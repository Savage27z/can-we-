"""Random-entry null model: what would trades with this strategy's stops and targets have
earned if they had been entered at random?

A strategy's result mixes two things: the value of its entry signal, and everything else about
the trade, such as its stop and target distances, the trading costs, and whatever the market
drifted while the trade ran. To isolate the first, each real signal is replayed many times at a
random candle instead of its real one. The replay keeps the signal's direction (unless asked to
randomise it), its risk, its reward and its invalidation offset, and is resolved under the same
exit policy and the same costs. If the real trades do no better than these, the entry signal has
added nothing.

Random entries are drawn from in-session H1 candles within `window_days` of the real entry, so
they see the same volatility and the same slow drift as the real trade did, rather than a
different market a decade away. A candle qualifies under the same session rule the strategy's own
confirmation candles must meet.

One replicate is one random entry for every real signal on the instrument; its statistic is the
mean net R over those trades, comparable to the real mean. Many replicates give the null
distribution, and the p-value is the share of replicates that did at least as well as the real
result.
"""
import zlib
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from backtest import sessions

from .strategy import BULLISH, Signal
from .vector import Arrays, resolve_batch

DEFAULT_WINDOW_DAYS = 60
CHUNK = 8_192                  # random trades resolved per batch, to bound memory
DIRECTIONS = ("same", "random")


@dataclass(frozen=True)
class Geometry:
    """What a real signal contributes to its replays: everything except when it happens."""
    entry_index: np.ndarray
    sign: np.ndarray             # +1 buy, -1 sell
    risk: np.ndarray             # entry to stop, price units
    reward: np.ndarray           # entry to target
    invalidation_offset: np.ndarray


def geometry_of(signals: Sequence[Signal]) -> Geometry:
    def level(s: Signal) -> float:
        return s.invalidation_price if s.invalidation_price is not None else s.stop_price

    return Geometry(
        entry_index=np.array([s.entry_index for s in signals], dtype=np.int64),
        sign=np.array([1.0 if s.direction == BULLISH else -1.0 for s in signals]),
        risk=np.array([s.risk for s in signals]),
        reward=np.array([abs(s.target_price - s.entry_price) for s in signals]),
        invalidation_offset=np.array([abs(s.entry_price - level(s)) for s in signals]),
    )


def _times_ns(frame: pd.DataFrame) -> np.ndarray:
    return frame["time"].to_numpy(dtype="datetime64[ns]").astype("int64")


def draw_entries(entry_index: np.ndarray, times_ns: np.ndarray, eligible: np.ndarray,
                 window_days: float, replicates: int, rng: np.random.Generator) -> np.ndarray:
    """(signals, replicates) candle indices: for each real entry, random eligible candles within
    the window either side of it. If nothing eligible is that close, anywhere eligible will do."""
    eligible_times = times_ns[eligible]
    half = int(window_days * 86_400 * 1e9)
    out = np.empty((len(entry_index), replicates), dtype=np.int64)
    for row, entry in enumerate(entry_index):
        t = times_ns[entry]
        lo = int(np.searchsorted(eligible_times, t - half, side="left"))
        hi = int(np.searchsorted(eligible_times, t + half, side="right"))
        if hi <= lo:
            lo, hi = 0, len(eligible)
        out[row] = eligible[rng.integers(lo, hi, size=replicates)]
    return out


@dataclass
class NullResult:
    sums: np.ndarray             # per replicate: total net R over its resolved trades
    counts: np.ndarray           # per replicate: how many of its trades resolved
    cost_sums: Optional[np.ndarray] = None    # per replicate: total cost in R over those trades

    def means(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.counts > 0, self.sums / self.counts, np.nan)

    def mean_cost(self) -> float:
        """Average cost in R per random trade, over every replicate."""
        if self.cost_sums is None or self.counts.sum() == 0:
            return float("nan")
        return float(self.cost_sums.sum() / self.counts.sum())


def null_replicates(signals: Sequence[Signal], h1: pd.DataFrame, policy: str, cost_model,
                    pip_size: float, replicates: int, window_days: float = DEFAULT_WINDOW_DAYS,
                    direction: str = "same", seed: int = 0, key: str = "") -> NullResult:
    """The null distribution for one instrument. `key` (e.g. the instrument name) keeps two
    instruments' random draws independent while each stays reproducible from `seed`."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    n_signals = len(signals)
    if n_signals == 0:
        return NullResult(np.zeros(replicates), np.zeros(replicates, dtype=np.int64),
                          np.zeros(replicates))

    arrays = Arrays.from_frame(h1)
    geo = geometry_of(signals)
    rng = np.random.default_rng([seed, zlib.crc32(key.encode())])
    eligible = np.flatnonzero(sessions.in_session_mask(h1["time"]).to_numpy())
    entries = draw_entries(geo.entry_index, _times_ns(h1), eligible, window_days, replicates, rng)
    if direction == "same":
        sign = np.broadcast_to(geo.sign[:, None], entries.shape)
    else:
        sign = rng.choice(np.array([-1.0, 1.0]), size=entries.shape)

    known = arrays.spread[np.isfinite(arrays.spread)]
    typical = float(np.median(known)) if len(known) else float("nan")

    e = entries.ravel()
    s = np.ascontiguousarray(sign).ravel()
    risk = np.repeat(geo.risk, replicates)
    reward = np.repeat(geo.reward, replicates)
    inv_offset = np.repeat(geo.invalidation_offset, replicates)

    net = np.empty(len(e))
    cost = np.empty(len(e))
    for start in range(0, len(e), CHUNK):
        part = slice(start, start + CHUNK)
        price = arrays.close[e[part]]
        exit_index, r_gross = resolve_batch(
            arrays, policy, entry_index=e[part], sign=s[part], entry_price=price,
            stop=price - s[part] * risk[part], target=price + s[part] * reward[part],
            invalidation=price - s[part] * inv_offset[part], risk=risk[part])
        cost_r = cost_model.price_array(arrays.spread[e[part]], typical, pip_size) / risk[part]
        net[part] = r_gross - cost_r                     # NaN stays NaN: the trade was still open
        cost[part] = cost_r
    net = net.reshape(entries.shape)
    resolved = np.isfinite(net)
    return NullResult(sums=np.where(resolved, net, 0.0).sum(axis=0),
                      counts=resolved.sum(axis=0).astype(np.int64),
                      cost_sums=np.where(resolved, cost.reshape(entries.shape), 0.0).sum(axis=0))


@dataclass(frozen=True)
class Comparison:
    observed: float              # the real mean net R per trade
    null_mean: float
    null_sd: float
    excess: float                # observed minus null_mean: what the entry signal added
    z: float
    p_value: float               # share of replicates at least as good as the real result
    replicates: int              # replicates that had any resolved trade


def compare(observed_mean: float, null: NullResult) -> Comparison:
    """Where the real result sits among the random-entry results (upper tail: better is rarer)."""
    means = null.means()
    means = means[np.isfinite(means)]
    if len(means) < 2 or not np.isfinite(observed_mean):
        nan = float("nan")
        return Comparison(observed_mean, nan, nan, nan, nan, nan, len(means))
    null_mean, null_sd = float(means.mean()), float(means.std(ddof=1))
    z = (observed_mean - null_mean) / null_sd if null_sd > 0 else float("nan")
    p = (1 + int((means >= observed_mean).sum())) / (len(means) + 1)
    return Comparison(observed_mean, null_mean, null_sd, observed_mean - null_mean, z, p, len(means))


def pool(nulls: Sequence[NullResult]) -> NullResult:
    """Replicate k across instruments combined: the trade-weighted null for the whole universe."""
    have_costs = all(n.cost_sums is not None for n in nulls)
    return NullResult(sums=np.sum([n.sums for n in nulls], axis=0),
                      counts=np.sum([n.counts for n in nulls], axis=0),
                      cost_sums=np.sum([n.cost_sums for n in nulls], axis=0) if have_costs else None)


def standardised_null(null: NullResult) -> Optional[np.ndarray]:
    """Each replicate's z-score against its own null distribution, for the max-statistic
    correction. None if the null has no spread to standardise by."""
    means = null.means()
    finite = means[np.isfinite(means)]
    if len(finite) < 2 or finite.std(ddof=1) == 0:
        return None
    return (means - finite.mean()) / finite.std(ddof=1)
