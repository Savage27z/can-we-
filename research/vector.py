"""Resolves many trades at once, for the null model.

The null model needs millions of random trades resolved, far too many for the one-trade-at-a-time
exit policies in exits.py. This resolves a whole batch with numpy, and must agree with those
scalar policies exactly (tests/test_vector.py checks it trade by trade).

The trick that keeps it short: a sell is a buy in mirrored prices. Multiplying every price by the
trade's sign (+1 buy, -1 sell) turns a short's stop above entry into a stop below it, and its
lows into highs, so one set of "buy" rules serves both.

Trades are resolved in blocks of candles, and a trade drops out of the batch as soon as it
ends, so the cost follows the trades that run long, not the number of trades times the longest.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

BLOCK = 128
POLICIES = ("close", "touch", "plan")


@dataclass(frozen=True)
class Arrays:
    """The execution frame as arrays."""
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    spread: np.ndarray          # NaN where unknown

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "Arrays":
        spread = (frame["spread"].to_numpy(dtype=float) if "spread" in frame.columns
                  else np.full(len(frame), np.nan))
        return cls(open=frame["open"].to_numpy(dtype=float), high=frame["high"].to_numpy(dtype=float),
                   low=frame["low"].to_numpy(dtype=float), close=frame["close"].to_numpy(dtype=float),
                   spread=spread)

    def __len__(self) -> int:
        return len(self.close)


def resolve_batch(arrays: Arrays, policy: str, entry_index: np.ndarray, sign: np.ndarray,
                  entry_price: np.ndarray, stop: np.ndarray, target: np.ndarray,
                  invalidation: np.ndarray, risk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ends each trade under `policy` ("close", "touch" or "plan", as in exits.py).

    Every argument is a 1-D array with one entry per trade, in price units; `sign` is +1 for a
    buy and -1 for a sell and `risk` is the distance from entry to stop (what one R is).
    Returns (exit_index, r_gross): the row of the candle the trade ended in (-1 while it is still
    open at the end of the data) and its result in R before costs (NaN while open).
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; choose from {POLICIES}")
    n, m = len(arrays), len(entry_index)
    exit_index = np.full(m, -1, dtype=np.int64)
    r_gross = np.full(m, np.nan)
    if m == 0:
        return exit_index, r_gross

    entry_s = sign * entry_price
    stop_s, target_s, invalidation_s = sign * stop, sign * target, sign * invalidation
    active = np.arange(m)
    first_candle = entry_index + 1
    steps = np.arange(BLOCK)

    while active.size:
        alive = first_candle[active] < n            # a trade with no candles left can never end
        active = active[alive]
        if not active.size:
            break
        idx = first_candle[active][:, None] + steps[None, :]
        valid = idx < n
        ic = np.minimum(idx, n - 1)
        s = sign[active][:, None]

        close_s = s * arrays.close[ic]
        a_stop, a_target, a_inv = (stop_s[active][:, None], target_s[active][:, None],
                                   invalidation_s[active][:, None])
        no_hit = np.zeros(idx.shape, dtype=bool)
        if policy == "close":
            inv_hit, target_hit, stop_hit = close_s < a_inv, close_s >= a_target, no_hit
        else:
            high_s = np.where(s > 0, arrays.high[ic], -arrays.low[ic])
            low_s = np.where(s > 0, arrays.low[ic], -arrays.high[ic])
            open_s = s * arrays.open[ic]
            stop_hit, target_hit = low_s <= a_stop, high_s >= a_target
            inv_hit = close_s < a_inv if policy == "plan" else no_hit

        hit = (stop_hit | target_hit | inv_hit) & valid
        ended = hit.any(axis=1)
        if ended.any():
            rows = np.flatnonzero(ended)
            col = hit[rows].argmax(axis=1)              # the first candle that ends the trade
            r_stop, r_target = stop_s[active][rows], target_s[active][rows]
            if policy == "close":
                price = np.where(inv_hit[rows, col], r_stop, r_target)   # scored at the level
            else:
                stop_fill = np.minimum(open_s[rows, col], r_stop)        # a gap fills at the worse open
                price = np.where(stop_hit[rows, col], stop_fill, r_target)
                if policy == "plan":
                    price = np.where(stop_hit[rows, col] | target_hit[rows, col], price,
                                     close_s[rows, col])                 # closed by hand at the close
            done = active[rows]
            exit_index[done] = ic[rows, col]
            r_gross[done] = (price - entry_s[done]) / risk[done]
        active = active[~ended]
        first_candle[active] += BLOCK
    return exit_index, r_gross
