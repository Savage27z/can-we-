"""Resolves many trades at once, for the null model.

The null model needs millions of random trades resolved, far too many for the one-trade-at-a-time
exit policies in exits.py. This resolves a whole batch with numpy, and must agree with those
scalar policies exactly (tests/test_vector.py checks it trade by trade).

The trick that keeps it short: a sell is a buy in mirrored prices. Negating every price turns a
short's stop above entry into a stop below it, and its lows into highs, so one set of "buy" rules
serves both. Buys and sells are resolved as two separate batches for that reason.

Trades are resolved in blocks of candles, and a trade drops out of the batch as soon as it
ends, so the cost follows the trades that run long, not the number of trades times the longest.
Each block is read as contiguous windows of the price arrays (padded with NaN past the end of the
data, and a comparison with NaN is never true, so a trade cannot end in candles that do not exist).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

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


class _Windows:
    """Sign-adjusted price arrays, viewed as windows of BLOCK candles starting at each row."""

    def __init__(self, arrays: Arrays, sign: float):
        pad = np.full(BLOCK, np.nan)

        def window(values: np.ndarray) -> np.ndarray:
            return sliding_window_view(np.concatenate([values, pad]), BLOCK)

        if sign > 0:
            self.open, self.high = window(arrays.open), window(arrays.high)
            self.low, self.close = window(arrays.low), window(arrays.close)
        else:                                          # mirrored: highs become lows, prices negate
            self.open, self.high = window(-arrays.open), window(-arrays.low)
            self.low, self.close = window(-arrays.high), window(-arrays.close)


def _resolve_buys(windows: _Windows, n: int, policy: str, entry_index: np.ndarray,
                  entry_s: np.ndarray, stop_s: np.ndarray, target_s: np.ndarray,
                  invalidation_s: np.ndarray, risk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The buy rules, over prices already mirrored for sells (so `entry_s` etc. are signed)."""
    m = len(entry_index)
    exit_index = np.full(m, -1, dtype=np.int64)
    r_gross = np.full(m, np.nan)
    first = entry_index + 1                            # the next candle to examine, per trade
    active = np.arange(m)

    while active.size:
        starts = first[active]
        alive = starts < n                             # a trade with no candles left can never end
        active, starts = active[alive], starts[alive]
        if not active.size:
            break
        a_stop, a_target = stop_s[active][:, None], target_s[active][:, None]
        close = windows.close[starts]
        if policy == "close":
            inv_hit, target_hit = close < invalidation_s[active][:, None], close >= a_target
            hit = inv_hit | target_hit
        else:
            high, low = windows.high[starts], windows.low[starts]
            stop_hit, target_hit = low <= a_stop, high >= a_target
            hit = stop_hit | target_hit
            if policy == "plan":
                inv_hit = close < invalidation_s[active][:, None]
                hit = hit | inv_hit
        ended = hit.any(axis=1)
        if ended.any():
            rows = np.flatnonzero(ended)
            col = hit[rows].argmax(axis=1)              # the first candle that ends the trade
            r_stop, r_target = stop_s[active][rows], target_s[active][rows]
            if policy == "close":
                price = np.where(inv_hit[rows, col], r_stop, r_target)   # scored at the level
            else:
                open_ = windows.open[starts[rows], col]
                stop_fill = np.minimum(open_, r_stop)                    # a gap fills at the worse open
                price = np.where(stop_hit[rows, col], stop_fill, r_target)
                if policy == "plan":
                    price = np.where(stop_hit[rows, col] | target_hit[rows, col], price,
                                     close[rows, col])                   # closed by hand at the close
            done = active[rows]
            exit_index[done] = starts[rows] + col
            r_gross[done] = (price - entry_s[done]) / risk[done]
        active = active[~ended]
        first[active] += BLOCK
    return exit_index, r_gross


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

    for side in (1.0, -1.0):
        picked = np.flatnonzero(sign == side)
        if not picked.size:
            continue
        exit_index[picked], r_gross[picked] = _resolve_buys(
            _Windows(arrays, side), n, policy, entry_index[picked], side * entry_price[picked],
            side * stop[picked], side * target[picked], side * invalidation[picked], risk[picked])
    return exit_index, r_gross
