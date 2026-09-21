"""Pieces the daily-timeframe strategies share: when a signal can be acted on, and how to turn an
entry and a volatility measure into a signal.

A daily candle is complete 24 hours after it opens (OANDA's daily candles run 17:00 to 17:00 New
York time, and the DST change happens at the weekend, while the market is shut and no candle
straddles it). A strategy that forms its signal on that candle's close cannot trade until after it,
so the entry here is the close of the first H1 candle opening at 07:00 UTC after the daily close:
London's open, clear of the New York rollover (when spreads are widest) and of the weekend gap.
"""
from typing import Optional

import numpy as np
import pandas as pd

from ..strategy import BEARISH, BULLISH, Signal

DAILY_PERIOD = pd.Timedelta(hours=24)
ENTRY_HOUR = 7                                   # UTC, London open
MAX_WAIT = pd.Timedelta(days=4)                  # a Friday signal waits for Monday, no longer
_SCAN = 130                                      # H1 candles to look through (a weekend is ~50)


def daily_close_times(daily: pd.DataFrame) -> pd.Series:
    return daily["time"] + DAILY_PERIOD


def _ns(times: pd.Series) -> np.ndarray:
    return times.to_numpy(dtype="datetime64[ns]").astype("int64")


def entry_after(h1: pd.DataFrame, close_times: pd.Series, hour: int = ENTRY_HOUR,
                max_wait: pd.Timedelta = MAX_WAIT) -> np.ndarray:
    """For each time a signal became known, the row of the first H1 candle that OPENS at `hour`
    UTC at or after it, or -1 if there is none within `max_wait` (the data ends or has a hole)."""
    h1_ns = _ns(h1["time"])
    hours = h1["time"].dt.hour.to_numpy()
    wait = int(max_wait.value)
    found = np.full(len(close_times), -1, dtype=np.int64)
    for k, when in enumerate(_ns(close_times)):
        first = int(np.searchsorted(h1_ns, when, side="left"))
        window = np.flatnonzero(hours[first:first + _SCAN] == hour)
        if len(window) and h1_ns[first + window[0]] - when <= wait:
            found[k] = first + window[0]
    return found


def value_before(open_times: pd.Series, close_times: pd.Series, values: np.ndarray,
                 moments: np.ndarray) -> np.ndarray:
    """`values` (one per daily candle) as of each moment in ns: from the latest daily candle that
    had CLOSED by then. NaN where none had."""
    ends = _ns(close_times)
    pos = np.searchsorted(ends, moments, side="right") - 1
    out = np.full(len(moments), np.nan)
    ok = pos >= 0
    out[ok] = values[pos[ok]]
    return out


def make_signal(instrument: str, direction: str, h1: pd.DataFrame, index: int, stop: float,
                target: float, meta: Optional[dict] = None) -> Optional[Signal]:
    """A signal entered at the close of H1 row `index`, or None if the stop and target are not on
    the right sides of the entry (a stop already passed, a target already reached)."""
    entry = float(h1["close"].iloc[index])
    ordered = (stop < entry < target) if direction == BULLISH else (target < entry < stop)
    if not ordered or not np.isfinite(stop) or not np.isfinite(target):
        return None
    return Signal(instrument=instrument, direction=direction, entry_index=int(index),
                  entry_time=h1["time"].iloc[index] + pd.Timedelta(hours=1), entry_price=entry,
                  stop_price=float(stop), target_price=float(target), meta=meta or {})


def months_of(h1: pd.DataFrame) -> float:
    """Length of the H1 data in months, for signals-per-month."""
    if len(h1) < 2:
        return 0.0
    return (h1["time"].iloc[-1] - h1["time"].iloc[0]).total_seconds() / (3600 * 24 * 30.44)


def hours_of(h1: pd.DataFrame) -> np.ndarray:
    return h1["time"].dt.hour.to_numpy()


__all__ = ["BEARISH", "BULLISH", "ENTRY_HOUR", "daily_close_times", "entry_after", "hours_of",
           "make_signal", "months_of", "value_before"]
