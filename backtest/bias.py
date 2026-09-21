"""§4 Daily bias — a filter only, computed from the most recently CLOSED Daily
candle as of a given moment.

"Most recently closed" requires the Daily period to have actually elapsed, not
just that the row's open time is <= the reference time — otherwise the engine
would use a still-forming Daily candle's close, which is look-ahead bias since
that close value isn't known yet in real time. A candle's true close is taken
as the NEXT candle's open time rather than a hardcoded 24 hours later, since
OANDA's default daily boundary (5pm America/New_York) shifts by an hour in UTC
across DST transitions, making some real days 23h or 25h long.

The newest stored row has no next row to read that from. It used to be treated as always
closed, so the same historical setup got a different bias depending on whether another day
had been appended yet (audit F01). Its next open is now derived from its own open time, by
the same rule the stored rows follow, so the answer no longer depends on how much history
follows the query time.
"""
import pandas as pd

from . import rules

NY = "America/New_York"
FRIDAY = 4


def expected_next_open(open_time: pd.Timestamp) -> pd.Timestamp:
    """When the Daily candle after the one opening at `open_time` opens: the same New York
    wall-clock time on the next calendar day (so DST days are 23h or 25h), or two days
    later when that lands on the Friday close, since the market then shuts for the weekend."""
    local = open_time.tz_convert(NY).tz_localize(None) + pd.Timedelta(days=1)
    close = local.tz_localize(NY, ambiguous=True, nonexistent="shift_forward")
    if close.weekday() == FRIDAY:
        close = (local + pd.Timedelta(days=2)).tz_localize(NY, ambiguous=True,
                                                           nonexistent="shift_forward")
    return close.tz_convert("UTC")


def _closed_at(times: pd.Series, i: int) -> pd.Timestamp:
    """When candle i closes: the next stored row's open, or, for the newest row, the open
    the next candle is expected to have."""
    return times.iloc[i + 1] if i + 1 < len(times) else expected_next_open(times.iloc[i])


def bias_asof(daily_df: pd.DataFrame, t: pd.Timestamp) -> str:
    times = daily_df["time"]
    closes = daily_df["close"].to_numpy()

    # Index of the daily candle currently covering (or most recently opened before) t.
    idx_current = times.searchsorted(t, side="right") - 1
    if idx_current < 0:
        return "neutral"

    # That candle may still be forming as of t; the most recently CLOSED candle is then
    # the one before it (whose close is this candle's open, already past).
    d = idx_current - 1 if t < _closed_at(times, idx_current) else idx_current

    if d < rules.DAILY_BIAS_CANDLES - 1:
        return "neutral"

    c0, c1, c2 = closes[d], closes[d - 1], closes[d - 2]
    if c0 > c1 > c2:
        return "bullish"
    if c0 < c1 < c2:
        return "bearish"
    return "neutral"
