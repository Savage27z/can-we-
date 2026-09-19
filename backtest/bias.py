"""§4 Daily bias — a filter only, computed from the most recently CLOSED Daily
candle as of a given moment.

"Most recently closed" requires the Daily period to have actually elapsed, not
just that the row's open time is <= the reference time — otherwise the engine
would use a still-forming Daily candle's close, which is look-ahead bias since
that close value isn't known yet in real time. A candle's true close is taken
as the NEXT candle's open time rather than a hardcoded 24 hours later, since
OANDA's default daily boundary (5pm America/New_York) shifts by an hour in UTC
across DST transitions, making some real days 23h or 25h long.
"""
import pandas as pd

from . import rules


def bias_asof(daily_df: pd.DataFrame, t: pd.Timestamp) -> str:
    times = daily_df["time"]
    closes = daily_df["close"].to_numpy()

    # Index of the daily candle currently covering (or most recently opened before) t.
    idx_current = times.searchsorted(t, side="right") - 1
    if idx_current < 0:
        return "neutral"

    # That candle may still be forming as of t; the most recently CLOSED candle
    # is therefore the one before it. A Daily candle's true close is the next
    # candle's open — NOT a hardcoded 24h later, since OANDA's default daily
    # boundary (5pm America/New_York) shifts by an hour across DST transitions,
    # making some real days 23h or 25h in UTC. Using the next row's open avoids
    # hardcoding a duration entirely. When idx_current is the newest row we
    # have (no next row yet, e.g. live use), it's safe to treat it as already
    # closed: storage only ever keeps candles OANDA reported as complete.
    if idx_current + 1 < len(times):
        true_close = times.iloc[idx_current + 1]
        d = idx_current - 1 if t < true_close else idx_current
    else:
        d = idx_current

    if d < rules.DAILY_BIAS_CANDLES - 1:
        return "neutral"

    c0, c1, c2 = closes[d], closes[d - 1], closes[d - 2]
    if c0 > c1 > c2:
        return "bullish"
    if c0 < c1 < c2:
        return "bearish"
    return "neutral"
