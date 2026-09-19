"""§5 Session filter: London 07:00-16:00 UTC or New York 12:00-21:00 UTC, checked
against a candle's OPEN timestamp.

Boundary convention (implementation decision, flagged for confirmation): the upper
bound of each session is treated as INCLUSIVE of the candle opening exactly on the
hour (hour <= end), not exclusive. Real H4 candles from OANDA land on a fixed
01/05/09/13/17/21:00 UTC grid, so the 21:00 candle sits exactly on NY's stated
"...-21:00" boundary — reading that as exclusive would silently drop every day's
21:00 H4 candle from ever qualifying as a sweep/FVG/confirmation candle, and would
also disqualify the FVG in strategy_rules.md's own §8.1 worked example (whose
third FVG candle, idx 14, follows the 17:00 candle and so lands at 21:00). Treating
the stated end hour as the last INCLUDED hour resolves this and matches the
worked example; strategy_rules.md doesn't state inclusive/exclusive explicitly, so
this is called out here rather than silently assumed.

Takes pandas Series/Timestamps (not raw numpy datetime64) throughout — converting
tz-aware pandas timestamps to numpy datetime64 silently drops/mishandles the tz in
some pandas/numpy version combinations, so `.dt.hour` / `.hour` (which pandas
computes correctly against the stored UTC instant) is used instead.
"""
import pandas as pd

from . import rules


def _in_session_hours(hour) -> bool:
    lo_start, lo_end = rules.LONDON_SESSION_UTC
    ny_start, ny_end = rules.NEWYORK_SESSION_UTC
    return (lo_start <= hour <= lo_end) or (ny_start <= hour <= ny_end)


def in_session(timestamp: pd.Timestamp) -> bool:
    return _in_session_hours(timestamp.hour)


def in_session_mask(times: pd.Series) -> pd.Series:
    """Vectorized §5 check over a pandas Series of UTC tz-aware timestamps."""
    hours = times.dt.hour
    lo_start, lo_end = rules.LONDON_SESSION_UTC
    ny_start, ny_end = rules.NEWYORK_SESSION_UTC
    return ((hours >= lo_start) & (hours <= lo_end)) | ((hours >= ny_start) & (hours <= ny_end))
