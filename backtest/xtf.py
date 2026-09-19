"""Cross-timeframe index lookups. Not part of strategy_rules.md directly, but
needed to align H4/H1/D indices without introducing look-ahead bias — an H4 (or
Daily) candle is only "known" once its own period has fully elapsed relative to
the H1 timestamp being evaluated.
"""
import pandas as pd


def h1_index_at_or_after(h1_times: pd.Series, t: pd.Timestamp) -> int:
    """First H1 index with open time >= t."""
    return int(h1_times.searchsorted(t, side="left"))


def h4_index_fully_closed_by(h4_times: pd.Series, t: pd.Timestamp, period_hours: int = 4) -> int:
    """Last H4 index whose candle has fully closed (open + period_hours <= t) by
    time t. Returns -1 if no H4 candle has closed yet by t.
    """
    cutoff = t - pd.Timedelta(hours=period_hours)
    return int(h4_times.searchsorted(cutoff, side="right")) - 1
