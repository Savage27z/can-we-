"""Cross-timeframe index lookups. Not part of strategy_rules.md directly, but
needed to align H4/H1/D indices without introducing look-ahead bias — an H4 (or
Daily) candle is only "known" once its own period has fully elapsed relative to
the H1 timestamp being evaluated.
"""
import pandas as pd


def h1_index_at_or_after(h1_times: pd.Series, t: pd.Timestamp) -> int:
    """First H1 index with open time >= t."""
    return int(h1_times.searchsorted(t, side="left"))


def h4_index_fully_closed_by(h4_times: pd.Series, t: pd.Timestamp,
                             period: pd.Timedelta = pd.Timedelta(hours=4)) -> int:
    """Last H4 index whose candle has fully closed (open + period <= t) by
    time t. Returns -1 if no H4 candle has closed yet by t. `period` is the higher
    timeframe's real candle duration, default 4h to match "H4" in the name — a caller
    running this same lookup on a different timeframe pair passes its real duration.
    """
    cutoff = t - period
    return int(h4_times.searchsorted(cutoff, side="right")) - 1
