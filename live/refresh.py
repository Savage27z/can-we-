"""Pulls the latest candles for a pair before computing live state. Thin wrapper
around data_pipeline.fetch_historical's incremental fetch (same logic Phase 1
uses) — the live engine and the backtester read the exact same parquet files, so
there is no separate "live data" representation to drift out of sync with.
"""
from datetime import timedelta
from typing import Optional

import pandas as pd

from data_pipeline import config, storage
from data_pipeline.fetch_historical import fetch_one

H4_PERIOD = timedelta(hours=4)
D_PERIOD = timedelta(hours=24)

# D/H4/H1 are fetched as three independent OANDA requests. If one fails (rate limit,
# transient network error) while the others succeed, the local files desynchronise —
# e.g. H1 catches up to 15:00 UTC while H4 is stuck at 09:00 UTC — and the live state
# would silently be built partly from stale H4 data.
#
# Staleness is measured by counting the H1 candles that exist AFTER the newest H4 (or
# Daily) candle closed, not by comparing clock times. Clock-time lags depend on the
# weekend closure and on how OANDA labels Daily candles (by their 17:00 New York
# open), so any fixed lag limit either false-alarms for most of Monday or misses real
# staleness midweek. A closure adds no H1 candles, so the count doesn't grow across a
# weekend, but it does grow steadily when a fetch is genuinely stale.
#
# Healthy: the newest complete H4 candle leaves only the (up to 4) H1 candles of the
# still-forming H4 candle after it, plus one for fetch timing.
MAX_H1_AFTER_H4 = 5
# Healthy: a still-forming day is 24 H1 candles (25 across a DST change), plus slack.
MAX_H1_AFTER_D = 30


class DataSyncError(RuntimeError):
    pass


def refresh_pair(pair: str, years: int = config.DEFAULT_LOOKBACK_YEARS) -> None:
    for tf_name, granularity in config.TIMEFRAMES.items():
        fetch_one(pair, tf_name, granularity, years, full=False)
    _verify_synced(pair)


def sync_problem(h1_times: pd.Series, h4_last: Optional[pd.Timestamp],
                 d_last: Optional[pd.Timestamp]) -> Optional[str]:
    """Describes why the three timeframes look desynchronised, or None if in sync."""
    if h1_times.empty:
        return "no H1 data available after refresh"
    if h4_last is None:
        return "no H4 data available after refresh"
    if d_last is None:
        return "no Daily data available after refresh"

    h4_uncovered = int((h1_times >= h4_last + H4_PERIOD).sum())
    if h4_uncovered > MAX_H1_AFTER_H4:
        return (f"H4 data is stale: {h4_uncovered} H1 candles exist after the newest H4 "
                f"candle closed (H4 up to {h4_last}, H1 up to {h1_times.max()}) — one of "
                f"the fetches likely failed or lagged; refusing to compute live state "
                f"from desynchronized timeframes.")
    d_uncovered = int((h1_times >= d_last + D_PERIOD).sum())
    if d_uncovered > MAX_H1_AFTER_D:
        return (f"Daily data is stale: {d_uncovered} H1 candles exist after the newest "
                f"Daily candle closed (Daily up to {d_last}, H1 up to {h1_times.max()}) — "
                f"refusing to compute live state from desynchronized timeframes.")
    return None


def _verify_synced(pair: str) -> None:
    problem = sync_problem(
        storage.load(pair, "H1")["time"],
        storage.last_timestamp(pair, "H4"),
        storage.last_timestamp(pair, "D"),
    )
    if problem:
        raise DataSyncError(f"{pair}: {problem}")
