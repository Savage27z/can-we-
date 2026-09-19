"""Pulls the latest candles for a pair before computing live state. Thin wrapper
around data_pipeline.fetch_historical's incremental fetch (same logic Phase 1
uses) — the live engine and the backtester read the exact same parquet files, so
there is no separate "live data" representation to drift out of sync with.
"""
from datetime import timedelta

from data_pipeline import config, storage
from data_pipeline.fetch_historical import fetch_one

# D/H4/H1 are fetched as three independent OANDA requests — if one fails (rate
# limit, transient network error) while the others succeed, the three local
# parquet files can end up desynchronized: e.g. H1 catches up to 15:00 UTC while
# H4 is stuck at 09:00 UTC because its own fetch errored. compute_state_from_market
# would then silently report a "current" state built partly from stale H4 data.
# These thresholds are generous enough to tolerate a normal weekend market
# closure (no candles Fri evening -> Sun evening) without false-positiving.
MAX_H4_LAG = timedelta(hours=52)
MAX_DAILY_LAG = timedelta(hours=76)


class DataSyncError(RuntimeError):
    pass


def refresh_pair(pair: str, years: int = config.DEFAULT_LOOKBACK_YEARS) -> None:
    for tf_name, granularity in config.TIMEFRAMES.items():
        fetch_one(pair, tf_name, granularity, years, full=False)
    _verify_synced(pair)


def _verify_synced(pair: str) -> None:
    h1_last = storage.last_timestamp(pair, "H1")
    h4_last = storage.last_timestamp(pair, "H4")
    daily_last = storage.last_timestamp(pair, "D")

    if h1_last is None:
        raise DataSyncError(f"{pair}: no H1 data available after refresh")
    if h4_last is None or h1_last - h4_last > MAX_H4_LAG:
        raise DataSyncError(
            f"{pair}: H4 data is stale relative to H1 (H1 up to {h1_last}, "
            f"H4 up to {h4_last}) — one of the two fetches likely failed or "
            f"lagged; refusing to compute live state from desynchronized "
            f"timeframes rather than silently reporting a stale H4 view as current."
        )
    if daily_last is None or h1_last - daily_last > MAX_DAILY_LAG:
        raise DataSyncError(
            f"{pair}: Daily data is stale relative to H1 (H1 up to {h1_last}, "
            f"Daily up to {daily_last}) — refusing to compute live state from "
            f"desynchronized timeframes."
        )
