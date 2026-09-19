"""CLI: fetch historical OHLCV candles from OANDA and store them as parquet.

Usage:
    python -m data_pipeline.fetch_historical
    python -m data_pipeline.fetch_historical --pairs EUR_USD GBP_USD USD_JPY --years 2
    python -m data_pipeline.fetch_historical --full   # ignore existing data, refetch all

Safe to re-run: each run only fetches candles newer than what's already stored
(unless --full is given), and merges are de-duplicated by timestamp.
"""
import argparse
from datetime import datetime, timedelta, timezone

from . import config, storage
from .oanda_client import OandaAPIError, fetch_candles


def fetch_one(instrument: str, tf_name: str, granularity: str, years: int, full: bool) -> int:
    now = datetime.now(timezone.utc)
    # Loaded once and reused for both the resume-point lookup and the merge
    # below (save_merged used to reload the same file a second time here).
    existing = storage.load(instrument, tf_name)

    if full or existing.empty:
        from_time = now - timedelta(days=365 * years)
    else:
        from_time = existing["time"].max().to_pydatetime()

    count = 0
    rows = []
    try:
        for candle in fetch_candles(instrument, granularity, from_time, now):
            rows.append(candle)
            count += 1
    except OandaAPIError:
        # Persist whatever was fetched before the error rather than discarding
        # it — a retry then only needs to cover what's still missing, instead
        # of re-fetching a long backfill from scratch on a transient failure.
        if rows:
            storage.save_merged(instrument, tf_name, rows, existing=existing)
            print(f"  {instrument} {tf_name}: saved {count} candles before the error below")
        raise

    if rows:
        merged = storage.save_merged(instrument, tf_name, rows, existing=existing)
        print(
            f"  {instrument} {tf_name}: fetched {count} new candles "
            f"(store now has {len(merged)} total, "
            f"{merged['time'].min()} -> {merged['time'].max()})"
        )
    else:
        print(f"  {instrument} {tf_name}: no new candles (already up to date)")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", default=config.PAIRS)
    parser.add_argument(
        "--timeframes", nargs="+", default=list(config.TIMEFRAMES.keys()),
        choices=list(config.TIMEFRAMES.keys()),
    )
    parser.add_argument("--years", type=int, default=config.DEFAULT_LOOKBACK_YEARS)
    parser.add_argument(
        "--full", action="store_true",
        help="Ignore existing parquet data and refetch the full lookback window.",
    )
    args = parser.parse_args()

    config.require_credentials()

    print(f"OANDA environment: {config.OANDA_ENVIRONMENT}")
    print(f"Pairs: {args.pairs}  Timeframes: {args.timeframes}  Lookback: {args.years}y\n")

    total = 0
    for pair in args.pairs:
        print(f"{pair}:")
        for tf_name in args.timeframes:
            granularity = config.TIMEFRAMES[tf_name]
            total += fetch_one(pair, tf_name, granularity, args.years, args.full)

    print(f"\nDone. {total} new candles fetched across {len(args.pairs)} pair(s).")


if __name__ == "__main__":
    main()
