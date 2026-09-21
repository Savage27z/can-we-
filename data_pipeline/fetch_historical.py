"""CLI: fetch historical candles from OANDA and store them as parquet.

Usage:
    python -m data_pipeline.fetch_historical
    python -m data_pipeline.fetch_historical --pairs EUR_USD GBP_USD USD_JPY --years 2
    python -m data_pipeline.fetch_historical --full          # refetch the whole lookback window
    python -m data_pipeline.fetch_historical --all --from-start   # every tradeable pair, all history

Safe to re-run: each run only fetches candles newer than what's already stored
(unless --full is given), and merges are de-duplicated by timestamp. --from-start goes
back to where OANDA's history for each instrument begins; once that has been done a
re-run just tops the data up.
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from . import config, instruments, storage
from .oanda_client import OandaAPIError, fetch_candles

# Stored history that begins within this long of an instrument's first candle counts as
# complete: the timeframes' first candles differ by a day or two, and re-fetching from
# the very start every run would be wasted work.
BACKFILL_TOLERANCE = timedelta(days=14)
UNIT_ATTEMPTS = 4          # per (pair, timeframe); each resumes from the last candle received
UNIT_WAIT_SECONDS = 30


def history_start(instrument: str) -> Optional[datetime]:
    """Where OANDA's history for the instrument begins, from the catalog (a day early,
    since a candle's date can be the day after it opens)."""
    known = instruments.find(instrument)
    if known is None or not known.history_start:
        return None
    day = datetime.fromisoformat(known.history_start).replace(tzinfo=timezone.utc)
    return day - timedelta(days=1)


def fetch_one(instrument: str, tf_name: str, granularity: str, years: int, full: bool,
              since: Optional[datetime] = None, attempts: int = 1,
              wait_seconds: float = UNIT_WAIT_SECONDS,
              sleep: Callable[[float], None] = time.sleep) -> int:
    """Fetches one instrument/timeframe and merges it into storage. Returns the number
    of candles fetched.

    `since` asks for history back to that instant: if the stored data starts later, it is
    fetched again from `since`; otherwise it is only topped up. A failure part-way keeps
    what was received and, with attempts > 1, continues from the last candle received
    (not from the newest stored one, which would leave a hole in a backfill).
    """
    now = datetime.now(timezone.utc)
    # Loaded once and reused for both the resume-point lookup and the merge
    # below (save_merged used to reload the same file a second time here).
    existing = storage.load(instrument, tf_name)

    if since is not None and (existing.empty or
                              existing["time"].min() > since + BACKFILL_TOLERANCE):
        from_time = since
    elif full or existing.empty:
        from_time = now - timedelta(days=365 * years)
    else:
        from_time = existing["time"].max().to_pydatetime()

    rows: list[dict] = []
    cursor = from_time
    for attempt in range(1, attempts + 1):
        try:
            for candle in fetch_candles(instrument, granularity, cursor, now):
                rows.append(candle)
            break
        except OandaAPIError:
            # Persist whatever was fetched rather than discarding it, so a failure
            # late in a long backfill doesn't cost the whole thing.
            if rows:
                cursor = rows[-1]["time"]
            if attempt == attempts:
                if rows:
                    storage.save_merged(instrument, tf_name, rows, existing=existing)
                    print(f"  {instrument} {tf_name}: saved {len(rows)} candles before the "
                          f"error below")
                raise
            print(f"  {instrument} {tf_name}: request failed (attempt {attempt}/{attempts}); "
                  f"retrying in {wait_seconds:g}s from {cursor}")
            sleep(wait_seconds)

    count = len({r["time"] for r in rows})   # a retry re-reads the boundary candle
    if rows:
        merged = storage.save_merged(instrument, tf_name, rows, existing=existing)
        print(
            f"  {instrument} {tf_name}: fetched {count} candles "
            f"(store now has {len(merged)} total, "
            f"{merged['time'].min()} -> {merged['time'].max()})"
        )
    else:
        print(f"  {instrument} {tf_name}: no new candles (already up to date)")
    return count


def _fetch_pair(pair: str, position: str, timeframes: list[str], years: int, full: bool,
                from_start: bool, attempts: int, wait_seconds: float,
                sleep: Callable[[float], None]) -> tuple[int, list[tuple[str, str, str]]]:
    print(f"[{position}] {pair}:")
    since = history_start(pair) if from_start else None
    if from_start and since is None:
        print(f"  {pair} is not in the catalog; using the {years}y lookback")
    total = 0
    failures: list[tuple[str, str, str]] = []
    for tf_name in timeframes:
        try:
            total += fetch_one(pair, tf_name, config.TIMEFRAMES[tf_name], years, full,
                               since=since, attempts=attempts, wait_seconds=wait_seconds,
                               sleep=sleep)
        except OandaAPIError as err:
            failures.append((pair, tf_name, str(err)[:200]))
            print(f"  {pair} {tf_name}: GIVING UP after {attempts} attempts: {str(err)[:200]}")
    return total, failures


def fetch_many(pairs: list[str], timeframes: list[str], years: int, full: bool,
               from_start: bool, attempts: int = UNIT_ATTEMPTS,
               wait_seconds: float = UNIT_WAIT_SECONDS,
               sleep: Callable[[float], None] = time.sleep,
               workers: int = 1) -> tuple[int, list[tuple[str, str, str]]]:
    """Fetches every pair/timeframe. A unit that keeps failing is recorded and skipped,
    so one bad pair doesn't stop a 68-pair run. Pairs are independent (each writes its
    own files), so `workers` > 1 fetches several at once. Returns (candles, failures)."""
    jobs = [(pair, f"{n}/{len(pairs)}") for n, pair in enumerate(pairs, start=1)]

    def run(job):
        pair, position = job
        return _fetch_pair(pair, position, timeframes, years, full, from_start, attempts,
                           wait_seconds, sleep)

    if workers <= 1:
        outcomes = [run(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(run, jobs))
    total = sum(count for count, _ in outcomes)
    failures = [f for _, unit_failures in outcomes for f in unit_failures]
    return total, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", nargs="+", default=config.PAIRS)
    parser.add_argument("--all", action="store_true",
                        help="Every instrument in the catalog (overrides --pairs).")
    parser.add_argument(
        "--timeframes", nargs="+", default=list(config.TIMEFRAMES.keys()),
        choices=list(config.TIMEFRAMES.keys()),
    )
    parser.add_argument("--years", type=int, default=config.DEFAULT_LOOKBACK_YEARS)
    parser.add_argument(
        "--full", action="store_true",
        help="Ignore existing parquet data and refetch the full lookback window.",
    )
    parser.add_argument(
        "--from-start", action="store_true",
        help="Go back to where OANDA's history for each instrument begins (from the "
             "catalog), then just top up on later runs.",
    )
    parser.add_argument("--workers", type=int, default=1,
                        help="Pairs to fetch at the same time (3-4 is polite to the API).")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):      # absent when output is captured or piped by a wrapper
        sys.stdout.reconfigure(encoding="utf-8")

    config.require_credentials()
    pairs = instruments.names() if args.all else args.pairs

    lookback = "from the start of history" if args.from_start else f"{args.years}y"
    print(f"OANDA environment: {config.OANDA_ENVIRONMENT}")
    print(f"{len(pairs)} pair(s)  Timeframes: {args.timeframes}  Lookback: {lookback}\n")

    total, failures = fetch_many(pairs, args.timeframes, args.years, args.full, args.from_start,
                                  workers=args.workers)

    print(f"\nDone. {total} candles fetched across {len(pairs)} pair(s).")
    if failures:
        print(f"{len(failures)} unit(s) failed; re-run the same command to resume them:")
        for pair, tf_name, message in failures:
            print(f"  {pair} {tf_name}: {message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
