"""Data-quality checks for the stored candles.

A backtest is only as trustworthy as its data, and a research platform that runs over
dozens of instruments will meet bad data (duplicated rows, holes, a spread that is
wrong for a whole year) that no single-pair test ever hit. These checks run per
instrument and timeframe and split findings in two:

- Hard errors, which make the data unfit to backtest until fixed: duplicate or unsorted
  timestamps, NaN or non-positive prices, an impossible candle (high below low, or
  the open/close outside the range), and candles off OANDA's time grid.
- Measurements, reported so a researcher can judge fitness: missing candles inside
  trading hours, flat candles, outlier ranges, spread statistics, and how well the
  stored H4 candles agree with the H1 candles they are made of.

OANDA aligns its Daily and H4 candles to 17:00 New York time, so in UTC the grid moves
by an hour twice a year (Daily opens 21:00 UTC in summer, 22:00 in winter). The grid
check therefore works in New York time, where it never moves.

    python -m data_pipeline.quality --all
    python -m data_pipeline.quality --pairs EUR_USD GBP_USD --out data/quality_report.csv
"""
import argparse
import sys
from typing import Optional

import numpy as np
import pandas as pd

from . import config, instruments, storage

NY = "America/New_York"
PERIOD = {
    "D": pd.Timedelta(hours=24),
    "H4": pd.Timedelta(hours=4),
    "H1": pd.Timedelta(hours=1),
}
HARD_CHECKS = ("duplicate_times", "unsorted", "nan_prices", "nonpositive_prices",
               "ohlc_invalid", "off_grid")
TOLERANCE = 1e-9
OUTLIER_RANGE_MULTIPLE = 20      # a candle this many times the recent median range
OUTLIER_WINDOW = 200
SPARSE_MISSING_PCT = 5.0         # more missing than this and the data is called sparse


def _local(times) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(times).tz_convert(NY)


def market_open(times) -> np.ndarray:
    """Whether a candle opening at each time falls inside the FX trading week, which
    runs from Sunday 17:00 to Friday 17:00 New York time."""
    local = _local(times)
    dow, hour = local.dayofweek, local.hour
    closed = ((dow == 4) & (hour >= 17)) | (dow == 5) | ((dow == 6) & (hour < 17))
    return ~np.asarray(closed)


def on_grid(times, timeframe: str) -> np.ndarray:
    """Whether each open time sits on OANDA's candle grid for the timeframe."""
    local = _local(times)
    whole_hour = (local.minute == 0) & (local.second == 0)
    if timeframe == "H1":
        return np.asarray(whole_hour)
    if timeframe == "H4":
        return np.asarray(whole_hour & (((local.hour - 17) % 4) == 0))
    if timeframe == "D":
        return np.asarray(whole_hour & (local.hour == 17))
    raise ValueError(f"unknown timeframe {timeframe!r}")


def missing_slots(times: pd.Series, timeframe: str) -> tuple[int, int, int]:
    """(missing candles, gaps, longest gap in candles), counting only candles that
    should exist: the weekend closure and the hours between Friday and Sunday
    are not holes."""
    period = PERIOD[timeframe]
    t = pd.Series(pd.to_datetime(times, utc=True)).drop_duplicates().sort_values().reset_index(drop=True)
    if len(t) < 2:
        return 0, 0, 0
    missing = gaps = longest = 0
    for i in np.flatnonzero((t.diff() > period).to_numpy()):
        first, last = t[i - 1] + period, t[i] - period
        if first > last:
            continue
        slots = pd.date_range(first, last, freq=period)
        n_missing = int(market_open(slots).sum())
        if n_missing:
            missing += n_missing
            gaps += 1
            longest = max(longest, n_missing)
    return missing, gaps, longest


def h1_vs_h4_mismatch(h1: pd.DataFrame, h4: pd.DataFrame) -> tuple[int, int]:
    """(H4 candles compared, of which disagree with the H1 candles they are made of).
    Only H4 candles whose four H1 candles are all present are compared."""
    if h1.empty or h4.empty:
        return 0, 0
    local_hour = pd.DatetimeIndex(h1["time"]).tz_convert(NY).hour
    offset = pd.to_timedelta((local_hour - 17) % 4, unit="h")
    bucket = h1["time"].reset_index(drop=True) - pd.Series(offset)
    grouped = h1.reset_index(drop=True).assign(bucket=bucket).groupby("bucket").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), n=("open", "size"))
    grouped = grouped[grouped["n"] == 4]
    joined = grouped.join(h4.set_index("time")[["open", "high", "low", "close"]],
                          how="inner", rsuffix="_h4")
    if joined.empty:
        return 0, 0
    differs = np.zeros(len(joined), dtype=bool)
    for col in ("open", "high", "low", "close"):
        differs |= ~np.isclose(joined[col], joined[f"{col}_h4"], rtol=0, atol=1e-8)
    return len(joined), int(differs.sum())


def check_frame(df: pd.DataFrame, timeframe: str, instrument: str,
                h1: Optional[pd.DataFrame] = None) -> dict:
    """All checks for one instrument/timeframe. `h1` enables the H1-vs-H4 comparison."""
    row: dict = {"instrument": instrument, "timeframe": timeframe, "rows": len(df)}
    if df.empty:
        return {**row, "first": None, "last": None, "years": 0.0, "hard_errors": 0,
                "issues": "no data"}

    t = df["time"]
    ohlc = df[["open", "high", "low", "close"]]
    pip = instruments.find(instrument).pip_size if instruments.find(instrument) else None

    row.update(first=t.min(), last=t.max(), years=round((t.max() - t.min()).days / 365.25, 2))
    row["duplicate_times"] = int(t.duplicated().sum())
    row["unsorted"] = int((t.diff().dropna() < pd.Timedelta(0)).sum())
    row["nan_prices"] = int(ohlc.isna().any(axis=1).sum())
    row["nonpositive_prices"] = int((ohlc <= 0).any(axis=1).sum())
    o, h, l, c = (df[k] for k in ("open", "high", "low", "close"))
    row["ohlc_invalid"] = int(((h < l) | (h < o - TOLERANCE) | (h < c - TOLERANCE)
                               | (l > o + TOLERANCE) | (l > c + TOLERANCE)).sum())
    row["off_grid"] = int((~on_grid(t, timeframe)).sum())
    row["hard_errors"] = sum(row[k] for k in HARD_CHECKS)

    missing, gaps, longest = missing_slots(t, timeframe)
    row.update(missing_candles=missing, gaps=gaps, longest_gap=longest,
               missing_pct=round(100 * missing / (missing + len(df)), 2))

    rng = h - l
    med = rng.rolling(OUTLIER_WINDOW, min_periods=50).median()
    row["flat_pct"] = round(100 * float((rng == 0).mean()), 2)
    row["outlier_bars"] = int(((rng > OUTLIER_RANGE_MULTIPLE * med) & (med > 0)).sum())
    row["zero_volume_pct"] = round(100 * float((df["volume"] == 0).mean()), 2)

    spread = df["spread"] if "spread" in df.columns else pd.Series(np.nan, index=df.index)
    known = spread.dropna()
    row["spread_missing_pct"] = round(100 * float(spread.isna().mean()), 2)
    row["spread_negative"] = int((known < 0).sum())
    if len(known) and pip:
        row["spread_median_pips"] = round(float(known.median()) / pip, 2)
        row["spread_p99_pips"] = round(float(known.quantile(0.99)) / pip, 2)

    if timeframe == "H4" and h1 is not None:
        compared, mismatched = h1_vs_h4_mismatch(h1, df)
        row["h4_compared"] = compared
        row["h4_mismatch_pct"] = round(100 * mismatched / compared, 2) if compared else None

    flags = [k for k in HARD_CHECKS if row[k]]
    if row["missing_pct"] > SPARSE_MISSING_PCT:
        flags.append("sparse")
    if row["outlier_bars"]:
        flags.append("outliers")
    if row.get("spread_missing_pct", 100) > 50:
        flags.append("no_spread")
    if row["spread_negative"]:
        flags.append("negative_spread")
    if row.get("h4_mismatch_pct"):
        flags.append("h4_vs_h1")
    row["issues"] = ", ".join(flags)
    return row


def check_instrument(instrument: str, timeframes: list[str]) -> list[dict]:
    h1 = storage.load(instrument, "H1") if "H4" in timeframes else None
    rows = []
    for tf in timeframes:
        df = h1 if tf == "H1" and h1 is not None else storage.load(instrument, tf)
        rows.append(check_frame(df, tf, instrument, h1=h1))
    return rows


def run(pairs: list[str], timeframes: list[str], progress=None) -> pd.DataFrame:
    rows = []
    for number, pair in enumerate(pairs, start=1):
        if progress:
            progress(f"[{number}/{len(pairs)}] {pair}")
        rows.extend(check_instrument(pair, timeframes))
    return pd.DataFrame(rows)


def summarise(report: pd.DataFrame) -> str:
    lines = [f"{report['instrument'].nunique()} instruments, {len(report)} instrument/timeframe "
             f"series, {int(report['rows'].sum()):,} candles checked"]
    empty = report[report["rows"] == 0]
    if len(empty):
        lines.append(f"NO DATA: {', '.join(sorted(set(empty['instrument'])))}")
    checked = report[report["rows"] > 0]
    if checked.empty:
        lines.append("\nNothing to check: none of these instruments has stored candles. "
                     "Fetch them first with python -m data_pipeline.fetch_historical.")
        return "\n".join(lines)

    hard = checked[checked["hard_errors"] > 0]
    lines.append(f"\nHARD ERRORS: {len(hard)} series" + (" (unfit to backtest until fixed)" if len(hard) else " (none)"))
    for _, r in hard.iterrows():
        detail = ", ".join(f"{k}={int(r[k])}" for k in HARD_CHECKS if r[k])
        lines.append(f"  {r['instrument']} {r['timeframe']}: {detail}")

    sparse = checked[(checked["timeframe"] == "H1") & (checked["missing_pct"] > SPARSE_MISSING_PCT)]
    lines.append(f"\nSPARSE H1 (>{SPARSE_MISSING_PCT:g}% of trading-hour candles missing): {len(sparse)}")
    for _, r in sparse.sort_values("missing_pct", ascending=False).iterrows():
        lines.append(f"  {r['instrument']}: {r['missing_pct']}% missing, longest gap {int(r['longest_gap'])} candles")

    h1 = checked[checked["timeframe"] == "H1"]
    if "spread_median_pips" in h1 and h1["spread_median_pips"].notna().any():
        widest = h1.sort_values("spread_median_pips", ascending=False).head(5)
        tightest = h1.sort_values("spread_median_pips").head(5)
        lines.append("\nMEDIAN H1 SPREAD (pips)  widest: " + ", ".join(
            f"{r['instrument']} {r['spread_median_pips']}" for _, r in widest.iterrows()))
        lines.append("                         tightest: " + ", ".join(
            f"{r['instrument']} {r['spread_median_pips']}" for _, r in tightest.iterrows()))
    no_spread = checked[checked["spread_missing_pct"] > 50]
    if len(no_spread):
        lines.append(f"\nMostly NO SPREAD data (re-fetch to add it): {len(no_spread)} series")

    h4 = checked[(checked["timeframe"] == "H4") & checked.get("h4_mismatch_pct", pd.Series(dtype=float)).notna()]
    if len(h4):
        lines.append(f"\nH4 vs H1 agreement: worst mismatch {h4['h4_mismatch_pct'].max()}% "
                     f"({h4.loc[h4['h4_mismatch_pct'].idxmax(), 'instrument']}), "
                     f"median {h4['h4_mismatch_pct'].median()}%")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", nargs="+", default=config.PAIRS)
    parser.add_argument("--all", action="store_true", help="Every instrument in the catalog.")
    parser.add_argument("--timeframes", nargs="+", default=list(config.TIMEFRAMES),
                        choices=list(config.TIMEFRAMES))
    parser.add_argument("--out", default=None, help="Write the full per-series report as CSV.")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    pairs = instruments.names() if args.all else args.pairs
    report = run(pairs, args.timeframes, progress=print)
    if args.out:
        report.to_csv(args.out, index=False)
        print(f"\nfull report written to {args.out}")
    print("\n" + summarise(report))
    if (report["rows"] > 0).any() and (report.loc[report["rows"] > 0, "hard_errors"] > 0).any():
        sys.exit(1)


if __name__ == "__main__":
    main()
