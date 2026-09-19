"""Parquet storage for fetched candles.

Layout: data/{instrument}/{timeframe}.parquet, one row per candle, columns
[time (UTC, tz-aware), open, high, low, close, volume], sorted ascending by time
with no duplicate timestamps. This is the file layout Phase 2's backtester reads.
"""
import os
import re
import tempfile
from pathlib import Path

import pandas as pd

from . import config

COLUMNS = ["time", "open", "high", "low", "close", "volume"]

# `instrument` and `timeframe` come from CLI args (fetch_historical.py's
# --pairs, live/narration's --pair) that aren't otherwise constrained against a
# fixed enum, and both get interpolated directly into a filesystem path below —
# validate here, at the actual point of use, rather than trusting every caller.
_INSTRUMENT_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
_TIMEFRAME_RE = re.compile(r"^[A-Za-z0-9]{1,8}$")


def path_for(instrument: str, timeframe: str) -> Path:
    if not _INSTRUMENT_RE.match(instrument):
        raise ValueError(
            f"invalid instrument {instrument!r}; expected OANDA format like 'EUR_USD'"
        )
    if not _TIMEFRAME_RE.match(timeframe):
        raise ValueError(f"invalid timeframe {timeframe!r}")
    return config.DATA_DIR / instrument / f"{timeframe}.parquet"


def load(instrument: str, timeframe: str) -> pd.DataFrame:
    p = path_for(instrument, timeframe)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS).astype({"time": "datetime64[ns, UTC]"})
    return pd.read_parquet(p)


def save_merged(instrument: str, timeframe: str, new_rows: list[dict],
                 existing: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge new_rows into the existing parquet for this instrument/timeframe,
    de-duplicating by timestamp (new_rows wins on conflict) and keeping sorted order.
    Returns the merged DataFrame that was written.

    Pass `existing` when the caller already loaded this file (e.g. to compute a
    resume point via last_timestamp) to avoid reading and deserializing the same
    parquet file twice in one call site.
    """
    if existing is None:
        existing = load(instrument, timeframe)
    new_df = pd.DataFrame(new_rows, columns=COLUMNS)
    if not new_df.empty:
        new_df["time"] = pd.to_datetime(new_df["time"], utc=True)

    combined = pd.concat([existing, new_df], ignore_index=True)
    if not combined.empty:
        combined = (
            combined.drop_duplicates(subset="time", keep="last")
            .sort_values("time")
            .reset_index(drop=True)
        )

    p = path_for(instrument, timeframe)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file in the same directory, then atomically rename over
    # the target — a crash, power loss, or full disk mid-write can otherwise
    # leave a parquet file (holding years of prior history) truncated/corrupt,
    # since to_parquet would have overwritten it in place.
    fd, tmp_path = tempfile.mkstemp(dir=p.parent, suffix=".parquet.tmp")
    os.close(fd)
    try:
        combined.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, p)
    except BaseException:
        os.unlink(tmp_path)
        raise
    return combined


def last_timestamp(instrument: str, timeframe: str):
    df = load(instrument, timeframe)
    if df.empty:
        return None
    return df["time"].max()
