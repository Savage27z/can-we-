"""Fetches and caches the Forex Factory weekly economic calendar (Phase 5).

The source is an unofficial, keyless JSON feed of the CURRENT calendar week only:
no historical data (so the news filter can't be backtested) and no next-week
file. Because it's unofficial and rate-limited, responses are cached on disk and
a stale cache is preferred over failing outright when a refresh errors.
"""
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

from data_pipeline import config as data_config

FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CACHE_TTL = timedelta(minutes=30)
REQUEST_TIMEOUT_SECONDS = 15


class NewsFeedError(RuntimeError):
    pass


@dataclass(frozen=True)
class CalendarEvent:
    title: str
    currency: str
    time_utc: datetime
    impact: str


@dataclass
class Calendar:
    events: list[CalendarEvent]
    fetched_at: datetime


def _cache_path() -> Path:
    return data_config.DATA_DIR / "news_cache" / "ff_calendar_thisweek.json"


def parse_events(raw: object) -> list[CalendarEvent]:
    if not isinstance(raw, list):
        raise NewsFeedError(f"calendar feed returned {type(raw).__name__}, expected a list")
    try:
        return [
            CalendarEvent(
                title=e["title"],
                currency=e["country"],
                time_utc=datetime.fromisoformat(e["date"]).astimezone(timezone.utc),
                impact=e["impact"],
            )
            for e in raw
        ]
    except (KeyError, ValueError, TypeError, AttributeError) as err:
        raise NewsFeedError(f"calendar feed has an unexpected event shape: {err!r}") from err


def _read_cache() -> Optional[Calendar]:
    try:
        payload = json.loads(_cache_path().read_text(encoding="utf-8"))
        return Calendar(
            events=parse_events(payload["events"]),
            fetched_at=datetime.fromisoformat(payload["fetched_at"]),
        )
    except (OSError, ValueError, KeyError, TypeError, NewsFeedError):
        return None


def _write_cache(raw: list, fetched_at: datetime) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"fetched_at": fetched_at.isoformat(), "events": raw}, f)
        os.replace(tmp_path, path)
    except BaseException:
        os.unlink(tmp_path)
        raise


def _download() -> list:
    resp = requests.get(FEED_URL, timeout=REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    try:
        return resp.json()
    except ValueError as err:
        raise NewsFeedError(f"calendar feed returned invalid JSON: {err}") from err


def load_calendar(now: Optional[datetime] = None) -> Calendar:
    """Fresh-enough cache -> use it; otherwise refresh; if the refresh fails, fall
    back to whatever cache exists (its `fetched_at` stays visible to callers so
    staleness isn't hidden), and only raise if there is nothing at all.
    """
    now = now or datetime.now(timezone.utc)
    cached = _read_cache()
    if cached is not None and now - cached.fetched_at < CACHE_TTL:
        return cached
    try:
        raw = _download()
        calendar = Calendar(events=parse_events(raw), fetched_at=now)
        _write_cache(raw, now)
        return calendar
    except (requests.RequestException, NewsFeedError) as err:
        if cached is not None:
            return cached
        raise NewsFeedError(f"could not fetch the economic calendar: {err}") from err
