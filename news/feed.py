"""Fetches and caches the Forex Factory weekly economic calendar (Phase 5).

The source is an unofficial, keyless JSON feed of the CURRENT calendar week only:
no historical data (so the news filter can't be backtested) and no next-week
file. Because it's unofficial and rate-limited, responses are cached on disk, a
stale cache is preferred over failing outright, and a failing feed is not hit
again for a few minutes.
"""
import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

from data_pipeline import config as data_config

log = logging.getLogger(__name__)

FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CACHE_TTL = timedelta(minutes=30)
REQUEST_TIMEOUT_SECONDS = 15
# After a failed refresh, don't contact the feed again for this long: it's
# rate-limited, and every pair/alert/report call would otherwise retry it.
RETRY_BACKOFF = timedelta(minutes=5)
# A cache stamped further in the future than this is clock skew or corruption, and
# would otherwise look "fresh" indefinitely.
MAX_CLOCK_SKEW = timedelta(minutes=5)

_lock = threading.Lock()
_last_failure: Optional[datetime] = None


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
    """Malformed events are skipped individually rather than discarding the whole
    calendar. An event with no UTC offset is rejected: `astimezone` would silently
    read it as the server's local time and shift the event."""
    if not isinstance(raw, list):
        raise NewsFeedError(f"calendar feed returned {type(raw).__name__}, expected a list")
    events, skipped = [], 0
    for item in raw:
        try:
            when = datetime.fromisoformat(item["date"])
            if when.tzinfo is None:
                raise ValueError("date has no UTC offset")
            events.append(CalendarEvent(
                title=item["title"],
                currency=item["country"],
                time_utc=when.astimezone(timezone.utc),
                impact=item["impact"],
            ))
        except (KeyError, ValueError, TypeError, AttributeError, OverflowError):
            skipped += 1
    if skipped:
        log.warning("skipped %d malformed calendar event(s) out of %d", skipped, len(raw))
    if not events:
        raise NewsFeedError("calendar feed contained no usable events")
    return events


def _read_cache() -> Optional[Calendar]:
    try:
        payload = json.loads(_cache_path().read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
        if fetched_at.tzinfo is None:
            return None
        return Calendar(events=parse_events(payload["events"]), fetched_at=fetched_at)
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
    back to whatever cache exists (its `fetched_at` stays visible so callers can
    judge staleness), and only raise if there is nothing at all.
    """
    global _last_failure
    now = now or datetime.now(timezone.utc)
    with _lock:  # concurrent callers wait, then reuse the first one's fresh cache
        cached = _read_cache()
        if cached is not None and cached.fetched_at - now > MAX_CLOCK_SKEW:
            cached = None
        if cached is not None and now - cached.fetched_at < CACHE_TTL:
            return cached

        if _last_failure is not None and now - _last_failure < RETRY_BACKOFF:
            if cached is not None:
                return cached
            raise NewsFeedError("calendar feed failed moments ago; not retrying yet")

        try:
            raw = _download()
            calendar = Calendar(events=parse_events(raw), fetched_at=now)
        except (requests.RequestException, NewsFeedError) as err:
            _last_failure = now
            if cached is not None:
                log.warning("calendar refresh failed, using the cached copy: %s", err)
                return cached
            raise NewsFeedError(f"could not fetch the economic calendar: {err}") from err

        _last_failure = None
        try:
            _write_cache(raw, now)
        except OSError as err:
            # The calendar is good; failing to cache it must not discard it.
            log.warning("could not write the calendar cache: %s", err)
        return calendar
