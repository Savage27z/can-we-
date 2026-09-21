"""News filter (Phase 5): flags or suppresses signals around high-impact events.

An overlay only — it never generates or changes a signal, bias, or level. Its
parameters are judgment defaults, NOT backtested: the calendar feed has no
historical data, so there is no way to measure whether these windows help.
"""
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from .feed import Calendar, NewsFeedError, load_calendar

log = logging.getLogger(__name__)

IMPACT_HIGH = "High"
BLACKOUT_BEFORE = timedelta(minutes=60)
BLACKOUT_AFTER = timedelta(minutes=60)
UPCOMING_HORIZON = timedelta(hours=72)
MAX_UPCOMING = 5
# The feed only holds the current calendar week; if "now" is further than this
# outside its date range, the feed hasn't rolled over and can't vouch for now.
COVERAGE_TOLERANCE = timedelta(days=1)
# A calendar older than this means refreshes have been failing and a cached copy is
# being served; event times don't change, but revisions and new events do.
MAX_CALENDAR_AGE = timedelta(hours=12)

_PAIR_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")


@dataclass
class NewsEventView:
    title: str
    currency: str
    time_utc: str
    when: str  # precomputed, e.g. "in 2h 15m" or "45m ago", so the LLM never does time math


@dataclass
class NewsStatus:
    status: str  # "clear" | "blackout" | "unavailable" | "not_checked"
    reason: Optional[str] = None
    blackout_events: list[NewsEventView] = field(default_factory=list)
    upcoming_events: list[NewsEventView] = field(default_factory=list)
    calendar_fetched_at: Optional[str] = None

    @classmethod
    def not_checked(cls) -> "NewsStatus":
        return cls(status="not_checked")


def relevant_currencies(pair: str) -> set[str]:
    if not _PAIR_RE.match(pair):
        raise ValueError(f"invalid pair {pair!r}; expected OANDA format like 'EUR_USD'")
    return set(pair.split("_")) | {"All"}


def describe_when(delta: timedelta) -> str:
    total_minutes = round(abs(delta).total_seconds() / 60)
    if total_minutes == 0:
        return "now"
    hours, minutes = divmod(total_minutes, 60)
    if hours and minutes:
        text = f"{hours}h {minutes}m"
    elif hours:
        text = f"{hours}h"
    else:
        text = f"{minutes}m"
    return f"in {text}" if delta > timedelta(0) else f"{text} ago"


def _view(event, now: datetime) -> NewsEventView:
    return NewsEventView(
        title=event.title,
        currency=event.currency,
        time_utc=event.time_utc.isoformat(),
        when=describe_when(event.time_utc - now),
    )


def evaluate(calendar: Calendar, pair: str, now: datetime,
             clock: Optional[datetime] = None) -> NewsStatus:
    """`now` is the moment judged; `clock` (default `now`) is the wall clock the
    calendar's age is measured against."""
    fetched_at = calendar.fetched_at.isoformat()
    if not calendar.events:
        return NewsStatus(status="unavailable", reason="calendar contains no events",
                          calendar_fetched_at=fetched_at)

    age = (clock or now) - calendar.fetched_at
    if age > MAX_CALENDAR_AGE:
        return NewsStatus(
            status="unavailable",
            reason=(f"calendar data is {round(age.total_seconds() / 3600)}h old; "
                    f"refreshing it has been failing"),
            calendar_fetched_at=fetched_at,
        )

    first = min(e.time_utc for e in calendar.events)
    last = max(e.time_utc for e in calendar.events)
    if not (first - COVERAGE_TOLERANCE <= now <= last + COVERAGE_TOLERANCE):
        return NewsStatus(
            status="unavailable",
            reason=(f"calendar covers {first.isoformat()} to {last.isoformat()}, which "
                    f"does not include the current time; the feed has likely not "
                    f"rolled over to the new week yet"),
            calendar_fetched_at=fetched_at,
        )

    currencies = relevant_currencies(pair)
    relevant = sorted(
        (e for e in calendar.events if e.impact == IMPACT_HIGH and e.currency in currencies),
        key=lambda e: e.time_utc,
    )
    blackout = [e for e in relevant
                if e.time_utc - BLACKOUT_BEFORE <= now <= e.time_utc + BLACKOUT_AFTER]
    upcoming = [e for e in relevant
                if e.time_utc > now and e not in blackout and e.time_utc <= now + UPCOMING_HORIZON]

    return NewsStatus(
        status="blackout" if blackout else "clear",
        blackout_events=[_view(e, now) for e in blackout],
        upcoming_events=[_view(e, now) for e in upcoming[:MAX_UPCOMING]],
        calendar_fetched_at=fetched_at,
    )


def check_news(pair: str, now: Optional[datetime] = None,
               at: Optional[datetime] = None) -> NewsStatus:
    """`now` is the wall clock: it drives calendar caching and freshness. `at` is the
    moment being judged (default: `now`), e.g. an earlier candle's confirmation time.
    Keeping them apart stops a historical `at` from discarding a good cache or
    stamping it with the wrong retrieval time."""
    now = now or datetime.now(timezone.utc)
    at = at or now
    relevant_currencies(pair)  # validate the pair up front, before any network call
    try:
        return evaluate(load_calendar(now), pair, at, clock=now)
    except NewsFeedError as err:
        return NewsStatus(status="unavailable", reason=str(err))
    except Exception as err:
        # News is an overlay: whatever goes wrong here must never take down an
        # otherwise valid report or alert, only mark the news check unavailable.
        log.exception("news check failed unexpectedly")
        return NewsStatus(status="unavailable", reason=f"news check failed ({type(err).__name__})")
