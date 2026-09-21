"""Whether a live snapshot is too old to act on, measured in market-open time.

Wall-clock age alone would call every weekend an outage, so the closure is skipped:
a snapshot is stale only if the market has been open for longer than `max_age`
since its newest candle closed. The closure is approximated as Friday 21:00 UTC to
Sunday 22:00 UTC, slightly wider than the real one, so it never raises a false alarm.
"""
from datetime import datetime, timedelta, timezone

_STEP = timedelta(minutes=15)


def market_open(moment: datetime) -> bool:
    moment = moment.astimezone(timezone.utc)
    day, hour = moment.weekday(), moment.hour
    if day == 4 and hour >= 21:
        return False
    if day == 5:
        return False
    if day == 6 and hour < 22:
        return False
    return True


def open_age(as_of: datetime, now: datetime) -> timedelta:
    """How long the market has been open between `as_of` and `now`."""
    total = timedelta(0)
    t = as_of
    while t < now:
        if market_open(t):
            total += min(_STEP, now - t)
        t += _STEP
    return total


def is_stale(as_of: datetime, now: datetime, max_age: timedelta) -> bool:
    if now - as_of <= max_age:  # cheap exit; also covers clock skew
        return False
    return open_age(as_of, now) > max_age
