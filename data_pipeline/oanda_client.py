"""Thin client for OANDA's v20 REST API /instruments/{instrument}/candles endpoint.

Only historical candle fetching is implemented here (Phase 1 scope). No order/account
endpoints are touched. Only OANDA_API_KEY is required for this endpoint; account ID is
not needed for candle data (it's captured in config for later phases that do need it,
e.g. streaming prices).
"""
import time
from datetime import datetime, timezone
from typing import Iterator

import requests

from . import config

MAX_CANDLES_PER_REQUEST = 5000
REQUEST_SLEEP_SECONDS = 0.3  # be polite to the API between paginated requests


class OandaAPIError(RuntimeError):
    pass


def _headers() -> dict:
    config.require_credentials()
    return {
        "Authorization": f"Bearer {config.OANDA_API_KEY}",
        "Accept-Datetime-Format": "RFC3339",
    }


def _to_rfc3339(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_time(s: str) -> datetime:
    # OANDA returns RFC3339 like "2024-01-02T03:00:00.000000000Z" (nanosecond precision).
    # Python's datetime only handles microseconds, so truncate the fractional part.
    if "." in s:
        head, frac = s.split(".", 1)
        frac = frac.rstrip("Z")[:6].ljust(6, "0")
        s = f"{head}.{frac}+00:00"
    else:
        s = s.rstrip("Z") + "+00:00"
    return datetime.fromisoformat(s)


def fetch_candles(
    instrument: str,
    granularity: str,
    from_time: datetime,
    to_time: datetime,
) -> Iterator[dict]:
    """Yield complete candles for `instrument`/`granularity` in [from_time, to_time].

    Handles OANDA's 5000-candle-per-request cap by paginating on `from`, and drops
    any still-forming ("complete": false) candle, since an in-progress candle's OHLC
    values would change on every re-fetch.
    """
    url = f"{config.oanda_host()}/v3/instruments/{instrument}/candles"
    cursor = from_time
    session = requests.Session()
    last_emitted_time: datetime | None = None

    while cursor <= to_time:
        params = {
            "granularity": granularity,
            "price": "M",  # midpoint OHLC
            "from": _to_rfc3339(cursor),
            "count": MAX_CANDLES_PER_REQUEST,
        }
        resp = session.get(url, headers=_headers(), params=params, timeout=30)
        if resp.status_code != 200:
            raise OandaAPIError(
                f"OANDA request failed ({resp.status_code}) for {instrument}/{granularity} "
                f"from={params['from']}: {resp.text}"
            )
        payload = resp.json()
        candles = payload.get("candles", [])
        if not candles:
            break

        page_last_time = cursor
        for c in candles:
            if not c.get("complete", False):
                continue
            t = _parse_time(c["time"])
            page_last_time = t
            if t > to_time:
                break
            if last_emitted_time is not None and t <= last_emitted_time:
                continue  # already emitted (pagination overlap on the `from` boundary)
            mid = c["mid"]
            yield {
                "time": t,
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": int(c["volume"]),
            }
            last_emitted_time = t

        if len(candles) < MAX_CANDLES_PER_REQUEST:
            break  # fewer than a full page means we've reached the latest available data

        if page_last_time <= cursor:
            break  # no forward progress; avoid an infinite loop

        cursor = page_last_time
        time.sleep(REQUEST_SLEEP_SECONDS)
