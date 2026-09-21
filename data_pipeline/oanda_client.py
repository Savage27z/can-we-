"""Thin, read-only client for OANDA's v20 REST API.

Two things are read: historical candles (/instruments/{instrument}/candles, which needs
only OANDA_API_KEY) and the list of instruments the account can trade
(/accounts/{id}/instruments, which also needs OANDA_ACCOUNT_ID). No order or position
endpoint is touched anywhere in this module.
"""
import time
from datetime import datetime, timezone
from typing import Iterator

import requests

from . import config

MAX_CANDLES_PER_REQUEST = 5000
REQUEST_SLEEP_SECONDS = 0.3  # be polite to the API between paginated requests

# A long backfill is thousands of requests; one dropped connection must not lose it.
# These statuses are transient (rate limit, server trouble); anything else (bad token,
# unknown instrument) will not fix itself and fails at once.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
BACKOFF_SECONDS = (1, 2, 4)            # waits between attempts, so 4 attempts in all
REQUEST_TIMEOUT = (10, 30)             # (connect, read) seconds


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


def get_json(session: requests.Session, url: str, params: dict | None = None) -> dict:
    """GET with retries on connection failures and transient statuses. Exhausted
    retries raise OandaAPIError, so callers that save partial progress on that error
    also cover a network outage, not only an API refusal."""
    last_problem = "no attempt made"
    for attempt in range(len(BACKOFF_SECONDS) + 1):
        if attempt:
            time.sleep(BACKOFF_SECONDS[attempt - 1])
        try:
            resp = session.get(url, headers=_headers(), params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as err:
            last_problem = f"{type(err).__name__}: {err}"
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in RETRY_STATUSES:
            last_problem = f"HTTP {resp.status_code}: {resp.text[:200]}"
            continue
        raise OandaAPIError(
            f"OANDA request failed ({resp.status_code}) for {url} params={params}: {resp.text}"
        )
    raise OandaAPIError(
        f"OANDA request failed after {len(BACKOFF_SECONDS) + 1} attempts for {url} "
        f"params={params}: {last_problem}"
    )


def _spread(candle: dict) -> float | None:
    """Closing ask minus closing bid, when the response carries both."""
    bid, ask = candle.get("bid"), candle.get("ask")
    if not bid or not ask:
        return None
    return round(float(ask["c"]) - float(bid["c"]), 10)


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

    Prices are the midpoint, as before; each candle also carries `spread` (closing ask
    minus closing bid) so a backtest can charge realistic trading costs.
    """
    url = f"{config.oanda_host()}/v3/instruments/{instrument}/candles"
    cursor = from_time
    session = requests.Session()
    last_emitted_time: datetime | None = None

    while cursor <= to_time:
        params = {
            "granularity": granularity,
            "price": "MBA",  # midpoint OHLC, plus bid and ask for the spread
            "from": _to_rfc3339(cursor),
            "count": MAX_CANDLES_PER_REQUEST,
        }
        payload = get_json(session, url, params)
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
            row = {
                "time": t,
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": int(c["volume"]),
            }
            spread = _spread(c)
            if spread is not None:
                row["spread"] = spread
            yield row
            last_emitted_time = t

        if len(candles) < MAX_CANDLES_PER_REQUEST:
            break  # fewer than a full page means we've reached the latest available data

        if page_last_time <= cursor:
            break  # no forward progress; avoid an infinite loop

        cursor = page_last_time
        time.sleep(REQUEST_SLEEP_SECONDS)


def fetch_instruments() -> list[dict]:
    """Raw instrument records for the configured account (read-only)."""
    if not config.OANDA_ACCOUNT_ID:
        raise RuntimeError(
            "OANDA_ACCOUNT_ID is not set; the instrument list is per account. Copy it "
            "from the OANDA Hub (the id looks like 101-004-1234567-001)."
        )
    url = f"{config.oanda_host()}/v3/accounts/{config.OANDA_ACCOUNT_ID}/instruments"
    return get_json(requests.Session(), url)["instruments"]


def first_candle_time(instrument: str, granularity: str = "D",
                      since: datetime = datetime(2000, 1, 1, tzinfo=timezone.utc)) -> datetime | None:
    """Open time of the earliest candle OANDA has for the instrument, or None."""
    url = f"{config.oanda_host()}/v3/instruments/{instrument}/candles"
    params = {"granularity": granularity, "price": "M", "from": _to_rfc3339(since), "count": 1}
    candles = get_json(requests.Session(), url, params).get("candles", [])
    return _parse_time(candles[0]["time"]) if candles else None
