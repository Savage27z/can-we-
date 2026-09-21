"""Tests for data_pipeline.oanda_client using mocked HTTP responses — no real
OANDA credentials or network access required. Exercises the pagination/dedup
logic that's hard to eyeball-verify but easy to get subtly wrong.
"""
import unittest
import requests
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from data_pipeline import config, oanda_client


def make_candle(t: datetime, o: float, h: float, l: float, c: float, v: int = 100,
                 complete: bool = True) -> dict:
    return {
        "time": t.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        "complete": complete,
        "volume": v,
        "mid": {"o": f"{o}", "h": f"{h}", "l": f"{l}", "c": f"{c}"},
    }


def fake_response(candles: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"candles": candles}
    return resp


class ParseTimeTests(unittest.TestCase):
    def test_nanosecond_precision_truncated_to_microseconds(self):
        dt = oanda_client._parse_time("2024-01-02T03:04:05.123456789Z")
        self.assertEqual(dt.year, 2024)
        self.assertEqual(dt.microsecond, 123456)
        self.assertEqual(dt.tzinfo.utcoffset(dt), timedelta(0))

    def test_no_fractional_seconds(self):
        dt = oanda_client._parse_time("2024-01-02T03:04:05Z")
        self.assertEqual(dt.second, 5)
        self.assertEqual(dt.microsecond, 0)


class FetchCandlesTests(unittest.TestCase):
    def setUp(self):
        config.OANDA_API_KEY = "test-token"

    def test_filters_incomplete_candles(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        candles = [
            make_candle(base, 1.0, 1.1, 0.9, 1.05),
            make_candle(base + timedelta(hours=1), 1.05, 1.1, 1.0, 1.08, complete=False),
        ]
        with patch.object(oanda_client.requests.Session, "get", return_value=fake_response(candles)):
            result = list(oanda_client.fetch_candles(
                "EUR_USD", "H1", base, base + timedelta(hours=2)
            ))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["close"], 1.05)

    def test_stops_at_to_time_bound(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        candles = [
            make_candle(base, 1.0, 1.1, 0.9, 1.0),
            make_candle(base + timedelta(hours=1), 1.0, 1.1, 0.9, 1.01),
            make_candle(base + timedelta(hours=2), 1.0, 1.1, 0.9, 1.02),  # beyond to_time
        ]
        to_time = base + timedelta(hours=1, minutes=30)
        with patch.object(oanda_client.requests.Session, "get", return_value=fake_response(candles)):
            result = list(oanda_client.fetch_candles("EUR_USD", "H1", base, to_time))
        self.assertEqual(len(result), 2)

    def test_pagination_dedups_boundary_candle(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # Page 1: a full page (forces a second request), last candle at hour 4999.
        page1 = [
            make_candle(base + timedelta(hours=i), 1.0, 1.1, 0.9, 1.0 + i / 10000)
            for i in range(oanda_client.MAX_CANDLES_PER_REQUEST)
        ]
        # Page 2 starts at the same timestamp as page 1's last candle (OANDA's `from`
        # is inclusive), then has one genuinely new candle.
        overlap_time = base + timedelta(hours=oanda_client.MAX_CANDLES_PER_REQUEST - 1)
        page2 = [
            make_candle(overlap_time, 1.0, 1.1, 0.9, 999.0),  # stale duplicate, must be dropped
            make_candle(overlap_time + timedelta(hours=1), 1.0, 1.1, 0.9, 2.0),
        ]
        responses = [fake_response(page1), fake_response(page2)]
        with patch.object(oanda_client.requests.Session, "get", side_effect=responses):
            result = list(oanda_client.fetch_candles(
                "EUR_USD", "H1", base, overlap_time + timedelta(hours=1)
            ))
        times = [r["time"] for r in result]
        self.assertEqual(len(times), len(set(times)), "duplicate timestamp emitted")
        self.assertEqual(len(result), oanda_client.MAX_CANDLES_PER_REQUEST + 1)
        self.assertEqual(result[-1]["close"], 2.0)  # not the stale 999.0 duplicate

    def test_raises_on_http_error(self):
        resp = MagicMock()
        resp.status_code = 401
        resp.text = "Unauthorized"
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        with patch.object(oanda_client.requests.Session, "get", return_value=resp):
            with self.assertRaises(oanda_client.OandaAPIError):
                list(oanda_client.fetch_candles("EUR_USD", "H1", base, base))


def status_response(code: int, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = code
    resp.text = text
    resp.json.return_value = {"candles": []}
    return resp


class RetryTests(unittest.TestCase):
    """A long backfill is thousands of requests; a dropped connection must not end it."""

    def setUp(self):
        config.OANDA_API_KEY = "test-token"
        sleeper = patch.object(oanda_client.time, "sleep")
        self.sleep = sleeper.start()
        self.addCleanup(sleeper.stop)
        self.session = oanda_client.requests.Session()

    def test_a_dropped_connection_is_retried_and_then_succeeds(self):
        good = fake_response([])
        with patch.object(oanda_client.requests.Session, "get",
                          side_effect=[requests.ConnectTimeout("no route"), good]) as get:
            body = oanda_client.get_json(self.session, "http://x", {})
        self.assertEqual(body, {"candles": []})
        self.assertEqual(get.call_count, 2)
        self.sleep.assert_called_once_with(oanda_client.BACKOFF_SECONDS[0])

    def test_transient_statuses_are_retried(self):
        for code in (429, 500, 502, 503, 504):
            with patch.object(oanda_client.requests.Session, "get",
                              side_effect=[status_response(code), fake_response([])]) as get:
                oanda_client.get_json(self.session, "http://x", {})
            self.assertEqual(get.call_count, 2, code)

    def test_a_permanent_error_fails_at_once_without_retrying(self):
        for code in (400, 401, 403, 404):
            with patch.object(oanda_client.requests.Session, "get",
                              return_value=status_response(code, "nope")) as get:
                with self.assertRaises(oanda_client.OandaAPIError):
                    oanda_client.get_json(self.session, "http://x", {})
            self.assertEqual(get.call_count, 1, code)

    def test_exhausted_retries_raise_the_api_error_type(self):
        attempts = len(oanda_client.BACKOFF_SECONDS) + 1
        with patch.object(oanda_client.requests.Session, "get",
                          side_effect=requests.ConnectionError("down")) as get:
            with self.assertRaises(oanda_client.OandaAPIError) as ctx:
                oanda_client.get_json(self.session, "http://x", {})
        self.assertEqual(get.call_count, attempts)
        self.assertIn("ConnectionError", str(ctx.exception))

    def test_the_token_never_appears_in_an_error_message(self):
        with patch.object(oanda_client.requests.Session, "get",
                          return_value=status_response(500, "boom")):
            with self.assertRaises(oanda_client.OandaAPIError) as ctx:
                oanda_client.get_json(self.session, "http://x", {})
        self.assertNotIn("test-token", str(ctx.exception))


class SpreadTests(unittest.TestCase):
    def setUp(self):
        config.OANDA_API_KEY = "test-token"

    def candle_with_quotes(self, t):
        c = make_candle(t, 1.0, 1.1, 0.9, 1.05)
        c["bid"] = {"o": "1.0", "h": "1.1", "l": "0.9", "c": "1.04994"}
        c["ask"] = {"o": "1.0", "h": "1.1", "l": "0.9", "c": "1.05006"}
        return c

    def test_spread_is_closing_ask_minus_closing_bid_without_float_noise(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        with patch.object(oanda_client.requests.Session, "get",
                          return_value=fake_response([self.candle_with_quotes(base)])):
            (row,) = oanda_client.fetch_candles("EUR_USD", "H1", base, base + timedelta(hours=1))
        self.assertEqual(row["spread"], 0.00012)
        self.assertEqual(row["close"], 1.05)                     # prices stay the midpoint

    def test_a_response_without_quotes_simply_has_no_spread(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        with patch.object(oanda_client.requests.Session, "get",
                          return_value=fake_response([make_candle(base, 1.0, 1.1, 0.9, 1.05)])):
            (row,) = oanda_client.fetch_candles("EUR_USD", "H1", base, base + timedelta(hours=1))
        self.assertNotIn("spread", row)

    def test_the_request_asks_for_mid_bid_and_ask(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        with patch.object(oanda_client.requests.Session, "get",
                          return_value=fake_response([])) as get:
            list(oanda_client.fetch_candles("EUR_USD", "H1", base, base))
        self.assertEqual(get.call_args.kwargs["params"]["price"], "MBA")


class InstrumentListTests(unittest.TestCase):
    def test_first_candle_time_reads_the_earliest_candle(self):
        config.OANDA_API_KEY = "test-token"
        payload = fake_response([make_candle(datetime(2002, 5, 6, 21, tzinfo=timezone.utc),
                                             1.0, 1.1, 0.9, 1.0)])
        with patch.object(oanda_client.requests.Session, "get", return_value=payload):
            found = oanda_client.first_candle_time("EUR_USD")
        self.assertEqual(found, datetime(2002, 5, 6, 21, tzinfo=timezone.utc))

    def test_first_candle_time_is_none_when_there_is_no_history(self):
        config.OANDA_API_KEY = "test-token"
        with patch.object(oanda_client.requests.Session, "get", return_value=fake_response([])):
            self.assertIsNone(oanda_client.first_candle_time("EUR_USD"))

    def test_the_instrument_list_needs_an_account_id(self):
        config.OANDA_API_KEY = "test-token"
        original = config.OANDA_ACCOUNT_ID
        config.OANDA_ACCOUNT_ID = None
        try:
            with self.assertRaises(RuntimeError):
                oanda_client.fetch_instruments()
        finally:
            config.OANDA_ACCOUNT_ID = original


if __name__ == "__main__":
    unittest.main()
