"""Tests for data_pipeline.oanda_client using mocked HTTP responses — no real
OANDA credentials or network access required. Exercises the pagination/dedup
logic that's hard to eyeball-verify but easy to get subtly wrong.
"""
import unittest
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


if __name__ == "__main__":
    unittest.main()
