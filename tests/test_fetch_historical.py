import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from data_pipeline import config, fetch_historical, storage
from data_pipeline.oanda_client import OandaAPIError

T0 = datetime(2020, 1, 1, tzinfo=timezone.utc)


def candle(hours: int, close: float = 1.0) -> dict:
    return {"time": T0 + timedelta(hours=hours), "open": close, "high": close, "low": close,
            "close": close, "volume": 1, "spread": 0.0001}


def seed(pair: str, tf: str, hours: list[int]) -> None:
    storage.save_merged(pair, tf, [candle(h) for h in hours])


class FakeFeed:
    """Stands in for fetch_candles. Each call returns the candles at or after the
    requested start; `fail_after` makes a call raise after yielding that many rows."""

    def __init__(self, hours: list[int], fail_after: dict[int, int] | None = None):
        self.hours = hours
        self.fail_after = fail_after or {}
        self.calls: list[datetime] = []

    def __call__(self, instrument, granularity, from_time, to_time):
        call_number = len(self.calls)
        self.calls.append(from_time)
        limit = self.fail_after.get(call_number)
        yielded = 0
        for h in self.hours:
            row = candle(h)
            if row["time"] < from_time:
                continue
            if limit is not None and yielded >= limit:
                raise OandaAPIError("connection dropped")
            yield row
            yielded += 1


class FetchTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._orig = config.DATA_DIR
        config.DATA_DIR = self._tmp
        self.addCleanup(lambda: (setattr(config, "DATA_DIR", self._orig),
                                 shutil.rmtree(self._tmp, ignore_errors=True)))
        quiet = patch("builtins.print")
        quiet.start()
        self.addCleanup(quiet.stop)
        self.sleeps: list[float] = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)

    def stored_hours(self, pair="EUR_USD", tf="H1"):
        df = storage.load(pair, tf)
        return [int((t - T0).total_seconds() // 3600) for t in df["time"]]


class BackfillTests(FetchTestCase):
    def test_data_that_starts_late_is_fetched_again_from_the_requested_start(self):
        seed("EUR_USD", "H1", [1000, 1001])
        feed = FakeFeed(list(range(0, 1002)))
        with patch.object(fetch_historical, "fetch_candles", feed):
            fetch_historical.fetch_one("EUR_USD", "H1", "H1", 8, False, since=T0)
        self.assertEqual(feed.calls, [T0])
        self.assertEqual(self.stored_hours(), list(range(0, 1002)))

    def test_data_that_already_reaches_back_is_only_topped_up(self):
        seed("EUR_USD", "H1", [0, 1, 2])
        feed = FakeFeed(list(range(0, 6)))
        with patch.object(fetch_historical, "fetch_candles", feed):
            fetch_historical.fetch_one("EUR_USD", "H1", "H1", 8, False, since=T0)
        self.assertEqual(feed.calls, [T0 + timedelta(hours=2)])     # from the newest stored
        self.assertEqual(self.stored_hours(), list(range(0, 6)))

    def test_a_start_within_the_tolerance_counts_as_reaching_back(self):
        seed("EUR_USD", "H1", [24 * 5, 24 * 5 + 1])                # five days after `since`
        feed = FakeFeed(list(range(0, 200)))
        with patch.object(fetch_historical, "fetch_candles", feed):
            fetch_historical.fetch_one("EUR_USD", "H1", "H1", 8, False, since=T0)
        self.assertEqual(feed.calls, [T0 + timedelta(hours=24 * 5 + 1)])

    def test_the_first_ever_fetch_uses_since_when_given(self):
        feed = FakeFeed(list(range(0, 5)))
        with patch.object(fetch_historical, "fetch_candles", feed):
            fetch_historical.fetch_one("EUR_USD", "H1", "H1", 8, False, since=T0)
        self.assertEqual(feed.calls, [T0])


class ResumeTests(FetchTestCase):
    def test_a_failure_part_way_resumes_from_the_last_candle_received(self):
        # A backfill over an existing recent store: resuming from the newest STORED
        # candle would skip everything between the failure point and the old data.
        seed("EUR_USD", "H1", [1000, 1001])
        feed = FakeFeed(list(range(0, 1002)), fail_after={0: 300})
        with patch.object(fetch_historical, "fetch_candles", feed):
            fetch_historical.fetch_one("EUR_USD", "H1", "H1", 8, False, since=T0,
                                       attempts=3, wait_seconds=7, sleep=self.sleep)
        self.assertEqual(feed.calls, [T0, T0 + timedelta(hours=299)])
        self.assertEqual(self.sleeps, [7])
        self.assertEqual(self.stored_hours(), list(range(0, 1002)))   # no hole

    def test_the_count_ignores_the_boundary_candle_read_twice(self):
        feed = FakeFeed(list(range(0, 10)), fail_after={0: 4})
        with patch.object(fetch_historical, "fetch_candles", feed):
            count = fetch_historical.fetch_one("EUR_USD", "H1", "H1", 8, False, since=T0,
                                               attempts=2, wait_seconds=0, sleep=self.sleep)
        self.assertEqual(count, 10)

    def test_giving_up_keeps_what_was_received_and_raises(self):
        feed = FakeFeed(list(range(0, 50)), fail_after={0: 20, 1: 0})
        with patch.object(fetch_historical, "fetch_candles", feed):
            with self.assertRaises(OandaAPIError):
                fetch_historical.fetch_one("EUR_USD", "H1", "H1", 8, False, since=T0,
                                           attempts=2, wait_seconds=0, sleep=self.sleep)
        self.assertEqual(self.stored_hours(), list(range(0, 20)))

    def test_a_single_attempt_still_saves_partial_progress_and_raises(self):
        feed = FakeFeed(list(range(0, 50)), fail_after={0: 5})
        with patch.object(fetch_historical, "fetch_candles", feed):
            with self.assertRaises(OandaAPIError):
                fetch_historical.fetch_one("EUR_USD", "H1", "H1", 8, False, since=T0)
        self.assertEqual(self.stored_hours(), list(range(0, 5)))
        self.assertEqual(self.sleeps, [])


class FetchManyTests(FetchTestCase):
    def test_a_pair_that_keeps_failing_is_recorded_and_the_next_pair_still_runs(self):
        def feed(instrument, granularity, from_time, to_time):
            if instrument == "GBP_USD":
                raise OandaAPIError("no such thing")
            yield candle(1)

        with patch.object(fetch_historical, "fetch_candles", feed):
            total, failures = fetch_historical.fetch_many(
                ["GBP_USD", "EUR_USD"], ["H1"], 8, False, False,
                attempts=2, wait_seconds=0, sleep=self.sleep)
        self.assertEqual(total, 1)
        self.assertEqual([(p, tf) for p, tf, _ in failures], [("GBP_USD", "H1")])
        self.assertEqual(self.stored_hours("EUR_USD"), [1])

    def test_from_start_uses_each_pairs_catalog_start(self):
        seen = {}

        def feed(instrument, granularity, from_time, to_time):
            seen[instrument] = from_time
            return iter(())

        with patch.object(fetch_historical, "fetch_candles", feed):
            fetch_historical.fetch_many(["EUR_USD", "USD_CNH"], ["D"], 8, False, True)
        self.assertLess(seen["EUR_USD"].year, 2003)          # history begins in 2002
        self.assertEqual(seen["USD_CNH"].year, 2011)


class ParallelTests(FetchTestCase):
    def test_several_workers_give_the_same_result_as_one(self):
        pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "NZD_USD"]

        def feed(instrument, granularity, from_time, to_time):
            for h in range(3):
                yield candle(h)

        with patch.object(fetch_historical, "fetch_candles", feed):
            total, failures = fetch_historical.fetch_many(pairs, ["H1", "H4"], 8, False, False,
                                                          workers=3)
        self.assertEqual(total, 5 * 2 * 3)
        self.assertEqual(failures, [])
        for pair in pairs:
            self.assertEqual(self.stored_hours(pair, "H1"), [0, 1, 2])
            self.assertEqual(self.stored_hours(pair, "H4"), [0, 1, 2])

    def test_failures_from_every_worker_are_collected(self):
        def feed(instrument, granularity, from_time, to_time):
            if instrument in ("GBP_USD", "AUD_USD"):
                raise OandaAPIError("bad")
            yield candle(0)

        with patch.object(fetch_historical, "fetch_candles", feed):
            _, failures = fetch_historical.fetch_many(
                ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"], ["H1"], 8, False, False,
                attempts=1, workers=2)
        self.assertEqual(sorted(p for p, _, _ in failures), ["AUD_USD", "GBP_USD"])


class HistoryStartTests(unittest.TestCase):
    def test_it_is_the_day_before_the_catalog_date(self):
        self.assertEqual(fetch_historical.history_start("EUR_USD"),
                         datetime(2002, 5, 5, tzinfo=timezone.utc))

    def test_an_unknown_instrument_has_none(self):
        self.assertIsNone(fetch_historical.history_start("XXX_YYY"))


if __name__ == "__main__":
    unittest.main()
