"""Tests for live.refresh's cross-timeframe sync check.

The scenarios use the real shapes of OANDA data: the weekend closure removes H1
candles (so the H4/Daily 'lag' in clock time is large on Monday), and Daily candles
are labelled by their 17:00 New York open, so on Monday the newest complete Daily
candle is labelled the previous Thursday.
"""
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from data_pipeline import config, storage
from live.refresh import DataSyncError, _verify_synced, sync_problem


def utc(y, m, d, h=0):
    return pd.Timestamp(datetime(y, m, d, h, tzinfo=timezone.utc))


def hourly(start, end):
    """H1 candle opens from start to end inclusive."""
    return pd.Series(pd.date_range(start, end, freq="1h"))


class SyncProblemTests(unittest.TestCase):
    def test_monday_afternoon_is_healthy(self):
        # Real case that a fixed clock-time limit rejected: H1 up to Mon 14:00, H4 up
        # to Mon 09:00, and the newest complete Daily candle labelled Thu 21:00.
        h1 = hourly(utc(2026, 9, 13, 21), utc(2026, 9, 14, 14))
        self.assertIsNone(sync_problem(h1, utc(2026, 9, 14, 9), utc(2026, 9, 10, 21)))

    def test_sunday_evening_reopen_is_healthy(self):
        # Market just reopened: H1 has its first candles, H4/Daily still end Friday.
        h1 = hourly(utc(2026, 9, 13, 21), utc(2026, 9, 13, 23))
        self.assertIsNone(sync_problem(h1, utc(2026, 9, 11, 17), utc(2026, 9, 10, 21)))

    def test_midweek_healthy(self):
        h1 = hourly(utc(2026, 9, 13, 21), utc(2026, 9, 16, 15))
        self.assertIsNone(sync_problem(h1, utc(2026, 9, 16, 7), utc(2026, 9, 15, 21)))

    def test_stale_h4_midweek_is_detected(self):
        h1 = hourly(utc(2026, 9, 13, 21), utc(2026, 9, 16, 15))
        problem = sync_problem(h1, utc(2026, 9, 16, 3), utc(2026, 9, 15, 21))
        self.assertIn("H4 data is stale", problem)

    def test_stale_daily_midweek_is_detected(self):
        h1 = hourly(utc(2026, 9, 13, 21), utc(2026, 9, 16, 15))
        problem = sync_problem(h1, utc(2026, 9, 16, 7), utc(2026, 9, 13, 21))
        self.assertIn("Daily data is stale", problem)

    def test_h4_boundary_is_exactly_five_uncovered_candles(self):
        h4_last = utc(2026, 9, 16, 7)  # closes 11:00, so H1 candles from 11:00 are uncovered
        five = hourly(utc(2026, 9, 15, 0), utc(2026, 9, 16, 15))  # 11..15 = 5 candles
        six = hourly(utc(2026, 9, 15, 0), utc(2026, 9, 16, 16))   # 11..16 = 6 candles
        d_last = utc(2026, 9, 15, 21)
        self.assertIsNone(sync_problem(five, h4_last, d_last))
        self.assertIn("H4", sync_problem(six, h4_last, d_last))

    def test_missing_data_is_reported(self):
        h1 = hourly(utc(2026, 9, 16, 0), utc(2026, 9, 16, 5))
        self.assertIn("no H1", sync_problem(pd.Series([], dtype="datetime64[ns, UTC]"),
                                            utc(2026, 9, 16), utc(2026, 9, 15)))
        self.assertIn("no H4", sync_problem(h1, None, utc(2026, 9, 15)))
        self.assertIn("no Daily", sync_problem(h1, utc(2026, 9, 16), None))

    def test_every_hour_of_a_realistic_week_is_healthy(self):
        # Market hours only: Sun 21:00 UTC through Fri 20:00 UTC. For each hour, the
        # newest complete H4 candle (opens on the 01/05/09/13/17/21 grid) and Daily
        # candle (labelled 21:00, complete 24h later) are what a healthy fetch holds.
        market = list(pd.date_range(utc(2026, 9, 13, 21), utc(2026, 9, 18, 20), freq="1h"))
        h4_opens = [t for t in pd.date_range(utc(2026, 9, 10, 1), utc(2026, 9, 18, 17), freq="4h")
                    if not (t.weekday() == 5 or (t.weekday() == 4 and t.hour >= 21)
                            or (t.weekday() == 6 and t.hour < 21))]
        d_opens = [utc(2026, 9, 9, 21), utc(2026, 9, 10, 21)] + [
            utc(2026, 9, 13, 21) + timedelta(days=i) for i in range(5)]
        for t in market:
            close = t + timedelta(hours=1)
            h1 = pd.Series([m for m in market if m <= t] + list(
                pd.date_range(utc(2026, 9, 10, 12), utc(2026, 9, 11, 20), freq="1h")))
            h1 = h1.sort_values().reset_index(drop=True)
            h4_last = max(o for o in h4_opens if o + timedelta(hours=4) <= close)
            d_last = max(o for o in d_opens if o + timedelta(hours=24) <= close)
            self.assertIsNone(sync_problem(h1, h4_last, d_last), f"false alarm at {t}")


class VerifySyncedTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._orig_data_dir = config.DATA_DIR
        config.DATA_DIR = self._tmp

    def tearDown(self):
        config.DATA_DIR = self._orig_data_dir
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _seed(self, tf, times):
        storage.save_merged("EUR_USD", tf, [
            {"time": t.to_pydatetime(), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
            for t in times
        ])

    def test_passes_on_healthy_stored_data(self):
        self._seed("H1", hourly(utc(2026, 9, 15, 0), utc(2026, 9, 16, 15)))
        self._seed("H4", [utc(2026, 9, 16, 7)])
        self._seed("D", [utc(2026, 9, 15, 21)])
        _verify_synced("EUR_USD")  # must not raise

    def test_raises_with_the_pair_named_when_h4_is_stale(self):
        self._seed("H1", hourly(utc(2026, 9, 15, 0), utc(2026, 9, 16, 15)))
        self._seed("H4", [utc(2026, 9, 16, 3)])
        self._seed("D", [utc(2026, 9, 15, 21)])
        with self.assertRaises(DataSyncError) as ctx:
            _verify_synced("EUR_USD")
        self.assertIn("EUR_USD", str(ctx.exception))

    def test_raises_when_nothing_is_stored(self):
        with self.assertRaises(DataSyncError):
            _verify_synced("EUR_USD")


if __name__ == "__main__":
    unittest.main()
