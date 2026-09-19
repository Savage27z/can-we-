"""Tests for live.refresh's cross-timeframe sync check (audit finding: D/H4/H1
are fetched as independent OANDA calls and can desync if one fails)."""
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from data_pipeline import config, storage
from live.refresh import DataSyncError, _verify_synced


class VerifySyncedTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._orig_data_dir = config.DATA_DIR
        config.DATA_DIR = self._tmp

    def tearDown(self):
        config.DATA_DIR = self._orig_data_dir
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _seed(self, pair, tf, t):
        storage.save_merged(pair, tf, [
            {"time": t, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        ])

    def test_passes_when_all_three_timeframes_are_current(self):
        now = datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc)  # a Monday
        self._seed("EUR_USD", "H1", now)
        self._seed("EUR_USD", "H4", now - timedelta(hours=3))
        self._seed("EUR_USD", "D", now - timedelta(hours=12))
        _verify_synced("EUR_USD")  # must not raise

    def test_passes_across_a_normal_weekend_gap(self):
        # Market has JUST reopened Sunday evening: H1 has its first post-weekend
        # candle, but H4's last candle is still Friday's pre-close one (its own
        # first post-weekend candle hasn't closed yet). ~49h gap, must not
        # false-positive as a sync failure.
        sunday_reopen = datetime(2024, 6, 9, 22, 0, tzinfo=timezone.utc)
        friday_close = datetime(2024, 6, 7, 21, 0, tzinfo=timezone.utc)
        self._seed("EUR_USD", "H1", sunday_reopen)
        self._seed("EUR_USD", "H4", friday_close)
        self._seed("EUR_USD", "D", friday_close)
        _verify_synced("EUR_USD")  # must not raise

    def test_raises_when_h4_fetch_lagged_while_h1_succeeded(self):
        # A Wednesday, not a weekend — a multi-day gap here can't be explained
        # by market closure, only by a stuck/failed H4 fetch.
        now = datetime(2024, 6, 12, 15, 0, tzinfo=timezone.utc)
        self._seed("EUR_USD", "H1", now)
        self._seed("EUR_USD", "H4", now - timedelta(hours=70))
        self._seed("EUR_USD", "D", now - timedelta(hours=12))
        with self.assertRaises(DataSyncError):
            _verify_synced("EUR_USD")

    def test_raises_when_daily_is_stale(self):
        now = datetime(2024, 6, 10, 15, 0, tzinfo=timezone.utc)
        self._seed("EUR_USD", "H1", now)
        self._seed("EUR_USD", "H4", now - timedelta(hours=1))
        self._seed("EUR_USD", "D", now - timedelta(days=10))
        with self.assertRaises(DataSyncError):
            _verify_synced("EUR_USD")

    def test_raises_when_h1_missing_entirely(self):
        with self.assertRaises(DataSyncError):
            _verify_synced("EUR_USD")


if __name__ == "__main__":
    unittest.main()
