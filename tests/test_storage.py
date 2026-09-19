"""Tests for data_pipeline.storage — merge/dedup logic against a temp directory,
no real data or credentials required.
"""
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from data_pipeline import config, storage


class StorageTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._orig_data_dir = config.DATA_DIR
        config.DATA_DIR = self._tmp

    def tearDown(self):
        config.DATA_DIR = self._orig_data_dir
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_load_missing_file_returns_empty_frame(self):
        df = storage.load("EUR_USD", "H1")
        self.assertTrue(df.empty)
        self.assertListEqual(list(df.columns), storage.COLUMNS)

    def test_last_timestamp_none_when_empty(self):
        self.assertIsNone(storage.last_timestamp("EUR_USD", "H1"))

    def test_save_and_reload_roundtrip(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = [
            {"time": base, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 10},
            {"time": base + timedelta(hours=1), "open": 1.05, "high": 1.2, "low": 1.0,
             "close": 1.1, "volume": 12},
        ]
        merged = storage.save_merged("EUR_USD", "H1", rows)
        self.assertEqual(len(merged), 2)

        reloaded = storage.load("EUR_USD", "H1")
        self.assertEqual(len(reloaded), 2)
        self.assertEqual(reloaded.iloc[0]["close"], 1.05)
        self.assertEqual(storage.last_timestamp("EUR_USD", "H1"), reloaded["time"].max())

    def test_merge_deduplicates_and_prefers_new_value(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        storage.save_merged("EUR_USD", "H1", [
            {"time": base, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 10},
        ])
        # Re-fetch of the same candle with a revised close (e.g. a re-run) should win.
        merged = storage.save_merged("EUR_USD", "H1", [
            {"time": base, "open": 1.0, "high": 1.15, "low": 0.9, "close": 1.09, "volume": 11},
        ])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.iloc[0]["close"], 1.09)

    def test_merge_keeps_sorted_order_regardless_of_insertion_order(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        storage.save_merged("EUR_USD", "H1", [
            {"time": base + timedelta(hours=2), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        ])
        merged = storage.save_merged("EUR_USD", "H1", [
            {"time": base, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
            {"time": base + timedelta(hours=1), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        ])
        self.assertListEqual(list(merged["time"]), sorted(merged["time"]))

    def test_separate_instruments_and_timeframes_do_not_collide(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        storage.save_merged("EUR_USD", "H1", [
            {"time": base, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        ])
        storage.save_merged("GBP_USD", "H1", [
            {"time": base, "open": 2, "high": 2, "low": 2, "close": 2, "volume": 2},
        ])
        storage.save_merged("EUR_USD", "D", [
            {"time": base, "open": 3, "high": 3, "low": 3, "close": 3, "volume": 3},
        ])
        self.assertEqual(storage.load("EUR_USD", "H1").iloc[0]["close"], 1)
        self.assertEqual(storage.load("GBP_USD", "H1").iloc[0]["close"], 2)
        self.assertEqual(storage.load("EUR_USD", "D").iloc[0]["close"], 3)

    def test_path_for_rejects_invalid_instrument(self):
        with self.assertRaises(ValueError):
            storage.path_for("../../secrets", "H1")
        with self.assertRaises(ValueError):
            storage.path_for("eur_usd", "H1")  # must be uppercase, OANDA format
        with self.assertRaises(ValueError):
            storage.path_for("EURUSD", "H1")  # missing underscore

    def test_path_for_rejects_invalid_timeframe(self):
        with self.assertRaises(ValueError):
            storage.path_for("EUR_USD", "../escape")

    def test_path_for_accepts_valid_pairs(self):
        # Must not raise.
        storage.path_for("EUR_USD", "H1")
        storage.path_for("USD_JPY", "D")

    def test_save_merged_accepts_preloaded_existing_to_skip_a_reload(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        preloaded = storage.load("EUR_USD", "H1")  # empty, file doesn't exist yet
        merged = storage.save_merged("EUR_USD", "H1", [
            {"time": base, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        ], existing=preloaded)
        self.assertEqual(len(merged), 1)
        self.assertEqual(storage.load("EUR_USD", "H1").iloc[0]["close"], 1)

    def test_save_writes_atomically_leaving_no_temp_file_behind(self):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        storage.save_merged("EUR_USD", "H1", [
            {"time": base, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        ])
        instrument_dir = self._tmp / "EUR_USD"
        leftover_tmp_files = list(instrument_dir.glob("*.tmp"))
        self.assertEqual(leftover_tmp_files, [])
        self.assertTrue((instrument_dir / "H1.parquet").exists())


if __name__ == "__main__":
    unittest.main()
