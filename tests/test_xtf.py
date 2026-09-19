import unittest

import pandas as pd

from backtest.xtf import h1_index_at_or_after, h4_index_fully_closed_by


class XtfTests(unittest.TestCase):
    def test_h1_index_at_or_after_exact_match(self):
        times = pd.Series(pd.to_datetime([
            "2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z",
        ]))
        self.assertEqual(h1_index_at_or_after(times, pd.Timestamp("2024-01-01T01:00:00Z")), 1)

    def test_h1_index_at_or_after_between_candles(self):
        times = pd.Series(pd.to_datetime([
            "2024-01-01T00:00:00Z", "2024-01-01T02:00:00Z",
        ]))
        self.assertEqual(h1_index_at_or_after(times, pd.Timestamp("2024-01-01T01:00:00Z")), 1)

    def test_h1_index_at_or_after_past_end(self):
        times = pd.Series(pd.to_datetime(["2024-01-01T00:00:00Z"]))
        self.assertEqual(h1_index_at_or_after(times, pd.Timestamp("2024-01-02T00:00:00Z")), 1)

    def test_h4_fully_closed_requires_full_period_elapsed(self):
        times = pd.Series(pd.to_datetime([
            "2024-01-01T00:00:00Z", "2024-01-01T04:00:00Z", "2024-01-01T08:00:00Z",
        ]))
        # At 07:59, candle index 1 (opens 04:00) has NOT fully closed yet (closes 08:00).
        self.assertEqual(h4_index_fully_closed_by(times, pd.Timestamp("2024-01-01T07:59:00Z")), 0)
        # At exactly 08:00, candle index 1 has just fully closed.
        self.assertEqual(h4_index_fully_closed_by(times, pd.Timestamp("2024-01-01T08:00:00Z")), 1)

    def test_h4_fully_closed_none_yet(self):
        times = pd.Series(pd.to_datetime(["2024-01-01T00:00:00Z"]))
        self.assertEqual(h4_index_fully_closed_by(times, pd.Timestamp("2024-01-01T02:00:00Z")), -1)


if __name__ == "__main__":
    unittest.main()
