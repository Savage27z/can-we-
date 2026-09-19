import unittest

import pandas as pd

from backtest.bias import bias_asof


def daily_df(closes):
    n = len(closes)
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="24h", tz="UTC"),
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": [1] * n,
    })


class DailyBiasTests(unittest.TestCase):
    def test_bullish_when_two_consecutive_higher_closes(self):
        df = daily_df([1.00, 1.05, 1.10, 1.15])
        # Reference time mid-way through day index 3 (not yet closed) -> uses days 0,1,2.
        t = pd.Timestamp("2024-01-04 12:00", tz="UTC")
        self.assertEqual(bias_asof(df, t), "bullish")

    def test_bearish_when_two_consecutive_lower_closes(self):
        df = daily_df([1.15, 1.10, 1.05, 1.00])
        t = pd.Timestamp("2024-01-04 12:00", tz="UTC")
        self.assertEqual(bias_asof(df, t), "bearish")

    def test_neutral_when_not_monotonic(self):
        df = daily_df([1.00, 1.10, 1.05, 1.20])
        t = pd.Timestamp("2024-01-04 12:00", tz="UTC")
        self.assertEqual(bias_asof(df, t), "neutral")

    def test_neutral_when_insufficient_history(self):
        df = daily_df([1.00, 1.05])
        t = pd.Timestamp("2024-01-02 12:00", tz="UTC")
        self.assertEqual(bias_asof(df, t), "neutral")

    def test_does_not_use_still_forming_candle(self):
        # Mid-backtest scenario: daily_df already holds a future row (day index 4,
        # as a full historical dataset always would), but `t` sits only 1 hour
        # into day index 3's period. Day 3's true close is day 4's open time —
        # long after `t` — so day 3 must NOT be used yet even though its (lower,
        # look-ahead) close is already sitting right there in the given data.
        df = daily_df([1.00, 1.05, 1.10, 0.50, 1.20])
        t = pd.Timestamp("2024-01-04 01:00", tz="UTC")  # day index 3 has barely opened
        self.assertEqual(bias_asof(df, t), "bullish")

    def test_treats_the_newest_available_row_as_already_closed(self):
        # Live-edge scenario: daily_df ends exactly at the row being evaluated
        # (no future row exists yet, since storage only ever holds candles OANDA
        # reported complete). Unlike the mid-backtest case above, this newest
        # row IS already closed — bias must use it, not skip it.
        df = daily_df([1.00, 1.05, 1.10, 1.20])
        t = pd.Timestamp("2024-01-04 01:00", tz="UTC")
        self.assertEqual(bias_asof(df, t), "bullish")  # uses days 1,2,3 (1.05<1.10<1.20)

    def test_uses_candle_once_fully_closed(self):
        df = daily_df([1.00, 1.05, 1.10, 1.15])
        t = pd.Timestamp("2024-01-05 00:00", tz="UTC")  # day index 3 has now fully closed
        self.assertEqual(bias_asof(df, t), "bullish")


if __name__ == "__main__":
    unittest.main()
