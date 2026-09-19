import unittest

import pandas as pd

from backtest.fractals import build_levels, find_swings


def make_df(highs, lows):
    n = len(highs)
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"),
        "open": highs,  # not used by swing/mitigation logic
        "high": highs,
        "low": lows,
        "close": highs,
        "volume": [1] * n,
    })


class SwingDetectionTests(unittest.TestCase):
    def test_simple_swing_high(self):
        highs = [1, 2, 3, 2, 1, 1, 1]
        lows = [0.5, 1, 1.5, 1, 0.5, 0.5, 0.5]
        df = find_swings(make_df(highs, lows))
        self.assertTrue(df["swing_high"].iloc[2])
        self.assertFalse(df["swing_high"].iloc[1])
        self.assertFalse(df["swing_high"].iloc[3])

    def test_simple_swing_low(self):
        highs = [3, 2, 1, 2, 3, 3, 3]
        lows = [2, 1, 0, 1, 2, 2, 2]
        df = find_swings(make_df(highs, lows))
        self.assertTrue(df["swing_low"].iloc[2])

    def test_requires_strict_extremity_on_both_sides(self):
        # A tie (not strictly greater) on either side must NOT count as a swing.
        highs = [1, 2, 2, 2, 1]
        lows = [0, 1, 1, 1, 0]
        df = find_swings(make_df(highs, lows))
        self.assertFalse(df["swing_high"].iloc[2])

    def test_edges_never_flagged(self):
        highs = [5, 1, 1, 1, 5]
        lows = [4, 0.5, 0.5, 0.5, 4]
        df = find_swings(make_df(highs, lows))
        self.assertFalse(df["swing_high"].iloc[0])
        self.assertFalse(df["swing_high"].iloc[-1])


class MitigationTests(unittest.TestCase):
    def test_swing_high_mitigated_when_later_high_exceeds_it(self):
        highs = [1, 2, 3, 2, 1, 4, 1]
        lows = [x - 0.5 for x in highs]
        df = find_swings(make_df(highs, lows))
        levels = build_levels(df)
        swing_high = next(lvl for lvl in levels if lvl.kind == "high")
        self.assertEqual(swing_high.index, 2)
        self.assertEqual(swing_high.confirmed_at, 4)
        self.assertEqual(swing_high.mitigated_at, 5)  # high[5] = 4 > 3

    def test_swing_high_never_mitigated_if_never_exceeded(self):
        highs = [1, 2, 3, 2, 1, 1, 1]
        lows = [x - 0.5 for x in highs]
        df = find_swings(make_df(highs, lows))
        levels = build_levels(df)
        swing_high = next(lvl for lvl in levels if lvl.kind == "high")
        self.assertIsNone(swing_high.mitigated_at)

    def test_unmitigated_as_of_boundary(self):
        highs = [1, 2, 3, 2, 1, 4, 1]
        lows = [x - 0.5 for x in highs]
        df = find_swings(make_df(highs, lows))
        levels = build_levels(df)
        lvl = next(l for l in levels if l.kind == "high")
        self.assertTrue(lvl.unmitigated_as_of(5))   # mitigation AT 5 itself is fine
        self.assertFalse(lvl.unmitigated_as_of(6))  # strictly after mitigation: dead

    def test_in_scope_rejects_levels_formed_after_reference_index(self):
        highs = [1, 2, 3, 2, 1, 1, 1]
        lows = [x - 0.5 for x in highs]
        df = find_swings(make_df(highs, lows))
        levels = build_levels(df)
        lvl = next(l for l in levels if l.kind == "high")  # formed at index 2
        self.assertFalse(lvl.in_scope_of(sweep_index=1, lookback=90))  # sweep before formation
        self.assertFalse(lvl.in_scope_of(sweep_index=2, lookback=90))  # sweep AT formation
        self.assertTrue(lvl.in_scope_of(sweep_index=3, lookback=90))


if __name__ == "__main__":
    unittest.main()
