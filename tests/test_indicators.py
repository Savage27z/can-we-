import unittest

import numpy as np
import pandas as pd

from research import indicators


def series(values):
    return pd.Series(values, dtype=float)


class TrueRangeTests(unittest.TestCase):
    def test_it_takes_the_largest_of_the_range_and_the_two_gaps_from_the_previous_close(self):
        high, low, close = series([10, 12, 9]), series([8, 10, 7]), series([9, 11, 8])
        tr = indicators.true_range(high, low, close)
        # candle 1: range 2, |12-9|=3, |10-9|=1 -> 3; candle 2: range 2, |9-11|=2, |7-11|=4 -> 4
        self.assertEqual(list(tr), [2.0, 3.0, 4.0])

    def test_the_first_candle_has_only_its_own_range(self):
        self.assertEqual(indicators.true_range(series([5]), series([3]), series([4])).iloc[0], 2.0)


class AtrTests(unittest.TestCase):
    def test_atr_is_the_simple_mean_of_the_last_n_true_ranges_and_nan_before_that(self):
        high, low, close = series([10, 12, 9, 13]), series([8, 10, 7, 9]), series([9, 11, 8, 12])
        atr = indicators.atr(high, low, close, length=3)
        self.assertTrue(atr.iloc[:2].isna().all())
        self.assertAlmostEqual(atr.iloc[2], (2 + 3 + 4) / 3)
        # candle 3: range 4, |13-8|=5, |9-8|=1 -> 5
        self.assertAlmostEqual(atr.iloc[3], (3 + 4 + 5) / 3)

    def test_a_flat_series_with_a_constant_range_has_that_range_as_its_atr(self):
        n = 30
        atr = indicators.atr(series([1.10] * n), series([1.09] * n), series([1.095] * n), 14)
        self.assertAlmostEqual(atr.iloc[-1], 0.01)


class RollingTests(unittest.TestCase):
    def test_sma(self):
        out = indicators.sma(series([1, 2, 3, 4, 5]), 3)
        self.assertTrue(out.iloc[:2].isna().all())
        self.assertEqual(list(out.iloc[2:]), [2.0, 3.0, 4.0])

    def test_rolling_std_is_the_population_deviation(self):
        out = indicators.rolling_std(series([2, 4, 4, 4, 5, 5, 7, 9]), 8)
        self.assertAlmostEqual(out.iloc[-1], 2.0)           # the textbook example

    def test_prior_max_and_min_exclude_the_current_row(self):
        values = series([1, 5, 2, 3, 9])
        self.assertEqual(indicators.prior_max(values, 2).iloc[4], 3.0)     # max of rows 2,3, not 9
        self.assertEqual(indicators.prior_min(values, 2).iloc[4], 2.0)
        self.assertTrue(indicators.prior_max(values, 2).iloc[:2].isna().all())


class NoLookaheadTests(unittest.TestCase):
    """Appending later data must never change an indicator's earlier values."""

    def test_values_up_to_i_are_unchanged_by_data_after_i(self):
        rng = np.random.default_rng(0)
        close = series(1.10 + np.cumsum(rng.normal(0, 0.001, 120)))
        high, low = close + 0.002, close - 0.002
        full = {"atr": indicators.atr(high, low, close, 14), "sma": indicators.sma(close, 20),
                "std": indicators.rolling_std(close, 20), "pmax": indicators.prior_max(high, 20),
                "pmin": indicators.prior_min(low, 20)}
        cut = 70
        part = {"atr": indicators.atr(high[:cut], low[:cut], close[:cut], 14),
                "sma": indicators.sma(close[:cut], 20), "std": indicators.rolling_std(close[:cut], 20),
                "pmax": indicators.prior_max(high[:cut], 20), "pmin": indicators.prior_min(low[:cut], 20)}
        for name, values in full.items():
            pd.testing.assert_series_equal(values.iloc[:cut], part[name], check_names=False,
                                           obj=name)


if __name__ == "__main__":
    unittest.main()
