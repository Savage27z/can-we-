import unittest

import numpy as np

from backtest.fvg import detect_fvg


class FVGTests(unittest.TestCase):
    def test_bullish_fvg(self):
        high = np.array([1.0, 1.05, 1.20])
        low = np.array([0.9, 1.0, 1.10])
        fvg = detect_fvg(high, low, mid_index=1)
        self.assertIsNotNone(fvg)
        self.assertEqual(fvg.direction, "bullish")
        self.assertEqual(fvg.low, 1.0)   # high[0]
        self.assertEqual(fvg.high, 1.10)  # low[2]
        self.assertEqual(fvg.confirmation_level, 1.0)  # §1.8: near/bottom edge

    def test_bearish_fvg(self):
        high = np.array([1.20, 1.05, 1.0])
        low = np.array([1.10, 0.95, 0.80])
        fvg = detect_fvg(high, low, mid_index=1)
        self.assertIsNotNone(fvg)
        self.assertEqual(fvg.direction, "bearish")
        self.assertEqual(fvg.low, 1.0)    # high[i+1] = high[2], the numeric lower bound
        self.assertEqual(fvg.high, 1.10)  # low[i-1] = low[0], the numeric upper bound
        self.assertEqual(fvg.confirmation_level, 1.10)  # §1.8: near/top edge = high[i+1]

    def test_no_gap_returns_none(self):
        high = np.array([1.0, 1.05, 1.06])
        low = np.array([0.9, 0.95, 0.96])
        self.assertIsNone(detect_fvg(high, low, mid_index=1))

    def test_out_of_bounds_returns_none(self):
        high = np.array([1.0, 1.05])
        low = np.array([0.9, 1.0])
        self.assertIsNone(detect_fvg(high, low, mid_index=0))
        self.assertIsNone(detect_fvg(high, low, mid_index=1))


if __name__ == "__main__":
    unittest.main()
