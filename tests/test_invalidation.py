import unittest

from backtest.invalidation import is_invalidated


class InvalidationTests(unittest.TestCase):
    def test_bullish_invalidated_when_close_below_extreme(self):
        self.assertTrue(is_invalidated("bullish", sweep_extreme=1.0800, close_price=1.0799))

    def test_bullish_not_invalidated_at_or_above_extreme(self):
        self.assertFalse(is_invalidated("bullish", sweep_extreme=1.0800, close_price=1.0800))
        self.assertFalse(is_invalidated("bullish", sweep_extreme=1.0800, close_price=1.0900))

    def test_bearish_invalidated_when_close_above_extreme(self):
        self.assertTrue(is_invalidated("bearish", sweep_extreme=1.2500, close_price=1.2501))

    def test_bearish_not_invalidated_at_or_below_extreme(self):
        self.assertFalse(is_invalidated("bearish", sweep_extreme=1.2500, close_price=1.2500))

    def test_invalid_direction_raises(self):
        with self.assertRaises(ValueError):
            is_invalidated("sideways", 1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
