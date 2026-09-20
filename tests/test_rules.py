import unittest

from backtest.rules import pip_size, stop_buffer, stop_price


class RulesTests(unittest.TestCase):
    def test_pip_size_standard_pair(self):
        self.assertAlmostEqual(pip_size("EUR_USD"), 0.0001)

    def test_pip_size_jpy_pair(self):
        self.assertAlmostEqual(pip_size("USD_JPY"), 0.01)

    def test_stop_buffer_standard_pair_is_5_pips(self):
        self.assertAlmostEqual(stop_buffer("GBP_USD"), 0.0005)

    def test_stop_buffer_jpy_pair_is_5_pips(self):
        self.assertAlmostEqual(stop_buffer("EUR_JPY"), 0.05)


class StopPriceTests(unittest.TestCase):
    def test_bullish_stop_sits_a_buffer_below_the_sweep_low(self):
        self.assertAlmostEqual(stop_price("EUR_USD", "bullish", 1.1455), 1.1450)

    def test_bearish_stop_sits_a_buffer_above_the_sweep_high(self):
        self.assertAlmostEqual(stop_price("EUR_USD", "bearish", 1.1500), 1.1505)

    def test_jpy_buffer_is_five_hundredths(self):
        self.assertAlmostEqual(stop_price("USD_JPY", "bullish", 157.20), 157.15)
        self.assertAlmostEqual(stop_price("USD_JPY", "bearish", 157.20), 157.25)


if __name__ == "__main__":
    unittest.main()
