import unittest

from backtest.rules import pip_size, stop_buffer


class RulesTests(unittest.TestCase):
    def test_pip_size_standard_pair(self):
        self.assertAlmostEqual(pip_size("EUR_USD"), 0.0001)

    def test_pip_size_jpy_pair(self):
        self.assertAlmostEqual(pip_size("USD_JPY"), 0.01)

    def test_stop_buffer_standard_pair_is_5_pips(self):
        self.assertAlmostEqual(stop_buffer("GBP_USD"), 0.0005)

    def test_stop_buffer_jpy_pair_is_5_pips(self):
        self.assertAlmostEqual(stop_buffer("EUR_JPY"), 0.05)


if __name__ == "__main__":
    unittest.main()
