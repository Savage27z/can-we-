"""No strategy may use information from after the moment it trades.

For a strategy that only looks backwards, running it on the data as it stood at time T (only the
candles that had closed by T) must give exactly the signals it gave in the full-history run for
trades entered by T. If a signal appears, disappears or changes when later candles are added, the
strategy was using the future, and any backtest of it is worthless.
"""
import unittest

import numpy as np
import pandas as pd

from data_pipeline import quality
from research.strategies import STRATEGIES, get_strategy
from tests.market_helpers import synthetic_market, truncate

CUTOFF_POSITIONS = (0.55, 0.65, 0.75, 0.85, 0.95)


def fingerprint(signals, cutoff):
    return sorted((s.entry_time.value, s.direction, round(s.entry_price, 9), round(s.stop_price, 9),
                   round(s.target_price, 9)) for s in signals if s.entry_time <= cutoff)


class SyntheticMarketTests(unittest.TestCase):
    """The helper itself must look like real data, or the tests built on it prove nothing."""

    @classmethod
    def setUpClass(cls):
        cls.market = synthetic_market(weeks=70, seed=5)        # spans the March DST change

    def test_all_three_timeframes_pass_the_data_quality_checks(self):
        for tf, frame in self.market.items():
            row = quality.check_frame(frame, tf, "EUR_USD", h1=self.market["H1"])
            self.assertEqual(row["hard_errors"], 0, f"{tf}: {row['issues']}")
            self.assertEqual(row["missing_candles"], 0, tf)

    def test_h4_and_daily_candles_agree_with_the_h1_candles(self):
        self.assertEqual(quality.h1_vs_h4_mismatch(self.market["H1"], self.market["H4"])[1], 0)

    def test_the_daily_grid_moves_with_daylight_saving_time_as_in_real_data(self):
        hours = set(self.market["D"]["time"].dt.hour)
        self.assertEqual(hours, {21, 22})


class NoLookaheadTests(unittest.TestCase):
    WEEKS = 150

    def check(self, name, seed=3, **params):
        market = synthetic_market(weeks=self.WEEKS, seed=seed)
        strategy = get_strategy(name, **params)
        full = strategy.generate("EUR_USD", market).signals
        times = market["H1"]["time"]
        compared = 0
        for position in CUTOFF_POSITIONS:
            cutoff = times.iloc[int(position * len(times))] + pd.Timedelta(hours=1)
            cut = strategy.generate("EUR_USD", truncate(market, cutoff)).signals
            expected, got = fingerprint(full, cutoff), fingerprint(cut, cutoff)
            self.assertEqual(got, expected,
                             f"{name}: signals differ when the data is cut at {cutoff}")
            compared += len(expected)
        return compared

    def test_donchian_trend(self):
        self.assertGreater(self.check("donchian_trend"), 3)

    def test_daily_reversion(self):
        self.assertGreater(self.check("daily_reversion", seed=4), 0)

    def test_range_breakout(self):
        self.assertGreater(self.check("range_breakout"), 20)

    def test_random_control(self):
        self.assertGreater(self.check("random_control"), 20)

    def test_sweep_fvg(self):
        # The original strategy, held to the same standard.
        self.assertGreater(self.check("sweep_fvg", seed=6), 0)

    def test_sweep_fvg_no_rollover(self):
        self.assertGreater(self.check("sweep_fvg_no_rollover", seed=6), 0)

    def test_every_registered_strategy_has_a_lookahead_test(self):
        covered = {"donchian_trend", "daily_reversion", "range_breakout", "random_control",
                   "sweep_fvg", "sweep_fvg_no_rollover"}
        self.assertEqual(set(STRATEGIES), covered,
                         "a new strategy needs a no-look-ahead test in this file")


class TheCheckCanFailTests(unittest.TestCase):
    """A test that cannot fail proves nothing, so show it catches a strategy that peeks."""

    def test_a_strategy_that_uses_the_future_is_caught(self):
        from research.strategy import Signal, StrategyOutput

        class Peeker:
            name, timeframes = "peeker", ("D", "H1")

            def generate(self, instrument, frames):
                h1 = frames["H1"]
                # Buy at every candle that is the lowest of the WHOLE data: it needs the future.
                low_row = int(h1["close"].idxmin())
                price = float(h1["close"].iloc[low_row])
                return StrategyOutput([Signal(instrument, "bullish", low_row,
                                              h1["time"].iloc[low_row] + pd.Timedelta(hours=1),
                                              price, price - 0.01, price + 0.02)])

        market = synthetic_market(weeks=60, seed=2)
        peeker = Peeker()
        full = peeker.generate("EUR_USD", market).signals
        row = full[0].entry_index
        # A moment BEFORE the candle it bought: as of then it could not have known it was the low.
        cutoff = market["H1"]["time"].iloc[row] - pd.Timedelta(hours=200)
        cut = peeker.generate("EUR_USD", truncate(market, cutoff)).signals
        self.assertNotEqual(fingerprint(cut, cutoff), fingerprint(full, cutoff))


if __name__ == "__main__":
    unittest.main()
