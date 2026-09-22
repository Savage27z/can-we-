"""Regressions for the htf_period/ltf_period generalisation (evaluate_setup, select_target,
xtf.h4_index_fully_closed_by): the default must reproduce the original H4=4h/H1=1h behaviour
exactly, and a different period must be honoured, not silently ignored."""
import unittest

import pandas as pd

from backtest.setup import evaluate_setup
from backtest.xtf import h4_index_fully_closed_by
from tests.test_signal_causality import worked_market


class DefaultPeriodUnchangedTests(unittest.TestCase):
    """The §8.1 worked example, run with no htf_period/ltf_period argument at all, must give
    exactly the same result as passing the 4h/1h defaults explicitly."""

    def test_explicit_default_matches_implicit_default(self):
        market = worked_market()
        implicit = evaluate_setup(market, "bullish", 11)
        explicit = evaluate_setup(market, "bullish", 11, htf_period=pd.Timedelta(hours=4),
                                  ltf_period=pd.Timedelta(hours=1))
        self.assertEqual(implicit.outcome, explicit.outcome)
        self.assertEqual(implicit.confirm_time, explicit.confirm_time)
        self.assertEqual(implicit.resolve_time, explicit.resolve_time)
        self.assertEqual(implicit.target_price, explicit.target_price)
        self.assertEqual(implicit.realized_r, explicit.realized_r)


class DifferentPeriodIsHonouredTests(unittest.TestCase):
    def test_confirm_time_uses_the_given_ltf_period_not_one_hour(self):
        market = worked_market()
        result = evaluate_setup(market, "bullish", 11, ltf_period=pd.Timedelta(minutes=1))
        # The confirming H1 row (row 6, per the §8.1 fixture) opens at fvg_formed_time+6h;
        # its close under a 1-minute ltf_period is 1 minute later, not 1 hour later.
        confirming_open = market.h1["time"].iloc[6]
        self.assertEqual(result.confirm_time, confirming_open + pd.Timedelta(minutes=1))

    def test_a_wrong_period_would_have_mistimed_it(self):
        market = worked_market()
        result = evaluate_setup(market, "bullish", 11, ltf_period=pd.Timedelta(minutes=1))
        confirming_open = market.h1["time"].iloc[6]
        self.assertNotEqual(result.confirm_time, confirming_open + pd.Timedelta(hours=1))


class XtfPeriodTests(unittest.TestCase):
    def test_default_period_is_four_hours(self):
        times = pd.Series(pd.to_datetime([
            "2024-01-01T00:00:00Z", "2024-01-01T04:00:00Z", "2024-01-01T08:00:00Z"]))
        self.assertEqual(h4_index_fully_closed_by(times, pd.Timestamp("2024-01-01T08:00:00Z")), 1)

    def test_a_five_minute_period_closes_five_minutes_after_open(self):
        times = pd.Series(pd.to_datetime([
            "2024-01-01T00:00:00Z", "2024-01-01T00:05:00Z", "2024-01-01T00:10:00Z"]))
        period = pd.Timedelta(minutes=5)
        self.assertEqual(h4_index_fully_closed_by(times, pd.Timestamp("2024-01-01T00:09:59Z"),
                                                  period=period), 0)
        self.assertEqual(h4_index_fully_closed_by(times, pd.Timestamp("2024-01-01T00:10:00Z"),
                                                  period=period), 1)


if __name__ == "__main__":
    unittest.main()
