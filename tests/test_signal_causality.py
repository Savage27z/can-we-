"""Regressions for audit findings F01-F03: what the engine may know at a given moment, and
how a live trade ends. F02 and F03 are live-layer opt-ins; every test that pins the default
also proves the backtest keeps the rules as originally scored."""
import unittest

import numpy as np
import pandas as pd

from backtest.bias import bias_asof
from backtest.resolution import resolve_on_plan
from backtest.setup import evaluate_setup, select_target
from live.state import compute_state_from_market
from research.exits import Candles, PlanExit
from research.strategy import Signal
from tests import test_worked_examples as worked


def utc(text):
    return pd.Timestamp(text, tz="UTC")


def daily_rows(opens, closes):
    return pd.DataFrame({"time": [utc(o) for o in opens], "open": closes, "high": closes,
                         "low": closes, "close": closes, "volume": 1})


def worked_market(edit=None):
    """The §8.1 bullish setup (entry 1.0815, stop 1.0790, target 1.0900, confirms at H1 row 6)."""
    case = worked.Section81BullishSignalTest()
    case.setUp()
    market = case.market
    if edit is not None:
        h1 = market.h1.copy()
        edit(h1)
        market.h1 = h1
    return market


class F01DailyBiasCausalityTests(unittest.TestCase):
    def test_the_newest_daily_row_cannot_supply_a_close_that_had_not_happened(self):
        df = daily_rows(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
                        [1.20, 1.10, 1.00, 1.30])
        # Jan 4's 1.30 closes on Jan 5; at Jan 4 noon only Jan 1-3 are known: falling closes.
        self.assertEqual(bias_asof(df, utc("2024-01-04 12:00")), "bearish")

    def test_a_row_that_has_closed_is_used_even_when_it_is_the_newest(self):
        df = daily_rows(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
                        [1.20, 1.10, 1.00, 1.30])
        self.assertEqual(bias_asof(df, utc("2024-01-05 00:00")), "neutral")   # 1.30 now counts

    def test_appending_history_cannot_change_an_earlier_moments_bias(self):
        closes = [1.00, 1.10, 1.20, 1.30, 0.50]
        opens = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        for query in ("2024-01-04 06:00", "2024-01-04 23:59"):
            shorter = bias_asof(daily_rows(opens[:4], closes[:4]), utc(query))
            longer = bias_asof(daily_rows(opens, closes), utc(query))
            self.assertEqual(shorter, longer, query)

    def test_the_weekend_does_not_make_the_answer_depend_on_the_next_row(self):
        # July 2024, New York summer: candles open 21:00 UTC. The one opening Thursday
        # closes Friday 17:00 New York, and the next opens Sunday.
        opens = ["2024-06-30 21:00", "2024-07-01 21:00", "2024-07-02 21:00",
                 "2024-07-03 21:00", "2024-07-04 21:00"]
        closes = [1.0, 1.1, 1.2, 1.3, 0.5]
        with_next = daily_rows(opens + ["2024-07-07 21:00"], closes + [0.4])
        without_next = daily_rows(opens, closes)
        for query in ("2024-07-05 12:00", "2024-07-06 12:00", "2024-07-07 20:00"):
            self.assertEqual(bias_asof(without_next, utc(query)), bias_asof(with_next, utc(query)),
                             query)


class F02PlanExitTests(unittest.TestCase):
    def test_the_default_still_scores_on_closes_only(self):
        def wick_through_the_stop(h1):
            h1.loc[8, "low"] = 1.0785        # touches the 1.0790 stop, closes back at 1.0835

        result = evaluate_setup(worked_market(wick_through_the_stop), "bullish", 11)
        self.assertEqual(result.outcome, "win")   # the original close-based scoring, unchanged

    def test_a_stop_touch_ends_the_trade_the_plan_told_you_to_place(self):
        def wick_through_the_stop(h1):
            h1.loc[8, "low"] = 1.0785

        result = evaluate_setup(worked_market(wick_through_the_stop), "bullish", 11,
                                plan_exits=True)
        self.assertEqual(result.outcome, "loss")
        self.assertAlmostEqual(result.realized_r, (1.0790 - 1.0815) / 0.0025, places=6)

    def test_a_target_touch_ends_the_trade_even_if_the_candle_closes_below_it(self):
        def touch_without_closing(h1):
            h1.loc[7:, "close"] = 1.0850
            h1.loc[7:, "open"] = 1.0850
            h1.loc[7:, "high"] = 1.0855
            h1.loc[7:, "low"] = 1.0845
            h1.loc[9, "high"] = 1.0910           # touches the 1.0900 target

        market = worked_market(touch_without_closing)
        self.assertEqual(evaluate_setup(market, "bullish", 11).outcome, "open")
        result = evaluate_setup(market, "bullish", 11, plan_exits=True)
        self.assertEqual(result.outcome, "win")
        self.assertAlmostEqual(result.realized_r, 3.4, places=6)

    def test_live_state_no_longer_shows_a_trade_whose_plan_has_ended(self):
        def wick_through_the_stop(h1):
            h1.loc[8, "low"] = 1.0785

        live = [s for s in compute_state_from_market(worked_market(wick_through_the_stop)).active_setups
                if s.status == "live_trade"]
        self.assertEqual(live, [])
        still_live = [s for s in compute_state_from_market(worked_market()).active_setups]
        self.assertTrue(all(s.status != "live_trade" for s in still_live))   # this one won

    def test_it_agrees_with_the_research_plan_exit(self):
        rng = np.random.default_rng(7)
        for trial in range(300):
            n = 40
            close = 1.10 + np.cumsum(rng.normal(0, 0.0010, n))
            open_ = np.concatenate([[1.10], close[:-1]]) + rng.normal(0, 0.0004, n)
            high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.0004, n))
            low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.0004, n))
            frame = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
                                  "open": open_, "high": high, "low": low, "close": close})
            for direction in ("bullish", "bearish"):
                entry = float(close[0])
                sign = 1 if direction == "bullish" else -1
                signal = Signal("EUR_USD", direction, 0, frame["time"][0], entry,
                                entry - sign * 0.0030, entry + sign * 0.0060,
                                invalidation_price=entry - sign * 0.0020)
                theirs = PlanExit().resolve(signal, Candles.from_frame(frame))
                ours = resolve_on_plan(direction, open_, high, low, close, 1,
                                       signal.invalidation_price, signal.stop_price,
                                       signal.target_price)
                if theirs.index is None:
                    self.assertIsNone(ours, (trial, direction))
                else:
                    self.assertEqual(ours, (theirs.index, theirs.reason, theirs.price),
                                     (trial, direction))


class F03TargetMitigationTests(unittest.TestCase):
    def _market_where_the_target_was_crossed_before_entry(self):
        def cross(h1):
            h1.loc[5, "high"] = 1.0910   # 06:00 candle: inside the still-forming H4, closes at 1.0811

        return worked_market(cross)

    def test_a_target_already_crossed_inside_the_forming_h4_is_not_eligible(self):
        market = self._market_where_the_target_was_crossed_before_entry()
        confirm = evaluate_setup(market, "bullish", 11).confirm_time
        with_h1 = evaluate_setup(market, "bullish", 11, h1_mitigation=True)
        self.assertEqual(with_h1.outcome, "no_target")
        self.assertEqual(with_h1.confirm_time, confirm)

    def test_the_default_keeps_the_original_rule_and_result(self):
        market = self._market_where_the_target_was_crossed_before_entry()
        self.assertEqual(evaluate_setup(market, "bullish", 11).outcome, "win")

    def test_a_cross_after_the_confirming_candle_does_not_count_at_entry(self):
        def later_cross(h1):
            h1.loc[9, "high"] = 1.0910   # after the confirming candle (row 6)

        market = worked_market(later_cross)
        result = evaluate_setup(market, "bullish", 11, h1_mitigation=True)
        self.assertNotEqual(result.outcome, "no_target")
        self.assertAlmostEqual(result.target_price, 1.0900)

    def test_the_confirming_candle_itself_counts(self):
        def cross_on_the_confirming_candle(h1):
            h1.loc[6, "high"] = 1.0910

        market = worked_market(cross_on_the_confirming_candle)
        self.assertEqual(evaluate_setup(market, "bullish", 11, h1_mitigation=True).outcome,
                         "no_target")

    def test_select_target_is_unchanged_without_the_opt_in(self):
        market = self._market_where_the_target_was_crossed_before_entry()
        level = select_target(market, "bullish", 11, 1.0815, 13)
        self.assertIsNotNone(level)


if __name__ == "__main__":
    unittest.main()
