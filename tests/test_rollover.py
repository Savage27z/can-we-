"""The rollover rule (strategy_rules.md §5, v1.7): the live bot does not trade a confirmation on the
H1 candle opening at 21:00 UTC. It is the variant that was backtested (research's SweepFvgNoRollover),
so the tests pin two things: the rule changes nothing except those confirmations, and it yields
exactly the trades the research measured."""
import unittest
from unittest.mock import patch

import pandas as pd

from backtest import rules
from backtest.engine import build_market_data, find_sweep_events
from backtest.setup import OUTCOMES_NO_TRADE, OUTCOMES_TRADED, evaluate_setup
from live.plan import _session_hours
from live.state import compute_state_from_market
from research.strategies.sweep_fvg import SweepFvg, SweepFvgNoRollover
from tests.market_helpers import synthetic_market, truncate


def market_of(frames):
    return build_market_data("EUR_USD", frames["D"], frames["H4"], frames["H1"])


def trade_key(r):
    return (r.direction, r.confirm_time, round(r.entry_price, 6), round(r.stop_price, 6),
            round(r.target_price, 6))


def signal_key(s):
    return (s.direction, s.entry_time, round(s.entry_price, 6), round(s.stop_price, 6),
            round(s.target_price, 6))


class RolloverRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = synthetic_market(weeks=160, seed=0)
        cls.market = market_of(cls.frames)
        cls.events = find_sweep_events(cls.market)
        cls.plain = [evaluate_setup(cls.market, d, i) for i, d in cls.events]
        cls.fixed = [evaluate_setup(cls.market, d, i, no_entry_hours=rules.NO_ENTRY_HOURS_UTC)
                     for i, d in cls.events]

    def confirm_hour(self, result):
        return self.market.h1["time"].iloc[result.confirm_index].hour

    def test_the_constant_is_the_hour_the_research_measured(self):
        self.assertEqual(rules.NO_ENTRY_HOURS_UTC, (21,))
        self.assertEqual(SweepFvgNoRollover().skip_open_hours, rules.NO_ENTRY_HOURS_UTC)

    def test_by_default_nothing_is_skipped_so_the_original_backtest_is_unchanged(self):
        self.assertNotIn("skipped_rollover", {r.outcome for r in self.plain})
        self.assertEqual({trade_key(r) for r in self.plain if r.outcome in OUTCOMES_TRADED},
                         {signal_key(s) for s in SweepFvg().generate("EUR_USD", self.frames).signals})

    def test_the_fixture_actually_exercises_the_rule(self):
        skipped = [r for r in self.fixed if r.outcome == "skipped_rollover"]
        self.assertGreater(len(skipped), 5)
        self.assertGreater(len([r for r in self.fixed if r.outcome in OUTCOMES_TRADED]), 5)

    def test_it_changes_exactly_the_confirmations_on_the_rollover_candle(self):
        for before, after in zip(self.plain, self.fixed):
            on_rollover = (before.confirm_index is not None and
                           self.confirm_hour(before) in rules.NO_ENTRY_HOURS_UTC)
            if on_rollover:
                self.assertEqual(after.outcome, "skipped_rollover")
                self.assertEqual(after.confirm_index, before.confirm_index)
                self.assertEqual(after.confirm_time, before.confirm_time)
                self.assertIsNone(after.target_price)              # it never went on to a plan
            else:
                self.assertEqual(after, before)                    # every other setup is identical

    def test_a_skipped_setup_is_dropped_not_left_waiting_for_a_later_candle(self):
        # This is the researched variant. Letting it wait would be a different, untested rule.
        for before, after in zip(self.plain, self.fixed):
            if after.outcome == "skipped_rollover":
                self.assertIsNotNone(before.confirm_index)         # the same candle confirmed it
                self.assertEqual(after.confirm_index, before.confirm_index)

    def test_it_yields_exactly_the_trades_of_the_researched_variant(self):
        trades = {trade_key(r) for r in self.fixed if r.outcome in OUTCOMES_TRADED}
        researched = SweepFvgNoRollover().generate("EUR_USD", self.frames)
        self.assertEqual(trades, {signal_key(s) for s in researched.signals})
        self.assertGreater(len(trades), 0)

    def test_the_fixed_trades_are_a_subset_of_the_original_trades(self):
        original = {trade_key(r) for r in self.plain if r.outcome in OUTCOMES_TRADED}
        fixed = {trade_key(r) for r in self.fixed if r.outcome in OUTCOMES_TRADED}
        self.assertLess(fixed, original)

    def test_skipping_is_a_no_trade_outcome(self):
        self.assertIn("skipped_rollover", OUTCOMES_NO_TRADE)

    def test_another_hour_only_skips_confirmations_on_that_hour(self):
        results = [evaluate_setup(self.market, d, i, no_entry_hours=(9,)) for i, d in self.events]
        for r in results:
            if r.outcome == "skipped_rollover":
                self.assertEqual(self.confirm_hour(r), 9)


class LiveStateAppliesTheRuleTests(unittest.TestCase):
    """The setup that the backtest default would show as a live trade must not appear as one."""

    @classmethod
    def setUpClass(cls):
        cls.frames = synthetic_market(weeks=160, seed=0)
        cls.market = market_of(cls.frames)

    def find_case(self):
        for index, direction in find_sweep_events(self.market):
            plain = evaluate_setup(self.market, direction, index)
            fixed = evaluate_setup(self.market, direction, index, no_entry_hours=(21,))
            if fixed.outcome == "skipped_rollover" and plain.outcome in OUTCOMES_TRADED:
                return plain
        self.fail("no setup that confirms on the rollover candle in the fixture")

    def test_a_confirmation_on_the_rollover_candle_is_not_a_live_trade(self):
        case = self.find_case()
        cutoff = case.confirm_time                       # the moment the candle closes
        cut = market_of(truncate(self.frames, cutoff))
        default_view = [evaluate_setup(cut, d, i) for i, d in find_sweep_events(cut)]
        self.assertTrue(any(r.outcome == "open" and r.sweep_time == case.sweep_time
                            for r in default_view),
                        "without the rule this setup is a live trade at that moment")

        state = compute_state_from_market(cut)
        self.assertNotIn(case.sweep_time.isoformat(),
                         [s.sweep_time for s in state.active_setups if s.status == "live_trade"])

    def test_the_backtest_engine_is_not_told_to_skip(self):
        # The live layer opts in; the shared engine does not, so research baselines are unchanged.
        with patch("live.state.evaluate_setup", wraps=evaluate_setup) as spy:
            compute_state_from_market(market_of(truncate(
                self.frames, pd.Timestamp("2022-06-01", tz="UTC"))))
        self.assertTrue(spy.called)
        for call in spy.call_args_list:
            self.assertEqual(call.kwargs.get("no_entry_hours"), rules.NO_ENTRY_HOURS_UTC)


class PlanTextHoursTests(unittest.TestCase):
    def test_the_last_candle_that_can_confirm_opens_at_2000(self):
        self.assertEqual(_session_hours(), "07:00–20:00 UTC")

    def test_without_the_rule_the_original_window_is_shown(self):
        with patch.object(rules, "NO_ENTRY_HOURS_UTC", ()):
            self.assertEqual(_session_hours(), "07:00–21:00 UTC")

    def test_a_hole_in_the_middle_is_shown_as_two_ranges(self):
        with patch.object(rules, "NO_ENTRY_HOURS_UTC", (12,)):
            self.assertEqual(_session_hours(), "07:00–11:00, 13:00–21:00 UTC")


if __name__ == "__main__":
    unittest.main()
