import unittest

from backtest import rules
from live.plan import MAX_PLANS_SHOWN, plan_for, plan_text, worst_entry
from live.state import ActiveSetup, LiveState
from news.filter import NewsStatus


def make_setup(status="live_trade", direction="bullish", pair="EUR_USD",
               sweep_time="2026-09-21T09:00:00+00:00", sweep_extreme=1.14550,
               confirmation_level=1.14600, entry=1.14650, stop=1.14500, target=1.15000,
               rr=2.333):
    live = status == "live_trade"
    return ActiveSetup(
        direction=direction, status=status, sweep_time=sweep_time, sweep_extreme=sweep_extreme,
        fvg_low=None if status == "pending_fvg" else 1.1460,
        fvg_high=None if status == "pending_fvg" else 1.1470,
        confirmation_level=None if status == "pending_fvg" else confirmation_level,
        entry_price=entry if live else None, stop_price=stop if live else None,
        target_price=target if live else None, rr=rr if live else None,
        h1_candles_to_confirm=None,
        confirm_time="2026-09-21T14:00:00+00:00" if live else None,
    )


def with_plan(setup, pair="EUR_USD", provisional=None):
    setup.plan = plan_for(pair, setup, provisional)
    return setup


def state_of(*setups, pair="EUR_USD"):
    return LiveState(pair=pair, as_of="2026-09-21T15:00:00+00:00", current_price=1.1470,
                     daily_bias="bullish", active_setups=list(setups),
                     liquidity_buy_side=[], liquidity_sell_side=[],
                     news=NewsStatus(status="clear"))


class WorstEntryTests(unittest.TestCase):
    def test_rr_at_the_worst_entry_is_exactly_the_minimum_for_a_buy(self):
        stop, target = 1.1450, 1.1500
        e = worst_entry(stop, target)
        self.assertAlmostEqual((target - e) / (e - stop), rules.MIN_RR)

    def test_rr_at_the_worst_entry_is_exactly_the_minimum_for_a_sell(self):
        stop, target = 1.1505, 1.1440
        e = worst_entry(stop, target)
        self.assertAlmostEqual((e - target) / (stop - e), rules.MIN_RR)


class LivePlanTests(unittest.TestCase):
    def test_buy_numbers(self):
        plan = plan_for("EUR_USD", make_setup())
        self.assertEqual(plan.side, "BUY")
        self.assertAlmostEqual(plan.stop_pips, 15.0)
        self.assertAlmostEqual(plan.target_pips, 35.0)
        self.assertAlmostEqual(plan.worst_entry, 1.14700)
        self.assertAlmostEqual(plan.rr, 2.333)
        self.assertFalse(plan.target_provisional)

    def test_buy_text_says_where_to_enter_stop_and_take_profit(self):
        s = with_plan(make_setup())
        text = plan_text(state_of(s))
        for expected in (
            "TRADE PLAN — BUY EUR/USD · LIVE",
            "closed above 1.14600 at Mon 21 Sep 14:00 UTC",
            "Entry: BUY at market. Signal price 1.14650.",
            "at or below 1.14700",
            "Stop-loss: 1.14500 (15.0 pips below entry)",
            "Take-profit: 1.15000 (35.0 pips above entry)",
            "R:R: 1 : 2.33",
            "CLOSES below 1.14550 (sweep low)",
        ):
            self.assertIn(expected, text)

    def test_sell_is_the_mirror_image(self):
        s = with_plan(make_setup(direction="bearish", sweep_extreme=1.15000,
                                 confirmation_level=1.14950, entry=1.14900,
                                 stop=1.15050, target=1.14400, rr=3.333))
        text = plan_text(state_of(s))
        for expected in (
            "SELL EUR/USD · LIVE",
            "closed below 1.14950",
            "at or above 1.14790",
            "Stop-loss: 1.15050 (15.0 pips above entry)",
            "Take-profit: 1.14400 (50.0 pips below entry)",
            "CLOSES above 1.15000 (sweep high)",
        ):
            self.assertIn(expected, text)

    def test_jpy_uses_three_decimals_and_hundredth_pips(self):
        s = with_plan(make_setup(pair="USD_JPY", sweep_extreme=157.200, confirmation_level=157.400,
                                 entry=157.500, stop=157.150, target=158.500, rr=2.857),
                      pair="USD_JPY")
        text = plan_text(state_of(s, pair="USD_JPY"))
        self.assertIn("Stop-loss: 157.150 (35.0 pips below entry)", text)
        self.assertIn("Take-profit: 158.500 (100.0 pips above entry)", text)

    def test_the_plan_reports_the_engines_own_rr_not_a_recomputation(self):
        self.assertEqual(plan_for("EUR_USD", make_setup(rr=2.5)).rr, 2.5)


class PendingConfirmationPlanTests(unittest.TestCase):
    def pending(self, provisional):
        return with_plan(make_setup("pending_confirmation"), provisional=provisional)

    def test_the_stop_comes_from_the_sweep_and_the_target_is_marked_provisional(self):
        plan = self.pending(1.15000).plan
        self.assertAlmostEqual(plan.stop, 1.14500)        # 1.14550 - 5 pips
        self.assertAlmostEqual(plan.entry, 1.14600)       # the trigger level
        self.assertTrue(plan.target_provisional)
        self.assertAlmostEqual(plan.rr, 4.0)              # (1.1500-1.1460)/(1.1460-1.1450)

    def test_text_explains_the_trigger_the_entry_and_the_cancel_level(self):
        text = plan_text(state_of(self.pending(1.15000)))
        for expected in (
            "BUY EUR/USD · NOT ACTIVE YET",
            "wait for an H1 candle to CLOSE above 1.14600",
            "07:00–20:00 UTC",
            "If the candle that confirms it opens at 21:00 UTC",
            "the setup is skipped.",
            "within 20 H1 candles",
            "BUY at market once it closes",
            "Skip it if the close is above 1.14700",
            "Stop-loss: 1.14500 (sweep low 1.14550 − 5 pips)",
            "Take-profit: 1.15000 — provisional",
            "R:R at the trigger level: 1 : 4.00",
            "Cancelled if an H1 candle CLOSES below 1.14550 (sweep low) first.",
        ):
            self.assertIn(expected, text)

    def test_a_plan_below_the_minimum_rr_warns_that_it_will_be_skipped(self):
        text = plan_text(state_of(self.pending(1.14700)))   # reward 10 pips vs risk 10 pips
        self.assertIn("R:R at the trigger level: 1 : 1.00", text)
        self.assertIn("expect this to be skipped", text)

    def test_no_target_is_stated_plainly_not_invented(self):
        s = self.pending(None)
        self.assertIsNone(s.plan.target)
        self.assertIsNone(s.plan.rr)
        self.assertIsNone(s.plan.worst_entry)
        text = plan_text(state_of(s))
        self.assertIn("no unmitigated target available", text)
        self.assertNotIn("Skip it if", text)

    def test_a_bearish_pending_plan_mirrors_it(self):
        s = with_plan(make_setup("pending_confirmation", direction="bearish", sweep_extreme=1.15000,
                                 confirmation_level=1.14900), provisional=1.14400)
        text = plan_text(state_of(s))
        self.assertIn("SELL EUR/USD · NOT ACTIVE YET", text)
        self.assertIn("CLOSE below 1.14900", text)
        self.assertIn("Skip it if the close is below 1.14790", text)
        self.assertIn("Stop-loss: 1.15050 (sweep high 1.15000 + 5 pips)", text)
        self.assertIn("Cancelled if an H1 candle CLOSES above 1.15000", text)


class PendingFvgPlanTests(unittest.TestCase):
    def test_no_entry_or_target_is_offered_before_a_gap_exists(self):
        s = with_plan(make_setup("pending_fvg"))
        self.assertAlmostEqual(s.plan.stop, 1.14500)
        for value in (s.plan.entry, s.plan.target, s.plan.rr, s.plan.worst_entry,
                      s.plan.stop_pips, s.plan.target_pips):
            self.assertIsNone(value)
        text = plan_text(state_of(s))
        self.assertIn("NOTHING TO ENTER YET", text)
        self.assertIn("bullish H4 fair value gap", text)
        self.assertIn("within 10 H4 candles", text)
        self.assertIn("the stop-loss goes at 1.14500 (sweep low 1.14550 − 5 pips)", text)
        self.assertIn("Cancelled if an H1 candle CLOSES below 1.14550", text)
        self.assertNotIn("Entry:", text)
        self.assertNotIn("Take-profit", text)


class PlanTextTests(unittest.TestCase):
    def test_no_setup_says_there_is_nothing_to_enter(self):
        text = plan_text(state_of())
        self.assertIn("none", text)
        self.assertIn("nothing to enter", text)

    def test_a_live_trade_leads_even_when_it_is_the_oldest_setup(self):
        waiting = with_plan(make_setup("pending_fvg", sweep_time="2026-09-21T09:00:00+00:00"))
        live = with_plan(make_setup("live_trade", sweep_time="2026-09-10T09:00:00+00:00"))
        text = plan_text(state_of(waiting, live))
        self.assertLess(text.index("· LIVE"), text.index("NOTHING TO ENTER YET"))

    def test_within_a_stage_the_newest_sweep_comes_first(self):
        older = with_plan(make_setup("pending_fvg", sweep_time="2026-09-18T09:00:00+00:00",
                                     sweep_extreme=1.13000))
        newer = with_plan(make_setup("pending_fvg", sweep_time="2026-09-21T09:00:00+00:00",
                                     sweep_extreme=1.14550))
        text = plan_text(state_of(older, newer))
        self.assertLess(text.index("1.14550"), text.index("1.13000"))

    def test_only_a_few_plans_are_shown(self):
        setups = [with_plan(make_setup("pending_fvg", sweep_time=f"2026-09-1{d}T09:00:00+00:00"))
                  for d in range(1, 6)]
        text = plan_text(state_of(*setups))
        self.assertEqual(text.count("🎯 TRADE PLAN"), MAX_PLANS_SHOWN)

    def test_a_state_without_plans_adds_nothing_instead_of_claiming_there_is_no_setup(self):
        s = make_setup()                       # built by hand, so no plan attached
        self.assertEqual(plan_text(state_of(s)), "")

    def test_building_a_plan_does_not_modify_the_setup(self):
        s = make_setup("pending_confirmation")
        before = dict(vars(s))
        plan_for("EUR_USD", s, 1.15)
        self.assertEqual(dict(vars(s)), before)


if __name__ == "__main__":
    unittest.main()
