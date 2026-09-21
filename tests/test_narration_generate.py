import json
import unittest
from unittest.mock import patch

from narration.generate import narrate_state
from live.state import LiveState, ActiveSetup
from news.filter import NewsEventView, NewsStatus


class NarrateStateTests(unittest.TestCase):
    def test_passes_full_state_json_to_the_model(self):
        state = LiveState(
            pair="EUR_USD", as_of="2026-01-01T00:00:00+00:00", current_price=1.05,
            daily_bias="bullish",
            active_setups=[ActiveSetup(
                direction="bullish", status="pending_confirmation",
                sweep_time="2026-01-01T09:00:00+00:00", sweep_extreme=1.0400,
                fvg_low=1.0410, fvg_high=1.0420, confirmation_level=1.0410,
                entry_price=None, stop_price=None, target_price=None, rr=None,
                h1_candles_to_confirm=None, confirm_time=None,
            )],
            liquidity_buy_side=[1.0600, 1.0650], liquidity_sell_side=[1.0300],
            news=NewsStatus(
                status="blackout",
                blackout_events=[NewsEventView(
                    title="FOMC Statement", currency="USD",
                    time_utc="2026-01-01T19:00:00+00:00", when="in 25m",
                )],
            ),
        )

        captured = {}

        def fake_chat(system_prompt, user_prompt, **kwargs):
            captured["system_prompt"] = system_prompt
            captured["user_prompt"] = user_prompt
            return "REPORT"

        with patch("narration.generate.deepseek_client.chat", side_effect=fake_chat):
            result = narrate_state(state)

        self.assertEqual(result, "REPORT")
        self.assertIn("EUR_USD", captured["user_prompt"])
        # The user prompt must carry the ACTUAL numbers, not a summary that could
        # drop precision or drift from what the engine computed.
        parsed = json.loads(captured["user_prompt"].split("\n\n", 1)[1])
        self.assertEqual(parsed["daily_bias"], "bullish")
        self.assertEqual(parsed["liquidity_buy_side"], [1.0600, 1.0650])
        self.assertEqual(parsed["active_setups"][0]["confirmation_level"], 1.0410)
        # News reaches the model with its precomputed "when" text, so the model
        # never has to do time arithmetic itself.
        self.assertEqual(parsed["news"]["status"], "blackout")
        self.assertEqual(parsed["news"]["blackout_events"][0]["when"], "in 25m")
        self.assertIn("You do not decide bias", captured["system_prompt"])
        self.assertIn("news.blackout_events", captured["system_prompt"])

    def test_the_trade_plan_is_kept_from_the_model(self):
        from live.plan import plan_for

        setup = ActiveSetup(
            direction="bullish", status="live_trade", sweep_time="2026-01-01T09:00:00+00:00",
            sweep_extreme=1.0400, fvg_low=1.0410, fvg_high=1.0420, confirmation_level=1.0410,
            entry_price=1.0430, stop_price=1.0395, target_price=1.0500, rr=2.0,
            h1_candles_to_confirm=1, confirm_time="2026-01-01T12:00:00+00:00",
        )
        setup.plan = plan_for("EUR_USD", setup)
        state = LiveState(pair="EUR_USD", as_of="2026-01-01T13:00:00+00:00", current_price=1.044,
                          daily_bias="bullish", active_setups=[setup], liquidity_buy_side=[],
                          liquidity_sell_side=[], news=NewsStatus(status="clear"))
        captured = {}

        def fake_chat(system_prompt, user_prompt, **kwargs):
            captured["user_prompt"] = user_prompt
            return "REPORT"

        with patch("narration.generate.deepseek_client.chat", side_effect=fake_chat):
            narrate_state(state)

        sent = json.loads(captured["user_prompt"].split("\n\n", 1)[1])
        self.assertNotIn("plan", sent["active_setups"][0])
        self.assertIsNotNone(setup.plan)   # stripping the model's copy must not touch the state

    def facts_sent_for(self, state):
        captured = {}

        def fake_chat(system_prompt, user_prompt, **kwargs):
            captured["user_prompt"] = user_prompt
            return "REPORT"

        with patch("narration.generate.deepseek_client.chat", side_effect=fake_chat):
            narrate_state(state)
        return json.loads(captured["user_prompt"].split("\n\n", 1)[1])

    def confirmed_state(self):
        setup = ActiveSetup(
            direction="bullish", status="live_trade", sweep_time="2026-06-10T21:00:00+00:00",
            sweep_extreme=1.1526, fvg_low=1.1541, fvg_high=1.1566, confirmation_level=1.1541,
            entry_price=1.1567, stop_price=1.1521, target_price=1.1645, rr=1.7,
            h1_candles_to_confirm=3, confirm_time="2026-06-12T08:00:00+00:00",
            fvg_start_time="2026-06-11T13:00:00+00:00")
        return LiveState(pair="EUR_USD", as_of="2026-06-12T08:00:00+00:00", current_price=1.1567,
                         daily_bias="neutral", active_setups=[setup], liquidity_buy_side=[],
                         liquidity_sell_side=[], news=NewsStatus(status="clear"))

    def test_times_reach_the_model_as_display_text_not_iso_strings(self):
        # Given an ISO string the model copies it into the report ("...on 2026-06-10T21:00:00+00:00").
        sent = self.facts_sent_for(self.confirmed_state())
        self.assertEqual(sent["as_of"], "Fri 12 Jun 08:00 UTC")
        setup = sent["active_setups"][0]
        self.assertEqual(setup["sweep_time"], "Wed 10 Jun 21:00 UTC")
        self.assertEqual(setup["confirm_time"], "Fri 12 Jun 08:00 UTC")

    def test_a_time_that_does_not_exist_yet_stays_null(self):
        state = self.confirmed_state()
        state.active_setups[0].confirm_time = None
        self.assertIsNone(self.facts_sent_for(state)["active_setups"][0]["confirm_time"])

    def test_numbers_and_the_diagnostic_time_are_untouched(self):
        setup = self.facts_sent_for(self.confirmed_state())["active_setups"][0]
        self.assertEqual(setup["entry_price"], 1.1567)
        self.assertEqual(setup["fvg_start_time"], "2026-06-11T13:00:00+00:00")   # never shown

    def test_formatting_the_model_copy_leaves_the_state_alone(self):
        state = self.confirmed_state()
        self.facts_sent_for(state)
        self.assertEqual(state.as_of, "2026-06-12T08:00:00+00:00")
        self.assertEqual(state.active_setups[0].sweep_time, "2026-06-10T21:00:00+00:00")
        self.assertEqual(state.active_setups[0].confirm_time, "2026-06-12T08:00:00+00:00")

    def test_the_prompt_tells_the_model_to_copy_the_times_exactly(self):
        from narration.prompt import SYSTEM_PROMPT
        self.assertIn("already formatted for display", SYSTEM_PROMPT)
        self.assertIn("never convert, shorten, reformat", SYSTEM_PROMPT)

    def test_the_prompt_no_longer_asks_for_stop_target_or_rr_lines(self):
        from narration.prompt import SYSTEM_PROMPT
        for old in (" • Invalidation:", " • Target:", " • Planned R:R:"):
            self.assertNotIn(old, SYSTEM_PROMPT)
        self.assertIn("trade-plan block", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
