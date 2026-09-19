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


if __name__ == "__main__":
    unittest.main()
