"""Tests for live.state — verifies the live engine surfaces a genuinely-detected
(via find_sweep_events, not a hand-picked index) pending setup with the right
structured fields, and that resolved/dead setups are excluded.
"""
import unittest

import pandas as pd

from backtest import rules
from backtest.fractals import build_levels, find_swings
from backtest.sessions import in_session_mask
from backtest.setup import MarketData
from live.state import compute_state_from_market


def build_eur_usd_market_with_pending_setup() -> MarketData:
    # Same H4 construction as tests/test_worked_examples.py's §8.1 regression test
    # (a real, fractal-detected swing low at 1.0800, swept at index 11, bullish FVG
    # forming at mid=13) — reused here because it's already proven to produce a
    # real sweep event via find_sweep_events, not just a hand-picked index.
    start = "2024-01-01 13:00"
    highs = [1.0850] * 12
    lows = [1.0840] * 12
    closes = [1.0845] * 12
    highs[1], highs[2], highs[3], highs[4], highs[5] = 1.0860, 1.0870, 1.0900, 1.0880, 1.0870
    lows[1], lows[2], lows[3], lows[4], lows[5] = 1.0850, 1.0860, 1.0890, 1.0870, 1.0860
    highs[6], highs[7], highs[8] = 1.0830, 1.0845, 1.0835
    lows[6], lows[7], lows[8] = 1.0800, 1.0815, 1.0805
    highs[9], lows[9] = 1.0845, 1.0820
    highs[10], lows[10] = 1.0850, 1.0810
    highs[11], lows[11], closes[11] = 1.0840, 1.0795, 1.0805

    highs += [1.0812, 1.0850, 1.0870]
    lows += [1.0803, 1.0806, 1.0820]
    closes += [1.0808, 1.0845, 1.0860]

    n = len(highs)
    h4 = pd.DataFrame({
        "time": pd.date_range(start, periods=n, freq="4h", tz="UTC"),
        "open": [(h + l) / 2 for h, l in zip(highs, lows)],
        "high": highs, "low": lows, "close": closes, "volume": [100] * n,
    })
    h4 = find_swings(h4).reset_index(drop=True)
    levels = build_levels(h4)

    # Only 2 H1 candles exist after the FVG forms (mid=13, formed at h4.time[14]+4h)
    # — nowhere near the 20-candle confirmation window, and neither confirms nor
    # invalidates, so this must surface as "pending_confirmation", not resolved.
    fvg_formed_time = h4["time"].iloc[14] + pd.Timedelta(hours=4)
    h1 = pd.DataFrame({
        "time": pd.date_range(fvg_formed_time, periods=2, freq="1h", tz="UTC"),
        "open": [1.0808, 1.0809], "high": [1.0810, 1.0811], "low": [1.0805, 1.0806],
        "close": [1.0808, 1.0809], "volume": [1, 1],
    })

    daily = pd.DataFrame({
        "time": pd.date_range("2023-12-25", periods=5, freq="24h", tz="UTC"),
        "open": [1.05, 1.06, 1.07, 1.08, 1.09], "high": [1.05, 1.06, 1.07, 1.08, 1.09],
        "low": [1.05, 1.06, 1.07, 1.08, 1.09], "close": [1.05, 1.06, 1.07, 1.08, 1.09],
        "volume": [1] * 5,
    })

    return MarketData(
        pair="EUR_USD", h4=h4, h1=h1, daily=daily, levels=levels,
        h4_session=in_session_mask(h4["time"]), h1_session=in_session_mask(h1["time"]),
    )


class LiveStateTests(unittest.TestCase):
    def setUp(self):
        self.market = build_eur_usd_market_with_pending_setup()

    def test_surfaces_the_pending_setup_with_correct_fields(self):
        state = compute_state_from_market(self.market)
        self.assertEqual(state.pair, "EUR_USD")
        # as_of is the latest H1 candle's CLOSE (open + 1h) — that candle is
        # stored complete, so its close is the true "known as of" instant.
        expected_as_of = self.market.h1["time"].iloc[-1] + pd.Timedelta(hours=1)
        self.assertEqual(state.as_of, expected_as_of.isoformat())
        self.assertEqual(state.daily_bias, "bullish")

        self.assertAlmostEqual(state.current_price, 1.0809)  # latest H1 close
        # The 1.0900 swing high from the §8.1-style H4 data is unmitigated and
        # above current price -> must appear on the buy side of the liquidity map.
        self.assertIn(1.0900, state.liquidity_buy_side)

        self.assertEqual(len(state.active_setups), 1)
        setup = state.active_setups[0]
        self.assertEqual(setup.direction, "bullish")
        self.assertEqual(setup.status, "pending_confirmation")
        self.assertAlmostEqual(setup.sweep_extreme, 1.0795)
        self.assertAlmostEqual(setup.confirmation_level, 1.0812)
        # Not yet confirmed -> no entry/stop/target/rr yet.
        self.assertIsNone(setup.entry_price)
        self.assertIsNone(setup.stop_price)
        self.assertIsNone(setup.target_price)
        self.assertIsNone(setup.rr)

    def test_every_setup_carries_a_plan_built_from_the_engines_own_numbers(self):
        state = compute_state_from_market(self.market)
        plan = state.active_setups[0].plan
        self.assertEqual(plan.side, "BUY")
        self.assertAlmostEqual(plan.stop, rules.stop_price("EUR_USD", "bullish", 1.0795))
        self.assertAlmostEqual(plan.invalidation, 1.0795)
        self.assertAlmostEqual(plan.entry, 1.0812)           # the trigger level
        self.assertTrue(plan.target_provisional)
        # The 1.0900 swing high is the nearest unmitigated one above the trigger.
        self.assertAlmostEqual(plan.target, 1.0900)
        self.assertAlmostEqual(plan.rr, (1.0900 - 1.0812) / (1.0812 - 1.0790))
        self.assertIs(type(plan.stop), float)                # plain floats, not numpy scalars

    def test_the_forecast_target_is_the_one_the_engine_picks_once_it_confirms(self):
        planned = compute_state_from_market(self.market).active_setups[0].plan

        confirming = pd.DataFrame({
            "time": [pd.Timestamp("2024-01-04 08:00", tz="UTC")],   # an in-session H1 candle
            "open": [1.0812], "high": [1.0818], "low": [1.0810], "close": [1.0815], "volume": [1],
        })
        self.market.h1 = pd.concat([self.market.h1, confirming], ignore_index=True)
        self.market.h1_session = in_session_mask(self.market.h1["time"])

        live = compute_state_from_market(self.market).active_setups[0]
        self.assertEqual(live.status, "live_trade")
        self.assertAlmostEqual(live.target_price, planned.target)
        self.assertAlmostEqual(live.plan.stop, planned.stop)
        self.assertFalse(live.plan.target_provisional)
        self.assertAlmostEqual(live.plan.entry, 1.0815)      # the confirming close, not the trigger

    def test_a_rejected_sweep_appears_as_a_recent_rejection(self):
        from tests.test_worked_examples import Section82NoFVGTest

        case = Section82NoFVGTest()
        case.setUp()
        state = compute_state_from_market(case.market)
        self.assertEqual(state.active_setups, [])
        self.assertTrue(state.recent_rejections)
        rejection = state.recent_rejections[0]
        self.assertEqual(rejection.outcome, "no_fvg")
        self.assertEqual(rejection.direction, "bearish")

    def test_no_recent_rejections_when_the_only_sweep_is_still_pending(self):
        state = compute_state_from_market(build_eur_usd_market_with_pending_setup())
        self.assertEqual(state.recent_rejections, [])

    def test_to_dict_is_json_serializable(self):
        import json
        state = compute_state_from_market(self.market)
        json.dumps(state.to_dict())  # must not raise


if __name__ == "__main__":
    unittest.main()
