"""Tests for the pending_fvg / pending_confirmation outcomes — the states Phase 3's
live engine depends on to distinguish "still waiting, more data could arrive" from
a definitively terminal outcome. These only show up when the available data runs
out mid-window, which happens constantly in live use (we're always at the "live
edge") but only ever at the tail of a historical backtest.
"""
import unittest

import pandas as pd

from backtest.fractals import build_levels, find_swings
from backtest.sessions import in_session_mask
from backtest.setup import MarketData, evaluate_setup


def h4_series(start, highs, lows, closes):
    n = len(highs)
    return pd.DataFrame({
        "time": pd.date_range(start, periods=n, freq="4h", tz="UTC"),
        "open": [(h + l) / 2 for h, l in zip(highs, lows)],
        "high": highs, "low": lows, "close": closes, "volume": [100] * n,
    })


def daily_series(start, closes):
    n = len(closes)
    return pd.DataFrame({
        "time": pd.date_range(start, periods=n, freq="24h", tz="UTC"),
        "open": closes, "high": closes, "low": closes, "close": closes, "volume": [1] * n,
    })


def build_market(pair, h4, h1, daily) -> MarketData:
    h4 = find_swings(h4).reset_index(drop=True)
    levels = build_levels(h4)
    return MarketData(
        pair=pair, h4=h4, h1=h1.reset_index(drop=True), daily=daily.reset_index(drop=True),
        levels=levels, h4_session=in_session_mask(h4["time"]), h1_session=in_session_mask(h1["time"]),
    )


class PendingFVGTest(unittest.TestCase):
    def test_data_runs_out_before_fvg_window_closes(self):
        start = "2024-01-01 13:00"  # index 6 -> 09:00 UTC (London), matches sweep needs
        highs = [1.0850, 1.0860, 1.0870, 1.0900, 1.0880, 1.0870, 1.0830]
        lows = [1.0840, 1.0850, 1.0860, 1.0890, 1.0870, 1.0860, 1.0800]
        closes = [1.0845, 1.0855, 1.0865, 1.0895, 1.0875, 1.0865, 1.0805]
        # index6 is the sweep candle (low 1.0800 pierces a swing low at ~1.0800-ish
        # formed earlier isn't needed for this test — we call evaluate_setup
        # directly with a fixed sweep_index, bypassing find_sweep_events).
        # Only ONE more candle exists after the sweep, and it doesn't form a gap.
        highs.append(1.0835)
        lows.append(1.0810)
        closes.append(1.0820)

        h4 = h4_series(start, highs, lows, closes)
        h1 = pd.DataFrame({
            "time": pd.date_range(h4["time"].iloc[-1], periods=1, freq="1h", tz="UTC"),
            "open": [1.0820], "high": [1.0825], "low": [1.0815], "close": [1.0820], "volume": [1],
        })
        daily = daily_series("2023-12-25", [1.05, 1.06, 1.07])

        market = build_market("EUR_USD", h4, h1, daily)
        result = evaluate_setup(market, "bullish", sweep_index=6)
        self.assertEqual(result.outcome, "pending_fvg")
        self.assertIsNone(result.fvg)


class PendingConfirmationTest(unittest.TestCase):
    def test_h1_data_runs_out_before_confirmation_window_closes(self):
        start = "2024-01-01 13:00"
        # index6 = sweep candle (evaluate_setup trusts the caller's sweep_index and
        # just reads low[6] as the extreme — it doesn't re-verify a real fractal
        # swing existed there, that's find_sweep_events' job, not evaluate_setup's).
        # FVG forms at mid=8 using candles 7/8/9: low[9]=1.0820 > high[7]=1.0812.
        highs = [1.0850, 1.0860, 1.0870, 1.0900, 1.0880, 1.0870, 1.0830, 1.0812, 1.0850, 1.0870]
        lows = [1.0840, 1.0850, 1.0860, 1.0890, 1.0870, 1.0860, 1.0800, 1.0803, 1.0806, 1.0820]
        closes = [1.0845, 1.0855, 1.0865, 1.0895, 1.0875, 1.0865, 1.0805, 1.0808, 1.0845, 1.0860]
        h4 = h4_series(start, highs, lows, closes)

        # Only 2 H1 candles exist after FVG formation — nowhere near the 20-candle
        # confirmation window — and neither confirms nor invalidates.
        fvg_formed_time = h4["time"].iloc[8] + pd.Timedelta(hours=4)
        h1 = pd.DataFrame({
            "time": pd.date_range(fvg_formed_time, periods=2, freq="1h", tz="UTC"),
            "open": [1.0808, 1.0809], "high": [1.0810, 1.0811], "low": [1.0805, 1.0806],
            "close": [1.0808, 1.0809], "volume": [1, 1],
        })
        daily = daily_series("2023-12-25", [1.05, 1.06, 1.07])

        market = build_market("EUR_USD", h4, h1, daily)
        result = evaluate_setup(market, "bullish", sweep_index=6)
        # Sanity: this must actually reach the confirmation stage (fvg found, bias ok)
        # for the test to be exercising what it claims to.
        self.assertIsNotNone(result.fvg, msg="test setup didn't actually form an FVG")
        self.assertEqual(result.outcome, "pending_confirmation")


if __name__ == "__main__":
    unittest.main()
