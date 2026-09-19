"""Regression tests built directly from strategy_rules.md §8's worked examples.
These exercise the full evaluate_setup pipeline end-to-end (fractals -> FVG ->
daily bias -> H1 confirmation/invalidation -> target -> R:R -> outcome), so if a
future edit to any module quietly breaks the doc's own reference scenarios, these
fail loudly.
"""
import unittest

import pandas as pd

from backtest.fractals import build_levels, find_swings
from backtest.sessions import in_session_mask
from backtest.setup import MarketData, evaluate_setup


def h4_series(start: str, hours, highs, lows, closes):
    n = len(highs)
    times = pd.date_range(start, periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({
        "time": times,
        "open": [(h + l) / 2 for h, l in zip(highs, lows)],
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [100] * n,
    })


def daily_series(start: str, closes):
    n = len(closes)
    return pd.DataFrame({
        "time": pd.date_range(start, periods=n, freq="24h", tz="UTC"),
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": [1] * n,
    })


def build_market(pair, h4, h1, daily) -> MarketData:
    h4 = find_swings(h4).reset_index(drop=True)
    levels = build_levels(h4)
    return MarketData(
        pair=pair, h4=h4, h1=h1.reset_index(drop=True), daily=daily.reset_index(drop=True),
        levels=levels,
        h4_session=in_session_mask(h4["time"]),
        h1_session=in_session_mask(h1["time"]),
    )


class Section81BullishSignalTest(unittest.TestCase):
    """§8.1: sweep of a swing low at 1.0800, bullish FVG forms, confirms above
    1.0812, target at a pre-existing swing high of 1.0900 -> a real signal, and
    it resolves as a win when price subsequently closes >= 1.0900.
    """

    def setUp(self):
        # Start hour chosen (13:00) so that index 11 (the sweep candle, per the
        # doc's table) lands on 09:00 UTC as in the worked example; see the H4
        # index/hour table in the doc for indices 10-14.
        start = "2024-01-01 13:00"

        highs = [1.0850] * 12
        lows = [1.0840] * 12
        closes = [1.0845] * 12

        # Pre-existing target swing high = 1.0900 at index 3 (needs 2 candles
        # each side with a lower high).
        highs[1], highs[2], highs[3], highs[4], highs[5] = 1.0860, 1.0870, 1.0900, 1.0880, 1.0870
        lows[1], lows[2], lows[3], lows[4], lows[5] = 1.0850, 1.0860, 1.0890, 1.0870, 1.0860

        # Swept swing low = 1.0800 at index 6 (needs 2 candles each side with a
        # higher low); confirmed_at = 8, well before the sweep at index 11.
        highs[4], highs[5], highs[6], highs[7], highs[8] = 1.0880, 1.0870, 1.0830, 1.0845, 1.0835
        lows[4], lows[5], lows[6], lows[7], lows[8] = 1.0870, 1.0860, 1.0800, 1.0815, 1.0805

        # idx 9, 10: lead-in candles, nothing special (must not exceed the swing
        # low's neighbors in a way that creates a competing fractal).
        highs[9], lows[9] = 1.0845, 1.0820
        highs[10], lows[10] = 1.0850, 1.0810  # doc's "idx 10" row

        # idx 11: the sweep candle (doc: high 1.0840, low 1.0795, close 1.0805).
        highs[11], lows[11], closes[11] = 1.0840, 1.0795, 1.0805

        # idx 12, 13, 14: FVG candles (doc: 1.0812/1.0803/1.0808, then
        # 1.0850/1.0806/1.0845, then the gap-closing candle).
        extra_highs = [1.0812, 1.0850, 1.0870, 1.0880, 1.0885]
        extra_lows = [1.0803, 1.0806, 1.0820, 1.0860, 1.0865]
        extra_closes = [1.0808, 1.0845, 1.0860, 1.0870, 1.0875]

        highs += extra_highs
        lows += extra_lows
        closes += extra_closes

        h4 = h4_series(start, None, highs, lows, closes)

        # H1 series covering from just after the FVG forms (idx 14 close = idx15
        # open, since idx14 is now index 14 in the extended array... the FVG mid
        # index is 13, formed-time reference is h4.time[14] + 4h) through the
        # confirmation and win.
        fvg_formed_time = h4["time"].iloc[14] + pd.Timedelta(hours=4)
        h1_times = pd.date_range(fvg_formed_time, periods=16, freq="1h", tz="UTC")
        # Hours 0-5 after formed-time are off-session or below the confirmation
        # level (1.0812); hour 6 lands at 07:00 UTC (London open) and confirms.
        h1_closes = [1.0808, 1.0809, 1.0810, 1.0810, 1.0811, 1.0811,
                     1.0815,  # confirmation candle
                     1.0820, 1.0835, 1.0850, 1.0865, 1.0880, 1.0895, 1.0905, 1.0910, 1.0915]
        h1 = pd.DataFrame({
            "time": h1_times,
            "open": h1_closes,
            "high": [c + 0.0002 for c in h1_closes],
            "low": [c - 0.0002 for c in h1_closes],
            "close": h1_closes,
            "volume": [10] * len(h1_closes),
        })

        daily = daily_series("2023-12-25", [1.05, 1.06, 1.07, 1.08, 1.09])

        self.market = build_market("EUR_USD", h4, h1, daily)

    def test_sweep_candle_is_a_valid_bullish_sweep(self):
        h4 = self.market.h4
        self.assertEqual(h4["low"].iloc[11], 1.0795)
        self.assertLess(h4["low"].iloc[11], 1.0800)   # pierces the swept level
        self.assertGreater(h4["close"].iloc[11], 1.0800)  # closes back above it
        self.assertTrue(self.market.h4_session.iloc[11])   # 09:00 UTC -> London

    def test_full_pipeline_produces_a_win(self):
        result = evaluate_setup(self.market, "bullish", sweep_index=11)
        self.assertEqual(result.outcome, "win", msg=result)
        self.assertAlmostEqual(result.sweep_extreme, 1.0795)
        self.assertIsNotNone(result.fvg)
        self.assertAlmostEqual(result.fvg.confirmation_level, 1.0812)
        self.assertAlmostEqual(result.entry_price, 1.0815)
        self.assertAlmostEqual(result.stop_price, 1.0790)  # 1.0795 - 5 pips
        self.assertAlmostEqual(result.target_price, 1.0900)
        self.assertAlmostEqual(result.r_price, 0.0025, places=6)
        self.assertAlmostEqual(result.reward_price, 0.0085, places=6)
        self.assertAlmostEqual(result.rr, 3.4, places=2)
        self.assertAlmostEqual(result.realized_r, 3.4, places=2)


class Section82NoFVGTest(unittest.TestCase):
    """§8.2: a valid bearish sweep occurs, but no 3-candle sequence in the
    following window forms a bearish FVG -> outcome must be 'no_fvg', no signal.
    """

    def setUp(self):
        start = "2024-01-01 13:00"
        n = 20
        highs = [1.2450] * n
        lows = [1.2430] * n
        closes = [1.2440] * n

        # Swing high = 1.2500 at index 3.
        highs[1], highs[2], highs[3], highs[4], highs[5] = 1.2460, 1.2470, 1.2500, 1.2480, 1.2470
        lows[1], lows[2], lows[3], lows[4], lows[5] = 1.2450, 1.2460, 1.2490, 1.2470, 1.2460

        # idx 8: bearish sweep candle (pierces 1.2500, closes back below).
        highs[8], lows[8], closes[8] = 1.2510, 1.2480, 1.2495

        # idx 9 through the end of the array (covers the full 10-candle FVG
        # window, whose last candidate mid=18 reads candles 17/18/19 — all of
        # them must stay flat, not just 9-18, or a gap forms at the edge):
        # keep highs/lows heavily overlapping candle-to-candle so no 3-candle
        # sequence ever gaps.
        for k in range(9, n):
            highs[k] = 1.2500
            lows[k] = 1.2470
            closes[k] = 1.2485

        h4 = h4_series(start, None, highs, lows, closes)
        h1 = pd.DataFrame({
            "time": pd.date_range(h4["time"].iloc[-1], periods=5, freq="1h", tz="UTC"),
            "open": [1.2485] * 5, "high": [1.2490] * 5, "low": [1.2480] * 5,
            "close": [1.2485] * 5, "volume": [1] * 5,
        })
        daily = daily_series("2023-12-25", [1.25, 1.24, 1.23])  # bearish bias, irrelevant here

        self.market = build_market("GBP_USD", h4, h1, daily)

    def test_sweep_candle_is_a_valid_bearish_sweep(self):
        h4 = self.market.h4
        self.assertGreater(h4["high"].iloc[8], 1.2500)
        self.assertLess(h4["close"].iloc[8], 1.2500)

    def test_no_fvg_forms_so_no_signal(self):
        result = evaluate_setup(self.market, "bearish", sweep_index=8)
        self.assertEqual(result.outcome, "no_fvg")
        self.assertIsNone(result.fvg)


if __name__ == "__main__":
    unittest.main()
