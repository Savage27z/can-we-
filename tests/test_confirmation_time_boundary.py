"""Regression test for the audit finding in backtest/setup.py: `confirm_time`
must be the confirming H1 candle's CLOSE (when entry_price is actually known),
not its open. Using the open time made the look-ahead cutoff passed to
xtf.h4_index_fully_closed_by up to an hour too conservative.

This scenario is built specifically so the confirming candle's true close lands
exactly on an H4 grid boundary — the one condition where the bug manifests.
tests/test_worked_examples.py's §8.1 regression does NOT exercise this (its
target level's confirmed_at/mitigated_at sit nowhere near the boundary), which
is why that bug shipped past 63 passing tests undetected; this test exists so
it can't happen again silently.

Precise construction (see the inline index comments): a swing high at index 2
(level 1.0900) becomes mitigated at index 7 (the first candle whose high
exceeds it). The confirming H1 candle opens at 08:00 and closes at 09:00 —
09:00 is exactly h4.time[7]. With confirm_time correctly set to the 09:00
close, h4_index_fully_closed_by resolves to index 7, so the level is correctly
recognized as already-mitigated (dead) and excluded as a target -> "no_target".
With the old bug (confirm_time = 08:00, the open), the cutoff resolves to
index 6 instead, one candle short — the level would incorrectly still look
"unmitigated" and get wrongly selected as a live target (target_price=1.0900),
producing a real (but wrong) trade signal instead of "no_target".
"""
import unittest

import pandas as pd

from backtest.fractals import build_levels, find_swings
from backtest.sessions import in_session_mask
from backtest.setup import MarketData, evaluate_setup


class ConfirmationTimeBoundaryTest(unittest.TestCase):
    def test_already_mitigated_target_is_excluded_at_the_exact_boundary(self):
        start = "2024-01-01 01:00"  # H4 grid: 01/05/09/13/17/21 UTC
        n = 12
        # idx: 0     1     2      3(sweep) 4      5      6      7      8-11 (filler)
        highs = [1.0850, 1.0860, 1.0900, 1.0880, 1.0812, 1.0850, 1.0870, 1.0920,
                 1.0900, 1.0900, 1.0900, 1.0900]
        lows = [1.0840, 1.0850, 1.0890, 1.0800, 1.0700, 1.0806, 1.0820, 1.0900,
                1.0880, 1.0880, 1.0880, 1.0880]
        closes = [1.0845, 1.0855, 1.0895, 1.0810, 1.0705, 1.0845, 1.0860, 1.0910,
                  1.0890, 1.0890, 1.0890, 1.0890]
        # index 2: swing high, level 1.0900, confirmed_at=4.
        # index 3: sweep candle (bullish), sweep_extreme = low[3] = 1.0800.
        # index 4,5,6: FVG candles (i-1, i, i+1) -> low[6]=1.0820 > high[4]=1.0812.
        # index 7: the candle that mitigates the index-2 swing high (first high > 1.0900).

        times = pd.date_range(start, periods=n, freq="4h", tz="UTC")
        h4 = pd.DataFrame({
            "time": times,
            "open": [(h + l) / 2 for h, l in zip(highs, lows)],
            "high": highs, "low": lows, "close": closes, "volume": [10] * n,
        })
        h4 = find_swings(h4).reset_index(drop=True)
        levels = build_levels(h4)

        target = next(lvl for lvl in levels if lvl.index == 2)
        self.assertEqual(target.mitigated_at, 7, "test construction assumption broke")

        # FVG forms at mid=5 (candles 4/5/6); formed-time reference is h4.time[6]+4h
        # = 2024-01-02 05:00 = h4.time[7] exactly.
        fvg_formed_time = h4["time"].iloc[6] + pd.Timedelta(hours=4)
        self.assertEqual(fvg_formed_time, h4["time"].iloc[7])

        # H1 candles from formed_time; confirmation happens at offset 3 (08:00,
        # session-qualified) with close 1.0815 > confirmation_level 1.0812.
        # That candle's CLOSE (09:00) is exactly h4.time[7] — the boundary.
        h1_closes = [1.0805, 1.0806, 1.0808, 1.0815, 1.0816, 1.0817]
        h1_times = pd.date_range(fvg_formed_time, periods=len(h1_closes), freq="1h", tz="UTC")
        h1 = pd.DataFrame({
            "time": h1_times, "open": h1_closes,
            "high": [c + 0.0002 for c in h1_closes], "low": [c - 0.0002 for c in h1_closes],
            "close": h1_closes, "volume": [1] * len(h1_closes),
        })

        daily = pd.DataFrame({
            "time": pd.date_range("2023-12-25", periods=3, freq="24h", tz="UTC"),
            "open": [1.05, 1.06, 1.07], "high": [1.05, 1.06, 1.07], "low": [1.05, 1.06, 1.07],
            "close": [1.05, 1.06, 1.07], "volume": [1] * 3,
        })

        market = MarketData(
            pair="EUR_USD", h4=h4, h1=h1, daily=daily, levels=levels,
            h4_session=in_session_mask(h4["time"]), h1_session=in_session_mask(h1["time"]),
        )

        result = evaluate_setup(market, "bullish", sweep_index=3)

        self.assertEqual(result.confirm_index, 3)
        # The whole point: confirm_time must be the CLOSE (09:00), not the
        # OPEN (08:00) of the confirming candle.
        self.assertEqual(result.confirm_time, pd.Timestamp("2024-01-02 09:00", tz="UTC"))
        # With confirm_time correct, the already-mitigated level at index 2 is
        # correctly excluded and there is no other eligible target -> no_target.
        # (With the pre-fix bug, this would instead resolve target_price=1.0900.)
        self.assertEqual(result.outcome, "no_target")
        self.assertIsNone(result.target_price)


if __name__ == "__main__":
    unittest.main()
