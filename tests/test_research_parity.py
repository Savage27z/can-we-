"""The research engine must reproduce the original backtest exactly.

Runs on a frozen copy of the 8-year, 3-pair data the original result (EUR_USD: 37 signals,
+0.34R) was measured on, kept out of git under data/_snapshots/phase2_8y. Without that
snapshot these tests are skipped; a newer, longer data store gives different numbers by
design, so it cannot serve as the reference.
"""
import math
import unittest
from collections import Counter
from pathlib import Path

from backtest.engine import find_sweep_events, load_market_data
from backtest.setup import evaluate_setup
from data_pipeline import config
from research import engine, exits, scoring
from research.costs import NoCost
from research.strategies import get_strategy

SNAPSHOT = Path(__file__).resolve().parent.parent / "data" / "_snapshots" / "phase2_8y"
TRADED = {"win", "loss", "open"}


@unittest.skipUnless(SNAPSHOT.exists(), "frozen 8-year snapshot not present")
class ParityWithTheOriginalBacktest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._original = config.DATA_DIR
        config.DATA_DIR = SNAPSHOT
        cls.legacy = {}
        cls.runs = {}
        for pair in ("EUR_USD", "GBP_USD", "USD_JPY"):
            market = load_market_data(pair)
            cls.legacy[pair] = [evaluate_setup(market, d, i) for i, d in find_sweep_events(market)]
            cls.runs[pair] = engine.run_instrument(get_strategy("sweep_fvg"), pair,
                                                   exits.get_exit("close"), NoCost())

    @classmethod
    def tearDownClass(cls):
        config.DATA_DIR = cls._original

    def test_the_funnel_is_identical(self):
        for pair, results in self.legacy.items():
            self.assertEqual(self.runs[pair].funnel, dict(Counter(r.outcome for r in results)), pair)

    def test_every_trade_matches_the_legacy_outcome_r_and_exit_time(self):
        for pair, results in self.legacy.items():
            traded = [r for r in results if r.outcome in TRADED]
            trades = self.runs[pair].trades
            self.assertEqual(len(trades), len(traded), pair)
            for legacy, (_, ours) in zip(traded, trades.iterrows()):
                context = f"{pair} sweep {legacy.sweep_time}"
                self.assertEqual(ours["sweep_time"], legacy.sweep_time, context)
                self.assertEqual(ours["outcome"], legacy.outcome, context)
                self.assertEqual(ours["entry_time"], legacy.confirm_time, context)
                self.assertEqual(ours["entry_price"], legacy.entry_price, context)
                if legacy.outcome == "open":
                    self.assertTrue(math.isnan(ours["r_gross"]), context)
                else:
                    self.assertEqual(ours["r_gross"], legacy.realized_r, context)
                    self.assertEqual(ours["exit_time"], legacy.resolve_time, context)

    def test_the_gate_result_is_reproduced(self):
        run = self.runs["EUR_USD"]
        summary = scoring.summarize(run.trades, run.months_spanned)
        self.assertEqual((summary["signals"], summary["wins"], summary["losses"]), (37, 14, 23))
        self.assertAlmostEqual(summary["expectancy_gross"], 0.34, places=2)
        self.assertAlmostEqual(summary["max_dd_gross"], 6.36, places=2)
        self.assertAlmostEqual(summary["avg_planned_rr"], 2.62, places=2)
        self.assertAlmostEqual(summary["signals_per_month"], 0.39, places=2)


if __name__ == "__main__":
    unittest.main()
