import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_pipeline import config, storage
from research import registry
from research.xs import inference, legs, run, universe
from research.xs.legs import CurrencyData
from research.xs.select import K, _extremes, select
from tests.market_helpers import synthetic_market, truncate


def data_of(markets):
    return {c: CurrencyData(c, m["D"], m["H1"]) for c, m in markets.items()}


def markets(currencies=16, weeks=130, seed=0):
    return {c: synthetic_market(weeks=weeks, seed=seed * 100 + i)
            for i, c in enumerate(universe.CURRENCIES[:currencies])}


class NoLookaheadTests(unittest.TestCase):
    """The ranking may use only what was known when the cohort was entered. Cutting the data off at a
    moment must not change the score or the selection of any cohort entered by then."""

    @classmethod
    def setUpClass(cls):
        cls.markets = markets(12, weeks=140, seed=1)
        cls.full = data_of(cls.markets)

    def compare(self, formation, hold):
        first = self.full["EUR"].h1["time"].iloc[0] + pd.Timedelta(days=100)
        last = self.full["EUR"].h1["time"].iloc[-1]
        entries = legs.monday_entries(first, last)
        full = legs.build_tables(self.full, formation, hold, entries)
        checked = 0
        for position in (0.5, 0.65, 0.8, 0.95):
            cutoff = entries[int(position * len(entries))] + pd.Timedelta(hours=1)
            cut_data = {c: CurrencyData(c, truncate({"D": m["D"], "H4": m["D"], "H1": m["H1"]}, cutoff)["D"],
                                        truncate({"D": m["D"], "H4": m["D"], "H1": m["H1"]}, cutoff)["H1"])
                        for c, m in self.markets.items()}
            early = entries[entries <= cutoff]
            part = legs.build_tables(cut_data, formation, hold, early)
            # Cohorts present in both (eligibility does not depend on the future) must match exactly.
            common = full.cohorts.isin(part.cohorts)
            np.testing.assert_array_equal(full.cohorts[common], part.cohorts[part.cohorts.isin(full.cohorts)])
            f_score = full.score[np.asarray(full.cohorts.isin(part.cohorts))]
            np.testing.assert_array_equal(f_score, part.score[np.asarray(part.cohorts.isin(full.cohorts))])
            f_rank = _extremes(np.where(np.isfinite(f_score), f_score, np.nan), K)
            p_rank = _extremes(np.where(np.isfinite(part.score), part.score, np.nan), K)
            np.testing.assert_array_equal(f_rank[0], p_rank[0])
            np.testing.assert_array_equal(f_rank[1], p_rank[1])
            checked += len(f_score)
        return checked

    def test_momentum_scores_and_selections_do_not_depend_on_later_data(self):
        self.assertGreater(self.compare(84, 4), 50)

    def test_reversal_scores_and_selections_do_not_depend_on_later_data(self):
        self.assertGreater(self.compare(7, 1), 50)

    def test_the_check_itself_catches_a_ranking_that_uses_a_candle_still_forming(self):
        # If the daily candle open at the entry were treated as closed, its finished close (known only
        # later) would leak into the score. The comparison must fail when that is switched on.
        with patch.object(legs, "DAILY_PERIOD_NS", 0):
            with self.assertRaises(AssertionError):
                self.compare(84, 4)

    def test_a_leg_that_finishes_inside_the_cut_matches_the_full_run(self):
        first = self.full["EUR"].h1["time"].iloc[0] + pd.Timedelta(days=100)
        entries = legs.monday_entries(first, self.full["EUR"].h1["time"].iloc[-1])
        full = legs.build_tables(self.full, 84, 4, entries)
        cutoff = entries[len(entries) // 2]
        cut_data = {c: CurrencyData(c, m["D"], truncate({"D": m["D"], "H4": m["D"], "H1": m["H1"]}, cutoff)["H1"])
                    for c, m in self.markets.items()}
        part = legs.build_tables(cut_data, 84, 4, entries[entries <= cutoff])
        rows_full = full.cohorts.get_indexer(part.cohorts)
        finite = np.isfinite(part.gross_long)
        self.assertTrue(finite.any())
        np.testing.assert_array_equal(part.gross_long[finite], full.gross_long[rows_full][finite])


class PipelineCalibrationTests(unittest.TestCase):
    """On random-walk currencies nothing can predict anything, so a rule's p-values must be uniform,
    through the whole path: prices to scores to legs to the permutation null."""

    def p_value(self, seed, rule="momentum", formation=84, hold=4):
        tables = legs.build_tables(data_of(markets(12, weeks=110, seed=seed)), formation, hold)
        null = inference.permutation_null(tables, replicates=300, seed=seed + 500)
        return inference.p_upper(select(tables, rule).gross.mean(), null.gross), tables

    def test_momentum_and_reversal_on_random_walks_get_uniform_p_values(self):
        for rule, formation, hold in (("momentum", 84, 4), ("reversal", 7, 1)):
            p = np.array([self.p_value(seed, rule, formation, hold)[0] for seed in range(24)])
            self.assertTrue(0.28 < p.mean() < 0.72, f"{rule}: mean p {p.mean():.3f}")
            self.assertLessEqual(int((p < 0.05).sum()), 6, rule)

    def test_the_oracle_is_detected_on_the_same_data(self):
        tables = legs.build_tables(data_of(markets(12, weeks=110, seed=3)), 84, 4)
        null = inference.permutation_null(tables, replicates=2000, seed=1)
        self.assertLess(inference.p_upper(select(tables, "oracle").gross.mean(), null.gross), 0.001)

    def test_random_assignments_are_uniform_against_the_null(self):
        tables = legs.build_tables(data_of(markets(12, weeks=110, seed=4)), 84, 4)
        null = inference.permutation_null(tables, replicates=2000, seed=2)
        p = inference.control_p_values(tables, null, draws=200, seed=6)
        self.assertLessEqual(float((p < 0.05).mean()), 0.09)
        self.assertTrue(0.44 < p.mean() < 0.56, p.mean())


class CommandLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.markets = markets(16, weeks=120, seed=7)

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._orig = config.DATA_DIR
        config.DATA_DIR = self._tmp
        self.addCleanup(lambda: (setattr(config, "DATA_DIR", self._orig),
                                 shutil.rmtree(self._tmp, ignore_errors=True)))
        for currency, market in self.markets.items():
            pair = universe.pair_of(currency)
            for tf in ("D", "H1"):
                storage.save_merged(pair, tf, market[tf].to_dict("records"))

    def run_cli(self, *extra, start="all"):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = run.main(["--start", start, "--replicates", "300", "--bootstrap", "300",
                             "--controls", "30", *extra])
        return code, out.getvalue()

    def test_it_runs_both_strategies_with_their_controls_and_says_who_is_a_candidate(self):
        code, output = self.run_cli("--no-register")
        self.assertEqual(code, 0, output[-500:])
        for expected in ("CROSS-SECTIONAL EXPERIMENT", "xs_momentum", "xs_reversal",
                         "formation 84 days, hold 4 week(s)", "formation 7 days, hold 1 week(s)",
                         "X0.", "X9.", "decision rules", "CANDIDATES:"):
            self.assertIn(expected, output)

    def test_the_dense_history_window_is_honoured_when_asked_for(self):
        code, output = self.run_cli("--no-register", start="auto")
        self.assertEqual(code, 0, output[-500:])
        self.assertIn("16 of 16 currencies loaded", output)

    def test_it_writes_the_legs_the_currency_tables_and_the_results(self):
        out_dir = self._tmp / "xs"
        code, _ = self.run_cli("--no-register", "--out-dir", str(out_dir))
        self.assertEqual(code, 0)
        for name in ("results.json", "legs_xs_momentum.csv", "legs_xs_reversal.csv",
                     "per_currency_xs_momentum.csv", "per_currency_xs_reversal.csv"):
            self.assertTrue((out_dir / name).exists(), name)
        saved = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
        momentum = saved["strategies"]["xs_momentum"]
        self.assertEqual(len(momentum["control_p"]), 30)
        self.assertIn("checks", momentum)
        legs_frame = pd.read_csv(out_dir / "legs_xs_momentum.csv")
        self.assertEqual(set(legs_frame["direction"]), {1, -1})

    def test_the_strategies_are_logged_and_the_controls_are_not_counted_as_looks(self):
        self.run_cli()
        runs = registry.load_runs()
        self.assertEqual(sorted(r["strategy"] for r in runs),
                         ["xs_momentum", "xs_momentum:controls", "xs_reversal", "xs_reversal:controls"])
        self.assertTrue(all(r["kind"] == "crosssection" for r in runs))
        counts = registry.trial_counts()
        self.assertEqual(counts["configs"], 2)                  # the controls are checks, not hypotheses
        self.assertEqual(counts["pair_tests"], 32)              # 16 pairs each

    def test_too_few_currencies_is_reported_not_a_crash(self):
        for pair in universe.PAIRS[3:]:
            for tf in ("D", "H1"):
                path = storage.path_for(pair, tf)
                path.unlink()
        code, output = self.run_cli("--no-register")
        self.assertEqual(code, 1)
        self.assertIn("too few currencies", output)


if __name__ == "__main__":
    unittest.main()
