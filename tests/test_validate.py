import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import config, storage
from research import registry, validate, walk_forward
from research.null_model import NullResult
from tests.test_research import section81_frames


def outcome(instrument, r_values, null_means, cost=0.05):
    """A worker's result: real trades (as R values) plus a null with the given replicate means."""
    trades = pd.DataFrame({"outcome": ["open" if np.isnan(r) else ("win" if r > 0 else "loss") for r in r_values],
                           "r_net": r_values, "cost_r": cost,
                           "entry_time": pd.date_range("2015-01-01", periods=len(r_values), freq="30D", tz="UTC")})
    n = len(r_values)
    null = np.asarray(null_means, dtype=float)
    return {"instrument": instrument, "trades": trades, "null_sums": null * n,
            "null_counts": np.full(len(null), n, dtype=np.int64),
            "data_from": pd.Timestamp("2010-01-01", tz="UTC")}


def spread_of_nulls(center, k=400, seed=0, sd=0.1):
    return np.random.default_rng(seed).normal(center, sd, k)


class TableTests(unittest.TestCase):
    def setUp(self):
        self.outcomes = [
            outcome("STRONG", [2.0, 1.5, 1.0, -1.0, 2.0, 1.5] * 5, spread_of_nulls(-0.1, seed=1)),
            outcome("WEAK", [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0] * 5, spread_of_nulls(0.0, seed=2)),
            outcome("BAD", [-1.0, -1.0, 0.5, -1.0, -1.0, 0.5] * 5, spread_of_nulls(0.0, seed=3)),
        ]

    def test_each_instrument_gets_its_real_null_excess_and_p(self):
        table, nulls = validate.per_instrument_table(self.outcomes)
        row = table.set_index("instrument")
        self.assertAlmostEqual(row.loc["STRONG", "real"], np.mean([2.0, 1.5, 1.0, -1.0, 2.0, 1.5]))
        self.assertAlmostEqual(row.loc["STRONG", "null"], -0.1, delta=0.02)
        self.assertAlmostEqual(row.loc["STRONG", "excess"], row.loc["STRONG", "real"] - row.loc["STRONG", "null"])
        self.assertLess(row.loc["STRONG", "p"], 0.01)
        self.assertGreater(row.loc["BAD", "p"], 0.9)
        self.assertEqual(set(nulls), {"STRONG", "WEAK", "BAD"})

    def test_each_instruments_null_spread_is_reported_for_the_power_statement(self):
        table, _ = validate.per_instrument_table(self.outcomes)
        self.assertTrue((table["null_sd"] > 0).all())
        self.assertAlmostEqual(table.set_index("instrument").loc["WEAK", "null_sd"], 0.1, delta=0.02)

    def test_corrections_are_never_smaller_than_the_raw_p(self):
        table, _ = validate.per_instrument_table(self.outcomes)
        for col in ("p_bh", "p_holm", "p_maxt"):
            self.assertTrue((table[col] >= table["p"] - 1e-12).all(), col)

    def test_a_strong_result_survives_correction_and_a_weak_one_does_not(self):
        table, _ = validate.per_instrument_table(self.outcomes)
        row = table.set_index("instrument")
        self.assertLess(row.loc["STRONG", "p_bh"], 0.05)
        self.assertLess(row.loc["STRONG", "p_maxt"], 0.05)
        self.assertGreater(row.loc["WEAK", "p_bh"], 0.05)

    def test_an_instrument_with_no_resolved_trades_is_left_out(self):
        empty = outcome("NONE", [np.nan, np.nan], spread_of_nulls(0.0, seed=4))
        table, _ = validate.per_instrument_table(self.outcomes + [empty])
        self.assertNotIn("NONE", set(table["instrument"]))

    def test_no_instruments_gives_an_empty_table(self):
        table, nulls = validate.per_instrument_table([])
        self.assertTrue(table.empty)
        self.assertEqual(nulls, {})

    def test_the_average_cost_per_trade_is_reported(self):
        table, _ = validate.per_instrument_table([outcome("A", [1.0, -1.0] * 5, spread_of_nulls(0, seed=5), cost=0.2)] + self.outcomes)
        self.assertAlmostEqual(table.set_index("instrument").loc["A", "avg_cost_r"], 0.2)


class FormattingTests(unittest.TestCase):
    def test_the_majors_are_shown_even_when_they_are_not_among_the_best(self):
        outcomes = [outcome(f"P{i:02d}", [1.0, -1.0, 2.0] * 8, spread_of_nulls(0.0, seed=i))
                    for i in range(20)]
        outcomes.append(outcome("EUR_USD", [-1.0, -1.0, 0.5] * 8, spread_of_nulls(0.0, seed=99)))
        table, _ = validate.per_instrument_table(outcomes)
        text = validate.format_table(table, rows=5)
        self.assertIn("EUR_USD", text)
        self.assertEqual(len([l for l in text.splitlines() if l.startswith("P")]), 5)

    def test_missing_numbers_print_as_placeholders_not_nan(self):
        table = pd.DataFrame([{"instrument": "X", "n": 3, "real": np.nan, "null": np.nan,
                               "excess": np.nan, "z": np.nan, "p": np.nan, "p_bh": np.nan,
                               "p_maxt": np.nan, "avg_cost_r": 0.1}])
        text = validate.format_table(table, rows=5)
        self.assertIn("n/a", text)
        self.assertNotIn("nan", text)

    def test_walk_forward_text_reports_the_test_and_its_p(self):
        rows = []
        rng = np.random.default_rng(0)
        for name in [f"P{i:02d}" for i in range(12)]:
            for year in range(2005, 2025):
                for _ in range(8):
                    rows.append((name, pd.Timestamp(f"{year}-06-01", tz="UTC"), rng.normal(0, 1)))
        df = pd.DataFrame(rows, columns=["instrument", "entry_time", "r_net"])
        df["exit_time"] = df["entry_time"] + pd.Timedelta(days=2)
        wf = walk_forward.walk_forward_selection(df, top_n=3, replicates=200)
        text = validate.format_walk_forward(wf, 3)
        self.assertIn("selected pairs, out of sample", text)
        self.assertIn("beat random picking with p =", text)

    def test_walk_forward_with_nothing_to_test_says_so(self):
        self.assertIn("not enough history", validate.format_walk_forward(walk_forward.WalkForwardResult(), 5))


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._orig = config.DATA_DIR
        config.DATA_DIR = self._tmp
        self.addCleanup(lambda: (setattr(config, "DATA_DIR", self._orig),
                                 shutil.rmtree(self._tmp, ignore_errors=True)))
        for tf, df in section81_frames().items():
            storage.save_merged("EUR_USD", tf, df.to_dict("records"))

    def run_cli(self, *args, pairs=("EUR_USD",)):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = validate.main(["--pairs", *pairs, "--start", "all", "--exit", "close",
                                  "--costs", "none", "--null-replicates", "60", *args])
        return code, out.getvalue()

    def test_it_runs_end_to_end_and_says_what_it_did(self):
        code, output = self.run_cli("--no-register")
        self.assertEqual(code, 0)
        for expected in ("VALIDATION", "POOLED", "WALK-FORWARD", "random-entry replays"):
            self.assertIn(expected, output)

    def test_it_writes_the_tables_and_the_settings(self):
        out_dir = self._tmp / "validation"
        code, _ = self.run_cli("--no-register", "--out-dir", str(out_dir))
        self.assertEqual(code, 0)
        for name in ("validation.csv", "walk_forward.csv", "trades.csv", "validation.json"):
            self.assertTrue((out_dir / name).exists(), name)
        saved = json.loads((out_dir / "validation.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["settings"]["null_replicates"], 60)

    def test_a_validation_is_logged_in_the_registry_with_its_null_settings(self):
        code, output = self.run_cli()
        self.assertEqual(code, 0)
        (run,) = registry.load_runs()
        self.assertEqual((run["kind"], run["exit"], run["costs"]), ("validation", "close", "none"))
        self.assertEqual(run["null"]["replicates"], 60)
        self.assertIn("REGISTRY: 1 runs", output)

    def test_no_register_leaves_the_registry_untouched(self):
        self.run_cli("--no-register")
        self.assertEqual(registry.load_runs(), [])

    def test_an_instrument_that_cannot_be_run_is_named_and_the_rest_still_count(self):
        code, output = self.run_cli("--no-register", pairs=("EUR_USD", "GBP_USD"))
        self.assertEqual(code, 0)
        self.assertIn("SKIPPED GBP_USD", output)
        self.assertIn("POOLED over 1 instruments", output)

    def test_when_nothing_can_be_run_the_exit_code_says_so(self):
        code, output = self.run_cli("--no-register", pairs=("GBP_USD",))
        self.assertEqual(code, 1)
        self.assertIn("FAILED GBP_USD", output)


class DecompositionTests(unittest.TestCase):
    def test_gross_is_net_plus_cost_for_real_and_random(self):
        resolved = pd.DataFrame({"r_net": [1.0, -1.5, 0.5], "cost_r": [0.1, 0.3, 0.2]})
        null = NullResult(sums=np.array([-3.0, -3.0]), counts=np.array([10, 10]),
                          cost_sums=np.array([2.0, 2.0]))
        d = validate.decompose(resolved, null)
        self.assertAlmostEqual(d["real"]["net"], 0.0)
        self.assertAlmostEqual(d["real"]["cost"], 0.2)
        self.assertAlmostEqual(d["real"]["gross"], 0.2)
        self.assertAlmostEqual(d["random"]["net"], -0.3)
        self.assertAlmostEqual(d["random"]["cost"], 0.2)
        self.assertAlmostEqual(d["random"]["gross"], -0.1)

    def test_the_text_shows_timing_and_cost_separately(self):
        d = {"real": {"gross": -0.037, "cost": 0.388, "net": -0.426},
             "random": {"gross": -0.019, "cost": 0.317, "net": -0.336}}
        text = validate.format_decomposition(d)
        self.assertIn("real minus random", text)
        last = text.splitlines()[-1]
        self.assertIn("-0.018", last)       # timing, before costs
        self.assertIn("-0.071", last)       # extra spread paid
        self.assertIn("-0.090", last)       # net


class EntryHoursTests(unittest.TestCase):
    def test_hours_are_the_candle_open_not_its_close_and_share_is_reported(self):
        # entry_time is the entry candle's close, so an entry closing at 08:00 opened at 07:00.
        times = ["2024-01-08 08:00"] * 6 + ["2024-01-08 22:00"] * 3 + ["2024-01-08 14:00"]
        resolved = pd.DataFrame({"entry_time": pd.to_datetime(times, utc=True)})
        text = validate.entry_hours(resolved)
        self.assertTrue(text.startswith("07:00 60%"), text)
        self.assertIn("21:00 30%", text)
        self.assertIn("13:00 10%", text)


class ReportIncludesTheDecompositionTests(unittest.TestCase):
    def test_the_command_line_report_shows_it(self):
        tmp = Path(tempfile.mkdtemp())
        original = config.DATA_DIR
        config.DATA_DIR = tmp
        try:
            for tf, df in section81_frames().items():
                storage.save_merged("EUR_USD", tf, df.to_dict("records"))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = validate.main(["--pairs", "EUR_USD", "--start", "all", "--exit", "close",
                                      "--costs", "none", "--null-replicates", "60", "--no-register"])
        finally:
            config.DATA_DIR = original
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertEqual(code, 0)
        self.assertIn("per trade, in R", out.getvalue())
        self.assertIn("would have detected an entry-timing excess", out.getvalue())
        self.assertIn("most common entry-candle opens", out.getvalue())


if __name__ == "__main__":
    unittest.main()
