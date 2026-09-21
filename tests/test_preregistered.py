import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research import preregistered as pre


def write_run(directory: Path, early=(0.1, 40), late=(0.1, 40), spread=1.0, pooled_p=0.001,
              excess=0.1, pair_p=None, settings=None, seed=0):
    """A saved validation run whose trades average `early`/`late` = (mean R, count) either side of 2016."""
    rng = np.random.default_rng(seed)
    rows = []
    for (mean, n), start in ((early, "2008-01-01"), (late, "2018-01-01")):
        r = rng.normal(mean, spread, n)
        r = r - r.mean() + mean                       # exact mean, so a boundary case is exact
        times = pd.date_range(start, periods=n, freq="10D", tz="UTC")     # 200 trades stay inside their half
        rows += [{"instrument": "EUR_USD", "entry_time": t, "outcome": "win" if x > 0 else "loss",
                  "planned_rr": 2.0, "r_gross": x, "cost_r": 0.0, "r_net": x}
                 for t, x in zip(times, r)]
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(directory / "trades.csv", index=False)
    pairs = pair_p if pair_p is not None else [0.3, 0.6, 0.5]
    pd.DataFrame({"instrument": [f"P{i}" for i in range(len(pairs))], "p": pairs}).to_csv(
        directory / "validation.csv", index=False)
    base = {"exit": "touch", "costs": "spread", "null_direction": "random"}
    (directory / "validation.json").write_text(json.dumps({
        "settings": {**base, **(settings or {})},
        "results": {"pooled": {"excess": excess, "p": pooled_p}}}), encoding="utf-8")
    return pre.load_run(directory)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def evaluate(self, **kwargs):
        return pre.evaluate_candidate(write_run(self.tmp / "run", **kwargs))

    def failed(self, checks):
        return [c.rule[:2] for c in checks if not c.passed]

    def test_a_strategy_meeting_every_rule_is_a_candidate(self):
        checks = self.evaluate(early=(0.5, 200), late=(0.5, 200), spread=1.0, pooled_p=0.001)
        self.assertEqual(pre.verdict(checks), "PASS", pre.format_checks("x", checks, ""))

    def test_a_positive_mean_whose_interval_includes_zero_fails_rule_1(self):
        checks = self.evaluate(early=(0.2, 15), late=(0.2, 15), spread=2.0)
        self.assertIn("1.", self.failed(checks))

    def test_a_negative_expectancy_fails_rule_1(self):
        checks = self.evaluate(early=(-0.3, 200), late=(-0.3, 200))
        self.assertIn("1.", self.failed(checks))

    def test_a_p_value_that_would_pass_alone_but_not_after_counting_three_strategies_fails(self):
        checks = self.evaluate(early=(0.5, 200), late=(0.5, 200), pooled_p=0.03)
        self.assertEqual(self.failed(checks), ["2."])

    def test_a_negative_excess_fails_rule_2_even_with_a_tiny_p(self):
        checks = self.evaluate(early=(0.5, 200), late=(0.5, 200), excess=-0.1, pooled_p=0.001)
        self.assertEqual(self.failed(checks), ["2."])

    def test_it_must_be_positive_in_both_halves(self):
        checks = self.evaluate(early=(1.0, 200), late=(-0.05, 200))
        self.assertIn("3.", self.failed(checks))

    def test_the_bonferroni_threshold_is_a_third_of_five_percent(self):
        self.assertAlmostEqual(pre.PRIMARY_P, 0.05 / 3)

    def test_a_run_with_the_wrong_null_is_flagged_as_a_deviation(self):
        checks = self.evaluate(early=(0.5, 200), late=(0.5, 200), settings={"null_direction": "same"})
        config = checks[0]
        self.assertFalse(config.passed)
        self.assertIn("DEVIATION", config.detail)
        self.assertEqual(pre.verdict(checks), "FAIL")

    def test_a_run_with_the_wrong_exit_or_costs_is_flagged_too(self):
        for change in ({"exit": "plan"}, {"costs": "none"}):
            checks = self.evaluate(early=(0.5, 200), late=(0.5, 200), settings=change)
            self.assertFalse(checks[0].passed, change)


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def evaluate(self, **kwargs):
        return pre.evaluate_control(write_run(self.tmp / "run", **kwargs))

    def test_a_control_that_looks_like_noise_passes(self):
        uniform = list(np.linspace(0.02, 0.98, 60))
        checks = self.evaluate(pooled_p=0.4, pair_p=uniform)
        self.assertEqual(pre.verdict(checks), "PASS", pre.format_checks("c", checks, ""))

    def test_an_extreme_pooled_p_fails(self):
        for p in (0.005, 0.995):
            checks = self.evaluate(pooled_p=p, pair_p=list(np.linspace(0.02, 0.98, 60)))
            self.assertFalse(checks[1].passed, p)

    def test_too_many_pairs_below_five_percent_fails(self):
        many_small = [0.01] * 12 + list(np.linspace(0.3, 0.9, 48))
        checks = self.evaluate(pooled_p=0.5, pair_p=many_small)
        self.assertFalse(checks[2].passed)

    def test_a_biased_mean_pair_p_fails(self):
        skewed = list(np.linspace(0.5, 0.99, 60))
        checks = self.evaluate(pooled_p=0.5, pair_p=skewed)
        self.assertFalse(checks[3].passed)


class CommandLineTests(unittest.TestCase):
    def test_it_prints_each_rule_and_returns_nonzero_when_a_run_fails(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            write_run(tmp / "good", early=(0.5, 200), late=(0.5, 200))
            write_run(tmp / "bad", early=(-0.5, 200), late=(-0.5, 200))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = pre.main(["good", "bad", "--runs-dir", str(tmp)])
            text = out.getvalue()
            self.assertEqual(code, 1)
            self.assertIn("good: candidate under the pre-registered rules: PASS", text)
            self.assertIn("bad: candidate under the pre-registered rules: FAIL", text)
            self.assertIn("[FAIL] 1.", text)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
