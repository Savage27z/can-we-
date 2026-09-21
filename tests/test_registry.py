import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from research import registry


class RegistryTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.dir, ignore_errors=True))

    def log(self, **over):
        args = dict(kind="backtest", strategy="sweep_fvg", exit="plan", costs="spread", start="auto",
                    instruments=["EUR_USD", "GBP_USD"], results={"pooled": 1}, directory=self.dir)
        return registry.record(**{**args, **over})


class RecordTests(RegistryTestCase):
    def test_an_entry_is_appended_and_read_back(self):
        entry = self.log(note="first")
        (loaded,) = registry.load_runs(self.dir)
        self.assertEqual(loaded["id"], entry["id"])
        self.assertEqual((loaded["strategy"], loaded["exit"], loaded["costs"], loaded["note"]),
                         ("sweep_fvg", "plan", "spread", "first"))
        self.assertEqual(loaded["instruments"], ["EUR_USD", "GBP_USD"])

    def test_the_log_only_grows(self):
        self.log()
        self.log(exit="close")
        self.assertEqual(len(registry.load_runs(self.dir)), 2)

    def test_the_code_version_is_recorded(self):
        entry = self.log()
        self.assertIn("git_commit", entry)
        self.assertIn(entry["git_dirty"], (True, False, None))

    def test_an_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            self.log(kind="guess")

    def test_no_registry_yet_means_no_runs(self):
        self.assertEqual(registry.load_runs(self.dir), [])

    def test_an_unreadable_line_is_skipped_and_the_rest_are_kept(self):
        self.log()
        with registry.registry_path(self.dir).open("a", encoding="utf-8") as f:
            f.write("{this is not json\n")
        self.log(exit="touch")
        with contextlib.redirect_stderr(io.StringIO()):
            runs = registry.load_runs(self.dir)
        self.assertEqual([r["exit"] for r in runs], ["plan", "touch"])

    def test_the_default_location_follows_the_data_directory(self):
        from data_pipeline import config
        original = config.DATA_DIR
        config.DATA_DIR = self.dir
        try:
            self.assertEqual(registry.registry_path(), self.dir / "registry" / "runs.jsonl")
        finally:
            config.DATA_DIR = original


class TrialCountTests(RegistryTestCase):
    def test_rerunning_a_configuration_is_not_a_new_hypothesis(self):
        self.log()
        self.log()
        self.assertEqual(registry.trial_counts(self.dir),
                         {"runs": 2, "configs": 1, "pair_tests": 2})

    def test_a_new_exit_or_cost_model_is_a_new_configuration(self):
        self.log()
        self.log(exit="close")
        self.log(costs="none")
        self.assertEqual(registry.trial_counts(self.dir)["configs"], 3)

    def test_the_same_configuration_on_a_new_instrument_adds_a_test(self):
        self.log(instruments=["EUR_USD"])
        self.log(instruments=["EUR_USD", "AUD_USD"])
        self.assertEqual(registry.trial_counts(self.dir), {"runs": 2, "configs": 1, "pair_tests": 2})

    def test_parameters_distinguish_configurations(self):
        self.log(params={"window": 90})
        self.log(params={"window": 60})
        self.log(params={"window": 90})
        self.assertEqual(registry.trial_counts(self.dir)["configs"], 2)

    def test_counts_can_be_limited_to_one_strategy(self):
        self.log()
        self.log(strategy="other")
        self.assertEqual(registry.trial_counts(self.dir, strategy="other")["configs"], 1)

    def test_an_empty_registry_has_zero_trials(self):
        self.assertEqual(registry.trial_counts(self.dir), {"runs": 0, "configs": 0, "pair_tests": 0})


class ImportTests(RegistryTestCase):
    def test_a_run_saved_before_the_registry_can_be_imported_with_its_original_time(self):
        out = self.dir / "old_run"
        out.mkdir()
        (out / "run.json").write_text(json.dumps({
            "created_at": "2026-09-21T01:00:00+00:00", "git_commit": "abc123",
            "strategy": "sweep_fvg", "exit": "close", "costs": "none", "start": "auto",
            "instruments": ["EUR_USD", "USD_JPY"], "pooled": {"expectancy_net": 0.06}}),
            encoding="utf-8")
        entry = registry.import_run(out, directory=self.dir)
        self.assertEqual(entry["created_at"], "2026-09-21T01:00:00+00:00")
        self.assertTrue(entry["imported"])
        self.assertEqual(entry["git_commit_at_run"], "abc123")
        self.assertEqual(entry["results"], {"expectancy_net": 0.06})
        self.assertEqual(registry.trial_counts(self.dir)["pair_tests"], 2)


class FormatTests(RegistryTestCase):
    def test_the_listing_shows_each_run(self):
        self.log(note="baseline")
        text = registry.format_runs(registry.load_runs(self.dir))
        self.assertIn("sweep_fvg", text)
        self.assertIn("baseline", text)


if __name__ == "__main__":
    unittest.main()
