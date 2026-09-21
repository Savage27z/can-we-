import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_pipeline import config, quality, storage
from research import engine, null_model, run_backtest, validate
from research.strategies import get_strategy
from tests.market_helpers import synthetic_market

NEW = ("donchian_trend", "range_breakout", "daily_reversion", "random_control",
       "sweep_fvg_no_rollover")


class PipelineTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.market = synthetic_market(weeks=260, seed=11)

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._orig = config.DATA_DIR
        config.DATA_DIR = self._tmp
        self.addCleanup(lambda: (setattr(config, "DATA_DIR", self._orig),
                                 shutil.rmtree(self._tmp, ignore_errors=True)))
        for tf, frame in self.market.items():
            storage.save_merged("EUR_USD", tf, frame.to_dict("records"))

    def run_validate(self, strategy, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = validate.main(["--strategy", strategy, "--pairs", "EUR_USD", "--start", "all",
                                  "--exit", "touch", "--costs", "spread", "--null-replicates", "30",
                                  "--no-register", *extra])
        return code, out.getvalue()


class ValidationRunsForEveryStrategy(PipelineTestCase):
    def test_each_strategy_runs_through_the_validation_cli(self):
        for name in NEW:
            code, output = self.run_validate(name)
            self.assertEqual(code, 0, f"{name}: {output[-400:]}")
            self.assertIn("POOLED", output, name)

    def test_the_null_direction_can_be_randomised_for_a_directional_strategy(self):
        code, output = self.run_validate("donchian_trend", "--null-direction", "random")
        self.assertEqual(code, 0)
        self.assertIn("direction random", output)

    def test_each_strategy_runs_through_the_backtest_cli(self):
        for name in NEW:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = run_backtest.main(["--strategy", name, "--pairs", "EUR_USD", "--start",
                                          "all", "--exit", "touch", "--costs", "spread"])
            self.assertEqual(code, 0, name)


class NullUsesTheStrategysOwnCandles(PipelineTestCase):
    def eligible_hours(self, strategy):
        seen = {}
        original = null_model.draw_entries

        def spy(entry_index, times_ns, eligible, *args, **kwargs):
            seen["eligible"] = eligible
            return original(entry_index, times_ns, eligible, *args, **kwargs)

        with patch.object(null_model, "draw_entries", spy):
            self.run_validate(strategy)
        h1 = storage.load("EUR_USD", "H1")
        return set(h1["time"].iloc[seen["eligible"]].dt.hour)

    def test_a_london_open_strategy_is_compared_with_random_london_open_entries(self):
        self.assertEqual(self.eligible_hours("donchian_trend"), {7})
        self.assertEqual(self.eligible_hours("daily_reversion"), {7})

    def test_a_morning_breakout_is_compared_with_random_morning_entries(self):
        self.assertEqual(self.eligible_hours("range_breakout"), {7, 8, 9, 10})

    def test_the_no_rollover_variant_never_draws_the_rollover_candle(self):
        hours = self.eligible_hours("sweep_fvg_no_rollover")
        self.assertNotIn(21, hours)
        self.assertIn(13, hours)

    def test_the_original_strategy_still_draws_from_the_whole_session(self):
        self.assertIn(21, self.eligible_hours("sweep_fvg"))


class WarmupTests(unittest.TestCase):
    def test_a_strategy_states_how_much_daily_history_it_needs(self):
        self.assertEqual(engine.warmup_of(get_strategy("daily_reversion")),
                         pd.Timedelta(days=get_strategy("daily_reversion").daily_warmup_days))
        self.assertEqual(engine.warmup_of(get_strategy("sweep_fvg")), engine.DAILY_WARMUP)

    def test_the_window_keeps_that_much_daily_history_before_the_start(self):
        dense = {"D": pd.Timestamp("2002-01-01", tz="UTC"), "H4": pd.Timestamp("2005-01-01", tz="UTC"),
                 "H1": pd.Timestamp("2005-01-01", tz="UTC")}
        with patch.object(quality, "usable_windows", return_value=dense):
            strategy = get_strategy("daily_reversion")
            window = engine.window_for(strategy.timeframes, "EUR_USD", "auto",
                                       engine.warmup_of(strategy))
            short = engine.window_for(strategy.timeframes, "EUR_USD", "auto")
        self.assertEqual(window.start, pd.Timestamp("2005-01-01", tz="UTC"))
        self.assertEqual(window.start - window.daily_start,
                         pd.Timedelta(days=strategy.daily_warmup_days))
        self.assertEqual(short.start - short.daily_start, engine.DAILY_WARMUP)

    def test_the_daily_history_never_reaches_back_before_the_daily_data_is_dense(self):
        dense = {"D": pd.Timestamp("2004-12-01", tz="UTC"), "H1": pd.Timestamp("2005-01-01", tz="UTC")}
        with patch.object(quality, "usable_windows", return_value=dense):
            window = engine.window_for(("D", "H1"), "EUR_USD", "auto", pd.Timedelta(days=300))
        self.assertEqual(window.daily_start, pd.Timestamp("2004-12-01", tz="UTC"))


if __name__ == "__main__":
    unittest.main()
