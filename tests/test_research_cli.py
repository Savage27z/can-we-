import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_pipeline import config, storage
from research import engine, run_backtest
from research.strategies import get_strategy
from tests.test_research import section81_frames

UTC = "UTC"


def candle_frame(times) -> pd.DataFrame:
    return pd.DataFrame({"time": times, "open": 1.1, "high": 1.11, "low": 1.09,
                         "close": 1.1, "volume": 1})


class TempStore(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._orig = config.DATA_DIR
        config.DATA_DIR = self._tmp
        self.addCleanup(lambda: (setattr(config, "DATA_DIR", self._orig),
                                 shutil.rmtree(self._tmp, ignore_errors=True)))

    def store(self, pair, tf, df):
        storage.save_merged(pair, tf, df.to_dict("records"))


class WindowTests(TempStore):
    def frames(self):
        return {"D": candle_frame(pd.date_range("2024-01-01", periods=120, freq="1D", tz=UTC)),
                "H4": candle_frame(pd.date_range("2024-01-01", periods=600, freq="4h", tz=UTC)),
                "H1": candle_frame(pd.date_range("2024-01-01", periods=2400, freq="1h", tz=UTC))}

    def test_a_window_cuts_h4_and_h1_at_the_start_and_daily_a_little_earlier(self):
        for tf, df in self.frames().items():
            self.store("EUR_USD", tf, df)
        start = pd.Timestamp("2024-03-01", tz=UTC)
        window = engine.window_from_start(start)
        loaded = engine.load_frames("EUR_USD", ("D", "H4", "H1"), window)
        self.assertEqual(loaded["H1"]["time"].iloc[0], start)
        self.assertEqual(loaded["H4"]["time"].iloc[0], start)
        self.assertEqual(loaded["D"]["time"].iloc[0], start - pd.Timedelta(days=14))
        for df in loaded.values():
            self.assertEqual(list(df.index), list(range(len(df))))

    def test_without_a_window_everything_is_loaded(self):
        for tf, df in self.frames().items():
            self.store("EUR_USD", tf, df)
        loaded = engine.load_frames("EUR_USD", ("D", "H4", "H1"))
        self.assertEqual([len(loaded[tf]) for tf in ("D", "H4", "H1")], [120, 600, 2400])

    def test_the_h1_frame_is_always_loaded_even_if_the_strategy_did_not_ask(self):
        self.store("EUR_USD", "H1", self.frames()["H1"])
        self.assertIn("H1", engine.load_frames("EUR_USD", ("H4",)))

    def test_a_window_that_leaves_no_h1_is_an_error(self):
        for tf, df in self.frames().items():
            self.store("EUR_USD", tf, df)
        window = engine.window_from_start(pd.Timestamp("2030-01-01", tz=UTC))
        with self.assertRaises(ValueError):
            engine.run_instrument(get_strategy("sweep_fvg"), "EUR_USD",
                                  run_backtest.get_exit("close"), run_backtest.get_cost("none"),
                                  window=window)

    def test_window_from_start(self):
        window = engine.window_from_start(pd.Timestamp("2020-06-15", tz=UTC))
        self.assertEqual(window.daily_start, pd.Timestamp("2020-06-01", tz=UTC))

    def test_no_data_means_no_usable_window(self):
        self.assertIsNone(engine.usable_window("EUR_USD", ("D", "H4", "H1")))


class CommandLineTests(TempStore):
    def setUp(self):
        super().setUp()
        for tf, df in section81_frames().items():
            self.store("EUR_USD", tf, df)

    def run_cli(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = run_backtest.main(["--pairs", "EUR_USD", "--start", "all", *args])
        return code, out.getvalue()

    def test_it_runs_and_prints_a_row_for_the_pair(self):
        code, output = self.run_cli()
        self.assertEqual(code, 0)
        self.assertIn("EUR_USD", output)
        self.assertIn("95% CI on netR", output)

    def test_it_can_write_the_trades_the_summary_and_a_run_record(self):
        out_dir = self._tmp / "run"
        code, _ = self.run_cli("--out-dir", str(out_dir))
        self.assertEqual(code, 0)
        trades = pd.read_csv(out_dir / "trades.csv")
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades["outcome"].iloc[0], "win")
        self.assertEqual(len(pd.read_csv(out_dir / "summary.csv")), 1)
        record = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
        self.assertEqual((record["strategy"], record["exit"], record["costs"], record["start"]),
                         ("sweep_fvg", "close", "none", "all"))
        self.assertEqual(record["instruments"], ["EUR_USD"])

    def test_an_instrument_with_no_data_is_reported_as_a_failure_and_the_exit_code_says_so(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = run_backtest.main(["--pairs", "EUR_USD", "GBP_USD", "--start", "all"])
        self.assertEqual(code, 1)
        self.assertIn("FAILED GBP_USD", out.getvalue())
        self.assertIn("EUR_USD", out.getvalue())          # the good instrument still ran

    def test_costs_without_spread_data_fail_loudly_for_that_instrument(self):
        code, output = self.run_cli("--costs", "spread")
        self.assertEqual(code, 1)
        self.assertIn("MissingSpreadData", output)

    def test_an_explicit_start_date_becomes_a_window(self):
        window = run_backtest._window_for("sweep_fvg", "EUR_USD", "2020-01-01")
        self.assertEqual(window.start, pd.Timestamp("2020-01-01", tz=UTC))
        self.assertIsNone(run_backtest._window_for("sweep_fvg", "EUR_USD", "all"))

    def test_auto_on_an_instrument_with_no_data_says_why(self):
        with self.assertRaises(ValueError) as ctx:
            run_backtest._window_for("sweep_fvg", "GBP_USD", "auto")
        self.assertIn("no dense history", str(ctx.exception))


class TableTests(unittest.TestCase):
    def test_the_table_shows_where_each_run_started(self):
        from research import scoring
        trades = pd.DataFrame({"outcome": ["win", "loss"], "planned_rr": [2.0, 2.0],
                               "r_gross": [2.0, -1.0], "r_net": [2.0, -1.0], "cost_r": [0.0, 0.0],
                               "entry_time": pd.date_range("2024-01-01", periods=2, tz=UTC)})
        row = run_backtest.summary_row("EUR_USD", scoring.summarize(trades),
                                       pd.Timestamp("2005-01-01", tz=UTC))
        table = run_backtest.format_table([row])
        self.assertIn("2005-01", table)
        self.assertIn("1/1", table)

    def test_an_instrument_with_no_trades_prints_placeholders_not_a_crash(self):
        from research import scoring
        empty = pd.DataFrame(columns=["outcome", "planned_rr", "r_gross", "r_net", "cost_r",
                                      "entry_time"])
        table = run_backtest.format_table([run_backtest.summary_row("XXX_YYY",
                                                                     scoring.summarize(empty))])
        self.assertIn("n/a", table)


if __name__ == "__main__":
    unittest.main()
