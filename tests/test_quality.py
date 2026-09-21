import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import config, quality, storage

NY = "America/New_York"


def utc(text: str, tz: str = "UTC") -> pd.Timestamp:
    return pd.Timestamp(text, tz=tz).tz_convert("UTC")


def trading_week(start_local: str, end_local: str, freq: str = "1h") -> pd.DatetimeIndex:
    """Candle open times from a Sunday 17:00 New York open to a Friday close."""
    return pd.date_range(pd.Timestamp(start_local, tz=NY), pd.Timestamp(end_local, tz=NY),
                         freq=freq).tz_convert("UTC")


SUMMER_WEEK = ("2024-07-07 17:00", "2024-07-12 16:00")     # H1 slots, Sun open .. Fri last hour
WINTER_WEEK = ("2024-01-07 17:00", "2024-01-12 16:00")


def frame(times, spread=0.0001) -> pd.DataFrame:
    n = len(times)
    df = pd.DataFrame({"time": pd.Series(times), "open": 1.1000, "high": 1.1005,
                       "low": 1.0995, "close": 1.1002, "volume": 10})
    if spread is not None:
        df["spread"] = spread
    assert len(df) == n
    return df


class MarketHoursTests(unittest.TestCase):
    def test_the_trading_week_runs_sunday_1700_to_friday_1700_new_york(self):
        cases = [
            ("2024-07-06 12:00", False),   # Saturday
            ("2024-07-07 16:00", False),   # Sunday, before the open
            ("2024-07-07 17:00", True),    # Sunday open
            ("2024-07-10 03:00", True),
            ("2024-07-12 16:00", True),    # Friday's last hour
            ("2024-07-12 17:00", False),   # Friday close
        ]
        for local, expected in cases:
            self.assertEqual(bool(quality.market_open([pd.Timestamp(local, tz=NY)])[0]),
                             expected, local)


class GridTests(unittest.TestCase):
    """OANDA aligns Daily and H4 to 17:00 New York, so the UTC grid moves with DST."""

    def test_daily_opens_at_2100_utc_in_summer_and_2200_in_winter(self):
        self.assertTrue(quality.on_grid([utc("2024-07-08 21:00")], "D")[0])
        self.assertTrue(quality.on_grid([utc("2024-01-08 22:00")], "D")[0])

    def test_a_daily_candle_on_the_other_seasons_hour_is_off_grid(self):
        self.assertFalse(quality.on_grid([utc("2024-01-08 21:00")], "D")[0])
        self.assertFalse(quality.on_grid([utc("2024-07-08 22:00")], "D")[0])

    def test_h4_grid_is_01_05_09_13_17_21_in_summer(self):
        times = [utc(f"2024-07-08 {h:02d}:00") for h in (1, 5, 9, 13, 17, 21)]
        self.assertTrue(quality.on_grid(times, "H4").all())

    def test_h4_grid_is_02_06_10_14_18_22_in_winter(self):
        times = [utc(f"2024-01-08 {h:02d}:00") for h in (2, 6, 10, 14, 18, 22)]
        self.assertTrue(quality.on_grid(times, "H4").all())
        self.assertFalse(quality.on_grid([utc("2024-01-08 01:00")], "H4")[0])

    def test_h1_must_start_on_the_hour(self):
        self.assertTrue(quality.on_grid([utc("2024-07-08 09:00")], "H1")[0])
        self.assertFalse(quality.on_grid([utc("2024-07-08 09:30")], "H1")[0])


class MissingCandleTests(unittest.TestCase):
    def test_a_complete_week_and_the_weekend_after_it_are_not_holes(self):
        two_weeks = trading_week("2024-07-07 17:00", "2024-07-19 16:00")
        two_weeks = two_weeks[quality.market_open(two_weeks)]
        self.assertEqual(quality.missing_slots(pd.Series(two_weeks), "H1"), (0, 0, 0))

    def test_a_missing_hour_inside_the_week_is_counted(self):
        times = trading_week(*SUMMER_WEEK)
        without = pd.Series(times.delete(30))
        self.assertEqual(quality.missing_slots(without, "H1"), (1, 1, 1))

    def test_a_run_of_missing_hours_is_one_gap_with_its_length(self):
        times = trading_week(*SUMMER_WEEK)
        without = pd.Series(times.delete([30, 31, 32]))
        self.assertEqual(quality.missing_slots(without, "H1"), (3, 1, 3))

    def test_missing_h4_candles_are_found_on_the_h4_grid(self):
        times = trading_week("2024-07-07 17:00", "2024-07-12 13:00", freq="4h")
        self.assertEqual(quality.missing_slots(pd.Series(times.delete(5)), "H4"), (1, 1, 1))

    def test_daily_candles_across_a_weekend_have_no_holes(self):
        days = [utc(f"2024-07-{d:02d} 17:00", NY) for d in (7, 8, 9, 10, 11, 14, 15)]
        self.assertEqual(quality.missing_slots(pd.Series(days), "D"), (0, 0, 0))

    def test_a_missing_weekday_daily_candle_is_counted(self):
        days = [utc(f"2024-07-{d:02d} 17:00", NY) for d in (7, 8, 10, 11)]     # no Tuesday
        self.assertEqual(quality.missing_slots(pd.Series(days), "D"), (1, 1, 1))

    def test_too_little_data_to_have_gaps(self):
        self.assertEqual(quality.missing_slots(pd.Series([utc("2024-07-08 09:00")]), "H1"),
                         (0, 0, 0))


class HardErrorTests(unittest.TestCase):
    def setUp(self):
        self.times = trading_week(*SUMMER_WEEK)

    def check(self, df):
        return quality.check_frame(df, "H1", "EUR_USD")

    def test_a_clean_week_has_no_issues_at_all(self):
        row = self.check(frame(self.times))
        self.assertEqual(row["hard_errors"], 0)
        self.assertEqual(row["missing_candles"], 0)
        self.assertEqual(row["issues"], "")
        self.assertEqual(row["rows"], 120)

    def test_duplicate_timestamps(self):
        df = frame(self.times)
        df = pd.concat([df, df.iloc[[5]]], ignore_index=True)
        self.assertEqual(self.check(df)["duplicate_times"], 1)

    def test_unsorted_timestamps(self):
        df = frame(self.times).iloc[::-1].reset_index(drop=True)
        self.assertGreater(self.check(df)["unsorted"], 0)

    def test_nan_and_non_positive_prices(self):
        df = frame(self.times)
        df.loc[3, "close"] = np.nan
        df.loc[4, "low"] = 0.0
        row = self.check(df)
        self.assertEqual(row["nan_prices"], 1)
        self.assertEqual(row["nonpositive_prices"], 1)

    def test_an_impossible_candle(self):
        df = frame(self.times)
        df.loc[2, "high"] = 1.0990                # below the low
        df.loc[3, "close"] = 1.2000               # above the high
        row = self.check(df)
        self.assertEqual(row["ohlc_invalid"], 2)
        self.assertIn("ohlc_invalid", row["issues"])

    def test_a_candle_off_the_time_grid(self):
        df = frame(self.times)
        df.loc[7, "time"] = df.loc[7, "time"] + pd.Timedelta(minutes=30)
        row = self.check(df)
        self.assertEqual(row["off_grid"], 1)
        self.assertEqual(row["hard_errors"], 1)

    def test_a_winter_daily_series_on_the_winter_grid_is_clean(self):
        days = [utc(f"2024-01-{d:02d} 17:00", NY) for d in (7, 8, 9, 10, 11)]
        self.assertEqual(quality.check_frame(frame(days), "D", "EUR_USD")["off_grid"], 0)

    def test_an_empty_frame_is_reported_not_crashed_on(self):
        row = quality.check_frame(storage.load("EUR_USD", "H1").iloc[0:0], "H1", "EUR_USD")
        self.assertEqual(row["rows"], 0)
        self.assertEqual(row["issues"], "no data")


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.times = trading_week(*SUMMER_WEEK)

    def test_missing_candles_make_the_series_sparse_above_the_threshold(self):
        keep = np.ones(len(self.times), dtype=bool)
        keep[10:30] = False                        # 20 of 120 gone
        row = quality.check_frame(frame(self.times[keep]), "H1", "EUR_USD")
        self.assertEqual(row["missing_candles"], 20)
        self.assertGreater(row["missing_pct"], quality.SPARSE_MISSING_PCT)
        self.assertIn("sparse", row["issues"])

    def test_spread_is_reported_in_pips(self):
        df = frame(self.times, spread=0.00012)
        row = quality.check_frame(df, "H1", "EUR_USD")
        self.assertEqual(row["spread_median_pips"], 1.2)
        self.assertEqual(row["spread_missing_pct"], 0.0)

    def test_a_jpy_spread_uses_its_own_pip_size(self):
        row = quality.check_frame(frame(self.times, spread=0.012), "H1", "USD_JPY")
        self.assertEqual(row["spread_median_pips"], 1.2)

    def test_a_series_without_spread_data_is_flagged(self):
        row = quality.check_frame(frame(self.times, spread=None), "H1", "EUR_USD")
        self.assertEqual(row["spread_missing_pct"], 100.0)
        self.assertIn("no_spread", row["issues"])

    def test_a_negative_spread_is_flagged(self):
        df = frame(self.times)
        df.loc[4, "spread"] = -0.0001
        row = quality.check_frame(df, "H1", "EUR_USD")
        self.assertEqual(row["spread_negative"], 1)
        self.assertIn("negative_spread", row["issues"])

    def test_a_spike_candle_is_an_outlier(self):
        df = frame(self.times)
        df.loc[100, "high"] = 1.2000               # a range hundreds of times normal
        self.assertEqual(quality.check_frame(df, "H1", "EUR_USD")["outlier_bars"], 1)


class H1VersusH4Tests(unittest.TestCase):
    def setUp(self):
        # Two H4 candles' worth of H1, from a Sunday 17:00 New York open.
        self.h1_times = pd.date_range(utc("2024-07-07 17:00", NY), periods=8, freq="1h")
        base = 1.1 + np.arange(8) * 1e-4
        self.h1 = pd.DataFrame({"time": self.h1_times, "open": base, "close": base + 5e-5,
                                "high": base + 1.5e-4, "low": base - 1e-4})
        rows = []
        for start in (0, 4):
            block = self.h1.iloc[start:start + 4]
            rows.append({"time": block["time"].iloc[0], "open": block["open"].iloc[0],
                         "high": block["high"].max(), "low": block["low"].min(),
                         "close": block["close"].iloc[-1]})
        self.h4 = pd.DataFrame(rows)

    def test_h4_candles_built_from_the_h1_candles_agree(self):
        self.assertEqual(quality.h1_vs_h4_mismatch(self.h1, self.h4), (2, 0))

    def test_a_wrong_h4_high_is_caught(self):
        self.h4.loc[1, "high"] += 0.001
        self.assertEqual(quality.h1_vs_h4_mismatch(self.h1, self.h4), (2, 1))

    def test_an_h4_candle_with_a_missing_h1_candle_is_not_compared(self):
        h1 = self.h1.drop(index=2)
        self.assertEqual(quality.h1_vs_h4_mismatch(h1, self.h4), (1, 0))

    def test_agreement_holds_on_the_winter_grid_too(self):
        winter = pd.date_range(utc("2024-01-07 17:00", NY), periods=8, freq="1h")
        h1 = self.h1.assign(time=winter)
        h4 = self.h4.assign(time=[winter[0], winter[4]])
        self.assertEqual(quality.h1_vs_h4_mismatch(h1, h4), (2, 0))

    def test_check_frame_reports_the_agreement(self):
        row = quality.check_frame(self.h4.assign(volume=1, spread=0.0001), "H4", "EUR_USD",
                                  h1=self.h1)
        self.assertEqual((row["h4_compared"], row["h4_mismatch_pct"]), (2, 0.0))


class RunTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._orig = config.DATA_DIR
        config.DATA_DIR = self._tmp
        self.addCleanup(lambda: (setattr(config, "DATA_DIR", self._orig),
                                 shutil.rmtree(self._tmp, ignore_errors=True)))

    def store(self, pair, tf, df):
        storage.save_merged(pair, tf, df.to_dict("records"))

    def test_run_and_summary_over_a_stored_pair(self):
        times = trading_week(*SUMMER_WEEK)
        self.store("EUR_USD", "H1", frame(times))
        report = quality.run(["EUR_USD"], ["H1"])
        self.assertEqual(len(report), 1)
        summary = quality.summarise(report)
        self.assertIn("HARD ERRORS: 0 series (none)", summary)

    def test_summary_lists_the_series_with_hard_errors(self):
        times = trading_week(*SUMMER_WEEK)
        bad = frame(times)
        bad.loc[2, "high"] = 1.0
        self.store("EUR_USD", "H1", bad)
        summary = quality.summarise(quality.run(["EUR_USD"], ["H1"]))
        self.assertIn("EUR_USD H1: ohlc_invalid=1", summary)

    def test_a_pair_with_no_data_is_named(self):
        summary = quality.summarise(quality.run(["GBP_USD"], ["H1"]))
        self.assertIn("NO DATA: GBP_USD", summary)


def weeks(first_sunday: str, count: int, keep_every: int = 1) -> pd.DatetimeIndex:
    """H1 candle open times for `count` consecutive trading weeks (120 candles each,
    Sunday 17:00 New York onward), optionally thinned to every `keep_every`th candle."""
    start = pd.Timestamp(first_sunday + " 17:00", tz=NY)
    pieces = []
    for k in range(count):
        week = pd.date_range(start + pd.DateOffset(weeks=k), periods=120, freq="1h")
        pieces.append(week[::keep_every])
    return pieces[0].append(pieces[1:]).tz_convert("UTC")


def stitched(*parts) -> pd.Series:
    return pd.Series(parts[0].append(parts[1:]))


class MonthlyDensityTests(unittest.TestCase):
    def test_a_complete_month_has_no_missing_share_and_a_thin_one_has_a_large_one(self):
        # Four full weeks starting Jan 7 carry January and Feb 1-2; four half-empty weeks follow.
        times = stitched(weeks("2024-01-07", 4), weeks("2024-02-04", 4, keep_every=2))
        share = quality.missing_share_by_month(times, "H1")
        self.assertEqual(share[pd.Period("2024-01")], 0.0)
        self.assertGreater(share[pd.Period("2024-02")], 0.3)
        self.assertLess(share[pd.Period("2024-02")], 0.6)

    def test_a_series_that_is_dense_throughout_is_usable_from_its_first_candle(self):
        times = weeks("2024-01-07", 12)
        self.assertEqual(quality.dense_from(pd.Series(times), "H1"), times[0])

    def test_a_sparse_start_pushes_the_usable_start_to_the_first_dense_month(self):
        # Sparse through February 23, then complete weeks from February 25.
        times = stitched(weeks("2024-01-07", 7, keep_every=8), weeks("2024-02-25", 10))
        self.assertEqual(quality.dense_from(times, "H1"), pd.Timestamp("2024-03-01", tz="UTC"))

    def test_a_sparse_patch_in_the_middle_pushes_the_start_past_it(self):
        times = stitched(weeks("2024-01-07", 4), weeks("2024-02-04", 4, keep_every=8),
                         weeks("2024-03-03", 8))
        self.assertEqual(quality.dense_from(times, "H1"), pd.Timestamp("2024-03-01", tz="UTC"))

    def test_a_series_still_sparse_in_its_last_month_has_no_reliable_data(self):
        times = stitched(weeks("2024-01-07", 8), weeks("2024-03-03", 4, keep_every=8))
        self.assertIsNone(quality.dense_from(times, "H1"))

    def test_no_data_has_no_reliable_start(self):
        self.assertIsNone(quality.dense_from(pd.Series([], dtype="datetime64[ns, UTC]"), "H1"))

    def test_the_sparseness_threshold_is_a_parameter(self):
        # February is half empty and is the last month: unusable when strict, fine when lax.
        times = stitched(weeks("2024-01-07", 4), weeks("2024-02-04", 4, keep_every=2))
        self.assertIsNone(quality.dense_from(times, "H1", sparse_share=0.10))
        self.assertEqual(quality.dense_from(times, "H1", sparse_share=0.9), times.iloc[0])

    def test_check_frame_reports_where_the_data_becomes_dense(self):
        times = stitched(weeks("2024-01-07", 7, keep_every=8), weeks("2024-02-25", 10))
        row = quality.check_frame(frame(times), "H1", "EUR_USD")
        self.assertEqual(row["dense_from"], "2024-03-01")


class UsableFromTests(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp())
        self._orig = config.DATA_DIR
        config.DATA_DIR = self._tmp
        self.addCleanup(lambda: (setattr(config, "DATA_DIR", self._orig),
                                 shutil.rmtree(self._tmp, ignore_errors=True)))

    def store(self, tf, times):
        storage.save_merged("EUR_USD", tf, frame(pd.Series(times)).to_dict("records"))

    def test_it_is_the_latest_of_the_timeframes_starts(self):
        dense = weeks("2024-01-07", 12)
        self.store("H1", dense)
        self.store("H4", dense[::4])
        # Daily: complete Sunday-to-Thursday candles, but starting a month later than the others.
        days = pd.date_range("2024-02-04 17:00", "2024-03-28 17:00", freq="1D", tz=NY)
        days = days[quality.market_open(days)].tz_convert("UTC")
        self.store("D", days)
        self.assertEqual(quality.usable_from("EUR_USD"), days[0])

    def test_a_missing_timeframe_means_nothing_is_usable(self):
        self.store("H1", weeks("2024-01-07", 4))
        self.assertIsNone(quality.usable_from("EUR_USD"))


if __name__ == "__main__":
    unittest.main()
