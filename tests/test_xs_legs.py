import unittest

import numpy as np
import pandas as pd

from research.xs import legs, universe
from research.xs.legs import CurrencyData, Tables

UTC = "UTC"
ENTRY = pd.Timestamp("2024-07-15 07:00", tz=UTC)            # a Monday
EXIT_4W = pd.Timestamp("2024-08-12 07:00", tz=UTC)          # four weeks later
EXIT_1W = pd.Timestamp("2024-07-22 07:00", tz=UTC)


def daily(n=260, step=0.001, start_price=1.00, half_range=0.005):
    """Daily candles opening 21:00 UTC each day, rising by `step`, with a constant range so the
    14-day ATR is exactly 2 x half_range."""
    close = start_price + step * np.arange(n)
    df = pd.DataFrame({"time": pd.date_range(pd.Timestamp("2024-01-01 21:00", tz=UTC), periods=n, freq="1D"),
                       "open": close, "high": close + half_range, "low": close - half_range,
                       "close": close, "volume": 1})
    return df


def h1(price=1.10, first="2024-06-01 00:00", periods=24 * 90, spread=0.0002, **overrides):
    """Hourly candles, flat at `price`. `overrides` maps a timestamp (string) to a dict of columns."""
    times = pd.date_range(pd.Timestamp(first, tz=UTC), periods=periods, freq="1h")
    df = pd.DataFrame({"time": times, "open": price, "high": price + 0.0001, "low": price - 0.0001,
                       "close": price, "volume": 1, "spread": spread})
    return df


def _utc(when):
    stamp = pd.Timestamp(when)
    return stamp if stamp.tzinfo is not None else stamp.tz_localize(UTC)


def set_candle(frame, when, **columns):
    row = frame.index[frame["time"] == _utc(when)][0]
    for name, value in columns.items():
        frame.loc[row, name] = value
    return frame


def columns(currency="EUR", formation=84, hold=4, daily_frame=None, h1_frame=None, first_entry=None,
            entries=(ENTRY,)):
    data = CurrencyData(currency, daily_frame if daily_frame is not None else daily(),
                        h1_frame if h1_frame is not None else h1(), first_entry)
    entry_ns = pd.DatetimeIndex(entries).to_numpy(dtype="datetime64[ns]").astype("int64")
    return legs.currency_columns(data, entry_ns, formation, hold)


class UniverseTests(unittest.TestCase):
    def test_sixteen_currencies_each_with_a_usd_pair(self):
        self.assertEqual(len(universe.CURRENCIES), 16)
        self.assertEqual(len(set(universe.PAIRS)), 16)

    def test_base_quoted_currencies_are_long_when_the_pair_is_long(self):
        for c in ("EUR", "GBP", "AUD", "NZD"):
            self.assertEqual((universe.pair_of(c), universe.orientation(c)), (f"{c}_USD", 1))

    def test_quote_quoted_currencies_are_long_when_the_pair_is_short(self):
        for c in ("JPY", "CAD", "CHF", "ZAR", "MXN"):
            self.assertEqual((universe.pair_of(c), universe.orientation(c)), (f"USD_{c}", -1))

    def test_the_excluded_currencies_are_not_in_the_universe(self):
        for c in ("HKD", "DKK", "CNH", "TRY", "USD"):
            self.assertNotIn(c, universe.CURRENCIES)
            with self.assertRaises(ValueError):
                universe.pair_of(c)

    def test_every_pair_is_in_the_instrument_catalog(self):
        from data_pipeline import instruments
        for pair in universe.PAIRS:
            self.assertIsNotNone(instruments.find(pair), pair)


class MondayEntryTests(unittest.TestCase):
    def test_entries_are_mondays_at_0700_utc(self):
        entries = legs.monday_entries(pd.Timestamp("2024-07-10", tz=UTC), pd.Timestamp("2024-08-06", tz=UTC))
        self.assertEqual([e.strftime("%a %Y-%m-%d %H:%M") for e in entries],
                         ["Mon 2024-07-15 07:00", "Mon 2024-07-22 07:00", "Mon 2024-07-29 07:00",
                          "Mon 2024-08-05 07:00"])


class ScoreTests(unittest.TestCase):
    def test_a_rising_base_quoted_currency_scores_positive_and_a_quote_quoted_one_negative(self):
        eur = columns("EUR")[0][0]
        jpy = columns("JPY")[0][0]
        self.assertGreater(eur, 0)
        self.assertAlmostEqual(jpy, -eur)                        # same pair moves, opposite currency

    def test_the_score_is_the_volatility_scaled_formation_return(self):
        # The last daily candle closed by the entry opened 2024-07-13 21:00 (row 194, close 1.194); the
        # start is the last one closed by 84 days earlier (row 110, close 1.110); ATR is exactly 0.01.
        expected = np.log(1.194 / 1.110) / (0.01 / 1.194)
        self.assertAlmostEqual(columns("EUR")[0][0], expected, places=9)

    def test_a_shorter_formation_window_uses_the_matching_daily_closes(self):
        expected = np.log(1.194 / 1.187) / (0.01 / 1.194)         # 7 days before: row 187
        self.assertAlmostEqual(columns("EUR", formation=7)[0][0], expected, places=9)

    def test_only_daily_candles_closed_before_the_entry_are_used(self):
        # Change the daily candle that is still open at the entry (row 195 opens 07-14 21:00, closes 07-15
        # 21:00): the score must not move.
        base = columns("EUR")[0][0]
        d = daily()
        d.loc[195, ["close", "high"]] = [5.0, 5.005]
        self.assertAlmostEqual(columns("EUR", daily_frame=d)[0][0], base, places=12)

    def test_too_little_daily_history_leaves_the_currency_ineligible(self):
        self.assertTrue(np.isnan(columns("EUR", daily_frame=daily(n=60))[0][0]))

    def test_a_hole_in_the_daily_data_around_the_formation_start_makes_it_ineligible(self):
        d = daily()
        d = d[~d["time"].between(pd.Timestamp("2024-04-10", tz=UTC), pd.Timestamp("2024-04-25", tz=UTC))]
        self.assertTrue(np.isnan(columns("EUR", daily_frame=d.reset_index(drop=True))[0][0]))

    def test_no_entry_candle_makes_it_ineligible(self):
        frame = h1()
        frame = frame[frame["time"] != ENTRY].reset_index(drop=True)
        self.assertTrue(np.isnan(columns("EUR", h1_frame=frame)[0][0]))

    def test_entries_before_the_dense_history_starts_are_ineligible(self):
        late = pd.Timestamp("2024-07-20", tz=UTC)
        self.assertTrue(np.isnan(columns("EUR", first_entry=late)[0][0]))
        self.assertFalse(np.isnan(columns("EUR", first_entry=pd.Timestamp("2024-07-01", tz=UTC))[0][0]))


class LegResultTests(unittest.TestCase):
    """R = signed price change / (2 x ATR) with ATR = 0.01, so one R is 0.02 and the stop is 0.05."""

    def frame_with_exit(self, exit_close, when=EXIT_4W):
        frame = h1()
        return set_candle(frame, when, close=exit_close, high=exit_close + 0.0001, low=exit_close - 0.0001,
                          open=exit_close)

    def test_a_rise_is_plus_two_r_for_a_long_and_minus_two_for_a_short(self):
        score, long_, short_, cost = columns("EUR", h1_frame=self.frame_with_exit(1.14))
        self.assertAlmostEqual(long_[0], 2.0)
        self.assertAlmostEqual(short_[0], -2.0)

    def test_for_a_quote_quoted_currency_the_legs_swap_because_long_the_currency_is_short_the_pair(self):
        _, long_, short_, _ = columns("JPY", h1_frame=self.frame_with_exit(1.14))
        self.assertAlmostEqual(long_[0], -2.0)
        self.assertAlmostEqual(short_[0], 2.0)

    def test_the_hold_length_sets_the_exit_candle(self):
        frame = self.frame_with_exit(1.12, when=EXIT_1W)
        _, long_1w, _, _ = columns("EUR", hold=1, h1_frame=frame)
        self.assertAlmostEqual(long_1w[0], 1.0)

    def test_a_stop_touched_on_the_way_fills_at_the_stop_and_only_hurts_that_direction(self):
        frame = self.frame_with_exit(1.14)
        set_candle(frame, "2024-07-20 12:00", low=1.04, open=1.09)              # touches 1.05
        _, long_, short_, _ = columns("EUR", h1_frame=frame)
        self.assertAlmostEqual(long_[0], (1.05 - 1.10) / 0.02)                  # -2.5R
        self.assertAlmostEqual(short_[0], -2.0)                                 # the short is unaffected

    def test_a_gap_through_the_stop_fills_at_the_worse_open(self):
        frame = self.frame_with_exit(1.14)
        set_candle(frame, "2024-07-20 12:00", low=1.02, open=1.03)
        _, long_, _, _ = columns("EUR", h1_frame=frame)
        self.assertAlmostEqual(long_[0], (1.03 - 1.10) / 0.02)                  # -3.5R

    def test_a_short_leg_is_stopped_above(self):
        frame = self.frame_with_exit(1.06)
        set_candle(frame, "2024-07-20 12:00", high=1.16, open=1.12)             # touches 1.15
        _, long_, short_, _ = columns("EUR", h1_frame=frame)
        self.assertAlmostEqual(short_[0], -(1.15 - 1.10) / 0.02)                # -2.5R
        self.assertAlmostEqual(long_[0], (1.06 - 1.10) / 0.02)

    def test_a_stop_touched_in_the_exit_candle_itself_counts(self):
        frame = self.frame_with_exit(1.14)
        set_candle(frame, EXIT_4W, low=1.04, open=1.11)
        _, long_, _, _ = columns("EUR", h1_frame=frame)
        self.assertAlmostEqual(long_[0], -2.5)

    def test_a_stop_touched_before_the_entry_or_after_the_exit_is_irrelevant(self):
        frame = self.frame_with_exit(1.14)
        set_candle(frame, "2024-07-14 12:00", low=1.00)
        set_candle(frame, "2024-08-12 08:00", low=1.00)
        _, long_, _, _ = columns("EUR", h1_frame=frame)
        self.assertAlmostEqual(long_[0], 2.0)

    def test_the_cost_is_one_spread_at_the_entry_candle_over_the_risk_unit(self):
        frame = self.frame_with_exit(1.14)
        set_candle(frame, ENTRY, spread=0.0004)
        set_candle(frame, EXIT_4W, spread=0.0100)                   # the exit spread is not charged
        cost = columns("EUR", h1_frame=frame)[3][0]
        self.assertAlmostEqual(cost, 0.0004 / 0.02)

    def test_a_missing_entry_spread_falls_back_to_the_typical_one(self):
        frame = self.frame_with_exit(1.14)
        set_candle(frame, ENTRY, spread=np.nan)
        self.assertAlmostEqual(columns("EUR", h1_frame=frame)[3][0], 0.0002 / 0.02)

    def test_no_exit_candle_leaves_the_score_but_no_result(self):
        frame = h1(periods=24 * 45)                                 # ends before the four-week exit
        score, long_, short_, cost = columns("EUR", h1_frame=frame)
        self.assertFalse(np.isnan(score[0]))
        self.assertTrue(np.isnan(long_[0]) and np.isnan(short_[0]))

    def test_the_exit_is_the_first_candle_at_or_after_the_target_within_three_days(self):
        frame = h1()
        near = frame[(frame["time"] < EXIT_4W) | (frame["time"] >= EXIT_4W + pd.Timedelta(hours=20))]
        near = set_candle(near.reset_index(drop=True), EXIT_4W + pd.Timedelta(hours=20), close=1.12,
                          high=1.1201, low=1.1199, open=1.12)
        self.assertAlmostEqual(columns("EUR", h1_frame=near)[1][0], 1.0)        # exits at +20h
        far = frame[(frame["time"] < EXIT_4W) | (frame["time"] >= EXIT_4W + pd.Timedelta(days=4))]
        self.assertTrue(np.isnan(columns("EUR", h1_frame=far.reset_index(drop=True))[1][0]))


class BuildTablesTests(unittest.TestCase):
    def data(self, n_currencies):
        names = universe.CURRENCIES[:n_currencies]
        return {c: CurrencyData(c, daily(), h1(periods=24 * 120)) for c in names}

    def test_a_cohort_needs_at_least_eight_eligible_currencies(self):
        entries = pd.DatetimeIndex([ENTRY])
        self.assertEqual(len(legs.build_tables(self.data(7), 84, 4, entries)), 0)
        tables = legs.build_tables(self.data(8), 84, 4, entries)
        self.assertEqual(len(tables), 1)
        self.assertEqual(int(tables.eligible.sum()), 8)

    def test_the_tables_are_cohorts_by_currencies(self):
        entries = pd.DatetimeIndex([ENTRY, ENTRY + pd.Timedelta(days=7)])
        tables = legs.build_tables(self.data(10), 84, 1, entries)
        self.assertEqual(tables.score.shape, (2, 10))
        self.assertEqual(tables.currencies, universe.CURRENCIES[:10])
        self.assertEqual((tables.hold_weeks, tables.formation_days), (1, 84))

    def test_cohorts_default_to_every_monday_the_data_covers(self):
        tables = legs.build_tables(self.data(9), 7, 1)
        self.assertTrue(all(t.dayofweek == 0 and t.hour == 7 for t in tables.cohorts))
        self.assertGreater(len(tables), 5)


if __name__ == "__main__":
    unittest.main()
