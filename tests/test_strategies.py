import unittest

import numpy as np
import pandas as pd

from research import indicators
from research.strategies import STRATEGIES, get_strategy
from research.strategies.common import entry_after, make_signal, value_before
from tests.market_helpers import daily_frame, h1_frame, synthetic_market
from tests.test_research import section81_frames

FIRST = "2024-03-01 21:00"          # opens of the hand-built daily candles: 21:00 UTC each day


def flat_days(n, high=1.10, low=1.09, close=1.095):
    return [(close, high, low, close)] * n


def signals_of(name, frames, **params):
    return get_strategy(name, **params).generate("EUR_USD", frames)


class EntryAfterTests(unittest.TestCase):
    def test_the_entry_is_the_first_0700_candle_at_or_after_the_signal_time(self):
        h1 = h1_frame("2024-04-01 12:00", 100)
        when = pd.Series([pd.Timestamp("2024-04-01 21:00", tz="UTC")])
        (row,) = entry_after(h1, when)
        self.assertEqual(h1["time"].iloc[row], pd.Timestamp("2024-04-02 07:00", tz="UTC"))

    def test_a_signal_exactly_at_0700_can_use_that_candle(self):
        h1 = h1_frame("2024-04-01 00:00", 60)
        (row,) = entry_after(h1, pd.Series([pd.Timestamp("2024-04-02 07:00", tz="UTC")]))
        self.assertEqual(h1["time"].iloc[row], pd.Timestamp("2024-04-02 07:00", tz="UTC"))

    def test_no_candle_within_the_wait_means_no_entry(self):
        h1 = h1_frame("2024-04-01 12:00", 5)               # the data ends before the next 07:00
        self.assertEqual(entry_after(h1, pd.Series([pd.Timestamp("2024-04-01 21:00", tz="UTC")]))[0], -1)

    def test_a_hole_in_the_data_longer_than_the_wait_means_no_entry(self):
        early = h1_frame("2024-04-01 12:00", 5)
        late = h1_frame("2024-04-20 05:00", 10)
        h1 = pd.concat([early, late], ignore_index=True)
        self.assertEqual(entry_after(h1, pd.Series([pd.Timestamp("2024-04-01 21:00", tz="UTC")]))[0], -1)

    def test_a_weekend_gap_is_waited_out(self):
        friday = h1_frame("2024-04-05 12:00", 10)          # Friday 12:00-21:00
        monday = h1_frame("2024-04-08 00:00", 30)
        h1 = pd.concat([friday, monday], ignore_index=True)
        (row,) = entry_after(h1, pd.Series([pd.Timestamp("2024-04-05 22:00", tz="UTC")]))
        self.assertEqual(h1["time"].iloc[row], pd.Timestamp("2024-04-08 07:00", tz="UTC"))


class ValueBeforeTests(unittest.TestCase):
    def test_a_value_is_available_only_once_its_daily_candle_has_closed(self):
        daily = daily_frame(flat_days(3))
        closes = daily["time"] + pd.Timedelta(hours=24)
        values = np.array([1.0, 2.0, 3.0])
        moments = np.array([pd.Timestamp("2024-03-02 20:59", tz="UTC").value,     # day 0 not closed yet
                            pd.Timestamp("2024-03-02 21:00", tz="UTC").value,     # day 0 just closed
                            pd.Timestamp("2024-03-04 10:00", tz="UTC").value])    # days 0 and 1 closed
        out = value_before(daily["time"], closes, values, moments)
        self.assertTrue(np.isnan(out[0]))
        self.assertEqual(list(out[1:]), [1.0, 2.0])


class MakeSignalTests(unittest.TestCase):
    def setUp(self):
        self.h1 = h1_frame("2024-04-01 00:00", 10, price=1.10)

    def test_a_valid_buy(self):
        s = make_signal("EUR_USD", "bullish", self.h1, 3, stop=1.09, target=1.12)
        self.assertEqual((s.entry_index, s.entry_price), (3, 1.10))
        self.assertEqual(s.entry_time, self.h1["time"].iloc[3] + pd.Timedelta(hours=1))

    def test_levels_on_the_wrong_side_give_no_signal_instead_of_an_error(self):
        self.assertIsNone(make_signal("EUR_USD", "bullish", self.h1, 3, stop=1.09, target=1.095))
        self.assertIsNone(make_signal("EUR_USD", "bearish", self.h1, 3, stop=1.09, target=1.08))
        self.assertIsNone(make_signal("EUR_USD", "bullish", self.h1, 3, stop=float("nan"), target=1.12))


class DonchianTests(unittest.TestCase):
    def frames(self, extra_rows, entry_price=1.12, h1_periods=200):
        rows = flat_days(30) + extra_rows
        daily = daily_frame(rows, FIRST)
        after = daily["time"].iloc[len(rows) - 1] + pd.Timedelta(hours=24)
        return {"D": daily, "H1": h1_frame(str(after), h1_periods, price=entry_price)}

    def test_a_breakout_above_the_channel_becomes_a_buy_with_atr_scaled_levels(self):
        frames = self.frames([(1.095, 1.125, 1.10, 1.12)])            # closes above the 1.10 channel top
        out = signals_of("donchian_trend", frames)
        (s,) = out.signals
        atr = (13 * 0.01 + 0.03) / 14                                  # hand-computed
        self.assertEqual(s.direction, "bullish")
        self.assertAlmostEqual(s.entry_price, 1.12)
        self.assertAlmostEqual(s.stop_price, 1.12 - 2 * atr)
        self.assertAlmostEqual(s.target_price, 1.12 + 4 * atr)
        self.assertAlmostEqual(s.planned_rr, 2.0)
        self.assertEqual(out.funnel["breakouts"], 1)

    def test_it_enters_at_the_next_london_open_after_the_daily_candle_closes(self):
        frames = self.frames([(1.095, 1.125, 1.10, 1.12)])
        (s,) = signals_of("donchian_trend", frames).signals
        entry_candle = frames["H1"]["time"].iloc[s.entry_index]
        self.assertEqual(entry_candle.hour, 7)
        daily_close = frames["D"]["time"].iloc[30] + pd.Timedelta(hours=24)
        self.assertGreaterEqual(entry_candle, daily_close)
        self.assertLess(entry_candle - daily_close, pd.Timedelta(hours=24))
        self.assertEqual(s.entry_time, entry_candle + pd.Timedelta(hours=1))

    def test_only_the_first_close_beyond_the_channel_is_a_signal(self):
        # Day 31 also closes beyond the (now higher) channel, but the breakout is not fresh.
        frames = self.frames([(1.095, 1.125, 1.10, 1.12), (1.12, 1.135, 1.12, 1.13)])
        self.assertEqual(len(signals_of("donchian_trend", frames).signals), 1)

    def test_a_new_breakout_after_falling_back_inside_the_channel_is_a_new_signal(self):
        frames = self.frames([(1.095, 1.125, 1.10, 1.12),              # breakout
                              (1.12, 1.12, 1.09, 1.095),               # back inside
                              (1.095, 1.135, 1.095, 1.13)])            # 1.13 clears the 1.125 channel top
        self.assertEqual(len(signals_of("donchian_trend", frames).signals), 2)

    def test_a_breakdown_below_the_channel_becomes_a_sell(self):
        frames = self.frames([(1.095, 1.09, 1.065, 1.07)], entry_price=1.07)
        (s,) = signals_of("donchian_trend", frames).signals
        self.assertEqual(s.direction, "bearish")
        self.assertLess(s.target_price, s.entry_price)
        self.assertGreater(s.stop_price, s.entry_price)
        self.assertAlmostEqual(s.planned_rr, 2.0)

    def test_a_close_that_only_reaches_the_channel_is_not_a_breakout(self):
        frames = self.frames([(1.095, 1.10, 1.09, 1.10)])              # equal to the top, not above
        self.assertEqual(signals_of("donchian_trend", frames).signals, [])

    def test_no_signal_until_a_full_channel_of_history_exists(self):
        daily = daily_frame(flat_days(10) + [(1.095, 1.125, 1.10, 1.12)], FIRST)
        after = daily["time"].iloc[-1] + pd.Timedelta(hours=24)
        out = signals_of("donchian_trend", {"D": daily, "H1": h1_frame(str(after), 100, price=1.12)})
        self.assertEqual(out.signals, [])

    def test_a_signal_with_no_candle_to_enter_on_is_counted_not_traded(self):
        out = signals_of("donchian_trend", self.frames([(1.095, 1.125, 1.10, 1.12)], h1_periods=5))
        self.assertEqual(out.signals, [])
        self.assertEqual(out.funnel["no_entry_candle"], 1)

    def test_it_declares_its_entry_candles_and_how_much_history_it_needs(self):
        strategy = get_strategy("donchian_trend")
        h1 = h1_frame("2024-04-01 00:00", 48)
        np.testing.assert_array_equal(np.flatnonzero(strategy.entry_mask(h1)), [7, 31])
        self.assertGreaterEqual(strategy.daily_warmup_days, 20 + 14)


class DailyReversionTests(unittest.TestCase):
    def rising(self, n=260, start=1.00, step=0.0005):
        return [(start + step * i,) * 4 for i in range(n)]

    def frames(self, rows, entry_price):
        daily = daily_frame([(o, max(o, c) + 0.001, min(o, c) - 0.001, c) for o, _, _, c in rows],
                            FIRST)
        after = daily["time"].iloc[-1] + pd.Timedelta(hours=24)
        return {"D": daily, "H1": h1_frame(str(after), 200, price=entry_price)}

    def test_a_dip_below_the_band_in_an_uptrend_is_a_buy_aiming_at_the_average(self):
        rows = self.rising() + [(1.13, 1.13, 1.10, 1.10)]
        frames = self.frames(rows, entry_price=1.10)
        (s,) = signals_of("daily_reversion", frames).signals
        close, high, low = frames["D"]["close"], frames["D"]["high"], frames["D"]["low"]
        centre = indicators.sma(close, 20).iloc[-1]
        atr = indicators.atr(high, low, close, 14).iloc[-1]
        self.assertEqual(s.direction, "bullish")
        self.assertAlmostEqual(s.target_price, centre)
        self.assertAlmostEqual(s.stop_price, 1.10 - 2 * atr)
        self.assertGreater(s.target_price, s.entry_price)

    def test_a_dip_in_a_downtrend_is_ignored_by_the_trend_filter(self):
        falling = [(1.20 - 0.0005 * i,) * 4 for i in range(260)]
        rows = falling + [(1.07, 1.07, 1.03, 1.03)]
        self.assertEqual(signals_of("daily_reversion", self.frames(rows, 1.03)).signals, [])

    def test_a_spike_above_the_band_in_a_downtrend_is_a_sell(self):
        falling = [(1.20 - 0.0005 * i,) * 4 for i in range(260)]
        rows = falling + [(1.07, 1.10, 1.07, 1.10)]
        (s,) = signals_of("daily_reversion", self.frames(rows, entry_price=1.10)).signals
        self.assertEqual(s.direction, "bearish")
        self.assertLess(s.target_price, s.entry_price)

    def test_a_signal_whose_target_is_already_reached_by_the_entry_is_skipped(self):
        rows = self.rising() + [(1.13, 1.13, 1.10, 1.10)]
        out = signals_of("daily_reversion", self.frames(rows, entry_price=1.20))
        self.assertEqual(out.signals, [])
        self.assertEqual(out.funnel["target_already_reached"], 1)

    def test_only_the_first_close_below_the_band_counts(self):
        rows = self.rising() + [(1.13, 1.13, 1.10, 1.10), (1.10, 1.10, 1.09, 1.09)]
        self.assertLessEqual(len(signals_of("daily_reversion", self.frames(rows, 1.09)).signals), 1)

    def test_without_200_days_of_history_there_is_no_trend_and_no_signal(self):
        rows = self.rising(n=100) + [(1.05, 1.05, 1.02, 1.02)]
        self.assertEqual(signals_of("daily_reversion", self.frames(rows, 1.02)).signals, [])

    def test_it_asks_for_enough_daily_history_and_enters_at_0700(self):
        strategy = get_strategy("daily_reversion")
        self.assertGreater(strategy.daily_warmup_days, 200)
        self.assertEqual(int(strategy.entry_mask(h1_frame("2024-04-01 00:00", 24)).sum()), 1)


class RangeBreakoutTests(unittest.TestCase):
    DAY = "2024-04-16 00:00"                   # a Tuesday

    def frames(self, atr=0.006, closes=None, highs=None, lows=None, drop_hours=()):
        """Thirty flat daily candles of range `atr`, then one day of H1 candles."""
        first = (pd.Timestamp(self.DAY, tz="UTC") - pd.Timedelta(days=33)).replace(hour=21)
        rows = [(1.10, 1.10 + atr / 2, 1.10 - atr / 2, 1.10)] * 30
        daily = daily_frame(rows, str(first))
        close = np.full(24, 1.1000)
        high, low = np.full(24, 1.1002), np.full(24, 1.0998)
        high[3], low[5] = 1.1010, 1.0990                       # the Asian range: 1.0990 - 1.1010
        for hour, value in (closes or {}).items():
            close[hour] = value
        for hour, value in (highs or {}).items():
            high[hour] = value
        for hour, value in (lows or {}).items():
            low[hour] = value
        h1 = h1_frame(self.DAY, 24, open=close, high=np.maximum(high, close),
                      low=np.minimum(low, close), close=close)
        h1 = h1[~h1["time"].dt.hour.isin(drop_hours)].reset_index(drop=True)
        return {"D": daily, "H1": h1}

    def test_a_close_above_the_range_in_the_london_morning_is_a_buy(self):
        out = signals_of("range_breakout", self.frames(closes={7: 1.1020}))
        (s,) = out.signals
        self.assertEqual(s.direction, "bullish")
        self.assertAlmostEqual(s.entry_price, 1.1020)
        self.assertAlmostEqual(s.stop_price, 1.0990)                 # the bottom of the range
        self.assertAlmostEqual(s.target_price, 1.1020 + 2 * 0.0030)
        self.assertAlmostEqual(s.planned_rr, 2.0)

    def test_a_close_below_the_range_is_a_sell_stopped_at_the_top(self):
        (s,) = signals_of("range_breakout", self.frames(closes={8: 1.0975})).signals
        self.assertEqual(s.direction, "bearish")
        self.assertAlmostEqual(s.stop_price, 1.1010)
        self.assertAlmostEqual(s.entry_price, 1.0975)
        self.assertAlmostEqual(s.target_price, 1.0975 - 2 * (1.1010 - 1.0975))

    def test_only_the_first_breakout_of_the_day_is_traded(self):
        out = signals_of("range_breakout", self.frames(closes={7: 1.1020, 9: 1.0970}))
        self.assertEqual(len(out.signals), 1)
        self.assertEqual(out.signals[0].direction, "bullish")

    def test_the_entry_time_is_the_breakout_candles_close(self):
        frames = self.frames(closes={8: 1.1020})
        (s,) = signals_of("range_breakout", frames).signals
        self.assertEqual(s.entry_time, pd.Timestamp(self.DAY, tz="UTC") + pd.Timedelta(hours=9))

    def test_a_breakout_after_the_morning_window_is_not_traded(self):
        out = signals_of("range_breakout", self.frames(closes={11: 1.1020}))
        self.assertEqual(out.signals, [])
        self.assertEqual(out.funnel["no_breakout"], 1)

    def test_a_day_that_stays_inside_the_range_has_no_trade(self):
        self.assertEqual(signals_of("range_breakout", self.frames()).signals, [])

    def test_a_range_too_wide_for_the_daily_volatility_is_skipped(self):
        out = signals_of("range_breakout", self.frames(atr=0.001, closes={7: 1.1020}))
        self.assertEqual(out.signals, [])
        self.assertEqual(out.funnel["risk_out_of_bounds"], 1)

    def test_a_range_too_tight_for_the_daily_volatility_is_skipped(self):
        out = signals_of("range_breakout", self.frames(atr=0.05, closes={7: 1.1020}))
        self.assertEqual(out.signals, [])
        self.assertEqual(out.funnel["risk_out_of_bounds"], 1)

    def test_a_day_with_too_few_range_candles_is_skipped(self):
        out = signals_of("range_breakout", self.frames(closes={7: 1.1020}, drop_hours=(0, 1)))
        self.assertEqual(out.signals, [])
        self.assertEqual(out.funnel["no_range"], 1)

    def test_it_declares_the_morning_candles_as_its_entry_candles(self):
        mask = get_strategy("range_breakout").entry_mask(h1_frame("2024-04-16 00:00", 24))
        np.testing.assert_array_equal(np.flatnonzero(mask), [7, 8, 9, 10])


class RandomControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.market = synthetic_market(weeks=200, seed=1)

    def test_about_one_day_in_ten_is_traded_in_both_directions(self):
        out = signals_of("random_control", self.market)
        self.assertTrue(70 <= len(out.signals) <= 140, len(out.signals))
        bullish = sum(1 for s in out.signals if s.direction == "bullish")
        self.assertTrue(0.3 < bullish / len(out.signals) < 0.7)

    def test_entries_are_only_at_the_hours_it_declares(self):
        out = signals_of("random_control", self.market)
        hours = {self.market["H1"]["time"].iloc[s.entry_index].hour for s in out.signals}
        self.assertTrue(hours <= set(range(7, 21)))
        self.assertGreater(len(hours), 6)

    def test_its_stops_are_two_atr_and_its_targets_two_r(self):
        out = signals_of("random_control", self.market)
        for s in out.signals[:20]:
            self.assertAlmostEqual(s.planned_rr, 2.0)
            self.assertAlmostEqual(s.risk, 2 * s.meta["atr"])

    def test_it_is_reproducible_and_depends_on_the_seed_and_the_instrument(self):
        a = signals_of("random_control", self.market)
        b = signals_of("random_control", self.market)
        self.assertEqual([s.entry_index for s in a.signals], [s.entry_index for s in b.signals])
        other_seed = signals_of("random_control", self.market, seed=1)
        self.assertNotEqual([s.entry_index for s in a.signals],
                            [s.entry_index for s in other_seed.signals])
        other_instrument = get_strategy("random_control").generate("GBP_USD", self.market)
        self.assertNotEqual([s.entry_index for s in a.signals],
                            [s.entry_index for s in other_instrument.signals])


class SweepFvgVariantTests(unittest.TestCase):
    def test_the_default_strategy_is_unchanged(self):
        out = signals_of("sweep_fvg", section81_frames())
        self.assertEqual(len(out.signals), 1)
        self.assertEqual(out.funnel, {"win": 1})

    def test_skipping_the_hour_of_the_entry_drops_the_signal_and_counts_it(self):
        out = signals_of("sweep_fvg", section81_frames(), skip_open_hours=(7,))     # it enters at 07:00
        self.assertEqual(out.signals, [])
        self.assertEqual(out.funnel, {"skipped_entry_hour": 1})

    def test_skipping_some_other_hour_changes_nothing(self):
        self.assertEqual(len(signals_of("sweep_fvg", section81_frames(),
                                        skip_open_hours=(21,)).signals), 1)

    def test_the_no_rollover_variant_excludes_the_2100_candle_from_its_entry_candles(self):
        h1 = h1_frame("2024-04-16 00:00", 24)
        plain = get_strategy("sweep_fvg").entry_mask(h1)
        fixed = get_strategy("sweep_fvg_no_rollover").entry_mask(h1)
        self.assertTrue(plain[21] and not fixed[21])
        np.testing.assert_array_equal(np.flatnonzero(plain & ~fixed), [21])

    def test_the_variant_is_registered_under_its_own_name(self):
        self.assertEqual(get_strategy("sweep_fvg_no_rollover").name, "sweep_fvg_no_rollover")


class RegistryTests(unittest.TestCase):
    def test_every_strategy_declares_what_the_engine_needs(self):
        for name in STRATEGIES:
            strategy = get_strategy(name)
            self.assertEqual(strategy.name, name)
            self.assertIn("H1", strategy.timeframes, name)
            self.assertTrue(hasattr(strategy, "entry_mask"), name)

    def test_the_preregistered_strategies_are_present_with_their_fixed_parameters(self):
        donchian = get_strategy("donchian_trend")
        self.assertEqual((donchian.channel, donchian.atr_length, donchian.stop_atr, donchian.reward_r),
                         (20, 14, 2.0, 2.0))
        reversion = get_strategy("daily_reversion")
        self.assertEqual((reversion.band_length, reversion.band_k, reversion.trend_length,
                          reversion.stop_atr), (20, 2.0, 200, 2.0))
        breakout = get_strategy("range_breakout")
        self.assertEqual((breakout.range_last_hour, breakout.entry_first_hour,
                          breakout.entry_last_hour, breakout.reward_r, breakout.min_range_atr,
                          breakout.max_range_atr), (6, 7, 10, 2.0, 0.25, 1.5))
        control = get_strategy("random_control")
        self.assertEqual((control.every_days, control.stop_atr, control.reward_r,
                          control.first_hour, control.last_hour), (10, 2.0, 2.0, 7, 20))


if __name__ == "__main__":
    unittest.main()
