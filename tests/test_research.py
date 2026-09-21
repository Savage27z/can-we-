import math
import unittest

import numpy as np
import pandas as pd

from research import costs, engine, exits, scoring
from research.costs import MissingSpreadData, NoCost, SpreadCost
from research.exits import Candles, Exit
from research.strategies import STRATEGIES, get_strategy
from research.strategy import Signal
from tests.test_worked_examples import Section81BullishSignalTest

T0 = pd.Timestamp("2024-01-01 07:00", tz="UTC")


def frame_of(rows, spreads=None):
    """H1 candles from (open, high, low, close) tuples; row 0 is the entry candle."""
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df.insert(0, "time", pd.date_range(T0, periods=len(rows), freq="1h"))
    df["volume"] = 1
    if spreads is not None:
        df["spread"] = spreads
    return df


def buy(**over):
    base = dict(instrument="EUR_USD", direction="bullish", entry_index=0,
                entry_time=T0 + pd.Timedelta(hours=1), entry_price=1.1000, stop_price=1.0950,
                target_price=1.1150, invalidation_price=1.0970)
    return Signal(**{**base, **over})


def sell(**over):
    base = dict(instrument="EUR_USD", direction="bearish", entry_index=0,
                entry_time=T0 + pd.Timedelta(hours=1), entry_price=1.1000, stop_price=1.1050,
                target_price=1.0850, invalidation_price=1.1030)
    return Signal(**{**base, **over})


ENTRY = (1.1000, 1.1000, 1.1000, 1.1000)


def resolve(policy, signal, *rows):
    return policy.resolve(signal, Candles.from_frame(frame_of([ENTRY, *rows])))


class SignalTests(unittest.TestCase):
    def test_risk_and_planned_rr(self):
        s = buy()
        self.assertAlmostEqual(s.risk, 0.005)
        self.assertAlmostEqual(s.planned_rr, 3.0)

    def test_a_buy_needs_stop_below_entry_below_target(self):
        for bad in (dict(stop_price=1.1010), dict(target_price=1.0990)):
            with self.assertRaises(ValueError):
                buy(**bad)

    def test_a_sell_needs_target_below_entry_below_stop(self):
        for bad in (dict(stop_price=1.0990), dict(target_price=1.1010)):
            with self.assertRaises(ValueError):
                sell(**bad)

    def test_an_unknown_direction_is_rejected(self):
        with self.assertRaises(ValueError):
            buy(direction="long")


class CloseExitTests(unittest.TestCase):
    policy = exits.CloseExit()

    def test_a_close_at_or_past_the_target_wins_and_is_scored_at_the_target(self):
        got = resolve(self.policy, buy(), (1.10, 1.12, 1.099, 1.116))
        self.assertEqual(got, Exit(1, 1.1150, "target"))

    def test_a_close_beyond_the_invalidation_level_loses_and_is_scored_at_the_stop(self):
        got = resolve(self.policy, buy(), (1.10, 1.10, 1.095, 1.096))
        self.assertEqual(got, Exit(1, 1.0950, "invalidation"))

    def test_wicks_do_not_end_a_close_based_trade(self):
        got = resolve(self.policy, buy(), (1.10, 1.120, 1.090, 1.099))    # pierces both, closes between
        self.assertEqual(got, exits.OPEN)

    def test_the_invalidation_level_defaults_to_the_stop_when_the_signal_has_none(self):
        signal = buy(invalidation_price=None)
        self.assertEqual(resolve(self.policy, signal, (1.10, 1.10, 1.094, 1.096)), exits.OPEN)
        self.assertEqual(resolve(self.policy, signal, (1.10, 1.10, 1.094, 1.094)).reason,
                         "invalidation")

    def test_sells_mirror_buys(self):
        self.assertEqual(resolve(self.policy, sell(), (1.10, 1.101, 1.08, 1.084)),
                         Exit(1, 1.0850, "target"))
        self.assertEqual(resolve(self.policy, sell(), (1.10, 1.105, 1.10, 1.104)),
                         Exit(1, 1.1050, "invalidation"))

    def test_only_candles_after_the_entry_candle_count(self):
        # The entry candle itself already closed at the entry price.
        got = self.policy.resolve(buy(), Candles.from_frame(frame_of([(1.10, 1.12, 1.09, 1.116)])))
        self.assertEqual(got, exits.OPEN)


class TouchExitTests(unittest.TestCase):
    policy = exits.TouchExit()

    def test_a_wick_to_the_target_wins_even_if_it_closes_lower(self):
        self.assertEqual(resolve(self.policy, buy(), (1.10, 1.116, 1.099, 1.105)),
                         Exit(1, 1.1150, "target"))

    def test_a_wick_to_the_stop_loses_at_the_stop(self):
        self.assertEqual(resolve(self.policy, buy(), (1.099, 1.100, 1.094, 1.098)),
                         Exit(1, 1.0950, "stop"))

    def test_a_gap_through_the_stop_fills_at_the_worse_open(self):
        self.assertEqual(resolve(self.policy, buy(), (1.093, 1.094, 1.092, 1.093)),
                         Exit(1, 1.0930, "stop"))

    def test_a_gap_past_the_target_still_fills_at_the_target_not_better(self):
        self.assertEqual(resolve(self.policy, buy(), (1.118, 1.119, 1.117, 1.118)),
                         Exit(1, 1.1150, "target"))

    def test_a_candle_touching_both_is_a_loss(self):
        self.assertEqual(resolve(self.policy, buy(), (1.10, 1.116, 1.094, 1.10)).reason, "stop")

    def test_sells_mirror_buys(self):
        self.assertEqual(resolve(self.policy, sell(), (1.101, 1.106, 1.099, 1.104)),
                         Exit(1, 1.1050, "stop"))
        self.assertEqual(resolve(self.policy, sell(), (1.10, 1.101, 1.084, 1.095)),
                         Exit(1, 1.0850, "target"))
        self.assertEqual(resolve(self.policy, sell(), (1.107, 1.108, 1.106, 1.107)),
                         Exit(1, 1.1070, "stop"))       # gaps up through the stop

    def test_a_trade_that_never_touches_either_level_stays_open(self):
        self.assertEqual(resolve(self.policy, buy(), (1.10, 1.105, 1.096, 1.10),
                                 (1.10, 1.108, 1.097, 1.104)), exits.OPEN)


class PlanExitTests(unittest.TestCase):
    policy = exits.PlanExit()

    def test_a_close_beyond_invalidation_exits_at_that_close_before_the_stop(self):
        # Low 1.0955 stays above the 1.0950 stop, but the 1.0960 close is beyond 1.0970.
        self.assertEqual(resolve(self.policy, buy(), (1.099, 1.099, 1.0955, 1.0960)),
                         Exit(1, 1.0960, "invalidation"))

    def test_a_touched_stop_still_exits_at_the_stop(self):
        self.assertEqual(resolve(self.policy, buy(), (1.099, 1.100, 1.094, 1.096)),
                         Exit(1, 1.0950, "stop"))

    def test_a_target_touched_in_the_same_candle_as_a_bad_close_wins(self):
        self.assertEqual(resolve(self.policy, buy(), (1.10, 1.116, 1.0955, 1.0960)).reason,
                         "target")

    def test_a_close_exactly_on_the_invalidation_level_does_not_exit(self):
        self.assertEqual(resolve(self.policy, buy(), (1.10, 1.101, 1.0975, 1.0970)), exits.OPEN)

    def test_sells_mirror_buys(self):
        self.assertEqual(resolve(self.policy, sell(), (1.101, 1.1045, 1.10, 1.1040)),
                         Exit(1, 1.1040, "invalidation"))


class ExitRegistryTests(unittest.TestCase):
    def test_names(self):
        self.assertEqual([exits.get_exit(n).name for n in ("close", "touch", "plan")],
                         ["close", "touch", "plan"])

    def test_unknown_name_lists_the_choices(self):
        with self.assertRaises(ValueError) as ctx:
            exits.get_exit("magic")
        self.assertIn("close", str(ctx.exception))


class CostTests(unittest.TestCase):
    def test_no_cost_is_free(self):
        self.assertEqual(NoCost().price(0.0002, 0.0002, 0.0001), 0.0)

    def test_one_spread_per_round_trip(self):
        self.assertAlmostEqual(SpreadCost().price(0.00012, 0.0002, 0.0001), 0.00012)

    def test_slippage_and_multiplier_add_on(self):
        model = SpreadCost(multiplier=2.0, slippage_pips=0.5)
        self.assertAlmostEqual(model.price(0.0001, 0.0002, 0.0001), 0.0002 + 0.00005)

    def test_a_missing_entry_spread_falls_back_to_the_typical_one(self):
        self.assertAlmostEqual(SpreadCost().price(math.nan, 0.0002, 0.0001), 0.0002)

    def test_no_spread_data_at_all_is_an_error_not_free_trading(self):
        with self.assertRaises(MissingSpreadData):
            SpreadCost().price(math.nan, math.nan, 0.0001)

    def test_get_cost(self):
        self.assertIsInstance(costs.get_cost("none"), NoCost)
        self.assertEqual(costs.get_cost("spread", slippage_pips=1.0).slippage_pips, 1.0)
        with self.assertRaises(ValueError):
            costs.get_cost("free-lunch")


class SimulateTests(unittest.TestCase):
    def run_sim(self, signals, rows, policy=None, cost=None, spreads=None):
        h1 = frame_of(rows, spreads)
        return engine.simulate(signals, h1, policy or exits.CloseExit(), cost or NoCost(),
                               "EUR_USD", "test")

    def test_a_close_based_win_scores_the_planned_rr_and_a_loss_minus_one(self):
        trades, _ = self.run_sim(
            [buy(), buy(entry_index=2)],
            [ENTRY, (1.10, 1.12, 1.10, 1.116), ENTRY, (1.10, 1.10, 1.095, 1.096)])
        self.assertEqual(list(trades["outcome"]), ["win", "loss"])
        self.assertAlmostEqual(trades["r_gross"].iloc[0], 3.0)
        self.assertAlmostEqual(trades["r_gross"].iloc[1], -1.0)

    def test_a_winning_sell_scores_positive(self):
        trades, _ = self.run_sim([sell()], [ENTRY, (1.10, 1.101, 1.08, 1.084)])
        self.assertAlmostEqual(trades["r_gross"].iloc[0], 3.0)

    def test_a_gap_loss_is_worse_than_minus_one(self):
        trades, _ = self.run_sim([buy()], [ENTRY, (1.093, 1.094, 1.092, 1.093)],
                                 policy=exits.TouchExit())
        self.assertAlmostEqual(trades["r_gross"].iloc[0], (1.093 - 1.100) / 0.005)

    def test_an_unresolved_trade_is_open_with_no_result(self):
        trades, _ = self.run_sim([buy()], [ENTRY, (1.10, 1.105, 1.098, 1.10)])
        row = trades.iloc[0]
        self.assertEqual(row["outcome"], "open")
        self.assertTrue(math.isnan(row["r_gross"]) and math.isnan(row["r_net"]))
        self.assertTrue(pd.isna(row["exit_time"]))

    def test_exit_time_is_the_close_of_the_exit_candle_and_bars_are_counted(self):
        trades, _ = self.run_sim([buy()], [ENTRY, (1.10, 1.10, 1.099, 1.10),
                                           (1.10, 1.12, 1.10, 1.116)])
        row = trades.iloc[0]
        self.assertEqual(row["exit_time"], T0 + pd.Timedelta(hours=3))
        self.assertEqual(row["bars_held"], 2)

    def test_spread_cost_is_taken_off_in_r(self):
        trades, _ = self.run_sim([buy()], [ENTRY, (1.10, 1.12, 1.10, 1.116)],
                                 cost=SpreadCost(), spreads=[0.0001, 0.0001])
        row = trades.iloc[0]
        self.assertAlmostEqual(row["cost_r"], 0.0001 / 0.005)
        self.assertAlmostEqual(row["r_net"], 3.0 - 0.02)
        self.assertAlmostEqual(row["spread_pips"], 1.0)

    def test_the_spread_used_is_the_entry_candles(self):
        trades, _ = self.run_sim([buy()], [ENTRY, (1.10, 1.12, 1.10, 1.116)],
                                 cost=SpreadCost(), spreads=[0.0003, 0.0009])
        self.assertAlmostEqual(trades["cost_r"].iloc[0], 0.0003 / 0.005)

    def test_a_missing_entry_spread_uses_the_median_and_is_counted(self):
        trades, fallbacks = self.run_sim([buy()], [ENTRY, (1.10, 1.12, 1.10, 1.116)],
                                         cost=SpreadCost(), spreads=[np.nan, 0.0002])
        self.assertEqual(fallbacks, 1)
        self.assertAlmostEqual(trades["cost_r"].iloc[0], 0.0002 / 0.005)

    def test_costs_without_any_spread_data_fail_loudly(self):
        with self.assertRaises(MissingSpreadData):
            self.run_sim([buy()], [ENTRY, (1.10, 1.12, 1.10, 1.116)], cost=SpreadCost())

    def test_no_cost_runs_without_spread_data(self):
        trades, fallbacks = self.run_sim([buy()], [ENTRY, (1.10, 1.12, 1.10, 1.116)])
        self.assertEqual((trades["cost_r"].iloc[0], fallbacks), (0.0, 0))

    def test_strategy_meta_is_carried_into_the_trade_log(self):
        trades, _ = self.run_sim([buy(meta={"sweep_time": "x"})], [ENTRY, (1.10, 1.12, 1.10, 1.116)])
        self.assertEqual(trades["sweep_time"].iloc[0], "x")

    def test_meta_cannot_overwrite_a_reserved_column(self):
        with self.assertRaises(ValueError):
            self.run_sim([buy(meta={"r_net": 99})], [ENTRY, (1.10, 1.12, 1.10, 1.116)])

    def test_an_entry_index_outside_the_data_is_an_error(self):
        with self.assertRaises(ValueError):
            self.run_sim([buy(entry_index=7)], [ENTRY, ENTRY])

    def test_no_signals_gives_an_empty_frame_with_the_standard_columns(self):
        trades, _ = self.run_sim([], [ENTRY, ENTRY])
        self.assertTrue(trades.empty)
        self.assertEqual(list(trades.columns), engine.TRADE_COLUMNS)


def trades_of(results, times=None):
    rows = []
    for i, r in enumerate(results):
        rows.append({"outcome": "open" if r is None else ("win" if r > 0 else "loss"),
                     "planned_rr": 2.0, "r_gross": np.nan if r is None else r,
                     "r_net": np.nan if r is None else r, "cost_r": 0.0,
                     "entry_time": T0 + pd.Timedelta(hours=i if times is None else times[i])})
    return pd.DataFrame(rows, columns=["outcome", "planned_rr", "r_gross", "r_net", "cost_r",
                                       "entry_time"])


class ScoringTests(unittest.TestCase):
    def test_basic_statistics(self):
        s = scoring.summarize(trades_of([3, -1, -1, 2, -1]), months_spanned=10)
        self.assertEqual((s["signals"], s["resolved"], s["wins"], s["losses"]), (5, 5, 2, 3))
        self.assertAlmostEqual(s["win_rate"], 0.4)
        self.assertAlmostEqual(s["expectancy_net"], 0.4)
        self.assertAlmostEqual(s["profit_factor_net"], 5 / 3)
        self.assertAlmostEqual(s["signals_per_month"], 0.5)

    def test_max_drawdown_is_the_deepest_fall_from_a_peak(self):
        # Running total: 3, 2, 1, 3, 2, 1, 0  -> peak 3, trough 0.
        s = scoring.summarize(trades_of([3, -1, -1, 2, -1, -1, -1]))
        self.assertAlmostEqual(s["max_dd_net"], 3.0)

    def test_a_losing_start_counts_as_drawdown_from_zero(self):
        self.assertAlmostEqual(scoring.max_drawdown(np.array([-1.0, -1.0, 3.0])), 2.0)

    def test_drawdown_follows_entry_order_not_input_order(self):
        # Entered 3rd, 2nd, 1st in time: the +3 comes last, so the losses run first.
        s = scoring.summarize(trades_of([3, -1, -1], times=[2, 1, 0]))
        self.assertAlmostEqual(s["max_dd_net"], 2.0)

    def test_open_trades_are_counted_but_not_scored(self):
        s = scoring.summarize(trades_of([3, None, -1]))
        self.assertEqual((s["signals"], s["resolved"], s["open"]), (3, 2, 1))
        self.assertAlmostEqual(s["expectancy_net"], 1.0)

    def test_no_trades(self):
        s = scoring.summarize(trades_of([]))
        self.assertEqual((s["signals"], s["expectancy_net"], s["ci_low"]), (0, None, None))

    def test_only_open_trades(self):
        self.assertIsNone(scoring.summarize(trades_of([None, None]))["expectancy_net"])

    def test_a_single_trade_has_no_interval(self):
        s = scoring.summarize(trades_of([2.0]))
        self.assertEqual((s["expectancy_net"], s["ci_low"]), (2.0, None))

    def test_all_winners_have_no_profit_factor(self):
        self.assertIsNone(scoring.summarize(trades_of([1, 2, 3]))["profit_factor_net"])

    def test_gross_and_net_differ_by_the_cost(self):
        trades = trades_of([3, -1, 2])
        trades["cost_r"] = 0.1
        trades["r_net"] = trades["r_gross"] - 0.1
        s = scoring.summarize(trades)
        self.assertAlmostEqual(s["expectancy_gross"] - s["expectancy_net"], 0.1)
        self.assertAlmostEqual(s["avg_cost_r"], 0.1)


class BootstrapTests(unittest.TestCase):
    def test_the_interval_brackets_the_mean_and_is_repeatable(self):
        values = np.array([3, -1, -1, 2, -1, -1, 3, -1, -1, 2.0])
        low, high, p = scoring.bootstrap_mean_interval(values, samples=2000)
        self.assertLess(low, values.mean())
        self.assertGreater(high, values.mean())
        self.assertEqual(scoring.bootstrap_mean_interval(values, samples=2000), (low, high, p))

    def test_a_small_sample_of_a_mixed_strategy_spans_zero(self):
        low, high, _ = scoring.bootstrap_mean_interval(np.array([2.5, -1, -1, -1, 2.5, -1.0]))
        self.assertLess(low, 0)
        self.assertGreater(high, 0)

    def test_identical_results_have_a_zero_width_interval(self):
        low, high, p = scoring.bootstrap_mean_interval(np.full(20, 0.5))
        self.assertEqual((low, high, p), (0.5, 0.5, 0.0))

    def test_more_trades_narrow_the_interval(self):
        rng = np.random.default_rng(1)
        small = rng.choice([2.0, -1.0], 30)
        large = rng.choice([2.0, -1.0], 3000)
        width = lambda v: np.subtract(*scoring.bootstrap_mean_interval(v, samples=1000)[1::-1])
        self.assertGreater(width(small), 3 * width(large))


class RegistryTests(unittest.TestCase):
    def test_sweep_fvg_is_registered(self):
        self.assertIn("sweep_fvg", STRATEGIES)
        self.assertEqual(get_strategy("sweep_fvg").timeframes, ("D", "H4", "H1"))

    def test_unknown_strategy_lists_the_choices(self):
        with self.assertRaises(ValueError) as ctx:
            get_strategy("nope")
        self.assertIn("sweep_fvg", str(ctx.exception))


def section81_frames():
    """The §8.1 worked example (a confirmed bullish signal that wins) as raw candle frames."""
    case = Section81BullishSignalTest("test_full_pipeline_produces_a_win")
    case.setUp()
    market = case.market
    return {"D": market.daily.copy(),
            "H4": market.h4[["time", "open", "high", "low", "close", "volume"]].copy(),
            "H1": market.h1.copy()}


class SweepFvgPluginTests(unittest.TestCase):
    def test_the_worked_example_becomes_the_expected_signal(self):
        out = get_strategy("sweep_fvg").generate("EUR_USD", section81_frames())
        (signal,) = out.signals
        self.assertEqual(signal.direction, "bullish")
        self.assertAlmostEqual(signal.entry_price, 1.0815)
        self.assertAlmostEqual(signal.stop_price, 1.0790)
        self.assertAlmostEqual(signal.target_price, 1.0900)
        self.assertAlmostEqual(signal.invalidation_price, 1.0795)      # the sweep extreme
        self.assertAlmostEqual(signal.planned_rr, 3.4, places=2)
        self.assertEqual(out.funnel, {"win": 1})
        self.assertIn("sweep_time", signal.meta)

    def test_the_signals_entry_candle_is_the_confirming_close(self):
        frames = section81_frames()
        (signal,) = get_strategy("sweep_fvg").generate("EUR_USD", frames).signals
        h1 = frames["H1"]
        self.assertAlmostEqual(h1["close"].iloc[signal.entry_index], signal.entry_price)
        self.assertEqual(signal.entry_time, h1["time"].iloc[signal.entry_index] + pd.Timedelta(hours=1))

    def test_close_exit_reproduces_the_legacy_result_and_other_exits_agree_here(self):
        frames = section81_frames()
        strategy = get_strategy("sweep_fvg")
        for name in ("close", "touch", "plan"):
            run = engine.run_instrument(strategy, "EUR_USD", exits.get_exit(name), NoCost(),
                                        frames=frames)
            (trade,) = run.trades.to_dict("records")
            self.assertEqual(trade["outcome"], "win", name)
            self.assertAlmostEqual(trade["r_gross"], 3.4, places=2)

    def test_run_instrument_reports_the_funnel_and_the_names(self):
        run = engine.run_instrument(get_strategy("sweep_fvg"), "EUR_USD", exits.CloseExit(),
                                    NoCost(), frames=section81_frames())
        self.assertEqual((run.strategy, run.exit, run.costs), ("sweep_fvg", "close", "none"))
        self.assertEqual(run.funnel, {"win": 1})
        self.assertGreater(run.months_spanned, 0)


if __name__ == "__main__":
    unittest.main()
