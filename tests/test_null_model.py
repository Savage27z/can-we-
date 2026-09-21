import unittest

import numpy as np
import pandas as pd

from backtest import sessions
from research import engine, exits, null_model
from research.costs import NoCost, SpreadCost
from research.null_model import NullResult
from research.strategy import Signal

T0 = pd.Timestamp("2020-01-06 00:00", tz="UTC")     # a Monday, midnight UTC


def walk(n, seed, sigma=0.0010, jump_share=0.01, spread=None):
    """A vectorised driftless random walk of H1 candles, with the occasional gap."""
    rng = np.random.default_rng(seed)
    gap = np.where(rng.random(n) < jump_share, rng.normal(0, sigma * 4, n), 0.0)
    step = rng.normal(0, sigma, n)
    close = 1.10 + np.cumsum(gap + step)
    open_ = np.concatenate([[1.10], close[:-1]]) + gap
    high = np.maximum(open_, close) + np.abs(rng.normal(0, sigma / 2, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, sigma / 2, n))
    frame = pd.DataFrame({"time": pd.date_range(T0, periods=n, freq="1h"), "open": open_,
                          "high": high, "low": low, "close": close, "volume": 1})
    if spread is not None:
        frame["spread"] = spread
    return frame


def signal_at(frame, entry, direction, risk=0.0025, reward=0.0050):
    price = float(frame["close"].iloc[entry])
    sign = 1.0 if direction == "bullish" else -1.0
    return Signal("EUR_USD", direction, int(entry), frame["time"].iloc[entry] + pd.Timedelta(hours=1),
                  price, price - sign * risk, price + sign * reward)


def in_session_indices(frame):
    return np.flatnonzero(sessions.in_session_mask(frame["time"]).to_numpy())


def p_value_for(frame, signals, policy="touch", replicates=200, seed=0, window_days=30):
    trades, _ = engine.simulate(signals, frame, exits.get_exit(policy), NoCost(), "EUR_USD", "t")
    observed = trades.loc[trades["outcome"] != "open", "r_net"].mean()
    null = null_model.null_replicates(signals, frame, policy, NoCost(), 0.0001, replicates,
                                      window_days=window_days, seed=seed, key="EUR_USD")
    return null_model.compare(observed, null)


class CalibrationTests(unittest.TestCase):
    """A null-model p-value is only worth reading if a strategy with no skill gets
    p-values spread evenly over 0-1, and a strategy with real skill gets tiny ones."""

    def test_signals_at_random_times_get_roughly_uniform_p_values(self):
        p_values = []
        for trial in range(120):
            rng = np.random.default_rng(1000 + trial)
            frame = walk(6000, seed=trial)
            eligible = in_session_indices(frame)
            entries = rng.choice(eligible[eligible < 5000], 40, replace=False)
            signals = [signal_at(frame, e, rng.choice(["bullish", "bearish"])) for e in entries]
            p_values.append(p_value_for(frame, signals, seed=trial).p_value)
        p = np.array(p_values)
        self.assertTrue(0.40 < p.mean() < 0.60, f"mean p {p.mean():.3f} (uniform would be 0.5)")
        self.assertLessEqual((p < 0.10).sum(), 27, "far too many false positives at 10%")
        self.assertGreaterEqual((p < 0.50).sum(), 45)
        self.assertLessEqual((p < 0.50).sum(), 75)

    def test_a_strategy_with_hindsight_skill_gets_tiny_p_values(self):
        for trial in range(6):
            frame = walk(6000, seed=500 + trial)
            eligible = in_session_indices(frame)
            close = frame["close"].to_numpy()
            signals = []
            for e in eligible[eligible < 5900][::7]:
                ahead = close[e + 12] - close[e]
                if abs(ahead) > 0.0025:                      # it peeks: enters only before a big move
                    signals.append(signal_at(frame, e, "bullish" if ahead > 0 else "bearish"))
            self.assertGreater(len(signals), 15)
            comparison = p_value_for(frame, signals[:60], seed=trial)
            self.assertLess(comparison.p_value, 0.02, f"trial {trial}: {comparison}")
            self.assertGreater(comparison.excess, 0.2)

    def test_a_strategy_that_always_enters_at_the_worst_time_gets_p_values_near_one(self):
        frame = walk(6000, seed=77)
        eligible = in_session_indices(frame)
        close = frame["close"].to_numpy()
        signals = []
        for e in eligible[eligible < 5900][::7]:
            ahead = close[e + 12] - close[e]
            if abs(ahead) > 0.0025:                          # enters AGAINST the coming move
                signals.append(signal_at(frame, e, "bearish" if ahead > 0 else "bullish"))
        comparison = p_value_for(frame, signals[:60])
        self.assertGreater(comparison.p_value, 0.95)
        self.assertLess(comparison.excess, -0.2)


class DrawEntriesTests(unittest.TestCase):
    def setUp(self):
        self.frame = walk(3000, seed=3)
        self.times = null_model._times_ns(self.frame)
        self.eligible = in_session_indices(self.frame)

    def draw(self, entries, window=10, k=50, seed=0):
        return null_model.draw_entries(np.array(entries), self.times, self.eligible, window, k,
                                       np.random.default_rng(seed))

    def test_every_draw_is_an_in_session_candle(self):
        drawn = self.draw([500, 1500])
        self.assertTrue(np.isin(drawn, self.eligible).all())

    def test_every_draw_is_within_the_window_of_the_real_entry(self):
        drawn = self.draw([1500], window=10)
        offset = np.abs(self.times[drawn] - self.times[1500]) / (86_400 * 1e9)
        self.assertLessEqual(offset.max(), 10.0001)
        self.assertGreater(offset.max(), 3, "draws should spread across the window")

    def test_a_shorter_window_stays_closer(self):
        near = self.draw([1500], window=2)
        far = self.draw([1500], window=20)
        span = lambda d: np.ptp(self.times[d])
        self.assertLess(span(near), span(far))

    def test_the_same_seed_gives_the_same_draws_and_another_seed_differs(self):
        np.testing.assert_array_equal(self.draw([800], seed=1), self.draw([800], seed=1))
        self.assertFalse(np.array_equal(self.draw([800], seed=1), self.draw([800], seed=2)))

    def test_with_no_eligible_candle_nearby_it_falls_back_to_anywhere(self):
        only_late = self.eligible[self.eligible > 2500]
        drawn = null_model.draw_entries(np.array([10]), self.times, only_late, 1, 30,
                                        np.random.default_rng(0))
        self.assertTrue(np.isin(drawn, only_late).all())

    def test_shape_is_signals_by_replicates(self):
        self.assertEqual(self.draw([100, 200, 300], k=7).shape, (3, 7))


class NullReplicateTests(unittest.TestCase):
    def setUp(self):
        self.frame = walk(4000, seed=11, spread=0.0001)
        eligible = in_session_indices(self.frame)
        self.signals = [signal_at(self.frame, e, "bullish" if i % 2 else "bearish")
                        for i, e in enumerate(eligible[eligible < 3500][::40][:30])]

    def run_null(self, **over):
        args = dict(signals=self.signals, h1=self.frame, policy="touch", cost_model=NoCost(),
                    pip_size=0.0001, replicates=60, seed=5, key="EUR_USD")
        return null_model.null_replicates(**{**args, **over})

    def test_the_same_seed_and_key_reproduce_exactly(self):
        a, b = self.run_null(), self.run_null()
        np.testing.assert_array_equal(a.sums, b.sums)

    def test_a_different_key_or_seed_gives_different_draws(self):
        base = self.run_null()
        self.assertFalse(np.array_equal(base.sums, self.run_null(key="GBP_USD").sums))
        self.assertFalse(np.array_equal(base.sums, self.run_null(seed=6).sums))

    def test_counts_are_at_most_one_trade_per_signal_per_replicate(self):
        null = self.run_null()
        self.assertEqual(null.sums.shape, (60,))
        self.assertLessEqual(null.counts.max(), len(self.signals))
        self.assertGreater(null.counts.min(), len(self.signals) // 2)

    def test_costs_lower_the_null_by_about_the_spread_in_r(self):
        free = self.run_null().means().mean()
        costly = self.run_null(cost_model=SpreadCost()).means().mean()
        self.assertAlmostEqual(free - costly, 0.0001 / 0.0025, delta=0.006)   # one pip over 25 pips

    def test_random_direction_is_a_different_null_from_the_same_direction(self):
        same = self.run_null(direction="same")
        randomised = self.run_null(direction="random")
        self.assertFalse(np.array_equal(same.sums, randomised.sums))

    def test_an_unknown_direction_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_null(direction="sideways")

    def test_no_signals_gives_an_empty_null(self):
        null = self.run_null(signals=[])
        self.assertEqual((null.sums.sum(), null.counts.sum()), (0, 0))
        self.assertTrue(np.isnan(null.means()).all())

    def test_costs_without_spread_data_fail_loudly(self):
        frame = walk(4000, seed=11)                          # no spread column
        with self.assertRaises(Exception) as ctx:
            self.run_null(h1=frame, cost_model=SpreadCost())
        self.assertIn("spread", str(ctx.exception))


class CompareTests(unittest.TestCase):
    def null_of(self, means):
        return NullResult(sums=np.array(means, dtype=float), counts=np.ones(len(means), dtype=np.int64))

    def test_a_result_above_every_replicate_gets_the_smallest_possible_p(self):
        c = null_model.compare(9.0, self.null_of(np.linspace(-1, 1, 99)))
        self.assertAlmostEqual(c.p_value, 1 / 100)
        self.assertGreater(c.z, 3)

    def test_a_result_below_every_replicate_gets_p_one(self):
        self.assertAlmostEqual(null_model.compare(-9.0, self.null_of(np.linspace(-1, 1, 99))).p_value, 1.0)

    def test_a_result_at_the_median_gets_a_middling_p(self):
        c = null_model.compare(0.0, self.null_of(np.linspace(-1, 1, 99)))
        self.assertTrue(0.45 < c.p_value < 0.55)
        self.assertAlmostEqual(c.excess, 0.0, places=6)

    def test_excess_is_observed_minus_the_null_mean(self):
        c = null_model.compare(0.3, self.null_of([0.0, 0.2, -0.2, 0.1]))
        self.assertAlmostEqual(c.excess, 0.3 - 0.025)

    def test_replicates_with_no_resolved_trades_are_left_out(self):
        null = NullResult(sums=np.array([1.0, 0.0, 3.0]), counts=np.array([1, 0, 1]))
        self.assertEqual(null_model.compare(0.0, null).replicates, 2)

    def test_too_few_replicates_or_a_missing_observation_give_nan_not_a_verdict(self):
        self.assertTrue(np.isnan(null_model.compare(1.0, self.null_of([0.5])).p_value))
        self.assertTrue(np.isnan(null_model.compare(float("nan"), self.null_of([0.1, 0.2])).p_value))


class PoolingTests(unittest.TestCase):
    def test_pooling_adds_sums_and_counts_replicate_by_replicate(self):
        a = NullResult(np.array([2.0, 4.0]), np.array([2, 4]))
        b = NullResult(np.array([1.0, 0.0]), np.array([1, 0]))
        pooled = null_model.pool([a, b])
        np.testing.assert_array_equal(pooled.sums, [3.0, 4.0])
        np.testing.assert_array_equal(pooled.counts, [3, 4])
        np.testing.assert_allclose(pooled.means(), [1.0, 1.0])

    def test_standardised_null_has_mean_zero_and_unit_spread(self):
        z = null_model.standardised_null(NullResult(np.array([1.0, 2.0, 3.0, 4.0]), np.ones(4, int)))
        self.assertAlmostEqual(float(np.nanmean(z)), 0.0)
        self.assertAlmostEqual(float(np.nanstd(z, ddof=1)), 1.0)

    def test_a_null_with_no_spread_cannot_be_standardised(self):
        self.assertIsNone(null_model.standardised_null(NullResult(np.full(5, 1.0), np.ones(5, int))))


class GeometryTests(unittest.TestCase):
    def test_a_signals_geometry_is_captured_and_direction_is_a_sign(self):
        frame = walk(500, seed=1)
        buy = signal_at(frame, 100, "bullish", risk=0.002, reward=0.005)
        sell = signal_at(frame, 200, "bearish", risk=0.003, reward=0.006)
        geo = null_model.geometry_of([buy, sell])
        np.testing.assert_array_equal(geo.sign, [1.0, -1.0])
        np.testing.assert_allclose(geo.risk, [0.002, 0.003])
        np.testing.assert_allclose(geo.reward, [0.005, 0.006])
        np.testing.assert_allclose(geo.invalidation_offset, [0.002, 0.003])   # defaults to the stop

    def test_the_invalidation_offset_uses_the_signals_own_level_when_it_has_one(self):
        frame = walk(500, seed=1)
        price = float(frame["close"].iloc[100])
        s = Signal("EUR_USD", "bullish", 100, T0, price, price - 0.003, price + 0.006,
                   invalidation_price=price - 0.0025)
        self.assertAlmostEqual(float(null_model.geometry_of([s]).invalidation_offset[0]), 0.0025)


if __name__ == "__main__":
    unittest.main()
