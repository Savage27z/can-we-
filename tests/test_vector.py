import unittest

import numpy as np
import pandas as pd

from research import exits, vector
from research.costs import MissingSpreadData, NoCost, SpreadCost
from research.exits import Candles
from research.strategy import Signal
from research.vector import Arrays, resolve_batch

T0 = pd.Timestamp("2020-01-06 00:00", tz="UTC")


def random_market(n=4000, seed=0, sigma=0.0010, jump_share=0.02):
    """A driftless random walk with realistic candles and the occasional price gap."""
    rng = np.random.default_rng(seed)
    close = np.empty(n)
    open_ = np.empty(n)
    last = 1.10
    for i in range(n):
        gap = rng.normal(0, sigma * 4) if rng.random() < jump_share else 0.0
        open_[i] = last + gap
        close[i] = open_[i] + rng.normal(0, sigma)
        last = close[i]
    body_hi, body_lo = np.maximum(open_, close), np.minimum(open_, close)
    high = body_hi + np.abs(rng.normal(0, sigma / 2, n))
    low = body_lo - np.abs(rng.normal(0, sigma / 2, n))
    return pd.DataFrame({"time": pd.date_range(T0, periods=n, freq="1h"), "open": open_,
                         "high": high, "low": low, "close": close, "volume": 1})


def random_signals(frame, count, seed=1):
    rng = np.random.default_rng(seed)
    signals = []
    for entry in rng.integers(0, len(frame) - 1, count):
        price = float(frame["close"].iloc[entry])
        risk = rng.uniform(0.0010, 0.0040)
        reward = risk * rng.uniform(1.5, 4.0)
        buffer = min(0.0005, risk * 0.4)
        if rng.random() < 0.5:
            stop, target, inv = price - risk, price + reward, price - risk + buffer
            direction = "bullish"
        else:
            stop, target, inv = price + risk, price - reward, price + risk - buffer
            direction = "bearish"
        signals.append(Signal("EUR_USD", direction, int(entry), T0, price, stop, target, inv))
    return signals


def batch_inputs(signals):
    sign = np.array([1.0 if s.direction == "bullish" else -1.0 for s in signals])
    return dict(
        entry_index=np.array([s.entry_index for s in signals], dtype=np.int64), sign=sign,
        entry_price=np.array([s.entry_price for s in signals]),
        stop=np.array([s.stop_price for s in signals]),
        target=np.array([s.target_price for s in signals]),
        invalidation=np.array([s.invalidation_price for s in signals]),
        risk=np.array([s.risk for s in signals]))


class ParityWithTheScalarExits(unittest.TestCase):
    """The null model's speed is worthless if it resolves trades differently from the real
    ones, so these compare it with exits.py on every trade."""

    @classmethod
    def setUpClass(cls):
        cls.frame = random_market()
        cls.signals = random_signals(cls.frame, 600)
        cls.candles = Candles.from_frame(cls.frame)
        cls.arrays = Arrays.from_frame(cls.frame)
        cls.inputs = batch_inputs(cls.signals)

    def check(self, policy):
        index, r = resolve_batch(self.arrays, policy, **self.inputs)
        scalar = exits.get_exit(policy)
        for i, signal in enumerate(self.signals):
            result = scalar.resolve(signal, self.candles)
            if result.index is None:
                self.assertEqual(index[i], -1, f"{policy} trade {i} should be open")
                self.assertTrue(np.isnan(r[i]))
                continue
            sign = 1.0 if signal.direction == "bullish" else -1.0
            expected_r = sign * (result.price - signal.entry_price) / signal.risk
            self.assertEqual(index[i], result.index, f"{policy} trade {i}: exit candle")
            self.assertEqual(r[i], expected_r, f"{policy} trade {i}: R")

    def test_close(self):
        self.check("close")

    def test_touch(self):
        self.check("touch")

    def test_plan(self):
        self.check("plan")

    def test_the_market_actually_exercises_gaps_and_both_outcomes(self):
        index, r = resolve_batch(self.arrays, "touch", **self.inputs)
        resolved = r[~np.isnan(r)]
        self.assertGreater((resolved > 0).sum(), 50)
        self.assertGreater((resolved < 0).sum(), 50)
        self.assertGreater((resolved < -1.0001).sum(), 0, "no gap-through-the-stop trades")


class BlockBoundaryTests(unittest.TestCase):
    """Trades are resolved in blocks of candles; the boundary must not shift or lose exits."""

    def flat_then_spike(self, spike_at, spike_high=None, spike_low=None, close=None):
        n = 600
        f = pd.DataFrame({"time": pd.date_range(T0, periods=n, freq="1h"), "open": 1.10,
                          "high": 1.1002, "low": 1.0998, "close": 1.10, "volume": 1})
        if spike_high is not None:
            f.loc[spike_at, "high"] = spike_high
        if spike_low is not None:
            f.loc[spike_at, "low"] = spike_low
        if close is not None:
            f.loc[spike_at, "close"] = close
        return f

    def buy(self, frame, policy):
        arrays = Arrays.from_frame(frame)
        return resolve_batch(arrays, policy, entry_index=np.array([0]), sign=np.array([1.0]),
                             entry_price=np.array([1.10]), stop=np.array([1.09]),
                             target=np.array([1.11]), invalidation=np.array([1.0925]),
                             risk=np.array([0.01]))

    def test_an_exit_is_found_wherever_it_falls_relative_to_a_block_edge(self):
        block = vector.BLOCK
        for spike_at in (1, block - 1, block, block + 1, 2 * block, 2 * block + 1, 4 * block + 7):
            frame = self.flat_then_spike(spike_at, spike_high=1.12)
            for policy in ("touch", "plan"):
                index, r = self.buy(frame, policy)
                self.assertEqual(index[0], spike_at, f"{policy} spike at {spike_at}")
                self.assertAlmostEqual(r[0], 1.0)
            frame = self.flat_then_spike(spike_at, close=1.12, spike_high=1.12)
            index, r = self.buy(frame, "close")
            self.assertEqual(index[0], spike_at)

    def test_a_trade_that_never_ends_is_open(self):
        index, r = self.buy(self.flat_then_spike(300), "plan")
        self.assertEqual(index[0], -1)
        self.assertTrue(np.isnan(r[0]))

    def test_a_trade_entered_on_the_last_candle_is_open_not_an_error(self):
        frame = self.flat_then_spike(300)
        arrays = Arrays.from_frame(frame)
        index, r = resolve_batch(arrays, "close", entry_index=np.array([len(frame) - 1]),
                                 sign=np.array([1.0]), entry_price=np.array([1.10]),
                                 stop=np.array([1.09]), target=np.array([1.11]),
                                 invalidation=np.array([1.0925]), risk=np.array([0.01]))
        self.assertEqual(index[0], -1)

    def test_an_empty_batch(self):
        arrays = Arrays.from_frame(self.flat_then_spike(300))
        empty = np.array([], dtype=float)
        index, r = resolve_batch(arrays, "close", np.array([], dtype=np.int64), empty, empty,
                                 empty, empty, empty, empty)
        self.assertEqual((len(index), len(r)), (0, 0))

    def test_an_unknown_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            self.buy(self.flat_then_spike(5), "magic")

    def test_sells_are_resolved_as_mirrored_buys(self):
        frame = self.flat_then_spike(50, spike_low=1.08)
        arrays = Arrays.from_frame(frame)
        index, r = resolve_batch(arrays, "touch", entry_index=np.array([0]),
                                 sign=np.array([-1.0]), entry_price=np.array([1.10]),
                                 stop=np.array([1.11]), target=np.array([1.09]),
                                 invalidation=np.array([1.1075]), risk=np.array([0.01]))
        self.assertEqual(index[0], 50)
        self.assertAlmostEqual(r[0], 1.0)


class ArrayCostTests(unittest.TestCase):
    def test_array_pricing_matches_scalar_pricing(self):
        spreads = np.array([0.0001, np.nan, 0.0003])
        for model in (NoCost(), SpreadCost(), SpreadCost(multiplier=2.0, slippage_pips=0.5)):
            got = model.price_array(spreads, 0.0002, 0.0001)
            want = [model.price(float(s), 0.0002, 0.0001) for s in spreads]
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-15)

    def test_no_spread_data_at_all_is_an_error_for_arrays_too(self):
        with self.assertRaises(MissingSpreadData):
            SpreadCost().price_array(np.array([np.nan, np.nan]), float("nan"), 0.0001)

    def test_no_cost_needs_no_spread_data(self):
        np.testing.assert_array_equal(NoCost().price_array(np.array([np.nan]), float("nan"), 0.0001),
                                      [0.0])


if __name__ == "__main__":
    unittest.main()
