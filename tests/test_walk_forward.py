import unittest

import numpy as np
import pandas as pd

from research.walk_forward import walk_forward_selection

UTC = "UTC"


def trades_frame(rows):
    df = pd.DataFrame(rows, columns=["instrument", "entry_time", "r_net"])
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["outcome"] = np.where(df["r_net"] > 0, "win", "loss")
    return df


def synthetic(true_means, seed=0, trades_per_year=8, years=range(2005, 2025), noise=1.0):
    """Every instrument trades a few times a year; its results are noise around a fixed mean."""
    rng = np.random.default_rng(seed)
    rows = []
    for name, mean in true_means.items():
        for year in years:
            for k in range(trades_per_year):
                when = pd.Timestamp(f"{year}-01-15", tz=UTC) + pd.Timedelta(days=int(rng.integers(0, 340)))
                rows.append((name, when, mean + rng.normal(0, noise)))
    return trades_frame(rows)


NAMES = [f"P{i:02d}" for i in range(20)]


class SelectionTests(unittest.TestCase):
    def test_a_pair_that_is_persistently_better_is_selected_and_the_test_notices(self):
        means = {name: 0.0 for name in NAMES}
        for name in NAMES[:3]:
            means[name] = 0.8
        result = walk_forward_selection(synthetic(means), top_n=3, replicates=2000)
        self.assertGreater(len(result.folds), 4)
        self.assertGreater(result.excess, 0.4)
        self.assertLess(result.p_value, 0.01)
        self.assertGreater(result.folds_won, len(result.folds) // 2)

    def test_when_every_pair_is_the_same_selection_adds_nothing(self):
        result = walk_forward_selection(synthetic({name: 0.1 for name in NAMES}, seed=4),
                                        top_n=3, replicates=2000)
        self.assertLess(abs(result.excess), 0.25)

    def test_p_values_are_roughly_uniform_when_there_is_nothing_to_find(self):
        p = []
        for seed in range(60):
            r = walk_forward_selection(synthetic({name: 0.0 for name in NAMES}, seed=seed, years=range(2005, 2021)),
                                       top_n=3, replicates=400, seed=seed)
            p.append(r.p_value)
        p = np.array(p)
        self.assertTrue(0.35 < p.mean() < 0.65, p.mean())
        self.assertLessEqual((p < 0.10).sum(), 14)

    def test_selection_uses_only_trades_before_the_block(self):
        # P00 is awful before 2013 and superb from 2013: it must not be chosen for the 2013 block.
        rows = []
        for name in NAMES:
            for year in range(2005, 2025):
                for k in range(8):
                    good = name == "P00" and year >= 2013
                    rows.append((name, pd.Timestamp(f"{year}-03-{k + 1:02d}", tz=UTC),
                                 5.0 if good else (-3.0 if name == "P00" else 0.0)))
        result = walk_forward_selection(trades_frame(rows), top_n=3, first_test_year=2013,
                                        test_years=2, replicates=200)
        first = result.folds[0]
        self.assertEqual(first.test_start.year, 2013)
        self.assertNotIn("P00", first.selected)

    def test_each_block_is_scored_only_on_its_own_years(self):
        result = walk_forward_selection(synthetic({name: 0.0 for name in NAMES}), top_n=3,
                                        first_test_year=2011, test_years=2, replicates=100)
        starts = [f.test_start.year for f in result.folds]
        self.assertEqual(starts, sorted(set(starts)))
        self.assertTrue(all(b - a == 2 for a, b in zip(starts, starts[1:])))
        for fold in result.folds:
            self.assertEqual(fold.test_end.year - fold.test_start.year, 2, fold)
            self.assertEqual((fold.test_start.month, fold.test_start.day), (1, 1))

    def test_a_pair_without_enough_training_trades_cannot_be_selected(self):
        rows = [(name, pd.Timestamp(f"{y}-06-01", tz=UTC), 0.0)
                for name in NAMES for y in range(2005, 2025) for _ in range(8)]
        # A pair with a spectacular but tiny record: 3 trades before 2011.
        rows += [("NEW", pd.Timestamp("2009-06-01", tz=UTC), 50.0)] * 3
        rows += [("NEW", pd.Timestamp(f"{y}-06-01", tz=UTC), 50.0) for y in range(2011, 2025)]
        result = walk_forward_selection(trades_frame(rows), top_n=3, replicates=100)
        self.assertNotIn("NEW", result.folds[0].selected)

    def test_open_trades_are_ignored(self):
        df = synthetic({name: 0.0 for name in NAMES})
        opened = df.copy()
        opened["r_net"] = np.nan
        opened["outcome"] = "open"
        combined = pd.concat([df, opened], ignore_index=True)
        a = walk_forward_selection(df, top_n=3, replicates=300, seed=1)
        b = walk_forward_selection(combined, top_n=3, replicates=300, seed=1)
        self.assertAlmostEqual(a.selected_mean, b.selected_mean)
        self.assertEqual(a.selected_trades, b.selected_trades)

    def test_with_too_few_pairs_to_choose_between_there_are_no_folds(self):
        result = walk_forward_selection(synthetic({"A": 0.0, "B": 0.0}), top_n=5, replicates=50)
        self.assertEqual(result.folds, [])
        self.assertTrue(np.isnan(result.p_value))

    def test_no_trades(self):
        empty = trades_frame([])
        self.assertEqual(walk_forward_selection(empty, replicates=10).folds, [])

    def test_the_same_seed_reproduces_and_the_selection_itself_is_deterministic(self):
        df = synthetic({name: 0.05 * i for i, name in enumerate(NAMES)}, seed=8)
        a = walk_forward_selection(df, top_n=3, replicates=500, seed=2)
        b = walk_forward_selection(df, top_n=3, replicates=500, seed=2)
        self.assertEqual((a.p_value, a.selected_mean), (b.p_value, b.selected_mean))
        self.assertEqual([f.selected for f in a.folds], [f.selected for f in b.folds])


if __name__ == "__main__":
    unittest.main()
