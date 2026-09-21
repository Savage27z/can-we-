import unittest

import numpy as np
import pandas as pd

from research.xs import inference, universe
from research.xs.inference import NullDistribution, block_bootstrap_interval, p_upper, permutation_null
from research.xs.legs import Tables
from research.xs.select import K, Selection, select


def make_tables(cohorts=300, currencies=16, seed=0, skill=0.0, start="2006-01-02", missing_exit=0.0,
                hold=4, formation=84):
    """Random tables. `skill` makes the score predict the forward result (momentum works when > 0)."""
    rng = np.random.default_rng(seed)
    times = pd.date_range(pd.Timestamp(start, tz="UTC") + pd.Timedelta(hours=7), periods=cohorts,
                          freq="7D")
    score = rng.normal(size=(cohorts, currencies))
    gross_long = skill * score + rng.normal(0, 1.2, size=(cohorts, currencies))
    gross_short = -gross_long
    cost = np.abs(rng.normal(0.03, 0.01, size=(cohorts, currencies)))
    if missing_exit:
        hole = rng.random((cohorts, currencies)) < missing_exit
        gross_long[hole], gross_short[hole] = np.nan, np.nan
    return Tables(times, universe.CURRENCIES[:currencies], score, gross_long, gross_short, cost,
                  hold, formation)


def small_tables():
    """Three cohorts, eight currencies, scores 0..7 in currency order (so the ranking is obvious)."""
    scores = np.tile(np.arange(8, dtype=float), (3, 1))
    times = pd.date_range("2010-01-04 07:00", periods=3, freq="7D", tz="UTC")
    gross_long = np.tile(np.arange(8, dtype=float) / 10, (3, 1))
    return Tables(times, universe.CURRENCIES[:8], scores, gross_long, -gross_long,
                  np.full((3, 8), 0.05), 4, 84)


class SelectionTests(unittest.TestCase):
    def names(self, selection, direction):
        keep = selection.direction == direction
        return sorted(universe.CURRENCIES[j] for j in selection.currency[keep & (selection.cohort == 0)])

    def test_momentum_is_long_the_strongest_and_short_the_weakest(self):
        sel = select(small_tables(), "momentum")
        self.assertEqual(self.names(sel, 1), sorted(universe.CURRENCIES[5:8]))
        self.assertEqual(self.names(sel, -1), sorted(universe.CURRENCIES[0:3]))

    def test_reversal_is_the_mirror_image(self):
        sel = select(small_tables(), "reversal")
        self.assertEqual(self.names(sel, 1), sorted(universe.CURRENCIES[0:3]))
        self.assertEqual(self.names(sel, -1), sorted(universe.CURRENCIES[5:8]))

    def test_every_cohort_has_three_longs_and_three_shorts(self):
        sel = select(make_tables(50), "momentum")
        for c in range(50):
            here = sel.cohort == c
            self.assertEqual(int((sel.direction[here] > 0).sum()), K)
            self.assertEqual(int((sel.direction[here] < 0).sum()), K)
            self.assertEqual(len(set(sel.currency[here])), 2 * K)

    def test_an_ineligible_currency_is_never_chosen(self):
        tables = small_tables()
        tables.score[:, 7] = np.nan                       # the strongest is missing
        chosen = set(select(tables, "momentum").currency)
        self.assertNotIn(7, chosen)
        self.assertIn(6, chosen)

    def test_the_oracle_ranks_by_the_forward_result(self):
        tables = small_tables()
        tables.gross_long[:] = tables.gross_long[:, ::-1]           # now currency 0 did best
        sel = select(tables, "oracle")
        self.assertEqual(self.names(sel, 1), sorted(universe.CURRENCIES[0:3]))

    def test_the_oracle_earns_more_than_any_honest_rule_on_random_data(self):
        tables = make_tables(200, seed=3)
        oracle = select(tables, "oracle").gross.mean()
        honest = select(tables, "momentum").gross.mean()
        self.assertGreater(oracle, honest + 0.5)

    def test_random_assignment_is_reproducible_and_depends_on_the_seed(self):
        tables = make_tables(60)
        a = select(tables, "random", seed=1)
        b = select(tables, "random", seed=1)
        c = select(tables, "random", seed=2)
        np.testing.assert_array_equal(a.currency, b.currency)
        self.assertFalse(np.array_equal(a.currency, c.currency))

    def test_random_assignment_uses_only_eligible_currencies(self):
        tables = make_tables(60)
        tables.score[:, :4] = np.nan
        self.assertTrue(np.all(select(tables, "random", seed=0).currency >= 4))

    def test_a_leg_with_no_exit_is_dropped_but_the_rest_of_the_cohort_remains(self):
        tables = small_tables()
        tables.gross_long[0, 7] = np.nan
        tables.gross_short[0, 7] = np.nan
        sel = select(tables, "momentum")
        self.assertEqual(int((sel.cohort == 0).sum()), 2 * K - 1)
        self.assertEqual(int((sel.cohort == 1).sum()), 2 * K)

    def test_ties_are_broken_by_currency_order(self):
        tables = small_tables()
        tables.score[:] = 1.0                                        # everyone equal
        sel = select(tables, "momentum")
        first = sel.cohort == 0
        # the lowest three columns tie for the bottom, and the highest three for the top
        self.assertEqual(sorted(sel.currency[first & (sel.direction < 0)]), [0, 1, 2])
        self.assertEqual(sorted(sel.currency[first & (sel.direction > 0)]), [5, 6, 7])

    def test_gross_comes_from_the_leg_direction(self):
        sel = select(small_tables(), "momentum")
        long_ = (sel.cohort == 0) & (sel.direction > 0)
        for j, g in zip(sel.currency[long_], sel.gross[long_]):
            self.assertAlmostEqual(g, j / 10)
        short_ = (sel.cohort == 0) & (sel.direction < 0)
        for j, g in zip(sel.currency[short_], sel.gross[short_]):
            self.assertAlmostEqual(g, -j / 10)

    def test_net_is_gross_minus_cost(self):
        sel = select(small_tables(), "momentum")
        np.testing.assert_allclose(sel.net, sel.gross - 0.05)

    def test_a_cohort_with_too_few_currencies_gets_no_legs(self):
        tables = small_tables()
        tables.score[1, :5] = np.nan                                 # only 3 eligible in cohort 1
        self.assertEqual(int((select(tables, "momentum").cohort == 1).sum()), 0)

    def test_an_unknown_rule_is_rejected(self):
        with self.assertRaises(ValueError):
            select(small_tables(), "vibes")


class PValueTests(unittest.TestCase):
    def test_the_observed_counts_itself(self):
        null = np.linspace(-1, 1, 99)
        self.assertAlmostEqual(p_upper(5.0, null), 1 / 100)
        self.assertAlmostEqual(p_upper(-5.0, null), 1.0)

    def test_non_finite_replicates_are_ignored(self):
        self.assertAlmostEqual(p_upper(0.0, np.array([np.nan, -1.0, 1.0, 2.0])), 3 / 4)


class PermutationNullTests(unittest.TestCase):
    def test_it_is_centred_on_zero_when_long_and_short_results_mirror_each_other(self):
        null = permutation_null(make_tables(300, seed=1), replicates=2000, seed=0)
        self.assertAlmostEqual(float(null.gross.mean()), 0.0, delta=0.01)

    def test_its_spread_matches_what_independent_legs_would_give(self):
        tables = make_tables(300, seed=2)
        null = permutation_null(tables, replicates=3000, seed=0)
        expected = 1.2 / np.sqrt(6 * 300)
        self.assertAlmostEqual(float(null.gross.std()), expected, delta=0.25 * expected)

    def test_net_is_gross_less_the_cost_of_the_random_legs(self):
        tables = make_tables(200, seed=4)
        null = permutation_null(tables, replicates=500, seed=0)
        self.assertAlmostEqual(float((null.gross - null.net).mean()), 0.03, delta=0.005)

    def test_legs_with_no_exit_are_left_out_of_the_means(self):
        null = permutation_null(make_tables(200, seed=5, missing_exit=0.2), replicates=300, seed=0)
        self.assertTrue(np.isfinite(null.gross).all())

    def test_it_is_reproducible_from_the_seed(self):
        tables = make_tables(60)
        a = permutation_null(tables, 200, seed=7)
        b = permutation_null(tables, 200, seed=7)
        np.testing.assert_array_equal(a.gross, b.gross)


class CalibrationTests(unittest.TestCase):
    """A p-value is only worth reading if a rule with no skill gets p-values spread evenly over
    0-1, and a rule with real skill gets tiny ones."""

    def test_momentum_with_no_skill_gets_roughly_uniform_p_values(self):
        p = []
        for seed in range(60):
            tables = make_tables(200, seed=seed)
            null = permutation_null(tables, replicates=400, seed=seed + 1000)
            p.append(p_upper(select(tables, "momentum").gross.mean(), null.gross))
        p = np.array(p)
        self.assertTrue(0.35 < p.mean() < 0.65, p.mean())
        self.assertLessEqual((p < 0.10).sum(), 12)
        self.assertLessEqual((p < 0.05).sum(), 8)

    def test_momentum_that_really_works_is_detected(self):
        tables = make_tables(300, seed=9, skill=0.4)
        null = permutation_null(tables, replicates=2000, seed=0)
        self.assertLess(p_upper(select(tables, "momentum").gross.mean(), null.gross), 0.005)

    def test_reversal_on_data_where_momentum_works_is_significantly_worse_than_random(self):
        tables = make_tables(300, seed=9, skill=0.4)
        null = permutation_null(tables, replicates=2000, seed=0)
        self.assertGreater(p_upper(select(tables, "reversal").gross.mean(), null.gross), 0.99)

    def test_the_oracle_is_detected_on_pure_noise(self):
        tables = make_tables(300, seed=11)
        null = permutation_null(tables, replicates=3000, seed=0)
        self.assertLess(p_upper(select(tables, "oracle").gross.mean(), null.gross), 0.001)

    def test_random_assignments_are_uniformly_distributed_against_the_null(self):
        tables = make_tables(250, seed=13)
        null = permutation_null(tables, replicates=2000, seed=99)
        p = inference.control_p_values(tables, null, draws=200, seed=5)
        self.assertLessEqual(float((p < 0.05).mean()), 0.09)
        self.assertTrue(0.44 < p.mean() < 0.56, p.mean())


class BlockBootstrapTests(unittest.TestCase):
    def test_the_interval_brackets_the_mean_and_is_reproducible(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0.1, 1.0, 300)
        counts = np.full(300, 6.0)
        low, high = block_bootstrap_interval(values * 6, counts, block=4, replicates=3000, seed=1)
        self.assertLess(low, values.mean())
        self.assertGreater(high, values.mean())
        self.assertEqual((low, high), block_bootstrap_interval(values * 6, counts, 4, 3000, seed=1))

    def test_serial_dependence_widens_the_interval(self):
        # AR(1) cohort means: adjacent cohorts move together, as overlapping holds make them.
        rng = np.random.default_rng(3)
        noise = rng.normal(size=600)
        series = np.zeros(600)
        for i in range(1, 600):
            series[i] = 0.8 * series[i - 1] + noise[i]
        counts = np.full(600, 6.0)
        naive = block_bootstrap_interval(series * 6, counts, block=1, replicates=4000, seed=0)
        blocked = block_bootstrap_interval(series * 6, counts, block=12, replicates=4000, seed=0)
        self.assertGreater(blocked[1] - blocked[0], 1.5 * (naive[1] - naive[0]))

    def test_a_constant_series_has_a_zero_width_interval(self):
        low, high = block_bootstrap_interval(np.full(50, 3.0), np.full(50, 6.0), 4, 500, 0)
        self.assertAlmostEqual(low, 0.5)
        self.assertAlmostEqual(high, 0.5)

    def test_cohorts_with_more_legs_weigh_more(self):
        sums = np.array([10.0] * 20 + [0.0] * 20)          # the first half: 10 legs of 1R; the rest: 1 leg
        counts = np.array([10.0] * 20 + [10.0] * 20)
        low, high = block_bootstrap_interval(sums, counts, 1, 2000, 0)
        self.assertTrue(0.3 < low < high < 0.7)

    def test_no_cohorts_gives_nan(self):
        low, high = block_bootstrap_interval(np.array([]), np.array([]))
        self.assertTrue(np.isnan(low) and np.isnan(high))

    def test_a_block_longer_than_the_series_is_shortened_not_an_error(self):
        low, high = block_bootstrap_interval(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0]), 10, 200, 0)
        self.assertTrue(1.0 <= low <= high <= 3.0)


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.tables = make_tables(120, seed=21, start="2013-01-07")           # spans 2013 into 2015
        self.null = permutation_null(self.tables, replicates=300, seed=0)

    def test_the_summary_matches_the_selected_legs(self):
        sel = select(self.tables, "momentum")
        r = inference.evaluate("m", self.tables, sel, self.null, bootstrap_replicates=300)
        self.assertEqual(r.legs, len(sel))
        self.assertEqual(r.cohorts, 120)
        self.assertAlmostEqual(r.gross_mean, sel.gross.mean())
        self.assertAlmostEqual(r.net_mean, sel.net.mean())
        self.assertAlmostEqual(r.net_mean, r.gross_mean - r.cost_mean)
        self.assertLess(r.ci_low, r.net_mean)
        self.assertGreater(r.ci_high, r.net_mean)

    def test_halves_are_split_by_cohort_date_at_2016(self):
        tables = make_tables(240, seed=22, start="2013-01-07")                # 2013 to mid-2017
        null = permutation_null(tables, replicates=100, seed=0)
        sel = select(tables, "momentum")
        r = inference.evaluate("m", tables, sel, null, bootstrap_replicates=100)
        early = np.asarray(tables.cohorts[sel.cohort] < inference.SPLIT)
        self.assertEqual((r.early_legs, r.late_legs), (int(early.sum()), int((~early).sum())))
        self.assertGreater(r.early_legs, 0)
        self.assertGreater(r.late_legs, 0)
        self.assertAlmostEqual(r.early_net, sel.net[early].mean())
        self.assertAlmostEqual(r.late_net, sel.net[~early].mean())

    def test_long_and_short_legs_are_reported_separately(self):
        sel = select(self.tables, "momentum")
        r = inference.evaluate("m", self.tables, sel, self.null, bootstrap_replicates=100)
        self.assertAlmostEqual(r.long_gross, sel.gross[sel.direction > 0].mean())
        self.assertAlmostEqual(r.short_gross, sel.gross[sel.direction < 0].mean())

    def test_the_per_currency_table_covers_every_chosen_currency(self):
        sel = select(self.tables, "momentum")
        r = inference.evaluate("m", self.tables, sel, self.null, bootstrap_replicates=100)
        self.assertEqual(int(r.per_currency["legs"].sum()), len(sel))
        self.assertTrue(set(r.per_currency["currency"]) <= set(universe.CURRENCIES))

    def test_the_detectable_effect_is_the_null_spread_times_the_one_sided_5_percent_z(self):
        sel = select(self.tables, "momentum")
        r = inference.evaluate("m", self.tables, sel, self.null, bootstrap_replicates=100)
        self.assertAlmostEqual(r.detectable, 1.645 * r.null_sd)


if __name__ == "__main__":
    unittest.main()
