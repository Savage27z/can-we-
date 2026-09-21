import unittest

import numpy as np

from research import multiple_testing as mt


class BenjaminiHochbergTests(unittest.TestCase):
    def test_a_worked_example(self):
        adjusted = mt.benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
        np.testing.assert_allclose(adjusted, [0.02, 0.04, 0.04, 0.02])

    def test_adjusted_values_are_never_below_the_raw_ones_or_above_one(self):
        p = np.random.default_rng(0).random(50)
        adjusted = mt.benjamini_hochberg(p)
        self.assertTrue((adjusted >= p - 1e-12).all())
        self.assertTrue((adjusted <= 1.0).all())

    def test_ordering_is_preserved(self):
        p = np.random.default_rng(1).random(30)
        adjusted = mt.benjamini_hochberg(p)
        self.assertTrue(np.all(np.diff(adjusted[np.argsort(p)]) >= -1e-12))

    def test_a_single_test_is_unchanged(self):
        np.testing.assert_allclose(mt.benjamini_hochberg([0.03]), [0.03])

    def test_nan_is_kept_and_does_not_count_as_a_test(self):
        adjusted = mt.benjamini_hochberg([0.01, np.nan, 0.04])
        self.assertTrue(np.isnan(adjusted[1]))
        np.testing.assert_allclose(adjusted[[0, 2]], [0.02, 0.04])     # m = 2

    def test_all_nan_or_empty(self):
        self.assertTrue(np.isnan(mt.benjamini_hochberg([np.nan, np.nan])).all())
        self.assertEqual(len(mt.benjamini_hochberg([])), 0)


class HolmTests(unittest.TestCase):
    def test_a_worked_example(self):
        np.testing.assert_allclose(mt.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])

    def test_holm_is_never_less_strict_than_benjamini_hochberg(self):
        p = np.random.default_rng(2).random(40) ** 2
        self.assertTrue((mt.holm(p) >= mt.benjamini_hochberg(p) - 1e-12).all())

    def test_the_smallest_p_is_multiplied_by_the_number_of_tests(self):
        self.assertAlmostEqual(mt.holm([0.001] + [0.9] * 9)[0], 0.01)

    def test_nan_is_kept(self):
        self.assertTrue(np.isnan(mt.holm([0.01, np.nan])[1]))


class MaxTTests(unittest.TestCase):
    def test_the_best_of_many_pays_a_price_for_being_the_best(self):
        rng = np.random.default_rng(3)
        null = rng.normal(size=(20, 4000))
        observed = np.zeros(20)
        observed[0] = 3.0                       # raw one-sided p about 0.00135
        adjusted = mt.max_t(observed, null)
        self.assertTrue(0.01 < adjusted[0] < 0.06, adjusted[0])       # about 20 x the raw p
        self.assertGreater(adjusted[1], 0.9)                          # a z of 0 is nothing special

    def test_a_z_beyond_every_null_maximum_gets_the_smallest_possible_p(self):
        null = np.random.default_rng(4).normal(size=(5, 999))
        self.assertAlmostEqual(mt.max_t([50.0, 0, 0, 0, 0], null)[0], 1 / 1000)

    def test_a_nan_observation_stays_nan(self):
        null = np.random.default_rng(5).normal(size=(3, 100))
        self.assertTrue(np.isnan(mt.max_t([np.nan, 1.0, 2.0], null)[0]))

    def test_nan_null_entries_are_ignored(self):
        null = np.random.default_rng(6).normal(size=(3, 100))
        null[0, :50] = np.nan
        self.assertTrue(np.isfinite(mt.max_t([0.5, 1.0, 2.0], null)).all())

    def test_the_shapes_must_agree(self):
        with self.assertRaises(ValueError):
            mt.max_t([1.0, 2.0], np.zeros((3, 10)))


if __name__ == "__main__":
    unittest.main()
