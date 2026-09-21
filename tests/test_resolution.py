import unittest

from backtest.resolution import resolve_on_closes


class BullishResolutionTests(unittest.TestCase):
    def resolve(self, closes, start=0, invalidation=1.09, target=1.15):
        return resolve_on_closes("bullish", closes, start, invalidation, target)

    def test_the_first_close_reaching_the_target_wins(self):
        self.assertEqual(self.resolve([1.10, 1.12, 1.16, 1.17]), (2, "target"))

    def test_a_close_below_the_invalidation_level_loses(self):
        self.assertEqual(self.resolve([1.10, 1.08, 1.16]), (1, "invalidation"))

    def test_a_close_exactly_on_the_target_counts(self):
        self.assertEqual(self.resolve([1.10, 1.15]), (1, "target"))

    def test_a_close_exactly_on_the_invalidation_level_does_not_invalidate(self):
        self.assertIsNone(self.resolve([1.10, 1.09, 1.10]))

    def test_neither_by_the_end_of_the_data_is_unresolved(self):
        self.assertIsNone(self.resolve([1.10, 1.11, 1.12]))

    def test_closes_before_the_start_index_are_ignored(self):
        self.assertEqual(self.resolve([1.08, 1.10, 1.16], start=1), (2, "target"))

    def test_start_beyond_the_data_is_unresolved(self):
        self.assertIsNone(self.resolve([1.10], start=5))

    def test_invalidation_is_checked_before_the_target(self):
        # A target below the invalidation level makes one close satisfy both.
        self.assertEqual(self.resolve([1.05], invalidation=1.09, target=1.02),
                         (0, "invalidation"))


class BearishResolutionTests(unittest.TestCase):
    def resolve(self, closes, invalidation=1.16, target=1.10):
        return resolve_on_closes("bearish", closes, 0, invalidation, target)

    def test_the_first_close_reaching_the_target_wins(self):
        self.assertEqual(self.resolve([1.15, 1.12, 1.09]), (2, "target"))

    def test_a_close_above_the_invalidation_level_loses(self):
        self.assertEqual(self.resolve([1.15, 1.17, 1.09]), (1, "invalidation"))

    def test_a_close_exactly_on_the_target_counts(self):
        self.assertEqual(self.resolve([1.12, 1.10]), (1, "target"))

    def test_a_close_exactly_on_the_invalidation_level_does_not_invalidate(self):
        self.assertIsNone(self.resolve([1.15, 1.16, 1.15]))


class DirectionTests(unittest.TestCase):
    def test_an_unknown_direction_is_an_error_not_a_silent_loss(self):
        with self.assertRaises(ValueError):
            resolve_on_closes("sideways", [1.0], 0, 1.0, 2.0)


if __name__ == "__main__":
    unittest.main()
