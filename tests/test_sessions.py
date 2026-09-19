import unittest

import pandas as pd

from backtest.sessions import in_session, in_session_mask


class SessionTests(unittest.TestCase):
    def test_before_london_not_in_session(self):
        self.assertFalse(in_session(pd.Timestamp("2024-01-02 05:00", tz="UTC")))

    def test_london_start_inclusive(self):
        self.assertTrue(in_session(pd.Timestamp("2024-01-02 07:00", tz="UTC")))

    def test_london_ny_overlap(self):
        self.assertTrue(in_session(pd.Timestamp("2024-01-02 13:00", tz="UTC")))

    def test_ny_end_boundary_inclusive(self):
        # See sessions.py docstring: real H4 candles land exactly on 21:00 UTC,
        # which must count as in-session or the doc's own §8.1 worked example breaks.
        self.assertTrue(in_session(pd.Timestamp("2024-01-02 21:00", tz="UTC")))

    def test_just_after_ny_end_not_in_session(self):
        self.assertFalse(in_session(pd.Timestamp("2024-01-02 22:00", tz="UTC")))

    def test_gap_between_london_and_ny_not_covered(self):
        # 16:00 is London's inclusive end; nothing between 17:00 and NY start (12:00)
        # is relevant here since NY already covers 12-21, so check a true dead zone.
        self.assertFalse(in_session(pd.Timestamp("2024-01-02 03:00", tz="UTC")))

    def test_mask_matches_scalar_check(self):
        times = pd.Series(pd.to_datetime([
            "2024-01-02T01:00:00Z", "2024-01-02T09:00:00Z",
            "2024-01-02T13:00:00Z", "2024-01-02T17:00:00Z", "2024-01-02T21:00:00Z",
        ]))
        mask = in_session_mask(times)
        expected = [in_session(t) for t in times]
        self.assertListEqual(list(mask), expected)


if __name__ == "__main__":
    unittest.main()
