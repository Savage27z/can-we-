import unittest
from datetime import datetime, timedelta, timezone

from live.freshness import is_stale, market_open

MAX = timedelta(hours=2)


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


class FreshnessTests(unittest.TestCase):
    def test_a_day_old_snapshot_on_a_trading_day_is_stale(self):
        self.assertTrue(is_stale(utc(2026, 9, 22, 15), utc(2026, 9, 23, 15), MAX))

    def test_a_recent_snapshot_is_fresh(self):
        self.assertFalse(is_stale(utc(2026, 9, 22, 15), utc(2026, 9, 22, 16, 30), MAX))

    def test_the_weekend_closure_is_not_staleness(self):
        # Friday's last candle, checked on Saturday and late Sunday.
        self.assertFalse(is_stale(utc(2026, 9, 18, 21), utc(2026, 9, 19, 12), MAX))
        self.assertFalse(is_stale(utc(2026, 9, 18, 21), utc(2026, 9, 20, 20), MAX))

    def test_a_snapshot_still_old_well_after_the_reopen_is_stale(self):
        self.assertTrue(is_stale(utc(2026, 9, 18, 21), utc(2026, 9, 21, 2), MAX))

    def test_market_hours(self):
        self.assertTrue(market_open(utc(2026, 9, 21, 12)))    # Monday
        self.assertFalse(market_open(utc(2026, 9, 19, 12)))   # Saturday
        self.assertFalse(market_open(utc(2026, 9, 18, 22)))   # Friday after the close
        self.assertTrue(market_open(utc(2026, 9, 20, 23)))    # Sunday after the open
