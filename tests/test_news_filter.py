import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from news import filter as nf
from news.feed import Calendar, CalendarEvent, NewsFeedError

T0 = datetime(2026, 9, 16, 18, 0, tzinfo=timezone.utc)  # FOMC statement time


def ev(title, currency, offset_minutes=0, impact="High"):
    return CalendarEvent(title, currency, T0 + timedelta(minutes=offset_minutes), impact)


def cal(*events):
    return Calendar(events=list(events), fetched_at=T0 - timedelta(hours=1))


class BlackoutWindowTests(unittest.TestCase):
    def setUp(self):
        self.calendar = cal(ev("FOMC Statement", "USD"))

    def test_blackout_at_the_exact_window_edges(self):
        before_edge = T0 - nf.BLACKOUT_BEFORE
        after_edge = T0 + nf.BLACKOUT_AFTER
        self.assertEqual(nf.evaluate(self.calendar, "EUR_USD", before_edge).status, "blackout")
        self.assertEqual(nf.evaluate(self.calendar, "EUR_USD", after_edge).status, "blackout")

    def test_clear_just_outside_the_window(self):
        just_before = T0 - nf.BLACKOUT_BEFORE - timedelta(seconds=1)
        just_after = T0 + nf.BLACKOUT_AFTER + timedelta(seconds=1)
        self.assertEqual(nf.evaluate(self.calendar, "EUR_USD", just_before).status, "clear")
        self.assertEqual(nf.evaluate(self.calendar, "EUR_USD", just_after).status, "clear")

    def test_blackout_event_carries_precomputed_when_text(self):
        status = nf.evaluate(self.calendar, "EUR_USD", T0 - timedelta(minutes=25))
        self.assertEqual(len(status.blackout_events), 1)
        self.assertEqual(status.blackout_events[0].when, "in 25m")
        self.assertEqual(status.blackout_events[0].currency, "USD")

    def test_clustered_events_merge_into_one_continuous_blackout(self):
        # Statement + press conference 30 min apart: still blacked out between them.
        calendar = cal(ev("FOMC Statement", "USD"), ev("FOMC Press Conference", "USD", 30))
        status = nf.evaluate(calendar, "EUR_USD", T0 + timedelta(minutes=15))
        self.assertEqual(status.status, "blackout")
        self.assertEqual(len(status.blackout_events), 2)


class RelevanceTests(unittest.TestCase):
    def test_only_high_impact_counts(self):
        calendar = cal(ev("Retail Sales", "USD", impact="Medium"),
                       ev("Minor Speech", "USD", impact="Low"),
                       ev("Anchor", "USD", 600))  # keeps coverage wide enough
        self.assertEqual(nf.evaluate(calendar, "EUR_USD", T0).status, "clear")

    def test_other_currencies_are_ignored(self):
        calendar = cal(ev("CPI m/m", "CAD"), ev("Anchor", "USD", 600))
        self.assertEqual(nf.evaluate(calendar, "EUR_USD", T0).status, "clear")

    def test_both_sides_of_the_pair_are_relevant(self):
        calendar = cal(ev("ECB Rate Decision", "EUR"))
        self.assertEqual(nf.evaluate(calendar, "EUR_USD", T0).status, "blackout")

    def test_all_currencies_high_impact_event_is_relevant(self):
        calendar = cal(ev("Global Event", "All"))
        self.assertEqual(nf.evaluate(calendar, "EUR_USD", T0).status, "blackout")

    def test_invalid_pair_raises(self):
        with self.assertRaises(ValueError):
            nf.relevant_currencies("../../x")


class UpcomingTests(unittest.TestCase):
    def test_upcoming_excludes_events_already_in_blackout(self):
        calendar = cal(ev("Soon", "USD"), ev("Later", "USD", 600))
        status = nf.evaluate(calendar, "EUR_USD", T0 - timedelta(minutes=30))
        self.assertEqual([e.title for e in status.blackout_events], ["Soon"])
        self.assertEqual([e.title for e in status.upcoming_events], ["Later"])

    def test_upcoming_respects_horizon_and_cap(self):
        events = [ev(f"E{i}", "USD", 120 * (i + 1)) for i in range(8)]
        events.append(ev("Far", "USD", int(nf.UPCOMING_HORIZON.total_seconds() / 60) + 600))
        status = nf.evaluate(cal(*events), "EUR_USD", T0)
        self.assertEqual(len(status.upcoming_events), nf.MAX_UPCOMING)
        self.assertNotIn("Far", [e.title for e in status.upcoming_events])

    def test_past_events_are_not_upcoming(self):
        calendar = cal(ev("Old", "USD", -600), ev("Anchor", "USD", 600))
        status = nf.evaluate(calendar, "EUR_USD", T0)
        self.assertEqual([e.title for e in status.upcoming_events], ["Anchor"])


class CoverageTests(unittest.TestCase):
    def test_now_far_outside_the_feed_range_is_unavailable_not_clear(self):
        calendar = cal(ev("FOMC Statement", "USD"))
        now = T0 + timedelta(days=9)
        calendar.fetched_at = now - timedelta(hours=1)  # freshly fetched, but last week's feed
        status = nf.evaluate(calendar, "EUR_USD", now)
        self.assertEqual(status.status, "unavailable")
        self.assertIn("rolled over", status.reason)

    def test_coverage_edge_is_exactly_one_day_past_the_last_event(self):
        calendar = cal(ev("FOMC Statement", "USD"))
        edge = T0 + nf.COVERAGE_TOLERANCE
        calendar.fetched_at = edge - timedelta(hours=1)
        self.assertEqual(nf.evaluate(calendar, "EUR_USD", edge).status, "clear")
        calendar.fetched_at = edge + timedelta(seconds=1) - timedelta(hours=1)
        self.assertEqual(nf.evaluate(calendar, "EUR_USD", edge + timedelta(seconds=1)).status,
                         "unavailable")

    def test_empty_calendar_is_unavailable(self):
        self.assertEqual(nf.evaluate(cal(), "EUR_USD", T0).status, "unavailable")


class StaleCalendarTests(unittest.TestCase):
    def test_a_long_stale_cached_calendar_is_reported_unavailable_not_trusted(self):
        # Refreshes have been failing and an old cache is being served.
        calendar = Calendar(events=[ev("FOMC Statement", "USD")],
                            fetched_at=T0 - nf.MAX_CALENDAR_AGE - timedelta(hours=1))
        status = nf.evaluate(calendar, "EUR_USD", T0)
        self.assertEqual(status.status, "unavailable")
        self.assertIn("old", status.reason)

    def test_a_calendar_just_inside_the_age_limit_is_used(self):
        calendar = Calendar(events=[ev("FOMC Statement", "USD")],
                            fetched_at=T0 - nf.MAX_CALENDAR_AGE + timedelta(minutes=1))
        self.assertEqual(nf.evaluate(calendar, "EUR_USD", T0).status, "blackout")


class CheckNewsTests(unittest.TestCase):
    def test_feed_failure_becomes_unavailable(self):
        with patch.object(nf, "load_calendar", side_effect=NewsFeedError("boom")):
            status = nf.check_news("EUR_USD", now=T0)
        self.assertEqual(status.status, "unavailable")
        self.assertIn("boom", status.reason)

    def test_an_unexpected_exception_is_contained_as_unavailable(self):
        # News is an overlay: an OSError/OverflowError/TypeError inside it must not
        # take down the report or alert that called it.
        for error in (OSError("disk"), OverflowError("date"), TypeError("naive")):
            with patch.object(nf, "load_calendar", side_effect=error):
                status = nf.check_news("EUR_USD", now=T0)
            self.assertEqual(status.status, "unavailable", repr(error))
            self.assertIn(type(error).__name__, status.reason)

    def test_invalid_pair_is_rejected_before_any_fetch(self):
        with patch.object(nf, "load_calendar") as fetch:
            with self.assertRaises(ValueError):
                nf.check_news("nope", now=T0)
        fetch.assert_not_called()

    def test_not_checked_status(self):
        self.assertEqual(nf.NewsStatus.not_checked().status, "not_checked")


class DescribeWhenTests(unittest.TestCase):
    def test_formats(self):
        d = timedelta
        self.assertEqual(nf.describe_when(d(minutes=45)), "in 45m")
        self.assertEqual(nf.describe_when(d(hours=2, minutes=15)), "in 2h 15m")
        self.assertEqual(nf.describe_when(d(hours=3)), "in 3h")
        self.assertEqual(nf.describe_when(-d(minutes=20)), "20m ago")
        self.assertEqual(nf.describe_when(d(seconds=10)), "now")


if __name__ == "__main__":
    unittest.main()
