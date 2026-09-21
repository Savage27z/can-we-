import logging
import unittest
from datetime import datetime, timedelta, timezone

from narration.deepseek_client import DeepSeekAPIError
from tests.tg_helpers import blackout_news, make_state, setup
from tgbot import config
from tgbot.run_bot import RedactingFormatter, seconds_until_next_check
from live.plan import SEPARATOR
from tgbot.service import ReportService, fallback_text
from tgbot.wording import NOTICE


def body_of(text):
    """The structural read: what follows the trade plan and precedes the closing notice."""
    closing = SEPARATOR + NOTICE
    assert text.endswith(closing), "every report must end with the notice"
    return text.split(SEPARATOR, 1)[-1][:-len(closing)]


class Clock:
    def __init__(self):
        self.now = datetime(2026, 9, 22, 15, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


def make_service(narrate, clock=None):
    calls = {"refresh": 0, "compute": 0}

    def refresh(pair):
        calls["refresh"] += 1

    def compute(pair):
        calls["compute"] += 1
        return make_state([setup("live_trade")])

    calls["chart"] = 0

    def chart(state):
        calls["chart"] += 1
        return b"PNG-BYTES"

    return ReportService(refresh, compute, narrate, clock or Clock(), chart), calls


class ReportServiceTests(unittest.TestCase):
    def test_report_is_cached_within_the_ttl(self):
        clock = Clock()
        service, calls = make_service(lambda s: "NARRATED", clock)
        self.assertEqual(body_of(service.get_report("EUR_USD")), "NARRATED")
        clock.now += config.REPORT_CACHE_TTL - timedelta(seconds=1)
        service.get_report("EUR_USD")
        self.assertEqual((calls["refresh"], calls["compute"]), (1, 1))

    def test_report_is_rebuilt_after_the_ttl(self):
        clock = Clock()
        service, calls = make_service(lambda s: "NARRATED", clock)
        service.get_report("EUR_USD")
        clock.now += config.REPORT_CACHE_TTL + timedelta(seconds=1)
        service.get_report("EUR_USD")
        self.assertEqual(calls["refresh"], 2)

    def test_falls_back_to_plain_summary_when_narration_fails(self):
        def failing(state):
            raise DeepSeekAPIError("down")

        service, _ = make_service(failing)
        text = service.get_report("EUR_USD")
        self.assertIn("plain summary", text)
        self.assertIn("Signal price 1.14650", text)  # the plan survives the fallback

    def test_data_refresh_failure_propagates_and_is_not_cached(self):
        service = ReportService(
            refresh=lambda pair: (_ for _ in ()).throw(RuntimeError("oanda down")),
            compute=lambda pair: make_state(), narrate=lambda s: "x", clock=Clock(),
            chart=lambda s: None,
        )
        with self.assertRaises(RuntimeError):
            service.get_report("EUR_USD")
        with self.assertRaises(RuntimeError):
            service.get_report("EUR_USD")


class NarrationFailureTests(unittest.TestCase):
    def test_a_fallback_report_is_cached_only_briefly_so_narration_is_retried(self):
        clock = Clock()
        outcomes = ["fail", "ok"]

        def narrate(state):
            if outcomes.pop(0) == "fail":
                raise DeepSeekAPIError("transient")
            return "NARRATED"

        service, calls = make_service(narrate, clock)
        self.assertIn("plain summary", service.get_report("EUR_USD"))
        clock.now += config.FALLBACK_CACHE_TTL - timedelta(seconds=1)
        self.assertIn("plain summary", service.get_report("EUR_USD"))
        self.assertEqual(calls["refresh"], 1)  # still cached

        clock.now += timedelta(seconds=2)  # past the short fallback TTL
        self.assertEqual(body_of(service.get_report("EUR_USD")), "NARRATED")
        self.assertEqual(calls["refresh"], 2)

    def test_none_or_empty_narration_falls_back_instead_of_being_sent_or_cached(self):
        for bad in (None, "", "   \n"):
            service, _ = make_service(lambda s, bad=bad: bad)
            text = service.get_report("EUR_USD")
            self.assertIn("plain summary", text, repr(bad))

    def test_unexpected_narration_exceptions_also_fall_back(self):
        def broken(state):
            raise KeyError("model changed its response")

        service, _ = make_service(broken)
        self.assertIn("plain summary", service.get_report("EUR_USD"))


class AnalysisChartTests(unittest.TestCase):
    def test_analysis_carries_the_text_and_the_chart(self):
        service, _ = make_service(lambda s: "NARRATED")
        analysis = service.get_analysis("EUR_USD")
        self.assertEqual(body_of(analysis.text), "NARRATED")
        self.assertEqual(analysis.chart, b"PNG-BYTES")

    def test_text_and_chart_are_cached_together(self):
        clock = Clock()
        service, calls = make_service(lambda s: "NARRATED", clock)
        service.get_analysis("EUR_USD")
        clock.now += config.REPORT_CACHE_TTL - timedelta(seconds=1)
        service.get_analysis("EUR_USD")
        self.assertEqual((calls["refresh"], calls["chart"]), (1, 1))

    def test_a_chart_failure_costs_only_the_picture(self):
        def broken_chart(state):
            raise RuntimeError("font cache unwritable")

        service = ReportService(lambda p: None, lambda p: make_state(), lambda s: "NARRATED",
                                Clock(), broken_chart)
        analysis = service.get_analysis("EUR_USD")
        self.assertEqual(body_of(analysis.text), "NARRATED")
        self.assertIsNone(analysis.chart)

    def test_the_chart_is_drawn_from_the_same_state_as_the_report(self):
        seen = {}
        state = make_state()

        def narrate(s):
            seen["narrated"] = s
            return "x"

        def chart(s):
            seen["charted"] = s
            return b"PNG"

        service = ReportService(lambda p: None, lambda p: state, narrate, Clock(), chart)
        service.get_analysis("EUR_USD")
        self.assertIs(seen["narrated"], seen["charted"])

    def test_get_report_still_returns_just_the_text(self):
        service, _ = make_service(lambda s: "NARRATED")
        self.assertEqual(body_of(service.get_report("EUR_USD")), "NARRATED")

    def test_chart_for_is_safe_to_call_directly_for_alerts(self):
        service = ReportService(lambda p: None, lambda p: make_state(), lambda s: "x",
                                Clock(), lambda s: (_ for _ in ()).throw(ValueError("no candles")))
        self.assertIsNone(service.chart_for(make_state()))


class StaleDataTests(unittest.TestCase):
    def test_a_day_old_snapshot_is_never_offered_as_a_live_entry(self):
        clock = Clock()
        clock.now += timedelta(hours=24)   # the state's as_of is 24h behind the clock
        service, _ = make_service(lambda s: "NARRATED READ", clock)
        text = service.get_report("EUR_USD")
        self.assertIn("STALE DATA", text)
        self.assertNotIn("TRADE PLAN", text)
        self.assertNotIn("at market", text)
        self.assertNotIn("NARRATED READ", text)

    def test_a_fresh_report_states_the_snapshot_time(self):
        service, _ = make_service(lambda s: "NARRATED READ")
        self.assertIn("Data as of Tue 22 Sep 15:00 UTC", service.get_report("EUR_USD"))

    def test_every_report_links_to_the_live_tradingview_chart(self):
        from tgbot.wording import tradingview_url

        service, _ = make_service(lambda s: "NARRATED READ")
        self.assertIn("Live chart: " + tradingview_url("EUR_USD"), service.get_report("EUR_USD"))

    def test_a_stale_report_links_to_it_too(self):
        clock = Clock()
        clock.now += timedelta(hours=24)
        service, _ = make_service(lambda s: "x", clock)
        self.assertIn("tradingview.com/chart/?symbol=OANDA%3AEURUSD", service.get_report("EUR_USD"))

    def test_the_link_uses_the_oanda_feed_and_the_pairs_symbol(self):
        from tgbot.wording import tradingview_url

        self.assertEqual(tradingview_url("USD_JPY"),
                         "https://www.tradingview.com/chart/?symbol=OANDA%3AUSDJPY&interval=60")

    def test_a_stale_report_is_retried_soon(self):
        clock = Clock()
        clock.now += timedelta(hours=24)
        service, calls = make_service(lambda s: "x", clock)
        service.get_report("EUR_USD")
        clock.now += config.FALLBACK_CACHE_TTL + timedelta(seconds=1)
        service.get_report("EUR_USD")
        self.assertEqual(calls["compute"], 2)


class TradePlanInReportTests(unittest.TestCase):
    def test_the_plan_comes_first_then_the_structural_read(self):
        service, _ = make_service(lambda s: "NARRATED READ")
        text = service.get_report("EUR_USD")
        self.assertTrue(text.startswith("🕒 Data as of"))
        self.assertTrue(body_of(text) == "NARRATED READ")
        self.assertLess(text.index("Data as of"), text.index("TRADE PLAN"))
        self.assertIn("Stop-loss: 1.14500", text)
        self.assertIn("Take-profit: 1.15000", text)
        self.assertEqual(body_of(text), "NARRATED READ")
        self.assertLess(text.index("TRADE PLAN"), text.index("NARRATED READ"))

    def test_the_plan_survives_a_narration_failure(self):
        def failing(state):
            raise DeepSeekAPIError("down")

        service, _ = make_service(failing)
        text = service.get_report("EUR_USD")
        self.assertIn("plain summary", text)
        self.assertIn("Stop-loss: 1.14500", text)

    def test_a_plan_rendering_bug_costs_only_the_plan(self):
        from unittest.mock import patch

        service, _ = make_service(lambda s: "NARRATED READ")
        with patch("tgbot.service.plan_text", side_effect=RuntimeError("bug")):
            text = service.get_report("EUR_USD")
        self.assertNotIn("TRADE PLAN", text)
        self.assertTrue(text.endswith("NARRATED READ" + SEPARATOR + NOTICE))   # no plan, but read and notice

    def test_alerts_render_the_same_report_including_the_plan(self):
        service, _ = make_service(lambda s: "NARRATED READ")
        self.assertIn("TRADE PLAN", service.render(make_state([setup("live_trade")])))

    def test_no_setup_still_says_there_is_nothing_to_enter(self):
        service = ReportService(lambda p: None, lambda p: make_state(), lambda s: "READ",
                                Clock(), lambda s: None)
        text = service.get_report("EUR_USD")
        self.assertIn("nothing to enter", text)
        self.assertEqual(body_of(text), "READ")


class NoticeTests(unittest.TestCase):
    """Every report says what it is: an analysis of fixed rules, not a recommendation."""

    def report(self, narrate, **kwargs):
        service, _ = make_service(narrate)
        return service.get_report("EUR_USD")

    def test_a_narrated_report_ends_with_the_notice_exactly_once(self):
        text = self.report(lambda s: "NARRATED")
        self.assertTrue(text.endswith(NOTICE))
        self.assertEqual(text.count(NOTICE), 1)

    def test_the_fallback_report_carries_it_too(self):
        def failing(state):
            raise DeepSeekAPIError("down")

        text = self.report(failing)
        self.assertIn("plain summary", text)
        self.assertTrue(text.endswith(NOTICE))

    def test_it_is_added_by_the_service_not_left_to_the_narration_model(self):
        # Whatever the model returns, and even if it returns nothing usable, the notice is there.
        for narrated in ("NARRATED", "", None):
            self.assertTrue(self.report(lambda s, n=narrated: n).endswith(NOTICE), repr(narrated))

    def test_the_alert_path_renders_the_same_report_with_the_notice(self):
        service, _ = make_service(lambda s: "NARRATED")
        self.assertTrue(service.render(make_state([setup("live_trade")])).endswith(NOTICE))

    def test_what_it_says_stays_within_what_the_research_supports(self):
        # It must not read as an endorsement, and it must not overclaim a result either way.
        for required in ("not a recommendation", "did not beat random entries",
                         "too short to tell skill from luck", "Not financial advice"):
            self.assertIn(required, NOTICE)
        for forbidden in ("profitable", "proven", "guaranteed", "win rate", "edge"):
            self.assertNotIn(forbidden, NOTICE.lower())

    def test_a_long_report_with_the_notice_still_fits_telegrams_chunks(self):
        from tgbot.messages import split_message

        text = self.report(lambda s: "\n".join(f"line {i}" for i in range(900)))
        chunks = split_message(text)
        self.assertTrue(all(len(c) <= 4000 for c in chunks))
        self.assertIn(NOTICE, chunks[-1])


class RefreshSerialisationTests(unittest.TestCase):
    def test_concurrent_fresh_state_calls_never_refresh_the_same_pair_at_once(self):
        import threading
        import time

        active = {"now": 0, "max": 0}
        guard = threading.Lock()

        def refresh(pair):
            with guard:
                active["now"] += 1
                active["max"] = max(active["max"], active["now"])
            time.sleep(0.05)
            with guard:
                active["now"] -= 1

        service = ReportService(refresh, lambda pair: make_state(), lambda s: "x", Clock(),
                                lambda s: None)
        threads = [threading.Thread(target=service.fresh_state, args=("EUR_USD",)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(active["max"], 1)

    def test_get_report_calling_fresh_state_does_not_deadlock(self):
        service, calls = make_service(lambda s: "NARRATED")
        self.assertEqual(body_of(service.get_report("EUR_USD")), "NARRATED")  # would hang if not re-entrant
        self.assertEqual(calls["refresh"], 1)


class BuildApplicationTests(unittest.TestCase):
    def test_registers_every_command_and_both_scheduled_jobs(self):
        from tgbot.run_bot import build_application

        app = build_application("123456:TEST-TOKEN")
        commands = set()
        for handler in app.handlers[0]:
            commands.update(getattr(handler, "commands", ()))
        self.assertEqual(commands, {"start", "help", "status", "analysis", "analyze"})
        self.assertEqual(len(app.job_queue.jobs()), 2)  # startup check + hourly check
        self.assertIn("service", app.bot_data)
        self.assertIn("alert_log", app.bot_data)


class FallbackTextTests(unittest.TestCase):
    def test_the_sweep_time_reads_as_a_date_not_an_iso_string(self):
        text = fallback_text(make_state([setup("pending_confirmation",
                                               sweep_time="2026-09-22T09:00:00+00:00")]))
        self.assertIn("(Tue 22 Sep 09:00 UTC)", text)
        self.assertNotIn("2026-09-22T", text)

    def test_no_setup(self):
        self.assertIn("No active setup", fallback_text(make_state()))

    def test_blackout_is_shown(self):
        text = fallback_text(make_state([setup("live_trade")], news=blackout_news()))
        self.assertIn("News blackout: USD FOMC Statement", text)


class RunnerHelperTests(unittest.TestCase):
    def test_next_check_is_five_past_the_hour(self):
        now = datetime(2026, 9, 22, 15, 2, 30, tzinfo=timezone.utc)
        self.assertEqual(seconds_until_next_check(now), 150)

    def test_next_check_rolls_to_the_next_hour(self):
        now = datetime(2026, 9, 22, 15, 5, 0, tzinfo=timezone.utc)
        self.assertEqual(seconds_until_next_check(now), 3600)

    def test_token_is_redacted_from_formatted_logs_including_tracebacks(self):
        formatter = RedactingFormatter("SECRET-TOKEN", "%(message)s")
        try:
            raise RuntimeError("failed calling https://api.telegram.org/botSECRET-TOKEN/getUpdates")
        except RuntimeError:
            import sys
            record = logging.LogRecord("x", logging.ERROR, __file__, 1,
                                       "url botSECRET-TOKEN/send", None, sys.exc_info())
        output = formatter.format(record)
        self.assertNotIn("SECRET-TOKEN", output)
        self.assertIn("<token>", output)


if __name__ == "__main__":
    unittest.main()
