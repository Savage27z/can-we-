import logging
import unittest
from datetime import datetime, timedelta, timezone

from narration.deepseek_client import DeepSeekAPIError
from tests.tg_helpers import blackout_news, make_state, setup
from tgbot import config
from tgbot.run_bot import RedactingFormatter, seconds_until_next_check
from tgbot.service import ReportService, fallback_text


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

    return ReportService(refresh, compute, narrate, clock or Clock()), calls


class ReportServiceTests(unittest.TestCase):
    def test_report_is_cached_within_the_ttl(self):
        clock = Clock()
        service, calls = make_service(lambda s: "NARRATED", clock)
        self.assertEqual(service.get_report("EUR_USD"), "NARRATED")
        clock.now += config.REPORT_CACHE_TTL - timedelta(seconds=1)
        service.get_report("EUR_USD")
        self.assertEqual(calls, {"refresh": 1, "compute": 1})

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
        self.assertIn("Entry 1.1465", text)

    def test_data_refresh_failure_propagates_and_is_not_cached(self):
        service = ReportService(
            refresh=lambda pair: (_ for _ in ()).throw(RuntimeError("oanda down")),
            compute=lambda pair: make_state(), narrate=lambda s: "x", clock=Clock(),
        )
        with self.assertRaises(RuntimeError):
            service.get_report("EUR_USD")
        with self.assertRaises(RuntimeError):
            service.get_report("EUR_USD")


class FallbackTextTests(unittest.TestCase):
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
