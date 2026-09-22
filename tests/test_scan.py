import unittest
from datetime import timedelta
from unittest.mock import AsyncMock, patch

from dataclasses import replace

from tests.test_tgbot_handlers import make_context, make_update, replies, run
from tests.test_tgbot_service import Clock
from tests.tg_helpers import NOW, blackout_news, make_state, setup
from tgbot import config, handlers
from tgbot.scan import pair_line, scan_text
from tgbot.service import ReportService


def state_for(pair, setups=None, bias="neutral", **kwargs):
    return replace(make_state(setups, **kwargs), pair=pair, daily_bias=bias)


class PairLineTests(unittest.TestCase):
    def test_no_setup(self):
        line = pair_line(state_for("EUR_USD"), NOW)
        self.assertIn("EUR/USD", line)
        self.assertIn("no setup in play", line)

    def test_a_live_trade_shows_entry_stop_target_and_reward(self):
        line = pair_line(state_for("EUR_USD", [setup("live_trade")], "bullish"), NOW)
        self.assertIn("LIVE bullish", line)
        self.assertIn("entry 1.14650", line)
        self.assertIn("stop 1.14500", line)
        self.assertIn("target 1.15000", line)
        self.assertIn("2.3R", line)

    def test_a_forming_setup_says_what_is_needed_and_how_far_away(self):
        line = pair_line(state_for("EUR_USD", [setup("pending_confirmation")]), NOW)
        self.assertIn("needs an H1 close above 1.14600", line)
        self.assertRegex(line, r"\(\d+ pips from price\)")

    def test_a_sweep_waiting_for_a_gap(self):
        line = pair_line(state_for("EUR_USD", [setup("pending_fvg")]), NOW)
        self.assertIn("waiting for a gap", line)
        self.assertIn("Tue 22 Sep 09:00 UTC", line)

    def test_the_most_advanced_setup_is_shown_and_the_rest_counted(self):
        line = pair_line(state_for("EUR_USD", [
            setup("pending_fvg", sweep_time="2026-09-22T12:00:00+00:00"),
            setup("live_trade", sweep_time="2026-09-22T09:00:00+00:00")]), NOW)
        self.assertIn("LIVE", line)
        self.assertIn("(+1 more)", line)

    def test_stale_data_is_flagged_not_read(self):
        line = pair_line(state_for("EUR_USD", [setup("live_trade")]), NOW + timedelta(hours=24))
        self.assertIn("stale", line)
        self.assertNotIn("LIVE", line)

    def test_a_news_blackout_is_flagged(self):
        line = pair_line(state_for("EUR_USD", news=blackout_news()), NOW)
        self.assertIn("news blackout", line)

    def test_jpy_prices_use_three_decimals(self):
        s = setup("live_trade")
        line = pair_line(state_for("USD_JPY", [s]), NOW)
        self.assertIn("entry 1.147", line)


class ScanTextTests(unittest.TestCase):
    def results(self):
        return {"EUR_USD": state_for("EUR_USD"),
                "GBP_USD": state_for("GBP_USD", [setup("live_trade")], "bullish"),
                "USD_JPY": state_for("USD_JPY", [setup("pending_fvg")])}

    def test_pairs_with_something_in_play_come_first(self):
        text = scan_text(self.results(), NOW)
        self.assertLess(text.index("GBP/USD"), text.index("USD/JPY"))
        self.assertLess(text.index("USD/JPY"), text.index("EUR/USD"))
        self.assertIn("2 of 3 pairs have something in play", text)

    def test_when_nothing_is_in_play_it_says_so_plainly(self):
        text = scan_text({"EUR_USD": state_for("EUR_USD"), "GBP_USD": state_for("GBP_USD")}, NOW)
        self.assertIn("Nothing is in play on any pair", text)

    def test_a_pair_that_failed_to_refresh_is_named_not_hidden(self):
        text = scan_text({"EUR_USD": RuntimeError("oanda"), "GBP_USD": state_for("GBP_USD")}, NOW)
        self.assertIn("EUR/USD", text)
        self.assertIn("couldn't refresh", text)

    def test_it_says_it_is_not_a_signal_and_how_to_get_the_full_read(self):
        text = scan_text(self.results(), NOW)
        self.assertIn("not a trade signal", text)
        self.assertIn("/analysis", text)
        self.assertIn("Data as of", text)


class ServiceScanTests(unittest.TestCase):
    def make(self, refresh=None):
        calls = []

        def compute(pair):
            calls.append(pair)
            return state_for(pair)

        clock = Clock()
        return ReportService(refresh or (lambda pair: None), compute, lambda s: "x", clock,
                             lambda s: None), calls, clock

    def test_it_reads_every_enabled_pair(self):
        service, calls, _ = self.make()
        results = service.scan()
        self.assertEqual(list(results), list(config.LIVE_PAIRS))
        self.assertEqual(calls, list(config.LIVE_PAIRS))

    def test_a_second_scan_within_the_ttl_reuses_the_first(self):
        service, calls, clock = self.make()
        service.scan()
        clock.now += config.REPORT_CACHE_TTL - timedelta(seconds=1)
        service.scan()
        self.assertEqual(len(calls), len(config.LIVE_PAIRS))
        clock.now += timedelta(seconds=2)
        service.scan()
        self.assertEqual(len(calls), 2 * len(config.LIVE_PAIRS))

    def test_one_failing_pair_does_not_lose_the_others(self):
        def refresh(pair):
            if pair == "GBP_USD":
                raise RuntimeError("oanda 504")

        service, _, _ = self.make(refresh)
        results = service.scan()
        self.assertIsInstance(results["GBP_USD"], RuntimeError)
        self.assertNotIsInstance(results["EUR_USD"], Exception)

    def test_a_total_failure_is_not_cached(self):
        def refresh(pair):
            raise RuntimeError("down")

        service, calls, _ = self.make(refresh)
        service.scan()
        service.scan()
        self.assertEqual(service._scan_cache, None)


class ScanHandlerTests(unittest.TestCase):
    def patched(self):
        patcher = patch.object(config, "allowed_chat_ids", return_value={111})
        patcher.start()
        self.addCleanup(patcher.stop)

    def context_with(self, results, args=None):
        ctx = make_context(args=args)
        ctx.application.bot_data["service"].scan = lambda: results
        return ctx

    def test_scan_replies_with_the_overview(self):
        self.patched()
        update = make_update()
        run(handlers.scan(update, self.context_with({"EUR_USD": state_for("EUR_USD")})))
        self.assertIn("Market scan", replies(update)[0])

    def test_analysis_all_is_the_same_scan(self):
        self.patched()
        update = make_update()
        run(handlers.analysis(update, self.context_with({"EUR_USD": state_for("EUR_USD")}, ["all"])))
        self.assertIn("Market scan", replies(update)[0])

    def test_a_scan_that_blows_up_says_so_instead_of_going_silent(self):
        self.patched()
        update = make_update()
        ctx = make_context()

        def boom():
            raise RuntimeError("x")

        ctx.application.bot_data["service"].scan = boom
        run(handlers.scan(update, ctx))
        self.assertIn("Couldn't scan", replies(update)[0])

    def test_only_authorised_chats_can_scan(self):
        update = make_update(chat_id=999)
        with patch.object(config, "allowed_chat_ids", return_value={111}):
            run(handlers.scan(update, self.context_with({})))
        self.assertIn("private", replies(update)[0])

    def test_the_help_lists_scan(self):
        self.assertIn("/scan", handlers.HELP_TEXT)


if __name__ == "__main__":
    unittest.main()
