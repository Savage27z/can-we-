import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tgbot import handlers
from tgbot.messages import split_message
from tgbot.pairs import normalize_pair
from tgbot.service import Analysis


def make_update(chat_id=111, edited=False):
    """Mimics python-telegram-bot's Update. For a normal message `message` and
    `effective_message` are the same object; for an EDITED message Telegram sends
    `edited_message`, so `update.message` is None while `effective_message` is set."""
    message = SimpleNamespace(reply_text=AsyncMock(), reply_html=AsyncMock(),
                              reply_photo=AsyncMock())
    return SimpleNamespace(
        effective_chat=SimpleNamespace(id=chat_id),
        effective_message=message,
        message=None if edited else message,
    )


def make_context(args=None, get_report=None, bot_data=None, chart=None):
    report = get_report or (lambda pair: f"REPORT for {pair}")
    service = SimpleNamespace(get_analysis=lambda pair: Analysis(text=report(pair), chart=chart))
    app = SimpleNamespace(bot_data={"service": service, **(bot_data or {})})
    return SimpleNamespace(args=args or [], application=app)


def run(coro):
    return asyncio.run(coro)


def replies(update):
    return [c.args[0] for c in update.effective_message.reply_text.await_args_list]


class NormalizePairTests(unittest.TestCase):
    def test_accepted_spellings(self):
        for raw in ["EUR_USD", "eurusd", "EUR/USD", "$EURUSD", " eur-usd "]:
            self.assertEqual(normalize_pair(raw), "EUR_USD", raw)

    def test_rejected_input(self):
        for raw in ["", "EUR", "EURUSDX", "../../x", "EUR_US1", "banana", "USDUSD"]:
            self.assertIsNone(normalize_pair(raw), raw)


class SplitMessageTests(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(split_message("a\nb"), ["a\nb"])

    def test_long_text_splits_on_line_boundaries_within_the_limit(self):
        text = "\n".join(["x" * 50] * 100)
        chunks = split_message(text, limit=500)
        self.assertTrue(all(len(c) <= 500 for c in chunks))
        self.assertEqual("\n".join(chunks), text)

    def test_single_oversized_line_is_hard_split(self):
        chunks = split_message("y" * 1200, limit=500)
        self.assertEqual([len(c) for c in chunks], [500, 500, 200])

    def test_blank_text_yields_no_chunks(self):
        self.assertEqual(split_message(""), [])
        self.assertEqual(split_message("\n\n  \n"), [])


class StartTests(unittest.TestCase):
    def test_start_text_lists_only_enabled_pairs_and_the_disclaimer(self):
        text = handlers.start_text()
        self.assertIn("FOREX STRUCTURE SCANNER", text)
        self.assertIn("<code>/analysis $EURUSD</code>", text)
        self.assertIn("• $EURUSD — Forex", text)
        self.assertIn("• $GBPUSD — Forex", text)
        self.assertIn("• $USDJPY — Forex", text)
        self.assertIn("• $USDCHF — Forex", text)
        self.assertIn("• $USDCAD — Forex", text)
        self.assertIn("• $AUDUSD — Forex", text)
        self.assertIn("• $NZDUSD — Forex", text)
        self.assertNotIn("EURGBP", text)
        self.assertIn("not financial advice", text)
        self.assertIn("Analysis, not signals", text)

    def test_authorized_start_replies_with_the_html_welcome(self):
        update = make_update()
        with patch.dict("os.environ", {"TELEGRAM_ALLOWED_CHAT_IDS": "111"}):
            run(handlers.start(update, make_context()))
        update.effective_message.reply_html.assert_awaited_once_with(handlers.start_text())

    def test_unauthorized_start_gets_no_welcome(self):
        update = make_update(chat_id=999)
        with patch.dict("os.environ", {"TELEGRAM_ALLOWED_CHAT_IDS": "111"}):
            run(handlers.start(update, make_context()))
        update.effective_message.reply_html.assert_not_awaited()
        self.assertEqual(replies(update), ["This bot is private."])


class AuthorizationTests(unittest.TestCase):
    def test_empty_allowlist_reveals_chat_id_and_does_nothing_else(self):
        update = make_update(chat_id=777)
        with patch.dict("os.environ", {"TELEGRAM_ALLOWED_CHAT_IDS": ""}):
            run(handlers.analysis(update, make_context()))
        self.assertIn("TELEGRAM_ALLOWED_CHAT_IDS=777", replies(update)[0])

    def test_unlisted_chat_is_refused_and_never_triggers_a_report(self):
        update = make_update(chat_id=999)
        called = []
        ctx = make_context(get_report=lambda pair: called.append(pair) or "x")
        with patch.dict("os.environ", {"TELEGRAM_ALLOWED_CHAT_IDS": "111"}):
            run(handlers.analysis(update, ctx))
        self.assertEqual(replies(update), ["This bot is private."])
        self.assertEqual(called, [])

    def test_help_and_status_are_also_protected(self):
        for handler in (handlers.help_command, handlers.status):
            update = make_update(chat_id=999)
            with patch.dict("os.environ", {"TELEGRAM_ALLOWED_CHAT_IDS": "111"}):
                run(handler(update, make_context()))
            self.assertEqual(replies(update), ["This bot is private."], handler.__name__)


class EditedMessageTests(unittest.TestCase):
    """Telegram delivers an edited command as `edited_message`; `update.message` is
    None. The handlers used it and crashed AFTER doing the paid work."""

    def test_edited_command_from_an_authorized_chat_still_gets_its_reply(self):
        update = make_update(edited=True)
        with patch.dict("os.environ", {"TELEGRAM_ALLOWED_CHAT_IDS": "111"}):
            run(handlers.analysis(update, make_context()))
        self.assertEqual(replies(update), ["REPORT for EUR_USD"])

    def test_edited_command_from_a_stranger_gets_the_private_reply_not_a_crash(self):
        update = make_update(chat_id=999, edited=True)
        with patch.dict("os.environ", {"TELEGRAM_ALLOWED_CHAT_IDS": "111"}):
            run(handlers.start(update, make_context()))
        self.assertEqual(replies(update), ["This bot is private."])


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict("os.environ", {"TELEGRAM_ALLOWED_CHAT_IDS": "111"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_defaults_to_eur_usd(self):
        update = make_update()
        run(handlers.analysis(update, make_context()))
        self.assertEqual(replies(update), ["REPORT for EUR_USD"])

    def test_the_report_is_sent_without_a_link_preview_card(self):
        update = make_update()
        run(handlers.analysis(update, make_context()))
        for call in update.effective_message.reply_text.await_args_list:
            self.assertTrue(call.kwargs["link_preview_options"].is_disabled)

    def test_accepts_flexible_pair_spelling(self):
        update = make_update()
        run(handlers.analysis(update, make_context(args=["$eurusd"])))
        self.assertEqual(replies(update), ["REPORT for EUR_USD"])

    def test_unreadable_pair(self):
        update = make_update()
        run(handlers.analysis(update, make_context(args=["banana"])))
        self.assertIn("Couldn't read", replies(update)[0])

    def test_a_huge_argument_is_not_echoed_back_in_full(self):
        update = make_update()
        run(handlers.analysis(update, make_context(args=["z" * 5000])))
        self.assertLess(len(replies(update)[0]), 200)

    def test_a_pair_that_is_not_enabled_is_refused(self):
        update = make_update()
        called = []
        ctx = make_context(args=["EURGBP"], get_report=lambda p: called.append(p) or "x")
        run(handlers.analysis(update, ctx))
        self.assertIn("isn't enabled", replies(update)[0])
        self.assertEqual(called, [])

    def test_the_refusal_does_not_claim_a_backtest_endorsed_the_enabled_pair(self):
        # It used to say the enabled pair "passed the backtest gate". The validation research no
        # longer supports that (no enabled pair is shown to have an edge).
        update = make_update()
        run(handlers.analysis(update, make_context(args=["EURGBP"])))
        reply = replies(update)[0]
        self.assertEqual(
            reply,
            "EUR_GBP isn't enabled. This bot covers EUR_USD, GBP_USD, USD_JPY, USD_CHF, "
            "USD_CAD, AUD_USD, NZD_USD only.",
        )
        for claim in ("passed", "gate", "negative expectancy"):
            self.assertNotIn(claim, reply)

    def test_service_failure_gets_a_generic_message_without_details(self):
        def boom(pair):
            raise RuntimeError("secret internal detail")

        update = make_update()
        run(handlers.analysis(update, make_context(get_report=boom)))
        self.assertIn("Couldn't build the report", replies(update)[0])
        self.assertNotIn("secret", replies(update)[0])


class ChartDeliveryTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict("os.environ", {"TELEGRAM_ALLOWED_CHAT_IDS": "111"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_chart_is_sent_before_the_text(self):
        order = []
        update = make_update()
        update.effective_message.reply_photo.side_effect = lambda **kw: order.append("photo")
        update.effective_message.reply_text.side_effect = lambda text, **kwargs: order.append("text")
        run(handlers.analysis(update, make_context(chart=b"PNG")))
        self.assertEqual(order, ["photo", "text"])
        update.effective_message.reply_photo.assert_awaited_once_with(photo=b"PNG")

    def test_no_photo_is_sent_when_there_is_no_chart(self):
        update = make_update()
        run(handlers.analysis(update, make_context(chart=None)))
        update.effective_message.reply_photo.assert_not_awaited()
        self.assertEqual(replies(update), ["REPORT for EUR_USD"])

    def test_a_failed_photo_upload_still_delivers_the_text(self):
        update = make_update()
        update.effective_message.reply_photo.side_effect = RuntimeError("upload failed")
        run(handlers.analysis(update, make_context(chart=b"PNG")))
        self.assertEqual(replies(update), ["REPORT for EUR_USD"])


class StatusTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict("os.environ", {"TELEGRAM_ALLOWED_CHAT_IDS": "111"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_before_the_first_check(self):
        update = make_update()
        run(handlers.status(update, make_context()))
        self.assertIn("No alert check has completed yet", replies(update)[0])

    def test_healthy(self):
        health = {"last_cycle": "2026-09-20T01:05:00+00:00", "last_ok": "2026-09-20T01:05:00+00:00",
                  "consecutive_failures": 0, "last_error": None}
        update = make_update()
        run(handlers.status(update, make_context(bot_data={"alert_health": health})))
        self.assertIn("Healthy", replies(update)[0])

    def test_failing_shows_the_streak_and_last_error(self):
        health = {"last_cycle": "2026-09-20T01:05:00+00:00", "last_ok": None,
                  "consecutive_failures": 4, "last_error": "OandaAPIError: 401"}
        update = make_update()
        run(handlers.status(update, make_context(bot_data={"alert_health": health})))
        self.assertIn("4 consecutive", replies(update)[0])
        self.assertIn("OandaAPIError: 401", replies(update)[0])


class ProfileTextTests(unittest.TestCase):
    def test_texts_fit_telegrams_limits(self):
        from tgbot import set_profile
        self.assertLessEqual(len(set_profile.DESCRIPTION), 512)
        self.assertLessEqual(len(set_profile.SHORT_DESCRIPTION), 120)

    def test_the_profile_says_analysis_and_does_not_oversell(self):
        from tgbot import set_profile
        self.assertIn("not signals", set_profile.DESCRIPTION)
        self.assertIn("did not beat random entries", set_profile.DESCRIPTION)
        self.assertIn("Not financial advice", set_profile.DESCRIPTION)


if __name__ == "__main__":
    unittest.main()
