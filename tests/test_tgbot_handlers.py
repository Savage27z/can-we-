import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tgbot import handlers
from tgbot.pairs import normalize_pair


def make_update(chat_id=111):
    message = SimpleNamespace(reply_text=AsyncMock(), reply_html=AsyncMock())
    return SimpleNamespace(effective_chat=SimpleNamespace(id=chat_id), message=message)


def make_context(args=None, get_report=None):
    service = SimpleNamespace(get_report=get_report or (lambda pair: f"REPORT for {pair}"))
    app = SimpleNamespace(bot_data={"service": service})
    return SimpleNamespace(args=args or [], application=app)


def run(coro):
    return asyncio.run(coro)


def replies(update):
    return [c.args[0] for c in update.message.reply_text.await_args_list]


class NormalizePairTests(unittest.TestCase):
    def test_accepted_spellings(self):
        for raw in ["EUR_USD", "eurusd", "EUR/USD", "$EURUSD", " eur-usd "]:
            self.assertEqual(normalize_pair(raw), "EUR_USD", raw)

    def test_rejected_input(self):
        for raw in ["", "EUR", "EURUSDX", "../../x", "EUR_US1", "banana", "USDUSD"]:
            self.assertIsNone(normalize_pair(raw), raw)


class SplitMessageTests(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(handlers.split_message("a\nb"), ["a\nb"])

    def test_long_text_splits_on_line_boundaries_within_the_limit(self):
        text = "\n".join(["x" * 50] * 100)
        chunks = handlers.split_message(text, limit=500)
        self.assertTrue(all(len(c) <= 500 for c in chunks))
        self.assertEqual("\n".join(chunks), text)

    def test_single_oversized_line_is_hard_split(self):
        chunks = handlers.split_message("y" * 1200, limit=500)
        self.assertEqual([len(c) for c in chunks], [500, 500, 200])


class AuthorizationTests(unittest.TestCase):
    def test_empty_allowlist_reveals_chat_id_and_does_nothing_else(self):
        update = make_update(chat_id=777)
        with patch.dict("os.environ", {"TELEGRAM_ALLOWED_CHAT_IDS": ""}):
            run(handlers.analysis(update, make_context()))
        self.assertIn("777", replies(update)[0])
        self.assertIn("TELEGRAM_ALLOWED_CHAT_IDS=777", replies(update)[0])

    def test_unlisted_chat_is_refused_and_never_triggers_a_report(self):
        update = make_update(chat_id=999)
        called = []
        ctx = make_context(get_report=lambda pair: called.append(pair) or "x")
        with patch.dict("os.environ", {"TELEGRAM_ALLOWED_CHAT_IDS": "111"}):
            run(handlers.analysis(update, ctx))
        self.assertEqual(replies(update), ["This bot is private."])
        self.assertEqual(called, [])


class StartTests(unittest.TestCase):
    def test_start_text_lists_only_enabled_pairs_and_the_disclaimer(self):
        text = handlers.start_text()
        self.assertIn("FOREX STRUCTURE SCANNER", text)
        self.assertIn("<code>/analysis $EURUSD</code>", text)
        self.assertIn("• $EURUSD — Forex", text)
        self.assertNotIn("GBPUSD", text)
        self.assertIn("not financial advice", text)

    def test_authorized_start_replies_with_the_html_welcome(self):
        update = make_update()
        with patch.dict("os.environ", {"TELEGRAM_ALLOWED_CHAT_IDS": "111"}):
            run(handlers.start(update, make_context()))
        update.message.reply_html.assert_awaited_once_with(handlers.start_text())

    def test_unauthorized_start_gets_no_welcome(self):
        update = make_update(chat_id=999)
        with patch.dict("os.environ", {"TELEGRAM_ALLOWED_CHAT_IDS": "111"}):
            run(handlers.start(update, make_context()))
        update.message.reply_html.assert_not_awaited()
        self.assertEqual(replies(update), ["This bot is private."])


class ProfileTextTests(unittest.TestCase):
    def test_texts_fit_telegrams_limits(self):
        from tgbot import set_profile
        self.assertLessEqual(len(set_profile.DESCRIPTION), 512)
        self.assertLessEqual(len(set_profile.SHORT_DESCRIPTION), 120)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict("os.environ", {"TELEGRAM_ALLOWED_CHAT_IDS": "111"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_defaults_to_eur_usd(self):
        update = make_update()
        run(handlers.analysis(update, make_context()))
        self.assertEqual(replies(update), ["REPORT for EUR_USD"])

    def test_accepts_flexible_pair_spelling(self):
        update = make_update()
        run(handlers.analysis(update, make_context(args=["$eurusd"])))
        self.assertEqual(replies(update), ["REPORT for EUR_USD"])

    def test_unreadable_pair(self):
        update = make_update()
        run(handlers.analysis(update, make_context(args=["banana"])))
        self.assertIn("Couldn't read", replies(update)[0])

    def test_pair_that_failed_the_backtest_gate_is_refused(self):
        update = make_update()
        called = []
        ctx = make_context(args=["GBPUSD"], get_report=lambda p: called.append(p) or "x")
        run(handlers.analysis(update, ctx))
        self.assertIn("isn't enabled", replies(update)[0])
        self.assertEqual(called, [])

    def test_service_failure_gets_a_generic_message_without_details(self):
        def boom(pair):
            raise RuntimeError("secret internal detail")

        update = make_update()
        run(handlers.analysis(update, make_context(get_report=boom)))
        self.assertIn("Couldn't build the report", replies(update)[0])
        self.assertNotIn("secret", replies(update)[0])


if __name__ == "__main__":
    unittest.main()
