import json
import unittest
from unittest.mock import patch

from narration import chat_prompt
from tests.tg_helpers import make_state, setup
from tests.test_tgbot_handlers import make_context, make_update, replies, run
from tgbot import chat, config, handlers


class ChatPromptTests(unittest.TestCase):
    """The guardrail text itself: what the model is told it must never do."""

    def test_declines_advice_seeking_questions(self):
        for phrase in ("should i buy", "should i sell", "is this a good trade",
                       "what would you do", "will it go up"):
            self.assertIn(phrase, chat_prompt.SYSTEM_PROMPT.lower())

    def test_forbids_inventing_facts_outside_the_json(self):
        self.assertIn("Never state a price, level, date", chat_prompt.SYSTEM_PROMPT)
        self.assertIn("say plainly that you don't have that information", chat_prompt.SYSTEM_PROMPT)

    def test_states_the_researched_result_when_asked_about_an_edge(self):
        self.assertIn("did not beat random entries", chat_prompt.SYSTEM_PROMPT)
        self.assertIn("Not financial advice", chat_prompt.SYSTEM_PROMPT)

    def test_the_user_prompt_carries_the_question_and_the_facts(self):
        prompt = chat_prompt.build_user_prompt("what's up with GBP/USD?", '{"GBP_USD": {}}')
        self.assertIn("what's up with GBP/USD?", prompt)
        self.assertIn('{"GBP_USD": {}}', prompt)


class ContextJsonTests(unittest.TestCase):
    def test_a_live_setups_plan_numbers_are_included(self):
        state = make_state([setup("live_trade")])
        facts = json.loads(chat._context_json({"EUR_USD": state}))
        self.assertEqual(facts["EUR_USD"]["active_setups"][0]["entry_price"], 1.14650)
        self.assertIn("plan", facts["EUR_USD"]["active_setups"][0])

    def test_a_failed_pair_is_named_not_dropped(self):
        facts = json.loads(chat._context_json({"EUR_USD": RuntimeError("oanda down")}))
        self.assertIn("error", facts["EUR_USD"])

    def test_recent_rejections_are_included_for_chat(self):
        from dataclasses import replace

        from live.state import RecentRejection

        state = replace(make_state(), recent_rejections=[
            RecentRejection(direction="bearish", sweep_time="2026-09-21T09:00:00+00:00",
                            outcome="low_rr", rr=0.8)])
        facts = json.loads(chat._context_json({"EUR_USD": state}))
        self.assertEqual(facts["EUR_USD"]["recent_rejections"][0]["outcome"], "low_rr")


class AnswerTests(unittest.TestCase):
    def test_calls_deepseek_with_the_chat_system_prompt_and_the_question(self):
        state = make_state()
        captured = {}

        def fake_chat(system_prompt, user_prompt, **kwargs):
            captured["system_prompt"] = system_prompt
            captured["user_prompt"] = user_prompt
            return "ANSWER"

        with patch("tgbot.chat.deepseek_client.chat", side_effect=fake_chat):
            result = chat.answer("what's happening?", {"EUR_USD": state})

        self.assertEqual(result, "ANSWER")
        self.assertEqual(captured["system_prompt"], chat_prompt.SYSTEM_PROMPT)
        self.assertIn("what's happening?", captured["user_prompt"])
        self.assertIn("EUR_USD", captured["user_prompt"])

    def test_a_very_long_question_is_truncated_before_it_reaches_the_model(self):
        captured = {}

        def fake_chat(system_prompt, user_prompt, **kwargs):
            captured["user_prompt"] = user_prompt
            return "x"

        with patch("tgbot.chat.deepseek_client.chat", side_effect=fake_chat):
            chat.answer("a" * 5000, {})

        sent_question = captured["user_prompt"].split("User's question: ", 1)[1]
        self.assertLessEqual(len(sent_question), chat.MAX_QUESTION_LENGTH)


class ChatMessageHandlerTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(config, "allowed_chat_ids", return_value={111})
        patcher.start()
        self.addCleanup(patcher.stop)

    def context_with(self, results, reply):
        ctx = make_context()
        ctx.application.bot_data["service"].scan = lambda: results
        patcher = patch("tgbot.handlers.answer_chat", return_value=reply)
        patcher.start()
        self.addCleanup(patcher.stop)
        return ctx

    def test_a_plain_text_message_gets_a_chat_reply(self):
        update = make_update(text="what's happening on GBP/USD?")
        ctx = self.context_with({"GBP_USD": make_state()}, "Nothing is currently live on GBP/USD.")
        run(handlers.chat_message(update, ctx))
        self.assertIn("Nothing is currently live", replies(update)[0])

    def test_an_empty_message_is_ignored(self):
        update = make_update(text="   ")
        ctx = self.context_with({}, "should not be called")
        run(handlers.chat_message(update, ctx))
        update.effective_message.reply_text.assert_not_awaited()

    def test_a_failure_says_so_instead_of_going_silent(self):
        update = make_update(text="hello")
        ctx = make_context()
        ctx.application.bot_data["service"].scan = lambda: (_ for _ in ()).throw(RuntimeError("x"))
        run(handlers.chat_message(update, ctx))
        self.assertIn("Couldn't answer", replies(update)[0])

    def test_an_empty_model_reply_does_not_send_a_blank_message(self):
        update = make_update(text="hello")
        ctx = self.context_with({}, "")
        run(handlers.chat_message(update, ctx))
        self.assertIn("Couldn't come up with an answer", replies(update)[0])

    def test_only_authorised_chats_get_an_answer(self):
        update = make_update(chat_id=999, text="hello")
        ctx = self.context_with({}, "should not be seen")
        run(handlers.chat_message(update, ctx))
        self.assertIn("private", replies(update)[0])

    def test_the_reply_has_no_link_preview(self):
        update = make_update(text="hello")
        ctx = self.context_with({}, "some answer")
        run(handlers.chat_message(update, ctx))
        self.assertTrue(
            update.effective_message.reply_text.await_args.kwargs["link_preview_options"].is_disabled)


if __name__ == "__main__":
    unittest.main()
