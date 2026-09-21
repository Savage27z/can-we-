import asyncio
import shutil
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from telegram.error import Forbidden, InvalidToken

from news.filter import NewsStatus
from tests.tg_helpers import NOW, blackout_news, make_state, setup
from tgbot import alerts, config


def select(state, notified=frozenset(), chat_id=111, now=NOW, in_news_window=False):
    with patch.object(alerts, "_confirmed_in_news_window", return_value=in_news_window):
        return alerts.select_new_alerts(state, set(notified), now, chat_id)


class SelectNewAlertsTests(unittest.TestCase):
    def test_only_confirmed_live_trades_alert(self):
        state = make_state([setup("pending_fvg"), setup("pending_confirmation"), setup("live_trade")])
        self.assertEqual([s.status for s in select(state)], ["live_trade"])

    def test_dedup_is_per_chat(self):
        s = setup("live_trade")
        state = make_state([s])
        sent_to_111 = {alerts.delivery_key(state.pair, s, 111)}
        self.assertEqual(select(state, sent_to_111, chat_id=111), [])
        self.assertEqual(len(select(state, sent_to_111, chat_id=222)), 1)

    def test_key_includes_the_pair(self):
        s = setup("live_trade")
        eur_key = alerts.delivery_key("EUR_USD", s, 111)
        gbp_state = make_state([s])
        gbp_state.pair = "GBP_USD"
        self.assertEqual(len(select(gbp_state, {eur_key})), 1)

    def test_stale_confirmation_is_not_pushed_as_new(self):
        s = setup("live_trade", confirmed_ago=config.MAX_ALERT_AGE + timedelta(minutes=1))
        self.assertEqual(select(make_state([s])), [])

    def test_news_blackout_defers_everything(self):
        state = make_state([setup("live_trade")], news=blackout_news())
        self.assertEqual(select(state), [])

    def test_deferred_alert_fires_once_the_blackout_has_passed(self):
        s = setup("live_trade")
        self.assertEqual(select(make_state([s], news=blackout_news())), [])
        self.assertEqual(len(select(make_state([s]))), 1)

    def test_trade_confirmed_inside_a_news_window_is_skipped(self):
        state = make_state([setup("live_trade")])
        self.assertEqual(select(state, in_news_window=True), [])

    def test_stale_data_suppresses_alerts(self):
        state = make_state([setup("live_trade")])
        late = NOW + config.MAX_DATA_AGE + timedelta(minutes=1)
        self.assertEqual(select(state, now=late), [])

    def test_unavailable_news_does_not_suppress(self):
        state = make_state([setup("live_trade")], news=NewsStatus(status="unavailable", reason="x"))
        self.assertEqual(len(select(state)), 1)


class ConfirmedInNewsWindowTests(unittest.TestCase):
    def test_true_only_for_a_blackout(self):
        with patch.object(alerts, "check_news", return_value=NewsStatus(status="blackout")):
            self.assertTrue(alerts._confirmed_in_news_window("EUR_USD", NOW))
        with patch.object(alerts, "check_news", return_value=NewsStatus(status="clear")):
            self.assertFalse(alerts._confirmed_in_news_window("EUR_USD", NOW))

    def test_the_confirmation_time_is_judged_without_moving_the_cache_clock(self):
        confirmed = NOW - timedelta(hours=2)
        with patch.object(alerts, "check_news",
                          return_value=NewsStatus(status="clear")) as check:
            alerts._confirmed_in_news_window("EUR_USD", confirmed)
        check.assert_called_once_with("EUR_USD", at=confirmed)

    def test_a_failing_news_check_never_blocks_an_alert(self):
        with patch.object(alerts, "check_news", side_effect=RuntimeError("boom")):
            self.assertFalse(alerts._confirmed_in_news_window("EUR_USD", NOW))


class AlertLogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.log = alerts.AlertLog(self.tmp / "alert_state.json")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_roundtrip(self):
        self.log.add(["a", "b"])
        self.log.add(["c"])
        self.assertEqual(self.log.load(), ["a", "b", "c"])

    def test_missing_or_corrupt_file_loads_empty(self):
        self.assertEqual(self.log.load(), [])
        self.log.path.write_text("not json", encoding="utf-8")
        self.assertEqual(self.log.load(), [])

    def test_history_is_capped(self):
        self.log.add([str(i) for i in range(alerts.MAX_REMEMBERED + 25)])
        keys = self.log.load()
        self.assertEqual(len(keys), alerts.MAX_REMEMBERED)
        self.assertEqual(keys[-1], str(alerts.MAX_REMEMBERED + 24))

    def test_a_failing_disk_does_not_raise_and_still_deduplicates(self):
        with patch.object(self.log, "_write_disk", side_effect=OSError("disk full")):
            self.log.add(["k1"])  # must not raise
        self.assertIn("k1", self.log.load())

    def test_history_survives_a_disk_that_cannot_be_read(self):
        self.log.add(["old"])
        self.log.path.write_text("not json", encoding="utf-8")
        self.log.add(["new"])
        self.assertEqual(self.log.load(), ["old", "new"])


def make_bot(send=None, get_me=None, send_photo=None):
    return SimpleNamespace(send_message=send or AsyncMock(), get_me=get_me or AsyncMock(),
                           send_photo=send_photo or AsyncMock())


def make_context(service, alert_log, bot, bot_data=None):
    # One persistent dict across cycles, like the real application's bot_data.
    data = bot_data if bot_data is not None else {}
    data.update(service=service, alert_log=alert_log)
    return SimpleNamespace(application=SimpleNamespace(bot_data=data), bot=bot)


class AlertJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.log = alerts.AlertLog(self.tmp / "alert_state.json")
        self.state = make_state([setup("live_trade")])
        self.render_text = "REPORT"
        self.chart = None
        self.service = SimpleNamespace(
            fresh_state=lambda pair: self.state, render=lambda state: self.render_text,
            chart_for=lambda state: self.chart)
        self.bot_data = {}
        for patcher in (
            patch.object(alerts, "_utcnow", return_value=NOW),
            patch.object(alerts, "_confirmed_in_news_window", return_value=False),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_job(self, bot, chat_ids="111,222", service=None, times=1):
        ctx = make_context(service or self.service, self.log, bot, self.bot_data)
        with patch.dict("os.environ", {"TELEGRAM_ALLOWED_CHAT_IDS": chat_ids}):
            for _ in range(times):
                asyncio.run(alerts.alert_job(ctx))
        return ctx

    def _many_setups(self):
        from live.plan import MAX_PLANS_SHOWN

        return [setup("live_trade", sweep_time=f"2026-09-22T0{i}:00:00+00:00")
                for i in range(MAX_PLANS_SHOWN + 1)]

    def test_the_alert_describes_only_the_setups_that_were_selected(self):
        fresh = setup("live_trade", sweep_time="2026-09-22T09:00:00+00:00")
        blocked = setup("live_trade", sweep_time="2026-09-22T08:00:00+00:00",
                        confirmed_ago=timedelta(minutes=90))
        self.state = make_state([fresh, blocked])
        blocked_at = blocked.confirm_time
        rendered, charted = [], []
        self.service.render = lambda st: rendered.append(st) or "REPORT"
        self.service.chart_for = lambda st: charted.append(st) or None
        with patch.object(alerts, "_confirmed_in_news_window",
                          side_effect=lambda pair, at: at.isoformat() == blocked_at):
            self.run_job(make_bot(AsyncMock()), chat_ids="111")
        self.assertTrue(rendered and charted)
        for st in rendered + charted:
            self.assertEqual([s.sweep_time for s in st.active_setups], [fresh.sweep_time])
        self.assertEqual(len(self.log.load()), 1)

    def test_more_setups_than_a_message_shows_are_sent_in_batches_and_all_recorded(self):
        from live.plan import MAX_PLANS_SHOWN

        self.state = make_state(self._many_setups())
        rendered = []
        self.service.render = lambda st: rendered.append(len(st.active_setups)) or "REPORT"
        send = AsyncMock()
        self.run_job(make_bot(send), chat_ids="111")
        self.assertTrue(all(n <= MAX_PLANS_SHOWN for n in rendered))
        self.assertEqual(sum(rendered), MAX_PLANS_SHOWN + 1)   # every recorded setup was shown
        self.assertEqual(send.await_count, 2)
        self.assertEqual(len(self.log.load()), MAX_PLANS_SHOWN + 1)

    def test_a_failed_batch_leaves_only_its_own_setups_unrecorded(self):
        from live.plan import MAX_PLANS_SHOWN

        self.state = make_state(self._many_setups())
        send = AsyncMock(side_effect=[None, RuntimeError("down")])
        self.run_job(make_bot(send), chat_ids="111")
        self.assertEqual(len(self.log.load()), MAX_PLANS_SHOWN)   # the rest retry next cycle

    def test_stale_data_is_a_failed_check_not_a_healthy_one(self):
        self.state = make_state([setup("live_trade")])
        with patch.object(alerts, "_utcnow", return_value=NOW + timedelta(hours=24)):
            ctx = self.run_job(make_bot(AsyncMock()), chat_ids="111")
        health = ctx.application.bot_data["alert_health"]
        self.assertEqual(health["consecutive_failures"], 1)
        self.assertIn("stale", health["last_error"])

    def test_the_weekend_closure_does_not_count_as_a_failure(self):
        friday_close = NOW.replace(day=18, hour=21)
        self.state = replace(make_state(), as_of=friday_close.isoformat())
        with patch.object(alerts, "_utcnow", return_value=friday_close + timedelta(hours=30)):
            ctx = self.run_job(make_bot(AsyncMock()), chat_ids="111")
        self.assertEqual(ctx.application.bot_data["alert_health"]["consecutive_failures"], 0)

    def test_the_push_opens_by_saying_it_is_analysis_not_a_trade_signal(self):
        from tgbot.wording import ALERT_HEADER

        send = AsyncMock()
        self.run_job(make_bot(send), chat_ids="111")
        text = send.await_args.kwargs["text"]
        self.assertTrue(text.startswith(ALERT_HEADER), text[:120])
        self.assertIn("not a trade signal", text.splitlines()[0])
        self.assertIn("REPORT", text)
        self.assertNotIn("🚨", text)

    def test_sends_to_every_allowed_chat_and_records_each_delivery(self):
        send = AsyncMock()
        self.run_job(make_bot(send))
        self.assertEqual(send.await_count, 2)
        self.assertIn("REPORT", send.await_args.kwargs["text"])
        self.assertEqual(len(self.log.load()), 2)

    def test_alert_text_is_sent_without_a_link_preview_card(self):
        send = AsyncMock()
        self.run_job(make_bot(send), chat_ids="111")
        self.assertTrue(send.await_args.kwargs["link_preview_options"].is_disabled)

    def test_a_second_cycle_does_not_resend(self):
        send = AsyncMock()
        self.run_job(make_bot(send), times=2)
        self.assertEqual(send.await_count, 2)

    def test_nothing_is_recorded_when_delivery_fails_so_it_retries(self):
        self.run_job(make_bot(AsyncMock(side_effect=RuntimeError("telegram down"))))
        self.assertEqual(self.log.load(), [])
        send = AsyncMock()
        self.run_job(make_bot(send))
        self.assertEqual(send.await_count, 2)

    def test_only_the_chat_that_failed_is_retried(self):
        async def first_cycle_send(chat_id, text, **kwargs):
            if chat_id == 111:
                raise RuntimeError("timeout")

        self.run_job(make_bot(first_cycle_send))
        retry_targets = []

        async def second_cycle_send(chat_id, text, **kwargs):
            retry_targets.append(chat_id)

        self.run_job(make_bot(second_cycle_send))
        self.assertEqual(retry_targets, [111])  # 222 already had it; no duplicate

    def test_a_blocked_chat_is_not_retried_forever(self):
        async def blocked(chat_id, text, **kwargs):
            raise Forbidden("bot was blocked by the user")

        self.run_job(make_bot(blocked), chat_ids="111")
        self.assertEqual(len(self.log.load()), 1)
        send = AsyncMock()
        self.run_job(make_bot(send), chat_ids="111")
        send.assert_not_awaited()

    def test_an_oversized_alert_is_split_into_telegram_sized_messages(self):
        self.render_text = "\n".join(["line " * 20] * 200)  # ~20k characters
        sent = []

        async def strict_send(chat_id, text, **kwargs):
            if len(text) > 4096:
                raise RuntimeError("Message is too long")
            sent.append(text)

        self.run_job(make_bot(strict_send), chat_ids="111")
        self.assertGreater(len(sent), 1)
        self.assertEqual(len(self.log.load()), 1)  # delivered, so recorded

    def test_overlapping_runs_send_each_alert_only_once(self):
        # The startup job and the hourly job are separate scheduler jobs and can overlap.
        sent = []

        async def slow_send(chat_id, text, **kwargs):
            await asyncio.sleep(0.01)
            sent.append(chat_id)

        ctx = make_context(self.service, self.log, make_bot(slow_send), self.bot_data)

        async def both():
            await asyncio.gather(alerts.alert_job(ctx), alerts.alert_job(ctx))

        with patch.dict("os.environ", {"TELEGRAM_ALLOWED_CHAT_IDS": "111,222"}):
            asyncio.run(both())
        self.assertEqual(sorted(sent), [111, 222])

    def test_the_chart_is_sent_before_the_alert_text(self):
        self.chart = b"PNG"
        order = []

        async def photo(chat_id, photo):
            order.append("photo")

        async def text(chat_id, text, **kwargs):
            order.append("text")

        self.run_job(make_bot(text, send_photo=photo), chat_ids="111")
        self.assertEqual(order, ["photo", "text"])

    def test_no_photo_is_sent_when_there_is_no_chart(self):
        send_photo = AsyncMock()
        self.run_job(make_bot(send_photo=send_photo), chat_ids="111")
        send_photo.assert_not_awaited()

    def test_a_failed_chart_never_blocks_the_alert_text_or_its_record(self):
        self.chart = b"PNG"
        send = AsyncMock()
        self.run_job(make_bot(send, send_photo=AsyncMock(side_effect=RuntimeError("upload"))),
                     chat_ids="111")
        send.assert_awaited_once()
        self.assertEqual(len(self.log.load()), 1)

    def test_a_chat_that_blocked_the_bot_is_not_retried_via_the_photo_either(self):
        self.chart = b"PNG"
        send = AsyncMock()
        self.run_job(make_bot(send, send_photo=AsyncMock(side_effect=Forbidden("blocked"))),
                     chat_ids="111")
        send.assert_not_awaited()
        self.assertEqual(len(self.log.load()), 1)  # recorded: no endless retries

    def test_no_allowed_chats_means_no_work(self):
        send = AsyncMock()
        self.run_job(make_bot(send), chat_ids="")
        send.assert_not_awaited()

    def test_a_rejected_token_exits_instead_of_leaving_a_deaf_bot_running(self):
        bot = make_bot(get_me=AsyncMock(side_effect=InvalidToken("rejected")))
        with patch.object(alerts, "_die") as die:
            die.side_effect = SystemExit(1)  # the real _die never returns
            with self.assertRaises(SystemExit):
                self.run_job(bot)
        die.assert_called_once()

    def test_a_network_blip_on_getme_does_not_stop_the_cycle(self):
        send = AsyncMock()
        self.run_job(make_bot(send, get_me=AsyncMock(side_effect=RuntimeError("timeout"))))
        self.assertEqual(send.await_count, 2)


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.log = alerts.AlertLog(self.tmp / "alert_state.json")
        self.failing = True
        self.bot_data = {}

        def fresh_state(pair):
            if self.failing:
                raise RuntimeError("OANDA down")
            return make_state()

        self.service = SimpleNamespace(fresh_state=fresh_state, render=lambda s: "x",
                                       chart_for=lambda s: None)
        patcher = patch.object(alerts, "_utcnow", return_value=NOW)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def cycle(self, send):
        ctx = make_context(self.service, self.log, make_bot(send), self.bot_data)
        with patch.dict("os.environ", {"TELEGRAM_ALLOWED_CHAT_IDS": "111"}):
            asyncio.run(alerts.alert_job(ctx))

    def test_a_failure_streak_warns_once_then_announces_recovery(self):
        send = AsyncMock()
        for _ in range(config.ALERT_FAILURES_BEFORE_WARNING - 1):
            self.cycle(send)
        send.assert_not_awaited()  # not yet

        self.cycle(send)  # the streak reaches the threshold
        self.assertEqual(send.await_count, 1)
        self.assertIn("failed", send.await_args.kwargs["text"])

        self.cycle(send)  # still failing: no repeated warning
        self.assertEqual(send.await_count, 1)

        self.failing = False
        self.cycle(send)
        self.assertEqual(send.await_count, 2)
        self.assertIn("recovered", send.await_args.kwargs["text"])
        self.assertEqual(self.bot_data["alert_health"]["consecutive_failures"], 0)

    def test_health_records_the_last_error_and_cycle_time(self):
        self.cycle(AsyncMock())
        health = self.bot_data["alert_health"]
        self.assertEqual(health["consecutive_failures"], 1)
        self.assertIn("OANDA down", health["last_error"])
        self.assertEqual(health["last_cycle"], NOW.isoformat())


if __name__ == "__main__":
    unittest.main()
