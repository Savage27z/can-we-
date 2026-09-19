import asyncio
import shutil
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tests.tg_helpers import NOW, blackout_news, make_state, setup
from tgbot import alerts, config


class SelectNewAlertsTests(unittest.TestCase):
    def test_only_confirmed_live_trades_alert(self):
        state = make_state([setup("pending_fvg"), setup("pending_confirmation"), setup("live_trade")])
        selected = alerts.select_new_alerts(state, set(), NOW)
        self.assertEqual([s.status for s in selected], ["live_trade"])

    def test_already_notified_setup_is_skipped(self):
        s = setup("live_trade")
        self.assertEqual(alerts.select_new_alerts(make_state([s]), {alerts.alert_key(s)}, NOW), [])

    def test_stale_confirmation_is_not_pushed_as_new(self):
        s = setup("live_trade", confirmed_ago=config.MAX_ALERT_AGE + timedelta(minutes=1))
        self.assertEqual(alerts.select_new_alerts(make_state([s]), set(), NOW), [])

    def test_news_blackout_suppresses_everything(self):
        state = make_state([setup("live_trade")], news=blackout_news())
        self.assertEqual(alerts.select_new_alerts(state, set(), NOW), [])

    def test_alert_fires_once_the_blackout_has_passed(self):
        s = setup("live_trade")
        during = make_state([s], news=blackout_news())
        after = make_state([s])
        self.assertEqual(alerts.select_new_alerts(during, set(), NOW), [])
        self.assertEqual(len(alerts.select_new_alerts(after, set(), NOW)), 1)


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


def fake_context(service, alert_log, send):
    app = SimpleNamespace(bot_data={"service": service, "alert_log": alert_log})
    return SimpleNamespace(application=app, bot=SimpleNamespace(send_message=send))


class AlertJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.log = alerts.AlertLog(self.tmp / "alert_state.json")
        self.state = make_state([setup("live_trade")])
        self.service = SimpleNamespace(fresh_state=lambda pair: self.state,
                                       render=lambda state: "REPORT")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_job(self, send, chat_ids="111,222"):
        with patch.dict("os.environ", {"TELEGRAM_ALLOWED_CHAT_IDS": chat_ids}), \
             patch.object(alerts, "_utcnow", return_value=NOW):
            asyncio.run(alerts.alert_job(fake_context(self.service, self.log, send)))

    def test_sends_to_every_allowed_chat_and_records_the_alert(self):
        send = AsyncMock()
        self.run_job(send)
        self.assertEqual(send.await_count, 2)
        self.assertIn("REPORT", send.await_args.kwargs["text"])
        self.assertEqual(len(self.log.load()), 1)

    def test_second_run_does_not_resend(self):
        send = AsyncMock()
        self.run_job(send)
        self.run_job(send)
        self.assertEqual(send.await_count, 2)  # still only the first run's two sends

    def test_not_recorded_when_no_delivery_succeeds(self):
        send = AsyncMock(side_effect=RuntimeError("telegram down"))
        self.run_job(send)
        self.assertEqual(self.log.load(), [])  # retried next cycle

    def test_one_failing_chat_does_not_block_the_others(self):
        calls = []

        async def send(chat_id, text):
            calls.append(chat_id)
            if chat_id == 111:
                raise RuntimeError("blocked by user")

        self.run_job(send)
        self.assertEqual(sorted(calls), [111, 222])
        self.assertEqual(len(self.log.load()), 1)

    def test_no_allowed_chats_means_no_work(self):
        send = AsyncMock()
        self.run_job(send, chat_ids="")
        send.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
