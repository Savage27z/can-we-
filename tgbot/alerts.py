"""Scheduled push alerts: notify once per newly confirmed live trade, per chat.

Design points:
- Deduplicated per (pair, direction, sweep time, chat), so one chat failing does not
  re-send to the others, and a second pair can never shadow the first.
- A news blackout DEFERS an alert (it is not recorded, so it can still go out once
  the blackout has passed). A trade whose confirmation candle itself fell inside a
  news window is skipped instead: the entry was made in exactly the conditions the
  filter exists to avoid.
- Skipped when the newest candle is stale, since the trade may have been invalidated.
- One cycle at a time (a lock), so the startup run and the hourly run can never
  both send the same alert.
"""
import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from telegram.error import Forbidden, InvalidToken

from data_pipeline import config as data_config
from live.state import ActiveSetup, LiveState
from news.filter import check_news

from . import config
from .messages import split_message
from .service import ReportService

log = logging.getLogger(__name__)

MAX_REMEMBERED = 200
MAX_STORED_ERROR_LENGTH = 200


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def delivery_key(pair: str, setup: ActiveSetup, chat_id: int) -> str:
    return f"{pair}|{setup.direction}|{setup.sweep_time}|{chat_id}"


def _confirmed_in_news_window(pair: str, confirmed_at: datetime) -> bool:
    try:
        return check_news(pair, now=confirmed_at).status == "blackout"
    except Exception:
        log.exception("could not check news at confirmation time; not suppressing")
        return False


def select_new_alerts(state: LiveState, notified: set[str], now: datetime,
                      chat_id: int) -> list[ActiveSetup]:
    live_trades = [s for s in state.active_setups if s.status == "live_trade"]
    if not live_trades:
        return []

    if state.news.status == "blackout":
        log.info("%d live trade(s) held back: high-impact news blackout in progress",
                 len(live_trades))
        return []

    as_of = datetime.fromisoformat(state.as_of)
    if now - as_of > config.MAX_DATA_AGE:
        log.warning("not alerting: newest candle closed %s ago, data is stale", now - as_of)
        return []

    fresh = []
    for setup in live_trades:
        if delivery_key(state.pair, setup, chat_id) in notified:
            continue
        confirmed_at = datetime.fromisoformat(setup.confirm_time)
        if now - confirmed_at > config.MAX_ALERT_AGE:
            log.info("live trade %s confirmed %s ago is too old to push",
                     setup.sweep_time, now - confirmed_at)
            continue
        if _confirmed_in_news_window(state.pair, confirmed_at):
            log.info("live trade %s confirmed inside a high-impact news window; not pushed",
                     setup.sweep_time)
            continue
        fresh.append(setup)
    return fresh


class AlertLog:
    """Remembers which alerts were already pushed, on disk and in memory.

    The in-memory copy means a failing disk (full volume, permissions) can no longer
    cause the same alert to be re-sent every hour; the disk copy means a restart
    doesn't forget."""

    def __init__(self, path: Path | None = None):
        self._path = path
        self._memory: list[str] = []

    @property
    def path(self) -> Path:
        return self._path or data_config.DATA_DIR / "alert_state.json"

    def _read_disk(self) -> list[str]:
        try:
            keys = json.loads(self.path.read_text(encoding="utf-8"))["notified"]
            return [k for k in keys if isinstance(k, str)]
        except (OSError, ValueError, KeyError, TypeError):
            return []

    def load(self) -> list[str]:
        return list(dict.fromkeys(self._read_disk() + self._memory))

    def _write_disk(self, keys: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"notified": keys}, f)
            os.replace(tmp, self.path)
        except BaseException:
            os.unlink(tmp)
            raise

    def add(self, new_keys: list[str]) -> None:
        keys = list(dict.fromkeys(self.load() + new_keys))[-MAX_REMEMBERED:]
        self._memory = keys
        try:
            self._write_disk(keys)
        except OSError:
            log.exception("could not persist the alert log; alerts stay deduplicated "
                          "in memory until restart")


async def _send(bot, chat_id: int, text: str, photo: Optional[bytes] = None) -> bool:
    """True once the chat needs no further attempts: delivered, or permanently
    unreachable (it blocked the bot). False means retry on the next cycle.
    The photo is best-effort: failing to send it never blocks the text."""
    if photo:
        try:
            await bot.send_photo(chat_id=chat_id, photo=photo)
        except Forbidden:
            log.warning("chat %s has blocked the bot; not retrying", chat_id)
            return True
        except Exception:
            log.exception("could not send the alert chart to chat %s; sending text only", chat_id)
    for chunk in split_message(text):
        try:
            await bot.send_message(chat_id=chat_id, text=chunk)
        except Forbidden:
            log.warning("chat %s has blocked the bot; not retrying", chat_id)
            return True
        except Exception:
            log.exception("could not deliver alert to chat %s", chat_id)
            return False
    return True


def _die() -> None:
    logging.shutdown()
    os._exit(1)


async def _run_cycle(context, chat_ids: set[int]) -> Optional[str]:
    """One check for every live pair. Returns None on success, else a short error."""
    service: ReportService = context.application.bot_data["service"]
    alert_log: AlertLog = context.application.bot_data["alert_log"]

    # A rejected token kills PTB's polling task while the process keeps running, and
    # the platform only restarts a process that exits. Exit non-zero so the failure
    # is visible instead of leaving a bot that looks healthy but is deaf.
    try:
        await context.bot.get_me()
    except InvalidToken:
        log.critical("Telegram rejected the bot token; exiting so the failure is visible")
        _die()
    except Exception as err:
        log.warning("Telegram getMe failed (continuing): %s", err)

    error = None
    for pair in config.LIVE_PAIRS:
        try:
            state = await asyncio.to_thread(service.fresh_state, pair)
        except Exception as err:
            log.exception("alert check failed for %s; will retry next cycle", pair)
            error = f"{type(err).__name__}: {err}"
            continue

        notified = set(alert_log.load())
        now = _utcnow()
        report = None
        chart = None
        for chat_id in sorted(chat_ids):
            new = select_new_alerts(state, notified, now, chat_id)
            if not new:
                continue
            if report is None:
                rendered = await asyncio.to_thread(service.render, state)
                report = "🚨 New live setup\n\n" + rendered
                chart = await asyncio.to_thread(service.chart_for, state)
            if await _send(context.bot, chat_id, report, chart):
                alert_log.add([delivery_key(state.pair, s, chat_id) for s in new])
            else:
                error = f"delivery to chat {chat_id} failed"
    return error


def _record_health(bot_data: dict, error: Optional[str]) -> Optional[str]:
    """Updates the health record shown by /status. Returns 'warn' exactly when the
    failure streak reaches the threshold, 'recovered' when a streak that had reached
    it ends, else None."""
    health = bot_data.setdefault("alert_health", {
        "last_cycle": None, "last_ok": None, "consecutive_failures": 0, "last_error": None,
    })
    stamp = _utcnow().isoformat()
    health["last_cycle"] = stamp
    if error is None:
        recovered = health["consecutive_failures"] >= config.ALERT_FAILURES_BEFORE_WARNING
        health.update(consecutive_failures=0, last_ok=stamp, last_error=None)
        return "recovered" if recovered else None
    health["consecutive_failures"] += 1
    health["last_error"] = error[:MAX_STORED_ERROR_LENGTH]
    if health["consecutive_failures"] == config.ALERT_FAILURES_BEFORE_WARNING:
        return "warn"
    return None


async def alert_job(context) -> None:
    """Runs on a schedule via the bot's JobQueue."""
    chat_ids = config.allowed_chat_ids()
    if not chat_ids:
        return

    bot_data = context.application.bot_data
    lock = bot_data.setdefault("alert_lock", asyncio.Lock())
    async with lock:
        error = await _run_cycle(context, chat_ids)
        outcome = _record_health(bot_data, error)

    if outcome == "warn":
        health = bot_data["alert_health"]
        text = (f"⚠️ The hourly alert check has failed {health['consecutive_failures']} times "
                f"in a row (last error: {health['last_error']}). New setups may be delayed "
                f"or missed until this is fixed.")
    elif outcome == "recovered":
        text = "✅ The hourly alert check has recovered."
    else:
        return
    for chat_id in sorted(chat_ids):
        await _send(context.bot, chat_id, text)
