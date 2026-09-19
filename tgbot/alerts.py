"""Scheduled push alerts: notify once per newly confirmed live trade.

Suppressed during a news blackout WITHOUT being marked as notified, so a setup that
confirms mid-blackout is still pushed afterwards if it's within MAX_ALERT_AGE.
"""
import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from data_pipeline import config as data_config
from live.state import ActiveSetup, LiveState

from . import config
from .service import ReportService

log = logging.getLogger(__name__)

MAX_REMEMBERED = 200


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def alert_key(setup: ActiveSetup) -> str:
    return f"{setup.direction}|{setup.sweep_time}"


def select_new_alerts(state: LiveState, notified: set[str], now: datetime) -> list[ActiveSetup]:
    if state.news.status == "blackout":
        return []
    fresh = []
    for setup in state.active_setups:
        if setup.status != "live_trade" or alert_key(setup) in notified:
            continue
        confirmed_at = datetime.fromisoformat(setup.confirm_time)
        if now - confirmed_at > config.MAX_ALERT_AGE:
            continue
        fresh.append(setup)
    return fresh


class AlertLog:
    """Persists which setups were already pushed, so a restart never re-sends them."""

    def __init__(self, path: Path | None = None):
        self._path = path

    @property
    def path(self) -> Path:
        return self._path or data_config.DATA_DIR / "alert_state.json"

    def load(self) -> list[str]:
        try:
            keys = json.loads(self.path.read_text(encoding="utf-8"))["notified"]
            return [k for k in keys if isinstance(k, str)]
        except (OSError, ValueError, KeyError, TypeError):
            return []

    def add(self, new_keys: list[str]) -> None:
        keys = (self.load() + new_keys)[-MAX_REMEMBERED:]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"notified": keys}, f)
            os.replace(tmp, self.path)
        except BaseException:
            os.unlink(tmp)
            raise


async def alert_job(context) -> None:
    """Runs on a schedule via the bot's JobQueue."""
    service: ReportService = context.application.bot_data["service"]
    alert_log: AlertLog = context.application.bot_data["alert_log"]
    chat_ids = config.allowed_chat_ids()
    if not chat_ids:
        return

    for pair in config.LIVE_PAIRS:
        try:
            state = await asyncio.to_thread(service.fresh_state, pair)
        except Exception:
            log.exception("alert check failed for %s; will retry next cycle", pair)
            continue

        new = select_new_alerts(state, set(alert_log.load()), _utcnow())
        if not new:
            continue

        text = "🚨 New live setup\n\n" + await asyncio.to_thread(service.render, state)
        delivered = 0
        for chat_id in chat_ids:
            try:
                await context.bot.send_message(chat_id=chat_id, text=text)
                delivered += 1
            except Exception:
                log.exception("could not deliver alert to chat %s", chat_id)
        if delivered:
            alert_log.add([alert_key(s) for s in new])
