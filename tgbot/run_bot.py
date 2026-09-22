"""Runs the Telegram bot: /analysis on demand plus hourly alert checks.

    python -m tgbot.run_bot
"""
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

from data_pipeline import config as data_config
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from . import config
from .alerts import AlertLog, alert_job
from .handlers import analysis, chat_message, help_command, scan, start, status
from .service import ReportService


class RedactingFormatter(logging.Formatter):
    """The bot token is embedded in every Telegram API URL, so it can surface in
    exception text or HTTP logs. Scrub it from the final formatted output."""

    def __init__(self, secret: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._secret = secret

    def format(self, record: logging.LogRecord) -> str:
        return super().format(record).replace(self._secret, "<token>")


def configure_logging(token: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(RedactingFormatter(token, "%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    # httpx logs full request URLs (token included) at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def seconds_until_next_check(now: datetime) -> float:
    target = now.replace(minute=config.ALERT_MINUTE_PAST_HOUR, second=0, microsecond=0)
    if target <= now:
        target += timedelta(hours=1)
    return (target - now).total_seconds()


def build_application(token: str) -> Application:
    app = (
        Application.builder()
        .token(token)
        # PTB's default 5s read timeout can report a message as failed even though
        # Telegram delivered it, which would then be retried as a duplicate.
        .read_timeout(20)
        .write_timeout(20)
        # Handle updates concurrently: with the default (one at a time) a single
        # slow /analysis stalls every other reply. Shared state is lock-protected.
        .concurrent_updates(True)
        .build()
    )
    app.bot_data["service"] = ReportService()
    app.bot_data["alert_log"] = AlertLog()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("scan", scan))
    app.add_handler(CommandHandler(["analysis", "analyze"], analysis))
    # Anything else typed as plain text (not a command): answered by the Q&A chat layer.
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_message))

    # One check shortly after startup (so a restart doesn't wait up to an hour),
    # then hourly just after each H1 close. The grace period stops APScheduler
    # from silently skipping a run that starts a moment late.
    job_kwargs = {"misfire_grace_time": config.ALERT_MISFIRE_GRACE_SECONDS, "coalesce": True}
    app.job_queue.run_once(alert_job, when=30, job_kwargs=job_kwargs)
    app.job_queue.run_repeating(
        alert_job, interval=3600,
        first=seconds_until_next_check(datetime.now(timezone.utc)),
        job_kwargs=job_kwargs,
    )
    return app


def warn_if_data_dir_is_not_a_volume(log: logging.Logger) -> None:
    """On Railway an unmounted /data silently falls back to the container's own
    disk, which is wiped on every deploy: the alert history and candles vanish and
    recent alerts get re-sent. Log loudly instead of running quietly like that."""
    data_dir = data_config.DATA_DIR
    log.info("data directory: %s", data_dir)
    if os.environ.get("RAILWAY_ENVIRONMENT") and not os.path.ismount(data_dir):
        log.warning("%s is not a mounted volume; alert history and candle data will "
                    "be lost on every redeploy", data_dir)


def main() -> None:
    token = config.require_token()
    configure_logging(token)
    log = logging.getLogger("tgbot")
    chat_count = len(config.allowed_chat_ids())
    log.info("starting; %d authorized chat(s) configured", chat_count)
    if chat_count == 0:
        log.warning("no authorized chats: the bot will not send alerts or answer commands")
    warn_if_data_dir_is_not_a_volume(log)
    try:
        # bootstrap_retries=-1: keep retrying if Telegram is unreachable at startup
        # instead of exiting and burning through the platform's restart budget.
        build_application(token).run_polling(bootstrap_retries=-1)
    except Exception:
        # PTB's exceptions can embed the token (e.g. "The token `...` was rejected").
        # Log through the redacting formatter rather than let Python's default
        # excepthook print the raw traceback to stderr.
        log.critical("bot stopped with an unhandled error", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
