"""Runs the Telegram bot: /analysis on demand plus hourly alert checks.

    python -m tgbot.run_bot
"""
import logging
from datetime import datetime, timedelta, timezone

from telegram.ext import Application, CommandHandler

from . import config
from .alerts import AlertLog, alert_job
from .handlers import analysis, help_command, start
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
    app = Application.builder().token(token).build()
    app.bot_data["service"] = ReportService()
    app.bot_data["alert_log"] = AlertLog()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler(["analysis", "analyze"], analysis))

    # One check shortly after startup (so a restart doesn't wait up to an hour),
    # then hourly just after each H1 close.
    app.job_queue.run_once(alert_job, when=30)
    app.job_queue.run_repeating(
        alert_job, interval=3600,
        first=seconds_until_next_check(datetime.now(timezone.utc)),
    )
    return app


def main() -> None:
    token = config.require_token()
    configure_logging(token)
    log = logging.getLogger("tgbot")
    log.info("starting; %d authorized chat(s) configured", len(config.allowed_chat_ids()))
    build_application(token).run_polling()


if __name__ == "__main__":
    main()
