import asyncio
import logging
from datetime import datetime, timezone

from . import config
from .messages import NO_LINK_PREVIEW, split_message
from .pairs import normalize_pair
from .scan import scan_text
from .service import ReportService
from .wording import ALERTS_LINE, not_enabled

log = logging.getLogger(__name__)

MAX_ECHOED_INPUT = 40  # keeps the "couldn't read that" reply far below Telegram's limit

HELP_TEXT = (
    "Commands:\n"
    "/scan — where every pair stands right now, nothing to wait for\n"
    "/analysis [pair] — full structural read and chart (default EUR_USD)\n"
    "/status — health of the hourly alert checks\n"
    "/help — this message\n\n"
    f"Enabled pairs: {', '.join(config.LIVE_PAIRS)}"
)


def start_text() -> str:
    """The /start reply (Telegram HTML). Enabled pairs come from config so this
    can't drift from what /analysis will actually accept."""
    enabled = "\n".join(
        f"• ${pair.replace('_', '')} — Forex" for pair in config.LIVE_PAIRS
    )
    example = f"/analysis ${config.DEFAULT_PAIR.replace('_', '')}"
    return (
        "📈 <b>FOREX STRUCTURE SCANNER</b>\n\n"
        "Get a structural read with:\n"
        f"<code>{example}</code>\n\n"
        f"Enabled:\n{enabled}\n\n"
        "<b>Analysis:</b>\n"
        "Daily → H4 → H1\n"
        "Liquidity sweeps + FVG + confirmation + news check\n\n"
        "<b>Alerts:</b>\n"
        f"{ALERTS_LINE}\n\n"
        "<i>Structural analysis, not financial advice.</i>"
    )


def status_text(health: dict | None) -> str:
    if not health or not health.get("last_cycle"):
        return "No alert check has completed yet since the bot started."
    lines = [f"Last alert check: {health['last_cycle']}"]
    if health["consecutive_failures"]:
        lines.append(
            f"⚠️ {health['consecutive_failures']} consecutive failing check(s). "
            f"Last error: {health['last_error']}"
        )
    else:
        lines.append("✅ Healthy — the last check completed without errors.")
    if health.get("last_ok"):
        lines.append(f"Last fully successful check: {health['last_ok']}")
    return "\n".join(lines)


# Handlers use `update.effective_message`, not `update.message`: Telegram delivers an
# EDITED command as `edited_message`, for which `update.message` is None. Using
# `.message` made an edited /analysis run the paid work and then crash on the reply.

async def _authorized(update) -> bool:
    chat_id = update.effective_chat.id
    allowed = config.allowed_chat_ids()
    if chat_id in allowed:
        return True
    message = update.effective_message
    if not allowed:
        await message.reply_text(
            f"This bot isn't set up yet. Your chat ID is {chat_id}.\n"
            f"Set TELEGRAM_ALLOWED_CHAT_IDS={chat_id} in the bot's environment "
            f"(.env locally, service variables on Railway) and restart it."
        )
    else:
        await message.reply_text("This bot is private.")
    return False


async def start(update, context) -> None:
    if not await _authorized(update):
        return
    await update.effective_message.reply_html(start_text())


async def help_command(update, context) -> None:
    if not await _authorized(update):
        return
    await update.effective_message.reply_text(HELP_TEXT)


async def status(update, context) -> None:
    if not await _authorized(update):
        return
    health = context.application.bot_data.get("alert_health")
    await update.effective_message.reply_text(status_text(health))


async def scan(update, context) -> None:
    if not await _authorized(update):
        return
    message = update.effective_message
    service: ReportService = context.application.bot_data["service"]
    try:
        results = await asyncio.to_thread(service.scan)
    except Exception:
        log.exception("scan failed")
        await message.reply_text("Couldn't scan the pairs right now. Try again in a few minutes.")
        return
    for chunk in split_message(scan_text(results, datetime.now(timezone.utc))):
        await message.reply_text(chunk, link_preview_options=NO_LINK_PREVIEW)


async def analysis(update, context) -> None:
    if not await _authorized(update):
        return
    message = update.effective_message

    raw = context.args[0] if context.args else config.DEFAULT_PAIR
    if raw.strip().lower() in ("all", "scan"):
        await scan(update, context)
        return
    pair = normalize_pair(raw)
    if pair is None:
        shown = raw if len(raw) <= MAX_ECHOED_INPUT else raw[:MAX_ECHOED_INPUT] + "…"
        await message.reply_text(
            f"Couldn't read {shown!r} as a currency pair. Try /analysis EUR_USD."
        )
        return
    if pair not in config.LIVE_PAIRS:
        await message.reply_text(not_enabled(pair, list(config.LIVE_PAIRS)))
        return

    service: ReportService = context.application.bot_data["service"]
    try:
        result = await asyncio.to_thread(service.get_analysis, pair)
    except Exception:
        log.exception("analysis failed for %s", pair)
        await message.reply_text(
            "Couldn't build the report right now (the data refresh failed). "
            "Try again in a few minutes."
        )
        return

    if result.chart:
        try:
            await message.reply_photo(photo=result.chart)
        except Exception:
            # The report matters more than its picture: carry on with the text.
            log.exception("could not send the chart for %s", pair)
    for chunk in split_message(result.text):
        await message.reply_text(chunk, link_preview_options=NO_LINK_PREVIEW)
