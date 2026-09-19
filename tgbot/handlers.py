import asyncio
import logging

from . import config
from .pairs import normalize_pair
from .service import ReportService

log = logging.getLogger(__name__)

TELEGRAM_MESSAGE_LIMIT = 4000  # Telegram's hard cap is 4096 characters

HELP_TEXT = (
    "Commands:\n"
    "/analysis [pair] — structural read (default EUR_USD)\n"
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
        "Pushed automatically when a new setup confirms "
        "(paused around high-impact news)\n\n"
        "<i>Structural analysis, not financial advice.</i>"
    )


def split_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    chunks, current = [], ""
    for line in text.split("\n"):
        while len(line) > limit:  # a single oversized line: hard-split it
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


async def _authorized(update) -> bool:
    chat_id = update.effective_chat.id
    allowed = config.allowed_chat_ids()
    if chat_id in allowed:
        return True
    if not allowed:
        await update.message.reply_text(
            f"This bot isn't set up yet. Your chat ID is {chat_id}.\n"
            f"Add TELEGRAM_ALLOWED_CHAT_IDS={chat_id} to the bot's .env and restart it."
        )
    else:
        await update.message.reply_text("This bot is private.")
    return False


async def start(update, context) -> None:
    if not await _authorized(update):
        return
    await update.message.reply_html(start_text())


async def help_command(update, context) -> None:
    if not await _authorized(update):
        return
    await update.message.reply_text(HELP_TEXT)


async def analysis(update, context) -> None:
    if not await _authorized(update):
        return

    raw = context.args[0] if context.args else config.DEFAULT_PAIR
    pair = normalize_pair(raw)
    if pair is None:
        await update.message.reply_text(
            f"Couldn't read {raw!r} as a currency pair. Try /analysis EUR_USD."
        )
        return
    if pair not in config.LIVE_PAIRS:
        await update.message.reply_text(
            f"{pair} isn't enabled. Only {', '.join(config.LIVE_PAIRS)} passed the "
            f"backtest gate; the other tested pairs showed negative expectancy."
        )
        return

    service: ReportService = context.application.bot_data["service"]
    try:
        text = await asyncio.to_thread(service.get_report, pair)
    except Exception:
        log.exception("analysis failed for %s", pair)
        await update.message.reply_text(
            "Couldn't build the report right now (the data refresh failed). "
            "Try again in a few minutes."
        )
        return

    for chunk in split_message(text):
        await update.message.reply_text(chunk)
