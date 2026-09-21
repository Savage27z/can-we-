from telegram import LinkPreviewOptions

TELEGRAM_MESSAGE_LIMIT = 4000  # Telegram's hard cap is 4096 characters

# Reports carry a TradingView link; without this Telegram would unfurl it into a large card.
NO_LINK_PREVIEW = LinkPreviewOptions(is_disabled=True)


def split_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Splits on line boundaries into chunks of at most `limit` characters.
    Whitespace-only input yields no chunks."""
    if not text.strip():
        return []
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
