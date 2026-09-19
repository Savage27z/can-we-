import re
from typing import Optional

_LETTERS_RE = re.compile(r"^[A-Z]{6}$")

# The major currencies OANDA quotes as forex pairs. Anything else (e.g. "banana")
# is treated as unreadable rather than as an unenabled pair.
KNOWN_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"}


def normalize_pair(text: str) -> Optional[str]:
    """Turns user input like 'eurusd', 'EUR/USD', '$EUR_USD' into OANDA's
    'EUR_USD' format, or None if it isn't a pair of known currencies."""
    cleaned = re.sub(r"[\s$/_\-]", "", text).upper()
    if not _LETTERS_RE.match(cleaned):
        return None
    base, quote = cleaned[:3], cleaned[3:]
    if base not in KNOWN_CURRENCIES or quote not in KNOWN_CURRENCIES or base == quote:
        return None
    return f"{base}_{quote}"
