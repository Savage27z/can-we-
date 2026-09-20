import logging
import os
from datetime import timedelta

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# Only pairs that passed the Phase 2 backtest gate (EUR_USD: +0.34R expectancy over
# 8 years; GBP_USD and USD_JPY were negative). Serving the others would present
# setups the backtest says lose money as if they were vetted signals.
LIVE_PAIRS = ["EUR_USD"]
DEFAULT_PAIR = LIVE_PAIRS[0]

# How much candle history the live bot keeps/bootstraps per pair. The backtest used
# 8 years; live only acts on recent structure (90-candle liquidity lookback, a
# 10-candle FVG window, 3 daily candles for bias), so 2 years is plenty and keeps a
# fresh deployment's first download small.
LIVE_HISTORY_YEARS = 2

# A /analysis report is reused for this long, so repeated commands don't each
# trigger an OANDA refresh plus a paid DeepSeek call.
REPORT_CACHE_TTL = timedelta(minutes=5)

# Scheduled alert check: H1 candles close on the hour; check a few minutes after
# so OANDA has published the just-closed candle.
ALERT_MINUTE_PAST_HOUR = 5

# A confirmed trade older than this is not pushed as "new" (e.g. the bot was down
# when it confirmed) — it's still visible via /analysis.
MAX_ALERT_AGE = timedelta(hours=3)

# Alerts are skipped when the newest candle is older than this: the trade could have
# been invalidated since, so pushing it as live would be wrong.
MAX_DATA_AGE = timedelta(hours=2)

# A plain-summary report (narration failed) is cached only briefly, so a transient
# DeepSeek error is retried soon instead of being served for the full cache TTL.
FALLBACK_CACHE_TTL = timedelta(seconds=60)

# After this many consecutive failing hourly checks, the allowed chats get one warning.
ALERT_FAILURES_BEFORE_WARNING = 3

# APScheduler skips a run delayed by more than its grace period (default: 1 second).
ALERT_MISFIRE_GRACE_SECONDS = 600


def parse_allowed_chat_ids(raw: str | None) -> set[int]:
    """Ignores entries that aren't integers (e.g. a typo like '@me') instead of
    raising: a bad value crashed startup, and a crash loop exhausts the restart
    policy and takes the whole bot down. Ignoring fails closed for that entry."""
    ids = set()
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.add(int(part))
        except ValueError:
            log.warning("ignoring invalid TELEGRAM_ALLOWED_CHAT_IDS entry %r", part)
    return ids


def allowed_chat_ids() -> set[int]:
    return parse_allowed_chat_ids(os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS"))


def require_token() -> str:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set. Create a bot with @BotFather and add "
            "the token to .env."
        )
    return TELEGRAM_BOT_TOKEN
