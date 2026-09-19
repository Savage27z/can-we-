"""Configuration for the OANDA historical data pipeline.

Pairs/timeframes/lookback here match strategy_rules.md's timeframe roles
(Daily = bias, H4 = sweep+FVG, H1 = confirmation) — Phase 2's backtester
reads directly from the parquet files this pipeline produces.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Overridable so a deployment can point this at a persistent volume; everything
# stored here (candle parquet files, news cache, alert log) lives under it.
DATA_DIR = Path(os.environ.get("DATA_DIR") or PROJECT_ROOT / "data")

# OANDA instrument codes (underscore format, e.g. "EUR_USD").
PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY"]

# Maps our internal timeframe name -> OANDA granularity string.
TIMEFRAMES = {
    "D": "D",
    "H4": "H4",
    "H1": "H1",
}

DEFAULT_LOOKBACK_YEARS = 8

OANDA_API_KEY = os.environ.get("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID")
OANDA_ENVIRONMENT = os.environ.get("OANDA_ENVIRONMENT", "practice")

_HOSTS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}


def oanda_host() -> str:
    try:
        return _HOSTS[OANDA_ENVIRONMENT]
    except KeyError:
        raise ValueError(
            f"OANDA_ENVIRONMENT must be 'practice' or 'live', got {OANDA_ENVIRONMENT!r}"
        )


def require_credentials() -> None:
    if not OANDA_API_KEY:
        raise RuntimeError(
            "OANDA_API_KEY is not set. Copy .env.example to .env and fill in your "
            "OANDA personal access token (Account Settings -> My Access Tokens on "
            "the OANDA practice or live site)."
        )
