"""Builds the reports the bot sends: refresh data, compute structured state, narrate.

Narration is optional at the edges: if DeepSeek is down or misbehaves, a plain
deterministic summary of the same state is sent instead, so a signal is never
lost just because the formatting layer failed.
"""
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

import requests

from live.refresh import refresh_pair
from live.state import LiveState, compute_current_state
from narration.deepseek_client import DeepSeekAPIError
from narration.generate import narrate_state

from . import config

log = logging.getLogger(__name__)


def _refresh_live_history(pair: str) -> None:
    refresh_pair(pair, years=config.LIVE_HISTORY_YEARS)


@dataclass
class _CachedReport:
    text: str
    built_at: datetime


def fallback_text(state: LiveState) -> str:
    lines = [f"📊 {state.pair} — Structural Read (plain summary)",
             f"Price {state.current_price}  |  Daily bias: {state.daily_bias.upper()}"]
    if not state.active_setups:
        lines.append("No active setup.")
    for s in state.active_setups:
        lines.append(f"\n{s.direction.upper()} setup — {s.status}")
        lines.append(f"Sweep extreme {s.sweep_extreme} ({s.sweep_time})")
        if s.confirmation_level is not None:
            lines.append(f"Confirmation level {s.confirmation_level}")
        if s.entry_price is not None:
            lines.append(f"Entry {s.entry_price}  Stop {s.stop_price}  "
                         f"Target {s.target_price}  R:R {s.rr:.2f}" if s.rr is not None
                         else f"Entry {s.entry_price}  Stop {s.stop_price}")
    news = state.news
    if news.status == "blackout":
        for e in news.blackout_events:
            lines.append(f"⚠️ News blackout: {e.currency} {e.title} — {e.when}")
    elif news.status == "unavailable":
        lines.append(f"News calendar unavailable: {news.reason}")
    return "\n".join(lines)


class ReportService:
    def __init__(
        self,
        refresh: Callable[[str], None] = _refresh_live_history,
        compute: Callable[[str], LiveState] = compute_current_state,
        narrate: Callable[[LiveState], str] = narrate_state,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self._refresh = refresh
        self._compute = compute
        self._narrate = narrate
        self._clock = clock
        self._cache: dict[str, _CachedReport] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, pair: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(pair, threading.Lock())

    def fresh_state(self, pair: str) -> LiveState:
        self._refresh(pair)
        return self._compute(pair)

    def render(self, state: LiveState) -> str:
        try:
            return self._narrate(state)
        except (DeepSeekAPIError, RuntimeError, requests.RequestException) as err:
            log.warning("narration failed for %s, sending plain summary: %s", state.pair, err)
            return fallback_text(state)

    def get_report(self, pair: str) -> str:
        with self._lock_for(pair):
            cached: Optional[_CachedReport] = self._cache.get(pair)
            now = self._clock()
            if cached is not None and now - cached.built_at < config.REPORT_CACHE_TTL:
                return cached.text
            text = self.render(self.fresh_state(pair))
            self._cache[pair] = _CachedReport(text=text, built_at=self._clock())
            return text
