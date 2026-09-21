"""Builds the reports the bot sends: refresh data, compute structured state, narrate.

Narration is optional at the edges: if DeepSeek is down or misbehaves, a plain
deterministic summary of the same state is sent instead, so a signal is never
lost just because the formatting layer failed.
"""
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from live.chart import load_candles, render_chart
from live.freshness import is_stale
from live.plan import SEPARATOR, plan_text, when
from live.refresh import refresh_pair
from live.state import LiveState, compute_current_state
from narration.generate import narrate_state

from . import config
from .wording import NOTICE

log = logging.getLogger(__name__)


def _refresh_live_history(pair: str) -> None:
    refresh_pair(pair, years=config.LIVE_HISTORY_YEARS)


def _default_chart(state: LiveState) -> Optional[bytes]:
    return render_chart(load_candles(state.pair), state)


@dataclass
class Analysis:
    text: str
    chart: Optional[bytes]  # PNG, or None when the chart couldn't be drawn


@dataclass
class _CachedReport:
    analysis: Analysis
    built_at: datetime
    ttl: timedelta


def stale_text(state: LiveState) -> str:
    """Shown instead of a plan when the snapshot is too old to act on: a plan built
    from it would tell the reader to enter at a price that may be long gone."""
    return (f"⚠️ STALE DATA — the newest candle closed {when(state.as_of)}, longer ago than "
            f"the market being open explains. Nothing here is current, so no trade plan "
            f"is shown.\n\nLast known: {state.pair} at {state.current_price}, "
            f"daily bias {state.daily_bias.upper()}, as of {when(state.as_of)}.")


def fallback_text(state: LiveState) -> str:
    lines = [f"📊 {state.pair} — Structural Read (plain summary)",
             f"Price {state.current_price}  |  Daily bias: {state.daily_bias.upper()}"]
    if not state.active_setups:
        lines.append("No active setup.")
    for s in state.active_setups:
        lines.append(f"\n{s.direction.upper()} setup — {s.status}")
        lines.append(f"Sweep extreme {s.sweep_extreme} ({when(s.sweep_time)})")
        if s.confirmation_level is not None:
            lines.append(f"Confirmation level {s.confirmation_level}")
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
        chart: Callable[[LiveState], Optional[bytes]] = _default_chart,
    ):
        self._refresh = refresh
        self._compute = compute
        self._narrate = narrate
        self._clock = clock
        self._chart = chart
        self._cache: dict[str, _CachedReport] = {}
        # Re-entrant: get_report holds the pair's lock and calls fresh_state, which
        # takes it again on the same thread.
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, pair: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(pair, threading.RLock())

    def fresh_state(self, pair: str) -> LiveState:
        # Serialised per pair: the hourly alert job and /analysis both refresh the
        # same parquet files, and two concurrent read-merge-write cycles can lose
        # candles (a stale writer landing last replaces newer data).
        with self._lock_for(pair):
            self._refresh(pair)
            return self._compute(pair)

    def _narrated_body(self, state: LiveState) -> tuple[str, bool]:
        """Returns (text, narrated). Anything short of usable narrated text —
        an exception, None, an empty string — degrades to the plain summary."""
        try:
            text = self._narrate(state)
            if isinstance(text, str) and text.strip():
                return text, True
            log.warning("narration returned no usable text for %s; sending plain summary",
                        state.pair)
        except Exception as err:
            log.warning("narration failed for %s, sending plain summary: %s", state.pair, err)
        return fallback_text(state), False

    def _render(self, state: LiveState) -> tuple[str, bool]:
        """The trade plan comes first (it is what you act on), then the structural read.
        The plan is computed by the engine, not the narration model, so it is still
        there when narration falls back; and if rendering it ever fails, the read is
        still sent."""
        if is_stale(datetime.fromisoformat(state.as_of), self._clock(), config.MAX_DATA_AGE):
            log.warning("%s data is stale (as of %s); not rendering a trade plan",
                        state.pair, state.as_of)
            return stale_text(state) + SEPARATOR + NOTICE, False
        body, narrated = self._narrated_body(state)
        try:
            plan = plan_text(state)
        except Exception:
            log.exception("could not render the trade plan for %s; sending the read alone",
                          state.pair)
            plan = ""
        text = plan + SEPARATOR + body if plan else body
        return f"🕒 Data as of {when(state.as_of)}\n\n" + text + SEPARATOR + NOTICE, narrated

    def render(self, state: LiveState) -> str:
        return self._render(state)[0]

    def chart_for(self, state: LiveState) -> Optional[bytes]:
        """The chart is decoration: any failure drawing it (a missing font, bad
        data) must cost the reader the picture only, never the report."""
        try:
            return self._chart(state)
        except Exception as err:
            log.warning("chart failed for %s, sending text only: %s", state.pair, err)
            return None

    def get_analysis(self, pair: str) -> Analysis:
        with self._lock_for(pair):
            cached: Optional[_CachedReport] = self._cache.get(pair)
            now = self._clock()
            if cached is not None and now - cached.built_at < cached.ttl:
                return cached.analysis
            state = self.fresh_state(pair)
            text, narrated = self._render(state)
            analysis = Analysis(text=text, chart=self.chart_for(state))
            ttl = config.REPORT_CACHE_TTL if narrated else config.FALLBACK_CACHE_TTL
            self._cache[pair] = _CachedReport(analysis=analysis, built_at=self._clock(), ttl=ttl)
            return analysis

    def get_report(self, pair: str) -> str:
        return self.get_analysis(pair).text
