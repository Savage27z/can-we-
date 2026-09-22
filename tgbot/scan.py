"""/scan: where every enabled pair stands against the rules right now, in one message.

This is the answer to "I don't want to wait for an alert": ask whenever you like and see, for
each pair, whether anything is forming and how far it is from confirming. It is a read of the
rules' state, computed from the same LiveState as /analysis, with no narration model involved
(so it is fast and costs nothing per call). It is not a signal: only a setup the rules have
confirmed is ever shown as live, and the rules are strict, so most of the time most pairs show
nothing in play.
"""
from datetime import datetime
from typing import Union

from backtest import rules
from live.freshness import is_stale
from live.plan import when
from live.state import ActiveSetup, LiveState

from . import config

BIAS_ICONS = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}
_STAGE_RANK = {"live_trade": 0, "pending_confirmation": 1, "pending_fvg": 2}

ScanResult = Union[LiveState, Exception]


def _price(pair: str, value: float) -> str:
    return format(value, f".{3 if pair.endswith('_JPY') else 5}f")


def _most_advanced(setups: list[ActiveSetup]) -> ActiveSetup:
    # Newest sweep first, then the furthest stage: a live trade beats a waiting one.
    ranked = sorted(setups, key=lambda s: s.sweep_time, reverse=True)
    ranked.sort(key=lambda s: _STAGE_RANK.get(s.status, len(_STAGE_RANK)))
    return ranked[0]


def _setup_phrase(state: LiveState, setup: ActiveSetup) -> str:
    pair = state.pair
    bullish = setup.direction == "bullish"
    if setup.status == "live_trade":
        reward = f", {setup.rr:.1f}R" if setup.rr is not None else ""
        return (f"LIVE {setup.direction}: entry {_price(pair, setup.entry_price)}, "
                f"stop {_price(pair, setup.stop_price)}, target {_price(pair, setup.target_price)}"
                f"{reward}")
    if setup.status == "pending_confirmation" and setup.confirmation_level is not None:
        pips = abs(setup.confirmation_level - state.current_price) / rules.pip_size(pair)
        side = "above" if bullish else "below"
        return (f"{setup.direction} setup forming: needs an H1 close {side} "
                f"{_price(pair, setup.confirmation_level)} ({pips:.0f} pips from price)")
    return f"{setup.direction} sweep {when(setup.sweep_time)}, waiting for a gap to form"


def pair_line(state: LiveState, now: datetime) -> str:
    icon = BIAS_ICONS.get(state.daily_bias, "⚪")
    head = f"{icon} {state.pair.replace('_', '/')} · {state.daily_bias}"
    if is_stale(datetime.fromisoformat(state.as_of), now, config.MAX_DATA_AGE):
        return f"{head} · ⚠️ data is stale (as of {when(state.as_of)}), not reading it"
    if not state.active_setups:
        text = "no setup in play"
    else:
        best = _most_advanced(state.active_setups)
        text = _setup_phrase(state, best)
        if len(state.active_setups) > 1:
            text += f" (+{len(state.active_setups) - 1} more)"
    if state.news.status == "blackout":
        text += " · ⚠️ news blackout"
    return f"{head}\n    {text}"


def _rank(pair: str, result: ScanResult) -> tuple:
    if isinstance(result, Exception) or not result.active_setups:
        return (len(_STAGE_RANK), config.LIVE_PAIRS.index(pair))
    best = _most_advanced(result.active_setups)
    return (_STAGE_RANK.get(best.status, len(_STAGE_RANK)), config.LIVE_PAIRS.index(pair))


def scan_text(results: dict[str, ScanResult], now: datetime) -> str:
    """Pairs with something in play first (live trades, then waiting setups), the rest after."""
    ordered = sorted(results.items(), key=lambda item: _rank(*item))
    lines = []
    for pair, result in ordered:
        if isinstance(result, Exception):
            lines.append(f"⚠️ {pair.replace('_', '/')}\n    couldn't refresh the data right now")
        else:
            lines.append(pair_line(result, now))

    states = [r for r in results.values() if not isinstance(r, Exception)]
    header = "📡 Market scan: where each pair stands against the rules"
    if states:
        oldest = min(states, key=lambda s: s.as_of)
        header += f"\n🕒 Data as of {when(oldest.as_of)}"
    in_play = sum(1 for r in states if r.active_setups)
    if states and in_play == 0:
        summary = ("Nothing is in play on any pair right now. The rules are strict, so this is "
                   "the usual state; an alert fires the moment one confirms.")
    else:
        summary = f"{in_play} of {len(results)} pairs have something in play."
    footer = "Full read and chart for one pair: /analysis PAIR (for example /analysis GBP_USD)."
    return "\n\n".join([header, summary, "\n\n".join(lines), footer,
                        "ℹ️ A read of the rules, not a trade signal or financial advice."])
