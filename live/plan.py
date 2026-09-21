"""Turns each active setup into a concrete, mechanical trade plan: where to enter,
where to put the stop-loss and the take-profit, and when the setup is off.

Everything here is arithmetic over numbers the engine already computed (§3, §6 of
strategy_rules.md); nothing is estimated and nothing is left to a language model. The
text is rendered here too, not by the narration layer, so a trader acting on it never
depends on the formatting model being up or well-behaved.

Three stages, matching the setup's status:
- live_trade: the engine's own entry, stop, target and R:R.
- pending_confirmation: the same, planned ahead. The stop is known (it comes from the
  sweep) and the entry is at least the trigger level; the target is what the engine
  would pick if it confirmed right now, so it is labelled provisional.
- pending_fvg: no entry exists yet, so only the stop and the cancel level are shown.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from backtest import rules

if TYPE_CHECKING:  # state.py imports this module, so import back for typing only
    from .state import ActiveSetup, LiveState

MAX_PLANS_SHOWN = 3
_STAGE_ORDER = {"live_trade": 0, "pending_confirmation": 1, "pending_fvg": 2}
SEPARATOR = "\n\n———\n\n"


@dataclass
class TradePlan:
    side: str                       # "BUY" or "SELL"
    stop: float                     # sweep extreme +/- buffer
    invalidation: float             # the sweep extreme: an H1 close beyond it cancels the setup
    entry: Optional[float] = None   # live: the signal price; pending_confirmation: the trigger level
    target: Optional[float] = None
    target_provisional: bool = False
    rr: Optional[float] = None
    worst_entry: Optional[float] = None  # entering beyond this leaves R:R under the minimum
    stop_pips: Optional[float] = None
    target_pips: Optional[float] = None


def worst_entry(stop: float, target: float) -> float:
    """The entry at which R:R is exactly the minimum. For a buy, anything higher is
    worse; for a sell, anything lower. Solves |target - e| = MIN_RR * |e - stop|."""
    return (target + rules.MIN_RR * stop) / (1 + rules.MIN_RR)


def _plain(value) -> Optional[float]:
    """Engine values can be numpy scalars; keep the plan (and its JSON) plain floats."""
    return None if value is None else float(value)


def plan_for(pair: str, setup: "ActiveSetup", provisional_target: Optional[float] = None) -> TradePlan:
    """`provisional_target` is only used while the setup awaits confirmation; the
    caller picks it with the engine's own target rule (it needs the market data)."""
    stop = _plain(setup.stop_price if setup.stop_price is not None
                  else rules.stop_price(pair, setup.direction, setup.sweep_extreme))
    plan = TradePlan(side="BUY" if setup.direction == "bullish" else "SELL",
                     stop=stop, invalidation=_plain(setup.sweep_extreme))

    if setup.status == "live_trade":
        plan.entry, plan.target = _plain(setup.entry_price), _plain(setup.target_price)
    elif setup.status == "pending_confirmation":
        plan.entry, plan.target = _plain(setup.confirmation_level), _plain(provisional_target)
        plan.target_provisional = True
    else:
        return plan

    if plan.entry is None:
        return plan
    pip = rules.pip_size(pair)
    plan.stop_pips = abs(plan.entry - stop) / pip
    if plan.target is not None:
        plan.target_pips = abs(plan.target - plan.entry) / pip
        risk = abs(plan.entry - stop)
        plan.rr = _plain(setup.rr) if setup.status == "live_trade" and setup.rr is not None \
            else (abs(plan.target - plan.entry) / risk if risk > 0 else None)
        plan.worst_entry = worst_entry(stop, plan.target)
    return plan


def _decimals(pair: str) -> int:
    return 3 if pair.endswith("_JPY") else 5


def _session_hours() -> str:
    """The open hours of the H1 candles that can confirm: both sessions (bounds inclusive, §5)
    minus the rollover hours the live bot does not trade (§5 v1.7), as ranges."""
    hours = sorted({h for lo, hi in (rules.LONDON_SESSION_UTC, rules.NEWYORK_SESSION_UTC)
                    for h in range(lo, hi + 1)} - set(rules.NO_ENTRY_HOURS_UTC))
    runs, start = [], hours[0]
    for previous, hour in zip(hours, hours[1:] + [None]):
        if hour != previous + 1:
            runs.append(f"{start:02d}:00–{previous:02d}:00")
            start = hour
    return ", ".join(runs) + " UTC"


def _rollover_note() -> str:
    hours = ", ".join(f"{h:02d}:00" for h in rules.NO_ENTRY_HOURS_UTC)
    return (f"If the candle that confirms it opens at {hours} UTC (it closes around the daily "
            f"rollover, when spreads are widest) the setup is skipped.")


def _when(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%a %d %b %H:%M UTC")


def _block(pair: str, setup: "ActiveSetup", plan: TradePlan) -> str:
    def price(value: float) -> str:
        return format(value, f".{_decimals(pair)}f")

    pip = rules.pip_size(pair)
    buffer_pips = round(rules.stop_buffer(pair) / pip)
    buy = plan.side == "BUY"
    pretty = pair.replace("_", "/")
    beyond = "below" if buy else "above"        # the side of the entry that loses
    through = "above" if buy else "below"       # the side of the entry that wins
    extreme_name = "sweep low" if buy else "sweep high"
    worst_side = "at or below" if buy else "at or above"
    cancel = (f"Cancelled if an H1 candle CLOSES {beyond} {price(plan.invalidation)} "
              f"({extreme_name}) first.")
    stop_note = f"{extreme_name} {price(plan.invalidation)} {'−' if buy else '+'} {buffer_pips} pips"

    if setup.status == "live_trade":
        lines = [f"🎯 TRADE PLAN — {plan.side} {pretty} · LIVE"]
        if setup.confirm_time and setup.confirmation_level is not None:
            lines.append(f"Confirmed by the H1 candle that closed {through} "
                         f"{price(setup.confirmation_level)} at {_when(setup.confirm_time)}.")
        lines.append(f"• Entry: {plan.side} at market. Signal price {price(plan.entry)}.")
        if plan.worst_entry is not None:
            lines.append(f"  Skip it if you can't fill {worst_side} {price(plan.worst_entry)} "
                         f"(R:R would fall under {rules.MIN_RR:g}).")
        lines.append(f"• Stop-loss: {price(plan.stop)} ({plan.stop_pips:.1f} pips {beyond} entry)")
        if plan.target is not None:
            lines.append(f"• Take-profit: {price(plan.target)} ({plan.target_pips:.1f} pips "
                         f"{through} entry)")
        if plan.rr is not None:
            lines.append(f"• R:R: 1 : {plan.rr:.2f}")
        lines.append(f"• Also close it by hand if an H1 candle CLOSES {beyond} "
                     f"{price(plan.invalidation)} ({extreme_name}) — the strategy's invalidation, "
                     f"which can come before your stop is hit.")
        return "\n".join(lines)

    if setup.status == "pending_confirmation":
        lines = [f"🎯 TRADE PLAN — {plan.side} {pretty} · NOT ACTIVE YET"]
        lines.append(f"• Trigger: wait for an H1 candle to CLOSE {through} {price(plan.entry)}. "
                     f"Only candles opening {_session_hours()} count, and the setup lapses "
                     f"if none does within {rules.CONFIRMATION_WINDOW_H1} H1 candles. "
                     f"{_rollover_note()}")
        entry = f"• Entry: {plan.side} at market once it closes, at about that candle's close."
        if plan.worst_entry is not None:
            entry += (f" Skip it if the close is {'above' if buy else 'below'} "
                      f"{price(plan.worst_entry)} (R:R under {rules.MIN_RR:g}).")
        lines.append(entry)
        lines.append(f"• Stop-loss: {price(plan.stop)} ({stop_note})")
        if plan.target is not None:
            lines.append(f"• Take-profit: {price(plan.target)} — provisional, the engine "
                         f"re-picks it when the trigger closes")
            if plan.rr is not None:
                lines.append(f"• R:R at the trigger level: 1 : {plan.rr:.2f}")
                if plan.rr < rules.MIN_RR:
                    lines.append(f"  ⚠️ Under the {rules.MIN_RR:g} minimum, so expect this to be skipped.")
        else:
            lines.append("• Take-profit: no unmitigated target available right now — the "
                         "strategy skips a trade that has none.")
        lines.append(f"• {cancel}")
        return "\n".join(lines)

    lines = [f"🎯 TRADE PLAN — {plan.side} {pretty} · NOTHING TO ENTER YET",
             f"• No entry, stop-loss or take-profit exist until a "
             f"{'bullish' if buy else 'bearish'} H4 fair value gap forms (within "
             f"{rules.FVG_FORMATION_WINDOW_H4} H4 candles of the sweep) and an H1 candle "
             f"closes through it.",
             f"• If that happens, the stop-loss goes at {price(plan.stop)} ({stop_note}).",
             f"• {cancel}"]
    return "\n".join(lines)


def plan_text(state: "LiveState") -> str:
    """The plan for every active setup, most actionable first. Empty when setups exist
    but carry no plan (a state built by hand rather than by the engine)."""
    if not state.active_setups:
        return "🎯 TRADE PLAN — none. There is no active setup, so there is nothing to enter."
    # Newest sweep first, then a stable sort by stage so live trades lead.
    ranked = sorted((s for s in state.active_setups if s.plan is not None),
                    key=lambda s: s.sweep_time, reverse=True)
    ranked.sort(key=lambda s: _STAGE_ORDER.get(s.status, len(_STAGE_ORDER)))
    return "\n\n".join(_block(state.pair, s, s.plan) for s in ranked[:MAX_PLANS_SHOWN])
