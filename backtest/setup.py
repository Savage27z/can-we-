"""Traces one sweep event (§2.1/§2.2) forward through FVG formation, daily-bias
filtering, H1 confirmation/invalidation (§2.3), target selection and R:R gating
(§6), to a final outcome (§6.5). Each sweep event is processed independently and
functionally — no shared mutable state between setups, which keeps concurrent
setups (e.g. two nearby sweeps a few candles apart) from interfering with each other,
since nothing in strategy_rules.md says a new sweep is blocked by another pending one.
"""
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from . import bias, invalidation, rules, xtf
from .fractals import LiquidityLevel
from .fvg import FVG, detect_fvg
from .resolution import resolve_on_closes, resolve_on_plan


@dataclass
class MarketData:
    pair: str
    h4: pd.DataFrame          # columns: time, open, high, low, close, swing_high, swing_low
    h1: pd.DataFrame          # columns: time, open, high, low, close
    daily: pd.DataFrame
    levels: list[LiquidityLevel]
    h4_session: pd.Series = field(repr=False)   # bool mask, per H4 candle open time
    h1_session: pd.Series = field(repr=False)   # bool mask, per H1 candle open time


OUTCOMES_NO_TRADE = {"no_fvg", "bias_block", "invalidated_pre_confirm", "expired", "no_target", "low_rr",
                     "skipped_rollover"}   # only when evaluate_setup is given no_entry_hours
OUTCOMES_TRADED = {"win", "loss", "open"}
# Non-terminal: the outcome isn't decided yet purely because we ran out of
# available data before the relevant window/resolution could complete — not
# because the strategy rules themselves ruled the setup out. In a full historical
# backtest these only show up in the last few candles of the dataset; in live use
# (Phase 3) this is the whole point — these ARE the actionable current states.
OUTCOMES_PENDING = {"pending_fvg", "pending_confirmation", "open"}


@dataclass
class SetupResult:
    pair: str
    direction: str            # "bullish" or "bearish"
    sweep_index: int
    sweep_time: pd.Timestamp
    sweep_extreme: float
    outcome: str
    fvg: Optional[FVG] = None
    h1_candles_to_confirm: Optional[int] = None
    confirm_index: Optional[int] = None
    confirm_time: Optional[pd.Timestamp] = None
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    r_price: Optional[float] = None
    reward_price: Optional[float] = None
    rr: Optional[float] = None
    resolve_index: Optional[int] = None
    resolve_time: Optional[pd.Timestamp] = None
    realized_r: Optional[float] = None


def _find_fvg(market: MarketData, sweep_index: int, required_direction: str) -> tuple[Optional[FVG], bool]:
    """§1.4/§1.5. Per §5 v1.3, the FVG's own candles are NOT session-checked —
    only the sweep and confirmation candles are (see strategy_rules.md §5 for why:
    requiring all 3 FVG candles to be session-qualified was killing ~half of all
    otherwise-valid gaps for no reason tied to the strategy's actual decision points).

    Returns (fvg_or_none, window_exhausted). window_exhausted is True only if every
    offset in the §1.5 window had real H4 data to check — False means we ran out of
    data (hit the end of the array) before the window closed out, which in live use
    means "still waiting", not "definitively no FVG" (§8.2's outcome).
    """
    high = market.h4["high"].to_numpy()
    low = market.h4["low"].to_numpy()
    for mid in range(sweep_index, sweep_index + rules.FVG_FORMATION_WINDOW_H4 + 1):
        if mid - 1 < 0 or mid + 1 >= len(high):
            return None, False
        candidate = detect_fvg(high, low, mid)
        if candidate is None or candidate.direction != required_direction:
            continue
        return candidate, True
    return None, True


def _crossed_by_h1(market: MarketData, kind: str, level: float, h4_ref_index: int,
                   through: pd.Timestamp, htf_period: pd.Timedelta) -> bool:
    """True if a completed H1 candle after the last closed H4 candle, up to `through`,
    has already traded through `level` (§1.6: a high is consumed by any later high above
    it, a low by any later low below it). Those H1 candles are known even though the H4
    candle they belong to is still forming; only what has actually happened is read."""
    h1_time = market.h1["time"]
    if h4_ref_index >= 0:
        start = market.h4["time"].iloc[h4_ref_index] + htf_period
    else:
        start = h1_time.iloc[0]
    lo = int(h1_time.searchsorted(start, side="left"))
    hi = int(h1_time.searchsorted(through, side="left"))   # candles opened before `through`
    if hi <= lo:
        return False
    if kind == "high":
        return bool(market.h1["high"].iloc[lo:hi].max() > level)
    return bool(market.h1["low"].iloc[lo:hi].min() < level)


def select_target(market: MarketData, direction: str, sweep_index: int,
                 entry_price: float, h4_ref_index: int,
                 h1_consumed_through: Optional[pd.Timestamp] = None,
                 htf_period: pd.Timedelta = pd.Timedelta(hours=4)) -> Optional[LiquidityLevel]:
    """§6.1. Public because the live plan asks the same question ("what would the
    target be at this entry?") before a trade has confirmed.

    Mitigation is judged on closed H4 candles. `h1_consumed_through` (opt-in, used by the
    live layer) additionally rejects a level that completed H1 candles inside the still-
    forming H4 candle have already traded through, up to that time (audit F03).

    `htf_period` is the higher timeframe's own candle duration (default 4h, matching "H4"
    in its name); a caller running the same logic on a different timeframe pair passes its
    real duration so "still forming" is judged correctly (see evaluate_setup)."""
    target_kind = "high" if direction == "bullish" else "low"
    candidates = [
        lvl for lvl in market.levels
        if lvl.kind == target_kind
        and lvl.confirmed_at <= h4_ref_index
        and lvl.in_scope_of(sweep_index, rules.LIQUIDITY_LOOKBACK_H4)
        and lvl.unmitigated_as_of(h4_ref_index + 1)
        and ((lvl.level > entry_price) if direction == "bullish" else (lvl.level < entry_price))
        and (h1_consumed_through is None
             or not _crossed_by_h1(market, lvl.kind, lvl.level, h4_ref_index, h1_consumed_through,
                                   htf_period))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda lvl: abs(lvl.level - entry_price))


def evaluate_setup(market: MarketData, direction: str, sweep_index: int,
                   no_entry_hours: tuple[int, ...] = (),
                   plan_exits: bool = False,
                   h1_mitigation: bool = False,
                   htf_period: pd.Timedelta = pd.Timedelta(hours=4),
                   ltf_period: pd.Timedelta = pd.Timedelta(hours=1)) -> SetupResult:
    """`no_entry_hours`: UTC open hours of an H1 candle that may not confirm a trade. A setup whose
    confirming candle opens at one of them ends as "skipped_rollover": it is dropped, not left
    waiting for a later candle, which is exactly the variant that was backtested
    (research's SweepFvgNoRollover). Empty (the default) is the rule as originally specified.

    `plan_exits` and `h1_mitigation` are the live layer's opt-ins (audit F02, F03); both default
    off, so the backtest and every research baseline keep the rules as originally scored.
    - plan_exits: resolve the trade the way the published plan runs it (hard stop and target
      that fire on a touch, plus a manual close on an invalidating H1 close) instead of on H1
      closes only, so a setup whose plan has ended is no longer reported as live.
    - h1_mitigation: a target already traded through by completed H1 candles inside the
      still-forming H4 candle is not eligible (§1.6).

    `htf_period`/`ltf_period`: the real candle duration of `market.h4`/`market.h1`, default 4h/1h
    to match their field names exactly. Every rule and every candle-count parameter (§1.5's FVG
    window, §2.3's confirmation window, §2.4's lookback) is unchanged by these — only the small
    number of places that convert an OPEN time to a CLOSE time (a candle's true close is its open
    plus its own duration) need the real value, so this same logic can run on any timeframe pair
    without silently mistiming those instants."""
    h4_time = market.h4["time"]
    h4_high = market.h4["high"].to_numpy()
    h4_low = market.h4["low"].to_numpy()

    sweep_time = h4_time.iloc[sweep_index]
    # §1.7: bearish sweep swept a swing high -> extreme is this candle's high;
    # bullish sweep swept a swing low -> extreme is this candle's low.
    sweep_extreme = h4_high[sweep_index] if direction == "bearish" else h4_low[sweep_index]

    base = dict(pair=market.pair, direction=direction, sweep_index=sweep_index,
                sweep_time=sweep_time, sweep_extreme=float(sweep_extreme))

    required_fvg_dir = direction  # §1.4: bullish sweep needs bullish FVG, bearish needs bearish
    fvg, fvg_window_exhausted = _find_fvg(market, sweep_index, required_fvg_dir)
    if fvg is None:
        return SetupResult(**base, outcome="no_fvg" if fvg_window_exhausted else "pending_fvg")

    # §2.3's reference point for the confirmation window: H4 candle (mid+1)'s close time.
    fvg_formed_time = h4_time.iloc[fvg.mid_index + 1] + htf_period

    # §4: daily bias filter, evaluated as of FVG formation (last gating condition, §2.1/§2.2 step 3).
    daily_bias = bias.bias_asof(market.daily, fvg_formed_time)
    if (direction == "bullish" and daily_bias == "bearish") or \
       (direction == "bearish" and daily_bias == "bullish"):
        return SetupResult(**base, outcome="bias_block", fvg=fvg)

    confirmation_level = fvg.confirmation_level
    h1_time = market.h1["time"]
    h1_close = market.h1["close"].to_numpy()
    j0 = xtf.h1_index_at_or_after(h1_time, fvg_formed_time)

    confirmed_index = None
    invalidated_pre_confirm = False
    confirmation_window_exhausted = False
    for offset in range(0, rules.CONFIRMATION_WINDOW_H1 + 1):
        j = j0 + offset
        if j >= len(h1_close):
            break
        c = h1_close[j]
        if invalidation.is_invalidated(direction, sweep_extreme, c):
            invalidated_pre_confirm = True
            break
        if market.h1_session.iloc[j]:
            confirmed = (c > confirmation_level) if direction == "bullish" else (c < confirmation_level)
            if confirmed:
                confirmed_index = j
                break
    else:
        confirmation_window_exhausted = True

    if confirmed_index is None:
        if invalidated_pre_confirm:
            outcome = "invalidated_pre_confirm"
        elif confirmation_window_exhausted:
            outcome = "expired"
        else:
            outcome = "pending_confirmation"
        return SetupResult(**base, outcome=outcome, fvg=fvg)

    if h1_time.iloc[confirmed_index].hour in no_entry_hours:
        return SetupResult(**base, outcome="skipped_rollover", fvg=fvg, confirm_index=confirmed_index,
                            confirm_time=h1_time.iloc[confirmed_index] + ltf_period,
                            entry_price=float(h1_close[confirmed_index]),
                            h1_candles_to_confirm=confirmed_index - j0)

    entry_price = float(h1_close[confirmed_index])
    # The confirming candle's CLOSE is the actual confirmation instant — that's
    # when entry_price is known, one hour after its open. Using the open time
    # here made every downstream "as of now" lookup (h4_ref_index below) up to
    # an hour too conservative, occasionally treating an already-mitigated
    # liquidity level as still tradeable (~1 in 4 confirmations, whenever the
    # true close lands exactly on an H4 boundary).
    confirm_time = h1_time.iloc[confirmed_index] + ltf_period
    h1_candles_to_confirm = confirmed_index - j0

    stop_price = rules.stop_price(market.pair, direction, sweep_extreme)

    h4_ref_index = xtf.h4_index_fully_closed_by(h4_time, confirm_time, period=htf_period)
    target_level = select_target(market, direction, sweep_index, entry_price, h4_ref_index,
                                 h1_consumed_through=confirm_time if h1_mitigation else None,
                                 htf_period=htf_period)
    if target_level is None:
        return SetupResult(**base, outcome="no_target", fvg=fvg, confirm_index=confirmed_index,
                            confirm_time=confirm_time, entry_price=entry_price, stop_price=stop_price,
                            h1_candles_to_confirm=h1_candles_to_confirm)

    target_price = target_level.level
    r_price = abs(entry_price - stop_price)
    reward_price = abs(target_price - entry_price)
    rr = (reward_price / r_price) if r_price > 0 else 0.0

    if rr < rules.MIN_RR:
        return SetupResult(**base, outcome="low_rr", fvg=fvg, confirm_index=confirmed_index,
                            confirm_time=confirm_time, entry_price=entry_price, stop_price=stop_price,
                            target_price=target_price, r_price=r_price, reward_price=reward_price,
                            rr=rr, h1_candles_to_confirm=h1_candles_to_confirm)

    # §6.5: resolve the trade, scanning H1 closes strictly after the confirmation candle.
    outcome = "open"
    resolve_index = None
    resolve_time = None
    fill_price = None
    if plan_exits:
        planned = resolve_on_plan(direction, market.h1["open"].to_numpy(),
                                  market.h1["high"].to_numpy(), market.h1["low"].to_numpy(),
                                  h1_close, confirmed_index + 1, sweep_extreme, stop_price,
                                  target_price)
        if planned is not None:
            resolve_index, reason, fill_price = planned
            outcome = "win" if reason == "target" else "loss"
            resolve_time = h1_time.iloc[resolve_index] + ltf_period
    else:
        resolution = resolve_on_closes(direction, h1_close, confirmed_index + 1,
                                       sweep_extreme, target_price)
        if resolution is not None:
            resolve_index, reason = resolution
            outcome = "loss" if reason == "invalidation" else "win"
            # close time, not open (see confirm_time above)
            resolve_time = h1_time.iloc[resolve_index] + ltf_period

    if fill_price is not None and r_price > 0:
        signed = (fill_price - entry_price) if direction == "bullish" else (entry_price - fill_price)
        realized_r = signed / r_price
    else:
        realized_r = rr if outcome == "win" else (-1.0 if outcome == "loss" else None)

    return SetupResult(**base, outcome=outcome, fvg=fvg, confirm_index=confirmed_index,
                        confirm_time=confirm_time, entry_price=entry_price, stop_price=stop_price,
                        target_price=target_price, r_price=r_price, reward_price=reward_price,
                        rr=rr, resolve_index=resolve_index, resolve_time=resolve_time,
                        realized_r=realized_r, h1_candles_to_confirm=h1_candles_to_confirm)
