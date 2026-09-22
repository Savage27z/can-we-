"""Computes the current structured state for a pair: Daily bias plus any setup
that isn't yet terminally resolved (§2.1-§2.3, §6 of strategy_rules.md). Reuses
the exact same evaluate_setup used by the backtester — the only difference in
live use is that OUTCOMES_PENDING results are the interesting ones (a fully
historical run would eventually resolve everything to a terminal outcome; live
is always sitting at the "not resolved yet" edge).

Per Phase 3's scope: outputs structured data (bias, level, invalidation, target)
only. No prose, no narration — that's Phase 4's job.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional

import pandas as pd

from backtest import bias, rules, xtf
from backtest.engine import find_sweep_events, load_market_data
from backtest.setup import (MarketData, OUTCOMES_NO_TRADE, OUTCOMES_PENDING, SetupResult,
                            evaluate_setup, select_target)
from news.filter import NewsStatus, check_news

from .plan import TradePlan, plan_for

_LIVE_STATUS_LABELS = {
    "pending_fvg": "pending_fvg",
    "pending_confirmation": "pending_confirmation",
    "open": "live_trade",
}

# How many liquidity levels to surface each side in the live view. Not a
# strategy rule (strategy_rules.md only ever needs the single nearest opposing
# level, §6.1) — this is purely a wider "map" for narration/display purposes,
# built from the same real, already-computed swing/mitigation data.
LIQUIDITY_MAP_SIZE = 5

# How far back /scan and chat explain a pair's "nothing in play" with the most recent
# rejected sweep. Not a strategy rule — purely how far back the read-only "why nothing
# happened" view looks; it never alters a setup's own outcome.
REJECTION_LOOKBACK_H4 = 24    # ~4 days of H4 candles
MAX_RECENT_REJECTIONS = 1


@dataclass
class ActiveSetup:
    direction: str
    status: str
    sweep_time: str
    sweep_extreme: float
    fvg_low: Optional[float]
    fvg_high: Optional[float]
    confirmation_level: Optional[float]
    entry_price: Optional[float]
    stop_price: Optional[float]
    target_price: Optional[float]
    rr: Optional[float]
    h1_candles_to_confirm: Optional[int]
    confirm_time: Optional[str]  # close time of the confirming H1 candle (ISO), None until confirmed
    fvg_start_time: Optional[str] = None  # open time of the FVG's first candle (ISO); chart placement only
    plan: Optional[TradePlan] = None      # rendered by live.plan, never by the narration model


@dataclass
class RecentRejection:
    """A sweep that reached a final no-trade outcome recently: why /scan or chat can say
    something more useful than "no setup in play" when nothing is currently active."""
    direction: str
    sweep_time: str
    outcome: str                       # one of backtest.setup.OUTCOMES_NO_TRADE
    rr: Optional[float] = None         # populated for "low_rr"
    confirm_time: Optional[str] = None  # populated once the setup had confirmed (e.g. "skipped_rollover")


@dataclass
class LiveState:
    pair: str
    as_of: str
    current_price: float
    daily_bias: str
    active_setups: list[ActiveSetup]
    liquidity_buy_side: list[float]   # nearest unmitigated swing highs above price
    liquidity_sell_side: list[float]  # nearest unmitigated swing lows below price
    news: NewsStatus                  # overlay only; never alters setups, bias, or levels
    recent_rejections: list[RecentRejection] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _to_active_setup(result: SetupResult, h4_times: pd.Series) -> ActiveSetup:
    return ActiveSetup(
        direction=result.direction,
        status=_LIVE_STATUS_LABELS[result.outcome],
        sweep_time=result.sweep_time.isoformat(),
        sweep_extreme=result.sweep_extreme,
        fvg_low=result.fvg.low if result.fvg else None,
        fvg_high=result.fvg.high if result.fvg else None,
        confirmation_level=result.fvg.confirmation_level if result.fvg else None,
        entry_price=result.entry_price,
        stop_price=result.stop_price,
        target_price=result.target_price,
        rr=result.rr,
        h1_candles_to_confirm=result.h1_candles_to_confirm,
        confirm_time=result.confirm_time.isoformat() if result.confirm_time is not None else None,
        fvg_start_time=(h4_times.iloc[result.fvg.mid_index - 1].isoformat()
                        if result.fvg else None),
    )


def _provisional_target(market: MarketData, result: SetupResult, as_of: pd.Timestamp) -> Optional[float]:
    """What §6.1 would pick if the setup confirmed right now, entering at the trigger
    level. The real target is chosen at the confirming candle's close, so this is
    a forecast; the plan labels it as such."""
    if result.fvg is None:
        return None
    h4_ref_index = xtf.h4_index_fully_closed_by(market.h4["time"], as_of)
    if h4_ref_index < 0:
        return None
    level = select_target(market, result.direction, result.sweep_index,
                          result.fvg.confirmation_level, h4_ref_index,
                          h1_consumed_through=as_of)
    return level.level if level is not None else None


def _liquidity_map(market: MarketData, as_of: pd.Timestamp, current_price: float):
    """Nearest unmitigated swing highs/lows to current price, using the exact same
    LiquidityLevel objects (and the same §2.4 lookback window) the strategy engine
    itself computes — this is a read-only wider view over real data, not a new
    source of levels the LLM could be tempted to invent.
    """
    h4_ref_index = xtf.h4_index_fully_closed_by(market.h4["time"], as_of)
    if h4_ref_index < 0:
        return [], []

    live_candidates = [
        lvl for lvl in market.levels
        if lvl.confirmed_at <= h4_ref_index
        and lvl.in_scope_of(h4_ref_index, rules.LIQUIDITY_LOOKBACK_H4)
        and lvl.unmitigated_as_of(h4_ref_index + 1)
    ]
    buy_side = sorted(
        (lvl.level for lvl in live_candidates if lvl.kind == "high" and lvl.level > current_price),
        key=lambda level: level - current_price,
    )[:LIQUIDITY_MAP_SIZE]
    sell_side = sorted(
        (lvl.level for lvl in live_candidates if lvl.kind == "low" and lvl.level < current_price),
        key=lambda level: current_price - level,
    )[:LIQUIDITY_MAP_SIZE]
    return buy_side, sell_side


def _recent_rejections(results: list[SetupResult], as_of: pd.Timestamp) -> list[RecentRejection]:
    """The most recent sweep(s) that reached a final no-trade outcome, newest first — reusing
    `results`, which the engine already computed for every sweep in view, not a second pass."""
    cutoff = as_of - pd.Timedelta(hours=4 * REJECTION_LOOKBACK_H4)
    rejected = sorted(
        (r for r in results if r.outcome in OUTCOMES_NO_TRADE and r.sweep_time >= cutoff),
        key=lambda r: r.sweep_time, reverse=True,
    )
    return [
        RecentRejection(
            direction=r.direction, sweep_time=r.sweep_time.isoformat(), outcome=r.outcome,
            rr=r.rr, confirm_time=r.confirm_time.isoformat() if r.confirm_time is not None else None,
        )
        for r in rejected[:MAX_RECENT_REJECTIONS]
    ]


def compute_state_from_market(market: MarketData, news: Optional[NewsStatus] = None) -> LiveState:
    events = find_sweep_events(market)
    # The live bot applies the §5 (v1.7) rollover rule and the v1.8 plan exits / H1 target
    # mitigation; the backtest defaults do not (see evaluate_setup).
    results = [evaluate_setup(market, direction, idx, no_entry_hours=rules.NO_ENTRY_HOURS_UTC,
                              plan_exits=True, h1_mitigation=True)
               for idx, direction in events]
    active = [r for r in results if r.outcome in OUTCOMES_PENDING]

    # The latest stored H1 row's `time` is its OPEN — since storage only ever
    # keeps complete candles, its close (one hour later) is the true "as of"
    # instant: that's when current_price (its close) actually became known.
    as_of: pd.Timestamp = market.h1["time"].iloc[-1] + pd.Timedelta(hours=1)
    current_price = float(market.h1["close"].iloc[-1])
    current_bias = bias.bias_asof(market.daily, as_of)
    buy_side, sell_side = _liquidity_map(market, as_of, current_price)

    setups = []
    for result in active:
        setup = _to_active_setup(result, market.h4["time"])
        provisional = (_provisional_target(market, result, as_of)
                       if setup.status == "pending_confirmation" else None)
        setup.plan = plan_for(market.pair, setup, provisional)
        setups.append(setup)

    return LiveState(
        pair=market.pair,
        as_of=as_of.isoformat(),
        current_price=current_price,
        daily_bias=current_bias,
        active_setups=setups,
        liquidity_buy_side=buy_side,
        liquidity_sell_side=sell_side,
        news=news if news is not None else NewsStatus.not_checked(),
        recent_rejections=_recent_rejections(results, as_of),
    )


def compute_current_state(pair: str, with_news: bool = True) -> LiveState:
    # News is checked against wall-clock time, not `as_of`: the data can be up to
    # an hour behind, but a release happening right now is what matters.
    news = check_news(pair) if with_news else None
    return compute_state_from_market(load_market_data(pair), news=news)
