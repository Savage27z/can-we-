"""Computes the current structured state for a pair: Daily bias plus any setup
that isn't yet terminally resolved (§2.1-§2.3, §6 of strategy_rules.md). Reuses
the exact same evaluate_setup used by the backtester — the only difference in
live use is that OUTCOMES_PENDING results are the interesting ones (a fully
historical run would eventually resolve everything to a terminal outcome; live
is always sitting at the "not resolved yet" edge).

Per Phase 3's scope: outputs structured data (bias, level, invalidation, target)
only. No prose, no narration — that's Phase 4's job.
"""
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd

from backtest import bias, rules, xtf
from backtest.engine import find_sweep_events, load_market_data
from backtest.setup import MarketData, OUTCOMES_PENDING, SetupResult, evaluate_setup
from news.filter import NewsStatus, check_news

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


def compute_state_from_market(market: MarketData, news: Optional[NewsStatus] = None) -> LiveState:
    events = find_sweep_events(market)
    results = [evaluate_setup(market, direction, idx) for idx, direction in events]
    active = [r for r in results if r.outcome in OUTCOMES_PENDING]

    # The latest stored H1 row's `time` is its OPEN — since storage only ever
    # keeps complete candles, its close (one hour later) is the true "as of"
    # instant: that's when current_price (its close) actually became known.
    as_of: pd.Timestamp = market.h1["time"].iloc[-1] + pd.Timedelta(hours=1)
    current_price = float(market.h1["close"].iloc[-1])
    current_bias = bias.bias_asof(market.daily, as_of)
    buy_side, sell_side = _liquidity_map(market, as_of, current_price)

    return LiveState(
        pair=market.pair,
        as_of=as_of.isoformat(),
        current_price=current_price,
        daily_bias=current_bias,
        active_setups=[_to_active_setup(r, market.h4["time"]) for r in active],
        liquidity_buy_side=buy_side,
        liquidity_sell_side=sell_side,
        news=news if news is not None else NewsStatus.not_checked(),
    )


def compute_current_state(pair: str, with_news: bool = True) -> LiveState:
    # News is checked against wall-clock time, not `as_of`: the data can be up to
    # an hour behind, but a release happening right now is what matters.
    news = check_news(pair) if with_news else None
    return compute_state_from_market(load_market_data(pair), news=news)
