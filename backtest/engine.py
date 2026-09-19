"""Orchestrates a full backtest for one pair: load data, compute swings/levels,
find sweep events (§1.3), and trace each one through setup.evaluate_setup.

Sweep-event detection collapses multiple simultaneously-swept stacked levels of the
same type into a single event per candle per direction. Physically, one wick can
dip below several unmitigated swing lows at once; treating each as a separate
"setup" would just duplicate-count the same price event into multiple identical
downstream trades (same sweep_extreme, same subsequent FVG/confirmation path) —
strategy_rules.md doesn't address this edge case explicitly, so this is a
deliberate implementation choice to avoid inflating signal counts, not a rule.
"""
from data_pipeline import storage

from . import rules, sessions
from .fractals import build_levels, find_swings
from .setup import MarketData, SetupResult, evaluate_setup


def load_market_data(pair: str) -> MarketData:
    daily = storage.load(pair, "D")
    h4 = storage.load(pair, "H4")
    h1 = storage.load(pair, "H1")

    h4 = find_swings(h4).reset_index(drop=True)
    h1 = h1.reset_index(drop=True)
    daily = daily.reset_index(drop=True)

    levels = build_levels(h4)

    h4_session = sessions.in_session_mask(h4["time"])
    h1_session = sessions.in_session_mask(h1["time"])

    return MarketData(pair=pair, h4=h4, h1=h1, daily=daily, levels=levels,
                       h4_session=h4_session, h1_session=h1_session)


def find_sweep_events(market: MarketData) -> list[tuple[int, str]]:
    high = market.h4["high"].to_numpy()
    low = market.h4["low"].to_numpy()
    close = market.h4["close"].to_numpy()
    session = market.h4_session
    n = len(market.h4)

    swing_lows = [lvl for lvl in market.levels if lvl.kind == "low"]
    swing_highs = [lvl for lvl in market.levels if lvl.kind == "high"]

    events: list[tuple[int, str]] = []
    for i in range(n):
        if not session.iloc[i]:
            continue

        for lvl in swing_lows:
            if lvl.index >= i:
                break
            if lvl.confirmed_at > i or not lvl.in_scope_of(i, rules.LIQUIDITY_LOOKBACK_H4):
                continue
            if not lvl.unmitigated_as_of(i):
                continue
            if low[i] < lvl.level < close[i]:
                events.append((i, "bullish"))
                break

        for lvl in swing_highs:
            if lvl.index >= i:
                break
            if lvl.confirmed_at > i or not lvl.in_scope_of(i, rules.LIQUIDITY_LOOKBACK_H4):
                continue
            if not lvl.unmitigated_as_of(i):
                continue
            if close[i] < lvl.level < high[i]:
                events.append((i, "bearish"))
                break

    return events


def run_pair_backtest(pair: str) -> list[SetupResult]:
    market = load_market_data(pair)
    events = find_sweep_events(market)
    return [evaluate_setup(market, direction, idx) for idx, direction in events]
