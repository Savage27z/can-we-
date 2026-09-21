"""The liquidity sweep + fair value gap strategy of strategy_rules.md, as a plug-in.

This wraps the existing implementation rather than reimplementing it: sweeps are found
by backtest.engine.find_sweep_events and each is traced through backtest.setup's
evaluate_setup, the same code the live bot and the original backtest run, so the signals
are identical by construction. What changes is only who ends the trades: the engine's exit
policy does, instead of evaluate_setup's own close-based resolution (which the "close"
exit reproduces exactly).

Only setups that reach a trade (a confirmed entry, a target, and R:R at or above the
minimum) become signals; everything else is counted in the funnel.
"""
from collections import Counter

import pandas as pd

from backtest.engine import build_market_data, find_sweep_events
from backtest.setup import evaluate_setup

from ..strategy import Signal, StrategyOutput

TRADED = {"win", "loss", "open"}


class SweepFvg:
    name = "sweep_fvg"
    timeframes = ("D", "H4", "H1")

    def generate(self, instrument: str, frames: dict[str, pd.DataFrame]) -> StrategyOutput:
        market = build_market_data(instrument, frames["D"], frames["H4"], frames["H1"])
        results = [evaluate_setup(market, direction, index)
                   for index, direction in find_sweep_events(market)]

        signals = [
            Signal(
                instrument=instrument, direction=r.direction, entry_index=r.confirm_index,
                entry_time=r.confirm_time, entry_price=r.entry_price, stop_price=r.stop_price,
                target_price=r.target_price, invalidation_price=r.sweep_extreme,
                meta={"sweep_time": r.sweep_time,
                      "h1_candles_to_confirm": r.h1_candles_to_confirm},
            )
            for r in results if r.outcome in TRADED
        ]
        h4_times = market.h4["time"]
        months = (h4_times.iloc[-1] - h4_times.iloc[0]).total_seconds() / (3600 * 24 * 30.44) \
            if len(h4_times) else 0.0
        return StrategyOutput(signals=signals, funnel=dict(Counter(r.outcome for r in results)),
                              months_spanned=months)
