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

import numpy as np
import pandas as pd

from backtest import sessions
from backtest.engine import build_market_data, find_sweep_events
from backtest.setup import evaluate_setup

from ..strategy import Signal, StrategyOutput

TRADED = {"win", "loss", "open"}


class SweepFvg:
    name = "sweep_fvg"
    timeframes = ("D", "H4", "H1")

    def __init__(self, skip_open_hours: tuple[int, ...] = ()):
        """`skip_open_hours`: drop signals whose entry candle opens at one of these UTC hours."""
        self.skip_open_hours = tuple(skip_open_hours)

    def entry_mask(self, h1: pd.DataFrame) -> np.ndarray:
        """The candles a confirmation can happen on: in-session, minus any skipped hour."""
        mask = sessions.in_session_mask(h1["time"]).to_numpy()
        if self.skip_open_hours:
            mask = mask & ~np.isin(h1["time"].dt.hour.to_numpy(), self.skip_open_hours)
        return mask

    def generate(self, instrument: str, frames: dict[str, pd.DataFrame]) -> StrategyOutput:
        market = build_market_data(instrument, frames["D"], frames["H4"], frames["H1"])
        results = [evaluate_setup(market, direction, index)
                   for index, direction in find_sweep_events(market)]

        h1_hours = market.h1["time"].dt.hour.to_numpy()
        funnel = Counter()
        signals = []
        for r in results:
            if r.outcome not in TRADED:
                funnel[r.outcome] += 1
            elif h1_hours[r.confirm_index] in self.skip_open_hours:
                funnel["skipped_entry_hour"] += 1
            else:
                funnel[r.outcome] += 1
                signals.append(Signal(
                    instrument=instrument, direction=r.direction, entry_index=r.confirm_index,
                    entry_time=r.confirm_time, entry_price=r.entry_price, stop_price=r.stop_price,
                    target_price=r.target_price, invalidation_price=r.sweep_extreme,
                    meta={"sweep_time": r.sweep_time,
                          "h1_candles_to_confirm": r.h1_candles_to_confirm}))
        h4_times = market.h4["time"]
        months = (h4_times.iloc[-1] - h4_times.iloc[0]).total_seconds() / (3600 * 24 * 30.44) \
            if len(h4_times) else 0.0
        return StrategyOutput(signals=signals, funnel=dict(funnel), months_spanned=months)


class SweepFvgNoRollover(SweepFvg):
    """The stage 3 hygiene fix: sweep/FVG exactly as before, minus entries in the 21:00 UTC candle,
    which closes at the New York rollover when spreads are widest."""
    name = "sweep_fvg_no_rollover"

    def __init__(self):
        super().__init__(skip_open_hours=(21,))
