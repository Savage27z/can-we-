"""Runs a strategy on an instrument: load the candles, ask the strategy for signals, end
each trade with an exit policy, charge trading costs, and return one row per trade.

Each signal is resolved on its own: the engine does not stop a new trade opening while an
earlier one on the same instrument is still running, exactly like the original backtest.
Portfolio limits (one trade at a time, a cap on correlated exposure) are a separate layer.
"""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest import rules
from data_pipeline import storage

from .exits import Candles, ExitPolicy
from .strategy import BULLISH, Signal, Strategy

EXECUTION_FRAME = "H1"
CANDLE_PERIOD = pd.Timedelta(hours=1)

TRADE_COLUMNS = [
    "instrument", "strategy", "direction", "entry_time", "entry_price", "stop_price",
    "target_price", "planned_rr", "exit_time", "exit_price", "exit_reason", "bars_held",
    "outcome", "r_gross", "spread_pips", "cost_r", "r_net",
]


@dataclass
class InstrumentRun:
    instrument: str
    strategy: str
    exit: str
    costs: str
    trades: pd.DataFrame
    funnel: dict
    months_spanned: float
    spread_fallbacks: int = 0     # trades whose entry candle had no spread, so the typical one was used


def load_frames(instrument: str, timeframes: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """The stored candle frames a strategy asked for, plus the H1 execution frame."""
    wanted = dict.fromkeys((*timeframes, EXECUTION_FRAME))      # unique, in order
    return {tf: storage.load(instrument, tf).reset_index(drop=True) for tf in wanted}


def simulate(signals: list[Signal], h1: pd.DataFrame, exit_policy: ExitPolicy, cost_model,
             instrument: str, strategy_name: str) -> tuple[pd.DataFrame, int]:
    """One row per signal. Returns (trades, number of entries that had no spread)."""
    candles = Candles.from_frame(h1)
    spreads = (h1["spread"].to_numpy(dtype=float) if "spread" in h1.columns
               else np.full(len(h1), np.nan))
    known = spreads[np.isfinite(spreads)]
    typical = float(np.median(known)) if len(known) else math.nan
    pip = rules.pip_size(instrument)

    rows: list[dict] = []
    fallbacks = 0
    for signal in signals:
        if not 0 <= signal.entry_index < len(h1):
            raise ValueError(f"{instrument}: signal entry_index {signal.entry_index} is outside "
                             f"the {len(h1)} H1 candles")
        clash = set(signal.meta) & set(TRADE_COLUMNS)
        if clash:
            raise ValueError(f"signal meta uses reserved column name(s) {sorted(clash)}")

        result = exit_policy.resolve(signal, candles)
        spread = float(spreads[signal.entry_index])
        if cost_model.uses_spread and math.isnan(spread):
            fallbacks += 1
        cost_r = cost_model.price(spread, typical, pip) / signal.risk

        if result.index is None:
            outcome, r_gross, r_net = "open", math.nan, math.nan
            exit_time, bars_held = pd.NaT, None
        else:
            sign = 1.0 if signal.direction == BULLISH else -1.0
            r_gross = sign * (result.price - signal.entry_price) / signal.risk
            r_net = r_gross - cost_r
            outcome = "win" if result.reason == "target" else "loss"
            exit_time = candles.times[result.index] + CANDLE_PERIOD   # close of the exit candle
            bars_held = result.index - signal.entry_index

        rows.append({
            "instrument": instrument, "strategy": strategy_name, "direction": signal.direction,
            "entry_time": signal.entry_time, "entry_price": signal.entry_price,
            "stop_price": signal.stop_price, "target_price": signal.target_price,
            "planned_rr": signal.planned_rr, "exit_time": exit_time,
            "exit_price": result.price, "exit_reason": result.reason, "bars_held": bars_held,
            "outcome": outcome, "r_gross": r_gross,
            "spread_pips": spread / pip if not math.isnan(spread) else math.nan,
            "cost_r": cost_r, "r_net": r_net,
            **signal.meta,
        })
    columns = TRADE_COLUMNS + sorted({k for r in rows for k in r} - set(TRADE_COLUMNS))
    return pd.DataFrame(rows, columns=columns), fallbacks


def run_instrument(strategy: Strategy, instrument: str, exit_policy: ExitPolicy, cost_model,
                   frames: dict[str, pd.DataFrame] | None = None) -> InstrumentRun:
    frames = frames if frames is not None else load_frames(instrument, strategy.timeframes)
    output = strategy.generate(instrument, frames)
    trades, fallbacks = simulate(output.signals, frames[EXECUTION_FRAME], exit_policy,
                                 cost_model, instrument, strategy.name)
    return InstrumentRun(instrument=instrument, strategy=strategy.name, exit=exit_policy.name,
                         costs=cost_model.name, trades=trades, funnel=dict(output.funnel),
                         months_spanned=output.months_spanned, spread_fallbacks=fallbacks)
