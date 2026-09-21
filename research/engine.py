"""Runs a strategy on an instrument: load the candles, ask the strategy for signals, end
each trade with an exit policy, charge trading costs, and return one row per trade.

Each signal is resolved on its own: the engine does not stop a new trade opening while an
earlier one on the same instrument is still running, exactly like the original backtest.
Portfolio limits (one trade at a time, a cap on correlated exposure) are a separate layer.
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from backtest import rules
from data_pipeline import quality, storage

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
    data_from: Optional[pd.Timestamp] = None    # first H1 candle used
    data_to: Optional[pd.Timestamp] = None      # last H1 candle used


DAILY_WARMUP = pd.Timedelta(days=14)


@dataclass(frozen=True)
class Window:
    """The part of the stored history a run may use. Sweeps come from H4 and their
    confirmation from H1, so both must begin where the LATER of them is dense; the Daily
    frame only supplies bias, so it may begin a little earlier, as warm-up."""
    start: pd.Timestamp
    daily_start: pd.Timestamp


def window_from_start(start: pd.Timestamp) -> Window:
    return Window(start=start, daily_start=start - DAILY_WARMUP)


def usable_window(instrument: str, timeframes: tuple[str, ...]) -> Optional[Window]:
    """The instrument's dense history across `timeframes`, or None if it has none."""
    starts = quality.usable_windows(instrument, tuple(dict.fromkeys((*timeframes, EXECUTION_FRAME))))
    if starts is None:
        return None
    start = max(starts.values())
    daily = max(starts["D"], start - DAILY_WARMUP) if "D" in starts else start
    return Window(start=start, daily_start=daily)


def window_for(timeframes: tuple[str, ...], instrument: str, start: str) -> Optional[Window]:
    """The window a run should use: "all" (no cut), "auto" (the instrument's dense history) or an
    explicit date. An instrument with no dense history is an error, not a silent empty run."""
    if start == "all":
        return None
    if start == "auto":
        window = usable_window(instrument, timeframes)
        if window is None:
            raise ValueError("no dense history in every timeframe (see python -m "
                             "data_pipeline.quality); use --start all to run on it anyway")
        return window
    return window_from_start(pd.Timestamp(start, tz="UTC"))


def load_frames(instrument: str, timeframes: tuple[str, ...],
                window: Optional[Window] = None) -> dict[str, pd.DataFrame]:
    """The stored candle frames a strategy asked for, plus the H1 execution frame,
    optionally cut to a window of reliable data."""
    wanted = dict.fromkeys((*timeframes, EXECUTION_FRAME))      # unique, in order
    frames = {tf: storage.load(instrument, tf) for tf in wanted}
    if window is not None:
        for tf, df in frames.items():
            begins = window.daily_start if tf == "D" else window.start
            frames[tf] = df[df["time"] >= begins]
    return {tf: df.reset_index(drop=True) for tf, df in frames.items()}


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
                   frames: dict[str, pd.DataFrame] | None = None,
                   window: Optional[Window] = None) -> InstrumentRun:
    frames = frames if frames is not None else load_frames(instrument, strategy.timeframes, window)
    h1 = frames[EXECUTION_FRAME]
    if h1.empty:
        raise ValueError(f"{instrument}: no H1 candles in the requested window")
    output = strategy.generate(instrument, frames)
    trades, fallbacks = simulate(output.signals, h1, exit_policy, cost_model, instrument,
                                 strategy.name)
    return InstrumentRun(instrument=instrument, strategy=strategy.name, exit=exit_policy.name,
                         costs=cost_model.name, trades=trades, funnel=dict(output.funnel),
                         months_spanned=output.months_spanned, spread_fallbacks=fallbacks,
                         data_from=h1["time"].iloc[0], data_to=h1["time"].iloc[-1])
