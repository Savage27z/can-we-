"""How a trade ends. A strategy proposes an entry, stop and target; the exit policy
walks the candles forward from the entry and says when and at what price it finished.

Three policies, because they answer different questions:

- close: the strategy's own scoring (§6.5). A win is an H1 CLOSE at or beyond the
  target; a loss is an H1 CLOSE beyond the invalidation level. Scored at exactly
  the target (+planned R) or the stop (-1R). This is what the original backtest did, and
  what the "37 signals, +0.34R" result is measured with.
- touch: hard stop-loss and take-profit orders. Either level is hit the moment price
  touches it, wicks included. If one candle touches both, the stop is assumed to have
  come first (the pessimistic reading, since the candle's internal order is unknown).
- plan: what the live trade plan tells a person to do: the hard stop and take-profit
  orders of "touch", plus closing by hand at the close of any H1 candle that closes
  beyond the invalidation level, which can come before the stop is touched.

Fill prices: a stop that price gaps through fills at the candle's open, the worse price.
A target fills at the target price, never better, even if price gapped past it. Both
choices lean pessimistic on purpose.
"""
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import pandas as pd

from backtest import invalidation
from backtest.resolution import resolve_on_closes

from .strategy import BULLISH, Signal


@dataclass(frozen=True)
class Candles:
    """The execution frame as arrays, so an exit policy loops over numbers, not rows."""
    times: pd.Series
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "Candles":
        return cls(times=frame["time"].reset_index(drop=True),
                   open=frame["open"].to_numpy(dtype=float),
                   high=frame["high"].to_numpy(dtype=float),
                   low=frame["low"].to_numpy(dtype=float),
                   close=frame["close"].to_numpy(dtype=float))

    def __len__(self) -> int:
        return len(self.close)


@dataclass(frozen=True)
class Exit:
    index: Optional[int]        # row of the candle the trade ended in; None while still open
    price: Optional[float]      # price it ended at, for R accounting
    reason: str                 # "target", "stop", "invalidation", or "open"


OPEN = Exit(index=None, price=None, reason="open")


class ExitPolicy(Protocol):
    name: str

    def resolve(self, signal: Signal, candles: Candles) -> Exit:
        ...


def _invalidation_level(signal: Signal) -> float:
    return signal.invalidation_price if signal.invalidation_price is not None else signal.stop_price


def _touches(signal: Signal, candles: Candles, j: int) -> Optional[Exit]:
    """Stop or target touched during candle j, stop first. None if neither."""
    stop, target = signal.stop_price, signal.target_price
    if signal.direction == BULLISH:
        if candles.low[j] <= stop:
            return Exit(j, float(min(candles.open[j], stop)), "stop")
        if candles.high[j] >= target:
            return Exit(j, target, "target")
    else:
        if candles.high[j] >= stop:
            return Exit(j, float(max(candles.open[j], stop)), "stop")
        if candles.low[j] <= target:
            return Exit(j, target, "target")
    return None


@dataclass(frozen=True)
class CloseExit:
    name: str = "close"

    def resolve(self, signal: Signal, candles: Candles) -> Exit:
        found = resolve_on_closes(signal.direction, candles.close, signal.entry_index + 1,
                                  _invalidation_level(signal), signal.target_price)
        if found is None:
            return OPEN
        index, reason = found
        # Scored at the stop or the target, not at the close that triggered it.
        price = signal.stop_price if reason == "invalidation" else signal.target_price
        return Exit(index, price, reason)


@dataclass(frozen=True)
class TouchExit:
    name: str = "touch"

    def resolve(self, signal: Signal, candles: Candles) -> Exit:
        for j in range(signal.entry_index + 1, len(candles)):
            hit = _touches(signal, candles, j)
            if hit is not None:
                return hit
        return OPEN


@dataclass(frozen=True)
class PlanExit:
    name: str = "plan"

    def resolve(self, signal: Signal, candles: Candles) -> Exit:
        level = _invalidation_level(signal)
        for j in range(signal.entry_index + 1, len(candles)):
            hit = _touches(signal, candles, j)
            if hit is not None:
                return hit
            if invalidation.is_invalidated(signal.direction, level, float(candles.close[j])):
                return Exit(j, float(candles.close[j]), "invalidation")
        return OPEN


EXITS = {"close": CloseExit, "touch": TouchExit, "plan": PlanExit}


def get_exit(name: str) -> ExitPolicy:
    try:
        return EXITS[name]()
    except KeyError:
        raise ValueError(f"unknown exit policy {name!r}; choose from {sorted(EXITS)}") from None
