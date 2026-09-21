"""The contract between a strategy and the backtest engine.

A strategy looks at candle frames and returns signals: where to enter, where the stop
and target are. It does NOT decide how the trade ends or what it costs; the engine
resolves each signal with an exit policy and a cost model, so the same strategy can be
scored under several sets of assumptions without changing it.
"""
from dataclasses import dataclass, field
from typing import Optional, Protocol

import pandas as pd

BULLISH, BEARISH = "bullish", "bearish"


@dataclass(frozen=True)
class Signal:
    instrument: str
    direction: str                  # "bullish" (buy) or "bearish" (sell)
    entry_index: int                # row of the H1 frame whose CLOSE is the entry price
    entry_time: pd.Timestamp        # that candle's close time: when the entry became known
    entry_price: float
    stop_price: float
    target_price: float
    invalidation_price: Optional[float] = None   # a close beyond this ends the trade, if the
                                                 # exit policy is close-based
    meta: dict = field(default_factory=dict)     # strategy-specific fields for the trade log

    def __post_init__(self):
        if self.direction == BULLISH:
            ordered = self.stop_price < self.entry_price < self.target_price
        elif self.direction == BEARISH:
            ordered = self.target_price < self.entry_price < self.stop_price
        else:
            raise ValueError(f"direction must be 'bullish' or 'bearish', got {self.direction!r}")
        if not ordered:
            raise ValueError(
                f"{self.instrument} {self.direction} signal has stop {self.stop_price}, entry "
                f"{self.entry_price}, target {self.target_price}: they must run stop < entry < "
                f"target for a buy and target < entry < stop for a sell")

    @property
    def risk(self) -> float:
        """Distance from entry to stop, in price units: what one R is."""
        return abs(self.entry_price - self.stop_price)

    @property
    def planned_rr(self) -> float:
        return abs(self.target_price - self.entry_price) / self.risk


@dataclass
class StrategyOutput:
    signals: list[Signal]
    funnel: dict = field(default_factory=dict)   # how many candidates fell out at each stage
    months_spanned: float = 0.0                  # length of the data examined, for signals/month


class Strategy(Protocol):
    """A strategy plug-in.

    `timeframes` names the candle frames it needs (keys of `frames`). The engine always
    provides "H1" as the execution frame, and every signal's `entry_index` is a row of
    it. Frames arrive with a fresh 0..n-1 index.

    Two optional attributes:
    - `daily_warmup_days`: how much Daily history before the first signal its indicators need
      (a 200-day average needs 200), so the engine keeps that much before the window starts.
    - `entry_mask(h1) -> bool array`: which H1 candles it could have entered on. The null model
      draws its random entries from these, so a strategy that only ever enters at 07:00 is compared
      with random 07:00 entries, not with entries at hours it never uses.
    """
    name: str
    timeframes: tuple[str, ...]

    def generate(self, instrument: str, frames: dict[str, pd.DataFrame]) -> StrategyOutput:
        ...
