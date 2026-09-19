"""§1.4 Fair Value Gap detection, §1.8 confirmation level.

No minimum gap size (§9 flags this as a candidate parameter to add later based on
backtest results — deliberately not implemented here yet).
"""
from dataclasses import dataclass


@dataclass
class FVG:
    mid_index: int   # the "i" in §1.4's (i-1, i, i+1)
    direction: str   # "bullish" or "bearish"
    low: float
    high: float

    @property
    def confirmation_level(self) -> float:
        """§1.8: the near/shallow edge of the gap."""
        return self.low if self.direction == "bullish" else self.high


def detect_fvg(high, low, mid_index: int) -> "FVG | None":
    """Check whether indices (mid_index-1, mid_index, mid_index+1) form an FVG.
    `high`/`low` are numpy arrays for the full H4 series (§1.10: no restriction on
    these candles overlapping a sweep candle elsewhere in the caller's logic).
    """
    i = mid_index
    if i - 1 < 0 or i + 1 >= len(high):
        return None
    if low[i + 1] > high[i - 1]:
        return FVG(i, "bullish", high[i - 1], low[i + 1])
    if high[i + 1] < low[i - 1]:
        return FVG(i, "bearish", high[i + 1], low[i - 1])
    return None
