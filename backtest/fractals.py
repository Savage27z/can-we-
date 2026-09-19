"""§1.1 swing high/low (2-candle fractal), §1.2 liquidity levels, §1.6 mitigation.

A swing formed at index k is only *knowable* starting at index k+2 (needs k+1, k+2
to have closed to evaluate the fractal condition) — `confirmed_at` encodes this so
the engine can avoid look-ahead bias when checking eligibility at some later index i.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def find_swings(df: pd.DataFrame) -> pd.DataFrame:
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    if n > 4:
        idx = np.arange(2, n - 2)
        swing_high[idx] = (
            (high[idx] > high[idx - 1]) & (high[idx] > high[idx - 2])
            & (high[idx] > high[idx + 1]) & (high[idx] > high[idx + 2])
        )
        swing_low[idx] = (
            (low[idx] < low[idx - 1]) & (low[idx] < low[idx - 2])
            & (low[idx] < low[idx + 1]) & (low[idx] < low[idx + 2])
        )
    out = df.copy()
    out["swing_high"] = swing_high
    out["swing_low"] = swing_low
    return out


@dataclass
class LiquidityLevel:
    index: int          # H4 index where the swing formed
    kind: str           # "high" or "low"
    level: float
    confirmed_at: int    # first index at which this swing is knowable (index + 2)
    mitigated_at: Optional[int]  # first index where price traded through it, or None

    def unmitigated_as_of(self, i: int) -> bool:
        """True if this level has not yet been mitigated strictly before index i.
        Mitigation happening AT i itself is fine: a sweep candle's own wick is what
        satisfies §1.6's mitigation condition for the level it sweeps, so the swept
        level's `mitigated_at` naturally equals the sweep index — that's expected,
        not a bug (see engine.py's sweep-detection comment for the full reasoning).
        """
        return self.mitigated_at is None or self.mitigated_at >= i

    def in_scope_of(self, sweep_index: int, lookback: int) -> bool:
        """§2.4: formed within the most recent `lookback` H4 candles *before* the
        sweep candle. Must reject levels formed at/after the sweep candle too —
        without the `self.index < sweep_index` check, a level formed after the
        sweep would give a negative distance that trivially satisfies `<= lookback`.
        """
        return 0 < sweep_index - self.index <= lookback


def build_levels(df: pd.DataFrame) -> list[LiquidityLevel]:
    """Compute every swing high/low with its confirmation index and its first
    mitigation index (§1.6: first later candle whose high/low trades through it).
    """
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)
    levels: list[LiquidityLevel] = []

    swing_high_idx = np.nonzero(df["swing_high"].to_numpy())[0]
    swing_low_idx = np.nonzero(df["swing_low"].to_numpy())[0]

    for k in swing_high_idx:
        L = high[k]
        future = high[k + 1:]
        mitigated_at = None
        if len(future):
            cm = np.maximum.accumulate(future)
            hits = np.nonzero(cm > L)[0]
            if len(hits):
                mitigated_at = int(k + 1 + hits[0])
        levels.append(LiquidityLevel(int(k), "high", float(L), int(k) + 2, mitigated_at))

    for k in swing_low_idx:
        L = low[k]
        future = low[k + 1:]
        mitigated_at = None
        if len(future):
            cm = np.minimum.accumulate(future)
            hits = np.nonzero(cm < L)[0]
            if len(hits):
                mitigated_at = int(k + 1 + hits[0])
        levels.append(LiquidityLevel(int(k), "low", float(L), int(k) + 2, mitigated_at))

    levels.sort(key=lambda lvl: lvl.index)
    return levels
