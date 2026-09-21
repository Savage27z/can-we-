"""A: daily breakout trend-following (pre-registered in research/PREREGISTRATION.md).

Buy when a daily candle closes above the highest high of the previous `channel` daily candles,
sell when it closes below the lowest low; only the first close beyond the channel counts (the
previous day's close was still inside its own channel), so one breakout is one signal, not one per
day of the trend. Enter at the next London open. The stop is `stop_atr` times the 14-day ATR from
entry and the target is `reward_r` times that distance the other way.

The stop is wide (two typical days' movement) so the spread is a small part of the risk, which is
the point of trying it after sweep/FVG's tight stops were eaten by costs.
"""
from collections import Counter

import numpy as np
import pandas as pd

from .. import indicators
from ..strategy import BEARISH, BULLISH, StrategyOutput
from .common import (ENTRY_HOUR, daily_close_times, entry_after, hours_of, make_signal,
                     months_of)


class DonchianTrend:
    name = "donchian_trend"
    timeframes = ("D", "H1")

    def __init__(self, channel: int = 20, atr_length: int = 14, stop_atr: float = 2.0,
                 reward_r: float = 2.0):
        self.channel, self.atr_length = channel, atr_length
        self.stop_atr, self.reward_r = stop_atr, reward_r
        self.daily_warmup_days = channel + atr_length + 30

    def entry_mask(self, h1: pd.DataFrame) -> np.ndarray:
        return hours_of(h1) == ENTRY_HOUR

    def generate(self, instrument: str, frames: dict[str, pd.DataFrame]) -> StrategyOutput:
        daily, h1 = frames["D"], frames["H1"]
        high, low, close = daily["high"], daily["low"], daily["close"]
        upper = indicators.prior_max(high, self.channel)
        lower = indicators.prior_min(low, self.channel)
        volatility = indicators.atr(high, low, close, self.atr_length)

        previous = close.shift(1)
        long_break = (close > upper) & ~(previous > upper.shift(1))
        short_break = (close < lower) & ~(previous < lower.shift(1))
        candidates = np.flatnonzero((long_break | short_break).to_numpy()
                                    & (volatility > 0).to_numpy())

        funnel: Counter = Counter(breakouts=len(candidates))
        entries = entry_after(h1, daily_close_times(daily).iloc[candidates])
        signals = []
        for k, day in enumerate(candidates):
            row = entries[k]
            if row < 0:
                funnel["no_entry_candle"] += 1
                continue
            direction = BULLISH if long_break.iloc[day] else BEARISH
            sign = 1.0 if direction == BULLISH else -1.0
            price = float(h1["close"].iloc[row])
            risk = self.stop_atr * float(volatility.iloc[day])
            signal = make_signal(instrument, direction, h1, row, stop=price - sign * risk,
                                 target=price + sign * self.reward_r * risk,
                                 meta={"atr": float(volatility.iloc[day])})
            if signal is None:
                funnel["invalid_levels"] += 1
                continue
            signals.append(signal)
        funnel["signals"] = len(signals)
        return StrategyOutput(signals=signals, funnel=dict(funnel), months_spanned=months_of(h1))
