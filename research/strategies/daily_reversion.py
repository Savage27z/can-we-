"""C: daily mean reversion, a pullback in the trend (pre-registered in
research/PREREGISTRATION.md).

Buy when a daily candle closes below the lower Bollinger Band (`band_length` days, `band_k`
standard deviations) while still above the `trend_length`-day average, i.e. a stretched dip inside an
uptrend; sell the mirror image. Only the first close beyond the band counts. Enter at the next
London open. The stop is `stop_atr` times the 14-day ATR from entry; the target is the band's
centre line (the 20-day average) as it stood at the signal, so the trade aims to capture the
snap back to the mean. A signal whose target has already been reached by the entry is skipped.
"""
from collections import Counter

import numpy as np
import pandas as pd

from .. import indicators
from ..strategy import BEARISH, BULLISH, StrategyOutput
from .common import (ENTRY_HOUR, daily_close_times, entry_after, hours_of, make_signal,
                     months_of)


class DailyReversion:
    name = "daily_reversion"
    timeframes = ("D", "H1")

    def __init__(self, band_length: int = 20, band_k: float = 2.0, trend_length: int = 200,
                 atr_length: int = 14, stop_atr: float = 2.0):
        self.band_length, self.band_k = band_length, band_k
        self.trend_length, self.atr_length, self.stop_atr = trend_length, atr_length, stop_atr
        self.daily_warmup_days = int(trend_length * 1.6) + 30       # 200 trading days is ~290 calendar

    def entry_mask(self, h1: pd.DataFrame) -> np.ndarray:
        return hours_of(h1) == ENTRY_HOUR

    def generate(self, instrument: str, frames: dict[str, pd.DataFrame]) -> StrategyOutput:
        daily, h1 = frames["D"], frames["H1"]
        high, low, close = daily["high"], daily["low"], daily["close"]
        centre = indicators.sma(close, self.band_length)
        spread = self.band_k * indicators.rolling_std(close, self.band_length)
        lower, upper = centre - spread, centre + spread
        trend = indicators.sma(close, self.trend_length)
        volatility = indicators.atr(high, low, close, self.atr_length)

        previous = close.shift(1)
        long_setup = (close < lower) & (close > trend) & ~(previous < lower.shift(1))
        short_setup = (close > upper) & (close < trend) & ~(previous > upper.shift(1))
        candidates = np.flatnonzero((long_setup | short_setup).to_numpy()
                                    & (volatility > 0).to_numpy())

        funnel: Counter = Counter(setups=len(candidates))
        entries = entry_after(h1, daily_close_times(daily).iloc[candidates])
        signals = []
        for k, day in enumerate(candidates):
            row = entries[k]
            if row < 0:
                funnel["no_entry_candle"] += 1
                continue
            direction = BULLISH if long_setup.iloc[day] else BEARISH
            sign = 1.0 if direction == BULLISH else -1.0
            price = float(h1["close"].iloc[row])
            signal = make_signal(instrument, direction, h1, row,
                                 stop=price - sign * self.stop_atr * float(volatility.iloc[day]),
                                 target=float(centre.iloc[day]),
                                 meta={"atr": float(volatility.iloc[day])})
            if signal is None:
                funnel["target_already_reached"] += 1
                continue
            signals.append(signal)
        funnel["signals"] = len(signals)
        return StrategyOutput(signals=signals, funnel=dict(funnel), months_spanned=months_of(h1))
