"""B: London breakout of the Asian range (pre-registered in research/PREREGISTRATION.md).

The range is the high and low of the H1 candles opening 00:00-06:59 UTC. The first candle
opening 07:00-10:59 UTC that closes beyond it is the entry, at that candle's close: a buy above the
range, a sell below it, at most one trade a day. The stop is the opposite side of the range and the
target is `reward_r` times the risk. A day is skipped if fewer than `min_candles` of the seven range
candles exist, or if the risk is under `min_range_atr` or over `max_range_atr` times the ATR
of the last completed daily candles (a range too tight to trade or too wide to mean anything).
"""
from collections import Counter

import numpy as np
import pandas as pd

from .. import indicators
from ..strategy import BEARISH, BULLISH, StrategyOutput
from .common import daily_close_times, hours_of, make_signal, months_of, value_before


class RangeBreakout:
    name = "range_breakout"
    timeframes = ("D", "H1")

    def __init__(self, range_last_hour: int = 6, entry_first_hour: int = 7,
                 entry_last_hour: int = 10, reward_r: float = 2.0, min_range_atr: float = 0.25,
                 max_range_atr: float = 1.5, min_candles: int = 6, atr_length: int = 14):
        self.range_last_hour = range_last_hour
        self.entry_first_hour, self.entry_last_hour = entry_first_hour, entry_last_hour
        self.reward_r = reward_r
        self.min_range_atr, self.max_range_atr = min_range_atr, max_range_atr
        self.min_candles, self.atr_length = min_candles, atr_length
        self.daily_warmup_days = atr_length + 30

    def entry_mask(self, h1: pd.DataFrame) -> np.ndarray:
        hours = hours_of(h1)
        return (hours >= self.entry_first_hour) & (hours <= self.entry_last_hour)

    def generate(self, instrument: str, frames: dict[str, pd.DataFrame]) -> StrategyOutput:
        daily, h1 = frames["D"], frames["H1"]
        volatility = indicators.atr(daily["high"], daily["low"], daily["close"],
                                    self.atr_length).to_numpy()
        daily_closed = daily_close_times(daily)

        hours = hours_of(h1)
        high, low, close = (h1[c].to_numpy(dtype=float) for c in ("high", "low", "close"))
        moments = h1["time"].to_numpy(dtype="datetime64[ns]").astype("int64")
        day = h1["time"].dt.floor("D").to_numpy(dtype="datetime64[ns]").astype("int64")
        edges = np.flatnonzero(np.diff(day)) + 1
        starts, ends = np.concatenate([[0], edges]), np.concatenate([edges, [len(h1)]])

        funnel: Counter = Counter()
        signals = []
        for start, end in zip(starts, ends):
            funnel["days"] += 1
            in_range = hours[start:end] <= self.range_last_hour
            if in_range.sum() < self.min_candles:
                funnel["no_range"] += 1
                continue
            range_high = high[start:end][in_range].max()
            range_low = low[start:end][in_range].min()

            row, direction = -1, None
            for w in np.flatnonzero((hours[start:end] >= self.entry_first_hour)
                                    & (hours[start:end] <= self.entry_last_hour)):
                candle = start + int(w)
                if close[candle] > range_high:
                    row, direction = candle, BULLISH
                    break
                if close[candle] < range_low:
                    row, direction = candle, BEARISH
                    break
            if row < 0:
                funnel["no_breakout"] += 1
                continue

            risk = close[row] - range_low if direction == BULLISH else range_high - close[row]
            typical = value_before(daily["time"], daily_closed, volatility, moments[row:row + 1])[0]
            if not np.isfinite(typical) or not (self.min_range_atr * typical <= risk
                                                <= self.max_range_atr * typical):
                funnel["risk_out_of_bounds"] += 1
                continue
            sign = 1.0 if direction == BULLISH else -1.0
            signal = make_signal(instrument, direction, h1, row,
                                 stop=range_low if direction == BULLISH else range_high,
                                 target=close[row] + sign * self.reward_r * risk,
                                 meta={"range_width": float(range_high - range_low)})
            if signal is None:
                funnel["invalid_levels"] += 1
                continue
            signals.append(signal)
        funnel["signals"] = len(signals)
        return StrategyOutput(signals=signals, funnel=dict(funnel), months_spanned=months_of(h1))
