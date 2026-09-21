"""D: the control (pre-registered in research/PREREGISTRATION.md).

On about one weekday in `every_days`, a random hour between `first_hour` and `last_hour` UTC
(the candle opening then), a random direction, a stop `stop_atr` times the ATR of the last completed
daily candles away and a target `reward_r` times that distance. It has no skill by construction.
If the validation pipeline reports anything but noise for it on real data, the pipeline is broken.

The draws are reproducible from `seed` and the instrument, and do not depend on the data beyond
which days and candles exist.
"""
import zlib
from collections import Counter

import numpy as np
import pandas as pd

from .. import indicators
from ..strategy import BEARISH, BULLISH, StrategyOutput
from .common import daily_close_times, hours_of, make_signal, months_of, value_before


class RandomControl:
    name = "random_control"
    timeframes = ("D", "H1")

    def __init__(self, every_days: int = 10, stop_atr: float = 2.0, reward_r: float = 2.0,
                 first_hour: int = 7, last_hour: int = 20, atr_length: int = 14, seed: int = 0):
        self.every_days, self.stop_atr, self.reward_r = every_days, stop_atr, reward_r
        self.first_hour, self.last_hour = first_hour, last_hour
        self.atr_length, self.seed = atr_length, seed
        self.daily_warmup_days = atr_length + 30

    def entry_mask(self, h1: pd.DataFrame) -> np.ndarray:
        hours = hours_of(h1)
        return (hours >= self.first_hour) & (hours <= self.last_hour)

    def generate(self, instrument: str, frames: dict[str, pd.DataFrame]) -> StrategyOutput:
        daily, h1 = frames["D"], frames["H1"]
        volatility = indicators.atr(daily["high"], daily["low"], daily["close"],
                                    self.atr_length).to_numpy()
        daily_closed = daily_close_times(daily)
        rng = np.random.default_rng([self.seed, zlib.crc32(instrument.encode())])

        allowed = self.entry_mask(h1)
        hours = hours_of(h1)
        moments = h1["time"].to_numpy(dtype="datetime64[ns]").astype("int64")
        day = h1["time"].dt.floor("D").to_numpy(dtype="datetime64[ns]").astype("int64")
        edges = np.flatnonzero(np.diff(day)) + 1
        starts, ends = np.concatenate([[0], edges]), np.concatenate([edges, [len(h1)]])

        funnel: Counter = Counter()
        signals = []
        for start, end in zip(starts, ends):
            # Every draw is made for every day, and the candle is chosen by its HOUR, so the choice
            # never depends on how many candles the day happens to have. A day only partly present
            # (the data ends mid-day) then picks the same candle it would with the whole day.
            pick_day = rng.random() < 1.0 / self.every_days
            side = rng.random() < 0.5
            hour = self.first_hour + int(rng.random() * (self.last_hour - self.first_hour + 1))
            if not pick_day:
                continue
            candles = start + np.flatnonzero(allowed[start:end] & (hours[start:end] == hour))
            if len(candles) == 0:
                funnel["hour_missing"] += 1
                continue
            funnel["picked"] += 1
            row = int(candles[0])
            typical = value_before(daily["time"], daily_closed, volatility, moments[row:row + 1])[0]
            if not np.isfinite(typical) or typical <= 0:
                funnel["no_volatility"] += 1
                continue
            direction = BULLISH if side else BEARISH
            sign = 1.0 if side else -1.0
            price = float(h1["close"].iloc[row])
            risk = self.stop_atr * typical
            signal = make_signal(instrument, direction, h1, row, stop=price - sign * risk,
                                 target=price + sign * self.reward_r * risk,
                                 meta={"atr": float(typical)})
            if signal is not None:
                signals.append(signal)
        funnel["signals"] = len(signals)
        return StrategyOutput(signals=signals, funnel=dict(funnel), months_spanned=months_of(h1))
