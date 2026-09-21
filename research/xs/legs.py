"""The leg tables: for every Monday cohort and every currency, its ranking score and what a long and
a short leg would have earned, following research/PREREGISTRATION_XS.md exactly.

- Entry is the close of the H1 candle opening at 07:00 UTC on the Monday.
- The ranking score is the currency's formation-window log return against the dollar divided by
  its 14-day ATR as a share of price, using only daily candles that had closed before the entry.
- A leg is held until the close of the first H1 candle opening at or after 07:00 UTC `hold_weeks`
  later (found within 3 days), with a disaster stop 5 x ATR from entry that acts on wicks (a gap
  fills at the worse open, and a candle touching the stop counts as stopped).
- Results are in R: signed price change divided by 2 x ATR(14 daily) at entry. Gross R is before
  spread; cost R is one spread taken at the entry candle over the same unit.

A currency with no entry candle, too little daily history, or (for its result) no exit candle is
simply missing for that cohort (NaN); the same holes apply to the strategy and to its null.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from backtest import rules

from .. import indicators
from ..costs import SpreadCost
from . import universe

DAY_NS = 86_400 * 10**9
DAILY_PERIOD_NS = DAY_NS
ENTRY_HOUR = 7
ATR_LENGTH = 14
RISK_ATRS = 2.0                  # one R is this many ATRs
STOP_ATRS = 5.0                  # the disaster stop
EXIT_SEARCH_NS = 3 * DAY_NS
STALE_NS = 5 * DAY_NS            # a daily candle older than this before its target date is a hole
MIN_ELIGIBLE = 8


@dataclass(frozen=True)
class CurrencyData:
    currency: str
    daily: pd.DataFrame
    h1: pd.DataFrame
    first_entry: Optional[pd.Timestamp] = None    # entries before this fall outside the dense history


@dataclass
class Tables:
    """One row per cohort, one column per currency. NaN marks a missing value."""
    cohorts: pd.DatetimeIndex          # entry times (Mondays 07:00 UTC)
    currencies: tuple[str, ...]
    score: np.ndarray                  # ranking score; finite exactly where the currency is eligible
    gross_long: np.ndarray             # R of being long the currency
    gross_short: np.ndarray            # R of being short it
    cost: np.ndarray                   # spread cost in R (the same for either direction)
    hold_weeks: int
    formation_days: int

    @property
    def eligible(self) -> np.ndarray:
        return np.isfinite(self.score)

    def __len__(self) -> int:
        return len(self.cohorts)


def _ns(times) -> np.ndarray:
    return pd.Series(times).to_numpy(dtype="datetime64[ns]").astype("int64")


def monday_entries(first: pd.Timestamp, last: pd.Timestamp) -> pd.DatetimeIndex:
    """Every Monday 07:00 UTC from `first` to `last`."""
    days = pd.date_range(first.floor("D"), last.floor("D"), freq="D", tz="UTC")
    return days[days.dayofweek == 0] + pd.Timedelta(hours=ENTRY_HOUR)


def currency_columns(data: CurrencyData, entry_ns: np.ndarray, formation_days: int,
                     hold_weeks: int) -> tuple[np.ndarray, ...]:
    """(score, gross_long, gross_short, cost) for one currency across the cohort entry times."""
    n_cohorts = len(entry_ns)
    score, gross_long, gross_short, cost = (np.full(n_cohorts, np.nan) for _ in range(4))

    daily, h1 = data.daily, data.h1
    d_close = daily["close"].to_numpy(dtype=float)
    atr = indicators.atr(daily["high"], daily["low"], daily["close"], ATR_LENGTH).to_numpy()
    d_closed_ns = _ns(daily["time"]) + DAILY_PERIOD_NS       # a daily candle has closed 24h after it opens

    h_ns = _ns(h1["time"])
    h_open, h_high, h_low, h_close = (h1[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close"))
    spread = (h1["spread"].to_numpy(dtype=float) if "spread" in h1.columns
              else np.full(len(h1), np.nan))
    known = spread[np.isfinite(spread)]
    typical = float(np.median(known)) if len(known) else float("nan")
    orient = universe.orientation(data.currency)
    pip = rules.pip_size(universe.pair_of(data.currency))
    n_h1, n_d = len(h1), len(daily)
    if n_h1 == 0 or n_d == 0:
        return score, gross_long, gross_short, cost

    entry_row = np.searchsorted(h_ns, entry_ns, side="left")
    entry_at = np.minimum(entry_row, n_h1 - 1)
    has_entry = (entry_row < n_h1) & (h_ns[entry_at] == entry_ns)
    if data.first_entry is not None:
        has_entry &= entry_ns >= data.first_entry.value

    last_end = np.searchsorted(d_closed_ns, entry_ns, side="right") - 1
    start_at = entry_ns - formation_days * DAY_NS
    last_start = np.searchsorted(d_closed_ns, start_at, side="right") - 1
    end_i, start_i = np.clip(last_end, 0, n_d - 1), np.clip(last_start, 0, n_d - 1)
    has_history = ((last_end >= 0) & (last_start >= 0)
                   & (d_closed_ns[end_i] >= entry_ns - STALE_NS)
                   & (d_closed_ns[start_i] >= start_at - STALE_NS))
    atr_end, close_end, close_start = atr[end_i], d_close[end_i], d_close[start_i]
    usable = has_entry & has_history & np.isfinite(atr_end) & (atr_end > 0)

    log_return = orient * np.log(close_end / close_start)
    score[usable] = (log_return / (atr_end / close_end))[usable]

    hold_ns = hold_weeks * 7 * DAY_NS
    unit = RISK_ATRS * atr_end
    stop_distance = STOP_ATRS * atr_end
    for c in np.flatnonzero(usable):
        target = entry_ns[c] + hold_ns
        exit_row = int(np.searchsorted(h_ns, target, side="left"))
        if exit_row >= n_h1 or h_ns[exit_row] - target > EXIT_SEARCH_NS or exit_row <= entry_row[c]:
            continue
        entry = h_close[entry_row[c]]
        opens = h_open[entry_row[c] + 1:exit_row + 1]
        lows, highs = h_low[entry_row[c] + 1:exit_row + 1], h_high[entry_row[c] + 1:exit_row + 1]

        stop = entry - stop_distance[c]
        hit = np.flatnonzero(lows <= stop)
        long_exit = min(opens[hit[0]], stop) if hit.size else h_close[exit_row]
        stop = entry + stop_distance[c]
        hit = np.flatnonzero(highs >= stop)
        short_exit = max(opens[hit[0]], stop) if hit.size else h_close[exit_row]

        pair_long = (long_exit - entry) / unit[c]           # R of being long the PAIR
        pair_short = -(short_exit - entry) / unit[c]        # R of being short the pair
        gross_long[c], gross_short[c] = (pair_long, pair_short) if orient > 0 else (pair_short, pair_long)
        paid = SpreadCost().price_array(np.array([spread[entry_row[c]]]), typical, pip)[0]
        cost[c] = paid / unit[c]
    return score, gross_long, gross_short, cost


def build_tables(data: dict[str, CurrencyData], formation_days: int, hold_weeks: int,
                 entries: Optional[pd.DatetimeIndex] = None) -> Tables:
    """The leg tables for one (formation, hold) configuration. Cohorts with fewer than MIN_ELIGIBLE
    eligible currencies are dropped."""
    currencies = tuple(c for c in universe.CURRENCIES if c in data)
    if entries is None:
        first = min(data[c].h1["time"].iloc[0] for c in currencies)
        last = max(data[c].h1["time"].iloc[-1] for c in currencies)
        entries = monday_entries(first, last)
    entry_ns = _ns(entries)

    columns = [currency_columns(data[c], entry_ns, formation_days, hold_weeks) for c in currencies]
    score, gross_long, gross_short, cost = (np.column_stack([col[i] for col in columns])
                                            for i in range(4))
    keep = np.isfinite(score).sum(axis=1) >= MIN_ELIGIBLE
    return Tables(cohorts=entries[keep], currencies=currencies, score=score[keep],
                  gross_long=gross_long[keep], gross_short=gross_short[keep], cost=cost[keep],
                  hold_weeks=hold_weeks, formation_days=formation_days)
