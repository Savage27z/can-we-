"""Indicators over candle series, for strategies to use.

Every function returns a series aligned to its input whose value at row i uses only rows up to i
(or, for the "prior" ones, only rows before i), and is NaN until it has enough history. That
alignment is what keeps a strategy from peeking: a signal computed at the close of daily candle i
may use indicator[i] and nothing later. tests/test_strategies_lookahead.py checks the strategies
built on them by cutting the data off and confirming no signal changes.
"""
import pandas as pd


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's true range: the largest of the candle's own range and its gap from the previous
    close. The first candle has no previous close, so its range is just high minus low."""
    previous_close = close.shift(1)
    return pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()],
                     axis=1).max(axis=1, skipna=True)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average true range: the simple mean of the last `length` true ranges, NaN before that. A
    typical daily move in price units, and so a volatility-scaled way to size a stop."""
    return true_range(high, low, close).rolling(length).mean()


def sma(values: pd.Series, length: int) -> pd.Series:
    return values.rolling(length).mean()


def rolling_std(values: pd.Series, length: int) -> pd.Series:
    """Population standard deviation over the last `length` values, as Bollinger Bands use."""
    return values.rolling(length).std(ddof=0)


def prior_max(values: pd.Series, length: int) -> pd.Series:
    """The highest value over the `length` rows BEFORE each row (not including it): the level a
    breakout has to clear."""
    return values.shift(1).rolling(length).max()


def prior_min(values: pd.Series, length: int) -> pd.Series:
    return values.shift(1).rolling(length).min()
