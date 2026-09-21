"""Synthetic candle data for testing strategies.

synthetic_market builds H1 candles for whole FX trading weeks (Sunday 17:00 to Friday 17:00 New
York time, so the DST changes fall while the market is shut, as in real data) and aggregates
them into H4 and Daily candles on OANDA's 17:00 New York grid. The frames are consistent with
one another the way the real ones are, which strategies that combine timeframes rely on.
"""
import numpy as np
import pandas as pd

NY = "America/New_York"
WEEK_CANDLES = 120                  # H1 candles from Sunday 17:00 to Friday 17:00


def synthetic_market(weeks: int = 120, seed: int = 0, sigma: float = 0.0008,
                     spread: float = 0.0001, first_sunday: str = "2022-01-02",
                     drift: float = 0.0, base: float = 1.10) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    pieces = []
    for w in range(weeks):
        start = pd.Timestamp(first_sunday + " 17:00", tz=NY) + pd.DateOffset(weeks=w)
        pieces.append(pd.date_range(start, periods=WEEK_CANDLES, freq="1h"))
    times = pieces[0].append(pieces[1:]).tz_convert("UTC")
    n = len(times)

    gap = np.zeros(n)
    gap[::WEEK_CANDLES] = rng.normal(0, sigma * 3, weeks)          # the weekend gap
    step = rng.normal(drift, sigma, n)
    close = base + np.cumsum(gap + step)
    open_ = np.concatenate([[base], close[:-1]]) + gap
    high = np.maximum(open_, close) + np.abs(rng.normal(0, sigma / 2, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, sigma / 2, n))
    h1 = pd.DataFrame({"time": times, "open": open_, "high": high, "low": low, "close": close,
                       "volume": 100, "spread": spread})
    return {"D": _aggregate(h1, 24), "H4": _aggregate(h1, 4), "H1": h1}


def _aggregate(h1: pd.DataFrame, hours: int) -> pd.DataFrame:
    """H1 candles into candles of `hours` starting on the 17:00 New York grid."""
    local = h1["time"].dt.tz_convert(NY)
    since = (local.dt.hour - 17) % hours
    bucket = h1["time"] - pd.to_timedelta(since, unit="h")
    grouped = h1.groupby(bucket)
    out = pd.DataFrame({
        "open": grouped["open"].first(), "high": grouped["high"].max(),
        "low": grouped["low"].min(), "close": grouped["close"].last(),
        "volume": grouped["volume"].sum(), "spread": grouped["spread"].last(),
        "count": grouped["open"].size()})
    out = out[out["count"] == hours].drop(columns="count")
    out.index.name = "time"
    return out.reset_index()


def truncate(frames: dict[str, pd.DataFrame], cutoff: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Only the candles that had CLOSED by `cutoff`: what was knowable at that moment."""
    period = {"H1": pd.Timedelta(hours=1), "H4": pd.Timedelta(hours=4), "D": pd.Timedelta(hours=24)}
    return {tf: df[df["time"] + period[tf] <= cutoff].reset_index(drop=True)
            for tf, df in frames.items()}


def daily_frame(rows, first_open: str = "2024-03-01 21:00") -> pd.DataFrame:
    """Daily candles from (open, high, low, close) tuples, one per calendar day."""
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df.insert(0, "time", pd.date_range(pd.Timestamp(first_open, tz="UTC"), periods=len(rows),
                                       freq="1D"))
    df["volume"] = 1
    df["spread"] = 0.0001
    return df


def h1_frame(start: str, periods: int, price: float = 1.10, **columns) -> pd.DataFrame:
    """Hourly candles from `start`, flat at `price` unless a column is given as a sequence."""
    df = pd.DataFrame({"time": pd.date_range(pd.Timestamp(start, tz="UTC"), periods=periods,
                                             freq="1h")})
    for name, default in (("open", price), ("high", price + 0.0002), ("low", price - 0.0002),
                          ("close", price)):
        df[name] = columns.get(name, default)
    df["volume"] = 1
    df["spread"] = columns.get("spread", 0.0001)
    return df
