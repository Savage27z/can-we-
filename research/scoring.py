"""Scores a set of trades: how often they win, what they earn per trade in R, how deep the
losing streaks run, and how much of that is distinguishable from luck.

Results are in R multiples (1R = the risk on that trade), so trades on different pairs
and stop distances can be pooled. Only resolved trades (a win or a loss) count towards
the statistics; a trade still open at the end of the data has no result yet.

The confidence interval is a bootstrap of the mean net R per trade: resample the trades
with replacement many times and read off the 2.5th and 97.5th percentiles of the means.
An expectancy whose interval spans zero has not been distinguished from no edge, however
good the point estimate looks; with a few dozen trades it almost always spans zero.
"""
from typing import Optional

import numpy as np
import pandas as pd

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_CHUNK = 250          # resamples drawn at a time, to bound memory on big pools


def bootstrap_mean_interval(values: np.ndarray, samples: int = BOOTSTRAP_SAMPLES,
                            seed: int = 0) -> tuple[float, float, float]:
    """(2.5th percentile, 97.5th percentile, share of resampled means at or below zero)."""
    n = len(values)
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for start in range(0, samples, BOOTSTRAP_CHUNK):
        size = min(BOOTSTRAP_CHUNK, samples - start)
        picks = rng.integers(0, n, size=(size, n))
        means[start:start + size] = values[picks].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high), float((means <= 0).mean())


def max_drawdown(results_in_order: np.ndarray) -> float:
    """Largest peak-to-trough fall of the running total, in R."""
    if len(results_in_order) == 0:
        return 0.0
    running = np.cumsum(results_in_order)
    peaks = np.maximum.accumulate(np.maximum(running, 0.0))
    return float((peaks - running).max())


def summarize(trades: pd.DataFrame, months_spanned: Optional[float] = None,
              bootstrap_samples: int = BOOTSTRAP_SAMPLES) -> dict:
    """Statistics for a trades frame as produced by the engine."""
    resolved = trades[trades["outcome"] != "open"]
    wins = int((resolved["outcome"] == "win").sum())
    losses = int((resolved["outcome"] == "loss").sum())
    n = len(resolved)

    summary: dict = {
        "signals": len(trades), "resolved": n, "open": len(trades) - n,
        "wins": wins, "losses": losses,
        "win_rate": wins / n if n else None,
        "avg_planned_rr": float(trades["planned_rr"].mean()) if len(trades) else None,
        "signals_per_month": (len(trades) / months_spanned
                              if months_spanned and months_spanned > 0 else None),
    }
    if n == 0:
        return {**summary, "expectancy_gross": None, "expectancy_net": None, "avg_cost_r": None,
                "sd_net": None, "ci_low": None, "ci_high": None, "p_le_zero": None,
                "profit_factor_net": None, "max_dd_gross": None, "max_dd_net": None}

    ordered = resolved.sort_values("entry_time", kind="stable")
    gross = ordered["r_gross"].to_numpy(dtype=float)
    net = ordered["r_net"].to_numpy(dtype=float)
    summary.update(
        expectancy_gross=float(gross.mean()),
        expectancy_net=float(net.mean()),
        avg_cost_r=float(ordered["cost_r"].mean()),
        sd_net=float(net.std(ddof=1)) if n > 1 else None,
        max_dd_gross=max_drawdown(gross),
        max_dd_net=max_drawdown(net),
    )
    losing = -net[net < 0].sum()
    summary["profit_factor_net"] = float(net[net > 0].sum() / losing) if losing > 0 else None
    if n > 1:
        low, high, p_le_zero = bootstrap_mean_interval(net, bootstrap_samples)
        summary.update(ci_low=low, ci_high=high, p_le_zero=p_le_zero)
    else:
        summary.update(ci_low=None, ci_high=None, p_le_zero=None)
    return summary
