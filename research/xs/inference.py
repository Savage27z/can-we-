"""Inference for the cross-sectional strategies, as fixed in research/PREREGISTRATION_XS.md.

The null is a cross-sectional permutation: the same cohorts, the same eligible currencies and the
same holds, but each cohort's long and short legs are drawn at random. That keeps the clustering by
week, the volatility regime and the common dollar factor, and destroys only the ranking information,
which is what is being tested.

The statistic for the ranking's skill is mean GROSS R per leg, so a difference in which currencies
the strategy and the null hold (and so in their spreads) is not mistaken for skill. Net R and the
interval around it, from a moving-block bootstrap over cohort weeks, answer the separate question
of whether it pays.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .legs import Tables
from .select import K, Selection

SPLIT = pd.Timestamp("2016-01-01", tz="UTC")
ONE_SIDED_5_PERCENT_Z = 1.645
BLOCK_WEEKS = 4


@dataclass
class NullDistribution:
    gross: np.ndarray                # mean gross R per leg, one value per replicate
    net: np.ndarray


def permutation_null(tables: Tables, replicates: int = 10_000, seed: int = 0,
                     k: int = K) -> NullDistribution:
    """Each replicate draws k long and k short currencies at random in every cohort and takes the
    mean gross (and net) R per leg over all cohorts."""
    rng = np.random.default_rng(seed)
    gross_sum, net_sum = np.zeros(replicates), np.zeros(replicates)
    count = np.zeros(replicates)
    for c in range(len(tables)):
        eligible = np.flatnonzero(tables.eligible[c])
        if len(eligible) < 2 * k:
            continue
        order = np.argsort(rng.random((replicates, len(eligible)), dtype=np.float32), axis=1)[:, :2 * k]
        chosen = eligible[order]
        gross = np.concatenate([tables.gross_long[c][chosen[:, :k]],
                                tables.gross_short[c][chosen[:, k:]]], axis=1)
        valid = np.isfinite(gross)
        gross_sum += np.where(valid, gross, 0.0).sum(axis=1)
        net_sum += np.where(valid, gross - tables.cost[c][chosen], 0.0).sum(axis=1)
        count += valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return NullDistribution(gross=gross_sum / count, net=net_sum / count)


def p_upper(observed: float, null_means: np.ndarray) -> float:
    """Share of the null at least as good as the observed value, counting the observed itself."""
    null_means = null_means[np.isfinite(null_means)]
    return (1 + int((null_means >= observed).sum())) / (len(null_means) + 1)


def block_bootstrap_interval(sums: np.ndarray, counts: np.ndarray, block: int = BLOCK_WEEKS,
                             replicates: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% interval for a mean per leg, resampling runs of `block` consecutive cohorts with
    replacement. Legs of one cohort share the week, and adjacent cohorts overlap in time when
    the hold is longer than a week, so resampling single legs would be far too optimistic.
    `sums` and `counts` are each cohort's total and number of legs, in time order."""
    n = len(sums)
    if n == 0:
        return float("nan"), float("nan")
    b = min(block, n)
    blocks = int(np.ceil(n / b))
    rng = np.random.default_rng(seed)
    cum_sum = np.concatenate([[0.0], np.cumsum(sums)])
    cum_count = np.concatenate([[0.0], np.cumsum(counts)])
    starts = rng.integers(0, n - b + 1, size=(replicates, blocks))
    drawn_sum = (cum_sum[starts + b] - cum_sum[starts]).sum(axis=1)
    drawn_count = (cum_count[starts + b] - cum_count[starts]).sum(axis=1)
    low, high = np.percentile(drawn_sum / drawn_count, [2.5, 97.5])
    return float(low), float(high)


@dataclass
class StrategyResult:
    name: str
    formation_days: int
    hold_weeks: int
    cohorts: int                     # cohorts with at least one leg
    legs: int
    gross_mean: float
    cost_mean: float
    net_mean: float
    ci_low: float                    # 95% block-bootstrap interval for net_mean
    ci_high: float
    p_gross: float                   # permutation p-value for the gross mean
    null_mean: float
    null_sd: float
    detectable: float                # smallest gross excess the test would have seen (one-sided 5%)
    early_net: float                 # cohorts before 2016
    late_net: float
    early_legs: int
    late_legs: int
    long_gross: float
    short_gross: float
    long_net: float
    short_net: float
    per_currency: pd.DataFrame = field(default_factory=pd.DataFrame)


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else float("nan")


def evaluate(name: str, tables: Tables, selection: Selection, null: NullDistribution,
             seed: int = 0, bootstrap_replicates: int = 10_000) -> StrategyResult:
    gross, net, cost = selection.gross, selection.net, selection.cost
    cohort_times = tables.cohorts[selection.cohort]
    early = np.asarray(cohort_times < SPLIT)
    longs = selection.direction > 0

    sums = np.bincount(selection.cohort, weights=net, minlength=len(tables))
    counts = np.bincount(selection.cohort, minlength=len(tables)).astype(float)
    used = counts > 0
    low, high = block_bootstrap_interval(sums[used], counts[used], BLOCK_WEEKS,
                                         bootstrap_replicates, seed)

    gross_null = null.gross[np.isfinite(null.gross)]
    null_sd = float(gross_null.std(ddof=1)) if len(gross_null) > 1 else float("nan")
    observed = _mean(gross)

    frame = pd.DataFrame({"currency": [tables.currencies[j] for j in selection.currency],
                          "gross": gross, "net": net})
    per_currency = (frame.groupby("currency").agg(legs=("gross", "size"), gross=("gross", "mean"),
                                                  net=("net", "mean")).reset_index())
    return StrategyResult(
        name=name, formation_days=tables.formation_days, hold_weeks=tables.hold_weeks,
        cohorts=int(used.sum()), legs=len(gross), gross_mean=observed, cost_mean=_mean(cost),
        net_mean=_mean(net), ci_low=low, ci_high=high, p_gross=p_upper(observed, null.gross),
        null_mean=float(gross_null.mean()) if len(gross_null) else float("nan"), null_sd=null_sd,
        detectable=ONE_SIDED_5_PERCENT_Z * null_sd, early_net=_mean(net[early]),
        late_net=_mean(net[~early]), early_legs=int(early.sum()), late_legs=int((~early).sum()),
        long_gross=_mean(gross[longs]), short_gross=_mean(gross[~longs]),
        long_net=_mean(net[longs]), short_net=_mean(net[~longs]), per_currency=per_currency)


def control_p_values(tables: Tables, null: NullDistribution, draws: int = 200,
                     seed: int = 0) -> np.ndarray:
    """X0: the p-value, against the permutation null, of each of `draws` random assignments."""
    from .select import select
    values = []
    for i in range(draws):
        picked = select(tables, "random", seed=seed * 1_000_003 + i)
        values.append(p_upper(_mean(picked.gross), null.gross))
    return np.array(values)
