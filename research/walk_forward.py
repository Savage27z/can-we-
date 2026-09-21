"""Walk-forward test of pair selection: does picking the best pairs on the past pay in the future?

The strategy's rules are fixed, so the thing being fitted to the data is the choice of pairs.
Someone who looked at a table of 65 pairs and chose the best few is doing exactly this, and the
question is whether that choice would have worked going forward.

The test replays it honestly. For each block of test years (two years at a time, from 2011), the
pairs are ranked by their net R per trade using ONLY trades that closed before the block began; the
top few are "selected"; and their trades inside the block, which the ranking never saw, are
scored. Blocks are chained forward, each training on everything before it.

The selected pairs' pooled result is compared with two references: all eligible pairs over the same
blocks (is selection better than not selecting?) and randomly chosen pairs of the same number,
many times over (could luck in choosing have done as well? this gives the p-value).
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_TOP_N = 5
DEFAULT_FIRST_TEST_YEAR = 2011
DEFAULT_TEST_YEARS = 2
DEFAULT_MIN_TRAIN_TRADES = 20


@dataclass
class Fold:
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    eligible: int                 # pairs with enough training trades to be ranked
    selected: list[str]
    selected_train_mean: float    # what the ranking saw
    selected_test_mean: Optional[float]
    selected_test_trades: int
    all_test_mean: Optional[float]
    all_test_trades: int


@dataclass
class WalkForwardResult:
    folds: list[Fold] = field(default_factory=list)
    selected_mean: float = float("nan")      # pooled over folds, trade-weighted
    all_mean: float = float("nan")
    excess: float = float("nan")             # selected minus all eligible pairs
    random_pick_mean: float = float("nan")   # average over random selections
    p_value: float = float("nan")            # share of random selections at least as good
    replicates: int = 0
    folds_won: int = 0                       # folds where the selection beat all pairs

    @property
    def selected_trades(self) -> int:
        return sum(f.selected_test_trades for f in self.folds)


def _mean(total: float, n: int) -> Optional[float]:
    return total / n if n else None


def walk_forward_selection(trades: pd.DataFrame, top_n: int = DEFAULT_TOP_N,
                           first_test_year: int = DEFAULT_FIRST_TEST_YEAR,
                           test_years: int = DEFAULT_TEST_YEARS,
                           min_train_trades: int = DEFAULT_MIN_TRAIN_TRADES,
                           replicates: int = 5000, seed: int = 0) -> WalkForwardResult:
    """`trades` needs the columns instrument, entry_time (tz-aware) and r_net; open trades
    (outcome "open", or no r_net) are ignored."""
    resolved = trades[np.isfinite(trades["r_net"].astype(float))]
    result = WalkForwardResult()
    if resolved.empty:
        return result
    times = resolved["entry_time"]
    last_year = int(times.max().year)

    per_fold_arrays = []      # (test_sum, test_n) per eligible instrument, for the random draws
    start_year = first_test_year
    while start_year <= last_year:
        start = pd.Timestamp(f"{start_year}-01-01", tz="UTC")
        end = start + pd.DateOffset(years=test_years)
        start_year += test_years

        train = resolved[times < start]
        test = resolved[(times >= start) & (times < end)]
        train_stats = train.groupby("instrument")["r_net"].agg(["mean", "count"])
        ranked = train_stats[train_stats["count"] >= min_train_trades]
        if len(ranked) <= top_n:                       # nothing to choose between
            continue
        ranked = ranked.sort_index().sort_values("mean", ascending=False, kind="stable")
        names = list(ranked.index)
        chosen = names[:top_n]

        test_stats = test.groupby("instrument")["r_net"].agg(["sum", "count"])
        test_sum = np.array([test_stats["sum"].get(name, 0.0) for name in names])
        test_n = np.array([int(test_stats["count"].get(name, 0)) for name in names])
        per_fold_arrays.append((test_sum, test_n))

        sel_sum, sel_n = float(test_sum[:top_n].sum()), int(test_n[:top_n].sum())
        all_sum, all_n = float(test_sum.sum()), int(test_n.sum())
        result.folds.append(Fold(
            test_start=start, test_end=end, eligible=len(names), selected=chosen,
            selected_train_mean=float(ranked["mean"].iloc[:top_n].mean()),
            selected_test_mean=_mean(sel_sum, sel_n), selected_test_trades=sel_n,
            all_test_mean=_mean(all_sum, all_n), all_test_trades=all_n))

    if not result.folds:
        return result

    sel_total = sum(float(s[:top_n].sum()) for s, _ in per_fold_arrays)
    sel_count = sum(int(n[:top_n].sum()) for _, n in per_fold_arrays)
    all_total = sum(float(s.sum()) for s, _ in per_fold_arrays)
    all_count = sum(int(n.sum()) for _, n in per_fold_arrays)
    if sel_count == 0 or all_count == 0:
        return result
    result.selected_mean = sel_total / sel_count
    result.all_mean = all_total / all_count
    result.excess = result.selected_mean - result.all_mean
    result.folds_won = sum(1 for f in result.folds
                           if f.selected_test_mean is not None and f.all_test_mean is not None
                           and f.selected_test_mean > f.all_test_mean)

    # The same selection made by lot: top_n random pairs in every fold.
    rng = np.random.default_rng(seed)
    random_totals = np.zeros(replicates)
    random_counts = np.zeros(replicates)
    for test_sum, test_n in per_fold_arrays:
        picks = rng.random((replicates, len(test_sum))).argsort(axis=1)[:, :top_n]
        random_totals += test_sum[picks].sum(axis=1)
        random_counts += test_n[picks].sum(axis=1)
    usable = random_counts > 0
    random_means = random_totals[usable] / random_counts[usable]
    result.replicates = int(usable.sum())
    result.random_pick_mean = float(random_means.mean())
    result.p_value = (1 + int((random_means >= result.selected_mean).sum())) / (result.replicates + 1)
    return result
