"""Which legs each rule picks. Every cohort takes K currencies long and K short.

- momentum:  long the K highest-scoring currencies, short the K lowest.
- reversal:  the opposite, long the K lowest and short the K highest.
- oracle:    ranks by the FORWARD result of holding the currency (information no one has); the
             power check, which must be detected.
- random:    the same number of longs and shorts assigned at random among the eligible currencies;
             the control, which must look like noise.

Ties are broken by currency order, so a rule is deterministic given the tables.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .legs import Tables

K = 3
RULES = ("momentum", "reversal", "oracle", "random")


@dataclass
class Selection:
    """The chosen legs, one row each. Legs whose exit could not be found are left out."""
    cohort: np.ndarray               # row of tables.cohorts
    currency: np.ndarray             # column of tables.currencies
    direction: np.ndarray            # +1 long the currency, -1 short
    gross: np.ndarray
    cost: np.ndarray
    n_cohorts: int                   # cohorts in the tables the selection was made from

    @property
    def net(self) -> np.ndarray:
        return self.gross - self.cost

    def __len__(self) -> int:
        return len(self.gross)


def _extremes(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per cohort, the columns of the k highest and k lowest finite values (highest first / lowest
    first); -1 throughout for a cohort with fewer than 2k finite values."""
    n = len(values)
    top, bottom = np.full((n, k), -1), np.full((n, k), -1)
    for c in range(n):
        ok = np.flatnonzero(np.isfinite(values[c]))
        if len(ok) < 2 * k:
            continue
        order = ok[np.argsort(values[c, ok], kind="stable")]
        bottom[c], top[c] = order[:k], order[::-1][:k]
    return top, bottom


def _random_assignment(eligible: np.ndarray, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = len(eligible)
    long_idx, short_idx = np.full((n, k), -1), np.full((n, k), -1)
    for c in range(n):
        ok = np.flatnonzero(eligible[c])
        if len(ok) < 2 * k:
            continue
        picked = rng.permutation(ok)[:2 * k]
        long_idx[c], short_idx[c] = picked[:k], picked[k:]
    return long_idx, short_idx


def _legs(tables: Tables, long_idx: np.ndarray, short_idx: np.ndarray) -> Selection:
    parts = []
    for direction, chosen, gross_table in ((1, long_idx, tables.gross_long),
                                           (-1, short_idx, tables.gross_short)):
        cohort, slot = np.nonzero(chosen >= 0)
        currency = chosen[cohort, slot]
        gross = gross_table[cohort, currency]
        keep = np.isfinite(gross)
        parts.append((cohort[keep], currency[keep], np.full(int(keep.sum()), direction),
                      gross[keep], tables.cost[cohort[keep], currency[keep]]))
    cohort, currency, direction, gross, cost = (np.concatenate([p[i] for p in parts]) for i in range(5))
    order = np.argsort(cohort, kind="stable")
    return Selection(cohort[order], currency[order], direction[order], gross[order], cost[order],
                     n_cohorts=len(tables))


def select(tables: Tables, rule: str, k: int = K, seed: Optional[int] = None) -> Selection:
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; choose from {RULES}")
    if rule == "random":
        long_idx, short_idx = _random_assignment(tables.eligible, k, np.random.default_rng(seed))
    elif rule == "oracle":
        forward = np.where(tables.eligible, tables.gross_long, np.nan)
        long_idx, short_idx = _extremes(forward, k)
    else:
        top, bottom = _extremes(np.where(tables.eligible, tables.score, np.nan), k)
        long_idx, short_idx = (top, bottom) if rule == "momentum" else (bottom, top)
    return _legs(tables, long_idx, short_idx)
