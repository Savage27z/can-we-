"""Corrections for testing many instruments at once.

Test 65 pairs at the 5% level and about three will "pass" by luck alone. Each function takes
the raw p-values (or z-scores) and returns adjusted ones that can be read at the ordinary
threshold: an adjusted p-value below 0.05 means the result survives having looked at every
pair.

- benjamini_hochberg: controls the false discovery rate, the expected share of the results
  called significant that are false. The usual choice when a few false leads are tolerable,
  and valid when the tests are positively correlated, as pairs sharing a currency are.
- holm: controls the family-wise error rate, the chance of even one false result. Stricter.
- max_t: the "best of N" correction. It asks how often the single best pair in a set of
  pairs with no edge would look at least this good, which is exactly the situation of picking
  the top pair from a table. Its random-entry draws are independent between pairs, while real
  pairs are positively correlated, so it is if anything conservative.
"""
import numpy as np


def _finite(p) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float)
    return p, np.isfinite(p)


def benjamini_hochberg(p_values) -> np.ndarray:
    """BH-adjusted p-values (q-values), in the input order; NaN stays NaN."""
    p, ok = _finite(p_values)
    out = np.full(p.shape, np.nan)
    m = int(ok.sum())
    if m == 0:
        return out
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]        # never adjust below a later rank
    result = np.empty(m)
    result[order] = np.minimum(adjusted, 1.0)
    out[ok] = result
    return out


def holm(p_values) -> np.ndarray:
    """Holm-adjusted p-values, in the input order; NaN stays NaN."""
    p, ok = _finite(p_values)
    out = np.full(p.shape, np.nan)
    m = int(ok.sum())
    if m == 0:
        return out
    order = np.argsort(p[ok])
    stepped = p[ok][order] * (m - np.arange(m))
    adjusted = np.maximum.accumulate(stepped)                   # never adjust below an earlier rank
    result = np.empty(m)
    result[order] = np.minimum(adjusted, 1.0)
    out[ok] = result
    return out


def max_t(observed_z, null_z) -> np.ndarray:
    """Max-statistic adjusted p-values.

    `observed_z` is each instrument's real z-score, shape (m,). `null_z` is each instrument's
    z-score in every random-entry replicate, shape (m, K). For instrument i the adjusted p is the
    share of replicates in which the BEST instrument reached at least z_i, so an instrument only
    stands out if it beats what the luckiest of m no-edge instruments typically manages.
    """
    observed = np.asarray(observed_z, dtype=float)
    null = np.asarray(null_z, dtype=float)
    if null.ndim != 2 or null.shape[0] != observed.shape[0]:
        raise ValueError("null_z must have one row per instrument and one column per replicate")
    with np.errstate(all="ignore"):
        best = np.nanmax(np.where(np.isfinite(null), null, -np.inf), axis=0)
    out = np.full(observed.shape, np.nan)
    for i, z in enumerate(observed):
        if np.isfinite(z):
            out[i] = (1 + int((best >= z).sum())) / (len(best) + 1)
    return out
