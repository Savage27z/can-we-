"""The decision rules of research/PREREGISTRATION_XS.md, applied by code.

The thresholds are copied from that (frozen) document. Changing one here is a deviation from it and
would have to be reported as one.
"""
import numpy as np

from ..preregistered import Check, format_checks, verdict
from .inference import StrategyResult, p_upper

ALPHA = 0.05
STRATEGIES_TESTED = 2
PRIMARY_P = ALPHA / STRATEGIES_TESTED                 # 0.025
CONTROL_MAX_SHARE = 0.09                              # of the 200 random assignments below 0.05
CONTROL_MEAN_P = (0.44, 0.56)
ORACLE_P = 0.001

__all__ = ["Check", "PRIMARY_P", "evaluate_candidate", "evaluate_controls", "format_checks", "verdict"]


def evaluate_candidate(result: StrategyResult) -> list[Check]:
    """A strategy is a candidate only if all three hold."""
    return [
        Check("1. mean net R positive, 95% block-bootstrap interval excludes zero",
              result.ci_low > 0 and result.net_mean > 0,
              f"{result.net_mean:+.4f}R per leg over {result.legs} legs, "
              f"interval [{result.ci_low:+.4f}, {result.ci_high:+.4f}]"),
        Check(f"2. gross permutation p < {PRIMARY_P}", result.p_gross < PRIMARY_P,
              f"gross {result.gross_mean:+.4f}R against a null of {result.null_mean:+.4f}R "
              f"(sd {result.null_sd:.4f}), p = {result.p_gross:.4f}"),
        Check("3. mean net R positive in both 2005-2015 and 2016-2026",
              result.early_net > 0 and result.late_net > 0,
              f"2005-2015 {result.early_net:+.4f}R (n={result.early_legs}), "
              f"2016-2026 {result.late_net:+.4f}R (n={result.late_legs})"),
    ]


def evaluate_controls(name: str, control_p: np.ndarray, oracle: StrategyResult) -> list[Check]:
    """The checks that must pass for the results on this configuration to be trusted."""
    share = float((control_p < ALPHA).mean())
    mean_p = float(control_p.mean())
    lo, hi = CONTROL_MEAN_P
    return [
        Check(f"X0. at most {CONTROL_MAX_SHARE:.0%} of {len(control_p)} random assignments below "
              f"p = {ALPHA} ({ALPHA:.0%} expected)", share <= CONTROL_MAX_SHARE, f"{share:.1%}"),
        Check(f"X0. mean p of the random assignments between {lo} and {hi}", lo < mean_p < hi,
              f"mean p = {mean_p:.3f}"),
        Check(f"X9. the oracle is detected: gross positive and p < {ORACLE_P}",
              oracle.gross_mean > 0 and oracle.p_gross < ORACLE_P,
              f"gross {oracle.gross_mean:+.4f}R, p = {oracle.p_gross:.5f}"),
    ]
