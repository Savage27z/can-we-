"""Applies the decision rules in research/PREREGISTRATION.md, mechanically, to saved validation runs.

The rules were fixed before the strategies were run, so they are applied by this code rather than
by reading the numbers and deciding afterwards. The thresholds below are copied from that
document; changing one here is a deviation from it and must be reported as one.

    python -m research.preregistered s4_donchian_random s4_breakout_random s4_reversion_random
    python -m research.preregistered --control s4_control_random
"""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data_pipeline import config

from . import scoring

ALPHA = 0.05
STRATEGIES_TESTED = 3
PRIMARY_P = ALPHA / STRATEGIES_TESTED                    # 0.0167, Bonferroni over the three strategies
SPLIT = pd.Timestamp("2016-01-01", tz="UTC")             # 2005-2015 against 2016-2026
CONTROL_POOLED_P = (0.02, 0.98)
CONTROL_MAX_RAW_SHARE = 0.12                             # 5% expected
CONTROL_MEAN_P = (0.40, 0.60)
PRIMARY_SETTINGS = {"exit": "touch", "costs": "spread", "null_direction": "random"}


@dataclass(frozen=True)
class Check:
    rule: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class Run:
    name: str
    trades: pd.DataFrame            # resolved trades
    pairs: pd.DataFrame             # validation.csv
    settings: dict
    results: dict


def load_run(run_dir: Path) -> Run:
    run_dir = Path(run_dir)
    saved = json.loads((run_dir / "validation.json").read_text(encoding="utf-8"))
    trades = pd.read_csv(run_dir / "trades.csv")
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades = trades[trades["outcome"] != "open"].reset_index(drop=True)
    return Run(name=run_dir.name, trades=trades, pairs=pd.read_csv(run_dir / "validation.csv"),
               settings=saved["settings"], results=saved["results"])


def configuration_check(run: Run, expected: dict) -> Check:
    """Whether the run used the configuration the document fixed. A mismatch is a deviation."""
    wrong = {k: (run.settings.get(k), v) for k, v in expected.items() if run.settings.get(k) != v}
    if wrong:
        return Check("run uses the pre-registered configuration", False,
                     "DEVIATION: " + ", ".join(f"{k} was {got!r}, registered {want!r}"
                                               for k, (got, want) in wrong.items()))
    return Check("run uses the pre-registered configuration", True,
                 ", ".join(f"{k}={v}" for k, v in expected.items()))


def evaluate_candidate(run: Run) -> list[Check]:
    """The three rules a strategy must pass, all of them, to be called a candidate."""
    summary = scoring.summarize(run.trades)
    pooled = run.results["pooled"]
    early = run.trades[run.trades["entry_time"] < SPLIT]["r_net"]
    late = run.trades[run.trades["entry_time"] >= SPLIT]["r_net"]
    ci_low, ci_high = summary["ci_low"], summary["ci_high"]
    return [
        configuration_check(run, PRIMARY_SETTINGS),
        Check("1. pooled net expectancy positive, 95% interval excludes zero",
              ci_low is not None and summary["expectancy_net"] > 0 and ci_low > 0,
              f"{summary['expectancy_net']:+.4f}R over {summary['resolved']} trades, "
              f"interval [{ci_low:+.4f}, {ci_high:+.4f}]"),
        Check(f"2. excess over the random-direction null positive with p < {PRIMARY_P:.4f}",
              pooled["excess"] > 0 and pooled["p"] < PRIMARY_P,
              f"excess {pooled['excess']:+.4f}R, p = {pooled['p']:.4f}"),
        Check("3. net expectancy positive in both 2005-2015 and 2016-2026",
              len(early) > 0 and len(late) > 0 and early.mean() > 0 and late.mean() > 0,
              f"2005-2015 {early.mean():+.4f}R (n={len(early)}), "
              f"2016-2026 {late.mean():+.4f}R (n={len(late)})"),
    ]


def evaluate_control(run: Run) -> list[Check]:
    """The pipeline's own calibration on real data: a strategy with no skill must look like noise."""
    pooled_p = run.results["pooled"]["p"]
    p = run.pairs["p"].dropna()
    raw_share = float((p < ALPHA).mean())
    mean_p = float(p.mean())
    lo, hi = CONTROL_POOLED_P
    mlo, mhi = CONTROL_MEAN_P
    return [
        configuration_check(run, PRIMARY_SETTINGS),
        Check(f"C1. pooled p-value between {lo} and {hi}", lo < pooled_p < hi,
              f"pooled p = {pooled_p:.4f}"),
        Check(f"C2. at most {CONTROL_MAX_RAW_SHARE:.0%} of pairs with raw p < {ALPHA} "
              f"({ALPHA:.0%} expected)", raw_share <= CONTROL_MAX_RAW_SHARE,
              f"{raw_share:.1%} of {len(p)} pairs"),
        Check(f"C3. mean pair p-value between {mlo} and {mhi}", mlo < mean_p < mhi,
              f"mean p = {mean_p:.3f}"),
    ]


def verdict(checks: list[Check]) -> str:
    return "PASS" if all(c.passed for c in checks) else "FAIL"


def format_checks(name: str, checks: list[Check], what: str) -> str:
    lines = [f"{name}: {what}: {verdict(checks)}"]
    for c in checks:
        lines.append(f"  [{'pass' if c.passed else 'FAIL'}] {c.rule}: {c.detail}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Run directory names under data/runs/.")
    parser.add_argument("--control", action="store_true", help="Apply the control's checks instead.")
    parser.add_argument("--runs-dir", default=None)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    base = Path(args.runs_dir) if args.runs_dir else config.DATA_DIR / "runs"
    exit_code = 0
    for name in args.runs:
        run = load_run(base / name)
        if args.control:
            checks, what = evaluate_control(run), "the pipeline is calibrated on real data"
        else:
            checks, what = evaluate_candidate(run), "candidate under the pre-registered rules"
        print(format_checks(name, checks, what) + "\n")
        exit_code |= int(verdict(checks) != "PASS")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
