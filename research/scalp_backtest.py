"""First-pass check: the exact sweep+FVG logic of strategy_rules.md, run on a scalp-scale
timeframe pair (M5 structure/sweep detection, M1 confirmation) instead of H4/H1.

This is deliberately lighter-weight than research/run_backtest.py's validated pipeline: one
pair, one window, no walk-forward split, no multiple-testing correction across configurations.
It answers one question honestly — does this rule set, unchanged, show any edge at scalp
scale — using the same trade resolution, cost model and random-entry null the rest of this
project holds every strategy to (backtest.setup.evaluate_setup, research.engine.simulate,
research.null_model), not a new, unvalidated shortcut.

    python -m research.scalp_backtest --pair EUR_USD

Needs M5/M1/D candles already fetched:
    python -m data_pipeline.fetch_historical --pairs EUR_USD --timeframes D M5 M1 --years 1
"""
import argparse
import sys
from collections import Counter

import pandas as pd

from backtest import rules
from backtest.engine import build_market_data, find_sweep_events
from backtest.setup import OUTCOMES_TRADED, evaluate_setup
from data_pipeline import storage

from . import null_model, scoring
from .costs import SpreadCost
from .engine import simulate
from .exits import get_exit
from .strategy import Signal

HTF_PERIOD = pd.Timedelta(minutes=5)     # M5 stands in for H4: same rule shape, 1/48th the duration
LTF_PERIOD = pd.Timedelta(minutes=1)     # M1 stands in for H1


def load_scalp_market(pair: str):
    daily = storage.load(pair, "D")
    m5 = storage.load(pair, "M5")
    m1 = storage.load(pair, "M1")
    return build_market_data(pair, daily, m5, m1)


def generate_signals(pair: str) -> tuple[list[Signal], dict, pd.DataFrame]:
    """The exact SweepFvg logic (research/strategies/sweep_fvg.py), pointed at M5/M1 instead
    of H4/H1. Returns (signals, funnel, m1_frame) — m1_frame is the execution frame simulate()
    and the null model resolve trades against."""
    market = load_scalp_market(pair)
    results = [evaluate_setup(market, direction, index,
                              htf_period=HTF_PERIOD, ltf_period=LTF_PERIOD)
              for index, direction in find_sweep_events(market)]

    funnel = Counter()
    signals = []
    for r in results:
        funnel[r.outcome] += 1
        if r.outcome in OUTCOMES_TRADED:
            signals.append(Signal(
                instrument=pair, direction=r.direction, entry_index=r.confirm_index,
                entry_time=r.confirm_time, entry_price=r.entry_price, stop_price=r.stop_price,
                target_price=r.target_price, invalidation_price=r.sweep_extreme,
                meta={"sweep_time": r.sweep_time}))
    return signals, dict(funnel), market.h1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", default="EUR_USD")
    parser.add_argument("--exit", default="plan", choices=["close", "touch", "plan"],
                        help="plan (default): hard stop/target plus a manual close on an "
                             "invalidating close, matching what the live bot would actually run.")
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    signals, funnel, m1 = generate_signals(args.pair)
    months = (m1["time"].iloc[-1] - m1["time"].iloc[0]).total_seconds() / (3600 * 24 * 30.44) \
        if len(m1) else 0.0

    print(f"{args.pair}: M5 sweep+FVG / M1 confirmation, {len(m1)} M1 candles spanning "
          f"{months:.1f} months\n")
    print("funnel:", {k: funnel[k] for k in sorted(funnel)})
    print(f"\n{len(signals)} signal(s) reached a trade "
          f"({len(signals) / months:.1f}/month)" if months else "")

    if not signals:
        print("\nNo trade ever confirmed at this scale in the fetched window. Nothing to score.")
        return 0

    exit_policy = get_exit(args.exit)
    cost_model = SpreadCost()
    trades, fallbacks = simulate(signals, m1, exit_policy, cost_model, args.pair,
                                 "sweep_fvg_scalp_m5m1")
    summary = scoring.summarize(trades, months)
    print(f"\nreal trades (exit={args.exit}, costs=spread):")
    for key in ("signals", "resolved", "wins", "losses", "win_rate", "expectancy_gross",
               "expectancy_net", "avg_cost_r", "ci_low", "ci_high", "p_le_zero", "max_dd_net"):
        print(f"  {key}: {summary[key]}")
    if fallbacks:
        print(f"  ({fallbacks} trade(s) had no spread on the entry candle; typical spread used)")

    null = null_model.null_replicates(signals, m1, args.exit, cost_model,
                                      rules.pip_size(args.pair), args.replicates,
                                      direction="same")
    resolved_net = trades[trades["outcome"] != "open"]["r_net"].to_numpy(dtype=float)
    comparison = null_model.compare(float(resolved_net.mean()) if len(resolved_net) else float("nan"),
                                    null)
    print(f"\nrandom-entry null ({args.replicates} replicates, same direction, real stops/targets):")
    print(f"  real mean net R: {comparison.observed:+.4f}  "
         f"null mean net R: {comparison.null_mean:+.4f}  (excess: {comparison.excess:+.4f})")
    print(f"  share of random replicates at least as good as the real result: "
         f"{comparison.p_value:.3f}")
    print(f"  null mean cost in R per trade: {null.mean_cost():.4f}")

    print("\nThis is a first-pass check on one pair and one window, not the same rigor as "
         "PREREGISTRATION.md's validated strategies (no walk-forward split, no "
         "multiple-testing correction). Treat any positive number here as a reason to look "
         "harder, not as a result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
