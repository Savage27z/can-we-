"""CLI: backtest a strategy over one or more instruments.

    python -m research.run_backtest --pairs EUR_USD --exit close --costs none
    python -m research.run_backtest --pairs EUR_USD GBP_USD --exit plan --costs spread
    python -m research.run_backtest --all --workers 8 --costs spread --out-dir data/runs/sweep_fvg_all

By default (--start auto) each instrument is run only over the part of its history where
H1, H4 and Daily are all dense (see python -m data_pipeline.quality): OANDA's earliest H4/H1
candles are mostly missing, and a strategy run over holes measures nothing real. Use
--start all for every stored candle, or --start 2015-01-01 for a fixed date.

To reproduce the original "37 signals, +0.34R" result, point it at the frozen snapshot:

    python -m research.run_backtest --data-dir data/_snapshots/phase2_8y --pairs EUR_USD

Exits: close = the strategy's own H1-close scoring, touch = hard stop/target orders,
plan = hard orders plus closing by hand on an invalidating close. Costs: none, or one
spread per trade (from the stored bid/ask spread) plus optional --slippage-pips.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from data_pipeline import config, instruments

from . import engine, registry, scoring
from .costs import get_cost
from .exits import get_exit
from .strategies import STRATEGIES, get_strategy


def _window_for(strategy: str, instrument: str, start: str):
    strat = get_strategy(strategy)
    return engine.window_for(strat.timeframes, instrument, start, engine.warmup_of(strat))


def _run_one(job: tuple) -> dict:
    """Runs in a worker process, so it takes and returns plain picklable values."""
    strategy, instrument, exit_name, cost_name, slippage, start = job
    try:
        run = engine.run_instrument(get_strategy(strategy), instrument, get_exit(exit_name),
                                    get_cost(cost_name, slippage_pips=slippage),
                                    window=_window_for(strategy, instrument, start))
    except Exception as err:      # one bad instrument must not end a 68-pair run
        return {"instrument": instrument, "error": f"{type(err).__name__}: {err}"}
    return {"instrument": instrument, "trades": run.trades, "funnel": run.funnel,
            "months": run.months_spanned, "fallbacks": run.spread_fallbacks,
            "data_from": run.data_from, "data_to": run.data_to}


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent.parent, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _fmt(value, spec: str, missing: str = "  n/a") -> str:
    return missing if value is None else format(value, spec)


def summary_row(instrument: str, summary: dict, data_from=None) -> dict:
    return {"instrument": instrument, "data_from": data_from, **summary}


def format_table(rows: list[dict]) -> str:
    header = (f"{'pair':8} {'from':>7} {'n':>4} {'W/L':>7} {'win%':>6} {'grossR':>7} {'netR':>7} "
              f"{'costR':>6} {'95% CI on netR':>17} {'maxDD':>6}")
    lines = [header, "-" * len(header)]
    for r in rows:
        ci = ("       n/a" if r["ci_low"] is None
              else f"[{r['ci_low']:+6.2f}, {r['ci_high']:+6.2f}]")
        wl = f"{r['wins']}/{r['losses']}"
        began = "" if r.get("data_from") is None else pd.Timestamp(r["data_from"]).strftime("%Y-%m")
        lines.append(f"{r['instrument']:8} {began:>7} {r['signals']:>4} {wl:>7} "
                     f"{_fmt(None if r['win_rate'] is None else 100 * r['win_rate'], '6.1f')} "
                     f"{_fmt(r['expectancy_gross'], '+7.2f')} {_fmt(r['expectancy_net'], '+7.2f')} "
                     f"{_fmt(r['avg_cost_r'], '6.3f')} {ci:>17} {_fmt(r['max_dd_net'], '6.2f')}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", default="sweep_fvg", choices=sorted(STRATEGIES))
    parser.add_argument("--pairs", nargs="+", default=["EUR_USD"])
    parser.add_argument("--all", action="store_true", help="Every instrument in the catalog.")
    parser.add_argument("--exit", default="close", choices=["close", "touch", "plan"])
    parser.add_argument("--costs", default="none", choices=["none", "spread"])
    parser.add_argument("--slippage-pips", type=float, default=0.0)
    parser.add_argument("--start", default="auto",
                        help="auto (each instrument's dense history), all (every stored "
                             "candle), or a date such as 2015-01-01.")
    parser.add_argument("--workers", type=int, default=1, help="Instruments to run at once.")
    parser.add_argument("--data-dir", default=None, help="Read candles from here instead.")
    parser.add_argument("--out-dir", default=None,
                        help="Write trades.csv, summary.csv and run.json here (and log the run "
                             "in the registry).")
    parser.add_argument("--no-register", action="store_true",
                        help="Do not log a saved run in the registry.")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):      # absent when output is captured or piped by a wrapper
        sys.stdout.reconfigure(encoding="utf-8")

    if args.data_dir:
        # Set both: this process reads config, spawned workers re-read the environment.
        os.environ["DATA_DIR"] = str(Path(args.data_dir).resolve())
        config.DATA_DIR = Path(args.data_dir).resolve()

    pairs = instruments.names() if args.all else args.pairs
    jobs = [(args.strategy, p, args.exit, args.costs, args.slippage_pips, args.start)
            for p in pairs]
    print(f"strategy {args.strategy}  exit {args.exit}  costs {args.costs}"
          + (f" (+{args.slippage_pips:g} pip slippage)" if args.slippage_pips else "")
          + f"  start {args.start}  data {config.DATA_DIR}  {len(pairs)} instrument(s)\n")

    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            outcomes = list(pool.map(_run_one, jobs))
    else:
        outcomes = [_run_one(job) for job in jobs]

    failed = [o for o in outcomes if "error" in o]
    done = [o for o in outcomes if "error" not in o]
    rows = [summary_row(o["instrument"], scoring.summarize(o["trades"], o["months"]),
                        o["data_from"]) for o in done]
    print(format_table(rows))

    all_trades = pd.concat([o["trades"] for o in done], ignore_index=True) if done else pd.DataFrame()
    pooled = scoring.summarize(all_trades) if len(all_trades) else None
    if pooled:
        # A running total over overlapping trades on many unrelated instruments is not an
        # equity curve anyone would experience, so its drawdown means nothing.
        pooled["max_dd_gross"] = pooled["max_dd_net"] = None
    if len(done) > 1 and pooled and pooled["resolved"]:
        print("\nPOOLED over all instruments (trades treated as independent):")
        print(format_table([summary_row("ALL", pooled)]))
        with_trades = [r for r in rows if r["resolved"]]
        positive = sum(1 for r in with_trades if r["expectancy_net"] > 0)
        print(f"\n{positive} of {len(with_trades)} instruments show positive net expectancy. "
              f"With no edge at all and independent instruments about half would, by chance;\n"
              f"picking the best few after the fact is how false edges get found. Rankings from "
              f"a run like this are exploratory, not validated.")
    fallbacks = sum(o["fallbacks"] for o in done)
    if fallbacks:
        print(f"\nnote: {fallbacks} trade(s) had no spread on the entry candle and were "
              f"charged the instrument's typical spread")
    for o in failed:
        print(f"\nFAILED {o['instrument']}: {o['error']}")

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        all_trades.to_csv(out / "trades.csv", index=False)
        pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
        record = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git_commit(), "strategy": args.strategy, "exit": args.exit,
            "costs": args.costs, "slippage_pips": args.slippage_pips, "start": args.start,
            "data_dir": str(config.DATA_DIR), "instruments": pairs,
            "failed": {o["instrument"]: o["error"] for o in failed},
            "pooled": pooled,
        }
        (out / "run.json").write_text(json.dumps(record, indent=1, default=str) + "\n",
                                      encoding="utf-8")
        print(f"\nwrote {out}/trades.csv, summary.csv, run.json")
        if not args.no_register:
            registry.record("backtest", args.strategy, args.exit, args.costs, args.start,
                            [o["instrument"] for o in done], pooled or {},
                            slippage_pips=args.slippage_pips, note=f"saved to {out.name}")
            counts = registry.trial_counts()
            print(f"registry: {counts['runs']} runs, {counts['configs']} configurations, "
                  f"{counts['pair_tests']} pair-tests logged so far")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
