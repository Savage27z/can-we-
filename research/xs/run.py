"""CLI: run the frozen cross-sectional experiment (research/PREREGISTRATION_XS.md).

    python -m research.xs.run --workers 8 --out-dir data/runs/xs_frozen

Loads the sixteen currencies, builds the leg tables for each strategy, and reports: the momentum
and reversal strategies with their interval, gross permutation p-value and halves; the two controls
on each strategy's own cohorts (random assignments, and the forward-looking oracle); and the
pre-registered verdicts. Every strategy is logged in the registry.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from data_pipeline import config

from .. import engine, registry
from . import inference, rules, universe
from .legs import CurrencyData, Tables, build_tables
from .select import K, select

WARMUP = pd.Timedelta(days=200)          # daily history kept before the first cohort: 84 days + ATR + slack

# name, formation days, hold weeks, rule: exactly the two strategies of the pre-registration.
STRATEGIES = (("xs_momentum", 84, 4, "momentum"), ("xs_reversal", 7, 1, "reversal"))


def _load(job: tuple) -> Optional[CurrencyData]:
    currency, start = job
    pair = universe.pair_of(currency)
    try:
        window = engine.window_for(("D", "H1"), pair, start, WARMUP)
        frames = engine.load_frames(pair, ("D", "H1"), window)
    except ValueError:
        return None
    if frames["H1"].empty or frames["D"].empty:
        return None
    return CurrencyData(currency, frames["D"], frames["H1"],
                        window.start if window is not None else None)


def load_universe(start: str, workers: int) -> dict[str, CurrencyData]:
    jobs = [(c, start) for c in universe.CURRENCIES]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            loaded = list(pool.map(_load, jobs))
    else:
        loaded = [_load(j) for j in jobs]
    return {c: d for c, d in zip(universe.CURRENCIES, loaded) if d is not None}


def eligibility_by_year(tables: Tables) -> str:
    counts = pd.Series(tables.eligible.sum(axis=1), index=tables.cohorts.year)
    yearly = counts.groupby(level=0).mean()
    return "  ".join(f"{y}:{v:.0f}" for y, v in yearly.items() if y % 3 == 2005 % 3 or y == yearly.index[-1])


def _row(label: str, r: inference.StrategyResult) -> str:
    return (f"{label:14} {r.legs:>6} {r.gross_mean:+9.4f} {-r.cost_mean:+9.4f} {r.net_mean:+9.4f}  "
            f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]  {r.p_gross:8.4f}")


def format_strategy(r: inference.StrategyResult) -> str:
    header = (f"{'':14} {'legs':>6} {'gross':>9} {'cost':>9} {'net':>9}  {'95% interval on net':>20}  "
              f"{'gross p':>8}")
    lines = [f"{r.name}: formation {r.formation_days} days, hold {r.hold_weeks} week(s), "
             f"{r.cohorts} cohorts", header, _row("R per leg", r)]
    lines.append(f"null (random assignment): gross {r.null_mean:+.4f}R, sd {r.null_sd:.4f}; the test "
                 f"would have detected a gross excess of about {r.detectable:.4f}R per leg")
    lines.append(f"halves, net R per leg: 2005-2015 {r.early_net:+.4f} (n={r.early_legs}), "
                 f"2016-2026 {r.late_net:+.4f} (n={r.late_legs})")
    lines.append(f"long legs gross {r.long_gross:+.4f} net {r.long_net:+.4f}; "
                 f"short legs gross {r.short_gross:+.4f} net {r.short_net:+.4f}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="auto", help="auto (dense history) or all.")
    parser.add_argument("--replicates", type=int, default=10_000, help="Permutation null replicates.")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--controls", type=int, default=200, help="Random assignments for X0.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--note", default="")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.data_dir:
        os.environ["DATA_DIR"] = str(Path(args.data_dir).resolve())
        config.DATA_DIR = Path(args.data_dir).resolve()

    data = load_universe(args.start, args.workers)
    missing = sorted(set(universe.CURRENCIES) - set(data))
    print(f"CROSS-SECTIONAL EXPERIMENT (frozen: research/PREREGISTRATION_XS.md)\n"
          f"{len(data)} of {len(universe.CURRENCIES)} currencies loaded"
          + (f"; missing {', '.join(missing)}" if missing else "") + f"; k = {K} long and {K} short\n")
    if len(data) < 8:
        print("too few currencies to rank")
        return 1

    report: dict = {"settings": {k: v for k, v in vars(args).items()}, "strategies": {}}
    all_checks_pass = True
    candidates = []
    for name, formation, hold, rule in STRATEGIES:
        tables = build_tables(data, formation, hold)
        null = inference.permutation_null(tables, args.replicates, seed=args.seed)
        chosen = select(tables, rule)
        result = inference.evaluate(name, tables, chosen, null, seed=args.seed,
                                    bootstrap_replicates=args.bootstrap)

        control_p = inference.control_p_values(tables, null, args.controls, seed=args.seed + 1)
        oracle_null = null           # the oracle is tested against the same permutation null
        oracle = inference.evaluate(f"{name}:oracle", tables, select(tables, "oracle"), oracle_null,
                                    seed=args.seed, bootstrap_replicates=1000)
        control_checks = rules.evaluate_controls(name, control_p, oracle)
        checks = rules.evaluate_candidate(result)

        print("=" * 100)
        print(format_strategy(result))
        print(f"eligible currencies per cohort (mean, by year): {eligibility_by_year(tables)}")
        print(f"\nchecks for this strategy's cohorts (must pass for the result to be trusted):")
        print(rules.format_checks(name, control_checks, "controls"))
        print(f"\ndecision rules:")
        print(rules.format_checks(name, checks, "candidate under the pre-registered rules"))
        print("\nper currency, exploratory only (legs, gross, net):")
        print(result.per_currency.round(4).to_string(index=False))
        print()

        trusted = rules.verdict(control_checks) == "PASS"
        all_checks_pass &= trusted
        if trusted and rules.verdict(checks) == "PASS":
            candidates.append(name)
        report["strategies"][name] = {
            "result": {k: v for k, v in asdict(result).items() if k != "per_currency"},
            "oracle": {k: v for k, v in asdict(oracle).items() if k != "per_currency"},
            "control_p": control_p.tolist(),
            "checks": {"controls": [asdict(c) for c in control_checks],
                       "decision": [asdict(c) for c in checks]},
            "controls_passed": trusted, "candidate": trusted and rules.verdict(checks) == "PASS"}
        if args.out_dir:
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            result.per_currency.to_csv(out / f"per_currency_{name}.csv", index=False)
            pd.DataFrame({"cohort": tables.cohorts[chosen.cohort],
                          "currency": [tables.currencies[j] for j in chosen.currency],
                          "direction": chosen.direction, "gross_r": chosen.gross,
                          "cost_r": chosen.cost}).to_csv(out / f"legs_{name}.csv", index=False)

        if not args.no_register:
            params = {"formation_days": formation, "hold_weeks": hold, "k": K, "rule": rule}
            summary = {k: v for k, v in asdict(result).items() if k != "per_currency"}
            registry.record("crosssection", name, "time", "spread", args.start,
                            [universe.pair_of(c) for c in data], summary, params=params,
                            note=args.note or "cross-sectional experiment (frozen pre-registration)")
            registry.record("crosssection", f"{name}:controls", "time", "spread", args.start,
                            [universe.pair_of(c) for c in data],
                            {"x0_mean_p": float(control_p.mean()), "oracle_p": oracle.p_gross},
                            params=params, extra={"control": True},
                            note="X0 random assignments and X9 oracle")

    print("=" * 100)
    if not all_checks_pass:
        print("SOME CONTROLS FAILED: the results above are not to be trusted.")
    print("CANDIDATES: " + (", ".join(candidates) if candidates else "none"))
    report["candidates"] = candidates
    if args.out_dir:
        (Path(args.out_dir) / "results.json").write_text(json.dumps(report, indent=1, default=str) + "\n",
                                                         encoding="utf-8")
        print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
