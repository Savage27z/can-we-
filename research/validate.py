"""CLI: is a strategy's result distinguishable from luck?

    python -m research.validate --all --exit plan --costs spread --workers 11
    python -m research.validate --pairs EUR_USD GBP_USD --null-replicates 2000

For each instrument the strategy is run as usual; then its signals are replayed at random
candles many times (research/null_model.py) to see what random entries with the same stops,
targets and costs would have earned. Reported per instrument and pooled:

    real     mean net R per trade of the real trades
    null     mean net R of the random-entry replays
    excess   real minus null: what the strategy's entry timing added
    p        share of replays at least as good as the real result
    p_BH     p corrected for testing many instruments (false discovery rate)
    p_maxT   p corrected for picking the best instrument (the "best of N" problem)

Then a walk-forward test of pair selection (research/walk_forward.py): would choosing the best
pairs on the past have paid in the future? Every validation is logged in the registry.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backtest import rules
from data_pipeline import config, instruments

from . import engine, multiple_testing, null_model, registry, walk_forward
from .costs import get_cost
from .exits import get_exit
from .null_model import NullResult
from .strategies import STRATEGIES, get_strategy

MAJORS = ("EUR_USD", "GBP_USD", "USD_JPY")
ONE_SIDED_5_PERCENT_Z = 1.645


def _validate_one(job: tuple) -> dict:
    """One instrument: the real trades and the random-entry replays. Runs in a worker process."""
    (strategy_name, instrument, exit_name, cost_name, slippage, start, replicates, window_days,
     direction, seed) = job
    try:
        strategy = get_strategy(strategy_name)
        window = engine.window_for(strategy.timeframes, instrument, start)
        frames = engine.load_frames(instrument, strategy.timeframes, window)
        h1 = frames[engine.EXECUTION_FRAME]
        if h1.empty:
            raise ValueError("no H1 candles in the requested window")
        output = strategy.generate(instrument, frames)
        cost = get_cost(cost_name, slippage_pips=slippage)
        trades, _ = engine.simulate(output.signals, h1, get_exit(exit_name), cost, instrument,
                                    strategy.name)
        null = null_model.null_replicates(output.signals, h1, exit_name, cost,
                                          rules.pip_size(instrument), replicates, window_days,
                                          direction, seed, key=instrument)
    except Exception as err:          # one bad instrument must not end a 68-pair run
        return {"instrument": instrument, "error": f"{type(err).__name__}: {err}"}
    return {"instrument": instrument, "trades": trades, "null_sums": null.sums,
            "null_counts": null.counts, "null_cost_sums": null.cost_sums,
            "data_from": h1["time"].iloc[0]}


def per_instrument_table(outcomes: list[dict]) -> tuple[pd.DataFrame, dict[str, NullResult]]:
    """One row per instrument with the real result, its null, and the corrected p-values."""
    rows, nulls = [], {}
    for o in outcomes:
        trades = o["trades"]
        resolved = trades[trades["outcome"] != "open"]
        if resolved.empty:
            continue
        null = NullResult(o["null_sums"], o["null_counts"], o.get("null_cost_sums"))
        c = null_model.compare(float(resolved["r_net"].mean()), null)
        rows.append({"instrument": o["instrument"], "data_from": o["data_from"],
                     "n": len(resolved), "real": c.observed, "null": c.null_mean,
                     "excess": c.excess, "z": c.z, "p": c.p_value, "null_sd": c.null_sd,
                     "avg_cost_r": float(resolved["cost_r"].mean())})
        nulls[o["instrument"]] = null
    table = pd.DataFrame(rows)
    if table.empty:
        return table, nulls
    table["p_bh"] = multiple_testing.benjamini_hochberg(table["p"])
    table["p_holm"] = multiple_testing.holm(table["p"])
    null_z = []
    for name in table["instrument"]:
        z = null_model.standardised_null(nulls[name])
        null_z.append(z if z is not None else np.full(len(nulls[name].sums), np.nan))
    table["p_maxt"] = multiple_testing.max_t(table["z"].to_numpy(), np.vstack(null_z))
    return table, nulls


def _num(value, spec: str, width: int) -> str:
    return f"{'n/a':>{width}}" if value is None or not np.isfinite(value) else format(value, spec)


def format_table(table: pd.DataFrame, rows: int) -> str:
    header = (f"{'pair':8} {'n':>4} {'real':>7} {'null':>7} {'excess':>7} {'z':>6} {'p':>7} "
              f"{'p_BH':>6} {'p_maxT':>7} {'costR':>6}")
    body = [header, "-" * len(header)]
    shown = table.sort_values("p").head(rows)
    extra = table[table["instrument"].isin(MAJORS) & ~table["instrument"].isin(shown["instrument"])]
    for _, r in pd.concat([shown, extra]).iterrows():
        body.append(f"{r['instrument']:8} {r['n']:>4} {_num(r['real'], '+7.3f', 7)} "
                    f"{_num(r['null'], '+7.3f', 7)} {_num(r['excess'], '+7.3f', 7)} "
                    f"{_num(r['z'], '+6.2f', 6)} {_num(r['p'], '7.4f', 7)} "
                    f"{_num(r['p_bh'], '6.3f', 6)} {_num(r['p_maxt'], '7.3f', 7)} "
                    f"{_num(r['avg_cost_r'], '6.3f', 6)}")
    return "\n".join(body)


def decompose(resolved: pd.DataFrame, pooled_null: NullResult) -> dict:
    """Splits real-versus-random into what entry timing did before costs and what the spread at the
    moment of entry cost, per trade in R. "cost" is a positive amount here, subtracted from gross."""
    real_cost = float(resolved["cost_r"].mean())
    real_net = float(resolved["r_net"].mean())
    null_cost = pooled_null.mean_cost()
    null_net = float(pooled_null.sums.sum() / pooled_null.counts.sum())
    return {"real": {"gross": real_net + real_cost, "cost": real_cost, "net": real_net},
            "random": {"gross": null_net + null_cost, "cost": null_cost, "net": null_net}}


def format_decomposition(d: dict) -> str:
    def row(label, r):
        return f"{label:18} {r['gross']:+8.3f} {-r['cost']:+8.3f} {r['net']:+8.3f}"
    diff = {k: d["real"][k] - d["random"][k] for k in ("gross", "cost", "net")}
    return "\n".join([
        f"{'per trade, in R':18} {'gross':>8} {'cost':>8} {'net':>8}",
        row("real trades", d["real"]), row("random entries", d["random"]),
        f"{'real minus random':18} {diff['gross']:+8.3f} {-diff['cost']:+8.3f} {diff['net']:+8.3f}"])


def entry_hours(resolved: pd.DataFrame, top: int = 4) -> str:
    """Where the real entries fall in the day (UTC hour the entry candle opened), since a strategy
    that funnels entries into a few hours pays for whatever those hours are like."""
    opened = (pd.to_datetime(resolved["entry_time"], utc=True) - pd.Timedelta(hours=1)).dt.hour
    share = opened.value_counts(normalize=True).head(top)
    return ", ".join(f"{hour:02d}:00 {100 * v:.0f}%" for hour, v in share.items())


def format_walk_forward(wf: walk_forward.WalkForwardResult, top_n: int) -> str:
    if not wf.folds or not np.isfinite(wf.selected_mean):
        return "not enough history to run the walk-forward test"
    lines = [f"{'test block':11} {'pairs ranked':>12} {'top-' + str(top_n) + ' trained':>13} "
             f"{'top-' + str(top_n) + ' test':>11} {'all test':>9}   selected"]
    for f in wf.folds:
        block = f"{f.test_start.year}-{f.test_end.year - 1}"
        lines.append(f"{block:11} {f.eligible:>12} {f.selected_train_mean:>+13.3f} "
                     f"{_num(f.selected_test_mean, '+11.3f', 11)} {_num(f.all_test_mean, '+9.3f', 9)}   "
                     f"{', '.join(f.selected)}")
    lines.append("")
    lines.append(f"selected pairs, out of sample: {wf.selected_mean:+.3f}R per trade over "
                 f"{wf.selected_trades} trades")
    lines.append(f"all ranked pairs, same blocks: {wf.all_mean:+.3f}R    "
                 f"randomly chosen pairs: {wf.random_pick_mean:+.3f}R")
    lines.append(f"selection beat random picking with p = {wf.p_value:.3f}; it beat 'all pairs' in "
                 f"{wf.folds_won} of {len(wf.folds)} blocks")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", default="sweep_fvg", choices=sorted(STRATEGIES))
    parser.add_argument("--pairs", nargs="+", default=["EUR_USD"])
    parser.add_argument("--all", action="store_true", help="Every instrument in the catalog.")
    parser.add_argument("--exit", default="plan", choices=["close", "touch", "plan"])
    parser.add_argument("--costs", default="spread", choices=["none", "spread"])
    parser.add_argument("--slippage-pips", type=float, default=0.0)
    parser.add_argument("--start", default="auto")
    parser.add_argument("--null-replicates", type=int, default=1000,
                        help="Random-entry replays per signal. 1000 resolves p-values down to 0.001.")
    parser.add_argument("--window-days", type=float, default=null_model.DEFAULT_WINDOW_DAYS,
                        help="Random entries fall within this many days of the real one.")
    parser.add_argument("--null-direction", default="same", choices=null_model.DIRECTIONS,
                        help="same: keep each signal's direction, testing entry timing. "
                             "random: randomise it too, testing timing and direction together.")
    parser.add_argument("--top-n", type=int, default=walk_forward.DEFAULT_TOP_N)
    parser.add_argument("--first-test-year", type=int, default=walk_forward.DEFAULT_FIRST_TEST_YEAR)
    parser.add_argument("--test-years", type=int, default=walk_forward.DEFAULT_TEST_YEARS)
    parser.add_argument("--min-train-trades", type=int, default=walk_forward.DEFAULT_MIN_TRAIN_TRADES)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows", type=int, default=15, help="Instruments to print (best p first).")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--note", default="", help="A note to store with the run in the registry.")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.data_dir:
        os.environ["DATA_DIR"] = str(Path(args.data_dir).resolve())
        config.DATA_DIR = Path(args.data_dir).resolve()

    pairs = instruments.names() if args.all else args.pairs
    jobs = [(args.strategy, p, args.exit, args.costs, args.slippage_pips, args.start,
             args.null_replicates, args.window_days, args.null_direction, args.seed) for p in pairs]
    print(f"VALIDATION  strategy {args.strategy}  exit {args.exit}  costs {args.costs}  "
          f"start {args.start}  {len(pairs)} instrument(s)\n"
          f"null: {args.null_replicates} random-entry replays per signal, entries within "
          f"+-{args.window_days:g} days, direction {args.null_direction}\n")

    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            outcomes = list(pool.map(_validate_one, jobs))
    else:
        outcomes = [_validate_one(job) for job in jobs]
    failed = [o for o in outcomes if "error" in o]
    done = [o for o in outcomes if "error" not in o]

    table, nulls = per_instrument_table(done)
    if table.empty:
        print("no instrument produced any resolved trade")
        for o in failed:
            print(f"FAILED {o['instrument']}: {o['error']}")
        return 1

    print(format_table(table, args.rows))

    all_trades = pd.concat([o["trades"] for o in done], ignore_index=True)
    resolved = all_trades[all_trades["outcome"] != "open"]
    pooled_null = null_model.pool([nulls[name] for name in table["instrument"]])
    pooled = null_model.compare(float(resolved["r_net"].mean()), pooled_null)
    print(f"\nPOOLED over {len(table)} instruments, {len(resolved)} trades: real "
          f"{pooled.observed:+.3f}R, random entries {pooled.null_mean:+.3f}R, excess "
          f"{pooled.excess:+.3f}R (z {pooled.z:+.1f}, p {pooled.p_value:.4f})")
    print(f"average cost per trade {resolved['cost_r'].mean():.3f}R "
          f"(median {resolved['cost_r'].median():.3f}R)\n")
    decomposition = decompose(resolved, pooled_null)
    print(format_decomposition(decomposition))
    # "Not significant" is only informative next to how large an effect the test could have seen.
    detectable = ONE_SIDED_5_PERCENT_Z * pooled.null_sd
    per_pair_sd = float(table["null_sd"].median())
    print(f"most common entry-candle opens (UTC): {entry_hours(resolved)}")
    print(f"power: the pooled test would have detected an entry-timing excess of about "
          f"{detectable:.3f}R per trade or more; a single pair (typical null spread "
          f"{per_pair_sd:.2f}R) only one above about {ONE_SIDED_5_PERCENT_Z * per_pair_sd:.2f}R")

    alpha = args.alpha
    counts = {"raw": int((table["p"] < alpha).sum()), "bh": int((table["p_bh"] < alpha).sum()),
              "holm": int((table["p_holm"] < alpha).sum()),
              "maxt": int((table["p_maxt"] < alpha).sum())}
    positive = table[table["excess"] > 0]
    print(f"\nInstruments beating random entry at {alpha:.0%}, of {len(table)}: "
          f"raw {counts['raw']}, after correction: BH {counts['bh']}, Holm {counts['holm']}, "
          f"max-t {counts['maxt']}   (about {alpha * len(table):.1f} would pass raw by luck alone)")
    print(f"{len(positive)} of {len(table)} instruments have a positive excess over random entry")

    print("\nWALK-FORWARD: choosing the best pairs on the past, scored on the future")
    wf = walk_forward.walk_forward_selection(
        all_trades, top_n=args.top_n, first_test_year=args.first_test_year,
        test_years=args.test_years, min_train_trades=args.min_train_trades, seed=args.seed)
    print(format_walk_forward(wf, args.top_n))

    results = {
        "pooled": {"trades": len(resolved), "real": pooled.observed, "null": pooled.null_mean,
                   "excess": pooled.excess, "z": pooled.z, "p": pooled.p_value,
                   "null_sd": pooled.null_sd, "min_detectable_excess": detectable},
        "per_pair_null_sd_median": per_pair_sd, "decomposition": decomposition,
        "survivors": counts, "instruments_tested": len(table),
        "walk_forward": {"selected_mean": wf.selected_mean, "all_mean": wf.all_mean,
                         "random_pick_mean": wf.random_pick_mean, "p": wf.p_value,
                         "folds": len(wf.folds), "folds_won": wf.folds_won,
                         "selected_trades": wf.selected_trades, "top_n": args.top_n},
    }
    if not args.no_register:
        registry.record("validation", args.strategy, args.exit, args.costs, args.start,
                        [o["instrument"] for o in done], results, slippage_pips=args.slippage_pips,
                        note=args.note, extra={"null": {"replicates": args.null_replicates,
                                        "window_days": args.window_days,
                                        "direction": args.null_direction, "seed": args.seed}})
        history = registry.trial_counts()
        best_p = float(table["p"].min())
        print(f"\nREGISTRY: {history['runs']} runs, {history['configs']} configurations and "
              f"{history['pair_tests']} pair-tests logged so far. Counting every one of them, the "
              f"best raw p of {best_p:.4f} would be at most {min(1.0, best_p * history['pair_tests']):.3f}.")

    for o in failed:
        print(f"\nSKIPPED {o['instrument']}: {o['error']}")

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "validation.csv", index=False)
        all_trades.to_csv(out / "trades.csv", index=False)
        pd.DataFrame([{**vars(f), "selected": ", ".join(f.selected)} for f in wf.folds]).to_csv(
            out / "walk_forward.csv", index=False)
        (out / "validation.json").write_text(json.dumps(
            {"settings": vars(args), "results": results}, indent=1, default=str) + "\n",
            encoding="utf-8")
        print(f"\nwrote {out}/validation.csv, walk_forward.csv, trades.csv, validation.json")
    return 1 if failed and not done else 0


if __name__ == "__main__":
    sys.exit(main())
