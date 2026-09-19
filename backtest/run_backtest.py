"""CLI: run the Phase 2 backtest for one or more pairs and print the report.

Usage:
    python -m backtest.run_backtest
    python -m backtest.run_backtest --pairs EUR_USD GBP_USD
    python -m backtest.run_backtest --trade-log trades.csv
"""
import argparse

import pandas as pd

from data_pipeline import config

from . import metrics
from .engine import load_market_data, find_sweep_events
from .setup import evaluate_setup


def run(pair: str):
    market = load_market_data(pair)
    events = find_sweep_events(market)
    results = [evaluate_setup(market, direction, idx) for idx, direction in events]

    h4_time = market.h4["time"]
    months_spanned = (h4_time.iloc[-1] - h4_time.iloc[0]).total_seconds() / (3600 * 24 * 30.44)

    report = metrics.build_report(pair, results, months_spanned)
    return report, results


def results_to_dataframe(results) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "pair": r.pair,
            "direction": r.direction,
            "sweep_time": r.sweep_time,
            "sweep_extreme": r.sweep_extreme,
            "outcome": r.outcome,
            "confirm_time": r.confirm_time,
            "h1_candles_to_confirm": r.h1_candles_to_confirm,
            "entry_price": r.entry_price,
            "stop_price": r.stop_price,
            "target_price": r.target_price,
            "rr": r.rr,
            "realized_r": r.realized_r,
            "resolve_time": r.resolve_time,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", default=config.PAIRS)
    parser.add_argument("--trade-log", help="Optional path to write the full per-setup log as CSV.")
    args = parser.parse_args()

    all_results = []
    for pair in args.pairs:
        report, results = run(pair)
        print(metrics.format_report(report))
        print()
        all_results.extend(results)

    if args.trade_log:
        df = results_to_dataframe(all_results)
        df.to_csv(args.trade_log, index=False)
        print(f"Wrote {len(df)} setup rows to {args.trade_log}")


if __name__ == "__main__":
    main()
