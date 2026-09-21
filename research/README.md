# Research platform

Runs a strategy over any of the 68 forex pairs OANDA offers, ends each trade under an
explicit exit rule, charges real spread costs, and scores the result in R. It is separate
from the live bot (`live/`, `tgbot/`), which it never touches.

```
data_pipeline/   fetch candles + spread, instrument catalog, data-quality checks
backtest/        the sweep/FVG rules (strategy_rules.md) and the original backtest
research/        strategy interface, exits, costs, engine, scoring, CLI
```

## Workflow

```bash
# 1. Data (read-only against OANDA; needs OANDA_API_KEY and OANDA_ACCOUNT_ID in .env)
python -m data_pipeline.instruments --refresh                 # the 68-pair catalog (committed)
python -m data_pipeline.fetch_historical --all --from-start --workers 4   # ~12M candles, resumable

# 2. Check it before trusting it
python -m data_pipeline.quality --all --out data/quality_report.csv

# 3. Backtest
python -m research.run_backtest --pairs EUR_USD GBP_USD --exit plan --costs spread
python -m research.run_backtest --all --workers 8 --exit plan --costs spread --out-dir data/runs/my_run
```

`--exit` is how a trade ends and `--costs` is what it costs; both matter (see below).
`--start auto` (the default) runs each pair only over the part of its history that is
dense in H1, H4 and Daily.

## Things to know before believing a number

- **Data starts sparse.** OANDA's earliest H4/H1 candles are mostly missing (the majors are
  ~95% empty until the end of 2004; USD_CNH until mid-2014; USD_TRY never becomes dense).
  `--start auto` excludes that; `--start all` does not.
- **Exits change the answer.** `close` is the strategy's own scoring (a win is an H1 close
  at the target). `touch` is hard stop/target orders, which act on wicks. `plan` is what the
  live trade plan tells a person to do. On EUR_USD 2005-2026 these give +0.34R, +0.23R and
  +0.23R before costs.
- **Costs are charged.** One spread per trade, taken at the entry candle, converted to R.
  It is a realistic floor, not an upper bound (spreads widen in fast markets).
- **Confidence intervals matter more than point estimates.** A few dozen trades almost never
  separate an edge from luck. Read the interval, not just the mean.
- **A table of 68 pairs is exploratory.** With no edge at all, about half would show a
  positive expectancy by chance, and the best few would look convincing. Do not pick
  winners from it; validation (out-of-sample, walk-forward, multiple-testing control) is
  what turns a ranking into evidence.
- **Trades are independent.** Each signal is resolved on its own, with no limit on how many
  are open at once, and pooled results treat correlated pairs (USD crosses) as independent.
- **DST.** OANDA aligns Daily and H4 to 17:00 New York, so the H4 session filter admits four
  candles a day in summer and three in winter (`strategy_rules.md` section 5).

## Reproducing the original result

The original backtest (EUR_USD: 37 signals, +0.34R over 8 years) was measured on data now
superseded by the longer store. A frozen copy is kept out of git in
`data/_snapshots/phase2_8y`:

```bash
python -m research.run_backtest --data-dir data/_snapshots/phase2_8y --pairs EUR_USD --start all
```

`tests/test_research_parity.py` checks, on that snapshot, that every one of 2,094 sweep
setups matches the legacy evaluator exactly (outcome, R, entry and exit times).

## Adding a strategy

1. Write a class in `research/strategies/` with `name`, `timeframes`, and
   `generate(instrument, frames) -> StrategyOutput`. Return `Signal`s: entry candle, entry
   price, stop, target. The strategy does not decide how a trade ends or what it costs.
   `Signal` rejects a stop, entry and target that are out of order.
2. Register it in `research/strategies/__init__.py`.
3. Test it on a hand-built market whose answer you know, as `tests/test_research.py` does
   for sweep/FVG with the strategy document's worked example.
4. Run it with `--exit close`, then `plan` and `touch`, with and without `--costs spread`.
   A strategy that only works under the most generous assumptions has not worked.

## Findings so far (data to 2026-09-21; sweep/FVG, one run of the platform)

- **Data:** 68 pairs, 12.0 million candles, no hard errors, H4 candles agree exactly with the
  H1 candles they are made of. The three TRY pairs have no dense H4/H1 history and are skipped.
- **On the strategy's own scoring** (`--exit close --costs none`), 65 pairs and 7,599 trades:
  pooled +0.06R per trade (interval +0.01 to +0.10, treating trades as independent, which
  correlated pairs are not); 40 of 65 pairs positive.
- **Under realistic assumptions** (`--exit plan --costs spread`): pooled -0.43R; 5 of 65 pairs
  positive, none with an interval wholly above zero, 32 wholly below. The median pair pays
  0.23R per trade in spread alone, because the stops are tight relative to the spread.
- **EUR_USD**, the pair the live bot uses: 116 trades 2005-2026 give +0.34R on the strategy's own
  scoring (interval 0.00 to +0.70) and +0.17R under realistic assumptions (-0.17 to +0.53).
  Its 2005-2018 half, which played no part in choosing it, gave +0.35R and +0.18R. It is one of
  the best few of 65 pairs, a rank that chance alone could produce; treat it as a hypothesis.
- **Pair differences persist over time** (correlation +0.29 between the 2005-2015 and 2016-2026
  halves, 1.3% of label shuffles do as well) and are not just drift (USD_ZAR earns the same on
  buys and sells), but they do not survive costs.

None of this validates the strategy. It shows the platform works, and that with costs charged
the strategy has not been distinguished from having no edge.

## Not built yet

Out-of-sample and walk-forward testing, a null model (random entries), correction for the
number of strategies and pairs tried, portfolio limits, and any order execution.
