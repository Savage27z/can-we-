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

# 4. Validate: is the result distinguishable from luck? (about 10 minutes for all 65 pairs)
python -m research.validate --all --workers 11 --exit plan --costs spread --out-dir data/runs/my_validation
python -m research.registry                                   # every run logged so far, and the trial count
python -m research.preregistered s4_donchian_random           # apply the pre-registered decision rules

# 5. Cross-sectional (16 currencies ranked against each other; the frozen experiment of stage 4b)
python -m research.xs.run --workers 8 --out-dir data/runs/xs_frozen
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

## How validation works

`research.validate` answers three questions, and every number it prints has to survive all three.

1. **Does the entry signal beat random entry?** Each real signal is replayed at random in-session
   candles within 60 days of the real one, keeping its direction, risk, reward and invalidation
   distance, under the same exit and cost rules (`null_model.py`). The result minus the random-entry
   result is the `excess`: what the entry timing added, everything else held equal. The p-value is
   the share of replays that did at least as well. This is calibrated: on 400 synthetic strategies
   with no skill the p-values are indistinguishable from uniform, and a strategy that peeks at the
   future gets tiny ones (`tests/test_null_model.py`).
2. **Does it survive having looked at 65 pairs?** The p-values are corrected for testing many
   instruments: Benjamini-Hochberg (false discovery rate), Holm (any false positive), and a
   max-statistic correction for "the best of N" (`multiple_testing.py`).
3. **Would choosing pairs on the past have paid?** The walk-forward test ranks pairs using only
   trades before each two-year block, picks the top five, and scores them on the block, against
   randomly chosen pairs (`walk_forward.py`). The strategy's rules are fixed, so pair selection is
   the only thing fitted to data.

Every saved run and validation is logged in `data/registry/runs.jsonl` with the code version, and
the trial count (configurations and instruments looked at) is printed with each validation. A
p-value means something different after 3 looks than after 300.

**Which null.** `--null-direction random` (randomise direction as well as timing) asks whether the
strategy beats coin-flip entries with the same stops, targets and costs. `same` (keep each signal's
direction) asks about timing alone, but it is biased: it applies a signal's direction to random dates
before the signal too, where that direction was only knowable in hindsight, which inflates the null
for momentum signals and deflates it for reversal signals. Use `random` for verdicts and treat `same` as a
diagnostic. A strategy can declare `entry_mask(h1)` so the null draws only from candles it could have
entered on, and `daily_warmup_days` for the history its indicators need.

The random-entry draws are independent between instruments while real pairs are correlated, which
makes the max-t correction if anything conservative. The pooled p-value treats trades as
independent, which overlapping trades on correlated pairs are not, so it is somewhat optimistic.

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
3. Test it on a hand-built market whose answer you know, as `tests/test_strategies.py` does, and add it
   to `tests/test_strategies_lookahead.py`, which cuts the data off at several moments and checks the
   strategy gives exactly the signals it gave in the full run (it fails a strategy that peeks, and
   fails the build if a registered strategy has no such test).
   Write the parameters and the decision rules down first, as `PREREGISTRATION.md` does.
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

### Validation (`research.validate`, 65 pairs, 1000 random-entry replays per signal)

- **Before costs the entry signal adds nothing.** On the strategy's own scoring the real trades
  earned +0.057R and random entries with the same stops and targets +0.060R: excess -0.002R
  (p = 0.51, 7,597 trades). The pooled test could have detected an excess of about 0.044R; a
  single pair only one above about 0.29R, so per-pair "not significant" says little.
- **After realistic exits and spread it is worse than random**: -0.426R against -0.337R, an
  excess of -0.088R. Of that, -0.016R is entry timing and -0.072R is extra spread: 23% of the
  strategy's entries are in the 21:00 UTC candle, which closes at the New York rollover, and 32%
  at 07:00, because signals fire just after H4 candles close.
- **No pair survives correction.** 3 of 65 beat random entry at 5% before correction, about
  what chance gives (3.2); none after Benjamini-Hochberg, Holm or max-t. 15 of 65 have a positive
  excess after costs and 30 of 65 before, again about half or fewer.
- **Choosing pairs on the past does work out of sample, but only as far as breakeven.** Ranking
  by net R before each two-year block and taking the top five: +0.279R out of sample on the
  strategy's own scoring (random picking +0.087R, p = 0.043) and -0.025R after realistic costs
  (random picking -0.340R, p = 0.001, 7 of 8 blocks). EUR_USD is among the five in 7 of 8 blocks.
- **Why pair differences persist.** Random entries in the same pairs earn positive R under the
  strategy's scoring too (USD_ZAR +0.14R, HKD_JPY +0.19R), so much of the persistence is the pair's
  payoff geometry and costs, not the entry signal.
- **Looks taken so far:** 6 configurations and 148 pair-tests (`python -m research.registry`).
  The best raw p-value, 0.008, is not significant counting all of them.

### Stage 4: three more strategies, pre-registered (`PREREGISTRATION.md`)

Wide ATR-scaled stops were tried because tight ones let the spread swamp everything (cost per trade
0.065R on average against 0.39R for sweep/FVG). The strategies, parameters and decision rules were
fixed and committed before any was run, and applied by code afterwards.

- **The control passed** (pooled p = 0.50, 4.6% of pairs at raw p < 0.05, mean pair p 0.52), so the
  validation pipeline is calibrated on real data.
- **None of the three is a candidate.** Daily Donchian breakout: -0.113R net, and 0.052R *worse* than
  random entries. London range breakout: -0.179R net; its direction call beats random entry by a
  statistically significant 0.012R per trade (p = 0.002, 217,443 trades) against 0.187R of
  spread, so real but worthless. Daily Bollinger reversion: -0.044R net, negative in both halves.
- **The rollover fix helps but does not rescue sweep/FVG:** dropping 21:00 UTC entries improves it
  from -0.426R to -0.310R and it is then indistinguishable from random entry. The live bot applies
  this rule from `strategy_rules.md` v1.7 (`backtest.rules.NO_ENTRY_HOURS_UTC`, opt-in so backtests
  are unchanged), checked to give exactly this variant's trades on real data.
- **Pairs differ mainly in how cheap they are to trade.** Across pairs net expectancy correlates
  -0.71 to -0.98 with average cost, including for the random-entry control; gross expectancy does not.
  The walk-forward "selection works" result is mostly this.

None of this validates any strategy. Across four strategies and 21 years, no entry signal has been
shown to make money after costs, and the only statistically significant effect (London range
breakout's direction call) is about a hundredth of a risk unit per trade. The rules were
pre-registered, so a failure here is a result, not a reason to loosen them.

### Stage 4b: cross-sectional momentum and reversal, pre-registered (`PREREGISTRATION_XS.md`)

The last test of price-based ideas. Each Monday 07:00 UTC, rank sixteen currencies (each against
USD) by volatility-scaled past return, go long the top three and short the bottom three, and hold to
the end of the period. Two strategies, fixed before the run: momentum (84 days back, hold 4 weeks)
and reversal (7 days back, hold 1 week). Ranking skill is tested on gross R against a permutation
null that assigns the legs at random; costs are judged separately.

- **The controls passed** (random assignments 5.5% and 4.0% below p = 0.05; an oracle that knows the
  outcome is detected at p = 0.0001), so the pipeline would have found a real ranking.
- **Neither strategy is a candidate; all three rules fail for both.** Momentum: gross +0.004R per leg
  (null +0.003R, p = 0.47), net -0.039R, interval [-0.086, +0.011]. Reversal: gross -0.001R (p = 0.49),
  net -0.041R, interval [-0.060, -0.021]. Both lose about their cost (0.04R per leg) and nothing more.
- **A flaw in the permutation null was found after the run** and is disclosed in the Outcome: it draws
  each week independently, but real rankings persist, so its spread is 1.4 to 1.8 times too small
  against the strategy's own. No verdict changes (both p-values were near 0.5) but a marginal pass
  from this null would not have been trustworthy.
- **The frozen stop rule applies.** With sweep/FVG, three stage 4 strategies and these two all failing,
  the question of an edge in forex price data at intraday to monthly horizons is closed for this
  data. Nothing further on FX price is run without a new hypothesis or new data.
- Registry now: 13 configurations and 505 configuration-by-instrument tests (the two cross-sectional
  configurations count 16 each; their controls are not counted).

## Not built yet

Portfolio limits (a cap on simultaneous and correlated trades), tuning strategy parameters
inside the walk-forward (every strategy so far has fixed, pre-registered parameters), strategies on
information other than price (carry, macro; interest-rate history would be needed for carry), and
any order execution. With no strategy having survived validation, there is nothing yet worth
executing.
