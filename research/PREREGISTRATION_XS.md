# Cross-sectional strategies: pre-registration (stage 4b)

**Status: FROZEN on 2026-09-21, approved as drafted.** No strategy code had been written and no
result looked at when this line was changed. From here on any change is an amendment, listed in an
Outcome section at the end, and any departure in the results is reported as a deviation.

## The question

Stage 4 found that four price-pattern strategies on intraday and daily horizons lose after costs,
mostly because spread is large next to the stop (median cost 0.02R for the daily ATR-scaled
strategies, 0.08R for the range breakout, 0.11R for sweep/FVG). At weekly-to-monthly holding periods the spread is small next to
the move. If price data contains an exploitable edge in forex at all, a slow cross-sectional
ranking of currencies is the place it can show up without being eaten by costs. This tests two
standard versions, each once, with no variations.

**Expectation, stated in advance:** a null. FX momentum has weakened since about 2010, and stage 4's
Donchian result (20-day breakouts did *worse* than random entries) points against momentum at
these horizons. My prior is roughly a 10-15% chance that either strategy is a candidate. A
significantly negative result for momentum would be reported as such, not flipped (see below).

## The universe

Each trade is a **leg**: long or short one currency against the US dollar through its USD pair (long
`EUR_USD` is long EUR; short `USD_JPY` is long JPY). A currency's return is the log change in its
value per dollar. Sixteen currencies: AUD, EUR, GBP, NZD, CAD, CHF, JPY, SGD, SEK, NOK, PLN, CZK, HUF,
MXN, THB, ZAR.

Excluded before looking at any return: **HKD and DKK** (hard pegs, with no independent price
discovery by design), **CNH** (managed, and dense history only from 2014) and **TRY** (no dense
history). This is a limitation in itself: TRY's collapses are where much of FX momentum's premium
has come from, so leaving it out biases the test against momentum.

A currency is eligible at a cohort date if its USD pair has dense H1 and Daily history then (the
`--start auto` rule) and at least the formation window of Daily history. A cohort with fewer than
8 eligible currencies is skipped. Eligibility counts by year will be reported.

## Common protocol

- **Cohorts:** every Monday. Entry is at the close of the H1 candle that opens at 07:00 UTC (London
  open, clear of the New York rollover and the weekend gap; the same rule as the stage 4 daily
  strategies), skipped if that candle does not exist. The ranking uses only Daily candles that had
  closed before that candle opened.
- **Ranking score:** the formation-window log return of the currency, divided by its 14-day ATR as
  a share of price (how many typical days it moved). Scaling by volatility stops the ranking simply
  preferring the most volatile currencies.
- **Selection:** the 3 highest and 3 lowest scores among the eligible currencies, three legs long
  and three short.
- **Exit:** at the close of the first H1 candle opening at or after 07:00 UTC on the Monday `H`
  weeks after entry, found within 3 days (a cohort with no such candle is skipped, for real and null
  alike). A disaster stop 5 x ATR(14 daily) from entry acts on wicks, a gap filling at the worse open.
  No target.
- **Unit of account:** R = signed price change divided by 2 x ATR(14 daily) at entry, so legs are
  volatility-scaled and comparable across currencies. It is a unit, not a stop; the disaster stop sits
  at -2.5R.
- **Costs:** one bid/ask spread per leg, taken at the entry candle, divided by 2 x ATR: the same
  model as stages 2-4. **Swap and financing are not charged** (see the limits).
- **Data:** the stored candles as of 2026-09-21, 2005-2026.

## The strategies (parameters fixed; no other values will be run)

**X1 `xs_momentum`.** Formation 12 weeks (84 days), hold 4 weeks. Long the 3 strongest, short the 3
weakest. A new cohort each Monday, so holds overlap.

**X2 `xs_reversal`.** Formation 1 week (7 days), hold 1 week. Long the 3 *weakest*, short the 3
strongest. Cohorts do not overlap.

The parameters are the conventional textbook horizons for each effect, and 3 of 16 is a round choice
for the extremes. They are not fitted to this data.

**Checks, which must pass or the results are not trusted:**

- **X0 `xs_random`:** the same cohorts with the long and short legs assigned at random, drawn 200
  times, each tested against the permutation null below. At most 9% of the 200 p-values may fall
  below 0.05 (5% expected) and their mean must lie between 0.44 and 0.56. This is an implementation
  check: it verifies the alignment and plumbing on real data, but by construction it cannot test the
  null's assumptions.
- **X9 `xs_oracle`:** ranks by the *forward* return over the hold, information no one has. It must
  be detected: mean gross R positive and permutation p < 0.001. This is the power check, showing the
  machinery can see an edge when one exists.
- **A no-look-ahead test** on the strategy code before any run: cohorts computed on data truncated at
  a moment must equal the full run's cohorts up to that moment.

## Inference

- **Primary null: cross-sectional permutation.** Same cohort dates, same eligible currencies, same
  holds, but each cohort's 3 long and 3 short legs are drawn at random from the eligible currencies.
  Leg results (both directions) are computed once, so 10,000 replicates are cheap. This keeps the
  clustering by week, the volatility regime and the common dollar factor, and destroys only the
  ranking information. That is the point of the test.
- **Test statistic for the ranking's skill: mean GROSS R per leg** (before spread). Volatility-scaled
  ranking and equal-sized legs still leave the strategy and the null holding different currencies,
  with different spreads in R, so testing net R would credit or blame the ranking for cost
  differences that are not skill. Costs are judged separately, in rule 1.
- **Interval for net R:** a moving-block bootstrap over cohort weeks (blocks of 4 weeks, which covers
  X1's overlapping holds), 10,000 resamples, 95% percentile interval. The stage 3 and 4 bootstrap
  treated trades as independent, which legs of one cohort are not.
- **Halves:** by cohort entry date, 2005-2015 and 2016-2026.
- **Reported alongside:** the null's spread as a minimum detectable effect, the gross / cost / net
  split, long legs against short legs, and a per-currency table (Benjamini-Hochberg corrected,
  exploratory only).

## Decision rules

A strategy is a **candidate** only if all three hold:

1. Mean **net** R per leg is positive and its 95% block-bootstrap interval excludes zero.
2. The **gross** permutation p-value is below 0.025 (0.05 divided by the two strategies).
3. Mean net R is positive in both 2005-2015 and 2016-2026.

Both strategies are reported whatever happens. Nothing is tuned to rescue a failure: a different
horizon, k, filter, universe or sign is a new look, and needs its own pre-registration. In
particular, if X1 is significantly *worse* than random (as Donchian was), that is reported and
momentum is not flipped into a reversal strategy after the fact.

**Stop rule.** If neither is a candidate, the question "is there an edge in forex price data at
intraday, daily and weekly-to-monthly horizons" is closed for this data, and no further FX price
strategy is run without a genuinely new hypothesis or new data. If one is a candidate, the next step
is a re-test that charges swap using historical interest rates, then a forward paper test on data
not yet seen. It is not execution.

## Before any run

A universe-level strategy interface (cohorts depend on all currencies at once), a time-based exit,
the leg tables, the permutation null, the block bootstrap and the tests above are written, tested and
committed first. Every run is logged in the registry, and any departure from this document is listed
in an Outcome section, as in stage 4.

## Known limits

- **Swap is not charged, and it is not small here.** A 4-week leg on a pair with a 2.5% annual rate
  gap pays about 0.19% of notional, roughly 0.15R at a 2 x ATR unit of about 1.3% of price. A
  candidate could be erased by it, which is why a candidate needs the swap-inclusive re-test.
- Overlapping cohorts and cross-currency correlation make the effective sample much smaller than the
  leg count; the block bootstrap and the permutation null are designed for that, the pooled intervals
  in earlier stages were not.
- Sixteen currencies, several of them managed, and TRY left out. Twenty-one years cover few
  independent regimes.
- Momentum and carry are correlated in FX; this tests price momentum only and cannot separate it from
  carry, whose data we lack.

## Outcome

Run once on 2026-09-21 with the code as committed (9e2bee8, clean tree) and the defaults of this
document: 16 of 16 currencies, cohorts from 2005-01-03 to 2026-08-17 (momentum) and 2026-09-07
(reversal), 10,000 permutation replicates, 10,000 bootstrap resamples, 200 random assignments, seed 0.
Nothing was rerun or adjusted after the result was seen. Both strategies and their controls are in
the registry; the outputs are in `data/runs/xs_frozen` (not committed).

**The controls passed for both strategies**, on each strategy's own cohorts. X0: 5.5% (momentum) and
4.0% (reversal) of random assignments fell below p = 0.05, against a limit of 9%, with mean p 0.518
and 0.484 (band 0.44 to 0.56). X9: the oracle was detected, gross +1.28R and +0.70R per leg, p = 0.0001
(the floor with 10,000 replicates). The pipeline is calibrated, and a ranking that carried information
would have shown up.

**Neither strategy is a candidate. All three rules fail for both.**

| | X1 momentum (84 days, 4 weeks) | X2 reversal (7 days, 1 week) |
|---|---|---|
| Legs (cohorts) | 6,726 (1,121) | 6,744 (1,124) |
| Gross R per leg | +0.0037 | -0.0005 |
| Null gross R (sd) | +0.0029 (0.0136) | -0.0008 (0.0072) |
| Rule 2: gross permutation p (needs < 0.025) | 0.472 | 0.486 |
| Cost per leg | 0.0426R | 0.0404R |
| Net R per leg | -0.0388 | -0.0409 |
| Rule 1: 95% block-bootstrap interval on net | [-0.086, +0.011] | [-0.060, -0.021] |
| Rule 3: net, 2005-2015 / 2016-2026 | +0.005 / -0.085 | -0.036 / -0.046 |
| Gross excess the test would have detected | about 0.022R | about 0.012R |

Reading: neither ranking carries measurable information about the next period's currency returns.
Reversal earns nothing before costs and loses its costs; its interval sits wholly below zero.
Momentum's net loss is also its costs, with an interval that includes zero. Its halves differ
(+0.005, then -0.085), but the interval is about 0.1R wide, so that is not a finding. Not
interpreted, and not acted on as the rules above require: momentum's long legs (gross -0.035R)
against its short legs (+0.042R), and the per-currency tables. THB's reversal cost is 0.28R per leg,
the width of that pair's spread against a one-week ATR unit.

**The stop rule applies.** Neither strategy is a candidate, so the question of an edge in forex
price data at intraday, daily and weekly-to-monthly horizons is closed for this data: sweep/FVG,
the three stage 4 strategies and these two all fail. No further FX price strategy is run without a
genuinely new hypothesis or new data. Carry is the obvious untested idea, but it is not price data
and we hold no history of interest rates.

### Deviations, clarifications, and a flaw found after the run

- **Missing exits.** This document did not say what happens to a leg whose exit cannot be found. The
  code drops that leg (per currency and cohort) identically for the strategy and the null, and skips
  a cohort with fewer than 8 eligible currencies. It made no difference: nothing was dropped, and
  every cohort has exactly six legs (6 x 1,121 = 6,726; 6 x 1,124 = 6,744).
- **Controls on each strategy's own cohorts.** X0 and X9 were run on the calendar of the strategy they
  vouch for, since the two calendars differ by a few cohorts and a control on another calendar would
  not prove anything about this one.
- **The per-currency table is not corrected.** This document promised Benjamini-Hochberg correction;
  the table prints legs, gross and net with no p-values, so there is nothing to correct and it is
  description only.
- **Code written after the freeze.** All of `research/xs/` was written after this document was
  frozen, as it said it would be, and nothing changed between commit 9e2bee8 and the run.
- **The permutation null understates the strategy's own sampling noise.** The null draws each
  cohort's assignment independently, but real rankings persist from week to week (the same
  currencies stay extreme while a trend lasts, and holds overlap) and currency returns are serially
  dependent. A diagnostic added after the run, not a decision (moving-block bootstrap of the
  saved legs, blocks of 4 cohorts, 4,000 resamples), puts the sampling sd of the strategy's mean gross
  R at 0.0250 (momentum) and 0.0099 (reversal), against null sds of 0.0136 and 0.0072: 1.8 and 1.4
  times larger. The permutation p-values are therefore too small for persistent rankings. This
  changes no verdict, because both were near 0.5 and the correction can only raise them. It changes
  what the test could see: scaled by these ratios, momentum's detectable gross excess is about 0.04R,
  roughly equal to its cost, so momentum was only just within reach of an edge that paid for itself
  (reversal, about 0.016R against a 0.040R cost, comfortably was). It also means a p-value just under
  0.025 from this design would not have been trustworthy, and X0 could not have revealed that,
  because its random assignments are drawn the same independent way as the null. A future ranking
  test needs a null that keeps the persistence, such as permuting whole runs of weeks, or the
  block-bootstrap spread as its yardstick.
