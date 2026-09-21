# Cross-sectional strategies: pre-registration (stage 4b)

**Status: DRAFT, awaiting approval.** It becomes binding when the status line is changed to FROZEN in
a commit, and no strategy code will be written and no result looked at before that. Changes made
before then are visible in the git history; changes after are amendments, listed at the end.

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
