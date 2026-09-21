# Stage 4 pre-registration

Written and committed before any of these strategies had been run on real data. Anything that
differs from this document in the results is a deviation and is reported as one.

## Why this exists

Stage 3 found that the sweep/FVG strategy's entry signal adds nothing over random entry, and that
spread cost dominates its result because its stops are tight (median cost 0.11R per trade, mean
0.39R). Every strategy, parameter and variant tried on the same 21 years is another look at the
same data, and enough looks will produce a "winner" by luck. So the strategies, their parameters,
the test conditions and the decision rules are fixed here first, and are not tuned afterwards.
The parameter values are conventional defaults, not fitted to this data.

## Common protocol

- **Universe:** the 65 forex pairs with dense H1, H4 and Daily history (`--start auto`); the three
  TRY pairs have none. Data as of 2026-09-21.
- **Exits and costs:** `--exit touch --costs spread`. Hard stop and target orders that act on
  wicks (a stop that gaps fills at the worse open; a candle touching both counts as a loss), and one
  bid/ask spread per trade taken at the entry candle. The strategies below have no separate
  invalidation level, so `plan` and `touch` coincide.
- **Entry timing:** signals are formed on completed candles only. Daily-timeframe signals are
  entered at the close of the first H1 candle that opens at 07:00 UTC after the daily candle has
  closed (London open, well clear of the New York rollover and of weekend gaps).
- **Null model:** `research.validate`, 1000 random-entry replays per signal (500 for
  `range_breakout`, which has far more signals), entries within 60 days of the real one, drawn
  from the candles the strategy could have entered on (the same hour rule). The **primary** null
  randomises direction as well as timing, so it asks "does this strategy beat coin-flip entries
  with the same stops, targets and costs?". The same-direction null (timing only) is a stricter
  secondary diagnostic. The same-direction null can absorb a trend-follower's edge, which is
  why it is not the primary.
- **Also reported:** per-pair p-values corrected by Benjamini-Hochberg and max-t, the walk-forward
  selection test (top 5 pairs, two-year blocks from 2011, at least 20 training trades), the gross /
  cost / net decomposition, and the detectable effect size.
- **Registry:** every run is logged; the number of configurations tried is stated with the results.

## Strategies (parameters fixed)

All four are direction-symmetric (long and short), one open decision at a time per signal, with
each signal resolved on its own like the earlier work.

**A. `donchian_trend`: daily breakout trend-following.** A daily close above the highest high of
the previous 20 daily candles (below the lowest low for a short), on the first close beyond the
channel (the previous day's close was inside its channel). Entry as above. Stop 2.0 x ATR(14 daily)
from entry; target 2.0R (4.0 x ATR) from entry.

**B. `range_breakout`: London breakout of the Asian range.** The high and low of the H1 candles
opening 00:00-06:59 UTC (at least 6 of the 7 present). The first H1 candle opening 07:00-10:59 UTC
that closes beyond the range is the entry, at that candle's close, one trade per day at most.
Stop at the opposite side of the range; target 2.0R. Skipped if the risk is under 0.25 x or over
1.5 x the ATR(14 daily).

**C. `daily_reversion`: pullback in the trend.** A daily close below the lower Bollinger Band
(20 days, 2.0 standard deviations) while above the 200-day average, on the first close beyond the
band, is a buy; the mirror image is a sell. Entry as above. Stop 2.0 x ATR(14 daily) from entry;
target the 20-day average at the signal, skipped if price has already reached it by the entry.

**D. `random_control`: the control.** On about one weekday in ten, a random candle opening
07:00-20:59 UTC, a random direction, stop 2.0 x ATR(14 daily), target 2.0R. It has no skill by
construction, so the pipeline must find nothing in it.

**E. `sweep_fvg_no_rollover`: the stage 3 hygiene fix.** The sweep/FVG strategy exactly as before,
except that signals whose entry candle opens at 21:00 UTC (it closes at the New York rollover)
are dropped, and the null's random entries avoid that hour too. Run under the stage 3
configuration (`plan` exit, spread costs) so it is comparable with the earlier result. This is
an execution fix suggested by looking at the data, not an independent hypothesis, so it cannot
be a pass, only a measurement of how much of the shortfall was rollover spread.

## Decision rules

A strategy (A, B or C) is a **candidate** only if all three hold on the primary configuration:

1. Pooled net expectancy after costs is positive and its 95% bootstrap interval excludes zero.
2. Pooled excess over the random-direction null is positive with p < 0.0167 (0.05 divided by the
   three strategies tested).
3. Pooled net expectancy is positive in both 2005-2015 and 2016-2026.

A candidate is evidence worth designing a paper-trading test around, not proof; it would still
need data the analysis has not seen. A strategy failing any of the three is rejected as it
stands. Nothing is tuned to rescue it: a variant would be a new look, registered and labelled as
one. Results are reported whether or not anything passes.

The pipeline itself is checked on the control (D): its pooled p-value must fall between 0.02
and 0.98, no more than 12% of its pairs may have a raw p-value below 0.05 (5% expected), and the
mean pair p-value must lie between 0.40 and 0.60. A control that fails means the null model is
biased on real data, and no other result in stage 4 would be trusted.

## Known limits

Pooled p-values treat trades as independent although overlapping trades on correlated pairs
are not, which is optimistic. The random draws are independent between pairs, which makes max-t
conservative. The 21 years cover a small number of independent market regimes. Daily strategies
trade rarely per pair (dozens per pair), so a single pair can only reveal a large edge; the pooled
test is the sensitive one.

---

# Outcome (added after the runs; everything above this line is unchanged)

Run on 2026-09-21 with the code and data as committed. Each run is in the registry
(`python -m research.registry`), and the rules were applied by `python -m research.preregistered`,
not by reading the numbers. After stage 4 the registry holds 19 runs, 11 distinct configurations
and 473 configuration-by-instrument tests.

## The control passed

The random-entry control (34,630 trades) earned -0.077R per trade and its random-direction null
-0.077R. Pooled p = 0.50; 4.6% of the 65 pairs had a raw p below 0.05 (5% expected); the mean pair
p was 0.52. All three control checks passed, so the null model is calibrated on real market data as
well as on synthetic random walks, and the other results can be taken at face value.

## No strategy is a candidate

| | trades | pooled net R | 95% interval | 1. net > 0, interval excludes 0 | excess over random-direction null | 2. p < 0.0167 | 3. positive in both halves |
|---|---|---|---|---|---|---|---|
| A `donchian_trend` | 30,235 | -0.113 | -0.130 to -0.096 | fail | -0.052R (z -6.4) | fail | fail (-0.108 / -0.119) |
| B `range_breakout` | 217,443 | -0.179 | -0.188 to -0.170 | fail | +0.012R (z +4.0), p = 0.002 | **pass** | fail (-0.245 / -0.119) |
| C `daily_reversion` | 6,487 | -0.044 | -0.073 to -0.016 | fail | +0.021R, p = 0.046 | fail | fail (-0.082 / -0.004) |

Rule 1 alone rejects all three, and it does not depend on any null model. Under the rules as
written nothing was tuned to rescue them.

- **A** loses 0.048R per trade before costs where random entries earn +0.004R: buying 20-day
  breakouts has done worse than a coin flip.
- **B** is the most instructive. Across 217,443 trades its direction call beats random entry with
  statistical significance, but by 0.012R per trade (0.009R gross), while spread costs 0.187R. A real
  effect that is economically worthless. That is why the decision rules require a positive net
  expectancy as well as significance.
- **C** is the closest to breakeven (+0.028R gross) but its excess over random misses the threshold
  and it is negative after costs in both halves.
- **Pairs:** no pair survives Benjamini-Hochberg or Holm for any strategy. One, `USD_CNH` under A
  (235 trades since 2014, +0.395R net, excess +0.32R, raw p = 0.001), survives max-t (0.021) but not
  Benjamini-Hochberg (0.065). One such result across four strategies and 65 pairs is not unusual by
  chance. It is a lead to watch, not a finding.

## E: the rollover fix, measured

Dropping sweep/FVG's 21:00 UTC entries removes 1,730 of 7,598 trades, cuts the average cost from
0.388R to 0.279R per trade and improves net from -0.426R to -0.310R. The "worse than random"
result disappears (excess -0.021R, p = 0.79). The strategy is still -0.31R net and no better than
random entry: a hygiene fix, not a rescue.

## Deviations and additions

- **The resolver was optimised mid-experiment.** `research/vector.py` was rewritten for speed
  (commit 5011510) after the control had finished and while `donchian_random` was running, so those
  two used the old code and every later run used the new. The outputs are bit-identical: the parity
  test against the scalar exits passes and 60,000 random trades on real EUR_USD data give exactly the
  same result under each exit policy.
- **Not pre-registered:** the `sweep_fvg` run against the random-direction null (it confirms stage 3:
  excess -0.089R against -0.088R with the same-direction null), and the cost-versus-pair analysis
  below. Both are additions, not decisions.
- **A caveat found after registering.** The same-direction null applies each signal's direction to
  random dates before the signal too, where that direction was only knowable in hindsight, so it is
  biased: it inflates the null for momentum signals (A's random same-direction entries "earn" +0.30R,
  a spurious finding) and deflates it for reversal signals (C's excess is +0.070R against +0.021R
  with the random-direction null). This is why the primary null randomises direction. The same-direction
  results are kept as diagnostics only.
- The machine went to sleep for several hours during the `rollover_same` run. That affects only
  wall-clock time.

## What persists across pairs is how cheap they are to trade (exploratory)

Across pairs, net expectancy correlates -0.71 (sweep/FVG), -0.89 (C), -0.91 (A), -0.96 (control) and
-0.98 (B) with the pair's average cost in R, and gross expectancy is uncorrelated with cost. So the
persistence found in stage 3, where choosing pairs on the past worked out of sample but only to
breakeven, is mostly a cost effect: the pairs that look good are the ones with the tightest spreads
relative to the stop. The random-entry control shows the same pattern.
