# Strategy Rules — Liquidity Sweep + FVG Reversal

Version: 1.6 (DST caveat in §5)
Scope: mechanical, deterministic rules only. Every condition below must be evaluable as a
pure function over OHLC(V) candle arrays and timestamps. No discretionary language.

**Revision log:**
- v1.1: Fixed a boundary-labeling error in §1.8 (confirmation level was described backwards
  relative to the range definition in §1.4). Added §1.9 clarifying that near-edge confirmation
  will often fire almost immediately after FVG formation (intended v1 behavior, to be checked
  in Phase 2 backtest, not assumed). Added §1.10 explicitly permitting the sweep candle to
  double as the FVG's origin candle. Pinned down window-counting semantics (§1.5) for the FVG
  formation window and confirmation window as plain chronological array-index distance, not
  session-time elapsed.
- v1.2: §2.3's second and third bullets previously duplicated Section 3's invalidation check,
  and did so inconsistently (one referenced an H1 close, the other an H4 close, against the
  same `sweep_extreme`). Collapsed into a single invalidation condition, with an explicit
  implementation note in Section 3 requiring one shared function used by both the
  pre-confirmation (§2.3) and post-confirmation (§3) code paths, so the two cannot drift apart
  under independent edits.
- v1.3: §5's session filter no longer applies to the FVG's three candles, only to the sweep
  and confirmation candles. See §5 for the full rationale and the Phase 2 backtest numbers that
  motivated this change.
- v1.4: Added §10, the news-filter overlay. It never changes a signal, bias, or level, so
  §§1-9 and every backtest result are unaffected.
- v1.5: Added §11, how the live trade plan (entry, stop, target) is derived and how it
  differs from the backtest's close-based scoring. Presentation only; §§1-10 unchanged.
- v1.6: Corrected §5's description of the H4 grid, which is fixed in UTC only in summer, and
  documented the resulting summer/winter difference in the H4 session filter. No rule or
  backtest result changed.

---

## 0. Timeframe roles (fixed)

| Timeframe | Role |
|---|---|
| Daily (D1) | Directional bias only. Never used for sweep/FVG/entry detection. |
| H4 | Sweep detection + FVG formation. |
| H1 | Confirmation close-through + entry trigger. |

A signal requires all three roles to agree in direction. There is no cross-substitution
(e.g. an H4 sweep can never be confirmed by a D1 close).

---

## 1. Definitions

### 1.1 Swing high / swing low (fractal, 2-candle)

On a given timeframe TF, candle at index `i` (0-indexed, chronological) is a:

- **Swing high** if `high[i] > high[i-1]` AND `high[i] > high[i-2]` AND `high[i] > high[i+1]`
  AND `high[i] > high[i+2]`.
- **Swing low** if `low[i] < low[i-1]` AND `low[i] < low[i-2]` AND `low[i] < low[i+1]` AND
  `low[i] < low[i+2]`.

A swing point only becomes confirmed/known once candles `i+1` and `i+2` have closed (i.e. it
cannot be identified in real time until 2 candles after it forms — this delay applies
identically in backtest and live).

Swing points are computed independently per timeframe (D1 swings for bias context only; H4
swings are the liquidity levels swept and targeted by this strategy).

### 1.2 Liquidity level

Any unmitigated H4 swing high or swing low (see 1.1) becomes a **liquidity level**. A level is
"unmitigated" until price trades through it (see 1.6 for consumption rule). All liquidity
levels considered by this strategy come from H4 swings, lookback-limited (see 2.4).

### 1.3 Liquidity sweep

A **sweep of a swing high** at price level `L` (H4) occurs on candle `i` (H4) if:
- `high[i] > L`, AND
- `close[i] < L`

(price pierces above the level intrabar but closes back below it — a bearish sweep, i.e. sell-side setup context).

A **sweep of a swing low** at price level `L` (H4) occurs on candle `i` (H4) if:
- `low[i] < L`, AND
- `close[i] > L`

(price pierces below the level intrabar but closes back above it — a bullish sweep, i.e. buy-side setup context).

The sweeping candle `i` must occur within the active session window (see Section 5). If the
candle's open time falls outside London/NY session hours, the sweep is disqualified and not
considered further.

### 1.4 Fair Value Gap (FVG)

On timeframe H4, a 3-candle sequence at indices `i-1, i, i+1` forms:

- **Bullish FVG** if `low[i+1] > high[i-1]`. The gap range is `(high[i-1], low[i+1])`.
- **Bearish FVG** if `high[i+1] < low[i-1]`. The gap range is `(high[i+1], low[i-1])`.

Minimum gap size: **none** (any non-zero gap, i.e. `low[i+1] - high[i-1] > 0` or
`low[i-1] - high[i+1] > 0`, qualifies). *Flagged for revision: if Phase 2 backtest shows
excessive low-quality signals, add a minimum size filter (fixed pips or a fraction of ATR(14)
on H4) here.*

An FVG is "in the opposite direction" of a sweep when: sweep of a swing high (bearish sweep
context) must be followed by a **bearish** FVG; sweep of a swing low (bullish sweep context)
must be followed by a **bullish** FVG.

### 1.5 FVG formation window

The FVG's middle candle (index `i` in 1.4) must occur on or after the sweep candle (1.3,
index `i`), and within **10 H4 candles** after the sweep candle, inclusive. If no qualifying
FVG forms in that window, the sweep setup expires and is discarded (no signal).

**Window counting rule (applies here and in 2.3):** "N candles" always means N consecutive
entries in the chronological H4 (or H1, for 2.3) candle array as pulled from the data source —
i.e. plain array-index distance (`j - i <= N`), counting every candle that exists in the data,
regardless of which session it falls in. Weekends and any period the market is closed
naturally produce no candles at all, so they never consume window budget. Candles that fall in
the Asia session (or the London/NY gap) still occupy array slots and count toward the window,
but cannot themselves be the disqualifying/qualifying sweep, FVG, or confirmation candle
(Section 5). This is a deliberate simplification: the window is a bar-count budget, not a
session-time budget, and must be implemented as simple index arithmetic, not calendar-aware
elapsed time.

### 1.6 Level mitigation / consumption

A liquidity level is consumed (removed from the active target pool) the first time any
candle's high/low trades through it after it was swept, per direction:
- A swing-high level is consumed once any subsequent candle's `high` exceeds it (regardless of close).
- A swing-low level is consumed once any subsequent candle's `low` goes below it.

Consumed levels are never reused as sweep sources or as targets for later signals.

### 1.7 Sweep extreme

The **sweep extreme** of a given sweep candle `i` (1.3) is:
- For a swing-high sweep: `high[i]` (the highest point reached during the sweep).
- For a swing-low sweep: `low[i]` (the lowest point reached during the sweep).

### 1.8 "Close back through a defined level" (confirmation level)

The **confirmation level** is the *near* boundary of the FVG — the edge closest to the price
action that preceded the gap (i.e. the edge price must cross first, coming from the sweep
side), not the far edge of the gap. Per the range definitions in 1.4:

- For a bullish FVG (buy-side setup), the gap range is `(high[i-1], low[i+1])` with
  `high[i-1]` as the **lower/bottom** boundary and `low[i+1]` as the **upper/top** boundary.
  The near boundary (closest to the sweep, which occurred below) is the **bottom**:
  confirmation level = `high[i-1]`.
- For a bearish FVG (sell-side setup), the gap range is `(high[i+1], low[i-1])` with
  `low[i-1]` as the **upper/top** boundary and `high[i+1]` as the **lower/bottom** boundary.
  The near boundary (closest to the sweep, which occurred above) is the **top**:
  confirmation level = `low[i-1]`.

(Note: this is the *near/shallow* edge of the gap, not the far edge and not the 50%
"consequent encroachment" midpoint. A close through the near edge is a materially easier
condition to satisfy than requiring a close through the far edge — see 1.9 below for why this
matters in practice.)

Confirmation is evaluated on **H1** candles only (Section 0). Given a bullish setup, the
setup is confirmed on the first H1 candle `j` (occurring at or after the FVG's formation on
H4) where `close[j] > confirmation_level`. Given a bearish setup, confirmed on the first H1
candle `j` where `close[j] < confirmation_level`.

The confirming H1 candle must open within the active session window (Section 5).

### 1.9 Confirmation timing (expected behavior, not a bug)

Because the near-edge confirmation level (1.8) is, by construction of the FVG itself (1.4),
already on the favorable side of price by the time the gap exists, the confirmation condition
will often already be satisfied by the first (or one of the first) H1 candle checked after the
FVG forms. In practice this means v1 of this strategy behaves closer to **"entry on FVG
completion"** than "entry after a pullback/retest into the gap." This is the intended v1
behavior (simplest, fewest free parameters) and not an error — but it should be explicitly
checked in the Phase 2 backtest output (e.g. distribution of H1 candles elapsed between FVG
formation and confirmation) rather than assumed. If backtest results show this produces poor
entries (e.g. immediate reversal against the position), a future revision may require price to
first retrace to the far edge or the 50% midpoint before confirming — that is an explicit
candidate for Section 9.

### 1.10 Sweep candle may double as the FVG's origin candle

The sweep candle (1.3) is **permitted** to also serve as the FVG's `i-1` or `i` candle in
1.4 — there is no rule forbidding overlap between the sweep candle and the FVG's three
candles. This is intentional: in this pattern the sweep (reversal) candle is frequently the
same candle that begins the imbalance, since the sharp reversal off the sweep is often what
creates the gap. No special-casing or exclusion is needed in the implementation.

---

## 2. Setup conditions

### 2.1 Bullish setup (long)

A bullish setup is created when, in order:
1. An H4 candle sweeps a swing low (1.3, bullish sweep) during an active session (Section 5).
2. Within the next 10 H4 candles (1.5), a **bullish FVG** (1.4) forms on H4.
3. The Daily bias (Section 4) is not bearish (i.e. Daily bias is bullish or neutral).

If all three hold, the setup enters "pending confirmation" state, tracking:
- `sweep_extreme` = 1.7 value from the sweep candle
- `confirmation_level` = 1.8 value from the FVG
- `fvg_range` = (low, high) of the FVG

### 2.2 Bearish setup (short) — mirror of 2.1

A bearish setup is created when, in order:
1. An H4 candle sweeps a swing high (1.3, bearish sweep) during an active session (Section 5).
2. Within the next 10 H4 candles (1.5), a **bearish FVG** (1.4) forms on H4.
3. The Daily bias (Section 4) is not bullish (i.e. Daily bias is bearish or neutral).

If all three hold, the setup enters "pending confirmation" state with the same tracked fields
as 2.1 (mirrored).

### 2.3 Setup expiry (pre-confirmation)

A pending setup is discarded (no signal ever generated) if either of the following occurs
before confirmation (1.8):
- No H1 close satisfies the confirmation condition within **20 H1 candles** after the FVG
  formed (H4 candle `i+1` close time). Window counting follows the rule in 1.5 (plain
  array-index distance over the H1 candle array, not session-time elapsed).
- The invalidation condition defined in Section 3 evaluates true (H1 close beyond
  `sweep_extreme`, in the invalidating direction) — applied identically here,
  pre-confirmation, as it is post-confirmation. There is exactly one invalidation check in
  this strategy, not two: see the implementation note at the end of Section 3.

### 2.4 Liquidity/lookback scope

Only H4 swing highs/lows (1.1) formed within the most recent **90 H4 candles** (~15 trading
days) before the sweep candle are eligible to be swept (2.1/2.2 step 1) or used as targets
(Section 6). Older, unmitigated levels outside this window are ignored by the strategy (they
may still exist on the chart but are not traded).

---

## 3. Invalidation

A **confirmed** setup (post 1.8) is invalidated, and the trade is closed at market/stop, if:

- **Bullish setup**: any H1 candle **closes** below `sweep_extreme` (the low of the sweep
  candle, 1.7). Condition: `close[j] < sweep_extreme`.
- **Bearish setup**: any H1 candle **closes** above `sweep_extreme` (the high of the sweep
  candle, 1.7). Condition: `close[j] > sweep_extreme`.

This is a close-based condition on H1 only — intrabar wicks through the sweep extreme do not
invalidate by themselves. Once invalidated, the setup is closed and produces no further
signals; it cannot be resurrected even if price later recrosses.

Stop-loss placement (for R:R bookkeeping, Section 6): stop is placed at `sweep_extreme` plus
a fixed buffer of **5 pips** (for JPY pairs: 0.05 in price terms; for all other pairs in this
strategy's scope: 0.0005 in price terms) beyond the extreme, in the invalidating direction.

**Implementation note (single source of truth):** this section's condition
(`close[j] < sweep_extreme` for bullish, `close[j] > sweep_extreme` for bearish, evaluated
per H1 candle) is used in exactly two places in the strategy — here (post-confirmation) and
in §2.3 (pre-confirmation setup expiry). These must be implemented as **one shared function**
(e.g. `is_invalidated(direction, sweep_extreme, h1_candle)`), called from both the
pre-confirmation and post-confirmation code paths, not duplicated. The only difference between
the two call sites is what happens *after* the check returns true (silently discard a pending
setup in §2.3, vs. close an open/live trade here) — the check itself must never be edited in
one place without the other.

---

## 4. Daily bias

Daily bias is computed once per Daily candle close, using the most recently closed Daily
candle, as:

- **Bullish** if `close[D] > close[D-1]` AND `close[D-1] > close[D-2]` (last two closed Daily
  candles both closed higher than the one before).
- **Bearish** if `close[D] < close[D-1]` AND `close[D-1] < close[D-2]`.
- **Neutral** otherwise.

Daily bias is a filter only (2.1/2.2 step 3) — it never generates a signal or overrides the
H4/H1 sweep-FVG-confirmation sequence.

---

## 5. Session filter

A candle qualifies as "in an active session" if its open timestamp (UTC) falls within either:
- **London**: 07:00–16:00 UTC, or
- **New York**: 12:00–21:00 UTC

(Both bounds inclusive of the hour: a candle opening exactly at 16:00 or 21:00 counts. In summer,
H4 candles land on a 01/05/09/13/17/21:00 UTC grid, so the 21:00 candle sits exactly on NY's
stated end — reading that boundary as exclusive would silently drop every day's 21:00 H4 candle
and would also break this doc's own §8.1 worked example, whose FVG's third candle lands there.)

**DST caveat (found by the data-quality grid check, v1.6):** OANDA aligns its Daily and H4 candles
to 17:00 New York time, not to a fixed UTC hour. In winter (US standard time) the H4 grid is
therefore 02/06/10/14/18/22:00 UTC and the Daily candle opens at 22:00 UTC rather than 21:00. The
session windows above are fixed in UTC, so the H4 session filter admits four candles a day in
summer (09, 13, 17, 21) but only three in winter (10, 14, 18): the candle opening at 17:00 New York
qualifies in summer and not in winter. H1 candles sit on the whole hour in both seasons and are
unaffected. This is a real inconsistency in the rule as specified, left unchanged so that every
backtest result stays comparable; a DST-aware session definition would be a strategy change to
test on its own.

This applies to: the **sweep candle** (1.3) and the **confirmation candle** (1.8) — each must
independently open within an active session, or that step is disqualified (sweep discarded, or
confirmation not yet satisfied and the setup continues waiting within its expiry window per
2.3).

**v1.3 revision:** earlier versions of this section also required all three of the FVG's
candles (1.4) to be individually session-qualified. Phase 2 backtesting showed this was overly
strict: on the fixed H4 session grid, only 2 of every 6 possible 3-candle windows per day have
all three candles session-qualified, and requiring it was responsible for roughly half of all
"no qualifying FVG" outcomes — killing real, correctly-directioned gaps for no reason tied to
the strategy's actual logic (the FVG is a structural fact about price, not an action being
taken, unlike the sweep and confirmation, which are the two decision points this filter is
meant to gate). The FVG candles' session status is no longer checked. Backtest impact of this
change: EUR_USD flipped from -0.39R to +0.39R expectancy (11 -> 21 signals, 18.2% -> 42.9% win
rate); GBP_USD improved from -0.27R to -0.18R but remained net negative (8 -> 18 signals, win
rate 25.0% -> 22.2%). Sample sizes (18-21 trades/pair) are still small enough that this is a
plausible fix, not a proven one — flagged for continued monitoring as more data/pairs are
added, not treated as fully validated.

---

## 6. Target and R:R bookkeeping

### 6.1 Target selection

The **target** is the nearest unmitigated (1.6), in-scope (2.4) H4 liquidity level (1.2) on
the opposite side of the trade direction, relative to the entry price:

- **Bullish setup**: nearest unmitigated H4 swing **high** above the entry price.
- **Bearish setup**: nearest unmitigated H4 swing **low** below the entry price.

If no such level exists within scope (2.4) at confirmation time, the setup is **not traded**
(no signal is generated) — target must exist and be identifiable at confirmation.

### 6.2 Entry price

Entry price = `close[j]`, the closing price of the confirming H1 candle (1.8).

### 6.3 Stop price

Stop price = `sweep_extreme` ± 5-pip buffer as defined in Section 3, in the direction away
from the trade (below entry for bullish, above entry for bearish).

### 6.4 R and R:R

- Risk in price terms: `R_price = abs(entry_price - stop_price)`.
- Reward in price terms: `Reward_price = abs(target_price - entry_price)`.
- R-multiple of the target: `Reward_price / R_price`.
- Minimum acceptable setup: `Reward_price / R_price >= 1.5`. If the computed R:R at
  confirmation is below 1.5, the setup is **not traded** (no signal generated), even though
  all prior conditions were met.

### 6.5 Trade outcome (for backtest scoring)

Once a signal is live (confirmed, R:R >= 1.5, target identified):
- **Win**: subsequent H1 close reaches/exceeds the target price before the invalidation
  condition (Section 3) triggers. Realized R = `Reward_price / R_price` (the planned R:R).
- **Loss**: invalidation condition (Section 3) triggers before target is reached. Realized
  R = -1 (full stop hit, since stop = sweep_extreme ± buffer by construction).
- **Open/unresolved**: neither has occurred by the end of available data (backtest) — excluded
  from win-rate calculation, reported separately as "open positions at end of test window."

No partial-target or breakeven-stop rules are defined in this version — a setup is fully win
or fully loss per 6.5. (Flagged for future revision post-Phase-2 if max-drawdown/variance is
too high.)

---

## 7. Explicit exclusions

The following are explicitly **not** part of this strategy and must not be implemented as
discretionary judgment calls:

- No "strength" or "quality" scoring of sweeps, FVGs, or structure based on visual impression.
- No subjective trendline, channel, or pattern (head-and-shoulders, wedge, etc.) analysis.
- No news/fundamental input into signal generation (news is a suppression filter only — see
  Phase 5 of the overall project, not part of this rules doc).
- No signals generated from sweeps/FVGs/confirmations occurring outside the London/NY session
  windows (Section 5) — Asia-session structure is ignored entirely by this version.
- No re-entry or scaling into a setup after invalidation (Section 3) — once invalidated, that
  swing/FVG combination is permanently dead for this strategy instance.
- No use of D1 or H1 timeframe for sweep/FVG detection, and no use of H4 for confirmation —
  timeframe roles (Section 0) are fixed, not adaptive.

---

## 8. Worked examples

### 8.1 Bullish example (signal generated)

Assume EUR/USD, all times UTC, H4 candles:

| H4 idx | Time | Open time in session? | High | Low | Close |
|---|---|---|---|---|---|
| 10 | 05:00 (prior day) | — | 1.0850 | 1.0810 | 1.0830 | (forms part of swing low context)
| 11 | 09:00 | ✔ London | 1.0840 | **1.0795** (sweep low, was prior swing low at 1.0800) | 1.0805 |
| 12 | 13:00 | ✔ NY overlap | 1.0812 | 1.0803 | 1.0808 |
| 13 | 17:00 | ✔ NY | 1.0850 | 1.0806 | 1.0845 |

- Prior confirmed swing low = 1.0800 (from earlier fractal, candles before idx 11).
- Candle 11: `low = 1.0795 < 1.0800` and `close = 1.0805 > 1.0800` → **bullish sweep** (1.3),
  occurs at 09:00 UTC → within London session → qualifies.
- Candles 12,13,14 (idx 12-14): suppose `low[14] > high[12]` → **bullish FVG** forms with
  gap range `(high[12]=1.0812, low[14])`. This is within 10 candles of the sweep (1.5) → setup
  created (2.1). Daily bias (Section 4) assumed bullish → condition 3 of 2.1 satisfied.
- Confirmation level = `high[12]` = 1.0812 (1.8).
- On H1, first candle closing above 1.0812 within 20 H1 candles, during London/NY session →
  confirmed. Say entry `close[j] = 1.0815`.
- Sweep extreme = `low[11]` = 1.0795. Stop = `1.0795 - 0.0005 = 1.0790`.
- R_price = `1.0815 - 1.0790 = 0.0025` (25 pips).
- Target: nearest unmitigated H4 swing high above 1.0815 within 90-candle lookback — say
  1.0900. Reward_price = `1.0900 - 1.0815 = 0.0085` (85 pips).
- R:R = `0.0085 / 0.0025 = 3.4` → ≥ 1.5 → **signal generated**: bullish, entry 1.0815, stop
  1.0790, target 1.0900, planned R:R 3.4.
- Outcome: if subsequent H1 closes reach 1.0900 before any H1 close < 1.0790 → **win**,
  realized R = +3.4. If an H1 close < 1.0790 occurs first → **loss**, realized R = -1.

### 8.2 Bearish example — invalidated, no signal

- H4 candle sweeps a swing high at 1.2500 (`high > 1.2500`, `close < 1.2500`) at 14:00 UTC
  (NY session) → bearish sweep qualifies.
- Within the next 10 H4 candles, no 3-candle sequence satisfies `high[i+1] < low[i-1])` — no
  bearish FVG forms → **setup expires at candle 10 per Section 1.5, no signal generated.**

This demonstrates the rule producing a clean no-signal outcome without requiring judgment.

---

## 9. Open parameters flagged for post-backtest revision

These were set to permissive/simple defaults per Phase 0 to avoid over-fitting before any
data exists. Revisit after Phase 2 backtest results:

- FVG minimum size (currently: none).
- FVG formation window (currently: 10 H4 candles).
- Confirmation window (currently: 20 H1 candles).
- Minimum R:R threshold (currently: 1.5).
- Liquidity lookback window (currently: 90 H4 candles).
- Stop buffer (currently: fixed 5 pips).

---

## 10. News filter (overlay, not a signal input)

A risk overlay applied at reporting time. It never generates, cancels, or modifies a setup,
bias, level, or backtest result; it only attaches a `news` status to the live state.

- **Source:** Forex Factory's unofficial weekly JSON feed (current calendar week only).
- **Relevant events:** impact `High`, and currency equal to either side of the pair (e.g. EUR
  or USD for EUR_USD), or `All`.
- **Blackout:** status is `blackout` while `now` is within 60 minutes before through 60 minutes
  after any relevant event (both edges inclusive). Clustered events merge into one blackout.
- **Upcoming:** relevant events after `now`, outside blackout, within 72 hours; at most 5.
- **Unavailable, not clear:** if the feed can't be fetched (and no cache exists), has no
  usable events, its date range is more than 1 day away from `now` (the feed hasn't rolled to
  the new week), or the calendar was last fetched more than 12 hours ago (refreshes are
  failing and a stale cache is being served), the status is `unavailable`. The filter never
  reports `clear` when it cannot see the current time, and an unexpected error inside the
  filter also yields `unavailable` rather than failing the report.
- **Alerts:** a blackout in progress defers a live-trade alert until it has passed. A trade
  whose confirmation candle itself fell inside a blackout window is not pushed. An
  `unavailable` status does not suppress alerts; the report says the news check is unavailable.
- **Not validated:** the feed has no historical data, so these windows are judgment defaults
  and cannot be backtested. Do not treat the filter as improving the strategy's edge.
- **Known gap:** near the weekend rollover the upcoming list can be empty even though the
  new week has events, because the feed hasn't published them yet.

---

## 11. Trade plan (presentation overlay, not a signal input)

Built by `live/plan.py` for every active setup and shown above the report. It is arithmetic
over the §3 and §6 values, computed by the engine and never by the narration model. It never
generates, cancels or modifies a setup, bias, level, or backtest result.

- **`live_trade`:** entry is the §6.2 signal price (the confirming H1 close), stop is §6.3,
  target is §6.1, R:R is §6.4.
- **Do-not-chase limit:** the entry at which R:R equals the §6.4 minimum for the same stop and
  target, `(target + 1.5 × stop) / 2.5`. A fill beyond it (higher for a buy, lower for a sell)
  is under the minimum, so a trader who cannot fill at or better than it skips the trade. This
  applies §6.4 to the actual fill rather than the signal price; it is not a backtested rule.
- **`pending_confirmation`:** the stop is §6.3 (it depends only on the sweep). The entry shown
  is the trigger level, the best case, since the confirming close can only be further along.
  The target is §6.1 evaluated as if the setup confirmed now at the trigger level. It is
  labelled provisional because the real target is chosen at the confirming candle's close.
- **`pending_fvg`:** no entry or target exists, so only the §6.3 stop and the §3 cancel level
  are shown.
- **Execution versus the backtest:** §6.5 scores a win or loss on H1 **closes**. A broker stop
  or limit order fires on any touch, so hard SL/TP orders will act on wicks the backtest
  ignores, and a demo account's results will differ from the backtest's. The §3 invalidation
  is close-based and can trigger before the stop (`sweep_extreme` ± 5 pips) is touched, so the
  plan tells the trader to close by hand on that close as well.
