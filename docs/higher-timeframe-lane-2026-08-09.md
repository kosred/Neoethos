# The higher-timeframe lane, re-measured — 2026-08-09

**THE STANDING BELIEF IS OVERTURNED FOR H1 AND NOT FOR H4. H1 carries rankable
signal and the prefilter was throwing all of it away — 0 columns earned on rank
before, 79 after, 106 of them clearing a Bonferroni bar on their own effective
sample size. H4 does not clear that bar on a single column, for a measured and
arithmetic reason that is not the old reason.**

The number this document replaces — *"base keeps 217/217, H1 keeps 40/217, H4
keeps 8/217, and none of the 8 earned their place"* — is **VOID**. It was not a
weak measurement. It was a measurement of column index. Nothing may cite it
again.

Re-run the measurement yourself before citing anything below:

```text
cargo test -p neoethos-search --release --test higher_timeframe_lane_measured \
    -- --ignored --nocapture
```

Source: `crates/neoethos-search/tests/higher_timeframe_lane_measured.rs`.
It is `#[ignore]`d and fails hard, naming the resolved store path, when the real
EURUSD M5/H1/H4 vortex store is absent — it never passes green without data.

---

## 1. What the old measurement said

That the prefilter, offered a multi-timeframe feature cube, kept every base
column, kept a minority of H1 columns, kept almost no H4 columns, and that the
handful of H4 survivors were all admitted by the per-timeframe quota or the
seed-template force-keep rather than by their own score. The natural reading —
and the reading acted on — was **higher timeframes carry no signal for a
5-minute base, so searching them is waste.**

## 2. Why it was wrong

Two independent facts met.

1. **`pearson_correlation` returned exactly `0.0` for any column containing a
   `NaN`.** The pre-repair statistic (f32, single pass) computed
   `den = sqrt((n·Sx² − Sx²)·(n·Sy² − Sy²))` and had the guard
   `if den == 0.0 || !den.is_finite() { 0.0 }`. One non-finite value anywhere in
   the column makes `Sx` `NaN`, makes `den` `NaN`, fires the guard, and reports
   the column as *uncorrelated* — indistinguishable from a genuinely useless
   feature.

2. **Every higher-timeframe column carries `NaN` by construction.**
   `core::features::align_features_by_ns` allocates the aligned array full of
   `f32::NAN` and only overwrites rows whose higher-timeframe bar has closed. The
   leading rows of every H1/H4/D1 column are therefore `NaN`, and the prefilter's
   in-sample slice starts at row 0. Normalisation, which would have converted the
   `NaN` to `0.0` upstream, is **off** in the operator's installed config.

So every higher-timeframe column scored exactly `0.0`, the stable sort broke the
resulting mass tie by original column index, and base columns — emitted first in
the cube — swept the whole top-K. **The prefilter was not ranking the higher
timeframes badly. It was not ranking them at all.**

### The reproduction, and it is worse than the report said

Measured under `legacy_prefix` (old f32 correlation, 1-bar forward return, 80%
leading prefix, `normalize_features=false` — the void number's exact regime):

| lane | columns scoring exactly 0.0 |
|---|---|
| H1 | **1,946 of 1,946 (100%)** |
| H4 | **1,946 of 1,946 (100%)** |

Under `legacy_cpcv` — the same old correlation but with today's triple-barrier
label and today's CPCV folds — **5,824 of 5,838 columns scored exactly 0.0,
including 1,932 of 1,946 BASE columns.** The proof that the ranking was column
index and not correlation is the median rank per timeframe:

| lane | median global rank | cube index range |
|---|---|---|
| base | 973 | 0 – 1,945 |
| H1 | 2,919 | 1,946 – 3,891 |
| H4 | 4,865 | 3,892 – 5,837 |

Those are the index midpoints, to the row.

And the historical keep counts reproduce with their mechanisms exposed:

| lane | offered | kept | on rank | quota | seed-template |
|---|---|---|---|---|---|
| H1 | 1,946 | 10 | **0** | 6 | 4 |
| H4 | 1,946 | 11 | **0** | 6 | 5 |

Identical in structure to the historical *"8 = 6 quota + 2 force-keep, none
earned it"*. The "none earned it" half was true. The half that mattered — *why*
— was not what anyone thought.

## 3. What the number is now

The correlation has been replaced: `neoethos_data::core::stats_f64`, f64
accumulation, two-pass centred form, pairwise-complete row skipping, and it
returns `used` / `skipped` so the caller can refuse a correlation computed from
nothing. Wired into the prefilter at `discovery.rs:5462`.

Under `new_cpcv` (repaired correlation, triple-barrier label, CPCV folds,
`normalize_features=false`), **no column scores exactly 0.0**:

| lane | offered | kept | on rank | quota | regime | seed-template |
|---|---|---|---|---|---|---|
| base | 1,946 | 151 | 137 | **0** | 14 | 0 |
| H1 | 1,946 | 83 | **79** | **0** | 0 | 4 |
| H4 | 1,946 | 29 | **24** | **0** | 0 | 5 |

Top-240 overlap between the void ranking and the repaired ranking: **13 of 240.**

**The per-timeframe quota is no longer binding.** It supplied 6 + 6 columns
before the repair and 0 + 0 after. `discovery.rs:5938-5945` states in its own
comment that this is the evidence to drop it.

### The distributions look equal — and that appearance is the trap

| lane | median \|r\| | p90 | max |
|---|---|---|---|
| base | 0.014641 | 0.031456 | 0.139935 |
| H1 | 0.014146 | 0.024869 | 0.142077 |
| H4 | 0.011936 | 0.020403 | **0.206786** |

The two highest-scoring columns in the entire 5,838-column cube are H4. Taken at
face value the old conclusion is not merely dead, it is inverted. **Do not take
it at face value.**

A higher-timeframe column is **forward-filled** onto base bars. One H4
observation becomes 48 bit-identical M5 rows, and `pearson_pairwise_f32` counts
all 48 as `used`. Counting instead the number of times a column actually
*changed* (contiguous runs of the bit-identical value):

| lane | rows `used` (median) | distinct observations (median) | runs / used |
|---|---|---|---|
| base | 137,564 | 137,564 | 0.9979 |
| H1 | ~136,700 | 11,559 | 0.0845 ≈ 1/12 |
| H4 | 133,235 | **2,818** | 0.0212 ≈ 1/48 |

Those ratios are exactly the M5:H1 and M5:H4 bar ratios. The sparse event
columns are far worse:

| column | \|r\| | rows `used` | real observations |
|---|---|---|---|
| `H4_range_breakout_signals_extra_bearish` | 0.2068 (#1 in the cube) | 696 | **15** |
| `H4_range_breakout_signals_bearish` | 0.1421 | — | **18** |
| `H4_standardized_psar_oscillator_regular_bullish` | 0.1369 | — | **12** |

`stats_f64::MIN_PAIRWISE_SAMPLES = 30` gates on `used`, so 696 sails through
while the actual evidence is fifteen points.

### The fair comparison

`t = |r|·√(n_eff − 2) / √(1 − r²)` with `n_eff` = distinct observations, against
a Bonferroni `|z|` of **4.451** for 5,838 simultaneous tests at family-wise 5%
(the prefilter runs one test per column and keeps the winners, which is exactly
the setting where an uncorrected maximum means nothing).

| lane | columns clearing the bar | share | max t |
|---|---|---|---|
| base | 953 of 1,880 | 50.7% | 48.93 |
| H1 | **106 of 1,888** | 5.6% | 15.43 |
| H4 | **0 of 1,855** | **0.0%** | 3.82 |

Ranked on `t`, the top 240 is **215 base + 25 H1 + 0 H4**.

Recomputing the correlation at each column's own resolution (one point per run,
label averaged over the run) removes forward-fill attenuation and lifts H4, but
does not rescue it: base 1,090 (58.5%), H1 182 (9.7%), H4 53 (2.9%); `t`-ranked
top 240 = **224 base + 16 H1 + 0 H4**.

## 4. What this means for the standing belief

**H1: the belief is overturned. Re-open the lane.**
H1 went from 0 columns earned on rank to 79, and 106 clear a Bonferroni bar on
their own effective sample size, topping out at `t = 15.4`. The winners are
`H1_fp_volume_z`, `H1_quant_rvol_50`, `H1_quant_rvol_20`,
`H1_normalized_volume_true_range` — volatility and volume *state*, which is
precisely what a higher timeframe should contribute to a lower-timeframe entry.
The prefilter was discarding all of it, and every discovery run before the repair
searched a feature space with the H1 contribution removed by an arithmetic
accident.

**H4: the belief survives on this symbol and this label — but for a different,
measured, and in principle fixable reason.**
Zero of 1,855 H4 columns clear the multiple-comparison bar on their own evidence.
The 24 that currently take top-240 places on rank do so on inflated row counts,
the worst of them on 12–18 real observations dressed as 568–840 rows. The reason
is arithmetic, not mystical: the CPCV cap of 200,000 M5 rows is only ~2,800 H4
bars, and a 35-bar (~3 hour) first-passage label **resolves inside a single H4
candle** — the feature is being asked to predict an outcome shorter than its own
bar. Admitting those columns unchanged would replace a false negative with a
false positive. If H4 is to be tested properly it needs a longer label horizon
and/or an H4-native base, not a quota.

**What must NOT be done with this document:** treat "H4 max |r| = 0.207" as
evidence that H4 is the strongest lane. It is the strongest *number* and the
weakest *evidence* in the cube, and the two facts have the same cause.

## 5. Open defects this measurement exposed

These are consequences of the measurement, recorded here so they are not lost.
None of them is fixed by this document.

1. **`MIN_PAIRWISE_SAMPLES` counts forward-filled rows as evidence.**
   `stats_f64.rs` gates rankability on `used`, which for a higher-timeframe
   column is the fill count, not the observation count. Until it counts distinct
   observations, the prefilter will keep promoting 15-observation event columns
   to the head of a 5,838-column cube. This is the single highest-value
   follow-up, and it must land **before** any early-reject predicate ranks on
   these scores.

2. **The per-timeframe quota is dead weight** (6 + 6 before the repair, 0 + 0
   after). Drop it in the same change as (1), not before — it is currently the
   only thing guaranteeing multi-timeframe seed templates resolve if the ranking
   swings back.

3. **`normalize_features` has two disagreeing defaults.** `neoethos-core`'s
   config default is `true` (`config.rs:2151`); `neoethos-data`'s is `false`
   (`lib.rs:25`); the operator's installed config says `false`. They produce
   materially different feature selections — top-240 overlap with the legacy
   ranking is 13/240 under `false` and 3/240 under `true`, H4 keeps 29 vs 18, and
   under `true` the old f32 function does not zero the higher-TF lane at all
   (only 55 H1 / 86 H4 exact zeros). Which regime a run lands in currently
   depends on which default wins. One default, chosen deliberately.

4. **Full-history multi-timeframe discovery cannot start on this machine.**
   With all 823,801 EURUSD M5 bars the run aborts before the prefilter:

   ```text
   INDICATOR VOCABULARY COLLAPSE at stage 'classic-ta' (69111 bars):
   145 indicator ids produced 318 columns, below the floor of 149 ids / 325 columns
   ```

   `enforce_floor` (`crates/neoethos-data/src/core/indicator_ledger.rs:604`)
   clamps the floor to `min(constant, what the budget afforded)`, and at full
   depth the budget afforded exactly 149 ids / 325 columns — so the floor equals
   the budget with **zero tolerance**, and any legitimate per-frame drop (here, 3
   `unknown_output` columns from `adaptive_bounds_rsi` on the H1 frame) is a hard
   error. Give the clamped floor a tolerance, or exclude the non-dispatch drop
   reasons from the comparison.

## 6. Measurement conditions

Stated because the conclusions are conditional on them.

| | |
|---|---|
| store | the operator's own vortex store, `%LOCALAPPDATA%\neoethos\data` |
| symbol / lanes | EURUSD, base M5, higher H1 + H4 |
| rows loaded | M5 823,801 · H1 69,111 · H4 17,335 |
| base rows measured | **last 200,000** — see below |
| cube | 200,000 rows × 5,838 columns (1,946 offered per timeframe; the *restored* vocabulary, not 217) |
| label | production triple-barrier first-passage, both directions, sl = 2.5 ATR, rr = 2.75, cost 1.5 pips, hold 35 bars |
| label census | long up 14,239 / down 120,803 / vertical 64,787 / ambiguous 170; short win 14,033 / loss 119,975 / vertical 65,768 / ambiguous 223; undefined 1 — **not degenerate**, no fallback to the 1-bar forward return |
| fit windows | production CPCV (8 splits, 2 test groups, 1% embargo, 2% purge, max_rows 200,000) → 7 of 28 folds, 138k–148k rows each |
| selection | global top-K 240 + `regime_` INFINITY exemption + per-TF quota 6 + seed-template force-keep, replicated verbatim |
| `normalize_features` | **false** (the operator's installed setting, and the regime the void number was made in) |
| rows the correlation used | base 135,795 used / 4,776 skipped (96.60%); H1 134,890 / 5,681 (95.96%); H4 131,307 / 9,264 (93.41%), averaged over the 7 folds |

**Why 200,000 base bars and not full history.** Two independent reasons. The
CPCV cap is 200,000 rows and it takes the tail, so the fit windows see the
identical bars either way — only indicator warm-up at the very start differs.
And the full-history run does not reach the prefilter at all (defect 4 above).

---

## 7. Citations of the void number — corrected or flagged

A void number left in a comment gets believed again. Every occurrence found in
the repository:

| location | status |
|---|---|
| `docs/pending-edits-forbidden-territory.md` §1 | **corrected** — annotated as void with a pointer here |
| `crates/neoethos-data/src/core/stats_f64.rs:30-31` (module doc, *"That reproduces the measured `base keeps 217/217, H1 keeps 40/217, H4 keeps 8/217` with no free parameters"*) | **NOT YET CORRECTED** — outside this workflow shard's file assignment. Replacement text in §7.1 below. |
| `crates/neoethos-search/examples/htf_prefilter_probe.rs:22` (doc comment naming the historical regime) | **NOT YET CORRECTED** — throwaway measurement probe, superseded by the test above; safe to delete along with `htf_effective_n_probe.rs` |

### 7.1 Replacement text for `stats_f64.rs`

The mechanism paragraph is correct and should stay. Only the sentence that cites
the keep-rates is void. Replace:

> That reproduces the measured "base keeps 217/217, H1 keeps 40/217, H4 keeps
> 8/217" with no free parameters.

with:

> Re-measured on real EURUSD M5/H1/H4 bars 2026-08-09: this scored **100% of H1
> and 100% of H4 columns exactly 0.0** — and, against the triple-barrier label,
> 99.3% of BASE columns too — so the resulting "ranking" was literally the cube's
> column index (median rank per timeframe landed on the index midpoints, 973 /
> 2,919 / 4,865). The older keep-rate figures quoted here previously
> ("217/217, 40/217, 8/217") are VOID; see
> `docs/higher-timeframe-lane-2026-08-09.md` and re-run
> `crates/neoethos-search/tests/higher_timeframe_lane_measured.rs`.

### 7.2 A separate, unrelated "217"

Several files carry the phrase *"at prefilter_top_k = 50 the base feature set
collapses from 217 columns to roughly 64"* — `config.rs:1374`,
`discovery.rs:46`, `discovery_tests.rs:2694`,
`shipped_config_matches_defaults.rs:219` and `:700`,
`docs/config-single-source-of-truth.md:156`,
`docs/knob-second-pass-2026-08-09.md:333`. That is a **different** claim: it is
about the width of the base cube, not about higher-timeframe keep rates, and it
is not void by the mechanism described here.

It is nonetheless **stale**, and this document is the wrong place to fix it: the
base cube is no longer 217 columns. The restored vocabulary offers **1,946
columns per timeframe** on this machine (5,838 across three lanes) and up to the
memory budget's 4,096 on a larger one, so a constant `top_k` of 240 now discards
a fraction that depends on the hardware. Those sites are flagged for the
`prefilter_top_k` work, not corrected here. The two docs occurrences carry that
flag in place.
