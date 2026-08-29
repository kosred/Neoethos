# Regime semantic v3: reviewed formula and exact-f64 contract

Status: approved semantic authority with CPU and resident-CUDA production
sources implemented. Source/rustfmt/contract checks do not constitute Cargo,
NVCC, real-device parity, or release validation; those remain explicitly open.

## 1. Authority and non-authority

`regime_detection.rs` semantic v2 is an untrusted migration input. It is not an
oracle for v3. In particular, v2 currently:

- labels a clipped volatility-ratio offset as a z-score;
- emits DX while describing the output as ADX-like trend strength;
- seeds DMI smoothing as an average and then applies the sum-form Wilder
  recurrence;
- marks row 14 valid in the f64 wrapper even though the 15-row legacy producer
  does not execute that calculation;
- uses sample standard deviation for Bollinger Bands;
- calls an ATR-normalized deviation a linear-regression squeeze momentum;
- calls a directional sign-count a mean-reversion probability;
- calls a candle-body/range balance DeMARK Range Expansion Index;
- uses ordinary high/low extrema instead of Dreiss true-high/true-low extrema;
- substitutes epsilon denominators and neutral numeric values for invalid cells.

V3 has two deliberately separate authorities:

1. **Formula semantics.** Creator, primary-paper, or official implementation
   sources decide what a named external formula means in real arithmetic.
2. **NeoEthos exact-f64 schedule.** The operation order in section 5 decides the
   exact CPU-f64 and GPU-f64 bits. It is a custom numerical implementation and
   must never be presented as part of the creator's formula.

External-formula tests may compare a high-precision implementation with an
explicit error bound. CPU-v3 versus CUDA-v3 parity is stricter: every valid
value and every validity byte must match exactly. A device which cannot satisfy
that contract must fail capability admission; it may not silently use CPU
values or widen a tolerance.

## 2. Independent source ledger

| Formula family | Primary, creator, or official evidence | V3 conclusion |
|---|---|---|
| Garman-Klass | Garman and Klass's paper gives the recommended practical estimator at eq. 19a: <https://www.cmegroup.com/trading/fx/files/a_estimation_of_security_price.pdf>; DOI metadata: <https://econpapers.repec.org/article/ucpjnlbus/v_3a53_3ay_3a1980_3ai_3a1_3ap_3a67-78.htm> | Use the eq. 19a per-bar variance component. The 10/50 rolling ratio, thresholds, and clipped offset remain NeoEthos custom semantics. `abs(variance)` is forbidden. |
| Z-score | NIST defines a z-score as deviation from a mean divided by a standard deviation: <https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm> | `ratio - 1` is not a z-score. Preserve the actual clipped offset only under a custom name. |
| Wilder DMI/ADX | Wilder's primary book identifies the Directional Movement Concept: <https://books.google.com/books/about/New_Concepts_in_Technical_Trading_System.html?id=WesJAQAAMAAJ>. TradingView publishes true range, Wilder sum recurrence, first ADX seed, and subsequent ADX recurrence: <https://www.tradingview.com/support/solutions/43000589099-average-directional-index-adx/>. MetaTrader documents DI, DX, and ADX as a strict Wilder correspondence: <https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/admiw> | Emit true 14-period Wilder ADX on the 0..100 scale. DI dominance direction and the ADX-25 directional state are explicitly custom derived signals. The displayed MetaTrader TR line differs from the standard true-range line; v3 follows the primary concept and TradingView's explicit `max(H-L, abs(H-Cprev), abs(L-Cprev))` schedule. |
| Bollinger Bands | John Bollinger specifies a 20-period mean, two bands, and **population** standard deviation: <https://www.bollingerbands.com/bollinger-bands> | Use population variance (`/20`), not sample variance (`/19`). |
| Keltner/TTM Squeeze | Official Keltner documentation describes ATR offsets around a moving average: <https://toslc.thinkorswim.com/center/reference/Tech-Indicators/studies-library/G-L/KeltnerChannels>. The official TTM Squeeze page states that its momentum histogram uses linear regression and other techniques: <https://toslc.thinkorswim.com/center/reference/Tech-Indicators/studies-library/T-U/TTM-Squeeze.html> | The complete proprietary TTM operation schedule is not public, and v2 does no linear regression. Preserve only v2's actual 20-period Bollinger/Keltner containment and ATR-normalized midline deviation under custom names. Do not use TTM in an identity. |
| Directional persistence | Lo and MacKinlay explicitly warn that rejecting a random walk does not establish mean reversion: <https://www.mit.edu/~alo/Papers/lo-mackinlay-88.html> | Preserve the sign balance of adjacent close differences, but call it directional persistence, not mean-reversion probability or autocorrelation. |
| DeMARK REI boundary | DeMARK states that REI compares current activity with two bars earlier, suppresses sideways/aberrant activity, and is proprietary: <https://demark.com/indicators-list/> | V2's eight-candle body/range balance is not REI. Preserve its actual math under a NeoEthos custom identity and remove `rei` from the v3 name/type. |
| Dreiss Choppiness | The published Dreiss description uses the sum of true ranges divided by highest true high minus lowest true low: <https://c.mql5.com/forextsd/forum/165/at_september_2009_choppiness_index.pdf>. NeuroShell publishes the same calculation and cites Dreiss: <https://www.neuroshell.com/manuals/ais1/choppinessindex.htm> | Use the true-high/true-low denominator and the creator's 0..100 scale. Do not clamp a gap-distorted result back into range. |
| CUSUM | Page's primary paper is E. S. Page, *Continuous Inspection Schemes*, Biometrika 41 (1954), DOI 10.1093/biomet/41.1-2.100: <https://academic.oup.com/biomet/article-abstract/41/1-2/100/456627>. NIST documents standardized CUSUM, reference value `k`, decision limit `h`, and two one-sided charts: <https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc3231.htm> | The rolling-50 z-score, `k=0.5`, `h=3`, and post-hit reset are a NeoEthos custom Page-style adaptation. Preserve and name the actual schedule; do not claim canonical Page calibration. |
| Shannon entropy | Shannon's primary paper defines `-sum(p log p)` and maximum `log(n)`: <https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1948.tb00917.x> | Shannon's entropy formula is used, but the 30 log-return, ten equal-width-bin discretization is NeoEthos custom and is named as such. |

## 3. Frozen 14-slot order and truthful identities

The slot count and slot order remain stable only to bound the migration. Names,
formula identities, scale, and warmups change, so v2 content is incompatible.
All invalid cells carry canonical quiet-NaN bits `0x7ff8000000000000`.

| Slot | Retired v2 name | Semantic-v3 name | Actual v3 value | First theoretical row |
|---:|---|---|---|---:|
| 0 | `regime_vol_state` | `neoethos_custom_gk_vol_ratio_state_10_50_v3` | `+1` if ratio `>1.5`, `-1` if ratio `<0.6`, otherwise valid `0` | 49 |
| 1 | `regime_vol_zscore` | `neoethos_custom_gk_vol_ratio_offset_10_50_v3` | `clamp(short_gk/long_gk - 1, -3, 3)`; this is not a z-score | 49 |
| 2 | `regime_trend_strength` | `regime_wilder_adx_14_v3` | true Wilder ADX on `0..100` | 27 |
| 3 | `regime_trend_direction` | `neoethos_custom_wilder_di_dominance_direction_14_v3` | `+1` for `+DI>-DI`, `-1` for `-DI>+DI`, valid `0` on a non-degenerate tie | 14 |
| 4 | `regime_trend_state` | `neoethos_custom_wilder_adx_direction_state_14_25_v3` | direction when ADX `>25`, otherwise valid `0` | 27 |
| 5 | `regime_squeeze` | `neoethos_custom_bollinger_keltner_squeeze_state_20_2_1p5_v3` | `+1` only for strict BB-inside-custom-KC containment, otherwise `-1` | 20 |
| 6 | `regime_squeeze_momentum` | `neoethos_custom_bollinger_midline_atr_deviation_20_v3` | `(close - population-BB mean) / arithmetic-mean TR` with no linear regression | 20 |
| 7 | `regime_mr_vs_momentum` | `neoethos_custom_directional_persistence_balance_20_v3` | `(same_direction_pairs - reversal_pairs) / nonzero_pairs` | 21 |
| 8 | `regime_rei` | `neoethos_custom_candle_body_range_balance_8_v3` | `clamp(sum(close-open)/sum(high-low), -1, 1)`; not DeMARK REI | 7 |
| 9 | `regime_choppiness` | `regime_dreiss_choppiness_index_14_v3` | `100*log10(sum(TR)/[max(true_high)-min(true_low)])/log10(14)` | 14 |
| 10 | `regime_cusum_up` | `neoethos_custom_standardized_cusum_up_50_0p5_3_v3` | post-reset upper standardized custom CUSUM state | 50 |
| 11 | `regime_cusum_down` | `neoethos_custom_standardized_cusum_down_50_0p5_3_v3` | post-reset lower standardized custom CUSUM state | 50 |
| 12 | `regime_change_signal` | `neoethos_custom_standardized_cusum_signal_50_0p5_3_v3` | strict `h=3`: `+1`, `-1`, or valid `0` | 50 |
| 13 | `regime_entropy` | `neoethos_custom_equal_width_log_return_entropy_30_10_v3` | normalized Shannon entropy of 30 log returns in ten custom equal-width bins | 30 |

The UTF-8 name bytes above total exactly 667. An isolated one-batch Regime
schema therefore uses a 448-byte pointer table, 120 bytes of 15 u64 name
offsets, and 667 name bytes: 1,235 bytes of pointer/schema metadata. In the full
workspace the pointer table is a maximum across producer batches rather than an
additive per-producer allocation.

## 4. Real-arithmetic formulas and validity

### 4.1 Garman-Klass ratio family

For bar `j`, let `u=ln(H_j)-ln(O_j)`, `d=ln(L_j)-ln(O_j)`, and
`c=ln(C_j)-ln(O_j)`. The creator-sourced practical variance component is:

`g_j = 0.5*(u-d)^2 - (2*ln(2)-1)*c^2`.

At row `i`, `short_gk=sqrt(sum(g[i-9..i])/10)` and
`long_gk=sqrt(sum(g[i-49..i])/50)`. A negative component is a compute failure;
v3 never applies `abs`. Zero long volatility is `ZeroDenominator` for both
slots. Threshold comparisons are strict.

### 4.2 Wilder DMI/ADX and custom derived states

For rows `i>=1`, use standard true range and mutually exclusive directional
movement. Seed the 14-period smoothed **sums** from rows 1 through 14. For each
subsequent row update each sum as `(previous - previous/14) + current`, in that
order. Compute `+DI`, `-DI`, and DX on the 0..100 scale. The first ADX at row 27
is the arithmetic mean of DX rows 14 through 27. Later ADX values use
`((previous_adx*13)+current_dx)/14` in that order.

If TR or `+DI + -DI` is zero, direction is `ZeroDenominator`. ADX seeding
requires 14 consecutive valid DX values. An invalid DX resets that seed; at or
after row 27 the ADX and state are `ZeroDenominator` until reseeded. Before the
first theoretical row, `Warmup` takes precedence.

This resolves the 15-row defect: with rows 0..14, a directional series has a
computed, valid slot-3 value at row 14, while slots 2 and 4 remain `Warmup`.

### 4.3 Custom Bollinger/Keltner family

At row `i>=20`, the close window is rows `i-19..i`. Bollinger mean is the
arithmetic mean and variance is the population variance (`/20`). Twenty true
ranges are rows `i-19..i`, which is why row 20 is the first row: every TR has a
previous close. The custom Keltner center is the same close mean and its width
is `1.5 * arithmetic_mean(TR)`.

The state retains v2's strict comparisons of explicitly rounded upper/lower
bands. ATR equal to zero is `ZeroDenominator` for both outputs; a constant
series is not an expansion signal. The deviation output uses the BB mean
directly and performs no regression.

### 4.4 Other custom bounded windows

- Directional persistence examines exactly 20 adjacent close-difference pairs.
  It compares signs without multiplying the differences, avoiding overflow and
  underflow while preserving the real-arithmetic sign test. Zero differences
  are omitted. No nonzero pairs is `ZeroDenominator`.
- Candle body/range balance uses exactly eight bars. Zero total high-low range
  is `ZeroDenominator`; a zero numerator with positive denominator is valid
  zero.
- Dreiss Choppiness uses 14 true ranges, `true_high=max(high, previous_close)`,
  and `true_low=min(low, previous_close)`. Numerator or denominator zero is
  `ZeroDenominator`. The creator result is not clamped.

### 4.5 Custom standardized CUSUM

At row `i>=50`, mean and sample variance (`/49`) use prior closes `i-50..i-1`;
the current close is not in its own baseline. For valid positive standard
deviation, `z=(close_i-mean)/std`, then in this exact logical order:

1. `candidate_up=max(previous_up + z - 0.5, 0)`;
2. `candidate_down=max(previous_down - z - 0.5, 0)`;
3. if `candidate_up > 3`, emit signal `+1` and reset emitted `up` to zero;
4. else if `candidate_down > 3`, emit signal `-1` and reset emitted `down` to zero;
5. otherwise emit signal zero and both candidates.

The emitted post-reset states become the next row's previous states. Equality
with three is not a hit. A zero-variance baseline marks all three cells
`ZeroDenominator` and resets both hidden states to zero; state is never carried
through an undefined standardization.

### 4.6 Custom binned log-return entropy

At row `i>=30`, form 30 returns `ln(C_j)-ln(C_{j-1})` for `j=i-29..i`. If all
returns are equal, emit valid zero: a point mass has Shannon entropy zero. For a
positive range, map each return with
`floor(((r-min)/range) * 9.999)`, clamp the integer index to `0..9`, count bins,
and evaluate bins in order 0 through 9. The result is
`-sum(p*ln(p))/ln(10)`. The equal-width binning is custom; only the entropy
functional is creator-sourced.

## 5. NeoEthos CPU-f64/GPU-f64 exact schedule v1

Schedule identity:
`neoethos.regime.semantic-v3.f64-rn-fixed-order-log49-neumaier-v1`.

1. Input lengths must agree and `N>0`. Every OHLC value must be finite and
   positive, and each row must satisfy
   `low <= min(open,close) <= max(open,close) <= high`. Violations refuse the
   entire producer before output allocation using a typed v3 input error. The
   nonfinite fixture therefore expects producer refusal, not fabricated
   per-cell values.
2. Difference-based formulas use a single dataset-wide exact power-of-two
   anchor derived from the greatest OHLC value's binary exponent. Each price is
   scaled by that power of two before subtraction. If a positive input cannot
   remain representable after scaling, admission fails with
   `ScaleRangeUnsupported`; no epsilon is substituted.
3. Every bounded sum uses chronological oldest-to-newest Neumaier summation:
   compute `t=sum+x`; update compensation using the magnitude branch; then set
   `sum=t`; finish with `sum+compensation`. Variance is two-pass: exact-schedule
   mean first, then chronological compensated squared deviations.
4. Positive logarithms use the already reviewed 49-denominator atanh schedule
   mirrored by VectorTA CHOP, generalized as `neoethos_ln_positive_exact_v1`:
   normalize subnormals by exact `2^54`, split exponent/mantissa by bits, set
   `z=(m-1)/(m+1)`, and accumulate `z + z^3/3 + ... + z^49/49` in increasing
   odd-denominator order. Return `exponent*LN2 + 2*sum`. `log10(x)` is
   `ln_exact(x)/LN10`. No host `ln`, `log10`, `ln_1p`, CUDA `log`, or CUDA
   `log10` may enter this producer.
5. Constants are bit-frozen: `LN2=0x3fe62e42fefa39ef`,
   `LN10=0x40026bb1bbb55515`, Garman-Klass coefficient
   `0x3fd8b90bfbe8e7bc`, entropy bin multiplier
   `0x4023ff7ced916873`, and canonical NaN
   `0x7ff8000000000000`.
6. CPU code uses strict ordinary f64 operations, never `mul_add`, fast-math, or
   algebraic reassociation. CUDA uses the corresponding round-to-nearest-even
   double intrinsics (`__dadd_rn`, `__dsub_rn`, `__dmul_rn`, `__ddiv_rn`,
   `__dsqrt_rn`) at schedule boundaries. NVCC must retain `--fmad=false`,
   `--ftz=false`, `--prec-div=true`, `--prec-sqrt=true`, and must not use
   `--use_fast_math`. NVIDIA documents the flags and intrinsics here:
   <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html> and
   <https://docs.nvidia.com/cuda/programming-guide/05-appendices/mathematical-functions.html>.
7. Any nonfinite intermediate or theoretically impossible negative variance is
   `ComputeFailure` with canonical NaN. Exact zero denominators use
   `ZeroDenominator`. There is no `1e-10`/`1e-12` denominator floor.

## 6. Validity precedence and edge cases

1. Whole-producer typed input refusal: empty input, length mismatch, nonfinite,
   nonpositive, OHLC-envelope violation, or unsupported scale range.
2. `Warmup` before a slot's first theoretical row, even if the partial inputs
   are degenerate.
3. `ZeroDenominator` once a complete theoretical window exists but its exact
   required denominator is zero.
4. `ComputeFailure` for a nonfinite intermediate or violated mathematical
   invariant after valid input.
5. `Valid` for finite computed values, including categorical zero and constant
   log-return entropy zero.

The checked RED fixture covers empty/nonfinite refusal, the exact 15-row DMI
boundary, every first theoretical row, constant windows, population-vs-sample
Bollinger variance, a price gap that distinguishes true-high/true-low
Choppiness, strict CUSUM hit/reset order, and constant-return entropy.

## 7. Fail-closed v2 migration

- `REGIME_SEMANTIC_VERSION` becomes 3 and the ordered v3 names are part of the
  feature-plan/content identity.
- The anonymous legacy 14-name family is designated semantic v2 only for
  refusal and offline regeneration.
- Any artifact, model frame, feature store, checkpoint, or cached route naming
  a retired v2 column or semantic version below 3 is rejected before search or
  model loading. Equal width and equal slot order do not authorize migration.
- There are no v2 aliases, dual emission, caller-selectable compatibility flag,
  CPU fallback on a GPU-present path, or automatic value conversion. The only
  migration is explicit regeneration from canonical OHLC under v3.

## 8. Resident CUDA and byte authority

Regime is one 14-column producer batch and borrows only the resident parent's
open, high, low, and close arrays, context, stream, identity, and ready event.

- retained values: `14*N*8 = 112N` bytes;
- retained logical validity: `14*N*1 = 14N` bytes;
- exact producer-retained total: `126N` bytes;
- additional retained bytes: `0`;
- allocator-visible scratch bytes: `0` (bounded windows are recomputed and
  recurrence state remains in registers);
- isolated pointer table: `14*4*8 = 448` bytes;
- isolated pointer/schema metadata with frozen names: `1,235` bytes;
- parent-input H2D bytes: `0`;
- feature-value/validity D2H bytes: `0`;
- ready-event count: `1`;
- native launch count: `2`.

Launch 1 is a row-parallel bounded-window kernel for slots 0, 1, 5, 6, 7, 8,
9, and 13. Launch 2 is one small chronological kernel with independent lanes
for DMI/ADX (slots 2..4) and CUSUM (slots 10..12). Launch 2 waits on launch 1
only through stream order; the single producer-ready event is recorded after
both. Neither launch materializes host feature values or synchronizes the
stream.

## 9. Frozen contract and production paths

The RED-first design froze:

- `docs/regime-semantic-v3.md`;
- `crates/neoethos-data/tests/fixtures/regime_semantic_v3_red_v1.json`;
- `crates/neoethos-data/tests/regime_semantic_v3_red_contract.rs`.

The approved production pass implements and source-contracts:

- `crates/neoethos-data/src/core/regime_detection.rs`;
- a single shared CPU exact-log authority chosen during implementation;
- `crates/neoethos-data/src/core/gpu_resident_regime_v3.rs`;
- `crates/neoethos-gpu-cuda/src/resident_regime_v3.rs`;
- `crates/neoethos-gpu-cuda/native/resident_regime_v3.cu`;
- Data/GPU `mod.rs`/`lib.rs`, CUDA `build.rs`, feature registry, shared resident
  store receipt, full-workspace preflight, and model/artifact migration paths.

## 10. Remaining validation blockers

1. The in-tree Rust/CUDA `log49` transcription is bound by one operation-token
   hash and independent marked-source hashes; Cargo/NVCC must still confirm the
   source seals survive the actual build inputs.
2. The exact Rust code-generation contract must be inspected after
   implementation to confirm no contraction/reassociation. The CUDA flags and
   explicit RN intrinsics are specified, but no build was run in this design
   turn.
3. CPU-v3 and real-device CUDA-v3 exact parity, first-divergence primitives,
   memory receipts, and launch/event evidence remain unvalidated until the
   native implementation is compiled and run on the target NVIDIA card.
4. The current-source census finds the retired 14 names only in the explicit
   refusal ledger, frozen migration fixture/spec, and one negative Search
   assertion; live Search consumers use v3 names and the 0..100 ADX scale.
   Cargo-level artifact/loading tests must still prove that no serialized v2
   schema bypasses registry rejection. No alias may hide a missed consumer.
