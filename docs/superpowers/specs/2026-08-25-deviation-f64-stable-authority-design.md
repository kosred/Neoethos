# NeoEthos Deviation f64 Stable Authority V2 Design

## Scope

Replace only the numerically unstable population-standard-deviation authority used by
VectorTA `deviation` devtypes 0 and 3. Preserve the public indicator id, default period,
warmup, output width, and independent f32 API. Correct the contradictory f64 non-finite and
period-one behavior explicitly. Devtypes 1 and 2 are not part of this change.

The result must be one versioned f64 authority shared by scalar, `Kernel::Auto`, AVX2,
AVX-512, streaming, and strict CUDA. Architecture-dependent reduction trees are not an
authority.

## Evidence and problem statement

The reviewed RTX fixture's first period-9 window produces three different answers:

- host AVX2/Auto: `0x3efaabdbb4eb0e79`;
- host scalar and strict CUDA raw-moment order: `0x3efaabdc4e7dc99f`;
- the correctly rounded result over the exact input f64 values: `0x3efaabdbf86838c1`.

The current formula, `E[x^2] - E[x]^2`, loses significant digits through cancellation.
Changing reduction order merely selects a different inaccurate answer. External numerical
analysis is an audit oracle only; vendored VectorTA remains the sole runtime.

## Considered approaches

### A. Preserve the raw-moment formula and choose one reduction order

This is O(N), simple, and bit-repeatable, but retains catastrophic cancellation. Rejected.

### B. Rolling central moments with periodic recomputation

This is O(N), but deterministic adversarial windows leave a positive residual after variance
collapses to zero. A periodic schedule does not bound that error between recomputations.
Rejected.

### C. Globally scaled, anchored two-pass authority per output window

Compute every output window independently. Derive an exact power-of-two scale from the largest
absolute input before subtracting any pair of values, normalize the inputs, then use an anchored
two-pass centered sum with Neumaier compensation. This is O(N * period), but avoids serial drift,
handles subnormal spreads and opposite-sign finite extremes, and parallelizes safely across
independent output windows. It measured at most 1 ULP over 150,480 exact-rational audit windows.
Selected.

## Stable authority V2

For each finite window of width `n`:

1. Scan oldest-to-newest, reject any non-finite input, and retain `max_abs_input` by comparing
   sign-cleared binary64 magnitude bits.
2. If `max_abs_input == 0`, emit positive zero. Otherwise derive `scale` from its bits as the
   largest positive power of two not greater than `max_abs_input`. This includes an exact
   subnormal-power path and uses no logarithm/libm.
3. Normalize the oldest value first and retain it as `anchor = x[0] / scale`.
4. In a second oldest-to-newest pass, form `d = x / scale - anchor` and update a Neumaier
   compensated sum of `d`.
5. Set `mean_delta = compensated_shifted_sum / n`.
6. In a third oldest-to-newest pass, form `z = (x / scale - anchor) - mean_delta`, square it
   with one specified fused operation, and accumulate `z^2` with the same Neumaier primitive.
7. Emit `scale * sqrt(compensated_normalized_square_sum / n)`. Every arithmetic operation,
   compensation branch, and loop order is part of the V2 authority.

There is no rolling M2 recurrence and no data-dependent cancellation threshold. Every output
is independent, so one bad window cannot contaminate later rows.

All non-fused operations have an explicit source order. CPU uses `f64::mul_add` exactly where
CUDA uses `__fma_rn`; CUDA uses explicit round-to-nearest double intrinsics for add, subtract,
divide, FMA, and square root. SIMD may vectorize across independent windows/series, never
horizontally across observations inside a window.

Scaling the raw inputs first is mandatory. Scaling `x - anchor` is not equivalent: subtraction
of opposite-sign finite values near `f64::MAX` can overflow before the scale is known. The V2
semantic identity is
`deviation_population_f64_global_pow2_anchored_neumaier_two_pass_fma_sqrt_rn_v2`.
Power-of-two normalization is correctly rounded binary64 arithmetic. It is not described as an
unconditional exact exponent shift because a component can underflow when a window spans more
than the representable exponent range. The frozen audit corpus includes such mixed-range cases.

The normalized square total must be finite and nonnegative. A negative/non-finite total or a
non-finite final result from finite inputs is a defect and emits canonical NaN; there is no clamp,
tolerance, rolling fallback, or free-form repair branch.

## Non-finite and lifecycle semantics

- Period one emits positive zero for a finite singleton and canonical NaN for NaN or infinity.
- A window containing NaN or either infinity emits canonical NaN.
- Every next window is evaluated independently, so recovery occurs on the first fully finite
  window without retained poisoned state.
- `DeviationStream` retains its ring and evaluates the current ring in logical oldest-to-newest
  order with the same two-pass authority. Batch and stream must be bit-identical after the same
  input prefix.
- Devtype 3 remains an alias of the same standard-deviation authority.

## Tests and acceptance

RED tests must precede production edits and prove:

1. The exact nine-value RTX fixture resolves to `0x3efaabdbf86838c1` for scalar, Auto,
   explicit AVX2/AVX-512 when supported, and stream.
2. Large-offset/tiny-variance, variance-collapse-to-constant, subnormal spread, opposite-sign
   `f64::MAX`/near-maximum values,
   mixed `[min_subnormal, 1, f64::MAX]` values and sign variants,
   monotone-drift, period 1/2/9/50/200, and interior non-finite fixtures compare against exact
   rational arithmetic over the input binary64 values. The audited corpus must remain within
   1 ULP for nonzero finite truth; this is an empirical acceptance bound, not a universal
   correctly-rounded theorem. Zero, NaN, and infinity use exact category/bit rules.
   Architecture lanes are bit-identical.
3. The source contract rejects raw moments and rolling M2 for the f64 authority and pins the
   global-input scaling, normalized anchor, compensation, two-pass schedule, explicit CUDA RN
   intrinsics, and independent-window launch in Rust and strict CUDA.
4. Existing devtype 1/2 and f32 behavior remains unchanged.

After local warning-denied tests, the RTX gate is a release no-run build, the exact reviewed
routeable-subset fixture, and the same filter under Compute Sanitizer with zero errors/leaks.
A focused before/after benchmark must measure the O(N * period) scalar cost and the recovered
throughput from CPU SIMD/GPU parallelism across independent outputs. Search-level regression,
not a synthetic scalar ratio alone, decides acceptance.

## Explicit boundaries

This design intentionally corrects period-one non-finite handling: finite singleton windows
emit positive zero, while NaN/infinity singletons emit canonical NaN. It does not make f32 an
oracle, does not add TA-Lib or any other runtime dependency,
does not alter indicator defaults or schema, and does not claim the whole Search pipeline is
complete. It closes one proven f64 mathematical and cross-device authority defect.
