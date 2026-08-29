# EHMA Deterministic f64 Authority V2 Design

## Scope

This change replaces only the vendored VectorTA EHMA f64 numerical authority.
The f32 CUDA kernels remain a separate precision contract and are not changed.
There is no TA-Lib dependency, backend, fallback, or runtime oracle.

The creator semantics are the Hann-windowed FIR described in John F. Ehlers'
September 2021 *Windowing* article and reproduced by the magazine's later
author-code examples:

```text
weight(k) = 1 - cos(2*pi*k/(period + 1)), k = 1..period
EHMA = sum(weight(k) * price(k)) / sum(weight(k))
```

Primary-source index and author-code evidence:

- https://technical.traders.com/archive/display2.asp?mo=SEP&yr=2021
- https://traders.com/Documentation/FEEDbk_docs/2021/12/TradersTips.html

External material is a semantics oracle only. Production continues to execute
the vendored VectorTA CPU or strict CUDA implementation.

## Root cause

The strict RTX fixture exposed row 13 as CPU
`0x3ff13338cd765d62` versus CUDA `0x3ff13338cd765d61`.
The CPU expected column does not call the direct `ehma()` API. It follows
`compute_cpu -> compute_cpu_batch -> ma_batch -> ehma_batch_with_kernel`.
That route builds forward recurrence weights and Auto selects an AVX four-lane
reduction. Strict CUDA builds the same forward recurrence but accumulates one
chronological FMA chain. The direct Scalar/Auto APIs reverse the recurrence
weights, while streaming uses a different rotating recurrence. These are four
different f64 contracts.

For the exact 14-price RTX window, an independent 200-digit evaluation of the
mathematical Ehlers formula rounds to `0x3ff13338cd765d61`. Therefore the CUDA
bit is accidentally correct for this row; copying the AVX reduction into CUDA
would make parity pass by moving away from the mathematical result.

## Rejected approaches

1. **Force the current CPU batch onto the current CUDA sum.** This fixes the
   observed row but leaves host-libm versus libdevice coefficients, recurrence
   asymmetry, streaming drift, and large-offset instability.
2. **Make CUDA reproduce AVX lane reassociation.** This produces the wrong
   rounded bit for the observed mathematical oracle and changes with period
   remainder and SIMD width.
3. **Keep libm/libdevice and accept a tolerance.** Exact-bit strict f64 parity
   is the required contract; widening tolerance would conceal route drift.

## Stable Authority V2

The version identity is:

```text
ehma_hann_f64_msun_ddangle_symmetric_pow2_anchored_dot2_v2
```

(`msun_ddangle` means the fixed msun transcription plus a double-double angle
tail.)

### Coefficients

1. Pin `PI_HI` and `PI_LO` by their IEEE-754 bits.
2. For `k = 1..ceil(period/2)`, compute the correctly rounded binary64
   half-angle `pi*k/(period+1)` with a quotient remainder and two-product
   residual. No host `sin`, `cos`, `sin_cos`, CUDA `sin`, `cos`, or `sincos`
   participates in the authority.
3. Evaluate sine with the existing FreeBSD-msun polynomial and medium pi/2
   reduction schedule, transcribed operation-for-operation in Rust and CUDA.
4. Form the equivalent Hann coefficient as `2 * sin(half_angle)^2`. This avoids
   cancellation in `1 - cos(theta)` at the smallest admitted angles.
5. Copy that coefficient bit-for-bit to the symmetric tap
   `period - k`. Do not regenerate the mirror through recurrence or trig.
6. Accumulate the normalization coefficient in chronological order with
   error-free `TwoSum` residuals.

All periods 1 through the strict CUDA bound 512 use this schedule. Period 1
has the exact coefficient 2.0.

### Window evaluation

For every finite window:

1. Find the largest absolute input and derive its exact floor power-of-two
   scale from its bits. Scale before subtraction, so opposite finite extremes
   cannot overflow and subnormal-only windows can be normalized.
2. Normalize the values by that scale and use the oldest chronological value
   as the anchor.
3. Accumulate each `(normalized_value - anchor) * weight` in chronological
   order. `TwoProd` uses an explicit fused residual and `TwoSum` captures the
   addition residual. Residuals are accumulated in the same order.
4. Divide the compensated shifted dot by the compensated coefficient, add the
   anchor, and rescale.

A NaN in a window produces the canonical quiet NaN. Infinite-input behavior
is kept on an explicit common non-finite fallback rather than being silently
treated as finite or narrowed. Warmup and recovery after a finite NaN gap stay
unchanged.

### Route unification

- Scalar, Auto, AVX2, AVX512, batch, `*_into`, and WASM f64 dispatch all call
  the same V2 coefficient builder and chronological evaluator. AVX API labels
  remain valid, but no f64 reassociation is permitted.
- `EhmaStream` stores the same mirrored weights and recomputes the logical
  ring window with the V2 evaluator after warmup. Its old rotating recurrence
  and drift-prone state are removed.
- `ehma_neo_batch_f64` implements the same constants, branches, and operation
  order with strict RN intrinsics where fusion is intended. The f64 translation
  unit remains compiled with precise division/sqrt, `-fmad=false`, and no FTZ.
- The f32 prefix of `ehma_kernel.cu` is byte-for-byte out of scope.

## Validation

RED-first source and numeric contracts pin:

- the exact RTX row-13 bit `0x3ff13338cd765d61`;
- coefficient symmetry for every period 1..512;
- constant signals, large same-sign offsets, subnormal-only inputs, mixed
  ordinary signs, period edges, a NaN gap and recovery;
- exact equality of direct Scalar/Auto/AVX, batch, into, and stream routes;
- absence of libm/libdevice and AVX reassociation from the f64 authority;
- preservation of all existing f32 entry points.

Local work is limited to source contracts, rustfmt, hashes, and static checks.
The parent RTX loop must then run the VectorTA release tests, the exact resident
Classic TA device fixture, Compute Sanitizer, and a before/after EHMA benchmark.
Correctness is the merge gate; any performance regression is measured and then
optimized without changing the V2 operation schedule.
