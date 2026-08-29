# FRAMA Large-Window Deque Design

## Goal

Make the optimized host FRAMA path for effective windows greater than 32 exactly equivalent to the published/direct two-half range formula, while keeping O(N) runtime and preserving the existing public API, warmup, NaN handling, smoothing constants, and non-fused final EMA arithmetic.

## Evidence and scope

The current `frama_scalar_deque` partitions the initial window into left and right monotonic deques, expires entries, and swaps the deques only once per half-window. On the authoritative 4096-row fixture this happens to remain equivalent for windows 50 and 100, but window 200 diverges from a direct two-half scan on 3,799 rows beginning at row 297. This is a mathematical window-membership defect, separate from the smaller host-plain versus CUDA-FMA rounding difference.

This change does not select CPU or CUDA as a numerical baseline. It does not change CUDA, transcendental functions, or the final host EMA expression.

## Chosen design

Maintain one monotonic max/min pair for each exact half-window:

- The left pair initially owns `[first, first + half)`.
- The right pair initially owns `[first + half, first + window)`.
- At output row `i`, these represent `[i - window, i - half)` and `[i - half, i)` respectively.
- After computing row `i`, expire `i - window` from the left pair, move `i - half` from the right boundary into the left pair, expire that boundary from the right pair, and add `i` to the right pair for row `i + 1`.
- Derive the full-window maximum/minimum from the two exact halves, so no third full-window deque is needed.

All deque additions retain the existing rule that a bar enters only when both high and low are non-NaN. Empty-half fallback values and invalid current-bar carry behavior remain unchanged.

## Testing

Add a RED-first unit test in `frama.rs` that builds a deterministic finite 4096-row fixture, computes window-200 output through `frama_scalar_deque`, computes an independent direct two-half reference using the same host arithmetic, and requires bit-for-bit equality for every valid output. Before the fix it must fail first at row 297; after the fix the complete row set must pass.

Run the focused unit test, then the existing FRAMA library tests if the focused compile remains lightweight. CUDA/device verification remains deferred to the real RTX run.
