// ============================================================================
// neo_f64_common.cuh — the f64 lane's shared vocabulary
// ============================================================================
//
// Included by every `*_kernel.cu` that carries an f64 entry point alongside its
// legacy f32 ones. It exists so that the three things the f32 lane got wrong
// are spelled ONCE and cannot drift per file:
//
//  1. THE QUIET NaN. The f32 kernels build their NaN as `__int_as_float(
//     0x7fffffff)` — an f32 bit pattern, and not even the CPU's quiet NaN. The
//     CPU allocates through `alloc_with_nan_prefix`, which writes
//     `f64::from_bits(0x7ff8_0000_0000_0000)`. `neo_qnan()` is that exact
//     pattern, so the warmup mask compares cell for cell rather than merely
//     "both are some NaN".
//
//  2. THE SENTINELS. `FLT_MAX` / `FLT_MIN` appear in the f32 kernels where the
//     CPU writes `f64::MAX` / `f64::MIN`. Those are NOT the same number and
//     `FLT_MIN` is not even the same KIND of number (it is the smallest
//     positive normal, while `f64::MIN` is the most negative finite). A running
//     minimum seeded with `FLT_MAX` and compared against real f64 prices is a
//     different algorithm. `NEO_F64_MAX` / `NEO_F64_MIN` are `f64::MAX` and
//     `f64::MIN` respectively.
//
//  3. NaN SEMANTICS. `neo_fmax` / `neo_fmin` are CUDA's `fmax`/`fmin`, which
//     return the non-NaN operand — the semantics of Rust's `f64::max`. Use them
//     ONLY where the CPU reference calls `f64::max`/`f64::min`. Where the CPU
//     writes an `if a < b { b = a }` comparison chain, reproduce the CHAIN:
//     a chain lets a NaN through untouched and that is the CPU's answer.
//
// NO f32 ANYWHERE below this line, and no fast-math intrinsic. Files that
// include this header must be added to `F64_LANE_SOURCES` in build.rs so nvcc
// is invoked with `-prec-div=true -prec-sqrt=true -fmad=false -ftz=false` and
// never with `--use_fast_math`.

#ifndef NEO_F64_COMMON_CUH
#define NEO_F64_COMMON_CUH

#include <math.h>

// `f64::from_bits(0x7ff8_0000_0000_0000)` — the exact quiet NaN the CPU's
// `alloc_with_nan_prefix` writes into the warmup prefix.
__device__ __forceinline__ double neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

// `f64::MAX` and `f64::MIN`. Deliberately spelled as literals rather than as
// `DBL_MAX`/`-DBL_MAX` so the value a reader checks against the Rust source is
// on the page.
#define NEO_F64_MAX (1.7976931348623157e308)
#define NEO_F64_MIN (-1.7976931348623157e308)

// Rust `f64::is_nan`.
__device__ __forceinline__ bool neo_is_nan(double x) {
    return x != x;
}

// Rust `f64::is_finite` — false for both infinities and for NaN. Bit-test
// rather than a comparison chain so a NaN cannot slip through a `<`.
__device__ __forceinline__ bool neo_is_finite(double x) {
    const long long EXP_MASK = 0x7ff0000000000000LL;
    return (__double_as_longlong(x) & EXP_MASK) != EXP_MASK;
}

// Rust `f64::max` / `f64::min`: the non-NaN operand wins.
__device__ __forceinline__ double neo_fmax(double a, double b) {
    return fmax(a, b);
}
__device__ __forceinline__ double neo_fmin(double a, double b) {
    return fmin(a, b);
}

// Fill `[0, warm)` of one output row with the CPU's quiet NaN.
__device__ __forceinline__ void neo_fill_warmup(double* row, int n, int warm) {
    if (warm > n) warm = n;
    for (int i = 0; i < warm; ++i) row[i] = neo_qnan();
}

// Fill an entire row. Used where the CPU's `*_prepare` would have returned an
// error and the caller receives an all-NaN column.
__device__ __forceinline__ void neo_fill_all(double* row, int n) {
    for (int i = 0; i < n; ++i) row[i] = neo_qnan();
}

// The per-thread ring bound shared by every f64 kernel that keeps a rolling
// window in a local array. A period beyond it is REFUSED by the host wrapper
// by name (`CudaF64IndicatorError::PeriodTooLarge`); the kernel never silently
// truncates a window, because a truncated window is a different indicator.
#ifndef NEO_MAX_RING
#define NEO_MAX_RING 512
#endif

#endif // NEO_F64_COMMON_CUH
