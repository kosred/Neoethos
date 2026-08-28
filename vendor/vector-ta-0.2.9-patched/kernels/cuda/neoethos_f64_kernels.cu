// ============================================================================
// NeoEthos f64 indicator lane — kernels/cuda/neoethos_f64_kernels.cu
// ============================================================================
//
// WHY THIS FILE EXISTS
//
// The ten indicators NeoEthos actually sweeps on the card — sma, ema, rsi,
// roc, mom, atr, adx, willr, cci, mfi — had NO double-precision kernel
// anywhere in this crate. Every one of their `*_kernel.cu` files is f32 in and
// f32 out. (`sma_kernel.cu` LOOKS f64 because five of its entry points carry a
// `_f64` suffix, but that suffix names an f64 ACCUMULATOR inside an f32
// pipeline: `sma_prefix_stage1_scan_f64` takes `const float* prices` and
// `sma_batch_from_prefix_f64` writes `float* out`. Counting `_f64` names is
// how this scope was mis-measured twice.)
//
// So the f32→f64 conversion here is not a port of an existing kernel; these
// are new kernels written against the crate's OWN f64 CPU implementations,
// which are the parity oracle:
//
//   sma   src/indicators/moving_averages/sma.rs   sma_scalar
//   ema   src/indicators/moving_averages/ema.rs   ema_scalar_into
//   rsi   src/indicators/rsi.rs                   rsi_compute_into_scalar
//   roc   src/indicators/roc.rs                   roc_scalar
//   mom   src/indicators/mom.rs                   mom_scalar
//   atr   src/indicators/atr.rs                   atr_compute_into_scalar
//   adx   src/indicators/adx.rs                   adx_scalar
//   willr src/indicators/willr.rs                 willr_scalar_naive
//   cci   src/indicators/cci.rs                   cci_scalar
//   mfi   src/indicators/mfi.rs                   mfi_scalar
//
// ----------------------------------------------------------------------------
// THE RULES THIS FILE IS WRITTEN UNDER
// ----------------------------------------------------------------------------
//
// 1. NO f32 ANYWHERE. Not a `float`, not a literal with an `f` suffix, not an
//    `expf`/`sqrtf`/`fmaxf`/`fabsf`/`__fdividef`/`__expf`/`rsqrtf`. Indicator
//    values feed `combined >= long_threshold`, so a one-ULP move flips a
//    trade. `moving_averages/ema_kernel.cu` is the cautionary example: it uses
//    `__fdividef` — the fast approximate divide — INSIDE the EMA recursion.
//
// 2. NO FAST MATH. `build.rs` lists this file in `F64_LANE_SOURCES`, which
//    means it is compiled with `-prec-div=true -prec-sqrt=true -fmad=false
//    -ftz=false` and NEVER with `--use_fast_math`, whatever `CUDA_FAST_MATH`
//    says. This is a correction, not a precaution: the shipped PTX already
//    contains 23 `rcp.approx.ftz.f64` instructions across 17 files, so fast
//    math degrades the crate's EXISTING f64 kernels today.
//
// 3. `-fmad=false` forbids the compiler from CONTRACTING a separate multiply
//    and add into an FMA. It does not disable an explicit `fma()` call. That
//    distinction is load-bearing: wherever the CPU reference uses
//    `f64::mul_add` this file calls `fma()` (fused, one rounding, matching),
//    and wherever the CPU reference writes `a * b + c` this file writes
//    `a * b + c` (two roundings, also matching). Contraction is what would
//    silently turn the second into the first.
//
// 4. ACCUMULATION ORDER IS PART OF THE ANSWER. Every indicator whose CPU
//    reference carries state across bars — ema, rsi, atr, adx, mfi, and the
//    running window sums inside sma and cci — is computed ONE THREAD PER
//    COLUMN, walking bars in ascending order, in exactly the CPU's order. A
//    parallel scan would give a mathematically equal but numerically different
//    answer and is forbidden here.
//
//    The three indicators whose CPU reference is genuinely per-bar independent
//    — roc, mom, willr — are parallel over (combo, bar). cci is split into a
//    sequential pass for its running mean and a parallel pass for the mean
//    absolute deviation, which is an independent fixed-order sum per bar.
//
//    OCCUPANCY CONSEQUENCE, stated rather than hidden: a sweep of K periods
//    launches K threads for the sequential kernels. At K=5 (hpc_ta's
//    [7,21,50,100,200]) that is five CUDA cores doing an O(n) walk each. This
//    lane's win at K=5 is CORRECTNESS (f64) and RESIDENCY (one upload per
//    frame, no device→host→device between indicators), not raw throughput;
//    throughput arrives when K is large, which is the GA lane's shape. Making
//    the sequential kernels wide would require a blocked-scan formulation and
//    a parity argument of its own — it is NOT a free change.
//
// 5. WARMUP IS BIT-EXACT. The CPU allocates via `alloc_with_nan_prefix(len,
//    warm)`, which writes the quiet NaN `0x7ff8000000000000` to `[0, warm)`
//    and leaves `[warm, len)` to the kernel. Every kernel below writes that
//    exact bit pattern (`QNAN`, via `__longlong_as_double`) over `[0, warm)`
//    and a real value at every index from `warm` on, so the NaN mask can be
//    compared cell for cell rather than merely "both are some NaN".
//
// ----------------------------------------------------------------------------
// CALLING CONVENTION
// ----------------------------------------------------------------------------
//
// Output is row-major `[n_combos][n]`: row `r` is the series for
// `periods[r]`. `first_valid` is computed on the HOST with the same rule the
// corresponding CPU `*_prepare` uses (first index whose inputs are all
// non-NaN) so the two sides cannot disagree about where the series begins.
//
// The period list is EXPLICIT, not a (start, end, step) range. The periods
// this codebase wants — [7, 21, 50, 100, 200] — are not an arithmetic
// progression, and the range API's only batched form would be 7..=200 step 1:
// 194 rows to keep 5, a 654 MB device allocation at 843k bars to keep 17 MB,
// with the size a function of the requested RANGE rather than of the hardware.
// An explicit list keeps peak memory at rows*cols and computes nothing that is
// thrown away.

#include <math.h>

// The exact quiet-NaN the CPU writes: `f64::from_bits(0x7ff8_0000_0000_0000)`.
__device__ __forceinline__ double neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

// Mirrors `ema.rs::is_finite_fast` — true for finite, false for inf and NaN.
__device__ __forceinline__ bool neo_is_finite(double x) {
    const long long EXP_MASK = 0x7ff0000000000000LL;
    return (__double_as_longlong(x) & EXP_MASK) != EXP_MASK;
}

// Fill `[0, warm)` of one output row with the CPU's quiet NaN.
__device__ __forceinline__ void neo_fill_warmup(double* row, int n, int warm) {
    if (warm > n) warm = n;
    for (int i = 0; i < warm; ++i) row[i] = neo_qnan();
}

// Exact operation-for-operation counterpart of sma.rs::compensated_add.
// `-fmad=false` is mandatory for this translation unit: contracting either
// subtraction/addition would change the correction term and therefore the
// SMA's semantic result.
__device__ __forceinline__ void neo_compensated_add(
    double value,
    double* sum,
    double* correction)
{
    const double adjusted = value - *correction;
    const double next = *sum + adjusted;
    *correction = (next - *sum) - adjusted;
    *sum = next;
}

// ============================================================================
// SMA — reference: sma.rs::sma_scalar
//   warm = first_valid + period - 1
//   period == 1 is a copy; otherwise a Kahan-compensated running window sum
//   seeded by a forward sum over [first, first+period).  Eviction and insertion
//   are deliberately TWO compensated operations; combining them first loses
//   the low bits before the correction can observe them.
//   The running sum is a cross-bar accumulation, so this is one thread per
//   column even though "an SMA" sounds embarrassingly parallel: recomputing
//   each window independently would give a DIFFERENT f64 value.
// ============================================================================
extern "C" __global__ void neoethos_sma_batch_f64(
    const double* __restrict__ prices,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    const int period = periods[r];
    double* row = out + (size_t)r * (size_t)n;
    if (period <= 0 || first_valid < 0 || first_valid >= n) {
        neo_fill_warmup(row, n, n);
        return;
    }

    if (period == 1) {
        neo_fill_warmup(row, n, first_valid);
        for (int i = first_valid; i < n; ++i) row[i] = prices[i];
        return;
    }

    const int warm = first_valid + period - 1;
    neo_fill_warmup(row, n, warm);
    if (warm >= n) return;

    double sum = 0.0;
    double correction = 0.0;
    for (int k = 0; k < period; ++k) {
        neo_compensated_add(prices[first_valid + k], &sum, &correction);
    }
    const double inv = 1.0 / (double)period;

    row[warm] = sum * inv;
    for (int i = first_valid + period; i < n; ++i) {
        neo_compensated_add(-prices[i - period], &sum, &correction);
        neo_compensated_add(prices[i], &sum, &correction);
        row[i] = sum * inv;
    }
}

// ============================================================================
// EMA — reference: ema.rs::ema_scalar_into
//   warm = first_valid  (only the pre-series is NaN; the seed bar is a value)
//   Two phases, both recursive:
//     [first+1, first+period) : running arithmetic mean over FINITE values
//                               only, `mean = ((vc-1)*mean + x) / vc`
//     [first+period, n)       : `prev = fma(beta, prev, alpha * x)`, which is
//                               the CPU's `beta.mul_add(prev, alpha * x)`
//   Non-finite bars carry the previous value forward WITHOUT updating state.
//   alpha = 2 / (period + 1); beta = 1 - alpha.
// ============================================================================
extern "C" __global__ void neoethos_ema_batch_f64(
    const double* __restrict__ prices,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    const int period = periods[r];
    double* row = out + (size_t)r * (size_t)n;
    if (period <= 0 || first_valid < 0 || first_valid >= n) {
        neo_fill_warmup(row, n, n);
        return;
    }

    neo_fill_warmup(row, n, first_valid);

    const double alpha = 2.0 / ((double)period + 1.0);
    const double beta = 1.0 - alpha;

    double mean = prices[first_valid];
    row[first_valid] = mean;
    long long valid_count = 1;

    int warmup_end = first_valid + period;
    if (warmup_end > n) warmup_end = n;

    for (int i = first_valid + 1; i < warmup_end; ++i) {
        const double x = prices[i];
        if (neo_is_finite(x)) {
            valid_count += 1;
            const double vc = (double)valid_count;
            mean = ((vc - 1.0) * mean + x) / vc;
        }
        row[i] = mean;
    }

    if (warmup_end < n) {
        double prev = mean;
        for (int i = warmup_end; i < n; ++i) {
            const double x = prices[i];
            if (neo_is_finite(x)) {
                prev = fma(beta, prev, alpha * x);
            }
            row[i] = prev;
        }
    }
}

// ============================================================================
// RSI — reference: rsi.rs::rsi_compute_into_scalar
//   warm = first_valid + period
//   Wilder smoothing: `avg = fma(avg, beta, inv_p * delta)`, matching the
//   CPU's `avg_gain.mul_add(beta, inv_p * g1)`. The CPU unrolls the steady
//   loop two at a time; unrolling does not change the per-index arithmetic or
//   its order, so the plain loop below is bit-identical to it.
//   A non-finite delta during seeding poisons both averages with NaN, exactly
//   as the CPU's `has_nan` branch does, and the NaN then propagates.
// ============================================================================
extern "C" __global__ void neoethos_rsi_batch_f64(
    const double* __restrict__ prices,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    const int period = periods[r];
    double* row = out + (size_t)r * (size_t)n;
    if (period <= 0 || first_valid < 0 || first_valid >= n) {
        neo_fill_warmup(row, n, n);
        return;
    }

    const int warm = first_valid + period;
    neo_fill_warmup(row, n, warm);
    if (warm >= n) return;

    const double inv_p = 1.0 / (double)period;
    const double beta = 1.0 - inv_p;

    double avg_gain = 0.0;
    double avg_loss = 0.0;
    bool has_nan = false;

    // `warm_last = min(first + period, len - 1)` on the CPU.
    int warm_last = first_valid + period;
    if (warm_last > n - 1) warm_last = n - 1;

    for (int i = first_valid + 1; i <= warm_last; ++i) {
        const double delta = prices[i] - prices[i - 1];
        if (!isfinite(delta)) { has_nan = true; break; }
        if (delta > 0.0) avg_gain += delta;
        else if (delta < 0.0) avg_loss -= delta;
    }

    const int idx0 = first_valid + period;
    if (has_nan) {
        avg_gain = neo_qnan();
        avg_loss = neo_qnan();
        if (idx0 < n) row[idx0] = neo_qnan();
    } else {
        avg_gain *= inv_p;
        avg_loss *= inv_p;
        if (idx0 < n) {
            const double denom = avg_gain + avg_loss;
            row[idx0] = (denom == 0.0) ? 50.0 : (100.0 * avg_gain / denom);
        }
    }

    for (int j = idx0 + 1; j < n; ++j) {
        const double d = prices[j] - prices[j - 1];
        const double g = (d > 0.0) ? d : 0.0;
        const double l = (d < 0.0) ? -d : 0.0;
        avg_gain = fma(avg_gain, beta, inv_p * g);
        avg_loss = fma(avg_loss, beta, inv_p * l);
        const double denom = avg_gain + avg_loss;
        row[j] = (denom == 0.0) ? 50.0 : (100.0 * avg_gain / denom);
    }
}

// ============================================================================
// ROC — reference: roc.rs::roc_scalar
//   warm = first_valid + period
//   Per-bar independent: `(c / p).mul_add(100.0, -100.0)`, and a zero or NaN
//   previous price yields 0.0 rather than an infinity. Parallel over
//   (combo, bar) because there is no cross-bar state at all.
// ============================================================================
extern "C" __global__ void neoethos_roc_batch_f64(
    const double* __restrict__ prices,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.y;
    if (r >= n_combos) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int period = periods[r];
    double* row = out + (size_t)r * (size_t)n;

    if (period <= 0 || first_valid < 0) { row[i] = neo_qnan(); return; }

    const int warm = first_valid + period;
    if (i < warm) { row[i] = neo_qnan(); return; }

    const double c = prices[i];
    const double p = prices[i - period];
    row[i] = (p == 0.0 || isnan(p)) ? 0.0 : fma(c / p, 100.0, -100.0);
}

// ============================================================================
// MOM — reference: mom.rs::mom_scalar
//   warm = first_valid + period; out[i] = d[i] - d[i-period]. Per-bar.
// ============================================================================
extern "C" __global__ void neoethos_mom_batch_f64(
    const double* __restrict__ prices,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.y;
    if (r >= n_combos) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int period = periods[r];
    double* row = out + (size_t)r * (size_t)n;

    if (period <= 0 || first_valid < 0) { row[i] = neo_qnan(); return; }

    const int warm = first_valid + period;
    if (i < warm) { row[i] = neo_qnan(); return; }

    row[i] = prices[i] - prices[i - period];
}

// ============================================================================
// ATR — reference: atr.rs::atr_compute_into_scalar
//   warm = first_valid + length - 1
//   Seed: sum of true range over [first, warm], divided by length.
//   Steady: `rma = fma(-alpha, rma, rma) + alpha * tr`, which is the CPU's
//   `(-alpha).mul_add(rma, rma) + alpha * tr` — NOT the algebraically equal
//   `rma * (1 - alpha) + alpha * tr`. Writing it the "obvious" way changes the
//   rounding at every bar and compounds across the whole series.
//   True range uses `fmax`-free explicit comparisons in the CPU; `if (hc > tr)
//   tr = hc;` reproduces that including NaN behaviour (a NaN comparison is
//   false, so tr keeps its earlier value — `fmax` would instead return the
//   non-NaN operand and diverge).
// ============================================================================
extern "C" __global__ void neoethos_atr_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    const int length = periods[r];
    double* row = out + (size_t)r * (size_t)n;
    if (length <= 0 || first_valid < 0 || first_valid >= n) {
        neo_fill_warmup(row, n, n);
        return;
    }

    const int warm = first_valid + length - 1;
    neo_fill_warmup(row, n, warm);
    if (warm >= n) return;

    const double alpha = 1.0 / (double)length;

    double sum_tr = high[first_valid] - low[first_valid];

    if (warm > first_valid) {
        double prev_c = close[first_valid];
        for (int i = first_valid + 1; i <= warm; ++i) {
            const double hi = high[i];
            const double lo = low[i];
            double tr = hi - lo;
            const double hc = fabs(hi - prev_c);
            if (hc > tr) tr = hc;
            const double lc = fabs(lo - prev_c);
            if (lc > tr) tr = lc;
            sum_tr += tr;
            prev_c = close[i];
        }
    }

    double rma = sum_tr / (double)length;
    row[warm] = rma;

    double prev_c = (warm + 1 > 0) ? close[warm] : close[0];
    for (int i = warm + 1; i < n; ++i) {
        const double hi = high[i];
        const double lo = low[i];
        double tr = hi - lo;
        const double hc = fabs(hi - prev_c);
        if (hc > tr) tr = hc;
        const double lc = fabs(lo - prev_c);
        if (lc > tr) tr = lc;
        rma = fma(-alpha, rma, rma) + alpha * tr;
        row[i] = rma;
        prev_c = close[i];
    }
}

// ============================================================================
// ADX — reference: adx.rs::adx_scalar, which the CPU calls on slices already
//   offset by `first_valid`, so every index below is absolute = first + r.
//   warm = first_valid + 2*period - 1, and that is EXACTLY the first index the
//   algorithm writes (dx_count reaches `period` at relative 2*period-1), so
//   the NaN prefix and the written region are contiguous with no gap.
//   Smoothing here is plain `atr * one_minus_rp + tr` — the CPU does NOT use
//   mul_add for adx, so neither does this, and `-fmad=false` keeps the
//   compiler from fusing it behind our back.
//
//   TRUE RANGE IS THE OPPOSITE CONVENTION FROM `atr`, and the two live 70 lines
//   apart. `adx.rs:339` and `adx.rs:390` both write `hl.max(hpc).max(lpc)`,
//   i.e. Rust `f64::max`, which returns the NON-NaN operand — so `fmax` is
//   correct here. `atr.rs:344-351` instead writes `if hc > tr { tr = hc; }`,
//   where a NaN comparison is false and the earlier value survives, so the
//   explicit `if` chain is correct THERE. Same numbers on clean bars, different
//   numbers the moment a bar carries a NaN high or low — and this store has
//   already shipped 14,240 corrupt bars, so that is not hypothetical. Carrying
//   `atr`'s convention into `adx` leaves `tr` NaN, which poisons `atr`,
//   `plus_di`, `minus_di`, `dx` and `last_adx` for EVERY REMAINING BAR while
//   the CPU carries on with a finite number.
// ============================================================================
extern "C" __global__ void neoethos_adx_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    const int period = periods[r];
    double* row = out + (size_t)r * (size_t)n;
    if (period <= 0 || first_valid < 0 || first_valid >= n) {
        neo_fill_warmup(row, n, n);
        return;
    }

    const int warm = first_valid + 2 * period - 1;
    neo_fill_warmup(row, n, warm);

    const int len_rel = n - first_valid;
    if (len_rel <= period) return;

    const double period_f = (double)period;
    const double rp = 1.0 / period_f;
    const double one_minus_rp = 1.0 - rp;
    const double period_minus_one = period_f - 1.0;

    double tr_sum = 0.0;
    double plus_dm_sum = 0.0;
    double minus_dm_sum = 0.0;

    double prev_h = high[first_valid];
    double prev_l = low[first_valid];
    double prev_c = close[first_valid];

    for (int rel = 1; rel <= period; ++rel) {
        const int i = first_valid + rel;
        const double ch = high[i];
        const double cl = low[i];

        const double hl = ch - cl;
        const double hpc = fabs(ch - prev_c);
        const double lpc = fabs(cl - prev_c);
        // adx.rs:339 `hl.max(hpc).max(lpc)` — f64::max, NOT an if-chain.
        const double tr = fmax(fmax(hl, hpc), lpc);

        const double up = ch - prev_h;
        const double down = prev_l - cl;
        if (up > down && up > 0.0) plus_dm_sum += up;
        if (down > up && down > 0.0) minus_dm_sum += down;
        tr_sum += tr;

        prev_h = ch;
        prev_l = cl;
        prev_c = close[i];
    }

    double atr = tr_sum;
    double plus_dm_smooth = plus_dm_sum;
    double minus_dm_smooth = minus_dm_sum;

    double plus_di_prev = 0.0;
    double minus_di_prev = 0.0;
    if (atr != 0.0) {
        plus_di_prev = (plus_dm_smooth / atr) * 100.0;
        minus_di_prev = (minus_dm_smooth / atr) * 100.0;
    }
    const double sum_di_prev = plus_di_prev + minus_di_prev;
    double dx_sum = (sum_di_prev != 0.0)
        ? (fabs(plus_di_prev - minus_di_prev) / sum_di_prev) * 100.0
        : 0.0;
    int dx_count = 1;
    double last_adx = 0.0;

    prev_h = high[first_valid + period];
    prev_l = low[first_valid + period];
    prev_c = close[first_valid + period];

    for (int rel = period + 1; rel < len_rel; ++rel) {
        const int i = first_valid + rel;
        const double ch = high[i];
        const double cl = low[i];

        const double hl = ch - cl;
        const double hpc = fabs(ch - prev_c);
        const double lpc = fabs(cl - prev_c);
        // adx.rs:390 `hl.max(hpc).max(lpc)` — f64::max, NOT an if-chain.
        const double tr = fmax(fmax(hl, hpc), lpc);

        const double up = ch - prev_h;
        const double down = prev_l - cl;
        const double plus_dm = (up > down && up > 0.0) ? up : 0.0;
        const double minus_dm = (down > up && down > 0.0) ? down : 0.0;

        atr = atr * one_minus_rp + tr;
        plus_dm_smooth = plus_dm_smooth * one_minus_rp + plus_dm;
        minus_dm_smooth = minus_dm_smooth * one_minus_rp + minus_dm;

        double plus_di = 0.0;
        double minus_di = 0.0;
        if (atr != 0.0) {
            plus_di = (plus_dm_smooth / atr) * 100.0;
            minus_di = (minus_dm_smooth / atr) * 100.0;
        }
        const double sum_di = plus_di + minus_di;
        const double dx = (sum_di != 0.0)
            ? (fabs(plus_di - minus_di) / sum_di) * 100.0
            : 0.0;

        if (dx_count < period) {
            dx_sum += dx;
            dx_count += 1;
            if (dx_count == period) {
                last_adx = dx_sum * rp;
                row[i] = last_adx;
            }
        } else {
            last_adx = (last_adx * period_minus_one + dx) * rp;
            row[i] = last_adx;
        }

        prev_h = ch;
        prev_l = cl;
        prev_c = close[i];
    }
}

// ============================================================================
// WILLR — reference: willr.rs::willr_scalar_naive
//   warm = first_valid + period - 1
//   Per-bar independent: max/min over the trailing window are exact, so the
//   deque form the CPU may pick and this recomputation agree bit for bit.
//   NaN handling is explicit and ORDERED: a NaN close short-circuits first,
//   then any NaN in the window, then a window that never updated h or l.
// ============================================================================
extern "C" __global__ void neoethos_willr_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.y;
    if (r >= n_combos) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int period = periods[r];
    double* row = out + (size_t)r * (size_t)n;

    if (period <= 0 || first_valid < 0) { row[i] = neo_qnan(); return; }

    const int warm = first_valid + period - 1;
    if (i < warm) { row[i] = neo_qnan(); return; }

    if (isnan(close[i])) { row[i] = neo_qnan(); return; }

    const int start = i + 1 - period;
    double h = -INFINITY;
    double l = INFINITY;
    bool has_nan = false;
    for (int j = start; j <= i; ++j) {
        const double hj = high[j];
        const double lj = low[j];
        if (isnan(hj) || isnan(lj)) { has_nan = true; break; }
        if (hj > h) h = hj;
        if (lj < l) l = lj;
    }

    if (has_nan || isinf(h) || isinf(l)) { row[i] = neo_qnan(); return; }

    const double denom = h - l;
    row[i] = (denom == 0.0) ? 0.0 : ((h - close[i]) / denom * -100.0);
}

// ============================================================================
// CCI — reference: cci.rs::cci_scalar (the general, non-14 path)
//   warm = first_valid + period - 1
//
//   Two kernels, because the reference has two different shapes inside it:
//
//   PASS 1 (`neoethos_cci_sma_f64`, one thread per column) reproduces the
//   RUNNING window sum exactly: seeded by a forward sum over
//   [first, first+period), then `sum = sum - d[i-period] + d[i]` in the CPU's
//   exact order and with the CPU's exact `sum * inv_p` rounding. This carries
//   state across bars, so it cannot be parallelised without changing the
//   answer. It writes the per-bar SMA into a scratch matrix.
//
//   PASS 2 (`neoethos_cci_from_sma_f64`, parallel over (combo, bar)) computes
//   the mean absolute deviation, which IS per-bar independent: the CPU
//   recomputes `sabs` from scratch over [i+1-period, i+1) every bar, ascending,
//   and this does the same in the same order. Splitting the work this way
//   keeps the expensive O(period) term parallel while leaving the numerically
//   fragile term sequential.
//
//   Peak memory: one extra rows*cols f64 scratch, i.e. exactly 2x the output
//   the caller already sized. Bounded by the same host-side chunking that
//   bounds the output, so it stays a function of hardware, not of parameters.
// ============================================================================
extern "C" __global__ void neoethos_cci_sma_f64(
    const double* __restrict__ prices,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ sma_out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    const int period = periods[r];
    double* row = sma_out + (size_t)r * (size_t)n;
    if (period <= 0 || first_valid < 0 || first_valid >= n) {
        neo_fill_warmup(row, n, n);
        return;
    }

    const int first_out = first_valid + period - 1;
    neo_fill_warmup(row, n, first_out);
    if (first_out >= n) return;

    const double inv_p = 1.0 / (double)period;

    double sum = 0.0;
    for (int k = first_valid; k < first_valid + period; ++k) sum += prices[k];
    row[first_out] = sum * inv_p;

    for (int i = first_out + 1; i < n; ++i) {
        sum = sum - prices[i - period] + prices[i];
        row[i] = sum * inv_p;
    }
}

extern "C" __global__ void neoethos_cci_from_sma_f64(
    const double* __restrict__ prices,
    const double* __restrict__ sma_in,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.y;
    if (r >= n_combos) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int period = periods[r];
    double* row = out + (size_t)r * (size_t)n;
    const double* sma_row = sma_in + (size_t)r * (size_t)n;

    if (period <= 0 || first_valid < 0) { row[i] = neo_qnan(); return; }

    const int first_out = first_valid + period - 1;
    if (i < first_out) { row[i] = neo_qnan(); return; }

    const double sma = sma_row[i];

    // The CPU sums the first window over [first, first+period) and every later
    // window over [i+1-period, i+1); for i == first_out those are the same
    // range, so one expression serves both.
    const int wstart = i + 1 - period;
    double sabs = 0.0;
    for (int j = wstart; j < i + 1; ++j) sabs += fabs(prices[j] - sma);

    const double denom = 0.015 * (sabs * (1.0 / (double)period));
    row[i] = (denom == 0.0) ? 0.0 : ((prices[i] - sma) / denom);
}

// ============================================================================
// MFI — reference: mfi.rs::mfi_scalar
//   warm = first_valid + period - 1
//   `typical_price` is the CPU's SOURCE series, which for the registry default
//   is hlc3 — NOT close. Feeding close computes a different indicator, not a
//   less precise one, so the host must upload hlc3 and pass it here.
//   The positive/negative money-flow ring buffer is a cross-bar accumulation
//   (`pos_sum -= old; pos_sum += new`), so this is one thread per column and
//   the ring is kept per-thread in a compile-time-bounded local array.
//
//   NEVER-OOM: `MFI_MAX_PERIOD` bounds the per-thread ring at a fixed size
//   chosen from the hardware's register/local budget, NOT from the caller's
//   period. A period beyond it is REFUSED by the host wrapper with a named
//   error rather than silently truncated or silently moved to the CPU.
// ============================================================================
#define MFI_MAX_PERIOD 512

extern "C" __global__ void neoethos_mfi_batch_f64(
    const double* __restrict__ typical_price,
    const double* __restrict__ volume,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    const int period = periods[r];
    double* row = out + (size_t)r * (size_t)n;

    if (period <= 0 || period > MFI_MAX_PERIOD || first_valid < 0 || first_valid >= n) {
        // The host refuses this before launching; writing an all-NaN row keeps
        // the buffer defined if it ever happens anyway.
        neo_fill_warmup(row, n, n);
        return;
    }

    const int warm = first_valid + period - 1;
    neo_fill_warmup(row, n, warm);
    if (warm >= n) return;

    double pos_ring[MFI_MAX_PERIOD];
    double neg_ring[MFI_MAX_PERIOD];
    for (int k = 0; k < period; ++k) { pos_ring[k] = 0.0; neg_ring[k] = 0.0; }

    double pos_sum = 0.0;
    double neg_sum = 0.0;
    double prev = typical_price[first_valid];
    int ring = 0;

    const int seed_start = first_valid + 1;
    const int seed_end = first_valid + period;
    for (int i = seed_start; i < seed_end; ++i) {
        const double tp_i = typical_price[i];
        const double flow = tp_i * volume[i];
        const double diff = tp_i - prev;
        prev = tp_i;

        const double gt = (diff > 0.0) ? 1.0 : 0.0;
        const double lt = (diff < 0.0) ? 1.0 : 0.0;
        const double pos_new = flow * gt;
        const double neg_new = flow * lt;

        pos_ring[ring] = pos_new;
        neg_ring[ring] = neg_new;
        pos_sum += pos_new;
        neg_sum += neg_new;

        ring += 1;
        if (ring == period) ring = 0;
    }

    const int idx0 = seed_end - 1;
    if (idx0 < n) {
        const double total = pos_sum + neg_sum;
        row[idx0] = (total < 1e-14) ? 0.0 : (100.0 * (pos_sum / total));
    }

    for (int i = seed_end; i < n; ++i) {
        pos_sum -= pos_ring[ring];
        neg_sum -= neg_ring[ring];

        const double tp_i = typical_price[i];
        const double flow = tp_i * volume[i];
        const double diff = tp_i - prev;
        prev = tp_i;

        const double gt = (diff > 0.0) ? 1.0 : 0.0;
        const double lt = (diff < 0.0) ? 1.0 : 0.0;
        const double pos_new = flow * gt;
        const double neg_new = flow * lt;

        pos_ring[ring] = pos_new;
        neg_ring[ring] = neg_new;
        pos_sum += pos_new;
        neg_sum += neg_new;

        const double total = pos_sum + neg_sum;
        row[i] = (total < 1e-14) ? 0.0 : (100.0 * (pos_sum / total));

        ring += 1;
        if (ring == period) ring = 0;
    }
}

// ############################################################################
// ############################################################################
//
//   SECOND BATCH — twenty more f64 entry points
//
// ############################################################################
// ############################################################################
//
// HOW THIS BATCH WAS CHOSEN, because "which twenty of the missing ones" is the
// whole question and a wrong answer here wastes every line below it.
//
// 179 of the 342 ids in `ALL_INDICATORS` have a `.cu` with no `_f64` entry
// point at all. Writing kernels in that order would be writing kernels for
// columns this system never sees. The set that MATTERS is the set that is
// actually REACHABLE, and that was measured by running the CPU dispatcher over
// all 342 ids exactly the way `hpc_ta` calls it — `output_id: None`,
// `params: &[]`:
//
//     232 of 342 succeed. 110 return Err or panic and are silently dropped.
//
// The 110 are dropped because `resolve_output_id` (dispatch/cpu_batch.rs:2185)
// returns `Err(InvalidParam { "output_id is required for multi-output
// indicators" })` whenever the registry declares more than one output, and
// `hpc_ta.rs:291` swallows that with `if let Ok(output) = compute_result`. So
// EVERY multi-output indicator contributes zero columns today.
//
// That measurement inverts the priority list this task started from.
// `bollinger_bands`, `donchian`, `aroon`, `di`, `dm`, `chandelier_exit`,
// `devstop` and `emd` are all multi-output, therefore all currently
// unreachable, therefore an f64 kernel for any of them would have no CPU column
// to be parity-checked against. They are NOT in this batch, and the reason is a
// measurement rather than a preference.
//
// The sharpest consequence is in `hpc_ta::MULTI_PERIOD_IDS`, the 18-indicator
// period sweep. Five of those 18 — stoch, macd, bollinger_bands, keltner,
// supertrend — are multi-output and emit nothing. Thirteen are reachable. Ten
// of the thirteen already had f64 kernels. The remaining three were `tsi`,
// `obv` and `vwap`, and they are the first three kernels below: with them the
// reachable sweep is 13 of 13 on the card and a frame never leaves the device.
//
// ----------------------------------------------------------------------------
// TWO PARITY HAZARDS THIS BATCH INTRODUCES THAT THE FIRST ONE DID NOT
// ----------------------------------------------------------------------------
//
// 1. `Kernel::Auto` IS NOT `Kernel::Scalar` IN THIS CRATE. Every kernel below
//    is written against a named `*_scalar` reference, but `hpc_ta` calls with
//    `Kernel::Auto`, and on x86_64 with the `nightly-avx` feature (which
//    `neoethos-data` enables) several indicators resolve Auto to `Avx2` or
//    `Avx512` instead — `obv_with_kernel` (obv.rs:164) maps `Auto => Avx2`
//    outright. An AVX kernel that reassociates a reduction is a DIFFERENT f64
//    number from its own scalar sibling, and no amount of care in a .cu file
//    can fix that.
//
//    This is not left for the card to discover. `gpu_indicators.rs` carries
//    `scalar_and_auto_agree_for_every_claimed_indicator`, which runs WITHOUT a
//    GPU and asserts bit-equality of `Kernel::Scalar` against `Kernel::Auto`
//    for every id registered in `F64_KERNELS`. If that test fails, the named
//    indicator comes OUT of the table until the divergence is explained — it is
//    not the kernel below that is wrong in that case.
//
// 2. TWO OF THE FIRST THREE IGNORE `period`, AND THAT IS FAITHFUL, NOT A BUG.
//    `compute_obv_batch` (cpu_batch.rs:3897) takes `|_params|` and builds
//    `ObvParams::default()`. `vwap` is anchored by calendar bucket and its only
//    parameter is `anchor`. So `obv_7 .. obv_200` are five byte-identical CPU
//    columns, and the kernel must produce five byte-identical rows. Computing
//    "obv with period 7" would be a different indicator and would fail parity
//    for a reason that looks like rounding and is not. Each such kernel is
//    marked PERIOD-INVARIANT and its thread computes the whole series ignoring
//    `periods[r]`. TSI is different: although vector-ta names its parameters
//    `long_period` and `short_period`, `hpc_ta` deliberately maps the requested
//    sweep anchor to that named pair, so its kernel must consume `periods[r]`.
//    `medprice` and `wclprice` are period-invariant for the simpler reason that
//    they have no period parameter at all.
//
// Everything in the first batch's header still binds here: no `float`, no `f`
// suffix, no f32 intrinsic, no fast math, `fma()` exactly where and only where
// the CPU writes `mul_add`, and the CPU's accumulation order preserved bar for
// bar. Grouping is part of that order: `wilders_scalar` seeds with
// `sum += *p0 + *p1 + *p2 + *p3`, which rounds differently from four separate
// `sum +=`, so `neoethos_wilders_batch_f64` groups by four too — while
// `smma_scalar`, its close cousin, sums one at a time and so does its kernel.

// The CPU's `f64::MIN` / `f64::MAX` window sentinels. `midpoint_scalar` seeds
// its window scan with these rather than with infinities, so an all-NaN window
// yields `(MIN + MAX) / 2` and not a NaN. Reproduced exactly, because the
// sentinel leaking into the answer is a real observable value.
#define NEO_F64_MIN (-1.7976931348623157e308)
#define NEO_F64_MAX (1.7976931348623157e308)

// Per-thread ring bound for `adxr`, whose CPU reference keeps `period` past ADX
// values. Same contract as `MFI_MAX_PERIOD`: a property of the KERNEL, not of
// the caller, and a larger period is refused by name on the host rather than
// truncated here.
#define ADXR_MAX_PERIOD 512

// NeoEthos uses the requested period as TSI's long-window anchor and scales
// the short window with the official/default 25:13 relationship.  Keep the
// defaults named here because they define that ratio; they are not hard-coded
// execution windows.
#define TSI_DEFAULT_LONG 25
#define TSI_DEFAULT_SHORT 13

// ============================================================================
// TSI — references: tsi.rs::tsi_scalar_classic and tsi_compute_into_inline
//   `periods[r]` is the named `long_period`; `short_period` is the same rounded
//   25:13 scaling performed by hpc_ta::sweep_params.  The default row (25/13)
//   takes `tsi_scalar_classic`; every other CPU row takes
//   `tsi_compute_into_inline`.  Both use the recurrence reproduced below.
//   Both EMA pairs are plain `alpha * x + (1 - alpha) * prev`; the CPU does NOT
//   use mul_add here, so neither does this, and `-fmad=false` is what stops
//   nvcc contracting it behind our back.
//   A non-finite bar writes NaN and CONTINUES without advancing `prev` or any
//   accumulator — an asymmetry with ema and efi, which carry state forward.
// ============================================================================
extern "C" __global__ void neoethos_tsi_batch_f64(
    const double* __restrict__ prices,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    const int long_period = periods[r];
    // For positive i32 periods this integer expression is exactly
    // round(long_period * 13.0 / 25.0): denominator 25 is odd, so a half tie
    // cannot occur.  Widen before multiplying to make overflow impossible.
    long long scaled_short =
        ((long long)long_period * TSI_DEFAULT_SHORT + (TSI_DEFAULT_LONG / 2)) /
        TSI_DEFAULT_LONG;
    const int short_period = (int)(scaled_short < 1 ? 1 : scaled_short);

    double* row = out + (size_t)r * (size_t)n;
    if (first_valid < 0 || first_valid >= n) { neo_fill_warmup(row, n, n); return; }

    const long long warm64 =
        (long long)first_valid + (long long)long_period + (long long)short_period;
    const int warm = warm64 < (long long)n ? (int)warm64 : n;
    neo_fill_warmup(row, n, warm);

    if (first_valid + 1 >= n) return;

    const double long_alpha = 2.0 / ((double)long_period + 1.0);
    const double short_alpha = 2.0 / ((double)short_period + 1.0);
    const double long_1minus = 1.0 - long_alpha;
    const double short_1minus = 1.0 - short_alpha;

    double prev = prices[first_valid];
    const double x1 = prices[first_valid + 1];
    if (!neo_is_finite(x1)) return;

    const double first_momentum = x1 - prev;
    prev = x1;

    double ema_long_num = first_momentum;
    double ema_short_num = first_momentum;
    double ema_long_den = fabs(first_momentum);
    double ema_short_den = fabs(first_momentum);

    for (int i = first_valid + 2; i < n; ++i) {
        const double cur = prices[i];
        if (!neo_is_finite(cur)) {
            row[i] = neo_qnan();
            continue;
        }
        const double momentum = cur - prev;
        prev = cur;

        ema_long_num  = long_alpha  * momentum       + long_1minus  * ema_long_num;
        ema_short_num = short_alpha * ema_long_num   + short_1minus * ema_short_num;
        ema_long_den  = long_alpha  * fabs(momentum) + long_1minus  * ema_long_den;
        ema_short_den = short_alpha * ema_long_den   + short_1minus * ema_short_den;

        if (i >= warm) {
            if (ema_short_den == 0.0) {
                row[i] = neo_qnan();
            } else {
                double v = 100.0 * (ema_short_num / ema_short_den);
                // Rust's `f64::clamp`, which is an ordered pair of comparisons
                // — NOT fmin/fmax, whose NaN handling differs.
                if (v < -100.0) v = -100.0;
                else if (v > 100.0) v = 100.0;
                row[i] = v;
            }
        }
    }
}

// ============================================================================
// OBV — official TA-Lib ta_OBV.c semantics
//   PERIOD-INVARIANT and lookback zero. The first output is the first valid
//   volume. Every later bar uses a strict sequential add/subtract, while flat
//   or unordered (NaN) close comparisons carry the accumulator. A prefix scan,
//   sign multiplication, or FMA would change IEEE-754 and edge-case semantics.
// ============================================================================
extern "C" __global__ void neoethos_obv_batch_f64(
    const double* __restrict__ close,
    const double* __restrict__ volume,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;
    (void)periods;  // PERIOD-INVARIANT: see the batch header.

    double* row = out + (size_t)r * (size_t)n;
    if (first_valid < 0 || first_valid >= n) { neo_fill_warmup(row, n, n); return; }

    neo_fill_warmup(row, n, first_valid);

    double prev_obv = volume[first_valid];
    double prev_close = close[first_valid];
    row[first_valid] = prev_obv;

    for (int i = first_valid + 1; i < n; ++i) {
        const double c = close[i];
        const double v = volume[i];
        if (c > prev_close) {
            prev_obv += v;
        } else if (c < prev_close) {
            prev_obv -= v;
        }
        row[i] = prev_obv;
        prev_close = c;
    }
}

// ============================================================================
// VWAP — reference: moving_averages/vwap.rs::vwap_scalar with anchor "1d",
//   i.e. count = 1, unit_char = 'd'.
//   PERIOD-INVARIANT, and the only kernel in this crate's f64 lane that reads a
//   non-f64 array.
//   warm = 0: `vwap_with_kernel` calls `alloc_with_nan_prefix(n, 0)`, so there
//   is NO NaN prefix — every bar carries a value, or NaN where the running
//   volume is not positive. `first_valid` is deliberately unused.
//
//   The anchor is a CALENDAR BUCKET, not a rolling window: `gid = ts /
//   bucket_ms` with `bucket_ms = 86_400_000`, and the accumulators reset when
//   the bucket changes. The CPU's two branches are reproduced exactly and they
//   are NOT equivalent:
//     ts >= 0 : reset when `ts >= next_cutoff || ts < window_start`
//     ts <  0 : reset when `gid != current_group_id`
//   Integer division truncates toward zero in both C and Rust, so bucket -0
//   and bucket +0 collide for negative timestamps — that is why the CPU needs
//   the second branch, and why collapsing the two would be wrong for any
//   pre-1970 bar.
//
//   `vol_price_sum = p.mul_add(v, vol_price_sum)` -> `fma`; `volume_sum += v`
//   stays a plain add.
// ============================================================================
extern "C" __global__ void neoethos_vwap_batch_f64(
    const long long* __restrict__ timestamps,
    const double* __restrict__ prices,
    const double* __restrict__ volume,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;
    (void)periods;      // PERIOD-INVARIANT: see the batch header.
    (void)first_valid;  // vwap has no warmup prefix — alloc_with_nan_prefix(n, 0).

    double* row = out + (size_t)r * (size_t)n;
    if (n <= 0) return;

    const long long DAY_MS = 86400000LL;
    const long long bucket_ms = DAY_MS;

    const long long I64_MIN = -9223372036854775807LL - 1LL;
    long long current_group_id = I64_MIN;
    long long window_start = 0;
    long long next_cutoff = I64_MIN;
    double volume_sum = 0.0;
    double vol_price_sum = 0.0;

    for (int i = 0; i < n; ++i) {
        const long long ts = timestamps[i];
        if (ts >= 0) {
            if (ts >= next_cutoff || ts < window_start) {
                const long long gid = ts / bucket_ms;
                current_group_id = gid;
                window_start = gid * bucket_ms;
                next_cutoff = window_start + bucket_ms;
                volume_sum = 0.0;
                vol_price_sum = 0.0;
            }
        } else {
            const long long gid = ts / bucket_ms;
            if (gid != current_group_id) {
                current_group_id = gid;
                window_start = gid * bucket_ms;
                next_cutoff = window_start + bucket_ms;
                volume_sum = 0.0;
                vol_price_sum = 0.0;
            }
        }
        const double v = volume[i];
        const double p = prices[i];
        volume_sum += v;
        vol_price_sum = fma(p, v, vol_price_sum);
        row[i] = (volume_sum > 0.0) ? (vol_price_sum / volume_sum) : neo_qnan();
    }
}

// ============================================================================
// WMA — reference: moving_averages/wma.rs::wma_scalar
//   warm = first_valid + period - 1
//   A RUNNING weighted sum, not a per-window recomputation. The CPU primes
//   `weight_sum` and `sum` over the first `period-1` bars, then per bar does
//   `weight_sum += v*period; sum += v; emit weight_sum/weights;
//    weight_sum -= sum; sum -= oldest`. The subtract-AFTER-emit order is what
//   makes the running form exact; reordering it moves every later value.
//   weights = period * (period + 1) * 0.5.
// ============================================================================
extern "C" __global__ void neoethos_wma_batch_f64(
    const double* __restrict__ prices,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    const int period = periods[r];
    double* row = out + (size_t)r * (size_t)n;
    if (period <= 0 || first_valid < 0 || first_valid >= n) { neo_fill_warmup(row, n, n); return; }

    const int lookback = period - 1;
    const int warm = first_valid + lookback;
    neo_fill_warmup(row, n, warm);
    if (warm >= n) return;

    const double period_f = (double)period;
    const double weights = period_f * (period_f + 1.0) * 0.5;

    double sum = 0.0;
    double weight_sum = 0.0;
    for (int k = 0; k < lookback; ++k) {
        const double v = prices[first_valid + k];
        weight_sum += v * ((double)k + 1.0);
        sum += v;
    }

    int in_new = first_valid + lookback;
    int in_old = first_valid;
    int o = warm;
    while (in_new < n) {
        const double v = prices[in_new];
        weight_sum += v * period_f;
        sum += v;
        row[o] = weight_sum / weights;
        weight_sum -= sum;
        sum -= prices[in_old];
        ++in_new;
        ++in_old;
        ++o;
    }
}

// ============================================================================
// WILDERS — reference: moving_averages/wilders.rs::wilders_scalar
//   warm = first_valid + period - 1
//   THE SEED SUM IS GROUPED BY FOUR. The CPU writes
//   `sum += *p0 + *p1 + *p2 + *p3` per group of four and then a 3/2/1
//   remainder: the parenthesised group is summed first and added once, which
//   rounds differently from four separate accumulations into `sum`. This kernel
//   groups identically. It is the single easiest thing to get wrong in this
//   file, and `smma` two definitions below deliberately does NOT do it.
//   Steady state: `y = (x - y).mul_add(alpha, y)` -> `fma(x - y, alpha, y)`.
//   period == 1 copies every bar after the seed.
// ============================================================================
extern "C" __global__ void neoethos_wilders_batch_f64(
    const double* __restrict__ prices,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    const int period = periods[r];
    double* row = out + (size_t)r * (size_t)n;
    if (period <= 0 || first_valid < 0 || first_valid >= n) { neo_fill_warmup(row, n, n); return; }

    const int wma_start = first_valid + period - 1;
    neo_fill_warmup(row, n, wma_start);
    if (wma_start >= n) return;

    double sum = 0.0;
    int p = first_valid;
    const int chunks4 = period / 4;
    for (int k = 0; k < chunks4; ++k) {
        sum += prices[p] + prices[p + 1] + prices[p + 2] + prices[p + 3];
        p += 4;
    }
    switch (period - chunks4 * 4) {
        case 3: sum += prices[p] + prices[p + 1] + prices[p + 2]; break;
        case 2: sum += prices[p] + prices[p + 1]; break;
        case 1: sum += prices[p]; break;
        default: break;
    }

    const double inv_n = 1.0 / (double)period;
    double y = sum * inv_n;
    row[wma_start] = y;

    if (period == 1) {
        for (int i = wma_start + 1; i < n; ++i) row[i] = prices[i];
        return;
    }

    const double alpha = inv_n;
    for (int i = wma_start + 1; i < n; ++i) {
        const double x = prices[i];
        y = fma(x - y, alpha, y);
        row[i] = y;
    }
}

// ============================================================================
// SMMA — reference: moving_averages/smma.rs::smma_scalar
//   warm = first_valid + period - 1
//   Seed is a PLAIN sequential sum over [first, first+period) — unlike
//   `wilders` above, which groups by four. Two indicators of the same family,
//   two different accumulation orders; copying one into the other is the trap
//   this pair exists to document.
//   Steady: `prev = f64::mul_add(prev, pm1, x) * inv_p` ->
//   `fma(prev, pm1, x) * inv_p`, with pm1 = period - 1.
//   period == 1 copies from `first` on.
// ============================================================================
extern "C" __global__ void neoethos_smma_batch_f64(
    const double* __restrict__ prices,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    const int period = periods[r];
    double* row = out + (size_t)r * (size_t)n;
    if (period <= 0 || first_valid < 0 || first_valid >= n) { neo_fill_warmup(row, n, n); return; }

    const int warm = first_valid + period - 1;
    neo_fill_warmup(row, n, warm);
    if (warm >= n) return;

    if (period == 1) {
        for (int i = first_valid; i < n; ++i) row[i] = prices[i];
        return;
    }

    const int end = first_valid + period;
    double sum = 0.0;
    for (int i = first_valid; i < end; ++i) sum += prices[i];

    const double pf = (double)period;
    const double pm1 = pf - 1.0;
    const double inv_p = 1.0 / pf;

    double prev = sum * inv_p;
    row[end - 1] = prev;

    for (int i = end; i < n; ++i) {
        prev = fma(prev, pm1, prices[i]) * inv_p;
        row[i] = prev;
    }
}

// ============================================================================
// DEMA — reference: moving_averages/dema.rs::dema_scalar, THE x86_64 VARIANT at
//   dema.rs:303 carrying `#[target_feature(enable = "fma")]`.
//   There is a SECOND `dema_scalar` at dema.rs:362, compiled only for non-x86,
//   which writes `ema1 * a + x * alpha` where the first writes
//   `ema1.mul_add(a, x * alpha)`. Those are different f64 numbers. This kernel
//   targets the x86_64 one because that is what the parity reference actually
//   executes on the machines this project runs on.
//   warm = first_valid + period - 1. The recursion starts at `first_valid` and
//   the CPU writes out[first_valid] before the NaN prefix is stamped over it,
//   so this kernel computes from `first_valid` and emits only from `warm`.
//   out = `ema1.mul_add(2.0, -ema2)` -> `fma(ema1, 2.0, -ema2)`.
// ============================================================================
extern "C" __global__ void neoethos_dema_batch_f64(
    const double* __restrict__ prices,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    const int period = periods[r];
    double* row = out + (size_t)r * (size_t)n;
    if (period <= 0 || first_valid < 0 || first_valid >= n) { neo_fill_warmup(row, n, n); return; }

    const int warm = first_valid + period - 1;
    neo_fill_warmup(row, n, warm);
    if (warm >= n) return;

    const double alpha = 2.0 / ((double)period + 1.0);
    const double a = 1.0 - alpha;

    double ema1 = prices[first_valid];
    double ema2 = ema1;
    if (first_valid >= warm) row[first_valid] = ema1;

    for (int i = first_valid + 1; i < n; ++i) {
        const double x = prices[i];
        ema1 = fma(ema1, a, x * alpha);
        ema2 = fma(ema2, a, ema1 * alpha);
        const double v = fma(ema1, 2.0, -ema2);
        if (i >= warm) row[i] = v;
    }
}

// ============================================================================
// TEMA — reference: moving_averages/tema.rs::tema_scalar
//   warm = first_valid + (period - 1) * 3
//   Three CASCADED emas, each started at a different bar, and the start offsets
//   are part of the answer: ema2 begins at first + (period-1), ema3 at
//   first + 2*(period-1). At its own start bar each cascade is seeded from the
//   stage below and then IMMEDIATELY stepped once —
//   `ema2 = ema1; ema2 = ema2 * per1 + ema1 * per;`. Dropping that extra step,
//   or seeding without it, shifts the whole series.
//   Plain `x * per1 + y * per` throughout — the CPU does NOT use mul_add for
//   tema, so `-fmad=false` is load-bearing here.
//   period == 1 copies from first_valid on.
// ============================================================================
extern "C" __global__ void neoethos_tema_batch_f64(
    const double* __restrict__ prices,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    const int period = periods[r];
    double* row = out + (size_t)r * (size_t)n;
    if (period <= 0 || first_valid < 0 || first_valid >= n) { neo_fill_warmup(row, n, n); return; }

    const int p1 = period - 1;
    const int warm = first_valid + p1 * 3;
    neo_fill_warmup(row, n, warm);

    if (period == 1) {
        for (int i = first_valid; i < n; ++i) row[i] = prices[i];
        return;
    }

    const double per = 2.0 / ((double)period + 1.0);
    const double per1 = 1.0 - per;

    const int start2 = first_valid + p1;
    const int start3 = first_valid + (p1 << 1);
    const int start_out = warm;

    double ema1 = prices[first_valid];
    double ema2 = 0.0;
    double ema3 = 0.0;

    const int end0 = (start2 < n) ? start2 : n;
    for (int i = first_valid; i < end0; ++i) {
        ema1 = ema1 * per1 + prices[i] * per;
    }

    if (start2 >= n) return;
    ema1 = ema1 * per1 + prices[start2] * per;
    ema2 = ema1;
    ema2 = ema2 * per1 + ema1 * per;

    const int end1 = (start3 < n) ? start3 : n;
    for (int i = start2 + 1; i < end1; ++i) {
        ema1 = ema1 * per1 + prices[i] * per;
        ema2 = ema2 * per1 + ema1 * per;
    }

    if (start3 >= n) return;
    ema1 = ema1 * per1 + prices[start3] * per;
    ema2 = ema2 * per1 + ema1 * per;
    ema3 = ema2;
    ema3 = ema3 * per1 + ema2 * per;

    const int end2 = (start_out < n) ? start_out : n;
    for (int i = start3 + 1; i < end2; ++i) {
        ema1 = ema1 * per1 + prices[i] * per;
        ema2 = ema2 * per1 + ema1 * per;
        ema3 = ema3 * per1 + ema2 * per;
    }

    for (int i = start_out; i < n; ++i) {
        ema1 = ema1 * per1 + prices[i] * per;
        ema2 = ema2 * per1 + ema1 * per;
        ema3 = ema3 * per1 + ema2 * per;
        row[i] = 3.0 * ema1 - 3.0 * ema2 + ema3;
    }
}

// ============================================================================
// ZLEMA — reference: moving_averages/zlema.rs::zlema_scalar
//   warm = first_valid + period - 1
//   lag = (period - 1) / 2, INTEGER division, so an even period lags by less
//   than half a period. The de-lagged input is `2*d[i] - d[i-lag]` only once i
//   has reached `first + lag`; before that it is the raw bar.
//   The recursion runs from `first_valid` regardless of warm, so the emitted
//   values depend on bars that are themselves never emitted.
//   Plain `alpha * val + (1 - alpha) * last` — no mul_add on the CPU side.
// ============================================================================
extern "C" __global__ void neoethos_zlema_batch_f64(
    const double* __restrict__ prices,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    const int period = periods[r];
    double* row = out + (size_t)r * (size_t)n;
    if (period <= 0 || first_valid < 0 || first_valid >= n) { neo_fill_warmup(row, n, n); return; }

    const int lag = (period - 1) / 2;
    const int warm = first_valid + period - 1;
    neo_fill_warmup(row, n, warm);
    if (warm >= n) return;

    const double alpha = 2.0 / ((double)period + 1.0);
    const double one_minus = 1.0 - alpha;

    double last_ema = prices[first_valid];
    for (int i = first_valid; i < n; ++i) {
        if (i > first_valid) {
            const double val = (i < first_valid + lag)
                ? prices[i]
                : (2.0 * prices[i] - prices[i - lag]);
            last_ema = alpha * val + one_minus * last_ema;
        }
        if (i >= warm) row[i] = last_ema;
    }
}

// ============================================================================
// VWMA — reference: moving_averages/vwma.rs::vwma_scalar
//   warm = first_valid + period - 1
//   A running volume-weighted window. The CPU keeps two running sums and
//   updates them with DIFFERENT shapes: `sum += p*v` then `sum -= p_old*v_old`
//   as two separate statements, but `vsum += v_new - v_old` as one fused
//   difference. Writing `vsum += v_new; vsum -= v_old;` rounds differently.
//   Both sums are plain adds; no mul_add on the CPU side.
//   A zero window volume yields an infinity or a NaN from the division, exactly
//   as the CPU does. This kernel does not "improve" on that.
// ============================================================================
extern "C" __global__ void neoethos_vwma_batch_f64(
    const double* __restrict__ prices,
    const double* __restrict__ volume,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    const int period = periods[r];
    double* row = out + (size_t)r * (size_t)n;
    if (period <= 0 || first_valid < 0 || first_valid >= n) { neo_fill_warmup(row, n, n); return; }

    const int warm = first_valid + period - 1;
    neo_fill_warmup(row, n, warm);
    if (warm >= n || n < period) return;

    double sum = 0.0;
    double vsum = 0.0;
    for (int k = 0; k < period; ++k) {
        const double p = prices[first_valid + k];
        const double v = volume[first_valid + k];
        sum += p * v;
        vsum += v;
    }
    row[warm] = sum / vsum;

    int new_idx = first_valid + period;
    int old_idx = first_valid;
    while (new_idx < n) {
        const double pn = prices[new_idx];
        const double vn = volume[new_idx];
        const double po = prices[old_idx];
        const double vo = volume[old_idx];
        sum += pn * vn;
        sum -= po * vo;
        vsum += vn - vo;
        row[new_idx] = sum / vsum;
        ++new_idx;
        ++old_idx;
    }
}

// ============================================================================
// NATR — reference: natr.rs::natr_scalar
//   warm = first_valid + period - 1
//   NOT "atr divided by close": the seed differs from `atr`'s and so does the
//   NaN behaviour of its true range. natr seeds with
//   `sum_tr = high[first] - low[first]`, adds true ranges over (first, warm],
//   divides by period once, and then smooths with a SINGLE FUSED operation:
//   `atr = (tr - atr).mul_add(inv_p, atr)` (natr.rs:344/354/364/374 in the
//   4x-unrolled body and natr.rs:392 in the tail) -> `fma(tr - atr, inv_p, atr)`.
//   NOT `(atr * (period - 1) + tr) / period`, which is algebraically equal and
//   three roundings instead of one. Inside a recursion that difference does not
//   stay at 1 ULP; it compounds over every bar of the series. Same trap as
//   `atr` twelve hundred lines up, different indicator.
//   The true range uses Rust's `.max()` chain, i.e. `f64::max`, which returns
//   the NON-NaN operand — so `fmax` is correct here. `atr.rs` instead uses
//   explicit `if (x > tr)` comparisons, where a NaN comparison is false and the
//   earlier value survives. Same numbers on clean data, different numbers on a
//   NaN bar, and carrying one convention into the other is invisible until a
//   gap appears in live data.
//   A zero or non-finite close yields NaN, not an infinity.
// ============================================================================
extern "C" __global__ void neoethos_natr_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    const int period = periods[r];
    double* row = out + (size_t)r * (size_t)n;
    if (period <= 0 || first_valid < 0 || first_valid >= n) { neo_fill_warmup(row, n, n); return; }

    const int warm_end = first_valid + period - 1;
    neo_fill_warmup(row, n, warm_end);
    if (warm_end >= n) return;

    const double inv_p = 1.0 / (double)period;

    double sum_tr = high[first_valid] - low[first_valid];
    for (int i = first_valid + 1; i <= warm_end; ++i) {
        const double hi = high[i];
        const double lo = low[i];
        const double pc = close[i - 1];
        const double tr = fmax(fmax(hi - lo, fabs(hi - pc)), fabs(lo - pc));
        sum_tr += tr;
    }

    double atr = sum_tr * inv_p;
    const double c_we = close[warm_end];
    row[warm_end] = (neo_is_finite(c_we) && c_we != 0.0) ? ((atr / c_we) * 100.0) : neo_qnan();

    for (int i = warm_end + 1; i < n; ++i) {
        const double hi = high[i];
        const double lo = low[i];
        const double pc = close[i - 1];
        const double tr = fmax(fmax(hi - lo, fabs(hi - pc)), fabs(lo - pc));
        // natr.rs:392 `atr = (tr - atr).mul_add(inv_p, atr)` — ONE rounding.
        atr = fma(tr - atr, inv_p, atr);
        const double c = close[i];
        row[i] = (neo_is_finite(c) && c != 0.0) ? ((atr / c) * 100.0) : neo_qnan();
    }
}

// ============================================================================
// ADXR — reference: adxr.rs::adxr_scalar
// External formula/lookback authorities (audit oracles only; never backends):
// https://raw.githubusercontent.com/TA-Lib/ta-lib/3800d9ed0006fa63cab818737fbea998219419ce/src/ta_func/ta_ADXR.c
// https://raw.githubusercontent.com/TA-Lib/ta-lib/3800d9ed0006fa63cab818737fbea998219419ce/src/ta_func/ta_ADX.c
//   warm = first_valid + 3 * period - 2
//   ADXR averages the current ADX with the ADX from `period - 1` bars ago, so
//   the kernel keeps a ring of the last `period - 1` ADX values. That ring is a
//   per-thread local array bounded by ADXR_MAX_PERIOD, which is a KERNEL
//   property; the host refuses a larger period BY NAME rather than truncating
//   the window. NEVER-OOM: peak memory is fixed by the kernel, never by the
//   caller's request.
//
//   Three details that differ from `adx.rs` and are easy to carry over wrongly:
//     * true range is `a.max(b).max(c)` — Rust `f64::max` -> `fmax`. `adx` uses
//       explicit `if` comparisons instead.
//     * the seed DX gate is `denom > 0.0`, not `denom != 0.0`.
//     * the smoothing is `atr.mul_add(om, tr)` -> `fma(atr, om, tr)`, with
//       om = 1 - 1/period. `adx.rs` writes `atr * one_minus_rp + tr` unfused.
//   The ring is pre-filled with the CPU's NaN and `prev_adx.is_finite()` gates
//   the output. For period == 1, the zero-lag ADXR is the current ADX.
// ============================================================================
extern "C" __global__ void neoethos_adxr_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    const int period = periods[r];
    double* row = out + (size_t)r * (size_t)n;
    if (period <= 0 || period > ADXR_MAX_PERIOD || first_valid < 0 || first_valid >= n) {
        // The host refuses this before launching; an all-NaN row keeps the
        // buffer defined if it ever happens anyway.
        neo_fill_warmup(row, n, n);
        return;
    }

    const int lag = period - 1;
    const int warmup_start = first_valid + 3 * period - 2;
    neo_fill_warmup(row, n, warmup_start);

    const double p = (double)period;
    const double rp = 1.0 / p;
    const double om = 1.0 - rp;
    const double pm1 = p - 1.0;

    double atr_sum = 0.0;
    double plus_dm_sum = 0.0;
    double minus_dm_sum = 0.0;

    int stop = first_valid + period;
    if (stop > n - 1) stop = n - 1;
    for (int i = first_valid + 1; i <= stop; ++i) {
        const double prev_close = close[i - 1];
        const double ch = high[i];
        const double cl = low[i];
        const double ph = high[i - 1];
        const double pl = low[i - 1];

        const double tr = fmax(fmax(ch - cl, fabs(ch - prev_close)), fabs(cl - prev_close));
        atr_sum += tr;

        const double up = ch - ph;
        const double down = pl - cl;
        if (up > down && up > 0.0) plus_dm_sum += up;
        if (down > up && down > 0.0) minus_dm_sum += down;
    }

    const double denom0 = plus_dm_sum + minus_dm_sum;
    const double initial_dx = (denom0 > 0.0)
        ? (100.0 * fabs(plus_dm_sum - minus_dm_sum) / denom0)
        : 0.0;

    double atr = atr_sum;
    double pdm_s = plus_dm_sum;
    double mdm_s = minus_dm_sum;

    double dx_sum = initial_dx;
    int dx_count = 1;
    double adx_last = neo_qnan();
    bool have_adx = false;

    double adx_ring[ADXR_MAX_PERIOD];
    for (int k = 0; k < lag; ++k) adx_ring[k] = neo_qnan();
    int head = 0;

    if (dx_count == period) {
        adx_last = dx_sum * rp;
        have_adx = true;

        double previous_adx;
        if (lag == 0) {
            previous_adx = adx_last;
        } else {
            previous_adx = adx_ring[head];
            adx_ring[head] = adx_last;
            head += 1;
            if (head == lag) head = 0;
        }

        const int initial_adx_index = first_valid + period;
        if (initial_adx_index >= warmup_start && initial_adx_index < n) {
            row[initial_adx_index] = 0.5 * (adx_last + previous_adx);
        }
    }

    for (int i = first_valid + period + 1; i < n; ++i) {
        const double prev_close = close[i - 1];
        const double ch = high[i];
        const double cl = low[i];
        const double ph = high[i - 1];
        const double pl = low[i - 1];

        const double tr = fmax(fmax(ch - cl, fabs(ch - prev_close)), fabs(cl - prev_close));

        const double up = ch - ph;
        const double down = pl - cl;
        const double plus_dm = (up > down && up > 0.0) ? up : 0.0;
        const double minus_dm = (down > up && down > 0.0) ? down : 0.0;

        atr = fma(atr, om, tr);
        pdm_s = fma(pdm_s, om, plus_dm);
        mdm_s = fma(mdm_s, om, minus_dm);

        const double denom = pdm_s + mdm_s;
        const double dx = (denom > 0.0) ? (100.0 * fabs(pdm_s - mdm_s) / denom) : 0.0;

        if (dx_count < period) {
            dx_sum += dx;
            dx_count += 1;

            if (i >= warmup_start) row[i] = neo_qnan();

            if (dx_count == period) {
                adx_last = dx_sum * rp;
                have_adx = true;

                double prev_adx;
                if (lag == 0) {
                    prev_adx = adx_last;
                } else {
                    prev_adx = adx_ring[head];
                    adx_ring[head] = adx_last;
                    head += 1;
                    if (head == lag) head = 0;
                }

                if (i >= warmup_start) {
                    row[i] = neo_is_finite(prev_adx) ? (0.5 * (adx_last + prev_adx)) : neo_qnan();
                }
            }
        } else if (have_adx) {
            const double adx_curr = (adx_last * pm1 + dx) * rp;
            adx_last = adx_curr;

            double prev_adx;
            if (lag == 0) {
                prev_adx = adx_curr;
            } else {
                prev_adx = adx_ring[head];
                adx_ring[head] = adx_curr;
                head += 1;
                if (head == lag) head = 0;
            }

            if (i >= warmup_start) {
                row[i] = neo_is_finite(prev_adx) ? (0.5 * (adx_curr + prev_adx)) : neo_qnan();
            }
        }
    }
}

// ============================================================================
// EFI — reference: efi.rs::efi_scalar
//   warm = first_valid_diff_index(price, volume, first_valid) — the first bar
//   `i > first_valid` where price[i], price[i-1] and volume[i] are ALL
//   non-NaN. That is a SCAN, not `first + period`, and this kernel performs the
//   same scan rather than trusting an arithmetic guess that would be wrong on
//   any frame with an interior gap.
//   At that bar the value is the RAW force `(p[i] - p[i-1]) * v[i]`, not
//   smoothed. Smoothing starts one bar later:
//   `prev = alpha.mul_add(diff, one_minus_alpha * prev)` ->
//   `fma(alpha, diff, one_minus_alpha * prev)`.
//   `prev_price` advances on EVERY bar including invalid ones, while `prev`
//   does not — the CPU uses a bitwise `&` of three NaN checks rather than a
//   short circuit, but the predicate is the same.
// ============================================================================
extern "C" __global__ void neoethos_efi_batch_f64(
    const double* __restrict__ prices,
    const double* __restrict__ volume,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    const int period = periods[r];
    double* row = out + (size_t)r * (size_t)n;
    if (period <= 0 || first_valid < 0 || first_valid >= n) { neo_fill_warmup(row, n, n); return; }

    // first_valid_diff_index, reproduced exactly.
    int start = first_valid + 1;
    while (start < n) {
        const double pc = prices[start];
        const double pp = prices[start - 1];
        const double vc = volume[start];
        if (!isnan(pc) && !isnan(pp) && !isnan(vc)) break;
        ++start;
    }
    if (start >= n) { neo_fill_warmup(row, n, n); return; }

    neo_fill_warmup(row, n, start);

    const double alpha = 2.0 / ((double)period + 1.0);
    const double one_minus_alpha = 1.0 - alpha;

    double prev = (prices[start] - prices[start - 1]) * volume[start];
    row[start] = prev;
    double prev_price = prices[start];

    for (int i = start + 1; i < n; ++i) {
        const double pc = prices[i];
        const double vc = volume[i];
        if (!isnan(pc) && !isnan(prev_price) && !isnan(vc)) {
            const double diff = (pc - prev_price) * vc;
            prev = fma(alpha, diff, one_minus_alpha * prev);
        }
        row[i] = prev;
        prev_price = pc;
    }
}

// ============================================================================
// MEDPRICE — reference: medprice.rs::medprice_scalar
//   warm = first_valid over HIGH AND LOW ONLY — not close, so this indicator
//   needs its own first-valid rule and its own device input shape. Feeding it
//   the hlc first-valid would shift the series wherever close is the late one.
//   `(h + l) * 0.5`, per-bar independent, parallel over (combo, bar).
//   PERIOD-INVARIANT: medprice has no period parameter at all.
// ============================================================================
extern "C" __global__ void neoethos_medprice_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.y;
    if (r >= n_combos) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    (void)periods;  // PERIOD-INVARIANT: medprice takes no period.

    double* row = out + (size_t)r * (size_t)n;
    if (first_valid < 0 || i < first_valid) { row[i] = neo_qnan(); return; }
    row[i] = (high[i] + low[i]) * 0.5;
}

// ============================================================================
// WCLPRICE — reference: wclprice.rs::wclprice_scalar
//   warm = first_valid over high, low AND close.
//   `c.mul_add(0.5, (h + l) * 0.25)` -> `fma(c, 0.5, (h + l) * 0.25)`.
//   The textbook form `(h + l + 2c) / 4` is algebraically identical and
//   numerically is not — one rounding versus three, on every bar.
//   PERIOD-INVARIANT: wclprice takes no period.
// ============================================================================
extern "C" __global__ void neoethos_wclprice_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.y;
    if (r >= n_combos) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    (void)periods;  // PERIOD-INVARIANT: wclprice takes no period.

    double* row = out + (size_t)r * (size_t)n;
    if (first_valid < 0 || i < first_valid) { row[i] = neo_qnan(); return; }
    row[i] = fma(close[i], 0.5, (high[i] + low[i]) * 0.25);
}

// ============================================================================
// MIDPOINT — reference: midpoint.rs::midpoint_scalar
//   warm = first_valid + period - 1
//   `(highest + lowest) / 2.0` over the trailing window of the price series.
//
//   THE SENTINELS ARE OBSERVABLE. The CPU seeds `highest = f64::MIN` and
//   `lowest = f64::MAX` — finite values, not infinities — so a window that is
//   entirely NaN emits `(f64::MIN + f64::MAX) / 2.0`, which is 0.0, rather than
//   NaN. Seeding with infinities here would emit NaN and disagree on every gap
//   while looking perfectly reasonable in review. This is exactly the class of
//   "magic constant copied without re-deriving" that the brief warns about, in
//   the direction nobody expects: the constant must be copied EXACTLY.
//
//   The CPU's `period == 14` fast path is the same fold unrolled and is
//   numerically identical, so one loop serves both.
//   Divides by 2.0; `midprice` below multiplies by 0.5. Both exact, both kept
//   as the CPU writes them.
// ============================================================================
extern "C" __global__ void neoethos_midpoint_batch_f64(
    const double* __restrict__ prices,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.y;
    if (r >= n_combos) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int period = periods[r];
    double* row = out + (size_t)r * (size_t)n;
    if (period <= 0 || first_valid < 0) { row[i] = neo_qnan(); return; }

    const int warm = first_valid + period - 1;
    if (i < warm) { row[i] = neo_qnan(); return; }

    double highest = NEO_F64_MIN;
    double lowest = NEO_F64_MAX;
    for (int j = i + 1 - period; j <= i; ++j) {
        const double v = prices[j];
        if (v > highest) highest = v;
        if (v < lowest) lowest = v;
    }
    row[i] = (highest + lowest) / 2.0;
}

// ============================================================================
// MIDPRICE — reference: midprice.rs::midprice_scalar
//   warm = first_valid + period - 1, over HIGH AND LOW only.
//   `(highest + lowest) * 0.5`, highest from `high` and lowest from `low`.
//   Unlike `midpoint` directly above, this one seeds with NEG_INFINITY /
//   INFINITY, so an all-NaN window emits NaN. Two neighbouring indicators, two
//   different sentinel conventions, both reproduced rather than harmonised.
//   The CPU has a `period <= 64` scan and a monotonic-deque path for longer
//   windows; max and min are exact so both give the same value, and one loop
//   serves both.
// ============================================================================
extern "C" __global__ void neoethos_midprice_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.y;
    if (r >= n_combos) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int period = periods[r];
    double* row = out + (size_t)r * (size_t)n;
    if (period <= 0 || first_valid < 0) { row[i] = neo_qnan(); return; }

    const int warm = first_valid + period - 1;
    if (i < warm) { row[i] = neo_qnan(); return; }

    double highest = -INFINITY;
    double lowest = INFINITY;
    for (int j = i + 1 - period; j <= i; ++j) {
        const double h = high[j];
        if (h > highest) highest = h;
        const double l = low[j];
        if (l < lowest) lowest = l;
    }
    row[i] = (highest + lowest) * 0.5;
}

// ============================================================================
// ROCP — reference: rocp.rs::rocp_scalar
//   warm = first_valid + period
//   `(c - p) / p`.
//   THIS IS NOT `roc`. `roc` guards a zero or NaN previous price and returns
//   0.0; `rocp` does NOT guard, and lets the division produce an infinity or a
//   NaN. Adding the guard here would be an "improvement" that fails parity on
//   exactly the corrupt zero-price bars this project has already been bitten
//   by. `rocr` immediately below DOES guard. Three neighbours, three different
//   rules, taken from the source rather than from the family resemblance.
// ============================================================================
extern "C" __global__ void neoethos_rocp_batch_f64(
    const double* __restrict__ prices,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.y;
    if (r >= n_combos) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int period = periods[r];
    double* row = out + (size_t)r * (size_t)n;
    if (period <= 0 || first_valid < 0) { row[i] = neo_qnan(); return; }

    const int warm = first_valid + period;
    if (i < warm) { row[i] = neo_qnan(); return; }

    const double c = prices[i];
    const double p = prices[i - period];
    row[i] = (c - p) / p;
}

// ============================================================================
// ROCR — reference: rocr.rs::rocr_scalar
//   warm = first_valid + period
//   `c / p`, with a zero or NaN previous price mapped to 0.0 — the guard `roc`
//   has and `rocp` does not.
// ============================================================================
extern "C" __global__ void neoethos_rocr_batch_f64(
    const double* __restrict__ prices,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.y;
    if (r >= n_combos) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int period = periods[r];
    double* row = out + (size_t)r * (size_t)n;
    if (period <= 0 || first_valid < 0) { row[i] = neo_qnan(); return; }

    const int warm = first_valid + period;
    if (i < warm) { row[i] = neo_qnan(); return; }

    const double past = prices[i - period];
    row[i] = (past == 0.0 || isnan(past)) ? 0.0 : (prices[i] / past);
}
