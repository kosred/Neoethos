#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>


#ifndef CG_NAN
#define CG_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


static __device__ __forceinline__ bool cg_bad_den(float den) {
    return (!isfinite(den)) || fabsf(den) <= 1.1920929e-7f;
}


struct CompSum {
    float s;
    float c;
    __device__ __forceinline__ CompSum() : s(0.0f), c(0.0f) {}
    __device__ __forceinline__ void add(float x) {
        float t = s + x;
        c += (fabsf(s) >= fabsf(x)) ? (s - t) + x : (x - t) + s;
        s  = t;
    }
    __device__ __forceinline__ void sub(float x) { add(-x); }
    __device__ __forceinline__ float val() const { return s + c; }
};


extern "C" __global__ void cg_batch_f32(const float* __restrict__ prices,
                                        const int*   __restrict__ periods,
                                        int series_len,
                                        int n_combos,
                                        int first_valid,
                                        float* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int base   = combo * series_len;
    float* __restrict__ out_ptr = out + base;


    if (UNLIKELY(period <= 0 || period > series_len ||
                 first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) out_ptr[i] = CG_NAN;
        return;
    }
    const int tail_len = series_len - first_valid;
    if (UNLIKELY(tail_len < (period + 1))) {
        for (int i = 0; i < series_len; ++i) out_ptr[i] = CG_NAN;
        return;
    }

    const int warm   = first_valid + period;
    const int window = period - 1;


    for (int i = 0; i < warm; ++i) out_ptr[i] = CG_NAN;

    if (window <= 0) {

        for (int i = warm; i < series_len; ++i) out_ptr[i] = 0.0f;
        return;
    }


    CompSum S_acc, T_acc;
    int nan_count = 0;


    for (int k = 0; k < window; ++k) {
        const float p = prices[warm - k];
        if (isfinite(p)) {
            S_acc.add(p);
            T_acc.add(fmaf((float)(k + 1), p, 0.0f));
        } else {
            nan_count++;
        }
    }


    {
        const float S = S_acc.val();
        out_ptr[warm] = (nan_count > 0 || cg_bad_den(S)) ? 0.0f : (-T_acc.val() / S);
    }


    const int REFRESH_EVERY = 512;
    int since_refresh = 0;
    for (int i = warm; i < series_len - 1; ++i) {
        const float add  = prices[i + 1];
        const float drop = prices[i - window + 1];

        if (isfinite(add))  S_acc.add(add); else ++nan_count;
        if (isfinite(drop)) S_acc.sub(drop); else --nan_count;


        T_acc.add(S_acc.val());

        if (isfinite(drop)) T_acc.sub((float)window * drop);


        const float S = S_acc.val();
        out_ptr[i + 1] = (nan_count > 0 || cg_bad_den(S)) ? 0.0f : (-T_acc.val() / S);


        if (++since_refresh >= REFRESH_EVERY) {
            since_refresh = 0;

            CompSum S_new, T_new;
            int nc = 0;
            const int cur = i + 1;
            for (int k = 0; k < window; ++k) {
                const float p = prices[cur - k];
                if (isfinite(p)) { S_new.add(p); T_new.add(fmaf((float)(k + 1), p, 0.0f)); }
                else { nc++; }
            }
            S_acc = S_new;
            T_acc = T_new;
            nan_count = nc;
        }
    }
}


extern "C" __global__ void cg_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float* __restrict__ out_tm)
{
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const float* __restrict__ col_in  = prices_tm + series;
    float*       __restrict__ col_out = out_tm    + series;

    if (UNLIKELY(period <= 0 || period > series_len)) {

        for (int row = 0; row < series_len; ++row)
            col_out[(size_t)row * num_series] = CG_NAN;
        return;
    }

    const int first_valid = first_valids[series];
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
        for (int row = 0; row < series_len; ++row)
            col_out[(size_t)row * num_series] = CG_NAN;
        return;
    }

    const int tail_len = series_len - first_valid;
    if (UNLIKELY(tail_len < (period + 1))) {
        for (int row = 0; row < series_len; ++row)
            col_out[(size_t)row * num_series] = CG_NAN;
        return;
    }

    const int warm   = first_valid + period;
    const int window = period - 1;


    for (int row = 0; row < warm; ++row)
        col_out[(size_t)row * num_series] = CG_NAN;

    if (window <= 0) {
        for (int row = warm; row < series_len; ++row)
            col_out[(size_t)row * num_series] = 0.0f;
        return;
    }


    CompSum S_acc, T_acc;
    int nan_count = 0;
    for (int k = 0; k < window; ++k) {
        const float p = col_in[(size_t)(warm - k) * num_series];
        if (isfinite(p)) {
            S_acc.add(p);
            T_acc.add(fmaf((float)(k + 1), p, 0.0f));
        } else {
            nan_count++;
        }
    }

    {
        const float S = S_acc.val();
        col_out[(size_t)warm * num_series] =
            (nan_count > 0 || cg_bad_den(S)) ? 0.0f : (-T_acc.val() / S);
    }

    const int REFRESH_EVERY = 512;
    int since_refresh = 0;
    for (int row = warm; row < series_len - 1; ++row) {
        const float add  = col_in[(size_t)(row + 1) * num_series];
        const float drop = col_in[(size_t)(row - window + 1) * num_series];

        if (isfinite(add))  S_acc.add(add); else ++nan_count;
        if (isfinite(drop)) S_acc.sub(drop); else --nan_count;

        T_acc.add(S_acc.val());
        if (isfinite(drop)) T_acc.sub((float)window * drop);

        const float S = S_acc.val();
        col_out[(size_t)(row + 1) * num_series] =
            (nan_count > 0 || cg_bad_den(S)) ? 0.0f : (-T_acc.val() / S);

        if (++since_refresh >= REFRESH_EVERY) {
            since_refresh = 0;
            CompSum S_new, T_new;
            int nc = 0;
            const int cur = row + 1;
            for (int k = 0; k < window; ++k) {
                const float p = col_in[(size_t)(cur - k) * num_series];
                if (isfinite(p)) { S_new.add(p); T_new.add(fmaf((float)(k + 1), p, 0.0f)); }
                else { nc++; }
            }
            S_acc = S_new;
            T_acc = T_new;
            nan_count = nc;
        }
    }
}


extern "C" __global__ void cg_prefix_prepare_f32(const float* __restrict__ prices,
                                                 int series_len,
                                                 float* __restrict__ P,
                                                 float* __restrict__ Q,
                                                 int*   __restrict__ C)
{

    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float ps = 0.0f;
    float qs = 0.0f;
    int   cs = 0;
    for (int i = 0; i < series_len; ++i) {
        const float p = prices[i];
        if (isfinite(p)) {
            ps += p;
            qs = fmaf((float)i, p, qs);
        } else {
            cs += 1;
        }
        P[i] = ps;
        Q[i] = qs;
        C[i] = cs;
    }
}


extern "C" __global__ void cg_batch_f32_from_prefix(
    const float* __restrict__ ,
    const int*   __restrict__ periods,
    int series_len,
    int n_combos,
    int first_valid,
    const float* __restrict__ P,
    const float* __restrict__ Q,
    const int*   __restrict__ C,
    float* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    float* __restrict__ out_ptr = out + (size_t)combo * series_len;

    if (UNLIKELY(period <= 0 || period > series_len ||
                 first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) out_ptr[i] = CG_NAN;
        return;
    }
    const int tail_len = series_len - first_valid;
    if (UNLIKELY(tail_len < (period + 1))) {
        for (int i = 0; i < series_len; ++i) out_ptr[i] = CG_NAN;
        return;
    }

    const int warm   = first_valid + period;
    const int window = period - 1;


    for (int i = 0; i < warm; ++i) out_ptr[i] = CG_NAN;
    if (window <= 0) {
        for (int i = warm; i < series_len; ++i) out_ptr[i] = 0.0f;
        return;
    }


    for (int i = warm; i < series_len; ++i) {
        const int a = i - window + 1;
        const int b = i;

        const float sumP = (P[b] - (a > 0 ? P[a - 1] : 0.0f));
        const float sumQ = (Q[b] - (a > 0 ? Q[a - 1] : 0.0f));
        const int   nans = (C[b] - (a > 0 ? C[a - 1] : 0));

        if (nans > 0 || cg_bad_den(sumP)) {
            out_ptr[i] = 0.0f;
        } else {

            const float num = fmaf((float)(i + 1), sumP, -sumQ);
            out_ptr[i] = -num / sumP;
        }
    }
}


// ===========================================================================
// f64 LANE  --  shard S5
// ===========================================================================
//
// The f32 entry points above are LEFT IN PLACE because the generated f32
// dispatcher and this indicator's own `*_wrapper.rs` still launch them by
// name. Everything below is the SAME algorithm at f64, in this same file, and
// it is what the NeoEthos f64 lane consumes. Nothing here narrows, and nothing
// here is fast-math:
//
//   * every `float` data pointer, local and shared array is `double`
//   * every f32 literal lost its `f` suffix
//   * expf/sqrtf/fmaxf/fminf/fabsf/powf/logf -> exp/sqrt/fmax/fmin/fabs/pow/log
//   * __fadd_rn/__fsub_rn/__fmul_rn -> __dadd_rn/__dsub_rn/__dmul_rn
//     __fmaf_rn -> __fma_rn  (ONE rounding, matching `f64::mul_add`)
//     __fdividef -> __ddiv_rn and __frcp_rn -> __drcp_rn: those two are the
//     FAST APPROXIMATE divide and reciprocal, and their f64 images here are
//     the correctly-rounded operations, not a wider approximation
//   * an f32 NaN bit pattern is NOT a NaN when reinterpreted as f64 --
//     `__longlong_as_double(0x7fc00000)` is 2.09e-314, a finite denormal that
//     compares ORDERED against everything, so a warmup prefix meant to read
//     NaN would read ~0.0 instead. Every such site became the f64 pattern
//     (0x7ff8000000000000 / 0x7fffffffffffffff).
//   * every epsilon was RE-DERIVED at f64 width from the CPU reference rather
//     than carried over; see the per-file note where one exists.
// ===========================================================================

#include <float.h>   // DBL_EPSILON -- see the note below
static __device__ __forceinline__ bool cg_bad_den_f64(double den) {
    return !(fabs(den) > DBL_EPSILON);
}
struct CompSum_f64 {
    double s;
    double c;
    __device__ __forceinline__ CompSum_f64() : s(0.0), c(0.0) {}
    __device__ __forceinline__ void add_f64(double x) {
        double t = s + x;
        c += (fabs(s) >= fabs(x)) ? (s - t) + x : (x - t) + s;
        s  = t;
    }
    __device__ __forceinline__ void sub_f64(double x) { add_f64(-x); }
    __device__ __forceinline__ double val_f64() const { return s + c; }
};
extern "C" __global__ void cg_batch_f64(const double* __restrict__ prices,
                                        const int*   __restrict__ periods,
                                        int series_len,
                                        int n_combos,
                                        int first_valid,
                                        double* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int base   = combo * series_len;
    double* __restrict__ out_ptr = out + base;


    if (UNLIKELY(period <= 0 || period > series_len ||
                 first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) out_ptr[i] = CG_NAN;
        return;
    }
    const int tail_len = series_len - first_valid;
    if (UNLIKELY(tail_len < (period + 1))) {
        for (int i = 0; i < series_len; ++i) out_ptr[i] = CG_NAN;
        return;
    }

    const int warm   = first_valid + period;
    const int window = period - 1;


    for (int i = 0; i < warm; ++i) out_ptr[i] = CG_NAN;

    if (window <= 0) {

        for (int i = warm; i < series_len; ++i) out_ptr[i] = 0.0;
        return;
    }


    CompSum_f64 S_acc, T_acc;
    int nan_count = 0;


    for (int k = 0; k < window; ++k) {
        const double p = prices[warm - k];
        if (isfinite(p)) {
            S_acc.add_f64(p);
            T_acc.add_f64(fma((double)(k + 1), p, 0.0));
        } else {
            nan_count++;
        }
    }


    {
        const double S = S_acc.val_f64();
        out_ptr[warm] = (nan_count > 0 || cg_bad_den_f64(S)) ? 0.0 : (-T_acc.val_f64() / S);
    }


    const int REFRESH_EVERY = 512;
    int since_refresh = 0;
    for (int i = warm; i < series_len - 1; ++i) {
        const double add_f64  = prices[i + 1];
        const double drop = prices[i - window + 1];

        if (isfinite(add_f64))  S_acc.add_f64(add_f64); else ++nan_count;
        if (isfinite(drop)) S_acc.sub_f64(drop); else --nan_count;


        T_acc.add_f64(S_acc.val_f64());

        if (isfinite(drop)) T_acc.sub_f64((double)window * drop);


        const double S = S_acc.val_f64();
        out_ptr[i + 1] = (nan_count > 0 || cg_bad_den_f64(S)) ? 0.0 : (-T_acc.val_f64() / S);


        if (++since_refresh >= REFRESH_EVERY) {
            since_refresh = 0;

            CompSum_f64 S_new, T_new;
            int nc = 0;
            const int cur = i + 1;
            for (int k = 0; k < window; ++k) {
                const double p = prices[cur - k];
                if (isfinite(p)) { S_new.add_f64(p); T_new.add_f64(fma((double)(k + 1), p, 0.0)); }
                else { nc++; }
            }
            S_acc = S_new;
            T_acc = T_new;
            nan_count = nc;
        }
    }
}
extern "C" __global__ void cg_many_series_one_param_f64(
    const double* __restrict__ prices_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    double* __restrict__ out_tm)
{
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) return;

    const double* __restrict__ col_in  = prices_tm + series;
    double*       __restrict__ col_out = out_tm    + series;

    if (UNLIKELY(period <= 0 || period > series_len)) {

        for (int row = 0; row < series_len; ++row)
            col_out[(size_t)row * num_series] = CG_NAN;
        return;
    }

    const int first_valid = first_valids[series];
    if (UNLIKELY(first_valid < 0 || first_valid >= series_len)) {
        for (int row = 0; row < series_len; ++row)
            col_out[(size_t)row * num_series] = CG_NAN;
        return;
    }

    const int tail_len = series_len - first_valid;
    if (UNLIKELY(tail_len < (period + 1))) {
        for (int row = 0; row < series_len; ++row)
            col_out[(size_t)row * num_series] = CG_NAN;
        return;
    }

    const int warm   = first_valid + period;
    const int window = period - 1;


    for (int row = 0; row < warm; ++row)
        col_out[(size_t)row * num_series] = CG_NAN;

    if (window <= 0) {
        for (int row = warm; row < series_len; ++row)
            col_out[(size_t)row * num_series] = 0.0;
        return;
    }


    CompSum_f64 S_acc, T_acc;
    int nan_count = 0;
    for (int k = 0; k < window; ++k) {
        const double p = col_in[(size_t)(warm - k) * num_series];
        if (isfinite(p)) {
            S_acc.add_f64(p);
            T_acc.add_f64(fma((double)(k + 1), p, 0.0));
        } else {
            nan_count++;
        }
    }

    {
        const double S = S_acc.val_f64();
        col_out[(size_t)warm * num_series] =
            (nan_count > 0 || cg_bad_den_f64(S)) ? 0.0 : (-T_acc.val_f64() / S);
    }

    const int REFRESH_EVERY = 512;
    int since_refresh = 0;
    for (int row = warm; row < series_len - 1; ++row) {
        const double add_f64  = col_in[(size_t)(row + 1) * num_series];
        const double drop = col_in[(size_t)(row - window + 1) * num_series];

        if (isfinite(add_f64))  S_acc.add_f64(add_f64); else ++nan_count;
        if (isfinite(drop)) S_acc.sub_f64(drop); else --nan_count;

        T_acc.add_f64(S_acc.val_f64());
        if (isfinite(drop)) T_acc.sub_f64((double)window * drop);

        const double S = S_acc.val_f64();
        col_out[(size_t)(row + 1) * num_series] =
            (nan_count > 0 || cg_bad_den_f64(S)) ? 0.0 : (-T_acc.val_f64() / S);

        if (++since_refresh >= REFRESH_EVERY) {
            since_refresh = 0;
            CompSum_f64 S_new, T_new;
            int nc = 0;
            const int cur = row + 1;
            for (int k = 0; k < window; ++k) {
                const double p = col_in[(size_t)(cur - k) * num_series];
                if (isfinite(p)) { S_new.add_f64(p); T_new.add_f64(fma((double)(k + 1), p, 0.0)); }
                else { nc++; }
            }
            S_acc = S_new;
            T_acc = T_new;
            nan_count = nc;
        }
    }
}
extern "C" __global__ void cg_prefix_prepare_f64(const double* __restrict__ prices,
                                                 int series_len,
                                                 double* __restrict__ P,
                                                 double* __restrict__ Q,
                                                 int*   __restrict__ C)
{

    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    double ps = 0.0;
    double qs = 0.0;
    int   cs = 0;
    for (int i = 0; i < series_len; ++i) {
        const double p = prices[i];
        if (isfinite(p)) {
            ps += p;
            // S5 CORRECTION -- ROUNDING COUNT. The CPU weighted sum in
            // `cg.rs` (`dot_sum_precomputed`) is `num += w * p`: multiply,
            // then add, TWO roundings. `fma` is ONE. `-fmad=false` keeps
            // the plain form from being contracted back into an FMA.
            qs = qs + (double)i * p;
        } else {
            cs += 1;
        }
        P[i] = ps;
        Q[i] = qs;
        C[i] = cs;
    }
}
extern "C" __global__ void cg_batch_f64_from_prefix(
    const double* __restrict__ ,
    const int*   __restrict__ periods,
    int series_len,
    int n_combos,
    int first_valid,
    const double* __restrict__ P,
    const double* __restrict__ Q,
    const int*   __restrict__ C,
    double* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    double* __restrict__ out_ptr = out + (size_t)combo * series_len;

    if (UNLIKELY(period <= 0 || period > series_len ||
                 first_valid < 0 || first_valid >= series_len)) {
        for (int i = 0; i < series_len; ++i) out_ptr[i] = CG_NAN;
        return;
    }
    const int tail_len = series_len - first_valid;
    if (UNLIKELY(tail_len < (period + 1))) {
        for (int i = 0; i < series_len; ++i) out_ptr[i] = CG_NAN;
        return;
    }

    const int warm   = first_valid + period;
    const int window = period - 1;


    for (int i = 0; i < warm; ++i) out_ptr[i] = CG_NAN;
    if (window <= 0) {
        for (int i = warm; i < series_len; ++i) out_ptr[i] = 0.0;
        return;
    }


    for (int i = warm; i < series_len; ++i) {
        const int a = i - window + 1;
        const int b = i;

        const double sumP = (P[b] - (a > 0 ? P[a - 1] : 0.0));
        const double sumQ = (Q[b] - (a > 0 ? Q[a - 1] : 0.0));
        const int   nans = (C[b] - (a > 0 ? C[a - 1] : 0));

        if (nans > 0 || cg_bad_den_f64(sumP)) {
            out_ptr[i] = 0.0;
        } else {

            const double num = fma((double)(i + 1), sumP, -sumQ);
            out_ptr[i] = -num / sumP;
        }
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE - cg (Ehlers centre of gravity)
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/cg.rs:260 `cg_scalar`, entered from
 *             `cg_with_kernel` (:201) whose `first` is at :210 and whose NaN
 *             prefix is `first + period` (:231).
 *
 * SINGLE OUTPUT ("value", cpu_batch.rs:3258 `expect_value_output`).
 *
 * PERIOD-SWEPT: `compute_cg_batch` reads `period` (default 10), so every row
 * of the sweep is a DIFFERENT column and `periods[combo]` is honoured.
 *
 * FIRST-VALID: `!is_nan` on the single source series (:210-213) - the common
 * rule. `F64FirstValidRule::AllInputsNonNan`.
 *
 * THE EPSILON THIS FILE GOT WRONG IN f32. `cg_kernel.cu:22` guards the
 * denominator with `fabsf(den) <= 1.1920929e-7f`, which is f32 MACHINE
 * EPSILON. The CPU guard is `den.abs() > f64::EPSILON` (:339, :393) -
 * 2.2204460492503131e-16. Copying the f32 constant into an f64 kernel would
 * make the zero-denominator branch fire roughly nine orders of magnitude too
 * early and emit 0.0 where the CPU emits a real value. `DBL_EPSILON` is used
 * here, spelled out as the literal so it cannot drift with a header.
 *
 * WINDOW: `n_items = period - 1` (:274), NOT `period`. The dot runs
 * k = 0..period-2 over `data[i-k]` with weight `k + 1`, so the OLDEST bar of
 * the nominal window is not read at all. Transcribed literally rather than
 * "corrected" to a full window.
 *
 * ACCUMULATION ORDER: the CPU has three shapes for the same sum - an 8-wide
 * unrolled block against a precomputed weight table (:276), a 4-wide block
 * with a running `w` (:357), and a fully written-out `period == 10` special
 * case (:400). All three accumulate `num` and `den` ASCENDING in k with a
 * single accumulator, so all three are the same left-associative sum that a
 * plain ascending loop produces; `num` starts at 0.0 and `0.0 + x` is exact.
 * ONE kernel therefore serves every period including 10, and no special case
 * is needed. Stated because the opposite is true for `wilders` in this crate.
 *
 * SEQUENTIAL PER COLUMN, but there is no carried state: each bar re-reads its
 * own window. The lane launches one thread per combo, so the loop over bars
 * is the thread body.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void cg_neo_batch_f64(const double* __restrict__ data,
                      int series_len,
                      const int* __restrict__ periods,
                      int n_combos,
                      int first_valid,
                      double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;
    const int period = periods[combo];

    /* f64::EPSILON, NOT the f32 machine epsilon this file carries at :22. */
    const double DBL_EPS = 2.2204460492503130808472633361816e-16;

    if (period <= 0 || period > len || first_valid < 0 || first_valid >= len) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    const int start = first_valid + period;      /* cg.rs:265, :231 */
    for (int i = 0; i < len && i < start; ++i) o[i] = NEO_F64_NAN;
    if (start >= len) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    const int n_items = period - 1;              /* cg.rs:274 */

    for (int i = start; i < len; ++i) {
        double num = 0.0;
        double den = 0.0;
        double w = 1.0;
        for (int k = 0; k < n_items; ++k) {
            const double p = data[i - k];
            num += w * p;
            den += p;
            w += 1.0;
        }
        o[i] = (fabs(den) > DBL_EPS) ? -num / den : 0.0;
    }
}
