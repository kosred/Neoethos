#include <cuda_runtime.h>
#include <math_constants.h>

#ifndef FORCE_INLINE
#define FORCE_INLINE __forceinline__ __device__
#endif

static inline __device__ float f_nan() { return CUDART_NAN_F; }

static constexpr int LEVELS = 9;


FORCE_INLINE void pivot_compute_levels_core(
    const int mode, const float h, const float l, const float c, const float o,
    float &r4, float &r3, float &r2, float &r1, float &pp, float &s1, float &s2, float &s3, float &s4)
{
    const float d = h - l;


    r4 = r3 = r2 = r1 = pp = s1 = s2 = s3 = s4 = f_nan();

    switch (mode) {

        case 0: {
            pp = (h + l + c) * (1.0f / 3.0f);
            const float t2 = pp + pp;
            r1 = t2 - l;
            r2 = pp + d;
            s1 = t2 - h;
            s2 = pp - d;
            break;
        }

        case 1: {
            pp = (h + l + c) * (1.0f / 3.0f);
            r1 = fmaf(d, 0.382f, pp);
            r2 = fmaf(d, 0.618f, pp);
            r3 = fmaf(d, 1.000f, pp);
            s1 = fmaf(d, -0.382f, pp);
            s2 = fmaf(d, -0.618f, pp);
            s3 = fmaf(d, -1.000f, pp);
            break;
        }

        case 2: {
            const float p_lt = (h + (l + l) + c) * 0.25f;
            const float p_gt = ((h + h) + l + c) * 0.25f;
            const float p_eq = (h + l + (c + c)) * 0.25f;
            if (c < o)      pp = p_lt;
            else if (c > o) pp = p_gt;
            else            pp = p_eq;

            const float n_lt = (h + (l + l) + c) * 0.5f;
            const float n_gt = ((h + h) + l + c) * 0.5f;
            const float n_eq = (h + l + (c + c)) * 0.5f;
            const float n = (c < o) ? n_lt : ((c > o) ? n_gt : n_eq);
            r1 = n - l;
            s1 = n - h;
            break;
        }

        case 3: {
            pp = (h + l + c) * (1.0f / 3.0f);

            const float c1 = 0.0916f, c2 = 0.183f, c3 = 0.275f, c4 = 0.55f;
            r1 = fmaf(d,  c1, c);
            r2 = fmaf(d,  c2, c);
            r3 = fmaf(d,  c3, c);
            r4 = fmaf(d,  c4, c);
            s1 = fmaf(d, -c1, c);
            s2 = fmaf(d, -c2, c);
            s3 = fmaf(d, -c3, c);
            s4 = fmaf(d, -c4, c);
            break;
        }

        case 4: {
            pp = (h + l + (o + o)) * 0.25f;
            const float t2p = pp + pp;
            const float t2l = l + l;
            const float t2h = h + h;
            r1 = t2p - l;
            r2 = fmaf(d,  1.0f, pp);
            r3 = (t2p - t2l) + h;
            r4 = fmaf(d,  1.0f, r3);
            s1 = t2p - h;
            s2 = fmaf(d, -1.0f, pp);
            s3 = (l + t2p) - t2h;
            s4 = fmaf(d, -1.0f, s3);
            break;
        }
        default: {  break; }
    }
}

extern "C" __global__ void pivot_extract_output_rows_f32(
    const float* __restrict__ packed,
    int num_combos,
    int series_len,
    int output_index,
    float* __restrict__ out)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_combos * series_len;
    if (idx >= total) return;

    const int row = idx / series_len;
    const int col = idx - row * series_len;
    const int packed_row = row * LEVELS + output_index;
    out[idx] = packed[packed_row * series_len + col];
}


extern "C" __global__
void pivot_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const float* __restrict__ open,
    const int*   __restrict__ modes,
    int n,
    int first_valid,
    int n_combos,
    float* __restrict__ out)
{

    if (blockIdx.y != 0) return;


    __shared__ unsigned char s_need_open_any;
    if (threadIdx.x == 0) {
        unsigned char f = 0;
        for (int j = 0; j < n_combos; ++j) {
            const int m = modes[j];
            f |= (m == 2) | (m == 4);
        }
        s_need_open_any = f;
    }
    __syncthreads();
    const bool need_open_any = (s_need_open_any != 0);


    const int stride = blockDim.x * gridDim.x;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride)
    {
        const float h = high[t];
        const float l = low[t];
        const float c = close[t];
        const float o = need_open_any ? open[t] : 0.0f;


        const bool base_ok = (t >= first_valid) && (h == h) && (l == l) && (c == c);


        for (int j = 0; j < n_combos; ++j) {
            const int mode = modes[j];
            const bool need_o = (mode == 2) || (mode == 4);
            const bool valid  = base_ok && (!need_o || (o == o));


            const int base = (j * LEVELS) * n + t;

            float r4, r3, r2, r1, pp, s1, s2, s3, s4;
            if (valid) {
                pivot_compute_levels_core(mode, h, l, c, o, r4, r3, r2, r1, pp, s1, s2, s3, s4);
            } else {
                r4 = r3 = r2 = r1 = pp = s1 = s2 = s3 = s4 = f_nan();
            }


            out[base + 0 * n] = r4;
            out[base + 1 * n] = r3;
            out[base + 2 * n] = r2;
            out[base + 3 * n] = r1;
            out[base + 4 * n] = pp;
            out[base + 5 * n] = s1;
            out[base + 6 * n] = s2;
            out[base + 7 * n] = s3;
            out[base + 8 * n] = s4;
        }
    }
}


extern "C" __global__
void pivot_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const float* __restrict__ open_tm,
    const int*   __restrict__ first_valids,
    int cols,
    int rows,
    int mode,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const bool need_o = (mode == 2) || (mode == 4);
    const int first_valid = first_valids[s];


    for (int t = 0; t < rows; ++t) {
        const int idx = t * cols + s;

        float r4, r3, r2, r1, pp, s1, s2, s3, s4;
        const float h = high_tm[idx];
        const float l = low_tm[idx];
        const float c = close_tm[idx];
        const float o = need_o ? open_tm[idx] : 0.0f;

        const bool valid = (t >= first_valid) && (h == h) && (l == l) && (c == c) && (!need_o || (o == o));
        if (valid) {
            pivot_compute_levels_core(mode, h, l, c, o, r4, r3, r2, r1, pp, s1, s2, s3, s4);
        } else {
            r4 = r3 = r2 = r1 = pp = s1 = s2 = s3 = s4 = f_nan();
        }


        out_tm[(0 * rows + t) * cols + s] = r4;
        out_tm[(1 * rows + t) * cols + s] = r3;
        out_tm[(2 * rows + t) * cols + s] = r2;
        out_tm[(3 * rows + t) * cols + s] = r1;
        out_tm[(4 * rows + t) * cols + s] = pp;
        out_tm[(5 * rows + t) * cols + s] = s1;
        out_tm[(6 * rows + t) * cols + s] = s2;
        out_tm[(7 * rows + t) * cols + s] = s3;
        out_tm[(8 * rows + t) * cols + s] = s4;
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

static inline __device__ double f_nan_f64() { return CUDART_NAN; }
FORCE_INLINE void pivot_compute_levels_core(
    const int mode, const double h, const double l, const double c, const double o,
    double &r4, double &r3, double &r2, double &r1, double &pp, double &s1, double &s2, double &s3, double &s4)
{
    const double d = h - l;


    r4 = r3 = r2 = r1 = pp = s1 = s2 = s3 = s4 = f_nan_f64();

    switch (mode) {

        case 0: {
            pp = (h + l + c) * (1.0 / 3.0);
            const double t2 = pp + pp;
            r1 = t2 - l;
            r2 = pp + d;
            s1 = t2 - h;
            s2 = pp - d;
            break;
        }

        case 1: {
            pp = (h + l + c) * (1.0 / 3.0);
            r1 = fma(d, 0.382, pp);
            r2 = fma(d, 0.618, pp);
            r3 = fma(d, 1.000, pp);
            s1 = fma(d, -0.382, pp);
            s2 = fma(d, -0.618, pp);
            s3 = fma(d, -1.000, pp);
            break;
        }

        case 2: {
            const double p_lt = (h + (l + l) + c) * 0.25;
            const double p_gt = ((h + h) + l + c) * 0.25;
            const double p_eq = (h + l + (c + c)) * 0.25;
            if (c < o)      pp = p_lt;
            else if (c > o) pp = p_gt;
            else            pp = p_eq;

            const double n_lt = (h + (l + l) + c) * 0.5;
            const double n_gt = ((h + h) + l + c) * 0.5;
            const double n_eq = (h + l + (c + c)) * 0.5;
            const double n = (c < o) ? n_lt : ((c > o) ? n_gt : n_eq);
            r1 = n - l;
            s1 = n - h;
            break;
        }

        case 3: {
            pp = (h + l + c) * (1.0 / 3.0);

            const double c1 = 0.0916, c2 = 0.183, c3 = 0.275, c4 = 0.55;
            r1 = fma(d,  c1, c);
            r2 = fma(d,  c2, c);
            r3 = fma(d,  c3, c);
            r4 = fma(d,  c4, c);
            s1 = fma(d, -c1, c);
            s2 = fma(d, -c2, c);
            s3 = fma(d, -c3, c);
            s4 = fma(d, -c4, c);
            break;
        }

        case 4: {
            pp = (h + l + (o + o)) * 0.25;
            const double t2p = pp + pp;
            const double t2l = l + l;
            const double t2h = h + h;
            r1 = t2p - l;
            r2 = fma(d,  1.0, pp);
            r3 = (t2p - t2l) + h;
            r4 = fma(d,  1.0, r3);
            s1 = t2p - h;
            s2 = fma(d, -1.0, pp);
            s3 = (l + t2p) - t2h;
            s4 = fma(d, -1.0, s3);
            break;
        }
        default: {  break; }
    }
}
extern "C" __global__ void pivot_extract_output_rows_f64(
    const double* __restrict__ packed,
    int num_combos,
    int series_len,
    int output_index,
    double* __restrict__ out)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_combos * series_len;
    if (idx >= total) return;

    const int row = idx / series_len;
    const int col = idx - row * series_len;
    const int packed_row = row * LEVELS + output_index;
    out[idx] = packed[packed_row * series_len + col];
}
extern "C" __global__
void pivot_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    const double* __restrict__ open,
    const int*   __restrict__ modes,
    int n,
    int first_valid,
    int n_combos,
    double* __restrict__ out)
{

    if (blockIdx.y != 0) return;


    __shared__ unsigned char s_need_open_any;
    if (threadIdx.x == 0) {
        unsigned char f = 0;
        for (int j = 0; j < n_combos; ++j) {
            const int m = modes[j];
            f |= (m == 2) | (m == 4);
        }
        s_need_open_any = f;
    }
    __syncthreads();
    const bool need_open_any = (s_need_open_any != 0);


    const int stride = blockDim.x * gridDim.x;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride)
    {
        const double h = high[t];
        const double l = low[t];
        const double c = close[t];
        const double o = need_open_any ? open[t] : 0.0;


        const bool base_ok = (t >= first_valid) && (h == h) && (l == l) && (c == c);


        for (int j = 0; j < n_combos; ++j) {
            const int mode = modes[j];
            const bool need_o = (mode == 2) || (mode == 4);
            const bool valid  = base_ok && (!need_o || (o == o));


            const int base = (j * LEVELS) * n + t;

            double r4, r3, r2, r1, pp, s1, s2, s3, s4;
            if (valid) {
                pivot_compute_levels_core(mode, h, l, c, o, r4, r3, r2, r1, pp, s1, s2, s3, s4);
            } else {
                r4 = r3 = r2 = r1 = pp = s1 = s2 = s3 = s4 = f_nan_f64();
            }


            out[base + 0 * n] = r4;
            out[base + 1 * n] = r3;
            out[base + 2 * n] = r2;
            out[base + 3 * n] = r1;
            out[base + 4 * n] = pp;
            out[base + 5 * n] = s1;
            out[base + 6 * n] = s2;
            out[base + 7 * n] = s3;
            out[base + 8 * n] = s4;
        }
    }
}
extern "C" __global__
void pivot_many_series_one_param_time_major_f64(
    const double* __restrict__ high_tm,
    const double* __restrict__ low_tm,
    const double* __restrict__ close_tm,
    const double* __restrict__ open_tm,
    const int*   __restrict__ first_valids,
    int cols,
    int rows,
    int mode,
    double* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const bool need_o = (mode == 2) || (mode == 4);
    const int first_valid = first_valids[s];


    for (int t = 0; t < rows; ++t) {
        const int idx = t * cols + s;

        double r4, r3, r2, r1, pp, s1, s2, s3, s4;
        const double h = high_tm[idx];
        const double l = low_tm[idx];
        const double c = close_tm[idx];
        const double o = need_o ? open_tm[idx] : 0.0;

        const bool valid = (t >= first_valid) && (h == h) && (l == l) && (c == c) && (!need_o || (o == o));
        if (valid) {
            pivot_compute_levels_core(mode, h, l, c, o, r4, r3, r2, r1, pp, s1, s2, s3, s4);
        } else {
            r4 = r3 = r2 = r1 = pp = s1 = s2 = s3 = s4 = f_nan_f64();
        }


        out_tm[(0 * rows + t) * cols + s] = r4;
        out_tm[(1 * rows + t) * cols + s] = r3;
        out_tm[(2 * rows + t) * cols + s] = r2;
        out_tm[(3 * rows + t) * cols + s] = r1;
        out_tm[(4 * rows + t) * cols + s] = pp;
        out_tm[(5 * rows + t) * cols + s] = s1;
        out_tm[(6 * rows + t) * cols + s] = s2;
        out_tm[(7 * rows + t) * cols + s] = s3;
        out_tm[(8 * rows + t) * cols + s] = s4;
    }
}

/* ===========================================================================
 * f64 LANE  --  closer 2, round 2                                     pivot
 * ---------------------------------------------------------------------------
 * CPU reference: `pivot_scalar`, src/indicators/pivot.rs:536, mode-3 arm at
 * :670-721, reached through `pivot_with_kernel` (:261).
 *
 * PERIOD-INVARIANT. `pivot` has no period: its only parameter is `mode`, an
 * integer selecting WHICH pivot formula runs, default 3 (cpu_batch.rs:16734).
 * The lane sweeps `periods`, and a period list cannot be a mode list -- a
 * sweep over {5, 10, 20, 50} would silently ask for modes 5, 10, 20 and 50,
 * three of which do not exist (:757 `_ => {}` leaves the row untouched). So
 * `periods` is unread and mode 3 is fixed, exactly as the CPU batch defaults
 * it. Every row is the same series.
 *
 * OUTPUT is `pp`, which is what the dispatcher returns for `output_id` "value"
 * as well as "pp" (cpu_batch.rs:16743-16745). The other eight series this
 * indicator emits are not reachable through the lane, which carries one matrix.
 *
 * INPUT is Hlc, NOT Ohlc4. `extract_ohlc_full_input` hands the CPU batch an
 * `open` slice, but the mode-3 arm never reads it (:670-721 touches high, low
 * and close only) and the first-valid scan is over the same three (:271-282).
 * Declaring Ohlc4 would make `open` an input to an indicator that does not have
 * one, and would adopt a first-valid index the CPU never computes.
 *
 * Bar-independent, but launched one thread per combo because that is the shape
 * this lane launches; the loop over bars is the thread body.
 *
 * `(h + l + c) * (1.0 / 3.0)` (:705) is a MULTIPLY BY THE RECIPROCAL, not a
 * divide by three. `1.0/3.0` is a single rounding of one third and then one
 * rounding of the product; `(h + l + c) / 3.0` would round differently. Copied
 * literally.
 *
 * `first_valid` is the lane AllInputsNonNan over (high, low, close), which is
 * the CPU scan at :271-282 for the first bar where none of the three is NaN.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void pivot_neo_batch_f64(const double* __restrict__ high,
                         const double* __restrict__ low,
                         const double* __restrict__ close,
                         int n,
                         const int* __restrict__ periods,
                         int n_combos,
                         int first_valid,
                         double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods;   /* pivot has no period -- see the header above. */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    if (first_valid < 0 || first_valid >= n) return;

    for (int i = first_valid; i < n; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];
        if (isnan(h) || isnan(l) || isnan(c)) {
            o[i] = NEO_F64_NAN;                       /* pivot.rs:690-700 */
            continue;
        }
        o[i] = (h + l + c) * (1.0 / 3.0);             /* pivot.rs:705-706 */
    }
}
