#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


__device__ __forceinline__ float inv_or_one(const float x) {
    return (x != 0.0f) ? __fdividef(1.0f, x) : 1.0f;
}

__device__ __forceinline__ void mode_weights(const int mode, float& w_intra, float& w_group) {


    if (mode == 0) { w_intra = 0.5f; w_group = 0.5f; }
    else if (mode == 2) { w_intra = 0.0f; w_group = 1.0f; }
    else { w_intra = 1.0f; w_group = 0.0f; }
}


__device__ __forceinline__ void kahan_add(float y, float& sum, float& c) {
    float t  = y - c;
    float ns = sum + t;
    c        = (ns - sum) - t;
    sum      = ns;
}


struct ModHelper {
    const int period;
    const bool is_pow2;
    const int mask;
    __device__ __forceinline__ ModHelper(int p)
        : period(p), is_pow2((p & (p - 1)) == 0), mask(p - 1) {}
    __device__ __forceinline__ int inc_wrap(int x) const {
        if (is_pow2) return (x + 1) & mask;
        int nx = x + 1; return (nx == period) ? 0 : nx;
    }
    __device__ __forceinline__ int dec_wrap(int x) const {
        if (is_pow2) return (x - 1) & mask;
        return (x == 0) ? (period - 1) : (x - 1);
    }
    __device__ __forceinline__ int mod(int x) const {
        return is_pow2 ? (x & mask) : (x % period);
    }
};


extern "C" __global__ void aso_batch_f32(
    const float* __restrict__ open,
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const int*   __restrict__ periods,
    const int*   __restrict__ modes,
    const int*   __restrict__ log2_tbl,
    const int*   __restrict__ level_offsets,
    const float* __restrict__ st_max,
    const float* __restrict__ st_min,
    int series_len,
    int first_valid,
    int level_count,
    int n_combos,
    float* __restrict__ out_bulls,
    float* __restrict__ out_bears)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;

    const int period = periods[combo];

    const int base = combo * series_len;


    auto fill_all_nan = [&]() {
        for (int i = threadIdx.x; i < series_len; i += blockDim.x) {
            out_bulls[base + i] = NAN;
            out_bears[base + i] = NAN;
        }
    };

    if (UNLIKELY(period <= 0 || first_valid < 0 || first_valid >= series_len)) {
        fill_all_nan();
        return;
    }

    const int warm = first_valid + period - 1;
    if (UNLIKELY(warm >= series_len)) {
        fill_all_nan();
        return;
    }


    const int k = log2_tbl[period];
    if (UNLIKELY(k < 0 || k >= level_count)) {
        fill_all_nan();
        return;
    }


    const int  mode = modes[combo];
    float w_intra, w_group;
    mode_weights(mode, w_intra, w_group);


    for (int i = threadIdx.x; i < warm; i += blockDim.x) {
        out_bulls[base + i] = NAN;
        out_bears[base + i] = NAN;
    }
    __syncthreads();


    if (threadIdx.x != 0) return;


    const int offset   = 1 << k;
    const int lvl_base = level_offsets[k];
    const float* __restrict__ st_max_lvl = st_max + lvl_base;
    const float* __restrict__ st_min_lvl = st_min + lvl_base;


    extern __shared__ float smem[];
    float* ring_b = smem;
    float* ring_e = smem + period;


    for (int i = 0; i < period; ++i) { ring_b[i] = 0.0f; ring_e[i] = 0.0f; }

    ModHelper mh(period);
    int   head   = 0;
    int   filled = 0;
    float sum_b  = 0.0f;
    float sum_e  = 0.0f;


    int start     = warm - period + 1;
    int idx_a     = start;
    int idx_b     = warm + 1 - offset;
    int gopen_idx = start;


    for (int t = warm; t < series_len; ++t, ++start, ++idx_a, ++idx_b, ++gopen_idx) {
        const float o = open[t];
        const float h = high[t];
        const float l = low[t];
        const float c = close[t];


        const float intrarange    = h - l;
        const float scale1        = 50.0f * inv_or_one(intrarange);
        const float intrabarbulls = fmaf((c - l) + (h - o), scale1, 0.0f);
        const float intrabarbears = fmaf((h - c) + (o - l), scale1, 0.0f);


        const float gh    = fmaxf(st_max_lvl[idx_a], st_max_lvl[idx_b]);
        const float gl    = fminf(st_min_lvl[idx_a], st_min_lvl[idx_b]);
        const float gopen = open[gopen_idx];
        const float gr    = gh - gl;
        const float scale2        = 50.0f * inv_or_one(gr);
        const float groupbulls    = fmaf((c - gl) + (gh - gopen), scale2, 0.0f);
        const float groupbears    = fmaf((gh - c) + (gopen - gl), scale2, 0.0f);


        const float b = fmaf(w_intra, intrabarbulls, w_group * groupbulls);
        const float e = fmaf(w_intra, intrabarbears, w_group * groupbears);

        const float old_b = (filled == period) ? ring_b[head] : 0.0f;
        const float old_e = (filled == period) ? ring_e[head] : 0.0f;


        sum_b += (b - old_b);
        sum_e += (e - old_e);

        ring_b[head] = b;
        ring_e[head] = e;
        head = mh.inc_wrap(head);
        if (filled < period) ++filled;

        const float n = (float)filled;
        out_bulls[base + t] = __fdividef(sum_b, n);
        out_bears[base + t] = __fdividef(sum_e, n);
    }
}


extern "C" __global__ void aso_many_series_one_param_f32(
    const float* __restrict__ open_tm,
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const int*   __restrict__ first_valids,
    int cols,
    int rows,
    int period,
    int mode,
    float* __restrict__ out_bulls_tm,
    float* __restrict__ out_bears_tm)
{
    const int s = blockIdx.x;
    if (s >= cols) return;

    auto fill_all_nan = [&]() {
        for (int t = threadIdx.x; t < rows; t += blockDim.x) {
            const int idx = t * cols + s;
            out_bulls_tm[idx] = NAN;
            out_bears_tm[idx] = NAN;
        }
    };

    if (UNLIKELY(period <= 0)) {
        fill_all_nan();
        return;
    }

    const int fv   = first_valids[s];
    if (UNLIKELY(fv < 0 || fv >= rows)) {
        fill_all_nan();
        return;
    }
    const int warm = fv + period - 1;
    if (UNLIKELY(warm >= rows)) {
        fill_all_nan();
        return;
    }


    for (int t = threadIdx.x; t < warm; t += blockDim.x) {
        const int idx = t * cols + s;
        out_bulls_tm[idx] = NAN;
        out_bears_tm[idx] = NAN;
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    float w_intra, w_group;
    mode_weights(mode, w_intra, w_group);

    extern __shared__ unsigned char smem_uc[];
    float* ring_b     = reinterpret_cast<float*>(smem_uc);
    float* ring_e     = ring_b + period;
    int*   dq_min_idx = reinterpret_cast<int*>(ring_e + period);
    int*   dq_max_idx = dq_min_idx + period;

    for (int i = 0; i < period; ++i) {
        ring_b[i] = 0.0f; ring_e[i] = 0.0f;
        dq_min_idx[i] = 0; dq_max_idx[i] = 0;
    }

    ModHelper mh(period);
    int head = 0, filled = 0;
    float sum_b = 0.0f, cb = 0.0f;
    float sum_e = 0.0f, ce = 0.0f;


    int min_head = 0, min_tail = 0, min_len = 0;
    int max_head = 0, max_tail = 0, max_len = 0;


    int idx       = fv * cols + s;
    int start_idx = idx - (period - 1) * cols;

    for (int t = fv; t < rows; ++t, idx += cols, start_idx += cols) {
        const float o = open_tm[idx];
        const float h = high_tm[idx];
        const float l = low_tm[idx];
        const float c = close_tm[idx];


        while (min_len > 0) {
            int back = mh.dec_wrap(min_tail);
            int j    = dq_min_idx[back];
            float lj = low_tm[j * cols + s];
            if (l <= lj) { min_tail = back; --min_len; } else { break; }
        }
        if (min_len == period) { min_head = mh.inc_wrap(min_head); --min_len; }
        dq_min_idx[min_tail] = t; min_tail = mh.inc_wrap(min_tail); ++min_len;


        while (max_len > 0) {
            int back = mh.dec_wrap(max_tail);
            int j    = dq_max_idx[back];
            float hj = high_tm[j * cols + s];
            if (h >= hj) { max_tail = back; --max_len; } else { break; }
        }
        if (max_len == period) { max_head = mh.inc_wrap(max_head); --max_len; }
        dq_max_idx[max_tail] = t; max_tail = mh.inc_wrap(max_tail); ++max_len;

        if (t >= warm) {
            const int start = t - period + 1;
            while (min_len > 0 && dq_min_idx[min_head] < start) { min_head = mh.inc_wrap(min_head); --min_len; }
            while (max_len > 0 && dq_max_idx[max_head] < start) { max_head = mh.inc_wrap(max_head); --max_len; }

            const float gl    = low_tm[dq_min_idx[min_head] * cols + s];
            const float gh    = high_tm[dq_max_idx[max_head] * cols + s];
            const float gopen = open_tm[start_idx];


            const float intrarange    = h - l;
            const float scale1        = 50.0f * inv_or_one(intrarange);
            const float intrabarbulls = fmaf((c - l) + (h - o), scale1, 0.0f);
            const float intrabarbears = fmaf((h - c) + (o - l), scale1, 0.0f);


            const float gr            = gh - gl;
            const float scale2        = 50.0f * inv_or_one(gr);
            const float groupbulls    = fmaf((c - gl) + (gh - gopen), scale2, 0.0f);
            const float groupbears    = fmaf((gh - c) + (gopen - gl), scale2, 0.0f);

            const float b = fmaf(w_intra, intrabarbulls, w_group * groupbulls);
            const float e = fmaf(w_intra, intrabarbears, w_group * groupbears);

            const float old_b = (filled == period) ? ring_b[head] : 0.0f;
            const float old_e = (filled == period) ? ring_e[head] : 0.0f;

            kahan_add(b - old_b, sum_b, cb);
            kahan_add(e - old_e, sum_e, ce);

            ring_b[head] = b; ring_e[head] = e;
            head = mh.inc_wrap(head);
            if (filled < period) ++filled;

            const float n = (float)filled;
            out_bulls_tm[idx] = __fdividef(sum_b, n);
            out_bears_tm[idx] = __fdividef(sum_e, n);
        }
    }
}

// ===========================================================================
// S3 f64 LANE — aso (Average Sentiment Oscillator, bulls line)
// ===========================================================================
// Reference: src/indicators/aso.rs
//   aso_prepare (:370)                   — first_valid + the Err branches
//   aso_selected_value (:490)            — the per-bar value
//   aso_scalar_output_selected (:527)    — the window scan and the running mean
// Batch defaults: period 10, mode 0. Inputs: OPEN, high, low, close.
//
// WHICH OUTPUT. aso is multi-output (bulls / bears). compute_aso_batch maps
// output_id "value" — the id this lane requests — to BULLS, so this kernel is
// the bulls line.
//
// FIRST-VALID IS CLOSE ALONE (:405-408):
//     close.iter().position(|x| !x.is_nan())
// Open, high and low are NEVER scanned. That is the same rule adxr uses, which
// is why the registry row says F64FirstValidRule::HlcCloseOnly and not
// AllInputsNonNan: on a frame where high or low starts late, the two rules name
// different bars and every value after the seed differs.
//
// WHICH BRANCH. period <= 64 (DEQUE_THRESHOLD, :544) takes the LINEAR-SCAN
// branch, and the batch default period is 10. The scan is O(period) per bar and
// is reproduced exactly; the >64 monotone-deque branch computes the same
// extremes by a different route and is not the path a default sweep takes. A
// period above 64 in the sweep would take the CPU's deque branch — which
// produces the same gl/gh values, since both are exact extreme selection over
// the same window under the same comparisons.
//
// THE SENTINELS ARE f64::MAX AND f64::MIN, AND THEY ARE LOAD-BEARING (:555-556).
//   gl starts at  f64::MAX =  1.7976931348623157e308
//   gh starts at  f64::MIN = -1.7976931348623157e308   (the most NEGATIVE
//                                                       finite double, not
//                                                       MIN_POSITIVE)
// with if-chains, not fmin/fmax: `if lj < gl` is false for a NaN lj, so NaNs
// are SKIPPED and an all-NaN window leaves the SENTINEL in place — a huge
// finite number, not NaN. fmin/fmax would give the identical answer on finite
// data and a DIFFERENT one here, so the chain is kept.
//
// The f32 kernels above cannot hold these sentinels at all: FLT_MAX is 3.4e38,
// so an f32 port silently substitutes a bound 270 orders of magnitude smaller.
//
// THE RING BUFFER IS NOT NEEDED. The CPU keeps ring[period] of past v values
// (:546, :579-581) so it can subtract the one leaving the mean. v is a PURE
// FUNCTION of the bar and its window, so the retired element is v(i - period),
// recomputed here. The head walk proves the index: at i == warm + k the CPU
// reads ring[head] only once filled == period, i.e. k >= period, and head has
// cycled to (k - period) mod period, which holds v(warm + k - period).
//
// filled is min(i - warm + 1, period) — the divisor GROWS for the first
// `period` outputs rather than being a fixed /period. Reproduced.
//
// k1 / k2 ARE ZERO GUARDS, NOT EPSILONS (:501, :503): intrarange == 0.0 → 1.0,
// tested exactly. No tolerance in the reference, none invented here.
//
// One thread per column: the running sum is carried across bars.
// ===========================================================================

#define NEO_S3_ASO_MODE 0

__device__ __forceinline__ double neo_s3_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

// aso_selected_value::<true> — the bulls branch, mode as a runtime value.
__device__ __forceinline__ double neo_s3_aso_bulls(
    double oi, double hi, double li, double ci,
    double gl, double gh, double gopen, int mode)
{
    const double intrarange = hi - li;
    const double k1 = (intrarange == 0.0) ? 1.0 : intrarange;
    const double gr = gh - gl;
    const double k2 = (gr == 0.0) ? 1.0 : gr;

    const double intrabar = (((ci - li) + (hi - oi)) * 50.0) / k1;
    const double group    = (((ci - gl) + (gh - gopen)) * 50.0) / k2;

    if (mode == 1) return intrabar;
    if (mode == 2) return group;
    return 0.5 * (intrabar + group);   // mode 0 and the CPU's catch-all
}

// The window value at bar `i`: gl/gh scanned over [i-period+1, i] with the
// CPU's sentinels and if-chains.
__device__ __forceinline__ double neo_s3_aso_value_at(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int i, int period, int mode)
{
    const int start = i + 1 - period;
    double gl =  1.7976931348623157e308;   // f64::MAX
    double gh = -1.7976931348623157e308;   // f64::MIN
    for (int j = start; j <= i; ++j) {
        const double lj = low[j];
        const double hj = high[j];
        if (lj < gl) gl = lj;
        if (hj > gh) gh = hj;
    }
    return neo_s3_aso_bulls(open[i], high[i], low[i], close[i], gl, gh, open[start], mode);
}

extern "C" __global__ void neoethos_aso_batch_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    double* __restrict__ row = out + (size_t)r * (size_t)n;
    const int period = periods[r];
    const int mode = NEO_S3_ASO_MODE;

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (period == 0) || (period > n) ||
        (mode > 2) ||
        ((n - first_valid) < period);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s3_qnan();
        return;
    }

    const int warm = first_valid + period - 1;
    // aso_output_into_slice fills the prefix with NaN; the scalar writes only
    // from `warm` on (:552).
    for (int i = 0; i < warm && i < n; ++i) row[i] = neo_s3_qnan();
    if (warm >= n) return;

    double sum = 0.0;
    for (int i = warm; i < n; ++i) {
        const double v = neo_s3_aso_value_at(open, high, low, close, i, period, mode);

        const int k = i - warm;                       // 0-based output index
        const int filled = (k < period) ? (k + 1) : period;
        const double old = (k >= period)
            ? neo_s3_aso_value_at(open, high, low, close, i - period, period, mode)
            : 0.0;

        sum += v - old;
        row[i] = sum / (double)filled;
    }
}
