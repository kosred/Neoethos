#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>


static __device__ __forceinline__ int find_first_finite(const float* v, int start, int len) {
    for (int i = start; i < len; ++i) {
        if (isfinite(v[i])) return i;
    }
    return len;
}


extern "C" __global__
void ott_apply_single_f32(const float* __restrict__ ma,
                          int series_len,
                          float percent,
                          float* __restrict__ out) {
    if (threadIdx.x != 0) return;
    if (series_len <= 0) return;

    const float fark = percent * 0.01f;
    const float scale_minus = 1.0f - percent * 0.005f;
    const float scale_plus = scale_minus + fark;


    int i = find_first_finite(ma, 0, series_len);
    if (i >= series_len) return;

    float m = ma[i];
    float long_stop = fmaf(-fark, m, m);
    float short_stop = fmaf( fark, m, m);
    int dir = 1;


    float mt0 = long_stop;
    float scale0 = (m > mt0) ? scale_plus : scale_minus;
    out[i] = mt0 * scale0;
    ++i;

    for (; i < series_len; ++i) {
        float mf = ma[i];
        if (!isfinite(mf)) continue;
        float mavg = mf;

        float cand_long = fmaf(-fark, mavg, mavg);
        float cand_short = fmaf( fark, mavg, mavg);

        float lprev = long_stop;
        float sprev = short_stop;


        if (mavg > lprev) {
            long_stop = (cand_long > lprev) ? cand_long : lprev;
        } else {
            long_stop = cand_long;
        }
        if (mavg < sprev) {
            short_stop = (cand_short < sprev) ? cand_short : sprev;
        } else {
            short_stop = cand_short;
        }


        if (dir == -1 && mavg > sprev) {
            dir = 1;
        } else if (dir == 1 && mavg < lprev) {
            dir = -1;
        }


        float mt = (dir == 1) ? long_stop : short_stop;
        float scale = (mavg > mt) ? scale_plus : scale_minus;
        out[i] = mt * scale;
    }
}


static __device__ __forceinline__ float vidya_alpha_base(int period) {
    return 2.0f / ((float)period + 1.0f);
}


extern "C" __global__
void ott_from_var_batch_f32(const float* __restrict__ prices,
                            const int*   __restrict__ periods,
                             const float* __restrict__ percents,
                             int series_len,
                             int n_combos,
                             float* __restrict__ out) {
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const float percent = percents[combo];
    if (period <= 0 || series_len <= 0 || !isfinite(percent)) return;

    float* __restrict__ out_row = out + combo * series_len;


    int first = -1;
    for (int i = 0; i < series_len; ++i) {
        if (isfinite(prices[i])) { first = i; break; }
    }
    if (first < 0) return;

    const float fark = percent * 0.01f;
    const float scale_minus = 1.0f - percent * 0.005f;
    const float scale_plus = scale_minus + fark;
    const float valpha_base = vidya_alpha_base(period);


    float ring_u[9];
    float ring_d[9];
    #pragma unroll
    for (int k = 0; k < 9; ++k) { ring_u[k] = 0.0; ring_d[k] = 0.0; }
    float u_sum = 0.0f, d_sum = 0.0f;
    int ridx = 0;


    float var = 0.0f;


    float long_stop = fmaf(-fark, var, var);
    float short_stop = fmaf( fark, var, var);
    int dir = 1;


    float mt0 = long_stop;
    float scale0 = (var > mt0) ? scale_plus : scale_minus;
    out_row[first] = mt0 * scale0;


    int pre_end = (first + 8 < series_len ? first + 8 : series_len - 1);
    for (int i = first + 1; i <= pre_end; ++i) {
        float a = prices[i - 1];
        float b = prices[i];
        if (!isfinite(a) || !isfinite(b)) continue;
        float up = b - a; if (up < 0.0f) up = 0.0f;
        float dn = a - b; if (dn < 0.0f) dn = 0.0f;
        ring_u[ridx] = up;  u_sum += up;
        ring_d[ridx] = dn;  d_sum += dn;
        if (++ridx == 9) ridx = 0;


        float cand_long = fmaf(-fark, var, var);
        float cand_short = fmaf( fark, var, var);
        float lprev = long_stop, sprev = short_stop;
        if (var > lprev) long_stop = (cand_long > lprev) ? cand_long : lprev; else long_stop = cand_long;
        if (var < sprev) short_stop = (cand_short < sprev) ? cand_short : sprev; else short_stop = cand_short;
        if (dir == -1 && var > sprev) dir = 1; else if (dir == 1 && var < lprev) dir = -1;
        float mt = (dir == 1) ? long_stop : short_stop;
        float scale = (var > mt) ? scale_plus : scale_minus;
        out_row[i] = mt * scale;
    }


    for (int i = first + 9; i < series_len; ++i) {
        float a = prices[i - 1];
        float b = prices[i];
        if (!isfinite(a) || !isfinite(b)) continue;
        float up = b - a; if (up < 0.0f) up = 0.0f;
        float dn = a - b; if (dn < 0.0f) dn = 0.0f;
        float old_u = ring_u[ridx];
        float old_d = ring_d[ridx];
        ring_u[ridx] = up; ring_d[ridx] = dn;
        if (++ridx == 9) ridx = 0;
        u_sum += up - old_u;
        d_sum += dn - old_d;
        float denom = u_sum + d_sum;
        float vcmo = (denom != 0.0f) ? ((u_sum - d_sum) / denom) : 0.0f;
        float avalpha = valpha_base * fabsf(vcmo);
        var = fmaf(avalpha, b, (1.0f - avalpha) * var);


        float cand_long = fmaf(-fark, var, var);
        float cand_short = fmaf( fark, var, var);
        float lprev = long_stop, sprev = short_stop;
        if (var > lprev) long_stop = (cand_long > lprev) ? cand_long : lprev; else long_stop = cand_long;
        if (var < sprev) short_stop = (cand_short < sprev) ? cand_short : sprev; else short_stop = cand_short;
        if (dir == -1 && var > sprev) dir = 1; else if (dir == 1 && var < lprev) dir = -1;
        float mt = (dir == 1) ? long_stop : short_stop;
        float scale = (var > mt) ? scale_plus : scale_minus;
        out_row[i] = mt * scale;
    }
}

extern "C" __global__
void ott_from_var_batch_f32_all_finite(const float* __restrict__ prices,
                                       const int*   __restrict__ periods,
                                       const float* __restrict__ percents,
                                       int series_len,
                                       int n_combos,
                                       float* __restrict__ out) {
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const float percent = percents[combo];
    if (period <= 0 || series_len <= 0 || !isfinite(percent)) return;

    float* __restrict__ out_row = out + combo * series_len;

    const float fark = percent * 0.01f;
    const float scale_minus = 1.0f - percent * 0.005f;
    const float scale_plus = scale_minus + fark;
    const float valpha_base = vidya_alpha_base(period);

    float ring_u[9];
    float ring_d[9];
    #pragma unroll
    for (int k = 0; k < 9; ++k) { ring_u[k] = 0.0f; ring_d[k] = 0.0f; }
    float u_sum = 0.0f, d_sum = 0.0f;
    int ridx = 0;

    float var = 0.0f;
    float long_stop = fmaf(-fark, var, var);
    float short_stop = fmaf( fark, var, var);
    int dir = 1;
    float mt0 = long_stop;
    float scale0 = (var > mt0) ? scale_plus : scale_minus;
    out_row[0] = mt0 * scale0;

    int pre_end = (8 < series_len ? 8 : series_len - 1);
    for (int i = 1; i <= pre_end; ++i) {
        float a = prices[i - 1];
        float b = prices[i];
        float up = b - a; if (up < 0.0f) up = 0.0f;
        float dn = a - b; if (dn < 0.0f) dn = 0.0f;
        ring_u[ridx] = up;  u_sum += up;
        ring_d[ridx] = dn;  d_sum += dn;
        if (++ridx == 9) ridx = 0;

        float cand_long = fmaf(-fark, var, var);
        float cand_short = fmaf( fark, var, var);
        float lprev = long_stop, sprev = short_stop;
        if (var > lprev) long_stop = (cand_long > lprev) ? cand_long : lprev; else long_stop = cand_long;
        if (var < sprev) short_stop = (cand_short < sprev) ? cand_short : sprev; else short_stop = cand_short;
        if (dir == -1 && var > sprev) dir = 1; else if (dir == 1 && var < lprev) dir = -1;
        float mt = (dir == 1) ? long_stop : short_stop;
        float scale = (var > mt) ? scale_plus : scale_minus;
        out_row[i] = mt * scale;
    }

    for (int i = 9; i < series_len; ++i) {
        float a = prices[i - 1];
        float b = prices[i];
        float up = b - a; if (up < 0.0f) up = 0.0f;
        float dn = a - b; if (dn < 0.0f) dn = 0.0f;
        float old_u = ring_u[ridx];
        float old_d = ring_d[ridx];
        ring_u[ridx] = up; ring_d[ridx] = dn;
        if (++ridx == 9) ridx = 0;
        u_sum += up - old_u;
        d_sum += dn - old_d;
        float denom = u_sum + d_sum;
        float vcmo = (denom != 0.0f) ? ((u_sum - d_sum) / denom) : 0.0f;
        float avalpha = valpha_base * fabsf(vcmo);
        var = fmaf(avalpha, b, (1.0f - avalpha) * var);

        float cand_long = fmaf(-fark, var, var);
        float cand_short = fmaf( fark, var, var);
        float lprev = long_stop, sprev = short_stop;
        if (var > lprev) long_stop = (cand_long > lprev) ? cand_long : lprev; else long_stop = cand_long;
        if (var < sprev) short_stop = (cand_short < sprev) ? cand_short : sprev; else short_stop = cand_short;
        if (dir == -1 && var > sprev) dir = 1; else if (dir == 1 && var < lprev) dir = -1;
        float mt = (dir == 1) ? long_stop : short_stop;
        float scale = (var > mt) ? scale_plus : scale_minus;
        out_row[i] = mt * scale;
    }
}

extern "C" __global__
void cmo9_from_prices_f32_all_finite(const float* __restrict__ prices,
                                     int series_len,
                                     float* __restrict__ vcmo_out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (series_len <= 0) return;

    vcmo_out[0] = 0.0f;
    if (series_len == 1) return;

    float ring_u[9];
    float ring_d[9];
    #pragma unroll
    for (int k = 0; k < 9; ++k) { ring_u[k] = 0.0f; ring_d[k] = 0.0f; }
    float u_sum = 0.0f, d_sum = 0.0f;
    int ridx = 0;

    int pre_end = (8 < series_len ? 8 : series_len - 1);
    for (int i = 1; i <= pre_end; ++i) {
        float a = prices[i - 1];
        float b = prices[i];
        float up = b - a; if (up < 0.0f) up = 0.0f;
        float dn = a - b; if (dn < 0.0f) dn = 0.0f;
        ring_u[ridx] = up;  u_sum += up;
        ring_d[ridx] = dn;  d_sum += dn;
        if (++ridx == 9) ridx = 0;
        vcmo_out[i] = 0.0f;
    }

    for (int i = 9; i < series_len; ++i) {
        float a = prices[i - 1];
        float b = prices[i];
        float up = b - a; if (up < 0.0f) up = 0.0f;
        float dn = a - b; if (dn < 0.0f) dn = 0.0f;
        float old_u = ring_u[ridx];
        float old_d = ring_d[ridx];
        ring_u[ridx] = up; ring_d[ridx] = dn;
        if (++ridx == 9) ridx = 0;
        u_sum += up - old_u;
        d_sum += dn - old_d;
        float denom = u_sum + d_sum;
        vcmo_out[i] = (denom != 0.0f) ? ((u_sum - d_sum) / denom) : 0.0f;
    }
}

extern "C" __global__
void ott_from_vcmo_batch_f32_all_finite(const float* __restrict__ prices,
                                        const float* __restrict__ vcmo,
                                        const int*   __restrict__ periods,
                                        const float* __restrict__ percents,
                                        int series_len,
                                        int n_combos,
                                        float* __restrict__ out) {
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const float percent = percents[combo];
    if (period <= 0 || series_len <= 0 || !isfinite(percent)) return;

    float* __restrict__ out_row = out + combo * series_len;

    const float fark = percent * 0.01f;
    const float scale_minus = 1.0f - percent * 0.005f;
    const float scale_plus = scale_minus + fark;
    const float valpha_base = vidya_alpha_base(period);

    float var = 0.0f;
    float long_stop = fmaf(-fark, var, var);
    float short_stop = fmaf( fark, var, var);
    int dir = 1;
    out_row[0] = 0.0f;

    int pre_end = (8 < series_len ? 8 : series_len - 1);
    for (int i = 1; i <= pre_end; ++i) {
        out_row[i] = 0.0f;
    }

    for (int i = 9; i < series_len; ++i) {
        float b = prices[i];
        float avalpha = valpha_base * fabsf(vcmo[i]);
        var = fmaf(avalpha, b, (1.0f - avalpha) * var);

        float cand_long = fmaf(-fark, var, var);
        float cand_short = fmaf( fark, var, var);
        float lprev = long_stop, sprev = short_stop;
        if (var > lprev) long_stop = (cand_long > lprev) ? cand_long : lprev; else long_stop = cand_long;
        if (var < sprev) short_stop = (cand_short < sprev) ? cand_short : sprev; else short_stop = cand_short;
        if (dir == -1 && var > sprev) dir = 1; else if (dir == 1 && var < lprev) dir = -1;
        float mt = (dir == 1) ? long_stop : short_stop;
        float scale = (var > mt) ? scale_plus : scale_minus;
        out_row[i] = mt * scale;
    }
}


extern "C" __global__
void ott_many_series_one_param_f32(const float* __restrict__ ma_tm,
                                   int cols,
                                   int rows,
                                   float percent,
                                   float* __restrict__ out_tm) {

    const int s = blockIdx.x;
    if (s >= cols || threadIdx.x != 0) return;
    if (rows <= 0) return;

    const float fark = percent * 0.01f;
    const float scale_minus = 1.0f - percent * 0.005f;
    const float scale_plus = scale_minus + fark;


    int t = 0;
    for (; t < rows; ++t) { if (isfinite(ma_tm[(size_t)t * (size_t)cols + s])) break; }
    if (t >= rows) return;

    float m = ma_tm[(size_t)t * (size_t)cols + s];
    float long_stop = fmaf(-fark, m, m);
    float short_stop = fmaf( fark, m, m);
    int dir = 1;
    float mt0 = long_stop;
    float scale0 = (m > mt0) ? scale_plus : scale_minus;
    out_tm[(size_t)t * (size_t)cols + s] = mt0 * scale0;
    ++t;
    for (; t < rows; ++t) {
        float mf = ma_tm[(size_t)t * (size_t)cols + s];
        if (!isfinite(mf)) continue;
        float mavg = mf;
        float cand_long = fmaf(-fark, mavg, mavg);
        float cand_short = fmaf( fark, mavg, mavg);
        float lprev = long_stop, sprev = short_stop;
        if (mavg > lprev) long_stop = (cand_long > lprev) ? cand_long : lprev; else long_stop = cand_long;
        if (mavg < sprev) short_stop = (cand_short < sprev) ? cand_short : sprev; else short_stop = cand_short;
        if (dir == -1 && mavg > sprev) dir = 1; else if (dir == 1 && mavg < lprev) dir = -1;
        float mt = (dir == 1) ? long_stop : short_stop;
        float scale = (mavg > mt) ? scale_plus : scale_minus;
        out_tm[(size_t)t * (size_t)cols + s] = mt * scale;
    }
}


extern "C" __global__
void ott_from_var_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                            int cols,
                                            int rows,
                                            int period,
                                            float percent,
                                            float* __restrict__ out_tm) {
    const int s = blockIdx.x;
    if (s >= cols || threadIdx.x != 0) return;

    const float fark = percent * 0.01f;
    const float scale_minus = 1.0f - percent * 0.005f;
    const float scale_plus = scale_minus + fark;
    const float valpha_base = vidya_alpha_base(period);


    int first = -1;
    for (int t = 0; t < rows; ++t) { if (isfinite(prices_tm[(size_t)t * (size_t)cols + s])) { first = t; break; } }
    if (first < 0) return;


    float ring_u[9];
    float ring_d[9];
    #pragma unroll
    for (int k = 0; k < 9; ++k) { ring_u[k] = 0.0; ring_d[k] = 0.0; }
    float u_sum = 0.0f, d_sum = 0.0f; int ridx = 0;
    float var = 0.0f;


    float long_stop = fmaf(-fark, var, var);
    float short_stop = fmaf( fark, var, var);
    int dir = 1;
    float mt0 = long_stop; float scale0 = (var > mt0) ? scale_plus : scale_minus;
    out_tm[(size_t)first * (size_t)cols + s] = mt0 * scale0;

    int pre_end = (first + 8 < rows ? first + 8 : rows - 1);
    for (int t = first + 1; t <= pre_end; ++t) {
        float a = prices_tm[(size_t)(t - 1) * (size_t)cols + s];
        float b = prices_tm[(size_t)t * (size_t)cols + s];
        if (!isfinite(a) || !isfinite(b)) continue;
        float up = b - a; if (up < 0.0f) up = 0.0f;
        float dn = a - b; if (dn < 0.0f) dn = 0.0f;
        ring_u[ridx] = up; u_sum += up; ring_d[ridx] = dn; d_sum += dn; if (++ridx == 9) ridx = 0;
        float cand_long = fmaf(-fark, var, var);
        float cand_short = fmaf( fark, var, var);
        float lprev = long_stop, sprev = short_stop;
        if (var > lprev) long_stop = (cand_long > lprev) ? cand_long : lprev; else long_stop = cand_long;
        if (var < sprev) short_stop = (cand_short < sprev) ? cand_short : sprev; else short_stop = cand_short;
        if (dir == -1 && var > sprev) dir = 1; else if (dir == 1 && var < lprev) dir = -1;
        float mt = (dir == 1) ? long_stop : short_stop;
        float scale = (var > mt) ? scale_plus : scale_minus;
        out_tm[(size_t)t * (size_t)cols + s] = mt * scale;
    }
    for (int t = first + 9; t < rows; ++t) {
        float a = prices_tm[(size_t)(t - 1) * (size_t)cols + s];
        float b = prices_tm[(size_t)t * (size_t)cols + s];
        if (!isfinite(a) || !isfinite(b)) continue;
        float up = b - a; if (up < 0.0f) up = 0.0f; float dn = a - b; if (dn < 0.0f) dn = 0.0f;
        float old_u = ring_u[ridx]; float old_d = ring_d[ridx]; ring_u[ridx] = up; ring_d[ridx] = dn; if (++ridx == 9) ridx = 0;
        u_sum += up - old_u; d_sum += dn - old_d; float denom = u_sum + d_sum; float vcmo = (denom != 0.0f) ? ((u_sum - d_sum) / denom) : 0.0f;
        float avalpha = valpha_base * fabsf(vcmo); var = fmaf(avalpha, b, (1.0f - avalpha) * var);
        float cand_long = fmaf(-fark, var, var); float cand_short = fmaf( fark, var, var);
        float lprev = long_stop, sprev = short_stop;
        if (var > lprev) long_stop = (cand_long > lprev) ? cand_long : lprev; else long_stop = cand_long;
        if (var < sprev) short_stop = (cand_short < sprev) ? cand_short : sprev; else short_stop = cand_short;
        if (dir == -1 && var > sprev) dir = 1; else if (dir == 1 && var < lprev) dir = -1;
        float mt = (dir == 1) ? long_stop : short_stop; float scale = (var > mt) ? scale_plus : scale_minus;
        out_tm[(size_t)t * (size_t)cols + s] = mt * scale;
    }
}


// ===========================================================================
// NEOETHOS f64 LANE  --  closer 4, round 3
//
// CPU reference: calculate_var_ma (src/indicators/ott.rs:627-700) feeding
// ott_scalar (:865-945), which is what ott_with_kernel (:275) takes for every
// period through ott_compute_into (:701).
//
// The period == 2 fast path at :283 (ott_default_var_2_1_4_into, :392) is NOT
// a second formula: with valpha = 2/3, fark = 0.014 and scale_minus = 0.993 it
// is the general path constant-folded, and the two agree bar for bar --
// var_ma[first] is 0.0, so ott_scalar seeds long_stop and short_stop at 0.0
// and emits 0.0 across the pre-warm window exactly as the fast path writes it.
// One body therefore serves both.
//
// PERIOD-SWEPT: PARAM_OTT (registry.rs:9472) declares `period` with default 2
// and `percent` with default 1.4, and ott has no cpu_batch arm -- its oracle
// is ott_with_kernel, which reads `period` directly. ma_type is "VAR"
// (ott.rs:145), which is the only arm this kernel reproduces; the SMA / EMA /
// WMA / TMA / WWMA / ZLEMA / TSF arms of calculate_moving_average (:500) are
// unreachable at the default.
//
// SHAPE: one thread per combo walking bars ASCENDING. The VAR moving average
// is a variable-alpha EMA driven by a nine-wide CMO ring, and the band is a
// ratchet whose long and short stops read their own previous values. Neither
// can be rebuilt bar-parallel without changing the rounding.
//
// The nine-wide ring is a CONSTANT of the indicator (:636), not a function of
// `period`, so this kernel declares no max_period and NEVER-OOM holds by
// construction.
// ===========================================================================

#define OTT_NEO_CMO_P 9

static __forceinline__ __device__ double ott_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__
void ott_neo_batch_f64(const double* __restrict__ data,
                       int n,
                       const int* __restrict__ periods,
                       int n_combos,
                       int first_valid,
                       double* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;
    if (n <= 0) return;

    double* __restrict__ row = out + (size_t)combo * (size_t)n;
    const double nn = ott_neo_qnan();

    const int period = periods[combo];
    const double percent = 1.4;   // registry.rs:9490 / ott.rs:138

    // ott_prepare, :349-352 -- `!is_nan` over the single close series, which
    // is F64FirstValidRule::AllInputsNonNan for CloseSlice.
    int first = first_valid;
    if (first < 0) first = 0;

    bool refused = false;
    if (first >= n) refused = true;
    if (period <= 0 || period > n) refused = true;            // :358
    if (!refused && (n - first) < period) refused = true;     // :365
    // percent is 1.4 -- finite and >= 0, so :372 never refuses here.

    if (refused) {
        for (int i = 0; i < n; ++i) row[i] = nn;
        return;
    }

    // The VAR moving average has its own first-non-NaN scan (:628) over the
    // same series, so ma_first == first and the NaN prefix is `first`.
    for (int i = 0; i < n; ++i) row[i] = nn;

    const double valpha = 2.0 / ((double)period + 1.0);
    const double fark = percent * 0.01;
    const double scale_minus = 1.0 - (percent * 0.005);

    double ring_u[OTT_NEO_CMO_P];
    double ring_d[OTT_NEO_CMO_P];
    for (int k = 0; k < OTT_NEO_CMO_P; ++k) { ring_u[k] = 0.0; ring_d[k] = 0.0; }
    double u_sum = 0.0, d_sum = 0.0;
    int idx = 0;

    // var_ma[first] = 0.0 (:643), and ott_scalar seeds from it.
    double var = 0.0;
    double long_stop = fma(0.0, -fark, 0.0);
    double short_stop = fma(0.0, fark, 0.0);
    int dir = 1;
    {
        const double mt0 = long_stop;
        const double scale0 = (0.0 > mt0) ? (scale_minus + fark) : scale_minus;
        row[first] = mt0 * scale0;
    }

    // :645-661 -- the nine-bar CMO warm-up. var stays 0.0 across it, so the
    // band emits its seed value at every one of those bars.
    const int pre_end_raw = first + 8;
    const int pre_end = pre_end_raw < (n - 1) ? pre_end_raw : (n - 1);
    for (int i = first + 1; i <= pre_end; ++i) {
        const double a = data[i - 1];
        const double b = data[i];
        if (!isnan(a) && !isnan(b)) {
            const double up = fmax(b - a, 0.0);
            const double down = fmax(a - b, 0.0);
            ring_u[idx] = up;
            u_sum += up;
            ring_d[idx] = down;
            d_sum += down;
            idx = (idx + 1) % OTT_NEO_CMO_P;
        }
        // ott_scalar, :904-943, with mavg == var == 0.0.
        const double mavg = var;
        const double cand_long = fma(mavg, -fark, mavg);
        const double cand_short = fma(mavg, fark, mavg);
        const double lprev = long_stop;
        const double sprev = short_stop;
        long_stop = (mavg > lprev) ? ((cand_long > lprev) ? cand_long : lprev) : cand_long;
        short_stop = (mavg < sprev) ? ((cand_short < sprev) ? cand_short : sprev) : cand_short;
        if (dir == -1 && mavg > sprev) dir = 1;
        else if (dir == 1 && mavg < lprev) dir = -1;
        const double mt = (dir == 1) ? long_stop : short_stop;
        const double scale = (mavg > mt) ? (scale_minus + fark) : scale_minus;
        row[i] = mt * scale;
    }

    if ((n - first) <= OTT_NEO_CMO_P) return;   // :663

    for (int i = first + OTT_NEO_CMO_P; i < n; ++i) {
        const double a = data[i - 1];
        const double b = data[i];
        // :670 -- a NaN pair leaves var_ma[i] carrying the previous value,
        // which ott_scalar then reads as a repeat of the previous mavg.
        if (!isnan(a) && !isnan(b)) {
            const double old_u = ring_u[idx];
            const double old_d = ring_d[idx];
            const double up = fmax(b - a, 0.0);
            const double down = fmax(a - b, 0.0);

            u_sum += up - old_u;
            d_sum += down - old_d;

            ring_u[idx] = up;
            ring_d[idx] = down;
            idx = (idx + 1) % OTT_NEO_CMO_P;

            const double denom = u_sum + d_sum;
            const double vcmo = (denom != 0.0) ? ((u_sum - d_sum) / denom) : 0.0;
            const double vabs = fabs(vcmo);
            var = valpha * vabs * b + (1.0 - valpha * vabs) * var;
        }

        const double mavg = var;
        const double cand_long = fma(mavg, -fark, mavg);
        const double cand_short = fma(mavg, fark, mavg);
        const double lprev = long_stop;
        const double sprev = short_stop;
        long_stop = (mavg > lprev) ? ((cand_long > lprev) ? cand_long : lprev) : cand_long;
        short_stop = (mavg < sprev) ? ((cand_short < sprev) ? cand_short : sprev) : cand_short;
        if (dir == -1 && mavg > sprev) dir = 1;
        else if (dir == 1 && mavg < lprev) dir = -1;
        const double mt = (dir == 1) ? long_stop : short_stop;
        const double scale = (mavg > mt) ? (scale_minus + fark) : scale_minus;
        row[i] = mt * scale;
    }
}
