#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

namespace {
__device__ inline float compute_alpha_f(int length) {
    return 2.0f / (static_cast<float>(length) + 1.0f);
}
__device__ inline float compute_beta_f(float a) { return 1.0f - a; }


__device__ inline double compute_alpha(int length) {
    return 2.0 / (static_cast<double>(length) + 1.0);
}
__device__ inline double compute_beta(double a) { return 1.0 - a; }

struct KahanState { float y; float c; };

__device__ inline void kahan_add(KahanState& s, float x) {
    float y = x - s.c;
    float t = s.y + y;
    s.c = (t - s.y) - y;
    s.y = t;
}

__device__ inline void ema_step(KahanState& ema, float alpha, float x) {
    float delta = fmaf(alpha, (x - ema.y), 0.0f);
    kahan_add(ema, delta);
}

__device__ inline float ec_step(KahanState& ec, float alpha, float ema_val, float src, float gain) {
    float delta = fmaf(alpha, (ema_val - ec.y) + gain * (src - ec.y), 0.0f);
    kahan_add(ec, delta);
    return ec.y;
}

__device__ inline float clamp_prev_ec_f(bool pine_mode, float ema_value) {
    return pine_mode ? 0.0f : ema_value;
}

__device__ inline double clamp_prev_ec(bool pine_mode, double ema_value) {
    return pine_mode ? 0.0 : ema_value;
}

__device__ inline float pick_src_f(const float* prices, int idx, bool confirmed) {
    int source_idx = confirmed && idx > 0 ? idx - 1 : idx;
    return prices[source_idx];
}

__device__ inline float pick_src_tm_f(
    const float* prices_tm,
    int idx,
    int series,
    int num_series,
    bool confirmed
) {
    int row = confirmed && idx > 0 ? idx - 1 : idx;
    return prices_tm[row * num_series + series];
}

__device__ inline double pick_src(const float* prices, int idx, bool confirmed) {
    int source_idx = confirmed && idx > 0 ? idx - 1 : idx;
    return static_cast<double>(prices[source_idx]);
}


__device__ __forceinline__ int quantize_to_0p1_tie_down_i(double gstar_times10, int gain_limit_times10) {
    int gi = __double2int_rd(gstar_times10);
    double frac = gstar_times10 - static_cast<double>(gi);
    if (frac > 0.5) gi += 1;
    if (gi >  gain_limit_times10) gi =  gain_limit_times10;
    if (gi < -gain_limit_times10) gi = -gain_limit_times10;
    return gi;
}

__device__ __forceinline__ float choose_best_gain_f(float alpha, float ema_val, float prev, float src, int gain_limit) {
    const double a = static_cast<double>(alpha);
    const double e = static_cast<double>(ema_val);
    const double p = static_cast<double>(prev);
    const double s = static_cast<double>(src);
    if (!isfinite(a) || !isfinite(e) || !isfinite(p) || !isfinite(s)) { return 0.0f; }

    const double base = a * e + (1.0 - a) * p;
    const double d = s - base;
    const double c = a * (s - p);
    const double sl = c * 0.1;
    const int gL = gain_limit;

    if (!isfinite(sl) || fabs(sl) <= DBL_MIN) {

        return static_cast<float>(-0.1 * static_cast<double>(gL));
    }

    const double k_cont = d / sl;
    if (k_cont <= -(static_cast<double>(gL) + 1.0)) {
        return static_cast<float>(-0.1 * static_cast<double>(gL));
    }
    if (k_cont >= (static_cast<double>(gL) + 1.0)) {
        return static_cast<float>(0.1 * static_cast<double>(gL));
    }

    int k0 = __double2int_rd(k_cont);
    int k1 = k0 + 1;
    if (k0 < -gL) k0 = -gL; else if (k0 > gL) k0 = gL;
    if (k1 < -gL) k1 = -gL; else if (k1 > gL) k1 = gL;

    const double e0 = fabs(d - sl * static_cast<double>(k0));
    const double e1 = fabs(d - sl * static_cast<double>(k1));

    const int k_best = (e1 < e0) ? k1 : k0;
    return static_cast<float>(0.1 * static_cast<double>(k_best));
}

__device__ __forceinline__ double choose_best_gain(double alpha, double ema_val, double prev, double src, int gain_limit) {
    const double a = alpha, e = ema_val, p = prev, s = src;
    if (!isfinite(a) || !isfinite(e) || !isfinite(p) || !isfinite(s)) { return 0.0; }

    const double base = a * e + (1.0 - a) * p;
    const double d = s - base;
    const double c = a * (s - p);
    const double sl = c * 0.1;
    const int gL = gain_limit;

    if (!isfinite(sl) || fabs(sl) <= DBL_MIN) {
        return -0.1 * static_cast<double>(gL);
    }

    const double k_cont = d / sl;
    if (k_cont <= -(static_cast<double>(gL) + 1.0)) { return -0.1 * static_cast<double>(gL); }
    if (k_cont >=  (static_cast<double>(gL) + 1.0)) { return  0.1 * static_cast<double>(gL); }

    int k0 = __double2int_rd(k_cont);
    int k1 = k0 + 1;
    if (k0 < -gL) k0 = -gL; else if (k0 > gL) k0 = gL;
    if (k1 < -gL) k1 = -gL; else if (k1 > gL) k1 = gL;

    const double e0 = fabs(d - sl * static_cast<double>(k0));
    const double e1 = fabs(d - sl * static_cast<double>(k1));
    const int k_best = (e1 < e0) ? k1 : k0;
    return 0.1 * static_cast<double>(k_best);
}

}

extern "C" __global__
void ehlers_ecema_batch_f32(const float* __restrict__ prices,
                            const int* __restrict__ lengths,
                            const int* __restrict__ gain_limits,
                            const unsigned char* __restrict__ pine_flags,
                            const unsigned char* __restrict__ confirmed_flags,
                            int series_len,
                            int n_combos,
                            int first_valid,
                            float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }

    const int length = lengths[combo];
    const int gain_limit = gain_limits[combo];
    const bool pine_mode = pine_flags[combo] != 0;
    const bool confirmed = confirmed_flags[combo] != 0;

    const int base = combo * series_len;
    const float nan_f = NAN;

    if (threadIdx.x == 0) {

        if (length <= 0 || gain_limit <= 0) {
            for (int i = 0; i < series_len; ++i) out[base + i] = nan_f;
        }
    }
    __syncthreads();
    if (length <= 0 || gain_limit <= 0) { return; }

    if (threadIdx.x == 0) {
        if (first_valid < 0 || first_valid >= series_len) {
            for (int i = 0; i < series_len; ++i) out[base + i] = nan_f;
        }
    }
    __syncthreads();
    if (first_valid < 0 || first_valid >= series_len) { return; }

    const int valid = series_len - first_valid;
    if (threadIdx.x == 0) {
        if (!pine_mode && valid < length) {
            for (int i = 0; i < series_len; ++i) out[base + i] = nan_f;
        }
    }
    __syncthreads();
    if (!pine_mode && valid < length) { return; }

    const int warm = pine_mode ? first_valid : first_valid + length - 1;
    if (warm >= series_len) {

        for (int idx = threadIdx.x; idx < series_len; idx += blockDim.x) {
            out[base + idx] = nan_f;
        }
        __syncthreads();
        return;
    }


    for (int idx = threadIdx.x; idx < warm; idx += blockDim.x) {
        out[base + idx] = nan_f;
    }
    __syncthreads();
    if (threadIdx.x != 0) { return; }

    const double alpha = compute_alpha(length);
    const double beta  = compute_beta(alpha);

    double ema = 0.0;
    double running_mean = 0.0;
    int mean_count = 0;
    const int warmup_end = pine_mode ? first_valid : ((first_valid + length) < series_len ? (first_valid + length) : series_len);

    bool has_prev = false; double prev_ec = 0.0;

    for (int i = first_valid; i < series_len; ++i) {
        const double price = static_cast<double>(prices[i]);
        double ema_value;

        if (pine_mode) {
            if (isfinite(price)) { ema = alpha * price + beta * ema; }
            ema_value = ema;
        } else {
            if (i == first_valid) {
                running_mean = price; mean_count = 1; ema = price; ema_value = price;
            } else if (i < warmup_end) {
                if (isfinite(price)) {
                    mean_count += 1;
                    running_mean = ((static_cast<double>(mean_count) - 1.0) * running_mean + price) / static_cast<double>(mean_count);
                }
                ema = running_mean; ema_value = running_mean;
            } else {
                if (isfinite(price)) { ema = beta * ema + alpha * price; }
                ema_value = ema;
            }
        }

        if (i < warm) { continue; }

        const double src = pick_src(prices, i, confirmed);
        const double prev = has_prev ? prev_ec : clamp_prev_ec(pine_mode, ema_value);

        const double best_gain = choose_best_gain(alpha, ema_value, prev, src, gain_limit);
        const double ec = alpha * (ema_value + best_gain * (src - prev)) + beta * prev;
        out[base + i] = static_cast<float>(ec);
        prev_ec = ec; has_prev = true;
    }
}


extern "C" __global__
void ehlers_ecema_batch_thread_per_combo_f32(const float* __restrict__ prices,
                                             const int* __restrict__ lengths,
                                             const int* __restrict__ gain_limits,
                                             const unsigned char* __restrict__ pine_flags,
                                             const unsigned char* __restrict__ confirmed_flags,
                                             int series_len,
                                             int n_combos,
                                             int first_valid,
                                             float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) { return; }

    const int length = lengths[combo];
    const int gain_limit = gain_limits[combo];
    const bool pine_mode = pine_flags[combo] != 0;
    const bool confirmed = confirmed_flags[combo] != 0;

    if (series_len <= 0 || length <= 0 || gain_limit <= 0) { return; }
    if (first_valid < 0 || first_valid >= series_len) { return; }
    const int valid = series_len - first_valid;
    if (!pine_mode && valid < length) { return; }

    const int base = combo * series_len;
    const float nan_f = NAN;

    const int warm = pine_mode ? first_valid : first_valid + length - 1;
    if (warm >= series_len) {
        for (int idx = 0; idx < series_len; ++idx) { out[base + idx] = nan_f; }
        return;
    }
    for (int t = 0; t < warm; ++t) { out[base + t] = nan_f; }

    const float alpha = compute_alpha_f(length);
    const float beta  = compute_beta_f(alpha);

    KahanState ema{0.0f, 0.0f};
    KahanState mean{0.0f, 0.0f};
    KahanState ec  {0.0f, 0.0f};
    const int warmup_end = pine_mode ? first_valid : ((first_valid + length) < series_len ? (first_valid + length) : series_len);

    bool has_prev = false;

    for (int i = first_valid; i < series_len; ++i) {
        const float price = prices[i];
        float ema_value;

        if (pine_mode) {
            if (isfinite(price)) { ema_step(ema, alpha, price); }
            ema_value = ema.y;
        } else {
            if (i == first_valid) {
                mean.y = price; mean.c = 0.0f; ema.y = price; ema.c = 0.0f; ema_value = price;
            } else if (i < warmup_end) {
                if (isfinite(price)) {
                    int count = (i - first_valid) + 1;
                    float inv = 1.0f / static_cast<float>(count);
                    float delta = (price - mean.y) * inv;
                    kahan_add(mean, delta);
                }
                ema.y = mean.y; ema.c = 0.0f; ema_value = ema.y;
            } else {
                if (isfinite(price)) { ema_step(ema, alpha, price); }
                ema_value = ema.y;
            }
        }

        if (i < warm) { continue; }

        const float src = pick_src_f(prices, i, confirmed);
        float prev = has_prev ? ec.y : clamp_prev_ec_f(pine_mode, ema_value);
        if (!has_prev) { ec.y = prev; ec.c = 0.0f; has_prev = true; }

        const float best_gain = choose_best_gain_f(alpha, ema_value, prev, src, gain_limit);
        float ec_val = ec_step(ec, alpha, ema_value, src, best_gain);
        out[base + i] = ec_val;
    }
}

extern "C" __global__
void ehlers_ecema_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int num_series,
    int series_len,
    int length,
    int gain_limit,
    unsigned char pine_flag,
    unsigned char confirmed_flag,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    const int series = blockIdx.x;
    if (series >= num_series) {
        return;
    }

    const float nan_f = NAN;

    if (threadIdx.x == 0) {
        if (length <= 0 || gain_limit <= 0) {
            for (int t = 0; t < series_len; ++t) out_tm[t * num_series + series] = nan_f;
        }
    }
    __syncthreads();
    if (length <= 0 || gain_limit <= 0) { return; }

    const int first_valid = first_valids[series];
    if (threadIdx.x == 0) {
        if (first_valid < 0 || first_valid >= series_len) {
            for (int t = 0; t < series_len; ++t) out_tm[t * num_series + series] = nan_f;
        }
    }
    __syncthreads();
    if (first_valid < 0 || first_valid >= series_len) { return; }

    const bool pine_mode = pine_flag != 0;
    const bool confirmed = confirmed_flag != 0;

    const int valid = series_len - first_valid;
    if (threadIdx.x == 0) {
        if (!pine_mode && valid < length) {
            for (int t = 0; t < series_len; ++t) out_tm[t * num_series + series] = nan_f;
        }
    }
    __syncthreads();
    if (!pine_mode && valid < length) { return; }

    const int warm = pine_mode ? first_valid : first_valid + length - 1;
    if (warm >= series_len) {

        for (int t = threadIdx.x; t < series_len; t += blockDim.x)
            out_tm[t * num_series + series] = nan_f;
        __syncthreads();
        return;
    }

    for (int t = threadIdx.x; t < warm; t += blockDim.x)
        out_tm[t * num_series + series] = nan_f;
    __syncthreads();
    if (threadIdx.x != 0) { return; }

    const float alpha = compute_alpha_f(length);
    const float beta  = compute_beta_f(alpha);

    KahanState ema{0.0f, 0.0f};
    KahanState mean{0.0f, 0.0f};
    KahanState ec  {0.0f, 0.0f};
    const int warmup_end = pine_mode ? first_valid : ((first_valid + length) < series_len ? (first_valid + length) : series_len);

    bool has_prev = false;

    for (int i = first_valid; i < series_len; ++i) {
        const float price = prices_tm[i * num_series + series];
        float ema_value;

        if (pine_mode) {
            if (isfinite(price)) { ema_step(ema, alpha, price); }
            ema_value = ema.y;
        } else {
            if (i == first_valid) {
                mean.y = price; mean.c = 0.0f; ema.y = price; ema.c = 0.0f; ema_value = price;
            } else if (i < warmup_end) {
                if (isfinite(price)) {
                    int count = (i - first_valid) + 1;
                    float inv = 1.0f / static_cast<float>(count);
                    float delta = (price - mean.y) * inv;
                    kahan_add(mean, delta);
                }
                ema.y = mean.y; ema.c = 0.0f; ema_value = ema.y;
            } else {
                if (isfinite(price)) { ema_step(ema, alpha, price); }
                ema_value = ema.y;
            }
        }

        if (i < warm) {
            continue;
        }

        const float src = pick_src_tm_f(prices_tm, i, series, num_series, confirmed);
        float prev = has_prev ? ec.y : clamp_prev_ec_f(pine_mode, ema_value);
        if (!has_prev) { ec.y = prev; ec.c = 0.0f; has_prev = true; }

        const float best_gain = choose_best_gain_f(alpha, ema_value, prev, src, gain_limit);
        float ec_val = ec_step(ec, alpha, ema_value, src, best_gain);
        out_tm[i * num_series + series] = ec_val;
    }
}


extern "C" __global__
void ehlers_ecema_many_series_one_param_1d_f32(
    const float* __restrict__ prices_tm,
    int num_series,
    int series_len,
    int length,
    int gain_limit,
    unsigned char pine_flag,
    unsigned char confirmed_flag,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) { return; }

    if (series_len <= 0 || length <= 0 || gain_limit <= 0) { return; }

    const bool pine_mode = pine_flag != 0;
    const bool confirmed = confirmed_flag != 0;
    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) { return; }
    const int valid = series_len - first_valid;
    if (!pine_mode && valid < length) { return; }

    const int stride = num_series;
    const float nan_f = NAN;

    const int warm = pine_mode ? first_valid : first_valid + length - 1;
    if (warm >= series_len) {
        for (int t = 0; t < series_len; ++t) { out_tm[t * stride + series] = nan_f; }
        return;
    }
    for (int t = 0; t < warm; ++t) { out_tm[t * stride + series] = nan_f; }

    const float alpha = compute_alpha_f(length);
    const float beta  = compute_beta_f(alpha);

    KahanState ema{0.0f, 0.0f};
    KahanState mean{0.0f, 0.0f};
    KahanState ec  {0.0f, 0.0f};
    const int warmup_end = pine_mode ? first_valid : ((first_valid + length) < series_len ? (first_valid + length) : series_len);
    bool has_prev = false;
    for (int i = first_valid; i < series_len; ++i) {
        const float price = prices_tm[i * stride + series];
        float ema_value;
        if (pine_mode) {
            if (isfinite(price)) { ema_step(ema, alpha, price); }
            ema_value = ema.y;
        } else {
            if (i == first_valid) { mean.y = price; mean.c = 0.0f; ema.y = price; ema.c = 0.0f; ema_value = price; }
            else if (i < warmup_end) {
                if (isfinite(price)) { int count = (i - first_valid) + 1; float inv = 1.0f / static_cast<float>(count); float delta = (price - mean.y) * inv; kahan_add(mean, delta); }
                ema.y = mean.y; ema.c = 0.0f; ema_value = mean.y;
            } else {
                if (isfinite(price)) { ema_step(ema, alpha, price); }
                ema_value = ema.y;
            }
        }

        if (i < warm) { continue; }
        const int idx_tm = i * stride + series;
        const int src_row = (confirmed && i > 0) ? (i - 1) : i;
        const float src = prices_tm[src_row * stride + series];
        float prev = has_prev ? ec.y : clamp_prev_ec_f(pine_mode, ema_value);
        if (!has_prev) { ec.y = prev; ec.c = 0.0f; has_prev = true; }

        const float best_gain = choose_best_gain_f(alpha, ema_value, prev, src, gain_limit);
        float ec_val = ec_step(ec, alpha, ema_value, src, best_gain);
        out_tm[idx_tm] = ec_val;
    }
}


extern "C" __global__
void ehlers_ecema_many_series_one_param_2d_f32(
    const float* __restrict__ prices_tm,
    int num_series,
    int series_len,
    int length,
    int gain_limit,
    unsigned char pine_flag,
    unsigned char confirmed_flag,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {

    const int tx = blockDim.x;
    const int ty = blockDim.y;
    const int series_per_grid_row = gridDim.x * tx;
    const int local_series = threadIdx.y * tx + threadIdx.x;
    const int series = blockIdx.y * series_per_grid_row + blockIdx.x * tx + local_series;
    if (series >= num_series) { return; }

    if (series_len <= 0 || length <= 0 || gain_limit <= 0) { return; }

    const bool pine_mode = pine_flag != 0;
    const bool confirmed = confirmed_flag != 0;
    const int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) { return; }
    const int valid = series_len - first_valid;
    if (!pine_mode && valid < length) { return; }

    const int stride = num_series;
    const float nan_f = NAN;

    const int warm = pine_mode ? first_valid : first_valid + length - 1;
    if (warm >= series_len) {
        for (int t = 0; t < series_len; ++t) { out_tm[t * stride + series] = nan_f; }
        return;
    }
    for (int t = 0; t < warm; ++t) { out_tm[t * stride + series] = nan_f; }

    const float alpha = compute_alpha_f(length);
    const float beta  = compute_beta_f(alpha);

    KahanState ema{0.0f, 0.0f}; KahanState mean{0.0f, 0.0f};
    KahanState ec{0.0f, 0.0f};
    const int warmup_end = pine_mode ? first_valid : ((first_valid + length) < series_len ? (first_valid + length) : series_len);
    bool has_prev = false;

    for (int i = first_valid; i < series_len; ++i) {
        const float price = prices_tm[i * stride + series];
        float ema_value;
        if (pine_mode) {
            if (isfinite(price)) { ema_step(ema, alpha, price); }
            ema_value = ema.y;
        } else {
            if (i == first_valid) { mean.y = price; mean.c = 0.0f; ema.y = price; ema.c = 0.0f; ema_value = price; }
            else if (i < warmup_end) {
                if (isfinite(price)) { int count = (i - first_valid) + 1; float inv = 1.0f / static_cast<float>(count); float delta = (price - mean.y) * inv; kahan_add(mean, delta); }
                ema.y = mean.y; ema.c = 0.0f; ema_value = mean.y;
            } else {
                if (isfinite(price)) { ema_step(ema, alpha, price); }
                ema_value = ema.y;
            }
        }
        if (i < warm) { continue; }
        const int idx_tm = i * stride + series;
        const int src_row = (confirmed && i > 0) ? (i - 1) : i;
        const float src = prices_tm[src_row * stride + series];
        float prev = has_prev ? ec.y : clamp_prev_ec_f(pine_mode, ema_value);
        if (!has_prev) { ec.y = prev; ec.c = 0.0f; has_prev = true; }

        const float best_gain = choose_best_gain_f(alpha, ema_value, prev, src, gain_limit);
        float ec_val = ec_step(ec, alpha, ema_value, src, best_gain);
        out_tm[idx_tm] = ec_val;
    }
}


// ===========================================================================
// S3 f64 LANE — ehlers_ecema (Error-Correcting EMA)
// ===========================================================================
// Reference: src/indicators/moving_averages/ehlers_ecema.rs
//   ehlers_ecema_prepare (:419)                        — first_valid, Err, alpha/beta
//   ehlers_ecema_scalar_direct_into_with_mode (:823)   — the arithmetic
// Defaults: length 20 (the SWEPT parameter, :246), gain_limit 50 (:251),
// confirmed_only FALSE (:208). warmup_end = first + length - 1.
//
// THE SEED IS A RUNNING MEAN OVER VALID BARS, NOT AN SMA (:844-853):
//     ema = ((vc - 1.0) * ema + x) / vc
// with vc counting only FINITE samples. That is a different number from
// sum/length whenever any bar in the seed window is non-finite, and it is a
// different ROUNDING even when none is — divide-per-step, not divide-once.
//
// THE GAIN SEARCH IS AN INTEGER MINIMISATION, AND IT IS EXACT.
// The CPU does not scan all 2*gain_limit+1 candidates: it computes the
// continuous optimum k_cont = d/s, clamps, and compares only floor(k_cont) and
// floor(k_cont)+1 by |d - s*k| — with `e0 <= e1` breaking ties toward the LOWER
// k (:906). Reproduced exactly, including that tie direction; reversing it
// changes the output whenever d sits exactly between two lattice points.
//
// THE DEGENERATE GUARD IS f64::MIN_POSITIVE, NOT AN EPSILON (:879):
//     if !s.is_finite() || !d.is_finite() || s.abs() <= f64::MIN_POSITIVE { -gL }
// f64::MIN_POSITIVE is 2.2250738585072014e-308 — the smallest NORMAL double,
// not the smallest representable and not a tolerance. An f32 port cannot hold
// it: FLT_MIN is 1.1754944e-38, 270 orders of magnitude larger, so the f32 lane
// takes the -gL branch on subnormal-but-meaningful steps the reference handles.
// DBL_MIN below is that exact constant.
//
// ROUNDING.
//   ema     = beta.mul_add(ema, alpha * x)        — product then ONE fma
//   base    = alpha.mul_add(ema, beta * prev_ec)  — same
//   prev_ec = (k as f64).mul_add(s, base)         — ONE fma
//
// NON-FINITE BARS FREEZE THE EMA rather than poisoning it (:862-864), the same
// polarity as wto. Transcribed.
//
// One thread per column.
// ===========================================================================

#define NEO_S3_ECEMA_GAIN_LIMIT     50
#define NEO_S3_ECEMA_CONFIRMED_ONLY 0

__device__ __forceinline__ double neo_s3_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void neoethos_ehlers_ecema_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    double* __restrict__ row = out + (size_t)r * (size_t)n;
    const int length = periods[r];
    const int gain_limit = NEO_S3_ECEMA_GAIN_LIMIT;
    const bool confirmed_only = (NEO_S3_ECEMA_CONFIRMED_ONLY != 0);

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        (length == 0) || (length > n) ||
        ((n - first_valid) < length) ||
        (gain_limit == 0);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s3_qnan();
        return;
    }

    const int start_idx = first_valid + length - 1;   // == warmup_end
    for (int i = 0; i < start_idx && i < n; ++i) row[i] = neo_s3_qnan();
    if (start_idx >= n) return;

    const double alpha = 2.0 / ((double)length + 1.0);
    const double beta = 1.0 - alpha;

    double ema = data[first_valid];
    int valid_count = 1;
    for (int i = first_valid + 1; i <= start_idx; ++i) {
        const double x = data[i];
        if (isfinite(x)) {
            valid_count += 1;
            const double vc = (double)valid_count;
            ema = ((vc - 1.0) * ema + x) / vc;
        }
    }

    const int gL = gain_limit;
    const double step_s = 0.1;
    double prev_ec = ema;

    for (int i = start_idx; i < n; ++i) {
        if (i > start_idx) {
            const double x = data[i];
            if (isfinite(x)) ema = fma(beta, ema, alpha * x);
        }

        const double src = (confirmed_only && i > 0) ? data[i - 1] : data[i];

        const double delta = src - prev_ec;
        const double c = alpha * delta;
        const double base = fma(alpha, ema, beta * prev_ec);
        const double d = src - base;
        const double s = c * step_s;

        int k_best;
        // 2.2250738585072014e-308 == f64::MIN_POSITIVE (DBL_MIN)
        if (!isfinite(s) || !isfinite(d) || fabs(s) <= 2.2250738585072014e-308) {
            k_best = -gL;
        } else {
            const double k_cont = d / s;
            if (k_cont <= (-(double)gL - 1.0)) {
                k_best = -gL;
            } else if (k_cont >= ((double)gL + 1.0)) {
                k_best = gL;
            } else {
                int k0 = (int)floor(k_cont);
                int k1 = k0 + 1;
                if (k0 < -gL) k0 = -gL; else if (k0 > gL) k0 = gL;
                if (k1 < -gL) k1 = -gL; else if (k1 > gL) k1 = gL;

                const double e0 = fabs(d - s * (double)k0);
                const double e1 = fabs(d - s * (double)k1);
                k_best = (e0 <= e1) ? k0 : k1;   // ties go to the LOWER k
            }
        }

        prev_ec = fma((double)k_best, s, base);
        row[i] = prev_ec;
    }
}
