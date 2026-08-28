#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

static __device__ __forceinline__ float sum_read(float s, float c) { return s + c; }

static __device__ __forceinline__ float wsum_norm_i32(int p) {
    long long t = (long long)p * (p + 1);

    return __int2float_rn((int)(t >> 1));
}


static __device__ __forceinline__ void kahan_add(float value, float& sum, float& comp) {
    float y = value - comp;
    float t = sum + y;
    comp = (t - sum) - y;
    sum = t;
}


static __device__ __forceinline__
float dma_quantized_best_gain(float x,
                              float e0_prev,
                              float ec_prev,
                              float alpha_e,
                              int   ema_gain_limit) {


    const float one_minus_alpha_e = 1.0f - alpha_e;
    const float base  = fmaf(alpha_e, e0_prev, one_minus_alpha_e * ec_prev);
    const float t     = alpha_e * (x - ec_prev);
    const float r     = x - base;

    const float EPS = 1e-20f;
    if (fabsf(t) <= EPS) return 0.0f;

    const float step = 0.1f;
    const int   limit = ema_gain_limit;
    float target = (r / t) / step;

    int i0 = (int)floorf(target);
    if (i0 < 0) i0 = 0; else if (i0 > limit) i0 = limit;
    int i1 = (i0 < limit) ? (i0 + 1) : i0;

    const float g0 = i0 * step;
    const float g1 = i1 * step;
    const float e0 = fabsf(r - t * g0);
    const float e1 = fabsf(r - t * g1);
    return (e0 <= e1) ? g0 : g1;
}


static __device__ __forceinline__
float dma_update_ec(float x,
                    float e0_prev,
                    float ec_prev,
                    float alpha_e,
                    int   ema_gain_limit) {
    const float g = dma_quantized_best_gain(x, e0_prev, ec_prev, alpha_e, ema_gain_limit);
    const float target = fmaf(g, x - ec_prev, e0_prev);
    return fmaf(alpha_e, target - ec_prev, ec_prev);
}

extern "C" __global__
void dma_batch_f32(const float* __restrict__ prices,
                   const int* __restrict__ hull_lengths,
                   const int* __restrict__ ema_lengths,
                   const int* __restrict__ ema_gain_limits,
                   const int* __restrict__ hull_types,
                   int series_len,
                   int n_combos,
                   int first_valid,
                   float* __restrict__ out) {
    const int combo = blockIdx.x;
    if (combo >= n_combos) {
        return;
    }


    if (threadIdx.x != 0) {
        return;
    }

    const int hull_length = hull_lengths[combo];
    const int ema_length = ema_lengths[combo];
    const int ema_gain_limit = ema_gain_limits[combo];
    const int hull_type = hull_types[combo];

    const int half = hull_length / 2;
    const int sqrt_len = static_cast<int>(floorf(sqrtf(static_cast<float>(hull_length)) + 0.5f));
    const float denom_half_f = (half        > 0 ? wsum_norm_i32(half)        : 1.0f);
    const float denom_full_f = (hull_length > 0 ? wsum_norm_i32(hull_length) : 1.0f);
    const float denom_sqrt_f = (sqrt_len    > 0 ? wsum_norm_i32(sqrt_len)    : 1.0f);
    const float inv_w_half   = 1.0f / denom_half_f;
    const float inv_w_full   = 1.0f / denom_full_f;
    const float inv_w_sqrt   = 1.0f / denom_sqrt_f;

    const int base_out = combo * series_len;
    if (series_len <= 0 || hull_length <= 0 || ema_length <= 0 || first_valid >= series_len) {
        for (int i = 0; i < series_len; ++i) {
            out[base_out + i] = NAN;
        }
        return;
    }

    for (int i = 0; i < first_valid; ++i) {
        out[base_out + i] = NAN;
    }


    extern __shared__ __align__(16) float shared[];
    float* diff_ring = shared;

    const float alpha_e = 2.0f / (static_cast<float>(ema_length) + 1.0f);
    const int i0_e = first_valid + (ema_length > 0 ? ema_length - 1 : 0);

    float e0_prev = 0.0f;
    bool e0_init_done = false;
    float ec_prev = 0.0f;
    bool ec_init_done = false;

    const int i0_half = first_valid + (half > 0 ? half - 1 : 0);
    const int i0_full = first_valid + (hull_length > 0 ? hull_length - 1 : 0);

    float a_half = 0.0f;
    float a_half_c = 0.0f;
    float s_half = 0.0f;
    bool half_ready = false;

    float a_full = 0.0f;
    float a_full_c = 0.0f;
    float s_full = 0.0f;
    bool full_ready = false;

    int diff_filled = 0;
    int diff_pos = 0;
    float diff_sum_seed = 0.0f;

    float a_diff = 0.0f;
    float s_diff = 0.0f;
    float a_diff_c = 0.0f;
    float s_diff_c = 0.0f;

    float diff_ema = 0.0f;
    bool diff_ema_init_done = false;
    const float alpha_sqrt = (sqrt_len > 0)
        ? 2.0f / (static_cast<float>(sqrt_len) + 1.0f)
        : 0.0f;

    float e_half_prev = 0.0f;
    float e_full_prev = 0.0f;
    bool e_half_init_done = false;
    bool e_full_init_done = false;
    const float alpha_half = (half > 0)
        ? 2.0f / (static_cast<float>(half) + 1.0f)
        : 0.0f;
    const float alpha_full = (hull_length > 0)
        ? 2.0f / (static_cast<float>(hull_length) + 1.0f)
        : 0.0f;

    const bool is_wma = (hull_type == 0);
    float hull_val = NAN;

    for (int i = first_valid; i < series_len; ++i) {
        const float x = prices[i];

        if (!e0_init_done) {
            if (i >= i0_e) {
                int start = i + 1 - ema_length;
                float sum = 0.0f;
                for (int k = start; k <= i; ++k) {
                    sum += prices[k];
                }
                e0_prev = sum / static_cast<float>(ema_length);
                e0_init_done = true;
            }
        } else {
            e0_prev = fmaf(alpha_e, x - e0_prev, e0_prev);
        }

        float diff_now = NAN;

        if (is_wma) {
            if (half > 0) {
                if (!half_ready) {
                    if (i >= i0_half) {
                        int start = i + 1 - half;
                        float sum = 0.0f;
                        float wsum_local = 0.0f;
                        for (int j = 0; j < half; ++j) {
                            const int idx = start + j;
                            const float w = static_cast<float>(j + 1);
                            const float v = prices[idx];
                            sum += v;
                            wsum_local = fmaf(w, v, wsum_local);
                        }
                        a_half = sum;
                        a_half_c = 0.0f;
                        s_half = wsum_local;
                        half_ready = true;
                    }
                } else {
                    const float a_prev = sum_read(a_half, a_half_c);
                    const float old = prices[i - half];
                    kahan_add(x - old, a_half, a_half_c);

                    s_half += fmaf(static_cast<float>(half), x, -a_prev);
                }
            }

            if (hull_length > 0) {
                if (!full_ready) {
                    if (i >= i0_full) {
                        int start = i + 1 - hull_length;
                        float sum = 0.0f;
                        float wsum_local = 0.0f;
                        for (int j = 0; j < hull_length; ++j) {
                            const int idx = start + j;
                            const float w = static_cast<float>(j + 1);
                            const float v = prices[idx];
                            sum += v;
                            wsum_local = fmaf(w, v, wsum_local);
                        }
                        a_full = sum;
                        a_full_c = 0.0f;
                        s_full = wsum_local;
                        full_ready = true;
                    }
                } else {
                    const float a_prev = sum_read(a_full, a_full_c);
                    const float old = prices[i - hull_length];
                    kahan_add(x - old, a_full, a_full_c);

                    s_full += fmaf(static_cast<float>(hull_length), x, -a_prev);
                }
            }

            if (half_ready && full_ready) {
                const float w_half = s_half * inv_w_half;
                const float w_full = s_full * inv_w_full;
                diff_now = 2.0f * w_half - w_full;
            }
        } else {
            if (half > 0) {
                if (!e_half_init_done) {
                    if (i >= i0_half) {
                        int start = i + 1 - half;
                        float sum = 0.0f;
                        for (int k = start; k <= i; ++k) {
                            sum += prices[k];
                        }
                        e_half_prev = sum / static_cast<float>(half);
                        e_half_init_done = true;
                    }
                } else {
                    e_half_prev = fmaf(alpha_half, x - e_half_prev, e_half_prev);
                }
            }

            if (hull_length > 0) {
                if (!e_full_init_done) {
                    if (i >= i0_full) {
                        int start = i + 1 - hull_length;
                        float sum = 0.0f;
                        for (int k = start; k <= i; ++k) {
                            sum += prices[k];
                        }
                        e_full_prev = sum / static_cast<float>(hull_length);
                        e_full_init_done = true;
                    }
                } else {
                    e_full_prev = fmaf(alpha_full, x - e_full_prev, e_full_prev);
                }
            }

            if (e_half_init_done && e_full_init_done) {
                diff_now = 2.0f * e_half_prev - e_full_prev;
            }
        }

        if (!isnan(diff_now) && sqrt_len > 0) {
            if (diff_filled < sqrt_len) {
                diff_ring[diff_filled] = diff_now;
                diff_sum_seed += diff_now;
                diff_filled += 1;

                if (diff_filled == sqrt_len) {
                    if (is_wma) {
                        a_diff = 0.0f;
                        s_diff = 0.0f;
                        a_diff_c = 0.0f;
                        s_diff_c = 0.0f;
                        for (int j = 0; j < sqrt_len; ++j) {
                            const float w = static_cast<float>(j + 1);
                            const float v = diff_ring[j];
                            kahan_add(v, a_diff, a_diff_c);
                            kahan_add(w * v, s_diff, s_diff_c);
                        }
                        hull_val = sum_read(s_diff, s_diff_c) * inv_w_sqrt;
                    } else {
                        diff_ema = diff_sum_seed / static_cast<float>(sqrt_len);
                        diff_ema_init_done = true;
                        hull_val = diff_ema;
                    }
                }
            } else {
                const float old = diff_ring[diff_pos];
                diff_ring[diff_pos] = diff_now;
                diff_pos += 1; if (diff_pos == sqrt_len) diff_pos = 0;

                if (is_wma) {
                    const float a_prev = sum_read(a_diff, a_diff_c);
                    kahan_add(diff_now - old, a_diff, a_diff_c);

                    kahan_add(fmaf(static_cast<float>(sqrt_len), diff_now, -a_prev), s_diff, s_diff_c);
                    hull_val = sum_read(s_diff, s_diff_c) * inv_w_sqrt;
                } else {
                    if (!diff_ema_init_done) {
                        diff_ema = diff_now;
                        diff_ema_init_done = true;
                    } else {
                        diff_ema = fmaf(alpha_sqrt, diff_now - diff_ema, diff_ema);
                    }
                    hull_val = diff_ema;
                }
            }
        }

        float ec_now = NAN;
        if (e0_init_done) {
            if (!ec_init_done) {
                ec_prev = e0_prev;
                ec_now = ec_prev;
                ec_init_done = true;
            } else {
                ec_now = dma_update_ec(x, e0_prev, ec_prev, alpha_e, ema_gain_limit);
                ec_prev = ec_now;
            }
        }

        float out_val = NAN;
        if (ec_init_done && diff_filled == sqrt_len) {
            out_val = 0.5f * (hull_val + ec_prev);
        }
        out[base_out + i] = out_val;
    }
}


template<int TX>
__device__ void dma_batch_tiled_f32_tx_core(const float* __restrict__ prices,
                                       const int* __restrict__ hull_lengths,
                                       const int* __restrict__ ema_lengths,
                                       const int* __restrict__ ema_gain_limits,
                                       const int* __restrict__ hull_types,
                                       int series_len,
                                       int n_combos,
                                       int first_valid,
                                       int combo_start,
                                       int sqrt_stride,
                                       float* __restrict__ out) {
    const int local = threadIdx.x;
    const int global_idx = combo_start + blockIdx.x * TX + local;
    if (global_idx >= n_combos) { return; }

    extern __shared__ __align__(16) float smem[];
    float* diff_ring = smem + local * sqrt_stride;

    const int hull_length = hull_lengths[global_idx];
    const int ema_length = ema_lengths[global_idx];
    const int ema_gain_limit = ema_gain_limits[global_idx];
    const int hull_type = hull_types[global_idx];

    if (series_len <= 0 || hull_length <= 0 || ema_length <= 0) {
        return;
    }

    const int half = hull_length / 2;
    const int sqrt_len = max(1, (int)floorf(sqrtf((float)hull_length) + 0.5f));
    const float denom_half_f = (half        > 0 ? wsum_norm_i32(half)        : 1.0f);
    const float denom_full_f = (hull_length > 0 ? wsum_norm_i32(hull_length) : 1.0f);
    const float denom_sqrt_f = (sqrt_len    > 0 ? wsum_norm_i32(sqrt_len)    : 1.0f);
    const float inv_w_half   = 1.0f / denom_half_f;
    const float inv_w_full   = 1.0f / denom_full_f;
    const float inv_w_sqrt   = 1.0f / denom_sqrt_f;

    const int base_out = global_idx * series_len;
    if (series_len <= 0 || hull_length <= 0 || ema_length <= 0 || first_valid >= series_len) {
        for (int i = 0; i < series_len; ++i) {
            out[base_out + i] = NAN;
        }
        return;
    }

    for (int i = 0; i < first_valid; ++i) {
        out[base_out + i] = NAN;
    }

    const float alpha_e = 2.0f / (float(ema_length) + 1.0f);
    const int i0_e = first_valid + (ema_length > 0 ? ema_length - 1 : 0);

    float e0_prev = 0.0f; bool e0_init_done = false;
    float ec_prev = 0.0f; bool ec_init_done = false;

    const int i0_half = first_valid + (half > 0 ? half - 1 : 0);
    const int i0_full = first_valid + (hull_length > 0 ? hull_length - 1 : 0);

    float a_half = 0.0f, a_half_c = 0.0f, s_half = 0.0f; bool half_ready = false;
    float a_full = 0.0f, a_full_c = 0.0f, s_full = 0.0f; bool full_ready = false;

    int diff_filled = 0, diff_pos = 0;
    float diff_sum_seed = 0.0f;
    float a_diff = 0.0f, s_diff = 0.0f;
    float a_diff_c = 0.0f, s_diff_c = 0.0f;
    float diff_ema = 0.0f; bool diff_ema_init_done = false;
    const float alpha_sqrt = (sqrt_len > 0) ? 2.0f / (float(sqrt_len) + 1.0f) : 0.0f;

    float e_half_prev = 0.0f, e_full_prev = 0.0f; bool e_half_init_done = false, e_full_init_done = false;
    const float alpha_half = (half > 0) ? 2.0f / (float(half) + 1.0f) : 0.0f;
    const float alpha_full = (hull_length > 0) ? 2.0f / (float(hull_length) + 1.0f) : 0.0f;

    const bool is_wma = (hull_type == 0);
    float hull_val = NAN;

    for (int i = first_valid; i < series_len; ++i) {
        const float x = prices[i];

        if (!e0_init_done) {
            if (i >= i0_e) {
                int start = i + 1 - ema_length;
                float sum = 0.0f;
                for (int k = start; k <= i; ++k) { sum += prices[k]; }
                e0_prev = sum / float(ema_length);
                e0_init_done = true;
            }
        } else {
            e0_prev = fmaf(alpha_e, x - e0_prev, e0_prev);
        }

        float diff_now = NAN;
        if (is_wma) {
            if (half > 0) {
                if (!half_ready) {
                    if (i >= i0_half) {
                        int start = i + 1 - half;
                        float sum = 0.0f, wsum = 0.0f;
                        for (int j = 0; j < half; ++j) {
                            const int idx = start + j; const float w = float(j + 1); const float v = prices[idx];
                            sum += v; wsum = fmaf(w, v, wsum);
                        }
                        a_half = sum; a_half_c = 0.0f; s_half = wsum; half_ready = true;
                    }
                } else {
                    const float a_prev = sum_read(a_half, a_half_c); const float old = prices[i - half];
                    kahan_add(x - old, a_half, a_half_c);
                    s_half += fmaf(float(half), x, -a_prev);
                }
            }
            if (hull_length > 0) {
                if (!full_ready) {
                    if (i >= i0_full) {
                        int start = i + 1 - hull_length;
                        float sum = 0.0f, wsum = 0.0f;
                        for (int j = 0; j < hull_length; ++j) {
                            const int idx = start + j; const float w = float(j + 1); const float v = prices[idx];
                            sum += v; wsum = fmaf(w, v, wsum);
                        }
                        a_full = sum; a_full_c = 0.0f; s_full = wsum; full_ready = true;
                    }
                } else {
                    const float a_prev = sum_read(a_full, a_full_c); const float old = prices[i - hull_length];
                    kahan_add(x - old, a_full, a_full_c);
                    s_full += fmaf(float(hull_length), x, -a_prev);
                }
            }
            if (half_ready && full_ready) {
                const float w_half = s_half * inv_w_half;
                const float w_full = s_full * inv_w_full;
                diff_now = 2.0f * w_half - w_full;
            }
        } else {
            if (half > 0) {
                if (!e_half_init_done) {
                    if (i >= i0_half) {
                        int start = i + 1 - half; float sum = 0.0f;
                        for (int k = start; k <= i; ++k) { sum += prices[k]; }
                        e_half_prev = sum / float(half); e_half_init_done = true;
                    }
                } else { e_half_prev = fmaf(alpha_half, x - e_half_prev, e_half_prev); }
            }
            if (hull_length > 0) {
                if (!e_full_init_done) {
                    if (i >= i0_full) {
                        int start = i + 1 - hull_length; float sum = 0.0f;
                        for (int k = start; k <= i; ++k) { sum += prices[k]; }
                        e_full_prev = sum / float(hull_length); e_full_init_done = true;
                    }
                } else { e_full_prev = fmaf(alpha_full, x - e_full_prev, e_full_prev); }
            }
            if (e_half_init_done && e_full_init_done) { diff_now = 2.0f * e_half_prev - e_full_prev; }
        }

        if (!isnan(diff_now) && sqrt_len > 0) {
            if (diff_filled < sqrt_len) {
                diff_ring[diff_filled] = diff_now; diff_sum_seed += diff_now;
                diff_filled += 1;
                if (diff_filled == sqrt_len) {
                    if (is_wma) {
                        a_diff = 0.0f; s_diff = 0.0f;
                        a_diff_c = 0.0f; s_diff_c = 0.0f;
                        for (int j = 0; j < sqrt_len; ++j) {
                            const float w = float(j + 1); const float v = diff_ring[j];
                            kahan_add(v, a_diff, a_diff_c); kahan_add(w * v, s_diff, s_diff_c);
                        }
                        hull_val = sum_read(s_diff, s_diff_c) * inv_w_sqrt;
                    } else {
                        diff_ema = diff_sum_seed / float(sqrt_len); diff_ema_init_done = true; hull_val = diff_ema;
                    }
                }
            } else {
                const float old = diff_ring[diff_pos]; diff_ring[diff_pos] = diff_now; diff_pos += 1; if (diff_pos == sqrt_len) diff_pos = 0;
                if (is_wma) {
                    const float a_prev = sum_read(a_diff, a_diff_c);
                    kahan_add(diff_now - old, a_diff, a_diff_c);
                    kahan_add(fmaf(float(sqrt_len), diff_now, -a_prev), s_diff, s_diff_c);
                    hull_val = sum_read(s_diff, s_diff_c) * inv_w_sqrt;
                } else {
                    if (!diff_ema_init_done) { diff_ema = diff_now; diff_ema_init_done = true; }
                    else { diff_ema = fmaf(alpha_sqrt, diff_now - diff_ema, diff_ema); }
                    hull_val = diff_ema;
                }
            }
        }

        float ec_now = NAN;
        if (e0_init_done) {
            if (!ec_init_done) { ec_prev = e0_prev; ec_now = ec_prev; ec_init_done = true; }
            else { ec_now = dma_update_ec(x, e0_prev, ec_prev, alpha_e, ema_gain_limit); ec_prev = ec_now; }
        }

        float out_val = NAN;
        if (ec_init_done && diff_filled == sqrt_len) {
            out_val = 0.5f * (hull_val + ec_prev);
        }
        out[base_out + i] = out_val;
    }
}

extern "C" {
__global__ void dma_batch_tiled_f32_tx32(
    const float* __restrict__ prices,
    const int* __restrict__ hull_lengths,
    const int* __restrict__ ema_lengths,
    const int* __restrict__ ema_gain_limits,
    const int* __restrict__ hull_types,
    int series_len,
    int n_combos,
    int first_valid,
    int combo_start,
    int sqrt_stride,
    float* __restrict__ out) {
    dma_batch_tiled_f32_tx_core<32>(prices, hull_lengths, ema_lengths, ema_gain_limits, hull_types,
                                    series_len, n_combos, first_valid, combo_start, sqrt_stride, out);
}
__global__ void dma_batch_tiled_f32_tx64(
    const float* __restrict__ prices,
    const int* __restrict__ hull_lengths,
    const int* __restrict__ ema_lengths,
    const int* __restrict__ ema_gain_limits,
    const int* __restrict__ hull_types,
    int series_len,
    int n_combos,
    int first_valid,
    int combo_start,
    int sqrt_stride,
    float* __restrict__ out) {
    dma_batch_tiled_f32_tx_core<64>(prices, hull_lengths, ema_lengths, ema_gain_limits, hull_types,
                                    series_len, n_combos, first_valid, combo_start, sqrt_stride, out);
}
__global__ void dma_batch_tiled_f32_tx128(
    const float* __restrict__ prices,
    const int* __restrict__ hull_lengths,
    const int* __restrict__ ema_lengths,
    const int* __restrict__ ema_gain_limits,
    const int* __restrict__ hull_types,
    int series_len,
    int n_combos,
    int first_valid,
    int combo_start,
    int sqrt_stride,
    float* __restrict__ out) {
    dma_batch_tiled_f32_tx_core<128>(prices, hull_lengths, ema_lengths, ema_gain_limits, hull_types,
                                     series_len, n_combos, first_valid, combo_start, sqrt_stride, out);
}
}


extern "C" __global__
void dma_many_series_one_param_f32(const float* __restrict__ prices_tm,
                                   int hull_length,
                                   int ema_length,
                                   int ema_gain_limit,
                                   int hull_type,
                                   int series_len,
                                   int num_series,
                                   const int* __restrict__ first_valids,
                                   int sqrt_len,
                                   float* __restrict__ out_tm) {
    const int series_idx = blockIdx.x;
    if (series_idx >= num_series) {
        return;
    }
    if (series_len <= 0 || hull_length <= 0 || ema_length <= 0) {
        return;
    }

    const int stride = num_series;
    const int base_out = series_idx;

    if (threadIdx.x == 0) {
        for (int i = 0; i < series_len; ++i) {
            out_tm[base_out + i * stride] = NAN;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    const int first_valid = first_valids[series_idx];
    if (first_valid >= series_len) {
        return;
    }

    extern __shared__ __align__(16) float diff_ring[];

    const int half = hull_length / 2;
    const int sqrt_len_clamped = (sqrt_len > 0) ? sqrt_len : 1;


    const float denom_half_f = (half        > 0 ? wsum_norm_i32(half)        : 1.0f);
    const float denom_full_f = (hull_length > 0 ? wsum_norm_i32(hull_length) : 1.0f);
    const float denom_sqrt_f = (sqrt_len_clamped > 0 ? wsum_norm_i32(sqrt_len_clamped) : 1.0f);
    const float inv_w_half   = 1.0f / denom_half_f;
    const float inv_w_full   = 1.0f / denom_full_f;
    const float inv_w_sqrt   = 1.0f / denom_sqrt_f;

    const float alpha_e = 2.0f / (static_cast<float>(ema_length) + 1.0f);
    const int i0_e = first_valid + (ema_length > 0 ? ema_length - 1 : 0);

    float e0_prev = 0.0f;
    bool e0_init_done = false;
    float ec_prev = 0.0f;
    bool ec_init_done = false;

    const int i0_half = first_valid + (half > 0 ? half - 1 : 0);
    const int i0_full = first_valid + (hull_length > 0 ? hull_length - 1 : 0);

    float a_half = 0.0f;
    float s_half = 0.0f;
    float a_half_c = 0.0f;
    float s_half_c = 0.0f;
    bool half_ready = false;

    float a_full = 0.0f;
    float s_full = 0.0f;
    float a_full_c = 0.0f;
    float s_full_c = 0.0f;
    bool full_ready = false;

    int diff_filled = 0;
    int diff_pos = 0;
    float diff_sum_seed = 0.0f;
    float diff_sum_seed_c = 0.0f;

    float a_diff = 0.0f;
    float s_diff = 0.0f;
    float a_diff_c = 0.0f;
    float s_diff_c = 0.0f;
    bool diff_wma_init_done = false;

    float diff_ema = 0.0f;
    bool diff_ema_init_done = false;
    const float alpha_sqrt = (sqrt_len_clamped > 0)
        ? 2.0f / (static_cast<float>(sqrt_len_clamped) + 1.0f)
        : 0.0f;

    float e_half_prev = 0.0f;
    float e_full_prev = 0.0f;
    bool e_half_init_done = false;
    bool e_full_init_done = false;
    const float alpha_half = (half > 0)
        ? 2.0f / (static_cast<float>(half) + 1.0f)
        : 0.0f;
    const float alpha_full = (hull_length > 0)
        ? 2.0f / (static_cast<float>(hull_length) + 1.0f)
        : 0.0f;

    const bool is_wma = (hull_type == 0);
    float hull_val = NAN;

    for (int i = first_valid; i < series_len; ++i) {
        const int idx = i * stride + series_idx;
        const float x = prices_tm[idx];

        if (!e0_init_done) {
            if (i >= i0_e) {
                int start = i + 1 - ema_length;
                float sum = 0.0f;
                float sum_c = 0.0f;
                for (int k = start; k <= i; ++k) {
                    kahan_add(prices_tm[k * stride + series_idx], sum, sum_c);
                }
                e0_prev = sum_read(sum, sum_c) / static_cast<float>(ema_length);
                e0_init_done = true;
            }
        } else {
            e0_prev = fmaf(alpha_e, x - e0_prev, e0_prev);
        }

        float diff_now = NAN;

        if (is_wma) {
            if (half > 0) {
                if (!half_ready) {
                    if (i >= i0_half) {
                        int start = i + 1 - half;
                        float sum = 0.0f;
                        float sum_c = 0.0f;
                        float wsum_local = 0.0f;
                        float wsum_c = 0.0f;
                        for (int j = 0; j < half; ++j) {
                            const int sidx = start + j;
                            const float w = static_cast<float>(j + 1);
                            const float v = prices_tm[sidx * stride + series_idx];
                            kahan_add(v, sum, sum_c);
                            kahan_add(w * v, wsum_local, wsum_c);
                        }
                        a_half = sum;
                        s_half = wsum_local;
                        a_half_c = sum_c;
                        s_half_c = wsum_c;
                        half_ready = true;
                    }
                } else {
                    const float a_prev = a_half;
                    const float old = prices_tm[(i - half) * stride + series_idx];
                    kahan_add(x - old, a_half, a_half_c);

                    kahan_add(fmaf(static_cast<float>(half), x, -a_prev), s_half, s_half_c);
                }
            }

            if (hull_length > 0) {
                if (!full_ready) {
                    if (i >= i0_full) {
                        int start = i + 1 - hull_length;
                        float sum = 0.0f;
                        float sum_c = 0.0f;
                        float wsum_local = 0.0f;
                        float wsum_c = 0.0f;
                        for (int j = 0; j < hull_length; ++j) {
                            const int sidx = start + j;
                            const float w = static_cast<float>(j + 1);
                            const float v = prices_tm[sidx * stride + series_idx];
                            kahan_add(v, sum, sum_c);
                            kahan_add(w * v, wsum_local, wsum_c);
                        }
                        a_full = sum;
                        s_full = wsum_local;
                        a_full_c = sum_c;
                        s_full_c = wsum_c;
                        full_ready = true;
                    }
                } else {
                    const float a_prev = a_full;
                    const float old = prices_tm[(i - hull_length) * stride + series_idx];
                    kahan_add(x - old, a_full, a_full_c);

                    kahan_add(fmaf(static_cast<float>(hull_length), x, -a_prev), s_full, s_full_c);
                }
            }

            if (half_ready && full_ready) {
                const float w_half = sum_read(s_half, s_half_c) * inv_w_half;
                const float w_full = sum_read(s_full, s_full_c) * inv_w_full;
                diff_now = 2.0f * w_half - w_full;
            }
        } else {
            if (half > 0) {
                if (!e_half_init_done) {
                    if (i >= i0_half) {
                        int start = i + 1 - half;
                        float sum = 0.0f;
                        float sum_c = 0.0f;
                        for (int k = start; k <= i; ++k) {
                            kahan_add(prices_tm[k * stride + series_idx], sum, sum_c);
                        }
                        e_half_prev = sum_read(sum, sum_c) / static_cast<float>(half);
                        e_half_init_done = true;
                    }
                } else {
                    e_half_prev = fmaf(alpha_half, x - e_half_prev, e_half_prev);
                }
            }

            if (hull_length > 0) {
                if (!e_full_init_done) {
                    if (i >= i0_full) {
                        int start = i + 1 - hull_length;
                        float sum = 0.0f;
                        float sum_c = 0.0f;
                        for (int k = start; k <= i; ++k) {
                            kahan_add(prices_tm[k * stride + series_idx], sum, sum_c);
                        }
                        e_full_prev = sum_read(sum, sum_c) / static_cast<float>(hull_length);
                        e_full_init_done = true;
                    }
                } else {
                    e_full_prev = fmaf(alpha_full, x - e_full_prev, e_full_prev);
                }
            }

            if (e_half_init_done && e_full_init_done) {
                diff_now = 2.0f * e_half_prev - e_full_prev;
            }
        }

        if (!isnan(diff_now) && sqrt_len_clamped > 0) {
            if (diff_filled < sqrt_len_clamped) {
                diff_ring[diff_filled] = diff_now;
                kahan_add(diff_now, diff_sum_seed, diff_sum_seed_c);
                diff_filled += 1;

                if (diff_filled == sqrt_len_clamped) {
                    if (is_wma) {
                        a_diff = 0.0f;
                        s_diff = 0.0f;
                        a_diff_c = 0.0f;
                        s_diff_c = 0.0f;
                        for (int j = 0; j < sqrt_len_clamped; ++j) {
                            const float w = static_cast<float>(j + 1);
                            const float v = diff_ring[j];
                            kahan_add(v, a_diff, a_diff_c);
                            kahan_add(w * v, s_diff, s_diff_c);
                        }
                        diff_wma_init_done = true;
                        hull_val = sum_read(s_diff, s_diff_c) * inv_w_sqrt;
                    } else {
                        diff_ema = sum_read(diff_sum_seed, diff_sum_seed_c) / static_cast<float>(sqrt_len_clamped);
                        diff_ema_init_done = true;
                        hull_val = diff_ema;
                    }
                }
            } else {
                const float old = diff_ring[diff_pos];
                diff_ring[diff_pos] = diff_now;
                diff_pos += 1; if (diff_pos == sqrt_len_clamped) diff_pos = 0;

                if (is_wma) {
                    if (!diff_wma_init_done) {
                        diff_wma_init_done = true;
                    }
                    const float a_prev = a_diff;
                    kahan_add(diff_now - old, a_diff, a_diff_c);

                    kahan_add(fmaf(static_cast<float>(sqrt_len_clamped), diff_now, -a_prev), s_diff, s_diff_c);
                    hull_val = sum_read(s_diff, s_diff_c) * inv_w_sqrt;
                } else {
                    if (!diff_ema_init_done) {
                        diff_ema = diff_now;
                        diff_ema_init_done = true;
                    } else {
                        diff_ema = fmaf(alpha_sqrt, diff_now - diff_ema, diff_ema);
                    }
                    hull_val = diff_ema;
                }
            }
        }

        float ec_now = NAN;
        if (e0_init_done) {
            if (!ec_init_done) {
                ec_prev = e0_prev;
                ec_now = ec_prev;
                ec_init_done = true;
            } else {
                ec_now = dma_update_ec(x, e0_prev, ec_prev, alpha_e, ema_gain_limit);
                ec_prev = ec_now;
            }
        }

        if (!isnan(hull_val) && !isnan(ec_now)) {
            out_tm[base_out + i * stride] = 0.5f * (hull_val + ec_now);
        }
    }
}


template<int TY>
__device__ void dma_ms1p_tiled_f32_tx1_ty_core(const float* __restrict__ prices_tm,
                                          int hull_length,
                                          int ema_length,
                                          int ema_gain_limit,
                                          int hull_type,
                                          int series_len,
                                          int num_series,
                                          const int* __restrict__ first_valids,
                                          int sqrt_len,
                                          float* __restrict__ out_tm) {
    const int base_series = blockIdx.x * TY;
    const int series_idx = base_series + threadIdx.y;
    if (series_idx >= num_series) { return; }
    if (series_len <= 0 || hull_length <= 0 || ema_length <= 0) { return; }

    extern __shared__ __align__(16) float smem[];
    float* diff_ring = smem + threadIdx.y * max(1, sqrt_len);

    const int stride = num_series;
    const int base_out = series_idx;

    if (threadIdx.x == 0) {
        for (int i = 0; i < series_len; ++i) { out_tm[base_out + i * stride] = NAN; }
    }
    if (threadIdx.x != 0) { return; }

    const int first_valid = first_valids[series_idx];
    if (first_valid >= series_len) { return; }

    const int half = hull_length / 2;
    const int sqrt_len_clamped = max(1, sqrt_len);
    const float denom_half_f = (half        > 0 ? wsum_norm_i32(half)              : 1.0f);
    const float denom_full_f = (hull_length > 0 ? wsum_norm_i32(hull_length)       : 1.0f);
    const float denom_sqrt_f = (sqrt_len_clamped > 0 ? wsum_norm_i32(sqrt_len_clamped) : 1.0f);
    const float inv_w_half   = 1.0f / denom_half_f;
    const float inv_w_full   = 1.0f / denom_full_f;
    const float inv_w_sqrt   = 1.0f / denom_sqrt_f;

    const float alpha_e = 2.0f / (float(ema_length) + 1.0f);
    const int i0_e = first_valid + (ema_length > 0 ? ema_length - 1 : 0);

    float e0_prev = 0.0f; bool e0_init_done = false;
    float ec_prev = 0.0f; bool ec_init_done = false;

    const int i0_half = first_valid + (half > 0 ? half - 1 : 0);
    const int i0_full = first_valid + (hull_length > 0 ? hull_length - 1 : 0);

    float a_half = 0.0f, s_half = 0.0f, a_half_c = 0.0f, s_half_c = 0.0f; bool half_ready = false;
    float a_full = 0.0f, s_full = 0.0f, a_full_c = 0.0f, s_full_c = 0.0f; bool full_ready = false;

    int diff_filled = 0, diff_pos = 0; float diff_sum_seed = 0.0f, diff_sum_seed_c = 0.0f;
    float a_diff = 0.0f, s_diff = 0.0f, a_diff_c = 0.0f, s_diff_c = 0.0f; bool diff_wma_init_done = false;
    float diff_ema = 0.0f; bool diff_ema_init_done = false;
    const float alpha_sqrt = (sqrt_len_clamped > 0) ? 2.0f / (float(sqrt_len_clamped) + 1.0f) : 0.0f;

    float e_half_prev = 0.0f, e_full_prev = 0.0f; bool e_half_init_done = false, e_full_init_done = false;
    const float alpha_half = (half > 0) ? 2.0f / (float(half) + 1.0f) : 0.0f;
    const float alpha_full = (hull_length > 0) ? 2.0f / (float(hull_length) + 1.0f) : 0.0f;

    const bool is_wma = (hull_type == 0);
    float hull_val = NAN;

    for (int i = first_valid; i < series_len; ++i) {
        const int idx = i * stride + series_idx; const float x = prices_tm[idx];

        if (!e0_init_done) {
            if (i >= i0_e) {
                int start = i + 1 - ema_length; float sum = 0.0f, sum_c = 0.0f;
                for (int k = start; k <= i; ++k) { kahan_add(prices_tm[k * stride + series_idx], sum, sum_c); }
                e0_prev = sum_read(sum, sum_c) / float(ema_length); e0_init_done = true;
            }
        } else { e0_prev = fmaf(alpha_e, x - e0_prev, e0_prev); }

        float diff_now = NAN;
        if (is_wma) {
            if (half > 0) {
                if (!half_ready) {
                    if (i >= i0_half) {
                        int start = i + 1 - half; float sum = 0.0f, sum_c = 0.0f, wsum = 0.0f, wsum_c = 0.0f;
                        for (int j = 0; j < half; ++j) {
                            const int sidx = start + j; const float w = float(j + 1);
                            const float v = prices_tm[sidx * stride + series_idx];
                            kahan_add(v, sum, sum_c); kahan_add(w * v, wsum, wsum_c);
                        }
                        a_half = sum; s_half = wsum; a_half_c = sum_c; s_half_c = wsum_c; half_ready = true;
                    }
                } else {
                    const float a_prev = a_half; const float old = prices_tm[(i - half) * stride + series_idx];
                    kahan_add(x - old, a_half, a_half_c);
                    kahan_add(float(half) * x, s_half, s_half_c); kahan_add(-a_prev, s_half, s_half_c);
                }
            }
            if (hull_length > 0) {
                if (!full_ready) {
                    if (i >= i0_full) {
                        int start = i + 1 - hull_length; float sum = 0.0f, sum_c = 0.0f, wsum = 0.0f, wsum_c = 0.0f;
                        for (int j = 0; j < hull_length; ++j) {
                            const int sidx = start + j; const float w = float(j + 1);
                            const float v = prices_tm[sidx * stride + series_idx];
                            kahan_add(v, sum, sum_c); kahan_add(w * v, wsum, wsum_c);
                        }
                        a_full = sum; s_full = wsum; a_full_c = sum_c; s_full_c = wsum_c; full_ready = true;
                    }
                } else {
                    const float a_prev = a_full; const float old = prices_tm[(i - hull_length) * stride + series_idx];
                    kahan_add(x - old, a_full, a_full_c);
                    kahan_add(float(hull_length) * x, s_full, s_full_c); kahan_add(-a_prev, s_full, s_full_c);
                }
            }
            if (half_ready && full_ready) {
                const float w_half = sum_read(s_half, s_half_c) * inv_w_half; const float w_full = sum_read(s_full, s_full_c) * inv_w_full;
                diff_now = 2.0f * w_half - w_full;
            }
        } else {
            if (half > 0) {
                if (!e_half_init_done) {
                    if (i >= i0_half) {
                        int start = i + 1 - half; float sum = 0.0f, sum_c = 0.0f;
                        for (int k = start; k <= i; ++k) { kahan_add(prices_tm[k * stride + series_idx], sum, sum_c); }
                        e_half_prev = sum_read(sum, sum_c) / float(half); e_half_init_done = true;
                    }
                } else { e_half_prev = fmaf(alpha_half, x - e_half_prev, e_half_prev); }
            }
            if (hull_length > 0) {
                if (!e_full_init_done) {
                    if (i >= i0_full) {
                        int start = i + 1 - hull_length; float sum = 0.0f, sum_c = 0.0f;
                        for (int k = start; k <= i; ++k) { kahan_add(prices_tm[k * stride + series_idx], sum, sum_c); }
                        e_full_prev = sum_read(sum, sum_c) / float(hull_length); e_full_init_done = true;
                    }
                } else { e_full_prev = fmaf(alpha_full, x - e_full_prev, e_full_prev); }
            }
            if (e_half_init_done && e_full_init_done) { diff_now = 2.0f * e_half_prev - e_full_prev; }
        }

        if (!isnan(diff_now) && sqrt_len_clamped > 0) {
            if (diff_filled < sqrt_len_clamped) {
                diff_ring[diff_filled] = diff_now; kahan_add(diff_now, diff_sum_seed, diff_sum_seed_c);
                diff_filled += 1;
                if (diff_filled == sqrt_len_clamped) {
                    if (is_wma) {
                        a_diff = 0.0f; s_diff = 0.0f; a_diff_c = 0.0f; s_diff_c = 0.0f;
                        for (int j = 0; j < sqrt_len_clamped; ++j) {
                            const float w = float(j + 1); const float v = diff_ring[j];
                            kahan_add(v, a_diff, a_diff_c); kahan_add(w * v, s_diff, s_diff_c);
                        }
                        diff_wma_init_done = true;
                        hull_val = sum_read(s_diff, s_diff_c) * inv_w_sqrt;
                    } else {
                        diff_ema = sum_read(diff_sum_seed, diff_sum_seed_c) / float(sqrt_len_clamped); diff_ema_init_done = true; hull_val = diff_ema;
                    }
                }
            } else {
                const float old = diff_ring[diff_pos]; diff_ring[diff_pos] = diff_now; diff_pos += 1; if (diff_pos == sqrt_len_clamped) diff_pos = 0;
                if (is_wma) {
                    if (!diff_wma_init_done) { diff_wma_init_done = true; }
                    const float a_prev = a_diff;
                    kahan_add(diff_now - old, a_diff, a_diff_c);
                    kahan_add(fmaf(float(sqrt_len_clamped), diff_now, -a_prev), s_diff, s_diff_c);
                    hull_val = sum_read(s_diff, s_diff_c) * inv_w_sqrt;
                } else {
                    if (!diff_ema_init_done) { diff_ema = diff_now; diff_ema_init_done = true; }
                    else { diff_ema = fmaf(alpha_sqrt, diff_now - diff_ema, diff_ema); }
                    hull_val = diff_ema;
                }
            }
        }

        float ec_now = NAN;
        if (e0_init_done) {
            if (!ec_init_done) { ec_prev = e0_prev; ec_now = ec_prev; ec_init_done = true; }
            else { ec_now = dma_update_ec(x, e0_prev, ec_prev, alpha_e, ema_gain_limit); ec_prev = ec_now; }
        }

        if (!isnan(hull_val) && !isnan(ec_now)) { out_tm[base_out + i * stride] = 0.5f * (hull_val + ec_now); }
    }
}

extern "C" {
__global__ void dma_ms1p_tiled_f32_tx1_ty2(const float* __restrict__ prices_tm,
                                           int hull_length,
                                           int ema_length,
                                           int ema_gain_limit,
                                           int hull_type,
                                           int series_len,
                                           int num_series,
                                           const int* __restrict__ first_valids,
                                           int sqrt_len,
                                           float* __restrict__ out_tm) {
    dma_ms1p_tiled_f32_tx1_ty_core<2>(prices_tm, hull_length, ema_length, ema_gain_limit, hull_type,
                                      series_len, num_series, first_valids, sqrt_len, out_tm);
}
__global__ void dma_ms1p_tiled_f32_tx1_ty4(const float* __restrict__ prices_tm,
                                           int hull_length,
                                           int ema_length,
                                           int ema_gain_limit,
                                           int hull_type,
                                           int series_len,
                                           int num_series,
                                           const int* __restrict__ first_valids,
                                           int sqrt_len,
                                           float* __restrict__ out_tm) {
    dma_ms1p_tiled_f32_tx1_ty_core<4>(prices_tm, hull_length, ema_length, ema_gain_limit, hull_type,
                                      series_len, num_series, first_valids, sqrt_len, out_tm);
}
}


// ===========================================================================
// NEOETHOS f64 LANE  --  closer 4, round 3
//
// CPU reference: dma_scalar (src/indicators/moving_averages/dma.rs:561-628),
// reached through dma_with_kernel (:296) -> dma_compute_into.
//
// PERIOD-SWEPT, and the swept int is the HULL length: ma_batch.rs:1868
// assigns sweep.hull_length = period_range while ema_length stays 20,
// ema_gain_limit stays 50 and hull_ma_type is pinned to "WMA" (:1871). The
// EMA-hull arm of dma_scalar (:700-737) is therefore unreachable from this
// lane and is not reproduced; mapping the swept int onto ema_length instead
// would compute a different indicator.
//
// SHAPE: one thread per combo walking bars ASCENDING. Five accumulators are
// carried across bars -- the EMA of price, the two weighted sliding sums that
// build the hull difference, the weighted sliding sum over the difference
// ring, and the gain-limited EC recursion whose g is chosen by comparing two
// candidate residuals. The creator-defined gain domain is symmetric,
// -ema_gain_limit..=+ema_gain_limit in integer tenths. None of that can be
// rebuilt bar-parallel without changing the rounding.
//
// The only per-thread array is the difference ring, whose length is
// round(sqrt(hull_length)); DMA_NEO_MAX_SQRT bounds it and
// F64Kernel::max_period REFUSES a larger period by name rather than
// truncating the window or moving the row to the host.
// ===========================================================================

#define DMA_NEO_MAX_SQRT 64
#define DMA_NEO_MAX_PERIOD 4160

static __forceinline__ __device__ double dma_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// dma.rs:797 -- `target.floor() as i64`. Rust's float-to-int cast SATURATES
// and maps NaN to 0; C's does neither, so the conversion is spelled out.
static __forceinline__ __device__ long long dma_neo_floor_to_i64(double v) {
    if (isnan(v)) return 0LL;
    const double f = floor(v);
    if (f <= -9223372036854775808.0) return -9223372036854775807LL - 1LL;
    if (f >= 9223372036854775808.0) return 9223372036854775807LL;
    return (long long)f;
}

extern "C" __global__
void dma_neo_batch_f64(const double* __restrict__ data,
                       int n,
                       const int* __restrict__ periods,
                       int n_combos,
                       int first_valid,
                       double* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;
    if (n <= 0) return;

    double* __restrict__ row = out + (size_t)combo * (size_t)n;
    const double nn = dma_neo_qnan();

    const int hull_length = periods[combo];
    const int ema_length = 20;      // ma_batch.rs:1865
    const int ema_gain_limit = 50;  // ma_batch.rs:1866

    // dma_prepare, :395-398 -- `!is_nan` over the single close series, which
    // is exactly F64FirstValidRule::AllInputsNonNan for CloseSlice.
    int first = first_valid;
    if (first < 0) first = 0;

    bool refused = false;
    if (first >= n) refused = true;
    if (hull_length <= 0 || hull_length > n) refused = true;   // :406
    if (ema_length <= 0 || ema_length > n) refused = true;     // :412
    if (hull_length > DMA_NEO_MAX_PERIOD) refused = true;

    int sqrt_len = 0;
    if (!refused) {
        sqrt_len = (int)round(sqrt((double)hull_length));      // :420
        const int longer = hull_length > ema_length ? hull_length : ema_length;
        const long long needed = (long long)longer + (long long)sqrt_len;
        if ((long long)(n - first) < needed) refused = true;   // :422
        if (sqrt_len > DMA_NEO_MAX_SQRT) refused = true;
    }

    if (refused) {
        for (int i = 0; i < n; ++i) row[i] = nn;
        return;
    }

    const int longer = hull_length > ema_length ? hull_length : ema_length;
    // :305 -- warmup_end = first + max(hull, ema) + sqrt_len - 1.
    long long warm_ll = (long long)first + (long long)longer + (long long)sqrt_len - 1;
    const int nan_end = warm_ll < (long long)n ? (int)warm_ll : n;
    for (int i = 0; i < nan_end; ++i) row[i] = nn;
    // Every bar past the warmup that dma_scalar does not write is left
    // uninitialised by alloc_with_nan_prefix; NaN is the only honest value.
    for (int i = nan_end; i < n; ++i) row[i] = nn;

    const double alpha_e = 2.0 / ((double)ema_length + 1.0);
    const double one_minus_alpha_e = 1.0 - alpha_e;
    const int i0_e = first + (ema_length > 0 ? ema_length - 1 : 0);
    double e0_prev = 0.0;
    bool e0_init_done = false;
    double ec_prev = 0.0;
    bool ec_init_done = false;

    const int half = hull_length / 2;
    double hull_val = nn;

    const int i0_half = first + (half > 0 ? half - 1 : 0);
    const int i0_full = first + (hull_length > 0 ? hull_length - 1 : 0);

    // :588 -- wsum(p) = (p * (p + 1)) as f64 / 2.0, then `.max(1.0)`.
    const double wsum_half = (double)((long long)half * (long long)(half + 1)) / 2.0;
    const double wsum_full =
        (double)((long long)hull_length * (long long)(hull_length + 1)) / 2.0;
    const double wsum_sqrt =
        (double)((long long)sqrt_len * (long long)(sqrt_len + 1)) / 2.0;
    const double den_half = fmax(wsum_half, 1.0);
    const double den_full = fmax(wsum_full, 1.0);
    const double den_sqrt = fmax(wsum_sqrt, 1.0);

    double a_half = 0.0, s_half = 0.0;
    bool half_ready = false;
    double a_full = 0.0, s_full = 0.0;
    bool full_ready = false;

    double diff_ring[DMA_NEO_MAX_SQRT];
    int diff_pos = 0;
    int diff_filled = 0;
    double a_diff = 0.0, s_diff = 0.0;

    for (int i = first; i < n; ++i) {
        const double x = data[i];

        // :633-645 -- SMA seed, then the one-fma EMA.
        if (!e0_init_done) {
            if (i >= i0_e) {
                const int start = i + 1 - ema_length;
                double sum = 0.0;
                for (int k = start; k <= i; ++k) sum += data[k];
                e0_prev = sum / (double)ema_length;
                e0_init_done = true;
            }
        } else {
            e0_prev = fma(x, alpha_e, one_minus_alpha_e * e0_prev);
        }

        double diff_now = nn;

        // :651-676 -- the half window.
        if (half > 0) {
            if (!half_ready) {
                if (i >= i0_half) {
                    const int start = i + 1 - half;
                    double sum = 0.0;
                    double wsum_local = 0.0;
                    for (int j = 0; j < half; ++j) {
                        const double w = (double)(j + 1);
                        const double v = data[start + j];
                        sum += v;
                        wsum_local += w * v;
                    }
                    a_half = sum;
                    s_half = wsum_local;
                    half_ready = true;
                }
            } else {
                const double a_prev = a_half;
                a_half = a_prev + x - data[i - half];
                s_half = s_half + (double)half * x - a_prev;
            }
        }

        // :678-700 -- the full window.
        if (hull_length > 0) {
            if (!full_ready) {
                if (i >= i0_full) {
                    const int start = i + 1 - hull_length;
                    double sum = 0.0;
                    double wsum_local = 0.0;
                    for (int j = 0; j < hull_length; ++j) {
                        const double w = (double)(j + 1);
                        const double v = data[start + j];
                        sum += v;
                        wsum_local += w * v;
                    }
                    a_full = sum;
                    s_full = wsum_local;
                    full_ready = true;
                }
            } else {
                const double a_prev = a_full;
                a_full = a_prev + x - data[i - hull_length];
                s_full = s_full + (double)hull_length * x - a_prev;
            }
        }

        if (half_ready && full_ready) {
            const double w_half = s_half / den_half;
            const double w_full = s_full / den_full;
            diff_now = 2.0 * w_half - w_full;   // :700
        }

        // :735-781 -- the difference ring and its weighted sum.
        if (isfinite(diff_now) && sqrt_len > 0) {
            if (diff_filled < sqrt_len) {
                diff_ring[diff_filled] = diff_now;
                diff_filled += 1;
                if (diff_filled == sqrt_len) {
                    a_diff = 0.0;
                    s_diff = 0.0;
                    for (int j = 0; j < sqrt_len; ++j) {
                        const double w = (double)(j + 1);
                        const double v = diff_ring[j];
                        a_diff += v;
                        s_diff += w * v;
                    }
                    hull_val = s_diff / den_sqrt;
                }
            } else {
                const double old = diff_ring[diff_pos];
                diff_ring[diff_pos] = diff_now;
                diff_pos = (diff_pos + 1) % sqrt_len;

                const double a_prev = a_diff;
                a_diff = a_prev + diff_now - old;
                s_diff = s_diff + (double)sqrt_len * diff_now - a_prev;
                hull_val = s_diff / den_sqrt;
            }
        }

        // :783-820 -- the gain-limited EC recursion.
        double ec_now = nn;
        if (e0_init_done) {
            if (!ec_init_done) {
                ec_prev = e0_prev;
                ec_init_done = true;
                ec_now = ec_prev;
            } else {
                const double dx = x - ec_prev;
                const double t = alpha_e * dx;
                const double base = fma(e0_prev, alpha_e, one_minus_alpha_e * ec_prev);
                const double r = x - base;

                double g_sel;
                if (t == 0.0) {
                    g_sel = 0.0;
                } else {
                    const long long limit_i = (long long)ema_gain_limit;
                    const long long lower_i = -limit_i;
                    const double target = (r / t) * 10.0;
                    long long i0 = dma_neo_floor_to_i64(target);
                    if (i0 < lower_i) i0 = lower_i;
                    else if (i0 > limit_i) i0 = limit_i;
                    const long long i1 = (i0 < limit_i) ? (i0 + 1) : i0;
                    // Dickson's definition constructs gain as value1 / 10.
                    const double g0 = (double)i0 / 10.0;
                    const double g1 = (double)i1 / 10.0;
                    const double e0 = fabs(r - t * g0);
                    const double e1 = fabs(r - t * g1);
                    g_sel = (e0 <= e1) ? g0 : g1;
                }

                ec_now = fma(e0_prev + g_sel * dx, alpha_e, one_minus_alpha_e * ec_prev);
                ec_prev = ec_now;
            }
        }

        if (isfinite(hull_val) && isfinite(ec_now)) {
            row[i] = 0.5 * (hull_val + ec_now);   // :824
        }
    }
}
