#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

#ifndef FRAMA_NAN
#define FRAMA_NAN (__int_as_float(0x7fffffff))
#endif

#ifndef FRAMA_MAX_WINDOW
#define FRAMA_MAX_WINDOW 1024
#endif


#ifndef FRAMA_USE_FAST_MATH

#define FRAMA_USE_FAST_MATH 0
#endif

#if FRAMA_USE_FAST_MATH
#  define FRAMA_LOG2F __log2f
#  define FRAMA_EXP2F __exp2f
#else
#  define FRAMA_LOG2F log2f
#  define FRAMA_EXP2F exp2f
#endif

__device__ __forceinline__ float frama_clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

struct MonoDeque {
    int* buf;
    int head;
    int tail;
    const float* data;
    int stride;
    int offset;
};

__device__ __forceinline__ int md_inc(int idx) {
    ++idx;
    if (idx >= FRAMA_MAX_WINDOW) {
        idx = 0;
    }
    return idx;
}

__device__ __forceinline__ int md_dec(int idx) {
    if (idx == 0) {
        idx = FRAMA_MAX_WINDOW;
    }
    return idx - 1;
}

__device__ __forceinline__ MonoDeque make_deque(
    int* storage,
    const float* data,
    int stride,
    int offset) {
    MonoDeque dq;
    dq.buf = storage;
    dq.head = 0;
    dq.tail = 0;
    dq.data = data;
    dq.stride = stride;
    dq.offset = offset;
    return dq;
}

__device__ __forceinline__ void md_clear(MonoDeque* dq) {
    dq->head = 0;
    dq->tail = 0;
}

__device__ __forceinline__ bool md_empty(const MonoDeque* dq) {
    return dq->head == dq->tail;
}

__device__ __forceinline__ int md_front(const MonoDeque* dq) {
    return dq->buf[dq->head];
}

__device__ __forceinline__ float md_value(const MonoDeque* dq, int idx) {
    return dq->data[idx * dq->stride + dq->offset];
}

__device__ __forceinline__ void md_expire(MonoDeque* dq, int idx_out) {
    if (!md_empty(dq) && dq->buf[dq->head] == idx_out) {
        dq->head = md_inc(dq->head);
    }
}

__device__ __forceinline__ void md_push_max(MonoDeque* dq, int idx) {
    const float cur = md_value(dq, idx);
    while (!md_empty(dq)) {
        int last_slot = md_dec(dq->tail);
        int last_idx = dq->buf[last_slot];
        if (md_value(dq, last_idx) >= cur) {
            break;
        }
        dq->tail = last_slot;
        if (dq->tail == dq->head) {
            break;
        }
    }
    dq->buf[dq->tail] = idx;
    dq->tail = md_inc(dq->tail);
}

__device__ __forceinline__ void md_push_min(MonoDeque* dq, int idx) {
    const float cur = md_value(dq, idx);
    while (!md_empty(dq)) {
        int last_slot = md_dec(dq->tail);
        int last_idx = dq->buf[last_slot];
        if (md_value(dq, last_idx) <= cur) {
            break;
        }
        dq->tail = last_slot;
        if (dq->tail == dq->head) {
            break;
        }
    }
    dq->buf[dq->tail] = idx;
    dq->tail = md_inc(dq->tail);
}

struct ExtremesPair {
    float maxv;
    float minv;
};

__device__ __forceinline__ ExtremesPair frama_front_or(
    const MonoDeque* dq_max,
    const MonoDeque* dq_min,
    float* prev_max,
    float* prev_min) {
    float maxv = *prev_max;
    float minv = *prev_min;
    if (!md_empty(dq_max)) {
        maxv = md_value(dq_max, md_front(dq_max));
    }
    if (!md_empty(dq_min)) {
        minv = md_value(dq_min, md_front(dq_min));
    }
    *prev_max = maxv;
    *prev_min = minv;
    ExtremesPair out = {maxv, minv};
    return out;
}

__device__ __forceinline__ void md_swap(MonoDeque* a, MonoDeque* b) {
    MonoDeque tmp = *a;
    *a = *b;
    *b = tmp;
}

extern "C" __global__ void frama_batch_f32(const float* __restrict__ high,
                                            const float* __restrict__ low,
                                            const float* __restrict__ close,
                                            const int* __restrict__ windows,
                                            const int* __restrict__ scs,
                                            const int* __restrict__ fcs,
                                            int series_len,
                                            int n_combos,
                                            int first_valid,
                                            float* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) {
        return;
    }

    int storage_full_max[FRAMA_MAX_WINDOW];
    int storage_full_min[FRAMA_MAX_WINDOW];
    int storage_left_max[FRAMA_MAX_WINDOW];
    int storage_left_min[FRAMA_MAX_WINDOW];
    int storage_right_max[FRAMA_MAX_WINDOW];
    int storage_right_min[FRAMA_MAX_WINDOW];

    MonoDeque d_full_max = make_deque(storage_full_max, high, 1, 0);
    MonoDeque d_full_min = make_deque(storage_full_min, low, 1, 0);
    MonoDeque d_left_max = make_deque(storage_left_max, high, 1, 0);
    MonoDeque d_left_min = make_deque(storage_left_min, low, 1, 0);
    MonoDeque d_right_max = make_deque(storage_right_max, high, 1, 0);
    MonoDeque d_right_min = make_deque(storage_right_min, low, 1, 0);

    float* row_out = out + combo * series_len;
    for (int i = 0; i < series_len; ++i) {
        row_out[i] = FRAMA_NAN;
    }

    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    int window = windows[combo];
    int sc = scs[combo];
    int fc = fcs[combo];
    if (window <= 0 || sc <= 0 || fc <= 0) {
        return;
    }

    int win = window;
    if (win & 1) {
        ++win;
    }
    if (win <= 1 || win > FRAMA_MAX_WINDOW) {
        return;
    }
    const int half = win / 2;
    if (half <= 0) {
        return;
    }

    const int tail_len = series_len - first_valid;
    if (tail_len < win) {
        return;
    }

    float seed = 0.0f;
    for (int j = 0; j < win; ++j) {
        seed += close[first_valid + j];
    }
    seed /= static_cast<float>(win);
    const int warm = first_valid + win - 1;
    row_out[warm] = seed;

    md_clear(&d_full_max);
    md_clear(&d_full_min);
    md_clear(&d_left_max);
    md_clear(&d_left_min);
    md_clear(&d_right_max);
    md_clear(&d_right_min);

    const int win_end = first_valid + win;
    for (int idx = first_valid; idx < win_end; ++idx) {
        const float hi = high[idx];
        const float lo = low[idx];
        if (isnan(hi) || isnan(lo)) {
            continue;
        }
        md_push_max(&d_full_max, idx);
        md_push_min(&d_full_min, idx);
        if (idx < first_valid + half) {
            md_push_max(&d_left_max, idx);
            md_push_min(&d_left_min, idx);
        } else {
            md_push_max(&d_right_max, idx);
            md_push_min(&d_right_min, idx);
        }
    }


    const float sc_f     = (float)sc;
    const float fc_f     = (float)fc;
    const float inv_half = 1.0f / (float)half;
    const float inv_win  = 1.0f / (float)win;
    const float log2_k   = FRAMA_LOG2F(2.0f / (sc_f + 1.0f));
    const float sc_lim   = 2.0f / (sc_f + 1.0f);
    const bool  sc_is_one = (sc == 1);

    float d_prev = 1.0f;

    float pm1 = FRAMA_NAN;
    float pm2 = FRAMA_NAN;
    float pm3 = FRAMA_NAN;
    float pn1 = FRAMA_NAN;
    float pn2 = FRAMA_NAN;
    float pn3 = FRAMA_NAN;

    int half_progress = 0;

    for (int i = warm + 1; i < series_len; ++i) {
        const int idx_out = i - win;
        md_expire(&d_full_max, idx_out);
        md_expire(&d_full_min, idx_out);
        md_expire(&d_left_max, idx_out);
        md_expire(&d_left_min, idx_out);
        md_expire(&d_right_max, idx_out + half);
        md_expire(&d_right_min, idx_out + half);

        const int newest = i - 1;
        const float hi = high[newest];
        const float lo = low[newest];
        if (!(isnan(hi) || isnan(lo))) {
            md_push_max(&d_full_max, newest);
            md_push_min(&d_full_min, newest);
            if (newest < idx_out + half) {
                md_push_max(&d_left_max, newest);
                md_push_min(&d_left_min, newest);
            } else {
                md_push_max(&d_right_max, newest);
                md_push_min(&d_right_min, newest);
            }
        }

        ExtremesPair right = frama_front_or(&d_right_max, &d_right_min, &pm1, &pn1);
        ExtremesPair left = frama_front_or(&d_left_max, &d_left_min, &pm2, &pn2);
        ExtremesPair full = frama_front_or(&d_full_max, &d_full_min, &pm3, &pn3);

        const float hi_i    = high[i];
        const float lo_i    = low[i];
        const float close_i = close[i];
        const float prev    = row_out[i - 1];

        if (!isnan(hi_i) && !isnan(lo_i) && !isnan(close_i) && !isnan(prev)) {

            const float n1 = (right.maxv - right.minv) * inv_half;
            const float n2 = (left .maxv - left .minv) * inv_half;
            const float n3 = (full .maxv - full .minv) * inv_win;

            float d_cur = d_prev;
            if (n1 > 0.0f && n2 > 0.0f && n3 > 0.0f) {
                d_cur = FRAMA_LOG2F(n1 + n2) - FRAMA_LOG2F(n3);
            }
            d_prev = d_cur;


            float alpha0 = FRAMA_EXP2F(log2_k * (d_cur - 1.0f));
            alpha0 = frama_clampf(alpha0, 0.1f, 1.0f);

            const float old_n = (2.0f - alpha0) / alpha0;
            float new_n = fc_f;
            if (!sc_is_one) {
                new_n = (sc_f - fc_f) * ((old_n - 1.0f) / (sc_f - 1.0f)) + fc_f;
            }
            float alpha = 2.0f / (new_n + 1.0f);
            alpha = frama_clampf(alpha, sc_lim, 1.0f);


            row_out[i] = fmaf(alpha, (close_i - prev), prev);
        } else {
            row_out[i] = prev;
        }

        ++half_progress;
        if (half_progress == half) {
            md_swap(&d_left_max, &d_right_max);
            md_swap(&d_left_min, &d_right_min);
            md_clear(&d_right_max);
            md_clear(&d_right_min);
            half_progress = 0;
        }
    }
}

extern "C" __global__ void frama_many_series_one_param_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const int* __restrict__ first_valids,
    int num_series,
    int series_len,
    int window,
    int sc,
    int fc,
    float* __restrict__ out_tm) {
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= num_series) {
        return;
    }

    int storage_full_max[FRAMA_MAX_WINDOW];
    int storage_full_min[FRAMA_MAX_WINDOW];
    int storage_left_max[FRAMA_MAX_WINDOW];
    int storage_left_min[FRAMA_MAX_WINDOW];
    int storage_right_max[FRAMA_MAX_WINDOW];
    int storage_right_min[FRAMA_MAX_WINDOW];

    MonoDeque d_full_max = make_deque(storage_full_max, high_tm, num_series, series);
    MonoDeque d_full_min = make_deque(storage_full_min, low_tm, num_series, series);
    MonoDeque d_left_max = make_deque(storage_left_max, high_tm, num_series, series);
    MonoDeque d_left_min = make_deque(storage_left_min, low_tm, num_series, series);
    MonoDeque d_right_max = make_deque(storage_right_max, high_tm, num_series, series);
    MonoDeque d_right_min = make_deque(storage_right_min, low_tm, num_series, series);

    float* col_out = out_tm + series;
    for (int row = 0; row < series_len; ++row) {
        col_out[row * num_series] = FRAMA_NAN;
    }

    if (window <= 0 || sc <= 0 || fc <= 0) {
        return;
    }

    int first_valid = first_valids[series];
    if (first_valid < 0 || first_valid >= series_len) {
        return;
    }

    int win = window;
    if (win & 1) {
        ++win;
    }
    if (win <= 1 || win > FRAMA_MAX_WINDOW) {
        return;
    }
    const int half = win / 2;
    if (half <= 0) {
        return;
    }

    const int tail_len = series_len - first_valid;
    if (tail_len < win) {
        return;
    }

    float seed = 0.0f;
    for (int j = 0; j < win; ++j) {
        seed += close_tm[(first_valid + j) * num_series + series];
    }
    seed /= static_cast<float>(win);
    const int warm = first_valid + win - 1;
    col_out[warm * num_series] = seed;

    md_clear(&d_full_max);
    md_clear(&d_full_min);
    md_clear(&d_left_max);
    md_clear(&d_left_min);
    md_clear(&d_right_max);
    md_clear(&d_right_min);

    const int win_end = first_valid + win;
    for (int idx = first_valid; idx < win_end; ++idx) {
        const float hi = high_tm[idx * num_series + series];
        const float lo = low_tm[idx * num_series + series];
        if (isnan(hi) || isnan(lo)) {
            continue;
        }
        md_push_max(&d_full_max, idx);
        md_push_min(&d_full_min, idx);
        if (idx < first_valid + half) {
            md_push_max(&d_left_max, idx);
            md_push_min(&d_left_min, idx);
        } else {
            md_push_max(&d_right_max, idx);
            md_push_min(&d_right_min, idx);
        }
    }


    const float sc_f     = (float)sc;
    const float fc_f     = (float)fc;
    const float inv_half = 1.0f / (float)half;
    const float inv_win  = 1.0f / (float)win;
    const float log2_k   = FRAMA_LOG2F(2.0f / (sc_f + 1.0f));
    const float sc_lim   = 2.0f / (sc_f + 1.0f);
    const bool  sc_is_one = (sc == 1);

    float d_prev = 1.0f;

    float pm1 = FRAMA_NAN;
    float pm2 = FRAMA_NAN;
    float pm3 = FRAMA_NAN;
    float pn1 = FRAMA_NAN;
    float pn2 = FRAMA_NAN;
    float pn3 = FRAMA_NAN;

    int half_progress = 0;

    for (int i = warm + 1; i < series_len; ++i) {
        const int idx_out = i - win;
        md_expire(&d_full_max, idx_out);
        md_expire(&d_full_min, idx_out);
        md_expire(&d_left_max, idx_out);
        md_expire(&d_left_min, idx_out);
        md_expire(&d_right_max, idx_out + half);
        md_expire(&d_right_min, idx_out + half);

        const int newest = i - 1;
        const float hi = high_tm[newest * num_series + series];
        const float lo = low_tm[newest * num_series + series];
        if (!(isnan(hi) || isnan(lo))) {
            md_push_max(&d_full_max, newest);
            md_push_min(&d_full_min, newest);
            if (newest < idx_out + half) {
                md_push_max(&d_left_max, newest);
                md_push_min(&d_left_min, newest);
            } else {
                md_push_max(&d_right_max, newest);
                md_push_min(&d_right_min, newest);
            }
        }

        ExtremesPair right = frama_front_or(&d_right_max, &d_right_min, &pm1, &pn1);
        ExtremesPair left = frama_front_or(&d_left_max, &d_left_min, &pm2, &pn2);
        ExtremesPair full = frama_front_or(&d_full_max, &d_full_min, &pm3, &pn3);

        const float hi_i    = high_tm [i * num_series + series];
        const float lo_i    = low_tm  [i * num_series + series];
        const float close_i = close_tm[i * num_series + series];
        const float prev    = col_out[(i - 1) * num_series];

        if (!isnan(hi_i) && !isnan(lo_i) && !isnan(close_i) && !isnan(prev)) {
            const float n1 = (right.maxv - right.minv) * inv_half;
            const float n2 = (left .maxv - left .minv) * inv_half;
            const float n3 = (full .maxv - full .minv) * inv_win;

            float d_cur = d_prev;
            if (n1 > 0.0f && n2 > 0.0f && n3 > 0.0f) {
                d_cur = FRAMA_LOG2F(n1 + n2) - FRAMA_LOG2F(n3);
            }
            d_prev = d_cur;

            float alpha0 = FRAMA_EXP2F(log2_k * (d_cur - 1.0f));
            alpha0 = frama_clampf(alpha0, 0.1f, 1.0f);

            const float old_n = (2.0f - alpha0) / alpha0;
            float new_n = fc_f;
            if (!sc_is_one) {
                new_n = (sc_f - fc_f) * ((old_n - 1.0f) / (sc_f - 1.0f)) + fc_f;
            }
            float alpha = 2.0f / (new_n + 1.0f);
            alpha = frama_clampf(alpha, sc_lim, 1.0f);

            col_out[i * num_series] = fmaf(alpha, (close_i - prev), prev);
        } else {
            col_out[i * num_series] = prev;
        }

        ++half_progress;
        if (half_progress == half) {
            md_swap(&d_left_max, &d_right_max);
            md_swap(&d_left_min, &d_right_min);
            md_clear(&d_right_max);
            md_clear(&d_right_min);
            half_progress = 0;
        }
    }
}

/* ===========================================================================
 * S4 f64 LANE — frama (fractal adaptive moving average)
 * ---------------------------------------------------------------------------
 * CPU authority: src/indicators/moving_averages/frama.rs
 * `frama-f64-v3-finite-hlc-segment-reset-even-window-stable-fma-v2`.
 * Every non-finite H/L/C row resets the seed, `d_prev`, previous output, and
 * extrema ownership. Each following maximal finite segment is a fresh FRAMA
 * run and emits only after `win` consecutive finite rows.
 *
 * WHAT THE f32 KERNEL ABOVE GETS WRONG, AND IS FIXED HERE
 *
 *  1. `__int_as_float(0x7f...)` is an f32 NaN bit pattern; here the prefix is
 *     `__longlong_as_double(0x7ff8...)`.
 *  2. The f64 lane carries extrema indices in four monotonic half-window
 *     deques, so it needs no type-sized extrema sentinels at all.
 *  3. `fmaxf`/`fminf` x1 each -> `fmax`/`fmin`. Non-finite values never reach
 *     the extrema scan under the v3 segment contract.
 *  4. `.clamp(0.1, 1.0)` IS NOT `fmin(fmax(x, 0.1), 1.0)`. Rust's `f64::clamp`
 *     is `if self < min {min} else if self > max {max} else {self}`, so a NaN
 *     PASSES THROUGH. `fmax`/`fmin` would replace it with a bound. Both clamps
 *     below are written as the CPU's comparison chain deliberately — this is
 *     the one place in this shard where the NaN rule inverts, and getting it
 *     "right" the usual way would be wrong here.
 *  5. `exp`/`ln` replace `expf`/`logf`. Two of these are unavoidable
 *     sub-ulp divergence sources between CUDA's libdevice and the host libm;
 *     declared, not discovered.
 *
 * Each update computes from `[i-win, i-half)` and `[i-half, i)` before it
 * expires `i-win`, moves `i-half` from right to left, and pushes `i` right.
 * Capacity 513 represents every index in a 512-row half without making a full
 * ring alias the empty `head == tail` state. `w_ln`, `sc_floor` and the parity
 * fix-up are hoisted exactly where the CPU hoists them. `d_prev` seeds at 1.0
 * at every finite segment boundary. One thread owns one column and walks rows
 * ascending. Requests whose evenized window exceeds 1024 fail closed in the
 * host and are rejected here again as kernel defense in depth.
 *
 * PARAMETERS: the swept int is `window` (CPU default 10). `sc` and `fc` are
 * held at the CPU batch defaults 300 and 1 (frama.rs:114,118) because the
 * sweep carries one integer and `window` is the length this indicator is
 * swept over.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define NEO_FRAMA_SC_DEFAULT 300
#define NEO_FRAMA_FC_DEFAULT 1
#define NEO_FRAMA_MAX_WINDOW 1024
#define NEO_FRAMA_HALF_DEQUE_CAPACITY 513

struct NeoFramaDequeF64V3 {
    int* storage;
    int head;
    int tail;
};

__device__ __forceinline__ NeoFramaDequeF64V3 neo_frama_make_deque_f64_v3(
    int* storage)
{
    NeoFramaDequeF64V3 deque = {storage, 0, 0};
    return deque;
}

__device__ __forceinline__ bool neo_frama_empty_f64_v3(
    const NeoFramaDequeF64V3* deque)
{
    return deque->head == deque->tail;
}

__device__ __forceinline__ void neo_frama_clear_f64_v3(
    NeoFramaDequeF64V3* deque)
{
    deque->head = 0;
    deque->tail = 0;
}

__device__ __forceinline__ int neo_frama_front_f64_v3(
    const NeoFramaDequeF64V3* deque)
{
    return deque->storage[deque->head];
}

__device__ __forceinline__ void neo_frama_expire_f64_v3(
    NeoFramaDequeF64V3* deque, int idx_out)
{
    if (!neo_frama_empty_f64_v3(deque) &&
        neo_frama_front_f64_v3(deque) == idx_out) {
        deque->head = (deque->head + 1) % NEO_FRAMA_HALF_DEQUE_CAPACITY;
    }
}

__device__ __forceinline__ void neo_frama_push_max_f64_v3(
    NeoFramaDequeF64V3* deque, int idx, const double* __restrict__ values)
{
    while (!neo_frama_empty_f64_v3(deque)) {
        const int last =
            (deque->tail + NEO_FRAMA_HALF_DEQUE_CAPACITY - 1) %
            NEO_FRAMA_HALF_DEQUE_CAPACITY;
        if (values[deque->storage[last]] >= values[idx]) break;
        deque->tail = last;
    }
    deque->storage[deque->tail] = idx;
    deque->tail =
        (deque->tail + 1) % NEO_FRAMA_HALF_DEQUE_CAPACITY;
}

__device__ __forceinline__ void neo_frama_push_min_f64_v3(
    NeoFramaDequeF64V3* deque, int idx, const double* __restrict__ values)
{
    while (!neo_frama_empty_f64_v3(deque)) {
        const int last =
            (deque->tail + NEO_FRAMA_HALF_DEQUE_CAPACITY - 1) %
            NEO_FRAMA_HALF_DEQUE_CAPACITY;
        if (values[deque->storage[last]] <= values[idx]) break;
        deque->tail = last;
    }
    deque->storage[deque->tail] = idx;
    deque->tail =
        (deque->tail + 1) % NEO_FRAMA_HALF_DEQUE_CAPACITY;
}

__device__ __forceinline__ double neo_frama_stable_update_f64_v2(
    double close, double previous, double alpha)
{
    return __fma_rn(alpha, __dsub_rn(close, previous), previous);
}

extern "C" __global__
void frama_neo_batch_f64(const double* __restrict__ high,
                         const double* __restrict__ low,
                         const double* __restrict__ close,
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
    const int window = periods[combo];

    const int sc = NEO_FRAMA_SC_DEFAULT;
    const int fc = NEO_FRAMA_FC_DEFAULT;

    if (len <= 0 || window <= 0 || window > len ||
        window > NEO_FRAMA_MAX_WINDOW) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }
    const int win = window + (window & 1);
    if (win > NEO_FRAMA_MAX_WINDOW) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }
    (void)first_valid;

    for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;

    int left_max_storage[NEO_FRAMA_HALF_DEQUE_CAPACITY];
    int left_min_storage[NEO_FRAMA_HALF_DEQUE_CAPACITY];
    int right_max_storage[NEO_FRAMA_HALF_DEQUE_CAPACITY];
    int right_min_storage[NEO_FRAMA_HALF_DEQUE_CAPACITY];
    NeoFramaDequeF64V3 left_max = neo_frama_make_deque_f64_v3(left_max_storage);
    NeoFramaDequeF64V3 left_min = neo_frama_make_deque_f64_v3(left_min_storage);
    NeoFramaDequeF64V3 right_max = neo_frama_make_deque_f64_v3(right_max_storage);
    NeoFramaDequeF64V3 right_min = neo_frama_make_deque_f64_v3(right_min_storage);

    const int half = win >> 1;
    const double win_f = (double)win;
    const double half_f = (double)half;
    const double w_ln = log(2.0 / ((double)sc + 1.0));
    const double sc_floor = 2.0 / ((double)sc + 1.0);
    const double LN_2 = 0.693147180559945309417232121458176568;

    int finite_run = 0;
    double seed = 0.0;
    double d_prev = 1.0;
    double previous = NEO_F64_NAN;

    for (int i = 0; i < len; ++i) {
        if (!isfinite(high[i]) || !isfinite(low[i]) || !isfinite(close[i])) {
            finite_run = 0;
            seed = 0.0;
            d_prev = 1.0;
            previous = NEO_F64_NAN;
            neo_frama_clear_f64_v3(&left_max);
            neo_frama_clear_f64_v3(&left_min);
            neo_frama_clear_f64_v3(&right_max);
            neo_frama_clear_f64_v3(&right_min);
            o[i] = NEO_F64_NAN;
            continue;
        }

        if (finite_run < win) {
            seed += close[i];
            if (finite_run < half) {
                neo_frama_push_max_f64_v3(&left_max, i, high);
                neo_frama_push_min_f64_v3(&left_min, i, low);
            } else {
                neo_frama_push_max_f64_v3(&right_max, i, high);
                neo_frama_push_min_f64_v3(&right_min, i, low);
            }
            finite_run += 1;
            if (finite_run == win) {
                previous = seed / (double)win;
                o[i] = previous;
            }
            continue;
        }

        const double max1 = high[neo_frama_front_f64_v3(&right_max)];
        const double min1 = low[neo_frama_front_f64_v3(&right_min)];
        const double max2 = high[neo_frama_front_f64_v3(&left_max)];
        const double min2 = low[neo_frama_front_f64_v3(&left_min)];

        const double max3 = fmax(max1, max2);
        const double min3 = fmin(min1, min2);

        const double n1 = (max1 - min1) / half_f;
        const double n2 = (max2 - min2) / half_f;
        const double n3 = (max3 - min3) / win_f;

        double d_cur;
        if (n1 > 0.0 && n2 > 0.0 && n3 > 0.0) {
            d_cur = (log(n1 + n2) - log(n3)) / LN_2;
        } else {
            d_cur = d_prev;
        }
        d_prev = d_cur;

        // Rust `clamp`: NaN passes through. NOT fmin/fmax. frama.rs:719.
        double alpha0 = exp(w_ln * (d_cur - 1.0));
        if (alpha0 < 0.1) alpha0 = 0.1; else if (alpha0 > 1.0) alpha0 = 1.0;

        const double old_n = (2.0 - alpha0) / alpha0;
        const double new_n =
            (double)(sc - fc) * ((old_n - 1.0) / ((double)sc - 1.0)) + (double)fc;

        double alpha = 2.0 / (new_n + 1.0);
        if (alpha < sc_floor) alpha = sc_floor; else if (alpha > 1.0) alpha = 1.0;

        previous = neo_frama_stable_update_f64_v2(close[i], previous, alpha);
        o[i] = previous;

        const int idx_out = i - win;
        const int crossing = i - half;
        neo_frama_expire_f64_v3(&left_max, idx_out);
        neo_frama_expire_f64_v3(&left_min, idx_out);
        neo_frama_expire_f64_v3(&right_max, crossing);
        neo_frama_expire_f64_v3(&right_min, crossing);
        neo_frama_push_max_f64_v3(&left_max, crossing, high);
        neo_frama_push_min_f64_v3(&left_min, crossing, low);
        neo_frama_push_max_f64_v3(&right_max, i, high);
        neo_frama_push_min_f64_v3(&right_min, i, low);
    }
}
