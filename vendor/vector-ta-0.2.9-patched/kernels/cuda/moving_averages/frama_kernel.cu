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
 * CPU oracle: src/indicators/moving_averages/frama.rs
 *   `frama_prepare`      (:253) — first_valid over the TRIPLE, the win parity
 *                                 fix-up, and every Err branch
 *   `frama_compute_into` (:320) — the seed, `mean(close[first .. first+win])`
 *   `frama_small_scan`   (:648) — the fractal-dimension scan and the EMA step
 *
 * The four `frama_small_scan_const::<10|14|20|32>` specialisations and
 * `frama_scalar_deque` (win > 32) are the SAME arithmetic: a const generic and
 * a monotonic deque change how the window max/min is found, not what it is —
 * max/min are exact — and the alpha/EMA lines are byte-identical in all three.
 * One kernel therefore serves every win, and no special case is written.
 *
 * WHAT THE f32 KERNEL ABOVE GETS WRONG, AND IS FIXED HERE
 *
 *  1. `__int_as_float(0x7f...)` is an f32 NaN bit pattern; here the prefix is
 *     `__longlong_as_double(0x7ff8...)`.
 *  2. `f64::MIN` / `f64::MAX` seeds. The CPU seeds max with `f64::MIN` —
 *     -1.797e308, the most negative FINITE double — NOT -infinity. `-DBL_MAX`
 *     is the f64 spelling. An f32 kernel that seeded with `-FLT_MAX` is an
 *     epsilon-class constant sized for the wrong type; this is the same bug
 *     in its largest form.
 *  3. `fmaxf`/`fminf` x1 each -> `fmax`/`fmin`. These MUST stay max/min and
 *     not become comparison chains: the CPU uses `f64::max`, which returns the
 *     NON-NaN operand, so one NaN bar inside the window is absorbed rather
 *     than poisoning `d_prev` for the rest of the series.
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
 * `w_ln`, `sc_floor` and the parity fix-up `if (win & 1) win += 1` are hoisted
 * exactly where the CPU hoists them. `d_prev` seeds at 1.0 and CARRIES across
 * bars, and `out[i]` reads `out[i-1]`: two carried scalars, so one thread per
 * column walking ascending, never a scan.
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

    int win = window;
    if (win & 1) win += 1;

    if (len <= 0 || window <= 0 || window > len ||
        first_valid < 0 || first_valid >= len ||
        (len - first_valid) < win) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    const int warm = first_valid + win - 1;
    for (int i = 0; i < len && i < warm; ++i) o[i] = NEO_F64_NAN;

    // frama.rs:337 — plain ascending sum of `win` closes, then one divide.
    double seed = 0.0;
    for (int j = first_valid; j < first_valid + win; ++j) seed += close[j];
    seed = seed / (double)win;
    o[warm] = seed;

    const int half = win >> 1;
    const double win_f = (double)win;
    const double half_f = (double)half;
    const double w_ln = log(2.0 / ((double)sc + 1.0));
    const double sc_floor = 2.0 / ((double)sc + 1.0);
    const double LN_2 = 0.693147180559945309417232121458176568;
    const double DBL_MIN_FINITE = -1.7976931348623157e308;  // f64::MIN
    const double DBL_MAX_FINITE =  1.7976931348623157e308;  // f64::MAX

    double d_prev = 1.0;

    for (int i = first_valid + win; i < len; ++i) {
        const int seg_start = i - win;
        const int mid = seg_start + half;

        double max1 = DBL_MIN_FINITE, min1 = DBL_MAX_FINITE;
        double max2 = DBL_MIN_FINITE, min2 = DBL_MAX_FINITE;

        int j = seg_start;
        while (j + 1 < mid) {
            max2 = fmax(max2, fmax(high[j], high[j + 1]));
            min2 = fmin(min2, fmin(low[j],  low[j + 1]));
            j += 2;
        }
        if (j < mid) {
            max2 = fmax(max2, high[j]);
            min2 = fmin(min2, low[j]);
        }

        j = mid;
        while (j + 1 < i) {
            max1 = fmax(max1, fmax(high[j], high[j + 1]));
            min1 = fmin(min1, fmin(low[j],  low[j + 1]));
            j += 2;
        }
        if (j < i) {
            max1 = fmax(max1, high[j]);
            min1 = fmin(min1, low[j]);
        }

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

        // frama.rs:724 — ONE fused rounding on the close term, a plain product
        // on the carry term. Not `alpha*c + (1-alpha)*prev`.
        o[i] = fma(close[i], alpha, (1.0 - alpha) * o[i - 1]);
    }
}
