#include <cuda_runtime.h>
#include <math_constants.h>

static __device__ __forceinline__ float f32_nan() {

    return __int_as_float(0x7fffffff);
}

struct KahanF32 {
    float s;
    float c;
    __device__ __forceinline__ KahanF32() : s(0.0f), c(0.0f) {}
    __device__ __forceinline__ void add(float x) {
        float y = x - c;
        float t = s + y;
        c = (t - s) - y;
        s = t;
    }
    __device__ __forceinline__ float value() const { return s + c; }
};

#ifndef CE_DQ_MAX
#define CE_DQ_MAX 256
#endif

extern "C" __global__ void chandelier_exit_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const int    len,
    const int    first_valid,
    const int*   __restrict__ periods,
    const float* __restrict__ mults,
    const int    rows,
    const int    use_close_flag,
    float*       __restrict__ out
)
{
    const int stride = blockDim.x * gridDim.x;
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += stride)
    {
        const int   period = periods[r];
        const float mult   = mults[r];


        if (period <= 0) {
            float* long_row_ptr  = out + (size_t)(2 * r)     * len;
            float* short_row_ptr = out + (size_t)(2 * r + 1) * len;
            for (int i = 0; i < len; ++i) {
                long_row_ptr[i]  = f32_nan();
                short_row_ptr[i] = f32_nan();
            }
            continue;
        }

        const int   warm = first_valid + period - 1;
        const float invP = 1.0f / (float)period;

        float* long_row_ptr  = out + (size_t)(2 * r)     * len;
        float* short_row_ptr = out + (size_t)(2 * r + 1) * len;


        bool  prev_close_set = false;
        float prev_close     = 0.0f;
        float atr            = CUDART_NAN_F;
        KahanF32 warm_sum;
        int   warm_count = 0;


        float long_raw_prev  = CUDART_NAN_F;
        float short_raw_prev = CUDART_NAN_F;
        int   dir_prev = 1;


        int   hi_idx = -1, lo_idx = -1;
        float hi_val = f32_nan(), lo_val = f32_nan();

        const bool use_dq_fast = (period <= CE_DQ_MAX);
        int max_head = 0, max_tail = 0, max_count = 0;
        int min_head = 0, min_tail = 0, min_count = 0;
        int max_idx_q[CE_DQ_MAX], min_idx_q[CE_DQ_MAX];
        float max_val_q[CE_DQ_MAX], min_val_q[CE_DQ_MAX];

        for (int i = 0; i < len; ++i) {
            const float h = high[i];
            const float l = low[i];
            const float c = close[i];


            if (i >= first_valid) {
                const float hl = fabsf(h - l);
                float tr;
                if (!prev_close_set) {
                    tr = hl;
                    prev_close = c;
                    prev_close_set = true;
                } else {
                    const float hc = fabsf(h - prev_close);
                    const float lc = fabsf(l - prev_close);
                    tr = fmaxf(hl, fmaxf(hc, lc));
                    prev_close = c;
                }

                if (warm_count < period) {
                    if (!isnan(tr)) warm_sum.add(tr);
                    ++warm_count;
                    if (warm_count == period) {
                        atr = warm_sum.value() * invP;
                    }
                } else {

                    if (!isnan(tr) && !isnan(atr)) {
                        atr += (tr - atr) * invP;
                    }
                }
            }


            const float x_max = use_close_flag ? c : h;
            const float x_min = use_close_flag ? c : l;

            const int start = (i - period + 1 > 0) ? (i - period + 1) : 0;

            if (use_dq_fast) {
                while (max_count > 0 && max_idx_q[max_head] < start) {
                    max_head = (max_head + 1) % CE_DQ_MAX;
                    --max_count;
                }
                while (min_count > 0 && min_idx_q[min_head] < start) {
                    min_head = (min_head + 1) % CE_DQ_MAX;
                    --min_count;
                }

                if (!isnan(x_max)) {
                    while (max_count > 0) {
                        const int back = (max_tail + CE_DQ_MAX - 1) % CE_DQ_MAX;
                        if (max_val_q[back] <= x_max) {
                            max_tail = back;
                            --max_count;
                        } else {
                            break;
                        }
                    }
                    max_idx_q[max_tail] = i;
                    max_val_q[max_tail] = x_max;
                    max_tail = (max_tail + 1) % CE_DQ_MAX;
                    ++max_count;
                }

                if (!isnan(x_min)) {
                    while (min_count > 0) {
                        const int back = (min_tail + CE_DQ_MAX - 1) % CE_DQ_MAX;
                        if (min_val_q[back] >= x_min) {
                            min_tail = back;
                            --min_count;
                        } else {
                            break;
                        }
                    }
                    min_idx_q[min_tail] = i;
                    min_val_q[min_tail] = x_min;
                    min_tail = (min_tail + 1) % CE_DQ_MAX;
                    ++min_count;
                }

                if (max_count > 0) {
                    hi_idx = max_idx_q[max_head];
                    hi_val = max_val_q[max_head];
                } else {
                    hi_idx = -1;
                    hi_val = f32_nan();
                }
                if (min_count > 0) {
                    lo_idx = min_idx_q[min_head];
                    lo_val = min_val_q[min_head];
                } else {
                    lo_idx = -1;
                    lo_val = f32_nan();
                }
            } else {
                if (!isnan(x_max) && (isnan(hi_val) || x_max >= hi_val)) { hi_val = x_max; hi_idx = i; }
                if (!isnan(x_min) && (isnan(lo_val) || x_min <= lo_val)) { lo_val = x_min; lo_idx = i; }

                if (hi_idx < start) {
                    hi_val = f32_nan(); hi_idx = -1;
                    for (int j = start; j <= i; ++j) {
                        const float v = use_close_flag ? close[j] : high[j];
                        if (!isnan(v) && (isnan(hi_val) || v > hi_val)) { hi_val = v; hi_idx = j; }
                    }
                }
                if (lo_idx < start) {
                    lo_val = f32_nan(); lo_idx = -1;
                    for (int j = start; j <= i; ++j) {
                        const float v = use_close_flag ? close[j] : low[j];
                        if (!isnan(v) && (isnan(lo_val) || v < lo_val)) { lo_val = v; lo_idx = j; }
                    }
                }
            }


            if (i < warm) {
                long_row_ptr[i]  = f32_nan();
                short_row_ptr[i] = f32_nan();
                continue;
            }


            if (isnan(atr) || isnan(hi_val) || isnan(lo_val)) {
                long_row_ptr[i]  = f32_nan();
                short_row_ptr[i] = f32_nan();
                continue;
            }


            const float ls0 = fmaf(-mult, atr, hi_val);
            const float ss0 = fmaf( mult, atr, lo_val);

            const float lsp = (i == warm || isnan(long_raw_prev))  ? ls0 : long_raw_prev;
            const float ssp = (i == warm || isnan(short_raw_prev)) ? ss0 : short_raw_prev;

            float ls = ls0, ss = ss0;
            if (i > warm) {
                const float pc = close[i - 1];
                if (pc > lsp) ls = (ls0 > lsp) ? ls0 : lsp;
                if (pc < ssp) ss = (ss0 < ssp) ? ss0 : ssp;
            }

            int d;
            if (c > ssp) d = 1;
            else if (c < lsp) d = -1;
            else d = dir_prev;

            long_raw_prev  = ls;
            short_raw_prev = ss;
            dir_prev = d;

            long_row_ptr[i]  = (d == 1)  ? ls : f32_nan();
            short_row_ptr[i] = (d == -1) ? ss : f32_nan();
        }
    }
}

extern "C" __global__ void chandelier_exit_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const int    cols,
    const int    rows,
    const int    period,
    const float  mult,
    const int*   __restrict__ first_valids,
    const int    use_close_flag,
    float*       __restrict__ out_tm
)
{
    const int stride = blockDim.x * gridDim.x;
    for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < cols; s += stride)
    {
        const int fv = first_valids[s];
        if (period <= 0) {

            float* long_mat  = out_tm + 0;
            float* short_mat = out_tm + (size_t)rows * cols;
            for (int t = 0; t < rows; ++t) {
                long_mat[t * cols + s]  = f32_nan();
                short_mat[t * cols + s] = f32_nan();
            }
            continue;
        }

        const int   warm_base = fv + period - 1;
        const float invP      = 1.0f / (float)period;

        float* long_mat  = out_tm + 0;
        float* short_mat = out_tm + (size_t)rows * cols;


        float atr = CUDART_NAN_F;
        KahanF32 warm_sum;
        int   warm_count = 0;
        float prev_close = 0.0f; bool prev_set = false;


        float long_raw_prev  = CUDART_NAN_F;
        float short_raw_prev = CUDART_NAN_F;
        int   dir_prev = 1;


        int   hi_idx = -1, lo_idx = -1;
        float hi_val = f32_nan(), lo_val = f32_nan();

        for (int t = 0; t < rows; ++t) {
            const int idx = t * cols + s;
            const float h = high_tm[idx];
            const float l = low_tm[idx];
            const float c = close_tm[idx];


            if (t >= fv) {
                const float hl = fabsf(h - l);
                float tr;
                if (!prev_set) { tr = hl; prev_close = c; prev_set = true; }
                else {
                    const float hc = fabsf(h - prev_close);
                    const float lc = fabsf(l - prev_close);
                    tr = fmaxf(hl, fmaxf(hc, lc));
                    prev_close = c;
                }

                if (warm_count < period) {
                    if (!isnan(tr)) warm_sum.add(tr);
                    ++warm_count;
                    if (warm_count == period) atr = warm_sum.value() * invP;
                } else {
                    if (!isnan(tr) && !isnan(atr)) {
                        const float delta = (tr - atr) * invP;
                        float corr = 0.0f;
                        float y = delta - corr;
                        float tt = atr + y;
                        corr = (tt - atr) - y;
                        atr = tt;
                    }
                }
            }


            const float x_max = use_close_flag ? c : h;
            const float x_min = use_close_flag ? c : l;

            if (!isnan(x_max) && (isnan(hi_val) || x_max >= hi_val)) { hi_val = x_max; hi_idx = t; }
            if (!isnan(x_min) && (isnan(lo_val) || x_min <= lo_val)) { lo_val = x_min; lo_idx = t; }

            const int start = (t - period + 1 > 0) ? (t - period + 1) : 0;
            if (hi_idx < start) {
                hi_val = f32_nan(); hi_idx = -1;
                for (int j = start; j <= t; ++j) {
                    const float v = use_close_flag ? close_tm[j * cols + s] : high_tm[j * cols + s];
                    if (!isnan(v) && (isnan(hi_val) || v > hi_val)) { hi_val = v; hi_idx = j; }
                }
            }
            if (lo_idx < start) {
                lo_val = f32_nan(); lo_idx = -1;
                for (int j = start; j <= t; ++j) {
                    const float v = use_close_flag ? close_tm[j * cols + s] : low_tm[j * cols + s];
                    if (!isnan(v) && (isnan(lo_val) || v < lo_val)) { lo_val = v; lo_idx = j; }
                }
            }


            if (t < warm_base) {
                long_mat[idx]  = f32_nan();
                short_mat[idx] = f32_nan();
                continue;
            }

            if (isnan(atr) || isnan(hi_val) || isnan(lo_val)) {
                long_mat[idx]  = f32_nan();
                short_mat[idx] = f32_nan();
                continue;
            }

            const float ls0 = fmaf(-mult, atr, hi_val);
            const float ss0 = fmaf( mult, atr, lo_val);

            const float lsp = (t == warm_base || isnan(long_raw_prev))  ? ls0 : long_raw_prev;
            const float ssp = (t == warm_base || isnan(short_raw_prev)) ? ss0 : short_raw_prev;

            float ls = ls0, ss = ss0;
            if (t > warm_base) {
                const float pc = close_tm[(t - 1) * cols + s];
                if (pc > lsp) ls = (ls0 > lsp) ? ls0 : lsp;
                if (pc < ssp) ss = (ss0 < ssp) ? ss0 : ssp;
            }

            int d;
            if (c > ssp) d = 1;
            else if (c < lsp) d = -1;
            else d = dir_prev;

            long_raw_prev  = ls;
            short_raw_prev = ss;
            dir_prev = d;

            long_mat[idx]  = (d == 1)  ? ls : f32_nan();
            short_mat[idx] = (d == -1) ? ss : f32_nan();
        }
    }
}


// ===========================================================================
// S2 f64 LANE — chandelier_exit
// ===========================================================================
// Reference: src/indicators/chandelier_exit.rs
//   `ce_first_valid`             (:626) — the first-valid rule
//   `ce_prepare`                 (:646) — refusals, warm = first + period - 1
//   `chandelier_exit_with_kernel`(:882) — the ATR call and the stop recurrence
//   src/indicators/atr.rs
//   `first_valid_hlc`            (:197) — ATR's OWN first-valid
//   `atr_compute_into_scalar`    (:319) — the Wilder seed and update
//   Canonical production sweeps `period`; mult = 3.0 and use_close = true.
//   The exact registered outputs are `long_stop` and `short_stop`, both emitted
//   by the resident pair entry below.
//
// FIRST-VALID: `HlcCloseOnly`, NOT the common rule.
//   With `use_close == true` — the batch default — `ce_first_valid` returns
//   `close.iter().position(|x| !x.is_nan())` and NEVER LOOKS AT high or low.
//   That is the `adxr` rule. (With `use_close == false` it is the MIN of the
//   three firsts, a fourth rule retained for the standalone/public ABI; the
//   canonical resident feature route fails closed before admitting it.)
//
// TWO DIFFERENT FIRST-VALID INDICES LIVE IN THIS ONE KERNEL.
//   The stop recurrence uses `ce_first_valid`. The ATR it consumes is produced
//   by `atr_with_kernel`, which computes its own `first_valid_hlc` — all three
//   series non-NaN SIMULTANEOUSLY — and its own warmup off that. On a frame
//   where high or low starts later than close the two indices differ, and
//   using one for both would move the ATR seed window. Both are derived inside
//   the kernel from the arrays it already has; the host's `first_valid` is
//   used only as a bounds check.
//
// ROUNDINGS.
//   ATR seed:  sum_tr accumulated ascending, then `rma = sum_tr / length` (1).
//   ATR step:  `rma = (-alpha).mul_add(rma, rma) + alpha * tr` — fma, mul, add,
//              THREE. NOT `(tr - rma).mul_add(alpha, rma)`, which is two. This
//              is the same class of error the brief names in natr; here the
//              CPU is the one taking three, so the kernel takes three.
//   stops:     `ls0 = ai.mul_add(-mult, highest)` and
//              `ss0 = ai.mul_add(mult, lowest)` — ONE fma each.
//
// `ls0.max(lsp)` / `ss0.min(ssp)` ARE `f64::max` / `f64::min` — they return the
// non-NaN operand — so they become `fmax` / `fmin`. The `if` guards around
// them (`close[i-1] > lsp`) are plain comparisons and stay plain.
//
// THE TRUE-RANGE MAX INSIDE ATR IS AN IF-CHAIN (`if hc > tr { tr = hc }`), so
// a NaN `hc` leaves `tr` alone — deliberately NOT fmax, which would be the
// same thing here only by accident of which operand is NaN.
// ===========================================================================

#define CE_MAX_PERIOD 512

__device__ __forceinline__ double neo_s2_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

__device__ __forceinline__ void chandelier_exit_row_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int n,
    int period,
    double mult,
    bool use_close,
    int first_valid,
    int* __restrict__ dq_max,
    int* __restrict__ dq_min,
    int deque_scratch_cap,
    double* __restrict__ row_long_stop,
    double* __restrict__ row_short_stop)
{
    const bool emit_short = row_short_stop != nullptr;
    for (int i = 0; i < n; ++i) {
        row_long_stop[i] = neo_s2_qnan();
        if (emit_short) row_short_stop[i] = neo_s2_qnan();
    }

    const bool declined =
        (n <= 0) ||
        (period <= 0) || (period > n) ||
        (first_valid < 0) || (first_valid >= n);
    if (declined) return;

    int cap = 1;
    while (cap < period && cap > 0) cap <<= 1;
    if (cap <= 0 || cap > deque_scratch_cap || dq_max == nullptr || dq_min == nullptr) return;

    // ---- ce_first_valid -------------------------------------------------
    int fc = -1;
    for (int i = 0; i < n; ++i) { if (!isnan(close[i])) { fc = i; break; } }
    int first;
    if (use_close) {
        if (fc < 0) return;
        first = fc;
    } else {
        int fh = -1, fl = -1;
        for (int i = 0; i < n; ++i) { if (!isnan(high[i])) { fh = i; break; } }
        for (int i = 0; i < n; ++i) { if (!isnan(low[i]))  { fl = i; break; } }
        if (fh < 0 || fl < 0 || fc < 0) return;
        first = fh < fl ? fh : fl;
        if (fc < first) first = fc;
    }
    if ((n - first) < period) return;

    const int warm = first + period - 1;

    // ---- ATR, with its OWN first_valid and warmup ------------------------
    // `atr_with_kernel` allocates a NaN prefix to `atr_warm` and writes from
    // there; values before it are NaN and feed the stop formula as NaN, which
    // is exactly what the CPU does.
    int atr_first = n;
    for (int i = 0; i < n; ++i) {
        if (!isnan(high[i]) && !isnan(low[i]) && !isnan(close[i])) { atr_first = i; break; }
    }
    if (atr_first >= n) return;
    if ((n - atr_first) < period) return;
    const int atr_warm = atr_first + period - 1;
    if (atr_warm >= n) return;

    const double alpha = 1.0 / (double)period;

    // ---- the stop recurrence, with ATR computed in step ------------------
    // ATR is a strictly forward recurrence and the stop loop walks the same
    // bars forward, so it is produced inline rather than in a scratch matrix.
    // Its seed needs bars [atr_first, atr_warm], which the loop reaches before
    // the first bar it is read at only when atr_warm <= warm; when it does not,
    // the seed is finished in this pre-pass instead.
    double sum_tr = high[atr_first] - low[atr_first];
    if (atr_warm > atr_first) {
        double prev_c = close[atr_first];
        for (int i = atr_first + 1; i <= atr_warm; ++i) {
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
    double rma = sum_tr / (double)period;
    double atr_prev_c = (atr_warm + 1 > 0) ? close[atr_warm] : close[0];
    int atr_next = atr_warm + 1;

    double long_raw_prev = neo_s2_qnan();
    double short_raw_prev = neo_s2_qnan();
    int prev_dir = 1;

    // Monotone deques over the max/min source. `cap` is the CPU's exact
    // `period.next_power_of_two()`, while production supplies runtime-sized
    // resident scratch and the preserved primary supplies its original cap.
    const int mask = cap - 1;
    unsigned hmax = 0u, tmax = 0u, hmin = 0u, tmin = 0u;

    const double* src_max = use_close ? close : high;
    const double* src_min = use_close ? close : low;

    for (int i = 0; i < n; ++i) {
        // Keep the ATR up to date for bar i.
        while (atr_next <= i && atr_next < n) {
            const int k = atr_next;
            const double hi = high[k];
            const double lo = low[k];
            double tr = hi - lo;
            const double hc = fabs(hi - atr_prev_c);
            if (hc > tr) tr = hc;
            const double lc = fabs(lo - atr_prev_c);
            if (lc > tr) tr = lc;
            rma = fma(-alpha, rma, rma) + alpha * tr;
            atr_prev_c = close[k];
            atr_next += 1;
        }
        const double ai = (i < atr_warm) ? neo_s2_qnan() : rma;

        while (hmax != tmax) {
            const int idx = dq_max[hmax & mask];
            if (idx + period <= i) hmax += 1u; else break;
        }
        while (hmin != tmin) {
            const int idx = dq_min[hmin & mask];
            if (idx + period <= i) hmin += 1u; else break;
        }

        const double v_max = src_max[i];
        if (!isnan(v_max)) {
            while (hmax != tmax) {
                const unsigned back_pos = (tmax - 1u) & (unsigned)mask;
                if (src_max[dq_max[back_pos]] < v_max) tmax -= 1u; else break;
            }
            dq_max[tmax & mask] = i;
            tmax += 1u;
        }
        const double v_min = src_min[i];
        if (!isnan(v_min)) {
            while (hmin != tmin) {
                const unsigned back_pos = (tmin - 1u) & (unsigned)mask;
                if (src_min[dq_min[back_pos]] > v_min) tmin -= 1u; else break;
            }
            dq_min[tmin & mask] = i;
            tmin += 1u;
        }

        if (i < warm) continue;

        const double highest = (hmax != tmax) ? src_max[dq_max[hmax & mask]] : neo_s2_qnan();
        const double lowest  = (hmin != tmin) ? src_min[dq_min[hmin & mask]] : neo_s2_qnan();

        const double ls0 = fma(ai, -mult, highest);
        const double ss0 = fma(ai, mult, lowest);

        const double lsp = (i == warm || isnan(long_raw_prev))  ? ls0 : long_raw_prev;
        const double ssp = (i == warm || isnan(short_raw_prev)) ? ss0 : short_raw_prev;

        const double ls = (i > warm && close[i - 1] > lsp) ? fmax(ls0, lsp) : ls0;
        const double ss = (i > warm && close[i - 1] < ssp) ? fmin(ss0, ssp) : ss0;

        int d;
        if (close[i] > ssp)      d = 1;
        else if (close[i] < lsp) d = -1;
        else                     d = prev_dir;

        long_raw_prev = ls;
        short_raw_prev = ss;
        prev_dir = d;

        row_long_stop[i] = (d == 1) ? ls : neo_s2_qnan();
        if (emit_short) row_short_stop[i] = (d == -1) ? ss : neo_s2_qnan();
    }
}

extern "C" __global__ void chandelier_exit_outputs_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int n,
    const int* __restrict__ periods,
    const double* __restrict__ mults,
    int use_close,
    int rows,
    int first_valid,
    int deque_scratch_cap,
    int* __restrict__ max_deque_scratch,
    int* __restrict__ min_deque_scratch,
    double* __restrict__ out_long_stop,
    double* __restrict__ out_short_stop)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || n <= 0) return;

    const size_t output_offset = (size_t)row * (size_t)n;
    const size_t scratch_offset = (size_t)row * (size_t)deque_scratch_cap;
    chandelier_exit_row_f64(
        high,
        low,
        close,
        n,
        periods[row],
        mults[row],
        use_close != 0,
        first_valid,
        max_deque_scratch + scratch_offset,
        min_deque_scratch + scratch_offset,
        deque_scratch_cap,
        out_long_stop + output_offset,
        out_short_stop + output_offset);
}

extern "C" __global__ void neoethos_chandelier_exit_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_combos || n <= 0) return;

    int dq_max[CE_MAX_PERIOD * 2];
    int dq_min[CE_MAX_PERIOD * 2];
    chandelier_exit_row_f64(
        high,
        low,
        close,
        n,
        periods[row],
        3.0,
        true,
        first_valid,
        dq_max,
        dq_min,
        CE_MAX_PERIOD,
        out + (size_t)row * (size_t)n,
        nullptr);
}
