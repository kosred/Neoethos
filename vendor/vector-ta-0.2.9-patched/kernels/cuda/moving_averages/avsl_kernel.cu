#include <cuda.h>
#include <cuda_runtime.h>

#ifndef AVSL_MAX_WIN
#define AVSL_MAX_WIN 200
#endif

#ifndef AVSL_MAX_PRE_RING

#define AVSL_MAX_PRE_RING 512
#endif

__device__ __forceinline__ float avsl_adj(float x) {

    if (x > -1.0f && x < 0.0f) return -1.0f;
    if (x >= 0.0f && x < 1.0f) return 1.0f;
    return x;
}


extern "C" __global__ void avsl_batch_f32(
    const float* __restrict__ close,
    const float* __restrict__ low,
    const float* __restrict__ volume,
    const int series_len,
    const int first_valid,
    const int* __restrict__ fast_periods,
    const int* __restrict__ slow_periods,
    const float* __restrict__ multipliers,
    float* __restrict__ out,
    const int rows)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const int fast = max(1, fast_periods[row]);
    const int slow = max(1, slow_periods[row]);
    const float mult = multipliers[row];

    const int base = first_valid + slow - 1;
    const int warmup2 = base + slow - 1;

    float* __restrict__ dst = out + (size_t)row * (size_t)series_len;


    if (base >= series_len) {
        for (int i = 0; i < series_len; ++i) dst[i] = __int_as_float(0x7fffffff);
        return;
    }


    double sum_close_f = 0.0, sum_close_s = 0.0;
    double sum_vol_f = 0.0, sum_vol_s = 0.0;
    double sum_cxv_f = 0.0, sum_cxv_s = 0.0;
    const double inv_fast = 1.0 / (double)fast;
    const double inv_slow = 1.0 / (double)slow;


    float ring_vpc[AVSL_MAX_WIN];
    float ring_vpr[AVSL_MAX_WIN];
    #pragma unroll
    for (int k = 0; k < AVSL_MAX_WIN; ++k) { ring_vpc[k] = 0.0f; ring_vpr[k] = 1.0f; }
    int ring_pos = 0;

    float pre_ring_local[AVSL_MAX_PRE_RING];
    int pre_pos = 0; int pre_cnt = 0;
    float pre_sum = 0.0f;

    for (int i = 0; i < series_len; ++i) {
        if (i >= first_valid) {
            const double c = (double)close[i];
            const double v = (double)volume[i];
            const double cv = c * v;
            sum_close_f += c; sum_vol_f += v; sum_cxv_f += cv;
            sum_close_s += c; sum_vol_s += v; sum_cxv_s += cv;

            if (i >= first_valid + fast) {
                const int k = i - fast;
                const float c_old = close[k];
                const float v_old = volume[k];
                sum_close_f -= c_old; sum_vol_f -= v_old; sum_cxv_f -= c_old * v_old;
            }
            if (i >= first_valid + slow) {
                const int k = i - slow;
                const float c_old = close[k];
                const float v_old = volume[k];
                sum_close_s -= c_old; sum_vol_s -= v_old; sum_cxv_s -= c_old * v_old;
            }
        }

        if (i >= base) {
            const double sma_f_d = sum_close_f * inv_fast;
            const double sma_s_d = sum_close_s * inv_slow;
            const double vwma_f_d = (sum_vol_f != 0.0) ? (sum_cxv_f / sum_vol_f) : sma_f_d;
            const double vwma_s_d = (sum_vol_s != 0.0) ? (sum_cxv_s / sum_vol_s) : sma_s_d;
            const double vpc_d = vwma_s_d - sma_s_d;
            const double vpr_d = (sma_f_d != 0.0) ? (vwma_f_d / sma_f_d) : 1.0;
            const double vol_f_d = sum_vol_f * inv_fast;
            const double vol_s_d = sum_vol_s * inv_slow;
            const double vm_d = (vol_s_d != 0.0) ? (vol_f_d / vol_s_d) : 1.0;
            const double vpci_d = vpc_d * vpr_d * vm_d;

            const float vpc = (float)vpc_d;
            const float vpr = (float)vpr_d;
            const float vm = (float)vm_d;
            const float vpci = (float)vpci_d;


            float t = (vpc < 0.0f) ? fabsf(vpci - 3.0f) : (vpci + 3.0f);

            float r = (t >= 0.0f) ? floorf(t + 0.5f) : ceilf(t - 0.5f);
            int len_v = (int)r;
            if (len_v < 1) len_v = 1;
            if (len_v > AVSL_MAX_WIN) len_v = AVSL_MAX_WIN;

            ring_vpc[ring_pos] = vpc;
            ring_vpr[ring_pos] = vpr;
            ring_pos += 1; if (ring_pos == AVSL_MAX_WIN) ring_pos = 0;

            const int take = (len_v < i + 1) ? len_v : (i + 1);
            const int hist_n = ((i - base + 1) < take) ? (i - base + 1) : take;
            const int pref_n = take - hist_n;

            float acc = 0.0f;
            if (hist_n > 0) {
                int rp = (ring_pos == 0) ? (AVSL_MAX_WIN - 1) : (ring_pos - 1);
                for (int j = 0; j < hist_n; ++j) {
                    const int idx_r = rp; rp = (rp == 0) ? (AVSL_MAX_WIN - 1) : (rp - 1);
                    const float adj = avsl_adj(ring_vpc[idx_r]);
                    const float r = ring_vpr[idx_r];
                    if (adj != 0.0f && r != 0.0f) {
                        acc += low[i - j] / (adj * r);
                    }
                }
            }
            if (pref_n > 0) {
                const int start_idx = i + 1 - (hist_n + pref_n);
                const int end_idx_excl = i + 1 - hist_n;
                float s = 0.0f;
                for (int k = start_idx; k < end_idx_excl; ++k) s += low[k];
                acc += s;
            }

            const float price_v = (acc / (float)len_v) * 0.01f;
            const float dev = (mult * vpci) * vm;
            const float pre_i = (low[i] - price_v) + dev;

            if (slow <= AVSL_MAX_PRE_RING) {
                if (pre_cnt < slow) {
                    pre_ring_local[pre_pos] = pre_i;
                    pre_sum += pre_i;
                    pre_pos += 1; if (pre_pos == slow) pre_pos = 0;
                    pre_cnt += 1;
                } else {
                    pre_sum -= pre_ring_local[pre_pos];
                    pre_ring_local[pre_pos] = pre_i;
                    pre_sum += pre_i;
                    pre_pos += 1; if (pre_pos == slow) pre_pos = 0;
                }
                if (i >= warmup2) dst[i] = pre_sum * (float)inv_slow;
            } else {


                if (i >= warmup2) {
                    float s = 0.0f;
                    for (int k = i - slow + 1; k <= i; ++k) {


                        s += pre_i;
                    }
                    dst[i] = s * (float)inv_slow;
                }
            }
        }
    }


    const int up = (warmup2 < series_len) ? warmup2 : series_len;
    for (int i = 0; i < up; ++i) dst[i] = __int_as_float(0x7fffffff);
}


extern "C" __global__ void avsl_many_series_one_param_f32(
    const float* __restrict__ close_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ volume_tm,
    const int* __restrict__ first_valids,
    const int cols,
    const int rows,
    const int fast,
    const int slow,
    const float multiplier,
    float* __restrict__ out_tm)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;

    const int first_valid = first_valids[col];
    const int base = first_valid + max(1, slow) - 1;
    const int warmup2 = base + max(1, slow) - 1;


    float sum_close_f = 0.0f, sum_close_s = 0.0f;
    float sum_vol_f = 0.0f, sum_vol_s = 0.0f;
    float sum_cxv_f = 0.0f, sum_cxv_s = 0.0f;
    const int f = max(1, fast);
    const int s = max(1, slow);
    const float inv_fast = 1.0f / (float)f;
    const float inv_slow = 1.0f / (float)s;

    float ring_vpc[AVSL_MAX_WIN];
    float ring_vpr[AVSL_MAX_WIN];
    #pragma unroll
    for (int k = 0; k < AVSL_MAX_WIN; ++k) { ring_vpc[k] = 0.0f; ring_vpr[k] = 1.0f; }
    int ring_pos = 0;

    float pre_ring_local[AVSL_MAX_PRE_RING];
    int pre_pos = 0; int pre_cnt = 0;
    float pre_sum = 0.0f;

    for (int i = 0; i < rows; ++i) {
        const int idx = i * cols + col;
        if (i >= first_valid) {
            const float c = close_tm[idx];
            const float v = volume_tm[idx];
            const float cv = c * v;
            sum_close_f += c; sum_vol_f += v; sum_cxv_f += cv;
            sum_close_s += c; sum_vol_s += v; sum_cxv_s += cv;

            if (i >= first_valid + f) {
                const int k = (i - f) * cols + col;
                const float c_old = close_tm[k];
                const float v_old = volume_tm[k];
                sum_close_f -= c_old; sum_vol_f -= v_old; sum_cxv_f -= c_old * v_old;
            }
            if (i >= first_valid + s) {
                const int k = (i - s) * cols + col;
                const float c_old = close_tm[k];
                const float v_old = volume_tm[k];
                sum_close_s -= c_old; sum_vol_s -= v_old; sum_cxv_s -= c_old * v_old;
            }
        }

        if (i >= base) {
            const float sma_f = sum_close_f * inv_fast;
            const float sma_s = sum_close_s * inv_slow;
            const float vwma_f = (sum_vol_f != 0.0f) ? (sum_cxv_f / sum_vol_f) : sma_f;
            const float vwma_s = (sum_vol_s != 0.0f) ? (sum_cxv_s / sum_vol_s) : sma_s;
            const float vpc = vwma_s - sma_s;
            const float vpr = (sma_f != 0.0f) ? (vwma_f / sma_f) : 1.0f;
            const float vol_f = sum_vol_f * inv_fast;
            const float vol_s = sum_vol_s * inv_slow;
            const float vm = (vol_s != 0.0f) ? (vol_f / vol_s) : 1.0f;
            const float vpci = vpc * vpr * vm;

            float t = (vpc < 0.0f) ? fabsf(vpci - 3.0f) : (vpci + 3.0f);
            float r = (t >= 0.0f) ? floorf(t + 0.5f) : ceilf(t - 0.5f);
            int len_v = (int)r;
            if (len_v < 1) len_v = 1;
            if (len_v > AVSL_MAX_WIN) len_v = AVSL_MAX_WIN;

            ring_vpc[ring_pos] = vpc; ring_vpr[ring_pos] = vpr;
            ring_pos += 1; if (ring_pos == AVSL_MAX_WIN) ring_pos = 0;

            const int take = (len_v < i + 1) ? len_v : (i + 1);
            const int hist_n = ((i - base + 1) < take) ? (i - base + 1) : take;
            const int pref_n = take - hist_n;
            float acc = 0.0f;
            if (hist_n > 0) {
                int rp = (ring_pos == 0) ? (AVSL_MAX_WIN - 1) : (ring_pos - 1);
                for (int j = 0; j < hist_n; ++j) {
                    const int idx_r = rp; rp = (rp == 0) ? (AVSL_MAX_WIN - 1) : (rp - 1);
                    const float adj = avsl_adj(ring_vpc[idx_r]);
                    const float r = ring_vpr[idx_r];
                    if (adj != 0.0f && r != 0.0f) {
                        const int idl = (i - j) * cols + col;
                        acc += low_tm[idl] / (adj * r);
                    }
                }
            }
            if (pref_n > 0) {
                const int start_i = i + 1 - (hist_n + pref_n);
                const int end_i = i + 1 - hist_n;
                float ssum = 0.0f;
                for (int k = start_i; k < end_i; ++k) {
                    ssum += low_tm[k * cols + col];
                }
                acc += ssum;
            }

            const float price_v = (acc / (float)len_v) * 0.01f;
            const float dev = (multiplier * vpci) * vm;
            const float pre_i = (low_tm[idx] - price_v) + dev;

            if (slow <= AVSL_MAX_PRE_RING) {
                if (pre_cnt < s) {
                    pre_ring_local[pre_pos] = pre_i;
                    pre_sum += pre_i;
                    pre_pos += 1; if (pre_pos == s) pre_pos = 0; pre_cnt += 1;
                } else {
                    pre_sum -= pre_ring_local[pre_pos];
                    pre_ring_local[pre_pos] = pre_i;
                    pre_sum += pre_i;
                    pre_pos += 1; if (pre_pos == s) pre_pos = 0;
                }
                if (i >= warmup2) out_tm[idx] = pre_sum * inv_slow;
            } else {
                if (i >= warmup2) {

                    float ssum = 0.0f;
                    for (int k = i - s + 1; k <= i; ++k) ssum += pre_i;
                    out_tm[idx] = ssum * inv_slow;
                }
            }
        }
    }


    const int up = (warmup2 < rows) ? warmup2 : rows;
    for (int i = 0; i < up; ++i) {
        const int idx = i * cols + col;
        out_tm[idx] = __int_as_float(0x7fffffff);
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE  --  closer 3, round 3
 *
 * Both existing entry points in this file are f32. This section adds the f64
 * lane entry point beside them; the f32 wrapper is untouched, and listing this
 * file in build.rs opts the WHOLE translation unit out of `--use_fast_math`.
 *
 * CPU REFERENCE: src/indicators/avsl.rs `avsl_scalar` (:505-716), with
 *   `avsl_prepare` (:339-425) for the warmup.
 *   Batch dispatcher: cpu_batch.rs:14118 -- `expect_value_output` means the
 *   only output is "value", which is the single `values` series.
 *
 * INPUT: (close, low, volume). The CPU batch calls `extract_hlcv_input`
 *   (:14123) and DISCARDS high (`let (_high, low, close, volume)`), and
 *   `avsl_scalar` never reads it. Registered as F64InputKind::Hlcv so the
 *   launch arm hands over four pointers in (high, low, close, volume) order;
 *   `high` is bound and left unread here, exactly as the CPU does.
 *
 * FIRST-VALID IGNORED, AND DERIVED HERE: `first_valid_max3` (:272-278) is the
 *   MAX of THREE INDEPENDENT first-non-NaN scans -- over close, low and volume
 *   separately. No declared `F64FirstValidRule` expresses that: it is not
 *   "the first index at which all three are non-NaN" (which is later whenever
 *   one series has a hole after another has started), and it is not any of the
 *   high/low pair rules. Rather than add a variant one indicator would use,
 *   the kernel derives it and declares the caller's value unused -- keeping
 *   both halves of one rule in one place.
 *
 * PERIOD-INVARIANT: the CPU batch reads `fast_period`, `slow_period` and
 *   `multiplier` (cpu_batch.rs:14126-14128) and never `period`. All three are
 *   pinned at the CPU defaults (12 / 26 / 2.0), so every row of a sweep is
 *   byte-identical.
 *
 * SHAPE: ONE THREAD PER COLUMN, bars ascending. Six sliding sums are
 *   maintained by subtract-then-add, a 200-deep history of (vpc, vpr) is
 *   replayed BACKWARDS at every bar, and a final sliding mean carries the
 *   result. The backward replay reads bars the current thread has already
 *   walked, so it is per-column and nothing else.
 *
 * ARITHMETIC taken verbatim:
 *   * the six sums add the new bar FIRST and subtract the departing one after
 *     (:566-588), which is the CPU's order and not the reverse.
 *   * `vwma = sum_cxv / sum_vol` falls back to the plain SMA when the volume
 *     sum is EXACTLY zero (:594, :599) -- an exact test, not a tolerance.
 *   * `len_v` (:612-625) rounds `(vpci - 3).abs()` or `(vpci + 3)` with Rust
 *     `f64::round`, which is round-half-AWAY-from-zero -- C `round()`, NOT
 *     `rint`/`nearbyint`, which round half to even and differ on every exact
 *     .5.
 *   * the backward accumulation walks the ring from the most recent entry
 *     (:641-660) and adds `low[i-j] / (adj * r)` one bar at a time, in that
 *     order.
 *   * the prefix sum (:663-681) is chunked FOUR WIDE with the association
 *     `s += a + b + c + d`, i.e. `s + (((a+b)+c)+d)`, then a scalar tail. That
 *     association is load-bearing and is reproduced exactly.
 *   * `dev = multiplier.mul_add(vpci, 0.0) * vm` (:685) is a fused
 *     multiply-add with a zero addend -- ONE rounding -- so `fma` is used, not
 *     a bare product, even though the two agree here.
 *   * `pre_i = (low[i] - price_v) + dev` (:686) -- the subtraction is
 *     parenthesised first.
 *
 * EPSILON: there is none. Every CPU guard is an exact `!= 0.0` test. The
 *   MAX_WIN cap of 200 (:545) is a HISTORY DEPTH, not a tolerance.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* cpu_batch.rs:14126-14128 */
#define NEO_AVSL_FAST_PERIOD 12
#define NEO_AVSL_SLOW_PERIOD 26
#define NEO_AVSL_MULTIPLIER  2.0
/* avsl.rs:545 */
#define NEO_AVSL_MAX_WIN 200

extern "C" __global__
void avsl_neo_batch_f64(const double* __restrict__ high,
                        const double* __restrict__ low,
                        const double* __restrict__ close,
                        const double* __restrict__ volume,
                        int n,
                        const int* __restrict__ periods,
                        int n_combos,
                        int first_valid,
                        double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)high;        /* bound and unread -- see header */
    (void)periods;     /* period-invariant -- see header */
    (void)first_valid; /* derived here -- see header */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int fast_period = NEO_AVSL_FAST_PERIOD;
    const int slow_period = NEO_AVSL_SLOW_PERIOD;
    const double multiplier = NEO_AVSL_MULTIPLIER;

    /* avsl_prepare (:385-396) */
    if (fast_period > n || slow_period > n) return;

    /* first_valid_max3 (:272) -- three INDEPENDENT scans, then the max. */
    int fc = -1, fl = -1, fv = -1;
    for (int i = 0; i < n; ++i) { if (!isnan(close[i]))  { fc = i; break; } }
    for (int i = 0; i < n; ++i) { if (!isnan(low[i]))    { fl = i; break; } }
    for (int i = 0; i < n; ++i) { if (!isnan(volume[i])) { fv = i; break; } }
    if (fc < 0 || fl < 0 || fv < 0) return;             /* AllValuesNaN */
    int first_val = fc; if (fl > first_val) first_val = fl; if (fv > first_val) first_val = fv;
    if (n - first_val < slow_period) return;            /* NotEnoughValidData */

    const int base    = first_val + slow_period - 1;
    const int warmup2 = base + slow_period - 1;
    if (base >= n) return;                              /* :524 -- all NaN */

    const double inv_fast = 1.0 / (double)fast_period;
    const double inv_slow = 1.0 / (double)slow_period;

    double sum_close_f = 0.0, sum_close_s = 0.0;
    double sum_vol_f   = 0.0, sum_vol_s   = 0.0;
    double sum_cxv_f   = 0.0, sum_cxv_s   = 0.0;

    double ring_vpc[NEO_AVSL_MAX_WIN];
    double ring_vpr[NEO_AVSL_MAX_WIN];
    for (int k = 0; k < NEO_AVSL_MAX_WIN; ++k) { ring_vpc[k] = 0.0; ring_vpr[k] = 1.0; }
    int ring_pos = 0;

    double pre_ring[NEO_AVSL_SLOW_PERIOD];
    for (int k = 0; k < slow_period; ++k) pre_ring[k] = 0.0;
    int    pre_pos = 0, pre_cnt = 0;
    double pre_sum = 0.0;

    for (int i = 0; i < n; ++i) {
        if (i >= first_val) {
            const double c  = close[i];
            const double v  = volume[i];
            const double cv = c * v;

            sum_close_f += c; sum_vol_f += v; sum_cxv_f += cv;
            sum_close_s += c; sum_vol_s += v; sum_cxv_s += cv;

            if (i + 1 > fast_period + first_val) {
                const int k = i + 1 - fast_period - 1;
                const double c_old = close[k];
                const double v_old = volume[k];
                sum_close_f -= c_old; sum_vol_f -= v_old; sum_cxv_f -= c_old * v_old;
            }
            if (i + 1 > slow_period + first_val) {
                const int k = i + 1 - slow_period - 1;
                const double c_old = close[k];
                const double v_old = volume[k];
                sum_close_s -= c_old; sum_vol_s -= v_old; sum_cxv_s -= c_old * v_old;
            }
        }

        if (i < base) continue;

        const double sma_f  = sum_close_f * inv_fast;
        const double sma_s  = sum_close_s * inv_slow;
        const double vwma_f = (sum_vol_f != 0.0) ? (sum_cxv_f / sum_vol_f) : sma_f;
        const double vwma_s = (sum_vol_s != 0.0) ? (sum_cxv_s / sum_vol_s) : sma_s;

        const double vpc = vwma_s - sma_s;
        const double vpr = (sma_f != 0.0) ? (vwma_f / sma_f) : 1.0;
        const double vol_f = sum_vol_f * inv_fast;
        const double vol_s = sum_vol_s * inv_slow;
        const double vm = (vol_s != 0.0) ? (vol_f / vol_s) : 1.0;
        const double vpci = vpc * vpr * vm;

        /* :612 -- Rust f64::round is round-half-away-from-zero == C round(). */
        double t = (vpc < 0.0) ? round(fabs(vpci - 3.0)) : round(vpci + 3.0);
        double m = (t < 1.0) ? 1.0 : t;
        if (m > (double)NEO_AVSL_MAX_WIN) m = (double)NEO_AVSL_MAX_WIN;
        const int len_v = (int)m;

        ring_vpc[ring_pos] = vpc;
        ring_vpr[ring_pos] = vpr;
        ring_pos += 1; if (ring_pos == NEO_AVSL_MAX_WIN) ring_pos = 0;

        const int take   = (len_v < (i + 1)) ? len_v : (i + 1);
        const int avail  = i - base + 1;
        const int hist_n = (avail < take) ? avail : take;
        const int pref_n = take - hist_n;

        double acc = 0.0;

        if (hist_n > 0) {
            int rp = (ring_pos == 0) ? (NEO_AVSL_MAX_WIN - 1) : (ring_pos - 1);
            for (int j = 0; j < hist_n; ++j) {
                const int idx_r = rp;
                rp = (rp == 0) ? (NEO_AVSL_MAX_WIN - 1) : (rp - 1);
                const double x = ring_vpc[idx_r];
                double adj;
                if (x > -1.0 && x < 0.0)      adj = -1.0;
                else if (x >= 0.0 && x < 1.0) adj =  1.0;
                else                          adj =  x;
                const double r = ring_vpr[idx_r];
                if (adj != 0.0 && r != 0.0) {
                    acc += low[i - j] / (adj * r);
                }
            }
        }

        if (pref_n > 0) {
            const int start_idx    = i + 1 - (hist_n + pref_n);
            const int end_idx_excl = i + 1 - hist_n;
            double s = 0.0;
            int k = start_idx;
            /* FOUR WIDE, association `s + (((a+b)+c)+d)` -- see header. */
            while (k + 4 <= end_idx_excl) {
                const double a = low[k];
                const double b = low[k + 1];
                const double c = low[k + 2];
                const double d = low[k + 3];
                s += a + b + c + d;
                k += 4;
            }
            while (k < end_idx_excl) { s += low[k]; k += 1; }
            acc += s;
        }

        const double price_v = (acc / (double)len_v) * 0.01;
        const double dev     = fma(multiplier, vpci, 0.0) * vm;
        const double pre_i   = (low[i] - price_v) + dev;

        pre_sum += pre_i;
        if (pre_cnt < slow_period) {
            pre_ring[pre_pos] = pre_i;
            pre_pos += 1; if (pre_pos == slow_period) pre_pos = 0;
            pre_cnt += 1;
        } else {
            pre_sum -= pre_ring[pre_pos];
            pre_ring[pre_pos] = pre_i;
            pre_pos += 1; if (pre_pos == slow_period) pre_pos = 0;
        }

        if (i >= warmup2) {
            o[i] = pre_sum * inv_slow;
        }
    }
}
