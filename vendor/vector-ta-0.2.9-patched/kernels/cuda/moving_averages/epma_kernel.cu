#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

namespace {


    __device__ __forceinline__ constexpr int kTile() { return 8; }


    __device__ __forceinline__ double epma_weight_sum(int p1, int offset) {
        return 0.5 * static_cast<double>(p1) *
               (static_cast<double>(p1) + 3.0 - 2.0 * static_cast<double>(offset));
    }
}

extern "C" __global__
void epma_batch_f32(const float* __restrict__ prices,
                    const int*   __restrict__ periods,
                    const int*   __restrict__ offsets,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int offset = offsets[combo];
    const int p1     = period - 1;
    if (p1 <= 0) return;

    const double bias = 2.0 - static_cast<double>(offset);
    const double wsum = epma_weight_sum(p1, offset);
    const double inv_wsum = (wsum == 0.0) ? 0.0 : (1.0 / wsum);


    const int warm = first_valid + period + offset + 1;

    const int base_out = combo * series_len;

    const int TILE = kTile();
    const int tile_span = blockDim.x * TILE;


    for (int base = blockIdx.x * tile_span; base < series_len; base += gridDim.x * tile_span) {
        int t_start = base + threadIdx.x * TILE;
        if (t_start >= series_len) continue;
        int t_end = t_start + TILE;
        if (t_end > series_len) t_end = series_len;


        const int pre_end = (warm < t_end ? (warm) : t_end);
        for (int t = t_start; t < pre_end; ++t) {
            out[base_out + t] = NAN;
        }
        if (t_end <= warm) continue;


        int t0 = (t_start < warm ? warm : t_start);


        int a = t0 + 1 - p1;
        int b = t0;


        double sumP  = 0.0;
        double sumIP = 0.0;

        #pragma unroll 4
        for (int k = 0; k < p1; ++k) {
            int idx = a + k;
            double pr = static_cast<double>(prices[idx]);
            sumP  += pr;
            sumIP  = fma(static_cast<double>(idx), pr, sumIP);
        }


        out[base_out + t0] = static_cast<float>((sumIP + (bias - static_cast<double>(a)) * sumP) * inv_wsum);


        for (int t = t0 + 1; t < t_end; ++t) {
            int old_a = a;
            a += 1;
            b += 1;

            double leaving  = static_cast<double>(prices[old_a]);
            double entering = static_cast<double>(prices[b]);

            sumP += entering - leaving;
            sumIP = fma(static_cast<double>(b),     entering, sumIP);
            sumIP = fma(-static_cast<double>(old_a), leaving,  sumIP);

            out[base_out + t] = static_cast<float>((sumIP + (bias - static_cast<double>(a)) * sumP) * inv_wsum);
        }
    }
}

extern "C" __global__
void epma_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    int period,
    int offset,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm)
{
    const int p1 = period - 1;
    if (p1 <= 0) return;

    const int s = blockIdx.y;
    if (s >= num_series) return;

    const int warm = first_valids[s] + period + offset + 1;

    const double bias = 2.0 - static_cast<double>(offset);
    const double wsum = epma_weight_sum(p1, offset);
    const double inv_wsum = (wsum == 0.0) ? 0.0 : (1.0 / wsum);

    const int TILE = kTile();
    const int tile_span = blockDim.x * TILE;


    auto load_tm = [&](int t) -> double {
        long long in_idx = static_cast<long long>(t) * static_cast<long long>(num_series) + static_cast<long long>(s);
        return static_cast<double>(prices_tm[in_idx]);
    };

    for (int base = blockIdx.x * tile_span; base < series_len; base += gridDim.x * tile_span) {
        int t_start = base + threadIdx.x * TILE;
        if (t_start >= series_len) continue;
        int t_end = t_start + TILE;
        if (t_end > series_len) t_end = series_len;


        const int pre_end = (warm < t_end ? (warm) : t_end);
        for (int t = t_start; t < pre_end; ++t) {
            long long out_idx = static_cast<long long>(t) * static_cast<long long>(num_series) + static_cast<long long>(s);
            out_tm[out_idx] = NAN;
        }
        if (t_end <= warm) continue;

        int t0 = (t_start < warm ? warm : t_start);

        int a = t0 + 1 - p1;
        int b = t0;

        double sumP  = 0.0;
        double sumIP = 0.0;

        #pragma unroll 4
        for (int k = 0; k < p1; ++k) {
            int idx = a + k;
            double pr = load_tm(idx);
            sumP  += pr;
            sumIP  = fma(static_cast<double>(idx), pr, sumIP);
        }

        {
            long long out_idx = static_cast<long long>(t0) * static_cast<long long>(num_series) + static_cast<long long>(s);
            out_tm[out_idx] = static_cast<float>((sumIP + (bias - static_cast<double>(a)) * sumP) * inv_wsum);
        }

        for (int t = t0 + 1; t < t_end; ++t) {
            int old_a = a;
            a += 1;
            b += 1;

            double leaving  = load_tm(old_a);
            double entering = load_tm(b);

            sumP += entering - leaving;
            sumIP = fma(static_cast<double>(b),     entering, sumIP);
            sumIP = fma(-static_cast<double>(old_a), leaving,  sumIP);

            long long out_idx = static_cast<long long>(t) * static_cast<long long>(num_series) + static_cast<long long>(s);
            out_tm[out_idx] = static_cast<float>((sumIP + (bias - static_cast<double>(a)) * sumP) * inv_wsum);
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

extern "C" __global__
void epma_batch_f64(const double* __restrict__ prices,
                    const int*   __restrict__ periods,
                    const int*   __restrict__ offsets,
                    int series_len,
                    int n_combos,
                    int first_valid,
                    double* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos) return;

    const int period = periods[combo];
    const int offset = offsets[combo];
    const int p1     = period - 1;
    if (p1 <= 0) return;

    const double bias = 2.0 - static_cast<double>(offset);
    const double wsum = epma_weight_sum(p1, offset);
    const double inv_wsum = (wsum == 0.0) ? 0.0 : (1.0 / wsum);


    const int warm = first_valid + period + offset + 1;

    const int base_out = combo * series_len;

    const int TILE = kTile();
    const int tile_span = blockDim.x * TILE;


    for (int base = blockIdx.x * tile_span; base < series_len; base += gridDim.x * tile_span) {
        int t_start = base + threadIdx.x * TILE;
        if (t_start >= series_len) continue;
        int t_end = t_start + TILE;
        if (t_end > series_len) t_end = series_len;


        const int pre_end = (warm < t_end ? (warm) : t_end);
        for (int t = t_start; t < pre_end; ++t) {
            out[base_out + t] = NAN;
        }
        if (t_end <= warm) continue;


        int t0 = (t_start < warm ? warm : t_start);


        int a = t0 + 1 - p1;
        int b = t0;


        double sumP  = 0.0;
        double sumIP = 0.0;

        #pragma unroll 4
        for (int k = 0; k < p1; ++k) {
            int idx = a + k;
            double pr = static_cast<double>(prices[idx]);
            sumP  += pr;
            sumIP  = fma(static_cast<double>(idx), pr, sumIP);
        }


        out[base_out + t0] = static_cast<double>((sumIP + (bias - static_cast<double>(a)) * sumP) * inv_wsum);


        for (int t = t0 + 1; t < t_end; ++t) {
            int old_a = a;
            a += 1;
            b += 1;

            double leaving  = static_cast<double>(prices[old_a]);
            double entering = static_cast<double>(prices[b]);

            sumP += entering - leaving;
            sumIP = fma(static_cast<double>(b),     entering, sumIP);
            sumIP = fma(-static_cast<double>(old_a), leaving,  sumIP);

            out[base_out + t] = static_cast<double>((sumIP + (bias - static_cast<double>(a)) * sumP) * inv_wsum);
        }
    }
}
extern "C" __global__
void epma_many_series_one_param_time_major_f64(
    const double* __restrict__ prices_tm,
    int period,
    int offset,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    double* __restrict__ out_tm)
{
    const int p1 = period - 1;
    if (p1 <= 0) return;

    const int s = blockIdx.y;
    if (s >= num_series) return;

    const int warm = first_valids[s] + period + offset + 1;

    const double bias = 2.0 - static_cast<double>(offset);
    const double wsum = epma_weight_sum(p1, offset);
    const double inv_wsum = (wsum == 0.0) ? 0.0 : (1.0 / wsum);

    const int TILE = kTile();
    const int tile_span = blockDim.x * TILE;


    auto load_tm = [&](int t) -> double {
        long long in_idx = static_cast<long long>(t) * static_cast<long long>(num_series) + static_cast<long long>(s);
        return static_cast<double>(prices_tm[in_idx]);
    };

    for (int base = blockIdx.x * tile_span; base < series_len; base += gridDim.x * tile_span) {
        int t_start = base + threadIdx.x * TILE;
        if (t_start >= series_len) continue;
        int t_end = t_start + TILE;
        if (t_end > series_len) t_end = series_len;


        const int pre_end = (warm < t_end ? (warm) : t_end);
        for (int t = t_start; t < pre_end; ++t) {
            long long out_idx = static_cast<long long>(t) * static_cast<long long>(num_series) + static_cast<long long>(s);
            out_tm[out_idx] = NAN;
        }
        if (t_end <= warm) continue;

        int t0 = (t_start < warm ? warm : t_start);

        int a = t0 + 1 - p1;
        int b = t0;

        double sumP  = 0.0;
        double sumIP = 0.0;

        #pragma unroll 4
        for (int k = 0; k < p1; ++k) {
            int idx = a + k;
            double pr = load_tm(idx);
            sumP  += pr;
            sumIP  = fma(static_cast<double>(idx), pr, sumIP);
        }

        {
            long long out_idx = static_cast<long long>(t0) * static_cast<long long>(num_series) + static_cast<long long>(s);
            out_tm[out_idx] = static_cast<double>((sumIP + (bias - static_cast<double>(a)) * sumP) * inv_wsum);
        }

        for (int t = t0 + 1; t < t_end; ++t) {
            int old_a = a;
            a += 1;
            b += 1;

            double leaving  = load_tm(old_a);
            double entering = load_tm(b);

            sumP += entering - leaving;
            sumIP = fma(static_cast<double>(b),     entering, sumIP);
            sumIP = fma(-static_cast<double>(old_a), leaving,  sumIP);

            long long out_idx = static_cast<long long>(t) * static_cast<long long>(num_series) + static_cast<long long>(s);
            out_tm[out_idx] = static_cast<double>((sumIP + (bias - static_cast<double>(a)) * sumP) * inv_wsum);
        }
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — epma
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/moving_averages/epma.rs:353 `epma_scalar`, which
 *   is what `epma_row_scalar` (:1188) — the row function the ScalarBatch lane
 *   takes (:1146-1157) — calls. `ma_batch.rs:1881` routes the id `epma` here
 *   with `offset` from params (default 4) and `period` from the SWEPT axis, so
 *   this is one of the three PERIOD-SWEPT indicators in this closer batch.
 *
 * THE FAST PATH IS NOT AN OPTIMISATION, IT IS A DIFFERENT NUMBER.
 *   `epma_scalar` (:360) diverts to `epma_scalar_default_stream` (:404) when
 *   period == 11 AND offset == 4 AND len >= 1024 AND there is no NaN at or
 *   after first_valid. That stream is KAHAN-COMPENSATED and slides its sums;
 *   the general path re-sums the window from scratch with plain fma. The two
 *   do not agree bit for bit. Both are reproduced here, gated on exactly the
 *   same four conditions, because a sweep that happens to include period 11
 *   would otherwise disagree with the CPU on that row alone — the hardest
 *   class of bug to find.
 *
 * Accumulation order, general path (:377-399): the CPU unrolls by four but
 *   every term is `sum = data[..].mul_add(w, sum)` into ONE accumulator in
 *   ascending i, so a plain loop of fma is the SAME association. Verified
 *   term by term rather than assumed.
 *
 * Warmup: `first_valid + period + offset + 1` (:377), matching `warm[row]`
 *   (:1069). Not `first + period - 1`.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* EpmaBatchRange offset axis default (ma_batch.rs:1882). */
#define NEO_EPMA_OFFSET 4

/* kahan_add, as the CPU helper of the same name. */
__device__ __forceinline__ void neo_epma_kahan(double* sum, double* c, double value)
{
    const double y = value - *c;
    const double t = *sum + y;
    *c = (t - *sum) - y;
    *sum = t;
}

extern "C" __global__
void epma_neo_batch_f64(const double* __restrict__ data,
                        int n,
                        const int* __restrict__ periods,
                        int n_combos,
                        int first_valid,
                        double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int period = periods[combo];
    const int offset = NEO_EPMA_OFFSET;
    if (period < 2 || offset >= period) return;      /* the CPU errors here */
    if (first_valid < 0 || first_valid >= n) return;

    /* epma_scalar_default_stream gate (:360-367) */
    if (period == 11 && offset == 4 && n >= 1024) {
        bool has_nan = false;
        for (int i = first_valid; i < n; ++i) { if (isnan(data[i])) { has_nan = true; break; } }
        if (!has_nan) {
            const int PERIOD = 11, OFF = 4, P1 = 10;
            const double C0 = -2.0, INV_WSUM = 0.04;
            double buffer[11];
            for (int i = 0; i < PERIOD; ++i) buffer[i] = 0.0;
            int head = 0; long long seen = 0; int included = 0;
            double sum = 0.0, sum_c = 0.0, ramp = 0.0, ramp_c = 0.0;

            for (int idx = first_valid; idx < n; ++idx) {
                const double value = data[idx];
                const int idx_out = (head + 1) % PERIOD;
                const double x_out = (included == P1) ? buffer[idx_out] : 0.0;

                buffer[head] = value;
                head = (head + 1) % PERIOD;
                ++seen;

                if (included < P1) {
                    const double m = (double)included;
                    neo_epma_kahan(&sum, &sum_c, value);
                    neo_epma_kahan(&ramp, &ramp_c, m * value);
                    ++included;
                } else {
                    const double s_old = sum;
                    neo_epma_kahan(&sum, &sum_c, value - x_out);
                    neo_epma_kahan(&ramp, &ramp_c, fma(9.0, value, x_out - s_old));
                }

                if (seen > (long long)(PERIOD + OFF + 1)) {
                    o[idx] = fma(C0, sum, ramp) * INV_WSUM;
                }
            }
            return;
        }
    }

    /* general path (:369-400) */
    const int    p1   = period - 1;
    const double c0   = 2.0 - (double)offset;
    const double p1f  = (double)p1;
    const double wsum = fma(p1f, c0, 0.5 * (p1f - 1.0) * p1f);
    const double inv  = 1.0 / wsum;

    for (int j = first_valid + period + offset + 1; j < n; ++j) {
        const int start = j + 1 - p1;
        double sum = 0.0;
        double wi = c0;
        for (int i = 0; i < p1; ++i) {
            sum = fma(data[start + i], wi, sum);
            wi += 1.0;
        }
        o[j] = sum * inv;
    }
}
