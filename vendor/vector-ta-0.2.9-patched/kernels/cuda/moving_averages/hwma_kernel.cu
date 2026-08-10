#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>


static __device__ __forceinline__ float qnan_f() { return __int_as_float(0x7fffffff); }

static __device__ __forceinline__ int clamp_int(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}


extern "C" __global__
void hwma_batch_f32(const float* __restrict__ prices,
                    const float* __restrict__ nas,
                    const float* __restrict__ nbs,
                    const float* __restrict__ ncs,
                    int first_valid,
                    int series_len,
                    int n_combos,
                    float* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || series_len <= 0) {
        return;
    }

    int first = clamp_int(first_valid, 0, series_len);

    const float na = nas[combo];
    const float nb = nbs[combo];
    const float nc = ncs[combo];

    const int base = combo * series_len;
    const float nan_f = qnan_f();
    for (int t = 0; t < first; ++t) { out[base + t] = nan_f; }
    if (first >= series_len) { return; }


    float f = prices[first];
    float v = 0.0f;
    float a = 0.0f;
    const float dh  = 0.5f;

    for (int t = first; t < series_len; ++t) {
        const float price = prices[t];
        const float s_prev = fmaf(dh, a, (f + v));
        const float f_new = fmaf(na, price, (1.0f - na) * s_prev);
        const float v_new = fmaf(nb, (f_new - f), (1.0f - nb) * (v + a));
        const float a_new = fmaf(nc, (v_new - v), (1.0f - nc) * a);
        const float s_new = fmaf(dh, a_new, (f_new + v_new));
        out[base + t] = s_new;
        f = f_new; v = v_new; a = a_new;
    }
}


extern "C" __global__ __launch_bounds__(256, 2)
void hwma_multi_series_one_param_f32(const float* __restrict__ prices_tm,
                                     float na,
                                     float nb,
                                     float nc,
                                     int num_series,
                                     int series_len,
                                     const int* __restrict__ first_valids,
                                     float* __restrict__ out_tm)
{
    for (int series_idx = blockIdx.x * blockDim.x + threadIdx.x;
         series_idx < num_series;
         series_idx += blockDim.x * gridDim.x)
    {
        if (series_len <= 0) return;

        const int stride = num_series;

        int first = clamp_int(first_valids[series_idx], 0, series_len);


        const float nan_f = qnan_f();
        int idx = series_idx;
        for (int t = 0; t < first; ++t, idx += stride) {
            out_tm[idx] = nan_f;
        }
        if (first >= series_len) continue;


        const double dna = (double)na;
        const double dnb = (double)nb;
        const double dnc = (double)nc;
        const double dh  = 0.5;


        int first_idx = first * stride + series_idx;
        double f = (double)prices_tm[first_idx];
        double v = 0.0;
        double a = 0.0;


        idx = first_idx;


        for (int t = first; t < series_len; ++t, idx += stride) {
            const double price = (double)prices_tm[idx];

            double s_prev = (f + v) + dh * a;

            double nap = dna * price;
            double f_new = fma((1.0 - dna), s_prev, nap);

            double vy    = v + a;
            double nbd   = dnb * (f_new - f);
            double v_new = fma((1.0 - dnb), vy, nbd);

            double ncv   = dnc * (v_new - v);
            double a_new = fma((1.0 - dnc), a, ncv);

            double s_new = (f_new + v_new) + dh * a_new;

            out_tm[idx] = (float)s_new;

            f = f_new;
            v = v_new;
            a = a_new;
        }
    }
}


// ---------------------------------------------------------------------------
// f64 lane.
//
// CPU reference: `moving_averages/hwma.rs::hwma_scalar` (l.489). Defaults
// na = 0.2, nb = 0.1, nc = 0.1 (`hwma.rs:198-208`). PERIOD-INVARIANT: hwma has
// no period parameter, so every row of a sweep is byte-identical.
//
// THE EXISTING f64 TWIN IN THIS FILE IS NOT THE ORACLE. `hwma_multi_series_one_param_f32`
// already computes in double (l.108-119) but with a DIFFERENT association:
// it writes `s_prev = (f + v) + dh * a` (two roundings) where the CPU writes
// `HALF.mul_add(a, f + v)` (one), and `f_new = fma(1.0 - na, s_prev, na*price)`
// which happens to be right while the f32 entry point writes
// `fmaf(na, price, (1.0f - na) * s_prev)` — the SAME two factors but the OTHER
// product rounded. Only the CPU line settles it. The five lines below are the
// CPU's, one for one:
//     s_prev = fma(HALF,     a,            f + v)
//     f_new  = fma(one_m_na, s_prev,       na * x)
//     v_new  = fma(nb,       f_new - f,    one_m_nb * (v + a))
//     a_new  = fma(nc,       v_new - v,    one_m_nc * a)
//     out    = fma(HALF,     a_new,        f_new + v_new)
// Five fused multiply-adds, five roundings from the products that feed them.
//
// Seed (hwma.rs:501-503): f = data[first_valid], v = 0.0, a = 0.0, and the
// walk starts AT first_valid.
//
// This is a serial recurrence, so it is ONE THREAD PER COLUMN walking bars in
// ascending order. There is no parallel-scan reformulation here: the four
// carried scalars make the step non-associative, and even for the linear part
// a scan would change the rounding.
//
// f32 -> f64 audit: pointers/locals widened; `fmaf` x5 -> `fma`; `0.5f`, `0.0f`,
// `1.0f` widened; `qnan_f()` (`__int_as_float(0x7fffffff)`) -> the f64
// quiet-NaN bit pattern; no fast-math intrinsic; no epsilon in this indicator;
// no min/max chain.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ double hwma_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__
void hwma_batch_f64(const double* __restrict__ prices,
                    int n,
                    const int*   __restrict__ periods,
                    int n_combos,
                    int first_valid,
                    double* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || n <= 0) return;
    (void)periods;   // period-invariant, see above

    const double nan_d = hwma_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);

    const int first = (first_valid < 0) ? 0 : first_valid;
    if (first >= n) {
        for (int t = 0; t < n; ++t) row[t] = nan_d;
        return;
    }
    for (int t = 0; t < first; ++t) row[t] = nan_d;

    const double na = 0.2;          // hwma.rs:199
    const double nb = 0.1;          // hwma.rs:203
    const double nc = 0.1;          // hwma.rs:207
    const double HALF = 0.5;
    const double one_m_na = 1.0 - na;
    const double one_m_nb = 1.0 - nb;
    const double one_m_nc = 1.0 - nc;

    double f = prices[first];
    double v = 0.0;
    double a = 0.0;

    for (int t = first; t < n; ++t) {
        const double x0 = prices[t];

        const double s_prev = fma(HALF, a, f + v);
        const double f_new  = fma(one_m_na, s_prev, na * x0);
        const double v_new  = fma(nb, f_new - f, one_m_nb * (v + a));
        const double a_new  = fma(nc, v_new - v, one_m_nc * a);

        row[t] = fma(HALF, a_new, f_new + v_new);

        f = f_new;
        v = v_new;
        a = a_new;
    }
}
