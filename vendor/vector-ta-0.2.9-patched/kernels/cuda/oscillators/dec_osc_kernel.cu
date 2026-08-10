#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

#ifndef LIKELY
#define LIKELY(x)   (__builtin_expect(!!(x), 1))
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif


static __forceinline__ __device__
void hp2_coeffs_f32(float period, float &c, float &two_oma, float &oma_sq) {
    const float p = fmaxf(period, 1.0f);
    float s, co;

    sincospif(1.4142135623730951f / p, &s, &co);
    const float alpha = 1.0f + ((s - 1.0f) / co);
    const float t = 1.0f - 0.5f * alpha;
    c = t * t;
    const float oma = 1.0f - alpha;
    two_oma = 2.0f * oma;
    oma_sq = oma * oma;
}


#ifndef DECOSC_TILE_T
#define DECOSC_TILE_T 2048
#endif


extern "C" __global__ void dec_osc_batch_f32(
    const float* __restrict__ prices,
    const int*   __restrict__ periods,
    const float* __restrict__ ks,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out
){

    const int blocks_needed = (n_combos + blockDim.x - 1) / blockDim.x;
    if (blockIdx.x >= blocks_needed) return;


    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = (combo < n_combos);

    __shared__ float s_prices[DECOSC_TILE_T];


    int    period    = 0;
    float  kf        = 0.0f;
    int    base_idx  = 0;
    if (active) {
        period   = periods[combo];
        kf       = ks[combo];
        base_idx = combo * series_len;


        const int prefix_len = (first_valid + 2 < series_len) ? (first_valid + 2) : series_len;
        for (int i = 0; i < prefix_len; ++i) {
            out[base_idx + i] = CUDART_NAN_F;
        }
    }


    bool valid = false;
    if (active) {
        valid = (period >= 2 && period <= series_len &&
                 first_valid >= 0 && first_valid < series_len &&
                 (series_len - first_valid) >= 2);
    }


    float c1=0, two_oma1=0, oma1_sq=0;
    float c2=0, two_oma2=0, oma2_sq=0;
    float scale = 0.0f;
    if (valid) {
        const float p  = (float)period;
        const float hp = 0.5f * p;
        hp2_coeffs_f32(p,  c1, two_oma1, oma1_sq);
        hp2_coeffs_f32(hp, c2, two_oma2, oma2_sq);
        scale = 100.0f * kf;
    }


    float x2=0.0f, x1=0.0f;
    float hp_prev_2=0.0f, hp_prev_1=0.0f;
    float decosc_prev_2=0.0f, decosc_prev_1=0.0f;

    if (valid) {
        const int i0 = first_valid;
        const int i1 = first_valid + 1;
        x2 = prices[i0];
        x1 = prices[i1];
        hp_prev_2 = x2;
        hp_prev_1 = x1;
        decosc_prev_2 = 0.0f;
        decosc_prev_1 = 0.0f;
    }


    for (int tile_start = first_valid + 2; tile_start < series_len; tile_start += DECOSC_TILE_T) {
        const int tile_end = min(series_len, tile_start + DECOSC_TILE_T);
        const int tile_len = tile_end - tile_start;


        for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
            s_prices[t] = prices[tile_start + t];
        }
        __syncthreads();

        if (valid) {

            for (int t = 0; t < tile_len; ++t) {
                const int i = tile_start + t;
                const float d0 = s_prices[t];

                const float dx  = d0 - 2.0f * x1 + x2;
                const float hp0 = fmaf(c1, dx, fmaf(two_oma1, hp_prev_1, -oma1_sq * hp_prev_2));

                const float dec    = d0 - hp0;
                const float d_dec1 = x1 - hp_prev_1;
                const float d_dec2 = x2 - hp_prev_2;
                const float decdx  = dec - 2.0f * d_dec1 + d_dec2;
                const float osc0   = fmaf(c2, decdx, fmaf(two_oma2, decosc_prev_1, -oma2_sq * decosc_prev_2));

                out[base_idx + i] = scale * (osc0 / d0);


                hp_prev_2      = hp_prev_1;
                hp_prev_1      = hp0;
                decosc_prev_2  = decosc_prev_1;
                decosc_prev_1  = osc0;
                x2 = x1;
                x1 = d0;
            }
        }

        __syncthreads();
    }
}


extern "C" __global__ void dec_osc_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids,
    int num_series,
    int series_len,
    int period,
    float k,
    float* __restrict__ out_tm
) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_series) return;

    const int first = first_valids[s];
    if (UNLIKELY(period < 2 || period > series_len ||
                 first < 0 || first >= series_len ||
                 series_len - first < 2)) {

        for (int t = 0; t < series_len; ++t) {
            out_tm[t * num_series + s] = CUDART_NAN_F;
        }
        return;
    }


    const int prefix_len = first + 2;
    for (int t = 0; t < prefix_len; ++t) {
        out_tm[t * num_series + s] = CUDART_NAN_F;
    }

    float c1, two_oma1, oma1_sq;
    float c2, two_oma2, oma2_sq;
    const float p  = (float)period;
    const float hp = 0.5f * p;
    hp2_coeffs_f32(p,  c1, two_oma1, oma1_sq);
    hp2_coeffs_f32(hp, c2, two_oma2, oma2_sq);

    const float scale = 100.0f * k;

    auto load_tm  = [&](int t) { return prices_tm[t * num_series + s]; };
    auto store_tm = [&](int t, float v) { out_tm[t * num_series + s] = v; };

    const int i0 = first;
    const int i1 = first + 1;

    float x2 = load_tm(i0);
    float x1 = load_tm(i1);
    float hp_prev_2 = x2;
    float hp_prev_1 = x1;
    float decosc_prev_2 = 0.0f;
    float decosc_prev_1 = 0.0f;

    for (int t = first + 2; t < series_len; ++t) {
        const float d0 = load_tm(t);
        const float dx  = d0 - 2.0f * x1 + x2;
        const float hp0 = fmaf(c1, dx, fmaf(two_oma1, hp_prev_1, -oma1_sq * hp_prev_2));

        const float dec    = d0 - hp0;
        const float d_dec1 = x1 - hp_prev_1;
        const float d_dec2 = x2 - hp_prev_2;
        const float decdx  = dec - 2.0f * d_dec1 + d_dec2;
        const float osc0   = fmaf(c2, decdx, fmaf(two_oma2, decosc_prev_1, -oma2_sq * decosc_prev_2));

        store_tm(t, scale * (osc0 / d0));

        hp_prev_2 = hp_prev_1;
        hp_prev_1 = hp0;
        decosc_prev_2 = decosc_prev_1;
        decosc_prev_1 = osc0;
        x2 = x1;
        x1 = d0;
    }
}

// ===========================================================================
// f64 LANE  --  shard S6
//
// THIS SECTION BELONGS IN THIS FILE AND NOWHERE ELSE. It was first written
// into a NEW `kernels/cuda/dec_osc_kernel.cu` at the kernels root, which
// `build.rs` has no `compile_kernel` call for -- build.rs:1364 compiles THIS
// file to `dec_osc_kernel.ptx`. The root file therefore produced no PTX, and
// `F64Kernel::DecOsc` resolved to a module that never contained its symbol:
// a `MissingKernelSymbol` at launch, not at build. The root file is deleted;
// keep the f64 entry point beside the f32 ones it shares a translation unit
// with, which is also what `F64_LANE_SOURCES` (build.rs:224) already covers,
// so this whole TU builds WITHOUT `--use_fast_math`.
//
// CPU reference: `dec_osc_scalar` (src/indicators/dec_osc.rs:352).
// `dec_osc_avx2` (:422) delegates to it verbatim, and `dec_osc_prepare` pins
// `Kernel::Auto -> Kernel::Scalar` (:268-271): one CPU answer, no association
// to settle.
//
// first_valid: `data.iter().position(|x| !x.is_nan())` (:244-246) over the
// single source series (default "close", :117) ->
// `F64FirstValidRule::AllInputsNonNan`.
//
// warm: the CPU writes NaN at `first` and `first + 1` explicitly (:386-387)
// and the emit loop starts at `first + 2` (:398). The swept `periods` value is
// `hp_period` (:120-122); `k` keeps its default of 1.0 (:124-126), so
// `scale = 100.0 * 1.0`.
//
// TWO CASCADED 2-POLE IIR FILTERS -- ONE THREAD PER COLUMN, ascending bars.
// Five scalars are carried across bars (hp_prev_1/2, decosc_prev_1/2,
// dec_prev_1/2 plus the x1/x2 price lags). This is exactly the shape the brief
// calls a serial recurrence: it is NOT bar-parallel and it must NOT be turned
// into a matrix-power scan, because the scan reassociates
// `c*dx + 2*oma*y1 - oma^2*y2` and moves the last bits of every subsequent bar.
//
// THE FILTER COEFFICIENTS ARE COMPUTED ONCE PER THREAD, IN THE CPU'S ORDER.
// :366-382 -- `angle = 2*PI*0.707/p`, `sin_cos`, `alpha = 1 + (sin-1)/cos`,
// `t = 1 - alpha*0.5`, `c = t*t`, `oma = 1-alpha`, `two_oma = oma+oma` (an
// ADDITION, not `2.0*oma`), `oma_sq = oma*oma`. Every one of those is
// reproduced as written: `two_oma = oma + oma` is exact either way, but the
// point of copying the expression is that nothing here is re-derived.
// `sincos()` is CUDA's correctly-rounded pair, matching `f64::sin_cos`.
// PI is the f64 `std::f64::consts::PI`, written to full f64 precision -- NOT
// an f32-width decimal, which is the classic silent constant loss when an f32
// kernel is widened.
//
// f32 -> f64 audit: the f32 lane above uses `fmaxf`. Below there is no
// f32-suffixed function, no f32 literal and no fast-math intrinsic --
// `__fdividef`, `__sinf` and friends are exactly what would destroy the
// coefficient derivation here, because `(sin1 - 1.0) / cos1` is a
// cancellation-sensitive expression whose error is amplified by the recursion.
// This indicator has no epsilon and no min/max chain; the final division by
// `d0` is left unguarded exactly as :408 leaves it.
// ===========================================================================

static __device__ __forceinline__ double dec_osc_qnan_f64() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__
void dec_osc_batch_f64(const double* __restrict__ data,
                       int n,
                       const int* __restrict__ periods,
                       int n_combos,
                       int first_valid,
                       double* __restrict__ out)
{
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos || n <= 0) return;

    const double nan_d = dec_osc_qnan_f64();
    double* __restrict__ row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);

    const int period = periods[combo];
    const int first  = (first_valid < 0) ? 0 : first_valid;

    for (int t = 0; t < n; ++t) row[t] = nan_d;

    // `dec_osc_prepare` rejects period < 3, period > len and len - first < 2.
    if (period < 3 || period > n || first >= n || (n - first) < 2) return;
    if (first + 1 >= n) return;                              // :359

    const double PI_F64 = 3.14159265358979323846264338327950288;
    const double p      = static_cast<double>(period);
    const double half_p = p * 0.5;                           // :364

    const double angle1 = 2.0 * PI_F64 * 0.707 / p;          // :366
    double sin1, cos1;
    sincos(angle1, &sin1, &cos1);                            // :367
    const double alpha1   = 1.0 + ((sin1 - 1.0) / cos1);     // :368
    const double t1       = 1.0 - alpha1 * 0.5;              // :369
    const double c1       = t1 * t1;                         // :370
    const double oma1     = 1.0 - alpha1;                    // :371
    const double two_oma1 = oma1 + oma1;                     // :372
    const double oma1_sq  = oma1 * oma1;                     // :373

    const double angle2 = 2.0 * PI_F64 * 0.707 / half_p;     // :375
    double sin2, cos2;
    sincos(angle2, &sin2, &cos2);                            // :376
    const double alpha2   = 1.0 + ((sin2 - 1.0) / cos2);     // :377
    const double t2       = 1.0 - alpha2 * 0.5;              // :378
    const double c2       = t2 * t2;                         // :379
    const double oma2     = 1.0 - alpha2;                    // :380
    const double two_oma2 = oma2 + oma2;                     // :381
    const double oma2_sq  = oma2 * oma2;                     // :382

    const double scale = 100.0 * 1.0;                        // :384, k defaults to 1.0

    // :389-396
    double x2 = data[first];
    double x1 = data[first + 1];
    double hp_prev_2 = x2;
    double hp_prev_1 = x1;
    double decosc_prev_2 = 0.0;
    double decosc_prev_1 = 0.0;
    double dec_prev_2 = x2 - hp_prev_2;
    double dec_prev_1 = x1 - hp_prev_1;

    for (int i = first + 2; i < n; ++i) {                    // :398-418
        const double d0 = data[i];

        const double dx  = d0 - 2.0 * x1 + x2;
        const double hp0 = c1 * dx + two_oma1 * hp_prev_1 - oma1_sq * hp_prev_2;

        const double dec   = d0 - hp0;
        const double decdx = dec - 2.0 * dec_prev_1 + dec_prev_2;
        const double osc0  = c2 * decdx + two_oma2 * decosc_prev_1 - oma2_sq * decosc_prev_2;

        row[i] = scale * osc0 / d0;

        hp_prev_2 = hp_prev_1;
        hp_prev_1 = hp0;
        decosc_prev_2 = decosc_prev_1;
        decosc_prev_1 = osc0;
        dec_prev_2 = dec_prev_1;
        dec_prev_1 = dec;
        x2 = x1;
        x1 = d0;
    }
}
