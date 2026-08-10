#include <cuda_runtime.h>
#include <math.h>


__device__ __forceinline__ bool finite_f(float x) { return isfinite(x); }


extern "C" __global__ void pvi_build_scale_f32(
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int len,
    int first_valid,
    float* __restrict__ scale_out)
{
    if (len <= 0) return;
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const int fv = first_valid < 0 ? 0 : first_valid;
    for (int i = 0; i < len; ++i) scale_out[i] = nanf("");
    if (fv >= len) return;

    scale_out[fv] = 1.0f;


    double prev_close = (double)close[fv];
    double prev_vol = (double)volume[fv];
    double accum = 1.0;

    for (int i = fv + 1; i < len; ++i) {
        const float cf = close[i];
        const float vf = volume[i];
        if (finite_f(cf) && finite_f(vf) && isfinite(prev_close) && isfinite(prev_vol)) {
            if ((double)vf > prev_vol) {
                const double c = (double)cf;
                const double r = (c - prev_close) / prev_close;

                accum += r * accum;
            }
            scale_out[i] = (float)accum;
            prev_close = (double)cf;
            prev_vol = (double)vf;
        } else {
            scale_out[i] = nanf("");
            if (finite_f(cf) && finite_f(vf)) {
                prev_close = (double)cf;
                prev_vol = (double)vf;
            }
        }
    }
}


extern "C" __global__ void pvi_build_scale_warp16_f32(
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int len,
    int first_valid,
    float* __restrict__ scale_out)
{
    if (len <= 0) return;
    if (blockIdx.x != 0) return;

    const int lane = threadIdx.x & 31;
    if (threadIdx.x >= 16) return;
    const unsigned mask = 0x0000ffffu;

    const int fv = first_valid < 0 ? 0 : first_valid;
    const float nan_f = nanf("");


    for (int i = lane; i < fv && i < len; i += 16) scale_out[i] = nan_f;
    if (fv >= len) return;

    if (lane == 0) scale_out[fv] = 1.0f;
    if (fv + 1 >= len) return;

    double accum0 = 1.0;

    for (int t0 = fv + 1; t0 < len; t0 += 16) {
        const int i = t0 + lane;
        double f = 1.0;
        if (i < len) {
            const float cf = close[i];
            const float c0 = close[i - 1];
            const float vf = volume[i];
            const float v0 = volume[i - 1];
            if ((double)vf > (double)v0) {
                const double c = (double)cf;
                const double prev = (double)c0;
                const double r = (c - prev) / prev;
                f = 1.0 + r;
            }
        }


        double prefix = f;
        for (int offset = 1; offset < 16; offset <<= 1) {
            double other = __shfl_up_sync(mask, prefix, offset, 16);
            if (lane >= offset) prefix *= other;
        }

        const double base = __shfl_sync(mask, accum0, 0, 16);
        if (i < len) scale_out[i] = (float)(base * prefix);

        const double tile_prod = __shfl_sync(mask, prefix, 15, 16);
        if (lane == 0) accum0 *= tile_prod;
    }
}


extern "C" __global__ void pvi_apply_scale_batch_f32(
    const float* __restrict__ scale,
    int len,
    int first_valid,
    const float* __restrict__ initial_values,
    int rows,
    float* __restrict__ out)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
    if (t >= len || r >= rows || rows <= 0) return;

    const float nan_f = nanf("");
    const size_t out_idx = (size_t)r * (size_t)len + (size_t)t;

    if (t < first_valid) {
        out[out_idx] = nan_f;
        return;
    }
    const float ivf = initial_values[r];
    if (t == first_valid) {
        out[out_idx] = ivf;
        return;
    }

    const float s = scale[t];
    if (!isfinite(s)) {
        out[out_idx] = nan_f;
        return;
    }

    const double iv = (double)ivf;
    const double sd = (double)s;
    out[out_idx] = (float)(iv * sd);
}


extern "C" __global__ void pvi_apply_batch_direct_f32(
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int len,
    int first_valid,
    const float* __restrict__ initial_values,
    int rows,
    float* __restrict__ out)
{
    if (rows <= 0 || len <= 0) return;
    const int fv = first_valid < 0 ? 0 : first_valid;
    const float nan_f = nanf("");

    const int stride = blockDim.x * gridDim.x;
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += stride) {

        for (int t = 0; t < min(fv, len); ++t) out[(size_t)r * len + t] = nan_f;
        if (fv >= len) continue;

        double pvi = (double)initial_values[r];
        out[(size_t)r * len + fv] = (float)pvi;
        if (fv + 1 >= len) continue;

        double prev_close = (double)close[fv];
        double prev_vol   = (double)volume[fv];
        for (int t = fv + 1; t < len; ++t) {
            const float cf = close[t];
            const float vf = volume[t];
            if (isfinite(cf) && isfinite(vf) && isfinite(prev_close) && isfinite(prev_vol)) {
                if ((double)vf > prev_vol) {
                    const double c = (double)cf;

                    pvi *= c / prev_close;
                }
                out[(size_t)r * len + t] = (float)pvi;
                prev_close = (double)cf;
                prev_vol   = (double)vf;
            } else {
                out[(size_t)r * len + t] = nan_f;
                if (isfinite(cf) && isfinite(vf)) {
                    prev_close = (double)cf;
                    prev_vol   = (double)vf;
                }
            }
        }
    }
}


extern "C" __global__ void pvi_many_series_one_param_f32(
    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    int cols,
    int rows,
    const int* __restrict__ first_valids,
    float initial_value,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols || rows <= 0) return;

    const int fv = first_valids[s] < 0 ? 0 : first_valids[s];
    const float nan_f = nanf("");


    for (int t = 0; t < fv && t < rows; ++t) {
        out_tm[t * cols + s] = nan_f;
    }
    if (fv >= rows) return;

    double pvi = (double)initial_value;
    out_tm[fv * cols + s] = (float)pvi;
    if (fv + 1 >= rows) return;

    double prev_close = (double)close_tm[fv * cols + s];
    double prev_vol = (double)volume_tm[fv * cols + s];

    for (int t = fv + 1; t < rows; ++t) {
        const float cf = close_tm[t * cols + s];
        const float vf = volume_tm[t * cols + s];
        if (isfinite(cf) && isfinite(vf) && isfinite(prev_close) && isfinite(prev_vol)) {
            if ((double)vf > prev_vol) {
                const double c = (double)cf;
                const double r = (c - prev_close) / prev_close;
                pvi += r * pvi;
            }
            out_tm[t * cols + s] = (float)pvi;
            prev_close = (double)cf;
            prev_vol = (double)vf;
        } else {
            out_tm[t * cols + s] = nan_f;
            if (isfinite(cf) && isfinite(vf)) {
                prev_close = (double)cf;
                prev_vol = (double)vf;
            }
        }
    }
}


// ===========================================================================
// S2 f64 LANE — pvi  (positive volume index)
// ===========================================================================
// Reference: src/indicators/pvi.rs
//   `pvi_with_kernel` (:229) — first_valid = first index where CLOSE AND
//                               VOLUME are both non-NaN; NaN prefix = that
//                               index; `valid < 2` is the refusal
//   `pvi_scalar`      (:419) — the recurrence
//   `PviInput::get_initial_value` (:144) -> unwrap_or(1000.0)
//
// PERIOD-INVARIANT. `pvi` has no period parameter at all; every row of a
// period sweep is the same series. Declared as such in `F64Kernel::
// is_period_invariant` so telemetry can explain the repeated work instead of
// leaving it to be discovered.
//
// THE NaN BRANCH IS THE WHOLE INDICATOR'S ROBUSTNESS AND IT IS ASYMMETRIC.
// On a bar where close or volume is NaN the CPU emits NaN, does NOT update
// `pvi`, and updates `prev_close`/`prev_vol` ONLY IF BOTH of the CURRENT
// values are non-NaN — which, inside that branch, can only be true when the
// PREVIOUS pair was the NaN one. Getting this wrong either freezes the
// comparison base forever or advances it to a NaN, and both poison every
// later bar. Reproduced branch for branch.
//
// ROUNDINGS: r = (c - prev_close) / prev_close -> sub + div (2);
//            pvi += r * pvi                    -> mul + add (2).
// NOT `pvi = pvi.mul_add(r, pvi)` — that would be one rounding where the CPU
// makes two.
//
// f32 hazards fixed here: 5 f32 kernels above, 6 f32-bit-pattern NaNs.
// ===========================================================================

__device__ __forceinline__ double neo_s2_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void neoethos_pvi_batch_f64(
    const double* __restrict__ close,
    const double* __restrict__ volume,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;
    (void)periods;   // pvi has no period parameter — see is_period_invariant.

    double* __restrict__ row = out + (size_t)r * (size_t)n;

    const bool declined =
        (n <= 0) ||
        (first_valid < 0) || (first_valid >= n) ||
        ((n - first_valid) < 2);
    if (declined) {
        for (int i = 0; i < n; ++i) row[i] = neo_s2_qnan();
        return;
    }

    for (int i = 0; i < first_valid; ++i) row[i] = neo_s2_qnan();

    double pvi = 1000.0;   // PviParams::get_initial_value -> unwrap_or(1000.0)
    row[first_valid] = pvi;

    double prev_close = close[first_valid];
    double prev_vol = volume[first_valid];

    for (int i = first_valid + 1; i < n; ++i) {
        const double c = close[i];
        const double v = volume[i];
        if (isnan(c) || isnan(v) || isnan(prev_close) || isnan(prev_vol)) {
            row[i] = neo_s2_qnan();
            if (!isnan(c) && !isnan(v)) {
                prev_close = c;
                prev_vol = v;
            }
            continue;
        }
        if (v > prev_vol) {
            const double rr = (c - prev_close) / prev_close;
            pvi += rr * pvi;
        }
        row[i] = pvi;
        prev_close = c;
        prev_vol = v;
    }
}
