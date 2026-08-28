#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

#ifndef VPT_SCAN_BLOCK_X
#define VPT_SCAN_BLOCK_X 256
#endif

#ifndef VPT_SCAN_ITEMS_PER_THREAD
#define VPT_SCAN_ITEMS_PER_THREAD 8
#endif

#define VPT_SCAN_TILE (VPT_SCAN_BLOCK_X * VPT_SCAN_ITEMS_PER_THREAD)


static __device__ __forceinline__ void kahan_add(float x, float &sum, float &c) {
    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}


extern "C" __global__ void vpt_scan_blocks_f32(
    const float* __restrict__ price,
    const float* __restrict__ volume,
    int len,
    int first_valid,
    float* __restrict__ out,
    double* __restrict__ block_sums)
{
    __shared__ double scan[VPT_SCAN_TILE];
    __shared__ double temp[VPT_SCAN_TILE];

    const int base = blockIdx.x * VPT_SCAN_TILE;
    const int tid = threadIdx.x;
    const float nan_f = CUDART_NAN_F;

    if (first_valid < 0) first_valid = 0;

    #pragma unroll
    for (int j = 0; j < VPT_SCAN_ITEMS_PER_THREAD; ++j) {
        const int lane = tid + j * VPT_SCAN_BLOCK_X;
        const int idx = base + lane;
        double inc = 0.0;
        if (idx >= first_valid && idx < len) {
            if (idx < 1) {
                inc = (double)nan_f;
            } else {
                const float p0 = price[idx - 1];
                const float p1 = price[idx];
                const float v1 = volume[idx];
                inc = (isfinite(p0) && p0 != 0.0f && isfinite(p1) && isfinite(v1))
                    ? (double)v1 * ((double)p1 - (double)p0) / (double)p0
                    : (double)nan_f;
            }
        }
        scan[lane] = inc;
    }
    __syncthreads();

    for (int offset = 1; offset < VPT_SCAN_TILE; offset <<= 1) {
        #pragma unroll
        for (int j = 0; j < VPT_SCAN_ITEMS_PER_THREAD; ++j) {
            const int lane = tid + j * VPT_SCAN_BLOCK_X;
            temp[lane] = scan[lane] + (lane >= offset ? scan[lane - offset] : 0.0);
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < VPT_SCAN_ITEMS_PER_THREAD; ++j) {
            const int lane = tid + j * VPT_SCAN_BLOCK_X;
            scan[lane] = temp[lane];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < VPT_SCAN_ITEMS_PER_THREAD; ++j) {
        const int lane = tid + j * VPT_SCAN_BLOCK_X;
        const int idx = base + lane;
        if (idx < len) out[idx] = idx <= first_valid ? nan_f : (float)scan[lane];
    }

    if (tid == 0) {
        int remaining = len - base;
        int count = remaining > VPT_SCAN_TILE ? VPT_SCAN_TILE : remaining;
        block_sums[blockIdx.x] = count > 0 ? scan[count - 1] : 0.0;
    }
}


extern "C" __global__ void vpt_scan_block_sums_f64(
    double* __restrict__ block_sums,
    int num_blocks)
{
    __shared__ double scan[VPT_SCAN_TILE];
    __shared__ double temp[VPT_SCAN_TILE];

    const int tid = threadIdx.x;
    #pragma unroll
    for (int j = 0; j < VPT_SCAN_ITEMS_PER_THREAD; ++j) {
        const int lane = tid + j * VPT_SCAN_BLOCK_X;
        scan[lane] = lane < num_blocks ? block_sums[lane] : 0.0;
    }
    __syncthreads();

    for (int offset = 1; offset < VPT_SCAN_TILE; offset <<= 1) {
        #pragma unroll
        for (int j = 0; j < VPT_SCAN_ITEMS_PER_THREAD; ++j) {
            const int lane = tid + j * VPT_SCAN_BLOCK_X;
            temp[lane] = scan[lane] + (lane >= offset ? scan[lane - offset] : 0.0);
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < VPT_SCAN_ITEMS_PER_THREAD; ++j) {
            const int lane = tid + j * VPT_SCAN_BLOCK_X;
            scan[lane] = temp[lane];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < VPT_SCAN_ITEMS_PER_THREAD; ++j) {
        const int lane = tid + j * VPT_SCAN_BLOCK_X;
        if (lane < num_blocks) block_sums[lane] = scan[lane];
    }
}


extern "C" __global__ void vpt_add_block_offsets_f32(
    float* __restrict__ out,
    int len,
    int first_valid,
    const double* __restrict__ block_sums)
{
    const int base = blockIdx.x * VPT_SCAN_TILE;
    if (blockIdx.x == 0) return;

    if (first_valid < 0) first_valid = 0;
    const double offset = block_sums[blockIdx.x - 1];
    const int tid = threadIdx.x;

    #pragma unroll
    for (int j = 0; j < VPT_SCAN_ITEMS_PER_THREAD; ++j) {
        const int lane = tid + j * VPT_SCAN_BLOCK_X;
        const int idx = base + lane;
        if (idx < len && idx > first_valid) out[idx] = (float)((double)out[idx] + offset);
    }
}


extern "C" __global__ void vpt_batch_f32(
    const float* __restrict__ price,
    const float* __restrict__ volume,
    int len,
    int first_valid,
    float* __restrict__ out)
{

    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (len <= 0) return;

    const float nan_f = CUDART_NAN_F;


    if (first_valid < 0) first_valid = 0;


    const int warm_end = (first_valid < len) ? first_valid : (len - 1);
    for (int i = 0; i <= warm_end; ++i) out[i] = nan_f;


    if (first_valid + 1 >= len) return;


    if (first_valid < 1) {
        for (int t = first_valid + 1; t < len; ++t) out[t] = nan_f;
        return;
    }


    float p0 = price[first_valid - 1];
    float p1 = price[first_valid];
    float v1 = volume[first_valid];


    bool ok = isfinite(p0) && isfinite(p1) && isfinite(v1) && (p0 != 0.0f);
    if (!ok) {
        for (int t = first_valid + 1; t < len; ++t) out[t] = nan_f;
        return;
    }

    float prev_p = p1;


    float sum = v1 * ((p1 - p0) / p0);
    float c = 0.0f;


    for (int t = first_valid + 1; t < len; ++t) {
        float pt = price[t];
        float vt = volume[t];

        bool good = isfinite(prev_p) && isfinite(pt) && isfinite(vt) && (prev_p != 0.0f);
        if (!good) {

            for (int j = t; j < len; ++j) out[j] = nan_f;
            return;
        }

        float cur = vt * ((pt - prev_p) / prev_p);
        kahan_add(cur, sum, c);
        out[t] = sum;

        prev_p = pt;
    }
}


extern "C" __global__ void vpt_many_series_one_param_f32(
    const float* __restrict__ price_tm,
    const float* __restrict__ volume_tm,
    int cols,
    int rows,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm)
{

    for (int s = blockIdx.x * blockDim.x + threadIdx.x;
         s < cols;
         s += blockDim.x * gridDim.x)
    {
        const float nan_f = CUDART_NAN_F;

        int fv = first_valids[s];
        if (fv < 0) fv = 0;


        float sum = 0.0f;
        float c = 0.0f;
        float prev_p = nan_f;
        bool sticky_nan = false;


        for (int t = 0; t < rows; ++t) {
            const int idx = t * cols + s;
            const float pt = price_tm[idx];
            const float vt = volume_tm[idx];

            if (t <= fv) {

                out_tm[idx] = nan_f;


                if (t == fv) {
                    if (fv < 1) {
                        sticky_nan = true;
                    } else {
                        const float p0 = price_tm[(t - 1) * cols + s];
                        const float v1 = vt;
                        const bool ok = isfinite(p0) && isfinite(pt) && isfinite(v1) && (p0 != 0.0f);
                        if (ok) {
                            sum = v1 * ((pt - p0) / p0);
                            c = 0.0f;
                            prev_p = pt;
                        } else {
                            sticky_nan = true;
                        }
                    }
                }
                continue;
            }


            if (sticky_nan) {
                out_tm[idx] = nan_f;
                continue;
            }

            const bool good = isfinite(prev_p) && isfinite(pt) && isfinite(vt) && (prev_p != 0.0f);
            if (!good) {
                sticky_nan = true;
                out_tm[idx] = nan_f;
                continue;
            }

            const float cur = vt * ((pt - prev_p) / prev_p);
            kahan_add(cur, sum, c);
            out_tm[idx] = sum;
            prev_p = pt;
        }
    }
}


// ===========================================================================
// S2 f64 LANE — vpt  (volume price trend)
// ===========================================================================
// Reference: src/indicators/vpt.rs
//   `vpt_first_valid`     (:148) — the start rule
//   `vpt_with_kernel`     (:165) — alloc_with_nan_prefix(len, first + 1)
//   `vpt_row_scalar_from` (:1103) — the seed and the running sum
//
// PERIOD-INVARIANT: `vpt` has no period parameter.
//
// ITS FIRST-VALID RULE IS ITS OWN, AND THE HOST'S VALUE IS NOT IT.
//   `vpt_first_valid` returns the first i >= 1 for which
//       price[i-1] is finite AND price[i-1] != 0 AND
//       price[i]   is finite AND volume[i] is finite
//   — three differences from the lane's common rule: it starts at 1, it
//   requires FINITE (so an infinity is rejected, not just a NaN), and it
//   rejects a ZERO previous price because that price is the divisor. The host
//   cannot express that in `F64FirstValidRule`, so this kernel DERIVES it
//   from the arrays it already has and uses the passed `first_valid` only as
//   a bounds sanity check. The registry row says `AllInputsNonNan` because
//   that is what the host computes and hands over; the value does not select
//   the window here, and this comment is the record of that.
//
// THE SEED IS NOT ZERO. `prev` starts as the ONE-BAR vpt term at index
// `start_i - 1` when `start_i >= 2`, and 0.0 only when `start_i == 1`. A
// kernel that seeded with 0.0 unconditionally would be off by one whole term
// for the entire series — a constant offset, which is exactly the kind of
// error a relative-tolerance parity check waves through.
//
// THE UNROLL BY FOUR IN THE CPU CHANGES NOTHING: the four bodies are
// identical and each depends on the one before, so a single loop reproduces
// the association exactly.
//
// ROUNDINGS: cur = v1 * ((p1 - p_prev) / p_prev) -> sub + div + mul (3);
//            val = cur + prev                    -> add             (1).
// The f32 kernels above compute the same thing with `__fmaf_rn`, which fuses
// the last multiply and add into ONE rounding — a different number even
// before the width changes.
// ===========================================================================

__device__ __forceinline__ double neo_s2_qnan() {
    return __longlong_as_double(0x7ff8000000000000LL);
}

extern "C" __global__ void neoethos_vpt_batch_f64(
    const double* __restrict__ price,
    const double* __restrict__ volume,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;
    (void)periods;
    (void)first_valid;   // see the header: vpt derives its own start.

    double* __restrict__ row = out + (size_t)r * (size_t)n;

    if (n <= 0) return;
    for (int i = 0; i < n; ++i) row[i] = neo_s2_qnan();

    // `valid_count < 2` refusal.
    int valid_count = 0;
    for (int i = 0; i < n; ++i) {
        if (!(isnan(price[i]) || isnan(volume[i]))) valid_count += 1;
        if (valid_count >= 2) break;
    }
    if (valid_count < 2) return;

    // `vpt_first_valid`.
    int first = -1;
    for (int i = 1; i < n; ++i) {
        const double p0 = price[i - 1];
        const double p1 = price[i];
        const double v1 = volume[i];
        if (isfinite(p0) && p0 != 0.0 && isfinite(p1) && isfinite(v1)) { first = i; break; }
    }
    if (first < 0) return;

    const int start_i = first + 1;
    if (start_i >= n) return;

    double prev;
    if (start_i >= 2) {
        const int k = start_i - 1;
        const double p0 = price[k - 1];
        const double p1 = price[k];
        const double v1 = volume[k];
        // The CPU tests `p0 != p0` (NaN) and `p0 == 0.0` — NOT `is_finite`.
        // An infinite p0 reaches the divide here, deliberately.
        if ((p0 != p0) || (p0 == 0.0) || (p1 != p1) || (v1 != v1)) {
            prev = neo_s2_qnan();
        } else {
            prev = v1 * ((p1 - p0) / p0);
        }
    } else {
        prev = 0.0;
    }

    double p_prev = price[start_i - 1];
    for (int i = start_i; i < n; ++i) {
        const double p1 = price[i];
        const double v1 = volume[i];
        double cur;
        if ((p_prev != p_prev) || (p_prev == 0.0) || (p1 != p1) || (v1 != v1)) {
            cur = neo_s2_qnan();
        } else {
            cur = v1 * ((p1 - p_prev) / p_prev);
        }
        const double val = cur + prev;
        row[i] = val;
        prev = val;
        p_prev = p1;
    }
}
