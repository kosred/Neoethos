#include <cuda_runtime.h>
#include <math.h>

#ifndef PRB_BATCH_CHUNK_LEN
#define PRB_BATCH_CHUNK_LEN 4096
#endif

extern "C" {


__constant__ float PRB_BINOM_SIGN[8][8] = {
    {  1.0f,  0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f },
    { -1.0f,  1.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f },
    {  1.0f, -2.0f,   1.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f },
    { -1.0f,  3.0f,  -3.0f,   1.0f,   0.0f,   0.0f,   0.0f,   0.0f },
    {  1.0f, -4.0f,   6.0f,  -4.0f,   1.0f,   0.0f,   0.0f,   0.0f },
    { -1.0f,  5.0f, -10.0f,  10.0f,  -5.0f,   1.0f,   0.0f,   0.0f },
    {  1.0f, -6.0f,  15.0f, -20.0f,  15.0f,  -6.0f,   1.0f,   0.0f },
    { -1.0f,  7.0f, -21.0f,  35.0f, -35.0f,  21.0f,  -7.0f,   1.0f }
};

__device__ __forceinline__ float qnan32() { return __int_as_float(0x7fffffff); }

__global__ void prb_ssf_filter_f32_serial(
    const float* __restrict__ data,
    const int len,
    const int first_valid,
    const int period,
    float* __restrict__ out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    const float nan = qnan32();
    if (!data || !out || len <= 0 || first_valid < 0 || first_valid >= len || period <= 0) {
        for (int i = 0; i < len; ++i) out[i] = nan;
        return;
    }

    for (int i = 0; i < first_valid; ++i) out[i] = nan;

    const float pi = 3.14159265358979323846f;
    const float omega = 2.0f * pi / (float)period;
    const float a = expf(-1.4142135623730951f * pi / (float)period);
    const float b = 2.0f * a * cosf(0.7071067811865476f * omega);
    const float c3 = -a * a;
    const float c2 = b;
    const float c1 = 1.0f - c2 - c3;

    float y1 = nan;
    float y2 = nan;
    for (int i = first_valid; i < len; ++i) {
        const float x = data[i];
        const float prev1 = isnan(y1) ? x : y1;
        const float prev2 = isnan(y2) ? prev1 : y2;
        const float y = c1 * x + c2 * prev1 + c3 * prev2;
        out[i] = y;
        y2 = y1;
        y1 = y;
    }
}

__global__ void prb_contig_valid_f32_serial(
    const float* __restrict__ data,
    const int len,
    int* __restrict__ out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int count = 0;
    for (int i = 0; i < len; ++i) {
        const float x = data[i];
        if (isnan(x)) {
            count = 0;
        } else {
            count += 1;
        }
        out[i] = count;
    }
}

__device__ __forceinline__ float horner_eval(const float* coeffs, int m, float x) {

    float acc = 0.0f;
    #pragma unroll
    for (int p = m - 1; p >= 0; --p) {
        acc = fmaf(acc, x, coeffs[p]);
    }
    return acc;
}


__device__ __forceinline__ float kahan_add(float sum, float x, float &c) {
    float y = x - c;
    float t = sum + y;
    c = (t - sum) - y;
    return t;
}

__device__ __forceinline__ void solve_coeffs_kahan(
    const float* __restrict__ arow,
    int max_m,
    int m,
    const float* __restrict__ S,
    float* __restrict__ coeffs) {
    #pragma unroll
    for (int r = 0; r < m; ++r) {
        float acc = 0.0f, c = 0.0f;
        const float* rowp = arow + r * max_m;
        #pragma unroll
        for (int cidx = 0; cidx < m; ++cidx) {
            acc = kahan_add(acc, rowp[cidx] * S[cidx], c);
        }
        coeffs[r] = acc;
    }
}

__global__ void prb_batch_f32(
    const float* __restrict__ data,
    const int len,
    const int first_valid,
    const int* __restrict__ periods,
    const int* __restrict__ orders,
    const int* __restrict__ offsets,
    const int combos,
    const int max_m,
    const float* __restrict__ a_inv,
    const int a_stride,
    const int* __restrict__ contig,
    const float ndev,
    const int* __restrict__ row_indices,
    float* __restrict__ out_main,
    float* __restrict__ out_up,
    float* __restrict__ out_lo)
{
    const int row = blockIdx.y;
    if (row >= combos) return;

    const int abs_row = row_indices ? row_indices[row] : row;
    const int n = periods[row];
    const int k = orders[row];
    const int m = k + 1;
    const int offset = offsets[row];
    const float x_pos = float(n) - float(offset);

    const float* arow = a_inv + row * a_stride;


    const int warm = first_valid + n - 1;
    const float nan = qnan32();


    float npow[8]; npow[0] = 1.0f;
    #pragma unroll
    for (int r = 1; r <= k; ++r) npow[r] = npow[r-1] * float(n);


    for (int i = 0; i < warm && i < len; ++i) {
        const int out_idx = abs_row * len + i;
        out_main[out_idx] = nan;
        out_up[out_idx]   = nan;
        out_lo[out_idx]   = nan;
    }
    if (warm >= len) return;


    if (contig[warm] < n) {
        for (int i = warm; i < len; ++i) {
            const int out_idx = abs_row * len + i;
            out_main[out_idx] = nan;
            out_up[out_idx]   = nan;
            out_lo[out_idx]   = nan;
        }
        return;
    }


    float S[8];
    float cS[8];
    #pragma unroll
    for (int r = 0; r < 8; ++r) { S[r] = 0.0f; cS[r] = 0.0f; }

    float sum = 0.0f, csum = 0.0f;
    float sumsq = 0.0f, csum2 = 0.0f;

    const int base0 = warm - n + 1;
    for (int j = 1; j <= n; ++j) {
        const float y = data[base0 + j - 1];

        sum   = kahan_add(sum, y, csum);
        sumsq = kahan_add(sumsq, y * y, csum2);


        float pwr = float(j);
        #pragma unroll
        for (int r = 1; r <= k; ++r) {
            S[r] = kahan_add(S[r], y * pwr, cS[r]);
            pwr *= float(j);
        }
    }
    S[0] = sum;


    {
        float coeffs[8];
        solve_coeffs_kahan(arow, max_m, m, S, coeffs);
        const float reg = horner_eval(coeffs, m, x_pos);
        const float invn = 1.0f / float(n);
        const float mean = sum * invn;
        float var = fmaf(sumsq, invn, -mean * mean);
        if (var < 0.0f) var = 0.0f;
        const float stdev = sqrtf(var);

        const int out_idx = abs_row * len + warm;
        out_main[out_idx] = reg;
        out_up[out_idx]   = reg + ndev * stdev;
        out_lo[out_idx]   = reg - ndev * stdev;
    }


    bool poisoned = false;
    float S_old[8];

    for (int i = warm + 1; i < len; ++i) {
        const int out_idx = abs_row * len + i;

        if (poisoned || contig[i] < n) {
            poisoned = true;
            out_main[out_idx] = nan;
            out_up[out_idx]   = nan;
            out_lo[out_idx]   = nan;
            continue;
        }


        #pragma unroll
        for (int r = 0; r <= k; ++r) S_old[r] = S[r];

        const float y_old = data[i - n];
        const float y_new = data[i];


        sum   = kahan_add(sum, -y_old, csum);
        sum   = kahan_add(sum,  y_new, csum);
        S[0]  = sum;
        sumsq = kahan_add(sumsq, -y_old * y_old, csum2);
        sumsq = kahan_add(sumsq,  y_new * y_new, csum2);


        #pragma unroll
        for (int r = 1; r <= k; ++r) {
            float acc = 0.0f, c = 0.0f;
            #pragma unroll
            for (int p = 0; p <= r; ++p) {
                acc = kahan_add(acc, PRB_BINOM_SIGN[r][p] * S_old[p], c);
            }

            S[r] = fmaf(y_new, npow[r], acc);
        }


        float coeffs[8];
        solve_coeffs_kahan(arow, max_m, m, S, coeffs);
        const float reg = horner_eval(coeffs, m, x_pos);
        const float invn = 1.0f / float(n);
        const float mean = sum * invn;
        float var = fmaf(sumsq, invn, -mean * mean);
        if (var < 0.0f) var = 0.0f;
        const float stdev = sqrtf(var);

        out_main[out_idx] = reg;
        out_up[out_idx]   = reg + ndev * stdev;
        out_lo[out_idx]   = reg - ndev * stdev;
    }
}


__global__ void prb_batch_chunked_f32(
    const float* __restrict__ data,
    const int len,
    const int first_valid,
    const int* __restrict__ periods,
    const int* __restrict__ orders,
    const int* __restrict__ offsets,
    const int combos,
    const int max_m,
    const float* __restrict__ a_inv,
    const int a_stride,
    const int* __restrict__ contig,
    const float ndev,
    const int* __restrict__ row_indices,
    float* __restrict__ out_main,
    float* __restrict__ out_up,
    float* __restrict__ out_lo)
{
    (void)contig;

    const int row = (int)blockIdx.y;
    if (row >= combos) return;

    const int chunk_id = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int chunk_start = chunk_id * PRB_BATCH_CHUNK_LEN;
    if (chunk_start >= len) return;
    const int chunk_end = min(chunk_start + PRB_BATCH_CHUNK_LEN, len);

    const int abs_row = row_indices ? row_indices[row] : row;
    const int n = periods[row];
    const int k = orders[row];
    const int m = k + 1;
    const int offset = offsets[row];
    const float x_pos = float(n) - float(offset);

    const float* arow = a_inv + row * a_stride;
    const int warm = first_valid + n - 1;
    const float nan = qnan32();


    float npow[8]; npow[0] = 1.0f;
    #pragma unroll
    for (int r = 1; r <= k; ++r) npow[r] = npow[r - 1] * float(n);


    if (chunk_end <= warm) {
        for (int i = chunk_start; i < chunk_end; ++i) {
            const int out_idx = abs_row * len + i;
            out_main[out_idx] = nan;
            out_up[out_idx]   = nan;
            out_lo[out_idx]   = nan;
        }
        return;
    }

    int i0 = chunk_start;
    for (; i0 < warm && i0 < chunk_end; ++i0) {
        const int out_idx = abs_row * len + i0;
        out_main[out_idx] = nan;
        out_up[out_idx]   = nan;
        out_lo[out_idx]   = nan;
    }
    if (i0 >= chunk_end) return;


    float S[8];
    float cS[8];
    #pragma unroll
    for (int r = 0; r < 8; ++r) { S[r] = 0.0f; cS[r] = 0.0f; }

    float sum = 0.0f, csum = 0.0f;
    float sumsq = 0.0f, csum2 = 0.0f;

    const int base0 = i0 - n + 1;
    for (int j = 1; j <= n; ++j) {
        const float y = data[base0 + j - 1];
        sum   = kahan_add(sum, y, csum);
        sumsq = kahan_add(sumsq, y * y, csum2);

        float pwr = float(j);
        #pragma unroll
        for (int r = 1; r <= k; ++r) {
            S[r] = kahan_add(S[r], y * pwr, cS[r]);
            pwr *= float(j);
        }
    }
    S[0] = sum;


    {
        float coeffs[8];
        solve_coeffs_kahan(arow, max_m, m, S, coeffs);
        const float reg = horner_eval(coeffs, m, x_pos);
        const float invn = 1.0f / float(n);
        const float mean = sum * invn;
        float var = fmaf(sumsq, invn, -mean * mean);
        if (var < 0.0f) var = 0.0f;
        const float stdev = sqrtf(var);

        const int out_idx = abs_row * len + i0;
        out_main[out_idx] = reg;
        out_up[out_idx]   = reg + ndev * stdev;
        out_lo[out_idx]   = reg - ndev * stdev;
    }


    float S_old[8];
    for (int i = i0 + 1; i < chunk_end; ++i) {
        const int out_idx = abs_row * len + i;

        #pragma unroll
        for (int r = 0; r <= k; ++r) S_old[r] = S[r];

        const float y_old = data[i - n];
        const float y_new = data[i];

        sum   = kahan_add(sum, -y_old, csum);
        sum   = kahan_add(sum,  y_new, csum);
        S[0]  = sum;
        sumsq = kahan_add(sumsq, -y_old * y_old, csum2);
        sumsq = kahan_add(sumsq,  y_new * y_new, csum2);

        #pragma unroll
        for (int r = 1; r <= k; ++r) {
            float acc = 0.0f, c = 0.0f;
            #pragma unroll
            for (int p = 0; p <= r; ++p) {
                acc = kahan_add(acc, PRB_BINOM_SIGN[r][p] * S_old[p], c);
            }
            S[r] = fmaf(y_new, npow[r], acc);
        }

        float coeffs[8];
        solve_coeffs_kahan(arow, max_m, m, S, coeffs);
        const float reg = horner_eval(coeffs, m, x_pos);
        const float invn = 1.0f / float(n);
        const float mean = sum * invn;
        float var = fmaf(sumsq, invn, -mean * mean);
        if (var < 0.0f) var = 0.0f;
        const float stdev = sqrtf(var);

        out_main[out_idx] = reg;
        out_up[out_idx]   = reg + ndev * stdev;
        out_lo[out_idx]   = reg - ndev * stdev;
    }
}

__global__ void prb_many_series_one_param_f32(
    const float* __restrict__ data_tm,
    const int cols,
    const int rows,
    const int period,
    const int order,
    const int offset,
    const int max_m,
    const float* __restrict__ a_inv,
    const int a_stride,
    const int* __restrict__ contig_tm,
    const int* __restrict__ first_valids,
    const float ndev,
    float* __restrict__ out_main_tm,
    float* __restrict__ out_up_tm,
    float* __restrict__ out_lo_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    const int n = period;
    const int k = order;
    const int m = k + 1;
    const float x_pos = float(n) - float(offset);
    const float* ainv = a_inv;

    const float nan = qnan32();
    const int fv = first_valids ? first_valids[s] : 0;
    const int warm = fv + n - 1;


    float npow[8]; npow[0] = 1.0f;
    #pragma unroll
    for (int r = 1; r <= k; ++r) npow[r] = npow[r-1] * float(n);


    for (int t = 0; t < rows && t < warm; ++t) {
        const int idx = t * cols + s;
        out_main_tm[idx] = nan;
        out_up_tm[idx]   = nan;
        out_lo_tm[idx]   = nan;
    }
    if (warm >= rows) return;


    if (contig_tm[warm * cols + s] < n) {
        for (int t = warm; t < rows; ++t) {
            const int idx = t * cols + s;
            out_main_tm[idx] = nan;
            out_up_tm[idx]   = nan;
            out_lo_tm[idx]   = nan;
        }
        return;
    }


    float S[8];
    float cS[8];
    #pragma unroll
    for (int r = 0; r < 8; ++r) { S[r] = 0.0f; cS[r] = 0.0f; }

    float sum = 0.0f, csum = 0.0f;
    float sumsq = 0.0f, csum2 = 0.0f;

    const int base0 = warm - n + 1;
    for (int j = 1; j <= n; ++j) {
        const float y = data_tm[(base0 + j - 1) * cols + s];
        sum   = kahan_add(sum, y, csum);
        sumsq = kahan_add(sumsq, y * y, csum2);

        float pwr = float(j);
        #pragma unroll
        for (int r = 1; r <= k; ++r) { S[r] = kahan_add(S[r], y * pwr, cS[r]); pwr *= float(j); }
    }
    S[0] = sum;


    {
        float coeffs[8];
        solve_coeffs_kahan(ainv, max_m, m, S, coeffs);
        const float reg = horner_eval(coeffs, m, x_pos);
        const float invn = 1.0f / float(n);
        const float mean = sum * invn;
        float var = fmaf(sumsq, invn, -mean * mean); if (var < 0.0f) var = 0.0f;
        const float stdev = sqrtf(var);

        const int idx = warm * cols + s;
        out_main_tm[idx] = reg;
        out_up_tm[idx]   = reg + ndev * stdev;
        out_lo_tm[idx]   = reg - ndev * stdev;
    }


    bool poisoned = false;
    float S_old[8];

    for (int t = warm + 1; t < rows; ++t) {
        const int idx = t * cols + s;

        if (poisoned || contig_tm[idx] < n) {
            poisoned = true;
            out_main_tm[idx] = nan;
            out_up_tm[idx]   = nan;
            out_lo_tm[idx]   = nan;
            continue;
        }

        #pragma unroll
        for (int r = 0; r <= k; ++r) S_old[r] = S[r];

        const float y_old = data_tm[(t - n) * cols + s];
        const float y_new = data_tm[t * cols + s];

        sum   = kahan_add(sum, -y_old, csum);
        sum   = kahan_add(sum,  y_new, csum);
        S[0]  = sum;
        sumsq = kahan_add(sumsq, -y_old * y_old, csum2);
        sumsq = kahan_add(sumsq,  y_new * y_new, csum2);

        #pragma unroll
        for (int r = 1; r <= k; ++r) {
            float acc = 0.0f, c = 0.0f;
            #pragma unroll
            for (int p = 0; p <= r; ++p) acc = kahan_add(acc, PRB_BINOM_SIGN[r][p] * S_old[p], c);
            S[r] = fmaf(y_new, npow[r], acc);
        }

        float coeffs[8]; solve_coeffs_kahan(ainv, max_m, m, S, coeffs);
        const float reg = horner_eval(coeffs, m, x_pos);
        const float invn = 1.0f / float(n);
        const float mean = sum * invn;
        float var = fmaf(sumsq, invn, -mean * mean); if (var < 0.0f) var = 0.0f;
        const float stdev = sqrtf(var);

        out_main_tm[idx] = reg;
        out_up_tm[idx]   = reg + ndev * stdev;
        out_lo_tm[idx]   = reg - ndev * stdev;
    }
}

}


// ===========================================================================
// NEOETHOS f64 LANE  --  closer 4, round 3
//
// CPU reference: prb_scalar (src/indicators/prb.rs:938-1108), reached through
// prb_with_kernel (:1378) -> prb_compute_into, with ssf_filter (:562) as the
// smoothing stage and lu_decomposition (:612) as the solver.
//
// OUTPUT: the `values` column -- compute_prb_batch (cpu_batch.rs:15857)
// resolves output_id == "value" to out.values.
//
// PERIOD-INVARIANT: that batch reads smooth_data (true), smooth_period (10),
// regression_period (100), polynomial_order (2), regression_offset (0), ndev
// (2.0) and equ_from (0) and NEVER `period` (cpu_batch.rs:15833-15839). A
// sweep of five periods gets five identical CPU columns, so this kernel
// writes five identical rows. Every one of those seven is a #define below,
// which is also why the design matrix is a fixed 3x3 and no allocation
// depends on a caller value -- NEVER-OOM by construction.
//
// SHAPE: one thread per combo walking bars ASCENDING. The super-smoother is a
// 2-pole IIR; the regression moments are ROLLED with the binomial shift at
// :1096-1103 rather than rebuilt, so their accumulation order is load-bearing.
// The ssf output is produced inside the same ascending walk and kept in a
// ring of REGRESSION_PERIOD + 1 entries -- the window plus the one bar ahead
// the roll consumes -- so it is never materialised for the whole series.
//
// EPSILON: the 1e-10 singular-matrix guard (:620, :645) is the CPU's own and
// is already f64-sized -- the normal-equation diagonal here is O(n^4) -- so it
// is carried across unchanged rather than rescaled from an f32 constant.
// ===========================================================================

#define PRB_NEO_SMOOTH_PERIOD 10
#define PRB_NEO_REG_PERIOD 100
#define PRB_NEO_ORDER 2
#define PRB_NEO_M 3           /* polynomial_order + 1 */
#define PRB_NEO_MAX_POW 4     /* 2 * polynomial_order */
#define PRB_NEO_OFFSET 0
#define PRB_NEO_EQU_FROM 0
#define PRB_NEO_RING (PRB_NEO_REG_PERIOD + 1)

static __forceinline__ __device__ double prb_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__
void prb_neo_batch_f64(const double* __restrict__ data,
                       int n,
                       const int* __restrict__ periods,
                       int n_combos,
                       int first_valid,
                       double* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;
    if (n <= 0) return;
    (void)periods;   // PERIOD-INVARIANT -- see the header.

    double* __restrict__ row = out + (size_t)combo * (size_t)n;
    const double nn = prb_neo_qnan();

    const int reg_n = PRB_NEO_REG_PERIOD;
    const int k = PRB_NEO_ORDER;
    const int m = PRB_NEO_M;
    const double ndev = 2.0;

    // prb_with_kernel, :1385-1388 -- `!is_nan` over the single close series,
    // which is F64FirstValidRule::AllInputsNonNan for CloseSlice.
    int first = first_valid;
    if (first < 0) first = 0;

    bool refused = false;
    if (first >= n) refused = true;
    if (reg_n <= 0 || reg_n > n) refused = true;            // :1410
    long long warm_ll = (long long)first + (long long)reg_n - 1 + PRB_NEO_EQU_FROM;
    if (!refused && warm_ll >= (long long)n) refused = true; // :1418

    if (refused) {
        for (int i = 0; i < n; ++i) row[i] = nn;
        return;
    }

    const int warmup = (int)warm_ll;
    for (int i = 0; i < n; ++i) row[i] = nn;

    // ------------------------------------------------------- the fixed design
    // :1013-1020 -- sx[p] = sum over j=1..n of j^p, built by repeated multiply
    // in exactly that order.
    double sx[PRB_NEO_MAX_POW + 1];
    for (int p = 0; p <= PRB_NEO_MAX_POW; ++p) sx[p] = 0.0;
    for (int j = 1; j <= reg_n; ++j) {
        const double jf = (double)j;
        double pwr = 1.0;
        sx[0] += 1.0;
        for (int p = 1; p <= PRB_NEO_MAX_POW; ++p) {
            pwr *= jf;
            sx[p] += pwr;
        }
    }

    double A[PRB_NEO_M * PRB_NEO_M];
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            A[i * m + j] = sx[i + j];

    // lu_decomposition, :612-654, including both singular refusals.
    double L[PRB_NEO_M * PRB_NEO_M];
    double U[PRB_NEO_M * PRB_NEO_M];
    for (int i = 0; i < m * m; ++i) { L[i] = 0.0; U[i] = 0.0; }
    for (int j = 0; j < m; ++j) U[j] = A[j];
    if (fabs(U[0]) < 1e-10) {
        return;   // SingularMatrix -- the CPU returns Err and no column at all
    }
    for (int i = 1; i < m; ++i) L[i * m] = A[i * m] / U[0];
    for (int i = 0; i < m; ++i) L[i * m + i] = 1.0;
    for (int i = 1; i < m; ++i) {
        for (int j = i; j < m; ++j) {
            double sum = 0.0;
            for (int kk = 0; kk < i; ++kk) sum += L[i * m + kk] * U[kk * m + j];
            U[i * m + j] = A[i * m + j] - sum;

            if (j > i) {
                double sum2 = 0.0;
                for (int kk = 0; kk < i; ++kk) sum2 += L[j * m + kk] * U[kk * m + i];
                if (fabs(U[i * m + i]) < 1e-10) return;
                L[j * m + i] = (A[j * m + i] - sum2) / U[i * m + i];
            }
        }
    }

    // :1029-1043 -- Pascal's triangle and the powers of n used by the shift.
    double binom[PRB_NEO_M * PRB_NEO_M];
    for (int i = 0; i < m * m; ++i) binom[i] = 0.0;
    for (int r = 0; r <= k; ++r) {
        const int r_off = r * m;
        binom[r_off + 0] = 1.0;
        binom[r_off + r] = 1.0;
        for (int c = 1; c < r; ++c) {
            const int prev = (r - 1) * m;
            binom[r_off + c] = binom[prev + (c - 1)] + binom[prev + c];
        }
    }
    double n_pow[PRB_NEO_M];
    n_pow[0] = 1.0;
    const double n_f = (double)reg_n;
    for (int r = 1; r <= k; ++r) n_pow[r] = n_pow[r - 1] * n_f;

    const double x_pos = n_f - (double)PRB_NEO_OFFSET + (double)PRB_NEO_EQU_FROM;
    const double inv_n = 1.0 / n_f;

    // ------------------------------------------------- the super-smoother
    // ssf_filter, :570-576. `a`, `b`, `c1..c3` are the CPU's own spellings.
    const double sp = (double)PRB_NEO_SMOOTH_PERIOD;
    const double omega = 2.0 * M_PI / sp;
    const double sqrt2 = 1.4142135623730951;   // std::f64::consts::SQRT_2
    const double ssf_a = exp(-sqrt2 * M_PI / sp);
    const double ssf_b = 2.0 * ssf_a * cos((sqrt2 / 2.0) * omega);
    const double c3 = -ssf_a * ssf_a;
    const double c2 = ssf_b;
    const double c1 = 1.0 - c2 - c3;

    double ring[PRB_NEO_RING];
    double y1, y2;
    bool ssf_phase2 = false;

    {
        const double x0 = data[first];
        const double y0 = c1 * x0 + c2 * x0 + c3 * x0;   // :579
        ring[first % PRB_NEO_RING] = y0;
        y1 = y0;
        y2 = y0;
    }

    // One ssf step at absolute index `idx`, reproducing the two-phase loop at
    // :583-609: the first phase runs until the first non-finite bar and takes
    // it as well, after which every later bar takes the NaN-guarded form.
    #define PRB_NEO_SSF_STEP(idx)                                            \
        do {                                                                 \
            const double xi = data[(idx)];                                   \
            double yv;                                                       \
            if (!ssf_phase2) {                                               \
                yv = c1 * xi + c2 * y1 + c3 * y2;                            \
                if (!isfinite(xi)) ssf_phase2 = true;                        \
            } else {                                                         \
                const double prev1 = isnan(y1) ? xi : y1;                    \
                const double prev2 = isnan(y2) ? prev1 : y2;                 \
                yv = c1 * xi + c2 * prev1 + c3 * prev2;                      \
            }                                                                \
            ring[(idx) % PRB_NEO_RING] = yv;                                 \
            y2 = y1;                                                         \
            y1 = yv;                                                         \
        } while (0)

    for (int idx = first + 1; idx <= warmup; ++idx) PRB_NEO_SSF_STEP(idx);

    // -------------------------------------------------------- the seed window
    // :1076-1093 -- start = warmup + 1 - n - equ_from, which is `first`.
    int start = warmup + 1 - reg_n - PRB_NEO_EQU_FROM;
    double s_xy[PRB_NEO_M];
    for (int r = 0; r < m; ++r) s_xy[r] = 0.0;
    double sum = 0.0, sumsq = 0.0;
    for (int t = 0; t < reg_n; ++t) {
        const double y = ring[(start + t) % PRB_NEO_RING];
        sum += y;
        sumsq += y * y;

        const double jf = (double)t + 1.0;
        s_xy[0] += y;
        double w = jf;
        for (int p = 1; p <= k; ++p) {
            s_xy[p] = fma(y, w, s_xy[p]);
            w *= jf;
        }
    }

    double tmp_y[PRB_NEO_M];
    double coeffs[PRB_NEO_M];
    double s_prev[PRB_NEO_M];
    for (int r = 0; r < m; ++r) { tmp_y[r] = 0.0; coeffs[r] = 0.0; s_prev[r] = 0.0; }

    for (int i = warmup; i < n; ++i) {
        // :1043-1058 -- forward then back substitution, subtracting term by
        // term in index order.
        for (int r = 0; r < m; ++r) {
            double acc = s_xy[r];
            const int rowo = r * m;
            for (int c = 0; c < r; ++c) acc -= L[rowo + c] * tmp_y[c];
            tmp_y[r] = acc / L[rowo + r];
        }
        for (int r = m - 1; r >= 0; --r) {
            const int rowo = r * m;
            double acc = tmp_y[r];
            for (int c = r + 1; c < m; ++c) acc -= U[rowo + c] * coeffs[c];
            coeffs[r] = acc / U[rowo + r];
        }

        // :1060-1063 -- Horner, highest order first, one fma per term.
        double reg = 0.0;
        for (int p = m - 1; p >= 0; --p) reg = fma(reg, x_pos, coeffs[p]);

        const double mean = sum * inv_n;
        const double var = (sumsq * inv_n) - mean * mean;
        const double stdev = (var > 0.0) ? sqrt(var) : 0.0;

        row[i] = reg;
        (void)ndev; (void)stdev;   // the bands are not this lane's columns

        if (i + 1 == n) break;                    // :1078
        const int y_new_idx = start + reg_n;
        if (y_new_idx >= n) break;                // :1082

        PRB_NEO_SSF_STEP(y_new_idx);              // smoothed[i + 1]

        const double y_old = ring[start % PRB_NEO_RING];
        const double y_new = ring[y_new_idx % PRB_NEO_RING];

        for (int r = 0; r < m; ++r) s_prev[r] = s_xy[r];

        s_xy[0] = s_prev[0] - y_old + y_new;
        sum = sum - y_old + y_new;
        sumsq = sumsq - y_old * y_old + y_new * y_new;

        // :1096-1103 -- the binomial shift that re-centres the moments.
        for (int r = 1; r <= k; ++r) {
            const int rowo = r * m;
            double acc = 0.0;
            for (int m2 = 0; m2 <= r; ++m2) {
                const double sign = (((r - m2) & 1) == 1) ? -1.0 : 1.0;
                acc += sign * binom[rowo + m2] * s_prev[m2];
            }
            s_xy[r] = acc + n_pow[r] * y_new;
        }

        start += 1;
    }

    #undef PRB_NEO_SSF_STEP
}
