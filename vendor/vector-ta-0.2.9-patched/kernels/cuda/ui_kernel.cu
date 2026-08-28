#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>
#include "ds_float2.cuh"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif


static __forceinline__ __device__ void neumaier_add(float y, float &sum, float &c) {
    float t = sum + y;
    if (fabsf(sum) >= fabsf(y)) c += (sum - t) + y; else c += (y - t) + sum;
    sum = t;
}


static __forceinline__ __device__ void kahan_add(float y, float &sum, float &c) {
    float y2 = y - c;
    float t  = sum + y2;
    c        = (t - sum) - y2;
    sum      = t;
}


extern "C" __global__ void ui_single_series_f32(
    const float* __restrict__ prices,
    int series_len,
    int first_valid,
    int period,
    float* __restrict__ out)
{
    if (series_len <= 0 || period <= 0) return;
    if (blockIdx.x != 0 || threadIdx.x != 0) return;


    extern __shared__ __align__(16) unsigned char shraw[];
    unsigned char* base = shraw;
    const int p = period;

    int* deq_idx = reinterpret_cast<int*>(base);

    size_t off = static_cast<size_t>(p) * sizeof(int);
    const size_t a = sizeof(double) - 1;
    off = (off + a) & ~a;

    double* sq_ring = reinterpret_cast<double*>(base + off);

    unsigned char* valid_ring = reinterpret_cast<unsigned char*>(base + off + static_cast<size_t>(p) * sizeof(double));

    const int fv = first_valid < 0 ? 0 : first_valid;
    const int warm_end = fv + (2 * p - 2);


    for (int i = 0; i < p; ++i) { sq_ring[i] = 0.0; valid_ring[i] = 0u; }

    const int warm_write = (warm_end < series_len) ? warm_end : series_len;
    for (int i = 0; i < warm_write; ++i) out[i] = CUDART_NAN_F;

    int head = 0, tail = 0, dsize = 0;
    int ring_idx = 0;
    double sum = 0.0;
    int count = 0;

    for (int i = fv; i < series_len; ++i) {
        const int start = (i + 1 >= p) ? (i + 1 - p) : 0;


        while (dsize != 0 && deq_idx[head] < start) {
            head = (head + 1); if (head == p) head = 0; dsize--;
        }

        const float xi = prices[i];
        const bool xi_finite = isfinite(xi);
        if (xi_finite) {

            while (dsize != 0) {
                int back = (tail == 0) ? (p - 1) : (tail - 1);
                const int j = deq_idx[back];
                const float xj = prices[j];
                if (xj <= xi) { tail = back; dsize--; }
                else break;
            }

            deq_idx[tail] = i;
            tail += 1; if (tail == p) tail = 0; dsize++;
        }


        unsigned char new_valid = 0u;
        float new_sq = 0.0f;
        if (i + 1 >= fv + p && dsize != 0) {
            const float m = prices[deq_idx[head]];
            if (xi_finite && isfinite(m) && fabsf(m) > 1e-20f) {
                const double dd = (static_cast<double>(xi) - static_cast<double>(m)) / static_cast<double>(m);
                new_sq = static_cast<float>(dd * dd);
                new_valid = 1u;
            }
        }


        if (new_valid)             { sum += (double)new_sq; count++; }
        if (valid_ring[ring_idx])  { sum -= sq_ring[ring_idx]; count--; }
        sq_ring[ring_idx] = (double)new_sq; valid_ring[ring_idx] = new_valid;
        ring_idx += 1; if (ring_idx == p) ring_idx = 0;


        if (i >= warm_end) {
            if (count == p) {
                double avg_d = sum / (double)p;
                if (avg_d < 0.0) avg_d = 0.0;
                out[i] = static_cast<float>(sqrt(avg_d));
            } else {
                out[i] = CUDART_NAN_F;
            }
        }
    }
}


extern "C" __global__ void ui_scale_rows_from_base_f32(
    const float* __restrict__ base,
    const float* __restrict__ scalars,
    int series_len,
    int n_rows,
    float* __restrict__ out)
{
    const int row = blockIdx.y;
    if (row >= n_rows) return;
    const float s = fabsf(scalars[row]);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    float* dst = out + row * series_len;
    for (int i = tid; i < series_len; i += stride) {
        const float v = base[i];
        dst[i] = static_cast<float>(static_cast<double>(v) * static_cast<double>(s));
    }
}


extern "C" __global__ void ui_many_series_one_param_time_major_f32(
    const float* __restrict__ prices_tm,
    const int*   __restrict__ first_valids,
    int cols,
    int rows,
    int period,
    float scalar,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.x;
    if (s >= cols || rows <= 0 || period <= 0) return;
    if (threadIdx.x != 0) return;


    extern __shared__ __align__(16) unsigned char shraw[];
    unsigned char* base = shraw;
    const int p = period;
    int* deq_idx = reinterpret_cast<int*>(base);
    size_t off = static_cast<size_t>(p) * sizeof(int);
    const size_t a = sizeof(double) - 1;
    off = (off + a) & ~a;
    float* deq_val = reinterpret_cast<float*>(base + off);
    float* sq_ring = reinterpret_cast<float*>(base + off + static_cast<size_t>(p) * sizeof(float));
    unsigned char* valid_ring = reinterpret_cast<unsigned char*>(base + off + static_cast<size_t>(p) * sizeof(double));

    const int fv = first_valids[s] < 0 ? 0 : first_valids[s];
    const int warm_end = fv + (2 * p - 2);
    for (int i = 0; i < p; ++i) { sq_ring[i] = 0.0f; valid_ring[i] = 0u; }
    for (int t = 0; t < rows && t < warm_end; ++t) { out_tm[t * cols + s] = CUDART_NAN_F; }

    int head = 0, tail = 0, dsize = 0;
    int ring_idx = 0;
    float sum = 0.0f, comp = 0.0f;
    int count = 0;
    const float s_abs = fabsf(scalar);

    for (int t = fv; t < rows; ++t) {
        const int start = (t + 1 >= p) ? (t + 1 - p) : 0;

        while (dsize != 0 && deq_idx[head] < start) {
            head = (head + 1); if (head == p) head = 0; dsize--;
        }
        const int idx = t * cols + s;
        const float xi = prices_tm[idx];
        const bool xi_finite = isfinite(xi);
        if (xi_finite) {
            while (dsize != 0) {
                int back = (tail == 0) ? (p - 1) : (tail - 1);
                const float xj = deq_val[back];
                if (xj <= xi) { tail = back; dsize--; } else { break; }
            }
            deq_idx[tail] = t; deq_val[tail] = xi;
            tail += 1; if (tail == p) tail = 0; dsize++;
        }

        unsigned char new_valid = 0u; float new_sq = 0.0f;
        if (t + 1 >= fv + p && dsize != 0) {
            const float m = deq_val[head];
            if (xi_finite && isfinite(m) && fabsf(m) > 1e-20f) {
                const float dd = (xi - m) / m;
                new_sq = dd * dd;
                new_valid = 1u;
            }
        }
        if (valid_ring[ring_idx]) { neumaier_add(-sq_ring[ring_idx], sum, comp); count--; }
        if (new_valid)             { neumaier_add( new_sq,               sum, comp); count++; }
        sq_ring[ring_idx] = new_sq; valid_ring[ring_idx] = new_valid;
        ring_idx += 1; if (ring_idx == p) ring_idx = 0;

        if (t >= warm_end) {
            if (count == p) {
                float avg = (sum + comp) / (float)p;
                if (avg < 0.0f) avg = 0.0f;
                out_tm[idx] = sqrtf(avg) * s_abs;
            } else {
                out_tm[idx] = CUDART_NAN_F;
            }
        }
    }
}


extern "C" __global__ void ui_one_series_many_params_f32(
    const float* __restrict__ prices,
    int series_len,
    const int*  __restrict__ periods,
    const float* __restrict__ scalars,
    int n_params,
    int first_valid,
    int max_period,
    float* __restrict__ out)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    int param_id = blockIdx.x * warps_per_block + warp;
    if (param_id >= n_params) return;


    extern __shared__ __align__(16) unsigned char shraw[];
    unsigned char* base = shraw;

    size_t stride_i = (size_t)warps_per_block * (size_t)max_period;
    int*   deq_idx_base = reinterpret_cast<int*>(base);
    size_t off = stride_i * sizeof(int);
    const size_t a = sizeof(double) - 1;
    off = (off + a) & ~a;
    float* deq_val_base = reinterpret_cast<float*>(base + off);
    float* sq_ring_base = reinterpret_cast<float*>(base + off + stride_i * sizeof(float));
    unsigned char* valid_base = reinterpret_cast<unsigned char*>(base + off + stride_i * sizeof(double));


    int*   deq_idx = deq_idx_base + warp * max_period;
    float* deq_val = deq_val_base + warp * max_period;
    float* sq_ring = sq_ring_base + warp * max_period;
    unsigned char* valid_ring = valid_base + warp * max_period;

    const int p = periods[param_id];
    if (p <= 0 || p > max_period) return;
    const int fv = first_valid < 0 ? 0 : first_valid;
    const int warm_end = fv + (2 * p - 2);


    for (int k = lane; k < p; k += WARP_SIZE) { sq_ring[k] = 0.0f; valid_ring[k] = 0u; }

    float* out_row = out + (size_t)param_id * (size_t)series_len;
    for (int i = lane; i < series_len && i < warm_end; i += WARP_SIZE) { out_row[i] = CUDART_NAN_F; }
    __syncwarp();


    if (lane == 0) {
        int head = 0, tail = 0, dsize = 0;
        int ring_idx = 0;
        float sum = 0.0f, comp = 0.0f;
        int count = 0;

        for (int i = fv; i < series_len; ++i) {
            const int start = (i + 1 >= p) ? (i + 1 - p) : 0;
            while (dsize != 0 && deq_idx[head] < start) { head = (head + 1); if (head == p) head = 0; dsize--; }

            const float xi = prices[i];
            const bool xi_finite = isfinite(xi);
            if (xi_finite) {
                while (dsize != 0) {
                    int back = (tail == 0) ? (p - 1) : (tail - 1);
                    const float xj = deq_val[back];
                    if (xj <= xi) { tail = back; dsize--; } else break;
                }
                deq_idx[tail] = i; deq_val[tail] = xi;
                tail += 1; if (tail == p) tail = 0; dsize++;
            }

            unsigned char new_valid = 0u; float new_sq = 0.0f;
            if (i + 1 >= fv + p && dsize != 0) {
                const float m = deq_val[head];
                if (xi_finite && isfinite(m) && fabsf(m) > 1e-20f) {
                    const float dd = (xi - m) / m;
                    new_sq = dd * dd; new_valid = 1u;
                }
            }

            if (valid_ring[ring_idx]) { neumaier_add(-sq_ring[ring_idx], sum, comp); count--; }
            if (new_valid)             { neumaier_add( new_sq,               sum, comp); count++; }
            sq_ring[ring_idx] = new_sq; valid_ring[ring_idx] = new_valid;
            ring_idx += 1; if (ring_idx == p) ring_idx = 0;

            if (i >= warm_end) {
                if (count == p) {
                    float avg = sum / (float)p;
                    if (avg < 0.0f) avg = 0.0f;
                    out_row[i] = sqrtf(avg) * fabsf(scalars[param_id]);
                } else {
                    out_row[i] = CUDART_NAN_F;
                }
            }
        }
    }
}


/* ===========================================================================
 * NEOETHOS f64 LANE — ui (ulcer index)
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/ui.rs:346 `ui_scalar`. Default scalar = 100.0
 * (ui.rs:115); our lane sweeps `period` with default params.
 *
 * THE EPSILON IS AN f64 EPSILON AND IT COMES FROM THE CPU. ui.rs:425 guards
 * the drawdown divide with `m.abs() > f64::EPSILON`, i.e. 2.220446049250313e-16.
 * The f32 lane's equivalent would be FLT_EPSILON, 1.19e-7 — eleven orders of
 * magnitude larger, which silently drops every bar whose rolling maximum is a
 * small price. Spelled out below rather than copied from the f32 file.
 *
 * `ui_scalar` has a `period <= 64` variant that tracks ring validity in a u64
 * bitmask and a general variant that uses a byte ring. The ARITHMETIC is
 * identical — same `sum` add/subtract, same `count` — so one implementation
 * serves both; only the bookkeeping container differs.
 *
 * `dd.mul_add(dd, 0.0)` (ui.rs:427) is a genuine fma with a zero addend, not a
 * plain square: fma(dd, dd, 0.0) rounds ONCE, `dd*dd` also rounds once, and
 * they agree — but the fma is written because that is what the CPU emits.
 *
 * Fixed window bound: the squares ring is a running total over `period`, so it
 * needs the ring. NEO_UI_MAX_PERIOD is refused BY NAME by the host wrapper
 * rather than truncated.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Must match NEO_UI_MAX_PERIOD in neoethos_f64_wrapper.rs. */
#define NEO_UI_MAX_PERIOD 512

extern "C" __global__
void ui_neo_batch_f64(const double* __restrict__ data,
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
    const int period = periods[combo];

    for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
    if (period <= 0 || period > NEO_UI_MAX_PERIOD || period > len ||
        first_valid < 0 || first_valid >= len) return;

    const double scalar_param = 100.0;               // ui.rs:115 default
    const double F64_EPS = 2.2204460492503131e-16;   // f64::EPSILON, NOT FLT_EPSILON
    const double inv_period = 1.0 / (double)period;
    const int warmup_end = first_valid + (period * 2 - 2);

    int deq[NEO_UI_MAX_PERIOD];                      // monotonic max-deque of indices
    double sq_ring[NEO_UI_MAX_PERIOD];
    unsigned char valid_ring[NEO_UI_MAX_PERIOD];
    for (int t = 0; t < period; ++t) { sq_ring[t] = 0.0; valid_ring[t] = 0; }

    const int cap = period;
    int head = 0, tail = 0, dsize = 0;
    int ring_idx = 0;
    double sum = 0.0;
    int count = 0;

    for (int i = first_valid; i < len; ++i) {
        const int start = (i + 1 >= period) ? (i + 1 - period) : 0;

        while (dsize != 0) {
            const int j = deq[head];
            if (j < start) { head += 1; if (head == cap) head = 0; dsize -= 1; }
            else break;
        }

        const double xi = data[i];
        const bool xi_finite = isfinite(xi);
        if (xi_finite) {
            while (dsize != 0) {
                int back = tail;
                if (back == 0) back = cap - 1; else back -= 1;
                const double xj = data[deq[back]];
                if (xj <= xi) { tail = back; dsize -= 1; }
                else break;
            }
            deq[tail] = i;
            tail += 1; if (tail == cap) tail = 0;
            dsize += 1;
        }

        unsigned char new_valid = 0;
        double new_sq = 0.0;
        if (i + 1 >= first_valid + period && dsize != 0) {
            const double m = data[deq[head]];
            if (xi_finite && isfinite(m) && fabs(m) > F64_EPS) {
                const double dd = (xi - m) * (scalar_param / m);
                new_sq = fma(dd, dd, 0.0);
                new_valid = 1;
            }
        }

        if (valid_ring[ring_idx] != 0) { sum -= sq_ring[ring_idx]; count -= 1; }
        if (new_valid != 0)            { sum += new_sq;            count += 1; }
        sq_ring[ring_idx] = new_sq;
        valid_ring[ring_idx] = new_valid;

        ring_idx += 1; if (ring_idx == period) ring_idx = 0;

        if (i >= warmup_end) {
            if (count == period) {
                double avg = sum * inv_period;
                if (avg < 0.0) avg = 0.0;
                o[i] = sqrt(avg);
            } else {
                o[i] = NEO_F64_NAN;
            }
        }
    }
}
