#include <cuda_runtime.h>
#include <math_constants.h>

/*
 * TA-Lib OBV is a strict recurrence with lookback zero:
 *   seed = volume[first_valid]
 *   close rises  -> accumulator += current volume
 *   close falls  -> accumulator -= current volume
 *   otherwise    -> carry accumulator
 *
 * A parallel prefix scan changes IEEE-754 association and is not the same
 * indicator. The f32 APIs therefore use a double sequential accumulator and
 * downcast only each public f32 output. Prefix NaNs remain parallel writes.
 */
extern "C" __global__
void obv_batch_f32_serial_ref(
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out)
{
    const int combo = blockIdx.y;
    if (combo >= n_combos || series_len <= 0) return;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int fv = first_valid < 0 ? 0 : first_valid;
    const int base = combo * series_len;

    for (int i = tid; i < fv && i < series_len; i += stride) {
        out[base + i] = CUDART_NAN_F;
    }

    if (tid == 0 && fv < series_len) {
        double prev_obv = (double)volume[fv];
        float prev_close = close[fv];
        out[base + fv] = (float)prev_obv;

        for (int i = fv + 1; i < series_len; ++i) {
            const float c = close[i];
            const double v = (double)volume[i];
            if (c > prev_close) {
                prev_obv += v;
            } else if (c < prev_close) {
                prev_obv -= v;
            }
            out[base + i] = (float)prev_obv;
            prev_close = c;
        }
    }
}

extern "C" __global__
void obv_many_series_one_param_time_major_f32(
    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    const int* __restrict__ first_valids,
    int cols,
    int rows,
    float* __restrict__ out_tm)
{
    const int series = blockIdx.x * blockDim.x + threadIdx.x;
    if (series >= cols || rows <= 0) return;

    const int fv = first_valids[series] < 0 ? 0 : first_valids[series];
    for (int row = 0; row < rows && row < fv; ++row) {
        out_tm[row * cols + series] = CUDART_NAN_F;
    }
    if (fv >= rows) return;

    const int idx0 = fv * cols + series;
    double prev_obv = (double)volume_tm[idx0];
    float prev_close = close_tm[idx0];
    out_tm[idx0] = (float)prev_obv;

    for (int row = fv + 1; row < rows; ++row) {
        const int idx = row * cols + series;
        const float c = close_tm[idx];
        const double v = (double)volume_tm[idx];
        if (c > prev_close) {
            prev_obv += v;
        } else if (c < prev_close) {
            prev_obv -= v;
        }
        out_tm[idx] = (float)prev_obv;
        prev_close = c;
    }
}

/*
 * Native f64 batch route. `periods` is accepted only because the shared
 * batch ABI is parameter-shaped; OBV itself is period-invariant.
 */
#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void obv_neo_batch_f64(
    const double* __restrict__ close,
    const double* __restrict__ volume,
    int series_len,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;
    (void)periods;

    double* __restrict__ row = out + (size_t)combo * (size_t)series_len;
    if (first_valid < 0 || first_valid >= series_len) {
        for (int i = 0; i < series_len; ++i) row[i] = NEO_F64_NAN;
        return;
    }

    for (int i = 0; i < first_valid; ++i) row[i] = NEO_F64_NAN;

    double prev_obv = volume[first_valid];
    double prev_close = close[first_valid];
    row[first_valid] = prev_obv;

    for (int i = first_valid + 1; i < series_len; ++i) {
        const double c = close[i];
        const double v = volume[i];
        if (c > prev_close) {
            prev_obv += v;
        } else if (c < prev_close) {
            prev_obv -= v;
        }
        row[i] = prev_obv;
        prev_close = c;
    }
}
