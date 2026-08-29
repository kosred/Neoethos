#include <cmath>
#include <cstddef>

/* ===========================================================================
 * NEOETHOS f64 LANE - didi_index
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/didi_index.rs
 *             `DidiIndexStream::update` over three `SmaWindow`s.
 *
 * The canonical production outputs are short, long, crossover and
 * crossunder. The preserved primary ABI emits short at the default 3:8:20
 * tuple; the full ABI consumes the exact dynamic RegistryRatio tuple. Both
 * entry points call the same row authority below.
 *
 * Every SMA is updated unconditionally before readiness is tested. During
 * fill the exact operation is `sum += value`; afterwards it is
 * `sum += value - old`. Re-summing a source window changes f64 bits and is not
 * an alternative implementation.
 *
 * A non-finite value resets all three rings and prior-cross state. The three
 * later early exits clear only `have_prev`, leaving the prior values intact,
 * exactly like the CPU stream. One sequential thread owns each tuple.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define DIDI_NEO_SHORT   3
#define DIDI_NEO_MEDIUM  8
#define DIDI_NEO_LONG   20

__device__ __forceinline__ void didi_index_store_f64(
    int index,
    double short_value,
    double long_value,
    double crossover,
    double crossunder,
    double* __restrict__ out_short,
    double* __restrict__ out_long,
    double* __restrict__ out_crossover,
    double* __restrict__ out_crossunder)
{
    if (out_short != nullptr) out_short[index] = short_value;
    if (out_long != nullptr) out_long[index] = long_value;
    if (out_crossover != nullptr) out_crossover[index] = crossover;
    if (out_crossunder != nullptr) out_crossunder[index] = crossunder;
}

__device__ __forceinline__ void didi_index_row_f64(
    const double* __restrict__ data,
    int len,
    int short_length,
    int medium_length,
    int long_length,
    double* __restrict__ s_val,
    int short_capacity,
    double* __restrict__ m_val,
    int medium_capacity,
    double* __restrict__ l_val,
    int long_capacity,
    double* __restrict__ out_short,
    double* __restrict__ out_long,
    double* __restrict__ out_crossover,
    double* __restrict__ out_crossunder)
{
    const bool declined =
        len <= 0 ||
        short_length <= 0 || short_length > short_capacity ||
        medium_length <= 0 || medium_length > medium_capacity ||
        long_length <= 0 || long_length > long_capacity;
    if (declined) {
        for (int i = 0; i < len; ++i) {
            didi_index_store_f64(
                i,
                NEO_F64_NAN,
                NEO_F64_NAN,
                NEO_F64_NAN,
                NEO_F64_NAN,
                out_short,
                out_long,
                out_crossover,
                out_crossunder);
        }
        return;
    }

    int s_idx = 0;
    int s_cnt = 0;
    int m_idx = 0;
    int m_cnt = 0;
    int l_idx = 0;
    int l_cnt = 0;
    double s_sum = 0.0;
    double m_sum = 0.0;
    double l_sum = 0.0;
    double prev_short = NEO_F64_NAN;
    double prev_long = NEO_F64_NAN;
    bool have_prev = false;

    for (int i = 0; i < len; ++i) {
        const double value = data[i];
        if (!isfinite(value)) {
            s_idx = 0;
            s_cnt = 0;
            s_sum = 0.0;
            m_idx = 0;
            m_cnt = 0;
            m_sum = 0.0;
            l_idx = 0;
            l_cnt = 0;
            l_sum = 0.0;
            prev_short = NEO_F64_NAN;
            prev_long = NEO_F64_NAN;
            have_prev = false;
            didi_index_store_f64(
                i,
                NEO_F64_NAN,
                NEO_F64_NAN,
                NEO_F64_NAN,
                NEO_F64_NAN,
                out_short,
                out_long,
                out_crossover,
                out_crossunder);
            continue;
        }

        bool s_ok;
        bool m_ok;
        bool l_ok;
        double short_ma;
        double medium_ma;
        double long_ma;

        if (s_cnt < short_length) {
            s_val[s_idx] = value;
            s_sum += value;
            s_cnt += 1;
            s_idx += 1;
            if (s_idx == short_length) s_idx = 0;
            s_ok = s_cnt == short_length;
        } else {
            const double old = s_val[s_idx];
            s_val[s_idx] = value;
            s_sum += value - old;
            s_idx += 1;
            if (s_idx == short_length) s_idx = 0;
            s_ok = true;
        }
        short_ma = s_sum / (double)short_length;

        if (m_cnt < medium_length) {
            m_val[m_idx] = value;
            m_sum += value;
            m_cnt += 1;
            m_idx += 1;
            if (m_idx == medium_length) m_idx = 0;
            m_ok = m_cnt == medium_length;
        } else {
            const double old = m_val[m_idx];
            m_val[m_idx] = value;
            m_sum += value - old;
            m_idx += 1;
            if (m_idx == medium_length) m_idx = 0;
            m_ok = true;
        }
        medium_ma = m_sum / (double)medium_length;

        if (l_cnt < long_length) {
            l_val[l_idx] = value;
            l_sum += value;
            l_cnt += 1;
            l_idx += 1;
            if (l_idx == long_length) l_idx = 0;
            l_ok = l_cnt == long_length;
        } else {
            const double old = l_val[l_idx];
            l_val[l_idx] = value;
            l_sum += value - old;
            l_idx += 1;
            if (l_idx == long_length) l_idx = 0;
            l_ok = true;
        }
        long_ma = l_sum / (double)long_length;

        if (!s_ok || !m_ok || !l_ok) {
            have_prev = false;
            didi_index_store_f64(
                i,
                NEO_F64_NAN,
                NEO_F64_NAN,
                NEO_F64_NAN,
                NEO_F64_NAN,
                out_short,
                out_long,
                out_crossover,
                out_crossunder);
            continue;
        }
        if (!isfinite(medium_ma) || medium_ma == 0.0) {
            have_prev = false;
            didi_index_store_f64(
                i,
                NEO_F64_NAN,
                NEO_F64_NAN,
                NEO_F64_NAN,
                NEO_F64_NAN,
                out_short,
                out_long,
                out_crossover,
                out_crossunder);
            continue;
        }

        const double short_value = short_ma / medium_ma;
        const double long_value = long_ma / medium_ma;
        if (!isfinite(short_value) || !isfinite(long_value)) {
            have_prev = false;
            didi_index_store_f64(
                i,
                NEO_F64_NAN,
                NEO_F64_NAN,
                NEO_F64_NAN,
                NEO_F64_NAN,
                out_short,
                out_long,
                out_crossover,
                out_crossunder);
            continue;
        }

        const double crossover =
            have_prev && short_value > long_value && prev_short <= prev_long ? 1.0 : 0.0;
        const double crossunder =
            have_prev && short_value < long_value && prev_short >= prev_long ? 1.0 : 0.0;
        didi_index_store_f64(
            i,
            short_value,
            long_value,
            crossover,
            crossunder,
            out_short,
            out_long,
            out_crossover,
            out_crossunder);
        prev_short = short_value;
        prev_long = long_value;
        have_prev = true;
    }
}

extern "C" __global__ void didi_index_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ short_lengths,
    const int* __restrict__ medium_lengths,
    const int* __restrict__ long_lengths,
    int rows,
    double* __restrict__ short_rings,
    int short_stride,
    double* __restrict__ medium_rings,
    int medium_stride,
    double* __restrict__ long_rings,
    int long_stride,
    double* __restrict__ out_short,
    double* __restrict__ out_long,
    double* __restrict__ out_crossover,
    double* __restrict__ out_crossunder)
{
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) return;

    double* __restrict__ row_short_ring =
        short_rings + (size_t)row * (size_t)short_stride;
    double* __restrict__ row_medium_ring =
        medium_rings + (size_t)row * (size_t)medium_stride;
    double* __restrict__ row_long_ring =
        long_rings + (size_t)row * (size_t)long_stride;
    double* __restrict__ row_short = out_short + (size_t)row * (size_t)len;
    double* __restrict__ row_long = out_long + (size_t)row * (size_t)len;
    double* __restrict__ row_crossover = out_crossover + (size_t)row * (size_t)len;
    double* __restrict__ row_crossunder = out_crossunder + (size_t)row * (size_t)len;
    didi_index_row_f64(
        data,
        len,
        short_lengths[row],
        medium_lengths[row],
        long_lengths[row],
        row_short_ring,
        short_stride,
        row_medium_ring,
        medium_stride,
        row_long_ring,
        long_stride,
        row_short,
        row_long,
        row_crossover,
        row_crossunder);
}

extern "C" __global__ void didi_index_neo_batch_f64(
    const double* __restrict__ data,
    int series_len,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;
    (void)periods;
    (void)first_valid;

    double short_ring[DIDI_NEO_SHORT];
    double medium_ring[DIDI_NEO_MEDIUM];
    double long_ring[DIDI_NEO_LONG];
    double* __restrict__ row_short = out + (size_t)combo * (size_t)series_len;
    didi_index_row_f64(
        data,
        series_len,
        DIDI_NEO_SHORT,
        DIDI_NEO_MEDIUM,
        DIDI_NEO_LONG,
        short_ring,
        DIDI_NEO_SHORT,
        medium_ring,
        DIDI_NEO_MEDIUM,
        long_ring,
        DIDI_NEO_LONG,
        row_short,
        nullptr,
        nullptr,
        nullptr);
}
