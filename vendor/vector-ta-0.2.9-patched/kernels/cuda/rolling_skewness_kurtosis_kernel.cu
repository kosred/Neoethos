#include <cmath>
#include <cstdint>

extern "C" __global__ void rolling_skewness_kurtosis_batch_f64(
    const double* data,
    int len,
    const int* lengths,
    const int* smooth_lengths,
    int rows,
    int max_length,
    int max_smooth_length,
    double* source_buffer,
    double* skew_buffer,
    double* kurt_buffer,
    double* out_skewness,
    double* out_kurtosis
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    int length = lengths[row];
    int smooth_length = smooth_lengths[row];
    if (length <= 0 || smooth_length <= 0) {
        return;
    }

    const double nan = NAN;
    double* source_ring = source_buffer + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* skew_ring = skew_buffer + static_cast<size_t>(row) * static_cast<size_t>(max_smooth_length);
    double* kurt_ring = kurt_buffer + static_cast<size_t>(row) * static_cast<size_t>(max_smooth_length);
    double* row_skew = out_skewness + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_kurt = out_kurtosis + static_cast<size_t>(row) * static_cast<size_t>(len);

    int source_head = 0;
    int source_count = 0;
    int skew_head = 0;
    int skew_count = 0;
    int kurt_head = 0;
    int kurt_count = 0;
    double skew_sum = 0.0;
    double kurt_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            source_head = 0;
            source_count = 0;
            skew_head = 0;
            skew_count = 0;
            kurt_head = 0;
            kurt_count = 0;
            skew_sum = 0.0;
            kurt_sum = 0.0;
            row_skew[i] = nan;
            row_kurt[i] = nan;
            continue;
        }

        source_ring[source_head] = value;
        source_head += 1;
        if (source_head == length) {
            source_head = 0;
        }
        if (source_count < length) {
            source_count += 1;
        }
        if (source_count < length) {
            row_skew[i] = nan;
            row_kurt[i] = nan;
            continue;
        }

        double n = static_cast<double>(length);
        double mean = 0.0;
        for (int j = 0; j < length; ++j) {
            mean += source_ring[j];
        }
        mean /= n;

        double m2 = 0.0;
        double m3 = 0.0;
        double m4 = 0.0;
        for (int j = 0; j < length; ++j) {
            double dev = source_ring[j] - mean;
            double dev2 = dev * dev;
            m2 += dev2;
            m3 += dev2 * dev;
            m4 += dev2 * dev2;
        }
        m2 /= n;
        if (!isfinite(m2) || m2 <= 2.2204460492503131e-16) {
            skew_head = 0;
            skew_count = 0;
            kurt_head = 0;
            kurt_count = 0;
            skew_sum = 0.0;
            kurt_sum = 0.0;
            row_skew[i] = nan;
            row_kurt[i] = nan;
            continue;
        }

        double sigma = sqrt(m2);
        double skew_raw = (m3 / n) / (sigma * sigma * sigma);
        double kurt_raw = (m4 / n) / (m2 * m2) - 3.0;

        if (skew_count == smooth_length) {
            skew_sum -= skew_ring[skew_head];
        } else {
            skew_count += 1;
        }
        skew_ring[skew_head] = skew_raw;
        skew_sum += skew_raw;
        skew_head += 1;
        if (skew_head == smooth_length) {
            skew_head = 0;
        }

        if (kurt_count == smooth_length) {
            kurt_sum -= kurt_ring[kurt_head];
        } else {
            kurt_count += 1;
        }
        kurt_ring[kurt_head] = kurt_raw;
        kurt_sum += kurt_raw;
        kurt_head += 1;
        if (kurt_head == smooth_length) {
            kurt_head = 0;
        }

        if (skew_count == smooth_length && kurt_count == smooth_length) {
            row_skew[i] = skew_sum / static_cast<double>(smooth_length);
            row_kurt[i] = kurt_sum / static_cast<double>(smooth_length);
        } else {
            row_skew[i] = nan;
            row_kurt[i] = nan;
        }
    }
}

// ===========================================================================
// NEOETHOS f64 LANE  --  closer 4
//
// CPU reference:
//   * arithmetic   : RollingSkewnessKurtosisStream::update,
//                    src/indicators/rolling_skewness_kurtosis.rs:349-409, and
//                    its unrolled twin compute_row_50_3_all_finite (:517-601),
//                    which is the path the defaults take. The two are
//                    arithmetically identical -- the unrolled one is the stream
//                    with length = 50 and smooth_length = 3 written out -- so
//                    this kernel implements the stream and gets both.
//   * refusals     : validate_common, :460-481.
//   * warmup       : warmup_prefix = length + smooth_length - 2 = 51
//                    (:417-426), applied by alloc_with_nan_prefix (:663).
//   * FIRST-VALID IGNORED. The CPU NEVER consults a first-valid index: it walks
//                    from index 0 and RESETS the whole accumulator on any
//                    non-finite bar (:350-353). Declaring AllInputsNonNan would
//                    be a claim the kernel does not honour, so the row is
//                    F64FirstValidRule::Ignored and the argument is discarded.
//   * emitted column: `skewness`. compute_rolling_skewness_kurtosis_batch
//                    (cpu_batch.rs:7978) accepts ONLY "skewness" and "kurtosis"
//                    -- there is no "value" arm (:8007-8015) -- so a parity run
//                    must ask the CPU for "skewness" explicitly.
//   * PERIOD-INVARIANT: the batch reads `length` (50) and `smooth_length` (3)
//                    and never `period` (cpu_batch.rs:7990-7992).
//
// EPSILON: `m2 <= f64::EPSILON` (:340 / :573) is ALREADY an f64-width constant
// on the host -- 2.220446049250313e-16 -- so it is carried across as
// DBL_EPSILON and NOT as any f32 value. This is the one guard in the file and
// re-deriving it means writing the f64 machine epsilon, which is what
// DBL_EPSILON is.
//
// SEQUENTIAL, one thread per column: the window sum, both SMA rings and the
// reset-on-non-finite state all carry across bars.
// ===========================================================================

#include <float.h>

#define RSK_NEO_LENGTH 50
#define RSK_NEO_SMOOTH 3

static __device__ __forceinline__ double rsk_neo_qnan() {
  return __longlong_as_double(0x7ff8000000000000ULL);
}

// SmaState::update, rolling_skewness_kurtosis.rs:285-309, and its unrolled twin
// update_sma3 (:484-514). Returns false while the window is still filling.
static __device__ __forceinline__ bool rsk_neo_sma_update(double value,
                                                          double* buf,
                                                          int* head,
                                                          int* count,
                                                          double* sum,
                                                          int period,
                                                          double* outv) {
  if (*count < period) {
    buf[*head] = value;
    *head += 1;
    if (*head == period) *head = 0;
    *count += 1;
    *sum += value;
    if (*count < period) return false;
    *outv = *sum / (double)period;
    return true;
  }
  const double old = buf[*head];
  buf[*head] = value;
  *head += 1;
  if (*head == period) *head = 0;
  *sum += value - old;
  *outv = *sum / (double)period;
  return true;
}

extern "C" __global__
void rolling_skewness_kurtosis_neo_batch_f64(const double* __restrict__ prices,
                                             int n,
                                             const int* __restrict__ periods,
                                             int n_combos,
                                             int first_valid,
                                             double* __restrict__ out) {
  const int combo = blockIdx.x * blockDim.x + threadIdx.x;
  if (combo >= n_combos) return;
  (void)periods;      // PERIOD-INVARIANT -- see the header.
  (void)first_valid;  // FIRST-VALID IGNORED -- see the header.

  if (n <= 0) return;
  double* __restrict__ row = out + (size_t)combo * (size_t)n;
  const double nn = rsk_neo_qnan();

  const int length = RSK_NEO_LENGTH;
  const int smooth = RSK_NEO_SMOOTH;
  const int needed = length + smooth - 1;      // warmup_needed, :417-420
  const int prefix = needed - 1;               // warmup_prefix,  :422-426
  (void)prefix;

  // The whole row starts NaN. That covers BOTH CPU branches: compute_row fills
  // everything with NaN when the series is not all finite (:625) and
  // alloc_with_nan_prefix + the explicit NaN writes cover the all-finite case,
  // where every index from `prefix` on is written with a value or with NaN.
  for (int i = 0; i < n; ++i) row[i] = nn;

  // validate_common, :460-481: empty, length > len, longest finite run == 0, or
  // shorter than `needed` -- every one of them is an Err, and an Err means the
  // caller gets no column at all.
  if (length > n) return;
  int best_run = 0, cur_run = 0;
  for (int i = 0; i < n; ++i) {
    if (isfinite(prices[i])) { cur_run += 1; if (cur_run > best_run) best_run = cur_run; }
    else cur_run = 0;
  }
  if (best_run == 0 || best_run < needed) return;

  double win[RSK_NEO_LENGTH];
  double skew_buf[RSK_NEO_SMOOTH], kurt_buf[RSK_NEO_SMOOTH];
  int head = 0, count = 0;
  double sum1 = 0.0;
  int skew_head = 0, kurt_head = 0, skew_count = 0, kurt_count = 0;
  double skew_sum = 0.0, kurt_sum = 0.0;
  for (int k = 0; k < length; ++k) win[k] = 0.0;
  for (int k = 0; k < smooth; ++k) { skew_buf[k] = 0.0; kurt_buf[k] = 0.0; }

  const double nf = (double)length;

  for (int i = 0; i < n; ++i) {
    const double value = prices[i];

    // :350-353 -- a non-finite bar resets the WHOLE accumulator and emits
    // nothing. This is why there is no first-valid index to honour.
    if (!isfinite(value)) {
      head = 0; count = 0; sum1 = 0.0;
      skew_head = 0; skew_count = 0; skew_sum = 0.0;
      kurt_head = 0; kurt_count = 0; kurt_sum = 0.0;
      continue;
    }

    if (count == length) {
      sum1 += value - win[head];
    } else {
      count += 1;
      sum1 += value;
    }
    win[head] = value;
    head += 1;
    if (head == length) head = 0;

    if (count < length) continue;

    const double mean = sum1 / nf;
    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    // SLOT ORDER, not chronological order: the CPU iterates `source_buf` as an
    // array (:333 / :557), so the ring rotation IS the summation order and a
    // chronological walk would round differently.
    for (int k = 0; k < length; ++k) {
      const double dev = win[k] - mean;
      const double dev2 = dev * dev;
      m2 += dev2;
      m3 += dev2 * dev;
      m4 += dev2 * dev2;
    }
    m2 /= nf;
    if (!isfinite(m2) || m2 <= DBL_EPSILON) {
      // :341-346 / :570-580 -- only the two SMA states reset here, never the
      // window.
      skew_head = 0; skew_count = 0; skew_sum = 0.0;
      kurt_head = 0; kurt_count = 0; kurt_sum = 0.0;
      continue;
    }
    const double sigma = sqrt(m2);
    const double skew_raw = (m3 / nf) / (sigma * sigma * sigma);
    const double kurt_raw = (m4 / nf) / (m2 * m2) - 3.0;

    double sk = 0.0, ku = 0.0;
    const bool have_sk =
        rsk_neo_sma_update(skew_raw, skew_buf, &skew_head, &skew_count, &skew_sum, smooth, &sk);
    const bool have_ku =
        rsk_neo_sma_update(kurt_raw, kurt_buf, &kurt_head, &kurt_count, &kurt_sum, smooth, &ku);
    // Both or neither -- :404-408 returns None unless BOTH are Some. No
    // `i >= prefix` guard here, because the CPU has none (:635): Some cannot
    // occur before index `prefix` anyway, since it needs `length` window bars
    // and then `smooth` more.
    if (have_sk && have_ku) row[i] = sk;
  }
}
