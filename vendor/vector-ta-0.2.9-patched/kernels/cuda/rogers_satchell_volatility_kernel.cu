// rogers_satchell_volatility — CUDA kernels.
//
// WHAT THIS REPLACES
// ------------------
// NOTHING. Before this file there was no `.cu` for this indicator at all, and
// `src/cuda/rogers_satchell_volatility_wrapper.rs` had ZERO `get_function`
// calls: it computed `compute_rs_row` / `compute_signal_row` on the HOST
// (:136, :178) and then `DeviceBuffer::from_slice`d the host answer at :263,
// :309 and :372, handing the caller three device pointers the card had never
// written. The inventory's `fallback.py` did not report it because its filter
// required at least one `get_function`, so a wrapper that never even pretended
// fell through the check.
//
// CPU REFERENCE
// -------------
//   src/cuda/rogers_satchell_volatility_wrapper.rs
//     :165 rs_term            :178 compute_rs_row
//     :136 compute_signal_row
//
// WHY TWO KERNELS
// ---------------
// `compute_rs_row` (:178) builds PREFIX SUMS over the whole series and then
// takes window differences. The prefix arrays do NOT depend on any swept
// parameter — `lookback` and `signal_length` only choose which differences to
// take — so they are built ONCE, by `rogers_satchell_prefix_f64`, and every
// parameter row reads them.
//
// The prefix pass is deliberately SINGLE-THREADED. A parallel scan would give a
// different summation tree and therefore different last bits, and the result is
// square-rooted and compared against a volatility threshold. One thread walking
// `len` bars is microseconds against the row work that follows, so the exact
// order costs nothing worth having.
//
// The second kernel is ONE THREAD PER PARAMETER ROW: the rs series is
// window-parallel, but `compute_signal_row` (:136) carries a running sum AND a
// validity count across bars, so the row is serial.
//
// ARITHMETIC
// ----------
// f64 throughout, and the file is listed in `F64_LANE_SOURCES` so it is never
// compiled with `--use_fast_math`. The host code it replaces accumulated in f64
// and stored f32; this stores f64, which is the point.

#include <cmath>
#include <cstdint>

__device__ __forceinline__ double rsv_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// rs_term (:165). Returns 0 where the CPU returns `None`.
__device__ __forceinline__ int rsv_term(
    double open, double high, double low, double close, double* out) {
    if (!(isfinite(open) && isfinite(high) && isfinite(low) && isfinite(close))) {
        return 0;
    }
    if (open <= 0.0 || high <= 0.0 || low <= 0.0 || close <= 0.0) {
        return 0;
    }
    *out = log(high / close) * log(high / open) + log(low / close) * log(low / open);
    return 1;
}

// Prefix pass. `prefix_sum` and `prefix_valid` are `len + 1` long, exactly as
// the CPU's `vec![0.0; len + 1]` (:186-187).
extern "C" __global__ void rogers_satchell_prefix_f64(
    const double* __restrict__ open,
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    double* __restrict__ prefix_sum,
    int* __restrict__ prefix_valid
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    prefix_sum[0] = 0.0;
    prefix_valid[0] = 0;
    for (int i = 0; i < len; ++i) {
        prefix_valid[i + 1] = prefix_valid[i];
        prefix_sum[i + 1] = prefix_sum[i];
        double term;
        if (rsv_term(open[i], high[i], low[i], close[i], &term)) {
            prefix_valid[i + 1] += 1;
            prefix_sum[i + 1] += term;
        }
    }
}

// The same prefix pass over the f32-typed inputs the EXISTING wrapper API
// takes. The widening happens on the DEVICE, inside `rsv_term`, which is
// exactly where the host code did it (`rs_term`, :165, takes `f32` and returns
// `f64`). Keeping this entry point means the f32-facing callers in
// `src/indicators/rogers_satchell_volatility.rs:1433` and `:1518` keep working
// while the card, not the host, does the arithmetic.
extern "C" __global__ void rogers_satchell_prefix_f32in(
    const float* __restrict__ open,
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int len,
    double* __restrict__ prefix_sum,
    int* __restrict__ prefix_valid
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    prefix_sum[0] = 0.0;
    prefix_valid[0] = 0;
    for (int i = 0; i < len; ++i) {
        prefix_valid[i + 1] = prefix_valid[i];
        prefix_sum[i + 1] = prefix_sum[i];
        double term;
        if (rsv_term(static_cast<double>(open[i]), static_cast<double>(high[i]),
                     static_cast<double>(low[i]), static_cast<double>(close[i]), &term)) {
            prefix_valid[i + 1] += 1;
            prefix_sum[i + 1] += term;
        }
    }
}

// Narrow the f64 result to the f32 the existing API returns. The host code did
// `variance.sqrt() as f32` (:212) and `(sum / n) as f32` (:158); this is the
// same single narrowing, performed on the device so no host value is ever
// uploaded as if the card had produced it.
extern "C" __global__ void rogers_satchell_narrow_f64_to_f32(
    const double* __restrict__ src,
    float* __restrict__ dst,
    long long total
) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (; i < total; i += stride) {
        dst[i] = static_cast<float>(src[i]);
    }
}

extern "C" __global__ void rogers_satchell_volatility_batch_f64(
    int len,
    const double* __restrict__ prefix_sum,
    const int* __restrict__ prefix_valid,
    const int* __restrict__ lookbacks,
    const int* __restrict__ signal_lengths,
    int rows,
    int slots,
    double* __restrict__ out_rs,
    double* __restrict__ out_signal
) {
    int slot = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (slot >= slots) {
        return;
    }

    const double nan_value = rsv_qnan();

    for (int row = slot; row < rows; row += slots) {
        int lookback = lookbacks[row];
        int signal_length = signal_lengths[row];

        size_t row_base = static_cast<size_t>(row) * static_cast<size_t>(len);
        double* rs = out_rs + row_base;
        double* signal = out_signal + row_base;

        for (int i = 0; i < len; ++i) {
            rs[i] = nan_value;
            signal[i] = nan_value;
        }
        if (lookback <= 0 || len == 0) {
            continue;
        }

        // compute_rs_row (:197)
        int warm = lookback - 1;
        if (warm > len) {
            warm = len;
        }
        for (int t = warm; t < len; ++t) {
            int end = t + 1;
            int start = end - lookback;
            if (start < 0) {
                continue;
            }
            if (prefix_valid[end] - prefix_valid[start] == lookback) {
                double variance =
                    (prefix_sum[end] - prefix_sum[start]) / static_cast<double>(lookback);
                if (variance < 0.0) {
                    variance = 0.0;
                }
                rs[t] = sqrt(variance);
            }
        }

        // compute_signal_row (:136)
        if (signal_length <= 0) {
            continue;
        }
        double sum = 0.0;
        int valid = 0;
        for (int i = 0; i < len; ++i) {
            double value = rs[i];
            if (isfinite(value)) {
                sum += value;
                valid += 1;
            }
            if (i >= signal_length) {
                double old = rs[i - signal_length];
                if (isfinite(old)) {
                    sum -= old;
                    valid -= 1;
                }
            }
            if (i + 1 >= signal_length && valid == signal_length) {
                signal[i] = sum / static_cast<double>(signal_length);
            }
        }
    }
}

// The time-major many-series form the existing API exposes
// (`rogers_satchell_volatility_many_series_one_param_time_major_dev`, :329):
// `cols` independent series of `rows` bars each, laid out `idx = t * cols + s`,
// all sharing ONE (lookback, signal_length).
//
// ONE THREAD PER SERIES. Each series needs its own prefix arrays, so the host
// plans slots the same way every other kernel here does and each thread loops
// `series = slot; series < cols; series += slots`.
extern "C" __global__ void rogers_satchell_many_series_time_major_f32in(
    const float* __restrict__ open_tm,
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int cols,
    int rows,
    int lookback,
    int signal_length,
    int slots,
    double* scratch_sum,
    int* scratch_valid,
    float* __restrict__ out_rs,
    float* __restrict__ out_signal
) {
    int slot = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (slot >= slots) {
        return;
    }

    // Narrowing the f64 quiet NaN, rather than an f32 bit pattern: this file
    // stays f64-first and the only float here is the OUTPUT type the existing
    // wrapper API returns.
    const float nan_f32 = static_cast<float>(rsv_qnan());
    double* prefix_sum = scratch_sum + static_cast<size_t>(slot) * static_cast<size_t>(rows + 1);
    int* prefix_valid = scratch_valid + static_cast<size_t>(slot) * static_cast<size_t>(rows + 1);

    for (int series = slot; series < cols; series += slots) {
        prefix_sum[0] = 0.0;
        prefix_valid[0] = 0;
        for (int t = 0; t < rows; ++t) {
            size_t idx = static_cast<size_t>(t) * static_cast<size_t>(cols) +
                         static_cast<size_t>(series);
            prefix_valid[t + 1] = prefix_valid[t];
            prefix_sum[t + 1] = prefix_sum[t];
            double term;
            if (rsv_term(static_cast<double>(open_tm[idx]), static_cast<double>(high_tm[idx]),
                         static_cast<double>(low_tm[idx]), static_cast<double>(close_tm[idx]),
                         &term)) {
                prefix_valid[t + 1] += 1;
                prefix_sum[t + 1] += term;
            }
        }

        for (int t = 0; t < rows; ++t) {
            size_t idx = static_cast<size_t>(t) * static_cast<size_t>(cols) +
                         static_cast<size_t>(series);
            out_rs[idx] = nan_f32;
            out_signal[idx] = nan_f32;
        }
        if (lookback <= 0 || rows == 0) {
            continue;
        }

        int warm = lookback - 1;
        if (warm > rows) {
            warm = rows;
        }
        for (int t = warm; t < rows; ++t) {
            int end = t + 1;
            int start = end - lookback;
            if (start < 0) {
                continue;
            }
            if (prefix_valid[end] - prefix_valid[start] == lookback) {
                double variance =
                    (prefix_sum[end] - prefix_sum[start]) / static_cast<double>(lookback);
                if (variance < 0.0) {
                    variance = 0.0;
                }
                size_t idx = static_cast<size_t>(t) * static_cast<size_t>(cols) +
                             static_cast<size_t>(series);
                out_rs[idx] = static_cast<float>(sqrt(variance));
            }
        }

        if (signal_length <= 0) {
            continue;
        }
        double sum = 0.0;
        int valid = 0;
        for (int t = 0; t < rows; ++t) {
            size_t idx = static_cast<size_t>(t) * static_cast<size_t>(cols) +
                         static_cast<size_t>(series);
            float value = out_rs[idx];
            if (isfinite(value)) {
                sum += static_cast<double>(value);
                valid += 1;
            }
            if (t >= signal_length) {
                size_t old_idx = static_cast<size_t>(t - signal_length) *
                                     static_cast<size_t>(cols) +
                                 static_cast<size_t>(series);
                float old = out_rs[old_idx];
                if (isfinite(old)) {
                    sum -= static_cast<double>(old);
                    valid -= 1;
                }
            }
            if (t + 1 >= signal_length && valid == signal_length) {
                out_signal[idx] = static_cast<float>(sum / static_cast<double>(signal_length));
            }
        }
    }
}

// ===========================================================================
// NEOETHOS f64 LANE  --  closer 4
//
// CPU reference:
//   * term        : rs_component_valid,
//                   src/indicators/rogers_satchell_volatility.rs:380-386, and
//                   its validating twin rs_component (:371-377).
//   * clean path  : compute_all_valid_rolling_fixed::<8, 8>, :643-708 -- the
//                   body compute_all_valid_rolling (:566-578) selects for the
//                   CPU DEFAULTS (lookback 8, signal_length 8), which is what
//                   this PERIOD-INVARIANT lane always has.
//   * dirty path  : build_term_prefixes (:480-500) + compute_rs_from_prefix
//                   (:503-531).
//   * selector    : rogers_satchell_volatility_compute_into, :713-742, on the
//                   `all_valid` flag prepare_input returns (:474, valid == len).
//   * refusals    : prepare_input, :388-478.
//   * emitted col : `rs`. compute_rogers_satchell_volatility_batch
//                   (cpu_batch.rs:6596) maps output_id "value" -> out.rs
//                   (:6688 region).
//   * PERIOD-INVARIANT: the batch reads `lookback` (8) and `signal_length` (8)
//                   and never `period` (cpu_batch.rs:6608-6610).
//
// BOTH PATHS ARE IMPLEMENTED AND THEY ARE NOT THE SAME ARITHMETIC. The clean
// path carries a RUNNING SUM and scales it by a precomputed reciprocal
// (`term_sum * inv_lookback`, :688); the dirty path takes a PREFIX-SUM
// DIFFERENCE and divides (`(prefix_sum[end] - prefix_sum[start]) / lookback`,
// :523). Those disagree in the last ulp on any real series, so choosing one
// unconditionally would be wrong for whichever frame took the other branch.
// The selector is reproduced exactly: `valid == len`, where valid counts bars
// whose four OHLC prices are ALL finite AND strictly positive.
//
// FIRST-VALID IGNORED: the CPU never computes a first-valid index for this
// indicator at all -- prepare_input counts valid bars but never locates the
// first one, and both compute paths start at index 0. Its validity rule
// (validate_ohlc, :366-368: finite AND > 0.0) is also the rule
// `garman_klass_volatility` already declares Ignored for in this lane, and for
// the same reason: no F64FirstValidRule variant expresses it.
//
// f64 END TO END: `log()` and `sqrt()`, never `logf`/`sqrtf`/`__logf`. The file
// is in build.rs::F64_LANE_SOURCES, so the three logarithms per bar are
// compiled with -prec-div=true, -fmad=false and WITHOUT --use_fast_math. That
// matters more here than almost anywhere else: the term is a difference of
// logarithms of near-equal prices, so it is already cancellation-prone and an
// approximate log would dominate the result.
// ===========================================================================

#define RSV_NEO_LOOKBACK 8
#define RSV_NEO_SIGNAL 8

static __device__ __forceinline__ double rsv_neo_qnan() {
  return __longlong_as_double(0x7ff8000000000000ULL);
}

// validate_ohlc, :366-368.
static __device__ __forceinline__ bool rsv_neo_ok(double v) {
  return isfinite(v) && v > 0.0;
}

// rs_component_valid, :380-386. Written term for term: three divides, three
// logs, then two multiply-adds the CPU does NOT fuse.
static __device__ __forceinline__ double rsv_neo_term(double o, double h, double l, double c) {
  const double high_close = log(h / c);
  const double low_close = log(l / c);
  const double close_open = log(c / o);
  return high_close * (high_close + close_open) + low_close * (low_close + close_open);
}

extern "C" __global__
void rogers_satchell_volatility_neo_batch_f64(const double* __restrict__ open,
                                              const double* __restrict__ high,
                                              const double* __restrict__ low,
                                              const double* __restrict__ close,
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
  const double nn = rsv_neo_qnan();
  for (int i = 0; i < n; ++i) row[i] = nn;

  const int lookback = RSV_NEO_LOOKBACK;

  // prepare_input, :388-478.
  if (lookback > n) return;
  int valid = 0;
  for (int i = 0; i < n; ++i) {
    if (rsv_neo_ok(open[i]) && rsv_neo_ok(high[i]) && rsv_neo_ok(low[i]) && rsv_neo_ok(close[i]))
      valid += 1;
  }
  if (valid == 0) return;
  if (valid < lookback) return;

  const bool all_valid = (valid == n);
  const int rs_warm = (lookback - 1 < n) ? (lookback - 1) : n;

  if (all_valid) {
    // compute_all_valid_rolling_fixed::<8, 8>, :643-708.
    double term_ring[RSV_NEO_LOOKBACK];
    const double inv_lookback = 1.0 / (double)lookback;
    double term_sum = 0.0;
    int term_idx = 0, term_count = 0;
    for (int k = 0; k < lookback; ++k) term_ring[k] = 0.0;

    for (int i = 0; i < n; ++i) {
      const double term = rsv_neo_term(open[i], high[i], low[i], close[i]);
      if (term_count == lookback) term_sum -= term_ring[term_idx];
      else term_count += 1;
      term_ring[term_idx] = term;
      term_sum += term;
      term_idx += 1;
      if (term_idx == lookback) term_idx = 0;

      if (term_count == lookback) {
        double variance = term_sum * inv_lookback;
        if (variance < 0.0) variance = 0.0;
        row[i] = sqrt(variance);
      }
    }
    return;
  }

  // build_term_prefixes + compute_rs_from_prefix, :480-531. The prefixes are
  // O(n) host allocations; on the card they are carried as a RUNNING pair,
  // which is the same recurrence -- prefix[i+1] = prefix[i] + term -- read at
  // two points. The trailing window is only `lookback` deep, so the kernel
  // keeps the last `lookback + 1` prefix entries in a per-thread ring instead
  // of an n-long array, which is what makes it an O(1)-memory kernel rather
  // than an unbounded allocation.
  double psum_ring[RSV_NEO_LOOKBACK + 1];
  int pvalid_ring[RSV_NEO_LOOKBACK + 1];
  const int cap = lookback + 1;
  double psum = 0.0;
  int pvalid = 0;
  psum_ring[0] = 0.0;
  pvalid_ring[0] = 0;
  for (int k = 1; k < cap; ++k) { psum_ring[k] = 0.0; pvalid_ring[k] = 0; }

  for (int i = 0; i < n; ++i) {
    if (rsv_neo_ok(open[i]) && rsv_neo_ok(high[i]) && rsv_neo_ok(low[i]) && rsv_neo_ok(close[i])) {
      pvalid += 1;
      psum += rsv_neo_term(open[i], high[i], low[i], close[i]);
    }
    const int end = i + 1;
    psum_ring[end % cap] = psum;
    pvalid_ring[end % cap] = pvalid;

    if (i < rs_warm) continue;
    const int start = end - lookback;
    const double seg_sum = psum - psum_ring[start % cap];
    const int seg_valid = pvalid - pvalid_ring[start % cap];
    if (seg_valid == lookback) {
      double variance = seg_sum / (double)lookback;
      if (variance < 0.0) variance = 0.0;
      row[i] = sqrt(variance);
    } else {
      row[i] = nn;
    }
  }
}
