// smooth_theil_sen — f64 CUDA kernel.
//
// WHAT THIS REPLACES
// ------------------
// One line:  extern "C" __global__ void smooth_theil_sen_batch_f64() {}
// plus a wrapper that resolved that empty symbol, computed on the host, and
// uploaded the host answer so the caller believed the card had produced it.
//
// CPU REFERENCE
// -------------
//   src/indicators/smooth_theil_sen.rs
//     :513  exponential_interpolation      :520  gaussian_kernel
//     :525  kernel_weights                 :548  median_in_place
//     :559  estimator                      :575  build_kernel_cache
//     :656  required_finite_segment        :672  segment_all_finite
//     :677  smooth_weighted_sorted         :744  compute_point   <- the body
//     :857  smooth_theil_sen_compute_into  :1541 batch_inner_into
//
// WHY THIS IS NOT "UNKERNELABLE"
// ------------------------------
// Theil-Sen is a MEDIAN OF PAIRWISE SLOPES — an order statistic. An order
// statistic is a standard GPU shape, not an exception: each (row, bar) owns its
// own candidate set, so the selection is per-thread over a private slab. With
// `length = 25` the slab is 300 slopes; a heapsort over it is 300*log2(300)
// comparisons, which is smaller than the 300 divisions that built it.
//
// THE SORT MUST BE `f64::total_cmp`, NOT `<`
// ------------------------------------------
// `estimator` (:559) sorts and then multiplies by POSITIONAL weights, so the
// permutation is part of the answer, not an implementation detail. The CPU
// sorts with `sort_unstable_by(f64::total_cmp)` — a TOTAL order that separates
// -0.0 from +0.0 and places NaN above everything. A plain `<` comparator gives
// a different permutation for those inputs and therefore a different weighted
// sum. `sts_total_key` below is `f64::total_cmp`'s exact bit transform, and the
// sort compares those keys. Because total_cmp only calls two values equal when
// their bit patterns are identical, an unstable sort and a stable one produce
// the same sequence, so heapsort is a faithful substitute for `sort_unstable`.
//
// SHAPE
// -----
// ONE THREAD PER PARAMETER ROW, walking bars ascending, looping
// `row = slot; row < rows; row += slots` so scratch is bounded by the card.
//
// ARITHMETIC
// ----------
// f64 throughout; the file is in `F64_LANE_SOURCES` so it is never compiled
// with `--use_fast_math`, and `fma()` appears only where the CPU writes
// `mul_add` (the Rmsd branch, :819). `1024^(-0.5)` is exactly 1/32, so
// `exponential_interpolation` is exact on both sides; `exp` and `sqrt` are the
// only transcendentals and `-prec-sqrt=true` pins the second of them.

#include <cmath>
#include <cstdint>

#define STS_STYLE_MEAN          0
#define STS_STYLE_SMOOTH_MEDIAN 1
#define STS_STYLE_MEDIAN        2

#define STS_DEV_MAD  0
#define STS_DEV_RMSD 1

__device__ __forceinline__ double sts_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

// `f64::total_cmp`: reinterpret as i64, then flip the low 63 bits of negatives.
// Comparing the results as signed integers is exactly the CPU's total order.
__device__ __forceinline__ long long sts_total_key(double value) {
    long long bits = __double_as_longlong(value);
    bits ^= static_cast<long long>((static_cast<unsigned long long>(bits >> 63)) >> 1);
    return bits;
}

__device__ __forceinline__ void sts_sift_down(double* a, int start, int end) {
    int root = start;
    while (2 * root + 1 <= end) {
        int child = 2 * root + 1;
        if (child + 1 <= end && sts_total_key(a[child]) < sts_total_key(a[child + 1])) {
            child += 1;
        }
        if (sts_total_key(a[root]) < sts_total_key(a[child])) {
            double tmp = a[root];
            a[root] = a[child];
            a[child] = tmp;
            root = child;
        } else {
            return;
        }
    }
}

// In-place ascending heapsort under `total_cmp`. No recursion, no extra memory.
__device__ __forceinline__ void sts_sort(double* a, int n) {
    if (n < 2) {
        return;
    }
    for (int start = (n - 2) / 2; start >= 0; --start) {
        sts_sift_down(a, start, n - 1);
    }
    for (int end = n - 1; end > 0; --end) {
        double tmp = a[0];
        a[0] = a[end];
        a[end] = tmp;
        sts_sift_down(a, 0, end - 1);
    }
}

// exponential_interpolation (:513). `k` is the style blend, always 0.5 here.
__device__ __forceinline__ double sts_exp_interp(double k, double endpoint) {
    double clamped = k < 0.0 ? 0.0 : (k > 1.0 ? 1.0 : k);
    const double min_value = 0.5;
    return (endpoint - min_value) * pow(1024.0, clamped - 1.0) + min_value;
}

// gaussian_kernel (:520)
__device__ __forceinline__ double sts_gaussian(double source, double bandwidth) {
    double ratio = source / bandwidth;
    return exp(-(ratio * ratio) / 4.0) / sqrt(2.0 * M_PI);
}

// kernel_weights (:525). Only SmoothMedian has weights; the other two styles
// get none and never read the buffer.
__device__ __forceinline__ void sts_kernel_weights(double* weights, int size, int style) {
    if (style != STS_STYLE_SMOOTH_MEDIAN || size == 0) {
        return;
    }
    double width = sts_exp_interp(0.5, static_cast<double>(size));
    double center = (static_cast<double>(size) - 1.0) * 0.5;
    double normalization = 0.0;
    for (int i = 0; i < size; ++i) {
        double position = static_cast<double>(i) - center;
        double weight = sts_gaussian(position, width);
        weights[i] = weight;
        normalization += weight;
    }
    if (normalization != 0.0) {
        for (int i = 0; i < size; ++i) {
            weights[i] /= normalization;
        }
    }
}

// estimator (:559)
__device__ __forceinline__ double sts_estimator(
    double* values, int n, int style, const double* weights) {
    if (style == STS_STYLE_MEAN) {
        // CPU sums in PUSH order and does not sort.
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += values[i];
        }
        return sum / static_cast<double>(n);
    }
    if (style == STS_STYLE_MEDIAN) {
        sts_sort(values, n);
        if (n % 2 == 1) {
            return values[n / 2];
        }
        return (values[n / 2 - 1] + values[n / 2]) * 0.5;
    }
    // SmoothMedian: sort, then a positional weighted sum in index order.
    sts_sort(values, n);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += values[i] * weights[i];
    }
    return sum;
}

extern "C" __global__ void smooth_theil_sen_batch_f64(
    const double* __restrict__ data,
    int len,
    int first,
    const int* __restrict__ lengths,
    const int* __restrict__ offsets,
    const double* __restrict__ multipliers,
    int slope_style,
    int residual_style,
    int deviation_style,
    int mad_style,
    int include_prediction,
    int rows,
    int slots,
    int slope_cap,
    int residual_cap,
    int error_cap,
    double* scratch,
    double* __restrict__ out_value,
    double* __restrict__ out_upper,
    double* __restrict__ out_lower,
    double* __restrict__ out_slope,
    double* __restrict__ out_intercept,
    double* __restrict__ out_deviation
) {
    int slot = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (slot >= slots) {
        return;
    }

    const double nan_value = sts_qnan();
    size_t per_slot = static_cast<size_t>(2 * slope_cap + 2 * residual_cap + 2 * error_cap);
    double* base = scratch + static_cast<size_t>(slot) * per_slot;
    double* slopes = base;
    double* slope_weights = slopes + slope_cap;
    double* residuals = slope_weights + slope_cap;
    double* residual_weights = residuals + residual_cap;
    double* errors = residual_weights + residual_cap;
    double* error_weights = errors + error_cap;

    for (int row = slot; row < rows; row += slots) {
        int length = lengths[row];
        int offset = offsets[row];
        double multiplier = multipliers[row];

        int pair_count = length * (length - 1) / 2;
        int error_len = length + (include_prediction ? offset : 0);

        // build_kernel_cache (:575) — once per row, exactly as the CPU does.
        sts_kernel_weights(slope_weights, pair_count, slope_style);
        sts_kernel_weights(residual_weights, length, residual_style);
        sts_kernel_weights(error_weights, error_len, mad_style);

        size_t row_base = static_cast<size_t>(row) * static_cast<size_t>(len);
        double* v_out = out_value + row_base;
        double* u_out = out_upper + row_base;
        double* l_out = out_lower + row_base;
        double* s_out = out_slope + row_base;
        double* i_out = out_intercept + row_base;
        double* d_out = out_deviation + row_base;

        // `warmup_bars` (:481) = length + offset - 1, from the first finite bar.
        int warmup = first + length + offset - 1;
        for (int idx = 0; idx < len; ++idx) {
            v_out[idx] = nan_value;
            u_out[idx] = nan_value;
            l_out[idx] = nan_value;
            s_out[idx] = nan_value;
            i_out[idx] = nan_value;
            d_out[idx] = nan_value;
        }

        for (int idx = warmup; idx < len; ++idx) {
            // required_finite_segment (:656) + segment_all_finite (:672)
            if (idx < offset) {
                continue;
            }
            int seg_base = idx - offset;
            int start = seg_base + 1 - length;
            if (start < 0) {
                continue;
            }
            int end = include_prediction ? idx : seg_base;
            bool all_finite = true;
            for (int i = start; i <= end; ++i) {
                if (!isfinite(data[i])) {
                    all_finite = false;
                    break;
                }
            }
            if (!all_finite) {
                continue;
            }

            // Pairwise slopes, in the CPU's push order (:769).
            int n = 0;
            for (int i = 0; i < length - 1; ++i) {
                double value_i = data[seg_base - i];
                for (int j = i + 1; j < length; ++j) {
                    double value_j = data[seg_base - j];
                    slopes[n] = (value_j - value_i) / static_cast<double>(j - i);
                    n += 1;
                }
            }
            double beta_1 = sts_estimator(slopes, n, slope_style, slope_weights);

            for (int j = 0; j < length; ++j) {
                residuals[j] = data[seg_base - j] - beta_1 * static_cast<double>(j);
            }
            double beta_0 = sts_estimator(residuals, length, residual_style, residual_weights);

            double predicted = beta_0 - beta_1 * static_cast<double>(offset);

            double deviation;
            if (deviation_style == STS_DEV_MAD) {
                int start_point = include_prediction ? -offset : 0;
                int count = 0;
                for (int point = start_point; point <= length - 1; ++point) {
                    int source_idx = idx - offset - point;
                    double predicted_point = beta_0 + beta_1 * static_cast<double>(point);
                    errors[count] = fabs(data[source_idx] - predicted_point);
                    count += 1;
                }
                deviation =
                    sts_estimator(errors, count, mad_style, error_weights) * multiplier;
            } else {
                int start_point = include_prediction ? -offset : 0;
                double square_errors = 0.0;
                int count = 0;
                for (int point = start_point; point <= length - 1; ++point) {
                    int source_idx = idx - offset - point;
                    double predicted_point = beta_0 + beta_1 * static_cast<double>(point);
                    double error = data[source_idx] - predicted_point;
                    square_errors += error * error;
                    count += 1;
                }
                // CPU: `square_errors.sqrt().mul_add(multiplier / (count as f64).sqrt(), 0.0)`
                deviation = fma(sqrt(square_errors),
                                multiplier / sqrt(static_cast<double>(count)), 0.0);
            }

            v_out[idx] = predicted;
            u_out[idx] = predicted + deviation;
            l_out[idx] = predicted - deviation;
            s_out[idx] = beta_1;
            i_out[idx] = beta_0;
            d_out[idx] = deviation;
        }
    }
}

// ===========================================================================
// NEOETHOS f64 LANE  --  closer 4
//
// The entry point above, `smooth_theil_sen_batch_f64`, is a full all-double
// port of src/indicators/smooth_theil_sen.rs written by this same workflow --
// but its ABI is the CRATE's batch ABI: five parameter arrays, five style
// selectors, a caller-allocated `scratch` pointer and SIX output matrices. The
// f64 lane launches exactly one shape,
//     (series..., n, periods, n_combos, first_valid, out)
// so a variant pointing at that symbol would read the stack. This entry point
// is the lane-shaped one. It reuses the device helpers above -- sts_sort,
// sts_kernel_weights, sts_estimator -- so there is ONE implementation of the
// arithmetic in this file, not two.
//
// CPU reference:
//   * arithmetic   : smooth_theil_sen.rs -- pairwise slopes (:769), the
//                    smooth-median estimator (:559), kernel_weights (:525),
//                    exponential_interpolation (:513), gaussian_kernel (:520).
//   * first-valid  : first_valid, :455 -- data.iter().position(|v|
//                    v.is_finite()). ONE price series scanned with is_finite,
//                    which is F64FirstValidRule::CloseFinite. LOAD-BEARING: the
//                    warmup is first + length + offset - 1 (:481-483, :502), so
//                    a different index shifts the whole series.
//   * refusals     : validate_input, :486-505.
//   * emitted col  : `value`, the predicted level. compute_smooth_theil_sen_
//                    batch (cpu_batch.rs:11201) maps output_id "value" ->
//                    out.value (:11337 region).
//   * PERIOD-INVARIANT: the batch reads `length` (25), `offset` (0),
//                    `multiplier` (2.0), `slope_style`, `residual_style`,
//                    `deviation_style`, `mad_style` and
//                    `include_prediction_in_deviation` -- never `period`
//                    (cpu_batch.rs:11213-11250).
//
// THE ORDER STATISTIC IS THE POINT. Theil-Sen's slope is the median of all
// length*(length-1)/2 = 300 pairwise slopes. On the card that is a per-thread
// local array of 300 doubles plus a heapsort (sts_sort above) -- the
// "per-column selection in a per-thread window" shape, with a COMPILE-TIME
// bound, so there is no dynamic allocation anywhere in the kernel. That bound
// is what makes this kernel possible at all rather than "irregular".
//
// PER-THREAD SCRATCH is 700 doubles (300 slopes + 300 slope weights + 25
// residuals + 25 residual weights + 25 errors + 25 error weights) = 5,600
// bytes. Fixed at compile time by the CPU DEFAULTS, which is the only
// parameter set this PERIOD-INVARIANT lane can be asked for.
// ===========================================================================

#define STS_NEO_LENGTH 25
#define STS_NEO_OFFSET 0
#define STS_NEO_MULTIPLIER 2.0
#define STS_NEO_PAIRS ((STS_NEO_LENGTH * (STS_NEO_LENGTH - 1)) / 2)

extern "C" __global__ void smooth_theil_sen_neo_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out) {
  const int combo = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (combo >= n_combos) return;
  (void)periods;  // PERIOD-INVARIANT -- see the header.

  if (n <= 0) return;
  double* __restrict__ row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);
  const double nan_value = sts_qnan();
  for (int i = 0; i < n; ++i) row[i] = nan_value;

  const int length = STS_NEO_LENGTH;
  const int offset = STS_NEO_OFFSET;
  const double multiplier = STS_NEO_MULTIPLIER;
  const int slope_style = STS_STYLE_SMOOTH_MEDIAN;
  const int residual_style = STS_STYLE_SMOOTH_MEDIAN;
  const int mad_style = STS_STYLE_SMOOTH_MEDIAN;
  const int include_prediction = 0;

  int first = first_valid;
  if (first < 0) first = 0;
  // validate_input, :486-505: no finite bar, or fewer than length + offset bars
  // after it, is an Err -- and an Err is no column at all.
  if (first >= n) return;
  if (n - first < length + offset) return;

  double slopes[STS_NEO_PAIRS];
  double slope_weights[STS_NEO_PAIRS];
  double residuals[STS_NEO_LENGTH];
  double residual_weights[STS_NEO_LENGTH];
  double errors[STS_NEO_LENGTH];
  double error_weights[STS_NEO_LENGTH];

  const int pair_count = STS_NEO_PAIRS;
  const int error_len = length + (include_prediction ? offset : 0);

  // build_kernel_cache (:575) -- once per row, exactly as the CPU does.
  sts_kernel_weights(slope_weights, pair_count, slope_style);
  sts_kernel_weights(residual_weights, length, residual_style);
  sts_kernel_weights(error_weights, error_len, mad_style);

  const int warmup = first + length + offset - 1;   // warmup_bars, :481-483

  for (int idx = warmup; idx < n; ++idx) {
    if (idx < offset) continue;
    const int seg_base = idx - offset;
    const int start = seg_base + 1 - length;
    if (start < 0) continue;
    const int end = include_prediction ? idx : seg_base;
    bool all_finite = true;
    for (int i = start; i <= end; ++i) {
      if (!isfinite(data[i])) { all_finite = false; break; }
    }
    if (!all_finite) continue;

    // Pairwise slopes, in the CPU's push order (:769).
    int cnt = 0;
    for (int i = 0; i < length - 1; ++i) {
      const double value_i = data[seg_base - i];
      for (int j = i + 1; j < length; ++j) {
        const double value_j = data[seg_base - j];
        slopes[cnt] = (value_j - value_i) / static_cast<double>(j - i);
        cnt += 1;
      }
    }
    const double beta_1 = sts_estimator(slopes, cnt, slope_style, slope_weights);

    for (int j = 0; j < length; ++j) {
      residuals[j] = data[seg_base - j] - beta_1 * static_cast<double>(j);
    }
    const double beta_0 = sts_estimator(residuals, length, residual_style, residual_weights);

    // The CPU computes the deviation to build `upper`/`lower`; this lane emits
    // `value`, which is `predicted` alone, so the MAD block is dead work here
    // and is omitted rather than computed and discarded.
    row[idx] = beta_0 - beta_1 * static_cast<double>(offset);
  }
  (void)errors;
  (void)multiplier;
}
