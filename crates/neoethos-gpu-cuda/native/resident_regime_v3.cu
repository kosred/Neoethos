#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kRegimeColumnsV3 = 14;
constexpr unsigned char kValidV3 = 0;
constexpr unsigned char kWarmupV3 = 1;
constexpr unsigned char kZeroDenominatorV3 = 5;
constexpr unsigned char kComputeFailureV3 = 8;
constexpr const char kRegimeOperationScheduleV1[] =
    "neoethos.regime.semantic-v3.f64-rn-fixed-order-log49-neumaier-v1";
constexpr const char kRegimeFixtureSha256V1[] =
    "f0f89c26727e90206bb85bdb4b3f6e11f59652176f7ba8475e9fbaa301548a93";
constexpr const char kRegimeLog49OperationTokensV1[] =
    "neoethos.regime.log49-mirror.v1|subnormal-scale=0x4350000000000000|mantissa-mask=0x000fffffffffffff|one-exponent=0x3ff0000000000000|ln2=0x3fe62e42fefa39ef|series-odd=3..49|order=normalize,bits,exponent,mantissa,z,z2,term,sum,loop,return|rounding=rn-no-fma";
constexpr const char kRegimeLog49OperationTokensSha256V1[] =
    "73002b6761d1ca425250a761fa4411cf3ae0d26c862caa964e93063c69c32080";
constexpr const char kRegimeLog49CudaMirrorSha256V1[] =
    "ec8299d718d7a3d5a189287380f042df603fde7bbed87b7378845d7ce73618fe";
static_assert(kRegimeColumnsV3 == 14);
static_assert(sizeof(kRegimeOperationScheduleV1) > 1U);
static_assert(sizeof(kRegimeFixtureSha256V1) == 65U);
static_assert(sizeof(kRegimeLog49OperationTokensV1) > 1U);
static_assert(sizeof(kRegimeLog49OperationTokensSha256V1) == 65U);
static_assert(sizeof(kRegimeLog49CudaMirrorSha256V1) == 65U);

__device__ __forceinline__ double from_bits_v3(std::uint64_t bits) {
  return __longlong_as_double(static_cast<long long>(bits));
}

__device__ __forceinline__ std::uint64_t to_bits_v3(double value) {
  return static_cast<std::uint64_t>(__double_as_longlong(value));
}

__device__ __forceinline__ bool finite_v3(double value) {
  return (to_bits_v3(value) & 0x7ff0000000000000ULL) !=
         0x7ff0000000000000ULL;
}

__device__ __forceinline__ double abs_v3(double value) {
  return value < 0.0 ? -value : value;
}

__device__ __forceinline__ double min_v3(double left, double right) {
  return left < right ? left : right;
}

__device__ __forceinline__ double max_v3(double left, double right) {
  return left > right ? left : right;
}

__device__ __forceinline__ double canonical_nan_v3() {
  return from_bits_v3(0x7ff8000000000000ULL);
}

// REGIME_LOG49_CUDA_MIRROR_BEGIN_V1
__device__ __forceinline__ double neoethos_ln_positive_exact_v1(double value) {
  const std::uint64_t source_bits = to_bits_v3(value);
  const bool subnormal = (source_bits & 0x7ff0000000000000ULL) == 0ULL;
  const double normalized =
      subnormal ? __dmul_rn(value, from_bits_v3(0x4350000000000000ULL)) : value;
  const int exponent_adjustment = subnormal ? -54 : 0;
  const std::uint64_t bits = to_bits_v3(normalized);
  const int exponent = static_cast<int>((bits >> 52U) & 0x7ffU) - 1023 +
                       exponent_adjustment;
  const std::uint64_t mantissa_bits =
      (bits & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL;
  const double mantissa = from_bits_v3(mantissa_bits);
  const double z = __ddiv_rn(__dsub_rn(mantissa, 1.0),
                             __dadd_rn(mantissa, 1.0));
  const double z_squared = __dmul_rn(z, z);
  double term = z;
  double sum = z;
  for (unsigned int denominator = 3U; denominator <= 49U; denominator += 2U) {
    term = __dmul_rn(term, z_squared);
    sum = __dadd_rn(sum,
                    __ddiv_rn(term, static_cast<double>(denominator)));
  }
  const double exponent_term = __dmul_rn(
      static_cast<double>(exponent), from_bits_v3(0x3fe62e42fefa39efULL));
  return __dadd_rn(exponent_term, __dmul_rn(2.0, sum));
}

__device__ __forceinline__ double neoethos_log10_positive_exact_v1(
    double value) {
  return __ddiv_rn(neoethos_ln_positive_exact_v1(value),
                   from_bits_v3(0x40026bb1bbb55515ULL));
}
// REGIME_LOG49_CUDA_MIRROR_END_V1

struct NeumaierV1 {
  double sum;
  double compensation;
};

__device__ __forceinline__ NeumaierV1 neumaier_zero_v1() {
  return {0.0, 0.0};
}

__device__ __forceinline__ void neumaier_add_v1(NeumaierV1* state,
                                                 double value) {
  const double next = __dadd_rn(state->sum, value);
  if (abs_v3(state->sum) >= abs_v3(value)) {
    state->compensation = __dadd_rn(
        state->compensation,
        __dadd_rn(__dsub_rn(state->sum, next), value));
  } else {
    state->compensation = __dadd_rn(
        state->compensation,
        __dadd_rn(__dsub_rn(value, next), state->sum));
  }
  state->sum = next;
}

__device__ __forceinline__ double neumaier_finish_v1(NeumaierV1 state) {
  return __dadd_rn(state.sum, state.compensation);
}

__device__ __forceinline__ double scaled_v3(const double* source,
                                             std::size_t row,
                                             double scale_anchor) {
  return __dmul_rn(source[row], scale_anchor);
}

__device__ __forceinline__ double true_range_v3(
    const double* high, const double* low, const double* close, std::size_t row,
    double scale_anchor) {
  const double h = scaled_v3(high, row, scale_anchor);
  const double l = scaled_v3(low, row, scale_anchor);
  const double previous_close = scaled_v3(close, row - 1U, scale_anchor);
  return max_v3(__dsub_rn(h, l),
                max_v3(abs_v3(__dsub_rn(h, previous_close)),
                       abs_v3(__dsub_rn(l, previous_close))));
}

__device__ __forceinline__ void set_invalid_v3(double* values,
                                                unsigned char* validity,
                                                std::size_t rows, int slot,
                                                std::size_t row,
                                                unsigned char reason) {
  const std::size_t index = static_cast<std::size_t>(slot) * rows + row;
  values[index] = canonical_nan_v3();
  validity[index] = reason;
}

__device__ __forceinline__ void set_valid_v3(double* values,
                                              unsigned char* validity,
                                              std::size_t rows, int slot,
                                              std::size_t row, double value) {
  if (finite_v3(value)) {
    const std::size_t index = static_cast<std::size_t>(slot) * rows + row;
    values[index] = value;
    validity[index] = kValidV3;
  } else {
    set_invalid_v3(values, validity, rows, slot, row, kComputeFailureV3);
  }
}

__device__ __forceinline__ void initialize_slot_v3(
    double* values, unsigned char* validity, std::size_t rows, int slot,
    std::size_t row) {
  set_invalid_v3(values, validity, rows, slot, row, kWarmupV3);
}

__device__ __forceinline__ bool gk_component_v3(
    const double* open, const double* high, const double* low,
    const double* close, std::size_t row, double scale_anchor,
    double* component) {
  const double o = scaled_v3(open, row, scale_anchor);
  const double h = scaled_v3(high, row, scale_anchor);
  const double l = scaled_v3(low, row, scale_anchor);
  const double c_value = scaled_v3(close, row, scale_anchor);
  const double log_open = neoethos_ln_positive_exact_v1(o);
  const double u = __dsub_rn(neoethos_ln_positive_exact_v1(h), log_open);
  const double d = __dsub_rn(neoethos_ln_positive_exact_v1(l), log_open);
  const double c =
      __dsub_rn(neoethos_ln_positive_exact_v1(c_value), log_open);
  const double range = __dsub_rn(u, d);
  *component = __dsub_rn(
      __dmul_rn(0.5, __dmul_rn(range, range)),
      __dmul_rn(from_bits_v3(0x3fd8b90bfbe8e7bcULL), __dmul_rn(c, c)));
  return finite_v3(*component) && *component >= 0.0;
}

__global__ void regime_independent_kernel_v3(
    const double* open, const double* high, const double* low,
    const double* close, std::size_t rows, double scale_anchor,
    double* values, unsigned char* validity) {
  const std::size_t first = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t row = first; row < rows; row += stride) {
    initialize_slot_v3(values, validity, rows, 0, row);
    initialize_slot_v3(values, validity, rows, 1, row);
    initialize_slot_v3(values, validity, rows, 5, row);
    initialize_slot_v3(values, validity, rows, 6, row);
    initialize_slot_v3(values, validity, rows, 7, row);
    initialize_slot_v3(values, validity, rows, 8, row);
    initialize_slot_v3(values, validity, rows, 9, row);
    initialize_slot_v3(values, validity, rows, 13, row);

    if (row >= 49U) {
      NeumaierV1 short_sum = neumaier_zero_v1();
      NeumaierV1 long_sum = neumaier_zero_v1();
      bool failed = false;
      for (std::size_t offset = 0; offset < 50U; ++offset) {
        double component = 0.0;
        if (!gk_component_v3(open, high, low, close, row - 49U + offset,
                             scale_anchor, &component)) {
          failed = true;
          break;
        }
        neumaier_add_v1(&long_sum, component);
        if (offset >= 40U) {
          neumaier_add_v1(&short_sum, component);
        }
      }
      if (failed) {
        set_invalid_v3(values, validity, rows, 0, row, kComputeFailureV3);
        set_invalid_v3(values, validity, rows, 1, row, kComputeFailureV3);
      } else {
        const double short_variance =
            __ddiv_rn(neumaier_finish_v1(short_sum), 10.0);
        const double long_variance =
            __ddiv_rn(neumaier_finish_v1(long_sum), 50.0);
        if (!finite_v3(short_variance) || !finite_v3(long_variance) ||
            short_variance < 0.0 || long_variance < 0.0) {
          set_invalid_v3(values, validity, rows, 0, row, kComputeFailureV3);
          set_invalid_v3(values, validity, rows, 1, row, kComputeFailureV3);
        } else {
          const double short_gk = __dsqrt_rn(short_variance);
          const double long_gk = __dsqrt_rn(long_variance);
          if (long_gk == 0.0) {
            set_invalid_v3(values, validity, rows, 0, row,
                           kZeroDenominatorV3);
            set_invalid_v3(values, validity, rows, 1, row,
                           kZeroDenominatorV3);
          } else {
            const double ratio = __ddiv_rn(short_gk, long_gk);
            if (!finite_v3(ratio)) {
              set_invalid_v3(values, validity, rows, 0, row,
                             kComputeFailureV3);
              set_invalid_v3(values, validity, rows, 1, row,
                             kComputeFailureV3);
            } else {
              const double state = ratio > 1.5 ? 1.0 : (ratio < 0.6 ? -1.0 : 0.0);
              const double offset =
                  min_v3(max_v3(__dsub_rn(ratio, 1.0), -3.0), 3.0);
              set_valid_v3(values, validity, rows, 0, row, state);
              set_valid_v3(values, validity, rows, 1, row, offset);
            }
          }
        }
      }
    }

    if (row >= 20U) {
      const std::size_t start = row - 19U;
      NeumaierV1 mean_sum = neumaier_zero_v1();
      for (std::size_t j = start; j <= row; ++j) {
        neumaier_add_v1(&mean_sum, scaled_v3(close, j, scale_anchor));
      }
      const double mean = __ddiv_rn(neumaier_finish_v1(mean_sum), 20.0);
      NeumaierV1 variance_sum = neumaier_zero_v1();
      NeumaierV1 tr_sum = neumaier_zero_v1();
      for (std::size_t j = start; j <= row; ++j) {
        const double deviation =
            __dsub_rn(scaled_v3(close, j, scale_anchor), mean);
        neumaier_add_v1(&variance_sum, __dmul_rn(deviation, deviation));
        neumaier_add_v1(&tr_sum,
                        true_range_v3(high, low, close, j, scale_anchor));
      }
      const double variance =
          __ddiv_rn(neumaier_finish_v1(variance_sum), 20.0);
      const double atr = __ddiv_rn(neumaier_finish_v1(tr_sum), 20.0);
      if (!finite_v3(mean) || !finite_v3(variance) || variance < 0.0 ||
          !finite_v3(atr)) {
        set_invalid_v3(values, validity, rows, 5, row, kComputeFailureV3);
        set_invalid_v3(values, validity, rows, 6, row, kComputeFailureV3);
      } else if (atr == 0.0) {
        set_invalid_v3(values, validity, rows, 5, row, kZeroDenominatorV3);
        set_invalid_v3(values, validity, rows, 6, row, kZeroDenominatorV3);
      } else {
        const double standard_deviation = __dsqrt_rn(variance);
        const double bb_width = __dmul_rn(2.0, standard_deviation);
        const double kc_width = __dmul_rn(1.5, atr);
        const double bb_upper = __dadd_rn(mean, bb_width);
        const double bb_lower = __dsub_rn(mean, bb_width);
        const double kc_upper = __dadd_rn(mean, kc_width);
        const double kc_lower = __dsub_rn(mean, kc_width);
        const double state =
            bb_upper < kc_upper && bb_lower > kc_lower ? 1.0 : -1.0;
        const double deviation = __ddiv_rn(
            __dsub_rn(scaled_v3(close, row, scale_anchor), mean), atr);
        set_valid_v3(values, validity, rows, 5, row, state);
        set_valid_v3(values, validity, rows, 6, row, deviation);
      }
    }

    if (row >= 21U) {
      unsigned int same = 0U;
      unsigned int reversal = 0U;
      for (std::size_t j = row - 19U; j <= row; ++j) {
        const double current =
            __dsub_rn(scaled_v3(close, j, scale_anchor),
                       scaled_v3(close, j - 1U, scale_anchor));
        const double previous =
            __dsub_rn(scaled_v3(close, j - 1U, scale_anchor),
                       scaled_v3(close, j - 2U, scale_anchor));
        if ((current > 0.0 && previous > 0.0) ||
            (current < 0.0 && previous < 0.0)) {
          ++same;
        } else if ((current > 0.0 && previous < 0.0) ||
                   (current < 0.0 && previous > 0.0)) {
          ++reversal;
        }
      }
      const unsigned int total = same + reversal;
      if (total == 0U) {
        set_invalid_v3(values, validity, rows, 7, row, kZeroDenominatorV3);
      } else {
        const double balance = __ddiv_rn(
            __dsub_rn(static_cast<double>(same),
                      static_cast<double>(reversal)),
            static_cast<double>(total));
        set_valid_v3(values, validity, rows, 7, row, balance);
      }
    }

    if (row >= 7U) {
      NeumaierV1 body_sum = neumaier_zero_v1();
      NeumaierV1 range_sum = neumaier_zero_v1();
      for (std::size_t j = row - 7U; j <= row; ++j) {
        neumaier_add_v1(
            &body_sum, __dsub_rn(scaled_v3(close, j, scale_anchor),
                                 scaled_v3(open, j, scale_anchor)));
        neumaier_add_v1(
            &range_sum, __dsub_rn(scaled_v3(high, j, scale_anchor),
                                  scaled_v3(low, j, scale_anchor)));
      }
      const double body = neumaier_finish_v1(body_sum);
      const double range = neumaier_finish_v1(range_sum);
      if (!finite_v3(body) || !finite_v3(range)) {
        set_invalid_v3(values, validity, rows, 8, row, kComputeFailureV3);
      } else if (range == 0.0) {
        set_invalid_v3(values, validity, rows, 8, row, kZeroDenominatorV3);
      } else {
        set_valid_v3(values, validity, rows, 8, row,
                     min_v3(max_v3(__ddiv_rn(body, range), -1.0), 1.0));
      }
    }

    if (row >= 14U) {
      const std::size_t start = row - 13U;
      NeumaierV1 tr_sum = neumaier_zero_v1();
      double highest_true_high = -from_bits_v3(0x7ff0000000000000ULL);
      double lowest_true_low = from_bits_v3(0x7ff0000000000000ULL);
      for (std::size_t j = start; j <= row; ++j) {
        neumaier_add_v1(&tr_sum,
                        true_range_v3(high, low, close, j, scale_anchor));
        const double previous_close = scaled_v3(close, j - 1U, scale_anchor);
        highest_true_high =
            max_v3(highest_true_high,
                   max_v3(scaled_v3(high, j, scale_anchor), previous_close));
        lowest_true_low =
            min_v3(lowest_true_low,
                   min_v3(scaled_v3(low, j, scale_anchor), previous_close));
      }
      const double numerator = neumaier_finish_v1(tr_sum);
      const double denominator =
          __dsub_rn(highest_true_high, lowest_true_low);
      if (!finite_v3(numerator) || !finite_v3(denominator)) {
        set_invalid_v3(values, validity, rows, 9, row, kComputeFailureV3);
      } else if (numerator == 0.0 || denominator == 0.0) {
        set_invalid_v3(values, validity, rows, 9, row,
                       kZeroDenominatorV3);
      } else {
        const double ratio = __ddiv_rn(numerator, denominator);
        if (!finite_v3(ratio) || ratio <= 0.0) {
          set_invalid_v3(values, validity, rows, 9, row, kComputeFailureV3);
        } else {
          const double chop = __ddiv_rn(
              __dmul_rn(100.0, neoethos_log10_positive_exact_v1(ratio)),
              neoethos_log10_positive_exact_v1(14.0));
          set_valid_v3(values, validity, rows, 9, row, chop);
        }
      }
    }

    if (row >= 30U) {
      double returns[30];
      bool failed = false;
      for (std::size_t offset = 0; offset < 30U; ++offset) {
        const std::size_t j = row - 29U + offset;
        returns[offset] = __dsub_rn(
            neoethos_ln_positive_exact_v1(scaled_v3(close, j, scale_anchor)),
            neoethos_ln_positive_exact_v1(
                scaled_v3(close, j - 1U, scale_anchor)));
        if (!finite_v3(returns[offset])) {
          failed = true;
          break;
        }
      }
      if (failed) {
        set_invalid_v3(values, validity, rows, 13, row, kComputeFailureV3);
      } else {
        double minimum = returns[0];
        double maximum = returns[0];
        for (std::size_t offset = 1; offset < 30U; ++offset) {
          minimum = min_v3(minimum, returns[offset]);
          maximum = max_v3(maximum, returns[offset]);
        }
        const double range = __dsub_rn(maximum, minimum);
        if (!finite_v3(range) || range < 0.0) {
          set_invalid_v3(values, validity, rows, 13, row,
                         kComputeFailureV3);
        } else if (range == 0.0) {
          set_valid_v3(values, validity, rows, 13, row, 0.0);
        } else {
          unsigned int bins[10] = {0U};
          for (std::size_t offset = 0; offset < 30U; ++offset) {
            const double coordinate = __dmul_rn(
                __ddiv_rn(__dsub_rn(returns[offset], minimum), range),
                from_bits_v3(0x4023ff7ced916873ULL));
            if (!finite_v3(coordinate) || coordinate < 0.0) {
              failed = true;
              break;
            }
            int bin = static_cast<int>(coordinate);
            bin = bin < 9 ? bin : 9;
            ++bins[bin];
          }
          if (failed) {
            set_invalid_v3(values, validity, rows, 13, row,
                           kComputeFailureV3);
          } else {
            NeumaierV1 entropy_sum = neumaier_zero_v1();
            for (int bin = 0; bin < 10; ++bin) {
              if (bins[bin] != 0U) {
                const double probability =
                    __ddiv_rn(static_cast<double>(bins[bin]), 30.0);
                neumaier_add_v1(
                    &entropy_sum,
                    __dmul_rn(probability,
                              neoethos_ln_positive_exact_v1(probability)));
              } else {
                neumaier_add_v1(&entropy_sum, 0.0);
              }
            }
            const double entropy = __ddiv_rn(
                -neumaier_finish_v1(entropy_sum),
                from_bits_v3(0x40026bb1bbb55515ULL));
            set_valid_v3(values, validity, rows, 13, row, entropy);
          }
        }
      }
    }
  }
}

__device__ void compute_wilder_lane_v3(
    const double* high, const double* low, const double* close,
    std::size_t rows, double scale_anchor, double* values,
    unsigned char* validity) {
  for (std::size_t row = 0; row < rows; ++row) {
    initialize_slot_v3(values, validity, rows, 2, row);
    initialize_slot_v3(values, validity, rows, 3, row);
    initialize_slot_v3(values, validity, rows, 4, row);
  }
  if (rows <= 14U) {
    return;
  }
  NeumaierV1 plus_seed = neumaier_zero_v1();
  NeumaierV1 minus_seed = neumaier_zero_v1();
  NeumaierV1 tr_seed = neumaier_zero_v1();
  for (std::size_t row = 1; row <= 14U; ++row) {
    const double up_move =
        __dsub_rn(scaled_v3(high, row, scale_anchor),
                   scaled_v3(high, row - 1U, scale_anchor));
    const double down_move =
        __dsub_rn(scaled_v3(low, row - 1U, scale_anchor),
                   scaled_v3(low, row, scale_anchor));
    const double plus_dm = up_move > down_move && up_move > 0.0 ? up_move : 0.0;
    const double minus_dm =
        down_move > up_move && down_move > 0.0 ? down_move : 0.0;
    neumaier_add_v1(&plus_seed, plus_dm);
    neumaier_add_v1(&minus_seed, minus_dm);
    neumaier_add_v1(&tr_seed,
                    true_range_v3(high, low, close, row, scale_anchor));
  }
  double plus_smooth = neumaier_finish_v1(plus_seed);
  double minus_smooth = neumaier_finish_v1(minus_seed);
  double tr_smooth = neumaier_finish_v1(tr_seed);
  double dx_seed[14] = {0.0};
  int dx_seed_count = 0;
  double adx = 0.0;
  bool adx_live = false;

  for (std::size_t row = 14U; row < rows; ++row) {
    if (row > 14U) {
      const double up_move =
          __dsub_rn(scaled_v3(high, row, scale_anchor),
                     scaled_v3(high, row - 1U, scale_anchor));
      const double down_move =
          __dsub_rn(scaled_v3(low, row - 1U, scale_anchor),
                     scaled_v3(low, row, scale_anchor));
      const double plus_dm =
          up_move > down_move && up_move > 0.0 ? up_move : 0.0;
      const double minus_dm =
          down_move > up_move && down_move > 0.0 ? down_move : 0.0;
      plus_smooth = __dadd_rn(
          __dsub_rn(plus_smooth, __ddiv_rn(plus_smooth, 14.0)), plus_dm);
      minus_smooth = __dadd_rn(
          __dsub_rn(minus_smooth, __ddiv_rn(minus_smooth, 14.0)), minus_dm);
      tr_smooth = __dadd_rn(
          __dsub_rn(tr_smooth, __ddiv_rn(tr_smooth, 14.0)),
          true_range_v3(high, low, close, row, scale_anchor));
    }

    unsigned char invalid_reason = kValidV3;
    double direction = 0.0;
    double dx = 0.0;
    bool dx_valid = false;
    if (!finite_v3(plus_smooth) || !finite_v3(minus_smooth) ||
        !finite_v3(tr_smooth)) {
      invalid_reason = kComputeFailureV3;
    } else if (tr_smooth == 0.0) {
      invalid_reason = kZeroDenominatorV3;
    } else {
      const double plus_di =
          __dmul_rn(__ddiv_rn(plus_smooth, tr_smooth), 100.0);
      const double minus_di =
          __dmul_rn(__ddiv_rn(minus_smooth, tr_smooth), 100.0);
      const double di_sum = __dadd_rn(plus_di, minus_di);
      if (!finite_v3(plus_di) || !finite_v3(minus_di) ||
          !finite_v3(di_sum)) {
        invalid_reason = kComputeFailureV3;
      } else if (di_sum == 0.0) {
        invalid_reason = kZeroDenominatorV3;
      } else {
        direction = plus_di > minus_di ? 1.0 : (minus_di > plus_di ? -1.0 : 0.0);
        dx = __dmul_rn(
            __ddiv_rn(abs_v3(__dsub_rn(plus_di, minus_di)), di_sum),
            100.0);
        if (finite_v3(dx)) {
          dx_valid = true;
        } else {
          invalid_reason = kComputeFailureV3;
        }
      }
    }

    if (dx_valid) {
      set_valid_v3(values, validity, rows, 3, row, direction);
    } else {
      set_invalid_v3(values, validity, rows, 3, row, invalid_reason);
      dx_seed_count = 0;
      adx_live = false;
      if (row >= 27U) {
        set_invalid_v3(values, validity, rows, 2, row, invalid_reason);
        set_invalid_v3(values, validity, rows, 4, row, invalid_reason);
      }
      continue;
    }

    if (adx_live) {
      adx = __ddiv_rn(__dadd_rn(__dmul_rn(adx, 13.0), dx), 14.0);
    } else {
      dx_seed[dx_seed_count++] = dx;
      if (dx_seed_count == 14) {
        NeumaierV1 adx_seed = neumaier_zero_v1();
        for (int index = 0; index < 14; ++index) {
          neumaier_add_v1(&adx_seed, dx_seed[index]);
        }
        adx = __ddiv_rn(neumaier_finish_v1(adx_seed), 14.0);
        adx_live = true;
        dx_seed_count = 0;
      }
    }
    if (row < 27U) {
      continue;
    }
    if (!adx_live) {
      set_invalid_v3(values, validity, rows, 2, row, kZeroDenominatorV3);
      set_invalid_v3(values, validity, rows, 4, row, kZeroDenominatorV3);
    } else if (!finite_v3(adx)) {
      adx_live = false;
      set_invalid_v3(values, validity, rows, 2, row, kComputeFailureV3);
      set_invalid_v3(values, validity, rows, 4, row, kComputeFailureV3);
    } else {
      set_valid_v3(values, validity, rows, 2, row, adx);
      set_valid_v3(values, validity, rows, 4, row,
                   adx > 25.0 ? direction : 0.0);
    }
  }
}

__device__ void compute_cusum_lane_v3(
    const double* close, std::size_t rows, double scale_anchor, double* values,
    unsigned char* validity) {
  for (std::size_t row = 0; row < rows; ++row) {
    initialize_slot_v3(values, validity, rows, 10, row);
    initialize_slot_v3(values, validity, rows, 11, row);
    initialize_slot_v3(values, validity, rows, 12, row);
  }
  double previous_up = 0.0;
  double previous_down = 0.0;
  for (std::size_t row = 50U; row < rows; ++row) {
    NeumaierV1 mean_sum = neumaier_zero_v1();
    for (std::size_t j = row - 50U; j < row; ++j) {
      neumaier_add_v1(&mean_sum, scaled_v3(close, j, scale_anchor));
    }
    const double mean = __ddiv_rn(neumaier_finish_v1(mean_sum), 50.0);
    NeumaierV1 variance_sum = neumaier_zero_v1();
    for (std::size_t j = row - 50U; j < row; ++j) {
      const double deviation =
          __dsub_rn(scaled_v3(close, j, scale_anchor), mean);
      neumaier_add_v1(&variance_sum, __dmul_rn(deviation, deviation));
    }
    const double variance =
        __ddiv_rn(neumaier_finish_v1(variance_sum), 49.0);
    unsigned char invalid_reason = kValidV3;
    if (!finite_v3(mean) || !finite_v3(variance) || variance < 0.0) {
      invalid_reason = kComputeFailureV3;
    } else if (variance == 0.0) {
      invalid_reason = kZeroDenominatorV3;
    }
    if (invalid_reason != kValidV3) {
      previous_up = 0.0;
      previous_down = 0.0;
      set_invalid_v3(values, validity, rows, 10, row, invalid_reason);
      set_invalid_v3(values, validity, rows, 11, row, invalid_reason);
      set_invalid_v3(values, validity, rows, 12, row, invalid_reason);
      continue;
    }
    const double standard_deviation = __dsqrt_rn(variance);
    const double z = __ddiv_rn(
        __dsub_rn(scaled_v3(close, row, scale_anchor), mean),
        standard_deviation);
    const double raw_up = __dsub_rn(__dadd_rn(previous_up, z), 0.5);
    const double raw_down = __dsub_rn(__dsub_rn(previous_down, z), 0.5);
    const double candidate_up = raw_up > 0.0 ? raw_up : 0.0;
    const double candidate_down = raw_down > 0.0 ? raw_down : 0.0;
    if (!finite_v3(standard_deviation) || !finite_v3(z) ||
        !finite_v3(candidate_up) || !finite_v3(candidate_down)) {
      previous_up = 0.0;
      previous_down = 0.0;
      set_invalid_v3(values, validity, rows, 10, row, kComputeFailureV3);
      set_invalid_v3(values, validity, rows, 11, row, kComputeFailureV3);
      set_invalid_v3(values, validity, rows, 12, row, kComputeFailureV3);
      continue;
    }
    double up = candidate_up;
    double down = candidate_down;
    double signal = 0.0;
    if (candidate_up > 3.0) {
      up = 0.0;
      signal = 1.0;
    } else if (candidate_down > 3.0) {
      down = 0.0;
      signal = -1.0;
    }
    previous_up = up;
    previous_down = down;
    set_valid_v3(values, validity, rows, 10, row, up);
    set_valid_v3(values, validity, rows, 11, row, down);
    set_valid_v3(values, validity, rows, 12, row, signal);
  }
}

__global__ void regime_recurrence_kernel_v3(
    const double* high, const double* low, const double* close,
    std::size_t rows, double scale_anchor, double* values,
    unsigned char* validity) {
  if (blockIdx.x != 0U) {
    return;
  }
  if (threadIdx.x == 0U) {
    compute_wilder_lane_v3(high, low, close, rows, scale_anchor, values,
                           validity);
  } else if (threadIdx.x == 1U) {
    compute_cusum_lane_v3(close, rows, scale_anchor, values, validity);
  }
}

int launch_status_v3() {
  return static_cast<int>(cudaGetLastError());
}

}  // namespace

extern "C" int neoethos_resident_regime_independent_f64_v3(
    const double* open, const double* high, const double* low,
    const double* close, std::size_t rows, double scale_anchor,
    double* feature_values, unsigned char* feature_validity_u8,
    cudaStream_t stream) {
  if (open == nullptr || high == nullptr || low == nullptr || close == nullptr ||
      feature_values == nullptr || feature_validity_u8 == nullptr || rows == 0U ||
      !std::isfinite(scale_anchor) || scale_anchor <= 0.0 || stream == nullptr) {
    return -1;
  }
  constexpr unsigned int threads = 256U;
  const std::size_t needed = (rows + threads - 1U) / threads;
  const unsigned int blocks = static_cast<unsigned int>(needed < 65535U ? needed : 65535U);
  regime_independent_kernel_v3<<<blocks, threads, 0, stream>>>(
      open, high, low, close, rows, scale_anchor, feature_values,
      feature_validity_u8);
  return launch_status_v3();
}

extern "C" int neoethos_resident_regime_recurrence_f64_v3(
    const double* high, const double* low, const double* close,
    std::size_t rows, double scale_anchor, double* feature_values,
    unsigned char* feature_validity_u8, cudaStream_t stream) {
  if (high == nullptr || low == nullptr || close == nullptr ||
      feature_values == nullptr || feature_validity_u8 == nullptr || rows == 0U ||
      !std::isfinite(scale_anchor) || scale_anchor <= 0.0 || stream == nullptr) {
    return -1;
  }
  regime_recurrence_kernel_v3<<<1, 2, 0, stream>>>(
      high, low, close, rows, scale_anchor, feature_values,
      feature_validity_u8);
  return launch_status_v3();
}
