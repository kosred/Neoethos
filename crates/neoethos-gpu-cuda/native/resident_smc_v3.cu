#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr int kSmcFeatureColumnsV3 = 46;
constexpr int kSmcParentSlotsV3 = 11;
constexpr int kSwingMemoryV3 = 15;
constexpr int kFvgMemoryV3 = 64;
// The scalar authority pushes one gap and only then evicts the oldest entry
// when the logical count exceeds 64. Keep one transient slot so that exact
// ordering never becomes an out-of-bounds write.
constexpr int kFvgStorageV3 = kFvgMemoryV3 + 1;
constexpr int kOrderBlockMemoryV3 = 64;
constexpr int kEventHistoryV3 = 64;
constexpr double kEpsilonV3 = 1.0e-12;
constexpr double kPositiveInfinityV3 = std::numeric_limits<double>::infinity();

enum SmcColumnV3 : int {
  kOb = 0,
  kFvg,
  kIfvg,
  kLiquiditySweep,
  kPdArray,
  kKillzone,
  kDisplacement,
  kBreakerBlock,
  kMitigationBlock,
  kMss,
  kVolumeImbalance,
  kBos,
  kEqh,
  kEql,
  kInducement,
  kAsianRange,
  kSilverBullet,
  kJudasSwing,
  kNwog,
  kNdog,
  kIctMacro,
  kFvgStrength,
  kDealingRangeWidth,
  kSwingRangePct,
  kObStrength,
  kTrendBias,
  kUnicornModel,
  kRejectionBlock,
  kPropulsionBlock,
  kFibTimeRatio,
  kFib236,
  kFib382,
  kFib500,
  kFib618,
  kFib705,
  kFib786,
  kFib886,
  kFib1272,
  kFib1414,
  kFib1618,
  kFib2000,
  kFib2618,
  kFvgMagnetDist,
  kFvgMagnetAge,
  kFvgInside,
  kFvgOpenCount,
};

enum SmcValidityV3 : unsigned char {
  kValid = 0,
  kWarmup = 1,
  kMissingInput = 2,
  kGap = 3,
  kStale = 4,
  kZeroDenominator = 5,
  kDegenerate = 6,
  kNonFinite = 7,
  kComputeFailure = 8,
  kAlignmentMissing = 9,
};

__device__ __forceinline__ double abs_v3(double value) {
  return value < 0.0 ? -value : value;
}

__device__ __forceinline__ double min_v3(double left, double right) {
  return left < right ? left : right;
}

__device__ __forceinline__ double max_v3(double left, double right) {
  return left > right ? left : right;
}

__device__ __forceinline__ double clamp_v3(double value, double low, double high) {
  return min_v3(max_v3(value, low), high);
}

__device__ __forceinline__ double canonical_nan_v3() {
  return __longlong_as_double(static_cast<long long>(0x7ff8000000000000ULL));
}

// Exact transcription of Data's semantic-v3 fixed-order authority. No libc,
// libdevice log1p, tolerance, FMA, or platform dispatch participates.
__device__ __forceinline__ double neoethos_smc_log1p_cpu_exact_v1(
    std::uint64_t age) {
  const double one_plus_age = static_cast<double>(age) + 1.0;
  const std::uint64_t bits = static_cast<std::uint64_t>(__double_as_longlong(one_plus_age));
  const int exponent = static_cast<int>((bits >> 52U) & 0x7ffU) - 1023;
  const std::uint64_t mantissa_bits =
      (bits & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL;
  const double mantissa = __longlong_as_double(static_cast<long long>(mantissa_bits));
  const double z = (mantissa - 1.0) / (mantissa + 1.0);
  const double z_squared = z * z;
  double term = z;
  double sum = z;
  for (unsigned int denominator = 3U; denominator <= 49U; denominator += 2U) {
    term = term * z_squared;
    sum = sum + term / static_cast<double>(denominator);
  }
  const double ln_two =
      __longlong_as_double(static_cast<long long>(0x3fe62e42fefa39efULL));
  return static_cast<double>(exponent) * ln_two + 2.0 * sum;
}

__device__ __forceinline__ std::int64_t floor_div_v3(std::int64_t value,
                                                     std::int64_t divisor) {
  std::int64_t quotient = value / divisor;
  const std::int64_t remainder = value % divisor;
  if (remainder < 0) {
    --quotient;
  }
  return quotient;
}

__device__ __forceinline__ std::int64_t positive_mod_v3(std::int64_t value,
                                                        std::int64_t divisor) {
  const std::int64_t remainder = value % divisor;
  return remainder < 0 ? remainder + divisor : remainder;
}

__device__ __forceinline__ bool leap_year_v3(std::int64_t year) {
  return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}

struct CivilTimeV3 {
  std::int64_t year;
  int month;
  int day;
  int ordinal;
  int hour;
  int minute;
  int iso_week;
};

__device__ __forceinline__ std::int64_t days_from_civil_v3(std::int64_t year,
                                                           unsigned int month,
                                                           unsigned int day) {
  year -= month <= 2U ? 1 : 0;
  const std::int64_t era = floor_div_v3(year, 400);
  const unsigned int year_of_era = static_cast<unsigned int>(year - era * 400);
  const unsigned int shifted_month = month > 2U ? month - 3U : month + 9U;
  const unsigned int day_of_year = (153U * shifted_month + 2U) / 5U + day - 1U;
  const unsigned int day_of_era = year_of_era * 365U + year_of_era / 4U -
                                  year_of_era / 100U + day_of_year;
  return era * 146097 + static_cast<std::int64_t>(day_of_era) - 719468;
}

__device__ __forceinline__ int weekday_monday_one_v3(std::int64_t epoch_days) {
  return static_cast<int>(positive_mod_v3(epoch_days + 3, 7)) + 1;
}

__device__ __forceinline__ int weeks_in_year_v3(std::int64_t year) {
  const int january_first = weekday_monday_one_v3(days_from_civil_v3(year, 1U, 1U));
  return january_first == 4 || (january_first == 3 && leap_year_v3(year)) ? 53 : 52;
}

__device__ __forceinline__ bool civil_time_v3(std::int64_t timestamp_ms,
                                              CivilTimeV3* output) {
  constexpr std::int64_t kMillisecondsPerDay = 86400000;
  constexpr std::int64_t kMillisecondsPerHour = 3600000;
  constexpr std::int64_t kMillisecondsPerMinute = 60000;
  const std::int64_t epoch_days = floor_div_v3(timestamp_ms, kMillisecondsPerDay);
  const std::int64_t millis_of_day =
      positive_mod_v3(timestamp_ms, kMillisecondsPerDay);

  std::int64_t shifted = epoch_days + 719468;
  const std::int64_t era = floor_div_v3(shifted, 146097);
  const unsigned int day_of_era = static_cast<unsigned int>(shifted - era * 146097);
  const unsigned int year_of_era =
      (day_of_era - day_of_era / 1460U + day_of_era / 36524U -
       day_of_era / 146096U) /
      365U;
  std::int64_t year = static_cast<std::int64_t>(year_of_era) + era * 400;
  const unsigned int day_of_year_march =
      day_of_era - (365U * year_of_era + year_of_era / 4U - year_of_era / 100U);
  const unsigned int shifted_month = (5U * day_of_year_march + 2U) / 153U;
  const unsigned int day =
      day_of_year_march - (153U * shifted_month + 2U) / 5U + 1U;
  const int month = static_cast<int>(shifted_month) + (shifted_month < 10U ? 3 : -9);
  year += month <= 2 ? 1 : 0;

  constexpr int kMonthOffsets[12] = {0, 31, 59, 90, 120, 151,
                                     181, 212, 243, 273, 304, 334};
  int ordinal = kMonthOffsets[month - 1] + static_cast<int>(day);
  if (month > 2 && leap_year_v3(year)) {
    ++ordinal;
  }
  const int weekday = weekday_monday_one_v3(epoch_days);
  int iso_week = (ordinal - weekday + 10) / 7;
  if (iso_week < 1) {
    iso_week = weeks_in_year_v3(year - 1);
  } else if (iso_week > weeks_in_year_v3(year)) {
    iso_week = 1;
  }

  output->year = year;
  output->month = month;
  output->day = static_cast<int>(day);
  output->ordinal = ordinal;
  output->hour = static_cast<int>(millis_of_day / kMillisecondsPerHour);
  output->minute = static_cast<int>(
      (millis_of_day % kMillisecondsPerHour) / kMillisecondsPerMinute);
  output->iso_week = iso_week;
  return true;
}

struct Sha256StateV3 {
  std::uint32_t state[8];
  unsigned char block[64];
  std::uint64_t total_bytes;
  unsigned int block_len;
};

__device__ __forceinline__ std::uint32_t rotate_right_v3(std::uint32_t value,
                                                         unsigned int shift) {
  return (value >> shift) | (value << (32U - shift));
}

__device__ __forceinline__ void sha256_initialize_v3(Sha256StateV3* state) {
  const std::uint32_t initial[8] = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U,
                                    0xa54ff53aU, 0x510e527fU, 0x9b05688cU,
                                    0x1f83d9abU, 0x5be0cd19U};
  for (int index = 0; index < 8; ++index) {
    state->state[index] = initial[index];
  }
  state->total_bytes = 0;
  state->block_len = 0;
}

__device__ __forceinline__ void sha256_compress_v3(Sha256StateV3* state) {
  constexpr std::uint32_t constants[64] = {
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
      0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
      0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
      0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
      0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
      0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
      0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
      0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
      0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
      0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
  std::uint32_t words[64];
  for (int index = 0; index < 16; ++index) {
    const int offset = index * 4;
    words[index] = (static_cast<std::uint32_t>(state->block[offset]) << 24U) |
                   (static_cast<std::uint32_t>(state->block[offset + 1]) << 16U) |
                   (static_cast<std::uint32_t>(state->block[offset + 2]) << 8U) |
                   static_cast<std::uint32_t>(state->block[offset + 3]);
  }
  for (int index = 16; index < 64; ++index) {
    const std::uint32_t s0 = rotate_right_v3(words[index - 15], 7U) ^
                             rotate_right_v3(words[index - 15], 18U) ^
                             (words[index - 15] >> 3U);
    const std::uint32_t s1 = rotate_right_v3(words[index - 2], 17U) ^
                             rotate_right_v3(words[index - 2], 19U) ^
                             (words[index - 2] >> 10U);
    words[index] = words[index - 16] + s0 + words[index - 7] + s1;
  }
  std::uint32_t a = state->state[0];
  std::uint32_t b = state->state[1];
  std::uint32_t c = state->state[2];
  std::uint32_t d = state->state[3];
  std::uint32_t e = state->state[4];
  std::uint32_t f = state->state[5];
  std::uint32_t g = state->state[6];
  std::uint32_t h = state->state[7];
  for (int index = 0; index < 64; ++index) {
    const std::uint32_t sigma1 = rotate_right_v3(e, 6U) ^ rotate_right_v3(e, 11U) ^
                                 rotate_right_v3(e, 25U);
    const std::uint32_t choice = (e & f) ^ ((~e) & g);
    const std::uint32_t temporary1 = h + sigma1 + choice + constants[index] + words[index];
    const std::uint32_t sigma0 = rotate_right_v3(a, 2U) ^ rotate_right_v3(a, 13U) ^
                                 rotate_right_v3(a, 22U);
    const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t temporary2 = sigma0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temporary1;
    d = c;
    c = b;
    b = a;
    a = temporary1 + temporary2;
  }
  state->state[0] += a;
  state->state[1] += b;
  state->state[2] += c;
  state->state[3] += d;
  state->state[4] += e;
  state->state[5] += f;
  state->state[6] += g;
  state->state[7] += h;
}

__device__ __forceinline__ void sha256_update_byte_v3(Sha256StateV3* state,
                                                       unsigned char byte) {
  state->block[state->block_len++] = byte;
  ++state->total_bytes;
  if (state->block_len == 64U) {
    sha256_compress_v3(state);
    state->block_len = 0;
  }
}

__device__ __forceinline__ void sha256_update_bytes_v3(Sha256StateV3* state,
                                                        const unsigned char* bytes,
                                                        std::size_t length) {
  for (std::size_t index = 0; index < length; ++index) {
    sha256_update_byte_v3(state, bytes[index]);
  }
}

__device__ __forceinline__ void sha256_finalize_v3(Sha256StateV3* state,
                                                    unsigned char* digest) {
  const std::uint64_t bit_length = state->total_bytes * 8U;
  sha256_update_byte_v3(state, 0x80U);
  while (state->block_len != 56U) {
    sha256_update_byte_v3(state, 0U);
  }
  for (int shift = 56; shift >= 0; shift -= 8) {
    sha256_update_byte_v3(state, static_cast<unsigned char>(bit_length >> shift));
  }
  for (int index = 0; index < 8; ++index) {
    digest[index * 4] = static_cast<unsigned char>(state->state[index] >> 24U);
    digest[index * 4 + 1] = static_cast<unsigned char>(state->state[index] >> 16U);
    digest[index * 4 + 2] = static_cast<unsigned char>(state->state[index] >> 8U);
    digest[index * 4 + 3] = static_cast<unsigned char>(state->state[index]);
  }
}

template <typename T>
__device__ __forceinline__ void sha256_array_v3(const T* values,
                                                std::size_t count,
                                                unsigned char* digest) {
  Sha256StateV3 state;
  sha256_initialize_v3(&state);
  sha256_update_bytes_v3(&state, reinterpret_cast<const unsigned char*>(values),
                         count * sizeof(T));
  sha256_finalize_v3(&state, digest);
}

__device__ __forceinline__ void shift_left_v3(double* values, int* indices,
                                              int count) {
  for (int index = 1; index < count; ++index) {
    values[index - 1] = values[index];
    if (indices != nullptr) {
      indices[index - 1] = indices[index];
    }
  }
}

__device__ __forceinline__ signed char quantize_direction_v3(double value,
                                                             unsigned char validity) {
  if (validity != kValid) {
    return 0;
  }
  if (value > 1.0e-9) {
    return 1;
  }
  if (value < -1.0e-9) {
    return -1;
  }
  return 0;
}

__device__ __forceinline__ signed char quantize_binary_v3(double value,
                                                          unsigned char validity) {
  return validity == kValid && value > 1.0e-9 ? 1 : 0;
}

__device__ __forceinline__ void mark_warmup_v3(unsigned char* validity,
                                                int column,
                                                std::size_t row,
                                                std::size_t required) {
  if (row < required) {
    validity[column] = kWarmup;
  }
}

__global__ void resident_smc_parent_features_f64_v3(
    const double* open, const double* high, const double* low,
    const double* close, const std::int64_t* timestamps, std::size_t rows,
    double* smc_feature_values, unsigned char* smc_feature_validity_u8,
    std::int64_t* months, std::int64_t* days, signed char* smc_parent_rows,
    unsigned char* generated_parent_hashes, unsigned int* device_error) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  *device_error = 0U;

  double swing_highs[kSwingMemoryV3];
  double swing_lows[kSwingMemoryV3];
  int swing_high_count = 0;
  int swing_low_count = 0;

  double buy_fvg_top[kFvgStorageV3];
  double buy_fvg_bottom[kFvgStorageV3];
  int buy_fvg_born[kFvgStorageV3];
  int buy_fvg_count = 0;
  double sell_fvg_top[kFvgStorageV3];
  double sell_fvg_bottom[kFvgStorageV3];
  int sell_fvg_born[kFvgStorageV3];
  int sell_fvg_count = 0;

  double buy_ob_top[kOrderBlockMemoryV3];
  double buy_ob_bottom[kOrderBlockMemoryV3];
  int buy_ob_born[kOrderBlockMemoryV3];
  int buy_ob_count = 0;
  double sell_ob_top[kOrderBlockMemoryV3];
  double sell_ob_bottom[kOrderBlockMemoryV3];
  int sell_ob_born[kOrderBlockMemoryV3];
  int sell_ob_count = 0;

  double history_ob[kEventHistoryV3] = {};
  double history_fvg[kEventHistoryV3] = {};
  double history_liquidity[kEventHistoryV3] = {};
  double history_mss[kEventHistoryV3] = {};
  double history_displacement[kEventHistoryV3] = {};
  double history_breaker[kEventHistoryV3] = {};
  signed char canonical_parent_trend_previous = 0;

  std::size_t latest_bull_sweep = 0;
  std::size_t latest_bear_sweep = 0;
  double last_confirmed_high = -kPositiveInfinityV3;
  double last_confirmed_low = kPositiveInfinityV3;
  double asian_high = -kPositiveInfinityV3;
  double asian_low = kPositiveInfinityV3;
  bool asian_range_set = false;
  double previous_day_close = canonical_nan_v3();
  double previous_week_close = canonical_nan_v3();
  std::int64_t last_day_key = -1;
  std::int64_t last_week_key = -1;
  double atr_sum = 0.0;
  std::size_t atr_count = 0;
  std::size_t consolidation_count = 0;

  for (std::size_t row = 0; row < rows; ++row) {
    double values[kSmcFeatureColumnsV3] = {};
    unsigned char validity[kSmcFeatureColumnsV3];
    for (int column = 0; column < kSmcFeatureColumnsV3; ++column) {
      validity[column] = kValid;
    }

    const double row_open = open[row];
    const double row_high = high[row];
    const double row_low = low[row];
    const double row_close = close[row];
    if (!isfinite(row_open) || !isfinite(row_high) || !isfinite(row_low) ||
        !isfinite(row_close) || row_open <= 0.0 || row_high <= 0.0 ||
        row_low <= 0.0 || row_close <= 0.0 ||
        row_low > min_v3(row_open, row_close) ||
        row_high < max_v3(row_open, row_close)) {
      *device_error = 1U;
      return;
    }

    if (row > 0) {
      const double true_range =
          max_v3(row_high - row_low,
                 max_v3(abs_v3(row_high - close[row - 1]),
                        abs_v3(row_low - close[row - 1])));
      atr_sum = atr_sum + true_range;
      ++atr_count;
    }
    const double running_atr =
        atr_count > 0 ? atr_sum / static_cast<double>(atr_count)
                      : max_v3(row_high - row_low, 1.0e-10);
    const double validity_running_atr =
        atr_count > 0 ? atr_sum / static_cast<double>(atr_count)
                      : row_high - row_low;

    CivilTimeV3 civil;
    if (!civil_time_v3(timestamps[row], &civil)) {
      *device_error = 2U;
      return;
    }
    months[row] = civil.year * 12 + civil.month;
    days[row] = civil.year * 10000 + civil.month * 100 + civil.day;
    const int hour = civil.hour;
    const int minute = civil.minute;
    const int hour_minute = hour * 60 + minute;
    values[kKillzone] =
        ((hour >= 7 && hour < 11) || (hour >= 13 && hour < 17)) ? 1.0 : 0.0;
    if ((hour_minute >= 590 && hour_minute <= 610) ||
        (hour_minute >= 650 && hour_minute <= 670) ||
        (hour_minute >= 790 && hour_minute <= 820) ||
        (hour_minute >= 890 && hour_minute <= 910) ||
        (hour_minute >= 915 && hour_minute <= 945)) {
      values[kIctMacro] = 1.0;
    }
    if (hour == 10 || hour == 14 || hour == 18) {
      values[kSilverBullet] = 1.0;
    }

    if (hour < 8) {
      asian_high = max_v3(asian_high, row_high);
      asian_low = min_v3(asian_low, row_low);
      asian_range_set = true;
    } else if (hour == 8 && minute == 0) {
      asian_range_set = true;
    }
    const bool asian_set_before_reset = asian_range_set;
    const double asian_range_before_reset = asian_high - asian_low;
    if (asian_range_set && asian_high > asian_low) {
      values[kAsianRange] = (row_close - asian_low) / (asian_high - asian_low);
    }

    const std::int64_t day_key = civil.ordinal + civil.year * 400;
    const std::int64_t week_key = civil.iso_week + civil.year * 60;
    const bool day_changed = day_key != last_day_key;
    const bool week_changed = week_key != last_week_key;
    if (day_changed) {
      if (isfinite(previous_day_close)) {
        values[kNdog] = (row_open - previous_day_close) / max_v3(running_atr, 1.0e-10);
      }
      previous_day_close = row > 0 ? close[row - 1] : row_close;
      last_day_key = day_key;
    }
    if (week_changed) {
      if (isfinite(previous_week_close)) {
        values[kNwog] = (row_open - previous_week_close) / max_v3(running_atr, 1.0e-10);
      }
      previous_week_close = row > 0 ? close[row - 1] : row_close;
      last_week_key = week_key;
    }
    if ((hour == 7 || hour == 13) && minute < 30 && row >= 1) {
      const double previous_close = close[row - 1];
      if (abs_v3(row_high - previous_close) > running_atr * 0.5 &&
          row_close < previous_close) {
        values[kJudasSwing] = -1.0;
      } else if (abs_v3(previous_close - row_low) > running_atr * 0.5 &&
                 row_close > previous_close) {
        values[kJudasSwing] = 1.0;
      }
    }

    if (row >= 1) {
      const double previous_open = open[row - 1];
      const double previous_close = close[row - 1];
      const double previous_body_top = max_v3(previous_open, previous_close);
      const double previous_body_bottom = min_v3(previous_open, previous_close);
      const double current_body_top = max_v3(row_open, row_close);
      const double current_body_bottom = min_v3(row_open, row_close);
      if (current_body_bottom > previous_body_top && row_low <= high[row - 1]) {
        values[kVolumeImbalance] = 1.0;
      } else if (current_body_top < previous_body_bottom && row_high >= low[row - 1]) {
        values[kVolumeImbalance] = -1.0;
      }
    }

    double dealing_high = -kPositiveInfinityV3;
    double dealing_low = kPositiveInfinityV3;
    if (row >= 40) {
      for (std::size_t index = row - 40; index < row; ++index) {
        dealing_high = max_v3(dealing_high, high[index]);
        dealing_low = min_v3(dealing_low, low[index]);
      }
      const double equilibrium = (dealing_high + dealing_low) / 2.0;
      values[kPdArray] = row_close > equilibrium ? 1.0 : -1.0;
      const double range = dealing_high - dealing_low;
      if (range > 1.0e-9) {
        const double dist236 = min_v3(abs_v3(row_close - (dealing_low + range * 0.236)) / range, 1.0);
        const double dist382 = min_v3(abs_v3(row_close - (dealing_low + range * 0.382)) / range, 1.0);
        const double dist500 = min_v3(abs_v3(row_close - equilibrium) / range, 1.0);
        const double dist618 = min_v3(abs_v3(row_close - (dealing_low + range * 0.618)) / range, 1.0);
        const double dist705 = min_v3(abs_v3(row_close - (dealing_low + range * 0.705)) / range, 1.0);
        const double dist786 = min_v3(abs_v3(row_close - (dealing_low + range * 0.786)) / range, 1.0);
        const double dist886 = min_v3(abs_v3(row_close - (dealing_low + range * 0.886)) / range, 1.0);
        const double dist1272 = min_v3(abs_v3(row_close - (dealing_low + range * 1.272)) / range, 1.0);
        const double dist1414 = min_v3(abs_v3(row_close - (dealing_low + range * 1.414)) / range, 1.0);
        const double dist1618 = min_v3(abs_v3(row_close - (dealing_low + range * 1.618)) / range, 1.0);
        const double dist2000 = min_v3(abs_v3(row_close - (dealing_low + range * 2.0)) / range, 1.0);
        const double dist2618 = min_v3(abs_v3(row_close - (dealing_low + range * 2.618)) / range, 1.0);
        values[kFib236] = max_v3(1.0 - 15.0 * dist236, 0.0);
        values[kFib382] = max_v3(1.0 - 15.0 * dist382, 0.0);
        values[kFib500] = max_v3(1.0 - 15.0 * dist500, 0.0);
        values[kFib618] = max_v3(1.0 - 15.0 * dist618, 0.0);
        values[kFib705] = max_v3(1.0 - 15.0 * dist705, 0.0);
        values[kFib786] = max_v3(1.0 - 15.0 * dist786, 0.0);
        values[kFib886] = max_v3(1.0 - 15.0 * dist886, 0.0);
        values[kFib1272] = max_v3(1.0 - 10.0 * dist1272, 0.0);
        values[kFib1414] = max_v3(1.0 - 10.0 * dist1414, 0.0);
        values[kFib1618] = max_v3(1.0 - 10.0 * dist1618, 0.0);
        values[kFib2000] = max_v3(1.0 - 5.0 * dist2000, 0.0);
        values[kFib2618] = max_v3(1.0 - 5.0 * dist2618, 0.0);
      }
    }

    if (row >= 20) {
      const double body = abs_v3(row_close - row_open);
      double average_body = 0.0;
      for (std::size_t index = row - 20; index < row; ++index) {
        average_body = average_body + abs_v3(close[index] - open[index]);
      }
      average_body = average_body / 20.0;
      if (average_body > 1.0e-12 && body >= 1.8 * average_body) {
        values[kDisplacement] = row_close > row_open ? 1.0 : -1.0;
      }
    }

    if (row >= 10) {
      const std::size_t center = row - 5;
      const double center_high = high[center];
      const double center_low = low[center];
      bool is_swing_high = true;
      bool is_swing_low = true;
      for (std::size_t index = center - 5; index <= center + 5; ++index) {
        if (index == center) {
          continue;
        }
        if (high[index] >= center_high) {
          is_swing_high = false;
        }
        if (low[index] <= center_low) {
          is_swing_low = false;
        }
      }
      if (is_swing_high) {
        if (swing_high_count == kSwingMemoryV3) {
          shift_left_v3(swing_highs, nullptr, swing_high_count);
          --swing_high_count;
        }
        swing_highs[swing_high_count++] = center_high;
      }
      if (is_swing_low) {
        if (swing_low_count == kSwingMemoryV3) {
          shift_left_v3(swing_lows, nullptr, swing_low_count);
          --swing_low_count;
        }
        swing_lows[swing_low_count++] = center_low;
      }
    }

    for (int index = 0; index < swing_high_count; ++index) {
      if (row_high > swing_highs[index] && row_close < swing_highs[index]) {
        values[kLiquiditySweep] = -1.0;
        latest_bear_sweep = row;
      }
    }
    for (int index = 0; index < swing_low_count; ++index) {
      if (row_low < swing_lows[index] && row_close > swing_lows[index]) {
        values[kLiquiditySweep] = 1.0;
        latest_bull_sweep = row;
      }
    }

    if (values[kDisplacement] == 1.0 && latest_bull_sweep > 0 &&
        row - latest_bull_sweep <= 15 && swing_high_count > 0) {
      const double recent_high = swing_highs[swing_high_count - 1];
      if (row_close > recent_high && close[row - 1] <= recent_high) {
        values[kMss] = 1.0;
      }
    }
    if (values[kDisplacement] == -1.0 && latest_bear_sweep > 0 &&
        row - latest_bear_sweep <= 15 && swing_low_count > 0) {
      const double recent_low = swing_lows[swing_low_count - 1];
      if (row_close < recent_low && close[row - 1] >= recent_low) {
        values[kMss] = -1.0;
      }
    }

    if (row >= 3) {
      const bool was_bull_sweep =
          history_liquidity[(row - 1) % kEventHistoryV3] == 1.0 ||
          history_liquidity[(row - 2) % kEventHistoryV3] == 1.0 ||
          history_liquidity[(row - 3) % kEventHistoryV3] == 1.0;
      const bool was_bear_sweep =
          history_liquidity[(row - 1) % kEventHistoryV3] == -1.0 ||
          history_liquidity[(row - 2) % kEventHistoryV3] == -1.0 ||
          history_liquidity[(row - 3) % kEventHistoryV3] == -1.0;
      const double previous_open = open[row - 1];
      const double previous_close = close[row - 1];
      const double previous_high = high[row - 1];
      const double previous_low = low[row - 1];
      if (was_bull_sweep && row_close > row_open && previous_close < previous_open &&
          row_close >= previous_high) {
        if (buy_ob_count == kOrderBlockMemoryV3) {
          *device_error = 3U;
          return;
        }
        values[kOb] = 1.0;
        buy_ob_born[buy_ob_count] = static_cast<int>(row);
        buy_ob_top[buy_ob_count] = previous_high;
        buy_ob_bottom[buy_ob_count] = previous_low;
        ++buy_ob_count;
      }
      if (was_bear_sweep && row_close < row_open && previous_close > previous_open &&
          row_close <= previous_low) {
        if (sell_ob_count == kOrderBlockMemoryV3) {
          *device_error = 4U;
          return;
        }
        values[kOb] = -1.0;
        sell_ob_born[sell_ob_count] = static_cast<int>(row);
        sell_ob_top[sell_ob_count] = previous_high;
        sell_ob_bottom[sell_ob_count] = previous_low;
        ++sell_ob_count;
      }
      if (!was_bull_sweep && row_close > row_open && previous_close < previous_open &&
          row_close >= previous_high && values[kDisplacement] == 1.0) {
        values[kMitigationBlock] = 1.0;
      }
      if (!was_bear_sweep && row_close < row_open && previous_close > previous_open &&
          row_close <= previous_low && values[kDisplacement] == -1.0) {
        values[kMitigationBlock] = -1.0;
      }
    }

    int write = 0;
    for (int index = 0; index < sell_ob_count; ++index) {
      if (row - static_cast<std::size_t>(sell_ob_born[index]) < 50 &&
          values[kDisplacement] == 1.0 && row_close > sell_ob_top[index]) {
        values[kBreakerBlock] = 1.0;
      } else if (row - static_cast<std::size_t>(sell_ob_born[index]) < 50) {
        sell_ob_born[write] = sell_ob_born[index];
        sell_ob_top[write] = sell_ob_top[index];
        sell_ob_bottom[write] = sell_ob_bottom[index];
        ++write;
      }
    }
    sell_ob_count = write;
    write = 0;
    for (int index = 0; index < buy_ob_count; ++index) {
      if (row - static_cast<std::size_t>(buy_ob_born[index]) < 50 &&
          values[kDisplacement] == -1.0 && row_close < buy_ob_bottom[index]) {
        values[kBreakerBlock] = -1.0;
      } else if (row - static_cast<std::size_t>(buy_ob_born[index]) < 50) {
        buy_ob_born[write] = buy_ob_born[index];
        buy_ob_top[write] = buy_ob_top[index];
        buy_ob_bottom[write] = buy_ob_bottom[index];
        ++write;
      }
    }
    buy_ob_count = write;

    if (row >= 2) {
      const double sell_bottom = row_high;
      const double sell_top = low[row - 2];
      if (sell_bottom < sell_top) {
        values[kFvg] = -1.0;
        sell_fvg_top[sell_fvg_count] = sell_top;
        sell_fvg_bottom[sell_fvg_count] = sell_bottom;
        sell_fvg_born[sell_fvg_count] = static_cast<int>(row);
        ++sell_fvg_count;
      }
      const double buy_top = row_low;
      const double buy_bottom = high[row - 2];
      if (buy_top > buy_bottom) {
        values[kFvg] = 1.0;
        buy_fvg_top[buy_fvg_count] = buy_top;
        buy_fvg_bottom[buy_fvg_count] = buy_bottom;
        buy_fvg_born[buy_fvg_count] = static_cast<int>(row);
        ++buy_fvg_count;
      }
    }

    write = 0;
    for (int index = 0; index < sell_fvg_count; ++index) {
      if (row_close > sell_fvg_top[index]) {
        values[kIfvg] = 1.0;
      } else {
        sell_fvg_top[write] = sell_fvg_top[index];
        sell_fvg_bottom[write] = sell_fvg_bottom[index];
        sell_fvg_born[write] = sell_fvg_born[index];
        ++write;
      }
    }
    sell_fvg_count = write;
    write = 0;
    for (int index = 0; index < buy_fvg_count; ++index) {
      if (row_close < buy_fvg_bottom[index]) {
        values[kIfvg] = -1.0;
      } else {
        buy_fvg_top[write] = buy_fvg_top[index];
        buy_fvg_bottom[write] = buy_fvg_bottom[index];
        buy_fvg_born[write] = buy_fvg_born[index];
        ++write;
      }
    }
    buy_fvg_count = write;
    if (buy_fvg_count > kFvgMemoryV3) {
      shift_left_v3(buy_fvg_top, buy_fvg_born, buy_fvg_count);
      shift_left_v3(buy_fvg_bottom, nullptr, buy_fvg_count);
      --buy_fvg_count;
    }
    if (sell_fvg_count > kFvgMemoryV3) {
      shift_left_v3(sell_fvg_top, sell_fvg_born, sell_fvg_count);
      shift_left_v3(sell_fvg_bottom, nullptr, sell_fvg_count);
      --sell_fvg_count;
    }

    double best_distance_atr = kPositiveInfinityV3;
    double best_signed = 0.0;
    double best_age = 0.0;
    const double atr_for_magnets = max_v3(running_atr, 1.0e-10);
    for (int index = 0; index < buy_fvg_count; ++index) {
      const double midpoint = (buy_fvg_top[index] + buy_fvg_bottom[index]) / 2.0;
      const double signed_distance = (midpoint - row_close) / atr_for_magnets;
      const double distance = abs_v3(signed_distance);
      if (distance < best_distance_atr) {
        best_distance_atr = distance;
        best_signed = clamp_v3(signed_distance, -10.0, 10.0);
        best_age = neoethos_smc_log1p_cpu_exact_v1(
                       row - static_cast<std::size_t>(buy_fvg_born[index])) /
                   9.21;
      }
      if (row_close <= buy_fvg_top[index] && row_close >= buy_fvg_bottom[index]) {
        values[kFvgInside] = 1.0;
      }
    }
    for (int index = 0; index < sell_fvg_count; ++index) {
      const double midpoint = (sell_fvg_top[index] + sell_fvg_bottom[index]) / 2.0;
      const double signed_distance = (midpoint - row_close) / atr_for_magnets;
      const double distance = abs_v3(signed_distance);
      if (distance < best_distance_atr) {
        best_distance_atr = distance;
        best_signed = clamp_v3(signed_distance, -10.0, 10.0);
        best_age = neoethos_smc_log1p_cpu_exact_v1(
                       row - static_cast<std::size_t>(sell_fvg_born[index])) /
                   9.21;
      }
      if (values[kFvgInside] == 0.0 && row_close <= sell_fvg_top[index] &&
          row_close >= sell_fvg_bottom[index]) {
        values[kFvgInside] = -1.0;
      }
    }
    if (isfinite(best_distance_atr)) {
      values[kFvgMagnetDist] = best_signed;
      values[kFvgMagnetAge] = clamp_v3(best_age, 0.0, 1.0);
    }
    values[kFvgOpenCount] =
        static_cast<double>(buy_fvg_count + sell_fvg_count) / 20.0;

    if (values[kFvg] != 0.0 && row >= 2) {
      const double gap_size = values[kFvg] > 0.0
                                  ? row_low - high[row - 2]
                                  : low[row - 2] - row_high;
      values[kFvgStrength] = abs_v3(gap_size) / max_v3(running_atr, 1.0e-10);
    }
    if (swing_high_count > 0) {
      const double recent_high = swing_highs[swing_high_count - 1];
      if (row_close > recent_high && recent_high > last_confirmed_high) {
        values[kBos] = 1.0;
        last_confirmed_high = recent_high;
      }
    }
    if (swing_low_count > 0) {
      const double recent_low = swing_lows[swing_low_count - 1];
      if (row_close < recent_low && recent_low < last_confirmed_low) {
        values[kBos] = -1.0;
        last_confirmed_low = recent_low;
      }
    }
    if (swing_high_count >= 2) {
      const double first = swing_highs[swing_high_count - 1];
      const double second = swing_highs[swing_high_count - 2];
      if (first > 0.0 && abs_v3(first - second) / first < 0.0005) {
        values[kEqh] = 1.0;
      }
    }
    if (swing_low_count >= 2) {
      const double first = swing_lows[swing_low_count - 1];
      const double second = swing_lows[swing_low_count - 2];
      if (first > 0.0 && abs_v3(first - second) / first < 0.0005) {
        values[kEql] = 1.0;
      }
    }
    if (row >= 3) {
      if (swing_high_count >= 2 && row_high > swing_highs[swing_high_count - 1] &&
          values[kDisplacement] == 0.0) {
        values[kInducement] = 1.0;
      }
      if (swing_low_count >= 2 && row_low < swing_lows[swing_low_count - 1] &&
          values[kDisplacement] == 0.0) {
        values[kInducement] = -1.0;
      }
    }
    if (values[kOb] != 0.0 && row >= 1) {
      const double previous_range = high[row - 1] - low[row - 1];
      const double previous_body = abs_v3(close[row - 1] - open[row - 1]);
      values[kObStrength] = previous_range > 1.0e-10 ? previous_body / previous_range : 0.0;
    }
    const double total_range = row_high - row_low;
    if (total_range > 1.0e-10) {
      const double body = abs_v3(row_close - row_open);
      const double wick_ratio = 1.0 - body / total_range;
      if (wick_ratio > 0.6) {
        const double upper_wick = row_high - max_v3(row_close, row_open);
        const double lower_wick = min_v3(row_close, row_open) - row_low;
        values[kRejectionBlock] = upper_wick > lower_wick ? -1.0 : 1.0;
      }
    }
    if (row >= 40) {
      values[kDealingRangeWidth] =
          row_close > 1.0e-10 ? (dealing_high - dealing_low) / row_close : 0.0;
    }
    if (swing_high_count > 0 && swing_low_count > 0) {
      values[kSwingRangePct] = row_close > 1.0e-10
                                   ? abs_v3(swing_highs[swing_high_count - 1] -
                                            swing_lows[swing_low_count - 1]) /
                                         row_close
                                   : 0.0;
    }
    const double bar_range = (row_high - row_low) / max_v3(running_atr, 1.0e-10);
    if (bar_range < 0.5) {
      ++consolidation_count;
    } else {
      if (consolidation_count >= 4 && values[kDisplacement] != 0.0) {
        values[kPropulsionBlock] = values[kDisplacement];
      }
      consolidation_count = 0;
    }
    if (row >= 50) {
      double fast_sum = 0.0;
      double slow_sum = 0.0;
      for (std::size_t index = row - 8; index <= row; ++index) {
        fast_sum = fast_sum + close[index];
      }
      for (std::size_t index = row - 50; index <= row; ++index) {
        slow_sum = slow_sum + close[index];
      }
      const double fast_average = fast_sum / 9.0;
      const double slow_average = slow_sum / 51.0;
      values[kTrendBias] = (fast_average - slow_average) / max_v3(running_atr, 1.0e-10);
    }
    if (row >= 5) {
      bool has_breaker = values[kBreakerBlock] != 0.0;
      bool has_fvg = values[kFvg] != 0.0;
      bool has_ob = values[kOb] != 0.0;
      for (std::size_t index = row - 5; index < row; ++index) {
        has_breaker = has_breaker || history_breaker[index % kEventHistoryV3] != 0.0;
        has_fvg = has_fvg || history_fvg[index % kEventHistoryV3] != 0.0;
        has_ob = has_ob || history_ob[index % kEventHistoryV3] != 0.0;
      }
      if (has_breaker && has_fvg && has_ob) {
        values[kUnicornModel] = row_close > row_open ? 1.0 : -1.0;
      }
    }
    if (row >= 10) {
      std::size_t bars_since_event = 0;
      if (values[kLiquiditySweep] == 0.0 && values[kMss] == 0.0 &&
          values[kDisplacement] == 0.0) {
        for (std::size_t index = row; index-- > 0;) {
          ++bars_since_event;
          if (history_liquidity[index % kEventHistoryV3] != 0.0 ||
              history_mss[index % kEventHistoryV3] != 0.0 ||
              history_displacement[index % kEventHistoryV3] != 0.0 ||
              bars_since_event > 60) {
            break;
          }
        }
      }
      constexpr int fib_times[5] = {8, 13, 21, 34, 55};
      for (int index = 0; index < 5; ++index) {
        if (bars_since_event == static_cast<std::size_t>(fib_times[index])) {
          values[kFibTimeRatio] = 1.0;
          break;
        }
        const std::int64_t distance =
            static_cast<std::int64_t>(bars_since_event) - fib_times[index];
        if ((distance < 0 ? -distance : distance) <= 1) {
          values[kFibTimeRatio] = 0.5;
        }
      }
    }

    if (hour == 0 && minute == 0) {
      asian_high = -kPositiveInfinityV3;
      asian_low = kPositiveInfinityV3;
      asian_range_set = false;
    }

    mark_warmup_v3(validity, kOb, row, 20);
    mark_warmup_v3(validity, kFvg, row, 2);
    mark_warmup_v3(validity, kIfvg, row, 2);
    mark_warmup_v3(validity, kLiquiditySweep, row, 10);
    mark_warmup_v3(validity, kBreakerBlock, row, 20);
    mark_warmup_v3(validity, kMitigationBlock, row, 20);
    mark_warmup_v3(validity, kMss, row, 20);
    mark_warmup_v3(validity, kVolumeImbalance, row, 1);
    mark_warmup_v3(validity, kBos, row, 10);
    mark_warmup_v3(validity, kEqh, row, 10);
    mark_warmup_v3(validity, kEql, row, 10);
    mark_warmup_v3(validity, kInducement, row, 10);
    mark_warmup_v3(validity, kUnicornModel, row, 5);
    mark_warmup_v3(validity, kFibTimeRatio, row, 10);
    mark_warmup_v3(validity, kFvgInside, row, 2);
    mark_warmup_v3(validity, kDisplacement, row, 20);
    mark_warmup_v3(validity, kPdArray, row, 40);
    mark_warmup_v3(validity, kDealingRangeWidth, row, 40);
    for (int column = kFib236; column <= kFib2618; ++column) {
      mark_warmup_v3(validity, column, row, 40);
    }
    mark_warmup_v3(validity, kFvgStrength, row, 2);
    mark_warmup_v3(validity, kTrendBias, row, 50);
    mark_warmup_v3(validity, kPropulsionBlock, row, 20);

    if (row >= 20) {
      double average_body = 0.0;
      for (std::size_t index = row - 20; index < row; ++index) {
        average_body = average_body + abs_v3(close[index] - open[index]);
      }
      average_body = average_body / 20.0;
      if (average_body <= kEpsilonV3) {
        validity[kDisplacement] = kZeroDenominator;
      }
    }
    if (row >= 40 && dealing_high - dealing_low <= kEpsilonV3) {
      validity[kPdArray] = kZeroDenominator;
      for (int column = kFib236; column <= kFib2618; ++column) {
        validity[column] = kZeroDenominator;
      }
    }
    if (row >= 2 && values[kFvg] != 0.0 && validity_running_atr <= kEpsilonV3) {
      validity[kFvgStrength] = kZeroDenominator;
    }
    if (total_range <= kEpsilonV3) {
      validity[kRejectionBlock] = kZeroDenominator;
    }
    if (row >= 50 && validity_running_atr <= kEpsilonV3) {
      validity[kTrendBias] = kZeroDenominator;
    }
    if (row >= 20 && validity_running_atr <= kEpsilonV3) {
      validity[kPropulsionBlock] = kZeroDenominator;
    }
    validity[kObStrength] = kAlignmentMissing;
    if (row >= 1 && values[kOb] != 0.0) {
      validity[kObStrength] = high[row - 1] - low[row - 1] > kEpsilonV3
                                  ? kValid
                                  : kZeroDenominator;
    }
    validity[kSwingRangePct] = kAlignmentMissing;
    if (row < 10) {
      validity[kSwingRangePct] = kWarmup;
    } else if (swing_high_count > 0 && swing_low_count > 0) {
      validity[kSwingRangePct] = kValid;
    }
    validity[kFvgMagnetDist] = kAlignmentMissing;
    validity[kFvgMagnetAge] = kAlignmentMissing;
    if (values[kFvgOpenCount] > 0.0) {
      validity[kFvgMagnetDist] = validity_running_atr > kEpsilonV3
                                     ? kValid
                                     : kZeroDenominator;
      validity[kFvgMagnetAge] = validity[kFvgMagnetDist];
    }
    validity[kAsianRange] = !asian_set_before_reset
                                ? kWarmup
                                : (asian_range_before_reset > kEpsilonV3
                                       ? kValid
                                       : kZeroDenominator);
    if (row == 0) {
      validity[kJudasSwing] = kWarmup;
      validity[kNwog] = kWarmup;
      validity[kNdog] = kWarmup;
    } else {
      if (day_changed && validity_running_atr <= kEpsilonV3) {
        validity[kNdog] = kZeroDenominator;
      }
      if (week_changed && validity_running_atr <= kEpsilonV3) {
        validity[kNwog] = kZeroDenominator;
      }
    }

    const signed char canonical_parent_trend =
        row >= 12
            ? (row_close > close[row - 12]
                   ? 1
                   : (row_close < close[row - 12] ? -1 : 0))
            : (row > 0
                   ? (row_close > close[row - 1]
                          ? 1
                          : (row_close < close[row - 1] ? -1 : 0))
                   : 0);
    const signed char canonical_parent_choch =
        row >= 1 && canonical_parent_trend != 0 &&
                canonical_parent_trend_previous != 0 &&
                canonical_parent_trend != canonical_parent_trend_previous
            ? canonical_parent_trend
            : 0;
    canonical_parent_trend_previous = canonical_parent_trend;
    const signed char canonical_parent_premium =
        row_close <= (row_high + row_low) * 0.5 ? 1 : -1;

    signed char parent[kSmcParentSlotsV3];
    parent[0] = quantize_direction_v3(values[kOb], validity[kOb]);
    if (parent[0] == 0) {
      parent[0] = quantize_direction_v3(values[kBos], validity[kBos]);
    }
    parent[1] = quantize_direction_v3(values[kFvg], validity[kFvg]);
    parent[2] = quantize_direction_v3(values[kLiquiditySweep], validity[kLiquiditySweep]);
    if (quantize_binary_v3(values[kEqh], validity[kEqh]) != 0) {
      parent[2] = -1;
    }
    if (quantize_binary_v3(values[kEql], validity[kEql]) != 0) {
      parent[2] = 1;
    }
    parent[3] = quantize_direction_v3(values[kTrendBias], validity[kTrendBias]);
    if (parent[3] == 0) {
      parent[3] = quantize_direction_v3(values[kBos], validity[kBos]);
    }
    if (parent[3] == 0) {
      parent[3] = quantize_direction_v3(values[kDisplacement], validity[kDisplacement]);
    }
    parent[4] = canonical_parent_premium;
    parent[5] = quantize_binary_v3(values[kInducement], validity[kInducement]);
    parent[6] = quantize_direction_v3(values[kBos], validity[kBos]);
    parent[7] = canonical_parent_choch;
    parent[8] = quantize_direction_v3(values[kEqh], validity[kEqh]);
    if (parent[8] == 0 && quantize_binary_v3(values[kEqh], validity[kEqh]) != 0) {
      parent[8] = -1;
    }
    parent[9] = quantize_direction_v3(values[kEql], validity[kEql]);
    if (parent[9] == 0 && quantize_binary_v3(values[kEql], validity[kEql]) != 0) {
      parent[9] = 1;
    }
    parent[10] = quantize_direction_v3(values[kDisplacement], validity[kDisplacement]);
    if (parent[10] != 0) {
      parent[5] = 1;
    }

    for (int column = 0; column < kSmcFeatureColumnsV3; ++column) {
      if (validity[column] == kValid && !isfinite(values[column])) {
        *device_error = 5U;
        return;
      }
      const std::size_t cell = static_cast<std::size_t>(column) * rows + row;
      smc_feature_validity_u8[cell] = validity[column];
      smc_feature_values[cell] = validity[column] == kValid ? values[column] : canonical_nan_v3();
    }
    for (int slot = 0; slot < kSmcParentSlotsV3; ++slot) {
      smc_parent_rows[row * kSmcParentSlotsV3 + static_cast<std::size_t>(slot)] = parent[slot];
    }

    history_ob[row % kEventHistoryV3] = values[kOb];
    history_fvg[row % kEventHistoryV3] = values[kFvg];
    history_liquidity[row % kEventHistoryV3] = values[kLiquiditySweep];
    history_mss[row % kEventHistoryV3] = values[kMss];
    history_displacement[row % kEventHistoryV3] = values[kDisplacement];
    history_breaker[row % kEventHistoryV3] = values[kBreakerBlock];
  }

  sha256_array_v3(months, rows, generated_parent_hashes);
  sha256_array_v3(days, rows, generated_parent_hashes + 32U);
  sha256_array_v3(smc_parent_rows, rows * kSmcParentSlotsV3,
                  generated_parent_hashes + 64U);
}

}  // namespace

extern "C" int neoethos_resident_smc_parent_features_f64_v3(
    const double* open, const double* high, const double* low,
    const double* close, const std::int64_t* timestamps, std::size_t rows,
    double* smc_feature_values, unsigned char* smc_feature_validity_u8,
    std::int64_t* months, std::int64_t* days, signed char* smc_parent_rows,
    unsigned char* generated_parent_hashes, unsigned int* device_error,
    cudaStream_t stream) {
  if (open == nullptr || high == nullptr || low == nullptr || close == nullptr ||
      timestamps == nullptr || smc_feature_values == nullptr ||
      smc_feature_validity_u8 == nullptr || months == nullptr || days == nullptr ||
      smc_parent_rows == nullptr || generated_parent_hashes == nullptr ||
      device_error == nullptr || stream == nullptr || rows == 0 ||
      rows >= (1ULL << 53U) ||
      rows > std::numeric_limits<std::size_t>::max() /
                 static_cast<std::size_t>(kSmcFeatureColumnsV3) ||
      rows > std::numeric_limits<std::size_t>::max() /
                 static_cast<std::size_t>(kSmcParentSlotsV3)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  resident_smc_parent_features_f64_v3<<<1, 1, 0, stream>>>(
      open, high, low, close, timestamps, rows, smc_feature_values,
      smc_feature_validity_u8, months, days, smc_parent_rows,
      generated_parent_hashes, device_error);
  return static_cast<int>(cudaGetLastError());
}
