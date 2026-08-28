#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

// One CUDA transcription of the frozen Sun fdlibm/OpenLibm e_log schedule.
// Both resident Quant-v3 and resident adaptive stops include this file; neither
// owns a private copy. Every arithmetic edge is explicit RN and mirrors
// neoethos-data/core/quant_exact_math_v3.rs.
// commit=82e90aef0657289192efe77be89791c07dea0775
// source-sha256=8996B789A4CBBCEF7CF7D568C1BE558CE9110900A40CA6C46FB4ED46C343CAFD
namespace neoethos_exact_math_v3 {

__device__ __forceinline__ double add_rn_v3(double left, double right) {
  return __dadd_rn(left, right);
}

__device__ __forceinline__ double sub_rn_v3(double left, double right) {
  return __dsub_rn(left, right);
}

__device__ __forceinline__ double mul_rn_v3(double left, double right) {
  return __dmul_rn(left, right);
}

__device__ __forceinline__ double div_rn_v3(double left, double right) {
  return __ddiv_rn(left, right);
}

__device__ __forceinline__ double with_high_word_v3(double value,
                                                     unsigned int high) {
  unsigned long long bits =
      static_cast<unsigned long long>(__double_as_longlong(value));
  bits = (bits & 0x00000000ffffffffULL) |
         (static_cast<unsigned long long>(high) << 32);
  return __longlong_as_double(static_cast<long long>(bits));
}

__device__ __forceinline__ bool exact_log_positive_f64_v3(double value,
                                                           double* output) {
  const double TWO54 = 1.80143985094819840000e+16;
  const double LN2_HI = 6.93147180369123816490e-01;
  const double LN2_LO = 1.90821492927058770002e-10;
  const double LG1 = 6.666666666666735130e-01;
  const double LG2 = 3.999999999940941908e-01;
  const double LG3 = 2.857142874366239149e-01;
  const double LG4 = 2.222219843214978396e-01;
  const double LG5 = 1.818357216161805012e-01;
  const double LG6 = 1.531383769920937332e-01;
  const double LG7 = 1.479819860511658591e-01;

  if (!isfinite(value) || !(value > 0.0)) return false;
  unsigned long long raw =
      static_cast<unsigned long long>(__double_as_longlong(value));
  int high = static_cast<int>(raw >> 32);
  const unsigned int low = static_cast<unsigned int>(raw);
  int exponent = 0;
  if (high < 0x00100000) {
    if (((static_cast<unsigned int>(high) & 0x7fffffffU) | low) == 0U)
      return false;
    exponent -= 54;
    value = mul_rn_v3(value, TWO54);
    raw = static_cast<unsigned long long>(__double_as_longlong(value));
    high = static_cast<int>(raw >> 32);
  }
  if (high >= 0x7ff00000) return false;

  exponent += (high >> 20) - 1023;
  high &= 0x000fffff;
  const int normalize = (high + 0x00095f64) & 0x00100000;
  value = with_high_word_v3(
      value, static_cast<unsigned int>(high | (normalize ^ 0x3ff00000)));
  exponent += normalize >> 20;

  const double fraction = sub_rn_v3(value, 1.0);
  if ((0x000fffff & (2 + high)) < 3) {
    if (fraction == 0.0) {
      if (exponent == 0) {
        *output = 0.0;
        return true;
      }
      const double exponent_f64 = static_cast<double>(exponent);
      *output = add_rn_v3(mul_rn_v3(exponent_f64, LN2_HI),
                          mul_rn_v3(exponent_f64, LN2_LO));
      return isfinite(*output);
    }
    const double square = mul_rn_v3(fraction, fraction);
    const double inner =
        sub_rn_v3(0.5, mul_rn_v3(0.33333333333333333, fraction));
    const double remainder = mul_rn_v3(square, inner);
    if (exponent == 0) {
      *output = sub_rn_v3(fraction, remainder);
      return isfinite(*output);
    }
    const double exponent_f64 = static_cast<double>(exponent);
    const double correction = sub_rn_v3(
        sub_rn_v3(remainder, mul_rn_v3(exponent_f64, LN2_LO)), fraction);
    *output = sub_rn_v3(mul_rn_v3(exponent_f64, LN2_HI), correction);
    return isfinite(*output);
  }

  const double scaled = div_rn_v3(fraction, add_rn_v3(2.0, fraction));
  const double exponent_f64 = static_cast<double>(exponent);
  const double square = mul_rn_v3(scaled, scaled);
  const int selector = (high - 0x0006147a) | (0x0006b851 - high);
  const double fourth = mul_rn_v3(square, square);
  const double even_inner = add_rn_v3(LG4, mul_rn_v3(fourth, LG6));
  const double even = mul_rn_v3(
      fourth, add_rn_v3(LG2, mul_rn_v3(fourth, even_inner)));
  const double odd_inner = add_rn_v3(LG5, mul_rn_v3(fourth, LG7));
  const double odd_middle = add_rn_v3(LG3, mul_rn_v3(fourth, odd_inner));
  const double odd = mul_rn_v3(
      square, add_rn_v3(LG1, mul_rn_v3(fourth, odd_middle)));
  const double remainder = add_rn_v3(odd, even);

  if (selector > 0) {
    const double half_square = mul_rn_v3(mul_rn_v3(0.5, fraction), fraction);
    const double scaled_sum =
        mul_rn_v3(scaled, add_rn_v3(half_square, remainder));
    if (exponent == 0) {
      *output = sub_rn_v3(fraction, sub_rn_v3(half_square, scaled_sum));
      return isfinite(*output);
    }
    const double low_term = mul_rn_v3(exponent_f64, LN2_LO);
    const double correction = sub_rn_v3(
        sub_rn_v3(half_square, add_rn_v3(scaled_sum, low_term)), fraction);
    *output = sub_rn_v3(mul_rn_v3(exponent_f64, LN2_HI), correction);
    return isfinite(*output);
  }

  const double scaled_remainder =
      mul_rn_v3(scaled, sub_rn_v3(fraction, remainder));
  if (exponent == 0) {
    *output = sub_rn_v3(fraction, scaled_remainder);
    return isfinite(*output);
  }
  const double correction = sub_rn_v3(
      sub_rn_v3(scaled_remainder, mul_rn_v3(exponent_f64, LN2_LO)),
      fraction);
  *output = sub_rn_v3(mul_rn_v3(exponent_f64, LN2_HI), correction);
  return isfinite(*output);
}

}  // namespace neoethos_exact_math_v3
