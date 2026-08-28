//! Platform-independent logarithm used by the SMC FVG-age feature.
//!
//! The old semantic-v2 lane called the platform `log1p` implementation. That
//! cannot be an exact-bit CPU/CUDA authority because libc and libdevice are
//! independent implementations. Semantic v3 range-reduces the positive
//! integer `1 + age` and evaluates a fixed 25-term atanh series. Every
//! operation and constant below has an identical CUDA transcription.

const LN_2_BITS_V1: u64 = 0x3fe6_2e42_fefa_39ef;
const MANTISSA_MASK: u64 = 0x000f_ffff_ffff_ffff;
const ONE_EXPONENT_BITS: u64 = 0x3ff0_0000_0000_0000;

/// Deterministic `ln(1 + age)` for an exactly represented non-negative age.
///
/// Production row counts are required to fit the exact integer range of f64.
/// The fixed series has `|z| <= 1/3`; its omitted tail is below f64 rounding
/// noise while avoiding every platform math-library call.
#[inline]
pub(crate) fn smc_log1p_exact_v1(age: u64) -> f64 {
    debug_assert!(age < (1_u64 << 53));
    let one_plus_age = age as f64 + 1.0;
    let bits = one_plus_age.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023;
    let mantissa = f64::from_bits((bits & MANTISSA_MASK) | ONE_EXPONENT_BITS);
    let z = (mantissa - 1.0) / (mantissa + 1.0);
    let z_squared = z * z;
    let mut term = z;
    let mut sum = z;
    let mut denominator = 3_u32;
    while denominator <= 49 {
        term *= z_squared;
        sum += term / denominator as f64;
        denominator += 2;
    }
    exponent as f64 * f64::from_bits(LN_2_BITS_V1) + 2.0 * sum
}
