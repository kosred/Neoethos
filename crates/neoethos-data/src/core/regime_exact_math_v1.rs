//! Exact positive logarithm authority for Regime semantic-v3.
//!
//! The CUDA resident producer carries a source-hash-sealed transcription of
//! the marked function body. The operation-token identity is deliberately
//! independent from libc and libdevice versions.

pub(crate) const REGIME_LOG49_OPERATION_TOKENS_V1: &str = "neoethos.regime.log49-mirror.v1|subnormal-scale=0x4350000000000000|mantissa-mask=0x000fffffffffffff|one-exponent=0x3ff0000000000000|ln2=0x3fe62e42fefa39ef|series-odd=3..49|order=normalize,bits,exponent,mantissa,z,z2,term,sum,loop,return|rounding=rn-no-fma";
pub(crate) const REGIME_LOG49_OPERATION_TOKENS_SHA256_V1: &str =
    "73002b6761d1ca425250a761fa4411cf3ae0d26c862caa964e93063c69c32080";
pub(crate) const REGIME_LOG49_RUST_MIRROR_SHA256_V1: &str =
    "f7d83af4d95a95c38cb360abcee96a223f4010aba2e3c679145c091e56db8fea";

const SUBNORMAL_SCALE_BITS_V1: u64 = 0x4350_0000_0000_0000;
const LN_2_BITS_V1: u64 = 0x3fe6_2e42_fefa_39ef;
const MANTISSA_MASK_V1: u64 = 0x000f_ffff_ffff_ffff;
const ONE_EXPONENT_BITS_V1: u64 = 0x3ff0_0000_0000_0000;

// REGIME_LOG49_RUST_MIRROR_BEGIN_V1
/// Deterministic positive natural logarithm for CPU/GPU exact-bit parity.
#[inline]
pub(crate) fn neoethos_ln_positive_exact_v1(value: f64) -> f64 {
    debug_assert!(value.is_finite() && value > 0.0);
    let (normalized, exponent_adjustment) = if value.is_subnormal() {
        (value * f64::from_bits(SUBNORMAL_SCALE_BITS_V1), -54)
    } else {
        (value, 0)
    };
    let bits = normalized.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023 + exponent_adjustment;
    let mantissa = f64::from_bits((bits & MANTISSA_MASK_V1) | ONE_EXPONENT_BITS_V1);
    let z = (mantissa - 1.0) / (mantissa + 1.0);
    let z_squared = z * z;
    let mut term = z;
    let mut sum = z;
    let mut denominator = 3_u32;
    while denominator <= 49 {
        term = term * z_squared;
        sum = sum + term / denominator as f64;
        denominator += 2;
    }
    exponent as f64 * f64::from_bits(LN_2_BITS_V1) + 2.0 * sum
}

#[inline]
pub(crate) fn neoethos_log10_positive_exact_v1(value: f64, ln_10: f64) -> f64 {
    neoethos_ln_positive_exact_v1(value) / ln_10
}
// REGIME_LOG49_RUST_MIRROR_END_V1
