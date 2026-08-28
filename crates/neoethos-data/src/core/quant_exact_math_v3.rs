//! Quant semantic-v3 exact logarithm authority.
//!
//! This covers the positive finite binary64 domain of the Sun fdlibm/OpenLibm
//! `e_log` schedule. Every arithmetic edge is named and kept in source order
//! so the CUDA transcription can use the corresponding explicit RN intrinsic.
//! The approximation is bounded-faithful to the real logarithm; exactness here
//! means identical CPU/CUDA bits for this frozen operation graph, not universal
//! correct rounding of the mathematical function.

pub const QUANT_OPENLIBM_COMMIT_V3: &str = "82e90aef0657289192efe77be89791c07dea0775";
pub const QUANT_OPENLIBM_E_LOG_SOURCE_SHA256_V3: &str =
    "8996B789A4CBBCEF7CF7D568C1BE558CE9110900A40CA6C46FB4ED46C343CAFD";
pub const QUANT_OPENLIBM_E_LOG_SOURCE_V3: &str = "vendor/vector-ta-0.2.9-patched/tests/fixtures/openlibm/e_log-82e90aef0657289192efe77be89791c07dea0775.c";
pub const QUANT_OPENLIBM_E_LOG_RECEIPT_V3: &str = "vendor/vector-ta-0.2.9-patched/tests/fixtures/openlibm/e_log-82e90aef0657289192efe77be89791c07dea0775.receipt.txt";
pub const QUANT_LOG_OPERATION_SCHEDULE_V3: &str = "neoethos.quant.log.semantic-v3;sun-fdlibm-openlibm-e_log;positive-finite-binary64;commit=82e90aef0657289192efe77be89791c07dea0775;source-sha256=8996B789A4CBBCEF7CF7D568C1BE558CE9110900A40CA6C46FB4ED46C343CAFD;rounding=rn-no-fma;cpu-cuda-bit-tolerance=zero;real-log-accuracy=bounded-faithful-max-1ulp-reviewed-wide-domain";

const TWO54_V3: f64 = 1.801_439_850_948_198_400_00e16;
const LN2_HI_V3: f64 = 6.931_471_803_691_238_164_90e-1;
const LN2_LO_V3: f64 = 1.908_214_929_270_587_700_02e-10;
const LG1_V3: f64 = 6.666_666_666_666_735_130e-1;
const LG2_V3: f64 = 3.999_999_999_940_941_908e-1;
const LG3_V3: f64 = 2.857_142_874_366_239_149e-1;
const LG4_V3: f64 = 2.222_219_843_214_978_396e-1;
const LG5_V3: f64 = 1.818_357_216_161_805_012e-1;
const LG6_V3: f64 = 1.531_383_769_920_937_332e-1;
const LG7_V3: f64 = 1.479_819_860_511_658_591e-1;

#[inline(always)]
fn add_rn_v3(left: f64, right: f64) -> f64 {
    left + right
}

#[inline(always)]
fn sub_rn_v3(left: f64, right: f64) -> f64 {
    left - right
}

#[inline(always)]
fn mul_rn_v3(left: f64, right: f64) -> f64 {
    left * right
}

#[inline(always)]
fn div_rn_v3(left: f64, right: f64) -> f64 {
    left / right
}

#[inline(always)]
fn with_high_word_v3(value: f64, high: u32) -> f64 {
    f64::from_bits((value.to_bits() & 0x0000_0000_ffff_ffff) | (u64::from(high) << 32))
}

/// Literal Sun fdlibm/OpenLibm e_log schedule for positive finite binary64.
#[inline]
pub fn quant_log_positive_f64_v3(mut value: f64) -> Option<f64> {
    if !value.is_finite() || value <= 0.0 {
        return None;
    }

    let mut high = (value.to_bits() >> 32) as i32;
    let low = value.to_bits() as u32;
    let mut exponent = 0_i32;
    if high < 0x0010_0000 {
        if (((high as u32) & 0x7fff_ffff) | low) == 0 {
            return None;
        }
        exponent -= 54;
        value = mul_rn_v3(value, TWO54_V3);
        high = (value.to_bits() >> 32) as i32;
    }
    if high >= 0x7ff0_0000 {
        return None;
    }

    exponent += (high >> 20) - 1023;
    high &= 0x000f_ffff;
    let normalize = (high + 0x0009_5f64) & 0x0010_0000;
    value = with_high_word_v3(value, (high | (normalize ^ 0x3ff0_0000)) as u32);
    exponent += normalize >> 20;

    let fraction = sub_rn_v3(value, 1.0);
    if (0x000f_ffff & (2 + high)) < 3 {
        if fraction == 0.0 {
            if exponent == 0 {
                return Some(0.0);
            }
            let exponent_f64 = f64::from(exponent);
            return Some(add_rn_v3(
                mul_rn_v3(exponent_f64, LN2_HI_V3),
                mul_rn_v3(exponent_f64, LN2_LO_V3),
            ));
        }
        let square = mul_rn_v3(fraction, fraction);
        let inner = sub_rn_v3(0.5, mul_rn_v3(0.333_333_333_333_333_33, fraction));
        let remainder = mul_rn_v3(square, inner);
        if exponent == 0 {
            return Some(sub_rn_v3(fraction, remainder));
        }
        let exponent_f64 = f64::from(exponent);
        let correction = sub_rn_v3(
            sub_rn_v3(remainder, mul_rn_v3(exponent_f64, LN2_LO_V3)),
            fraction,
        );
        return Some(sub_rn_v3(mul_rn_v3(exponent_f64, LN2_HI_V3), correction));
    }

    let scaled = div_rn_v3(fraction, add_rn_v3(2.0, fraction));
    let exponent_f64 = f64::from(exponent);
    let square = mul_rn_v3(scaled, scaled);
    let selector = (high - 0x0006_147a) | (0x0006_b851 - high);
    let fourth = mul_rn_v3(square, square);
    let even_inner = add_rn_v3(LG4_V3, mul_rn_v3(fourth, LG6_V3));
    let even = mul_rn_v3(fourth, add_rn_v3(LG2_V3, mul_rn_v3(fourth, even_inner)));
    let odd_inner = add_rn_v3(LG5_V3, mul_rn_v3(fourth, LG7_V3));
    let odd_middle = add_rn_v3(LG3_V3, mul_rn_v3(fourth, odd_inner));
    let odd = mul_rn_v3(square, add_rn_v3(LG1_V3, mul_rn_v3(fourth, odd_middle)));
    let remainder = add_rn_v3(odd, even);

    let result = if selector > 0 {
        let half_square = mul_rn_v3(mul_rn_v3(0.5, fraction), fraction);
        let scaled_sum = mul_rn_v3(scaled, add_rn_v3(half_square, remainder));
        if exponent == 0 {
            sub_rn_v3(fraction, sub_rn_v3(half_square, scaled_sum))
        } else {
            let low_term = mul_rn_v3(exponent_f64, LN2_LO_V3);
            let correction = sub_rn_v3(
                sub_rn_v3(half_square, add_rn_v3(scaled_sum, low_term)),
                fraction,
            );
            sub_rn_v3(mul_rn_v3(exponent_f64, LN2_HI_V3), correction)
        }
    } else {
        let scaled_remainder = mul_rn_v3(scaled, sub_rn_v3(fraction, remainder));
        if exponent == 0 {
            sub_rn_v3(fraction, scaled_remainder)
        } else {
            let correction = sub_rn_v3(
                sub_rn_v3(scaled_remainder, mul_rn_v3(exponent_f64, LN2_LO_V3)),
                fraction,
            );
            sub_rn_v3(mul_rn_v3(exponent_f64, LN2_HI_V3), correction)
        }
    };
    result.is_finite().then_some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ordered_bits(value: f64) -> u64 {
        let bits = value.to_bits();
        if bits >> 63 == 0 {
            bits | (1_u64 << 63)
        } else {
            !bits
        }
    }

    #[test]
    fn frozen_sun_checkpoints_are_exact() {
        for (input_bits, expected_bits) in [
            (0x3ff0_0000_0000_0000, 0x0000_0000_0000_0000),
            (0x0000_0000_0000_0001, 0xc087_4385_446d_71c3),
            (0x4000_0000_0000_0000, 0x3fe6_2e42_fefa_39ef),
            (0x3fe0_0000_0000_0000, 0xbfe6_2e42_fefa_39ef),
            (0x3f90_d8b0_1d6a_1591, 0xc010_6de8_9959_7cd8),
            (0x3fa8_6023_0080_6d1d, 0xc008_5ba3_1b96_26ee),
            (0x3fb8_4482_7417_a07c, 0xc002_d928_b548_dbe4),
            (0x3fc7_6c46_f3ca_9d14, 0xbffb_2c4a_e8c3_fca8),
        ] {
            let actual = quant_log_positive_f64_v3(f64::from_bits(input_bits))
                .expect("positive finite checkpoint is in domain");
            assert_eq!(actual.to_bits(), expected_bits, "input=0x{input_bits:016x}");
        }
    }

    #[test]
    fn wide_domain_matches_high_precision_checkpoints_within_one_ulp() {
        // Expected cells were rounded from 250-decimal-digit arbitrary-
        // precision values. This audits accuracy only; CPU/CUDA must match exactly.
        for (input_bits, correctly_rounded_bits) in [
            (0x0000_0000_0000_0001, 0xc087_4385_446d_71c3),
            (0x000f_ffff_ffff_ffff, 0xc086_232b_dd7a_bcd2),
            (0x0010_0000_0000_0000, 0xc086_232b_dd7a_bcd2),
            (0x3b10_0000_0000_0000, 0xc04b_0861_a6c0_f69c),
            (0x3fb9_9999_9999_999a, 0xc002_6bb1_bbb5_5515),
            (0x3fe0_0000_0000_0000, 0xbfe6_2e42_fefa_39ef),
            (0x3fef_ffff_ffff_ffff, 0xbca0_0000_0000_0000),
            (0x3ff0_0000_0000_0000, 0x0000_0000_0000_0000),
            (0x3ff0_0000_0000_0001, 0x3caf_ffff_ffff_ffff),
            (0x3ff6_a09e_667f_3bcd, 0x3fd6_2e42_fefa_39f0),
            (0x4000_0000_0000_0000, 0x3fe6_2e42_fefa_39ef),
            (0x4005_bf0a_8b14_5769, 0x3ff0_0000_0000_0000),
            (0x4024_0000_0000_0000, 0x4002_6bb1_bbb5_5516),
            (0x4350_0000_0000_0000, 0x4042_b708_8723_20e2),
            (0x5fef_ffff_ffff_ffff, 0x4076_2e42_fefa_39ef),
            (0x7fef_ffff_ffff_ffff, 0x4086_2e42_fefa_39ef),
        ] {
            let actual = quant_log_positive_f64_v3(f64::from_bits(input_bits))
                .expect("positive finite checkpoint is in domain");
            let ulp =
                ordered_bits(actual).abs_diff(ordered_bits(f64::from_bits(correctly_rounded_bits)));
            assert!(ulp <= 1, "input=0x{input_bits:016x} ulp={ulp}");
        }
    }

    #[test]
    fn nonpositive_and_nonfinite_values_fail_closed() {
        for value in [0.0, -0.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert!(quant_log_positive_f64_v3(value).is_none());
        }
    }

    #[test]
    fn openlibm_authority_metadata_is_frozen() {
        assert_eq!(
            QUANT_OPENLIBM_COMMIT_V3,
            "82e90aef0657289192efe77be89791c07dea0775"
        );
        assert_eq!(
            QUANT_OPENLIBM_E_LOG_SOURCE_SHA256_V3,
            "8996B789A4CBBCEF7CF7D568C1BE558CE9110900A40CA6C46FB4ED46C343CAFD"
        );
        assert!(QUANT_OPENLIBM_E_LOG_SOURCE_V3.ends_with(".c"));
        assert!(QUANT_OPENLIBM_E_LOG_RECEIPT_V3.ends_with(".receipt.txt"));
        assert!(
            QUANT_LOG_OPERATION_SCHEDULE_V3.contains("cpu-cuda-bit-tolerance=zero")
                && QUANT_LOG_OPERATION_SCHEDULE_V3.contains("rounding=rn-no-fma")
        );
    }
}
