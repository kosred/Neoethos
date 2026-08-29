#[path = "../src/core/smc_log1p_exact_v1.rs"]
mod authority;

use authority::smc_log1p_exact_v1;

#[test]
fn fixed_order_log1p_has_platform_independent_golden_bits() {
    for (age, expected) in [
        (0_u64, 0x0000_0000_0000_0000),
        (1, 0x3fe6_2e42_fefa_39ef),
        (2, 0x3ff1_93ea_7aad_030b),
        (3, 0x3ff6_2e42_fefa_39ef),
        (7, 0x4000_a2b2_3f3b_ab73),
        (63, 0x4010_a2b2_3f3b_ab73),
        (64, 0x4010_b292_9394_2975),
        (4095, 0x4020_a2b2_3f3b_ab73),
        (4096, 0x4020_a2d2_3e3b_b61d),
        (5_270_000, 0x402e_f480_44b6_37b7),
    ] {
        assert_eq!(smc_log1p_exact_v1(age).to_bits(), expected, "age {age}");
    }
}

#[test]
fn fixed_order_log1p_is_monotonic_and_close_to_mathematical_log1p() {
    let mut previous = f64::NEG_INFINITY;
    for age in 0_u64..=100_000 {
        let actual = smc_log1p_exact_v1(age);
        assert!(actual >= previous, "age {age} broke monotonicity");
        assert!(
            (actual - (age as f64).ln_1p()).abs() <= 4.0 * f64::EPSILON * actual.abs().max(1.0),
            "age {age} exceeded the deterministic approximation error bound"
        );
        previous = actual;
    }
}
