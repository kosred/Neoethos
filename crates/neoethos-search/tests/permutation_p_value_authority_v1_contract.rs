use std::fs;
use std::path::PathBuf;

const PRIMARY_SOURCE_V1: &str = "https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf";
const PERMUTATION_COUNT_V1: usize = 50;
const REJECTION_THRESHOLD_V1: f64 = 0.05;

fn discovery_source() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/discovery.rs");
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn primary_monte_carlo_p_value_v1(beats: usize, permutations: usize) -> Option<f64> {
    if permutations == 0 || beats > permutations {
        return None;
    }
    let numerator = beats.checked_add(1)?;
    let denominator = permutations.checked_add(1)?;
    Some(numerator as f64 / denominator as f64)
}

#[test]
fn primary_source_identity_and_plus_one_estimator_are_pinned() {
    assert!(PRIMARY_SOURCE_V1.ends_with("PermPValuesPreprint.pdf"));
    assert_eq!(primary_monte_carlo_p_value_v1(0, 50), Some(1.0 / 51.0));
    assert_eq!(primary_monte_carlo_p_value_v1(50, 50), Some(1.0));
    assert_eq!(primary_monte_carlo_p_value_v1(51, 50), None);
    assert_eq!(primary_monte_carlo_p_value_v1(0, 0), None);
}

#[test]
fn b_zero_one_two_fixture_pins_the_m50_decision_boundary() {
    let p0 = primary_monte_carlo_p_value_v1(0, PERMUTATION_COUNT_V1).unwrap();
    let p1 = primary_monte_carlo_p_value_v1(1, PERMUTATION_COUNT_V1).unwrap();
    let p2 = primary_monte_carlo_p_value_v1(2, PERMUTATION_COUNT_V1).unwrap();

    assert!((p0 - 1.0 / 51.0).abs() <= f64::EPSILON);
    assert!((p1 - 2.0 / 51.0).abs() <= f64::EPSILON);
    assert!((p2 - 3.0 / 51.0).abs() <= f64::EPSILON);
    assert!(p0 < REJECTION_THRESHOLD_V1);
    assert!(p1 < REJECTION_THRESHOLD_V1);
    assert!(p2 >= REJECTION_THRESHOLD_V1);
}

#[test]
fn production_uses_the_versioned_primary_estimator_without_the_zero_p_value_formula() {
    let source = discovery_source();
    for required in [
        "PERMUTATION_MONTE_CARLO_P_VALUE_SEMANTICS_V1",
        "neoethos.permutation-monte-carlo-p-value.v1",
        PRIMARY_SOURCE_V1,
        "fn permutation_monte_carlo_p_value_v1(",
        "if permutations == 0 || beats > permutations",
        "let numerator = beats.checked_add(1)",
        "let denominator = permutations.checked_add(1)",
        "permutation_monte_carlo_p_value_v1(beats, N_PERM)",
    ] {
        assert!(
            source.contains(required),
            "production permutation authority is missing {required:?}"
        );
    }
    assert!(
        !source.contains("let p_value = beats as f64 / N_PERM as f64;"),
        "the biased b/m estimator can return the impossible p=0 and must be retired"
    );
}
