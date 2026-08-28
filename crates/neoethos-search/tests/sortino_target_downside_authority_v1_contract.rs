use std::fs;
use std::path::PathBuf;

const PRIMARY_SOURCE_V1: &str =
    "https://www.cmegroup.com/education/files/rr-sortino-a-sharper-ratio.pdf";
const TARGET_RETURN_V1: f64 = 0.0;

fn quality_source() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/quality.rs");
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn official_target_downside_sortino_v1(
    returns: &[f64],
    annualization: f64,
) -> Result<f64, &'static str> {
    if returns.len() < 2 {
        return Ok(0.0);
    }
    if !annualization.is_finite()
        || annualization <= 0.0
        || returns.iter().any(|value| !value.is_finite())
    {
        return Err("non-finite Sortino input");
    }

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let target_downside_variance = returns
        .iter()
        .map(|value| (value - TARGET_RETURN_V1).min(0.0).powi(2))
        .sum::<f64>()
        / returns.len() as f64;
    if !mean_return.is_finite() || !target_downside_variance.is_finite() {
        return Err("non-finite Sortino intermediate");
    }
    let target_downside_deviation = target_downside_variance.sqrt();
    if target_downside_deviation <= 1.0e-9 {
        return Ok(0.0);
    }
    let ratio = ((mean_return - TARGET_RETURN_V1) / target_downside_deviation) * annualization;
    ratio
        .is_finite()
        .then_some(ratio)
        .ok_or("non-finite Sortino result")
}

#[test]
fn primary_source_identity_and_target_downside_formula_are_pinned() {
    assert!(PRIMARY_SOURCE_V1.ends_with("rr-sortino-a-sharper-ratio.pdf"));
    let actual = official_target_downside_sortino_v1(&[0.20, 0.10, -0.10], 1.0).unwrap();
    let expected = (0.20_f64 / 3.0) / (0.01_f64 / 3.0).sqrt();
    assert!((actual - expected).abs() <= 1.0e-12);
}

#[test]
fn one_downside_observation_is_defined_and_all_observations_remain_in_n() {
    let actual = official_target_downside_sortino_v1(&[0.20, -0.10], 1.0).unwrap();
    let expected = 0.05_f64 / (0.01_f64 / 2.0).sqrt();
    assert!((actual - expected).abs() <= 1.0e-12);

    let with_two_non_downside_observations =
        official_target_downside_sortino_v1(&[-0.10, 0.0, 0.20, 0.30], 1.0).unwrap();
    assert!((with_two_non_downside_observations - 2.0).abs() <= 1.0e-12);
}

#[test]
fn nonfinite_inputs_fail_closed_instead_of_becoming_a_plausible_ratio() {
    assert!(official_target_downside_sortino_v1(&[0.1, f64::NAN], 1.0).is_err());
    assert!(official_target_downside_sortino_v1(&[0.1, -0.1], f64::NAN).is_err());
}

#[test]
fn production_uses_all_n_observations_and_the_exact_zero_target() {
    let source = quality_source();
    for required in [
        "TARGET_DOWNSIDE_SORTINO_SEMANTICS_V1",
        "neoethos.target-downside-sortino.v1",
        PRIMARY_SOURCE_V1,
        "const SORTINO_TARGET_RETURN_V1: f64 = 0.0;",
        "let shortfall = (period_return - SORTINO_TARGET_RETURN_V1).min(0.0);",
        "downside_sum_squares / returns.len() as f64",
        "(mean_return - SORTINO_TARGET_RETURN_V1) / target_downside_deviation",
    ] {
        assert!(
            source.contains(required),
            "production target-downside authority is missing {required:?}"
        );
    }
    assert!(
        !source.contains(
            "let downside: Vec<f64> = returns.iter().cloned().filter(|v| *v < 0.0).collect();"
        ),
        "dropping non-downside observations changes N and is not target downside deviation"
    );
    assert!(
        !source.contains("stddev_sample(&downside, 0.0)"),
        "sample SD of the negative subset is not target downside deviation"
    );
}

#[test]
fn production_nonfinite_sortino_is_an_explicit_existing_gate_rejection() {
    let source = quality_source();
    for required in [
        "const INVALID_TARGET_DOWNSIDE_SORTINO_V1: f64 = f64::NEG_INFINITY;",
        "if !sortino.is_finite()",
        "let mut invalid = empty_metrics(strategy_id);",
        "invalid.sortino_ratio = INVALID_TARGET_DOWNSIDE_SORTINO_V1;",
        "invalid.quality_score = f64::NEG_INFINITY;",
        "invalid.recommendation = \"INVALID_SORTINO_INPUT\".to_string();",
        "return invalid;",
    ] {
        assert!(
            source.contains(required),
            "non-finite Sortino is not fail-closed through the existing quality boundary: {required:?}"
        );
    }
}
