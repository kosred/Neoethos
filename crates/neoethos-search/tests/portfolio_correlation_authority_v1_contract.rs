use std::fs;
use std::path::PathBuf;

const SCIPY_PEARSONR_AUTHORITY_V1: &str =
    "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html";
const SCIPY_SPEARMANR_AUTHORITY_V1: &str =
    "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html";
const NIST_TIE_CORRECTED_SPEARMAN_AUTHORITY_V1: &str =
    "https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/rankcorr.htm";
const SCIPY_NEAR_CONSTANT_RELATIVE_NORM_V1: f64 = 1.0e-13;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-search"))
}

fn source(path: &str) -> String {
    let path = manifest_dir().join(path);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read required source {}: {error}", path.display()))
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let (_, tail) = source
        .split_once(start)
        .unwrap_or_else(|| panic!("missing source boundary {start:?}"));
    tail.split_once(end)
        .unwrap_or_else(|| panic!("missing source boundary {end:?} after {start:?}"))
        .0
}

fn require_all(source: &str, required: &[&str]) {
    for token in required {
        assert!(source.contains(token), "missing required token {token:?}");
    }
}

fn midranks(values: &[i8]) -> Vec<f64> {
    values
        .iter()
        .map(|value| {
            let before = values.iter().filter(|candidate| *candidate < value).count() as f64;
            let equal = values
                .iter()
                .filter(|candidate| *candidate == value)
                .count() as f64;
            before + (equal + 1.0) / 2.0
        })
        .collect()
}

fn pearson(values_a: &[f64], values_b: &[f64]) -> Option<f64> {
    if values_a.len() != values_b.len() || values_a.len() < 2 {
        return None;
    }
    let n = values_a.len() as f64;
    let mean_a = values_a.iter().sum::<f64>() / n;
    let mean_b = values_b.iter().sum::<f64>() / n;
    let centered_a = values_a
        .iter()
        .map(|value| (value - mean_a).powi(2))
        .sum::<f64>();
    let centered_b = values_b
        .iter()
        .map(|value| (value - mean_b).powi(2))
        .sum::<f64>();
    if centered_a == 0.0 || centered_b == 0.0 {
        return None;
    }
    if centered_a.sqrt() < SCIPY_NEAR_CONSTANT_RELATIVE_NORM_V1 * mean_a.abs()
        || centered_b.sqrt() < SCIPY_NEAR_CONSTANT_RELATIVE_NORM_V1 * mean_b.abs()
    {
        return None;
    }
    let numerator = values_a
        .iter()
        .zip(values_b)
        .map(|(a, b)| (a - mean_a) * (b - mean_b))
        .sum::<f64>();
    let correlation = numerator / (centered_a.sqrt() * centered_b.sqrt());
    correlation.is_finite().then_some(correlation)
}

#[test]
fn official_authority_anchors_define_the_versioned_boundary() {
    let discovery = source("src/discovery.rs");
    require_all(
        &discovery,
        &[
            "neoethos.portfolio-correlation-authority.v1",
            SCIPY_PEARSONR_AUTHORITY_V1,
            SCIPY_SPEARMANR_AUTHORITY_V1,
            NIST_TIE_CORRECTED_SPEARMAN_AUTHORITY_V1,
            "const SCIPY_NEAR_CONSTANT_RELATIVE_NORM_V1: f64 = 1.0e-13;",
            "enum CorrelationUndefinedV1",
            "LengthMismatch",
            "InsufficientPairedObservations",
            "ConstantInput",
            "NearConstantInput",
            "NonFiniteResult",
            "InvalidThreshold",
        ],
    );
}

#[test]
fn pearson_and_tie_corrected_spearman_return_typed_undefined_outcomes() {
    let discovery = source("src/discovery.rs");
    let pearson_source = section(
        &discovery,
        "fn pearson_corr_i8(",
        "\npub fn ensure_portfolio_export_ready",
    );
    let spearman_source = section(&discovery, "fn spearman_corr_i8(", "\nfn pearson_corr_i8(");

    for correlation_source in [pearson_source, spearman_source] {
        require_all(
            correlation_source,
            &[
                "Result<f64, CorrelationUndefinedV1>",
                "validate_paired_correlation_shape_v1(a, b)?",
                "classify_centered_correlation_input_v1(",
                "finish_correlation_v1(",
            ],
        );
        assert!(
            !correlation_source.contains(".len().min("),
            "correlation must not truncate mismatched inputs"
        );
        assert!(
            !correlation_source.contains("return 0.0"),
            "undefined correlation must not become plausible zero"
        );
    }
}

#[test]
fn first_candidate_and_every_pair_use_one_fail_closed_gate() {
    let discovery = source("src/discovery.rs");
    let canonical_selection = section(
        &discovery,
        "let mut portfolio = Vec::new();",
        "progress_fn(DiscoveryProgress::PortfolioSelected",
    );
    let best_effort_selection = section(
        &discovery,
        "for ((_, gene), sig) in best_effort_fallback {",
        "funnel.record_stage(\"fallback_best_effort\"",
    );

    for selection in [canonical_selection, best_effort_selection] {
        require_all(
            selection,
            &[
                "portfolio_signal_is_correlation_rankable_v1(&sig)",
                "pairwise_portfolio_correlation_decision_v1(",
                "PortfolioCorrelationDecisionV1::Accept",
            ],
        );
        let rankability = selection
            .find("portfolio_signal_is_correlation_rankable_v1(&sig)")
            .expect("rankability gate");
        let accept = selection
            .find("portfolio_signals.push(sig)")
            .expect("portfolio acceptance");
        assert!(
            rankability < accept,
            "rankability must be checked before even the first candidate is accepted"
        );
    }
}

#[test]
fn threshold_equality_rejects_and_greedy_candidate_order_is_preserved() {
    let discovery = source("src/discovery.rs");
    let pair_gate = section(
        &discovery,
        "fn pairwise_portfolio_correlation_decision_v1(",
        "\nfn i8_midranks(",
    );
    require_all(
        pair_gate,
        &[
            "!threshold.is_finite()",
            "CorrelationUndefinedV1::InvalidThreshold",
            "pearson.abs() >= threshold || spearman.abs() >= threshold",
            "PortfolioCorrelationDecisionV1::RejectThreshold",
        ],
    );

    let selection = section(
        &discovery,
        "for (idx, ((_, gene), sig)) in filtered.into_iter().zip(signals_map).enumerate() {",
        "progress_fn(DiscoveryProgress::PortfolioSelected",
    );
    let compare = selection
        .find("for existing in &portfolio_signals")
        .expect("ranked greedy comparison loop");
    let accept = selection
        .find("portfolio_signals.push(sig)")
        .expect("ranked greedy acceptance");
    assert!(
        compare < accept,
        "candidate order must remain greedy and stable"
    );
}

#[test]
fn primary_formula_fixtures_cover_ties_constants_and_near_constants() {
    let x = [1_i8, 2, 3, 4, 5];
    let y = [5_i8, 6, 7, 8, 7];
    let x_ranks = midranks(&x);
    let y_ranks = midranks(&y);
    let spearman = pearson(&x_ranks, &y_ranks).expect("defined tie-corrected Spearman");
    assert!((spearman - 0.820_782_681_668_123_3).abs() < 1.0e-15);

    let constant = [0.0_f64; 5];
    assert_eq!(pearson(&[1.0, 2.0, 3.0, 4.0, 5.0], &constant), None);
    assert_eq!(pearson(&[1.0, 2.0], &[1.0]), None);

    let mean = 1.0_f64;
    let centered_sum_squares = (0.5 * SCIPY_NEAR_CONSTANT_RELATIVE_NORM_V1).powi(2);
    assert!(centered_sum_squares.sqrt() < SCIPY_NEAR_CONSTANT_RELATIVE_NORM_V1 * mean.abs());
}

#[test]
fn stale_constant_zero_fixture_is_retired_without_weakening_tie_parity() {
    let tests = source("src/discovery_tests.rs");
    require_all(
        &tests,
        &[
            "spearman_corr_i8(&a, &constant)",
            "Err(CorrelationUndefinedV1::ConstantInput)",
            "0.820_782_681_668_123_3",
            "classify_centered_correlation_input_v1(",
            "CorrelationUndefinedV1::NearConstantInput",
        ],
    );
    assert!(
        !tests.contains("assert_eq!(spearman_corr_i8(&a, &constant), 0.0)"),
        "constant correlation must not remain a plausible zero fixture"
    );
}
