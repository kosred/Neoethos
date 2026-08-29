use neoethos_search::scoring::{ga_fitness, ga_fitness_growth};
use neoethos_search::{
    BacktestMetrics, CANONICAL_BACKTEST_ARTIFACT_KIND, CANONICAL_BACKTEST_SCHEMA_VERSION,
    FORWARD_TEST_VALIDATION_ARTIFACT_KIND, FORWARD_TEST_VALIDATION_SCHEMA_VERSION,
    LIVE_EXECUTION_SIMULATION_ARTIFACT_KIND, LIVE_EXECUTION_SIMULATION_SCHEMA_VERSION,
};
use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn source(relative: &str) -> String {
    let path = manifest_dir().join(relative);
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

fn canonical_metric_row(monthly_target_hit_rate: f64) -> [f64; 11] {
    [
        12_000.0,
        1.75,
        112_000.0,
        0.04,
        0.62,
        1.8,
        120.0,
        monthly_target_hit_rate,
        120.0,
        0.82,
        0.01,
    ]
}

#[test]
fn all_eleven_metric_slots_round_trip_without_reordering_or_loss() {
    let raw = canonical_metric_row(0.73);
    let typed = BacktestMetrics::from_metric_array(raw);

    let wire = serde_json::to_value(typed).expect("serialize typed metrics");
    assert_eq!(wire["monthly_target_hit_rate"], 0.73);
    assert_eq!(typed.to_metric_array(), raw);
    assert_eq!(<[f64; 11]>::from(BacktestMetrics::from(raw)), raw);
}

#[test]
fn ga_and_growth_ranking_consumers_observe_the_exact_round_trip_row() {
    let raw = canonical_metric_row(0.73);
    let round_trip = BacktestMetrics::from_metric_array(raw).to_metric_array();
    let without_monthly_hit = canonical_metric_row(0.0);

    assert!(
        ga_fitness(&raw) > ga_fitness(&without_monthly_hit),
        "slot seven must remain the dominant ga_fitness monthly-hit reward"
    );
    assert_eq!(ga_fitness(&round_trip), ga_fitness(&raw));
    assert_eq!(ga_fitness_growth(&round_trip), ga_fitness_growth(&raw));
}

#[test]
fn typed_wire_maps_slot_seven_without_default_or_schema_width_change() {
    let eval = source("src/eval.rs");
    let typed = section(&eval, "pub struct BacktestMetrics {", "\n}");
    for required in [
        "pub monthly_target_hit_rate: f64,",
        "pub trade_count: usize,",
        "pub consistency: f64,",
    ] {
        assert!(typed.contains(required), "typed metrics omit {required:?}");
    }
    assert!(
        !typed.contains("serde(default"),
        "legacy rows must not silently deserialize monthly hit as zero"
    );

    let conversion = section(
        &eval,
        "impl BacktestMetrics {",
        "\nimpl From<[f64; 11]> for BacktestMetrics",
    );
    for required in [
        "monthly_target_hit_rate: metrics[Self::MONTHLY_TARGET_HIT_RATE_INDEX]",
        "self.monthly_target_hit_rate,",
        "pub fn from_metric_array(metrics: [f64; 11]) -> Self",
        "pub fn to_metric_array(self) -> [f64; 11]",
    ] {
        assert!(
            conversion.contains(required),
            "11-slot conversion omits {required:?}"
        );
    }
    assert!(
        !conversion.contains("slot 7: monthly_target_hit_rate is not modelled"),
        "retired lossy slot-seven path remains"
    );
}

#[test]
fn persisted_artifacts_use_new_versions_and_reject_legacy_before_typed_use() {
    assert_eq!(CANONICAL_BACKTEST_SCHEMA_VERSION, 3);
    assert_eq!(FORWARD_TEST_VALIDATION_SCHEMA_VERSION, 3);
    assert_eq!(LIVE_EXECUTION_SIMULATION_SCHEMA_VERSION, 2);
    assert!(CANONICAL_BACKTEST_ARTIFACT_KIND.ends_with(".v3"));
    assert!(FORWARD_TEST_VALIDATION_ARTIFACT_KIND.ends_with(".v3"));
    assert!(LIVE_EXECUTION_SIMULATION_ARTIFACT_KIND.ends_with(".v2"));

    let validation = source("src/validation.rs");
    for required in [
        "struct CanonicalBacktestPayloadV3",
        "struct ForwardTestValidationPayloadV3",
        "reject_legacy_metric_artifact_payload_v1(bytes, \"canonical backtest\", 3)?;",
        "reject_legacy_metric_artifact_payload_v1(bytes, \"forward test\", 3)?;",
        "reject_legacy_live_execution_simulation_v1(&bytes, 2)?;",
    ] {
        assert!(
            validation.contains(required),
            "versioned metric persistence omits {required:?}"
        );
    }
}

#[test]
fn stale_zero_loss_fixtures_are_replaced_without_touching_gpu_producers() {
    let oracle = source("src/gpu_native/prototype_population_oracle.rs");
    let oracle_test = section(
        &oracle,
        "fn oracle_metric_row_keeps_raw_monthly_hit_rate_in_slot_seven()",
        "\n    #[test]",
    );
    assert!(
        oracle_test.contains("BacktestMetrics::from_metric_array(row.values).to_metric_array()[7]")
    );
    assert!(
        oracle_test.contains("1.0"),
        "device-oracle roundtrip must preserve the producer's slot seven"
    );
    assert!(
        !oracle_test.contains("0.0"),
        "device-oracle fixture still blesses slot-seven loss"
    );

    let app_tests = source("../neoethos-app/src/app_services/discovery_tests.rs");
    let fixture = section(
        &app_tests,
        "let lo_oos_metrics = BacktestMetrics {",
        "let (search_input_receipt, selection_scope)",
    );
    assert_eq!(
        fixture.matches("monthly_target_hit_rate:").count(),
        1,
        "the one full App BacktestMetrics literal must name monthly hit; the update literal inherits it"
    );
}
