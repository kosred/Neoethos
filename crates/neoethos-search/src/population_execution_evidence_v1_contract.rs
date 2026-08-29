use crate::data_selection::{
    CanonicalSearchArtifactScopeV2, CanonicalSearchInputReceiptV2, CanonicalSearchWindowRoleV1,
};
use crate::engine_identity::PopulationEvalEngine;
use crate::eval::BacktestSettings;
use crate::exact_resident_dataset_authority_v1::ExactResidentDatasetViewRequestV1;
use crate::population_execution_evidence_v1::{
    ExactPopulationExecutionErrorCodeV1, begin_exact_population_execution_run_v1,
};
use neoethos_data::{FeatureCellValidity, FeatureColumnF64, FeatureFrame, Ohlcv};

fn parent(rows: usize) -> (CanonicalSearchArtifactScopeV2, FeatureFrame, Ohlcv) {
    parent_with_changed_feature(rows, None)
}

fn parent_with_changed_feature(
    rows: usize,
    changed_row: Option<usize>,
) -> (CanonicalSearchArtifactScopeV2, FeatureFrame, Ohlcv) {
    let timestamps = neoethos_data::test_fixtures::canonical_test_timestamps(rows);
    let features = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
        timestamps.clone(),
        vec![
            FeatureColumnF64::new(
                "population_evidence_a",
                (0..rows)
                    .map(|row| {
                        let value = row as f64 + 0.25;
                        if changed_row == Some(row) {
                            f64::from_bits(value.to_bits() + 1)
                        } else {
                            value
                        }
                    })
                    .collect(),
                vec![FeatureCellValidity::Valid; rows],
            )
            .unwrap(),
            FeatureColumnF64::new(
                "population_evidence_b",
                (0..rows).map(|row| row as f64 * -0.5).collect(),
                vec![FeatureCellValidity::Valid; rows],
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let close = (0..rows)
        .map(|row| 1.10 + row as f64 * 0.000_01)
        .collect::<Vec<_>>();
    let ohlcv = Ohlcv {
        timestamp: Some(timestamps),
        open: close.iter().map(|value| value - 0.000_01).collect(),
        high: close.iter().map(|value| value + 0.000_10).collect(),
        low: close.iter().map(|value| value - 0.000_10).collect(),
        close,
        volume: Some((0..rows).map(|row| row as f64 + 100.0).collect()),
    };
    let anchor = features.provenance().bindings()[0]
        .dataset_identity()
        .clone();
    let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &features).unwrap();
    let scope = CanonicalSearchArtifactScopeV2::for_entire_receipt(
        CanonicalSearchWindowRoleV1::DiscoveryInput,
        receipt,
    )
    .unwrap();
    (scope, features, ohlcv)
}

#[test]
fn one_run_seals_full_range_and_index_views_against_one_exact_parent() {
    let (scope, features, ohlcv) = parent(12);
    let admission = crate::acquire_strict_discovery_device_admission_v1().unwrap();
    let run =
        begin_exact_population_execution_run_v1(admission, &scope, &features, &ohlcv).unwrap();
    let settings = BacktestSettings::default();

    let full = run
        .seal_evaluation(&settings, ExactResidentDatasetViewRequestV1::Full)
        .unwrap();
    let range = run
        .seal_evaluation(
            &settings,
            ExactResidentDatasetViewRequestV1::ContiguousRange { start: 2, end: 9 },
        )
        .unwrap();
    let indices = run
        .seal_evaluation(
            &settings,
            ExactResidentDatasetViewRequestV1::OrderedIndices(&[0, 2, 6, 11]),
        )
        .unwrap();

    assert_eq!(full.authority().parent_row_count(), 12);
    assert_eq!(range.authority().view().row_count(), 7);
    assert_eq!(indices.authority().view().row_count(), 4);
    assert_eq!(
        full.authority().parent_dataset_identity_sha256(),
        range.authority().parent_dataset_identity_sha256()
    );
    assert_eq!(
        range.authority().parent_dataset_identity_sha256(),
        indices.authority().parent_dataset_identity_sha256()
    );
}

#[test]
fn population_auto_reads_parent_and_route_facts_from_the_existing_run_without_a_probe() {
    let (scope, features, ohlcv) = parent(12);
    let admission = crate::acquire_strict_discovery_device_admission_v1().unwrap();
    let run =
        begin_exact_population_execution_run_v1(admission, &scope, &features, &ohlcv).unwrap();
    let facts = run
        .population_auto_sizing_primitives_v1()
        .expect("run-owned sizing primitives");
    assert_eq!(facts.resident_parent_rows, 12);
    assert_eq!(facts.feature_count, 2);
    assert_eq!(facts.parent_canonical_scope_identity_sha256.len(), 64);
    assert_eq!(facts.parent_dataset_identity_sha256.len(), 64);
    assert!(matches!(
        facts.route,
        crate::PopulationAutoSizingRouteV1::CpuNoCompatibleGpu { .. }
            | crate::PopulationAutoSizingRouteV1::NativeCuda { .. }
    ));

    let source = include_str!("population_execution_evidence_v1.rs");
    let method = source
        .split("pub(crate) fn population_auto_sizing_primitives_v1")
        .nth(1)
        .expect("sizing primitive method");
    let body = method
        .split("pub(crate) fn seal_evaluation")
        .next()
        .expect("method body");
    for forbidden in [
        "acquire_strict_discovery_device_admission_v1",
        "mem_get_info",
        "gpu_submission_ceiling",
        "device_count",
    ] {
        assert!(
            !body.contains(forbidden),
            "run-bound sizing must not probe again: {forbidden}"
        );
    }
}

#[cfg(feature = "gpu-b-native")]
#[test]
fn admitted_population_auto_runs_generation_zero_on_real_cuda() {
    if std::env::var("NEOETHOS_RUN_CUDA_SEARCH_TESTS").as_deref() != Ok("1") {
        eprintln!(
            "SKIPPED admitted_population_auto_runs_generation_zero_on_real_cuda — \
             set NEOETHOS_RUN_CUDA_SEARCH_TESTS=1 on the RTX host"
        );
        return;
    }

    crate::backend::install_evaluation_backend(
        crate::backend::EvaluationBackend::parse("cuda_required")
            .expect("typed CUDA-required backend"),
    )
    .expect("install CUDA-required evaluation backend");
    crate::genetic::set_migration_enabled(false);

    let (scope, features, ohlcv) = parent(400);
    let admission = crate::acquire_strict_discovery_device_admission_v1()
        .expect("real RTX strict-device admission");
    let run = begin_exact_population_execution_run_v1(admission, &scope, &features, &ohlcv)
        .expect("seal resident parent on the admitted RTX route");

    let mut config = crate::discovery::DiscoveryConfig::default();
    config.population = 200;
    config.population_auto = true;
    config.generations = 0;
    config.max_indicators = 5;
    config.evaluation_symbol = "EURUSD".to_owned();
    config.timeframe_label = "M15".to_owned();
    config.evaluation_spread_pips = 1.2;
    config.evaluation_commission_per_trade = 7.0;
    config.target_profile.min_payoff_ratio = 0.5;

    let stage1_len = ((features.n_samples() as f64
        * config.runtime_overrides.resolved_funnel_stage1_pct()) as usize)
        .min(features.n_samples());
    assert!(stage1_len > 0 && stage1_len < features.n_samples());
    let stage1_start = 0usize;
    let stage1_end = stage1_len;
    let stage1_features = features
        .row_window(stage1_start, stage1_end)
        .expect("exact Stage1 feature view");
    let stage1_ohlcv = Ohlcv {
        timestamp: ohlcv
            .timestamp
            .as_ref()
            .map(|values| values[stage1_start..stage1_end].to_vec()),
        open: ohlcv.open[stage1_start..stage1_end].to_vec(),
        high: ohlcv.high[stage1_start..stage1_end].to_vec(),
        low: ohlcv.low[stage1_start..stage1_end].to_vec(),
        close: ohlcv.close[stage1_start..stage1_end].to_vec(),
        volume: ohlcv
            .volume
            .as_ref()
            .map(|values| values[stage1_start..stage1_end].to_vec()),
    };

    let primitives = run
        .population_auto_sizing_primitives_v1()
        .expect("borrow admitted no-probe sizing facts");
    assert!(matches!(
        primitives.route,
        crate::PopulationAutoSizingRouteV1::NativeCuda { .. }
    ));
    let stage1_window =
        crate::population_auto_sizing_receipt_v1::seal_population_auto_stage1_window_v1(
            &primitives.parent_dataset_identity_sha256,
            "selection_stage1",
            stage1_start,
            stage1_end,
        )
        .expect("seal exact Stage1 sizing identity");
    let receipt = crate::population_auto_sizing_receipt_v1::seal_population_auto_sizing_receipt_v1(
        crate::population_auto_sizing_receipt_v1::PopulationAutoSizingRequestV1 {
            population_auto: config.population_auto,
            configured_population: config.population,
            resident_parent_rows: primitives.resident_parent_rows,
            evaluation_rows: stage1_len,
            feature_count: primitives.feature_count,
            month_capacity: crate::eval::current_backtest_runtime_overrides().month_capacity,
            requested_max_indicators: config.max_indicators,
            migration_enabled: false,
            parent_canonical_scope_identity_sha256: primitives
                .parent_canonical_scope_identity_sha256,
            parent_dataset_identity_sha256: primitives.parent_dataset_identity_sha256,
            stage1_window,
            route: primitives.route,
        },
    )
    .expect("seal admitted population-auto sizing receipt");
    assert_eq!(receipt.configured_population(), 200);
    assert!(
        receipt.resolved_population() > receipt.configured_population(),
        "the real RTX receipt must widen this M15-shaped generation-0 fixture"
    );

    let mut evaluation_config = config.evaluation_config(stage1_ohlcv.close.last().copied());
    // This synthetic fixture has no installed broker research contract. Give
    // it explicit EURUSD contract economics so the strict authority and the
    // native evaluator exercise the same finite money-per-pip inputs.
    evaluation_config.pip_value = 0.0001;
    evaluation_config.pip_value_per_lot = 10.0;
    let authority = crate::run_identity::build_population_auto_search_authority_v1(
        &config,
        &receipt,
        evaluation_config.pip_value_per_lot,
        neoethos_data::current_data_runtime_overrides().normalize_features,
    )
    .expect("build strict receipt-linked search authority");
    let result = crate::genetic::evolve_search_with_progress_and_limits_exact(
        &stage1_features,
        &stage1_ohlcv,
        receipt.resolved_population(),
        0,
        config.max_indicators,
        None,
        Some(evaluation_config),
        &run,
        ExactResidentDatasetViewRequestV1::ContiguousRange {
            start: stage1_start,
            end: stage1_end,
        },
        &authority,
        |_, _, _, _, _| {},
    )
    .expect("execute exact generation zero on the admitted CUDA route");
    assert_eq!(result.genes.len(), receipt.resolved_population());

    let completed = run.finish().expect("complete exact CUDA execution receipt");
    assert!(
        completed
            .engines()
            .contains(&PopulationEvalEngine::CudaNativeF64),
        "generation zero must record a successful native f64 population"
    );
}

#[test]
fn one_parent_is_sealed_once_and_views_derive_without_raw_parent_access() {
    let source = include_str!("population_execution_evidence_v1.rs");
    let begin = source
        .find("pub(crate) fn begin_exact_population_execution_run_v1")
        .expect("exact run constructor");
    let seal = source
        .find("pub(crate) fn seal_evaluation")
        .expect("exact view seal");
    let finish = source[seal..]
        .find("pub(crate) fn finish")
        .map(|offset| seal + offset)
        .expect("run finish delimiter");
    let begin_body = &source[begin..seal];
    let seal_body = &source[seal..finish];

    assert_eq!(
        begin_body
            .matches("seal_exact_resident_dataset_parent_v1(")
            .count(),
        1,
        "the complete parent is sealed once when the run begins"
    );
    assert!(seal_body.contains("derive_exact_resident_dataset_authority_v1("));
    for forbidden in [
        "seal_exact_resident_dataset_parent_v1(",
        "FeatureFrame",
        "Ohlcv",
        "dense_window(",
        "hash_parent(",
    ] {
        assert!(
            !seal_body.contains(forbidden),
            "view sealing must derive from the opaque parent: {forbidden}"
        );
    }
}

#[test]
fn an_unsampled_parent_mutation_changes_the_native_resident_identity() {
    let (scope_a, features_a, ohlcv_a) = parent_with_changed_feature(600, None);
    let (scope_b, features_b, ohlcv_b) = parent_with_changed_feature(600, Some(417));
    let admission_a = crate::acquire_strict_discovery_device_admission_v1().unwrap();
    let admission_b = crate::acquire_strict_discovery_device_admission_v1().unwrap();
    let run_a =
        begin_exact_population_execution_run_v1(admission_a, &scope_a, &features_a, &ohlcv_a)
            .unwrap();
    let run_b =
        begin_exact_population_execution_run_v1(admission_b, &scope_b, &features_b, &ohlcv_b)
            .unwrap();
    let settings = BacktestSettings::default();
    let evaluation_a = run_a
        .seal_evaluation(&settings, ExactResidentDatasetViewRequestV1::Full)
        .unwrap();
    let evaluation_b = run_b
        .seal_evaluation(&settings, ExactResidentDatasetViewRequestV1::Full)
        .unwrap();

    assert_ne!(
        evaluation_a.resident_identity_sha256(),
        evaluation_b.resident_identity_sha256(),
        "a formerly-unsampled feature bit must force a distinct resident upload"
    );
}

#[test]
fn evaluation_layout_must_match_the_sealed_view_before_any_engine_call() {
    let (scope, features, ohlcv) = parent(10);
    let admission = crate::acquire_strict_discovery_device_admission_v1().unwrap();
    let run =
        begin_exact_population_execution_run_v1(admission, &scope, &features, &ohlcv).unwrap();
    let evaluation = run
        .seal_evaluation(
            &BacktestSettings::default(),
            ExactResidentDatasetViewRequestV1::ContiguousRange { start: 3, end: 8 },
        )
        .unwrap();

    evaluation.validate_population_layout(5, 2).unwrap();
    for (rows, columns) in [(4, 2), (5, 1), (10, 2)] {
        let error = evaluation
            .validate_population_layout(rows, columns)
            .expect_err("detached or parent-shaped inputs must not enter the sealed view");
        assert_eq!(
            error.code(),
            ExactPopulationExecutionErrorCodeV1::ViewLayoutMismatch
        );
    }
}

#[test]
fn failed_or_wrong_cardinality_native_output_never_records_cuda() {
    let (scope, features, ohlcv) = parent(8);
    let admission = crate::acquire_strict_discovery_device_admission_v1().unwrap();
    let run =
        begin_exact_population_execution_run_v1(admission, &scope, &features, &ohlcv).unwrap();
    let evaluation = run
        .seal_evaluation(
            &BacktestSettings::default(),
            ExactResidentDatasetViewRequestV1::Full,
        )
        .unwrap();

    let mismatch = evaluation
        .record_successful_population(PopulationEvalEngine::CudaNativeF64, 4, 3)
        .expect_err("wrong output cardinality is not successful native evidence");
    assert_eq!(
        mismatch.code(),
        ExactPopulationExecutionErrorCodeV1::EngineReceipt
    );
    assert_eq!(
        run.finish().unwrap_err().code(),
        ExactPopulationExecutionErrorCodeV1::EngineReceipt
    );
}

#[test]
fn exact_success_is_run_scoped_and_finishes_to_the_existing_receipt() {
    let (scope_a, features_a, ohlcv_a) = parent(8);
    let (scope_b, features_b, ohlcv_b) = parent(9);
    let admission_a = crate::acquire_strict_discovery_device_admission_v1().unwrap();
    let admission_b = crate::acquire_strict_discovery_device_admission_v1().unwrap();
    let run_a =
        begin_exact_population_execution_run_v1(admission_a, &scope_a, &features_a, &ohlcv_a)
            .unwrap();
    let run_b =
        begin_exact_population_execution_run_v1(admission_b, &scope_b, &features_b, &ohlcv_b)
            .unwrap();
    let evaluation_a = run_a
        .seal_evaluation(
            &BacktestSettings::default(),
            ExactResidentDatasetViewRequestV1::Full,
        )
        .unwrap();
    let evaluation_b = run_b
        .seal_evaluation(
            &BacktestSettings::default(),
            ExactResidentDatasetViewRequestV1::Full,
        )
        .unwrap();

    evaluation_a
        .record_successful_population(PopulationEvalEngine::Cpu, 3, 3)
        .unwrap();
    evaluation_b
        .record_successful_population(PopulationEvalEngine::Cpu, 2, 2)
        .unwrap();
    let receipt_a = run_a.finish().unwrap();
    let receipt_b = run_b.finish().unwrap();
    assert_eq!(receipt_a.engines(), &[PopulationEvalEngine::Cpu]);
    assert_eq!(receipt_b.engines(), &[PopulationEvalEngine::Cpu]);
    assert_ne!(
        receipt_a.canonical_scope_identity_sha256(),
        receipt_b.canonical_scope_identity_sha256()
    );
}

#[test]
fn prototype_b_records_only_after_exact_output_validation() {
    let source = include_str!("gpu_native/prototype_b_population_eval.rs");
    let production = source.split("#[cfg(test)]").next().unwrap();
    let wrapper = production
        .find("pub(crate) fn try_evaluate_population_b(")
        .expect("sealed Prototype-B wrapper");
    let tail = &production[wrapper..];
    let raw_call = tail.find("evaluate_population_b_raw_v1(").unwrap();
    let exact_check = tail
        .find("require_exact_native_population_rows_v1(")
        .unwrap();
    let record = tail.find("record_successful_population(").unwrap();
    // The raw call is the argument of the exact-cardinality wrapper, so the
    // outer call token appears first in source while Rust evaluates the inner
    // raw call first. Both must complete before the receipt is recorded.
    assert!(exact_check < raw_call && raw_call < record);
    assert!(tail.contains(".validate_population_layout("));
    let evidence_source = include_str!("population_execution_evidence_v1.rs");
    let native_run_source =
        include_str!("population_execution_evidence_v1/native_cuda_resident_v1.rs");
    let native_bind = &evidence_source[evidence_source
        .find("pub(crate) fn bind_exact_native_population_view_v1")
        .expect("sealed native bind boundary")..];
    assert!(native_bind.contains("self.native_residency.bind_exact_native_population_view_v1("));
    assert!(native_bind.contains("&self.resident_identity_sha256"));
    assert!(native_run_source.contains("current_view_identity_sha256"));
    assert!(native_run_source.contains("Some(resident_execution_identity_sha256)"));
    let signature_end = tail.find(") -> Result<Vec<[f64; 11]>>").unwrap();
    let signature = &tail[..signature_end];
    for detached in [
        "close: &[f64]",
        "high: &[f64]",
        "low: &[f64]",
        "indicators: ArrayView2",
        "month_idx: &[i64]",
        "day_idx: &[i64]",
        "timestamps: &[i64]",
        "smc_data: &[SmcRow]",
        "settings: &BacktestSettings",
    ] {
        assert!(
            !signature.contains(detached),
            "Prototype-B must consume the buffers bound inside sealed evidence: {detached}"
        );
    }
    assert!(!production.contains("pub(crate) fn evaluate_population_b_raw_v1"));
    for stale_key in ["key: String", "sample_hash", "dataset_key"] {
        assert!(
            !production.contains(stale_key),
            "Prototype-B production retains stale cache-key authority `{stale_key}`"
        );
    }
}

#[test]
fn process_global_observation_and_unsealed_success_routes_are_absent() {
    let engine = include_str!("engine_identity.rs");
    let eval = include_str!("eval.rs");
    let cubecl = include_str!("cubecl_eval.rs");
    let evidence = include_str!("population_execution_evidence_v1.rs");
    let discovery = include_str!("discovery.rs");
    let funnel = include_str!("funnel_profile.rs");
    assert!(!engine.contains("AtomicU8"));
    assert!(!engine.contains("static OBSERVED"));
    assert!(!engine.contains("observed_population_engines"));
    assert!(!engine.contains("record_population_engine"));
    assert!(!engine.contains("reset_observed_population_engines"));
    assert!(!eval.contains("record_population_engine"));
    assert!(!cubecl.contains("record_population_engine"));
    assert!(!evidence.contains("Deserialize"));
    assert!(!evidence.contains("impl Default for ExactPopulationExecution"));
    assert!(!evidence.contains("OnceLock"));
    assert!(!evidence.contains("static CURRENT"));
    assert!(evidence.contains("pub(crate) fn begin_exact_population_execution_run_v1"));
    assert!(funnel.contains("attach_population_execution_run_receipt_v2"));
    assert!(funnel.contains("fn population_execution_run_receipt_v2("));
    assert!(discovery.contains(
        "attach_population_execution_run_receipt_v2(population_execution_run_receipt_v2)"
    ));
    assert!(discovery.contains(".population_execution_run_receipt_v2()"));
    assert!(discovery.contains("pub population_execution_run_receipt_v2:"));
    assert!(discovery.contains(
        "population_execution_run_receipt_v2: population_execution_run_receipt_v2.cloned()"
    ));
    assert!(!discovery.contains("pub population_engine_run_receipt:"));
}
