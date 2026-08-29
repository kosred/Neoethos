use super::execution_receipt_continuation_v1_tests::{
    native_receipt_from_value_v1, native_receipt_value_v1, unchecked_native_receipt_from_value_v1,
};
use super::payload_v1_tests::{gene_for_metrics_v1, metric_row_v1, valid_evaluation_config_v1};
#[allow(unused_imports)]
use super::*;
use crate::canonical_native_discovery_request_v1::{
    CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_SCHEMA_V1,
    CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_VERSION_V1, CanonicalNativeDiscoveryRequestV1,
    CanonicalNativeGenerationZeroOverridesV1, CanonicalResearchContractArtifactRefV1,
    resolve_canonical_native_discovery_request_v1,
};
use crate::canonical_native_runtime_authority_v1::install_and_seal_canonical_native_runtime_authority_v1;
use crate::canonical_trendbar_research::{
    CanonicalTrendbarResearchCostAssumptionsV2, CanonicalTrendbarResearchExecutionContractV3,
};
use crate::data_selection::{
    CanonicalGpuResidentSearchInputReceiptV3, CanonicalSearchInput, CanonicalSearchInputReceiptV2,
};
use crate::genetic::{EvaluationConfig, SearchResult};
use crate::historical_research::{
    HistoricalResearchArtifactClassV1, HistoricalResearchPromotionEligibilityV1,
};
use neoethos_core::Settings;
use neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1;
use neoethos_data::core::features::FeatureBuildOptions;
use neoethos_data::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalDatasetSeriesReceiptV1,
    CanonicalOhlcvPublishRequest, CanonicalTimeframe, CanonicalVolumeRef,
    SelectedDatasetGenerationV1, publish_canonical_ohlcv_generation,
};
use neoethos_gpu_cuda::{PopulationMetricsOnlyPlanV1, PopulationResidencyCountersV1};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

const ARTIFACT_V1: &str = "research/contracts/chunk2a2-\".json";

struct HighLevelResultFixtureV1 {
    _root: TempDir,
    request: CanonicalNativeDiscoveryRequestV1,
    financial_contract: CanonicalTrendbarResearchExecutionContractV3,
    native_receipt_v3: CanonicalGpuResidentSearchInputReceiptV3,
    sizing_receipt_v2: ResidentPopulationAutoSizingReceiptV2,
    evaluation_config: EvaluationConfig,
    milestone: crate::prepared_discovery_run_input_v3::ResidentGenerationZeroMilestoneV1,
}

fn compatible_settings_v1(root: &std::path::Path) -> Settings {
    let mut settings = Settings::default();
    settings.system.data_dir = root.to_owned();
    settings.system.symbol = "EURUSD".to_owned();
    settings.system.account_currency = "USD".to_owned();
    settings.system.base_timeframe = "M1".to_owned();
    settings.system.higher_timeframes = vec!["H4".to_owned()];
    settings.models.prop_search_population = 10;
    settings.models.prop_search_population_auto = false;
    settings.models.prop_search_generations = 1;
    settings.models.prop_search_max_indicators = 5;
    settings.models.prop_search_max_rows = 0;
    settings.models.prop_search_max_rows_by_tf.clear();
    settings.models.prop_search_min_payoff_ratio = 0.0;
    settings.models.prop_search_device = "auto".to_owned();
    settings.models.discovery_runtime.prefilter_top_k = 0;
    settings.models.discovery_runtime.min_history_years = 0;
    settings.models.discovery_runtime.adaptive_thresholds = false;
    settings.models.discovery_ledger.enabled = false;
    settings.models.gene_stop_bounds.atr_scaled = false;
    settings
}

fn publish_source_v1(
    root: &std::path::Path,
    timeframe: CanonicalTimeframe,
    tag: &str,
) -> SelectedDatasetGenerationV1 {
    let identity = CanonicalDatasetIdentity::external(
        "neoethos-chunk2a2-high-level-result",
        "EURUSD",
        timeframe,
        BarTimestampConvention::BarOpen,
    )
    .unwrap();
    let provenance = ProducerProvenanceEnvelopeV1::new(
        "neoethos.search.chunk2a2-high-level-result-fixture.v1",
        tag.as_bytes().to_vec(),
    )
    .unwrap();
    let bars = neoethos_data::test_fixtures::ctrader_sample_ohlcv();
    let published = publish_canonical_ohlcv_generation(CanonicalOhlcvPublishRequest {
        configured_root: root,
        identity: &identity,
        expected_generation: None,
        provenance: &provenance,
        ohlcv: &bars,
        volume: CanonicalVolumeRef::Float64(bars.volume.as_deref().unwrap()),
        rows_per_chunk: 2,
    })
    .unwrap();
    SelectedDatasetGenerationV1::from_manifest(published.manifest()).unwrap()
}

fn write_contract_v1(
    root: &std::path::Path,
    anchor: &SelectedDatasetGenerationV1,
    higher: &SelectedDatasetGenerationV1,
) -> String {
    let series =
        CanonicalDatasetSeriesReceiptV1::new(anchor.clone(), vec![anchor.clone(), higher.clone()])
            .unwrap();
    let options = FeatureBuildOptions {
        higher_tfs: vec!["H4".to_owned()],
        ..FeatureBuildOptions::default()
    };
    let input = CanonicalSearchInput::from_exact_series_receipt(
        root,
        &series,
        CanonicalTimeframe::M1,
        &options,
    )
    .unwrap();
    let mut receipt_value = serde_json::to_value(input.receipt().unwrap()).unwrap();
    let segments = receipt_value["source_bindings"][0]["segments"]
        .as_array_mut()
        .unwrap();
    let original = segments[0].clone();
    let row_start = original["row_start"].as_u64().unwrap();
    let row_end = original["row_end"].as_u64().unwrap();
    let timestamp_start = original["timestamp_start_ms"].as_i64().unwrap();
    let timestamp_end = original["timestamp_end_ms"].as_i64().unwrap();
    assert!(row_end - row_start >= 2 && timestamp_start < timestamp_end);
    let row_split = row_start + 1;
    let timestamp_split = timestamp_start + 1;
    segments[0] = serde_json::json!({
        "row_start": row_start,
        "row_end": row_split,
        "timestamp_start_ms": timestamp_start,
        "timestamp_end_ms": timestamp_start,
    });
    segments.insert(
        1,
        serde_json::json!({
            "row_start": row_split,
            "row_end": row_end,
            "timestamp_start_ms": timestamp_split,
            "timestamp_end_ms": timestamp_end,
        }),
    );
    let receipt: CanonicalSearchInputReceiptV2 = serde_json::from_value(receipt_value).unwrap();
    receipt.validate().unwrap();
    let assumption_sha = format!("{:x}", Sha256::digest(b"chunk2a2-high-level-financials"));
    let contract = CanonicalTrendbarResearchExecutionContractV3::new(
        receipt,
        CanonicalTrendbarResearchCostAssumptionsV2 {
            symbol: "EURUSD",
            account_currency: "USD",
            assumption_source_id: "neoethos.test.chunk2a2-high-level-financials.v1",
            assumption_source_sha256: &assumption_sha,
            pip_size: 0.0001,
            pip_value_per_lot: 10.0,
            full_spread_pips_assumption: 1.2,
            slippage_pips_per_fill_assumption: 0.1,
            commission_account_per_lot_per_fill_assumption: 3.5,
            swap_long_pips_per_day: -0.2,
            swap_short_pips_per_day: -0.1,
            pnl_conversion_fee_rate: 0.0,
        },
    )
    .unwrap();
    let bytes = serde_json::to_vec(&contract).unwrap();
    let path = root.join(ARTIFACT_V1);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, &bytes).unwrap();
    format!("{:x}", Sha256::digest(bytes))
}

fn request_evidence_from_request_v1(
    request: &CanonicalNativeDiscoveryRequestV1,
) -> CanonicalNativeGenerationZeroRequestEvidenceV1 {
    let loaded = request.loaded_contract();
    let scope = request.scope();
    let limits = request.limits();
    CanonicalNativeGenerationZeroRequestEvidenceV1 {
        execution_scope: scope.execution_scope(),
        artifact_class: HistoricalResearchArtifactClassV1::ResearchOnly,
        promotion_eligibility: HistoricalResearchPromotionEligibilityV1::NotPromotionEligible,
        authorization_issued: false,
        contract_artifact_reference_schema: CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_SCHEMA_V1
            .to_owned(),
        contract_artifact_reference_version: CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_VERSION_V1,
        contract_artifact_relative_path: loaded.relative_path().to_owned(),
        contract_artifact_expected_sha256: loaded.exact_artifact_sha256().to_owned(),
        contract_artifact_exact_file_sha256: loaded.exact_artifact_sha256().to_owned(),
        contract_artifact_exact_file_byte_count: loaded.byte_len(),
        contract_domain_identity_sha256: loaded.contract_identity_sha256().to_owned(),
        startup_settings_id: request.startup_settings_sha256().to_owned(),
        runtime_install_receipt_id: request
            .runtime_install_receipt()
            .identity_sha256()
            .to_owned(),
        generation_zero_runtime_authority_id: request
            .runtime_authority()
            .identity_sha256()
            .to_owned(),
        unused_full_search_scope_id: scope.identity_sha256().to_owned(),
        raw_generations: scope.raw_legacy_generations_unused_full_search(),
        clamped_generations: scope.clamped_legacy_generations_unused_full_search(),
        cost_band_status: scope.cost_band_status(),
        cost_band: scope.cost_band_pips_unused_generation_zero(),
        configured_population_cap: limits.configured_population_cap(),
        resolved_population_cap: limits.resolved_population_cap(),
        term_cap: limits.term_cap(),
        string_bytes_cap: limits.string_bytes_cap(),
        vector_elements_cap: limits.vector_elements_cap(),
        source_count_cap: limits.source_count_cap(),
        result_bytes_cap: limits.result_bytes_cap(),
    }
}

fn fixed_shape_from_request_v1(
    request: &CanonicalNativeDiscoveryRequestV1,
) -> CanonicalNativeGenerationZeroFixedMetadataShapeV1 {
    let loaded = request.loaded_contract();
    let bindings = loaded.source_projection().bindings();
    let total_source_segment_count = bindings
        .iter()
        .try_fold(0_usize, |total, binding| {
            total.checked_add(binding.segments().len())
        })
        .unwrap();
    CanonicalNativeGenerationZeroFixedMetadataShapeV1 {
        contract_compact_json_bytes: checked_compact_json_byte_count_v1(loaded.contract()).unwrap(),
        contract_artifact_relative_path_compact_json_bytes:
            checked_compact_json_string_byte_count_v1(loaded.relative_path()).unwrap(),
        source_count: bindings.len(),
        total_source_segment_count,
    }
}

fn assert_preflight_matches_oracle_v1(
    actual: &CanonicalNativeGenerationZeroResultPreflightV1,
    oracle: &CanonicalNativeGenerationZeroResultPreflightV1,
) {
    assert_eq!(
        actual.prepared_feature_count(),
        oracle.prepared_feature_count()
    );
    assert_eq!(
        actual.raw_configured_max_indicators(),
        oracle.raw_configured_max_indicators()
    );
    assert_eq!(
        actual.resolved_max_indicators(),
        oracle.resolved_max_indicators()
    );
    assert_eq!(actual.term_cap(), oracle.term_cap());
    assert_eq!(
        actual.configured_population(),
        oracle.configured_population()
    );
    assert_eq!(actual.population_cap(), oracle.population_cap());
    assert_eq!(
        actual.fixed_metadata_upper_bound_with_empty_arrays_bytes(),
        oracle.fixed_metadata_upper_bound_with_empty_arrays_bytes()
    );
    for population in [actual.configured_population(), actual.population_cap()] {
        assert_eq!(
            actual
                .checked_upper_bound_for_population(population)
                .unwrap(),
            oracle
                .checked_upper_bound_for_population(population)
                .unwrap()
        );
    }
}

fn evaluation_config_for_contract_v1(
    contract: &CanonicalTrendbarResearchExecutionContractV3,
) -> EvaluationConfig {
    let mut evaluation = valid_evaluation_config_v1(false);
    evaluation.pip_value = contract.pip_size();
    evaluation.pip_value_per_lot = contract.pip_value_per_lot();
    evaluation.spread_pips = contract.screening_spread_and_slippage_round_trip_pips();
    evaluation.commission_per_trade = contract.round_trip_commission_account_per_lot();
    evaluation.swap_long_pips_per_day = contract.swap_long_pips_per_day();
    evaluation.swap_short_pips_per_day = contract.swap_short_pips_per_day();
    evaluation.pnl_conversion_fee_rate = contract.pnl_conversion_fee_rate();
    evaluation
}

fn milestone_v1(
    native_identity: String,
    sizing_receipt: &ResidentPopulationAutoSizingReceiptV2,
) -> crate::prepared_discovery_run_input_v3::ResidentGenerationZeroMilestoneV1 {
    let population = sizing_receipt.resolved_population();
    let scenario_count = sizing_receipt.max_concurrent_scenario_count();
    let launch_count = population.div_ceil(scenario_count);
    let mut genes = Vec::with_capacity(population);
    let mut metrics = Vec::with_capacity(population);
    for candidate in 0..population {
        let mut row = metric_row_v1();
        row[0] += candidate as f64;
        row[2] += candidate as f64;
        let mut gene = gene_for_metrics_v1(&row, false);
        gene.strategy_id = format!("gen0-strategy-{candidate}");
        genes.push(gene);
        metrics.push(row);
    }
    let metric_bytes = PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(
        population,
        u32::try_from(sizing_receipt.month_capacity()).unwrap(),
    )
    .unwrap()
    .metric_rows_bytes();
    let raw_counters = [
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        launch_count as u64,
        launch_count as u64,
        population as u64,
        metric_bytes,
        0,
        0,
        0,
        0,
        0,
    ];
    // SAFETY: the source census freezes repr(C), 18 ordered u64 fields,
    // size 144, and alignment 8 for this test-only fixture.
    let counters =
        unsafe { std::mem::transmute::<[u64; 18], PopulationResidencyCountersV1>(raw_counters) };
    let receipts = (0..launch_count)
        .map(|launch| [u8::try_from(launch + 1).unwrap(); 32])
        .collect();
    crate::prepared_discovery_run_input_v3::ResidentGenerationZeroMilestoneV1::test_fixture_v1(
        0,
        native_identity,
        sizing_receipt.identity_sha256().to_owned(),
        population,
        sizing_receipt.term_cap(),
        sizing_receipt.stage1_row_start(),
        sizing_receipt.stage1_row_end(),
        receipts,
        None,
        counters,
        SearchResult {
            genes,
            metrics,
            effective_smc_gate_threshold: 0.5,
        },
    )
}

fn high_level_fixture_v1() -> HighLevelResultFixtureV1 {
    let root = TempDir::new().unwrap();
    let anchor = publish_source_v1(root.path(), CanonicalTimeframe::M1, "anchor");
    let higher = publish_source_v1(root.path(), CanonicalTimeframe::H4, "higher");
    let artifact_sha = write_contract_v1(root.path(), &anchor, &higher);
    let settings = compatible_settings_v1(root.path());
    crate::genetic::set_migration_enabled(false);
    let install = install_and_seal_canonical_native_runtime_authority_v1(&settings).unwrap();
    let reference =
        CanonicalResearchContractArtifactRefV1::checked_new(ARTIFACT_V1, artifact_sha).unwrap();
    let overrides =
        CanonicalNativeGenerationZeroOverridesV1::checked_new(Some(10), Some(false), Some(0))
            .unwrap();
    let request =
        resolve_canonical_native_discovery_request_v1(&settings, &install, reference, overrides)
            .unwrap();
    let financial_contract = request.loaded_contract().contract().clone();
    let native_value = native_receipt_value_v1(&financial_contract);
    let native_receipt_v3 = native_receipt_from_value_v1(&native_value);
    let projection = crate::resident_population_auto_sizing_receipt_v2::
        canonical_pinned_source_projection_from_search_receipt_v1(
            financial_contract.input_receipt(),
        )
        .unwrap();
    let sizing_receipt_v2 = crate::resident_population_auto_sizing_receipt_v2::tests::
        canonical_result_fixture_receipt_v2(
            financial_contract.identity_sha256().unwrap(),
            financial_contract.input_receipt_sha256().to_owned(),
            projection.identity_sha256(),
            0,
            native_receipt_v3.row_count(),
            0,
            native_receipt_v3.row_count(),
            4,
        );
    let evaluation_config = evaluation_config_for_contract_v1(&financial_contract);
    let milestone = milestone_v1(
        native_receipt_v3.identity_sha256().unwrap(),
        &sizing_receipt_v2,
    );
    HighLevelResultFixtureV1 {
        _root: root,
        request,
        financial_contract,
        native_receipt_v3,
        sizing_receipt_v2,
        evaluation_config,
        milestone,
    }
}

#[test]
fn high_level_preflight_and_sealer_match_private_oracles_and_reject_forged_authority() {
    let fixture = high_level_fixture_v1();
    let request = &fixture.request;
    let prepared_feature_count = fixture.sizing_receipt_v2.feature_count();
    let shape = fixed_shape_from_request_v1(request);
    assert!(shape.source_count > 1);
    assert!(shape.total_source_segment_count > shape.source_count);

    let oracle_preflight = checked_preflight_from_fixed_metadata_shape_v1(
        prepared_feature_count,
        request.config().max_indicators,
        request.config().population,
        shape,
    )
    .unwrap();
    let high_level_preflight =
        preflight_canonical_native_generation_zero_result_v1(request, prepared_feature_count)
            .unwrap();
    assert_preflight_matches_oracle_v1(&high_level_preflight, &oracle_preflight);

    let evaluation_evidence =
        CanonicalNativeGenerationZeroEvaluationEvidenceV1::checked_from_evaluation_config_v1(
            &fixture.evaluation_config,
            crate::discovery::DiscoveryMode::PropFirm,
        )
        .unwrap();
    let (oracle_view, oracle_seal) =
        checked_seal_canonical_native_generation_zero_research_result_from_evidence_v1(
            oracle_preflight,
            request_evidence_from_request_v1(request),
            fixture.financial_contract.clone(),
            fixture.native_receipt_v3.clone(),
            fixture.sizing_receipt_v2.clone(),
            evaluation_evidence,
            &fixture.milestone,
        )
        .unwrap();
    let (high_level_view, high_level_seal) =
        seal_canonical_native_generation_zero_research_result_v1(
            request,
            high_level_preflight,
            fixture.financial_contract.clone(),
            fixture.native_receipt_v3.clone(),
            fixture.sizing_receipt_v2.clone(),
            fixture.evaluation_config.clone(),
            &fixture.milestone,
        )
        .unwrap();
    assert_eq!(high_level_seal, oracle_seal);
    assert_eq!(
        high_level_view.evidence_identity_sha256(),
        oracle_view.evidence_identity_sha256()
    );
    let mut oracle_bytes = Vec::new();
    let mut high_level_bytes = Vec::new();
    write_canonical_native_generation_zero_research_result_v1(
        &oracle_view,
        &mut oracle_bytes,
        MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1,
    )
    .unwrap();
    write_canonical_native_generation_zero_research_result_v1(
        &high_level_view,
        &mut high_level_bytes,
        MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1,
    )
    .unwrap();
    assert_eq!(high_level_bytes, oracle_bytes);
    let empty_count = high_level_view
        .checked_fixed_metadata_with_empty_arrays_byte_count_v1()
        .unwrap();
    assert!(
        empty_count
            <= high_level_view
                .preflight()
                .fixed_metadata_upper_bound_with_empty_arrays_bytes()
    );
    assert!(
        high_level_seal.byte_count()
            <= high_level_view
                .preflight()
                .checked_upper_bound_for_population(
                    fixture.sizing_receipt_v2.resolved_population(),
                )
                .unwrap()
    );
    drop(high_level_view);
    drop(oracle_view);

    let mut underdeclared_shapes = Vec::new();
    underdeclared_shapes.push(CanonicalNativeGenerationZeroFixedMetadataShapeV1 {
        contract_compact_json_bytes: shape.contract_compact_json_bytes - 1,
        ..shape
    });
    underdeclared_shapes.push(CanonicalNativeGenerationZeroFixedMetadataShapeV1 {
        contract_artifact_relative_path_compact_json_bytes: shape
            .contract_artifact_relative_path_compact_json_bytes
            - 1,
        ..shape
    });
    underdeclared_shapes.push(CanonicalNativeGenerationZeroFixedMetadataShapeV1 {
        source_count: shape.source_count - 1,
        ..shape
    });
    underdeclared_shapes.push(CanonicalNativeGenerationZeroFixedMetadataShapeV1 {
        total_source_segment_count: shape.total_source_segment_count - 1,
        ..shape
    });
    for forged_shape in underdeclared_shapes {
        let forged_preflight = checked_preflight_from_fixed_metadata_shape_v1(
            prepared_feature_count,
            request.config().max_indicators,
            request.config().population,
            forged_shape,
        )
        .unwrap();
        assert!(
            seal_canonical_native_generation_zero_research_result_v1(
                request,
                forged_preflight,
                fixture.financial_contract.clone(),
                fixture.native_receipt_v3.clone(),
                fixture.sizing_receipt_v2.clone(),
                fixture.evaluation_config.clone(),
                &fixture.milestone,
            )
            .is_err(),
            "high-level sealer accepted an underdeclared fixed-metadata preflight"
        );
    }

    let fresh_preflight = || {
        preflight_canonical_native_generation_zero_result_v1(request, prepared_feature_count)
            .unwrap()
    };
    let mut forged_contract_value = serde_json::to_value(&fixture.financial_contract).unwrap();
    forged_contract_value["input_receipt_sha256"] = serde_json::json!("9".repeat(64));
    let forged_contract = serde_json::from_value(forged_contract_value).unwrap();
    assert!(
        seal_canonical_native_generation_zero_research_result_v1(
            request,
            fresh_preflight(),
            forged_contract,
            fixture.native_receipt_v3.clone(),
            fixture.sizing_receipt_v2.clone(),
            fixture.evaluation_config.clone(),
            &fixture.milestone,
        )
        .is_err(),
        "high-level sealer skipped contract self-validation"
    );

    let mut forged_native_value = serde_json::to_value(&fixture.native_receipt_v3).unwrap();
    forged_native_value["source_bindings"][0]["segments"][0]["row_end"] =
        serde_json::json!(u64::MAX);
    let forged_native = unchecked_native_receipt_from_value_v1(forged_native_value);
    assert!(
        seal_canonical_native_generation_zero_research_result_v1(
            request,
            fresh_preflight(),
            fixture.financial_contract.clone(),
            forged_native,
            fixture.sizing_receipt_v2.clone(),
            fixture.evaluation_config.clone(),
            &fixture.milestone,
        )
        .is_err(),
        "high-level sealer skipped native V3 validation/projection binding"
    );

    let forged_sizing = crate::resident_population_auto_sizing_receipt_v2::tests::
        canonical_result_forge_feature_count_without_rehash_v2_for_test(
            fixture.sizing_receipt_v2.clone(),
        );
    assert!(
        seal_canonical_native_generation_zero_research_result_v1(
            request,
            fresh_preflight(),
            fixture.financial_contract.clone(),
            fixture.native_receipt_v3.clone(),
            forged_sizing,
            fixture.evaluation_config.clone(),
            &fixture.milestone,
        )
        .is_err(),
        "high-level sealer skipped sizing V2 self-validation"
    );

    let forged_milestone = milestone_v1("9".repeat(64), &fixture.sizing_receipt_v2);
    assert!(
        seal_canonical_native_generation_zero_research_result_v1(
            request,
            fresh_preflight(),
            fixture.financial_contract.clone(),
            fixture.native_receipt_v3.clone(),
            fixture.sizing_receipt_v2.clone(),
            fixture.evaluation_config.clone(),
            &forged_milestone,
        )
        .is_err(),
        "high-level sealer skipped milestone/native identity binding"
    );
}
