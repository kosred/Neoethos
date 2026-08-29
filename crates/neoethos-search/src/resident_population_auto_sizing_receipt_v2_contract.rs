#[test]
fn resident_population_auto_v2_surface_is_available_to_the_staged_cuda_route() {
    use crate::resident_population_auto_sizing_receipt_v2::{
        ResidentPopulationAutoSizingReceiptV2, ResidentPopulationAutoSizingRequestV2,
        evaluation_config_from_canonical_trendbar_contract_v2,
        seal_resident_population_auto_for_canonical_trendbar_research_v2,
    };

    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ResidentPopulationAutoSizingReceiptV2>();
    assert_send_sync::<ResidentPopulationAutoSizingRequestV2>();
    let _ = seal_resident_population_auto_for_canonical_trendbar_research_v2;
    let _ = evaluation_config_from_canonical_trendbar_contract_v2;
}

#[test]
fn staged_cuda_v5_retains_the_population_sizing_receipt() {
    use crate::prepared_discovery_run_input_v3::PreparedCanonicalDiscoveryRunInputV5;

    assert!(std::mem::size_of::<Option<PreparedCanonicalDiscoveryRunInputV5>>() > 0);
    fn assert_full_authorities(input: &PreparedCanonicalDiscoveryRunInputV5) {
        let _ = input.population_sizing_receipt_v2();
        let _ = input.financial_contract_v3();
        let _ = input.exact_evaluation_config_v2();
    }
    let _ = assert_full_authorities;
}

#[test]
fn resident_v2_binds_exact_financial_value_authority() {
    use crate::resident_population_auto_sizing_receipt_v2::ResidentPopulationAutoSizingReceiptV2;

    fn assert_financial_authority(receipt: &ResidentPopulationAutoSizingReceiptV2) {
        assert!(receipt.pip_value_per_lot().is_finite());
        assert!(!receipt.financial_authority_identity_sha256().is_empty());
    }
    let _ = assert_financial_authority;

    let source = include_str!("resident_population_auto_sizing_receipt_v2.rs");
    assert!(
        !source.contains("evaluation_config(None)"),
        "resident sizing must not substitute metadata/typical price for exact Stage1 financial truth"
    );
    assert!(source.contains("CanonicalTrendbarResearchExecutionContractV3"));
    assert!(source.contains("config.session_spread_pips.is_none()"));
}

#[test]
fn resident_v2_distinguishes_requested_effective_and_typed_adaptive_evidence() {
    use crate::resident_population_auto_sizing_receipt_v2::ResidentPopulationAutoSizingReceiptV2;
    use neoethos_gpu_cuda::{ResidentAdaptiveBaseRequestV1, ResidentAdaptiveBaseViewTokenV1};

    fn assert_adaptive_authority(
        receipt: &ResidentPopulationAutoSizingReceiptV2,
        request: &ResidentAdaptiveBaseRequestV1,
        token: &ResidentAdaptiveBaseViewTokenV1,
    ) {
        let _ = receipt.adaptive_stops_requested_for_run();
        let _ = receipt.adaptive_base_effective_for_stage1();
        let _ = receipt.adaptive_resolution_reason();
        let _ = receipt.resident_adaptive_semantic_v1();
        let _ = receipt.stop_target_log_operation_schedule_v3();
        let _ = receipt.resident_adaptive_view_and_request_v2();
        let _ = receipt.validate_resident_adaptive_view_token_v2(request, token);
    }
    let _ = assert_adaptive_authority;

    let source = include_str!("resident_population_auto_sizing_receipt_v2.rs");
    assert!(source.contains("AdaptiveTailCapExceeded"));
    assert!(source.contains("ResidentAdaptiveBaseRequestV1::MIN_VIEW_ROWS_V1"));
    assert!(source.contains("RESIDENT_ADAPTIVE_BASE_SEMANTIC_V1"));
}

#[test]
fn v5_exposes_only_the_native_generation_zero_milestone() {
    use crate::prepared_discovery_run_input_v3::{
        ResidentGenerationZeroMilestoneV1,
        run_prepared_canonical_trendbar_research_generation_zero_v5,
    };

    fn assert_send<T: Send>() {}
    assert_send::<ResidentGenerationZeroMilestoneV1>();
    let _ =
        run_prepared_canonical_trendbar_research_generation_zero_v5::<fn(crate::DiscoveryProgress)>;

    let source = include_str!("prepared_discovery_run_input_v3.rs");
    assert!(source.contains("V5 requires one admitted native CUDA run"));
    assert!(!source.contains("PreparedCanonicalDiscoveryRunInputV5::Cpu"));
    assert!(!source.contains("config.evaluation_config"));
    assert!(source.contains("does not yet have a resident adaptive-threshold reduction"));
    assert!(source.contains("does not yet have a resident median-ATR gene-band reduction"));
    assert!(source.contains("clear_adaptive_threshold_ladder"));
    assert!(source.contains("clear_gene_stop_atr_scale"));
    assert!(source.contains("requires resident trim/remap before sizing"));
    assert!(source.contains("requires resident feature prefilter/remap before sizing"));
    assert!(source.contains("payoff floor is unreachable"));
    assert!(source.contains("does not yet carry the canonical discovery-ledger seed authority"));
}
