use neoethos_search::{
    CanonicalNativeCancellationTokenV1, CanonicalNativeDiscoveryExecutionErrorCodeV1,
    CanonicalNativeDiscoveryExecutionStageV1, PublishedCanonicalNativeGenerationZeroResearchV1,
    run_canonical_native_discovery_generation_zero_from_ref_v1,
};
#[cfg(all(target_os = "linux", not(feature = "gpu-cuda")))]
use neoethos_search::{
    CanonicalNativeGenerationZeroOverridesV1, CanonicalResearchContractArtifactRefV1,
    install_and_seal_canonical_native_runtime_authority_v1,
};

#[test]
fn canonical_native_executor_public_api_exists() {
    let _ = std::mem::size_of::<CanonicalNativeCancellationTokenV1>();
    let _ = std::mem::size_of::<CanonicalNativeDiscoveryExecutionErrorCodeV1>();
    let _ = std::mem::size_of::<CanonicalNativeDiscoveryExecutionStageV1>();
    let _ = std::mem::size_of::<PublishedCanonicalNativeGenerationZeroResearchV1>();
    let _ = std::any::type_name_of_val(
        &run_canonical_native_discovery_generation_zero_from_ref_v1::<
            fn(neoethos_search::DiscoveryProgress),
        >,
    );
}

#[test]
fn cancellation_is_cloneable_monotonic_and_observable() {
    let first = CanonicalNativeCancellationTokenV1::new();
    let second = first.clone();
    assert!(!first.is_cancelled());
    second.cancel();
    assert!(first.is_cancelled());
    first.cancel();
    assert!(second.is_cancelled());
}

#[test]
fn public_failure_taxonomy_names_every_executor_boundary() {
    use CanonicalNativeDiscoveryExecutionErrorCodeV1 as Code;
    use CanonicalNativeDiscoveryExecutionStageV1 as Stage;

    let stages = [
        Stage::NativeCapabilityGate,
        Stage::RuntimeInstallReceipt,
        Stage::SearchGpuExecutionLease,
        Stage::ContractReferenceValidation,
        Stage::ContractArtifactRead,
        Stage::ContractArtifactHash,
        Stage::ContractSchemaValidation,
        Stage::ExactSourcePin,
        Stage::NativePreflight,
        Stage::NativeAdmission,
        Stage::ResidentDataMaterialization,
        Stage::NativeReceiptBinding,
        Stage::GenerationZeroEvaluation,
        Stage::ConsumerCompletion,
        Stage::ResultPublication,
    ];
    let codes = [
        Code::UnsupportedPlatform,
        Code::NativeCudaRequired,
        Code::Cancelled,
        Code::InvalidRequest,
        Code::RuntimeAuthorityInvalid,
        Code::ArtifactUnavailable,
        Code::ArtifactHashMismatch,
        Code::ContractInvalid,
        Code::ExactGenerationConflict,
        Code::PreflightRejected,
        Code::AdmissionRejected,
        Code::MaterializationRejected,
        Code::ReceiptRejected,
        Code::EvaluationRejected,
        Code::CompletionRejected,
        Code::ResultSealingRejected,
        Code::PublicationRejected,
    ];
    assert_eq!(stages.len(), 15);
    assert_eq!(codes.len(), 17);
}

#[test]
fn source_contract_is_one_staged_native_pipeline_without_population_clone_or_reopen() {
    let source = include_str!("../src/canonical_native_discovery_run_v1.rs");
    let result_source = include_str!("../src/canonical_native_generation_zero_result_v1.rs");
    let pipeline = source
        .split_once("fn run_canonical_native_discovery_generation_zero_cuda_v1")
        .expect("missing CUDA executor helper")
        .1;
    let common_boundary = source
        .split_once("pub fn run_canonical_native_discovery_generation_zero_from_ref_v1")
        .expect("missing public executor boundary")
        .1
        .split_once("struct ExecutorCancellationMarkerV1")
        .expect("missing CUDA cancellation marker")
        .0;
    let unsupported = common_boundary
        .find("#[cfg(not(target_os = \"linux\"))]")
        .unwrap();
    let no_cuda = common_boundary
        .find("#[cfg(all(target_os = \"linux\", not(feature = \"gpu-cuda\")))]")
        .unwrap();
    let cuda = common_boundary
        .find("#[cfg(all(target_os = \"linux\", feature = \"gpu-cuda\"))]")
        .unwrap();
    assert!(unsupported < no_cuda && no_cuda < cuda);
    let unsupported_branch = &common_boundary[unsupported..no_cuda];
    assert!(unsupported_branch.contains("UnsupportedPlatform"));
    for forbidden in [
        "resolve_canonical_native_discovery_request_v1",
        "canonical_root",
        "data_dir",
    ] {
        assert!(!unsupported_branch.contains(forbidden));
    }
    for required in [
        "resolve_canonical_native_discovery_request_v1",
        "pin_exact_canonical_series_v1",
        "preflight_gpu_only_feature_workspace_v3",
        "prepare_gpu_only_feature_materialization_v3",
        "preflight_canonical_native_generation_zero_result_v1",
        "prepare_prepared_canonical_trendbar_research_run_input_capped_v5",
        "materialize_prepared_gpu_only_feature_store_for_data_population_v3",
        "CanonicalGpuResidentSearchInputReceiptV3::from_resident_store",
        "run_prepared_canonical_trendbar_research_generation_zero_gated_typed_v5",
        "seal_canonical_native_generation_zero_research_result_v1",
        "publish_canonical_native_generation_zero_research_result_v1",
    ] {
        assert!(
            pipeline.contains(required),
            "missing production stage `{required}`"
        );
    }

    let ordered = [
        "pin_exact_canonical_series_v1",
        "preflight_gpu_only_feature_workspace_v3",
        "prepare_gpu_only_feature_materialization_v3",
        "preflight_canonical_native_generation_zero_result_v1",
        "prepare_prepared_canonical_trendbar_research_run_input_capped_v5",
        "run_prepared_canonical_trendbar_research_generation_zero_gated_typed_v5",
        "seal_canonical_native_generation_zero_research_result_v1",
        "publish_canonical_native_generation_zero_research_result_v1",
    ];
    let mut previous = 0;
    for marker in ordered {
        let position = pipeline.find(marker).unwrap();
        assert!(position >= previous, "stage `{marker}` is out of order");
        previous = position;
    }

    for forbidden in [
        "run_batch",
        "FallbackPolicy",
        "EvaluationBackend::CPU",
        "DevicePreference::Auto",
        "fs::read",
        "SearchResult::clone",
        "search_result().clone",
        "genes.to_vec",
        "metrics.to_vec",
        "run_prepared_canonical_trendbar_research_with_holdout",
    ] {
        assert!(
            !source.contains(forbidden),
            "forbidden executor path `{forbidden}`"
        );
    }

    assert!(pipeline.matches("probe_cancellation_v1(").count() >= 6);
    assert!(pipeline.contains("ExecutorCancellationMarkerV1"));
    assert!(
        pipeline.contains("native_config.max_indicators = preflight.resolved_max_indicators()")
    );
    assert!(!pipeline.contains("native_config.population ="));
    assert!(!pipeline.contains("native_config.population_auto ="));
    assert!(
        result_source
            .contains("preflight.resolved_max_indicators() == sizing.requested_max_indicators()")
    );
    assert!(!result_source.contains(
        "preflight.raw_configured_max_indicators() == sizing.requested_max_indicators()"
    ));
}

#[cfg(all(target_os = "linux", not(feature = "gpu-cuda")))]
#[test]
fn default_linux_gate_wins_before_cancelled_or_unreadable_artifact() {
    let settings = neoethos_core::Settings::default();
    let install = install_and_seal_canonical_native_runtime_authority_v1(&settings).unwrap();
    let reference = CanonicalResearchContractArtifactRefV1::checked_new(
        "research/contracts/does-not-exist.json",
        "1".repeat(64),
    )
    .unwrap();
    let cancellation = CanonicalNativeCancellationTokenV1::new();
    cancellation.cancel();
    let error = run_canonical_native_discovery_generation_zero_from_ref_v1(
        &settings,
        &install,
        reference,
        CanonicalNativeGenerationZeroOverridesV1::default(),
        &cancellation,
        |_| {},
    )
    .unwrap_err();
    assert_eq!(
        error.stage(),
        CanonicalNativeDiscoveryExecutionStageV1::NativeCapabilityGate
    );
    assert_eq!(
        error.code(),
        CanonicalNativeDiscoveryExecutionErrorCodeV1::NativeCudaRequired
    );
}
