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

fn let_binding_for_call<'a>(source: &'a str, call_marker: &str) -> &'a str {
    let call_index = source
        .find(call_marker)
        .unwrap_or_else(|| panic!("missing call `{call_marker}`"));
    let let_index = source[..call_index]
        .rfind("let ")
        .unwrap_or_else(|| panic!("missing owning let binding for `{call_marker}`"));
    let mut binding = source[let_index + 4..call_index].trim_start();
    if let Some(rest) = binding.strip_prefix("mut ") {
        binding = rest;
    }
    let end = binding
        .find(|character: char| character.is_whitespace() || character == ':' || character == '=')
        .unwrap_or(binding.len());
    let binding = &binding[..end];
    assert!(
        !binding.is_empty(),
        "empty owning let binding for `{call_marker}`"
    );
    binding
}

fn parenthesized_call<'a>(source: &'a str, call_marker: &str) -> &'a str {
    let start = source
        .find(call_marker)
        .unwrap_or_else(|| panic!("missing call `{call_marker}`"));
    let open_offset = source[start..]
        .find('(')
        .unwrap_or_else(|| panic!("missing call opening parenthesis for `{call_marker}`"));
    let open = start + open_offset;
    let mut depth = 0_u64;
    for (offset, byte) in source.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("missing call closing parenthesis for `{call_marker}`")
}

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
        "begin_prepared_gpu_only_feature_two_pass_v2",
        "stream_score_batches_v2",
        "seal_selected_map_v2",
        "materialize_compact_selected_store_v2",
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
        "begin_prepared_gpu_only_feature_two_pass_v2",
        "stream_score_batches_v2",
        "seal_selected_map_v2",
        "preflight_canonical_native_generation_zero_result_v1",
        "prepare_prepared_canonical_trendbar_research_run_input_capped_v5",
        "materialize_compact_selected_store_v2",
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

    let seal_position = pipeline.find("seal_selected_map_v2(").unwrap();
    let selected_map_binding = let_binding_for_call(pipeline, "seal_selected_map_v2(");
    let selected_count_call = format!("{selected_map_binding}.selected_column_count()");
    let selected_count_position = pipeline.find(&selected_count_call).unwrap_or_else(|| {
        panic!("sealed receipt `{selected_map_binding}` never supplies its count")
    });
    let result_preflight_position = pipeline
        .find("preflight_canonical_native_generation_zero_result_v1(")
        .unwrap();
    assert!(
        seal_position < selected_count_position
            && selected_count_position < result_preflight_position,
        "selected-map seal and compact selected count must precede result preflight"
    );

    let result_preflight = parenthesized_call(
        pipeline,
        "preflight_canonical_native_generation_zero_result_v1(",
    );
    if !result_preflight.contains(&selected_count_call) {
        let compact_count_binding = let_binding_for_call(pipeline, &selected_count_call);
        assert!(
            result_preflight.contains(compact_count_binding),
            "result preflight must consume the compact selected count"
        );
    }

    let compact_materialization =
        parenthesized_call(pipeline, "materialize_compact_selected_store_v2(");
    assert!(
        compact_materialization.contains(selected_map_binding),
        "compact materialization must consume the exact sealed selected-map receipt"
    );
    for forbidden in [
        format!("&{selected_map_binding}"),
        format!("{selected_map_binding}.clone()"),
        format!("Arc::clone(&{selected_map_binding})"),
    ] {
        assert!(
            !compact_materialization.contains(&forbidden),
            "Search must move the sealed selected-map receipt by value, not `{forbidden}`"
        );
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
    assert!(
        !pipeline.contains("materialize_prepared_gpu_only_feature_store_for_data_population_v3"),
        "production native pipeline must not reopen the unfiltered resident-store materializer"
    );

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
