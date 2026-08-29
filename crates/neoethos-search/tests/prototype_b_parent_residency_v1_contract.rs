use neoethos_search::{ExactPopulationExecutionRunReceiptV2, NativePopulationResidencyReceiptV1};

const EVIDENCE: &str = include_str!("../src/population_execution_evidence_v1.rs");
const NATIVE_RUN: &str =
    include_str!("../src/population_execution_evidence_v1/native_cuda_resident_v1.rs");
const NATIVE_RECEIPT: &str = include_str!("../src/native_population_residency_receipt_v1.rs");
const RUN_RECEIPT_V2: &str = include_str!("../src/population_execution_run_receipt_v2.rs");
const ADAPTER: &str = include_str!("../src/gpu_native/prototype_b_population_eval.rs");
const FUNNEL: &str = include_str!("../src/funnel_profile.rs");
const DISCOVERY: &str = include_str!("../src/discovery.rs");
const LIB: &str = include_str!("../src/lib.rs");
const ENGINE_RECEIPT_V1: &str = include_str!("../src/population_engine_run_receipt_v1.rs");

fn source_between<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start = source
        .find(start)
        .unwrap_or_else(|| panic!("missing source boundary `{start}`"));
    let tail = &source[start..];
    let end = tail
        .find(end)
        .unwrap_or_else(|| panic!("missing source boundary `{end}` after `{start}`"));
    &tail[..end]
}

#[test]
fn receipt_types_are_public_serializable_outputs_not_publicly_constructible_authority() {
    fn assert_serializable<T: serde::Serialize>() {}
    assert_serializable::<NativePopulationResidencyReceiptV1>();
    assert_serializable::<ExactPopulationExecutionRunReceiptV2>();

    for source in [NATIVE_RECEIPT, RUN_RECEIPT_V2] {
        assert!(!source.contains("Deserialize"));
        assert!(!source.contains("impl Default"));
        assert!(!source.contains("pub fn new("));
        assert!(!source.contains("from_hash"));
        assert!(!source.contains("from_v1"));
        assert!(!source.contains("current("));
    }
}

#[test]
fn sealed_execution_run_owns_one_parent_native_session_without_global_registry() {
    assert!(EVIDENCE.contains("native_residency: NativePopulationResidencyRunV1"));
    assert!(EVIDENCE.contains("begin_native_population_residency_v1("));
    assert!(NATIVE_RUN.contains("parent_dataset_identity_sha256"));
    assert!(NATIVE_RUN.contains("Mutex<Option<"));
    assert!(!NATIVE_RUN.contains("static SLOT"));
    assert!(!NATIVE_RUN.contains("OnceLock"));
    let adapter_production = &ADAPTER[ADAPTER
        .find("\nfn evaluate_population_b_batch(")
        .expect("native batch boundary")..];
    assert!(!adapter_production.contains("fn resident_slot()"));
    assert!(!adapter_production.contains("active_residency_scopes"));
}

#[test]
fn native_parent_is_built_only_from_the_private_sealed_p1b_run() {
    assert!(EVIDENCE.contains("mod native_cuda_resident_v1;"));
    assert!(NATIVE_RUN.contains("pub(super) fn begin_native_population_residency_v1"));
    assert!(NATIVE_RUN.contains("SealedExactResidentDatasetParentV1"));
    assert!(NATIVE_RUN.contains("PopulationParentDatasetV1"));
    for forbidden in [
        "pub fn begin_native_population_residency_v1",
        "pub(crate) fn begin_native_population_residency_v1",
        "caller_parent_sha256",
        "sample_hash",
        "dataset_key",
    ] {
        assert!(
            !NATIVE_RUN.contains(forbidden),
            "forbidden authority `{forbidden}`"
        );
    }
}

#[test]
fn full_range_and_ordered_views_derive_only_from_the_sealed_authority() {
    for required in [
        "ExactResidentDatasetViewV1::Full",
        "ExactResidentDatasetViewV1::ContiguousRange",
        "ExactResidentDatasetViewV1::OrderedIndices",
        "PopulationEvaluationViewV1::full",
        "PopulationEvaluationViewV1::contiguous_range",
        "PopulationEvaluationViewV1::ordered_indices",
        "PopulationTimestampModeV1::DisabledIndexDelta",
    ] {
        assert!(
            NATIVE_RUN.contains(required),
            "missing sealed view route `{required}`"
        );
    }
    assert!(!NATIVE_RUN.contains("view_kind: u32"));
    assert!(!NATIVE_RUN.contains("indices_sha256"));
}

#[test]
fn ordered_native_view_never_materializes_or_reuploads_gathered_parent_arrays() {
    let ordered = source_between(
        EVIDENCE,
        "ExactResidentDatasetViewV1::OrderedIndices",
        "let timestamps = match timestamp_mode",
    );
    assert!(!ordered.contains("Array2::from_shape_fn"));
    assert!(!ordered.contains("Cow::Owned"));
    assert!(!ordered.contains("self.ohlcv.close[*index]"));

    let adapter_call = &ADAPTER[ADAPTER
        .find("fn evaluate_population_b_batch(")
        .expect("native batch boundary")..];
    for forbidden in [
        "close.to_vec()",
        "high.to_vec()",
        "low.to_vec()",
        "indicators.iter().copied().collect()",
        ".upload_dataset(PopulationDatasetView",
    ] {
        assert!(
            !adapter_call.contains(forbidden),
            "native route gathers `{forbidden}`"
        );
    }
    assert!(adapter_call.contains("bind_exact_native_population_view_v1"));
}

#[test]
fn native_success_records_exact_parent_view_and_all_intermediate_readback_counters() {
    for required in [
        "parent_upload_count",
        "parent_upload_bytes",
        "view_binding_count",
        "ordered_index_upload_bytes",
        "adaptive_upload_bytes",
        "stream_creation_count",
        "explicit_synchronization_count",
        "metric_rows_readback_count",
        "metric_rows_readback_rows",
        "metric_rows_readback_bytes",
        "diagnostic_readback_count",
        "diagnostic_readback_rows",
        "diagnostic_readback_bytes",
        "accepted_trade_total_readback_count",
        "accepted_trade_total_readback_bytes",
    ] {
        assert!(
            NATIVE_RECEIPT.contains(required),
            "receipt omits `{required}`"
        );
    }
    assert!(!NATIVE_RECEIPT.contains(concat!("compact_result_", "readback_count")));
    assert!(ADAPTER.contains("record_successful_native_population_v1"));
    assert!(ADAPTER.contains("read_residency_counters_v1"));
    assert!(ADAPTER.contains("require_exact_native_population_rows"));
    for required in [
        "selected_device_ordinal",
        "device_identity_sha256",
        "native_abi_version",
        "cuda_build_manifest_sha256",
        "neoethos.cuda-native-build.v1",
        "architectures",
        "gencode",
        "sass_targets",
        "ptx_targets",
        "nvcc_version",
        "cuobjdump_version",
        "required_sass_target",
    ] {
        assert!(
            NATIVE_RECEIPT.contains(required),
            "receipt omits native device/build binding `{required}`"
        );
    }
}

#[test]
fn v2_wraps_the_unchanged_v1_receipt_without_upcast_default_or_wire_mutation() {
    assert!(RUN_RECEIPT_V2.contains("engine_receipt_v1: PopulationEngineRunReceiptV1"));
    assert!(
        RUN_RECEIPT_V2
            .contains("native_residency_receipt_v1: Option<NativePopulationResidencyReceiptV1>")
    );
    assert!(RUN_RECEIPT_V2.contains("EXACT_POPULATION_EXECUTION_RUN_RECEIPT_SCHEMA_VERSION_V2"));
    assert!(RUN_RECEIPT_V2.contains("identity_sha256"));
    assert!(!RUN_RECEIPT_V2.contains("unwrap_or_default"));
    assert!(!RUN_RECEIPT_V2.contains("From<PopulationEngineRunReceiptV1>"));
    assert!(!ENGINE_RECEIPT_V1.contains("native_residency"));
}

#[test]
fn cuda_engine_requires_a_native_receipt_and_cpu_only_runs_cannot_forge_one() {
    assert!(RUN_RECEIPT_V2.contains("CudaNativeF64"));
    assert!(RUN_RECEIPT_V2.contains("MissingNativeResidencyReceipt"));
    assert!(RUN_RECEIPT_V2.contains("UnexpectedNativeResidencyReceipt"));
    assert!(NATIVE_RECEIPT.contains("NoSuccessfulNativePopulation"));
    assert!(NATIVE_RECEIPT.contains("ParentUploadCountMismatch"));
    assert!(NATIVE_RECEIPT.contains("ParentIdentityMismatch"));
    assert!(NATIVE_RECEIPT.contains("TransferAccountingMismatch"));
    assert!(NATIVE_RECEIPT.contains("InvalidCudaBuildManifest"));
    assert!(NATIVE_RECEIPT.contains("parent_upload_bytes() == 0"));
    assert!(NATIVE_RECEIPT.contains("checked_add"));
    assert!(NATIVE_RECEIPT.contains("view_binding_count() != exact_view_binding_total"));
    assert!(NATIVE_RECEIPT.contains("ordered_binding_count() == 0"));
    assert!(NATIVE_RECEIPT.contains("ordered_index_upload_bytes() == 0"));
    assert!(
        NATIVE_RECEIPT.contains(
            "successful_native_population_count != counters.metric_rows_readback_count()"
        )
    );
    assert!(NATIVE_RUN.contains("metric_readback_delta"));
}

#[test]
fn funnel_and_discovery_persist_the_full_v2_receipt_not_v1_or_logs() {
    assert!(FUNNEL.contains("ExactPopulationExecutionRunReceiptV2"));
    assert!(FUNNEL.contains("attach_population_execution_run_receipt_v2"));
    assert!(DISCOVERY.contains("pub population_execution_run_receipt_v2:"));
    assert!(DISCOVERY.contains("population_execution_run_receipt_v2.cloned()"));
    assert!(DISCOVERY.contains(".engine_receipt_v1().engines()"));
    assert!(!DISCOVERY.contains("pub population_engine_run_receipt:"));
    assert!(!FUNNEL.contains("attach_population_engine_run_receipt("));
}

#[test]
fn production_exports_only_immutable_receipts_and_has_no_fallback_or_fake_native_path() {
    for required in [
        "pub use native_population_residency_receipt_v1::NativePopulationResidencyReceiptV1",
        "pub use population_execution_run_receipt_v2::ExactPopulationExecutionRunReceiptV2",
    ] {
        assert!(
            LIB.contains(required),
            "missing immutable export `{required}`"
        );
    }
    for source in [NATIVE_RUN, ADAPTER, RUN_RECEIPT_V2] {
        for forbidden in [
            "RecomputeOnCpu",
            "FallbackDecision",
            "synthetic",
            "fake_permit",
            "std::env",
        ] {
            assert!(
                !source.contains(forbidden),
                "forbidden production path `{forbidden}`"
            );
        }
    }
}
