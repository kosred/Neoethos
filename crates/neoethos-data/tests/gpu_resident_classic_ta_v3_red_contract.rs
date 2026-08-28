use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("neoethos-data must live under the workspace root")
        .to_path_buf()
}

fn read(path: impl AsRef<Path>) -> String {
    fs::read_to_string(workspace_root().join(path)).unwrap_or_default()
}

#[test]
fn classic_ta_is_now_real_and_the_remaining_capability_census_stays_fail_closed() {
    let contracts = read("crates/neoethos-gpu-contracts/src/resident_feature_store_v3.rs");
    let data = read("crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs");
    let executor = read("crates/neoethos-gpu-cuda/src/resident_classic_ta_v3.rs");

    assert!(contracts.contains("Self::ClassicTa"));
    assert!(contracts.contains("Self::Smc"));
    assert!(executor.contains("ResidentFeatureProducerV3::ClassicTa"));
    assert!(data.contains("resident_classic_ta_capability_v3()?"));
    assert!(data.contains("resident_smc_capability_v3()?"));
    assert!(data.contains("EXPECTED_MISSING_AFTER_REAL_RESIDENT_PRODUCERS_V3"));
}

#[test]
fn classic_ta_recipe_preflight_projects_the_frozen_run_plan_before_workspace_admission() {
    let hpc = read("crates/neoethos-data/src/core/hpc_ta.rs");
    let data = read("crates/neoethos-data/src/core/gpu_resident_classic_ta_v3.rs");

    for token in [
        "ClassicTaResidentPlanProjectionV3",
        "ClassicTaRunPlan",
        "admitted_indicator_ids",
        "extended_groups",
        "available_bytes_at_admission",
        "route_plan_sha256",
        "working_set",
    ] {
        assert!(
            hpc.contains(token) || data.contains(token),
            "Classic TA resident preflight omitted `{token}`"
        );
    }
    for token in [
        "preflight_resident_classic_ta_v3",
        "before_full_workspace_admission",
        "no_second_budget_probe",
        "FeatureCellValidity::Warmup.code()",
        "FeatureCellValidity::ComputeFailure.code()",
    ] {
        assert!(
            data.contains(token),
            "Classic TA phase-one authority omitted `{token}`"
        );
    }
}

#[test]
fn classic_ta_borrows_the_one_shot_context_stream_and_parent_uploads() {
    let classic = read("crates/neoethos-data/src/core/gpu_resident_classic_ta_v3.rs");
    let executor = read("crates/neoethos-gpu-cuda/src/resident_classic_ta_v3.rs");
    let vector_runtime = read("vendor/vector-ta-0.2.9-patched/src/cuda/runtime.rs");

    assert!(vector_runtime.contains("pub fn from_parts("));
    for token in [
        "GpuOnlyRunDeviceAdmissionV3",
        "producer_ready_event",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "timestamps",
        "hlc3",
        "hl2",
        "hlcc4",
    ] {
        assert!(
            classic.contains(token) || executor.contains(token),
            "Classic TA resident input authority omitted `{token}`"
        );
    }
    assert!(!classic.contains("GpuIndicatorEngine::new(ohlcv, 0)"));
    assert!(!classic.contains("upload_ohlcv_f64"));
    assert!(!classic.contains("Context::new"));
    assert!(!classic.contains("Stream::new"));
}

#[test]
fn classic_ta_streams_at_most_64_outputs_with_exact_device_validity_and_no_download() {
    let classic = read("crates/neoethos-data/src/core/gpu_resident_classic_ta_v3.rs");
    let executor = read("crates/neoethos-gpu-cuda/src/resident_classic_ta_v3.rs");
    let runtime = read("crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    let native = read("crates/neoethos-gpu-cuda/native/resident_classic_ta_v3.cu");
    let legacy = read("crates/neoethos-data/src/core/classic_cuda_plan.rs");
    let named = read("vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");

    for token in [
        "MAX_RESIDENT_PRODUCER_BATCH_COLUMNS_V3: usize = 64",
        "ResidentFeatureProducerV3::ClassicTa",
    ] {
        assert!(
            classic.contains(token),
            "Classic TA bounded resident batch authority omitted `{token}`"
        );
    }
    for token in [
        "ResidentF64FeatureBatchV3",
        "append_resident_classic_ta_recipe_v3",
        "neoethos_resident_classic_validity_u8_v3",
        "kWarmupV3",
        "kNonFiniteV3",
        "0xffU",
        "resident_classic_ta_capability_v3",
    ] {
        assert!(
            executor.contains(token) || runtime.contains(token) || native.contains(token),
            "Classic TA device executor omitted `{token}`"
        );
    }
    assert!(named.contains("into_resident_parts_v3"));
    assert!(!classic.contains("download_primary_output_f64"));
    assert!(!classic.contains("download_named_outputs_f64"));
    assert!(!classic.contains("synchronize()"));
    assert!(legacy.contains("download_primary_output_f64"));
    assert!(legacy.contains("download_named_outputs_f64"));
}

#[test]
fn classic_ta_vendor_lifecycle_uses_explicit_cust_types_and_frozen_edition_syntax() {
    let wrapper = read("vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");

    assert!(wrapper.contains("use cust::error::CudaResult;"));
    assert!(wrapper.contains("DeviceBuffer, DeviceCopy, LockedBuffer"));
    assert!(wrapper.contains("if let Some(maximum) = kernel.max_period() {"));
    assert!(wrapper.contains("if let Some(too_large) = periods"));
    assert!(
        !wrapper.contains("&& let Some(too_large)"),
        "the vendored edition must not depend on Rust 2024 let-chain syntax"
    );
    assert_eq!(
        wrapper
            .matches("pub const fn device_id(&self) -> u32")
            .count(),
        1,
        "only the direct scalar field getter may remain const"
    );
    assert_eq!(
        wrapper.matches("pub fn device_id(&self) -> u32").count(),
        2,
        "the Arc-backed named-parts getter must remain visible without an invalid const qualifier"
    );
}

#[test]
fn classic_preflight_emits_only_move_only_local_route_drafts() {
    let data =
        read("crates/neoethos-data/src/core/gpu_resident_classic_ta_v3.rs").replace("\r\n", "\n");
    let recipe = read("crates/neoethos-data/src/core/gpu_resident_feature_recipe_v4.rs")
        .replace("\r\n", "\n");

    for required in [
        "ResidentClassicTaLocalDraftV4",
        "ResidentRouteDraftV4",
        "ResidentCanonicalParameterV4",
        "RESIDENT_CLASSIC_TA_LOCAL_ROUTE_DOMAIN_V4",
        "local_destination_column",
        "local_output_identity_sha256",
    ] {
        assert!(
            data.contains(required),
            "Classic local-draft preflight omitted {required:?}"
        );
    }
    let preflight = data
        .split_once("pub(crate) fn preflight_resident_classic_ta_v3(")
        .expect("Classic preflight")
        .1;
    for forbidden in [
        "ResidentFeatureRouteV3::new",
        "route_receipt_sha256(",
        "let ordinal =",
        "route_id =",
    ] {
        assert!(
            !preflight.contains(forbidden),
            "Classic preflight still mints global authority through {forbidden:?}"
        );
    }
    assert!(!data.contains("use neoethos_gpu_contracts::resident_feature_store_v3::{\n    ResidentFeatureProducerV3, ResidentFeatureRouteV3,"));
    for move_only in [
        "ResidentRouteDraftV4",
        "ResidentProducerBatchDraftV4",
        "ResidentProducerDraftV4",
    ] {
        assert!(recipe.contains(move_only));
        assert!(
            !recipe.contains(&format!("impl Clone for {move_only}")),
            "recipe authority became cloneable"
        );
    }
}

/// Primary and warmup launches now share one exact owner plan. This remains the
/// compile-safe full-component RED until every named multi-output VectorTA
/// owner exposes the same pure all-output/parameter/scratch allocation plan.
#[test]
#[ignore = "RED: every named VectorTA owner must expose exact all-output allocation plans"]
fn red_classic_memory_receipt_comes_from_the_runtime_allocation_owner() {
    let runtime = read("crates/neoethos-gpu-cuda/src/resident_classic_ta_v3.rs");
    let vector = read("vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");

    for required in [
        "ResidentClassicTaPreDeviceMemoryReceiptV4",
        "preflight_resident_classic_ta_memory_v4",
        "F64ResidentSingleSweepAllocationPlanV4",
        "preflight_resident_single_sweep_allocation_v4",
        "selected_value_bytes",
        "all_output_retained_bytes",
        "additional_retained_bytes",
        "retained_scratch_bytes",
        "validity_scratch_bytes",
        "derived_input_bytes",
        "ready_event_count",
    ] {
        assert!(
            runtime.contains(required) || vector.contains(required),
            "exact pre-device Classic memory authority omitted {required:?}"
        );
    }
    assert!(runtime.contains("runtime_memory_receipt_v4 != pre_device_memory_receipt_v4"));
    assert!(vector.contains("preflight_resident_named_allocation_v4"));
    assert!(runtime.contains("all_named_output_ownership_bytes"));
}
