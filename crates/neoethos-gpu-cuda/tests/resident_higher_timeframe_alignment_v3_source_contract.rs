use std::fs;
use std::path::{Path, PathBuf};

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("neoethos-gpu-cuda lives under crates/")
        .to_path_buf()
}

fn read(path: &str) -> String {
    fs::read_to_string(repository_root().join(path))
        .unwrap_or_else(|error| panic!("read {path}: {error}"))
}

fn compact(source: &str) -> String {
    source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

#[test]
fn htf_v3_device_availability_uses_a_device_safe_saturating_bound() {
    let cuda = read("crates/neoethos-gpu-cuda/native/resident_higher_timeframe_alignment_v3.cu");
    let helper_start = cuda
        .find("__device__ __forceinline__ bool availability_at_v3")
        .expect("HTF CUDA must define the fixed/calendar availability helper");
    let helper_end = cuda[helper_start..]
        .find("__device__ __forceinline__ bool latest_available_parent_row_v3")
        .map(|offset| helper_start + offset)
        .expect("HTF CUDA must delimit the availability helper");
    let helper = &cuda[helper_start..helper_end];

    assert!(
        !helper.contains("std::numeric_limits"),
        "device availability must not depend on host-only std::numeric_limits"
    );
    assert!(
        helper.contains("INT64_MAX - segment.fixed_period_ms"),
        "fixed availability must retain its explicit saturating int64 bound"
    );
}

#[test]
fn htf_v3_data_owner_seals_host_recipe_before_device_then_binds_exact_capture() {
    let data = read("crates/neoethos-data/src/core/gpu_resident_higher_timeframe_alignment_v3.rs");

    for required in [
        "ResidentHigherTimeframeHostParentRecipeV3",
        "ResidentHigherTimeframeParentBatchMemoryFormulaV3",
        "prepare_resident_higher_timeframe_host_parent_v3",
        "prepare_resident_higher_timeframe_direct_parent_owner_v3",
        "PreparedResidentHigherTimeframeDirectParentCaptureTemplateV3",
        "PendingResidentHigherTimeframeDirectParentCaptureLaunchV3",
        "ValidatedResidentHigherTimeframeDirectParentCaptureV3",
        "ResidentColumnSchemaAssemblerV4::default()",
        "ResidentFeatureColumnBindingV3::from_admitted_route",
        "into_smc_preparation_parts_v3",
        "capture_direct_parent_v3",
        "expected_retained_parent_device_bytes",
        "parent_open_ms",
        "ResidentHigherTimeframeGlobalBatchV3",
        "ResidentHigherTimeframeGlobalParentSegmentV3",
        "build_global_batches_v3",
        "MAX_RESIDENT_PRODUCER_BATCH_COLUMNS_V4",
        "parent_segments",
        "native_kernel_launch_count",
        "pointer_table_h2d_bytes",
        "PendingResidentHigherTimeframeRuntimeV3",
        "PreparedResidentHigherTimeframeAppendV3",
        "bind_captured_parents_v3",
        "bind_captured_parent_v3",
        "capture.retained_device_bytes()",
        "capture.quant_runtime_receipt()",
        "capture.session_runtime_receipt()",
        "capture.regime_runtime_receipt()",
        "capture.footprint_runtime_receipt()",
        "bind_current_native_v3",
        "validate_native_receipt",
        "assembler.append_resident_higher_timeframe_alignment_v3",
    ] {
        assert!(
            data.contains(required),
            "Data HTF host/native split omitted `{required}`"
        );
    }

    let preflight_start = data
        .find("pub(crate) fn preflight_resident_higher_timeframe_alignment_v3")
        .expect("HTF host-only preflight must exist");
    let preflight_signature_end = data[preflight_start..]
        .find(") -> Result<")
        .map(|offset| preflight_start + offset)
        .expect("HTF host-only preflight signature must be bounded");
    let preflight_signature = &data[preflight_start..preflight_signature_end];
    assert!(preflight_signature.contains("Vec<ResidentHigherTimeframeHostParentRecipeV3>"));
    for forbidden in [
        "GpuOnlyRunDeviceAdmissionV3",
        "PendingResidentHigherTimeframeDirectParentCaptureV3",
        "parent_context_process_token",
        "parent_stream_process_token",
    ] {
        assert!(
            !preflight_signature.contains(forbidden),
            "host-only HTF preflight illegally depends on `{forbidden}`"
        );
    }
    let preflight_end = data[preflight_start..]
        .find("fn validate_canonical_cpu_route_order_v3")
        .map(|offset| preflight_start + offset)
        .expect("HTF host-only preflight must end before route validation helpers");
    let preflight = &data[preflight_start..preflight_end];
    for forbidden in [
        "run_device",
        "capture",
        "base_context_process_token",
        "base_stream_process_token",
    ] {
        assert!(
            !preflight.contains(forbidden),
            "HTF recipe draft depends on post-device `{forbidden}`"
        );
    }
    let input_identity_start = data
        .find("fn htf_input_identity_v3")
        .expect("HTF host input identity must exist");
    let input_identity = &data[input_identity_start..];
    assert!(input_identity.contains("expected_retained_parent_device_bytes"));
    assert!(!input_identity.contains("parent_context_process_token"));
    assert!(!input_identity.contains("parent_stream_process_token"));

    let owner_helper_start = data
        .find("pub(crate) fn prepare_resident_higher_timeframe_direct_parent_owner_v3")
        .expect("HTF direct-parent owner helper must exist");
    let owner_helper_signature_end = data[owner_helper_start..]
        .find(") -> Result<")
        .map(|offset| owner_helper_start + offset)
        .expect("HTF direct-parent owner helper signature must be bounded");
    let owner_helper_signature = &data[owner_helper_start..owner_helper_signature_end];
    assert!(owner_helper_signature.contains("&MaterializedPinnedResidentCanonicalSourceV1"));
    for forbidden in [
        "source_binding_sha256",
        "route_receipt_sha256",
        "canonical_parameter_tuple_sha256",
        "global_ordinal",
    ] {
        assert!(
            !owner_helper_signature.contains(forbidden),
            "HTF owner helper accepts caller-mintable `{forbidden}`"
        );
    }

    assert!(!data.contains("copy_to("));
    assert!(!data.contains("synchronize()"));
    assert!(!data.contains("HTF_ROUTE_COUNT"));
    assert!(!data.contains("101"));
}

#[test]
fn htf_v3_opaque_direct_parent_capture_moves_existing_batches_without_pack_or_copy() {
    let native = read("crates/neoethos-gpu-cuda/src/resident_higher_timeframe_alignment_v3.rs");
    let smc = read("crates/neoethos-gpu-cuda/src/resident_smc_v3.rs");
    let classic = read("crates/neoethos-gpu-cuda/src/resident_classic_ta_v3.rs");
    let data = read("crates/neoethos-data/src/core/gpu_resident_higher_timeframe_alignment_v3.rs");

    for required in [
        "ResidentHigherTimeframeDirectParentLaunchPlanV3",
        "PendingResidentHigherTimeframeDirectParentCaptureV3",
        "ResidentHigherTimeframeCapturedRouteV3",
        "capture_resident_higher_timeframe_direct_parent_v3",
        "ResidentFeatureProducerV3::Smc",
        "ResidentFeatureProducerV3::ClassicTa",
        "ResidentFeatureProducerV3::Quant",
        "ResidentFeatureProducerV3::Session",
        "ResidentFeatureProducerV3::Regime",
        "ResidentFeatureProducerV3::Footprint",
        "launch_resident_quant_v3",
        "launch_resident_session_v2",
        "launch_resident_regime_v3",
        "launch_resident_footprint_v2",
        "ResidentClassicTaExecutorV3::new_v4",
        "next_pending_batch_v3",
        "route_descriptors",
        "retained_device_bytes",
        "quant_runtime_receipt",
        "session_runtime_receipt",
        "regime_runtime_receipt",
        "footprint_runtime_receipt",
        "into_direct_parent",
        "std::mem::forget(producer_batches)",
        "std::mem::forget(parent_source)",
    ] {
        assert!(
            native.contains(required),
            "opaque HTF parent capture omitted `{required}`"
        );
    }
    assert!(smc.contains("into_higher_timeframe_parent_parts_v3"));
    assert!(classic.contains("detach_shared_derived_input_charge_v3"));
    assert!(data.contains("bind_captured_parent_v3"));
    assert!(data.contains("PendingResidentHigherTimeframeDirectParentCaptureV3"));

    let capture_start = native
        .find("pub fn capture_resident_higher_timeframe_direct_parent_v3")
        .expect("HTF capture entrypoint must exist");
    let capture_end = native[capture_start..]
        .find("impl Drop for PendingResidentHigherTimeframeDirectParentCaptureV3")
        .map(|offset| capture_start + offset)
        .expect("HTF capture entrypoint must end at its move-only drop guard");
    let capture = &native[capture_start..capture_end];
    let mut cursor = 0;
    for producer in [
        "ResidentFeatureProducerV3::Smc",
        "ResidentFeatureProducerV3::ClassicTa",
        "ResidentFeatureProducerV3::Quant",
        "ResidentFeatureProducerV3::Session",
        "ResidentFeatureProducerV3::Regime",
        "ResidentFeatureProducerV3::Footprint",
    ] {
        let next = capture[cursor..]
            .find(producer)
            .map(|offset| cursor + offset)
            .unwrap_or_else(|| panic!("capture omitted canonical producer `{producer}`"));
        assert!(next >= cursor, "capture producer order regressed");
        cursor = next + producer.len();
    }
    for forbidden in ["append_batch", "copy_to(", "synchronize()"] {
        assert!(
            !capture.contains(forbidden),
            "direct-parent capture must not use `{forbidden}`"
        );
    }
}

#[test]
fn htf_v3_native_abi_is_variable_width_and_owns_direct_parent_carriers() {
    let rust = read("crates/neoethos-gpu-cuda/src/resident_higher_timeframe_alignment_v3.rs");
    let abi =
        read("crates/neoethos-gpu-cuda/native/resident_higher_timeframe_alignment_v3_abi.cuh");
    let cuda = read("crates/neoethos-gpu-cuda/native/resident_higher_timeframe_alignment_v3.cu");

    for required in [
        "ResidentHigherTimeframeDirectParentV3",
        "ResidentHigherTimeframeDirectParentConstructionGuardV3",
        "Box<dyn ResidentParentDatasetSourceV3>",
        "Vec<Box<dyn ResidentF64FeatureBatchV3>>",
        "parent_feature_column_count",
        "source_value_buffers_device",
        "source_validity_buffers_device",
        "source_value_offsets_device",
        "source_validity_offsets_device",
        "ResidentHigherTimeframeLaunchAuthorityV3",
        "ResidentHigherTimeframeRuntimeReceiptV3",
        "feature_value_d2h_bytes: 0",
        "feature_validity_d2h_bytes: 0",
        "parent_feature_h2d_bytes: 0",
        "host_synchronize_count: 0",
        "producer_ready_event_synchronize_count: 0",
        "logical_validity_codes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]",
        "ResidentFeatureProducerV3::HigherTimeframeAlignment",
        "RESIDENT_HTF_IMPLEMENTATION_ID_V3",
        "RESIDENT_HTF_EXACT_MATH_AUTHORITY_V3",
    ] {
        assert!(
            rust.contains(required),
            "native HTF Rust omitted `{required}`"
        );
    }

    for required in [
        "NeoResidentHigherTimeframeLaunchV3",
        "NeoResidentHigherTimeframeParentSegmentV3",
        "base_row_count",
        "feature_column_count",
        "parent_segment_count",
        "parent_segments_host",
        "first_column",
        "column_count",
        "parent_row_count",
        "availability_rule",
        "fixed_period_ms",
        "max_age_ms",
        "base_open_ms",
        "parent_open_ms",
        "source_value_buffers_device",
        "source_validity_buffers_device",
        "source_value_offsets_device",
        "source_validity_offsets_device",
        "feature_values",
        "feature_validity_u8",
        "neoethos_resident_higher_timeframe_alignment_f64_v3",
    ] {
        assert!(
            abi.contains(required),
            "native HTF ABI omitted `{required}`"
        );
    }

    for required in [
        "kAlignmentMissingV3 = 9U",
        "kStaleV3 = 4U",
        "canonical_nan_v3",
        "available_at_ms <= base_timestamp_ms",
        "age_ms > segment.max_age_ms",
        "source_validity <= kAlignmentMissingV3",
        "source_validity == kValidV3",
        "parent_open_ms[parent_row + 1U]",
        "open_ms + segment.fixed_period_ms",
        "segment.first_column + local_column",
        "segment.first_column != next_first_column",
        "segment.column_count",
        "resident_higher_timeframe_alignment_f64_v3",
    ] {
        assert!(
            cuda.contains(required),
            "native HTF CUDA omitted `{required}`"
        );
    }
    let resolve_parent = cuda
        .find("const bool parent_resolved")
        .expect("HTF CUDA must resolve one parent row per base-row segment");
    let gather_columns = cuda
        .find("for (std::uint64_t local_column")
        .expect("HTF CUDA must gather the variable-width segment");
    assert!(
        resolve_parent < gather_columns,
        "HTF CUDA repeated causal binary search inside the feature-column loop"
    );

    let compact_rust = compact(&rust);
    assert!(compact_rust.contains("POINTER_FIELDS_PER_COLUMN_V3:usize=4"));
    assert!(compact_rust.contains("max_batch_columns.checked_mul(POINTER_FIELDS_PER_COLUMN_V3)"));
    assert!(compact_rust.contains("pointer_table_entries.checked_mul(U64_BYTES_V3)"));
    assert!(compact_rust.contains("self.base_row_count.checked_mul(columns)"));
    assert!(
        compact_rust.contains("cells.checked_mul(F64_BYTES_V3+LOGICAL_VALIDITY_BYTES_PER_CELL_V3)")
    );
    assert!(!rust.contains("HTF_ROUTE_COUNT"));
    assert!(!rust.contains("101"));
    assert!(!rust.contains("copy_to("));
    assert!(!rust.contains("synchronize()"));
    assert!(!rust.contains("#[derive(Clone"));
    assert!(!rust.contains("#[derive(Copy"));
    assert!(rust.contains("column_sources"));
    assert!(rust.contains("parent_segments"));
    assert!(rust.contains("native_kernel_launch_count"));
    assert!(rust.contains("std::mem::forget(feature_batches)"));
    assert!(rust.contains("std::mem::forget(parent_source)"));
}

#[test]
fn htf_v3_launch_binds_shape_route_order_identity_and_exact_allocation_receipts() {
    let rust = read("crates/neoethos-gpu-cuda/src/resident_higher_timeframe_alignment_v3.rs");
    let data_owner =
        read("crates/neoethos-data/src/core/gpu_resident_higher_timeframe_alignment_v3.rs");
    for required in [
        "selected_parent_order",
        "canonical_cpu_producer_order",
        "source_route_receipt_sha256",
        "output_route_receipt_sha256",
        "input_identity_sha256",
        "source_binding_sha256",
        "parent_store_identity_sha256",
        "parent_context_process_token",
        "parent_stream_process_token",
        "base_context_process_token",
        "base_stream_process_token",
        "retained_parent_device_bytes",
        "retained_feature_device_bytes",
        "pointer_table_device_bytes",
        "pointer_table_h2d_bytes",
        "native_launch_count",
        "producer_ready_event_count",
        "availability_rule",
        "fixed_open_plus_period_v1",
        "next_direct_bar_open_v1",
        "forward_fill",
        "canonical_qnan_bits: 0x7ff8_0000_0000_0000",
    ] {
        assert!(rust.contains(required), "HTF receipt omitted `{required}`");
    }
    assert!(data_owner.contains("period_ms.saturating_mul(2)"));
    assert!(!data_owner.contains("HTF fixed max-age overflow"));
}

#[test]
fn htf_v3_runtime_peak_includes_the_live_direct_parent_graph() {
    let htf = read("crates/neoethos-gpu-cuda/src/resident_higher_timeframe_alignment_v3.rs");
    let store = read("crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    let compact_htf = compact(&htf);
    let method_start = store
        .find("pub fn append_resident_higher_timeframe_alignment_v3")
        .expect("feature store must expose its crate-owned HTF append method");
    let method_end = store[method_start..]
        .find("/// Launch and append the atomic twenty-three-column Session-v2")
        .map(|offset| method_start + offset)
        .expect("HTF append method must end before Session-v2");
    let compact_append = compact(&store[method_start..method_end]);
    let compact_store = compact(&store);

    assert!(compact_htf.contains(
        "pub(crate)constfnretained_parent_device_bytes(&self)->usize{self.retained_parent_device_bytes}"
    ));
    assert!(
        compact_append
            .contains("letexternal_live_device_bytes=executor.retained_parent_device_bytes();")
    );
    assert!(compact_append.contains("self.append_batch_with_external_live_bytes("));
    assert!(compact_append.contains("Box::new(batch),external_live_device_bytes,"));
    assert!(
        compact_store
            .contains("batch.retained_device_bytes().checked_add(external_live_device_bytes)")
    );
    assert!(compact_store.contains(
        "self.max_live_producer_bytes=self.max_live_producer_bytes.max(live_device_bytes)"
    ));
}

#[test]
fn htf_v3_real_device_gate_covers_cross_parent_values_and_all_validity_codes() {
    let fixture = read(
        "crates/neoethos-gpu-cuda/src/resident_higher_timeframe_alignment_v3_device_fixture.rs",
    );
    for required in [
        "#![cfg(feature = \"cuda-device-fixtures\")]",
        "resident_higher_timeframe_v3_device_route_value_and_validity_parity",
        "for validity_code in 0_u8..=9",
        "parent_segment_count: parent_segments.len() as u32",
        "availability_rule: 1",
        "availability_rule: 2",
        "fixed_period_ms: 5",
        "max_age_ms: 10",
        "fixed_period_ms: 0",
        "max_age_ms: -1",
        "2_002.0_f64.to_bits()",
        "3_002.0_f64.to_bits()",
        "stream.synchronize()?",
        "device_values.copy_to(&mut values)?",
        "device_validity.copy_to(&mut validity_u8)?",
    ] {
        assert!(
            fixture.contains(required),
            "HTF RTX fixture omitted `{required}`"
        );
    }
    assert!(fixture.contains("Production has no aligned feature/validity D2H path"));
    let release = read(
        "crates/neoethos-gpu-cuda/tests/fixtures/resident_higher_timeframe_alignment_v3.release.txt",
    );
    for required in [
        "verified=false",
        "feature_column_count=live_recipe_v4_flattened_parent_schema",
        "real_device_route_value_validity_parity=not_run",
        "compute_sanitizer_memcheck=not_run",
        "capability_registered=false",
    ] {
        assert!(
            release.contains(required),
            "HTF release guard omitted `{required}`"
        );
    }
    assert!(!release.contains("verified=true"));

    let runtime_handoff = read(
        "crates/neoethos-gpu-cuda/tests/fixtures/resident_higher_timeframe_alignment_v3.runtime-assembler-handoff.rs.txt",
    );
    for required in [
        "append_resident_higher_timeframe_alignment_v3",
        "ResidentHigherTimeframeExecutorV3::new(",
        "while let Some(batch) = executor.next_pending_batch_v3()?",
        "self.append_batch(Box::new(batch))?",
        "while !self.try_retire_completed_batch()?",
        "executor.finish_v3()",
        "do not register capability yet",
    ] {
        assert!(
            runtime_handoff.contains(required),
            "HTF runtime handoff omitted `{required}`"
        );
    }
}
