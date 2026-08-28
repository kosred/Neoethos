use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("neoethos-gpu-cuda must live under the workspace root")
        .to_path_buf()
}

fn read(path: impl AsRef<Path>) -> String {
    fs::read_to_string(workspace_root().join(path)).unwrap_or_default()
}

#[test]
fn classic_ta_executor_is_gpu_cuda_owned_and_borrows_the_one_shot_carrier() {
    let runtime = read("crates/neoethos-gpu-cuda/src/resident_classic_ta_v3.rs");

    for token in [
        "ResidentClassicTaRecipeV3",
        "ResidentClassicTaExecutorV3",
        "GpuOnlyRunDeviceAdmissionV3",
        "ResidentParentDatasetSourceV3",
        "CudaSession::from_parts",
        "CudaF64Indicators::from_session",
        "primary_context_for_resident_producer_v3",
        "run_stream_for_resident_producer_v3",
        "producer_ready_event",
    ] {
        assert!(
            runtime.contains(token),
            "resident Classic TA executor omitted `{token}`"
        );
    }

    for forbidden in [
        "Context::new",
        "Stream::new",
        "upload_ohlcv",
        "download_",
        ".synchronize(",
        "compute_cpu_batch",
        "f32",
        "fallback",
    ] {
        assert!(
            !runtime.contains(forbidden),
            "strict resident Classic TA reintroduced forbidden `{forbidden}`"
        );
    }
}

#[test]
fn classic_ta_batches_are_monotonic_bounded_opaque_and_stream_retired() {
    let runtime = read("crates/neoethos-gpu-cuda/src/resident_classic_ta_v3.rs");
    let vector_ta = read("vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");

    for token in [
        "MAX_RESIDENT_CLASSIC_TA_BATCH_COLUMNS_V3: usize = 64",
        "PendingResidentClassicTaBatchV3",
        "ResidentF64FeatureBatchV3",
        "first_destination_column",
        "next_destination_column",
        "producer_ready_event",
        "enqueue_nonblocking_release",
        "retained_device_bytes",
        "retained_scratch_bytes",
        "sweep_resident_v3",
        "into_resident_parts_v3",
        "validate_observed_route_v3",
        "F64_EXACT_MATH_AUTHORITY_V3",
    ] {
        assert!(runtime.contains(token), "bounded batch omitted `{token}`");
    }
    assert!(vector_ta.contains("into_resident_parts_v3"));
    assert!(!runtime.contains("pub fn value_buffer"));
    assert!(!runtime.contains("pub fn producer_context"));
    assert!(!runtime.contains("pub fn producer_stream"));
}

#[test]
fn classic_ta_derived_inputs_and_validity_are_device_owned_exact_f64() {
    let runtime = read("crates/neoethos-gpu-cuda/src/resident_classic_ta_v3.rs");
    let native = read("crates/neoethos-gpu-cuda/native/resident_classic_ta_v3.cu");
    let build = read("crates/neoethos-gpu-cuda/build.rs");

    for token in [
        "hlc3",
        "hl2",
        "hlcc4",
        "classic_validity_u8_v3",
        "FeatureCellValidity",
        "Warmup",
        "Valid",
        "NonFinite",
        "ComputeFailure",
        "neoethos_resident_classic_fill_nan_f64_v3",
        "isinf(value)",
        "atomicCAS(device_error, 0U, 3U)",
        "0xffU",
    ] {
        assert!(
            runtime.contains(token) || native.contains(token),
            "device Classic TA input/validity authority omitted `{token}`"
        );
    }
    assert!(native.contains("neoethos_resident_classic_derived_inputs_f64_v3"));
    assert!(native.contains("neoethos_resident_classic_validity_u8_v3"));
    assert!(build.contains("native/resident_classic_ta_v3.cu"));
    assert!(!native.contains("--use_fast_math"));
}

#[test]
fn classic_ta_device_maxima_are_typed_without_relaxed_constexpr() {
    let native = read("crates/neoethos-gpu-cuda/native/resident_classic_ta_v3.cu");
    let build = read("crates/neoethos-gpu-cuda/build.rs");

    assert_eq!(
        native
            .matches("const std::size_t size_max_v3 = ~std::size_t{0};")
            .count(),
        2,
        "both device validity kernels need an exact typed size_t maximum"
    );
    assert_eq!(
        native
            .matches("std::numeric_limits<std::size_t>::max()")
            .count(),
        1,
        "only the host launch validator may retain numeric_limits<size_t>::max()"
    );
    assert!(native.contains("const std::uint64_t u64_max_v3 = ~std::uint64_t{0};"));
    assert!(!native.contains("std::numeric_limits<unsigned long long>::max()"));
    for source in [&native, &build] {
        assert!(
            !source.contains("expt-relaxed-constexpr"),
            "the Classic TA native build must not weaken CUDA constexpr authority"
        );
    }
}

#[test]
fn classic_ta_compiled_owner_uses_crate_event_authority_without_fake_debug_or_mutability() {
    let runtime =
        read("crates/neoethos-gpu-cuda/src/resident_classic_ta_v3.rs").replace("\r\n", "\n");
    let store = read("crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs");

    assert!(store.contains("pub(crate) fn wait_before_read("));
    assert!(!runtime.contains("F64Kernel, F64ResidentNamedPartsV3"));
    assert!(!runtime.contains("#[derive(Debug)]\npub(crate) struct ResidentClassicTaExecutorV3"));
    for forbidden in [
        "let mut hlc3 = ResidentClassicTaDeviceBufferV3",
        "let mut hl2 = ResidentClassicTaDeviceBufferV3",
        "let mut hlcc4 = ResidentClassicTaDeviceBufferV3",
        "let mut first_finite_rows",
        "let mut validity_u8",
        "let mut validity_device_error",
        "let mut output = ResidentClassicTaDeviceBufferV3",
    ] {
        assert!(
            !runtime.contains(forbidden),
            "compiled Classic TA owner retained unnecessary `{forbidden}`"
        );
    }
}

#[test]
fn classic_ta_optional_vector_ta_edge_and_module_are_integrated() {
    let manifest = read("crates/neoethos-gpu-cuda/Cargo.toml");
    let library = read("crates/neoethos-gpu-cuda/src/lib.rs");

    assert!(manifest.contains("vector-ta = { version = \"0.2.9\", optional = true"));
    assert!(manifest.contains("cuda = [\"dep:cust\", \"dep:vector-ta\"]"));
    assert!(library.contains("pub mod resident_classic_ta_v3;"));
}

#[test]
fn classic_ta_named_route_owned_launches_cannot_be_stolen_by_primary_specs() {
    let runtime =
        read("crates/neoethos-gpu-cuda/src/resident_classic_ta_v3.rs").replace("\r\n", "\n");

    assert!(runtime.contains(
        "launch.first_valid_rule() != ResidentClassicTaFirstValidRuleV3::NamedRouteOwned"
    ));
    assert!(runtime.contains("\"cvi\" =>"));
    assert!(runtime.contains("require_parameter_keys_v3(launch, &[\"period\"])?;"));
    assert!(runtime.contains("self.engine.cvi_production_output("));
}

#[test]
fn classic_ta_primary_input_guard_preserves_semantic_source_kinds() {
    let runtime = read("crates/neoethos-gpu-cuda/src/resident_classic_ta_v3.rs");
    let compact: String = runtime
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();

    for exact_pair in [
        "(ResidentClassicTaInputV3::Hlcc4,F64InputKind::Hlcc4Slice)",
        "(ResidentClassicTaInputV3::Volume,F64InputKind::VolumeSlice)",
        "(ResidentClassicTaInputV3::Hlcc4Volume,F64InputKind::Hlcc4Volume)",
    ] {
        assert!(
            compact.contains(exact_pair),
            "resident Classic TA input guard omitted semantic pair `{exact_pair}`"
        );
    }

    for collapsed_pair in [
        "(ResidentClassicTaInputV3::Hlcc4,F64InputKind::CloseSlice)",
        "(ResidentClassicTaInputV3::Volume,F64InputKind::CloseSlice)",
        "(ResidentClassicTaInputV3::Hlcc4Volume,F64InputKind::CloseVolume)",
    ] {
        assert!(
            !compact.contains(collapsed_pair),
            "resident Classic TA input guard collapsed semantic kind to ABI shape `{collapsed_pair}`"
        );
    }
}

#[test]
fn classic_primary_and_warmup_memory_are_preflighted_before_the_carrier_allocates() {
    let runtime =
        read("crates/neoethos-gpu-cuda/src/resident_classic_ta_v3.rs").replace("\r\n", "\n");
    let vector = read("vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    let compact_runtime: String = runtime
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();

    for required in [
        "ResidentClassicTaPreDeviceMemoryReceiptV4",
        "ResidentClassicTaLaunchMemoryPlanV4",
        "preflight_resident_classic_ta_memory_v4",
        "F64ResidentSingleSweepAllocationPlanV4",
        "preflight_resident_single_sweep_allocation_v4",
        "selected_value_bytes",
        "all_output_retained_bytes",
        "additional_retained_bytes",
        "validity_bytes",
        "validity_scratch_bytes",
        "derived_input_bytes",
        "derived_ready_event_count",
        "ready_event_count",
    ] {
        assert!(
            runtime.contains(required) || vector.contains(required),
            "Classic pre-device plan omitted {required:?}"
        );
    }
    assert!(compact_runtime.contains("rows.checked_mul(3)"));
    assert!(compact_runtime.contains("columns.checked_mul(25)"));
    assert!(runtime.contains("sweep_resident_preplanned_v4"));
    assert!(runtime.contains("runtime_memory_receipt_v4 != pre_device_memory_receipt_v4"));
    let equality_guard = runtime
        .find("runtime_memory_receipt_v4 != pre_device_memory_receipt_v4")
        .expect("exact pre-device receipt equality guard");
    let first_allocation = runtime
        .find("ResidentClassicTaDerivedInputsV3::launch(")
        .expect("first retained Classic allocation");
    assert!(
        equality_guard < first_allocation,
        "the complete primary/warmup receipt must compare equal before any Classic allocation"
    );
    for move_only in [
        "ResidentClassicTaLaunchMemoryPlanV4",
        "ResidentClassicTaPreDeviceMemoryReceiptV4",
    ] {
        assert!(!runtime.contains(&format!("impl Clone for {move_only}")));
        assert!(!runtime.contains(&format!("impl Copy for {move_only}")));
    }
}

#[test]
fn classic_runtime_uses_admitted_global_bindings_after_smc_not_local_zero_ordinals() {
    let runtime = read("crates/neoethos-gpu-cuda/src/resident_classic_ta_v3.rs");
    let store = read("crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    let data = read("crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs");
    let compact_runtime: String = runtime
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();

    assert!(store.contains("append_resident_classic_ta_recipe_v4"));
    assert!(runtime.contains("admitted_global_bindings"));
    assert!(runtime.contains("validate_admitted_global_bindings_v4"));
    assert!(runtime.contains("classic_global_start == 0"));
    for exact_guard in [
        "admitted_global_bindings.len() != recipe.output_count()",
        "binding.ordinal != expected_global_ordinal",
        "binding.feature_name != route.feature_name()",
        "binding.canonical_parameter_tuple_sha256",
        "!= route.canonical_parameter_tuple_sha256()",
        "binding.route_receipt_sha256.iter().all(|byte| *byte == 0)",
    ] {
        assert!(
            compact_runtime.contains(&exact_guard.replace(' ', "")),
            "global Classic binding validation omitted {exact_guard:?}"
        );
    }
    assert!(!compact_runtime.contains(
        ".map(|route|ResidentFeatureColumnBindingV3{ordinal:route.destination_column(),"
    ));

    let smc = data
        .find("pending_smc_batch.append_to(&mut assembler)?")
        .expect("SMC append");
    let classic = data
        .find("assembler.append_resident_classic_ta_recipe_v4(")
        .expect("globally bound Classic append");
    assert!(
        smc < classic,
        "SMC must occupy its admitted span before Classic"
    );

    let memory_preflight = data
        .find("preflight_resident_classic_ta_memory_v4(")
        .expect("Classic pre-device memory preflight");
    let carrier_move = data
        .find("admitted_run.into_gpu_only_run_device_admission_v3()")
        .expect("one-shot run-device carrier move");
    assert!(
        memory_preflight < carrier_move,
        "Classic memory authority must fail closed before the one-shot carrier moves"
    );
}
