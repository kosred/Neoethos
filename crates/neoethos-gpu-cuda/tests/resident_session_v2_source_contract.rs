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
fn native_abi_is_fixed_width_and_carries_exactly_one_atomic_23_column_launch() {
    let source = read("crates/neoethos-gpu-cuda/native/resident_session_v2_abi.cuh");
    for required in [
        "NEOETHOS_RESIDENT_SESSION_ABI_VERSION_V2",
        "NeoResidentSessionLaunchV2",
        "semantic_version",
        "feature_column_count",
        "row_count",
        "const double* open",
        "const double* high",
        "const double* low",
        "const double* close",
        "const double* volume",
        "const std::int64_t* timestamps_ms",
        "double* feature_values",
        "std::uint8_t* feature_validity_u8",
        "static_assert(sizeof(NeoResidentSessionLaunchV2) == 88U)",
        "offsetof(NeoResidentSessionLaunchV2, timestamps_ms) == 64U",
        "neoethos_resident_session_f64_v2",
    ] {
        assert!(source.contains(required), "native ABI omitted `{required}`");
    }
    assert_eq!(
        source.matches("neoethos_resident_session_f64_v2").count(),
        1
    );
}

#[test]
fn cuda_is_one_sequential_dual_clock_state_machine_with_cpu_ordered_math() {
    let source = read("crates/neoethos-gpu-cuda/native/resident_session_v2.cu");
    let compact_source = compact(&source);
    for required in [
        "resident_session_all_f64_v2",
        "<<<1U, 1U, 0U, stream>>>",
        "0x7ff8000000000000ULL",
        "kFeatureColumnCountV2 = 23ULL",
        "kRetainedBytesPerRowV2 = 207ULL",
        "kValidV2 = 0U",
        "kWarmupV2 = 1U",
        "kZeroDenominatorV2 = 5U",
        "const std::int64_t millis_in_day",
        "const unsigned hour",
        "const unsigned minute",
        "const bool asian_open = hour == 0U && minute == 0U",
        "const bool london_open = hour == 7U && minute == 0U",
        "const bool new_york_open = hour == 12U && minute == 0U",
        "hour >= 12U && hour < 16U",
        "cumulative ATR",
        "value clock consumes the admitted millisecond inference",
        "validity clock consumes the original canonical millisecond timestamp",
    ] {
        assert!(
            source.contains(required),
            "CUDA source omitted `{required}`"
        );
    }
    assert_eq!(source.matches("__global__").count(), 1);
    for forbidden in [
        "cudaMalloc",
        "cudaFree",
        "cudaMemcpy",
        "cudaStreamSynchronize",
        "cudaDeviceSynchronize",
        "thrust::",
        "std::vector",
        "atomicAdd",
    ] {
        assert!(
            !compact_source.contains(forbidden),
            "Session-v2 CUDA contains forbidden `{forbidden}`"
        );
    }
}

#[test]
fn rust_owner_prevalidates_extents_and_closes_every_semantic_source_before_capability() {
    let source = read("crates/neoethos-gpu-cuda/src/resident_session_v2.rs");
    let compact_source = compact(&source);
    for required in [
        "SealedResidentSessionSourceClosureV2",
        "seal_resident_session_source_closure_v2",
        "ResidentSessionLaunchAuthorityV2",
        "resident_session_capability_v2",
        "neoethos_resident_session_f64_v2",
        "launch_resident_session_v2",
        "ResidentSessionRuntimeReceiptV2",
        "ResidentF64FeatureBatchV3",
        "validate_parent_extents_v2(parent, rows)?",
        "checked_mul(RESIDENT_SESSION_COLUMN_NAMES_V2.len())",
        "parent.producer_ready_event().wait_before_read",
        "parent.producer_context().as_raw() != context.as_raw()",
        "parent.producer_stream().as_inner() != stream.as_inner()",
        "retained_feature_device_bytes",
        "parent_input_h2d_bytes: 0",
        "feature_value_d2h_bytes: 0",
        "scratch_device_bytes: 0",
        "native_launch_count: 1",
        "producer_ready_event_count: 1",
        "size_of::<NeoResidentSessionLaunchV2>() == 88",
        "offset_of!(NeoResidentSessionLaunchV2, timestamps_ms) == 64",
        "include_bytes!(\"../native/resident_session_v2_abi.cuh\")",
        "include_bytes!(\"../native/resident_session_v2.cu\")",
        "../../neoethos-data/src/core/session_features.rs",
        "../../neoethos-data/src/core/timestamps.rs",
        "../../neoethos-data/src/core/features.rs",
        "../../neoethos-data/src/core/gpu_resident_session_v2.rs",
    ] {
        assert!(source.contains(required), "Rust owner omitted `{required}`");
    }
    let validation = compact_source
        .find("validate_parent_extents_v2(parent,rows)?")
        .expect("parent preallocation validation");
    let allocation = compact_source
        .find("StreamOrderedSessionBufferV2::<f64>::uninitialized_async")
        .expect("resident value allocation");
    assert!(
        validation < allocation,
        "parent validation must precede allocation"
    );
    for forbidden in ["derive(Clone", "derive(Copy", ".copy_to(", ".synchronize("] {
        assert!(
            !compact_source.contains(forbidden),
            "Rust Session-v2 owner contains forbidden `{forbidden}`"
        );
    }
}

#[test]
fn required_card_release_receipt_is_source_sealed_and_gates_capability() {
    let source = read("crates/neoethos-gpu-cuda/src/resident_session_v2.rs");
    let compact_source = compact(&source);
    let receipt = read(
        "crates/neoethos-gpu-cuda/tests/fixtures/resident_session_v2_device_parity.release.txt",
    );
    let required_exact_fields = [
        "verified=true",
        "semantic_version=2",
        "feature_column_count=23",
        "cpu_cuda_value_bit_mismatches=0",
        "cpu_cuda_validity_mismatches=0",
        "compute_sanitizer_errors=0",
        "compute_sanitizer_leaked_bytes=0",
        "racecheck_errors=0",
        "kernel_launch_count=257",
        "kernel_rows=4096",
        "kernel_feature_columns=23",
        "kernel_median_ns=35152055",
        "kernel_p95_ns=35166075",
        "input_h2d_copy_count=6",
        "feature_d2h_bytes=0",
        "device_identity_sha256=4f66f5daf6514123d6d17a634716f8ee428af60abc64136ef97b5fbc36748489",
        "parity_log_sha256=acc2a11184f079fce7c18b2c1f9b6a8469b81860f7fafdc97548fb56988e3cb8",
        "sanitizer_log_sha256=ffbfdf28bc7a9b39b403260f50a2cbf5765d3472f40809a215994ee618b68792",
        "racecheck_log_sha256=b04d3fdd7a7abbbe24ebdf21cc9ddfceffbebefc7dc253452b441f51098ed1fb",
        "nsys_report_sha256=757768625e0d46206dac2752d0464e9870ee3770b6d2f9e907e95c0fa72cd8b3",
    ];
    assert_eq!(receipt.lines().count(), required_exact_fields.len());
    assert_eq!(receipt.lines().collect::<Vec<_>>(), required_exact_fields);

    for required in [
        "resident_session_v2_device_parity.release.txt",
        "validate_resident_session_release_receipt_v2",
        "receipt_has_nonzero_sha256_v2",
        "release_receipt_accepts_exact_frozen_evidence",
        "release_receipt_rejects_drift_or_zero_hashes",
    ] {
        assert!(
            source.contains(required),
            "Session-v2 release authority omitted `{required}`"
        );
    }
    let receipt_closure = compact_source
        .find("include_bytes!(\"../tests/fixtures/resident_session_v2_device_parity.release.txt\")")
        .expect("release receipt must enter the source closure");
    let exact_math_closure = compact_source
        .find("implementation.update(RESIDENT_SESSION_EXACT_MATH_AUTHORITY_V2.as_bytes())")
        .expect("exact-math authority must enter the source closure");
    assert!(receipt_closure < exact_math_closure);

    let capability_gate = source
        .find("validate_resident_session_release_receipt_v2(device_receipt)?")
        .expect("required-card receipt validation");
    let capability_mint = source
        .find("let closure = seal_resident_session_source_closure_v2();")
        .expect("source closure before capability mint");
    assert!(
        capability_gate < capability_mint,
        "required-card evidence must fail closed before capability mint"
    );
}

#[test]
fn data_owner_consumes_native_closure_and_runtime_receipt_without_caller_capability_bits() {
    let source = read("crates/neoethos-data/src/core/gpu_resident_session_v2.rs");
    for required in [
        "preflight_current_native_resident_session_v2",
        "PreparedCurrentNativeResidentSessionProducerV2",
        "PreparedResidentSessionRuntimeV2",
        "into_recipe_parts",
        "seal_resident_session_source_closure_v2",
        "resident_session_capability_v2",
        "ResidentSessionLaunchAuthorityV2::seal",
        "append_resident_session_v2",
        "../../../neoethos-gpu-cuda/native/resident_session_v2_abi.cuh",
        "../../../neoethos-gpu-cuda/native/resident_session_v2.cu",
        "../../../neoethos-gpu-cuda/src/resident_session_v2.rs",
    ] {
        assert!(source.contains(required), "Data owner omitted `{required}`");
    }
    let store = read("crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs");
    assert!(
        store.contains("session_runtime.append_to(&mut assembler, session_bindings)?"),
        "crate-owned materializer does not consume the prepared Session runtime"
    );
    let compact_source = compact(&source);
    for required in [
        "structPreparedResidentSessionRuntimeV2{runtime_admission:ResidentSessionRuntimeAdmissionV2,launch_authority:ResidentSessionLaunchAuthorityV2,}",
        "fninto_recipe_parts(",
        "ResidentProducerDraftV4,PreparedResidentSessionRuntimeV2",
        "letreceipt=assembler.append_resident_session_v2(bindings,launch_authority)?;",
        "runtime_admission.validate_native_receipt(&receipt)",
    ] {
        assert!(
            compact_source.contains(required),
            "Session runtime split omitted `{required}`"
        );
    }
    assert!(!source.contains("allow_cpu_fallback"));
}

#[test]
fn build_store_and_device_fixture_are_connected_without_feature_d2h_in_production() {
    let build = read("crates/neoethos-gpu-cuda/build.rs");
    let library = read("crates/neoethos-gpu-cuda/src/lib.rs");
    let store = read("crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    let fixture = read("crates/neoethos-gpu-cuda/src/resident_session_v2_device_fixture.rs");
    assert!(build.contains("native/resident_session_v2.cu"));
    assert!(build.contains("native/resident_session_v2_abi.cuh"));
    assert!(library.contains("pub mod resident_session_v2;"));
    assert!(library.contains("pub mod resident_session_v2_device_fixture;"));
    for required in [
        "append_resident_session_v2",
        "ResidentSessionRuntimeReceiptV2",
        "session_runtime_receipt_v2",
    ] {
        assert!(
            store.contains(required),
            "resident store omitted `{required}`"
        );
    }
    for required in [
        "cfg(feature = \"cuda-device-fixtures\")",
        "run_resident_session_v2_device_fixture",
        "run_resident_session_v2_device_perf_fixture",
        "copy_to",
        "test-only parity D2H",
        "zero feature D2H",
    ] {
        assert!(
            fixture.contains(required),
            "device fixture omitted `{required}`"
        );
    }
}

#[test]
fn cpu_oracle_covers_all_session_boundaries_gaps_zero_denominators_and_unit_inference() {
    let oracle = read("crates/neoethos-data/tests/resident_session_v2_oracle.rs");
    for required in [
        "asian_london_new_york_overlap_and_utc_day_boundaries",
        "sparse_observed_rows_preserve_exact_open_only_resets",
        "zero_atr_and_zero_volume_emit_canonical_nan_with_typed_reason",
        "legacy_value_clock_infers_seconds_while_typed_resident_authority_rejects_them",
        "rtx_device_fixture_matches_all_session_v2_value_bits_and_validity_codes",
        "rtx_session_v2_4096_resident_kernel_perf_fixture",
    ] {
        assert!(oracle.contains(required), "CPU oracle omitted `{required}`");
    }
}
