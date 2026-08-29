use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

fn read(path: impl AsRef<Path>) -> String {
    let path = workspace_root().join(path);
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

fn function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("missing function signature `{signature}`"));
    let open = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .expect("function has no opening brace");
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[open..].iter().copied().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth = depth.checked_sub(1).expect("unbalanced closing brace");
                if depth == 0 {
                    return &source[open + 1..open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("function `{signature}` has no closing brace");
}

#[test]
fn resident_smc_is_one_same_carrier_cuda_parent_and_feature_factory() {
    let runtime = read("crates/neoethos-gpu-cuda/src/resident_smc_v3.rs");
    let native = read("crates/neoethos-gpu-cuda/native/resident_smc_v3.cu");
    let data = read("crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs");

    for token in [
        "pub const RESIDENT_SMC_COLUMN_NAMES_V3: [&str; 46]",
        "pub const RESIDENT_SMC_PARENT_SLOT_NAMES_V3: [&str; 11]",
        "pub fn prepare_resident_smc_parent_v3(",
        "pub fn begin_resident_smc_store_v3(",
        "pub struct PendingResidentSmcBatchV3",
        "pub fn append_to(",
        "GpuOnlyRunDeviceAdmissionV3",
        "primary_context_for_resident_producer_v3",
        "run_stream_for_resident_producer_v3",
        "ResidentProducerReadyEventV3::record(",
        "sealed_validity_u4_logical_bytes",
        "sealed_validity_u4_padded_device_bytes",
        "one_time_input_h2d_bytes",
        "transient_device_bytes",
        "peak_device_bytes",
        "producer_ready_event_count: 1",
        "unsafe impl ResidentParentDatasetSourceV3",
        "unsafe impl ResidentF64FeatureBatchV3",
        "neoethos_resident_smc_parent_features_f64_v3(",
        "enqueue_nonblocking_release",
        "drop_async",
    ] {
        assert!(
            runtime.contains(token),
            "missing resident SMC runtime `{token}`"
        );
    }
    for token in [
        "neoethos_resident_smc_parent_features_f64_v3",
        "kSmcFeatureColumnsV3 = 46",
        "kSmcParentSlotsV3 = 11",
        "smc_feature_values",
        "smc_feature_validity_u8",
        "smc_parent_rows",
        "canonical_nan_v3",
        "neoethos_smc_log1p_cpu_exact_v1",
        "stream == nullptr",
    ] {
        assert!(
            native.contains(token),
            "missing resident SMC CUDA `{token}`"
        );
    }
    for token in [
        "prepare_resident_smc_parent_v3",
        "ResidentFeatureProducerV3::Smc",
        "ResidentSmcMaterializationV3",
    ] {
        assert!(
            data.contains(token),
            "Data does not own SMC handoff `{token}`"
        );
    }

    for forbidden in [
        "compute_smc_feature_columns(",
        "compute_smc_feature_columns_f64(",
        "FeatureFrame",
        "stream.synchronize()",
        "Context::new(",
        "Stream::new(",
        "vec![0_i8; rows * 11]",
        "DeviceBuffer::from_slice(",
    ] {
        assert!(
            !runtime.contains(forbidden),
            "resident SMC path reintroduced forbidden host/fallback seam `{forbidden}`"
        );
    }
    assert!(!native.contains("::log1p("));
    assert!(!native.contains("smc_parent_rows[cell] = 0"));
    assert!(!native.contains("fallback_"));
    assert!(!runtime.contains("into_resident_sources_v3"));
    assert!(!runtime.contains("Box<dyn ResidentParentDatasetSourceV3>"));
    assert!(!runtime.contains("Box<dyn ResidentF64FeatureBatchV3>"));
    assert_eq!(
        runtime.matches("async_copy_from(").count(),
        6,
        "one-time parent upload is exact OHLCV plus timestamps"
    );
    assert_eq!(runtime.matches("copy_to(").count(), 2);
}

#[test]
fn resident_smc_pins_canonical_order_validity_and_parent_slot_derivation() {
    let cpu = read("crates/neoethos-data/src/core/smc.rs");
    let registry = read("crates/neoethos-data/src/core/feature_registry.rs");
    let runtime = read("crates/neoethos-gpu-cuda/src/resident_smc_v3.rs");
    let native = read("crates/neoethos-gpu-cuda/native/resident_smc_v3.cu");

    let names = [
        "smc_ob",
        "smc_fvg",
        "smc_ifvg",
        "smc_liq_sweep",
        "smc_pd_array",
        "smc_killzone",
        "smc_displacement",
        "smc_breaker_block",
        "smc_mitigation_block",
        "smc_mss",
        "smc_volume_imbalance",
        "smc_bos",
        "smc_eqh",
        "smc_eql",
        "smc_inducement",
        "smc_asian_range",
        "smc_silver_bullet",
        "smc_judas_swing",
        "smc_nwog",
        "smc_ndog",
        "smc_ict_macro",
        "smc_fvg_strength",
        "smc_dealing_range_width",
        "smc_swing_range_pct",
        "smc_ob_strength",
        "smc_trend_bias",
        "smc_unicorn_model",
        "smc_rejection_block",
        "smc_propulsion_block",
        "smc_fib_time_ratio",
        "smc_fib_236",
        "smc_fib_382",
        "smc_fib_500",
        "smc_fib_618",
        "smc_fib_705",
        "smc_fib_786",
        "smc_fib_886",
        "smc_fib_1272",
        "smc_fib_1414",
        "smc_fib_1618",
        "smc_fib_2000",
        "smc_fib_2618",
        "smc_fvg_magnet_dist",
        "smc_fvg_magnet_age",
        "smc_fvg_inside",
        "smc_fvg_open_count",
    ];
    for name in names {
        assert!(cpu.contains(name), "CPU SMC authority omitted `{name}`");
        assert!(
            registry.contains(name),
            "registry SMC authority omitted `{name}`"
        );
        assert!(
            runtime.contains(name),
            "resident SMC order omitted `{name}`"
        );
    }
    for token in [
        "enum SmcValidityV3 : unsigned char",
        "kValid = 0",
        "kWarmup = 1",
        "kMissingInput = 2",
        "kGap = 3",
        "kStale = 4",
        "kZeroDenominator = 5",
        "kDegenerate = 6",
        "kNonFinite = 7",
        "kComputeFailure = 8",
        "kAlignmentMissing = 9",
    ] {
        assert!(
            native.contains(token),
            "resident SMC validity omitted `{token}`"
        );
    }
    assert!(runtime.contains("logical-u8.codes-0-through-9.v3"));
    assert!(runtime.contains("physical-u4.low-nibble-first.v3"));
    for token in [
        "let timestamp_in_range =",
        "let timestamp_is_strictly_increasing =",
        "if !timestamp_in_range || !timestamp_is_strictly_increasing",
    ] {
        assert!(
            runtime.contains(token),
            "resident SMC timestamp authority omitted `{token}`"
        );
    }
    for token in [
        "SMC_SLOT_ORDER_V3",
        "smc_ob",
        "smc_fvg",
        "smc_liq_sweep",
        "smc_trend_bias",
        "smc_pd_array",
        "smc_inducement",
        "smc_bos",
        "smc_mss",
        "smc_eqh",
        "smc_eql",
        "smc_displacement",
    ] {
        assert!(
            runtime.contains(token) || native.contains(token),
            "resident parent SMC slot authority omitted `{token}`"
        );
    }
}

#[test]
fn resident_smc_semantic_v3_is_an_explicit_content_identity_migration() {
    let cpu = read("crates/neoethos-data/src/core/smc.rs");
    let exact_log = read("crates/neoethos-data/src/core/smc_log1p_exact_v1.rs");
    let registry = read("crates/neoethos-data/src/core/feature_registry.rs");
    let native = read("crates/neoethos-gpu-cuda/native/resident_smc_v3.cu");
    for token in [
        "pub const SMC_SEMANTIC_VERSION: u32 = 3",
        "SMC_V2_ARTIFACT_MIGRATION_POLICY",
        "refuse semantic-v2 SMC artifacts",
        "smc_log1p_exact_v1((i - born) as u64)",
    ] {
        assert!(
            cpu.contains(token),
            "SMC v3 CPU authority omitted `{token}`"
        );
    }
    assert!(registry.contains("super::smc::SMC_SEMANTIC_VERSION"));
    assert!(registry.contains("smc_log1p_exact_v1.rs"));
    assert!(exact_log.contains("denominator <= 49"));
    assert!(native.contains("neoethos_smc_log1p_cpu_exact_v1"));
    assert!(native.contains("denominator <= 49U"));
    assert!(!cpu.contains(".ln_1p()"));
    assert!(!native.contains("::log1p("));
}

#[test]
fn resident_smc_uses_the_portable_exact_f64_infinity_identity() {
    let native = read("crates/neoethos-gpu-cuda/native/resident_smc_v3.cu");

    assert!(native.contains("#include <limits>"));
    assert!(native.contains(
        "constexpr double kPositiveInfinityV3 = std::numeric_limits<double>::infinity();"
    ));
    assert!(native.contains("double last_confirmed_high = -kPositiveInfinityV3;"));
    assert!(native.contains("double last_confirmed_low = kPositiveInfinityV3;"));
    assert!(
        !native.contains("CUDART_INF"),
        "NVCC 13 does not make CUDART_INF available through this translation unit"
    );
}

#[test]
fn resident_parent_v4_retains_and_hashes_the_complete_classic_ohlcv_input_once() {
    let contracts = read("crates/neoethos-gpu-contracts/src/resident_feature_store_v3.rs");
    let store = read("crates/neoethos-gpu-cuda/src/resident_feature_store_v3.rs");
    let runtime = read("crates/neoethos-gpu-cuda/src/resident_smc_v3.rs");
    let data = read("crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs");

    for token in [
        "ResidentParentDatasetLayoutV4",
        "open_sha256",
        "volume_sha256",
        "neoethos.resident-parent-dataset-layout.v4",
        "checked_mul(row_count, 5, \"resident parent OHLCV arrays\")",
    ] {
        assert!(
            contracts.contains(token),
            "complete resident parent contract omitted `{token}`"
        );
    }
    for token in [
        "fn open(&self) -> &DeviceBuffer<f64>",
        "fn volume(&self) -> &DeviceBuffer<f64>",
        "(\"open\", parent.open().len())",
        "(\"volume\", parent.volume().len())",
    ] {
        assert!(
            store.contains(token),
            "resident parent runtime omitted `{token}`"
        );
    }
    for token in [
        "volume: LockedBuffer<f64>",
        "volume: StreamOrderedSmcBufferV3<f64>",
        "open_sha256 = hash_f64_bits_le_v3(open)",
        "volume_sha256 = hash_f64_bits_le_v3(volume)",
        "open: transient_open",
        "volume: retained_volume",
        "5 * std::mem::size_of::<f64>()",
    ] {
        assert!(
            runtime.contains(token),
            "one-upload SMC parent successor omitted `{token}`"
        );
    }
    assert!(data.contains("source.volume.as_deref()"));
    assert!(data.contains("volume,"));
    assert!(!runtime.contains("drop(transient_open);"));
}

#[test]
fn resident_smc_rejects_nonfinite_or_negative_volume_before_upload() {
    let runtime = read("crates/neoethos-gpu-cuda/src/resident_smc_v3.rs");
    let validation = function_body(&runtime, "fn validate_inputs_v3(");

    for token in [
        "let row_volume = volume[row];",
        "!row_volume.is_finite()",
        "row_volume < 0.0",
    ] {
        assert!(
            validation.contains(token),
            "canonical OHLCV validation omitted `{token}`"
        );
    }
}

#[test]
fn only_real_smc_capability_is_admitted_and_the_other_nine_refuse_in_order() {
    let data = read("crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs");
    for token in [
        "resident_smc_capability_v3()",
        "assert_eq!(missing, EXPECTED_MISSING_AFTER_SMC_V3)",
        "ClassicTa",
        "Quant",
        "Session",
        "Regime",
        "Footprint",
        "HigherTimeframeAlignment",
        "RobustNormalization",
        "CanonicalContentSha256",
        "FeatureMajorToBarMajor",
    ] {
        assert!(data.contains(token), "SMC-only census omitted `{token}`");
    }
    assert!(!data.contains("assert_eq!(missing, ResidentFeatureProducerV3::ALL)"));
}

#[test]
fn resident_smc_cuda_unit_is_build_linked_and_exported() {
    let build = read("crates/neoethos-gpu-cuda/build.rs");
    let library = read("crates/neoethos-gpu-cuda/src/lib.rs");
    assert!(build.contains("native/resident_smc_v3.cu"));
    assert!(library.contains("#[cfg(feature = \"cuda\")]\npub mod resident_smc_v3;"));
}
