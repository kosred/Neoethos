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

const QUANT_COLUMN_NAMES_V3: [&str; 63] = [
    "quant_close",
    "quant_return_1",
    "quant_return_2",
    "quant_return_3",
    "quant_return_5",
    "quant_return_8",
    "quant_return_13",
    "quant_return_21",
    "quant_log_return",
    "quant_log_volatility",
    "quant_realized_vol_5",
    "quant_realized_vol_10",
    "quant_realized_vol_20",
    "quant_realized_vol_50",
    "quant_gk_vol_10",
    "quant_gk_vol_20",
    "quant_parkinson_vol_10",
    "quant_parkinson_vol_20",
    "quant_vol_ratio",
    "quant_hurst_100",
    "quant_autocorr_1",
    "quant_autocorr_5",
    "quant_autocorr_10",
    "quant_efficiency_ratio_10",
    "quant_efficiency_ratio_20",
    "quant_skewness_30",
    "quant_kurtosis_30",
    "quant_kyle_lambda",
    "quant_vpin",
    "quant_amihud_illiquidity",
    "quant_roll_spread",
    "quant_consec_up",
    "quant_consec_down",
    "quant_inside_bar",
    "quant_outside_bar",
    "quant_body_ratio",
    "quant_upper_shadow",
    "quant_lower_shadow",
    "quant_prev_day_h_dist",
    "quant_prev_day_l_dist",
    "quant_prev_week_h_dist",
    "quant_prev_week_l_dist",
    "quant_orb_4",
    "quant_orb_8",
    "quant_orb_12",
    "quant_amd_phase",
    "quant_wyckoff",
    "quant_engulfing_vol",
    "quant_pivot_dist",
    "quant_r1_dist",
    "quant_r2_dist",
    "quant_s1_dist",
    "quant_s2_dist",
    "quant_cam_r3_dist",
    "quant_cam_s3_dist",
    "quant_zscore_20",
    "quant_zscore_50",
    "quant_fractal_dim",
    "quant_rvol_10",
    "quant_rvol_20",
    "quant_rvol_50",
    "quant_delta_volume",
    "quant_cum_delta_zscore",
];

#[test]
fn census_freezes_31_preserved_and_32_versioned_migrations_in_schema_order() {
    let source = read("crates/neoethos-gpu-cuda/src/resident_quant_v3_census.rs");
    let compact_source = compact(&source);
    for name in QUANT_COLUMN_NAMES_V3 {
        assert!(source.contains(name), "Quant-v3 census omitted {name}");
    }
    for required in [
        "RESIDENT_QUANT_COLUMN_NAMES_V3: [&str; 63]",
        "RESIDENT_QUANT_V2_BITWISE_PRESERVED_ROUTE_COUNT_V3: usize = 31",
        "RESIDENT_QUANT_V3_EXACT_LOG_MIGRATION_ROUTE_COUNT_V3: usize = 10",
        "RESIDENT_QUANT_V3_ANNUALIZED_EXACT_LOG_MIGRATION_ROUTE_COUNT_V3: usize = 8",
        "RESIDENT_QUANT_V3_TEMPORAL_MIGRATION_ROUTE_COUNT_V3: usize = 14",
        "RESIDENT_QUANT_V3_MIGRATED_ROUTE_COUNT_V3: usize = 32",
        "RESIDENT_QUANT_V3_CHANGED_COLUMN_NAMES_V3: [&str; 32]",
        "trading_sessions_per_year=252",
        "unversioned-artifacts=fail-closed",
        "never-label-as-bitwise-v2-parity",
        "V2BitwisePreserved",
        "V3ExactLogMigration",
        "V3AnnualizedExactLogMigration",
        "V3TemporalSessionMigration",
    ] {
        assert!(source.contains(required), "census omitted `{required}`");
    }
    assert!(!compact_source.contains("MissingInput"));
}

#[test]
fn native_abi_is_fixed_width_typed_and_has_one_63_column_entrypoint() {
    let source = read("crates/neoethos-gpu-cuda/native/resident_quant_v3_abi.cuh");
    for required in [
        "NEOETHOS_RESIDENT_QUANT_ABI_VERSION_V3",
        "NeoResidentQuantLaunchV3",
        "semantic_version",
        "feature_column_count",
        "row_count",
        "timeframe_millis",
        "bars_per_asian_session",
        "bars_per_utc_day",
        "bars_per_trading_week",
        "trading_sessions_per_year",
        "annualization_periods_per_year",
        "const double* open",
        "const double* volume",
        "const std::int64_t* timestamps",
        "double* feature_values",
        "std::uint8_t* feature_validity_u8",
        "static_assert(sizeof(NeoResidentQuantLaunchV3) == 136)",
        "static_assert(offsetof(NeoResidentQuantLaunchV3, open) == 72)",
        "static_assert(offsetof(NeoResidentQuantLaunchV3, feature_validity_u8) == 128)",
        "neoethos_resident_quant_f64_v3",
    ] {
        assert!(source.contains(required), "native ABI omitted `{required}`");
    }
    assert_eq!(source.matches("neoethos_resident_quant_f64_v3").count(), 1);
}

#[test]
fn native_kernel_is_one_bounded_deterministic_launch_without_alloc_sync_or_d2h() {
    let source = read("crates/neoethos-gpu-cuda/native/resident_quant_v3.cu");
    let compact_source = compact(&source);
    for required in [
        "resident_quant_all_f64_v3",
        "<<<1U, 1U, 0U, stream>>>",
        "0x7ff8000000000000ULL",
        "kTradingSessionsPerYearV3 = 252ULL",
        "kFeatureColumnCountV3 = 63ULL",
        "kValidV3 = 0U",
        "kWarmupV3 = 1U",
        "kZeroDenominatorV3 = 5U",
        "sqrt_rn_v3(static_cast<double>(launch.annualization_periods_per_year))",
        "previous completed UTC day",
        "first N observed Asian-session bars",
        "fixed maximum lookback of 500 bars",
    ] {
        assert!(
            source.contains(required),
            "native kernel omitted `{required}`"
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
    ] {
        assert!(
            !compact_source.contains(forbidden),
            "native Quant-v3 contains forbidden `{forbidden}`"
        );
    }
}

#[test]
fn rust_owner_binds_real_abi_before_minting_capability_and_retains_same_stream_buffers() {
    let source = read("crates/neoethos-gpu-cuda/src/resident_quant_v3.rs");
    let compact_source = compact(&source);
    for required in [
        "SealedResidentQuantMigrationClosureV3",
        "seal_resident_quant_migration_closure_v3",
        "ResidentQuantLaunchAuthorityV3",
        "resident_quant_capability_v3",
        "resident_quant_v3_device_parity.release.txt",
        "verified=true",
        "cpu_cuda_value_bit_mismatches=0",
        "cpu_cuda_validity_mismatches=0",
        "compute_sanitizer_errors=0",
        "compute_sanitizer_leaked_bytes=0",
        "device_identity_sha256",
        "parity_log_sha256",
        "sanitizer_log_sha256",
        "nsys_report_sha256",
        "receipt_has_nonzero_sha256_v3",
        "neoethos_resident_quant_f64_v3",
        "launch_resident_quant_v3",
        "ResidentQuantRuntimeReceiptV3",
        "ResidentF64FeatureBatchV3",
        "assert!(std::mem::size_of::<NeoResidentQuantLaunchV3>() == 136)",
        "assert!(std::mem::offset_of!(NeoResidentQuantLaunchV3, open) == 72)",
        "assert!(std::mem::offset_of!(NeoResidentQuantLaunchV3, feature_validity_u8) == 128)",
        "parent.producer_ready_event().wait_before_read",
        "parent.producer_context().as_raw() != context.as_raw()",
        "parent.producer_stream().as_inner() != stream.as_inner()",
        "feature_value_d2h_bytes: 0",
        "parent_input_h2d_bytes: 0",
        "native_launch_count: 1",
        "producer_ready_event_count: 1",
        "include_bytes!(\"../native/resident_quant_v3_abi.cuh\")",
        "include_bytes!(\"../native/resident_quant_v3.cu\")",
    ] {
        assert!(
            source.contains(required),
            "Rust native owner omitted `{required}`"
        );
    }
    for required in [
        "include_bytes!(\"../../neoethos-data/src/core/quant_features.rs\")",
        "include_bytes!(\"../../neoethos-data/src/core/gpu_resident_quant_v3.rs\")",
    ] {
        assert!(
            compact_source.contains(required),
            "Rust native owner omitted whitespace-insensitive `{required}`"
        );
    }
    for forbidden in ["derive(Copy", ".copy_to(", ".synchronize("] {
        assert!(
            !compact_source.contains(forbidden),
            "Rust Quant-v3 owner contains forbidden `{forbidden}`"
        );
    }
}

#[test]
fn data_connection_consumes_native_closure_and_never_accepts_caller_capability_bits() {
    let source = read("crates/neoethos-data/src/core/gpu_resident_quant_v3.rs");
    let compact_source = compact(&source);
    for required in [
        "preflight_current_native_resident_quant_v3",
        "PreparedCurrentNativeResidentQuantProducerV3",
        "PreparedResidentQuantRuntimeV3",
        "seal_resident_quant_migration_closure_v3",
        "resident_quant_capability_v3",
        "ResidentQuantLaunchAuthorityV3::seal",
        "append_resident_quant_v3",
        "into_recipe_parts",
        "pub(crate) fn append_to(",
    ] {
        assert!(
            source.contains(required),
            "Data Quant connection omitted `{required}`"
        );
    }
    assert!(compact_source.contains("self.runtime_admission.validate_native_receipt(&receipt)"));
    assert!(!source.contains("into_launched_parts"));
    assert!(!source.contains("allow_cpu_fallback"));
}

#[test]
fn oracle_and_device_fixtures_cover_values_validity_gaps_nonfinite_and_boundaries() {
    let oracle = read("crates/neoethos-data/tests/resident_quant_v3_oracle.rs");
    let device = read("crates/neoethos-gpu-cuda/src/resident_quant_v3_device_fixture.rs");
    for required in [
        "ordinary_m30_fixture",
        "exact_grid_gap_fixture",
        "nonfinite_input_fails_closed",
        "positive_prices_below_the_legacy_floor_keep_zero_values_without_widening_validity",
        "flat_close_transition_fixture",
        "kyle_lambda_keeps_rust_signum_value_denominator_and_zero_delta_validity",
        "utc_day_boundary_fixture",
        "five_day_week_boundary_fixture",
        "asian_session_orb_boundary_fixture",
        "assert_bitwise_preserved_v2_routes",
        "assert_migrated_v3_routes",
        "rtx_device_fixture_profiles_the_single_lane_bounded_schedule",
        "NEOETHOS_QUANT_PERF_ROWS",
        "canonical quiet NaN",
    ] {
        assert!(
            oracle.contains(required),
            "oracle fixture omitted `{required}`"
        );
    }
    for required in [
        "cfg(feature = \"cuda-device-fixtures\")",
        "run_resident_quant_v3_device_fixture",
        "run_resident_quant_v3_device_perf_fixture",
        "zero feature D2H",
        "copy_to",
        "test-only parity D2H",
    ] {
        assert!(
            device.contains(required),
            "device fixture omitted `{required}`"
        );
    }
}

#[test]
fn device_fixtures_check_the_usize_to_u64_row_boundary() {
    let device = read("crates/neoethos-gpu-cuda/src/resident_quant_v3_device_fixture.rs");
    assert_eq!(
        device
            .matches("let row_count = u64::try_from(close.len())")
            .count(),
        2,
        "parity and performance fixtures must each check the row-count conversion"
    );
    assert_eq!(
        device
            .matches("Quant fixture row count exceeds u64")
            .count(),
        2,
        "both checked conversions must propagate through the fixture Result"
    );
    assert!(!device.contains("close.len() as u64"));
    assert!(!device.contains("u64::try_from(close.len()).unwrap"));
}

#[test]
fn vol_ratio_oracle_and_cuda_share_the_exact_log_zero_denominator_predicate() {
    let oracle = read("crates/neoethos-data/tests/resident_quant_v3_oracle.rs");
    let native = read("crates/neoethos-gpu-cuda/native/resident_quant_v3.cu");
    let census = read("crates/neoethos-gpu-cuda/src/resident_quant_v3_census.rs");

    for required in [
        "let mut vol_ratio_validity = fixed_validity(rows, 20);",
        "vol_ratio_validity[row] = FeatureCellValidity::ZeroDenominator;",
        "replace_column(columns, \"quant_vol_ratio\", vol_ratio, vol_ratio_validity)?;",
        "vol_ratio_boundary_uses_exact_migrated_log_validity",
    ] {
        assert!(
            oracle.contains(required),
            "Quant-v3 CPU oracle omitted `{required}`"
        );
    }
    assert!(!oracle.contains("cloned_validity(columns, \"quant_vol_ratio\")?"));
    for required in [
        "for (std::size_t row = 20U; row < rows; ++row)",
        "if (long_squared <= kValidityEpsilonV3)",
        "set_invalid_v3(launch, 18, row, kZeroDenominatorV3)",
    ] {
        assert!(
            native.contains(required),
            "Quant-v3 CUDA slot 18 omitted `{required}`"
        );
    }
    assert!(census.contains("FixedWarmupOrZeroDenominator"));
}

#[test]
fn autocorrelation_oracle_and_cuda_share_the_exact_log_zero_denominator_predicate() {
    let oracle = read("crates/neoethos-data/tests/resident_quant_v3_oracle.rs");
    let native = read("crates/neoethos-gpu-cuda/native/resident_quant_v3.cu");

    for required in [
        "let mut autocorrelation_validity = fixed_validity(rows, 50 + lag);",
        "autocorrelation_validity[row] = FeatureCellValidity::ZeroDenominator;",
        "replace_column(columns, &name, autocorrelation, autocorrelation_validity)?;",
        "autocorrelation_boundaries_use_exact_migrated_log_validity",
    ] {
        assert!(
            oracle.contains(required),
            "Quant-v3 autocorrelation oracle omitted `{required}`"
        );
    }
    assert!(!oracle.contains("cloned_validity(columns, &name)?"));
    for required in [
        "const int autocorrelation_lags[3] = {1, 5, 10};",
        "for (std::size_t row = 50U + lag; row < rows; ++row)",
        "if (denominator <= kValidityEpsilonV3)",
        "set_invalid_v3(launch, slot, row, kZeroDenominatorV3)",
    ] {
        assert!(
            native.contains(required),
            "Quant-v3 CUDA autocorrelation omitted `{required}`"
        );
    }
}

#[test]
fn remaining_exact_log_validity_audit_closes_only_derived_schedule_seams() {
    let oracle = read("crates/neoethos-data/tests/resident_quant_v3_oracle.rs");
    let native = read("crates/neoethos-gpu-cuda/native/resident_quant_v3.cu");

    for required in [
        "let rows = 140;",
        "let mut hurst_validity = fixed_validity(rows, 100);",
        "hurst_validity[row] = FeatureCellValidity::ZeroDenominator;",
        "let mut skewness_validity = fixed_validity(rows, 30);",
        "let mut kurtosis_validity = fixed_validity(rows, 30);",
        "skewness_validity[row] = FeatureCellValidity::ZeroDenominator;",
        "kurtosis_validity[row] = FeatureCellValidity::ZeroDenominator;",
        "remaining_exact_log_derived_validities_use_exact_migrated_intermediates",
    ] {
        assert!(
            oracle.contains(required),
            "remaining Quant-v3 CPU validity audit omitted `{required}`"
        );
    }
    for mixed_authority in [
        "cloned_validity(columns, \"quant_hurst_100\")?",
        "cloned_validity(columns, \"quant_skewness_30\")?",
        "cloned_validity(columns, \"quant_kurtosis_30\")?",
    ] {
        assert!(
            !oracle.contains(mixed_authority),
            "derived exact-log validity still clones legacy authority `{mixed_authority}`"
        );
    }
    for schedule_independent in [
        "cloned_validity(columns, \"quant_log_return\")?",
        "cloned_validity(columns, \"quant_log_volatility\")?",
        "cloned_validity(columns, \"quant_fractal_dim\")?",
    ] {
        assert!(
            oracle.contains(schedule_independent),
            "raw-input validity boundary drifted `{schedule_independent}`"
        );
    }
    assert_eq!(oracle.matches("cloned_validity(columns").count(), 3);

    for required in [
        "if (deviation <= kValidityEpsilonV3 ||",
        "set_invalid_v3(launch, 19, row, kZeroDenominatorV3)",
        "if (standard_deviation <= kValidityEpsilonV3)",
        "set_invalid_v3(launch, 25, row, kZeroDenominatorV3)",
        "set_invalid_v3(launch, 26, row, kZeroDenominatorV3)",
        "const double range = sub_rn_v3(launch.high[row], launch.low[row]);",
        "set_invalid_v3(launch, 9, row, kZeroDenominatorV3)",
        "const double range = sub_rn_v3(maximum, minimum);",
        "set_invalid_v3(launch, 57, row, kZeroDenominatorV3)",
    ] {
        assert!(
            native.contains(required),
            "Quant-v3 CUDA validity schedule omitted `{required}`"
        );
    }
}

#[test]
fn cumulative_delta_preserves_the_split_value_and_validity_schedules() {
    let oracle = read("crates/neoethos-data/tests/resident_quant_v3_oracle.rs");
    let native = read("crates/neoethos-gpu-cuda/native/resident_quant_v3.cu");
    let compact_native = compact(&native);
    let census = read("crates/neoethos-gpu-cuda/src/resident_quant_v3_census.rs");

    for required in [
        "cum_delta_zscore_preserves_floor_aware_values_and_unfloored_validity",
        "legacy.values[row].to_bits() == 0.0_f64.to_bits()",
        "legacy.validity[row] == FeatureCellValidity::Valid",
        "let mut mismatch_census = Vec::new();",
        "value_bits",
        "validity_code",
        "mismatch_census.join(\"\\n\")",
    ] {
        assert!(
            oracle.contains(required),
            "Quant-v3 oracle/parity census omitted `{required}`"
        );
    }

    let device_test = oracle
        .split("fn rtx_device_fixture_matches_all_quant_v3_value_bits_and_validity_codes()")
        .nth(1)
        .expect("real-device Quant-v3 parity test")
        .split("fn rtx_device_fixture_profiles_the_single_lane_bounded_schedule()")
        .next()
        .expect("end of real-device Quant-v3 parity test");
    assert_eq!(
        device_test.matches("assert!(").count(),
        1,
        "device parity must report one complete mismatch census"
    );
    assert!(
        !device_test.contains("assert_eq!("),
        "device parity must not stop on the first route mismatch"
    );

    for required in [
        "doublevalidity_cumulative_delta=0.0;",
        "doublevalidity_cumulative_ring[50]={0.0};",
        "validity_cumulative_delta=add_rn_v3(validity_cumulative_delta,validity_delta);",
        "validity_cumulative_ring[index%50U]",
        "value_standard_deviation>kValueFloorV3",
        "validity_standard_deviation<=kValidityEpsilonV3",
    ] {
        assert!(
            compact_native.contains(required),
            "Quant-v3 CUDA cumulative-delta compatibility schedule omitted `{required}`"
        );
    }

    let route = census
        .split("\"quant_cum_delta_zscore\"")
        .nth(1)
        .expect("cumulative-delta census route");
    assert!(route.contains("V2BitwisePreserved"));
    assert!(route.contains("CumulativeDeltaPrefixOrZeroDenominator"));
}

#[test]
fn verified_release_receipt_is_pinned_before_exact_capability_minting() {
    let owner = read("crates/neoethos-gpu-cuda/src/resident_quant_v3.rs");
    let receipt =
        read("crates/neoethos-gpu-cuda/tests/fixtures/resident_quant_v3_device_parity.release.txt");

    for required in [
        "RESIDENT_QUANT_VERIFIED_RELEASE_RECEIPT_SHA256_V3",
        "0x0c, 0x26, 0x97, 0xa8",
        "Sha256::digest(device_receipt.as_bytes())",
        "resident Quant-v3 capability refuses an unpinned release receipt",
        "racecheck_log_sha256",
        "feature_d2h_bytes=0",
        "verified_release_receipt_mints_exact_quant_capability_v3",
        "capability.producer()",
        "capability.implementation_id()",
        "capability.implementation_sha256()",
        "capability.exact_math_authority()",
    ] {
        assert!(
            owner.contains(required),
            "Quant-v3 promoted capability proof omitted `{required}`"
        );
    }
    assert!(!owner.contains("held closed pending"));

    for required in [
        "verified=true",
        "cpu_cuda_value_bit_mismatches=0",
        "cpu_cuda_validity_mismatches=0",
        "compute_sanitizer_errors=0",
        "compute_sanitizer_leaked_bytes=0",
        "racecheck_log_sha256=2f392cd9a0b57e55401e28eb5681812123a66882ecb4c80d9efe0a122bc9321c",
        "kernel_launch_count=33",
        "kernel_median_ns=1071856591",
        "kernel_p95_ns=1072183163",
        "feature_d2h_bytes=0",
    ] {
        assert!(
            receipt.lines().any(|line| line == required),
            "promoted Quant-v3 release receipt omitted `{required}`"
        );
    }
}
