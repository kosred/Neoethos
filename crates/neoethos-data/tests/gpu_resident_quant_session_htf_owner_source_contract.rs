use std::fs;
use std::path::{Path, PathBuf};

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("neoethos-data lives under crates/")
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
fn quant_v4_owner_freezes_all_63_routes_and_typed_temporal_inputs() {
    let source = read("crates/neoethos-data/src/core/gpu_resident_quant_v3.rs");
    let compact_source = compact(&source);
    for required in [
        "RESIDENT_QUANT_FEATURE_SEMANTIC_VERSION_V4: u32",
        "RESIDENT_QUANT_COLUMN_NAMES_V3: [&str; 63]",
        "TRADING_SESSIONS_PER_YEAR_V3",
        "annualization_periods_per_year",
        "bars_per_asian_session",
        "bars_per_utc_day",
        "bars_per_trading_week",
        "ResidentFeatureProducerV3::Quant",
        "ResidentFeatureStageV3::Derived",
        "neoethos.data.resident-quant-route.semantic-v4",
        "ResidentProducerDraftV4::from_owner_preflight",
        "feature_value_d2h_bytes: 0",
        "validate_native_receipt",
        "native_capability.implementation_id() == RESIDENT_QUANT_IMPLEMENTATION_ID_V4",
        "native_capability.exact_math_authority() == RESIDENT_QUANT_EXACT_MATH_AUTHORITY_V4",
        "RESIDENT_QUANT_V3_TO_V4_MIGRATION_POLICY",
    ] {
        assert!(
            source.contains(required),
            "Quant-v3 owner omitted `{required}`"
        );
    }
    assert!(
        compact_source.contains("ResidentProducerBatchDraftV4::from_owner_preflight(0,63,0,0,)")
    );
    for name in [
        "quant_realized_vol_5",
        "quant_gk_vol_20",
        "quant_parkinson_vol_20",
        "quant_prev_day_h_dist",
        "quant_prev_week_l_dist",
        "quant_orb_4",
        "quant_orb_8",
        "quant_orb_12",
        "quant_pivot_dist",
        "quant_cam_s3_dist",
    ] {
        assert!(
            source.contains(name),
            "Quant-v3 route census omitted {name}"
        );
    }
    assert!(!source.contains("FeatureCellValidity::MissingInput"));
    assert!(!source.contains("#[derive(Clone"));
    assert!(!source.contains("#[derive(Copy"));
}

#[test]
fn session_v2_owner_is_atomic_sequential_and_keeps_the_dual_clock_contract() {
    let source = read("crates/neoethos-data/src/core/gpu_resident_session_v2.rs");
    let compact_source = compact(&source);
    for required in [
        "RESIDENT_SESSION_SEMANTIC_VERSION_V2: u32 = 2",
        "RESIDENT_SESSION_COLUMN_NAMES_V2: [&str; 23]",
        "RESIDENT_SESSION_RETAINED_BYTES_PER_ROW_V2: u64 = 207",
        "RESIDENT_SESSION_POINTER_TABLE_DEVICE_BYTES_V2: u64 = 736",
        "RESIDENT_SESSION_ISOLATED_POINTER_SCHEMA_BYTES_V2: u64 = 1_377",
        "infer_timestamp_unit(timestamps) != Some(TimestampUnit::Milliseconds)",
        "neoethos.data.resident-session-route.semantic-v2",
        "ResidentFeatureProducerV3::Session",
        "native_launch_count: 1",
        "producer_ready_event_count: 1",
        "producer_ready_event_synchronize_count: 0",
        "feature_value_d2h_bytes: 0",
        "logical_validity_codes: [0, 1, 5]",
        "validate_native_receipt",
        "preflight_current_native_resident_session_v2",
        "PreparedCurrentNativeResidentSessionProducerV2",
        "seal_resident_session_source_closure_v2",
        "resident_session_capability_v2",
        "ResidentSessionLaunchAuthorityV2::seal",
        "append_resident_session_v2",
        "validate_native_receipt",
    ] {
        assert!(
            source.contains(required),
            "Session-v2 owner omitted `{required}`"
        );
    }
    assert!(
        compact_source.contains("ResidentProducerBatchDraftV4::from_owner_preflight(0,23,0,0,)")
    );
    assert!(!source.contains("FeatureCellValidity::MissingInput"));
    assert!(!source.contains("#[derive(Clone"));
    assert!(!source.contains("#[derive(Copy"));
}

#[test]
fn htf_owner_retains_opaque_parents_and_binds_causal_alignment_per_route() {
    let source =
        read("crates/neoethos-data/src/core/gpu_resident_higher_timeframe_alignment_v3.rs");
    for required in [
        "HIGHER_TIMEFRAME_ALIGNMENT_SEMANTIC_VERSION_V3: u32 = 3",
        "RetainedResidentHigherTimeframeParentV3<P>",
        "parent: P",
        "selected_parent_order",
        "canonical_cpu_producer_order",
        "ResidentFeatureProducerV3::HigherTimeframeAlignment",
        "ResidentFeatureStageV3::HigherTimeframeAligned",
        "fixed_open_plus_period_v1",
        "next_direct_bar_open_v1",
        "availability_lag_ms",
        "max_age_ms",
        "retained_parent_device_bytes",
        "feature_value_d2h_bytes: 0",
        "validate_native_receipt",
        "bind_captured_parents_v3",
        "parent_context_process_token",
        "parent_stream_process_token",
    ] {
        assert!(source.contains(required), "HTF owner omitted `{required}`");
    }
    assert!(!source.contains("impl<P: Clone>"));
    assert!(!source.contains("#[derive(Clone"));
    assert!(!source.contains("#[derive(Copy"));
}

#[test]
fn quant_and_session_move_launch_authority_into_htf_but_retain_receipt_admission() {
    let quant = read("crates/neoethos-data/src/core/gpu_resident_quant_v3.rs");
    for required in [
        "PendingResidentQuantHigherTimeframeParentV3",
        "ResidentQuantHigherTimeframeBatchMemoryV3",
        "into_higher_timeframe_parent_v3",
        "higher_timeframe_batch_memory_v3",
        "take_launch_authority_v3",
        "validate_captured_parent_receipt_v3",
        "runtime_admission: Option<ResidentQuantRuntimeAdmissionV3>",
        "launch_authority: Option<ResidentQuantLaunchAuthorityV3>",
        "validate_native_receipt(receipt)",
    ] {
        assert!(
            quant.contains(required),
            "Quant HTF split omitted `{required}`"
        );
    }

    let session = read("crates/neoethos-data/src/core/gpu_resident_session_v2.rs");
    for required in [
        "PendingResidentSessionHigherTimeframeParentV2",
        "ResidentSessionHigherTimeframeBatchMemoryV2",
        "into_higher_timeframe_parent_v2",
        "higher_timeframe_batch_memory_v2",
        "take_launch_authority_v2",
        "validate_captured_parent_receipt_v2",
        "runtime_admission: Option<ResidentSessionRuntimeAdmissionV2>",
        "launch_authority: Option<ResidentSessionLaunchAuthorityV2>",
        "validate_native_receipt(receipt)",
    ] {
        assert!(
            session.contains(required),
            "Session HTF split omitted `{required}`"
        );
    }
}

#[test]
fn temporal_grid_fails_closed() {
    let temporal = read("crates/neoethos-data/src/core/gpu_resident_temporal_grid_v1.rs");
    for required in [
        "TRADING_SESSIONS_PER_YEAR_V3: u64 = 252",
        "ASIAN_SESSION_MILLIS_V2",
        "UTC_DAY_MILLIS_V2",
        "TRADING_DAYS_PER_WEEK_V3",
        "at least twelve bars",
        "timestamp.rem_euclid(timeframe_millis) != 0",
        "gap.rem_euclid(timeframe_millis) != 0",
    ] {
        assert!(
            temporal.contains(required),
            "temporal owner omitted `{required}`"
        );
    }
}

#[test]
fn owner_modules_do_not_mint_global_ordinals_plans_or_source_provenance() {
    let sources = [
        read("crates/neoethos-data/src/core/gpu_resident_quant_v3.rs"),
        read("crates/neoethos-data/src/core/gpu_resident_session_v2.rs"),
        read("crates/neoethos-data/src/core/gpu_resident_higher_timeframe_alignment_v3.rs"),
    ];
    for source in sources {
        for forbidden in [
            "ResidentFeatureRouteV3::new",
            "FeaturePlanV1::new",
            "DatasetFeatureArtifactProvenanceV1::new",
            "global_column",
        ] {
            assert!(
                !source.contains(forbidden),
                "owner minted forbidden `{forbidden}`"
            );
        }
    }
}
