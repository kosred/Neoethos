//! RED-first contracts for the approved Regime semantic-v3 design.
//!
//! The fixture and source contracts were frozen before production work. They
//! now bind the CPU, CUDA, receipt and fail-closed migration implementation to
//! the same reviewed identities and exact operation schedule.

use neoethos_data::{Ohlcv, REGIME_FEATURE_NAMES_V3, compute_regime_feature_columns_f64};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

const SPEC_PATH: &str = "docs/regime-semantic-v3.md";
const FIXTURE_PATH: &str = "crates/neoethos-data/tests/fixtures/regime_semantic_v3_red_v1.json";
const FIXTURE_SHA256: &str = "f0f89c26727e90206bb85bdb4b3f6e11f59652176f7ba8475e9fbaa301548a93";
const OPERATION_SCHEDULE: &str = "neoethos.regime.semantic-v3.f64-rn-fixed-order-log49-neumaier-v1";
const CANONICAL_NAN_BITS: u64 = 0x7ff8_0000_0000_0000;
const LN_2_BITS: u64 = 0x3fe6_2e42_fefa_39ef;
const LN_10_BITS: u64 = 0x4002_6bb1_bbb5_5515;
const GK_COEFFICIENT_BITS: u64 = 0x3fd8_b90b_fbe8_e7bc;
const LOG49_MIRROR_OPERATION_TOKENS_V1: &str = "neoethos.regime.log49-mirror.v1|subnormal-scale=0x4350000000000000|mantissa-mask=0x000fffffffffffff|one-exponent=0x3ff0000000000000|ln2=0x3fe62e42fefa39ef|series-odd=3..49|order=normalize,bits,exponent,mantissa,z,z2,term,sum,loop,return|rounding=rn-no-fma";
const LOG49_MIRROR_OPERATION_TOKENS_SHA256_V1: &str =
    "73002b6761d1ca425250a761fa4411cf3ae0d26c862caa964e93063c69c32080";
const LOG49_RUST_MIRROR_SHA256_V1: &str =
    "f7d83af4d95a95c38cb360abcee96a223f4010aba2e3c679145c091e56db8fea";
const LOG49_CUDA_MIRROR_SHA256_V1: &str =
    "ec8299d718d7a3d5a189287380f042df603fde7bbed87b7378845d7ce73618fe";

const V3_NAMES: [&str; 14] = [
    "neoethos_custom_gk_vol_ratio_state_10_50_v3",
    "neoethos_custom_gk_vol_ratio_offset_10_50_v3",
    "regime_wilder_adx_14_v3",
    "neoethos_custom_wilder_di_dominance_direction_14_v3",
    "neoethos_custom_wilder_adx_direction_state_14_25_v3",
    "neoethos_custom_bollinger_keltner_squeeze_state_20_2_1p5_v3",
    "neoethos_custom_bollinger_midline_atr_deviation_20_v3",
    "neoethos_custom_directional_persistence_balance_20_v3",
    "neoethos_custom_candle_body_range_balance_8_v3",
    "regime_dreiss_choppiness_index_14_v3",
    "neoethos_custom_standardized_cusum_up_50_0p5_3_v3",
    "neoethos_custom_standardized_cusum_down_50_0p5_3_v3",
    "neoethos_custom_standardized_cusum_signal_50_0p5_3_v3",
    "neoethos_custom_equal_width_log_return_entropy_30_10_v3",
];

const V2_NAMES: [&str; 14] = [
    "regime_vol_state",
    "regime_vol_zscore",
    "regime_trend_strength",
    "regime_trend_direction",
    "regime_trend_state",
    "regime_squeeze",
    "regime_squeeze_momentum",
    "regime_mr_vs_momentum",
    "regime_rei",
    "regime_choppiness",
    "regime_cusum_up",
    "regime_cusum_down",
    "regime_change_signal",
    "regime_entropy",
];

const FIRST_THEORETICAL_ROWS: [u64; 14] = [49, 49, 27, 14, 27, 20, 20, 21, 7, 14, 50, 50, 50, 30];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("neoethos-data must live two levels below the workspace root")
        .to_path_buf()
}

fn read(path: &str) -> String {
    let absolute = workspace_root().join(path);
    fs::read_to_string(&absolute)
        .unwrap_or_else(|error| panic!("read {}: {error}", absolute.display()))
}

fn read_or_empty(path: &str) -> String {
    fs::read_to_string(workspace_root().join(path)).unwrap_or_default()
}

fn fixture_bytes() -> Vec<u8> {
    let absolute = workspace_root().join(FIXTURE_PATH);
    fs::read(&absolute).unwrap_or_else(|error| panic!("read {}: {error}", absolute.display()))
}

fn fixture() -> Value {
    serde_json::from_slice(&fixture_bytes()).expect("Regime semantic-v3 fixture must be JSON")
}

fn case<'a>(fixture: &'a Value, id: &str) -> &'a Value {
    fixture["cases"]
        .as_array()
        .expect("fixture cases array")
        .iter()
        .find(|candidate| candidate["id"].as_str() == Some(id))
        .unwrap_or_else(|| panic!("missing fixture case {id}"))
}

fn parse_bits(value: &Value) -> f64 {
    let text = value.as_str().expect("f64 bits must be a string");
    let bits = u64::from_str_radix(
        text.strip_prefix("0x")
            .expect("f64 bits must start with 0x"),
        16,
    )
    .expect("valid hexadecimal f64 bits");
    f64::from_bits(bits)
}

fn ln_positive_exact_v1(value: f64) -> f64 {
    assert!(value.is_finite() && value > 0.0);
    let (normalized, exponent_adjustment) = if value.is_subnormal() {
        (value * f64::from_bits(0x4350_0000_0000_0000), -54)
    } else {
        (value, 0)
    };
    let bits = normalized.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023 + exponent_adjustment;
    let mantissa = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    let z = (mantissa - 1.0) / (mantissa + 1.0);
    let z_squared = z * z;
    let mut term = z;
    let mut sum = z;
    let mut denominator = 3_u32;
    while denominator <= 49 {
        term = term * z_squared;
        sum = sum + term / denominator as f64;
        denominator += 2;
    }
    exponent as f64 * f64::from_bits(LN_2_BITS) + 2.0 * sum
}

fn log10_positive_exact_v1(value: f64) -> f64 {
    ln_positive_exact_v1(value) / f64::from_bits(LN_10_BITS)
}

fn neumaier_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;
    for value in values {
        let next = sum + value;
        if sum.abs() >= value.abs() {
            compensation += (sum - next) + value;
        } else {
            compensation += (value - next) + sum;
        }
        sum = next;
    }
    sum + compensation
}

fn cusum_step(previous_up: f64, previous_down: f64, z: f64) -> (f64, f64, f64) {
    let candidate_up = ((previous_up + z) - 0.5).max(0.0);
    let candidate_down = ((previous_down - z) - 0.5).max(0.0);
    if candidate_up > 3.0 {
        (0.0, candidate_down, 1.0)
    } else if candidate_down > 3.0 {
        (candidate_up, 0.0, -1.0)
    } else {
        (candidate_up, candidate_down, 0.0)
    }
}

#[test]
fn fixture_hash_schema_and_exact_slot_order_are_frozen() {
    let bytes = fixture_bytes();
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), FIXTURE_SHA256);

    let fixture: Value = serde_json::from_slice(&bytes).expect("valid fixture JSON");
    assert_eq!(
        fixture["schema"].as_str(),
        Some("neoethos.regime-semantic-v3.red-fixtures.v1")
    );
    assert_eq!(fixture["semantic_version"].as_u64(), Some(3));
    assert_eq!(
        fixture["operation_schedule"].as_str(),
        Some(OPERATION_SCHEDULE)
    );
    assert_eq!(
        parse_bits(&fixture["canonical_nan_bits"]).to_bits(),
        CANONICAL_NAN_BITS
    );

    let columns = fixture["columns"].as_array().expect("columns array");
    assert_eq!(columns.len(), 14);
    let mut identities = HashSet::new();
    for (slot, column) in columns.iter().enumerate() {
        assert_eq!(column["slot"].as_u64(), Some(slot as u64));
        assert_eq!(column["retired_v2_name"].as_str(), Some(V2_NAMES[slot]));
        assert_eq!(column["semantic_v3_name"].as_str(), Some(V3_NAMES[slot]));
        assert_eq!(
            column["first_theoretical_row"].as_u64(),
            Some(FIRST_THEORETICAL_ROWS[slot])
        );
        assert!(
            identities.insert(
                column["formula_identity"]
                    .as_str()
                    .expect("formula identity")
            ),
            "duplicate formula identity at slot {slot}"
        );
    }

    for (slot, name) in V3_NAMES.iter().enumerate() {
        assert!(name.ends_with("_v3"), "slot {slot} lacks v3 identity");
        if slot != 2 && slot != 9 {
            assert!(
                name.starts_with("neoethos_custom_"),
                "non-creator slot {slot} is not explicitly custom"
            );
        }
        for misleading in ["zscore", "_rei_", "ttm", "mean_reversion"] {
            assert!(
                !name.contains(misleading),
                "v3 slot {slot} retained misleading token {misleading}"
            );
        }
    }

    let name_bytes = V3_NAMES.iter().map(|name| name.len()).sum::<usize>();
    assert_eq!(name_bytes, 667);
    assert_eq!(14 * 4 * 8 + 15 * 8 + name_bytes, 1_235);
}

#[test]
fn fixture_covers_short_nonfinite_constant_gap_recurrence_and_entropy_boundaries() {
    let fixture = fixture();
    let ids = fixture["cases"]
        .as_array()
        .expect("cases array")
        .iter()
        .map(|case| case["id"].as_str().expect("case id"))
        .collect::<HashSet<_>>();
    for required in [
        "empty_input_refuses_before_allocation",
        "nonfinite_close_refuses_before_allocation",
        "monotone_fifteen_rows_exposes_v2_validity_defect",
        "constant_sixty_rows_freezes_denominators",
        "garman_klass_primary_component_exact_schedule",
        "bollinger_population_not_sample_variance",
        "dreiss_gap_uses_true_high_and_true_low",
        "cusum_strict_threshold_equality_does_not_hit",
        "cusum_up_hit_emits_post_reset_state",
        "cusum_down_hit_emits_post_reset_state",
        "constant_log_returns_have_valid_zero_entropy",
    ] {
        assert!(ids.contains(required), "fixture omitted {required}");
    }

    let fifteen = case(&fixture, "monotone_fifteen_rows_exposes_v2_validity_defect");
    let row_14 = fifteen["expect"]
        .as_array()
        .expect("15-row expectations")
        .iter()
        .filter(|expectation| expectation["row"].as_u64() == Some(14))
        .collect::<Vec<_>>();
    assert_eq!(row_14.len(), 3);
    assert_eq!(row_14[0]["validity"].as_str(), Some("warmup"));
    assert_eq!(row_14[1]["validity"].as_str(), Some("valid"));
    assert_eq!(
        parse_bits(&row_14[1]["value_bits"]).to_bits(),
        1.0_f64.to_bits()
    );
    assert_eq!(row_14[2]["validity"].as_str(), Some("warmup"));

    let columns = fixture["columns"].as_array().expect("columns array");
    for (slot, column) in columns.iter().enumerate() {
        let constant = &column["constant_at_first"];
        if slot == 13 {
            assert_eq!(constant["validity"].as_str(), Some("valid"));
            assert_eq!(parse_bits(&constant["value_bits"]).to_bits(), 0);
        } else {
            assert_eq!(
                constant["validity"].as_str(),
                Some("zero_denominator"),
                "constant slot {slot}"
            );
            assert_eq!(
                parse_bits(&constant["value_bits"]).to_bits(),
                CANONICAL_NAN_BITS
            );
        }
    }
}

#[test]
fn fixture_formula_bits_match_the_frozen_independent_schedule() {
    let fixture = fixture();

    let gk = case(&fixture, "garman_klass_primary_component_exact_schedule");
    let open = parse_bits(&gk["input"]["open_bits"]);
    let high = parse_bits(&gk["input"]["high_bits"]);
    let low = parse_bits(&gk["input"]["low_bits"]);
    let close = parse_bits(&gk["input"]["close_bits"]);
    let u = ln_positive_exact_v1(high) - ln_positive_exact_v1(open);
    let d = ln_positive_exact_v1(low) - ln_positive_exact_v1(open);
    let c = ln_positive_exact_v1(close) - ln_positive_exact_v1(open);
    let range = u - d;
    let component = 0.5 * (range * range) - f64::from_bits(GK_COEFFICIENT_BITS) * (c * c);
    assert_eq!(
        component.to_bits(),
        parse_bits(&gk["expected_component_bits"]).to_bits()
    );
    assert!(component >= 0.0);

    let bollinger = case(&fixture, "bollinger_population_not_sample_variance");
    let closes = (1_u32..=20).map(f64::from).collect::<Vec<_>>();
    let mean = neumaier_sum(closes.iter().copied()) / 20.0;
    let variance = neumaier_sum(closes.iter().map(|value| {
        let deviation = *value - mean;
        deviation * deviation
    })) / 20.0;
    let std = variance.sqrt();
    assert_eq!(
        mean.to_bits(),
        parse_bits(&bollinger["expected_mean_bits"]).to_bits()
    );
    assert_eq!(
        variance.to_bits(),
        parse_bits(&bollinger["expected_population_variance_bits"]).to_bits()
    );
    assert_eq!(
        std.to_bits(),
        parse_bits(&bollinger["expected_population_std_bits"]).to_bits()
    );
    assert_ne!(
        variance.to_bits(),
        parse_bits(&bollinger["rejected_sample_variance_bits"]).to_bits()
    );

    let gap = case(&fixture, "dreiss_gap_uses_true_high_and_true_low");
    let sum_true_range = parse_bits(&gap["sum_true_range_bits"]);
    let true_range = parse_bits(&gap["true_high_minus_true_low_bits"]);
    let ratio = sum_true_range / true_range;
    let value = (100.0 * log10_positive_exact_v1(ratio)) / log10_positive_exact_v1(14.0);
    assert_eq!(
        value.to_bits(),
        parse_bits(&gap["expected_value_bits"]).to_bits()
    );
    let rejected_ratio =
        sum_true_range / parse_bits(&gap["rejected_raw_high_low_denominator_bits"]);
    let rejected =
        (100.0 * log10_positive_exact_v1(rejected_ratio)) / log10_positive_exact_v1(14.0);
    assert_eq!(
        rejected.to_bits(),
        parse_bits(&gap["rejected_raw_formula_value_bits"]).to_bits()
    );
    assert_ne!(value.to_bits(), rejected.to_bits());

    for id in [
        "cusum_strict_threshold_equality_does_not_hit",
        "cusum_up_hit_emits_post_reset_state",
        "cusum_down_hit_emits_post_reset_state",
    ] {
        let recurrence = case(&fixture, id);
        let actual = cusum_step(
            parse_bits(&recurrence["input"]["previous_up_bits"]),
            parse_bits(&recurrence["input"]["previous_down_bits"]),
            parse_bits(&recurrence["input"]["z_bits"]),
        );
        let expected = &recurrence["expected"];
        assert_eq!(
            actual.0.to_bits(),
            parse_bits(&expected["up_bits"]).to_bits(),
            "{id} up"
        );
        assert_eq!(
            actual.1.to_bits(),
            parse_bits(&expected["down_bits"]).to_bits(),
            "{id} down"
        );
        assert_eq!(
            actual.2.to_bits(),
            parse_bits(&expected["signal_bits"]).to_bits(),
            "{id} signal"
        );
    }
}

#[test]
fn red_log49_rust_cuda_mirror_is_source_hash_sealed() {
    let rust = read_or_empty("crates/neoethos-data/src/core/regime_exact_math_v1.rs");
    let native = read_or_empty("crates/neoethos-gpu-cuda/native/resident_regime_v3.cu");

    for source in [&rust, &native] {
        assert!(source.contains(LOG49_MIRROR_OPERATION_TOKENS_V1));
        assert!(source.contains(LOG49_MIRROR_OPERATION_TOKENS_SHA256_V1));
        for token in [
            "0x4350000000000000",
            "0x000fffffffffffff",
            "0x3ff0000000000000",
            "0x3fe62e42fefa39ef",
            "denominator",
            "49",
        ] {
            assert!(source.contains(token), "log49 mirror omitted {token:?}");
        }
    }

    assert!(rust.contains("REGIME_LOG49_RUST_MIRROR_BEGIN_V1"));
    assert!(rust.contains("REGIME_LOG49_RUST_MIRROR_END_V1"));
    assert!(rust.contains("REGIME_LOG49_RUST_MIRROR_SHA256_V1"));
    assert!(native.contains("REGIME_LOG49_CUDA_MIRROR_BEGIN_V1"));
    assert!(native.contains("REGIME_LOG49_CUDA_MIRROR_END_V1"));
    assert!(native.contains("kRegimeLog49CudaMirrorSha256V1"));

    let rust_body = rust
        .split_once("// REGIME_LOG49_RUST_MIRROR_BEGIN_V1")
        .and_then(|(_, rest)| rest.split_once("// REGIME_LOG49_RUST_MIRROR_END_V1"))
        .map(|(body, _)| body)
        .expect("marked Rust log49 mirror");
    let cuda_body = native
        .split_once("// REGIME_LOG49_CUDA_MIRROR_BEGIN_V1")
        .and_then(|(_, rest)| rest.split_once("// REGIME_LOG49_CUDA_MIRROR_END_V1"))
        .map(|(body, _)| body)
        .expect("marked CUDA log49 mirror");
    assert_eq!(
        format!("{:x}", Sha256::digest(rust_body.as_bytes())),
        LOG49_RUST_MIRROR_SHA256_V1
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(cuda_body.as_bytes())),
        LOG49_CUDA_MIRROR_SHA256_V1
    );
    assert!(rust.contains(LOG49_RUST_MIRROR_SHA256_V1));
    assert!(native.contains(LOG49_CUDA_MIRROR_SHA256_V1));
}

#[test]
fn red_cpu_v3_must_replace_the_untrusted_legacy_bridge() {
    let cpu = read("crates/neoethos-data/src/core/regime_detection.rs");
    for token in [
        "pub const REGIME_SEMANTIC_VERSION: u32 = 3;",
        "pub const REGIME_FEATURE_NAMES_V3: [&str; 14]",
        "pub const REGIME_OPERATION_SCHEDULE_V1: &str",
        "pub const REGIME_SEMANTIC_V3_FIXTURE_SHA256: &str",
        "pub const REGIME_V2_ARTIFACT_MIGRATION_POLICY: &str",
        "refuse semantic-v2 Regime artifacts",
        OPERATION_SCHEDULE,
        FIXTURE_SHA256,
        "neoethos_ln_positive_exact_v1",
        "REGIME_CANONICAL_NAN_BITS_V3",
        "RegimeInputRefusalV3",
        "ScaleRangeUnsupported",
        "ordered_neumaier_sum_v1",
    ] {
        assert!(cpu.contains(token), "Regime CPU v3 omitted {token:?}");
    }
    for name in V3_NAMES {
        assert!(cpu.contains(name), "Regime CPU v3 omitted slot {name}");
    }

    let v3_lane = cpu
        .split_once("pub fn compute_regime_feature_columns_f64(")
        .map(|(_, body)| body)
        .expect("the explicit-validity Regime v3 CPU lane must remain source-visible");
    for forbidden in [
        "compute_regime_feature_columns(ohlcv)",
        ".abs().sqrt()",
        "1e-10",
        "1e-12",
        ".ln()",
        ".log10()",
        ".ln_1p()",
        "/ (bb_period as f64 - 1.0)",
    ] {
        assert!(
            !v3_lane.contains(forbidden),
            "Regime CPU v3 retained forbidden v2 seam {forbidden:?}"
        );
    }
}

#[test]
fn red_resident_cuda_v3_must_be_exact_f64_and_host_materialization_free() {
    let runtime = read_or_empty("crates/neoethos-gpu-cuda/src/resident_regime_v3.rs");
    let native = read_or_empty("crates/neoethos-gpu-cuda/native/resident_regime_v3.cu");

    for token in [
        "RESIDENT_REGIME_COLUMN_NAMES_V3",
        "resident_regime_capability_v3",
        "launch_resident_regime_v3",
        "ResidentParentDatasetSourceV3",
        "ResidentProducerReadyEventV3::record(",
        "retained_feature_device_bytes",
        "additional_retained_device_bytes: 0",
        "scratch_device_bytes: 0",
        "parent_input_h2d_bytes: 0",
        "feature_value_d2h_bytes: 0",
        "producer_ready_event_count: 1",
        "native_launch_count: 2",
        "126",
        "448",
        "1235",
        OPERATION_SCHEDULE,
        FIXTURE_SHA256,
    ] {
        assert!(
            runtime.contains(token),
            "resident Regime runtime omitted {token:?}"
        );
    }
    for token in [
        "neoethos_resident_regime_independent_f64_v3",
        "neoethos_resident_regime_recurrence_f64_v3",
        "kRegimeColumnsV3 = 14",
        "neoethos_ln_positive_exact_v1",
        "__dadd_rn",
        "__dsub_rn",
        "__dmul_rn",
        "__ddiv_rn",
        "__dsqrt_rn",
        "0x7ff8000000000000ULL",
        "kWarmupV3",
        "kZeroDenominatorV3",
        "kComputeFailureV3",
    ] {
        assert!(
            native.contains(token),
            "resident Regime CUDA omitted {token:?}"
        );
    }
    for forbidden in [
        "compute_regime_feature_columns",
        "FeatureColumnF64",
        "copy_to(",
        "stream.synchronize()",
        "Context::new(",
        "Stream::new(",
        "fallback_",
        "log(",
        "log10(",
        "__log",
    ] {
        assert!(
            !runtime.contains(forbidden) && !native.contains(forbidden),
            "resident Regime reintroduced forbidden seam {forbidden:?}"
        );
    }
}

#[test]
fn red_v2_artifacts_and_missing_resident_receipts_must_fail_closed() {
    let registry = read("crates/neoethos-data/src/core/feature_registry.rs");
    let data = read("crates/neoethos-data/src/core/gpu_resident_feature_store_v3.rs");
    let preflight =
        read("crates/neoethos-data/src/core/gpu_only_feature_workspace_preflight_v3.rs");
    let build = read("crates/neoethos-gpu-cuda/build.rs");
    let library = read("crates/neoethos-gpu-cuda/src/lib.rs");

    for token in [
        "super::regime_detection::REGIME_SEMANTIC_VERSION",
        "REGIME_V2_ARTIFACT_MIGRATION_POLICY",
    ] {
        assert!(registry.contains(token), "registry omitted {token:?}");
    }
    for token in [
        "resident_regime_capability_v3()?",
        "SealedRegimeComponentReceiptV3",
        "regime: seal_token.regime",
        "regime.validate_runtime_evidence(",
    ] {
        assert!(data.contains(token), "resident store omitted {token:?}");
    }
    assert!(
        !preflight.contains("ResidentFeatureProducerV3::Regime,"),
        "Regime must leave the pending census only with its sealed v3 receipt"
    );
    assert!(build.contains("native/resident_regime_v3.cu"));
    assert!(library.contains("pub mod resident_regime_v3;"));

    let spec = read(SPEC_PATH);
    for retired in V2_NAMES {
        assert!(
            spec.contains(retired),
            "migration spec omitted retired name {retired}"
        );
    }
    for forbidden_policy in ["alias v2", "automatic value conversion", "dual emission"] {
        assert!(
            !registry.contains(forbidden_policy),
            "registry added forbidden compatibility policy {forbidden_policy:?}"
        );
    }
}

fn ohlc(open: Vec<f64>, high: Vec<f64>, low: Vec<f64>, close: Vec<f64>) -> Ohlcv {
    Ohlcv {
        timestamp: None,
        open,
        high,
        low,
        close,
        volume: None,
    }
}

#[test]
fn cpu_v3_executes_the_frozen_short_constant_gap_and_refusal_fixtures() {
    let empty = ohlc(Vec::new(), Vec::new(), Vec::new(), Vec::new());
    assert_eq!(
        compute_regime_feature_columns_f64(&empty)
            .expect_err("empty Regime input must refuse")
            .to_string(),
        "RegimeInputRefusalV3::EmptyInput"
    );

    let nonfinite = ohlc(vec![100.0], vec![100.0], vec![100.0], vec![f64::NAN]);
    assert_eq!(
        compute_regime_feature_columns_f64(&nonfinite)
            .expect_err("nonfinite Regime input must refuse")
            .to_string(),
        "RegimeInputRefusalV3::NonFiniteOhlc{row:0,field:close}"
    );

    let monotone = ohlc(
        (0..15).map(|i| 100.0 + f64::from(i)).collect(),
        (0..15).map(|i| 102.0 + f64::from(i)).collect(),
        (0..15).map(|i| 99.0 + f64::from(i)).collect(),
        (0..15).map(|i| 101.0 + f64::from(i)).collect(),
    );
    let columns = compute_regime_feature_columns_f64(&monotone).expect("15-row Regime v3");
    assert_eq!(
        columns
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>(),
        REGIME_FEATURE_NAMES_V3
    );
    assert_eq!(columns[2].validity[14].as_str(), "warmup");
    assert_eq!(columns[3].validity[14].as_str(), "valid");
    assert_eq!(columns[3].values[14].to_bits(), 1.0_f64.to_bits());
    assert_eq!(columns[4].validity[14].as_str(), "warmup");

    let constant = ohlc(
        vec![100.0; 60],
        vec![100.0; 60],
        vec![100.0; 60],
        vec![100.0; 60],
    );
    let columns = compute_regime_feature_columns_f64(&constant).expect("constant Regime v3");
    for (slot, row) in FIRST_THEORETICAL_ROWS.into_iter().enumerate() {
        let row = row as usize;
        if slot == 13 {
            assert_eq!(columns[slot].validity[row].as_str(), "valid");
            assert_eq!(columns[slot].values[row].to_bits(), 0);
        } else {
            assert_eq!(columns[slot].validity[row].as_str(), "zero_denominator");
            assert_eq!(columns[slot].values[row].to_bits(), CANONICAL_NAN_BITS);
        }
    }

    let mut open = vec![110.0; 15];
    let mut high = vec![111.0; 15];
    let mut low = vec![109.0; 15];
    let mut close = vec![110.0; 15];
    open[0] = 100.0;
    high[0] = 101.0;
    low[0] = 99.0;
    close[0] = 100.0;
    let gap = ohlc(open, high, low, close);
    let columns = compute_regime_feature_columns_f64(&gap).expect("gap Regime v3");
    assert_eq!(columns[9].validity[14].as_str(), "valid");
    assert_eq!(columns[9].values[14].to_bits(), 0x4046_fb6c_35d3_f6e6);
}
