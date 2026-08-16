//! What the vocabulary restoration actually bought, measured on the real
//! on-disk store rather than estimated.
//!
//! Non-negotiable #4: restoring the vocabulary CHANGES WHAT THE SEARCH
//! EXPLORES, so the before/after column counts must be reported, and every
//! historical discovery result — produced on the 66-column vocabulary — is not
//! comparable to anything produced after this lands. This test is where that
//! number comes from.
//!
//! Skips (does not fail) when the store is absent, so a machine without the
//! data still runs the suite.
//!
//! ```text
//! cargo test -p neoethos-data --test vocabulary_restoration_measured -- --nocapture
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;

use neoethos_data::core::all_indicators::ALL_INDICATORS;
use neoethos_data::core::feature_budget::{VocabularyBudget, column_bytes};
#[cfg(feature = "gpu-cuda")]
use neoethos_data::core::hpc_ta::compute_classic_ta_columns_with_policy;
use neoethos_data::core::hpc_ta::{
    ALT_PERIODS, ClassicTaExecutionReport, IndicatorComputePolicy, MULTI_PERIOD_IDS,
    compute_classic_ta_columns_with_policy_report,
};
use neoethos_data::core::indicator_ledger::{
    EXPECTED_NON_PRODUCING, EXPECTED_NON_PRODUCING_OUTPUTS, has_finite_variation,
    series_fingerprint,
};
use serde_json::json;

/// Bars to measure on. Long enough to clear every period in `ALT_PERIODS`
/// (200 * 1.25 = 250) by three orders of magnitude, so nothing is
/// warmup-limited, and short enough to run in seconds.
const DEFAULT_BARS: usize = 200_000;

fn requested_bars() -> usize {
    std::env::var("NEOETHOS_TASK1_BARS")
        .map(|raw| {
            raw.parse::<usize>()
                .unwrap_or_else(|e| panic!("invalid NEOETHOS_TASK1_BARS={raw:?}: {e}"))
        })
        .unwrap_or(DEFAULT_BARS)
}

fn required_task1_env(name: &str) -> String {
    std::env::var(name)
        .unwrap_or_else(|error| panic!("{name} is required for auditable Task-1 evidence: {error}"))
}

#[cfg(feature = "gpu-cuda")]
fn require_task1_true_env(name: &str) {
    let value = required_task1_env(name);
    assert!(
        matches!(value.as_str(), "1" | "true" | "TRUE" | "True"),
        "{name} must force the auditable Task-1 lane, got {value:?}"
    );
}

fn duration_ns(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).expect("Task-1 duration exceeds the versioned u64-ns field")
}

/// Load the real bars, or FAIL naming the resolved path.
///
/// This used to return `Option` and the test printed "SKIPPED" and passed. That
/// is the defect this whole workstream exists to kill, wearing a test's
/// clothing: a green result that distinguishes "measured 825 columns on real
/// bars" from "measured nothing" only in stdout nobody reads. The test is
/// `#[ignore]`d instead, so it never runs where the store is absent — and when
/// it IS run, a missing store is a hard failure with the path in the message.
fn real_vortex_path() -> PathBuf {
    if let Some(path) = std::env::var_os("NEOETHOS_TASK1_VORTEX") {
        return PathBuf::from(path);
    }
    let base = std::env::var_os("LOCALAPPDATA")
        .expect("LOCALAPPDATA is unset and NEOETHOS_TASK1_VORTEX was not supplied");
    PathBuf::from(base).join("neoethos/data/symbol=EURUSD/timeframe=M5/data.vortex")
}

fn real_ohlcv() -> (neoethos_data::Ohlcv, PathBuf) {
    let path = real_vortex_path();
    let bars = requested_bars();
    assert!(bars > 0, "NEOETHOS_TASK1_BARS must be positive");
    let o = neoethos_data::load_vortex(&path)
        .unwrap_or_else(|e| panic!("no readable Vortex store at {}: {e:#}", path.display()));
    let total = o.close.len();
    assert!(
        total >= bars,
        "the store at {} holds {total} bars, fewer than the {bars} this measurement needs",
        path.display()
    );
    let lo = total - bars;
    (
        neoethos_data::Ohlcv {
            timestamp: o.timestamp.as_ref().map(|t| t[lo..].to_vec()),
            open: o.open[lo..].to_vec(),
            high: o.high[lo..].to_vec(),
            low: o.low[lo..].to_vec(),
            close: o.close[lo..].to_vec(),
            volume: o.volume.as_ref().map(|v| v[lo..].to_vec()),
        },
        path,
    )
}

fn canonical_series_equal(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(&left, &right)| {
            (left.is_nan() && right.is_nan())
                || (left == 0.0 && right == 0.0)
                || left.to_bits() == right.to_bits()
        })
}

fn write_task1_ledger(
    source_path: &std::path::Path,
    ohlcv: &neoethos_data::Ohlcv,
    columns: &[(String, Vec<f64>)],
    execution: &ClassicTaExecutionReport,
    elapsed: std::time::Duration,
) {
    let names: Vec<&str> = columns.iter().map(|(name, _)| name.as_str()).collect();
    let schema_hash =
        neoethos_core::storage::json::stable_json_hash(&names).expect("hash Task-1 feature schema");

    let mut indices_by_fingerprint: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    let mut duplicates = Vec::new();
    let mut all_non_finite = Vec::new();
    let mut constant = Vec::new();
    let mut warmup_or_gap = Vec::new();
    let mut truncated = Vec::new();
    for (index, (name, values)) in columns.iter().enumerate() {
        if values.len() != ohlcv.close.len() {
            truncated.push(json!({"name": name, "values": values.len()}));
        }

        if values.iter().any(|value| value.is_finite()) {
            if !has_finite_variation(values) {
                constant.push(name.clone());
            }
        } else {
            all_non_finite.push(name.clone());
        }

        let invalid_total = values.iter().filter(|value| !value.is_finite()).count();
        if invalid_total > 0 {
            let leading_invalid = values.iter().take_while(|value| !value.is_finite()).count();
            warmup_or_gap.push(json!({
                "name": name,
                "leading_invalid": leading_invalid,
                "invalid_total": invalid_total,
            }));
        }

        let fingerprint = series_fingerprint(values);
        let matching_index = indices_by_fingerprint
            .get(&fingerprint)
            .and_then(|indices| {
                indices.iter().copied().find(|&candidate| {
                    canonical_series_equal(columns[candidate].1.as_slice(), values.as_slice())
                })
            });
        if let Some(first_index) = matching_index {
            duplicates.push(json!({
                "name": name,
                "duplicate_of": columns[first_index].0,
            }));
            continue;
        }
        indices_by_fingerprint
            .entry(fingerprint)
            .or_default()
            .push(index);
    }

    #[cfg(feature = "gpu-cuda")]
    let claimed_cuda_sweep_ids: Vec<&str> = neoethos_data::core::gpu_indicators::GPU_SWEEP_SPECS
        .iter()
        .map(|spec| spec.id)
        .collect();
    #[cfg(not(feature = "gpu-cuda"))]
    let claimed_cuda_sweep_ids: Vec<&str> = Vec::new();
    let claimed_cuda_sweep_count = claimed_cuda_sweep_ids.len();

    let symbol = required_task1_env("NEOETHOS_TASK1_SYMBOL");
    let timeframe = required_task1_env("NEOETHOS_TASK1_TIMEFRAME");
    let source_sha256 = required_task1_env("NEOETHOS_TASK1_SOURCE_SHA256");
    let source_release = required_task1_env("NEOETHOS_TASK1_SOURCE_RELEASE");
    let source_asset_sha256 = required_task1_env("NEOETHOS_TASK1_ASSET_SHA256");
    let source_url = required_task1_env("NEOETHOS_TASK1_SOURCE_URL");
    let execution_json = json!({
        "budget_rows": execution.budget_rows,
        "budget_available_bytes_at_admission": execution.available_bytes_at_admission,
        "budget_max_columns": execution.max_columns,
        "admitted_indicator_ids": &execution.admitted_indicator_ids,
        "budget_deferred_indicator_ids": &execution.budget_deferred_indicator_ids,
        "planned_base_columns": execution.planned_base_columns,
        "admitted_base_columns": execution.admitted_base_columns,
        "historical_sweep_reserved_columns": execution.historical_sweep_reserved_columns,
        "historical_sweep_produced_columns": execution.historical_sweep_produced_columns,
        "extended_mode": execution.extended_mode,
        "extended_admitted_indicator_ids": &execution.extended_admitted_indicator_ids,
        "extended_budget_deferred_indicator_ids": &execution.extended_budget_deferred_indicator_ids,
        "extended_budget_columns": execution.extended_budget_columns,
        "extended_planned_columns": execution.extended_planned_columns,
    });
    let ledger = json!({
        "schema": "neoethos.task1.indicator_ledger.v2",
        "source_path": source_path,
        "source_sha256": source_sha256,
        "source_asset_sha256": source_asset_sha256,
        "source_url": source_url,
        "source_release": source_release,
        "source_class": "quarantined_external_or_legacy_vortex",
        "financial_evaluation_allowed": false,
        "symbol": symbol,
        "timeframe": timeframe,
        "rows": ohlcv.close.len(),
        "first_timestamp": ohlcv.timestamp.as_ref().and_then(|v| v.first()),
        "last_timestamp": ohlcv.timestamp.as_ref().and_then(|v| v.last()),
        "elapsed_ns": duration_ns(elapsed),
        "attempted_indicator_ids": ALL_INDICATORS.len(),
        "expected_nonproducing": EXPECTED_NON_PRODUCING,
        "expected_nonproducing_outputs": EXPECTED_NON_PRODUCING_OUTPUTS,
        "execution": execution_json,
        "produced_columns": columns.len(),
        "schema_hash": schema_hash,
        "truncated_columns": truncated,
        "all_non_finite_columns": all_non_finite,
        "constant_columns": constant,
        "warmup_or_gap_columns": warmup_or_gap,
        "duplicate_columns": duplicates,
        "compiled_with_gpu_cuda": cfg!(feature = "gpu-cuda"),
        "claimed_cuda_sweep_ids": claimed_cuda_sweep_ids,
        "claimed_cuda_sweep_count": claimed_cuda_sweep_count,
        "multi_period_sweep_ids": MULTI_PERIOD_IDS,
    });

    let encoded = serde_json::to_vec_pretty(&ledger).expect("serialize Task-1 indicator ledger");
    if let Some(path) = std::env::var_os("NEOETHOS_TASK1_LEDGER_OUTPUT") {
        let path = PathBuf::from(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create Task-1 ledger output directory");
        }
        std::fs::write(&path, &encoded).expect("write Task-1 indicator ledger");
        eprintln!("Task-1 indicator ledger: {}", path.display());
    }
    eprintln!("Task-1 feature schema hash: {schema_hash}");
}

/// A column belongs to the multi-period sweep iff its name starts with a swept
/// id followed by `_<period>`. Everything else came from the base
/// ALL_INDICATORS pass.
fn is_sweep_column(name: &str) -> bool {
    MULTI_PERIOD_IDS.iter().any(|id| {
        ALT_PERIODS.iter().any(|p| {
            let prefix = format!("{id}_{p}");
            name == prefix || name.starts_with(&format!("{prefix}_"))
        })
    })
}

#[test]
fn task1_duplicate_comparison_matches_indicator_fingerprint_semantics() {
    let left = [f64::NAN, -0.0, 1.25];
    let right = [f64::from_bits(0x7ff8_0000_0000_0001), 0.0, 1.25];
    assert_eq!(series_fingerprint(&left), series_fingerprint(&right));
    assert!(canonical_series_equal(&left, &right));
    assert!(!canonical_series_equal(&left, &[f64::NAN, 0.0, 1.5]));
}

#[test]
#[ignore = "requires the real EURUSD M5 vortex store; run it explicitly before any production \
            discovery run — `cargo test -p neoethos-data --test vocabulary_restoration_measured \
            -- --ignored --nocapture`"]
fn measure_the_restored_vocabulary_on_real_bars() {
    neoethos_core::logging::setup_minimal_logging(false)
        .expect("Task-1 production INFO/WARN ledger must be visible");
    let (ohlcv, source_path) = real_ohlcv();
    let n = ohlcv.close.len();

    let t0 = std::time::Instant::now();
    let run = compute_classic_ta_columns_with_policy_report(&ohlcv, IndicatorComputePolicy::Cpu)
        .expect("the repaired base pass must clear its own vocabulary floor on real bars");
    let elapsed = t0.elapsed();
    let cols = &run.columns;

    assert_eq!(
        run.report.admitted_indicator_ids.len() + run.report.budget_deferred_indicator_ids.len(),
        ALL_INDICATORS.len(),
        "the production execution report must account for every base indicator exactly once"
    );

    let sweep: Vec<&String> = cols
        .iter()
        .map(|(k, _)| k)
        .filter(|k| is_sweep_column(k))
        .collect();
    let base_count = cols.len() - sweep.len();

    // Distinct producing ids in the base pass: the column name for a
    // single-output indicator IS the id; a multi-output one is `<id>_<output>`.
    let mut per_id: BTreeMap<String, usize> = BTreeMap::new();
    for (name, _) in cols.iter().filter(|(k, _)| !is_sweep_column(k)) {
        let id = neoethos_data::core::all_indicators::ALL_INDICATORS
            .iter()
            .filter(|id| name == *id || name.starts_with(&format!("{id}_")))
            // Longest match wins, so `adaptive_macd_signal` is not attributed
            // to `adaptive_macd`'s shorter cousins.
            .max_by_key(|id| id.len())
            .map(|s| s.to_string())
            .unwrap_or_else(|| name.clone());
        *per_id.entry(id).or_insert(0) += 1;
    }

    let bytes = column_bytes(n) * cols.len() as u64;
    let budget = VocabularyBudget::for_frame(n);

    write_task1_ledger(&source_path, &ohlcv, cols, &run.report, elapsed);

    eprintln!("\n=== RESTORED INDICATOR VOCABULARY, MEASURED ===");
    eprintln!("bars                    : {n}");
    eprintln!("wall clock              : {:.1}s", elapsed.as_secs_f64());
    eprintln!(
        "ALL_INDICATORS ids      : {}",
        neoethos_data::core::all_indicators::ALL_INDICATORS.len()
    );
    eprintln!(
        "producing ids (base)    : {}   (was 1: ttm_trend)",
        per_id.len()
    );
    eprintln!("base columns            : {base_count}   (was 1)");
    eprintln!("sweep columns           : {}   (was 65)", sweep.len());
    eprintln!("TOTAL columns           : {}   (was 66)", cols.len());
    eprintln!(
        "f64 staging cost        : {:.2} GB at {n} bars ({:.2} MB per column)",
        bytes as f64 / 1e9,
        column_bytes(n) as f64 / 1e6
    );
    eprintln!(
        "budget on this machine  : {} columns max ({:.1} GB free)",
        budget.max_columns,
        budget.available_bytes as f64 / 1e9
    );
    eprintln!(
        "at 843,456 bars         : {:.2} GB for {} columns",
        (column_bytes(843_456) * cols.len() as u64) as f64 / 1e9,
        cols.len()
    );
    eprintln!("===============================================\n");

    // The measurement is the point, but these are the claims it must support.
    assert!(
        cols.len() > 400,
        "the restored vocabulary is {} columns — the whole exercise was to get past 66",
        cols.len()
    );
    assert!(
        per_id.len() > 250,
        "only {} of {} indicator ids produced a column",
        per_id.len(),
        neoethos_data::core::all_indicators::ALL_INDICATORS.len()
    );
    // Every column full length: the cube copy in lib.rs zero-pads otherwise.
    for (name, v) in cols {
        assert_eq!(v.len(), n, "column '{name}' is {} values", v.len());
    }
}

#[cfg(feature = "gpu-cuda")]
#[test]
#[ignore = "requires a real NVIDIA card and NEOETHOS_TASK1_VORTEX; CPU fallback is forbidden"]
fn task1_full_feature_frame_cpu_cuda_parity_on_real_vortex() {
    neoethos_core::logging::setup_minimal_logging(false)
        .expect("Task-1 production INFO/WARN ledger must be visible");
    require_task1_true_env("VECTOR_TA_CUDA_FORCE_FATBIN");
    require_task1_true_env("CUDA_MODULE_LOAD_DEBUG");
    let (ohlcv, source_path) = real_ohlcv();

    let cpu_started = std::time::Instant::now();
    let cpu = compute_classic_ta_columns_with_policy(&ohlcv, IndicatorComputePolicy::Cpu)
        .expect("CPU feature frame must build on the quarantined Vortex fixture");
    let cpu_elapsed = cpu_started.elapsed();

    let cuda_started = std::time::Instant::now();
    let cuda = compute_classic_ta_columns_with_policy(&ohlcv, IndicatorComputePolicy::RequireGpu)
        .expect("RequireGpu must execute the real CUDA sweep without a CPU fallback");
    let cuda_elapsed = cuda_started.elapsed();

    let cpu_names: Vec<&str> = cpu.iter().map(|(name, _)| name.as_str()).collect();
    let cuda_names: Vec<&str> = cuda.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(
        cuda_names, cpu_names,
        "CPU/CUDA feature names or ordering differ"
    );

    let mut compared_finite_cells = 0u64;
    let mut different_f64_bits = 0u64;
    let mut worst_absolute_delta = 0.0f64;
    let mut worst_relative_delta = 0.0f64;
    let mut worst_column = String::new();
    let mut worst_row = 0usize;
    let mut worst_relative_column = String::new();
    let mut worst_relative_row = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for ((name, cpu_values), (_, cuda_values)) in cpu.iter().zip(&cuda) {
        if cuda_values.len() != cpu_values.len() {
            failures.push(format!(
                "{name}: CPU/CUDA lengths differ (cpu={}, cuda={})",
                cpu_values.len(),
                cuda_values.len()
            ));
            continue;
        }
        let mut validity_mismatches = 0usize;
        let mut first_validity_mismatch: Option<(usize, f64, f64)> = None;
        let mut non_finite_values = 0usize;
        let mut first_non_finite: Option<(usize, f64, f64)> = None;
        let mut tolerance_failures = 0usize;
        let mut first_tolerance_failure: Option<(usize, f64, f64, f64, f64)> = None;
        for (row, (&expected, &actual)) in cpu_values.iter().zip(cuda_values).enumerate() {
            if actual.is_nan() != expected.is_nan() {
                validity_mismatches += 1;
                first_validity_mismatch.get_or_insert((row, expected, actual));
                continue;
            }
            if expected.is_nan() {
                continue;
            }
            if !expected.is_finite() || !actual.is_finite() {
                non_finite_values += 1;
                first_non_finite.get_or_insert((row, expected, actual));
                continue;
            }
            compared_finite_cells += 1;
            different_f64_bits += u64::from(expected.to_bits() != actual.to_bits());
            let absolute = (expected - actual).abs();
            let relative = absolute / expected.abs().max(f64::MIN_POSITIVE);
            if relative > worst_relative_delta {
                worst_relative_delta = relative;
                worst_relative_column = name.clone();
                worst_relative_row = row;
            }
            if absolute > worst_absolute_delta {
                worst_absolute_delta = absolute;
                worst_column = name.clone();
                worst_row = row;
            }
            let allowed = 1e-12 + 1e-12 * expected.abs();
            if absolute > allowed {
                tolerance_failures += 1;
                first_tolerance_failure.get_or_insert((row, expected, actual, absolute, allowed));
            }
        }
        if let Some((row, expected, actual)) = first_validity_mismatch {
            failures.push(format!(
                "{name}: {validity_mismatches} CPU/CUDA validity-mask mismatch(es); first at \
                 [{row}] cpu={expected}, cuda={actual}"
            ));
        }
        if let Some((row, expected, actual)) = first_non_finite {
            failures.push(format!(
                "{name}: {non_finite_values} non-finite non-warmup value(s); first at [{row}] \
                 cpu={expected}, cuda={actual}"
            ));
        }
        if let Some((row, expected, actual, absolute, allowed)) = first_tolerance_failure {
            failures.push(format!(
                "{name}: {tolerance_failures} f64 tolerance failure(s); first at [{row}] delta \
                 {absolute:e} exceeds {allowed:e} (cpu={expected:.17e}, cuda={actual:.17e})"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} full-frame CPU/CUDA defect group(s):\n{}",
        failures.len(),
        failures.join("\n")
    );

    let schema_hash =
        neoethos_core::storage::json::stable_json_hash(&cpu_names).expect("hash parity schema");
    let symbol = required_task1_env("NEOETHOS_TASK1_SYMBOL");
    let timeframe = required_task1_env("NEOETHOS_TASK1_TIMEFRAME");
    let source_sha256 = required_task1_env("NEOETHOS_TASK1_SOURCE_SHA256");
    let source_release = required_task1_env("NEOETHOS_TASK1_SOURCE_RELEASE");
    let source_asset_sha256 = required_task1_env("NEOETHOS_TASK1_ASSET_SHA256");
    let source_url = required_task1_env("NEOETHOS_TASK1_SOURCE_URL");
    let gpu_name = required_task1_env("NEOETHOS_TASK1_GPU_NAME");
    let gpu_uuid = required_task1_env("NEOETHOS_TASK1_GPU_UUID");
    let gpu_compute_capability = required_task1_env("NEOETHOS_TASK1_GPU_COMPUTE_CAPABILITY");
    let gpu_driver = required_task1_env("NEOETHOS_TASK1_GPU_DRIVER");
    let cuda_toolkit = required_task1_env("NEOETHOS_TASK1_CUDA_TOOLKIT");
    let report = json!({
        "schema": "neoethos.task1.cpu_cuda_full_feature_parity.v1",
        "source_path": source_path,
        "source_sha256": source_sha256,
        "source_asset_sha256": source_asset_sha256,
        "source_url": source_url,
        "source_release": source_release,
        "source_class": "quarantined_external_or_legacy_vortex",
        "financial_evaluation_allowed": false,
        "symbol": symbol,
        "timeframe": timeframe,
        "rows": ohlcv.len(),
        "columns": cpu.len(),
        "schema_hash": schema_hash,
        "cpu_elapsed_ns": duration_ns(cpu_elapsed),
        "cuda_elapsed_ns": duration_ns(cuda_elapsed),
        "compared_finite_cells": compared_finite_cells,
        "different_f64_bits": different_f64_bits,
        "worst_absolute_delta": worst_absolute_delta,
        "worst_relative_delta": worst_relative_delta,
        "worst_column": worst_column,
        "worst_row": worst_row,
        "worst_relative_column": worst_relative_column,
        "worst_relative_row": worst_relative_row,
        "required_device_lane": true,
        "execution_class": "hybrid_cuda_sweep_plus_cpu_unclaimed_nodes",
        "full_frame_executed_entirely_on_gpu": false,
        "performance_comparison_valid": false,
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "gpu_compute_capability": gpu_compute_capability,
        "gpu_driver": gpu_driver,
        "cuda_toolkit": cuda_toolkit,
        "compiled_archs": vector_ta::cuda::module_loader::COMPILED_ARCHS,
        "compiled_ptx_arch": vector_ta::cuda::module_loader::COMPILED_PTX_ARCH,
        "module_load_path": "forced_fatbin_no_ptx_fallback",
        "claimed_cuda_sweep_count": neoethos_data::core::gpu_indicators::GPU_SWEEP_SPECS.len(),
        "multi_period_sweep_count": MULTI_PERIOD_IDS.len(),
        "claimed_cuda_sweep_ids": neoethos_data::core::gpu_indicators::GPU_SWEEP_SPECS
            .iter()
            .map(|spec| spec.id)
            .collect::<Vec<_>>(),
    });
    let encoded = serde_json::to_vec_pretty(&report).expect("serialize CPU/CUDA parity report");
    if let Some(path) = std::env::var_os("NEOETHOS_TASK1_GPU_LEDGER_OUTPUT") {
        let path = PathBuf::from(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create CPU/CUDA report directory");
        }
        std::fs::write(&path, &encoded).expect("write CPU/CUDA parity report");
        eprintln!("Task-1 CPU/CUDA parity report: {}", path.display());
    }
    eprintln!("NEOETHOS_TASK1_GPU_PARITY={report}");
}
