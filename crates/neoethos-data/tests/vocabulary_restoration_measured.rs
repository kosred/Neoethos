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
use neoethos_data::core::hpc_ta::{
    ALT_PERIODS, ClassicTaExecutionReport, IndicatorComputePolicy, MULTI_PERIOD_IDS,
    SWEEP_POINT_EXCLUSIONS, compute_classic_ta_columns_with_policy_report,
};
use neoethos_data::core::indicator_ledger::{
    EXPECTED_NON_PRODUCING, PRODUCTION_OUTPUT_EXCLUSIONS, has_finite_variation, series_fingerprint,
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

#[derive(Debug)]
struct Task1QualityFindings {
    exact_duplicates: Vec<(String, String)>,
    all_non_finite: Vec<String>,
}

fn write_task1_ledger(
    source_path: &std::path::Path,
    ohlcv: &neoethos_data::Ohlcv,
    columns: &[(String, Vec<f64>)],
    execution: &ClassicTaExecutionReport,
    elapsed: std::time::Duration,
) -> Task1QualityFindings {
    let names: Vec<&str> = columns.iter().map(|(name, _)| name.as_str()).collect();
    let schema_hash =
        neoethos_core::storage::json::stable_json_hash(&names).expect("hash Task-1 feature schema");

    let mut indices_by_fingerprint: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    let mut exact_duplicates = Vec::new();
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
            exact_duplicates.push((name.clone(), columns[first_index].0.clone()));
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
        "schema": "neoethos.task1.indicator_ledger.v3",
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
        "production_output_exclusions": PRODUCTION_OUTPUT_EXCLUSIONS,
        "sweep_point_exclusions": SWEEP_POINT_EXCLUSIONS,
        "execution": execution_json,
        "produced_columns": columns.len(),
        "schema_hash": schema_hash,
        "truncated_columns": truncated,
        "all_non_finite_columns": &all_non_finite,
        "constant_columns": constant,
        "warmup_or_gap_columns": warmup_or_gap,
        "duplicate_columns": exact_duplicates.iter().map(|(name, duplicate_of)| json!({
            "name": name,
            "duplicate_of": duplicate_of,
        })).collect::<Vec<_>>(),
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
    Task1QualityFindings {
        exact_duplicates,
        all_non_finite,
    }
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

/// Resolve a classic-TA column to `(indicator, optional swept period, output)`.
/// The output is empty for a single-output series. Longest-id matching keeps
/// underscores inside indicator ids from being mistaken for separators.
fn classic_sweep_identity(name: &str) -> Option<(&'static str, Option<usize>, &str)> {
    let indicator = ALL_INDICATORS
        .iter()
        .copied()
        .filter(|id| name == *id || name.starts_with(&format!("{id}_")))
        .max_by_key(|id| id.len())?;
    let suffix = name
        .strip_prefix(indicator)?
        .strip_prefix('_')
        .unwrap_or_default();
    if suffix.is_empty() {
        return Some((indicator, None, ""));
    }

    let (first, remainder) = suffix
        .split_once('_')
        .map_or((suffix, ""), |(first, remainder)| (first, remainder));
    if let Ok(period) = first.parse::<usize>()
        && ALT_PERIODS.contains(&period)
    {
        return Some((indicator, Some(period), remainder));
    }
    Some((indicator, None, suffix))
}

fn structurally_duplicate_sweep_pairs(duplicates: &[(String, String)]) -> Vec<(String, String)> {
    duplicates
        .iter()
        .filter_map(|(name, duplicate_of)| {
            let (id, period, output) = classic_sweep_identity(name)?;
            let (first_id, first_period, first_output) = classic_sweep_identity(duplicate_of)?;
            (id == first_id
                && output == first_output
                && period != first_period
                && (period.is_some() || first_period.is_some()))
            .then(|| (name.clone(), duplicate_of.clone()))
        })
        .collect()
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
fn structural_sweep_duplicate_classifier_ignores_cross_output_corpus_coincidence() {
    let duplicates = vec![
        (
            "ehlers_itrend_100".to_string(),
            "ehlers_itrend_50".to_string(),
        ),
        (
            "adaptive_bounds_rsi_200_upper_signal".to_string(),
            "adaptive_bounds_rsi_200_lower_signal".to_string(),
        ),
        ("rsi_21".to_string(), "rsi".to_string()),
    ];
    assert_eq!(
        structurally_duplicate_sweep_pairs(&duplicates),
        vec![
            (
                "ehlers_itrend_100".to_string(),
                "ehlers_itrend_50".to_string()
            ),
            ("rsi_21".to_string(), "rsi".to_string()),
        ]
    );
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
    let run =
        compute_classic_ta_columns_with_policy_report(&ohlcv, IndicatorComputePolicy::CpuOnly)
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

    let findings = write_task1_ledger(&source_path, &ohlcv, cols, &run.report, elapsed);
    let structural_sweep_duplicates =
        structurally_duplicate_sweep_pairs(&findings.exact_duplicates);

    assert!(
        structural_sweep_duplicates.is_empty(),
        "real-corpus sweep still contains exact aliases of the same indicator/output at another \
         period: {structural_sweep_duplicates:#?}. Repair the static period plan; never drop \
         columns based on this frame"
    );
    assert!(
        findings.all_non_finite.is_empty(),
        "real-corpus production schema still contains all-nonfinite outputs: {:#?}. Enable a \
         hand-reviewed formula or exclude the disabled output statically with a named reason",
        findings.all_non_finite
    );

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
