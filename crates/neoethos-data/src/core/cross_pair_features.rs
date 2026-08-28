//! Cross-pair feature engineering — features that USE multiple
//! symbols together.
//!
//! Phase F3. Adds inter-symbol relationship features to the
//! feature pipeline. The 34-model ensemble can already learn
//! per-symbol indicators (RSI, MACD, regime, etc.); adding
//! cross-pair features lets it ALSO learn:
//!
//! - **Rolling correlation** between log-returns of the base
//!   symbol and each related symbol over multiple windows. High
//!   correlation = co-moving regime; sudden correlation breakdown
//!   = signal worth attending.
//! - **Log-price spread** between base and related. Mean-
//!   reverting spreads are the textbook basis for pairs trading;
//!   the ML stack can learn when a wide spread predicts a
//!   reversion in EITHER side.
//! - **Z-scored spread** = (spread - rolling_mean) / rolling_std
//!   over a window. Bounds the raw spread to comparable scale
//!   across symbol pairs.
//!
//! ## Why pure Rust
//!
//! Per operator directive (2026-05-18): zero Python, all-Rust for
//! performance. This module uses only `std` + the in-crate
//! `Ohlcv` type — no ndarray, no external numerics.
//!
//! ## Timestamp alignment
//!
//! Symbols traded at the same broker share a bar timeline most of
//! the time but occasionally drift (one symbol's bar might be
//! late by a few minutes around a feed hiccup). The aligner
//! `align_related_to_base_index` does a forward-search by
//! timestamp_ms: for each base bar, it finds the related-symbol
//! bar with the LARGEST timestamp <= base.timestamp_ms. Missing
//! related-symbol data emits NaN at that index; downstream feature
//! columns propagate NaN, and the training pipeline's normalisers
//! handle NaN as a missing-value sentinel.
//!
//! ## What this module does NOT do
//!
//! - It does NOT modify the existing `compute_hpc_feature_frame`
//!   pipeline. F3 ships the helper functions; wiring them into
//!   the multi-symbol training orchestrator (Phase B5
//!   MultiSymbolTrainingOrchestrator) is a follow-up commit
//!   when the operator's training run wants cross-pair features.
//! - It does NOT solve the FeatureFrame's "single Ohlcv input"
//!   shape. Callers compose: build base FeatureFrame, then append
//!   cross-pair columns from `compute_cross_pair_features` and
//!   re-emit as an augmented FeatureFrame.

use super::super::Ohlcv;
use super::features::{FeatureCellValidity, FeatureColumnF64};
use super::timestamps::validate_canonical_millisecond_timestamps;
use anyhow::{Context, Result, ensure};
use std::collections::HashSet;

/// Default rolling windows for cross-pair correlation and
/// z-scored spread. Mid-frequency forex defaults: 10, 20, 50,
/// 100 bars cover scalping (10–20), swing (50), and position
/// (100) horizons.
pub const DEFAULT_CROSS_PAIR_WINDOWS: &[usize] = &[10, 20, 50, 100];

/// Version 2 removes index fallback and numeric warmup sentinels, adds bounded
/// causal as-of alignment, and carries typed validity through every output.
pub const CROSS_PAIR_TRANSFORM_SEMANTIC_VERSION: u32 = 2;

/// Build the internal f64 value-plus-validity cross-pair columns used while the
/// public feature/model boundary is migrated atomically in Tasks 5B-9.
///
/// Unlike [`compute_cross_pair_features`], this boundary never guesses row
/// alignment when timestamps are absent, never encodes warmup/staleness as a
/// numeric zero or an untyped NaN, and never forward-fills an observation past
/// `max_staleness_ms`. Every related observation is selected causally with
/// `related_timestamp <= base_timestamp`.
pub fn compute_cross_pair_feature_columns_f64(
    base: &Ohlcv,
    related: &[(String, &Ohlcv)],
    windows: &[usize],
    max_staleness_ms: i64,
) -> Result<Vec<FeatureColumnF64>> {
    ensure!(
        max_staleness_ms >= 0,
        "cross-pair max staleness must be non-negative"
    );
    if base.is_empty() || related.is_empty() {
        return Ok(Vec::new());
    }
    ensure!(!windows.is_empty(), "cross-pair windows must not be empty");
    let mut unique_windows = HashSet::with_capacity(windows.len());
    for &window in windows {
        ensure!(window >= 2, "cross-pair windows must be at least two bars");
        ensure!(
            unique_windows.insert(window),
            "duplicate cross-pair window {window}"
        );
    }

    let base_timestamps = validate_cross_pair_input("base", base)?;
    let base_returns = typed_log_returns(&base.close);
    let mut columns = Vec::with_capacity(related.len() * (1 + windows.len() * 2));
    let mut column_symbols = HashSet::with_capacity(related.len());

    for (related_name, related_ohlcv) in related {
        let sanitized = sanitize_symbol_for_column_name(related_name);
        ensure!(
            !sanitized.is_empty(),
            "related symbol `{related_name}` has no safe column identity"
        );
        ensure!(
            column_symbols.insert(sanitized.clone()),
            "related symbol `{related_name}` collides with another cross-pair column identity `{sanitized}`"
        );
        let related_timestamps = validate_cross_pair_input(related_name, related_ohlcv)?;
        let aligned = align_related_close_typed(
            base_timestamps,
            related_timestamps,
            &related_ohlcv.close,
            max_staleness_ms,
        );
        let related_returns = typed_log_returns_from_cells(&aligned);

        for &window in windows {
            columns.push(typed_rolling_pearson(
                format!("xcorr_{sanitized}_{window}"),
                &base_returns,
                &related_returns,
                window,
            )?);
        }

        let spread = typed_log_spread(format!("spread_{sanitized}"), &base.close, &aligned)?;
        columns.push(spread.clone());
        for &window in windows {
            columns.push(typed_rolling_zscore(
                format!("spread_z_{sanitized}_{window}"),
                &spread,
                window,
            )?);
        }
    }

    Ok(columns)
}

fn validate_cross_pair_input<'a>(label: &str, input: &'a Ohlcv) -> Result<&'a [i64]> {
    let timestamps = input
        .timestamp
        .as_deref()
        .with_context(|| format!("cross-pair input `{label}` has no canonical timestamp column"))?;
    ensure!(
        timestamps.len() == input.close.len(),
        "cross-pair input `{label}` has {} timestamps but {} close values",
        timestamps.len(),
        input.close.len()
    );
    validate_canonical_millisecond_timestamps(timestamps)
        .with_context(|| format!("validate cross-pair input `{label}` timestamps"))?;
    Ok(timestamps)
}

#[derive(Debug, Clone, Copy)]
struct FeatureCellF64 {
    value: f64,
    validity: FeatureCellValidity,
}

impl FeatureCellF64 {
    const fn invalid(validity: FeatureCellValidity) -> Self {
        Self {
            value: f64::NAN,
            validity,
        }
    }

    fn from_positive_price(value: f64) -> Self {
        if value.is_finite() && value > 0.0 {
            Self {
                value,
                validity: FeatureCellValidity::Valid,
            }
        } else {
            Self::invalid(FeatureCellValidity::NonFinite)
        }
    }
}

fn align_related_close_typed(
    base_timestamps: &[i64],
    related_timestamps: &[i64],
    related_close: &[f64],
    max_staleness_ms: i64,
) -> Vec<FeatureCellF64> {
    let mut output = Vec::with_capacity(base_timestamps.len());
    let mut related_index = 0usize;

    for &base_timestamp in base_timestamps {
        while related_index + 1 < related_timestamps.len()
            && related_timestamps[related_index + 1] <= base_timestamp
        {
            related_index += 1;
        }
        if related_timestamps[related_index] > base_timestamp {
            output.push(FeatureCellF64::invalid(
                FeatureCellValidity::AlignmentMissing,
            ));
            continue;
        }
        if base_timestamp - related_timestamps[related_index] > max_staleness_ms {
            output.push(FeatureCellF64::invalid(FeatureCellValidity::Stale));
            continue;
        }
        output.push(FeatureCellF64::from_positive_price(
            related_close[related_index],
        ));
    }
    output
}

fn typed_log_returns(closes: &[f64]) -> Vec<FeatureCellF64> {
    let cells = closes
        .iter()
        .copied()
        .map(FeatureCellF64::from_positive_price)
        .collect::<Vec<_>>();
    typed_log_returns_from_cells(&cells)
}

fn typed_log_returns_from_cells(cells: &[FeatureCellF64]) -> Vec<FeatureCellF64> {
    let mut output = Vec::with_capacity(cells.len());
    if cells.is_empty() {
        return output;
    }
    output.push(FeatureCellF64::invalid(FeatureCellValidity::Warmup));
    for pair in cells.windows(2) {
        if !pair[0].validity.is_valid() {
            output.push(FeatureCellF64::invalid(pair[0].validity));
        } else if !pair[1].validity.is_valid() {
            output.push(FeatureCellF64::invalid(pair[1].validity));
        } else {
            output.push(FeatureCellF64 {
                value: pair[1].value.ln() - pair[0].value.ln(),
                validity: FeatureCellValidity::Valid,
            });
        }
    }
    output
}

fn typed_log_spread(
    name: String,
    base_close: &[f64],
    related_close: &[FeatureCellF64],
) -> Result<FeatureColumnF64> {
    ensure!(
        base_close.len() == related_close.len(),
        "cross-pair spread inputs have different lengths"
    );
    let mut values = Vec::with_capacity(base_close.len());
    let mut validity = Vec::with_capacity(base_close.len());
    for (&base, related) in base_close.iter().zip(related_close) {
        if !related.validity.is_valid() {
            values.push(f64::NAN);
            validity.push(related.validity);
        } else if !base.is_finite() || base <= 0.0 {
            values.push(f64::NAN);
            validity.push(FeatureCellValidity::NonFinite);
        } else {
            values.push(base.ln() - related.value.ln());
            validity.push(FeatureCellValidity::Valid);
        }
    }
    FeatureColumnF64::new(name, values, validity)
}

fn first_invalidity(
    mut cells: impl Iterator<Item = FeatureCellValidity>,
) -> Option<FeatureCellValidity> {
    cells.find(|validity| !validity.is_valid())
}

fn typed_rolling_pearson(
    name: String,
    left: &[FeatureCellF64],
    right: &[FeatureCellF64],
    window: usize,
) -> Result<FeatureColumnF64> {
    ensure!(
        left.len() == right.len(),
        "correlation inputs differ in length"
    );
    let mut values = vec![f64::NAN; left.len()];
    let mut validity = vec![FeatureCellValidity::Warmup; left.len()];
    if left.len() < window {
        return FeatureColumnF64::new(name, values, validity);
    }

    for end in (window - 1)..left.len() {
        let start = end + 1 - window;
        if let Some(reason) = first_invalidity(
            left[start..=end]
                .iter()
                .chain(&right[start..=end])
                .map(|cell| cell.validity),
        ) {
            validity[end] = reason;
            continue;
        }
        let count = window as f64;
        let mean_left = left[start..=end].iter().map(|cell| cell.value).sum::<f64>() / count;
        let mean_right = right[start..=end]
            .iter()
            .map(|cell| cell.value)
            .sum::<f64>()
            / count;
        let mut left_ss = 0.0;
        let mut right_ss = 0.0;
        let mut cross = 0.0;
        for (left_cell, right_cell) in left[start..=end].iter().zip(&right[start..=end]) {
            let left_delta = left_cell.value - mean_left;
            let right_delta = right_cell.value - mean_right;
            left_ss += left_delta * left_delta;
            right_ss += right_delta * right_delta;
            cross += left_delta * right_delta;
        }
        let denominator = (left_ss * right_ss).sqrt();
        if !denominator.is_finite() {
            validity[end] = FeatureCellValidity::NonFinite;
        } else if denominator == 0.0 {
            validity[end] = FeatureCellValidity::ZeroDenominator;
        } else {
            let value = (cross / denominator).clamp(-1.0, 1.0);
            if value.is_finite() {
                values[end] = value;
                validity[end] = FeatureCellValidity::Valid;
            } else {
                validity[end] = FeatureCellValidity::NonFinite;
            }
        }
    }
    FeatureColumnF64::new(name, values, validity)
}

fn typed_rolling_zscore(
    name: String,
    source: &FeatureColumnF64,
    window: usize,
) -> Result<FeatureColumnF64> {
    let mut values = vec![f64::NAN; source.len()];
    let mut validity = vec![FeatureCellValidity::Warmup; source.len()];
    if source.len() < window {
        return FeatureColumnF64::new(name, values, validity);
    }

    for end in (window - 1)..source.len() {
        let start = end + 1 - window;
        if let Some(reason) = first_invalidity(source.validity[start..=end].iter().copied()) {
            validity[end] = reason;
            continue;
        }
        let count = window as f64;
        let mean = source.values[start..=end].iter().sum::<f64>() / count;
        let sum_squared_deviation = source.values[start..=end]
            .iter()
            .map(|value| {
                let delta = *value - mean;
                delta * delta
            })
            .sum::<f64>();
        let standard_deviation = (sum_squared_deviation / count).sqrt();
        if !standard_deviation.is_finite() {
            validity[end] = FeatureCellValidity::NonFinite;
        } else if standard_deviation == 0.0 {
            validity[end] = FeatureCellValidity::ZeroDenominator;
        } else {
            let value = (source.values[end] - mean) / standard_deviation;
            if value.is_finite() {
                values[end] = value;
                validity[end] = FeatureCellValidity::Valid;
            } else {
                validity[end] = FeatureCellValidity::NonFinite;
            }
        }
    }
    FeatureColumnF64::new(name, values, validity)
}

/// Build cross-pair feature columns for one base symbol against
/// any number of related symbols.
///
/// Returns named columns suitable for appending to the base
/// symbol's existing FeatureFrame. Column naming convention:
///
/// - `xcorr_<RELATED>_<WIN>`: rolling Pearson correlation of
///   1-bar log returns over the last `WIN` bars
/// - `spread_<RELATED>`: instantaneous log-price spread,
///   `ln(base.close) - ln(related.close)`
/// - `spread_z_<RELATED>_<WIN>`: z-scored spread over `WIN`
///   bars: `(spread - mean) / std`
///
/// Where `<RELATED>` is uppercased + alphanumeric-only for safe
/// embedding in column names. Windows come from the supplied
/// `windows` slice; pass [`DEFAULT_CROSS_PAIR_WINDOWS`] for the
/// standard set.
///
/// **Output length** equals `base.len()` — each cross-pair column
/// is aligned to the base's bar timeline. Bars where the related
/// symbol has no aligned data emit NaN.
///
/// **Empty inputs** return an empty Vec. The caller is responsible
/// for not asking for cross-pair features when there's nothing to
/// cross with.
pub fn compute_cross_pair_features(
    base: &Ohlcv,
    related: &[(String, &Ohlcv)],
    windows: &[usize],
) -> Vec<(String, Vec<f64>)> {
    let base_len = base.len();
    if base_len == 0 || related.is_empty() {
        return Vec::new();
    }
    let base_log_returns = log_returns(&base.close);
    let mut columns: Vec<(String, Vec<f64>)> = Vec::new();

    for (related_name, related_ohlcv) in related {
        if related_ohlcv.is_empty() {
            continue;
        }
        let sanitized = sanitize_symbol_for_column_name(related_name);
        // Align the related symbol's close prices onto the base's
        // bar index. NaN where misaligned / missing.
        let aligned_close = align_related_to_base_index(base, related_ohlcv);
        let aligned_log_returns = log_returns(&aligned_close);

        // 1. Rolling Pearson correlation of log returns.
        for &win in windows {
            let col_name = format!("xcorr_{}_{}", sanitized, win);
            let values = rolling_pearson_correlation(&base_log_returns, &aligned_log_returns, win);
            columns.push((col_name, values));
        }

        // 2. Instantaneous log-price spread.
        let mut spread = Vec::with_capacity(base_len);
        for i in 0..base_len {
            let b = base.close[i];
            let r = aligned_close[i];
            if b > 0.0 && r > 0.0 && r.is_finite() {
                spread.push(b.ln() - r.ln());
            } else {
                spread.push(f64::NAN);
            }
        }
        columns.push((format!("spread_{}", sanitized), spread.clone()));

        // 3. Z-scored spread per window.
        for &win in windows {
            let col_name = format!("spread_z_{}_{}", sanitized, win);
            let zs = rolling_zscore(&spread, win);
            columns.push((col_name, zs));
        }
    }

    columns
}

// ---------------------------------------------------------------------------
// Helpers (private)
// ---------------------------------------------------------------------------

/// Sanitise a symbol name for safe embedding in a column name.
/// `EUR/USD` → `EURUSD`, `eur.usd` → `EURUSD`, etc.
fn sanitize_symbol_for_column_name(symbol: &str) -> String {
    symbol
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_uppercase())
        .collect()
}

/// Compute log-returns from a price series. The first element is
/// 0.0 (no previous bar to diff against); subsequent elements are
/// `ln(close[i]) - ln(close[i-1])`. Zero or negative prices emit
/// NaN for that index.
fn log_returns(closes: &[f64]) -> Vec<f64> {
    let n = closes.len();
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }
    out.push(0.0);
    for i in 1..n {
        let prev = closes[i - 1];
        let curr = closes[i];
        if prev > 0.0 && curr > 0.0 && prev.is_finite() && curr.is_finite() {
            out.push(curr.ln() - prev.ln());
        } else {
            out.push(f64::NAN);
        }
    }
    out
}

/// Align `related`'s close prices onto `base`'s bar index by
/// timestamp_ms forward-search.
///
/// For each base bar at index `i`, finds the related-symbol bar
/// whose `timestamp_ms` is the LARGEST value `<= base.timestamp[i]`.
/// Emits NaN when:
/// - The base or related has no timestamp vector
/// - No related bar with `timestamp <= base[i]` exists (the
///   related symbol hasn't started yet)
fn align_related_to_base_index(base: &Ohlcv, related: &Ohlcv) -> Vec<f64> {
    let base_len = base.len();
    let mut out = vec![f64::NAN; base_len];
    let (Some(base_ts), Some(rel_ts)) = (base.timestamp.as_ref(), related.timestamp.as_ref())
    else {
        // No timestamps available — fall back to index-by-index
        // alignment for as far as both vectors go.
        let len = base_len.min(related.close.len());
        for i in 0..len {
            out[i] = related.close[i];
        }
        return out;
    };
    // Two-pointer walk since both timestamp vectors are sorted
    // (broker bars are chronological).
    let mut j = 0usize;
    for i in 0..base_len {
        let target = base_ts[i];
        while j + 1 < rel_ts.len() && rel_ts[j + 1] <= target {
            j += 1;
        }
        if j < rel_ts.len() && rel_ts[j] <= target {
            out[i] = related.close[j];
        }
    }
    out
}

/// Rolling Pearson correlation between two series of equal
/// length over a sliding window.
///
/// Returns `(n,)` shaped vector. Indices `< window - 1` emit
/// NaN (warmup), as do windows that contain NaN values in either
/// series.
fn rolling_pearson_correlation(a: &[f64], b: &[f64], window: usize) -> Vec<f64> {
    let n = a.len();
    let mut out = vec![f64::NAN; n];
    if window < 2 || n < window || n != b.len() {
        return out;
    }
    for end in (window - 1)..n {
        let start = end + 1 - window;
        let mut sum_a = 0.0;
        let mut sum_b = 0.0;
        let mut sum_aa = 0.0;
        let mut sum_bb = 0.0;
        let mut sum_ab = 0.0;
        let mut any_nan = false;
        for k in start..=end {
            let ai = a[k];
            let bi = b[k];
            if !ai.is_finite() || !bi.is_finite() {
                any_nan = true;
                break;
            }
            sum_a += ai;
            sum_b += bi;
            sum_aa += ai * ai;
            sum_bb += bi * bi;
            sum_ab += ai * bi;
        }
        if any_nan {
            continue;
        }
        let w = window as f64;
        let mean_a = sum_a / w;
        let mean_b = sum_b / w;
        let var_a = (sum_aa / w) - mean_a * mean_a;
        let var_b = (sum_bb / w) - mean_b * mean_b;
        let cov_ab = (sum_ab / w) - mean_a * mean_b;
        if var_a <= 0.0 || var_b <= 0.0 {
            // Constant series → undefined correlation. Emit NaN
            // (caller's normaliser handles it).
            continue;
        }
        let denom = (var_a * var_b).sqrt();
        if denom > 0.0 {
            out[end] = (cov_ab / denom).clamp(-1.0, 1.0);
        }
    }
    out
}

/// Rolling z-score: `(x[i] - mean[i-window+1..i+1]) / std[...]`.
/// Indices `< window - 1` emit NaN. NaN propagates through the
/// window — any NaN in the window makes the output NaN.
fn rolling_zscore(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![f64::NAN; n];
    if window < 2 || n < window {
        return out;
    }
    for end in (window - 1)..n {
        let start = end + 1 - window;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut any_nan = false;
        for k in start..=end {
            let v = values[k];
            if !v.is_finite() {
                any_nan = true;
                break;
            }
            sum += v;
            sum_sq += v * v;
        }
        if any_nan {
            continue;
        }
        let w = window as f64;
        let mean = sum / w;
        let var = (sum_sq / w) - mean * mean;
        if var <= 0.0 {
            continue;
        }
        let std = var.sqrt();
        if std > 0.0 {
            out[end] = (values[end] - mean) / std;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ohlcv(closes: Vec<f64>, timestamps: Vec<i64>) -> Ohlcv {
        let n = closes.len();
        Ohlcv {
            timestamp: Some(timestamps),
            open: closes.clone(),
            high: closes.iter().map(|c| c + 0.01).collect(),
            low: closes.iter().map(|c| c - 0.01).collect(),
            close: closes,
            volume: Some(vec![1.0; n]),
        }
    }

    // ── sanitize_symbol_for_column_name ─────────────────────────────

    #[test]
    fn sanitize_removes_punctuation_and_uppercases() {
        assert_eq!(sanitize_symbol_for_column_name("eur/usd"), "EURUSD");
        assert_eq!(sanitize_symbol_for_column_name("EUR.USD"), "EURUSD");
        assert_eq!(sanitize_symbol_for_column_name("eur-usd"), "EURUSD");
        assert_eq!(sanitize_symbol_for_column_name("USDJPY"), "USDJPY");
    }

    // ── log_returns ─────────────────────────────────────────────────

    #[test]
    fn log_returns_first_is_zero_then_diffs() {
        let lr = log_returns(&[1.0, 1.05, 1.05]);
        assert!((lr[0] - 0.0).abs() < 1e-12);
        assert!((lr[1] - (1.05f64.ln() - 1.0f64.ln())).abs() < 1e-12);
        assert!((lr[2] - 0.0).abs() < 1e-12); // no change
    }

    #[test]
    fn log_returns_emits_nan_on_zero_or_negative_prices() {
        let lr = log_returns(&[1.0, 0.0, 1.0]);
        assert_eq!(lr[0], 0.0);
        assert!(lr[1].is_nan());
        assert!(lr[2].is_nan()); // prev was 0
    }

    // ── align_related_to_base_index ─────────────────────────────────

    #[test]
    fn align_forward_search_picks_largest_le_timestamp() {
        let base = ohlcv(vec![1.0, 1.0, 1.0], vec![100, 200, 300]);
        let related = ohlcv(vec![10.0, 20.0, 30.0], vec![100, 200, 300]);
        let aligned = align_related_to_base_index(&base, &related);
        assert_eq!(aligned, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn align_handles_late_starting_related_symbol() {
        // Related starts at timestamp 200 — base bar 0 (ts=100)
        // has no aligned related data → NaN.
        let base = ohlcv(vec![1.0, 1.0, 1.0], vec![100, 200, 300]);
        let related = ohlcv(vec![20.0, 30.0], vec![200, 300]);
        let aligned = align_related_to_base_index(&base, &related);
        assert!(aligned[0].is_nan());
        assert_eq!(aligned[1], 20.0);
        assert_eq!(aligned[2], 30.0);
    }

    #[test]
    fn align_forward_searches_when_related_drifts() {
        // Related has timestamps 100, 150 (irregular) — for base
        // bar at ts=200 we should pick related[1]=150 (largest <=200).
        let base = ohlcv(vec![1.0, 1.0], vec![100, 200]);
        let related = ohlcv(vec![10.0, 15.0], vec![100, 150]);
        let aligned = align_related_to_base_index(&base, &related);
        assert_eq!(aligned, vec![10.0, 15.0]);
    }

    // ── rolling_pearson_correlation ─────────────────────────────────

    #[test]
    fn perfect_positive_correlation_is_one() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // b = 2a
        let corr = rolling_pearson_correlation(&a, &b, 3);
        // First two are NaN (warmup), then perfect correlation.
        assert!(corr[0].is_nan());
        assert!(corr[1].is_nan());
        for &v in &corr[2..] {
            assert!((v - 1.0).abs() < 1e-9, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn perfect_negative_correlation_is_minus_one() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr = rolling_pearson_correlation(&a, &b, 3);
        for &v in &corr[2..] {
            assert!((v + 1.0).abs() < 1e-9, "expected -1.0, got {v}");
        }
    }

    #[test]
    fn correlation_with_nan_in_window_emits_nan() {
        let a = vec![1.0, 2.0, f64::NAN, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let corr = rolling_pearson_correlation(&a, &b, 3);
        // window ending at index 2 includes NaN → NaN
        assert!(corr[2].is_nan());
    }

    #[test]
    fn correlation_clamps_to_unit_interval() {
        // Floating-point drift can produce slight overshoot above
        // 1.0; the clamp guards against that.
        let a = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let b = a.clone();
        let corr = rolling_pearson_correlation(&a, &b, 4);
        for &v in &corr {
            if !v.is_nan() {
                assert!(v <= 1.0 && v >= -1.0);
            }
        }
    }

    // ── rolling_zscore ──────────────────────────────────────────────

    #[test]
    fn rolling_zscore_is_zero_for_constant_window() {
        let v = vec![5.0, 5.0, 5.0, 5.0];
        let z = rolling_zscore(&v, 3);
        // Constant window → var = 0 → NaN (undefined z-score).
        assert!(z[2].is_nan());
        assert!(z[3].is_nan());
    }

    #[test]
    fn rolling_zscore_recovers_known_value() {
        // Window: [1, 2, 3]. mean=2, var=2/3, std≈0.816.
        // Last value 3 → z = (3 - 2) / 0.816 ≈ 1.225.
        let v = vec![1.0, 2.0, 3.0];
        let z = rolling_zscore(&v, 3);
        assert!(z[0].is_nan());
        assert!(z[1].is_nan());
        let expected = (3.0 - 2.0) / (2.0_f64 / 3.0).sqrt();
        assert!(
            (z[2] - expected).abs() < 1e-9,
            "expected {expected}, got {}",
            z[2]
        );
    }

    // ── compute_cross_pair_features ─────────────────────────────────

    #[test]
    fn empty_base_returns_no_columns() {
        let base = ohlcv(vec![], vec![]);
        let related = ohlcv(vec![1.0, 2.0], vec![1, 2]);
        let cols = compute_cross_pair_features(
            &base,
            &[("XYZ".to_string(), &related)],
            DEFAULT_CROSS_PAIR_WINDOWS,
        );
        assert!(cols.is_empty());
    }

    #[test]
    fn empty_related_list_returns_no_columns() {
        let base = ohlcv(vec![1.0, 1.0, 1.0], vec![1, 2, 3]);
        let cols = compute_cross_pair_features(&base, &[], DEFAULT_CROSS_PAIR_WINDOWS);
        assert!(cols.is_empty());
    }

    #[test]
    fn produces_expected_column_names_and_count() {
        // 1 related symbol × (4 windows for xcorr + 1 spread + 4 windows for spread_z)
        // = 9 columns.
        let base = ohlcv(vec![1.0; 200], (0..200).collect());
        let related = ohlcv(vec![1.0; 200], (0..200).collect());
        let cols = compute_cross_pair_features(
            &base,
            &[("GBPUSD".to_string(), &related)],
            DEFAULT_CROSS_PAIR_WINDOWS,
        );
        assert_eq!(cols.len(), 4 + 1 + 4);
        let names: Vec<&str> = cols.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"xcorr_GBPUSD_10"));
        assert!(names.contains(&"xcorr_GBPUSD_20"));
        assert!(names.contains(&"xcorr_GBPUSD_50"));
        assert!(names.contains(&"xcorr_GBPUSD_100"));
        assert!(names.contains(&"spread_GBPUSD"));
        assert!(names.contains(&"spread_z_GBPUSD_10"));
        assert!(names.contains(&"spread_z_GBPUSD_20"));
        assert!(names.contains(&"spread_z_GBPUSD_50"));
        assert!(names.contains(&"spread_z_GBPUSD_100"));
    }

    #[test]
    fn output_columns_match_base_length() {
        let base = ohlcv(vec![1.0; 50], (0..50).collect());
        let related = ohlcv(vec![1.0; 50], (0..50).collect());
        let cols =
            compute_cross_pair_features(&base, &[("USDJPY".to_string(), &related)], &[10, 20]);
        for (name, values) in &cols {
            assert_eq!(values.len(), 50, "column {name} has wrong length");
        }
    }

    #[test]
    fn perfectly_correlated_symbols_emit_unit_correlation() {
        // Generate identical price series — correlation must be 1.0
        // (or NaN warmup) at every window.
        let prices: Vec<f64> = (0..50).map(|i| 1.0 + (i as f64) * 0.001).collect();
        let timestamps: Vec<i64> = (0..50).collect();
        let base = ohlcv(prices.clone(), timestamps.clone());
        let related = ohlcv(prices.clone(), timestamps.clone());
        let cols = compute_cross_pair_features(&base, &[("PERFECT".to_string(), &related)], &[10]);
        let xcorr = cols
            .iter()
            .find(|(n, _)| n == "xcorr_PERFECT_10")
            .expect("xcorr column present");
        // After warmup, expect ~1.0 everywhere.
        for &v in xcorr.1.iter().skip(15) {
            assert!((v - 1.0).abs() < 1e-6, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn multiple_related_symbols_emit_disjoint_column_names() {
        let base = ohlcv(vec![1.0; 30], (0..30).collect());
        let r1 = ohlcv(vec![1.0; 30], (0..30).collect());
        let r2 = ohlcv(vec![1.0; 30], (0..30).collect());
        let cols = compute_cross_pair_features(
            &base,
            &[("GBPUSD".to_string(), &r1), ("USDJPY".to_string(), &r2)],
            &[10],
        );
        let names: Vec<&str> = cols.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"xcorr_GBPUSD_10"));
        assert!(names.contains(&"xcorr_USDJPY_10"));
        assert!(names.contains(&"spread_GBPUSD"));
        assert!(names.contains(&"spread_USDJPY"));
    }
}
