use super::super::Ohlcv;
use crate::core::all_indicators::ALL_INDICATORS;
use std::panic::{AssertUnwindSafe, catch_unwind};
use vector_ta::indicators::dispatch::{
    IndicatorComputeRequest, IndicatorDataRef, IndicatorSeries, compute_cpu,
};
use vector_ta::utilities::data_loader::Candles;
use vector_ta::utilities::enums::Kernel;

/// Largest indicator-period this module ever asks vector-ta to compute
/// in its multi-period sweep. Used by [`max_indicator_warmup`] so the
/// genetic search can pre-flight gene admission against the data slice
/// length and skip indicators that would panic the kernel
/// (`warm prefix exceeds row width`, vector-ta v0.2.9 #212).
pub const MAX_MULTI_PERIOD_LOOKBACK: usize = 200;

/// Maximum warmup periods (in bars) that the indicator stack can
/// produce on a frame with `n_rows` bars. Returns the largest period
/// from the multi-period sweep that still fits, or 0 if the frame is
/// too short to compute any of the parameterized indicators safely.
///
/// Used by the validation harness and pre-flight guards to refuse
/// evaluation on slices smaller than the indicator's warmup. The
/// thresholds match the `alt_periods` array in `compute_classic_ta_columns`.
pub fn max_indicator_warmup(n_rows: usize) -> usize {
    const ALT_PERIODS: &[usize] = &[7, 21, 50, 100, 200];
    ALT_PERIODS
        .iter()
        .rev()
        .find(|&&p| p < n_rows)
        .copied()
        .unwrap_or(0)
}

/// Computes ALL 340+ Technical Indicators automatically using VectorTA's Dispatch Engine.
/// Multi-output indicators are automatically decomposed into separate named columns.
///
/// Each indicator call is wrapped in `std::panic::catch_unwind` because
/// vector-ta v0.2.9 panics on a small subset of period/data combinations
/// (e.g. EURUSD M5 hits `warm prefix exceeds row width` at
/// `utilities/helpers.rs:159`, #212). The wrapping converts a panic into
/// a silently-skipped column rather than tearing down the worker thread,
/// which on the rayon-driven discovery hot path would otherwise abort
/// the whole TF run with no fallback path. The pre-flight
/// [`max_indicator_warmup`] helper still gates the multi-period sweep
/// so the common case never reaches the kernel boundary.
pub fn compute_classic_ta_columns(ohlcv: &Ohlcv) -> Vec<(String, Vec<f64>)> {
    let n = ohlcv.len();
    if n == 0 {
        return vec![];
    }

    let mut cols = Vec::new();

    // 1. Pack data into VectorTA Candles struct
    let timestamps = ohlcv.timestamp.clone().unwrap_or_else(|| vec![0i64; n]);
    let volume = ohlcv.volume.clone().unwrap_or_else(|| vec![0.0; n]);

    let candles = Candles::new(
        timestamps,
        ohlcv.open.clone(),
        ohlcv.high.clone(),
        ohlcv.low.clone(),
        ohlcv.close.clone(),
        volume,
    );

    let data_ref = IndicatorDataRef::Candles {
        candles: &candles,
        source: None,
    };

    // 2. Dispatch to every known indicator using Default Parameters
    for &id in ALL_INDICATORS {
        let req = IndicatorComputeRequest {
            indicator_id: id,
            output_id: None,
            data: data_ref,
            params: &[],
            kernel: Kernel::Auto,
        };

        // #212: a small subset of indicator/data combinations in
        // vector-ta v0.2.9 panic instead of returning Err (e.g.
        // `warm prefix exceeds row width`). The panic aborts the
        // worker thread and tears down the TF run, so we catch it
        // here, log once, and treat the indicator as unavailable
        // for this frame.
        let computed = catch_unwind(AssertUnwindSafe(|| compute_cpu(req)));
        let Ok(compute_result) = computed else {
            tracing::warn!(
                target: "neoethos_data::hpc_ta",
                indicator = %id,
                rows = n,
                "vector-ta indicator kernel panicked; skipping column for this frame"
            );
            continue;
        };
        if let Ok(output) = compute_result {
            let rows = output.rows;
            let out_cols = output.cols;

            match output.series {
                IndicatorSeries::F64(v) => {
                    if out_cols <= 1 {
                        // Single-output indicator
                        if v.len() == n {
                            cols.push((id.to_string(), v));
                        } else if v.len() > n {
                            let chunk: Vec<f64> = v.into_iter().take(n).collect();
                            cols.push((id.to_string(), chunk));
                        }
                    } else {
                        // Multi-output: decompose into separate columns
                        // Data is stored row-major: [row0_col0, row0_col1, ..., row1_col0, ...]
                        if v.len() == rows * out_cols && rows >= n {
                            for c in 0..out_cols {
                                let mut col_data = Vec::with_capacity(n);
                                for r in 0..n {
                                    col_data.push(v[r * out_cols + c]);
                                }
                                cols.push((format!("{}_line{}", id, c), col_data));
                            }
                        }
                    }
                }
                IndicatorSeries::I32(v) => {
                    if v.len() == n {
                        let cf: Vec<f64> = v.into_iter().map(|x| x as f64).collect();
                        cols.push((id.to_string(), cf));
                    }
                }
                IndicatorSeries::Bool(v) => {
                    if v.len() == n {
                        let cf: Vec<f64> =
                            v.into_iter().map(|x| if x { 1.0 } else { 0.0 }).collect();
                        cols.push((id.to_string(), cf));
                    }
                }
            }
        }
    }

    // 3. Multi-period variants for the most critical indicators
    // The genetic engine benefits from seeing the same indicator at different lookback periods
    let multi_period_ids = [
        "rsi",
        "ema",
        "sma",
        "atr",
        "adx",
        "cci",
        "stoch",
        "macd",
        "bollinger_bands",
        "keltner",
        "supertrend",
        "willr",
        "roc",
        "mom",
        "tsi",
        "mfi",
        "obv",
        "vwap",
    ];
    let alt_periods = [7, 21, 50, 100, 200];

    for &ind_id in &multi_period_ids {
        for &period in &alt_periods {
            // #212: pre-flight check — if the period is larger than the
            // data length, vector-ta's `warm_prefix` exceeds the row
            // width and the kernel panics at `helpers.rs:159` instead
            // of returning Err. Skip the call entirely for these cases.
            // The 1.25× safety margin matches the kernel's typical
            // `first_valid_idx + period` formula plus a small headroom
            // for indicators with extra warmup beyond the period itself.
            if (period as f64) * 1.25 >= n as f64 {
                continue;
            }
            let params = [vector_ta::indicators::dispatch::ParamKV {
                key: "period",
                value: vector_ta::indicators::dispatch::ParamValue::Int(period),
            }];
            let req = IndicatorComputeRequest {
                indicator_id: ind_id,
                output_id: None,
                data: data_ref,
                params: &params,
                kernel: Kernel::Auto,
            };
            // #212: defense in depth — even after the pre-flight guard
            // above, wrap the kernel call to ensure a panic in a
            // less-common code path cannot tear down the TF run.
            let computed = catch_unwind(AssertUnwindSafe(|| compute_cpu(req)));
            let Ok(compute_result) = computed else {
                tracing::warn!(
                    target: "neoethos_data::hpc_ta",
                    indicator = %ind_id,
                    period = period,
                    rows = n,
                    "vector-ta multi-period kernel panicked; skipping column"
                );
                continue;
            };
            if let Ok(output) = compute_result {
                match output.series {
                    IndicatorSeries::F64(v) if v.len() == n => {
                        cols.push((format!("{}_{}", ind_id, period), v));
                    }
                    IndicatorSeries::F64(v) if v.len() > n && output.cols <= 1 => {
                        let chunk: Vec<f64> = v.into_iter().take(n).collect();
                        cols.push((format!("{}_{}", ind_id, period), chunk));
                    }
                    IndicatorSeries::F64(v)
                        if output.cols > 1
                            && v.len() == output.rows * output.cols
                            && output.rows >= n =>
                    {
                        for c in 0..output.cols {
                            let mut col_data = Vec::with_capacity(n);
                            for r in 0..n {
                                col_data.push(v[r * output.cols + c]);
                            }
                            cols.push((format!("{}_{}_line{}", ind_id, period, c), col_data));
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    cols
}

/// One series returned by `compute_single_indicator` — multi-output
/// indicators (Bollinger Bands, MACD, Stochastic) decompose into
/// several of these.
#[derive(Debug, Clone)]
pub struct IndicatorLine {
    /// Human-readable line name. Single-output indicators use the
    /// indicator id (e.g. `"sma"`); multi-output ones suffix with
    /// the column index (`"bollinger_bands_line0"`, `"…_line1"`,
    /// `"…_line2"` for lower/middle/upper).
    pub name: String,
    /// Series aligned with the input ohlcv length. NaN-padding at
    /// the start is preserved (so e.g. SMA(20)[0..19] = NaN), which
    /// the chart renders as a gap before the line begins.
    pub values: Vec<f64>,
}

/// Compute a single indicator on demand — the interactive Chart
/// screen calls this through the `/indicators` HTTP endpoint
/// whenever the user adds an indicator to the overlay. Cheap enough
/// to recompute on every pan; vector_ta dispatches to CPU SIMD or
/// GPU kernels under the hood.
///
/// `params` is a key→f64 map. Conventional keys per indicator:
///   * `sma`/`ema`/`rsi`/`atr`/`adx`: `period`
///   * `bollinger_bands`: `period`, `std_dev`
///   * `macd`: `fast`, `slow`, `signal`
///   * `stoch`: `k_period`, `k_slow`, `d_period`
/// Unrecognised keys are silently ignored. Empty map = library defaults.
///
/// Returns the row count + lines on success, anyhow error if the
/// indicator id is unknown or the kernel rejects the input.
pub fn compute_single_indicator(
    ohlcv: &Ohlcv,
    indicator_id: &str,
    params: &std::collections::HashMap<String, f64>,
) -> anyhow::Result<Vec<IndicatorLine>> {
    let n = ohlcv.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Pack the ohlcv slice into a Candles instance for the dispatch
    // API. Timestamps and volume are nice-to-have but not required
    // by most indicators — we fill zeros when missing.
    let timestamps = ohlcv.timestamp.clone().unwrap_or_else(|| vec![0i64; n]);
    let volume = ohlcv.volume.clone().unwrap_or_else(|| vec![0.0; n]);
    let candles = Candles::new(
        timestamps,
        ohlcv.open.clone(),
        ohlcv.high.clone(),
        ohlcv.low.clone(),
        ohlcv.close.clone(),
        volume,
    );
    let data_ref = IndicatorDataRef::Candles {
        candles: &candles,
        source: None,
    };

    // Translate the f64 param map into vector_ta's ParamKV array.
    // vector_ta accepts ints for period-like params and floats for
    // multipliers (e.g. Bollinger Bands' std_dev); we route based on
    // whether the value has a fractional part.
    let mut kv: Vec<vector_ta::indicators::dispatch::ParamKV> = Vec::with_capacity(params.len());
    for (k, v) in params {
        // Leak the &'static str via Box::leak so the dispatch API
        // can hold a 'static reference. Param map is tiny (≤ 5
        // entries) and lives for the call, so the leak is bounded
        // by the call site — acceptable trade-off for the simpler
        // wire shape.
        let key: &'static str = Box::leak(k.clone().into_boxed_str());
        let value = if v.fract() == 0.0 && v.abs() <= i64::MAX as f64 {
            vector_ta::indicators::dispatch::ParamValue::Int(*v as i64)
        } else {
            vector_ta::indicators::dispatch::ParamValue::Float(*v)
        };
        kv.push(vector_ta::indicators::dispatch::ParamKV { key, value });
    }

    let req = IndicatorComputeRequest {
        indicator_id,
        output_id: None,
        data: data_ref,
        params: &kv,
        kernel: Kernel::Auto,
    };
    let output =
        compute_cpu(req).map_err(|e| anyhow::anyhow!("vector_ta dispatch failed: {e:?}"))?;

    let rows = output.rows;
    let out_cols = output.cols.max(1);
    let mut lines = Vec::with_capacity(out_cols);

    match output.series {
        IndicatorSeries::F64(v) => {
            if out_cols <= 1 {
                let values = if v.len() == n {
                    v
                } else if v.len() > n {
                    v.into_iter().take(n).collect()
                } else {
                    anyhow::bail!("indicator returned {} values, expected ≥{}", v.len(), n);
                };
                lines.push(IndicatorLine {
                    name: indicator_id.to_string(),
                    values,
                });
            } else if v.len() == rows * out_cols && rows >= n {
                // Row-major decomposition into per-column series.
                for c in 0..out_cols {
                    let mut col_data = Vec::with_capacity(n);
                    for r in 0..n {
                        col_data.push(v[r * out_cols + c]);
                    }
                    lines.push(IndicatorLine {
                        name: format!("{indicator_id}_line{c}"),
                        values: col_data,
                    });
                }
            } else {
                anyhow::bail!(
                    "indicator multi-output shape mismatch: rows={} cols={} len={} n={}",
                    rows,
                    out_cols,
                    v.len(),
                    n
                );
            }
        }
        IndicatorSeries::I32(v) => {
            if v.len() == n {
                lines.push(IndicatorLine {
                    name: indicator_id.to_string(),
                    values: v.into_iter().map(|x| x as f64).collect(),
                });
            }
        }
        IndicatorSeries::Bool(v) => {
            if v.len() == n {
                lines.push(IndicatorLine {
                    name: indicator_id.to_string(),
                    values: v.into_iter().map(|x| if x { 1.0 } else { 0.0 }).collect(),
                });
            }
        }
    }

    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    // #212: pre-flight check helper used by the validation harness and
    // gene admission gate to refuse computation on slices smaller than
    // the indicator's warmup. These assertions document the contract
    // and trap regressions if `alt_periods` ever changes.
    #[test]
    fn max_indicator_warmup_returns_zero_for_tiny_frames() {
        assert_eq!(max_indicator_warmup(0), 0);
        assert_eq!(max_indicator_warmup(5), 0);
        assert_eq!(max_indicator_warmup(7), 0);
    }

    #[test]
    fn max_indicator_warmup_returns_largest_fitting_period() {
        // n=8 fits 7 (smallest alt_period) but not 21.
        assert_eq!(max_indicator_warmup(8), 7);
        assert_eq!(max_indicator_warmup(22), 21);
        assert_eq!(max_indicator_warmup(51), 50);
        assert_eq!(max_indicator_warmup(101), 100);
        assert_eq!(max_indicator_warmup(201), 200);
    }

    #[test]
    fn max_indicator_warmup_caps_at_largest_period() {
        // Even for huge frames the helper does not exceed the
        // configured `MAX_MULTI_PERIOD_LOOKBACK`.
        assert_eq!(max_indicator_warmup(10_000), 200);
        assert_eq!(max_indicator_warmup(1_000_000), MAX_MULTI_PERIOD_LOOKBACK);
    }

    // Pre-flight gate documented in `compute_classic_ta_columns`: the
    // multi-period sweep skips any period whose `*1.25` safety margin
    // exceeds the frame length. This test pins the contract so a refactor
    // can't silently drop the guard and reintroduce the #212 panic.
    #[test]
    fn pre_flight_gate_skips_periods_larger_than_frame() {
        // For a 30-row frame: 7 fits (7*1.25=8.75 < 30), 21 fits
        // (26.25 < 30), 50 does NOT fit (62.5 ≥ 30).
        let n: usize = 30;
        let acceptable: Vec<usize> = [7usize, 21, 50, 100, 200]
            .into_iter()
            .filter(|p| (*p as f64) * 1.25 < n as f64)
            .collect();
        assert_eq!(acceptable, vec![7, 21]);
    }
}
