use super::super::Ohlcv;
use crate::core::all_indicators::ALL_INDICATORS;
use vector_ta::indicators::dispatch::{
    IndicatorComputeRequest, IndicatorDataRef, IndicatorSeries, compute_cpu,
};
use vector_ta::utilities::data_loader::Candles;
use vector_ta::utilities::enums::Kernel;

/// Computes ALL 340+ Technical Indicators automatically using VectorTA's Dispatch Engine.
/// Multi-output indicators are automatically decomposed into separate named columns.
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

        if let Ok(output) = compute_cpu(req) {
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
            if let Ok(output) = compute_cpu(req) {
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
