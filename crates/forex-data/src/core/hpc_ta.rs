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
    let timestamps = vec![0i64; n];
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
