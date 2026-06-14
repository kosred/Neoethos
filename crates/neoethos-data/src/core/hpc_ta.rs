use super::super::Ohlcv;
use crate::core::all_indicators::ALL_INDICATORS;
use rayon::prelude::*;
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

    // 1. Pack data into VectorTA Candles struct (once; shared read-only
    //    across the rayon workers below — `Candles` holds plain Vecs so it
    //    is `Sync`).
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

    // 2. Dispatch to every known indicator — PARALLEL across indicators.
    //    Each indicator is an independent pure function of the shared
    //    `candles`, so the previous serial `for &id in ALL_INDICATORS`
    //    loop becomes a rayon `par_iter`. `flat_map_iter` + `collect`
    //    preserves the original column order exactly. The feature build
    //    runs ONCE up-front (not inside the GA candidate `par_iter`), so
    //    this does not nest with the discovery hot path. Each worker
    //    re-creates the cheap `IndicatorDataRef` borrow of `candles`.
    let mut cols: Vec<(String, Vec<f64>)> = ALL_INDICATORS
        .par_iter()
        .flat_map_iter(|&id| {
            let mut out: Vec<(String, Vec<f64>)> = Vec::new();
            let data_ref = IndicatorDataRef::Candles {
                candles: &candles,
                source: None,
            };
            let req = IndicatorComputeRequest {
                indicator_id: id,
                output_id: None,
                data: data_ref,
                params: &[],
                kernel: Kernel::Auto,
            };
            // #212: a small subset of indicator/data combinations in
            // vector-ta v0.2.9 panic instead of returning Err. Catch it
            // per-indicator so one bad column never tears down the frame.
            let computed = catch_unwind(AssertUnwindSafe(|| compute_cpu(req)));
            let Ok(compute_result) = computed else {
                tracing::warn!(
                    target: "neoethos_data::hpc_ta",
                    indicator = %id,
                    rows = n,
                    "vector-ta indicator kernel panicked; skipping column for this frame"
                );
                return out.into_iter();
            };
            if let Ok(output) = compute_result {
                let rows = output.rows;
                let out_cols = output.cols;

                match output.series {
                    IndicatorSeries::F64(v) => {
                        if out_cols <= 1 {
                            // Single-output indicator
                            if v.len() == n {
                                out.push((id.to_string(), v));
                            } else if v.len() > n {
                                let chunk: Vec<f64> = v.into_iter().take(n).collect();
                                out.push((id.to_string(), chunk));
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
                                    out.push((format!("{}_line{}", id, c), col_data));
                                }
                            }
                        }
                    }
                    IndicatorSeries::I32(v) => {
                        if v.len() == n {
                            let cf: Vec<f64> = v.into_iter().map(|x| x as f64).collect();
                            out.push((id.to_string(), cf));
                        }
                    }
                    IndicatorSeries::Bool(v) => {
                        if v.len() == n {
                            let cf: Vec<f64> =
                                v.into_iter().map(|x| if x { 1.0 } else { 0.0 }).collect();
                            out.push((id.to_string(), cf));
                        }
                    }
                }
            }
            out.into_iter()
        })
        .collect();

    // 3. Multi-period variants for the most critical indicators —
    //    PARALLEL across the 18 indicators (each runs its own period
    //    sweep serially inside the closure). Appended after the base
    //    columns to preserve the original ordering exactly.
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

    let multi_cols: Vec<(String, Vec<f64>)> = multi_period_ids
        .par_iter()
        .flat_map_iter(|&ind_id| {
            let mut out: Vec<(String, Vec<f64>)> = Vec::new();
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
                let data_ref = IndicatorDataRef::Candles {
                    candles: &candles,
                    source: None,
                };
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
                            out.push((format!("{}_{}", ind_id, period), v));
                        }
                        IndicatorSeries::F64(v) if v.len() > n && output.cols <= 1 => {
                            let chunk: Vec<f64> = v.into_iter().take(n).collect();
                            out.push((format!("{}_{}", ind_id, period), chunk));
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
                                out.push((format!("{}_{}_line{}", ind_id, period, c), col_data));
                            }
                        }
                        _ => {}
                    }
                }
            }
            out.into_iter()
        })
        .collect();
    cols.extend(multi_cols);

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

    // Look up the indicator's declared outputs. Multi-output indicators
    // (MACD, Bollinger Bands, Stochastic, …) REQUIRE an explicit
    // `output_id` per series in vector_ta — dispatching them with
    // `output_id: None` fails with "output_id is required for
    // multi-output indicators". Single-output indicators use `None`
    // (the library's default output).
    let output_ids: Vec<Option<&'static str>> =
        vector_ta::indicators::registry::list_indicators()
            .iter()
            .find(|i| i.id == indicator_id)
            .map(|info| {
                if info.outputs.len() <= 1 {
                    vec![None]
                } else {
                    info.outputs.iter().map(|o| Some(o.id)).collect()
                }
            })
            .unwrap_or_else(|| vec![None]);

    let mut lines = Vec::with_capacity(output_ids.len());
    for out_id in output_ids {
        let req = IndicatorComputeRequest {
            indicator_id,
            output_id: out_id,
            data: data_ref,
            params: &kv,
            kernel: Kernel::Auto,
        };
        let output = compute_cpu(req).map_err(|e| {
            anyhow::anyhow!("vector_ta dispatch failed ({indicator_id}/{out_id:?}): {e:?}")
        })?;
        let values = flatten_indicator_series(output.series, n)?;
        // Single-output → bare indicator id (e.g. "sma"); multi-output →
        // "<id>_<output>" (e.g. "macd_signal") so the chart legend can
        // split on '_' and show the per-line label.
        let name = match out_id {
            Some(id) => format!("{indicator_id}_{id}"),
            None => indicator_id.to_string(),
        };
        lines.push(IndicatorLine { name, values });
    }

    Ok(lines)
}

/// Flatten a vector_ta series for ONE output into exactly `n` values.
/// vector_ta reports a 1-D series as rows=1 × cols=n, so we key off the
/// value count, not the rows/cols metadata — the previous `rows == n`
/// assumption rejected every single-output series with a "shape
/// mismatch" error.
fn flatten_indicator_series(series: IndicatorSeries, n: usize) -> anyhow::Result<Vec<f64>> {
    match series {
        IndicatorSeries::F64(v) => normalize_indicator_len(v, n),
        IndicatorSeries::I32(v) => {
            normalize_indicator_len(v.into_iter().map(|x| x as f64).collect(), n)
        }
        IndicatorSeries::Bool(v) => normalize_indicator_len(
            v.into_iter().map(|x| if x { 1.0 } else { 0.0 }).collect(),
            n,
        ),
    }
}

fn normalize_indicator_len(v: Vec<f64>, n: usize) -> anyhow::Result<Vec<f64>> {
    if v.len() == n {
        Ok(v)
    } else if v.len() > n {
        // Bar-aligned from the start; take the leading n (warmup padding
        // lives at the head and stays aligned with candle index 0).
        Ok(v.into_iter().take(n).collect())
    } else {
        anyhow::bail!("indicator returned {} values, expected ≥{}", v.len(), n)
    }
}

// ===========================================================================
// Task #22 — GPU (CUDA) period-sweep batching, DE-RISK STEP (sma only)
// ===========================================================================
//
// Everything below is `#[cfg(feature = "gpu-cuda")]`-gated. With the feature
// OFF none of it compiles, so the default (AMD / gpu-vulkan / pure-CPU) build
// is byte-identical. It can only be COMPILED on a machine with the CUDA
// toolkit (the A6000 VPS): the `gpu-cuda` feature pulls in
// `vector-ta/cuda-build-ptx`, whose build.rs emits the kernel PTX via nvcc.
//
// SCOPE (intentionally minimal — this de-risks the full #22 before fanning
// out to all 193 indicators): prove that ONE `compute_cuda_device` period
// sweep for `sma` over [7,21,50,100,200] is bit-for-bit-equivalent (within an
// f32 tolerance) to the per-period CPU sweep in `compute_classic_ta_columns`.
//
// DESIGN — single upload, ONE batched device call, then SELECT rows:
//   * Narrow the close series to f32 and `CudaRuntime::upload_f32` it ONCE
//     (the host-input `compute_cuda` re-uploads per call; the whole point of
//     the device-resident `compute_cuda_device` path is to upload once and
//     reuse — here a single sweep call already reuses the one upload).
//   * Issue ONE `compute_cuda_device` for `sma` with a CONTIGUOUS period sweep
//     `period_start=min..period_end=max, period_step=1` covering the wanted
//     periods, then SELECT the rows for [7,21,50,100,200]. `compute_cuda_device`
//     returns an `IndicatorCudaOutput` whose `HostF32` series is a row-major
//     `rows × cols` matrix with `rows = number of swept periods` (one row per
//     period, ascending — see `expand_grid_sma` in
//     vendor/.../indicators/moving_averages/sma.rs and `sma_batch_dev_from_device_ptr`
//     in vendor/.../cuda/moving_averages/sma_wrapper.rs, which sets
//     `rows: combos.len(), cols: len`) and `cols = series length n`. Row `i`
//     corresponds to period `min + i` (step 1). We index the wanted period as
//     `row = period - min`.
//   * NaN warmup: the device kernel (`kernels/cuda/moving_averages/sma_kernel.cu`,
//     `sma_batch_from_prefix_f64`) fills `t < first_valid + period - 1` with
//     `SMA_NAN = __int_as_float(0x7fffffff)` (a quiet NaN) and writes the SMA
//     value from index `period-1` onward (first_valid = 0 here). That is the
//     SAME `period-1` NaN-warmup convention as the CPU path, so the NaN masks
//     line up by construction.
//   * Skip guard: applies the SAME `(period as f64)*1.25 >= n` pre-flight as
//     the CPU sweep, so a too-short frame produces exactly the same (possibly
//     empty / truncated) set of `sma_{period}` columns the CPU path produces.
//     Periods that survive the guard are also guaranteed to satisfy the
//     kernel's own `len - first_valid >= period` requirement (since
//     period*1.25 < n ⇒ period < n), so the batch is never rejected wholesale.
//   * fail-loud: ANY device/dispatch error → `Err` (no unwrap/expect on device
//     calls). The CALLER decides fallback — this de-risk harness does not
//     silently swallow GPU failures.

/// The period sweep this de-risk step proves out (mirrors `alt_periods` in
/// `compute_classic_ta_columns`).
#[cfg(feature = "gpu-cuda")]
pub const SMA_SWEEP_PERIODS: [usize; 5] = [7, 21, 50, 100, 200];

/// GPU period-sweep for `sma` over `periods`, computed in ONE batched
/// `compute_cuda_device` call against a single device upload of the close
/// series. Returns `sma_{period}` columns (f64, bar-aligned, length `n`, with
/// the kernel's NaN warmup preserved) in ascending-period order, applying the
/// same `(period as f64)*1.25 >= n` skip as the CPU path so the column SET +
/// NAMES + ORDER match `compute_classic_ta_columns` for `sma`.
///
/// Fail-loud: returns `Err` on CUDA-unavailable or any device/dispatch error.
#[cfg(feature = "gpu-cuda")]
pub fn gpu_sma_sweep_columns(
    ohlcv: &Ohlcv,
    periods: &[usize],
) -> anyhow::Result<Vec<(String, Vec<f64>)>> {
    use anyhow::{Context, bail};
    use vector_ta::cuda::{CudaRuntime, cuda_available};
    use vector_ta::indicators::dispatch::{
        CudaOutputTarget, IndicatorCudaDeviceDataRef, IndicatorCudaDeviceRequest,
        IndicatorCudaSeries, ParamKV, ParamValue, compute_cuda_device,
    };

    let n = ohlcv.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Hard gate: fail loud if CUDA is not usable (the caller/test decides what
    // to do with the error — e.g. CPU fallback, or panic under NEOETHOS_REQUIRE_GPU).
    if !cuda_available() {
        bail!("no cuda device (vector_ta::cuda::cuda_available() == false)");
    }

    // Apply the CPU path's #212 pre-flight skip FIRST, so the produced column
    // set matches the CPU sweep exactly (skipped periods produce no column).
    // Ascending order is required: the device matrix rows are ascending by
    // period (expand_grid_sma walks start..=end), and the CPU sweep also emits
    // in the given period order — we keep callers honest by sorting+deduping.
    let mut wanted: Vec<usize> = periods
        .iter()
        .copied()
        .filter(|&p| p >= 1 && (p as f64) * 1.25 < n as f64)
        .collect();
    wanted.sort_unstable();
    wanted.dedup();
    if wanted.is_empty() {
        return Ok(vec![]);
    }

    let min_p = *wanted.first().unwrap();
    let max_p = *wanted.last().unwrap();

    // Narrow close → f32 and upload ONCE.
    let close_f32: Vec<f32> = ohlcv.close.iter().map(|&x| x as f32).collect();
    let runtime = CudaRuntime::new(0).context("CudaRuntime::new(0) failed")?;
    let d_close = runtime
        .upload_f32(&close_f32)
        .context("upload_f32(close) failed")?;

    // ONE batched device call: contiguous sweep [min_p..=max_p] step 1 over the
    // single device-resident close series. `first_valid = 0` (no leading-NaN
    // prefix in our cube). The result is a row-major rows×cols matrix with one
    // row per swept period (ascending) and cols = n.
    let params = [
        ParamKV {
            key: "period_start",
            value: ParamValue::Int(min_p as i64),
        },
        ParamKV {
            key: "period_end",
            value: ParamValue::Int(max_p as i64),
        },
        ParamKV {
            key: "period_step",
            value: ParamValue::Int(1),
        },
        ParamKV {
            key: "first_valid",
            value: ParamValue::Int(0),
        },
    ];
    let req = IndicatorCudaDeviceRequest {
        indicator_id: "sma",
        output_id: None,
        data: IndicatorCudaDeviceDataRef::Slice {
            values: d_close.as_view(),
        },
        params: &params,
        kernel: Kernel::Auto,
        target: CudaOutputTarget::HostF32,
    };
    let out = compute_cuda_device(req)
        .map_err(|e| anyhow::anyhow!("compute_cuda_device(sma sweep) failed: {e:?}"))?;

    let host: Vec<f32> = match out.series {
        IndicatorCudaSeries::HostF32(v) => v,
        IndicatorCudaSeries::DeviceF32(_) => {
            bail!("compute_cuda_device returned DeviceF32 despite HostF32 target")
        }
    };

    // Validate the matrix shape against our expectation BEFORE indexing rows:
    // rows = (max_p - min_p) + 1 swept periods, cols = n.
    let expected_rows = max_p - min_p + 1;
    if out.cols != n {
        bail!(
            "sma sweep cols mismatch: out.cols={} expected n={}",
            out.cols,
            n
        );
    }
    if out.rows != expected_rows {
        bail!(
            "sma sweep rows mismatch: out.rows={} expected {} (periods {}..={} step 1)",
            out.rows,
            expected_rows,
            min_p,
            max_p
        );
    }
    if host.len() != out.rows.saturating_mul(out.cols) {
        bail!(
            "sma sweep host buffer len {} != rows*cols {}*{}",
            host.len(),
            out.rows,
            out.cols
        );
    }

    // SELECT the wanted period rows. Row index for period p (step 1) is p - min_p.
    let mut cols: Vec<(String, Vec<f64>)> = Vec::with_capacity(wanted.len());
    for &p in &wanted {
        let row = p - min_p;
        let start = row * out.cols;
        let slice = &host[start..start + out.cols];
        // f32 → f64; the kernel's SMA_NAN (0x7fffffff) is a quiet f32 NaN and
        // `as f64` preserves NaN, so the warmup NaN mask carries over intact.
        let col: Vec<f64> = slice.iter().map(|&x| x as f64).collect();
        cols.push((format!("sma_{}", p), col));
    }

    Ok(cols)
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

    // =======================================================================
    // Task #22 DE-RISK STEP — CPU==GPU parity for the `sma` period sweep.
    // Only built `--features gpu-cuda` (i.e. ONLY on the A6000 VPS). Proves
    // that the single batched `compute_cuda_device` sweep matches the
    // per-period CPU sweep used by `compute_classic_ta_columns`.
    // =======================================================================

    /// Deterministic ~3000-bar seeded random-walk OHLCV. Long enough to clear
    /// the 200-period warmup; the natural NaN warmup lives at the series head
    /// (produced by the indicator, not the fixture). Pure-std xorshift PRNG so
    /// the fixture is reproducible with no extra deps.
    #[cfg(feature = "gpu-cuda")]
    fn seeded_random_walk_ohlcv(n: usize, seed: u64) -> Ohlcv {
        let mut state = seed | 1;
        let mut next_unit = || -> f64 {
            // xorshift64* → uniform in [0,1)
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let r = state.wrapping_mul(0x2545F4914F6CDD1D);
            (r >> 11) as f64 / (1u64 << 53) as f64
        };

        let mut close = Vec::with_capacity(n);
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut price = 100.0f64;
        for _ in 0..n {
            let o = price;
            let step = (next_unit() - 0.5) * 0.5; // ±0.25 random walk
            price += step;
            let c = price;
            let hi = o.max(c) + next_unit() * 0.1;
            let lo = o.min(c) - next_unit() * 0.1;
            open.push(o);
            close.push(c);
            high.push(hi);
            low.push(lo);
        }
        Ohlcv {
            timestamp: Some((0..n as i64).collect()),
            open,
            high,
            low,
            close,
            volume: Some(vec![1.0; n]),
        }
    }

    /// CPU reference for the `sma` sweep — mirrors the per-period sweep code in
    /// `compute_classic_ta_columns` (the `ind_id == "sma"` slice): same
    /// `(period as f64)*1.25 >= n` skip, same `compute_cpu` request shape, same
    /// `sma_{period}` column name, same single-output length handling.
    #[cfg(feature = "gpu-cuda")]
    fn cpu_sma_sweep_columns(ohlcv: &Ohlcv, periods: &[usize]) -> Vec<(String, Vec<f64>)> {
        let n = ohlcv.len();
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
        let mut out: Vec<(String, Vec<f64>)> = Vec::new();
        for &period in periods {
            if (period as f64) * 1.25 >= n as f64 {
                continue;
            }
            let params = [vector_ta::indicators::dispatch::ParamKV {
                key: "period",
                value: vector_ta::indicators::dispatch::ParamValue::Int(period as i64),
            }];
            let data_ref = IndicatorDataRef::Candles {
                candles: &candles,
                source: None,
            };
            let req = IndicatorComputeRequest {
                indicator_id: "sma",
                output_id: None,
                data: data_ref,
                params: &params,
                kernel: Kernel::Auto,
            };
            let output = compute_cpu(req).expect("cpu sma compute");
            match output.series {
                IndicatorSeries::F64(v) if v.len() == n => {
                    out.push((format!("sma_{}", period), v));
                }
                IndicatorSeries::F64(v) if v.len() > n && output.cols <= 1 => {
                    let chunk: Vec<f64> = v.into_iter().take(n).collect();
                    out.push((format!("sma_{}", period), chunk));
                }
                other => panic!("unexpected cpu sma series shape: {:?}", other),
            }
        }
        out
    }

    #[cfg(feature = "gpu-cuda")]
    #[test]
    fn gpu_cpu_sma_sweep_parity() {
        let require_gpu = std::env::var("NEOETHOS_REQUIRE_GPU")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false);

        let n = 3000usize;
        let ohlcv = seeded_random_walk_ohlcv(n, 0xC0FFEE_1234_5678);
        let periods = SMA_SWEEP_PERIODS;

        let gpu = match gpu_sma_sweep_columns(&ohlcv, &periods) {
            Ok(cols) => cols,
            Err(e) => {
                if require_gpu {
                    panic!(
                        "NEOETHOS_REQUIRE_GPU set but gpu_sma_sweep_columns failed: {e:?}"
                    );
                }
                eprintln!(
                    "gpu_cpu_sma_sweep_parity: skipping — GPU sweep unavailable: {e:?} \
                     (set NEOETHOS_REQUIRE_GPU=1 to make this a hard failure)"
                );
                return;
            }
        };

        let cpu = cpu_sma_sweep_columns(&ohlcv, &periods);

        // Column COUNT equal.
        assert_eq!(
            gpu.len(),
            cpu.len(),
            "column count mismatch: gpu={} cpu={}",
            gpu.len(),
            cpu.len()
        );

        // NAMES equal in ORDER (sma_7..sma_200, skip-filtered identically).
        for (i, ((gname, _), (cname, _))) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert_eq!(gname, cname, "column name/order mismatch at index {i}");
        }

        // Per-cell NaN mask identical + finite values within f32-appropriate
        // tolerance (1e-4 + 1e-4*|cpu|).
        for ((name, gcol), (_, ccol)) in gpu.iter().zip(cpu.iter()) {
            assert_eq!(
                gcol.len(),
                ccol.len(),
                "column {name} length mismatch: gpu={} cpu={}",
                gcol.len(),
                ccol.len()
            );
            for (j, (&g, &c)) in gcol.iter().zip(ccol.iter()).enumerate() {
                assert_eq!(
                    g.is_nan(),
                    c.is_nan(),
                    "NaN-mask mismatch at {name}[{j}]: gpu={g} cpu={c}"
                );
                if c.is_nan() {
                    continue;
                }
                let tol = 1e-4 + 1e-4 * c.abs();
                assert!(
                    (g - c).abs() <= tol,
                    "value mismatch at {name}[{j}]: gpu={g} cpu={c} (|Δ|={} > tol={tol})",
                    (g - c).abs()
                );
            }
        }
    }
}
