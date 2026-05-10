use anyhow::{Context, Result, bail};
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;
use forex_core::TrainingPrecision;
use half::bf16;
use ndarray::ArrayView2;

use crate::eval::{BacktestSettings, SmcRow};

const SMC_WIDTH: usize = 11;
const BACKTEST_CORE_METRIC_WIDTH: usize = 7;

#[cube(launch)]
fn synthesize_signals_kernel<F: Float>(
    indicators: &Array<F>,
    gene_offsets: &Array<i32>,
    gene_indices: &Array<i32>,
    gene_weights: &Array<F>,
    long_thr: &Array<F>,
    short_thr: &Array<F>,
    smc_data: &Array<i32>,
    gene_smc_flags: &Array<i32>,
    smc_weights: &Array<F>,
    output: &mut Array<i32>,
    n_samples: u32,
    gate_threshold: F,
) {
    if ABSOLUTE_POS < output.len() {
        let pos = ABSOLUTE_POS;
        let gene = pos / n_samples;
        let sample = pos % n_samples;

        let start = gene_offsets[gene] as u32;
        let end = gene_offsets[gene + 1] as u32;
        let mut combined = F::new(0.0);
        let mut i = start;
        while i < end {
            let idx = gene_indices[i] as u32;
            let weight = gene_weights[i];
            let indicator = indicators[idx * n_samples + sample];
            combined += weight * indicator;
            i += 1;
        }

        let lt = long_thr[gene];
        let st = short_thr[gene];
        let mut sig = 0i32;
        if combined >= lt {
            sig = 1;
        } else if combined <= st {
            sig = -1;
        }

        if sig == 0 {
            output[pos] = 0;
            return;
        }

        let flag_base = gene * SMC_WIDTH as u32;
        let smc_base = sample * SMC_WIDTH as u32;
        let mut active_sum = F::new(0.0);
        let mut j = 0u32;
        while j < SMC_WIDTH as u32 {
            if gene_smc_flags[flag_base + j] != 0 {
                active_sum += smc_weights[j];
            }
            j += 1;
        }

        if active_sum <= F::new(0.0) {
            output[pos] = sig;
            return;
        }

        let gate = if active_sum < gate_threshold {
            active_sum
        } else {
            gate_threshold
        };
        let mut score = F::new(0.0);
        let mut k = 0u32;
        while k < SMC_WIDTH as u32 {
            if gene_smc_flags[flag_base + k] != 0 {
                let smc_value = smc_data[smc_base + k];
                if k == 5 {
                    if smc_value == 1 {
                        score += smc_weights[k];
                    }
                } else if smc_value == sig {
                    score += smc_weights[k];
                }
            }
            k += 1;
        }

        if score >= gate {
            output[pos] = sig;
        } else {
            output[pos] = 0;
        }
    }
}

#[cube(launch)]
fn backtest_population_kernel(
    close_pips: &Array<f32>,
    high_pips: &Array<f32>,
    low_pips: &Array<f32>,
    signals_flat: &Array<i32>,
    timestamp_deltas_ms: &Array<i32>,
    month_idx: &Array<i32>,
    day_idx: &Array<i32>,
    sl_pips: &Array<f32>,
    tp_pips: &Array<f32>,
    metrics_out: &mut Array<f32>,
    trade_counts_out: &mut Array<i32>,
    monthly_pnls_out: &mut Array<f32>,
    month_counts_out: &mut Array<i32>,
    n_samples: u32,
    month_capacity: u32,
    initial_equity: f32,
    max_hold_bars: u32,
    min_hold_bars: u32,
    max_trades_per_day: u32,
    gap_threshold_ms: i32,
    use_timestamps: i32,
    trailing_enabled: i32,
    trailing_atr_multiplier: f32,
    trailing_be_trigger_r: f32,
    spread_pips: f32,
    commission_per_trade: f32,
    pip_value_per_lot: f32,
) {
    if ABSOLUTE_POS < trade_counts_out.len() {
        let gene = ABSOLUTE_POS;
        let signal_base = gene * n_samples;
        let month_base = gene * month_capacity;
        let metric_base = gene * BACKTEST_CORE_METRIC_WIDTH as u32;

        let mut zero_idx = 0u32;
        while zero_idx < month_capacity {
            monthly_pnls_out[month_base + zero_idx] = 0.0;
            zero_idx += 1;
        }
        month_counts_out[gene] = 0;
        trade_counts_out[gene] = 0;

        if n_samples == 0 {
            let mut j = 0u32;
            while j < BACKTEST_CORE_METRIC_WIDTH as u32 {
                metrics_out[metric_base + j] = 0.0;
                j += 1;
            }
            return;
        }

        let sl_distance = sl_pips[gene];
        let tp_distance = tp_pips[gene];

        let mut equity = initial_equity;
        let mut peak_equity = initial_equity;
        let mut max_dd = 0.0f32;
        let mut trade_count = 0i32;
        let mut wins = 0i32;
        let mut gross_profit = 0.0f32;
        let mut gross_loss = 0.0f32;

        let mut last_month = -1i32;
        let mut current_month_pnl = 0.0f32;
        let mut month_ptr = -1i32;

        let mut last_day = -1i32;
        let mut day_peak = equity;
        let mut day_low = equity;
        let mut max_daily_dd = 0.0f32;
        let mut day_trade_count = 0u32;

        let mut in_pos = 0i32;
        let mut entry_px = 0.0f32;
        let mut entry_idx = -1i32;
        let mut trail_px = 0.0f32;

        let mut i = 1u32;
        while i < n_samples {
            let m_val = month_idx[i];
            if m_val != last_month {
                if last_month != -1 {
                    month_ptr += 1;
                    if month_ptr >= 0 && month_ptr < month_capacity as i32 {
                        monthly_pnls_out[month_base + month_ptr as u32] = current_month_pnl;
                    }
                }
                current_month_pnl = 0.0;
                last_month = m_val;
            }

            let d_val = day_idx[i];
            if d_val != last_day {
                if last_day != -1 && day_peak > 0.0 {
                    let dd = (day_peak - day_low) / day_peak;
                    if dd > max_daily_dd {
                        max_daily_dd = dd;
                    }
                }
                last_day = d_val;
                day_peak = equity;
                day_low = equity;
                day_trade_count = 0;
            }

            if in_pos != 0
                && use_timestamps != 0
                && gap_threshold_ms > 0
                && timestamp_deltas_ms[i] >= gap_threshold_ms
            {
                let mut pnl = if in_pos == 1 {
                    (close_pips[i] - entry_px) * pip_value_per_lot
                } else {
                    (entry_px - close_pips[i]) * pip_value_per_lot
                };
                pnl -= commission_per_trade + (spread_pips * 0.5 * pip_value_per_lot);
                equity += pnl;
                current_month_pnl += pnl;
                trade_count += 1;
                if pnl > 0.0 {
                    wins += 1;
                    gross_profit += pnl;
                } else {
                    gross_loss += -pnl;
                }
                in_pos = 0;
                if equity > peak_equity {
                    peak_equity = equity;
                }
                if equity < day_low {
                    day_low = equity;
                }
                let current_dd = if peak_equity > 0.0 {
                    (peak_equity - equity) / peak_equity
                } else {
                    0.0
                };
                if current_dd > max_dd {
                    max_dd = current_dd;
                }
            }

            if in_pos != 0 {
                let lo = low_pips[i];
                let hi = high_pips[i];

                let worst_float_pnl = if in_pos == 1 {
                    (lo - entry_px) * pip_value_per_lot
                } else {
                    (entry_px - hi) * pip_value_per_lot
                };
                if (equity + worst_float_pnl) < day_low {
                    day_low = equity + worst_float_pnl;
                }

                let best_float_pnl = if in_pos == 1 {
                    (hi - entry_px) * pip_value_per_lot
                } else {
                    (entry_px - lo) * pip_value_per_lot
                };
                if (equity + best_float_pnl) > peak_equity {
                    peak_equity = equity + best_float_pnl;
                }

                let current_dd = if peak_equity > 0.0 {
                    (peak_equity - (equity + worst_float_pnl)) / peak_equity
                } else {
                    0.0
                };
                if current_dd > max_dd {
                    max_dd = current_dd;
                }

                let mut pnl = 0.0f32;
                let mut exit = false;
                let bars_held = i as i32 - entry_idx;
                let past_min_hold = min_hold_bars == 0 || bars_held >= min_hold_bars as i32;

                if past_min_hold && in_pos == 1 {
                    let mut sl = entry_px - sl_distance;
                    let tp = entry_px + tp_distance;
                    if trailing_enabled != 0 {
                        let mv = hi - entry_px;
                        if mv >= (trailing_be_trigger_r * sl_distance) {
                            let candidate = hi - (trailing_atr_multiplier * sl_distance);
                            if trail_px == 0.0 || candidate > trail_px {
                                trail_px = candidate;
                            }
                            if trail_px > sl {
                                sl = trail_px;
                            }
                        }
                    }
                    if lo <= sl {
                        pnl = (sl - entry_px) * pip_value_per_lot;
                        exit = true;
                    } else if hi >= tp {
                        pnl = (tp - entry_px) * pip_value_per_lot;
                        exit = true;
                    }
                } else if past_min_hold {
                    let mut sl = entry_px + sl_distance;
                    let tp = entry_px - tp_distance;
                    if trailing_enabled != 0 {
                        let mv = entry_px - lo;
                        if mv >= (trailing_be_trigger_r * sl_distance) {
                            let candidate = lo + (trailing_atr_multiplier * sl_distance);
                            if trail_px == 0.0 || candidate < trail_px {
                                trail_px = candidate;
                            }
                            if trail_px < sl {
                                sl = trail_px;
                            }
                        }
                    }
                    if hi >= sl {
                        pnl = (entry_px - sl) * pip_value_per_lot;
                        exit = true;
                    } else if lo <= tp {
                        pnl = (entry_px - tp) * pip_value_per_lot;
                        exit = true;
                    }
                }

                if !exit && past_min_hold && max_hold_bars > 0 && bars_held >= max_hold_bars as i32
                {
                    pnl = if in_pos == 1 {
                        (close_pips[i] - entry_px) * pip_value_per_lot
                    } else {
                        (entry_px - close_pips[i]) * pip_value_per_lot
                    };
                    exit = true;
                }

                if exit {
                    pnl -= commission_per_trade + (spread_pips * 0.5 * pip_value_per_lot);
                    equity += pnl;
                    current_month_pnl += pnl;
                    trade_count += 1;
                    if pnl > 0.0 {
                        wins += 1;
                        gross_profit += pnl;
                    } else {
                        gross_loss += -pnl;
                    }
                    in_pos = 0;
                    if equity > peak_equity {
                        peak_equity = equity;
                    }
                    if equity < day_low {
                        day_low = equity;
                    }

                    let current_dd = if peak_equity > 0.0 {
                        (peak_equity - equity) / peak_equity
                    } else {
                        0.0
                    };
                    if current_dd > max_dd {
                        max_dd = current_dd;
                    }
                }
            } else {
                // Causal entry: read PRIOR-bar signal, fill at CURRENT-bar
                // close. Mirrors `eval.rs::simulate_trades_core` exactly so
                // CUDA backtest is semantically equivalent to CPU canonical.
                // Reading `signals_flat[signal_base + i]` (current bar) would
                // re-introduce intra-bar look-ahead.
                let s = signals_flat[signal_base + i - 1];
                if s != 0 {
                    if !(max_trades_per_day > 0 && day_trade_count >= max_trades_per_day) {
                        in_pos = s;
                        entry_px = close_pips[i] + (s as f32) * spread_pips * 0.5;
                        entry_idx = i as i32;
                        trail_px = 0.0;
                        day_trade_count += 1;
                    }
                }
            }

            i += 1;
        }

        let net_profit = equity - initial_equity;
        let win_rate = if trade_count > 0 {
            wins as f32 / trade_count as f32
        } else {
            0.0
        };
        let pf = if gross_loss > 0.0 {
            (gross_profit / gross_loss).min(10.0)
        } else if gross_profit > 0.0 {
            10.0
        } else {
            0.0
        };
        let expectancy = if trade_count > 0 {
            net_profit / trade_count as f32
        } else {
            0.0
        };
        let filled_months = if month_ptr >= 0 {
            let raw = month_ptr + 1;
            if raw < month_capacity as i32 {
                raw
            } else {
                month_capacity as i32
            }
        } else {
            0
        };

        metrics_out[metric_base] = net_profit;
        metrics_out[metric_base + 1] = peak_equity;
        metrics_out[metric_base + 2] = max_dd;
        metrics_out[metric_base + 3] = win_rate;
        metrics_out[metric_base + 4] = pf;
        metrics_out[metric_base + 5] = expectancy;
        metrics_out[metric_base + 6] = max_daily_dd;
        trade_counts_out[gene] = trade_count;
        month_counts_out[gene] = filled_months;
    }
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    // Phase 64 — both CPU and GPU paths now share the canonical
    // `forex_core::utils::mean_std` so CPU/GPU rank parity cannot drift
    // due to a math-helper divergence.
    let (mean, std) = forex_core::utils::mean_std(values);
    if !mean.is_finite() || !std.is_finite() {
        return (0.0, 0.0);
    }
    (mean, std)
}

fn parse_training_precision(value: &str) -> Option<TrainingPrecision> {
    match value.trim().to_ascii_lowercase().as_str() {
        "fp32" | "f32" | "float32" => Some(TrainingPrecision::Fp32),
        "fp16" | "f16" | "float16" | "half" => Some(TrainingPrecision::Fp16),
        "bf16" | "bfloat16" => Some(TrainingPrecision::Bf16),
        "fp8" | "float8" => Some(TrainingPrecision::Fp8),
        "bf4" => Some(TrainingPrecision::Bf4),
        _ => None,
    }
}

fn requested_eval_precision() -> TrainingPrecision {
    [
        "FOREX_BOT_SEARCH_EVAL_PRECISION",
        "FOREX_BOT_TRAIN_PRECISION",
        "FOREX_TRAIN_PRECISION",
    ]
    .iter()
    .find_map(|key| std::env::var(key).ok())
    .and_then(|value| parse_training_precision(&value))
    .unwrap_or(TrainingPrecision::Fp32)
}

fn prefers_bf16(requested: TrainingPrecision) -> bool {
    matches!(
        requested,
        TrainingPrecision::Bf16 | TrainingPrecision::Fp8 | TrainingPrecision::Bf4
    )
}

pub(crate) fn cuda_eval_signal_kernel_enabled() -> bool {
    !matches!(
        std::env::var("FOREX_BOT_SEARCH_EVAL_CUDA_KERNEL")
            .ok()
            .map(|value| value.trim().to_ascii_lowercase()),
        Some(value) if matches!(value.as_str(), "0" | "false" | "off" | "disable" | "disabled")
    )
}

pub(crate) fn cuda_eval_backtest_kernel_enabled() -> bool {
    !matches!(
        std::env::var("FOREX_BOT_SEARCH_BACKTEST_CUDA_KERNEL")
            .ok()
            .map(|value| value.trim().to_ascii_lowercase()),
        Some(value) if matches!(value.as_str(), "0" | "false" | "off" | "disable" | "disabled")
    )
}

fn signal_kernel_units(client: &ComputeClient<CudaRuntime>) -> u32 {
    let max_units = client.properties().hardware.max_units_per_cube.max(1);
    std::env::var("FOREX_BOT_SEARCH_EVAL_KERNEL_UNITS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(max_units)
        .min(max_units)
        .max(1)
}

fn backtest_kernel_units(client: &ComputeClient<CudaRuntime>) -> u32 {
    let max_units = client.properties().hardware.max_units_per_cube.max(1);
    std::env::var("FOREX_BOT_SEARCH_BACKTEST_KERNEL_UNITS")
        .ok()
        .or_else(|| std::env::var("FOREX_BOT_SEARCH_EVAL_KERNEL_UNITS").ok())
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(max_units)
        .min(max_units)
        .max(1)
}

fn cuda_device_id() -> usize {
    std::env::var("FOREX_BOT_SEARCH_EVAL_CUDA_DEVICE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
}

fn flatten_i32_rows(rows: &[SmcRow]) -> Vec<i32> {
    let mut out = Vec::with_capacity(rows.len().saturating_mul(SMC_WIDTH));
    for row in rows {
        for value in row {
            out.push(*value as i32);
        }
    }
    out
}

fn flatten_i32_flags(rows: &[SmcRow]) -> Vec<i32> {
    flatten_i32_rows(rows)
}

fn launch_signal_kernel<F>(
    client: &ComputeClient<CudaRuntime>,
    indicators_flat: &[F],
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[F],
    long_thr: &[F],
    short_thr: &[F],
    smc_data: &[i32],
    gene_smc_flags: &[i32],
    smc_weights: &[F],
    n_genes: usize,
    n_samples: usize,
    gate_threshold: F,
) -> Result<Vec<i32>>
where
    F: Float + CubeElement,
{
    let total = n_genes.saturating_mul(n_samples);
    if total == 0 {
        return Ok(Vec::new());
    }

    let indicators_handle = client.create_from_slice(F::as_bytes(indicators_flat));
    let gene_offsets_handle = client.create_from_slice(i32::as_bytes(gene_offsets));
    let gene_indices_handle = client.create_from_slice(i32::as_bytes(gene_indices));
    let gene_weights_handle = client.create_from_slice(F::as_bytes(gene_weights));
    let long_thr_handle = client.create_from_slice(F::as_bytes(long_thr));
    let short_thr_handle = client.create_from_slice(F::as_bytes(short_thr));
    let smc_data_handle = client.create_from_slice(i32::as_bytes(smc_data));
    let gene_smc_flags_handle = client.create_from_slice(i32::as_bytes(gene_smc_flags));
    let smc_weights_handle = client.create_from_slice(F::as_bytes(smc_weights));
    let output_handle = client.empty(total.saturating_mul(std::mem::size_of::<i32>()));

    let units = signal_kernel_units(client);
    let cubes = (total as u32).div_ceil(units);
    synthesize_signals_kernel::launch::<F, CudaRuntime>(
        client,
        CubeCount::Static(cubes, 1, 1),
        CubeDim::new_1d(units),
        unsafe { ArrayArg::from_raw_parts::<F>(&indicators_handle, indicators_flat.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<i32>(&gene_offsets_handle, gene_offsets.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<i32>(&gene_indices_handle, gene_indices.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<F>(&gene_weights_handle, gene_weights.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<F>(&long_thr_handle, long_thr.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<F>(&short_thr_handle, short_thr.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<i32>(&smc_data_handle, smc_data.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<i32>(&gene_smc_flags_handle, gene_smc_flags.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<F>(&smc_weights_handle, smc_weights.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<i32>(&output_handle, total, 1) },
        ScalarArg::new(n_samples as u32),
        ScalarArg::new(gate_threshold),
    )
    .context("launch cuda evaluator signal kernel")?;

    let bytes = client.read_one(output_handle);
    Ok(i32::from_bytes(&bytes).to_vec())
}

fn materialize_i8_rows(flat: &[i32], n_genes: usize, n_samples: usize) -> Vec<Vec<i8>> {
    flat.chunks(n_samples)
        .take(n_genes)
        .map(|row| {
            row.iter()
                .map(|value| value.clamp(-1, 1) as i8)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn try_generate_signal_flat_cuda(
    indicators: ArrayView2<'_, f32>,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f32],
    long_thr: &[f32],
    short_thr: &[f32],
    smc_data: &[SmcRow],
    gene_smc_flags: &[SmcRow],
    gate_threshold: f32,
    smc_weights: &[f32; SMC_WIDTH],
) -> Result<Vec<i32>> {
    let n_genes = long_thr.len();
    let n_samples = indicators.ncols();
    if n_genes == 0 || n_samples == 0 {
        return Ok(Vec::new());
    }
    if gene_offsets.len() != n_genes + 1 {
        bail!(
            "cuda evaluator signal kernel gene_offsets mismatch: expected {}, received {}",
            n_genes + 1,
            gene_offsets.len()
        );
    }
    if short_thr.len() != n_genes
        || gene_smc_flags.len() != n_genes
        || smc_data.len() != n_samples
        || indicators.nrows() == 0
    {
        bail!("cuda evaluator signal kernel received inconsistent dimensions");
    }

    let device_id = cuda_device_id();
    let device_count = tch::Cuda::device_count();
    if device_count <= device_id as i64 {
        bail!(
            "cuda evaluator signal kernel requested device {} but only {} CUDA devices are available",
            device_id,
            device_count
        );
    }

    let indicators_flat = indicators.iter().copied().collect::<Vec<_>>();
    let smc_data_flat = flatten_i32_rows(smc_data);
    let gene_smc_flags_flat = flatten_i32_flags(gene_smc_flags);
    let precision = requested_eval_precision();

    let device = CudaDevice::new(device_id);
    let client = CudaRuntime::client(&device);

    if prefers_bf16(precision) {
        let indicators_bf16 = indicators_flat
            .iter()
            .map(|value| bf16::from_f32(*value))
            .collect::<Vec<_>>();
        let gene_weights_bf16 = gene_weights
            .iter()
            .map(|value| bf16::from_f32(*value))
            .collect::<Vec<_>>();
        let long_thr_bf16 = long_thr
            .iter()
            .map(|value| bf16::from_f32(*value))
            .collect::<Vec<_>>();
        let short_thr_bf16 = short_thr
            .iter()
            .map(|value| bf16::from_f32(*value))
            .collect::<Vec<_>>();
        let smc_weights_bf16 = smc_weights
            .iter()
            .map(|value| bf16::from_f32(*value))
            .collect::<Vec<_>>();

        match launch_signal_kernel::<bf16>(
            &client,
            &indicators_bf16,
            gene_offsets,
            gene_indices,
            &gene_weights_bf16,
            &long_thr_bf16,
            &short_thr_bf16,
            &smc_data_flat,
            &gene_smc_flags_flat,
            &smc_weights_bf16,
            n_genes,
            n_samples,
            bf16::from_f32(gate_threshold),
        ) {
            Ok(flat) => return Ok(flat),
            Err(err) => {
                tracing::debug!(
                    "cuda evaluator bf16 signal kernel unavailable, falling back to fp32: {err}"
                );
            }
        }
    }

    launch_signal_kernel::<f32>(
        &client,
        &indicators_flat,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        &smc_data_flat,
        &gene_smc_flags_flat,
        smc_weights,
        n_genes,
        n_samples,
        gate_threshold,
    )
    .context("launch fp32 cuda evaluator signal kernel")
}

pub(crate) fn try_generate_signal_rows_cuda(
    indicators: ArrayView2<'_, f32>,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f32],
    long_thr: &[f32],
    short_thr: &[f32],
    smc_data: &[SmcRow],
    gene_smc_flags: &[SmcRow],
    gate_threshold: f32,
    smc_weights: &[f32; SMC_WIDTH],
) -> Result<Vec<Vec<i8>>> {
    let n_genes = long_thr.len();
    let n_samples = indicators.ncols();
    let flat = try_generate_signal_flat_cuda(
        indicators,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        smc_data,
        gene_smc_flags,
        gate_threshold,
        smc_weights,
    )?;
    Ok(materialize_i8_rows(&flat, n_genes, n_samples))
}

fn saturating_i32(value: i64) -> i32 {
    value.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

fn timestamp_delta_ms(timestamps: &[i64], n_samples: usize) -> (Vec<i32>, bool) {
    let mut deltas = vec![0i32; n_samples];
    if timestamps.len() != n_samples {
        return (deltas, false);
    }
    for i in 1..n_samples {
        let delta = timestamps[i].saturating_sub(timestamps[i - 1]).max(0);
        deltas[i] = saturating_i32(delta);
    }
    (deltas, true)
}

fn normalize_prices_to_pips(prices: &[f64], pip_value: f64) -> Vec<f32> {
    let safe_pip = if pip_value.abs() < 1e-12 {
        1e-12
    } else {
        pip_value
    };
    prices
        .iter()
        .map(|price| (*price / safe_pip) as f32)
        .collect()
}

fn launch_backtest_kernel(
    client: &ComputeClient<CudaRuntime>,
    close_pips: &[f32],
    high_pips: &[f32],
    low_pips: &[f32],
    signals_flat: &[i32],
    timestamp_deltas_ms: &[i32],
    use_timestamps: bool,
    month_idx: &[i32],
    day_idx: &[i32],
    sl_pips: &[f32],
    tp_pips: &[f32],
    settings: &BacktestSettings,
    month_capacity: usize,
) -> Result<(Vec<f32>, Vec<i32>, Vec<f32>, Vec<i32>)> {
    let n_samples = close_pips.len();
    let n_genes = sl_pips.len();
    if n_samples == 0 || n_genes == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
    }
    if high_pips.len() != n_samples
        || low_pips.len() != n_samples
        || timestamp_deltas_ms.len() != n_samples
        || month_idx.len() != n_samples
        || day_idx.len() != n_samples
        || tp_pips.len() != n_genes
        || signals_flat.len() != n_genes.saturating_mul(n_samples)
    {
        bail!("cuda evaluator backtest kernel received inconsistent dimensions");
    }

    let close_handle = client.create_from_slice(f32::as_bytes(close_pips));
    let high_handle = client.create_from_slice(f32::as_bytes(high_pips));
    let low_handle = client.create_from_slice(f32::as_bytes(low_pips));
    let signals_handle = client.create_from_slice(i32::as_bytes(signals_flat));
    let timestamp_delta_handle = client.create_from_slice(i32::as_bytes(timestamp_deltas_ms));
    let month_handle = client.create_from_slice(i32::as_bytes(month_idx));
    let day_handle = client.create_from_slice(i32::as_bytes(day_idx));
    let sl_handle = client.create_from_slice(f32::as_bytes(sl_pips));
    let tp_handle = client.create_from_slice(f32::as_bytes(tp_pips));

    let metrics_len = n_genes.saturating_mul(BACKTEST_CORE_METRIC_WIDTH);
    let monthly_len = n_genes.saturating_mul(month_capacity);
    let metrics_handle = client.empty(metrics_len.saturating_mul(std::mem::size_of::<f32>()));
    let trade_counts_handle = client.empty(n_genes.saturating_mul(std::mem::size_of::<i32>()));
    let monthly_handle = client.empty(monthly_len.saturating_mul(std::mem::size_of::<f32>()));
    let month_counts_handle = client.empty(n_genes.saturating_mul(std::mem::size_of::<i32>()));

    let units = backtest_kernel_units(client);
    let cubes = (n_genes as u32).div_ceil(units);
    backtest_population_kernel::launch::<CudaRuntime>(
        client,
        CubeCount::Static(cubes, 1, 1),
        CubeDim::new_1d(units),
        unsafe { ArrayArg::from_raw_parts::<f32>(&close_handle, n_samples, 1) },
        unsafe { ArrayArg::from_raw_parts::<f32>(&high_handle, n_samples, 1) },
        unsafe { ArrayArg::from_raw_parts::<f32>(&low_handle, n_samples, 1) },
        unsafe { ArrayArg::from_raw_parts::<i32>(&signals_handle, signals_flat.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<i32>(&timestamp_delta_handle, n_samples, 1) },
        unsafe { ArrayArg::from_raw_parts::<i32>(&month_handle, month_idx.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<i32>(&day_handle, day_idx.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<f32>(&sl_handle, sl_pips.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<f32>(&tp_handle, tp_pips.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<f32>(&metrics_handle, metrics_len, 1) },
        unsafe { ArrayArg::from_raw_parts::<i32>(&trade_counts_handle, n_genes, 1) },
        unsafe { ArrayArg::from_raw_parts::<f32>(&monthly_handle, monthly_len, 1) },
        unsafe { ArrayArg::from_raw_parts::<i32>(&month_counts_handle, n_genes, 1) },
        ScalarArg::new(n_samples as u32),
        ScalarArg::new(month_capacity as u32),
        ScalarArg::new(settings.initial_equity() as f32),
        ScalarArg::new(settings.max_hold_bars as u32),
        ScalarArg::new(settings.min_hold_bars as u32),
        ScalarArg::new(settings.max_trades_per_day as u32),
        ScalarArg::new(saturating_i32(settings.gap_threshold_ms)),
        ScalarArg::new(if use_timestamps { 1i32 } else { 0i32 }),
        ScalarArg::new(if settings.trailing_enabled {
            1i32
        } else {
            0i32
        }),
        ScalarArg::new(settings.trailing_atr_multiplier as f32),
        ScalarArg::new(settings.trailing_be_trigger_r as f32),
        ScalarArg::new(settings.spread_pips as f32),
        ScalarArg::new(settings.commission_per_trade as f32),
        ScalarArg::new(settings.pip_value_per_lot as f32),
    )
    .context("launch cuda evaluator backtest kernel")?;

    let metrics_bytes = client.read_one(metrics_handle);
    let trade_counts_bytes = client.read_one(trade_counts_handle);
    let monthly_bytes = client.read_one(monthly_handle);
    let month_counts_bytes = client.read_one(month_counts_handle);

    Ok((
        f32::from_bytes(&metrics_bytes).to_vec(),
        i32::from_bytes(&trade_counts_bytes).to_vec(),
        f32::from_bytes(&monthly_bytes).to_vec(),
        i32::from_bytes(&month_counts_bytes).to_vec(),
    ))
}

pub(crate) fn try_evaluate_population_cuda(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    indicators: ArrayView2<'_, f32>,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f32],
    long_thr: &[f32],
    short_thr: &[f32],
    month_idx: &[i64],
    day_idx: &[i64],
    timestamps: &[i64],
    sl_pips: &[f64],
    tp_pips: &[f64],
    smc_data: &[SmcRow],
    gene_smc_flags: &[SmcRow],
    gate_threshold: f32,
    smc_weights: &[f32; SMC_WIDTH],
    settings: &BacktestSettings,
) -> Result<Vec<[f64; 11]>> {
    let n_genes = long_thr.len();
    let n_samples = close.len();
    if n_genes == 0 || n_samples == 0 {
        return Ok(vec![ZERO_METRICS; n_genes]);
    }
    if high.len() != n_samples
        || low.len() != n_samples
        || month_idx.len() != n_samples
        || day_idx.len() != n_samples
        || indicators.ncols() != n_samples
        || sl_pips.len() != n_genes
        || tp_pips.len() != n_genes
    {
        bail!("cuda population evaluate path received inconsistent dimensions");
    }

    let signals_flat = try_generate_signal_flat_cuda(
        indicators,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        smc_data,
        gene_smc_flags,
        gate_threshold,
        smc_weights,
    )?;

    let device = CudaDevice::new(cuda_device_id());
    let client = CudaRuntime::client(&device);
    let close_pips = normalize_prices_to_pips(close, settings.pip_value);
    let high_pips = normalize_prices_to_pips(high, settings.pip_value);
    let low_pips = normalize_prices_to_pips(low, settings.pip_value);
    let (timestamp_deltas_ms, use_timestamps) = timestamp_delta_ms(timestamps, n_samples);
    let month_idx = month_idx
        .iter()
        .map(|value| saturating_i32(*value))
        .collect::<Vec<_>>();
    let day_idx = day_idx
        .iter()
        .map(|value| saturating_i32(*value))
        .collect::<Vec<_>>();
    let sl_pips = sl_pips
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let tp_pips = tp_pips
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let month_capacity = settings.month_capacity();

    let (metrics_flat, trade_counts, monthly_flat, month_counts) = launch_backtest_kernel(
        &client,
        &close_pips,
        &high_pips,
        &low_pips,
        &signals_flat,
        &timestamp_deltas_ms,
        use_timestamps,
        &month_idx,
        &day_idx,
        &sl_pips,
        &tp_pips,
        settings,
        month_capacity,
    )?;

    let mut results = Vec::with_capacity(n_genes);
    for g in 0..n_genes {
        let metric_base = g * BACKTEST_CORE_METRIC_WIDTH;
        let month_base = g.saturating_mul(month_capacity);
        let month_count = month_counts.get(g).copied().unwrap_or_default().max(0) as usize;
        let month_limit = month_count.min(month_capacity);
        let month_returns = monthly_flat[month_base..month_base + month_limit]
            .iter()
            .map(|value| *value as f64)
            .collect::<Vec<_>>();
        let (avg_m, std_m) = mean_std(&month_returns);
        let sharpe = if std_m > 0.0 {
            (avg_m / std_m) * 3.4641
        } else {
            0.0
        };
        let consistency = if std_m > 0.0 {
            (avg_m / std_m).clamp(0.0, 1.0)
        } else if avg_m > 0.0 && month_returns.len() < 2 {
            1.0
        } else {
            0.0
        };

        results.push([
            metrics_flat[metric_base] as f64,
            sharpe,
            metrics_flat[metric_base + 1] as f64,
            metrics_flat[metric_base + 2] as f64,
            metrics_flat[metric_base + 3] as f64,
            metrics_flat[metric_base + 4] as f64,
            metrics_flat[metric_base + 5] as f64,
            0.0,
            trade_counts.get(g).copied().unwrap_or_default() as f64,
            consistency,
            metrics_flat[metric_base + 6] as f64,
        ]);
    }

    Ok(results)
}

const ZERO_METRICS: [f64; 11] = [0.0; 11];
