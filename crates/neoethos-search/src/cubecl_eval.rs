use anyhow::{Context, Result, bail};
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;
use neoethos_core::TrainingPrecision;
use half::bf16;
use ndarray::ArrayView2;

use crate::eval::{BacktestSettings, SmcRow};

const SMC_WIDTH: usize = 11;
const BACKTEST_CORE_METRIC_WIDTH: usize = 7;

#[cube(launch)]
fn synthesize_signals_kernel<F: Float + CubeElement>(
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
    // cubecl 0.9: ABSOLUTE_POS and Array::len() are `usize`, and array
    // indexing also expects `usize`. Coerce all u32 kernel parameters
    // to usize at the top of the kernel so the rest reads naturally.
    //
    // For mutable scalar accumulators (`combined`, `active_sum`,
    // `score`, `sig`) we must use RuntimeCell because cubecl 0.9's
    // `assign` and `assign_op` paths both reject const-initialized
    // `let mut` bindings.
    let pos = ABSOLUTE_POS;
    if pos < output.len() {
        let n_samples = n_samples as usize;
        let gene = pos / n_samples;
        let sample = pos % n_samples;

        let start = gene_offsets[gene] as usize;
        let end = gene_offsets[gene + 1] as usize;
        let combined = RuntimeCell::<F>::new(F::new(0.0));
        for i in start..end {
            let idx = gene_indices[i] as usize;
            let weight = gene_weights[i];
            let indicator = indicators[idx * n_samples + sample];
            combined.store(combined.read() + weight * indicator);
        }

        let lt = long_thr[gene];
        let st = short_thr[gene];
        let combined_val = combined.read();
        let sig = RuntimeCell::<i32>::new(0);
        if combined_val >= lt {
            sig.store(1);
        } else if combined_val <= st {
            sig.store(-1);
        }

        let sig_val = sig.read();
        if sig_val == 0 {
            output[pos] = 0;
            terminate!();
        }

        let flag_base = gene * SMC_WIDTH;
        let smc_base = sample * SMC_WIDTH;
        let active_sum = RuntimeCell::<F>::new(F::new(0.0));
        for j in 0..SMC_WIDTH {
            if gene_smc_flags[flag_base + j] != 0 {
                active_sum.store(active_sum.read() + smc_weights[j]);
            }
        }

        let active_sum_val = active_sum.read();
        if active_sum_val <= F::new(0.0) {
            output[pos] = sig_val;
            terminate!();
        }

        let gate = if active_sum_val < gate_threshold {
            active_sum_val
        } else {
            gate_threshold
        };
        let score = RuntimeCell::<F>::new(F::new(0.0));
        for k in 0..SMC_WIDTH {
            if gene_smc_flags[flag_base + k] != 0 {
                let smc_value = smc_data[smc_base + k];
                if k == 5 {
                    if smc_value == 1 {
                        score.store(score.read() + smc_weights[k]);
                    }
                } else if smc_value == sig_val {
                    score.store(score.read() + smc_weights[k]);
                }
            }
        }

        if score.read() >= gate {
            output[pos] = sig_val;
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
    // cubecl 0.9: index arithmetic is usize; coerce u32 params at the top.
    // Every scalar accumulator that gets reassigned must use RuntimeCell —
    // `let mut x = literal;` and `let mut x = param;` both produce
    // immutable bindings in cubecl 0.9, and any later `=`/`+=` panics.
    if ABSOLUTE_POS < trade_counts_out.len() {
        let gene = ABSOLUTE_POS;
        let n_samples = n_samples as usize;
        let month_capacity = month_capacity as usize;
        let max_hold_bars = max_hold_bars as usize;
        let min_hold_bars = min_hold_bars as usize;
        let max_trades_per_day = max_trades_per_day as usize;
        let signal_base = gene * n_samples;
        let month_base = gene * month_capacity;
        let metric_base = gene * BACKTEST_CORE_METRIC_WIDTH;

        for zero_idx in 0..month_capacity {
            monthly_pnls_out[month_base + zero_idx] = 0.0;
        }
        month_counts_out[gene] = 0;
        trade_counts_out[gene] = 0;

        if n_samples == 0 {
            for j in 0..BACKTEST_CORE_METRIC_WIDTH {
                metrics_out[metric_base + j] = 0.0;
            }
            terminate!();
        }

        let sl_distance = sl_pips[gene];
        let tp_distance = tp_pips[gene];

        let equity = RuntimeCell::<f32>::new(initial_equity);
        let peak_equity = RuntimeCell::<f32>::new(initial_equity);
        let max_dd = RuntimeCell::<f32>::new(0.0);
        let trade_count = RuntimeCell::<i32>::new(0);
        let wins = RuntimeCell::<i32>::new(0);
        let gross_profit = RuntimeCell::<f32>::new(0.0);
        let gross_loss = RuntimeCell::<f32>::new(0.0);

        let last_month = RuntimeCell::<i32>::new(-1);
        let current_month_pnl = RuntimeCell::<f32>::new(0.0);
        let month_ptr = RuntimeCell::<i32>::new(-1);

        let last_day = RuntimeCell::<i32>::new(-1);
        let day_peak = RuntimeCell::<f32>::new(initial_equity);
        let day_low = RuntimeCell::<f32>::new(initial_equity);
        let max_daily_dd = RuntimeCell::<f32>::new(0.0);
        let day_trade_count = RuntimeCell::<u32>::new(0);

        let in_pos = RuntimeCell::<i32>::new(0);
        let entry_px = RuntimeCell::<f32>::new(0.0);
        let entry_idx = RuntimeCell::<i32>::new(-1);
        let trail_px = RuntimeCell::<f32>::new(0.0);

        for i in 1..n_samples {
            let m_val = month_idx[i];
            let last_month_v = last_month.read();
            if m_val != last_month_v {
                if last_month_v != -1 {
                    let next_ptr = month_ptr.read() + 1;
                    month_ptr.store(next_ptr);
                    if next_ptr >= 0 && next_ptr < month_capacity as i32 {
                        monthly_pnls_out[month_base + next_ptr as usize] = current_month_pnl.read();
                    }
                }
                current_month_pnl.store(0.0);
                last_month.store(m_val);
            }

            let d_val = day_idx[i];
            let last_day_v = last_day.read();
            if d_val != last_day_v {
                if last_day_v != -1 && day_peak.read() > 0.0 {
                    let dd = (day_peak.read() - day_low.read()) / day_peak.read();
                    if dd > max_daily_dd.read() {
                        max_daily_dd.store(dd);
                    }
                }
                last_day.store(d_val);
                day_peak.store(equity.read());
                day_low.store(equity.read());
                day_trade_count.store(0);
            }

            let in_pos_v = in_pos.read();
            if in_pos_v != 0
                && use_timestamps != 0
                && gap_threshold_ms > 0
                && timestamp_deltas_ms[i] >= gap_threshold_ms
            {
                let entry_px_v = entry_px.read();
                let pnl_cell = RuntimeCell::<f32>::new(0.0);
                if in_pos_v == 1 {
                    pnl_cell.store((close_pips[i] - entry_px_v) * pip_value_per_lot);
                } else {
                    pnl_cell.store((entry_px_v - close_pips[i]) * pip_value_per_lot);
                }
                pnl_cell.store(
                    pnl_cell.read()
                        - commission_per_trade
                        - (spread_pips * 0.5 * pip_value_per_lot),
                );
                let pnl = pnl_cell.read();
                equity.store(equity.read() + pnl);
                current_month_pnl.store(current_month_pnl.read() + pnl);
                trade_count.store(trade_count.read() + 1);
                if pnl > 0.0 {
                    wins.store(wins.read() + 1);
                    gross_profit.store(gross_profit.read() + pnl);
                } else {
                    gross_loss.store(gross_loss.read() - pnl);
                }
                in_pos.store(0);
                let eq = equity.read();
                if eq > peak_equity.read() {
                    peak_equity.store(eq);
                }
                if eq < day_low.read() {
                    day_low.store(eq);
                }
                let pe = peak_equity.read();
                let current_dd = RuntimeCell::<f32>::new(0.0);
                if pe > 0.0 {
                    current_dd.store((pe - eq) / pe);
                }
                if current_dd.read() > max_dd.read() {
                    max_dd.store(current_dd.read());
                }
            }

            let in_pos_v2 = in_pos.read();
            if in_pos_v2 != 0 {
                let lo = low_pips[i];
                let hi = high_pips[i];
                let entry_px_v = entry_px.read();

                let worst_float_pnl = if in_pos_v2 == 1 {
                    (lo - entry_px_v) * pip_value_per_lot
                } else {
                    (entry_px_v - hi) * pip_value_per_lot
                };
                let eq = equity.read();
                if (eq + worst_float_pnl) < day_low.read() {
                    day_low.store(eq + worst_float_pnl);
                }

                let best_float_pnl = if in_pos_v2 == 1 {
                    (hi - entry_px_v) * pip_value_per_lot
                } else {
                    (entry_px_v - lo) * pip_value_per_lot
                };
                if (eq + best_float_pnl) > peak_equity.read() {
                    peak_equity.store(eq + best_float_pnl);
                }

                let pe = peak_equity.read();
                let current_dd = RuntimeCell::<f32>::new(0.0);
                if pe > 0.0 {
                    current_dd.store((pe - (eq + worst_float_pnl)) / pe);
                }
                if current_dd.read() > max_dd.read() {
                    max_dd.store(current_dd.read());
                }

                let pnl_cell = RuntimeCell::<f32>::new(0.0);
                let exit_cell = RuntimeCell::<u32>::new(0);
                let bars_held = i as i32 - entry_idx.read();
                let past_min_hold = min_hold_bars == 0 || bars_held >= min_hold_bars as i32;

                if past_min_hold && in_pos_v2 == 1 {
                    let sl_cell = RuntimeCell::<f32>::new(entry_px_v - sl_distance);
                    let tp = entry_px_v + tp_distance;
                    if trailing_enabled != 0 {
                        let mv = hi - entry_px_v;
                        if mv >= (trailing_be_trigger_r * sl_distance) {
                            let candidate = hi - (trailing_atr_multiplier * sl_distance);
                            if trail_px.read() == 0.0 || candidate > trail_px.read() {
                                trail_px.store(candidate);
                            }
                            if trail_px.read() > sl_cell.read() {
                                sl_cell.store(trail_px.read());
                            }
                        }
                    }
                    let sl_v = sl_cell.read();
                    if lo <= sl_v {
                        pnl_cell.store((sl_v - entry_px_v) * pip_value_per_lot);
                        exit_cell.store(1);
                    } else if hi >= tp {
                        pnl_cell.store((tp - entry_px_v) * pip_value_per_lot);
                        exit_cell.store(1);
                    }
                } else if past_min_hold {
                    let sl_cell = RuntimeCell::<f32>::new(entry_px_v + sl_distance);
                    let tp = entry_px_v - tp_distance;
                    if trailing_enabled != 0 {
                        let mv = entry_px_v - lo;
                        if mv >= (trailing_be_trigger_r * sl_distance) {
                            let candidate = lo + (trailing_atr_multiplier * sl_distance);
                            if trail_px.read() == 0.0 || candidate < trail_px.read() {
                                trail_px.store(candidate);
                            }
                            if trail_px.read() < sl_cell.read() {
                                sl_cell.store(trail_px.read());
                            }
                        }
                    }
                    let sl_v = sl_cell.read();
                    if hi >= sl_v {
                        pnl_cell.store((entry_px_v - sl_v) * pip_value_per_lot);
                        exit_cell.store(1);
                    } else if lo <= tp {
                        pnl_cell.store((entry_px_v - tp) * pip_value_per_lot);
                        exit_cell.store(1);
                    }
                }

                if exit_cell.read() == 0
                    && past_min_hold
                    && max_hold_bars > 0
                    && bars_held >= max_hold_bars as i32
                {
                    if in_pos_v2 == 1 {
                        pnl_cell.store((close_pips[i] - entry_px_v) * pip_value_per_lot);
                    } else {
                        pnl_cell.store((entry_px_v - close_pips[i]) * pip_value_per_lot);
                    }
                    exit_cell.store(1);
                }

                if exit_cell.read() != 0 {
                    pnl_cell.store(
                        pnl_cell.read()
                            - commission_per_trade
                            - (spread_pips * 0.5 * pip_value_per_lot),
                    );
                    let pnl = pnl_cell.read();
                    equity.store(equity.read() + pnl);
                    current_month_pnl.store(current_month_pnl.read() + pnl);
                    trade_count.store(trade_count.read() + 1);
                    if pnl > 0.0 {
                        wins.store(wins.read() + 1);
                        gross_profit.store(gross_profit.read() + pnl);
                    } else {
                        gross_loss.store(gross_loss.read() - pnl);
                    }
                    in_pos.store(0);
                    let eq2 = equity.read();
                    if eq2 > peak_equity.read() {
                        peak_equity.store(eq2);
                    }
                    if eq2 < day_low.read() {
                        day_low.store(eq2);
                    }
                    let pe2 = peak_equity.read();
                    let current_dd = RuntimeCell::<f32>::new(0.0);
                    if pe2 > 0.0 {
                        current_dd.store((pe2 - eq2) / pe2);
                    }
                    if current_dd.read() > max_dd.read() {
                        max_dd.store(current_dd.read());
                    }
                }
            } else {
                // Causal entry: read PRIOR-bar signal, fill at CURRENT-bar close.
                let s = signals_flat[signal_base + i - 1];
                if s != 0 {
                    if !(max_trades_per_day > 0
                        && (day_trade_count.read() as usize) >= max_trades_per_day)
                    {
                        in_pos.store(s);
                        entry_px.store(close_pips[i] + (s as f32) * spread_pips * 0.5);
                        entry_idx.store(i as i32);
                        trail_px.store(0.0);
                        day_trade_count.store(day_trade_count.read() + 1);
                    }
                }
            }
        }

        let final_equity = equity.read();
        let final_peak = peak_equity.read();
        let final_max_dd = max_dd.read();
        let final_trade_count = trade_count.read();
        let final_wins = wins.read();
        let final_gp = gross_profit.read();
        let final_gl = gross_loss.read();
        let final_max_daily_dd = max_daily_dd.read();
        let final_month_ptr = month_ptr.read();

        let net_profit = final_equity - initial_equity;
        let win_rate_cell = RuntimeCell::<f32>::new(0.0);
        if final_trade_count > 0 {
            win_rate_cell.store(final_wins as f32 / final_trade_count as f32);
        }
        let pf_cell = RuntimeCell::<f32>::new(0.0);
        if final_gl > 0.0 {
            pf_cell.store((final_gp / final_gl).min(10.0));
        } else if final_gp > 0.0 {
            pf_cell.store(10.0);
        }
        let expectancy_cell = RuntimeCell::<f32>::new(0.0);
        if final_trade_count > 0 {
            expectancy_cell.store(net_profit / final_trade_count as f32);
        }
        let filled_months_cell = RuntimeCell::<i32>::new(0);
        if final_month_ptr >= 0 {
            let raw = final_month_ptr + 1;
            if raw < month_capacity as i32 {
                filled_months_cell.store(raw);
            } else {
                filled_months_cell.store(month_capacity as i32);
            }
        }

        metrics_out[metric_base] = net_profit;
        metrics_out[metric_base + 1] = final_peak;
        metrics_out[metric_base + 2] = final_max_dd;
        metrics_out[metric_base + 3] = win_rate_cell.read();
        metrics_out[metric_base + 4] = pf_cell.read();
        metrics_out[metric_base + 5] = expectancy_cell.read();
        metrics_out[metric_base + 6] = final_max_daily_dd;
        trade_counts_out[gene] = final_trade_count;
        month_counts_out[gene] = filled_months_cell.read();
    }
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    // Phase 64 — both CPU and GPU paths now share the canonical
    // `neoethos_core::utils::mean_std` so CPU/GPU rank parity cannot drift
    // due to a math-helper divergence.
    let (mean, std) = neoethos_core::utils::mean_std(values);
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
    match std::env::var("FOREX_BOT_SEARCH_EVAL_CUDA_DEVICE") {
        // Not set: pick device 0 silently — the canonical default.
        Err(_) => 0,
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(value) => value,
            Err(_) => {
                // The user explicitly set the env var but it did not
                // parse as a usize ("auto", "all", "GPU0" — typos like
                // these used to silently fall back to device 0,
                // running the search on the wrong card without telling
                // anyone. Now we shout, then default.
                tracing::warn!(
                    target: "neoethos_search::gpu",
                    raw = %raw,
                    "FOREX_BOT_SEARCH_EVAL_CUDA_DEVICE is set but not a valid \
                     non-negative integer; falling back to device 0."
                );
                0
            }
        },
    }
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
                .map(|value| (*value).clamp(-1, 1) as i8)
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
    // Note — emit a one-line WARN when we actually saturate
    // so the operator can detect it (was previously silent). The four
    // callsites (timestamp deltas, gap-threshold config, month/day idx)
    // all expect values that comfortably fit in i32 for normal trading
    // data; if we ever DO saturate, the kernel result is wrong and we
    // want it in the log. The cost (one branch per element on the rare
    // path) is negligible vs. the cost of debugging a silent wrong-
    // result later.
    if value > i32::MAX as i64 || value < i32::MIN as i64 {
        tracing::warn!(
            target: "neoethos_search::cubecl_eval",
            value = value,
            "i64 → i32 saturation in cubecl_eval kernel input: value clamped — \
             check upstream data magnitudes (timestamp delta > 24.8 days? \
             gap_threshold_ms > i32::MAX? month/day idx out of range?)"
        );
    }
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
