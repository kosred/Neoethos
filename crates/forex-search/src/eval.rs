#[cfg(feature = "gpu")]
use crate::cubecl_eval::{
    cuda_eval_backtest_kernel_enabled, cuda_eval_signal_kernel_enabled,
    try_evaluate_population_cuda, try_generate_signal_rows_cuda,
};
use crate::genetic::strategy_gene::infer_market_cost_profile;
use crate::quality::Trade;
use ndarray::ArrayView2;
use rayon::prelude::*;
use std::env;
use std::sync::Once;

pub type SmcRow = [i8; 11];

pub struct PopulationEvalInputs<'a> {
    pub close: &'a [f64],
    pub high: &'a [f64],
    pub low: &'a [f64],
    pub indicators: ArrayView2<'a, f32>,
    pub gene_offsets: &'a [i32],
    pub gene_indices: &'a [i32],
    pub gene_weights: &'a [f32],
    pub long_thr: &'a [f32],
    pub short_thr: &'a [f32],
    pub month_idx: &'a [i64],
    pub day_idx: &'a [i64],
    pub sl_pips: &'a [f64],
    pub tp_pips: &'a [f64],
    pub smc_data: &'a [SmcRow],
    pub gene_smc_flags: &'a [SmcRow],
    pub gate_threshold: f32,
    pub weights: &'a [f32; 11],
    pub settings: &'a BacktestSettings,
}

static RAYON_INIT: Once = Once::new();

fn init_rayon() {
    RAYON_INIT.call_once(|| {
        let threads = env::var("FOREX_BOT_RUST_THREADS")
            .ok()
            .or_else(|| env::var("RAYON_NUM_THREADS").ok())
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0);
        if let Some(n) = threads {
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build_global();
        }
    });
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    if values.len() < 2 {
        return (0.0, 0.0);
    }
    let n = values.len() as f64;
    let sum: f64 = values.iter().sum();
    let mean = sum / n;
    let var = values
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / (n - 1.0);
    (mean, var.sqrt())
}

#[derive(Debug, Clone)]
pub struct BacktestSettings {
    pub sl_pips: f64,
    pub tp_pips: f64,
    pub max_hold_bars: usize,
    pub trailing_enabled: bool,
    pub trailing_atr_multiplier: f64,
    pub trailing_be_trigger_r: f64,
    pub pip_value: f64,
    pub spread_pips: f64,
    pub commission_per_trade: f64,
    pub pip_value_per_lot: f64,
    pub kill_zones_enabled: bool,
}

impl Default for BacktestSettings {
    fn default() -> Self {
        let profile = infer_market_cost_profile("", "", None, None, None);
        Self {
            sl_pips: 20.0,
            tp_pips: 40.0,
            max_hold_bars: 0,
            trailing_enabled: false,
            trailing_atr_multiplier: 1.0,
            trailing_be_trigger_r: 1.0,
            pip_value: profile.pip_value,
            spread_pips: profile.spread_pips,
            commission_per_trade: profile.commission_per_trade,
            pip_value_per_lot: profile.pip_value_per_lot,
            kill_zones_enabled: false,
        }
    }
}

impl BacktestSettings {
    pub fn initial_equity(&self) -> f64 {
        env::var("FOREX_BOT_BACKTEST_INITIAL_EQUITY")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(100000.0)
    }

    pub fn month_capacity(&self) -> usize {
        env::var("FOREX_BOT_BACKTEST_MAX_MONTH_BUCKETS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(240)
    }
}

pub fn fast_evaluate_strategy_core(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    signals: &[i8],
    month_idx: &[i64],
    day_idx: &[i64],
    settings: &BacktestSettings,
) -> [f64; 11] {
    let n = close.len();
    if n == 0 {
        return [0.0; 11];
    }

    let initial_equity = settings.initial_equity();
    let month_capacity = settings.month_capacity();

    let mut equity = initial_equity;
    let mut peak_equity = initial_equity;
    let mut max_dd = 0.0;
    let mut trade_count = 0usize;
    let mut wins = 0usize;
    let mut gross_profit = 0.0;
    let mut gross_loss = 0.0;

    let mut last_month = -1i64;
    let mut current_month_pnl = 0.0;
    let mut monthly_pnls = vec![0.0; month_capacity];
    let mut month_ptr = -1i64;

    let mut last_day = -1i64;
    let mut day_peak = equity;
    let mut day_low = equity;
    let mut max_daily_dd = 0.0;

    let mut in_pos = 0i8;
    let mut entry_px = 0.0;
    let mut entry_idx = -1i64;
    let mut trail_px = 0.0;

    let pip = if settings.pip_value.abs() < 1e-12 {
        1e-12
    } else {
        settings.pip_value
    };

    for i in 1..n {
        let m_val = *month_idx.get(i).unwrap_or(&last_month);
        if m_val != last_month {
            if last_month != -1 {
                month_ptr += 1;
                if month_ptr < month_capacity as i64 {
                    monthly_pnls[month_ptr as usize] = current_month_pnl;
                }
            }
            current_month_pnl = 0.0;
            last_month = m_val;
        }

        let d_val = *day_idx.get(i).unwrap_or(&last_day);
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
        }

        if in_pos != 0 {
            let lo = low[i];
            let hi = high[i];
            let worst_float_pnl = if in_pos == 1 {
                (lo - entry_px) / pip * settings.pip_value_per_lot
            } else {
                (entry_px - hi) / pip * settings.pip_value_per_lot
            };
            if (equity + worst_float_pnl) < day_low {
                day_low = equity + worst_float_pnl;
            }

            let best_float_pnl = if in_pos == 1 {
                (hi - entry_px) / pip * settings.pip_value_per_lot
            } else {
                (entry_px - lo) / pip * settings.pip_value_per_lot
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

            let mut pnl = 0.0;
            let mut exit = false;

            if in_pos == 1 {
                let mut sl = entry_px - (settings.sl_pips * pip);
                let tp = entry_px + (settings.tp_pips * pip);
                if settings.trailing_enabled {
                    let mv = hi - entry_px;
                    if mv >= (settings.trailing_be_trigger_r * settings.sl_pips * pip) {
                        let candidate =
                            hi - (settings.trailing_atr_multiplier * settings.sl_pips * pip);
                        if trail_px == 0.0 || candidate > trail_px {
                            trail_px = candidate;
                        }
                        if trail_px > sl {
                            sl = trail_px;
                        }
                    }
                }
                if lo <= sl {
                    pnl = (sl - entry_px) / pip * settings.pip_value_per_lot;
                    exit = true;
                } else if hi >= tp {
                    pnl = (tp - entry_px) / pip * settings.pip_value_per_lot;
                    exit = true;
                }
            } else {
                let mut sl = entry_px + (settings.sl_pips * pip);
                let tp = entry_px - (settings.tp_pips * pip);
                if settings.trailing_enabled {
                    let mv = entry_px - lo;
                    if mv >= (settings.trailing_be_trigger_r * settings.sl_pips * pip) {
                        let candidate =
                            lo + (settings.trailing_atr_multiplier * settings.sl_pips * pip);
                        if trail_px == 0.0 || candidate < trail_px {
                            trail_px = candidate;
                        }
                        if trail_px < sl {
                            sl = trail_px;
                        }
                    }
                }
                if hi >= sl {
                    pnl = (entry_px - sl) / pip * settings.pip_value_per_lot;
                    exit = true;
                } else if lo <= tp {
                    pnl = (entry_px - tp) / pip * settings.pip_value_per_lot;
                    exit = true;
                }
            }

            if !exit
                && settings.max_hold_bars > 0
                && (i as i64 - entry_idx) >= settings.max_hold_bars as i64
            {
                pnl = if in_pos == 1 {
                    (close[i] - entry_px) / pip * settings.pip_value_per_lot
                } else {
                    (entry_px - close[i]) / pip * settings.pip_value_per_lot
                };
                exit = true;
            }

            if exit {
                pnl -= settings.commission_per_trade
                    + (settings.spread_pips * settings.pip_value_per_lot);
                equity += pnl;
                current_month_pnl += pnl;
                trade_count += 1;
                if pnl > 0.0 {
                    wins += 1;
                    gross_profit += pnl;
                } else {
                    gross_loss += pnl.abs();
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
            let s = signals[i];
            if s != 0 {
                in_pos = s;
                entry_px = close[i];
                entry_idx = i as i64;
                trail_px = 0.0;
            }
        }
    }

    let net_profit = equity - initial_equity;
    let win_rate = if trade_count > 0 {
        wins as f64 / trade_count as f64
    } else {
        0.0
    };
    let pf = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else {
        if gross_profit > 0.0 { 10.0 } else { 0.0 }
    };
    let expectancy = if trade_count > 0 {
        net_profit / trade_count as f64
    } else {
        0.0
    };

    let mut month_returns = Vec::new();
    if month_ptr >= 0 {
        let limit = month_ptr.min(month_capacity.saturating_sub(1) as i64) as usize;
        month_returns.extend_from_slice(&monthly_pnls[..=limit]);
    }
    let (avg_m, std_m) = mean_std(&month_returns);

    // Annualize Sharpe factor: sqrt(12) = 3.4641
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

    [
        net_profit,
        sharpe,
        peak_equity,
        max_dd,
        win_rate,
        pf,
        expectancy,
        0.0,
        trade_count as f64,
        consistency,
        max_daily_dd,
    ]
}

pub fn simulate_trades_core(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    timestamps: &[i64],
    signals: &[i8],
    settings: &BacktestSettings,
) -> Vec<Trade> {
    let n = close
        .len()
        .min(high.len())
        .min(low.len())
        .min(timestamps.len())
        .min(signals.len());
    if n == 0 {
        return Vec::new();
    }

    let initial_balance = settings.initial_equity();
    let pip = if settings.pip_value.abs() < 1e-12 {
        1e-12
    } else {
        settings.pip_value
    };

    let mut trades = Vec::new();
    let mut in_pos = 0i8;
    let mut entry_px = 0.0;
    let mut entry_idx = 0usize;
    let mut trail_px = 0.0;

    for i in 1..n {
        if in_pos != 0 {
            let lo = low[i];
            let hi = high[i];
            let mut pnl = 0.0;
            let mut exit = false;

            // Session-Aware Trading (Idea #4.4) - Force exit on Friday 20:00+
            let ts = timestamps.get(i).copied().unwrap_or_default();
            if ts > 0 && settings.kill_zones_enabled {
                let sec_in_day = ts % 86400;
                let hour = sec_in_day / 3600;
                let days_since_epoch = ts / 86400;
                let weekday = (days_since_epoch + 4) % 7; // 0=Sun, 1=Mon, 5=Fri

                if weekday == 5 && hour >= 20 {
                    exit = true; // Force exit before weekend
                    pnl = if in_pos == 1 {
                        (close[i] - entry_px) / pip * settings.pip_value_per_lot
                    } else {
                        (entry_px - close[i]) / pip * settings.pip_value_per_lot
                    };
                }
            }

            if in_pos == 1 && !exit {
                let mut sl = entry_px - (settings.sl_pips * pip);
                let tp = entry_px + (settings.tp_pips * pip);
                if settings.trailing_enabled {
                    let mv = hi - entry_px;
                    if mv >= (settings.trailing_be_trigger_r * settings.sl_pips * pip) {
                        let candidate =
                            hi - (settings.trailing_atr_multiplier * settings.sl_pips * pip);
                        if trail_px == 0.0 || candidate > trail_px {
                            trail_px = candidate;
                        }
                        if trail_px > sl {
                            sl = trail_px;
                        }
                    }
                }
                if lo <= sl {
                    pnl = (sl - entry_px) / pip * settings.pip_value_per_lot;
                    exit = true;
                } else if hi >= tp {
                    pnl = (tp - entry_px) / pip * settings.pip_value_per_lot;
                    exit = true;
                }
            } else {
                let mut sl = entry_px + (settings.sl_pips * pip);
                let tp = entry_px - (settings.tp_pips * pip);
                if settings.trailing_enabled {
                    let mv = entry_px - lo;
                    if mv >= (settings.trailing_be_trigger_r * settings.sl_pips * pip) {
                        let candidate =
                            lo + (settings.trailing_atr_multiplier * settings.sl_pips * pip);
                        if trail_px == 0.0 || candidate < trail_px {
                            trail_px = candidate;
                        }
                        if trail_px < sl {
                            sl = trail_px;
                        }
                    }
                }
                if hi >= sl {
                    pnl = (entry_px - sl) / pip * settings.pip_value_per_lot;
                    exit = true;
                } else if lo <= tp {
                    pnl = (entry_px - tp) / pip * settings.pip_value_per_lot;
                    exit = true;
                }
            }

            if !exit && settings.max_hold_bars > 0 && (i - entry_idx) >= settings.max_hold_bars {
                pnl = if in_pos == 1 {
                    (close[i] - entry_px) / pip * settings.pip_value_per_lot
                } else {
                    (entry_px - close[i]) / pip * settings.pip_value_per_lot
                };
                exit = true;
            }

            if exit {
                pnl -= settings.commission_per_trade
                    + (settings.spread_pips * settings.pip_value_per_lot);
                let entry_time = timestamps.get(entry_idx).copied().unwrap_or_default();
                let exit_time = timestamps.get(i).copied().unwrap_or(entry_time);
                let duration_hours = if exit_time >= entry_time {
                    Some((exit_time - entry_time) as f64 / 3_600_000.0)
                } else {
                    None
                };
                trades.push(Trade {
                    entry_time,
                    exit_time: Some(exit_time),
                    pnl,
                    pnl_pct: Some(pnl / initial_balance),
                    duration_hours,
                });
                in_pos = 0;
            }
        } else if signals[i] != 0 {
            // Session-Aware Trading (Idea #4.4) - Block entries in Kill Zones
            let mut block_entry = false;
            let ts = timestamps.get(i).copied().unwrap_or_default();
            if ts > 0 && settings.kill_zones_enabled {
                let sec_in_day = ts % 86400;
                let hour = sec_in_day / 3600;
                let min = (sec_in_day % 3600) / 60;
                let days_since_epoch = ts / 86400;
                let weekday = (days_since_epoch + 4) % 7; // 0=Sun, 1=Mon, 5=Fri

                let is_friday_kill = weekday == 5 && hour >= 20;
                let is_monday_kill = weekday == 1 && hour == 0 && min < 30;
                if is_friday_kill || is_monday_kill {
                    block_entry = true;
                }
            }

            if !block_entry {
                in_pos = signals[i];
                entry_px = close[i];
                entry_idx = i;
                trail_px = 0.0;
            }
        }
    }

    trades
}

fn synthesize_signals_cpu(
    indicators: ArrayView2<'_, f32>,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f32],
    long_thr: &[f32],
    short_thr: &[f32],
    smc_data: &[SmcRow],
    gene_smc_flags: &[SmcRow],
    gate_threshold: f32,
    weights: &[f32; 11],
    gene_index: usize,
    n_samples: usize,
) -> Vec<i8> {
    let mut combined = vec![0.0_f32; n_samples];
    let start = gene_offsets[gene_index] as usize;
    let end = gene_offsets[gene_index + 1] as usize;
    for i in start..end {
        let idx = gene_indices[i] as usize;
        let w = gene_weights[i];
        if idx < indicators.nrows() {
            let row = indicators.row(idx);
            for (j, &v) in row.iter().enumerate() {
                combined[j] += w * v;
            }
        }
    }

    let mut signals = vec![0i8; n_samples];
    let lt = long_thr[gene_index];
    let st = short_thr[gene_index];
    let flags = gene_smc_flags[gene_index];
    let active_sum: f32 = flags
        .iter()
        .enumerate()
        .map(|(i, &f)| if f != 0 { weights[i] } else { 0.0 })
        .sum();
    let gate = gate_threshold.min(active_sum);

    for i in 0..n_samples {
        let v = combined[i];
        let sig = if v >= lt {
            1
        } else if v <= st {
            -1
        } else {
            0
        };
        if sig == 0 {
            continue;
        }

        if active_sum > 0.0 {
            let mut score = 0.0f32;
            let smc = smc_data[i];
            for j in 0..11 {
                if flags[j] != 0 {
                    if j == 5 {
                        if smc[j] == 1 {
                            score += weights[j];
                        }
                    } else if smc[j] == sig {
                        score += weights[j];
                    }
                }
            }
            if score >= gate {
                signals[i] = sig;
            }
        } else {
            signals[i] = sig;
        }
    }

    signals
}

pub fn evaluate_population_core(
    inputs: PopulationEvalInputs<'_>,
) -> Result<Vec<[f64; 11]>, String> {
    let PopulationEvalInputs {
        close,
        high,
        low,
        indicators,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        month_idx,
        day_idx,
        sl_pips,
        tp_pips,
        smc_data,
        gene_smc_flags,
        gate_threshold,
        weights,
        settings,
    } = inputs;
    init_rayon();
    let n_genes = long_thr.len();
    let n_samples = close.len();

    #[cfg(feature = "gpu")]
    if cuda_eval_signal_kernel_enabled() && cuda_eval_backtest_kernel_enabled() {
        match try_evaluate_population_cuda(
            close,
            high,
            low,
            indicators,
            gene_offsets,
            gene_indices,
            gene_weights,
            long_thr,
            short_thr,
            month_idx,
            day_idx,
            sl_pips,
            tp_pips,
            smc_data,
            gene_smc_flags,
            gate_threshold,
            weights,
            settings,
        ) {
            Ok(results) => return Ok(results),
            Err(err) => {
                tracing::warn!(
                    "full cuda evaluator unavailable, falling back to partial gpu/cpu evaluation: {err}"
                );
            }
        }
    }

    #[cfg(feature = "gpu")]
    let gpu_signal_rows = if cuda_eval_signal_kernel_enabled() {
        match try_generate_signal_rows_cuda(
            indicators,
            gene_offsets,
            gene_indices,
            gene_weights,
            long_thr,
            short_thr,
            smc_data,
            gene_smc_flags,
            gate_threshold,
            weights,
        ) {
            Ok(rows) => Some(rows),
            Err(err) => {
                tracing::warn!(
                    "cuda evaluator signal kernel unavailable, falling back to cpu evaluator synthesis: {err}"
                );
                None
            }
        }
    } else {
        None
    };

    let results: Vec<[f64; 11]> = (0..n_genes)
        .into_par_iter()
        .map(|g| {
            #[cfg(feature = "gpu")]
            let signals = if let Some(signal_rows) = gpu_signal_rows.as_ref() {
                signal_rows[g].clone()
            } else {
                synthesize_signals_cpu(
                    indicators,
                    gene_offsets,
                    gene_indices,
                    gene_weights,
                    long_thr,
                    short_thr,
                    smc_data,
                    gene_smc_flags,
                    gate_threshold,
                    weights,
                    g,
                    n_samples,
                )
            };

            #[cfg(not(feature = "gpu"))]
            let signals = synthesize_signals_cpu(
                indicators,
                gene_offsets,
                gene_indices,
                gene_weights,
                long_thr,
                short_thr,
                smc_data,
                gene_smc_flags,
                gate_threshold,
                weights,
                g,
                n_samples,
            );

            let mut gene_settings = settings.clone();
            gene_settings.sl_pips = sl_pips[g];
            gene_settings.tp_pips = tp_pips[g];
            fast_evaluate_strategy_core(
                close,
                high,
                low,
                &signals,
                month_idx,
                day_idx,
                &gene_settings,
            )
        })
        .collect();

    Ok(results)
}
