//! Test-only causal trade/risk trace specialization for parity levels 4-9.
//!
//! This module is compiled only by the test harness. The production compact
//! backtest kernel has no diagnostic branch and no diagnostic output buffers.

use crate::gpu_native::parity_hierarchy::{
    EntryEventTrace, ExitEventTrace, FloatTolerance, ParityPolicy, ParityTrace, RiskStateTrace,
    TradeTrace, compare_traces,
};
#[cfg(any(feature = "gpu-cuda", feature = "gpu-vulkan"))]
use cubecl::prelude::*;
#[cfg(feature = "gpu-cuda")]
type DiagnosticRuntime = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "gpu-vulkan", not(feature = "gpu-cuda")))]
type DiagnosticRuntime = cubecl::wgpu::WgpuRuntime;

const EXIT_STOP: u32 = 1;
const EXIT_TARGET: u32 = 2;
const EXIT_MAX_HOLD: u32 = 3;

const RISK_NEW_DAY: u32 = 1 << 0;
const RISK_NEW_MONTH: u32 = 1 << 1;
const RISK_DAILY_LOSS_BREACH: u32 = 1 << 2;
const RISK_TOTAL_DRAWDOWN_BREACH: u32 = 1 << 3;
const RISK_PROFIT_TARGET_REACHED: u32 = 1 << 4;

#[derive(Debug, Clone)]
struct TradeTraceSettings {
    initial_equity: f32,
    max_hold_bars: u32,
    min_hold_bars: u32,
    max_trades_per_day: u32,
    trailing_enabled: bool,
    trailing_atr_multiplier: f32,
    trailing_be_trigger_r: f32,
    spread_pips: f32,
    commission_per_trade: f32,
    pip_value_per_lot: f32,
    swap_long_pips_per_day: f32,
    swap_short_pips_per_day: f32,
    pnl_conversion_fee_rate: f32,
    risk_based_sizing: bool,
    risk_per_trade_min: f32,
    risk_per_trade_max: f32,
    high_quality_confidence: f32,
    adaptive_rr: f32,
    ftmo_daily_loss_fraction: f32,
    ftmo_overall_drawdown_fraction: f32,
    ftmo_profit_target_fraction: f32,
}

impl Default for TradeTraceSettings {
    fn default() -> Self {
        let ftmo = neoethos_core::domain::prop_firm::PropFirmConstraints::FTMO_STANDARD;
        Self {
            initial_equity: 100_000.0,
            max_hold_bars: 12,
            min_hold_bars: 0,
            max_trades_per_day: 0,
            trailing_enabled: false,
            trailing_atr_multiplier: 1.0,
            trailing_be_trigger_r: 1.0,
            spread_pips: 0.0,
            commission_per_trade: 0.0,
            pip_value_per_lot: 10.0,
            swap_long_pips_per_day: 0.0,
            swap_short_pips_per_day: 0.0,
            pnl_conversion_fee_rate: 0.0,
            risk_based_sizing: false,
            risk_per_trade_min: 0.005,
            risk_per_trade_max: 0.01,
            high_quality_confidence: 0.65,
            adaptive_rr: 2.0,
            ftmo_daily_loss_fraction: ftmo.max_daily_loss_pct,
            ftmo_overall_drawdown_fraction: ftmo.max_overall_drawdown_pct,
            ftmo_profit_target_fraction: ftmo.challenge_profit_target_pct,
        }
    }
}

#[derive(Debug, Clone)]
struct TradeTraceInput {
    candidate_id: u64,
    scenario_id: u64,
    close_pips: Vec<f32>,
    high_pips: Vec<f32>,
    low_pips: Vec<f32>,
    signals: Vec<i32>,
    confidences: Vec<f32>,
    timestamp_deltas_ms: Vec<i32>,
    month_ids: Vec<i32>,
    day_ids: Vec<i32>,
    adaptive_base_pips: Vec<f32>,
    stop_pips: f32,
    target_pips: f32,
    stop_vol_multiplier: f32,
    settings: TradeTraceSettings,
}

impl TradeTraceInput {
    fn bars(&self) -> usize {
        self.close_pips.len()
    }

    fn validate(&self) -> anyhow::Result<()> {
        let bars = self.bars();
        if bars < 2 {
            anyhow::bail!("trade trace requires at least two bars");
        }
        for (field, actual) in [
            ("high_pips", self.high_pips.len()),
            ("low_pips", self.low_pips.len()),
            ("signals", self.signals.len()),
            ("confidences", self.confidences.len()),
            ("timestamp_deltas_ms", self.timestamp_deltas_ms.len()),
            ("month_ids", self.month_ids.len()),
            ("day_ids", self.day_ids.len()),
            ("adaptive_base_pips", self.adaptive_base_pips.len()),
        ] {
            if actual != bars {
                anyhow::bail!(
                    "trade trace {field} length {actual} does not match bar count {bars}"
                );
            }
        }
        if self
            .close_pips
            .iter()
            .chain(&self.high_pips)
            .chain(&self.low_pips)
            .chain(&self.confidences)
            .chain(&self.adaptive_base_pips)
            .any(|value| !value.is_finite())
        {
            anyhow::bail!("trade trace rejects non-finite arrays");
        }
        Ok(())
    }
}

fn diagnostic_policy() -> ParityPolicy {
    let f32_state = FloatTolerance {
        absolute: 1.0e-4,
        relative: 2.0e-6,
        max_ulps: 16,
    };
    ParityPolicy {
        size_and_costs: f32_state,
        pnl_and_equity: f32_state,
        risk_state: f32_state,
        ..ParityPolicy::default()
    }
}

fn cpu_trade_trace(input: &TradeTraceInput) -> anyhow::Result<ParityTrace> {
    input.validate()?;
    let settings = &input.settings;
    let mut trace = ParityTrace::default();
    let mut equity = settings.initial_equity;
    let mut peak_equity = equity;
    let mut day_start_equity = equity;
    let mut month_start_equity = equity;
    let mut last_day = i32::MIN;
    let mut last_month = i32::MIN;
    let mut day_trade_count = 0_u32;
    let mut current_day_pnl = 0.0_f32;
    let mut max_daily_loss = 0.0_f32;
    let mut eod_peak = equity;
    let mut max_eod_drawdown = 0.0_f32;
    let mut total_pnl = 0.0_f32;
    let mut in_position = 0_i32;
    let mut entry_bar = 0_usize;
    let mut entry_price = 0.0_f32;
    let mut position_size = 1.0_f32;
    let mut position_days = 0.0_f32;
    let mut stop_distance = input.stop_pips;
    let mut target_distance = input.target_pips;
    let mut trail_price = 0.0_f32;

    for bar in 1..input.bars() {
        let mut risk_flags = 0_u32;
        let month_id = input.month_ids[bar];
        if month_id != last_month {
            last_month = month_id;
            month_start_equity = equity;
            risk_flags |= RISK_NEW_MONTH;
        }
        let day_id = input.day_ids[bar];
        if day_id != last_day {
            if last_day != i32::MIN {
                finalize_ftmo_day(
                    current_day_pnl,
                    equity,
                    &mut max_daily_loss,
                    &mut eod_peak,
                    &mut max_eod_drawdown,
                );
            }
            last_day = day_id;
            day_start_equity = equity;
            day_trade_count = 0;
            current_day_pnl = 0.0;
            risk_flags |= RISK_NEW_DAY;
        }

        if in_position != 0 && input.timestamp_deltas_ms[bar] > 0 {
            position_days += input.timestamp_deltas_ms[bar] as f32 / 86_400_000.0;
        }

        if in_position != 0 {
            let best_float = if in_position == 1 {
                (input.high_pips[bar] - entry_price) * settings.pip_value_per_lot * position_size
            } else {
                (entry_price - input.low_pips[bar]) * settings.pip_value_per_lot * position_size
            };
            peak_equity = peak_equity.max(equity + best_float);

            let bars_held = bar - entry_bar;
            let past_min_hold =
                settings.min_hold_bars == 0 || bars_held >= settings.min_hold_bars as usize;
            let mut exit_reason = 0_u32;
            let mut price_pnl_per_lot = 0.0_f32;
            if past_min_hold && in_position == 1 {
                let mut stop = entry_price - stop_distance;
                let target = entry_price + target_distance;
                if settings.trailing_enabled && trail_price > 0.0 && trail_price > stop {
                    stop = trail_price;
                }
                if input.low_pips[bar] <= stop {
                    price_pnl_per_lot = (stop - entry_price) * settings.pip_value_per_lot;
                    exit_reason = EXIT_STOP;
                } else if input.high_pips[bar] >= target {
                    price_pnl_per_lot = (target - entry_price) * settings.pip_value_per_lot;
                    exit_reason = EXIT_TARGET;
                }
                if exit_reason == 0 && settings.trailing_enabled {
                    let favorable = input.high_pips[bar] - entry_price;
                    if favorable >= settings.trailing_be_trigger_r * stop_distance {
                        let candidate =
                            input.high_pips[bar] - settings.trailing_atr_multiplier * stop_distance;
                        if trail_price == 0.0 || candidate > trail_price {
                            trail_price = candidate;
                        }
                    }
                }
            } else if past_min_hold && in_position == -1 {
                let mut stop = entry_price + stop_distance;
                let target = entry_price - target_distance;
                if settings.trailing_enabled && trail_price > 0.0 && trail_price < stop {
                    stop = trail_price;
                }
                if input.high_pips[bar] >= stop {
                    price_pnl_per_lot = (entry_price - stop) * settings.pip_value_per_lot;
                    exit_reason = EXIT_STOP;
                } else if input.low_pips[bar] <= target {
                    price_pnl_per_lot = (entry_price - target) * settings.pip_value_per_lot;
                    exit_reason = EXIT_TARGET;
                }
                if exit_reason == 0 && settings.trailing_enabled {
                    let favorable = entry_price - input.low_pips[bar];
                    if favorable >= settings.trailing_be_trigger_r * stop_distance {
                        let candidate =
                            input.low_pips[bar] + settings.trailing_atr_multiplier * stop_distance;
                        if trail_price == 0.0 || candidate < trail_price {
                            trail_price = candidate;
                        }
                    }
                }
            }
            if exit_reason == 0
                && past_min_hold
                && settings.max_hold_bars > 0
                && bars_held >= settings.max_hold_bars as usize
            {
                price_pnl_per_lot = if in_position == 1 {
                    (input.close_pips[bar] - entry_price) * settings.pip_value_per_lot
                } else {
                    (entry_price - input.close_pips[bar]) * settings.pip_value_per_lot
                };
                exit_reason = EXIT_MAX_HOLD;
            }

            if exit_reason != 0 {
                let explicit_cost_per_lot = settings.commission_per_trade
                    + settings.spread_pips * 0.5 * settings.pip_value_per_lot;
                let scaled_after_explicit =
                    (price_pnl_per_lot - explicit_cost_per_lot) * position_size;
                let swap_per_day = if in_position == 1 {
                    settings.swap_long_pips_per_day
                } else {
                    settings.swap_short_pips_per_day
                };
                let swap_credit =
                    swap_per_day * position_days * settings.pip_value_per_lot * position_size;
                let before_conversion = scaled_after_explicit + swap_credit;
                let conversion_fee = if settings.pnl_conversion_fee_rate > 0.0
                    && settings.pnl_conversion_fee_rate < 1.0
                {
                    before_conversion * settings.pnl_conversion_fee_rate
                } else {
                    0.0
                };
                let pnl = before_conversion - conversion_fee;
                // The entry price embeds the entry half-spread and the PnL
                // calculation above deducts the exit half-spread. Report the
                // complete round-trip spread at level 7.
                let reported_cost_per_lot = settings.commission_per_trade
                    + settings.spread_pips * settings.pip_value_per_lot;
                let costs = reported_cost_per_lot * position_size - swap_credit + conversion_fee;
                equity += pnl;
                current_day_pnl += pnl;
                total_pnl += pnl;
                peak_equity = peak_equity.max(equity);
                trace.exits.push(ExitEventTrace {
                    candidate_id: input.candidate_id,
                    scenario_id: input.scenario_id,
                    bar: bar as u32,
                    reason: exit_reason,
                });
                trace.accepted_trades.push(TradeTrace {
                    candidate_id: input.candidate_id,
                    scenario_id: input.scenario_id,
                    entry_bar: entry_bar as u32,
                    exit_bar: bar as u32,
                    direction: in_position as i8,
                    exit_reason,
                    size: position_size as f64,
                    costs: costs as f64,
                    pnl: pnl as f64,
                    equity_after: equity as f64,
                });
                in_position = 0;
            }
        } else {
            let signal = input.signals[bar - 1];
            if signal != 0
                && !(settings.max_trades_per_day > 0
                    && day_trade_count >= settings.max_trades_per_day)
            {
                in_position = signal;
                entry_bar = bar;
                entry_price = input.close_pips[bar] + signal as f32 * settings.spread_pips * 0.5;
                trail_price = 0.0;
                position_days = 0.0;
                stop_distance = input.stop_pips;
                target_distance = input.target_pips;
                if input.stop_vol_multiplier > 0.0 {
                    let adaptive_stop =
                        input.stop_vol_multiplier * input.adaptive_base_pips[bar - 1];
                    if adaptive_stop.is_finite() && adaptive_stop > 0.0 {
                        stop_distance = adaptive_stop;
                        target_distance = settings.adaptive_rr * adaptive_stop;
                    }
                }
                position_size = if settings.risk_based_sizing {
                    risk_based_size(input.confidences[bar - 1], equity, stop_distance, settings)
                } else {
                    1.0
                };
                day_trade_count += 1;
                trace.entries.push(EntryEventTrace {
                    candidate_id: input.candidate_id,
                    scenario_id: input.scenario_id,
                    bar: bar as u32,
                    direction: signal as i8,
                });
            }
        }

        if bar + 1 == input.bars() {
            finalize_ftmo_day(
                current_day_pnl,
                equity,
                &mut max_daily_loss,
                &mut eod_peak,
                &mut max_eod_drawdown,
            );
        }
        if max_daily_loss >= settings.initial_equity * settings.ftmo_daily_loss_fraction {
            risk_flags |= RISK_DAILY_LOSS_BREACH;
        }
        if max_eod_drawdown >= settings.ftmo_overall_drawdown_fraction {
            risk_flags |= RISK_TOTAL_DRAWDOWN_BREACH;
        }
        if total_pnl >= settings.initial_equity * settings.ftmo_profit_target_fraction {
            risk_flags |= RISK_PROFIT_TARGET_REACHED;
        }
        trace.risk_states.push(RiskStateTrace {
            candidate_id: input.candidate_id,
            scenario_id: input.scenario_id,
            bar: bar as u32,
            day_id: day_id as i64,
            month_id: month_id as i64,
            equity: equity as f64,
            peak_equity: peak_equity as f64,
            day_start_equity: day_start_equity as f64,
            month_start_equity: month_start_equity as f64,
            flags: risk_flags,
        });
    }
    Ok(trace)
}

fn finalize_ftmo_day(
    day_pnl: f32,
    equity: f32,
    max_daily_loss: &mut f32,
    eod_peak: &mut f32,
    max_eod_drawdown: &mut f32,
) {
    if day_pnl < 0.0 {
        *max_daily_loss = (*max_daily_loss).max(-day_pnl);
    }
    *eod_peak = (*eod_peak).max(equity);
    if *eod_peak > 0.0 {
        *max_eod_drawdown = (*max_eod_drawdown).max((*eod_peak - equity) / *eod_peak);
    }
}

fn risk_based_size(
    confidence: f32,
    equity: f32,
    stop_pips: f32,
    settings: &TradeTraceSettings,
) -> f32 {
    let confidence = confidence.clamp(0.0, 1.0);
    let confidence_scale = if settings.high_quality_confidence > 0.0 {
        (confidence / settings.high_quality_confidence).min(1.0)
    } else {
        1.0
    };
    let risk_fraction = settings.risk_per_trade_min
        + (settings.risk_per_trade_max - settings.risk_per_trade_min) * confidence_scale;
    let denominator = stop_pips.max(1.0) * settings.pip_value_per_lot;
    if equity > 0.0 && denominator > 1.0e-12 {
        ((risk_fraction * equity) / denominator).clamp(0.0, 100.0)
    } else {
        0.0
    }
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-vulkan"))]
#[cube(launch)]
fn trade_trace_kernel(
    close_pips: &Array<f32>,
    high_pips: &Array<f32>,
    low_pips: &Array<f32>,
    signals: &Array<i32>,
    confidences: &Array<f32>,
    timestamp_deltas_ms: &Array<i32>,
    month_ids: &Array<i32>,
    day_ids: &Array<i32>,
    adaptive_base_pips: &Array<f32>,
    identity_in: &Array<u32>,
    identity_out: &mut Array<u32>,
    entry_directions: &mut Array<i32>,
    exit_reasons: &mut Array<i32>,
    trade_entry_bars: &mut Array<i32>,
    trade_directions: &mut Array<i32>,
    trade_sizes: &mut Array<f32>,
    trade_costs: &mut Array<f32>,
    trade_pnls: &mut Array<f32>,
    trade_equities: &mut Array<f32>,
    risk_equities: &mut Array<f32>,
    risk_peaks: &mut Array<f32>,
    risk_day_starts: &mut Array<f32>,
    risk_month_starts: &mut Array<f32>,
    risk_flags_out: &mut Array<i32>,
    n_samples: u32,
    initial_equity: f32,
    stop_pips: f32,
    target_pips: f32,
    stop_vol_multiplier: f32,
    adaptive_rr: f32,
    max_hold_bars: u32,
    min_hold_bars: u32,
    max_trades_per_day: u32,
    trailing_enabled: i32,
    trailing_atr_multiplier: f32,
    trailing_be_trigger_r: f32,
    spread_pips: f32,
    commission_per_trade: f32,
    pip_value_per_lot: f32,
    swap_long_pips_per_day: f32,
    swap_short_pips_per_day: f32,
    pnl_conversion_fee_rate: f32,
    risk_based_sizing: i32,
    risk_per_trade_min: f32,
    risk_per_trade_max: f32,
    high_quality_confidence: f32,
    ftmo_daily_loss_fraction: f32,
    ftmo_overall_drawdown_fraction: f32,
    ftmo_profit_target_fraction: f32,
) {
    if ABSOLUTE_POS == 0 {
        // One candidate/scenario is intentionally traced per diagnostic launch.
        // Copy both u64 IDs as low/high u32 words so every host-side event is
        // reconstructed from device output rather than stamped from host input.
        for word in 0..4 {
            identity_out[word] = identity_in[word];
        }
        let n_samples = n_samples as usize;
        for bar in 0..n_samples {
            entry_directions[bar] = 0;
            exit_reasons[bar] = 0;
            trade_entry_bars[bar] = -1;
            trade_directions[bar] = 0;
            trade_sizes[bar] = 0.0;
            trade_costs[bar] = 0.0;
            trade_pnls[bar] = 0.0;
            trade_equities[bar] = 0.0;
            risk_equities[bar] = initial_equity;
            risk_peaks[bar] = initial_equity;
            risk_day_starts[bar] = initial_equity;
            risk_month_starts[bar] = initial_equity;
            risk_flags_out[bar] = 0;
        }

        let equity = RuntimeCell::<f32>::new(initial_equity);
        let peak_equity = RuntimeCell::<f32>::new(initial_equity);
        let day_start_equity = RuntimeCell::<f32>::new(initial_equity);
        let month_start_equity = RuntimeCell::<f32>::new(initial_equity);
        let last_day = RuntimeCell::<i32>::new(-2147483647);
        let last_month = RuntimeCell::<i32>::new(-2147483647);
        let day_trade_count = RuntimeCell::<u32>::new(0);
        let current_day_pnl = RuntimeCell::<f32>::new(0.0);
        let max_daily_loss = RuntimeCell::<f32>::new(0.0);
        let eod_peak = RuntimeCell::<f32>::new(initial_equity);
        let max_eod_drawdown = RuntimeCell::<f32>::new(0.0);
        let total_pnl = RuntimeCell::<f32>::new(0.0);
        let in_position = RuntimeCell::<i32>::new(0);
        let entry_bar = RuntimeCell::<i32>::new(0);
        let entry_price = RuntimeCell::<f32>::new(0.0);
        let position_size = RuntimeCell::<f32>::new(1.0);
        let position_days = RuntimeCell::<f32>::new(0.0);
        let stop_distance = RuntimeCell::<f32>::new(stop_pips);
        let target_distance = RuntimeCell::<f32>::new(target_pips);
        let trail_price = RuntimeCell::<f32>::new(0.0);

        for bar in 1..n_samples {
            let risk_flags = RuntimeCell::<i32>::new(0);
            let month_id = month_ids[bar];
            if month_id != last_month.read() {
                last_month.store(month_id);
                month_start_equity.store(equity.read());
                risk_flags.store(risk_flags.read() + RISK_NEW_MONTH as i32);
            }
            let day_id = day_ids[bar];
            if day_id != last_day.read() {
                if last_day.read() != -2147483647 {
                    if current_day_pnl.read() < 0.0
                        && 0.0 - current_day_pnl.read() > max_daily_loss.read()
                    {
                        max_daily_loss.store(0.0 - current_day_pnl.read());
                    }
                    if equity.read() > eod_peak.read() {
                        eod_peak.store(equity.read());
                    }
                    if eod_peak.read() > 0.0 {
                        let drawdown = (eod_peak.read() - equity.read()) / eod_peak.read();
                        if drawdown > max_eod_drawdown.read() {
                            max_eod_drawdown.store(drawdown);
                        }
                    }
                }
                last_day.store(day_id);
                day_start_equity.store(equity.read());
                day_trade_count.store(0);
                current_day_pnl.store(0.0);
                risk_flags.store(risk_flags.read() + RISK_NEW_DAY as i32);
            }

            if in_position.read() != 0 && timestamp_deltas_ms[bar] > 0 {
                position_days
                    .store(position_days.read() + timestamp_deltas_ms[bar] as f32 / 86_400_000.0);
            }

            if in_position.read() != 0 {
                let best_float = RuntimeCell::<f32>::new(
                    (entry_price.read() - low_pips[bar]) * pip_value_per_lot * position_size.read(),
                );
                if in_position.read() == 1 {
                    best_float.store(
                        (high_pips[bar] - entry_price.read())
                            * pip_value_per_lot
                            * position_size.read(),
                    );
                }
                if equity.read() + best_float.read() > peak_equity.read() {
                    peak_equity.store(equity.read() + best_float.read());
                }

                let bars_held = bar as i32 - entry_bar.read();
                let past_min_hold = min_hold_bars == 0 || bars_held >= min_hold_bars as i32;
                let exit_reason = RuntimeCell::<i32>::new(0);
                let price_pnl_per_lot = RuntimeCell::<f32>::new(0.0);
                if past_min_hold && in_position.read() == 1 {
                    let stop = RuntimeCell::<f32>::new(entry_price.read() - stop_distance.read());
                    let target = entry_price.read() + target_distance.read();
                    if trailing_enabled != 0
                        && trail_price.read() > 0.0
                        && trail_price.read() > stop.read()
                    {
                        stop.store(trail_price.read());
                    }
                    if low_pips[bar] <= stop.read() {
                        price_pnl_per_lot
                            .store((stop.read() - entry_price.read()) * pip_value_per_lot);
                        exit_reason.store(EXIT_STOP as i32);
                    } else if high_pips[bar] >= target {
                        price_pnl_per_lot.store((target - entry_price.read()) * pip_value_per_lot);
                        exit_reason.store(EXIT_TARGET as i32);
                    }
                    if exit_reason.read() == 0 && trailing_enabled != 0 {
                        let favorable = high_pips[bar] - entry_price.read();
                        if favorable >= trailing_be_trigger_r * stop_distance.read() {
                            let candidate =
                                high_pips[bar] - trailing_atr_multiplier * stop_distance.read();
                            if trail_price.read() == 0.0 || candidate > trail_price.read() {
                                trail_price.store(candidate);
                            }
                        }
                    }
                } else if past_min_hold {
                    let stop = RuntimeCell::<f32>::new(entry_price.read() + stop_distance.read());
                    let target = entry_price.read() - target_distance.read();
                    if trailing_enabled != 0
                        && trail_price.read() > 0.0
                        && trail_price.read() < stop.read()
                    {
                        stop.store(trail_price.read());
                    }
                    if high_pips[bar] >= stop.read() {
                        price_pnl_per_lot
                            .store((entry_price.read() - stop.read()) * pip_value_per_lot);
                        exit_reason.store(EXIT_STOP as i32);
                    } else if low_pips[bar] <= target {
                        price_pnl_per_lot.store((entry_price.read() - target) * pip_value_per_lot);
                        exit_reason.store(EXIT_TARGET as i32);
                    }
                    if exit_reason.read() == 0 && trailing_enabled != 0 {
                        let favorable = entry_price.read() - low_pips[bar];
                        if favorable >= trailing_be_trigger_r * stop_distance.read() {
                            let candidate =
                                low_pips[bar] + trailing_atr_multiplier * stop_distance.read();
                            if trail_price.read() == 0.0 || candidate < trail_price.read() {
                                trail_price.store(candidate);
                            }
                        }
                    }
                }
                if exit_reason.read() == 0
                    && past_min_hold
                    && max_hold_bars > 0
                    && bars_held >= max_hold_bars as i32
                {
                    if in_position.read() == 1 {
                        price_pnl_per_lot
                            .store((close_pips[bar] - entry_price.read()) * pip_value_per_lot);
                    } else {
                        price_pnl_per_lot
                            .store((entry_price.read() - close_pips[bar]) * pip_value_per_lot);
                    }
                    exit_reason.store(EXIT_MAX_HOLD as i32);
                }

                if exit_reason.read() != 0 {
                    let explicit_cost_per_lot =
                        commission_per_trade + spread_pips * 0.5 * pip_value_per_lot;
                    let scaled_after_explicit =
                        (price_pnl_per_lot.read() - explicit_cost_per_lot) * position_size.read();
                    let swap_per_day = RuntimeCell::<f32>::new(swap_short_pips_per_day);
                    if in_position.read() == 1 {
                        swap_per_day.store(swap_long_pips_per_day);
                    }
                    let swap_credit = swap_per_day.read()
                        * position_days.read()
                        * pip_value_per_lot
                        * position_size.read();
                    let before_conversion = scaled_after_explicit + swap_credit;
                    let conversion_fee = RuntimeCell::<f32>::new(0.0);
                    if pnl_conversion_fee_rate > 0.0 && pnl_conversion_fee_rate < 1.0 {
                        conversion_fee.store(before_conversion * pnl_conversion_fee_rate);
                    }
                    let pnl = before_conversion - conversion_fee.read();
                    let reported_cost_per_lot =
                        commission_per_trade + spread_pips * pip_value_per_lot;
                    let costs = reported_cost_per_lot * position_size.read() - swap_credit
                        + conversion_fee.read();
                    equity.store(equity.read() + pnl);
                    current_day_pnl.store(current_day_pnl.read() + pnl);
                    total_pnl.store(total_pnl.read() + pnl);
                    if equity.read() > peak_equity.read() {
                        peak_equity.store(equity.read());
                    }
                    exit_reasons[bar] = exit_reason.read();
                    trade_entry_bars[bar] = entry_bar.read();
                    trade_directions[bar] = in_position.read();
                    trade_sizes[bar] = position_size.read();
                    trade_costs[bar] = costs;
                    trade_pnls[bar] = pnl;
                    trade_equities[bar] = equity.read();
                    in_position.store(0);
                }
            } else {
                let signal = signals[bar - 1];
                if signal != 0
                    && !(max_trades_per_day > 0 && day_trade_count.read() >= max_trades_per_day)
                {
                    in_position.store(signal);
                    entry_bar.store(bar as i32);
                    entry_price.store(close_pips[bar] + signal as f32 * spread_pips * 0.5);
                    trail_price.store(0.0);
                    position_days.store(0.0);
                    stop_distance.store(stop_pips);
                    target_distance.store(target_pips);
                    if stop_vol_multiplier > 0.0 {
                        let adaptive_stop = stop_vol_multiplier * adaptive_base_pips[bar - 1];
                        if adaptive_stop > 0.0 {
                            stop_distance.store(adaptive_stop);
                            target_distance.store(adaptive_rr * adaptive_stop);
                        }
                    }
                    if risk_based_sizing != 0 {
                        let confidence = RuntimeCell::<f32>::new(confidences[bar - 1]);
                        if confidence.read() < 0.0 {
                            confidence.store(0.0);
                        }
                        if confidence.read() > 1.0 {
                            confidence.store(1.0);
                        }
                        let confidence_scale = RuntimeCell::<f32>::new(1.0);
                        if high_quality_confidence > 0.0 {
                            let ratio = confidence.read() / high_quality_confidence;
                            if ratio < 1.0 {
                                confidence_scale.store(ratio);
                            }
                        }
                        let risk_fraction = risk_per_trade_min
                            + (risk_per_trade_max - risk_per_trade_min) * confidence_scale.read();
                        let effective_stop = RuntimeCell::<f32>::new(1.0);
                        if stop_distance.read() > 1.0 {
                            effective_stop.store(stop_distance.read());
                        }
                        let denominator = effective_stop.read() * pip_value_per_lot;
                        let size = RuntimeCell::<f32>::new(0.0);
                        if equity.read() > 0.0 && denominator > 1.0e-12 {
                            size.store((risk_fraction * equity.read()) / denominator);
                        }
                        if size.read() > 100.0 {
                            size.store(100.0);
                        }
                        if size.read() < 0.0 {
                            size.store(0.0);
                        }
                        position_size.store(size.read());
                    } else {
                        position_size.store(1.0);
                    }
                    day_trade_count.store(day_trade_count.read() + 1);
                    entry_directions[bar] = signal;
                }
            }

            if bar + 1 == n_samples {
                if current_day_pnl.read() < 0.0
                    && 0.0 - current_day_pnl.read() > max_daily_loss.read()
                {
                    max_daily_loss.store(0.0 - current_day_pnl.read());
                }
                if equity.read() > eod_peak.read() {
                    eod_peak.store(equity.read());
                }
                if eod_peak.read() > 0.0 {
                    let drawdown = (eod_peak.read() - equity.read()) / eod_peak.read();
                    if drawdown > max_eod_drawdown.read() {
                        max_eod_drawdown.store(drawdown);
                    }
                }
            }
            if max_daily_loss.read() >= initial_equity * ftmo_daily_loss_fraction {
                risk_flags.store(risk_flags.read() + RISK_DAILY_LOSS_BREACH as i32);
            }
            if max_eod_drawdown.read() >= ftmo_overall_drawdown_fraction {
                risk_flags.store(risk_flags.read() + RISK_TOTAL_DRAWDOWN_BREACH as i32);
            }
            if total_pnl.read() >= initial_equity * ftmo_profit_target_fraction {
                risk_flags.store(risk_flags.read() + RISK_PROFIT_TARGET_REACHED as i32);
            }
            risk_equities[bar] = equity.read();
            risk_peaks[bar] = peak_equity.read();
            risk_day_starts[bar] = day_start_equity.read();
            risk_month_starts[bar] = month_start_equity.read();
            risk_flags_out[bar] = risk_flags.read();
        }
    }
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-vulkan"))]
fn gpu_trade_trace(
    input: &TradeTraceInput,
) -> Result<ParityTrace, crate::gpu_native::engine::EngineError> {
    #[cfg(feature = "gpu-cuda")]
    {
        let client = create_diagnostic_client()?;
        return launch_trade_trace::<cubecl::cuda::CudaRuntime>(&client, input);
    }
    #[cfg(all(feature = "gpu-vulkan", not(feature = "gpu-cuda")))]
    {
        let client = create_diagnostic_client()?;
        launch_trade_trace::<cubecl::wgpu::WgpuRuntime>(&client, input)
    }
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-vulkan"))]
fn create_diagnostic_client()
-> Result<ComputeClient<DiagnosticRuntime>, crate::gpu_native::engine::EngineError> {
    crate::cubecl_eval::create_gpu_client(None).map_err(|error| {
        let message = error.to_string();
        if crate::gpu_native::prototype_a::is_known_no_adapter_error(&message) {
            crate::gpu_native::engine::EngineError::UnsupportedCapability {
                operation: "diagnostic_gpu_adapter",
                detail: message,
            }
        } else {
            crate::gpu_native::engine::EngineError::Backend(message)
        }
    })
}

#[cfg(any(feature = "gpu-cuda", feature = "gpu-vulkan"))]
fn launch_trade_trace<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TradeTraceInput,
) -> Result<ParityTrace, crate::gpu_native::engine::EngineError> {
    input
        .validate()
        .map_err(|error| crate::gpu_native::engine::EngineError::Backend(error.to_string()))?;
    let n = input.bars();
    let close = client.create_from_slice(f32::as_bytes(&input.close_pips));
    let high = client.create_from_slice(f32::as_bytes(&input.high_pips));
    let low = client.create_from_slice(f32::as_bytes(&input.low_pips));
    let signals = client.create_from_slice(i32::as_bytes(&input.signals));
    let confidences = client.create_from_slice(f32::as_bytes(&input.confidences));
    let timestamp_deltas = client.create_from_slice(i32::as_bytes(&input.timestamp_deltas_ms));
    let month_ids = client.create_from_slice(i32::as_bytes(&input.month_ids));
    let day_ids = client.create_from_slice(i32::as_bytes(&input.day_ids));
    let adaptive_base = client.create_from_slice(f32::as_bytes(&input.adaptive_base_pips));
    let identity_words = [
        input.candidate_id as u32,
        (input.candidate_id >> 32) as u32,
        input.scenario_id as u32,
        (input.scenario_id >> 32) as u32,
    ];
    let identity_in = client.create_from_slice(u32::as_bytes(&identity_words));
    let identity_out = client.empty(identity_words.len() * core::mem::size_of::<u32>());
    let i32_bytes = n * core::mem::size_of::<i32>();
    let f32_bytes = n * core::mem::size_of::<f32>();
    let entry_directions = client.empty(i32_bytes);
    let exit_reasons = client.empty(i32_bytes);
    let trade_entry_bars = client.empty(i32_bytes);
    let trade_directions = client.empty(i32_bytes);
    let trade_sizes = client.empty(f32_bytes);
    let trade_costs = client.empty(f32_bytes);
    let trade_pnls = client.empty(f32_bytes);
    let trade_equities = client.empty(f32_bytes);
    let risk_equities = client.empty(f32_bytes);
    let risk_peaks = client.empty(f32_bytes);
    let risk_day_starts = client.empty(f32_bytes);
    let risk_month_starts = client.empty(f32_bytes);
    let risk_flags = client.empty(i32_bytes);
    let settings = &input.settings;

    trade_trace_kernel::launch::<R>(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(close, n) },
        unsafe { ArrayArg::from_raw_parts(high, n) },
        unsafe { ArrayArg::from_raw_parts(low, n) },
        unsafe { ArrayArg::from_raw_parts(signals, n) },
        unsafe { ArrayArg::from_raw_parts(confidences, n) },
        unsafe { ArrayArg::from_raw_parts(timestamp_deltas, n) },
        unsafe { ArrayArg::from_raw_parts(month_ids, n) },
        unsafe { ArrayArg::from_raw_parts(day_ids, n) },
        unsafe { ArrayArg::from_raw_parts(adaptive_base, n) },
        unsafe { ArrayArg::from_raw_parts(identity_in, identity_words.len()) },
        unsafe { ArrayArg::from_raw_parts(identity_out.clone(), identity_words.len()) },
        unsafe { ArrayArg::from_raw_parts(entry_directions.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(exit_reasons.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(trade_entry_bars.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(trade_directions.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(trade_sizes.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(trade_costs.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(trade_pnls.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(trade_equities.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(risk_equities.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(risk_peaks.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(risk_day_starts.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(risk_month_starts.clone(), n) },
        unsafe { ArrayArg::from_raw_parts(risk_flags.clone(), n) },
        n as u32,
        settings.initial_equity,
        input.stop_pips,
        input.target_pips,
        input.stop_vol_multiplier,
        settings.adaptive_rr,
        settings.max_hold_bars,
        settings.min_hold_bars,
        settings.max_trades_per_day,
        i32::from(settings.trailing_enabled),
        settings.trailing_atr_multiplier,
        settings.trailing_be_trigger_r,
        settings.spread_pips,
        settings.commission_per_trade,
        settings.pip_value_per_lot,
        settings.swap_long_pips_per_day,
        settings.swap_short_pips_per_day,
        settings.pnl_conversion_fee_rate,
        i32::from(settings.risk_based_sizing),
        settings.risk_per_trade_min,
        settings.risk_per_trade_max,
        settings.high_quality_confidence,
        settings.ftmo_daily_loss_fraction,
        settings.ftmo_overall_drawdown_fraction,
        settings.ftmo_profit_target_fraction,
    );

    let read_i32 = |handle| {
        client
            .read_one(handle)
            .map(|bytes| i32::from_bytes(bytes.as_ref()).to_vec())
            .map_err(|error| {
                crate::gpu_native::engine::EngineError::Backend(format!(
                    "read diagnostic i32 trace: {error:?}"
                ))
            })
    };
    let read_f32 = |handle| {
        client
            .read_one(handle)
            .map(|bytes| f32::from_bytes(bytes.as_ref()).to_vec())
            .map_err(|error| {
                crate::gpu_native::engine::EngineError::Backend(format!(
                    "read diagnostic f32 trace: {error:?}"
                ))
            })
    };
    let identity_words = client
        .read_one(identity_out)
        .map(|bytes| u32::from_bytes(bytes.as_ref()).to_vec())
        .map_err(|error| {
            crate::gpu_native::engine::EngineError::Backend(format!(
                "read diagnostic identity trace: {error:?}"
            ))
        })?;
    if identity_words.len() != 4 {
        return Err(crate::gpu_native::engine::EngineError::Backend(format!(
            "diagnostic identity output has {} words, expected 4",
            identity_words.len()
        )));
    }
    let candidate_id = u64::from(identity_words[0]) | (u64::from(identity_words[1]) << 32);
    let scenario_id = u64::from(identity_words[2]) | (u64::from(identity_words[3]) << 32);
    let entry_directions = read_i32(entry_directions)?;
    let exit_reasons = read_i32(exit_reasons)?;
    let trade_entry_bars = read_i32(trade_entry_bars)?;
    let trade_directions = read_i32(trade_directions)?;
    let trade_sizes = read_f32(trade_sizes)?;
    let trade_costs = read_f32(trade_costs)?;
    let trade_pnls = read_f32(trade_pnls)?;
    let trade_equities = read_f32(trade_equities)?;
    let risk_equities = read_f32(risk_equities)?;
    let risk_peaks = read_f32(risk_peaks)?;
    let risk_day_starts = read_f32(risk_day_starts)?;
    let risk_month_starts = read_f32(risk_month_starts)?;
    let risk_flags = read_i32(risk_flags)?;

    let mut trace = ParityTrace::default();
    for bar in 1..n {
        let direction = entry_directions[bar];
        if direction != 0 {
            trace.entries.push(EntryEventTrace {
                candidate_id,
                scenario_id,
                bar: bar as u32,
                direction: direction as i8,
            });
        }
        let reason = exit_reasons[bar];
        if reason != 0 {
            trace.exits.push(ExitEventTrace {
                candidate_id,
                scenario_id,
                bar: bar as u32,
                reason: reason as u32,
            });
            trace.accepted_trades.push(TradeTrace {
                candidate_id,
                scenario_id,
                entry_bar: trade_entry_bars[bar] as u32,
                exit_bar: bar as u32,
                direction: trade_directions[bar] as i8,
                exit_reason: reason as u32,
                size: trade_sizes[bar] as f64,
                costs: trade_costs[bar] as f64,
                pnl: trade_pnls[bar] as f64,
                equity_after: trade_equities[bar] as f64,
            });
        }
        trace.risk_states.push(RiskStateTrace {
            candidate_id,
            scenario_id,
            bar: bar as u32,
            day_id: input.day_ids[bar] as i64,
            month_id: input.month_ids[bar] as i64,
            equity: risk_equities[bar] as f64,
            peak_equity: risk_peaks[bar] as f64,
            day_start_equity: risk_day_starts[bar] as f64,
            month_start_equity: risk_month_starts[bar] as f64,
            flags: risk_flags[bar] as u32,
        });
    }
    Ok(trace)
}

fn fixed_same_bar_fixture() -> TradeTraceInput {
    TradeTraceInput {
        candidate_id: 10,
        scenario_id: 100,
        close_pips: vec![100.0; 8],
        high_pips: vec![100.0, 100.0, 103.0, 100.0, 103.0, 100.0, 100.0, 100.0],
        low_pips: vec![100.0, 100.0, 97.0, 100.0, 97.0, 100.0, 100.0, 100.0],
        signals: vec![1, 0, -1, 0, 0, 0, 0, 0],
        confidences: vec![0.5; 8],
        timestamp_deltas_ms: vec![0; 8],
        month_ids: vec![0; 8],
        day_ids: vec![0; 8],
        adaptive_base_pips: vec![0.0; 8],
        stop_pips: 2.0,
        target_pips: 2.0,
        stop_vol_multiplier: 0.0,
        settings: TradeTraceSettings::default(),
    }
}

fn adaptive_cost_sizing_fixture() -> TradeTraceInput {
    let mut settings = TradeTraceSettings::default();
    settings.spread_pips = 2.0;
    settings.commission_per_trade = 3.0;
    settings.swap_long_pips_per_day = -0.5;
    settings.pnl_conversion_fee_rate = 0.005;
    settings.risk_based_sizing = true;
    TradeTraceInput {
        candidate_id: 20,
        scenario_id: 200,
        close_pips: vec![100.0; 10],
        high_pips: vec![
            100.0, 100.0, 104.0, 114.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
        ],
        low_pips: vec![
            100.0, 100.0, 98.0, 99.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
        ],
        signals: vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        confidences: vec![0.65; 10],
        timestamp_deltas_ms: vec![0, 0, 43_200_000, 43_200_000, 0, 0, 0, 0, 0, 0],
        month_ids: vec![0; 10],
        day_ids: vec![0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        adaptive_base_pips: vec![3.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        stop_pips: 2.0,
        target_pips: 4.0,
        stop_vol_multiplier: 2.0,
        settings,
    }
}

fn trailing_calendar_ftmo_fixture() -> TradeTraceInput {
    let mut settings = TradeTraceSettings::default();
    settings.initial_equity = 1_000.0;
    settings.trailing_enabled = true;
    settings.trailing_be_trigger_r = 1.0;
    settings.trailing_atr_multiplier = 1.0;
    settings.pip_value_per_lot = 20.0;
    settings.max_trades_per_day = 1;
    TradeTraceInput {
        candidate_id: 30,
        scenario_id: 300,
        close_pips: vec![100.0; 12],
        high_pips: vec![
            100.0, 100.0, 105.0, 101.0, 100.0, 106.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
        ],
        low_pips: vec![
            100.0, 100.0, 99.0, 99.5, 100.0, 99.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
        ],
        signals: vec![1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        confidences: vec![0.5; 12],
        timestamp_deltas_ms: vec![0; 12],
        month_ids: vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        day_ids: vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        adaptive_base_pips: vec![0.0; 12],
        stop_pips: 5.0,
        target_pips: 20.0,
        stop_vol_multiplier: 0.0,
        settings,
    }
}

fn ftmo_profit_target_fixture() -> TradeTraceInput {
    let mut settings = TradeTraceSettings::default();
    settings.initial_equity = 1_000.0;
    settings.pip_value_per_lot = 20.0;
    TradeTraceInput {
        candidate_id: 0x1_0000_001e,
        scenario_id: 0x2_0000_012c,
        close_pips: vec![100.0; 5],
        high_pips: vec![100.0, 100.0, 111.0, 100.0, 100.0],
        low_pips: vec![100.0; 5],
        signals: vec![1, 0, 0, 0, 0],
        confidences: vec![0.5; 5],
        timestamp_deltas_ms: vec![0; 5],
        month_ids: vec![0; 5],
        day_ids: vec![0; 5],
        adaptive_base_pips: vec![0.0; 5],
        stop_pips: 5.0,
        target_pips: 10.0,
        stop_vol_multiplier: 0.0,
        settings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_diagnostic_fixtures_cover_levels_four_through_nine() {
        let fixed = cpu_trade_trace(&fixed_same_bar_fixture()).unwrap();
        assert_eq!(fixed.entries.len(), 2);
        assert_eq!(fixed.exits[0].reason, EXIT_STOP);
        assert_eq!(fixed.exits[1].reason, EXIT_STOP);

        let adaptive = cpu_trade_trace(&adaptive_cost_sizing_fixture()).unwrap();
        assert_eq!(adaptive.exits[0].reason, EXIT_TARGET);
        let adaptive_trade = &adaptive.accepted_trades[0];
        assert!((adaptive_trade.size - 16.666_666).abs() < 1.0e-4);
        assert!((adaptive_trade.costs - 475.166_66).abs() < 1.0e-2);
        assert!((adaptive_trade.pnl - 1_691.5).abs() < 1.0e-1);

        let trailing = cpu_trade_trace(&trailing_calendar_ftmo_fixture()).unwrap();
        assert_eq!(trailing.accepted_trades[0].exit_reason, EXIT_STOP);
        assert_eq!(trailing.accepted_trades[0].pnl, 0.0);
        let first_bar = trailing
            .risk_states
            .iter()
            .find(|state| state.bar == 1)
            .unwrap();
        assert_eq!(first_bar.flags & (RISK_NEW_DAY | RISK_NEW_MONTH), 3);
        let day_rollover = trailing
            .risk_states
            .iter()
            .find(|state| state.bar == 3)
            .unwrap();
        assert_eq!(day_rollover.flags & RISK_NEW_DAY, RISK_NEW_DAY);
        let month_rollover = trailing
            .risk_states
            .iter()
            .find(|state| state.bar == 4)
            .unwrap();
        assert_eq!(month_rollover.flags & RISK_NEW_MONTH, RISK_NEW_MONTH);
        let loss_finalized = trailing
            .risk_states
            .iter()
            .find(|state| state.bar == 6)
            .unwrap();
        assert_eq!(
            loss_finalized.flags & (RISK_DAILY_LOSS_BREACH | RISK_TOTAL_DRAWDOWN_BREACH),
            RISK_DAILY_LOSS_BREACH | RISK_TOTAL_DRAWDOWN_BREACH
        );

        let profit = cpu_trade_trace(&ftmo_profit_target_fixture()).unwrap();
        assert_eq!(profit.accepted_trades[0].pnl, 200.0);
        assert!(profit.risk_states.iter().any(|state| {
            state.flags & RISK_PROFIT_TARGET_REACHED == RISK_PROFIT_TARGET_REACHED
        }));
        assert!(
            compare_traces("cpu", "cpu", &trailing, &trailing, diagnostic_policy(),).is_match()
        );
    }

    #[cfg(any(feature = "gpu-cuda", feature = "gpu-vulkan"))]
    #[test]
    fn direct_trade_trace_levels_four_through_nine_match_cpu() {
        for input in [
            fixed_same_bar_fixture(),
            adaptive_cost_sizing_fixture(),
            trailing_calendar_ftmo_fixture(),
            ftmo_profit_target_fixture(),
        ] {
            let expected = cpu_trade_trace(&input).unwrap();
            let actual = match gpu_trade_trace(&input) {
                Ok(actual) => actual,
                Err(crate::gpu_native::engine::EngineError::UnsupportedCapability {
                    operation: "diagnostic_gpu_adapter",
                    detail,
                }) => {
                    eprintln!("trade trace GPU test skipped: {detail}");
                    return;
                }
                Err(error) => panic!("trade trace GPU execution failed: {error}"),
            };
            let report = compare_traces(
                "cpu_diagnostic",
                "gpu_diagnostic",
                &expected,
                &actual,
                diagnostic_policy(),
            );
            assert!(
                report.is_match(),
                "causal trade trace diverged: {:?}",
                report.first_divergence
            );
        }
    }
}
