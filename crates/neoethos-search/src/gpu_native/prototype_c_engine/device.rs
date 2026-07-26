//! Device-resident CubeCL kernels and engine for Prototype C.
//!
//! This module is compiled only for builds that bind a CubeCL runtime. The
//! host-side projection and result widening live in the parent module and are
//! compiled everywhere.

use super::{
    C_EXIT_GAP, C_EXIT_MAX_HOLD, C_EXIT_NONE, C_EXIT_STOP, C_EXIT_TARGET, C_METRIC_WIDTH,
    CPopulationHostBuffers, SMC_WIDTH, survivor_summary_from_device_metrics,
};
use crate::gpu_native::engine::{
    BacktestEngine, DatasetHandle, DeviceEventHandle, DeviceFilterPolicy, DeviceMetricsHandle,
    DeviceSelectionHandle, EngineCapabilities, EngineError, EngineIdentity, EngineStatus,
    GeneBufferHandle, GpuDiscoverySession, HostSurvivorSummary, ScenarioBufferHandle,
    SynchronizationMode, TypedDeviceHandle,
};
use crate::gpu_native::prototype_a::{
    PrototypeADatasetUpload, PrototypeAGeneUpload, PrototypeAScenarioUpload,
};
use crate::gpu_native::prototype_b_engine::validate_population_eligibility;
use crate::gpu_native::prototype_bc::{PrototypeKind, prototype_c_capabilities};
use cubecl::prelude::*;

#[cfg(feature = "gpu-cuda")]
pub type PrototypeCActiveRuntime = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "gpu-vulkan", not(feature = "gpu-cuda")))]
pub type PrototypeCActiveRuntime = cubecl::wgpu::WgpuRuntime;

#[cfg(feature = "gpu-cuda")]
const PROTOTYPE_C_BACKEND_ID: u32 = 21;
#[cfg(all(feature = "gpu-vulkan", not(feature = "gpu-cuda")))]
const PROTOTYPE_C_BACKEND_ID: u32 = 22;

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

/// Candidate x bar signal synthesis. Terms accumulate in ascending CSR order so
/// the device reproduces the canonical `f32` accumulation order.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn c_population_signals_kernel(
    indicators: &Array<f32>,
    gene_offsets: &Array<i32>,
    gene_indices: &Array<i32>,
    gene_weights: &Array<f32>,
    long_thresholds: &Array<f32>,
    short_thresholds: &Array<f32>,
    smc_rows: &Array<i32>,
    smc_flags: &Array<i32>,
    smc_weights: &Array<f32>,
    signals_out: &mut Array<i32>,
    confidences_out: &mut Array<f32>,
    n_bars: u32,
    population: u32,
    gate_threshold: f32,
    smc_gate_disabled: u32,
) {
    let bars = n_bars as usize;
    let total = population as usize * bars;
    if ABSOLUTE_POS < total {
        let flat = ABSOLUTE_POS;
        let candidate = flat / bars;
        let bar = flat - candidate * bars;

        let combined = RuntimeCell::<f32>::new(0.0);
        let start = gene_offsets[candidate] as usize;
        let end = gene_offsets[candidate + 1] as usize;
        for term in start..end {
            let feature = gene_indices[term] as usize;
            combined.store(combined.read() + gene_weights[term] * indicators[feature * bars + bar]);
        }

        let long_threshold = long_thresholds[candidate];
        let short_threshold = short_thresholds[candidate];
        let value = combined.read();
        let signal = RuntimeCell::<i32>::new(0);
        if value >= long_threshold {
            signal.store(1);
        } else if value <= short_threshold {
            signal.store(-1);
        }

        let emitted = RuntimeCell::<i32>::new(0);
        let confidence = RuntimeCell::<f32>::new(0.0);
        if signal.read() != 0 {
            let raw_gap = f32::abs(long_threshold - short_threshold);
            let gap = RuntimeCell::<f32>::new(1.0e-6);
            if raw_gap > 1.0e-6 {
                gap.store(raw_gap);
            }
            let margin = RuntimeCell::<f32>::new(short_threshold - value);
            if signal.read() == 1 {
                margin.store(value - long_threshold);
            }
            let scaled = margin.read() / gap.read();
            let clamped = RuntimeCell::<f32>::new(scaled);
            if scaled < 0.0 {
                clamped.store(0.0);
            } else if scaled > 1.0 {
                clamped.store(1.0);
            }
            confidence.store(clamped.read());

            let active_sum = RuntimeCell::<f32>::new(0.0);
            for slot in 0..SMC_WIDTH {
                if smc_flags[candidate * SMC_WIDTH + slot] != 0 {
                    active_sum.store(active_sum.read() + smc_weights[slot]);
                }
            }
            if smc_gate_disabled != 0 {
                active_sum.store(0.0);
            }
            let gate = RuntimeCell::<f32>::new(active_sum.read());
            if gate_threshold < active_sum.read() {
                gate.store(gate_threshold);
            }

            let passes = RuntimeCell::<i32>::new(1);
            if active_sum.read() > 0.0 {
                let score = RuntimeCell::<f32>::new(0.0);
                for slot in 0..SMC_WIDTH {
                    if smc_flags[candidate * SMC_WIDTH + slot] != 0 {
                        let row = smc_rows[bar * SMC_WIDTH + slot];
                        if slot == 5 {
                            if row == 1 {
                                score.store(score.read() + smc_weights[slot]);
                            }
                        } else if row == signal.read() {
                            score.store(score.read() + smc_weights[slot]);
                        }
                    }
                }
                if score.read() < gate.read() {
                    passes.store(0);
                }
            }
            if passes.read() != 0 {
                emitted.store(signal.read());
            } else {
                confidence.store(0.0);
            }
        }

        signals_out[flat] = emitted.read();
        confidences_out[flat] = confidence.read();
    }
}

/// Per-candidate causal entry count. One unit per candidate keeps the count and
/// the later emission in the same canonical bar order.
#[cube(launch)]
fn c_population_count_events_kernel(
    signals: &Array<i32>,
    counts_out: &mut Array<i32>,
    n_bars: u32,
    population: u32,
) {
    if ABSOLUTE_POS < population as usize {
        let candidate = ABSOLUTE_POS;
        let bars = n_bars as usize;
        let base = candidate * bars;
        let count = RuntimeCell::<i32>::new(0);
        for bar in 1..bars {
            if signals[base + bar - 1] != 0 {
                count.store(count.read() + 1);
            }
        }
        counts_out[candidate] = count.read();
    }
}

/// Deterministic exclusive scan of the per-candidate counts. The population axis
/// is small relative to the bar axis, so one sequential device pass is both
/// exact and cheap. It never truncates: the total lands in `offsets_out[P]`.
#[cube(launch)]
fn c_population_scan_offsets_kernel(
    counts: &Array<i32>,
    offsets_out: &mut Array<i32>,
    total_out: &mut Array<i32>,
    population: u32,
) {
    if ABSOLUTE_POS < 1 {
        let running = RuntimeCell::<i32>::new(0);
        for candidate in 0..population as usize {
            offsets_out[candidate] = running.read();
            running.store(running.read() + counts[candidate]);
        }
        offsets_out[population as usize] = running.read();
        // One control scalar: the host reads this to size the sparse passes and
        // to enforce the event capacity. The candidate-indexed offsets stay
        // resident and are never read back.
        total_out[0] = running.read();
    }
}

/// Causal entry emission in candidate-major, bar-ascending order.
///
/// Same-bar precedence is stop-first for every Stage-1 event, so it is a kernel
/// invariant rather than a per-event field.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn emit_c_population_events(
    close_pips: &Array<f32>,
    adaptive_base_pips: &Array<f32>,
    stop_pips: &Array<f32>,
    target_pips: &Array<f32>,
    stop_vol_multipliers: &Array<f32>,
    signals: &Array<i32>,
    event_offsets: &Array<i32>,
    event_candidate_out: &mut Array<i32>,
    event_scenario_out: &mut Array<i32>,
    event_entry_out: &mut Array<i32>,
    event_last_out: &mut Array<i32>,
    event_direction_out: &mut Array<i32>,
    event_stop_out: &mut Array<f32>,
    event_target_out: &mut Array<f32>,
    n_bars: u32,
    population: u32,
    max_hold_bars: u32,
    min_hold_bars: u32,
    half_spread_pips: f32,
    adaptive_rr: f32,
    has_adaptive_base: u32,
) {
    if ABSOLUTE_POS < population as usize {
        let candidate = ABSOLUTE_POS;
        let bars = n_bars as usize;
        let last_dataset_bar = bars - 1;
        let signal_base = candidate * bars;
        let write = RuntimeCell::<i32>::new(event_offsets[candidate]);
        let multiplier = stop_vol_multipliers[candidate];

        for bar in 1..bars {
            let direction = signals[signal_base + bar - 1];
            if direction != 0 {
                let slot = write.read() as usize;
                let signal_bar = bar - 1;
                let entry_pips = close_pips[bar] + direction as f32 * half_spread_pips;

                let stop_distance = RuntimeCell::<f32>::new(stop_pips[candidate]);
                let target_distance = RuntimeCell::<f32>::new(target_pips[candidate]);
                if multiplier > 0.0 && has_adaptive_base != 0 {
                    let adaptive_stop = multiplier * adaptive_base_pips[signal_bar];
                    let adaptive_target = adaptive_rr * adaptive_stop;
                    if adaptive_stop > 0.0 && adaptive_target > 0.0 {
                        stop_distance.store(adaptive_stop);
                        target_distance.store(adaptive_target);
                    }
                }

                let stop_level = RuntimeCell::<f32>::new(entry_pips + stop_distance.read());
                let target_level = RuntimeCell::<f32>::new(entry_pips - target_distance.read());
                if direction > 0 {
                    stop_level.store(entry_pips - stop_distance.read());
                    target_level.store(entry_pips + target_distance.read());
                }

                let last_bar = RuntimeCell::<i32>::new(last_dataset_bar as i32);
                if max_hold_bars > 0 {
                    let hold = RuntimeCell::<i32>::new(min_hold_bars as i32);
                    if max_hold_bars > min_hold_bars {
                        hold.store(max_hold_bars as i32);
                    }
                    let scheduled = bar as i32 + hold.read();
                    if scheduled < last_dataset_bar as i32 {
                        last_bar.store(scheduled);
                    }
                }

                event_candidate_out[slot] = candidate as i32;
                event_scenario_out[slot] = candidate as i32;
                event_entry_out[slot] = bar as i32;
                event_last_out[slot] = last_bar.read();
                event_direction_out[slot] = direction;
                event_stop_out[slot] = stop_level.read();
                event_target_out[slot] = target_level.read();
                write.store(write.read() + 1);
            }
        }
    }
}

/// Compact first-hit search: one unit per event.
///
/// Within a bar the canonical resolver drains gaps first, then stop, then
/// target, then the max-hold sweep, so the earliest bar with any reason wins and
/// the priority above breaks ties on the same bar.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn c_population_first_hit_kernel(
    high_pips: &Array<f32>,
    low_pips: &Array<f32>,
    gap_flags: &Array<i32>,
    event_entry: &Array<i32>,
    event_last: &Array<i32>,
    event_direction: &Array<i32>,
    event_stop: &Array<f32>,
    event_target: &Array<f32>,
    exit_bar_out: &mut Array<i32>,
    exit_reason_out: &mut Array<i32>,
    event_count: u32,
    n_bars: u32,
    min_hold_bars: u32,
    max_hold_bars: u32,
) {
    if ABSOLUTE_POS < event_count as usize {
        let event = ABSOLUTE_POS;
        exit_bar_out[event] = -1;
        exit_reason_out[event] = C_EXIT_NONE;

        let entry_bar = event_entry[event];
        let raw_last = event_last[event];
        let last_bar = RuntimeCell::<i32>::new(raw_last);
        if raw_last > n_bars as i32 - 1 {
            last_bar.store(n_bars as i32 - 1);
        }

        if entry_bar < last_bar.read() {
            let hold_min = RuntimeCell::<i32>::new(1);
            if min_hold_bars > 1 {
                hold_min.store(min_hold_bars as i32);
            }
            let level_activation = entry_bar + hold_min.read();

            let max_hold_exit = RuntimeCell::<i32>::new(-1);
            if max_hold_bars > 0 {
                let hold = RuntimeCell::<i32>::new(min_hold_bars as i32);
                if max_hold_bars > min_hold_bars {
                    hold.store(max_hold_bars as i32);
                }
                let scheduled = entry_bar + hold.read();
                if scheduled <= last_bar.read() {
                    max_hold_exit.store(scheduled);
                }
            }

            let direction = event_direction[event];
            let stop_level = event_stop[event];
            let target_level = event_target[event];
            let found_bar = RuntimeCell::<i32>::new(-1);
            let found_reason = RuntimeCell::<i32>::new(C_EXIT_NONE);

            let scan_start = (entry_bar + 1) as usize;
            let scan_end = (last_bar.read() + 1) as usize;
            for bar in scan_start..scan_end {
                if found_bar.read() < 0 {
                    let reason = RuntimeCell::<i32>::new(C_EXIT_NONE);
                    if gap_flags[bar] != 0 {
                        reason.store(C_EXIT_GAP);
                    } else {
                        let stop_hit = RuntimeCell::<i32>::new(0);
                        let target_hit = RuntimeCell::<i32>::new(0);
                        if bar as i32 >= level_activation {
                            let high = high_pips[bar];
                            let low = low_pips[bar];
                            if direction > 0 {
                                if low <= stop_level {
                                    stop_hit.store(1);
                                }
                                if high >= target_level {
                                    target_hit.store(1);
                                }
                            } else {
                                if high >= stop_level {
                                    stop_hit.store(1);
                                }
                                if low <= target_level {
                                    target_hit.store(1);
                                }
                            }
                        }
                        if stop_hit.read() != 0 {
                            reason.store(C_EXIT_STOP);
                        } else if target_hit.read() != 0 {
                            reason.store(C_EXIT_TARGET);
                        } else if max_hold_exit.read() == bar as i32 {
                            reason.store(C_EXIT_MAX_HOLD);
                        }
                    }
                    if reason.read() != C_EXIT_NONE {
                        found_bar.store(bar as i32);
                        found_reason.store(reason.read());
                    }
                }
            }

            exit_bar_out[event] = found_bar.read();
            exit_reason_out[event] = found_reason.read();
        }
    }
}

/// Deterministic non-overlapping trade stitching.
///
/// Acceptance depends only on position state, the daily trade cap and the
/// calendar, never on equity, so it is a separate device pass whose output the
/// metric reduction consumes verbatim.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn stitch_c_population_trades(
    day_idx: &Array<i32>,
    event_offsets: &Array<i32>,
    event_entry: &Array<i32>,
    exit_bar: &Array<i32>,
    exit_reason: &Array<i32>,
    accepted_event_out: &mut Array<i32>,
    accepted_count_out: &mut Array<i32>,
    n_bars: u32,
    population: u32,
    max_trades_per_day: u32,
) {
    if ABSOLUTE_POS < population as usize {
        let candidate = ABSOLUTE_POS;
        let range_start = event_offsets[candidate] as usize;
        let range_end = event_offsets[candidate + 1] as usize;
        // An open position with no exit inside its window blocks every later
        // entry, exactly as the canonical walk does.
        let never_exits = n_bars as i32 + 1;

        let accepted = RuntimeCell::<i32>::new(0);
        let last_day = RuntimeCell::<i32>::new(-1);
        let day_trade_count = RuntimeCell::<u32>::new(0);
        let open_exit_bar = RuntimeCell::<i32>::new(-1);
        let open_exit_reason = RuntimeCell::<i32>::new(C_EXIT_NONE);

        // One pass over this candidate's events in canonical order. Iterating
        // events rather than bars is equivalent because acceptance can only
        // change at a bar that carries an event, and it keeps the loop bounds
        // free of data-dependent cursor advancement.
        for event in range_start..range_end {
            let entry = event_entry[event];

            if day_idx[entry as usize] != last_day.read() {
                last_day.store(day_idx[entry as usize]);
                day_trade_count.store(0);
            }

            let blocked = RuntimeCell::<i32>::new(0);
            if entry < open_exit_bar.read() {
                blocked.store(1);
            } else if entry == open_exit_bar.read() && open_exit_reason.read() != C_EXIT_GAP {
                // A regular stop/target/max-hold exit consumes the bar; the
                // canonical walk cannot re-enter on it.
                blocked.store(1);
            }

            if blocked.read() == 0
                && (max_trades_per_day == 0 || day_trade_count.read() < max_trades_per_day)
            {
                let exit = RuntimeCell::<i32>::new(never_exits);
                if exit_bar[event] >= 0 {
                    exit.store(exit_bar[event]);
                }
                open_exit_bar.store(exit.read());
                open_exit_reason.store(exit_reason[event]);
                day_trade_count.store(day_trade_count.read() + 1);
                accepted_event_out[range_start + accepted.read() as usize] = event as i32;
                accepted.store(accepted.read() + 1);
            }
        }

        accepted_count_out[candidate] = accepted.read();
    }
}

/// Exact cost, sizing, equity, calendar and metric reduction.
///
/// `month_workspace` holds the completed monthly P&L in its first half and the
/// month starting equity in its second half, so one buffer carries both.
/// `timestamp_pair` holds `[day_since_epoch, ms_of_day]` per bar, which keeps
/// the overnight carry term exact without an `f64` device type.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn reduce_c_population_metrics(
    close_pips: &Array<f32>,
    high_pips: &Array<f32>,
    low_pips: &Array<f32>,
    month_idx: &Array<i32>,
    day_idx: &Array<i32>,
    timestamp_pair: &Array<i32>,
    confidences: &Array<f32>,
    event_entry: &Array<i32>,
    event_direction: &Array<i32>,
    event_stop: &Array<f32>,
    event_target: &Array<f32>,
    exit_bar: &Array<i32>,
    exit_reason: &Array<i32>,
    accepted_event: &Array<i32>,
    accepted_count: &Array<i32>,
    event_offsets: &Array<i32>,
    month_workspace: &mut Array<f32>,
    metrics_out: &mut Array<f32>,
    n_bars: u32,
    population: u32,
    month_capacity: u32,
    initial_equity: f32,
    half_spread_pips: f32,
    commission_per_trade: f32,
    pip_value_per_lot: f32,
    swap_long_pips_per_day: f32,
    swap_short_pips_per_day: f32,
    pnl_conversion_fee_rate: f32,
    risk_based_sizing: u32,
    risk_per_trade_min: f32,
    risk_per_trade_max: f32,
    high_quality_confidence: f32,
) {
    if ABSOLUTE_POS < population as usize {
        let candidate = ABSOLUTE_POS;
        let bars = n_bars as usize;
        let capacity = month_capacity as usize;
        let month_base = candidate * capacity;
        let start_base = population as usize * capacity + month_base;
        for slot in 0..capacity {
            month_workspace[month_base + slot] = 0.0;
            month_workspace[start_base + slot] = initial_equity;
        }

        let half_spread_cost = half_spread_pips * pip_value_per_lot;
        let confidence_base = candidate * bars;
        let range_start = event_offsets[candidate] as usize;
        let accepted_total = accepted_count[candidate];

        let equity = RuntimeCell::<f32>::new(initial_equity);
        let peak_equity = RuntimeCell::<f32>::new(initial_equity);
        let max_drawdown = RuntimeCell::<f32>::new(0.0);
        let trade_count = RuntimeCell::<i32>::new(0);
        let wins = RuntimeCell::<i32>::new(0);
        let gross_profit = RuntimeCell::<f32>::new(0.0);
        let gross_loss = RuntimeCell::<f32>::new(0.0);

        let last_month = RuntimeCell::<i32>::new(-1);
        let current_month_pnl = RuntimeCell::<f32>::new(0.0);
        let current_month_start_equity = RuntimeCell::<f32>::new(initial_equity);
        let month_ptr = RuntimeCell::<i32>::new(-1);

        let last_day = RuntimeCell::<i32>::new(-1);
        let day_peak = RuntimeCell::<f32>::new(initial_equity);
        let day_low = RuntimeCell::<f32>::new(initial_equity);
        let max_daily_drawdown = RuntimeCell::<f32>::new(0.0);

        let accepted_cursor = RuntimeCell::<i32>::new(0);
        let has_position = RuntimeCell::<i32>::new(0);
        let position_entry_pips = RuntimeCell::<f32>::new(0.0);
        let position_lots = RuntimeCell::<f32>::new(0.0);
        let position_direction = RuntimeCell::<i32>::new(0);
        let position_stop = RuntimeCell::<f32>::new(0.0);
        let position_target = RuntimeCell::<f32>::new(0.0);
        let position_exit_bar = RuntimeCell::<i32>::new(-1);
        let position_exit_reason = RuntimeCell::<i32>::new(C_EXIT_NONE);
        let position_entry_bar = RuntimeCell::<i32>::new(-1);

        for bar in 1..bars {
            let month = month_idx[bar];
            if month != last_month.read() {
                if last_month.read() != -1 {
                    let next_ptr = month_ptr.read() + 1;
                    month_ptr.store(next_ptr);
                    if next_ptr >= 0 && next_ptr < capacity as i32 {
                        month_workspace[month_base + next_ptr as usize] = current_month_pnl.read();
                        month_workspace[start_base + next_ptr as usize] =
                            current_month_start_equity.read();
                    }
                }
                current_month_pnl.store(0.0);
                current_month_start_equity.store(equity.read());
                last_month.store(month);
            }

            let day = day_idx[bar];
            if day != last_day.read() {
                if last_day.read() != -1 && day_peak.read() > 0.0 {
                    let drawdown = (day_peak.read() - day_low.read()) / day_peak.read();
                    if drawdown > max_daily_drawdown.read() {
                        max_daily_drawdown.store(drawdown);
                    }
                }
                last_day.store(day);
                day_peak.store(equity.read());
                day_low.store(equity.read());
            }

            let flat_after_gap = RuntimeCell::<i32>::new(0);
            let blocked = RuntimeCell::<i32>::new(0);
            if has_position.read() != 0 {
                let realize = RuntimeCell::<i32>::new(0);
                if position_exit_bar.read() == bar as i32
                    && position_exit_reason.read() == C_EXIT_GAP
                {
                    realize.store(1);
                    flat_after_gap.store(1);
                } else {
                    let low = low_pips[bar];
                    let high = high_pips[bar];
                    let worst = RuntimeCell::<f32>::new(0.0);
                    let best = RuntimeCell::<f32>::new(0.0);
                    if position_direction.read() > 0 {
                        worst.store((low - position_entry_pips.read()) * pip_value_per_lot);
                        best.store((high - position_entry_pips.read()) * pip_value_per_lot);
                    } else {
                        worst.store((position_entry_pips.read() - high) * pip_value_per_lot);
                        best.store((position_entry_pips.read() - low) * pip_value_per_lot);
                    }
                    let worst_pnl = worst.read() * position_lots.read();
                    let best_pnl = best.read() * position_lots.read();

                    if equity.read() + best_pnl > peak_equity.read() {
                        peak_equity.store(equity.read() + best_pnl);
                    }
                    if equity.read() + best_pnl > day_peak.read() {
                        if day_peak.read() > 0.0 {
                            let drawdown = (day_peak.read() - day_low.read()) / day_peak.read();
                            if drawdown > max_daily_drawdown.read() {
                                max_daily_drawdown.store(drawdown);
                            }
                        }
                        day_peak.store(equity.read() + best_pnl);
                        day_low.store(equity.read() + worst_pnl);
                    } else if equity.read() + worst_pnl < day_low.read() {
                        day_low.store(equity.read() + worst_pnl);
                    }
                    if peak_equity.read() > 0.0 {
                        let drawdown =
                            (peak_equity.read() - (equity.read() + worst_pnl)) / peak_equity.read();
                        if drawdown > max_drawdown.read() {
                            max_drawdown.store(drawdown);
                        }
                    }

                    if position_exit_bar.read() == bar as i32
                        && position_exit_reason.read() != C_EXIT_NONE
                    {
                        realize.store(1);
                    }
                    blocked.store(1);
                }

                if realize.read() != 0 {
                    let exit_level = RuntimeCell::<f32>::new(close_pips[bar]);
                    if position_exit_reason.read() == C_EXIT_STOP {
                        exit_level.store(position_stop.read());
                    } else if position_exit_reason.read() == C_EXIT_TARGET {
                        exit_level.store(position_target.read());
                    }
                    let price_pnl = RuntimeCell::<f32>::new(
                        (position_entry_pips.read() - exit_level.read()) * pip_value_per_lot,
                    );
                    if position_direction.read() > 0 {
                        price_pnl.store(
                            (exit_level.read() - position_entry_pips.read()) * pip_value_per_lot,
                        );
                    }
                    let gross_scaled = price_pnl.read() * position_lots.read()
                        - (commission_per_trade + half_spread_cost) * position_lots.read();

                    let entry_bar_index = position_entry_bar.read() as usize;
                    let entry_days = timestamp_pair[entry_bar_index * 2];
                    let entry_ms = timestamp_pair[entry_bar_index * 2 + 1];
                    let exit_days = timestamp_pair[bar * 2];
                    let exit_ms = timestamp_pair[bar * 2 + 1];
                    let overnight_days = RuntimeCell::<f32>::new(0.0);
                    if exit_days > entry_days || (exit_days == entry_days && exit_ms > entry_ms) {
                        let day_part = (exit_days - entry_days) as f32;
                        let ms_part = (exit_ms - entry_ms) as f32 / 86_400_000.0;
                        overnight_days.store(day_part + ms_part);
                    }
                    let swap_pips = RuntimeCell::<f32>::new(swap_short_pips_per_day);
                    if position_direction.read() > 0 {
                        swap_pips.store(swap_long_pips_per_day);
                    }
                    let with_carry = gross_scaled
                        + swap_pips.read()
                            * overnight_days.read()
                            * pip_value_per_lot
                            * position_lots.read();
                    let pnl = RuntimeCell::<f32>::new(with_carry);
                    if pnl_conversion_fee_rate > 0.0 && pnl_conversion_fee_rate < 1.0 {
                        pnl.store(with_carry * (1.0 - pnl_conversion_fee_rate));
                    }

                    equity.store(equity.read() + pnl.read());
                    current_month_pnl.store(current_month_pnl.read() + pnl.read());
                    trade_count.store(trade_count.read() + 1);
                    if pnl.read() > 0.0 {
                        wins.store(wins.read() + 1);
                        gross_profit.store(gross_profit.read() + pnl.read());
                    } else {
                        gross_loss.store(gross_loss.read() + f32::abs(pnl.read()));
                    }

                    if equity.read() > peak_equity.read() {
                        peak_equity.store(equity.read());
                    }
                    if equity.read() > day_peak.read() {
                        if day_peak.read() > 0.0 {
                            let drawdown = (day_peak.read() - day_low.read()) / day_peak.read();
                            if drawdown > max_daily_drawdown.read() {
                                max_daily_drawdown.store(drawdown);
                            }
                        }
                        day_peak.store(equity.read());
                        day_low.store(equity.read());
                    } else if equity.read() < day_low.read() {
                        day_low.store(equity.read());
                    }
                    if peak_equity.read() > 0.0 {
                        let drawdown = (peak_equity.read() - equity.read()) / peak_equity.read();
                        if drawdown > max_drawdown.read() {
                            max_drawdown.store(drawdown);
                        }
                    }
                    has_position.store(0);
                }
            }

            if blocked.read() == 0 || flat_after_gap.read() != 0 {
                if accepted_cursor.read() < accepted_total {
                    let event =
                        accepted_event[range_start + accepted_cursor.read() as usize] as usize;
                    if event_entry[event] == bar as i32 {
                        accepted_cursor.store(accepted_cursor.read() + 1);
                        let direction = event_direction[event];
                        let entry_pips = close_pips[bar] + direction as f32 * half_spread_pips;
                        let stop_level = event_stop[event];
                        let stop_distance = f32::abs(stop_level - entry_pips);
                        let lots = RuntimeCell::<f32>::new(1.0);
                        if risk_based_sizing != 0 {
                            let confidence = confidences[confidence_base + bar - 1];
                            let clamped = RuntimeCell::<f32>::new(confidence);
                            if confidence < 0.0 {
                                clamped.store(0.0);
                            } else if confidence > 1.0 {
                                clamped.store(1.0);
                            }
                            let scale = RuntimeCell::<f32>::new(1.0);
                            if high_quality_confidence > 0.0 {
                                let ratio = clamped.read() / high_quality_confidence;
                                if ratio < 1.0 {
                                    scale.store(ratio);
                                }
                            }
                            let risk = risk_per_trade_min
                                + (risk_per_trade_max - risk_per_trade_min) * scale.read();
                            let guarded_stop = RuntimeCell::<f32>::new(1.0);
                            if stop_distance > 1.0 {
                                guarded_stop.store(stop_distance);
                            }
                            let denominator = guarded_stop.read() * pip_value_per_lot;
                            let sized = RuntimeCell::<f32>::new(0.0);
                            if equity.read() > 0.0 && f32::abs(denominator) > 1.0e-12 {
                                sized.store(risk * equity.read() / denominator);
                            }
                            if sized.read() < 0.0 {
                                sized.store(0.0);
                            } else if sized.read() > 100.0 {
                                sized.store(100.0);
                            }
                            lots.store(sized.read());
                        }

                        has_position.store(1);
                        position_entry_pips.store(entry_pips);
                        position_lots.store(lots.read());
                        position_direction.store(direction);
                        position_stop.store(stop_level);
                        position_target.store(event_target[event]);
                        position_exit_bar.store(exit_bar[event]);
                        position_exit_reason.store(exit_reason[event]);
                        position_entry_bar.store(bar as i32);
                    }
                }
            }
        }

        if last_day.read() != -1 && day_peak.read() > 0.0 {
            let drawdown = (day_peak.read() - day_low.read()) / day_peak.read();
            if drawdown > max_daily_drawdown.read() {
                max_daily_drawdown.store(drawdown);
            }
        }

        let net_profit = equity.read() - initial_equity;
        let win_rate = RuntimeCell::<f32>::new(0.0);
        let expectancy = RuntimeCell::<f32>::new(0.0);
        if trade_count.read() > 0 {
            win_rate.store(wins.read() as f32 / trade_count.read() as f32);
            expectancy.store(net_profit / trade_count.read() as f32);
        }
        let profit_factor = RuntimeCell::<f32>::new(0.0);
        if gross_loss.read() > 0.0 {
            profit_factor.store(gross_profit.read() / gross_loss.read());
        } else if gross_profit.read() > 0.0 {
            profit_factor.store(10.0);
        }

        let limit = RuntimeCell::<i32>::new(-1);
        if month_ptr.read() >= 0 && capacity > 0 {
            let cap_limit = capacity as i32 - 1;
            if month_ptr.read() < cap_limit {
                limit.store(month_ptr.read());
            } else {
                limit.store(cap_limit);
            }
        }

        let monthly_mean = RuntimeCell::<f32>::new(0.0);
        let monthly_std = RuntimeCell::<f32>::new(0.0);
        if limit.read() >= 1 {
            let count = limit.read() + 1;
            let sum = RuntimeCell::<f32>::new(0.0);
            for index in 0..count as usize {
                sum.store(sum.read() + month_workspace[month_base + index]);
            }
            let mean = sum.read() / count as f32;
            let variance = RuntimeCell::<f32>::new(0.0);
            for index in 0..count as usize {
                let delta = month_workspace[month_base + index] - mean;
                variance.store(variance.read() + delta * delta);
            }
            monthly_mean.store(mean);
            monthly_std.store(f32::sqrt(variance.read() / (count - 1) as f32));
        }

        let sharpe = RuntimeCell::<f32>::new(0.0);
        let consistency = RuntimeCell::<f32>::new(0.0);
        if monthly_std.read() > 0.0 {
            let ratio = monthly_mean.read() / monthly_std.read();
            sharpe.store(ratio * 3.4641);
            let clamped = RuntimeCell::<f32>::new(ratio);
            if ratio < 0.0 {
                clamped.store(0.0);
            } else if ratio > 1.0 {
                clamped.store(1.0);
            }
            consistency.store(clamped.read());
        } else if monthly_mean.read() > 0.0 && limit.read() < 1 {
            consistency.store(1.0);
        }

        let monthly_target_hit_rate = RuntimeCell::<f32>::new(0.0);
        if limit.read() >= 0 {
            let hits = RuntimeCell::<i32>::new(0);
            let counted = RuntimeCell::<i32>::new(0);
            for index in 0..(limit.read() + 1) as usize {
                let base_equity = month_workspace[start_base + index];
                if base_equity > 0.0 {
                    counted.store(counted.read() + 1);
                    if month_workspace[month_base + index] / base_equity >= 0.04 {
                        hits.store(hits.read() + 1);
                    }
                }
            }
            if counted.read() > 0 {
                monthly_target_hit_rate.store(hits.read() as f32 / counted.read() as f32);
            }
        }

        let metric_base = candidate * C_METRIC_WIDTH;
        metrics_out[metric_base] = net_profit;
        metrics_out[metric_base + 1] = sharpe.read();
        metrics_out[metric_base + 2] = peak_equity.read();
        metrics_out[metric_base + 3] = max_drawdown.read();
        metrics_out[metric_base + 4] = win_rate.read();
        metrics_out[metric_base + 5] = profit_factor.read();
        metrics_out[metric_base + 6] = expectancy.read();
        metrics_out[metric_base + 7] = monthly_target_hit_rate.read();
        metrics_out[metric_base + 8] = trade_count.read() as f32;
        metrics_out[metric_base + 9] = consistency.read();
        metrics_out[metric_base + 10] = max_daily_drawdown.read();
    }
}

// ---------------------------------------------------------------------------
// Device-resident session
// ---------------------------------------------------------------------------

struct DatasetResident {
    handle: DatasetHandle,
    upload: PrototypeADatasetUpload,
    close: cubecl::server::Handle,
    high: cubecl::server::Handle,
    low: cubecl::server::Handle,
    indicators: cubecl::server::Handle,
    months: cubecl::server::Handle,
    days: cubecl::server::Handle,
    timestamp_pair: cubecl::server::Handle,
    gap_flags: cubecl::server::Handle,
    smc_rows: cubecl::server::Handle,
    adaptive_base: cubecl::server::Handle,
    adaptive_base_len: usize,
    has_adaptive_base: bool,
    bars: usize,
    feature_count: usize,
    indicator_len: usize,
}

struct GenesResident {
    handle: GeneBufferHandle,
    upload: PrototypeAGeneUpload,
    offsets: cubecl::server::Handle,
    indices: cubecl::server::Handle,
    weights: cubecl::server::Handle,
    long_thresholds: cubecl::server::Handle,
    short_thresholds: cubecl::server::Handle,
    stop_pips: cubecl::server::Handle,
    target_pips: cubecl::server::Handle,
    stop_vol_multipliers: cubecl::server::Handle,
    smc_flags: cubecl::server::Handle,
    smc_weights: cubecl::server::Handle,
    signals: cubecl::server::Handle,
    confidences: cubecl::server::Handle,
    event_counts: cubecl::server::Handle,
    event_offsets: cubecl::server::Handle,
    event_total: cubecl::server::Handle,
    month_workspace: cubecl::server::Handle,
    metrics: cubecl::server::Handle,
    accepted_count: cubecl::server::Handle,
    term_len: usize,
    signal_len: usize,
    month_workspace_len: usize,
    metrics_len: usize,
    month_capacity: usize,
    population: usize,
}

struct EventsResident {
    candidate: cubecl::server::Handle,
    scenario: cubecl::server::Handle,
    entry: cubecl::server::Handle,
    last: cubecl::server::Handle,
    direction: cubecl::server::Handle,
    stop: cubecl::server::Handle,
    target: cubecl::server::Handle,
    exit_bar: cubecl::server::Handle,
    exit_reason: cubecl::server::Handle,
    accepted_event: cubecl::server::Handle,
    capacity: usize,
}

struct ScenarioSlot {
    handle: ScenarioBufferHandle,
    upload: PrototypeAScenarioUpload,
}

struct MetricsSlot {
    handle: DeviceMetricsHandle,
    dataset: DatasetHandle,
    genes: GeneBufferHandle,
    scenarios: ScenarioBufferHandle,
    event: DeviceEventHandle,
    emitted_events: usize,
}

struct SelectionSlot {
    handle: DeviceSelectionHandle,
    metrics: DeviceMetricsHandle,
    event: DeviceEventHandle,
}

pub struct PrototypeCResources<R: Runtime> {
    client: ComputeClient<R>,
    dataset: Option<DatasetResident>,
    genes: Option<GenesResident>,
    events: Option<EventsResident>,
    scenarios: Option<ScenarioSlot>,
    metrics: Option<MetricsSlot>,
    selection: Option<SelectionSlot>,
}

pub struct PrototypeCBacktestEngine<R: Runtime> {
    session: GpuDiscoverySession<PrototypeCResources<R>>,
    max_events: usize,
}

pub fn create_prototype_c_engine(
    device_override: Option<usize>,
    session_id: u64,
    max_events: usize,
) -> Result<PrototypeCBacktestEngine<PrototypeCActiveRuntime>, EngineError> {
    if max_events == 0 {
        return Err(EngineError::Backend(
            "Prototype C max event capacity must be non-zero".into(),
        ));
    }
    let client = crate::cubecl_eval::create_gpu_client(device_override).map_err(|error| {
        let message = error.to_string();
        if crate::gpu_native::prototype_a::is_known_no_adapter_error(&message) {
            EngineError::UnsupportedCapability {
                operation: "prototype_c_gpu_adapter",
                detail: message,
            }
        } else {
            EngineError::Backend(message)
        }
    })?;
    let identity = EngineIdentity {
        session_id,
        backend_id: PROTOTYPE_C_BACKEND_ID,
        device_id: device_override.unwrap_or(0) as u32,
    };
    Ok(PrototypeCBacktestEngine {
        session: GpuDiscoverySession::with_backend(
            identity,
            SynchronizationMode::ExplicitEvents,
            PrototypeCResources {
                client,
                dataset: None,
                genes: None,
                events: None,
                scenarios: None,
                metrics: None,
                selection: None,
            },
        ),
        max_events,
    })
}

fn launch_dims<R: Runtime>(client: &ComputeClient<R>, work_items: usize) -> (CubeCount, CubeDim) {
    let units = client.properties().hardware.max_units_per_cube.clamp(1, 64);
    let cubes = (work_items.max(1) as u32).div_ceil(units);
    (
        CubeCount::Static(cubes.max(1), 1, 1),
        CubeDim::new_1d(units),
    )
}

impl<R: Runtime> PrototypeCBacktestEngine<R> {
    fn validate_current<H: TypedDeviceHandle>(
        &self,
        operation: &'static str,
        actual: H,
        expected: H,
    ) -> Result<(), EngineError> {
        self.session.validate(actual)?;
        if actual.token() != expected.token() {
            return Err(EngineError::UnexpectedHandle {
                operation,
                expected: expected.token(),
                actual: actual.token(),
            });
        }
        Ok(())
    }

    fn validate_wait_event(
        &self,
        operation: &'static str,
        actual: Option<DeviceEventHandle>,
        expected: DeviceEventHandle,
    ) -> Result<(), EngineError> {
        let actual = actual.ok_or_else(|| EngineError::UnexpectedHandle {
            operation,
            expected: expected.token(),
            actual: expected.token(),
        })?;
        self.validate_current(operation, actual, expected)
    }

    /// Emitted event count of the most recent evaluation.
    pub fn emitted_events(&self) -> usize {
        self.session
            .backend()
            .metrics
            .as_ref()
            .map_or(0, |metrics| metrics.emitted_events)
    }

    /// Submit the resident sparse chain and return the emitted event count.
    ///
    /// Exactly one control scalar is read between kernels: the event total the
    /// scan writes. It sizes the sparse passes and enforces the declared event
    /// capacity before a single event is written, so an over-capacity population
    /// is a typed refusal rather than an out-of-range device write.
    fn submit_population_chain(&mut self) -> Result<usize, EngineError> {
        let (bars, population, month_capacity) = {
            let resources = self.session.backend();
            let dataset = resources
                .dataset
                .as_ref()
                .ok_or_else(|| EngineError::Backend("Prototype C dataset is missing".into()))?;
            let genes = resources
                .genes
                .as_ref()
                .ok_or_else(|| EngineError::Backend("Prototype C genes are missing".into()))?;
            (dataset.bars, genes.population, genes.month_capacity)
        };
        let settings = {
            let resources = self.session.backend();
            resources
                .dataset
                .as_ref()
                .expect("dataset validated above")
                .upload
                .settings
                .to_settings()
        };
        let pip = if settings.pip_value.abs() < 1.0e-12 {
            1.0e-12
        } else {
            settings.pip_value
        };
        let _ = pip;
        let half_spread_pips = (settings.spread_pips * 0.5) as f32;

        {
            let resources = self.session.backend();
            let client = &resources.client;
            let dataset = resources.dataset.as_ref().expect("dataset validated above");
            let genes = resources.genes.as_ref().expect("genes validated above");

            let (signal_cubes, signal_dim) = launch_dims(client, genes.signal_len);
            c_population_signals_kernel::launch::<R>(
                client,
                signal_cubes,
                signal_dim,
                unsafe {
                    ArrayArg::from_raw_parts(dataset.indicators.clone(), dataset.indicator_len)
                },
                unsafe { ArrayArg::from_raw_parts(genes.offsets.clone(), population + 1) },
                unsafe { ArrayArg::from_raw_parts(genes.indices.clone(), genes.term_len.max(1)) },
                unsafe { ArrayArg::from_raw_parts(genes.weights.clone(), genes.term_len.max(1)) },
                unsafe { ArrayArg::from_raw_parts(genes.long_thresholds.clone(), population) },
                unsafe { ArrayArg::from_raw_parts(genes.short_thresholds.clone(), population) },
                unsafe { ArrayArg::from_raw_parts(dataset.smc_rows.clone(), bars * SMC_WIDTH) },
                unsafe {
                    ArrayArg::from_raw_parts(genes.smc_flags.clone(), population * SMC_WIDTH)
                },
                unsafe { ArrayArg::from_raw_parts(genes.smc_weights.clone(), SMC_WIDTH) },
                unsafe { ArrayArg::from_raw_parts(genes.signals.clone(), genes.signal_len) },
                unsafe { ArrayArg::from_raw_parts(genes.confidences.clone(), genes.signal_len) },
                bars as u32,
                population as u32,
                genes.upload.gate_threshold,
                u32::from(crate::genetic::smc_gate_disabled()),
            );

            let (count_cubes, count_dim) = launch_dims(client, population);
            c_population_count_events_kernel::launch::<R>(
                client,
                count_cubes,
                count_dim,
                unsafe { ArrayArg::from_raw_parts(genes.signals.clone(), genes.signal_len) },
                unsafe { ArrayArg::from_raw_parts(genes.event_counts.clone(), population) },
                bars as u32,
                population as u32,
            );

            c_population_scan_offsets_kernel::launch::<R>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                unsafe { ArrayArg::from_raw_parts(genes.event_counts.clone(), population) },
                unsafe { ArrayArg::from_raw_parts(genes.event_offsets.clone(), population + 1) },
                unsafe { ArrayArg::from_raw_parts(genes.event_total.clone(), 1) },
                population as u32,
            );
        }

        let total_bytes = {
            let resources = self.session.backend();
            let genes = resources.genes.as_ref().expect("genes validated above");
            resources
                .client
                .read_one(genes.event_total.clone())
                .map_err(|error| {
                    EngineError::Backend(format!(
                        "Prototype C event total readback failed: {error}"
                    ))
                })?
        };
        let total_values = i32::from_bytes(total_bytes.as_ref());
        let emitted = total_values
            .first()
            .copied()
            .ok_or_else(|| EngineError::Backend("Prototype C event total is empty".into()))?;
        if emitted < 0 {
            return Err(EngineError::Backend(format!(
                "Prototype C emitted a negative event total {emitted}"
            )));
        }
        let emitted = emitted as usize;
        if emitted > self.max_events {
            return Err(EngineError::UnsupportedCapability {
                operation: "event_capacity",
                detail: format!(
                    "the population emits {emitted} causal entries, above the session capacity \
                     of {}; raise the capacity rather than truncating the workload",
                    self.max_events
                ),
            });
        }

        self.ensure_event_capacity(emitted.max(1))?;

        {
            let resources = self.session.backend();
            let client = &resources.client;
            let dataset = resources.dataset.as_ref().expect("dataset validated above");
            let genes = resources.genes.as_ref().expect("genes validated above");
            let events = resources.events.as_ref().expect("events allocated above");
            let capacity = events.capacity;

            let (emit_cubes, emit_dim) = launch_dims(client, population);
            emit_c_population_events::launch::<R>(
                client,
                emit_cubes,
                emit_dim,
                unsafe { ArrayArg::from_raw_parts(dataset.close.clone(), bars) },
                unsafe {
                    ArrayArg::from_raw_parts(
                        dataset.adaptive_base.clone(),
                        dataset.adaptive_base_len,
                    )
                },
                unsafe { ArrayArg::from_raw_parts(genes.stop_pips.clone(), population) },
                unsafe { ArrayArg::from_raw_parts(genes.target_pips.clone(), population) },
                unsafe { ArrayArg::from_raw_parts(genes.stop_vol_multipliers.clone(), population) },
                unsafe { ArrayArg::from_raw_parts(genes.signals.clone(), genes.signal_len) },
                unsafe { ArrayArg::from_raw_parts(genes.event_offsets.clone(), population + 1) },
                unsafe { ArrayArg::from_raw_parts(events.candidate.clone(), capacity) },
                unsafe { ArrayArg::from_raw_parts(events.scenario.clone(), capacity) },
                unsafe { ArrayArg::from_raw_parts(events.entry.clone(), capacity) },
                unsafe { ArrayArg::from_raw_parts(events.last.clone(), capacity) },
                unsafe { ArrayArg::from_raw_parts(events.direction.clone(), capacity) },
                unsafe { ArrayArg::from_raw_parts(events.stop.clone(), capacity) },
                unsafe { ArrayArg::from_raw_parts(events.target.clone(), capacity) },
                bars as u32,
                population as u32,
                settings.max_hold_bars as u32,
                settings.min_hold_bars as u32,
                half_spread_pips,
                settings.adaptive_rr as f32,
                u32::from(dataset.has_adaptive_base),
            );

            if emitted > 0 {
                let (hit_cubes, hit_dim) = launch_dims(client, emitted);
                c_population_first_hit_kernel::launch::<R>(
                    client,
                    hit_cubes,
                    hit_dim,
                    unsafe { ArrayArg::from_raw_parts(dataset.high.clone(), bars) },
                    unsafe { ArrayArg::from_raw_parts(dataset.low.clone(), bars) },
                    unsafe { ArrayArg::from_raw_parts(dataset.gap_flags.clone(), bars) },
                    unsafe { ArrayArg::from_raw_parts(events.entry.clone(), capacity) },
                    unsafe { ArrayArg::from_raw_parts(events.last.clone(), capacity) },
                    unsafe { ArrayArg::from_raw_parts(events.direction.clone(), capacity) },
                    unsafe { ArrayArg::from_raw_parts(events.stop.clone(), capacity) },
                    unsafe { ArrayArg::from_raw_parts(events.target.clone(), capacity) },
                    unsafe { ArrayArg::from_raw_parts(events.exit_bar.clone(), capacity) },
                    unsafe { ArrayArg::from_raw_parts(events.exit_reason.clone(), capacity) },
                    emitted as u32,
                    bars as u32,
                    settings.min_hold_bars as u32,
                    settings.max_hold_bars as u32,
                );
            }

            let (stitch_cubes, stitch_dim) = launch_dims(client, population);
            stitch_c_population_trades::launch::<R>(
                client,
                stitch_cubes,
                stitch_dim,
                unsafe { ArrayArg::from_raw_parts(dataset.days.clone(), bars) },
                unsafe { ArrayArg::from_raw_parts(genes.event_offsets.clone(), population + 1) },
                unsafe { ArrayArg::from_raw_parts(events.entry.clone(), capacity) },
                unsafe { ArrayArg::from_raw_parts(events.exit_bar.clone(), capacity) },
                unsafe { ArrayArg::from_raw_parts(events.exit_reason.clone(), capacity) },
                unsafe { ArrayArg::from_raw_parts(events.accepted_event.clone(), capacity) },
                unsafe { ArrayArg::from_raw_parts(genes.accepted_count.clone(), population) },
                bars as u32,
                population as u32,
                settings.max_trades_per_day as u32,
            );

            let (reduce_cubes, reduce_dim) = launch_dims(client, population);
            reduce_c_population_metrics::launch::<R>(
                client,
                reduce_cubes,
                reduce_dim,
                unsafe { ArrayArg::from_raw_parts(dataset.close.clone(), bars) },
                unsafe { ArrayArg::from_raw_parts(dataset.high.clone(), bars) },
                unsafe { ArrayArg::from_raw_parts(dataset.low.clone(), bars) },
                unsafe { ArrayArg::from_raw_parts(dataset.months.clone(), bars) },
                unsafe { ArrayArg::from_raw_parts(dataset.days.clone(), bars) },
                unsafe { ArrayArg::from_raw_parts(dataset.timestamp_pair.clone(), bars * 2) },
                unsafe { ArrayArg::from_raw_parts(genes.confidences.clone(), genes.signal_len) },
                unsafe { ArrayArg::from_raw_parts(events.entry.clone(), capacity) },
                unsafe { ArrayArg::from_raw_parts(events.direction.clone(), capacity) },
                unsafe { ArrayArg::from_raw_parts(events.stop.clone(), capacity) },
                unsafe { ArrayArg::from_raw_parts(events.target.clone(), capacity) },
                unsafe { ArrayArg::from_raw_parts(events.exit_bar.clone(), capacity) },
                unsafe { ArrayArg::from_raw_parts(events.exit_reason.clone(), capacity) },
                unsafe { ArrayArg::from_raw_parts(events.accepted_event.clone(), capacity) },
                unsafe { ArrayArg::from_raw_parts(genes.accepted_count.clone(), population) },
                unsafe { ArrayArg::from_raw_parts(genes.event_offsets.clone(), population + 1) },
                unsafe {
                    ArrayArg::from_raw_parts(
                        genes.month_workspace.clone(),
                        genes.month_workspace_len,
                    )
                },
                unsafe { ArrayArg::from_raw_parts(genes.metrics.clone(), genes.metrics_len) },
                bars as u32,
                population as u32,
                month_capacity as u32,
                crate::eval::current_backtest_runtime_overrides().initial_equity as f32,
                half_spread_pips,
                settings.commission_per_trade as f32,
                settings.pip_value_per_lot as f32,
                settings.swap_long_pips_per_day as f32,
                settings.swap_short_pips_per_day as f32,
                settings.pnl_conversion_fee_rate as f32,
                u32::from(settings.risk_based_sizing),
                settings.risk_per_trade_min as f32,
                settings.risk_per_trade_max as f32,
                settings.high_quality_confidence as f32,
            );
        }

        Ok(emitted)
    }

    /// Allocate the sparse event workspace for the observed event count.
    ///
    /// The buffers are reused whenever the existing capacity already covers the
    /// request, so a repeated benchmark iteration performs no new allocation.
    fn ensure_event_capacity(&mut self, required: usize) -> Result<(), EngineError> {
        let existing = self
            .session
            .backend()
            .events
            .as_ref()
            .map_or(0, |events| events.capacity);
        if existing >= required {
            return Ok(());
        }
        let capacity = required
            .max(existing.saturating_mul(2))
            .min(self.max_events)
            .max(required);
        let client = &self.session.backend().client;
        let events = EventsResident {
            candidate: client.empty(capacity * size_of::<i32>()),
            scenario: client.empty(capacity * size_of::<i32>()),
            entry: client.empty(capacity * size_of::<i32>()),
            last: client.empty(capacity * size_of::<i32>()),
            direction: client.empty(capacity * size_of::<i32>()),
            stop: client.empty(capacity * size_of::<f32>()),
            target: client.empty(capacity * size_of::<f32>()),
            exit_bar: client.empty(capacity * size_of::<i32>()),
            exit_reason: client.empty(capacity * size_of::<i32>()),
            accepted_event: client.empty(capacity * size_of::<i32>()),
            capacity,
        };
        self.session.transfers().record_workspace_allocations(10);
        self.session.backend_mut().events = Some(events);
        Ok(())
    }
}

impl<R: Runtime> BacktestEngine for PrototypeCBacktestEngine<R> {
    type Backend = PrototypeCResources<R>;

    fn status(&self) -> EngineStatus {
        EngineStatus::Ready
    }

    fn capabilities(&self) -> EngineCapabilities {
        prototype_c_capabilities()
    }

    fn session(&self) -> &GpuDiscoverySession<Self::Backend> {
        &self.session
    }

    fn upload_dataset(&mut self, bytes: &[u8]) -> Result<DatasetHandle, EngineError> {
        if self.session.backend().dataset.is_some() {
            return Err(EngineError::UnsupportedCapability {
                operation: "dataset_reupload",
                detail: "a Prototype C session accepts exactly one logical dataset upload".into(),
            });
        }
        let upload = PrototypeADatasetUpload::decode(bytes)
            .map_err(|error| EngineError::Backend(error.to_string()))?;
        let buffers = CPopulationHostBuffers::from_dataset(&upload)?;
        let handle = self.session.allocate_handle::<DatasetHandle>();
        let resident = {
            let client = &self.session.backend().client;
            DatasetResident {
                handle,
                close: client.create_from_slice(f32::as_bytes(&buffers.close_pips)),
                high: client.create_from_slice(f32::as_bytes(&buffers.high_pips)),
                low: client.create_from_slice(f32::as_bytes(&buffers.low_pips)),
                indicators: client.create_from_slice(f32::as_bytes(&upload.indicators)),
                months: client.create_from_slice(i32::as_bytes(&buffers.months)),
                days: client.create_from_slice(i32::as_bytes(&buffers.days)),
                timestamp_pair: client.create_from_slice(i32::as_bytes(&buffers.timestamp_pair)),
                gap_flags: client.create_from_slice(i32::as_bytes(&buffers.gap_flags)),
                smc_rows: client.create_from_slice(i32::as_bytes(&buffers.smc_rows)),
                adaptive_base: client.create_from_slice(f32::as_bytes(&buffers.adaptive_base_pips)),
                adaptive_base_len: buffers.adaptive_base_pips.len(),
                has_adaptive_base: buffers.has_adaptive_base,
                bars: buffers.bars,
                feature_count: buffers.feature_count,
                indicator_len: upload.indicators.len(),
                upload,
            }
        };
        let upload_bytes =
            ((resident.bars * 3 + resident.indicator_len + resident.adaptive_base_len)
                * size_of::<f32>()
                + (resident.bars * 5 + resident.bars * SMC_WIDTH) * size_of::<i32>())
                as u64;
        self.session.transfers().record_dataset_upload(upload_bytes);
        self.session.backend_mut().dataset = Some(resident);
        Ok(handle)
    }

    fn upload_genes(&mut self, bytes: &[u8]) -> Result<GeneBufferHandle, EngineError> {
        let upload = PrototypeAGeneUpload::decode(bytes)
            .map_err(|error| EngineError::Backend(error.to_string()))?;
        let (bars, feature_count, month_capacity) = {
            let dataset = self.session.backend().dataset.as_ref().ok_or_else(|| {
                EngineError::Backend("upload the Prototype C dataset first".into())
            })?;
            (
                dataset.bars,
                dataset.feature_count,
                dataset.upload.settings.to_settings().month_capacity(),
            )
        };
        if let Some((position, index)) = upload
            .indices
            .iter()
            .copied()
            .enumerate()
            .find(|(_, index)| *index < 0 || *index as usize >= feature_count)
        {
            return Err(EngineError::Backend(format!(
                "Prototype C gene term {position} references feature {index}, outside \
                 0..{feature_count}"
            )));
        }
        if month_capacity == 0 {
            return Err(EngineError::Backend(
                "Prototype C requires a non-zero month capacity".into(),
            ));
        }

        let population = upload.population();
        let smc_flags = upload
            .smc_flags
            .iter()
            .flatten()
            .map(|value| i32::from(*value))
            .collect::<Vec<i32>>();
        let stop_pips = upload
            .stop_pips
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<f32>>();
        let target_pips = upload
            .target_pips
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<f32>>();
        let stop_vol_multipliers = upload
            .stop_vol_multipliers
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<f32>>();

        let signal_len = population * bars;
        let month_workspace_len = population * month_capacity * 2;
        let metrics_len = population * C_METRIC_WIDTH;
        let term_len = upload.indices.len();
        let handle = self.session.allocate_handle::<GeneBufferHandle>();
        let resident = {
            let client = &self.session.backend().client;
            GenesResident {
                handle,
                offsets: client.create_from_slice(i32::as_bytes(&upload.offsets)),
                indices: client.create_from_slice(i32::as_bytes(&pad_i32(&upload.indices))),
                weights: client.create_from_slice(f32::as_bytes(&pad_f32(&upload.weights))),
                long_thresholds: client.create_from_slice(f32::as_bytes(&upload.long_thresholds)),
                short_thresholds: client.create_from_slice(f32::as_bytes(&upload.short_thresholds)),
                stop_pips: client.create_from_slice(f32::as_bytes(&stop_pips)),
                target_pips: client.create_from_slice(f32::as_bytes(&target_pips)),
                stop_vol_multipliers: client
                    .create_from_slice(f32::as_bytes(&stop_vol_multipliers)),
                smc_flags: client.create_from_slice(i32::as_bytes(&smc_flags)),
                smc_weights: client.create_from_slice(f32::as_bytes(&upload.smc_weights)),
                signals: client.empty(signal_len * size_of::<i32>()),
                confidences: client.empty(signal_len * size_of::<f32>()),
                event_counts: client.empty(population * size_of::<i32>()),
                event_offsets: client.empty((population + 1) * size_of::<i32>()),
                event_total: client.empty(size_of::<i32>()),
                month_workspace: client.empty(month_workspace_len * size_of::<f32>()),
                metrics: client.empty(metrics_len * size_of::<f32>()),
                accepted_count: client.empty(population * size_of::<i32>()),
                term_len,
                signal_len,
                month_workspace_len,
                metrics_len,
                month_capacity,
                population,
                upload,
            }
        };
        let upload_bytes =
            ((resident.upload.offsets.len() + term_len + smc_flags.len()) * size_of::<i32>()
                + (term_len + population * 5 + SMC_WIDTH) * size_of::<f32>()) as u64;
        self.session.transfers().record_gene_upload(upload_bytes);
        self.session.transfers().record_workspace_allocations(8);
        let resources = self.session.backend_mut();
        resources.metrics = None;
        resources.selection = None;
        resources.scenarios = None;
        resources.events = None;
        resources.genes = Some(resident);
        Ok(handle)
    }

    fn upload_scenarios(&mut self, bytes: &[u8]) -> Result<ScenarioBufferHandle, EngineError> {
        let upload = PrototypeAScenarioUpload::decode(bytes)
            .map_err(|error| EngineError::Backend(error.to_string()))?;
        {
            let resources = self.session.backend();
            let dataset = resources.dataset.as_ref().ok_or_else(|| {
                EngineError::Backend("upload the Prototype C dataset first".into())
            })?;
            let genes = resources
                .genes
                .as_ref()
                .ok_or_else(|| EngineError::Backend("upload Prototype C genes first".into()))?;
            // Pre-execution capability gate, identical to Prototype B's.
            validate_population_eligibility(
                &dataset.upload,
                &genes.upload,
                &upload,
                PrototypeKind::CSparseFirstHit,
            )?;
        }
        let handle = self.session.allocate_handle::<ScenarioBufferHandle>();
        self.session
            .transfers()
            .record_scenario_upload((upload.scenarios.len() * size_of::<u64>()) as u64);
        let resources = self.session.backend_mut();
        resources.metrics = None;
        resources.selection = None;
        resources.scenarios = Some(ScenarioSlot { handle, upload });
        Ok(handle)
    }

    fn evaluate(
        &mut self,
        dataset: DatasetHandle,
        genes: GeneBufferHandle,
        scenarios: ScenarioBufferHandle,
        wait_for: Option<DeviceEventHandle>,
    ) -> Result<(DeviceMetricsHandle, Option<DeviceEventHandle>), EngineError> {
        if wait_for.is_some() {
            return Err(EngineError::UnsupportedCapability {
                operation: "evaluate_wait_event",
                detail: "the first Stage-1 evaluate has no predecessor event".into(),
            });
        }
        {
            let resources = self.session.backend();
            let dataset_slot = resources.dataset.as_ref().ok_or_else(|| {
                EngineError::Backend("Prototype C dataset is not uploaded".into())
            })?;
            let gene_slot = resources
                .genes
                .as_ref()
                .ok_or_else(|| EngineError::Backend("Prototype C genes are not uploaded".into()))?;
            let scenario_slot = resources.scenarios.as_ref().ok_or_else(|| {
                EngineError::Backend("Prototype C scenarios are not uploaded".into())
            })?;
            self.validate_current("evaluate_dataset", dataset, dataset_slot.handle)?;
            self.validate_current("evaluate_genes", genes, gene_slot.handle)?;
            self.validate_current("evaluate_scenarios", scenarios, scenario_slot.handle)?;
        }

        let emitted = self.submit_population_chain()?;
        self.session.transfers().record_synchronization_event();

        let metrics_handle = self.session.allocate_handle::<DeviceMetricsHandle>();
        let event_handle = self.session.allocate_handle::<DeviceEventHandle>();
        let resources = self.session.backend_mut();
        resources.metrics = Some(MetricsSlot {
            handle: metrics_handle,
            dataset,
            genes,
            scenarios,
            event: event_handle,
            emitted_events: emitted,
        });
        resources.selection = None;
        Ok((metrics_handle, Some(event_handle)))
    }

    fn filter(
        &mut self,
        metrics: DeviceMetricsHandle,
        policy: DeviceFilterPolicy,
        wait_for: Option<DeviceEventHandle>,
    ) -> Result<(DeviceSelectionHandle, Option<DeviceEventHandle>), EngineError> {
        let metrics_slot =
            self.session.backend().metrics.as_ref().ok_or_else(|| {
                EngineError::Backend("Prototype C metrics are not available".into())
            })?;
        self.validate_current("filter_metrics", metrics, metrics_slot.handle)?;
        self.validate_wait_event("filter_wait_event", wait_for, metrics_slot.event)?;
        if policy != DeviceFilterPolicy::All {
            return Err(EngineError::UnsupportedCapability {
                operation: "device_filtering",
                detail: format!("{policy:?} is reserved for Stage 2; no CPU filtering was run"),
            });
        }
        let selection_handle = self.session.allocate_handle::<DeviceSelectionHandle>();
        let event_handle = self.session.allocate_handle::<DeviceEventHandle>();
        self.session.backend_mut().selection = Some(SelectionSlot {
            handle: selection_handle,
            metrics,
            event: event_handle,
        });
        Ok((selection_handle, Some(event_handle)))
    }

    fn readback_compact(
        &mut self,
        selection: DeviceSelectionHandle,
        wait_for: Option<DeviceEventHandle>,
    ) -> Result<HostSurvivorSummary, EngineError> {
        let (metrics_handle, dataset_handle, gene_handle, scenario_handle) = {
            let selection_slot = self.session.backend().selection.as_ref().ok_or_else(|| {
                EngineError::Backend("Prototype C selection is not available".into())
            })?;
            self.validate_current("readback_selection", selection, selection_slot.handle)?;
            self.validate_wait_event("readback_wait_event", wait_for, selection_slot.event)?;
            let metrics_handle = selection_slot.metrics;
            let metrics_slot = self.session.backend().metrics.as_ref().ok_or_else(|| {
                EngineError::Backend("Prototype C metrics are not available".into())
            })?;
            self.validate_current("readback_metrics", metrics_handle, metrics_slot.handle)?;
            (
                metrics_handle,
                metrics_slot.dataset,
                metrics_slot.genes,
                metrics_slot.scenarios,
            )
        };
        let _ = metrics_handle;

        let expected_dataset = self
            .session
            .backend()
            .dataset
            .as_ref()
            .ok_or_else(|| EngineError::Backend("Prototype C dataset is not uploaded".into()))?
            .handle;
        let expected_genes = self
            .session
            .backend()
            .genes
            .as_ref()
            .ok_or_else(|| EngineError::Backend("Prototype C genes are not uploaded".into()))?
            .handle;
        let expected_scenarios = self
            .session
            .backend()
            .scenarios
            .as_ref()
            .ok_or_else(|| EngineError::Backend("Prototype C scenarios are not uploaded".into()))?
            .handle;
        self.validate_current("readback_dataset_parent", dataset_handle, expected_dataset)?;
        self.validate_current("readback_gene_parent", gene_handle, expected_genes)?;
        self.validate_current(
            "readback_scenario_parent",
            scenario_handle,
            expected_scenarios,
        )?;

        let (metrics_bytes, metrics_len) = {
            let resources = self.session.backend();
            let genes = resources
                .genes
                .as_ref()
                .expect("gene parent validated above");
            let bytes = resources
                .client
                .read_one(genes.metrics.clone())
                .map_err(|error| {
                    EngineError::Backend(format!("Prototype C metric readback failed: {error}"))
                })?;
            (bytes, genes.metrics_len)
        };
        let metrics = f32::from_bytes(metrics_bytes.as_ref()).to_vec();
        if metrics.len() != metrics_len {
            return Err(EngineError::Backend(format!(
                "Prototype C metric readback returned {} values, expected {metrics_len}",
                metrics.len()
            )));
        }
        self.session
            .transfers()
            .record_compact_readback((metrics.len() * size_of::<f32>()) as u64);

        let (genes_upload, scenarios_upload) = {
            let resources = self.session.backend();
            (
                resources
                    .genes
                    .as_ref()
                    .expect("gene parent validated above")
                    .upload
                    .clone(),
                resources
                    .scenarios
                    .as_ref()
                    .expect("scenario parent validated above")
                    .upload
                    .clone(),
            )
        };
        let summary =
            survivor_summary_from_device_metrics(&metrics, &genes_upload, &scenarios_upload)?;
        let resources = self.session.backend_mut();
        resources.metrics = None;
        resources.selection = None;
        Ok(summary)
    }
}

/// CubeCL rejects zero-length array bindings, so an empty CSR term list is
/// padded to one inert element. The kernels never read it: every CSR range is
/// derived from the offsets, which stay empty.
fn pad_i32(values: &[i32]) -> Vec<i32> {
    if values.is_empty() {
        vec![0]
    } else {
        values.to_vec()
    }
}

fn pad_f32(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        vec![0.0]
    } else {
        values.to_vec()
    }
}
