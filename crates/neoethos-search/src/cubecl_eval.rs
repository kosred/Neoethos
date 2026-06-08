use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
#[cfg(feature = "gpu-cuda")]
use cubecl::cuda::{CudaDevice, CudaRuntime};
#[cfg(all(feature = "gpu-vulkan", not(feature = "gpu-cuda")))]
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::prelude::*;
use half::bf16;
use ndarray::ArrayView2;
use neoethos_core::TrainingPrecision;

use crate::eval::{BacktestSettings, SmcRow};

const SMC_WIDTH: usize = 11;
const BACKTEST_CORE_METRIC_WIDTH: usize = 7;

// ─── F-CORE3 consolidation — CUDA env-var registry ──────────────────
//
// **2026-05-25**: the 7 inline `std::env::var(...)` reads previously
// scattered across `requested_eval_precision`, `cuda_eval_signal_kernel_enabled`,
// `cuda_eval_backtest_kernel_enabled`, `signal_kernel_units`,
// `backtest_kernel_units`, and `cuda_device_id` now route through this
// typed registry. Same canonical pattern as
// `crates/neoethos-app/src/app_services/env_overrides.rs` and
// `crates/neoethos-search/src/genetic/runtime_overrides.rs`.
//
// The `CudaEnvKnobs` struct is built once on first access via
// `cuda_env_knobs()` (lazy OnceLock) so the env-var reads happen at
// most once per process. Mirrors the audit-baseline `HardwareRuntimeOverrides`
// shape.
//
// Knobs covered (env-var name → typed field):
// - `NEOETHOS_BOT_SEARCH_EVAL_PRECISION` / `NEOETHOS_BOT_TRAIN_PRECISION` /
//   `FOREX_TRAIN_PRECISION` → `requested_precision: TrainingPrecision`
// - `NEOETHOS_BOT_SEARCH_EVAL_CUDA_KERNEL` → `eval_kernel_enabled: bool`
// - `NEOETHOS_BOT_SEARCH_BACKTEST_CUDA_KERNEL` → `backtest_kernel_enabled: bool`
// - `NEOETHOS_BOT_SEARCH_EVAL_KERNEL_UNITS` → `eval_kernel_units_override: Option<u32>`
// - `NEOETHOS_BOT_SEARCH_BACKTEST_KERNEL_UNITS` → `backtest_kernel_units_override: Option<u32>`
// - `NEOETHOS_BOT_SEARCH_EVAL_CUDA_DEVICE` → `cuda_device_id: usize`
//
// Vulkan note: the wgpu-vulkan backend is wired via feature
// aggregation (`vulkan` cargo feature). It doesn't read any of these
// env vars — the cubecl runtime selects Vulkan at compile time when
// the `vulkan` feature is on. No env-knob registry needed for Vulkan
// today; if one becomes necessary it'd live in a sibling
// `wgpu_eval.rs` file with the same typed-registry pattern.

#[derive(Debug, Clone, Copy)]
struct CudaEnvKnobs {
    requested_precision: TrainingPrecision,
    eval_kernel_enabled: bool,
    backtest_kernel_enabled: bool,
    eval_kernel_units_override: Option<u32>,
    backtest_kernel_units_override: Option<u32>,
    // Read only by the `gpu-cuda` device selector; unused on the wgpu/Vulkan
    // path (which uses `WgpuDevice::DefaultDevice`).
    #[allow(dead_code)]
    cuda_device_id: usize,
}

impl CudaEnvKnobs {
    fn from_env() -> Self {
        Self {
            requested_precision: read_requested_precision_from_env(),
            eval_kernel_enabled: read_kernel_enabled_from_env(
                "NEOETHOS_BOT_SEARCH_EVAL_CUDA_KERNEL",
            ),
            backtest_kernel_enabled: read_kernel_enabled_from_env(
                "NEOETHOS_BOT_SEARCH_BACKTEST_CUDA_KERNEL",
            ),
            eval_kernel_units_override: read_kernel_units_from_env(
                "NEOETHOS_BOT_SEARCH_EVAL_KERNEL_UNITS",
            ),
            // Backtest units fall back to eval units when the explicit
            // backtest knob is unset — preserves the original semantics
            // (`signal_kernel_units` and `backtest_kernel_units` both
            // honoured EVAL_KERNEL_UNITS as the umbrella default).
            backtest_kernel_units_override: read_kernel_units_from_env(
                "NEOETHOS_BOT_SEARCH_BACKTEST_KERNEL_UNITS",
            )
            .or_else(|| read_kernel_units_from_env("NEOETHOS_BOT_SEARCH_EVAL_KERNEL_UNITS")),
            cuda_device_id: read_cuda_device_id_from_env(),
        }
    }
}

static CUDA_ENV_KNOBS: OnceLock<CudaEnvKnobs> = OnceLock::new();

fn cuda_env_knobs() -> CudaEnvKnobs {
    *CUDA_ENV_KNOBS.get_or_init(CudaEnvKnobs::from_env)
}

fn read_requested_precision_from_env() -> TrainingPrecision {
    [
        "NEOETHOS_BOT_SEARCH_EVAL_PRECISION",
        "NEOETHOS_BOT_TRAIN_PRECISION",
        "FOREX_TRAIN_PRECISION",
    ]
    .iter()
    .find_map(|key| std::env::var(key).ok())
    .and_then(|value| parse_training_precision(&value))
    .unwrap_or(TrainingPrecision::Fp32)
}

fn read_kernel_enabled_from_env(name: &str) -> bool {
    !matches!(
        std::env::var(name)
            .ok()
            .map(|value| value.trim().to_ascii_lowercase()),
        Some(value) if matches!(value.as_str(), "0" | "false" | "off" | "disable" | "disabled")
    )
}

fn read_kernel_units_from_env(name: &str) -> Option<u32> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value > 0)
}

fn read_cuda_device_id_from_env() -> usize {
    match std::env::var("NEOETHOS_BOT_SEARCH_EVAL_CUDA_DEVICE") {
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
                    "NEOETHOS_BOT_SEARCH_EVAL_CUDA_DEVICE is set but not a valid \
                     non-negative integer; falling back to device 0."
                );
                0
            }
        },
    }
}

/// Create a `ComputeClient` for the active GPU runtime. The concrete runtime is
/// chosen at COMPILE time by the GPU feature flag — CUDA under `gpu-cuda`,
/// wgpu/Vulkan under `gpu-vulkan` — so every downstream kernel launch stays
/// generic over `R: Runtime` and runs unchanged on whichever backend was built.
/// (When both features are on, CUDA wins.)
#[cfg(feature = "gpu-cuda")]
fn create_gpu_client(device_override: Option<usize>) -> Result<ComputeClient<CudaRuntime>> {
    let device_id = device_override.unwrap_or_else(cuda_device_id);
    let device_count = tch::Cuda::device_count();
    if device_count <= device_id as i64 {
        bail!(
            "GPU evaluator requested CUDA device {} but only {} CUDA devices are available",
            device_id,
            device_count
        );
    }
    let device = CudaDevice::new(device_id);
    Ok(CudaRuntime::client(&device))
}

/// wgpu/Vulkan twin of the `gpu-cuda` client factory above. Uses cubecl's
/// `wgpu` (naga → SPIR-V) path, NOT `wgpu-spirv`: the direct SPIR-V passthrough
/// crashes AMD's Vulkan driver, so naga emits validated SPIR-V.
///
/// MULTI-GPU (2026-06-06): `NEOETHOS_BOT_SEARCH_EVAL_WGPU_DEVICE=<n>` pins this
/// process to discrete GPU `n` (`WgpuDevice::DiscreteGpu(n)`) — combo-level
/// parallelism: launch one discovery process per A6000 (env=0 and env=1) and
/// both cards run concurrently. Unset ⇒ `DefaultDevice` (the best adapter; the
/// first discrete GPU on a multi-GPU box).
#[cfg(all(feature = "gpu-vulkan", not(feature = "gpu-cuda")))]
fn create_gpu_client(device_override: Option<usize>) -> Result<ComputeClient<WgpuRuntime>> {
    // Multi-GPU sharding (Stage 2): an explicit `device_override` (one per lane)
    // wins; otherwise the legacy singular env var; otherwise the best adapter.
    let env_device = std::env::var("NEOETHOS_BOT_SEARCH_EVAL_WGPU_DEVICE")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok());
    let device = match device_override.or(env_device) {
        Some(n) => WgpuDevice::DiscreteGpu(n),
        None => WgpuDevice::DefaultDevice,
    };
    Ok(WgpuRuntime::client(&device))
}

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
    // Phase 2 (2026-06-06): per-bar confidence of the raw threshold crossing,
    // mirrors `synthesize_signals_and_confidence_cpu` (eval.rs:1543-1544).
    // f32 regardless of `F` so the backtest kernel's risk-sizing reads the same
    // precision the CPU uses; written only where the final signal survives.
    confidences_out: &mut Array<f32>,
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
            confidences_out[pos] = 0.0;
            terminate!();
        }

        // Confidence of the raw threshold crossing (pre-gate); computed in `F`
        // (cast to f32 only at the store) to match the CPU
        // `synthesize_signals_and_confidence_cpu` (eval.rs:1507/1543-1544):
        // gap = |lt - st| guarded >= 1e-6; margin = (combined-lt) long / (st-combined) short;
        // conf = (margin/gap).clamp(0,1). Written only where the final signal survives.
        let gap_raw = lt - st;
        let gap_abs = if gap_raw >= F::new(0.0) {
            gap_raw
        } else {
            F::new(0.0) - gap_raw
        };
        let gap = if gap_abs < F::new(1e-6) {
            F::new(1e-6)
        } else {
            gap_abs
        };
        let margin = if sig_val == 1 {
            combined_val - lt
        } else {
            st - combined_val
        };
        let conf_f = margin / gap;
        let conf = if conf_f < F::new(0.0) {
            F::new(0.0)
        } else if conf_f > F::new(1.0) {
            F::new(1.0)
        } else {
            conf_f
        };

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
            confidences_out[pos] = f32::cast_from(conf);
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
            confidences_out[pos] = f32::cast_from(conf);
        } else {
            output[pos] = 0;
            confidences_out[pos] = 0.0;
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
    // Phase C.3 (2026-05-28) — broker-supplied carry costs mirrored
    // from the CPU `apply_carry_and_fee` helper. Sign convention
    // matches the broker: positive = credit, negative = charge per
    // overnight day. `pnl_conversion_fee_rate` is a fraction (0.005
    // = 0.5%); skipped if non-finite / out-of-range so a missing-
    // broker-data run still produces a backtest, matching the CPU
    // kernel's fail-safe default behaviour.
    swap_long_pips_per_day: f32,
    swap_short_pips_per_day: f32,
    pnl_conversion_fee_rate: f32,
    // Phase 2 (2026-06-06): risk-based, confidence-scaled position sizing —
    // mirrors `risk_based_pos_lots` + the pos_lots multiply sites on the CPU
    // (eval.rs:657-685, 1049-1054, 809/865-880/979/627). `confidences_flat` is
    // per-gene-per-bar (same layout as `signals_flat`). When `risk_based_sizing`
    // is 0 the kernel forces pos_lots = 1.0 (legacy fixed-1-lot parity).
    confidences_flat: &Array<f32>,
    risk_based_sizing: i32,
    risk_per_trade_min: f32,
    risk_per_trade_max: f32,
    high_quality_confidence: f32,
    // Per-month STARTING equity (sibling of monthly_pnls_out); the host divides
    // monthly_pnls_out / month_start_equities_out to get monthly_target_hit_rate
    // (metric slot 7), matching eval.rs:1110-1131.
    month_start_equities_out: &mut Array<f32>,
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
            month_start_equities_out[month_base + zero_idx] = initial_equity;
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
        // Per-month starting equity (slot-7 monthly_target_hit_rate). Mirrors the
        // CPU `current_month_start_equity` carried at each month boundary (eval.rs:732/778).
        let current_month_start_equity = RuntimeCell::<f32>::new(initial_equity);
        // Confidence-scaled lot size, captured ONCE at entry, held for the trade
        // (eval.rs:1049-1054). 1.0 = legacy fixed-1-lot.
        let pos_lots = RuntimeCell::<f32>::new(1.0);

        let last_day = RuntimeCell::<i32>::new(-1);
        let day_peak = RuntimeCell::<f32>::new(initial_equity);
        let day_low = RuntimeCell::<f32>::new(initial_equity);
        let max_daily_dd = RuntimeCell::<f32>::new(0.0);
        let day_trade_count = RuntimeCell::<u32>::new(0);

        let in_pos = RuntimeCell::<i32>::new(0);
        let entry_px = RuntimeCell::<f32>::new(0.0);
        let entry_idx = RuntimeCell::<i32>::new(-1);
        let trail_px = RuntimeCell::<f32>::new(0.0);
        // Phase C.3: accumulated days in position. Resets to 0 at entry,
        // each in-position bar adds `timestamp_deltas_ms[i] / 86_400_000`.
        // f32 precision loss on the cast is bounded by ~5 ms per bar
        // (cast of values up to 86.4M ms into 24-bit mantissa); over
        // a year of D1 bars this accumulates to <$0.001 of swap error
        // at typical EURUSD pip values — negligible vs the $122/year
        // swap charge being modelled.
        let position_days = RuntimeCell::<f32>::new(0.0);

        for i in 1..n_samples {
            // Phase C.3: accumulate carry duration while in position.
            // Runs BEFORE any exit logic so the close branches use the
            // total time held, INCLUDING the delta into the current bar.
            if in_pos.read() != 0 && use_timestamps != 0 && timestamp_deltas_ms[i] > 0 {
                let delta_days = timestamp_deltas_ms[i] as f32 / 86_400_000.0;
                position_days.store(position_days.read() + delta_days);
            }

            let m_val = month_idx[i];
            let last_month_v = last_month.read();
            if m_val != last_month_v {
                if last_month_v != -1 {
                    let next_ptr = month_ptr.read() + 1;
                    month_ptr.store(next_ptr);
                    if next_ptr >= 0 && next_ptr < month_capacity as i32 {
                        monthly_pnls_out[month_base + next_ptr as usize] = current_month_pnl.read();
                        month_start_equities_out[month_base + next_ptr as usize] =
                            current_month_start_equity.read();
                    }
                }
                current_month_pnl.store(0.0);
                // New month starts at the equity carried in (eval.rs:778). Runs
                // BEFORE this bar's exits mutate equity — preserve the ordering.
                current_month_start_equity.store(equity.read());
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
                // Phase 2: scale (gross - commission - half_spread) by pos_lots
                // BEFORE swap (swap scaled in its own term). Matches eval.rs:809.
                pnl_cell.store(pnl_cell.read() * pos_lots.read());
                // Phase C.3: broker swap (signed: + = credit, − = charge).
                let swap_per_day_gap = if in_pos_v == 1 {
                    swap_long_pips_per_day
                } else {
                    swap_short_pips_per_day
                };
                let swap_credit_gap =
                    swap_per_day_gap * position_days.read() * pip_value_per_lot * pos_lots.read();
                pnl_cell.store(pnl_cell.read() + swap_credit_gap);
                // PnL conversion fee applied last; skip if out-of-range.
                if pnl_conversion_fee_rate > 0.0 && pnl_conversion_fee_rate < 1.0 {
                    pnl_cell.store(pnl_cell.read() * (1.0 - pnl_conversion_fee_rate));
                }
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

                // Phase 2: float PnL scaled by pos_lots (eval.rs:865-880) so the
                // equity-based DD tracks the sized position.
                let worst_base = if in_pos_v2 == 1 {
                    (lo - entry_px_v) * pip_value_per_lot
                } else {
                    (entry_px_v - hi) * pip_value_per_lot
                };
                let worst_float_pnl = worst_base * pos_lots.read();
                let eq = equity.read();
                if (eq + worst_float_pnl) < day_low.read() {
                    day_low.store(eq + worst_float_pnl);
                }

                let best_base = if in_pos_v2 == 1 {
                    (hi - entry_px_v) * pip_value_per_lot
                } else {
                    (entry_px_v - lo) * pip_value_per_lot
                };
                let best_float_pnl = best_base * pos_lots.read();
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
                    // Apply only the trail from PRIOR bars (no intra-bar look-ahead — must
                    // match the CPU eval: this bar's high can't move the stop its own low is
                    // checked against). `trail_px == 0.0` is the unset sentinel.
                    if trailing_enabled != 0 && trail_px.read() > 0.0 && trail_px.read() > sl_cell.read() {
                        sl_cell.store(trail_px.read());
                    }
                    let sl_v = sl_cell.read();
                    if lo <= sl_v {
                        pnl_cell.store((sl_v - entry_px_v) * pip_value_per_lot);
                        exit_cell.store(1);
                    } else if hi >= tp {
                        pnl_cell.store((tp - entry_px_v) * pip_value_per_lot);
                        exit_cell.store(1);
                    }
                    // AFTER the exit check: ratchet the trail up from THIS bar's high.
                    if exit_cell.read() == 0 && trailing_enabled != 0 {
                        let mv = hi - entry_px_v;
                        if mv >= (trailing_be_trigger_r * sl_distance) {
                            let candidate = hi - (trailing_atr_multiplier * sl_distance);
                            if trail_px.read() == 0.0 || candidate > trail_px.read() {
                                trail_px.store(candidate);
                            }
                        }
                    }
                } else if past_min_hold {
                    let sl_cell = RuntimeCell::<f32>::new(entry_px_v + sl_distance);
                    let tp = entry_px_v - tp_distance;
                    if trailing_enabled != 0 && trail_px.read() > 0.0 && trail_px.read() < sl_cell.read() {
                        sl_cell.store(trail_px.read());
                    }
                    let sl_v = sl_cell.read();
                    if hi >= sl_v {
                        pnl_cell.store((entry_px_v - sl_v) * pip_value_per_lot);
                        exit_cell.store(1);
                    } else if lo <= tp {
                        pnl_cell.store((entry_px_v - tp) * pip_value_per_lot);
                        exit_cell.store(1);
                    }
                    // AFTER the exit check: ratchet the trail down from THIS bar's low.
                    if exit_cell.read() == 0 && trailing_enabled != 0 {
                        let mv = entry_px_v - lo;
                        if mv >= (trailing_be_trigger_r * sl_distance) {
                            let candidate = lo + (trailing_atr_multiplier * sl_distance);
                            if trail_px.read() == 0.0 || candidate < trail_px.read() {
                                trail_px.store(candidate);
                            }
                        }
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
                    // Phase 2: scale (gross - commission - half_spread) by pos_lots
                    // BEFORE swap. Matches eval.rs:979.
                    pnl_cell.store(pnl_cell.read() * pos_lots.read());
                    // Phase C.3: broker swap (signed: + = credit, − = charge).
                    let swap_per_day = if in_pos_v2 == 1 {
                        swap_long_pips_per_day
                    } else {
                        swap_short_pips_per_day
                    };
                    let swap_credit =
                        swap_per_day * position_days.read() * pip_value_per_lot * pos_lots.read();
                    pnl_cell.store(pnl_cell.read() + swap_credit);
                    if pnl_conversion_fee_rate > 0.0 && pnl_conversion_fee_rate < 1.0 {
                        pnl_cell.store(pnl_cell.read() * (1.0 - pnl_conversion_fee_rate));
                    }
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
                        // Phase C.3: reset carry accumulator at new entry.
                        position_days.store(0.0);
                        // Phase 2: risk-based, confidence-scaled lot size, captured
                        // at entry from running equity + the prior-bar confidence
                        // (causal i-1, same shift as the signal read). Mirrors
                        // risk_based_pos_lots (eval.rs:657-685) term-for-term.
                        if risk_based_sizing != 0 {
                            // RuntimeCell idiom (cubecl rejects if-as-value that mixes
                            // array-read cube values with bare literals).
                            let conf_c = RuntimeCell::<f32>::new(confidences_flat[signal_base + i - 1]);
                            if conf_c.read() < 0.0 {
                                conf_c.store(0.0);
                            }
                            if conf_c.read() > 1.0 {
                                conf_c.store(1.0);
                            }
                            // conf_scale = (conf/hq).min(1.0), guarded for hq<=0 (=> 1.0).
                            let conf_scale = RuntimeCell::<f32>::new(1.0);
                            if high_quality_confidence > 0.0 {
                                let r = conf_c.read() / high_quality_confidence;
                                if r < 1.0 {
                                    conf_scale.store(r);
                                }
                            }
                            let risk_pct = risk_per_trade_min
                                + (risk_per_trade_max - risk_per_trade_min) * conf_scale.read();
                            // eff_sl = sl_distance.max(1.0)
                            let eff_sl = RuntimeCell::<f32>::new(1.0);
                            if sl_distance > 1.0 {
                                eff_sl.store(sl_distance);
                            }
                            let denom = eff_sl.read() * pip_value_per_lot;
                            let eq_now = equity.read();
                            // lots = (risk_pct*equity)/denom if valid, else 0; clamp [0,100].
                            let lots = RuntimeCell::<f32>::new(0.0);
                            if eq_now > 0.0 && denom > 1e-12 {
                                lots.store((risk_pct * eq_now) / denom);
                            }
                            if lots.read() > 100.0 {
                                lots.store(100.0);
                            }
                            if lots.read() < 0.0 {
                                lots.store(0.0);
                            }
                            pos_lots.store(lots.read());
                        } else {
                            pos_lots.store(1.0);
                        }
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
    // F-CORE3 closure: typed boundary via `cuda_env_knobs()`.
    cuda_env_knobs().requested_precision
}

fn prefers_bf16(requested: TrainingPrecision) -> bool {
    matches!(
        requested,
        TrainingPrecision::Bf16 | TrainingPrecision::Fp8 | TrainingPrecision::Bf4
    )
}

pub(crate) fn cuda_eval_signal_kernel_enabled() -> bool {
    // F-CORE3 closure: typed boundary via `cuda_env_knobs()`.
    cuda_env_knobs().eval_kernel_enabled
}

pub(crate) fn cuda_eval_backtest_kernel_enabled() -> bool {
    // F-CORE3 closure: typed boundary via `cuda_env_knobs()`.
    cuda_env_knobs().backtest_kernel_enabled
}

// **2026-05-25 — task #261**: switched from concrete `CudaRuntime` to a
// generic `R: Runtime` parameter so these helpers (and the kernel-launch
// fns below) compile against any cubecl runtime — CUDA (NVIDIA), Vulkan
// (cross-vendor via cubecl-wgpu/spirv), and ROCm/HIP. The CUDA-specific
// env knobs stay because they're plain numbers (kernel unit count,
// device id) — semantically valid for any backend; the `cuda_` prefix
// just reflects the env-var name and is a follow-up cosmetic rename.
fn signal_kernel_units<R: Runtime>(client: &ComputeClient<R>) -> u32 {
    let max_units = client.properties().hardware.max_units_per_cube.max(1);
    cuda_env_knobs()
        .eval_kernel_units_override
        .unwrap_or(max_units)
        .min(max_units)
        .max(1)
}

fn backtest_kernel_units<R: Runtime>(client: &ComputeClient<R>) -> u32 {
    let max_units = client.properties().hardware.max_units_per_cube.max(1);
    cuda_env_knobs()
        .backtest_kernel_units_override
        .unwrap_or(max_units)
        .min(max_units)
        .max(1)
}

#[allow(dead_code)] // gpu-cuda device selection only; wgpu uses DefaultDevice
fn cuda_device_id() -> usize {
    // F-CORE3 closure: typed boundary via `cuda_env_knobs()`. The
    // tracing::warn for unparseable values fires once at first read
    // (inside `read_cuda_device_id_from_env`) rather than on every
    // kernel launch.
    cuda_env_knobs().cuda_device_id
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

/// Host-side validation that every index the kernel will compute stays
/// within its array. This is the contract `synthesize_signals_kernel`
/// implicitly assumes; without these checks a single bad GA gene
/// silently reads garbage memory in the CUDA kernel, which produces
/// **wrong trading signals** with no error (panics in CUDA kernels
/// surface as CUDA driver errors or simply corrupt data — both far
/// worse than a clean `Err` for a real-money trading system).
///
/// Invariants (mirroring `synthesize_signals_kernel`):
///   1. `gene_offsets.len() == n_genes + 1` (CSR layout, last entry is total)
///   2. `gene_offsets` is monotonically non-decreasing
///   3. `gene_offsets[n_genes] as usize <= gene_indices.len()` and ≤ gene_weights.len()
///   4. every `gene_indices[i]` is in `[0, indicators_flat.len() / n_samples)`
///   5. `long_thr.len() == n_genes` and `short_thr.len() == n_genes`
///   6. `gene_smc_flags.len() == n_genes * SMC_WIDTH`
///   7. `smc_data.len() == n_samples * SMC_WIDTH`
///   8. `smc_weights.len() == SMC_WIDTH`
///   9. `indicators_flat.len() % n_samples == 0`
fn validate_signal_kernel_inputs<F>(
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
) -> Result<()> {
    if gene_offsets.len() != n_genes + 1 {
        bail!(
            "gene_offsets length {} must equal n_genes + 1 = {}",
            gene_offsets.len(),
            n_genes + 1
        );
    }
    if long_thr.len() != n_genes {
        bail!("long_thr length {} must equal n_genes {}", long_thr.len(), n_genes);
    }
    if short_thr.len() != n_genes {
        bail!("short_thr length {} must equal n_genes {}", short_thr.len(), n_genes);
    }
    if gene_smc_flags.len() != n_genes * SMC_WIDTH {
        bail!(
            "gene_smc_flags length {} must equal n_genes * SMC_WIDTH = {}",
            gene_smc_flags.len(),
            n_genes * SMC_WIDTH
        );
    }
    if smc_weights.len() != SMC_WIDTH {
        bail!(
            "smc_weights length {} must equal SMC_WIDTH = {}",
            smc_weights.len(),
            SMC_WIDTH
        );
    }
    if smc_data.len() != n_samples * SMC_WIDTH {
        bail!(
            "smc_data length {} must equal n_samples * SMC_WIDTH = {}",
            smc_data.len(),
            n_samples * SMC_WIDTH
        );
    }
    if n_samples == 0 {
        bail!("n_samples must be > 0");
    }
    if indicators_flat.len() % n_samples != 0 {
        bail!(
            "indicators_flat length {} must be a multiple of n_samples {}",
            indicators_flat.len(),
            n_samples
        );
    }
    let n_indicators = indicators_flat.len() / n_samples;
    let total_entries = gene_offsets[n_genes];
    if total_entries < 0 {
        bail!("gene_offsets[n_genes] = {} must be non-negative", total_entries);
    }
    if (total_entries as usize) > gene_indices.len() {
        bail!(
            "gene_offsets[n_genes] = {} exceeds gene_indices length {}",
            total_entries,
            gene_indices.len()
        );
    }
    if (total_entries as usize) > gene_weights.len() {
        bail!(
            "gene_offsets[n_genes] = {} exceeds gene_weights length {}",
            total_entries,
            gene_weights.len()
        );
    }
    // Monotonicity: gene_offsets[g] <= gene_offsets[g+1] for every g.
    for g in 0..n_genes {
        if gene_offsets[g] > gene_offsets[g + 1] {
            bail!(
                "gene_offsets must be non-decreasing: gene_offsets[{}]={} > gene_offsets[{}]={}",
                g,
                gene_offsets[g],
                g + 1,
                gene_offsets[g + 1]
            );
        }
        if gene_offsets[g] < 0 {
            bail!("gene_offsets[{}] = {} is negative", g, gene_offsets[g]);
        }
    }
    // Every gene_indices entry must reference a valid indicator row.
    // Checking up to `total_entries` because anything past that isn't read.
    let used_entries = total_entries as usize;
    for (i, &idx) in gene_indices.iter().take(used_entries).enumerate() {
        if idx < 0 {
            bail!("gene_indices[{}] = {} is negative", i, idx);
        }
        if (idx as usize) >= n_indicators {
            bail!(
                "gene_indices[{}] = {} exceeds n_indicators = {} (indicators_flat.len()/n_samples)",
                i,
                idx,
                n_indicators
            );
        }
    }
    Ok(())
}

// `F` first so callers can `launch_signal_kernel::<f32>(&client, ...)` and have
// `R` inferred from the client (turbofish is positional; the runtime is never
// named at the call sites).
/// Conservative max ELEMENTS per single GPU storage buffer. wgpu caps a storage
/// buffer at `max_storage_buffer_binding_size` (WebGPU default 128MB); exceeding
/// it raises "wgpu error: Out of Memory". We stay under it and window/batch the
/// GA's big buffers (the indicators matrix and the per-gene signal series) so
/// even huge-row timeframes (M1: ~5.3M rows) run on the GPU instead of falling
/// back to CPU. Overridable per box via `NEOETHOS_BOT_SEARCH_GPU_BUFFER_MB`
/// (raise it where the device's real limit is higher than the 128MB default).
fn gpu_buffer_elem_cap() -> usize {
    const DEFAULT_MB: usize = 120;
    let mb = std::env::var("NEOETHOS_BOT_SEARCH_GPU_BUFFER_MB")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|m| *m > 0)
        .unwrap_or(DEFAULT_MB);
    (mb.saturating_mul(1024 * 1024) / 4).max(1) // 4 bytes/element (i32/f32)
}

/// SAMPLE columns the signal-synth kernel can process in one launch so no buffer
/// exceeds the cap. Per-window buffers: indicators window (`n_indicators × W`),
/// signal/confidence outputs (`n_genes × W`), SMC window (`W × SMC_WIDTH`).
fn signal_window_size(n_indicators: usize, n_genes: usize, n_samples: usize) -> usize {
    let per_sample = n_indicators.max(n_genes).max(SMC_WIDTH).max(1);
    (gpu_buffer_elem_cap() / per_sample).clamp(1, n_samples.max(1))
}

/// GENES the backtest kernel can process in one launch — its signal buffer is
/// `B × n_samples`, so `B = cap / n_samples`.
fn backtest_gene_batch(n_genes: usize, n_samples: usize) -> usize {
    (gpu_buffer_elem_cap() / n_samples.max(1)).clamp(1, n_genes.max(1))
}

/// Gather indicator columns `[s0, s1)` out of the indicator-major flat matrix
/// (`[n_indicators × n_samples]`, indexed `idx*n_samples+s`) into a contiguous
/// `[n_indicators × (s1-s0)]` window the kernel indexes as `idx*W + (s-s0)`.
fn gather_indicator_window<F: Copy>(
    indicators_flat: &[F],
    n_indicators: usize,
    n_samples: usize,
    s0: usize,
    s1: usize,
) -> Vec<F> {
    let wlen = s1 - s0;
    let mut out = Vec::with_capacity(n_indicators.saturating_mul(wlen));
    for idx in 0..n_indicators {
        let base = idx * n_samples;
        out.extend_from_slice(&indicators_flat[base + s0..base + s1]);
    }
    out
}

/// Run the (stateless, per-sample) signal-synth kernel over SAMPLE-windows so the
/// indicators/signal buffers stay under the wgpu cap, assembling the full
/// `[n_genes × n_samples]` signal + confidence series on the host. Windowing is
/// exact (each sample is independent of the others) so the result is identical
/// to a single whole-series launch — CPU↔GPU parity is preserved.
fn windowed_signal_synth<F, R: Runtime>(
    client: &ComputeClient<R>,
    indicators_flat: &[F],
    n_indicators: usize,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[F],
    long_thr: &[F],
    short_thr: &[F],
    smc_data_flat: &[i32],
    gene_smc_flags_flat: &[i32],
    smc_weights: &[F],
    n_genes: usize,
    n_samples: usize,
    gate_threshold: F,
) -> Result<(Vec<i32>, Vec<f32>)>
where
    F: Float + CubeElement,
{
    let w = signal_window_size(n_indicators, n_genes, n_samples);
    let mut signals = vec![0i32; n_genes.saturating_mul(n_samples)];
    let mut conf = vec![0f32; n_genes.saturating_mul(n_samples)];
    let mut s0 = 0;
    while s0 < n_samples {
        let s1 = (s0 + w).min(n_samples);
        let wlen = s1 - s0;
        let ind_window = gather_indicator_window(indicators_flat, n_indicators, n_samples, s0, s1);
        let smc_window = &smc_data_flat[s0 * SMC_WIDTH..s1 * SMC_WIDTH];
        let (sig_w, conf_w) = launch_signal_kernel::<F, R>(
            client,
            &ind_window,
            gene_offsets,
            gene_indices,
            gene_weights,
            long_thr,
            short_thr,
            smc_window,
            gene_smc_flags_flat,
            smc_weights,
            n_genes,
            wlen,
            gate_threshold,
        )?;
        for gene in 0..n_genes {
            let dst = gene * n_samples + s0;
            let src = gene * wlen;
            signals[dst..dst + wlen].copy_from_slice(&sig_w[src..src + wlen]);
            conf[dst..dst + wlen].copy_from_slice(&conf_w[src..src + wlen]);
        }
        s0 = s1;
    }
    Ok((signals, conf))
}

#[cfg(test)]
mod window_tests {
    use super::{backtest_gene_batch, gather_indicator_window, signal_window_size};

    #[test]
    fn gather_indicator_window_extracts_strided_columns() {
        // 3 indicators × 5 samples, indicator-major flat: idx*5 + s.
        let flat: Vec<i32> = (0..15).collect(); // ind0=[0..5] ind1=[5..10] ind2=[10..15]
        // Window samples [1, 4) -> each indicator's [1,2,3] offset.
        let w = gather_indicator_window(&flat, 3, 5, 1, 4);
        assert_eq!(w, vec![1, 2, 3, 6, 7, 8, 11, 12, 13]);
        // Full window == original.
        assert_eq!(gather_indicator_window(&flat, 3, 5, 0, 5), flat);
    }

    #[test]
    fn small_combo_is_one_window_and_one_batch() {
        // n_samples small => the whole series fits one window / one batch.
        assert_eq!(signal_window_size(64, 50, 10), 10);
        assert_eq!(backtest_gene_batch(200, 100), 200);
    }

    #[test]
    fn huge_combo_splits_into_multiple_windows_and_batches() {
        // M1-like: 5.27M samples => sub-series windows + small gene batches.
        let w = signal_window_size(64, 50, 5_270_000);
        assert!(w >= 1 && w < 5_270_000, "got {w}");
        let b = backtest_gene_batch(200, 5_270_000);
        assert!(b >= 1 && b < 200, "got {b}");
        // The full series is covered by an integer number of windows.
        let windows = 5_270_000usize.div_ceil(w);
        assert!(windows >= 2);
    }
}

fn launch_signal_kernel<F, R: Runtime>(
    client: &ComputeClient<R>,
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
) -> Result<(Vec<i32>, Vec<f32>)>
where
    F: Float + CubeElement,
{
    let total = n_genes.saturating_mul(n_samples);
    if total == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    // **CRITICAL for real-money trading**: validate every kernel input
    // BEFORE handing buffers to the GPU. Bad GA-evolved indices silently
    // read garbage memory in CUDA, producing wrong trading signals with
    // no error path. The check is O(n_genes + total_entries) — negligible
    // next to the kernel's own work.
    validate_signal_kernel_inputs(
        indicators_flat,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        smc_data,
        gene_smc_flags,
        smc_weights,
        n_genes,
        n_samples,
    )
    .context("signal kernel input validation failed")?;

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
    let conf_handle = client.empty(total.saturating_mul(std::mem::size_of::<f32>()));

    let units = signal_kernel_units(client);
    let cubes = (total as u32).div_ceil(units);
    // cubecl 0.10: `from_raw_parts(handle, len)` takes the Handle BY VALUE (no
    // generic, no vectorization arg), so clone each (cheap, Arc-backed) to keep
    // the originals alive for the read-back. Scalars are passed as raw values
    // (the 0.10 `LaunchArg for T` impl, replacing 0.9's `ScalarArg::new`). The
    // generated `launch` is infallible (returns `()`), so no `.context()?`.
    synthesize_signals_kernel::launch::<F, R>(
        client,
        CubeCount::Static(cubes, 1, 1),
        CubeDim::new_1d(units),
        unsafe { ArrayArg::from_raw_parts(indicators_handle.clone(), indicators_flat.len()) },
        unsafe { ArrayArg::from_raw_parts(gene_offsets_handle.clone(), gene_offsets.len()) },
        unsafe { ArrayArg::from_raw_parts(gene_indices_handle.clone(), gene_indices.len()) },
        unsafe { ArrayArg::from_raw_parts(gene_weights_handle.clone(), gene_weights.len()) },
        unsafe { ArrayArg::from_raw_parts(long_thr_handle.clone(), long_thr.len()) },
        unsafe { ArrayArg::from_raw_parts(short_thr_handle.clone(), short_thr.len()) },
        unsafe { ArrayArg::from_raw_parts(smc_data_handle.clone(), smc_data.len()) },
        unsafe { ArrayArg::from_raw_parts(gene_smc_flags_handle.clone(), gene_smc_flags.len()) },
        unsafe { ArrayArg::from_raw_parts(smc_weights_handle.clone(), smc_weights.len()) },
        unsafe { ArrayArg::from_raw_parts(output_handle.clone(), total) },
        unsafe { ArrayArg::from_raw_parts(conf_handle.clone(), total) },
        n_samples as u32,
        gate_threshold,
    );

    let bytes = client.read_one_unchecked(output_handle);
    let conf_bytes = client.read_one_unchecked(conf_handle);
    Ok((
        i32::from_bytes(&bytes).to_vec(),
        f32::from_bytes(&conf_bytes).to_vec(),
    ))
}

// Signal-only GPU path: kept for a future "GPU signals + CPU backtest" hybrid
// lane. The current hybrid uses the full-eval kernel, so these are unused today.
#[allow(dead_code)]
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
    device_override: Option<usize>,
) -> Result<(Vec<i32>, Vec<f32>)> {
    let n_genes = long_thr.len();
    let n_samples = indicators.ncols();
    if n_genes == 0 || n_samples == 0 {
        return Ok((Vec::new(), Vec::new()));
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

    let client = create_gpu_client(device_override)?;

    let indicators_flat = indicators.iter().copied().collect::<Vec<_>>();
    let smc_data_flat = flatten_i32_rows(smc_data);
    let gene_smc_flags_flat = flatten_i32_flags(gene_smc_flags);
    let n_indicators = indicators.nrows();
    let precision = requested_eval_precision();

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

        match windowed_signal_synth::<bf16, _>(
            &client,
            &indicators_bf16,
            n_indicators,
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
            Ok(pair) => return Ok(pair),
            Err(err) => {
                tracing::debug!(
                    "cuda evaluator bf16 signal kernel unavailable, falling back to fp32: {err}"
                );
            }
        }
    }

    windowed_signal_synth::<f32, _>(
        &client,
        &indicators_flat,
        n_indicators,
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

#[allow(dead_code)] // signal-only GPU path; see materialize_i8_rows note above
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
    let (flat, _conf) = try_generate_signal_flat_cuda(
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
        None,
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

fn launch_backtest_kernel<R: Runtime>(
    client: &ComputeClient<R>,
    close_pips: &[f32],
    high_pips: &[f32],
    low_pips: &[f32],
    signals_flat: &[i32],
    confidences_flat: &[f32],
    timestamp_deltas_ms: &[i32],
    use_timestamps: bool,
    month_idx: &[i32],
    day_idx: &[i32],
    sl_pips: &[f32],
    tp_pips: &[f32],
    settings: &BacktestSettings,
    month_capacity: usize,
) -> Result<(Vec<f32>, Vec<i32>, Vec<f32>, Vec<i32>, Vec<f32>)> {
    let n_samples = close_pips.len();
    let n_genes = sl_pips.len();
    if n_samples == 0 || n_genes == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()));
    }
    if high_pips.len() != n_samples
        || low_pips.len() != n_samples
        || timestamp_deltas_ms.len() != n_samples
        || month_idx.len() != n_samples
        || day_idx.len() != n_samples
        || tp_pips.len() != n_genes
        || signals_flat.len() != n_genes.saturating_mul(n_samples)
        || confidences_flat.len() != n_genes.saturating_mul(n_samples)
    {
        bail!("cuda evaluator backtest kernel received inconsistent dimensions");
    }

    let close_handle = client.create_from_slice(f32::as_bytes(close_pips));
    let high_handle = client.create_from_slice(f32::as_bytes(high_pips));
    let low_handle = client.create_from_slice(f32::as_bytes(low_pips));
    let signals_handle = client.create_from_slice(i32::as_bytes(signals_flat));
    let conf_handle = client.create_from_slice(f32::as_bytes(confidences_flat));
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
    let month_start_eq_handle =
        client.empty(monthly_len.saturating_mul(std::mem::size_of::<f32>()));
    let month_counts_handle = client.empty(n_genes.saturating_mul(std::mem::size_of::<i32>()));

    let units = backtest_kernel_units(client);
    let cubes = (n_genes as u32).div_ceil(units);
    // cubecl 0.10 migration: Handle-by-value `from_raw_parts(handle, len)`
    // (clone to keep originals for read-back), raw-value scalars (no
    // `ScalarArg::new`), infallible `launch` (no `.context()?`).
    backtest_population_kernel::launch::<R>(
        client,
        CubeCount::Static(cubes, 1, 1),
        CubeDim::new_1d(units),
        unsafe { ArrayArg::from_raw_parts(close_handle.clone(), n_samples) },
        unsafe { ArrayArg::from_raw_parts(high_handle.clone(), n_samples) },
        unsafe { ArrayArg::from_raw_parts(low_handle.clone(), n_samples) },
        unsafe { ArrayArg::from_raw_parts(signals_handle.clone(), signals_flat.len()) },
        unsafe { ArrayArg::from_raw_parts(timestamp_delta_handle.clone(), n_samples) },
        unsafe { ArrayArg::from_raw_parts(month_handle.clone(), month_idx.len()) },
        unsafe { ArrayArg::from_raw_parts(day_handle.clone(), day_idx.len()) },
        unsafe { ArrayArg::from_raw_parts(sl_handle.clone(), sl_pips.len()) },
        unsafe { ArrayArg::from_raw_parts(tp_handle.clone(), tp_pips.len()) },
        unsafe { ArrayArg::from_raw_parts(metrics_handle.clone(), metrics_len) },
        unsafe { ArrayArg::from_raw_parts(trade_counts_handle.clone(), n_genes) },
        unsafe { ArrayArg::from_raw_parts(monthly_handle.clone(), monthly_len) },
        unsafe { ArrayArg::from_raw_parts(month_counts_handle.clone(), n_genes) },
        n_samples as u32,
        month_capacity as u32,
        settings.initial_equity() as f32,
        settings.max_hold_bars as u32,
        settings.min_hold_bars as u32,
        settings.max_trades_per_day as u32,
        saturating_i32(settings.gap_threshold_ms),
        if use_timestamps { 1i32 } else { 0i32 },
        if settings.trailing_enabled { 1i32 } else { 0i32 },
        settings.trailing_atr_multiplier as f32,
        settings.trailing_be_trigger_r as f32,
        settings.spread_pips as f32,
        settings.commission_per_trade as f32,
        settings.pip_value_per_lot as f32,
        // Phase C.3 (2026-05-28) — broker-supplied carry costs.
        settings.swap_long_pips_per_day as f32,
        settings.swap_short_pips_per_day as f32,
        settings.pnl_conversion_fee_rate as f32,
        // Phase 2 (2026-06-06) — confidence-scaled risk-based sizing knobs +
        // per-month start-equity output for slot-7. ORDER MUST MATCH the kernel
        // signature appended after pnl_conversion_fee_rate.
        unsafe { ArrayArg::from_raw_parts(conf_handle.clone(), confidences_flat.len()) },
        if settings.risk_based_sizing { 1i32 } else { 0i32 },
        settings.risk_per_trade_min as f32,
        settings.risk_per_trade_max as f32,
        settings.high_quality_confidence as f32,
        unsafe { ArrayArg::from_raw_parts(month_start_eq_handle.clone(), monthly_len) },
    );

    let metrics_bytes = client.read_one_unchecked(metrics_handle);
    let trade_counts_bytes = client.read_one_unchecked(trade_counts_handle);
    let monthly_bytes = client.read_one_unchecked(monthly_handle);
    let month_counts_bytes = client.read_one_unchecked(month_counts_handle);
    let month_start_eq_bytes = client.read_one_unchecked(month_start_eq_handle);

    Ok((
        f32::from_bytes(&metrics_bytes).to_vec(),
        i32::from_bytes(&trade_counts_bytes).to_vec(),
        f32::from_bytes(&monthly_bytes).to_vec(),
        i32::from_bytes(&month_counts_bytes).to_vec(),
        f32::from_bytes(&month_start_eq_bytes).to_vec(),
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
    device_override: Option<usize>,
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

    let (signals_flat, confidences_flat) = try_generate_signal_flat_cuda(
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
        device_override,
    )?;

    let client = create_gpu_client(device_override)?;
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

    // Gene-batch the backtest so each per-gene signal buffer (`B × n_samples`)
    // stays under the wgpu storage-buffer cap — huge-row TFs (M1/M3) no longer
    // OOM. Each gene is evaluated WHOLE-SERIES (the kernel is unchanged), so the
    // concatenated result is identical to a single launch — CPU↔GPU parity holds.
    let mut metrics_flat: Vec<f32> = Vec::with_capacity(n_genes * BACKTEST_CORE_METRIC_WIDTH);
    let mut trade_counts: Vec<i32> = Vec::with_capacity(n_genes);
    let mut monthly_flat: Vec<f32> = Vec::with_capacity(n_genes * month_capacity);
    let mut month_counts: Vec<i32> = Vec::with_capacity(n_genes);
    let mut month_start_eq_flat: Vec<f32> = Vec::with_capacity(n_genes * month_capacity);
    let batch = backtest_gene_batch(n_genes, n_samples);
    let mut g0 = 0usize;
    while g0 < n_genes {
        let g1 = (g0 + batch).min(n_genes);
        let (m, tc, mo, mc, mse) = launch_backtest_kernel(
            &client,
            &close_pips,
            &high_pips,
            &low_pips,
            &signals_flat[g0 * n_samples..g1 * n_samples],
            &confidences_flat[g0 * n_samples..g1 * n_samples],
            &timestamp_deltas_ms,
            use_timestamps,
            &month_idx,
            &day_idx,
            &sl_pips[g0..g1],
            &tp_pips[g0..g1],
            settings,
            month_capacity,
        )?;
        metrics_flat.extend_from_slice(&m);
        trade_counts.extend_from_slice(&tc);
        monthly_flat.extend_from_slice(&mo);
        month_counts.extend_from_slice(&mc);
        month_start_eq_flat.extend_from_slice(&mse);
        g0 = g1;
    }

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

        // Slot 7: monthly_target_hit_rate — fraction of COMPLETE months whose
        // return >= 4% of that month's STARTING equity. Computed host-side in f64
        // from the kernel's per-month PnL + start-equity buffers, byte-for-byte
        // matching the CPU (eval.rs:1110-1131): base>0 counts, no-trade months miss.
        const MONTHLY_RETURN_TARGET: f64 = 0.04;
        let mut hit = 0usize;
        let mut counted = 0usize;
        for idx in 0..month_limit {
            let base = month_start_eq_flat[month_base + idx] as f64;
            if base > 0.0 {
                counted += 1;
                if (monthly_flat[month_base + idx] as f64) / base >= MONTHLY_RETURN_TARGET {
                    hit += 1;
                }
            }
        }
        let monthly_hit = if counted > 0 {
            hit as f64 / counted as f64
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
            monthly_hit,
            trade_counts.get(g).copied().unwrap_or_default() as f64,
            consistency,
            metrics_flat[metric_base + 6] as f64,
        ]);
    }

    Ok(results)
}

const ZERO_METRICS: [f64; 11] = [0.0; 11];
