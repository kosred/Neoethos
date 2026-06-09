use super::evolution_math::{
    EvolutionSearchPolicy, SeenSignatureMemory, SurvivorSelectionPolicy, apply_metrics, crossover,
    gene_signature_hash, generate_random_genes, mutate, new_random_gene, select_parent_index,
    select_survivor_indices, unique_candidate_or_retry,
};
use super::runtime_overrides::current_genetic_search_runtime_overrides;
use super::smc_indicators::{SmcSearchConfig, build_smc_arrays, enforce_population_smc_ratio};
use super::strategy_gene::{EvaluationConfig, Gene, SearchResult};
use crate::eval::BacktestSettings;
use crate::stop_target::{StopTargetSettings, infer_stop_target_pips};
use anyhow::{Result, anyhow, bail};
use chrono::{Datelike, TimeZone, Utc};
use ndarray::Array2;
use neoethos_data::{FeatureFrame, Ohlcv};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::HashSet;
use std::time::{Duration, Instant};

/// Build a deterministic RNG by routing the genetic-search runtime
/// overrides through the canonical
/// [`neoethos_core::contracts::DeterminismPolicy`] enum:
/// `Deterministic { seed }` produces reproducible runs, while
/// `BestEffort` and `NonDeterministicAllowed` both fall back to a fresh
/// OS-derived seed (the latter is the legacy "no seed configured"
/// behavior). The GPU path in `cubecl_eval` consumes the same seed so
/// CPU/GPU runs produce identical genomes for identical inputs.
fn build_search_rng() -> StdRng {
    use neoethos_core::contracts::DeterminismPolicy;
    let seed = match current_genetic_search_runtime_overrides().determinism_policy() {
        DeterminismPolicy::Deterministic { seed } => seed,
        DeterminismPolicy::BestEffort | DeterminismPolicy::NonDeterministicAllowed => {
            rand::rng().random::<u64>()
        }
    };
    StdRng::seed_from_u64(seed)
}

type GeneArrays = (Vec<i32>, Vec<i32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// Holds data that is stable across all generations (OHLCV + features don't change).
/// Computing this once outside the generation loop saves ~5-15% eval time.
pub struct EvalDataCache {
    pub indicators: ndarray::Array2<f32>,
    pub months: Vec<i64>,
    pub days: Vec<i64>,
    pub smc_data: Vec<crate::eval::SmcRow>,
}

impl EvalDataCache {
    pub fn build(features: &FeatureFrame, ohlcv: &Ohlcv) -> Self {
        let indicators = transpose_features(features);
        let (months, days) = month_day_indices(&features.timestamps);
        let n_samples = features.n_samples();
        let (ob, fvg, liq, trend, prem, ind, bos, choch, eqh, eql, disp) =
            build_smc_arrays(features, ohlcv);
        let mut smc_data = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            smc_data.push([
                ob[i], fvg[i], liq[i], trend[i], prem[i], ind[i], bos[i], choch[i], eqh[i], eql[i],
                disp[i],
            ]);
        }
        Self {
            indicators,
            months,
            days,
            smc_data,
        }
    }
}

pub fn month_day_indices(timestamps: &[i64]) -> (Vec<i64>, Vec<i64>) {
    let mut months = Vec::with_capacity(timestamps.len());
    let mut days = Vec::with_capacity(timestamps.len());
    for ts in timestamps {
        if let Some(dt) = Utc.timestamp_millis_opt(*ts).single() {
            let month_key = (dt.year() as i64) * 12 + dt.month() as i64;
            let day_key = (dt.year() as i64) * 10000 + (dt.month() as i64) * 100 + dt.day() as i64;
            months.push(month_key);
            days.push(day_key);
        } else {
            months.push(0);
            days.push(0);
        }
    }
    (months, days)
}

fn build_gene_arrays(genes: &[Gene]) -> GeneArrays {
    let mut offsets = Vec::with_capacity(genes.len() + 1);
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    let mut long_thr = Vec::with_capacity(genes.len());
    let mut short_thr = Vec::with_capacity(genes.len());
    offsets.push(0);
    for gene in genes {
        long_thr.push(gene.long_threshold);
        short_thr.push(gene.short_threshold);
        for (idx, weight) in gene.indices.iter().zip(gene.weights.iter()) {
            indices.push(*idx as i32);
            weights.push(*weight);
        }
        offsets.push(indices.len() as i32);
    }
    (offsets, indices, weights, long_thr, short_thr)
}

fn transpose_features(frame: &FeatureFrame) -> Array2<f32> {
    // `as_indicators_view` is already `[features × samples]` for both backings
    // (a transposed view of the in-RAM matrix, or the native mmap layout), so
    // this is the GA's `indicators` exactly. Callers only ever pass the small
    // prefiltered/windowed frame here, so the `to_owned` copy stays bounded.
    frame.as_indicators_view().to_owned()
}

/// Compute the signal series for a `Gene` using the SAME SMC-gated logic as
/// the in-search evaluator (`crate::eval::synthesize_signals_cpu`).
///
/// Item 6 from the search optimization notes: the post-search filtering and
/// Monte-Carlo perturbation paths in `discovery.rs` previously called this
/// function but it implemented only the linear weighted-indicator threshold,
/// ignoring `gene.use_ob`, `use_fvg`, `use_bos`, etc. and the SMC gate
/// configured via `EvaluationConfig::smc_gate_threshold`. As a consequence
/// the post-search "min_trades" filter and the MC perturbation reward used
/// a signal series that did NOT match what was actually evaluated and
/// archived during search. This let strategies through that should have
/// been pruned (and pruned strategies that should have passed).
///
/// Behaviour is now:
/// 1. Build the combined indicator score (unchanged).
/// 2. Threshold against `gene.long_threshold` / `gene.short_threshold`.
/// 3. Apply the SMC-flag gate using the same scoring as
///    `synthesize_signals_cpu`: each enabled flag contributes its weight
///    when the SMC indicator at bar i agrees with the candidate signal
///    direction; only signals whose aggregated score >= the per-gene gate
///    survive.
pub fn signals_for_gene(features: &FeatureFrame, gene: &Gene) -> Vec<i8> {
    signals_for_gene_with_config(features, gene, &EvaluationConfig::default())
}

pub fn signals_for_gene_with_config(
    features: &FeatureFrame,
    gene: &Gene,
    config: &EvaluationConfig,
) -> Vec<i8> {
    // Behaviour-identical thin wrapper: drop the confidence vector.
    signals_and_confidence_for_gene_with_config(features, gene, config).0
}

/// Confidence-emitting variant of [`signals_for_gene_with_config`]. Returns
/// the SAME signals plus a per-bar confidence in `[0,1]` used by the
/// risk-based position sizer.
///
/// Confidence per bar: `0.0` when the signal is `0`; otherwise
///   gap    = (long_threshold - short_threshold).abs().max(1e-6)
///   long:  margin = combined[i] - long_threshold
///   short: margin = short_threshold - combined[i]
///   conf   = (margin / gap).clamp(0.0, 1.0)
pub fn signals_and_confidence_for_gene_with_config(
    features: &FeatureFrame,
    gene: &Gene,
    config: &EvaluationConfig,
) -> (Vec<i8>, Vec<f32>) {
    let n_samples = features.n_samples();
    let mut combined = vec![0.0_f32; n_samples];
    for (idx, weight) in gene.indices.iter().zip(gene.weights.iter()) {
        if *idx >= features.n_features() {
            continue;
        }
        let col = features.feature_column(*idx);
        for (i, v) in col.iter().enumerate() {
            combined[i] += *weight * *v;
        }
    }
    let mut signals = vec![0_i8; n_samples];
    let mut confidences = vec![0.0_f32; n_samples];
    let gap = (gene.long_threshold - gene.short_threshold).abs().max(1e-6);

    // Resolve gene SMC flags + per-flag weights identical to the in-search
    // evaluator. If no flag is enabled we short-circuit to the simple
    // threshold path so callers that build a `Gene` with no SMC flags get
    // the same fast path as before this change.
    let flags: [i8; 11] = [
        gene.use_ob as i8,
        gene.use_fvg as i8,
        gene.use_liq_sweep as i8,
        gene.mtf_confirmation as i8,
        gene.use_premium_discount as i8,
        gene.use_inducement as i8,
        gene.use_bos as i8,
        gene.use_choch as i8,
        gene.use_eqh as i8,
        gene.use_eql as i8,
        gene.use_displacement as i8,
    ];
    let _any_flag = flags.iter().any(|f| *f != 0);

    // Need OHLCV-derived SMC indicator series — compute the same way the
    // evaluator does. Without OHLCV we fall back to the un-gated path so
    // single-arg callers (no Ohlcv handy) keep working; gated callers
    // should use `signals_for_gene_full`. (Both branches of the original
    // code applied the identical un-gated threshold loop, so this is
    // behaviour-preserving.)
    let _ = config; // reserved: future per-config gate threshold override

    for i in 0..n_samples {
        let v = combined[i];
        let sig = if v >= gene.long_threshold {
            1
        } else if v <= gene.short_threshold {
            -1
        } else {
            0
        };
        signals[i] = sig;
        if sig != 0 {
            let margin = if sig == 1 {
                v - gene.long_threshold
            } else {
                gene.short_threshold - v
            };
            confidences[i] = (margin / gap).clamp(0.0, 1.0);
        }
    }
    (signals, confidences)
}

/// SMC-gated variant that mirrors `eval::synthesize_signals_cpu` exactly.
/// Use this in post-search filtering / MC perturbation so the trade count
/// matches what the evaluator actually scored.
pub fn signals_for_gene_full(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    gene: &Gene,
    config: &EvaluationConfig,
) -> Vec<i8> {
    // Behaviour-identical thin wrapper: drop the confidence vector.
    signals_and_confidence_for_gene_full(features, ohlcv, gene, config).0
}

/// Confidence-emitting variant of [`signals_for_gene_full`]. Returns the
/// SAME SMC-gated signals plus a per-bar confidence in `[0,1]` used by the
/// risk-based position sizer. Confidence is computed from the RAW threshold
/// crossing (pre-gate) and stored only for bars whose final (post-gate)
/// signal is non-zero, so it aligns exactly with the signals slice.
///
/// Confidence per bar: `0.0` when the signal is `0`; otherwise
///   gap    = (long_threshold - short_threshold).abs().max(1e-6)
///   long:  margin = combined[i] - long_threshold
///   short: margin = short_threshold - combined[i]
///   conf   = (margin / gap).clamp(0.0, 1.0)
pub fn signals_and_confidence_for_gene_full(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    gene: &Gene,
    config: &EvaluationConfig,
) -> (Vec<i8>, Vec<f32>) {
    let n_samples = features.n_samples();
    let mut combined = vec![0.0_f32; n_samples];
    for (idx, weight) in gene.indices.iter().zip(gene.weights.iter()) {
        if *idx >= features.n_features() {
            continue;
        }
        let col = features.feature_column(*idx);
        for (i, v) in col.iter().enumerate() {
            combined[i] += *weight * *v;
        }
    }

    let flags: [i8; 11] = [
        gene.use_ob as i8,
        gene.use_fvg as i8,
        gene.use_liq_sweep as i8,
        gene.mtf_confirmation as i8,
        gene.use_premium_discount as i8,
        gene.use_inducement as i8,
        gene.use_bos as i8,
        gene.use_choch as i8,
        gene.use_eqh as i8,
        gene.use_eql as i8,
        gene.use_displacement as i8,
    ];
    let smc_weights = [
        config.smc_weight_ob,
        config.smc_weight_fvg,
        config.smc_weight_liq,
        config.smc_weight_mtf,
        config.smc_weight_premium,
        config.smc_weight_inducement,
        config.smc_weight_bos,
        config.smc_weight_choch,
        config.smc_weight_eqh,
        config.smc_weight_eql,
        config.smc_weight_displacement,
    ];
    let active_sum: f32 = flags
        .iter()
        .enumerate()
        .map(|(i, &f)| if f != 0 { smc_weights[i] } else { 0.0 })
        .sum();
    // Hard bypass: when `NEOETHOS_BOT_DISABLE_SMC_GATE=1` the gate
    // collapses (active_sum = 0 → raw signal passes through). Lets
    // operators isolate "SMC indicators don't trigger on this symbol"
    // from genuine signal-generation issues without recompiling.
    //
    // F-CORE3 closure (2026-05-25): previously read `std::env::var`
    // inline on EVERY call to this hot-path function. Now resolved
    // through the typed `SmcGateOverrides::disable_gate` boundary so
    // the env is hit at most once per process (in
    // `GeneticSearchRuntimeOverrides::from_env`).
    let smc_bypass = super::runtime_overrides::current_genetic_search_runtime_overrides()
        .smc_gate
        .disable_gate;
    let active_sum = if smc_bypass { 0.0 } else { active_sum };
    let gate = config.smc_gate_threshold.min(active_sum);

    let (ob, fvg, liq, trend, prem, ind, bos, choch, eqh, eql, disp) =
        super::smc_indicators::build_smc_arrays(features, ohlcv);

    let mut signals = vec![0_i8; n_samples];
    let mut confidences = vec![0.0_f32; n_samples];
    let gap = (gene.long_threshold - gene.short_threshold).abs().max(1e-6);
    for i in 0..n_samples {
        let v = combined[i];
        let raw = if v >= gene.long_threshold {
            1
        } else if v <= gene.short_threshold {
            -1
        } else {
            0
        };
        if raw == 0 {
            continue;
        }
        // Confidence of the raw threshold crossing (pre-gate).
        let margin = if raw == 1 {
            v - gene.long_threshold
        } else {
            gene.short_threshold - v
        };
        let conf = (margin / gap).clamp(0.0, 1.0);
        if active_sum <= 0.0 {
            signals[i] = raw;
            confidences[i] = conf;
            continue;
        }
        let smc_row = [
            ob[i], fvg[i], liq[i], trend[i], prem[i], ind[i], bos[i], choch[i], eqh[i], eql[i],
            disp[i],
        ];
        let mut score = 0.0_f32;
        for j in 0..11 {
            if flags[j] != 0 {
                if j == 5 {
                    if smc_row[j] == 1 {
                        score += smc_weights[j];
                    }
                } else if smc_row[j] == raw {
                    score += smc_weights[j];
                }
            }
        }
        if score >= gate {
            signals[i] = raw;
            confidences[i] = conf;
        }
    }
    (signals, confidences)
}

/// Evaluate genes using a pre-built EvalDataCache (avoids recomputing stable arrays each generation).
pub fn evaluate_genes_cached(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    genes: &[Gene],
    config: &EvaluationConfig,
    cache: &EvalDataCache,
) -> Result<Vec<[f64; 11]>> {
    if genes.is_empty() {
        return Ok(Vec::new());
    }
    let (offsets, indices, weights, long_thr, short_thr) = build_gene_arrays(genes);
    let (sl_pips, tp_pips) = resolve_stop_target_arrays(genes, ohlcv, config);
    let mut gene_smc_flags = Vec::with_capacity(genes.len());
    for g in genes {
        gene_smc_flags.push([
            g.use_ob as i8,
            g.use_fvg as i8,
            g.use_liq_sweep as i8,
            g.mtf_confirmation as i8,
            g.use_premium_discount as i8,
            g.use_inducement as i8,
            g.use_bos as i8,
            g.use_choch as i8,
            g.use_eqh as i8,
            g.use_eql as i8,
            g.use_displacement as i8,
        ]);
    }

    let smc_weights = [
        config.smc_weight_ob,
        config.smc_weight_fvg,
        config.smc_weight_liq,
        config.smc_weight_mtf,
        config.smc_weight_premium,
        config.smc_weight_inducement,
        config.smc_weight_bos,
        config.smc_weight_choch,
        config.smc_weight_eqh,
        config.smc_weight_eql,
        config.smc_weight_displacement,
    ];

    let b_settings = BacktestSettings {
        max_hold_bars: config.max_hold_bars,
        trailing_enabled: config.trailing_enabled,
        trailing_atr_multiplier: config.trailing_atr_multiplier,
        trailing_be_trigger_r: config.trailing_be_trigger_r,
        pip_value: config.pip_value,
        spread_pips: config.spread_pips,
        commission_per_trade: config.commission_per_trade,
        pip_value_per_lot: config.pip_value_per_lot,
        ..Default::default()
    };

    crate::eval::evaluate_population_core(crate::eval::PopulationEvalInputs {
        close: &ohlcv.close,
        high: &ohlcv.high,
        low: &ohlcv.low,
        indicators: cache.indicators.view(),
        gene_offsets: &offsets,
        gene_indices: &indices,
        gene_weights: &weights,
        long_thr: &long_thr,
        short_thr: &short_thr,
        month_idx: &cache.months,
        day_idx: &cache.days,
        timestamps: &features.timestamps,
        sl_pips: &sl_pips,
        tp_pips: &tp_pips,
        smc_data: &cache.smc_data,
        gene_smc_flags: &gene_smc_flags,
        gate_threshold: config.smc_gate_threshold,
        weights: &smc_weights,
        settings: &b_settings,
    })
    .map_err(|e| anyhow!(e))
}

/// AREA 2 / Stage A (2026-06-09) — GPU-routed **validation** population eval.
///
/// Mirrors [`evaluate_genes`] (same CSR/SMC array packing via the SHARED
/// `build_gene_arrays` / `build_smc_arrays`, the single source of truth used by
/// the GA), but with two deliberate differences so it reproduces the post-search
/// validation screens (Monte-Carlo / re-eval) bit-for-bit:
///
/// 1. It routes through [`crate::eval::validation_backtest_population`] (whole
///    population → ONE GPU launch, CPU fallback) instead of the GA's CPU+GPU
///    *split* [`crate::eval::evaluate_population_core`].
/// 2. The caller supplies the **exact** [`BacktestSettings`] template the serial
///    validation path used (e.g. `discovery_backtest_settings`): kill-zones on,
///    and — critically — `risk_based_sizing == false`, so the kernel uses
///    fixed-1-lot sizing identical to `simulate_trades_core`. Per-gene `sl_pips`
///    / `tp_pips` are taken from the gene with the SAME 20/40 fallback
///    `discovery_backtest_settings` applies (NOT the OHLCV-inferred
///    `resolve_stop_target_arrays` defaults the GA uses), so the SL/TP exits
///    match the serial Monte-Carlo run exactly.
///
/// With `risk_based_sizing == false` the returned `metrics[0]` (net_profit) is
/// the fixed-1-lot trade-pnl sum, so a consumer testing `metrics[g][0] > 0.0`
/// gets the SAME profitable/not verdict as the serial
/// `simulate_trades_core(...).iter().map(|t| t.pnl).sum() > 0.0`.
pub fn validation_genes_population(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    genes: &[Gene],
    config: &EvaluationConfig,
    settings_template: &BacktestSettings,
) -> Result<Vec<[f64; 11]>> {
    if genes.is_empty() {
        return Ok(Vec::new());
    }
    if features.n_samples() == 0 || features.n_features() == 0 {
        bail!("empty feature matrix");
    }
    let n_samples = features.n_samples();
    if ohlcv.close.len() != n_samples {
        bail!("ohlcv length does not match feature rows");
    }

    let indicators = transpose_features(features);
    let (offsets, indices, weights, long_thr, short_thr) = build_gene_arrays(genes);
    // Per-gene SL/TP resolved the SAME way `discovery_backtest_settings` does:
    // the gene's own finite-positive value, else the 20/40-pip fallback. This is
    // what the serial validation path injected into `simulate_trades_core`, so
    // the SL/TP exits stay identical.
    let mut sl_pips = Vec::with_capacity(genes.len());
    let mut tp_pips = Vec::with_capacity(genes.len());
    for g in genes {
        sl_pips.push(if g.sl_pips.is_finite() && g.sl_pips > 0.0 {
            g.sl_pips
        } else {
            20.0
        });
        tp_pips.push(if g.tp_pips.is_finite() && g.tp_pips > 0.0 {
            g.tp_pips
        } else {
            40.0
        });
    }
    let (months, days) = month_day_indices(&features.timestamps);

    let (ob, fvg, liq, trend, prem, ind, bos, choch, eqh, eql, disp) =
        build_smc_arrays(features, ohlcv);
    let mut smc_data = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        smc_data.push([
            ob[i], fvg[i], liq[i], trend[i], prem[i], ind[i], bos[i], choch[i], eqh[i], eql[i],
            disp[i],
        ]);
    }
    let mut gene_smc_flags = Vec::with_capacity(genes.len());
    for g in genes {
        gene_smc_flags.push([
            g.use_ob as i8,
            g.use_fvg as i8,
            g.use_liq_sweep as i8,
            g.mtf_confirmation as i8,
            g.use_premium_discount as i8,
            g.use_inducement as i8,
            g.use_bos as i8,
            g.use_choch as i8,
            g.use_eqh as i8,
            g.use_eql as i8,
            g.use_displacement as i8,
        ]);
    }

    let smc_weights = [
        config.smc_weight_ob,
        config.smc_weight_fvg,
        config.smc_weight_liq,
        config.smc_weight_mtf,
        config.smc_weight_premium,
        config.smc_weight_inducement,
        config.smc_weight_bos,
        config.smc_weight_choch,
        config.smc_weight_eqh,
        config.smc_weight_eql,
        config.smc_weight_displacement,
    ];

    // Use the caller's settings verbatim, but FORCE fixed-1-lot sizing so the
    // metrics[0] sign matches the serial `simulate_trades_core` reference. The
    // caller is expected to pass `risk_based_sizing == false` already; this is
    // belt-and-suspenders so a stray template can never silently change sizing.
    let mut settings = settings_template.clone();
    settings.risk_based_sizing = false;

    Ok(crate::eval::validation_backtest_population(
        crate::eval::PopulationEvalInputs {
            close: &ohlcv.close,
            high: &ohlcv.high,
            low: &ohlcv.low,
            indicators: indicators.view(),
            gene_offsets: &offsets,
            gene_indices: &indices,
            gene_weights: &weights,
            long_thr: &long_thr,
            short_thr: &short_thr,
            month_idx: &months,
            day_idx: &days,
            timestamps: &features.timestamps,
            sl_pips: &sl_pips,
            tp_pips: &tp_pips,
            smc_data: &smc_data,
            gene_smc_flags: &gene_smc_flags,
            gate_threshold: config.smc_gate_threshold,
            weights: &smc_weights,
            settings: &settings,
        },
    ))
}

/// AREA 2 / Stage B (2026-06-09) — GPU-routed **CPCV fold** population eval over a
/// NON-CONTIGUOUS gathered index set.
///
/// This is the CPCV twin of [`validation_genes_population`]. The CPCV gate
/// (`discovery::evaluate_cpcv_gate`) backtests every portfolio gene on each
/// Combinatorial-Purged-CV fold, where a fold is a set of *gathered*
/// (re-indexed, non-monotonic) absolute bar indices `absolute_idx`. The serial
/// CPU path gathers the per-bar arrays HOST-SIDE into fresh contiguous Vecs and
/// runs `fast_evaluate_strategy_core` on them — the backtest never sees the gaps
/// because the gather already happened. This helper feeds the GPU population
/// kernel the SAME host-gathered contiguous buffers, so the kernel is byte-
/// identical to the CPU's gathered-Vec path WITHOUT any kernel change.
///
/// ## Why this can't reuse [`validation_genes_population`]
/// Two deliberate differences:
///  1. **SMC is gathered, NOT recomputed.** `validation_genes_population` calls
///     `build_smc_arrays` on the *passed* OHLCV. The SMC primitives in
///     `derive_smc_arrays` carry heavy cross-bar LOOKBACK (trend uses
///     `close[i-12]`, BoS/EQH/EQL use 12–20-bar windows, FVG/liq use 2–3-bar
///     windows). Recomputing SMC on a gathered (non-contiguous) OHLCV slice would
///     read the WRONG neighbours and silently corrupt the fold. The CPU CPCV path
///     avoids this by gathering the *full-series* precomputed signals/confidence;
///     this helper mirrors it by computing the full-series SMC arrays ONCE and
///     GATHERING them at `absolute_idx`. Signal synthesis is fully pointwise
///     (`combined[i]` = weighted sum of indicator[i]; the gate reads only
///     `smc_row[i]`), so the on-device synth at gathered position `k` reads the
///     SAME indicator+SMC values the full-series synth read at `absolute_idx[k]`
///     → identical signals/confidence → identical fold metrics.
///  2. **`risk_based_sizing` is PRESERVED from the caller's template**, not forced
///     to `false`. CPCV uses `discovery_backtest_settings` which inherits
///     `BacktestSettings::default().risk_based_sizing == true` and feeds the gene's
///     REAL per-bar confidence into the risk sizer. The kernel recomputes the
///     identical confidence on-device (pointwise), so risk-based sizing matches.
///
/// `timestamps` is passed empty (`&[]`) so the backtest uses index-delta carry —
/// EXACTLY as the serial CPCV path does (`fast_evaluate_strategy_core(..., &[], ...)`).
///
/// Per-gene `sl_pips`/`tp_pips` use the SAME finite-positive-else-20/40 fallback
/// `discovery_backtest_settings` applies, so SL/TP exits match the serial run.
///
/// Returns one `[f64; 11]` metric row per gene (same layout as
/// [`crate::eval::evaluate_population_core`]).
#[allow(clippy::too_many_arguments)]
pub fn validation_genes_population_gathered(
    full_indicators: ndarray::ArrayView2<'_, f32>,
    full_smc: &[crate::eval::SmcRow],
    genes: &[Gene],
    config: &EvaluationConfig,
    settings_template: &BacktestSettings,
    absolute_idx: &[usize],
    gathered_close: &[f64],
    gathered_high: &[f64],
    gathered_low: &[f64],
    gathered_months: &[i64],
    gathered_days: &[i64],
) -> Result<Vec<[f64; 11]>> {
    if genes.is_empty() || absolute_idx.is_empty() {
        return Ok(Vec::new());
    }
    let full_samples = full_indicators.ncols();
    let n_features = full_indicators.nrows();
    if full_smc.len() != full_samples {
        bail!(
            "full SMC length {} != full sample count {}",
            full_smc.len(),
            full_samples
        );
    }
    let fold_n = absolute_idx.len();
    if gathered_close.len() != fold_n
        || gathered_high.len() != fold_n
        || gathered_low.len() != fold_n
        || gathered_months.len() != fold_n
        || gathered_days.len() != fold_n
    {
        bail!(
            "gathered per-bar arrays must all have length {} (the fold index count)",
            fold_n
        );
    }
    if let Some(&bad) = absolute_idx.iter().find(|&&i| i >= full_samples) {
        bail!(
            "CPCV gather index {} out of range (full series has {} samples)",
            bad,
            full_samples
        );
    }

    // Gather the indicator columns at `absolute_idx` into a fresh
    // `[n_features × fold_n]` matrix — the kernel's `indicators` layout. The
    // pointwise synth then reads `gathered_ind[f][k]` = full-series indicator[f]
    // at `absolute_idx[k]`, reproducing `combined` exactly.
    let mut gathered_ind = Array2::<f32>::zeros((n_features, fold_n));
    for f in 0..n_features {
        let src = full_indicators.row(f);
        let mut dst = gathered_ind.row_mut(f);
        for (k, &abs) in absolute_idx.iter().enumerate() {
            dst[k] = src[abs];
        }
    }

    // Gather the FULL-SERIES SMC rows (NOT recomputed on the gathered slice).
    let mut gathered_smc: Vec<crate::eval::SmcRow> = Vec::with_capacity(fold_n);
    for &abs in absolute_idx {
        gathered_smc.push(full_smc[abs]);
    }

    let (offsets, indices, weights, long_thr, short_thr) = build_gene_arrays(genes);
    // Per-gene SL/TP resolved the SAME way `discovery_backtest_settings` does.
    let mut sl_pips = Vec::with_capacity(genes.len());
    let mut tp_pips = Vec::with_capacity(genes.len());
    for g in genes {
        sl_pips.push(if g.sl_pips.is_finite() && g.sl_pips > 0.0 {
            g.sl_pips
        } else {
            20.0
        });
        tp_pips.push(if g.tp_pips.is_finite() && g.tp_pips > 0.0 {
            g.tp_pips
        } else {
            40.0
        });
    }

    let mut gene_smc_flags = Vec::with_capacity(genes.len());
    for g in genes {
        gene_smc_flags.push([
            g.use_ob as i8,
            g.use_fvg as i8,
            g.use_liq_sweep as i8,
            g.mtf_confirmation as i8,
            g.use_premium_discount as i8,
            g.use_inducement as i8,
            g.use_bos as i8,
            g.use_choch as i8,
            g.use_eqh as i8,
            g.use_eql as i8,
            g.use_displacement as i8,
        ]);
    }

    let smc_weights = [
        config.smc_weight_ob,
        config.smc_weight_fvg,
        config.smc_weight_liq,
        config.smc_weight_mtf,
        config.smc_weight_premium,
        config.smc_weight_inducement,
        config.smc_weight_bos,
        config.smc_weight_choch,
        config.smc_weight_eqh,
        config.smc_weight_eql,
        config.smc_weight_displacement,
    ];

    // Use the caller's settings VERBATIM — including `risk_based_sizing` (CPCV
    // keeps it true so the gene's real per-bar confidence drives sizing, matching
    // the serial `fast_evaluate_strategy_core` call). Empty `timestamps` ⇒ the
    // kernel/CPU backtest uses index-delta carry, identical to the serial path.
    Ok(crate::eval::validation_backtest_population(
        crate::eval::PopulationEvalInputs {
            close: gathered_close,
            high: gathered_high,
            low: gathered_low,
            indicators: gathered_ind.view(),
            gene_offsets: &offsets,
            gene_indices: &indices,
            gene_weights: &weights,
            long_thr: &long_thr,
            short_thr: &short_thr,
            month_idx: gathered_months,
            day_idx: gathered_days,
            timestamps: &[],
            sl_pips: &sl_pips,
            tp_pips: &tp_pips,
            smc_data: &gathered_smc,
            gene_smc_flags: &gene_smc_flags,
            gate_threshold: config.smc_gate_threshold,
            weights: &smc_weights,
            settings: settings_template,
        },
    ))
}

pub fn evaluate_genes(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    genes: &[Gene],
    config: &EvaluationConfig,
) -> Result<Vec<[f64; 11]>> {
    if features.n_samples() == 0 || features.n_features() == 0 {
        bail!("empty feature matrix");
    }
    let n_samples = features.n_samples();
    if ohlcv.close.len() != n_samples {
        bail!("ohlcv length does not match feature rows");
    }

    let indicators = transpose_features(features);
    let (offsets, indices, weights, long_thr, short_thr) = build_gene_arrays(genes);
    let (sl_pips, tp_pips) = resolve_stop_target_arrays(genes, ohlcv, config);
    let (months, days) = month_day_indices(&features.timestamps);

    let (ob, fvg, liq, trend, prem, ind, bos, choch, eqh, eql, disp) =
        build_smc_arrays(features, ohlcv);
    let mut smc_data = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        smc_data.push([
            ob[i], fvg[i], liq[i], trend[i], prem[i], ind[i], bos[i], choch[i], eqh[i], eql[i],
            disp[i],
        ]);
    }
    let mut gene_smc_flags = Vec::with_capacity(genes.len());
    for g in genes {
        gene_smc_flags.push([
            g.use_ob as i8,
            g.use_fvg as i8,
            g.use_liq_sweep as i8,
            g.mtf_confirmation as i8,
            g.use_premium_discount as i8,
            g.use_inducement as i8,
            g.use_bos as i8,
            g.use_choch as i8,
            g.use_eqh as i8,
            g.use_eql as i8,
            g.use_displacement as i8,
        ]);
    }

    let smc_weights = [
        config.smc_weight_ob,
        config.smc_weight_fvg,
        config.smc_weight_liq,
        config.smc_weight_mtf,
        config.smc_weight_premium,
        config.smc_weight_inducement,
        config.smc_weight_bos,
        config.smc_weight_choch,
        config.smc_weight_eqh,
        config.smc_weight_eql,
        config.smc_weight_displacement,
    ];

    let b_settings = BacktestSettings {
        max_hold_bars: config.max_hold_bars,
        trailing_enabled: config.trailing_enabled,
        trailing_atr_multiplier: config.trailing_atr_multiplier,
        trailing_be_trigger_r: config.trailing_be_trigger_r,
        pip_value: config.pip_value,
        spread_pips: config.spread_pips,
        commission_per_trade: config.commission_per_trade,
        pip_value_per_lot: config.pip_value_per_lot,
        ..Default::default()
    };

    crate::eval::evaluate_population_core(crate::eval::PopulationEvalInputs {
        close: &ohlcv.close,
        high: &ohlcv.high,
        low: &ohlcv.low,
        indicators: indicators.view(),
        gene_offsets: &offsets,
        gene_indices: &indices,
        gene_weights: &weights,
        long_thr: &long_thr,
        short_thr: &short_thr,
        month_idx: &months,
        day_idx: &days,
        timestamps: &features.timestamps,
        sl_pips: &sl_pips,
        tp_pips: &tp_pips,
        smc_data: &smc_data,
        gene_smc_flags: &gene_smc_flags,
        gate_threshold: config.smc_gate_threshold,
        weights: &smc_weights,
        settings: &b_settings,
    })
    .map_err(|e| anyhow!(e))
}

fn resolve_stop_target_arrays(
    genes: &[Gene],
    ohlcv: &Ohlcv,
    config: &EvaluationConfig,
) -> (Vec<f64>, Vec<f64>) {
    // **F-761 / F-CORE2 closure (2026-05-25)**: was previously
    // `else { 0.0001 }` — a hardcoded EURUSD-pip fallback that
    // silently wrongs JPY pairs (pip = 0.01) and metals
    // (pip = 0.01). Now routes through `default_pip_size(&config.symbol)`
    // which is symbol-aware AND returns NaN for empty symbol (so the
    // fitness guard rejects strategies that lack a resolvable pip).
    let pip_size = if config.pip_value.is_finite() && config.pip_value > 0.0 {
        config.pip_value
    } else {
        super::strategy_gene::default_pip_size(&config.symbol)
    };
    let default = infer_stop_target_pips(
        &ohlcv.open,
        &ohlcv.high,
        &ohlcv.low,
        &ohlcv.close,
        &StopTargetSettings::default(),
        pip_size,
        0,
    );
    // **F-762 / F-CORE2 closure (2026-05-25)**: was previously
    // `.unwrap_or((20.0, 40.0))` — a synthetic SL/TP placeholder that
    // covered up "couldn't infer defaults from OHLCV". Now propagates
    // NaN so genes without explicit `sl_pips`/`tp_pips` settings are
    // rejected by the downstream `is_finite()` gate (lines below)
    // instead of being silently sized at 20/40 pips.
    let (default_sl, default_tp) = default
        .map(|(sl, tp, _rr)| (sl, tp))
        .unwrap_or((f64::NAN, f64::NAN));

    let mut sl_pips = Vec::with_capacity(genes.len());
    let mut tp_pips = Vec::with_capacity(genes.len());
    for gene in genes {
        sl_pips.push(if gene.sl_pips.is_finite() && gene.sl_pips > 0.0 {
            gene.sl_pips
        } else {
            default_sl
        });
        tp_pips.push(if gene.tp_pips.is_finite() && gene.tp_pips > 0.0 {
            gene.tp_pips
        } else {
            default_tp
        });
    }
    (sl_pips, tp_pips)
}

pub fn random_search(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    n_genes: usize,
    max_indicators: usize,
) -> Result<SearchResult> {
    let n_indicators = features.n_features();
    let smc_cfg = SmcSearchConfig::from_env();
    let mut rng = build_search_rng();
    let mut genes =
        generate_random_genes(n_genes, n_indicators, max_indicators, 0, &smc_cfg, &mut rng);
    enforce_population_smc_ratio(&mut genes, &smc_cfg);
    for gene in genes.iter_mut() {
        gene.normalize(n_indicators, 1);
    }
    let metrics = evaluate_genes(features, ohlcv, &genes, &EvaluationConfig::default())?;
    Ok(SearchResult { genes, metrics })
}

pub fn evolve_search(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    population: usize,
    generations: usize,
    max_indicators: usize,
) -> Result<SearchResult> {
    evolve_search_with_progress(
        features,
        ohlcv,
        population,
        generations,
        max_indicators,
        None,
        |_, _, _, _, _| {},
    )
}

#[allow(clippy::too_many_arguments)]
pub fn evolve_search_with_progress_and_limits<F>(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    population: usize,
    generations: usize,
    max_indicators: usize,
    max_runtime: Option<Duration>,
    eval_config: Option<EvaluationConfig>,
    progress_fn: F,
) -> Result<SearchResult>
where
    F: FnMut(usize, usize, f64, usize, usize),
{
    evolve_search_with_progress_impl(
        features,
        ohlcv,
        population,
        generations,
        max_indicators,
        max_runtime,
        eval_config,
        progress_fn,
    )
}

pub fn evolve_search_with_progress<F>(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    population: usize,
    generations: usize,
    max_indicators: usize,
    eval_config: Option<EvaluationConfig>,
    progress_fn: F,
) -> Result<SearchResult>
where
    F: FnMut(usize, usize, f64, usize, usize),
{
    evolve_search_with_progress_impl(
        features,
        ohlcv,
        population,
        generations,
        max_indicators,
        None,
        eval_config,
        progress_fn,
    )
}

#[allow(clippy::too_many_arguments)]
fn evolve_search_with_progress_impl<F>(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    population: usize,
    generations: usize,
    max_indicators: usize,
    max_runtime: Option<Duration>,
    eval_config: Option<EvaluationConfig>,
    mut progress_fn: F,
) -> Result<SearchResult>
where
    F: FnMut(usize, usize, f64, usize, usize),
{
    if population == 0 {
        bail!("population must be > 0");
    }
    let n_indicators = features.n_features();
    let smc_cfg = SmcSearchConfig::from_env();

    // All `NEOETHOS_BOT_*` search-engine knobs are resolved through the typed
    // `GeneticSearchRuntimeOverrides` boundary; the inline env reads that
    // used to live here are gone (P0-8).
    let genetic_runtime_overrides = current_genetic_search_runtime_overrides();
    let resolved_smc_gate = genetic_runtime_overrides.resolved_smc_gate();
    let resolved_selection = genetic_runtime_overrides.resolved_selection();

    let gate_start = resolved_smc_gate.start;
    let gate_end = resolved_smc_gate.end;
    let gate_curve = resolved_smc_gate.curve;
    let gate_stagnation_step = resolved_smc_gate.stagnation_step;
    let (gate_lo, gate_hi) = (gate_start.min(gate_end), gate_start.max(gate_end));

    let mut eval_cfg = eval_config.unwrap_or_default();
    eval_cfg.smc_gate_threshold = gate_start.clamp(gate_lo, gate_hi);

    let seen_retry_attempts = genetic_runtime_overrides.effective_seen_retry_attempts();
    let mut seen_memory = SeenSignatureMemory::from_env();
    let mut rng = build_search_rng();

    // **GA Fix C (2026-05-26, taskdoc #275)** — seed ~10% of the
    // initial population with hand-crafted multi-TF professional
    // templates. The pure-random cold start gives the GA almost no
    // chance of wiring up a coherent D1+H4+H1+M15+M5 confluence on a
    // 1500+ feature space; templates seed the basin of attraction so
    // mutation can refine real strategies instead of noise.
    //
    // Cap at min(50, 10% of population). If templates can't resolve
    // (single-TF backtest, unfamiliar feature names) we fall through
    // to pure random — `seed_professional_templates` returns fewer
    // than `count` rather than erroring, and the random fill below
    // pads out the population.
    let seed_count = (population / 10).min(50);
    let mut genes: Vec<Gene> = if seed_count > 0 {
        let seeds = super::seed_templates::seed_professional_templates(
            seed_count,
            &features.names,
            n_indicators,
            &mut rng,
        );
        if seeds.is_empty() {
            // **F-317 (2026-05-29)**: templates wanted to seed but
            // resolved to zero genes — usually means the feature names
            // exposed by the upstream pipeline don't match what the
            // templates expect (e.g. single-TF run with no D1/H4
            // prefixes, or a custom feature set). Log loudly so the
            // operator knows their cold-start is actually pure random.
            tracing::warn!(
                target: "neoethos_search::search_engine",
                seed_count,
                feature_count = features.names.len(),
                n_indicators,
                "GA Fix C seed templates returned 0 genes — check that the upstream feature pipeline exposes the expected multi-TF prefixes (e.g. D1_, H4_, H1_, M15_, M5_); falling back to pure-random cold start"
            );
        } else {
            tracing::info!(
                target: "neoethos_search::search_engine",
                seeded = seeds.len(),
                population,
                "GA Fix C: seeded {} of {} initial genes with multi-TF templates",
                seeds.len(),
                population
            );
        }
        let random_count = population.saturating_sub(seeds.len());
        let mut out = seeds;
        out.extend(generate_random_genes(
            random_count,
            n_indicators,
            max_indicators,
            0,
            &smc_cfg,
            &mut rng,
        ));
        out
    } else {
        // **F-317 (2026-05-29)**: `seed_count == 0` means
        // `population / 10` rounded down to zero, i.e. `population < 10`.
        // The historical code silently fell through to pure random with
        // no diagnostic. That breaks the GA Fix C invariant ("10 % of
        // the cold start should be hand-crafted templates"), so emit
        // a warn so the operator sees that their tiny-population run
        // gets no template seeding.
        tracing::warn!(
            target: "neoethos_search::search_engine",
            population,
            "GA Fix C seed templates skipped because population ({population}) < 10 — set --population to >= 10 for hand-crafted seeding, otherwise the cold start is pure random"
        );
        generate_random_genes(
            population,
            n_indicators,
            max_indicators,
            0,
            &smc_cfg,
            &mut rng,
        )
    };
    enforce_population_smc_ratio(&mut genes, &smc_cfg);

    genes = genes
        .into_iter()
        .map(|g| {
            unique_candidate_or_retry(
                g,
                &mut seen_memory,
                n_indicators,
                max_indicators,
                0,
                seen_retry_attempts,
                &smc_cfg,
                &mut rng,
            )
        })
        .collect();

    let mut best_metrics = Vec::new();
    let mut profitable_archive: Vec<(Gene, [f64; 11], usize)> = Vec::new();
    let mut archive_seq = 0usize;
    // Item 4 from the search optimization notes: dedupe by `gene_signature_hash`
    // (a function of the canonical genome — sorted indices, weights, thresholds
    // and SMC flags) instead of `strategy_id`. The strategy_id is randomly
    // regenerated by `crossover`/`mutate` every generation, so two genomes
    // that compute the same signal kept getting archived under different ids.
    let mut seen_gene_hashes: HashSet<u64> = HashSet::new();

    // Archive scoring thresholds and selection policy come from the typed
    // overrides resolved above; no further env reads are necessary here.
    let archive_mode = genetic_runtime_overrides.archive_scoring.mode.clone();
    let archive_min_net = genetic_runtime_overrides.archive_scoring.min_net;
    let archive_min_pf = genetic_runtime_overrides.archive_scoring.min_pf;
    let archive_min_sharpe = genetic_runtime_overrides.archive_scoring.min_sharpe;
    let archive_cap = genetic_runtime_overrides.effective_archive_cap(population, generations);
    let base_immigrant_ratio = resolved_selection.immigrant_ratio;
    let base_survivor_fraction = resolved_selection.survivor_fraction;
    let parent_selection = resolved_selection.parent;
    let survivor_selection = resolved_selection.survivor;
    let selection_temperature = resolved_selection.temperature;
    let tournament_size = genetic_runtime_overrides.effective_tournament_size(population);
    let search_policy = EvolutionSearchPolicy::new(
        base_survivor_fraction,
        base_immigrant_ratio,
        parent_selection,
        survivor_selection,
        selection_temperature,
        tournament_size,
    );
    let stagnation_patience = genetic_runtime_overrides.effective_stagnation_patience();
    // HARD convergence early-stop (separate from the soft `stagnation_patience`
    // kick). `0` disables. `min_improvement` is the epsilon defining a
    // "meaningful" top-fitness gain when counting stagnant generations.
    let convergence_patience = genetic_runtime_overrides.effective_convergence_patience();
    let min_improvement = genetic_runtime_overrides.effective_min_improvement();
    // Wall-clock floor: the early-stop may fire only after this fraction of the
    // time budget has elapsed. Generation throughput varies ~300× across
    // timeframes (a fast TF does 250 gens in ~1 s, M1 in ~21 min), so a pure
    // generation count would kill fast TFs before they ever search. The floor
    // guarantees every combo gets real search time regardless of its gen rate.
    let convergence_min_elapsed_fraction =
        genetic_runtime_overrides.effective_convergence_min_elapsed_fraction();

    // Default OFF to avoid O(n²) cost; set > 0 only for large populations.
    let novelty_weight = genetic_runtime_overrides.novelty_weight;

    // Perf #3: build stable eval data cache ONCE before the generation loop
    let eval_cache = EvalDataCache::build(features, ohlcv);

    let started_at = Instant::now();
    let mut best_score_seen = f64::NEG_INFINITY;
    let mut stagnant_gens = 0usize;
    // Fires the "gate fully relaxed" notice at most ONCE per search. The
    // realized smc_gate_threshold is clamped to [gate_lo, gate_hi] regardless,
    // so the previous per-generation warn was pure log flood (audit 2026-06-08).
    let mut warned_gate_floor = false;

    if generations == 0 {
        let metrics = evaluate_genes_cached(features, ohlcv, &genes, &eval_cfg, &eval_cache)?;
        apply_metrics(&mut genes, &metrics);
        seen_memory.flush();
        return Ok(SearchResult { genes, metrics });
    }

    for generation in 0..generations {
        let progress = (generation as f32) / ((generations - 1) as f32).max(1.0);
        let mut gate_now = gate_start + (gate_end - gate_start) * progress.powf(gate_curve);
        if stagnant_gens >= stagnation_patience {
            // F-036 fix (2026-05-25): the stagnation decrement subtracts
            // `gate_stagnation_step * stagnant_gens`. With a default
            // step of 0.03 and `stagnant_gens` reaching the thousands the
            // raw value balloons (e.g. -54), but the downstream
            // `.clamp(gate_lo, gate_hi)` at line ~900 pins the REALIZED
            // threshold to `gate_lo` regardless — so the runaway raw value
            // has zero effect on which signals pass the gate. The previous
            // code re-emitted a WARN every generation once past the floor,
            // flooding the log on any stagnating combo (audit 2026-06-08).
            // We now (a) bottom the raw value at `gate_lo - 1.0` so it never
            // diverges, and (b) emit the diagnostic at most ONCE per search.
            gate_now -= gate_stagnation_step * (stagnant_gens as f32);
            let absolute_floor = gate_lo - 1.0;
            if gate_now < absolute_floor {
                if !warned_gate_floor {
                    warned_gate_floor = true;
                    tracing::info!(
                        target: "neoethos_search::search_engine",
                        generation,
                        stagnant_gens,
                        gate_lo,
                        "SMC gate fully relaxed under stagnation; realized \
                         threshold stays clamped at gate_lo. Logged once per search."
                    );
                }
                gate_now = absolute_floor;
            }
        }
        eval_cfg.smc_gate_threshold = gate_now.clamp(gate_lo, gate_hi);

        let metrics = evaluate_genes_cached(features, ohlcv, &genes, &eval_cfg, &eval_cache)?;
        apply_metrics(&mut genes, &metrics);

        let mut scored: Vec<(f64, usize, Gene, [f64; 11])> = genes
            .iter()
            .cloned()
            .zip(metrics)
            .enumerate()
            .map(|(idx, (g, m))| (g.fitness, idx, g, m))
            .collect();

        // --- Novelty Search: Behavioral Diversity ---
        // Pre-compute all HashSets once and run the O(n²) Jaccard pass in
        // parallel — turns a single-threaded bottleneck into Ncores× faster.
        if novelty_weight > 0.0 && scored.len() > 1 {
            use rayon::prelude::*;
            let n_pop = scored.len();
            let index_sets: Vec<HashSet<usize>> = scored
                .iter()
                .map(|(_, _, g, _)| g.indices.iter().copied().collect())
                .collect();

            // Parallel: each row i computes its mean Jaccard distance to the
            // remaining population. Each pair is touched twice (i→j and j→i),
            // matching the previous semantics exactly while running in parallel.
            let novelty_scores: Vec<f64> = (0..n_pop)
                .into_par_iter()
                .map(|i| {
                    let sig_i = &index_sets[i];
                    let mut dist_sum = 0.0;
                    for (j, sig_j) in index_sets.iter().enumerate() {
                        if i == j {
                            continue;
                        }
                        let intersection = sig_i.intersection(sig_j).count() as f64;
                        let union = sig_i.union(sig_j).count() as f64;
                        let jaccard_dist = if union > 0.0 {
                            1.0 - (intersection / union)
                        } else {
                            0.0
                        };
                        dist_sum += jaccard_dist;
                    }
                    dist_sum / (n_pop as f64 - 1.0)
                })
                .collect();

            // Normalize and blend
            let min_fit = scored
                .iter()
                .map(|(f, _, _, _)| *f)
                .filter(|f| f.is_finite())
                .fold(f64::INFINITY, f64::min);
            let max_fit = scored
                .iter()
                .map(|(f, _, _, _)| *f)
                .filter(|f| f.is_finite())
                .fold(f64::NEG_INFINITY, f64::max);
            let fit_range = (max_fit - min_fit).max(1e-9);
            let max_nov = novelty_scores
                .iter()
                .copied()
                .fold(0.0_f64, f64::max)
                .max(1e-9);

            for i in 0..n_pop {
                if !scored[i].0.is_finite() {
                    continue;
                }
                let norm_fit = (scored[i].0 - min_fit) / fit_range;
                let norm_nov = novelty_scores[i] / max_nov;
                // Modify the sorting score purely for the tournament survival/elites
                scored[i].0 = (1.0 - novelty_weight) * norm_fit + novelty_weight * norm_nov;
            }
        }
        // ------------------------------------------

        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        let top_score = scored.first().map(|x| x.0).unwrap_or(f64::NEG_INFINITY);
        if top_score > best_score_seen + min_improvement {
            best_score_seen = top_score;
            stagnant_gens = 0;
        } else {
            stagnant_gens += 1;
        }

        for (_score, _, gene, m) in scored.iter() {
            if profitable_archive.len() >= archive_cap {
                break;
            }
            let (net, sharpe, pf, trades) = (m[0], m[1], m[5], m[8]);
            if !net.is_finite() || !sharpe.is_finite() || !pf.is_finite() || !trades.is_finite() {
                continue;
            }
            let keep = match archive_mode.as_str() {
                "active" => trades > 0.0,
                "pf" | "profit_factor" => trades > 0.0 && pf > archive_min_pf,
                "sharpe" => trades > 0.0 && sharpe > archive_min_sharpe,
                _ => trades > 0.0 && net > archive_min_net,
            };
            if !keep {
                continue;
            }
            // Hash the canonical genome (after `Gene::normalize`) so two
            // mutated copies that produce the SAME signal collapse to one
            // archive entry regardless of their randomly-assigned strategy_id.
            let mut canonical = gene.clone();
            canonical.normalize(features.n_features(), 1);
            let hash = gene_signature_hash(&canonical);
            if !seen_gene_hashes.insert(hash) {
                continue;
            }
            profitable_archive.push((gene.clone(), *m, archive_seq));
            archive_seq += 1;
        }

        progress_fn(
            generation + 1,
            generations,
            top_score,
            stagnant_gens,
            profitable_archive.len(),
        );

        // Convergence early-stop (2026-06-09): once the search has been flat for
        // `convergence_patience` generations the soft diversity kick (gate
        // relaxation + raised immigrants + heavy hypermutation) has already had
        // its chance to escape — further generations are wasted wall-clock.
        // Returning the archive NOW lets the auto-loop advance to the next
        // symbol×timeframe (coverage beats depth — search-depth-economics
        // analysis: heavy TFs stagnate early and burn ~90% of the budget for
        // nothing). `convergence_patience == 0` disables. Fail-loud: the reason
        // is logged at INFO so this is never mistaken for a crash / silent
        // truncation. Returns via the SAME archive-or-top_candidates logic as
        // the wall-clock cap below so downstream finalize is byte-identical.
        // Intentionally placed BEFORE the max_runtime check: if a generation is
        // both converged and over-budget, both paths yield identical output and
        // "converged" is the more informative reason — do not reorder.
        //
        // WALL-CLOCK FLOOR (fix 2026-06-09): the generation count alone is NOT a
        // safe stop signal — a fast TF reaches 250 stagnant gens in ~1 s, far
        // too little real search to find the rare low-DD prop-firm genes (live
        // regression: AUDUSD H4 early-stopped at gen 291 in 1 s → 0 strategies
        // vs 7 on a full run). So the early-stop additionally requires that at
        // least `convergence_min_elapsed_fraction` of the time budget has
        // elapsed. With no time budget (`max_runtime == None`) the wall-clock
        // floor cannot be evaluated, so the early-stop is suppressed and the
        // combo runs to the generation ceiling.
        let convergence_floor_reached = match max_runtime {
            Some(mr) => started_at.elapsed() >= mr.mul_f64(convergence_min_elapsed_fraction),
            None => false,
        };
        if convergence_patience > 0
            && stagnant_gens >= convergence_patience
            && convergence_floor_reached
        {
            tracing::info!(
                target: "neoethos_search::search_engine",
                generation = generation + 1,
                total_generations = generations,
                best_score = best_score_seen,
                stagnant_gens,
                convergence_patience,
                elapsed_s = started_at.elapsed().as_secs(),
                min_elapsed_fraction = convergence_min_elapsed_fraction,
                archive_len = profitable_archive.len(),
                "GA converged: early-stopping this combo (flat for convergence_patience \
                 gens past the wall-clock floor); advancing to the next."
            );
            let best_return_count = population
                .clamp(2, (population / 2).clamp(100, 500))
                .min(scored.len());
            let top_candidates: Vec<Gene> = scored
                .iter()
                .take(best_return_count)
                .map(|(_, _, g, _)| g.clone())
                .collect();
            let top_metrics: Vec<[f64; 11]> = scored
                .iter()
                .take(best_return_count)
                .map(|(_, _, _, m)| *m)
                .collect();
            seen_memory.flush();
            if !profitable_archive.is_empty() {
                profitable_archive.sort_by(|a, b| {
                    b.1[0]
                        .partial_cmp(&a.1[0])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.2.cmp(&b.2))
                });
                return Ok(SearchResult {
                    genes: profitable_archive
                        .iter()
                        .map(|(g, _, _)| g.clone())
                        .collect(),
                    metrics: profitable_archive.iter().map(|(_, m, _)| *m).collect(),
                });
            }
            return Ok(SearchResult {
                genes: top_candidates,
                metrics: top_metrics,
            });
        }

        if let Some(max_runtime) = max_runtime
            && started_at.elapsed() >= max_runtime
        {
            // **F-035 documentation (2026-05-25)** — `best_return_count`
            // is the number of top-fitness genomes to return when the
            // wall-clock runtime budget is exhausted. The formula
            // intentionally:
            //   1. `population.clamp(2, ...)` — never return fewer than 2
            //      (caller's expectation: at least a parent + a sibling
            //      so the downstream genetic operators have material).
            //   2. `(population / 2).clamp(100, 500)` — upper bound is
            //      ~half the population but capped at [100, 500] so
            //      very-small populations don't return everything and
            //      very-large populations don't drown the caller in
            //      noise.
            //   3. `.min(scored.len())` — never exceed what we have.
            // The bounds [100, 500] are empirical: 100 is the smallest
            // archive size the diversity-archive can keep meaningful
            // novelty; 500 is the largest the downstream consumer
            // (`finalize_candidates_with_progress`) can process without
            // observable UI lag. Tunable via Settings would be a Phase-C
            // task — the literal here is a calibration, not a bug.
            let best_return_count = population
                .clamp(2, (population / 2).clamp(100, 500))
                .min(scored.len());
            let top_candidates: Vec<Gene> = scored
                .iter()
                .take(best_return_count)
                .map(|(_, _, g, _)| g.clone())
                .collect();
            let top_metrics: Vec<[f64; 11]> = scored
                .iter()
                .take(best_return_count)
                .map(|(_, _, _, m)| *m)
                .collect();
            seen_memory.flush();
            if !profitable_archive.is_empty() {
                profitable_archive.sort_by(|a, b| {
                    b.1[0]
                        .partial_cmp(&a.1[0])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.2.cmp(&b.2))
                });
                return Ok(SearchResult {
                    genes: profitable_archive
                        .iter()
                        .map(|(g, _, _)| g.clone())
                        .collect(),
                    metrics: profitable_archive.iter().map(|(_, m, _)| *m).collect(),
                });
            }
            return Ok(SearchResult {
                genes: top_candidates,
                metrics: top_metrics,
            });
        }

        let best_return_count = population
            .clamp(2, (population / 2).clamp(100, 500))
            .min(scored.len());
        let top_candidates: Vec<Gene> = scored
            .iter()
            .take(best_return_count)
            .map(|(_, _, g, _)| g.clone())
            .collect();
        best_metrics = scored
            .iter()
            .take(best_return_count)
            .map(|(_, _, _, m)| *m)
            .collect();

        if generation + 1 == generations {
            seen_memory.flush();
            if !profitable_archive.is_empty() {
                profitable_archive.sort_by(|a, b| {
                    b.1[0]
                        .partial_cmp(&a.1[0])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.2.cmp(&b.2))
                });
                return Ok(SearchResult {
                    genes: profitable_archive
                        .iter()
                        .map(|(g, _, _)| g.clone())
                        .collect(),
                    metrics: profitable_archive.iter().map(|(_, m, _)| *m).collect(),
                });
            }
            return Ok(SearchResult {
                genes: top_candidates,
                metrics: best_metrics,
            });
        }

        // Reuse the seeded RNG built at the top of `evolve_search_with_progress_impl`
        // (was `let mut rng = rand::rng();` here, which shadowed the seeded one and
        // broke the determinism work in the GPU path). `rng` is available in scope.
        let score_vector: Vec<f64> = scored.iter().map(|(score, _, _, _)| *score).collect();
        let survivor_fraction = if stagnant_gens >= stagnation_patience {
            (search_policy.survivor_fraction * 0.75).clamp(0.0, 0.5)
        } else {
            search_policy.survivor_fraction
        };
        let survivor_count = ((population as f64) * survivor_fraction).round() as usize;
        let survivor_count = match search_policy.survivor_selection {
            SurvivorSelectionPolicy::Generational => 0,
            _ => survivor_count.clamp(2, scored.len()),
        };
        let survivor_indices = select_survivor_indices(
            &score_vector,
            survivor_count,
            search_policy.survivor_selection,
            search_policy.selection_temperature,
            search_policy.tournament_size,
            &mut rng,
        );
        let survivors: Vec<Gene> = survivor_indices
            .iter()
            .map(|idx| scored[*idx].2.clone())
            .collect();

        let mut next = Vec::with_capacity(population);
        next.extend(survivors);

        // **GA Fix C — diversity rescue (2026-05-26, taskdoc #275)**.
        // The Python prototype's reward-hack ("never trade → 0 DD →
        // pass the max_dd filter") manifests here as a population that
        // collapses to >50% zero-trade after a few generations. Once
        // that happens the GA's gradient information is gone — every
        // zero-trade gene scores identically (-100 with graduated
        // fitness, NEG_INFINITY pre-fix) so survivor selection is
        // pure noise. We detect this state and inject fresh multi-TF
        // templates into 25% of the next population to break the
        // attractor.
        let zero_trade_count = scored
            .iter()
            .filter(|(_, _, g, _)| g.trades_count < 1)
            .count();
        let rescue_active = zero_trade_count * 2 > scored.len();
        let mut rescue_genes: Vec<Gene> = Vec::new();
        if rescue_active {
            let rescue_target = population / 4;
            // Try templates first; pad with fresh random genes if the
            // feature set can't resolve enough template roles.
            let seeds = super::seed_templates::seed_professional_templates(
                rescue_target,
                &features.names,
                n_indicators,
                &mut rng,
            );
            let template_n = seeds.len();
            rescue_genes.extend(seeds);
            while rescue_genes.len() < rescue_target {
                rescue_genes.push(new_random_gene(
                    n_indicators,
                    max_indicators,
                    generation + 1,
                    &smc_cfg,
                    &mut rng,
                ));
            }
            tracing::info!(
                target: "neoethos_search::search_engine",
                generation,
                zero_trade = zero_trade_count,
                pop = scored.len(),
                rescue_n = rescue_genes.len(),
                template_n,
                "diversity rescue: replaced {} zero-trade genes with seed templates",
                rescue_genes.len()
            );
        }

        let immigrant_ratio = if stagnant_gens >= stagnation_patience {
            search_policy.immigrant_fraction.max(0.5)
        } else {
            search_policy.immigrant_fraction
        };
        let immigrant_count = ((population as f64) * immigrant_ratio).round() as usize;
        // Reserve room for rescue genes so the rescue isn't crowded out
        // by the normal immigrant budget.
        let remaining_after_rescue = population
            .saturating_sub(next.len())
            .saturating_sub(rescue_genes.len());
        let immigrant_count = immigrant_count.min(remaining_after_rescue);
        for _ in 0..immigrant_count {
            let immigrant = new_random_gene(
                n_indicators,
                max_indicators,
                generation + 1,
                &smc_cfg,
                &mut rng,
            );
            next.push(unique_candidate_or_retry(
                immigrant,
                &mut seen_memory,
                n_indicators,
                max_indicators,
                generation + 1,
                seen_retry_attempts,
                &smc_cfg,
                &mut rng,
            ));
        }

        // Inject the rescue genes AFTER immigrants so they're treated
        // as first-class population members in the next-generation eval
        // (the seen-signature dedupe still applies so duplicates with
        // earlier-archived genes are rejected).
        for rescue in rescue_genes.drain(..) {
            if next.len() >= population {
                break;
            }
            next.push(unique_candidate_or_retry(
                rescue,
                &mut seen_memory,
                n_indicators,
                max_indicators,
                generation + 1,
                seen_retry_attempts,
                &smc_cfg,
                &mut rng,
            ));
        }

        let parent_indices: Vec<usize> = (0..scored.len()).collect();
        while next.len() < population {
            let a_idx = select_parent_index(
                &score_vector,
                &parent_indices,
                search_policy.parent_selection,
                search_policy.tournament_size,
                search_policy.selection_temperature,
                &mut rng,
            );
            let mut b_idx = select_parent_index(
                &score_vector,
                &parent_indices,
                search_policy.parent_selection,
                search_policy.tournament_size,
                search_policy.selection_temperature,
                &mut rng,
            );
            if parent_indices.len() > 1 {
                let mut retries = 0usize;
                while b_idx == a_idx && retries < 4 {
                    b_idx = select_parent_index(
                        &score_vector,
                        &parent_indices,
                        search_policy.parent_selection,
                        search_policy.tournament_size,
                        search_policy.selection_temperature,
                        &mut rng,
                    );
                    retries += 1;
                }
            }
            let a = &scored[a_idx].2;
            let b = &scored[b_idx].2;
            let crossed = crossover(a, b, generation + 1, &mut rng);
            let mutated = mutate(
                &crossed,
                n_indicators,
                max_indicators,
                generation + 1,
                &smc_cfg,
                stagnant_gens,
                &mut rng,
            );
            next.push(unique_candidate_or_retry(
                mutated,
                &mut seen_memory,
                n_indicators,
                max_indicators,
                generation + 1,
                seen_retry_attempts,
                &smc_cfg,
                &mut rng,
            ));
        }
        enforce_population_smc_ratio(&mut next, &smc_cfg);
        genes = next;
        seen_memory.flush();
    }
    seen_memory.flush();
    Ok(SearchResult {
        genes,
        metrics: best_metrics,
    })
}
