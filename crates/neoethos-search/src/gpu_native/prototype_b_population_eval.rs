//! Prototype B as the discovery pipeline's population evaluator.
//!
//! Until now B existed only inside the benchmark harness, while the shipped
//! discovery lane ran Prototype A in f32. A 2026-07-28 measurement on one card
//! settled which of those belongs in production:
//!
//! | bars | A-f32 | A-f64 | B |
//! |---|---|---|---|
//! | 4 096 | wrong, 95.0 M cand-bars/s | wrong, 13.5 M | exact, 49.7 M |
//! | 20 000 | wrong, 136.4 M | wrong, 15.4 M | exact, 47.8 M |
//! | 200 000 | wrong by 54 %, 138.9 M | wrong by 0.19 %, 11.5 M | exact, 47.4 M |
//!
//! B is the only lane that reproduces the canonical CPU engine, and it is also
//! 3-4x faster than A once A is given the double precision it needs to be even
//! approximately right. This module is the adapter that lets the discovery
//! pipeline call it, presenting exactly the signature the existing CubeCL entry
//! point uses so the call sites do not have to care which engine runs.
//!
//! Nothing here changes what is computed — B was proven bit-exact against the
//! CPU at 4 096, 20 000 and 200 000 bars on real EURUSD data. It changes which
//! engine computes it.

use anyhow::{Result, anyhow, bail};
use ndarray::ArrayView2;

use neoethos_gpu_contracts::device::ScenarioDescriptor;

use crate::eval::{BacktestSettings, SmcRow};
use crate::gpu_native::prototype_a::{PrototypeADatasetUpload, PrototypeAGeneUpload};
use crate::gpu_native::prototype_b_engine::PrototypeBPopulationInputs;
use crate::gpu_native::prototype_population_oracle::population_settings_for_dataset;
use crate::gpu_native::snapshot_fixture::SnapshotSettingsDto;

use neoethos_gpu_cuda::{PopulationDatasetView, PopulationGeneView, PopulationSession};

/// Metric row shape shared with the CPU and CubeCL lanes.
const ZERO_METRICS: [f64; 11] = [0.0; 11];

/// Is a CUDA device present and the native population engine usable?
pub(crate) fn prototype_b_available() -> bool {
    neoethos_gpu_cuda::runtime_available() && neoethos_gpu_cuda::device_count() > 0
}

/// Event capacity for a session, sized from the hardware rather than from the
/// caller's parameters.
///
/// The never-OOM invariant is that peak memory is a function of the available
/// device, never of what the user asked for: a run may be slow, but it must not
/// die. `max_events` is the one knob that would otherwise let a large population
/// dictate an allocation, so it is derived here from free VRAM and clamped, and
/// the per-candidate workspaces are subtracted first.
fn event_capacity(device: usize, population: usize, bars: usize) -> Result<usize> {
    let free = neoethos_gpu_cuda::device_free_memory_bytes(device)
        .ok_or_else(|| anyhow!("prototype B: cannot read free device memory to size the session"))?;
    // Leave headroom for context and fragmentation.
    let budget = (free / 10) * 7;
    // signals (1 B) + confidences (4 B) per candidate-bar, monthly buckets and
    // metric rows per candidate, plus a fixed reserve.
    let per_candidate_bar = 5u64;
    let per_candidate = 3_840u64 + 104;
    let fixed = population as u64 * bars as u64 * per_candidate_bar
        + population as u64 * per_candidate
        + 64 * 1024 * 1024;
    let room = budget.saturating_sub(fixed);
    // Each event costs an event record plus an outcome record.
    let capacity = room / 72;
    if capacity < 1_024 {
        bail!(
            "prototype B: {} candidates x {} bars leaves no room for events on this device \
             (free {} MiB) — evaluate a smaller batch",
            population,
            bars,
            free / (1024 * 1024)
        );
    }
    Ok(capacity as usize)
}

/// Evaluate a population on Prototype B.
///
/// Mirrors `cubecl_eval::try_evaluate_population_cuda` argument for argument so
/// the two are interchangeable at the call site, and returns the same
/// `[f64; 11]` rows in candidate order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_evaluate_population_b(
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
    stop_vol_mult: &[f64],
    smc_data: &[SmcRow],
    gene_smc_flags: &[SmcRow],
    gate_threshold: f32,
    smc_weights: &[f32; 11],
    settings: &BacktestSettings,
    device_override: Option<usize>,
) -> Result<Vec<[f64; 11]>> {
    let n_genes = long_thr.len();
    let bars = close.len();
    if n_genes == 0 || bars == 0 {
        return Ok(vec![ZERO_METRICS; n_genes]);
    }
    // Same optional-contract handling as the CubeCL lane: an empty
    // `stop_vol_mult` means "no adaptive stops", and every downstream slice
    // would otherwise index out of range.
    let stop_vol_fallback = crate::eval::normalized_stop_vol_mult(stop_vol_mult, n_genes);
    let stop_vol_mult = stop_vol_fallback.as_deref().unwrap_or(stop_vol_mult);

    // Indicators arrive as a `[feature][bar]` view; B wants that same layout
    // contiguous. `as_slice` succeeds when the view is already standard layout
    // and the copy is only paid when it is not.
    let feature_count = indicators.nrows();
    let indicators_flat: Vec<f32> = match indicators.as_slice() {
        Some(flat) => flat.to_vec(),
        None => indicators.iter().copied().collect(),
    };

    let dataset = PrototypeADatasetUpload {
        close: close.to_vec(),
        high: high.to_vec(),
        low: low.to_vec(),
        indicators: indicators_flat,
        feature_count,
        months: month_idx.to_vec(),
        days: day_idx.to_vec(),
        timestamps: timestamps.to_vec(),
        smc_data: smc_data.to_vec(),
        settings: SnapshotSettingsDto::from_settings(settings),
    };
    let genes = PrototypeAGeneUpload {
        candidate_ids: (0..n_genes as u64).collect(),
        offsets: gene_offsets.to_vec(),
        indices: gene_indices.to_vec(),
        weights: gene_weights.to_vec(),
        long_thresholds: long_thr.to_vec(),
        short_thresholds: short_thr.to_vec(),
        stop_pips: sl_pips.to_vec(),
        target_pips: tp_pips.to_vec(),
        stop_vol_multipliers: stop_vol_mult.to_vec(),
        smc_flags: gene_smc_flags.to_vec(),
        smc_weights: *smc_weights,
        gate_threshold,
    };

    let inputs = PrototypeBPopulationInputs::from_uploads(&dataset, &genes)
        .map_err(|error| anyhow!("prototype B: {error}"))?;
    let native_settings = population_settings_for_dataset(&dataset)
        .map_err(|error| anyhow!("prototype B settings: {error}"))?;

    let smc_rows: Vec<i8> = dataset.smc_data.iter().flatten().copied().collect();
    let adaptive_base = dataset.settings.to_settings().adaptive_base_pips.clone();

    let device = device_override.unwrap_or(0) as i32;
    let capacity = event_capacity(device as usize, n_genes, bars)?;
    let mut session = PopulationSession::create(device, capacity)
        .map_err(|error| anyhow!("prototype B session: {error}"))?;

    session
        .upload_dataset(PopulationDatasetView {
            close: &dataset.close,
            high: &dataset.high,
            low: &dataset.low,
            indicators: &dataset.indicators,
            feature_count: dataset.feature_count,
            months: &dataset.months,
            days: &dataset.days,
            timestamps: &dataset.timestamps,
            smc_rows: &smc_rows,
            adaptive_base_pips: adaptive_base.as_deref(),
        })
        .map_err(|error| anyhow!("prototype B dataset upload: {error}"))?;

    session
        .upload_genes(PopulationGeneView {
            descriptors: &inputs.descriptors,
            offsets: &genes.offsets,
            indices: &genes.indices,
            weights: &genes.weights,
            stop_pips: &genes.stop_pips,
            target_pips: &genes.target_pips,
            stop_vol_multipliers: &genes.stop_vol_multipliers,
            smc_flags: &inputs.smc_flags,
            smc_weights: &genes.smc_weights,
            gate_threshold: genes.gate_threshold,
            smc_gate_disabled: crate::genetic::smc_gate_disabled(),
        })
        .map_err(|error| anyhow!("prototype B gene upload: {error}"))?;

    // One full-window scenario per candidate. Costs, spread and slippage are
    // carried by the settings, not the scenario, so every per-scenario knob
    // stays zero — the engine rejects anything else, which is what keeps this
    // lane exactly the evaluation the CPU performs.
    let scenarios: Vec<ScenarioDescriptor> = (0..n_genes as u64)
        .map(|candidate| ScenarioDescriptor {
            base_candidate_id: candidate,
            scenario_id: candidate,
            rng_counter: 0,
            window_offset: 0,
            window_len: bars as u32,
            scenario_type: 0,
            spread_ticks: 0,
            slippage_ticks: 0,
            commission_micros: 0,
            perturbation_offset: 0,
            perturbation_count: 0,
            reserved: 0,
        })
        .collect();
    session
        .upload_scenarios(&scenarios)
        .map_err(|error| anyhow!("prototype B scenario upload: {error}"))?;

    let (event_id, _counters) = session
        .evaluate(&native_settings)
        .map_err(|error| anyhow!("prototype B evaluate: {error}"))?;
    session
        .wait(event_id)
        .map_err(|error| anyhow!("prototype B wait: {error}"))?;
    let rows = session
        .read_metrics()
        .map_err(|error| anyhow!("prototype B readback: {error}"))?;

    if rows.len() != n_genes {
        bail!(
            "prototype B returned {} metric rows for {} candidates",
            rows.len(),
            n_genes
        );
    }
    // Rows carry their candidate id, so order is restored explicitly rather
    // than assumed — a silently permuted population would misattribute every
    // metric to the wrong gene.
    let mut out = vec![ZERO_METRICS; n_genes];
    let mut seen = vec![false; n_genes];
    for row in &rows {
        let idx = row.candidate_id as usize;
        if idx >= n_genes {
            bail!(
                "prototype B returned candidate id {} outside the population of {}",
                row.candidate_id,
                n_genes
            );
        }
        if seen[idx] {
            bail!("prototype B returned candidate id {idx} twice");
        }
        seen[idx] = true;
        out[idx] = row.values;
    }
    if let Some(missing) = seen.iter().position(|hit| !hit) {
        bail!("prototype B returned no metric row for candidate {missing}");
    }
    Ok(out)
}
