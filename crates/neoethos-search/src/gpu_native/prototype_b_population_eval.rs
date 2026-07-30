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

use neoethos_gpu_contracts::device::{NeoPopulationEvent, ScenarioDescriptor};

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
fn event_capacity(
    device: usize,
    population: usize,
    bars: usize,
    feature_count: usize,
) -> Result<usize> {
    let free = neoethos_gpu_cuda::device_free_memory_bytes(device)
        .ok_or_else(|| anyhow!("prototype B: cannot read free device memory to size the session"))?;
    // Leave headroom for context and fragmentation.
    let budget = (free / 10) * 7;
    // The resident dataset. This dominates on a dense timeframe and omitting it
    // is not a rounding error: EURUSD M1 is 5.27 M bars against 257 features,
    // which is ~5.4 GB of indicators alone. Sizing that ignored the dataset
    // would hand back an event capacity the card cannot host and turn the
    // never-OOM invariant into a crash at exactly the workload it exists for.
    //   close/high/low f64, months/days/timestamps i64, SMC_SLOTS bytes per bar,
    //   plus feature-major f32 indicators.
    let dataset = bars as u64 * (3 * 8 + 3 * 8 + 11 + feature_count as u64 * 4);
    // signals (1 B) + confidences (4 B) per candidate-bar, monthly buckets and
    // metric rows per candidate, plus a fixed reserve.
    let per_candidate_bar = 5u64;
    let per_candidate = 3_840u64 + 104;
    let fixed = dataset
        + population as u64 * bars as u64 * per_candidate_bar
        + population as u64 * per_candidate
        + 64 * 1024 * 1024;
    let room = budget.saturating_sub(fixed);
    // Each event costs an event record plus an outcome record. Taken from the
    // types rather than written as a number: this was hardcoded at 72 bytes,
    // and when the contract grew to carry per-trade P&L and excursion the real
    // cost became 120 — so the session asked for two thirds more device memory
    // than it had budgeted for and the allocation failed with a bare status
    // code. A literal here is a silent trap the next time the contract changes.
    let bytes_per_event = (std::mem::size_of::<NeoPopulationEvent>()
        + std::mem::size_of::<neoethos_gpu_contracts::device::NeoPopulationOutcome>())
        as u64;
    let capacity = room / bytes_per_event;
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


// ── Resident session ─────────────────────────────────────────────────────────
//
// Measured on an RTX 2080 Ti, 2026-07-28: building a session and re-uploading
// the dataset on every call cost 2.7 M candidate-bars/s at 4 096 bars against
// 49.5 M for the same kernel driven by a session that stays alive — an 18x loss,
// falling to 23 % at 200 000 bars because the overhead is per call rather than
// per bar. That shape is exactly wrong here: the Monte-Carlo quality screen
// calls this evaluator once per surviving candidate (7 793 in a real AUDUSD H4
// run) and the GA calls it every generation.
//
// The native session refuses a second `upload_dataset`, so reuse means keeping
// the session alive while the dataset is unchanged and rebuilding it when it is
// not. Genes and scenarios carry no such restriction and are re-uploaded every
// call, which is correct — they are what actually varies.

/// Cheap identity for an uploaded dataset: length plus a strided sample.
///
/// Same approach the CubeCL resident cache uses, for the same reason — hashing
/// every byte of a 200 000-bar dataset on every call would reintroduce exactly
/// the per-call cost this cache exists to remove. Floats are hashed by their
/// bit pattern, so two datasets that differ only in a NaN payload are still
/// treated as different.
fn sample_hash<T, F>(values: &[T], to_bits: F, hasher: &mut impl std::hash::Hasher)
where
    F: Fn(&T) -> u64,
{
    use std::hash::Hash;
    values.len().hash(hasher);
    if values.is_empty() {
        return;
    }
    const SAMPLES: usize = 256;
    if values.len() <= SAMPLES {
        for value in values {
            to_bits(value).hash(hasher);
        }
    } else {
        let stride = values.len() / SAMPLES;
        for index in 0..SAMPLES {
            to_bits(&values[index * stride]).hash(hasher);
        }
        to_bits(&values[values.len() - 1]).hash(hasher);
    }
}

#[allow(clippy::too_many_arguments)]
fn dataset_key(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    indicators: &ArrayView2<'_, f32>,
    feature_count: usize,
    months: &[i64],
    days: &[i64],
    timestamps: &[i64],
    smc_data: &[SmcRow],
    settings: &BacktestSettings,
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for slice in [close, high, low] {
        sample_hash(slice, |v| v.to_bits(), &mut hasher);
    }
    // The view is sampled through its iterator so a non-contiguous layout does
    // not have to be flattened just to decide whether it is the same data.
    indicators.len().hash(&mut hasher);
    let stride = (indicators.len() / 256).max(1);
    for (i, v) in indicators.iter().enumerate() {
        if i % stride == 0 {
            v.to_bits().hash(&mut hasher);
        }
    }
    feature_count.hash(&mut hasher);
    for slice in [months, days, timestamps] {
        sample_hash(slice, |v| *v as u64, &mut hasher);
    }
    sample_hash(smc_data, |row| row.iter().map(|v| *v as u64).sum(), &mut hasher);
    // Settings participate because `adaptive_base_pips` is uploaded with the
    // dataset: two runs over identical bars but different adaptive stops are
    // different datasets on the device.
    format!("{settings:?}").hash(&mut hasher);
    hasher.finish()
}

struct ResidentSession {
    session: PopulationSession,
    key: u64,
    device: i32,
    capacity: usize,
    /// The host-side dataset, staged once and kept.
    ///
    /// Building it copies every bar of close/high/low and the whole
    /// feature-major indicator matrix — for M3 that is ~1.75 M bars against 64
    /// features. Rebuilding that per call, while the device copy sat resident
    /// and unchanged, was pure waste: the bars do not change between
    /// generations, only the genes do. It stays in RAM, ready.
    dataset: PrototypeADatasetUpload,
    smc_rows: Vec<i8>,
    native_settings: neoethos_gpu_contracts::device::NeoPopulationSettings,
}

/// The session holds a raw device handle, which is why it is not `Send` by
/// itself. Sending it between threads is sound here for a specific reason:
/// every native entry point begins with `cudaSetDevice(session->device)`, so a
/// call from a different thread binds the right device before touching
/// anything, and all access is serialized by the mutex below. Without that
/// per-entry binding this would be unsound.
struct SendResident(ResidentSession);
unsafe impl Send for SendResident {}

fn resident_slot() -> &'static std::sync::Mutex<Option<SendResident>> {
    static SLOT: std::sync::OnceLock<std::sync::Mutex<Option<SendResident>>> =
        std::sync::OnceLock::new();
    SLOT.get_or_init(|| std::sync::Mutex::new(None))
}

/// Largest population known to have fitted the card's event budget.
///
/// Discovering the limit by failing costs the whole attempt: the kernel runs,
/// exhausts capacity, and its work is discarded before the halves are retried.
/// Paying that once is the price of not knowing how many trades a population
/// emits; paying it every generation is waste. A 2026-07-29 M3 run spent 391 s
/// on a single population evaluation that way, against a benchmark rate that
/// would place it near 4 s.
///
/// So the size that worked is remembered and used as the starting point.
/// `usize::MAX` means "not yet learned" — the first call tries the whole
/// population, as it should.
fn learned_batch_limit() -> &'static std::sync::atomic::AtomicUsize {
    static LIMIT: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(usize::MAX);
    &LIMIT
}

/// Evaluate a population on Prototype B, splitting when the card cannot hold
/// the trades it produces.
///
/// Event capacity is bounded by VRAM, but how many entries a population emits
/// is not knowable before running it — a dense set of signals over 351 518 bars
/// can exceed any budget. The engine reports that distinctly
/// (`is_capacity_exhausted`) rather than as a generic failure, so the answer is
/// to give it less work rather than to give up: the population is halved and
/// each half evaluated in turn.
///
/// Genes are independent, so a split result equals the whole-population one.
/// This is the never-OOM rule applied as intended — peak memory follows the
/// hardware, and an oversized workload gets slower instead of failing.
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
    use std::sync::atomic::Ordering as AtomicOrd;
    let n_genes = long_thr.len();

    // Start at the size already known to fit rather than rediscovering the
    // limit by throwing away a full evaluation every generation.
    let learned = learned_batch_limit().load(AtomicOrd::Relaxed);
    if n_genes > learned && n_genes > 1 {
        return split_and_evaluate(
            close, high, low, indicators, gene_offsets, gene_indices, gene_weights, long_thr,
            short_thr, month_idx, day_idx, timestamps, sl_pips, tp_pips, stop_vol_mult, smc_data,
            gene_smc_flags, gate_threshold, smc_weights, settings, device_override,
        );
    }

    let attempt = evaluate_population_b_batch(
        close, high, low, indicators, gene_offsets, gene_indices, gene_weights, long_thr,
        short_thr, month_idx, day_idx, timestamps, sl_pips, tp_pips, stop_vol_mult, smc_data,
        gene_smc_flags, gate_threshold, smc_weights, settings, device_override,
    );
    let Err(error) = attempt else {
        // This size fits; keep it as the starting point for the next call.
        learned_batch_limit().fetch_max(n_genes, AtomicOrd::Relaxed);
        return attempt;
    };
    // Only a capacity exhaustion is worth retrying smaller. Anything else is a
    // fault, and halving the work would just hide it behind a slower failure.
    if !format!("{error}").contains("exceeded the session capacity") || n_genes < 2 {
        return Err(error);
    }
    // Remember the ceiling so the next generation does not pay this again.
    learned_batch_limit().fetch_min(n_genes / 2, AtomicOrd::Relaxed);
    tracing::info!(
        target: "neoethos_search::eval",
        n_genes,
        learned = learned_batch_limit().load(AtomicOrd::Relaxed),
        "population emitted more trades than the card can hold — splitting, and          remembering the limit so later generations start there"
    );
    return split_and_evaluate(
        close, high, low, indicators, gene_offsets, gene_indices, gene_weights, long_thr,
        short_thr, month_idx, day_idx, timestamps, sl_pips, tp_pips, stop_vol_mult, smc_data,
        gene_smc_flags, gate_threshold, smc_weights, settings, device_override,
    );
}

/// Halve the population and evaluate each side.
///
/// CSR arrays are sliced by gene with their term windows carried along, and the
/// tail's offsets rebased so its first gene starts at zero — getting that wrong
/// evaluates the wrong terms silently rather than erroring.
#[allow(clippy::too_many_arguments)]
fn split_and_evaluate(
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
    let half = n_genes / 2;

    // CSR gene arrays are sliced by gene, and the term ranges follow the
    // offsets, so a split has to carry the right window of both.
    let split_end = gene_offsets[half] as usize;
    let mut head = try_evaluate_population_b(
        close, high, low, indicators, &gene_offsets[..=half], &gene_indices[..split_end],
        &gene_weights[..split_end], &long_thr[..half], &short_thr[..half], month_idx, day_idx,
        timestamps, &sl_pips[..half], &tp_pips[..half], &stop_vol_mult[..half], smc_data,
        &gene_smc_flags[..half], gate_threshold, smc_weights, settings, device_override,
    )?;
    // The tail's offsets must be rebased so its first gene starts at zero.
    let tail_offsets: Vec<i32> = gene_offsets[half..]
        .iter()
        .map(|offset| offset - gene_offsets[half])
        .collect();
    let tail = try_evaluate_population_b(
        close, high, low, indicators, &tail_offsets, &gene_indices[split_end..],
        &gene_weights[split_end..], &long_thr[half..], &short_thr[half..], month_idx, day_idx,
        timestamps, &sl_pips[half..], &tp_pips[half..], &stop_vol_mult[half..], smc_data,
        &gene_smc_flags[half..], gate_threshold, smc_weights, settings, device_override,
    )?;
    head.extend(tail);
    Ok(head)
}

/// Evaluate a population on Prototype B.
///
/// Mirrors `cubecl_eval::try_evaluate_population_cuda` argument for argument so
/// the two are interchangeable at the call site, and returns the same
/// `[f64; 11]` rows in candidate order.
#[allow(clippy::too_many_arguments)]
fn evaluate_population_b_batch(
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

    // Identity is decided from the caller's slices, before anything is copied,
    // so a repeat call over the same bars costs a sampled hash instead of a
    // full rebuild of the dataset.
    let feature_count = indicators.nrows();
    let key = dataset_key(
        close,
        high,
        low,
        &indicators,
        feature_count,
        month_idx,
        day_idx,
        timestamps,
        smc_data,
        settings,
    );
    let device = device_override.unwrap_or(0) as i32;
    let capacity = event_capacity(device as usize, n_genes, bars, feature_count)?;

    // Hold the slot for the whole evaluation. That serializes device access the
    // way the CubeCL launch lock does, which is deliberate: the quality screen
    // calls this from a rayon `par_iter`, and one session per worker thread
    // would multiply VRAM by the thread count.
    let mut slot = resident_slot()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let reusable = slot
        .as_ref()
        .is_some_and(|r| r.0.key == key && r.0.device == device && r.0.capacity >= capacity);

    if !reusable {
        // Drop the old session before building the new one so the two never
        // hold device memory at the same time.
        *slot = None;

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
        let native_settings = population_settings_for_dataset(&dataset)
            .map_err(|error| anyhow!("prototype B settings: {error}"))?;
        let smc_rows: Vec<i8> = dataset.smc_data.iter().flatten().copied().collect();
        let adaptive_base = dataset.settings.to_settings().adaptive_base_pips.clone();

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

        *slot = Some(SendResident(ResidentSession {
            session,
            key,
            device,
            capacity,
            dataset,
            smc_rows,
            native_settings,
        }));
    }

    let resident = &mut slot
        .as_mut()
        .expect("resident session was just installed")
        .0;
    let native_settings = resident.native_settings;

    // Genes and scenarios are what actually change between calls, so they are
    // the only things rebuilt and re-uploaded.
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
    let inputs = PrototypeBPopulationInputs::from_uploads(&resident.dataset, &genes)
        .map_err(|error| anyhow!("prototype B: {error}"))?;
    let session = &mut resident.session;

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

    // How much of the event budget the population actually needed.
    //
    // This is the number that decides three open questions at once. The kernel
    // benchmarks at 47-50 M candidate-bars/s; inside discovery it managed 0.46 M,
    // which is slower than the 128-core CPU. The suspected cause is that the
    // population overruns the event budget and gets split into many small
    // launches, so fixed per-launch cost dominates. The same overrun would mean
    // each gene opens a position every few bars — which would also explain an
    // MFE capture of 18 % and a walk-forward gate that never passes.
    //
    // Logged once per process: it is a property of the data, not of the call.
    {
        static LOGGED: std::sync::Once = std::sync::Once::new();
        let emitted = session.emitted_events();
        LOGGED.call_once(|| {
            let per_candidate = emitted as f64 / n_genes.max(1) as f64;
            tracing::info!(
                target: "neoethos_search::eval",
                n_genes,
                bars,
                event_capacity = capacity,
                emitted_events = emitted,
                events_per_candidate = format!("{per_candidate:.0}"),
                bars_per_trade = format!("{:.1}", bars as f64 / per_candidate.max(1e-9)),
                capacity_used_pct = format!(
                    "{:.1}",
                    100.0 * emitted as f64 / capacity.max(1) as f64
                ),
                "event budget usage for this population"
            );
        });
    }

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
