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

use anyhow::{Context, Result, anyhow, bail};
use ndarray::ArrayView2;

use neoethos_gpu_contracts::device::{NeoPopulationEvent, ScenarioDescriptor};

use crate::eval::{BacktestSettings, SmcRow};
use crate::gpu_native::prototype_a::{PrototypeADatasetUpload, PrototypeAGeneUpload};
use crate::gpu_native::prototype_b_engine::PrototypeBPopulationInputs;
use crate::gpu_native::prototype_population_oracle::population_settings_for_dataset;
use crate::gpu_native::snapshot_fixture::SnapshotSettingsDto;

use neoethos_gpu_cuda::{
    CudaPopulationError, PopulationDatasetView, PopulationGeneView, PopulationSession,
};

/// Metric row shape shared with the CPU and CubeCL lanes.
const ZERO_METRICS: [f64; 11] = [0.0; 11];

/// Is a CUDA device present and the native population engine usable?
pub(crate) fn prototype_b_available() -> bool {
    neoethos_gpu_cuda::runtime_available() && neoethos_gpu_cuda::device_count() > 0
}

/// Candidates the card can host at once, from free VRAM rather than from the
/// caller's population.
///
/// `event_capacity` below budgets for an event buffer the kernel no longer
/// allocates. What it allocates instead is `population * MAX_TRADES_PER_
/// CANDIDATE` outcome records — ~590 KB per candidate — so peak memory became a
/// function of the requested population. That is the never-OOM invariant
/// inverted, and removing the event buffer is what inverted it.
///
/// Measured: validation asks for ~25 000 candidates in one call (250 folds x
/// 100 Monte-Carlo runs). At 1.03 MB each over 87 715 bars that is ~25 GB on a
/// 24 GB card — it failed, the retry halved it, the halves failed too because
/// the first failure had already left the context unusable, and 25 000 of
/// 25 250 items ran on the CPU after 30 s of wasted attempts.
///
/// Deciding the size up front costs one query and removes the failure entirely.
/// The batch to use when free memory cannot be read and nothing has worked yet.
///
/// At ~0.68 MB per candidate over 87 715 bars this is around 700 MB — small
/// enough for any discrete card, large enough to keep the reduce busy. It is a
/// floor for a blind moment, not a target, and
/// `the_blind_batch_is_small_enough_for_any_card` holds it to that whatever a
/// candidate comes to cost.
const CONSERVATIVE_BATCH: usize = 1_024;

/// The last size the card was known to accept, for when the query stops
/// answering. Better evidence than a constant, and it costs one atomic.
fn last_known_fit() -> &'static std::sync::atomic::AtomicUsize {
    static LAST: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    &LAST
}

fn candidates_that_fit(device: usize, bars: usize, feature_count: usize) -> Option<usize> {
    candidates_for_free_memory(
        neoethos_gpu_cuda::device_free_memory_bytes(device)?,
        bars,
        feature_count,
    )
}

/// Device bytes the resident dataset occupies, in one place.
///
/// `upload_dataset` allocates, per bar: `close`/`high`/`low` as `double`,
/// `months`/`days`/`timestamps` as `long long`, `SMC_SLOTS` signed bytes,
/// `gap_flags` as one unsigned byte, `adaptive_base_pips` as a `double` when
/// the settings carry one, and `feature_count` `float` indicators.
///
/// Both budgets below wrote this expression out separately and both stopped at
/// 59 + 4F — the gap flags and the adaptive base column were missing. That is
/// 9 B a bar, which is nothing on H1 and 47 MB on M1's 5.27 M bars, against a
/// fixed reserve of 64 MB. The adaptive column is charged unconditionally
/// because over-charging shrinks a batch while under-charging fails an
/// allocation, and adaptive stops are on by default.
fn dataset_device_bytes(bars: usize, feature_count: usize) -> u64 {
    const PRICES: u64 = 3 * 8;
    const CALENDAR: u64 = 3 * 8;
    const SMC_ROW: u64 = neoethos_gpu_cuda::SMC_SLOTS as u64;
    const GAP_FLAG: u64 = 1;
    const ADAPTIVE_BASE: u64 = 8;
    bars as u64
        * (PRICES + CALENDAR + SMC_ROW + GAP_FLAG + ADAPTIVE_BASE + feature_count as u64 * 4)
}

/// Monthly buckets a candidate is budgeted for.
///
/// `NeoPopulationSettings::month_capacity` is a runtime value, but every
/// production path resolves it to this — `population_settings_for_dataset`
/// takes it from the backtest runtime overrides, whose default is pinned at 240
/// by `prototype_population_oracle`. Budgeting for the default is deliberate:
/// two `f64` arrays of 240 is 3 840 B against the ~590 KB of trade slots beside
/// it, so a wrong month capacity moves the answer by well under a percent while
/// querying the settings here would need the dataset the caller has not staged
/// yet.
const MONTH_BUCKETS_BUDGETED: u64 = 240;

/// Device bytes one candidate occupies over `bars`, in one place.
///
/// There were two copies of this and both charged 5 bytes per candidate-bar —
/// one for the `signed char` signal column and four for a `float` confidence
/// column. The kernel stopped allocating the confidence column and recomputes
/// the value at the entry bar instead, but nothing here noticed: `fits` never
/// grew, the batch never changed, and a measured H1 run moved 127.8 s -> 127.5 s
/// for a kernel that had just given up four of the five bytes it held per
/// candidate-bar. The cheaper kernel was never asked for more work.
///
/// So the per-candidate-bar figure now comes from
/// `WORKSPACE_BYTES_PER_CANDIDATE_BAR`, which
/// `workspace_bytes_per_candidate_bar_match_the_kernel` reads out of the `.cu`
/// the device actually compiles. Host and device can no longer disagree
/// silently about what a candidate costs.
fn per_candidate_device_bytes(bars: usize) -> u64 {
    use neoethos_gpu_contracts::device::{NeoPopulationMetricRow, NeoPopulationOutcome};
    // The trade slots: `population * MAX_TRADES_PER_CANDIDATE` outcome records,
    // and by far the largest term — ~590 KB of the 0.68 MB an H1 candidate
    // costs once the confidence column is gone.
    let trade_slots = neoethos_gpu_cuda::MAX_TRADES_PER_CANDIDATE
        * std::mem::size_of::<NeoPopulationOutcome>() as u64;
    // The bar-indexed columns, priced by the kernel rather than by convention.
    let bar_columns = bars as u64 * neoethos_gpu_cuda::WORKSPACE_BYTES_PER_CANDIDATE_BAR;
    // `monthly_pnls` and `month_start_equities`, plus the one metric row read
    // back per candidate.
    let monthly = 2 * MONTH_BUCKETS_BUDGETED * std::mem::size_of::<f64>() as u64;
    trade_slots + bar_columns + monthly + std::mem::size_of::<NeoPopulationMetricRow>() as u64
}

/// The arithmetic, separated from the device query so it can be checked without
/// a card. The numbers it produces decide whether a run uses the GPU at all.
fn candidates_for_free_memory(free: u64, bars: usize, feature_count: usize) -> Option<usize> {
    // Same headroom convention as `event_capacity`: leave three tenths for
    // context, fragmentation and the allocator's own bookkeeping.
    let budget = (free / 10) * 7;
    let dataset = dataset_device_bytes(bars, feature_count);
    let room = budget
        .saturating_sub(dataset)
        .saturating_sub(64 * 1024 * 1024);
    let fits = room / per_candidate_device_bytes(bars).max(1);
    // Below this the card cannot do useful work and the CPU lane is the honest
    // answer; `None` leaves the decision where it already is.
    (fits >= 16).then_some(fits as usize)
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
    // `dataset_device_bytes` carries the enumeration; a second copy of it here
    // is how the gap-flag and adaptive-base columns came to be missing from
    // both.
    let dataset = dataset_device_bytes(bars, feature_count);
    // The bar-indexed columns, priced from the kernel's own allocation list
    // rather than written here as a number. This said 5 — one byte of signal
    // plus four of confidence — and went on saying it after the kernel stopped
    // allocating the confidence column, which is why removing that column
    // bought nothing.
    let per_candidate_bar = neoethos_gpu_cuda::WORKSPACE_BYTES_PER_CANDIDATE_BAR;
    // Deliberately NOT the whole of `per_candidate_device_bytes`: the trade
    // slots are excluded here. What this function sizes is `max_events`, whose
    // only consumer is a kernel that is no longer launched, and its live effect
    // is the `bail!` below acting as a coarse "this population has no room at
    // all" gate. Charging the ~590 KB of trade slots as well would make that
    // gate fire on exactly the batch `candidates_for_free_memory` just approved
    // — it approves the largest population whose slots fit, leaving a remainder
    // smaller than one candidate — and turn a correct batch into a spurious
    // split. The trade-slot reservation is where this wants to end up, and it
    // gets there when the reservation stops being a compile-time constant.
    let per_candidate = 2 * MONTH_BUCKETS_BUDGETED * std::mem::size_of::<f64>() as u64
        + std::mem::size_of::<neoethos_gpu_contracts::device::NeoPopulationMetricRow>() as u64;
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
    /// The population the device workspace was actually allocated for.
    ///
    /// The signal, confidence and outcome arrays are sized `population * bars`
    /// when the workspace is built, and the kernel indexes them by the CURRENT
    /// population — which `upload_genes` overwrites on every call. Reusing a
    /// session for a larger population therefore writes past the end of every
    /// one of those arrays, into whatever the allocator placed next.
    ///
    /// Nothing stopped that. The reuse test was `capacity >= capacity`, and
    /// `event_capacity` SUBTRACTS the per-candidate workspaces, so a larger
    /// population asks for LESS capacity and passes more easily — the check
    /// was not merely silent about population, it was inverted with respect
    /// to it. A session built for 256 candidates was reused for 25 600.
    workspace_population: usize,
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
    // What the card can hold is knowable before asking it, so ask first. The
    // retry below still exists for what cannot be predicted — how many trades a
    // population actually emits — but it should never be reached for a size
    // that was arithmetic all along.
    let fits = match candidates_that_fit(
        device_override.unwrap_or(0),
        close.len(),
        indicators.nrows(),
    ) {
        Some(fits) => {
            last_known_fit().store(fits, AtomicOrd::Relaxed);
            fits
        }
        // Not knowing how much room there is is a reason to ask for less, not
        // for everything. This read `unwrap_or(usize::MAX)` — unknown meant
        // unlimited — and a measured run launched 24 700 candidates against a
        // card that holds ~16 300, died with a stream synchronization failure,
        // and left the CUDA context unusable: the 27 evaluations after it could
        // not even read free memory, so 31 859 items went to the CPU from one
        // bad guess.
        None => {
            let fallback = match last_known_fit().load(AtomicOrd::Relaxed) {
                0 => CONSERVATIVE_BATCH,
                known => known,
            };
            tracing::warn!(
                target: "neoethos_search::eval",
                n_genes,
                fallback,
                "cannot read free device memory — sizing the batch conservatively"
            );
            fallback
        }
    };
    if fits < n_genes {
        tracing::info!(
            target: "neoethos_search::eval",
            n_genes,
            fits,
            "population exceeds what the card can host — splitting before asking"
        );
    }
    let learned = learned.min(fits);
    if n_genes > learned && n_genes > 1 {
        return split_and_evaluate(
            close, high, low, indicators, gene_offsets, gene_indices, gene_weights, long_thr,
            short_thr, month_idx, day_idx, timestamps, sl_pips, tp_pips, stop_vol_mult, smc_data,
            gene_smc_flags, gate_threshold, smc_weights, settings, device_override, learned,
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
    if !is_capacity_exhaustion(&error) || n_genes < 2 {
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
        gene_smc_flags, gate_threshold, smc_weights, settings, device_override, n_genes / 2,
    );
}

/// Whether the card ran out of room, so halving the work is worth a retry.
///
/// This compared the message text against `"exceeded the session capacity"`,
/// which is the event buffer's wording. Removing the event buffer moved the
/// first allocation to fail to the outcome array, which says `"device
/// allocation failed"` — so the retry stopped firing and every oversized
/// population went to the CPU. The optimisation disabled its own safety net,
/// quietly, because the two agreed on a string rather than a type.
///
/// Asking the error what it is removes that coupling. The capacity check in
/// `event_capacity_for` raises a plain `anyhow` error rather than a native one,
/// so it is matched separately and by a phrase it owns.
///
/// This only works because the call sites attach context rather than formatting
/// the error into a new one — `anyhow!("...: {error}")` renders the source and
/// throws the value away, which left this check unable to find anything and
/// made it dead code the moment it was written.
fn is_capacity_exhaustion(error: &anyhow::Error) -> bool {
    if error
        .downcast_ref::<CudaPopulationError>()
        .is_some_and(CudaPopulationError::is_capacity_exhausted)
    {
        return true;
    }
    format!("{error}").contains("leaves no room for events on this device")
}

#[cfg(test)]
mod capacity_detection_tests {
    use super::*;

    /// Whatever it approves has to fit in the budget it was given.
    ///
    /// This used to assert `fits < 25_000` on a 24 GB card, because the measured
    /// failure was validation asking for ~25 000 candidates at 1.03 MB each —
    /// ~25 GB on a 24 GB card. That number was a consequence of what a candidate
    /// cost, not an invariant: with the confidence column gone a candidate costs
    /// 0.68 MB at these dimensions and 25 000 of them genuinely do fit, so the
    /// old assertion would now forbid a batch that is correct.
    ///
    /// The durable statement is the one that number stood in for: the approved
    /// batch, priced at what the device actually allocates, must sit inside the
    /// same seven tenths of free memory the function budgeted with, and one more
    /// candidate must not.
    #[test]
    fn whatever_it_approves_fits_inside_the_budget_it_was_given() {
        for (free, bars, features) in [
            (24u64 * 1024 * 1024 * 1024, 87_715usize, 257usize),
            (24 * 1024 * 1024 * 1024, 1_757_261, 64),
            (24 * 1024 * 1024 * 1024, 5_270_000, 64),
            (8 * 1024 * 1024 * 1024, 87_715, 257),
        ] {
            let fits = candidates_for_free_memory(free, bars, features)
                .unwrap_or_else(|| panic!("{bars} bars on {free} B should host a batch"));
            let budget = (free / 10) * 7;
            let dataset = dataset_device_bytes(bars, features);
            const RESERVE: u64 = 64 * 1024 * 1024;
            let footprint = fits as u64 * per_candidate_device_bytes(bars) + dataset + RESERVE;
            assert!(
                footprint <= budget,
                "{bars} bars: approved {fits} candidates costing {footprint} B (with the \
                 reserve) against a budget of {budget} B"
            );
            // And it is not leaving the card half empty either: one more
            // candidate has to breach that same budget.
            let one_more = (fits as u64 + 1) * per_candidate_device_bytes(bars) + dataset + RESERVE;
            assert!(
                one_more > budget,
                "{bars} bars: approved {fits}, but {} would also have fitted",
                fits + 1
            );
        }
    }

    /// The reuse test was inverted with respect to population.
    ///
    /// `event_capacity` subtracts the per-candidate-bar columns and the monthly
    /// buckets from the budget, so asking for MORE candidates yields a SMALLER
    /// required capacity — and `capacity >= capacity` therefore passed exactly
    /// when it should have failed. This pins the arithmetic that made the old
    /// check unsafe, so a future edit to the budget cannot quietly restore it.
    #[test]
    fn a_bigger_population_demands_less_capacity_which_is_why_that_test_was_unsafe() {
        const BARS: usize = 87_715;
        const FEATURES: usize = 81;
        const FREE: u64 = 24 * 1024 * 1024 * 1024;

        // Same shape as `event_capacity`, without the device query.
        let capacity_for = |population: u64| -> u64 {
            let budget = (FREE / 10) * 7;
            let dataset = dataset_device_bytes(BARS, FEATURES);
            let per_candidate = 2 * MONTH_BUCKETS_BUDGETED * 8 + 104;
            let fixed = dataset
                + population * BARS as u64 * neoethos_gpu_cuda::WORKSPACE_BYTES_PER_CANDIDATE_BAR
                + population * per_candidate
                + 64 * 1024 * 1024;
            budget.saturating_sub(fixed) / 128
        };

        let small = capacity_for(256);
        let large = capacity_for(25_600);
        assert!(
            large < small,
            "25 600 candidates asked for {large} and 256 asked for {small} — if this ever              reverses, re-read why the population check exists"
        );
    }

    /// A blind moment must not approve everything.
    ///
    /// The unreadable-memory branch used to yield `usize::MAX`. That single
    /// default launched 24 700 candidates at a card holding ~16 300, which
    /// failed mid-stream and left the CUDA context unusable for the rest of the
    /// run. Whatever it yields, it has to be a batch the smallest sensible card
    /// could host.
    #[test]
    fn the_blind_batch_is_small_enough_for_any_card() {
        const BARS: usize = 87_715;
        let blind = CONSERVATIVE_BATCH as u64 * per_candidate_device_bytes(BARS);
        assert!(
            blind < 2 * 1024 * 1024 * 1024,
            "a blind batch wants {} MiB, which is not safe on a small card",
            blind / (1024 * 1024)
        );
        assert!(CONSERVATIVE_BATCH >= 256, "and it still has to keep the card busy");
    }

    /// The host must charge exactly what the kernel allocates per candidate-bar.
    ///
    /// This is the defect the whole change is about. `signal_confidences` was
    /// removed from the `.cu` and the host went on charging for it, so `fits`
    /// never moved and the batch never grew — the kernel got cheaper and nothing
    /// asked for more work. `neoethos_gpu_cuda`'s own
    /// `workspace_bytes_per_candidate_bar_match_the_kernel` pins the constant to
    /// the kernel source; this pins the host's use of it, so the two halves of
    /// the fix cannot come apart again.
    #[test]
    fn the_bar_proportional_charge_is_what_the_kernel_allocates() {
        let near = per_candidate_device_bytes(1_000);
        let far = per_candidate_device_bytes(1_001_000);
        assert_eq!(
            far - near,
            1_000_000 * neoethos_gpu_cuda::WORKSPACE_BYTES_PER_CANDIDATE_BAR,
            "a million more bars must cost a million times what the kernel allocates per \
             candidate-bar, and nothing else"
        );
        // 5 B was one byte of signal plus four of confidence. If this ever holds
        // again, either the column is back or the host is paying for one the
        // device does not allocate — and the second is what happened.
        assert_ne!(
            neoethos_gpu_cuda::WORKSPACE_BYTES_PER_CANDIDATE_BAR,
            5,
            "the confidence column is gone from the kernel"
        );
    }

    /// Dropping the confidence column has to show up as a bigger batch.
    ///
    /// Measured before this landed: 12 702-16 709 candidates per launch on H1,
    /// ~13 % of an RTX 3090's 125 952 resident threads, with the reduce running
    /// one thread per candidate. The kernel change on its own moved none of it,
    /// because the host kept charging for the column the kernel had stopped
    /// allocating. This is what proves that half is in place.
    #[test]
    fn removing_the_confidence_column_raises_the_batch() {
        const FREE: u64 = 24 * 1024 * 1024 * 1024;
        // What the host used to charge: today's price plus the 4 B/candidate-bar
        // confidence column.
        let fits_when_confidence_was_charged = |bars: usize, features: usize| -> u64 {
            let budget = (FREE / 10) * 7;
            let dataset = dataset_device_bytes(bars, features);
            let room = budget
                .saturating_sub(dataset)
                .saturating_sub(64 * 1024 * 1024);
            room / (per_candidate_device_bytes(bars) + bars as u64 * 4).max(1)
        };
        for (bars, features, least) in [
            (87_715usize, 257usize, 1.4f64),
            (1_757_261, 64, 3.0),
            (5_270_000, 64, 4.0),
        ] {
            let before = fits_when_confidence_was_charged(bars, features);
            let after = candidates_for_free_memory(FREE, bars, features)
                .unwrap_or_else(|| panic!("{bars} bars should host a batch"))
                as u64;
            let ratio = after as f64 / before.max(1) as f64;
            assert!(
                ratio >= least,
                "{bars} bars: {before} -> {after} candidates is {ratio:.2}x, and the column \
                 that went away was worth at least {least:.1}x here"
            );
        }
    }

    /// Peak memory must follow the hardware, never the request.
    #[test]
    fn a_smaller_card_approves_a_smaller_batch() {
        const BARS: usize = 87_715;
        const FEATURES: usize = 257;
        let big = candidates_for_free_memory(24 * 1024 * 1024 * 1024, BARS, FEATURES).unwrap();
        let small = candidates_for_free_memory(8 * 1024 * 1024 * 1024, BARS, FEATURES).unwrap();
        assert!(small < big, "8 GB approved {small}, 24 GB approved {big}");

        // A card with no room to work says so rather than approving a batch it
        // cannot host — the CPU lane is the honest answer there.
        assert_eq!(
            candidates_for_free_memory(64 * 1024 * 1024, BARS, FEATURES),
            None
        );
    }

    /// Built exactly as the engine builds it, so the test exercises the real
    /// shape rather than a convenient stand-in.
    fn native_error(status: i32) -> CudaPopulationError {
        CudaPopulationError::Native {
            operation: "evaluate",
            status,
            message: neoethos_gpu_cuda::population_status_message(status),
        }
    }

    /// The call sites attach context; this proves the typed error survives it.
    ///
    /// It did not: every site formatted the error into a fresh `anyhow!`, so
    /// the retry-smaller check could never find it. The population that could
    /// not fit went to the CPU instead of being halved, and a measured run put
    /// 770 500 of 778 205 validation items there with the card idle.
    #[test]
    fn out_of_memory_survives_the_context_the_call_sites_attach() {
        let native: Result<()> = Err(native_error(neoethos_gpu_cuda::STATUS_ALLOCATION_FAILED))
            .map_err(anyhow::Error::new)
            .context("prototype B evaluate");
        let error = native.expect_err("constructed as an error");
        assert!(
            is_capacity_exhaustion(&error),
            "an out-of-memory has to stay recognisable through the context: {error:#}"
        );

        // The rendering the log uses must still name the cause, or the next
        // investigation starts blind again.
        let rendered = format!("{error:#}");
        assert!(rendered.contains("prototype B evaluate"), "{rendered}");
        assert!(rendered.contains("device allocation failed"), "{rendered}");
    }

    /// A launch failure is a fault, not a size problem.
    #[test]
    fn a_launch_failure_is_not_retried_smaller() {
        let error = anyhow::Error::new(native_error(neoethos_gpu_cuda::STATUS_LAUNCH_FAILED))
            .context("prototype B evaluate");
        assert!(!is_capacity_exhaustion(&error));
    }
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
    head_len: usize,
) -> Result<Vec<[f64; 11]>> {
    let n_genes = long_thr.len();
    // Cut at the size that fits rather than in half.
    //
    // Halving overshoots whenever the population is a little over the limit: a
    // measured run had 25 600 candidates against a card holding 12 334, halved
    // to 12 800, halved again to 6 400 — four launches each using half the
    // card, where three full ones would do. The caller knows the limit when the
    // split is pre-emptive; after a capacity failure it does not, and passes
    // half.
    let half = head_len.clamp(1, n_genes.saturating_sub(1));

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

    let reusable = slot.as_ref().is_some_and(|r| {
        r.0.key == key
            && r.0.device == device
            && r.0.capacity >= capacity
            // Never hand the kernel more candidates than the workspace holds.
            && r.0.workspace_population >= n_genes
    });

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
            .map_err(anyhow::Error::new)
        .context("prototype B settings")?;
        let smc_rows: Vec<i8> = dataset.smc_data.iter().flatten().copied().collect();
        let adaptive_base = dataset.settings.to_settings().adaptive_base_pips.clone();

        let mut session = PopulationSession::create(device, capacity)
            .map_err(anyhow::Error::new)
        .context("prototype B session")?;
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
            .map_err(anyhow::Error::new)
        .context("prototype B dataset upload")?;

        *slot = Some(SendResident(ResidentSession {
            session,
            key,
            device,
            capacity,
            workspace_population: n_genes,
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
        .map_err(anyhow::Error::new)
        .context("prototype B")?;
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
        .map_err(anyhow::Error::new)
        .context("prototype B gene upload")?;

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
        .map_err(anyhow::Error::new)
        .context("prototype B scenario upload")?;

    let (event_id, _counters) = session
        .evaluate(&native_settings)
        .map_err(anyhow::Error::new)
        .context("prototype B evaluate")?;
    session
        .wait(event_id)
        .map_err(anyhow::Error::new)
        .context("prototype B wait")?;
    let rows = session
        .read_metrics()
        .map_err(anyhow::Error::new)
        .context("prototype B readback")?;

    // How full the trade slots actually are.
    //
    // Every candidate reserves MAX_TRADES_PER_CANDIDATE slots — 590 KB of the
    // 0.68 MB an H1 candidate now costs, so with the confidence column gone this
    // one array is 87 % of a candidate's device memory. The reservation is a
    // compile-time constant while what a candidate records is not, and nothing
    // measured the difference, so nothing could tell whether the card was full
    // of trades or of empty space.
    //
    // `accepted_trade_count` in the counters looks like the answer and is
    // always zero: the kernel never fills that field, it only stores the total
    // on the session after `wait`. Slot 8 of a metric row is the same fact per
    // candidate, already read back.
    let trade_counts = rows.iter().map(|row| row.values[8]).filter(|count| count.is_finite());
    let (peak, total) = trade_counts.fold((0.0f64, 0.0f64), |(peak, total), count| {
        (peak.max(count), total + count)
    });
    tracing::info!(
        target: "neoethos_search::eval",
        n_genes,
        reserved_slots = neoethos_gpu_cuda::MAX_TRADES_PER_CANDIDATE,
        busiest_candidate = peak as u64,
        mean_trades = (total / (rows.len() as f64).max(1.0)) as u64,
        peak_fill_pct = format!(
            "{:.2}",
            peak * 100.0 / neoethos_gpu_cuda::MAX_TRADES_PER_CANDIDATE as f64
        ),
        "trade slot usage — what was reserved per candidate against what was recorded"
    );

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
