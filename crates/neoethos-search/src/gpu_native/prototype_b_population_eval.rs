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

use anyhow::{Context, Result, bail};
use ndarray::ArrayView2;

// `NeoPopulationEvent` is deliberately not imported. It was here only so the
// deleted event budget could take its size, and the population lane touches no
// event record anywhere else — the type stays alive for the diagnostic readback
// contract, which is a different lane.
use neoethos_gpu_contracts::device::ScenarioDescriptor;

use crate::eval::{BacktestSettings, SmcRow};
use crate::gpu_native::prototype_a::{PrototypeADatasetUpload, PrototypeAGeneUpload};
use crate::gpu_native::scenario;
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
/// What the session allocates per candidate is `MAX_TRADES_PER_CANDIDATE`
/// outcome records — ~590 KB — plus the monthly buckets and its metric row, so
/// peak memory is a function of the requested population. That is the never-OOM
/// invariant inverted, and it is why this ceiling is computed from the device
/// instead: the caller asks for what the hardware has room for, never for what
/// it wants.
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
/// At ~0.6 MB per candidate this is around 600 MB — small enough for any
/// discrete card, large enough to keep the reduce busy. It is a floor for a
/// blind moment, not a target.
const CONSERVATIVE_BATCH: usize = 1_024;

/// Sustained device throughput, in scenario-bars per second.
///
/// Measured 2026-08 on an RTX 3090 at populations 16 384 and 131 072: 843-966 M
/// candidate-bars/s. The lower end is used, so the time estimate errs toward
/// SMALLER launches — an underestimate costs one extra launch, an overestimate
/// costs an unobservable hour.
///
/// PRE-FUSION MEASUREMENT — RE-MEASURE ON THE CARD. That number was taken while
/// the walk READ a precomputed `signal_values[bar]`. The fused walk now runs the
/// CSR accumulation, both thresholds and the SMC gate inside the same thread, so
/// its throughput is unmeasured and every launch-length estimate below is
/// arithmetic against a stale constant. It is only ever used to LOWER the
/// approved count, so a wrong value cannot cause an out-of-memory — it can only
/// make a launch longer or shorter than the target claims.
const SCENARIO_BARS_PER_SECOND: u64 = 843_000_000;

/// How long one submission should take.
///
/// This constant exists because fusing signal synthesis into the walk removed
/// the thing that used to bound a launch. Per-scenario device memory fell from
/// 4 811 048 B to 593 768 B, so a 24 GB card now fits on the order of 30 000
/// scenarios at 87 715 bars — and if the trade slots are ever dropped for a
/// metrics-only mode, over four million. At ~1 000 scenarios/second a four
/// million scenario launch runs for more than an hour with NO host observation
/// point: no progress, no telemetry line, no chance to cancel, and a failure
/// anywhere in it discards all of it.
///
/// So sizing gained a second term. Memory says what the card can HOLD; this
/// says what the operator can WATCH. The launch takes the smaller.
const TARGET_LAUNCH_SECONDS: u64 = 20;

/// Below this a launch stops filling the card, so the time term must never push
/// under it — the occupancy knee measured on the 3090.
///
/// It cannot cause an out-of-memory: the time term is only ever combined with
/// the memory term by `min`, so the memory ceiling still wins outright.
const OCCUPANCY_KNEE: u64 = 16_384;

/// Scenarios that keep one launch inside [`TARGET_LAUNCH_SECONDS`].
///
/// Peak memory is untouched by this: it can only lower the count.
///
/// The occupancy floor WINS over the target when the series is long enough, and
/// it says so rather than leaving the operator to infer it: at 5 270 000 bars
/// the honest term is ~3 200 scenarios and the floor lifts it to 16 384, an
/// estimated 102 s launch against a 20 s target. That is the right trade — a
/// launch that does not fill the card wastes more than it saves — but an
/// unobservable 102 s is exactly what this term exists to stop being a surprise.
fn candidates_for_target_launch(bars: usize) -> u64 {
    let bars = (bars as u64).max(1);
    let raw = SCENARIO_BARS_PER_SECOND.saturating_mul(TARGET_LAUNCH_SECONDS) / bars;
    if raw < OCCUPANCY_KNEE {
        tracing::warn!(
            target: "neoethos_search::eval",
            bars,
            target_seconds = TARGET_LAUNCH_SECONDS,
            time_term = raw,
            floor = OCCUPANCY_KNEE,
            estimated_seconds = OCCUPANCY_KNEE.saturating_mul(bars) / SCENARIO_BARS_PER_SECOND,
            "the occupancy floor overrides the launch-length target — this launch will run \
             longer than the target with no host observation point"
        );
    }
    raw.max(OCCUPANCY_KNEE)
}

/// What the sizing arithmetic concluded.
///
/// THE TWO "NO ANSWER" CASES ARE NOT THE SAME ANSWER. This was an `Option`, and
/// `None` meant both "free VRAM is unreadable" and "this card has no room for
/// the dataset at all" — so the branch taken when the card is provably too small
/// was the branch that re-used the LAST SUCCESSFUL LARGE BATCH. That is the
/// `unwrap_or(usize::MAX)` incident rebuilt with more steps: a 12 GB card asked
/// for an M1 dataset it cannot hold would have launched whatever an earlier M5
/// run had fitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sizing {
    /// The card can host this many scenarios in one launch.
    Fits(usize),
    /// `cudaMemGetInfo` did not answer. Size conservatively and carry on.
    Unreadable,
    /// The card answered and it has no usable room. Not a batching problem.
    NoRoom { dataset_bytes: u64, budget_bytes: u64 },
}

fn candidates_that_fit(
    device: usize,
    bars: usize,
    feature_count: usize,
    month_capacity: usize,
) -> Sizing {
    match neoethos_gpu_cuda::device_free_memory_bytes(device) {
        Some(free) => candidates_for_free_memory(free, bars, feature_count, month_capacity),
        None => Sizing::Unreadable,
    }
}

/// The largest candidate count worth putting in ONE submission to this engine,
/// for callers that pick their own chunk size before calling the evaluator.
///
/// This is a BATCHING number, never a search number: genes are independent, so
/// any chunking of a population produces bit-identical per-gene metrics — the
/// only thing a caller changes by consulting this is how many launches the work
/// takes. The quality screen used a constant 256 here, which ran the card at
/// ~42 M candidate-bars/s where a launch sized to this ceiling sustains
/// 843-966 M (measured 2026-08, RTX 3090, populations 16 384 / 131 072).
///
/// `None` means "no card, or its free memory cannot be read" — the caller keeps
/// whatever conservative constant it had, and nothing about the run changes.
/// The evaluator below still re-checks and splits on its own, so a stale answer
/// here costs one split, never an OOM.
///
/// Since the walk stopped materialising per-scenario signal columns, this is
/// bounded by [`TARGET_LAUNCH_SECONDS`] as often as by VRAM — a launch is kept
/// short enough to be observed, not merely small enough to fit.
pub(crate) fn submission_ceiling(device: usize, bars: usize, feature_count: usize) -> Option<usize> {
    if !prototype_b_available() {
        return None;
    }
    match candidates_that_fit(
        device,
        bars,
        feature_count,
        crate::eval::current_backtest_runtime_overrides().month_capacity,
    ) {
        Sizing::Fits(fits) => Some(fits),
        Sizing::Unreadable | Sizing::NoRoom { .. } => None,
    }
}

/// The arithmetic, separated from the device query so it can be checked without
/// a card. The numbers it produces decide whether a run uses the GPU at all.
///
/// TWO terms now, and they answer different questions:
///
///   * memory — what the card can HOLD. Unchanged in intent, but the
///     per-scenario cost collapsed: `bars * 5` is gone because the `signal_values`
///     (i8) and `signal_confidences` (f32) columns are gone. Those two carried a
///     value from the signal kernel to the reduce and nothing else; the walk
///     now produces it in registers as it advances. At 843 456 bars they were
///     4 217 280 B per scenario — 87.7 % of the old 4 811 048 B, and the sole
///     reason a 24 GB card resolved to 3 316 scenarios and the quality screen
///     grew a six-chunk loop. What is left is real: 8 192 trade slots at 72 B
///     plus 3 944 B of monthly buckets and metric row = 593 768 B.
///
///   * time — what the operator can WATCH. See [`TARGET_LAUNCH_SECONDS`].
///     Memory no longer binds anywhere near as early, so without this a launch
///     could run for an hour with no observation point.
///
/// The launch takes the SMALLER. Peak memory therefore stays a function of the
/// hardware alone — the time term can only ever lower the count, never raise it
/// past what the card was measured to hold.
fn candidates_for_free_memory(
    free: u64,
    bars: usize,
    feature_count: usize,
    month_capacity: usize,
) -> Sizing {
    // Leave three tenths for context, fragmentation and the allocator's own
    // bookkeeping.
    let budget = (free / 10) * 7;
    // The dataset is charged TWICE for the indicator matrix on purpose. The
    // device uploads it feature-major into a staging buffer and transposes it
    // into the permanent bar-major one, so both exist for the duration of one
    // kernel at upload. Budgeting for the peak is what keeps that transient
    // from being the allocation that fails.
    let indicators = bars as u64 * feature_count as u64 * 4;
    // Every bars-scaled array `upload_dataset` allocates, named so the next one
    // added has an obvious place to go:
    //   close + high + low                    3 x f64
    //   months + days + timestamps            3 x i64
    //   adaptive_base_pips                    1 x f64   (WAS MISSING)
    //   gap_flags                             1 x u8    (WAS MISSING)
    //   smc_rows                             11 x i8
    //   indicators, twice (staging+permanent)
    let dataset = bars as u64 * (3 * 8 + 3 * 8 + 8 + 1 + 11) + 2 * indicators;
    // The card answered and the DATASET alone does not fit. That is not a
    // batching problem: no work list, however small, makes a 5.4 GB indicator
    // matrix smaller. Saying so here is what stops the caller from sizing a
    // launch out of a stale number and then discovering it by failing.
    if dataset.saturating_add(64 * 1024 * 1024) >= budget {
        return Sizing::NoRoom { dataset_bytes: dataset, budget_bytes: budget };
    }
    // The 64 MiB is the reserve for allocator fragmentation, the context, and
    // the GENE arrays — which are not bars-scaled and not scenario-scaled:
    // `population * 51 B + (population + 1) * 4 B + terms * 8 B`. The largest
    // gene array any lane uploads is the quality screen's 131 072 clones, which
    // is ~7 MB of that reserve; a work list is scenarios, and those are charged
    // per scenario below.
    let room = budget
        .saturating_sub(dataset)
        .saturating_sub(64 * 1024 * 1024);
    // Per SCENARIO: its trade slots, its monthly buckets, its metric row, and
    // the nine scenario-descriptor arrays the upload stages on the device.
    // No signal column, no confidence column, no event buffer — none of them
    // exists on the device any more.
    //
    // `month_capacity` is a PARAMETER here, not the literal 3 840 it used to be
    // folded into. The device allocates `scenario_count * month_capacity`
    // doubles twice, and `month_capacity` is an operator-configurable runtime
    // override with no upper bound — so hardcoding the default made peak device
    // memory a function of a user parameter, in the one function that exists to
    // stop exactly that.
    let outcome = std::mem::size_of::<neoethos_gpu_contracts::device::NeoPopulationOutcome>() as u64;
    let metric_row =
        std::mem::size_of::<neoethos_gpu_contracts::device::NeoPopulationMetricRow>() as u64;
    // 8 (base id) + 8 (scenario id) + 8 (rng counter) + 8 (window offset)
    // + 4 (window len) + 4 (type) + 4 (spread) + 4 (slippage) + 8 (commission)
    const SCENARIO_UPLOAD_BYTES: u64 = 56;
    let per_candidate = neoethos_gpu_cuda::MAX_TRADES_PER_CANDIDATE * outcome
        + 2 * month_capacity as u64 * 8
        + metric_row
        + SCENARIO_UPLOAD_BYTES;
    let fits = room / per_candidate.max(1);
    // Below this the card cannot do useful work and the CPU lane is the honest
    // answer.
    if fits < 16 {
        return Sizing::NoRoom { dataset_bytes: dataset, budget_bytes: budget };
    }
    Sizing::Fits(fits.min(candidates_for_target_launch(bars)) as usize)
}

/// The `max_events` the native session is created with — a formality.
///
/// There WAS an `event_capacity` here. It read free VRAM, subtracted the
/// dataset and the per-candidate workspaces, divided the remainder by
/// `size_of::<NeoPopulationEvent>() + size_of::<NeoPopulationOutcome>()`, and
/// `bail!`ed the whole evaluation when the answer came out under 1 024.
///
/// Every one of those bytes was imaginary. `session->events` is declared in the
/// native session and freed in `release_workspace()`, and there is no
/// `device_alloc` for it anywhere in the allocation block — the only kernel
/// that ever filled it has been commented out since the reduce started opening
/// positions from the signal directly. The engine has not allocated one event
/// record in a long time.
///
/// So the budget guarded nothing and cost three real things: it could abort an
/// evaluation over a buffer that does not exist; it was half the session-reuse
/// predicate, so a session was rebuilt whenever the imaginary number moved; and
/// because it SUBTRACTS `population * bars * 5`, a larger population asked for
/// a smaller capacity — the arithmetic ran backwards with respect to the one
/// quantity that actually matters.
///
/// What still bounds device memory is [`candidates_for_free_memory`], which
/// counts only allocations that exist. This constant just satisfies the ABI's
/// "must be non-zero" check; the device stores it nowhere and reads it never.
const VESTIGIAL_SESSION_MAX_EVENTS: usize = 1;


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
    // Indicators, by INDEXING, exactly as `sample_hash` does it.
    //
    // This walked the whole matrix — `indicators.iter().enumerate()` with an
    // `i % stride` test per element — to keep 256 of them. On EURUSD M5 that is
    // 843 456 bars x 64 features = 53 981 184 elements iterated and 53 981 184
    // remainders computed, on EVERY launch, to select 256 values. The sibling
    // `sample_hash` twenty lines up already did the obvious thing: compute the
    // stride, then index. There was no reason for the two to differ.
    //
    // `ArrayView2` indexing is O(1) whatever the layout, so nothing is given up
    // by not iterating: a non-contiguous view is addressed by its strides just
    // as a contiguous one is. The sampled positions are the same ones the
    // modulo walk selected (0, stride, 2*stride, ...) plus the final element,
    // which `sample_hash` has always included and this had not. The key is
    // process-local and never persisted, so the absolute value is free to
    // change; what matters is that it still separates two different matrices,
    // and one more sample can only help.
    let flat = indicators.len();
    flat.hash(&mut hasher);
    if flat > 0 {
        const SAMPLES: usize = 256;
        let stride = (flat / SAMPLES).max(1);
        let rows = indicators.nrows();
        let cols = indicators.ncols();
        let mut index = 0usize;
        while index < flat {
            // Logical row-major order over the view, matching what `iter()`
            // yields for a standard C-ordered `ArrayView2`.
            let value = indicators[(index / cols.max(1), index % cols.max(1))];
            value.to_bits().hash(&mut hasher);
            index += stride;
        }
        // The last element is cheap and pins a change in the tail, which a
        // stride that divides the length would otherwise never visit.
        indicators[(rows - 1, cols - 1)].to_bits().hash(&mut hasher);
    }
    feature_count.hash(&mut hasher);
    for slice in [months, days, timestamps] {
        sample_hash(slice, |v| *v as u64, &mut hasher);
    }
    // SMC flags are -1/0/+1, and `-1i8 as u64` is 18 446 744 073 709 551 615:
    // two of them overflow the sum. Release wrapped it silently and every parity
    // run passed; a debug build panics, which is how it surfaced — after the 18
    // parity tests had been reported green three times from release runs.
    //
    // Summing also threw position away — [1, -1] and [-1, 1] hashed the same.
    // Folding fixes both. `wrapping_*` is the intent here, not an oversight:
    // this is a hash and mixing is what it is for.
    sample_hash(
        smc_data,
        |row| {
            row.iter().fold(0u64, |acc, v| {
                acc.wrapping_mul(31).wrapping_add(*v as i64 as u64)
            })
        },
        &mut hasher,
    );
    settings_key(settings, &mut hasher);
    hasher.finish()
}

/// Hash the settings that identify a resident session, field by field.
///
/// This was `format!("{settings:?}").hash(..)`. `BacktestSettings` carries
/// `adaptive_base_pips: Option<Arc<[f64]>>` — 843 456 f64 on EURUSD M5 — and
/// `Debug` renders every one of them, so each launch built a ~17 MB `String`
/// and then SipHashed all of it, purely to decide whether the dataset already
/// on the card was the same one. That is the single most expensive line in a
/// lane whose entire purpose is to avoid re-uploading data.
///
/// It was also wrong in the other direction. `Debug` folds in `sl_pips` and
/// `tp_pips`, which do not travel to the device at all — they are per-gene, in
/// the `stop_pips` / `target_pips` arrays, and `NeoPopulationSettings` has no
/// field for either. The chunk loop takes them from the FIRST GENE of the chunk
/// (`discovery.rs`), so two chunks over the same bars produced two different
/// keys, and 16 bytes of computationally dead scalar destroyed a 261 MiB
/// resident dataset and rebuilt it.
///
/// The destructure is EXHAUSTIVE on purpose, and that is the point of this
/// function rather than a hand-written field list: add a field to
/// `BacktestSettings` and this stops compiling. A field silently left out of a
/// cache key is not a missing feature, it is a wrong cache HIT — the run
/// continues, on the card, against the wrong data, with nothing to show for it.
fn settings_key(settings: &BacktestSettings, hasher: &mut impl std::hash::Hasher) {
    use std::hash::Hash;
    let BacktestSettings {
        // ── Deliberately NOT hashed ──────────────────────────────────────────
        //
        // These two are per-GENE, not per-dataset. They reach the device in the
        // `stop_pips` / `target_pips` arrays uploaded with the genes on every
        // call, and `NeoPopulationSettings` has no field for either — the DTO
        // the device settings are built from (`SnapshotSettingsDto`) drops them
        // outright. Nothing that is resident depends on them, so including them
        // could only ever cause a spurious rebuild, and it did.
        sl_pips: _,
        tp_pips: _,

        // ── Everything else, by bits ─────────────────────────────────────────
        //
        // All of these DO reach the device, either in the resident
        // `adaptive_base_pips` upload or in the `NeoPopulationSettings` struct
        // that is stored on the resident session and replayed on every
        // `evaluate`. Keeping them here is what makes that stored copy safe:
        // change any one of them and the key changes, the session is rebuilt,
        // and the stored settings are rebuilt with it. Drop one from this list
        // and the run would keep using the previous spread, commission or hold
        // limit while reporting the new one.
        max_hold_bars,
        min_hold_bars,
        max_trades_per_day,
        gap_threshold_ms,
        trailing_enabled,
        trailing_atr_multiplier,
        trailing_be_trigger_r,
        trailing_min_lock_pips,
        pip_value,
        spread_pips,
        commission_per_trade,
        pip_value_per_lot,
        kill_zones_enabled,
        session_spread_profile,
        swap_long_pips_per_day,
        swap_short_pips_per_day,
        pnl_conversion_fee_rate,
        risk_based_sizing,
        risk_per_trade_min,
        risk_per_trade_max,
        high_quality_confidence,
        adaptive_base_pips,
        adaptive_vol_mult,
        adaptive_rr,
    } = settings;

    max_hold_bars.hash(hasher);
    min_hold_bars.hash(hasher);
    max_trades_per_day.hash(hasher);
    gap_threshold_ms.hash(hasher);
    trailing_enabled.hash(hasher);
    // Floats by bit pattern, never by value: `f64` is not `Hash`, and `-0.0 ==
    // 0.0` while `NaN != NaN`, so a value comparison would both merge distinct
    // settings and refuse to recognise identical ones.
    trailing_atr_multiplier.to_bits().hash(hasher);
    trailing_be_trigger_r.to_bits().hash(hasher);
    trailing_min_lock_pips.to_bits().hash(hasher);
    pip_value.to_bits().hash(hasher);
    spread_pips.to_bits().hash(hasher);
    commission_per_trade.to_bits().hash(hasher);
    pip_value_per_lot.to_bits().hash(hasher);
    kill_zones_enabled.hash(hasher);
    match session_spread_profile {
        // The discriminant is hashed separately from the buckets so a profile
        // whose three values happen to equal the flat `spread_pips` is still a
        // different session from no profile at all.
        Some(profile) => {
            1u8.hash(hasher);
            profile.asian_pips.to_bits().hash(hasher);
            profile.overlap_pips.to_bits().hash(hasher);
            profile.late_ny_pips.to_bits().hash(hasher);
        }
        None => 0u8.hash(hasher),
    }
    swap_long_pips_per_day.to_bits().hash(hasher);
    swap_short_pips_per_day.to_bits().hash(hasher);
    pnl_conversion_fee_rate.to_bits().hash(hasher);
    risk_based_sizing.hash(hasher);
    risk_per_trade_min.to_bits().hash(hasher);
    risk_per_trade_max.to_bits().hash(hasher);
    high_quality_confidence.to_bits().hash(hasher);
    // The one series in here, and the reason settings are in the key at all:
    // `adaptive_base_pips` is uploaded WITH the dataset, so two runs over
    // identical bars but different adaptive stops are different datasets on the
    // device. Sampled the same way as close/high/low — length plus a strided
    // sample — never by `Debug`, which is what made this line cost 17 MB.
    match adaptive_base_pips {
        Some(base) => {
            1u8.hash(hasher);
            sample_hash(base, |v| v.to_bits(), hasher);
        }
        None => 0u8.hash(hasher),
    }
    adaptive_vol_mult.to_bits().hash(hasher);
    adaptive_rr.to_bits().hash(hasher);
}

struct ResidentSession {
    session: PopulationSession,
    key: u64,
    device: i32,
    /// The SCENARIO COUNT the device workspace was actually allocated for.
    ///
    /// The outcome array is sized `scenarios * MAX_TRADES_PER_CANDIDATE` when
    /// the workspace is built, and the kernel indexes it by the CURRENT
    /// scenario count — which `upload_scenarios` overwrites on every call.
    /// Reusing a session for a longer work list therefore writes past its end,
    /// into whatever the allocator placed next. (There used to be two more
    /// arrays with the same hazard, `signal_values` and `signal_confidences`,
    /// sized `population * bars`; the walk now synthesises the signal in
    /// registers and they do not exist.)
    ///
    /// This counted GENES until the scenario became the unit of work, and the
    /// two are no longer the same number: a quality-screen launch is 174 genes
    /// and 17 574 scenarios, so a gene-count guard would have approved a reuse
    /// that overruns the workspace by a factor of a hundred.
    ///
    /// Nothing stopped that at all before either guard existed. The reuse test
    /// was `capacity >= capacity` on the old event budget, and that budget
    /// SUBTRACTED `population * bars * 5`, so a larger population asked for LESS
    /// capacity and passed more easily — the check was not merely silent about
    /// size, it was inverted with respect to it. A session built for 256 was
    /// reused for 25 600. The phantom budget is gone; this field is what
    /// replaced it, and it is the ONLY thing standing between a reused session
    /// and that write.
    ///
    /// It records what the DEVICE workspace holds, not what the last call
    /// asked for. The native side grows the workspace instead of matching it
    /// exactly (`workspace_scenarios < scenario_count`), so a shorter work list
    /// reuses a larger workspace and this value must not be lowered to follow
    /// it — lowering it would be harmless for safety but would force a rebuild
    /// the device no longer performs, and the two sides would disagree about
    /// what is resident.
    workspace_scenarios: usize,
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

/// What a learned launch size is a fact ABOUT.
///
/// A fit is measured in scenarios, and the room available for scenarios is
/// `budget - dataset(bars, features)` — which changes by an order of magnitude
/// between an M5 and an M1 dataset, and between a 12 GB and a 24 GB card. A
/// single process-wide `AtomicUsize` keyed to nothing therefore replayed one
/// symbol/timeframe's fit as another's launch size, and one dataset's FAILURE
/// permanently capped every other dataset's launches for the life of the
/// process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct LimitKey {
    device: usize,
    bars: usize,
    feature_count: usize,
    month_capacity: usize,
}

/// Largest work list known to have fitted THIS (device, dataset shape).
///
/// Discovering the limit by failing costs the whole attempt: the kernel runs,
/// exhausts capacity, and its work is discarded before the halves are retried.
/// Paying that once is the price of not knowing how many trades a population
/// emits; paying it every generation is waste. A 2026-07-29 M3 run spent 391 s
/// on a single population evaluation that way, against a benchmark rate that
/// would place it near 4 s.
///
/// So the size that worked is remembered and used as the starting point — for
/// the shape it was learned on, and no other. An absent entry means "not yet
/// learned": the first call tries the whole work list, as it should.
fn learned_batch_limits()
-> &'static std::sync::Mutex<std::collections::HashMap<LimitKey, usize>> {
    static LIMITS: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<LimitKey, usize>>,
    > = std::sync::OnceLock::new();
    LIMITS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn learned_batch_limit(key: LimitKey) -> usize {
    learned_batch_limits()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&key)
        .copied()
        .unwrap_or(usize::MAX)
}

/// Raise the learned ceiling: this size was accepted.
fn learn_batch_success(key: LimitKey, scenarios: usize) {
    let mut limits = learned_batch_limits()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let entry = limits.entry(key).or_insert(scenarios);
    if *entry != usize::MAX {
        *entry = (*entry).max(scenarios);
    }
}

/// Lower the learned ceiling: this size was refused for capacity.
fn learn_batch_failure(key: LimitKey, ceiling: usize) {
    let mut limits = learned_batch_limits()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let entry = limits.entry(key).or_insert(ceiling);
    *entry = (*entry).min(ceiling);
}

/// A failure that must NEVER be retried by making the work list smaller.
///
/// `upload_dataset` returns `STATUS_ALLOCATION_FAILED` exactly like a workspace
/// exhaustion, and `is_capacity_exhaustion` cannot tell them apart from the
/// status alone — so a dataset that does not fit was halved, and halved, and
/// halved, down to `n_scenarios < 2`. For a 17 748-scenario quality screen that
/// is ~35 000 failed launches, each one re-attempting the same multi-gigabyte
/// `cudaMalloc` that just failed, and the lane's own history records that one
/// such mid-stream failure left the CUDA context unusable for the 27
/// evaluations after it.
///
/// Slicing the descriptor array cannot change the dataset by one byte. This
/// marker is attached to every upload error so the retry can say so.
#[derive(Debug)]
struct NotAWorkListSizeProblem(&'static str);

impl std::fmt::Display for NotAWorkListSizeProblem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} does not depend on the work list size — splitting it cannot help",
            self.0
        )
    }
}

impl std::error::Error for NotAWorkListSizeProblem {}

/// Evaluate one full-series scenario per gene — the identity work list.
///
/// This is what every caller outside the quality screen wants, and it is the
/// parity floor for the whole scenario change: a descriptor array of nothing but
/// `base_scenario` describes exactly the evaluation the engine performed before
/// scenarios existed. The zeroed fields the old code wrote by hand are now
/// written by [`scenario::base_scenario`], which matters for one of them —
/// `spread_ticks` and `commission_micros` are now `-1` ("no override") where the
/// literal was `0`, and `0` has since become a REAL override meaning "charge no
/// spread". Constructing descriptors anywhere but through the builders is how
/// that would silently become a free-trading backtest.
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
    let scenarios: Vec<ScenarioDescriptor> = (0..n_genes as u64)
        .map(|candidate| scenario::base_scenario(candidate, candidate, bars))
        .collect();
    try_evaluate_scenarios_b(
        close, high, low, indicators, gene_offsets, gene_indices, gene_weights, long_thr,
        short_thr, month_idx, day_idx, timestamps, sl_pips, tp_pips, stop_vol_mult, smc_data,
        gene_smc_flags, gate_threshold, smc_weights, settings, device_override, &scenarios,
    )
}

/// Evaluate an arbitrary work list on Prototype B, splitting when the card
/// cannot hold the trades it produces.
///
/// THE SCENARIO IS THE UNIT OF WORK. The gene arrays stay whole and resident;
/// what is sized, split and submitted is the DESCRIPTOR ARRAY. That is the
/// difference that turns the quality screen's seven launches — six Monte-Carlo
/// chunks and one sensitivity pass — into one.
///
/// The device allocation is bounded by VRAM, and `candidates_for_free_memory`
/// predicts it — but a prediction can still be beaten by fragmentation or by
/// another process taking memory between the query and the allocation. The
/// engine reports an out-of-memory distinctly (`is_capacity_exhausted`) rather
/// than as a generic failure, so the answer is to give it less work rather than
/// to give up: the work list is cut and each part evaluated in turn.
///
/// Scenarios are independent — each is one thread reading shared, read-only gene
/// and dataset arrays — so a split result equals the whole one. This is the
/// never-OOM rule applied as intended: peak memory follows the hardware, and an
/// oversized workload gets slower instead of failing.
///
/// Splitting the SCENARIOS rather than the genes is also strictly simpler than
/// what it replaces. The old split sliced the CSR gene arrays and rebased the
/// tail's offsets — get that wrong and it evaluates the wrong terms silently
/// rather than erroring. Slicing a descriptor array cannot be got wrong that
/// way: every descriptor carries its own gene index, so a slice is still a
/// correct work list whatever it contains.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_evaluate_scenarios_b(
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
    scenarios: &[ScenarioDescriptor],
) -> Result<Vec<[f64; 11]>> {
    let n_genes = long_thr.len();
    // What the launch is sized by, from here down.
    let n_scenarios = scenarios.len();
    if n_scenarios == 0 {
        return Ok(Vec::new());
    }
    // Refused here, before anything is uploaded, because the device cannot
    // detect either fault: a gene index past the end of the population is an
    // out-of-bounds read of thresholds and CSR offsets that still produces a
    // metric row, and a window past the end of the series is an out-of-bounds
    // read of prices. The native side checks again — two independent guards,
    // like the workspace-population pair — but this one can name the index.
    if let Err(detail) = scenario::validate_scenarios(scenarios, n_genes, close.len()) {
        bail!("prototype B scenario list: {detail}");
    }

    // Everything the launch size is a fact about, in one key.
    let limit_key = LimitKey {
        device: device_override.unwrap_or(0),
        bars: close.len(),
        feature_count: indicators.nrows(),
        month_capacity: crate::eval::current_backtest_runtime_overrides().month_capacity,
    };
    // Start at the size already known to fit THIS shape rather than
    // rediscovering the limit by throwing away a full evaluation every
    // generation.
    let learned = learned_batch_limit(limit_key);
    // What the card can hold is knowable before asking it, so ask first. The
    // retry below still exists for what cannot be predicted — how many trades a
    // population actually emits — but it should never be reached for a size
    // that was arithmetic all along.
    let fits = match candidates_that_fit(
        limit_key.device,
        limit_key.bars,
        limit_key.feature_count,
        limit_key.month_capacity,
    ) {
        Sizing::Fits(fits) => fits,
        // The card was READ and it has no room. Sizing a launch here — from a
        // constant, or worse from a fit learned on some other dataset — is how
        // a 12 GB card came to be handed a batch that needed 15.9 GB of
        // workspace on top of a dataset it could not hold either. There is no
        // batch that fixes it, so say so and let the caller's own policy
        // decide: with NEOETHOS_REQUIRE_GPU set that is a loud failure, and
        // without it the CPU lane is the honest answer.
        Sizing::NoRoom { dataset_bytes, budget_bytes } => {
            bail!(
                "prototype B: this device has no room for the dataset — it needs \
                 {dataset_bytes} B (close/high/low, months/days/timestamps, the adaptive stop \
                 base, gap flags, SMC rows and the indicator matrix twice for the bar-major \
                 transpose) against a {budget_bytes} B budget of free VRAM. {n_scenarios} \
                 scenarios over {} bars x {} features. Splitting the work list cannot help: \
                 the dataset is the same size whatever the launch asks for.",
                limit_key.bars,
                limit_key.feature_count
            );
        }
        // Not knowing how much room there is is a reason to ask for less, not
        // for everything. This read `unwrap_or(usize::MAX)` — unknown meant
        // unlimited — and a measured run launched 24 700 candidates against a
        // card that holds ~16 300, died with a stream synchronization failure,
        // and left the CUDA context unusable: the 27 evaluations after it could
        // not even read free memory, so 31 859 items went to the CPU from one
        // bad guess. It then read `last_known_fit()`, which is the same mistake
        // with a better disguise — the last SUCCESSFUL large batch is exactly
        // the wrong guess to make when the card has stopped answering.
        Sizing::Unreadable => {
            tracing::warn!(
                target: "neoethos_search::eval",
                n_genes,
                n_scenarios,
                fallback = CONSERVATIVE_BATCH,
                "cannot read free device memory — sizing the batch conservatively"
            );
            CONSERVATIVE_BATCH
        }
    };
    if fits < n_scenarios {
        tracing::info!(
            target: "neoethos_search::eval",
            n_genes,
            n_scenarios,
            fits,
            "work list exceeds what the card can host — splitting before asking"
        );
    }
    let learned = learned.min(fits);
    if n_scenarios > learned && n_scenarios > 1 {
        return split_and_evaluate(
            close, high, low, indicators, gene_offsets, gene_indices, gene_weights, long_thr,
            short_thr, month_idx, day_idx, timestamps, sl_pips, tp_pips, stop_vol_mult, smc_data,
            gene_smc_flags, gate_threshold, smc_weights, settings, device_override, scenarios,
            learned,
        );
    }

    let attempt = evaluate_population_b_batch(
        close, high, low, indicators, gene_offsets, gene_indices, gene_weights, long_thr,
        short_thr, month_idx, day_idx, timestamps, sl_pips, tp_pips, stop_vol_mult, smc_data,
        gene_smc_flags, gate_threshold, smc_weights, settings, device_override, scenarios,
    );
    let Err(error) = attempt else {
        // This size fits; keep it as the starting point for the next call over
        // this same shape.
        learn_batch_success(limit_key, n_scenarios);
        return attempt;
    };
    // Only a capacity exhaustion is worth retrying smaller. Anything else is a
    // fault, and halving the work would just hide it behind a slower failure.
    //
    // "Capacity exhaustion" now excludes the uploads. `upload_dataset` reports
    // the same `STATUS_ALLOCATION_FAILED` as a workspace exhaustion, so without
    // the marker a dataset that does not fit was halved down to one scenario —
    // ~35 000 failed launches for a 17 748-scenario screen, each re-attempting
    // the identical multi-gigabyte `cudaMalloc`.
    if !is_capacity_exhaustion(&error) || n_scenarios < 2 {
        return Err(error);
    }
    // Remember the ceiling so the next generation does not pay this again.
    learn_batch_failure(limit_key, n_scenarios / 2);
    tracing::info!(
        target: "neoethos_search::eval",
        n_genes,
        n_scenarios,
        learned = learned_batch_limit(limit_key),
        "the work list emitted more trades than the card can hold — splitting, and \
         remembering the limit so later launches start there"
    );
    return split_and_evaluate(
        close, high, low, indicators, gene_offsets, gene_indices, gene_weights, long_thr,
        short_thr, month_idx, day_idx, timestamps, sl_pips, tp_pips, stop_vol_mult, smc_data,
        gene_smc_flags, gate_threshold, smc_weights, settings, device_override, scenarios,
        n_scenarios / 2,
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
/// Asking the error what it is removes that coupling.
///
/// There was a second arm here matching the string `"leaves no room for events
/// on this device"` — the host-side event budget's own `bail!`. That budget was
/// sizing a buffer the kernel never allocated, so it is gone and so is the
/// phrase; a substring match against a message no code can produce is dead
/// weight that reads like a live path.
///
/// This only works because the call sites attach context rather than formatting
/// the error into a new one — `anyhow!("...: {error}")` renders the source and
/// throws the value away, which left this check unable to find anything and
/// made it dead code the moment it was written.
fn is_capacity_exhaustion(error: &anyhow::Error) -> bool {
    // An UPLOAD that ran out of memory is never a work-list size problem. The
    // dataset, the genes and the descriptor array are the same size whatever
    // the launch asks for, and the split halves only the descriptor array — so
    // retrying a failed `upload_dataset` smaller re-attempts the identical
    // `cudaMalloc` at every leaf of the recursion. See
    // [`NotAWorkListSizeProblem`].
    //
    // `anyhow::Error::downcast_ref` is what searches the context chain.
    // `error.chain()` does NOT work here: it yields anyhow's internal
    // `ContextError` wrapper as `&dyn Error`, and downcasting THAT to the
    // marker fails — the check compiles, always answers "no", and the marker
    // silently does nothing. `a_dataset_allocation_failure_is_never_split`
    // caught exactly that.
    if error.downcast_ref::<NotAWorkListSizeProblem>().is_some() {
        return false;
    }
    error
        .downcast_ref::<CudaPopulationError>()
        .is_some_and(CudaPopulationError::is_capacity_exhausted)
}

#[cfg(test)]
mod capacity_detection_tests {
    use super::*;

    /// Every sizing test drives the arithmetic at the DEFAULT month capacity
    /// unless it is testing that knob, so the number under test is the one
    /// production computes.
    const MONTHS: usize = 240;

    fn fits_or_panic(free: u64, bars: usize, features: usize) -> usize {
        match candidates_for_free_memory(free, bars, features, MONTHS) {
            Sizing::Fits(fits) => fits,
            other => panic!("expected a fit, got {other:?}"),
        }
    }

    /// What fusion bought, stated as a number rather than as a claim.
    ///
    /// This test used to assert `fits < 25_000` — that the population
    /// validation asks for (250 folds x 100 Monte-Carlo runs) could NOT be done
    /// in one launch on a 24 GB card. That was true while every scenario also
    /// carried a `bars`-long i8 signal column and an f32 confidence column:
    /// 4 217 280 B of the 4 811 048 B per scenario at 843 456 bars, 87.7 % of
    /// the total, existing only to hand a value between two kernels.
    ///
    /// Those columns are gone — the walk synthesises the signal in registers —
    /// so the per-scenario cost is 593 768 B and the same card hosts the whole
    /// request. Pinning the OLD ceiling would pin the defect.
    ///
    /// What must stay true is the shape: a finite ceiling, well clear of the
    /// request, and still derived from the card rather than from the caller.
    #[test]
    fn the_population_validation_asks_for_now_fits_a_24gb_card() {
        const BARS: usize = 87_715;
        const FEATURES: usize = 257;
        let fits = fits_or_panic(24 * 1024 * 1024 * 1024, BARS, FEATURES);
        assert!(
            fits >= 25_000,
            "deleting 4.2 MB of per-scenario handoff must let the 25 000-candidate \
             validation call run in one launch: {fits}"
        );
        // And it is still a ceiling, not "unlimited". The old failure mode was
        // `unwrap_or(usize::MAX)` — unknown treated as unbounded — which
        // launched 24 700 candidates at a card holding ~16 300 and left the
        // CUDA context unusable for the rest of the run.
        assert!(
            fits < 10_000_000,
            "the ceiling has to remain a real bound: {fits}"
        );
    }

    /// Memory stopped being the binding constraint, so time became one.
    ///
    /// At EURUSD M5 dimensions the memory term alone approves ~29 000
    /// scenarios; the operator-visibility term cuts that to ~20 000, which is
    /// one launch of about `TARGET_LAUNCH_SECONDS`. Both must be in play, and
    /// the smaller must win.
    #[test]
    fn a_launch_stays_observable_once_memory_stops_binding() {
        // Real EURUSD M5: 843 456 bars, 64 features.
        const BARS: usize = 843_456;
        const FEATURES: usize = 64;
        let approved = fits_or_panic(24 * 1024 * 1024 * 1024, BARS, FEATURES);
        let time_term = candidates_for_target_launch(BARS) as usize;
        assert_eq!(
            approved, time_term,
            "at these dimensions the launch length binds, not the card"
        );
        // ~843 M scenario-bars/s over 843 456 bars is ~1 000 scenarios/s, so
        // the approved batch has to land near the target rather than an hour
        // away from it.
        let seconds = approved as u64 * BARS as u64 / SCENARIO_BARS_PER_SECOND;
        assert!(
            seconds <= TARGET_LAUNCH_SECONDS,
            "an approved launch estimates at {seconds} s against a {TARGET_LAUNCH_SECONDS} s target"
        );

        // The time term must never be the reason a launch exceeds what the card
        // holds — it is combined by `min`, so a tiny card still wins.
        let small = fits_or_panic(4 * 1024 * 1024 * 1024, BARS, FEATURES);
        assert!(
            small < time_term,
            "a 4 GB card approved {small} where the time term alone would allow {time_term}"
        );
    }

    /// Short series make the time term generous, and the occupancy knee is the
    /// floor that keeps it from ever being the thing that starves the card.
    #[test]
    fn the_time_term_never_starves_a_short_series() {
        assert_eq!(
            candidates_for_target_launch(usize::MAX),
            OCCUPANCY_KNEE,
            "even an absurd bar count must not push the launch under the knee"
        );
        assert!(candidates_for_target_launch(4_096) > OCCUPANCY_KNEE);
        // The floor really does OVERRIDE the target on a long series, and that
        // is a deliberate trade rather than a bound: at EURUSD M1 dimensions the
        // honest time term is ~3 200 scenarios and the floor lifts it to 16 384,
        // an estimated 102 s launch against a 20 s target. It is pinned here so
        // the warning `candidates_for_target_launch` emits stays true, and so
        // that nobody reads the target as a guarantee.
        const M1_BARS: u64 = 5_270_000;
        assert!(
            SCENARIO_BARS_PER_SECOND * TARGET_LAUNCH_SECONDS / M1_BARS < OCCUPANCY_KNEE,
            "the long-series case is what the override warning exists for"
        );
    }

    /// A workspace may be reused DOWNWARD and grown UPWARD, and neither costs a
    /// dataset re-upload.
    ///
    /// The old `event_capacity` subtracted `population * bars * 5 + population *
    /// 3944` from the budget, so asking for MORE candidates yielded a SMALLER
    /// required capacity — and `capacity >= capacity` therefore passed exactly
    /// when it should have failed. A session built for 256 candidates was
    /// reused for 25 600.
    ///
    /// What replaced it was `workspace_scenarios >= n_scenarios` in the HOST
    /// reuse predicate, and that over-corrected. `workspace_scenarios` is
    /// written only at session creation, so the first LARGER launch tore the
    /// session down and re-uploaded the entire dataset — up to 10.8 GB of
    /// traffic — in order to grow a 594 B/scenario workspace that the device
    /// grows by itself. The guard that matters is the device's
    /// (`workspace_scenarios < scenario_count`), and it is exact.
    #[test]
    fn a_workspace_is_reusable_downward_and_grown_upward() {
        // The device predicate, verbatim, and the host record that follows it.
        let device_reallocates = |workspace: usize, requested: usize| workspace < requested;
        let host_record_after = |workspace: usize, requested: usize| workspace.max(requested);

        // The measured case: a session built for 25 600 then asked for 12 800
        // and 6 400 by the recursive split. None re-allocates, and the record
        // does not follow them down.
        for requested in [25_600usize, 12_800, 6_400, 3_300, 1] {
            assert!(
                !device_reallocates(25_600, requested),
                "asking for {requested} against a 25 600 workspace must not re-allocate \
                 — that is the 15.9 GB free-and-realloc this fix exists to remove"
            );
            assert_eq!(host_record_after(25_600, requested), 25_600);
        }

        // Growth re-allocates the WORKSPACE, and only the workspace; the host
        // record then matches what the device holds.
        assert!(device_reallocates(256, 25_600));
        assert_eq!(host_record_after(256, 25_600), 25_600);

        // 1 000 -> 5 000 -> 1 000 -> 5 000 is ONE device re-allocation and no
        // dataset re-upload at all. Under the old host predicate the first
        // 5 000 rebuilt the whole session.
        let mut record = 1_000usize;
        let mut reallocations = 0;
        for requested in [1_000usize, 5_000, 1_000, 5_000] {
            if device_reallocates(record, requested) {
                reallocations += 1;
            }
            record = host_record_after(record, requested);
        }
        assert_eq!(reallocations, 1, "the workspace grows once and is then reused");
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
        // No `bars * 5` term: the per-scenario signal and confidence columns
        // that made the blind batch depend on the series length are gone, so
        // this is now a flat 593 768 B per candidate whatever the dataset.
        let per_candidate =
            neoethos_gpu_cuda::MAX_TRADES_PER_CANDIDATE * 72 + 2 * MONTHS as u64 * 8 + 104 + 56;
        let blind = CONSERVATIVE_BATCH as u64 * per_candidate;
        assert!(
            blind < 2 * 1024 * 1024 * 1024,
            "a blind batch wants {} MiB, which is not safe on a small card",
            blind / (1024 * 1024)
        );
        assert!(CONSERVATIVE_BATCH >= 256, "and it still has to keep the card busy");
    }

    /// Peak memory must follow the hardware, never the request.
    #[test]
    fn a_smaller_card_approves_a_smaller_batch() {
        const BARS: usize = 87_715;
        const FEATURES: usize = 257;
        let big = fits_or_panic(24 * 1024 * 1024 * 1024, BARS, FEATURES);
        let small = fits_or_panic(8 * 1024 * 1024 * 1024, BARS, FEATURES);
        assert!(small < big, "8 GB approved {small}, 24 GB approved {big}");
    }

    /// "No room" and "cannot read the card" are DIFFERENT ANSWERS.
    ///
    /// They were the same `None`, and the caller's `None` arm sized the launch
    /// up to `last_known_fit()` — so the branch taken when the card is provably
    /// too small was the branch that replayed the biggest batch that had ever
    /// worked, on any dataset. That is the `unwrap_or(usize::MAX)` incident
    /// rebuilt with more steps.
    #[test]
    fn no_room_is_not_the_same_answer_as_cannot_read() {
        const BARS: usize = 87_715;
        const FEATURES: usize = 257;
        assert!(matches!(
            candidates_for_free_memory(64 * 1024 * 1024, BARS, FEATURES, MONTHS),
            Sizing::NoRoom { .. }
        ));

        // The worked shape: a 12 GB card asked for EURUSD M1 at 257 features.
        // The indicator matrix alone is 5.4 GB and it is charged twice for the
        // bar-major transpose, so the DATASET does not fit — and no work list,
        // however small, makes a dataset smaller.
        assert!(matches!(
            candidates_for_free_memory(12 * 1024 * 1024 * 1024, 5_270_000, 257, MONTHS),
            Sizing::NoRoom { .. }
        ));
    }

    /// `month_capacity` is an operator knob with no upper bound, and the device
    /// allocates `scenario_count * month_capacity` doubles TWICE. Folding the
    /// default 240 into a literal 3 944 made peak device memory a function of a
    /// user parameter — the never-OOM invariant inverted, inside the one
    /// function that exists to enforce it.
    #[test]
    fn the_month_capacity_knob_is_charged_rather_than_assumed() {
        const BARS: usize = 87_715;
        const FEATURES: usize = 257;
        const FREE: u64 = 24 * 1024 * 1024 * 1024;
        let default = fits_or_panic(FREE, BARS, FEATURES);
        let doubled = match candidates_for_free_memory(FREE, BARS, FEATURES, 2 * MONTHS) {
            Sizing::Fits(fits) => fits,
            other => panic!("expected a fit, got {other:?}"),
        };
        assert!(
            doubled < default,
            "doubling month_capacity adds 3 840 B per scenario, so it must LOWER the \
             approved count: {doubled} vs {default}"
        );
        // The default still costs what the old literal did — 2 * 240 * 8 + 104
        // = 3 944 — plus the 56 B of scenario-descriptor arrays that were never
        // charged at all.
        assert_eq!(2 * MONTHS * 8 + 104, 3_944);
    }

    /// A fit learned on one dataset must never size a launch on another.
    ///
    /// `last_known_fit` and `learned_batch_limit` were process-wide atomics
    /// keyed to nothing, so an M5 fit of 26 777 scenarios was replayed as the
    /// batch size for an M1 dataset whose indicator matrix alone is ten times
    /// larger — and one dataset's capacity failure permanently capped every
    /// other dataset's launches for the life of the process.
    #[test]
    fn a_learned_limit_belongs_to_one_shape() {
        let m5 = LimitKey { device: 0, bars: 843_456, feature_count: 64, month_capacity: 240 };
        let m1 = LimitKey { device: 0, bars: 5_270_000, feature_count: 257, month_capacity: 240 };
        learn_batch_success(m5, 26_777);
        assert_eq!(learned_batch_limit(m5), 26_777);
        assert_eq!(
            learned_batch_limit(m1),
            usize::MAX,
            "an M1 dataset has no learned limit just because an M5 one does"
        );
        // Nor does a failure on one shape cap the other.
        learn_batch_failure(m1, 512);
        assert_eq!(learned_batch_limit(m1), 512);
        assert_eq!(learned_batch_limit(m5), 26_777);
    }

    /// A DATASET allocation failure must not be retried by halving the work.
    ///
    /// `upload_dataset` returns the same `STATUS_ALLOCATION_FAILED` as a
    /// workspace exhaustion, so the retry could not tell them apart and split a
    /// 17 748-scenario screen down to one scenario — ~35 000 launches, each
    /// re-attempting the identical multi-gigabyte `cudaMalloc`, on a context the
    /// first failure may already have left unusable.
    #[test]
    fn a_dataset_allocation_failure_is_never_split() {
        let workspace: Result<()> =
            Err(native_error(neoethos_gpu_cuda::STATUS_ALLOCATION_FAILED))
                .map_err(anyhow::Error::new)
                .context("prototype B evaluate");
        assert!(
            is_capacity_exhaustion(&workspace.expect_err("constructed as an error")),
            "a workspace exhaustion IS worth retrying smaller"
        );

        let dataset: Result<()> = Err(native_error(neoethos_gpu_cuda::STATUS_ALLOCATION_FAILED))
            .map_err(anyhow::Error::new)
            .context(NotAWorkListSizeProblem("the dataset upload"))
            .context("prototype B dataset upload");
        let error = dataset.expect_err("constructed as an error");
        assert!(
            !is_capacity_exhaustion(&error),
            "a dataset upload failure is the same size at every leaf: {error:#}"
        );
        // And the reason still reaches the log.
        let rendered = format!("{error:#}");
        assert!(rendered.contains("prototype B dataset upload"), "{rendered}");
        assert!(rendered.contains("does not depend on the work list size"), "{rendered}");
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

    fn key_of(settings: &BacktestSettings) -> u64 {
        use std::hash::Hasher;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        settings_key(settings, &mut hasher);
        hasher.finish()
    }

    /// `BacktestSettings::default()` leaves `pip_value`, `spread_pips`,
    /// `commission_per_trade` and `pip_value_per_lot` as NaN sentinels — a
    /// deliberate trap so a caller that never binds a real symbol is caught by
    /// the NaN-fitness guard. That makes the default useless as a hashing
    /// baseline: `NaN + 0.3` is NaN with the same bit pattern, so a test built
    /// on it would report "the key ignores spread" no matter what the key did.
    /// Fill the cost fields with real numbers first.
    fn priced_settings() -> BacktestSettings {
        let mut settings = BacktestSettings::default();
        settings.pip_value = 0.0001;
        settings.spread_pips = 1.2;
        settings.commission_per_trade = 3.5;
        settings.pip_value_per_lot = 10.0;
        settings
    }

    /// The key must separate exactly the settings the device can see.
    ///
    /// Both directions are failures, and they fail differently. A field that
    /// reaches `NeoPopulationSettings` but not the key is a WRONG CACHE HIT:
    /// the resident session keeps last chunk's spread or commission and the run
    /// continues on the card, quietly pricing trades with the wrong numbers. A
    /// field that reaches the key but not the device is pure waste: it tears
    /// down a 261 MiB resident dataset and re-uploads it to change nothing.
    ///
    /// `sl_pips` and `tp_pips` were the second kind. They are per-gene — they
    /// travel in `stop_pips` / `target_pips`, re-uploaded every call — and the
    /// chunk loop takes them from the first gene of the chunk, so every chunk
    /// over the same bars rebuilt the dataset over 16 dead bytes.
    #[test]
    fn the_key_separates_settings_that_reach_the_device() {
        let base = priced_settings();
        let baseline = key_of(&base);

        // Per-gene, invisible to `NeoPopulationSettings`: must NOT rebuild.
        let mut per_gene = base.clone();
        per_gene.sl_pips = base.sl_pips + 17.5;
        per_gene.tp_pips = base.tp_pips + 42.25;
        assert_eq!(
            key_of(&per_gene),
            baseline,
            "sl_pips/tp_pips are per-gene and have no field in NeoPopulationSettings — \
             changing them must not destroy the resident dataset"
        );

        // Everything that does reach the device: must rebuild.
        let mut spread = base.clone();
        spread.spread_pips = base.spread_pips + 0.3;
        assert_ne!(key_of(&spread), baseline, "spread reaches the kernel");

        let mut commission = base.clone();
        commission.commission_per_trade = base.commission_per_trade + 1.0;
        assert_ne!(key_of(&commission), baseline, "commission reaches the kernel");

        let mut hold = base.clone();
        hold.max_hold_bars = base.max_hold_bars + 1;
        assert_ne!(key_of(&hold), baseline, "max_hold_bars reaches the kernel");

        // `trailing_min_lock_pips` HAS a field in `NeoPopulationSettings`
        // (`population_settings_for_dataset` writes it), so it belongs in the
        // key. Note for whoever chases a trailing-stop parity gap: the value
        // the device receives is currently NOT this one. `SnapshotSettingsDto`
        // has no field for it, and `to_settings()` rebuilds from
        // `BacktestSettings::default()`, so the round trip through the DTO
        // silently substitutes the default. That is a separate defect in the
        // DTO, not in this key — the key is what must change if the setting
        // changes, and fixing the DTO must not require touching this line.
        let mut trailing = base.clone();
        trailing.trailing_min_lock_pips = base.trailing_min_lock_pips + 1.0;
        assert_ne!(
            key_of(&trailing),
            baseline,
            "trailing_min_lock_pips has a field in NeoPopulationSettings"
        );

        // The session profile is a distinct session from a flat spread even
        // when its buckets all equal that flat value.
        let mut profile = base.clone();
        profile.session_spread_profile = Some(crate::eval::SessionSpreadProfile {
            asian_pips: base.spread_pips,
            overlap_pips: base.spread_pips,
            late_ny_pips: base.spread_pips,
        });
        assert_ne!(
            key_of(&profile),
            baseline,
            "a profile whose buckets equal the flat spread is still a different session"
        );
    }

    /// `adaptive_base_pips` is uploaded WITH the dataset, so it is the one
    /// settings field whose change genuinely is a different device dataset.
    ///
    /// It is also the field that made the old key cost 17 MB: `Debug` rendered
    /// all 843 456 values into a `String` on every launch just to compare them.
    /// Sampled hashing has to keep separating them.
    #[test]
    fn the_adaptive_base_series_is_part_of_the_dataset_identity() {
        use std::sync::Arc;
        let base = priced_settings();

        let mut none = base.clone();
        none.adaptive_base_pips = None;

        let mut flat = base.clone();
        flat.adaptive_base_pips = Some(Arc::from(vec![10.0f64; 4_096].as_slice()));

        let mut shifted = base.clone();
        let mut values = vec![10.0f64; 4_096];
        // Move a sampled position: with 4 096 values and 256 samples the stride
        // is 16, so index 2 048 is one of them.
        values[2_048] = 11.0;
        shifted.adaptive_base_pips = Some(Arc::from(values.as_slice()));

        let mut longer = base.clone();
        longer.adaptive_base_pips = Some(Arc::from(vec![10.0f64; 8_192].as_slice()));

        assert_ne!(key_of(&none), key_of(&flat), "absent is not the same as flat");
        assert_ne!(key_of(&flat), key_of(&shifted), "a changed stop distance is a different dataset");
        assert_ne!(key_of(&flat), key_of(&longer), "a different length is a different dataset");
        assert_eq!(
            key_of(&flat),
            key_of(&flat.clone()),
            "and identical series must still hit the resident session"
        );
    }
}

/// Cut the WORK LIST and evaluate each part.
///
/// This used to slice the CSR gene arrays and rebase the tail's offsets so its
/// first gene started at zero — a manoeuvre that, done wrong, evaluates the
/// wrong terms silently rather than erroring. None of that is needed any more:
/// every descriptor carries its own gene index, so the genes stay whole and
/// resident and a slice of the descriptor array is still a correct work list
/// whatever it contains. The class of bug is gone rather than guarded.
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
    scenarios: &[ScenarioDescriptor],
    head_len: usize,
) -> Result<Vec<[f64; 11]>> {
    // Everything launched below this point is a split leaf. Without it the
    // telemetry sees only the outer entry: a `calls=7` line covered an unknown
    // number of real launches, and "unknown" is what let the recursive split be
    // mistaken for a single submission twice over. The guard nests, so a leaf
    // three halvings deep still reports as a leaf.
    let _split = crate::eval_telemetry::SplitScope::enter();
    let n_scenarios = scenarios.len();
    // Cut at the size that fits rather than in half.
    //
    // Halving overshoots whenever the work list is a little over the limit: a
    // measured run had 25 600 items against a card holding 12 334, halved to
    // 12 800, halved again to 6 400 — four launches each using half the card,
    // where three full ones would do. The caller knows the limit when the split
    // is pre-emptive; after a capacity failure it does not, and passes half.
    let cut = head_len.clamp(1, n_scenarios.saturating_sub(1));

    // The gene arrays are passed through UNCHANGED to both halves. That is the
    // point: a descriptor's `base_candidate_id` indexes the full population, so
    // slicing the genes would invalidate every index in the work list. Uploading
    // the whole gene array twice costs a few megabytes; slicing it wrongly costs
    // a run that reports numbers from the wrong strategies.
    let mut head = try_evaluate_scenarios_b(
        close, high, low, indicators, gene_offsets, gene_indices, gene_weights, long_thr,
        short_thr, month_idx, day_idx, timestamps, sl_pips, tp_pips, stop_vol_mult, smc_data,
        gene_smc_flags, gate_threshold, smc_weights, settings, device_override,
        &scenarios[..cut],
    )?;
    let tail = try_evaluate_scenarios_b(
        close, high, low, indicators, gene_offsets, gene_indices, gene_weights, long_thr,
        short_thr, month_idx, day_idx, timestamps, sl_pips, tp_pips, stop_vol_mult, smc_data,
        gene_smc_flags, gate_threshold, smc_weights, settings, device_override,
        &scenarios[cut..],
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
    scenarios: &[ScenarioDescriptor],
) -> Result<Vec<[f64; 11]>> {
    // Host prep starts here, not at the first upload.
    //
    // Everything between this line and the submission below is the card
    // waiting: hashing the dataset key, sizing the session, staging genes and
    // scenarios, and — when the session is not reusable — freeing and
    // re-uploading the entire workspace. It was never measured, so "the launch
    // took 23 s" and "the kernel took 0.30 s" could both be reported without
    // anyone being able to say where the other 22.7 s went.
    let host_prep_started = std::time::Instant::now();
    let n_genes = long_thr.len();
    let n_scenarios = scenarios.len();
    let bars = close.len();
    if n_genes == 0 || bars == 0 || n_scenarios == 0 {
        return Ok(vec![ZERO_METRICS; n_scenarios]);
    }
    // Same optional-contract handling as the CubeCL lane: an empty
    // `stop_vol_mult` means "no adaptive stops", and every downstream slice
    // would otherwise index out of range.
    let stop_vol_fallback = crate::eval::normalized_stop_vol_mult(stop_vol_mult, n_genes);
    let stop_vol_mult = stop_vol_fallback.as_deref().unwrap_or(stop_vol_mult);

    // And the same contract for timestamps: the CPU reference treats an empty
    // slice as "no timestamps" (session spread profile off, gap detection off,
    // entry/exit stamps zero — `use_timestamps` in `fast_evaluate_strategy_core`),
    // and the CPCV gathered-fold lane passes exactly that. The device dataset has
    // no empty-slice notion — `PopulationDatasetView::validate` demands one stamp
    // per bar — so zeros go up instead. That is bit-identical, not approximate:
    // the kernel's `timestamp_ms <= 0` guard resolves spread to the scalar, the
    // gap kernel's `current > previous` never fires on equal stamps, and the
    // trade stamps come back 0 exactly as the CPU's disabled path writes them.
    // Before this shim every production CPCV fold eval was refused at upload and
    // silently recomputed on the CPU — the card never saw the CPCV gate at all.
    let timestamps_fallback: Option<Vec<i64>> =
        if timestamps.is_empty() { Some(vec![0; bars]) } else { None };
    let timestamps = timestamps_fallback.as_deref().unwrap_or(timestamps);

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

    // Hold the slot for the whole evaluation. That serializes device access the
    // way the CubeCL launch lock does, which is deliberate: the quality screen
    // calls this from a rayon `par_iter`, and one session per worker thread
    // would multiply VRAM by the thread count.
    let mut slot = resident_slot()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    // Three terms, and each one is load-bearing.
    //
    // There was a fourth, `capacity >= capacity`, comparing the event budget
    // the session was created with against the one this call computed. It
    // guarded a buffer that is never allocated, and because that budget
    // subtracted `population * bars * 5` it moved every time the population
    // did — so a session was torn down and rebuilt over an imaginary number.
    // It is gone.
    //
    // What is NOT gone is the size term. That is the one this predicate exists
    // for: the device workspace is sized at allocation and every kernel indexes
    // it by the current SCENARIO COUNT, so handing a reused session a LONGER
    // work list writes past the end of the outcome array — into `monthly_pnls`,
    // which is what sharpe and consistency are computed from, with `sanitize()`
    // turning the consequence into a plausible 0.0. The native side refuses that
    // too (`workspace_scenarios < scenario_count` re-allocates), so this is the
    // outer of two independent guards, not the only one.
    // THE SIZE TERM IS NOT IN THIS PREDICATE, and that is deliberate.
    //
    // It was: `workspace_scenarios >= n_scenarios`. `workspace_scenarios` is
    // only ever written at session CREATION, so the first launch LARGER than
    // the one the session was built for tore the whole session down — a fresh
    // `PopulationSession::create`, a full `upload_dataset` (up to 10.8 GB of
    // traffic at M1/F=257 counting the bar-major transpose) and a full rebuild
    // of the host staging copy — to grow a workspace that costs 594 B per
    // scenario and that the DEVICE grows for free: `upload`-side, cu:2165 reads
    // `session->workspace_scenarios < scenario_count` and re-allocates the
    // workspace alone, leaving the dataset resident.
    //
    // So the host guard was not a second safety net at all. It could not stop
    // an overrun the device does not already stop — the device decides the
    // allocation — and what it actually did was convert a free workspace growth
    // into the "go back to the card and it is empty" dead time. The GA lane and
    // the validation tail share one resident slot and their launch sizes
    // alternate, so this fired constantly.
    //
    // `workspace_scenarios` stays, as a RECORD of what the device allocated,
    // raised after each successful evaluate (below). It is what the telemetry
    // and the tests read; it is no longer a rebuild trigger.
    let reusable = slot
        .as_ref()
        .is_some_and(|r| r.0.key == key && r.0.device == device);

    // Counted, not just taken: a rebuild frees and re-allocates the whole
    // workspace, and every `cudaFree` is an implicit device-wide sync. A run
    // that rebuilds on every launch and one that never rebuilds produced the
    // same output, and the difference is the largest single cost in the lane.
    let session_rebuilt = !reusable;

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

        let mut session = PopulationSession::create(device, VESTIGIAL_SESSION_MAX_EVENTS)
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
            .context(NotAWorkListSizeProblem("the dataset upload"))
        .context("prototype B dataset upload")?;

        *slot = Some(SendResident(ResidentSession {
            session,
            key,
            device,
            workspace_scenarios: n_scenarios,
            dataset,
            smc_rows,
            native_settings,
        }));
    }

    let resident = &mut slot
        .as_mut()
        .expect("resident session was just installed")
        .0;
    // The device settings come from the RESIDENT copy, built when the session
    // was, and this is only sound because the key covers them.
    //
    // The question the key rewrite raises: `dataset_key` no longer hashes
    // `sl_pips` and `tp_pips`, so could a settings change now slip past the
    // predicate and leave this copy stale — the run charging last chunk's
    // spread while reporting this chunk's? There were two ways to close that:
    // keep every scalar in the key, or rebuild `native_settings` on every call
    // and store nothing.
    //
    // Keeping them in the key is the choice, for a reason specific to the two
    // fields that were dropped: they cannot reach here. `native_settings` is
    // built by `population_settings_for_dataset`, which reads
    // `SnapshotSettingsDto` — and that DTO has no `sl_pips` or `tp_pips` field
    // at all. Per-gene stops travel in the `stop_pips` / `target_pips` arrays,
    // re-uploaded with the genes on every call, and `NeoPopulationSettings` has
    // nowhere to put them. So the two excluded fields are provably invisible to
    // this struct, while `settings_key` hashes every other field of
    // `BacktestSettings` by an EXHAUSTIVE destructure that stops compiling when
    // a field is added. Any scalar that can change what the device charges
    // changes the key, rebuilds the session, and rebuilds this copy with it.
    //
    // The alternative — refresh per call — would also be correct, and it would
    // additionally stop a spread change from tearing down a 261 MiB resident
    // dataset. It is not taken here because it would leave the key and the
    // stored settings free to disagree, and a compiler-enforced destructure in
    // one place is a stronger guarantee than a rebuild that a future edit can
    // quietly skip. See `the_key_separates_settings_that_reach_the_device`.
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
        .context(NotAWorkListSizeProblem("the gene upload"))
        .context("prototype B gene upload")?;

    // The work list, as the caller built it.
    //
    // THIS BLOCK USED TO CONSTRUCT THE DESCRIPTORS ITSELF — one per gene, with
    // every field except `scenario_id` a literal zero, and a comment explaining
    // that costs were carried by the settings "so every per-scenario knob stays
    // zero". The knobs were zero because nothing read them; the engine did not
    // "reject anything else", it ignored everything else. That is what made a
    // screen wanting three treatments of the same genes need three launches.
    //
    // The caller now decides. `try_evaluate_population_b` still builds one
    // full-series `base_scenario` per gene and that path is bit-identical to the
    // old literal — with one field genuinely changed: `spread_ticks` and
    // `commission_micros` are `-1` ("no override") where they were `0`, and `0`
    // now means "charge nothing". Any descriptor built outside
    // `scenario::base_scenario` / `cost_scenario` / `perturb_scenario` is how
    // that becomes a free-trading backtest, which is why there is no third
    // construction site.
    session
        .upload_scenarios(scenarios)
        .map_err(anyhow::Error::new)
        .context("prototype B scenario upload")?;

    // The host/device boundary. Everything above was prep; everything from here
    // to the readback is the only part of the call the card is in.
    let host_prep = host_prep_started.elapsed();
    let device_started = std::time::Instant::now();

    // The engine counts its own submissions and synchronizations (the native
    // side increments them at each kernel launch and each stream sync) and
    // returns them here. The host read them into `_counters` and dropped them
    // on the next line, so the one authoritative answer to "how many kernels
    // did this launch actually submit, and how many times did it stop the
    // stream" was produced by the device, handed to the host, and thrown away
    // — every launch, for the whole life of this lane.
    let (event_id, counters) = session
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
    let device_elapsed = device_started.elapsed();

    // The device grew its workspace if it needed to (cu:2165 re-allocates when
    // `workspace_scenarios < scenario_count`), so the host record follows it
    // upward and never downward: a shorter work list reuses a larger workspace,
    // and the two sides must not disagree about what is resident.
    resident.workspace_scenarios = resident.workspace_scenarios.max(n_scenarios);

    crate::eval_telemetry::record_launch(
        crate::eval_telemetry::current_lane(),
        crate::eval_telemetry::LaunchRecord {
            host_prep,
            device: device_elapsed,
            kernel_submissions: counters.kernel_submissions,
            synchronization_events: counters.synchronization_events,
            session_rebuilt,
            split_leaf: crate::eval_telemetry::inside_split(),
        },
    );

    // How full the trade slots actually are.
    //
    // Every candidate reserves MAX_TRADES_PER_CANDIDATE slots — 589 824 B of
    // the 593 768 B it now costs. Since the signal and confidence columns were
    // deleted this array is 99.3 % of the per-scenario footprint, so it is the
    // ONLY thing left standing between the card and a far larger population,
    // and the reservation is a constant while what a candidate records is not.
    // Nothing measured the difference, so nothing could tell whether the card
    // was full of trades or of empty space.
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
        n_scenarios,
        reserved_slots = neoethos_gpu_cuda::MAX_TRADES_PER_CANDIDATE,
        busiest_candidate = peak as u64,
        mean_trades = (total / (rows.len() as f64).max(1.0)) as u64,
        peak_fill_pct = format!(
            "{:.2}",
            peak * 100.0 / neoethos_gpu_cuda::MAX_TRADES_PER_CANDIDATE as f64
        ),
        "trade slot usage — what was reserved per candidate against what was recorded"
    );

    // Where this launch's time went.
    //
    // This was behind a `std::sync::Once` — "it is a property of the data, not
    // of the call". That was wrong twice over. The population size changes on
    // every recursive split, so the first launch is the LEAST representative one
    // there is; and the number that actually matters is not the event budget at
    // all but the ratio below. A once-per-process line cannot show a ratio that
    // varies per launch, and a lane whose cost is per-call is exactly the lane
    // where a single sample is worthless.
    //
    // So it is per launch now. The GA drives a few hundred of these in a run,
    // which is the same order as the per-call `trade slot usage` line already
    // emitted above, and INFO rather than DEBUG because the app installs its own
    // subscriber and never enables DEBUG — a diagnostic nobody sees is not a
    // diagnostic.
    //
    // Three fields are GONE from this line: `event_capacity`, `emitted_events`
    // and `capacity_used_pct`. They were the most confidently wrong numbers in
    // the run. `emitted_events` is not a count of events — nothing emits events
    // — it is `population * MAX_TRADES_PER_CANDIDATE`, the size of the outcome
    // array, so it was a constant multiple of the population dressed as a
    // measurement. `event_capacity` was a budget for a buffer with no
    // allocation. Their ratio, printed as `capacity_used_pct`, was therefore a
    // pure function of the population and told an investigator precisely
    // nothing while looking exactly like the answer. What a candidate actually
    // records is the `trade slot usage` line above, which reads real metric
    // rows.
    {
        let host_prep_ms = host_prep.as_secs_f64() * 1e3;
        let device_ms = device_elapsed.as_secs_f64() * 1e3;
        let accounted = host_prep_ms + device_ms;
        tracing::info!(
            target: "neoethos_search::eval",
            lane = crate::eval_telemetry::current_lane(),
            n_genes,
            // Both, because they are no longer the same number and the ratio is
            // the measurement: 174 genes carrying 17 574 scenarios is one launch
            // where it used to be seven.
            n_scenarios,
            bars,
            session_rebuilt,
            split_leaf = crate::eval_telemetry::inside_split(),
            kernel_submissions = counters.kernel_submissions,
            sync_events = counters.synchronization_events,
            host_prep_ms = format!("{host_prep_ms:.1}"),
            device_ms = format!("{device_ms:.1}"),
            device_pct = format!(
                "{:.1}",
                if accounted > 0.0 { 100.0 * device_ms / accounted } else { 0.0 }
            ),
            "launch anatomy — host prep against device time for THIS launch"
        );
    }

    if rows.len() != n_scenarios {
        bail!(
            "prototype B returned {} metric rows for {} scenarios",
            rows.len(),
            n_scenarios
        );
    }
    // Rows come back in scenario order, and each one is checked against the
    // descriptor that asked for it.
    //
    // The old check demultiplexed by `candidate_id`, which worked only while
    // there was exactly one scenario per gene: a Monte-Carlo work list has 100
    // scenarios sharing one `candidate_id`, so "seen this id twice" would fire
    // on a correct result and the id would no longer identify a row. Matching
    // `scenario_id` positionally is a STRICTLY STRONGER check — it catches the
    // same permutation without requiring gene ids to be unique — and it is the
    // reason the descriptor carries an id the host chose rather than a position
    // the host has to trust.
    let mut out = vec![ZERO_METRICS; n_scenarios];
    for (index, row) in rows.iter().enumerate() {
        let expected = scenarios[index].scenario_id;
        if row.scenario_id != expected {
            bail!(
                "prototype B returned scenario {} at position {index}, where the work list \
                 asked for scenario {expected} — the results are permuted and every metric \
                 would be attributed to the wrong run",
                row.scenario_id
            );
        }
        out[index] = row.values;
    }
    Ok(out)
}
