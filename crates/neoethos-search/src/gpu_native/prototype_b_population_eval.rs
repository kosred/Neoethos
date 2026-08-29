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

// `NeoPopulationEvent` is deliberately not imported. It was here only so the
// deleted event budget could take its size, and the population lane touches no
// event record anywhere else — the type stays alive for the diagnostic readback
// contract, which is a different lane.
use neoethos_gpu_contracts::device::{GeneDescriptor, ScenarioDescriptor};

use crate::eval::SmcRow;
use crate::gpu_native::prototype_a::PrototypeAGeneUpload;
use crate::gpu_native::prototype_population_oracle::population_settings_for_settings;
use crate::gpu_native::scenario;

use neoethos_gpu_cuda::{CudaPopulationError, PopulationGeneView, PopulationResidencyCountersV1};

/// Metric row shape shared with the CPU and CubeCL lanes.
const ZERO_METRICS: [f64; 11] = [0.0; 11];

struct NativePopulationBatchV1 {
    rows: Vec<[f64; 11]>,
    counters: PopulationResidencyCountersV1,
}

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
    NoRoom {
        dataset_bytes: u64,
        budget_bytes: u64,
    },
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
fn prototype_b_dataset_peak_bytes(bars: usize, feature_count: usize) -> u64 {
    let bars = bars as u64;
    let indicator_elements = bars.saturating_mul(feature_count as u64);
    let indicator_bytes = indicator_elements.saturating_mul(std::mem::size_of::<f64>() as u64);
    let fixed_per_bar = (3 * std::mem::size_of::<f64>()
        + 3 * std::mem::size_of::<i64>()
        + std::mem::size_of::<f64>()
        + std::mem::size_of::<u8>()
        + 11 * std::mem::size_of::<i8>()) as u64;
    bars.saturating_mul(fixed_per_bar)
        .saturating_add(indicator_bytes)
}

fn candidates_for_free_memory(
    free: u64,
    bars: usize,
    feature_count: usize,
    month_capacity: usize,
) -> Sizing {
    // Leave three tenths for context, fragmentation and the allocator's own
    // bookkeeping.
    let budget = (free / 10) * 7;
    // The sealed parent owns exactly one feature-major indicator matrix. The V1
    // native walk consumes it directly, so neither a staging transpose nor a
    // second resident indicator copy belongs in this production budget.
    // Every bars-scaled array `upload_parent_dataset_v1` allocates, named so the next one
    // added has an obvious place to go:
    //   close + high + low                    3 x f64
    //   months + days + timestamps            3 x i64
    //   adaptive_base_pips                    1 x f64   (WAS MISSING)
    //   gap_flags                             1 x u8    (WAS MISSING)
    //   smc_rows                             11 x i8
    //   indicators, once (immutable feature-major parent)
    let dataset = prototype_b_dataset_peak_bytes(bars, feature_count);
    // The card answered and the DATASET alone does not fit. That is not a
    // batching problem: no work list, however small, makes a 10.8 GB f64
    // indicator matrix (21.6 GB at the transpose peak) smaller. Saying so here
    // is what stops the caller from sizing a launch out of a stale number and
    // then discovering it by failing.
    if dataset.saturating_add(64 * 1024 * 1024) >= budget {
        return Sizing::NoRoom {
            dataset_bytes: dataset,
            budget_bytes: budget,
        };
    }
    // The 64 MiB is the reserve for allocator fragmentation, the context, and
    // the GENE arrays — which are not bars-scaled and not scenario-scaled:
    // `population * 59 B + (population + 1) * 4 B + terms * 12 B + 88 B`.
    // The largest gene array any lane uploads is the quality screen's 131 072
    // clones; the fixed per-gene portion is ~7.4 MiB before its CSR terms and
    // stays inside this reserve. A work list is scenarios, charged below.
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
    let outcome =
        std::mem::size_of::<neoethos_gpu_contracts::device::NeoPopulationOutcome>() as u64;
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
        return Sizing::NoRoom {
            dataset_bytes: dataset,
            budget_bytes: budget,
        };
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
fn learned_batch_limits() -> &'static std::sync::Mutex<std::collections::HashMap<LimitKey, usize>> {
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
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f64],
    long_thr: &[f64],
    short_thr: &[f64],
    sl_pips: &[f64],
    tp_pips: &[f64],
    stop_vol_mult: &[f64],
    gene_smc_flags: &[SmcRow],
    gate_threshold: f64,
    smc_weights: &[f64; 11],
) -> Result<Vec<[f64; 11]>> {
    evidence
        .validate_population_layout(evidence.row_count(), evidence.feature_count())
        .map_err(anyhow::Error::new)?;
    let expected = long_thr.len();
    let rows = require_exact_native_population_rows_v1(
        evaluate_population_b_raw_v1(
            evidence,
            gene_offsets,
            gene_indices,
            gene_weights,
            long_thr,
            short_thr,
            sl_pips,
            tp_pips,
            stop_vol_mult,
            gene_smc_flags,
            gate_threshold,
            smc_weights,
        ),
        expected,
    )?;
    if expected > 0 {
        evidence.record_successful_native_population_v1(
            expected,
            rows.rows.len(),
            rows.counters,
        )?;
        evidence
            .record_successful_population(
                crate::engine_identity::PopulationEvalEngine::CudaNativeF64,
                expected,
                rows.rows.len(),
            )
            .map_err(anyhow::Error::new)?;
    }
    Ok(rows.rows)
}

fn require_exact_native_population_rows_v1(
    outcome: Result<NativePopulationBatchV1>,
    expected: usize,
) -> Result<NativePopulationBatchV1> {
    let rows = outcome?;
    if rows.rows.len() != expected {
        bail!(
            "prototype B returned {} metric rows; expected {expected}",
            rows.rows.len()
        );
    }
    Ok(rows)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_population_b_raw_v1(
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f64],
    long_thr: &[f64],
    short_thr: &[f64],
    sl_pips: &[f64],
    tp_pips: &[f64],
    stop_vol_mult: &[f64],
    gene_smc_flags: &[SmcRow],
    gate_threshold: f64,
    smc_weights: &[f64; 11],
) -> Result<NativePopulationBatchV1> {
    let n_genes = long_thr.len();
    let bars = evidence.row_count();
    if n_genes == 0 || bars == 0 {
        return Ok(NativePopulationBatchV1 {
            rows: vec![ZERO_METRICS; n_genes],
            counters: PopulationResidencyCountersV1::default(),
        });
    }
    let scenarios: Vec<ScenarioDescriptor> = (0..n_genes as u64)
        .map(|candidate| scenario::base_scenario(candidate, candidate, bars))
        .collect();
    evaluate_scenarios_b_raw_v1(
        evidence,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        sl_pips,
        tp_pips,
        stop_vol_mult,
        gene_smc_flags,
        gate_threshold,
        smc_weights,
        &scenarios,
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
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f64],
    long_thr: &[f64],
    short_thr: &[f64],
    sl_pips: &[f64],
    tp_pips: &[f64],
    stop_vol_mult: &[f64],
    gene_smc_flags: &[SmcRow],
    gate_threshold: f64,
    smc_weights: &[f64; 11],
    scenarios: &[ScenarioDescriptor],
) -> Result<Vec<[f64; 11]>> {
    evidence
        .validate_population_layout(evidence.row_count(), evidence.feature_count())
        .map_err(anyhow::Error::new)?;
    let expected = scenarios.len();
    let rows = require_exact_native_population_rows_v1(
        evaluate_scenarios_b_raw_v1(
            evidence,
            gene_offsets,
            gene_indices,
            gene_weights,
            long_thr,
            short_thr,
            sl_pips,
            tp_pips,
            stop_vol_mult,
            gene_smc_flags,
            gate_threshold,
            smc_weights,
            scenarios,
        ),
        expected,
    )?;
    if expected > 0 {
        evidence.record_successful_native_population_v1(
            expected,
            rows.rows.len(),
            rows.counters,
        )?;
        evidence
            .record_successful_population(
                crate::engine_identity::PopulationEvalEngine::CudaNativeF64,
                expected,
                rows.rows.len(),
            )
            .map_err(anyhow::Error::new)?;
    }
    Ok(rows.rows)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_scenarios_b_raw_v1(
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f64],
    long_thr: &[f64],
    short_thr: &[f64],
    sl_pips: &[f64],
    tp_pips: &[f64],
    stop_vol_mult: &[f64],
    gene_smc_flags: &[SmcRow],
    gate_threshold: f64,
    smc_weights: &[f64; 11],
    scenarios: &[ScenarioDescriptor],
) -> Result<NativePopulationBatchV1> {
    let n_genes = long_thr.len();
    // What the launch is sized by, from here down.
    let n_scenarios = scenarios.len();
    if n_scenarios == 0 {
        return Ok(NativePopulationBatchV1 {
            rows: Vec::new(),
            counters: PopulationResidencyCountersV1::default(),
        });
    }
    // Refused here, before anything is uploaded, because the device cannot
    // detect either fault: a gene index past the end of the population is an
    // out-of-bounds read of thresholds and CSR offsets that still produces a
    // metric row, and a window past the end of the series is an out-of-bounds
    // read of prices. The native side checks again — two independent guards,
    // like the workspace-population pair — but this one can name the index.
    if let Err(detail) = scenario::validate_scenarios(scenarios, n_genes, evidence.row_count()) {
        bail!("prototype B scenario list: {detail}");
    }

    let selected_ordinal = evidence
        .require_exact_cuda_device_ordinal_v1()?
        .selected_ordinal();
    let device = usize::try_from(selected_ordinal).map_err(|_| {
        anyhow::anyhow!("sealed CUDA ordinal {selected_ordinal} does not fit this process")
    })?;

    // Everything the launch size is a fact about, in one key.
    let limit_key = LimitKey {
        device,
        bars: evidence.row_count(),
        feature_count: evidence.feature_count(),
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
        Sizing::NoRoom {
            dataset_bytes,
            budget_bytes,
        } => {
            bail!(
                "prototype B: this device has no room for the dataset — it needs \
                 {dataset_bytes} B (close/high/low, months/days/timestamps, the adaptive stop \
                 base, gap flags, SMC rows and one immutable feature-major indicator matrix) \
                 against a {budget_bytes} B budget of free VRAM. {n_scenarios} \
                 scenarios over {} bars x {} features. Splitting the work list cannot help: \
                 the dataset is the same size whatever the launch asks for.",
                limit_key.bars,
                limit_key.feature_count
            );
        }
        Sizing::Unreadable => bail!(
            "prototype B could not read free memory on sealed CUDA ordinal {selected_ordinal}; \
             a runtime/device probe fault cannot authorize a guessed batch or CPU substitution"
        ),
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
            evidence,
            gene_offsets,
            gene_indices,
            gene_weights,
            long_thr,
            short_thr,
            sl_pips,
            tp_pips,
            stop_vol_mult,
            gene_smc_flags,
            gate_threshold,
            smc_weights,
            scenarios,
            learned,
        );
    }

    let attempt = evaluate_population_b_batch(
        evidence,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        sl_pips,
        tp_pips,
        stop_vol_mult,
        gene_smc_flags,
        gate_threshold,
        smc_weights,
        device,
        scenarios,
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
        evidence,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        sl_pips,
        tp_pips,
        stop_vol_mult,
        gene_smc_flags,
        gate_threshold,
        smc_weights,
        scenarios,
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

    #[test]
    fn sealed_parent_sizes_one_f64_indicator_matrix_without_transpose_staging() {
        const BARS: usize = 37;
        const FEATURES: usize = 13;
        let fixed_per_bar = 3 * std::mem::size_of::<f64>()
            + 3 * std::mem::size_of::<i64>()
            + std::mem::size_of::<f64>()
            + std::mem::size_of::<u8>()
            + 11 * std::mem::size_of::<i8>();
        let expected = BARS * fixed_per_bar + BARS * FEATURES * std::mem::size_of::<f64>();
        assert_eq!(
            prototype_b_dataset_peak_bytes(BARS, FEATURES),
            expected as u64
        );
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
    /// session down and re-uploaded the entire dataset — more than 10 GB of H2D
    /// traffic plus a same-sized transient transpose allocation — in order to
    /// grow a 594 B/scenario workspace that the device grows by itself. The
    /// guard that matters is the device's
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
        assert_eq!(
            reallocations, 1,
            "the workspace grows once and is then reused"
        );
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
        // The f64 indicator matrix alone is 10.8 GB and it is charged twice for
        // the bar-major transpose, so the DATASET does not fit — and no work
        // list, however small, makes a dataset smaller.
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
        let m5 = LimitKey {
            device: 0,
            bars: 843_456,
            feature_count: 64,
            month_capacity: 240,
        };
        let m1 = LimitKey {
            device: 0,
            bars: 5_270_000,
            feature_count: 257,
            month_capacity: 240,
        };
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
        let workspace: Result<()> = Err(native_error(neoethos_gpu_cuda::STATUS_ALLOCATION_FAILED))
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
        assert!(
            rendered.contains("prototype B dataset upload"),
            "{rendered}"
        );
        assert!(
            rendered.contains("does not depend on the work list size"),
            "{rendered}"
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

    #[test]
    fn resident_reuse_is_owned_by_the_run_scoped_sealed_native_boundary() {
        let source = include_str!("prototype_b_population_eval.rs");
        let production = &source[source
            .find("fn evaluate_population_b_batch(")
            .expect("native population adapter boundary")..];
        assert!(production.contains("bind_exact_native_population_view_v1"));
        assert!(!production.contains("fn resident_slot()"));
        assert!(!production.contains("static RESIDENT"));
        assert!(!production.contains("sample_hash"));
        assert!(!production.contains("dataset_key"));
    }

    #[test]
    fn prototype_b_f64_adapter_contract_has_no_cubecl_diversion() {
        let source = include_str!("../cubecl_eval.rs").to_ascii_lowercase();
        for forbidden in [
            "prototype_b",
            "prototype b",
            "prototype-b",
            "gpu-b-adapter",
            "try_evaluate_population_b",
        ] {
            assert!(
                !source.contains(forbidden),
                "CubeCL must remain a separate engine; found stale diversion token {forbidden}"
            );
        }
    }

    #[test]
    fn prototype_b_f64_adapter_contract_real_parity_is_fail_loud() {
        let source = include_str!("../eval.rs");
        let start = source
            .find("fn gpu_matches_cpu_with_a_trailing_stop()")
            .expect("real Prototype B parity test must exist");
        let rest = &source[start..];
        let end = rest
            .find("fn uniform_buckets_are_a_scalar_by_another_name()")
            .expect("next parity test must delimit the real trailing-stop test");
        let parity_test = &rest[..end];

        assert!(
            !parity_test.contains("skipping trailing parity"),
            "a device error must fail the paid real-device parity test, never report a skip"
        );
        assert!(
            parity_test.contains("Prototype B real-device parity failed"),
            "the parity test must surface a device failure with an explicit fail-loud diagnostic"
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
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f64],
    long_thr: &[f64],
    short_thr: &[f64],
    sl_pips: &[f64],
    tp_pips: &[f64],
    stop_vol_mult: &[f64],
    gene_smc_flags: &[SmcRow],
    gate_threshold: f64,
    smc_weights: &[f64; 11],
    scenarios: &[ScenarioDescriptor],
    head_len: usize,
) -> Result<NativePopulationBatchV1> {
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
    let mut head = evaluate_scenarios_b_raw_v1(
        evidence,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        sl_pips,
        tp_pips,
        stop_vol_mult,
        gene_smc_flags,
        gate_threshold,
        smc_weights,
        &scenarios[..cut],
    )?;
    let tail = evaluate_scenarios_b_raw_v1(
        evidence,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        sl_pips,
        tp_pips,
        stop_vol_mult,
        gene_smc_flags,
        gate_threshold,
        smc_weights,
        &scenarios[cut..],
    )?;
    head.rows.extend(tail.rows);
    head.counters = tail.counters;
    Ok(head)
}

/// Evaluate a population on Prototype B.
///
/// This is the canonical f64 native-CUDA adapter. CubeCL remains a separate
/// engine and never diverts into this path.
#[allow(clippy::too_many_arguments)]
fn evaluate_population_b_batch(
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f64],
    long_thr: &[f64],
    short_thr: &[f64],
    sl_pips: &[f64],
    tp_pips: &[f64],
    stop_vol_mult: &[f64],
    gene_smc_flags: &[SmcRow],
    gate_threshold: f64,
    smc_weights: &[f64; 11],
    device: usize,
    scenarios: &[ScenarioDescriptor],
) -> Result<NativePopulationBatchV1> {
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
    let bars = evidence.row_count();
    if n_genes == 0 || bars == 0 || n_scenarios == 0 {
        return Ok(NativePopulationBatchV1 {
            rows: vec![ZERO_METRICS; n_scenarios],
            counters: PopulationResidencyCountersV1::default(),
        });
    }
    // Same optional-contract handling as the CubeCL lane: an empty
    // `stop_vol_mult` means "no adaptive stops", and every downstream slice
    // would otherwise index out of range.
    let stop_vol_fallback = crate::eval::normalized_stop_vol_mult(stop_vol_mult, n_genes);
    let stop_vol_mult = stop_vol_fallback.as_deref().unwrap_or(stop_vol_mult);

    let native_device = i32::try_from(device)
        .map_err(|_| anyhow::anyhow!("sealed CUDA ordinal {device} does not fit the native ABI"))?;
    let native_settings = population_settings_for_settings(evidence.settings())
        .map_err(anyhow::Error::new)
        .context("prototype B settings")?;

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
    let descriptors = genes
        .candidate_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(index, candidate_id)| {
            let start = genes.offsets[index] as u32;
            let end = genes.offsets[index + 1] as u32;
            GeneDescriptor {
                candidate_id,
                term_offset: start,
                term_count: end.saturating_sub(start),
                long_threshold: genes.long_thresholds[index],
                short_threshold: genes.short_thresholds[index],
                stop_ticks: 0,
                target_ticks: 0,
                stop_vol_multiplier: genes.stop_vol_multipliers[index],
                flags: 0,
                reserved: 0,
            }
        })
        .collect::<Vec<_>>();
    let smc_flags = genes
        .smc_flags
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();

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
    let ((rows, counters, host_prep, device_elapsed, adapter_counters), residency_counters) =
        evidence.bind_exact_native_population_view_v1(native_device, |session| {
            session
                .upload_genes(PopulationGeneView {
                    descriptors: &descriptors,
                    offsets: &genes.offsets,
                    indices: &genes.indices,
                    weights: &genes.weights,
                    stop_pips: &genes.stop_pips,
                    target_pips: &genes.target_pips,
                    stop_vol_multipliers: &genes.stop_vol_multipliers,
                    smc_flags: &smc_flags,
                    smc_weights: &genes.smc_weights,
                    gate_threshold: genes.gate_threshold,
                    smc_gate_disabled: crate::genetic::smc_gate_disabled(),
                })
                .map_err(anyhow::Error::new)
                .context(NotAWorkListSizeProblem("the gene upload"))
                .context("prototype B gene upload")?;
            session
                .upload_scenarios(scenarios)
                .map_err(anyhow::Error::new)
                .context("prototype B scenario upload")?;

            // The parent upload/view bind and all changing input staging are
            // included in host prep. The device interval still includes the
            // full population-metric D2H readback; P1-E must eliminate that
            // intermediate transfer before this can be final-only evidence.
            let host_prep = host_prep_started.elapsed();
            let device_started = std::time::Instant::now();
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
            let adapter_counters = session
                .read_residency_counters_v1()
                .map_err(anyhow::Error::new)
                .context("prototype B residency counter readback")?;
            Ok((
                rows,
                counters,
                host_prep,
                device_started.elapsed(),
                adapter_counters,
            ))
        })?;
    if adapter_counters != residency_counters {
        bail!("native population residency counters changed without an intervening operation");
    }
    let session_rebuilt = residency_counters.parent_upload_count() == 1
        && residency_counters.view_binding_count() == 1;

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
    let trade_counts = rows
        .iter()
        .map(|row| row.values[8])
        .filter(|count| count.is_finite());
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
    Ok(NativePopulationBatchV1 {
        rows: out,
        counters: residency_counters,
    })
}
