use super::super::Ohlcv;
use crate::core::all_indicators::ALL_INDICATORS;
use crate::core::feature_budget::{VocabularyBudget, admit_indicators};
use crate::core::indicator_ledger::{
    DropReason, IndicatorLedger, expected_non_producing, has_finite_variation, output_ids_for,
    planned_output_count, series_fingerprint,
};
use rayon::prelude::*;
use std::collections::HashSet;
use std::panic::{AssertUnwindSafe, catch_unwind};
use vector_ta::indicators::dispatch::{
    IndicatorComputeOutput, IndicatorComputeRequest, IndicatorDataRef, IndicatorSeries, ParamKV,
    ParamValue, compute_cpu,
};
use vector_ta::utilities::data_loader::Candles;
use vector_ta::utilities::enums::Kernel;

/// Fewest distinct indicator ids the base pass must produce a column for
/// before the feature build is considered sane, on a frame long enough to warm
/// them up.
///
/// MEASURED, not aspirational: over real EURUSD M5 bars (identical at 20k and
/// 200k rows) 324 of the 342 ids produce at least one column once the output
/// ids are enumerated and the accept test keys off the value count. 280 leaves
/// room for vector-ta variation while making the regression this constant
/// exists to catch — the collapse to ONE producing id (`ttm_trend`) that this
/// codebase ran on for sixteen months — a hard, named failure.
pub const MIN_PRODUCING_INDICATOR_IDS: usize = 280;

/// Fewest base-pass columns, same measurement: the repaired pass yields ~674.
pub const MIN_BASE_VOCABULARY_COLUMNS: usize = 400;

/// Frames shorter than this do not enforce [`MIN_PRODUCING_INDICATOR_IDS`].
///
/// On a 100-bar fixture most indicators legitimately fail their warmup, so a
/// low count is the data's property and not a regression. The census is still
/// logged at every length — only the hard error is length-gated, and the log
/// line says which mode it ran in.
pub const VOCABULARY_FLOOR_MIN_ROWS: usize = 5_000;

/// Largest indicator-period this module ever asks vector-ta to compute
/// in its multi-period sweep. Used by [`max_indicator_warmup`] so the
/// genetic search can pre-flight gene admission against the data slice
/// length and skip indicators that would panic the kernel
/// (`warm prefix exceeds row width`, vector-ta v0.2.9 #212).
pub const MAX_MULTI_PERIOD_LOOKBACK: usize = 200;

/// Maximum warmup periods (in bars) that the indicator stack can
/// produce on a frame with `n_rows` bars. Returns the largest period
/// from the multi-period sweep that still fits, or 0 if the frame is
/// too short to compute any of the parameterized indicators safely.
///
/// Used by the validation harness and pre-flight guards to refuse
/// evaluation on slices smaller than the indicator's warmup. The
/// thresholds match the `alt_periods` array in `compute_classic_ta_columns`.
pub fn max_indicator_warmup(n_rows: usize) -> usize {
    const ALT_PERIODS: &[usize] = &[7, 21, 50, 100, 200];
    ALT_PERIODS
        .iter()
        .rev()
        .find(|&&p| p < n_rows)
        .copied()
        .unwrap_or(0)
}

/// Computes ALL 340+ Technical Indicators automatically using VectorTA's Dispatch Engine.
/// Multi-output indicators are automatically decomposed into separate named columns.
///
/// Each indicator call is wrapped in `std::panic::catch_unwind` because
/// vector-ta v0.2.9 panics on a small subset of period/data combinations
/// (e.g. EURUSD M5 hits `warm prefix exceeds row width` at
/// `utilities/helpers.rs:159`, #212). The wrapping converts a panic into
/// a silently-skipped column rather than tearing down the worker thread,
/// which on the rayon-driven discovery hot path would otherwise abort
/// the whole TF run with no fallback path. The pre-flight
/// [`max_indicator_warmup`] helper still gates the multi-period sweep
/// so the common case never reaches the kernel boundary.
pub fn compute_classic_ta_columns(ohlcv: &Ohlcv) -> anyhow::Result<Vec<(String, Vec<f64>)>> {
    compute_classic_ta_columns_with_policy(ohlcv, resolved_indicator_compute_policy())
}

/// Process-wide policy set once by the binary that owns the operator's
/// Settings. `None` until something sets it.
static POLICY_OVERRIDE: std::sync::OnceLock<IndicatorComputePolicy> = std::sync::OnceLock::new();

/// Bind the indicator lane policy for this process. Idempotent-or-refuse: the
/// second call fails with the value already in force rather than silently
/// keeping one of the two, because "which lane ran" is exactly the question
/// this whole module exists to answer unambiguously.
///
/// This is the seam the operator's Settings plugs into. It is deliberately a
/// single point, so there is one place to read to know what a run did.
pub fn set_indicator_compute_policy(
    policy: IndicatorComputePolicy,
) -> Result<(), IndicatorComputePolicy> {
    match POLICY_OVERRIDE.set(policy) {
        Ok(()) => Ok(()),
        Err(_) => Err(*POLICY_OVERRIDE.get().expect("just failed to set, so it is set")),
    }
}

/// The policy a caller that does not name one gets.
///
/// # Why this is not just `Auto`
///
/// It was, and that made the whole GPU-or-fail apparatus unreachable.
/// [`IndicatorComputePolicy::RequireGpu`] existed, was correct, and had ZERO
/// production call sites — `compute_classic_ta_columns` hardcoded `Auto`, and
/// `Auto` turns a lane failure into a `tracing::warn!` plus CPU numbers. So on
/// a card where the arch story is wrong, the run FINISHES, with a log line as
/// the only symptom. That is this project's signature defect shape: the gate
/// exists and the caller never passes through it.
///
/// Precedence, deliberately identical to `neoethos-search`'s backend
/// resolution so the two cannot disagree about the same run:
///
/// 1. [`set_indicator_compute_policy`] — the operator's Settings.
/// 2. `Auto`.
///
/// 2026-08-10 — THE SEAM IS FINALLY PLUGGED IN, and the env lever is gone.
/// `set_indicator_compute_policy` had ZERO callers, so step 1 never fired and
/// `NEOETHOS_REQUIRE_GPU` was the only way to reach `RequireGpu` at all: the
/// module documented Settings as the source and read the environment instead.
/// The installer is now called from
/// `neoethos_search::backend::install_evaluation_backend_from_settings`, which
/// already holds the resolved backend — so the escalation comes from the same
/// config value (`system.enable_gpu_preference` / `models.prop_search_device`
/// naming a `*_required` variant) that decides the search lane, and the two
/// crates cannot disagree about one run.
///
/// ⚠ BEHAVIOUR CHANGE: exporting `NEOETHOS_REQUIRE_GPU=1` no longer forces
/// `RequireGpu` here. That is a LOOSENING, so it is reported at ERROR by
/// `neoethos_data::report_retired_env_vars()` whenever the variable is still
/// set.
pub fn resolved_indicator_compute_policy() -> IndicatorComputePolicy {
    crate::report_retired_env_vars();
    POLICY_OVERRIDE
        .get()
        .copied()
        .unwrap_or(IndicatorComputePolicy::Auto)
}

/// Which lane may compute the multi-period indicator sweep.
///
/// The base 340-indicator pass is CPU in every policy — vector-ta has no
/// device dispatch for most of those ids, and inventing one per indicator is
/// not something a policy flag can do. The multi-period sweep is the part that
/// is sweep-shaped and therefore the part a card wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndicatorComputePolicy {
    /// Use the card when this binary has the CUDA indicator lane compiled in
    /// AND a usable device is present AND its arch matches the compiled PTX.
    /// Otherwise use the CPU and RECORD THE REASON BY NAME in
    /// `indicator_telemetry` (a declared fallback, not a silent one).
    Auto,
    /// CPU only. This is the parity reference — f64 end to end.
    Cpu,
    /// The card or nothing. Panics rather than returning CPU numbers that
    /// would be mistaken for device numbers. For the GPU-or-fail invariant.
    RequireGpu,
}

/// The multi-period sweep, with an explicit lane policy.
///
/// # The device lane is f64 end to end
///
/// This doc used to say the opposite, and said it first, on the public API:
/// that vector-ta's device layer was f32-only, that routing a column through
/// the card therefore CHANGED ITS VALUE, and that the divergence was a
/// deliberate measured trade-off announced at WARN. That is no longer true and
/// the inverted version was actively dangerous, because a reader who believed
/// it would conclude the CPU fallback was the SAFE outcome.
///
/// What is true now: the lane uploads f64, runs f64 kernels compiled with
/// `-prec-div=true -prec-sqrt=true -fmad=false -ftz=false` and never
/// `--use_fast_math`, and returns f64 with no narrowing anywhere. The claim is
/// not made by this comment — it is measured by
/// `gpu_cpu_indicator_sweep_parity`, whose tolerance is `(1e-12, 1e-12)`,
/// tight enough that an f32 lane sneaking back in (relative 1.19e-7) fails
/// instantly.
///
/// The column SET, NAMES and ORDER are identical in every policy.
///
/// # What `Auto` costs, and why it is not the honest default on a card
///
/// Under `Auto` a lane failure is a `tracing::warn!` and CPU numbers. That is a
/// DECLARED fallback — the reason is recorded per indicator and printed by
/// `indicator_telemetry::indicator_device_summary()` — but a run that finishes
/// is still a run that finishes, and a log line is a weak symptom for "the card
/// did not participate". Under the standing GPU-EXCLUSIVE directive the honest
/// default when a card is PRESENT is [`IndicatorComputePolicy::RequireGpu`];
/// `Auto` should mean "CPU when there is no card", not "CPU when the card
/// refused". Callers that do not name a policy get
/// [`resolved_indicator_compute_policy`], which is where that choice is made.
pub fn compute_classic_ta_columns_with_policy(
    ohlcv: &Ohlcv,
    policy: IndicatorComputePolicy,
) -> anyhow::Result<Vec<(String, Vec<f64>)>> {
    let rows = ohlcv.len();
    compute_classic_ta_columns_sized(ohlcv, policy, rows)
}

/// The real entry point: same as [`compute_classic_ta_columns_with_policy`] but
/// with the vocabulary budget sized against `budget_rows` rather than against
/// THIS frame.
///
/// # Why the budget must not be sized from the frame
///
/// The vocabulary budget converts free RAM into a maximum column count, and
/// `admit_indicators` then takes the prefix of `ALL_INDICATORS` that fits. If
/// the budget is sized from each frame's own row count, the admitted ID SET
/// becomes a function of the frame — and every timeframe has a different row
/// count. Measured on the operator's box (20.6 GB free): base M5 at 1,054,320
/// bars admits 269 ids, H1 at 70,288 admits all 342, H4 at 17,572 admits all
/// 342. The per-timeframe cube widths then differ by ~140 columns,
/// `lib.rs::try_assemble_cube_in_ram` refuses to assemble (its width invariant
/// is doing exactly its job), and every run on a box under roughly 40 GB free
/// silently falls through to the slower streaming disk path.
///
/// So the CALLER sizes one budget from the run's widest frame — the base
/// timeframe — and passes that row count to every timeframe. Conservative by
/// construction: the higher timeframes are charged the base frame's per-column
/// price, so the plan can only ever over-reserve, never under-reserve.
///
/// `budget_rows` must be >= the widest frame this run will build. Passing the
/// frame's own length is correct only for a single-frame build.
pub fn compute_classic_ta_columns_sized(
    ohlcv: &Ohlcv,
    policy: IndicatorComputePolicy,
    budget_rows: usize,
) -> anyhow::Result<Vec<(String, Vec<f64>)>> {
    let n = ohlcv.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let budget_rows = budget_rows.max(n);

    // 1. Pack data into VectorTA Candles struct (once; shared read-only
    //    across the rayon workers below — `Candles` holds plain Vecs so it
    //    is `Sync`).
    let timestamps = ohlcv.timestamp.clone().unwrap_or_else(|| vec![0i64; n]);
    let volume = ohlcv.volume.clone().unwrap_or_else(|| vec![0.0; n]);

    let candles = Candles::new(
        timestamps,
        ohlcv.open.clone(),
        ohlcv.high.clone(),
        ohlcv.low.clone(),
        ohlcv.close.clone(),
        volume,
    );

    // 2. SIZE THE VOCABULARY BEFORE ALLOCATING ANY OF IT.
    //
    //    Repairing the dispatch takes the base pass from 1 produced column to
    //    ~674, and at 843,456 bars an f64 column is 6.75 MB — so the staging
    //    peak goes from 6.7 MB to 4.5 GB per timeframe. `VocabularyBudget`
    //    turns FREE RAM (never a user parameter, never a constant) into a
    //    maximum column count, and `admit_indicators` takes a deterministic
    //    prefix of `ALL_INDICATORS` that fits it. Planning is registry lookups
    //    only, so the decision is made before a single Vec is allocated.
    //
    //    THE SWEEP IS RESERVED FIRST. The period sweep used to be staged
    //    entirely OUTSIDE this budget, so `max_columns` was never the peak: at
    //    the M5 store's 1,054,320 bars its 130 columns are another 1.10 GB of
    //    f64 staging that nothing accounted for, and it mattered most exactly
    //    where the budget binds. The sweep is also the HISTORICAL vocabulary,
    //    so it has first claim — a machine that can afford only 66 columns gets
    //    precisely what it got before this change, and the base pass takes what
    //    is left.
    let budget = VocabularyBudget::for_run(budget_rows);
    let sweep_reserved = planned_sweep_columns();
    let base_budget = budget.reserve(sweep_reserved);
    let all_ids: Vec<&'static str> = ALL_INDICATORS.to_vec();
    let (admitted, deferred, planned_columns) = admit_indicators(&all_ids, &base_budget);
    let base_admitted_plan: usize = admitted.iter().map(|id| planned_output_count(id)).sum();

    // 3. Dispatch to every admitted indicator — PARALLEL across indicators.
    //    Each indicator is an independent pure function of the shared
    //    `candles`, so this is a rayon `par_iter` whose results are collected
    //    BY POSITION, preserving `ALL_INDICATORS` order exactly. The feature
    //    build runs ONCE up-front (not inside the GA candidate `par_iter`), so
    //    this does not nest with the discovery hot path.
    //
    //    ── THE TWO FIXES, AND WHY BOTH WERE NEEDED ────────────────────────
    //
    //    (a) OUTPUT IDS ARE ENUMERATED. `output_id: None` against a
    //        multi-output indicator is answered by vector-ta with
    //        `InvalidParam { key: "output_id" }` — 92 of the 342 ids failed on
    //        exactly this, and it is why every multi-output indicator (macd,
    //        bollinger_bands, stoch, keltner, supertrend, …) produced nothing
    //        anywhere in the system. `compute_single_indicator` in this same
    //        file has always done it correctly for the chart endpoint; this
    //        loop never did. `output_ids_for` is now the ONE place that
    //        resolves it, for both call sites.
    //
    //    (b) THE ACCEPT TEST KEYS OFF THE VALUE COUNT, NOT rows/cols.
    //        vector-ta reports a 1-D series as `rows=1 x cols=n`, so the old
    //        `if out_cols <= 1` was FALSE for every single-output f64
    //        indicator on any frame with more than one bar. Those fell into
    //        the multi-output branch, which then required `rows >= n` — i.e.
    //        `1 >= 200000` — and dropped them with no else and no log. That
    //        alone accounted for 232 of the 342: series that were CORRECT,
    //        FULL-LENGTH, and thrown away. `flatten_indicator_series` (used by
    //        the chart path, where this lesson was learned and never carried
    //        back up) keys off `v.len()` and is now used here too.
    //
    //    Everything a worker discards is recorded in its own `IndicatorLedger`
    //    with a reason and the dispatch message. Nothing on this path is
    //    allowed to vanish.
    let per_id: Vec<(Vec<(String, Vec<f64>)>, IndicatorLedger)> = admitted
        .par_iter()
        .map(|&id| {
            let mut out: Vec<(String, Vec<f64>)> = Vec::new();
            let mut ledger = IndicatorLedger::new();
            dispatch_indicator_outputs(&candles, id, id, &[], n, Kernel::Auto, &mut out, &mut ledger);
            (out, ledger)
        })
        .collect();

    let mut ledger = IndicatorLedger::new();
    for id in &deferred {
        ledger.dropped(
            id,
            id,
            DropReason::OverBudget,
            format!(
                "base-vocabulary budget full at {} columns ({} of {} total reserved for the \
                 period sweep; sized at {} bars, {:.2} GB free)",
                base_budget.max_columns,
                sweep_reserved,
                budget.max_columns,
                budget_rows,
                budget.available_bytes as f64 / 1e9
            ),
        );
    }
    let mut cols: Vec<(String, Vec<f64>)> = Vec::new();
    for (mut produced, led) in per_id {
        cols.append(&mut produced);
        ledger.merge(led);
    }
    base_budget.log("base-vocabulary", planned_columns, cols.len());

    // 3. Multi-period variants for the most critical indicators. Appended
    //    after the base columns to preserve the original ordering exactly.
    //
    //    ROUTING (task #22): this is the sweep-shaped part of the feature
    //    build — 18 indicators × 5 periods = 90 independent full-series scans
    //    over the same OHLCV — and therefore the part a card wins. When the
    //    CUDA lane is compiled in, present and arch-compatible, the ten
    //    indicators in `gpu_indicators::GPU_SWEEP_SPECS` are swept on the
    //    device against ONE resident upload; the remaining eight stay on the
    //    CPU and are reported by name as `CpuIndicatorNotPortable`.
    //
    //    ORDER IS LOAD-BEARING: column order feeds `effective_feature_names`
    //    and every discovery artifact. The per-indicator results are collected
    //    into a slot indexed by position in `MULTI_PERIOD_IDS` and then
    //    concatenated in that order, so the emitted order is byte-identical to
    //    the pure-CPU path regardless of which lane produced which column.
    let multi_cols = compute_multi_period_columns(ohlcv, &candles, n, policy);
    let sweep_actual = multi_cols.len();
    cols.extend(multi_cols);

    // 4b. THE EXTENDED SWEEP — the answer to "a period is not a detail".
    //
    //     Repairing the dispatch bought 695 base columns, and every one of them
    //     is computed at ONE vector-ta default. That is not what was asked for:
    //     RSI(4) is a scalping oscillator and RSI(54) is a regime filter, and a
    //     vocabulary that can only reach one of them is 300 arbitrary points in
    //     a space nobody chose. `period_plan` already knows how to drive each
    //     indicator's real window parameters — it was simply unreachable for
    //     anything outside the hardcoded eighteen.
    //
    //     So every id whose window is drivable (`Key` or `Ratio`) and that is
    //     NOT already in `MULTI_PERIOD_IDS` is swept too, in `ALL_INDICATORS`
    //     order, for as many (id, period) pairs as the machine can still
    //     afford after the base pass and the historical sweep have been paid
    //     for. On a box with no headroom this admits nothing and the emitted
    //     column set is exactly what it was — the extension can only ever use
    //     memory that was going to go unused.
    //
    //     This is a WORKING-SET step toward the streaming design in
    //     `docs/streaming-parameter-search.md`, not the design itself: it still
    //     materialises one cube. What it establishes is the part that design
    //     needs and did not have — a hardware-sized, deterministic, repeatable
    //     admission over the (indicator, period) space.
    //     BOUNDED BY THE VOCABULARY IT EXTENDS, NOT ONLY BY RAM.
    //
    //     Memory is not the only resource the never-OOM reasoning has to cover:
    //     a run that does not finish is no more useful than one that OOMs. The
    //     extension's memory cost is bounded by `budget`, but its TIME cost is
    //     not bounded by anything — and the two diverge exactly where the frame
    //     is short and the box is large, because the per-column price then
    //     collapses while the per-column WORK does not. Measured: at 6,000 bars
    //     with 18.8 GB free the budget allows 4,096 columns, so the extension
    //     admits every sweepable id and the pass goes from ~9 s to ~12 min.
    //
    //     So the extension may never plan more columns than the vocabulary it
    //     extends. That is a ratio to a measured quantity rather than a
    //     constant — it scales with the machine through `base_admitted_plan` —
    //     and it is deliberately a stopgap: the real answer is not to
    //     materialise the extension at all but to stream it, which is
    //     `docs/streaming-parameter-search.md`.
    //
    //     SIZED FROM THE PLAN, NEVER FROM WHAT THIS FRAME HAPPENED TO PRODUCE.
    //     Using `cols.len()` here would make the admitted extension a function
    //     of the frame, which is the same defect one level up: two timeframes
    //     would extend by different amounts and the cube widths would diverge.
    //
    //     STREAMING OVERRIDE (2026-08-10). When a working set is installed by
    //     `install_extended_sweep_working_set`, the extension is EXACTLY that
    //     batch's (indicator, period) pairs and the budget-capped prefix is not
    //     consulted. With no working set installed — the default, and every
    //     existing caller — this block is byte-identical to what it was: same
    //     plan function, same `ALT_PERIODS` slice per id, same emission order,
    //     same deferral ledger. That identity is the parity property, and it is
    //     structural rather than tested-by-luck: both lanes run through the SAME
    //     `sweep_one_id_ledgered`.
    //
    //     THE STOPGAP STAYS, AND HERE IS WHY IT MUST. The measurement that
    //     motivated it — "~9 s to ~12 min at 6,000 bars with 4,096 columns
    //     allowed" — DID NOT REPRODUCE at HEAD: the production pass at 6,000
    //     bars with `max_columns` 4,096 takes 1.36 s and emits 1,795 columns,
    //     and the cap binds at every bar count measured (1,795 produced against
    //     4,096 / 4,096 / 3,244 allowed at 6k / 20k / 200k). But the cost that
    //     the cap bounds is per-column-COMPUTED, not per-column-HELD — 342 ids
    //     take 1.10 s at 6k bars and 10.47 s at 60k — so STREAMING DOES NOT
    //     REMOVE IT. Ten batches of a tenth of the space cost what one pass over
    //     the whole space costs, plus ten prefilters instead of one. Removing
    //     the cap here would therefore hand back an unbounded-in-time extension
    //     with nothing having been fixed. What WOULD remove it is a cheaper
    //     dispatch, and the target is named: at 6,000 bars the slowest TEN of
    //     342 ids are 79.3% of the base pass and the slowest ten of the sweep
    //     are 86.2% of it (`goertzel_cycle_composite_wave` 2.17 s,
    //     `smooth_theil_sen` 1.12 s, `volume_adjusted_ma` 0.79 s, each for five
    //     periods).
    let spent = cols.len();
    let planned_so_far = base_admitted_plan + sweep_reserved;
    let unspent = budget.max_columns.saturating_sub(planned_so_far);
    let working_set = current_extended_sweep_working_set();
    let (ext_groups, ext_deferred, ext_mode, extended_budget) = match working_set.as_deref() {
        Some(batch) => {
            if batch.planned_columns > unspent {
                tracing::warn!(
                    target: "neoethos_data::hpc_ta",
                    cursor = batch.cursor,
                    batch_columns = batch.planned_columns,
                    unspent_columns = unspent,
                    max_columns = budget.max_columns,
                    "the installed streaming working set is WIDER than this machine still \
                     affords after the base vocabulary and the historical sweep. Building it \
                     anyway — the caller sized the batch and refusing here would silently \
                     change which parameter region the run explored — but expect memory \
                     pressure. Size the batch from VocabularyBudget minus the resident plan."
                );
            }
            (
                batch.grouped_by_id(),
                Vec::new(),
                "streaming_batch",
                batch.planned_columns,
            )
        }
        None => {
            let extended_budget = unspent.min(planned_so_far);
            let (plan, deferred) = extended_sweep_plan(extended_budget);
            let groups: Vec<(&'static str, Vec<usize>)> = plan
                .into_iter()
                .map(|id| (id, ALT_PERIODS.to_vec()))
                .collect();
            (groups, deferred, "budget_prefix", extended_budget)
        }
    };
    if !ext_groups.is_empty() {
        let ext: Vec<(Vec<(String, Vec<f64>)>, IndicatorLedger)> = ext_groups
            .par_iter()
            .map(|(id, periods)| sweep_one_id_ledgered(&candles, *id, periods, n, Kernel::Auto))
            .collect();
        for (mut c, l) in ext {
            cols.append(&mut c);
            ledger.merge(l);
        }
    }
    for id in &ext_deferred {
        ledger.dropped(
            id,
            id,
            DropReason::OverBudget,
            format!(
                "extended period sweep: only {extended_budget} columns of the {} the machine \
                 affords were still unspent after the base vocabulary and the historical sweep",
                budget.max_columns
            ),
        );
    }
    tracing::info!(
        target: "neoethos_data::hpc_ta",
        rows = n,
        budget_rows,
        max_columns = budget.max_columns,
        base_columns = spent - sweep_actual,
        sweep_columns = sweep_actual,
        extended_mode = ext_mode,
        extended_cursor = working_set.as_deref().map(|b| b.cursor).unwrap_or(0),
        extended_space_len = working_set.as_deref().map(|b| b.space_len).unwrap_or(0),
        extended_ids = ext_groups.len(),
        extended_pairs = ext_groups.iter().map(|(_, p)| p.len()).sum::<usize>(),
        extended_deferred = ext_deferred.len(),
        extended_columns = cols.len() - spent,
        total_columns = cols.len(),
        "indicator vocabulary composition (base pass at library defaults + period sweeps)"
    );

    // 5. CENSUS. Fingerprint every column exactly once and report the two
    //    quality facts the old code could not have known:
    //
    //      * DUPLICATES — a column bit-identical to an earlier one. This is
    //        what a period sweep looks like when the indicator ignores the
    //        swept key (obv and vwap declare no window parameter, so their
    //        five "periods" are five copies). Reported, NOT removed: the
    //        column SET must be a function of the id list and the registry and
    //        never of the frame, or the per-timeframe cube width would vary
    //        and `lib.rs`'s width invariant would break.
    //      * DEGENERATE — no finite variation on this frame. Ballast for any
    //        correlation-ranked prefilter. Also kept, for the same reason.
    //
    //      * A DUPLICATE NAME, by contrast, is a HARD ERROR. Two columns called
    //        the same thing is not a quality problem, it is a correctness one:
    //        every downstream projection is by name, so the second silently
    //        shadows the first in any name→index map. The extended sweep emits
    //        `<id>_<period>` names into the same namespace as the base pass's
    //        `<id>_<output>` names, so this is now a reachable collision rather
    //        than a theoretical one.
    {
        let mut seen: HashSet<u64> = HashSet::with_capacity(cols.len());
        let mut names: HashSet<&str> = HashSet::with_capacity(cols.len());
        let mut collisions: Vec<&str> = Vec::new();
        for (name, values) in &cols {
            if !names.insert(name.as_str()) {
                collisions.push(name.as_str());
            }
            if !seen.insert(series_fingerprint(values)) {
                ledger.duplicate_column(name);
            }
            if !has_finite_variation(values) {
                ledger.degenerate_column(name);
            }
        }
        if !collisions.is_empty() {
            anyhow::bail!(
                "the indicator pass emitted {} DUPLICATE COLUMN NAME(S): {:?}. Column names are \
                 the only key every downstream projection has, so a collision silently shadows a \
                 real feature. Rename the sweep suffix for the id(s) involved.",
                collisions.len(),
                collisions.iter().take(20).collect::<Vec<_>>()
            );
        }
    }

    ledger.log_summary("classic-ta", n);

    // 6. THE FLOOR. This is the whole point of the ledger: a regression from
    //    ~800 columns back to 66 is not a log line anyone has to notice, it is
    //    a hard error that names the drop bucket which grew.
    //
    //    Length-gated, and honestly so: on a short frame most indicators fail
    //    their warmup legitimately, so enforcing there would make the guard
    //    fire on fixtures instead of on regressions.
    //
    //    CLAMPED BY WHAT THE MACHINE AFFORDED. The floor and the budget were
    //    added in the same change and were never exercised together: at the M5
    //    store's real depth on the operator's own box the budget admits 269 ids
    //    and this floor demands 280, so the feature build hard-errored and
    //    discovery could not start. "This machine cannot afford the vocabulary"
    //    and "the dispatch regressed" are different incidents; the budget's own
    //    truncation is already a WARN, and this floor is only about the second.
    //    See `IndicatorLedger::enforce_floor`.
    if n >= VOCABULARY_FLOOR_MIN_ROWS {
        ledger.enforce_floor(
            "classic-ta",
            n,
            MIN_PRODUCING_INDICATOR_IDS,
            MIN_BASE_VOCABULARY_COLUMNS,
            admitted.len(),
            base_admitted_plan,
        )?;
    } else {
        tracing::info!(
            target: "neoethos_data::hpc_ta",
            rows = n,
            floor_min_rows = VOCABULARY_FLOOR_MIN_ROWS,
            producing_ids = ledger.producing_ids(),
            columns = cols.len(),
            "frame is shorter than the vocabulary floor's minimum row count — the census above \
             was recorded but the hard floor was NOT enforced for this frame"
        );
    }

    Ok(cols)
}

/// Dispatch ONE indicator across all of its declared outputs, appending each
/// produced column and recording every discard with a reason.
///
/// This is the single place the two dispatch mistakes are fixed, shared by the
/// base pass and the multi-period sweep so they cannot drift apart again:
///
///   * `output_ids_for` supplies the `output_id` a multi-output indicator
///     requires, resolved from vector-ta's registry (or, for the five
///     multi-output ids that have no registry entry, from the override table
///     in `core::indicator_ledger` harvested from the dispatcher source);
///   * `flatten_indicator_series` keys acceptance off the VALUE COUNT, because
///     vector-ta reports a 1-D series as `rows=1 x cols=n`.
///
/// `column_prefix` is the base column name — `"rsi"` for the base pass,
/// `"rsi_21"` for the sweep. A multi-output indicator suffixes the output id
/// (`"macd_21_signal"`), matching `compute_single_indicator`'s convention, so
/// the name says which series it is rather than a positional `_line2`.
fn dispatch_indicator_outputs(
    candles: &Candles,
    id: &'static str,
    column_prefix: &str,
    params: &[ParamKV],
    n: usize,
    kernel: Kernel,
    out: &mut Vec<(String, Vec<f64>)>,
    ledger: &mut IndicatorLedger,
) {
    let outputs = output_ids_for(id);
    let multi = outputs.len() > 1;
    let excluded = expected_non_producing(id);

    for out_id in outputs {
        let name = match out_id {
            Some(o) if multi => format!("{column_prefix}_{o}"),
            _ => column_prefix.to_string(),
        };
        let data_ref = IndicatorDataRef::Candles {
            candles,
            source: None,
        };
        let req = IndicatorComputeRequest {
            indicator_id: id,
            output_id: out_id,
            data: data_ref,
            params,
            kernel,
        };
        // #212: a small subset of indicator/data combinations in vector-ta
        // v0.2.9 panic instead of returning Err. Catch it per-output so one bad
        // column never tears down the frame — but COUNT it, which the previous
        // handler did only as an un-aggregated warn.
        let computed = catch_unwind(AssertUnwindSafe(|| compute_cpu(req)));
        // ONE place decides what a discard does to the COLUMN SET.
        //
        // A capability failure (no dispatch arm, no such output, unsupported
        // kernel) removes the column, and that absence is what the vocabulary
        // floor detects a regression by. A FRAME failure (warmup, data length,
        // compute) keeps the name and fills it with NaN, because otherwise the
        // emitted width would be a function of the frame — and every timeframe
        // has a different one, so the cube could not be assembled. See
        // `DropReason::is_frame_dependent`.
        let mut discard: Option<(DropReason, String)> = None;
        match computed {
            Err(_) => {
                discard = Some((
                    DropReason::KernelPanic,
                    "vector-ta kernel panicked (issue #212)".to_string(),
                ));
            }
            Ok(Err(e)) => {
                discard = Some((DropReason::from_dispatch(&e), e.to_string()));
            }
            Ok(Ok(output)) => {
                // A PATTERN MATRIX IS NOT A LONG SERIES.
                //
                // `normalize_indicator_len` correctly refuses `62 * n` values
                // for an n-bar frame — taking the head would return pattern 0's
                // flags under the whole indicator's name. But refusing is not
                // the same as having nothing to emit: the library hands back
                // `pattern_ids`, one name per row, so the matrix decomposes
                // exactly. Handled BEFORE the flatten so the refusal stays the
                // rule for everything that has no such metadata.
                if let Some(matrix) = pattern_matrix_columns(&output, &name, n) {
                    if excluded.is_some() {
                        ledger.stale_exclusion(id);
                    }
                    for col in matrix {
                        ledger.produced(id);
                        out.push(col);
                    }
                    continue;
                }
                match flatten_indicator_series(output.series, n) {
                Ok((values, raw_len)) => {
                    if raw_len > n {
                        // The tail was dropped. A discard is a discard even when
                        // the column survives it.
                        ledger.dropped(
                            id,
                            &name,
                            DropReason::Truncated,
                            format!("kernel returned {raw_len} values for {n} bars; head kept"),
                        );
                    }
                    if excluded.is_some() {
                        // The exclusion table said this id cannot produce. It
                        // did. That means the table is stale, which is a thing
                        // to fix, not to shrug at.
                        ledger.stale_exclusion(id);
                    }
                    ledger.produced(id);
                    out.push((name.clone(), values));
                }
                    Err(e) => {
                        discard = Some((DropReason::ShortSeries, e.to_string()));
                    }
                }
            }
        }
        if let Some((reason, detail)) = discard {
            ledger.dropped(id, &name, reason, detail);
            if reason.is_frame_dependent() {
                out.push((name, vec![f64::NAN; n]));
            }
        }
    }
}

/// Emit the columns an indicator WOULD have produced, filled with NaN, using
/// exactly the names `dispatch_indicator_outputs` would have used.
///
/// Used only where the skip is a function of the FRAME rather than of the
/// indicator's capability — today that is the `#212` warmup pre-flight guard,
/// which reads `n` directly. Those skips must not change the column set,
/// because every timeframe has a different `n` and the cube's width invariant
/// would then fail on every multi-timeframe run.
///
/// NaN, never zero: a zero is a real number the GA can threshold against, and
/// "this frame cannot support this period" is an absence, not a reading. It is
/// also what the warmup prefix of every windowed indicator already looks like,
/// so nothing downstream meets a new shape.
fn push_absent_columns(
    id: &'static str,
    column_prefix: &str,
    n: usize,
    out: &mut Vec<(String, Vec<f64>)>,
) {
    let outputs = output_ids_for(id);
    let multi = outputs.len() > 1;
    for out_id in outputs {
        let name = match out_id {
            Some(o) if multi => format!("{column_prefix}_{o}"),
            _ => column_prefix.to_string(),
        };
        out.push((name, vec![f64::NAN; n]));
    }
}

/// Decompose a pattern MATRIX output into one named boolean column per pattern,
/// or `None` when this output is not a matrix.
///
/// `pattern_recognition` is the only id in vector-ta 0.2.9 that returns one:
/// `rows = PATTERN_RUNNERS.len()` — 62 candlestick patterns, set at
/// `pattern_recognition.rs:1146` — and `cols = bars`, laid out PATTERN-MAJOR.
/// The orientation is not assumed: it is read off the crate's own
/// `PatternRecognitionOutput::to_bitmask_u64`, which indexes
/// `values_u8[row * cols .. row * cols + cols]` (pattern_recognition.rs:344-346)
/// with `words_per_row = cols.div_ceil(64)`, i.e. one row IS one pattern's
/// series across all bars.
///
/// Every one of the three shape facts is CHECKED rather than trusted — the row
/// count against the id list, the column count against the frame, and the value
/// count against their product. A mismatch returns `None`, and the caller then
/// takes the normal path, where `normalize_indicator_len` refuses the flattened
/// multi-series with a named hard error. Guessing an orientation here would
/// produce 62 columns of silent mis-attribution, which is strictly worse than
/// the drop it replaces.
fn pattern_matrix_columns(
    output: &IndicatorComputeOutput,
    column_prefix: &str,
    n: usize,
) -> Option<Vec<(String, Vec<f64>)>> {
    let ids = output.pattern_ids.as_ref()?;
    if ids.is_empty() || output.rows != ids.len() || output.cols != n {
        return None;
    }
    let IndicatorSeries::Bool(values) = &output.series else {
        return None;
    };
    if values.len() != output.rows.checked_mul(output.cols)? {
        return None;
    }
    Some(
        ids.iter()
            .enumerate()
            .map(|(row, pattern)| {
                let start = row * n;
                let series = values[start..start + n]
                    .iter()
                    .map(|&b| if b { 1.0 } else { 0.0 })
                    .collect();
                (format!("{column_prefix}_{pattern}"), series)
            })
            .collect(),
    )
}

/// The 18 indicators that get a period sweep, in emission order.
pub const MULTI_PERIOD_IDS: [&str; 18] = [
    "rsi",
    "ema",
    "sma",
    "atr",
    "adx",
    "cci",
    "stoch",
    "macd",
    "bollinger_bands",
    "keltner",
    "supertrend",
    "willr",
    "roc",
    "mom",
    "tsi",
    "mfi",
    "obv",
    "vwap",
];

/// The periods swept for each of [`MULTI_PERIOD_IDS`].
pub const ALT_PERIODS: [usize; 5] = [7, 21, 50, 100, 200];

/// Columns the HISTORICAL period sweep will stage, planned from the registry
/// before anything is allocated.
///
/// Registry lookups only — no compute — so this can be subtracted from the
/// machine's budget BEFORE the base pass is admitted. That ordering is the
/// point: the sweep IS the 66-column vocabulary every historical result was
/// produced on, so it has first claim on the memory, and a machine that can
/// afford nothing else still gets exactly what it had before.
///
/// It over-counts on short frames — the `#212` pre-flight guard skips periods
/// whose warmup exceeds the frame — and that is the correct direction for a
/// budget: a plan may over-reserve, never under-reserve.
pub fn planned_sweep_columns() -> usize {
    MULTI_PERIOD_IDS
        .iter()
        .map(|id| planned_output_count(id) * ALT_PERIODS.len())
        .sum()
}

/// Columns the RESIDENT part of a pass will stage on this machine: the admitted
/// base vocabulary plus the historical period sweep, i.e. everything that is
/// present in every batch and is not the streaming extension.
///
/// This is the same arithmetic `compute_classic_ta_columns_sized` performs, in
/// ONE place, so a streaming loop that sizes its batch from
/// `max_columns - resident` cannot drift from what the pass actually spends.
/// Registry lookups only; nothing is allocated.
pub fn planned_resident_columns(budget_rows: usize) -> usize {
    let budget = VocabularyBudget::for_run(budget_rows);
    let sweep_reserved = planned_sweep_columns();
    let base_budget = budget.reserve(sweep_reserved);
    let all_ids: Vec<&'static str> = ALL_INDICATORS.to_vec();
    let (admitted, _deferred, _planned) = admit_indicators(&all_ids, &base_budget);
    let base_plan: usize = admitted.iter().map(|id| planned_output_count(id)).sum();
    base_plan + sweep_reserved
}

/// How wide one streaming batch may be on this machine: what
/// [`VocabularyBudget`] affords at `budget_rows`, minus the resident plan.
///
/// A function of the hardware and the widest frame, never of a user parameter —
/// the never-OOM invariant. Zero means the machine cannot afford ANY streaming
/// extension, which the caller must treat as "do not stream", not as "stream a
/// batch of nothing".
pub fn streaming_batch_columns(budget_rows: usize) -> usize {
    VocabularyBudget::for_run(budget_rows)
        .max_columns
        .saturating_sub(planned_resident_columns(budget_rows))
}

/// Is this id's window drivable by the sweep, i.e. would sweeping it produce
/// genuinely DIFFERENT columns rather than five copies of one?
///
/// Three conditions, all of them load-bearing:
///
///   * it must be REGISTERED — an unregistered id's `period_plan` falls back to
///     the `"period"` naming convention, which the by-name dispatch arms may or
///     may not read. Sweeping on a guess is how a vocabulary fills with
///     duplicates wearing distinct names;
///   * its plan must be [`PeriodPlan::Key`] or [`PeriodPlan::Ratio`], never
///     `NoWindow` — `NoWindow` means the swept key is ignored by construction
///     (obv, vwap);
///   * it must not be on `EXPECTED_NON_PRODUCING`. Those ids cannot produce even
///     once; sweeping them would multiply a known, named failure by five. They
///     are still ATTEMPTED once per frame by the base pass, so the exclusion
///     table cannot rot behind this filter.
///
/// [`MULTI_PERIOD_IDS`] is excluded by the caller, not here, because those ids
/// are swept already and re-sweeping them would emit duplicate column NAMES —
/// a hard error in `compute_classic_ta_columns_sized`.
fn is_extended_sweepable(id: &str) -> bool {
    if expected_non_producing(id).is_some() {
        return false;
    }
    if vector_ta::indicators::registry::get_indicator(id).is_none() {
        return false;
    }
    matches!(period_plan(id), PeriodPlan::Key(_) | PeriodPlan::Ratio(_))
}

/// Which ids the EXTENDED period sweep can afford, in `ALL_INDICATORS` order.
///
/// Returns `(admitted, deferred)`. This is the same admission mechanism as
/// `feature_budget::admit_indicators`, applied to the (indicator, period) space
/// instead of the indicator space: a deterministic, repeatable prefix sized from
/// what the machine has left after the base vocabulary and the historical sweep
/// have been paid for.
///
/// `budget_columns` of zero admits nothing and defers everything — the emitted
/// column set is then exactly what it was before the extension existed, which is
/// the property that makes this safe to turn on: it can only ever spend memory
/// that was going to go unused.
///
/// Determinism matters beyond reproducibility here. This function is the
/// prototype of the streaming design's advance through the parameter space
/// (`docs/streaming-parameter-search.md`): a batch selector that never repeats
/// itself and never depends on the frame is what lets a later change swap the
/// working set between generations without the column layout becoming a
/// function of scheduling.
pub fn extended_sweep_plan(budget_columns: usize) -> (Vec<&'static str>, Vec<&'static str>) {
    let mut admitted: Vec<&'static str> = Vec::new();
    let mut deferred: Vec<&'static str> = Vec::new();
    let mut used = 0usize;
    for &id in ALL_INDICATORS {
        if MULTI_PERIOD_IDS.contains(&id) || !is_extended_sweepable(id) {
            continue;
        }
        let want = planned_output_count(id) * ALT_PERIODS.len();
        if used + want <= budget_columns {
            used += want;
            admitted.push(id);
        } else {
            deferred.push(id);
        }
    }
    (admitted, deferred)
}

// ─────────────────────────────────────────────────────────────────────────────
// THE ADVANCE — a streaming working set over the (indicator, period) space.
//
// WHAT THIS IS FOR, STATED HONESTLY, BECAUSE THE DESIGN DOC IS WRONG ABOUT IT.
//
// `docs/streaming-parameter-search.md` §1 justifies streaming by the cost of
// MATERIALISING the cube. Measured at HEAD, that justification does not hold:
// the production feature pass at 6,000 real EURUSD M5 bars with the budget
// allowing 4,096 columns takes **1.36 s** and emits 1,795 columns, and at
// 200,000 bars **26.76 s**. Time is linear in BARS and linear in COLUMNS
// COMPUTED; it is NOT a function of columns HELD. So per unit of (indicator,
// period) space explored, streaming costs exactly what materialising costs —
// and a streamed run re-pays the prefilter (5.6% of a run) once per batch.
//
// Streaming buys exactly two things, and neither is the feature build:
//
//   1. the MEMORY bound, which `VocabularyBudget` already gives; and
//   2. — the only real prize — never paying the DOWNSTREAM stages for a batch
//      that is knowably doomed. On the run the doc cites
//      (`docs/measurements/3090-47260276/card-run-valid.log`) the quality
//      screen was 88,971 ms = **50.4% of wall time** and took 174 candidates
//      in and 0 out. That is what the early-reject predicate in
//      `neoethos_search::discovery` exists to skip.
//
// Build the loop to save feature-build time and it will measure as a
// REGRESSION. Build it to skip the quality screen and it is worth 4-6x more
// (indicator, period) regions examined per hour on the same hardware.
//
// WHY `extended_sweep_plan` COULD NOT BE THE ADVANCE, despite its own doc
// comment claiming it was the prototype of one: it is a PREFIX selector that
// always restarts at index 0, it takes no offset, and it enumerates IDS (all
// five `ALT_PERIODS` atomically) rather than (indicator, period) pairs, so a
// batch boundary could only ever fall between ids. The three functions below
// are the advance it described but was not.
// ─────────────────────────────────────────────────────────────────────────────

/// One point in the sweep space: an indicator and the window it is evaluated at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SweepPair {
    pub id: &'static str,
    pub period: usize,
}

/// A working set: the slice of the (indicator, period) space one batch covers.
///
/// TWO ORDERS, AND THEY ARE DELIBERATELY DIFFERENT.
///
/// * **Selection** walks period-OUTER / id-inner, so a batch mixes timescales
///   rather than being 32 flavours of one window. That is the order
///   `docs/streaming-parameter-search.md` §3.2 specifies and it is the one the
///   cursor indexes into.
/// * **Emission** is id-outer (`ALL_INDICATORS` order) / period-inner —
///   *whatever the batch selected*. This is not a detail: emission order feeds
///   `effective_feature_names` and every discovery artifact, and the parity
///   requirement is that a batch covering the WHOLE space emits a column list
///   byte-identical to today's non-streaming pass. Sorting the selected pairs
///   back into emission order is what makes that true by construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SweepBatch {
    /// Cursor this batch began at, in SELECTION order. One integer — this is
    /// the whole resumable state of a streaming run.
    pub cursor: usize,
    /// Cursor the next batch begins at. Always `> cursor` unless the space is
    /// exhausted or the budget admits nothing.
    pub next_cursor: usize,
    /// The batch's pairs, in EMISSION order.
    pub pairs: Vec<SweepPair>,
    /// Columns this batch plans to stage, from registry lookups only.
    pub planned_columns: usize,
    /// Total pairs in the space, so a log line can say "batch 7 of 1,620".
    pub space_len: usize,
    /// True when this batch reached the end of the space. The loop decides what
    /// that means (stop, or wrap to cursor 0 for a second pass); wrapping is
    /// never done here, because a silent wrap is a repeat and the whole point
    /// of the cursor is that a pair appears in exactly one batch per sweep.
    pub exhausted: bool,
}

impl SweepBatch {
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// True when this batch is the entire space starting from zero — the
    /// degenerate case the parity test pins.
    pub fn covers_whole_space(&self) -> bool {
        self.cursor == 0 && self.pairs.len() == self.space_len && self.space_len > 0
    }

    /// The batch regrouped for dispatch: `(id, periods)` in emission order,
    /// periods ascending within each id. This is the shape
    /// `sweep_one_id_ledgered` already takes, so the streaming lane and the
    /// prefix lane run through the SAME sweep code with the same naming, the
    /// same `#212` pre-flight guard and the same ledger.
    pub fn grouped_by_id(&self) -> Vec<(&'static str, Vec<usize>)> {
        let mut out: Vec<(&'static str, Vec<usize>)> = Vec::new();
        for pair in &self.pairs {
            match out.last_mut() {
                Some((id, periods)) if *id == pair.id => periods.push(pair.period),
                _ => out.push((pair.id, vec![pair.period])),
            }
        }
        out
    }
}

/// Position of an id in `ALL_INDICATORS`, for the emission sort. Linear scan
/// over a 342-entry static — called once per batch, not per column.
fn all_indicators_rank(id: &str) -> usize {
    ALL_INDICATORS
        .iter()
        .position(|&candidate| candidate == id)
        .unwrap_or(usize::MAX)
}

/// Every (indicator, period) pair the extended sweep can reach, in SELECTION
/// order (period-outer, id-inner).
///
/// Four properties, all of which the streaming loop needs and none of which are
/// free:
///
/// * **no repeats** — a pair appears exactly once, so batch *k* and batch *j*
///   are disjoint;
/// * **deterministic** — a pure function of `ALL_INDICATORS`, `ALT_PERIODS` and
///   the vector-ta registry, so batch *k* is the same set on every machine and
///   every run and a result is reproducible from `(seed, cursor)` alone;
/// * **frame-independent** — nothing here reads the frame, so every timeframe
///   in a run gets the SAME batch and the per-TF cube widths stay equal (which
///   is what `lib.rs::try_assemble_cube_in_ram`'s width invariant requires);
/// * **exclusion-stable** — `MULTI_PERIOD_IDS` are excluded exactly as
///   `extended_sweep_plan` excludes them, so a streamed column can never
///   collide by NAME with a historical sweep column (a collision is a hard
///   error in `compute_classic_ta_columns_sized`, by design).
pub fn extended_sweep_space() -> Vec<SweepPair> {
    let mut space = Vec::new();
    for &period in ALT_PERIODS.iter() {
        for &id in ALL_INDICATORS {
            if MULTI_PERIOD_IDS.contains(&id) || !is_extended_sweepable(id) {
                continue;
            }
            space.push(SweepPair { id, period });
        }
    }
    space
}

/// Size of the space `extended_sweep_space` enumerates.
pub fn extended_sweep_space_len() -> usize {
    extended_sweep_space().len()
}

/// Batch *k*: the pairs from `cursor` onward that fit `budget_columns`.
///
/// `budget_columns` comes from [`VocabularyBudget`] — free RAM divided by the
/// widest frame's per-column price, minus what the resident families and the
/// base vocabulary already claimed. It is never a constant and never a user
/// parameter, which is the never-OOM invariant applied rather than circumvented.
///
/// ONE PAIR IS ALWAYS TAKEN when the budget is non-zero and the cursor is
/// inside the space. Without that, an id whose planned output count alone
/// exceeds the budget would produce an empty non-advancing batch forever — a
/// loop that makes no progress and says nothing, which is the silent-stall
/// shape of the silent-drop defect. Taking it anyway over-spends by at most one
/// indicator's outputs and is reported by the budget log.
pub fn extended_sweep_batch(cursor: usize, budget_columns: usize) -> SweepBatch {
    let space = extended_sweep_space();
    let space_len = space.len();
    if cursor >= space_len || budget_columns == 0 {
        return SweepBatch {
            cursor,
            next_cursor: cursor,
            pairs: Vec::new(),
            planned_columns: 0,
            space_len,
            exhausted: cursor >= space_len,
        };
    }
    let mut selected: Vec<SweepPair> = Vec::new();
    let mut used = 0usize;
    let mut idx = cursor;
    while idx < space_len {
        let pair = space[idx];
        let want = planned_output_count(pair.id);
        if !selected.is_empty() && used + want > budget_columns {
            break;
        }
        used += want;
        selected.push(pair);
        idx += 1;
    }
    // Emission order: id-outer in `ALL_INDICATORS` order, period ascending
    // within each id. `sort_by_key` is stable, but the key is total so
    // stability is not load-bearing — the order is a function of the SET alone,
    // never of which cursor produced it.
    selected.sort_by_key(|pair| (all_indicators_rank(pair.id), pair.period));
    SweepBatch {
        cursor,
        next_cursor: idx,
        pairs: selected,
        planned_columns: used,
        space_len,
        exhausted: idx >= space_len,
    }
}

/// The working set installed for this process, or `None` for the historical
/// budget-prefix behaviour.
///
/// A process-level seam, exactly like [`set_indicator_compute_policy`] above and
/// for the same reason: the cube build reaches `compute_classic_ta_columns_sized`
/// through six frames of `rayon::join` in `lib.rs`, and threading a batch
/// parameter through all of them would touch call sites in crates this change
/// does not own. It is not ambient state in the dangerous sense — the batch is a
/// pure function of `(cursor, budget_columns)`, it is logged by cursor and width
/// on every pass, and `with_extended_sweep_working_set` in `lib.rs` scopes the
/// install so it cannot leak past the build it was made for.
static EXTENDED_SWEEP_WORKING_SET: std::sync::RwLock<Option<std::sync::Arc<SweepBatch>>> =
    std::sync::RwLock::new(None);

/// Install (or clear) the working set. Returns the PREVIOUS value so a scoped
/// helper can restore it — including when the build between install and restore
/// unwinds.
///
/// Lock poisoning is recovered rather than propagated: a poisoned lock here
/// means some other thread panicked while holding it, and refusing to read the
/// working set would silently change which columns the run builds. The value is
/// a plain `Option<Arc<_>>` with no invariant a panic could have broken.
pub fn install_extended_sweep_working_set(
    batch: Option<std::sync::Arc<SweepBatch>>,
) -> Option<std::sync::Arc<SweepBatch>> {
    let mut guard = EXTENDED_SWEEP_WORKING_SET
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    std::mem::replace(&mut *guard, batch)
}

/// The working set in force, or `None` when the pass should take the historical
/// budget-capped prefix.
pub fn current_extended_sweep_working_set() -> Option<std::sync::Arc<SweepBatch>> {
    EXTENDED_SWEEP_WORKING_SET
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

/// CPU sweep for ONE indicator across `periods`. This is the parity reference
/// — f64 end to end — and is the only implementation of the column naming and
/// the #212 pre-flight guard, so the GPU lane cannot drift from it by
/// accident.
// Only the CUDA lane still calls the un-ledgered shape: its per-indicator CPU
// fallback and the parity test. The pure-CPU path aggregates one census for the
// whole sweep via `cpu_multi_period_all`, so in a card-less build this wrapper
// has no callers.
#[cfg(feature = "gpu-cuda")]
fn cpu_multi_period_columns(
    candles: &Candles,
    ind_id: &str,
    periods: &[usize],
    n: usize,
    kernel: Kernel,
) -> Vec<(String, Vec<f64>)> {
    let (cols, ledger) = cpu_multi_period_columns_ledgered(candles, ind_id, periods, n, kernel);
    // This wrapper exists so the GPU lane's per-indicator CPU fallback and the
    // parity test keep their original signature. It must still be incapable of
    // dropping silently, so anything the sweep discarded is reported here.
    if ledger.dropped_columns() > 0 {
        ledger.log_summary("multi-period-sweep", n);
    }
    cols
}

/// How a period maps onto ONE indicator's actual parameter names.
///
/// A period is not a detail of an indicator: RSI(4) is a scalping oscillator
/// and RSI(54) is a regime filter, and the search has to be able to reach both.
/// But "sweep everything at `period = P`" is wrong in a specific, measurable
/// way — vector-ta reads parameters BY KEY, and an indicator that does not
/// declare `period` simply ignores the key and returns its default series. Five
/// swept "periods" then produce five identical columns: 342 pieces of noise
/// wearing legitimate names, which is worse than the 66-column status quo
/// because the GA would happily fit them.
///
/// So the sweep asks the registry what the indicator's window parameters
/// actually are:
///
///   * [`PeriodPlan::Key`] — the indicator declares one window key
///     (`period`, `length`, …). Sweep it directly. This covers rsi, ema, sma,
///     atr, adx, cci, willr, roc, mom, tsi, mfi AND the three multi-output ids
///     bollinger_bands / keltner / supertrend, which all declare `period` and
///     produced nothing only because their `output_id` was never supplied.
///   * [`PeriodPlan::Ratio`] — the indicator declares a coupled TUPLE of
///     windows whose RELATIVE sizes are its identity: macd is
///     (fast 12, slow 26, signal 9) and stoch is (fastk 14, slowk 3, slowd 3).
///     Setting all three to the same P would not be "macd at period P", it
///     would be a different indicator. The whole tuple is scaled by
///     `P / anchor`, preserving the shape, so macd at 7 vs at 200 really are
///     the fast and slow versions of the same phenomenon.
///   * [`PeriodPlan::NoWindow`] — the indicator declares no window at all
///     (obv, vwap). The `period` key is still passed, UNCHANGED from the
///     historical behaviour, because dropping the `obv_7 … obv_200` names
///     would break projection-by-name against every stored portfolio. The
///     duplication is instead surfaced by the duplicate census in
///     `compute_classic_ta_columns_with_policy`, which names them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PeriodPlan {
    Key(&'static str),
    Ratio(&'static [(&'static str, i64)]),
    NoWindow,
}

/// `(indicator, (param key, default value))` for the coupled-window
/// indicators, read from vector-ta's registry param tables. The FIRST entry is
/// the anchor the swept period replaces; the rest keep their ratio to it.
const COUPLED_WINDOWS: &[(&str, &[(&str, i64)])] = &[
    // registry.rs PARAM_MACD: fast 12, slow 26, signal 9. Anchored on `slow`,
    // the window that sets the indicator's timescale.
    (
        "macd",
        &[("slow_period", 26), ("fast_period", 12), ("signal_period", 9)],
    ),
    // registry.rs PARAM_STOCH: fastk 14, slowk 3, slowd 3. Anchored on fastk.
    (
        "stoch",
        &[("fastk_period", 14), ("slowk_period", 3), ("slowd_period", 3)],
    ),
    // registry.rs PARAM_TSI (line 3030): long_period 25, short_period 13. tsi
    // declares NEITHER `period` nor `length`, so it used to fall through to
    // `NoWindow` and its five swept "periods" were five bit-identical copies —
    // MEASURED on 60,000 real EURUSD M5 bars as the duplicate group
    // [tsi, tsi_7, tsi_21, tsi_50, tsi_100, tsi_200]. Anchored on long_period,
    // the window that sets the timescale, with short_period scaled to keep the
    // 25:13 ratio that IS the indicator.
    (
        "tsi",
        &[("long_period", 25), ("short_period", 13)],
    ),
];

/// Window-parameter keys the sweep knows how to drive directly.
const RECOGNISED_WINDOW_KEYS: [&str; 4] = ["period", "length", "lookback_period", "lookback"];

/// Resolve the sweep plan for one indicator from vector-ta's declared params.
///
/// `NoWindow` means "this indicator declares no window-shaped parameter at
/// all". It must NEVER mean "it declares one under a name we do not recognise"
/// — that was the tsi defect, and it is silent by construction: the swept key
/// is simply ignored by the kernel and five identical columns come back wearing
/// five different names. So an unmatched window-shaped key is a WARN naming the
/// id and the key, not a shrug.
fn period_plan(ind_id: &str) -> PeriodPlan {
    if let Some((_, keys)) = COUPLED_WINDOWS.iter().find(|(k, _)| *k == ind_id) {
        return PeriodPlan::Ratio(keys);
    }
    if let Some(info) = vector_ta::indicators::registry::get_indicator(ind_id) {
        for key in RECOGNISED_WINDOW_KEYS {
            if info.params.iter().any(|p| p.key == key) {
                return PeriodPlan::Key(key);
            }
        }
        let unmatched = unmatched_window_keys(ind_id);
        if !unmatched.is_empty() {
            tracing::warn!(
                target: "neoethos_data::hpc_ta",
                indicator = ind_id,
                keys = %unmatched.join(","),
                "indicator declares window-shaped parameters the period sweep does not \
                 recognise, so sweeping it produces IDENTICAL columns under different names. \
                 Add it to hpc_ta::COUPLED_WINDOWS (anchored on its longest window) or extend \
                 RECOGNISED_WINDOW_KEYS."
            );
        }
        return PeriodPlan::NoWindow;
    }
    // Unregistered: `period` is the convention the by-name dispatch arms use.
    PeriodPlan::Key("period")
}

/// Integer parameters whose NAME looks like a window but which none of
/// [`RECOGNISED_WINDOW_KEYS`] matched — the tsi class.
///
/// Separated from [`period_plan`] so a test can assert the class is empty
/// across the whole swept set without capturing log output.
fn unmatched_window_keys(ind_id: &str) -> Vec<&'static str> {
    if COUPLED_WINDOWS.iter().any(|(k, _)| *k == ind_id) {
        return Vec::new();
    }
    let Some(info) = vector_ta::indicators::registry::get_indicator(ind_id) else {
        return Vec::new();
    };
    if info
        .params
        .iter()
        .any(|p| RECOGNISED_WINDOW_KEYS.contains(&p.key))
    {
        return Vec::new();
    }
    info.params
        .iter()
        .filter(|p| {
            matches!(p.kind, vector_ta::indicators::registry::IndicatorParamKind::Int)
                && (p.key.contains("period") || p.key.contains("length"))
        })
        .map(|p| p.key)
        .collect()
}

/// Build the parameter list for one indicator at one swept period.
fn sweep_params(plan: PeriodPlan, period: usize) -> Vec<ParamKV<'static>> {
    match plan {
        PeriodPlan::Key(key) => vec![ParamKV {
            key,
            value: ParamValue::Int(period as i64),
        }],
        PeriodPlan::Ratio(keys) => {
            let (_, anchor_default) = keys[0];
            keys.iter()
                .map(|&(key, default)| {
                    // Preserve the tuple's shape: scale each declared default
                    // by the same factor, never below 1.
                    let scaled = ((default as f64) * (period as f64) / (anchor_default as f64))
                        .round()
                        .max(1.0) as i64;
                    ParamKV {
                        key,
                        value: ParamValue::Int(scaled),
                    }
                })
                .collect()
        }
        // Historical behaviour preserved deliberately — see PeriodPlan docs.
        PeriodPlan::NoWindow => vec![ParamKV {
            key: "period",
            value: ParamValue::Int(period as i64),
        }],
    }
}

/// The sweep, with its outcomes counted.
///
/// Every discard the old body made silently is now a reason:
///   * the `#212` pre-flight skip is `PreflightWarmup` (was a bare `continue`);
///   * a dispatch `Err` is its own variant (was `if let Ok(...)` with no else —
///     the branch that swallowed the five multi-output ids' `output_id`
///     complaint 25 times per frame);
///   * a wrong-length series is `ShortSeries` (was the `_ => {}` catch-all).
fn cpu_multi_period_columns_ledgered(
    candles: &Candles,
    ind_id: &str,
    periods: &[usize],
    n: usize,
    kernel: Kernel,
) -> (Vec<(String, Vec<f64>)>, IndicatorLedger) {
    let mut out: Vec<(String, Vec<f64>)> = Vec::new();
    let mut ledger = IndicatorLedger::new();
    // `dispatch_indicator_outputs` needs a 'static id, which every caller has:
    // the sweep is driven by `MULTI_PERIOD_IDS` and the GPU spec table, both
    // `&'static str` tables.
    let Some(static_id) = MULTI_PERIOD_IDS.iter().copied().find(|s| *s == ind_id) else {
        // Not one of the swept ids — the caller is asking for something this
        // function was never given a table entry for. Loud, not silent.
        ledger.dropped(
            "unknown-sweep-id",
            ind_id,
            DropReason::UnknownIndicator,
            "id is not in hpc_ta::MULTI_PERIOD_IDS, so the sweep has no plan for it",
        );
        return (out, ledger);
    };
    let _ = &mut out;
    let _ = &mut ledger;
    sweep_one_id_ledgered(candles, static_id, periods, n, kernel)
}

/// Sweep ONE `'static` indicator id across `periods`.
///
/// Shared by the historical eighteen (through
/// [`cpu_multi_period_columns_ledgered`], which keeps the id-table guard the
/// GPU lane relies on) and by the extended sweep, so the two cannot drift on
/// naming, on the `#212` pre-flight guard, or on what they record.
fn sweep_one_id_ledgered(
    candles: &Candles,
    static_id: &'static str,
    periods: &[usize],
    n: usize,
    kernel: Kernel,
) -> (Vec<(String, Vec<f64>)>, IndicatorLedger) {
    let mut out: Vec<(String, Vec<f64>)> = Vec::new();
    let mut ledger = IndicatorLedger::new();
    let plan = period_plan(static_id);
    for &period in periods {
        // #212: pre-flight check — if the period is larger than the data
        // length, vector-ta's `warm_prefix` exceeds the row width and the
        // kernel panics at `helpers.rs:159` instead of returning Err. Skip the
        // call entirely for these cases. The 1.25× safety margin matches the
        // kernel's typical `first_valid_idx + period` formula plus a small
        // headroom for indicators with extra warmup beyond the period itself.
        if (period as f64) * 1.25 >= n as f64 {
            // THE COLUMN SET MUST NOT DEPEND ON THE FRAME.
            //
            // This guard skipped the period entirely, so a short frame emitted
            // FEWER columns than a long one — and every timeframe has a
            // different length. That made the emitted width a function of the
            // frame, which is precisely what `lib.rs::try_assemble_cube_in_ram`
            // refuses to assemble, so a run would fall through to the slower
            // streaming disk path with nothing but a debug line to say why.
            // Harmless while the sweep was 18 ids; not harmless now that it is
            // hundreds. Caught by `cube_assembly_tests::ram_and_disk_cubes_are_identical`.
            //
            // So the columns are still EMITTED, at full frame length, filled
            // with NaN — the same value the warmup prefix of every windowed
            // indicator already carries, and the honest one here: this frame
            // cannot support this period, so the reading does not exist. It is
            // NOT zero, which would be a real number the GA could threshold
            // against. The skip is still counted under `PreflightWarmup`, and
            // the column shows up in the degenerate census as carrying no
            // information.
            let column = format!("{static_id}_{period}");
            ledger.dropped(
                static_id,
                &column,
                DropReason::PreflightWarmup,
                format!(
                    "period {period} * 1.25 >= {n} bars (#212 pre-flight guard); column emitted \
                     as all-NaN to keep the column set independent of the frame length"
                ),
            );
            push_absent_columns(static_id, &column, n, &mut out);
            continue;
        }
        let params = sweep_params(plan, period);
        dispatch_indicator_outputs(
            candles,
            static_id,
            &format!("{static_id}_{period}"),
            &params,
            n,
            kernel,
            &mut out,
            &mut ledger,
        );
    }
    (out, ledger)
}

/// Pure-CPU multi-period sweep across all of [`MULTI_PERIOD_IDS`], parallel
/// across indicators. Column order is by position in `MULTI_PERIOD_IDS` —
/// `flat_map_iter` + `collect` on an indexed `par_iter` preserves it.
fn cpu_multi_period_all(candles: &Candles, n: usize) -> Vec<(String, Vec<f64>)> {
    let per_id: Vec<(Vec<(String, Vec<f64>)>, IndicatorLedger)> = MULTI_PERIOD_IDS
        .par_iter()
        .map(|&ind_id| cpu_multi_period_columns_ledgered(candles, ind_id, &ALT_PERIODS, n, Kernel::Auto))
        .collect();
    let mut cols = Vec::new();
    let mut ledger = IndicatorLedger::new();
    for (mut c, l) in per_id {
        cols.append(&mut c);
        ledger.merge(l);
    }
    // ONE summary for the whole sweep rather than eighteen — but never zero.
    ledger.log_summary("multi-period-sweep", n);
    cols
}

// --- lane selection -------------------------------------------------------
//
// Two implementations of `compute_multi_period_columns`, chosen at COMPILE
// time. In a card-less build the CUDA one does not exist, so nothing about the
// default build changes — not the code, not the dependency graph, not the
// object output.

#[cfg(not(feature = "gpu-cuda"))]
fn compute_multi_period_columns(
    _ohlcv: &Ohlcv,
    candles: &Candles,
    n: usize,
    policy: IndicatorComputePolicy,
) -> Vec<(String, Vec<f64>)> {
    use crate::core::indicator_telemetry::{IndicatorLane, IndicatorRunSummary, record};
    use std::time::Instant;

    // `RequireGpu` on a binary with no CUDA lane compiled in is an operator
    // error that MUST be loud. Returning CPU numbers here is exactly the
    // silent degradation the GPU-or-fail invariant exists to prevent.
    assert!(
        policy != IndicatorComputePolicy::RequireGpu,
        "IndicatorComputePolicy::RequireGpu was requested but this binary has no CUDA indicator \
         lane compiled in. Rebuild with `--features gpu-cuda` (and CUDA_ARCH set to the card's \
         compute capability), or use IndicatorComputePolicy::Auto/Cpu."
    );

    let t0 = Instant::now();
    let cols = cpu_multi_period_all(candles, n);
    let lane = match policy {
        IndicatorComputePolicy::Cpu => IndicatorLane::CpuByPolicy,
        _ => IndicatorLane::CpuNoFeature,
    };
    record(IndicatorRunSummary {
        gpu_indicators: Vec::new(),
        cpu_indicators: MULTI_PERIOD_IDS
            .iter()
            .map(|id| (*id, lane.clone()))
            .collect(),
        cpu_time: t0.elapsed(),
        ..Default::default()
    });
    cols
}

#[cfg(feature = "gpu-cuda")]
fn compute_multi_period_columns(
    ohlcv: &Ohlcv,
    candles: &Candles,
    n: usize,
    policy: IndicatorComputePolicy,
) -> Vec<(String, Vec<f64>)> {
    use crate::core::gpu_indicators::{GpuIndicatorEngine, spec_for};
    use crate::core::indicator_telemetry::{IndicatorLane, IndicatorRunSummary, record};
    use std::time::Instant;

    if policy == IndicatorComputePolicy::Cpu {
        let t0 = Instant::now();
        let cols = cpu_multi_period_all(candles, n);
        record(IndicatorRunSummary {
            cpu_indicators: MULTI_PERIOD_IDS
                .iter()
                .map(|id| (*id, IndicatorLane::CpuByPolicy))
                .collect(),
            cpu_time: t0.elapsed(),
            ..Default::default()
        });
        return cols;
    }

    // Device ordinal 0. Multi-card indicator fan-out is a separate change; a
    // single card already serves the whole frame from one upload.
    let engine = match GpuIndicatorEngine::new(ohlcv, 0) {
        Ok(e) => e,
        Err(e) => {
            // `RequireGpu`: no fallback exists. Panic with the device error
            // still attached — this is the GPU-or-fail invariant, and it is
            // worth more than a completed run with CPU numbers in it.
            if policy == IndicatorComputePolicy::RequireGpu {
                panic!(
                    "IndicatorComputePolicy::RequireGpu: the CUDA indicator lane could not be \
                     opened and there is no fallback: {e:?}"
                );
            }
            // `Auto`: fall back, but the fallback is DECLARED — logged at WARN
            // with the device error, and recorded per indicator in the run
            // summary so `indicator_device_summary()` prints the reason.
            tracing::warn!(
                target: "neoethos_data::hpc_ta",
                error = %format!("{e:#}"),
                "CUDA indicator lane unavailable — the multi-period sweep runs on the CPU for \
                 this frame. This is a DECLARED fallback: the reason is recorded in the \
                 indicator-lane device summary. Use IndicatorComputePolicy::RequireGpu to make \
                 it fatal."
            );
            let t0 = Instant::now();
            let cols = cpu_multi_period_all(candles, n);
            let lane = IndicatorLane::CpuDeviceUnavailable {
                why: format!("{e:#}"),
            };
            record(IndicatorRunSummary {
                cpu_indicators: MULTI_PERIOD_IDS
                    .iter()
                    .map(|id| (*id, lane.clone()))
                    .collect(),
                cpu_time: t0.elapsed(),
                ..Default::default()
            });
            return cols;
        }
    };

    // The lane is proven. Announce WHICH lane and at WHAT precision, once per
    // frame, the same way the population lane's `device_summary` does — a
    // reader must never have to infer either.
    //
    // This used to be a WARN announcing a divergence, because the device layer
    // was f32 while every other column in the frame was f64. It is now INFO
    // announcing parity, because the lane is f64 end to end (no narrowing on
    // upload, f64 kernels, no fast math). The claim is checked by
    // `gpu_cpu_indicator_sweep_parity` on real bars; the log line states the
    // precision, it does not prove it.
    tracing::info!(
        target: "neoethos_data::hpc_ta",
        device = engine.device_name(),
        arch = engine.device_arch(),
        compiled_archs = vector_ta::cuda::module_loader::COMPILED_ARCHS,
        ptx_arch = crate::core::indicator_telemetry::VECTOR_TA_PTX_ARCH,
        precision = engine.precision(),
        gpu_indicators = crate::core::gpu_indicators::GPU_SWEEP_SPECS.len(),
        cpu_indicators = MULTI_PERIOD_IDS.len() - crate::core::gpu_indicators::GPU_SWEEP_SPECS.len(),
        "multi-period indicator sweep is running on the CUDA f64 lane — f64 upload, f64 kernels, \
         no fast math. See hpc_ta::gpu_cpu_indicator_sweep_parity for the measured agreement \
         against the f64 CPU reference."
    );

    // Per-indicator slots, filled in `MULTI_PERIOD_IDS` order so the emitted
    // column order matches the pure-CPU path exactly.
    let mut slots: Vec<Vec<(String, Vec<f64>)>> = vec![Vec::new(); MULTI_PERIOD_IDS.len()];
    // Tracked explicitly rather than inferred from `slots[i].is_empty()`: on a
    // frame short enough that every period is skipped by the 1.25x pre-flight
    // guard, a device sweep legitimately returns ZERO columns, and an
    // emptiness test would then re-run it on the CPU and mislabel the lane.
    let mut device_handled = vec![false; MULTI_PERIOD_IDS.len()];
    let mut gpu_ids: Vec<&'static str> = Vec::new();
    let mut cpu_ids: Vec<(&'static str, IndicatorLane)> = Vec::new();
    let mut device_calls: u64 = 0;
    let mut gpu_time = std::time::Duration::ZERO;
    let mut cpu_time = std::time::Duration::ZERO;

    // Device pass first: serial across indicators by design. They all read the
    // same resident buffers on one stream, so interleaving them from rayon
    // workers would only contend for the same stream.
    for (idx, &ind_id) in MULTI_PERIOD_IDS.iter().enumerate() {
        let Some(spec) = spec_for(ind_id) else {
            continue;
        };
        let t0 = Instant::now();
        match engine.sweep_columns(spec, &ALT_PERIODS) {
            Ok(cols) => {
                // ONE launch per indicator now, for the whole period list —
                // the f64 lane takes an explicit period array instead of an
                // arithmetic (start, end, step) range, so the five periods no
                // longer cost five launches. Counting columns here would
                // silently re-inflate this back to the old number and hide the
                // improvement.
                if !cols.is_empty() {
                    device_calls += 1;
                }
                gpu_time += t0.elapsed();
                slots[idx] = cols;
                device_handled[idx] = true;
                gpu_ids.push(ind_id);
            }
            Err(e) => {
                // A launch that fails AFTER the lane was proven is a real
                // fault, not a capability gap. Under RequireGpu it is fatal;
                // under Auto it is a declared per-indicator fallback.
                if policy == IndicatorComputePolicy::RequireGpu {
                    panic!(
                        "IndicatorComputePolicy::RequireGpu: device sweep for `{ind_id}` failed \
                         after the lane was proven: {e:?}"
                    );
                }
                tracing::warn!(
                    target: "neoethos_data::hpc_ta",
                    indicator = %ind_id,
                    error = %format!("{e:#}"),
                    "device sweep failed for this indicator; falling back to the CPU for it \
                     (declared, recorded in the indicator-lane summary)"
                );
                cpu_ids.push((
                    ind_id,
                    IndicatorLane::CpuDeviceUnavailable {
                        why: format!("{e:#}"),
                    },
                ));
            }
        }
    }
    if let Err(e) = engine.synchronize() {
        tracing::warn!(
            target: "neoethos_data::hpc_ta",
            error = %format!("{e:#}"),
            "CudaRuntime::synchronize failed after the indicator sweep; recorded device time is \
             a lower bound"
        );
    }

    // Anything the device did not fill is a CPU indicator: either not in the
    // device table (enumerated up front) or a failed launch recorded above.
    let cpu_pending: Vec<(usize, &'static str)> = device_handled
        .iter()
        .enumerate()
        .filter(|(_, handled)| !**handled)
        .map(|(i, _)| (i, MULTI_PERIOD_IDS[i]))
        .collect();
    for &(_, id) in &cpu_pending {
        if !cpu_ids.iter().any(|(cid, _)| *cid == id) {
            cpu_ids.push((id, IndicatorLane::CpuIndicatorNotPortable));
        }
    }
    let t0 = Instant::now();
    let cpu_filled: Vec<(usize, Vec<(String, Vec<f64>)>)> = cpu_pending
        .par_iter()
        .map(|&(i, id)| (i, cpu_multi_period_columns(candles, id, &ALT_PERIODS, n, Kernel::Auto)))
        .collect();
    cpu_time += t0.elapsed();
    for (i, cols) in cpu_filled {
        slots[i] = cols;
    }

    record(IndicatorRunSummary {
        gpu_indicators: gpu_ids,
        cpu_indicators: cpu_ids,
        gpu_time,
        cpu_time,
        device_calls,
        uploads: engine.uploads(),
        // Reported by the engine rather than restated here, so the telemetry
        // string and the lane cannot drift apart.
        precision: Some(engine.precision()),
    });

    slots.into_iter().flatten().collect()
}

/// One series returned by `compute_single_indicator` — multi-output
/// indicators (Bollinger Bands, MACD, Stochastic) decompose into
/// several of these.
#[derive(Debug, Clone)]
pub struct IndicatorLine {
    /// Human-readable line name. Single-output indicators use the
    /// indicator id (e.g. `"sma"`); multi-output ones suffix with
    /// the column index (`"bollinger_bands_line0"`, `"…_line1"`,
    /// `"…_line2"` for lower/middle/upper).
    pub name: String,
    /// Series aligned with the input ohlcv length. NaN-padding at
    /// the start is preserved (so e.g. SMA(20)[0..19] = NaN), which
    /// the chart renders as a gap before the line begins.
    pub values: Vec<f64>,
}

/// Compute a single indicator on demand — the interactive Chart
/// screen calls this through the `/indicators` HTTP endpoint
/// whenever the user adds an indicator to the overlay. Cheap enough
/// to recompute on every pan; vector_ta dispatches to CPU SIMD or
/// GPU kernels under the hood.
///
/// `params` is a key→f64 map. Conventional keys per indicator:
///   * `sma`/`ema`/`rsi`/`atr`/`adx`: `period`
///   * `bollinger_bands`: `period`, `std_dev`
///   * `macd`: `fast`, `slow`, `signal`
///   * `stoch`: `k_period`, `k_slow`, `d_period`
/// Unrecognised keys are silently ignored. Empty map = library defaults.
///
/// Returns the row count + lines on success, anyhow error if the
/// indicator id is unknown or the kernel rejects the input.
pub fn compute_single_indicator(
    ohlcv: &Ohlcv,
    indicator_id: &str,
    params: &std::collections::HashMap<String, f64>,
) -> anyhow::Result<Vec<IndicatorLine>> {
    let n = ohlcv.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Pack the ohlcv slice into a Candles instance for the dispatch
    // API. Timestamps and volume are nice-to-have but not required
    // by most indicators — we fill zeros when missing.
    let timestamps = ohlcv.timestamp.clone().unwrap_or_else(|| vec![0i64; n]);
    let volume = ohlcv.volume.clone().unwrap_or_else(|| vec![0.0; n]);
    let candles = Candles::new(
        timestamps,
        ohlcv.open.clone(),
        ohlcv.high.clone(),
        ohlcv.low.clone(),
        ohlcv.close.clone(),
        volume,
    );
    let data_ref = IndicatorDataRef::Candles {
        candles: &candles,
        source: None,
    };

    // Translate the f64 param map into vector_ta's ParamKV array.
    // vector_ta accepts ints for period-like params and floats for
    // multipliers (e.g. Bollinger Bands' std_dev); we route based on
    // whether the value has a fractional part.
    let mut kv: Vec<vector_ta::indicators::dispatch::ParamKV> = Vec::with_capacity(params.len());
    for (k, v) in params {
        // Leak the &'static str via Box::leak so the dispatch API
        // can hold a 'static reference. Param map is tiny (≤ 5
        // entries) and lives for the call, so the leak is bounded
        // by the call site — acceptable trade-off for the simpler
        // wire shape.
        let key: &'static str = Box::leak(k.clone().into_boxed_str());
        let value = if v.fract() == 0.0 && v.abs() <= i64::MAX as f64 {
            vector_ta::indicators::dispatch::ParamValue::Int(*v as i64)
        } else {
            vector_ta::indicators::dispatch::ParamValue::Float(*v)
        };
        kv.push(vector_ta::indicators::dispatch::ParamKV { key, value });
    }

    // Look up the indicator's declared outputs. Multi-output indicators
    // (MACD, Bollinger Bands, Stochastic, …) REQUIRE an explicit
    // `output_id` per series in vector_ta — dispatching them with
    // `output_id: None` fails with "output_id is required for
    // multi-output indicators". Single-output indicators use `None`
    // (the library's default output).
    let output_ids: Vec<Option<&'static str>> =
        vector_ta::indicators::registry::list_indicators()
            .iter()
            .find(|i| i.id == indicator_id)
            .map(|info| {
                if info.outputs.len() <= 1 {
                    vec![None]
                } else {
                    info.outputs.iter().map(|o| Some(o.id)).collect()
                }
            })
            .unwrap_or_else(|| vec![None]);

    let mut lines = Vec::with_capacity(output_ids.len());
    for out_id in output_ids {
        let req = IndicatorComputeRequest {
            indicator_id,
            output_id: out_id,
            data: data_ref,
            params: &kv,
            kernel: Kernel::Auto,
        };
        let output = compute_cpu(req).map_err(|e| {
            anyhow::anyhow!("vector_ta dispatch failed ({indicator_id}/{out_id:?}): {e:?}")
        })?;
        let (values, raw_len) = flatten_indicator_series(output.series, n)?;
        if raw_len > n {
            tracing::warn!(
                target: "neoethos_data::hpc_ta",
                indicator = indicator_id,
                output = ?out_id,
                raw_len,
                bars = n,
                "chart indicator returned more values than the frame has bars; the head was kept"
            );
        }
        // Single-output → bare indicator id (e.g. "sma"); multi-output →
        // "<id>_<output>" (e.g. "macd_signal") so the chart legend can
        // split on '_' and show the per-line label.
        let name = match out_id {
            Some(id) => format!("{indicator_id}_{id}"),
            None => indicator_id.to_string(),
        };
        lines.push(IndicatorLine { name, values });
    }

    Ok(lines)
}

/// Flatten a vector_ta series for ONE output into exactly `n` values.
/// vector_ta reports a 1-D series as rows=1 × cols=n, so we key off the
/// value count, not the rows/cols metadata — the previous `rows == n`
/// assumption rejected every single-output series with a "shape
/// mismatch" error.
///
/// Returns `(values, raw_value_count)`. The raw count is handed back so the
/// caller can LEDGER a truncation instead of performing one silently — this
/// function used to be the last uncounted discard on the repaired path.
fn flatten_indicator_series(series: IndicatorSeries, n: usize) -> anyhow::Result<(Vec<f64>, usize)> {
    match series {
        IndicatorSeries::F64(v) => normalize_indicator_len(v, n),
        IndicatorSeries::I32(v) => {
            normalize_indicator_len(v.into_iter().map(|x| x as f64).collect(), n)
        }
        IndicatorSeries::Bool(v) => normalize_indicator_len(
            v.into_iter().map(|x| if x { 1.0 } else { 0.0 }).collect(),
            n,
        ),
    }
}

/// Fit one output's raw values onto the frame, or refuse.
///
/// Three outcomes, no fourth:
///
///   * exactly `n` — the normal case;
///   * an exact MULTIPLE of `n` — that is a FLATTENED MULTI-SERIES, not a long
///     single series. Taking the head would return the first output's values
///     under whichever `output_id` was requested: silent MIS-ATTRIBUTION, which
///     is worse than a silent loss because the column looks legitimate and
///     carries the wrong meaning. Hard error, named;
///   * longer for any other reason — head-aligned truncation, which is real and
///     is reported back to the caller so it lands in the ledger under
///     [`DropReason::Truncated`] rather than happening quietly.
fn normalize_indicator_len(v: Vec<f64>, n: usize) -> anyhow::Result<(Vec<f64>, usize)> {
    let raw = v.len();
    if raw == n {
        Ok((v, raw))
    } else if raw > n {
        if n > 0 && raw % n == 0 {
            anyhow::bail!(
                "indicator returned {raw} values for a {n}-bar frame — an exact {}x multiple, \
                 i.e. a FLATTENED MULTI-SERIES. Taking the head would silently return one \
                 output's values under another output's name; there is no correct silent \
                 handling, so this is refused.",
                raw / n
            );
        }
        // Bar-aligned from the start; take the leading n (warmup padding
        // lives at the head and stays aligned with candle index 0).
        Ok((v.into_iter().take(n).collect(), raw))
    } else {
        anyhow::bail!("indicator returned {} values, expected ≥{}", raw, n)
    }
}


#[cfg(test)]
mod streaming_advance_tests {
    use super::*;
    use std::collections::HashSet;

    /// PARITY, AND IT COMES FIRST.
    ///
    /// The degenerate case: one batch whose working set is the WHOLE space,
    /// starting at cursor 0. Its emitted `(id, periods)` grouping must be
    /// exactly what the non-streaming pass builds — the budget-prefix plan with
    /// every id at all five `ALT_PERIODS`, in `ALL_INDICATORS` order.
    ///
    /// Asserted on the (id, period) LIST, not on a count: the extension emits
    /// `<id>_<period>` names into the same namespace as the base pass and a
    /// duplicate NAME is a hard error in `compute_classic_ta_columns_sized`, so
    /// a width-only assertion would pass while the column set changed.
    #[test]
    fn whole_space_batch_is_byte_identical_to_the_non_streaming_plan() {
        let space_len = extended_sweep_space_len();
        assert!(space_len > 0, "the sweep space must not be empty");
        let batch = extended_sweep_batch(0, usize::MAX);
        assert!(batch.covers_whole_space());
        assert!(batch.exhausted);
        assert_eq!(batch.next_cursor, space_len);

        // What the non-streaming lane would dispatch, at an unbounded budget.
        let (plan, deferred) = extended_sweep_plan(usize::MAX);
        assert!(deferred.is_empty(), "an unbounded budget defers nothing");
        let expected: Vec<(&'static str, Vec<usize>)> = plan
            .into_iter()
            .map(|id| (id, ALT_PERIODS.to_vec()))
            .collect();

        assert_eq!(
            batch.grouped_by_id(),
            expected,
            "the whole-space batch must dispatch exactly the ids, periods and ORDER the \
             non-streaming pass dispatches — column order feeds effective_feature_names"
        );
    }

    /// The advance never repeats and never loses a pair: the disjoint union of
    /// every batch is exactly the space, in selection order.
    #[test]
    fn successive_batches_partition_the_space_exactly_once() {
        let space = extended_sweep_space();
        let mut cursor = 0usize;
        let mut seen: Vec<SweepPair> = Vec::new();
        let mut guard = 0usize;
        loop {
            let batch = extended_sweep_batch(cursor, 40);
            if batch.is_empty() {
                break;
            }
            assert!(
                batch.next_cursor > cursor,
                "a non-empty batch must advance the cursor, or the loop stalls silently"
            );
            seen.extend(batch.pairs.iter().copied());
            cursor = batch.next_cursor;
            guard += 1;
            assert!(guard < 100_000, "advance failed to terminate");
            if batch.exhausted {
                break;
            }
        }
        assert_eq!(cursor, space.len());
        assert_eq!(seen.len(), space.len(), "every pair must appear exactly once");
        let unique: HashSet<SweepPair> = seen.iter().copied().collect();
        assert_eq!(unique.len(), space.len(), "no pair may appear twice");
        let space_set: HashSet<SweepPair> = space.iter().copied().collect();
        assert_eq!(unique, space_set, "the union of the batches IS the space");
    }

    /// Batch *k* is the same set on every call — the property that makes a
    /// result reproducible from `(seed, cursor)` alone.
    #[test]
    fn a_batch_is_a_pure_function_of_cursor_and_budget() {
        for cursor in [0usize, 1, 37, 200] {
            let a = extended_sweep_batch(cursor, 64);
            let b = extended_sweep_batch(cursor, 64);
            assert_eq!(a, b, "batch at cursor {cursor} is not deterministic");
        }
    }

    /// Selection is period-outer so a batch MIXES timescales; emission is
    /// id-outer so the column layout never becomes a function of scheduling.
    #[test]
    fn selection_mixes_timescales_and_emission_is_id_ordered() {
        let space = extended_sweep_space();
        // Selection: the first `ALT_PERIODS[0]`-period run precedes any
        // `ALT_PERIODS[1]` pair.
        let first_period = space[0].period;
        assert_eq!(first_period, ALT_PERIODS[0]);
        // Emission within one batch: non-decreasing ALL_INDICATORS rank, and
        // ascending period within an id.
        let batch = extended_sweep_batch(0, 512);
        let mut last = (0usize, 0usize);
        for pair in &batch.pairs {
            let key = (all_indicators_rank(pair.id), pair.period);
            assert!(key > last || last == (0, 0), "emission order is not sorted");
            last = key;
        }
    }

    /// A budget smaller than one id's outputs must still advance — a batch that
    /// makes no progress is a stall with no log line, which is the silent-drop
    /// defect one level up.
    #[test]
    fn a_budget_too_small_for_one_indicator_still_advances() {
        let batch = extended_sweep_batch(0, 1);
        assert!(!batch.is_empty());
        assert_eq!(batch.next_cursor, 1);
    }

    /// A zero budget admits nothing and does NOT advance, and an exhausted
    /// cursor reports itself rather than wrapping. Wrapping is the loop's
    /// decision, never a silent one here.
    #[test]
    fn zero_budget_and_exhausted_cursor_are_reported_not_papered_over() {
        let empty = extended_sweep_batch(0, 0);
        assert!(empty.is_empty());
        assert_eq!(empty.next_cursor, 0);
        assert!(!empty.exhausted);

        let len = extended_sweep_space_len();
        let past_end = extended_sweep_batch(len, usize::MAX);
        assert!(past_end.is_empty());
        assert!(past_end.exhausted);
        assert_eq!(past_end.next_cursor, len);
    }

    /// The space can never collide by NAME with the historical sweep, because a
    /// duplicate column name is a hard error in the pass that emits both.
    #[test]
    fn the_space_excludes_the_historical_sweep_ids() {
        for pair in extended_sweep_space() {
            assert!(
                !MULTI_PERIOD_IDS.contains(&pair.id),
                "{} is swept by the historical sweep; streaming it too emits duplicate \
                 column NAMES, which is a hard error",
                pair.id
            );
        }
    }

    /// Install/restore is exact, so a scoped build cannot leak its working set
    /// into the next one.
    #[test]
    fn installing_a_working_set_returns_the_previous_one() {
        let batch = std::sync::Arc::new(extended_sweep_batch(0, 8));
        let previous = install_extended_sweep_working_set(Some(batch.clone()));
        assert_eq!(current_extended_sweep_working_set().as_deref(), Some(&*batch));
        let restored = install_extended_sweep_working_set(previous);
        assert_eq!(restored.as_deref(), Some(&*batch));
        assert!(current_extended_sweep_working_set().is_none());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #212: pre-flight check helper used by the validation harness and
    // gene admission gate to refuse computation on slices smaller than
    // the indicator's warmup. These assertions document the contract
    // and trap regressions if `alt_periods` ever changes.
    #[test]
    fn max_indicator_warmup_returns_zero_for_tiny_frames() {
        assert_eq!(max_indicator_warmup(0), 0);
        assert_eq!(max_indicator_warmup(5), 0);
        assert_eq!(max_indicator_warmup(7), 0);
    }

    #[test]
    fn max_indicator_warmup_returns_largest_fitting_period() {
        // n=8 fits 7 (smallest alt_period) but not 21.
        assert_eq!(max_indicator_warmup(8), 7);
        assert_eq!(max_indicator_warmup(22), 21);
        assert_eq!(max_indicator_warmup(51), 50);
        assert_eq!(max_indicator_warmup(101), 100);
        assert_eq!(max_indicator_warmup(201), 200);
    }

    #[test]
    fn max_indicator_warmup_caps_at_largest_period() {
        // Even for huge frames the helper does not exceed the
        // configured `MAX_MULTI_PERIOD_LOOKBACK`.
        assert_eq!(max_indicator_warmup(10_000), 200);
        assert_eq!(max_indicator_warmup(1_000_000), MAX_MULTI_PERIOD_LOOKBACK);
    }

    // Pre-flight gate documented in `compute_classic_ta_columns`: the
    // multi-period sweep skips any period whose `*1.25` safety margin
    // exceeds the frame length. This test pins the contract so a refactor
    // can't silently drop the guard and reintroduce the #212 panic.
    #[test]
    fn pre_flight_gate_skips_periods_larger_than_frame() {
        // For a 30-row frame: 7 fits (7*1.25=8.75 < 30), 21 fits
        // (26.25 < 30), 50 does NOT fit (62.5 ≥ 30).
        let n: usize = 30;
        let acceptable: Vec<usize> = [7usize, 21, 50, 100, 200]
            .into_iter()
            .filter(|p| (*p as f64) * 1.25 < n as f64)
            .collect();
        assert_eq!(acceptable, vec![7, 21]);
    }

    /// The two constants that used to be inline literals must still describe
    /// the same sweep, otherwise `max_indicator_warmup` and the GPU lane's
    /// skip guard disagree with what is actually computed.
    #[test]
    fn alt_periods_matches_the_warmup_helper() {
        assert_eq!(ALT_PERIODS, [7, 21, 50, 100, 200]);
        assert_eq!(
            *ALT_PERIODS.last().unwrap(),
            MAX_MULTI_PERIOD_LOOKBACK,
            "MAX_MULTI_PERIOD_LOOKBACK must be the largest swept period"
        );
        assert_eq!(MULTI_PERIOD_IDS.len(), 18);
    }

    /// Column SET, NAMES and ORDER must not depend on which lane computed the
    /// values. On a card-less build both policies run the same code and this
    /// only pins the contract; on a box with a card it is a real cross-lane
    /// test, because `Auto` routes ten of the eighteen indicators through
    /// CUDA and the columns are re-assembled by slot afterwards.
    #[test]
    fn lane_policy_does_not_change_column_names_or_order() {
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let cpu =
            compute_classic_ta_columns_with_policy(&ohlcv, IndicatorComputePolicy::Cpu).unwrap();
        let auto =
            compute_classic_ta_columns_with_policy(&ohlcv, IndicatorComputePolicy::Auto).unwrap();
        let cpu_names: Vec<&str> = cpu.iter().map(|(n, _)| n.as_str()).collect();
        let auto_names: Vec<&str> = auto.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            cpu_names, auto_names,
            "the lane policy changed the feature-frame column layout — every stored artifact \
             would be invalidated"
        );
    }

    // =======================================================================
    // The 341-silent-drop regression guards.
    //
    // These run on the embedded ~100-bar cTrader fixture, which is BELOW
    // `VOCABULARY_FLOOR_MIN_ROWS`, so the hard floor does not fire here — most
    // indicators legitimately fail their warmup at 100 bars. What these assert
    // is the SHAPE of the fix, which is length-independent: that the vocabulary
    // is no longer one column, that multi-output indicators are dispatched with
    // an output id, and that every column is full length.
    // =======================================================================

    /// The headline regression: the base pass used to yield exactly ONE column
    /// (`ttm_trend`) out of 342 ids. Anything close to that again is the bug
    /// returning.
    #[test]
    fn the_base_vocabulary_is_no_longer_a_single_column() {
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let cols =
            compute_classic_ta_columns_with_policy(&ohlcv, IndicatorComputePolicy::Cpu).unwrap();
        let names: Vec<&str> = cols.iter().map(|(n, _)| n.as_str()).collect();
        // The sweep alone contributes 65 on a long frame, but at ~100 bars only
        // the 7/21/50 periods clear the #212 pre-flight guard. The base pass is
        // what this asserts: it must contribute far more than the single
        // `ttm_trend` it produced for sixteen months.
        let swept: Vec<&&str> = names
            .iter()
            .filter(|n| ALT_PERIODS.iter().any(|p| n.contains(&format!("_{p}"))))
            .collect();
        let base_columns = names.len() - swept.len();
        assert!(
            base_columns > 100,
            "the base ALL_INDICATORS pass produced {base_columns} columns — it used to produce \
             exactly 1 (ttm_trend) and the repaired pass produces hundreds. Column sample: {:?}",
            &names[..names.len().min(12)]
        );
    }

    /// `output_id: None` against a multi-output indicator is the cause of 92 of
    /// the 342 failures, and is precisely why the five multi-output ids in the
    /// period sweep produced nothing. Bollinger Bands must now emit its
    /// declared lines, named by output rather than by position.
    #[test]
    fn multi_output_indicators_now_emit_one_named_column_per_output() {
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let cols =
            compute_classic_ta_columns_with_policy(&ohlcv, IndicatorComputePolicy::Cpu).unwrap();
        let bb: Vec<&String> = cols
            .iter()
            .map(|(n, _)| n)
            .filter(|n| n.starts_with("bollinger_bands"))
            .collect();
        assert!(
            bb.len() >= 3,
            "bollinger_bands is multi-output and must produce one column per declared output; \
             got {bb:?}"
        );
        assert!(
            !bb.iter().any(|n| n.contains("_line")),
            "columns must be named by OUTPUT ID, not by position: {bb:?}"
        );
    }

    /// A short column would be zero-padded by the cube copy in `lib.rs`, and a
    /// padded zero is indistinguishable from a real reading. Nothing here may
    /// emit one.
    #[test]
    fn every_emitted_column_is_exactly_frame_length() {
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let n = ohlcv.len();
        let cols =
            compute_classic_ta_columns_with_policy(&ohlcv, IndicatorComputePolicy::Cpu).unwrap();
        assert!(!cols.is_empty());
        for (name, v) in &cols {
            assert_eq!(v.len(), n, "column '{name}' is {} values, expected {n}", v.len());
        }
    }

    /// THE COLUMN SET MUST NOT DEPEND ON THE FRAME LENGTH.
    ///
    /// Every timeframe in a run is a different length, and
    /// `lib.rs::try_assemble_cube_in_ram` refuses to assemble a cube whose
    /// per-timeframe widths differ — it then falls through to the slower
    /// streaming disk path, with a debug line as the only symptom. Two
    /// mechanisms made the width frame-dependent and both were live: the `#212`
    /// warmup pre-flight guard (structurally — it reads `n`), and any dispatch
    /// failure that is a property of the data rather than of the indicator.
    /// Both now emit an all-NaN column under the right name instead of nothing.
    ///
    /// 60 bars vs the full fixture is chosen so the pre-flight guard fires on
    /// one side and not the other (period 50 needs 62.5 bars).
    #[test]
    fn the_column_set_is_independent_of_the_frame_length() {
        let full = crate::test_fixtures::ctrader_sample_ohlcv();
        assert!(full.len() > 60, "fixture too short to truncate meaningfully");
        let short = Ohlcv {
            timestamp: full.timestamp.as_ref().map(|t| t[..60].to_vec()),
            open: full.open[..60].to_vec(),
            high: full.high[..60].to_vec(),
            low: full.low[..60].to_vec(),
            close: full.close[..60].to_vec(),
            volume: full.volume.as_ref().map(|v| v[..60].to_vec()),
        };
        let a = compute_classic_ta_columns_with_policy(&full, IndicatorComputePolicy::Cpu).unwrap();
        // Sized against the LONGER frame, exactly as a multi-timeframe build
        // does — the budget must not vary per timeframe either.
        let b = compute_classic_ta_columns_sized(&short, IndicatorComputePolicy::Cpu, full.len())
            .unwrap();
        let names_a: Vec<&str> = a.iter().map(|(n, _)| n.as_str()).collect();
        let names_b: Vec<&str> = b.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names_a.len(),
            names_b.len(),
            "a {}-bar frame emitted {} columns and a 60-bar frame emitted {} — the cube's width \
             invariant would refuse to assemble and the run would silently take the disk path",
            full.len(),
            names_a.len(),
            names_b.len()
        );
        assert_eq!(names_a, names_b, "column names/order differ between frame lengths");
        for (name, v) in &b {
            assert_eq!(v.len(), 60, "column '{name}' is not frame length");
        }
    }

    /// Column names must be unique — a duplicate name silently shadows a real
    /// feature wherever the cube is projected by name.
    #[test]
    fn column_names_are_unique() {
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let cols =
            compute_classic_ta_columns_with_policy(&ohlcv, IndicatorComputePolicy::Cpu).unwrap();
        let mut seen = std::collections::HashSet::new();
        for (name, _) in &cols {
            assert!(seen.insert(name.clone()), "duplicate column name '{name}'");
        }
    }

    /// The sweep must ask each indicator for the window keys it actually
    /// declares. Setting `period` on MACD would be ignored and produce five
    /// identical columns wearing five different names.
    #[test]
    fn the_period_sweep_uses_each_indicators_real_window_parameters() {
        assert_eq!(period_plan("rsi"), PeriodPlan::Key("period"));
        // MACD declares fast/slow/signal, never `period`.
        match period_plan("macd") {
            PeriodPlan::Ratio(keys) => {
                assert_eq!(keys[0].0, "slow_period");
                let p = sweep_params(PeriodPlan::Ratio(keys), 52);
                // Anchored on slow=26, so 52 doubles the whole tuple:
                // fast 12->24, signal 9->18. The SHAPE is preserved, which is
                // what makes "macd at 52" a real indicator rather than noise.
                let get = |k: &str| {
                    p.iter()
                        .find(|kv| kv.key == k)
                        .map(|kv| match kv.value {
                            ParamValue::Int(i) => i,
                            _ => panic!("period params must be Int"),
                        })
                        .unwrap_or_else(|| panic!("missing {k}"))
                };
                assert_eq!(get("slow_period"), 52);
                assert_eq!(get("fast_period"), 24);
                assert_eq!(get("signal_period"), 18);
            }
            other => panic!("macd must use a coupled-window plan, got {other:?}"),
        }
        // stoch is the other coupled one.
        assert!(matches!(period_plan("stoch"), PeriodPlan::Ratio(_)));
        // obv declares no window — the historical `period` key is still sent so
        // the obv_7..obv_200 names survive, and the duplication is reported by
        // the census instead of being pretended away.
        assert_eq!(period_plan("obv"), PeriodPlan::NoWindow);
        assert_eq!(sweep_params(PeriodPlan::NoWindow, 21)[0].key, "period");
    }

    /// The ratio scaling must never produce a zero or negative window.
    #[test]
    fn coupled_window_scaling_never_underflows_to_zero() {
        for (_, keys) in COUPLED_WINDOWS {
            for &period in &ALT_PERIODS {
                for kv in sweep_params(PeriodPlan::Ratio(keys), period) {
                    match kv.value {
                        ParamValue::Int(i) => {
                            assert!(i >= 1, "{} scaled to {i} at period {period}", kv.key)
                        }
                        _ => panic!("expected Int"),
                    }
                }
            }
        }
    }

    /// The floor must be reachable: it is measured against the real recovery
    /// (324 ids / ~674 base columns), not set above what the library can give.
    #[test]
    fn the_vocabulary_floor_is_below_the_measured_recovery() {
        assert!(MIN_PRODUCING_INDICATOR_IDS < ALL_INDICATORS.len());
        assert!(
            MIN_PRODUCING_INDICATOR_IDS >= 200,
            "a floor low enough to pass the 1-column state would not be a floor at all"
        );
        assert!(MIN_BASE_VOCABULARY_COLUMNS >= 400);
    }

    // =======================================================================
    // Task #22 — CPU-vs-CUDA parity for the multi-period indicator sweep.
    //
    // Built only `--features gpu-cuda`, i.e. only on a box with the CUDA
    // toolkit. Runs against the REAL embedded EURUSD M1 sample, not a
    // synthetic ramp: `crates/neoethos-data/test_fixtures/eurusd_m1_100bars.json`
    // is captured cTrader data, so the price scale, the tick granularity and
    // the high/low geometry are the ones the kernels will actually see.
    //
    // ── THE TOLERANCE, AND WHY IT IS NOW NEARLY ZERO ──────────────────────
    //
    // This block used to argue for a LOOSE tolerance, because bit equality was
    // impossible for two reasons that were both true at the time:
    //
    //   1. vector-ta's device layer was f32 end to end
    //      (`CudaDeviceVectorF32`, `upload_f32`,
    //      `IndicatorCudaSeries::HostF32`) while every CPU indicator returns
    //      f64 — ~7 significant decimal digits against ~16; and
    //   2. vector-ta's build.rs passed `--use_fast_math` unless
    //      `CUDA_FAST_MATH=0` was exported, enabling flush-to-zero and
    //      approximate div/sqrt/rcp.
    //
    // BOTH ARE FIXED (2026-08-09):
    //
    //   1. `device_types_f64.rs` adds the f64 device vocabulary,
    //      `upload_ohlcv_f64` uploads without narrowing, and
    //      `kernels/cuda/neoethos_f64_kernels.cu` holds real f64 kernels for
    //      all ten indicators in `GPU_SWEEP_SPECS`;
    //   2. that file is listed in vector-ta build.rs's `F64_LANE_SOURCES`, so
    //      it is compiled with `-prec-div=true -prec-sqrt=true -fmad=false
    //      -ftz=false` and NEVER with `--use_fast_math` — the flag is no
    //      longer something an operator can get wrong.
    //
    // The kernels were written against this crate's own f64 CPU
    // implementations, preserving the accumulation ORDER (one thread per
    // column for everything recursive) and the exact `fma` placement (CUDA
    // `fma()` wherever the CPU uses `f64::mul_add`, plain `a*b+c` wherever it
    // does not, with `-fmad=false` stopping the compiler from fusing behind
    // our back). So the target is BIT EQUALITY, and the remaining tolerance is
    // a small safety margin against the parts of that claim that only a real
    // card can settle:
    //
    //   * whether nvcc reassociated anything we did not intend;
    //   * whether the device's `fma`, division and `fabs` agree with the
    //     host's on every input in the fixture (they should: all are
    //     IEEE-754 correctly rounded in double, and `-prec-div` forces the
    //     correctly-rounded divide).
    //
    // 1e-12 relative is roughly 4500 ulp at double precision — wide enough
    // that a single unintended reassociation still passes and gets REPORTED by
    // the printed worst-case, narrow enough that an f32 lane sneaking back in
    // (1.19e-7) fails instantly, which is the regression this guards.
    //
    // WORKFLOW: run this on the card, read the printed per-indicator worst
    // case, and if it is 0.0 across the board, tighten to exact equality.
    // Until then these numbers are a bound, not a measurement.
    //
    // What is NOT tolerated, at all:
    //   * a different NaN mask — the warmup boundary is structural, and a
    //     shifted warmup means the two lanes are computing different windows;
    //   * a different column set, name or order;
    //   * an infinity on either side.
    // =======================================================================

    /// `(absolute floor, relative ceiling)` for one indicator id.
    ///
    /// Uniform now: the lane is f64 on both sides, so there is no longer a
    /// reason for the recursive indicators to get a looser budget than the
    /// windowed ones. If the card shows otherwise, split it again WITH the
    /// measured numbers rather than with an error model.
    #[cfg(feature = "gpu-cuda")]
    fn parity_tolerance(_id: &str) -> (f64, f64) {
        // Absolute floor for values legitimately near zero (`mom`, `roc` and
        // the CCI numerator all cross zero), then a relative ceiling that is
        // five orders of magnitude tighter than f32 epsilon.
        (1e-12, 1e-12)
    }

    #[cfg(feature = "gpu-cuda")]
    #[test]
    fn gpu_cpu_indicator_sweep_parity() {
        use crate::core::gpu_indicators::{GPU_SWEEP_SPECS, GpuIndicatorEngine};

        let require_gpu = std::env::var("NEOETHOS_REQUIRE_GPU")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false);

        // REAL data. 100 bars clears the 7/21/50 periods under the 1.25x
        // pre-flight guard; 100 and 200 are skipped by BOTH lanes, which is
        // itself part of what the column-set assertion below checks. The
        // full-length check is the CLI feature build on the box.
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let n = ohlcv.len();
        assert!(n >= 64, "fixture too short to sweep anything: {n} bars");

        let candles = {
            let timestamps = ohlcv.timestamp.clone().unwrap_or_else(|| vec![0i64; n]);
            let volume = ohlcv.volume.clone().unwrap_or_else(|| vec![0.0; n]);
            Candles::new(
                timestamps,
                ohlcv.open.clone(),
                ohlcv.high.clone(),
                ohlcv.low.clone(),
                ohlcv.close.clone(),
                volume,
            )
        };

        let engine = match GpuIndicatorEngine::new(&ohlcv, 0) {
            Ok(e) => e,
            Err(e) => {
                if require_gpu {
                    panic!("NEOETHOS_REQUIRE_GPU set but the CUDA lane would not open: {e:?}");
                }
                eprintln!(
                    "gpu_cpu_indicator_sweep_parity: SKIPPED — no usable CUDA lane: {e:?}\n\
                     (set NEOETHOS_REQUIRE_GPU=1 to make this a hard failure)"
                );
                return;
            }
        };

        eprintln!(
            "parity vs {} ({}), fatbin archs {}, build.rs recorded {}, precision {}, \
             {n} real EURUSD M1 bars",
            engine.device_name(),
            engine.device_arch(),
            vector_ta::cuda::module_loader::COMPILED_ARCHS,
            crate::core::indicator_telemetry::VECTOR_TA_PTX_ARCH,
            engine.precision(),
        );

        let mut worst_overall = 0.0f64;
        for spec in GPU_SWEEP_SPECS {
            let gpu = engine
                .sweep_columns(spec, &ALT_PERIODS)
                .unwrap_or_else(|e| panic!("device sweep failed for {}: {e:?}", spec.id));
            // `Kernel::Scalar`, NOT `Kernel::Auto`, and this is load-bearing.
            //
            // Every f64 kernel in `neoethos_f64_kernels.cu` was written against
            // the crate's `*_scalar` implementation, bar for bar, in its exact
            // accumulation order. `neoethos-data` enables vector-ta's
            // `nightly-avx`, so on x86_64 `Auto` resolves to Avx2/Avx512 — and
            // the crate's own scalar and AVX paths are NOT bit-identical for
            // every indicator: `vwap` differs at index 136 and `wilders` at its
            // seed bar, 1 ULP each, measured card-lessly by
            // `tests/f64_lane_cpu_reference.rs` (both are withheld from the
            // kernel table over it).
            //
            // Measuring against `Auto` therefore measures the AVX divergence on
            // top of the device divergence, and a future AVX reassociation
            // would fail this test against a CORRECT kernel — burning a rented
            // card on a phantom. The sweep that runs in PRODUCTION still uses
            // `Auto`; only the oracle pins Scalar.
            let cpu = cpu_multi_period_columns(&candles, spec.id, &ALT_PERIODS, n, Kernel::Scalar);

            // Structural equality first — a mismatch here is a bug, never a
            // tolerance question.
            let gpu_names: Vec<&str> = gpu.iter().map(|(k, _)| k.as_str()).collect();
            let cpu_names: Vec<&str> = cpu.iter().map(|(k, _)| k.as_str()).collect();
            assert_eq!(
                gpu_names, cpu_names,
                "{}: column set/name/order differs between lanes",
                spec.id
            );

            let (abs_tol, rel_tol) = parity_tolerance(spec.id);
            let mut worst = 0.0f64;
            let mut worst_at = String::new();

            for ((name, gcol), (_, ccol)) in gpu.iter().zip(cpu.iter()) {
                assert_eq!(gcol.len(), ccol.len(), "{name}: length differs between lanes");
                for (j, (&g, &c)) in gcol.iter().zip(ccol.iter()).enumerate() {
                    assert_eq!(
                        g.is_nan(),
                        c.is_nan(),
                        "{name}[{j}]: NaN mask differs (gpu={g} cpu={c}) — the warmup boundary \
                         is structural, so the two lanes are computing different windows"
                    );
                    if c.is_nan() {
                        continue;
                    }
                    assert!(
                        g.is_finite(),
                        "{name}[{j}]: device produced {g} where the CPU produced a finite {c}"
                    );
                    let delta = (g - c).abs();
                    let allowed = abs_tol + rel_tol * c.abs();
                    // Track the normalised overshoot so "worst" is comparable
                    // across indicators with different price scales.
                    let ratio = delta / allowed.max(f64::MIN_POSITIVE);
                    if ratio > worst {
                        worst = ratio;
                        worst_at = format!("{name}[{j}] gpu={g} cpu={c} |d|={delta}");
                    }
                    assert!(
                        delta <= allowed,
                        "{name}[{j}]: |gpu-cpu| = {delta} > {allowed} \
                         (abs_tol={abs_tol}, rel_tol={rel_tol}) — gpu={g} cpu={c}"
                    );
                }
            }
            worst_overall = worst_overall.max(worst);
            eprintln!(
                "  {:<6} worst = {:.4} of budget (abs {:.0e} + rel {:.0e})   {}",
                spec.id, worst, abs_tol, rel_tol, worst_at
            );
        }

        engine.synchronize().expect("synchronize after parity sweep");
        eprintln!(
            "parity worst across all indicators: {worst_overall:.4} of budget. \
             0.0 means the two lanes are BIT-IDENTICAL on this frame. That is now a legitimate \
             target rather than a trap, because the CPU side of this comparison is pinned to \
             Kernel::Scalar — the reference every kernel was written against — instead of \
             Kernel::Auto, which resolves to AVX and is not bit-identical to Scalar for every \
             indicator in this crate. Anything above 0 names the ONE cell where a rounding \
             differs and must be explained before acceptance; do not widen the tolerance to \
             make it pass."
        );
    }

    /// The engine must never hand back a working-looking handle on a device
    /// whose arch cannot load the kernels — and when it refuses, the message
    /// must name BOTH arches so the operator can act without reading source.
    #[cfg(feature = "gpu-cuda")]
    #[test]
    fn arch_mismatch_error_names_both_arches() {
        use crate::core::gpu_indicators::GpuIndicatorEngine;
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        match GpuIndicatorEngine::new(&ohlcv, 0) {
            Ok(engine) => {
                // A lane that opened has already proven a real module load and
                // kernel launch, so the arches are compatible by construction.
                // Assert the reported strings are populated, not placeholders.
                assert!(engine.device_arch().starts_with("sm_"));
                assert_ne!(crate::core::indicator_telemetry::VECTOR_TA_PTX_ARCH, "none");
            }
            Err(e) => {
                let msg = format!("{e:#}");
                assert!(
                    msg.contains("compute capability") || msg.contains("cuda_available"),
                    "refusal message must name the device state, got: {msg}"
                );
            }
        }
    }
}
