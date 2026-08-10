//! Hardware-derived ceiling on how wide the indicator vocabulary may get.
//!
//! # The number that forced this module to exist
//!
//! Repairing the two dispatch mistakes in `hpc_ta` takes the base-timeframe
//! vocabulary from **66 columns to roughly 800** — a 12x multiplication of the
//! widest object the system builds. That is the point of the fix, but a
//! vocabulary that OOMs is not an improvement, so the width is capped by the
//! MACHINE and never by a user parameter.
//!
//! ## The arithmetic, at the real shapes
//!
//! One f64 column is `rows * 8` bytes. At the 843,456-bar reference frame:
//!
//! | columns | bytes/column | per timeframe |
//! |--------:|-------------:|--------------:|
//! |      66 |     6.75 MB  |    **445 MB** (today) |
//! |     233 |     6.75 MB  |    **1.57 GB** (shape fix alone) |
//! |     804 |     6.75 MB  |    **5.43 GB** (shape fix + output enumeration) |
//!
//! The EURUSD M5 store actually holds **1,054,320** bars, where one column is
//! 8.43 MB and 804 columns cost **6.78 GB per timeframe**.
//!
//! Downstream the cube narrows to f32 (`lib.rs` documents that narrowing), so
//! the assembled 3-timeframe cube at 804 columns/TF is
//! `843,456 * 2412 * 4 = 8.14 GB` — and `should_build_cube_in_ram` already
//! sizes THAT against free RAM. What it does not size, and what this module
//! does, is the f64 STAGING peak inside `compute_classic_ta_columns`: every
//! column of one timeframe is live as a `Vec<f64>` before any of it reaches the
//! cube.
//!
//! ## What is capped, and by what
//!
//! [`VocabularyBudget::for_frame`] converts free RAM into a maximum column
//! count. It spends at most [`STAGING_RAM_FRACTION`] of *available* memory,
//! never a constant, so the same binary produces a 66-column vocabulary on a
//! laptop and an 800-column one on a 128 GB box without any flag being set.
//!
//! Two deliberate properties:
//!
//! * **The floor is the status quo.** [`MIN_COLUMNS`] is 66 — the vocabulary
//!   every historical result was produced on. A machine too small to afford
//!   more still gets exactly what it got before, with a WARN saying so. The cap
//!   can never make the system worse than it was.
//! * **The cut is deterministic and counted.** Ids are admitted in
//!   `ALL_INDICATORS` order until the planned column count reaches the cap;
//!   everything past it is recorded as `DropReason::OverBudget` with its name.
//!   A truncated vocabulary is loud, not silent.
//!
//! ## Two properties that were WRONG in the first cut of this module
//!
//! Both were found by review and both were load-bearing:
//!
//! 1. **The budget must not be sized from the FRAME.** `for_frame(n)` called
//!    once per timeframe made the admitted id set a function of that
//!    timeframe's row count — base M5 (1,054,320 rows) truncated while H1
//!    (70,288) and H4 (17,572) did not, so the per-TF cube widths diverged by
//!    ~140 columns and `lib.rs`'s width invariant bailed the in-RAM cube to the
//!    disk path on every run on any box under ~40 GB free. The column SET must
//!    be a function of the id list and the machine, never of the frame. The
//!    caller therefore sizes ONE budget from the run's WIDEST frame (the base
//!    timeframe) via [`VocabularyBudget::for_run`] and passes it to every
//!    timeframe — see `hpc_ta::compute_classic_ta_columns_sized`.
//!
//! 2. **The cap must cover EVERY stage that stages columns.** The period sweep
//!    used to run entirely outside the budget, so `max_columns` was not the
//!    peak: at the M5 store's depth its 130 columns are another 1.10 GB of
//!    staging that nothing accounted for. [`VocabularyBudget::reserve`] carves
//!    the sweep's planned width out FIRST — it is the historical vocabulary and
//!    therefore has first claim — and the base pass is admitted against what is
//!    left.

use crate::core::indicator_ledger::planned_output_count;

/// Bytes per value on the staging path. f64, always — the f32 lane was
/// measured 54% wrong at 200k bars and is never an option for anything the
/// search consumes.
pub const STAGING_VALUE_BYTES: u64 = 8;

/// Share of *available* RAM the f64 indicator staging buffers may occupy.
///
/// A quarter, not a half: the same process is about to hold the f32 cube
/// (sized separately by `should_build_cube_in_ram` at 1.5x + 2 GB), the source
/// OHLCV, and rayon's per-worker transients. Peak is therefore a function of
/// the machine at every step, which is the never-OOM invariant.
pub const STAGING_RAM_FRACTION: f64 = 0.25;

/// Never plan fewer columns than the vocabulary the system already had.
///
/// 66 = 13 swept indicators x 5 periods + `ttm_trend`. Below this the cap would
/// be a regression rather than a guard, so on a machine that cannot afford 66
/// columns we take the memory risk and say so at WARN — the alternative is
/// silently searching a smaller space than last week.
pub const MIN_COLUMNS: usize = 66;

/// Absolute ceiling regardless of how much RAM the box has.
///
/// `ALL_INDICATORS` is 342 ids and the widest declares twelve outputs, so the
/// theoretical maximum is a few thousand. This bound exists to make a runaway
/// registry change (an id that suddenly declares 500 outputs) hit a wall with a
/// message rather than an allocator.
pub const MAX_COLUMNS_HARD_CEILING: usize = 4096;

/// The reference frame used in this module's documented arithmetic — the bar
/// count the operator's sizing question was asked at.
pub const REFERENCE_ROWS: usize = 843_456;

/// Bytes one f64 column of `rows` values costs.
pub const fn column_bytes(rows: usize) -> u64 {
    rows as u64 * STAGING_VALUE_BYTES
}

/// Bytes `columns` f64 columns of `rows` values cost.
pub const fn staging_bytes(rows: usize, columns: usize) -> u64 {
    column_bytes(rows) * columns as u64
}

/// A sized, hardware-derived plan for one timeframe's indicator vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VocabularyBudget {
    /// Bars in the frame being built.
    pub rows: usize,
    /// Free RAM the decision was made against, in bytes. `0` means the probe
    /// failed, in which case the budget collapses to [`MIN_COLUMNS`].
    pub available_bytes: u64,
    /// Bytes the staging buffers are allowed to occupy.
    pub budget_bytes: u64,
    /// Maximum columns this frame may stage.
    pub max_columns: usize,
    /// True when [`MIN_COLUMNS`] had to override a smaller hardware answer, i.e.
    /// the machine cannot really afford even the historical vocabulary.
    pub floor_binds: bool,
}

impl VocabularyBudget {
    /// Size the vocabulary from the machine this run is actually on, against
    /// the row count of THIS frame.
    ///
    /// Correct only for a single-frame build. A multi-timeframe build must use
    /// [`VocabularyBudget::for_run`] with the widest frame's row count and pass
    /// the SAME budget to every timeframe — otherwise the admitted id set (and
    /// therefore the column width) varies per timeframe. See this module's
    /// header, property 1.
    pub fn for_frame(rows: usize) -> Self {
        Self::from_available(rows, neoethos_core::available_memory_bytes())
    }

    /// Size ONE budget for a whole multi-timeframe run from its WIDEST frame.
    ///
    /// `widest_rows` is the base timeframe's bar count: every other timeframe
    /// is a resampling of it and therefore shorter, so a budget that fits the
    /// base fits all of them, and the admitted id set is identical across the
    /// run. Conservative by construction — the short frames are charged the
    /// base frame's per-column price.
    pub fn for_run(widest_rows: usize) -> Self {
        Self::for_frame(widest_rows)
    }

    /// Carve `reserved` columns out of this budget for a stage planned
    /// separately, returning the budget the remaining stage may spend.
    ///
    /// Used to give the period sweep first claim: the sweep IS the historical
    /// 66-column vocabulary, so on a machine that can afford nothing else the
    /// system still gets exactly what it had before, and the base pass takes
    /// what is left. The remainder never goes below 1 — a zero-width budget
    /// would make `admit_indicators` defer everything and turn a memory
    /// decision into a vocabulary collapse.
    pub fn reserve(&self, reserved: usize) -> Self {
        VocabularyBudget {
            max_columns: self.max_columns.saturating_sub(reserved).max(1),
            ..self.clone()
        }
    }

    /// Deterministic core of [`VocabularyBudget::for_frame`], separated so the
    /// arithmetic can be tested without a machine probe.
    pub fn from_available(rows: usize, available_bytes: u64) -> Self {
        let budget_bytes = (available_bytes as f64 * STAGING_RAM_FRACTION) as u64;
        let per_column = column_bytes(rows).max(1);
        let affordable = (budget_bytes / per_column) as usize;
        let capped = affordable.min(MAX_COLUMNS_HARD_CEILING);
        let floor_binds = capped < MIN_COLUMNS;
        VocabularyBudget {
            rows,
            available_bytes,
            budget_bytes,
            max_columns: capped.max(MIN_COLUMNS),
            floor_binds,
        }
    }

    /// Bytes the fully-admitted plan will stage.
    pub fn planned_bytes(&self, columns: usize) -> u64 {
        staging_bytes(self.rows, columns)
    }

    /// Announce the budget once per frame, with the arithmetic visible.
    ///
    /// The operator must never have to infer why a run searched a narrower
    /// vocabulary than the last one: the numbers that produced the cap are in
    /// the line.
    pub fn log(&self, stage: &str, planned_columns: usize, admitted_columns: usize) {
        tracing::info!(
            target: "neoethos_data::feature_budget",
            stage,
            rows = self.rows,
            available_gb = format!("{:.2}", self.available_bytes as f64 / 1e9),
            budget_gb = format!("{:.2}", self.budget_bytes as f64 / 1e9),
            column_mb = format!("{:.2}", column_bytes(self.rows) as f64 / 1e6),
            max_columns = self.max_columns,
            planned_columns,
            admitted_columns,
            admitted_gb = format!("{:.2}", self.planned_bytes(admitted_columns) as f64 / 1e9),
            "indicator vocabulary budget (f64 staging peak, derived from free RAM — never from a \
             user parameter)"
        );
        if self.floor_binds {
            tracing::warn!(
                target: "neoethos_data::feature_budget",
                stage,
                rows = self.rows,
                available_gb = format!("{:.2}", self.available_bytes as f64 / 1e9),
                min_columns = MIN_COLUMNS,
                "this machine cannot afford even the historical {MIN_COLUMNS}-column vocabulary \
                 within its staging budget; building it anyway because narrowing below the \
                 status quo would be a silent regression. Expect memory pressure."
            );
        }
        if admitted_columns < planned_columns {
            tracing::warn!(
                target: "neoethos_data::feature_budget",
                stage,
                planned_columns,
                admitted_columns,
                deferred = planned_columns - admitted_columns,
                "the indicator vocabulary was TRUNCATED to fit this machine — the search will \
                 explore a narrower feature space than a larger box would. The deferred ids are \
                 listed by name under reason=over_budget in the indicator census."
            );
        }
    }
}

/// Which ids fit the budget, in deterministic `ALL_INDICATORS` order.
///
/// Returns `(admitted, deferred, planned_total_columns)`. Planning uses only
/// registry lookups, so nothing is allocated before the decision is made — the
/// cap is applied BEFORE the memory is spent, not after.
pub fn admit_indicators<'a>(
    ids: &[&'a str],
    budget: &VocabularyBudget,
) -> (Vec<&'a str>, Vec<&'a str>, usize) {
    let mut admitted = Vec::with_capacity(ids.len());
    let mut deferred = Vec::new();
    let mut used = 0usize;
    let mut planned_total = 0usize;
    for &id in ids {
        let want = planned_output_count(id);
        planned_total += want;
        if used + want <= budget.max_columns {
            used += want;
            admitted.push(id);
        } else {
            deferred.push(id);
        }
    }
    (admitted, deferred, planned_total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn documented_arithmetic_is_actually_what_the_code_computes() {
        // The rows of the table in this module's docs, recomputed rather than
        // restated — a doc comment that drifts from the code is how a sizing
        // claim becomes a lie.
        assert_eq!(column_bytes(REFERENCE_ROWS), 6_747_648);
        assert_eq!(staging_bytes(REFERENCE_ROWS, 66), 445_344_768);
        assert_eq!(staging_bytes(REFERENCE_ROWS, 233), 1_572_201_984);
        assert_eq!(staging_bytes(REFERENCE_ROWS, 804), 5_425_108_992);
        // The M5 store's real depth.
        assert_eq!(column_bytes(1_054_320), 8_434_560);
        assert_eq!(staging_bytes(1_054_320, 804), 6_781_386_240);
    }

    #[test]
    fn full_vocabulary_costs_about_five_and_a_half_gigabytes_per_timeframe() {
        let bytes = staging_bytes(REFERENCE_ROWS, 804);
        let gb = bytes as f64 / 1e9;
        assert!(
            (5.4..5.5).contains(&gb),
            "804 f64 columns at {REFERENCE_ROWS} bars should cost ~5.42 GB, got {gb:.3} GB"
        );
        // And at the store's real M5 depth.
        let real = staging_bytes(1_054_320, 804) as f64 / 1e9;
        assert!(
            (6.7..6.9).contains(&real),
            "at the M5 store's 1,054,320 bars it should cost ~6.78 GB, got {real:.3} GB"
        );
    }

    #[test]
    fn budget_is_a_function_of_hardware_not_of_a_constant() {
        let small = VocabularyBudget::from_available(REFERENCE_ROWS, 8_000_000_000);
        let large = VocabularyBudget::from_available(REFERENCE_ROWS, 128_000_000_000);
        assert!(
            large.max_columns > small.max_columns,
            "a bigger machine must be allowed a wider vocabulary ({} vs {})",
            large.max_columns,
            small.max_columns
        );
        // 8 GB free -> 2 GB budget / 6.75 MB per column = 296 columns.
        assert_eq!(small.max_columns, 296);
        // 128 GB free -> 32 GB budget -> hits the hard ceiling, not the RAM.
        assert_eq!(large.max_columns, MAX_COLUMNS_HARD_CEILING);
    }

    #[test]
    fn a_failed_memory_probe_falls_back_to_the_historical_vocabulary_not_to_zero() {
        let b = VocabularyBudget::from_available(REFERENCE_ROWS, 0);
        assert_eq!(b.max_columns, MIN_COLUMNS);
        assert!(b.floor_binds);
    }

    #[test]
    fn the_floor_can_never_narrow_the_search_below_what_it_already_had() {
        // A machine with 100 MB free would arithmetically afford 3 columns.
        let b = VocabularyBudget::from_available(REFERENCE_ROWS, 100_000_000);
        assert_eq!(b.max_columns, MIN_COLUMNS);
        assert!(b.floor_binds, "the floor overriding hardware must be reported");
    }

    #[test]
    fn admission_is_deterministic_and_reports_what_it_deferred() {
        let ids = ["rsi", "ema", "sma", "atr"];
        // Every one of these is single-output, so the plan is 1 column each.
        let budget = VocabularyBudget {
            rows: 1000,
            available_bytes: 0,
            budget_bytes: 0,
            max_columns: 2,
            floor_binds: false,
        };
        let (adm, def, planned) = admit_indicators(&ids, &budget);
        assert_eq!(adm, vec!["rsi", "ema"]);
        assert_eq!(def, vec!["sma", "atr"]);
        assert_eq!(planned, 4);
        // Order-stable across calls.
        let (adm2, _, _) = admit_indicators(&ids, &budget);
        assert_eq!(adm, adm2);
    }

    #[test]
    fn a_generous_budget_admits_the_whole_indicator_list() {
        let budget = VocabularyBudget::from_available(200_000, 64_000_000_000);
        let ids: Vec<&str> = crate::core::all_indicators::ALL_INDICATORS.to_vec();
        let (adm, def, _planned) = admit_indicators(&ids, &budget);
        assert!(def.is_empty(), "{} ids deferred on a 64 GB box", def.len());
        assert_eq!(adm.len(), ids.len());
    }
}
