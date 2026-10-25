//! Hardware-derived working-set width for the Classic/vector-ta producer.
//!
//! The frozen run plan admits a deterministic subset of indicator/period
//! outputs against one measurement, sized at the largest direct source frame.
//! Every timeframe then uses that same vocabulary; later memory pressure must
//! never silently change its columns. Deferred IDs remain explicit in the
//! admission report and can be covered by the existing streaming search.
//!
//! This budget covers f64 staging, not every producer's scratch or the final
//! f64-plus-validity cube. The retained-output preflight and higher-timeframe
//! scheduler in lib.rs perform additional checks before feature computation.
//! On Windows the live budget is bounded by process commit headroom as well as
//! physical RAM. A probe is not an OS/process-wide reservation.
//!
//! MIN_COLUMNS preserves the historical planning floor; it is NOT permission
//! to execute an unaffordable plan. A frozen plan that no longer fits must be
//! refused or streamed without dropping its admitted columns.

use crate::core::indicator_ledger::planned_output_count;

/// Bytes per value on the staging path. f64, always — the f32 lane was
/// measured 54% wrong at 200k bars and is never an option for anything the
/// search consumes.
pub const STAGING_VALUE_BYTES: u64 = 8;

/// Share of *available* RAM the f64 indicator staging buffers may occupy.
///
/// A quarter leaves room for the f64-plus-validity cube, source OHLCV and
/// per-worker transients. This planning fraction is not an OOM guarantee.
pub const STAGING_RAM_FRACTION: f64 = 0.25;

/// Never plan fewer columns than the vocabulary the system already had.
///
/// 66 = 13 swept indicators x 5 periods + `ttm_trend`. Below this the cap would
/// change the requested search. Keep the plan and let execution preflight
/// reject it if the measured allocation headroom is insufficient.
pub const MIN_COLUMNS: usize = 66;

/// Operator override for the planning headroom, for controlled comparisons.
///
/// The vocabulary budget is derived from a live memory probe, so two runs of the
/// SAME settings can admit a different number of indicator columns (and a
/// different streaming batch width) when the machine's free RAM differs. That
/// makes A/B comparisons of two search recipes unsound: the frame content, not
/// the recipe, decided the outcome. Setting this variable to a byte count pins
/// the planning headroom, so the admitted vocabulary and the streaming shape
/// become a pure function of the dataset and the pinned value.
///
/// The pin is planning only. Execution preflight still measures real headroom
/// and refuses an unaffordable complete plan; when the current probe is smaller
/// than the pin the shortfall is reported loudly (see `log`).
pub const ENV_PINNED_PLANNING_HEADROOM_BYTES: &str =
    "NEOETHOS_PINNED_FEATURE_BUDGET_HEADROOM_BYTES";

/// Absolute ceiling regardless of how much RAM the box has.
///
/// `ALL_INDICATORS` is 342 ids and the widest declares twelve outputs, so the
/// theoretical maximum is a few thousand. This bound exists to make a runaway
/// registry change (an id that suddenly declares 500 outputs) hit a wall with a
/// message rather than an allocator.
pub const MAX_COLUMNS_HARD_CEILING: usize = 4096;

/// Historical reference frame retained by the arithmetic regression tests.
pub const REFERENCE_ROWS: usize = 843_456;

/// Bytes one f64 column of `rows` values costs.
pub const fn column_bytes(rows: usize) -> u64 {
    (rows as u64).saturating_mul(STAGING_VALUE_BYTES)
}

/// Bytes `columns` f64 columns of `rows` values cost.
pub const fn staging_bytes(rows: usize, columns: usize) -> u64 {
    column_bytes(rows).saturating_mul(columns as u64)
}

/// A sized, hardware-derived plan for one timeframe's indicator vocabulary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VocabularyBudget {
    /// Bars in the frame being built.
    pub rows: usize,
    /// Allocation headroom at planning time, in bytes. `0` means exhausted or
    /// unknown; the planning floor is not permission to allocate it.
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
    /// therefore the column width) varies per timeframe.
    pub fn for_frame(rows: usize) -> Self {
        let actual = neoethos_core::allocation_headroom_bytes();
        Self::for_frame_from_headroom(rows, actual, pinned_planning_headroom_bytes_from_env())
    }

    /// Deterministic core of [`VocabularyBudget::for_frame`] with the planning
    /// headroom resolved explicitly, so a pinned budget can be tested without a
    /// machine probe or process-global environment mutation.
    ///
    /// When `pinned` is present it REPLACES the probe; the actual probe is kept
    /// only so [`VocabularyBudget::log`] can report a shortfall. Never adds
    /// permission: the execution preflight still measures real headroom.
    pub fn for_frame_from_headroom(rows: usize, actual: u64, pinned: Option<u64>) -> Self {
        Self::from_available(rows, pinned.unwrap_or(actual))
    }

    /// Planning headroom this budget was sized from, and the live probe it
    /// replaced (if any).
    pub fn planning_headroom(&self) -> PlanningHeadroomV1 {
        PlanningHeadroomV1 {
            headroom_bytes: self.available_bytes,
            pinned: pinned_planning_headroom_bytes_from_env().is_some(),
            probe_bytes: neoethos_core::allocation_headroom_bytes(),
        }
    }

    /// Size ONE budget for a whole multi-timeframe run from its WIDEST frame.
    ///
    /// `widest_rows` is the largest independently downloaded timeframe row
    /// count. A budget that fits that direct frame fits every smaller direct
    /// frame, and the admitted id set is identical across the run.
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
    pub fn log(&self, stage: &str, planned_columns: usize, admitted_planned_columns: usize) {
        tracing::info!(
            target: "neoethos_data::feature_budget",
            stage,
            rows = self.rows,
            available_gb = format!("{:.2}", self.available_bytes as f64 / 1e9),
            budget_gb = format!("{:.2}", self.budget_bytes as f64 / 1e9),
            column_mb = format!("{:.2}", column_bytes(self.rows) as f64 / 1e6),
            max_columns = self.max_columns,
            planned_columns,
            admitted_columns = admitted_planned_columns,
            admitted_gb = format!(
                "{:.2}",
                self.planned_bytes(admitted_planned_columns) as f64 / 1e9
            ),
            "indicator vocabulary budget (f64 staging, derived from allocation headroom — never from a \
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
                 within its staging budget; the planning floor does not authorize allocation. \
                 Execution preflight must still admit the complete frozen output."
            );
        }
        let provenance = self.planning_headroom();
        if provenance.pinned {
            let shortfall = provenance.shortfall_bytes();
            tracing::info!(
                target: "neoethos_data::feature_budget",
                stage,
                rows = self.rows,
                pinned = true,
                planned_headroom_gb = format!("{:.2}", provenance.headroom_bytes as f64 / 1e9),
                probe_headroom_gb = format!("{:.2}", provenance.probe_bytes as f64 / 1e9),
                shortfall_gb = format!("{:.2}", shortfall as f64 / 1e9),
                "planning headroom is PINNED by {ENV_PINNED_PLANNING_HEADROOM_BYTES}: the admitted \
                 vocabulary and streaming batch are a function of the dataset and the pin, not of this \
                 machine's free RAM. Execution preflight still measures the real headroom."
            );
            if shortfall > 0 {
                tracing::warn!(
                    target: "neoethos_data::feature_budget",
                    stage,
                    pinned_headroom_gb = format!("{:.2}", provenance.headroom_bytes as f64 / 1e9),
                    probe_headroom_gb = format!("{:.2}", provenance.probe_bytes as f64 / 1e9),
                    shortfall_gb = format!("{:.2}", shortfall as f64 / 1e9),
                    "the pinned planning headroom EXCEEDS the current allocation probe — the complete \
                     pinned plan may be refused by the execution preflight. This is a machine-capacity \
                     shortfall, not a vocabulary change: the pinned recipe is unchanged."
                );
            }
        }
        if admitted_planned_columns < planned_columns {
            tracing::warn!(
                target: "neoethos_data::feature_budget",
                stage,
                planned_columns,
                admitted_columns = admitted_planned_columns,
                deferred = planned_columns - admitted_planned_columns,
                "the indicator vocabulary was TRUNCATED to fit this machine — the search will \
                 explore a narrower feature space than a larger box would. The deferred ids are \
                 listed by name under reason=over_budget in the indicator census."
            );
        }
    }
}

/// Where a budget's planning headroom came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanningHeadroomV1 {
    /// Headroom the budget was sized from (the pin when one is set).
    pub headroom_bytes: u64,
    /// True when an operator pin replaced the live probe.
    pub pinned: bool,
    /// The live probe at the moment the budget was built.
    pub probe_bytes: u64,
}

impl PlanningHeadroomV1 {
    /// The live probe is smaller than the pinned planning headroom, so the
    /// complete plan may be refused by the execution preflight.
    pub fn shortfall_bytes(&self) -> u64 {
        if self.pinned {
            self.headroom_bytes.saturating_sub(self.probe_bytes)
        } else {
            0
        }
    }
}

/// Parse a pinned planning headroom. Zero and unparsable values are ignored so
/// a typo cannot silently plan an empty vocabulary.
pub fn pinned_planning_headroom_from_value(value: Option<&str>) -> Option<u64> {
    value
        .map(str::trim)
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|bytes| *bytes > 0)
}

/// Operator pin for the planning headroom, when set in this process.
pub fn pinned_planning_headroom_bytes_from_env() -> Option<u64> {
    pinned_planning_headroom_from_value(
        std::env::var(ENV_PINNED_PLANNING_HEADROOM_BYTES)
            .ok()
            .as_deref(),
    )
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
    fn staging_size_overflow_cannot_wrap_into_an_affordable_plan() {
        assert_eq!(staging_bytes(usize::MAX, usize::MAX), u64::MAX);
        let budget = VocabularyBudget::from_available(usize::MAX, 1_000);
        assert!(budget.floor_binds);
        assert!(budget.planned_bytes(MIN_COLUMNS) > budget.budget_bytes);
    }

    #[test]
    fn the_floor_can_never_narrow_the_search_below_what_it_already_had() {
        // A machine with 100 MB free would arithmetically afford 3 columns.
        let b = VocabularyBudget::from_available(REFERENCE_ROWS, 100_000_000);
        assert_eq!(b.max_columns, MIN_COLUMNS);
        assert!(
            b.floor_binds,
            "the floor overriding hardware must be reported"
        );
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
    fn pinned_headroom_replaces_the_probe_and_keeps_the_arithmetic() {
        let actual = 8_000_000_000u64;
        let pinned = 32_000_000_000u64;
        let budget =
            VocabularyBudget::for_frame_from_headroom(REFERENCE_ROWS, actual, Some(pinned));
        assert_eq!(budget.available_bytes, pinned);
        assert_eq!(
            budget.max_columns,
            VocabularyBudget::from_available(REFERENCE_ROWS, pinned).max_columns
        );
        assert!(budget.max_columns > VocabularyBudget::from_available(REFERENCE_ROWS, actual).max_columns);
    }

    #[test]
    fn a_pin_makes_the_recipe_independent_of_the_live_probe() {
        let pinned = 16_000_000_000u64;
        let starved = VocabularyBudget::for_frame_from_headroom(REFERENCE_ROWS, 4_000_000_000, Some(pinned));
        let generous = VocabularyBudget::for_frame_from_headroom(REFERENCE_ROWS, 60_000_000_000, Some(pinned));
        assert_eq!(starved, generous, "same pin, same rows must plan the same vocabulary");
        // Without a pin the probe decides, which is exactly the unsound case.
        let probed_small = VocabularyBudget::for_frame_from_headroom(REFERENCE_ROWS, 4_000_000_000, None);
        assert_ne!(probed_small, starved);
    }

    #[test]
    fn pin_parsing_ignores_zero_and_garbage() {
        assert_eq!(pinned_planning_headroom_from_value(Some("34359738368")), Some(34_359_738_368));
        assert_eq!(pinned_planning_headroom_from_value(Some("  1073741824 ")), Some(1_073_741_824));
        assert_eq!(pinned_planning_headroom_from_value(Some("0")), None);
        assert_eq!(pinned_planning_headroom_from_value(Some("twelve")), None);
        assert_eq!(pinned_planning_headroom_from_value(None), None);
    }

    #[test]
    fn shortfall_is_reported_only_when_a_pin_exceeds_the_probe() {
        let under = PlanningHeadroomV1 {
            headroom_bytes: 16_000_000_000,
            pinned: true,
            probe_bytes: 20_000_000_000,
        };
        assert_eq!(under.shortfall_bytes(), 0);
        let over = PlanningHeadroomV1 {
            headroom_bytes: 32_000_000_000,
            pinned: true,
            probe_bytes: 12_000_000_000,
        };
        assert_eq!(over.shortfall_bytes(), 20_000_000_000);
        let unpinned = PlanningHeadroomV1 {
            headroom_bytes: 32_000_000_000,
            pinned: false,
            probe_bytes: 12_000_000_000,
        };
        assert_eq!(unpinned.shortfall_bytes(), 0, "an unpinned budget always sized from the probe");
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
