//! Counted, reasoned outcomes for every indicator column the feature build
//! attempts — the mechanism that makes a silent drop impossible.
//!
//! # Why this module exists
//!
//! `hpc_ta` used to iterate all 342 ids in [`crate::core::all_indicators`] and
//! keep whatever came back through an `if let Ok(output)` with no `else` and an
//! `if v.len() == n` with no `else`. Exactly ONE of the 342 (`ttm_trend`)
//! produced a column. The other 341 were discarded with no log line, no
//! counter, and no way for any run to know. That state survived sixteen months
//! because nothing in the system was capable of noticing it.
//!
//! Measured causes (real EURUSD M5 bars, identical at 20k and 200k rows):
//!
//! | count | cause |
//! |------:|-------|
//! |   232 | `Ok` with a correct full-length f64 series, thrown away because the accept test read vector-ta's `rows`/`cols` (a 1-D series is reported `rows=1 x cols=n`, so `cols <= 1` was false and the multi-output branch then demanded `rows >= n`, i.e. `1 >= 200000`) |
//! |    92 | `output_id: None` against a multi-output indicator — vector-ta requires an explicit output id |
//! |     8 | `UnknownIndicator` — no dispatch arm in vector-ta 0.2.9 |
//! |     5 | `UnknownOutput` — unregistered id whose implicit `"value"` output does not exist |
//! |     4 | `UnsupportedCapability { capability: "cpu_batch" }` |
//! |     1 | kept (`ttm_trend`, whose Bool arm never consulted `cols`) |
//! |     0 | panics |
//! |     0 | caused by `params: &[]` — vector-ta fills every declared default itself |
//!
//! So the fix is two mechanical corrections (key off the value count; enumerate
//! the declared outputs) — but the DURABLE fix is this module. Every discard on
//! the feature path now carries a [`DropReason`] and a name, is counted, is
//! summarised once per frame at INFO, and trips a hard error when the produced
//! vocabulary falls through the floor. A future regression from 800 columns
//! back to 66 cannot be missed: it fails the build of the feature frame and
//! names the reason bucket that grew.

use std::collections::{BTreeMap, BTreeSet};

use vector_ta::indicators::dispatch::IndicatorDispatchError;
use vector_ta::indicators::registry::get_indicator;

// ---------------------------------------------------------------------------
// Indicators that CANNOT produce a column, named explicitly with the reason.
// ---------------------------------------------------------------------------

/// Ids in [`crate::core::all_indicators::ALL_INDICATORS`] that vector-ta 0.2.9
/// cannot serve through `compute_cpu`, each with the measured reason.
///
/// These are still ATTEMPTED on every frame. Excluding them from the attempt
/// would make the table rot silently the day vector-ta gains an arm — the exact
/// failure mode this module exists to end. Instead:
///
///   * a failure here is an EXPECTED failure: counted, reported by name, and
///     not charged against the production floor;
///   * a SUCCESS here is reported at WARN, because it means this table is stale
///     and the id should be promoted back to the expected-to-produce set.
pub const EXPECTED_NON_PRODUCING: &[(&str, &str)] = &[
    // ---- not indicators at all: moving-average FAMILY dispatch entry points.
    // vector-ta exposes the MA family through one `ma`-shaped selector that
    // takes a `ma_type` enum; the concrete members (sma, ema, wma, hma, alma,
    // …) are already listed individually in ALL_INDICATORS, so these three are
    // pure duplication of a dispatch mechanism, not vocabulary.
    ("ma", "moving-average family selector, not an indicator; members listed individually"),
    ("ma_batch", "moving-average family batch selector, not an indicator"),
    ("ma_stream", "moving-average family streaming selector, not an indicator"),
    // ---- `UnknownIndicator`: present in ALL_INDICATORS, absent from
    // vector-ta 0.2.9's dispatch table entirely. Real gaps in the library.
    ("cvi", "UnknownIndicator: no dispatch arm in vector-ta 0.2.9"),
    ("insync_index", "UnknownIndicator: no dispatch arm in vector-ta 0.2.9"),
    ("marketefi", "UnknownIndicator: no dispatch arm in vector-ta 0.2.9"),
    ("sar", "UnknownIndicator: no dispatch arm in vector-ta 0.2.9"),
    ("trend_follower", "UnknownIndicator: no dispatch arm in vector-ta 0.2.9"),
    // ---- `UnsupportedCapability { capability: "cpu_batch" }`: these DO have
    // registry entries, but `cpu_single.rs` routes every non-pattern
    // request through `compute_cpu_batch`, and `dispatch_cpu_batch_by_indicator`
    // has no arm for them, so they fall to the catch-all. Reachable only by
    // calling their `*_with_kernel` functions directly, which is a separate,
    // larger change against the vendored crate.
    ("dec_osc", "UnsupportedCapability 'cpu_batch': registered but no cpu_batch dispatch arm"),
    ("decycler", "UnsupportedCapability 'cpu_batch': registered but no cpu_batch dispatch arm"),
    ("ott", "UnsupportedCapability 'cpu_batch': registered but no cpu_batch dispatch arm"),
    ("rsx", "UnsupportedCapability 'cpu_batch': registered but no cpu_batch dispatch arm"),
    // ---- `UnknownOutput`: vector-ta CONTRADICTS ITSELF about this id. The
    // registry declares eight named F64 outputs for it
    // (`OUTPUTS_VDUBUS_DIVERGENCE_WAVE_PATTERN_GENERATOR`, registry.rs:1240 —
    // fast_standard, fast_climax, fast_rounded, fast_predator, slow_*), but its
    // dispatch arm (cpu_batch.rs:5095) opens with
    // `expect_value_output(...)`, which accepts ONLY the literal "value"
    // (cpu_batch.rs:17247). So every output_id the registry hands us is rejected
    // with `UnknownOutput`, and "value" is not a name the registry offers.
    // The id is unreachable through `compute_cpu` in either direction.
    //
    // NOT a silent absence: it is attempted every frame, counted under
    // DropReason::UnknownOutput, and named here. If a future vector-ta
    // reconciles the two, the id starts producing and `stale_exclusion` fires a
    // WARN telling us to delete this line.
    (
        "vdubus_divergence_wave_pattern_generator",
        "UnknownOutput: registry declares 8 named outputs but the dispatch arm's \
         expect_value_output() accepts only \"value\" — vector-ta 0.2.9 self-contradiction",
    ),
];

/// Individual OUTPUTS that vector-ta 0.2.9 declares in its registry but cannot
/// serve, on ids whose OTHER outputs work fine.
///
/// The id-level table above cannot express this case, and the case is real: an
/// indicator that drops from eight declared series to five still "produces", so
/// every id-level check — including the vocabulary floor — is blind to it. That
/// is the original 341-column defect at output granularity, and it is exactly
/// the granularity the defect actually had (92 of the 342 failures were
/// `output_id`-shaped).
///
/// Same contract as [`EXPECTED_NON_PRODUCING`]: these are still ATTEMPTED on
/// every frame, a failure is counted under its real `DropReason`, and a SUCCESS
/// means this table is stale.
pub const EXPECTED_NON_PRODUCING_OUTPUTS: &[(&str, &str, &str)] = &[
    // vector-ta CONTRADICTS ITSELF on three of this id's eight declared
    // outputs. `registry.rs:1504` (OUTPUTS_ADAPTIVE_BOUNDS_RSI) declares
    // `lower`, `middle` and `upper` — the shared OUTPUT_LOWER / OUTPUT_MIDDLE /
    // OUTPUT_UPPER constants at registry.rs:434-453 — but the dispatch arm's
    // own name table, `adaptive_bounds_rsi_field` at cpu_batch.rs:12549, spells
    // the same three series `lower_bound`, `mid` and `upper_bound` and rejects
    // everything else with `UnknownOutput`. The other five (rsi, lower_mid,
    // upper_mid, regime, regime_flip) agree between the two and DO produce.
    //
    // Reachable only by requesting a name the registry does not offer, which
    // would mean this crate second-guessing the registry — a much worse
    // precedent than three named, counted absences.
    (
        "adaptive_bounds_rsi",
        "lower",
        "UnknownOutput: registry declares `lower` (registry.rs:1504/449) but the dispatch arm's \
         adaptive_bounds_rsi_field (cpu_batch.rs:12549) accepts only `lower_bound` — vector-ta \
         0.2.9 self-contradiction",
    ),
    (
        "adaptive_bounds_rsi",
        "middle",
        "UnknownOutput: registry declares `middle` (registry.rs:1504/439) but the dispatch arm's \
         adaptive_bounds_rsi_field (cpu_batch.rs:12549) accepts only `mid` — vector-ta 0.2.9 \
         self-contradiction",
    ),
    (
        "adaptive_bounds_rsi",
        "upper",
        "UnknownOutput: registry declares `upper` (registry.rs:1504/434) but the dispatch arm's \
         adaptive_bounds_rsi_field (cpu_batch.rs:12549) accepts only `upper_bound` — vector-ta \
         0.2.9 self-contradiction",
    ),
];

/// Is this `(id, output_id)` pair one of the [`EXPECTED_NON_PRODUCING_OUTPUTS`]?
pub fn expected_non_producing_output(id: &str, output_id: &str) -> Option<&'static str> {
    EXPECTED_NON_PRODUCING_OUTPUTS
        .iter()
        .find(|(i, o, _)| *i == id && *o == output_id)
        .map(|(_, _, why)| *why)
}

/// Is this id one of the [`EXPECTED_NON_PRODUCING`] set?
pub fn expected_non_producing(id: &str) -> Option<&'static str> {
    EXPECTED_NON_PRODUCING
        .iter()
        .find(|(k, _)| *k == id)
        .map(|(_, why)| *why)
}

// ---------------------------------------------------------------------------
// Output-id resolution.
// ---------------------------------------------------------------------------

/// Multi-output indicators that have NO vector-ta registry entry, so
/// `get_indicator` cannot enumerate their outputs.
///
/// `compute_cpu_batch` falls through to a by-name match with `output_id`
/// defaulted to the literal `"value"`, which these five reject with
/// `UnknownOutput` because `"value"` is not one of their series. The names
/// below are harvested from the dispatcher source itself
/// (`vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs`), and
/// an id that names an output this table gets wrong will fail LOUDLY with
/// `UnknownOutput` rather than vanishing.
pub const UNREGISTERED_MULTI_OUTPUTS: &[(&str, &[&str])] = &[
    // cpu_batch.rs: `compute_rolling_skewness_kurtosis_batch`
    ("rolling_skewness_kurtosis", &["skewness", "kurtosis"]),
    // cpu_batch.rs: `compute_rolling_z_score_trend_batch`
    ("rolling_z_score_trend", &["zscore", "momentum"]),
    // cpu_batch.rs: `compute_historical_volatility_percentile_batch`
    ("historical_volatility_percentile", &["hvp", "hvp_sma"]),
    // cpu_batch.rs: `compute_ehlers_data_sampling_relative_strength_indicator_batch`
    (
        "ehlers_data_sampling_relative_strength_indicator",
        &["ds_rsi", "original_rsi", "signal"],
    ),
    // cpu_batch.rs: `compute_ict_propulsion_block_batch` (twelve fields)
    (
        "ict_propulsion_block",
        &[
            "bullish_high",
            "bullish_low",
            "bullish_kind",
            "bullish_active",
            "bullish_mitigated",
            "bullish_new",
            "bearish_high",
            "bearish_low",
            "bearish_kind",
            "bearish_active",
            "bearish_mitigated",
            "bearish_new",
        ],
    ),
];

/// Every `output_id` this indicator must be dispatched with, in a stable order.
///
/// * a registered SINGLE-output indicator gets `[None]` — the library default;
/// * a registered MULTI-output indicator gets one `Some(id)` per declared
///   output, because vector-ta answers `output_id: None` for those with
///   `InvalidParam { key: "output_id" }`. This is the single mechanism that
///   `compute_single_indicator` (the chart endpoint) has always used and the
///   ALL_INDICATORS loop never did;
/// * an UNREGISTERED id gets its entry from [`UNREGISTERED_MULTI_OUTPUTS`], or
///   `[None]` (which `compute_cpu_batch` turns into the literal `"value"`).
pub fn output_ids_for(id: &str) -> Vec<Option<&'static str>> {
    if let Some(info) = get_indicator(id) {
        if info.outputs.len() <= 1 {
            return vec![None];
        }
        return info.outputs.iter().map(|o| Some(o.id)).collect();
    }
    if let Some((_, outs)) = UNREGISTERED_MULTI_OUTPUTS.iter().find(|(k, _)| *k == id) {
        return outs.iter().map(|o| Some(*o)).collect();
    }
    vec![None]
}

/// How many columns this id will attempt, used to plan the memory budget
/// BEFORE anything is allocated. Cheap: registry lookup only, no compute.
///
/// `pattern_recognition` is the one id whose column count is not its output
/// count: it declares a single `matrix` output and returns 62 x n booleans,
/// which `hpc_ta::pattern_matrix_columns` decomposes into one column per
/// candlestick pattern. Counting it as 1 would leave 61 columns — 514 MB at the
/// M5 store's depth — outside the never-OOM budget, i.e. peak memory would stop
/// being a function of what was planned. The count comes from the library's own
/// pattern list, not from a number written down here.
pub fn planned_output_count(id: &str) -> usize {
    if id == "pattern_recognition" {
        return vector_ta::indicators::pattern_recognition::list_patterns()
            .len()
            .max(1);
    }
    output_ids_for(id).len()
}

// ---------------------------------------------------------------------------
// Drop reasons.
// ---------------------------------------------------------------------------

/// Why one attempted column did not become a column.
///
/// Every discard on the feature-build data path must carry one of these. There
/// is deliberately no `Other`-without-detail and no `Unknown`: an unmatched
/// dispatch error still records its full `Display` text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DropReason {
    /// vector-ta panicked instead of returning `Err` (issue #212).
    KernelPanic,
    /// `IndicatorDispatchError::UnknownIndicator`
    UnknownIndicator,
    /// `IndicatorDispatchError::UnknownOutput`
    UnknownOutput,
    /// `IndicatorDispatchError::MissingRequiredInput`
    MissingRequiredInput,
    /// `IndicatorDispatchError::InvalidParam`
    InvalidParam,
    /// `IndicatorDispatchError::UnsupportedCapability`
    UnsupportedCapability,
    /// `IndicatorDispatchError::DataLengthMismatch`
    DataLengthMismatch,
    /// `IndicatorDispatchError::KernelUnavailable`
    KernelUnavailable,
    /// `IndicatorDispatchError::ComputeFailed`
    ComputeFailed,
    /// A dispatch error variant this enum does not name yet. The detail string
    /// still carries the full message, so it can never be a mystery.
    OtherDispatchError,
    /// The kernel returned fewer values than the frame has bars. Never padded:
    /// a short series is a build defect, and zero-padding it would hand the GA
    /// a real number it could threshold against.
    ShortSeries,
    /// The kernel returned MORE values than the frame has bars and the head was
    /// kept. Not a lost column — a lost TAIL — but still a discard, and the one
    /// place on the repaired path that used to do it with no counter and no
    /// log. (An exact integer multiple of the frame length is not truncated at
    /// all: that is a flattened multi-series and taking the head would
    /// mis-attribute one output's values to another's name, so it is a hard
    /// error inside `normalize_indicator_len`.)
    Truncated,
    /// Skipped before the call because the indicator's warmup exceeds the frame
    /// (the #212 pre-flight guard). Expected on short frames, a defect on long
    /// ones — which is why it is counted rather than `continue`d.
    PreflightWarmup,
    /// Not attempted: the hardware-derived vocabulary budget was already full.
    /// Peak memory is a function of the machine, so this is the never-OOM
    /// invariant doing its job — but it is still a discard and still counted.
    OverBudget,
}

impl DropReason {
    /// Stable snake_case label used as the log field and the summary key.
    pub fn label(self) -> &'static str {
        match self {
            DropReason::KernelPanic => "kernel_panic",
            DropReason::UnknownIndicator => "unknown_indicator",
            DropReason::UnknownOutput => "unknown_output",
            DropReason::MissingRequiredInput => "missing_required_input",
            DropReason::InvalidParam => "invalid_param",
            DropReason::UnsupportedCapability => "unsupported_capability",
            DropReason::DataLengthMismatch => "data_length_mismatch",
            DropReason::KernelUnavailable => "kernel_unavailable",
            DropReason::ComputeFailed => "compute_failed",
            DropReason::OtherDispatchError => "other_dispatch_error",
            DropReason::ShortSeries => "short_series",
            DropReason::Truncated => "truncated",
            DropReason::PreflightWarmup => "preflight_warmup",
            DropReason::OverBudget => "over_budget",
        }
    }

    /// Is this discard a property of the FRAME rather than of the indicator?
    ///
    /// The distinction decides whether the caller must emit an all-NaN
    /// placeholder column in the drop's place, and it is load-bearing:
    ///
    ///   * a CAPABILITY failure — no dispatch arm, no such output, an
    ///     unsupported kernel — is a property of the `(indicator, output)` pair
    ///     and happens identically on every frame. Emitting a placeholder for
    ///     it would manufacture a permanently-dead column AND blind the
    ///     output-level vocabulary floor, which detects a regression by the
    ///     ABSENCE of a name;
    ///   * a DATA failure — the frame is too short to warm the window, the
    ///     kernel could not compute on these values — is a property of THIS
    ///     frame, and every timeframe has a different length. Skipping it makes
    ///     the emitted column SET a function of the frame, so the per-timeframe
    ///     cube widths diverge and `lib.rs::try_assemble_cube_in_ram` refuses to
    ///     assemble — a run then falls through to the slower streaming disk path
    ///     with nothing but a debug line to say why. Measured: caught by
    ///     `cube_assembly_tests::ram_and_disk_cubes_are_identical` the moment the
    ///     base vocabulary went from 1 column to ~750.
    ///
    /// The drop is COUNTED either way. A placeholder is not a produced column:
    /// `produced()` is never called for one, so the vocabulary floor still sees
    /// the real number.
    pub fn is_frame_dependent(self) -> bool {
        match self {
            DropReason::PreflightWarmup
            | DropReason::DataLengthMismatch
            | DropReason::ComputeFailed
            | DropReason::KernelPanic
            | DropReason::ShortSeries => true,
            DropReason::UnknownIndicator
            | DropReason::UnknownOutput
            | DropReason::MissingRequiredInput
            | DropReason::InvalidParam
            | DropReason::UnsupportedCapability
            | DropReason::KernelUnavailable
            | DropReason::OtherDispatchError
            | DropReason::Truncated
            | DropReason::OverBudget => false,
        }
    }

    /// Classify a vector-ta dispatch error without losing its text.
    pub fn from_dispatch(e: &IndicatorDispatchError) -> Self {
        match e {
            IndicatorDispatchError::UnknownIndicator { .. } => DropReason::UnknownIndicator,
            IndicatorDispatchError::UnknownOutput { .. } => DropReason::UnknownOutput,
            IndicatorDispatchError::MissingRequiredInput { .. } => {
                DropReason::MissingRequiredInput
            }
            IndicatorDispatchError::InvalidParam { .. } => DropReason::InvalidParam,
            IndicatorDispatchError::UnsupportedCapability { .. } => {
                DropReason::UnsupportedCapability
            }
            IndicatorDispatchError::DataLengthMismatch { .. } => DropReason::DataLengthMismatch,
            IndicatorDispatchError::KernelUnavailable { .. } => DropReason::KernelUnavailable,
            IndicatorDispatchError::ComputeFailed { .. } => DropReason::ComputeFailed,
            _ => DropReason::OtherDispatchError,
        }
    }
}

// ---------------------------------------------------------------------------
// The ledger.
// ---------------------------------------------------------------------------

/// Per-reason accounting for one drop bucket.
#[derive(Clone, Debug, Default)]
struct Bucket {
    columns: usize,
    ids: BTreeSet<String>,
    /// First few `(column, detail)` pairs, so the log names real examples
    /// instead of only a number.
    examples: Vec<(String, String)>,
}

/// How many distinct examples each reason bucket keeps for the log line.
const EXAMPLES_PER_REASON: usize = 5;

/// A counted, reasoned record of everything one indicator pass produced and
/// everything it discarded.
///
/// Cheap to merge, so each rayon worker keeps its own and the parent folds
/// them — no shared mutex on the hot path.
#[derive(Clone, Debug, Default)]
pub struct IndicatorLedger {
    /// id -> number of columns it produced.
    produced: BTreeMap<String, usize>,
    /// reason -> bucket.
    drops: BTreeMap<DropReason, Bucket>,
    /// Ids in [`EXPECTED_NON_PRODUCING`] that nevertheless produced a column.
    /// A non-empty set means the exclusion table is stale.
    stale_exclusions: BTreeSet<String>,
    /// Columns whose values are bit-identical to an earlier column in the same
    /// pass. NOT removed — the column SET must stay a function of the id list
    /// and the registry, never of the frame, or the cube width would vary by
    /// timeframe. Reported so a period sweep that the indicator silently
    /// ignores shows up as duplication instead of as vocabulary.
    duplicate_columns: Vec<String>,
    /// Columns with no finite variation on this frame (all-NaN or constant).
    /// Also not removed, for the same width-invariance reason.
    degenerate_columns: Vec<String>,
}

impl IndicatorLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one produced column.
    pub fn produced(&mut self, id: &str) {
        *self.produced.entry(id.to_string()).or_insert(0) += 1;
    }

    /// Record one discard, with its reason and enough detail to act on.
    pub fn dropped(&mut self, id: &str, column: &str, reason: DropReason, detail: impl Into<String>) {
        let b = self.drops.entry(reason).or_default();
        b.columns += 1;
        b.ids.insert(id.to_string());
        if b.examples.len() < EXAMPLES_PER_REASON {
            b.examples.push((column.to_string(), detail.into()));
        }
    }

    /// Record that an id on the expected-to-fail list actually worked.
    pub fn stale_exclusion(&mut self, id: &str) {
        self.stale_exclusions.insert(id.to_string());
    }

    pub fn duplicate_column(&mut self, name: &str) {
        self.duplicate_columns.push(name.to_string());
    }

    pub fn degenerate_column(&mut self, name: &str) {
        self.degenerate_columns.push(name.to_string());
    }

    /// Fold another worker's ledger into this one.
    pub fn merge(&mut self, other: IndicatorLedger) {
        for (id, n) in other.produced {
            *self.produced.entry(id).or_insert(0) += n;
        }
        for (reason, b) in other.drops {
            let dst = self.drops.entry(reason).or_default();
            dst.columns += b.columns;
            dst.ids.extend(b.ids);
            for e in b.examples {
                if dst.examples.len() < EXAMPLES_PER_REASON {
                    dst.examples.push(e);
                }
            }
        }
        self.stale_exclusions.extend(other.stale_exclusions);
        self.duplicate_columns.extend(other.duplicate_columns);
        self.degenerate_columns.extend(other.degenerate_columns);
    }

    /// Number of distinct indicator ids that produced at least one column.
    pub fn producing_ids(&self) -> usize {
        self.produced.len()
    }

    /// Total columns produced.
    pub fn produced_columns(&self) -> usize {
        self.produced.values().sum()
    }

    /// Total columns discarded, all reasons.
    pub fn dropped_columns(&self) -> usize {
        self.drops.values().map(|b| b.columns).sum()
    }

    pub fn duplicate_count(&self) -> usize {
        self.duplicate_columns.len()
    }

    pub fn degenerate_count(&self) -> usize {
        self.degenerate_columns.len()
    }

    /// One INFO line for the pass, then one INFO line per drop reason naming
    /// the bucket size, the ids involved and up to five real examples.
    ///
    /// This is the thing that makes a regression from 800 columns to 66
    /// impossible to miss: the reason bucket that grew is printed by name with
    /// the exact dispatch message that caused it.
    pub fn log_summary(&self, stage: &str, rows: usize) {
        tracing::info!(
            target: "neoethos_data::indicator_ledger",
            stage,
            rows,
            producing_ids = self.producing_ids(),
            produced_columns = self.produced_columns(),
            dropped_columns = self.dropped_columns(),
            duplicate_columns = self.duplicate_count(),
            degenerate_columns = self.degenerate_count(),
            "indicator vocabulary census"
        );
        for (reason, b) in &self.drops {
            let examples = b
                .examples
                .iter()
                .map(|(c, d)| if d.is_empty() { c.clone() } else { format!("{c}: {d}") })
                .collect::<Vec<_>>()
                .join(" | ");
            tracing::info!(
                target: "neoethos_data::indicator_ledger",
                stage,
                reason = reason.label(),
                columns = b.columns,
                ids = b.ids.len(),
                %examples,
                "indicator columns discarded"
            );
        }
        if !self.stale_exclusions.is_empty() {
            tracing::warn!(
                target: "neoethos_data::indicator_ledger",
                stage,
                ids = %self.stale_exclusions.iter().cloned().collect::<Vec<_>>().join(","),
                "indicators on EXPECTED_NON_PRODUCING produced columns — the exclusion table in \
                 core::indicator_ledger is STALE and should be trimmed"
            );
        }
        if !self.duplicate_columns.is_empty() {
            tracing::warn!(
                target: "neoethos_data::indicator_ledger",
                stage,
                count = self.duplicate_columns.len(),
                names = %preview(&self.duplicate_columns),
                "columns are bit-identical to an earlier column — usually an indicator that \
                 IGNORES the swept parameter, so the sweep bought duplication rather than \
                 vocabulary. They are KEPT (the column set must not depend on the frame) but \
                 they carry no information."
            );
        }
        if !self.degenerate_columns.is_empty() {
            tracing::info!(
                target: "neoethos_data::indicator_ledger",
                stage,
                count = self.degenerate_columns.len(),
                names = %preview(&self.degenerate_columns),
                "columns have no finite variation on this frame (all-NaN or constant) — kept for \
                 width invariance, but they are ballast for any correlation-ranked prefilter"
            );
        }
    }

    /// Hard-fail when the produced vocabulary falls through the floor.
    ///
    /// `min_ids` and `min_columns` are MEASURED floors, not aspirations — see
    /// `hpc_ta::MIN_PRODUCING_INDICATOR_IDS`. The check is skipped on frames
    /// too short to warm the indicators up, because there a low count is the
    /// data's fault and not a regression; the caller decides that with
    /// `enforce`.
    ///
    /// # Why the floor is clamped by what the machine afforded
    ///
    /// The hardware vocabulary budget and this floor were added in the same
    /// change and were never exercised together, and they contradict each other
    /// on a real machine at real depth. Measured: 20.6 GB free, the M5 store's
    /// 1,054,320 bars, one f64 column = 8.43 MB, so `max_columns = 580` and
    /// `admit_indicators` admits **269** of the 342 ids. The absolute floor
    /// demands 280 producing ids, so `compute_classic_ta_columns` returned Err
    /// and discovery could not start at all — two safety mechanisms
    /// deadlocking, on the operator's own box.
    ///
    /// So the floor is `min(constant, what was attempted)`. A machine that
    /// cannot afford the vocabulary is a DIFFERENT INCIDENT from a dispatch
    /// regression, and the two must never share an error message: the first is
    /// a sizing fact already reported at WARN by
    /// `feature_budget::VocabularyBudget::log`, the second is the 341-column
    /// silent drop returning. `afforded_ids` / `afforded_columns` are what the
    /// budget admitted; pass `usize::MAX` for both when no budget applies.
    pub fn enforce_floor(
        &self,
        stage: &str,
        rows: usize,
        min_ids: usize,
        min_columns: usize,
        afforded_ids: usize,
        afforded_columns: usize,
    ) -> anyhow::Result<()> {
        let effective_ids = min_ids.min(afforded_ids);
        let effective_columns = min_columns.min(afforded_columns);
        let budget_bound = effective_ids < min_ids || effective_columns < min_columns;
        if self.producing_ids() >= effective_ids && self.produced_columns() >= effective_columns {
            return Ok(());
        }
        let worst = self
            .drops
            .iter()
            // OverBudget is the budget doing its job, not a dispatch failure —
            // naming it as "the largest drop bucket" would point the reader at
            // the wrong incident entirely.
            .filter(|(r, _)| **r != DropReason::OverBudget)
            .max_by_key(|(_, b)| b.columns)
            .map(|(r, b)| {
                let ex = b
                    .examples
                    .iter()
                    .map(|(c, d)| format!("{c} ({d})"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{} dropped {} columns across {} ids; e.g. {ex}", r.label(), b.columns, b.ids.len())
            })
            .unwrap_or_else(|| {
                "no drops were recorded at all besides the budget's own, which is itself a defect"
                    .into()
            });
        let sizing = if budget_bound {
            format!(
                " NOTE: the hardware vocabulary budget only admitted {afforded_ids} ids / \
                 {afforded_columns} columns on this machine, so the floor was clamped from \
                 {min_ids}/{min_columns} down to {effective_ids}/{effective_columns}. The pass \
                 failed even THAT, so this is a dispatch regression and not a sizing problem — \
                 see the feature_budget WARN line for the sizing."
            )
        } else {
            String::new()
        };
        anyhow::bail!(
            "INDICATOR VOCABULARY COLLAPSE at stage '{stage}' ({rows} bars): {} indicator ids \
             produced {} columns, below the floor of {effective_ids} ids / {effective_columns} \
             columns. This is the 341-silent-drop regression returning. Largest drop bucket: \
             {worst}.{sizing} Full per-reason census is on the INFO lines above \
             (target=neoethos_data::indicator_ledger).",
            self.producing_ids(),
            self.produced_columns(),
        )
    }
}

fn preview(names: &[String]) -> String {
    let head: Vec<&str> = names.iter().take(8).map(|s| s.as_str()).collect();
    if names.len() > head.len() {
        format!("{} … (+{} more)", head.join(","), names.len() - head.len())
    } else {
        head.join(",")
    }
}

/// FNV-1a over the raw bits, NaN-canonicalised so two all-NaN series hash the
/// same. f64 bits throughout — no narrowing anywhere on this path.
pub fn series_fingerprint(v: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in v {
        let bits = if x.is_nan() {
            0x7ff8_0000_0000_0000u64
        } else if x == 0.0 {
            // -0.0 and 0.0 must fingerprint identically.
            0u64
        } else {
            x.to_bits()
        };
        for b in bits.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    }
    h
}

/// Does this series carry any information at all on this frame?
pub fn has_finite_variation(v: &[f64]) -> bool {
    let mut first: Option<f64> = None;
    for &x in v {
        if !x.is_finite() {
            continue;
        }
        match first {
            None => first = Some(x),
            Some(f) => {
                if x != f {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_expected_non_producing_entry_carries_a_reason() {
        for (id, why) in EXPECTED_NON_PRODUCING {
            assert!(!id.is_empty(), "empty id in EXPECTED_NON_PRODUCING");
            assert!(
                why.len() > 20,
                "'{id}' is excluded without a real reason — a name with no reason is how the \
                 next 341-column drop starts"
            );
        }
    }

    #[test]
    fn every_expected_non_producing_output_names_a_real_declared_output_and_a_reason() {
        for (id, output, why) in EXPECTED_NON_PRODUCING_OUTPUTS {
            assert!(
                crate::core::all_indicators::ALL_INDICATORS.contains(id),
                "'{id}' has an output exclusion but is not in ALL_INDICATORS"
            );
            let info = get_indicator(id)
                .unwrap_or_else(|| panic!("'{id}' has an output exclusion but is unregistered"));
            assert!(
                info.outputs.iter().any(|o| o.id == *output),
                "'{id}.{output}' is excluded but the registry does not declare that output — the \
                 entry is dead and would hide a real absence"
            );
            assert!(
                why.len() > 20,
                "'{id}.{output}' is excluded without a real reason"
            );
        }
        assert_eq!(
            expected_non_producing_output("adaptive_bounds_rsi", "regime"),
            None,
            "an output that WORKS must never be on the exclusion list"
        );
    }

    #[test]
    fn expected_non_producing_ids_are_all_real_entries_of_all_indicators() {
        // A typo here would silently do nothing, which is precisely the class
        // of defect this module exists to prevent.
        for (id, _) in EXPECTED_NON_PRODUCING {
            assert!(
                crate::core::all_indicators::ALL_INDICATORS.contains(id),
                "'{id}' is on the exclusion list but is not in ALL_INDICATORS — the entry is dead"
            );
        }
    }

    #[test]
    fn unregistered_output_tables_name_indicators_that_are_really_unregistered() {
        for (id, outs) in UNREGISTERED_MULTI_OUTPUTS {
            assert!(!outs.is_empty(), "'{id}' override table has no output names");
            assert!(
                crate::core::all_indicators::ALL_INDICATORS.contains(id),
                "'{id}' has an output override but is not in ALL_INDICATORS"
            );
        }
    }

    #[test]
    fn multi_output_indicators_get_one_request_per_declared_output() {
        // The measured cause of 92 of the 342 failures: `output_id: None`
        // against a multi-output indicator. bollinger_bands has three.
        let outs = output_ids_for("bollinger_bands");
        assert!(
            outs.len() > 1,
            "bollinger_bands must enumerate its outputs, got {outs:?}"
        );
        assert!(outs.iter().all(|o| o.is_some()));
        // A single-output indicator still uses the library default.
        assert_eq!(output_ids_for("rsi"), vec![None]);
    }

    #[test]
    fn unregistered_multi_output_ids_resolve_through_the_override_table() {
        let outs = output_ids_for("rolling_skewness_kurtosis");
        assert_eq!(outs, vec![Some("skewness"), Some("kurtosis")]);
    }

    #[test]
    fn ledger_merges_and_reports_every_bucket() {
        let mut a = IndicatorLedger::new();
        a.produced("rsi");
        a.produced("rsi");
        a.dropped("sar", "sar", DropReason::UnknownIndicator, "unknown indicator: sar");
        let mut b = IndicatorLedger::new();
        b.produced("ema");
        b.dropped("rsx", "rsx", DropReason::UnsupportedCapability, "cpu_batch");
        a.merge(b);
        assert_eq!(a.producing_ids(), 2);
        assert_eq!(a.produced_columns(), 3);
        assert_eq!(a.dropped_columns(), 2);
    }

    #[test]
    fn floor_violation_is_a_hard_error_that_names_the_worst_bucket() {
        let mut l = IndicatorLedger::new();
        l.produced("ttm_trend");
        for i in 0..341 {
            l.dropped(
                "x",
                &format!("col{i}"),
                DropReason::InvalidParam,
                "output_id is required for multi-output indicators",
            );
        }
        let err = l
            .enforce_floor("base-vocabulary", 200_000, 280, 400, usize::MAX, usize::MAX)
            .expect_err("1 producing id must not pass a floor of 280");
        let msg = format!("{err}");
        assert!(msg.contains("INDICATOR VOCABULARY COLLAPSE"), "{msg}");
        assert!(msg.contains("invalid_param"), "{msg}");
        assert!(msg.contains("output_id is required"), "{msg}");
    }

    #[test]
    fn floor_passes_when_the_vocabulary_is_healthy() {
        let mut l = IndicatorLedger::new();
        for i in 0..300 {
            let id = format!("ind{i}");
            l.produced(&id);
            l.produced(&id);
        }
        l.enforce_floor("base-vocabulary", 200_000, 280, 400, usize::MAX, usize::MAX)
            .unwrap();
    }

    /// THE DEADLOCK THIS CLAMP EXISTS FOR.
    ///
    /// 20.6 GB free, 1,054,320 bars (the M5 store's real depth): the budget
    /// admits 269 ids / 580 columns, every one of them produces, and the
    /// absolute floor of 280/400 would still hard-error — on the operator's own
    /// machine, before discovery could start. The floor must be clamped by what
    /// the machine afforded.
    #[test]
    fn a_budget_truncated_vocabulary_that_fully_produced_is_not_a_collapse() {
        let mut l = IndicatorLedger::new();
        for i in 0..269 {
            let id = format!("ind{i}");
            l.produced(&id);
            l.produced(&id);
        }
        for i in 0..73 {
            l.dropped(
                &format!("deferred{i}"),
                &format!("deferred{i}"),
                DropReason::OverBudget,
                "vocabulary budget full",
            );
        }
        l.enforce_floor("base-vocabulary", 1_054_320, 280, 400, 269, 580)
            .expect(
                "269 of 269 afforded ids produced — that is the budget working, not a collapse",
            );
    }

    /// …but a clamped floor is still a floor: a real dispatch regression inside
    /// the admitted set must still be a hard error, and the message must say
    /// which incident it is.
    #[test]
    fn a_clamped_floor_still_catches_a_dispatch_regression_and_names_it_as_one() {
        let mut l = IndicatorLedger::new();
        l.produced("ttm_trend");
        for i in 0..268 {
            l.dropped(
                "x",
                &format!("col{i}"),
                DropReason::InvalidParam,
                "output_id is required for multi-output indicators",
            );
        }
        for i in 0..73 {
            l.dropped(
                &format!("deferred{i}"),
                &format!("deferred{i}"),
                DropReason::OverBudget,
                "vocabulary budget full",
            );
        }
        let err = l
            .enforce_floor("base-vocabulary", 1_054_320, 280, 400, 269, 580)
            .expect_err("1 producing id of 269 afforded is a collapse at any clamp");
        let msg = format!("{err}");
        assert!(msg.contains("clamped from 280/400 down to 269/400"), "{msg}");
        // The budget's own drops must NOT be blamed for the collapse.
        assert!(msg.contains("invalid_param"), "{msg}");
        assert!(!msg.contains("Largest drop bucket: over_budget"), "{msg}");
    }

    #[test]
    fn fingerprint_is_stable_and_distinguishes_real_differences() {
        assert_eq!(series_fingerprint(&[1.0, 2.0]), series_fingerprint(&[1.0, 2.0]));
        assert_ne!(series_fingerprint(&[1.0, 2.0]), series_fingerprint(&[1.0, 2.5]));
        // NaN canonicalisation: two all-NaN series of the same length collide,
        // which is what makes duplicate detection catch dead sweeps.
        assert_eq!(
            series_fingerprint(&[f64::NAN, f64::NAN]),
            series_fingerprint(&[f64::NAN, f64::NAN])
        );
        // -0.0 must not read as a different column from 0.0.
        assert_eq!(series_fingerprint(&[-0.0]), series_fingerprint(&[0.0]));
    }

    #[test]
    fn variation_detector_ignores_the_nan_warmup_prefix() {
        assert!(!has_finite_variation(&[f64::NAN, f64::NAN]));
        assert!(!has_finite_variation(&[f64::NAN, 3.0, 3.0]));
        assert!(has_finite_variation(&[f64::NAN, 3.0, 3.5]));
    }
}
