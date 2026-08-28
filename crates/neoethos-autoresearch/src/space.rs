//! Axis A — the search-configuration space, and the frozen-field wall.
//!
//! This module owns three things and nothing else:
//!
//! 1. **The declared axis-A factor set** (`FactorId`, §6 of
//!    `docs/autoresearch-loop.md`) — eleven categorical factors with 3–5 levels
//!    each, every level a *named constant in a table in this file*. There is no
//!    constructor that takes a free-form value: `SearchConfigDelta`'s only field
//!    is private and its only constructor validates every level index against
//!    the factor's arity. A proposer that cannot express an arbitrary
//!    configuration cannot express an arbitrary objective either.
//! 2. **`FROZEN_FIELDS`** — every `DiscoveryConfig` field the loop may never
//!    write: the goal mirrors, the whole cost model, the risk band, and the
//!    validation geometry. Disjointness from everything `apply` writes is
//!    asserted by a unit test over the *same tables that drive `apply`*, and
//!    then again at runtime by [`money_path_audit`], which diffs the produced
//!    config against the reference field by field. A declaration can drift from
//!    an implementation; a diff cannot.
//! 3. **`Capabilities`** — what the *compiled* `neoethos-search` can actually
//!    express. Every knob the design needs but the search does not yet have is
//!    behind a cargo feature whose only effect, when enabled, is to compile a
//!    probe that reads the field. So the feature cannot be turned on before the
//!    field exists: the crate stops compiling and the error names the field.
//!    With the feature off, the level or variant that needs it is **excluded
//!    from the drawable space and named in the space report** — never drawn and
//!    silently reduced to the default, which would make the loop appear to run
//!    a variant it is not running.
//!
//! The distinction this file exists to make mechanical:
//!
//! > The loop may vary **what the SEARCH refuses**. It may never vary **what
//! > the JUDGE refuses**, what a trade **costs**, or what the **goal** is.

use serde::{Deserialize, Serialize};

use neoethos_search::discovery::DiscoveryConfig;
use neoethos_search::orchestration::StreamingPlan;

// ───────────────────────────────────────────────────────────────────────────
// Constants (design §15, the proposer's half). The judge's and the shuffle
// control's constants are NOT here — they belong to their owners' modules, and
// duplicating a threshold is how two of them end up disagreeing.
// ───────────────────────────────────────────────────────────────────────────

/// One sweep = 100 searches. The operator's own unit.
pub const SWEEP_SEARCHES: usize = 100;
/// Slots per sweep drawn uniformly at random over the whole space, forever.
/// **Never annealed** — see [`crate::proposer`] for why.
pub const EXPLORATION_FLOOR: usize = 25;
/// Trust region: at most this many factors may differ from the parent.
pub const MAX_DELTA_FACTORS: usize = 3;
/// Sweeps' worth of draws each axis-B variant needs before U4 may be claimed.
pub const B_MIN_DRAWS: usize = 3;
/// Consecutive colliding draws before the space counts as enumerated (U5).
pub const PROPOSER_RETRIES: usize = 64;

// ───────────────────────────────────────────────────────────────────────────
// Capability probes — absence fails at COMPILE time, and names the field.
// ───────────────────────────────────────────────────────────────────────────

// Each probe is dead code by construction; dead code is still type-checked, so
// enabling the feature against a `neoethos-search` that lacks the field is a
// compile error naming that field. That is the whole mechanism.

/// `docs/autoresearch-loop.md` §14.6 — `StreamingPlan::start_cursor`.
#[cfg(feature = "search-streaming-start-cursor")]
#[allow(dead_code)]
fn probe_streaming_start_cursor(plan: &StreamingPlan) -> usize {
    plan.start_cursor
}

/// `docs/autoresearch-loop.md` §14.3 — `ResolvedConfigStamp::ga_seed`.
#[cfg(feature = "search-stamp-ga-seed")]
#[allow(dead_code)]
fn probe_stamp_ga_seed(stamp: &neoethos_search::run_identity::ResolvedConfigStamp) -> u64 {
    stamp.ga_seed
}

/// What the compiled `neoethos-search` can express.
///
/// The **only** public constructor is [`Capabilities::compiled`], which reads
/// `cfg!(feature = …)` and nothing else. The loop cannot declare a capability
/// it does not have, because the declaration is not an argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities {
    /// A sweep can start the streaming working set at a chosen cursor.
    pub streaming_start_cursor: bool,
    /// `ResolvedConfigStamp` includes the GA seed, so replicates get distinct
    /// `config_hash`es. When false the identity is the pair
    /// `(config_hash, replicate_seed)` and the session header must say so.
    pub stamp_ga_seed: bool,
    /// `DiscoveryConfig` carries the triple-barrier label horizon (B4).
    pub label_horizon: bool,
    /// The GA's fitness table is selectable (B0, and B1's in-market term).
    pub fitness_table: bool,
    /// Candidates can be scored at the pessimistic edge of the cost band (B3).
    pub cost_edge_scoring: bool,
    /// A conditioning set can be declared before the run (B2).
    pub conditioning_set: bool,
    /// Net-per-bar-in-market is available as the GA objective (B1).
    pub in_market_fitness: bool,
    /// The expectancy t-statistic is available as the GA objective (B7).
    pub t_stat_objective: bool,
    /// The portfolio, after correlation pruning, is the scored unit (B8).
    pub portfolio_objective: bool,
}

impl Capabilities {
    /// Read from the compiled feature set. The only production constructor.
    pub fn compiled() -> Self {
        Self {
            streaming_start_cursor: cfg!(feature = "search-streaming-start-cursor"),
            stamp_ga_seed: cfg!(feature = "search-stamp-ga-seed"),
            label_horizon: cfg!(feature = "search-label-horizon"),
            fitness_table: cfg!(feature = "search-fitness-table"),
            cost_edge_scoring: cfg!(feature = "search-cost-edge-scoring"),
            conditioning_set: cfg!(feature = "search-conditioning-set"),
            in_market_fitness: cfg!(feature = "search-in-market-fitness"),
            t_stat_objective: cfg!(feature = "search-t-stat-objective"),
            portfolio_objective: cfg!(feature = "search-portfolio-objective"),
        }
    }

    /// Every capability on. Crate-private and test-only **on purpose**: a
    /// production path that could fabricate a capability would be the silent
    /// no-op this design exists to prevent.
    #[cfg(test)]
    pub(crate) fn all_for_tests() -> Self {
        Self {
            streaming_start_cursor: true,
            stamp_ga_seed: true,
            label_horizon: true,
            fitness_table: true,
            cost_edge_scoring: true,
            conditioning_set: true,
            in_market_fitness: true,
            t_stat_objective: true,
            portfolio_objective: true,
        }
    }
}

/// One knob the design needs that the compiled search may not have.
///
/// Carried in the space report so a reader can see, per level and per variant,
/// exactly which symbol is missing and where the edit is written down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequiredKnob {
    StreamingStartCursor,
    StampGaSeed,
    LabelHorizon,
    FitnessTable,
    CostEdgeScoring,
    ConditioningSet,
    InMarketFitness,
    TStatObjective,
    PortfolioObjective,
}

impl RequiredKnob {
    pub fn available(self, caps: &Capabilities) -> bool {
        match self {
            Self::StreamingStartCursor => caps.streaming_start_cursor,
            Self::StampGaSeed => caps.stamp_ga_seed,
            Self::LabelHorizon => caps.label_horizon,
            Self::FitnessTable => caps.fitness_table,
            Self::CostEdgeScoring => caps.cost_edge_scoring,
            Self::ConditioningSet => caps.conditioning_set,
            Self::InMarketFitness => caps.in_market_fitness,
            Self::TStatObjective => caps.t_stat_objective,
            Self::PortfolioObjective => caps.portfolio_objective,
        }
    }

    /// The symbol whose absence excludes this knob, named exactly.
    pub fn symbol(self) -> &'static str {
        match self {
            Self::StreamingStartCursor => {
                "neoethos_search::orchestration::StreamingPlan::start_cursor"
            }
            Self::StampGaSeed => "neoethos_search::run_identity::ResolvedConfigStamp::ga_seed",
            Self::LabelHorizon => {
                "neoethos_search::discovery::DiscoveryConfig::label_max_hold_bars"
            }
            Self::FitnessTable => "neoethos_search::discovery::DiscoveryConfig::fitness_table",
            Self::CostEdgeScoring => {
                "neoethos_search::discovery::DiscoveryConfig::score_at_cost_band_edge"
            }
            Self::ConditioningSet => {
                "neoethos_search::discovery::DiscoveryConfig::conditioning_set"
            }
            Self::InMarketFitness => "neoethos_search::scoring::FitnessTable::NetPerBarInMarket",
            Self::TStatObjective => "neoethos_search::scoring::FitnessTable::ExpectancyTStat",
            Self::PortfolioObjective => {
                "neoethos_search::discovery::DiscoveryConfig::score_portfolio_after_pruning"
            }
        }
    }

    /// The cargo feature that turns the knob on once the symbol lands.
    pub fn feature(self) -> &'static str {
        match self {
            Self::StreamingStartCursor => "search-streaming-start-cursor",
            Self::StampGaSeed => "search-stamp-ga-seed",
            Self::LabelHorizon => "search-label-horizon",
            Self::FitnessTable => "search-fitness-table",
            Self::CostEdgeScoring => "search-cost-edge-scoring",
            Self::ConditioningSet => "search-conditioning-set",
            Self::InMarketFitness => "search-in-market-fitness",
            Self::TStatObjective => "search-t-stat-objective",
            Self::PortfolioObjective => "search-portfolio-objective",
        }
    }

    /// Where the exact edit is written out.
    pub fn required_elsewhere_section(self) -> &'static str {
        match self {
            Self::StampGaSeed => "docs/autoresearch-loop.md §14.3",
            Self::StreamingStartCursor => "docs/autoresearch-loop.md §14.6",
            _ => "docs/autoresearch-loop.md §14.7",
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// The frozen wall.
// ───────────────────────────────────────────────────────────────────────────

/// Fields whose value decides what a trade EARNS or what it COSTS, or what the
/// goal is. A proposer that could move one of these could reach any number.
pub const MONEY_PATH_FIELDS: &[&str] = &[
    "evaluation_symbol",
    "evaluation_account_currency",
    "evaluation_spread_pips",
    "evaluation_commission_per_trade",
    "session_spread_pips",
    "cost_band_pips",
    "swap_long_pips_per_day",
    "swap_short_pips_per_day",
    "initial_balance",
    "risk_per_trade_min",
    "risk_per_trade_max",
    "risky_risk_band",
    "prop_firm_risk_band",
    "risky_start_balance",
    "risky_target_balance",
    "risky_horizon_days",
    "prop_firm_min_pass_rate",
    "max_regime_loss_pct",
    "mode",
    // The one gate that says "the average trade must make money". Deliberately
    // NOT in the refusal vector (§7.9): it is the floor under everything else.
    "target_profile.min_net_expectancy_per_trade",
];

/// Fields that belong to the JUDGE. A loop that could widen its own folds could
/// reach any number too — and by a quieter route.
pub const VALIDATION_GEOMETRY_FIELDS: &[&str] = &[
    "enable_cpcv",
    "cpcv_n_splits",
    "cpcv_n_test_groups",
    "cpcv_embargo_pct",
    "cpcv_purge_pct",
    "cpcv_min_phi",
    "cpcv_max_rows",
    "walkforward_splits",
    "embargo_minutes",
    "require_walkforward_for_export",
    "mc_runs",
    "mc_min_profitable",
    "sensitivity_spread_pips",
    "sensitivity_commission_per_lot",
];

/// The union. `SearchConfigDelta::apply` and every objective override are
/// asserted disjoint from this, by test and again at runtime.
///
/// `max_pbo` is **not** here, and that is deliberate and load-bearing: it is the
/// per-candidate CPCV number the SEARCH refuses on (`§7.9`,
/// `candidate_pbo_cap`), a different object from the judge's session-level CSCV
/// PBO, which lives in the judge's own thresholds and is not a `DiscoveryConfig`
/// field at all.
pub const FROZEN_FIELDS: &[&str] = &[
    // money path
    "evaluation_symbol",
    "evaluation_account_currency",
    "evaluation_spread_pips",
    "evaluation_commission_per_trade",
    "session_spread_pips",
    "cost_band_pips",
    "swap_long_pips_per_day",
    "swap_short_pips_per_day",
    "initial_balance",
    "risk_per_trade_min",
    "risk_per_trade_max",
    "risky_risk_band",
    "prop_firm_risk_band",
    "risky_start_balance",
    "risky_target_balance",
    "risky_horizon_days",
    "prop_firm_min_pass_rate",
    "max_regime_loss_pct",
    "mode",
    "target_profile.min_net_expectancy_per_trade",
    // validation geometry
    "enable_cpcv",
    "cpcv_n_splits",
    "cpcv_n_test_groups",
    "cpcv_embargo_pct",
    "cpcv_purge_pct",
    "cpcv_min_phi",
    "cpcv_max_rows",
    "walkforward_splits",
    "embargo_minutes",
    "require_walkforward_for_export",
    "mc_runs",
    "mc_min_profitable",
    "sensitivity_spread_pips",
    "sensitivity_commission_per_lot",
];

/// A field the delta wrote that it may not write, or a field it moved without
/// declaring. Both are the same defect: the run is not the proposal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoneyPathViolation {
    pub field: String,
    pub reference: String,
    pub proposed: String,
}

/// Diff the produced config against the reference over every frozen field.
///
/// This is the mechanical half of non-negotiable #1. The declaration tables can
/// drift from the code; a field-by-field diff of the actual structs cannot.
/// Floats are compared by bit pattern so a `NaN` cannot slip through as "equal
/// to nothing, therefore fine".
pub fn money_path_audit(
    reference: &DiscoveryConfig,
    proposed: &DiscoveryConfig,
) -> Result<(), Vec<MoneyPathViolation>> {
    let mut out = Vec::new();

    macro_rules! cmp {
        ($field:literal, $a:expr, $b:expr) => {
            if $a != $b {
                out.push(MoneyPathViolation {
                    field: $field.to_string(),
                    reference: format!("{:?}", $a),
                    proposed: format!("{:?}", $b),
                });
            }
        };
    }
    macro_rules! cmpf {
        ($field:literal, $a:expr, $b:expr) => {
            if $a.to_bits() != $b.to_bits() {
                out.push(MoneyPathViolation {
                    field: $field.to_string(),
                    reference: format!("{:?}", $a),
                    proposed: format!("{:?}", $b),
                });
            }
        };
    }

    cmp!(
        "evaluation_symbol",
        reference.evaluation_symbol,
        proposed.evaluation_symbol
    );
    cmp!(
        "evaluation_account_currency",
        reference.evaluation_account_currency,
        proposed.evaluation_account_currency
    );
    cmpf!(
        "evaluation_spread_pips",
        reference.evaluation_spread_pips,
        proposed.evaluation_spread_pips
    );
    cmpf!(
        "evaluation_commission_per_trade",
        reference.evaluation_commission_per_trade,
        proposed.evaluation_commission_per_trade
    );
    cmp!(
        "session_spread_pips",
        reference.session_spread_pips.map(|v| v.map(f64::to_bits)),
        proposed.session_spread_pips.map(|v| v.map(f64::to_bits))
    );
    cmp!(
        "cost_band_pips",
        reference
            .cost_band_pips
            .map(|(a, b)| (a.to_bits(), b.to_bits())),
        proposed
            .cost_band_pips
            .map(|(a, b)| (a.to_bits(), b.to_bits()))
    );
    cmpf!(
        "swap_long_pips_per_day",
        reference.swap_long_pips_per_day,
        proposed.swap_long_pips_per_day
    );
    cmpf!(
        "swap_short_pips_per_day",
        reference.swap_short_pips_per_day,
        proposed.swap_short_pips_per_day
    );
    cmpf!(
        "initial_balance",
        reference.initial_balance,
        proposed.initial_balance
    );
    cmpf!(
        "risk_per_trade_min",
        reference.risk_per_trade_min,
        proposed.risk_per_trade_min
    );
    cmpf!(
        "risk_per_trade_max",
        reference.risk_per_trade_max,
        proposed.risk_per_trade_max
    );
    cmp!(
        "risky_risk_band",
        reference
            .risky_risk_band
            .map(|(a, b)| (a.to_bits(), b.to_bits())),
        proposed
            .risky_risk_band
            .map(|(a, b)| (a.to_bits(), b.to_bits()))
    );
    cmp!(
        "prop_firm_risk_band",
        reference
            .prop_firm_risk_band
            .map(|(a, b)| (a.to_bits(), b.to_bits())),
        proposed
            .prop_firm_risk_band
            .map(|(a, b)| (a.to_bits(), b.to_bits()))
    );
    cmpf!(
        "risky_start_balance",
        reference.risky_start_balance,
        proposed.risky_start_balance
    );
    cmpf!(
        "risky_target_balance",
        reference.risky_target_balance,
        proposed.risky_target_balance
    );
    cmpf!(
        "risky_horizon_days",
        reference.risky_horizon_days,
        proposed.risky_horizon_days
    );
    cmpf!(
        "prop_firm_min_pass_rate",
        reference.prop_firm_min_pass_rate,
        proposed.prop_firm_min_pass_rate
    );
    cmpf!(
        "max_regime_loss_pct",
        reference.max_regime_loss_pct,
        proposed.max_regime_loss_pct
    );
    cmp!(
        "mode",
        format!("{:?}", reference.mode),
        format!("{:?}", proposed.mode)
    );
    cmpf!(
        "target_profile.min_net_expectancy_per_trade",
        reference.target_profile.min_net_expectancy_per_trade,
        proposed.target_profile.min_net_expectancy_per_trade
    );

    cmp!("enable_cpcv", reference.enable_cpcv, proposed.enable_cpcv);
    cmp!(
        "cpcv_n_splits",
        reference.cpcv_n_splits,
        proposed.cpcv_n_splits
    );
    cmp!(
        "cpcv_n_test_groups",
        reference.cpcv_n_test_groups,
        proposed.cpcv_n_test_groups
    );
    cmpf!(
        "cpcv_embargo_pct",
        reference.cpcv_embargo_pct,
        proposed.cpcv_embargo_pct
    );
    cmpf!(
        "cpcv_purge_pct",
        reference.cpcv_purge_pct,
        proposed.cpcv_purge_pct
    );
    cmpf!(
        "cpcv_min_phi",
        reference.cpcv_min_phi,
        proposed.cpcv_min_phi
    );
    cmp!(
        "cpcv_max_rows",
        reference.cpcv_max_rows,
        proposed.cpcv_max_rows
    );
    cmp!(
        "walkforward_splits",
        reference.walkforward_splits,
        proposed.walkforward_splits
    );
    cmp!(
        "embargo_minutes",
        reference.embargo_minutes,
        proposed.embargo_minutes
    );
    cmp!(
        "require_walkforward_for_export",
        reference.require_walkforward_for_export,
        proposed.require_walkforward_for_export
    );
    cmp!("mc_runs", reference.mc_runs, proposed.mc_runs);
    cmp!(
        "mc_min_profitable",
        reference.mc_min_profitable,
        proposed.mc_min_profitable
    );
    cmpf!(
        "sensitivity_spread_pips",
        reference.sensitivity_spread_pips,
        proposed.sensitivity_spread_pips
    );
    cmpf!(
        "sensitivity_commission_per_lot",
        reference.sensitivity_commission_per_lot,
        proposed.sensitivity_commission_per_lot
    );

    if out.is_empty() { Ok(()) } else { Err(out) }
}

// ───────────────────────────────────────────────────────────────────────────
// Timeframes.
// ───────────────────────────────────────────────────────────────────────────

/// The canonical timeframe set (2026-08-09 decision: native from the server,
/// never resampled). Anything else is refused by name rather than guessed at.
pub const CANONICAL_TIMEFRAMES: &[(&str, u32)] = &[
    ("M1", 1),
    ("M3", 3),
    ("M5", 5),
    ("M15", 15),
    ("M30", 30),
    ("H1", 60),
    ("H4", 240),
    ("D1", 1440),
];

/// Minutes per bar for a canonical timeframe label, or `None` — never a guess.
pub fn timeframe_minutes(label: &str) -> Option<u32> {
    CANONICAL_TIMEFRAMES
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(label))
        .map(|(_, m)| *m)
}

// ───────────────────────────────────────────────────────────────────────────
// The factors.
// ───────────────────────────────────────────────────────────────────────────

/// Where in the (indicator, period) space a sweep's working set starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CursorPolicy {
    /// Start at 0 — the only policy today's `run_streaming_working_set`
    /// expresses.
    Start,
    /// Resume from the cursor the parent sweep stopped at.
    Resumed,
    /// A random offset into the space, so the loop does not re-search the
    /// prefix of the space forever.
    RandomOffset,
}

pub const POPULATION_LEVELS: &[usize] = &[512, 2048, 4096, 8192];
pub const GENERATION_LEVELS: &[usize] = &[20, 50, 100, 200];
pub const GENE_WIDTH_LEVELS: &[usize] = &[4, 8, 12];
pub const PREFILTER_WIDTH_LEVELS: &[usize] = &[120, 240, 480, 960];
pub const PREFILTER_INSAMPLE_FRAC_LEVELS: &[f64] = &[0.60, 0.70, 0.80];
pub const BASE_TIMEFRAME_LEVELS: &[&str] = &["M5", "M15", "M30", "H1"];
/// H4 is deliberately absent as a lane on its own (§6): zero of 1,855 H4
/// columns clear a Bonferroni bar on their own effective sample size. It is
/// reachable only inside a multi-lane level, and only when the label horizon
/// spans at least one lane candle — see [`lane_horizon_conflict`].
pub const HIGHER_LANE_LEVELS: &[&[&str]] = &[&[], &["H1"], &["H1", "H4"], &["H1", "D1"]];
/// `0` = a single whole-vocabulary pass; anything else = stream that many
/// batches. The batch WIDTH is never a level — it comes from free RAM.
pub const STREAMING_BATCH_LEVELS: &[usize] = &[0, 4, 12, 32];
pub const CURSOR_LEVELS: &[CursorPolicy] = &[
    CursorPolicy::Start,
    CursorPolicy::Resumed,
    CursorPolicy::RandomOffset,
];
pub const CANDIDATE_COUNT_LEVELS: &[usize] = &[64, 256, 1024];
pub const PORTFOLIO_SHAPE_LEVELS: &[(usize, f64)] = &[(1, 1.0), (5, 0.7), (12, 0.5)];

/// The eleven axis-A factors, in declaration order. The order is part of the
/// wire format (`SearchConfigDelta` is an array indexed by it), so appending is
/// safe and reordering is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactorId {
    Population,
    Generations,
    GeneWidth,
    PrefilterWidth,
    PrefilterInsampleFrac,
    BaseTimeframe,
    HigherLanes,
    StreamingBatches,
    WorkingSetCursor,
    CandidateCount,
    PortfolioShape,
}

pub const FACTOR_COUNT: usize = 11;

impl FactorId {
    pub const ALL: [FactorId; FACTOR_COUNT] = [
        FactorId::Population,
        FactorId::Generations,
        FactorId::GeneWidth,
        FactorId::PrefilterWidth,
        FactorId::PrefilterInsampleFrac,
        FactorId::BaseTimeframe,
        FactorId::HigherLanes,
        FactorId::StreamingBatches,
        FactorId::WorkingSetCursor,
        FactorId::CandidateCount,
        FactorId::PortfolioShape,
    ];

    pub fn index(self) -> usize {
        match self {
            Self::Population => 0,
            Self::Generations => 1,
            Self::GeneWidth => 2,
            Self::PrefilterWidth => 3,
            Self::PrefilterInsampleFrac => 4,
            Self::BaseTimeframe => 5,
            Self::HigherLanes => 6,
            Self::StreamingBatches => 7,
            Self::WorkingSetCursor => 8,
            Self::CandidateCount => 9,
            Self::PortfolioShape => 10,
        }
    }

    pub fn arity(self) -> usize {
        match self {
            Self::Population => POPULATION_LEVELS.len(),
            Self::Generations => GENERATION_LEVELS.len(),
            Self::GeneWidth => GENE_WIDTH_LEVELS.len(),
            Self::PrefilterWidth => PREFILTER_WIDTH_LEVELS.len(),
            Self::PrefilterInsampleFrac => PREFILTER_INSAMPLE_FRAC_LEVELS.len(),
            Self::BaseTimeframe => BASE_TIMEFRAME_LEVELS.len(),
            Self::HigherLanes => HIGHER_LANE_LEVELS.len(),
            Self::StreamingBatches => STREAMING_BATCH_LEVELS.len(),
            Self::WorkingSetCursor => CURSOR_LEVELS.len(),
            Self::CandidateCount => CANDIDATE_COUNT_LEVELS.len(),
            Self::PortfolioShape => PORTFOLIO_SHAPE_LEVELS.len(),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Population => "population",
            Self::Generations => "generations",
            Self::GeneWidth => "gene_width",
            Self::PrefilterWidth => "prefilter_width",
            Self::PrefilterInsampleFrac => "prefilter_insample_frac",
            Self::BaseTimeframe => "base_timeframe",
            Self::HigherLanes => "higher_lanes",
            Self::StreamingBatches => "streaming_batches",
            Self::WorkingSetCursor => "working_set_cursor",
            Self::CandidateCount => "candidate_count",
            Self::PortfolioShape => "portfolio_shape",
        }
    }

    /// The `DiscoveryConfig` fields this factor writes — the *declared* half of
    /// the frozen-field wall. [`money_path_audit`] is the enforced half.
    pub fn writes(self) -> &'static [&'static str] {
        match self {
            // `population_auto` is written alongside `population`: a level that
            // set the population and left auto-sizing on would not be the level
            // it says it is.
            Self::Population => &["population", "population_auto"],
            Self::Generations => &["generations"],
            Self::GeneWidth => &["max_indicators"],
            Self::PrefilterWidth => &["runtime_overrides.prefilter_top_k"],
            Self::PrefilterInsampleFrac => &["runtime_overrides.prefilter_insample_frac"],
            Self::BaseTimeframe => &["timeframe_label"],
            Self::HigherLanes => &["higher_timeframes"],
            // The plan is handed to the runner beside the config; it is not a
            // `DiscoveryConfig` field, and the batch WIDTH is never touched.
            Self::StreamingBatches => &["streaming_plan.max_batches"],
            Self::WorkingSetCursor => &["streaming_plan.start_cursor"],
            Self::CandidateCount => &["candidate_count"],
            Self::PortfolioShape => &["portfolio_size", "corr_threshold"],
        }
    }

    /// A human-readable name for one level, for the census and the report.
    pub fn level_label(self, level: u8) -> String {
        let l = level as usize;
        match self {
            Self::Population => POPULATION_LEVELS.get(l).map(|v| v.to_string()),
            Self::Generations => GENERATION_LEVELS.get(l).map(|v| v.to_string()),
            Self::GeneWidth => GENE_WIDTH_LEVELS.get(l).map(|v| v.to_string()),
            Self::PrefilterWidth => PREFILTER_WIDTH_LEVELS.get(l).map(|v| v.to_string()),
            Self::PrefilterInsampleFrac => PREFILTER_INSAMPLE_FRAC_LEVELS
                .get(l)
                .map(|v| format!("{v:.2}")),
            Self::BaseTimeframe => BASE_TIMEFRAME_LEVELS.get(l).map(|v| v.to_string()),
            Self::HigherLanes => HIGHER_LANE_LEVELS.get(l).map(|v| {
                if v.is_empty() {
                    "none".to_string()
                } else {
                    v.join("+")
                }
            }),
            Self::StreamingBatches => STREAMING_BATCH_LEVELS.get(l).map(|v| {
                if *v == 0 {
                    "single_pass".to_string()
                } else {
                    v.to_string()
                }
            }),
            Self::WorkingSetCursor => CURSOR_LEVELS.get(l).map(|v| format!("{v:?}")),
            Self::CandidateCount => CANDIDATE_COUNT_LEVELS.get(l).map(|v| v.to_string()),
            Self::PortfolioShape => PORTFOLIO_SHAPE_LEVELS
                .get(l)
                .map(|(n, c)| format!("{n}@corr{c:.2}")),
        }
        .unwrap_or_else(|| format!("<out-of-range:{level}>"))
    }

    /// The knob a level needs, when it needs one the search may not have.
    pub fn level_requires(self, level: u8) -> Option<RequiredKnob> {
        match self {
            Self::WorkingSetCursor => match CURSOR_LEVELS.get(level as usize) {
                Some(CursorPolicy::Start) | None => None,
                Some(_) => Some(RequiredKnob::StreamingStartCursor),
            },
            _ => None,
        }
    }

    /// Whether this level can be drawn at all under `caps`.
    pub fn level_available(self, level: u8, caps: &Capabilities) -> bool {
        match self.level_requires(level) {
            None => true,
            Some(k) => k.available(caps),
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// The delta.
// ───────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpaceError {
    /// A level index outside the factor's declared arity. There is no clamping:
    /// a proposal the space cannot express is refused, not rounded into it.
    LevelOutOfRange {
        factor: FactorId,
        level: u8,
        arity: usize,
    },
    /// A level whose knob the compiled search does not have.
    LevelUnavailable {
        factor: FactorId,
        level: u8,
        knob: RequiredKnob,
    },
    /// A timeframe label outside the canonical set.
    UnknownTimeframe { label: String },
    /// A higher lane whose candle is longer than the label horizon: the feature
    /// would be asked to predict an outcome shorter than its own bar.
    LaneHorizonConflict {
        lane: String,
        lane_bars_in_base: u32,
        horizon_bars: usize,
    },
    /// **U5.** The reachable space is enumerated: every uniform draw collided
    /// with a configuration already run, so this slot had no unexplored point
    /// left to draw.
    ///
    /// It is a `SpaceError` and not a proposer bookkeeping flag because that is
    /// exactly what it is — a statement about the SPACE, not about the sampler.
    /// Carrying it here means the proposer emits one refusal record **per
    /// abandoned slot**, so the slots a shortened sweep never drew are counted
    /// in the journal and named with the reason, instead of vanishing into the
    /// difference between `SWEEP_SEARCHES` and `proposals.len()`.
    SpaceEnumerated {
        slot: usize,
        drawn: usize,
        of: usize,
        consecutive_collisions: usize,
    },
}

impl std::fmt::Display for SpaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LevelOutOfRange {
                factor,
                level,
                arity,
            } => write!(
                f,
                "axis-A factor `{}` has {arity} levels; level {level} does not exist",
                factor.label()
            ),
            Self::LevelUnavailable {
                factor,
                level,
                knob,
            } => write!(
                f,
                "axis-A factor `{}` level `{}` needs `{}` (cargo feature `{}`, edit in {}), \
                 which the compiled neoethos-search does not provide",
                factor.label(),
                factor.level_label(*level),
                knob.symbol(),
                knob.feature(),
                knob.required_elsewhere_section()
            ),
            Self::UnknownTimeframe { label } => write!(
                f,
                "`{label}` is not one of the canonical timeframes {:?}",
                CANONICAL_TIMEFRAMES
                    .iter()
                    .map(|(n, _)| *n)
                    .collect::<Vec<_>>()
            ),
            Self::LaneHorizonConflict {
                lane,
                lane_bars_in_base,
                horizon_bars,
            } => write!(
                f,
                "higher lane `{lane}` is {lane_bars_in_base} base bars long but the label \
                 horizon is {horizon_bars} bars: the label resolves inside a single {lane} \
                 candle, so the lane's columns would be ranked on an outcome shorter than \
                 their own bar (docs/higher-timeframe-lane-2026-08-09.md §3-4)"
            ),
            Self::SpaceEnumerated {
                slot,
                drawn,
                of,
                consecutive_collisions,
            } => write!(
                f,
                "slot {slot} was ABANDONED: {consecutive_collisions} consecutive uniform draws \
                 all collided with configurations this session has already run, so the reachable \
                 space is enumerated (U5, ProposerExhausted). The sweep stopped at {drawn} of \
                 {of} searches; slots {slot}..{of} were never drawn. Counted and named here, one \
                 record per abandoned slot, rather than left as the difference between a sweep's \
                 declared size and its actual one"
            ),
        }
    }
}

impl std::error::Error for SpaceError {}

/// A point in axis-A space: one level index per factor.
///
/// The field is **private** and the only constructor validates. That is the
/// whole of "typed, bounded, frozen-field-free": there is no way to build a
/// delta that names a field, only one that names a level of a declared factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SearchConfigDelta {
    levels: [u8; FACTOR_COUNT],
}

impl SearchConfigDelta {
    /// Validated construction. Every index is checked against its factor's
    /// arity and against `caps`; an out-of-range or unavailable level is an
    /// error, never a clamp.
    pub fn new(levels: [u8; FACTOR_COUNT], caps: &Capabilities) -> Result<Self, SpaceError> {
        for f in FactorId::ALL {
            let l = levels[f.index()];
            if (l as usize) >= f.arity() {
                return Err(SpaceError::LevelOutOfRange {
                    factor: f,
                    level: l,
                    arity: f.arity(),
                });
            }
            if let Some(knob) = f.level_requires(l)
                && !knob.available(caps)
            {
                return Err(SpaceError::LevelUnavailable {
                    factor: f,
                    level: l,
                    knob,
                });
            }
        }
        Ok(Self { levels })
    }

    pub fn level(&self, factor: FactorId) -> u8 {
        self.levels[factor.index()]
    }

    pub fn levels(&self) -> [u8; FACTOR_COUNT] {
        self.levels
    }

    /// Set one factor's level, validated. Used by the trust region, which
    /// reverts factors toward the parent one at a time.
    pub fn with_level(
        mut self,
        factor: FactorId,
        level: u8,
        caps: &Capabilities,
    ) -> Result<Self, SpaceError> {
        self.levels[factor.index()] = level;
        Self::new(self.levels, caps)
    }

    /// Which factors differ from `other`. The trust region's input.
    pub fn differing_factors(&self, other: &Self) -> Vec<FactorId> {
        FactorId::ALL
            .into_iter()
            .filter(|f| self.levels[f.index()] != other.levels[f.index()])
            .collect()
    }

    // ── resolved values ────────────────────────────────────────────────────

    pub fn population(&self) -> usize {
        POPULATION_LEVELS[self.level(FactorId::Population) as usize]
    }
    pub fn generations(&self) -> usize {
        GENERATION_LEVELS[self.level(FactorId::Generations) as usize]
    }
    pub fn gene_width(&self) -> usize {
        GENE_WIDTH_LEVELS[self.level(FactorId::GeneWidth) as usize]
    }
    pub fn prefilter_width(&self) -> usize {
        PREFILTER_WIDTH_LEVELS[self.level(FactorId::PrefilterWidth) as usize]
    }
    pub fn prefilter_insample_frac(&self) -> f64 {
        PREFILTER_INSAMPLE_FRAC_LEVELS[self.level(FactorId::PrefilterInsampleFrac) as usize]
    }
    pub fn base_timeframe(&self) -> &'static str {
        BASE_TIMEFRAME_LEVELS[self.level(FactorId::BaseTimeframe) as usize]
    }
    pub fn higher_lanes(&self) -> &'static [&'static str] {
        HIGHER_LANE_LEVELS[self.level(FactorId::HigherLanes) as usize]
    }
    pub fn streaming_batches(&self) -> usize {
        STREAMING_BATCH_LEVELS[self.level(FactorId::StreamingBatches) as usize]
    }
    pub fn cursor_policy(&self) -> CursorPolicy {
        CURSOR_LEVELS[self.level(FactorId::WorkingSetCursor) as usize]
    }
    pub fn candidate_count(&self) -> usize {
        CANDIDATE_COUNT_LEVELS[self.level(FactorId::CandidateCount) as usize]
    }
    pub fn portfolio_shape(&self) -> (usize, f64) {
        PORTFOLIO_SHAPE_LEVELS[self.level(FactorId::PortfolioShape) as usize]
    }

    pub fn streaming_plan(&self) -> StreamingPlan {
        match self.streaming_batches() {
            0 => StreamingPlan::disabled(),
            n => StreamingPlan::streaming(n),
        }
    }

    /// Apply the structural half of the proposal.
    ///
    /// **Ordering matters and is not negotiable.** This runs on the config
    /// *before* `DiscoveryConfig::apply_mode_overrides`, because that function
    /// scales `filtering.min_trades_per_month` by `timeframe_label` and rewrites
    /// `min_trades_per_day` in PropFirm mode. Applying the timeframe after the
    /// mode overrides would scale the floors for a timeframe the run is not
    /// using; applying the objective's floors before them would have the mode
    /// silently overwrite the objective. So the sequence is:
    ///
    /// ```text
    ///   baseline -> apply_structural -> apply_mode_overrides -> objective/refusals -> audit
    /// ```
    ///
    /// and [`crate::proposal::materialise`] is the only place that runs it.
    pub fn apply_structural(&self, cfg: &mut DiscoveryConfig) -> Result<(), SpaceError> {
        let tf = self.base_timeframe();
        let tf_min = timeframe_minutes(tf).ok_or_else(|| SpaceError::UnknownTimeframe {
            label: tf.to_string(),
        })?;
        for lane in self.higher_lanes() {
            let lane_min = timeframe_minutes(lane).ok_or_else(|| SpaceError::UnknownTimeframe {
                label: (*lane).to_string(),
            })?;
            if lane_min <= tf_min {
                return Err(SpaceError::UnknownTimeframe {
                    label: format!("{lane} is not HIGHER than the base timeframe {tf}"),
                });
            }
        }

        cfg.population = self.population();
        // Auto-sizing would raise the population to the card's ceiling and the
        // level would decide nothing. A factor that does not decide its own
        // value is a factor that lies in the coverage table.
        cfg.population_auto = false;
        cfg.generations = self.generations();
        cfg.max_indicators = self.gene_width();
        cfg.runtime_overrides.prefilter_top_k = self.prefilter_width();
        cfg.runtime_overrides.prefilter_insample_frac = self.prefilter_insample_frac();
        cfg.timeframe_label = tf.to_string();
        cfg.higher_timeframes = self
            .higher_lanes()
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        cfg.candidate_count = self.candidate_count();
        let (portfolio_size, corr_threshold) = self.portfolio_shape();
        cfg.portfolio_size = portfolio_size;
        cfg.corr_threshold = corr_threshold;
        Ok(())
    }
}

/// Refuse a `(base timeframe, lane, label horizon)` triple in which the lane's
/// candle is longer than the whole label.
///
/// `docs/higher-timeframe-lane-2026-08-09.md` §3–4 measured this exactly: a
/// 35-bar (~3 hour) label resolves inside a single H4 candle, and the H4 columns
/// that took top-240 places did so on 12–18 real observations dressed as
/// 568–840 forward-filled rows. Admitting the lane unchanged replaces a false
/// negative with a false positive, and this loop would find it and promote it.
/// So the lane is reachable only through a label horizon that can express it —
/// axis-B variant B4 — which is the mechanism that document names.
pub fn lane_horizon_conflict(
    base_timeframe: &str,
    lanes: &[&str],
    horizon_bars: usize,
) -> Result<(), SpaceError> {
    let base_min =
        timeframe_minutes(base_timeframe).ok_or_else(|| SpaceError::UnknownTimeframe {
            label: base_timeframe.to_string(),
        })?;
    for lane in lanes {
        let lane_min = timeframe_minutes(lane).ok_or_else(|| SpaceError::UnknownTimeframe {
            label: (*lane).to_string(),
        })?;
        let lane_bars_in_base = lane_min.div_ceil(base_min.max(1));
        if (horizon_bars as u32) < lane_bars_in_base {
            return Err(SpaceError::LaneHorizonConflict {
                lane: (*lane).to_string(),
                lane_bars_in_base,
                horizon_bars,
            });
        }
    }
    Ok(())
}

/// One excluded level, named, with the edit that would restore it.
///
/// A level can be excluded for two different kinds of reason and the report
/// must not blur them: a **missing knob** (the compiled search cannot express
/// it — `knob` is `Some`, and the symbol, the feature and the section that
/// carries the edit are all named), or a **measured impossibility** (an
/// unreachable payoff floor — `knob` is `None` and `reason` carries the
/// numbers). Either way the level is never silently substituted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExcludedLevel {
    pub axis: String,
    pub factor: String,
    pub level: String,
    pub knob: Option<RequiredKnob>,
    pub symbol: Option<String>,
    pub feature: Option<String>,
    pub required_elsewhere: Option<String>,
    pub reason: String,
}

impl ExcludedLevel {
    /// A level the compiled search cannot express.
    pub fn missing_knob(axis: &str, factor: &str, level: String, knob: RequiredKnob) -> Self {
        Self {
            axis: axis.to_string(),
            factor: factor.to_string(),
            level,
            knob: Some(knob),
            symbol: Some(knob.symbol().to_string()),
            feature: Some(knob.feature().to_string()),
            required_elsewhere: Some(knob.required_elsewhere_section().to_string()),
            reason: format!(
                "requires `{}`, absent from the compiled neoethos-search",
                knob.symbol()
            ),
        }
    }

    /// A level this configuration's own measured geometry cannot reach.
    pub fn unreachable(axis: &str, factor: &str, level: String, reason: String) -> Self {
        Self {
            axis: axis.to_string(),
            factor: factor.to_string(),
            level,
            knob: None,
            symbol: None,
            feature: None,
            required_elsewhere: None,
            reason,
        }
    }
}

/// The drawable space, and everything excluded from it.
///
/// Rendered into the session header and into every verdict. `"unreachable"` is
/// only ever a claim about a *searched* space (§10.2), so the report must say
/// which one — including the parts that were never searchable at all.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpaceReport {
    pub factors: Vec<FactorReport>,
    pub excluded: Vec<ExcludedLevel>,
    pub identity_source: String,
    pub drawable_cells: u128,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactorReport {
    pub factor: String,
    pub levels: Vec<String>,
    pub drawable: Vec<String>,
}

/// The name the loop's `contracts.rs` and the verdict use for [`SpaceReport`].
/// One type, two names — never two types.
pub type SpaceDescription = SpaceReport;

/// Build the axis-A half of the space report under `caps`.
pub fn axis_a_report(caps: &Capabilities) -> (Vec<FactorReport>, Vec<ExcludedLevel>, u128) {
    let mut factors = Vec::with_capacity(FACTOR_COUNT);
    let mut excluded = Vec::new();
    let mut cells: u128 = 1;
    for f in FactorId::ALL {
        let mut all = Vec::with_capacity(f.arity());
        let mut drawable = Vec::new();
        for l in 0..f.arity() as u8 {
            let name = f.level_label(l);
            all.push(name.clone());
            if f.level_available(l, caps) {
                drawable.push(name);
            } else if let Some(knob) = f.level_requires(l) {
                excluded.push(ExcludedLevel::missing_knob("A", f.label(), name, knob));
            }
        }
        cells = cells.saturating_mul(drawable.len().max(1) as u128);
        factors.push(FactorReport {
            factor: f.label().to_string(),
            levels: all,
            drawable,
        });
    }
    (factors, excluded, cells)
}

/// How a proposal is identified, and why.
///
/// With `ResolvedConfigStamp::ga_seed` present (§14.3) the `config_hash` alone
/// is the identity. Without it, two proposals that differ only in their
/// replicate seed hash identically, and deduping on the hash alone would
/// **silently delete every replicate** — which is exactly the within-cell
/// variance the judge needs. So the fallback identity is the pair, and the
/// session header says so in as many words.
pub fn identity_source(caps: &Capabilities) -> &'static str {
    if caps.stamp_ga_seed {
        "config_hash (run_identity.rs carries ga_seed)"
    } else {
        "config_hash + external replicate_seed (run_identity.rs lacks ga_seed)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps() -> Capabilities {
        Capabilities::all_for_tests()
    }

    #[test]
    fn every_written_field_is_disjoint_from_the_frozen_wall() {
        for f in FactorId::ALL {
            for field in f.writes() {
                assert!(
                    !FROZEN_FIELDS.contains(field),
                    "axis-A factor `{}` writes frozen field `{field}`",
                    f.label()
                );
            }
        }
    }

    #[test]
    fn money_path_and_validation_geometry_are_both_inside_the_frozen_wall() {
        for field in MONEY_PATH_FIELDS.iter().chain(VALIDATION_GEOMETRY_FIELDS) {
            assert!(
                FROZEN_FIELDS.contains(field),
                "`{field}` is not in FROZEN_FIELDS"
            );
        }
    }

    #[test]
    fn factor_indices_match_declaration_order() {
        for (i, f) in FactorId::ALL.into_iter().enumerate() {
            assert_eq!(f.index(), i, "{} is out of order", f.label());
            assert!(f.arity() >= 3, "{} has fewer than three levels", f.label());
            assert!(f.arity() <= 5, "{} has more than five levels", f.label());
        }
    }

    #[test]
    fn a_level_outside_the_arity_is_refused_not_clamped() {
        let mut levels = [0u8; FACTOR_COUNT];
        levels[FactorId::Population.index()] = 99;
        let err = SearchConfigDelta::new(levels, &caps()).unwrap_err();
        assert!(matches!(
            err,
            SpaceError::LevelOutOfRange {
                factor: FactorId::Population,
                level: 99,
                ..
            }
        ));
    }

    #[test]
    fn a_cursor_level_without_the_knob_is_unavailable_and_names_the_symbol() {
        let mut none = caps();
        none.streaming_start_cursor = false;
        let mut levels = [0u8; FACTOR_COUNT];
        levels[FactorId::WorkingSetCursor.index()] = 2; // RandomOffset
        let err = SearchConfigDelta::new(levels, &none).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("StreamingPlan::start_cursor"), "{text}");
        assert!(text.contains("search-streaming-start-cursor"), "{text}");
        // …and `Start` stays drawable, so the loop still runs.
        levels[FactorId::WorkingSetCursor.index()] = 0;
        assert!(SearchConfigDelta::new(levels, &none).is_ok());
    }

    #[test]
    fn excluded_levels_are_named_in_the_space_report() {
        let mut none = caps();
        none.streaming_start_cursor = false;
        let (_, excluded, _) = axis_a_report(&none);
        assert_eq!(
            excluded.len(),
            2,
            "Resumed and RandomOffset should both be excluded"
        );
        assert!(
            excluded
                .iter()
                .all(|e| e.knob == Some(RequiredKnob::StreamingStartCursor))
        );
        assert!(excluded.iter().all(|e| e.required_elsewhere.is_some()));
    }

    #[test]
    fn an_h4_lane_under_a_35_bar_label_is_refused_by_name() {
        // M5 base: an H4 candle is 48 base bars. The shipped 35-bar label
        // resolves INSIDE one of them.
        let err = lane_horizon_conflict("M5", &["H1", "H4"], 35).unwrap_err();
        assert!(matches!(
            err,
            SpaceError::LaneHorizonConflict {
                lane_bars_in_base: 48,
                horizon_bars: 35,
                ..
            }
        ));
        // 120 bars (10 hours) spans it.
        assert!(lane_horizon_conflict("M5", &["H1", "H4"], 120).is_ok());
        // D1 needs 288 base bars, so 120 is still not enough.
        assert!(lane_horizon_conflict("M5", &["H1", "D1"], 120).is_err());
        assert!(lane_horizon_conflict("M5", &["H1", "D1"], 480).is_ok());
    }

    #[test]
    fn the_identity_source_is_stated_either_way() {
        let mut c = caps();
        assert!(identity_source(&c).contains("carries ga_seed"));
        c.stamp_ga_seed = false;
        assert!(identity_source(&c).contains("lacks ga_seed"));
    }

    #[test]
    fn an_abandoned_slot_names_itself_and_the_size_of_the_shortfall() {
        // The U5 record. `break 'slots` used to leave the abandoned slots as the
        // difference between the sweep's declared size and its actual one, which
        // is the silent drop the surrounding `bail!` explicitly refuses to make
        // on the other early-exit path.
        let e = SpaceError::SpaceEnumerated {
            slot: 12,
            drawn: 12,
            of: SWEEP_SEARCHES,
            consecutive_collisions: PROPOSER_RETRIES,
        };
        let text = e.to_string();
        assert!(text.contains("slot 12 was ABANDONED"), "{text}");
        assert!(
            text.contains(&format!("{PROPOSER_RETRIES} consecutive")),
            "{text}"
        );
        assert!(text.contains(&format!("12 of {SWEEP_SEARCHES}")), "{text}");
        assert!(text.contains("enumerated"), "{text}");
    }

    #[test]
    fn differing_factors_is_the_trust_regions_input() {
        let a = SearchConfigDelta::new([0; FACTOR_COUNT], &caps()).unwrap();
        let b = a
            .with_level(FactorId::Population, 2, &caps())
            .unwrap()
            .with_level(FactorId::Generations, 1, &caps())
            .unwrap();
        let d = b.differing_factors(&a);
        assert_eq!(d, vec![FactorId::Population, FactorId::Generations]);
    }
}
