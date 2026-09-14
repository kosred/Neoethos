//! Resolved search configuration and cost/exit-geometry diagnostics.
//!
//! A full-TP/full-SL payoff ratio describes only those two terminal outcomes.
//! It is not an upper bound on the average winning/losing trade ratio when
//! trailing, time limits or session exits can realize smaller losses. A prior
//! sample's observed payoff is likewise not a bound on a different strategy.
//! Search admission validates configuration domains; actual candidate results
//! still have to satisfy the unchanged quality, validation and OOS targets.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::discovery::DiscoveryConfig;

pub const RESOLVED_CONFIG_STAMP_SCHEMA_VERSION_V2: u16 = 2;
pub const POPULATION_AUTO_SEARCH_AUTHORITY_SCHEMA_VERSION_V1: u16 = 1;

/// Mandatory algorithm version in the private hash body, not a user setting.
/// Binds the three-way evidence partition, unique selected-candidate labels,
/// and price-gross-only currency-conversion fees even with unchanged knobs.
/// Both stamp producers use this token; old stamps fail their current self-hash
/// and cannot carry old two-way ledgers into the CPU selection authority.
const SEARCH_ALGORITHM_SEMANTICS_V1: &str = "neoethos.search.algorithm.is-calibration-holdout-80-10-10.unique-labels.price-gross-conversion-fee.v1";

/// Include corrected execution/stop/admission arithmetic in every config key.
/// Computed once, not once per candidate. This separates existing cached scores
/// and checkpoints even when the user's knobs and dataset are unchanged.
fn cpu_execution_source_sha256() -> &'static str {
    use sha2::{Digest, Sha256};
    static IDENTITY: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    IDENTITY.get_or_init(|| {
        let mut hash = Sha256::new();
        hash.update(b"neoethos.cpu-execution-source.v1\0");
        for source in [
            include_bytes!("eval.rs").as_slice(),
            include_bytes!("stop_target.rs").as_slice(),
            include_bytes!("run_identity.rs").as_slice(),
        ] {
            hash.update((source.len() as u64).to_le_bytes());
            hash.update(source);
        }
        format!("{:x}", hash.finalize())
    })
}

fn deserialize_required_option<'de, D, T>(
    deserializer: D,
) -> std::result::Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

/// Historical sample statistic retained for API compatibility and diagnosis.
/// It is NOT a mathematical ceiling and must never authorize a search refusal.
pub const MEASURED_TRAILING_PAYOFF_CEILING: f64 = 1.08;

/// The inputs the payoff ceiling is computed from. Every one of them is a
/// RESOLVED value — what the run will actually use, not what a default says.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PayoffCeilingInputs {
    /// Tightest stop the search space can express, in pips
    /// (`ResolvedGeneStopBounds::sl_min_pips` — ATR-scaled when a scale is
    /// installed for this dataset).
    pub sl_min_pips: f64,
    /// Widest stop the search space can express, in pips.
    pub sl_max_pips: f64,
    /// Tightest take-profit the search space can express, in pips.
    pub tp_min_pips: f64,
    /// Widest take-profit the search space can express, in pips
    /// (`ResolvedGeneStopBounds::tp_max_pips`).
    pub tp_max_pips: f64,
    /// Highest reward:risk the INITIALISER samples
    /// (`ResolvedGeneStopBounds::rr_max`). Mutation is not bound by it — it
    /// clamps SL and TP independently — so this bounds generation 0 only.
    pub initializer_rr_max: f64,
    /// Lowest reward:risk the INITIALISER samples
    /// (`ResolvedGeneStopBounds::rr_min`).
    ///
    /// It plays NO part in the payoff ceiling — the ceiling is set by the widest
    /// TP against the tightest SL, so only `rr_max` can bind it. It is carried
    /// here because it decides something else entirely: the PREFILTER's label
    /// geometry is `label_rr = 0.5 × (rr_min + rr_max)`
    /// (`discovery.rs`, the `label_sl_atr_mult` / `label_rr` block), so two runs
    /// that differ only in `rr_min` rank features differently and therefore
    /// search different spaces. Before this field existed `rr_min` appeared
    /// nowhere in the stamp and was not recoverable from it — `rr_max` can be
    /// read back as `tp_max / sl_max`, but `rr_min` cannot — so those two runs
    /// produced the SAME `config_hash`. A stamp that gives one hash to two
    /// different experiments is worse than no stamp, because it invites a false
    /// equality.
    pub initializer_rr_min: f64,
    /// The median ATR, in pips, the band was scaled to, or `None` when the
    /// absolute pip band is in force. Recorded because the same four pip
    /// numbers mean completely different things on M5 and on H4, and every
    /// prior run's artifacts were silent about which was which.
    pub atr_pips: Option<f64>,
    /// Round-trip cost charged per trade, in pips:
    /// `spread_pips + commission_per_trade / pip_value_per_lot`.
    /// `eval.rs` charges the entry half-spread into `entry_px` and the exit half
    /// plus the full `commission_per_trade` at exit, so this is the whole
    /// round trip.
    pub cost_pips_round_trip: f64,
    pub trailing_enabled: bool,
    /// R multiple at which the trail arms (`trailing_be_trigger_r`).
    pub trailing_be_trigger_r: f64,
    /// Give-back from the running extreme, as a multiple of the gene's OWN stop
    /// distance (`trailing_atr_multiplier`). Despite the name it is not ATR
    /// based: `eval.rs:1034` computes `hi − multiplier × pos_sl_pips × pip`.
    pub trailing_give_back_r: f64,
    /// Floor the armed trail is clamped to, in pips above entry
    /// (`BacktestSettings::trailing_min_lock_pips`, default 2.0).
    pub trailing_min_lock_pips: f64,
}

/// Which term of the arithmetic actually pins the ceiling. Naming it is the
/// difference between "raise the TP clamp" and "turn the trail off".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BindingConstraint {
    /// The take-profit clamp's upper bound caps the numerator.
    TakeProfitClamp,
    /// The stop clamp's lower bound floors the denominator.
    StopClamp,
    /// The fixed round-trip charge exceeds the widest take-profit distance.
    CostExceedsTakeProfit,
    /// Legacy diagnostic variant. Current code never treats a sample's
    /// trailing payoff as an admission constraint.
    TrailingGiveBack,
}

impl BindingConstraint {
    pub fn label(self) -> &'static str {
        match self {
            Self::TakeProfitClamp => "take-profit clamp upper bound",
            Self::StopClamp => "stop clamp lower bound",
            Self::CostExceedsTakeProfit => "charged cost exceeds the widest take-profit",
            Self::TrailingGiveBack => "trailing give-back (measured, not arithmetic)",
        }
    }
}

/// The computed ceiling and every number the operator needs to argue with it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PayoffCeiling {
    /// Full-TP/full-SL payoff reference: (tp_max - c) / (sl_min + c).
    /// This does not bound average win/loss ratios with partial/session exits.
    pub arithmetic_ceiling: f64,
    /// Full-TP/full-SL reference restricted to the initializer's upper RR band.
    /// It does not bound generation 0's realized average winning/losing trades.
    pub initializer_ceiling: f64,
    /// The TP that produced [`Self::arithmetic_ceiling`], in pips.
    pub ceiling_tp_pips: f64,
    /// The SL that produced [`Self::arithmetic_ceiling`], in pips.
    pub ceiling_sl_pips: f64,
    /// `Some` when the trail arms at or before it starts conceding — i.e.
    /// `give_back >= be_trigger`, so the armed stop sits at or below entry and
    /// is clamped to the min-lock floor. The value is the payoff of a run in
    /// which every armed trade exits at that floor:
    /// `(min_lock − c) / (sl_min + c)`. It is the degenerate value the trail
    /// collapses toward, NOT a maximum.
    #[serde(deserialize_with = "deserialize_required_option")]
    pub trailing_armed_floor_payoff: Option<f64>,
    /// Legacy serialized field. Now equals the barrier reference and is
    /// advisory only; it is not enforced as a search-admission ceiling.
    pub enforced_ceiling: f64,
    pub binding: BindingConstraint,
    /// True whenever trailing is enabled: there is no proved universal
    /// average-win/loss ceiling for any trailing geometry.
    pub trailing_ceiling_unmeasured: bool,
    /// Win rate at which a strategy sitting EXACTLY at the configured floor
    /// breaks even in gross-R terms: `1 / (1 + floor)`.
    pub required_win_rate_at_floor: f64,
    /// Cost-charged break-even win rate at the ceiling barriers:
    /// `(sl + c) / (sl + tp)`.
    pub breakeven_win_rate_at_ceiling: f64,
    /// Driftless first-passage win rate at the ceiling barriers: `sl / (sl + tp)`.
    /// This is what a coin gets. Anything above it must be paid for by edge.
    pub zero_edge_base_rate: f64,
}

impl PayoffCeiling {
    /// Edge, in win-rate points, the configuration demands before it can even
    /// return its money — the gap between what a coin produces at these
    /// barriers and what cost-recovery requires.
    pub fn edge_points_required_to_break_even(&self) -> f64 {
        self.breakeven_win_rate_at_ceiling - self.zero_edge_base_rate
    }
}

fn finite(name: &str, v: f64) -> Result<f64> {
    if !v.is_finite() {
        bail!("payoff-ceiling input `{name}` is not finite ({v})");
    }
    Ok(v)
}

/// Full-stop/full-target geometry as a pure diagnostic. The box reference
/// maximizes (tp - cost) / (sl + cost); the initializer reference additionally
/// respects its upper RR constraint. Neither bounds the realized average
/// win/loss ratio when some exits occur before a full SL or TP.
pub fn max_achievable_payoff(inputs: &PayoffCeilingInputs) -> Result<PayoffCeiling> {
    let sl_min = finite("sl_min_pips", inputs.sl_min_pips)?;
    let sl_max = finite("sl_max_pips", inputs.sl_max_pips)?;
    let tp_min = finite("tp_min_pips", inputs.tp_min_pips)?;
    let tp_max = finite("tp_max_pips", inputs.tp_max_pips)?;
    let rr_min = finite("initializer_rr_min", inputs.initializer_rr_min)?;
    let rr_max = finite("initializer_rr_max", inputs.initializer_rr_max)?;
    let cost = finite("cost_pips_round_trip", inputs.cost_pips_round_trip)?;
    let be_trigger = finite("trailing_be_trigger_r", inputs.trailing_be_trigger_r)?;
    let give_back = finite("trailing_give_back_r", inputs.trailing_give_back_r)?;
    let min_lock = finite("trailing_min_lock_pips", inputs.trailing_min_lock_pips)?;

    if sl_min <= 0.0 || sl_max <= 0.0 || sl_min > sl_max {
        bail!("payoff-ceiling stop clamp must satisfy 0 < min <= max (got {sl_min}..={sl_max})");
    }
    if tp_min <= 0.0 || tp_max <= 0.0 || tp_min > tp_max {
        bail!(
            "payoff-ceiling take-profit clamp must satisfy 0 < min <= max (got {tp_min}..={tp_max})"
        );
    }
    if rr_min <= 0.0 || rr_max <= 0.0 || rr_min > rr_max {
        bail!(
            "payoff-ceiling initializer RR band must satisfy 0 < min <= max (got {rr_min}..={rr_max})"
        );
    }
    if let Some(atr_pips) = inputs.atr_pips {
        let atr_pips = finite("atr_pips", atr_pips)?;
        if atr_pips <= 0.0 {
            bail!("payoff-ceiling input `atr_pips` must be > 0 (got {atr_pips})");
        }
    }
    if cost < 0.0 {
        bail!("payoff-ceiling input `cost_pips_round_trip` must be >= 0 (got {cost})");
    }
    if be_trigger < 0.0 || give_back < 0.0 || min_lock < 0.0 {
        bail!(
            "payoff-ceiling trailing geometry must be nonnegative (trigger={be_trigger}, \
             give_back={give_back}, min_lock_pips={min_lock})"
        );
    }

    let denom = sl_min + cost;
    let numer = tp_max - cost;
    let arithmetic_ceiling = if numer <= 0.0 { 0.0 } else { numer / denom };

    // With a positive fixed cost, an RR-constrained payoff increases with
    // SL until TP hits its cap, then decreases. The maximum is at that kink
    // (clamped to the allowed SL interval), not necessarily at the smallest SL.
    let init_sl = (tp_max / rr_max).clamp(sl_min, sl_max);
    let init_tp = tp_max.min(rr_max * init_sl);
    let init_numer = init_tp - cost;
    let initializer_ceiling = if init_numer <= 0.0 {
        0.0
    } else {
        init_numer / (init_sl + cost)
    };

    // Which term pins it. Cost first (it can dominate outright), then whichever
    // clamp the operator would have to move. Moving `tp_max` raises the
    // numerator 1:1; moving `sl_min` down raises the ratio faster whenever
    // `sl_min < tp_max`, which is the whole configured band, so the TP clamp is
    // the honest "what do I change" answer only when the stop is already at the
    // floor of what a broker will accept. We report the TP clamp when the
    // numerator is the smaller lever and the stop clamp otherwise.
    let binding = if numer <= 0.0 {
        BindingConstraint::CostExceedsTakeProfit
    } else if tp_max <= sl_min {
        BindingConstraint::TakeProfitClamp
    } else {
        BindingConstraint::StopClamp
    };

    // A trailing exit can realize a small loss or reach TP before retracing.
    // No observed sample average bounds every possible sequence of such exits.
    let trail_on = inputs.trailing_enabled;
    let trailing_armed_floor_payoff = (trail_on && give_back >= be_trigger && give_back > 0.0)
        .then_some((min_lock - cost) / denom);
    let trailing_ceiling_unmeasured = trail_on;
    let enforced_ceiling = arithmetic_ceiling; // legacy field: diagnostic only

    Ok(PayoffCeiling {
        arithmetic_ceiling,
        initializer_ceiling,
        ceiling_tp_pips: tp_max,
        ceiling_sl_pips: sl_min,
        trailing_armed_floor_payoff,
        enforced_ceiling,
        binding,
        trailing_ceiling_unmeasured,
        // `w · avg_win = (1 − w) · avg_loss` at `avg_win / avg_loss = floor`
        // gives `w = 1 / (1 + floor)`. Filled by the caller that knows the floor;
        // recomputed there. Placeholder values are overwritten in
        // `assert_payoff_floor_reachable`, and this function is also called
        // directly by the stamp, which does not need them — so they are computed
        // against the ceiling itself, which is the honest reading of "the win
        // rate this configuration would require if it got everything it asked
        // for".
        required_win_rate_at_floor: 1.0 / (1.0 + enforced_ceiling),
        breakeven_win_rate_at_ceiling: denom / (sl_min + tp_max),
        zero_edge_base_rate: sl_min / (sl_min + tp_max),
    })
}

/// Validate domains and return the configured payoff diagnostic.
///
/// Kept under the historical API name for existing callers. A measured sample
/// or the full-stop barrier ratio cannot prove that a target average win/loss
/// ratio is unreachable. The target itself is unchanged and is checked against
/// the candidate's realized trades in the quality/validation funnel.
pub fn assert_payoff_floor_reachable(
    configured_floor: f64,
    inputs: &PayoffCeilingInputs,
) -> Result<PayoffCeiling> {
    let mut diagnostic = max_achievable_payoff(inputs)?;
    if !configured_floor.is_finite() {
        bail!("configured payoff floor is not finite ({configured_floor})");
    }
    if configured_floor > 0.0 {
        diagnostic.required_win_rate_at_floor = 1.0 / (1.0 + configured_floor);
    }
    Ok(diagnostic)
}

// ---------------------------------------------------------------------------
// Resolved-config stamp.
// ---------------------------------------------------------------------------

/// A legacy decision-critical subset, resolved. Written into the discovery
/// ledger so an operator can attribute the major search controls after the
/// fact. This is not an exhaustive experiment identity: the population-sizing
/// receipt, canonical input receipt, and exact stage-1 view remain separate
/// authorities.
///
/// Deliberately NOT the whole `DiscoveryConfig` (that is
/// `DiscoveryRunProfile`'s job, and it lands next to the portfolio JSON): this
/// is the short list an operator reads first. Its hash compares only this
/// stamped subset; the strict S3b search authority adds the sizing/stage facts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedConfigStamp {
    /// Strict schema 2: all authority fields are required and unknown fields
    /// are rejected. Legacy/default-filled stamps are not current authority.
    pub schema_version: u16,
    /// `fnv64:…` over every other field of this legacy subset, in declaration
    /// order. Equality does not claim that every DiscoveryConfig field matches.
    pub config_hash: String,
    pub symbol: String,
    pub timeframe: String,
    pub mode: String,

    // ── THE PRIMARY GATE ──────────────────────────────────────────────────
    //
    // Added 2026-08-09 after the review pointed out that the stamp recorded the
    // DEMOTED payoff floor and not the check that now decides survival. A stamp
    // that omits the primary gate is worse than no stamp: it invites a false
    // equality between two runs that searched under different rules.
    /// `TargetProfile::min_net_expectancy_per_trade`. Checked UNCONDITIONALLY,
    /// ahead of the payoff floor: `0.0` means "strictly positive".
    pub min_net_expectancy_per_trade: f64,
    /// `TargetProfile::min_expectancy_t_stat`. `0.0` = the significance bar is
    /// OFF, which is the shipped state.
    pub min_expectancy_t_stat: f64,

    // ── the gate that decided the LAST run's outcome, now demoted ─────────
    pub payoff_floor: f64,
    pub min_win_rate: f64,
    pub max_in_market: f64,

    // ── the search space those floors are judged against ──────────────────
    pub sl_clamp_pips: (f64, f64),
    pub tp_clamp_pips: (f64, f64),
    pub initializer_rr_max: f64,
    /// The initialiser's LOWER reward:risk bound. Not recoverable from the four
    /// pip clamps (`rr_max` is `tp_max / sl_max`; `rr_min` is not expressible),
    /// and it moves the prefilter's label geometry — see
    /// `PayoffCeilingInputs::initializer_rr_min`. Without it two runs that
    /// ranked features differently hashed identically.
    pub initializer_rr_min: f64,
    /// Median ATR (pips) the band was scaled to; `None` = absolute pip band.
    /// Without this the four pip numbers above are un-interpretable across
    /// timeframes.
    #[serde(deserialize_with = "deserialize_required_option")]
    pub band_atr_pips: Option<f64>,
    pub trailing_enabled: bool,
    pub trailing_be_trigger_r: f64,
    pub trailing_give_back_r: f64,
    pub trailing_min_lock_pips: f64,

    // ── what a trade is charged ───────────────────────────────────────────
    pub spread_pips: f64,
    pub commission_per_trade: f64,
    pub pip_value_per_lot: f64,
    pub cost_pips_round_trip: f64,
    pub swap_long_pips_per_day: f64,
    pub swap_short_pips_per_day: f64,
    /// Whether the evaluator force-closes and blocks entries around the
    /// weekend boundary. This changes trade outcomes, so an older stamp that
    /// did not bind the decision is intentionally not deserializable as the
    /// current authority.
    pub kill_zones_enabled: bool,
    /// The per-UTC-bucket spread curve, or `None` for a FLAT spread charged at
    /// 03:00 Tokyo and at the London open alike. Without this in the hash, a
    /// flat-spread run and a per-hour-curve run hash identically.
    #[serde(deserialize_with = "deserialize_required_option")]
    pub session_spread_pips: Option<[f64; 3]>,
    /// The band every reported result is measured against, so a ledger can say
    /// what its own `cost_band_*` counts were measured at.
    #[serde(deserialize_with = "deserialize_required_option")]
    pub cost_band_pips: Option<(f64, f64)>,

    // ── the shape of the search ───────────────────────────────────────────
    pub prefilter_top_k: usize,
    pub prefilter_insample_frac: f64,
    pub prefilter_min_per_timeframe: usize,
    /// CPCV settings. They now decide the PREFILTER's refit windows as well as
    /// the validation folds, so two runs that differ here explored different
    /// feature sets and are not the same experiment.
    pub enable_cpcv: bool,
    pub cpcv_n_splits: usize,
    pub cpcv_n_test_groups: usize,
    pub cpcv_embargo_pct: f64,
    pub cpcv_purge_pct: f64,
    pub cpcv_max_rows: usize,
    pub population: usize,
    pub population_auto: bool,
    pub max_indicators: usize,
    pub max_rows: usize,
    pub max_rows_by_timeframe: BTreeMap<String, usize>,
    pub max_hours: f64,
    pub funnel_stage1_pct: f64,
    /// Stable schema token: `earliest` or `most_recent` (never `Debug`).
    pub stage1_window: String,
    pub generations: usize,
    pub candidate_count: usize,
    pub portfolio_size: usize,
    pub mc_runs: u32,
    pub mc_min_profitable: u32,
    pub initial_balance: f64,
    pub risk_per_trade_min: f64,
    pub risk_per_trade_max: f64,

    // ── the two normalisation flags whose disagreement makes a multi-term
    //    gene equal to its largest-magnitude term ────────────────────────────
    pub adaptive_thresholds: bool,
    pub normalize_features: bool,

    /// The gate's own verdict, recorded so a ledger says what the run was
    /// permitted to find.
    pub payoff_ceiling: PayoffCeiling,
}

/// Everything except the hash, so the hash can be computed over it.
#[derive(Serialize)]
struct StampBody<'a> {
    cpu_execution_source_sha256: &'a str,
    search_algorithm_semantics: &'static str,
    schema_version: u16,
    symbol: &'a str,
    timeframe: &'a str,
    mode: &'a str,
    min_net_expectancy_per_trade: f64,
    min_expectancy_t_stat: f64,
    payoff_floor: f64,
    min_win_rate: f64,
    max_in_market: f64,
    sl_clamp_pips: (f64, f64),
    tp_clamp_pips: (f64, f64),
    initializer_rr_max: f64,
    initializer_rr_min: f64,
    band_atr_pips: Option<f64>,
    trailing_enabled: bool,
    trailing_be_trigger_r: f64,
    trailing_give_back_r: f64,
    trailing_min_lock_pips: f64,
    spread_pips: f64,
    commission_per_trade: f64,
    pip_value_per_lot: f64,
    cost_pips_round_trip: f64,
    swap_long_pips_per_day: f64,
    swap_short_pips_per_day: f64,
    kill_zones_enabled: bool,
    session_spread_pips: Option<[f64; 3]>,
    cost_band_pips: Option<(f64, f64)>,
    prefilter_top_k: usize,
    prefilter_insample_frac: f64,
    prefilter_min_per_timeframe: usize,
    enable_cpcv: bool,
    cpcv_n_splits: usize,
    cpcv_n_test_groups: usize,
    cpcv_embargo_pct: f64,
    cpcv_purge_pct: f64,
    cpcv_max_rows: usize,
    population: usize,
    population_auto: bool,
    max_indicators: usize,
    max_rows: usize,
    max_rows_by_timeframe: &'a BTreeMap<String, usize>,
    max_hours: f64,
    funnel_stage1_pct: f64,
    stage1_window: &'a str,
    generations: usize,
    candidate_count: usize,
    portfolio_size: usize,
    mc_runs: u32,
    mc_min_profitable: u32,
    initial_balance: f64,
    risk_per_trade_min: f64,
    risk_per_trade_max: f64,
    adaptive_thresholds: bool,
    normalize_features: bool,
}

impl ResolvedConfigStamp {
    fn hash_body_v2(&self) -> StampBody<'_> {
        StampBody {
            cpu_execution_source_sha256: cpu_execution_source_sha256(),
            search_algorithm_semantics: SEARCH_ALGORITHM_SEMANTICS_V1,
            schema_version: self.schema_version,
            symbol: &self.symbol,
            timeframe: &self.timeframe,
            mode: &self.mode,
            min_net_expectancy_per_trade: self.min_net_expectancy_per_trade,
            min_expectancy_t_stat: self.min_expectancy_t_stat,
            payoff_floor: self.payoff_floor,
            min_win_rate: self.min_win_rate,
            max_in_market: self.max_in_market,
            sl_clamp_pips: self.sl_clamp_pips,
            tp_clamp_pips: self.tp_clamp_pips,
            initializer_rr_max: self.initializer_rr_max,
            initializer_rr_min: self.initializer_rr_min,
            band_atr_pips: self.band_atr_pips,
            trailing_enabled: self.trailing_enabled,
            trailing_be_trigger_r: self.trailing_be_trigger_r,
            trailing_give_back_r: self.trailing_give_back_r,
            trailing_min_lock_pips: self.trailing_min_lock_pips,
            spread_pips: self.spread_pips,
            commission_per_trade: self.commission_per_trade,
            pip_value_per_lot: self.pip_value_per_lot,
            cost_pips_round_trip: self.cost_pips_round_trip,
            swap_long_pips_per_day: self.swap_long_pips_per_day,
            swap_short_pips_per_day: self.swap_short_pips_per_day,
            kill_zones_enabled: self.kill_zones_enabled,
            session_spread_pips: self.session_spread_pips,
            cost_band_pips: self.cost_band_pips,
            prefilter_top_k: self.prefilter_top_k,
            prefilter_insample_frac: self.prefilter_insample_frac,
            prefilter_min_per_timeframe: self.prefilter_min_per_timeframe,
            enable_cpcv: self.enable_cpcv,
            cpcv_n_splits: self.cpcv_n_splits,
            cpcv_n_test_groups: self.cpcv_n_test_groups,
            cpcv_embargo_pct: self.cpcv_embargo_pct,
            cpcv_purge_pct: self.cpcv_purge_pct,
            cpcv_max_rows: self.cpcv_max_rows,
            population: self.population,
            population_auto: self.population_auto,
            max_indicators: self.max_indicators,
            max_rows: self.max_rows,
            max_rows_by_timeframe: &self.max_rows_by_timeframe,
            max_hours: self.max_hours,
            funnel_stage1_pct: self.funnel_stage1_pct,
            stage1_window: &self.stage1_window,
            generations: self.generations,
            candidate_count: self.candidate_count,
            portfolio_size: self.portfolio_size,
            mc_runs: self.mc_runs,
            mc_min_profitable: self.mc_min_profitable,
            initial_balance: self.initial_balance,
            risk_per_trade_min: self.risk_per_trade_min,
            risk_per_trade_max: self.risk_per_trade_max,
            adaptive_thresholds: self.adaptive_thresholds,
            normalize_features: self.normalize_features,
        }
    }

    fn computed_config_hash_v2(&self) -> Result<String> {
        crate::artifact_io::stable_json_hash(&self.hash_body_v2())
    }

    fn validate_persisted_float_domains_v2(&self) -> Result<()> {
        let required_finite = [
            (
                "min_net_expectancy_per_trade",
                self.min_net_expectancy_per_trade,
            ),
            ("min_expectancy_t_stat", self.min_expectancy_t_stat),
            ("payoff_floor", self.payoff_floor),
            ("min_win_rate", self.min_win_rate),
            ("max_in_market", self.max_in_market),
            ("sl_clamp_pips.min", self.sl_clamp_pips.0),
            ("sl_clamp_pips.max", self.sl_clamp_pips.1),
            ("tp_clamp_pips.min", self.tp_clamp_pips.0),
            ("tp_clamp_pips.max", self.tp_clamp_pips.1),
            ("initializer_rr_max", self.initializer_rr_max),
            ("initializer_rr_min", self.initializer_rr_min),
            ("trailing_be_trigger_r", self.trailing_be_trigger_r),
            ("trailing_give_back_r", self.trailing_give_back_r),
            ("trailing_min_lock_pips", self.trailing_min_lock_pips),
            ("spread_pips", self.spread_pips),
            ("commission_per_trade", self.commission_per_trade),
            ("pip_value_per_lot", self.pip_value_per_lot),
            ("cost_pips_round_trip", self.cost_pips_round_trip),
            ("swap_long_pips_per_day", self.swap_long_pips_per_day),
            ("swap_short_pips_per_day", self.swap_short_pips_per_day),
            ("prefilter_insample_frac", self.prefilter_insample_frac),
            ("cpcv_embargo_pct", self.cpcv_embargo_pct),
            ("cpcv_purge_pct", self.cpcv_purge_pct),
            ("max_hours", self.max_hours),
            ("funnel_stage1_pct", self.funnel_stage1_pct),
            ("initial_balance", self.initial_balance),
            ("risk_per_trade_min", self.risk_per_trade_min),
            ("risk_per_trade_max", self.risk_per_trade_max),
            (
                "payoff_ceiling.arithmetic_ceiling",
                self.payoff_ceiling.arithmetic_ceiling,
            ),
            (
                "payoff_ceiling.initializer_ceiling",
                self.payoff_ceiling.initializer_ceiling,
            ),
            (
                "payoff_ceiling.ceiling_tp_pips",
                self.payoff_ceiling.ceiling_tp_pips,
            ),
            (
                "payoff_ceiling.ceiling_sl_pips",
                self.payoff_ceiling.ceiling_sl_pips,
            ),
            (
                "payoff_ceiling.enforced_ceiling",
                self.payoff_ceiling.enforced_ceiling,
            ),
            (
                "payoff_ceiling.required_win_rate_at_floor",
                self.payoff_ceiling.required_win_rate_at_floor,
            ),
            (
                "payoff_ceiling.breakeven_win_rate_at_ceiling",
                self.payoff_ceiling.breakeven_win_rate_at_ceiling,
            ),
            (
                "payoff_ceiling.zero_edge_base_rate",
                self.payoff_ceiling.zero_edge_base_rate,
            ),
        ];
        for (name, value) in required_finite {
            anyhow::ensure!(
                value.is_finite(),
                "resolved-config stamp field `{name}` is not finite"
            );
        }
        if let Some(value) = self.band_atr_pips {
            anyhow::ensure!(
                value.is_finite() && value > 0.0,
                "resolved-config stamp ATR scale must be finite and > 0"
            );
        }
        if let Some(value) = self.payoff_ceiling.trailing_armed_floor_payoff {
            anyhow::ensure!(
                value.is_finite(),
                "resolved-config stamp trailing armed-floor payoff is not finite"
            );
        }
        if let Some(curve) = self.session_spread_pips {
            for (bucket, value) in ["asian", "overlap", "late_ny"].into_iter().zip(curve) {
                anyhow::ensure!(
                    value.is_finite() && value >= 0.0,
                    "resolved-config stamp session spread `{bucket}` must be finite and >= 0"
                );
            }
        }
        if let Some((lo, hi)) = self.cost_band_pips {
            anyhow::ensure!(
                lo.is_finite() && hi.is_finite() && 0.0 <= lo && lo <= hi,
                "resolved-config stamp cost band must satisfy 0 <= lo <= hi"
            );
        }

        // Negative target floors are intentionally retained as disabled or
        // lenient sentinels by direct DiscoveryConfig callers. They must be
        // finite, but only the fields that are fractions have an upper domain.
        anyhow::ensure!(
            self.min_win_rate <= 1.0 && self.max_in_market <= 1.0,
            "resolved-config stamp target fractions must be <= 1"
        );
        anyhow::ensure!(
            self.prefilter_insample_frac > 0.0 && self.prefilter_insample_frac <= 1.0,
            "resolved-config stamp prefilter fraction must be in (0, 1]"
        );
        anyhow::ensure!(
            (0.0..=1.0).contains(&self.cpcv_embargo_pct)
                && (0.0..=1.0).contains(&self.cpcv_purge_pct),
            "resolved-config stamp CPCV percentages must be in [0, 1]"
        );
        anyhow::ensure!(
            (0.01..=1.0).contains(&self.funnel_stage1_pct),
            "resolved-config stamp stage1 percentage must be in [0.01, 1]"
        );
        anyhow::ensure!(
            self.max_hours >= 0.0,
            "resolved-config stamp max-hours must be >= 0"
        );
        anyhow::ensure!(
            self.initial_balance > 0.0,
            "resolved-config stamp initial balance must be > 0"
        );
        anyhow::ensure!(
            0.0 <= self.risk_per_trade_min
                && self.risk_per_trade_min <= self.risk_per_trade_max
                && self.risk_per_trade_max <= 1.0,
            "resolved-config stamp risk band must satisfy 0 <= min <= max <= 1"
        );
        anyhow::ensure!(
            self.spread_pips >= 0.0
                && self.commission_per_trade >= 0.0
                && self.pip_value_per_lot > 0.0
                && self.cost_pips_round_trip >= 0.0,
            "resolved-config stamp cost inputs must be nonnegative with positive pip value"
        );
        anyhow::ensure!(
            self.trailing_be_trigger_r >= 0.0
                && self.trailing_give_back_r >= 0.0
                && self.trailing_min_lock_pips >= 0.0,
            "resolved-config stamp trailing geometry must be nonnegative"
        );
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == RESOLVED_CONFIG_STAMP_SCHEMA_VERSION_V2,
            "unsupported resolved-config stamp schema {}",
            self.schema_version
        );
        anyhow::ensure!(
            self.config_hash == self.computed_config_hash_v2()?,
            "resolved-config stamp self-hash mismatch"
        );
        anyhow::ensure!(
            !self.symbol.is_empty()
                && !self.timeframe.is_empty()
                && self.population > 0
                && self.max_indicators > 0,
            "resolved-config stamp has an empty identity or zero search extent"
        );
        anyhow::ensure!(
            matches!(self.mode.as_str(), "strict" | "prop_firm" | "risky"),
            "resolved-config stamp has an unknown mode token"
        );
        anyhow::ensure!(
            matches!(self.stage1_window.as_str(), "earliest" | "most_recent"),
            "resolved-config stamp has an unknown stage1-window token"
        );
        self.validate_persisted_float_domains_v2()?;
        let inputs = PayoffCeilingInputs {
            sl_min_pips: self.sl_clamp_pips.0,
            sl_max_pips: self.sl_clamp_pips.1,
            tp_min_pips: self.tp_clamp_pips.0,
            tp_max_pips: self.tp_clamp_pips.1,
            initializer_rr_max: self.initializer_rr_max,
            initializer_rr_min: self.initializer_rr_min,
            atr_pips: self.band_atr_pips,
            cost_pips_round_trip: self.cost_pips_round_trip,
            trailing_enabled: self.trailing_enabled,
            trailing_be_trigger_r: self.trailing_be_trigger_r,
            trailing_give_back_r: self.trailing_give_back_r,
            trailing_min_lock_pips: self.trailing_min_lock_pips,
        };
        anyhow::ensure!(
            self.cost_pips_round_trip
                == cost_pips_round_trip(
                    self.spread_pips,
                    self.commission_per_trade,
                    self.pip_value_per_lot,
                ),
            "resolved-config stamp raw cost fields do not reconcile to round-trip cost"
        );
        anyhow::ensure!(
            self.payoff_ceiling == assert_payoff_floor_reachable(self.payoff_floor, &inputs)?,
            "resolved-config stamp payoff ceiling is detached from its inputs"
        );
        Ok(())
    }
}

/// Round-trip cost in pips, from the resolved cost model.
///
/// `eval.rs` charges the entry half-spread into `entry_px` and the exit half
/// plus the full `commission_per_trade` at exit, so spread is charged once per
/// round trip and commission once per round trip.
///
/// KNOWN GAP, not fixed here (change #5 owns it): `commission_per_trade` is a
/// single round-trip number, while a real 45 USD/1M schedule is ~0.62 pips PER
/// SIDE, ~1.24 pips round trip before spread. Do not read any single number
/// here as "the true cost" — results should be reported against a 1.6–2.4 pip
/// band, and a result that survives only at the optimistic edge is not a result.
pub fn cost_pips_round_trip(
    spread_pips: f64,
    commission_per_trade: f64,
    pip_value_per_lot: f64,
) -> f64 {
    if !pip_value_per_lot.is_finite() || pip_value_per_lot.abs() < f64::EPSILON {
        // No conversion available: charge the spread alone and let the caller's
        // NaN guards speak. Never silently pretend commission is zero without
        // saying so — callers log this via the stamp's `pip_value_per_lot`.
        return spread_pips;
    }
    spread_pips + commission_per_trade / pip_value_per_lot
}

/// Build the [`PayoffCeilingInputs`] this run resolves to.
///
/// Every input is READ FROM THE SAME ACCESSOR THE SEARCH READS, never
/// re-declared here:
///   * the SL/TP band from [`crate::genetic::current_gene_stop_bounds`], so the
///     gate sees the ATR-scaled band this dataset actually installed — not the
///     absolute M5 pip literals that were true until 2026-08-09;
///   * the trailing geometry from
///     [`crate::genetic::current_strategy_evaluation_runtime_overrides`]'s
///     `exit_policy`, the config recipient that replaced the hardcoded
///     `trailing_enabled: true` in `strategy_gene.rs`;
///   * `pip_value_per_lot` from `DiscoveryConfig::evaluation_config`, the
///     evaluator's own cost model.
///
/// A gate that re-declared any of these would be checking a different search
/// than the one about to run, which is the whole failure mode it exists to stop.
///
/// MUST be called AFTER the per-run ATR scale is installed
/// (`install_gene_stop_atr_scale` / `clear_gene_stop_atr_scale`), otherwise it
/// reads a previous combo's band.
pub fn payoff_inputs_for_config(
    config: &DiscoveryConfig,
    pip_value_per_lot: f64,
) -> PayoffCeilingInputs {
    let bounds = crate::genetic::current_gene_stop_bounds();
    let exit = crate::genetic::current_strategy_evaluation_runtime_overrides().exit_policy;
    PayoffCeilingInputs {
        sl_min_pips: bounds.sl_min_pips,
        sl_max_pips: bounds.sl_max_pips,
        tp_min_pips: bounds.tp_min_pips,
        tp_max_pips: bounds.tp_max_pips,
        initializer_rr_max: bounds.rr_max,
        initializer_rr_min: bounds.rr_min,
        atr_pips: bounds.atr_pips,
        cost_pips_round_trip: cost_pips_round_trip(
            config.evaluation_spread_pips,
            config.evaluation_commission_per_trade,
            pip_value_per_lot,
        ),
        trailing_enabled: exit.trailing_enabled,
        trailing_be_trigger_r: exit.trailing_be_trigger_r,
        // `ExitPolicyOverrides` names it `trailing_stop_multiplier` because it
        // multiplies the position's own stop distance; downstream it is copied
        // into the field called `trailing_atr_multiplier`, which is not ATR.
        trailing_give_back_r: exit.trailing_stop_multiplier,
        trailing_min_lock_pips: exit.trailing_min_lock_pips,
    }
}

/// This run's `config_hash`, for callers that need only the identity.
///
/// Same function, same inputs, same hash as the stamp written into the ledger —
/// so a trial-returns matrix and the ledger beside it can be proved to belong to
/// one run rather than assumed to. Returns `None` (never a placeholder) when the
/// stamp cannot be built: a caller must treat that as "cannot be attributed".
pub fn config_hash_for(
    config: &DiscoveryConfig,
    pip_value_per_lot: f64,
    normalize_features: bool,
) -> Option<String> {
    let inputs = payoff_inputs_for_config(config, pip_value_per_lot);
    let ceiling =
        assert_payoff_floor_reachable(config.target_profile.min_payoff_ratio, &inputs).ok()?;
    stamp_resolved_config(
        config,
        &inputs,
        ceiling,
        pip_value_per_lot,
        normalize_features,
    )
    .ok()
    .map(|s| s.config_hash)
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PopulationAutoSelectionSemanticsV1 {
    schema_version: u16,
    resolved_config_stamp_hash: String,
    population_auto: bool,
    configured_population: u64,
    resolved_population: u64,
    requested_max_indicators: u64,
    term_cap: u64,
    month_capacity: u64,
    migration_enabled_for_run: bool,
    stage1_role: String,
    stage1_row_start: u64,
    stage1_row_end: u64,
    stage1_identity_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    growth_goal: Option<crate::scoring::RiskyGrowthGoal>,
}

/// Strict persisted authority for the selection-changing population decision.
///
/// The full sizing receipt remains embedded so device/admission facts are not
/// lost. Its hardware-only fields are intentionally excluded from
/// `search_config_hash`: two valid cards that resolve the same P/K/stage view
/// execute the same search. The containing result additionally cross-links the
/// exact sizing-receipt identity to its completed execution receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PopulationAutoSearchAuthorityV1 {
    schema_version: u16,
    resolved_config_stamp: ResolvedConfigStamp,
    population_auto_sizing_receipt:
        crate::population_auto_sizing_receipt_v1::PopulationAutoSizingReceiptV1,
    search_config_hash: String,
    /// Legacy authorities omit the goal; new goal-aware runs bind it separately
    /// from the unchanged resolved-config stamp schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    growth_goal: Option<crate::scoring::RiskyGrowthGoal>,
}

impl PopulationAutoSearchAuthorityV1 {
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub const fn resolved_config_stamp(&self) -> &ResolvedConfigStamp {
        &self.resolved_config_stamp
    }

    pub const fn population_auto_sizing_receipt(
        &self,
    ) -> &crate::population_auto_sizing_receipt_v1::PopulationAutoSizingReceiptV1 {
        &self.population_auto_sizing_receipt
    }

    pub fn search_config_hash(&self) -> &str {
        &self.search_config_hash
    }

    pub const fn growth_goal(&self) -> Option<crate::scoring::RiskyGrowthGoal> {
        self.growth_goal
    }

    fn semantic_projection_unchecked_v1(&self) -> Result<PopulationAutoSelectionSemanticsV1> {
        let receipt = &self.population_auto_sizing_receipt;
        let stage1 = receipt.stage1_window();
        Ok(PopulationAutoSelectionSemanticsV1 {
            schema_version: self.schema_version,
            resolved_config_stamp_hash: self.resolved_config_stamp.config_hash.clone(),
            population_auto: receipt.population_auto(),
            configured_population: u64::try_from(receipt.configured_population())?,
            resolved_population: u64::try_from(receipt.resolved_population())?,
            requested_max_indicators: u64::try_from(receipt.requested_max_indicators())?,
            term_cap: u64::try_from(receipt.term_cap())?,
            month_capacity: u64::try_from(receipt.month_capacity())?,
            migration_enabled_for_run: receipt.migration_enabled_for_run(),
            stage1_role: stage1.role().to_owned(),
            stage1_row_start: stage1.row_start(),
            stage1_row_end: stage1.row_end(),
            stage1_identity_sha256: stage1.identity_sha256().to_owned(),
            growth_goal: self.growth_goal,
        })
    }

    fn computed_search_config_hash_v1(&self) -> Result<String> {
        crate::artifact_io::stable_json_hash(&self.semantic_projection_unchecked_v1()?)
    }

    pub fn selection_semantics_v1(&self) -> Result<PopulationAutoSelectionSemanticsV1> {
        self.validate()?;
        self.semantic_projection_unchecked_v1()
    }

    pub fn semantically_matches(&self, other: &Self) -> Result<bool> {
        Ok(self.selection_semantics_v1()? == other.selection_semantics_v1()?)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == POPULATION_AUTO_SEARCH_AUTHORITY_SCHEMA_VERSION_V1,
            "unsupported population-auto search authority schema {}",
            self.schema_version
        );
        self.resolved_config_stamp.validate()?;
        self.population_auto_sizing_receipt
            .validate()
            .map_err(anyhow::Error::new)?;
        let stamp = &self.resolved_config_stamp;
        let receipt = &self.population_auto_sizing_receipt;
        if let Some(goal) = self.growth_goal {
            anyhow::ensure!(
                stamp.mode == "risky",
                "growth goal requires Risky selection mode"
            );
            goal.validate().map_err(anyhow::Error::msg)?;
        }
        anyhow::ensure!(
            stamp.population == receipt.resolved_population(),
            "resolved-config population is detached from its sizing receipt"
        );
        anyhow::ensure!(
            stamp.population_auto == receipt.population_auto(),
            "resolved-config population_auto is detached from its sizing receipt"
        );
        anyhow::ensure!(
            stamp.max_indicators == receipt.requested_max_indicators(),
            "resolved-config max_indicators is detached from its sizing receipt"
        );

        let resident_rows = receipt.resident_parent_rows();
        let expected_stage1_rows =
            ((resident_rows as f64 * stamp.funnel_stage1_pct) as usize).min(resident_rows);
        anyhow::ensure!(
            expected_stage1_rows == receipt.evaluation_rows(),
            "resolved stage1 percentage is detached from the exact evaluation extent"
        );
        let stage1 = receipt.stage1_window();
        let expected_range = match stamp.stage1_window.as_str() {
            "earliest" => (0, expected_stage1_rows),
            "most_recent" => (
                resident_rows.saturating_sub(expected_stage1_rows),
                resident_rows,
            ),
            other => anyhow::bail!("unknown strict stage1 window token {other}"),
        };
        anyhow::ensure!(
            stage1.row_start() == u64::try_from(expected_range.0)?
                && stage1.row_end() == u64::try_from(expected_range.1)?,
            "resolved stage1 policy is detached from the exact stage1 range"
        );
        anyhow::ensure!(
            self.search_config_hash == self.computed_search_config_hash_v1()?,
            "population-auto search semantic hash mismatch"
        );
        Ok(())
    }
}

pub fn build_population_auto_search_authority_v1(
    config: &DiscoveryConfig,
    receipt: &crate::population_auto_sizing_receipt_v1::PopulationAutoSizingReceiptV1,
    pip_value_per_lot: f64,
    normalize_features: bool,
) -> Result<PopulationAutoSearchAuthorityV1> {
    receipt.validate().map_err(anyhow::Error::new)?;
    let growth_goal = matches!(config.mode, crate::discovery::DiscoveryMode::Risky).then_some(
        crate::scoring::RiskyGrowthGoal {
            start_balance: config.risky_start_balance,
            target_balance: config.risky_target_balance,
            horizon_days: config.risky_horizon_days,
        },
    );
    if let Some(goal) = growth_goal {
        goal.validate().map_err(anyhow::Error::msg)?;
    }
    anyhow::ensure!(
        config.population == receipt.configured_population(),
        "requested DiscoveryConfig population does not match the sizing receipt"
    );
    anyhow::ensure!(
        config.population_auto == receipt.population_auto(),
        "requested DiscoveryConfig population_auto does not match the sizing receipt"
    );
    anyhow::ensure!(
        config.max_indicators == receipt.requested_max_indicators(),
        "requested DiscoveryConfig max_indicators does not match the sizing receipt"
    );
    let resolved_config = DiscoveryConfig {
        population: receipt.resolved_population(),
        ..config.clone()
    };
    let inputs = payoff_inputs_for_config(&resolved_config, pip_value_per_lot);
    let ceiling =
        assert_payoff_floor_reachable(resolved_config.target_profile.min_payoff_ratio, &inputs)?;
    let resolved_config_stamp = stamp_resolved_config(
        &resolved_config,
        &inputs,
        ceiling,
        pip_value_per_lot,
        normalize_features,
    )?;
    let mut authority = PopulationAutoSearchAuthorityV1 {
        schema_version: POPULATION_AUTO_SEARCH_AUTHORITY_SCHEMA_VERSION_V1,
        resolved_config_stamp,
        population_auto_sizing_receipt: receipt.clone(),
        search_config_hash: String::new(),
        growth_goal,
    };
    authority.search_config_hash = authority.computed_search_config_hash_v1()?;
    authority.validate()?;
    Ok(authority)
}

/// Receipt-linked sizing/search semantic hash.
///
/// This deliberately excludes the selected device, free-memory snapshot, and
/// probe identities: those are persisted and validated in the sizing receipt,
/// but a different card that resolves the same population and stage-1 search
/// does not create a different search. It remains narrower than an exhaustive
/// `DiscoveryConfig` experiment identity because the legacy stamp has known
/// pre-existing omissions outside S3b.
pub fn population_auto_semantic_config_hash_for_v1(
    config: &DiscoveryConfig,
    receipt: &crate::population_auto_sizing_receipt_v1::PopulationAutoSizingReceiptV1,
    pip_value_per_lot: f64,
    normalize_features: bool,
) -> Result<String> {
    Ok(build_population_auto_search_authority_v1(
        config,
        receipt,
        pip_value_per_lot,
        normalize_features,
    )?
    .search_config_hash)
}

#[cfg(all(test, feature = "gpu-b-adapter"))]
pub(crate) fn recompute_population_auto_search_authority_hash_for_test_v1(
    authority: &mut PopulationAutoSearchAuthorityV1,
) -> Result<()> {
    authority.search_config_hash = authority.computed_search_config_hash_v1()?;
    Ok(())
}

#[cfg(all(test, feature = "gpu-b-adapter"))]
pub(crate) fn recompute_resolved_config_stamp_hash_for_test_v2(
    authority: &mut PopulationAutoSearchAuthorityV1,
) -> Result<()> {
    authority.resolved_config_stamp.config_hash =
        authority.resolved_config_stamp.computed_config_hash_v2()?;
    authority.search_config_hash = authority.computed_search_config_hash_v1()?;
    Ok(())
}

/// Stamp the resolved configuration. Pure: takes every ambient value as an
/// argument so it is testable without a process-wide install.
pub fn stamp_resolved_config(
    config: &DiscoveryConfig,
    inputs: &PayoffCeilingInputs,
    ceiling: PayoffCeiling,
    pip_value_per_lot: f64,
    normalize_features: bool,
) -> Result<ResolvedConfigStamp> {
    let canonical_ceiling =
        assert_payoff_floor_reachable(config.target_profile.min_payoff_ratio, inputs)?;
    anyhow::ensure!(
        ceiling == canonical_ceiling,
        "resolved-config stamp ceiling must come from the canonical payoff-floor gate"
    );
    let mode = match config.mode {
        crate::discovery::DiscoveryMode::Strict => "strict",
        crate::discovery::DiscoveryMode::PropFirm => "prop_firm",
        crate::discovery::DiscoveryMode::Risky => "risky",
    };
    let max_rows_by_timeframe = config
        .max_rows_by_timeframe
        .iter()
        .map(|(timeframe, rows)| (timeframe.clone(), *rows))
        .collect::<BTreeMap<_, _>>();
    let stage1_window = match config.runtime_overrides.stage1_window {
        crate::discovery::Stage1Window::Earliest => "earliest",
        crate::discovery::Stage1Window::MostRecent => "most_recent",
    };
    let body = StampBody {
        cpu_execution_source_sha256: cpu_execution_source_sha256(),
        search_algorithm_semantics: SEARCH_ALGORITHM_SEMANTICS_V1,
        schema_version: RESOLVED_CONFIG_STAMP_SCHEMA_VERSION_V2,
        symbol: &config.evaluation_symbol,
        timeframe: &config.timeframe_label,
        mode,
        min_net_expectancy_per_trade: config.target_profile.min_net_expectancy_per_trade,
        min_expectancy_t_stat: config.target_profile.min_expectancy_t_stat,
        payoff_floor: config.target_profile.min_payoff_ratio,
        min_win_rate: config.target_profile.min_win_rate,
        max_in_market: config.target_profile.max_in_market,
        sl_clamp_pips: (inputs.sl_min_pips, inputs.sl_max_pips),
        tp_clamp_pips: (inputs.tp_min_pips, inputs.tp_max_pips),
        initializer_rr_max: inputs.initializer_rr_max,
        initializer_rr_min: inputs.initializer_rr_min,
        band_atr_pips: inputs.atr_pips,
        trailing_enabled: inputs.trailing_enabled,
        trailing_be_trigger_r: inputs.trailing_be_trigger_r,
        trailing_give_back_r: inputs.trailing_give_back_r,
        trailing_min_lock_pips: inputs.trailing_min_lock_pips,
        spread_pips: config.evaluation_spread_pips,
        commission_per_trade: config.evaluation_commission_per_trade,
        pip_value_per_lot,
        cost_pips_round_trip: inputs.cost_pips_round_trip,
        swap_long_pips_per_day: config.swap_long_pips_per_day,
        swap_short_pips_per_day: config.swap_short_pips_per_day,
        kill_zones_enabled: config.kill_zones_enabled,
        session_spread_pips: config.session_spread_pips,
        cost_band_pips: config.cost_band_pips,
        prefilter_top_k: config.runtime_overrides.prefilter_top_k,
        prefilter_insample_frac: config.runtime_overrides.resolved_prefilter_insample_frac(),
        prefilter_min_per_timeframe: config.runtime_overrides.prefilter_min_per_timeframe,
        enable_cpcv: config.enable_cpcv,
        cpcv_n_splits: config.cpcv_n_splits,
        cpcv_n_test_groups: config.cpcv_n_test_groups,
        cpcv_embargo_pct: config.cpcv_embargo_pct,
        cpcv_purge_pct: config.cpcv_purge_pct,
        cpcv_max_rows: config.cpcv_max_rows,
        population: config.population,
        population_auto: config.population_auto,
        max_indicators: config.max_indicators,
        max_rows: config.max_rows,
        max_rows_by_timeframe: &max_rows_by_timeframe,
        max_hours: config.max_hours,
        funnel_stage1_pct: config.runtime_overrides.resolved_funnel_stage1_pct(),
        stage1_window,
        generations: config.generations,
        candidate_count: config.candidate_count,
        portfolio_size: config.portfolio_size,
        mc_runs: config.mc_runs,
        mc_min_profitable: config.mc_min_profitable,
        initial_balance: config.initial_balance,
        risk_per_trade_min: config.risk_per_trade_min,
        risk_per_trade_max: config.risk_per_trade_max,
        adaptive_thresholds: config.adaptive_thresholds,
        normalize_features,
    };
    let config_hash = crate::artifact_io::stable_json_hash(&body)?;
    let stamp = ResolvedConfigStamp {
        schema_version: body.schema_version,
        config_hash,
        symbol: body.symbol.to_string(),
        timeframe: body.timeframe.to_string(),
        mode: body.mode.to_string(),
        min_net_expectancy_per_trade: body.min_net_expectancy_per_trade,
        min_expectancy_t_stat: body.min_expectancy_t_stat,
        payoff_floor: body.payoff_floor,
        min_win_rate: body.min_win_rate,
        max_in_market: body.max_in_market,
        sl_clamp_pips: body.sl_clamp_pips,
        tp_clamp_pips: body.tp_clamp_pips,
        initializer_rr_max: body.initializer_rr_max,
        initializer_rr_min: body.initializer_rr_min,
        band_atr_pips: body.band_atr_pips,
        trailing_enabled: body.trailing_enabled,
        trailing_be_trigger_r: body.trailing_be_trigger_r,
        trailing_give_back_r: body.trailing_give_back_r,
        trailing_min_lock_pips: body.trailing_min_lock_pips,
        spread_pips: body.spread_pips,
        commission_per_trade: body.commission_per_trade,
        pip_value_per_lot: body.pip_value_per_lot,
        cost_pips_round_trip: body.cost_pips_round_trip,
        swap_long_pips_per_day: body.swap_long_pips_per_day,
        swap_short_pips_per_day: body.swap_short_pips_per_day,
        kill_zones_enabled: body.kill_zones_enabled,
        session_spread_pips: body.session_spread_pips,
        cost_band_pips: body.cost_band_pips,
        prefilter_top_k: body.prefilter_top_k,
        prefilter_insample_frac: body.prefilter_insample_frac,
        prefilter_min_per_timeframe: body.prefilter_min_per_timeframe,
        enable_cpcv: body.enable_cpcv,
        cpcv_n_splits: body.cpcv_n_splits,
        cpcv_n_test_groups: body.cpcv_n_test_groups,
        cpcv_embargo_pct: body.cpcv_embargo_pct,
        cpcv_purge_pct: body.cpcv_purge_pct,
        cpcv_max_rows: body.cpcv_max_rows,
        population: body.population,
        population_auto: body.population_auto,
        max_indicators: body.max_indicators,
        max_rows: body.max_rows,
        max_rows_by_timeframe: max_rows_by_timeframe.clone(),
        max_hours: body.max_hours,
        funnel_stage1_pct: body.funnel_stage1_pct,
        stage1_window: body.stage1_window.to_owned(),
        generations: body.generations,
        candidate_count: body.candidate_count,
        portfolio_size: body.portfolio_size,
        mc_runs: body.mc_runs,
        mc_min_profitable: body.mc_min_profitable,
        initial_balance: body.initial_balance,
        risk_per_trade_min: body.risk_per_trade_min,
        risk_per_trade_max: body.risk_per_trade_max,
        adaptive_thresholds: body.adaptive_thresholds,
        normalize_features: body.normalize_features,
        payoff_ceiling: ceiling,
    };
    stamp.validate()?;
    Ok(stamp)
}

// ---------------------------------------------------------------------------
// Tests. The gate must be verifiable without a run — that is the point.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The configuration that produced "174 candidates screened, 0 survived".
    fn production_inputs() -> PayoffCeilingInputs {
        PayoffCeilingInputs {
            sl_min_pips: 6.0,
            sl_max_pips: 20.0,
            tp_min_pips: 12.0,
            tp_max_pips: 45.0,
            initializer_rr_max: 2.5,
            initializer_rr_min: 1.5,
            atr_pips: None,
            cost_pips_round_trip: 2.89,
            trailing_enabled: true,
            trailing_be_trigger_r: 1.0,
            trailing_give_back_r: 1.0,
            trailing_min_lock_pips: 2.0,
        }
    }

    #[test]
    fn barrier_arithmetic_matches_the_hand_calculation() {
        let mut i = production_inputs();
        i.trailing_enabled = false;
        let c = max_achievable_payoff(&i).unwrap();
        // (45 - 2.89) / (6 + 2.89) = 42.11 / 8.89
        assert!((c.arithmetic_ceiling - (42.11 / 8.89)).abs() < 1e-9);
        // Initializer optimum: SL=45/2.5=18, TP=45. Moving SL below the
        // TP/RR kink increases the relative cost and REDUCES this ratio.
        assert!((c.initializer_ceiling - (42.11 / 20.89)).abs() < 1e-9);
        assert!(c.trailing_armed_floor_payoff.is_none());
        assert_eq!(c.binding, BindingConstraint::StopClamp);
        // With the trail off the enforced ceiling is the arithmetic one.
        assert!((c.enforced_ceiling - c.arithmetic_ceiling).abs() < 1e-12);
    }

    #[test]
    fn the_verdicts_rr_inequality_holds() {
        // "the floor needs tp >= 2*sl + 3*c": at sl 20 and c 2.89 that is
        // 48.67, outside the [12, 45] take-profit clamp.
        let c = 2.89_f64;
        let sl = 20.0_f64;
        let needed_tp = 2.0 * sl + 3.0 * c;
        assert!((needed_tp - 48.67).abs() < 1e-9);
        assert!(needed_tp > 45.0, "the clamp excludes the required RR");
        // And the payoff at exactly that TP is exactly 2.0.
        assert!((((needed_tp - c) / (sl + c)) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn trailing_sample_average_does_not_replace_the_barrier_diagnostic() {
        let i = production_inputs();
        let c = max_achievable_payoff(&i).unwrap();
        assert_eq!(c.binding, BindingConstraint::StopClamp);
        assert_eq!(c.enforced_ceiling, c.arithmetic_ceiling);
        assert!(c.trailing_ceiling_unmeasured);
        // The armed trade's floor exit is (2.0 - 2.89) / 8.89 — NEGATIVE. The
        // trail turns a would-be loser into a small loss dressed as a win, which
        // is exactly the mechanism that pinned the measured payoff near 1.0.
        let floor_payoff = c.trailing_armed_floor_payoff.expect("armed regime");
        assert!(floor_payoff < 0.0, "min-lock is below the charged cost");
        assert!((floor_payoff - (-0.89 / 8.89)).abs() < 1e-9);
    }

    #[test]
    fn valid_payoff_targets_are_evaluated_not_prejudged_by_a_historical_sample() {
        let inputs = production_inputs();
        for target in [2.0, 5.0, 20.0] {
            let diagnostic = assert_payoff_floor_reachable(target, &inputs)
                .expect("a sample average is not a universal performance bound");
            assert_eq!(diagnostic.required_win_rate_at_floor, 1.0 / (1.0 + target));
        }
        assert!(assert_payoff_floor_reachable(f64::NAN, &inputs).is_err());
    }

    #[test]
    fn floor_zero_disables_the_gate() {
        let i = production_inputs();
        assert!(assert_payoff_floor_reachable(0.0, &i).is_ok());
    }

    #[test]
    fn trail_off_admits_the_two_point_zero_floor() {
        // This is the honest boundary: with the trail off the floor IS
        // reachable, because mutation clamps SL and TP independently and can
        // reach sl 6 / tp 45. The gate must not refuse it.
        let mut i = production_inputs();
        i.trailing_enabled = false;
        let c = assert_payoff_floor_reachable(2.0, &i).expect("reachable with the trail off");
        assert!(c.enforced_ceiling > 2.0);
        // A partial loss at a timeout can make the realized payoff larger
        // than the full-SL reference even with trailing disabled.
        assert!(assert_payoff_floor_reachable(5.0, &i).is_ok());
    }

    #[test]
    fn tight_and_loose_trails_both_require_realized_trade_evidence() {
        for give_back in [0.4, 1.0, 3.0] {
            let mut inputs = production_inputs();
            inputs.trailing_give_back_r = give_back;
            let diagnostic = assert_payoff_floor_reachable(2.0, &inputs).unwrap();
            assert!(diagnostic.trailing_ceiling_unmeasured);
            assert_ne!(diagnostic.binding, BindingConstraint::TrailingGiveBack);
            assert_eq!(diagnostic.enforced_ceiling, diagnostic.arithmetic_ceiling);
            assert_eq!(
                diagnostic.trailing_armed_floor_payoff.is_some(),
                give_back >= 1.0
            );
        }
    }

    #[test]
    fn initializer_reference_matches_an_independent_grid_maximum() {
        let mut inputs = production_inputs();
        inputs.trailing_enabled = false;
        for cost in [0.0, 0.5, 2.89] {
            inputs.cost_pips_round_trip = cost;
            let result = max_achievable_payoff(&inputs).unwrap();
            let mut observed = 0.0_f64;
            for step in 0..=14_000 {
                let sl = 6.0 + f64::from(step) / 1_000.0;
                let tp = (2.5 * sl).min(45.0);
                observed = observed.max((tp - cost) / (sl + cost));
            }
            assert!((result.initializer_ceiling - observed).abs() < 1e-12);
        }
    }

    #[test]
    fn cost_exceeding_the_take_profit_is_named_not_swallowed() {
        let mut i = production_inputs();
        i.trailing_enabled = false;
        i.cost_pips_round_trip = 60.0;
        let c = max_achievable_payoff(&i).unwrap();
        assert_eq!(c.arithmetic_ceiling, 0.0);
        assert_eq!(c.binding, BindingConstraint::CostExceedsTakeProfit);
        // Costs still appear explicitly; candidate net profitability remains
        // mandatory downstream rather than inferred from this coarse input.
        assert!(assert_payoff_floor_reachable(0.5, &i).is_ok());
    }

    #[test]
    fn non_finite_inputs_fail_loudly_rather_than_producing_nan() {
        let mut i = production_inputs();
        i.cost_pips_round_trip = f64::NAN;
        let err = max_achievable_payoff(&i).unwrap_err();
        assert!(format!("{err}").contains("cost_pips_round_trip"));

        let mut i = production_inputs();
        i.sl_min_pips = 0.0;
        assert!(max_achievable_payoff(&i).is_err());
    }

    #[test]
    fn win_rate_arithmetic_is_the_textbook_one() {
        let mut i = production_inputs();
        i.trailing_enabled = false;
        let c = assert_payoff_floor_reachable(2.0, &i).unwrap();
        assert!((c.required_win_rate_at_floor - 1.0 / 3.0).abs() < 1e-12);
        assert!((c.zero_edge_base_rate - 6.0 / 51.0).abs() < 1e-12);
        assert!((c.breakeven_win_rate_at_ceiling - 8.89 / 51.0).abs() < 1e-12);
        assert!(c.edge_points_required_to_break_even() > 0.0);
    }

    #[test]
    fn cost_conversion_charges_commission_in_pips() {
        // 45 USD per 1M ~= 4.50 USD per 0.1M... the shape that matters is that a
        // commission of one pip-value-per-lot adds exactly one pip.
        assert!((cost_pips_round_trip(1.5, 10.0, 10.0) - 2.5).abs() < 1e-12);
        // No conversion available → spread alone, never a silent zero.
        assert!((cost_pips_round_trip(1.5, 10.0, 0.0) - 1.5).abs() < 1e-12);
        assert!((cost_pips_round_trip(1.5, 10.0, f64::NAN) - 1.5).abs() < 1e-12);
    }

    /// Two runs that search different spaces must not share a `config_hash`.
    ///
    /// `rr_min` moves the prefilter's label geometry
    /// (`label_rr = 0.5 × (rr_min + rr_max)`), so it decides which features the
    /// GA is even shown. It is NOT recoverable from the four pip clamps —
    /// `rr_max` is `tp_max / sl_max`, `rr_min` is not expressible — so before it
    /// was stamped these two runs hashed identically and a ledger asserted they
    /// were the same experiment.
    #[test]
    fn the_stamp_separates_runs_that_differ_only_in_the_lower_rr_bound() {
        let mut config = crate::discovery::DiscoveryConfig::default();
        config.target_profile.min_payoff_ratio = 0.5;
        config.evaluation_symbol = "EURUSD".to_owned();
        config.evaluation_spread_pips = 1.89;
        config.evaluation_commission_per_trade = 10.0;
        let stamp_for = |rr_min: f64| {
            let mut inputs = production_inputs();
            inputs.initializer_rr_min = rr_min;
            inputs.cost_pips_round_trip = cost_pips_round_trip(
                config.evaluation_spread_pips,
                config.evaluation_commission_per_trade,
                10.0,
            );
            let ceiling = assert_payoff_floor_reachable(0.5, &inputs).expect("reachable floor");
            stamp_resolved_config(&config, &inputs, ceiling, 10.0, false).expect("stamp")
        };
        let a = stamp_for(1.5);
        let b = stamp_for(2.0);
        assert_eq!(a.initializer_rr_min, 1.5);
        assert_eq!(b.initializer_rr_min, 2.0);
        assert_ne!(
            a.config_hash, b.config_hash,
            "rr_min changes the prefilter's label geometry and therefore the \
             feature ranking; two such runs must not claim the same identity"
        );
        // And the same inputs must still be stable, or the hash is noise.
        assert_eq!(stamp_for(1.5).config_hash, a.config_hash);
    }

    #[test]
    fn legacy_arithmetic_cannot_reuse_the_current_config_cache_key() {
        let mut config = DiscoveryConfig::default();
        config.evaluation_symbol = "EURUSD".to_owned();
        config.target_profile.min_payoff_ratio = 0.5;
        config.evaluation_spread_pips = 1.89;
        config.evaluation_commission_per_trade = 10.0;
        let mut inputs = production_inputs();
        inputs.cost_pips_round_trip = cost_pips_round_trip(
            config.evaluation_spread_pips,
            config.evaluation_commission_per_trade,
            10.0,
        );
        let diagnostic = assert_payoff_floor_reachable(0.5, &inputs).unwrap();
        let mut stamp = stamp_resolved_config(&config, &inputs, diagnostic, 10.0, false).unwrap();
        let mut old_body = serde_json::to_value(stamp.hash_body_v2()).unwrap();
        let source = old_body
            .as_object_mut()
            .unwrap()
            .remove("cpu_execution_source_sha256")
            .unwrap();
        assert_eq!(source.as_str(), Some(cpu_execution_source_sha256()));
        let old_hash = crate::artifact_io::stable_json_hash(&old_body).unwrap();
        assert_ne!(stamp.config_hash, old_hash);
        stamp.config_hash = old_hash;
        assert!(
            stamp
                .validate()
                .unwrap_err()
                .to_string()
                .contains("self-hash mismatch")
        );
    }

    #[test]
    fn the_stamp_separates_population_auto_and_stage1_search_semantics() {
        let mut base = crate::discovery::DiscoveryConfig::default();
        base.target_profile.min_payoff_ratio = 0.5;
        base.evaluation_symbol = "EURUSD".to_owned();
        base.evaluation_spread_pips = 1.89;
        base.evaluation_commission_per_trade = 10.0;
        let stamp_for = |config: &crate::discovery::DiscoveryConfig| {
            let mut inputs = production_inputs();
            inputs.cost_pips_round_trip = cost_pips_round_trip(
                config.evaluation_spread_pips,
                config.evaluation_commission_per_trade,
                10.0,
            );
            let ceiling = assert_payoff_floor_reachable(0.5, &inputs).expect("reachable floor");
            stamp_resolved_config(config, &inputs, ceiling, 10.0, false)
                .expect("resolved config stamp")
                .config_hash
        };
        let expected = stamp_for(&base);

        let mut mutations: Vec<(&str, crate::discovery::DiscoveryConfig)> = Vec::new();
        let mut config = base.clone();
        config.population_auto = !config.population_auto;
        mutations.push(("population_auto", config));
        let mut config = base.clone();
        config.max_indicators = config.max_indicators.saturating_add(1);
        mutations.push(("max_indicators", config));
        let mut config = base.clone();
        config.max_rows = config.max_rows.saturating_add(1);
        mutations.push(("max_rows", config));
        let mut config = base.clone();
        config
            .max_rows_by_timeframe
            .insert("M15".to_string(), 123_456);
        mutations.push(("max_rows_by_timeframe", config));
        let mut config = base.clone();
        config.max_hours += 1.0;
        mutations.push(("max_hours", config));
        let mut config = base.clone();
        config.runtime_overrides.funnel_stage1_pct = 0.5;
        mutations.push(("funnel_stage1_pct", config));
        let mut config = base.clone();
        config.runtime_overrides.stage1_window = crate::discovery::Stage1Window::MostRecent;
        mutations.push(("stage1_window", config));

        for (field, config) in mutations {
            assert_ne!(
                stamp_for(&config),
                expected,
                "selection-changing {field} must change the config hash"
            );
        }

        let mut first_order = base.clone();
        first_order
            .max_rows_by_timeframe
            .insert("H1".to_string(), 50_000);
        first_order
            .max_rows_by_timeframe
            .insert("M15".to_string(), 200_000);
        let mut opposite_order = base;
        opposite_order
            .max_rows_by_timeframe
            .insert("M15".to_string(), 200_000);
        opposite_order
            .max_rows_by_timeframe
            .insert("H1".to_string(), 50_000);
        assert_eq!(
            stamp_for(&first_order),
            stamp_for(&opposite_order),
            "HashMap insertion order must not change the canonical config hash"
        );
    }

    #[test]
    fn current_algorithm_token_is_mandatory_and_rejects_old_or_missing_versions() {
        let config = DiscoveryConfig {
            evaluation_symbol: "EURUSD".to_owned(),
            evaluation_spread_pips: 1.89,
            evaluation_commission_per_trade: 10.0,
            ..DiscoveryConfig::default()
        };
        let mut inputs = production_inputs();
        inputs.cost_pips_round_trip = cost_pips_round_trip(
            config.evaluation_spread_pips,
            config.evaluation_commission_per_trade,
            10.0,
        );
        let ceiling =
            assert_payoff_floor_reachable(config.target_profile.min_payoff_ratio, &inputs).unwrap();
        let stamp = stamp_resolved_config(&config, &inputs, ceiling, 10.0, false).unwrap();
        assert_eq!(stamp.config_hash, stamp.computed_config_hash_v2().unwrap());
        assert_eq!(
            stamp.config_hash,
            stamp_resolved_config(&config, &inputs, ceiling, 10.0, false)
                .unwrap()
                .config_hash
        );

        let mut old_body = stamp.hash_body_v2();
        old_body.search_algorithm_semantics = "neoethos.search.algorithm.two-way.v0";
        let old_hash = crate::artifact_io::stable_json_hash(&old_body).unwrap();
        let encoded = serde_json::to_string(&stamp.hash_body_v2()).unwrap();
        let field = format!(
            "\"search_algorithm_semantics\":{},",
            serde_json::to_string(SEARCH_ALGORITHM_SEMANTICS_V1).unwrap()
        );
        assert_eq!(encoded.matches(&field).count(), 1);
        // Preserve every other byte and its order, including the CURRENT source
        // digest: refusal must depend on the algorithm token, not map ordering.
        let missing = encoded.replacen(&field, "", 1);
        let missing_hash = format!(
            "fnv64:{:016x}",
            crate::artifact_io::fnv1a64(missing.as_bytes())
        );
        for obsolete_hash in [old_hash, missing_hash] {
            assert_ne!(obsolete_hash, stamp.config_hash);
            let mut obsolete = stamp.clone();
            obsolete.config_hash = obsolete_hash;
            let decoded: ResolvedConfigStamp =
                serde_json::from_slice(&serde_json::to_vec(&obsolete).unwrap()).unwrap();
            assert!(
                decoded
                    .validate()
                    .unwrap_err()
                    .to_string()
                    .contains("self-hash mismatch")
            );
        }
    }

    fn cpu_selection_authority_fixture() -> (
        DiscoveryConfig,
        crate::population_auto_sizing_receipt_v1::PopulationAutoSizingReceiptV1,
    ) {
        use crate::population_auto_sizing_receipt_v1::{
            CpuPopulationAutoCalibrationV1, PopulationAutoSizingRequestV1,
            PopulationAutoSizingRouteV1, seal_cpu_population_auto_plan_v1,
            seal_population_auto_sizing_receipt_v1, seal_population_auto_stage1_window_v1,
        };
        let config = DiscoveryConfig {
            population: 200,
            population_auto: true,
            max_indicators: 4,
            evaluation_symbol: "EURUSD".to_owned(),
            evaluation_spread_pips: 1.89,
            evaluation_commission_per_trade: 10.0,
            runtime_overrides: crate::discovery::DiscoveryRuntimeOverrides {
                funnel_stage1_pct: 0.25,
                stage1_window: crate::discovery::Stage1Window::Earliest,
                ..Default::default()
            },
            ..DiscoveryConfig::default()
        };
        // Synthetic measurements exercise the real CPU receipt/authority
        // builders, not a device probe, financial permit, or runtime benchmark.
        let cpu_plan = seal_cpu_population_auto_plan_v1(
            CpuPopulationAutoCalibrationV1 {
                worker_count: 2,
                calibration_candidates: 8,
                calibration_elapsed_ns: 1_000_000_000,
                available_memory_bytes: 8 * 1024 * 1024 * 1024,
                total_memory_bytes: 8 * 1024 * 1024 * 1024,
            },
            25,
            12,
            4,
        )
        .unwrap();
        let receipt = seal_population_auto_sizing_receipt_v1(PopulationAutoSizingRequestV1 {
            population_auto: true,
            configured_population: 200,
            resident_parent_rows: 100,
            evaluation_rows: 25,
            feature_count: 4,
            month_capacity: 12,
            requested_max_indicators: 4,
            migration_enabled: false,
            parent_canonical_scope_identity_sha256: "a".repeat(64),
            parent_dataset_identity_sha256: "b".repeat(64),
            stage1_window: seal_population_auto_stage1_window_v1(
                &"b".repeat(64),
                "selection_stage1",
                0,
                25,
            )
            .unwrap(),
            route: PopulationAutoSizingRouteV1::CpuExplicitResearch {
                contract_identity_sha256: "c".repeat(64),
                input_receipt_sha256: "d".repeat(64),
            },
            cpu_plan: Some(cpu_plan),
        })
        .unwrap();
        (config, receipt)
    }

    #[test]
    fn algorithm_version_propagates_through_the_actual_cpu_selection_hash() {
        let (config, receipt) = cpu_selection_authority_fixture();
        let current =
            build_population_auto_search_authority_v1(&config, &receipt, 10.0, false).unwrap();
        current.validate().unwrap();
        assert_eq!(
            current.search_config_hash(),
            population_auto_semantic_config_hash_for_v1(&config, &receipt, 10.0, false).unwrap()
        );
        let mut old_body = current.resolved_config_stamp.hash_body_v2();
        old_body.search_algorithm_semantics = "neoethos.search.algorithm.two-way.v0";
        let old_hash = crate::artifact_io::stable_json_hash(&old_body).unwrap();
        let mut old = current.clone();
        old.resolved_config_stamp.config_hash = old_hash;
        // Even recomputing the outer unkeyed identity cannot repair an old
        // inner algorithm stamp. This is the method the real CPU path hashes.
        old.search_config_hash = old.computed_search_config_hash_v1().unwrap();
        assert_ne!(old.search_config_hash, current.search_config_hash);
        assert!(
            old.validate()
                .unwrap_err()
                .to_string()
                .contains("self-hash mismatch")
        );
    }

    #[test]
    fn population_auto_growth_goal_changes_selection_hash_and_roundtrips() {
        let (mut config, receipt) = cpu_selection_authority_fixture();
        config.mode = crate::discovery::DiscoveryMode::Risky;
        config.risky_start_balance = 100.0;
        config.risky_target_balance = 50_000.0;
        config.risky_horizon_days = 180.0;
        let original =
            build_population_auto_search_authority_v1(&config, &receipt, 10.0, false).unwrap();
        let expected_goal = crate::scoring::RiskyGrowthGoal {
            start_balance: 100.0,
            target_balance: 50_000.0,
            horizon_days: 180.0,
        };
        assert_eq!(original.growth_goal(), Some(expected_goal));
        let encoded = serde_json::to_vec(&original).unwrap();
        let restored: PopulationAutoSearchAuthorityV1 = serde_json::from_slice(&encoded).unwrap();
        restored.validate().unwrap();
        assert_eq!(restored, original);

        for field in 0..3 {
            let mut changed = config.clone();
            match field {
                0 => changed.risky_start_balance = 200.0,
                1 => changed.risky_target_balance = 60_000.0,
                _ => changed.risky_horizon_days = 240.0,
            }
            let authority =
                build_population_auto_search_authority_v1(&changed, &receipt, 10.0, false).unwrap();
            // The old stamp does not contain these fields: the additional
            // semantic projection must bind all three independently.
            assert_eq!(
                authority.resolved_config_stamp(),
                original.resolved_config_stamp()
            );
            assert_ne!(
                authority.search_config_hash(),
                original.search_config_hash()
            );
            assert!(!authority.semantically_matches(&original).unwrap());
        }
        let mut tampered = original.clone();
        tampered.growth_goal.as_mut().unwrap().horizon_days = 240.0;
        assert!(
            tampered
                .validate()
                .unwrap_err()
                .to_string()
                .contains("semantic hash mismatch")
        );
    }

    #[test]
    fn population_auto_growth_goal_rejects_malformed_and_wrong_mode() {
        let (mut config, receipt) = cpu_selection_authority_fixture();
        config.mode = crate::discovery::DiscoveryMode::Risky;
        config.risky_start_balance = 100.0;
        config.risky_target_balance = 50_000.0;
        config.risky_horizon_days = 180.0;
        let original =
            build_population_auto_search_authority_v1(&config, &receipt, 10.0, false).unwrap();
        for field in 0..3 {
            let mut invalid = config.clone();
            match field {
                0 => invalid.risky_start_balance = 0.0,
                1 => invalid.risky_target_balance = invalid.risky_start_balance,
                _ => invalid.risky_horizon_days = f64::NAN,
            }
            assert!(
                build_population_auto_search_authority_v1(&invalid, &receipt, 10.0, false).is_err()
            );
        }
        let mut malformed = original.clone();
        malformed.growth_goal.as_mut().unwrap().horizon_days = 0.0;
        malformed.search_config_hash = malformed.computed_search_config_hash_v1().unwrap();
        assert!(malformed.validate().is_err());
        for mode in [
            crate::discovery::DiscoveryMode::PropFirm,
            crate::discovery::DiscoveryMode::Strict,
        ] {
            config.mode = mode;
            let mut wrong_mode =
                build_population_auto_search_authority_v1(&config, &receipt, 10.0, false).unwrap();
            assert_eq!(wrong_mode.growth_goal(), None);
            wrong_mode.growth_goal = original.growth_goal();
            wrong_mode.search_config_hash = wrong_mode.computed_search_config_hash_v1().unwrap();
            assert!(
                wrong_mode
                    .validate()
                    .unwrap_err()
                    .to_string()
                    .contains("requires Risky selection mode")
            );
        }
    }

    #[test]
    fn population_auto_growth_goal_preserves_none_wire_and_matches_live_policy() {
        let (mut config, receipt) = cpu_selection_authority_fixture();
        config.mode = crate::discovery::DiscoveryMode::Risky;
        config.risky_start_balance = 100.0;
        config.risky_target_balance = 50_000.0;
        config.risky_horizon_days = 180.0;
        let authority =
            build_population_auto_search_authority_v1(&config, &receipt, 10.0, false).unwrap();
        let stamp = authority.resolved_config_stamp();
        // Use explicit local scalars; the test must not resolve ambient broker
        // metadata or mutate the operation's settings to exercise this boundary.
        let mut evaluation = crate::genetic::EvaluationConfig {
            symbol: stamp.symbol.clone(),
            account_currency: "USD".to_owned(),
            initial_equity: stamp.initial_balance,
            trailing_enabled: stamp.trailing_enabled,
            trailing_atr_multiplier: stamp.trailing_give_back_r,
            trailing_be_trigger_r: stamp.trailing_be_trigger_r,
            trailing_min_lock_pips: stamp.trailing_min_lock_pips,
            pip_value: 0.0001,
            spread_pips: stamp.spread_pips,
            commission_per_trade: stamp.commission_per_trade,
            pip_value_per_lot: stamp.pip_value_per_lot,
            swap_long_pips_per_day: stamp.swap_long_pips_per_day,
            swap_short_pips_per_day: stamp.swap_short_pips_per_day,
            pnl_conversion_fee_rate: 0.0,
            kill_zones_enabled: stamp.kill_zones_enabled,
            session_spread_pips: stamp.session_spread_pips,
            risk_per_trade_min: stamp.risk_per_trade_min,
            risk_per_trade_max: stamp.risk_per_trade_max,
            growth_objective: true,
            growth_goal: authority.growth_goal(),
            ..crate::genetic::EvaluationConfig::default()
        };
        let adaptive = crate::stop_target::ResolvedAdaptiveStopsPolicyV1::checked_new(
            crate::stop_target::StopTargetSettings {
                vol_estimator: "parkinson".to_owned(),
                atr_stop_multiplier: 1.5,
                ..crate::stop_target::StopTargetSettings::default()
            },
            true,
            2.0,
        )
        .unwrap();
        let policy = crate::live_portfolio::LiveTradingPolicyV1::from_search_authority(
            &authority,
            &evaluation,
            false,
            &adaptive,
        )
        .unwrap();
        assert_eq!(
            policy.sealed_evaluation_config().unwrap().growth_goal,
            authority.growth_goal()
        );
        evaluation.growth_goal.as_mut().unwrap().horizon_days = 240.0;
        assert!(
            crate::live_portfolio::LiveTradingPolicyV1::from_search_authority(
                &authority,
                &evaluation,
                false,
                &adaptive,
            )
            .unwrap_err()
            .to_string()
            .contains("growth goal disagrees")
        );
        evaluation.growth_goal = None;
        assert!(
            crate::live_portfolio::LiveTradingPolicyV1::from_search_authority(
                &authority,
                &evaluation,
                false,
                &adaptive,
            )
            .is_err()
        );

        // Compatibility is for current-source legacy None framing, not a
        // relaxation of the resolved stamp's compiled-source identity check.
        let mut legacy = authority.clone();
        legacy.growth_goal = None;
        legacy.search_config_hash = legacy.computed_search_config_hash_v1().unwrap();
        legacy.validate().unwrap();
        let mut old_projection =
            serde_json::to_value(authority.selection_semantics_v1().unwrap()).unwrap();
        old_projection
            .as_object_mut()
            .unwrap()
            .remove("growth_goal");
        assert_eq!(
            serde_json::to_value(legacy.selection_semantics_v1().unwrap()).unwrap(),
            old_projection
        );
        let encoded = serde_json::to_vec(&legacy).unwrap();
        assert!(!String::from_utf8_lossy(&encoded).contains("growth_goal"));
        let restored: PopulationAutoSearchAuthorityV1 = serde_json::from_slice(&encoded).unwrap();
        restored.validate().unwrap();
        assert_eq!(serde_json::to_vec(&restored).unwrap(), encoded);
        crate::live_portfolio::LiveTradingPolicyV1::from_search_authority(
            &restored,
            &evaluation,
            false,
            &adaptive,
        )
        .unwrap();
        evaluation.growth_goal = authority.growth_goal();
        assert!(
            crate::live_portfolio::LiveTradingPolicyV1::from_search_authority(
                &restored,
                &evaluation,
                false,
                &adaptive,
            )
            .is_err()
        );
    }

    #[test]
    fn public_stamp_constructor_never_emits_a_cost_contradiction() {
        let mut config = crate::discovery::DiscoveryConfig::default();
        config.evaluation_symbol = "EURUSD".to_owned();
        config.evaluation_spread_pips = 1.89;
        config.evaluation_commission_per_trade = 10.0;
        config.target_profile.min_payoff_ratio = 0.5;
        let mut inputs = production_inputs();
        inputs.cost_pips_round_trip = cost_pips_round_trip(
            config.evaluation_spread_pips,
            config.evaluation_commission_per_trade,
            10.0,
        );
        let ceiling = assert_payoff_floor_reachable(0.5, &inputs).expect("reachable floor");
        let valid = stamp_resolved_config(&config, &inputs, ceiling, 10.0, false)
            .expect("consistent stamp");
        valid.validate().expect("constructor output self-validates");

        let mut contradictory = inputs;
        contradictory.cost_pips_round_trip += 0.1;
        let contradictory_ceiling =
            assert_payoff_floor_reachable(0.5, &contradictory).expect("reachable floor");
        assert!(
            stamp_resolved_config(&config, &contradictory, contradictory_ceiling, 10.0, false,)
                .is_err(),
            "caller-supplied derived cost cannot contradict the raw config fields"
        );
    }

    #[test]
    fn payoff_geometry_rejects_reversed_or_nonpositive_search_bounds() {
        let base = production_inputs();
        let mut cases = Vec::new();
        let mut inputs = base;
        inputs.sl_max_pips = inputs.sl_min_pips - 1.0;
        cases.push(("reversed stop clamp", inputs));
        let mut inputs = base;
        inputs.tp_min_pips = inputs.tp_max_pips + 1.0;
        cases.push(("reversed take-profit clamp", inputs));
        let mut inputs = base;
        inputs.sl_max_pips = 0.0;
        cases.push(("nonpositive stop maximum", inputs));
        let mut inputs = base;
        inputs.tp_min_pips = 0.0;
        cases.push(("nonpositive take-profit minimum", inputs));
        let mut inputs = base;
        inputs.initializer_rr_min = inputs.initializer_rr_max + 0.1;
        cases.push(("reversed initializer RR", inputs));
        let mut inputs = base;
        inputs.initializer_rr_min = 0.0;
        cases.push(("nonpositive initializer RR minimum", inputs));
        let mut inputs = base;
        inputs.initializer_rr_max = 0.0;
        cases.push(("nonpositive initializer RR maximum", inputs));
        let mut inputs = base;
        inputs.atr_pips = Some(0.0);
        cases.push(("nonpositive ATR scale", inputs));
        let mut inputs = base;
        inputs.atr_pips = Some(f64::NAN);
        cases.push(("nonfinite ATR scale", inputs));

        for (name, inputs) in cases {
            assert!(
                max_achievable_payoff(&inputs).is_err(),
                "{name} must not mint a payoff/search authority"
            );
        }
    }

    #[test]
    fn strict_stamp_rejects_every_nonfinite_or_out_of_domain_persisted_float() {
        fn valid_stamp() -> ResolvedConfigStamp {
            let mut config = crate::discovery::DiscoveryConfig::default();
            config.evaluation_symbol = "EURUSD".to_owned();
            config.evaluation_spread_pips = 1.89;
            config.evaluation_commission_per_trade = 10.0;
            config.target_profile.min_payoff_ratio = 0.5;
            let mut inputs = production_inputs();
            inputs.cost_pips_round_trip = cost_pips_round_trip(
                config.evaluation_spread_pips,
                config.evaluation_commission_per_trade,
                10.0,
            );
            let ceiling =
                assert_payoff_floor_reachable(0.5, &inputs).expect("reachable payoff floor");
            stamp_resolved_config(&config, &inputs, ceiling, 10.0, false)
                .expect("valid strict stamp")
        }

        fn reject(name: &str, mutate: fn(&mut ResolvedConfigStamp)) {
            let mut stamp = valid_stamp();
            mutate(&mut stamp);
            stamp.config_hash = stamp
                .computed_config_hash_v2()
                .expect("an attacker can recompute the unkeyed inner hash");
            assert!(
                stamp.validate().is_err(),
                "rehashed invalid persisted field `{name}` must fail closed"
            );
        }

        let nonfinite: &[(&str, fn(&mut ResolvedConfigStamp))] = &[
            ("min_net_expectancy_per_trade", |s| {
                s.min_net_expectancy_per_trade = f64::NAN
            }),
            ("min_expectancy_t_stat", |s| {
                s.min_expectancy_t_stat = f64::NAN
            }),
            ("payoff_floor", |s| s.payoff_floor = f64::NAN),
            ("min_win_rate", |s| s.min_win_rate = f64::NAN),
            ("max_in_market", |s| s.max_in_market = f64::NAN),
            ("sl_clamp_pips.0", |s| s.sl_clamp_pips.0 = f64::NAN),
            ("sl_clamp_pips.1", |s| s.sl_clamp_pips.1 = f64::NAN),
            ("tp_clamp_pips.0", |s| s.tp_clamp_pips.0 = f64::NAN),
            ("tp_clamp_pips.1", |s| s.tp_clamp_pips.1 = f64::NAN),
            ("initializer_rr_max", |s| s.initializer_rr_max = f64::NAN),
            ("initializer_rr_min", |s| s.initializer_rr_min = f64::NAN),
            ("band_atr_pips", |s| s.band_atr_pips = Some(f64::NAN)),
            ("trailing_be_trigger_r", |s| {
                s.trailing_be_trigger_r = f64::NAN
            }),
            ("trailing_give_back_r", |s| {
                s.trailing_give_back_r = f64::NAN
            }),
            ("trailing_min_lock_pips", |s| {
                s.trailing_min_lock_pips = f64::NAN
            }),
            ("spread_pips", |s| s.spread_pips = f64::NAN),
            ("commission_per_trade", |s| {
                s.commission_per_trade = f64::NAN
            }),
            ("pip_value_per_lot", |s| s.pip_value_per_lot = f64::NAN),
            ("cost_pips_round_trip", |s| {
                s.cost_pips_round_trip = f64::NAN
            }),
            ("swap_long_pips_per_day", |s| {
                s.swap_long_pips_per_day = f64::NAN
            }),
            ("swap_short_pips_per_day", |s| {
                s.swap_short_pips_per_day = f64::NAN
            }),
            ("session_spread_pips[0]", |s| {
                s.session_spread_pips = Some([f64::NAN, 1.0, 1.0])
            }),
            ("session_spread_pips[1]", |s| {
                s.session_spread_pips = Some([1.0, f64::NAN, 1.0])
            }),
            ("session_spread_pips[2]", |s| {
                s.session_spread_pips = Some([1.0, 1.0, f64::NAN])
            }),
            ("cost_band_pips.0", |s| {
                s.cost_band_pips = Some((f64::NAN, 2.4))
            }),
            ("cost_band_pips.1", |s| {
                s.cost_band_pips = Some((1.6, f64::NAN))
            }),
            ("prefilter_insample_frac", |s| {
                s.prefilter_insample_frac = f64::NAN
            }),
            ("cpcv_embargo_pct", |s| s.cpcv_embargo_pct = f64::NAN),
            ("cpcv_purge_pct", |s| s.cpcv_purge_pct = f64::NAN),
            ("max_hours", |s| s.max_hours = f64::NAN),
            ("max_hours positive infinity", |s| {
                s.max_hours = f64::INFINITY
            }),
            ("funnel_stage1_pct", |s| s.funnel_stage1_pct = f64::NAN),
            ("initial_balance", |s| s.initial_balance = f64::NAN),
            ("risk_per_trade_min", |s| s.risk_per_trade_min = f64::NAN),
            ("risk_per_trade_max", |s| s.risk_per_trade_max = f64::NAN),
            ("payoff_ceiling.arithmetic_ceiling", |s| {
                s.payoff_ceiling.arithmetic_ceiling = f64::NAN
            }),
            ("payoff_ceiling.initializer_ceiling", |s| {
                s.payoff_ceiling.initializer_ceiling = f64::NAN
            }),
            ("payoff_ceiling.ceiling_tp_pips", |s| {
                s.payoff_ceiling.ceiling_tp_pips = f64::NAN
            }),
            ("payoff_ceiling.ceiling_sl_pips", |s| {
                s.payoff_ceiling.ceiling_sl_pips = f64::NAN
            }),
            ("payoff_ceiling.trailing_armed_floor_payoff", |s| {
                s.payoff_ceiling.trailing_armed_floor_payoff = Some(f64::NAN)
            }),
            ("payoff_ceiling.enforced_ceiling", |s| {
                s.payoff_ceiling.enforced_ceiling = f64::NAN
            }),
            ("payoff_ceiling.required_win_rate_at_floor", |s| {
                s.payoff_ceiling.required_win_rate_at_floor = f64::NAN
            }),
            ("payoff_ceiling.breakeven_win_rate_at_ceiling", |s| {
                s.payoff_ceiling.breakeven_win_rate_at_ceiling = f64::NAN
            }),
            ("payoff_ceiling.zero_edge_base_rate", |s| {
                s.payoff_ceiling.zero_edge_base_rate = f64::NAN
            }),
        ];
        for (name, mutate) in nonfinite {
            reject(name, *mutate);
        }

        let invalid_domains: &[(&str, fn(&mut ResolvedConfigStamp))] = &[
            ("win-rate above one", |s| s.min_win_rate = 1.1),
            ("in-market fraction above one", |s| s.max_in_market = 1.1),
            ("zero ATR scale", |s| s.band_atr_pips = Some(0.0)),
            ("negative trail trigger", |s| s.trailing_be_trigger_r = -0.1),
            ("negative trail give-back", |s| {
                s.trailing_give_back_r = -0.1
            }),
            ("negative trail lock", |s| s.trailing_min_lock_pips = -0.1),
            ("negative spread", |s| s.spread_pips = -0.1),
            ("negative commission", |s| s.commission_per_trade = -0.1),
            ("nonpositive pip value", |s| s.pip_value_per_lot = 0.0),
            ("negative session spread", |s| {
                s.session_spread_pips = Some([-0.1, 1.0, 1.0])
            }),
            ("reversed cost band", |s| {
                s.cost_band_pips = Some((2.4, 1.6))
            }),
            ("negative cost band", |s| {
                s.cost_band_pips = Some((-0.1, 2.4))
            }),
            ("zero prefilter fraction", |s| {
                s.prefilter_insample_frac = 0.0
            }),
            ("prefilter fraction above one", |s| {
                s.prefilter_insample_frac = 1.1
            }),
            ("CPCV embargo above one", |s| s.cpcv_embargo_pct = 1.1),
            ("negative CPCV embargo", |s| s.cpcv_embargo_pct = -0.1),
            ("negative CPCV purge", |s| s.cpcv_purge_pct = -0.1),
            ("CPCV purge above one", |s| s.cpcv_purge_pct = 1.1),
            ("negative max-hours", |s| s.max_hours = -0.1),
            ("nonpositive initial balance", |s| s.initial_balance = 0.0),
            ("negative risk minimum", |s| s.risk_per_trade_min = -0.1),
            ("risk maximum above one", |s| s.risk_per_trade_max = 1.1),
            ("reversed risk band", |s| {
                s.risk_per_trade_min = 0.2;
                s.risk_per_trade_max = 0.1;
            }),
        ];
        for (name, mutate) in invalid_domains {
            reject(name, *mutate);
        }

        let stamp = valid_stamp();
        let encoded = serde_json::to_string(&stamp).expect("strict stamp serializes");
        let decoded: ResolvedConfigStamp =
            serde_json::from_str(&encoded).expect("strict stamp roundtrips");
        assert_eq!(decoded, stamp, "strict stamp must roundtrip losslessly");
        decoded.validate().expect("roundtripped stamp validates");
    }
}
