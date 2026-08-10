//! **Run identity**: what this run was configured to look for, and whether that
//! is arithmetically possible.
//!
//! Two things live here, both landed 2026-08-09 after the adversarial review of
//! the "174 candidates screened, 0 survived" run:
//!
//! 1. [`ResolvedConfigStamp`] — the ~30 decision-critical values the run
//!    actually resolved, plus a hash over them. A prior run could not be
//!    attributed to a config file after the fact: the repo `config.yaml` and the
//!    operator's store config disagreed on `prefilter_top_k` (240 vs 50) and on
//!    the payoff floor (2.0 vs 0.0), and nothing in the artifacts said which one
//!    had been in force. Every run now writes what it resolved.
//!
//! 2. [`assert_payoff_floor_reachable`] — the CONFIG-IDENTITY GATE. It refuses
//!    to start a run whose configured payoff floor cannot be reached under that
//!    run's own trailing settings, SL/TP clamps and charged cost. The whole
//!    point is to make "the screen is broken" impossible to confuse with "the
//!    market said no".
//!
//! ## What this is NOT
//!
//! Nothing here has any expected value in money. The arithmetic below decides
//! whether a question is ASKABLE, not whether the answer is profitable. Exit
//! geometry, clamps, RR and timeframe REDISTRIBUTE the (win-rate, payoff) split;
//! on a driftless price the product is fixed at −cost. Measured on real EURUSD
//! bars: expectancy was −4.15 pips per trade in EVERY trailing configuration
//! while the payoff ratio moved from 0.91 to 2.53. A future reader must not
//! read this module as a profit-seeking change.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::discovery::DiscoveryConfig;

/// Maximum payoff ratio EVER observed on real bars under the production
/// trailing configuration (`trailing_enabled: true`, `be_trigger_r: 1.0`,
/// `atr_multiplier: 1.0`), across the full SL/TP grid and every timeframe.
///
/// MEASUREMENT (2026-08-09, real EURUSD bars, cost charged): payoff 0.87 at
/// sl6/tp45 AND 0.87 at sl6/tp300 — avg_win 6.10 vs 6.11 pips. The take-profit
/// is dead code under this trail: the stop sits at entry the instant a trade
/// touches +1R, and it is tested BEFORE the take-profit on every subsequent
/// bar, so reaching 3R requires climbing 1R→3R without ever retracing 1R from
/// the running high. The best cell anywhere in the grid was 1.08.
///
/// This is an EMPIRICAL constant, not arithmetic. It is used only to refuse a
/// run whose floor sits above a number no cell of a real-bar grid has ever
/// reached under this exit geometry — which is exactly the run that produced
/// "0 of 174" and reported it as a verdict on the market. Re-measure it if the
/// trailing implementation changes; the arithmetic ceiling below does not
/// depend on it.
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
    /// Cost alone exceeds the widest expressible take-profit: no exit can pay.
    CostExceedsTakeProfit,
    /// The trail arms and then concedes the exit before the take-profit is ever
    /// tested. Not an arithmetic bound — an empirical one, see
    /// [`MEASURED_TRAILING_PAYOFF_CEILING`].
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
pub struct PayoffCeiling {
    /// Highest payoff ratio the search space can express AT ALL, over the box
    /// mutation can reach: `(tp_max − c) / (sl_min + c)`.
    ///
    /// This is an exact upper bound with the trail OFF. No exit pays more than
    /// the widest take-profit minus cost; no full-stop loss is smaller than the
    /// tightest stop plus cost.
    pub arithmetic_ceiling: f64,
    /// Same bound restricted to what the INITIALISER can draw, where TP is
    /// additionally capped at `initializer_rr_max × sl`. Generation 0 cannot
    /// exceed this even though later generations can.
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
    pub trailing_armed_floor_payoff: Option<f64>,
    /// The ceiling the gate actually enforces: the arithmetic one, lowered to
    /// [`MEASURED_TRAILING_PAYOFF_CEILING`] when the trail is in the
    /// `give_back <= be_trigger` regime that measurement covered.
    pub enforced_ceiling: f64,
    pub binding: BindingConstraint,
    /// `true` when the trail is ON and LOOSER than its arm trigger
    /// (`give_back > be_trigger`). No real-bar measurement covers that geometry,
    /// so [`Self::enforced_ceiling`] is the barrier arithmetic alone and a floor
    /// that passes here has not been checked against anything empirical. It is
    /// also the reward-hack corner — payoff 2.53 at expectancy −4.18 pips per
    /// trade — which only the net-expectancy gate can refuse.
    #[serde(default)]
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

/// The arithmetic, as a pure function so the gate is verifiable without a run.
///
/// Maximising `(tp − c) / (sl + c)` over the reachable box is monotone in both
/// arguments, so the maximiser is `tp = tp_max`, `sl = sl_min`. Mutation clamps
/// SL and TP INDEPENDENTLY (`evolution_math` mutation arm 2), so the initialiser's
/// reward:risk ceiling does not bind the reachable box — it is reported
/// separately rather than folded in, because a gate must only refuse what is
/// PROVABLY unreachable.
pub fn max_achievable_payoff(inputs: &PayoffCeilingInputs) -> Result<PayoffCeiling> {
    let sl_min = finite("sl_min_pips", inputs.sl_min_pips)?;
    let _sl_max = finite("sl_max_pips", inputs.sl_max_pips)?;
    let _tp_min = finite("tp_min_pips", inputs.tp_min_pips)?;
    let tp_max = finite("tp_max_pips", inputs.tp_max_pips)?;
    let rr_max = finite("initializer_rr_max", inputs.initializer_rr_max)?;
    let cost = finite("cost_pips_round_trip", inputs.cost_pips_round_trip)?;
    let be_trigger = finite("trailing_be_trigger_r", inputs.trailing_be_trigger_r)?;
    let give_back = finite("trailing_give_back_r", inputs.trailing_give_back_r)?;
    let min_lock = finite("trailing_min_lock_pips", inputs.trailing_min_lock_pips)?;

    if sl_min <= 0.0 {
        bail!("payoff-ceiling input `sl_min_pips` must be > 0 (got {sl_min})");
    }
    if tp_max <= 0.0 {
        bail!("payoff-ceiling input `tp_max_pips` must be > 0 (got {tp_max})");
    }
    if cost < 0.0 {
        bail!("payoff-ceiling input `cost_pips_round_trip` must be >= 0 (got {cost})");
    }

    let denom = sl_min + cost;
    let numer = tp_max - cost;
    let arithmetic_ceiling = if numer <= 0.0 { 0.0 } else { numer / denom };

    let init_tp = tp_max.min((rr_max * sl_min).max(0.0));
    let init_numer = init_tp - cost;
    let initializer_ceiling = if init_numer <= 0.0 {
        0.0
    } else {
        init_numer / denom
    };

    // Which term pins it. Cost first (it can dominate outright), then whichever
    // clamp the operator would have to move. Moving `tp_max` raises the
    // numerator 1:1; moving `sl_min` down raises the ratio faster whenever
    // `sl_min < tp_max`, which is the whole configured band, so the TP clamp is
    // the honest "what do I change" answer only when the stop is already at the
    // floor of what a broker will accept. We report the TP clamp when the
    // numerator is the smaller lever and the stop clamp otherwise.
    let mut binding = if numer <= 0.0 {
        BindingConstraint::CostExceedsTakeProfit
    } else if tp_max <= sl_min {
        BindingConstraint::TakeProfitClamp
    } else {
        BindingConstraint::StopClamp
    };

    // WHICH TRAIL REGIME THE MEASUREMENT COVERS. Corrected 2026-08-09 after the
    // adversarial review, in BOTH directions — the first cut was wrong twice:
    //
    //   * `give_back <= be_trigger` (the measured point is give_back == trigger
    //     == 1.0, and anything tighter concedes the exit EARLIER, so its payoff
    //     is bounded by the measured one). The first cut tested
    //     `give_back >= be_trigger`, which let `trailing_stop_multiplier: 0.5`
    //     walk straight past the gate into a configuration whose realised payoff
    //     is STRICTLY LOWER than the one being refused.
    //
    //   * `give_back > be_trigger` is NOT covered and must not be treated as if
    //     it were. It is looser, so it pays MORE: the same review measured
    //     payoff 2.53 at `trail_mult = 3`. Enforcing 1.08 there would refuse a
    //     configuration that demonstrably reaches 2.0 — a false refusal, which
    //     is exactly the failure this gate exists to end, pointed the other way.
    //
    // At and below the trigger, `hi − give_back × sl` sits at or below entry the
    // instant the trail arms, so the `max(entry + min_lock)` floor binds
    // immediately and every armed trade can exit for `min_lock` pips gross.
    // Because the trailed stop is tested BEFORE the take-profit on every later
    // bar (eval.rs:1013-1041), the take-profit only pays on paths that never
    // retrace `give_back × sl` after arming.
    // Two DIFFERENT conditions, deliberately not collapsed into one:
    //   * `armed_at_or_below_entry` — the armed stop is at/below entry, so the
    //     min-lock floor binds and the collapse value is computable. Needs
    //     `give_back >= be_trigger`.
    //   * `trail_bounded_by_measurement` — the realised payoff cannot exceed the
    //     measured 1.08. Needs `give_back <= be_trigger` (tighter than measured
    //     concedes the exit earlier and therefore pays less).
    // They coincide only at `give_back == be_trigger`, which is the point that
    // was actually measured.
    let trail_on = inputs.trailing_enabled;
    let armed_at_or_below_entry = trail_on && give_back >= be_trigger && give_back > 0.0;
    let trail_degenerate = trail_on && give_back <= be_trigger;
    // The loose-trail corner: no measurement bounds it, so the gate falls back
    // to barrier arithmetic. That is the REWARD HACK corner (payoff 2.53 at
    // expectancy −4.18 pips per trade). What stops it is not this gate — it is
    // the unconditional net-expectancy check in `TargetProfile::evaluate`. This
    // flag exists so a run that passed the gate for this reason is identifiable
    // in the ledger afterwards rather than indistinguishable from a measured one.
    let trailing_ceiling_unmeasured = trail_on && give_back > be_trigger;
    let trailing_armed_floor_payoff = if armed_at_or_below_entry {
        Some((min_lock - cost) / denom)
    } else {
        None
    };

    let mut enforced_ceiling = arithmetic_ceiling;
    if trail_degenerate && MEASURED_TRAILING_PAYOFF_CEILING < enforced_ceiling {
        enforced_ceiling = MEASURED_TRAILING_PAYOFF_CEILING;
        binding = BindingConstraint::TrailingGiveBack;
    }
    if trailing_ceiling_unmeasured {
        tracing::warn!(
            target: "neoethos_search::run_identity",
            trailing_be_trigger_r = be_trigger,
            trailing_give_back_r = give_back,
            enforced_ceiling,
            "the trail is LOOSER than the arm trigger — no measurement bounds this exit \
             geometry, so the enforced payoff ceiling is the barrier arithmetic alone. The \
             review measured payoff 2.53 at give-back 3.0 with expectancy −4.18 pips per \
             trade: a payoff floor cannot refuse that configuration and only the \
             net-expectancy gate can."
        );
    }

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

/// THE CONFIG-IDENTITY GATE.
///
/// Refuses to start a run whose configured payoff floor exceeds what its own
/// resolved (trailing config, SL clamp, TP clamp, cost) can produce. A floor of
/// `0.0` disables the profile gate entirely and is always accepted.
///
/// BEHAVIOUR CHANGE, stated explicitly: a configuration that previously ran for
/// hours and reported "0 survived" now REFUSES TO START and prints why. It
/// permits nothing new. It refuses exactly one class of run: the one whose
/// answer was fixed before a bar was read.
pub fn assert_payoff_floor_reachable(
    configured_floor: f64,
    inputs: &PayoffCeilingInputs,
) -> Result<PayoffCeiling> {
    let mut ceiling = max_achievable_payoff(inputs)?;
    if !configured_floor.is_finite() {
        bail!("configured payoff floor is not finite ({configured_floor})");
    }
    // The floor's own break-even win rate, which is what the operator is really
    // asking for when they set it.
    ceiling.required_win_rate_at_floor = if configured_floor > 0.0 {
        1.0 / (1.0 + configured_floor)
    } else {
        1.0 / (1.0 + ceiling.enforced_ceiling)
    };

    if configured_floor <= 0.0 {
        return Ok(ceiling);
    }
    if configured_floor <= ceiling.enforced_ceiling {
        return Ok(ceiling);
    }

    let trail_note = match ceiling.trailing_armed_floor_payoff {
        Some(floor_payoff) => format!(
            "\n  trailing:           ENABLED, arms at {:.2}R and gives back {:.2}R \
             (give-back >= trigger, so the armed stop is floored at entry + {:.1} pips \
             and is tested BEFORE the take-profit on every later bar).\n  \
             every armed trade that is not a clean unretraced run to the take-profit \
             exits for {:.2} pips net, i.e. a payoff of {:.3}.\n  \
             MEASURED on real EURUSD bars under exactly this configuration: payoff 0.87 \
             at sl6/tp45 AND 0.87 at sl6/tp300 (avg_win 6.10 vs 6.11 pips — the \
             take-profit is dead code); best cell anywhere in the grid {:.2}.",
            inputs.trailing_be_trigger_r,
            inputs.trailing_give_back_r,
            inputs.trailing_min_lock_pips,
            inputs.trailing_min_lock_pips - inputs.cost_pips_round_trip,
            floor_payoff,
            MEASURED_TRAILING_PAYOFF_CEILING,
        ),
        None if ceiling.trailing_ceiling_unmeasured => format!(
            "\n  trailing:           ENABLED and LOOSER than its trigger (arms at {:.2}R, \
             gives back {:.2}R). No real-bar measurement covers this geometry, so the ceiling \
             above is the barrier arithmetic alone — the review measured payoff 2.53 here at \
             expectancy -4.18 pips per trade, which a payoff floor cannot refuse.",
            inputs.trailing_be_trigger_r, inputs.trailing_give_back_r,
        ),
        None => "\n  trailing:           off — the ceiling above is the exact barrier \
                 arithmetic."
            .to_string(),
    };

    bail!(
        "REFUSING TO START: the configured payoff floor is unreachable under this run's \
         own settings. This is a report on the configuration, not on the market.\n\
         \n  configured floor:   {floor:.3}  (models.prop_search_min_payoff_ratio)\n  \
         achievable ceiling: {ceiling_v:.3}\n  \
         binding constraint: {binding}\n\
         \n  arithmetic ceiling (trail off): (tp_max {tp:.1} - cost {cost:.2}) / \
         (sl_min {sl:.1} + cost {cost:.2}) = {arith:.3}\n  \
         initialiser-only ceiling (rr <= {rr:.2}):                                  {init:.3}\
         {trail_note}\n\
         \n  WIN RATES at the ceiling barriers (sl {sl:.1} / tp {tp:.1}, cost {cost:.2} pips):\n    \
         required by the configured floor : {req_wr:.2}%  (break-even at payoff {floor:.3})\n    \
         cost-charged break-even          : {be_wr:.2}%\n    \
         zero-edge base rate (driftless)  : {base_wr:.2}%\n    \
         edge that must be paid for       : {edge:+.2} win-rate points\n\
         \n  Every survival path in the quality screen is gated by \
         `target_profile.accepts()` (discovery.rs), so with this floor the run's \
         outcome is decided before a bar is read.\n  \
         Fix one of: lower `models.prop_search_min_payoff_ratio`; turn the discovery \
         trailing off; widen the take-profit clamp; or tighten the stop clamp. \
         NOTE: none of those has any expected value in money — measured, expectancy \
         held at -4.15 pips per trade while the payoff ratio moved from 0.91 to 2.53. \
         They only make the question askable.",
        floor = configured_floor,
        ceiling_v = ceiling.enforced_ceiling,
        binding = ceiling.binding.label(),
        tp = inputs.tp_max_pips,
        sl = inputs.sl_min_pips,
        cost = inputs.cost_pips_round_trip,
        arith = ceiling.arithmetic_ceiling,
        rr = inputs.initializer_rr_max,
        init = ceiling.initializer_ceiling,
        trail_note = trail_note,
        req_wr = ceiling.required_win_rate_at_floor * 100.0,
        be_wr = ceiling.breakeven_win_rate_at_ceiling * 100.0,
        base_wr = ceiling.zero_edge_base_rate * 100.0,
        edge = ceiling.edge_points_required_to_break_even() * 100.0,
    )
}

// ---------------------------------------------------------------------------
// Resolved-config stamp.
// ---------------------------------------------------------------------------

/// The decision-critical values, resolved. Written into the discovery ledger
/// every run so a result can be attributed to a configuration after the fact.
///
/// Deliberately NOT the whole `DiscoveryConfig` (that is
/// `DiscoveryRunProfile`'s job, and it lands next to the portfolio JSON): this
/// is the short list an operator reads first, and the hash that says whether
/// two runs were the same experiment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedConfigStamp {
    /// `fnv64:…` over every other field of this struct, in declaration order.
    /// Two runs with the same hash searched under the same decisions.
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
    #[serde(default)]
    pub min_net_expectancy_per_trade: f64,
    /// `TargetProfile::min_expectancy_t_stat`. `0.0` = the significance bar is
    /// OFF, which is the shipped state.
    #[serde(default)]
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
    #[serde(default)]
    pub initializer_rr_min: f64,
    /// Median ATR (pips) the band was scaled to; `None` = absolute pip band.
    /// Without this the four pip numbers above are un-interpretable across
    /// timeframes.
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
    /// The per-UTC-bucket spread curve, or `None` for a FLAT spread charged at
    /// 03:00 Tokyo and at the London open alike. Without this in the hash, a
    /// flat-spread run and a per-hour-curve run hash identically.
    #[serde(default)]
    pub session_spread_pips: Option<[f64; 3]>,
    /// The band every reported result is measured against, so a ledger can say
    /// what its own `cost_band_*` counts were measured at.
    #[serde(default)]
    pub cost_band_pips: Option<(f64, f64)>,

    // ── the shape of the search ───────────────────────────────────────────
    pub prefilter_top_k: usize,
    pub prefilter_insample_frac: f64,
    pub prefilter_min_per_timeframe: usize,
    /// CPCV settings. They now decide the PREFILTER's refit windows as well as
    /// the validation folds, so two runs that differ here explored different
    /// feature sets and are not the same experiment.
    #[serde(default)]
    pub enable_cpcv: bool,
    #[serde(default)]
    pub cpcv_n_splits: usize,
    #[serde(default)]
    pub cpcv_n_test_groups: usize,
    #[serde(default)]
    pub cpcv_embargo_pct: f64,
    #[serde(default)]
    pub cpcv_purge_pct: f64,
    #[serde(default)]
    pub cpcv_max_rows: usize,
    pub population: usize,
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
    let ceiling = max_achievable_payoff(&inputs).ok()?;
    stamp_resolved_config(config, &inputs, ceiling, pip_value_per_lot, normalize_features)
        .ok()
        .map(|s| s.config_hash)
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
    let mode = format!("{:?}", config.mode);
    let body = StampBody {
        symbol: &config.evaluation_symbol,
        timeframe: &config.timeframe_label,
        mode: &mode,
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
    Ok(ResolvedConfigStamp {
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
    })
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
        // Initialiser can only draw tp <= 2.5 * 6 = 15 → (15 - 2.89)/8.89.
        assert!((c.initializer_ceiling - (12.11 / 8.89)).abs() < 1e-9);
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
    fn trailing_regime_lowers_the_enforced_ceiling_to_the_measured_one() {
        let i = production_inputs();
        let c = max_achievable_payoff(&i).unwrap();
        assert_eq!(c.binding, BindingConstraint::TrailingGiveBack);
        assert!((c.enforced_ceiling - MEASURED_TRAILING_PAYOFF_CEILING).abs() < 1e-12);
        // The armed trade's floor exit is (2.0 - 2.89) / 8.89 — NEGATIVE. The
        // trail turns a would-be loser into a small loss dressed as a win, which
        // is exactly the mechanism that pinned the measured payoff near 1.0.
        let floor_payoff = c.trailing_armed_floor_payoff.expect("armed regime");
        assert!(floor_payoff < 0.0, "min-lock is below the charged cost");
        assert!((floor_payoff - (-0.89 / 8.89)).abs() < 1e-9);
    }

    #[test]
    fn the_gate_refuses_the_run_that_produced_zero_of_174() {
        let i = production_inputs();
        let err = assert_payoff_floor_reachable(2.0, &i).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("REFUSING TO START"), "{msg}");
        // Both numbers must be printed.
        assert!(msg.contains("2.000"), "configured floor missing: {msg}");
        assert!(msg.contains("1.080"), "ceiling missing: {msg}");
        // The binding constraint must be named.
        assert!(msg.contains("trailing give-back"), "{msg}");
        // The required win rate next to the zero-edge base rate.
        assert!(msg.contains("zero-edge base rate"), "{msg}");
        assert!(msg.contains("33.33%"), "1/(1+2) missing: {msg}");
        // 6 / (6 + 45) = 11.7647%
        assert!(msg.contains("11.76%"), "driftless base rate missing: {msg}");
        // (6 + 2.89) / 51 = 17.4314%
        assert!(msg.contains("17.43%"), "break-even win rate missing: {msg}");
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
        // …and it must still refuse a floor above the arithmetic ceiling.
        assert!(assert_payoff_floor_reachable(5.0, &i).is_err());
    }

    /// UPDATED 2026-08-09 — the previous version of this test encoded the hole
    /// the review found. It asserted that `give_back = 0.4` escapes the measured
    /// ceiling, on the reasoning that the armed stop sits ABOVE entry and so the
    /// geometry differs from the measured one. The geometry does differ; the
    /// BOUND does not. A tighter give-back concedes the exit EARLIER, so its
    /// average win — and therefore its payoff — cannot exceed the measured
    /// point's. The old rule let `trailing_stop_multiplier: 0.5` walk past the
    /// gate into a configuration whose realised payoff is strictly LOWER than
    /// the one being refused.
    #[test]
    fn a_tighter_trail_than_the_measured_one_is_still_bounded_by_it() {
        let mut i = production_inputs();
        i.trailing_give_back_r = 0.4;
        let c = max_achievable_payoff(&i).unwrap();
        assert_eq!(c.binding, BindingConstraint::TrailingGiveBack);
        assert!((c.enforced_ceiling - MEASURED_TRAILING_PAYOFF_CEILING).abs() < 1e-12);
        assert!(!c.trailing_ceiling_unmeasured);
        // The armed stop is above entry here, so the min-lock collapse value is
        // NOT applicable and must not be reported as if it were.
        assert!(c.trailing_armed_floor_payoff.is_none());
        // And the gate refuses the 2.0 floor for it, as it does for the measured
        // point — the whole reason the hole mattered.
        assert!(assert_payoff_floor_reachable(2.0, &i).is_err());
    }

    /// The other direction, which the review's proposed fix would have got
    /// wrong: a LOOSER trail pays MORE. The same review measured payoff 2.53 at
    /// give-back 3.0, so borrowing the 1.08 number there would be a FALSE
    /// REFUSAL of a configuration that demonstrably reaches 2.0. The gate must
    /// fall back to barrier arithmetic and SAY that nothing measured covers it.
    #[test]
    fn a_looser_trail_than_the_measured_one_is_unmeasured_not_refused() {
        let mut i = production_inputs();
        i.trailing_give_back_r = 3.0;
        let c = max_achievable_payoff(&i).unwrap();
        assert!(
            c.trailing_ceiling_unmeasured,
            "the ledger must be able to identify a run that passed for this reason"
        );
        assert!((c.enforced_ceiling - c.arithmetic_ceiling).abs() < 1e-12);
        assert_ne!(c.binding, BindingConstraint::TrailingGiveBack);
        // Payoff 2.53 was measured here at expectancy -4.18 pips per trade. The
        // payoff floor cannot refuse it; only the net-expectancy gate can.
        assert!(assert_payoff_floor_reachable(2.0, &i).is_ok());
    }

    #[test]
    fn cost_exceeding_the_take_profit_is_named_not_swallowed() {
        let mut i = production_inputs();
        i.trailing_enabled = false;
        i.cost_pips_round_trip = 60.0;
        let c = max_achievable_payoff(&i).unwrap();
        assert_eq!(c.arithmetic_ceiling, 0.0);
        assert_eq!(c.binding, BindingConstraint::CostExceedsTakeProfit);
        assert!(assert_payoff_floor_reachable(0.5, &i).is_err());
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
        let config = crate::discovery::DiscoveryConfig::default();
        let stamp_for = |rr_min: f64| {
            let mut inputs = production_inputs();
            inputs.initializer_rr_min = rr_min;
            let ceiling = max_achievable_payoff(&inputs).expect("finite inputs");
            stamp_resolved_config(&config, &inputs, ceiling, 10.0, false).expect("stamp")
        };
        let a = stamp_for(1.5);
        let b = stamp_for(3.0);
        assert_eq!(a.initializer_rr_min, 1.5);
        assert_eq!(b.initializer_rr_min, 3.0);
        assert_ne!(
            a.config_hash, b.config_hash,
            "rr_min changes the prefilter's label geometry and therefore the \
             feature ranking; two such runs must not claim the same identity"
        );
        // And the same inputs must still be stable, or the hash is noise.
        assert_eq!(stamp_for(1.5).config_hash, a.config_hash);
    }
}
