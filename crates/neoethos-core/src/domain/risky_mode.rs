//! Risky Mode — autonomous compounding from a small starting balance
//! to a large target ($20 → $50,000 default), per the 2026-05-17
//! operator directive.
//!
//! **This is NOT an FTMO / prop-firm variant.** It is a completely
//! separate operating mode with explicit high-risk semantics:
//!
//! - The operator starts with a small bankroll (default $20) and accepts a
//!   higher failure rate than PropFirm mode, but risk is still sized from the
//!   strategy's measured edge. Deliberately forcing near-certain ruin is not a
//!   trading objective.
//! - The autonomous live loop owns every sizing and entry decision it
//!   sends. Manual-order policy belongs to the separate manual order
//!   route; this manager neither receives nor pretends to gate it.
//! - Risk per trade is a positive fraction no greater than 50 % of the current
//!   bankroll. The shipped fallback is 10 % (half-Kelly for the fallback
//!   52 % / 1.5R edge); production live execution replaces it with the exact
//!   held-out half-Kelly fraction carried by the promoted portfolio.
//!
//! ## Strategy framing — scalp many times, net profit after expenses
//!
//! Earlier drafts of this module described Risky Mode as a "20 pips
//! per day, single high-conviction trade per session" strategy. The
//! 2026-05-17 operator directive retired that framing as a bottleneck:
//! a fixed pip target per day caps the upside before slippage,
//! commission and swap have been considered, and rules out the
//! scalping cadence the bot actually has an edge on. The replacement
//! contract is:
//!
//! - The bot may take **as many trades per day as the signal source
//!   produces**; there is no hardcoded trade-count cap. The per-stage
//!   kill switches (daily-loss, weekly-DD, presend-sanity) bound the
//!   downside; nothing in this module bounds the upside.
//! - The optimisation target is **net profit after expenses**
//!   (commission + spread + swap) accumulated toward
//!   `target_capital_usd`, not a fixed pip count. The producer's job
//!   is to filter to expected-value-positive scalps after costs; the
//!   manager's job is to size them and stop the bleeding.
//! - The [`DEFAULT_RISKY_TRADES_PER_DAY`] constant is now a
//!   throughput *assumption* used only by the days-to-target
//!   estimator — it has no enforcement effect. Operators can tune it
//!   via [`RiskyModeConfig::expected_trades_per_day`] to reflect
//!   their broker's typical fill cadence.
//!
//! The live order path in
//! `crates/neoethos-app/src/app_services/live_trading.rs` constructs and
//! consults this manager.
//!
//! ## Composition with the rest of the risk stack
//!
//! Risky Mode does NOT replace [`crate::domain::risk::RiskManager`]
//! or [`crate::domain::prop_firm::PropFirmConstraints`]. When a
//! `RiskyModeManager` is bound on a `TradingSession`, it is the
//! STRICTER outer gate: `RiskyModeManager::check_trade_allowed`
//! runs first, so a Risky Mode rejection blocks the order before
//! anything downstream sees it.
//!
//! **CORRECTION 2026-08-09 (audit #142/#137).** This paragraph used
//! to say the gate ran *"BEFORE `risk_gate::prop_firm_pre_trade_check`"*
//! and that *"the prop-firm constraints stay in place for the
//! manual-trading path"*. Both halves are false and were false when
//! written:
//!
//! - `risk_gate::prop_firm_pre_trade_check` **does not exist**
//!   anywhere in `crates/`, `desktop/`, `mesh/` or `mcp/`.
//! - [`crate::domain::risk::RiskManager`], which owns the prop-firm
//!   tiers, has **no production constructor** — see the warning on
//!   that struct. Nothing downstream of this manager applies them.
//!
//! So `RiskyModeManager` is not the outer of two gates. On the live
//! path it is, together with the breakers in
//! `neoethos-app/src/app_services/live_trading.rs`, the ONLY gate.
//! Read it that way.
//!
//! ## Numeric convention (operator directive §7.2)
//!
//! All bankroll, price, PnL and risk-fraction values in this module
//! are `f64`. f64 carries ~15-16 decimal digits of mantissa, which
//! is enough to keep cents accurate at the $50,000-target scale.
//! The earlier f32 build is retired with this rebaseline.

use anyhow::{Result, bail};

// ---------------------------------------------------------------------------
// Defaults — operator-directive-derived (2026-05-17 framing).
// Every constant is `pub const` so the live producer and the UI can render
// the canonical value from a single source of truth.
// ---------------------------------------------------------------------------

/// Default starting bankroll in USD. The operator's "$20" framing
/// from `risky_mode_compounding_research.md` §4.1.
pub const DEFAULT_STARTING_CAPITAL_USD: f64 = 20.0;

/// Default target bankroll in USD. The operator's "$50,000" goal
/// from the same source.
pub const DEFAULT_TARGET_CAPITAL_USD: f64 = 50_000.0;

/// Default geometric step between consecutive stages. The bankroll
/// roughly doubles per stage (`$20 → $40 → $80 → …`). Chosen so that
/// the $20 → $50,000 span resolves to ~11 stages at the default
/// factor.
pub const DEFAULT_DOUBLING_FACTOR: f64 = 2.0;

/// Fallback risk-per-trade fraction used only when no promoted strategy edge is
/// available (for example, a standalone scenario projection). With the
/// fallback 52 % win rate and 1.5 reward:risk, full Kelly is 20 % and
/// half-Kelly is 10 %. Live trading does not invent this value: schema-v5
/// portfolios carry their own held-out half-Kelly evidence.
pub const RISKY_MODE_DEFAULT_RISK_PER_TRADE_FRACTION: f64 = 0.10;

/// There is no evidence-based universal positive floor. Zero is exposed as the
/// conceptual lower boundary; validated active stages still require `> 0`.
pub const RISKY_MODE_MIN_RISK_PER_TRADE_FRACTION: f64 = 0.0;

/// Upper bound on the per-trade risk fraction in Risky Mode. Above
/// 50 % a single loss destroys at least half the bankroll. This remains a hard
/// defence-in-depth ceiling, not a target or a minimum.
pub const RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION: f64 = 0.50;

/// Half-Kelly for a binary payoff described by win rate and reward:risk,
/// bounded by the Risky hard ceiling. A non-positive edge returns zero.
pub fn half_kelly_fraction_from_win_rate_reward_to_risk(win_rate: f64, reward_to_risk: f64) -> f64 {
    if !win_rate.is_finite()
        || !reward_to_risk.is_finite()
        || win_rate <= 0.0
        || win_rate >= 1.0
        || reward_to_risk <= 0.0
    {
        return 0.0;
    }
    let loss_rate = 1.0 - win_rate;
    let full_kelly = (win_rate * reward_to_risk - loss_rate) / reward_to_risk;
    (0.5 * full_kelly).clamp(0.0, RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION)
}

/// Pre-broker-send sanity ceiling as a fraction of bankroll. Even
/// with Risky Mode armed and a 50 % per-trade target, no single
/// order whose implied risk exceeds this fraction may leave the
/// process. Defence-in-depth against bugs in our own sizing.
pub const DEFAULT_PRESEND_SANITY_CEILING_FRACTION: f64 = 0.55;

/// Default cumulative monthly-loss cap as a fraction of the current
/// bankroll. At 0.99 it is intentionally looser than the per-day cap.
pub const DEFAULT_MONTHLY_LOSS_CAP_FRACTION: f64 = 0.99;

/// Default trades-per-day **assumption** for the days-to-target
/// estimator. This is a throughput *projection* — it has no gating /
/// rate-limiting effect on the live producer. A scalping cadence of
/// 10 trades/day is a deliberately middle-of-the-road default; the
/// operator can override per-session via
/// [`RiskyModeConfig::expected_trades_per_day`] to reflect their own
/// broker's typical fill latency and the producer's signal frequency.
/// The estimator multiplies trades-to-target by this value for the
/// scenarios API and dashboard.
pub const DEFAULT_RISKY_TRADES_PER_DAY: f64 = 10.0;

/// Default expected win-rate of the bot's signal source AFTER
/// commission + spread + swap have been deducted from each trade's
/// expected value. 0.52 is the honest baseline for a retail
/// scalping setup — slightly better than a coin flip but not by
/// much; the operator can tighten it via
/// [`RiskyModeConfig::expected_win_rate`] when they have empirical
/// evidence of a stronger edge. The §6.4 / §7.1 ruin-probability
/// estimate uses this together with [`DEFAULT_EXPECTED_REWARD_TO_RISK`]
/// and the per-stage `risk_per_trade_fraction` to compute the
/// Brownian-motion estimate; with these defaults the early-stage
/// estimate is ≥ 0.99 (matching the operator-stated 99 % ruin
/// acceptance).
pub const DEFAULT_EXPECTED_WIN_RATE: f64 = 0.52;

/// Default expected reward-to-risk ratio per trade. 1.5 reflects a
/// scalping cadence where take-profit targets are ~1.5× the
/// stop-loss distance. Conservative — the operator can raise it via
/// [`RiskyModeConfig::expected_reward_to_risk`] when their producer
/// targets larger excursions; raising it pulls the ruin-probability
/// estimate DOWN, which is why we default to the conservative side.
pub const DEFAULT_EXPECTED_REWARD_TO_RISK: f64 = 1.5;

// ---------------------------------------------------------------------------
// Stage descriptor (research §4.2).
// ---------------------------------------------------------------------------

/// One bankroll stage in the Risky Mode taper.
///
/// Stages tile the bankroll range from `starting_capital_usd` up to
/// (or past) `target_capital_usd`; the manager picks the active
/// stage by where the live bankroll lands. Each stage carries its
/// own sizing fraction and kill-switch caps. Bankroll stages control capital
/// protection; they do not manufacture a larger bet merely because the account
/// is small. The live fraction comes from held-out strategy evidence.
///
/// Convention: ranges are half-open `[bankroll_lower_usd,
/// bankroll_upper_usd)` so consecutive stages tile the line without
/// overlap. The last stage's `bankroll_upper_usd` is set to
/// `target_capital_usd` (or just past it for hysteresis).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RiskyStage {
    /// Zero-based stage index (`0 = S1` in the research-doc table).
    pub stage_idx: u8,
    /// Inclusive lower bankroll bound (USD).
    pub bankroll_lower_usd: f64,
    /// Exclusive upper bankroll bound (USD).
    pub bankroll_upper_usd: f64,
    /// Fraction of the current bankroll the bot may risk on a
    /// single trade. Must lie in
    /// `(RISKY_MODE_MIN_RISK_PER_TRADE_FRACTION,
    /// RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION]` (i.e. `(0, 0.50]`).
    /// The default table is flat; an explicitly supplied table may taper.
    pub risk_per_trade_fraction: f64,
    /// Maximum number of simultaneously open positions at this
    /// stage. Defaults to 1 in the table built by
    /// [`build_logarithmic_stages`] — Risky Mode in the v0.4.5 build
    /// runs a single-position scalping cadence (one trade at a time,
    /// many trades per day). Multi-position regimes that allow a
    /// second concurrent trade at later stages will set this >1 in a
    /// custom table.
    pub max_concurrent_positions: u8,
    /// Per-pair exposure cap as a fraction of bankroll. Identical
    /// to `risk_per_trade_fraction` in the single-position regime;
    /// kept as a separate knob so multi-position variants in v0.5+
    /// can shrink it independently.
    pub max_pair_exposure_fraction: f64,
    /// Daily-loss kill switch threshold as a fraction of the
    /// stage-entry bankroll. With per-trade risk at 0.30–0.50 the
    /// daily cap has to be generous (a single losing trade can hit
    /// 50 % already); default range is 0.80 → 0.50.
    pub daily_loss_cap_fraction: f64,
    /// Weekly-drawdown kill switch threshold as a fraction of
    /// stage-entry bankroll. Default range 0.95 → 0.60.
    pub weekly_drawdown_cap_fraction: f64,
}

// ---------------------------------------------------------------------------
// Top-level Risky Mode configuration.
// ---------------------------------------------------------------------------

/// Risky Mode configuration consumed by [`RiskyModeManager::new`].
#[derive(Debug, Clone, PartialEq)]
pub struct RiskyModeConfig {
    /// Starting bankroll in USD.
    pub starting_capital_usd: f64,
    /// Target bankroll in USD.
    pub target_capital_usd: f64,
    /// Per-stage bankroll multiplier.
    pub stage_doubling_factor: f64,
    /// Pre-computed stage table; built by
    /// [`build_logarithmic_stages`].
    pub stages: Vec<RiskyStage>,
    /// Cumulative realized loss over the calendar month, as a fraction of the
    /// CURRENT bankroll, above which [`KillSwitchTier::PerMonth`] trips.
    /// At the default [`DEFAULT_MONTHLY_LOSS_CAP_FRACTION`] (0.99) this tier is
    /// effectively inert because the per-day cap binds first.
    pub monthly_loss_cap_fraction: f64,
    /// Pre-broker-send sanity check fraction. No single order's
    /// implied risk may exceed this fraction of the current
    /// bankroll, regardless of stage sizing.
    pub presend_sanity_ceiling_fraction: f64,
    /// Operator-tuned **expected** scalping cadence — used only by
    /// [`RiskyModeManager::estimated_days_to_target`] to convert
    /// trades-to-target into a "trading days" estimate for the
    /// scenarios API. Has no gating effect on the live producer:
    /// the dispatch gate accepts as many signals per day as the
    /// producer emits. Default
    /// [`DEFAULT_RISKY_TRADES_PER_DAY`] = 10.0.
    pub expected_trades_per_day: f64,
    /// Operator-tuned **expected** win-rate after commission, spread,
    /// and swap expenses. Drives the Brownian-motion ruin-probability
    /// estimate via [`RiskyModeManager::current_ruin_probability_estimate`]
    /// and the days-to-target projection. Must lie in `(0.0, 1.0)`.
    /// Default [`DEFAULT_EXPECTED_WIN_RATE`] = 0.52.
    pub expected_win_rate: f64,
    /// Operator-tuned **expected** reward-to-risk ratio per trade —
    /// the TP-distance / SL-distance the producer typically targets.
    /// Must be positive. Default
    /// [`DEFAULT_EXPECTED_REWARD_TO_RISK`] = 1.5.
    pub expected_reward_to_risk: f64,
}

impl Default for RiskyModeConfig {
    /// Returns `$20 → $50,000` in a logarithmic stage table with the
    /// evidence-compatible 10 % fallback risk.
    fn default() -> Self {
        let stages = build_logarithmic_stages(
            DEFAULT_STARTING_CAPITAL_USD,
            DEFAULT_TARGET_CAPITAL_USD,
            DEFAULT_DOUBLING_FACTOR,
        );
        Self {
            starting_capital_usd: DEFAULT_STARTING_CAPITAL_USD,
            target_capital_usd: DEFAULT_TARGET_CAPITAL_USD,
            stage_doubling_factor: DEFAULT_DOUBLING_FACTOR,
            stages,
            monthly_loss_cap_fraction: DEFAULT_MONTHLY_LOSS_CAP_FRACTION,
            presend_sanity_ceiling_fraction: DEFAULT_PRESEND_SANITY_CEILING_FRACTION,
            expected_trades_per_day: DEFAULT_RISKY_TRADES_PER_DAY,
            expected_win_rate: DEFAULT_EXPECTED_WIN_RATE,
            expected_reward_to_risk: DEFAULT_EXPECTED_REWARD_TO_RISK,
        }
    }
}

impl RiskyModeConfig {
    /// Validate the config against the operator-directive hard
    /// floors and the structural invariants the manager relies on.
    pub fn validate(&self) -> Result<()> {
        if !self.starting_capital_usd.is_finite() || self.starting_capital_usd <= 0.0 {
            bail!(
                "starting_capital_usd must be positive and finite, got {}",
                self.starting_capital_usd
            );
        }
        if !self.target_capital_usd.is_finite()
            || self.target_capital_usd <= self.starting_capital_usd
        {
            bail!(
                "target_capital_usd ({}) must be > starting_capital_usd ({})",
                self.target_capital_usd,
                self.starting_capital_usd
            );
        }
        if !self.stage_doubling_factor.is_finite() || self.stage_doubling_factor <= 1.0 {
            bail!(
                "stage_doubling_factor must be > 1.0, got {}",
                self.stage_doubling_factor
            );
        }
        if self.stages.is_empty() {
            bail!("stages must contain at least one RiskyStage");
        }

        // Stage table monotonicity + sizing bounds. Risk may taper but cannot
        // rise as bankroll grows, and every active stage must remain positive
        // and under the catastrophic-loss ceiling.
        for window in self.stages.windows(2) {
            let a = &window[0];
            let b = &window[1];
            if b.bankroll_lower_usd < a.bankroll_upper_usd - f64::EPSILON {
                bail!(
                    "stages must be monotonically increasing: stage {} ends at {} but stage {} starts at {}",
                    a.stage_idx,
                    a.bankroll_upper_usd,
                    b.stage_idx,
                    b.bankroll_lower_usd
                );
            }
            // Risk fraction is allowed to taper down (it normally
            // does); a stage that increases its risk fraction is
            // rejected as a configuration error.
            if b.risk_per_trade_fraction > a.risk_per_trade_fraction + f64::EPSILON {
                bail!(
                    "risk_per_trade_fraction must be non-increasing: stage {} = {} vs stage {} = {}",
                    a.stage_idx,
                    a.risk_per_trade_fraction,
                    b.stage_idx,
                    b.risk_per_trade_fraction
                );
            }
        }
        for stage in &self.stages {
            if stage.bankroll_upper_usd <= stage.bankroll_lower_usd {
                bail!(
                    "stage {} has bankroll_upper_usd ({}) <= bankroll_lower_usd ({})",
                    stage.stage_idx,
                    stage.bankroll_upper_usd,
                    stage.bankroll_lower_usd
                );
            }
            if !stage.risk_per_trade_fraction.is_finite()
                || stage.risk_per_trade_fraction <= RISKY_MODE_MIN_RISK_PER_TRADE_FRACTION
                || stage.risk_per_trade_fraction > RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION
            {
                bail!(
                    "risk_per_trade_fraction must be in ({}, {}], stage {} = {}",
                    RISKY_MODE_MIN_RISK_PER_TRADE_FRACTION,
                    RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION,
                    stage.stage_idx,
                    stage.risk_per_trade_fraction
                );
            }
            if stage.max_concurrent_positions == 0 || stage.max_concurrent_positions > 5 {
                bail!(
                    "max_concurrent_positions must be in [1, 5], stage {} = {}",
                    stage.stage_idx,
                    stage.max_concurrent_positions
                );
            }
            if !(0.0..=1.0).contains(&stage.max_pair_exposure_fraction)
                || stage.max_pair_exposure_fraction <= 0.0
            {
                bail!(
                    "max_pair_exposure_fraction must be in (0, 1.0], stage {} = {}",
                    stage.stage_idx,
                    stage.max_pair_exposure_fraction
                );
            }
            if !(0.0..=1.0).contains(&stage.daily_loss_cap_fraction)
                || stage.daily_loss_cap_fraction <= 0.0
            {
                bail!(
                    "daily_loss_cap_fraction must be in (0, 1.0], stage {} = {}",
                    stage.stage_idx,
                    stage.daily_loss_cap_fraction
                );
            }
            if !(0.0..=1.0).contains(&stage.weekly_drawdown_cap_fraction)
                || stage.weekly_drawdown_cap_fraction <= 0.0
            {
                bail!(
                    "weekly_drawdown_cap_fraction must be in (0, 1.0], stage {} = {}",
                    stage.stage_idx,
                    stage.weekly_drawdown_cap_fraction
                );
            }
        }

        if !self.monthly_loss_cap_fraction.is_finite()
            || self.monthly_loss_cap_fraction <= 0.0
            || self.monthly_loss_cap_fraction > 1.0
        {
            bail!(
                "monthly_loss_cap_fraction must be in (0, 1.0], got {}",
                self.monthly_loss_cap_fraction
            );
        }
        if !self.presend_sanity_ceiling_fraction.is_finite()
            || self.presend_sanity_ceiling_fraction <= 0.0
            || self.presend_sanity_ceiling_fraction > 1.0
        {
            bail!(
                "presend_sanity_ceiling_fraction must be in (0, 1.0], got {}",
                self.presend_sanity_ceiling_fraction
            );
        }
        if !self.expected_trades_per_day.is_finite() || self.expected_trades_per_day <= 0.0 {
            bail!(
                "expected_trades_per_day must be positive and finite, got {}",
                self.expected_trades_per_day
            );
        }
        if !self.expected_win_rate.is_finite()
            || self.expected_win_rate <= 0.0
            || self.expected_win_rate >= 1.0
        {
            bail!(
                "expected_win_rate must be in (0.0, 1.0), got {}",
                self.expected_win_rate
            );
        }
        if !self.expected_reward_to_risk.is_finite() || self.expected_reward_to_risk <= 0.0 {
            bail!(
                "expected_reward_to_risk must be positive and finite, got {}",
                self.expected_reward_to_risk
            );
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Kill-switch hierarchy (research §5).
// ---------------------------------------------------------------------------

/// Tiered kill-switch identifier. Returned by
/// [`RiskyModeManager::check_trade_allowed`] on rejection so the
/// caller can render the right UI banner and tag the right telemetry
/// event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KillSwitchTier {
    /// Per-trade SL was missing or invalid (research §5.1).
    PerTrade,
    /// Cumulative daily loss exceeded the stage cap (research §5.2).
    PerDay,
    /// Cumulative loss for the current ISO week exceeded the stage cap.
    PerWeek,
    /// Bankroll dropped below the previous stage's lower boundary
    /// (research §5.3 — auto-retreat).
    PerStage,
    /// Cumulative monthly drawdown exceeded the ack-ceiling
    /// (research §5.4).
    PerMonth,
    /// Pre-broker-send sanity check rejected the order because its
    /// implied risk exceeded `presend_sanity_ceiling_fraction` of
    /// the bankroll (research §5.7).
    PreSendSanity,
}

// ---------------------------------------------------------------------------
// Live manager.
// ---------------------------------------------------------------------------

/// Live Risky Mode state machine. Owns the bankroll cursor, the
/// accumulated-loss ledgers, and the sticky kill-switch flag.
#[derive(Debug, Clone)]
pub struct RiskyModeManager {
    config: RiskyModeConfig,
    current_stage_idx: u8,
    /// Highest stage this manager has ever occupied. Only ever advances.
    /// The per-stage retreat trigger measures against THIS, not the live
    /// cursor — see [`RiskyModeManager::check_trade_allowed`].
    high_water_stage_idx: u8,
    current_bankroll_usd: f64,
    daily_loss_accumulated_usd: f64,
    weekly_loss_accumulated_usd: f64,
    monthly_loss_accumulated_usd: f64,
    consecutive_losses: u32,
}

impl RiskyModeManager {
    /// Build a new manager from a validated config and an initial
    /// bankroll. The bankroll determines the starting stage.
    pub fn new(config: RiskyModeConfig, initial_bankroll_usd: f64) -> Result<Self> {
        config.validate()?;
        if !initial_bankroll_usd.is_finite() || initial_bankroll_usd <= 0.0 {
            bail!(
                "initial_bankroll_usd must be positive and finite, got {}",
                initial_bankroll_usd
            );
        }
        let stage_idx = locate_stage_idx(&config.stages, initial_bankroll_usd);
        Ok(Self {
            config,
            current_stage_idx: stage_idx,
            high_water_stage_idx: stage_idx,
            current_bankroll_usd: initial_bankroll_usd,
            daily_loss_accumulated_usd: 0.0,
            weekly_loss_accumulated_usd: 0.0,
            monthly_loss_accumulated_usd: 0.0,
            consecutive_losses: 0,
        })
    }

    /// Read-only access to the underlying config.
    pub fn config(&self) -> &RiskyModeConfig {
        &self.config
    }

    /// Current bankroll in USD.
    pub fn current_bankroll_usd(&self) -> f64 {
        self.current_bankroll_usd
    }

    /// Cumulative daily loss in USD (positive number = loss).
    pub fn daily_loss_accumulated_usd(&self) -> f64 {
        self.daily_loss_accumulated_usd
    }

    /// Cumulative weekly loss in USD.
    pub fn weekly_loss_accumulated_usd(&self) -> f64 {
        self.weekly_loss_accumulated_usd
    }

    /// Cumulative monthly loss in USD.
    pub fn monthly_loss_accumulated_usd(&self) -> f64 {
        self.monthly_loss_accumulated_usd
    }

    /// Active stage descriptor.
    pub fn current_stage(&self) -> &RiskyStage {
        &self.config.stages[self.current_stage_idx as usize]
    }

    /// Returns `Ok(())` if a new order at `size_usd` is allowed.
    /// On rejection returns the tripped tier. The size argument is
    /// the notional USD at risk if the SL fires
    /// (`lot_pip_value * sl_pips`).
    pub fn check_trade_allowed(
        &self,
        size_usd: f64,
        proposed_sl_pips: f64,
        proposed_tp_pips: f64,
    ) -> std::result::Result<(), KillSwitchTier> {
        // Per-trade SL must be present (research §5.1).
        if !proposed_sl_pips.is_finite()
            || proposed_sl_pips <= 0.0
            || !proposed_tp_pips.is_finite()
            || proposed_tp_pips <= 0.0
        {
            return Err(KillSwitchTier::PerTrade);
        }

        // Pre-send sanity ceiling (research §5.7).
        let ceiling_usd = self.current_bankroll_usd * self.config.presend_sanity_ceiling_fraction;
        if size_usd >= ceiling_usd {
            return Err(KillSwitchTier::PreSendSanity);
        }

        // Per-day cap (research §5.2).
        let stage = self.current_stage();
        let daily_cap_usd = stage.daily_loss_cap_fraction * self.current_bankroll_usd;
        if self.daily_loss_accumulated_usd >= daily_cap_usd {
            return Err(KillSwitchTier::PerDay);
        }

        // Per-week cap. The accumulator is account-wide and ISO-week scoped at
        // the live call site. Keeping this as its own tier matters: a daily
        // reset must not silently reopen trading after the week is already
        // spent.
        let weekly_cap_usd = stage.weekly_drawdown_cap_fraction * self.current_bankroll_usd;
        if self.weekly_loss_accumulated_usd >= weekly_cap_usd {
            return Err(KillSwitchTier::PerWeek);
        }

        // Per-stage retreat trigger (research §5.3).
        //
        // **Fixed 2026-08-09 — this tier was unreachable.** It used to compare
        // against `self.current_stage_idx`, but EVERY write to that cursor is
        // `locate_stage_idx(stages, bankroll)` (`new`, `sync_bankroll`,
        // `record_trade_outcome`), which by construction returns an index whose
        // `bankroll_lower_usd` is <= the bankroll. The predecessor's lower bound
        // is lower still, so `bankroll < prev.lower` could never hold and
        // `PerStage` could never fire — while the ARMED banner advertised it.
        //
        // A retreat is measured against the HIGH-WATER stage, which only ever
        // advances. Reaching stage N and then falling below stage N-1's floor
        // is the event this tier was written for.
        if self.high_water_stage_idx > 0 {
            let prev = &self.config.stages[(self.high_water_stage_idx as usize) - 1];
            if self.current_bankroll_usd < prev.bankroll_lower_usd {
                return Err(KillSwitchTier::PerStage);
            }
        }

        // Per-month cap (research §5.4).
        let monthly_cap_usd = self.config.monthly_loss_cap_fraction * self.current_bankroll_usd;
        if self.monthly_loss_accumulated_usd >= monthly_cap_usd {
            return Err(KillSwitchTier::PerMonth);
        }

        Ok(())
    }

    /// Position-size in USD for the next trade.
    ///
    /// Returns `bankroll * stage.risk_per_trade_fraction * confidence`,
    /// clamped to `[0, stage.max_pair_exposure_fraction * bankroll]`.
    /// The operator-directive sizing rule — 30 %–50 % of the current
    /// bankroll per trade — is encoded directly in the stage's
    /// `risk_per_trade_fraction`; the confidence multiplier only
    /// *shrinks* the size (so a low-conviction signal trades smaller),
    /// never grows it.
    ///
    /// Returns `0.0` when `entry_confidence <= 0.0`, meaning the
    /// caller should treat the signal as "no trade".
    pub fn calculate_position_size_usd(&self, entry_confidence: f64) -> f64 {
        if !entry_confidence.is_finite() || entry_confidence <= 0.0 {
            return 0.0;
        }
        let stage = self.current_stage();
        let confidence = entry_confidence.clamp(0.0, 1.0);
        let raw = stage.risk_per_trade_fraction * confidence;
        let capped = raw.min(stage.max_pair_exposure_fraction);
        capped * self.current_bankroll_usd
    }

    /// Reconcile the bankroll cursor to an externally observed balance,
    /// WITHOUT touching the loss accumulators or the kill-switch history.
    ///
    /// **Added 2026-08-09 (W3) because wiring the gate without it produces
    /// false refusals.** [`Self::record_trade_outcome`] only sees the trades
    /// THIS engine opened. The real account also moves from manual orders,
    /// other concurrently running engines, deposits, withdrawals and swap — so
    /// `current_bankroll_usd` drifts away from the balance the live sizing
    /// actually multiplies. Left unreconciled, an account that grew from
    /// another engine's wins would size at `0.50 × new_balance` and be measured
    /// against a pre-send ceiling of `0.55 × stale_bankroll`, and
    /// [`Self::check_trade_allowed`] would return
    /// [`KillSwitchTier::PreSendSanity`] on a perfectly correctly sized order.
    ///
    /// Callers should call this with the broker's fresh balance immediately
    /// before `check_trade_allowed`, so every tier is evaluated against the
    /// money that actually exists.
    ///
    /// Deliberately does NOT reset the accumulators: a day's losses are a day's
    /// losses regardless of where the balance came from, and the day cap
    /// (`daily_loss_cap_fraction × bankroll`) correctly tightens as the
    /// bankroll falls and loosens as it recovers.
    ///
    /// Non-finite or non-positive input is ignored — a failed balance fetch
    /// must never silently move the cursor to zero and trip `PerStage`.
    pub fn sync_bankroll(&mut self, bankroll_usd: f64) {
        if !bankroll_usd.is_finite() || bankroll_usd <= 0.0 {
            return;
        }
        self.current_bankroll_usd = bankroll_usd;
        self.current_stage_idx = locate_stage_idx(&self.config.stages, bankroll_usd);
        self.high_water_stage_idx = self.high_water_stage_idx.max(self.current_stage_idx);
    }

    /// Raise the period loss accumulators to AT LEAST the supplied values,
    /// never lowering them.
    ///
    /// **Added 2026-08-09 because the ledger was per-ENGINE while the money is
    /// per-ACCOUNT.** [`Self::record_trade_outcome`] only ever sees the trades
    /// the engine holding this manager opened, but `POST /autonomous/start`
    /// spawns one engine per portfolio and they all trade the SAME cTrader
    /// account. With N engines the account could lose roughly N × the intended
    /// daily cap before any single engine's `daily_loss_accumulated_usd`
    /// reached its own. The same gap made the ledger evaporate on restart,
    /// while the halt it produces is persisted.
    ///
    /// The caller supplies the ACCOUNT's realized losses for the current UTC
    /// day / ISO week / calendar month — the trade journal is the durable,
    /// account-wide record of exactly that. `max` rather than `=` is what makes
    /// the two sources safe to combine: a trade this engine already booked and
    /// the journal has also recorded counts once, not twice, and a close the
    /// journal has not caught up with yet is not forgotten.
    ///
    /// Non-finite or negative inputs are ignored — an unreadable journal must
    /// never be able to LOWER a loss already counted.
    pub fn raise_period_losses(&mut self, day_usd: f64, week_usd: f64, month_usd: f64) {
        let raise = |slot: &mut f64, v: f64| {
            if v.is_finite() && v > *slot {
                *slot = v;
            }
        };
        raise(&mut self.daily_loss_accumulated_usd, day_usd);
        raise(&mut self.weekly_loss_accumulated_usd, week_usd);
        raise(&mut self.monthly_loss_accumulated_usd, month_usd);
    }

    /// Highest stage index this manager has ever occupied (for logging).
    pub fn high_water_stage_idx(&self) -> u8 {
        self.high_water_stage_idx
    }

    /// Update bankroll after a closed trade. Advances or retreats
    /// the stage cursor as needed. Positive `pnl_usd` increases
    /// bankroll, negative decreases.
    ///
    /// The bankroll it computes is a running estimate from THIS engine's
    /// trades; [`Self::sync_bankroll`] overwrites it with the broker's truth at
    /// the next entry. The part that only this method can provide, and the
    /// reason it must be called on every closed trade, is the daily / weekly /
    /// monthly loss ledger the kill switch trips on.
    pub fn record_trade_outcome(&mut self, pnl_usd: f64) {
        if !pnl_usd.is_finite() {
            return;
        }
        self.current_bankroll_usd = (self.current_bankroll_usd + pnl_usd).max(0.0);
        if pnl_usd < 0.0 {
            let loss = -pnl_usd;
            self.daily_loss_accumulated_usd += loss;
            self.weekly_loss_accumulated_usd += loss;
            self.monthly_loss_accumulated_usd += loss;
            self.consecutive_losses = self.consecutive_losses.saturating_add(1);
        } else {
            self.consecutive_losses = 0;
        }
        self.current_stage_idx = locate_stage_idx(&self.config.stages, self.current_bankroll_usd);
        self.high_water_stage_idx = self.high_water_stage_idx.max(self.current_stage_idx);
    }

    /// Reset the daily-loss accumulator (caller decides when —
    /// research §5.2 says 00:00 CET).
    pub fn reset_daily_accumulator(&mut self) {
        self.daily_loss_accumulated_usd = 0.0;
    }

    /// Reset the weekly-loss accumulator.
    pub fn reset_weekly_accumulator(&mut self) {
        self.weekly_loss_accumulated_usd = 0.0;
    }

    /// Reset the monthly-loss accumulator.
    pub fn reset_monthly_accumulator(&mut self) {
        self.monthly_loss_accumulated_usd = 0.0;
    }

    /// Ruin probability estimate using the per-stage
    /// `risk_per_trade_fraction` and the operator-configured
    /// [`RiskyModeConfig::expected_win_rate`] /
    /// [`RiskyModeConfig::expected_reward_to_risk`] pair (defaulting
    /// to the honest 0.52 / 1.5 scalping baseline). With those
    /// defaults the early-stage estimate is ≥ 0.99 — the estimator's
    /// role is to confirm the operator's §7.1 framing numerically,
    /// not to give them a "you might be OK" hope.
    ///
    /// Formula (Brownian-motion ruin, research §9.3):
    /// ```text
    /// P(ruin) ≈ exp(-2 * mu_log * ln(B / B_min) / sigma_sq_log)
    /// ```
    pub fn current_ruin_probability_estimate(&self) -> f64 {
        let stage = self.current_stage();
        let p: f64 = self.config.expected_win_rate;
        let r: f64 = self.config.expected_reward_to_risk;
        let f_eff = stage.risk_per_trade_fraction;
        if f_eff <= 0.0 || f_eff >= 1.0 {
            return 1.0;
        }
        let up = (1.0 + r * f_eff).ln();
        let down_arg = 1.0 - f_eff;
        if down_arg <= 0.0 {
            return 1.0;
        }
        let down = down_arg.ln();
        let mu_log = p * up + (1.0 - p) * down;
        let sigma_sq = p * (1.0 - p) * (up - down).powi(2);
        if mu_log <= 0.0 || sigma_sq <= 0.0 {
            return 1.0;
        }
        // Distance to ruin in log-bankroll units: how far the
        // current bankroll is above $1 (research §6.3 "ruined"
        // definition).
        let b_min: f64 = 1.0;
        if self.current_bankroll_usd <= b_min {
            return 1.0;
        }
        let log_distance = (self.current_bankroll_usd / b_min).ln();
        let exponent = -2.0 * mu_log * log_distance / sigma_sq;
        exponent.exp().clamp(0.0, 1.0)
    }

    /// Estimated trading days to reach the target from the current
    /// bankroll, given the expected per-trade log-growth at the
    /// current stage's `risk_per_trade_fraction` and the operator's
    /// expected scalping cadence
    /// ([`RiskyModeConfig::expected_trades_per_day`]). Mildly
    /// optimistic — late stages are slightly less aggressive, so real
    /// trajectories will run longer than the number returned.
    ///
    /// The estimator assumes the bot scalps `expected_trades_per_day`
    /// times per session (each producing the per-trade log-growth
    /// implied by `risk_per_trade_fraction` and the (p, r) win-rate /
    /// reward-to-risk pair). It does **not** assume any fixed pip
    /// target per trade — strategy framing per operator directive
    /// 2026-05-17.
    ///
    /// Returns `None` when:
    /// - `current_bankroll >= target` (already at or past the goal),
    /// - expected per-trade log-growth is non-positive,
    /// - the configured cadence is non-positive (rejected by
    ///   `validate` but defensively re-checked here).
    pub fn estimated_days_to_target(&self) -> Option<u32> {
        let target = self.config.target_capital_usd;
        let current = self.current_bankroll_usd;
        if !current.is_finite() || !target.is_finite() {
            return None;
        }
        if current >= target {
            return Some(0);
        }
        let stage = self.current_stage();
        let p: f64 = self.config.expected_win_rate;
        let r: f64 = self.config.expected_reward_to_risk;
        let f_eff = stage.risk_per_trade_fraction;
        if f_eff <= 0.0 {
            return None;
        }
        let up = (1.0 + r * f_eff).ln();
        let down_arg = 1.0 - f_eff;
        if down_arg <= 0.0 {
            return None;
        }
        let down = down_arg.ln();
        let mu_log = p * up + (1.0 - p) * down;
        if mu_log <= 0.0 {
            return None;
        }
        let cadence = self.config.expected_trades_per_day;
        if !cadence.is_finite() || cadence <= 0.0 {
            return None;
        }
        let log_distance = (target / current).ln();
        let trades_to_target = log_distance / mu_log;
        let days = (trades_to_target / cadence).ceil();
        if !days.is_finite() || days <= 0.0 {
            return None;
        }
        let capped = days.min(u32::MAX as f64);
        Some(capped as u32)
    }

    /// Percentile-based first-passage-time estimate. Returns the
    /// number of trading days `n` such that
    /// `P(bankroll_n ≥ target) = percentile`, i.e. only `percentile`
    /// fraction of Monte Carlo paths reach `target` by trade `n` or
    /// faster. Operator-facing meaning: "if you got a top-10% run
    /// (`percentile = 0.10`), this is how fast you could hit target".
    ///
    /// Math: under the geometric-Brownian-motion approximation,
    /// `log(B_n) ~ Normal(log(B_0) + μ*n, σ²*n)` where μ and σ are the
    /// per-trade log-return mean and standard deviation. Solve
    /// `μ*n + z*σ*√n = log(target/B_0)` for n, where
    /// `z = Φ⁻¹(1 - percentile)` (positive for low percentiles → faster).
    /// Substituting `u = √n` gives a quadratic in u:
    /// `μ*u² + z*σ*u − log_distance = 0`, solved via the quadratic
    /// formula. Returns `None` in the same regime
    /// `estimated_days_to_target` returns `None` (negative-EV, etc.).
    ///
    /// `percentile` must be in `(0.0, 1.0)`. Common values:
    /// - `0.10` → optimistic "best case among credible runs"
    /// - `0.25` → top-quartile of survivor paths
    /// - `0.50` → median survivor (very close to `estimated_days_to_target`)
    /// - `0.75` → conservative tail
    ///
    /// Surface this through the scenarios API so a "$100→$100K"
    /// configuration produces a meaningful range (best/median/conservative)
    /// rather than a single deterministic number that hides the
    /// variance.
    pub fn estimated_days_to_target_percentile(&self, percentile: f64) -> Option<u32> {
        if !(0.0..1.0).contains(&percentile) || percentile <= 0.0 {
            return None;
        }
        let target = self.config.target_capital_usd;
        let current = self.current_bankroll_usd;
        if !current.is_finite() || !target.is_finite() {
            return None;
        }
        if current >= target {
            return Some(0);
        }
        let stage = self.current_stage();
        let p: f64 = self.config.expected_win_rate;
        let r: f64 = self.config.expected_reward_to_risk;
        let f_eff = stage.risk_per_trade_fraction;
        if f_eff <= 0.0 {
            return None;
        }
        let up = (1.0 + r * f_eff).ln();
        let down_arg = 1.0 - f_eff;
        if down_arg <= 0.0 {
            return None;
        }
        let down = down_arg.ln();
        let mu_log = p * up + (1.0 - p) * down;
        if mu_log <= 0.0 {
            return None;
        }
        let sigma_sq = p * (1.0 - p) * (up - down).powi(2);
        if sigma_sq <= 0.0 {
            return None;
        }
        let sigma = sigma_sq.sqrt();
        let cadence = self.config.expected_trades_per_day;
        if !cadence.is_finite() || cadence <= 0.0 {
            return None;
        }
        let log_distance = (target / current).ln();
        // z = Φ⁻¹(1 - percentile). Use a rational approximation
        // (Beasley-Springer-Moro) good to ~1e-5 over (0, 1).
        let z = inverse_standard_normal_cdf(1.0 - percentile);
        // Quadratic: μ*u² + z*σ*u − log_distance = 0 (u = √n).
        // Discriminant = (z*σ)² + 4*μ*log_distance.
        let disc = (z * sigma).powi(2) + 4.0 * mu_log * log_distance;
        if disc < 0.0 {
            return None;
        }
        let u = (-z * sigma + disc.sqrt()) / (2.0 * mu_log);
        if !u.is_finite() || u <= 0.0 {
            return None;
        }
        let trades = u * u;
        let days = (trades / cadence).ceil();
        if !days.is_finite() || days <= 0.0 {
            return None;
        }
        Some(days.min(u32::MAX as f64) as u32)
    }

    /// Operator-facing time-to-target SCENARIO triple.
    /// Returns `(best_case_days, expected_days, conservative_days,
    /// ruin_probability)`. The API exposes this triple so the full
    /// distribution is visible, not just the mean:
    ///
    /// - `best_case_days` = 10th-percentile (a lucky top-10% run)
    /// - `expected_days` = deterministic (50th-percentile-ish)
    /// - `conservative_days` = 75th-percentile (still a "successful"
    ///   run but on the slow side)
    /// - `ruin_probability` = probability the account hits $1 before
    ///   reaching target, per the Brownian-motion barrier estimate
    ///
    /// All three day numbers are `None` when expected log-growth is
    /// non-positive (matches `estimated_days_to_target` semantics).
    pub fn time_to_target_scenarios(&self) -> TimeToTargetScenarios {
        TimeToTargetScenarios {
            best_case_days: self.estimated_days_to_target_percentile(0.10),
            expected_days: self.estimated_days_to_target(),
            conservative_days: self.estimated_days_to_target_percentile(0.75),
            ruin_probability: self.current_ruin_probability_estimate(),
        }
    }
}

/// Operator-facing time-to-target estimate covering the full
/// distribution rather than a single deterministic number. Returned
/// by [`RiskyModeManager::time_to_target_scenarios`] and surfaced by the
/// Risky Mode dashboard so the operator sees variance honestly.
///
/// All `_days` fields are `None` when the configured edge produces
/// non-positive expected log-growth (the strategy cannot reach the
/// target on average — see `risky_mode_compounding_research.md` §10.5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeToTargetScenarios {
    /// 10th-percentile first-passage time (days). "If everything
    /// breaks right — top 10 % of credible runs — this is how fast
    /// you could plausibly hit target." NEVER claim faster than this.
    pub best_case_days: Option<u32>,
    /// Deterministic mean first-passage time (days). Close to the
    /// median for moderate-variance configurations.
    pub expected_days: Option<u32>,
    /// 75th-percentile first-passage time (days). The "still
    /// successful but slow" case — useful to set operator expectations
    /// about how long a successful run can drag.
    pub conservative_days: Option<u32>,
    /// Brownian-motion barrier estimate of ruin probability from the
    /// current bankroll.
    pub ruin_probability: f64,
}

/// Beasley-Springer-Moro rational approximation to the inverse of
/// the standard-normal CDF Φ⁻¹(p). Accurate to ~1e-5 over `(0, 1)`.
/// Used by [`RiskyModeManager::estimated_days_to_target_percentile`]
/// to convert a percentile into the corresponding z-score without
/// pulling a stats crate into `neoethos-core`'s dependency closure.
fn inverse_standard_normal_cdf(p: f64) -> f64 {
    // Coefficients for the central-region approximation.
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    // Coefficients for the tail-region approximation.
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

// ---------------------------------------------------------------------------
// Stage-table construction.
// ---------------------------------------------------------------------------

/// Build the default logarithmic capital-protection ladder. Position risk is
/// flat at [`RISKY_MODE_DEFAULT_RISK_PER_TRADE_FRACTION`]; bankroll alone is
/// not evidence that the strategy's edge changed.
pub fn build_logarithmic_stages(
    starting_capital_usd: f64,
    target_capital_usd: f64,
    doubling_factor: f64,
) -> Vec<RiskyStage> {
    build_logarithmic_stages_at_risk(
        starting_capital_usd,
        target_capital_usd,
        doubling_factor,
        RISKY_MODE_DEFAULT_RISK_PER_TRADE_FRACTION,
    )
}

/// Build the same capital ladder at one explicit, evidence-derived risk
/// fraction. Invalid inputs remain visible in the returned fallback stage so
/// [`RiskyModeConfig::validate`] can refuse them rather than silently clamp.
pub fn build_logarithmic_stages_at_risk(
    starting_capital_usd: f64,
    target_capital_usd: f64,
    doubling_factor: f64,
    risk_per_trade_fraction: f64,
) -> Vec<RiskyStage> {
    // Defensive: bad inputs produce a single trivial stage at the
    // bankroll itself so the manager can still be constructed
    // (validation will reject, but we don't want to panic here).
    if !starting_capital_usd.is_finite()
        || starting_capital_usd <= 0.0
        || !target_capital_usd.is_finite()
        || target_capital_usd <= starting_capital_usd
        || !doubling_factor.is_finite()
        || doubling_factor <= 1.0
    {
        return vec![RiskyStage {
            stage_idx: 0,
            bankroll_lower_usd: starting_capital_usd.max(0.0),
            bankroll_upper_usd: target_capital_usd.max(starting_capital_usd + 1.0),
            risk_per_trade_fraction,
            max_concurrent_positions: 1,
            max_pair_exposure_fraction: risk_per_trade_fraction,
            daily_loss_cap_fraction: 0.80,
            weekly_drawdown_cap_fraction: 0.95,
        }];
    }

    let span = (target_capital_usd / starting_capital_usd).ln();
    let step = doubling_factor.ln();
    let stage_count = (span / step).ceil().max(1.0) as usize;

    let mut stages = Vec::with_capacity(stage_count);
    for i in 0..stage_count {
        let lower = starting_capital_usd * doubling_factor.powi(i as i32);
        let upper_unbounded = lower * doubling_factor;
        let upper = if i + 1 == stage_count {
            upper_unbounded.max(target_capital_usd)
        } else {
            upper_unbounded
        };

        let taper_t: f64 = if stage_count <= 1 {
            0.0
        } else {
            (i as f64) / ((stage_count - 1) as f64)
        };

        // Daily loss cap: 0.80 -> 0.50 linear taper.
        let daily_cap = 0.80 - taper_t * (0.80 - 0.50);
        // Weekly DD cap: 0.95 -> 0.60 linear taper.
        let weekly_cap = 0.95 - taper_t * (0.95 - 0.60);

        stages.push(RiskyStage {
            stage_idx: i as u8,
            bankroll_lower_usd: lower,
            bankroll_upper_usd: upper,
            risk_per_trade_fraction,
            max_concurrent_positions: 1,
            // Single-position regime; pair-exposure equals the
            // per-trade risk fraction at this stage.
            max_pair_exposure_fraction: risk_per_trade_fraction,
            daily_loss_cap_fraction: daily_cap.max(0.50),
            weekly_drawdown_cap_fraction: weekly_cap.max(0.60),
        });
    }
    stages
}

/// Resolve the fallback per-trade fraction for a bankroll stage. The default
/// ladder is flat, so this remains a compatibility helper for callers that do
/// not yet carry promoted OOS sizing evidence.
///
/// Returns `None` for degenerate inputs (non-finite, `target <= start`,
/// `doubling <= 1.0`, or a non-positive bankroll) so the caller can fall
/// back to its own configured fraction instead of trading a wrong size.
pub fn stage_risk_fraction_for_bankroll(
    start_usd: f64,
    target_usd: f64,
    doubling: f64,
    bankroll_usd: f64,
) -> Option<f64> {
    // Reject the degenerate inputs that `build_logarithmic_stages` would
    // paper over with a single fallback stage — a live caller wants a clean
    // `None` so it keeps its own (safer) configured fraction.
    if !start_usd.is_finite()
        || start_usd <= 0.0
        || !target_usd.is_finite()
        || target_usd <= start_usd
        || !doubling.is_finite()
        || doubling <= 1.0
        || !bankroll_usd.is_finite()
        || bankroll_usd <= 0.0
    {
        return None;
    }
    let stages = build_logarithmic_stages(start_usd, target_usd, doubling);
    let idx = locate_stage_idx(&stages, bankroll_usd) as usize;
    stages.get(idx).map(|s| s.risk_per_trade_fraction)
}

// ---------------------------------------------------------------------------
// Internal helpers.
// ---------------------------------------------------------------------------

/// Locate the stage index whose `[lower, upper)` range contains
/// `bankroll_usd`. Bankrolls below the first stage clamp to stage
/// 0; bankrolls above the last stage clamp to the last stage.
fn locate_stage_idx(stages: &[RiskyStage], bankroll_usd: f64) -> u8 {
    if stages.is_empty() {
        return 0;
    }
    if bankroll_usd < stages[0].bankroll_lower_usd {
        return 0;
    }
    for (i, stage) in stages.iter().enumerate() {
        if bankroll_usd >= stage.bankroll_lower_usd && bankroll_usd < stage.bankroll_upper_usd {
            return i as u8;
        }
    }
    (stages.len() - 1) as u8
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_bankroll_relocates_the_stage_without_touching_the_ledgers() {
        // W3 (2026-08-09). `sync_bankroll` exists so the live gate measures a
        // proposed order against the balance that actually exists, not against
        // a cursor that only ever saw one engine's trades.
        let mut m = RiskyModeManager::new(RiskyModeConfig::default(), 20.0).expect("manager");
        assert_eq!(m.current_stage().stage_idx, 0);
        m.record_trade_outcome(-5.0);
        let ledger = m.daily_loss_accumulated_usd();
        assert!((ledger - 5.0).abs() < 1e-9);

        m.sync_bankroll(1_000.0);
        assert!((m.current_bankroll_usd() - 1_000.0).abs() < 1e-9);
        assert!(
            m.current_stage().stage_idx > 0,
            "a 50x bankroll must advance the stage cursor"
        );
        assert!(
            (m.daily_loss_accumulated_usd() - ledger).abs() < 1e-9,
            "reconciling the balance must not launder the day's losses"
        );
    }

    #[test]
    fn sync_bankroll_ignores_a_failed_balance_read() {
        // 0.0 is what `fetch_account_runtime_blocking` yields on failure.
        // Accepting it would zero the bankroll and trip PerStage on the next
        // entry of every running engine.
        let mut m = RiskyModeManager::new(RiskyModeConfig::default(), 640.0).expect("manager");
        let before = m.current_bankroll_usd();
        let stage_before = m.current_stage().stage_idx;
        m.sync_bankroll(0.0);
        m.sync_bankroll(-12.0);
        m.sync_bankroll(f64::NAN);
        m.sync_bankroll(f64::INFINITY);
        assert!((m.current_bankroll_usd() - before).abs() < 1e-9);
        assert_eq!(m.current_stage().stage_idx, stage_before);
    }

    #[test]
    fn config_default_has_operator_directive_constants() {
        let cfg = RiskyModeConfig::default();
        assert_eq!(cfg.starting_capital_usd, 20.0);
        assert_eq!(cfg.target_capital_usd, 50_000.0);
        assert_eq!(cfg.stage_doubling_factor, 2.0);
        assert_eq!(
            cfg.monthly_loss_cap_fraction,
            DEFAULT_MONTHLY_LOSS_CAP_FRACTION
        );
    }

    #[test]
    fn config_default_stage_table_tiles_20_to_50k() {
        let cfg = RiskyModeConfig::default();
        assert!(cfg.stages.len() >= 2, "expected at least 2 stages");
        // First stage starts at $20.
        assert!((cfg.stages[0].bankroll_lower_usd - 20.0).abs() < 1e-9);
        // Last stage ends at or past $50,000.
        let last = cfg.stages.last().expect("at least one stage");
        assert!(
            last.bankroll_upper_usd >= 50_000.0,
            "last stage must extend to target, got {}",
            last.bankroll_upper_usd
        );
        // Stages are monotonically increasing.
        for w in cfg.stages.windows(2) {
            assert!(w[0].bankroll_upper_usd <= w[1].bankroll_lower_usd + 1e-9);
        }
        // The fallback risk is flat across bankroll stages and stays under the
        // hard ceiling. Capital alone is not a new estimate of edge.
        for stage in &cfg.stages {
            assert!(
                (RISKY_MODE_MIN_RISK_PER_TRADE_FRACTION..=RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION)
                    .contains(&stage.risk_per_trade_fraction),
                "stage {} risk_per_trade out of band: {}",
                stage.stage_idx,
                stage.risk_per_trade_fraction
            );
        }
        for w in cfg.stages.windows(2) {
            assert!(
                (w[0].risk_per_trade_fraction - w[1].risk_per_trade_fraction).abs() < 1e-9,
                "default risk_per_trade must remain flat"
            );
        }
    }

    #[test]
    fn new_accepts_a_valid_default_config() {
        let mgr = RiskyModeManager::new(RiskyModeConfig::default(), 20.0)
            .expect("must accept valid config");
        assert_eq!(mgr.current_bankroll_usd(), 20.0);
        assert_eq!(mgr.current_stage().stage_idx, 0);
    }

    #[test]
    fn validate_rejects_non_positive_risk_fraction() {
        let mut cfg = RiskyModeConfig::default();
        cfg.stages[0].risk_per_trade_fraction = 0.0;
        let err = cfg.validate().expect_err("must reject");
        assert!(
            err.to_string().contains("risk_per_trade_fraction"),
            "wrong error: {err}"
        );
    }

    #[test]
    fn validate_rejects_risk_fraction_above_50_percent() {
        let mut cfg = RiskyModeConfig::default();
        cfg.stages[0].risk_per_trade_fraction = 0.51;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_non_monotonic_risk_fraction() {
        let mut cfg = RiskyModeConfig::default();
        // Make stage 0 less aggressive than stage 1.
        cfg.stages[0].risk_per_trade_fraction = 0.30;
        if cfg.stages.len() > 1 {
            cfg.stages[1].risk_per_trade_fraction = 0.45;
            let err = cfg.validate().expect_err("must reject");
            assert!(err.to_string().contains("non-increasing"));
        }
    }

    #[test]
    fn check_trade_allowed_passes_at_default_state() {
        let cfg = RiskyModeConfig::default();
        let mgr = RiskyModeManager::new(cfg, 20.0).expect("manager");
        // Tiny order well inside the per-stage cap.
        let ok = mgr.check_trade_allowed(1.0, 10.0, 30.0);
        assert!(ok.is_ok(), "unexpected reject: {:?}", ok);
    }

    #[test]
    fn check_trade_allowed_rejects_missing_sl() {
        let cfg = RiskyModeConfig::default();
        let mgr = RiskyModeManager::new(cfg, 20.0).expect("manager");
        let res = mgr.check_trade_allowed(1.0, 0.0, 30.0);
        assert_eq!(res, Err(KillSwitchTier::PerTrade));
    }

    #[test]
    fn check_trade_allowed_rejects_when_size_breaches_presend_ceiling() {
        let cfg = RiskyModeConfig::default();
        let mgr = RiskyModeManager::new(cfg, 100.0).expect("manager");
        // presend ceiling default is 0.55 -> 55 USD.
        let res = mgr.check_trade_allowed(60.0, 10.0, 30.0);
        assert_eq!(res, Err(KillSwitchTier::PreSendSanity));
    }

    #[test]
    fn check_trade_allowed_rejects_after_daily_cap_exceeded() {
        let mut mgr = RiskyModeManager::new(RiskyModeConfig::default(), 100.0).expect("manager");
        // Stage 0 daily cap is generous (0.80 in the default table)
        // -> 80 USD. Push the accumulator past it.
        mgr.daily_loss_accumulated_usd = 81.0;
        let res = mgr.check_trade_allowed(1.0, 10.0, 30.0);
        assert_eq!(res, Err(KillSwitchTier::PerDay));
    }

    #[test]
    fn check_trade_allowed_enforces_weekly_cap_until_weekly_reset() {
        let mut mgr = RiskyModeManager::new(RiskyModeConfig::default(), 100.0).expect("manager");
        let weekly_cap = mgr.current_stage().weekly_drawdown_cap_fraction * 100.0;

        mgr.raise_period_losses(0.0, weekly_cap - 0.01, 0.0);
        assert!(
            mgr.check_trade_allowed(1.0, 10.0, 30.0).is_ok(),
            "a weekly loss below the configured cap must remain eligible"
        );

        mgr.raise_period_losses(0.0, weekly_cap, 0.0);
        assert_eq!(
            mgr.check_trade_allowed(1.0, 10.0, 30.0),
            Err(KillSwitchTier::PerWeek),
            "the weekly cap must bind exactly at its configured threshold"
        );

        mgr.reset_daily_accumulator();
        assert_eq!(
            mgr.check_trade_allowed(1.0, 10.0, 30.0),
            Err(KillSwitchTier::PerWeek),
            "a day rollover must not launder a spent ISO week"
        );

        mgr.reset_weekly_accumulator();
        assert!(
            mgr.check_trade_allowed(1.0, 10.0, 30.0).is_ok(),
            "the ISO-week rollover must reopen the weekly gate"
        );
    }

    #[test]
    fn calculate_position_size_uses_risk_per_trade_fraction_times_bankroll() {
        let cfg = RiskyModeConfig::default();
        let mgr = RiskyModeManager::new(cfg, 100.0).expect("manager");
        // At full confidence at stage 0 with 0.50 risk fraction,
        // size should be 50.0.
        let size = mgr.calculate_position_size_usd(1.0);
        let stage0 = mgr.current_stage();
        let expected = stage0.risk_per_trade_fraction * 100.0;
        assert!(
            (size - expected).abs() < 1e-9,
            "size mismatch: got {size}, expected {expected}"
        );
    }

    #[test]
    fn calculate_position_size_scales_with_confidence() {
        let mgr = RiskyModeManager::new(RiskyModeConfig::default(), 100.0).expect("manager");
        let full = mgr.calculate_position_size_usd(1.0);
        let half = mgr.calculate_position_size_usd(0.5);
        assert!(
            (full - 2.0 * half).abs() < 1e-9,
            "confidence scaling broken: full={full}, half={half}"
        );
    }

    #[test]
    fn calculate_position_size_returns_zero_for_non_positive_confidence() {
        let mgr = RiskyModeManager::new(RiskyModeConfig::default(), 100.0).expect("manager");
        assert_eq!(mgr.calculate_position_size_usd(0.0), 0.0);
        assert_eq!(mgr.calculate_position_size_usd(-0.5), 0.0);
        assert_eq!(mgr.calculate_position_size_usd(f64::NAN), 0.0);
    }

    #[test]
    fn half_kelly_default_avoids_near_certain_ruin() {
        // 52% / 1.5R implies full Kelly 20% and half-Kelly 10%. The default
        // must preserve positive log growth instead of deliberately selecting
        // the old near-certain-ruin 30-50% band.
        let mgr = RiskyModeManager::new(RiskyModeConfig::default(), 20.0).expect("manager");
        let p = mgr.current_ruin_probability_estimate();
        assert!(
            p < 0.01,
            "half-Kelly fallback should keep modeled ruin below 1%, got {p}"
        );
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn ruin_probability_drops_as_bankroll_grows() {
        // Bankrolls near the start ($20) sit in the most aggressive
        // stage (f=0.5) where the model's expected log-growth is
        // negative at the honest 0.52/1.5 defaults → P(ruin) = 1.0
        // by the early-return branch. Bankrolls deep in the taper
        // (e.g. $25k, where f≈0.32) flip the sign of mu_log → the
        // formula returns an exp(-…)-shaped finite probability much
        // smaller than 1. This is the cross-stage sanity property
        // we want pinned: as the operator climbs the stage ladder
        // the ruin estimate eases off.
        let mgr_small = RiskyModeManager::new(RiskyModeConfig::default(), 20.0).expect("small");
        let mgr_large = RiskyModeManager::new(RiskyModeConfig::default(), 25_000.0).expect("large");
        let p_small = mgr_small.current_ruin_probability_estimate();
        let p_large = mgr_large.current_ruin_probability_estimate();
        assert!(
            p_large < p_small,
            "ruin prob did not decrease with bankroll: small={p_small} large={p_large}"
        );
    }

    #[test]
    fn fallback_default_is_half_kelly_for_fallback_edge() {
        let derived = half_kelly_fraction_from_win_rate_reward_to_risk(
            DEFAULT_EXPECTED_WIN_RATE,
            DEFAULT_EXPECTED_REWARD_TO_RISK,
        );
        assert!(
            (RISKY_MODE_DEFAULT_RISK_PER_TRADE_FRACTION - 0.10).abs() < 1e-9,
            "half-Kelly fallback must remain 0.10, got {}",
            RISKY_MODE_DEFAULT_RISK_PER_TRADE_FRACTION
        );
        assert!((derived - RISKY_MODE_DEFAULT_RISK_PER_TRADE_FRACTION).abs() < 1e-12);
    }

    #[test]
    fn ruin_probability_eases_with_stronger_operator_edge() {
        // If the operator empirically demonstrates a stronger edge
        // (raise expected_win_rate to 0.60 with reward_to_risk 2.0)
        // the model's ruin estimate at the same stage / bankroll
        // must drop. Pins the (p, r) sensitivity so a future refactor
        // that drops the config wiring gets caught.
        let mut cfg_thin = RiskyModeConfig::default();
        cfg_thin.expected_win_rate = 0.52;
        cfg_thin.expected_reward_to_risk = 1.5;
        let mut cfg_fat = RiskyModeConfig::default();
        cfg_fat.expected_win_rate = 0.60;
        cfg_fat.expected_reward_to_risk = 2.0;
        // Pick a bankroll where the taper has reduced f enough that
        // BOTH parameter combinations produce mu_log > 0 — that way
        // we're comparing two finite estimates, not 1.0 vs anything.
        let bankroll = 25_000.0;
        let mgr_thin = RiskyModeManager::new(cfg_thin, bankroll).expect("thin");
        let mgr_fat = RiskyModeManager::new(cfg_fat, bankroll).expect("fat");
        let p_thin = mgr_thin.current_ruin_probability_estimate();
        let p_fat = mgr_fat.current_ruin_probability_estimate();
        assert!(
            p_fat < p_thin,
            "stronger edge must drop ruin estimate: thin={p_thin} fat={p_fat}"
        );
    }

    #[test]
    fn validate_rejects_bad_expected_win_rate() {
        let mut cfg = RiskyModeConfig::default();
        cfg.expected_win_rate = 0.0;
        assert!(cfg.validate().is_err());
        cfg.expected_win_rate = 1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_non_positive_reward_to_risk() {
        let mut cfg = RiskyModeConfig::default();
        cfg.expected_reward_to_risk = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn days_to_target_returns_some_zero_when_at_or_past_target() {
        let cfg = RiskyModeConfig::default();
        let target = cfg.target_capital_usd;
        let mgr = RiskyModeManager::new(cfg, target).expect("manager");
        assert_eq!(mgr.estimated_days_to_target(), Some(0));
    }

    #[test]
    fn days_to_target_is_finite_at_half_kelly_fallback() {
        let mgr = RiskyModeManager::new(RiskyModeConfig::default(), DEFAULT_STARTING_CAPITAL_USD)
            .expect("manager");
        assert!(mgr.estimated_days_to_target().is_some());
    }

    #[test]
    fn no_positive_edge_does_not_invent_a_time_to_profit() {
        let mut cfg = RiskyModeConfig::default();
        cfg.expected_win_rate = 0.4;
        cfg.expected_reward_to_risk = 1.0;
        let mgr = RiskyModeManager::new(cfg, DEFAULT_STARTING_CAPITAL_USD).unwrap();
        assert_eq!(mgr.estimated_days_to_target(), None);
        assert_eq!(
            half_kelly_fraction_from_win_rate_reward_to_risk(0.4, 1.0),
            0.0
        );
    }

    #[test]
    fn operator_risk_below_old_thirty_percent_floor_is_preserved() {
        let mut cfg = RiskyModeConfig::default();
        cfg.stages = build_logarithmic_stages_at_risk(
            cfg.starting_capital_usd,
            cfg.target_capital_usd,
            DEFAULT_DOUBLING_FACTOR,
            0.02,
        );
        cfg.validate().unwrap();
        assert!(
            cfg.stages
                .iter()
                .all(|stage| (stage.risk_per_trade_fraction - 0.02).abs() < 1e-12)
        );
    }

    #[test]
    fn days_to_target_returns_some_when_operator_demonstrates_edge() {
        // Once the operator empirically shows a strong-enough edge
        // (e.g. win-rate 0.60, RR 2.0 — the kind of edge that warrants
        // raising the defaults) the estimator produces a finite
        // figure. We don't band the exact value tightly — the
        // property under test is "estimator works for a credible
        // positive-EV configuration".
        let mut cfg = RiskyModeConfig::default();
        cfg.expected_win_rate = 0.60;
        cfg.expected_reward_to_risk = 2.0;
        let mgr = RiskyModeManager::new(cfg, DEFAULT_STARTING_CAPITAL_USD).expect("manager");
        let days = mgr
            .estimated_days_to_target()
            .expect("finite estimate with edge");
        assert!(
            (1..100_000).contains(&days),
            "20->50k estimate out of plausible band: {days}"
        );
    }

    #[test]
    fn time_to_target_scenarios_returns_ordered_triple() {
        // With a credible positive-EV configuration, the scenarios
        // must come back ordered: best_case < expected < conservative.
        // This pins the percentile-time-to-target semantics: the
        // 10th-percentile (best_case) is faster than the deterministic
        // expectation, which is faster than the 75th-percentile
        // (conservative). Property under test — exact magnitudes
        // depend on the operator's edge so we don't band tightly.
        let mut cfg = RiskyModeConfig::default();
        cfg.expected_win_rate = 0.55;
        cfg.expected_reward_to_risk = 2.0;
        let mgr = RiskyModeManager::new(cfg, DEFAULT_STARTING_CAPITAL_USD).expect("manager");
        let scenarios = mgr.time_to_target_scenarios();
        let best = scenarios.best_case_days.expect("best case finite");
        let expected = scenarios.expected_days.expect("expected finite");
        let conservative = scenarios.conservative_days.expect("conservative finite");
        assert!(
            best <= expected,
            "best_case ({best}) must be ≤ expected ({expected})"
        );
        assert!(
            expected <= conservative,
            "expected ({expected}) must be ≤ conservative ({conservative})"
        );
        // Ruin probability must be a valid probability.
        assert!(
            (0.0..=1.0).contains(&scenarios.ruin_probability),
            "ruin_probability out of [0,1]: {}",
            scenarios.ruin_probability
        );
    }

    #[test]
    fn time_to_target_scenarios_handles_user_chosen_target_100k() {
        // Operator-supplied target test: $100 → $100,000 (= 1000×
        // growth). With a credible strong-edge configuration
        // (win-rate 0.55, RR 2.0, 30%-50% stage risk, 10 trades/day)
        // the deterministic expectation is ~7-8 days, the 10th-
        // percentile is faster (a lucky run), and the 75th-percentile
        // is slower. Property under test: the user can pick ANY
        // positive target larger than start and get a meaningful triple.
        let mut cfg = RiskyModeConfig::default();
        cfg.starting_capital_usd = 100.0;
        cfg.target_capital_usd = 100_000.0;
        cfg.stages = build_logarithmic_stages(100.0, 100_000.0, 2.0);
        cfg.expected_win_rate = 0.55;
        cfg.expected_reward_to_risk = 2.0;
        let mgr = RiskyModeManager::new(cfg, 100.0).expect("manager");
        let scenarios = mgr.time_to_target_scenarios();
        // Deterministic expectation must be in plausible band.
        let expected = scenarios.expected_days.expect("expected days finite");
        assert!(
            (1..=365).contains(&expected),
            "expected days {expected} out of plausible band for $100→$100K strong edge"
        );
        // best_case_days strictly less than expected_days for any
        // realistic variance — pins the percentile ordering for the
        // operator-facing UI.
        let best = scenarios.best_case_days.expect("best case finite");
        assert!(
            best < expected,
            "best_case ({best}) must be strictly less than expected ({expected})"
        );
    }

    #[test]
    fn time_to_target_scenarios_handles_user_chosen_target_50k() {
        // Same operator-facing scenario for the $100 → $50K target
        // (= 500× growth) — must also produce a sensible triple. Pins
        // that the user can configure either common
        // milestone ($50K, $100K, etc.) and the estimator works.
        let mut cfg = RiskyModeConfig::default();
        cfg.starting_capital_usd = 100.0;
        cfg.target_capital_usd = 50_000.0;
        cfg.stages = build_logarithmic_stages(100.0, 50_000.0, 2.0);
        cfg.expected_win_rate = 0.55;
        cfg.expected_reward_to_risk = 2.0;
        let mgr = RiskyModeManager::new(cfg, 100.0).expect("manager");
        let scenarios = mgr.time_to_target_scenarios();
        let expected = scenarios.expected_days.expect("expected days finite");
        assert!(
            (1..=365).contains(&expected),
            "expected days {expected} out of plausible band for $100→$50K strong edge"
        );
    }

    #[test]
    fn time_to_target_scenarios_are_finite_at_half_kelly_fallback() {
        let mgr = RiskyModeManager::new(RiskyModeConfig::default(), DEFAULT_STARTING_CAPITAL_USD)
            .expect("manager");
        let scenarios = mgr.time_to_target_scenarios();
        assert!(scenarios.best_case_days.is_some());
        assert!(scenarios.expected_days.is_some());
        assert!(scenarios.conservative_days.is_some());
        assert!(scenarios.ruin_probability < 0.01);
    }

    #[test]
    fn inverse_standard_normal_cdf_matches_known_values() {
        // Sanity-check the Beasley-Springer-Moro approximation
        // against canonical reference values from any stats text.
        // Tolerance ~1e-4 matches the approximation's stated accuracy.
        let cases = [
            (0.5, 0.0),
            (0.975, 1.96),  // canonical 95% two-sided z
            (0.95, 1.6449), // 90% two-sided
            (0.90, 1.2816), // 80% two-sided / top-10% one-sided
            (0.75, 0.6745), // upper quartile
            (0.025, -1.96), // lower 2.5%
        ];
        for (p, expected_z) in cases {
            let z = inverse_standard_normal_cdf(p);
            assert!(
                (z - expected_z).abs() < 1e-3,
                "Φ⁻¹({p}): expected {expected_z}, got {z}"
            );
        }
    }

    #[test]
    fn days_to_target_scales_inversely_with_cadence() {
        // Use a positive-EV configuration so the estimator returns
        // finite values for both cadences. Then a 10× slower cadence
        // must yield a strictly larger days-to-target figure. Pins
        // the cadence semantics: it's a divisor on trades-to-target,
        // not a gating cap.
        let mut cfg_fast = RiskyModeConfig::default();
        cfg_fast.expected_win_rate = 0.60;
        cfg_fast.expected_reward_to_risk = 2.0;
        cfg_fast.expected_trades_per_day = 50.0;
        let mut cfg_slow = cfg_fast.clone();
        cfg_slow.expected_trades_per_day = 5.0;
        let mgr_fast = RiskyModeManager::new(cfg_fast, DEFAULT_STARTING_CAPITAL_USD).expect("fast");
        let mgr_slow = RiskyModeManager::new(cfg_slow, DEFAULT_STARTING_CAPITAL_USD).expect("slow");
        let fast = mgr_fast.estimated_days_to_target().expect("fast estimate");
        let slow = mgr_slow.estimated_days_to_target().expect("slow estimate");
        assert!(
            slow > fast,
            "slower cadence must yield larger days estimate: fast={fast} slow={slow}"
        );
    }

    #[test]
    fn validate_rejects_non_positive_expected_trades_per_day() {
        let mut cfg = RiskyModeConfig::default();
        cfg.expected_trades_per_day = 0.0;
        let err = cfg.validate().expect_err("must reject zero cadence");
        assert!(
            err.to_string().contains("expected_trades_per_day"),
            "wrong error: {err}"
        );
    }

    #[test]
    fn record_trade_outcome_advances_stage_after_large_profit() {
        let mut mgr = RiskyModeManager::new(RiskyModeConfig::default(), 20.0).expect("manager");
        assert_eq!(mgr.current_stage().stage_idx, 0);
        // Push bankroll into stage 1's range.
        let stage1_lower = mgr.config.stages[1].bankroll_lower_usd;
        let pnl = stage1_lower - mgr.current_bankroll_usd + 0.5;
        mgr.record_trade_outcome(pnl);
        assert!(mgr.current_stage().stage_idx >= 1);
    }

    #[test]
    fn record_trade_outcome_retreats_stage_after_large_loss() {
        let mut mgr = RiskyModeManager::new(RiskyModeConfig::default(), 40.0).expect("manager");
        let starting_idx = mgr.current_stage().stage_idx;
        // Burn the bankroll back to <20 -> stage 0.
        mgr.record_trade_outcome(-mgr.current_bankroll_usd + 1.0);
        assert!(mgr.current_bankroll_usd() > 0.0);
        assert!(
            mgr.current_stage().stage_idx <= starting_idx,
            "stage did not retreat: was {starting_idx} now {}",
            mgr.current_stage().stage_idx
        );
    }

    #[test]
    fn record_trade_outcome_accumulates_losses_separately() {
        let mut mgr = RiskyModeManager::new(RiskyModeConfig::default(), 100.0).expect("manager");
        mgr.record_trade_outcome(-5.0);
        mgr.record_trade_outcome(-7.0);
        mgr.record_trade_outcome(3.0); // ignored by daily accumulator
        assert!((mgr.daily_loss_accumulated_usd() - 12.0).abs() < 1e-9);
        assert!((mgr.weekly_loss_accumulated_usd() - 12.0).abs() < 1e-9);
        assert!((mgr.monthly_loss_accumulated_usd() - 12.0).abs() < 1e-9);
    }

    #[test]
    fn build_logarithmic_stages_handles_degenerate_inputs() {
        let stages = build_logarithmic_stages(-1.0, 100.0, 2.0);
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].stage_idx, 0);
    }

    #[test]
    fn stage_risk_fraction_is_flat_and_rejects_bad_inputs() {
        let early = stage_risk_fraction_for_bankroll(100.0, 50_000.0, 2.0, 120.0).unwrap();
        let late = stage_risk_fraction_for_bankroll(100.0, 50_000.0, 2.0, 40_000.0).unwrap();
        assert_eq!(early, late, "bankroll alone must not change measured edge");
        assert!(
            (RISKY_MODE_MIN_RISK_PER_TRADE_FRACTION..=RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION)
                .contains(&early)
        );
        assert!(
            (RISKY_MODE_MIN_RISK_PER_TRADE_FRACTION..=RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION)
                .contains(&late)
        );
        // Bankroll below the ladder clamps to the first (most aggressive) stage.
        let below = stage_risk_fraction_for_bankroll(100.0, 50_000.0, 2.0, 10.0).unwrap();
        assert_eq!(below, early);
        // Degenerate inputs → None so the live caller keeps its safe fraction.
        assert!(stage_risk_fraction_for_bankroll(100.0, 50.0, 2.0, 100.0).is_none());
        assert!(stage_risk_fraction_for_bankroll(0.0, 50_000.0, 2.0, 100.0).is_none());
        assert!(stage_risk_fraction_for_bankroll(100.0, 50_000.0, 1.0, 100.0).is_none());
        assert!(stage_risk_fraction_for_bankroll(100.0, 50_000.0, 2.0, 0.0).is_none());
    }

    // ── 2026-08-09: the three tiers the review proved were dead ──────────────

    /// `PerStage` could never fire. Every write to `current_stage_idx` is
    /// `locate_stage_idx(stages, bankroll)`, which guarantees
    /// `bankroll >= stages[idx].lower`, and the predecessor's lower bound is
    /// lower still — so the old `bankroll < prev.lower` test was unsatisfiable.
    /// This pins the corrected rule: the trigger is a retreat from the
    /// HIGH-WATER stage.
    #[test]
    fn per_stage_fires_on_a_retreat_below_the_previous_rung_of_the_high_water_stage() {
        let mut cfg = RiskyModeConfig::default();
        cfg.starting_capital_usd = 100.0;
        cfg.target_capital_usd = 50_000.0;
        cfg.stages = build_logarithmic_stages(100.0, 50_000.0, DEFAULT_DOUBLING_FACTOR);
        // Start ON the first rung: no stage has been left behind, so a retreat
        // is not yet definable and the tier must stay silent.
        let mut m = RiskyModeManager::new(cfg, 100.0).expect("manager");
        assert_eq!(m.high_water_stage_idx(), 0);
        m.sync_bankroll(40.0);
        assert!(
            m.check_trade_allowed(1.0, 20.0, 40.0).is_ok(),
            "no stage was ever cleared — PerStage must not fire"
        );

        // Climb to stage 1 (>= 200), then fall back below stage 0's floor.
        m.sync_bankroll(250.0);
        assert_eq!(m.high_water_stage_idx(), 1);
        m.sync_bankroll(150.0);
        assert!(
            m.check_trade_allowed(1.0, 20.0, 40.0).is_ok(),
            "150 is still above stage 0's floor of 100 — not a retreat"
        );
        m.sync_bankroll(90.0);
        assert_eq!(
            m.check_trade_allowed(1.0, 20.0, 40.0),
            Err(KillSwitchTier::PerStage),
            "climbed to stage 1 then fell below stage 0's floor — retreat"
        );
    }

    /// The high-water cursor only advances; a falling balance must not reset it
    /// (that is precisely how the old cursor erased the retreat).
    #[test]
    fn the_high_water_stage_never_walks_back_down() {
        let mut cfg = RiskyModeConfig::default();
        cfg.stages = build_logarithmic_stages(100.0, 50_000.0, DEFAULT_DOUBLING_FACTOR);
        cfg.starting_capital_usd = 100.0;
        cfg.target_capital_usd = 50_000.0;
        let mut m = RiskyModeManager::new(cfg, 100.0).expect("manager");
        m.sync_bankroll(1_700.0);
        let peak = m.high_water_stage_idx();
        assert!(
            peak >= 4,
            "1700 should sit on stage 4 of the 100->50k ladder"
        );
        m.record_trade_outcome(-1_600.0);
        assert_eq!(m.high_water_stage_idx(), peak);
        assert!(m.current_stage().stage_idx < peak);
    }

    #[test]
    fn the_monthly_cap_defaults_and_remains_configurable() {
        let cfg = RiskyModeConfig::default();
        assert_eq!(
            cfg.monthly_loss_cap_fraction, DEFAULT_MONTHLY_LOSS_CAP_FRACTION,
            "default monthly loss cap changed"
        );
        let mut cfg = RiskyModeConfig::default();
        cfg.monthly_loss_cap_fraction = 0.50;
        let mut m = RiskyModeManager::new(cfg, 100.0).expect("manager");
        m.raise_period_losses(0.0, 0.0, 40.0);
        assert!(
            m.check_trade_allowed(1.0, 20.0, 40.0).is_ok(),
            "40 < 0.50 * 100 — under the monthly cap"
        );
        m.raise_period_losses(0.0, 0.0, 60.0);
        assert_eq!(
            m.check_trade_allowed(1.0, 20.0, 40.0),
            Err(KillSwitchTier::PerMonth)
        );
        // Rejected by the validator, not silently clamped.
        let mut bad = RiskyModeConfig::default();
        bad.monthly_loss_cap_fraction = 0.0;
        assert!(RiskyModeManager::new(bad, 100.0).is_err());
    }

    /// The account-wide ledger hook: another engine's loss must be able to
    /// close this engine's day, and it must never double-count or subtract.
    #[test]
    fn raise_period_losses_takes_the_max_and_never_lowers() {
        let mut m = RiskyModeManager::new(RiskyModeConfig::default(), 100.0).expect("manager");
        m.record_trade_outcome(-10.0); // this engine's own loss
        assert!((m.daily_loss_accumulated_usd() - 10.0).abs() < 1e-9);
        // The journal reports the same trade — must count ONCE, not twice.
        m.raise_period_losses(10.0, 10.0, 10.0);
        assert!((m.daily_loss_accumulated_usd() - 10.0).abs() < 1e-9);
        // A sibling engine lost 25 more on the same account.
        m.raise_period_losses(35.0, 35.0, 35.0);
        assert!((m.daily_loss_accumulated_usd() - 35.0).abs() < 1e-9);
        // An unreadable / stale journal must never lower a booked loss.
        m.raise_period_losses(0.0, 0.0, 0.0);
        m.raise_period_losses(f64::NAN, -5.0, f64::INFINITY);
        assert!((m.daily_loss_accumulated_usd() - 35.0).abs() < 1e-9);
        assert!((m.weekly_loss_accumulated_usd() - 35.0).abs() < 1e-9);
    }
}
