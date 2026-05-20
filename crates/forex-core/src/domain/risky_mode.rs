//! Risky Mode — autonomous compounding from a small starting balance
//! to a large target ($20 → $50,000 default), per the 2026-05-17
//! operator directive.
//!
//! **This is NOT an FTMO / prop-firm variant.** It is a completely
//! separate operating mode with explicit informed-consent semantics:
//!
//! - The operator starts with a small bankroll (default $20) and
//!   accepts that they are extremely likely (≥ 99 % probability per
//!   the §6.4 acceptance ceiling) to lose the starting balance while
//!   attempting to reach a much larger target.
//! - The bot — not the operator — owns every sizing and entry
//!   decision once Risky Mode is armed. Manual BUY/SELL orders are
//!   REJECTED at the gate; only [`crate::domain::risky_mode`]-aware
//!   AI signals can place trades. This is the
//!   `autonomous_only_contract` invariant enforced by
//!   [`RiskyModeConfig::autonomous_only_contract_accepted`].
//! - Risk per trade is in the operator-stated band 30 %–50 % of the
//!   *current* bankroll (default 40 %). This is two orders of
//!   magnitude larger than any prop-firm guidance — the operator
//!   has signed the §6.4 acknowledgement that this is expected to
//!   wipe out the starting capital in the typical case.
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
//! The auto-trade producer in
//! `crates/forex-app/src/app_services/trading/auto_trade.rs` is the
//! consumer of this manager.
//!
//! ## Composition with the rest of the risk stack
//!
//! Risky Mode does NOT replace [`crate::domain::risk::RiskManager`]
//! or [`crate::domain::prop_firm::PropFirmConstraints`]. When a
//! `RiskyModeManager` is bound on a `TradingSession`, it is the
//! STRICTER outer gate: `RiskyModeManager::check_trade_allowed`
//! runs BEFORE `risk_gate::prop_firm_pre_trade_check`, so a Risky
//! Mode rejection blocks the order before it ever reaches the prop-
//! firm rules. The prop-firm constraints stay in place for the
//! manual-trading (non-Risky-Mode) path.
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
// Every constant is `pub const` so the wizard, the auto-trade producer,
// and the UI can render the canonical value from a single source of
// truth.
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

/// Default risk-per-trade fraction (middle of the operator-stated
/// 30 %–50 % band). This is the fraction of the *current* bankroll
/// the bot is allowed to risk on a single trade — i.e. the SL
/// distance × lot value implied by this fraction is what gets sent
/// to the broker. Per the operator directive this is two orders of
/// magnitude larger than any prop-firm-style sizing and is expected
/// to wipe out the starting capital in the typical case.
pub const RISKY_MODE_DEFAULT_RISK_PER_TRADE_FRACTION: f64 = 0.40;

/// Lower bound on the per-trade risk fraction in Risky Mode. The
/// operator-stated band is 30 %–50 %; anything below 30 % degenerates
/// into "FTMO with a different name" and is rejected by the config
/// validator. Operators wanting a more conservative profile should
/// disable Risky Mode and rely on the standard
/// [`crate::domain::risk::RiskManager`] path.
pub const RISKY_MODE_MIN_RISK_PER_TRADE_FRACTION: f64 = 0.30;

/// Upper bound on the per-trade risk fraction in Risky Mode. Above
/// 50 % the single-loss-wipes-the-account regime gets so degenerate
/// that the per-day kill switch is the only thing preventing total
/// loss; the validator rejects anything beyond this ceiling.
pub const RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION: f64 = 0.50;

/// Pre-broker-send sanity ceiling as a fraction of bankroll. Even
/// with Risky Mode armed and a 50 % per-trade target, no single
/// order whose implied risk exceeds this fraction may leave the
/// process. Defence-in-depth against bugs in our own sizing.
pub const DEFAULT_PRESEND_SANITY_CEILING_FRACTION: f64 = 0.55;

/// Default minimum AI-ensemble confidence required for an entry.
/// The auto-trade producer must clear this before its signal reaches
/// the dispatch gate. Lower confidence → noisier scalps → expenses
/// (commission + spread + swap) eat the net edge; the floor keeps
/// the producer honest about what counts as actionable.
pub const DEFAULT_SWARM_CONFIDENCE_MIN: f64 = 0.65;

/// Default pairwise correlation cap. Concurrent positions in
/// directionally-correlated pairs effectively concentrate risk; the
/// gate refuses to open a second position when the abs correlation
/// with an existing position exceeds this fraction.
pub const DEFAULT_CORRELATION_CAP: f64 = 0.7;

/// Default volatility-sigma threshold for the per-stage pause. When
/// the rolling 30-day ATR exceeds this many sigmas above its mean,
/// the per-stage kill switch fires (research §4.6.2).
pub const DEFAULT_VOLATILITY_SIGMA_PAUSE: f64 = 3.0;

/// Operator-acknowledged tail-risk ceiling on the *initial-stage*
/// ruin probability (per `risky_mode_compounding_research.md` §6.4
/// and the 2026-05-17 operator directive §7.1). The operator
/// explicitly accepts that this fraction of attempts will lose the
/// starting balance — 99 % is the directive value, capturing the
/// "you will almost certainly lose your $20" honesty floor.
pub const MAX_ACCEPTABLE_INITIAL_RUIN_PROBABILITY: f64 = 0.99;

/// Default trades-per-day **assumption** for the days-to-target
/// estimator. This is a throughput *projection* — it has no gating /
/// rate-limiting effect on the live producer. A scalping cadence of
/// 10 trades/day is a deliberately middle-of-the-road default; the
/// operator can override per-session via
/// [`RiskyModeConfig::expected_trades_per_day`] to reflect their own
/// broker's typical fill latency and the producer's signal frequency.
/// The estimator multiplies trades-to-target by this value to surface
/// the wizard's "approximately N trading days at M trades/day" figure.
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
/// own sizing fraction and kill-switch caps. Unlike the
/// previous Kelly-tapered build, the per-trade risk fraction is a
/// direct knob (`risk_per_trade_fraction`) rather than a Kelly
/// multiplier — the operator-directive sizing is "30–50 % of the
/// current bankroll per trade", which is the variable Risky Mode
/// tunes; Kelly is irrelevant because the framing accepts ≥99 %
/// ruin probability up-front.
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
    /// `[RISKY_MODE_MIN_RISK_PER_TRADE_FRACTION,
    /// RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION]` (i.e. `[0.30, 0.50]`).
    /// Tapers from 0.50 at the first stage to 0.30 at the last
    /// stage in the default table — the small-bankroll early stages
    /// are the most aggressive (the operator needs the geometric
    /// kick) and the late stages soften so a wiped late-game stage
    /// gives the bot a chance to retreat rather than blow up at the
    /// finish line.
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

/// Operator-tunable Risky Mode configuration. Constructed by the
/// wizard's `AutonomyRisk` step (Step 9.5) and consumed by
/// [`RiskyModeManager::new`].
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
    /// Operator-acknowledged tail-risk ceiling. Default
    /// [`MAX_ACCEPTABLE_INITIAL_RUIN_PROBABILITY`] (0.99).
    pub acknowledged_ruin_probability_ceiling: f64,
    /// Pre-broker-send sanity check fraction. No single order's
    /// implied risk may exceed this fraction of the current
    /// bankroll, regardless of stage sizing.
    pub presend_sanity_ceiling_fraction: f64,
    /// **Autonomous-only contract acceptance.** When set, the
    /// manager rejects every manual order via
    /// [`Self::rejects_manual_orders`]; only AI signals from the
    /// auto-trade producer can place trades. The operator
    /// affirmatively ticks this in the wizard's `AutonomyRisk`
    /// step. False by default — a Risky Mode session whose
    /// `enable_risky_mode` call is made with this field unset is
    /// rejected by the validator.
    pub autonomous_only_contract_accepted: bool,
    /// Whether Risky Mode is allowed to drive a live broker. False
    /// by default — paper trading first per research §10.3.
    pub allow_live_broker: bool,
    /// Minimum AI ensemble confidence for an entry.
    pub require_swarm_confidence_min: f64,
    /// Require regime filter (research §4.6.4).
    pub require_regime_filter: bool,
    /// Require news blackout (research §4.6.3).
    pub require_news_blackout: bool,
    /// Pairwise correlation ceiling for concurrent positions.
    pub correlation_cap: f64,
    /// Volatility-sigma threshold for the per-stage pause.
    pub volatility_sigma_pause: f64,
    /// Operator-tuned **expected** scalping cadence — used only by
    /// [`RiskyModeManager::estimated_days_to_target`] to convert
    /// trades-to-target into a "trading days" estimate for the
    /// wizard's surface. Has no gating effect on the live producer:
    /// the dispatch gate accepts as many signals per day as the
    /// producer emits. Default
    /// [`DEFAULT_RISKY_TRADES_PER_DAY`] = 10.0.
    pub expected_trades_per_day: f64,
    /// Operator-tuned **expected** win-rate AFTER expenses (commission
    /// + spread + swap). Drives the Brownian-motion ruin-probability
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
    /// Returns the operator-directive default: `$20 → $50,000` in a
    /// logarithmic stage table, 40 % per-trade default, autonomous-
    /// only contract UNACCEPTED (the wizard step must explicitly
    /// flip it), paper trading only, all upstream filter gates on.
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
            acknowledged_ruin_probability_ceiling: MAX_ACCEPTABLE_INITIAL_RUIN_PROBABILITY,
            presend_sanity_ceiling_fraction: DEFAULT_PRESEND_SANITY_CEILING_FRACTION,
            autonomous_only_contract_accepted: false,
            allow_live_broker: false,
            require_swarm_confidence_min: DEFAULT_SWARM_CONFIDENCE_MIN,
            require_regime_filter: true,
            require_news_blackout: true,
            correlation_cap: DEFAULT_CORRELATION_CAP,
            volatility_sigma_pause: DEFAULT_VOLATILITY_SIGMA_PAUSE,
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

        // Stage table monotonicity + sizing bounds. Operator
        // directive 30–50 % per-trade band is enforced PER STAGE so
        // a misconfigured table cannot silently fall outside it.
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
            // 30 %–50 % per-trade band per operator directive §7.1.
            if !(RISKY_MODE_MIN_RISK_PER_TRADE_FRACTION..=RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION)
                .contains(&stage.risk_per_trade_fraction)
            {
                bail!(
                    "risk_per_trade_fraction must be in [{}, {}] per operator directive §7.1, stage {} = {}",
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

        if !self.acknowledged_ruin_probability_ceiling.is_finite()
            || self.acknowledged_ruin_probability_ceiling <= 0.0
            || self.acknowledged_ruin_probability_ceiling > 1.0
        {
            bail!(
                "acknowledged_ruin_probability_ceiling must be in (0, 1.0], got {}",
                self.acknowledged_ruin_probability_ceiling
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
        if !(0.0..=1.0).contains(&self.require_swarm_confidence_min) {
            bail!(
                "require_swarm_confidence_min must be in [0, 1.0], got {}",
                self.require_swarm_confidence_min
            );
        }
        if !(0.0..=1.0).contains(&self.correlation_cap) {
            bail!(
                "correlation_cap must be in [0, 1.0], got {}",
                self.correlation_cap
            );
        }
        if !self.volatility_sigma_pause.is_finite() || self.volatility_sigma_pause <= 0.0 {
            bail!(
                "volatility_sigma_pause must be positive and finite, got {}",
                self.volatility_sigma_pause
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
    /// Bankroll dropped below the previous stage's lower boundary
    /// (research §5.3 — auto-retreat).
    PerStage,
    /// Cumulative monthly drawdown exceeded the ack-ceiling
    /// (research §5.4).
    PerMonth,
    /// Operator hit HALT — manual kill switch from the UI
    /// (research §5.5).
    Manual,
    /// Hardware / connection-loss flatten (research §5.6).
    HardwareConnLoss,
    /// Pre-broker-send sanity check rejected the order because its
    /// implied risk exceeded `presend_sanity_ceiling_fraction` of
    /// the bankroll (research §5.7).
    PreSendSanity,
    /// **Operator attempted a manual BUY/SELL while Risky Mode
    /// armed the autonomous-only contract.** Manual orders are
    /// strictly forbidden in that mode — only AI signals from the
    /// auto-trade producer can place trades.
    ManualOrderWhileAutonomousOnly,
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
    current_bankroll_usd: f64,
    daily_loss_accumulated_usd: f64,
    weekly_loss_accumulated_usd: f64,
    monthly_loss_accumulated_usd: f64,
    consecutive_losses: u32,
    last_kill_switch_trip: Option<(KillSwitchTier, chrono::DateTime<chrono::Utc>)>,
}

impl RiskyModeManager {
    /// Build a new manager from a validated config and an initial
    /// bankroll. The bankroll determines the starting stage.
    pub fn new(config: RiskyModeConfig, initial_bankroll_usd: f64) -> Result<Self> {
        config.validate()?;
        if !config.autonomous_only_contract_accepted {
            bail!(
                "RiskyModeManager rejects construction without \
                 autonomous_only_contract_accepted=true — the operator must \
                 have explicitly signed the wizard's AutonomyRisk acknowledgement \
                 (§7.1 informed-consent gate) before Risky Mode can run."
            );
        }
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
            current_bankroll_usd: initial_bankroll_usd,
            daily_loss_accumulated_usd: 0.0,
            weekly_loss_accumulated_usd: 0.0,
            monthly_loss_accumulated_usd: 0.0,
            consecutive_losses: 0,
            last_kill_switch_trip: None,
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

    /// Last kill-switch trip, if any.
    pub fn last_kill_switch_trip(&self) -> Option<(KillSwitchTier, chrono::DateTime<chrono::Utc>)> {
        self.last_kill_switch_trip
    }

    /// Active stage descriptor.
    pub fn current_stage(&self) -> &RiskyStage {
        &self.config.stages[self.current_stage_idx as usize]
    }

    /// `true` iff Risky Mode is in autonomous-only mode and a
    /// manual BUY/SELL order from the operator must be rejected at
    /// the gate. The `&` borrow is intentional — callers in the
    /// trading session inspect this before forwarding an order to
    /// the broker fill path.
    ///
    /// Returns `false` when:
    /// - the config flag is not set (Risky Mode is "armed-by-default"
    ///   to autonomous, but the wizard contract must be signed
    ///   first), or
    /// - the operator never enabled Risky Mode in the first place
    ///   (in which case there is no manager and this method is not
    ///   reachable).
    pub fn rejects_manual_orders(&self) -> bool {
        self.config.autonomous_only_contract_accepted
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
        // Sticky manual / hardware halts — even after their time
        // window expires the operator must explicitly clear them.
        if let Some((tier, _ts)) = self.last_kill_switch_trip
            && (tier == KillSwitchTier::Manual || tier == KillSwitchTier::HardwareConnLoss)
        {
            return Err(tier);
        }

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

        // Per-stage retreat trigger (research §5.3).
        if self.current_stage_idx > 0 {
            let prev = &self.config.stages[(self.current_stage_idx as usize) - 1];
            if self.current_bankroll_usd < prev.bankroll_lower_usd {
                return Err(KillSwitchTier::PerStage);
            }
        }

        // Per-month cap (research §5.4).
        let monthly_cap_usd =
            self.config.acknowledged_ruin_probability_ceiling * self.current_bankroll_usd;
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

    /// Update bankroll after a closed trade. Advances or retreats
    /// the stage cursor as needed. Positive `pnl_usd` increases
    /// bankroll, negative decreases.
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

    /// Manual operator kill-switch.
    pub fn trip_manual_halt(&mut self) {
        self.last_kill_switch_trip = Some((KillSwitchTier::Manual, chrono::Utc::now()));
    }

    /// Hardware/connection-loss flatten signal.
    pub fn trip_hardware_kill(&mut self) {
        self.last_kill_switch_trip = Some((KillSwitchTier::HardwareConnLoss, chrono::Utc::now()));
    }

    /// Clear a sticky halt.
    pub fn clear_halt(&mut self) {
        self.last_kill_switch_trip = None;
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
}

// ---------------------------------------------------------------------------
// Stage-table construction.
// ---------------------------------------------------------------------------

/// Build a logarithmic stage table from `starting_capital_usd` to
/// (or past) `target_capital_usd`, doubling the bankroll at each
/// step by `doubling_factor`. The per-trade risk fraction tapers
/// linearly from [`RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION`] (0.50)
/// at the first stage to [`RISKY_MODE_MIN_RISK_PER_TRADE_FRACTION`]
/// (0.30) at the last. Returns at least one stage.
pub fn build_logarithmic_stages(
    starting_capital_usd: f64,
    target_capital_usd: f64,
    doubling_factor: f64,
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
            risk_per_trade_fraction: RISKY_MODE_DEFAULT_RISK_PER_TRADE_FRACTION,
            max_concurrent_positions: 1,
            max_pair_exposure_fraction: RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION,
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

        // Per-trade risk: 0.50 -> 0.30 linear across stages.
        let risk_per_trade = RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION
            - taper_t
                * (RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION - RISKY_MODE_MIN_RISK_PER_TRADE_FRACTION);

        // Daily loss cap: 0.80 -> 0.50 linear taper.
        let daily_cap = 0.80 - taper_t * (0.80 - 0.50);
        // Weekly DD cap: 0.95 -> 0.60 linear taper.
        let weekly_cap = 0.95 - taper_t * (0.95 - 0.60);

        stages.push(RiskyStage {
            stage_idx: i as u8,
            bankroll_lower_usd: lower,
            bankroll_upper_usd: upper,
            risk_per_trade_fraction: risk_per_trade,
            max_concurrent_positions: 1,
            // Single-position regime; pair-exposure equals the
            // per-trade risk fraction at this stage.
            max_pair_exposure_fraction: risk_per_trade,
            daily_loss_cap_fraction: daily_cap.max(0.50),
            weekly_drawdown_cap_fraction: weekly_cap.max(0.60),
        });
    }
    stages
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

    /// Build a default config with the autonomous contract explicitly
    /// accepted — the test harness equivalent of the operator ticking
    /// the wizard acknowledgement. New() rejects without this.
    fn signed_default_config() -> RiskyModeConfig {
        let mut cfg = RiskyModeConfig::default();
        cfg.autonomous_only_contract_accepted = true;
        cfg
    }

    #[test]
    fn config_default_has_operator_directive_constants() {
        let cfg = RiskyModeConfig::default();
        assert_eq!(cfg.starting_capital_usd, 20.0);
        assert_eq!(cfg.target_capital_usd, 50_000.0);
        assert_eq!(cfg.stage_doubling_factor, 2.0);
        assert_eq!(
            cfg.acknowledged_ruin_probability_ceiling,
            MAX_ACCEPTABLE_INITIAL_RUIN_PROBABILITY
        );
        // Default is paper trading only, autonomous contract NOT
        // accepted — the wizard must flip these before live use.
        assert!(!cfg.allow_live_broker);
        assert!(!cfg.autonomous_only_contract_accepted);
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
        // Risk fraction tapers DOWN across stages and stays in band.
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
                w[0].risk_per_trade_fraction >= w[1].risk_per_trade_fraction - 1e-9,
                "risk_per_trade must be non-increasing"
            );
        }
    }

    #[test]
    fn new_rejects_when_autonomous_contract_unsigned() {
        let cfg = RiskyModeConfig::default(); // autonomous_only_contract_accepted = false
        let err =
            RiskyModeManager::new(cfg, 20.0).expect_err("must reject without autonomous contract");
        assert!(
            err.to_string()
                .contains("autonomous_only_contract_accepted"),
            "wrong error: {err}"
        );
    }

    #[test]
    fn new_accepts_when_autonomous_contract_signed() {
        let cfg = signed_default_config();
        let mgr = RiskyModeManager::new(cfg, 20.0).expect("must accept signed config");
        assert!(mgr.rejects_manual_orders());
        assert_eq!(mgr.current_bankroll_usd(), 20.0);
        assert_eq!(mgr.current_stage().stage_idx, 0);
    }

    #[test]
    fn validate_rejects_risk_fraction_below_30_percent() {
        let mut cfg = signed_default_config();
        cfg.stages[0].risk_per_trade_fraction = 0.29;
        let err = cfg.validate().expect_err("must reject");
        assert!(
            err.to_string().contains("risk_per_trade_fraction"),
            "wrong error: {err}"
        );
    }

    #[test]
    fn validate_rejects_risk_fraction_above_50_percent() {
        let mut cfg = signed_default_config();
        cfg.stages[0].risk_per_trade_fraction = 0.51;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_non_monotonic_risk_fraction() {
        let mut cfg = signed_default_config();
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
        let cfg = signed_default_config();
        let mgr = RiskyModeManager::new(cfg, 20.0).expect("manager");
        // Tiny order well inside the per-stage cap.
        let ok = mgr.check_trade_allowed(1.0, 10.0, 30.0);
        assert!(ok.is_ok(), "unexpected reject: {:?}", ok);
    }

    #[test]
    fn check_trade_allowed_rejects_missing_sl() {
        let cfg = signed_default_config();
        let mgr = RiskyModeManager::new(cfg, 20.0).expect("manager");
        let res = mgr.check_trade_allowed(1.0, 0.0, 30.0);
        assert_eq!(res, Err(KillSwitchTier::PerTrade));
    }

    #[test]
    fn check_trade_allowed_rejects_when_size_breaches_presend_ceiling() {
        let cfg = signed_default_config();
        let mgr = RiskyModeManager::new(cfg, 100.0).expect("manager");
        // presend ceiling default is 0.55 -> 55 USD.
        let res = mgr.check_trade_allowed(60.0, 10.0, 30.0);
        assert_eq!(res, Err(KillSwitchTier::PreSendSanity));
    }

    #[test]
    fn check_trade_allowed_rejects_after_daily_cap_exceeded() {
        let mut mgr = RiskyModeManager::new(signed_default_config(), 100.0).expect("manager");
        // Stage 0 daily cap is generous (0.80 in the default table)
        // -> 80 USD. Push the accumulator past it.
        mgr.daily_loss_accumulated_usd = 81.0;
        let res = mgr.check_trade_allowed(1.0, 10.0, 30.0);
        assert_eq!(res, Err(KillSwitchTier::PerDay));
    }

    #[test]
    fn manual_halt_makes_all_subsequent_trades_reject_with_manual() {
        let mut mgr = RiskyModeManager::new(signed_default_config(), 100.0).expect("manager");
        mgr.trip_manual_halt();
        let res = mgr.check_trade_allowed(1.0, 10.0, 30.0);
        assert_eq!(res, Err(KillSwitchTier::Manual));
        mgr.clear_halt();
        assert!(mgr.check_trade_allowed(1.0, 10.0, 30.0).is_ok());
    }

    #[test]
    fn calculate_position_size_uses_risk_per_trade_fraction_times_bankroll() {
        let cfg = signed_default_config();
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
        let mgr = RiskyModeManager::new(signed_default_config(), 100.0).expect("manager");
        let full = mgr.calculate_position_size_usd(1.0);
        let half = mgr.calculate_position_size_usd(0.5);
        assert!(
            (full - 2.0 * half).abs() < 1e-9,
            "confidence scaling broken: full={full}, half={half}"
        );
    }

    #[test]
    fn calculate_position_size_returns_zero_for_non_positive_confidence() {
        let mgr = RiskyModeManager::new(signed_default_config(), 100.0).expect("manager");
        assert_eq!(mgr.calculate_position_size_usd(0.0), 0.0);
        assert_eq!(mgr.calculate_position_size_usd(-0.5), 0.0);
        assert_eq!(mgr.calculate_position_size_usd(f64::NAN), 0.0);
    }

    #[test]
    fn ruin_probability_is_extreme_at_default_sizing() {
        // With 30-50% per trade and the operator's honest
        // 0.52 / 1.5 (win-rate / reward-to-risk) defaults, the
        // Brownian-motion model returns P(ruin) ≈ 1.0 at stage 0 —
        // negative expected log-growth → guaranteed-ruin in the
        // model's idealisation. This is exactly the §7.1 framing
        // the operator signed for; if a future refactor inflates
        // the defaults back to overly-optimistic values this
        // assertion will catch it.
        let mgr = RiskyModeManager::new(signed_default_config(), 20.0).expect("manager");
        let p = mgr.current_ruin_probability_estimate();
        assert!(
            p > 0.95,
            "ruin probability should be ≥0.95 at default sizing, got {p}"
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
        let mgr_small = RiskyModeManager::new(signed_default_config(), 20.0).expect("small");
        let mgr_large = RiskyModeManager::new(signed_default_config(), 25_000.0).expect("large");
        let p_small = mgr_small.current_ruin_probability_estimate();
        let p_large = mgr_large.current_ruin_probability_estimate();
        assert!(
            p_large < p_small,
            "ruin prob did not decrease with bankroll: small={p_small} large={p_large}"
        );
    }

    #[test]
    fn ruin_probability_eases_with_stronger_operator_edge() {
        // If the operator empirically demonstrates a stronger edge
        // (raise expected_win_rate to 0.60 with reward_to_risk 2.0)
        // the model's ruin estimate at the same stage / bankroll
        // must drop. Pins the (p, r) sensitivity so a future refactor
        // that drops the config wiring gets caught.
        let mut cfg_thin = signed_default_config();
        cfg_thin.expected_win_rate = 0.52;
        cfg_thin.expected_reward_to_risk = 1.5;
        let mut cfg_fat = signed_default_config();
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
        let mut cfg = signed_default_config();
        cfg.expected_win_rate = 0.0;
        assert!(cfg.validate().is_err());
        cfg.expected_win_rate = 1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_non_positive_reward_to_risk() {
        let mut cfg = signed_default_config();
        cfg.expected_reward_to_risk = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn days_to_target_returns_some_zero_when_at_or_past_target() {
        let cfg = signed_default_config();
        let target = cfg.target_capital_usd;
        let mgr = RiskyModeManager::new(cfg, target).expect("manager");
        assert_eq!(mgr.estimated_days_to_target(), Some(0));
    }

    #[test]
    fn days_to_target_returns_none_at_default_negative_growth() {
        // With the honest §7.1 defaults (win-rate 0.52, RR 1.5) the
        // stage-0 per-trade log-growth is NEGATIVE: the model's
        // expected outcome is ruin, not target. The estimator must
        // refuse to invent a "days to target" number in that regime
        // (research §10.5 — don't surface optimistic projections
        // when the math says target is unreachable in expectation).
        let mgr = RiskyModeManager::new(signed_default_config(), DEFAULT_STARTING_CAPITAL_USD)
            .expect("manager");
        assert!(
            mgr.estimated_days_to_target().is_none(),
            "estimator must return None when expected log-growth is non-positive"
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
        let mut cfg = signed_default_config();
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
    fn days_to_target_scales_inversely_with_cadence() {
        // Use a positive-EV configuration so the estimator returns
        // finite values for both cadences. Then a 10× slower cadence
        // must yield a strictly larger days-to-target figure. Pins
        // the cadence semantics: it's a divisor on trades-to-target,
        // not a gating cap.
        let mut cfg_fast = signed_default_config();
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
        let mut cfg = signed_default_config();
        cfg.expected_trades_per_day = 0.0;
        let err = cfg.validate().expect_err("must reject zero cadence");
        assert!(
            err.to_string().contains("expected_trades_per_day"),
            "wrong error: {err}"
        );
    }

    #[test]
    fn record_trade_outcome_advances_stage_after_large_profit() {
        let mut mgr = RiskyModeManager::new(signed_default_config(), 20.0).expect("manager");
        assert_eq!(mgr.current_stage().stage_idx, 0);
        // Push bankroll into stage 1's range.
        let stage1_lower = mgr.config.stages[1].bankroll_lower_usd;
        let pnl = stage1_lower - mgr.current_bankroll_usd + 0.5;
        mgr.record_trade_outcome(pnl);
        assert!(mgr.current_stage().stage_idx >= 1);
    }

    #[test]
    fn record_trade_outcome_retreats_stage_after_large_loss() {
        let mut mgr = RiskyModeManager::new(signed_default_config(), 40.0).expect("manager");
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
        let mut mgr = RiskyModeManager::new(signed_default_config(), 100.0).expect("manager");
        mgr.record_trade_outcome(-5.0);
        mgr.record_trade_outcome(-7.0);
        mgr.record_trade_outcome(3.0); // ignored by daily accumulator
        assert!((mgr.daily_loss_accumulated_usd() - 12.0).abs() < 1e-9);
        assert!((mgr.weekly_loss_accumulated_usd() - 12.0).abs() < 1e-9);
        assert!((mgr.monthly_loss_accumulated_usd() - 12.0).abs() < 1e-9);
    }

    #[test]
    fn rejects_manual_orders_is_authoritative_for_autonomous_contract() {
        let mut cfg = signed_default_config();
        let mgr = RiskyModeManager::new(cfg.clone(), 20.0).expect("manager");
        assert!(mgr.rejects_manual_orders());

        // If the contract were unsigned the new() call rejects, so
        // we can't construct a manager without it. This is the
        // intended invariant.
        cfg.autonomous_only_contract_accepted = false;
        assert!(RiskyModeManager::new(cfg, 20.0).is_err());
    }

    #[test]
    fn build_logarithmic_stages_handles_degenerate_inputs() {
        let stages = build_logarithmic_stages(-1.0, 100.0, 2.0);
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].stage_idx, 0);
    }
}
