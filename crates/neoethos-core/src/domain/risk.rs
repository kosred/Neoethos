//! Prop-firm risk manager — the entry gate and size clamp for the
//! `system.trading_mode = "prop_firm"` live path.
//!
//! # WIRED 2026-08-10 (audit #137 / #192 / #294)
//!
//! Until today this file was 884 lines of correct-looking prop-firm risk logic
//! with **no production constructor**: every `RiskManager::new` call in the
//! workspace sat inside this file's own `#[cfg(test)]` module. It was the decoy
//! that let audit finding #204 survive — a reader looking for "where is the
//! risk gate" found it here and stopped looking.
//!
//! The operator's decision (`docs/audit-ledger.json` #137) was **WIRE IT**. So:
//!
//! * [`RiskManager::from_settings`] is the only constructor. It takes
//!   `&Settings` and the live account balance and resolves EVERY number from
//!   the operator's config or from the selected prop-firm preset. There is no
//!   `new(rules, flag, balance)` any more — a caller cannot hand this type a
//!   limit nobody chose.
//! * `crates/neoethos-app/src/app_services/live_trading.rs` builds one per
//!   engine when the trading mode is NOT `risky` (Risky Mode has its own
//!   [`crate::domain::risky_mode::RiskyModeManager`]; running both would apply
//!   two incompatible rulebooks to one order), calls
//!   [`RiskManager::check_trade_allowed`] before every entry and clamps the
//!   entry's risk fraction with [`RiskManager::calculate_position_size`].
//! * The three knobs the ledger listed as having no qualified reader —
//!   `risk.challenge_mode`, `risk.challenge_phase`,
//!   `risk.recovery_mode_enabled` — resolve here and nowhere else.
//!
//! # What was DELETED rather than wired, and why
//!
//! Replacing something means deleting the old. Each of these had a real
//! implementation and **no producer for its input**, so wiring it would have
//! meant inventing the number it judges:
//!
//! * **Trading-session window and night-session block** (`set_session_times`,
//!   `set_night_block`, `is_trading_session`). No `Settings` key and no preset
//!   table names a session window, and the night block needs a measured
//!   volatility the live loop does not compute. Weekend/session exclusion is
//!   already owned by `risk.kill_zones_enabled` in `live_trading.rs`.
//! * **News kill window** (`update_kill_window`). A setter with no producer.
//! * **Strategy-rank / strategy-Sharpe drawdown-recovery tiers.** Nothing in
//!   the live path ranks the running strategies against each other, so these
//!   arms could only ever be passed `None`. The drawdown bands they sat in
//!   still bite — through the sizing multipliers in
//!   [`RiskManager::calculate_position_size`], which need no ranking.
//! * **A second Kelly implementation** inside `calculate_position_size`.
//!   [`crate::domain::kelly`] is the Kelly in this workspace.
//! * **A second daily-trade cap.** `risk.max_trades_per_day` is enforced
//!   account-wide by [`crate::domain::daily_entry_cap`], behind
//!   `risk.max_trades_per_day_enabled`. A per-engine copy here would have
//!   silently re-armed a cap the operator deliberately left disarmed.
//! * **`PropFirmRules`, `ChallengeRiskPreset` and `resolve_challenge_risk_preset`.**
//!   `resolve_challenge_risk_preset` was a phase table that ignored the
//!   configured preset; [`PropFirmPhaseRiskDefaults::for_preset`] is the
//!   preset-aware one and is what this file now calls. `PropFirmRules` carried
//!   nine fields of which three were ever read.
//! * **The daily profit lock** (`daily_profit_lock_pct`). Nothing read it. It
//!   describes a REFUSAL — "stop trading once today's gain reaches X" — and
//!   adding a refusal nobody asked for is not wiring, it is inventing. If the
//!   operator wants it, it needs a `Settings` key and a decision, not a
//!   resurrected field.
//!
//! # Which way the money moves
//!
//! Every path through this file can only REFUSE an entry or make it SMALLER.
//! [`RiskManager::calculate_position_size`] returns a risk fraction that the
//! caller applies with `min`, so it is a clamp and never a lift.

use crate::config::Settings;
use crate::domain::prop_firm::{
    PropFirmConstraints, PropFirmPhaseRiskDefaults, PropFirmPreset, PropFirmRuntimeDefaults,
};

// ─────────────────────── revenge-trade detector policy ───────────────────────
//
// These are behavioural constants, not risk limits: they describe what
// "revenge trading" looks like, the way `RISKY_MODE_MAX_RISK_PER_TRADE_FRACTION`
// describes the risky ladder's ceiling. They are named and public so the
// numbers are visible rather than buried inline, and so a future `Settings` key
// has somewhere obvious to land.

/// How many recent closed trades the detector keeps.
pub const REVENGE_TRADES_TRACKED: usize = 10;
/// Re-entering sooner than this after a LOSS is revenge trading.
pub const REVENGE_MIN_MINUTES_AFTER_LOSS: f64 = 15.0;
/// A losing streak of this length trips the detector outside the tolerated
/// hours below.
pub const REVENGE_CONSECUTIVE_LOSS_STREAK: usize = 3;
/// How far back the streak is counted.
pub const REVENGE_STREAK_LOOKBACK: usize = 5;
/// UTC hour windows in which a losing streak alone does NOT trip the detector —
/// the London and New York opens, where a streak is more likely to be the
/// market than the trader.
pub const REVENGE_TOLERATED_HOURS_UTC: [(u32, u32); 2] = [(7, 9), (13, 15)];
/// Sizing up by more than this multiple of the recent mean, straight after a
/// loss, is revenge trading.
pub const REVENGE_SIZE_ESCALATION_FACTOR: f64 = 1.5;
/// Two same-direction losses closer together than this are revenge trading.
pub const REVENGE_SAME_DIRECTION_GAP_MINUTES: f64 = 30.0;

// ───────────────────────────── sizing policy ─────────────────────────────────

/// At or above this confidence the signal multiplier is 1.0.
pub const CONFIDENCE_FULL_SIZE: f64 = 0.80;
/// Below this confidence the multiplier is flat at [`LOW_CONFIDENCE_MULTIPLIER`].
pub const CONFIDENCE_SCALING_FLOOR: f64 = 0.60;
/// Multiplier applied below [`CONFIDENCE_SCALING_FLOOR`].
pub const LOW_CONFIDENCE_MULTIPLIER: f64 = 0.30;
/// Multiplier at exactly [`CONFIDENCE_SCALING_FLOOR`], rising linearly to 1.0
/// at [`CONFIDENCE_FULL_SIZE`].
pub const MID_CONFIDENCE_BASE: f64 = 0.50;

/// Fraction of the day's drawdown budget above which size is cut hardest.
pub const DD_BUDGET_TIGHT_FRACTION: f64 = 0.75;
/// Multiplier applied above [`DD_BUDGET_TIGHT_FRACTION`].
pub const DD_BUDGET_TIGHT_MULTIPLIER: f64 = 0.35;
/// Fraction of the day's drawdown budget above which size is cut.
pub const DD_BUDGET_HALF_FRACTION: f64 = 0.50;
/// Multiplier applied above [`DD_BUDGET_HALF_FRACTION`].
pub const DD_BUDGET_HALF_MULTIPLIER: f64 = 0.60;
/// Floor on the linear taper applied below the first recovery band, so a small
/// total drawdown never sizes the account to nothing.
pub const TOTAL_DD_TAPER_FLOOR: f64 = 0.30;
/// How far equity must climb back toward the day's start before recovery mode
/// releases, as a fraction of the day's starting equity.
pub const RECOVERY_EXIT_EQUITY_TOLERANCE: f64 = 0.005;

// ────────────────────────────── trade records ────────────────────────────────

/// One closed trade, as the revenge detector sees it.
///
/// `size` and `direction` come from the entry snapshot the live loop keeps.
/// When that snapshot is missing (engine restarted while a position was open)
/// the caller passes `size: 0.0` / `direction: None`; both checks that use them
/// are guarded so an absent value can only make the detector quieter, never
/// produce a false refusal.
#[derive(Debug, Clone, Copy)]
pub struct ClosedTrade {
    pub entry_time_sec: u64,
    pub exit_time_sec: u64,
    /// Realized net PnL in the account's deposit currency.
    pub pnl: f64,
    /// Position size actually sent, in lots.
    pub size: f64,
    /// `1` long, `-1` short, `None` when unknown.
    pub direction: Option<i32>,
}

/// Rolling window of closed trades plus the "is this revenge trading" test.
#[derive(Debug, Clone)]
pub struct RevengeTradeDetector {
    recent_trades: Vec<ClosedTrade>,
    max_trades_tracked: usize,
}

impl Default for RevengeTradeDetector {
    fn default() -> Self {
        Self {
            recent_trades: Vec::new(),
            max_trades_tracked: REVENGE_TRADES_TRACKED,
        }
    }
}

impl RevengeTradeDetector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Trades currently in the window (diagnostics / tests).
    pub fn tracked(&self) -> usize {
        self.recent_trades.len()
    }

    pub fn record_trade(&mut self, trade: ClosedTrade) {
        self.recent_trades.push(trade);
        if self.recent_trades.len() > self.max_trades_tracked {
            self.recent_trades.remove(0);
        }
    }

    pub fn is_revenge_trading(&self, current_time_sec: u64, current_hour: u32) -> bool {
        if self.recent_trades.len() < 2 {
            return false;
        }

        // **2026-05-25 unwrap audit**: the `len() < 2` guard makes the
        // `.last()` infallible. Pattern-match per the no-panic doctrine — a
        // future refactor breaking the invariant now returns false (no
        // revenge-trade flag) instead of panicking the gate.
        let Some(last_trade) = self.recent_trades.last() else {
            return false;
        };
        let time_since_last_min =
            (current_time_sec.saturating_sub(last_trade.exit_time_sec)) as f64 / 60.0;

        if time_since_last_min < REVENGE_MIN_MINUTES_AFTER_LOSS && last_trade.pnl < 0.0 {
            return true;
        }

        let mut consecutive_losses = 0usize;
        for trade in self.recent_trades.iter().rev().take(REVENGE_STREAK_LOOKBACK) {
            if trade.pnl < 0.0 {
                consecutive_losses += 1;
            } else {
                break;
            }
        }

        if consecutive_losses >= REVENGE_CONSECUTIVE_LOSS_STREAK {
            let tolerated = REVENGE_TOLERATED_HOURS_UTC
                .iter()
                .any(|(from, to)| (*from..*to).contains(&current_hour));
            if !tolerated {
                return true;
            }
        }

        if self.recent_trades.len() >= 3 {
            let recent_idx = self.recent_trades.len() - 3;
            let recent = &self.recent_trades[recent_idx..];
            let mut sum_prev_sizes = 0.0;
            let mut count_prev = 0;
            for t in &recent[..recent.len() - 1] {
                sum_prev_sizes += t.size;
                count_prev += 1;
            }
            if count_prev > 0 {
                // **2026-05-25 unwrap audit**: `recent` is a slice taken from
                // `recent_trades` after the `len() >= 3` guard, so
                // `recent.len() == 3` always. The pattern below is defensive.
                let Some(last_recent) = recent.last() else {
                    return false;
                };
                let mean_prev = sum_prev_sizes / (count_prev as f64);
                let last_size = last_recent.size;
                let prev_pnl = recent[recent.len() - 2].pnl;
                if mean_prev > 0.0
                    && last_size > REVENGE_SIZE_ESCALATION_FACTOR * mean_prev
                    && prev_pnl < 0.0
                {
                    return true;
                }
            }
        }

        if self.recent_trades.len() >= 3 {
            let n = self.recent_trades.len();
            let t1 = &self.recent_trades[n - 3];
            let t2 = &self.recent_trades[n - 2];
            let t3 = &self.recent_trades[n - 1];

            if t1.direction.is_some()
                && t1.direction == t2.direction
                && t2.direction == t3.direction
                && t3.pnl < 0.0
                && t2.pnl < 0.0
            {
                return true;
            }
        }

        {
            let n = self.recent_trades.len();
            let prev = &self.recent_trades[n - 2];
            let last = &self.recent_trades[n - 1];
            let gap_min = (last.entry_time_sec.saturating_sub(prev.exit_time_sec)) as f64 / 60.0;

            if gap_min < REVENGE_SAME_DIRECTION_GAP_MINUTES
                && last.pnl < 0.0
                && prev.pnl < 0.0
                && last.direction.is_some()
                && last.direction == prev.direction
            {
                return true;
            }
        }

        false
    }
}

// ─────────────────────────────── gate inputs ─────────────────────────────────

/// Everything [`RiskManager::check_trade_allowed`] judges, and nothing it has
/// to invent.
///
/// Every field is MEASURED at the call site. `confidence` is `Option` because
/// it only exists when `models.live_ml_gate` is on — with the gate off there is
/// no confidence number anywhere in the live loop, and passing `1.0` would be a
/// invented measurement that silently disables the phase's confidence floor.
#[derive(Debug, Clone, Copy)]
pub struct TradeGateInput {
    /// Live account balance in the deposit currency.
    pub equity: f64,
    /// Blended signal confidence, when the ML gate produced one.
    pub confidence: Option<f64>,
    /// Wall clock, for the revenge detector.
    pub current_time_sec: u64,
    /// UTC hour, for the revenge detector's tolerated windows.
    pub current_hour: u32,
    /// Entries already opened on the ACCOUNT this UTC day
    /// ([`crate::domain::daily_entry_cap`] is the counter). Account-wide, not
    /// per engine — several engines share one broker account.
    pub entries_today: usize,
}

/// Inputs to the size clamp. Same rule: measured or absent.
#[derive(Debug, Clone, Copy)]
pub struct PositionSizingInput {
    pub equity: f64,
    /// The risk fraction the caller intends to use, before this clamp.
    pub base_risk_pct: f64,
    /// Blended signal confidence, when the ML gate produced one.
    pub confidence: Option<f64>,
}

/// Why an entry was refused. `rule` is a stable id for the log and the status
/// line; `detail` carries the numbers, because a refusal the operator cannot
/// explain is a control he will switch off.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TradeGateRefusal {
    pub rule: &'static str,
    pub detail: String,
}

impl TradeGateRefusal {
    fn new(rule: &'static str, detail: String) -> Self {
        Self { rule, detail }
    }
}

// ───────────────────────────── the risk manager ──────────────────────────────

/// Prop-firm daily-loss / total-drawdown / drawdown-recovery / revenge-trade
/// gate, plus the position-size clamp.
///
/// Construct with [`RiskManager::from_settings`]. There is no other
/// constructor: every threshold below is either an operator key or a value of
/// the selected [`PropFirmPreset`], and a struct literal would be a way to
/// smuggle in a limit nobody chose.
#[derive(Debug, Clone)]
pub struct RiskManager {
    // ── resolved policy (never mutated after construction) ──
    /// `risk.preset` — which firm's table seeded the numbers below.
    pub preset: PropFirmPreset,
    /// `risk.challenge_phase` — selects the phase risk table.
    pub challenge_phase: String,
    /// `risk.challenge_mode`. `true` measures total drawdown from the FIXED
    /// challenge starting equity (how FTMO measures it) and gates on the
    /// challenge profit target; `false` measures it from the running equity
    /// PEAK — a trailing drawdown, which is the stricter of the two once the
    /// account is in profit — and gates on the monthly target instead.
    pub challenge_mode: bool,
    /// `risk.recovery_mode_enabled`. Arms the drawdown-recovery behaviour: the
    /// early HALT band, the recovery trade cap and the recovery size
    /// multipliers. It does NOT gate `risk.daily_drawdown_limit` or
    /// `risk.total_drawdown_limit` — those are the operator's written-down caps
    /// and are never optional.
    pub recovery_mode_enabled: bool,
    /// Preset runtime table: the recovery bands and their size multipliers.
    pub runtime: PropFirmRuntimeDefaults,
    /// `risk.total_drawdown_limit`.
    pub max_total_loss_pct: f64,
    /// `risk.daily_drawdown_limit`.
    pub daily_dd_stop_trading_pct: f64,
    /// Where recovery mode arms, derived by holding the PRESET's
    /// warning:stop ratio against the operator's stop — so lowering the stop
    /// lowers the warning with it instead of leaving the warning above it.
    pub daily_dd_warning_pct: f64,
    /// The per-trade risk ceiling that binds: **the operator's**
    /// (`risk.prop_firm_max_risk_per_trade`, falling back to
    /// `risk.max_risk_per_trade`).
    ///
    /// RULED 2026-08-10: "1% for prop firm, 30% for risky mode."
    ///
    /// It first shipped as `min(operator, phase_table)`, which read the phase
    /// table as a LOCK. That is wrong twice over. It contradicts the rule this
    /// codebase applies to every other preset-seeded limit — `reconcile_one`
    /// says in as many words that a preset is a SEED, NOT A LOCK, names both
    /// numbers at ERROR when the operator's is looser, and then runs HIS — and
    /// it silently overrode a number he had typed on purpose: his 0.010 became
    /// the FTMO phase-1 table's 0.005, half the size he chose, with nothing but
    /// a log line to say so.
    ///
    /// A firm's published rule is worth stating loudly. It is not worth
    /// enforcing against the account owner's explicit instruction, because then
    /// the config stops describing what the machine does — which is the defect
    /// this whole project has been unwinding.
    pub max_risk_per_trade: f64,
    /// The operator's own ceiling. Equal to [`Self::max_risk_per_trade`]; kept
    /// separate so the disagreement log can name both sides.
    pub operator_max_risk_per_trade: f64,
    /// What the phase table would have imposed. ADVISORY: reported, never
    /// enforced. When it is the tighter of the two, that fact is logged at
    /// ERROR with both numbers, because exceeding a firm's per-trade rule is
    /// how a challenge is failed — but it is the operator's call to make.
    pub phase_max_risk_per_trade: f64,
    /// Phase confidence floor. Only bites when the ML gate measures one.
    pub min_confidence_threshold: f64,
    /// `risk.monthly_profit_target_pct`.
    pub monthly_profit_target_pct: f64,
    /// Preset challenge profit target.
    pub challenge_target_return_pct: f64,

    // ── running state ──
    /// Anchor for total drawdown in challenge mode (0.0 when not in one).
    pub challenge_start_equity: f64,
    pub total_peak_equity: f64,
    pub day_start_equity: f64,
    pub day_peak_equity: f64,
    pub month_start_equity: f64,
    pub last_session_date_id: Option<u32>,
    pub last_month_id: Option<u32>,
    /// Latched by a TOTAL-drawdown breach. Deliberately never cleared: a blown
    /// account must not resume trading because equity ticked back up. Restart
    /// the engine after reviewing the account.
    pub circuit_breaker_triggered: bool,
    pub recovery_mode: bool,
    pub monthly_target_hit: bool,
    pub challenge_target_hit: bool,
    pub revenge_detector: RevengeTradeDetector,
}

impl RiskManager {
    /// The production constructor. Resolves every threshold from `settings`
    /// and the preset it names, anchored on the live account balance.
    ///
    /// Fails LOUD rather than substituting a default: a manager built from
    /// numbers that cannot bound anything is worse than no manager, because it
    /// looks like a gate.
    pub fn from_settings(settings: &Settings, live_equity: f64) -> Result<Self, String> {
        if !live_equity.is_finite() || live_equity <= 0.0 {
            return Err(format!(
                "RiskManager needs a live account balance to anchor the daily and total \
                 drawdown limits; got {live_equity}"
            ));
        }

        let preset = settings.risk.preset;
        let constraints = PropFirmConstraints::for_preset(preset);
        let runtime = PropFirmRuntimeDefaults::for_preset(preset);
        let challenge_phase = settings.risk.challenge_phase.clone();
        let phase = PropFirmPhaseRiskDefaults::for_preset(preset, &challenge_phase);

        let max_total_loss_pct = settings.risk.total_drawdown_limit;
        if !max_total_loss_pct.is_finite() || max_total_loss_pct <= 0.0 || max_total_loss_pct > 1.0 {
            return Err(format!(
                "risk.total_drawdown_limit must be a fraction in (0, 1]; got \
                 {max_total_loss_pct}. It is the hard stop on this account — there is no \
                 sane substitute for it"
            ));
        }
        let daily_dd_stop_trading_pct = settings.risk.daily_drawdown_limit;
        if !daily_dd_stop_trading_pct.is_finite()
            || daily_dd_stop_trading_pct <= 0.0
            || daily_dd_stop_trading_pct > 1.0
        {
            return Err(format!(
                "risk.daily_drawdown_limit must be a fraction in (0, 1]; got \
                 {daily_dd_stop_trading_pct}"
            ));
        }

        // Recovery arms at the preset's warning:stop RATIO applied to the
        // operator's stop. Taking the preset's warning literally would leave it
        // ABOVE the stop whenever the operator tightens the stop, i.e. recovery
        // mode would never arm before trading halted.
        let daily_dd_warning_pct = if runtime.daily_dd_stop_trading_pct > 0.0 {
            daily_dd_stop_trading_pct
                * (runtime.daily_dd_warning_pct / runtime.daily_dd_stop_trading_pct)
        } else {
            daily_dd_stop_trading_pct
        };

        // Per-trade ceiling. THE LOWER ONE BINDS — the same rule #209/#210
        // settled for the risky ladder: a limit the operator wrote down is
        // never silently raised, and a preset's phase table never lifts it
        // either.
        let operator_max_risk_per_trade = settings
            .risk
            .prop_firm_max_risk_per_trade
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(settings.risk.max_risk_per_trade);
        if !operator_max_risk_per_trade.is_finite()
            || operator_max_risk_per_trade <= 0.0
            || operator_max_risk_per_trade > 1.0
        {
            return Err(format!(
                "risk.max_risk_per_trade (or risk.prop_firm_max_risk_per_trade) must be a \
                 fraction in (0, 1]; got {operator_max_risk_per_trade}"
            ));
        }
        let phase_max_risk_per_trade = phase.max_risk_per_trade;
        // THE OPERATOR'S NUMBER BINDS. The phase table is advisory — see the
        // field docs for the 2026-08-10 ruling and why `min` was wrong. When
        // the firm's rule is the tighter one, say so at ERROR with both numbers
        // and run his anyway: that is exactly what `reconcile_one` does for
        // every other preset-seeded limit, and two rules for one idea is how
        // this codebase got into trouble in the first place.
        let max_risk_per_trade = operator_max_risk_per_trade;
        if phase_max_risk_per_trade.is_finite()
            && phase_max_risk_per_trade > 0.0
            && operator_max_risk_per_trade > phase_max_risk_per_trade
        {
            tracing::error!(
                target: "neoethos_core::risk",
                key = "risk.prop_firm_max_risk_per_trade",
                preset = preset.as_str(),
                phase = ?challenge_phase,
                your_value = operator_max_risk_per_trade,
                firm_rule = phase_max_risk_per_trade,
                "TWO READINGS OF ONE LIMIT: your per-trade ceiling is LOOSER than the selected \
                 preset's published rule for this phase. Your value is used — a preset is a \
                 seed, not a lock — but the firm's number is the one that fails a challenge."
            );
        }

        let monthly_profit_target_pct = settings.risk.monthly_profit_target_pct;
        let challenge_target_return_pct = constraints.challenge_profit_target_pct as f64;

        // Total drawdown is anchored on the account the operator declared, not
        // on whatever the balance happens to be when an engine starts — that is
        // the same anchor the inline breaker this replaces used.
        let declared_balance = settings.risk.initial_balance;
        let anchor = if declared_balance.is_finite() && declared_balance > 0.0 {
            declared_balance
        } else {
            live_equity
        };

        let challenge_mode = settings.risk.challenge_mode;

        Ok(Self {
            preset,
            challenge_phase,
            challenge_mode,
            recovery_mode_enabled: settings.risk.recovery_mode_enabled,
            runtime,
            max_total_loss_pct,
            daily_dd_stop_trading_pct,
            daily_dd_warning_pct,
            max_risk_per_trade,
            operator_max_risk_per_trade,
            phase_max_risk_per_trade,
            min_confidence_threshold: phase.min_confidence_threshold,
            monthly_profit_target_pct,
            challenge_target_return_pct,
            challenge_start_equity: if challenge_mode { anchor } else { 0.0 },
            total_peak_equity: live_equity.max(anchor),
            day_start_equity: live_equity,
            day_peak_equity: live_equity,
            month_start_equity: live_equity,
            last_session_date_id: None,
            last_month_id: None,
            circuit_breaker_triggered: false,
            recovery_mode: false,
            monthly_target_hit: false,
            challenge_target_hit: false,
            revenge_detector: RevengeTradeDetector::new(),
        })
    }

    /// Roll the day and month cursors. Call once per entry consideration with
    /// the SAME `yyyymmdd` key the account-wide entry cap uses, so the two
    /// rules can never disagree about which day a trade belongs to.
    ///
    /// A new day re-seeds the day's starting and peak equity, which is what
    /// makes the daily stop resume next UTC day. A new month re-seeds the
    /// monthly anchor and clears the monthly-target latch. The TOTAL-drawdown
    /// latch is not touched — nothing clears that but a restart.
    pub fn roll_periods(&mut self, date_id: u32, month_id: u32, equity: f64) {
        if !equity.is_finite() || equity <= 0.0 {
            return;
        }
        if self.last_session_date_id != Some(date_id) {
            self.day_start_equity = equity;
            self.day_peak_equity = equity;
            self.last_session_date_id = Some(date_id);
        }
        if self.last_month_id != Some(month_id) {
            self.month_start_equity = equity;
            self.monthly_target_hit = false;
            self.last_month_id = Some(month_id);
        }
    }

    /// `(daily_dd, intraday_dd, dd_used, dd_limit, total_dd)`, all fractions.
    ///
    /// `total_dd` is measured from the fixed challenge start in challenge mode
    /// and from the running equity peak otherwise.
    pub fn drawdown_state(&self, equity: f64) -> (f64, f64, f64, f64, f64) {
        let daily_dd_pct = if self.day_start_equity > 0.0 {
            (self.day_start_equity - equity) / self.day_start_equity
        } else {
            0.0
        };
        let intraday_dd_pct = if self.day_peak_equity > 0.0 {
            (self.day_peak_equity - equity) / self.day_peak_equity
        } else {
            0.0
        };
        let total_dd_pct = if self.challenge_start_equity > 0.0 {
            (self.challenge_start_equity - equity) / self.challenge_start_equity
        } else if self.total_peak_equity > 0.0 {
            (self.total_peak_equity - equity) / self.total_peak_equity
        } else {
            0.0
        };
        let dd_used = daily_dd_pct.max(intraday_dd_pct).max(0.0);
        let dd_limit = self.daily_dd_stop_trading_pct.max(1e-9);
        (
            daily_dd_pct,
            intraday_dd_pct,
            dd_used,
            dd_limit,
            total_dd_pct,
        )
    }

    /// Arm / release recovery mode from the day's drawdown.
    ///
    /// A no-op when `risk.recovery_mode_enabled` is `false` — that is the whole
    /// job of the key, and it is why the key was inert before today.
    pub fn update_recovery_state(&mut self, equity: f64) {
        if !self.recovery_mode_enabled {
            self.recovery_mode = false;
            return;
        }
        if self.day_start_equity <= 0.0 {
            return;
        }
        let daily_dd_pct = (self.day_start_equity - equity) / self.day_start_equity;

        if daily_dd_pct >= self.daily_dd_warning_pct {
            self.recovery_mode = true;
        } else if self.recovery_mode {
            let half_warning = self.daily_dd_warning_pct / 2.0;
            if equity >= (self.day_start_equity * (1.0 - RECOVERY_EXIT_EQUITY_TOLERANCE))
                || daily_dd_pct <= half_warning
            {
                self.recovery_mode = false;
            }
        }
    }

    /// The entry gate. `Ok(())` admits; `Err` names the rule and the numbers.
    pub fn check_trade_allowed(
        &mut self,
        input: TradeGateInput,
    ) -> Result<(), TradeGateRefusal> {
        // ── profit targets: stop when the goal is met ──
        if self.challenge_mode {
            if self.challenge_start_equity > 0.0 {
                let ret = (input.equity - self.challenge_start_equity) / self.challenge_start_equity;
                if ret >= self.challenge_target_return_pct {
                    self.challenge_target_hit = true;
                }
            }
            if self.challenge_target_hit {
                return Err(TradeGateRefusal::new(
                    "risk.challenge_target_reached",
                    format!(
                        "challenge profit target {:.2}% reached — no further entries this \
                         challenge",
                        self.challenge_target_return_pct * 100.0
                    ),
                ));
            }
        } else {
            // `monthly_profit_target_pct == 0.0` means NO monthly stop, not
            // "stop at zero profit". This is not a `max_`-style sentinel: a
            // profit TARGET of zero has one honest reading, and the other one
            // would halt every account on its first bar. The repo's root
            // `config.yaml` ships exactly 0.0 here.
            if self.month_start_equity > 0.0 && self.monthly_profit_target_pct > 0.0 {
                let ret = (input.equity - self.month_start_equity) / self.month_start_equity;
                if ret >= self.monthly_profit_target_pct {
                    self.monthly_target_hit = true;
                }
            }
            if self.monthly_target_hit {
                return Err(TradeGateRefusal::new(
                    "risk.monthly_profit_target_pct",
                    format!(
                        "monthly profit target {:.2}% reached — no further entries until the \
                         next UTC month",
                        self.monthly_profit_target_pct * 100.0
                    ),
                ));
            }
        }

        if self.circuit_breaker_triggered {
            return Err(TradeGateRefusal::new(
                "risk.total_drawdown_limit",
                "circuit breaker latched by an earlier total-drawdown breach — restart the \
                 engine after reviewing the account"
                    .to_string(),
            ));
        }

        if self
            .revenge_detector
            .is_revenge_trading(input.current_time_sec, input.current_hour)
        {
            return Err(TradeGateRefusal::new(
                "risk.revenge_trading",
                "revenge-trade pattern in the recent closed trades (re-entry too soon after a \
                 loss, a losing streak outside the tolerated hours, a size escalation after a \
                 loss, or repeated same-direction losses)"
                    .to_string(),
            ));
        }

        let (daily_dd, intraday_dd, _dd_used, _dd_limit, total_dd) =
            self.drawdown_state(input.equity);

        // The hard total cap first — it is the most severe condition, and it
        // latches.
        if total_dd >= self.max_total_loss_pct {
            self.circuit_breaker_triggered = true;
            return Err(TradeGateRefusal::new(
                "risk.total_drawdown_limit",
                format!(
                    "total drawdown {:.2}% reached the limit {:.2}% — ALL new entries halted \
                     (exit management continues). Restart the engine after reviewing the \
                     account",
                    total_dd * 100.0,
                    self.max_total_loss_pct * 100.0
                ),
            ));
        }

        // Drawdown-recovery behaviour, armed by risk.recovery_mode_enabled.
        if self.recovery_mode_enabled {
            if total_dd >= self.runtime.recovery_halt_drawdown_pct {
                self.circuit_breaker_triggered = true;
                return Err(TradeGateRefusal::new(
                    "risk.recovery_halt",
                    format!(
                        "drawdown recovery: HALT trading (drawdown {:.2}% > {:.2}%)",
                        total_dd * 100.0,
                        self.runtime.recovery_halt_drawdown_pct * 100.0
                    ),
                ));
            }
            if total_dd >= self.runtime.recovery_top_strategy_drawdown_pct
                && input.entries_today >= self.runtime.recovery_max_trades_per_day
            {
                return Err(TradeGateRefusal::new(
                    "risk.recovery_max_trades_per_day",
                    format!(
                        "drawdown recovery: drawdown {:.2}% > {:.2}%, at most {} entries per \
                         UTC day account-wide (already {})",
                        total_dd * 100.0,
                        self.runtime.recovery_top_strategy_drawdown_pct * 100.0,
                        self.runtime.recovery_max_trades_per_day,
                        input.entries_today
                    ),
                ));
            }
        }

        if daily_dd >= self.daily_dd_stop_trading_pct {
            return Err(TradeGateRefusal::new(
                "risk.daily_drawdown_limit",
                format!(
                    "daily loss {:.2}% reached the limit {:.2}% — new entries blocked until \
                     the next UTC day (exits continue)",
                    daily_dd * 100.0,
                    self.daily_dd_stop_trading_pct * 100.0
                ),
            ));
        }
        if intraday_dd >= self.daily_dd_stop_trading_pct {
            return Err(TradeGateRefusal::new(
                "risk.daily_drawdown_limit.intraday",
                format!(
                    "intraday give-back {:.2}% from the day's peak reached the limit {:.2}% — \
                     new entries blocked until the next UTC day (exits continue)",
                    intraday_dd * 100.0,
                    self.daily_dd_stop_trading_pct * 100.0
                ),
            ));
        }

        // Phase confidence floor. Only evaluable when something MEASURED a
        // confidence — with `models.live_ml_gate` off nothing does, and
        // inventing 1.0 here would silently retire the floor.
        if let Some(confidence) = input.confidence
            && confidence < self.min_confidence_threshold
        {
            return Err(TradeGateRefusal::new(
                "risk.challenge_phase.min_confidence_threshold",
                format!(
                    "signal confidence {:.2} below the {} phase floor {:.2}",
                    confidence, self.challenge_phase, self.min_confidence_threshold
                ),
            ));
        }

        Ok(())
    }

    /// Fold a closed trade into the drawdown and revenge state.
    ///
    /// `equity` is the account balance AFTER the close.
    pub fn record_closed_trade(&mut self, trade: ClosedTrade, equity: f64) {
        self.revenge_detector.record_trade(trade);
        self.on_equity_update(equity);
    }

    /// Track peaks and re-evaluate recovery mode from a fresh balance.
    pub fn on_equity_update(&mut self, equity: f64) {
        if !equity.is_finite() || equity <= 0.0 {
            return;
        }
        if equity > self.total_peak_equity {
            self.total_peak_equity = equity;
        }
        if equity > self.day_peak_equity {
            self.day_peak_equity = equity;
        }
        self.update_recovery_state(equity);
    }

    /// The size clamp: the risk fraction this account may put on THIS entry.
    ///
    /// The caller applies it with `min`, so this can only shrink an entry. It
    /// returns exactly `0.0` once the total-loss limit (or the recovery halt,
    /// when armed) is reached — a hard "no size", not a small one.
    pub fn calculate_position_size(&mut self, input: PositionSizingInput) -> f64 {
        // Confidence scaling, only when a confidence was measured.
        let signal_multiplier = match input.confidence {
            Some(c) if c >= CONFIDENCE_FULL_SIZE => 1.0,
            Some(c) if c >= CONFIDENCE_SCALING_FLOOR => {
                let span = CONFIDENCE_FULL_SIZE - CONFIDENCE_SCALING_FLOOR;
                let slope = if span > 0.0 {
                    (1.0 - MID_CONFIDENCE_BASE) / span
                } else {
                    0.0
                };
                MID_CONFIDENCE_BASE + (c - CONFIDENCE_SCALING_FLOOR) * slope
            }
            Some(_) => LOW_CONFIDENCE_MULTIPLIER,
            None => 1.0,
        };

        let mut risk_pct = input.base_risk_pct * signal_multiplier;

        // Recovery mode halves the ceiling before anything else is applied.
        let mut current_cap = self.max_risk_per_trade;
        if self.recovery_mode_enabled && self.recovery_mode {
            current_cap = self.max_risk_per_trade * self.runtime.recovery_mode_risk_multiplier;
        }
        risk_pct = risk_pct.min(current_cap);

        let (_, _, dd_used, dd_limit, total_dd_pct) = self.drawdown_state(input.equity);
        let dd_frac = dd_used / dd_limit.max(1e-9);

        if dd_frac >= DD_BUDGET_TIGHT_FRACTION {
            risk_pct *= DD_BUDGET_TIGHT_MULTIPLIER;
        } else if dd_frac >= DD_BUDGET_HALF_FRACTION {
            risk_pct *= DD_BUDGET_HALF_MULTIPLIER;
        }

        let max_total_loss = self.max_total_loss_pct.max(1e-6);
        if total_dd_pct >= max_total_loss {
            return 0.0;
        }
        if self.recovery_mode_enabled {
            if total_dd_pct >= self.runtime.recovery_halt_drawdown_pct {
                return 0.0;
            } else if total_dd_pct >= self.runtime.recovery_top_strategy_drawdown_pct {
                risk_pct *= self.runtime.recovery_mode_risk_multiplier;
            } else if total_dd_pct >= self.runtime.recovery_min_sharpe_drawdown_pct {
                risk_pct *= self.runtime.defensive_mode_risk_multiplier;
            } else if total_dd_pct >= self.runtime.recovery_top_three_drawdown_pct {
                risk_pct *= self.runtime.caution_mode_risk_multiplier;
            } else if total_dd_pct > 0.0 {
                let scale = 1.0 - (total_dd_pct / max_total_loss);
                risk_pct *= scale.max(TOTAL_DD_TAPER_FLOOR);
            }
        } else if total_dd_pct > 0.0 {
            let scale = 1.0 - (total_dd_pct / max_total_loss);
            risk_pct *= scale.max(TOTAL_DD_TAPER_FLOOR);
        }

        risk_pct.clamp(0.0, self.max_risk_per_trade)
    }
}

#[cfg(test)]
mod tests {
    use super::{ClosedTrade, PositionSizingInput, RiskManager, TradeGateInput};
    use crate::config::Settings;
    use crate::domain::prop_firm::{PropFirmPhaseRiskDefaults, PropFirmPreset};

    const DAY: u32 = 2026_08_10;
    const MONTH: u32 = 2026_08;
    const NOW: u64 = 1_700_000_000;

    /// A `Settings` with the risk section pinned to values the tests reason
    /// about. Built through `Settings::default()` so the seal stays the single
    /// construction point.
    fn settings(mutate: impl FnOnce(&mut Settings)) -> Settings {
        let mut s = Settings::default();
        s.risk.preset = PropFirmPreset::Ftmo;
        s.risk.initial_balance = 10_000.0;
        s.risk.total_drawdown_limit = 0.10;
        s.risk.daily_drawdown_limit = 0.04;
        s.risk.max_risk_per_trade = 0.030;
        s.risk.prop_firm_max_risk_per_trade = None;
        s.risk.challenge_mode = true;
        s.risk.challenge_phase = "phase_1".to_string();
        s.risk.recovery_mode_enabled = true;
        s.risk.monthly_profit_target_pct = 0.04;
        mutate(&mut s);
        s
    }

    fn manager(mutate: impl FnOnce(&mut Settings)) -> RiskManager {
        RiskManager::from_settings(&settings(mutate), 10_000.0)
            .expect("the pinned test settings must build a manager")
    }

    fn gate(equity: f64) -> TradeGateInput {
        TradeGateInput {
            equity,
            confidence: None,
            current_time_sec: NOW,
            current_hour: 10,
            entries_today: 0,
        }
    }

    /// #137: the production constructor exists and the manager it builds is
    /// made of the operator's numbers, not of literals.
    #[test]
    fn from_settings_takes_its_limits_from_the_operator_config() {
        let m = manager(|_| {});
        assert_eq!(m.max_total_loss_pct, 0.10);
        assert_eq!(m.daily_dd_stop_trading_pct, 0.04);
        assert_eq!(m.monthly_profit_target_pct, 0.04);
        // FTMO's warning:stop ratio (0.035 : 0.040) held against the
        // operator's 0.04 stop.
        assert!((m.daily_dd_warning_pct - 0.035).abs() < 1e-12);
    }

    /// #137 / #294: `risk.challenge_phase` is no longer inert — it selects the
    /// phase risk table, which supplies the confidence floor and the ADVISORY
    /// ceiling reported alongside the operator's own. Since the 2026-08-10
    /// ruling it no longer decides the BINDING ceiling; that is his number.
    #[test]
    fn challenge_phase_selects_the_phase_risk_table() {
        let p1 = manager(|s| s.risk.challenge_phase = "phase_1".to_string());
        let p2 = manager(|s| s.risk.challenge_phase = "phase_2".to_string());

        assert_eq!(
            p1.phase_max_risk_per_trade,
            PropFirmPhaseRiskDefaults::FTMO_PHASE_1.max_risk_per_trade
        );
        assert_eq!(
            p2.phase_max_risk_per_trade,
            PropFirmPhaseRiskDefaults::FTMO_PHASE_2.max_risk_per_trade
        );
        assert_ne!(p1.phase_max_risk_per_trade, p2.phase_max_risk_per_trade);
        assert_ne!(p1.min_confidence_threshold, p2.min_confidence_threshold);
        // And the binding ceiling does NOT move with the phase: it is the
        // operator's, on both. This assertion is the ruling — if it ever fails,
        // a phase table has started overriding him again.
        assert_eq!(p1.max_risk_per_trade, p2.max_risk_per_trade);
        assert_eq!(p1.max_risk_per_trade, p1.operator_max_risk_per_trade);
    }

    /// THE OPERATOR'S NUMBER BINDS. A phase table advises and is logged; it
    /// never silently replaces a ceiling he typed.
    #[test]
    fn the_operators_per_trade_ceiling_binds_and_the_phase_table_only_advises() {
        // RULED 2026-08-10: "1% for prop firm, 30% for risky mode."
        //
        // This first shipped as `min(operator, phase)`, which silently halved a
        // number the operator had typed on purpose — his 0.010 became FTMO
        // phase-1's 0.005. A preset is a SEED, NOT A LOCK, exactly as
        // `reconcile_one` says for every other preset-seeded limit.
        let looser = manager(|s| s.risk.prop_firm_max_risk_per_trade = Some(0.01));
        assert_eq!(
            looser.max_risk_per_trade, 0.01,
            "the operator wrote 1%; the phase table's 0.5% must NOT override it"
        );
        assert!(
            looser.max_risk_per_trade > looser.phase_max_risk_per_trade,
            "this case only tests anything while the operator's value is the looser one"
        );

        // Tighter than the firm's rule: still his, unchanged. The rule is
        // "his number binds", not "the larger number binds".
        let tighter = manager(|s| s.risk.prop_firm_max_risk_per_trade = Some(0.001));
        assert_eq!(tighter.max_risk_per_trade, 0.001);

        // And the firm's number is still CARRIED, so the disagreement can be
        // reported rather than forgotten.
        assert_eq!(
            looser.phase_max_risk_per_trade,
            PropFirmPhaseRiskDefaults::FTMO_PHASE_1.max_risk_per_trade
        );
    }

    /// The other half of the same ruling: risky mode sizes from the risky
    /// ladder, and no prop-firm phase table may reach it.
    #[test]
    fn risky_mode_is_not_bounded_by_a_prop_firm_phase_table() {
        let s = {
            let mut s = Settings::default();
            s.system.trading_mode = "risky".to_string();
            s.risk.risky_max_risk_per_trade = Some(0.30);
            s
        };
        assert_eq!(
            s.risk.risky_max_risk_per_trade,
            Some(0.30),
            "30% is the risky ladder's ceiling and is the operator's own number"
        );
        // The prop-firm manager is not the one that runs in risky mode at all —
        // `live_trading` selects `RiskyModeManager` on this trading_mode, so a
        // phase ceiling of 0.005 can never bound a 0.30 entry.
        assert!(
            PropFirmPhaseRiskDefaults::FTMO_PHASE_1.max_risk_per_trade < 0.30,
            "if this ever stopped being true the two ladders would have merged"
        );
    }

    /// #294: `risk.challenge_mode` decides WHICH anchor total drawdown is
    /// measured from, and therefore what gets refused.
    #[test]
    fn challenge_mode_anchors_total_drawdown_on_the_challenge_start() {
        let m = manager(|s| s.risk.challenge_mode = true);
        let (_, _, _, _, total_dd) = m.drawdown_state(9_250.0);
        assert!((total_dd - 0.075).abs() < 1e-9);

        // Not in a challenge: the anchor is the running PEAK, so an account
        // that grew to 12k and fell back to 10k is 16.7% down, not 0%.
        let mut off = manager(|s| s.risk.challenge_mode = false);
        assert_eq!(off.challenge_start_equity, 0.0);
        off.on_equity_update(12_000.0);
        let (_, _, _, _, total_dd_peak) = off.drawdown_state(10_000.0);
        assert!((total_dd_peak - (2_000.0 / 12_000.0)).abs() < 1e-9);
    }

    /// The hard total cap refuses and LATCHES.
    #[test]
    fn total_drawdown_limit_refuses_and_latches_the_circuit_breaker() {
        let mut m = manager(|_| {});
        let refusal = m
            .check_trade_allowed(gate(9_000.0))
            .expect_err("10% total drawdown must refuse");
        assert_eq!(refusal.rule, "risk.total_drawdown_limit");
        assert!(m.circuit_breaker_triggered);

        // Recovering the balance does NOT re-open the engine.
        let again = m
            .check_trade_allowed(gate(10_000.0))
            .expect_err("the latch must survive a recovery in equity");
        assert_eq!(again.rule, "risk.total_drawdown_limit");
    }

    /// The daily stop is not a latch: it resumes on the next UTC day.
    #[test]
    fn daily_drawdown_limit_resumes_next_utc_day() {
        let mut m = manager(|_| {});
        m.roll_periods(DAY, MONTH, 10_000.0);
        let refusal = m
            .check_trade_allowed(gate(9_600.0))
            .expect_err("a 4% daily loss must refuse");
        assert_eq!(refusal.rule, "risk.daily_drawdown_limit");
        assert!(!m.circuit_breaker_triggered);

        m.roll_periods(DAY + 1, MONTH, 9_600.0);
        assert!(m.check_trade_allowed(gate(9_600.0)).is_ok());
    }

    /// #294: `risk.recovery_mode_enabled` is the toggle it claims to be. With
    /// it OFF the recovery HALT band does not fire; the operator's own
    /// `total_drawdown_limit` still does.
    #[test]
    fn recovery_mode_enabled_arms_the_recovery_halt_band() {
        // 6% TOTAL drawdown from the 10k challenge anchor, but only 1.05% of
        // the day — so the daily stop is not what decides this, the recovery
        // band is.
        let mut on = manager(|s| s.risk.recovery_mode_enabled = true);
        on.roll_periods(DAY, MONTH, 9_500.0);
        let refusal = on
            .check_trade_allowed(gate(9_400.0))
            .expect_err("recovery halt must fire when the toggle is on");
        assert_eq!(refusal.rule, "risk.recovery_halt");

        let mut off = manager(|s| s.risk.recovery_mode_enabled = false);
        off.roll_periods(DAY, MONTH, 9_500.0);
        assert!(off.check_trade_allowed(gate(9_400.0)).is_ok());
    }

    /// The size clamp returns a hard zero once the total-loss limit is hit,
    /// and never exceeds the resolved per-trade ceiling.
    #[test]
    fn position_size_is_zero_at_the_limit_and_capped_below_it() {
        let mut m = manager(|_| {});
        assert_eq!(
            m.calculate_position_size(PositionSizingInput {
                equity: 9_000.0,
                base_risk_pct: 0.01,
                confidence: None,
            }),
            0.0
        );

        let mut healthy = manager(|_| {});
        let sized = healthy.calculate_position_size(PositionSizingInput {
            equity: 10_000.0,
            base_risk_pct: 0.030,
            confidence: None,
        });
        assert!(sized <= healthy.max_risk_per_trade + 1e-12);
        assert!(sized > 0.0);
    }

    /// #192: the clamp is a clamp. Whatever the caller intended, the returned
    /// fraction never exceeds the resolved ceiling, so `min`-ing it can only
    /// make an order smaller.
    #[test]
    fn position_size_never_exceeds_the_resolved_ceiling() {
        let mut m = manager(|_| {});
        for base in [0.0, 0.001, 0.01, 0.30, 1.0] {
            let sized = m.calculate_position_size(PositionSizingInput {
                equity: 10_000.0,
                base_risk_pct: base,
                confidence: Some(1.0),
            });
            assert!(
                sized <= m.max_risk_per_trade + 1e-12,
                "base {base} produced {sized}, above the ceiling {}",
                m.max_risk_per_trade
            );
        }
    }

    /// An absent confidence must not silently retire the phase floor by
    /// scoring as a perfect signal — and must not refuse either.
    #[test]
    fn absent_confidence_neither_refuses_nor_scales() {
        let mut m = manager(|_| {});
        assert!(m.check_trade_allowed(gate(10_000.0)).is_ok());

        let low = m
            .check_trade_allowed(TradeGateInput {
                confidence: Some(0.10),
                ..gate(10_000.0)
            })
            .expect_err("a measured confidence below the phase floor must refuse");
        assert_eq!(low.rule, "risk.challenge_phase.min_confidence_threshold");

        let unmeasured = m.calculate_position_size(PositionSizingInput {
            equity: 10_000.0,
            base_risk_pct: 0.001,
            confidence: None,
        });
        let perfect = m.calculate_position_size(PositionSizingInput {
            equity: 10_000.0,
            base_risk_pct: 0.001,
            confidence: Some(0.95),
        });
        assert_eq!(unmeasured, perfect);
    }

    /// The revenge detector reaches the gate.
    #[test]
    fn revenge_pattern_refuses_the_next_entry() {
        let mut m = manager(|_| {});
        m.record_closed_trade(
            ClosedTrade {
                entry_time_sec: NOW - 3_600,
                exit_time_sec: NOW - 1_800,
                pnl: -50.0,
                size: 0.01,
                direction: Some(1),
            },
            9_950.0,
        );
        m.record_closed_trade(
            ClosedTrade {
                entry_time_sec: NOW - 600,
                exit_time_sec: NOW - 60,
                pnl: -50.0,
                size: 0.01,
                direction: Some(1),
            },
            9_900.0,
        );
        let refusal = m
            .check_trade_allowed(gate(9_900.0))
            .expect_err("re-entering one minute after a loss must refuse");
        assert_eq!(refusal.rule, "risk.revenge_trading");
    }

    /// The monthly target gate is what `challenge_mode: false` selects, and it
    /// clears on the next month.
    #[test]
    fn monthly_target_stops_trading_until_the_month_rolls() {
        let mut m = manager(|s| s.risk.challenge_mode = false);
        m.roll_periods(DAY, MONTH, 10_000.0);
        let refusal = m
            .check_trade_allowed(gate(10_400.0))
            .expect_err("a 4% monthly gain must stop new entries");
        assert_eq!(refusal.rule, "risk.monthly_profit_target_pct");

        m.roll_periods(DAY + 30, MONTH + 1, 10_400.0);
        assert!(m.check_trade_allowed(gate(10_400.0)).is_ok());
    }

    /// A manager that cannot bound anything must not be built.
    #[test]
    fn from_settings_fails_loud_on_limits_that_cannot_bound() {
        assert!(RiskManager::from_settings(&settings(|_| {}), 0.0).is_err());
        assert!(
            RiskManager::from_settings(&settings(|s| s.risk.total_drawdown_limit = 0.0), 10_000.0)
                .is_err()
        );
        assert!(
            RiskManager::from_settings(&settings(|s| s.risk.daily_drawdown_limit = -1.0), 10_000.0)
                .is_err()
        );
        assert!(
            RiskManager::from_settings(
                &settings(|s| {
                    s.risk.max_risk_per_trade = 0.0;
                    s.risk.prop_firm_max_risk_per_trade = None;
                }),
                10_000.0
            )
            .is_err()
        );
    }
}
