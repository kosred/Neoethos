//! Cross-platform first-run wizard. Spec:
//! `docs/audits/research/installer_wizard_ux_spec.md` (1010 LOC).
//! Competitive analysis & Step 9.5 additions:
//! `docs/audits/research/wizard_onboarding_competitive_analysis.md`
//! (1701 LOC).
//!
//! Architecture (spec §1.4):
//! - The wizard lives entirely inside `forex-app::ui::wizard`. No new
//!   crate is introduced.
//! - The state machine (`WizardController`) is renderer-agnostic. The
//!   egui surface here and the ratatui surface in
//!   `forex-cli/src/tui/wizard.rs` both drive the same controller.
//! - Persistence (the `wizard_state.json` schema in spec §5) is
//!   declared in `state.rs` (`WizardStateFile::{read_from, write_to,
//!   default_path}`) and driven by `summary.rs::run_apply` at
//!   Step 10. The IO uses
//!   `forex_core::storage::json::write_json_atomic` for crash
//!   safety (audit-cleaned at F-CORE2-018).
//!
//! Operator invariants enforced here (NOT in step files):
//! - 11 canonical timeframes from `forex_core::contracts::temporal`
//!   (NO H2). See `symbols.rs`.
//! - 4 % monthly profit *floor* from
//!   `forex_core::domain::prop_firm::PropFirmConstraints::FTMO_STANDARD`.
//!   See `account_profile.rs`.

use eframe::egui;
use std::path::PathBuf;

pub mod account_profile;
pub mod autonomy_risk;
pub mod autostart;
pub mod hardware;
pub mod historical;
pub mod migration;
pub mod news_api;
pub mod oauth;
pub mod path;
pub mod state;
pub mod summary;
pub mod symbols;
pub mod welcome;

#[allow(unused_imports)]
pub use state::{
    InstallMetadata, RiskAcknowledgement, WIZARD_STATE_FILENAME, WIZARD_STATE_FILE_VERSION,
    WizardError, WizardState, WizardStateFile, WizardStepStatus,
};

/// Step-render result. Every step file returns one of these to the
/// controller so the controller (not the step) owns the actual state
/// transition. Spec §2 "Next / Back / Cancel" contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepResult {
    /// User clicked Continue / Next / Enter.
    NextRequested,
    /// User clicked Back / Esc-Back.
    BackRequested,
    /// User clicked Skip — only valid when `is_skippable()` is true.
    SkipRequested,
    /// User clicked Cancel / closed the modal.
    CancelRequested,
    /// User is still on this step (e.g. typing in a text field).
    /// Default for "no-op this frame".
    StayHere,
}

/// User-visible wizard configuration. Populated step-by-step; written
/// to `<data_path>/config.yaml` at Step 10. Operator invariants live
/// here so that `WizardController::current` is the single source of
/// truth for the renderers and the Apply writer.
///
/// Defaults follow operator policy (4 % monthly floor, no H2 in
/// canonical timeframes) and are `pub const` at the relevant step
/// file so they're greppable from a code review.
#[derive(Debug, Clone)]
pub struct WizardConfig {
    pub license_accepted: bool,
    pub data_path: Option<PathBuf>,
    pub install_metadata: InstallMetadata,

    // Step 3.
    pub operator_name: Option<String>,
    pub prop_firm_preset: String,
    pub monthly_profit_target: f32,
    pub trading_mode: TradingMode,
    pub interface_mode: InterfaceMode,
    pub risk_profile_slider: u8,        // 1..=10
    pub require_stop_loss: bool,
    pub per_trade_max_risk_pct: f32,
    pub daily_loss_reset_timezone: String,

    // Step 4.
    pub ctrader_client_id: Option<String>,
    pub ctrader_client_secret_set: bool,
    pub ctrader_environment: CTraderEnvironment,
    pub selected_ctid_trader_account_id: Option<u64>,
    pub additional_account_ids: Vec<u64>,

    // Step 5.
    pub selected_template: SymbolTemplate,
    pub selected_symbols: Vec<String>,
    pub selected_timeframes: Vec<String>,

    // Step 6.
    pub history_months: u8,

    // Step 7.
    pub forced_backend: Option<String>,

    // Step 8.
    pub news_filter_enabled: bool,
    pub news_blackout_minutes: u32,
    pub maintenance_window_enabled: bool,
    pub correlation_cap: f32,
    pub volatility_sigma_pause: f32,

    // Step 9.
    pub autostart_enabled: bool,
    pub start_minimized: bool,

    // Step 9.5.
    pub autonomous_mode_enabled: bool,
    pub equity_stop_pct: f32,
    pub capital_at_risk_disclosure: Option<f32>,
    pub risk_acknowledgement: Option<RiskAcknowledgement>,
    /// Risky Mode arming flag (research §4 + §7.1). Set to `true`
    /// only when the operator explicitly acknowledges the ruin-
    /// probability ceiling. The Step 10 Apply writer reads this and
    /// calls `session.enable_risky_mode(...)` iff `true`. Closes
    /// audit gap clarification #2 — AutonomyRisk wizard step now
    /// wires explicitly to RiskyMode instead of relying on a stale
    /// risk-slider == 10 heuristic.
    pub risky_mode_armed: bool,
    /// Operator-acknowledged ruin-probability ceiling (e.g. 0.99 for
    /// the operator-directive 99% S1 ruin ceiling). `None` until the
    /// operator ticks the checkbox in the AutonomyRisk step. Cleared
    /// when `risky_mode_armed` flips back to false so a re-arm always
    /// re-prompts. f64 to match the rebuilt forex_core::RiskyModeConfig
    /// numeric convention (operator directive §7.2).
    pub risky_mode_ruin_ceiling_acknowledged: Option<f64>,

    // Step 10.
    pub telemetry_opt_in: bool,
    /// Typed-signature gate (Live trading mode only). Spec / competitive
    /// analysis §1.1 TradingView pattern.
    pub live_typed_signature: Option<String>,
}

impl Default for WizardConfig {
    fn default() -> Self {
        Self {
            license_accepted: false,
            data_path: None,
            install_metadata: InstallMetadata::default(),

            operator_name: None,
            prop_firm_preset: account_profile::WIZARD_DEFAULT_PROP_FIRM_PRESET.to_string(),
            monthly_profit_target: account_profile::WIZARD_DEFAULT_MONTHLY_PROFIT_TARGET,
            trading_mode: TradingMode::default(),
            interface_mode: InterfaceMode::default(),
            risk_profile_slider: account_profile::WIZARD_DEFAULT_RISK_PROFILE,
            require_stop_loss: account_profile::WIZARD_DEFAULT_REQUIRE_SL,
            per_trade_max_risk_pct: account_profile::WIZARD_DEFAULT_PER_TRADE_RISK_PCT,
            daily_loss_reset_timezone: account_profile::WIZARD_DEFAULT_DLL_RESET_TZ.to_string(),

            ctrader_client_id: None,
            ctrader_client_secret_set: false,
            ctrader_environment: CTraderEnvironment::Demo,
            selected_ctid_trader_account_id: None,
            additional_account_ids: Vec::new(),

            selected_template: SymbolTemplate::Custom,
            selected_symbols: symbols::WIZARD_DEFAULT_SYMBOLS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            selected_timeframes: symbols::WIZARD_DEFAULT_TIMEFRAMES
                .iter()
                .map(|s| (*s).to_string())
                .collect(),

            history_months: historical::WIZARD_DEFAULT_HISTORY_MONTHS,

            forced_backend: None,

            news_filter_enabled: news_api::WIZARD_DEFAULT_NEWS_ENABLED,
            news_blackout_minutes: news_api::WIZARD_DEFAULT_NEWS_BLACKOUT_MINUTES,
            maintenance_window_enabled: news_api::WIZARD_DEFAULT_MAINTENANCE_WINDOW_ENABLED,
            correlation_cap: news_api::WIZARD_DEFAULT_CORRELATION_CAP,
            volatility_sigma_pause: news_api::WIZARD_DEFAULT_VOLATILITY_SIGMA,

            autostart_enabled: autostart::WIZARD_DEFAULT_AUTOSTART_ENABLED,
            start_minimized: autostart::WIZARD_DEFAULT_START_MINIMIZED,

            risky_mode_armed: false,
            risky_mode_ruin_ceiling_acknowledged: None,
            autonomous_mode_enabled: autonomy_risk::WIZARD_DEFAULT_AUTONOMOUS_MODE_ENABLED,
            equity_stop_pct: autonomy_risk::WIZARD_DEFAULT_EQUITY_STOP_PCT,
            capital_at_risk_disclosure: None,
            risk_acknowledgement: None,

            telemetry_opt_in: summary::WIZARD_DEFAULT_TELEMETRY_OPT_IN,
            live_typed_signature: None,
        }
    }
}

/// Spec §2 Step 3 — three-way radio Backtest / Forward (default) / Live.
/// Default = Forward, per spec §11 acceptance criterion 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingMode {
    Backtest,
    Forward,
    Live,
}

impl Default for TradingMode {
    fn default() -> Self {
        TradingMode::Forward
    }
}

impl TradingMode {
    pub fn as_str(self) -> &'static str {
        match self {
            TradingMode::Backtest => "backtest",
            TradingMode::Forward => "forward",
            TradingMode::Live => "live",
        }
    }
}

/// Competitive analysis §9.1 / §3.2 — Beginner / Advanced interface
/// gate. Beginner hides Polars debug panes, raw OAuth tokens, raw
/// protobuf inspector. Default = Beginner per the analysis doc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterfaceMode {
    #[default]
    Beginner,
    Advanced,
}

impl InterfaceMode {
    pub fn as_str(self) -> &'static str {
        match self {
            InterfaceMode::Beginner => "beginner",
            InterfaceMode::Advanced => "advanced",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CTraderEnvironment {
    Live,
    Demo,
}

impl CTraderEnvironment {
    pub fn as_str(self) -> &'static str {
        match self {
            CTraderEnvironment::Live => "live",
            CTraderEnvironment::Demo => "demo",
        }
    }
}

/// Step 5 template gallery — competitive analysis §8.4. Six entries.
/// `Custom` falls through to the existing symbol/timeframe picker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolTemplate {
    ScalpingEurusdM1,
    ScalpingMajorsM5,
    SwingD1Majors,
    TrendH1Baskets,
    MeanReversionH1Majors,
    Custom,
}

impl SymbolTemplate {
    pub fn all() -> &'static [SymbolTemplate] {
        &[
            SymbolTemplate::ScalpingEurusdM1,
            SymbolTemplate::ScalpingMajorsM5,
            SymbolTemplate::SwingD1Majors,
            SymbolTemplate::TrendH1Baskets,
            SymbolTemplate::MeanReversionH1Majors,
            SymbolTemplate::Custom,
        ]
    }

    pub fn label(self) -> &'static str {
        match self {
            SymbolTemplate::ScalpingEurusdM1 => "Scalping EURUSD M1",
            SymbolTemplate::ScalpingMajorsM5 => "Scalping majors M5",
            SymbolTemplate::SwingD1Majors => "Swing trading D1 majors",
            SymbolTemplate::TrendH1Baskets => "Trend following H1 baskets",
            SymbolTemplate::MeanReversionH1Majors => "Mean-reversion H1 majors",
            SymbolTemplate::Custom => "Custom (start from blank)",
        }
    }

    /// Slider position 1–10 per competitive analysis §8.4 each entry.
    pub fn risk_score(self) -> u8 {
        match self {
            SymbolTemplate::ScalpingEurusdM1 => 4,
            SymbolTemplate::ScalpingMajorsM5 => 3,
            SymbolTemplate::SwingD1Majors => 4,
            SymbolTemplate::TrendH1Baskets => 5,
            SymbolTemplate::MeanReversionH1Majors => 4,
            SymbolTemplate::Custom => 4,
        }
    }
}

/// Renderer-agnostic state machine. Both `wizard_ui` (egui) and
/// `forex-cli wizard` (ratatui) own one of these per session.
///
/// `current` is the active step. `state_file` is the in-memory shadow
/// of `wizard_state.json`; it's only written on Apply (Step 10) or
/// explicit Skip (spec §3 rule 1).
#[derive(Debug, Clone)]
pub struct WizardController {
    pub current: WizardState,
    pub config: WizardConfig,
    pub state_file: WizardStateFile,
    pub last_error: Option<WizardError>,
    pub cancelled: bool,
    pub finished: bool,
}

impl Default for WizardController {
    fn default() -> Self {
        Self::new()
    }
}

impl WizardController {
    pub fn new() -> Self {
        Self {
            current: WizardState::Welcome,
            config: WizardConfig::default(),
            state_file: WizardStateFile::new(),
            last_error: None,
            cancelled: false,
            finished: false,
        }
    }

    /// Resume an existing wizard from a previously-persisted file.
    /// Spec §5.2 — start at the first incomplete step.
    pub fn resume_from(state_file: WizardStateFile) -> Self {
        let current = state_file.first_incomplete_step();
        Self {
            current,
            config: WizardConfig::default(),
            state_file,
            last_error: None,
            cancelled: false,
            finished: false,
        }
    }

    /// Is the current step skippable?
    ///
    /// - `Welcome` is never skippable (spec §2 Step 1).
    /// - `Summary` is the terminal step — Apply / Cancel only.
    /// - `AutonomyRisk` is non-skippable iff `trading_mode == Live`
    ///   OR `autonomous_mode_enabled` (competitive analysis §9.2).
    pub fn is_skippable(&self) -> bool {
        match self.current {
            WizardState::Welcome | WizardState::Summary => false,
            WizardState::AutonomyRisk => {
                !(self.config.trading_mode == TradingMode::Live
                    || self.config.autonomous_mode_enabled)
            }
            _ => self.current.is_skippable_default(),
        }
    }

    /// Apply a `StepResult` from a renderer.
    pub fn apply(&mut self, result: StepResult) {
        match result {
            StepResult::NextRequested => self.advance(),
            StepResult::BackRequested => self.go_back(),
            StepResult::SkipRequested => self.skip(),
            StepResult::CancelRequested => {
                self.cancelled = true;
            }
            StepResult::StayHere => {}
        }
    }

    fn advance(&mut self) {
        // Spec §3 rule 1 — never silently skip. Mark complete then
        // transition. Welcome can't be marked complete unless license
        // is accepted (gated at the step's render).
        if !self.state_file.completed_steps.contains(&self.current) {
            self.state_file.completed_steps.push(self.current);
        }
        match self.current.next() {
            Some(next) => self.current = next,
            None => self.finished = true,
        }
    }

    fn go_back(&mut self) {
        if let Some(prev) = self.current.previous() {
            // Going back un-completes the current step so a re-traverse
            // forwards re-marks it (spec §5 — "Preserve user selections
            // through navigation" — selections survive, completion
            // status does not).
            self.state_file.completed_steps.retain(|s| *s != self.current);
            self.current = prev;
        }
    }

    fn skip(&mut self) {
        if !self.is_skippable() {
            return;
        }
        if !self.state_file.skipped_steps.contains(&self.current) {
            self.state_file.skipped_steps.push(self.current);
        }
        match self.current.next() {
            Some(next) => self.current = next,
            None => self.finished = true,
        }
    }

    /// Total step count for the progress tracker. Spec §9 mockups
    /// show "Step N / 10" — Autonomy & Risk is rendered as "9.5 / 10"
    /// rather than bumping the denominator.
    pub fn total_steps(&self) -> usize {
        WizardState::ordered().len()
    }

    pub fn step_index(&self) -> usize {
        self.current.index()
    }
}

/// egui entry point. Called from `forex-app::main` when the wizard
/// gate fires (first-run sentinel missing OR `--wizard` CLI flag).
///
/// Renders the current step. Renderers are wrapped in
/// `egui::Window::new("forex-ai Setup Wizard")` to match the modal
/// chrome described in spec §9 mockups.
pub fn wizard_ui(ctx: &egui::Context, controller: &mut WizardController) {
    if controller.cancelled || controller.finished {
        return;
    }

    egui::CentralPanel::default().show(ctx, |ui| {
        // Progress tracker (spec §9 — "Step N / 10").
        let total = controller.total_steps();
        let idx = controller.step_index() + 1;
        ui.heading(format!(
            "forex-ai Setup Wizard — Step {} / {} · {}",
            idx,
            total,
            controller.current.label()
        ));
        ui.separator();

        let result = match controller.current {
            WizardState::Welcome => welcome::render(ui, controller),
            WizardState::Path => path::render(ui, controller),
            WizardState::AccountProfile => account_profile::render(ui, controller),
            WizardState::OAuth => oauth::render(ui, controller),
            WizardState::Symbols => symbols::render(ui, controller),
            WizardState::Historical => historical::render(ui, controller),
            WizardState::Hardware => hardware::render(ui, controller),
            WizardState::NewsApi => news_api::render(ui, controller),
            WizardState::Autostart => autostart::render(ui, controller),
            WizardState::AutonomyRisk => autonomy_risk::render(ui, controller),
            WizardState::Summary => summary::render(ui, controller),
        };
        controller.apply(result);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_controller_starts_at_welcome() {
        let c = WizardController::new();
        assert_eq!(c.current, WizardState::Welcome);
        assert!(!c.cancelled);
        assert!(!c.finished);
    }

    #[test]
    fn next_advances_through_every_step() {
        let mut c = WizardController::new();
        let states = WizardState::ordered();
        for expected in states.iter().skip(1) {
            c.apply(StepResult::NextRequested);
            assert_eq!(c.current, *expected, "advance to {:?}", expected);
        }
        c.apply(StepResult::NextRequested);
        assert!(c.finished, "advancing past Summary marks finished");
    }

    #[test]
    fn back_returns_to_previous_step() {
        let mut c = WizardController::new();
        c.apply(StepResult::NextRequested); // Welcome -> Path
        c.apply(StepResult::NextRequested); // Path -> AccountProfile
        c.apply(StepResult::BackRequested); // back to Path
        assert_eq!(c.current, WizardState::Path);
    }

    #[test]
    fn back_at_first_step_is_a_no_op() {
        let mut c = WizardController::new();
        c.apply(StepResult::BackRequested);
        assert_eq!(c.current, WizardState::Welcome);
    }

    #[test]
    fn skip_advances_and_records_in_skipped_steps() {
        let mut c = WizardController::new();
        c.apply(StepResult::NextRequested); // -> Path
        c.apply(StepResult::SkipRequested); // skip Path
        assert_eq!(c.current, WizardState::AccountProfile);
        assert!(c.state_file.skipped_steps.contains(&WizardState::Path));
    }

    #[test]
    fn welcome_is_never_skippable() {
        let mut c = WizardController::new();
        assert!(!c.is_skippable());
        c.apply(StepResult::SkipRequested);
        assert_eq!(c.current, WizardState::Welcome, "skip on Welcome is a no-op");
    }

    #[test]
    fn autonomy_risk_is_blocked_when_trading_mode_live() {
        let mut c = WizardController::new();
        c.config.trading_mode = TradingMode::Live;
        c.current = WizardState::AutonomyRisk;
        assert!(!c.is_skippable());
    }

    #[test]
    fn autonomy_risk_is_blocked_when_autonomous_mode_enabled() {
        let mut c = WizardController::new();
        c.config.autonomous_mode_enabled = true;
        c.current = WizardState::AutonomyRisk;
        assert!(!c.is_skippable());
    }

    #[test]
    fn autonomy_risk_is_skippable_in_forward_mode_without_autonomy() {
        let mut c = WizardController::new();
        c.current = WizardState::AutonomyRisk;
        assert!(c.is_skippable());
    }

    #[test]
    fn cancel_marks_controller_cancelled() {
        let mut c = WizardController::new();
        c.apply(StepResult::CancelRequested);
        assert!(c.cancelled);
    }

    #[test]
    fn resume_from_jumps_to_first_incomplete_step() {
        let mut file = WizardStateFile::new();
        file.completed_steps
            .extend_from_slice(&[WizardState::Welcome, WizardState::Path]);
        let c = WizardController::resume_from(file);
        assert_eq!(c.current, WizardState::AccountProfile);
    }

    #[test]
    fn default_config_meets_operator_invariants() {
        let cfg = WizardConfig::default();
        // Operator policy: monthly target floor = 4 %.
        assert!(
            (cfg.monthly_profit_target - 0.04).abs() < f32::EPSILON,
            "monthly target defaults to operator floor"
        );
        // Default trading mode = Forward (spec §11 #1).
        assert_eq!(cfg.trading_mode, TradingMode::Forward);
        // 11 canonical timeframes — default selection is a subset and
        // must not contain H2.
        assert!(!cfg.selected_timeframes.iter().any(|tf| tf == "H2"));
    }
}
