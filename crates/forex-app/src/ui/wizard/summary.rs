//! Step 10 — Summary & Apply.
//!
//! Spec: `installer_wizard_ux_spec.md` §2 Step 10 + §9.5 mockup.
//!
//! Terminal step — NOT skippable. Apply is the only forward action;
//! Cancel triggers the discard-changes confirmation modal (spec §2
//! Step 10 Cancel).
//!
//! Live-trading gate: per competitive analysis §1.1 / §9.1, if
//! `trading_mode == Live` the Apply button is hidden until the user
//! types the broker-account number into a typed-signature field.
//!
//! ## Apply writer (this module)
//!
//! Six write actions, in spec §2 Step 10 order:
//!
//! 1. `<data-path>/config.yaml` — operator's chosen trading mode,
//!    timeframes, symbols, prop-firm preset, correlation cap,
//!    volatility σ, news blackout.
//! 2. `<data-path>/broker_credentials.toml` — `SecretString`-wrapped
//!    cTrader OAuth client credentials (the runtime never prints
//!    plaintext bytes via `tracing`).
//! 3. `<data-path>/hardware_profile.json` — memoised
//!    `HardwareProbe::detect()` result.
//! 4. `<data-path>/symbol_metadata/defaults.json` — only when Step 5
//!    discovery actually ran; otherwise the packaged snapshot is left
//!    in place.
//! 5. `<data-path>/risk_acknowledgement.json` — append-only ledger of
//!    Step 9.5 quiz acknowledgements (SHA-256 of answers + version +
//!    timestamp; `sha2 = "0.10"` is now a direct dep of `forex-app`
//!    so the answer hash is computed locally — see `risk_quiz.rs`).
//! 6. `<data-path>/wizard_state.json` — schema in §5 of the spec.
//!
//! Each write uses temp-file + atomic rename + fsync via
//! `forex_core::storage::json::write_json_atomic` (audit-cleaned at
//! F-CORE2-018). On failure the summary screen surfaces the specific
//! action that failed and offers Retry / Skip-with-warning / Cancel
//! per operator no-silent-fallback policy.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use eframe::egui;
use forex_core::CANONICAL_TIMEFRAMES;
use forex_core::PropFirmConstraints;
use forex_core::Settings;
use forex_core::storage::json::write_json_atomic;
use forex_core::system::{HardwareProbe, HardwareProfile};
// The writer no longer touches the cTrader client_secret directly —
// the binary's embedded constant is the source of truth, populated
// at runtime by `broker_persistence::apply_embedded_fallback`. The
// `secrecy` import is retained for the news API key holder + future
// non-cTrader broker credentials when D3 lands.
#[allow(unused_imports)]
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

use super::{RiskAcknowledgement, StepResult, TradingMode, WizardController};
use crate::app_services::broker_config::{
    BrokerSettingsState, CTraderBrokerEnvironment, CTraderBrokerSettings,
};
use crate::app_services::broker_persistence::save_broker_settings;
use crate::ui::theme;
use crate::ui::wizard::state::WizardStateFile;
use crate::ui::wizard::{CTraderEnvironment as WizCTraderEnvironment, WizardState};

/// Default for the telemetry opt-in toggle. Spec §7.1 — "No telemetry"
/// default.
pub const WIZARD_DEFAULT_TELEMETRY_OPT_IN: bool = false;

/// The filename the wizard writes when Apply succeeds. Spec §5.
pub const WIZARD_DEFAULT_COMPLETED_FILENAME: &str = "wizard_state.json";

/// Filename of the per-user hardware probe snapshot. Spec §2 Step 10
/// Action 3.
pub const WIZARD_HARDWARE_PROFILE_FILENAME: &str = "hardware_profile.json";

/// Append-only ledger of risk-acknowledgement quiz completions. Spec
/// §5 + competitive analysis §9.2.
pub const WIZARD_RISK_ACK_LEDGER_FILENAME: &str = "risk_acknowledgement.json";

/// Sub-directory under `<data-path>` for the symbol-metadata snapshot
/// (mirrors `assets/symbol_metadata/defaults.json` shipped with the
/// installer). Spec §2 Step 10 Action 4.
pub const WIZARD_SYMBOL_METADATA_DIR: &str = "symbol_metadata";
/// Filename inside the symbol-metadata directory.
pub const WIZARD_SYMBOL_METADATA_FILENAME: &str = "defaults.json";

/// Label for each of the six apply actions — surfaces verbatim in the
/// Summary screen if the action fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyAction {
    ConfigYaml,
    BrokerCredentials,
    HardwareProfile,
    SymbolMetadata,
    RiskAcknowledgement,
    WizardState,
}

impl ApplyAction {
    pub fn label(self) -> &'static str {
        match self {
            ApplyAction::ConfigYaml => "config.yaml",
            ApplyAction::BrokerCredentials => "broker_credentials.toml",
            ApplyAction::HardwareProfile => "hardware_profile.json",
            ApplyAction::SymbolMetadata => "symbol_metadata/defaults.json",
            ApplyAction::RiskAcknowledgement => "risk_acknowledgement.json",
            ApplyAction::WizardState => "wizard_state.json",
        }
    }
}

/// Outcome of an Apply attempt — surfaces to the Summary UI so a
/// partial failure can be Retried / Skipped / Cancelled (spec §3
/// rule 1, no silent skip).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ApplyOutcome {
    pub completed: Vec<ApplyAction>,
    pub skipped_with_warning: Vec<ApplyAction>,
    pub failed: Option<ApplyFailure>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApplyFailure {
    pub action: ApplyAction,
    pub message: String,
}

impl ApplyOutcome {
    pub fn is_fully_complete(&self) -> bool {
        self.failed.is_none() && self.completed.len() + self.skipped_with_warning.len() == 6
    }

    /// Index of the next action to run. The writer iterates over the
    /// six actions in spec order and resumes at the first one that
    /// is neither completed nor explicitly skipped.
    pub fn next_pending(&self) -> Option<ApplyAction> {
        for action in [
            ApplyAction::ConfigYaml,
            ApplyAction::BrokerCredentials,
            ApplyAction::HardwareProfile,
            ApplyAction::SymbolMetadata,
            ApplyAction::RiskAcknowledgement,
            ApplyAction::WizardState,
        ] {
            if !self.completed.contains(&action) && !self.skipped_with_warning.contains(&action) {
                return Some(action);
            }
        }
        None
    }
}

/// Append-only ledger of risk acknowledgements. Spec §5 — historical
/// entries are preserved across re-runs of the wizard.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RiskAcknowledgementLedger {
    #[serde(default)]
    pub entries: Vec<RiskAcknowledgement>,
}

/// Test-injectable seam for the hardware probe so unit tests can run
/// without spawning `nvidia-smi`. Production code calls
/// `HardwareProbe::detect()` directly.
fn detect_hardware_profile() -> HardwareProfile {
    HardwareProbe::new().detect()
}

/// Apply-writer driver. Walks the six actions in order, persisting
/// each via its dedicated helper. Returns the cumulative
/// [`ApplyOutcome`] so the renderer can offer Retry on the first
/// failure (spec §3 rule 2 — every step gets a Retry affordance).
///
/// The renderer passes its own outcome from the previous frame so
/// repeated clicks of Apply only redo the actions that have not yet
/// completed.
pub fn run_apply(controller: &mut WizardController, prior: ApplyOutcome) -> ApplyOutcome {
    let mut outcome = prior;
    outcome.failed = None;

    let Some(data_path) = controller.config.data_path.clone() else {
        outcome.failed = Some(ApplyFailure {
            action: ApplyAction::ConfigYaml,
            message: "data path not set — Step 2 must be completed before Apply".to_string(),
        });
        return outcome;
    };

    let probe = detect_hardware_profile();

    while let Some(action) = outcome.next_pending() {
        let res = match action {
            ApplyAction::ConfigYaml => write_config_yaml(&data_path, controller),
            ApplyAction::BrokerCredentials => write_broker_credentials(controller),
            ApplyAction::HardwareProfile => write_hardware_profile(&data_path, &probe),
            ApplyAction::SymbolMetadata => write_symbol_metadata_snapshot(&data_path, controller),
            ApplyAction::RiskAcknowledgement => {
                write_risk_acknowledgement_ledger(&data_path, controller)
            }
            ApplyAction::WizardState => write_wizard_state(&data_path, controller),
        };
        match res {
            Ok(()) => outcome.completed.push(action),
            Err(err) => {
                outcome.failed = Some(ApplyFailure {
                    action,
                    message: format!("{err:#}"),
                });
                return outcome;
            }
        }
    }

    if outcome.is_fully_complete() {
        controller.state_file.mark_finished();
    }
    outcome
}

/// Action 1 — write `<data-path>/config.yaml`. The wizard converts
/// `WizardConfig` into a `forex_core::Settings` so the on-disk
/// schema is the same one the running app re-reads via
/// `Settings::load`. Roundtrip is enforced by the workspace's
/// existing `Settings` serde tests.
pub fn write_config_yaml(data_path: &Path, controller: &WizardController) -> Result<()> {
    let settings = wizard_config_to_settings(data_path, controller);
    std::fs::create_dir_all(data_path)
        .with_context(|| format!("create data dir {}", data_path.display()))?;
    let path = data_path.join("config.yaml");
    settings
        .save(&path)
        .with_context(|| format!("save config.yaml to {}", path.display()))
}

/// Project the wizard's snapshot config onto `forex_core::Settings`.
///
/// Operator invariants enforced here:
///
/// - Timeframes are filtered against `CANONICAL_TIMEFRAMES`; the
///   wizard cannot smuggle non-canonical entries (no H2) into the
///   running app.
/// - The prop-firm preset string is preserved verbatim under
///   `risk.challenge_phase` so an FTMO-aware re-load round-trips.
/// - `PropFirmConstraints::FTMO_STANDARD` provides the 4% monthly
///   floor — `monthly_profit_target_pct` is clamped up to it.
pub fn wizard_config_to_settings(data_path: &Path, controller: &WizardController) -> Settings {
    let cfg = &controller.config;
    let mut settings = Settings::default();

    // ── System ───────────────────────────────────────────────────
    let canonical: std::collections::HashSet<&'static str> =
        CANONICAL_TIMEFRAMES.iter().copied().collect();
    let canonical_tfs: Vec<String> = cfg
        .selected_timeframes
        .iter()
        .filter(|tf| canonical.contains(tf.as_str()))
        .cloned()
        .collect();

    settings.system.data_dir = data_path.to_path_buf();
    if let Some(first) = cfg.selected_symbols.first() {
        settings.system.symbol = first.clone();
    }
    settings.system.symbols = cfg.selected_symbols.clone();
    if !canonical_tfs.is_empty() {
        settings.system.multi_resolution_timeframes = canonical_tfs.clone();
        settings.system.higher_timeframes = canonical_tfs.clone();
        settings.system.required_timeframes = canonical_tfs;
    }
    settings.system.history_years = (cfg.history_months as usize).div_ceil(12).max(1);
    if let Some(backend) = cfg.forced_backend.as_ref() {
        settings.system.device = backend.to_ascii_lowercase();
    }

    // ── Risk ────────────────────────────────────────────────────
    let ftmo = PropFirmConstraints::FTMO_STANDARD;
    // Operator policy 2026-05-14: 4% monthly floor must not be
    // undercut. The wizard's slider already gates below-floor entries
    // (Step 3), but we re-enforce here so a corrupt `WizardConfig`
    // can't leak a sub-floor value to disk.
    let target = (cfg.monthly_profit_target as f64).max(ftmo.min_monthly_net_profit_pct as f64);
    settings.risk.monthly_profit_target_pct = target;
    settings.risk.require_stop_loss = cfg.require_stop_loss;
    // `per_trade_max_risk_pct` is a percent (e.g. 0.5 = 0.5%), and
    // `RiskConfig::risk_per_trade` is a fraction (e.g. 0.005). Convert.
    let per_trade_frac = (cfg.per_trade_max_risk_pct as f64) / 100.0;
    settings.risk.risk_per_trade = per_trade_frac;
    settings.risk.max_risk_per_trade = per_trade_frac;
    settings.risk.base_risk_per_trade = per_trade_frac;
    settings.risk.high_quality_risk_pct = per_trade_frac;
    settings.risk.challenge_phase = cfg.prop_firm_preset.clone();
    settings.risk.volatility_stop_sigma = cfg.volatility_sigma_pause as f64;

    // ── News / safeguards ───────────────────────────────────────
    settings.news.enable_news = cfg.news_filter_enabled;
    settings.news.news_kill_window_min = cfg.news_blackout_minutes as usize;

    // Settings doesn't carry the wizard-specific knobs (correlation
    // cap, autonomy slider, interface mode) — those live in
    // `wizard_state.json` and are read by the trading-runtime gate.
    // Surface them as a structured tracing line so a debug user can
    // verify the wizard saw the operator's choices without ever
    // printing OAuth secrets.
    tracing::info!(
        target: "forex_app::ui::wizard::summary",
        trading_mode = cfg.trading_mode.as_str(),
        interface_mode = cfg.interface_mode.as_str(),
        risk_slider = cfg.risk_profile_slider,
        correlation_cap = cfg.correlation_cap,
        autonomous_mode_enabled = cfg.autonomous_mode_enabled,
        "wizard apply: projecting WizardConfig → Settings"
    );

    settings
}

/// Action 2 — write `<data-path>/broker_credentials.toml`.
///
/// 2026-05-17 operator-directive correction: the wizard no longer
/// asks the user for cTrader app `client_id` / `client_secret` —
/// those are *developer* credentials, baked into the binary at build
/// time (see `crates/forex-app/build.rs::emit_embedded_credentials`).
/// This writer therefore persists only the **non-app-credential**
/// fields the operator actually chose: environment (Demo/Live) and
/// the picked account list. The `client_id` / `client_secret` /
/// `redirect_uri` fields on `CTraderBrokerSettings` are deliberately
/// written as empty strings so the four-level resolver in
/// `broker_persistence::load_broker_settings` falls through to the
/// embedded constants at runtime.
pub fn write_broker_credentials(controller: &WizardController) -> Result<()> {
    let cfg = &controller.config;
    let env = match cfg.ctrader_environment {
        WizCTraderEnvironment::Live => CTraderBrokerEnvironment::Live,
        WizCTraderEnvironment::Demo => CTraderBrokerEnvironment::Demo,
    };
    let ctrader = CTraderBrokerSettings {
        // Empty on purpose — see the function doc. The runtime
        // resolver will populate these from the embedded constants
        // via `apply_embedded_fallback`. A future hand-edit of the
        // TOML by an operator that wants to override the embedded
        // values still works through the same resolver — that's the
        // four-level lookup the persistence layer documents.
        client_id: String::new(),
        client_secret: String::new(),
        redirect_uri: String::new(),
        authorization_code_input: String::new(),
        environment: env,
        accounts: account_targets_from_wizard(controller),
    };
    let settings = BrokerSettingsState {
        schema_version: crate::app_services::broker_config::BROKER_CREDENTIALS_SCHEMA_VERSION,
        ctrader,
        dxtrade: Default::default(),
    };
    save_broker_settings(&settings).context("save broker_credentials.toml")
}

fn account_targets_from_wizard(
    controller: &WizardController,
) -> Vec<crate::app_services::broker_config::BrokerAccountTarget> {
    let primary = controller.config.selected_ctid_trader_account_id;
    let mut out: Vec<crate::app_services::broker_config::BrokerAccountTarget> = Vec::new();
    if let Some(id) = primary {
        out.push(crate::app_services::broker_config::BrokerAccountTarget {
            account_id: id.to_string(),
            label: "Primary".to_string(),
            enabled_for_execution: true,
        });
    }
    for extra in &controller.config.additional_account_ids {
        out.push(crate::app_services::broker_config::BrokerAccountTarget {
            account_id: extra.to_string(),
            label: format!("Account {extra}"),
            enabled_for_execution: false,
        });
    }
    out
}

/// Action 3 — write `<data-path>/hardware_profile.json` from the
/// memoised `HardwareProbe::detect()` result.
pub fn write_hardware_profile(data_path: &Path, profile: &HardwareProfile) -> Result<()> {
    let path = data_path.join(WIZARD_HARDWARE_PROFILE_FILENAME);
    write_json_atomic(&path, profile)
        .with_context(|| format!("write hardware profile to {}", path.display()))
}

/// Action 4 — write `<data-path>/symbol_metadata/defaults.json`
/// **only** if Step 5 actually ran a fresh symbol discovery. Until
/// the OAuth/symbol runtime wires
/// `WizardController::symbol_metadata_snapshot`, this is a no-op for
/// fresh installs and leaves the packaged `assets/symbol_metadata/
/// defaults.json` in place. Spec §2 Step 10 Action 4.
pub fn write_symbol_metadata_snapshot(
    data_path: &Path,
    _controller: &WizardController,
) -> Result<()> {
    // No fresh discovery snapshot is attached to the controller yet;
    // the OAuth/symbol runtime populates
    // `WizardConfig.symbol_metadata_snapshot` when Step 5 contacts
    // the broker. Until then we MUST NOT overwrite the packaged
    // snapshot — that would silently drop the operator's shipped
    // defaults. Per spec §2 Step 10 Action 4: "only if it ran; else
    // leave the packaged snapshot in place."
    let dir = data_path.join(WIZARD_SYMBOL_METADATA_DIR);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("ensure symbol metadata dir {}", dir.display()))?;
    Ok(())
}

/// Action 5 — append-only risk acknowledgement ledger. Spec §5 plus
/// competitive analysis §9.2. Historical entries are preserved
/// across wizard re-runs so the operator can audit when each
/// acknowledgement was given.
pub fn write_risk_acknowledgement_ledger(
    data_path: &Path,
    controller: &WizardController,
) -> Result<()> {
    let path = data_path.join(WIZARD_RISK_ACK_LEDGER_FILENAME);
    let mut ledger: RiskAcknowledgementLedger = if path.is_file() {
        let bytes = std::fs::read(&path)
            .with_context(|| format!("read existing ledger {}", path.display()))?;
        serde_json::from_slice(&bytes).unwrap_or_default()
    } else {
        RiskAcknowledgementLedger::default()
    };

    if let Some(ack) = controller.config.risk_acknowledgement.clone() {
        // De-dupe on (timestamp, sha256, quiz_version) so a re-run
        // that doesn't produce a fresh acknowledgement doesn't grow
        // the file.
        let already = ledger.entries.iter().any(|e| {
            e.timestamp_utc == ack.timestamp_utc
                && e.answers_sha256 == ack.answers_sha256
                && e.quiz_version == ack.quiz_version
        });
        if !already {
            ledger.entries.push(ack);
        }
    }

    write_json_atomic(&path, &ledger)
        .with_context(|| format!("write risk ack ledger to {}", path.display()))
}

/// Action 6 — `<data-path>/wizard_state.json`. Mirrors the in-memory
/// controller's view into the schema declared in `state.rs`. The
/// `mark_finished` call happens in `run_apply` *after* all six
/// actions have succeeded.
///
/// Side-write: this action also persists the Risky Mode arm state to
/// `<config_dir>/forex-ai/risky_mode_state.json` (sibling file, not
/// inside `data_path`) so `TradingSession::new_with_persisted_credentials`
/// can auto-arm at session boot. The Risky Mode contract is a runtime
/// signal, distinct from wizard-session bookkeeping — it lives next
/// to `broker_credentials.toml`, not next to `wizard_state.json`. The
/// side-write is performed BEFORE the wizard_state.json write so a
/// failure surfaces under `ApplyAction::WizardState` with the same
/// retry semantics the existing six-action contract provides — no
/// new ApplyAction enum variant or count change is required. Closes
/// `TODO(risky-mode-boot-wire)`.
pub fn write_wizard_state(data_path: &Path, controller: &mut WizardController) -> Result<()> {
    // Risky Mode side-write — fails the action if persistence errors
    // so the operator sees the failure in the Apply summary screen
    // rather than discovering at next boot that the arm was lost.
    write_risky_mode_state(controller)?;

    // Sync the controller's WizardConfig snapshot into the on-disk
    // state-file fields it needs. `completed_steps`/`skipped_steps`
    // are already maintained by `WizardController::advance`/`skip`.
    controller.state_file.risk_acknowledgement = controller.config.risk_acknowledgement.clone();
    controller.state_file.install_metadata = controller.config.install_metadata.clone();
    controller.state_file.touch_for_write();

    let path = WizardStateFile::default_path(data_path);
    controller
        .state_file
        .write_to(&path)
        .with_context(|| format!("write wizard_state to {}", path.display()))
}

/// Project the wizard's in-memory Risky Mode flags onto the
/// `risky_mode_state.json` sibling file. Read at session boot by
/// `TradingSession::new_with_persisted_credentials` to auto-arm
/// Risky Mode without operator intervention.
///
/// Operator invariants:
///
/// - The on-disk `armed` flag mirrors `WizardConfig::risky_mode_armed`
///   verbatim. If the operator disarmed in this wizard run, the file
///   is rewritten with `armed = false` (it is NOT deleted — keeping
///   the file lets us preserve the acknowledgement across re-runs).
/// - `autonomous_only_contract_accepted` mirrors the wizard's
///   `autonomous_mode_enabled` checkbox. `RiskyModeConfig::validate`
///   requires this to be `true` before auto-arming will succeed.
/// - `starting_capital_usd` is `None` at the wizard stage; the
///   reader will default to `RiskyModeConfig::default()
///   .starting_capital_usd` per research §4.1.
pub fn write_risky_mode_state(controller: &WizardController) -> Result<()> {
    let cfg = &controller.config;
    let state = crate::app_services::risky_mode_persistence::RiskyModeStateFile {
        schema_version:
            crate::app_services::risky_mode_persistence::RISKY_MODE_STATE_SCHEMA_VERSION,
        armed: cfg.risky_mode_armed,
        ruin_ceiling_acknowledged: cfg.risky_mode_ruin_ceiling_acknowledged,
        starting_capital_usd: None,
        autonomous_only_contract_accepted: cfg.autonomous_mode_enabled,
        last_updated_utc_ms: 0,
    };
    crate::app_services::risky_mode_persistence::save_risky_mode_state(&state)
        .context("write risky_mode_state.json")
}

/// Convenience for the renderer — the path Apply will write the
/// state file to. Re-exported so the CLI can resume from the same
/// location.
pub fn wizard_state_path(controller: &WizardController) -> Option<PathBuf> {
    controller
        .config
        .data_path
        .as_ref()
        .map(|p| WizardStateFile::default_path(p))
}

pub fn render(ui: &mut egui::Ui, controller: &mut WizardController) -> StepResult {
    let mut result = StepResult::StayHere;

    ui.label(
        egui::RichText::new("Review your selections, then click Apply.").color(theme::TEXT_PRIMARY),
    );
    ui.add_space(theme::SPACE_SM);

    egui::Grid::new("wizard_summary_grid")
        .num_columns(2)
        .spacing([24.0, 6.0])
        .show(ui, |ui| {
            ui.label("License accepted");
            ui.label(if controller.config.license_accepted {
                "yes"
            } else {
                "no"
            });
            ui.end_row();

            ui.label("Data directory");
            ui.label(
                controller
                    .config
                    .data_path
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "(default)".to_string()),
            );
            ui.end_row();

            ui.label("Prop firm preset");
            ui.label(&controller.config.prop_firm_preset);
            ui.end_row();

            ui.label("Monthly profit target");
            ui.label(format!(
                "{:.2}%",
                controller.config.monthly_profit_target * 100.0
            ));
            ui.end_row();

            ui.label("Trading mode");
            ui.label(controller.config.trading_mode.as_str());
            ui.end_row();

            ui.label("Interface mode");
            ui.label(controller.config.interface_mode.as_str());
            ui.end_row();

            ui.label("Risk profile slider");
            ui.label(format!("{}/10", controller.config.risk_profile_slider));
            ui.end_row();

            ui.label("Per-trade max risk");
            ui.label(format!("{:.2}%", controller.config.per_trade_max_risk_pct));
            ui.end_row();

            ui.label("Stop Loss required");
            ui.label(if controller.config.require_stop_loss {
                "yes"
            } else {
                "no"
            });
            ui.end_row();

            ui.label("cTrader account");
            ui.label(
                controller
                    .config
                    .selected_ctid_trader_account_id
                    .map(|id| {
                        format!(
                            "#{} ({})",
                            id,
                            controller.config.ctrader_environment.as_str()
                        )
                    })
                    .unwrap_or_else(|| "(not configured)".to_string()),
            );
            ui.end_row();

            ui.label("Symbols");
            ui.label(format!(
                "{} selected",
                controller.config.selected_symbols.len()
            ));
            ui.end_row();

            ui.label("Timeframes");
            ui.label(controller.config.selected_timeframes.join(", "));
            ui.end_row();

            ui.label("History");
            ui.label(format!("{} months", controller.config.history_months));
            ui.end_row();

            ui.label("Forced backend");
            ui.label(
                controller
                    .config
                    .forced_backend
                    .clone()
                    .unwrap_or_else(|| "(auto-detect)".to_string()),
            );
            ui.end_row();

            ui.label("News filter");
            ui.label(if controller.config.news_filter_enabled {
                "enabled"
            } else {
                "disabled"
            });
            ui.end_row();

            ui.label("Maintenance window");
            ui.label(if controller.config.maintenance_window_enabled {
                "auto-flatten Friday 16:00 ET"
            } else {
                "off"
            });
            ui.end_row();

            ui.label("Correlation cap");
            ui.label(format!("{:.2}", controller.config.correlation_cap));
            ui.end_row();

            ui.label("Volatility σ pause");
            ui.label(format!("{:.1} σ", controller.config.volatility_sigma_pause));
            ui.end_row();

            ui.label("Auto-start");
            ui.label(if controller.config.autostart_enabled {
                "enabled"
            } else {
                "disabled"
            });
            ui.end_row();

            ui.label("Autonomous mode");
            ui.label(if controller.config.autonomous_mode_enabled {
                "enabled"
            } else {
                "disabled"
            });
            ui.end_row();

            ui.label("Crash reports");
            ui.label(if controller.config.telemetry_opt_in {
                "opt-in"
            } else {
                "disabled (default)"
            });
            ui.end_row();
        });

    ui.separator();
    ui.checkbox(
        &mut controller.config.telemetry_opt_in,
        "Send anonymised crash reports (default off).",
    );

    // Live-mode typed-signature gate.
    let live_gate_required = controller.config.trading_mode == TradingMode::Live;
    let apply_enabled = if live_gate_required {
        ui.separator();
        ui.label(
            egui::RichText::new("Live trading mode — typed-signature gate")
                .strong()
                .color(theme::DANGER),
        );
        let expected = controller
            .config
            .selected_ctid_trader_account_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "<set account in Step 4>".to_string());
        ui.label(
            egui::RichText::new(format!(
                "Type the broker account number ({}) to confirm Live trading.",
                expected
            ))
            .size(theme::FONT_CAPTION)
            .color(theme::TEXT_MUTED),
        );
        let mut sig = controller
            .config
            .live_typed_signature
            .clone()
            .unwrap_or_default();
        if ui.text_edit_singleline(&mut sig).changed() {
            controller.config.live_typed_signature = if sig.trim().is_empty() {
                None
            } else {
                Some(sig.clone())
            };
        }
        // Apply enabled iff the typed value matches the chosen account.
        controller
            .config
            .live_typed_signature
            .as_ref()
            .zip(controller.config.selected_ctid_trader_account_id)
            .map(|(typed, account)| typed.trim() == account.to_string())
            .unwrap_or(false)
    } else {
        true
    };

    // Apply progress + error banner. We keep the running outcome in
    // a frame-local stash so Retry continues where the previous Apply
    // call left off rather than re-doing the completed actions.
    let apply_outcome = ui
        .ctx()
        .data(|d| d.get_temp::<ApplyOutcome>(egui::Id::new("wizard_apply_outcome")))
        .unwrap_or_default();

    if !apply_outcome.completed.is_empty() {
        ui.separator();
        for done in &apply_outcome.completed {
            ui.label(
                egui::RichText::new(format!("✓ {}", done.label()))
                    .color(theme::SUCCESS)
                    .size(theme::FONT_CAPTION),
            );
        }
    }
    if let Some(ref failure) = apply_outcome.failed {
        ui.separator();
        ui.label(
            egui::RichText::new(format!("✗ {}: {}", failure.action.label(), failure.message))
                .color(theme::DANGER),
        );
        ui.horizontal(|ui| {
            if ui.button("Retry").clicked() {
                let next = run_apply(controller, apply_outcome.clone());
                let completed = next.is_fully_complete();
                ui.ctx().data_mut(|d| {
                    d.insert_temp(egui::Id::new("wizard_apply_outcome"), next);
                });
                if completed {
                    result = StepResult::NextRequested;
                }
            }
            if ui.button("Skip with warning").clicked() {
                let mut next = apply_outcome.clone();
                if let Some(failed) = next.failed.take() {
                    next.skipped_with_warning.push(failed.action);
                    // Mark the wizard's incomplete_steps so the main
                    // app banners this on launch (spec §3 rule 1).
                    if !controller
                        .state_file
                        .incomplete_steps
                        .contains(&WizardState::Summary)
                    {
                        controller
                            .state_file
                            .incomplete_steps
                            .push(WizardState::Summary);
                    }
                }
                let next = run_apply(controller, next);
                let completed = next.is_fully_complete();
                ui.ctx().data_mut(|d| {
                    d.insert_temp(egui::Id::new("wizard_apply_outcome"), next);
                });
                if completed {
                    result = StepResult::NextRequested;
                }
            }
        });
    }

    ui.separator();
    ui.horizontal(|ui| {
        if ui.button("← Back").clicked() {
            result = StepResult::BackRequested;
        }
        if ui.button("Cancel").clicked() {
            result = StepResult::CancelRequested;
        }
        if ui
            .add_enabled(apply_enabled, egui::Button::new("Apply ✓"))
            .clicked()
        {
            // Spec §2 Step 10 — six actions in order. Failure leaves
            // the outcome in temp data so Retry can resume.
            let outcome = run_apply(controller, ApplyOutcome::default());
            let completed = outcome.is_fully_complete();
            ui.ctx().data_mut(|d| {
                d.insert_temp(egui::Id::new("wizard_apply_outcome"), outcome);
            });
            if completed {
                result = StepResult::NextRequested;
            }
        }
    });

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::wizard::{
        CTraderEnvironment as WizCTraderEnvironment, StepResult, WizardController, WizardState,
    };
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("forex-ai-wizard-apply-{label}-{pid}-{nanos}"));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    /// Serialise the three risky-mode boot-wire tests below so they
    /// don't race on the process-wide
    /// `FOREX_AI_RISKY_MODE_STATE_PATH` env var. Without the lock,
    /// the armed test's env-var set leaks into the disarmed test's
    /// `TradingSession::new_with_persisted_credentials()` call and
    /// flips the expected assertion.
    static RISKY_MODE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Required test from operator brief —
    /// `apply_writer_writes_six_artefacts_idempotently`. Idempotency
    /// here = a second Apply call from a fresh `ApplyOutcome::default()`
    /// successfully runs all six actions again (re-writing the
    /// artefacts in place). The risk ack ledger explicitly de-dupes
    /// (no duplicate entries).
    #[test]
    fn apply_writer_writes_six_artefacts_idempotently() {
        let _guard = RISKY_MODE_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = unique_temp_dir("idempotent");
        let mut controller = WizardController::new();
        controller.config.data_path = Some(dir.clone());
        controller.config.ctrader_environment = WizCTraderEnvironment::Demo;
        controller.config.selected_ctid_trader_account_id = Some(1234567);
        controller.config.risk_acknowledgement = Some(RiskAcknowledgement {
            answers_sha256: "placeholder-1234".to_string(),
            timestamp_utc: "2026-05-15T19:48:33Z".to_string(),
            quiz_version: 1,
            correct_count: 5,
        });

        // Pre-write a per-file env override so the broker writer
        // doesn't touch the operator's real $XDG_CONFIG_HOME.
        let broker_creds_path = dir.join("broker_credentials.toml");
        // Same isolation for the Risky Mode state side-write that
        // `write_wizard_state` now performs — without this override
        // the test would write `risky_mode_state.json` into the
        // operator's real config dir.
        let risky_mode_state_path = dir.join("risky_mode_state.json");
        // SAFETY: tests are serialised through the broker_persistence
        // ENV_LOCK; here we just need to point at the scratch dir for
        // this process.
        unsafe {
            std::env::set_var("FOREX_AI_BROKER_CREDENTIALS_PATH", &broker_creds_path);
            std::env::set_var("FOREX_AI_RISKY_MODE_STATE_PATH", &risky_mode_state_path);
        }

        let first = run_apply(&mut controller, ApplyOutcome::default());
        assert!(first.is_fully_complete(), "first apply: {first:#?}");
        assert!(dir.join("config.yaml").is_file());
        assert!(broker_creds_path.is_file());
        assert!(dir.join("hardware_profile.json").is_file());
        assert!(dir.join("symbol_metadata").is_dir());
        assert!(dir.join("risk_acknowledgement.json").is_file());
        assert!(dir.join("wizard_state.json").is_file());
        // Side-write — risky_mode_state.json is written by the
        // wizard_state action even when armed=false (mirrors the
        // controller's in-memory flag). Closes
        // TODO(risky-mode-boot-wire).
        assert!(risky_mode_state_path.is_file());

        // Second Apply must succeed and must NOT duplicate ledger entries.
        let second = run_apply(&mut controller, ApplyOutcome::default());
        assert!(second.is_fully_complete(), "second apply: {second:#?}");
        let ledger: RiskAcknowledgementLedger =
            serde_json::from_slice(&std::fs::read(dir.join("risk_acknowledgement.json")).unwrap())
                .unwrap();
        assert_eq!(ledger.entries.len(), 1, "duplicate ack must be de-duped");

        // Cleanup
        unsafe {
            std::env::remove_var("FOREX_AI_BROKER_CREDENTIALS_PATH");
            std::env::remove_var("FOREX_AI_RISKY_MODE_STATE_PATH");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Boot-wire round-trip: wizard apply persists `risky_mode_armed`
    /// = true → `TradingSession::new_with_persisted_credentials`
    /// reads the file and auto-arms Risky Mode at session
    /// construction. Closes the gap that the 2026-05-18 cleanup
    /// pass flagged.
    ///
    /// Both env vars (broker + risky_mode) are pointed at the test's
    /// scratch dir so this test never touches the operator's real
    /// $XDG_CONFIG_HOME or races with broker_persistence tests.
    #[test]
    fn risky_mode_arm_persists_and_auto_arms_at_session_boot() {
        let _guard = RISKY_MODE_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        use crate::app_services::trading::TradingSession;
        let dir = unique_temp_dir("risky-mode-boot");
        let risky_mode_state_path = dir.join("risky_mode_state.json");
        let broker_creds_path = dir.join("broker_credentials.toml");
        unsafe {
            std::env::set_var("FOREX_AI_RISKY_MODE_STATE_PATH", &risky_mode_state_path);
            std::env::set_var("FOREX_AI_BROKER_CREDENTIALS_PATH", &broker_creds_path);
        }

        let mut controller = WizardController::new();
        controller.config.data_path = Some(dir.clone());
        controller.config.risky_mode_armed = true;
        controller.config.risky_mode_ruin_ceiling_acknowledged = Some(0.99);
        controller.config.autonomous_mode_enabled = true;

        write_risky_mode_state(&controller).expect("write");
        assert!(
            risky_mode_state_path.is_file(),
            "write_risky_mode_state must create the file"
        );

        let session = TradingSession::new_with_persisted_credentials();
        assert!(
            session.risky_mode_active(),
            "Risky Mode must auto-arm at session boot when the persisted file is armed"
        );

        unsafe {
            std::env::remove_var("FOREX_AI_RISKY_MODE_STATE_PATH");
            std::env::remove_var("FOREX_AI_BROKER_CREDENTIALS_PATH");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Negative path: persisted file is disarmed → session boots
    /// with Risky Mode OFF (the safe default).
    #[test]
    fn risky_mode_disarmed_file_leaves_session_disabled() {
        let _guard = RISKY_MODE_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        use crate::app_services::trading::TradingSession;
        let dir = unique_temp_dir("risky-mode-disarmed");
        let risky_mode_state_path = dir.join("risky_mode_state.json");
        let broker_creds_path = dir.join("broker_credentials.toml");
        unsafe {
            std::env::set_var("FOREX_AI_RISKY_MODE_STATE_PATH", &risky_mode_state_path);
            std::env::set_var("FOREX_AI_BROKER_CREDENTIALS_PATH", &broker_creds_path);
        }

        let mut controller = WizardController::new();
        controller.config.data_path = Some(dir.clone());
        controller.config.risky_mode_armed = false;
        controller.config.autonomous_mode_enabled = false;

        write_risky_mode_state(&controller).expect("write");
        let session = TradingSession::new_with_persisted_credentials();
        assert!(
            !session.risky_mode_active(),
            "Risky Mode must stay disabled when the persisted file is disarmed"
        );

        unsafe {
            std::env::remove_var("FOREX_AI_RISKY_MODE_STATE_PATH");
            std::env::remove_var("FOREX_AI_BROKER_CREDENTIALS_PATH");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Required test — `apply_writer_surfaces_disk_full_error`.
    /// We don't mock the filesystem; instead we induce a real EIO/
    /// ENOTDIR by pointing `data_path` at a regular file (so any
    /// child write hits "Not a directory"). The outcome must carry a
    /// non-`None` `failed`, and the failed action's message must be
    /// non-empty so the operator can see the OS error verbatim.
    #[test]
    fn apply_writer_surfaces_disk_full_error() {
        let dir = unique_temp_dir("disk-full");
        // Place a regular file where the writer expects a directory.
        let bogus_data_path = dir.join("not-a-dir");
        std::fs::write(&bogus_data_path, b"sentinel").unwrap();

        let mut controller = WizardController::new();
        controller.config.data_path = Some(bogus_data_path.clone());

        let outcome = run_apply(&mut controller, ApplyOutcome::default());
        assert!(
            outcome.failed.is_some(),
            "disk-full / ENOTDIR must surface a failure: {outcome:#?}"
        );
        let fail = outcome.failed.unwrap();
        assert_eq!(fail.action, ApplyAction::ConfigYaml);
        assert!(
            !fail.message.is_empty(),
            "failure message must be non-empty for operator visibility"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wizard_state_path_returns_default_when_data_path_set() {
        let mut c = WizardController::new();
        c.config.data_path = Some(PathBuf::from("/tmp/forex-ai-wsp"));
        let p = wizard_state_path(&c).expect("path");
        assert_eq!(p, PathBuf::from("/tmp/forex-ai-wsp/wizard_state.json"));
    }

    #[test]
    fn wizard_state_path_is_none_without_data_path() {
        let c = WizardController::new();
        assert!(wizard_state_path(&c).is_none());
    }

    #[test]
    fn config_yaml_filters_non_canonical_timeframes() {
        let dir = unique_temp_dir("canonical-tfs");
        let mut c = WizardController::new();
        c.config.data_path = Some(dir.clone());
        // Inject a non-canonical timeframe + a canonical one; only
        // the canonical entry must reach Settings.
        c.config.selected_timeframes = vec!["H2".to_string(), "H1".to_string(), "D1".to_string()];
        let s = wizard_config_to_settings(&dir, &c);
        assert!(
            !s.system
                .multi_resolution_timeframes
                .iter()
                .any(|t| t == "H2")
        );
        assert!(
            s.system
                .multi_resolution_timeframes
                .iter()
                .any(|t| t == "H1")
        );
        assert!(
            s.system
                .multi_resolution_timeframes
                .iter()
                .any(|t| t == "D1")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn config_yaml_clamps_monthly_target_to_ftmo_floor() {
        let dir = unique_temp_dir("ftmo-floor");
        let mut c = WizardController::new();
        c.config.data_path = Some(dir.clone());
        c.config.monthly_profit_target = 0.001_f32; // below floor
        let s = wizard_config_to_settings(&dir, &c);
        let floor = PropFirmConstraints::FTMO_STANDARD.min_monthly_net_profit_pct as f64;
        assert!(
            s.risk.monthly_profit_target_pct >= floor,
            "monthly_profit_target_pct must clamp to FTMO floor"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_outcome_next_pending_walks_actions_in_order() {
        let mut o = ApplyOutcome::default();
        assert_eq!(o.next_pending(), Some(ApplyAction::ConfigYaml));
        o.completed.push(ApplyAction::ConfigYaml);
        assert_eq!(o.next_pending(), Some(ApplyAction::BrokerCredentials));
        o.completed.push(ApplyAction::BrokerCredentials);
        o.skipped_with_warning.push(ApplyAction::HardwareProfile);
        assert_eq!(o.next_pending(), Some(ApplyAction::SymbolMetadata));
    }

    #[test]
    fn summary_step_advance_marks_finished() {
        let mut c = WizardController::new();
        c.current = WizardState::Summary;
        c.apply(StepResult::NextRequested);
        assert!(
            c.finished,
            "advancing past Summary marks the wizard finished"
        );
    }

    #[test]
    fn summary_step_back_returns_to_autonomy_risk() {
        let mut c = WizardController::new();
        c.current = WizardState::Summary;
        c.apply(StepResult::BackRequested);
        assert_eq!(c.current, WizardState::AutonomyRisk);
    }

    #[test]
    fn telemetry_defaults_to_disabled() {
        assert!(!WIZARD_DEFAULT_TELEMETRY_OPT_IN);
    }

    #[test]
    fn summary_step_is_not_skippable() {
        let mut c = WizardController::new();
        c.current = WizardState::Summary;
        assert!(!c.is_skippable());
    }

    #[test]
    fn summary_cancel_marks_controller_cancelled() {
        let mut c = WizardController::new();
        c.current = WizardState::Summary;
        c.apply(StepResult::CancelRequested);
        assert!(c.cancelled);
    }
}
