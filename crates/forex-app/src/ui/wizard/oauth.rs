//! Step 4 — cTrader OAuth onboarding (4.1 register → 4.4 account probe).
//!
//! Spec: `installer_wizard_ux_spec.md` §2 Step 4 + §9.2 mockup.
//!
//! This step DOES NOT perform a live broker round-trip — the actual
//! flow is driven by `ProductionCTraderLiveAuthBackend` in
//! `crates/forex-app/src/app_services/ctrader_live_auth.rs`. The
//! wizard's responsibility here is the UI plumbing + the CSRF
//! state-machine handoff. Spec §11 acceptance criterion 4:
//! "OAuth tokens are persisted only after the flow completes —
//! no half-written `broker_credentials.toml`."

use eframe::egui;

use super::{CTraderEnvironment, StepResult, WizardController};
use crate::ui::theme;

/// Spec §2 Step 4.2 — loopback port allocator. RFC 8252 §7.3 fallback
/// list. Must match `CTraderLoopbackConfig` at
/// `app_services/ctrader_live_auth.rs:28`.
pub const WIZARD_DEFAULT_OAUTH_LOOPBACK_PORTS: &[u16] = &[7777, 7878, 8989];

/// Spec §2 Step 4.2 — browser callback timeout (matches
/// `CTRADER_CALLBACK_TIMEOUT` at `ctrader_live_auth.rs:24`).
pub const WIZARD_DEFAULT_OAUTH_CALLBACK_TIMEOUT_SECONDS: u64 = 300;

/// Sub-step within the OAuth screen. The wizard re-renders the same
/// step until the user clicks "Continue" — the sub-step is internal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OAuthSubStep {
    /// 4.1 Register app — credentials text fields.
    RegisterApp,
    /// 4.2 Sign in with cTID — browser handoff.
    SignIn,
    /// 4.3 Account picker.
    PickAccount,
    /// 4.4 Per-account auth probe.
    AuthProbe,
}

pub fn render(ui: &mut egui::Ui, controller: &mut WizardController) -> StepResult {
    let mut result = StepResult::StayHere;

    // Live/Demo mode banner — competitive analysis §1.1 (TradingView
    // colour-codes live vs paper at login). The wizard tints the
    // surrounding label accordingly. The actual modal-background tint
    // is `// TODO(wizard-mode-banner)` for now; the badge surfaces
    // the colour signal.
    let env_color = match controller.config.ctrader_environment {
        CTraderEnvironment::Live => theme::DANGER,
        CTraderEnvironment::Demo => theme::TEXT_MUTED,
    };
    let env_label = match controller.config.ctrader_environment {
        CTraderEnvironment::Live => "LIVE",
        CTraderEnvironment::Demo => "DEMO",
    };
    ui.horizontal(|ui| {
        ui.label("cTrader environment:");
        if ui
            .selectable_label(
                controller.config.ctrader_environment == CTraderEnvironment::Demo,
                "Demo",
            )
            .clicked()
        {
            controller.config.ctrader_environment = CTraderEnvironment::Demo;
        }
        if ui
            .selectable_label(
                controller.config.ctrader_environment == CTraderEnvironment::Live,
                "Live",
            )
            .clicked()
        {
            controller.config.ctrader_environment = CTraderEnvironment::Live;
        }
        ui.label(egui::RichText::new(env_label).color(env_color).strong());
    });

    ui.separator();
    ui.label(
        egui::RichText::new("4.1 Register your application at openapi.ctrader.com")
            .strong()
            .color(theme::TEXT_PRIMARY),
    );
    ui.label(
        egui::RichText::new(
            "Sign in to your cTID → Applications → Add Application. Set redirect URI to \
             http://127.0.0.1:7777/ctrader/callback (or 7878 / 8989). Paste the IDs below.",
        )
        .color(theme::TEXT_MUTED)
        .size(theme::FONT_CAPTION),
    );

    let mut client_id = controller.config.ctrader_client_id.clone().unwrap_or_default();
    ui.horizontal(|ui| {
        ui.label("Client ID:");
        if ui
            .add(egui::TextEdit::singleline(&mut client_id).desired_width(320.0))
            .changed()
        {
            controller.config.ctrader_client_id = if client_id.trim().is_empty() {
                None
            } else {
                Some(client_id.clone())
            };
        }
    });

    // We do NOT persist the actual client secret on the controller —
    // only a boolean "user filled it in". The real secret would be
    // wrapped via `secrecy::SecretString` once Step 10 Apply fires.
    // Spec §11 #4 — no half-written secrets.
    let mut secret_placeholder = if controller.config.ctrader_client_secret_set {
        "•••••••••".to_string()
    } else {
        String::new()
    };
    ui.horizontal(|ui| {
        ui.label("Client Secret:");
        let response = ui.add(
            egui::TextEdit::singleline(&mut secret_placeholder)
                .password(true)
                .desired_width(320.0),
        );
        if response.changed() {
            controller.config.ctrader_client_secret_set =
                !secret_placeholder.trim().is_empty();
        }
    });

    ui.separator();
    ui.label(
        egui::RichText::new("4.2 Sign in with cTID")
            .strong()
            .color(theme::TEXT_PRIMARY),
    );
    ui.label(
        egui::RichText::new(format!(
            "The wizard will bind a loopback listener on the first free port of {:?} and open \
             the system browser. Callback timeout: {} s.",
            WIZARD_DEFAULT_OAUTH_LOOPBACK_PORTS, WIZARD_DEFAULT_OAUTH_CALLBACK_TIMEOUT_SECONDS
        ))
        .color(theme::TEXT_MUTED)
        .size(theme::FONT_CAPTION),
    );

    if ui.button("Sign in with cTID").clicked() {
        // TODO(wizard-oauth-runtime): wire this button to
        // `ProductionCTraderLiveAuthBackend::authorize` and update
        // `controller.config.selected_ctid_trader_account_id` on
        // success. The state-machine plumbing (request / response /
        // CSRF mismatch surface) lives in the controller; this button
        // is a hook point.
    }

    ui.separator();
    ui.label(
        egui::RichText::new("4.3 / 4.4 Account picker + auth probe")
            .strong()
            .color(theme::TEXT_PRIMARY),
    );
    if let Some(account_id) = controller.config.selected_ctid_trader_account_id {
        ui.label(format!(
            "Primary account: #{} ({})",
            account_id,
            controller.config.ctrader_environment.as_str()
        ));
    } else {
        ui.label(
            egui::RichText::new("No account picked yet. Complete 4.2 to populate this list.")
                .color(theme::TEXT_MUTED)
                .size(theme::FONT_CAPTION),
        );
    }

    ui.separator();
    ui.horizontal(|ui| {
        if ui.button("← Back").clicked() {
            result = StepResult::BackRequested;
        }
        if ui.button("Skip cTrader").clicked() {
            result = StepResult::SkipRequested;
        }
        if ui.button("Continue →").clicked() {
            result = StepResult::NextRequested;
        }
    });

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::wizard::{StepResult, WizardController, WizardState};

    #[test]
    fn default_loopback_ports_match_rfc8252_three_port_fallback() {
        assert_eq!(WIZARD_DEFAULT_OAUTH_LOOPBACK_PORTS.len(), 3);
    }

    #[test]
    fn callback_timeout_matches_spec_300_seconds() {
        assert_eq!(WIZARD_DEFAULT_OAUTH_CALLBACK_TIMEOUT_SECONDS, 300);
    }

    #[test]
    fn oauth_step_advances_to_symbols_on_next() {
        let mut c = WizardController::new();
        c.current = WizardState::OAuth;
        c.apply(StepResult::NextRequested);
        assert_eq!(c.current, WizardState::Symbols);
    }

    #[test]
    fn oauth_step_skip_records_under_ctrader_oauth_key() {
        let mut c = WizardController::new();
        c.current = WizardState::OAuth;
        c.apply(StepResult::SkipRequested);
        assert!(c.state_file.skipped_steps.contains(&WizardState::OAuth));
        assert_eq!(c.current, WizardState::Symbols);
    }

    #[test]
    fn oauth_back_returns_to_account_profile() {
        let mut c = WizardController::new();
        c.current = WizardState::OAuth;
        c.apply(StepResult::BackRequested);
        assert_eq!(c.current, WizardState::AccountProfile);
    }
}
