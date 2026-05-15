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

use eframe::egui;

use super::{StepResult, TradingMode, WizardController};
use crate::ui::theme;

/// Default for the telemetry opt-in toggle. Spec §7.1 — "No telemetry"
/// default.
pub const WIZARD_DEFAULT_TELEMETRY_OPT_IN: bool = false;

/// The filename the wizard writes when Apply succeeds. Spec §5.
pub const WIZARD_DEFAULT_COMPLETED_FILENAME: &str = "wizard_state.json";

pub fn render(ui: &mut egui::Ui, controller: &mut WizardController) -> StepResult {
    let mut result = StepResult::StayHere;

    ui.label(
        egui::RichText::new("Review your selections, then click Apply.")
            .color(theme::TEXT_PRIMARY),
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
            ui.label(if controller.config.require_stop_loss { "yes" } else { "no" });
            ui.end_row();

            ui.label("cTrader account");
            ui.label(
                controller
                    .config
                    .selected_ctid_trader_account_id
                    .map(|id| format!("#{} ({})", id, controller.config.ctrader_environment.as_str()))
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
            // TODO(wizard-apply-writer): write config.yaml,
            // broker_credentials.toml, hardware_profile.json,
            // wizard_state.json. Spec §2 Step 10 Actions 1–6.
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
    fn summary_step_advance_marks_finished() {
        let mut c = WizardController::new();
        c.current = WizardState::Summary;
        c.apply(StepResult::NextRequested);
        assert!(c.finished, "advancing past Summary marks the wizard finished");
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
