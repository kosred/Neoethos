//! Step 3 — Account & profile (extended per competitive analysis §9.1).
//!
//! Spec: `installer_wizard_ux_spec.md` §2 Step 3 +
//! `wizard_onboarding_competitive_analysis.md` §9.1 additions:
//!   - Risk-profile slider 1–10 (default 4) — competitive analysis §8.3.
//!   - SL-required toggle — competitive analysis §7 must-have row.
//!   - Per-trade max risk slider 0.1–2.0 % — competitive analysis §7.
//!   - Daily-loss reset timezone — competitive analysis §4.1.
//!   - Interface mode Beginner / Advanced — competitive analysis §3.2.
//!
//! Operator-locked invariants enforced here:
//! - Monthly profit target slider **floor** is 4 % per
//!   `PropFirmConstraints::FTMO_STANDARD.min_monthly_net_profit_pct`.
//!   Typing 3 % surfaces the validator from spec §3 error matrix.

use eframe::egui;
use forex_core::domain::prop_firm::PropFirmConstraints;

use super::{InterfaceMode, StepResult, TradingMode, WizardController};
use crate::ui::theme;

// ─── Operator-locked policy constants ──────────────────────────────
//
// Reviewers can grep `WIZARD_DEFAULT_` to audit operator policy from
// a single sweep.

/// Monthly profit target *floor* — operator directive 2026-05-14 (verbatim
/// at `forex-core/src/domain/prop_firm.rs:36`). The slider cannot go
/// below this.
pub const WIZARD_DEFAULT_MONTHLY_PROFIT_FLOOR: f32 =
    PropFirmConstraints::FTMO_STANDARD.min_monthly_net_profit_pct;

/// Monthly profit target slider default. Equal to the floor — the
/// wizard does NOT pre-suggest a higher target (no over-promising).
pub const WIZARD_DEFAULT_MONTHLY_PROFIT_TARGET: f32 = WIZARD_DEFAULT_MONTHLY_PROFIT_FLOOR;

/// Monthly profit target slider upper bound. Spec §2 Step 3 4-row
/// mockup: "slider 4 %–25 %".
pub const WIZARD_DEFAULT_MONTHLY_PROFIT_CEILING: f32 = 0.25;

/// Default prop-firm preset. Only FTMO Standard is operator-approved
/// today (spec §10.6 — "Aggressive" hidden until constants exist).
pub const WIZARD_DEFAULT_PROP_FIRM_PRESET: &str = "FTMO_STANDARD";

/// Risk-profile slider default position (1–10) — competitive analysis
/// §8.3 table row "4 (moderate)".
pub const WIZARD_DEFAULT_RISK_PROFILE: u8 = 4;

/// Lowest slider position the wizard exposes. 1 = conservative
/// (competitive analysis §8.3).
pub const WIZARD_DEFAULT_RISK_PROFILE_MIN: u8 = 1;

/// Highest slider position. 10 = Risky Mode unlock. Owned by the
/// Risky Mode agent (see TODO below).
pub const WIZARD_DEFAULT_RISK_PROFILE_MAX: u8 = 10;

/// Default for SL-required toggle (competitive analysis §7 must-have).
/// On for FTMO presets.
pub const WIZARD_DEFAULT_REQUIRE_SL: bool = true;

/// Default per-trade max risk %, competitive analysis §7
/// "Per-trade max loss". The slider's *floor* is operator-policy
/// (0.1 %); the ceiling is conservative for the default profile.
pub const WIZARD_DEFAULT_PER_TRADE_RISK_PCT: f32 = 0.75;
pub const WIZARD_DEFAULT_PER_TRADE_RISK_FLOOR: f32 = 0.1;
pub const WIZARD_DEFAULT_PER_TRADE_RISK_CEILING: f32 = 2.0;

/// Daily-loss reset timezone — FTMO convention is CE(S)T
/// (competitive analysis §4.1). Stored as IANA TZ string.
pub const WIZARD_DEFAULT_DLL_RESET_TZ: &str = "Europe/Prague";

pub fn render(ui: &mut egui::Ui, controller: &mut WizardController) -> StepResult {
    let mut result = StepResult::StayHere;

    ui.label(
        egui::RichText::new("Tell forex-ai a bit about how you want to trade.")
            .color(theme::TEXT_PRIMARY),
    );
    ui.add_space(theme::SPACE_SM);

    // Operator name.
    let mut name = controller
        .config
        .operator_name
        .clone()
        .unwrap_or_default();
    ui.horizontal(|ui| {
        ui.label("Operator name (optional):");
        if ui
            .text_edit_singleline(&mut name)
            .on_hover_text("Used as the journal tag. Stored on this machine only.")
            .changed()
        {
            controller.config.operator_name = if name.trim().is_empty() {
                None
            } else {
                Some(name.clone())
            };
        }
    });

    // Prop-firm preset — FTMO Standard only for now (spec §10.6).
    ui.horizontal(|ui| {
        ui.label("Prop-firm preset:");
        egui::ComboBox::from_id_salt("wizard_prop_firm_preset")
            .selected_text(&controller.config.prop_firm_preset)
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut controller.config.prop_firm_preset,
                    WIZARD_DEFAULT_PROP_FIRM_PRESET.to_string(),
                    "FTMO Standard (recommended)",
                );
            });
    });

    // Monthly profit target slider with operator floor.
    ui.horizontal(|ui| {
        ui.label("Monthly net profit target:");
        let mut pct = controller.config.monthly_profit_target * 100.0;
        let response = ui.add(
            egui::Slider::new(
                &mut pct,
                (WIZARD_DEFAULT_MONTHLY_PROFIT_FLOOR * 100.0)
                    ..=(WIZARD_DEFAULT_MONTHLY_PROFIT_CEILING * 100.0),
            )
            .suffix(" %"),
        );
        if response.changed() {
            let new_target = pct / 100.0;
            controller.config.monthly_profit_target =
                new_target.max(WIZARD_DEFAULT_MONTHLY_PROFIT_FLOOR);
        }
        ui.label(
            egui::RichText::new(format!(
                "(floor {:.0}% per operator policy 2026-05-14)",
                WIZARD_DEFAULT_MONTHLY_PROFIT_FLOOR * 100.0
            ))
            .size(theme::FONT_CAPTION)
            .color(theme::TEXT_MUTED),
        );
    });

    // Trading mode.
    ui.horizontal(|ui| {
        ui.label("Trading mode:");
        ui.selectable_value(
            &mut controller.config.trading_mode,
            TradingMode::Backtest,
            "Backtest",
        );
        ui.selectable_value(
            &mut controller.config.trading_mode,
            TradingMode::Forward,
            "Forward test",
        );
        ui.selectable_value(
            &mut controller.config.trading_mode,
            TradingMode::Live,
            "Live",
        );
    });

    // Interface mode — competitive analysis §3.2.
    ui.horizontal(|ui| {
        ui.label("Interface mode:");
        ui.selectable_value(
            &mut controller.config.interface_mode,
            InterfaceMode::Beginner,
            "Beginner",
        );
        ui.selectable_value(
            &mut controller.config.interface_mode,
            InterfaceMode::Advanced,
            "Advanced",
        );
    });

    ui.separator();
    ui.label(
        egui::RichText::new("Risk-profile & safeguards")
            .strong()
            .color(theme::TEXT_PRIMARY),
    );

    // Risk-profile slider 1–10 — competitive analysis §8.3.
    ui.horizontal(|ui| {
        ui.label("Risk profile (1–10):");
        ui.add(
            egui::Slider::new(
                &mut controller.config.risk_profile_slider,
                WIZARD_DEFAULT_RISK_PROFILE_MIN..=WIZARD_DEFAULT_RISK_PROFILE_MAX,
            ),
        );
        if controller.config.risk_profile_slider == WIZARD_DEFAULT_RISK_PROFILE_MAX {
            ui.label(
                egui::RichText::new("Risky Mode")
                    .color(theme::DANGER)
                    .strong(),
            );
            // Risky Mode unlock — research §8.2 wizard branch panel.
            // The actual `RiskyModeConfig` is constructed by the
            // Step 10 Apply path from `RiskyModeConfig::default()`
            // (research §4.1 — $20 → $50_000 with paper-trading
            // default ON). Until summary.rs::apply is wired by the
            // wizard-apply-writer agent (Phase 2B), this label is
            // the operator-facing signal that the slider has armed
            // the mode.
            //
            // FIXME(risky-mode-apply): summary.rs needs session
            // access — gated on Agent B wizard apply writer landing.
            // The Apply writer should call
            // `session.enable_risky_mode(RiskyModeConfig::default(),
            // starting_bankroll)` where `starting_bankroll` comes
            // from the broker-reported balance at Apply time, or
            // `RiskyModeConfig::default().starting_capital_usd` ($20)
            // when no broker is connected yet.
        }
    });

    // SL-required toggle.
    ui.checkbox(
        &mut controller.config.require_stop_loss,
        "Require Stop Loss on every order (recommended for FTMO).",
    );

    // Per-trade max risk slider.
    ui.horizontal(|ui| {
        ui.label("Per-trade max risk:");
        ui.add(
            egui::Slider::new(
                &mut controller.config.per_trade_max_risk_pct,
                WIZARD_DEFAULT_PER_TRADE_RISK_FLOOR..=WIZARD_DEFAULT_PER_TRADE_RISK_CEILING,
            )
            .suffix(" %"),
        );
    });

    // Daily-loss reset timezone.
    ui.horizontal(|ui| {
        ui.label("Daily-loss reset timezone:");
        ui.text_edit_singleline(&mut controller.config.daily_loss_reset_timezone)
            .on_hover_text("IANA TZ string. FTMO convention: Europe/Prague (CE(S)T).");
    });

    ui.separator();
    ui.horizontal(|ui| {
        if ui.button("← Back").clicked() {
            result = StepResult::BackRequested;
        }
        if ui.button("Skip").clicked() {
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
    fn monthly_profit_floor_matches_operator_constant() {
        // Operator invariant — the floor is sourced from prop_firm.rs.
        assert!(
            (WIZARD_DEFAULT_MONTHLY_PROFIT_FLOOR
                - PropFirmConstraints::FTMO_STANDARD.min_monthly_net_profit_pct)
                .abs()
                < f32::EPSILON
        );
        assert!((WIZARD_DEFAULT_MONTHLY_PROFIT_FLOOR - 0.04f32).abs() < f32::EPSILON);
    }

    #[test]
    fn risk_profile_slider_bounds_are_one_to_ten() {
        assert_eq!(WIZARD_DEFAULT_RISK_PROFILE_MIN, 1);
        assert_eq!(WIZARD_DEFAULT_RISK_PROFILE_MAX, 10);
        assert!(WIZARD_DEFAULT_RISK_PROFILE >= WIZARD_DEFAULT_RISK_PROFILE_MIN);
        assert!(WIZARD_DEFAULT_RISK_PROFILE <= WIZARD_DEFAULT_RISK_PROFILE_MAX);
    }

    #[test]
    fn account_profile_advances_on_next() {
        let mut c = WizardController::new();
        c.current = WizardState::AccountProfile;
        c.apply(StepResult::NextRequested);
        assert_eq!(c.current, WizardState::OAuth);
    }

    #[test]
    fn account_profile_back_goes_to_path() {
        let mut c = WizardController::new();
        c.current = WizardState::AccountProfile;
        c.apply(StepResult::BackRequested);
        assert_eq!(c.current, WizardState::Path);
    }

    #[test]
    fn default_per_trade_risk_within_floor_and_ceiling() {
        assert!(WIZARD_DEFAULT_PER_TRADE_RISK_PCT >= WIZARD_DEFAULT_PER_TRADE_RISK_FLOOR);
        assert!(WIZARD_DEFAULT_PER_TRADE_RISK_PCT <= WIZARD_DEFAULT_PER_TRADE_RISK_CEILING);
    }

    #[test]
    fn require_stop_loss_default_is_on_for_ftmo() {
        assert!(WIZARD_DEFAULT_REQUIRE_SL);
    }
}
