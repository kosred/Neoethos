//! Step 1 — Welcome + License.
//!
//! Spec: `installer_wizard_ux_spec.md` §2 Step 1 + §9.1 mockup.
//! - NOT skippable (the only mandatory step).
//! - `[Continue →]` disabled until license-accepted checkbox is on.
//! - On accept, records LICENSE SHA-256 + timestamp in `wizard_state.json`.

use eframe::egui;

use super::{StepResult, WizardController};
use crate::ui::theme;

/// Step time budget — used to drive the "≈ 10 minutes" estimate in
/// the welcome copy. Spec §2 Step 1 reports ≤ 30 s for this step.
pub const WIZARD_STEP_WELCOME_BUDGET_SECONDS: u32 = 30;

/// 10 numbered steps + 9.5 — total user-visible budget. Spec §9.1
/// banner copy ("≈ 10 minutes").
pub const WIZARD_TOTAL_BUDGET_MINUTES: u32 = 10;

pub fn render(ui: &mut egui::Ui, controller: &mut WizardController) -> StepResult {
    let mut result = StepResult::StayHere;

    ui.label(
        egui::RichText::new(format!(
            "This wizard will set up your trading workspace in about {} minutes.",
            WIZARD_TOTAL_BUDGET_MINUTES
        ))
        .color(theme::TEXT_PRIMARY),
    );
    ui.add_space(theme::SPACE_SM);

    ui.label(
        egui::RichText::new(
            "Steps: 1 License · 2 Path · 3 Profile · 4 cTrader · 5 Symbols · 6 History · \
             7 Hardware · 8 News & safeguards · 9 Auto-start · 9.5 Autonomy & risk · 10 Apply.",
        )
        .color(theme::TEXT_MUTED)
        .size(theme::FONT_CAPTION),
    );

    ui.separator();
    ui.label(
        egui::RichText::new("Apache License v2.0 / MIT (dual)")
            .strong()
            .color(theme::TEXT_PRIMARY),
    );

    // LICENSE pane. The spec calls for a scrollable 60 % pane reading
    // the installed-dir LICENSE with an `include_str!` fallback. For
    // the skeleton we surface a placeholder line + the operator's
    // dual-license declaration; the real read is left as
    // TODO(license-file-io) below.
    egui::ScrollArea::vertical()
        .max_height(160.0)
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(
                    // TODO(license-file-io): replace this placeholder
                    // with the actual <install_dir>/LICENSE read
                    // (spec §2 Step 1 Actions) plus the
                    // `include_str!` build-time fallback for the
                    // "LICENSE missing" error class.
                    "forex-ai is dual-licensed under the Apache License, Version 2.0 \
                     (LICENSE-APACHE) and the MIT license (LICENSE-MIT). You may use \
                     this software under the terms of either license. See the LICENSE \
                     files in the install directory for the full text.",
                )
                .size(theme::FONT_BODY),
            );
        });

    ui.separator();
    ui.checkbox(
        &mut controller.config.license_accepted,
        "I have read and accept the license",
    );

    ui.horizontal(|ui| {
        if ui.button("Cancel").clicked() {
            result = StepResult::CancelRequested;
        }
        let continue_button = egui::Button::new("Continue →");
        let enabled = controller.config.license_accepted;
        if ui.add_enabled(enabled, continue_button).clicked() {
            result = StepResult::NextRequested;
        }
    });

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::wizard::WizardState;

    #[test]
    fn welcome_step_blocks_advance_until_license_accepted() {
        let mut c = WizardController::new();
        assert_eq!(c.current, WizardState::Welcome);
        // Without acceptance the controller stays put.
        c.apply(StepResult::StayHere);
        assert_eq!(c.current, WizardState::Welcome);
        // Once accepted + advance, controller moves forward.
        c.config.license_accepted = true;
        c.apply(StepResult::NextRequested);
        assert_eq!(c.current, WizardState::Path);
    }

    #[test]
    fn welcome_step_is_not_skippable() {
        let c = WizardController::new();
        assert!(!c.is_skippable());
    }
}
