//! Step 6 — Historical data download.
//!
//! Spec: `installer_wizard_ux_spec.md` §2 Step 6 + §9.4 mockup.
//!
//! Drives `crates/forex-app/src/app_services/ctrader_history.rs::
//! fetch_historical_bars`. The wizard owns the token-bucket gate at
//! 5 req/s per `ctrader_api_full_reference.md` §3.2 ("a maximum of 5
//! requests per second per connection for any historical data
//! requests"). On cancel, the wizard writes a `.partial` sentinel
//! beside the Parquet so the main app can prompt to Resume.

use eframe::egui;

use super::{StepResult, WizardController};
use crate::ui::theme;

/// Operator default — months of history seeded. Spec §2 Step 6
/// "default 6".
pub const WIZARD_DEFAULT_HISTORY_MONTHS: u8 = 6;

/// Allowed slider stops. Spec §9.4 mockup row "1   3   6   12   18   24".
pub const WIZARD_DEFAULT_HISTORY_MONTH_OPTIONS: &[u8] = &[1, 3, 6, 12, 18, 24];

/// Token-bucket rate for historical-data requests.
/// `ctrader_api_full_reference.md` §3.2 — 5 req/s per connection.
pub const WIZARD_DEFAULT_HISTORY_RATE_LIMIT_RPS: u8 = 5;

/// Backoff after `REQUEST_FREQUENCY_EXCEEDED` (108). Spec §3 error
/// matrix — "30 s backoff + resume".
pub const WIZARD_DEFAULT_HISTORY_BACKOFF_SECONDS: u32 = 30;

/// File sentinel suffix indicating an interrupted download. Spec §2
/// Step 6 Actions — "Output … `.partial` (on Cancel)".
pub const WIZARD_PARTIAL_SENTINEL_SUFFIX: &str = ".partial";

/// File sentinel suffix marking a completed download. Spec §2 Step 6
/// Actions — "Output … `.complete`".
pub const WIZARD_COMPLETE_SENTINEL_SUFFIX: &str = ".complete";

pub fn render(ui: &mut egui::Ui, controller: &mut WizardController) -> StepResult {
    let mut result = StepResult::StayHere;

    ui.label(
        egui::RichText::new(format!(
            "Download history for {} symbols × {} timeframes (rate-limited to {} req/s).",
            controller.config.selected_symbols.len(),
            controller.config.selected_timeframes.len(),
            WIZARD_DEFAULT_HISTORY_RATE_LIMIT_RPS,
        ))
        .color(theme::TEXT_PRIMARY),
    );

    ui.add_space(theme::SPACE_SM);

    ui.horizontal(|ui| {
        ui.label("Months of history:");
        for option in WIZARD_DEFAULT_HISTORY_MONTH_OPTIONS {
            if ui
                .selectable_label(
                    controller.config.history_months == *option,
                    format!("{}", option),
                )
                .clicked()
            {
                controller.config.history_months = *option;
            }
        }
    });

    let pair_count =
        controller.config.selected_symbols.len() * controller.config.selected_timeframes.len();
    let estimated_seconds = (pair_count as u32)
        .saturating_mul(controller.config.history_months as u32)
        .div_ceil(WIZARD_DEFAULT_HISTORY_RATE_LIMIT_RPS as u32);
    ui.label(
        egui::RichText::new(format!(
            "Estimated ≈ {} requests, ≈ {} s at {} req/s.",
            pair_count.saturating_mul(controller.config.history_months as usize),
            estimated_seconds,
            WIZARD_DEFAULT_HISTORY_RATE_LIMIT_RPS,
        ))
        .color(theme::TEXT_MUTED)
        .size(theme::FONT_CAPTION),
    );

    ui.separator();
    ui.label(
        egui::RichText::new(
            "Cancel preserves already-downloaded bars (.partial sentinel). \
             No synthetic fill is ever written.",
        )
        .size(theme::FONT_CAPTION)
        .color(theme::WARNING),
    );

    ui.separator();
    ui.horizontal(|ui| {
        if ui.button("← Back").clicked() {
            result = StepResult::BackRequested;
        }
        if ui.button("Begin download").clicked() {
            // TODO(wizard-history-runtime): wire to
            // `app_services::ctrader_history::fetch_historical_bars`
            // with the controller's token bucket. The skeleton
            // surfaces only the UI plumbing.
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
    fn default_history_months_equals_operator_default() {
        assert_eq!(WIZARD_DEFAULT_HISTORY_MONTHS, 6);
        assert!(WIZARD_DEFAULT_HISTORY_MONTH_OPTIONS.contains(&WIZARD_DEFAULT_HISTORY_MONTHS));
    }

    #[test]
    fn rate_limit_matches_ctrader_api_documentation() {
        // ctrader_api_full_reference.md §3.2 — 5 req/s per connection.
        assert_eq!(WIZARD_DEFAULT_HISTORY_RATE_LIMIT_RPS, 5);
    }

    #[test]
    fn historical_step_advances_to_hardware() {
        let mut c = WizardController::new();
        c.current = WizardState::Historical;
        c.apply(StepResult::NextRequested);
        assert_eq!(c.current, WizardState::Hardware);
    }

    #[test]
    fn historical_step_skip_records_under_historical_key() {
        let mut c = WizardController::new();
        c.current = WizardState::Historical;
        c.apply(StepResult::SkipRequested);
        assert!(c.state_file.skipped_steps.contains(&WizardState::Historical));
    }

    #[test]
    fn sentinel_suffixes_are_distinguishable() {
        assert_ne!(WIZARD_PARTIAL_SENTINEL_SUFFIX, WIZARD_COMPLETE_SENTINEL_SUFFIX);
    }
}
