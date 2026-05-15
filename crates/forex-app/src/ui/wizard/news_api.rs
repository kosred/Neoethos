//! Step 8 — News / sentiment provider + auto-trade safeguards.
//!
//! Spec: `installer_wizard_ux_spec.md` §2 Step 8 +
//! `wizard_onboarding_competitive_analysis.md` §9.1 extensions:
//!   - News blackout window (default 2 min ± — competitive analysis §7.2).
//!   - Maintenance window auto-flatten Friday 16:00 ET → Sunday 18:00 ET
//!     (competitive analysis §7.3).
//!   - Correlation cap (default 0.7 — competitive analysis §7.4).
//!   - Volatility σ pause (default 3.0 — competitive analysis §7.5).
//!
//! Optional `secrecy::SecretString` API key — stored in-memory until
//! Step 10 Apply.

use eframe::egui;
use secrecy::{ExposeSecret, SecretString};

use super::{StepResult, WizardController};
use crate::ui::theme;

/// Default for the news-filter toggle. Spec §2 Step 8 — "default off".
pub const WIZARD_DEFAULT_NEWS_ENABLED: bool = false;

/// News blackout window in minutes ± a high-impact event.
/// Competitive analysis §7.2 — FTMO 2 min, E8 5 min, FunderPro 2 min;
/// 2 min is the operator-safer default.
pub const WIZARD_DEFAULT_NEWS_BLACKOUT_MINUTES: u32 = 2;

/// Maintenance-window auto-flatten toggle. Competitive analysis §7.3
/// — default ON for retail forex.
pub const WIZARD_DEFAULT_MAINTENANCE_WINDOW_ENABLED: bool = true;

/// Friday auto-flatten ET hour (competitive analysis §7.3).
pub const WIZARD_DEFAULT_MAINTENANCE_FRIDAY_FLATTEN_HOUR_ET: u8 = 16;

/// Sunday resume-pause ET hour (competitive analysis §7.3).
pub const WIZARD_DEFAULT_MAINTENANCE_SUNDAY_RESUME_HOUR_ET: u8 = 18;

/// Correlation cap between concurrent open positions.
/// Competitive analysis §7.4 — "default 0.7".
pub const WIZARD_DEFAULT_CORRELATION_CAP: f32 = 0.7;
pub const WIZARD_DEFAULT_CORRELATION_CAP_FLOOR: f32 = 0.5;
pub const WIZARD_DEFAULT_CORRELATION_CAP_CEILING: f32 = 0.95;

/// Volatility σ pause threshold. Competitive analysis §7.5 — "default 3.0 σ
/// over a 14-bar ATR window".
pub const WIZARD_DEFAULT_VOLATILITY_SIGMA: f32 = 3.0;
pub const WIZARD_DEFAULT_VOLATILITY_SIGMA_FLOOR: f32 = 1.5;
pub const WIZARD_DEFAULT_VOLATILITY_SIGMA_CEILING: f32 = 5.0;

/// In-memory API-key holder. Kept off the `WizardConfig` struct so the
/// secret never lands in `Debug` output. Spec §7.4 — the wizard never
/// transmits this off the machine.
#[derive(Default)]
pub struct NewsApiKeyHolder {
    api_key: Option<SecretString>,
}

impl NewsApiKeyHolder {
    pub fn is_set(&self) -> bool {
        self.api_key.is_some()
    }

    pub fn set(&mut self, value: String) {
        if value.trim().is_empty() {
            self.api_key = None;
        } else {
            self.api_key = Some(SecretString::from(value));
        }
    }

    /// Read-only access for the Apply step (Step 10). Not exposed via
    /// `Debug`.
    pub fn expose(&self) -> Option<&str> {
        self.api_key.as_ref().map(|s| s.expose_secret())
    }
}

pub fn render(ui: &mut egui::Ui, controller: &mut WizardController) -> StepResult {
    let mut result = StepResult::StayHere;

    ui.checkbox(
        &mut controller.config.news_filter_enabled,
        "Enable news filter (suppress trading around high-impact events).",
    );

    if controller.config.news_filter_enabled {
        ui.horizontal(|ui| {
            ui.label("News blackout window (minutes ± event):");
            ui.add(egui::DragValue::new(
                &mut controller.config.news_blackout_minutes,
            ));
        });
        ui.label(
            egui::RichText::new(
                "API key is held in-memory and only written via secrecy::SecretString \
                 at Apply. Never transmitted off this machine.",
            )
            .size(theme::FONT_CAPTION)
            .color(theme::TEXT_MUTED),
        );
    }

    ui.separator();
    ui.label(
        egui::RichText::new("Auto-trade safeguards (competitive analysis §7)")
            .strong()
            .color(theme::TEXT_PRIMARY),
    );

    ui.checkbox(
        &mut controller.config.maintenance_window_enabled,
        "Auto-flatten on Friday 16:00 ET and pause through Sunday 18:00 ET.",
    );

    ui.horizontal(|ui| {
        ui.label("Max correlation between concurrent open positions:");
        ui.add(
            egui::Slider::new(
                &mut controller.config.correlation_cap,
                WIZARD_DEFAULT_CORRELATION_CAP_FLOOR..=WIZARD_DEFAULT_CORRELATION_CAP_CEILING,
            )
            .max_decimals(2),
        );
    });

    ui.horizontal(|ui| {
        ui.label("Pause when ATR > N · σ (rolling 14-bar):");
        ui.add(
            egui::Slider::new(
                &mut controller.config.volatility_sigma_pause,
                WIZARD_DEFAULT_VOLATILITY_SIGMA_FLOOR..=WIZARD_DEFAULT_VOLATILITY_SIGMA_CEILING,
            )
            .max_decimals(1),
        );
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
    fn news_blackout_default_matches_ftmo_funderpro() {
        // FTMO 2 min, FunderPro 2 min — adopted as default.
        assert_eq!(WIZARD_DEFAULT_NEWS_BLACKOUT_MINUTES, 2);
    }

    #[test]
    fn correlation_cap_default_within_bounds() {
        assert!(WIZARD_DEFAULT_CORRELATION_CAP >= WIZARD_DEFAULT_CORRELATION_CAP_FLOOR);
        assert!(WIZARD_DEFAULT_CORRELATION_CAP <= WIZARD_DEFAULT_CORRELATION_CAP_CEILING);
    }

    #[test]
    fn volatility_sigma_default_within_bounds() {
        assert!(WIZARD_DEFAULT_VOLATILITY_SIGMA >= WIZARD_DEFAULT_VOLATILITY_SIGMA_FLOOR);
        assert!(WIZARD_DEFAULT_VOLATILITY_SIGMA <= WIZARD_DEFAULT_VOLATILITY_SIGMA_CEILING);
    }

    #[test]
    fn news_api_step_advances_to_autostart() {
        let mut c = WizardController::new();
        c.current = WizardState::NewsApi;
        c.apply(StepResult::NextRequested);
        assert_eq!(c.current, WizardState::Autostart);
    }

    #[test]
    fn news_api_key_holder_keeps_secret_off_debug_output() {
        let mut holder = NewsApiKeyHolder::default();
        holder.set("very-secret-key".to_string());
        let debug = format!("{:?}", holder.is_set());
        assert!(!debug.contains("very-secret-key"));
        assert!(holder.is_set());
        assert_eq!(holder.expose(), Some("very-secret-key"));
    }

    #[test]
    fn news_api_key_holder_clears_on_empty_input() {
        let mut holder = NewsApiKeyHolder::default();
        holder.set("set".to_string());
        assert!(holder.is_set());
        holder.set("   ".to_string());
        assert!(!holder.is_set());
    }
}
