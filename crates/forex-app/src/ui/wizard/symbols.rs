//! Step 5 — Symbol & timeframe defaults + template gallery.
//!
//! Spec: `installer_wizard_ux_spec.md` §2 Step 5 + §9.3 mockup +
//! `wizard_onboarding_competitive_analysis.md` §8.4 (six-template
//! gallery + Custom).
//!
//! OPERATOR INVARIANT (load-bearing): the timeframe list is sourced
//! from `forex_core::contracts::temporal::CANONICAL_TIMEFRAMES` —
//! 11 entries, NO H2. Test below pins this.

use eframe::egui;
use forex_core::contracts::temporal::CANONICAL_TIMEFRAMES;

use super::{StepResult, SymbolTemplate, WizardController};
use crate::ui::theme;

/// Default symbols pre-selected when the user hasn't picked a template.
/// Spec §9.3 mockup: "Default selection EURUSD".
pub const WIZARD_DEFAULT_SYMBOLS: &[&str] = &["EURUSD"];

/// Default timeframes pre-selected. Spec §9.3: "Default selection
/// M5, M15, H1, H4, D1".
pub const WIZARD_DEFAULT_TIMEFRAMES: &[&str] = &["M5", "M15", "H1", "H4", "D1"];

/// Top-28 majors preset — competitive analysis §8.4 / spec §9.3
/// "Preset: Top 28 ▼".  Conservative subset; the actual top-28 list
/// is `// TODO(symbol-universe-canon)` until the operator pins it.
pub const WIZARD_DEFAULT_TOP_28_PRESET: &[&str] = &[
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    "NZDJPY", "NZDCHF", "NZDCAD",
    "CADJPY", "CADCHF", "CHFJPY",
];

/// Top-7 majors used by the Scalping-majors-M5 template.
pub const WIZARD_DEFAULT_TOP_7_MAJORS: &[&str] = &[
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
];

pub fn render(ui: &mut egui::Ui, controller: &mut WizardController) -> StepResult {
    let mut result = StepResult::StayHere;

    ui.label(
        egui::RichText::new(
            "Pick symbols and timeframes to seed the local cache. \
             You can change these any time.",
        )
        .color(theme::TEXT_PRIMARY),
    );
    ui.add_space(theme::SPACE_SM);

    ui.label(
        egui::RichText::new("Strategy template")
            .strong()
            .color(theme::TEXT_PRIMARY),
    );

    for template in SymbolTemplate::all() {
        let selected = controller.config.selected_template == *template;
        if ui
            .selectable_label(
                selected,
                format!(
                    "{} (risk {} / 10)",
                    template.label(),
                    template.risk_score()
                ),
            )
            .clicked()
        {
            controller.config.selected_template = *template;
            apply_template(*template, controller);
        }
    }

    ui.separator();
    ui.label(
        egui::RichText::new("Symbols")
            .strong()
            .color(theme::TEXT_PRIMARY),
    );
    if ui.button("Apply Top 28 majors").clicked() {
        controller.config.selected_symbols =
            WIZARD_DEFAULT_TOP_28_PRESET.iter().map(|s| (*s).to_string()).collect();
        controller.config.selected_template = SymbolTemplate::Custom;
    }

    // Symbol multi-select. The real wizard populates this from
    // `ProtoOASymbolsListReq` (2114); the skeleton uses the static
    // canonical list above.
    egui::ScrollArea::vertical()
        .max_height(160.0)
        .id_salt("wizard_symbols_scroll")
        .show(ui, |ui| {
            for symbol in WIZARD_DEFAULT_TOP_28_PRESET {
                let mut on = controller
                    .config
                    .selected_symbols
                    .iter()
                    .any(|s| s == *symbol);
                if ui.checkbox(&mut on, *symbol).changed() {
                    if on {
                        controller
                            .config
                            .selected_symbols
                            .push((*symbol).to_string());
                    } else {
                        controller
                            .config
                            .selected_symbols
                            .retain(|s| s != *symbol);
                    }
                }
            }
        });

    ui.separator();
    ui.label(
        egui::RichText::new("Timeframes (11 canonical — NO H2 per operator policy)")
            .strong()
            .color(theme::TEXT_PRIMARY),
    );
    for tf in CANONICAL_TIMEFRAMES {
        let mut on = controller
            .config
            .selected_timeframes
            .iter()
            .any(|s| s == *tf);
        if ui.checkbox(&mut on, *tf).changed() {
            if on {
                controller
                    .config
                    .selected_timeframes
                    .push((*tf).to_string());
            } else {
                controller.config.selected_timeframes.retain(|s| s != *tf);
            }
        }
    }

    ui.add_space(theme::SPACE_SM);
    let pair_count =
        controller.config.selected_symbols.len() * controller.config.selected_timeframes.len();
    ui.label(
        egui::RichText::new(format!(
            "{} symbols × {} tfs = {} pairs",
            controller.config.selected_symbols.len(),
            controller.config.selected_timeframes.len(),
            pair_count
        ))
        .color(theme::TEXT_MUTED)
        .size(theme::FONT_CAPTION),
    );

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

/// Apply a template's symbol/timeframe defaults to the controller.
/// Competitive analysis §8.4.
pub fn apply_template(template: SymbolTemplate, controller: &mut WizardController) {
    match template {
        SymbolTemplate::ScalpingEurusdM1 => {
            controller.config.selected_symbols = vec!["EURUSD".to_string()];
            controller.config.selected_timeframes = vec!["M1", "M3", "M5"]
                .into_iter()
                .map(String::from)
                .collect();
        }
        SymbolTemplate::ScalpingMajorsM5 => {
            controller.config.selected_symbols = WIZARD_DEFAULT_TOP_7_MAJORS
                .iter()
                .map(|s| (*s).to_string())
                .collect();
            controller.config.selected_timeframes = vec!["M5", "M15"]
                .into_iter()
                .map(String::from)
                .collect();
        }
        SymbolTemplate::SwingD1Majors => {
            controller.config.selected_symbols = WIZARD_DEFAULT_TOP_7_MAJORS
                .iter()
                .map(|s| (*s).to_string())
                .collect();
            controller.config.selected_timeframes = vec!["H4", "D1", "W1"]
                .into_iter()
                .map(String::from)
                .collect();
        }
        SymbolTemplate::TrendH1Baskets => {
            controller.config.selected_symbols = WIZARD_DEFAULT_TOP_28_PRESET
                .iter()
                .map(|s| (*s).to_string())
                .collect();
            controller.config.selected_timeframes = vec!["H1", "H4"]
                .into_iter()
                .map(String::from)
                .collect();
        }
        SymbolTemplate::MeanReversionH1Majors => {
            controller.config.selected_symbols = WIZARD_DEFAULT_TOP_7_MAJORS
                .iter()
                .map(|s| (*s).to_string())
                .collect();
            controller.config.selected_timeframes = vec!["M30", "H1", "H4"]
                .into_iter()
                .map(String::from)
                .collect();
        }
        SymbolTemplate::Custom => {
            // No-op — preserve whatever the user picked manually.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::wizard::{StepResult, WizardController, WizardState};

    #[test]
    fn canonical_timeframes_have_eleven_entries_no_h2() {
        assert_eq!(CANONICAL_TIMEFRAMES.len(), 11);
        assert!(
            !CANONICAL_TIMEFRAMES.iter().any(|t| *t == "H2"),
            "H2 forbidden per operator policy"
        );
    }

    #[test]
    fn default_timeframes_subset_of_canonical_and_no_h2() {
        for tf in WIZARD_DEFAULT_TIMEFRAMES {
            assert!(
                CANONICAL_TIMEFRAMES.contains(tf),
                "{} must be canonical",
                tf
            );
            assert_ne!(*tf, "H2", "H2 forbidden");
        }
    }

    #[test]
    fn template_gallery_has_six_entries() {
        assert_eq!(SymbolTemplate::all().len(), 6);
    }

    #[test]
    fn scalping_eurusd_m1_template_uses_lowest_three_tfs() {
        let mut c = WizardController::new();
        apply_template(SymbolTemplate::ScalpingEurusdM1, &mut c);
        assert_eq!(c.config.selected_symbols, vec!["EURUSD".to_string()]);
        assert_eq!(c.config.selected_timeframes, vec!["M1", "M3", "M5"]);
    }

    #[test]
    fn swing_d1_majors_template_includes_w1() {
        let mut c = WizardController::new();
        apply_template(SymbolTemplate::SwingD1Majors, &mut c);
        assert!(c.config.selected_timeframes.contains(&"W1".to_string()));
        assert!(!c.config.selected_timeframes.contains(&"H2".to_string()));
    }

    #[test]
    fn symbols_step_advances_to_historical() {
        let mut c = WizardController::new();
        c.current = WizardState::Symbols;
        c.apply(StepResult::NextRequested);
        assert_eq!(c.current, WizardState::Historical);
    }

    #[test]
    fn no_template_ever_includes_h2_timeframe() {
        let mut c = WizardController::new();
        for template in SymbolTemplate::all() {
            apply_template(*template, &mut c);
            assert!(
                !c.config.selected_timeframes.iter().any(|t| t == "H2"),
                "template {:?} introduced H2",
                template
            );
        }
    }
}
