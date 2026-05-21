//! Renderer for the cTrader 14-step connection state machine — P8.
//!
//! Shows each step on its own row with a status glyph, name, latest
//! message, request_id (when present), and a Retry button on failed
//! steps that surfaces the configured retry hint.
//!
//! Drop this into the BrokerSetup page (or any panel) by passing the
//! state-machine instance:
//!
//! ```ignore
//! ctrader_state_view::render(ui, &mut state.ctrader_sm);
//! ```

use eframe::egui;

use crate::app_services::ctrader_state_machine::{CTraderStateMachine, CTraderStepStatus};
use crate::ui::theme;

pub fn render(ui: &mut egui::Ui, sm: &mut CTraderStateMachine) {
    ui.add_space(theme::SPACE_MD);
    ui.label(
        egui::RichText::new("cTrader connection")
            .size(theme::FONT_SUBTITLE)
            .strong()
            .color(theme::TEXT_PRIMARY),
    );
    ui.add_space(theme::SPACE_XS);

    let current = sm.current_step();
    let summary = if sm.is_fully_connected() {
        ("● Connected".to_string(), theme::SUCCESS)
    } else if let Some(idx) = current {
        (format!("● Stuck on step {}/14", idx), theme::WARNING)
    } else {
        ("○ Not started".to_string(), theme::TEXT_MUTED)
    };
    ui.label(
        egui::RichText::new(summary.0)
            .size(theme::FONT_BODY)
            .color(summary.1),
    );
    ui.add_space(theme::SPACE_SM);

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .max_height(360.0)
        .show(ui, |ui| {
            for step in &sm.steps {
                render_one_step(ui, step);
            }
        });

    ui.add_space(theme::SPACE_SM);
    ui.horizontal(|ui| {
        if theme::small_button(ui, "Reset state", theme::ButtonKind::Secondary).clicked() {
            sm.reset();
        }
    });
}

fn render_one_step(
    ui: &mut egui::Ui,
    step: &crate::app_services::ctrader_state_machine::CTraderStep,
) {
    let (glyph_color, label_color) = match step.status {
        CTraderStepStatus::Pending => (theme::TEXT_FAINT, theme::TEXT_MUTED),
        CTraderStepStatus::InFlight => (theme::ACCENT, theme::TEXT_PRIMARY),
        CTraderStepStatus::Ok => (theme::SUCCESS, theme::TEXT_PRIMARY),
        CTraderStepStatus::Failed => (theme::DANGER, theme::DANGER),
        CTraderStepStatus::Skipped => (theme::TEXT_FAINT, theme::TEXT_FAINT),
    };
    egui::Frame::new()
        .inner_margin(egui::Margin::symmetric(theme::SPACE_SM as i8, 4))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(step.status.glyph())
                        .size(theme::FONT_SUBTITLE)
                        .color(glyph_color)
                        .strong(),
                );
                ui.label(
                    egui::RichText::new(format!("{:>2}.", step.index))
                        .size(theme::FONT_CAPTION)
                        .color(theme::TEXT_FAINT)
                        .monospace(),
                );
                ui.label(
                    egui::RichText::new(&step.name)
                        .size(theme::FONT_BODY)
                        .color(label_color),
                );
                if let Some(rid) = &step.request_id {
                    ui.label(
                        egui::RichText::new(format!("[req={}]", rid))
                            .size(theme::FONT_CAPTION)
                            .color(theme::TEXT_FAINT)
                            .monospace(),
                    );
                }
            });
            if let Some(message) = &step.message {
                ui.label(
                    egui::RichText::new(format!("    {}", message))
                        .size(theme::FONT_CAPTION)
                        .color(if matches!(step.status, CTraderStepStatus::Failed) {
                            theme::DANGER
                        } else {
                            theme::TEXT_MUTED
                        }),
                );
            }
            if let Some(hint) = &step.retry_hint {
                ui.label(
                    egui::RichText::new(format!("    → fix: {}", hint))
                        .size(theme::FONT_CAPTION)
                        .color(theme::WARNING),
                );
            }
        });
}
