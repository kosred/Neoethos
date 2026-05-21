//! Welcome / onboarding view — replaces the 11-step modal wizard with a
//! non-blocking checklist of task cards.
//!
//! Design (task #8):
//! - The Welcome tab shows 4 cards that route the operator into the
//!   real working tabs (BrokerSetup / DataBootstrap / Risk / Training).
//! - Cards are read-only routers: the heavy logic (OAuth, file pickers,
//!   risk-cap forms, training pipeline) stays in the destination tabs.
//! - Status (☐ / ✓) is derived live from existing state — no separate
//!   onboarding state file to drift out of sync.
//! - Once everything is set up the cards turn into a quiet "All set ✓"
//!   banner; the tab stays accessible from the sidebar so the operator
//!   can revisit if anything regresses.
//!
//! Future Flutter migration: each card is a pure data → button mapping,
//! so the pattern translates 1:1 to any Flutter list / grid layout.

use crate::app_services::trading::TradingSession;
use crate::app_state::AppState;
use crate::ui::theme;
use crate::workspace::WorkspaceTab;
use eframe::egui;

/// One row of the onboarding checklist.
struct Card {
    title: &'static str,
    description: &'static str,
    target_tab: WorkspaceTab,
    done: bool,
    cta: &'static str,
}

/// Result of rendering the Welcome view. The caller (`WorkspaceViewer`)
/// translates `requested_tab` into a `WorkspaceState::focus_tab` call so
/// the dock actually switches.
pub struct WelcomeAction {
    pub requested_tab: Option<WorkspaceTab>,
}

/// Decide whether each onboarding step is "done" by reading existing
/// state. Keeping these inline (instead of a struct method) makes the
/// checks greppable and the dependencies obvious.
fn build_cards(state: &AppState, session: &TradingSession) -> [Card; 4] {
    // Card 1 — broker. Considered done once the session reports a live
    // connection. We use `is_connected` because that's the runtime
    // truth; a persisted-creds-but-not-yet-OAuthed state still requires
    // an action from the operator (click the button), so showing it as
    // "to-do" is honest.
    let broker_done = session.is_connected();

    // Card 2 — data folder. Considered done once we have a non-empty
    // data dir that actually exists on disk. The empty-string fallback
    // (set when neither wizard nor settings have run) is treated as
    // missing.
    let data_done = !state.runtime.data_dir.as_os_str().is_empty()
        && state.runtime.data_dir.exists();

    // Card 3 — risk caps. The Risk panel always renders with defaults,
    // so "done" here means the operator actually engaged with it. We
    // detect engagement two ways:
    //   - `require_stop_loss` is still on (the prop-firm-safe default),
    //     AND
    //   - at least one numeric cap has been moved off its factory
    //     default (max_lot_size != 10.0 OR risk_per_trade != 0.03).
    // A factory-fresh AppState therefore stays "to-do" and an operator
    // who has slid even one slider becomes "done". Cheap, no extra
    // state file to drift.
    // Risk caps are stored as f64. We use f64::EPSILON to detect any
    // movement off the factory defaults; the operator only needs to
    // touch ONE slider for the card to flip to "done".
    let risk_touched = (state.risk.max_lot_size - 10.0).abs() > f64::EPSILON
        || (state.risk.risk_per_trade - 0.03).abs() > f64::EPSILON;
    let risk_done = state.risk.require_stop_loss && risk_touched;

    // Card 4 — first training. We check the disk artifacts dir; that's
    // the same evidence the trading session itself uses to decide
    // whether auto-trade has anything to consult. We deliberately do
    // NOT key off the in-memory `training_job` snapshot — a freshly
    // launched app has it `None` even though models exist on disk from
    // a previous session.
    let training_done = training_artifact_exists_on_disk(&state.runtime.data_dir);

    [
        Card {
            title: "Connect cTrader broker",
            description:
                "OAuth into your cTrader account so NeoEthos can stream quotes \
                 and (in live mode) place orders on your behalf.",
            target_tab: WorkspaceTab::BrokerSetup,
            done: broker_done,
            cta: "Open Broker Setup →",
        },
        Card {
            title: "Pick your data folder",
            description:
                "Point NeoEthos at the directory that holds (or will hold) \
                 your historical OHLCV data. The Data Bootstrap tab also \
                 lets you download missing history from the broker.",
            target_tab: WorkspaceTab::DataBootstrap,
            done: data_done,
            cta: "Open Data Bootstrap →",
        },
        Card {
            title: "Confirm your risk caps",
            description:
                "Daily / total drawdown limits, per-trade max risk, \
                 mandatory stop-loss policy. The defaults are prop-firm \
                 compliant, but you must acknowledge them before live \
                 trading is enabled.",
            target_tab: WorkspaceTab::Risk,
            done: risk_done,
            cta: "Open Risk Settings →",
        },
        Card {
            title: "Run your first training",
            description:
                "Kick off discovery + training on the selected symbol so \
                 the engine has at least one model to consult before \
                 auto-trade is allowed to fire.",
            target_tab: WorkspaceTab::Training,
            done: training_done,
            cta: "Open Training →",
        },
    ]
}

/// Walk `<data_dir>/models/` for any `.json` / `.bin` artifact. We don't
/// validate the artifact format here — that's the registry's job. We
/// just want a yes/no signal for "has anything ever been trained".
fn training_artifact_exists_on_disk(data_dir: &std::path::Path) -> bool {
    let models_dir = data_dir.join("models");
    let Ok(read_dir) = std::fs::read_dir(&models_dir) else {
        return false;
    };
    for entry in read_dir.flatten() {
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if matches!(ext, "json" | "bin" | "onnx" | "safetensors") {
                    return true;
                }
            }
        }
        if path.is_dir() {
            // One level of subdirs (e.g. <data>/models/EURUSD-M1/...).
            // Two levels is enough for our cross-session check.
            if let Ok(sub) = std::fs::read_dir(&path) {
                for inner in sub.flatten() {
                    if inner.path().is_file() {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Render the Welcome tab. Returns the tab the operator clicked into,
/// if any — the dock layer picks this up and calls `focus_tab` so the
/// click actually navigates.
pub fn render(
    ui: &mut egui::Ui,
    state: &AppState,
    session: &TradingSession,
) -> WelcomeAction {
    let cards = build_cards(state, session);
    let done_count = cards.iter().filter(|c| c.done).count();
    let total = cards.len();
    let mut action = WelcomeAction { requested_tab: None };

    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.add_space(theme::SPACE_LG);
        ui.vertical_centered(|ui| {
            ui.heading("Welcome to NeoEthos");
            ui.add_space(theme::SPACE_SM);
            ui.label(
                egui::RichText::new(
                    "A disciplined multi-model ML engine for FX strategy research \
                     and risk-aware execution.",
                )
                .color(theme::TEXT_FAINT)
                .italics(),
            );
        });
        ui.add_space(theme::SPACE_LG);

        // Progress strip — quiet, no big celebratory animation. The
        // operator can scan it at a glance.
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(format!("Setup progress: {done_count}/{total}"))
                    .color(theme::TEXT_PRIMARY)
                    .strong(),
            );
            ui.add_space(theme::SPACE_MD);
            let progress = done_count as f32 / total.max(1) as f32;
            ui.add(
                egui::ProgressBar::new(progress)
                    .desired_width(220.0)
                    .show_percentage(),
            );
        });
        ui.add_space(theme::SPACE_LG);

        // Render cards. We deliberately do NOT use a grid widget —
        // egui's `Grid` enforces uniform row heights and the card
        // bodies wrap to different lengths. A vertical stack of cards
        // gives each one room to breathe and lays out cleanly at any
        // dock width.
        for card in &cards {
            render_card(ui, card, &mut action);
            ui.add_space(theme::SPACE_MD);
        }

        if done_count == total {
            ui.add_space(theme::SPACE_LG);
            ui.vertical_centered(|ui| {
                ui.label(
                    egui::RichText::new("All set ✓ — happy trading.")
                        .color(theme::ACCENT)
                        .size(16.0)
                        .strong(),
                );
                ui.add_space(theme::SPACE_SM);
                ui.label(
                    egui::RichText::new(
                        "You can close this tab from the sidebar; it stays \
                         accessible if anything regresses.",
                    )
                    .color(theme::TEXT_FAINT),
                );
            });
        }

        ui.add_space(theme::SPACE_LG);
    });

    action
}

fn render_card(ui: &mut egui::Ui, card: &Card, action: &mut WelcomeAction) {
    egui::Frame::new()
        .fill(theme::PANEL_BG)
        .stroke(egui::Stroke::new(
            1.0,
            if card.done {
                theme::ACCENT.linear_multiply(0.5)
            } else {
                theme::BORDER
            },
        ))
        .corner_radius(egui::CornerRadius::same(theme::RADIUS_SM))
        .inner_margin(egui::Margin::same(theme::SPACE_MD as i8))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                // Status glyph: filled circle for done, hollow for to-do.
                let glyph = if card.done { "●" } else { "○" };
                let glyph_colour = if card.done {
                    theme::ACCENT
                } else {
                    theme::TEXT_FAINT
                };
                ui.label(
                    egui::RichText::new(glyph)
                        .color(glyph_colour)
                        .size(18.0)
                        .strong(),
                );
                ui.add_space(theme::SPACE_SM);
                ui.vertical(|ui| {
                    ui.label(
                        egui::RichText::new(card.title)
                            .color(theme::TEXT_PRIMARY)
                            .size(15.0)
                            .strong(),
                    );
                    ui.add_space(2.0);
                    ui.label(
                        egui::RichText::new(card.description)
                            .color(theme::TEXT_FAINT),
                    );
                    ui.add_space(theme::SPACE_SM);
                    if ui.button(card.cta).clicked() {
                        action.requested_tab = Some(card.target_tab);
                    }
                });
            });
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_artifact_check_handles_missing_dir() {
        // No models dir → no artifact.
        let dir = std::env::temp_dir().join(format!(
            "neoethos_welcome_missing_{}",
            std::process::id()
        ));
        assert!(!training_artifact_exists_on_disk(&dir));
    }

    #[test]
    fn training_artifact_check_finds_file_in_subdir() {
        use std::fs;
        let root = std::env::temp_dir().join(format!(
            "neoethos_welcome_artifact_{}",
            std::process::id()
        ));
        let models = root.join("models").join("EURUSD-M1");
        fs::create_dir_all(&models).expect("temp model dir");
        fs::write(models.join("model.bin"), b"x").expect("write artifact");
        assert!(training_artifact_exists_on_disk(&root));
        let _ = fs::remove_dir_all(&root);
    }
}
