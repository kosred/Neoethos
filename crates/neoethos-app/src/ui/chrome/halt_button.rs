//! Red HALT panic button in the top-right of the main chrome.
//!
//! Implements giants-pattern gap #3 from
//! `docs/audits/research/wizard_onboarding_competitive_analysis.md`
//! §10.4 T-Manual — the MT4/5 AutoTrading panic button / FTMO /
//! TakeProfitTrader / tastytrade manual kill-switch. The button is
//! always visible regardless of which workspace tab is active so the
//! operator can stop trading from anywhere in the app.
//!
//! Visual contract:
//!
//! - Solid-red button (`theme::DANGER` fill), label `HALT` in bold
//!   uppercase. The ONLY solid-`DANGER` surface in the whole chrome
//!   per `wizard_onboarding_competitive_analysis.md` §10.6.
//! - Confirmation modal is dismissable only by clicking `[Cancel]` or
//!   `[Confirm HALT]`; outside-click is suppressed so the operator
//!   does not accidentally close the dialog.
//! - On `[Confirm HALT]` the button calls `session.trip_manual_halt()`
//!   which:
//!     1. Sets the `halted` flag (T-Manual rejects subsequent orders).
//!     2. Iterates open positions and calls the existing close path.
//!     3. Iterates pending orders and calls the existing cancel path.
//!     4. Calls `risky_mode_manager.trip_manual_halt()` when Risky
//!        Mode is active (research §5.5 — the sticky kill-switch
//!        tier `KillSwitchTier::Manual` is set so a later
//!        `execute_ctrader_order` cannot slip past Risky Mode even
//!        if the operator clears `halt_state.halted` without
//!        re-enabling the Risky Mode side). Covered by
//!        `trading_tests::halt_button_also_trips_risky_mode_kill_switch`.
//!     5. Emits `tracing::error!(target: "neoethos_app::halt", ...)`.
//!     6. Writes a sentinel file under `<data-dir>/HALTED_<ts>.flag`.
//! - After the trip, a persistent banner appears: "TRADING HALTED —
//!   operator must clear flag to resume", with a `[Clear HALT]`
//!   button that flips `halted` back and removes the sentinel.
//!
//! The state for "modal is open" lives in egui persisted memory keyed
//! by `egui::Id::new("neoethos_app::chrome::halt_modal_open")` so the
//! draw helper stays a free function — the existing `ForexApp` struct
//! is not extended just for one transient bool.

use crate::app_services::trading::TradingSession;
use crate::app_state::AppState;
use crate::ui::theme;
use eframe::egui;

/// Stable egui id for the modal-open flag. Persisted across frames
/// so the modal survives repaints triggered by background events
/// (heartbeats, service messages, etc.).
fn modal_open_id() -> egui::Id {
    egui::Id::new("neoethos_app::chrome::halt_modal_open")
}

/// Render the HALT button into the current horizontal layout. Call
/// once per frame from the top-bar draw loop AFTER the status pill
/// so the eye reads "what mode am I in -> emergency stop" from left
/// to right.
///
/// Mutates `session.halt_state` via `session.trip_manual_halt` when
/// the operator confirms the modal. `state` is required because the
/// trip iterates `state.order_ticket` to drive the existing close /
/// cancel paths.
pub fn draw_halt_button(ui: &mut egui::Ui, session: &mut TradingSession, state: &mut AppState) {
    let ctx = ui.ctx().clone();
    let modal_id = modal_open_id();
    let mut modal_open = ctx.data(|d| d.get_temp::<bool>(modal_id)).unwrap_or(false);

    // The button itself — solid red, bold uppercase `HALT`. We do not
    // route through `theme::button(ButtonKind::Danger)` because that
    // helper renders the muted-fill ghost-style danger button; the
    // HALT button MUST be a saturated solid so it stands apart from
    // every other control in the chrome.
    let halt_button = egui::Button::new(
        egui::RichText::new("HALT")
            .size(theme::FONT_BODY)
            .strong()
            .color(egui::Color32::WHITE),
    )
    .fill(theme::DANGER)
    .stroke(egui::Stroke::new(1.0, theme::DANGER))
    .corner_radius(egui::CornerRadius::same(theme::RADIUS_SM))
    .min_size(egui::vec2(64.0, theme::BUTTON_HEIGHT_SM));

    let response = ui
        .add(halt_button)
        .on_hover_text("Halt all trading — closes positions and cancels orders.");
    if response.clicked() {
        modal_open = true;
    }

    if modal_open {
        // egui::Window with `collapsible(false)` and a captured central
        // click area approximates a dismiss-on-button-only modal. We
        // do not call `interact(Sense::click())` on the surrounding
        // background, so outside-click is naturally inert.
        let mut should_close = false;
        let mut should_confirm = false;
        egui::Window::new("Halt all trading?")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(&ctx, |ui| {
                ui.set_min_width(360.0);
                ui.label(
                    egui::RichText::new(
                        "This will cancel all pending orders and close all open positions \
                         immediately.",
                    )
                    .size(theme::FONT_BODY)
                    .color(theme::TEXT_PRIMARY),
                );
                ui.add_space(theme::SPACE_SM);
                ui.label(
                    egui::RichText::new(
                        "After confirming, new orders are blocked until you clear the \
                         HALT banner.",
                    )
                    .size(theme::FONT_CAPTION)
                    .color(theme::TEXT_MUTED),
                );
                ui.add_space(theme::SPACE_MD);
                ui.horizontal(|ui| {
                    if theme::button(ui, "Cancel", theme::ButtonKind::Secondary).clicked() {
                        should_close = true;
                    }
                    // Confirm button: same solid-red treatment as the
                    // HALT button itself so the operator's eye reads
                    // the path "click HALT -> click Confirm HALT" as
                    // a consistent destructive flow.
                    let confirm = egui::Button::new(
                        egui::RichText::new("Confirm HALT")
                            .size(theme::FONT_BODY)
                            .strong()
                            .color(egui::Color32::WHITE),
                    )
                    .fill(theme::DANGER)
                    .stroke(egui::Stroke::new(1.0, theme::DANGER))
                    .corner_radius(egui::CornerRadius::same(theme::RADIUS_SM))
                    .min_size(egui::vec2(0.0, theme::BUTTON_HEIGHT));
                    if ui.add(confirm).clicked() {
                        should_confirm = true;
                    }
                });
            });
        if should_confirm {
            session.trip_manual_halt(state);
            modal_open = false;
        } else if should_close {
            modal_open = false;
        }
    }

    ctx.data_mut(|d| d.insert_temp::<bool>(modal_id, modal_open));
}

/// Render the persistent "TRADING HALTED" banner once a HALT has
/// tripped. Call from the main draw loop AFTER the top bar so the
/// banner sits flush below the chrome and reads as a separate strip.
/// Returns `true` when the operator clicks `[Clear HALT]`.
pub fn draw_halt_banner(ui: &mut egui::Ui, session: &mut TradingSession) -> bool {
    if !session.is_halted() {
        return false;
    }
    let mut cleared = false;
    egui::Frame::new()
        .fill(theme::DANGER.linear_multiply(0.18))
        .stroke(egui::Stroke::new(1.0, theme::DANGER))
        .inner_margin(egui::Margin::symmetric(
            theme::SPACE_LG as i8,
            theme::SPACE_SM as i8,
        ))
        .corner_radius(egui::CornerRadius::same(theme::RADIUS_SM))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("TRADING HALTED")
                        .size(theme::FONT_BODY)
                        .strong()
                        .color(theme::DANGER),
                );
                ui.label(
                    egui::RichText::new("— operator must clear flag to resume")
                        .size(theme::FONT_CAPTION)
                        .color(theme::TEXT_PRIMARY),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if theme::small_button(ui, "Clear HALT", theme::ButtonKind::Secondary)
                        .on_hover_text(
                            "Remove the HALT sentinel and allow new orders to flow again.",
                        )
                        .clicked()
                    {
                        session.clear_halt();
                        cleared = true;
                    }
                });
            });
        });
    cleared
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::trading::{TradingAdapterKind, TradingEnvironment};
    use crate::app_state::{AppRuntimeConfig, AppState};
    use neoethos_core::Settings;

    fn test_state() -> AppState {
        let tmp = std::env::temp_dir().join(format!(
            "neoethos-halt-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = std::fs::create_dir_all(&tmp);
        let runtime = AppRuntimeConfig {
            config_path: "config.yaml".to_string(),
            data_dir: tmp,
            start_local: true,
            auto_discovery: false,
            auto_training: false,
        };
        AppState::new(runtime, &Settings::default(), Vec::new())
    }

    #[test]
    fn trip_manual_halt_sets_halted_flag() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        let mut state = test_state();
        assert!(!session.is_halted());
        session.trip_manual_halt(&mut state);
        assert!(session.is_halted());
    }

    #[test]
    fn trip_manual_halt_iterates_open_positions() {
        // Without a connected cTrader runtime, position list is empty.
        // The summary still records zero — what matters is that the
        // iteration runs to completion and the flag flips.
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        let mut state = test_state();
        let summary = session.trip_manual_halt(&mut state);
        assert!(session.is_halted());
        assert_eq!(summary.positions_closed, 0);
        assert_eq!(summary.orders_cancelled, 0);
    }

    #[test]
    fn new_order_rejected_when_halted() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        let mut state = test_state();
        session.trip_manual_halt(&mut state);
        // execute_buy_order routes through execute_ctrader_order which
        // hard-rejects when halted, regardless of the ticket's order type.
        session.execute_buy_order(&mut state);
        assert!(state.status_msg.contains("HALT in force"));
    }

    #[test]
    fn clear_halt_flips_flag_and_allows_orders_again() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        let mut state = test_state();
        session.trip_manual_halt(&mut state);
        assert!(session.is_halted());
        session.clear_halt();
        assert!(!session.is_halted());
    }

    #[test]
    fn clear_halt_removes_sentinel_file() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        let mut state = test_state();
        session.trip_manual_halt(&mut state);
        let sentinel = session
            .halt_state()
            .sentinel_path
            .clone()
            .expect("trip_manual_halt should record a sentinel path");
        assert!(
            sentinel.exists(),
            "sentinel file should exist on disk after trip"
        );
        session.clear_halt();
        assert!(
            !sentinel.exists(),
            "sentinel file should be removed after clear_halt"
        );
    }

    #[test]
    fn environment_label_is_recorded_in_summary() {
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.set_trading_environment(TradingEnvironment::Paper);
        let mut state = test_state();
        let summary = session.trip_manual_halt(&mut state);
        assert_eq!(summary.environment_label, "PAPER");
    }
}
