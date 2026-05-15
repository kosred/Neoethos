//! Persistent Demo/Paper/Live status pill rendered in the main top
//! bar, left of the account dropdown.
//!
//! Implements giants-pattern gap #1 from
//! `docs/audits/research/wizard_onboarding_competitive_analysis.md`
//! §10 — the ThinkOrSwim paperMoney pill / TradingView Trading Panel
//! gray-vs-red convention. The pill is ALWAYS visible; the operator
//! must never have to scan more than one location to answer
//! "am I trading real money right now?".
//!
//! Visual contract (cited from `docs/audits/research/ui_ux_design_spec.md`
//! §5.1):
//!
//! | Variant         | Token            | Hex      | Pill text             |
//! |-----------------|------------------|----------|------------------------|
//! | Demo            | `TEXT_MUTED`     | `#9AA4B2`| `DEMO · <account>`     |
//! | Paper           | `WARNING`        | `#F4B400`| `PAPER · <account>`    |
//! | LiveSmall       | `DANGER`         | `#EF5350`| `LIVE SMALL · <account>` |
//! | LiveFull        | `DANGER` (bold)  | `#EF5350`| `LIVE · <account>`     |
//!
//! `DANGER` is the same `--candle-down` red used by bear candles, per
//! the design-spec §5.1 mapping `--status-danger -> --candle-down`.
//! The pill renders as a `status_badge` (low-alpha fill + colored
//! stroke + colored bold UPPERCASE text), which is visually distinct
//! from the solid-red HALT button so the two co-exist without
//! ambiguity (HALT remains the ONLY `DANGER`-solid surface in the
//! whole chrome per `wizard_onboarding_competitive_analysis.md`
//! §10.6).

use crate::app_services::trading::TradingEnvironment;
use crate::ui::theme;
use eframe::egui;

/// Render the persistent trading-environment pill into the current
/// horizontal layout. Call once per frame from the top-bar draw loop.
///
/// The pill is always visible and never collapses. `account_label`
/// is shown after a middle-dot separator; pass `""` to suppress it
/// (e.g. when the broker session has not discovered any accounts
/// yet).
pub fn draw_status_pill(
    ui: &mut egui::Ui,
    env: TradingEnvironment,
    account_label: &str,
) {
    let color = pill_color_for(env);
    let text = pill_text_for(env, account_label);
    // `status_badge` is the canonical "compact colored pill" primitive
    // from `theme.rs:334`. Reusing it keeps the new pill on the same
    // visual grammar as the existing "PRO" badge in the top bar — both
    // are low-alpha fill + strong border + colored UPPERCASE text.
    theme::status_badge(ui, &text, color);
}

/// Token lookup so tests can assert the variant -> color mapping
/// without driving egui. The function is intentionally simple so
/// the unit tests in `chrome/status_pill.rs` document the contract.
pub fn pill_color_for(env: TradingEnvironment) -> egui::Color32 {
    match env {
        TradingEnvironment::Demo => theme::STATUS_DEMO,
        TradingEnvironment::Paper => theme::STATUS_PAPER,
        TradingEnvironment::LiveSmall | TradingEnvironment::LiveFull => theme::STATUS_LIVE,
    }
}

/// Text the pill displays. Public for testability; the production
/// renderer goes through `draw_status_pill`.
pub fn pill_text_for(env: TradingEnvironment, account_label: &str) -> String {
    let trimmed = account_label.trim();
    if trimmed.is_empty() {
        env.pill_label().to_string()
    } else {
        // Middle-dot · separator — the same glyph used in the
        // existing top-bar ribbon (`compact_status_text` /
        // engine label rows in `main.rs`) so the typography stays
        // uniform.
        format!("{} · {}", env.pill_label(), trimmed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_pill_uses_muted_text_token() {
        assert_eq!(pill_color_for(TradingEnvironment::Demo), theme::STATUS_DEMO);
    }

    #[test]
    fn paper_pill_uses_warning_amber_token() {
        assert_eq!(
            pill_color_for(TradingEnvironment::Paper),
            theme::STATUS_PAPER
        );
    }

    #[test]
    fn live_small_pill_uses_danger_red_token() {
        assert_eq!(
            pill_color_for(TradingEnvironment::LiveSmall),
            theme::STATUS_LIVE
        );
    }

    #[test]
    fn live_full_pill_uses_danger_red_token() {
        assert_eq!(
            pill_color_for(TradingEnvironment::LiveFull),
            theme::STATUS_LIVE
        );
    }

    #[test]
    fn pill_text_demo_with_account() {
        assert_eq!(
            pill_text_for(TradingEnvironment::Demo, "12345"),
            "DEMO · 12345"
        );
    }

    #[test]
    fn pill_text_live_full_without_account_omits_separator() {
        assert_eq!(pill_text_for(TradingEnvironment::LiveFull, ""), "LIVE");
    }

    #[test]
    fn pill_text_live_small_trims_whitespace() {
        assert_eq!(
            pill_text_for(TradingEnvironment::LiveSmall, "  acct-7  "),
            "LIVE SMALL · acct-7"
        );
    }
}
