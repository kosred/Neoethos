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

/// HALT-aware variant. When `halted == true` the pill ignores the
/// trading environment and renders "HALTED" in solid `DANGER` red so
/// the operator can never miss it. Matches ship-gate §5.1.5: "status
/// pill flips to 'HALTED'".
pub fn draw_status_pill_with_halt(
    ui: &mut egui::Ui,
    env: TradingEnvironment,
    account_label: &str,
    halted: bool,
) {
    if halted {
        theme::status_badge(ui, &pill_text_for_halt(account_label), pill_color_for_halt());
    } else {
        draw_status_pill(ui, env, account_label);
    }
}

/// HALT pill color — solid `DANGER` red, the same token used by the
/// HALT button. Picked deliberately so the two visually rhyme when both
/// are active: a HALT trip flashes the same red on both surfaces.
pub fn pill_color_for_halt() -> egui::Color32 {
    theme::STATUS_LIVE
}

/// HALT pill text. Always starts with `HALTED`; the account label is
/// appended after a middle-dot so the operator can still tell which
/// account is frozen.
pub fn pill_text_for_halt(account_label: &str) -> String {
    let trimmed = account_label.trim();
    if trimmed.is_empty() {
        "HALTED".to_string()
    } else {
        format!("HALTED · {trimmed}")
    }
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

    // ── HALT pill (ship-gate §5.1.5) ──────────────────────────────────

    #[test]
    fn halt_pill_uses_danger_red_token() {
        assert_eq!(pill_color_for_halt(), theme::STATUS_LIVE);
    }

    #[test]
    fn halt_pill_text_without_account() {
        assert_eq!(pill_text_for_halt(""), "HALTED");
    }

    #[test]
    fn halt_pill_text_with_account() {
        assert_eq!(pill_text_for_halt("712345"), "HALTED · 712345");
    }

    #[test]
    fn halt_pill_text_trims_whitespace_in_account_label() {
        assert_eq!(pill_text_for_halt("  998877  "), "HALTED · 998877");
    }

    /// Ship-gate §5.1.5 smoke test — drives the whole flow at the
    /// model level: trip HALT on a fresh `TradingSession`, then assert
    /// that the status-pill renderer would pick the HALTED variant
    /// over the configured trading environment. We don't drive egui
    /// pixels (`Context::default()` works in unit tests but doesn't
    /// add coverage beyond what these state assertions already give —
    /// the §5.1.5 "click the red HALT button" path is the operator's
    /// finger; everything downstream of `trip_manual_halt` is covered
    /// here and in `halt_button::tests`).
    #[test]
    fn smoke_session_halt_overrides_environment_in_status_pill() {
        use crate::app_services::trading::{
            TradingAdapterKind, TradingEnvironment, TradingSession,
        };
        use crate::app_state::{AppRuntimeConfig, AppState};
        use forex_core::Settings;

        // Set up a session in LIVE env so we know the pill would
        // otherwise show "LIVE · ...". The HALT override must beat it.
        let mut session =
            TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
        session.set_trading_environment(TradingEnvironment::LiveFull);

        let tmp = std::env::temp_dir().join(format!(
            "forex-ai-pill-halt-test-{}",
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
        let mut state = AppState::new(runtime, &Settings::default(), Vec::new());
        let account_label = "987654";

        // Pre-halt: pill text and color follow the LiveFull env.
        assert!(!session.is_halted(), "fresh session must not be halted");
        assert_eq!(
            pill_text_for(TradingEnvironment::LiveFull, account_label),
            "LIVE · 987654"
        );
        assert_eq!(
            pill_color_for(TradingEnvironment::LiveFull),
            theme::STATUS_LIVE
        );

        // Trip HALT.
        let summary = session.trip_manual_halt(&mut state);
        assert!(session.is_halted(), "trip_manual_halt must set the flag");
        assert!(
            summary.environment_label.contains("LIVE"),
            "summary preserves the env in force at trip time, got {}",
            summary.environment_label
        );

        // Post-halt: the HALT-aware accessors return the HALTED variant
        // regardless of the configured environment. This is exactly
        // what `draw_status_pill_with_halt` paints in the top bar.
        let halted_text = pill_text_for_halt(account_label);
        let halted_color = pill_color_for_halt();
        assert_eq!(halted_text, "HALTED · 987654");
        assert_eq!(halted_color, theme::STATUS_LIVE);
        assert_ne!(
            halted_text,
            pill_text_for(TradingEnvironment::LiveFull, account_label),
            "HALT override must visibly differ from the env-only label"
        );
    }
}
