//! TUI wizard skeleton — `forex-cli wizard`.
//!
//! Spec: `docs/audits/research/installer_wizard_ux_spec.md` §8 (CLI
//! parity) — the wizard's state machine is shared between the egui
//! front-end (`forex-app::ui::wizard`) and this ratatui front-end.
//! Each GUI step has a TUI counterpart (§8.1 mapping table).
//!
//! Keybindings (spec §8.2):
//! - → / Enter — Continue (Next)
//! - ←        — Back
//! - Tab/S-Tab — Cycle fields within current step
//! - Space    — Toggle checkbox / radio
//! - s        — Skip current step (only when skippable)
//! - r        — Retry the last failed action
//! - Esc      — Open Cancel confirmation
//! - ?        — Open inline help
//! - q        — Same as Esc
//!
//! No-tty mode (§8.3): if `stdin` is not a tty, this entry point
//! refuses to start and prints to stderr.
//!
//! FIXME(wizard-tui): port from desktop. The desktop wizard in
//! `forex-app/src/ui/wizard/` defines the canonical state machine
//! (`WizardController`) and per-step renderer signatures; this file
//! is the placeholder so that `forex-cli wizard` is wired into
//! `main.rs` as a recognised subcommand. The actual ratatui draw
//! routines (§8.1 mapping table) require a `forex-cli` ↔ `forex-app`
//! dep on the shared `WizardController` data type, which today
//! lives behind `forex_app::ui::wizard` — that requires either
//! lifting the controller into a shared crate or a `pub use` path
//! across crates. Spec §1.4 notes the controller is "shared between
//! GUI and TUI fronts; only the rendering layer differs".

use anyhow::Result;

/// stderr message printed when `stdin` is not a tty. Spec §8.3.
pub const WIZARD_TUI_NO_TTY_MESSAGE: &str =
    "forex-cli wizard requires a tty; use `forex-cli init` for headless first-run setup.";

/// Refuse-to-run guard for non-tty stdin. Returns `Err` so callers
/// surface the message at the process-exit boundary.
pub fn run_wizard_tui() -> Result<()> {
    if !is_stdin_tty() {
        eprintln!("{}", WIZARD_TUI_NO_TTY_MESSAGE);
        return Err(anyhow::anyhow!("forex-cli wizard: stdin is not a tty"));
    }

    // FIXME(wizard-tui): port the per-step ratatui pages from the
    // desktop wizard (see `forex-app/src/ui/wizard/`):
    //   - mod.rs       → state machine
    //   - welcome.rs   → §8.1 row "1 — License" (pager + Y/N)
    //   - path.rs      → §8.1 row "2 — Path" (tab-complete textinput)
    //   - account_profile.rs → "3 — Profile" (three list selectors)
    //   - oauth.rs     → "4 — OAuth" (browser launch + copy-paste)
    //   - symbols.rs   → "5 — Symbols" (two-pane multi-select)
    //   - historical.rs → "6 — History" (ratatui::Gauge)
    //   - hardware.rs  → "7 — Hardware" (card-style block)
    //   - news_api.rs  → "8 — News" (masked input)
    //   - autostart.rs → "9 — Autostart" (single toggle)
    //   - autonomy_risk.rs → "9.5 — Quiz + autonomy"
    //   - summary.rs   → "10 — Summary" (scrollable table)
    //
    // For now we just print the placeholder so the subcommand is a
    // recognised entry point without crashing.
    println!("forex-cli wizard: TUI rendering not yet ported.");
    println!("Run the desktop wizard via `forex-app --wizard` for now.");
    Ok(())
}

fn is_stdin_tty() -> bool {
    use std::io::IsTerminal;
    std::io::stdin().is_terminal()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_tty_message_mentions_init_subcommand() {
        // §8.3 — verbatim message must point users at `forex-cli init`.
        assert!(WIZARD_TUI_NO_TTY_MESSAGE.contains("forex-cli init"));
    }

    #[test]
    fn run_wizard_returns_err_when_stdin_is_not_tty() {
        // The cargo test harness pipes stdin → IsTerminal is false.
        let result = run_wizard_tui();
        assert!(result.is_err(), "must refuse on non-tty stdin");
    }
}
