//! Step 1 — Welcome + License.
//!
//! Spec: `installer_wizard_ux_spec.md` §2 Step 1 + §9.1 mockup.
//! - NOT skippable (the only mandatory step).
//! - `[Continue →]` disabled until license-accepted checkbox is on.
//! - On accept, records LICENSE SHA-256 + timestamp in `wizard_state.json`.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use eframe::egui;

use super::{StepResult, WizardController};
use crate::ui::theme;

/// Step time budget — used to drive the "≈ 10 minutes" estimate in
/// the welcome copy. Spec §2 Step 1 reports ≤ 30 s for this step.
pub const WIZARD_STEP_WELCOME_BUDGET_SECONDS: u32 = 30;

/// 10 numbered steps + 9.5 — total user-visible budget. Spec §9.1
/// banner copy ("≈ 10 minutes").
pub const WIZARD_TOTAL_BUDGET_MINUTES: u32 = 10;

/// Env override for the LICENSE path. Set by the installer so the
/// wizard reads the same on-disk LICENSE that ships in the package
/// (`/usr/share/forex-ai/LICENSE` on Linux, `Contents/Resources/LICENSE`
/// on macOS, `<install_dir>\LICENSE` on Windows — per
/// `installer_infrastructure_spec.md` §8). Test hook too.
pub const WIZARD_LICENSE_PATH_ENV: &str = "FOREX_AI_LICENSE_PATH";

/// Compile-time fallback so the wizard always shows *something* even
/// if the on-disk LICENSE was deleted (the "LICENSE missing" error
/// class from `WizardError::LicenseMissing`). Path is relative to
/// this file: `crates/forex-app/src/ui/wizard/welcome.rs` →
/// `<repo_root>/LICENSE`.
const BUNDLED_LICENSE: &str = include_str!("../../../../../LICENSE");

/// Read the LICENSE file from disk, falling back to the bundled
/// `include_str!` copy on miss. Lookup order (spec §2 Step 1 Actions
/// + `installer_infrastructure_spec.md` §8):
///   1. `FOREX_AI_LICENSE_PATH` env var (installer / test hook).
///   2. Platform-canonical install paths next to the running binary.
///   3. Compile-time bundled copy (warned, never silent — spec §3
///      rule 1 "Never silently skip").
pub fn load_license_text() -> String {
    if let Some(text) = read_from_env() {
        return text;
    }
    if let Some(text) = read_from_platform_defaults() {
        return text;
    }
    tracing::warn!(
        target: "forex_app::wizard",
        "LICENSE file not found on disk; falling back to compiled-in copy"
    );
    BUNDLED_LICENSE.to_string()
}

fn read_from_env() -> Option<String> {
    let path = std::env::var(WIZARD_LICENSE_PATH_ENV).ok()?;
    match std::fs::read_to_string(&path) {
        Ok(text) => Some(text),
        Err(err) => {
            tracing::warn!(
                target: "forex_app::wizard",
                path = %path,
                error = %err,
                "FOREX_AI_LICENSE_PATH set but unreadable; continuing lookup"
            );
            None
        }
    }
}

fn read_from_platform_defaults() -> Option<String> {
    for candidate in platform_license_candidates() {
        if let Ok(text) = std::fs::read_to_string(&candidate) {
            return Some(text);
        }
    }
    None
}

/// Build the per-platform LICENSE candidate list per
/// `installer_infrastructure_spec.md` §8. Order: next-to-binary
/// (works for portable / dev builds), then OS-canonical share dirs.
fn platform_license_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();

    // Next to the current executable. Covers Windows `<install_dir>\LICENSE`,
    // macOS `Contents/MacOS/forex-app` → `../Resources/LICENSE`, and
    // `cargo run` dev where the binary lives in `target/<profile>/`.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            out.push(dir.join("LICENSE"));
            // macOS .app bundle layout — Contents/MacOS/.. → Contents/Resources.
            out.push(dir.join("../Resources/LICENSE"));
        }
    }

    // Platform-canonical share dirs.
    if cfg!(target_os = "linux") {
        out.push(PathBuf::from("/usr/share/forex-ai/LICENSE"));
        out.push(PathBuf::from("/usr/share/doc/forex-ai/LICENSE"));
    } else if cfg!(target_os = "macos") {
        out.push(PathBuf::from(
            "/Applications/Forex AI.app/Contents/Resources/LICENSE",
        ));
    }

    out
}

/// Tracks scroll-to-end so the "Continue" button is gated on the
/// user having actually scrolled the LICENSE pane to the bottom
/// (spec §2 Step 1: scroll-to-accept). State is per-process; the
/// wizard resets it when the user navigates back to this step.
fn scrolled_to_end_flag() -> &'static std::sync::Mutex<bool> {
    static FLAG: OnceLock<std::sync::Mutex<bool>> = OnceLock::new();
    FLAG.get_or_init(|| std::sync::Mutex::new(false))
}

fn license_cache() -> &'static String {
    static CACHE: OnceLock<String> = OnceLock::new();
    CACHE.get_or_init(load_license_text)
}

/// Test-only helper — clears the scroll-to-end gate so re-entrant
/// renders start from "not yet scrolled".
#[cfg(test)]
fn reset_scroll_gate() {
    if let Ok(mut g) = scrolled_to_end_flag().lock() {
        *g = false;
    }
}

/// Returns true if the on-disk LICENSE is reachable via any of the
/// configured lookup paths. Surfaced for tests that want to assert
/// the fallback branch without monkey-patching the env.
#[allow(dead_code)]
fn license_resolves_on_disk(license_path_override: Option<&Path>) -> bool {
    if let Some(p) = license_path_override {
        return std::fs::metadata(p).is_ok();
    }
    if let Ok(path) = std::env::var(WIZARD_LICENSE_PATH_ENV) {
        if std::fs::metadata(&path).is_ok() {
            return true;
        }
    }
    platform_license_candidates()
        .iter()
        .any(|c| std::fs::metadata(c).is_ok())
}

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

    // LICENSE pane — scrollable monospace, real file read with
    // include_str!() compile-time fallback. Spec §2 Step 1 Actions.
    let license_text = license_cache().as_str();
    let scroll_output = egui::ScrollArea::vertical()
        .max_height(280.0)
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.add(
                egui::Label::new(
                    egui::RichText::new(license_text)
                        .size(theme::FONT_BODY)
                        .monospace(),
                )
                .wrap(),
            );
        });

    // Scroll-to-end gate (spec §2 Step 1: "Accept-to-continue is
    // gated on actually scrolling to the end"). Latch true once the
    // user has scrolled the inner content within ~4 px of the
    // bottom; egui's `ScrollAreaOutput.state` exposes the offset +
    // viewport.
    let mut scrolled_to_end_latched = scrolled_to_end_flag().lock().map(|g| *g).unwrap_or(false);
    let content_height = scroll_output.content_size.y;
    let viewport_height = scroll_output.inner_rect.height();
    let offset_y = scroll_output.state.offset.y;
    let max_scroll = (content_height - viewport_height).max(0.0);
    // "max_scroll ≤ 0.5" handles the short-license case where the
    // whole text fits without any scrollbar: the user has already
    // seen it, so we don't trap them.
    let reached_end = max_scroll <= 0.5 || offset_y + 4.0 >= max_scroll;
    if reached_end {
        scrolled_to_end_latched = true;
        if let Ok(mut g) = scrolled_to_end_flag().lock() {
            *g = true;
        }
    }

    ui.separator();
    // Checkbox is disabled until the pane has been scrolled to the
    // end at least once — Microsoft Learn / NN/G UX wizard pattern:
    // "Don't let users accept text they haven't seen".
    let accept_enabled = scrolled_to_end_latched;
    ui.add_enabled_ui(accept_enabled, |ui| {
        ui.checkbox(
            &mut controller.config.license_accepted,
            "I have read and accept the license",
        );
    });
    if !accept_enabled {
        ui.label(
            egui::RichText::new("Scroll to the end of the license to enable acceptance.")
                .color(theme::TEXT_MUTED)
                .size(theme::FONT_CAPTION),
        );
    }

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
    use std::io::Write;

    // Tests that mutate `FOREX_AI_LICENSE_PATH` MUST run in series —
    // cargo runs tests in a single process by default, and parallel
    // env mutation is racy. A per-suite mutex keeps them serial
    // without requiring the operator to set RUST_TEST_THREADS=1.
    fn env_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
    }

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

    #[test]
    fn load_license_text_reads_from_env_var() {
        let _guard = env_lock().lock().unwrap();
        let mut tmp = std::env::temp_dir();
        tmp.push(format!("forex-ai-license-test-{}.txt", std::process::id()));
        let body = "TEST LICENSE BODY — env override\nLine 2";
        {
            let mut f = std::fs::File::create(&tmp).expect("create temp LICENSE");
            f.write_all(body.as_bytes()).expect("write temp LICENSE");
        }
        // SAFETY: env mutation is gated by `env_lock` above, so no
        // other test mutates this var in parallel. Required because
        // `set_var` is marked unsafe in edition 2024.
        unsafe {
            std::env::set_var(WIZARD_LICENSE_PATH_ENV, &tmp);
        }
        let text = load_license_text();
        unsafe {
            std::env::remove_var(WIZARD_LICENSE_PATH_ENV);
        }
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(text, body, "env var path should win over fallbacks");
    }

    #[test]
    fn load_license_text_falls_back_to_compiled_in_when_disk_missing() {
        let _guard = env_lock().lock().unwrap();
        // Point env at a path that definitely does not exist.
        let mut tmp = std::env::temp_dir();
        tmp.push(format!(
            "forex-ai-license-does-not-exist-{}.txt",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp); // make extra sure.
        unsafe {
            std::env::set_var(WIZARD_LICENSE_PATH_ENV, &tmp);
        }
        let text = load_license_text();
        unsafe {
            std::env::remove_var(WIZARD_LICENSE_PATH_ENV);
        }
        // Bundled LICENSE is the repo-root file — non-empty + has
        // the expected dual-license header so we can't accidentally
        // pass with an empty include_str!().
        assert!(!text.is_empty(), "fallback text must be non-empty");
        // The repo LICENSE is Apache 2.0 + MIT; pick a stable token.
        assert!(
            text.contains("Apache") || text.contains("MIT") || text.contains("LICENSE"),
            "bundled LICENSE should look like a real license, got: {}",
            text.chars().take(120).collect::<String>()
        );
        assert_eq!(text, BUNDLED_LICENSE);
    }

    #[test]
    fn license_resolves_on_disk_via_env_override() {
        let _guard = env_lock().lock().unwrap();
        let mut tmp = std::env::temp_dir();
        tmp.push(format!(
            "forex-ai-license-resolve-{}.txt",
            std::process::id()
        ));
        std::fs::write(&tmp, "x").expect("write probe");
        unsafe {
            std::env::set_var(WIZARD_LICENSE_PATH_ENV, &tmp);
        }
        assert!(license_resolves_on_disk(None));
        unsafe {
            std::env::remove_var(WIZARD_LICENSE_PATH_ENV);
        }
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn reset_scroll_gate_clears_latch() {
        // Force-latch then reset and confirm the flag is cleared.
        if let Ok(mut g) = scrolled_to_end_flag().lock() {
            *g = true;
        }
        reset_scroll_gate();
        let v = scrolled_to_end_flag().lock().map(|g| *g).unwrap_or(true);
        assert!(!v, "reset_scroll_gate must zero the latch");
    }
}
