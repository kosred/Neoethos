//! Step 2 — Path selection.
//!
//! Spec: `installer_wizard_ux_spec.md` §2 Step 2.
//!
//! Defaults to OS-canonical paths via the `directories` crate (already
//! a workspace dependency via `dirs`):
//! - Windows: `%LOCALAPPDATA%\forex-ai\`
//! - macOS:   `~/Library/Application Support/forex-ai/`
//! - Linux:   `$XDG_DATA_HOME/forex-ai/` (i.e. `~/.local/share/forex-ai/`)
//!
//! On entry the wizard also runs `migration::detect_portable_install`
//! to surface the legacy `~/.forex-ai/` layout (spec §6).

use eframe::egui;
use std::path::PathBuf;
use std::sync::OnceLock;

use super::{StepResult, WizardController, migration};
use crate::ui::theme;

/// Spec §3 — amber disk-free banner triggers below this many GiB.
pub const WIZARD_DEFAULT_DISK_FREE_AMBER_GIB: u64 = 20;
/// Spec §3 — red disk-free banner triggers below this many GiB.
pub const WIZARD_DEFAULT_DISK_FREE_RED_GIB: u64 = 5;
/// Folder name appended under the OS-canonical root.
pub const WIZARD_DEFAULT_DATA_FOLDER: &str = "forex-ai";

/// Resolve the OS-canonical default data path. Returns `None` if the
/// crate can't determine a per-user directory (e.g. on a stripped-down
/// container).
pub fn default_data_path() -> Option<PathBuf> {
    dirs::data_local_dir().map(|d| d.join(WIZARD_DEFAULT_DATA_FOLDER))
}

/// Test seam — when set, `pick_folder_via_rfd` returns the override
/// instead of opening the real native dialog. Lets the unit tests
/// drive the wiring without needing a windowing system.
#[cfg(test)]
fn rfd_override() -> &'static std::sync::Mutex<Option<PathBuf>> {
    static OVERRIDE: std::sync::OnceLock<std::sync::Mutex<Option<PathBuf>>> =
        std::sync::OnceLock::new();
    OVERRIDE.get_or_init(|| std::sync::Mutex::new(None))
}

/// Open the native folder-picker. Sync API per `rfd 0.17.2`
/// (`src/file_dialog.rs:132`: `pub fn pick_folder(self) -> Option<PathBuf>`).
/// Returns `None` on user-cancel and also on platforms where rfd
/// silently can't open a dialog (headless Linux, no-GTK build,
/// SSH session without DISPLAY) — caller logs + falls back to the
/// text-box flow.
fn pick_folder_via_rfd(seed: &std::path::Path) -> Option<PathBuf> {
    #[cfg(test)]
    {
        if let Ok(g) = rfd_override().lock() {
            if let Some(p) = g.as_ref() {
                return Some(p.clone());
            }
        }
    }
    rfd::FileDialog::new()
        .set_title("Choose forex-ai data folder")
        .set_directory(seed)
        .pick_folder()
}

pub fn render(ui: &mut egui::Ui, controller: &mut WizardController) -> StepResult {
    let mut result = StepResult::StayHere;

    if controller.config.data_path.is_none() {
        controller.config.data_path = default_data_path();
    }

    ui.label(
        egui::RichText::new("Choose where forex-ai will store data, models, and OAuth tokens.")
            .color(theme::TEXT_PRIMARY),
    );
    ui.add_space(theme::SPACE_SM);

    // Show the resolved default + an editable text box so the user
    // can override. A real implementation drives this with the
    // `rfd` folder picker (already a dependency).
    let mut path_string = controller
        .config
        .data_path
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    ui.horizontal(|ui| {
        ui.label("Data directory:");
        let response = ui.add(egui::TextEdit::singleline(&mut path_string).desired_width(420.0));
        if response.changed() {
            controller.config.data_path = if path_string.is_empty() {
                None
            } else {
                Some(PathBuf::from(&path_string))
            };
        }
        if ui.button("Browse…").clicked() {
            // Native folder-picker via `rfd 0.17` (verified against the
            // crate's `src/file_dialog.rs` — sync `FileDialog::new`
            // → `.set_title(..)` → `.set_directory(..)` →
            // `.pick_folder() -> Option<PathBuf>`). On Linux without
            // GTK / on headless SSH sessions rfd's `pick_folder`
            // returns `None`; we surface a hint instead of panicking
            // so the operator can fall back to the text box.
            let seed = controller
                .config
                .data_path
                .clone()
                .or_else(default_data_path)
                .unwrap_or_else(|| std::path::PathBuf::from("."));
            match pick_folder_via_rfd(&seed) {
                Some(folder) => {
                    controller.config.data_path = Some(folder);
                }
                None => {
                    tracing::info!(
                        target: "forex_app::wizard",
                        "rfd::pick_folder returned None (cancelled or no GUI environment)"
                    );
                }
            }
        }
    });

    // Spec §6 — surface portable-install migration banner. Memoised
    // because filesystem-walking the candidate roots every frame is
    // wasteful (and racy on a network home dir).
    static PORTABLE_REPORT: OnceLock<Option<migration::PortableMigrationReport>> = OnceLock::new();
    let portable = PORTABLE_REPORT.get_or_init(migration::detect_portable_install);
    if let Some(report) = portable {
        ui.separator();
        ui.label(
            egui::RichText::new("Legacy ~/.forex-ai detected")
                .strong()
                .color(theme::WARNING),
        );
        for line in report.summary_lines() {
            ui.label(
                egui::RichText::new(line)
                    .size(theme::FONT_CAPTION)
                    .color(theme::TEXT_MUTED),
            );
        }
        ui.label(
            egui::RichText::new(
                "Migration UI is rendered here when wired; for now the wizard surfaces \
                 detection only.",
            )
            .size(theme::FONT_CAPTION)
            .color(theme::TEXT_FAINT),
        );
    }

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
    fn default_data_path_ends_in_forex_ai_when_resolvable() {
        if let Some(p) = default_data_path() {
            assert!(
                p.ends_with(WIZARD_DEFAULT_DATA_FOLDER),
                "default path must end in folder name, got {:?}",
                p
            );
        }
    }

    #[test]
    fn path_step_advances_on_next() {
        let mut c = WizardController::new();
        c.current = WizardState::Path;
        c.apply(StepResult::NextRequested);
        assert_eq!(c.current, WizardState::AccountProfile);
    }

    #[test]
    fn path_step_back_returns_to_welcome() {
        let mut c = WizardController::new();
        c.current = WizardState::Path;
        c.apply(StepResult::BackRequested);
        assert_eq!(c.current, WizardState::Welcome);
    }

    #[test]
    fn path_step_skip_records_in_skipped_steps() {
        let mut c = WizardController::new();
        c.current = WizardState::Path;
        c.apply(StepResult::SkipRequested);
        assert!(c.state_file.skipped_steps.contains(&WizardState::Path));
        assert_eq!(c.current, WizardState::AccountProfile);
    }

    #[test]
    fn disk_thresholds_keep_operator_default_order() {
        assert!(WIZARD_DEFAULT_DISK_FREE_RED_GIB < WIZARD_DEFAULT_DISK_FREE_AMBER_GIB);
    }

    #[test]
    fn rfd_picker_returns_override_in_tests() {
        // Drive the rfd wrapper without opening a real dialog — set
        // the test override, call the wrapper, assert it returned
        // the seeded value. The headless CI runner has no native
        // dialog backend, so this is the only way to keep the
        // wiring under test.
        let probe = std::env::temp_dir().join("forex-ai-rfd-test-path");
        *rfd_override().lock().unwrap() = Some(probe.clone());
        let picked = pick_folder_via_rfd(std::path::Path::new("."));
        *rfd_override().lock().unwrap() = None;
        assert_eq!(picked, Some(probe));
    }

    #[test]
    fn rfd_override_unset_by_default() {
        // Sanity — if a previous test leaks the override, this test
        // would fail. Keeps the seam from being silently sticky.
        let g = rfd_override().lock().unwrap();
        assert!(g.is_none(), "test override leaked across tests");
    }
}
