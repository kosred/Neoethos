//! Step 9 — Auto-start at login.
//!
//! Spec: `installer_wizard_ux_spec.md` §2 Step 9.
//!
//! Per-platform artefact paths:
//! - Linux: `~/.config/autostart/neoethos-app.desktop` (freedesktop
//!   Autostart Specification).
//! - macOS: `~/Library/LaunchAgents/ai.forex.app.plist`.
//! - Windows: per-user shortcut in
//!   `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\` or
//!   `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`.
//!
//! All three paths are user-scoped — no UAC / sudo prompt.

use eframe::egui;
use std::path::PathBuf;

use super::{StepResult, WizardController};
use crate::ui::theme;

pub const WIZARD_DEFAULT_AUTOSTART_ENABLED: bool = false;
pub const WIZARD_DEFAULT_START_MINIMIZED: bool = false;

/// Linux .desktop filename. Spec §2 Step 9.
/// `#[allow(dead_code)]`: Linux build of the Apply writer + packaging
/// metadata in `Cargo.toml [package.metadata.deb]` reads this name; the
/// Rust compiler in a cross-platform release build can't see those
/// non-Rust call sites.
#[allow(dead_code)]
pub const WIZARD_DEFAULT_LINUX_AUTOSTART_FILENAME: &str = "neoethos-app.desktop";

/// macOS plist filename. Spec §2 Step 9.
#[allow(dead_code)] // macOS-only — see Linux sibling const above
pub const WIZARD_DEFAULT_MACOS_LAUNCHAGENT_FILENAME: &str = "ai.forex.app.plist";

/// Windows shortcut filename inside the per-user Startup folder.
pub const WIZARD_DEFAULT_WINDOWS_AUTOSTART_FILENAME: &str = "neoethos-app.lnk";

/// Path the wizard would write for the current platform. Returns
/// `None` if the per-user config dir can't be resolved.
pub fn target_autostart_path() -> Option<PathBuf> {
    #[cfg(target_os = "linux")]
    {
        return dirs::config_dir().map(|d| {
            d.join("autostart")
                .join(WIZARD_DEFAULT_LINUX_AUTOSTART_FILENAME)
        });
    }
    #[cfg(target_os = "macos")]
    {
        return dirs::home_dir().map(|d| {
            d.join("Library/LaunchAgents")
                .join(WIZARD_DEFAULT_MACOS_LAUNCHAGENT_FILENAME)
        });
    }
    #[cfg(target_os = "windows")]
    {
        return dirs::config_dir().map(|d| {
            d.join("Microsoft/Windows/Start Menu/Programs/Startup")
                .join(WIZARD_DEFAULT_WINDOWS_AUTOSTART_FILENAME)
        });
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

pub fn render(ui: &mut egui::Ui, controller: &mut WizardController) -> StepResult {
    let mut result = StepResult::StayHere;

    ui.checkbox(
        &mut controller.config.autostart_enabled,
        "Start neoethos-app on system login.",
    );
    if controller.config.autostart_enabled {
        ui.checkbox(
            &mut controller.config.start_minimized,
            "Start minimised to system tray.",
        );
    } else {
        // Reset minimised flag if auto-start is toggled off.
        controller.config.start_minimized = false;
    }

    if let Some(path) = target_autostart_path() {
        ui.label(
            egui::RichText::new(format!("Artefact will be written to: {}", path.display()))
                .size(theme::FONT_CAPTION)
                .color(theme::TEXT_MUTED),
        );
    } else {
        ui.label(
            egui::RichText::new(
                "Cannot determine the autostart path for this platform; the wizard will \
                 surface a warning at Apply.",
            )
            .size(theme::FONT_CAPTION)
            .color(theme::WARNING),
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
    fn autostart_default_is_off() {
        assert!(!WIZARD_DEFAULT_AUTOSTART_ENABLED);
        assert!(!WIZARD_DEFAULT_START_MINIMIZED);
    }

    #[test]
    fn autostart_step_advances_to_autonomy_risk() {
        let mut c = WizardController::new();
        c.current = WizardState::Autostart;
        c.apply(StepResult::NextRequested);
        assert_eq!(c.current, WizardState::AutonomyRisk);
    }

    #[test]
    fn linux_filename_uses_desktop_extension() {
        assert!(WIZARD_DEFAULT_LINUX_AUTOSTART_FILENAME.ends_with(".desktop"));
    }
}
