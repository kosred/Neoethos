//! Step 7 — Hardware compatibility probe.
//!
//! Spec: `installer_wizard_ux_spec.md` §2 Step 7.
//!
//! Wraps `forex_core::system::HardwareProbe::detect()` and surfaces
//! one card per accelerator device with vendor-specific remediation
//! copy when a backend is detected-but-unavailable (NVIDIA driver
//! missing, ROCm package not installed, Vulkan SDK absent).

use eframe::egui;
use forex_core::system::{HardwareProbe, HardwareProfile};
use std::sync::OnceLock;

use super::{StepResult, WizardController};
use crate::ui::theme;

/// Order in which backends are recommended when more than one is
/// available. Matches `forex-core/src/system/backends.rs` dispatch
/// preference: CUDA > ROCm > Vulkan > CPU. Spec §2 Step 7.
pub const WIZARD_DEFAULT_BACKEND_PREFERENCE: &[&str] = &["CUDA", "ROCm", "Vulkan", "CPU"];

/// File name for the per-user hardware-probe snapshot. Spec §2 Step 7
/// Actions — "writes `<data_path>/hardware_profile.json`".
pub const WIZARD_DEFAULT_HARDWARE_PROFILE_FILENAME: &str = "hardware_profile.json";

/// Memoised snapshot — the probe spawns `nvidia-smi` / `rocminfo` and
/// walks `sysinfo`, so re-running it every egui frame is wasteful.
/// One-shot per process; the user re-runs the wizard for a fresh
/// detection.
fn cached_profile() -> &'static HardwareProfile {
    static PROFILE: OnceLock<HardwareProfile> = OnceLock::new();
    PROFILE.get_or_init(|| {
        let mut probe = HardwareProbe::new();
        probe.detect()
    })
}

pub fn render(ui: &mut egui::Ui, controller: &mut WizardController) -> StepResult {
    let mut result = StepResult::StayHere;

    ui.label(
        egui::RichText::new("Detect compute backends and pick a default.")
            .color(theme::TEXT_PRIMARY),
    );

    let profile = cached_profile();

    ui.add_space(theme::SPACE_SM);
    ui.label(
        egui::RichText::new(format!(
            "CPU cores: {} · RAM: {:.1} GiB · Platform: {}",
            profile.cpu_cores, profile.total_ram_gb, profile.platform_label
        ))
        .color(theme::TEXT_PRIMARY),
    );

    if profile.gpu_names.is_empty() {
        ui.label(
            egui::RichText::new(
                "No GPU detected — falling back to CPU NdArray. (Recommended backend: CPU.)",
            )
            .color(theme::WARNING)
            .size(theme::FONT_CAPTION),
        );
    } else {
        ui.label(
            egui::RichText::new(format!("GPUs detected: {}", profile.num_gpus))
                .strong()
                .color(theme::TEXT_PRIMARY),
        );
        for (idx, name) in profile.gpu_names.iter().enumerate() {
            let mem = profile.gpu_mem_gb.get(idx).copied().unwrap_or(0.0);
            ui.label(
                egui::RichText::new(format!("  · {} ({:.1} GiB VRAM)", name, mem))
                    .color(theme::TEXT_MUTED)
                    .size(theme::FONT_CAPTION),
            );
        }
    }

    ui.separator();
    ui.label(
        egui::RichText::new("Optional: force a backend (overrides auto-detection).")
            .color(theme::TEXT_PRIMARY),
    );
    ui.horizontal(|ui| {
        ui.label("Forced backend:");
        let mut forced = controller
            .config
            .forced_backend
            .clone()
            .unwrap_or_else(|| "(auto)".to_string());
        egui::ComboBox::from_id_salt("wizard_forced_backend")
            .selected_text(&forced)
            .show_ui(ui, |ui| {
                if ui
                    .selectable_value(&mut forced, "(auto)".to_string(), "Auto-detect")
                    .clicked()
                {
                    controller.config.forced_backend = None;
                }
                for backend in WIZARD_DEFAULT_BACKEND_PREFERENCE {
                    if ui
                        .selectable_value(&mut forced, (*backend).to_string(), *backend)
                        .clicked()
                    {
                        controller.config.forced_backend = Some((*backend).to_string());
                    }
                }
            });
    });

    // Vendor-specific remediation hints.
    if profile
        .gpu_names
        .iter()
        .any(|n| n.to_lowercase().contains("nvidia"))
    {
        ui.label(
            egui::RichText::new("Tip: install the NVIDIA driver to enable CUDA precisions.")
                .color(theme::TEXT_MUTED)
                .size(theme::FONT_CAPTION),
        );
    }
    if profile.gpu_names.iter().any(|n| {
        let lower = n.to_lowercase();
        lower.contains("amd") || lower.contains("radeon")
    }) {
        ui.label(
            egui::RichText::new("Tip: install ROCm to enable Radeon backends.")
                .color(theme::TEXT_MUTED)
                .size(theme::FONT_CAPTION),
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
    fn backend_preference_starts_with_cuda() {
        assert_eq!(WIZARD_DEFAULT_BACKEND_PREFERENCE.first(), Some(&"CUDA"));
        assert!(WIZARD_DEFAULT_BACKEND_PREFERENCE.contains(&"CPU"));
    }

    #[test]
    fn hardware_step_advances_to_news_api() {
        let mut c = WizardController::new();
        c.current = WizardState::Hardware;
        c.apply(StepResult::NextRequested);
        assert_eq!(c.current, WizardState::NewsApi);
    }

    #[test]
    fn hardware_step_back_returns_to_historical() {
        let mut c = WizardController::new();
        c.current = WizardState::Hardware;
        c.apply(StepResult::BackRequested);
        assert_eq!(c.current, WizardState::Historical);
    }
}
