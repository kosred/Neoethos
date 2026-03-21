use crate::app_state::HardwareState;
use eframe::egui;
use std::ops::RangeInclusive;

pub fn cpu_slider_bounds() -> RangeInclusive<i32> {
    1..=252
}

pub fn render(ui: &mut egui::Ui, hardware: &mut HardwareState) {
    ui.heading("Hardware Allocation");
    ui.separator();
    ui.add(egui::Slider::new(&mut hardware.cpu_cores, cpu_slider_bounds()).text("CPU Cores"));
    ui.checkbox(&mut hardware.gpu_enabled, "Enable GPU Acceleration (CUDA)");
}

