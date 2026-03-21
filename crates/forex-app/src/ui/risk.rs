use crate::app_state::RiskState;
use eframe::egui;
use std::ops::RangeInclusive;

pub fn drawdown_slider_bounds() -> RangeInclusive<f32> {
    0.1..=10.0
}

pub fn lot_size_slider_bounds() -> RangeInclusive<f32> {
    0.01..=50.0
}

pub fn render(ui: &mut egui::Ui, risk: &mut RiskState) {
    ui.heading("Prop-Firm Risk Guard");
    ui.separator();
    ui.add(
        egui::Slider::new(&mut risk.daily_drawdown_limit, drawdown_slider_bounds())
            .text("Daily Drawdown Limit (%)"),
    );
    ui.add(
        egui::Slider::new(&mut risk.max_lot_size, lot_size_slider_bounds())
            .text("Max Lot Size"),
    );
}

