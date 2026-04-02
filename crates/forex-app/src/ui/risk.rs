use eframe::egui;
use forex_core::config::RiskConfig;
use std::ops::RangeInclusive;

pub fn drawdown_slider_bounds() -> RangeInclusive<f64> {
    0.01..=0.20
}

pub fn total_drawdown_slider_bounds() -> RangeInclusive<f64> {
    0.05..=0.50
}

pub fn risk_per_trade_slider_bounds() -> RangeInclusive<f64> {
    0.005..=0.10
}

pub fn lot_size_slider_bounds() -> RangeInclusive<f64> {
    0.01..=50.0
}

pub fn render(ui: &mut egui::Ui, risk: &mut RiskConfig) {
    ui.heading("Prop-Firm Risk Guard");
    ui.separator();
    ui.add(
        egui::Slider::new(&mut risk.daily_drawdown_limit, drawdown_slider_bounds())
            .text("Daily Drawdown Limit (%)"),
    );
    ui.add(
        egui::Slider::new(
            &mut risk.total_drawdown_limit,
            total_drawdown_slider_bounds(),
        )
        .text("Total Drawdown Limit (%)"),
    );
    ui.add(
        egui::Slider::new(&mut risk.risk_per_trade, risk_per_trade_slider_bounds())
            .text("Risk Per Trade (Ratio)"),
    );
    ui.add(
        egui::Slider::new(&mut risk.max_lot_size, lot_size_slider_bounds()).text("Max Lot Size"),
    );
    ui.checkbox(&mut risk.require_stop_loss, "Require Stop-Loss (Prop Firm)");
}
