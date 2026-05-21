use eframe::egui;
use neoethos_core::config::RiskConfig;
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

    // All three sliders previously displayed raw decimals like
    // "0.0400" for a 4 % limit, which was the most visible amateur
    // tell in the UI. The custom formatter renders the slider value
    // as a percentage string with one decimal so traders read
    // "4.0 %" / "10.0 %" / "3.0 %" — the same convention TradingView
    // and cTrader use throughout.
    ui.add(
        egui::Slider::new(&mut risk.daily_drawdown_limit, drawdown_slider_bounds())
            .custom_formatter(|v, _| format!("{:.1} %", v * 100.0))
            .text("Daily drawdown limit"),
    );
    ui.add(
        egui::Slider::new(
            &mut risk.total_drawdown_limit,
            total_drawdown_slider_bounds(),
        )
        .custom_formatter(|v, _| format!("{:.1} %", v * 100.0))
        .text("Total drawdown limit"),
    );
    ui.add(
        egui::Slider::new(&mut risk.risk_per_trade, risk_per_trade_slider_bounds())
            .custom_formatter(|v, _| format!("{:.2} %", v * 100.0))
            .text("Risk per trade"),
    );
    ui.add(
        egui::Slider::new(&mut risk.max_lot_size, lot_size_slider_bounds())
            .custom_formatter(|v, _| format!("{v:.2} lots"))
            .text("Max lot size"),
    );
    ui.checkbox(&mut risk.require_stop_loss, "Require stop-loss (prop firm)");
}
