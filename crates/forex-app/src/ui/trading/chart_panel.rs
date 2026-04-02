use crate::app_services::trading::{ChartCandle, MarketChartSnapshot, TradingSession};
use crate::app_state::AppState;
use crate::ui::theme;
use eframe::egui;

#[derive(Debug, Clone, PartialEq)]
pub struct ChartPanel {
    pub symbol: String,
    pub timeframe: String,
    pub available_timeframes: Vec<String>,
    pub candle_count: usize,
    pub headline: String,
    pub overlay_status: String,
    pub warnings: Vec<String>,
}

pub fn build_chart_panel(snapshot: &MarketChartSnapshot) -> ChartPanel {
    ChartPanel {
        symbol: snapshot.symbol.clone(),
        timeframe: snapshot.timeframe.clone(),
        available_timeframes: snapshot.available_timeframes.clone(),
        candle_count: snapshot.candles.len(),
        headline: snapshot.headline.clone(),
        overlay_status: snapshot.overlay_status.clone(),
        warnings: snapshot.warnings.clone(),
    }
}

pub fn render(ui: &mut egui::Ui, state: &mut AppState, session: &mut TradingSession) {
    let mut snapshot = session.market_chart_snapshot(state);
    if snapshot.timeframe != state.chart_timeframe {
        state.chart_timeframe = snapshot.timeframe.clone();
    }
    let panel = build_chart_panel(&snapshot);

    ui.vertical(|ui| {
        ui.horizontal(|ui| {
            ui.strong(format!("{} Market Chart", panel.symbol));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                for timeframe in panel.available_timeframes.iter().rev() {
                    let selected = timeframe == &state.chart_timeframe;
                    if ui
                        .add_sized(
                            [56.0, 24.0],
                            egui::Button::new(
                                egui::RichText::new(timeframe).color(theme::TEXT_PRIMARY),
                            )
                            .selected(selected),
                        )
                        .clicked()
                    {
                        state.chart_timeframe = timeframe.clone();
                        snapshot = session.market_chart_snapshot(state);
                    }
                }
            });
        });

        ui.add_space(4.0);
        ui.label(egui::RichText::new(panel.headline.clone()).color(theme::TEXT_MUTED));

        ui.add_space(8.0);
        let desired = egui::vec2(ui.available_width(), 320.0);
        let (rect, _response) = ui.allocate_exact_size(desired, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 18.0, theme::SURFACE_BG);
        painter.rect_stroke(
            rect,
            18.0,
            egui::Stroke::new(1.0, theme::BORDER),
            egui::StrokeKind::Outside,
        );

        if snapshot.candles.is_empty() {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "No market candles available for this symbol/timeframe.",
                egui::TextStyle::Body.resolve(ui.style()),
                theme::TEXT_MUTED,
            );
        } else {
            paint_grid(ui, &painter, rect);
            paint_candles(
                &painter,
                rect,
                &snapshot.candles,
                snapshot.price_min,
                snapshot.price_max,
            );
            paint_overlays(ui, &painter, rect, &snapshot);
        }

        ui.add_space(8.0);
        ui.label(egui::RichText::new(panel.overlay_status).color(theme::TEXT_MUTED));
        for warning in panel.warnings {
            ui.label(egui::RichText::new(warning).color(theme::WARNING));
        }
    });
}

fn paint_grid(ui: &egui::Ui, painter: &egui::Painter, rect: egui::Rect) {
    let _ = ui;
    let grid_color = egui::Color32::from_white_alpha(18);
    for idx in 1..6 {
        let y = egui::lerp(rect.top()..=rect.bottom(), idx as f32 / 6.0);
        painter.line_segment(
            [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
            egui::Stroke::new(1.0, grid_color),
        );
    }
    for idx in 1..10 {
        let x = egui::lerp(rect.left()..=rect.right(), idx as f32 / 10.0);
        painter.line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
            egui::Stroke::new(1.0, grid_color),
        );
    }
}

fn paint_candles(
    painter: &egui::Painter,
    rect: egui::Rect,
    candles: &[ChartCandle],
    price_min: f64,
    price_max: f64,
) {
    let chart_width = rect.width();
    let candle_width = (chart_width / candles.len().max(1) as f32).clamp(4.0, 10.0);
    let body_width = candle_width * 0.58;
    let price_span = (price_max - price_min).max(0.000_000_1);

    for (idx, candle) in candles.iter().enumerate() {
        let center_x = rect.left() + candle_width * (idx as f32 + 0.5);
        let high_y = map_price_to_y(rect, candle.high, price_min, price_span);
        let low_y = map_price_to_y(rect, candle.low, price_min, price_span);
        let open_y = map_price_to_y(rect, candle.open, price_min, price_span);
        let close_y = map_price_to_y(rect, candle.close, price_min, price_span);
        let color = if candle.close >= candle.open {
            theme::SUCCESS
        } else {
            theme::DANGER
        };

        painter.line_segment(
            [egui::pos2(center_x, high_y), egui::pos2(center_x, low_y)],
            egui::Stroke::new(1.0, color),
        );

        let top = open_y.min(close_y);
        let bottom = open_y.max(close_y).max(top + 1.5);
        let body = egui::Rect::from_center_size(
            egui::pos2(center_x, (top + bottom) / 2.0),
            egui::vec2(body_width, (bottom - top).max(2.0)),
        );
        painter.rect_filled(body, 2.0, color.linear_multiply(0.65));
        painter.rect_stroke(
            body,
            2.0,
            egui::Stroke::new(1.0, color),
            egui::StrokeKind::Outside,
        );
    }
}

fn paint_overlays(
    ui: &egui::Ui,
    painter: &egui::Painter,
    rect: egui::Rect,
    snapshot: &MarketChartSnapshot,
) {
    if snapshot.candles.is_empty() {
        return;
    }

    let price_span = (snapshot.price_max - snapshot.price_min).max(0.000_000_1);
    let candle_width = (rect.width() / snapshot.candles.len().max(1) as f32).clamp(4.0, 10.0);
    for overlay in &snapshot.overlays {
        if overlay.candle_index >= snapshot.candles.len() {
            continue;
        }
        let center_x = rect.left() + candle_width * (overlay.candle_index as f32 + 0.5);
        let price_y = map_price_to_y(rect, overlay.price, snapshot.price_min, price_span);
        painter.circle_filled(egui::pos2(center_x, price_y), 5.0, theme::ACCENT);
        painter.text(
            egui::pos2(center_x + 8.0, price_y - 10.0),
            egui::Align2::LEFT_CENTER,
            &overlay.label,
            egui::TextStyle::Small.resolve(ui.style()),
            theme::TEXT_PRIMARY,
        );
    }
}

fn map_price_to_y(rect: egui::Rect, price: f64, price_min: f64, price_span: f64) -> f32 {
    let normalized = ((price - price_min) / price_span).clamp(0.0, 1.0) as f32;
    rect.bottom() - normalized * rect.height()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::trading::{ChartOverlay, MarketChartSnapshot};

    #[test]
    fn chart_panel_uses_market_snapshot_summary_and_overlay_status() {
        let snapshot = MarketChartSnapshot {
            symbol: "EURUSD".to_string(),
            timeframe: "M5".to_string(),
            available_timeframes: vec!["M1".to_string(), "M5".to_string()],
            candles: vec![ChartCandle {
                timestamp: Some(1),
                open: 1.1,
                high: 1.2,
                low: 1.0,
                close: 1.15,
            }],
            overlays: vec![ChartOverlay {
                label: "BOT BUY".to_string(),
                candle_index: 0,
                price: 1.15,
            }],
            price_min: 1.0,
            price_max: 1.2,
            headline: "1 candles · latest close 1.15000 · range 1.00000-1.20000".to_string(),
            overlay_status: "Trade overlays will appear here once execution events are available."
                .to_string(),
            warnings: vec!["Execution timeline unavailable".to_string()],
        };

        let panel = build_chart_panel(&snapshot);

        assert_eq!(panel.symbol, "EURUSD");
        assert_eq!(panel.timeframe, "M5");
        assert_eq!(panel.candle_count, 1);
        assert!(panel.headline.contains("latest close 1.15000"));
        assert!(panel.overlay_status.contains("Trade overlays"));
        assert_eq!(panel.warnings, vec!["Execution timeline unavailable"]);
    }
}
