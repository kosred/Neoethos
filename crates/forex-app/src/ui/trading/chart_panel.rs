use crate::app_services::trading::{MarketChartSnapshot, TradingSession};
use crate::app_services::ServiceEvent;
use crate::app_state::AppState;
use crate::ui::theme;
use eframe::egui;
use tokio::sync::mpsc;

const PRICE_AXIS_WIDTH: f32 = 72.0;
const VOLUME_PANEL_RATIO: f32 = 0.18; // volume takes bottom 18% of chart area

pub fn render(
    ui: &mut egui::Ui,
    state: &mut AppState,
    session: &mut TradingSession,
    tx: &mpsc::Sender<ServiceEvent>,
) {
    let mut snapshot = session.market_chart_snapshot(state, Some(tx));
    if snapshot.timeframe != state.chart_timeframe {
        state.chart_timeframe = snapshot.timeframe.clone();
    }

    ui.vertical(|ui| {
        // ── Instrument header ───────────────────────────────────────
        render_symbol_header(ui, &snapshot);
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            ui.spacing_mut().button_padding = egui::vec2(7.0, 3.0);
            for timeframe in snapshot.available_timeframes.clone() {
                let selected = timeframe == state.chart_timeframe;
                let text = egui::RichText::new(&timeframe)
                    .size(11.0)
                    .color(if selected {
                        theme::ACCENT
                    } else {
                        theme::TEXT_MUTED
                    });
                let btn = egui::Button::new(text)
                    .selected(selected)
                    .corner_radius(egui::CornerRadius::same(3));
                if ui.add_sized([32.0, 18.0], btn).clicked() {
                    state.chart_timeframe = timeframe.clone();
                    snapshot = session.market_chart_snapshot(state, Some(tx));
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    egui::RichText::new(&snapshot.headline)
                        .size(10.5)
                        .color(theme::TEXT_MUTED),
                );
            });
        });
        ui.add_space(2.0);

        // ── Main chart area ─────────────────────────────────────────
        let total_h = (ui.available_height() - 8.0).max(200.0);
        let candle_h = total_h * (1.0 - VOLUME_PANEL_RATIO);
        let vol_h = total_h * VOLUME_PANEL_RATIO;
        let total_w = ui.available_width();
        let chart_w = total_w - PRICE_AXIS_WIDTH;

        let outer_rect = ui
            .allocate_exact_size(egui::vec2(total_w, total_h), egui::Sense::hover())
            .0;
        let painter = ui.painter_at(outer_rect);
        painter.rect_filled(outer_rect, 0.0, theme::CHART_BG);

        let candle_rect = egui::Rect::from_min_size(outer_rect.min, egui::vec2(chart_w, candle_h));
        let vol_rect = egui::Rect::from_min_size(
            egui::pos2(outer_rect.left(), outer_rect.top() + candle_h),
            egui::vec2(chart_w, vol_h),
        );
        let price_axis_rect = egui::Rect::from_min_size(
            egui::pos2(outer_rect.left() + chart_w, outer_rect.top()),
            egui::vec2(PRICE_AXIS_WIDTH, total_h),
        );

        if snapshot.candles.is_empty() {
            // Grid lines even when empty
            paint_grid(&painter, candle_rect);
            let cx = outer_rect.center();
            painter.text(
                egui::pos2(cx.x, cx.y - 18.0),
                egui::Align2::CENTER_CENTER,
                format!("{}  ·  {}", snapshot.symbol, state.chart_timeframe),
                egui::FontId::new(15.0, egui::FontFamily::Proportional),
                theme::TEXT_PRIMARY,
            );
            painter.text(
                egui::pos2(cx.x, cx.y + 4.0),
                egui::Align2::CENTER_CENTER,
                "No candles - connect to cTrader or load local data",
                egui::TextStyle::Small.resolve(ui.style()),
                theme::TEXT_MUTED,
            );
        } else {
            paint_grid(&painter, candle_rect);
            paint_candles(&painter, candle_rect, &snapshot);
            paint_volume_bars(&painter, vol_rect, &snapshot);
            paint_price_axis(ui, &painter, price_axis_rect, &snapshot);
            paint_overlays(ui, &painter, candle_rect, &snapshot);
        }

        // separator between candles and volume
        painter.line_segment(
            [
                egui::pos2(candle_rect.left(), candle_rect.bottom()),
                egui::pos2(candle_rect.right(), candle_rect.bottom()),
            ],
            egui::Stroke::new(1.0, theme::BORDER),
        );

        // bid/ask horizontal lines
        if !snapshot.candles.is_empty() {
            let price_span = (snapshot.price_max - snapshot.price_min).max(1e-7);
            if let Some(ask) = snapshot.ask {
                let y = price_y(candle_rect, ask, snapshot.price_min, price_span);
                painter.line_segment(
                    [
                        egui::pos2(candle_rect.left(), y),
                        egui::pos2(candle_rect.right(), y),
                    ],
                    egui::Stroke::new(1.0, theme::SUCCESS.linear_multiply(0.7)),
                );
            }
            if let Some(bid) = snapshot.bid {
                let y = price_y(candle_rect, bid, snapshot.price_min, price_span);
                painter.line_segment(
                    [
                        egui::pos2(candle_rect.left(), y),
                        egui::pos2(candle_rect.right(), y),
                    ],
                    egui::Stroke::new(1.0, theme::DANGER.linear_multiply(0.7)),
                );
            }
        }

        // Warnings
        for warning in &snapshot.warnings {
            ui.label(egui::RichText::new(warning).color(theme::WARNING).small());
        }
    });
}

/// Top info bar — TradingView-style: symbol large, OHLC inline, bid/ask/spread right-aligned
fn render_symbol_header(ui: &mut egui::Ui, snapshot: &MarketChartSnapshot) {
    let last = snapshot.candles.last();
    let (last_close, last_open) = last.map(|c| (c.close, c.open)).unwrap_or((0.0, 0.0));
    let is_up = last_close >= last_open;
    let price_color = if is_up { theme::SUCCESS } else { theme::DANGER };

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;

        // Symbol name — bold, prominent
        ui.strong(
            egui::RichText::new(&snapshot.symbol)
                .size(14.0)
                .color(theme::TEXT_PRIMARY),
        );

        if let Some(c) = last {
            // Last price — colored + change%
            ui.label(
                egui::RichText::new(format!("{:.5}", c.close))
                    .size(14.0)
                    .color(price_color)
                    .strong(),
            );

            if let Some(pct) = snapshot.price_change_pct {
                let sign = if pct >= 0.0 { "+" } else { "" };
                ui.label(
                    egui::RichText::new(format!("{}{:.2}%", sign, pct))
                        .size(11.0)
                        .color(price_color),
                );
            }

            // OHLC inline (compact)
            ui.add(egui::Separator::default().vertical().spacing(4.0));
            for (lbl, val) in [("O", c.open), ("H", c.high), ("L", c.low), ("C", c.close)] {
                ui.label(egui::RichText::new(lbl).size(10.0).color(theme::TEXT_MUTED));
                ui.label(
                    egui::RichText::new(format!("{:.5}", val))
                        .size(11.0)
                        .color(theme::TEXT_PRIMARY),
                );
            }

            // Bid / Ask / Spread — right-aligned
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                match (snapshot.bid, snapshot.ask) {
                    (Some(bid), Some(ask)) => {
                        let spread_pips = ((ask - bid) * 100_000.0).abs();
                        ui.label(
                            egui::RichText::new(format!("{:.1}p", spread_pips))
                                .size(10.0)
                                .color(theme::TEXT_MUTED),
                        );
                        ui.label(
                            egui::RichText::new(format!("{:.5}", ask))
                                .size(11.0)
                                .color(theme::SUCCESS),
                        );
                        ui.label(egui::RichText::new("A").size(10.0).color(theme::TEXT_MUTED));
                        ui.label(
                            egui::RichText::new(format!("{:.5}", bid))
                                .size(11.0)
                                .color(theme::DANGER),
                        );
                        ui.label(egui::RichText::new("B").size(10.0).color(theme::TEXT_MUTED));
                    }
                    _ => {
                        ui.label(
                            egui::RichText::new("No live quote")
                                .size(10.0)
                                .color(theme::TEXT_MUTED),
                        );
                    }
                }
            });
        } else {
            ui.label(
                egui::RichText::new("No data — connect or load local")
                    .size(11.0)
                    .color(theme::TEXT_MUTED),
            );
        }
    });
}

fn paint_grid(painter: &egui::Painter, rect: egui::Rect) {
    let grid_color = theme::GRID.linear_multiply(0.75);
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

fn paint_candles(painter: &egui::Painter, rect: egui::Rect, snapshot: &MarketChartSnapshot) {
    let candles = &snapshot.candles;
    let price_span = (snapshot.price_max - snapshot.price_min).max(1e-7);
    let chart_width = rect.width();
    let candle_width = (chart_width / candles.len().max(1) as f32).clamp(2.0, 12.0);
    let body_width = (candle_width * 0.6).max(1.5);

    for (idx, candle) in candles.iter().enumerate() {
        let cx = rect.left() + candle_width * (idx as f32 + 0.5);
        let high_y = price_y(rect, candle.high, snapshot.price_min, price_span);
        let low_y = price_y(rect, candle.low, snapshot.price_min, price_span);
        let open_y = price_y(rect, candle.open, snapshot.price_min, price_span);
        let close_y = price_y(rect, candle.close, snapshot.price_min, price_span);

        let bullish = candle.close >= candle.open;
        let color = if bullish {
            theme::SUCCESS
        } else {
            theme::DANGER
        };

        // Wick
        painter.line_segment(
            [egui::pos2(cx, high_y), egui::pos2(cx, low_y)],
            egui::Stroke::new(1.0, color),
        );

        // Body
        let top = open_y.min(close_y);
        let bottom = open_y.max(close_y).max(top + 1.5);
        let body = egui::Rect::from_center_size(
            egui::pos2(cx, (top + bottom) / 2.0),
            egui::vec2(body_width, (bottom - top).max(2.0)),
        );
        painter.rect_filled(
            body,
            1.0,
            color.linear_multiply(if bullish { 0.6 } else { 0.75 }),
        );
        painter.rect_stroke(
            body,
            1.0,
            egui::Stroke::new(1.0, color),
            egui::StrokeKind::Outside,
        );
    }
}

fn paint_volume_bars(painter: &egui::Painter, rect: egui::Rect, snapshot: &MarketChartSnapshot) {
    let candles = &snapshot.candles;
    if candles.is_empty() {
        return;
    }
    let max_vol = candles
        .iter()
        .map(|c| c.volume)
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let chart_width = rect.width();
    let bar_width = (chart_width / candles.len().max(1) as f32).clamp(2.0, 12.0);
    let body_width = (bar_width * 0.6).max(1.0);

    for (idx, candle) in candles.iter().enumerate() {
        if candle.volume <= 0.0 {
            continue;
        }
        let cx = rect.left() + bar_width * (idx as f32 + 0.5);
        let ratio = (candle.volume / max_vol).clamp(0.0, 1.0) as f32;
        let bar_h = (rect.height() * ratio).max(1.0);
        let vol_rect = egui::Rect::from_min_size(
            egui::pos2(cx - body_width / 2.0, rect.bottom() - bar_h),
            egui::vec2(body_width, bar_h),
        );
        let bullish = candle.close >= candle.open;
        let color = if bullish {
            theme::SUCCESS
        } else {
            theme::DANGER
        };
        painter.rect_filled(vol_rect, 0.0, color.linear_multiply(0.4));
    }
}

fn paint_price_axis(
    ui: &egui::Ui,
    painter: &egui::Painter,
    rect: egui::Rect,
    snapshot: &MarketChartSnapshot,
) {
    let price_span = (snapshot.price_max - snapshot.price_min).max(1e-7);

    painter.rect_filled(rect, 0.0, theme::PANEL_BG);
    painter.line_segment(
        [
            egui::pos2(rect.left(), rect.top()),
            egui::pos2(rect.left(), rect.bottom()),
        ],
        egui::Stroke::new(1.0, theme::BORDER),
    );

    let candle_h = rect.height() * (1.0 - VOLUME_PANEL_RATIO);
    let candle_axis_rect = egui::Rect::from_min_size(rect.min, egui::vec2(rect.width(), candle_h));

    let steps = 5u32;
    for i in 0..=steps {
        let frac = i as f32 / steps as f32;
        let price = snapshot.price_min + (1.0 - frac as f64) * price_span;
        let y = egui::lerp(candle_axis_rect.top()..=candle_axis_rect.bottom(), frac);
        painter.line_segment(
            [egui::pos2(rect.left(), y), egui::pos2(rect.left() + 5.0, y)],
            egui::Stroke::new(1.0, theme::BORDER),
        );
        painter.text(
            egui::pos2(rect.left() + 8.0, y),
            egui::Align2::LEFT_CENTER,
            format!("{:.4}", price),
            egui::TextStyle::Small.resolve(ui.style()),
            theme::TEXT_MUTED,
        );
    }

    if let Some(last) = snapshot.candles.last() {
        let y = price_y(candle_axis_rect, last.close, snapshot.price_min, price_span);
        let bullish = last.close >= last.open;
        let color = if bullish {
            theme::SUCCESS
        } else {
            theme::DANGER
        };
        let label_rect = egui::Rect::from_min_size(
            egui::pos2(rect.left(), y - 9.0),
            egui::vec2(rect.width(), 18.0),
        );
        painter.rect_filled(label_rect, 3.0, color);
        painter.text(
            label_rect.center(),
            egui::Align2::CENTER_CENTER,
            format!("{:.4}", last.close),
            egui::TextStyle::Small.resolve(ui.style()),
            egui::Color32::WHITE,
        );
    }

    let bid_ask_colors = [
        (snapshot.ask, theme::SUCCESS, "A"),
        (snapshot.bid, theme::DANGER, "B"),
    ];
    for (price_opt, color, prefix) in bid_ask_colors {
        if let Some(price) = price_opt {
            let y = price_y(candle_axis_rect, price, snapshot.price_min, price_span);
            let label_rect = egui::Rect::from_min_size(
                egui::pos2(rect.left() + 4.0, y - 9.0),
                egui::vec2(rect.width() - 8.0, 18.0),
            );
            painter.rect_filled(label_rect, 3.0, color.linear_multiply(0.18));
            painter.rect_stroke(
                label_rect,
                3.0,
                egui::Stroke::new(1.0, color.linear_multiply(0.65)),
                egui::StrokeKind::Outside,
            );
            painter.text(
                label_rect.center(),
                egui::Align2::CENTER_CENTER,
                format!("{} {:.4}", prefix, price),
                egui::TextStyle::Small.resolve(ui.style()),
                color,
            );
        }
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
    let price_span = (snapshot.price_max - snapshot.price_min).max(1e-7);
    let candle_width = (rect.width() / snapshot.candles.len().max(1) as f32).clamp(2.0, 12.0);

    for overlay in &snapshot.overlays {
        if overlay.candle_index >= snapshot.candles.len() {
            continue;
        }
        let cx = rect.left() + candle_width * (overlay.candle_index as f32 + 0.5);
        let py = price_y(rect, overlay.price, snapshot.price_min, price_span);
        painter.circle_filled(egui::pos2(cx, py), 5.0, theme::ACCENT);
        painter.text(
            egui::pos2(cx + 8.0, py - 10.0),
            egui::Align2::LEFT_CENTER,
            &overlay.label,
            egui::TextStyle::Small.resolve(ui.style()),
            theme::TEXT_PRIMARY,
        );
    }
}

fn price_y(rect: egui::Rect, price: f64, price_min: f64, price_span: f64) -> f32 {
    let normalized = ((price - price_min) / price_span).clamp(0.0, 1.0) as f32;
    rect.bottom() - normalized * rect.height()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::trading::{ChartCandle, ChartOverlay, MarketChartSnapshot};

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
                volume: 1500.0,
            }],
            overlays: vec![ChartOverlay {
                label: "BOT BUY".to_string(),
                candle_index: 0,
                price: 1.15,
            }],
            price_min: 1.0,
            price_max: 1.2,
            bid: Some(1.14990),
            ask: Some(1.15010),
            price_change_pct: Some(0.45),
            headline: "1 candles · latest close 1.15000 · range 1.00000-1.20000".to_string(),
            overlay_status: "Trade overlays will appear here once execution events are available."
                .to_string(),
            warnings: vec!["Execution timeline unavailable".to_string()],
        };

        // Verify the snapshot carries all fields the chart renderer needs.
        assert_eq!(snapshot.symbol, "EURUSD");
        assert_eq!(snapshot.timeframe, "M5");
        assert_eq!(snapshot.candles.len(), 1);
        assert!(snapshot.headline.contains("latest close 1.15000"));
        assert!(snapshot.overlay_status.contains("Trade overlays"));
        assert_eq!(snapshot.warnings, vec!["Execution timeline unavailable"]);
        assert!(snapshot.bid.is_some());
        assert!(snapshot.ask.is_some());
        assert!(snapshot.price_change_pct.is_some());
    }

    #[test]
    fn price_y_maps_min_to_bottom_and_max_to_top() {
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(400.0, 300.0));
        assert_eq!(price_y(rect, 1.0, 1.0, 0.2), 300.0); // min price → bottom
        assert_eq!(price_y(rect, 1.2, 1.0, 0.2), 0.0); // max price → top
        assert!((price_y(rect, 1.1, 1.0, 0.2) - 150.0).abs() < 1.0); // mid price → mid
    }
}
