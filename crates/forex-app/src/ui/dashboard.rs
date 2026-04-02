use crate::ui::theme;
use egui::{Color32, Pos2, Stroke, Ui, vec2};

#[derive(Default, Clone, Debug)]
pub struct DashboardPanel {
    pub equity_curve: Vec<f64>,
}

impl DashboardPanel {
    pub fn new() -> Self {
        Self {
            equity_curve: Vec::new(),
        }
    }

    pub fn show(
        &mut self,
        ui: &mut Ui,
        auto_trade_enabled: bool,
        account_balance: f64,
        account_equity: f64,
    ) {
        ui.heading("Operator Overview");
        ui.add_space(8.0);

        theme::section_frame(ui.style()).show(ui, |ui| {
            ui.strong("Execution Posture");
            if auto_trade_enabled {
                ui.label(
                    egui::RichText::new(
                        "AI auto-trade is armed. Model-originated execution may be dispatched.",
                    )
                    .color(theme::DANGER)
                    .strong(),
                );
            } else {
                ui.label(
                    egui::RichText::new(
                        "Manual-safe mode. Models may score opportunities but cannot dispatch live trades.",
                    )
                    .color(theme::TEXT_MUTED),
                );
            }
        });

        ui.add_space(12.0);
        ui.strong("Account Snapshot");
        ui.add_space(4.0);

        if account_balance > 0.0 {
            ui.label(format!("Account Balance: ${:.2}", account_balance));
        } else {
            ui.label("Account Balance: Unavailable");
        }
        if account_equity > 0.0 {
            ui.label(format!("Current Equity: ${:.2}", account_equity));
        } else {
            ui.label("Current Equity: Unavailable");
        }

        let (rect, _response) =
            ui.allocate_exact_size(vec2(ui.available_width(), 150.0), egui::Sense::hover());

        if self.equity_curve.len() > 1 {
            let min_equity = self
                .equity_curve
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            let max_equity = self
                .equity_curve
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let width = rect.width();
            let height = rect.height();
            let range = (max_equity - min_equity).max(1.0);

            let points: Vec<Pos2> = self
                .equity_curve
                .iter()
                .enumerate()
                .map(|(i, &val)| {
                    let x = rect.left() + (i as f32 / (self.equity_curve.len() - 1) as f32) * width;
                    let y = rect.bottom() - ((val - min_equity) as f32 / range as f32) * height;
                    Pos2::new(x, y)
                })
                .collect();

            let color = if self.equity_curve.first() <= self.equity_curve.last() {
                Color32::from_rgb(0, 200, 0)
            } else {
                Color32::from_rgb(200, 0, 0)
            };
            ui.label(format!("Peak Watermark: ${:.2}", max_equity));
            ui.painter()
                .add(egui::Shape::line(points, Stroke::new(2.0, color)));
        } else {
            let message = if account_equity > 0.0 {
                "Equity history not available yet."
            } else {
                "No live equity history is connected."
            };
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                message,
                egui::FontId::proportional(14.0),
                theme::TEXT_MUTED,
            );
        }
    }
}
