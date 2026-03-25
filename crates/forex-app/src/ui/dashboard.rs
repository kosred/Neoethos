use egui::{Ui, Color32, Stroke, Rect, Pos2, vec2};
use crate::ui::theme;

#[derive(Default, Clone, Debug)]
pub struct DashboardPanel {
    pub equity_curve: Vec<f64>,
}

impl DashboardPanel {
    pub fn new() -> Self {
        Self {
            equity_curve: vec![100000.0, 100050.0, 100100.0, 100020.0, 100300.0, 100450.0],
        }
    }

    pub fn push_equity(&mut self, val: f64) {
        self.equity_curve.push(val);
        if self.equity_curve.len() > 100 {
            self.equity_curve.remove(0);
        }
    }

    pub fn show(&mut self, ui: &mut Ui, auto_trade: &mut bool) {
        ui.heading("Live Equity Curve & Tracker");
        ui.add_space(8.0);
        
        theme::section_frame(ui.style()).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.strong("AI Auto-Trade Master Switch");
                ui.add_space(8.0);
                ui.checkbox(auto_trade, "Enable Live Execution");
            });
            if *auto_trade {
                ui.label(egui::RichText::new("WARNING: Models are currently authorized to dispatch live trades!").color(theme::DANGER).strong());
            } else {
                ui.label(egui::RichText::new("Safe Mode: Models can evaluate opportunities but cannot trade.").color(theme::TEXT_MUTED));
            }
        });
        
        ui.add_space(12.0);
        
        let min_equity = self.equity_curve.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_equity = self.equity_curve.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        let current = self.equity_curve.last().copied().unwrap_or(0.0);
        ui.label(format!("Current Equity: ${:.2}", current));
        ui.label(format!("Peak Watermark: ${:.2}", max_equity));
        
        // Custom simple line chart using egui basic shapes to avoid adding heavy egui_plot dependency
        let (rect, _response) = ui.allocate_exact_size(vec2(ui.available_width(), 150.0), egui::Sense::hover());
        
        if self.equity_curve.len() > 1 {
            let width = rect.width();
            let height = rect.height();
            let range = (max_equity - min_equity).max(1.0);
            
            let points: Vec<Pos2> = self.equity_curve.iter().enumerate().map(|(i, &val)| {
                let x = rect.left() + (i as f32 / (self.equity_curve.len() - 1) as f32) * width;
                let y = rect.bottom() - ((val - min_equity) as f32 / range as f32) * height;
                Pos2::new(x, y)
            }).collect();

            let color = if self.equity_curve.first() <= self.equity_curve.last() {
                Color32::from_rgb(0, 200, 0)
            } else {
                Color32::from_rgb(200, 0, 0)
            };
            
            ui.painter().add(egui::Shape::line(
                points,
                Stroke::new(2.0, color),
            ));
        }
    }
}
