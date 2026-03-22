use crate::app_state::AppState;
use crate::ui::theme;
use eframe::egui;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChartPanel {
    pub symbol: String,
    pub timeframes: Vec<&'static str>,
    pub markers: Vec<String>,
}

pub fn build_chart_panel(state: &AppState) -> ChartPanel {
    ChartPanel {
        symbol: state.selected_pair.clone(),
        timeframes: vec!["M1", "M5", "H1"],
        markers: vec![
            format!("BOT BUY · {}", state.selected_pair),
            format!("BOT EXIT · {}", state.selected_pair),
        ],
    }
}

pub fn render(ui: &mut egui::Ui, state: &AppState) {
    let panel = build_chart_panel(state);

    ui.vertical(|ui| {
        ui.horizontal(|ui| {
            ui.strong(format!("{} Chart Surface", panel.symbol));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                for timeframe in panel.timeframes.iter().rev() {
                    ui.add_sized(
                        [52.0, 24.0],
                        egui::Button::new(egui::RichText::new(*timeframe).color(theme::TEXT_PRIMARY)),
                    );
                }
            });
        });

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

        let points = [
            egui::pos2(rect.left() + 22.0, rect.bottom() - 120.0),
            egui::pos2(rect.left() + rect.width() * 0.18, rect.bottom() - 150.0),
            egui::pos2(rect.left() + rect.width() * 0.32, rect.bottom() - 132.0),
            egui::pos2(rect.left() + rect.width() * 0.46, rect.bottom() - 190.0),
            egui::pos2(rect.left() + rect.width() * 0.62, rect.bottom() - 176.0),
            egui::pos2(rect.left() + rect.width() * 0.79, rect.bottom() - 230.0),
            egui::pos2(rect.right() - 26.0, rect.bottom() - 248.0),
        ];
        painter.add(egui::Shape::line(
            points.to_vec(),
            egui::Stroke::new(3.0, theme::ACCENT),
        ));

        let buy_marker = points[3];
        let exit_marker = points[5];
        painter.circle_filled(buy_marker, 6.0, theme::SUCCESS);
        painter.circle_filled(exit_marker, 6.0, theme::DANGER);

        painter.text(
            buy_marker + egui::vec2(-24.0, -24.0),
            egui::Align2::LEFT_CENTER,
            "BOT BUY",
            egui::TextStyle::Small.resolve(ui.style()),
            theme::SUCCESS,
        );
        painter.text(
            exit_marker + egui::vec2(-24.0, -24.0),
            egui::Align2::LEFT_CENTER,
            "BOT EXIT",
            egui::TextStyle::Small.resolve(ui.style()),
            theme::DANGER,
        );

        ui.add_space(8.0);
        ui.label(
            egui::RichText::new("Chart engine placeholder: the next tranche will add a real plotting/annotation layer on this surface.")
                .color(theme::TEXT_MUTED),
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_state::{AppRuntimeConfig, AppState, HardwareState, RiskState};
    use std::path::PathBuf;

    fn sample_state() -> AppState {
        AppState {
            runtime: AppRuntimeConfig {
                config_path: "config.yaml".to_string(),
                data_dir: PathBuf::from("data"),
                start_local: true,
            },
            data_source: crate::app_state::DataSource::Local,
            status_msg: "Local Mode".to_string(),
            selected_pair: "EURUSD".to_string(),
            available_symbols: vec!["EURUSD".to_string(), "GBPUSD".to_string()],
            discovery_job: None,
            training_job: None,
            canonical_log_path: PathBuf::from("logs").join("forex-ai.log"),
            hardware: HardwareState::default(),
            risk: RiskState::default(),
        }
    }

    #[test]
    fn chart_panel_shows_timeframes_and_bot_marker_placeholders() {
        let panel = build_chart_panel(&sample_state());

        assert_eq!(panel.timeframes, vec!["M1", "M5", "H1"]);
        assert!(panel
            .markers
            .iter()
            .any(|marker: &String| marker.contains("BOT BUY")));
        assert!(panel
            .markers
            .iter()
            .any(|marker: &String| marker.contains("BOT EXIT")));
    }
}
