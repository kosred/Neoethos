use crate::app_services::trading::TradingSession;
use crate::app_state::AppState;
use crate::ui::components::{
    DashboardCard, render_report, render_status_badge, render_summary_cards, render_view_header,
};
use crate::ui::system::shared::{failed_bootstrap_snapshot, parse_bootstrap_list};
use crate::ui::theme;
use eframe::egui;

pub fn render(
    ui: &mut egui::Ui,
    state: &mut AppState,
    session: &mut TradingSession,
    tx: &tokio::sync::mpsc::Sender<crate::app_services::ServiceEvent>,
) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        let summary_cards = vec![
            DashboardCard {
                label: "Data Root".to_string(),
                value: state.runtime.data_dir.display().to_string(),
            },
            DashboardCard {
                label: "Pairs".to_string(),
                value: state.bootstrap_form.pairs_input.clone(),
            },
            DashboardCard {
                label: "Timeframes".to_string(),
                value: state.bootstrap_form.timeframes_input.clone(),
            },
            DashboardCard {
                label: "Years".to_string(),
                value: state.bootstrap_form.years.to_string(),
            },
        ];

        render_view_header(
            ui,
            "Data Bootstrap",
            "Prepare local vortex data without mixing broker setup or intelligence controls into the same page.",
        );
        ui.separator();
        render_summary_cards(ui, "Bootstrap Snapshot", &summary_cards);

        ui.add_space(10.0);
        theme::section_frame(ui.style()).show(ui, |ui| {
            render_status_badge(ui, "Bootstrap", state.bootstrap_job.as_ref());
            ui.add_space(8.0);
            ui.strong("Historical Data Import");
            ui.label(
                "Fetch missing historical bars into the local vortex cache for research and training.",
            );

            ui.horizontal(|ui| {
                ui.label("Pairs");
                ui.text_edit_singleline(&mut state.bootstrap_form.pairs_input);
            });
            ui.horizontal(|ui| {
                ui.label("Timeframes");
                ui.text_edit_singleline(&mut state.bootstrap_form.timeframes_input);
            });
            ui.horizontal(|ui| {
                ui.label("Years");
                ui.add(egui::DragValue::new(&mut state.bootstrap_form.years).range(1..=25));
            });

            if ui.button("Start cTrader Bootstrap").clicked() {
                let symbols = parse_bootstrap_list(&state.bootstrap_form.pairs_input);
                let timeframes = parse_bootstrap_list(&state.bootstrap_form.timeframes_input);
                if symbols.is_empty() || timeframes.is_empty() {
                    state.bootstrap_job = Some(failed_bootstrap_snapshot(anyhow::anyhow!(
                        "bootstrap requires at least one symbol and one timeframe"
                    )));
                } else {
                    let _ = session.start_ctrader_bootstrap_batch(
                        state.runtime.data_dir.clone(),
                        symbols,
                        timeframes,
                        state.bootstrap_form.years,
                        tx.clone(),
                    );
                }
            }
        });

        if let Some(snapshot) = state.bootstrap_job.as_ref() {
            ui.add_space(10.0);
            theme::section_frame(ui.style()).show(ui, |ui| {
                render_report(ui, snapshot);
            });
        }
    });
}
