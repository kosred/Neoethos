use crate::app_services::{
    discovery::{failed_snapshot, start_discovery_job, DiscoveryJobHandle, DiscoveryRequest},
    jobs::JobState,
    ServiceEvent,
};
use crate::app_state::AppState;
use crate::ui::components::{open_log, render_report, render_status_badge};
use eframe::egui;
use tokio::sync::mpsc;

pub fn render(
    ui: &mut egui::Ui,
    state: &mut AppState,
    tx: &mpsc::UnboundedSender<ServiceEvent>,
    handle: &mut Option<DiscoveryJobHandle>,
) {
    ui.heading("Strategy Discovery Engine");
    ui.separator();

    ui.horizontal(|ui| {
        ui.label("Target Pair:");
        egui::ComboBox::from_label("")
            .selected_text(&state.selected_pair)
            .show_ui(ui, |ui| {
                for sym in &state.available_symbols {
                    ui.selectable_value(&mut state.selected_pair, sym.clone(), sym);
                }
            });
    });

    render_status_badge(ui, "Discovery", state.discovery_job.as_ref());

    if let Some(snapshot) = state.discovery_job.as_ref() {
        render_report(ui, snapshot);
    } else {
        ui.label("No active discovery job.");
    }

    ui.separator();
    ui.horizontal(|ui| {
        let running = state
            .discovery_job
            .as_ref()
            .map(|snapshot| matches!(snapshot.state, JobState::Queued | JobState::Running))
            .unwrap_or(false);

        if !running && ui.button("🔥 Start Genetic Discovery").clicked() {
            let request = DiscoveryRequest {
                data_root: state.runtime.data_dir.clone(),
                symbol: state.selected_pair.clone(),
                base_tf: "M1".to_string(),
                higher_tfs: vec!["M5".to_string(), "M15".to_string(), "H1".to_string()],
                config: forex_search::DiscoveryConfig {
                    population: 100,
                    generations: 5,
                    max_indicators: 12,
                    candidate_count: 200,
                    portfolio_size: 100,
                    corr_threshold: 0.7,
                    min_trades_per_day: 1.0,
                    filtering: Default::default(),
                },
            };

            match start_discovery_job(request, tx.clone()) {
                Ok(job_handle) => {
                    state.discovery_job = Some(job_handle.snapshot.clone());
                    *handle = Some(job_handle);
                }
                Err(err) => {
                    state.discovery_job = Some(failed_snapshot(
                        crate::app_services::jobs::JobKind::Discovery,
                        err,
                    ));
                }
            }
        }

        if running && ui.button("Stop Search").clicked() {
            if let Some(handle) = handle.as_ref() {
                handle.cancel.request();
            }
        }

        if ui.button("Open Log").clicked() {
            if let Err(err) = open_log(&state.canonical_log_path) {
                state.discovery_job = Some(failed_snapshot(
                    crate::app_services::jobs::JobKind::Discovery,
                    anyhow::anyhow!("failed to open log {}: {}", state.canonical_log_path.display(), err),
                ));
            }
        }
    });
}
