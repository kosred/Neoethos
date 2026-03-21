use crate::app_services::{
    jobs::JobState,
    training::{failed_snapshot, start_training_job, TrainingJobHandle, TrainingRequest},
    ServiceEvent,
};
use crate::app_state::AppState;
use crate::ui::components::{open_log, render_report, render_status_badge};
use eframe::egui;
use std::path::PathBuf;
use tokio::sync::mpsc;

pub fn render(
    ui: &mut egui::Ui,
    state: &mut AppState,
    tx: &mpsc::UnboundedSender<ServiceEvent>,
    handle: &mut Option<TrainingJobHandle>,
) {
    ui.heading("Model Swarm Training");
    ui.separator();
    ui.label("Train and update the model swarm using the current runtime configuration.");

    render_status_badge(ui, "Training", state.training_job.as_ref());

    if let Some(snapshot) = state.training_job.as_ref() {
        render_report(ui, snapshot);
    } else {
        ui.label("No active training job.");
    }

    ui.separator();
    ui.horizontal(|ui| {
        let running = state
            .training_job
            .as_ref()
            .map(|snapshot| matches!(snapshot.state, JobState::Queued | JobState::Running))
            .unwrap_or(false);

        if !running && ui.button("🚀 Run Swarm Training").clicked() {
            let request = TrainingRequest {
                config_path: state.runtime.config_path.clone(),
                models_dir: PathBuf::from("models"),
                symbol: state.selected_pair.clone(),
                base_tf: "M1".to_string(),
            };

            match start_training_job(request, tx.clone()) {
                Ok(job_handle) => {
                    state.training_job = Some(job_handle.snapshot.clone());
                    *handle = Some(job_handle);
                }
                Err(err) => {
                    state.training_job = Some(failed_snapshot(err));
                }
            }
        }

        if running && ui.button("Stop Training").clicked() {
            if let Some(handle) = handle.as_ref() {
                handle.cancel.request();
            }
        }

        if ui.button("Open Log").clicked() {
            if let Err(err) = open_log(&state.canonical_log_path) {
                state.training_job = Some(failed_snapshot(anyhow::anyhow!(
                    "failed to open log {}: {}",
                    state.canonical_log_path.display(),
                    err
                )));
            }
        }
    });
}
