use crate::app_services::{
    ServiceEvent,
    jobs::{JobSnapshot, JobState},
    training::{TrainingJobHandle, TrainingRequest, failed_snapshot, start_training_job},
};
use crate::app_state::AppState;
use crate::ui::components::{
    DashboardCard, DashboardSection, open_log, render_dashboard_sections, render_report,
    render_status_badge, render_summary_cards, render_view_header,
};
use eframe::egui;
use std::path::PathBuf;
use tokio::sync::mpsc;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct TrainingDashboard {
    summary_cards: Vec<DashboardCard>,
    sections: Vec<DashboardSection>,
}

type SectionRows = Vec<(String, String)>;
type TrainingEntryGroups = (SectionRows, SectionRows, SectionRows);

pub fn render(
    ui: &mut egui::Ui,
    state: &mut AppState,
    tx: &mpsc::Sender<ServiceEvent>,
    handle: &mut Option<TrainingJobHandle>,
) {
    render_view_header(
        ui,
        "Model Swarm Training",
        "Execute the Rust-owned training backend against the active runtime configuration and monitor model outcomes live.",
    );
    ui.separator();

    ui.horizontal(|ui| {
        ui.label("Symbol:");
        egui::ComboBox::from_id_salt("training_symbol")
            .selected_text(&state.selected_pair)
            .show_ui(ui, |ui| {
                for sym in &state.available_symbols {
                    ui.selectable_value(&mut state.selected_pair, sym.clone(), sym);
                }
            });
        ui.add_space(12.0);
        ui.label("Base TF:");
        let timeframes = neoethos_core::CANONICAL_TIMEFRAMES;
        egui::ComboBox::from_id_salt("training_tf")
            .selected_text(&state.chart_timeframe)
            .show_ui(ui, |ui| {
                for tf in timeframes {
                    ui.selectable_value(&mut state.chart_timeframe, tf.to_string(), *tf);
                }
            });
    });
    ui.add_space(4.0);

    render_status_badge(ui, "Training", state.training_job.as_ref());

    if let Some(snapshot) = state.training_job.as_ref() {
        render_training_dashboard(ui, snapshot);
        egui::CollapsingHeader::new("Detailed Report & Events")
            .default_open(true)
            .show(ui, |ui| render_report(ui, snapshot));
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
                base_tf: state.chart_timeframe.clone(),
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

        // Task #10 — Discovery → Training hand-off. Reads the
        // `model_targets.json` written by `start_discovery_job` for
        // the currently-selected (symbol, base_tf) and previews the
        // discovered portfolio so the operator can see what they're
        // about to train against. Clicking starts training with the
        // pair the discovery job locked in (which may differ from
        // the chrome's currently-selected pair if the operator has
        // since navigated away).
        if !running {
            let targets_path = crate::app_services::discovery::model_targets_path_for(
                &state.runtime.data_dir,
                &state.selected_pair,
                &state.chart_timeframe,
            );
            let targets_exists = targets_path.exists();
            ui.add_enabled_ui(targets_exists, |ui| {
                let label = if targets_exists {
                    "📋 Load discovered targets"
                } else {
                    "📋 No discovered targets yet"
                };
                if ui
                    .button(label)
                    .on_hover_text(if targets_exists {
                        "Read model_targets.json from the latest discovery run for this symbol/timeframe, and start a training job against the discovered portfolio."
                    } else {
                        "Run Discovery for this symbol/timeframe first — it writes model_targets.json on success."
                    })
                    .clicked()
                {
                    match load_model_targets(&targets_path) {
                        Ok(file) => {
                            // Sync chrome state to whatever Discovery
                            // actually targeted (defensive against
                            // operator nav since the discovery run).
                            state.selected_pair = file.symbol.clone();
                            state.chart_timeframe = file.base_tf.clone();
                            let req = TrainingRequest {
                                config_path: state.runtime.config_path.clone(),
                                models_dir: PathBuf::from("models"),
                                symbol: file.symbol,
                                base_tf: file.base_tf,
                            };
                            tracing::info!(
                                target: "neoethos_app::training::auto_trigger",
                                portfolio_size = file.portfolio.len(),
                                discovered_at = %file.discovered_at_utc,
                                "training auto-triggered from model_targets.json"
                            );
                            match start_training_job(req, tx.clone()) {
                                Ok(job_handle) => {
                                    state.training_job = Some(job_handle.snapshot.clone());
                                    *handle = Some(job_handle);
                                }
                                Err(err) => {
                                    state.training_job = Some(failed_snapshot(err));
                                }
                            }
                        }
                        Err(err) => {
                            state.training_job = Some(failed_snapshot(anyhow::anyhow!(
                                "model_targets.json at {} is malformed: {err}",
                                targets_path.display()
                            )));
                        }
                    }
                }
            });
        }

        if running
            && ui.button("Stop Training").clicked()
            && let Some(handle) = handle.as_ref()
        {
            handle.cancel.request();
        }

        if ui.button("Open Log").clicked()
            && let Err(err) = open_log(&state.canonical_log_path)
        {
            state.training_job = Some(failed_snapshot(anyhow::anyhow!(
                "failed to open log {}: {}",
                state.canonical_log_path.display(),
                err
            )));
        }
    });
}

/// Read `<data_root>/discovery_targets/<symbol>_<base_tf>_model_targets.json`
/// and assert on the schema version. Task #10 hand-off from Discovery.
fn load_model_targets(
    path: &std::path::Path,
) -> anyhow::Result<crate::app_services::discovery::ModelTargetsFile> {
    let bytes = std::fs::read(path)?;
    let file: crate::app_services::discovery::ModelTargetsFile =
        serde_json::from_slice(&bytes)?;
    if file.schema_version != crate::app_services::discovery::MODEL_TARGETS_SCHEMA_VERSION {
        anyhow::bail!(
            "model_targets.json schema_version mismatch: file={}, expected={}",
            file.schema_version,
            crate::app_services::discovery::MODEL_TARGETS_SCHEMA_VERSION
        );
    }
    Ok(file)
}

fn render_training_dashboard(ui: &mut egui::Ui, snapshot: &JobSnapshot) {
    let dashboard = build_training_dashboard(snapshot);
    render_summary_cards(ui, "Training Overview", &dashboard.summary_cards);
    render_dashboard_sections(
        ui,
        &format!("training_dashboard_{:?}", snapshot.id),
        &dashboard.sections,
    );
}

fn build_training_dashboard(snapshot: &JobSnapshot) -> TrainingDashboard {
    let requested_models = counter_value(snapshot, "requested_models")
        .or_else(|| counter_value(snapshot, "planned_models"))
        .unwrap_or(0);
    let completed_models = counter_value(snapshot, "completed_models").unwrap_or(0);

    let mut summary_cards = vec![
        DashboardCard {
            label: "State".to_string(),
            value: format!("{:?}", snapshot.state),
        },
        DashboardCard {
            label: "Stage".to_string(),
            value: if snapshot.progress.stage.is_empty() {
                "idle".to_string()
            } else {
                snapshot.progress.stage.clone()
            },
        },
        DashboardCard {
            label: "Symbol".to_string(),
            value: highlight_value(snapshot, "symbol")
                .unwrap_or("-")
                .to_string(),
        },
        DashboardCard {
            label: "Completion".to_string(),
            value: format!("{completed_models} / {requested_models}"),
        },
    ];

    if requested_models == 0 {
        summary_cards[3].value = "0 / 0".to_string();
    }

    let mut sections = Vec::new();

    let mut target_rows = Vec::new();
    push_highlight_row(snapshot, &mut target_rows, "base_tf", "Base TF");
    push_highlight_row(snapshot, &mut target_rows, "config_path", "Config");
    push_highlight_row(snapshot, &mut target_rows, "models_dir", "Models Dir");
    push_section(&mut sections, "Training Target", target_rows);

    let mut runtime_rows = Vec::new();
    push_counter_row(
        snapshot,
        &mut runtime_rows,
        "requested_models",
        "Requested Models",
    );
    push_counter_row(
        snapshot,
        &mut runtime_rows,
        "planned_models",
        "Planned Models",
    );
    push_counter_row(
        snapshot,
        &mut runtime_rows,
        "completed_models",
        "Completed Models",
    );
    push_counter_row(
        snapshot,
        &mut runtime_rows,
        "failed_models",
        "Failed Models",
    );
    push_section(&mut sections, "Execution Summary", runtime_rows);

    let (planned_models, completed_entries, failed_entries) = parse_training_entries(snapshot);
    push_section(&mut sections, "Planned Models", planned_models);
    push_section(&mut sections, "Completed Models", completed_entries);
    push_section(&mut sections, "Failed Models", failed_entries);

    TrainingDashboard {
        summary_cards,
        sections,
    }
}

fn parse_training_entries(snapshot: &JobSnapshot) -> TrainingEntryGroups {
    let mut planned = Vec::new();
    let mut completed = Vec::new();
    let mut failed = Vec::new();

    for entry in &snapshot.report.entries {
        let parts: Vec<&str> = entry.split(" | ").collect();
        match parts.as_slice() {
            ["planned", model] => planned.push(((*model).to_string(), "planned".to_string())),
            ["completed", model] => completed.push(((*model).to_string(), "completed".to_string())),
            ["failed", model, error] => failed.push(((*model).to_string(), (*error).to_string())),
            _ => {}
        }
    }

    (planned, completed, failed)
}

fn counter_value(snapshot: &JobSnapshot, name: &str) -> Option<u64> {
    snapshot
        .report
        .counters
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| *value)
}

fn highlight_value<'a>(snapshot: &'a JobSnapshot, name: &str) -> Option<&'a str> {
    snapshot
        .report
        .highlights
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.as_str())
}

fn push_counter_row(
    snapshot: &JobSnapshot,
    rows: &mut Vec<(String, String)>,
    key: &str,
    label: &str,
) {
    if let Some(value) = counter_value(snapshot, key) {
        rows.push((label.to_string(), value.to_string()));
    }
}

fn push_highlight_row(
    snapshot: &JobSnapshot,
    rows: &mut Vec<(String, String)>,
    key: &str,
    label: &str,
) {
    if let Some(value) = highlight_value(snapshot, key) {
        rows.push((label.to_string(), value.to_string()));
    }
}

fn push_section(sections: &mut Vec<DashboardSection>, title: &str, rows: Vec<(String, String)>) {
    if !rows.is_empty() {
        sections.push(DashboardSection {
            title: title.to_string(),
            rows,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::jobs::{JobKind, JobProgress, JobSnapshot};

    #[test]
    fn training_dashboard_groups_runtime_models_and_failures() {
        let mut snapshot = JobSnapshot::new(JobKind::Training);
        snapshot.state = JobState::Degraded;
        snapshot.progress = JobProgress {
            percent: Some(0.79),
            stage: "backend_running".to_string(),
            message: "backend running with 2 completed and 1 failed model(s) out of 4".to_string(),
        };
        snapshot.report.counters = vec![
            ("requested_models".to_string(), 4),
            ("planned_models".to_string(), 4),
            ("completed_models".to_string(), 2),
            ("failed_models".to_string(), 1),
        ];
        snapshot.report.highlights = vec![
            ("symbol".to_string(), "EURUSD".to_string()),
            ("base_tf".to_string(), "M1".to_string()),
            ("config_path".to_string(), "config.yaml".to_string()),
            ("models_dir".to_string(), "models".to_string()),
        ];
        snapshot.report.entries = vec![
            "planned | xgboost".to_string(),
            "planned | mlp".to_string(),
            "planned | elasticnet".to_string(),
            "completed | xgboost".to_string(),
            "completed | elasticnet".to_string(),
            "failed | mlp | cuda oom".to_string(),
        ];

        let dashboard = build_training_dashboard(&snapshot);

        assert_eq!(dashboard.summary_cards[0].label, "State");
        assert_eq!(dashboard.summary_cards[0].value, "Degraded");
        assert_eq!(dashboard.summary_cards[1].label, "Stage");
        assert_eq!(dashboard.summary_cards[1].value, "backend_running");
        assert_eq!(dashboard.summary_cards[2].label, "Symbol");
        assert_eq!(dashboard.summary_cards[2].value, "EURUSD");
        assert_eq!(dashboard.summary_cards[3].label, "Completion");
        assert_eq!(dashboard.summary_cards[3].value, "2 / 4");

        assert_eq!(dashboard.sections[0].title, "Training Target");
        assert_eq!(
            dashboard.sections[0].rows,
            vec![
                ("Base TF".to_string(), "M1".to_string()),
                ("Config".to_string(), "config.yaml".to_string()),
                ("Models Dir".to_string(), "models".to_string()),
            ]
        );

        assert_eq!(dashboard.sections[1].title, "Execution Summary");
        assert_eq!(
            dashboard.sections[1].rows,
            vec![
                ("Requested Models".to_string(), "4".to_string()),
                ("Planned Models".to_string(), "4".to_string()),
                ("Completed Models".to_string(), "2".to_string()),
                ("Failed Models".to_string(), "1".to_string()),
            ]
        );

        assert_eq!(dashboard.sections[2].title, "Planned Models");
        assert_eq!(
            dashboard.sections[2].rows,
            vec![
                ("xgboost".to_string(), "planned".to_string()),
                ("mlp".to_string(), "planned".to_string()),
                ("elasticnet".to_string(), "planned".to_string()),
            ]
        );

        assert_eq!(dashboard.sections[3].title, "Completed Models");
        assert_eq!(
            dashboard.sections[3].rows,
            vec![
                ("xgboost".to_string(), "completed".to_string()),
                ("elasticnet".to_string(), "completed".to_string()),
            ]
        );

        assert_eq!(dashboard.sections[4].title, "Failed Models");
        assert_eq!(
            dashboard.sections[4].rows,
            vec![("mlp".to_string(), "cuda oom".to_string()),]
        );
    }

    #[test]
    fn training_dashboard_omits_model_sections_when_entries_are_missing() {
        let mut snapshot = JobSnapshot::new(JobKind::Training);
        snapshot.state = JobState::Queued;
        snapshot.report.highlights = vec![("symbol".to_string(), "GBPUSD".to_string())];
        snapshot.report.counters = vec![("planned_models".to_string(), 0)];

        let dashboard = build_training_dashboard(&snapshot);

        assert_eq!(dashboard.summary_cards.len(), 4);
        assert_eq!(
            dashboard
                .sections
                .iter()
                .map(|section| section.title.as_str())
                .collect::<Vec<_>>(),
            vec!["Execution Summary"]
        );
    }
}
