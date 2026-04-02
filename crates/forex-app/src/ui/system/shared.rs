use crate::app_services::ServiceEvent;
use crate::app_services::broker_config::BrokerAccountTarget;
use crate::app_services::jobs::{JobEventLevel, JobKind, JobSnapshot, JobState, push_recent_event};
use crate::app_state::AppState;
use eframe::egui;

pub fn labeled_text_edit(ui: &mut egui::Ui, label: &str, value: &mut String) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add_sized(
            [ui.available_width().max(200.0), 24.0],
            egui::TextEdit::singleline(value),
        );
    });
}

pub fn render_account_targets(
    ui: &mut egui::Ui,
    accounts: &mut Vec<BrokerAccountTarget>,
    default_prefix: &str,
) {
    ui.add_space(6.0);
    ui.strong("Execution Targets");
    for (idx, account) in accounts.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.checkbox(&mut account.enabled_for_execution, "");
            ui.label(format!("Target {}", idx + 1));
            ui.add_sized(
                [120.0, 24.0],
                egui::TextEdit::singleline(&mut account.account_id),
            );
            ui.add_sized(
                [160.0, 24.0],
                egui::TextEdit::singleline(&mut account.label),
            );
        });
    }
    if ui.button("+ Add Account Target").clicked() {
        let next = accounts.len() + 1;
        accounts.push(BrokerAccountTarget {
            account_id: String::new(),
            label: format!("{default_prefix} {next}"),
            enabled_for_execution: false,
        });
    }
}

pub fn parse_bootstrap_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_ascii_uppercase())
        .collect()
}

pub fn failed_bootstrap_snapshot(err: anyhow::Error) -> JobSnapshot {
    let mut snapshot = JobSnapshot::new(JobKind::Bootstrap);
    snapshot.state = JobState::Failed;
    snapshot.progress.stage = "bootstrap_failed".to_string();
    snapshot.progress.message = err.to_string();
    snapshot.report.summary = format!("Bootstrap failed: {}", err);
    snapshot.report.errors.push(err.to_string());
    snapshot.report.events = push_recent_event(
        &snapshot.report.events,
        JobEventLevel::Error,
        snapshot.report.summary.clone(),
    );
    snapshot
}

pub fn sync_news_now(state: &AppState, tx: &tokio::sync::mpsc::Sender<ServiceEvent>) {
    let pair = state.selected_pair.clone();
    let mut filter_clone = state.llm_news_filter.clone();
    let tx_clone = tx.clone();
    std::thread::spawn(move || {
        if let Ok(status) = filter_clone.poll_llm_news_sentiment(&pair) {
            let _ = tx_clone.blocking_send(ServiceEvent::LlmNewsUpdated(status));
        }
    });
}
