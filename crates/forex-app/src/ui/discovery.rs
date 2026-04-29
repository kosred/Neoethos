use crate::app_services::{
    ServiceEvent,
    discovery::{DiscoveryJobHandle, DiscoveryRequest, failed_snapshot, start_discovery_job},
    jobs::{JobSnapshot, JobState},
};
use crate::app_state::AppState;
use crate::ui::components::{
    DashboardCard, DashboardSection, open_log, render_dashboard_sections, render_report,
    render_status_badge, render_summary_cards, render_view_header,
};
use eframe::egui;
use tokio::sync::mpsc;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct DiscoveryDashboard {
    summary_cards: Vec<DashboardCard>,
    sections: Vec<DashboardSection>,
}

pub fn render(
    ui: &mut egui::Ui,
    state: &mut AppState,
    tx: &mpsc::Sender<ServiceEvent>,
    handle: &mut Option<DiscoveryJobHandle>,
) {
    render_view_header(
        ui,
        "Strategy Discovery Engine",
        "Search, rank, and assemble tradable strategy portfolios from the active symbol universe.",
    );
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

    egui::CollapsingHeader::new("🚀 Discovery Parameters")
        .default_open(true)
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label("Base Timeframe:");
                    ui.text_edit_singleline(&mut state.discovery_form.base_tf);
                });
                ui.horizontal(|ui| {
                    ui.label("Higher Timeframes (CSV):");
                    ui.text_edit_singleline(&mut state.discovery_form.higher_tfs);
                });
                ui.add(
                    egui::Slider::new(&mut state.discovery_form.max_indicators, 1..=30)
                        .text("Max Indicators"),
                );
                ui.add(
                    egui::Slider::new(&mut state.discovery_form.population, 10..=500)
                        .text("Population"),
                );
                ui.add(
                    egui::Slider::new(&mut state.discovery_form.generations, 1..=100)
                        .text("Generations"),
                );
                ui.add(
                    egui::Slider::new(&mut state.discovery_form.target_candidates, 10..=1000)
                        .text("Target Candidates"),
                );
                ui.add(
                    egui::Slider::new(&mut state.discovery_form.portfolio_size, 1..=500)
                        .text("Portfolio Size"),
                );
                ui.add(
                    egui::Slider::new(&mut state.discovery_form.correlation_threshold, 0.0..=1.0)
                        .text("Correlation Threshold"),
                );
                ui.add(
                    egui::Slider::new(&mut state.discovery_form.min_trades_per_day, 0.1..=10.0)
                        .text("Min Trades/Day"),
                );
            });
        });

    render_status_badge(ui, "Discovery", state.discovery_job.as_ref());

    egui::ScrollArea::vertical().show(ui, |ui| {
        if let Some(snapshot) = state.discovery_job.as_ref() {
            render_discovery_dashboard(ui, snapshot);
            egui::CollapsingHeader::new("Detailed Report & Events")
                .default_open(true)
                .show(ui, |ui| render_report(ui, snapshot));
        } else {
            ui.label("No active discovery job.");
        }
    });

    ui.separator();
    ui.horizontal(|ui| {
        let running = state
            .discovery_job
            .as_ref()
            .map(|snapshot| matches!(snapshot.state, JobState::Queued | JobState::Running))
            .unwrap_or(false);

        if !running && ui.button("🔥 Start Genetic Discovery").clicked() {
            let higher_tfs: Vec<String> = state
                .discovery_form
                .higher_tfs
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            let request = DiscoveryRequest {
                data_root: state.runtime.data_dir.clone(),
                symbol: state.selected_pair.clone(),
                base_tf: state.discovery_form.base_tf.clone(),
                higher_tfs,
                config: forex_search::DiscoveryConfig {
                    timeframe_label: state.discovery_form.base_tf.clone(),
                    population: state.discovery_form.population as usize,
                    generations: state.discovery_form.generations as usize,
                    max_indicators: state.discovery_form.max_indicators as usize,
                    candidate_count: state.discovery_form.target_candidates as usize,
                    portfolio_size: state.discovery_form.portfolio_size as usize,
                    corr_threshold: state.discovery_form.correlation_threshold as f64,
                    min_trades_per_day: state.discovery_form.min_trades_per_day as f64,
                    ..forex_search::DiscoveryConfig::default()
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

        if running
            && ui.button("Stop Search").clicked()
            && let Some(handle) = handle.as_ref()
        {
            handle.cancel.request();
        }

        if ui.button("Open Log").clicked()
            && let Err(err) = open_log(&state.canonical_log_path)
        {
            state.discovery_job = Some(failed_snapshot(
                crate::app_services::jobs::JobKind::Discovery,
                anyhow::anyhow!(
                    "failed to open log {}: {}",
                    state.canonical_log_path.display(),
                    err
                ),
            ));
        }
    });
}

fn render_discovery_dashboard(ui: &mut egui::Ui, snapshot: &JobSnapshot) {
    let dashboard = build_discovery_dashboard(snapshot);
    render_summary_cards(ui, "Discovery Overview", &dashboard.summary_cards);
    render_dashboard_sections(
        ui,
        &format!("discovery_dashboard_{:?}", snapshot.id),
        &dashboard.sections,
    );
}

fn build_discovery_dashboard(snapshot: &JobSnapshot) -> DiscoveryDashboard {
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
    ];
    if let Some(symbol) = highlight_value(snapshot, "symbol") {
        summary_cards.push(DashboardCard {
            label: "Symbol".to_string(),
            value: symbol.to_string(),
        });
    }
    if let Some(best_strategy) = highlight_value(snapshot, "best_strategy") {
        summary_cards.push(DashboardCard {
            label: "Best Strategy".to_string(),
            value: best_strategy.to_string(),
        });
    }

    let mut sections = Vec::new();

    let mut plan_rows = Vec::new();
    push_counter_row(
        snapshot,
        &mut plan_rows,
        "target_candidates",
        "Target Candidates",
    );
    push_counter_row(
        snapshot,
        &mut plan_rows,
        "target_portfolio",
        "Target Portfolio",
    );
    push_counter_row(snapshot, &mut plan_rows, "population", "Population");
    push_counter_row(snapshot, &mut plan_rows, "generations", "Generations");
    push_highlight_row(snapshot, &mut plan_rows, "base_tf", "Base TF");
    push_highlight_row(snapshot, &mut plan_rows, "higher_tfs", "Higher TFs");
    push_section(&mut sections, "Discovery Plan", plan_rows);

    let mut search_rows = Vec::new();
    if let Some(current_generation) = counter_value(snapshot, "generation") {
        let total_generations = counter_value(snapshot, "generations")
            .map(|value| value.to_string())
            .unwrap_or_else(|| "?".to_string());
        search_rows.push((
            "Current Generation".to_string(),
            format!("{current_generation} / {total_generations}"),
        ));
    }
    push_counter_row(
        snapshot,
        &mut search_rows,
        "archived_profitable",
        "Archived Profitable",
    );
    push_counter_row(
        snapshot,
        &mut search_rows,
        "stagnant_generations",
        "Stagnant Generations",
    );
    push_counter_row(
        snapshot,
        &mut search_rows,
        "truncated_candidates",
        "Truncated Candidates",
    );
    push_section(&mut sections, "Search Runtime", search_rows);

    let mut selection_rows = Vec::new();
    push_counter_row(
        snapshot,
        &mut selection_rows,
        "candidates",
        "Ranked Candidates",
    );
    push_counter_row(
        snapshot,
        &mut selection_rows,
        "filtered_candidates",
        "Filtered Candidates",
    );
    push_counter_row(snapshot, &mut selection_rows, "portfolio", "Portfolio Size");
    push_counter_row(
        snapshot,
        &mut selection_rows,
        "rejected_by_correlation",
        "Rejected By Correlation",
    );
    push_section(&mut sections, "Selection Funnel", selection_rows);

    let mut best_rows = Vec::new();
    push_highlight_row(snapshot, &mut best_rows, "best_strategy", "Strategy");
    push_highlight_row(snapshot, &mut best_rows, "best_sharpe", "Sharpe");
    push_highlight_row(snapshot, &mut best_rows, "best_win_rate", "Win Rate");
    push_section(&mut sections, "Best Candidate", best_rows);

    DiscoveryDashboard {
        summary_cards,
        sections,
    }
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
    fn discovery_dashboard_groups_runtime_metrics_into_operator_sections() {
        let mut snapshot = JobSnapshot::new(JobKind::Discovery);
        snapshot.state = JobState::Running;
        snapshot.progress = JobProgress {
            percent: Some(0.88),
            stage: "search_generations".to_string(),
            message: "generation 3/5 complete".to_string(),
        };
        snapshot.report.counters = vec![
            ("target_candidates".to_string(), 200),
            ("target_portfolio".to_string(), 100),
            ("population".to_string(), 100),
            ("generations".to_string(), 5),
            ("generation".to_string(), 3),
            ("archived_profitable".to_string(), 17),
            ("stagnant_generations".to_string(), 1),
            ("candidates".to_string(), 200),
            ("truncated_candidates".to_string(), 200),
            ("filtered_candidates".to_string(), 33),
            ("portfolio".to_string(), 12),
            ("rejected_by_correlation".to_string(), 9),
        ];
        snapshot.report.highlights = vec![
            ("symbol".to_string(), "EURUSD".to_string()),
            ("base_tf".to_string(), "M1".to_string()),
            ("higher_tfs".to_string(), "M5, M15, H1".to_string()),
            ("best_strategy".to_string(), "trend-breakout-7".to_string()),
            ("best_sharpe".to_string(), "1.94".to_string()),
            ("best_win_rate".to_string(), "0.62".to_string()),
        ];

        let dashboard = build_discovery_dashboard(&snapshot);

        assert_eq!(dashboard.summary_cards[0].label, "State");
        assert_eq!(dashboard.summary_cards[0].value, "Running");
        assert_eq!(dashboard.summary_cards[1].label, "Stage");
        assert_eq!(dashboard.summary_cards[1].value, "search_generations");
        assert_eq!(dashboard.summary_cards[2].label, "Symbol");
        assert_eq!(dashboard.summary_cards[2].value, "EURUSD");
        assert_eq!(dashboard.summary_cards[3].label, "Best Strategy");
        assert_eq!(dashboard.summary_cards[3].value, "trend-breakout-7");

        assert_eq!(dashboard.sections[0].title, "Discovery Plan");
        assert_eq!(
            dashboard.sections[0].rows,
            vec![
                ("Target Candidates".to_string(), "200".to_string()),
                ("Target Portfolio".to_string(), "100".to_string()),
                ("Population".to_string(), "100".to_string()),
                ("Generations".to_string(), "5".to_string()),
                ("Base TF".to_string(), "M1".to_string()),
                ("Higher TFs".to_string(), "M5, M15, H1".to_string()),
            ]
        );

        assert_eq!(dashboard.sections[1].title, "Search Runtime");
        assert_eq!(
            dashboard.sections[1].rows,
            vec![
                ("Current Generation".to_string(), "3 / 5".to_string()),
                ("Archived Profitable".to_string(), "17".to_string()),
                ("Stagnant Generations".to_string(), "1".to_string()),
                ("Truncated Candidates".to_string(), "200".to_string()),
            ]
        );

        assert_eq!(dashboard.sections[2].title, "Selection Funnel");
        assert_eq!(
            dashboard.sections[2].rows,
            vec![
                ("Ranked Candidates".to_string(), "200".to_string()),
                ("Filtered Candidates".to_string(), "33".to_string()),
                ("Portfolio Size".to_string(), "12".to_string()),
                ("Rejected By Correlation".to_string(), "9".to_string()),
            ]
        );

        assert_eq!(dashboard.sections[3].title, "Best Candidate");
        assert_eq!(
            dashboard.sections[3].rows,
            vec![
                ("Strategy".to_string(), "trend-breakout-7".to_string()),
                ("Sharpe".to_string(), "1.94".to_string()),
                ("Win Rate".to_string(), "0.62".to_string()),
            ]
        );
    }

    #[test]
    fn discovery_dashboard_omits_empty_sections() {
        let mut snapshot = JobSnapshot::new(JobKind::Discovery);
        snapshot.state = JobState::Queued;
        snapshot.report.highlights = vec![("symbol".to_string(), "GBPUSD".to_string())];

        let dashboard = build_discovery_dashboard(&snapshot);

        assert_eq!(dashboard.summary_cards.len(), 3);
        assert!(
            dashboard
                .sections
                .iter()
                .all(|section| !section.rows.is_empty())
        );
        assert!(
            dashboard
                .sections
                .iter()
                .all(|section| section.title != "Best Candidate")
        );
    }
}
