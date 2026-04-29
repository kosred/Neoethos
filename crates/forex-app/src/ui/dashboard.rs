use crate::app_services::jobs::{JobSnapshot, JobState};
use crate::ui::ai_insights::AiInsightsPanel;
use crate::ui::components::{DashboardCard, render_status_badge, render_summary_cards};
use crate::ui::theme;
use egui::{Pos2, Stroke, Ui, vec2};

#[derive(Default, Clone, Debug)]
pub struct DashboardPanel {
    pub equity_curve: Vec<f64>,
}

pub struct DashboardInputs<'a> {
    pub auto_trade_enabled: bool,
    pub account_balance: f64,
    pub account_equity: f64,
    pub discovery_job: Option<&'a JobSnapshot>,
    pub training_job: Option<&'a JobSnapshot>,
    pub ai_insights: &'a AiInsightsPanel,
}

impl DashboardPanel {
    pub fn new() -> Self {
        Self {
            equity_curve: Vec::new(),
        }
    }

    pub fn show(&mut self, ui: &mut Ui, inputs: DashboardInputs<'_>) {
        let DashboardInputs {
            auto_trade_enabled,
            account_balance,
            account_equity,
            discovery_job,
            training_job,
            ai_insights,
        } = inputs;

        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Operator Overview");
            ui.add_space(8.0);

            let summary_cards = vec![
                DashboardCard {
                    label: "Auto-Trade".to_string(),
                    value: if auto_trade_enabled {
                        "Armed".to_string()
                    } else {
                        "Manual-safe".to_string()
                    },
                },
                DashboardCard {
                    label: "Balance".to_string(),
                    value: if account_balance > 0.0 {
                        format!("${:.2}", account_balance)
                    } else {
                        "Unavailable".to_string()
                    },
                },
                DashboardCard {
                    label: "Equity".to_string(),
                    value: if account_equity > 0.0 {
                        format!("${:.2}", account_equity)
                    } else {
                        "Unavailable".to_string()
                    },
                },
                DashboardCard {
                    label: "Discovery".to_string(),
                    value: job_state_label(discovery_job),
                },
                DashboardCard {
                    label: "Training".to_string(),
                    value: job_state_label(training_job),
                },
            ];
            render_summary_cards(ui, "Operator Snapshot", &summary_cards);

            ui.add_space(8.0);
            ui.columns(2, |columns| {
                theme::section_frame(columns[0].style()).show(&mut columns[0], |ui| {
                    ui.strong(egui::RichText::new("Execution Posture").color(theme::TEXT_PRIMARY));
                    ui.add_space(6.0);
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
                                "Manual-safe mode. Models may score but cannot dispatch live trades.",
                            )
                            .color(theme::TEXT_MUTED),
                        );
                    }
                });

                theme::section_frame(columns[1].style()).show(&mut columns[1], |ui| {
                    ui.strong(
                        egui::RichText::new("AI Ensemble Signals").color(theme::TEXT_PRIMARY),
                    );
                    ui.add_space(6.0);
                    if let (Some(buy), Some(sell), Some(neutral)) = (
                        ai_insights.prob_buy,
                        ai_insights.prob_sell,
                        ai_insights.prob_neutral,
                    ) {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Buy ").color(theme::TEXT_MUTED));
                            ui.add(
                                egui::ProgressBar::new(buy)
                                    .text(format!("{:.0}%", buy * 100.0))
                                    .fill(theme::SUCCESS),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Sell").color(theme::TEXT_MUTED));
                            ui.add(
                                egui::ProgressBar::new(sell)
                                    .text(format!("{:.0}%", sell * 100.0))
                                    .fill(theme::DANGER),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Hold").color(theme::TEXT_MUTED));
                            ui.add(
                                egui::ProgressBar::new(neutral)
                                    .text(format!("{:.0}%", neutral * 100.0))
                                    .fill(theme::TEXT_MUTED),
                            );
                        });
                    } else {
                        ui.label(
                            egui::RichText::new("Prediction signals unavailable.")
                                .color(theme::TEXT_MUTED),
                        );
                        ui.label(
                            egui::RichText::new("Run swarm training to populate the ensemble.")
                                .color(theme::TEXT_MUTED)
                                .small(),
                        );
                    }
                });
            });

            ui.add_space(8.0);
            ui.columns(2, |columns| {
                theme::section_frame(columns[0].style()).show(&mut columns[0], |ui| {
                    ui.strong(egui::RichText::new("Job Activity").color(theme::TEXT_PRIMARY));
                    ui.add_space(6.0);
                    render_status_badge(ui, "Discovery", discovery_job);
                    ui.add_space(4.0);
                    render_status_badge(ui, "Training", training_job);
                    if let Some(job) = discovery_job.or(training_job) {
                        if !job.progress.stage.is_empty() {
                            ui.add_space(4.0);
                            ui.label(
                                egui::RichText::new(format!("Stage: {}", job.progress.stage))
                                    .color(theme::TEXT_MUTED)
                                    .small(),
                            );
                        }
                        if let Some(pct) = job.progress.percent {
                            ui.add(egui::ProgressBar::new(pct));
                        }
                    }
                });

                theme::section_frame(columns[1].style()).show(&mut columns[1], |ui| {
                    ui.strong(
                        egui::RichText::new("Account Snapshot").color(theme::TEXT_PRIMARY),
                    );
                    ui.add_space(6.0);
                    egui::Grid::new("dashboard_account_grid")
                        .num_columns(2)
                        .spacing([12.0, 8.0])
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("Balance").color(theme::TEXT_MUTED));
                            if account_balance > 0.0 {
                                ui.strong(
                                    egui::RichText::new(format!("${:.2}", account_balance))
                                        .color(theme::TEXT_PRIMARY),
                                );
                            } else {
                                ui.label(
                                    egui::RichText::new("Unavailable").color(theme::TEXT_MUTED),
                                );
                            }
                            ui.end_row();

                            ui.label(egui::RichText::new("Equity").color(theme::TEXT_MUTED));
                            if account_equity > 0.0 {
                                let color = if account_equity >= account_balance {
                                    theme::SUCCESS
                                } else {
                                    theme::DANGER
                                };
                                ui.strong(
                                    egui::RichText::new(format!("${:.2}", account_equity))
                                        .color(color),
                                );
                            } else {
                                ui.label(
                                    egui::RichText::new("Unavailable").color(theme::TEXT_MUTED),
                                );
                            }
                            ui.end_row();
                        });
                });
            });

            ui.add_space(8.0);
            let (rect, _) =
                ui.allocate_exact_size(vec2(ui.available_width(), 120.0), egui::Sense::hover());

            if self.equity_curve.len() > 1 {
                let min_eq = self
                    .equity_curve
                    .iter()
                    .cloned()
                    .fold(f64::INFINITY, f64::min);
                let max_eq = self
                    .equity_curve
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                let width = rect.width();
                let height = rect.height();
                let range = (max_eq - min_eq).max(1.0);

                let points: Vec<Pos2> = self
                    .equity_curve
                    .iter()
                    .enumerate()
                    .map(|(i, &val)| {
                        let x = rect.left()
                            + (i as f32 / (self.equity_curve.len() - 1) as f32) * width;
                        let y = rect.bottom() - ((val - min_eq) as f32 / range as f32) * height;
                        Pos2::new(x, y)
                    })
                    .collect();

                let color = if self.equity_curve.first() <= self.equity_curve.last() {
                    theme::SUCCESS
                } else {
                    theme::DANGER
                };
                ui.painter()
                    .add(egui::Shape::line(points, Stroke::new(2.0, color)));
                ui.label(
                    egui::RichText::new(format!("Peak: ${:.2}", max_eq))
                        .color(theme::TEXT_MUTED)
                        .small(),
                );
            } else {
                ui.painter().text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    if account_equity > 0.0 {
                        "Equity history not available yet."
                    } else {
                        "No live equity history is connected."
                    },
                    egui::FontId::proportional(14.0),
                    theme::TEXT_MUTED,
                );
            }
        });
    }
}

fn job_state_label(job: Option<&JobSnapshot>) -> String {
    match job {
        None => "Idle".to_string(),
        Some(snapshot) => match snapshot.state {
            JobState::Queued => "Queued".to_string(),
            JobState::Running => {
                if snapshot.progress.stage.is_empty() {
                    "Running".to_string()
                } else {
                    snapshot.progress.stage.clone()
                }
            }
            JobState::Succeeded => "Succeeded".to_string(),
            JobState::Degraded => "Degraded".to_string(),
            JobState::Failed => "Failed".to_string(),
            JobState::Cancelled => "Cancelled".to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::jobs::{JobKind, JobProgress, JobSnapshot};

    #[test]
    fn job_state_label_returns_idle_for_none() {
        assert_eq!(job_state_label(None), "Idle");
    }

    #[test]
    fn job_state_label_uses_stage_when_running() {
        let mut snapshot = JobSnapshot::new(JobKind::Discovery);
        snapshot.state = JobState::Running;
        snapshot.progress = JobProgress {
            percent: Some(0.5),
            stage: "search_generations".to_string(),
            message: String::new(),
        };
        assert_eq!(job_state_label(Some(&snapshot)), "search_generations");
    }

    #[test]
    fn job_state_label_falls_back_to_running_when_stage_is_empty() {
        let mut snapshot = JobSnapshot::new(JobKind::Training);
        snapshot.state = JobState::Running;
        assert_eq!(job_state_label(Some(&snapshot)), "Running");
    }

    #[test]
    fn job_state_label_shows_terminal_states() {
        let mut s = JobSnapshot::new(JobKind::Discovery);
        s.state = JobState::Succeeded;
        assert_eq!(job_state_label(Some(&s)), "Succeeded");
        s.state = JobState::Failed;
        assert_eq!(job_state_label(Some(&s)), "Failed");
        s.state = JobState::Degraded;
        assert_eq!(job_state_label(Some(&s)), "Degraded");
    }
}
