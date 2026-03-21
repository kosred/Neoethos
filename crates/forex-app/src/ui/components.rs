use crate::app_services::jobs::{JobEventLevel, JobSnapshot, JobState};
use crate::ui::theme;
use eframe::egui;
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DashboardCard {
    pub label: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DashboardSection {
    pub title: String,
    pub rows: Vec<(String, String)>,
}

pub fn status_color(state: JobState) -> egui::Color32 {
    match state {
        JobState::Queued => theme::TEXT_MUTED,
        JobState::Running => theme::ACCENT,
        JobState::Succeeded => theme::SUCCESS,
        JobState::Degraded => theme::WARNING,
        JobState::Failed => theme::DANGER,
        JobState::Cancelled => egui::Color32::from_rgb(255, 128, 128),
    }
}

pub fn render_view_header(ui: &mut egui::Ui, title: &str, subtitle: &str) {
    ui.heading(egui::RichText::new(title).color(theme::TEXT_PRIMARY));
    if !subtitle.trim().is_empty() {
        ui.label(egui::RichText::new(subtitle).color(theme::TEXT_MUTED));
    }
}

pub fn render_status_badge(ui: &mut egui::Ui, label: &str, snapshot: Option<&JobSnapshot>) {
    let (text, color) = match snapshot {
        Some(snapshot) => (format!("{label}: {:?}", snapshot.state), status_color(snapshot.state)),
        None => (format!("{label}: Idle"), theme::TEXT_MUTED),
    };
    let mut frame = theme::card_frame(ui.style());
    frame.fill = color.linear_multiply(0.14);
    frame.stroke = egui::Stroke::new(1.0, color);
    frame.show(ui, |ui| {
        ui.label(egui::RichText::new(text).color(color).strong());
    });
}

pub fn render_summary_cards(ui: &mut egui::Ui, title: &str, cards: &[DashboardCard]) {
    if cards.is_empty() {
        return;
    }

    ui.separator();
    ui.strong(egui::RichText::new(title).color(theme::TEXT_PRIMARY));
    ui.horizontal_wrapped(|ui| {
        for card in cards {
            theme::card_frame(ui.style()).show(ui, |ui| {
                ui.set_min_size(egui::vec2(165.0, 68.0));
                ui.label(egui::RichText::new(&card.label).small().color(theme::TEXT_MUTED));
                ui.strong(egui::RichText::new(&card.value).color(theme::TEXT_PRIMARY));
            });
        }
    });
}

pub fn render_dashboard_sections(
    ui: &mut egui::Ui,
    id_prefix: &str,
    sections: &[DashboardSection],
) {
    if sections.is_empty() {
        return;
    }

    ui.separator();
    ui.columns(2, |columns| {
        for (idx, section) in sections.iter().enumerate() {
            columns[idx % 2].add_space(2.0);
            theme::section_frame(columns[idx % 2].style()).show(&mut columns[idx % 2], |ui| {
                ui.set_min_width(280.0);
                ui.strong(egui::RichText::new(&section.title).color(theme::TEXT_PRIMARY));
                ui.add_space(8.0);
                egui::Grid::new(format!("{id_prefix}_{idx}"))
                    .num_columns(2)
                    .spacing([12.0, 8.0])
                    .show(ui, |ui| {
                        for (label, value) in &section.rows {
                            ui.label(egui::RichText::new(label).color(theme::TEXT_MUTED));
                            ui.strong(egui::RichText::new(value).color(theme::TEXT_PRIMARY));
                            ui.end_row();
                        }
                    });
            });
        }
    });
}

pub fn render_report(ui: &mut egui::Ui, snapshot: &JobSnapshot) {
    if let Some(percent) = snapshot.progress.percent {
        ui.add(egui::ProgressBar::new(percent).text(snapshot.progress.stage.clone()));
    } else if !snapshot.progress.stage.is_empty() {
        ui.label(format!("Stage: {}", snapshot.progress.stage));
    }

    if !snapshot.progress.message.is_empty() {
        ui.label(&snapshot.progress.message);
    }

    if !snapshot.report.counters.is_empty() {
        ui.separator();
        ui.strong("Counters");
        egui::Grid::new(format!("report_counters_{:?}", snapshot.kind))
            .num_columns(2)
            .spacing([16.0, 6.0])
            .show(ui, |ui| {
                for (name, value) in &snapshot.report.counters {
                    ui.label(name);
                    ui.monospace(value.to_string());
                    ui.end_row();
                }
            });
    }

    if !snapshot.report.highlights.is_empty() {
        ui.separator();
        ui.strong("Highlights");
        egui::Grid::new(format!("report_highlights_{:?}", snapshot.kind))
            .num_columns(2)
            .spacing([16.0, 6.0])
            .show(ui, |ui| {
                for (name, value) in &snapshot.report.highlights {
                    ui.label(name);
                    ui.strong(value);
                    ui.end_row();
                }
            });
    }

    if !snapshot.report.summary.is_empty() {
        ui.separator();
        ui.strong("Summary");
        ui.label(&snapshot.report.summary);
    }

    if !snapshot.report.entries.is_empty() {
        ui.separator();
        ui.strong("Latest Results");
        for entry in &snapshot.report.entries {
            ui.label(format!("• {entry}"));
        }
    }

    if !snapshot.report.events.is_empty() {
        ui.separator();
        ui.strong("Live Events");
        for event in snapshot.report.events.iter().rev() {
            ui.colored_label(
                event_color(event.level),
                format!("• {}", event.message),
            );
        }
    }

    if !snapshot.report.warnings.is_empty() {
        ui.separator();
        ui.colored_label(theme::WARNING, "Warnings");
        for warning in &snapshot.report.warnings {
            ui.label(warning);
        }
    }

    if !snapshot.report.errors.is_empty() {
        ui.separator();
        ui.colored_label(theme::DANGER, "Errors");
        for error in &snapshot.report.errors {
            ui.label(error);
        }
    }

    if let Some(log_path) = snapshot.report.log_path.as_ref() {
        ui.separator();
        ui.label(format!("Log: {log_path}"));
    }
}

pub fn open_log(path: &Path) -> anyhow::Result<()> {
    let (command, args) = open_command(path);
    let status = Command::new(command).args(args).status()?;
    if !status.success() {
        anyhow::bail!("failed to open log path {}", path.display());
    }
    Ok(())
}

fn event_color(level: JobEventLevel) -> egui::Color32 {
    match level {
        JobEventLevel::Info => egui::Color32::LIGHT_BLUE,
        JobEventLevel::Warning => theme::WARNING,
        JobEventLevel::Error => theme::DANGER,
    }
}

#[cfg(target_os = "windows")]
fn open_command(path: &Path) -> (&'static str, Vec<String>) {
    (
        "cmd",
        vec![
            "/C".to_string(),
            "start".to_string(),
            "".to_string(),
            path.display().to_string(),
        ],
    )
}

#[cfg(target_os = "macos")]
fn open_command(path: &Path) -> (&'static str, Vec<String>) {
    ("open", vec![path.display().to_string()])
}

#[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
fn open_command(path: &Path) -> (&'static str, Vec<String>) {
    ("xdg-open", vec![path.display().to_string()])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_services::jobs::JobState;
    use std::path::Path;

    #[test]
    fn status_color_maps_terminal_states() {
        assert_eq!(status_color(JobState::Succeeded), theme::SUCCESS);
        assert_eq!(status_color(JobState::Failed), theme::DANGER);
        assert_eq!(
            status_color(JobState::Cancelled),
            egui::Color32::from_rgb(255, 128, 128)
        );
    }

    #[test]
    fn event_color_maps_event_levels() {
        assert_eq!(event_color(JobEventLevel::Info), egui::Color32::LIGHT_BLUE);
        assert_eq!(event_color(JobEventLevel::Warning), theme::WARNING);
        assert_eq!(event_color(JobEventLevel::Error), theme::DANGER);
    }

    #[test]
    fn dashboard_card_and_section_preserve_values() {
        let card = DashboardCard {
            label: "Adapter".to_string(),
            value: "cTrader".to_string(),
        };
        let section = DashboardSection {
            title: "Runtime".to_string(),
            rows: vec![("Mode".to_string(), "Remote Open API".to_string())],
        };

        assert_eq!(card.label, "Adapter");
        assert_eq!(card.value, "cTrader");
        assert_eq!(section.title, "Runtime");
        assert_eq!(section.rows[0], ("Mode".to_string(), "Remote Open API".to_string()));
    }

    #[test]
    fn open_command_targets_the_canonical_log_path() {
        let path = Path::new("logs/forex-ai.log");
        let (command, args) = open_command(path);

        assert!(!command.is_empty());
        assert!(args.iter().any(|arg| arg.contains("forex-ai.log")));
    }
}
