use crate::app_services::jobs::{JobEventLevel, JobSnapshot, JobState};
use eframe::egui;
use std::path::Path;
use std::process::Command;

pub fn status_color(state: JobState) -> egui::Color32 {
    match state {
        JobState::Queued => egui::Color32::GRAY,
        JobState::Running => egui::Color32::YELLOW,
        JobState::Succeeded => egui::Color32::GREEN,
        JobState::Degraded => egui::Color32::from_rgb(255, 165, 0),
        JobState::Failed => egui::Color32::RED,
        JobState::Cancelled => egui::Color32::LIGHT_RED,
    }
}

pub fn render_status_badge(ui: &mut egui::Ui, label: &str, snapshot: Option<&JobSnapshot>) {
    let (text, color) = match snapshot {
        Some(snapshot) => (format!("{label}: {:?}", snapshot.state), status_color(snapshot.state)),
        None => (format!("{label}: Idle"), egui::Color32::GRAY),
    };
    ui.colored_label(color, text);
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
        ui.colored_label(egui::Color32::from_rgb(255, 165, 0), "Warnings");
        for warning in &snapshot.report.warnings {
            ui.label(warning);
        }
    }

    if !snapshot.report.errors.is_empty() {
        ui.separator();
        ui.colored_label(egui::Color32::RED, "Errors");
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
        JobEventLevel::Warning => egui::Color32::from_rgb(255, 165, 0),
        JobEventLevel::Error => egui::Color32::RED,
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
        assert_eq!(status_color(JobState::Succeeded), egui::Color32::GREEN);
        assert_eq!(status_color(JobState::Failed), egui::Color32::RED);
        assert_eq!(status_color(JobState::Cancelled), egui::Color32::LIGHT_RED);
    }

    #[test]
    fn event_color_maps_event_levels() {
        assert_eq!(event_color(JobEventLevel::Info), egui::Color32::LIGHT_BLUE);
        assert_eq!(event_color(JobEventLevel::Warning), egui::Color32::from_rgb(255, 165, 0));
        assert_eq!(event_color(JobEventLevel::Error), egui::Color32::RED);
    }

    #[test]
    fn open_command_targets_the_canonical_log_path() {
        let path = Path::new("logs/forex-ai.log");
        let (command, args) = open_command(path);

        assert!(!command.is_empty());
        assert!(args.iter().any(|arg| arg.contains("forex-ai.log")));
    }
}
