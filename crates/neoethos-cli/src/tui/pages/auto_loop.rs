//! Auto-loop page — shows the current checkpoint
//! (`cache/auto_loop_checkpoint.json`) and lets the operator launch
//! `neoethos-cli auto-loop` from the TUI with the same JobManager that
//! Discover/Train use.

use crossterm::event::KeyCode;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Padding, Paragraph, Widget};

use crate::tui::app::{AppShared, Hit, HitAction};
use crate::tui::jobs::JobStatus;
use crate::tui::theme;

const JOB_LABEL_PREFIX: &str = "auto-loop";
/// Graceful-stop flag the auto-loop polls between jobs. Creating it (FIX D's
/// `K`) tells a running loop to halt after the current job finishes.
const STOP_FLAG_PATH: &str = "cache/auto_loop_stop.flag";

pub fn draw(area: Rect, buf: &mut Buffer, shared: &mut AppShared) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .margin(1)
        .spacing(1)
        .split(area);

    render_controls(cols[0], buf, shared);
    render_state(cols[1], buf, shared);
}

fn render_controls(area: Rect, buf: &mut Buffer, shared: &mut AppShared) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER))
        .title(Span::styled(
            " AUTO-LOOP CONTROLS ",
            theme::caption_style().add_modifier(Modifier::BOLD),
        ))
        .style(theme::panel_block_style())
        .padding(Padding::new(2, 2, 1, 1));
    let inner = block.inner(area);
    block.render(area, buf);

    let job_alive = shared.jobs.has_running(JOB_LABEL_PREFIX);
    let stop_flag = std::path::Path::new(STOP_FLAG_PATH);

    let mut lines = vec![
        Line::raw(""),
        Line::from(vec![
            Span::styled("  Pipeline: ", theme::caption_style()),
            Span::styled(
                "import → discover → train → next (sym, tf)",
                theme::primary_style(),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Checkpoint: ", theme::caption_style()),
            Span::styled("cache/auto_loop_checkpoint.json", theme::primary_style()),
        ]),
        Line::from(vec![
            Span::styled("  Stop flag: ", theme::caption_style()),
            Span::styled(
                if stop_flag.exists() {
                    "PRESENT — loop will stop after current job"
                } else {
                    "absent (press K for graceful stop)"
                },
                if stop_flag.exists() {
                    theme::sell_style()
                } else {
                    theme::muted_style()
                },
            ),
        ]),
        Line::raw(""),
    ];

    let launch_y = inner.y + lines.len() as u16;
    if job_alive {
        lines.push(Line::from(vec![Span::styled(
            "  ⏳ auto-loop running — see live log on the right  ",
            Style::default()
                .fg(theme::TEXT_MUTED)
                .bg(theme::SURFACE_ALT)
                .add_modifier(Modifier::BOLD),
        )]));
    } else {
        lines.push(Line::from(vec![Span::styled(
            "  [ Start auto-loop ]   click / press L  ",
            Style::default()
                .fg(theme::APP_BG)
                .bg(theme::ACCENT)
                .add_modifier(Modifier::BOLD),
        )]));
    }

    Paragraph::new(lines).render(inner, buf);

    if !job_alive && launch_y < inner.y + inner.height {
        shared.hits.push(Hit {
            rect: Rect {
                x: inner.x,
                y: launch_y,
                width: inner.width,
                height: 1,
            },
            action: HitAction::Activate,
        });
    }
}

fn render_state(area: Rect, buf: &mut Buffer, shared: &AppShared) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER))
        .title(Span::styled(
            " CHECKPOINT / LIVE LOG ",
            theme::caption_style().add_modifier(Modifier::BOLD),
        ))
        .style(theme::panel_block_style())
        .padding(Padding::new(2, 2, 1, 1));
    let inner = block.inner(area);
    block.render(area, buf);

    let job = shared.jobs.latest_for(JOB_LABEL_PREFIX);

    let mut lines: Vec<Line> = Vec::new();
    if let Ok(text) = std::fs::read_to_string("cache/auto_loop_checkpoint.json") {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
            let started = v.get("started_at").and_then(|s| s.as_str()).unwrap_or("?");
            let updated = v.get("updated_at").and_then(|s| s.as_str()).unwrap_or("?");
            let remaining = v.get("remaining").and_then(|n| n.as_u64()).unwrap_or(0);
            let completed = v
                .get("completed")
                .and_then(|c| c.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            lines.push(Line::from(vec![
                Span::styled("  started:   ", theme::caption_style()),
                Span::styled(started.to_string(), theme::primary_style()),
            ]));
            lines.push(Line::from(vec![
                Span::styled("  updated:   ", theme::caption_style()),
                Span::styled(updated.to_string(), theme::primary_style()),
            ]));
            lines.push(Line::from(vec![
                Span::styled("  completed: ", theme::caption_style()),
                Span::styled(completed.to_string(), theme::buy_style()),
                Span::styled("   remaining: ", theme::caption_style()),
                Span::styled(remaining.to_string(), theme::accent_style()),
            ]));
            lines.push(Line::raw(""));
        }
    } else {
        lines.push(Line::styled(
            "  No checkpoint yet (no auto-loop has run)",
            theme::muted_style(),
        ));
        lines.push(Line::raw(""));
    }

    if let Some(job) = job {
        lines.push(Line::styled(
            format!(
                "  Live job — {} · {}s",
                match job.status {
                    JobStatus::Running => "RUNNING",
                    JobStatus::Completed => "COMPLETED",
                    JobStatus::Failed => "FAILED",
                    JobStatus::Stopped => "STOPPED",
                },
                job.elapsed_seconds()
            ),
            theme::caption_style().add_modifier(Modifier::BOLD),
        ));
        lines.push(Line::raw(""));
        let visible = (inner.height as usize).saturating_sub(lines.len());
        for l in job.tail(visible) {
            lines.push(Line::styled(
                super::discover::strip_ansi_for_display(l),
                theme::muted_style(),
            ));
        }
    }

    Paragraph::new(lines).render(inner, buf);
}

pub fn handle_key(code: KeyCode, shared: &mut AppShared) -> bool {
    match code {
        KeyCode::Char('l') | KeyCode::Char('L') | KeyCode::Enter => {
            launch_now(shared);
            true
        }
        KeyCode::Char('K') => {
            // Graceful-stop: create the stop flag the auto-loop polls between
            // jobs (FIX D), mirroring how Discover's K stops a run. Behind the
            // FIX-A Y/N confirmation since it changes a long-running pipeline.
            if std::path::Path::new(STOP_FLAG_PATH).exists() {
                shared.status = "Stop flag already present — loop will stop after current job"
                    .to_string();
            } else if shared.jobs.has_running(JOB_LABEL_PREFIX) {
                shared.pending_confirmation =
                    Some(crate::tui::app::PendingAction::AutoLoopStop);
                shared.status = "Confirm graceful stop (create stop flag)? [Y]es / [N]o".to_string();
            } else {
                // No live job, but still let the operator pre-arm the flag so a
                // resumed loop stops; confirm anyway for consistency.
                shared.pending_confirmation =
                    Some(crate::tui::app::PendingAction::AutoLoopStop);
                shared.status = "Confirm create stop flag? [Y]es / [N]o".to_string();
            }
            true
        }
        _ => false,
    }
}

/// Create the auto-loop stop flag so the loop halts gracefully after the
/// current job (called once the user confirms — FIX D). The auto-loop polls for
/// this file between jobs; touching it is the documented graceful-stop path.
pub fn do_create_stop_flag(shared: &mut AppShared) {
    let path = std::path::Path::new(STOP_FLAG_PATH);
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            shared.status = format!("Could not create {}: {e}", parent.display());
            return;
        }
    }
    match std::fs::write(path, b"stop\n") {
        Ok(()) => {
            shared.status =
                "Stop flag created — auto-loop will halt after the current job".to_string();
        }
        Err(e) => {
            shared.status = format!("Failed to write {STOP_FLAG_PATH}: {e}");
        }
    }
}

pub fn launch_now(shared: &mut AppShared) {
    if shared.jobs.has_running(JOB_LABEL_PREFIX) {
        shared.status = "auto-loop already running".to_string();
        return;
    }
    let root = shared.data_root.display().to_string();
    let args = vec!["auto-loop".to_string(), "--root".to_string(), root];
    shared.jobs.spawn("auto-loop", args);
    shared.status = "Spawned auto-loop".to_string();
}
