//! Train — editable parameter form + live training log.
//! Same form pattern as Discover: ↑↓ focus · Enter edit · Esc cancel
//! · l (or click [ Launch ]) to spawn the subprocess.

use crossterm::event::KeyCode;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Padding, Paragraph, Widget};

use crate::tui::app::{AppShared, Hit, HitAction};
use crate::tui::jobs::JobStatus;
use crate::tui::theme;

const JOB_LABEL_PREFIX: &str = "train";

pub fn draw(area: Rect, buf: &mut Buffer, shared: &mut AppShared) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .margin(1)
        .spacing(1)
        .split(area);

    render_form(cols[0], buf, shared);
    if shared.jobs.latest_for(JOB_LABEL_PREFIX).is_some() {
        render_live_log(cols[1], buf, shared);
    } else {
        render_registry(cols[1], buf);
    }
}

fn render_form(area: Rect, buf: &mut Buffer, shared: &mut AppShared) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER))
        .title(Span::styled(
            " TRAIN — ↑↓ field · Enter edit · Esc cancel · l launch ",
            theme::caption_style().add_modifier(Modifier::BOLD),
        ))
        .style(theme::panel_block_style())
        .padding(Padding::new(2, 2, 1, 1));
    let inner = block.inner(area);
    block.render(area, buf);

    let focused = shared.train_form.focused;
    let editing = shared.train_form.editing;

    let mut y = inner.y;
    for (idx, field) in shared.train_form.fields.iter().enumerate() {
        if y >= inner.y + inner.height {
            break;
        }
        let is_focused = idx == focused;
        let is_editing = is_focused && editing;
        let marker = if is_focused { ">" } else { " " };
        let value_render = if field.value.is_empty() {
            format!("({})", field.default_value)
        } else {
            field.value.clone()
        };
        let value_with_cursor = if is_editing {
            format!("{}█", value_render)
        } else {
            value_render
        };
        let label_color = if is_focused {
            theme::ACCENT
        } else {
            theme::TEXT_MUTED
        };
        let value_bg = if is_editing {
            theme::SURFACE_ALT
        } else if is_focused {
            theme::ACCENT_SOFT
        } else {
            theme::PANEL_BG
        };
        let line = Line::from(vec![
            Span::styled(
                format!(" {} ", marker),
                Style::default()
                    .fg(theme::ACCENT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("{:>14}  ", field.label),
                Style::default().fg(label_color),
            ),
            Span::styled(
                format!(" {} ", value_with_cursor),
                Style::default().fg(theme::TEXT_PRIMARY).bg(value_bg),
            ),
        ]);
        Paragraph::new(line).render(
            Rect {
                x: inner.x,
                y,
                width: inner.width,
                height: 1,
            },
            buf,
        );
        shared.hits.push(Hit {
            rect: Rect {
                x: inner.x,
                y,
                width: inner.width,
                height: 1,
            },
            action: HitAction::FocusField {
                page: crate::tui::pages::Page::Train,
                index: idx,
            },
        });
        if y + 1 < inner.y + inner.height {
            Paragraph::new(Line::from(vec![
                Span::raw("                  "),
                Span::styled(field.hint, theme::caption_style()),
            ]))
            .render(
                Rect {
                    x: inner.x,
                    y: y + 1,
                    width: inner.width,
                    height: 1,
                },
                buf,
            );
        }
        y += 3;
    }

    if y < inner.y + inner.height {
        let job_alive = shared.jobs.has_running(JOB_LABEL_PREFIX);
        let (text, fg, bg) = if job_alive {
            (
                format!(
                    "  ⏳ Training for {}s — press [K] to Stop · live log →  ",
                    shared
                        .jobs
                        .latest_for(JOB_LABEL_PREFIX)
                        .map(|j| j.elapsed_seconds())
                        .unwrap_or(0)
                ),
                theme::TEXT_MUTED,
                theme::SURFACE_ALT,
            )
        } else {
            (
                "  [ Launch train ]   l / click  ".to_string(),
                theme::APP_BG,
                theme::ACCENT,
            )
        };
        let line = Line::from(vec![Span::styled(
            text,
            Style::default().fg(fg).bg(bg).add_modifier(Modifier::BOLD),
        )]);
        Paragraph::new(line).render(
            Rect {
                x: inner.x,
                y,
                width: inner.width,
                height: 1,
            },
            buf,
        );
        if !job_alive {
            shared.hits.push(Hit {
                rect: Rect {
                    x: inner.x,
                    y,
                    width: inner.width,
                    height: 1,
                },
                action: HitAction::Activate,
            });
        }
    }
}

fn render_live_log(area: Rect, buf: &mut Buffer, shared: &AppShared) {
    let title = match shared.jobs.latest_for(JOB_LABEL_PREFIX) {
        Some(j) => format!(
            " TRAINING · {} · {}s ",
            match j.status {
                JobStatus::Running => "RUNNING",
                JobStatus::Completed => "COMPLETED",
                JobStatus::Failed => "FAILED",
                JobStatus::Stopped => "STOPPED",
            },
            j.elapsed_seconds()
        ),
        None => " TRAINING ".to_string(),
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER))
        .title(Span::styled(
            title,
            theme::caption_style().add_modifier(Modifier::BOLD),
        ))
        .style(theme::panel_block_style())
        .padding(Padding::new(2, 2, 1, 1));
    let inner = block.inner(area);
    block.render(area, buf);

    let lines: Vec<Line> = if let Some(job) = shared.jobs.latest_for(JOB_LABEL_PREFIX) {
        let visible = inner.height.max(1) as usize;
        job.tail(visible)
            .map(|l| {
                let lower = l.to_lowercase();
                let style = if lower.contains("error")
                    || lower.contains("panic")
                    || lower.contains("failed")
                {
                    theme::sell_style()
                } else if lower.contains("complete") || lower.contains("saved") {
                    theme::buy_style()
                } else if lower.contains("epoch") || lower.contains("training") {
                    theme::accent_style()
                } else {
                    theme::muted_style()
                };
                Line::styled(super::discover::strip_ansi_for_display(l), style)
            })
            .collect()
    } else {
        Vec::new()
    };
    Paragraph::new(lines).render(inner, buf);
}

fn render_registry(area: Rect, buf: &mut Buffer) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER))
        .title(Span::styled(
            " MODEL REGISTRY ",
            theme::caption_style().add_modifier(Modifier::BOLD),
        ))
        .style(theme::panel_block_style())
        .padding(Padding::new(2, 2, 1, 1));
    let inner = block.inner(area);
    block.render(area, buf);

    // The store uses the nested models/<SYMBOL>/<TF>/<model_name>/ layout
    // (same contract as the app's scan_models_dir) — listing only the top
    // level would show bare symbol directories and hide every trained
    // model. Walk two levels down and report "SYMBOL/TF/name"; keep any
    // top-level artifact files as-is.
    let mut entries: Vec<String> = Vec::new();
    let root = std::path::PathBuf::from("cache").join("models");
    if let Ok(read) = std::fs::read_dir(&root) {
        for e in read.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with('_') {
                continue; // sentinel files/dirs
            }
            if !e.path().is_dir() {
                entries.push(name);
                continue;
            }
            let mut found_nested = false;
            if let Ok(tfs) = std::fs::read_dir(e.path()) {
                for tf in tfs.flatten() {
                    let tf_name = tf.file_name().to_string_lossy().to_string();
                    if !tf.path().is_dir() || tf_name.starts_with('_') {
                        continue;
                    }
                    if let Ok(models) = std::fs::read_dir(tf.path()) {
                        for m in models.flatten() {
                            let m_name = m.file_name().to_string_lossy().to_string();
                            if m_name.starts_with('_') {
                                continue;
                            }
                            entries.push(format!("{name}/{tf_name}/{m_name}"));
                            found_nested = true;
                        }
                    }
                }
            }
            if !found_nested {
                entries.push(name);
            }
        }
    }
    entries.sort();
    let lines: Vec<Line> = if entries.is_empty() {
        vec![
            Line::raw(""),
            Line::styled(
                "  No trained models in cache/models/ yet.",
                theme::muted_style(),
            ),
            Line::raw(""),
            Line::styled(
                "  Set Symbol/Base TF on the left, then press l",
                theme::muted_style(),
            ),
            Line::styled(
                "  (or click [ Launch ]) to start training.",
                theme::muted_style(),
            ),
        ]
    } else {
        entries
            .into_iter()
            .map(|n| Line::styled(format!("  · {}", n), theme::primary_style()))
            .collect()
    };
    Paragraph::new(lines).render(inner, buf);
}

pub fn handle_key(code: KeyCode, shared: &mut AppShared) -> bool {
    let form = &mut shared.train_form;

    if form.editing {
        match code {
            KeyCode::Esc => {
                form.stop_editing(false);
                shared.status = "Edit cancelled".to_string();
                return true;
            }
            KeyCode::Enter => {
                form.stop_editing(true);
                shared.status = format!(
                    "Saved {} = {}",
                    form.fields[form.focused].label,
                    form.fields[form.focused].effective()
                );
                return true;
            }
            KeyCode::Backspace => {
                form.backspace();
                return true;
            }
            KeyCode::Char(c) => {
                form.type_char(c);
                return true;
            }
            _ => return false,
        }
    }

    match code {
        KeyCode::Up | KeyCode::Char('k') => {
            form.focus_prev();
            true
        }
        KeyCode::Down | KeyCode::Char('j') => {
            form.focus_next();
            true
        }
        KeyCode::Enter => {
            let label = form.fields[form.focused].label;
            form.start_editing();
            shared.status = format!("Editing {} (Esc to cancel)", label);
            true
        }
        KeyCode::Char('x') => {
            form.clear_focused();
            shared.status = "Field cleared".to_string();
            true
        }
        KeyCode::Char('l') | KeyCode::Char('L') => {
            launch_now(shared);
            true
        }
        KeyCode::Char('K') => {
            shared.status = if shared.jobs.stop_latest(JOB_LABEL_PREFIX) {
                "Stopping training…".to_string()
            } else {
                "No running training to stop".to_string()
            };
            true
        }
        _ => false,
    }
}

pub fn launch_now(shared: &mut AppShared) {
    if shared.jobs.has_running(JOB_LABEL_PREFIX) {
        shared.status = "training already running".to_string();
        return;
    }
    let form = &shared.train_form;
    let symbol = form.value_for("Symbol").unwrap_or("EURUSD").to_string();
    let base = form.value_for("Base TF").unwrap_or("M30").to_string();
    let root = form.value_for("Data root").unwrap_or("data").to_string();
    let models_dir = form
        .value_for("Models dir")
        .unwrap_or("cache/models")
        .to_string();

    // train command doesn't honor --root; the training_orchestrator reads
    // NEOETHOS_BOT_DATA_ROOT instead. We inject it on the child subprocess
    // only (via Command::env in spawn_with_env) — NEVER on the parent TUI
    // process. The TUI is already multi-threaded by the time this runs
    // (tokio runtime, rayon worker pool, ratatui input thread, ...) and
    // per std::env::set_var docs, on Linux/macOS the only safe option is
    // to never mutate the parent env after threads have spawned.
    let args = vec![
        "train".to_string(),
        "--symbol".to_string(),
        symbol,
        "--base".to_string(),
        base,
        "--models-dir".to_string(),
        models_dir,
    ];
    let envs = vec![("NEOETHOS_BOT_DATA_ROOT".to_string(), root)];
    shared.jobs.spawn_with_env("train", args, envs);
    shared.status = "Spawned train".to_string();
}
