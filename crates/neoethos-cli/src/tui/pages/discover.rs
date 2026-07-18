//! Discover page — editable parameter form + launch + live progress.
//!
//! v3 (this file): every parameter is a real editable form field.
//! Up/Down navigates fields; Enter on a field opens edit mode (type to
//! modify, Esc to cancel, Enter to commit); when the focus marker is
//! on the LAUNCH row, Enter spawns the subprocess. Mouse: click any
//! field to focus it, click "[ Launch ]" to spawn.

use crossterm::event::KeyCode;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Padding, Paragraph, Widget};

use crate::tui::app::{AppShared, Hit, HitAction};
use crate::tui::jobs::JobStatus;
use crate::tui::theme;

const JOB_LABEL_PREFIX: &str = "discover";

pub fn draw(area: Rect, buf: &mut Buffer, shared: &mut AppShared) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .margin(1)
        .spacing(1)
        .split(area);

    render_form(cols[0], buf, shared);
    render_status(cols[1], buf, shared);
}

fn render_form(area: Rect, buf: &mut Buffer, shared: &mut AppShared) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER))
        .title(Span::styled(
            " PARAMETERS — ↑↓ field · Enter edit/launch · Esc cancel ",
            theme::caption_style().add_modifier(Modifier::BOLD),
        ))
        .style(theme::panel_block_style())
        .padding(Padding::new(2, 2, 1, 1));
    let inner = block.inner(area);
    block.render(area, buf);

    let n_fields = shared.discover_form.fields.len();
    let focused = shared.discover_form.focused;
    let editing = shared.discover_form.editing;

    // Each field uses 2 rows: label/value line + hint line. Plus a
    // blank separator after every field.
    let mut y = inner.y;
    for (idx, field) in shared.discover_form.fields.iter().enumerate() {
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
        // `is_focused` and unfocused both render TEXT_PRIMARY — the focused
        // state is signalled by `value_bg` below (ACCENT_SOFT vs PANEL_BG),
        // not by the value foreground colour. Collapsed 2026-05-26.
        let value_color = if is_editing {
            theme::ACCENT
        } else {
            theme::TEXT_PRIMARY
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
                Style::default().fg(value_color).bg(value_bg),
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

        // Click hit-rect for the field — clicking focuses (and starts
        // editing if already focused).
        shared.hits.push(Hit {
            rect: Rect {
                x: inner.x,
                y,
                width: inner.width,
                height: 1,
            },
            action: HitAction::FocusField {
                page: crate::tui::pages::Page::Discover,
                index: idx,
            },
        });

        // Hint line (read-only, muted).
        if y + 1 < inner.y + inner.height {
            let hint = Paragraph::new(Line::from(vec![
                Span::raw("                  "),
                Span::styled(field.hint, theme::caption_style()),
            ]));
            hint.render(
                Rect {
                    x: inner.x,
                    y: y + 1,
                    width: inner.width,
                    height: 1,
                },
                buf,
            );
        }
        y += 3; // 1 value + 1 hint + 1 blank
        let _ = idx;
        let _ = n_fields;
    }

    // Validation / status message.
    if let Some(msg) = &shared.discover_form.message {
        if y < inner.y + inner.height {
            Paragraph::new(Line::from(vec![Span::styled(
                msg.clone(),
                theme::sell_style(),
            )]))
            .render(
                Rect {
                    x: inner.x,
                    y,
                    width: inner.width,
                    height: 1,
                },
                buf,
            );
            y += 2;
        }
    }

    // Launch / running indicator on the bottom of the form panel.
    if y < inner.y + inner.height {
        let job_alive = shared.jobs.has_running(JOB_LABEL_PREFIX);
        let (text, fg, bg) = if job_alive {
            (
                format!(
                    "  ⏳ Running for {}s — press [K] to Stop · LIVE LOG →  ",
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
                "  [ Launch batch-discover ]   Enter / click  ".to_string(),
                theme::APP_BG,
                theme::ACCENT,
            )
        };
        let launch_line = Line::from(vec![Span::styled(
            text,
            Style::default().fg(fg).bg(bg).add_modifier(Modifier::BOLD),
        )]);
        Paragraph::new(launch_line).render(
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

fn render_status(area: Rect, buf: &mut Buffer, shared: &AppShared) {
    let title = match shared.jobs.latest_for(JOB_LABEL_PREFIX) {
        Some(j) => format!(
            " LIVE LOG · {} · {}s ",
            match j.status {
                JobStatus::Running => "RUNNING",
                JobStatus::Completed => "COMPLETED",
                JobStatus::Failed => "FAILED",
                JobStatus::Stopped => "STOPPED",
            },
            j.elapsed_seconds()
        ),
        None => " LIVE LOG (no run yet) ".to_string(),
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
                } else if lower.contains("portfolio_size=")
                    || lower.contains("found ")
                    || lower.contains("complete")
                {
                    theme::buy_style()
                } else if lower.contains("processing symbol") || lower.contains("timeframe:") {
                    theme::accent_style()
                } else if lower.contains("funnel") || lower.contains("prop-firm") {
                    theme::primary_style()
                } else {
                    theme::muted_style()
                };
                Line::styled(strip_ansi(l), style)
            })
            .collect()
    } else {
        vec![
            Line::raw(""),
            Line::styled(
                "  Edit fields on the left, then press Enter on the",
                theme::muted_style(),
            ),
            Line::styled(
                "  [ Launch ] row — log streams here in real time.",
                theme::muted_style(),
            ),
        ]
    };
    Paragraph::new(lines).render(inner, buf);
}

/// Page-local key handler. Returns true if consumed.
pub fn handle_key(code: KeyCode, shared: &mut AppShared) -> bool {
    let form = &mut shared.discover_form;

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
            // Enter on a focused field → start editing. Launch is
            // bound to 'l' (or the clickable [Launch] pill).
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
            // Stopping a discovery kills the subprocess — stage a Y/N
            // confirmation rather than killing immediately (FIX A). Only prompt
            // if there's actually a running job to stop.
            if shared.jobs.has_running(JOB_LABEL_PREFIX) {
                shared.pending_confirmation =
                    Some(crate::tui::app::PendingAction::DiscoverStop);
                shared.status = "Confirm stop discovery? [Y]es / [N]o".to_string();
            } else {
                shared.status = "No running discovery to stop".to_string();
            }
            true
        }
        _ => false,
    }
}

/// Stop the running discovery (called once the user confirms — FIX A).
pub fn do_stop(shared: &mut AppShared) {
    shared.status = if shared.jobs.stop_latest(JOB_LABEL_PREFIX) {
        "Stopping discovery…".to_string()
    } else {
        "No running discovery to stop".to_string()
    };
}

pub fn launch_now(shared: &mut AppShared) {
    if shared.jobs.has_running(JOB_LABEL_PREFIX) {
        shared.status = "discovery already running".to_string();
        return;
    }
    let form = &shared.discover_form;
    let symbols = form.value_for("Symbols").unwrap_or("").to_string();
    let timeframes = form
        .value_for("Timeframes")
        .unwrap_or("M30,H1,H4,D1")
        .to_string();
    let root = form.value_for("Data root").unwrap_or("data").to_string();
    let out_dir = form
        .value_for("Out dir")
        .unwrap_or("cache/discovery")
        .to_string();

    let mut args = vec![
        "batch-discover".to_string(),
        "--root".to_string(),
        root,
        "--out-dir".to_string(),
        out_dir,
        "--timeframes".to_string(),
        timeframes,
    ];
    if !symbols.trim().is_empty() {
        args.push("--symbols".to_string());
        args.push(symbols);
    }
    // Forward the numeric form fields as explicit overrides so they actually
    // take effect (parity fix: these were silently dropped before, making the
    // form fields dead). Each is passed only when the user entered a value.
    for (field, flag) in [
        ("Population", "--population"),
        ("Generations", "--generations"),
        ("Portfolio size", "--portfolio-size"),
    ] {
        if let Some(v) = form.value_for(field) {
            let v = v.trim();
            if !v.is_empty() && v.parse::<usize>().is_ok() {
                args.push(flag.to_string());
                args.push(v.to_string());
            }
        }
    }

    shared.jobs.spawn("discover", args);
    shared.status = "Spawned batch-discover".to_string();
}

pub(super) fn strip_ansi_for_display(line: &str) -> String {
    strip_ansi(line)
}

fn strip_ansi(line: &str) -> String {
    // Iterate over CHARS, not bytes: pushing `byte as char` re-interprets
    // each raw UTF-8 byte as a Latin-1 codepoint, turning every multi-byte
    // character (→ · ⏳, Greek text) into mojibake in the live-log panels.
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' && chars.peek() == Some(&'[') {
            chars.next(); // consume '['
            for c2 in chars.by_ref() {
                if c2.is_ascii_alphabetic() {
                    break; // CSI final byte ends the escape sequence
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::strip_ansi;

    #[test]
    fn strip_ansi_preserves_multibyte_utf8() {
        assert_eq!(strip_ansi("gen 5 → 6 · ολοκληρώθηκε"), "gen 5 → 6 · ολοκληρώθηκε");
    }

    #[test]
    fn strip_ansi_removes_color_codes() {
        assert_eq!(strip_ansi("\u{1b}[1;32mOK\u{1b}[0m done"), "OK done");
    }
}
