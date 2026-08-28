//! Canonical Native Research control page.
//!
//! Every action is a short-lived CLI HTTP client subprocess. The app process
//! remains the sole owner of the Native Research handle and execution lease;
//! cancellation first reads the live opaque token and then requests an exact
//! token match through the dedicated native endpoint.

use crossterm::event::KeyCode;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Padding, Paragraph, Widget};

use crate::tui::app::{AppShared, Hit, HitAction, PendingAction};
use crate::tui::form::{Field, FormState};
use crate::tui::jobs::JobStatus;
use crate::tui::theme;

const JOB_LABEL_PREFIX: &str = "native-research";

pub fn make_form() -> FormState {
    FormState::new(vec![
        Field::new(
            "Contract path",
            "",
            "Required path relative to the sealed canonical authority root",
        ),
        Field::new(
            "Expected SHA-256",
            "",
            "Required exact 64-character lowercase artifact digest",
        ),
        Field::new(
            "Population",
            "",
            "Optional positive override; blank preserves the contract/settings value",
        ),
        Field::new(
            "Population auto",
            "",
            "Optional true/false override; blank preserves the contract/settings value",
        ),
        Field::new(
            "Max indicators",
            "",
            "Optional positive override; blank preserves the contract/settings value",
        ),
        Field::new(
            "API base",
            "",
            "Blank reads the running desktop port, then falls back to loopback:7423",
        ),
    ])
}

pub fn draw(area: Rect, buf: &mut Buffer, shared: &mut AppShared) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(47), Constraint::Percentage(53)])
        .margin(1)
        .spacing(1)
        .split(area);
    render_form(columns[0], buf, shared);
    render_status(columns[1], buf, shared);
}

fn render_form(area: Rect, buf: &mut Buffer, shared: &mut AppShared) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER))
        .title(Span::styled(
            " NATIVE RESEARCH — exact contract · no model handoff ",
            theme::caption_style().add_modifier(Modifier::BOLD),
        ))
        .style(theme::panel_block_style())
        .padding(Padding::new(2, 2, 1, 1));
    let inner = block.inner(area);
    block.render(area, buf);

    let focused = shared.native_research_form.focused;
    let editing = shared.native_research_form.editing;
    let mut y = inner.y;
    for (index, field) in shared.native_research_form.fields.iter().enumerate() {
        if y >= inner.y + inner.height {
            break;
        }
        let active = index == focused;
        let value = if field.value.is_empty() {
            "(blank)".to_owned()
        } else if field.label == "Expected SHA-256" && field.value.len() > 18 {
            format!("{}…", field.value.chars().take(17).collect::<String>())
        } else {
            field.value.clone()
        };
        let value = if active && editing {
            format!("{value}█")
        } else {
            value
        };
        Paragraph::new(Line::from(vec![
            Span::styled(
                if active { " > " } else { "   " },
                Style::default()
                    .fg(theme::ACCENT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("{:>18}  ", field.label),
                Style::default().fg(if active {
                    theme::ACCENT
                } else {
                    theme::TEXT_MUTED
                }),
            ),
            Span::styled(
                value,
                Style::default().fg(theme::TEXT_PRIMARY).bg(if active {
                    theme::ACCENT_SOFT
                } else {
                    theme::PANEL_BG
                }),
            ),
        ]))
        .render(
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
                page: crate::tui::pages::Page::NativeResearch,
                index,
            },
        });
        if y + 1 < inner.y + inner.height {
            Paragraph::new(Line::styled(field.hint.as_str(), theme::caption_style())).render(
                Rect {
                    x: inner.x + 3,
                    y: y + 1,
                    width: inner.width.saturating_sub(3),
                    height: 1,
                },
                buf,
            );
        }
        y += 3;
    }

    if y < inner.y + inner.height {
        Paragraph::new(Line::from(vec![
            Span::styled(
                " [L] Start ",
                Style::default()
                    .fg(theme::APP_BG)
                    .bg(theme::ACCENT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled("[S] Status", theme::accent_style()),
            Span::raw("  "),
            Span::styled("[K] Cancel", theme::sell_style()),
        ]))
        .render(
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
                width: 11.min(inner.width),
                height: 1,
            },
            action: HitAction::Activate,
        });
    }
}

fn render_status(area: Rect, buf: &mut Buffer, shared: &AppShared) {
    let title = match shared.jobs.latest_for(JOB_LABEL_PREFIX) {
        Some(job) => format!(
            " APP-OWNED NATIVE STATUS · {} ",
            match job.status {
                JobStatus::Running => "REQUESTING",
                JobStatus::Completed => "UPDATED",
                JobStatus::Failed => "REQUEST FAILED",
                JobStatus::Stopped => "CLIENT STOPPED",
            }
        ),
        None => " APP-OWNED NATIVE STATUS · press S ".to_owned(),
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

    let lines: Vec<Line> = match shared.jobs.latest_for(JOB_LABEL_PREFIX) {
        Some(job) => job
            .tail(inner.height.max(1) as usize)
            .map(|line| {
                let clean = super::discover::strip_ansi_for_display(line);
                let lower = clean.to_ascii_lowercase();
                let style = if lower.contains("failure") || lower.contains("rejected") {
                    theme::sell_style()
                } else if lower.contains("published") || lower.contains("accepted") {
                    theme::buy_style()
                } else {
                    theme::muted_style()
                };
                Line::styled(clean, style)
            })
            .collect(),
        None => vec![
            Line::styled(
                "Status is read from /engines/status in the running app process.",
                theme::muted_style(),
            ),
            Line::styled(
                "Published evidence and stable failure stage are bounded before display.",
                theme::muted_style(),
            ),
        ],
    };
    Paragraph::new(lines).render(inner, buf);
}

pub fn handle_key(code: KeyCode, shared: &mut AppShared) -> bool {
    let form = &mut shared.native_research_form;
    if form.editing {
        match code {
            KeyCode::Enter => form.stop_editing(true),
            KeyCode::Esc => form.stop_editing(false),
            KeyCode::Backspace => form.backspace(),
            KeyCode::Char(character) => form.type_char(character),
            KeyCode::Up => form.focus_prev(),
            KeyCode::Down | KeyCode::Tab => form.focus_next(),
            _ => {}
        }
        return true;
    }
    match code {
        KeyCode::Up => form.focus_prev(),
        KeyCode::Down => form.focus_next(),
        KeyCode::Enter => form.start_editing(),
        KeyCode::Char('l') | KeyCode::Char('L') => launch_now(shared),
        KeyCode::Char('s') | KeyCode::Char('S') => request_status(shared),
        KeyCode::Char('k') | KeyCode::Char('K') => {
            shared.pending_confirmation = Some(PendingAction::NativeResearchCancel);
            shared.status =
                "Confirm exact-token Native Research cancellation? [Y]es / [N]o".to_owned();
        }
        _ => return false,
    }
    true
}

pub fn launch_now(shared: &mut AppShared) {
    if shared.jobs.has_running(JOB_LABEL_PREFIX) {
        shared.status = "A Native Research control request is already in flight".to_owned();
        return;
    }
    match start_args(&shared.native_research_form) {
        Ok(args) => {
            shared.jobs.spawn("native-research-start", args);
            shared.status = "Requested canonical Native Research start".to_owned();
        }
        Err(message) => {
            shared.native_research_form.message = Some(message.clone());
            shared.status = message;
        }
    }
}

fn request_status(shared: &mut AppShared) {
    let mut args = vec!["native-research".to_string(), "status".to_string()];
    append_api_base(&shared.native_research_form, &mut args);
    shared.jobs.spawn("native-research-status", args);
    shared.status = "Refreshing canonical Native Research status".to_owned();
}

pub fn do_cancel(shared: &mut AppShared) {
    let mut args = vec!["native-research".to_string(), "cancel".to_string()];
    append_api_base(&shared.native_research_form, &mut args);
    shared.jobs.spawn("native-research-cancel", args);
    shared.status = "Requesting exact-token Native Research cancellation".to_owned();
}

fn start_args(form: &FormState) -> Result<Vec<String>, String> {
    let path = form.value_for("Contract path").unwrap_or("").trim();
    if path.is_empty() {
        return Err("Contract path is required".to_owned());
    }
    let sha = form.value_for("Expected SHA-256").unwrap_or("").trim();
    if sha.len() != 64
        || !sha
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("Expected SHA-256 must be exactly 64 lowercase hex characters".to_owned());
    }
    let mut args = vec![
        "native-research".to_string(),
        "start".to_string(),
        "--contract-relative-path".to_string(),
        path.to_owned(),
        "--expected-sha256".to_string(),
        sha.to_owned(),
    ];
    append_positive_override(form, "Population", "--population", &mut args)?;
    if let Some(value) = nonblank(form, "Population auto") {
        if !matches!(value, "true" | "false") {
            return Err("Population auto must be true, false, or blank".to_owned());
        }
        args.extend(["--population-auto".to_owned(), value.to_owned()]);
    }
    append_positive_override(form, "Max indicators", "--max-indicators", &mut args)?;
    append_api_base(form, &mut args);
    Ok(args)
}

fn append_positive_override(
    form: &FormState,
    label: &str,
    flag: &str,
    args: &mut Vec<String>,
) -> Result<(), String> {
    let Some(value) = nonblank(form, label) else {
        return Ok(());
    };
    if value
        .parse::<usize>()
        .ok()
        .filter(|number| *number > 0)
        .is_none()
    {
        return Err(format!("{label} must be a positive integer or blank"));
    }
    args.extend([flag.to_owned(), value.to_owned()]);
    Ok(())
}

fn append_api_base(form: &FormState, args: &mut Vec<String>) {
    if let Some(base) = nonblank(form, "API base") {
        args.extend(["--api-base".to_owned(), base.to_owned()]);
    }
}

fn nonblank<'a>(form: &'a FormState, label: &str) -> Option<&'a str> {
    form.value_for(label)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_args_target_only_the_native_command_and_keep_optional_values_explicit() {
        let mut form = make_form();
        form.fields[0].value = "contracts/run.json".to_owned();
        form.fields[1].value = "ab".repeat(32);
        form.fields[2].value = "2048".to_owned();
        form.fields[3].value = "false".to_owned();
        form.fields[4].value = "17".to_owned();
        form.fields[5].value = "http://127.0.0.1:54321".to_owned();

        let args = start_args(&form).expect("valid form");
        assert_eq!(&args[0..2], ["native-research", "start"]);
        assert!(args.windows(2).any(|pair| pair == ["--population", "2048"]));
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--population-auto", "false"])
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--max-indicators", "17"])
        );
        assert!(!args.iter().any(|arg| arg.contains("discover")));
        assert!(!args.iter().any(|arg| arg == "train"));
    }
}
