//! Logs page — views the canonical sectioned log (`logs/neoethos.log`).
//! Follows the tail by default; ↑↓/PgUp/PgDn scroll back through history and
//! `F` jumps back to following the newest lines.

use crossterm::event::KeyCode;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Padding, Paragraph, Widget};

use crate::tui::app::AppShared;
use crate::tui::theme;

pub fn handle_key(code: KeyCode, shared: &mut AppShared) -> bool {
    match code {
        // Scroll UP = further into the past (larger offset from the tail).
        KeyCode::Up => {
            shared.logs_scroll = shared.logs_scroll.saturating_add(1);
            true
        }
        KeyCode::Down => {
            shared.logs_scroll = shared.logs_scroll.saturating_sub(1);
            true
        }
        KeyCode::PageUp => {
            shared.logs_scroll = shared.logs_scroll.saturating_add(15);
            true
        }
        KeyCode::PageDown => {
            shared.logs_scroll = shared.logs_scroll.saturating_sub(15);
            true
        }
        // Follow: jump back to the newest lines.
        KeyCode::Char('F') => {
            shared.logs_scroll = 0;
            true
        }
        _ => false,
    }
}

/// Cap on how much of the log file is read per frame. The draw runs in
/// the synchronous render loop (~30 fps) — an unbounded read_to_string
/// of a multi-hundred-MB log from a long discovery run would hammer the
/// disk and freeze the TUI. 2 MB of tail is thousands of lines, far more
/// than the operator can scroll through anyway.
const LOG_TAIL_BYTES: u64 = 2 * 1024 * 1024;

/// Read at most the last `LOG_TAIL_BYTES` of the file, aligned to the
/// first complete line after the seek point. Shared with the Dashboard's
/// recent-activity panel, which reads the same canonical log per frame.
pub(crate) fn read_log_tail(path: &std::path::Path) -> Option<String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).ok()?;
    let len = f.metadata().ok()?.len();
    let truncated = len > LOG_TAIL_BYTES;
    if truncated {
        f.seek(SeekFrom::End(-(LOG_TAIL_BYTES as i64))).ok()?;
    }
    let mut bytes = Vec::with_capacity(LOG_TAIL_BYTES.min(len) as usize + 1);
    f.read_to_end(&mut bytes).ok()?;
    // Lossy-tolerant decode: a mid-file seek can land inside a UTF-8 sequence.
    let mut body = String::from_utf8_lossy(&bytes).into_owned();
    if truncated {
        // Drop the partial first line so styling/alignment stay sane.
        if let Some(nl) = body.find('\n') {
            body.replace_range(..=nl, "");
        }
    }
    Some(body)
}

pub fn draw(area: Rect, buf: &mut Buffer, shared: &AppShared) {
    let log_path = std::path::PathBuf::from("logs").join("neoethos.log");
    let body = read_log_tail(&log_path)
        .unwrap_or_else(|| "(canonical log not found at logs/neoethos.log)".to_string());

    let all: Vec<&str> = body.lines().collect();
    let total = all.len();

    // Reserve the inner height for log lines.
    let block_title = if shared.logs_scroll == 0 {
        " LOGS — logs/neoethos.log · following tail · [↑↓/PgUp/PgDn] scroll ".to_string()
    } else {
        format!(
            " LOGS — {} lines back · [F] follow tail · [↑↓] scroll ",
            shared.logs_scroll
        )
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER))
        .title(Span::styled(
            block_title,
            theme::caption_style().add_modifier(Modifier::BOLD),
        ))
        .style(theme::panel_block_style())
        .padding(Padding::new(2, 2, 1, 1));
    let inner = block.inner(area);
    block.render(area, buf);

    let visible = inner.height.max(1) as usize;
    let max_scroll = total.saturating_sub(visible);
    let scroll = shared.logs_scroll.min(max_scroll);
    let end = total.saturating_sub(scroll);
    let start = end.saturating_sub(visible);

    let lines: Vec<Line> = all[start..end]
        .iter()
        .map(|l| {
            let lower = l.to_lowercase();
            let style =
                if lower.contains("error") || lower.contains("failed") || lower.contains("panic") {
                    theme::sell_style()
                } else if lower.contains("status=success") || lower.contains("complete") {
                    theme::buy_style()
                } else if lower.contains("operation=") || lower.contains("====") {
                    theme::accent_style()
                } else if lower.contains("subsystem=") {
                    theme::primary_style()
                } else {
                    theme::muted_style()
                };
            Line::styled((*l).to_string(), style)
        })
        .collect();
    Paragraph::new(lines).render(inner, buf);
}
