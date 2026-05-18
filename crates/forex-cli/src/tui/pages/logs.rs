//! Logs page — tails the canonical sectioned log
//! (`logs/forex-ai.log`) with simple filter buckets so the operator
//! can see what every subsystem has emitted lately.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Padding, Paragraph, Widget};

use crate::tui::app::AppShared;
use crate::tui::theme;

pub fn draw(area: Rect, buf: &mut Buffer, _shared: &AppShared) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER))
        .title(Span::styled(
            " LOGS — logs/forex-ai.log (last 200 lines) ",
            theme::caption_style().add_modifier(Modifier::BOLD),
        ))
        .style(theme::panel_block_style())
        .padding(Padding::new(2, 2, 1, 1));
    let inner = block.inner(area);
    block.render(area, buf);

    let log_path = std::path::PathBuf::from("logs").join("forex-ai.log");
    let body = std::fs::read_to_string(&log_path)
        .unwrap_or_else(|_| "(canonical log not found at logs/forex-ai.log)".to_string());

    let visible = inner.height.max(1) as usize;
    let lines: Vec<Line> = body
        .lines()
        .rev()
        .take(visible)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
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
            Line::styled(l.to_string(), style)
        })
        .collect();
    Paragraph::new(lines).render(inner, buf);
}
