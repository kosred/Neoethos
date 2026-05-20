//! Config page — renders [`forex_core::resolved_config::ResolvedConfig`]
//! as a five-column table (section / field / raw / resolved / source).
//! P6: surfaces every backend setting that affects discovery.

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
            " RESOLVED CONFIG — section · field · raw · resolved · source ",
            theme::caption_style().add_modifier(Modifier::BOLD),
        ))
        .style(theme::panel_block_style())
        .padding(Padding::new(2, 2, 1, 1));
    let inner = block.inner(area);
    block.render(area, buf);

    let settings = forex_core::Settings::load().unwrap_or_default();
    let resolved = forex_core::resolved_config::ResolvedConfig::from_settings(&settings);

    let mut lines: Vec<Line> = vec![
        Line::from(vec![
            Span::styled(
                format!(" {:<10} ", "section"),
                theme::caption_style().add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("{:<24} ", "field"),
                theme::caption_style().add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("{:<22} ", "raw"),
                theme::caption_style().add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("{:<22} ", "resolved"),
                theme::caption_style().add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("{:<10}", "source"),
                theme::caption_style().add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::raw(""),
    ];

    for f in &resolved.display_fields {
        let source_color = match f.source {
            forex_core::resolved_config::ResolvedSource::Config => theme::TEXT_PRIMARY,
            forex_core::resolved_config::ResolvedSource::SentinelExpanded => theme::ACCENT,
            forex_core::resolved_config::ResolvedSource::EnvOverride => theme::SELL,
            forex_core::resolved_config::ResolvedSource::Default => theme::TEXT_MUTED,
        };
        lines.push(Line::from(vec![
            Span::styled(format!(" {:<10} ", f.section), theme::muted_style()),
            Span::styled(
                format!("{:<24} ", f.field),
                Style::default()
                    .fg(theme::TEXT_PRIMARY)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("{:<22} ", f.raw), theme::muted_style()),
            Span::styled(
                format!("{:<22} ", f.resolved),
                Style::default().fg(theme::TEXT_PRIMARY),
            ),
            Span::styled(
                format!("{:<10}", f.source.label()),
                Style::default().fg(source_color),
            ),
        ]));
        if let Some(note) = &f.note {
            lines.push(Line::from(vec![
                Span::raw("            "),
                Span::styled(format!("↳ {}", note), theme::caption_style()),
            ]));
        }
    }

    Paragraph::new(lines).render(inner, buf);
}
