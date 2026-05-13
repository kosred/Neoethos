//! KPI card — a small bordered block with a caption, a big value,
//! and an optional sub-label. Used by the Dashboard for "Active
//! jobs", "Last discovery", "Models registered", etc.

use ratatui::{
    buffer::Buffer,
    layout::{Alignment, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Padding, Paragraph, Widget, Wrap},
};

use crate::tui::theme;

pub struct Kpi<'a> {
    pub caption: &'a str,
    pub value: String,
    pub sub: Option<String>,
    pub value_style: Style,
}

impl<'a> Kpi<'a> {
    pub fn new(caption: &'a str, value: impl Into<String>) -> Self {
        Self {
            caption,
            value: value.into(),
            sub: None,
            value_style: theme::title_style(),
        }
    }

    pub fn sub(mut self, sub: impl Into<String>) -> Self {
        self.sub = Some(sub.into());
        self
    }

    pub fn value_style(mut self, style: Style) -> Self {
        self.value_style = style;
        self
    }
}

impl<'a> Widget for Kpi<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::BORDER))
            .style(theme::panel_block_style())
            .padding(Padding::new(1, 1, 0, 0));
        let inner = block.inner(area);
        block.render(area, buf);

        let caption = self.caption.to_uppercase();
        let mut lines = vec![Line::styled(
            caption,
            theme::caption_style().add_modifier(Modifier::BOLD),
        )];
        lines.push(Line::styled(self.value, self.value_style));
        if let Some(sub) = self.sub {
            lines.push(Line::styled(sub, theme::muted_style()));
        }
        Paragraph::new(lines)
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: true })
            .render(inner, buf);
    }
}

/// Force two spans together with a fixed gap. Mostly used for
/// "label: value" pairs in narrow widgets.
pub fn _label_value<'a>(label: &'a str, value: &'a str) -> Line<'a> {
    Line::from(vec![
        Span::styled(label, theme::muted_style()),
        Span::raw("  "),
        Span::styled(value, theme::primary_style()),
    ])
}
