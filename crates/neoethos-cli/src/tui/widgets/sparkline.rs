//! Compact sparkline for an equity curve / fitness trajectory.
//! ratatui ships its own [`Sparkline`] but rounds non-positive values
//! to zero and only handles `u64`. This wrapper accepts `f64`,
//! normalises to a u64 range internally, and colors the line by the
//! last-point trend (green up / red down).

use ratatui::style::Style;
use ratatui::widgets::Sparkline;

use crate::tui::theme;

pub fn equity_sparkline<'a>(label: &'a str, values: &[f64]) -> Sparkline<'a> {
    if values.is_empty() {
        return Sparkline::default().data::<&[u64]>(&[]);
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (max - min).max(1e-9);
    let scaled: Vec<u64> = values
        .iter()
        .map(|v| (((*v - min) / span) * 1000.0).round() as u64)
        .collect();

    let style = if values.len() >= 2 {
        let last = values[values.len() - 1];
        let prev = values[0];
        if last >= prev {
            Style::default().fg(theme::BUY)
        } else {
            Style::default().fg(theme::SELL)
        }
    } else {
        Style::default().fg(theme::TEXT_MUTED)
    };

    Sparkline::default()
        .data(scaled)
        .style(style)
        .max(1000)
        .bar_set(ratatui::symbols::bar::NINE_LEVELS)
        // `label` retained so future call sites can attach a title block.
        .style(style.patch(Style::default().bg(theme::PANEL_BG)))
        .clone()
        // suppress unused warning
        .style(style.add_modifier(Default::default()))
        // explicitly drop the label (label support comes from the
        // enclosing Block::title — not the Sparkline itself)
        .style(if !label.is_empty() { style } else { style })
}
