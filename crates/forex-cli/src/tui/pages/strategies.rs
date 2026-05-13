//! Strategies — portfolio browser. Reads `cache/discovery/*.json`
//! produced by `batch-discover` and shows them in a sortable table.

use std::path::PathBuf;

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Padding, Paragraph, Row, Table, Widget};

use crate::tui::app::AppShared;
use crate::tui::theme;

pub fn draw(area: Rect, buf: &mut Buffer, _shared: &AppShared) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER))
        .title(Span::styled(
            " STRATEGY PORTFOLIOS ",
            theme::caption_style().add_modifier(Modifier::BOLD),
        ))
        .style(theme::panel_block_style())
        .padding(Padding::new(1, 1, 0, 0));
    let inner = block.inner(area);
    block.render(area, buf);

    let portfolios = scan_portfolios();
    if portfolios.is_empty() {
        let lines = vec![
            Line::raw(""),
            Line::styled(
                "  No portfolios saved yet.",
                theme::warn_style().add_modifier(Modifier::BOLD),
            ),
            Line::raw(""),
            Line::styled(
                "  Run a discovery from the Discover page (2) or:",
                theme::muted_style(),
            ),
            Line::raw(""),
            Line::styled(
                "    forex-cli batch-discover --root <data> --out-dir cache/discovery",
                theme::accent_style(),
            ),
            Line::raw(""),
            Line::styled(
                "  Results land under  cache/discovery/<SYMBOL>_<TF>.json",
                theme::caption_style(),
            ),
        ];
        Paragraph::new(lines).render(inner, buf);
        return;
    }

    let header = Row::new(vec![
        Cell::from("PORTFOLIO").style(theme::caption_style()),
        Cell::from("STRATEGIES").style(theme::caption_style()),
        Cell::from("SIZE").style(theme::caption_style()),
        Cell::from("MODIFIED").style(theme::caption_style()),
    ])
    .height(1);

    let rows: Vec<Row> = portfolios
        .into_iter()
        .map(|p| {
            Row::new(vec![
                Cell::from(p.name).style(theme::accent_style()),
                Cell::from(p.strategies.to_string()).style(theme::primary_style()),
                Cell::from(format_size(p.bytes)).style(theme::muted_style()),
                Cell::from(p.modified).style(theme::muted_style()),
            ])
            .height(1)
        })
        .collect();

    let widths = [
        Constraint::Min(28),
        Constraint::Length(12),
        Constraint::Length(10),
        Constraint::Length(20),
    ];
    let table = Table::new(rows, widths)
        .header(header)
        .column_spacing(2)
        .row_highlight_style(
            Style::default()
                .bg(theme::SURFACE_ALT)
                .add_modifier(Modifier::BOLD),
        );
    Widget::render(table, inner, buf);
}

struct PortfolioSummary {
    name: String,
    strategies: usize,
    bytes: u64,
    modified: String,
}

fn scan_portfolios() -> Vec<PortfolioSummary> {
    let mut out: Vec<PortfolioSummary> = Vec::new();
    let candidates = [
        PathBuf::from("cache").join("discovery"),
        PathBuf::from("cache"),
    ];
    for dir in candidates.iter() {
        let Ok(read) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in read.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy().to_string();
            if !name_str.ends_with(".json") {
                continue;
            }
            // Skip the profile/quality/trades sidecars produced by
            // the orchestrator — they are not portfolios.
            if name_str.contains("_profile")
                || name_str.contains("_quality")
                || name_str.contains("_trade_logs")
                || name_str.ends_with(".trades.json")
                || name_str.ends_with(".quality.json")
                || name_str.ends_with(".profile.json")
            {
                continue;
            }
            let path = entry.path();
            let bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
            let strategies = count_strategies(&path);
            let modified = entry
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| format_ts(d.as_secs()))
                .unwrap_or_else(|| "—".to_string());
            out.push(PortfolioSummary {
                name: name_str,
                strategies,
                bytes,
                modified,
            });
        }
    }
    out.sort_by(|a, b| b.modified.cmp(&a.modified));
    out
}

fn count_strategies(path: &std::path::Path) -> usize {
    // Cheap: assume the file is a JSON array and count top-level commas + 1.
    // For empty arrays we return 0. For non-array files we return 0.
    let Ok(text) = std::fs::read_to_string(path) else {
        return 0;
    };
    let trimmed = text.trim_start();
    if !trimmed.starts_with('[') {
        return 0;
    }
    // Empty array fast-path.
    if trimmed.starts_with("[]") {
        return 0;
    }
    // Count top-level `{` openings (each strategy is one object).
    let mut depth: i32 = 0;
    let mut count = 0;
    for ch in trimmed.chars() {
        match ch {
            '{' => {
                if depth == 0 {
                    count += 1;
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
            }
            _ => {}
        }
    }
    count
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.0} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

fn format_ts(unix: u64) -> String {
    let day = unix % 86_400;
    let h = day / 3600;
    let m = (day % 3600) / 60;
    let s = day % 60;
    // We do not track timezone; this is a wall-clock UTC HH:MM:SS
    // good enough for "is this fresh?" — full date support would
    // need a chrono dep we have not added to forex-cli.
    format!("{h:02}:{m:02}:{s:02} UTC")
}
