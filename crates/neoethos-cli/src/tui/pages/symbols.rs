//! Symbols — dataset inventory. Reads `<root>/symbol=*/timeframe=*/`
//! and shows a grid of available bars per symbol × timeframe.

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Row, Table, Widget};

use crate::tui::app::AppShared;
use crate::tui::theme;

pub fn draw(area: Rect, buf: &mut Buffer, shared: &AppShared) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER))
        .title(Span::styled(
            " DATASET INVENTORY ",
            theme::caption_style().add_modifier(Modifier::BOLD),
        ))
        .style(theme::panel_block_style());
    let inner = block.inner(area);
    block.render(area, buf);

    let rows = collect_inventory(&shared.data_root);
    if rows.is_empty() {
        let empty = ratatui::widgets::Paragraph::new(vec![
            Line::raw(""),
            Line::styled(
                "  No data found.",
                theme::warn_style().add_modifier(Modifier::BOLD),
            ),
            Line::raw(""),
            Line::styled(
                format!("  Expected layout at: {}", shared.data_root.display()),
                theme::muted_style(),
            ),
            Line::styled(
                "      symbol=EURUSD/timeframe=M1/data.vortex",
                theme::muted_style(),
            ),
        ]);
        empty.render(inner, buf);
        return;
    }

    let header = Row::new(vec![
        Cell::from("SYMBOL").style(theme::caption_style()),
        Cell::from("TIMEFRAMES").style(theme::caption_style()),
        Cell::from("FILES").style(theme::caption_style()),
        Cell::from("SIZE").style(theme::caption_style()),
    ])
    .height(1);

    let body_rows: Vec<Row> = rows
        .into_iter()
        .map(|(sym, tfs, files, bytes)| {
            Row::new(vec![
                Cell::from(sym).style(theme::accent_style()),
                Cell::from(tfs).style(theme::primary_style()),
                Cell::from(files.to_string()).style(theme::muted_style()),
                Cell::from(format_size(bytes)).style(theme::muted_style()),
            ])
            .height(1)
        })
        .collect();

    let widths = [
        Constraint::Length(10),
        Constraint::Min(40),
        Constraint::Length(8),
        Constraint::Length(10),
    ];
    let table = Table::new(body_rows, widths)
        .header(header)
        .column_spacing(2)
        .row_highlight_style(
            Style::default()
                .bg(theme::SURFACE_ALT)
                .add_modifier(Modifier::BOLD),
        );
    Widget::render(table, inner, buf);
}

fn collect_inventory(root: &std::path::Path) -> Vec<(String, String, usize, u64)> {
    let mut out: Vec<(String, String, usize, u64)> = Vec::new();
    if let Ok(read) = std::fs::read_dir(root) {
        for entry in read.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(sym) = name.strip_prefix("symbol=") {
                let mut tfs: Vec<String> = Vec::new();
                let mut files = 0usize;
                let mut bytes = 0u64;
                if let Ok(inner) = std::fs::read_dir(entry.path()) {
                    for tf_entry in inner.flatten() {
                        let tf_name = tf_entry.file_name();
                        let tf_name = tf_name.to_string_lossy();
                        if let Some(tf) = tf_name.strip_prefix("timeframe=") {
                            tfs.push(tf.to_string());
                            if let Ok(files_read) = std::fs::read_dir(tf_entry.path()) {
                                for f in files_read.flatten() {
                                    files += 1;
                                    if let Ok(meta) = f.metadata() {
                                        bytes += meta.len();
                                    }
                                }
                            }
                        }
                    }
                }
                tfs.sort_by(timeframe_sort_key);
                out.push((sym.to_string(), tfs.join(" "), files, bytes));
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn timeframe_sort_key(a: &String, b: &String) -> std::cmp::Ordering {
    // Sort canonical chart timeframes by minute value so the
    // operator reads them as M1, M5, M15, H1, H4, D1, W1, MN1.
    fn to_minutes(tf: &str) -> u64 {
        let (kind, num) = tf.split_at(1);
        let n: u64 = num.parse().unwrap_or(0);
        match kind {
            "M" => n,
            "H" => n * 60,
            "D" => n * 1440,
            "W" => n * 10_080,
            _ => 100_000, // MN1 etc.
        }
    }
    to_minutes(a).cmp(&to_minutes(b))
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.0} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.0} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
