//! Dashboard — landing page. Shows dataset summary, active jobs,
//! recent runs, and KPI cards.

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Padding, Paragraph, Widget, Wrap};

use crate::tui::app::AppShared;
use crate::tui::theme;
use crate::tui::widgets::kpi::Kpi;

pub fn draw(area: Rect, buf: &mut Buffer, shared: &AppShared) {
    // Top row: 4 KPI cards.
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5), // KPI strip
            Constraint::Min(8),    // recent activity
        ])
        .margin(1)
        .split(area);

    let kpi_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .spacing(1)
        .split(rows[0]);

    let (sym_count, tf_count) = dataset_summary(shared);
    Kpi::new("Symbols", sym_count.to_string())
        .sub(format!("{tf_count} timeframes each"))
        .value_style(theme::accent_style())
        .render(kpi_cols[0], buf);

    let active = active_job_count();
    Kpi::new("Active jobs", active.to_string())
        .sub(if active == 0 { "Idle" } else { "Running" })
        .value_style(if active == 0 {
            theme::muted_style().add_modifier(Modifier::BOLD)
        } else {
            theme::buy_style()
        })
        .render(kpi_cols[1], buf);

    let portfolios = portfolio_count(shared);
    Kpi::new("Portfolios", portfolios.to_string())
        .sub("Strategies saved")
        .value_style(theme::accent_style())
        .render(kpi_cols[2], buf);

    let up_min = shared.started_at.elapsed().as_secs() / 60;
    Kpi::new("TUI uptime", format!("{up_min}m"))
        .sub(shared.data_root.display().to_string())
        .value_style(theme::primary_style())
        .render(kpi_cols[3], buf);

    // Bottom: recent activity + getting-started.
    let bottom_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .spacing(1)
        .split(rows[1]);

    render_recent_activity(bottom_cols[0], buf, shared);
    render_quick_start(bottom_cols[1], buf);
}

fn render_recent_activity(area: Rect, buf: &mut Buffer, shared: &AppShared) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER))
        .title(Span::styled(
            " RECENT ACTIVITY ",
            theme::caption_style().add_modifier(Modifier::BOLD),
        ))
        .style(theme::panel_block_style())
        .padding(Padding::new(1, 1, 0, 0));
    let inner = block.inner(area);
    block.render(area, buf);

    let lines = recent_activity_lines(shared);
    let body = if lines.is_empty() {
        vec![Line::styled(
            "No recent runs. Tab to Discover (2) or Train (5) to start one.",
            theme::muted_style(),
        )]
    } else {
        lines
    };
    Paragraph::new(body)
        .wrap(Wrap { trim: true })
        .render(inner, buf);
}

fn render_quick_start(area: Rect, buf: &mut Buffer) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER))
        .title(Span::styled(
            " QUICK START ",
            theme::caption_style().add_modifier(Modifier::BOLD),
        ))
        .style(theme::panel_block_style())
        .padding(Padding::new(1, 1, 0, 0));
    let inner = block.inner(area);
    block.render(area, buf);

    let lines = vec![
        Line::styled(
            "1.  Symbols (4)",
            Style::default()
                .fg(theme::TEXT_PRIMARY)
                .add_modifier(Modifier::BOLD),
        ),
        Line::styled("    inspect dataset inventory", theme::muted_style()),
        Line::raw(""),
        Line::styled(
            "2.  Discover (2)",
            Style::default()
                .fg(theme::TEXT_PRIMARY)
                .add_modifier(Modifier::BOLD),
        ),
        Line::styled("    search for strategies", theme::muted_style()),
        Line::raw(""),
        Line::styled(
            "3.  Strategies (3)",
            Style::default()
                .fg(theme::TEXT_PRIMARY)
                .add_modifier(Modifier::BOLD),
        ),
        Line::styled("    browse + rank discovered", theme::muted_style()),
        Line::raw(""),
        Line::styled(
            "4.  Train (5)",
            Style::default()
                .fg(theme::TEXT_PRIMARY)
                .add_modifier(Modifier::BOLD),
        ),
        Line::styled("    train models on a portfolio", theme::muted_style()),
    ];
    Paragraph::new(lines).render(inner, buf);
}

fn dataset_summary(shared: &AppShared) -> (usize, usize) {
    let mut symbols = 0usize;
    let mut max_tfs = 0usize;
    if let Ok(read) = std::fs::read_dir(&shared.data_root) {
        for entry in read.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !name.starts_with("symbol=") {
                continue;
            }
            symbols += 1;
            let tfs = std::fs::read_dir(entry.path())
                .map(|inner| {
                    inner
                        .flatten()
                        .filter(|e| e.file_name().to_string_lossy().starts_with("timeframe="))
                        .count()
                })
                .unwrap_or(0);
            if tfs > max_tfs {
                max_tfs = tfs;
            }
        }
    }
    (symbols, max_tfs)
}

fn active_job_count() -> usize {
    // No live job-state IPC yet. Count any cli.exe sibling
    // processes via tasklist-like info is not portable; for now
    // we just return 0. Refresh button on the page can later
    // probe a `~/.forex-ai/jobs.json` index.
    0
}

fn portfolio_count(shared: &AppShared) -> usize {
    // Look under <cwd>/cache/discovery and the data root's
    // `cache/` sibling for any *.json portfolio file.
    let mut total = 0;
    for candidate in [
        std::path::PathBuf::from("cache").join("discovery"),
        shared
            .data_root
            .parent()
            .map(|p| p.join("cache"))
            .unwrap_or_default(),
    ] {
        if let Ok(read) = std::fs::read_dir(&candidate) {
            for entry in read.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.ends_with(".json") && !name.contains("profile") {
                    total += 1;
                }
            }
        }
    }
    total
}

fn recent_activity_lines(shared: &AppShared) -> Vec<Line<'static>> {
    // Read the canonical sectioned log if it exists; surface the
    // last 8 distinct entries across SECTION DISCOVERY + CLI.
    let log_path = std::path::PathBuf::from("logs").join("forex-ai.log");
    let mut lines: Vec<Line<'static>> = Vec::new();
    if let Ok(content) = std::fs::read_to_string(&log_path) {
        // Very simple parser — sectioned logs are key=value blocks
        // separated by blank lines + section dividers.
        for block in content.split("--- CURRENT ---").skip(1) {
            // Pull the first 8 lines of this block.
            let mut op = String::new();
            let mut status = String::new();
            let mut msg = String::new();
            for line in block.lines().take(20) {
                if let Some(v) = line.strip_prefix("operation=") {
                    op = v.to_string();
                } else if let Some(v) = line.strip_prefix("status=") {
                    status = v.to_string();
                } else if let Some(v) = line.strip_prefix("message=") {
                    msg = v.to_string();
                    break;
                }
            }
            if op.is_empty() {
                continue;
            }
            let status_style = match status.as_str() {
                "SUCCESS" => theme::buy_style(),
                "FAILED" => theme::sell_style(),
                "STARTED" => theme::accent_style(),
                _ => theme::muted_style(),
            };
            lines.push(Line::from(vec![
                Span::styled(format!("  {:9}  ", status), status_style),
                Span::styled(format!("{:24}  ", op), theme::primary_style()),
                Span::styled(msg, theme::muted_style()),
            ]));
            if lines.len() >= 8 {
                break;
            }
        }
    }
    let _ = shared; // shared not yet used; placeholder for future enrichment.
    lines
}
