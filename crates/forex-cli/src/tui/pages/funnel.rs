//! Funnel page — reads the most recent `*_funnel.json` files written
//! by discovery and renders the 16-stage rejection funnel as a stacked
//! count-in/out/rejected table.

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Padding, Paragraph, Widget};

use crate::tui::app::AppShared;
use crate::tui::theme;

pub fn draw(area: Rect, buf: &mut Buffer, _shared: &AppShared) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .margin(1)
        .spacing(1)
        .split(area);

    render_run_list(cols[0], buf);
    render_latest_funnel(cols[1], buf);
}

fn render_run_list(area: Rect, buf: &mut Buffer) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER))
        .title(Span::styled(
            " RECENT FUNNELS ",
            theme::caption_style().add_modifier(Modifier::BOLD),
        ))
        .style(theme::panel_block_style())
        .padding(Padding::new(2, 2, 1, 1));
    let inner = block.inner(area);
    block.render(area, buf);

    let funnels = collect_funnel_files();
    let lines: Vec<Line> = if funnels.is_empty() {
        vec![
            Line::raw(""),
            Line::styled("  No funnel JSONs found.", theme::muted_style()),
            Line::raw(""),
            Line::styled(
                "  Run discover/batch-discover; each work-unit",
                theme::muted_style(),
            ),
            Line::styled(
                "  saves <symbol>_<tf>_funnel.json next to its",
                theme::muted_style(),
            ),
            Line::styled(
                "  portfolio JSON. They show every gate that",
                theme::muted_style(),
            ),
            Line::styled(
                "  rejected candidates and the bottleneck stage.",
                theme::muted_style(),
            ),
        ]
    } else {
        funnels
            .iter()
            .take(20)
            .map(|f| Line::styled(format!("  · {}", f.display()), theme::primary_style()))
            .collect()
    };
    Paragraph::new(lines).render(inner, buf);
}

fn render_latest_funnel(area: Rect, buf: &mut Buffer) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER))
        .title(Span::styled(
            " LATEST FUNNEL — count_in · count_out · rejected ",
            theme::caption_style().add_modifier(Modifier::BOLD),
        ))
        .style(theme::panel_block_style())
        .padding(Padding::new(2, 2, 1, 1));
    let inner = block.inner(area);
    block.render(area, buf);

    let funnels = collect_funnel_files();
    let Some(latest) = funnels.first() else {
        Paragraph::new(vec![
            Line::raw(""),
            Line::styled("  (no funnel data yet)", theme::muted_style()),
        ])
        .render(inner, buf);
        return;
    };

    let lines: Vec<Line> = match std::fs::read_to_string(latest) {
        Ok(text) => match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(v) => render_funnel_value(&v, latest),
            Err(_) => vec![Line::styled("  (parse failed)", theme::sell_style())],
        },
        Err(_) => vec![Line::styled("  (read failed)", theme::sell_style())],
    };
    Paragraph::new(lines).render(inner, buf);
}

fn render_funnel_value(v: &serde_json::Value, path: &std::path::Path) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    let symbol = v.get("symbol").and_then(|s| s.as_str()).unwrap_or("?");
    let tf = v.get("timeframe").and_then(|s| s.as_str()).unwrap_or("?");
    let outcome = v.get("outcome").and_then(|s| s.as_str()).unwrap_or("?");
    let bottleneck = v
        .get("bottleneck_stage")
        .and_then(|s| s.as_str())
        .unwrap_or("");
    let bottleneck_rejected = v
        .get("bottleneck_rejected")
        .and_then(|n| n.as_u64())
        .unwrap_or(0);

    out.push(Line::from(vec![
        Span::styled(
            format!("  {} {} ", symbol, tf),
            Style::default()
                .fg(theme::ACCENT)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("· outcome={} ", outcome),
            Style::default().fg(if outcome == "export_ready" {
                theme::BUY
            } else {
                theme::SELL
            }),
        ),
        Span::styled(
            format!(
                "· bottleneck={} ({} rejected)",
                bottleneck, bottleneck_rejected
            ),
            theme::muted_style(),
        ),
    ]));
    out.push(Line::raw(""));

    if let Some(stages) = v.get("stages").and_then(|s| s.as_array()) {
        for s in stages {
            let name = s.get("name").and_then(|n| n.as_str()).unwrap_or("?");
            let cin = s.get("count_in").and_then(|n| n.as_u64()).unwrap_or(0);
            let cout = s.get("count_out").and_then(|n| n.as_u64()).unwrap_or(0);
            let rej = s.get("rejected").and_then(|n| n.as_u64()).unwrap_or(0);
            let style = if rej == bottleneck_rejected && rej > 0 {
                Style::default()
                    .fg(theme::SELL)
                    .add_modifier(Modifier::BOLD)
            } else if cout > 0 && cin > 0 && cout == cin {
                Style::default().fg(theme::BUY)
            } else if cin == 0 {
                theme::muted_style()
            } else {
                Style::default().fg(theme::TEXT_PRIMARY)
            };
            out.push(Line::styled(
                format!(
                    "  {:<28}  in={:>6}  out={:>6}  rejected={:>6}",
                    name, cin, cout, rej
                ),
                style,
            ));
        }
    }
    out.push(Line::raw(""));
    out.push(Line::styled(
        format!("  source: {}", path.display()),
        theme::caption_style(),
    ));
    out
}

fn collect_funnel_files() -> Vec<std::path::PathBuf> {
    let mut found: Vec<(std::time::SystemTime, std::path::PathBuf)> = Vec::new();
    for root in &["cache/discovery", "cache/discovery_test", "cache/auto_loop"] {
        let p = std::path::Path::new(root);
        if !p.exists() {
            continue;
        }
        if let Ok(read) = std::fs::read_dir(p) {
            for entry in read.flatten() {
                let path = entry.path();
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.ends_with("_funnel.json") {
                    let mtime = entry
                        .metadata()
                        .and_then(|m| m.modified())
                        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                    found.push((mtime, path));
                }
            }
        }
    }
    found.sort_by(|a, b| b.0.cmp(&a.0));
    found.into_iter().map(|(_, p)| p).collect()
}
