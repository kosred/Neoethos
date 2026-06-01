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
                "    neoethos-cli batch-discover --root <data> --out-dir cache/discovery",
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

/// Array fields, in preference order, that hold the strategy objects in
/// the various portfolio shapes we write:
///   - `portfolio`  — the curated set (modern `discovery.rs` output)
///   - `best_genes` — talib-knowledge files
///   - `genes`      — GA checkpoints / `strategy_gene` dumps
///   - `candidates` / `survivors` / `strategies` — other writers
/// A bare `[...]` array (no wrapper object) is also handled.
const STRATEGY_ARRAY_KEYS: &[&str] = &[
    "portfolio",
    "best_genes",
    "genes",
    "strategies",
    "candidates",
    "survivors",
];

/// (mtime_secs, len) → count cache so the 71 MB knowledge file isn't
/// re-read and re-scanned on every redraw. Keyed per path; a changed
/// mtime or size invalidates the entry.
fn count_cache()
-> &'static std::sync::Mutex<std::collections::HashMap<PathBuf, (u64, u64, usize)>> {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<PathBuf, (u64, u64, usize)>>,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn count_strategies(path: &std::path::Path) -> usize {
    let meta = std::fs::metadata(path).ok();
    let len = meta.as_ref().map(|m| m.len()).unwrap_or(0);
    let mtime = meta
        .as_ref()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);

    if let Ok(cache) = count_cache().lock() {
        if let Some(&(c_mtime, c_len, c_count)) = cache.get(path) {
            if c_mtime == mtime && c_len == len {
                return c_count;
            }
        }
    }

    let count = compute_strategy_count(path);
    if let Ok(mut cache) = count_cache().lock() {
        cache.insert(path.to_path_buf(), (mtime, len, count));
    }
    count
}

fn compute_strategy_count(path: &std::path::Path) -> usize {
    let Ok(text) = std::fs::read_to_string(path) else {
        return 0;
    };
    let trimmed = text.trim_start();
    // Two shapes in the wild: a bare `[ {...}, … ]` array, or an object
    // that wraps the strategies in one of `STRATEGY_ARRAY_KEYS`. Locate
    // the relevant array's opening `[`, then count the objects directly
    // inside it.
    let array_start = if trimmed.starts_with('[') {
        Some(0)
    } else {
        STRATEGY_ARRAY_KEYS.iter().find_map(|key| {
            let needle = format!("\"{key}\"");
            let kpos = trimmed.find(&needle)?;
            trimmed[kpos..].find('[').map(|rel| kpos + rel)
        })
    };
    let Some(start) = array_start else {
        return 0;
    };
    count_objects_in_array(&trimmed[start..])
}

/// `s` begins at the `[` of a strategy array. Count the objects whose
/// opening `{` sits at the array's immediate element depth. String-aware
/// so braces inside quoted values (indicator names, notes) don't inflate
/// the count; stops at the array's matching `]`.
fn count_objects_in_array(s: &str) -> usize {
    let mut in_str = false;
    let mut escaped = false;
    let mut bracket_depth: i32 = 0; // [] nesting
    let mut brace_depth: i32 = 0; // {} nesting
    let mut count = 0usize;
    for ch in s.chars() {
        if in_str {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_str = false;
            }
            continue;
        }
        match ch {
            '"' => in_str = true,
            '[' => bracket_depth += 1,
            ']' => {
                bracket_depth -= 1;
                if bracket_depth == 0 {
                    break;
                }
            }
            '{' => {
                if bracket_depth == 1 && brace_depth == 0 {
                    count += 1;
                }
                brace_depth += 1;
            }
            '}' => brace_depth -= 1,
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
    // need a chrono dep we have not added to neoethos-cli.
    format!("{h:02}:{m:02}:{s:02} UTC")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_wrapped_best_genes_object() {
        // The talib-knowledge shape that used to report 0 — an object
        // whose strategies live in a `best_genes` array.
        let json = r#"{
            "generated_at": "2026-02-16T19:59:16Z",
            "symbol": "EURUSD",
            "best_genes": [
                {"indicators": ["CDLSHORTLINE"], "score": 1.2},
                {"indicators": ["CDLKICKING"], "score": 0.9},
                {"indicators": ["CDLTRISTAR"], "score": 0.7}
            ]
        }"#;
        let start = json.find('[').unwrap();
        assert_eq!(count_objects_in_array(&json[start..]), 3);
    }

    #[test]
    fn counts_bare_array() {
        let json = r#"[ {"a":1}, {"b":2} ]"#;
        assert_eq!(count_objects_in_array(json), 2);
    }

    #[test]
    fn prefers_portfolio_over_candidates() {
        // Modern discovery output carries both; the STRATEGIES column
        // should reflect the curated `portfolio`, not the wider pool.
        let json = r#"{
            "portfolio": [ {"id":1}, {"id":2} ],
            "candidates": [ {"id":3}, {"id":4}, {"id":5}, {"id":6} ]
        }"#;
        let trimmed = json.trim_start();
        let key = STRATEGY_ARRAY_KEYS
            .iter()
            .find(|k| trimmed.contains(&format!("\"{k}\"")))
            .unwrap();
        assert_eq!(*key, "portfolio");
        let kpos = trimmed.find("\"portfolio\"").unwrap();
        let start = kpos + trimmed[kpos..].find('[').unwrap();
        assert_eq!(count_objects_in_array(&trimmed[start..]), 2);
    }

    #[test]
    fn ignores_braces_inside_strings() {
        // A brace inside a quoted value must not inflate the count.
        let json = r#"[ {"note":"a { brace } here"}, {"note":"plain"} ]"#;
        assert_eq!(count_objects_in_array(json), 2);
    }

    #[test]
    fn empty_array_is_zero() {
        assert_eq!(count_objects_in_array("[]"), 0);
    }
}
