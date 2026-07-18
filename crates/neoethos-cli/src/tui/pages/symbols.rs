//! Symbols — dataset inventory. Reads `<root>/symbol=*/timeframe=*/`
//! and shows a grid of available bars per symbol × timeframe.

use crossterm::event::KeyCode;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table, Widget};

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

    // Reserve a bottom import bar so data can be brought in without leaving the
    // TUI — shown in both the populated and empty-dataset states.
    let import_h = 4u16.min(inner.height);
    let content = Rect {
        height: inner.height.saturating_sub(import_h),
        ..inner
    };
    let bar = Rect {
        y: inner.y + content.height,
        height: import_h,
        ..inner
    };
    draw_inventory(content, buf, shared);
    draw_import_bar(bar, buf, shared);
}

fn draw_inventory(inner: Rect, buf: &mut Buffer, shared: &AppShared) {
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

fn draw_import_bar(area: Rect, buf: &mut Buffer, shared: &AppShared) {
    let field = &shared.import_form.fields[0];
    let editing = shared.import_form.editing;
    let value = if editing {
        format!("{}▌", field.value)
    } else if field.value.trim().is_empty() {
        "<press E to set a source path>".to_string()
    } else {
        field.value.clone()
    };
    let vstyle = if editing {
        Style::default().fg(theme::APP_BG).bg(theme::ACCENT)
    } else if field.value.trim().is_empty() {
        theme::muted_style()
    } else {
        Style::default().fg(theme::TEXT_PRIMARY)
    };
    let running = shared.jobs.has_running("import");
    let hint = if running {
        "  importing… see status bar / Logs page".to_string()
    } else {
        "  [E] edit source   [I] import → data/   (CSV/TSV/JSON/Parquet/Vortex auto-detected)"
            .to_string()
    };
    let lines = vec![
        Line::from(vec![
            Span::styled(
                "  IMPORT  ",
                theme::caption_style().add_modifier(Modifier::BOLD),
            ),
            Span::styled("source: ", theme::muted_style()),
            Span::styled(value, vstyle),
        ]),
        Line::from(vec![Span::styled(hint, theme::caption_style())]),
    ];
    Paragraph::new(lines).render(area, buf);
}

pub fn handle_key(code: KeyCode, shared: &mut AppShared) -> bool {
    if shared.import_form.editing {
        let f = &mut shared.import_form;
        match code {
            KeyCode::Enter => f.stop_editing(true),
            KeyCode::Esc => f.stop_editing(false),
            KeyCode::Backspace => f.backspace(),
            KeyCode::Char(c) => f.type_char(c),
            _ => return false,
        }
        return true;
    }
    match code {
        KeyCode::Char('E') => {
            shared.import_form.focus(0);
            shared.import_form.start_editing();
            true
        }
        KeyCode::Char('I') => {
            // Import writes into the data/ layout — stage a Y/N confirmation
            // rather than launching immediately (FIX A). Guard against a blank
            // source up front so the prompt only appears for a real action.
            let src = shared
                .import_form
                .value_for("Import source")
                .unwrap_or("")
                .trim()
                .to_string();
            if src.is_empty() {
                shared.status = "Set an import source first (press E)".to_string();
            } else if shared.jobs.has_running("import") {
                shared.status = "import already running".to_string();
            } else {
                shared.pending_confirmation =
                    Some(crate::tui::app::PendingAction::SymbolsImport);
                shared.status = "Confirm data import? [Y]es / [N]o".to_string();
            }
            true
        }
        _ => false,
    }
}

pub fn launch_import(shared: &mut AppShared) {
    let src = shared
        .import_form
        .value_for("Import source")
        .unwrap_or("")
        .trim()
        .to_string();
    if src.is_empty() {
        shared.status = "Set an import source first (press E)".to_string();
        return;
    }
    if shared.jobs.has_running("import") {
        shared.status = "import already running".to_string();
        return;
    }
    let root = shared.data_root.display().to_string();
    shared.jobs.spawn(
        "import",
        vec![
            "import".to_string(),
            "--source".to_string(),
            src,
            "--root".to_string(),
            root,
        ],
    );
    shared.status = "Spawned data import — converting to data/ layout".to_string();
}

fn collect_inventory(root: &std::path::Path) -> Vec<(String, String, usize, u64)> {
    // draw() runs per frame (~30 fps); a full read_dir + per-file metadata
    // walk of the data tree every frame is tens of thousands of syscalls
    // per second on a large dataset. Memoize for 2 s — inventory changes
    // on import timescales, not frame timescales.
    use std::sync::Mutex;
    use std::time::{Duration, Instant};
    type InventoryCache = Option<(Instant, std::path::PathBuf, Vec<(String, String, usize, u64)>)>;
    static CACHE: Mutex<InventoryCache> = Mutex::new(None);
    {
        let guard = CACHE.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((at, cached_root, rows)) = guard.as_ref() {
            if cached_root == root && at.elapsed() < Duration::from_secs(2) {
                return rows.clone();
            }
        }
    }
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
    *CACHE.lock().unwrap_or_else(|p| p.into_inner()) =
        Some((Instant::now(), root.to_path_buf(), out.clone()));
    out
}

fn timeframe_sort_key(a: &String, b: &String) -> std::cmp::Ordering {
    // Sort canonical chart timeframes by minute value so the
    // operator reads them as M1, M5, M15, H1, H4, D1, W1, MN1.
    fn to_minutes(tf: &str) -> u64 {
        // MN1 must be checked BEFORE the single-letter split: "MN1" split
        // at 1 is ("M", "N1"), whose parse-failure used to collapse the
        // monthly timeframe to 0 minutes and sort it before M1.
        if tf.starts_with("MN") {
            return 100_000;
        }
        let (kind, num) = tf.split_at(1);
        let n: u64 = num.parse().unwrap_or(0);
        match kind {
            "M" => n,
            "H" => n * 60,
            "D" => n * 1440,
            "W" => n * 10_080,
            _ => 100_000,
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
