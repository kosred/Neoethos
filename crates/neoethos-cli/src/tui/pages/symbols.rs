//! Symbols — manifest-only canonical Vortex identity inventory.

use crossterm::event::KeyCode;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table, Widget};

use crate::tui::app::AppShared;
use crate::tui::theme;

#[derive(Clone, Debug)]
struct InventoryRow {
    symbol: String,
    timeframe: String,
    dataset_identity: String,
    generation: String,
    manifest_binding_sha256: String,
    verification: neoethos_data::DataVerificationStatus,
    size_bytes: u64,
}

#[derive(Clone, Debug, Default)]
struct InventorySnapshot {
    rows: Vec<InventoryRow>,
    rejected: Vec<String>,
    scan_error: Option<String>,
}

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
    let import_h = 9u16.min(inner.height);
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
    let snapshot = collect_inventory(&shared.data_root);
    let mut notices = snapshot
        .scan_error
        .iter()
        .map(|error| Line::styled(format!("  SCAN ERROR: {error}"), theme::warn_style()))
        .collect::<Vec<_>>();
    notices.extend(
        snapshot
            .rejected
            .iter()
            .map(|rejected| Line::styled(format!("  REJECTED {rejected}"), theme::warn_style())),
    );
    let notice_height = u16::try_from(notices.len())
        .unwrap_or(u16::MAX)
        .min(inner.height);
    let areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(notice_height), Constraint::Min(0)])
        .split(inner);
    if !notices.is_empty() {
        Paragraph::new(notices).render(areas[0], buf);
    }
    let table_area = areas[1];

    if snapshot.rows.is_empty() {
        let empty = ratatui::widgets::Paragraph::new(vec![
            Line::raw(""),
            Line::styled(
                "  No canonical manifest entries found.",
                theme::warn_style().add_modifier(Modifier::BOLD),
            ),
            Line::raw(""),
            Line::styled(
                format!("  Expected layout at: {}", shared.data_root.display()),
                theme::muted_style(),
            ),
            Line::styled(
                "      d1-<exact-canonical-identity>/data.vortex.complete",
                theme::muted_style(),
            ),
        ]);
        empty.render(table_area, buf);
        return;
    }

    let header = Row::new(vec![
        Cell::from("SYMBOL / TF").style(theme::caption_style()),
        Cell::from("EXACT IDENTITY / GENERATION").style(theme::caption_style()),
        Cell::from("MANIFEST BINDING / VERIFY").style(theme::caption_style()),
        Cell::from("SIZE").style(theme::caption_style()),
    ])
    .height(1);

    let body_rows: Vec<Row> = snapshot
        .rows
        .into_iter()
        .map(|entry| {
            Row::new(vec![
                Cell::from(format!("{}\n{}", entry.symbol, entry.timeframe))
                    .style(theme::accent_style()),
                Cell::from(format!("{}\n{}", entry.dataset_identity, entry.generation))
                    .style(theme::primary_style()),
                Cell::from(format!(
                    "{}\n{:?}",
                    entry.manifest_binding_sha256, entry.verification
                ))
                .style(theme::muted_style()),
                Cell::from(format_size(entry.size_bytes)).style(theme::muted_style()),
            ])
            .height(2)
        })
        .collect();

    let widths = [
        Constraint::Length(12),
        Constraint::Percentage(43),
        Constraint::Percentage(42),
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
    Widget::render(table, table_area, buf);
}

fn draw_import_bar(area: Rect, buf: &mut Buffer, shared: &AppShared) {
    let running = shared.jobs.has_running("import");
    let hint = if running {
        "  importing… CPU + SourceSeal admission is held through verified Vortex publication"
            .to_string()
    } else {
        "  [↑/↓] select  [E/Enter] edit  [I] import one explicit source → verified Vortex"
            .to_string()
    };
    let mut lines = Vec::with_capacity(shared.import_form.fields.len() + 2);
    lines.push(Line::from(vec![Span::styled(
        "  IMPORT CONTRACT — no inference is used for publication",
        theme::caption_style().add_modifier(Modifier::BOLD),
    )]));
    for (index, field) in shared.import_form.fields.iter().enumerate() {
        let focused = index == shared.import_form.focused;
        let editing = focused && shared.import_form.editing;
        let value = if editing {
            format!("{}▌", field.value)
        } else if field.value.trim().is_empty() {
            "<required>".to_string()
        } else {
            field.value.clone()
        };
        let value_style = if editing {
            Style::default().fg(theme::APP_BG).bg(theme::ACCENT)
        } else if field.value.trim().is_empty() {
            theme::warn_style()
        } else {
            Style::default().fg(theme::TEXT_PRIMARY)
        };
        lines.push(Line::from(vec![
            Span::styled(if focused { "  ▸ " } else { "    " }, theme::accent_style()),
            Span::styled(format!("{}: ", field.label), theme::muted_style()),
            Span::styled(value, value_style),
        ]));
    }
    lines.push(Line::from(vec![Span::styled(hint, theme::caption_style())]));
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
        KeyCode::Up => {
            shared.import_form.focus_prev();
            true
        }
        KeyCode::Down | KeyCode::Tab => {
            shared.import_form.focus_next();
            true
        }
        KeyCode::Char('E') | KeyCode::Enter => {
            shared.import_form.start_editing();
            true
        }
        KeyCode::Char('I') => {
            // Import writes into the data/ layout — stage a Y/N confirmation
            // rather than launching immediately (FIX A). Guard against a blank
            // source up front so the prompt only appears for a real action.
            if let Some(label) = missing_import_field(shared) {
                shared.status = format!("Set {label} before import (↑/↓ then E)");
            } else if shared.jobs.has_running("import") {
                shared.status = "import already running".to_string();
            } else {
                shared.pending_confirmation = Some(crate::tui::app::PendingAction::SymbolsImport);
                shared.status = "Confirm data import? [Y]es / [N]o".to_string();
            }
            true
        }
        _ => false,
    }
}

pub fn launch_import(shared: &mut AppShared) {
    if let Some(label) = missing_import_field(shared) {
        shared.status = format!("Set {label} before import (↑/↓ then E)");
        return;
    }
    if shared.jobs.has_running("import") {
        shared.status = "import already running".to_string();
        return;
    }
    let root = shared.data_root.display().to_string();
    let value = |label: &str| {
        shared
            .import_form
            .value_for(label)
            .expect("the explicit import form contains every required field")
            .trim()
            .to_string()
    };
    let source_format = value("Source format");
    if source_format
        .parse::<neoethos_data::core::import_provenance::ImportSourceFormat>()
        .is_err()
    {
        shared.status = "Source format is not one of the exact supported labels".to_string();
        return;
    }
    let bar_timestamps = value("Bar timestamps");
    if bar_timestamps != "bar_open" {
        shared.status =
            "Bar timestamps must be explicitly evidenced as exactly `bar_open`".to_string();
        return;
    }
    shared.jobs.spawn(
        "import",
        vec![
            "import".to_string(),
            "--source".to_string(),
            value("Import source"),
            "--format".to_string(),
            source_format,
            "--source-namespace".to_string(),
            value("Source namespace"),
            "--symbol".to_string(),
            value("Symbol"),
            "--timeframe".to_string(),
            value("Timeframe"),
            "--bar-timestamps".to_string(),
            bar_timestamps,
            "--root".to_string(),
            root,
        ],
    );
    shared.status =
        "Spawned explicit import — acknowledgement requires verified canonical Vortex reopen"
            .to_string();
}

fn missing_import_field(shared: &AppShared) -> Option<&'static str> {
    [
        "Import source",
        "Source format",
        "Source namespace",
        "Symbol",
        "Timeframe",
        "Bar timestamps",
    ]
    .into_iter()
    .find(|label| {
        shared
            .import_form
            .value_for(label)
            .is_none_or(|value| value.trim().is_empty())
    })
}

fn collect_inventory(root: &std::path::Path) -> InventorySnapshot {
    // draw() runs per frame (~30 fps); even a bounded manifest inventory must
    // not become per-frame filesystem churn. Memoize for 2 s — inventory
    // changes on import timescales, not frame timescales.
    use std::sync::Mutex;
    use std::time::{Duration, Instant};
    type InventoryCache = Option<(Instant, std::path::PathBuf, InventorySnapshot)>;
    static CACHE: Mutex<InventoryCache> = Mutex::new(None);
    {
        let guard = CACHE.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((at, cached_root, rows)) = guard.as_ref() {
            if cached_root == root && at.elapsed() < Duration::from_secs(2) {
                return rows.clone();
            }
        }
    }
    let mut snapshot = InventorySnapshot::default();
    match neoethos_data::DatasetDiscovery::scan_metadata(root) {
        Ok(discovery) => {
            for skipped in discovery.skipped {
                snapshot.rejected.push(format!(
                    "path={} category={} detail={:?}",
                    skipped.path.display(),
                    skipped.reason.category(),
                    skipped.reason
                ));
            }
            for entry in discovery.entries {
                let (Some(symbol), Some(timeframe)) = (entry.symbol, entry.timeframe) else {
                    snapshot.rejected.push(format!(
                        "path={} category=invalid_inventory_entry detail=missing symbol/timeframe for identity={} generation={}",
                        entry.path.display(), entry.dataset_identity, entry.generation
                    ));
                    continue;
                };
                snapshot.rows.push(InventoryRow {
                    symbol,
                    timeframe,
                    dataset_identity: entry.dataset_identity,
                    generation: entry.generation,
                    manifest_binding_sha256: entry.manifest_binding_sha256,
                    verification: entry.verification,
                    size_bytes: entry.size_bytes,
                });
            }
        }
        Err(error) => snapshot.scan_error = Some(format!("{}: {error:#}", root.display())),
    }
    snapshot.rows.sort_by(|left, right| {
        left.symbol
            .cmp(&right.symbol)
            .then_with(|| timeframe_sort_key(&left.timeframe, &right.timeframe))
            .then_with(|| left.dataset_identity.cmp(&right.dataset_identity))
    });
    snapshot.rejected.sort();
    *CACHE.lock().unwrap_or_else(|p| p.into_inner()) =
        Some((Instant::now(), root.to_path_buf(), snapshot.clone()));
    snapshot
}

fn timeframe_sort_key(a: &String, b: &String) -> std::cmp::Ordering {
    let protocol_code = |timeframe: &str| {
        timeframe
            .parse::<neoethos_data::CanonicalTimeframe>()
            .map(|timeframe| timeframe.ctrader_protocol_code())
            .unwrap_or(i32::MAX)
    };
    protocol_code(a).cmp(&protocol_code(b))
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
