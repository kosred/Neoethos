//! Symbols — manifest-only canonical Vortex identity inventory.

use crossterm::event::KeyCode;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Widget, Wrap};

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

#[derive(Clone, Debug)]
struct InventoryNotice {
    path: std::path::PathBuf,
    category: &'static str,
    detail: String,
}

#[derive(Clone, Debug, Default)]
struct InventorySnapshot {
    rows: Vec<InventoryRow>,
    not_runnable: Vec<InventoryNotice>,
    scan_error: Option<String>,
}

pub fn draw(area: Rect, buf: &mut Buffer, shared: &mut AppShared) {
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
    // Key handling follows the viewport actually rendered, including resize.
    shared.symbols_viewport = content;
    draw_inventory(content, buf, shared);
    draw_import_bar(bar, buf, shared);
}

fn draw_inventory(inner: Rect, buf: &mut Buffer, shared: &AppShared) {
    let snapshot = collect_inventory(&shared.data_root);
    render_inventory(
        inner,
        buf,
        &snapshot,
        &shared.data_root,
        shared.symbols_scroll,
    );
}

fn render_inventory(
    inner: Rect,
    buf: &mut Buffer,
    snapshot: &InventorySnapshot,
    root: &std::path::Path,
    scroll: u16,
) {
    // The same bounded metadata is wrapped and scrollable. No identity,
    // generation, path, or rejection detail is shortened or suppressed.
    Paragraph::new(inventory_lines(snapshot, root))
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0))
        .render(inner, buf);
}

fn inventory_lines(snapshot: &InventorySnapshot, root: &std::path::Path) -> Vec<Line<'static>> {
    let mut lines = vec![
        Line::styled(
            format!(
                "{} IDs · {} not runnable · metadata only",
                snapshot.rows.len(),
                snapshot.not_runnable.len(),
            ),
            theme::caption_style(),
        ),
        Line::styled(
            "PgUp/PgDn scroll · Home top · Up/Down import fields",
            theme::muted_style(),
        ),
    ];
    if let Some(error) = &snapshot.scan_error {
        lines.push(Line::styled(
            format!("SCAN ERROR: {error}"),
            theme::warn_style(),
        ));
    }
    for notice in &snapshot.not_runnable {
        let name = notice
            .path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();
        lines.push(Line::styled(
            format!("SKIPPED {name} [{}]", notice.category),
            theme::warn_style(),
        ));
        lines.push(Line::styled(
            format!("Detail: {}", notice.detail),
            theme::warn_style(),
        ));
        lines.push(Line::styled(
            format!("Path: {}", notice.path.display()),
            theme::muted_style(),
        ));
    }
    if snapshot.rows.is_empty() {
        lines.push(Line::styled(
            "No canonical manifest entries found.",
            theme::warn_style(),
        ));
        lines.push(Line::styled(
            format!("Expected root: {}", root.display()),
            theme::muted_style(),
        ));
        lines.push(Line::styled(
            "Layout: d1-<exact-canonical-identity>/data.vortex.complete",
            theme::muted_style(),
        ));
    }
    for entry in &snapshot.rows {
        lines.push(Line::raw(""));
        lines.push(Line::styled(
            format!(
                "{} {} · {} · {}",
                entry.symbol,
                entry.timeframe,
                format_size(entry.size_bytes),
                entry.verification.as_str(),
            ),
            theme::accent_style(),
        ));
        lines.push(Line::styled(
            format!("Identity: {}", entry.dataset_identity),
            theme::primary_style(),
        ));
        lines.push(Line::styled(
            format!("Generation: {}", entry.generation),
            theme::primary_style(),
        ));
        lines.push(Line::styled(
            format!("Manifest binding: {}", entry.manifest_binding_sha256),
            theme::muted_style(),
        ));
    }
    lines
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
    // Keep one rendered row of overlap. A one-row viewport advances one row;
    // a not-yet-rendered or empty viewport cannot skip invisible content.
    let viewport = shared.symbols_viewport;
    let page_step = if viewport.width == 0 || viewport.height == 0 {
        0
    } else {
        viewport.height.saturating_sub(1).max(1)
    };
    match code {
        KeyCode::PageUp => {
            shared.symbols_scroll = shared.symbols_scroll.saturating_sub(page_step);
            true
        }
        KeyCode::PageDown => {
            shared.symbols_scroll = shared.symbols_scroll.saturating_add(page_step);
            true
        }
        KeyCode::Home => {
            shared.symbols_scroll = 0;
            true
        }
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
                snapshot.not_runnable.push(InventoryNotice {
                    path: skipped.path,
                    category: skipped.reason.category(),
                    detail: skipped.reason.detail().to_owned(),
                });
            }
            for entry in discovery.entries {
                let (Some(symbol), Some(timeframe)) = (entry.symbol, entry.timeframe) else {
                    snapshot.not_runnable.push(InventoryNotice {
                        path: entry.path,
                        category: "invalid_inventory_entry",
                        detail: format!(
                            "missing symbol/timeframe for identity={} generation={}",
                            entry.dataset_identity, entry.generation
                        ),
                    });
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
    snapshot
        .not_runnable
        .sort_by(|left, right| left.path.cmp(&right.path));
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

#[cfg(test)]
pub(in crate::tui) mod tests {
    use super::*;

    #[test]
    fn canonical_scan_retains_loose_sidecars_and_actual_invalid_identity_with_distinct_categories()
    {
        let root = std::env::temp_dir().join(format!(
            "neoethos-tui-inventory-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        for name in ["symbol_metadata.json", "spread_stats.json"] {
            std::fs::write(root.join(name), b"{}").unwrap();
        }
        std::fs::create_dir(root.join("d1-invalid")).unwrap();
        let snapshot = collect_inventory(&root);
        assert!(snapshot.rows.is_empty());
        assert_eq!(snapshot.not_runnable.len(), 3);
        for name in ["symbol_metadata.json", "spread_stats.json"] {
            let notice = snapshot
                .not_runnable
                .iter()
                .find(|notice| notice.path == root.join(name))
                .unwrap();
            assert_eq!(notice.category, "import_required");
            assert!(notice.detail.contains("explicit import"));
        }
        assert_eq!(
            snapshot
                .not_runnable
                .iter()
                .find(|notice| notice.path == root.join("d1-invalid"))
                .unwrap()
                .category,
            "invalid_canonical_identity",
        );
        for name in ["symbol_metadata.json", "spread_stats.json"] {
            std::fs::remove_file(root.join(name)).unwrap();
        }
        std::fs::remove_dir(root.join("d1-invalid")).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

    fn paging_fixture() -> (InventorySnapshot, &'static std::path::Path) {
        let root = std::path::Path::new("display-fixture");
        let mut snapshot = InventorySnapshot::default();
        snapshot.not_runnable.push(InventoryNotice {
            path: root
                .join("long-directory/".repeat(8))
                .join("broken-generation"),
            category: "unverified_generation",
            detail: format!(
                "{} terminal-detail-sentinel",
                "mismatched generation hash ".repeat(8)
            ),
        });
        for index in 0..9 {
            snapshot.rows.push(InventoryRow {
                symbol: "EURUSD".to_owned(),
                timeframe: format!("TF{index}"),
                dataset_identity: format!("identity-{index}-{}-end", "x".repeat(96)),
                generation: format!("generation-{index}-{}", "g".repeat(64)),
                manifest_binding_sha256: format!("binding-{index}-{}", "a".repeat(64)),
                verification: neoethos_data::DataVerificationStatus::ManifestOnly,
                size_bytes: 1024,
            });
        }
        (snapshot, root)
    }

    // The App key-path regression injects display metadata only, without
    // publishing datasets or mutating the process-global inventory cache.
    pub(in crate::tui) fn paging_fixture_rows(
        width: u16,
        viewport_height: Option<u16>,
        scroll: u16,
    ) -> Vec<String> {
        assert!(width > 0);
        let (snapshot, root) = paging_fixture();
        // One row per source character plus each newline is a conservative
        // finite bound for this metadata fixture, not a production cap.
        let height = viewport_height.unwrap_or_else(|| {
            let bound: usize = inventory_lines(&snapshot, root)
                .iter()
                .map(|line| {
                    1 + line
                        .spans
                        .iter()
                        .map(|span| span.content.chars().count())
                        .sum::<usize>()
                })
                .sum();
            u16::try_from(bound).unwrap()
        });
        let area = Rect::new(0, 0, width, height);
        let mut buffer = Buffer::empty(area);
        render_inventory(area, &mut buffer, &snapshot, root, scroll);
        let mut rows: Vec<String> = buffer
            .content
            .chunks(usize::from(width))
            .map(|row| row.iter().map(|cell| cell.symbol()).collect())
            .collect();
        if viewport_height.is_none() {
            while rows.last().is_some_and(|row| row.trim().is_empty()) {
                rows.pop();
            }
        }
        rows
    }

    #[test]
    fn inventory_fixture_preserves_full_diagnostics_and_all_nine_identities() {
        let (snapshot, root) = paging_fixture();
        let lines = inventory_lines(&snapshot, root);
        let source: String = lines
            .iter()
            .flat_map(|line| line.spans.iter().map(|span| span.content.as_ref()))
            .collect();
        for entry in &snapshot.rows {
            assert!(source.contains(&entry.dataset_identity));
            assert!(source.contains(&entry.generation));
            assert!(source.contains(&entry.manifest_binding_sha256));
        }
        for width in [40, 80] {
            let visible = paging_fixture_rows(width, None, 0).concat();
            assert!(visible.contains("broken-generation"));
            assert!(visible.contains("unverified_generation"));
            assert!(visible.contains("terminal-detail-sentinel"));
            for index in 0..9 {
                assert!(visible.contains(&format!("EURUSD TF{index}")));
                assert!(visible.contains(&format!("identity-{index}-")));
            }
        }
        let empty = Rect::new(0, 0, 0, 0);
        render_inventory(empty, &mut Buffer::empty(empty), &snapshot, root, 0);
    }
}
