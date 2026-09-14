//! Dashboard — landing page. Shows dataset summary, active jobs,
//! recent runs, and KPI cards.

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Padding, Paragraph, Widget};

use crate::tui::app::AppShared;
use crate::tui::theme;
use crate::tui::widgets::kpi::Kpi;

#[derive(Clone, Debug, Default)]
struct DatasetSummary {
    symbol_count: usize,
    max_timeframes: usize,
    exact_entries: Vec<String>,
    rejections: Vec<String>,
    scan_error: Option<String>,
}

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

    let dataset = dataset_summary(shared);
    Kpi::new("Symbols", dataset.symbol_count.to_string())
        .sub(format!(
            "{} max TF · {} exact IDs",
            dataset.max_timeframes,
            dataset.exact_entries.len()
        ))
        .value_style(theme::accent_style())
        .render(kpi_cols[0], buf);

    let active = shared.jobs.running_count();
    Kpi::new("Active jobs", active.to_string())
        .sub(if active == 0 { "Idle" } else { "Running" })
        .value_style(if active == 0 {
            theme::muted_style().add_modifier(Modifier::BOLD)
        } else {
            theme::buy_style()
        })
        .render(kpi_cols[1], buf);

    let (files, detail) = match super::strategies::portfolio_file_counts(&shared.cache_root) {
        Ok((files, unavailable)) => (
            files.to_string(),
            format!("{unavailable} unavailable · not trading approval"),
        ),
        Err(_) => (
            "?".to_owned(),
            "Cache unreadable · see Strategies".to_owned(),
        ),
    };
    Kpi::new("Portfolio files", files)
        .sub(detail)
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

    render_recent_activity(bottom_cols[0], buf, &dataset);
    render_quick_start(bottom_cols[1], buf);
}

fn render_recent_activity(area: Rect, buf: &mut Buffer, dataset: &DatasetSummary) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER))
        .title(Span::styled(
            " DATASET INVENTORY / RECENT ACTIVITY ",
            theme::caption_style().add_modifier(Modifier::BOLD),
        ))
        .style(theme::panel_block_style())
        .padding(Padding::new(1, 1, 0, 0));
    let inner = block.inner(area);
    block.render(area, buf);

    let lines = activity_panel_lines(dataset, recent_activity_lines(), inner.height as usize);
    // One compact row per entry: wrapped hashes must never push all activity
    // off the fixed-height Dashboard. Symbols retains the full inventory.
    Paragraph::new(lines).render(inner, buf);
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
        Line::raw(""),
        Line::styled(
            "5.  Promote ([P] in Strategies)",
            Style::default()
                .fg(theme::TEXT_PRIMARY)
                .add_modifier(Modifier::BOLD),
        ),
        Line::styled("    merge into the live portfolio", theme::muted_style()),
        Line::raw(""),
        // This TUI is the discovery / training / config console. Live trading,
        // positions and orders live in the NeoEthos desktop app (which owns the
        // broker connection) — by design, not a gap.
        Line::styled(
            "Live trading & positions",
            theme::caption_style().add_modifier(Modifier::BOLD),
        ),
        Line::styled("    → NeoEthos desktop app", theme::muted_style()),
    ];
    Paragraph::new(lines).render(inner, buf);
}

fn dataset_summary(shared: &AppShared) -> DatasetSummary {
    use std::sync::Mutex;
    use std::time::{Duration, Instant};
    type InventoryCache = Option<(Instant, std::path::PathBuf, DatasetSummary)>;
    static CACHE: Mutex<InventoryCache> = Mutex::new(None);
    {
        let guard = CACHE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((updated_at, cached_root, cached)) = guard.as_ref()
            && cached_root == &shared.data_root
            && updated_at.elapsed() < Duration::from_secs(2)
        {
            return cached.clone();
        }
    }
    let summary = build_dataset_summary(&shared.data_root);
    *CACHE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) =
        Some((Instant::now(), shared.data_root.clone(), summary.clone()));
    summary
}

fn build_dataset_summary(data_root: &std::path::Path) -> DatasetSummary {
    let mut summary = DatasetSummary::default();
    let report = match neoethos_data::DatasetDiscovery::scan_metadata(data_root) {
        Ok(report) => report,
        Err(error) => {
            summary.scan_error = Some(format!("root={} detail={error:#}", data_root.display()));
            return summary;
        }
    };
    let mut timeframes_by_symbol =
        std::collections::BTreeMap::<String, std::collections::BTreeSet<String>>::new();
    for entry in report.entries {
        let (Some(symbol), Some(timeframe)) = (entry.symbol, entry.timeframe) else {
            summary.rejections.push(format!(
                "path={} category=invalid_inventory_entry detail=missing symbol/timeframe for identity={} generation={} manifest_binding_sha256={} verification={:?}",
                entry.path.display(),
                entry.dataset_identity,
                entry.generation,
                entry.manifest_binding_sha256,
                entry.verification
            ));
            continue;
        };
        timeframes_by_symbol
            .entry(symbol.clone())
            .or_default()
            .insert(timeframe.clone());
        summary.exact_entries.push(format!(
            "{symbol} {timeframe} identity={} generation={} manifest_binding_sha256={} verification={:?}",
            entry.dataset_identity,
            entry.generation,
            entry.manifest_binding_sha256,
            entry.verification
        ));
    }
    summary.symbol_count = timeframes_by_symbol.len();
    summary.max_timeframes = timeframes_by_symbol
        .values()
        .map(std::collections::BTreeSet::len)
        .max()
        .unwrap_or(0);
    summary
        .rejections
        .extend(report.skipped.into_iter().map(|skipped| {
            format!(
                "path={} category={} detail={:?}",
                skipped.path.display(),
                skipped.reason.category(),
                skipped.reason
            )
        }));
    summary
}

fn dataset_inventory_lines(summary: &DatasetSummary) -> Vec<Line<'static>> {
    let mut lines = vec![Line::styled(
        format!(
            "{} IDs · {} symbols · {} not runnable · metadata only",
            summary.exact_entries.len(),
            summary.symbol_count,
            summary.rejections.len()
        ),
        theme::caption_style(),
    )];
    if let Some(error) = &summary.scan_error {
        lines.push(Line::styled(
            format!("Inventory error: {error}"),
            theme::sell_style(),
        ));
    } else {
        lines.push(Line::styled(
            "Full identity/generation/skipped-path details: Symbols",
            theme::muted_style(),
        ));
    }
    lines
}

fn activity_panel_lines(
    summary: &DatasetSummary,
    mut activity: Vec<Line<'static>>,
    height: usize,
) -> Vec<Line<'static>> {
    if height == 0 {
        return Vec::new();
    }
    if activity.is_empty() {
        activity.push(Line::styled(
            "No complete subsystem events in today's UTC log (INFO may be filtered).",
            theme::muted_style(),
        ));
    }
    let mut lines: Vec<_> = dataset_inventory_lines(summary)
        .into_iter()
        .take(height.saturating_sub(1).min(2))
        .collect();
    lines.extend(
        activity
            .into_iter()
            .take(height.saturating_sub(lines.len())),
    );
    lines
}

#[derive(Debug)]
struct Activity {
    status: String,
    operation: String,
    message: String,
    finished_field: bool,
}

/// Consume the current SectionedRunRecord text emitted through tracing.
/// Only closed blocks are shown; a tail beginning/ending midway through a
/// record or ordinary interleaved tracing messages cannot borrow its fields.
fn parse_activity(content: &str) -> Vec<Activity> {
    let mut current: Option<Activity> = None;
    let mut recent = std::collections::VecDeque::with_capacity(8);
    for line in content.lines() {
        if let Some(header) = line.strip_prefix("> [") {
            current = header.split_once("] ").and_then(|(_, rest)| {
                let (label, run_id) = rest.rsplit_once("  |  run_id=")?;
                if run_id.trim().is_empty() {
                    return None;
                }
                let label = label.split("  |  ").next()?;
                let (status, operation) = label.split_once(' ')?;
                if status.is_empty() || operation.is_empty() {
                    return None;
                }
                Some(Activity {
                    status: status.to_owned(),
                    operation: operation.to_owned(),
                    message: String::new(),
                    finished_field: false,
                })
            });
        } else if line == "=".repeat(78) {
            if let Some(record) = current.take().filter(|record| record.finished_field) {
                if recent.len() == 8 {
                    recent.pop_front();
                }
                recent.push_back(record);
            }
        } else if let Some(record) = current.as_mut() {
            if line.starts_with("  finished: ") {
                record.finished_field = true;
            } else if let Some(message) = line.strip_prefix("  message: ") {
                record.message = message.to_owned();
            }
        }
    }
    recent.into_iter().rev().collect()
}

fn activity_lines(content: &str) -> Vec<Line<'static>> {
    parse_activity(content)
        .into_iter()
        .map(|record| {
            let status_style = match record.status.as_str() {
                "SUCCESS" => theme::buy_style(),
                "FAILED" => theme::sell_style(),
                "STARTED" => theme::accent_style(),
                _ => theme::muted_style(),
            };
            Line::from(vec![
                Span::styled(format!("{} ", record.status), status_style),
                Span::styled(format!("{} · ", record.operation), theme::primary_style()),
                Span::styled(record.message, theme::muted_style()),
            ])
        })
        .collect()
}

fn recent_activity_lines() -> Vec<Line<'static>> {
    let log_path = neoethos_core::logging::canonical_log_path();
    super::logs::read_log_tail(&log_path)
        .map(|content| activity_lines(&content))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn produced_log() -> (std::path::PathBuf, String) {
        use neoethos_core::sectioned_log::{SectionedRunRecord, SubsystemSection};
        let root = std::env::temp_dir().join(format!(
            "neoethos-tui-activity-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let path = root.join("activity.log");
        let writer = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .without_time()
            .with_writer(move || writer.try_clone().unwrap())
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            for index in 0..12 {
                let status = ["STARTED", "SUCCESS", "FAILED"][index % 3];
                neoethos_core::logging::write_subsystem_record(
                    SubsystemSection::Cli,
                    SectionedRunRecord {
                        run_id: format!("fixture-{index}"),
                        parent_run_id: None,
                        started_at: "2026-09-07T23:30:00Z".to_owned(),
                        finished_at: "2026-09-07T23:30:01Z".to_owned(),
                        subsystem: SubsystemSection::Cli,
                        operation: format!("operation-{index}"),
                        status: status.to_owned(),
                        symbol: None,
                        timeframe: None,
                        error_code: None,
                        message: format!("μήνυμα-{index}"),
                        body:
                            "multiline body\nmessage: not a record field\n> [CLI] false body header"
                                .to_owned(),
                    },
                )
                .unwrap();
                tracing::info!("ordinary interleaved event without a subsystem record");
            }
        });
        let content = std::fs::read_to_string(path).unwrap();
        (root, content)
    }

    #[test]
    fn actual_subsystem_producer_reaches_activity_newest_first_without_cross_record_fields() {
        let (root, content) = produced_log();
        let records = parse_activity(&content);
        assert_eq!(records.len(), 8);
        for (record, index) in records.iter().zip((4..12).rev()) {
            assert_eq!(record.operation, format!("operation-{index}"));
            assert_eq!(record.message, format!("μήνυμα-{index}"));
            assert_eq!(record.status, ["STARTED", "SUCCESS", "FAILED"][index % 3]);
        }
        let first_header = content.find("> [CLI]").unwrap();
        let next_header = content[first_header + 1..].find("\n> [CLI]").unwrap() + first_header + 2;
        let fragment = &content[first_header + 5..next_header];
        assert!(
            parse_activity(fragment).is_empty(),
            "incomplete header/tail must not invent an event"
        );
        let mut partial = content.clone();
        partial.push_str("\n> [CLI] FAILED incomplete  |  run_id=partial\n  finished: now\n  message: unfinished");
        assert_eq!(parse_activity(&partial)[0].operation, "operation-11");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn nine_dataset_identities_do_not_hide_newest_activity_in_compact_dashboard() {
        let (root, content) = produced_log();
        let summary = DatasetSummary {
            symbol_count: 3,
            max_timeframes: 3,
            exact_entries: (0..9)
                .map(|i| {
                    format!(
                        "identity={i} generation={} hash={}",
                        "g".repeat(128),
                        "h".repeat(64)
                    )
                })
                .collect(),
            rejections: vec!["rejected generation".to_owned()],
            scan_error: None,
        };
        for (width, height) in [(42, 12), (66, 18), (12, 3), (0, 0)] {
            let area = Rect::new(0, 0, width, height);
            let mut buffer = Buffer::empty(area);
            let rows = activity_panel_lines(&summary, activity_lines(&content), height as usize);
            Paragraph::new(rows).render(area, &mut buffer);
            let text: String = buffer.content.iter().map(|cell| cell.symbol()).collect();
            if width >= 42 {
                assert!(
                    text.contains("FAILED operation-11"),
                    "{width}x{height}: {text}"
                );
                assert!(text.contains("9 IDs"));
                assert!(text.contains("1 not runnable"));
            }
        }
        assert!(
            activity_panel_lines(&summary, Vec::new(), 1)[0].spans[0]
                .content
                .contains("No complete")
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
