//! Strategies — strict live-portfolio browser. Reads only the exact
//! `*.live_portfolio.json` artifacts emitted by discovery and shows them in a
//! sortable table.

use std::path::PathBuf;

use crossterm::event::KeyCode;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Cell, Padding, Paragraph, Row, StatefulWidget, Table, TableState, Widget,
};

use crate::tui::app::AppShared;
use crate::tui::theme;

/// Move the Strategies selection / validate the selected portfolio. Returns
/// whether the key was consumed.
pub fn handle_key(code: KeyCode, shared: &mut AppShared) -> bool {
    let portfolios = match scan_portfolios(&shared.cache_root) {
        Ok(portfolios) => portfolios,
        Err(error) => {
            shared.status = error;
            return true;
        }
    };
    if portfolios.is_empty() {
        return false;
    }
    let count = portfolios.len();
    match code {
        KeyCode::Up => {
            shared.strategies_selected = shared.strategies_selected.saturating_sub(1);
            true
        }
        KeyCode::Down => {
            shared.strategies_selected = (shared.strategies_selected + 1).min(count - 1);
            true
        }
        KeyCode::Char('V') => {
            let sel = shared.strategies_selected.min(count - 1);
            launch_validate(shared, &portfolios[sel].path);
            true
        }
        KeyCode::Char('P') => {
            let sel = shared.strategies_selected.min(count - 1);
            launch_promote(shared, &portfolios[sel].path);
            true
        }
        _ => false,
    }
}

/// Validate the selected portfolio on real data via `trader-replay`, which
/// replays the discovery's own genes from the selected strict v3 artifact so
/// the user can confirm a portfolio out-of-sample without leaving the TUI.
fn launch_validate(shared: &mut AppShared, portfolio_path: &std::path::Path) {
    if !portfolio_path.exists() {
        shared.status = "Selected strict v3 live portfolio no longer exists".to_string();
        return;
    }
    if shared.jobs.has_running("validate") {
        shared.status = "validation already running".to_string();
        return;
    }
    shared.jobs.spawn(
        "validate",
        vec![
            "trader-replay".to_string(),
            "--portfolio".to_string(),
            portfolio_path.display().to_string(),
        ],
    );
    shared.status = "Spawned trader-replay validation — see Logs / status".to_string();
}

/// Promote the selected portfolio to the live set via `discovery-promote-weekly`
/// (B3/#4 parity). The selected strict v3 artifact carries the exact search
/// receipt and config identity; no neighboring path or filename is authority.
fn launch_promote(shared: &mut AppShared, portfolio_path: &std::path::Path) {
    if shared.jobs.has_running("promote") {
        shared.status = "promotion already running".to_string();
        return;
    }
    if !portfolio_path.exists() {
        shared.status = "Selected strict v3 live portfolio no longer exists".to_string();
        return;
    }

    shared.jobs.spawn(
        "promote",
        vec![
            "discovery-promote-weekly".to_string(),
            "--portfolio".to_string(),
            portfolio_path.display().to_string(),
        ],
    );
    shared.status =
        "Spawned exact discovery-promote-weekly — see live log for the verdict".to_string();
}

pub fn draw(area: Rect, buf: &mut Buffer, shared: &AppShared) {
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

    let portfolios = match scan_portfolios(&shared.cache_root) {
        Ok(portfolios) => portfolios,
        Err(error) => {
            Paragraph::new(Line::styled(
                format!("Portfolio inventory unavailable: {error}"),
                theme::sell_style(),
            ))
            .render(inner, buf);
            return;
        }
    };
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
                format!(
                    "  Configured discovery cache: {}",
                    shared.cache_root.join("discovery").display()
                ),
                theme::accent_style(),
            ),
            Line::raw(""),
            Line::styled(
                "  Selectable artifacts end in  .live_portfolio.json",
                theme::caption_style(),
            ),
        ];
        Paragraph::new(lines).render(inner, buf);
        return;
    }

    let sel = shared
        .strategies_selected
        .min(portfolios.len().saturating_sub(1));

    // Split: portfolio table on top, the selected portfolio's per-strategy
    // metrics below — so the user can actually SEE what a discovery found.
    let detail_h = (inner.height / 2)
        .clamp(0, 14)
        .min(inner.height.saturating_sub(4));
    let table_area = Rect {
        height: inner.height.saturating_sub(detail_h),
        ..inner
    };
    let detail_area = Rect {
        y: inner.y + table_area.height,
        height: detail_h,
        ..inner
    };

    let header = Row::new(vec![
        Cell::from("PORTFOLIO").style(theme::caption_style()),
        Cell::from("STRATEGIES").style(theme::caption_style()),
        Cell::from("SIZE").style(theme::caption_style()),
        Cell::from("MODIFIED").style(theme::caption_style()),
    ])
    .height(1);

    let rows: Vec<Row> = portfolios
        .iter()
        .map(|p| {
            Row::new(vec![
                Cell::from(p.name.clone()).style(theme::accent_style()),
                Cell::from(
                    p.strategies
                        .as_ref()
                        .map(|count| count.to_string())
                        .unwrap_or_else(|_| "?".to_owned()),
                )
                .style(theme::primary_style()),
                Cell::from(format_size(p.bytes)).style(theme::muted_style()),
                Cell::from(p.modified.clone()).style(theme::muted_style()),
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
        )
        .highlight_symbol("▸ ");
    let mut state = TableState::default();
    state.select(Some(sel));
    StatefulWidget::render(table, table_area, buf, &mut state);

    draw_details(detail_area, buf, &portfolios[sel]);
}

#[derive(Clone, Debug, serde::Deserialize)]
struct StratMetrics {
    strategy_id: String,
    initial_capital: f64,
    net_profit: f64,
    #[serde(rename = "total_return_pct")]
    return_pct: f64,
    #[serde(rename = "sharpe_ratio")]
    sharpe: f64,
    #[serde(rename = "max_drawdown_pct")]
    max_dd: f64,
    #[serde(rename = "win_rate")]
    win: f64,
}

type FileStamp = Option<(std::time::SystemTime, u64)>;

/// Cache the ID-bound display projection, not the large receipt/equity arrays.
/// Both files participate: changing the selected portfolio must invalidate an
/// unchanged sidecar, and neither file is decoded on every redraw.
#[derive(Clone)]
struct QualityEntry {
    portfolio_stamp: FileStamp,
    quality_source: PathBuf,
    quality_stamp: FileStamp,
    metrics: Result<Vec<StratMetrics>, String>,
}

fn quality_cache() -> &'static std::sync::Mutex<std::collections::HashMap<PathBuf, QualityEntry>> {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<PathBuf, QualityEntry>>,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn file_stamp(path: &std::path::Path) -> FileStamp {
    let meta = std::fs::metadata(path).ok()?;
    Some((meta.modified().ok()?, meta.len()))
}

#[derive(serde::Deserialize)]
struct ResearchQualityProjection {
    schema_version: u32,
    artifact_class: String,
    promotion_eligibility: String,
    discovery_result: ResearchQualityBody,
}

#[derive(serde::Deserialize)]
struct ResearchQualityBody {
    quality_metrics: Vec<StratMetrics>,
}

fn load_quality(portfolio_path: &std::path::Path) -> Result<Vec<StratMetrics>, String> {
    let name = portfolio_path.to_string_lossy();
    let stem = name
        .strip_suffix(".live_portfolio.json")
        .ok_or_else(|| "Selected artifact is not a live portfolio".to_string())?;
    let sidecar = PathBuf::from(format!("{stem}.quality.json"));
    // Only a missing legacy sidecar permits the current App producer's
    // embedded research source. A malformed/unreadable existing sidecar fails.
    let embedded =
        !sidecar.try_exists().map_err(|error| error.to_string())? && stem.ends_with(".research");
    let quality_source = if embedded {
        PathBuf::from(format!("{stem}.json"))
    } else {
        sidecar
    };
    let portfolio_stamp = file_stamp(portfolio_path);
    let quality_stamp = file_stamp(&quality_source);
    if let Ok(cache) = quality_cache().lock() {
        if let Some(entry) = cache.get(portfolio_path) {
            if entry.portfolio_stamp == portfolio_stamp
                && entry.quality_source == quality_source
                && entry.quality_stamp == quality_stamp
            {
                return entry.metrics.clone();
            }
        }
    }
    let metrics = (|| {
        let portfolio =
            neoethos_search::live_portfolio::load_live_portfolio_json(portfolio_path)
                .map_err(|_| "Selected portfolio identity is unavailable or invalid".to_string())?;
        let selected_ids: Vec<String> = portfolio
            .genes
            .into_iter()
            .map(|gene| gene.strategy_id)
            .collect();
        let reader = std::io::BufReader::new(
            std::fs::File::open(&quality_source)
                .map_err(|_| "Quality diagnostics are missing or unreadable".to_owned())?,
        );
        let rows = if embedded {
            let projection: ResearchQualityProjection = serde_json::from_reader(reader)
                .map_err(|_| "Research quality rows are incomplete or malformed".to_owned())?;
            if projection.schema_version != 3
                || projection.artifact_class != "research_only"
                || projection.promotion_eligibility != "not_promotion_eligible"
            {
                return Err("Unsupported research quality envelope".to_owned());
            }
            projection.discovery_result.quality_metrics
        } else {
            serde_json::from_reader(reader)
                .map_err(|_| "Quality sidecar rows are incomplete or malformed".to_owned())?
        };
        bind_strategy_metrics(rows, &selected_ids)
    })();
    if let Ok(mut cache) = quality_cache().lock() {
        cache.insert(
            portfolio_path.to_path_buf(),
            QualityEntry {
                portfolio_stamp,
                quality_source,
                quality_stamp,
                metrics: metrics.clone(),
            },
        );
    }
    metrics
}

fn draw_details(area: Rect, buf: &mut Buffer, p: &PortfolioSummary) {
    if area.height == 0 {
        return;
    }
    let metrics = load_quality(&p.path);
    let lines = quality_detail_lines(
        &p.name,
        metrics.as_deref().map_err(String::as_str),
        area.height.saturating_sub(4) as usize,
    );
    Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::TOP)
                .border_style(Style::default().fg(theme::BORDER)),
        )
        .render(area, buf);
}

fn quality_detail_lines(
    name: &str,
    metrics: Result<&[StratMetrics], &str>,
    max_rows: usize,
) -> Vec<Line<'static>> {
    let mut header = vec![Span::styled(
        format!(" {name} "),
        theme::accent_style().add_modifier(Modifier::BOLD),
    )];
    if let Ok(rows) = metrics {
        header.push(Span::styled(
            format!("· {} selected  ", rows.len()),
            theme::muted_style(),
        ));
    }
    header.push(Span::styled(
        "[↑↓] [V]alidate [P]romote",
        theme::caption_style(),
    ));
    let mut lines: Vec<Line> = vec![Line::from(header)];

    let metrics = match metrics {
        Ok(rows) => rows,
        Err(reason) => {
            lines.push(Line::styled(
                format!("  Per-strategy metrics unavailable: {reason}"),
                theme::warn_style(),
            ));
            return lines;
        }
    };
    if metrics.is_empty() {
        lines.push(Line::styled(
            "  No selected strategies with quality metrics.",
            theme::caption_style(),
        ));
    } else {
        // These are independently sized IS backtests, not a shared-account
        // ledger. Neither a sidecar nor a research projection is OOS proof.
        lines.push(Line::styled(
            "  Unsealed IS diagnostics · full balance per strategy · account units · NOT portfolio PnL",
            theme::caption_style(),
        ));
        lines.push(Line::from(vec![Span::styled(
            format!(
                "  {:<20}{:>10}{:>10}{:>8}{:>8}{:>7}{:>7}",
                "Strategy ID", "Start", "Net", "Ret%", "Sharpe", "DD%", "Win%"
            ),
            theme::caption_style().add_modifier(Modifier::BOLD),
        )]));
        for m in metrics.iter().take(max_rows) {
            let dd_pct = m.max_dd * 100.0;
            // Operator's low-DD lens: green ≤6%, amber ≤10%, red above.
            let dd_style = if dd_pct <= 6.0 {
                theme::primary_style()
            } else if dd_pct <= 10.0 {
                theme::warn_style()
            } else {
                Style::default().fg(theme::SELL)
            };
            let net_style = if m.net_profit >= 0.0 {
                theme::buy_style()
            } else {
                Style::default().fg(theme::SELL)
            };
            lines.push(Line::from(vec![
                Span::styled(format!("  {:<20.20}", m.strategy_id), theme::muted_style()),
                Span::styled(
                    format!("{:>10}", fmt_money(m.initial_capital)),
                    theme::muted_style(),
                ),
                Span::styled(format!("{:>10}", fmt_money(m.net_profit)), net_style),
                Span::styled(
                    format!("{:>7.1}%", m.return_pct * 100.0),
                    theme::primary_style(),
                ),
                Span::styled(format!("{:>8.2}", m.sharpe), theme::primary_style()),
                Span::styled(format!("{:>6.1}%", dd_pct), dd_style),
                Span::styled(format!("{:>6.0}%", m.win * 100.0), theme::muted_style()),
            ]));
        }
    }
    lines
}

/// Keep every metric attached to its selected object/ID. Unknown large fields
/// are skipped by serde; no token-order or row-index join is permitted.
#[cfg(test)]
fn extract_strategy_metrics(
    text: &str,
    selected_ids: &[String],
) -> Result<Vec<StratMetrics>, String> {
    let rows: Vec<StratMetrics> = serde_json::from_str(text)
        .map_err(|_| "Quality sidecar rows are incomplete or malformed".to_string())?;
    bind_strategy_metrics(rows, selected_ids)
}

fn bind_strategy_metrics(
    rows: Vec<StratMetrics>,
    selected_ids: &[String],
) -> Result<Vec<StratMetrics>, String> {
    let selected: std::collections::HashSet<&str> =
        selected_ids.iter().map(String::as_str).collect();
    if selected.len() != selected_ids.len() || selected.iter().any(|id| id.trim().is_empty()) {
        return Err("Selected strategy IDs are missing or ambiguous".to_string());
    }
    let mut by_id = std::collections::HashMap::with_capacity(selected.len());
    for row in rows {
        if !selected.contains(row.strategy_id.as_str()) {
            continue;
        }
        if row.initial_capital <= 0.0
            || [
                row.initial_capital,
                row.net_profit,
                row.return_pct * 100.0,
                row.sharpe,
                row.max_dd * 100.0,
                row.win * 100.0,
            ]
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(format!(
                "Invalid quality metrics for selected ID {}",
                row.strategy_id
            ));
        }
        let id = row.strategy_id.clone();
        if by_id.insert(id.clone(), row).is_some() {
            return Err(format!("Ambiguous quality rows for selected ID {id}"));
        }
    }
    selected_ids
        .iter()
        .map(|id| {
            by_id
                .remove(id)
                .ok_or_else(|| format!("No quality row for selected ID {id}"))
        })
        .collect()
}

/// Compact money formatting for the narrow details panel: thousands as `k`,
/// millions as `M`, with sign preserved.
fn fmt_money(v: f64) -> String {
    let a = v.abs();
    if a >= 1_000_000.0 {
        format!("{:.2}M", v / 1_000_000.0)
    } else if a >= 1_000.0 {
        format!("{:.1}k", v / 1_000.0)
    } else {
        format!("{:.0}", v)
    }
}

struct PortfolioSummary {
    name: String,
    strategies: Result<usize, String>,
    bytes: u64,
    modified: String,
    path: PathBuf,
}

/// Existing producer locations under the configured cache only; no CWD guesses
/// and no recursive scan into models, temporary trees or unrelated profiles.
pub(super) fn artifact_dirs(cache_root: &std::path::Path) -> [PathBuf; 6] {
    [
        cache_root.to_path_buf(),
        cache_root.join("discovery"),
        cache_root.join("discovery").join("research"),
        cache_root.join("auto_loop"),
        cache_root.join("schedule"),
        cache_root.join("discovery_test"),
    ]
}

pub(super) fn collect_artifact_files(
    cache_root: &std::path::Path,
    accepts: impl Fn(&str) -> bool,
) -> Result<Vec<PathBuf>, String> {
    let mut found = Vec::new();
    for dir in artifact_dirs(cache_root) {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(format!("{}: {error}", dir.display())),
        };
        for entry in entries {
            let entry = entry.map_err(|error| format!("{}: {error}", dir.display()))?;
            let path = entry.path();
            if !path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(&accepts)
            {
                continue;
            }
            let metadata = entry
                .metadata()
                .map_err(|error| format!("{}: {error}", path.display()))?;
            if metadata.is_file() {
                let modified = metadata
                    .modified()
                    .map_err(|error| format!("{}: {error}", path.display()))?;
                found.push((modified, path));
            }
        }
    }
    found.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    found.dedup_by(|a, b| a.1 == b.1);
    Ok(found.into_iter().map(|(_, path)| path).collect())
}

fn scan_portfolios(cache_root: &std::path::Path) -> Result<Vec<PortfolioSummary>, String> {
    collect_artifact_files(cache_root, |name| name.ends_with(".live_portfolio.json"))?
        .into_iter()
        .map(|path| {
            let metadata =
                std::fs::metadata(&path).map_err(|error| format!("{}: {error}", path.display()))?;
            let modified = metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|duration| format_ts(duration.as_secs()))
                .unwrap_or_else(|| "—".to_owned());
            Ok(PortfolioSummary {
                name: path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
                strategies: count_strategies(&path),
                bytes: metadata.len(),
                modified,
                path,
            })
        })
        .collect()
}

type PortfolioFileCensus = Result<(usize, usize), String>;
type PortfolioCensusCache = Option<(std::time::Instant, PathBuf, PortfolioFileCensus)>;

/// A file census, NOT a certificate that these portfolios can trade.
/// Cache this compact Dashboard projection for two seconds, including errors;
/// redraws must not enumerate all producer directories every frame.
pub(super) fn portfolio_file_counts(cache_root: &std::path::Path) -> PortfolioFileCensus {
    static CACHE: std::sync::Mutex<PortfolioCensusCache> = std::sync::Mutex::new(None);
    let mut cache = CACHE.lock().unwrap_or_else(|error| error.into_inner());
    cached_portfolio_file_counts(&mut cache, cache_root, std::time::Instant::now, || {
        let files = scan_portfolios(cache_root)?;
        Ok((
            files.len(),
            files.iter().filter(|file| file.strategies.is_err()).count(),
        ))
    })
}

fn cached_portfolio_file_counts(
    cache: &mut PortfolioCensusCache,
    cache_root: &std::path::Path,
    clock: impl Fn() -> std::time::Instant,
    scan: impl FnOnce() -> PortfolioFileCensus,
) -> PortfolioFileCensus {
    if let Some((at, root, counts)) = cache.as_ref() {
        if root == cache_root
            && clock().saturating_duration_since(*at) < std::time::Duration::from_secs(2)
        {
            return counts.clone();
        }
    }
    let counts = scan();
    // TTL starts after the scan, so a slow decode still receives a full pause.
    *cache = Some((clock(), cache_root.to_path_buf(), counts.clone()));
    counts
}

type CountEntry = (FileStamp, Result<usize, String>);

fn count_cache() -> &'static std::sync::Mutex<std::collections::HashMap<PathBuf, CountEntry>> {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<PathBuf, CountEntry>>,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn count_strategies(path: &std::path::Path) -> Result<usize, String> {
    let stamp = file_stamp(path);
    if let Ok(cache) = count_cache().lock() {
        if let Some((cached_stamp, count)) = cache.get(path) {
            if *cached_stamp == stamp {
                return count.clone();
            }
        }
    }
    // The same validated V6/legacy loader used by selected quality details.
    // A random payload/array or malformed artifact is unknown, never zero.
    let count = neoethos_search::live_portfolio::load_live_portfolio_json(path)
        .map(|portfolio| portfolio.genes.len())
        .map_err(|error| format!("Portfolio unavailable: {error:#}"));
    if let Ok(mut cache) = count_cache().lock() {
        cache.insert(path.to_path_buf(), (stamp, count.clone()));
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

    fn quality_row(id: &str, net: f64, return_fraction: f64) -> serde_json::Value {
        serde_json::json!({
            "strategy_id": id,
            "initial_capital": 10_000.0,
            "net_profit": net,
            "total_return_pct": return_fraction,
            "sharpe_ratio": 1.5,
            "max_drawdown_pct": 0.06,
            "win_rate": 0.55,
            // An actual writer field ignored by the compact display projection.
            "equity_curve": [10_000.0, 10_000.0 + net]
        })
    }

    fn details_text(metrics: Result<&[StratMetrics], &str>) -> String {
        quality_detail_lines("fixture.live_portfolio.json", metrics, 10)
            .into_iter()
            .map(|line| {
                line.spans
                    .into_iter()
                    .map(|span| span.content.into_owned())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn selected_quality_rows_are_id_bound_and_standalone_profits_are_not_added() {
        // All three rows belong to the quality census, but only A/B were
        // selected. Each replay used the same complete 10,000-unit balance;
        // neither 903,000 nor 3,000 is a measured shared-account portfolio PnL.
        let text = serde_json::to_string(&vec![
            quality_row("selected-b", 2_000.0, 0.20),
            quality_row("unselected-outlier", 900_000.0, 90.0),
            quality_row("selected-a", 1_000.0, 0.10),
        ])
        .unwrap();
        let metrics =
            extract_strategy_metrics(&text, &["selected-a".to_string(), "selected-b".to_string()])
                .unwrap();
        assert_eq!(metrics.len(), 2);
        assert_eq!(metrics[0].strategy_id, "selected-a");
        assert_eq!(metrics[0].net_profit, 1_000.0);
        assert_eq!(metrics[1].strategy_id, "selected-b");
        assert_eq!(metrics[1].net_profit, 2_000.0);
        assert!(metrics.iter().all(|row| row.initial_capital == 10_000.0));

        let rendered = details_text(Ok(&metrics));
        assert!(rendered.contains("2 selected"));
        assert!(rendered.contains("Unsealed IS diagnostics"));
        assert!(rendered.contains("full balance per strategy"));
        assert!(rendered.contains("account units"));
        assert!(rendered.contains("NOT portfolio PnL"));
        assert!(rendered.contains("1.0k"));
        assert!(rendered.contains("2.0k"));
        assert!(rendered.contains("10.0%"), "0.10 must render as 10.0%");
        assert!(rendered.contains("20.0%"));
        assert!(!rendered.contains("0.1%"));
        assert!(!rendered.contains("unselected-outlier"));
        assert!(!rendered.contains("900.0k"));
        assert!(!rendered.contains("903.0k"));
        assert!(!rendered.contains("3.0k"));
        assert!(!rendered.contains("Σnet"));
        assert!(!rendered.contains('€'));
    }

    #[test]
    fn missing_or_ambiguous_selected_quality_ids_are_unavailable() {
        let selected = ["selected-a".to_string(), "selected-b".to_string()];
        for rows in [
            vec![quality_row("selected-a", 1_000.0, 0.10)],
            vec![
                quality_row("selected-a", 1_000.0, 0.10),
                quality_row("selected-b", 2_000.0, 0.20),
                quality_row("selected-a", 4_000.0, 0.40),
            ],
        ] {
            let error = extract_strategy_metrics(&serde_json::to_string(&rows).unwrap(), &selected)
                .expect_err("missing or duplicate selected ID must not guess a row");
            let rendered = details_text(Err(&error));
            assert!(rendered.contains("Per-strategy metrics unavailable"));
            assert!(!rendered.contains("1.0k"));
            assert!(!rendered.contains("2.0k"));
        }
        assert!(extract_strategy_metrics("[]", &["".to_string()]).is_err());
        assert!(
            extract_strategy_metrics("[]", &["selected-a".to_string(), "selected-a".to_string()])
                .is_err()
        );
    }

    #[test]
    fn incomplete_quality_objects_never_borrow_another_rows_numbers() {
        let mut first = quality_row("selected-a", 1_000.0, 0.10);
        first.as_object_mut().unwrap().remove("total_return_pct");
        let text =
            serde_json::to_string(&vec![first, quality_row("selected-b", 2_000.0, 0.20)]).unwrap();
        let error =
            extract_strategy_metrics(&text, &["selected-a".to_string(), "selected-b".to_string()])
                .expect_err("object-bound decoding must reject the former token-scan cross-wire");
        assert!(details_text(Err(&error)).contains("incomplete or malformed"));
    }

    #[test]
    fn selected_quality_display_preserves_zero_and_negative_results() {
        let text = serde_json::to_string(&vec![
            quality_row("zero", 0.0, 0.0),
            quality_row("loss", -1_000.0, -0.10),
        ])
        .unwrap();
        let metrics =
            extract_strategy_metrics(&text, &["zero".to_string(), "loss".to_string()]).unwrap();
        assert_eq!(metrics[0].net_profit, 0.0);
        assert_eq!(metrics[1].net_profit, -1_000.0);
        let rendered = details_text(Ok(&metrics));
        assert!(rendered.contains("-1.0k"));
        assert!(rendered.contains("-10.0%"));
        assert!(!rendered.contains("Σnet"));
    }

    #[test]
    fn configured_research_cache_uses_real_writer_portfolio_and_rejects_fake_arrays() {
        let root = std::env::temp_dir().join(format!(
            "neoethos-tui-portfolio-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let research = root.join("discovery/research");
        std::fs::create_dir_all(&research).unwrap();
        let portfolio = research.join("actual.research.live_portfolio.json");
        let bytes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../neoethos-search/test_fixtures/strategy_report_v6/8510cec7a31d6c18b3ed61709a9315fee0ea2e4c9b7c4fb2220f73d314d7f02d.research.live_portfolio.json"
        ));
        std::fs::write(&portfolio, bytes).unwrap();
        std::fs::write(research.join("actual.research.json"), include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"),
            "/../neoethos-search/test_fixtures/strategy_report_v6/8510cec7a31d6c18b3ed61709a9315fee0ea2e4c9b7c4fb2220f73d314d7f02d.research.json"))).unwrap();
        std::fs::write(research.join("actual.research.costs.json"), b"{}").unwrap();
        let invalid = research.join("invalid.live_portfolio.json");
        std::fs::write(&invalid, br#"{"payload":[{},{}],"candidates":[{},{}]}"#).unwrap();
        assert_eq!(count_strategies(&portfolio).unwrap(), 2);
        let quality = load_quality(&portfolio).unwrap();
        assert_eq!(quality.len(), 2);
        assert_eq!(quality[0].strategy_id, "report-selected-a");
        assert_eq!(quality[0].net_profit, 50.0);
        assert_eq!(quality[1].strategy_id, "report-selected-b");
        assert_eq!(quality[1].net_profit, 100.0);
        assert!(
            quality
                .iter()
                .all(|row| row.strategy_id != "report-not-selected")
        );
        let rendered = details_text(Ok(&quality));
        assert!(rendered.contains("NOT portfolio PnL"));
        assert!(!rendered.contains("27.0k"));
        let sidecar = research.join("actual.research.quality.json");
        std::fs::write(&sidecar, b"malformed existing sidecar").unwrap();
        assert!(
            load_quality(&portfolio).is_err(),
            "malformed sidecar must not silently fall back"
        );
        std::fs::remove_file(sidecar).unwrap();
        assert_eq!(load_quality(&portfolio).unwrap().len(), 2);
        assert!(count_strategies(&invalid).is_err());
        assert_eq!(portfolio_file_counts(&root).unwrap(), (2, 1));
        assert_eq!(scan_portfolios(&root).unwrap().len(), 2);
        // A different configured root must not reuse this profile's file list/count.
        let other = root.join("other-cache");
        std::fs::create_dir(&other).unwrap();
        assert_eq!(portfolio_file_counts(&other).unwrap(), (0, 0));
        std::fs::write(&portfolio, b"{}").unwrap();
        assert!(
            count_strategies(&portfolio).is_err(),
            "changed file invalidates cached count"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn portfolio_census_is_root_bound_and_caches_success_and_errors_for_two_seconds() {
        use std::cell::Cell;
        use std::time::{Duration, Instant};

        let start = Instant::now();
        let now = Cell::new(start);
        let scans = Cell::new(0);
        let mut cache = None;
        let first = std::path::Path::new("first-configured-cache");
        let second = std::path::Path::new("second-configured-cache");
        let read = |cache: &mut PortfolioCensusCache,
                    root: &std::path::Path,
                    result: PortfolioFileCensus| {
            cached_portfolio_file_counts(
                cache,
                root,
                || now.get(),
                || {
                    scans.set(scans.get() + 1);
                    result
                },
            )
        };

        assert_eq!(read(&mut cache, first, Ok((2, 1))), Ok((2, 1)));
        now.set(start + Duration::from_millis(1_999));
        assert_eq!(
            read(&mut cache, first, Err("must not scan".into())),
            Ok((2, 1))
        );
        assert_eq!(scans.get(), 1);

        let unavailable = Err("unreadable configured cache".to_owned());
        assert_eq!(read(&mut cache, second, unavailable.clone()), unavailable);
        assert_eq!(read(&mut cache, second, Ok((0, 0))), unavailable);
        assert_eq!(scans.get(), 2, "a cached error remains unknown, not zero");
        assert_eq!(read(&mut cache, first, Ok((3, 0))), Ok((3, 0)));
        assert_eq!(
            scans.get(),
            3,
            "changing roots cannot reuse another inventory"
        );

        now.set(start + Duration::from_millis(3_999));
        assert_eq!(read(&mut cache, first, unavailable.clone()), unavailable);
        assert_eq!(scans.get(), 4, "exactly two seconds must refresh");
        now.set(start + Duration::from_millis(5_998));
        assert_eq!(read(&mut cache, first, Ok((0, 0))), unavailable);
        assert_eq!(scans.get(), 4);
        now.set(start + Duration::from_millis(5_999));
        assert_eq!(read(&mut cache, first, Ok((0, 0))), Ok((0, 0)));
        assert_eq!(scans.get(), 5, "expired errors must permit recovery");
    }

    #[test]
    fn portfolio_inventory_reports_unreadable_location_instead_of_false_zero() {
        let root = std::env::temp_dir().join(format!(
            "neoethos-tui-invalid-cache-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&root, b"not a directory").unwrap();
        assert!(portfolio_file_counts(&root).is_err());
        std::fs::remove_file(root).unwrap();
    }
}
