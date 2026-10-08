//! Saved diagnostics from actual discovery writers: standalone *_funnel.json
//! and the embedded funnel in current *.research.json envelopes. Read-only
//! metadata display is never portfolio admission or execution evidence.

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Padding, Paragraph, Widget, Wrap};

use crate::tui::app::AppShared;
use crate::tui::theme;

pub fn handle_key(code: crossterm::event::KeyCode, shared: &mut AppShared) -> bool {
    use crossterm::event::KeyCode;
    match code {
        KeyCode::PageUp => {
            shared.funnel_scroll = shared.funnel_scroll.saturating_sub(10);
            return true;
        }
        KeyCode::PageDown => {
            shared.funnel_scroll = shared.funnel_scroll.saturating_add(10);
            return true;
        }
        KeyCode::Home => {
            shared.funnel_scroll = 0;
            return true;
        }
        KeyCode::Up | KeyCode::Down => {}
        _ => return false,
    }
    let files = match collect_funnel_files(&shared.cache_root) {
        Ok(files) => files,
        Err(error) => {
            shared.status = error;
            return true;
        }
    };
    if files.is_empty() {
        return true;
    }
    let current = shared
        .funnel_selected_path
        .as_ref()
        .and_then(|path| files.iter().position(|file| file == path))
        .unwrap_or(0);
    let next = if code == KeyCode::Up {
        current.saturating_sub(1)
    } else {
        (current + 1).min(files.len() - 1)
    };
    shared.funnel_selected_path = Some(files[next].clone());
    shared.funnel_scroll = 0;
    true
}

pub fn draw(area: Rect, buf: &mut Buffer, shared: &AppShared) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .margin(1)
        .spacing(1)
        .split(area);
    // One root-bound inventory snapshot for both panels, not two independent scans.
    match collect_funnel_files(&shared.cache_root) {
        Ok(files) => {
            let selected = match shared.funnel_selected_path.as_ref() {
                Some(path) => files.iter().find(|file| *file == path),
                None => files.first(),
            };
            render_run_list(cols[0], buf, &files, selected);
            render_selected_funnel(cols[1], buf, selected, shared.funnel_scroll);
        }
        Err(error) => {
            Paragraph::new(Line::styled(
                format!("Funnel inventory unavailable: {error}"),
                theme::sell_style(),
            ))
            .render(area, buf);
        }
    }
}

fn render_run_list(
    area: Rect,
    buf: &mut Buffer,
    files: &[std::path::PathBuf],
    selected: Option<&std::path::PathBuf>,
) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER))
        .title(Span::styled(
            " SAVED FUNNELS · ↑↓ select ",
            theme::caption_style(),
        ))
        .style(theme::panel_block_style())
        .padding(Padding::new(1, 1, 0, 0));
    let inner = block.inner(area);
    block.render(area, buf);
    let selected_index = selected
        .and_then(|path| files.iter().position(|file| file == path))
        .unwrap_or(0);
    let start = selected_index.saturating_sub((inner.height as usize).saturating_sub(1));
    let lines: Vec<Line> = if files.is_empty() {
        vec![Line::styled(
            "No saved funnels in the configured cache.",
            theme::muted_style(),
        )]
    } else {
        files
            .iter()
            .skip(start)
            .take(inner.height as usize)
            .map(|path| {
                let marker = if Some(path) == selected { "▸" } else { " " };
                Line::styled(
                    format!(
                        "{marker} {}",
                        path.file_name().unwrap_or_default().to_string_lossy()
                    ),
                    if Some(path) == selected {
                        theme::accent_style()
                    } else {
                        theme::primary_style()
                    },
                )
            })
            .collect()
    };
    Paragraph::new(lines).render(inner, buf);
}

fn render_selected_funnel(
    area: Rect,
    buf: &mut Buffer,
    selected: Option<&std::path::PathBuf>,
    scroll: u16,
) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER))
        .title(Span::styled(
            " SAVED DIAGNOSTICS · PgUp/PgDn · Home ",
            theme::caption_style(),
        ))
        .style(theme::panel_block_style())
        .padding(Padding::new(1, 1, 0, 0));
    let inner = block.inner(area);
    block.render(area, buf);
    let lines = match selected {
        Some(path) => match load_funnel_file(path) {
            Ok((research, value)) => {
                let mut lines = vec![
                    Line::styled(
                        if research {
                            "ResearchOnly / NotPromotionEligible — saved metadata, not execution proof"
                        } else {
                            "Saved funnel diagnostics — not execution or promotion evidence"
                        },
                        theme::warn_style(),
                    ),
                    Line::styled(
                        format!("source: {}", path.display()),
                        theme::caption_style(),
                    ),
                ];
                lines.extend(render_funnel_value(&value, path));
                lines
            }
            Err(error) => vec![Line::styled(
                format!("{}: {error}", path.display()),
                theme::sell_style(),
            )],
        },
        None => vec![Line::styled(
            "No selected funnel available. Choose an existing run with ↑↓.",
            theme::muted_style(),
        )],
    };
    Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0))
        .render(inner, buf);
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
    let bottleneck_rejected = v.get("bottleneck_rejected").and_then(|n| n.as_u64());

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
                "· bottleneck={} ({} not advanced)",
                bottleneck,
                count_text(bottleneck_rejected)
            ),
            theme::muted_style(),
        ),
    ]));
    out.push(Line::styled(
        "  Not advanced is not necessarily failed: caps and untested candidates are separate.",
        theme::muted_style(),
    ));
    out.push(Line::styled(
        "  Missing counts are '?'; zero alone does not prove a stage ran.",
        theme::muted_style(),
    ));
    out.push(Line::raw(""));

    if let Some(census) = v
        .get("candidate_census")
        .and_then(|value| value.as_object())
    {
        // These counts come from the saved run, not differences between stage
        // rows (which can count bars, features, candidates or gene/fold tests).
        const GROUPS: &[&[(&str, &str)]] = &[
            &[
                ("ga_returned_candidates", "GA pool"),
                ("quality_evaluated", "Quality evaluated"),
            ],
            &[
                ("validation_candidate_limit", "Validation limit (0=all)"),
                ("validation_candidates_admitted", "Admitted"),
            ],
            &[("validation_candidates_capped", "Not tested / budget cap")],
            &[
                ("walkforward_tested", "WF tested"),
                ("walkforward_passed", "Passed"),
                ("walkforward_failed", "Failed"),
            ],
            &[("walkforward_not_tested", "WF not tested")],
            &[
                ("correlation_tested", "Correlation checked"),
                ("rejected_by_correlation", "Rejected"),
            ],
            &[
                ("portfolio_selected", "Selected at reported stage"),
                ("portfolio_capacity_not_selected", "Not selected / capacity"),
            ],
            &[("robustness_removed", "Robustness removed")],
        ];
        for fields in GROUPS {
            let observations = fields
                .iter()
                .filter_map(|(key, label)| {
                    census
                        .get(*key)
                        .map(|value| format!("{label}: {}", count_text(value.as_u64())))
                })
                .collect::<Vec<_>>();
            if !observations.is_empty() {
                out.push(Line::styled(
                    format!("  {}", observations.join(" · ")),
                    theme::primary_style(),
                ));
            }
        }
    } else {
        out.push(Line::styled(
            "  Candidate coverage not recorded in this funnel.",
            theme::muted_style(),
        ));
    }
    out.push(Line::raw(""));

    if let Some(stages) = v.get("stages").and_then(|s| s.as_array()) {
        for s in stages {
            let name = s.get("name").and_then(|n| n.as_str()).unwrap_or("?");
            let cin = s.get("count_in").and_then(|n| n.as_u64());
            let cout = s.get("count_out").and_then(|n| n.as_u64());
            let rej = s.get("rejected").and_then(|n| n.as_u64());
            let style = if name == bottleneck && rej.is_some_and(|value| value > 0) {
                Style::default()
                    .fg(theme::ACCENT)
                    .add_modifier(Modifier::BOLD)
            } else if cout.is_some_and(|value| value > 0) && cout == cin {
                Style::default().fg(theme::BUY)
            } else if cin.is_none() || cin == Some(0) {
                theme::muted_style()
            } else {
                Style::default().fg(theme::TEXT_PRIMARY)
            };
            out.push(Line::styled(
                format!(
                    "  {}  in={}  out={}  not advanced={}",
                    name.replace('_', " "),
                    count_text(cin),
                    count_text(cout),
                    count_text(rej)
                ),
                style,
            ));
            if let Some(reasons) = s.get("top_reasons").and_then(|value| value.as_array()) {
                for reason in reasons {
                    let label = reason
                        .get(0)
                        .and_then(|value| value.as_str())
                        .unwrap_or("unknown reason");
                    let count = reason.get(1).and_then(|value| value.as_u64());
                    let disposition = if label == "portfolio_capacity_not_selected" {
                        "Not selected / capacity"
                    } else if label.starts_with("not_tested")
                        || label == "validation_candidates_capped"
                    {
                        "Not tested"
                    } else {
                        "Reason"
                    };
                    out.push(Line::styled(
                        format!(
                            "    {}: {} = {}",
                            disposition,
                            label.replace('_', " "),
                            count_text(count),
                        ),
                        if disposition != "Reason" {
                            theme::accent_style()
                        } else {
                            theme::muted_style()
                        },
                    ));
                }
            }
            if let Some(omitted) = s
                .get("reasons_truncated")
                .and_then(|value| value.as_u64())
                .filter(|value| *value > 0)
            {
                out.push(Line::styled(
                    format!("    {omitted} additional reason buckets were not saved."),
                    theme::warn_style(),
                ));
            }
        }
    }
    out.push(Line::raw(""));
    out.push(Line::styled(
        format!("  source: {}", path.display()),
        theme::caption_style(),
    ));
    out
}

fn count_text(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "?".to_owned())
}

fn collect_funnel_files(cache_root: &std::path::Path) -> Result<Vec<std::path::PathBuf>, String> {
    use std::sync::Mutex;
    use std::time::{Duration, Instant};
    type Cache = Option<(
        Instant,
        std::path::PathBuf,
        Result<Vec<std::path::PathBuf>, String>,
    )>;
    static CACHE: Mutex<Cache> = Mutex::new(None);
    {
        let guard = CACHE.lock().unwrap_or_else(|error| error.into_inner());
        if let Some((at, root, files)) = guard.as_ref() {
            if root == cache_root && at.elapsed() < Duration::from_secs(2) {
                return files.clone();
            }
        }
    }
    let files = super::strategies::collect_artifact_files(cache_root, |name| {
        name.ends_with("_funnel.json") || name.ends_with(".research.json")
    });
    *CACHE.lock().unwrap_or_else(|error| error.into_inner()) =
        Some((Instant::now(), cache_root.to_path_buf(), files.clone()));
    files
}

#[derive(serde::Deserialize)]
struct ResearchFunnelProjection {
    schema_version: u32,
    artifact_class: String,
    promotion_eligibility: String,
    discovery_result: ResearchFunnelBody,
}

#[derive(serde::Deserialize)]
struct ResearchFunnelBody {
    funnel_profile: Option<serde_json::Value>,
}

fn load_funnel_file(path: &std::path::Path) -> Result<(bool, serde_json::Value), String> {
    use std::io::BufReader;
    type Stamp = (std::time::SystemTime, u64);
    type Cached = Option<(
        std::path::PathBuf,
        Stamp,
        Result<(bool, serde_json::Value), String>,
    )>;
    static CACHE: std::sync::Mutex<Cached> = std::sync::Mutex::new(None);
    let metadata = std::fs::metadata(path).map_err(|error| error.to_string())?;
    let stamp = (
        metadata.modified().map_err(|error| error.to_string())?,
        metadata.len(),
    );
    {
        let cache = CACHE.lock().unwrap_or_else(|error| error.into_inner());
        if let Some((cached_path, cached_stamp, result)) = cache.as_ref() {
            if cached_path == path && *cached_stamp == stamp {
                return result.clone();
            }
        }
    }
    let result = (|| {
        let file = std::fs::File::open(path).map_err(|error| error.to_string())?;
        let reader = BufReader::new(file);
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".research.json"))
        {
            // Streaming serde skips the large metrics/ledger/feature fields.
            // Only the compact funnel is retained. This is NOT authority validation.
            let projected: ResearchFunnelProjection =
                serde_json::from_reader(reader).map_err(|error| error.to_string())?;
            if projected.schema_version != 3
                || projected.artifact_class != "research_only"
                || projected.promotion_eligibility != "not_promotion_eligible"
            {
                return Err(
                    "Unsupported research envelope; no authority or eligibility inferred"
                        .to_owned(),
                );
            }
            let value = projected
                .discovery_result
                .funnel_profile
                .ok_or_else(|| "Research envelope did not record a funnel".to_owned())?;
            Ok((true, value))
        } else {
            serde_json::from_reader(reader)
                .map(|value| (false, value))
                .map_err(|error| error.to_string())
        }
    })();
    *CACHE.lock().unwrap_or_else(|error| error.into_inner()) =
        Some((path.to_path_buf(), stamp, result.clone()));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_cache_reads_actual_research_writer_and_actual_standalone_funnel_writer() {
        let root = std::env::temp_dir().join(format!(
            "neoethos-tui-funnels-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let research_dir = root.join("discovery/research");
        std::fs::create_dir_all(&research_dir).unwrap();
        let research_path = research_dir.join("actual.research.json");
        let bytes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../neoethos-search/test_fixtures/strategy_report_v6/a2d3ef812dd8b341f863f7a57df3f71172a94c95269fe51c49ece25e3623d327.research.json"
        ));
        std::fs::write(&research_path, bytes).unwrap();
        // Neither metadata sidecar is a separate result/funnel.
        std::fs::write(research_dir.join("actual.research.costs.json"), b"{}").unwrap();
        std::fs::write(
            research_dir.join("actual.research.live_portfolio.json"),
            b"{}",
        )
        .unwrap();
        let mut profile = neoethos_search::funnel_profile::FunnelProfile::new("EURUSD", "M5");
        profile.record_stage("passed_quality", 5, 2);
        profile.add_reject_reason("passed_quality", "quality_rejection_fixture", 3);
        let portfolio_path = root.join("discovery/ordinary.json");
        profile.save_next_to(&portfolio_path).unwrap();
        let standalone = neoethos_search::funnel_profile::funnel_path_for(&portfolio_path);
        let files = collect_funnel_files(&root).unwrap();
        assert_eq!(files.len(), 2);
        assert!(files.contains(&research_path));
        assert!(files.contains(&standalone));
        let (is_research, value) = load_funnel_file(&research_path).unwrap();
        assert!(is_research);
        assert_eq!(
            value["outcome"],
            "synthetic-report-fixture-not-trading-evidence"
        );
        assert!(!value.as_object().unwrap().contains_key("quality_metrics"));
        let (is_research, value) = load_funnel_file(&standalone).unwrap();
        assert!(!is_research);
        assert!(rendered(value).contains("quality rejection fixture = 3"));
        assert!(
            collect_funnel_files(&root.join("another-cache"))
                .unwrap()
                .is_empty()
        );
        // Same selected path, changed bytes: never return a previous good result.
        std::fs::write(&research_path, b"{}").unwrap();
        assert!(load_funnel_file(&research_path).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn malformed_research_cannot_be_presented_as_a_plain_funnel() {
        let root = std::env::temp_dir().join(format!(
            "neoethos-tui-refused-funnel-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let path = root.join("invalid.research.json");
        for bytes in [
            br#"{"stages":[]}"#.as_slice(),
            br#"{"schema_version":3,"artifact_class":"live","promotion_eligibility":"not_promotion_eligible","discovery_result":{"funnel_profile":{"stages":[]}}}"#.as_slice(),
            br#"{"schema_version":3,"artifact_class":"research_only","promotion_eligibility":"not_promotion_eligible","discovery_result":{"funnel_profile":null}}"#.as_slice(),
        ] {
            std::fs::write(&path, bytes).unwrap();
            assert!(load_funnel_file(&path).is_err());
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    fn rendered(value: serde_json::Value) -> String {
        render_funnel_value(&value, std::path::Path::new("fixture_funnel.json"))
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
    fn funnel_shows_saved_reasons_without_calling_budget_caps_oos_failures() {
        let text = rendered(serde_json::json!({
            "symbol": "EURUSD", "timeframe": "M5", "outcome": "no_candidates",
            "bottleneck_stage": "validation_budget", "bottleneck_rejected": 9800,
            "stages": [
                {"name":"validation_budget", "count_in":10000, "count_out":200, "rejected":9800,
                 "top_reasons":[["validation_candidates_capped",9800]], "reasons_truncated":0},
                {"name":"passed_walkforward", "count_in":200, "count_out":170, "rejected":30,
                 "top_reasons":[["insufficient_oos_trades",20],["negative_oos_return",10]], "reasons_truncated":2}
            ]
        }));
        assert!(text.contains("validation budget  in=10000  out=200  not advanced=9800"));
        assert!(text.contains("Not tested: validation candidates capped = 9800"));
        assert!(text.contains("Reason: insufficient oos trades = 20"));
        assert!(text.contains("Reason: negative oos return = 10"));
        assert!(text.contains("2 additional reason buckets were not saved"));
        assert!(!text.contains("9800 failed"));
    }

    #[test]
    fn missing_or_malformed_counts_are_unknown_not_fabricated_zeroes() {
        let text = rendered(serde_json::json!({
            "stages":[{"name":"passed_walkforward", "count_in":-1, "count_out":0,
                       "top_reasons":[["not_tested_cancelled", null]]}]
        }));
        assert!(text.contains("passed walkforward  in=?  out=0  not advanced=?"));
        assert!(text.contains("Not tested: not tested cancelled = ?"));
    }

    #[test]
    fn renderer_preserves_all_persisted_reason_buckets() {
        let reasons: Vec<_> = (0..32)
            .map(|index| serde_json::json!([format!("reason_{index}"), index]))
            .collect();
        let text = rendered(serde_json::json!({"stages":[{
            "name":"passed_quality", "count_in":100, "count_out":0, "rejected":100,
            "top_reasons":reasons
        }]}));
        for index in 0..32 {
            assert!(text.contains(&format!("Reason: reason {index} = {index}")));
        }
    }

    #[test]
    fn persisted_census_keeps_untested_separate_from_failed_and_final_capacity() {
        let text = rendered(serde_json::json!({"candidate_census":{
            "ga_returned_candidates":10000,
            "validation_candidate_limit":200,
            "validation_candidates_admitted":200,
            "validation_candidates_capped":9800,
            "quality_evaluated":180,
            "walkforward_tested":120,
            "walkforward_passed":100,
            "walkforward_failed":20,
            "walkforward_not_tested":80,
            "correlation_tested":100,
            "rejected_by_correlation":0,
            "portfolio_capacity_not_selected":96,
            "robustness_removed":2,
            "portfolio_selected":2
        }}));
        assert!(text.contains("GA pool: 10000 · Quality evaluated: 180"));
        assert!(text.contains("Validation limit (0=all): 200 · Admitted: 200"));
        assert!(text.contains("Not tested / budget cap: 9800"));
        assert!(text.contains("WF tested: 120 · Passed: 100 · Failed: 20"));
        assert!(text.contains("WF not tested: 80"));
        assert!(text.contains("Selected at reported stage: 2 · Not selected / capacity: 96"));
        assert!(text.contains("Robustness removed: 2"));
        assert!(!text.contains("Final selected"));
        assert!(
            !text.contains("CPCV tested"),
            "no invented gene/fold counts"
        );
        assert!(rendered(serde_json::json!({})).contains("Candidate coverage not recorded"));
        let earlier = rendered(serde_json::json!({"candidate_census":{
            "portfolio_selected":4
        }}));
        assert!(earlier.contains("Selected at reported stage: 4"));
        assert!(!earlier.contains("Robustness removed"));
        assert!(!earlier.contains("Final selected"));
    }

    #[test]
    fn zero_validation_limit_means_all_without_fabricating_completed_counts() {
        let text = rendered(serde_json::json!({"candidate_census":{
            "validation_candidate_limit":0,
            "quality_evaluated":0
        }}));
        assert!(text.contains("Validation limit (0=all): 0"));
        assert!(text.contains("Quality evaluated: 0"));
        assert!(!text.contains("Admitted:"));
        assert!(!text.contains("WF tested:"));
    }
}
