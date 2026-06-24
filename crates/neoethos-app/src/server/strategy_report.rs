//! `/strategy/list` + `/strategy/report` — honest per-strategy reporting built
//! from the stored discovery artifacts (trades + profile + quality), which the
//! engine writes but never aggregated into a monthly journal or surfaced with
//! its own validation verdict.
//!
//! For each discovered strategy we reconstruct the €1000 equity curve month by
//! month from the recorded per-trade returns, derive CAGR / max-drawdown / span,
//! attach the broker-mode (from the cache sub-dir), the engine's own validation
//! flags (cpcv / walkforward / completeness), and sanity flags (out-of-sample
//! failure, low sample size, impossible compounded return = a units bug).

use std::path::{Path, PathBuf};

use axum::Json;
use axum::extract::{Query, State};
use serde::{Deserialize, Serialize};

use neoethos_core::Settings;

use super::state::AppApiState;

const SEED: f64 = 1000.0;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MonthRow {
    pub month: String, // YYYY-MM
    pub balance: f64,
    pub return_pct: f64,
    pub trades: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StrategyEntry {
    pub mode: String, // risky | prop_firm
    pub dir: String,  // cache sub-dir name (provenance)
    pub symbol: String,
    pub timeframe: String,
    pub base: String, // SYMBOL_TF
    pub trades: usize,
    pub win_rate: Option<f64>,
    pub profit_factor: Option<f64>,
    pub sharpe: Option<f64>,
    pub cpcv_passed: Option<bool>,
    pub walkforward_passed: Option<bool>,
    pub validation_complete: Option<bool>,
    pub span_start: Option<String>,
    pub span_end: Option<String>,
    pub years: f64,
    pub cagr_pct: f64,
    pub final_from_1000: f64,
    pub max_dd_pct: f64,
    /// Honest red flags (out-of-sample failed, low sample, units anomaly, …).
    pub flags: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StrategyListDto {
    pub count: usize,
    pub strategies: Vec<StrategyEntry>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StrategyReportDto {
    #[serde(flatten)]
    pub head: StrategyEntry,
    pub monthly: Vec<MonthRow>,
    pub yearly: Vec<MonthRow>,
}

fn cache_dir(state: &AppApiState) -> PathBuf {
    Settings::from_yaml(state.config_path())
        .map(|s| s.system.cache_dir)
        .unwrap_or_else(|_| PathBuf::from("cache"))
}

fn auto_loop_dirs(cache: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(cache) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() && p.file_name().and_then(|n| n.to_str()).map(|n| n.starts_with("auto_loop")).unwrap_or(false) {
                out.push(p);
            }
        }
    }
    out
}

fn mode_of(dir: &Path) -> String {
    let n = dir.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if n.contains("propfirm") || n.contains("prop_firm") {
        "prop_firm".to_string()
    } else {
        "risky".to_string()
    }
}

fn read_json(p: &Path) -> Option<serde_json::Value> {
    std::fs::read_to_string(p).ok().and_then(|t| serde_json::from_str(&t).ok())
}

fn month_of(ms: i64) -> String {
    let secs = ms / 1000;
    let days = secs / 86400;
    // civil date from days since epoch (Howard Hinnant's algorithm)
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    let _ = d;
    format!("{:04}-{:02}", y, m)
}

/// Pick the gene with the most trades from a trades.json value.
fn best_gene(v: &serde_json::Value) -> Option<&Vec<serde_json::Value>> {
    let genes = v.as_array()?;
    genes
        .iter()
        .filter_map(|g| g.get("trades").and_then(|t| t.as_array()))
        .max_by_key(|t| t.len())
}

/// Build a strategy entry (+ optional monthly/yearly rows) from one base.
fn build(dir: &Path, base: &str, with_monthly: bool) -> Option<(StrategyEntry, Vec<MonthRow>, Vec<MonthRow>)> {
    let trades_v = read_json(&dir.join(format!("{base}.json.trades.json")))?;
    let gene = best_gene(&trades_v)?;
    // chronological (entry_time)
    let mut rows: Vec<(i64, i64, f64)> = gene
        .iter()
        .filter_map(|t| {
            Some((
                t.get("entry_time")?.as_i64()?,
                t.get("exit_time").and_then(|x| x.as_i64()).unwrap_or(0),
                t.get("pnl_pct").and_then(|x| x.as_f64()).unwrap_or(0.0),
            ))
        })
        .collect();
    rows.sort_by_key(|r| r.0);
    if rows.is_empty() {
        return None;
    }

    let (sym, tf) = base.split_once('_').unwrap_or((base, ""));
    let span_start = month_of(rows.first().unwrap().0);
    let span_end = month_of(rows.last().unwrap().1.max(rows.last().unwrap().0));
    let start_ms = rows.first().unwrap().0;
    let end_ms = rows.last().unwrap().1.max(rows.last().unwrap().0);
    let years = ((end_ms - start_ms) as f64 / 1000.0 / 86400.0 / 365.25).max(0.05);

    // monthly compounding + max drawdown
    let mut eq = SEED;
    let mut peak = SEED;
    let mut max_dd = 0.0f64;
    let mut monthly: Vec<MonthRow> = Vec::new();
    let mut cur_month = String::new();
    let mut month_open = SEED;
    let mut month_trades = 0usize;
    for (_e, x, pct) in &rows {
        eq *= 1.0 + pct;
        peak = peak.max(eq);
        if peak > 0.0 {
            max_dd = max_dd.max((peak - eq) / peak);
        }
        let m = month_of(*x);
        if m != cur_month {
            if !cur_month.is_empty() {
                monthly.push(MonthRow {
                    month: cur_month.clone(),
                    balance: round2(eq),
                    return_pct: round2((eq / month_open - 1.0) * 100.0),
                    trades: month_trades,
                });
            }
            cur_month = m;
            month_open = eq / (1.0 + pct); // open = balance before this month's first trade
            month_trades = 0;
        }
        month_trades += 1;
    }
    if !cur_month.is_empty() {
        monthly.push(MonthRow {
            month: cur_month.clone(),
            balance: round2(eq),
            return_pct: round2((eq / month_open - 1.0) * 100.0),
            trades: month_trades,
        });
    }

    let cagr_pct = ((eq / SEED).powf(1.0 / years) - 1.0) * 100.0;

    // profile (validation flags)
    let prof = read_json(&dir.join(format!("{base}.json.profile.json")));
    let pf = |k: &str| prof.as_ref().and_then(|p| p.get(k)).and_then(|v| v.as_bool());
    let cpcv = pf("cpcv_passed");
    let wf = pf("walkforward_passed");
    let complete = pf("validation_evidence_complete");

    // quality (headline metrics for the most-traded gene)
    let qual = read_json(&dir.join(format!("{base}.json.quality.json")));
    let qbest = qual.as_ref().and_then(|q| q.as_array()).and_then(|a| {
        a.iter().max_by(|x, y| {
            let tx = x.get("total_trades").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let ty = y.get("total_trades").and_then(|v| v.as_f64()).unwrap_or(0.0);
            tx.partial_cmp(&ty).unwrap_or(std::cmp::Ordering::Equal)
        })
    });
    let qf = |k: &str| qbest.and_then(|q| q.get(k)).and_then(|v| v.as_f64());

    // honest flags
    let mut flags = Vec::new();
    if wf == Some(false) {
        flags.push("out-of-sample (walkforward) FAILED — mostly in-sample".to_string());
    }
    if complete == Some(false) {
        flags.push("validation incomplete".to_string());
    }
    if rows.len() < 100 {
        flags.push(format!("low sample ({} trades) — overfit risk", rows.len()));
    }
    if cagr_pct.abs() > 1000.0 || !cagr_pct.is_finite() {
        flags.push("impossible compounded return — likely a pips↔profit units bug".to_string());
    }
    if (qf("sharpe_ratio").unwrap_or(0.0)) > 20.0 {
        flags.push("implausibly high Sharpe — overfit".to_string());
    }

    let head = StrategyEntry {
        mode: mode_of(dir),
        dir: dir.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string(),
        symbol: sym.to_string(),
        timeframe: tf.to_string(),
        base: base.to_string(),
        trades: rows.len(),
        win_rate: qf("win_rate"),
        profit_factor: qf("profit_factor"),
        sharpe: qf("sharpe_ratio"),
        cpcv_passed: cpcv,
        walkforward_passed: wf,
        validation_complete: complete,
        span_start: Some(span_start),
        span_end: Some(span_end),
        years: round2(years),
        cagr_pct: if cagr_pct.is_finite() { round2(cagr_pct) } else { 0.0 },
        final_from_1000: if eq.is_finite() { round2(eq) } else { 0.0 },
        max_dd_pct: round2(max_dd * 100.0),
        flags,
    };

    let yearly = if with_monthly {
        let mut yr: Vec<MonthRow> = Vec::new();
        let mut last_year = String::new();
        for m in &monthly {
            let y = m.month[..4].to_string();
            if y != last_year {
                yr.push(MonthRow { month: y.clone(), balance: m.balance, return_pct: 0.0, trades: 0 });
                last_year = y;
            } else if let Some(l) = yr.last_mut() {
                l.balance = m.balance;
            }
        }
        yr
    } else {
        Vec::new()
    };

    Some((head, if with_monthly { monthly } else { Vec::new() }, yearly))
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn list_bases(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            if let Some(n) = e.file_name().to_str() {
                if let Some(base) = n.strip_suffix(".json.trades.json") {
                    out.push(base.to_string());
                }
            }
        }
    }
    out.sort();
    out
}

pub async fn list(State(state): State<AppApiState>) -> Json<StrategyListDto> {
    let cache = cache_dir(&state);
    let mut strategies = Vec::new();
    for dir in auto_loop_dirs(&cache) {
        for base in list_bases(&dir) {
            if let Some((head, _, _)) = build(&dir, &base, false) {
                strategies.push(head);
            }
        }
    }
    strategies.sort_by(|a, b| b.trades.cmp(&a.trades));
    Json(StrategyListDto { count: strategies.len(), strategies })
}

#[derive(Debug, Deserialize)]
pub struct ReportQuery {
    pub dir: String,
    pub base: String,
}

pub async fn report(
    State(state): State<AppApiState>,
    Query(q): Query<ReportQuery>,
) -> Result<Json<StrategyReportDto>, axum::http::StatusCode> {
    let cache = cache_dir(&state);
    // sanitise: only allow auto_loop* dirs + a plain base name
    if !q.dir.starts_with("auto_loop") || q.base.contains('/') || q.base.contains('\\') {
        return Err(axum::http::StatusCode::BAD_REQUEST);
    }
    let dir = cache.join(&q.dir);
    let (head, monthly, yearly) = build(&dir, &q.base, true).ok_or(axum::http::StatusCode::NOT_FOUND)?;
    Ok(Json(StrategyReportDto { head, monthly, yearly }))
}
