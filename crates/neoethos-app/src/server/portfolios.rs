//! `/portfolios/list` — discovered strategy portfolios the autopilot can run.
//! Scans the cache dir for `*live_portfolio.json` artifacts and surfaces each
//! with its absolute path + symbol/base-TF + size, so the user can pick an
//! EXISTING strategy to replay or trade live (with clear provenance).

use std::path::{Path, PathBuf};

use axum::Json;
use axum::extract::State;
use serde::Serialize;

use neoethos_core::Settings;

use super::state::AppApiState;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PortfolioEntry {
    pub path: String,
    pub file_name: String,
    pub symbol: Option<String>,
    pub base_tf: Option<String>,
    pub gene_count: Option<usize>,
    pub size_bytes: u64,
    pub modified_ms: Option<i64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PortfoliosDto {
    pub count: usize,
    pub portfolios: Vec<PortfolioEntry>,
}

fn find_artifacts(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    let mut visited = 0usize;
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for ent in rd.flatten() {
            visited += 1;
            if visited > 200_000 {
                break;
            }
            let p = ent.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.to_string_lossy().ends_with("live_portfolio.json") {
                out.push(p);
            }
        }
    }
    out
}

fn read_entry(p: &Path) -> PortfolioEntry {
    let meta = std::fs::metadata(p).ok();
    let size_bytes = meta.as_ref().map(|m| m.len()).unwrap_or(0);
    let modified_ms = meta
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as i64);

    let (mut symbol, mut base_tf, mut gene_count) = (None, None, None);
    if let Ok(txt) = std::fs::read_to_string(p) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) {
            symbol = v.get("symbol").and_then(|x| x.as_str()).map(String::from);
            base_tf = v
                .get("base_tf")
                .or_else(|| v.get("base_timeframe"))
                .and_then(|x| x.as_str())
                .map(String::from);
            gene_count = v
                .get("genes")
                .or_else(|| v.get("full_genes"))
                .and_then(|x| x.as_array())
                .map(|a| a.len());
        }
    }

    PortfolioEntry {
        path: std::fs::canonicalize(p)
            .map(|c| c.display().to_string().trim_start_matches(r"\\?\").to_string())
            .unwrap_or_else(|_| p.display().to_string()),
        file_name: p.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default(),
        symbol,
        base_tf,
        gene_count,
        size_bytes,
        modified_ms,
    }
}

pub async fn list(State(state): State<AppApiState>) -> Json<PortfoliosDto> {
    let cache_dir = Settings::from_yaml(state.config_path())
        .map(|s| s.system.cache_dir)
        .unwrap_or_else(|_| PathBuf::from("cache"));
    let mut portfolios: Vec<PortfolioEntry> = find_artifacts(&cache_dir).iter().map(|p| read_entry(p)).collect();
    portfolios.sort_by(|a, b| b.modified_ms.cmp(&a.modified_ms));
    Json(PortfoliosDto { count: portfolios.len(), portfolios })
}
