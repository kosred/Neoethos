//! `/storage/paths` — every file/dir the app reads or writes, with the
//! resolved absolute path + size + last-modified + item count, so the user
//! can find and open anything (data, models, strategies, journal, logs,
//! config, credentials). Pairs with the desktop `open_folder` command.

use std::path::{Path, PathBuf};

use axum::Json;
use axum::extract::State;
use serde::Serialize;

use neoethos_core::Settings;

use super::state::AppApiState;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StorageEntry {
    pub key: String,
    pub label: String,
    pub path: String,
    pub exists: bool,
    pub is_dir: bool,
    pub size_bytes: u64,
    pub item_count: usize,
    pub last_modified_ms: Option<i64>,
    /// data | models | strategies | journal | logs | config | secret | cache
    pub kind: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StoragePathsDto {
    pub entries: Vec<StorageEntry>,
}

fn mtime_ms(meta: &std::fs::Metadata) -> Option<i64> {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as i64)
}

/// Recursive size + top-level item count + newest mtime. Bounded to avoid
/// pathological deep trees; symlinks are not followed.
fn stats(path: &Path) -> (bool, bool, u64, usize, Option<i64>) {
    let Ok(meta) = std::fs::metadata(path) else {
        return (false, false, 0, 0, None);
    };
    if meta.is_file() {
        return (true, false, meta.len(), 1, mtime_ms(&meta));
    }
    // directory
    let mut size = 0u64;
    let mut newest = mtime_ms(&meta);
    let mut top_count = 0usize;
    let mut stack: Vec<PathBuf> = vec![path.to_path_buf()];
    let mut visited = 0usize;
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for ent in rd.flatten() {
            visited += 1;
            if visited > 200_000 {
                break; // safety cap
            }
            if dir == path {
                top_count += 1;
            }
            let Ok(m) = ent.metadata() else { continue };
            if let Some(t) = mtime_ms(&m) {
                newest = Some(newest.map_or(t, |n| n.max(t)));
            }
            if m.is_dir() {
                stack.push(ent.path());
            } else {
                size += m.len();
            }
        }
    }
    (true, true, size, top_count, newest)
}

fn abs(p: &Path) -> String {
    std::fs::canonicalize(p)
        .map(|c| c.display().to_string().trim_start_matches(r"\\?\").to_string())
        .unwrap_or_else(|_| p.display().to_string())
}

fn entry(key: &str, label: &str, kind: &str, path: PathBuf) -> StorageEntry {
    let (exists, is_dir, size_bytes, item_count, last_modified_ms) = stats(&path);
    StorageEntry {
        key: key.to_string(),
        label: label.to_string(),
        path: abs(&path),
        exists,
        is_dir,
        size_bytes,
        item_count,
        last_modified_ms,
        kind: kind.to_string(),
    }
}

/// Count files matching a suffix anywhere under `root` (bounded).
fn count_suffix(root: &Path, suffix: &str) -> usize {
    let mut n = 0usize;
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
            } else if p.to_string_lossy().ends_with(suffix) {
                n += 1;
            }
        }
    }
    n
}

pub async fn paths(State(state): State<AppApiState>) -> Json<StoragePathsDto> {
    let settings = Settings::from_yaml(state.config_path()).ok();
    let data_dir = settings
        .as_ref()
        .map(|s| s.system.data_dir.clone())
        .unwrap_or_else(|| PathBuf::from("data"));
    let cache_dir = settings
        .as_ref()
        .map(|s| s.system.cache_dir.clone())
        .unwrap_or_else(|| PathBuf::from("cache"));
    let models_dir = PathBuf::from("models");

    let mut entries = vec![
        entry("config", "Engine config (config.yaml)", "config", state.config_path().to_path_buf()),
        entry("data", "Market data (Vortex)", "data", data_dir.clone()),
        entry("models", "Trained models", "models", models_dir),
        entry("cache", "Engine cache", "cache", cache_dir.clone()),
        entry("journal", "Trade journal", "journal", data_dir.join("journal")),
        entry("logs", "Logs", "logs", neoethos_core::logging::default_log_dir()),
    ];

    // Strategies = the discovered live-portfolio artifacts under cache/.
    let strat_count = count_suffix(&cache_dir, "live_portfolio.json")
        + count_suffix(&cache_dir, "model_targets.json");
    entries.push(StorageEntry {
        key: "strategies".to_string(),
        label: "Discovered strategies".to_string(),
        path: abs(&cache_dir),
        exists: cache_dir.exists(),
        is_dir: true,
        size_bytes: 0,
        item_count: strat_count,
        last_modified_ms: None,
        kind: "strategies".to_string(),
    });

    if let Ok(creds) = neoethos_core::broker_config::credentials_file_path() {
        entries.push(entry("credentials", "Broker credentials", "secret", creds));
    }

    Json(StoragePathsDto { entries })
}
