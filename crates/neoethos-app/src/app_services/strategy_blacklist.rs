//! Permanent strategy blacklist — the auto-cull "graveyard".
//!
//! When a live engine's strategy loses too many trades in a row (operator
//! directive 2026-07-01), it is *retired*: recorded here so it can NEVER be
//! selected for live trading again, and filtered out of the discovery/portfolio
//! listings so it is not re-surfaced. **Nothing is deleted** — the strategy file
//! stays on disk; this is a record, not a removal (respects the "never delete
//! strategies/data" invariant).
//!
//! Identity = a content fingerprint of the portfolio file (stable for the same
//! file on disk, and identical for a byte-identical re-export of the same
//! strategy). We also keep the path for human readability + a fast path match.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// One retired strategy. Append-only; never removed automatically.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BlacklistEntry {
    /// Content fingerprint of the portfolio file (see [`fingerprint_bytes`]).
    pub fingerprint: String,
    /// The portfolio file path at retirement time (for display + a path match).
    pub portfolio_path: String,
    pub symbol: Option<String>,
    /// Why it was retired, e.g. "6 consecutive losing trades (demo/live)".
    pub reason: String,
    pub consecutive_losses: u32,
    pub net_pnl: f64,
    pub retired_at_unix_ms: i64,
}

/// Canonical on-disk path: `<data_dir>/strategy_blacklist.json`. Honors the
/// live `config.yaml` data_dir; `None` (skip) on any config failure so a
/// blacklist hiccup never breaks the trading loop.
pub fn blacklist_path() -> Option<PathBuf> {
    let cfg = crate::server::state::current_config_path();
    neoethos_core::Settings::from_yaml(&cfg)
        .ok()
        .map(|s| s.system.data_dir.join("strategy_blacklist.json"))
}

/// Stable content fingerprint of a portfolio file's bytes. The same file on
/// disk always maps to the same value, and a byte-identical re-export of the
/// same strategy maps to the same value (discovery's serializer is
/// deterministic) — so a re-discovered clone is caught too.
pub fn fingerprint_bytes(bytes: &[u8]) -> String {
    let mut h = DefaultHasher::new();
    bytes.hash(&mut h);
    format!("{:016x}", h.finish())
}

/// Fingerprint a portfolio file by path; `None` if unreadable.
pub fn fingerprint_file(path: impl AsRef<Path>) -> Option<String> {
    std::fs::read(path.as_ref()).ok().map(|b| fingerprint_bytes(&b))
}

/// Load all retired entries; empty on any failure (best-effort).
pub fn load() -> Vec<BlacklistEntry> {
    let Some(path) = blacklist_path() else { return Vec::new() };
    let Ok(raw) = std::fs::read_to_string(&path) else { return Vec::new() };
    serde_json::from_str(&raw).unwrap_or_default()
}

/// True if this portfolio (by content fingerprint OR its recorded path) is
/// retired. Used by Autopilot (block selection) + discovery listings (hide it).
pub fn is_blacklisted(portfolio_path: &str) -> bool {
    let entries = load();
    if entries.is_empty() {
        return false;
    }
    let norm = normalize_path(portfolio_path);
    let fp = fingerprint_file(portfolio_path);
    entries.iter().any(|e| {
        (fp.as_deref().is_some() && Some(e.fingerprint.as_str()) == fp.as_deref())
            || normalize_path(&e.portfolio_path) == norm
    })
}

/// Record a strategy as retired (idempotent on fingerprint). Best-effort:
/// logs + swallows I/O errors so culling never destabilizes the engine.
pub fn retire(entry: BlacklistEntry) {
    let Some(path) = blacklist_path() else { return };
    let mut entries = load();
    if entries.iter().any(|e| e.fingerprint == entry.fingerprint) {
        return; // already retired
    }
    entries.push(entry);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match serde_json::to_string_pretty(&entries) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, json) {
                tracing::warn!(
                    target: "neoethos_app::strategy_blacklist",
                    error = %e, path = %path.display(),
                    "failed to write strategy blacklist"
                );
            }
        }
        Err(e) => tracing::warn!(
            target: "neoethos_app::strategy_blacklist",
            error = %e, "failed to serialize strategy blacklist"
        ),
    }
}

fn normalize_path(p: &str) -> String {
    p.replace('\\', "/").to_ascii_lowercase()
}
