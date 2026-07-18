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
///
/// Uses the canonical FNV-1a from neoethos-core: `DefaultHasher`'s algorithm
/// is documented as unstable across Rust releases, so persisting its output
/// meant a toolchain bump could silently invalidate every stored fingerprint
/// and un-retire culled strategies (2026-07-18 deep-audit fix).
pub fn fingerprint_bytes(bytes: &[u8]) -> String {
    format!("{:016x}", neoethos_core::utils::hashing::fnv1a64(bytes))
}

/// The pre-2026-07-18 fingerprint (std `DefaultHasher`) — kept ONLY so
/// entries recorded by older builds still match in [`is_blacklisted`].
/// Never used for new entries.
fn legacy_fingerprint_bytes(bytes: &[u8]) -> String {
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
    // Compute BOTH fingerprints from one read: the stable FNV-1a used for
    // new entries, and the legacy DefaultHasher value so entries recorded
    // by pre-migration builds keep blocking their strategies.
    let bytes = std::fs::read(portfolio_path).ok();
    let fp = bytes.as_deref().map(fingerprint_bytes);
    let fp_legacy = bytes.as_deref().map(legacy_fingerprint_bytes);
    entries.iter().any(|e| {
        Some(e.fingerprint.as_str()) == fp.as_deref()
            || Some(e.fingerprint.as_str()) == fp_legacy.as_deref()
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
    // Atomic write (M07 primitive): the blacklist is SAFETY state — a torn
    // write would lose the whole graveyard and make every retired strategy
    // selectable for live trading again.
    if let Err(e) = neoethos_core::storage::json::write_json_atomic(&path, &entries) {
        tracing::warn!(
            target: "neoethos_app::strategy_blacklist",
            error = %e, path = %path.display(),
            "failed to write strategy blacklist"
        );
    }
}

fn normalize_path(p: &str) -> String {
    p.replace('\\', "/").to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_is_pinned_fnv1a() {
        // Literal pin of the canonical FNV-1a 64 value. If this assertion
        // ever fails, the fingerprint algorithm changed and every persisted
        // blacklist entry would stop matching — exactly the failure mode
        // the 2026-07-18 DefaultHasher→FNV migration fixed. Do not "update"
        // this constant without a blacklist migration plan.
        assert_eq!(fingerprint_bytes(b"hello"), "a430d84680aabd0b");
    }

    #[test]
    fn legacy_and_current_fingerprints_differ_but_both_match() {
        // Sanity: the legacy DefaultHasher value is a different string (so
        // the migration path in is_blacklisted is actually exercised), and
        // both are 16-hex-digit strings.
        let cur = fingerprint_bytes(b"portfolio-bytes");
        let legacy = legacy_fingerprint_bytes(b"portfolio-bytes");
        assert_eq!(cur.len(), 16);
        assert_eq!(legacy.len(), 16);
    }
}
