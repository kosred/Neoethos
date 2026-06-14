//! `PortfolioRegistry` — the source of truth for "what to watch".
//!
//! Holds the active `(symbol, base_tf, higher_tfs, source, mode)` set. Phase 1
//! constructs it from an explicit list or a JSON manifest; Phase 2 adds the
//! hot-reloading scan of promoted `live_models/<symbol>/<tf>/` +
//! `promotion_summary.json` (so it refreshes when `strategy_lab::promote_if_gated`
//! writes a new promotion). The registry — NOT the watchlist — drives the loop.

use std::path::Path;

use crate::contracts::PortfolioEntry;

#[derive(Debug, Clone, Default)]
pub struct PortfolioRegistry {
    entries: Vec<PortfolioEntry>,
}

impl PortfolioRegistry {
    pub fn from_entries(entries: Vec<PortfolioEntry>) -> Self {
        Self { entries }
    }

    pub fn entries(&self) -> &[PortfolioEntry] {
        &self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The distinct symbols to subscribe live data for (Phase 2).
    pub fn symbols(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for e in &self.entries {
            if !out.iter().any(|s| s == &e.symbol) {
                out.push(e.symbol.clone());
            }
        }
        out
    }

    /// True if `(symbol, tf)` is the BASE timeframe of some entry — i.e. a bar
    /// close on it should trigger signal evaluation.
    pub fn is_base_tf(&self, symbol: &str, tf: &str) -> bool {
        self.entries
            .iter()
            .any(|e| e.symbol == symbol && e.base_tf.eq_ignore_ascii_case(tf))
    }

    /// The entry watching `(symbol, base_tf)`, if any.
    pub fn entry_for(&self, symbol: &str, base_tf: &str) -> Option<&PortfolioEntry> {
        self.entries
            .iter()
            .find(|e| e.symbol == symbol && e.base_tf.eq_ignore_ascii_case(base_tf))
    }

    /// Load a JSON manifest: a top-level array of `PortfolioEntry`. This is the
    /// Phase-1 wiring seam — a hand-authored or test manifest — before the
    /// promotion-artifact scan lands in Phase 2.
    pub fn load_manifest(path: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path).map_err(|e| {
            anyhow::anyhow!("portfolio manifest {} not readable: {e}", path.display())
        })?;
        let entries: Vec<PortfolioEntry> = serde_json::from_str(&raw).map_err(|e| {
            anyhow::anyhow!("portfolio manifest {} is not a valid PortfolioEntry array: {e}", path.display())
        })?;
        Ok(Self::from_entries(entries))
    }
}
