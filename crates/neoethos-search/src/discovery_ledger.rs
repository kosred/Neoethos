//! Search-memory + weekly-refresh **discovery ledger** (2026-06-06).
//!
//! THE OPERATOR GOAL: each weekly discovery run should ADD new diverse
//! strategies to a growing library instead of re-discovering ones it already
//! found. This module is the persistent record of "what was searched" per
//! symbol/TF — every run writes a [`DiscoverySearchLedger`] (the promoted
//! portfolio + the top archive genes, each with its canonical gene-signature
//! hash + fitness + indicator names + SMC flags); the NEXT run's start loads
//! the prior ledger and seeds the GA's seen-signature memory with those hashes
//! so the engine SKIPS re-evolving duplicates.
//!
//! ADDITIVE BY DESIGN. This module does NOT touch the GA core
//! (`genetic::search_engine` / `genetic::evolution_math` evolution loop /
//! `eval` / `scoring`). It reuses two existing seams from the engine:
//!   - [`crate::genetic::gene_signature_hash`] — the canonical FNV-1a genome
//!     hash over indices/weights/thresholds/SMC-flags/SL-TP. We MUST use the
//!     exact same function so a seeded hash matches what the GA produces for an
//!     equivalent gene (otherwise dedup silently fails).
//!   - [`crate::genetic::SeenSignatureMemory`] + its file persistence — the GA
//!     builds its own `SeenSignatureMemory::from_env()` and (when an on-disk
//!     `file_path` is configured via `models.seen_signature_runtime.file_path`)
//!     loads previously-persisted hashes from that file at construction. We seed
//!     into a `SeenSignatureMemory` and let that same file-persistence path
//!     carry the hashes to the engine. When no `file_path` is configured (the
//!     default, in-memory only), the seed step still runs but the engine's fresh
//!     in-memory set won't see the seeded hashes — set a file_path for true
//!     cross-run dedup.
//!
//! Purity: the (de)serialization helpers do NOT read the clock. The caller
//! computes `timestamp_ms` and passes it in, so the module is fully testable.

use std::path::PathBuf;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::artifact_io::write_json_atomic;
use crate::discovery::{DiscoveryConfig, DiscoveryResult};
use crate::genetic::{Gene, SeenSignatureMemory, gene_signature_hash};

/// One recorded strategy gene. `hash` is the decimal string form of the u64
/// [`gene_signature_hash`] — kept as a string so very large hashes survive any
/// JSON tooling that treats numbers as f64.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct GeneRecord {
    pub hash: String,
    pub fitness: f64,
    pub trades: f64,
    pub sharpe: f64,
    pub indicator_names: Vec<String>,
    /// Pipe-joined active SMC flags, e.g. `"OB|FVG|BOS"`. Empty when none active.
    pub smc_flags: String,
}

/// Bookkeeping about the search that produced this ledger (so a future run /
/// audit can tell whether the seen-set was built under comparable settings).
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct SearchMetadata {
    pub population: usize,
    pub generations: usize,
    pub prefilter_feature_names: Vec<String>,
}

/// The full per-symbol/TF ledger written after each discovery run.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct DiscoverySearchLedger {
    pub timestamp_ms: i64,
    pub symbol: String,
    pub base_tf: String,
    /// The promoted portfolio, as records.
    pub portfolio: Vec<GeneRecord>,
    /// Top archive (non-portfolio) genes, capped by `archive_top_n`.
    pub archive: Vec<GeneRecord>,
    pub search_meta: SearchMetadata,
}

/// `<cache_dir>/{SYMBOL}_{TF}.discovery_ledger.json`. Symbol + TF are
/// upper-cased so the path is stable regardless of how the caller cased them.
pub fn ledger_path(cache_dir: &str, symbol: &str, tf: &str) -> PathBuf {
    let mut p = PathBuf::from(cache_dir);
    p.push(format!(
        "{}_{}.discovery_ledger.json",
        symbol.trim().to_ascii_uppercase(),
        tf.trim().to_ascii_uppercase()
    ));
    p
}

/// Build the pipe-joined active-SMC-flag string for a gene, e.g. `"OB|FVG|BOS"`.
/// Order is fixed so equal flag-sets always produce equal strings.
fn smc_flags_string(gene: &Gene) -> String {
    let mut parts: Vec<&str> = Vec::new();
    if gene.use_ob {
        parts.push("OB");
    }
    if gene.use_fvg {
        parts.push("FVG");
    }
    if gene.use_liq_sweep {
        parts.push("LIQ_SWEEP");
    }
    if gene.mtf_confirmation {
        parts.push("MTF");
    }
    if gene.use_premium_discount {
        parts.push("PREMIUM_DISCOUNT");
    }
    if gene.use_inducement {
        parts.push("INDUCEMENT");
    }
    if gene.use_bos {
        parts.push("BOS");
    }
    if gene.use_choch {
        parts.push("CHOCH");
    }
    if gene.use_eqh {
        parts.push("EQH");
    }
    if gene.use_eql {
        parts.push("EQL");
    }
    if gene.use_displacement {
        parts.push("DISPLACEMENT");
    }
    parts.join("|")
}

/// Map a gene's `indices` to indicator names via `effective_feature_names` (the
/// post-prefilter column names the indices reference — exactly the mapping
/// `GeneExport` / the live-portfolio artifact use). Out-of-range indices are
/// skipped (the same defensive behavior as `build_portfolio_exports`).
fn indicator_names_for(gene: &Gene, effective_feature_names: &[String]) -> Vec<String> {
    let mut names = Vec::with_capacity(gene.indices.len());
    for idx in &gene.indices {
        if let Some(name) = effective_feature_names.get(*idx) {
            names.push(name.clone());
        }
    }
    names
}

/// Build a [`GeneRecord`] from a gene + the effective feature names. The hash is
/// computed with the SAME `gene_signature_hash` the GA uses, so a seeded hash
/// matches what the engine would produce for an equivalent gene.
fn gene_record(gene: &Gene, effective_feature_names: &[String]) -> GeneRecord {
    GeneRecord {
        hash: gene_signature_hash(gene).to_string(),
        fitness: gene.fitness,
        trades: gene.trades_count as f64,
        sharpe: gene.sharpe_ratio,
        indicator_names: indicator_names_for(gene, effective_feature_names),
        smc_flags: smc_flags_string(gene),
    }
}

/// Read + deserialize the prior ledger for `symbol`/`tf`. Returns `None` when
/// the file is absent or invalid (fail soft — a corrupt ledger must never abort
/// a discovery run; it just means we can't seed from it). Logs a warn on a
/// present-but-unreadable/unparseable file.
pub fn load_prior_ledger(
    cache_dir: &str,
    symbol: &str,
    tf: &str,
) -> Option<DiscoverySearchLedger> {
    let path = ledger_path(cache_dir, symbol, tf);
    if !path.exists() {
        return None;
    }
    let raw = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(err) => {
            tracing::warn!(
                target: "neoethos_search::discovery_ledger",
                path = %path.display(),
                error = %err,
                "prior discovery ledger present but unreadable; skipping seed"
            );
            return None;
        }
    };
    match serde_json::from_str::<DiscoverySearchLedger>(&raw) {
        Ok(ledger) => Some(ledger),
        Err(err) => {
            tracing::warn!(
                target: "neoethos_search::discovery_ledger",
                path = %path.display(),
                error = %err,
                "prior discovery ledger is not valid JSON for this schema; skipping seed"
            );
            None
        }
    }
}

/// Build the ledger for THIS run from the [`DiscoveryResult`] + config and write
/// it atomically. `timestamp_ms` is passed in (callers compute it via the same
/// clock they stamp other artifacts with) so the module stays pure/testable.
///
/// The portfolio records come from `result.portfolio`; the archive records come
/// from the top `config.discovery_ledger_archive_top_n` of `result.candidates`
/// (ranked by fitness, descending) that are NOT already in the portfolio (by
/// hash) — so the seen-set grows beyond just the promoted strategies.
pub fn save_discovery_ledger(
    cache_dir: &str,
    symbol: &str,
    tf: &str,
    result: &DiscoveryResult,
    config: &DiscoveryConfig,
    timestamp_ms: i64,
) -> Result<()> {
    let names = &result.effective_feature_names;

    let portfolio: Vec<GeneRecord> = result
        .portfolio
        .iter()
        .map(|g| gene_record(g, names))
        .collect();

    let portfolio_hashes: std::collections::HashSet<String> =
        portfolio.iter().map(|r| r.hash.clone()).collect();

    // Top-N archive genes by fitness, excluding anything already promoted.
    let archive_top_n = config.discovery_ledger_archive_top_n;
    let mut archive_sorted: Vec<&Gene> = result.candidates.iter().collect();
    archive_sorted.sort_by(|a, b| {
        b.fitness
            .partial_cmp(&a.fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut archive: Vec<GeneRecord> = Vec::new();
    for gene in archive_sorted {
        if archive.len() >= archive_top_n {
            break;
        }
        let rec = gene_record(gene, names);
        if portfolio_hashes.contains(&rec.hash) {
            continue;
        }
        archive.push(rec);
    }

    let ledger = DiscoverySearchLedger {
        timestamp_ms,
        symbol: symbol.trim().to_ascii_uppercase(),
        base_tf: tf.trim().to_ascii_uppercase(),
        portfolio,
        archive,
        search_meta: SearchMetadata {
            population: config.population,
            generations: config.generations,
            prefilter_feature_names: names.clone(),
        },
    };

    let path = ledger_path(cache_dir, symbol, tf);
    write_json_atomic(&path, &ledger)
}

/// Seed `seen` with every hash recorded in `ledger` (portfolio + archive) so the
/// GA's dedup skips re-discovering them. Each `GeneRecord.hash` is parsed back to
/// the u64 the engine compares against; unparseable hashes are skipped (fail
/// soft). Returns the number of hashes actually inserted (new to `seen`).
pub fn seed_seen_from_ledger(ledger: &DiscoverySearchLedger, seen: &mut SeenSignatureMemory) -> usize {
    let mut inserted = 0usize;
    for rec in ledger.portfolio.iter().chain(ledger.archive.iter()) {
        match rec.hash.parse::<u64>() {
            Ok(h) => {
                if seen.insert_hash(h) {
                    inserted += 1;
                }
            }
            Err(err) => {
                tracing::warn!(
                    target: "neoethos_search::discovery_ledger",
                    hash = %rec.hash,
                    error = %err,
                    "discovery-ledger record has an unparseable signature hash; skipping"
                );
            }
        }
    }
    inserted
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ledger() -> DiscoverySearchLedger {
        DiscoverySearchLedger {
            timestamp_ms: 1_717_000_000_000,
            symbol: "EURUSD".to_string(),
            base_tf: "D1".to_string(),
            portfolio: vec![GeneRecord {
                hash: "12345678901234567890".to_string(),
                fitness: 1.5,
                trades: 42.0,
                sharpe: 1.1,
                indicator_names: vec!["rsi_14".to_string(), "atr_20".to_string()],
                smc_flags: "OB|FVG".to_string(),
            }],
            archive: vec![GeneRecord {
                hash: "987654321".to_string(),
                fitness: 0.7,
                trades: 10.0,
                sharpe: 0.4,
                indicator_names: vec!["ema_50".to_string()],
                smc_flags: String::new(),
            }],
            search_meta: SearchMetadata {
                population: 1000,
                generations: 50,
                prefilter_feature_names: vec!["rsi_14".to_string(), "atr_20".to_string()],
            },
        }
    }

    #[test]
    fn ledger_round_trip_save_load_equal() {
        let dir = std::env::temp_dir().join(format!(
            "neoethos_ledger_rt_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let cache_dir = dir.to_string_lossy().to_string();

        let original = sample_ledger();
        let path = ledger_path(&cache_dir, &original.symbol, &original.base_tf);
        // Write directly (save_discovery_ledger needs a DiscoveryResult; the
        // round-trip we care about is the on-disk JSON shape, which both paths
        // share via write_json_atomic + serde).
        write_json_atomic(&path, &original).unwrap();

        let loaded =
            load_prior_ledger(&cache_dir, &original.symbol, &original.base_tf).expect("ledger");
        assert_eq!(loaded, original);

        // Path convention is exactly {SYMBOL}_{TF}.discovery_ledger.json.
        assert!(
            path.file_name().unwrap().to_string_lossy() == "EURUSD_D1.discovery_ledger.json"
        );

        // Absent ledger → None (fail soft).
        assert!(load_prior_ledger(&cache_dir, "NOPE", "M1").is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn seed_inserts_known_hash_so_ga_would_skip_it() {
        let ledger = sample_ledger();
        // Build the seen-memory the way the engine does (`from_env()` →
        // overrides default to max_entries = 3_000_000). A bare
        // `SeenSignatureMemory::default()` has max_entries = 0, whose eviction
        // loop would empty the set immediately — not how the GA constructs it.
        let mut seen = SeenSignatureMemory {
            max_entries: 3_000_000,
            ..Default::default()
        };
        let inserted = seed_seen_from_ledger(&ledger, &mut seen);
        // Both records have valid, distinct u64 hashes → 2 inserted.
        assert_eq!(inserted, 2);

        // The seen-memory now contains the portfolio gene's hash, so the GA's
        // `insert_gene` would report it as a duplicate (returns false on a
        // hash already present).
        let portfolio_hash: u64 = ledger.portfolio[0].hash.parse().unwrap();
        assert!(seen.all.contains(&portfolio_hash));
        let archive_hash: u64 = ledger.archive[0].hash.parse().unwrap();
        assert!(seen.all.contains(&archive_hash));

        // Re-seeding the same ledger inserts nothing new (idempotent dedup).
        assert_eq!(seed_seen_from_ledger(&ledger, &mut seen), 0);
    }
}
