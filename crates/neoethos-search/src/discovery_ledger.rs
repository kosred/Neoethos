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
    /// MEASUREMENT SLICE (2026-08-09). The RESOLVED configuration this run
    /// searched under, plus a hash over it.
    ///
    /// Why it exists: a prior run could not be attributed to a config file
    /// after the fact. The repo `config.yaml` and the operator's store config
    /// disagreed on `prefilter_top_k` (240 vs 50) and on the payoff floor
    /// (2.0 vs 0.0), and no artifact said which had been in force — so "the run
    /// found nothing" could not be separated from "the run was configured to
    /// find nothing". Two ledgers with the same `config_hash` are the same
    /// experiment.
    ///
    /// `#[serde(default)]` so ledgers written before this field exists still
    /// load and still seed the seen-set — an old ledger must never break a run.
    #[serde(default)]
    pub resolved_config: Option<crate::run_identity::ResolvedConfigStamp>,
    /// MEASUREMENT SLICE (2026-08-09). Accounting for the per-trial per-period
    /// return matrix written beside this ledger — how many trials were offered,
    /// how many were persisted, how many were dropped and why.
    ///
    /// `None` means the sidecar was not produced (the quality screen did not
    /// run, or the write failed) — recorded as absence rather than left to be
    /// inferred from a missing file.
    #[serde(default)]
    pub trial_returns: Option<crate::trial_returns::TrialReturnsManifest>,
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

    // ── MEASUREMENT SLICE (2026-08-09): stamp the resolved configuration ──
    //
    // Read through the SAME accessors the search read: the ATR-scaled stop band
    // from `current_gene_stop_bounds()`, the trailing geometry from the
    // installed `ExitPolicyOverrides`, the cost model from
    // `DiscoveryConfig::evaluation_config`. A stamp that re-derived any of them
    // would record a configuration nobody ran.
    //
    // Non-fatal: a stamp failure must not lose the seen-set the ledger exists to
    // carry. It is recorded as `None` and logged, never silently omitted.
    let pip_value_per_lot = config.evaluation_config(None).pip_value_per_lot;
    let inputs = crate::run_identity::payoff_inputs_for_config(config, pip_value_per_lot);
    let normalize_features = neoethos_data::current_data_runtime_overrides().normalize_features;
    let resolved_config = match crate::run_identity::max_achievable_payoff(&inputs)
        .and_then(|ceiling| {
            crate::run_identity::stamp_resolved_config(
                config,
                &inputs,
                ceiling,
                pip_value_per_lot,
                normalize_features,
            )
        }) {
        Ok(stamp) => {
            tracing::info!(
                target: "neoethos_search::discovery_ledger",
                config_hash = %stamp.config_hash,
                payoff_floor = stamp.payoff_floor,
                payoff_ceiling = stamp.payoff_ceiling.enforced_ceiling,
                sl_band = ?stamp.sl_clamp_pips,
                tp_band = ?stamp.tp_clamp_pips,
                band_atr_pips = ?stamp.band_atr_pips,
                trailing_enabled = stamp.trailing_enabled,
                cost_pips_round_trip = stamp.cost_pips_round_trip,
                prefilter_top_k = stamp.prefilter_top_k,
                adaptive_thresholds = stamp.adaptive_thresholds,
                normalize_features = stamp.normalize_features,
                "resolved-config stamp for this run"
            );
            Some(stamp)
        }
        Err(err) => {
            tracing::warn!(
                target: "neoethos_search::discovery_ledger",
                error = %err,
                "could not stamp the resolved config — the ledger records its ABSENCE. \
                 A result that cannot name its configuration is not attributable."
            );
            None
        }
    };

    // The per-trial return matrix is written by the quality screen (it is the
    // only place that holds every trial's trades). Here we only embed its
    // accounting, so the ledger says how many trials were persisted out of how
    // many were run.
    //
    // IDENTITY CHECK (2026-08-09 review): the manifest is keyed only on
    // `{SYMBOL}_{TF}.trial_returns.json`, so a run that skipped the quality
    // screen, ran under a different configuration, or whose matrix write failed
    // (non-fatal by design) would otherwise silently attach the PREVIOUS run's
    // matrix to itself — and then this ledger would assert on that basis that
    // DSR and PBO are computable. A manifest whose `config_hash` disagrees with
    // this run's stamp is REFUSED and its absence recorded, which is the honest
    // state. A `None` hash (written by a caller that had no stamp) is refused
    // for the same reason: it cannot be attributed.
    let expected_hash = resolved_config.as_ref().map(|s| s.config_hash.as_str());
    let trial_returns = match crate::trial_returns::load_manifest(cache_dir, symbol, tf) {
        Some(m) if m.config_hash.as_deref() == expected_hash && expected_hash.is_some() => Some(m),
        Some(m) => {
            tracing::warn!(
                target: "neoethos_search::discovery_ledger",
                symbol = %symbol,
                tf = %tf,
                manifest_config_hash = ?m.config_hash,
                ledger_config_hash = ?expected_hash,
                manifest_timestamp_ms = m.timestamp_ms,
                ledger_timestamp_ms = timestamp_ms,
                "trial-returns manifest beside this ledger belongs to a DIFFERENT run — \
                 REFUSED rather than embedded. DSR and PBO are NOT computable for this run."
            );
            None
        }
        None => None,
    };
    if trial_returns.is_none() {
        tracing::warn!(
            target: "neoethos_search::discovery_ledger",
            symbol = %symbol,
            tf = %tf,
            "no usable trial-returns manifest beside this ledger — DSR and PBO are NOT \
             computable for this run. Expected when the quality screen did not run."
        );
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
        resolved_config,
        trial_returns,
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
            resolved_config: None,
            trial_returns: None,
        }
    }

    #[test]
    fn a_ledger_written_before_the_stamp_existed_still_loads() {
        // Old ledgers carry the seen-set that stops a weekly run re-discovering
        // what it already found. Adding measurement fields must never make one
        // unreadable — the `#[serde(default)]` on both new fields is what
        // guarantees it, and this is the test that holds it.
        let legacy = serde_json::json!({
            "timestamp_ms": 1_717_000_000_000_i64,
            "symbol": "EURUSD",
            "base_tf": "D1",
            "portfolio": [],
            "archive": [],
            "search_meta": {
                "population": 10,
                "generations": 2,
                "prefilter_feature_names": []
            }
        });
        let parsed: DiscoverySearchLedger =
            serde_json::from_value(legacy).expect("legacy ledger must still parse");
        assert!(parsed.resolved_config.is_none());
        assert!(parsed.trial_returns.is_none());
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
