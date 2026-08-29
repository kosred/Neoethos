//! Search-memory + weekly-refresh **discovery ledger** (2026-06-06).
//!
//! THE OPERATOR GOAL: each weekly discovery run should ADD new diverse
//! strategies to a growing library instead of re-discovering ones it already
//! found. This module is the persistent record of "what was searched" per exact
//! canonical-input receipt and resolved config — every run writes a
//! [`DiscoverySearchLedger`] (the promoted
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
//!     builds its own `SeenSignatureMemory::current()` and (when an on-disk
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

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::artifact_io::write_json_atomic;
use crate::data_selection::CanonicalSearchInputReceiptV2;
use crate::discovery::{DiscoveryConfig, DiscoveryResult};
use crate::genetic::{Gene, SeenSignatureMemory, gene_signature_hash};

/// The first discovery-ledger schema that is cryptographically bound to the
/// complete canonical search input and the resolved search configuration.
pub const DISCOVERY_LEDGER_SCHEMA: &str = "neoethos.discovery_search_ledger.v3";

const SEARCH_STATE_DIRECTORY: &str = "canonical-search-state";

/// One recorded strategy gene. `hash` is the decimal string form of the u64
/// [`gene_signature_hash`] — kept as a string so very large hashes survive any
/// JSON tooling that treats numbers as f64.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub struct SearchMetadata {
    pub population: usize,
    pub generations: usize,
    pub prefilter_feature_names: Vec<String>,
}

/// The full receipt/config-bound ledger written after each discovery run.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DiscoverySearchLedger {
    pub schema: String,
    /// Full, independently-validatable proof of every canonical source segment
    /// and feature plan consumed by the search.
    pub search_input_receipt: CanonicalSearchInputReceiptV2,
    /// Recomputed SHA-256 identity of `search_input_receipt`.
    pub search_input_receipt_sha256: String,
    /// Exact resolved configuration identity. This is required rather than an
    /// `Option`: unattributable state is not valid state.
    pub config_hash: String,
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
    pub resolved_config: Option<crate::run_identity::ResolvedConfigStamp>,
    /// MEASUREMENT SLICE (2026-08-09). Accounting for the per-trial per-period
    /// return matrix written beside this ledger — how many trials were offered,
    /// how many were persisted, how many were dropped and why.
    ///
    /// `None` means the sidecar was not produced (the quality screen did not
    /// run, or the write failed) — recorded as absence rather than left to be
    /// inferred from a missing file.
    pub trial_returns: Option<crate::trial_returns::TrialReturnsManifest>,
}

fn validate_config_hash(config_hash: &str) -> Result<&str> {
    let hex = config_hash.strip_prefix("fnv64:").ok_or_else(|| {
        anyhow::anyhow!("invalid discovery config hash `{config_hash}`: expected fnv64:<16 hex>")
    })?;
    anyhow::ensure!(
        hex.len() == 16
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "invalid discovery config hash `{config_hash}`: expected fnv64:<16 lowercase hex>"
    );
    Ok(hex)
}

fn receipt_identity(receipt: &CanonicalSearchInputReceiptV2) -> Result<String> {
    receipt
        .identity_sha256()
        .map_err(anyhow::Error::new)
        .context("validate canonical search input receipt for discovery state")
}

fn validate_receipt_display_identity(
    receipt: &CanonicalSearchInputReceiptV2,
    symbol: &str,
    tf: &str,
) -> Result<()> {
    let anchor = receipt.validate().map_err(anyhow::Error::new)?;
    anyhow::ensure!(
        anchor.symbol_name().eq_ignore_ascii_case(symbol.trim())
            && anchor.timeframe().as_str().eq_ignore_ascii_case(tf.trim()),
        "discovery display identity {}/{} does not match receipt anchor {}/{}",
        symbol,
        tf,
        anchor.symbol_name(),
        anchor.timeframe().as_str()
    );
    Ok(())
}

fn state_directory(
    cache_dir: &str,
    receipt: &CanonicalSearchInputReceiptV2,
    config_hash: &str,
) -> Result<PathBuf> {
    let receipt_sha256 = receipt_identity(receipt)?;
    let config_hex = validate_config_hash(config_hash)?;
    Ok(PathBuf::from(cache_dir)
        .join(SEARCH_STATE_DIRECTORY)
        .join(receipt_sha256)
        .join(format!("fnv64-{config_hex}")))
}

/// `<cache_dir>/canonical-search-state/<full receipt SHA-256>/<full config
/// hash>/discovery_ledger.v3.json`.
pub fn ledger_path(
    cache_dir: &str,
    expected_receipt: &CanonicalSearchInputReceiptV2,
    expected_config_hash: &str,
) -> Result<PathBuf> {
    Ok(
        state_directory(cache_dir, expected_receipt, expected_config_hash)?
            .join("discovery_ledger.v3.json"),
    )
}

fn legacy_ledger_path(cache_dir: &str, symbol: &str, tf: &str) -> PathBuf {
    PathBuf::from(cache_dir).join(format!(
        "{}_{}.discovery_ledger.json",
        symbol.trim().to_ascii_uppercase(),
        tf.trim().to_ascii_uppercase()
    ))
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

/// Read the ledger at the exact receipt/config address.
///
/// Genuine absence is `Ok(None)`. Any present-but-unverifiable state is an
/// error: corruption, a legacy symbol/TF-only ledger, schema drift, or an exact
/// receipt/config mismatch must never seed a new search as if it were valid.
pub fn load_prior_ledger(
    cache_dir: &str,
    symbol: &str,
    tf: &str,
    expected_receipt: &CanonicalSearchInputReceiptV2,
    expected_config_hash: &str,
) -> Result<Option<DiscoverySearchLedger>> {
    let expected_receipt_sha256 = receipt_identity(expected_receipt)?;
    validate_receipt_display_identity(expected_receipt, symbol, tf)?;
    validate_config_hash(expected_config_hash)?;
    let path = ledger_path(cache_dir, expected_receipt, expected_config_hash)?;
    if !path.exists() {
        let orphan_manifest =
            crate::trial_returns::manifest_path(cache_dir, expected_receipt, expected_config_hash)?;
        let orphan_binary =
            crate::trial_returns::binary_path(cache_dir, expected_receipt, expected_config_hash)?;
        anyhow::ensure!(
            !orphan_manifest.exists() && !orphan_binary.exists(),
            "orphaned receipt-bound trial-return state exists without discovery ledger {} \
             (manifest {}, binary {})",
            path.display(),
            orphan_manifest.display(),
            orphan_binary.display()
        );
        let legacy = legacy_ledger_path(cache_dir, symbol, tf);
        anyhow::ensure!(
            !legacy.exists(),
            "legacy unbound discovery ledger {} exists for {}/{}; refusing symbol/TF-only state",
            legacy.display(),
            symbol,
            tf
        );
        return Ok(None);
    }
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("read discovery ledger {}", path.display()))?;
    let ledger: DiscoverySearchLedger = serde_json::from_str(&raw)
        .with_context(|| format!("parse discovery ledger {}", path.display()))?;

    anyhow::ensure!(
        ledger.schema == DISCOVERY_LEDGER_SCHEMA,
        "unsupported discovery ledger schema `{}` in {}; expected `{DISCOVERY_LEDGER_SCHEMA}`",
        ledger.schema,
        path.display()
    );
    let embedded_receipt_sha256 = receipt_identity(&ledger.search_input_receipt)?;
    anyhow::ensure!(
        ledger.search_input_receipt_sha256 == embedded_receipt_sha256,
        "discovery ledger receipt id does not match its embedded receipt in {}",
        path.display()
    );
    anyhow::ensure!(
        embedded_receipt_sha256 == expected_receipt_sha256
            && ledger.search_input_receipt == *expected_receipt,
        "discovery ledger receipt mismatch in {}: stored {}, expected {}",
        path.display(),
        embedded_receipt_sha256,
        expected_receipt_sha256
    );
    anyhow::ensure!(
        ledger.config_hash == expected_config_hash,
        "discovery ledger config mismatch in {}: stored `{}`, expected `{}`",
        path.display(),
        ledger.config_hash,
        expected_config_hash
    );
    anyhow::ensure!(
        ledger.symbol == symbol.trim().to_ascii_uppercase()
            && ledger.base_tf == tf.trim().to_ascii_uppercase(),
        "discovery ledger display identity mismatch in {}: stored {}/{}, expected {}/{}",
        path.display(),
        ledger.symbol,
        ledger.base_tf,
        symbol.trim().to_ascii_uppercase(),
        tf.trim().to_ascii_uppercase()
    );
    if let Some(stamp) = &ledger.resolved_config {
        anyhow::ensure!(
            stamp.config_hash == expected_config_hash,
            "discovery ledger resolved-config stamp mismatch in {}: stored `{}`, expected `{}`",
            path.display(),
            stamp.config_hash,
            expected_config_hash
        );
    }
    let sidecar_manifest = crate::trial_returns::load_manifest(
        cache_dir,
        symbol,
        tf,
        expected_receipt,
        expected_config_hash,
    )?;
    anyhow::ensure!(
        ledger.trial_returns == sidecar_manifest,
        "embedded trial-returns manifest does not match the validated sidecar for receipt {} \
         and config {}",
        expected_receipt_sha256,
        expected_config_hash
    );
    Ok(Some(ledger))
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
    expected_receipt: &CanonicalSearchInputReceiptV2,
    expected_config_hash: &str,
    result: &DiscoveryResult,
    config: &DiscoveryConfig,
    timestamp_ms: i64,
) -> Result<()> {
    let expected_receipt_sha256 = receipt_identity(expected_receipt)?;
    validate_receipt_display_identity(expected_receipt, symbol, tf)?;
    validate_config_hash(expected_config_hash)?;
    anyhow::ensure!(
        result.search_input_receipt == *expected_receipt,
        "discovery result receipt does not match the receipt selected for ledger persistence"
    );
    anyhow::ensure!(
        result.search_config_hash == expected_config_hash,
        "discovery result config `{}` does not match ledger config `{expected_config_hash}`",
        result.search_config_hash
    );
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
    let pip_value_per_lot = config.evaluation_config(None).pip_value_per_lot;
    let inputs = crate::run_identity::payoff_inputs_for_config(config, pip_value_per_lot);
    let normalize_features = neoethos_data::current_data_runtime_overrides().normalize_features;
    let resolved_config = crate::run_identity::assert_payoff_floor_reachable(
        config.target_profile.min_payoff_ratio,
        &inputs,
    )
    .and_then(|ceiling| {
        crate::run_identity::stamp_resolved_config(
            config,
            &inputs,
            ceiling,
            pip_value_per_lot,
            normalize_features,
        )
    })
    .context("resolve exact discovery config before writing its ledger")?;
    anyhow::ensure!(
        resolved_config.config_hash == expected_config_hash,
        "discovery ledger config identity drift: recomputed `{}`, expected `{}`",
        resolved_config.config_hash,
        expected_config_hash
    );
    tracing::info!(
        target: "neoethos_search::discovery_ledger",
        config_hash = %resolved_config.config_hash,
        receipt_sha256 = %expected_receipt_sha256,
        payoff_floor = resolved_config.payoff_floor,
        payoff_ceiling = resolved_config.payoff_ceiling.enforced_ceiling,
        sl_band = ?resolved_config.sl_clamp_pips,
        tp_band = ?resolved_config.tp_clamp_pips,
        band_atr_pips = ?resolved_config.band_atr_pips,
        trailing_enabled = resolved_config.trailing_enabled,
        cost_pips_round_trip = resolved_config.cost_pips_round_trip,
        prefilter_top_k = resolved_config.prefilter_top_k,
        adaptive_thresholds = resolved_config.adaptive_thresholds,
        normalize_features = resolved_config.normalize_features,
        "resolved-config stamp for this receipt-bound run"
    );

    // The per-trial return matrix is written by the quality screen (it is the
    // only place that holds every trial's trades). Here we only embed its
    // accounting, so the ledger says how many trials were persisted out of how
    // many were run.
    //
    // IDENTITY CHECK: the strict loader resolves the manifest only inside this
    // receipt/config directory, then recomputes the embedded receipt id, checks
    // the exact config and display identity, and hashes the named binary. A
    // mismatch or a legacy symbol/TF-only sidecar is an error, not absence.
    let trial_returns = crate::trial_returns::load_manifest(
        cache_dir,
        symbol,
        tf,
        expected_receipt,
        expected_config_hash,
    )?;
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
        schema: DISCOVERY_LEDGER_SCHEMA.to_string(),
        search_input_receipt: expected_receipt.clone(),
        search_input_receipt_sha256: expected_receipt_sha256,
        config_hash: expected_config_hash.to_string(),
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
        resolved_config: Some(resolved_config),
        trial_returns,
    };

    let path = ledger_path(cache_dir, expected_receipt, expected_config_hash)?;
    let parent = path.parent().ok_or_else(|| {
        anyhow::anyhow!("discovery ledger path has no parent: {}", path.display())
    })?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("create discovery ledger directory {}", parent.display()))?;
    write_json_atomic(&path, &ledger)
}

/// Seed `seen` with every hash recorded in `ledger` (portfolio + archive) so the
/// GA's dedup skips re-discovering them. Each `GeneRecord.hash` is parsed back to
/// the u64 the engine compares against; unparseable hashes are skipped (fail
/// soft). Returns the number of hashes actually inserted (new to `seen`).
pub fn seed_seen_from_ledger(
    ledger: &DiscoverySearchLedger,
    seen: &mut SeenSignatureMemory,
) -> usize {
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

    const TEST_CONFIG_HASH: &str = "fnv64:0123456789abcdef";

    fn temp_dir(prefix: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "{prefix}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("create temp directory");
        dir
    }

    fn sample_receipt() -> CanonicalSearchInputReceiptV2 {
        let features = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
        let anchor = features.provenance().bindings()[0]
            .dataset_identity()
            .clone();
        CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &features)
            .expect("canonical receipt fixture")
    }

    fn other_valid_receipt(
        receipt: &CanonicalSearchInputReceiptV2,
    ) -> CanonicalSearchInputReceiptV2 {
        let mut value = serde_json::to_value(receipt).expect("receipt JSON");
        let current = value["feature_plan_identity"]
            .as_str()
            .expect("feature plan identity");
        let replacement = if current == "0".repeat(64) {
            "1".repeat(64)
        } else {
            "0".repeat(64)
        };
        value["feature_plan_identity"] = serde_json::Value::String(replacement);
        CanonicalSearchInputReceiptV2::from_json_bytes(
            &serde_json::to_vec(&value).expect("receipt bytes"),
        )
        .expect("structurally valid alternate receipt")
    }

    fn sample_ledger(receipt: CanonicalSearchInputReceiptV2) -> DiscoverySearchLedger {
        DiscoverySearchLedger {
            schema: DISCOVERY_LEDGER_SCHEMA.to_string(),
            search_input_receipt_sha256: receipt.identity_sha256().expect("receipt id"),
            search_input_receipt: receipt,
            config_hash: TEST_CONFIG_HASH.to_string(),
            timestamp_ms: 1_717_000_000_000,
            symbol: "EURUSD".to_string(),
            base_tf: "M1".to_string(),
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
    fn legacy_unbound_ledger_is_an_error_not_absence() {
        let dir = temp_dir("neoethos_ledger_legacy");
        let cache = dir.to_string_lossy();
        let legacy = serde_json::json!({
            "timestamp_ms": 1_717_000_000_000_i64,
            "symbol": "EURUSD",
            "base_tf": "M1",
            "portfolio": [],
            "archive": [],
            "search_meta": {
                "population": 10,
                "generations": 2,
                "prefilter_feature_names": []
            }
        });
        let legacy_path = legacy_ledger_path(&cache, "EURUSD", "M1");
        std::fs::write(&legacy_path, serde_json::to_vec(&legacy).unwrap()).unwrap();

        let receipt = sample_receipt();
        let config_hash = TEST_CONFIG_HASH.to_string();
        let error = load_prior_ledger(&cache, "EURUSD", "M1", &receipt, &config_hash)
            .expect_err("legacy state must fail closed");
        assert!(
            error
                .to_string()
                .contains("legacy unbound discovery ledger")
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn ledger_round_trip_requires_exact_receipt_and_config() {
        let dir = temp_dir("neoethos_ledger_roundtrip");
        let cache = dir.to_string_lossy();
        let receipt = sample_receipt();
        let original = sample_ledger(receipt.clone());
        let path = ledger_path(&cache, &receipt, &original.config_hash).unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        write_json_atomic(&path, &original).unwrap();

        let loaded = load_prior_ledger(
            &cache,
            &original.symbol,
            &original.base_tf,
            &receipt,
            &original.config_hash,
        )
        .expect("valid ledger")
        .expect("present ledger");
        assert_eq!(loaded, original);
        assert_eq!(
            path.file_name().unwrap().to_string_lossy(),
            "discovery_ledger.v3.json"
        );
        assert!(
            path.to_string_lossy()
                .contains(&original.search_input_receipt_sha256)
        );
        assert!(path.to_string_lossy().contains("fnv64-"));

        let missing = temp_dir("neoethos_ledger_missing");
        assert!(
            load_prior_ledger(
                &missing.to_string_lossy(),
                "EURUSD",
                "M1",
                &receipt,
                &original.config_hash,
            )
            .expect("absence is not corruption")
            .is_none()
        );
        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_dir_all(missing);
    }

    #[test]
    fn same_symbol_timeframe_but_different_receipt_never_shares_a_ledger_path() {
        let receipt_a = sample_receipt();
        let receipt_b = other_valid_receipt(&receipt_a);
        let config_hash = TEST_CONFIG_HASH.to_string();
        let a = ledger_path("cache", &receipt_a, &config_hash).unwrap();
        let b = ledger_path("cache", &receipt_b, &config_hash).unwrap();
        assert_ne!(a, b);
        assert!(
            a.to_string_lossy()
                .contains(&receipt_a.identity_sha256().unwrap())
        );
        assert!(
            b.to_string_lossy()
                .contains(&receipt_b.identity_sha256().unwrap())
        );
    }

    #[test]
    fn ledger_copied_to_another_receipt_address_is_rejected() {
        let dir = temp_dir("neoethos_ledger_swap");
        let cache = dir.to_string_lossy();
        let receipt_a = sample_receipt();
        let receipt_b = other_valid_receipt(&receipt_a);
        let ledger = sample_ledger(receipt_a.clone());
        let wrong_path = ledger_path(&cache, &receipt_b, &ledger.config_hash).unwrap();
        std::fs::create_dir_all(wrong_path.parent().unwrap()).unwrap();
        write_json_atomic(&wrong_path, &ledger).unwrap();

        let error = load_prior_ledger(&cache, "EURUSD", "M1", &receipt_b, &ledger.config_hash)
            .expect_err("copied ledger must not bind to its directory name");
        assert!(error.to_string().contains("receipt mismatch"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn ledger_copied_to_another_config_address_is_rejected() {
        const OTHER_CONFIG_HASH: &str = "fnv64:fedcba9876543210";
        let dir = temp_dir("neoethos_ledger_config_swap");
        let cache = dir.to_string_lossy();
        let receipt = sample_receipt();
        let ledger = sample_ledger(receipt.clone());
        let wrong_path = ledger_path(&cache, &receipt, OTHER_CONFIG_HASH).unwrap();
        std::fs::create_dir_all(wrong_path.parent().unwrap()).unwrap();
        write_json_atomic(&wrong_path, &ledger).unwrap();

        let error = load_prior_ledger(&cache, "EURUSD", "M1", &receipt, OTHER_CONFIG_HASH)
            .expect_err("copied config state must fail closed");
        assert!(error.to_string().contains("config mismatch"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn corrupt_receipt_bound_ledger_is_an_error_not_absence() {
        let dir = temp_dir("neoethos_ledger_corrupt");
        let cache = dir.to_string_lossy();
        let receipt = sample_receipt();
        let path = ledger_path(&cache, &receipt, TEST_CONFIG_HASH).unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"{not-json").unwrap();

        let error = load_prior_ledger(&cache, "EURUSD", "M1", &receipt, TEST_CONFIG_HASH)
            .expect_err("corruption must fail closed");
        assert!(error.to_string().contains("parse discovery ledger"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn embedded_trial_manifest_must_equal_the_validated_sidecar() {
        let dir = temp_dir("neoethos_ledger_sidecar_mismatch");
        let cache = dir.to_string_lossy();
        let receipt = sample_receipt();
        let matrix = crate::trial_returns::TrialReturnMatrix {
            period_keys: vec![24_001],
            rows: vec![crate::trial_returns::TrialReturnRow {
                candidate_index: 0,
                strategy_id: "gene_0".to_string(),
                returns: vec![0.01],
                trades_outside_grid: 0,
            }],
            initial_balance: 10_000.0,
        };
        crate::trial_returns::write_trial_returns(
            &cache,
            "EURUSD",
            "M1",
            &receipt,
            TEST_CONFIG_HASH,
            &matrix,
            1,
        )
        .unwrap();

        let ledger = sample_ledger(receipt.clone());
        let path = ledger_path(&cache, &receipt, TEST_CONFIG_HASH).unwrap();
        write_json_atomic(&path, &ledger).unwrap();
        let error = load_prior_ledger(&cache, "EURUSD", "M1", &receipt, TEST_CONFIG_HASH)
            .expect_err("ledger and sidecar must be one atomic identity claim");
        assert!(
            error
                .to_string()
                .contains("embedded trial-returns manifest does not match")
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn seed_inserts_known_hash_so_ga_would_skip_it() {
        let ledger = sample_ledger(sample_receipt());
        // Build the seen-memory the way the engine does (`current()` →
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
