//! Permanent strategy blacklist — the auto-cull "graveyard".
//!
//! When a live engine's strategy loses too many trades in a row (operator
//! directive 2026-07-01), it is *retired*: recorded here so it can NEVER be
//! selected for live trading again, and filtered out of the discovery/portfolio
//! listings so it is not re-surfaced. **Nothing is deleted** — the strategy file
//! stays on disk; this is a record, not a removal (respects the "never delete
//! strategies/data" invariant).
//!
//! Identity = a fingerprint of the strategy the portfolio file describes. We
//! also keep the path for human readability + a fast path match.
//!
//! # Why identity is the GENE and not the file — item #218, 2026-08-09
//!
//! It used to be the file's raw bytes, on the reasoning that "discovery's
//! serializer is deterministic, so a re-discovered clone is caught too". The
//! serializer is deterministic; the **content** is not. A
//! `neoethos_search::genetic::strategy_gene::Gene` carries the trading rule
//! (`indices`, `weights`, thresholds, the SMC flags, `tp_pips`/`sl_pips`,
//! `stop_vol_mult`) **and, in the same struct, that run's measurements**:
//! `fitness`, `sharpe_ratio`, `win_rate`, `max_drawdown`, `profit_factor`,
//! `expectancy`, `trades_count`, `generation`, `strategy_id`,
//! `slice_pass_rate`, `consistency`.
//!
//! So a fresh discovery run that rediscovers the *identical* trading rule
//! writes a byte-different artifact — a different `generation`, a different
//! `strategy_id`, a fitness that moved in the fourth decimal — and the strategy
//! auto-culled after six consecutive real losses came back with a clean record.
//! Nothing in the loop noticed, because the loop was comparing file bytes.
//!
//! [`gene_fingerprint_bytes`] hashes the rule and drops the measurements. It
//! also resolves each gene's positional `indices` through
//! `effective_feature_names` first, because an index is meaningless without the
//! column list it indexes into — the same `[3, 17, 42]` is a different strategy
//! under a different prefilter.
//!
//! **The old fingerprints are never removed.** [`is_blacklisted`] matches on
//! gene identity, current file bytes, the pre-2026-07-18 `DefaultHasher` bytes,
//! and the recorded path — four arms, and it logs which one caught.
//!
//! # Where the identity itself now lives — item #219, 2026-08-10
//!
//! In [`neoethos_core::strategy_identity`], not here. This module is in
//! `neoethos-app`, the top of the dependency graph, so while the definition sat
//! here `neoethos-search` could not see it: retirement blocked SELECTION but
//! discovery was free to re-derive the culled rule on the very run the
//! retirement queued. `neoethos-search` now filters the live portfolio against
//! the same identity (`live_portfolio::from_discovery`), reading this same
//! blacklist file. The copy that used to live here is DELETED, not wrapped —
//! two implementations of "same strategy" would reopen the hole the day they
//! drifted.

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

// GENE IDENTITY MOVED 2026-08-10 (#219) to
// `neoethos_core::strategy_identity`, and DELETED here.
//
// `GENE_FINGERPRINT_PREFIX`, `GENE_MEASUREMENT_FIELDS`,
// `PORTFOLIO_NON_IDENTITY_FIELDS`, `write_canonical`, `resolve_indices`,
// `gene_rule_identity` and `gene_fingerprint_bytes` all lived in THIS file,
// which sits at the top of the dependency graph. So the identity that decides
// "this is the retired strategy" was unreachable from `neoethos-search`: the
// GA could re-derive a culled rule on the very run the retirement queued, and
// nothing in the loop noticed. The definition now lives in `neoethos-core`,
// which both crates depend on, and there is exactly ONE of it.
//
// `gene_fingerprint_bytes` keeps its name and behaviour for every caller here
// (it is what `retire` stores and `is_blacklisted` matches on) and is a
// one-line delegation.

/// See [`neoethos_core::strategy_identity::GENE_FINGERPRINT_PREFIX`].
pub use neoethos_core::strategy_identity::GENE_FINGERPRINT_PREFIX;

/// Fingerprint the STRATEGY a live-portfolio artifact describes, independent of
/// which run produced the file.
///
/// `None` when the bytes are not a live-portfolio artifact (unparseable, or no
/// `genes` array) — the caller then falls back to the file-bytes fingerprint,
/// so an unrecognised shape degrades to the old behaviour instead of silently
/// producing no identity at all.
pub fn gene_fingerprint_bytes(bytes: &[u8]) -> Option<String> {
    neoethos_core::strategy_identity::portfolio_gene_fingerprint(bytes)
}

/// Fingerprint a portfolio file by path; `None` if unreadable.
///
/// Prefers the GENE identity (#218) and falls back to the file-bytes
/// fingerprint for anything that is not a recognisable live-portfolio artifact,
/// so nothing that used to get an identity stops getting one.
pub fn fingerprint_file(path: impl AsRef<Path>) -> Option<String> {
    let bytes = std::fs::read(path.as_ref()).ok()?;
    Some(gene_fingerprint_bytes(&bytes).unwrap_or_else(|| fingerprint_bytes(&bytes)))
}

/// Load all retired entries; empty on any failure (best-effort).
pub fn load() -> Vec<BlacklistEntry> {
    let Some(path) = blacklist_path() else { return Vec::new() };
    let Ok(raw) = std::fs::read_to_string(&path) else { return Vec::new() };
    serde_json::from_str(&raw).unwrap_or_default()
}

/// True if this portfolio is retired, by GENE identity, by file content, or by
/// its recorded path. Used by Autopilot (block selection) + discovery listings
/// (hide it).
///
/// Four arms, checked in that order, and **none of the three old ones was
/// removed** — entries written by any previous build keep blocking their
/// strategies. A hit is logged with the arm that caught it, so "why is this
/// strategy not selectable" is answerable from the log rather than by
/// reasoning about hashes.
pub fn is_blacklisted(portfolio_path: &str) -> bool {
    let entries = load();
    if entries.is_empty() {
        return false;
    }
    let norm = normalize_path(portfolio_path);
    // ONE read, four fingerprints: the gene identity used for new entries
    // (#218), the stable FNV-1a file hash, and the legacy DefaultHasher value
    // so entries recorded by pre-2026-07-18 builds still match.
    let bytes = std::fs::read(portfolio_path).ok();
    let fp_gene = bytes.as_deref().and_then(gene_fingerprint_bytes);
    let fp = bytes.as_deref().map(fingerprint_bytes);
    let fp_legacy = bytes.as_deref().map(legacy_fingerprint_bytes);

    for entry in &entries {
        let stored = Some(entry.fingerprint.as_str());
        let matched_on = if stored == fp_gene.as_deref() {
            "gene_identity"
        } else if stored == fp.as_deref() {
            "file_bytes_fnv1a"
        } else if stored == fp_legacy.as_deref() {
            "file_bytes_legacy_defaulthasher"
        } else if normalize_path(&entry.portfolio_path) == norm {
            "recorded_path"
        } else {
            continue;
        };
        tracing::info!(
            target: "neoethos_app::strategy_blacklist",
            portfolio_path = %portfolio_path,
            matched_on,
            retired_path = %entry.portfolio_path,
            reason = %entry.reason,
            consecutive_losses = entry.consecutive_losses,
            net_pnl = entry.net_pnl,
            "BLACKLIST HIT: this strategy is retired and will not be selected"
        );
        return true;
    }
    false
}

/// Record a strategy as retired (idempotent on fingerprint). Best-effort:
/// logs + swallows I/O errors so culling never destabilizes the engine.
pub fn retire(entry: BlacklistEntry) {
    let Some(path) = blacklist_path() else { return };
    let mut entries = load();
    if entries.iter().any(|e| e.fingerprint == entry.fingerprint) {
        return; // already retired
    }
    // Say WHICH identity was recorded. A `gene:` entry blocks the strategy
    // across future discovery runs; a bare file-bytes entry blocks only that
    // exact artifact, which is the #218 hole and must be visible when it happens.
    let kind = if entry.fingerprint.starts_with(GENE_FINGERPRINT_PREFIX) {
        "gene_identity (blocks re-discovery of the same rule)"
    } else {
        "file_bytes (this artifact only — the portfolio did not parse as a \
         live-portfolio JSON, so the gene rule could not be extracted)"
    };
    tracing::warn!(
        target: "neoethos_app::strategy_blacklist",
        portfolio_path = %entry.portfolio_path,
        fingerprint = %entry.fingerprint,
        fingerprint_kind = kind,
        reason = %entry.reason,
        consecutive_losses = entry.consecutive_losses,
        net_pnl = entry.net_pnl,
        "RETIRING strategy to the permanent blacklist"
    );
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

    /// An artifact as discovery writes it: the rule, plus that run's
    /// measurements interleaved in the same struct.
    fn artifact(generation: u64, fitness: f64, strategy_id: &str, features: &[&str]) -> String {
        format!(
            r#"{{
              "schema_version": 3,
              "symbol": "EURUSD",
              "base_tf": "M5",
              "higher_tfs": ["H1"],
              "effective_feature_names": {},
              "normalize_features": false,
              "genes": [{{
                "indices": [0, 2],
                "weights": [0.5, -0.25],
                "long_threshold": 0.6,
                "short_threshold": -0.6,
                "fitness": {fitness},
                "sharpe_ratio": 1.7,
                "win_rate": 0.44,
                "max_drawdown": 0.11,
                "profit_factor": 1.6,
                "expectancy": 3.2,
                "trades_count": 431,
                "generation": {generation},
                "strategy_id": "{strategy_id}",
                "use_ob": true, "use_fvg": false, "use_liq_sweep": false,
                "mtf_confirmation": true, "use_premium_discount": false,
                "use_inducement": false, "use_bos": true, "use_choch": false,
                "use_eqh": false, "use_eql": false, "use_displacement": false,
                "tp_pips": 24.0, "sl_pips": 12.0,
                "slice_pass_rate": 0.8, "consistency": 0.7,
                "stop_vol_mult": 0.0
              }}]
            }}"#,
            serde_json::to_string(features).expect("feature list"),
        )
    }

    const FEATURES: [&str; 3] = ["rsi_14", "atr_14", "ema_50"];

    /// THE DEFECT #218 CLOSES. Two discovery runs, the same trading rule,
    /// different run metadata. The file bytes differ — which is exactly how a
    /// strategy culled after six consecutive real losses became eligible again
    /// with a clean record.
    #[test]
    fn a_rediscovered_identical_rule_keeps_its_identity() {
        let first = artifact(41, 2.7180000001, "gene-a1b2", &FEATURES);
        let second = artifact(88, 2.7180000009, "gene-9f9f", &FEATURES);

        assert_ne!(
            fingerprint_bytes(first.as_bytes()),
            fingerprint_bytes(second.as_bytes()),
            "the premise: the raw bytes differ between runs"
        );
        let a = gene_fingerprint_bytes(first.as_bytes()).expect("a live-portfolio artifact");
        let b = gene_fingerprint_bytes(second.as_bytes()).expect("a live-portfolio artifact");
        assert_eq!(a, b, "the same rule must keep the same identity across runs");
        assert!(a.starts_with(GENE_FINGERPRINT_PREFIX));
    }

    /// The other direction, and the one that must NOT over-fire: a different
    /// stop distance is a different strategy, however similar the rest looks.
    #[test]
    fn a_changed_rule_is_a_different_strategy() {
        let base = artifact(41, 2.7, "gene-a1b2", &FEATURES);
        let wider_stop = base.replace(r#""sl_pips": 12.0"#, r#""sl_pips": 30.0"#);
        assert_ne!(base, wider_stop, "the fixture must actually differ");
        assert_ne!(
            gene_fingerprint_bytes(base.as_bytes()),
            gene_fingerprint_bytes(wider_stop.as_bytes())
        );
    }

    /// An index is meaningless without the column list it indexes into. Two
    /// runs whose prefilters produced different orderings describe different
    /// strategies even with identical `indices`.
    #[test]
    fn indices_are_resolved_through_the_feature_names() {
        let a = artifact(1, 1.0, "x", &["rsi_14", "atr_14", "ema_50"]);
        // Same [0, 2], different columns → a different strategy.
        let b = artifact(1, 1.0, "x", &["macd", "atr_14", "bb_width"]);
        assert_ne!(
            gene_fingerprint_bytes(a.as_bytes()),
            gene_fingerprint_bytes(b.as_bytes())
        );

        // And the same columns reached by different positions ARE the same
        // strategy: [0, 2] over [rsi, atr, ema] selects rsi_14 + ema_50.
        let reordered = a
            .replace(r#""indices": [0, 2]"#, r#""indices": [2, 0]"#)
            .replace(
                r#"["rsi_14","atr_14","ema_50"]"#,
                r#"["ema_50","atr_14","rsi_14"]"#,
            );
        assert_eq!(
            gene_fingerprint_bytes(a.as_bytes()),
            gene_fingerprint_bytes(reordered.as_bytes()),
            "same columns, same weights, same order → same strategy"
        );
    }

    /// Key order and whitespace in the file must not change identity, or the
    /// fingerprint is a file hash wearing a different name.
    #[test]
    fn formatting_does_not_change_identity() {
        let pretty = artifact(1, 1.0, "x", &FEATURES);
        let value: serde_json::Value = serde_json::from_str(&pretty).expect("valid");
        let compact = serde_json::to_string(&value).expect("re-serialise");
        assert_eq!(
            gene_fingerprint_bytes(pretty.as_bytes()),
            gene_fingerprint_bytes(compact.as_bytes())
        );
    }

    /// Anything that is not a live-portfolio artifact must degrade to the old
    /// behaviour, not silently lose its identity.
    #[test]
    fn a_non_portfolio_file_falls_back_to_the_bytes() {
        assert!(gene_fingerprint_bytes(b"not json at all").is_none());
        assert!(gene_fingerprint_bytes(br#"{"no":"genes here"}"#).is_none());
        assert!(gene_fingerprint_bytes(br#"[1,2,3]"#).is_none());
    }

    /// An unresolvable index must be kept, never dropped — dropping a column
    /// would collapse two different strategies onto one fingerprint.
    #[test]
    fn an_out_of_range_index_is_kept_verbatim() {
        let names = ["rsi_14"];
        let resolved = neoethos_core::strategy_identity::resolve_indices(
            &serde_json::json!([0, 7]),
            &names,
        );
        assert_eq!(resolved, serde_json::json!(["rsi_14", 7]));
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
