//! Canonical identity of a discovered strategy — the ONE definition of "this is
//! the same trading rule", shared by everything that has to recognise one.
//!
//! # Why this is in `neoethos-core` — item #219, 2026-08-10
//!
//! The fingerprint was written in `neoethos-app`'s `app_services::
//! strategy_blacklist`, which is the top of the dependency graph. So the auto-
//! cull loop could RETIRE a strategy and refuse to select it again, while
//! `neoethos-search` — which cannot see `neoethos-app` — had zero references to
//! the blacklist and was free to re-derive the identical rule on the very run
//! the retirement queued. The loop looked closed and was not.
//!
//! Moving the definition down here (and DELETING the copy up there, rather than
//! leaving two) is what lets the search consult the same identity the live side
//! blacklists on. There is exactly one implementation; a second one would
//! reintroduce the defect the first day the two drifted.
//!
//! # What identity means
//!
//! A `neoethos_search::genetic::strategy_gene::Gene` carries the trading rule
//! (`indices`, `weights`, thresholds, the SMC flags, `tp_pips`/`sl_pips`,
//! `stop_vol_mult`) **and, in the same struct, that run's measurements**
//! (`fitness`, `sharpe_ratio`, `generation`, `strategy_id`, ...). So a fresh
//! run that rediscovers the identical rule writes a byte-different artifact.
//! Identity therefore = the rule with the measurements removed and the
//! positional `indices` resolved through `effective_feature_names`, because an
//! index is meaningless without the column list it indexes into.

use std::collections::HashSet;
use std::path::Path;

use serde_json::{Map, Value};

/// Marks a fingerprint as a GENE-identity hash rather than a file-bytes hash,
/// so a stored entry says which kind it is and the two can never collide.
pub const GENE_FINGERPRINT_PREFIX: &str = "gene:";

/// Marks a fingerprint as a SINGLE gene's rule identity, distinct from the
/// whole-artifact hash above. A portfolio of three genes has one
/// [`GENE_FINGERPRINT_PREFIX`] hash and three of these.
pub const GENE_RULE_FINGERPRINT_PREFIX: &str = "rule:";

/// Fields on a `Gene` that record HOW THAT RUN WENT, not what the strategy is.
///
/// Every one of these moves between two discovery runs that find the same
/// trading rule, which is precisely why a file-bytes fingerprint let a culled
/// strategy back in (#218). They are excluded from the identity.
///
/// **This is a deny-list, deliberately.** Anything not named here JOINS the
/// identity, so a new *rule* field added to `Gene` is covered automatically.
/// The failure mode of a stale deny-list is an over-specific fingerprint —
/// which never blocks a strategy that was not culled. An allow-list would fail
/// the other way: a new rule field silently ignored, two genuinely different
/// strategies sharing one fingerprint, and a strategy blocked that nobody
/// retired.
///
/// **If you add a per-run measurement to `Gene`, add it here.**
pub const GENE_MEASUREMENT_FIELDS: &[&str] = &[
    "fitness",
    "sharpe_ratio",
    "win_rate",
    "max_drawdown",
    "profit_factor",
    "expectancy",
    "trades_count",
    "generation",
    "strategy_id",
    "slice_pass_rate",
    "consistency",
];

/// Top-level artifact fields excluded from the identity.
///
/// `schema_version` is the file format, not the strategy. `effective_feature_names`
/// is not dropped so much as CONSUMED — it is folded into each gene by resolving
/// `indices` to names, which is the only way a positional index means anything.
pub const PORTFOLIO_NON_IDENTITY_FIELDS: &[&str] = &["schema_version", "effective_feature_names"];

/// Deterministic textual form of a JSON value: object keys sorted, no
/// whitespace. Written explicitly rather than relying on `serde_json`'s map
/// ordering, which is a Cargo-feature (`preserve_order`) away from changing and
/// would silently invalidate every stored fingerprint if it did.
fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => {
            // Delegate escaping to serde_json so quotes/control chars cannot
            // forge a boundary between two different values.
            out.push_str(&Value::String(s.clone()).to_string());
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String((*key).clone()).to_string());
                out.push(':');
                if let Some(v) = map.get(*key) {
                    write_canonical(v, out);
                }
            }
            out.push('}');
        }
    }
}

/// Replace a gene's positional `indices` with the feature NAMES they select.
///
/// An index only means something against the column list discovery produced, so
/// two runs whose prefilters ordered features differently describe different
/// strategies with identical index arrays. An index with no name (list absent,
/// or out of range) is kept VERBATIM rather than dropped — dropping a column
/// would make two different strategies hash the same.
pub fn resolve_indices(indices: &Value, names: &[&str]) -> Value {
    let Some(items) = indices.as_array() else {
        return indices.clone();
    };
    Value::Array(
        items
            .iter()
            .map(|index| {
                match index
                    .as_u64()
                    .and_then(|position| names.get(position as usize))
                {
                    Some(name) => Value::String((*name).to_string()),
                    None => index.clone(),
                }
            })
            .collect(),
    )
}

/// One gene reduced to the trading rule: measurements removed, indices named.
pub fn gene_rule_identity(gene: &Value, names: &[&str]) -> Value {
    let Some(fields) = gene.as_object() else {
        return gene.clone();
    };
    let mut rule = Map::new();
    for (key, value) in fields {
        if GENE_MEASUREMENT_FIELDS.contains(&key.as_str()) {
            continue;
        }
        if key == "indices" {
            rule.insert("features".to_string(), resolve_indices(value, names));
            continue;
        }
        rule.insert(key.clone(), value.clone());
    }
    Value::Object(rule)
}

/// Fingerprint of ONE gene's trading rule, independent of the run that produced
/// it and of whatever other genes shipped beside it in a portfolio.
///
/// This is the identity discovery filters on: a retired strategy that reappears
/// bundled with two different genes hashes differently as an ARTIFACT but
/// identically as a RULE, which is the hole `is_blacklisted` alone could not
/// close.
pub fn gene_rule_fingerprint(gene: &Value, names: &[&str]) -> String {
    let mut canonical = String::new();
    write_canonical(&gene_rule_identity(gene, names), &mut canonical);
    format!(
        "{GENE_RULE_FINGERPRINT_PREFIX}{:016x}",
        crate::utils::hashing::fnv1a64(canonical.as_bytes())
    )
}

/// Fingerprint the STRATEGY a live-portfolio artifact describes, independent of
/// which run produced the file.
///
/// `None` when the bytes are not a live-portfolio artifact (unparseable, or no
/// `genes` array) — the caller then falls back to the file-bytes fingerprint,
/// so an unrecognised shape degrades to the old behaviour instead of silently
/// producing no identity at all.
pub fn portfolio_gene_fingerprint(bytes: &[u8]) -> Option<String> {
    let parsed: Value = serde_json::from_slice(bytes).ok()?;
    let fields = parsed.as_object()?;
    let genes = fields.get("genes")?.as_array()?;
    let names = effective_feature_names(fields);

    // Everything else on the artifact — symbol, base_tf, higher_tfs,
    // normalize_features — IS identity: it changes what the rule does. Copying
    // by iteration rather than by an allow-list means a future field joins the
    // identity by default.
    let mut identity = Map::new();
    for (key, value) in fields {
        if key == "genes" || PORTFOLIO_NON_IDENTITY_FIELDS.contains(&key.as_str()) {
            continue;
        }
        identity.insert(key.clone(), value.clone());
    }
    identity.insert(
        "genes".to_string(),
        Value::Array(
            genes
                .iter()
                .map(|gene| gene_rule_identity(gene, &names))
                .collect(),
        ),
    );

    let mut canonical = String::new();
    write_canonical(&Value::Object(identity), &mut canonical);
    Some(format!(
        "{GENE_FINGERPRINT_PREFIX}{:016x}",
        crate::utils::hashing::fnv1a64(canonical.as_bytes())
    ))
}

/// Every PER-GENE rule fingerprint in a live-portfolio artifact's bytes.
///
/// Empty when the bytes are not a recognisable artifact — the caller must treat
/// that as "no identities learned from this file", never as "no strategies are
/// retired".
pub fn gene_rule_fingerprints(bytes: &[u8]) -> Vec<String> {
    let Ok(parsed) = serde_json::from_slice::<Value>(bytes) else {
        return Vec::new();
    };
    let Some(fields) = parsed.as_object() else {
        return Vec::new();
    };
    let Some(genes) = fields.get("genes").and_then(|g| g.as_array()) else {
        return Vec::new();
    };
    let names = effective_feature_names(fields);
    genes
        .iter()
        .map(|gene| gene_rule_fingerprint(gene, &names))
        .collect()
}

fn effective_feature_names(fields: &Map<String, Value>) -> Vec<&str> {
    fields
        .get("effective_feature_names")
        .and_then(|v| v.as_array())
        .map(|list| list.iter().map(|v| v.as_str().unwrap_or("")).collect())
        .unwrap_or_default()
}

/// The rule fingerprints of every strategy the auto-cull retired.
///
/// Built by reading `<data_dir>/strategy_blacklist.json` and, for each entry,
/// the artifact it names. Nothing is deleted when a strategy is retired (the
/// "never delete strategies/data" invariant), so the file is normally still
/// there; an entry whose file is gone contributes no rule identity and says so.
#[derive(Debug, Clone, Default)]
pub struct RetiredRules {
    rules: HashSet<String>,
    /// Entries in the blacklist file whose artifact could not be read, so their
    /// rules are NOT in `rules`. Reported, never silently swallowed: this is the
    /// count of retired strategies discovery can still re-derive.
    pub unreadable_entries: usize,
    /// Total entries seen in the blacklist file.
    pub entries: usize,
}

impl RetiredRules {
    /// Canonical file name under the data dir.
    pub const FILE_NAME: &'static str = "strategy_blacklist.json";

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    pub fn len(&self) -> usize {
        self.rules.len()
    }

    pub fn contains(&self, rule_fingerprint: &str) -> bool {
        self.rules.contains(rule_fingerprint)
    }

    /// Read the blacklist under `data_dir` and resolve every entry to its
    /// per-gene rule identities.
    ///
    /// A missing file is the normal case (nothing has ever been retired) and
    /// yields an empty set. A file that exists but cannot be parsed is an
    /// ERROR: it means retirements were recorded and are now not being
    /// honoured, which the operator must see.
    pub fn load_from_data_dir(data_dir: impl AsRef<Path>) -> Self {
        let path = data_dir.as_ref().join(Self::FILE_NAME);
        let raw = match std::fs::read_to_string(&path) {
            Ok(raw) => raw,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Self::default(),
            Err(err) => {
                tracing::error!(
                    target: "neoethos_core::strategy_identity",
                    path = %path.display(), error = %err,
                    "the strategy blacklist exists but could not be read — retired strategies \
                     are NOT being excluded on this run"
                );
                return Self::default();
            }
        };
        let entries: Vec<Value> = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(err) => {
                tracing::error!(
                    target: "neoethos_core::strategy_identity",
                    path = %path.display(), error = %err,
                    "the strategy blacklist is not readable JSON — retired strategies are NOT \
                     being excluded on this run"
                );
                return Self::default();
            }
        };

        let mut out = Self {
            entries: entries.len(),
            ..Self::default()
        };
        for entry in &entries {
            let Some(portfolio_path) = entry
                .get("portfolioPath")
                .or_else(|| entry.get("portfolio_path"))
                .and_then(|v| v.as_str())
            else {
                out.unreadable_entries += 1;
                continue;
            };
            match std::fs::read(portfolio_path) {
                Ok(bytes) => {
                    let rules = gene_rule_fingerprints(&bytes);
                    if rules.is_empty() {
                        out.unreadable_entries += 1;
                    }
                    out.rules.extend(rules);
                }
                Err(err) => {
                    out.unreadable_entries += 1;
                    tracing::warn!(
                        target: "neoethos_core::strategy_identity",
                        portfolio_path, error = %err,
                        "a retired strategy's artifact is unreadable, so its RULE cannot be \
                         recognised — discovery can re-derive this one"
                    );
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn artifact(fitness: f64, generation: u64) -> Value {
        json!({
            "schema_version": 1,
            "symbol": "EURUSD",
            "base_tf": "M5",
            "effective_feature_names": ["rsi_14", "atr_14", "ema_50"],
            "normalize_features": false,
            "genes": [{
                "indices": [0, 2],
                "weights": [0.5, -0.25],
                "tp_pips": 20.0,
                "sl_pips": 10.0,
                "fitness": fitness,
                "generation": generation,
                "strategy_id": format!("gen{generation}"),
            }],
        })
    }

    #[test]
    fn the_same_rule_rediscovered_has_the_same_rule_fingerprint() {
        let a = serde_json::to_vec(&artifact(1.5, 3)).expect("serialize");
        let b = serde_json::to_vec(&artifact(1.9, 41)).expect("serialize");
        assert_eq!(
            gene_rule_fingerprints(&a),
            gene_rule_fingerprints(&b),
            "measurements must not be part of the rule identity — that is #218"
        );
    }

    #[test]
    fn a_different_rule_hashes_differently() {
        let mut other = artifact(1.5, 3);
        other["genes"][0]["tp_pips"] = json!(40.0);
        let a = serde_json::to_vec(&artifact(1.5, 3)).expect("serialize");
        let b = serde_json::to_vec(&other).expect("serialize");
        assert_ne!(gene_rule_fingerprints(&a), gene_rule_fingerprints(&b));
    }

    #[test]
    fn indices_are_resolved_through_the_feature_names() {
        // Same positional indices, different column list ⇒ different strategy.
        let mut renamed = artifact(1.5, 3);
        renamed["effective_feature_names"] = json!(["macd", "atr_14", "ema_200"]);
        let a = serde_json::to_vec(&artifact(1.5, 3)).expect("serialize");
        let b = serde_json::to_vec(&renamed).expect("serialize");
        assert_ne!(gene_rule_fingerprints(&a), gene_rule_fingerprints(&b));
    }

    #[test]
    fn unrecognisable_bytes_yield_no_identities() {
        assert!(gene_rule_fingerprints(b"not json").is_empty());
        assert!(portfolio_gene_fingerprint(b"not json").is_none());
    }

    #[test]
    fn a_missing_blacklist_is_an_empty_set_not_an_error() {
        let dir = std::env::temp_dir().join("neoethos-retired-rules-missing");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        let retired = RetiredRules::load_from_data_dir(&dir);
        assert!(retired.is_empty());
        assert_eq!(retired.entries, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
