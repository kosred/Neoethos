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

use serde::Deserialize;
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
    write_canonical_chunks(value, &mut |chunk| out.push_str(chunk));
}

fn write_canonical_chunks(value: &Value, out: &mut impl FnMut(&str)) {
    match value {
        Value::Null => out("null"),
        Value::Bool(b) => out(if *b { "true" } else { "false" }),
        Value::Number(n) => out(&n.to_string()),
        Value::String(s) => {
            // Delegate escaping to serde_json so quotes/control chars cannot
            // forge a boundary between two different values.
            out(&Value::String(s.clone()).to_string());
        }
        Value::Array(items) => {
            out("[");
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out(",");
                }
                write_canonical_chunks(item, out);
            }
            out("]");
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out("{");
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out(",");
                }
                out(&Value::String((*key).clone()).to_string());
                out(":");
                if let Some(v) = map.get(*key) {
                    write_canonical_chunks(v, out);
                }
            }
            out("}");
        }
    }
}

// Fingerprinting is not Search's receipt/strategy validation authority. These
// private DTOs recognize its exact transport shape without a Core -> Search
// dependency. Unknown/malformed shared formats must never fall back to legacy.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SharedPortfolioWireV1 {
    shared_receipt_schema_version: u16,
    artifact_kind: String,
    input_receipt: Value,
    portfolio: SharedPortfolioBodyV1,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SharedPortfolioBodyV1 {
    schema_version: u16,
    portfolio_schema_version: u32,
    portfolio_identity_sha256: String,
    search_scope: SharedScopeV1,
    // V5 has no final reservation. A V6 body must carry the real typed scope;
    // explicit null is not a legacy omission and must not pass as one.
    #[serde(default, deserialize_with = "present_shared_scope")]
    final_holdout_scope: Option<SharedScopeV1>,
    search_config_hash: String,
    live_trading_policy: Value,
    symbol: String,
    base_tf: String,
    higher_tfs: Vec<String>,
    effective_feature_names: Vec<String>,
    normalize_features: bool,
    genes: Vec<Value>,
    sizing_evidence: Vec<SharedSizingV1>,
    cost_band: Value,
}

#[derive(Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct SharedScopeV1 {
    schema_version: u16,
    receipt_sha256: String,
    scope_sha256: String,
    evaluated_window: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SharedSizingV1 {
    schema_version: u16,
    forward_test_artifact_kind: String,
    forward_test_schema_version: u32,
    scope: SharedScopeV1,
    search_config_hash: String,
    strategy_identity: Value,
    summary: Value,
}

enum PortfolioWire {
    Legacy(Value),
    Shared(SharedPortfolioWireV1),
}

fn present_wire_field<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<bool, D::Error> {
    serde::de::IgnoredAny::deserialize(deserializer)?;
    Ok(true)
}

fn present_shared_scope<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<SharedScopeV1>, D::Error> {
    SharedScopeV1::deserialize(deserializer).map(Some)
}

fn valid_shared_window(value: &Value, allow_selection_validation: bool) -> bool {
    let Some(fields) = value.as_object() else {
        return false;
    };
    let Some(start) = fields.get("row_start").and_then(Value::as_u64) else {
        return false;
    };
    let Some(end) = fields.get("row_end").and_then(Value::as_u64) else {
        return false;
    };
    let Some(first) = fields.get("timestamp_start_ms").and_then(Value::as_i64) else {
        return false;
    };
    let Some(last) = fields.get("timestamp_end_ms").and_then(Value::as_i64) else {
        return false;
    };
    fields.len() == 5
        && start < end
        && first <= last
        && (matches!(
            fields.get("role").and_then(Value::as_str),
            Some(
                "discovery_input"
                    | "in_sample"
                    | "holdout"
                    | "walk_forward_train"
                    | "walk_forward_validation"
                    | "forward_test"
                    | "live_simulation"
                    | "prop_firm_risk"
            )
        ) || (allow_selection_validation
            && fields.get("role").and_then(Value::as_str) == Some("selection_validation")))
}

// Recognize the V6 transport's three distinct windows without claiming Search's
// receipt/content validation authority. All three scopes must refer to the same
// receipt, and every sizing row must describe the same intervening calibration.
fn valid_shared_v6_partition(body: &SharedPortfolioBodyV1) -> bool {
    let Some(final_scope) = &body.final_holdout_scope else {
        return false;
    };
    let Some(first_sizing) = body.sizing_evidence.first() else {
        return false;
    };
    let selected = &body.search_scope.evaluated_window;
    let final_window = &final_scope.evaluated_window;
    selected["role"].as_str() == Some("in_sample")
        && final_window["role"].as_str() == Some("holdout")
        && final_scope.receipt_sha256 == body.search_scope.receipt_sha256
        && selected["row_end"].as_u64() < final_window["row_start"].as_u64()
        && selected["timestamp_end_ms"].as_i64() < final_window["timestamp_start_ms"].as_i64()
        && body.sizing_evidence.iter().all(|evidence| {
            let calibration = &evidence.scope.evaluated_window;
            evidence.scope == first_sizing.scope
                && calibration["role"].as_str() == Some("selection_validation")
                && selected["row_end"] == calibration["row_start"]
                && selected["timestamp_end_ms"].as_i64()
                    < calibration["timestamp_start_ms"].as_i64()
                && calibration["row_end"] == final_window["row_start"]
                && calibration["timestamp_end_ms"].as_i64()
                    < final_window["timestamp_start_ms"].as_i64()
        })
}

fn portfolio_wire(bytes: &[u8]) -> Option<PortfolioWire> {
    #[derive(Default, Deserialize)]
    struct Probe {
        #[serde(default, deserialize_with = "present_wire_field")]
        shared_receipt_schema_version: bool,
        #[serde(default, deserialize_with = "present_wire_field")]
        artifact_kind: bool,
        #[serde(default, deserialize_with = "present_wire_field")]
        input_receipt: bool,
        #[serde(default, deserialize_with = "present_wire_field")]
        portfolio: bool,
        #[serde(default, deserialize_with = "present_wire_field")]
        portfolio_schema_version: bool,
        #[serde(default, deserialize_with = "present_wire_field")]
        portfolio_identity_sha256: bool,
    }
    let probe: Probe = serde_json::from_slice(bytes).ok()?;
    if !(probe.shared_receipt_schema_version
        || probe.artifact_kind
        || probe.input_receipt
        || probe.portfolio
        || probe.portfolio_schema_version
        || probe.portfolio_identity_sha256)
    {
        return serde_json::from_slice(bytes)
            .ok()
            .map(PortfolioWire::Legacy);
    }
    let wire: SharedPortfolioWireV1 = serde_json::from_slice(bytes).ok()?;
    let body = &wire.portfolio;
    let sha256 =
        |value: &str| value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit());
    let valid_scope = |scope: &SharedScopeV1| {
        scope.schema_version == 1
            && sha256(&scope.receipt_sha256)
            && sha256(&scope.scope_sha256)
            && valid_shared_window(&scope.evaluated_window, body.portfolio_schema_version == 6)
    };
    if wire.shared_receipt_schema_version != 1
        || wire.artifact_kind != "neoethos.live-portfolio-shared-receipt.v1"
        || wire
            .input_receipt
            .get("schema_version")
            .and_then(Value::as_u64)
            != Some(2)
        || body.schema_version != 1
        || !matches!(body.portfolio_schema_version, 5 | 6)
        || !sha256(&body.portfolio_identity_sha256)
        || !valid_scope(&body.search_scope)
        || body.symbol.is_empty()
        || body.base_tf.is_empty()
        || body.search_config_hash.is_empty()
        || !body.live_trading_policy.is_object()
        || body.effective_feature_names.is_empty()
        || body.effective_feature_names.iter().any(String::is_empty)
        || body
            .effective_feature_names
            .iter()
            .collect::<HashSet<_>>()
            .len()
            != body.effective_feature_names.len()
        || body.genes.iter().any(|gene| !gene.is_object())
        || body.genes.len() != body.sizing_evidence.len()
        || body.cost_band.as_array()?.len() != body.genes.len()
        || body.sizing_evidence.iter().any(|evidence| {
            evidence.schema_version != 1
                || evidence.forward_test_schema_version != 3
                || evidence.forward_test_artifact_kind
                    != "neoethos.search.forward-test-validation.v3"
                || !valid_scope(&evidence.scope)
                || evidence.scope.receipt_sha256 != body.search_scope.receipt_sha256
                || evidence.search_config_hash != body.search_config_hash
                || !evidence.strategy_identity.is_object()
                || !evidence.summary.is_object()
        })
    {
        return None;
    }
    match body.portfolio_schema_version {
        5 if body.final_holdout_scope.is_none() => {}
        6 if body.final_holdout_scope.as_ref().is_some_and(valid_scope)
            && valid_shared_v6_partition(body) => {}
        _ => return None,
    }
    Some(PortfolioWire::Shared(wire))
}

// Small borrowed virtual JSON nodes reconstruct the unchanged V5/V6 identity. Each
// scope references ONE parsed receipt; neither an N-receipt Value tree nor the
// expanded canonical JSON string is ever allocated.
enum CanonicalPart<'a> {
    Value(&'a Value),
    Text(&'a str),
    Literal(&'static str),
    Object(Vec<(&'static str, CanonicalPart<'a>)>),
    Array(Vec<CanonicalPart<'a>>),
    Rules(&'a [Value], &'a [String]),
}

impl CanonicalPart<'_> {
    fn write(&self, out: &mut impl FnMut(&str)) {
        match self {
            Self::Value(value) => write_canonical_chunks(value, out),
            Self::Text(text) => out(&serde_json::to_string(text).expect("serialize text")),
            Self::Literal(text) => out(text),
            Self::Object(fields) => {
                let mut ordered = fields.iter().collect::<Vec<_>>();
                ordered.sort_by_key(|(key, _)| *key);
                out("{");
                for (index, (key, value)) in ordered.into_iter().enumerate() {
                    if index != 0 {
                        out(",");
                    }
                    Self::Text(key).write(out);
                    out(":");
                    value.write(out);
                }
                out("}");
            }
            Self::Array(values) => {
                out("[");
                for (index, value) in values.iter().enumerate() {
                    if index != 0 {
                        out(",");
                    }
                    value.write(out);
                }
                out("]");
            }
            Self::Rules(genes, names) => {
                let names = names.iter().map(String::as_str).collect::<Vec<_>>();
                out("[");
                for (index, gene) in genes.iter().enumerate() {
                    if index != 0 {
                        out(",");
                    }
                    write_canonical_chunks(&gene_rule_identity(gene, &names), out);
                }
                out("]");
            }
        }
    }
}

fn expanded_scope<'a>(scope: &'a SharedScopeV1, receipt: &'a Value) -> CanonicalPart<'a> {
    use CanonicalPart::{Literal, Object, Text, Value as Part};
    Object(vec![
        ("schema_version", Literal("2")),
        ("receipt", Part(receipt)),
        ("receipt_sha256", Text(&scope.receipt_sha256)),
        ("evaluated_window", Part(&scope.evaluated_window)),
    ])
}

fn shared_portfolio_fingerprint(wire: &SharedPortfolioWireV1) -> String {
    use CanonicalPart::{Array, Literal, Object, Rules, Text, Value as Part};
    let body = &wire.portfolio;
    let sizing = body
        .sizing_evidence
        .iter()
        .map(|evidence| {
            Object(vec![(
                "forward_test",
                Object(vec![
                    ("schema_version", Literal("2")),
                    ("artifact_kind", Text(&evidence.forward_test_artifact_kind)),
                    (
                        "scope",
                        expanded_scope(&evidence.scope, &wire.input_receipt),
                    ),
                    ("search_config_hash", Text(&evidence.search_config_hash)),
                    (
                        "payload",
                        Object(vec![
                            ("schema_version", Literal("3")),
                            ("strategy_identity", Part(&evidence.strategy_identity)),
                            ("summary", Part(&evidence.summary)),
                        ]),
                    ),
                ]),
            )])
        })
        .collect();
    let mut fields = vec![
        (
            "search_scope",
            expanded_scope(&body.search_scope, &wire.input_receipt),
        ),
        ("search_config_hash", Text(&body.search_config_hash)),
        ("live_trading_policy", Part(&body.live_trading_policy)),
        ("symbol", Text(&body.symbol)),
        ("base_tf", Text(&body.base_tf)),
        (
            "higher_tfs",
            Array(body.higher_tfs.iter().map(|tf| Text(tf)).collect()),
        ),
        (
            "normalize_features",
            Literal(if body.normalize_features {
                "true"
            } else {
                "false"
            }),
        ),
        ("genes", Rules(&body.genes, &body.effective_feature_names)),
        ("sizing_evidence", Array(sizing)),
        ("cost_band", Part(&body.cost_band)),
    ];
    if let Some(final_scope) = &body.final_holdout_scope {
        fields.push((
            "final_holdout_scope",
            expanded_scope(final_scope, &wire.input_receipt),
        ));
    }
    let identity = Object(fields);
    let mut hash = crate::utils::hashing::fnv1a64(&[]);
    identity.write(&mut |chunk| {
        hash = crate::utils::hashing::fnv1a64_update(hash, chunk.as_bytes());
    });
    format!("{GENE_FINGERPRINT_PREFIX}{hash:016x}")
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
    let parsed = match portfolio_wire(bytes)? {
        PortfolioWire::Legacy(parsed) => parsed,
        PortfolioWire::Shared(wire) => return Some(shared_portfolio_fingerprint(&wire)),
    };
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
    let Some(wire) = portfolio_wire(bytes) else {
        return Vec::new();
    };
    let parsed = match wire {
        PortfolioWire::Legacy(parsed) => parsed,
        PortfolioWire::Shared(wire) => {
            let names = wire
                .portfolio
                .effective_feature_names
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            return wire
                .portfolio
                .genes
                .iter()
                .map(|gene| gene_rule_fingerprint(gene, &names))
                .collect();
        }
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

    // Exact shared transport layout, independently paired with its legacy V5
    // expansion. Core cannot manufacture Search validation authority; the
    // actual Search serializer is also checked in its live_portfolio tests.
    fn shared_fixture() -> (Value, Value) {
        let receipt = json!({"schema_version": 2, "feature_content_sha256": "a".repeat(64)});
        let window = json!({
            "role": "in_sample", "row_start": 0, "row_end": 80,
            "timestamp_start_ms": 1000, "timestamp_end_ms": 80000,
        });
        let holdout = json!({
            "role": "holdout", "row_start": 80, "row_end": 100,
            "timestamp_start_ms": 81000, "timestamp_end_ms": 100000,
        });
        let reference = |window: &Value| {
            json!({
                "schema_version": 1, "receipt_sha256": "b".repeat(64),
                "scope_sha256": "c".repeat(64), "evaluated_window": window,
            })
        };
        let expanded = |window: &Value| {
            json!({
                "schema_version": 2, "receipt": receipt,
                "receipt_sha256": "b".repeat(64), "evaluated_window": window,
            })
        };
        let mut legacy = artifact(1.5, 3);
        legacy["schema_version"] = json!(5);
        legacy["search_scope"] = expanded(&window);
        legacy["search_config_hash"] = json!("fnv64:0123456789abcdef");
        legacy["live_trading_policy"] = json!({"schema_version": 1, "trailing_enabled": true});
        legacy["higher_tfs"] = json!(["H1"]);
        legacy["cost_band"] = json!([["gen3", "cost_band_survives"]]);
        let strategy = json!({"schema_version": 2, "exact_gene_hash": "fnv64:fedcba9876543210"});
        let summary = json!({"bars": 20, "metrics": {"net_profit": 42.5}});
        legacy["sizing_evidence"] = json!([{"forward_test": {
            "schema_version": 2,
            "artifact_kind": "neoethos.search.forward-test-validation.v3",
            "scope": expanded(&holdout),
            "search_config_hash": legacy["search_config_hash"],
            "payload": {"schema_version": 3, "strategy_identity": strategy, "summary": summary},
        }}]);
        let mut body = legacy.clone();
        body["schema_version"] = json!(1);
        body["portfolio_schema_version"] = json!(5);
        body["portfolio_identity_sha256"] = json!("d".repeat(64));
        body["search_scope"] = reference(&window);
        body["sizing_evidence"] = json!([{
            "schema_version": 1,
            "forward_test_artifact_kind": "neoethos.search.forward-test-validation.v3",
            "forward_test_schema_version": 3,
            "scope": reference(&holdout),
            "search_config_hash": legacy["search_config_hash"],
            "strategy_identity": strategy, "summary": summary,
        }]);
        let shared = json!({
            "shared_receipt_schema_version": 1,
            "artifact_kind": "neoethos.live-portfolio-shared-receipt.v1",
            "input_receipt": receipt,
            "portfolio": body,
        });
        (legacy, shared)
    }

    // These are shape/identity fixtures, not validated receipt or OOS evidence.
    // Search separately tests this reader with its real raw/normalized serializer.
    fn shared_v6_fixture() -> (Value, Value) {
        let (mut plain, mut shared) = shared_fixture();
        let calibration = json!({
            "role": "selection_validation", "row_start": 80, "row_end": 90,
            "timestamp_start_ms": 81000, "timestamp_end_ms": 90000,
        });
        let final_window = json!({
            "role": "holdout", "row_start": 90, "row_end": 100,
            "timestamp_start_ms": 91000, "timestamp_end_ms": 100000,
        });
        plain["schema_version"] = json!(6);
        plain["final_holdout_scope"] = plain["search_scope"].clone();
        plain["final_holdout_scope"]["evaluated_window"] = final_window.clone();
        plain["sizing_evidence"][0]["forward_test"]["scope"]["evaluated_window"] =
            calibration.clone();
        plain["sizing_evidence"][0]["forward_test"]["payload"]["summary"]["bars"] = json!(10);
        shared["portfolio"]["portfolio_schema_version"] = json!(6);
        shared["portfolio"]["final_holdout_scope"] = shared["portfolio"]["search_scope"].clone();
        shared["portfolio"]["final_holdout_scope"]["evaluated_window"] = final_window;
        shared["portfolio"]["sizing_evidence"][0]["scope"]["evaluated_window"] = calibration;
        shared["portfolio"]["sizing_evidence"][0]["summary"]["bars"] = json!(10);
        (plain, shared)
    }

    #[test]
    fn shared_portfolio_preserves_legacy_fingerprint_and_per_gene_rules() {
        let (legacy, shared) = shared_fixture();
        let legacy = serde_json::to_vec(&legacy).unwrap();
        let shared = serde_json::to_vec_pretty(&shared).unwrap();
        let old = portfolio_gene_fingerprint(&legacy).unwrap();
        assert_eq!(portfolio_gene_fingerprint(&shared), Some(old));
        let rules = gene_rule_fingerprints(&legacy);
        assert_eq!(rules.len(), 1);
        assert_eq!(gene_rule_fingerprints(&shared), rules);

        let (_, mut changed) = shared_fixture();
        changed["portfolio"]["genes"][0]["sl_pips"] = json!(11.0);
        let changed = serde_json::to_vec(&changed).unwrap();
        assert_ne!(gene_rule_fingerprints(&changed), rules);
        assert_ne!(
            portfolio_gene_fingerprint(&changed),
            portfolio_gene_fingerprint(&shared)
        );
    }

    #[test]
    fn shared_v6_preserves_plain_identity_and_rules_with_distinct_final_scope() {
        for normalized in [false, true] {
            let (mut plain, mut shared) = shared_v6_fixture();
            plain["normalize_features"] = json!(normalized);
            shared["portfolio"]["normalize_features"] = json!(normalized);
            let plain_bytes = serde_json::to_vec(&plain).unwrap();
            let shared_bytes = serde_json::to_vec(&shared).unwrap();
            let identity = portfolio_gene_fingerprint(&plain_bytes).unwrap();
            let rules = gene_rule_fingerprints(&plain_bytes);
            assert_eq!(
                portfolio_gene_fingerprint(&shared_bytes),
                Some(identity.clone())
            );
            assert_eq!(gene_rule_fingerprints(&shared_bytes), rules);

            // The reserved tail is part of the full artifact identity, not the
            // trading rule. Changing it must not be lost in compact expansion.
            plain["final_holdout_scope"]["evaluated_window"]["row_end"] = json!(101);
            plain["final_holdout_scope"]["evaluated_window"]["timestamp_end_ms"] = json!(101000);
            shared["portfolio"]["final_holdout_scope"]["evaluated_window"]["row_end"] = json!(101);
            shared["portfolio"]["final_holdout_scope"]["evaluated_window"]["timestamp_end_ms"] =
                json!(101000);
            let changed_plain = serde_json::to_vec(&plain).unwrap();
            let changed_shared = serde_json::to_vec(&shared).unwrap();
            let changed_identity = portfolio_gene_fingerprint(&changed_plain).unwrap();
            assert_ne!(changed_identity, identity);
            assert_eq!(
                portfolio_gene_fingerprint(&changed_shared),
                Some(changed_identity)
            );
            assert_eq!(gene_rule_fingerprints(&changed_shared), rules);
        }
    }

    #[test]
    fn shared_v6_rejects_mixed_versions_missing_final_and_calibration_rebinding() {
        let (plain, shared) = shared_v6_fixture();
        let mutations: &[(&str, fn(&mut Value))] = &[
            ("missing final scope", |v| {
                v["portfolio"]
                    .as_object_mut()
                    .unwrap()
                    .remove("final_holdout_scope");
            }),
            ("null final scope", |v| {
                v["portfolio"]["final_holdout_scope"] = Value::Null
            }),
            ("V5 carrying a V6 final scope", |v| {
                v["portfolio"]["portfolio_schema_version"] = json!(5)
            }),
            ("unknown portfolio version", |v| {
                v["portfolio"]["portfolio_schema_version"] = json!(7)
            }),
            ("unknown final field", |v| {
                v["portfolio"]["final_holdout_scope"]["unknown"] = json!(true)
            }),
            ("final scope reference version", |v| {
                v["portfolio"]["final_holdout_scope"]["schema_version"] = json!(2)
            }),
            ("final scope malformed hash", |v| {
                v["portfolio"]["final_holdout_scope"]["scope_sha256"] = json!("not-sha256")
            }),
            ("final scope different receipt", |v| {
                v["portfolio"]["final_holdout_scope"]["receipt_sha256"] = json!("e".repeat(64))
            }),
            ("final is calibration", |v| {
                v["portfolio"]["final_holdout_scope"]["evaluated_window"]["role"] =
                    json!("selection_validation")
            }),
            ("search is not IS", |v| {
                v["portfolio"]["search_scope"]["evaluated_window"]["role"] =
                    json!("discovery_input")
            }),
            ("old holdout used for sizing", |v| {
                v["portfolio"]["sizing_evidence"][0]["scope"]["evaluated_window"]["role"] =
                    json!("holdout")
            }),
            ("calibration starts after a gap", |v| {
                v["portfolio"]["sizing_evidence"][0]["scope"]["evaluated_window"]["row_start"] =
                    json!(81)
            }),
            ("calibration overlaps final", |v| {
                v["portfolio"]["sizing_evidence"][0]["scope"]["evaluated_window"]["row_end"] =
                    json!(91)
            }),
            ("final timestamp overlaps calibration", |v| {
                v["portfolio"]["final_holdout_scope"]["evaluated_window"]["timestamp_start_ms"] =
                    json!(90000)
            }),
            ("calibration timestamp overlaps IS", |v| {
                v["portfolio"]["sizing_evidence"][0]["scope"]["evaluated_window"]["timestamp_start_ms"] =
                    json!(80000)
            }),
            ("different per-gene calibration scope", |v| {
                let second_gene = v["portfolio"]["genes"][0].clone();
                v["portfolio"]["genes"]
                    .as_array_mut()
                    .unwrap()
                    .push(second_gene);
                let mut second_sizing = v["portfolio"]["sizing_evidence"][0].clone();
                second_sizing["scope"]["scope_sha256"] = json!("e".repeat(64));
                v["portfolio"]["sizing_evidence"]
                    .as_array_mut()
                    .unwrap()
                    .push(second_sizing);
                let second_cost = v["portfolio"]["cost_band"][0].clone();
                v["portfolio"]["cost_band"]
                    .as_array_mut()
                    .unwrap()
                    .push(second_cost);
            }),
        ];
        for (case, mutate) in mutations {
            let mut changed = shared.clone();
            mutate(&mut changed);
            let bytes = serde_json::to_vec(&changed).unwrap();
            assert!(portfolio_gene_fingerprint(&bytes).is_none(), "{case}");
            assert!(gene_rule_fingerprints(&bytes).is_empty(), "{case}");
            changed["genes"] = plain["genes"].clone();
            let bytes = serde_json::to_vec(&changed).unwrap();
            assert!(
                portfolio_gene_fingerprint(&bytes).is_none(),
                "fallback: {case}"
            );
            assert!(
                gene_rule_fingerprints(&bytes).is_empty(),
                "fallback: {case}"
            );
        }
        let (_, mut v5) = shared_fixture();
        v5["portfolio"]["final_holdout_scope"] = Value::Null;
        let bytes = serde_json::to_vec(&v5).unwrap();
        assert!(portfolio_gene_fingerprint(&bytes).is_none());
        assert!(gene_rule_fingerprints(&bytes).is_empty());
    }

    #[test]
    fn legacy_portfolio_canonical_identity_bytes_remain_unchanged() {
        let bytes = br#"{"schema_version":3,"symbol":"EURUSD","effective_feature_names":["x"],"genes":[{"indices":[0],"sl_pips":10.0,"fitness":2.5}]}"#;
        let canonical = br#"{"genes":[{"features":["x"],"sl_pips":10.0}],"symbol":"EURUSD"}"#;
        assert_eq!(
            portfolio_gene_fingerprint(bytes),
            Some(format!(
                "gene:{:016x}",
                crate::utils::hashing::fnv1a64(canonical)
            ))
        );
    }

    #[test]
    fn unknown_or_malformed_shared_envelopes_never_fall_back_to_legacy_genes() {
        let (legacy, shared) = shared_fixture();
        let mutations: &[fn(&mut Value)] = &[
            |v| v["shared_receipt_schema_version"] = json!(2),
            |v| v["shared_receipt_schema_version"] = Value::Null,
            |v| v["artifact_kind"] = json!("unrecognized.v1"),
            |v| {
                v.as_object_mut().unwrap().remove("artifact_kind");
            },
            |v| {
                v.as_object_mut()
                    .unwrap()
                    .remove("shared_receipt_schema_version");
            },
            |v| v["input_receipt"] = Value::Null,
            |v| v["input_receipt"]["schema_version"] = json!(3),
            |v| v["portfolio"]["schema_version"] = json!(2),
            |v| v["portfolio"]["portfolio_schema_version"] = json!(6),
            |v| v["portfolio"]["portfolio_identity_sha256"] = json!("not-a-hash"),
            |v| v["portfolio"]["unknown_math"] = json!(true),
            |v| v["portfolio"]["search_scope"]["schema_version"] = json!(2),
            |v| v["portfolio"]["search_scope"]["evaluated_window"]["row_end"] = json!(0),
            |v| {
                v["portfolio"]["sizing_evidence"][0]["scope"]["receipt_sha256"] =
                    json!("e".repeat(64))
            },
            |v| v["portfolio"]["sizing_evidence"][0]["forward_test_schema_version"] = json!(4),
            |v| v["portfolio"]["sizing_evidence"][0]["forward_test_artifact_kind"] = json!("wrong"),
            |v| v["portfolio"]["sizing_evidence"] = json!([]),
            |v| v["portfolio"]["cost_band"] = json!([]),
            |v| v["portfolio"]["effective_feature_names"] = json!(["duplicate", "duplicate"]),
            |v| v["portfolio"]["genes"][0] = Value::Null,
        ];
        for (case, mutate) in mutations.iter().enumerate() {
            let mut changed = shared.clone();
            mutate(&mut changed);
            let bytes = serde_json::to_vec(&changed).unwrap();
            assert!(portfolio_gene_fingerprint(&bytes).is_none(), "case {case}");
            assert!(gene_rule_fingerprints(&bytes).is_empty(), "case {case}");
            // A misleading top-level legacy body cannot authorize fallback.
            changed["genes"] = legacy["genes"].clone();
            let bytes = serde_json::to_vec(&changed).unwrap();
            assert!(
                portfolio_gene_fingerprint(&bytes).is_none(),
                "fallback case {case}"
            );
            assert!(
                gene_rule_fingerprints(&bytes).is_empty(),
                "fallback case {case}"
            );
        }
        let bare_body = serde_json::to_vec(&shared["portfolio"]).unwrap();
        assert!(portfolio_gene_fingerprint(&bare_body).is_none());
        assert!(gene_rule_fingerprints(&bare_body).is_empty());
        let duplicate = serde_json::to_string(&shared).unwrap().replacen(
            '{',
            "{\"shared_receipt_schema_version\":1,",
            1,
        );
        assert!(portfolio_gene_fingerprint(duplicate.as_bytes()).is_none());
        assert!(gene_rule_fingerprints(duplicate.as_bytes()).is_empty());
    }

    #[test]
    fn retired_rules_learn_compact_artifact_rules_without_search_dependency() {
        static NEXT_FIXTURE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let sequence = NEXT_FIXTURE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "neoethos-compact-retired-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir(&root).unwrap();
        let (_, shared) = shared_fixture();
        let bytes = serde_json::to_vec(&shared).unwrap();
        let artifact_path = root.join("hash-only.research.live_portfolio.json");
        std::fs::write(&artifact_path, &bytes).unwrap();
        std::fs::write(
            root.join(RetiredRules::FILE_NAME),
            serde_json::to_vec(&json!([
                {"portfolioPath": artifact_path.to_str().unwrap()}
            ]))
            .unwrap(),
        )
        .unwrap();
        let retired = RetiredRules::load_from_data_dir(&root);
        assert_eq!(retired.entries, 1);
        assert_eq!(retired.unreadable_entries, 0);
        assert_eq!(retired.len(), 1);
        assert!(retired.contains(&gene_rule_fingerprints(&bytes)[0]));
        std::fs::remove_dir_all(&root).unwrap();
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
