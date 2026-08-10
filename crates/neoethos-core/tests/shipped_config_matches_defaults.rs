//! A default that contradicts the config a run actually reads is how this
//! project loses eight months.
//!
//! # What the previous version of this test did, and why it was not enough
//!
//! It guarded **3 keys of 390**, all inside `models.discovery_runtime`. It
//! **skipped absent keys by design** — which is precisely the 150-missing-keys
//! defect, since an absent key is exactly how a file silently takes a default
//! nobody looked at. It never compared the two shipped files against each
//! other. And `shipped_configs()` could not reach `%LOCALAPPDATA%`, so it never
//! looked at **the only file a run reads**.
//!
//! Its own motivating example proves the gap: `prefilter_top_k: 50` is live in
//! the operator's store right now, and this test was green.
//!
//! # What it does now
//!
//! 1. **No unknown keys.** Every key in the repo config and in the live store
//!    must be a field of `Settings`. This catches tombstones — fields deleted
//!    from the code whose keys sat on in the operator's file for months
//!    (`export_onnx`, `news_kill_window_min`, `news_lookahead_minutes`,
//!    `perplexity_enabled`) — and typos, which today parse, save, and report
//!    "saved (verbatim)".
//! 2. **Pinned keys may not diverge unregistered.** For the gate and
//!    search-shaping keys, a value that differs from the code default must
//!    appear in [`ROOT_REGISTERED`] or [`LIVE_REGISTERED`] with the exact
//!    value and the reason. A divergence then stops being drift and becomes a
//!    dated decision that neither side can move without editing this file.
//! 3. **The live store is read when present.** Its divergences are named here
//!    in the same table as the repo's.
//!
//! The desktop seed is no longer compared: it is GENERATED from
//! `Settings::default()` (see `generated_seed_is_current.rs`), so it cannot
//! disagree with the defaults and there is nothing left to compare.
//!
//! # Why the registry stores the FILE's value and not the DEFAULT's
//!
//! Deliberately. The default side is guarded by the generator test, which
//! fails on any `Default` change. This test guards the file side. Storing a
//! default here would mean two places to update for one change, which is the
//! shape of the bug, not the fix.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use neoethos_core::config::{DiscoveryRuntimeConfig, Settings};

/// Keys whose value selects **what is searched** or **what is refused**, as
/// opposed to how fast it happens. A silent divergence on any of these means
/// the run the operator thinks they configured is not the run that happened.
///
/// Adding a path here is how you say "this one must never drift again".
const PINNED: &[&str] = &[
    // --- export and consistency gates -------------------------------------
    "models.require_walkforward_for_export",
    "models.prop_firm_min_pass_rate",
    "models.discovery_runtime.prop_firm_gate.pass_rate",
    "models.regime_router_enabled",
    "system.multi_resolution_enabled",
    "models.l1_feature_selection_enabled",
    "models.l1_feature_selection_per_regime",
    "risk.challenge_mode",
    "risk.max_trades_per_day_enabled",
    "models.tree_device_preference",
    // --- what the search is allowed to see --------------------------------
    "models.discovery_runtime.prefilter_top_k",
    "models.discovery_runtime.prefilter_insample_frac",
    "models.discovery_runtime.prefilter_min_per_timeframe",
    "models.discovery_runtime.adaptive_thresholds",
    "models.data_runtime.normalize_features",
    "models.cpcv_max_rows",
    // --- money ------------------------------------------------------------
    "models.prop_search_min_payoff_ratio",
    "models.prop_search_device",
    "risk.daily_drawdown_limit",
    "risk.total_drawdown_limit",
    "risk.max_portfolio_risk",
    "risk.atr_stop_multiplier",
    "risk.min_risk_reward",
    "risk.commission_per_lot_is_per_side",
    "risk.trailing_enabled",
];

/// Free-form maps whose keys are data, not schema. Flattening into them would
/// produce paths that are correctly absent from `Settings::default()`.
const OPAQUE_MAPS: &[&str] = &[
    "models.hpo_trials_by_model",
    "models.max_epochs_by_model",
    "models.model_param_overrides",
    "models.prop_search_max_rows_by_tf",
];

/// Divergences in the repo's own `config.yaml` that are DECIDED, not drifted.
/// `(path, the value the file must carry, why)`.
const ROOT_REGISTERED: &[(&str, &str, &str)] = &[
    (
        "models.require_walkforward_for_export",
        "false",
        "2026-06-06 regime-diversity mandate, quoted verbatim in the file: a hard all-period OOS \
         export gate kills regime specialists, which is what the library is collecting. \
         Walk-forward still RUNS and is recorded per strategy. Default is true. STILL OPEN — \
         docs/audit-status-2026-08-09.md #322 asks whether false is the right posture and says \
         it needs an OOS measurement.",
    ),
    (
        "models.prop_firm_min_pass_rate",
        "0.0",
        "Same 2026-06-06 mandate: 0.65 -> 0.40 -> 0.0, where 0.0 is RANKING-ONLY (the window \
         pass-rate still orders candidates, it kills none). Default is 0.40.",
    ),
    (
        "models.regime_router_enabled",
        "true",
        "Shipped on against a Default of false. Enabling a router is the stricter, more \
         selective side, so the safer value wins here and the disagreement is recorded rather \
         than reconciled.",
    ),
    (
        "system.multi_resolution_enabled",
        "false",
        "The seed comment calls the true path 'the pre-GA wall that stopped combos completing \
         on laptop AND VPS' — every combo serially reloaded M1's 5.27M rows and fed M1 noise \
         into coarse-TF strategies. Default is true, so a key-less install RE-CREATES the wall. \
         The operator's live store has true.",
    ),
    (
        "risk.challenge_mode",
        "true",
        "UNWIRED — RiskManager has no production constructor and fills its challenge targets \
         from a hardcoded FTMO_STANDARD. Default is false; both repo files shipped true, i.e. \
         an armed mode that does not exist. RETAINED AS INTENT: deleting it deletes the \
         prop-firm goal, not garbage.",
    ),
    (
        "risk.daily_drawdown_limit",
        "0.08",
        "CORRECTED 2026-08-10 from 0.10000000149011612 — an f32 value widened to f64, i.e. \
         machine-written, not chosen. 0.08 is NONE_OWN_MONEY.daily_dd_stop_trading_pct, which \
         is the right source because this file runs `preset: none`. NOT 0.04 — that is FTMO's \
         number and belongs in the live store, which runs `preset: ftmo`. Registered because \
         if the Default is preset-derived these agree, and if it is not, the disagreement is \
         a decision rather than drift.",
    ),
    (
        "risk.total_drawdown_limit",
        "0.14",
        "CORRECTED 2026-08-10 from 0.20000000298023224 — same f32 fingerprint, same writer. \
         0.14 = NONE_OWN_MONEY.max_overall_drawdown_pct 0.20 x the 0.7 buffer. The live store's \
         0.07 is FTMO's 0.10 x 0.7 and is correct there. The two files were answering for two \
         different presets and nothing said so.",
    ),
    (
        "risk.max_portfolio_risk",
        "0.34",
        "The risky profile's portfolio cap. NOTE the live store carries 0.0, which under the \
         current reading means NO CAP AT ALL rather than 'no risk' — that ambiguity is being \
         turned into a loud startup error naming both readings, never a silent correction.",
    ),
    (
        "risk.trailing_enabled",
        "true",
        "UNWIRED / SHADOWED DUPLICATE. The search reads models.exit_policy.trailing_enabled; \
         this RiskConfig copy is what the operator's live store sets, and live execution trails \
         unconditionally with no config recipient at all. Marked, not deleted — deleting it \
         would turn a visibly wrong value into an invisible hardcode on the path that spends \
         real money.",
    ),
];

/// Divergences in `%LOCALAPPDATA%\neoethos\config.yaml` — the only file a run
/// reads. Every entry here is something the OPERATOR chose or inherited, and
/// every one is surfaced by `scripts/migrate_live_config.ps1` for him to
/// sanction. Nothing in this list is corrected by code.
const LIVE_REGISTERED: &[(&str, &str, &str)] = &[
    (
        "models.prop_search_min_payoff_ratio",
        "0.0",
        "MONEY. 0.0 means THE PAYOFF FLOOR IS OFF — the state that let the search buy trade \
         volume with no payoff margin. Default is now 2.0 (reject any strategy whose realized \
         average win is under 2x its average loss). His file predates the floor.",
    ),
    (
        "models.discovery_runtime.prefilter_top_k",
        "50",
        "MONEY-ADJACENT. At 50 the base feature set collapses from 217 columns to roughly 64 \
         and the SMC, session and footprint families die first. This is the exact key the old \
         version of this test was written to protect, and it has been live at 50 the whole \
         time because that test could not see this file.",
    ),
    (
        "models.discovery_runtime.prefilter_insample_frac",
        "0.7",
        "70/30 in-sample split against a Default of 0.8. Operator tuning; reported, not \
         changed.",
    ),
    (
        "models.require_walkforward_for_export",
        "false",
        "MONEY. The out-of-sample EXPORT GATE is OFF. Matches the repo's 2026-06-06 mandate, \
         so this is consistent rather than drifted — but it is the gate that decides whether an \
         in-sample-overfit strategy can reach live money, so it is listed with the money items.",
    ),
    (
        "models.prop_firm_min_pass_rate",
        "0.0",
        "Ranking-only, per the same mandate. Consistent with the repo profile.",
    ),
    (
        "risk.max_portfolio_risk",
        "0.0",
        "MONEY. Under today's reading this is NO CAP AT ALL, not 'no risk'. Being turned into \
         a loud startup error naming both readings rather than silently re-interpreted, because \
         picking either meaning changes how much of the account can be at risk at once.",
    ),
    (
        "risk.trailing_enabled",
        "true",
        "MONEY. This is the ORPHANED RiskConfig copy. He also hand-tuned \
         trailing_atr_multiplier: 0.4 and trailing_be_trigger_r: 0.1 here — deliberate values \
         that move nothing, because the search reads models.exit_policy and live execution \
         trails unconditionally.",
    ),
    (
        "models.discovery_runtime.adaptive_thresholds",
        "false",
        "The static coarse-threshold ladder documents itself as calibrated for z-score \
         normalised features, while raw magnitudes span ~1e5:1 — so a 0.35 threshold is \
         unreachable for an ATR term and always-on for an RSI term. Default is now true. \
         Changing it changes WHAT IS SEARCHED, so it is reported, not corrected.",
    ),
    (
        "models.data_runtime.normalize_features",
        "false",
        "The other half of the same defect: the GA's 5:1 weight ladder cannot reorder feature \
         terms that differ by 1e5, so a multi-indicator gene equalled its single \
         largest-magnitude term. Default is now true. Anything fitted on the raw cube must be \
         retrained, which is why this is his decision and not a migration's.",
    ),
    (
        "models.cpcv_max_rows",
        "0",
        "0 = UNBOUNDED. CPCV is the heaviest serial validation in the run; Default is 200000. \
         Reported because 'unbounded' is a runtime and memory decision, not a correctness one.",
    ),
    (
        "system.multi_resolution_enabled",
        "true",
        "On, matching the Default, against the repo profile's false. Recorded on both sides so \
         the disagreement is visible from either end.",
    ),
    (
        "models.regime_router_enabled",
        "true",
        "Matches the repo profile; Default is false.",
    ),
    (
        "risk.challenge_mode",
        "true",
        "UNWIRED — see the root entry. Retained as intent.",
    ),
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root is two levels above the crate manifest")
        .to_path_buf()
}

fn root_config() -> PathBuf {
    repo_root().join("config.yaml")
}

/// The operator's live store — **the only file a run reads**. Absent on CI and
/// on a fresh machine, which is why every check that uses it is conditional
/// rather than skipped-by-design.
fn live_store() -> Option<PathBuf> {
    let base = std::env::var_os("LOCALAPPDATA")?;
    let p = PathBuf::from(base).join("neoethos").join("config.yaml");
    p.exists().then_some(p)
}

fn flatten(prefix: &str, v: &serde_yaml_ng::Value, out: &mut BTreeMap<String, serde_yaml_ng::Value>) {
    match v {
        serde_yaml_ng::Value::Mapping(m) if !m.is_empty() && !OPAQUE_MAPS.contains(&prefix) => {
            for (k, val) in m {
                let key = k.as_str().map(str::to_owned).unwrap_or_else(|| format!("{k:?}"));
                let path = if prefix.is_empty() { key } else { format!("{prefix}.{key}") };
                flatten(&path, val, out);
            }
        }
        other => {
            out.insert(prefix.to_string(), other.clone());
        }
    }
}

fn leaves_of_file(path: &Path) -> BTreeMap<String, serde_yaml_ng::Value> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("config {} is unreadable: {e}", path.display()));
    let doc: serde_yaml_ng::Value = serde_yaml_ng::from_str(&text)
        .unwrap_or_else(|e| panic!("config {} is not valid YAML: {e}", path.display()));
    let mut out = BTreeMap::new();
    flatten("", &doc, &mut out);
    out
}

fn default_leaves() -> BTreeMap<String, serde_yaml_ng::Value> {
    let v = serde_yaml_ng::to_value(Settings::default()).expect("Settings must serialise");
    let mut out = BTreeMap::new();
    flatten("", &v, &mut out);
    out
}

/// Numeric-tolerant equality, so `0.8` and `0.80` do not read as a drift.
fn same(a: &serde_yaml_ng::Value, b: &serde_yaml_ng::Value) -> bool {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) => (x - y).abs() < 1e-12,
        _ => a == b,
    }
}

fn registered<'a>(
    table: &'a [(&'a str, &'a str, &'a str)],
    path: &str,
) -> Option<(serde_yaml_ng::Value, &'a str)> {
    table.iter().find(|(p, _, _)| *p == path).map(|(_, v, why)| {
        let parsed: serde_yaml_ng::Value = serde_yaml_ng::from_str(v)
            .unwrap_or_else(|e| panic!("registry value `{v}` for {path} is not valid YAML: {e}"));
        (parsed, *why)
    })
}

// ---------------------------------------------------------------------------
// 1. No unknown keys. This is what makes a typo and a tombstone loud.
// ---------------------------------------------------------------------------

fn assert_no_unknown_keys(path: &Path, remedy: &str) {
    let defaults = default_leaves();
    let file = leaves_of_file(path);
    let unknown: Vec<&String> = file
        .keys()
        .filter(|k| {
            !defaults.contains_key(*k)
                // a key that IS an opaque map is fine even though its contents
                // never appear in the defaults
                && !OPAQUE_MAPS.contains(&k.as_str())
                && !OPAQUE_MAPS.iter().any(|m| k.starts_with(&format!("{m}.")))
        })
        .collect();

    assert!(
        unknown.is_empty(),
        "\n{} contains {} key(s) that are NOT fields of Settings:\n{}\n\n\
         Today serde ignores these silently — no struct sets deny_unknown_fields, so a \
         misspelled `trailing_enabeld:` parses, saves, and the raw-YAML editor reports \
         'saved (verbatim)'. That editor is the only route to 364 of the 390 knobs.\n\n\
         Once deny_unknown_fields lands these become a HARD LOAD FAILURE at startup.\n\n\
         {remedy}\n",
        path.display(),
        unknown.len(),
        unknown
            .iter()
            .map(|k| format!("  - {k}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
}

#[test]
fn repo_config_contains_no_key_that_is_not_a_settings_field() {
    assert_no_unknown_keys(
        &root_config(),
        "Remedy: delete the key from config.yaml, or restore the field it names.",
    );
}

#[test]
fn the_live_store_contains_no_key_that_is_not_a_settings_field() {
    let Some(path) = live_store() else {
        return; // no live store on this machine — nothing to check
    };
    assert_no_unknown_keys(
        &path,
        "Remedy: run `pwsh scripts/migrate_live_config.ps1` and accept the DROP section. It \
         backs the file up first, shows the full diff before writing, refuses to run \
         unattended, and lists money-path changes separately. Do NOT hand-edit the live store.",
    );
}

// ---------------------------------------------------------------------------
// 2. Pinned keys: a divergence must be a registered decision.
// ---------------------------------------------------------------------------

fn assert_pinned(path: &Path, table: &[(&str, &str, &str)], which: &str) {
    let defaults = default_leaves();
    let file = leaves_of_file(path);
    let mut failures: Vec<String> = Vec::new();

    for key in PINNED {
        let Some(want_default) = defaults.get(*key) else {
            // The path is not a field at all. Reported by its own test below.
            continue;
        };
        let Some(got) = file.get(*key) else {
            continue; // absent: the default applies, and it is the default by construction
        };
        if same(got, want_default) {
            continue;
        }
        match registered(table, key) {
            Some((expected, _why)) if same(got, &expected) => {}
            Some((expected, _)) => failures.push(format!(
                "{key}: file has {got:?}, but the registry in this test records {expected:?}. \
                 The value moved without the decision moving. Update the registry entry AND say \
                 why, or put the value back."
            )),
            None => failures.push(format!(
                "{key}: file has {got:?}, the code default is {want_default:?}, and this \
                 divergence is NOT REGISTERED.\n    This is exactly the shape of the defect \
                 being closed: two plausible numbers, both documented, nothing comparing them. \
                 Either make them agree, or add an entry to {which} in this file stating the \
                 value and the reason."
            )),
        }
    }

    assert!(
        failures.is_empty(),
        "\n{}\n\n{}\n",
        path.display(),
        failures.join("\n\n")
    );
}

#[test]
fn repo_config_divergences_from_the_defaults_are_registered_decisions() {
    assert_pinned(&root_config(), ROOT_REGISTERED, "ROOT_REGISTERED");
}

/// The check nothing performed before: the live store is the ONLY file a run
/// reads, and until now it was the only one nothing compared.
#[test]
fn live_store_divergences_from_the_defaults_are_registered_decisions() {
    let Some(path) = live_store() else {
        return;
    };
    assert_pinned(&path, LIVE_REGISTERED, "LIVE_REGISTERED");
}

// ---------------------------------------------------------------------------
// 3. The tables themselves must stay alive.
// ---------------------------------------------------------------------------

#[test]
fn every_pinned_path_is_a_real_field() {
    let defaults = default_leaves();
    let dead: Vec<&&str> = PINNED.iter().filter(|k| !defaults.contains_key(**k)).collect();
    assert!(
        dead.is_empty(),
        "these pinned paths are not fields of Settings, so they guard nothing — the field was \
         renamed or deleted and this list was not updated: {dead:?}"
    );
}

#[test]
fn every_registered_path_is_pinned() {
    let mut orphans = Vec::new();
    for (table, name) in [(ROOT_REGISTERED, "ROOT_REGISTERED"), (LIVE_REGISTERED, "LIVE_REGISTERED")]
    {
        for (p, _, _) in table {
            if !PINNED.contains(p) {
                orphans.push(format!("{name}: {p}"));
            }
        }
    }
    assert!(
        orphans.is_empty(),
        "these registry entries name paths that are not in PINNED, so nothing consults them — a \
         registered decision that guards nothing is worse than no entry, because it reads as \
         coverage: {orphans:?}"
    );
}

// ---------------------------------------------------------------------------
// 4. The specific value that started all of this.
// ---------------------------------------------------------------------------

#[test]
fn prefilter_top_k_default_is_the_shipped_240_not_the_old_50() {
    let d = DiscoveryRuntimeConfig::default();
    assert_eq!(
        d.prefilter_top_k, 240,
        "the code default for prefilter_top_k must be 240. At 50 the base feature set collapses \
         from 217 columns to roughly 64 and the SMC, session and footprint families are the \
         first to go. NOTE the operator's live store still carries 50 — that is his to change, \
         and scripts/migrate_live_config.ps1 puts it in front of him."
    );
}

#[test]
fn the_repo_config_exists_and_parses() {
    let p = root_config();
    assert!(p.exists(), "repo config missing: {}", p.display());
    let leaves = leaves_of_file(&p);
    assert!(!leaves.is_empty(), "{} parsed to nothing", p.display());
}
