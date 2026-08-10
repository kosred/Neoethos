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
//! 4. **The repo profile is collapsed to OVERRIDES ONLY**
//!    (`the_repo_profile_carries_only_its_overrides`). A key whose value
//!    already equals the compiled default is deleted from the file — which
//!    cannot change any effective value, because the loader supplies the
//!    identical number from `Default`. What is left is exactly the set of
//!    disagreements, so a value can only appear in a file when it means
//!    something. This is the other half of the collapse: the desktop seed is
//!    generated, this one is reduced.
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
    // `risk.trailing_enabled` removed 2026-08-10 with the field itself (#206):
    // the trail is `models.exit_policy.trailing_enabled` and there is no second
    // copy left to pin. A store still carrying the old key is named at WARN by
    // `RETIRED_KEYS`.
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
        "The RISKY ladder's concurrent-risk cap, and correct here because this file is \
         `trading_mode: risky` with `preset: none`. Since 2026-08-10 it is also the SEED for \
         that mode, so the file and the code now agree by construction rather than by \
         coincidence. Under a prop firm the seed is the preset's daily stop instead — carrying \
         0.34 there is 8.5x FTMO's daily limit, which one correlated move spends in full. \
         See portfolio_cap_follows_the_mode.rs.",
    ),
    (
        "models.tree_device_preference",
        "gpu",
        "2026-08-10. Default is `auto`. The repo profile asks for the card explicitly, which is \
         the standing GPU invariant (#35): a machine WITH a card must run on it or fail loudly, \
         never quietly fall back to CPU and report a number that took 20x longer. Registered \
         rather than reconciled because `auto` and `gpu` are NOT the same request — the \
         refuters overturned merging this with `models.tree_runtime.device`, and the two keys \
         stay distinct.",
    ),
    (
        "models.l1_feature_selection_enabled",
        "true",
        "2026-08-10. Default is false. This changes WHAT IS SEARCHED — L1 selection prunes the \
         feature set before the GA sees it — so it is recorded, not reconciled. The value has \
         been in this profile throughout the runs the current results came from; flipping it to \
         match the default would silently change the search on the next run, which is precisely \
         what this table exists to prevent.",
    ),
    (
        "models.l1_feature_selection_per_regime",
        "true",
        "2026-08-10. Default is false. The per-regime half of the same decision; see above. \
         Both move together — enabling selection globally while disabling it per regime is a \
         combination nothing in the profile intended.",
    ),
];

/// Divergences in `%LOCALAPPDATA%\neoethos\config.yaml` — the only file a run
/// reads. Every entry here is something the OPERATOR chose or inherited, and
/// every one is surfaced by `neoethos-cli config normalize` for him to
/// sanction. Nothing in this list is corrected by code.
const LIVE_REGISTERED: &[(&str, &str, &str)] = &[
    (
        "models.discovery_runtime.prefilter_insample_frac",
        "0.7",
        "70/30 in-sample split against a Default of 0.8. Operator tuning; reported, not \
         changed.",
    ),
    (
        "risk.max_portfolio_risk",
        "0.04",
        "MONEY, and no longer a divergence — it is the SEED for this file's ftmo preset under \
         prop_firm, so it equals the default and is listed only because money keys always are. \
         History worth keeping: this store carried 0.0 until 2026-08-10, and so did the \
         DEFAULT, which was the actual finding — on a knob named max_ that reads as NO CAP AT \
         ALL, shipped on every install and chosen by nobody. It was then briefly set to 0.34, \
         which is the RISKY ladder's number and 8.5x FTMO's daily stop; the operator caught it \
         the same day. The cap is now seeded per preset AND per trading_mode, at the daily \
         stop, because if correlated positions stop out together the day's loss IS the total \
         open risk. See portfolio_cap_follows_the_mode.rs.",
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
    (
        "models.l1_feature_selection_enabled",
        "true",
        "2026-08-10. Default is false. Matches the repo profile, so this is consistent rather \
         than drifted — recorded on both sides so the decision is visible from either end. \
         Changes WHAT IS SEARCHED; not corrected by code.",
    ),
    (
        "models.l1_feature_selection_per_regime",
        "true",
        "2026-08-10. Default is false. The per-regime half of the same decision; matches the \
         repo profile.",
    ),
    (
        "models.tree_device_preference",
        "gpu",
        "2026-08-10. Default is `auto`. His box has a 3090 and the standing GPU invariant (#35) \
         is that a card present means the card runs it or the run fails loudly. `gpu` is the \
         stricter, louder side of `auto`, so the safer value is already the one in the file and \
         nothing is being raised here.",
    ),
    // `risk.total_drawdown_limit` USED TO BE REGISTERED HERE, at "0.07" against
    // a default of 0.07000000104308128 — the same number, differing only by an
    // f32 constant widened to f64 inside `RiskConfig::default()`. The entry
    // said the fingerprint should stay on the record rather than be "fixed".
    //
    // DELETED 2026-08-10 (#214). The fingerprint was fixed at its source
    // instead: `PropFirmConstraints::buffered_total_drawdown_limit()` widens
    // once and rounds, so the compiled default IS 0.07 and there is no
    // divergence left for this table to hold. A registered decision that can
    // never fire reads as coverage — the test's own words, two functions down
    // in `every_registered_path_is_pinned`. The history lives in that helper's
    // doc comment, where a reader looking at the number will find it.
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

/// `the_repo_profile_carries_only_its_overrides` REWRITES the repo config, and
/// three other tests in this same binary read it — on parallel threads. Without
/// a lock a reader can observe a half-written file and fail for a reason that
/// has nothing to do with what it guards. Poisoning is ignored deliberately: a
/// panicking assert must not convert the rest of the suite into lock errors.
static ROOT_CONFIG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock_root_config() -> std::sync::MutexGuard<'static, ()> {
    ROOT_CONFIG_LOCK.lock().unwrap_or_else(|e| e.into_inner())
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

/// `min_sharpe` -> `minSharpe`. One segment only.
fn snake_to_camel(segment: &str) -> String {
    let mut out = String::with_capacity(segment.len());
    let mut upper_next = false;
    for ch in segment.chars() {
        if ch == '_' {
            upper_next = true;
        } else if upper_next {
            out.extend(ch.to_uppercase());
            upper_next = false;
        } else {
            out.push(ch);
        }
    }
    out
}

/// The canonical (serialised) path for a key as it is spelled IN A FILE, or
/// `None` when no field accepts that spelling.
///
/// ## Why this exists — it nearly cost five money-path thresholds
///
/// These checks compare file keys against the paths of `Settings::default()`
/// SERIALISED. That silently assumes the serialised spelling is the only
/// accepted one. It is not: `PromotionGateConfig` is
/// `#[serde(rename_all = "camelCase")]` with a snake_case `alias` on every
/// field, precisely so the operator's YAML reads like the rest of the file.
/// Both spellings deserialise to the same field.
///
/// So `models.promotion_gate.min_sharpe` — a real, loaded, ENFORCED promotion
/// threshold — did not match the serialised `...minSharpe` and was reported as
/// "NOT a field of Settings", whose printed remedy is "delete the key from
/// config.yaml". Following that remedy would have deleted the five thresholds
/// the promotion gate is judged by, from the profile, on the grounds that they
/// do not exist. They do.
///
/// Resolving the alias mechanically beats an exception list: any future field
/// that follows the same camelCase-with-snake-alias convention is covered
/// without anyone remembering to add it here.
/// `rename_all` is per STRUCT, not per path: `ModelsConfig` keeps
/// `promotion_gate` in snake_case while `PromotionGateConfig` renames its own
/// fields to camelCase, so the real default path is
/// `models.promotion_gate.minSharpe` — neither all-snake nor all-camel.
/// Each segment is therefore resolved independently, and with paths at most a
/// few levels deep, enumerating the 2^n spellings is the honest way to say
/// "either spelling, at every level".
fn canonical_path(defaults: &BTreeMap<String, serde_yaml_ng::Value>, path: &str) -> Option<String> {
    if defaults.contains_key(path) {
        return Some(path.to_string());
    }
    let segments: Vec<&str> = path.split('.').collect();
    // Guard against a pathological path turning this into a huge search.
    if segments.len() > 8 {
        return None;
    }
    for mask in 0u32..(1u32 << segments.len()) {
        let candidate = segments
            .iter()
            .enumerate()
            .map(|(i, seg)| {
                if mask & (1 << i) == 0 {
                    (*seg).to_string()
                } else {
                    snake_to_camel(seg)
                }
            })
            .collect::<Vec<_>>()
            .join(".");
        if defaults.contains_key(&candidate) {
            return Some(candidate);
        }
    }
    None
}

/// The compiled default for a key spelled as a file spells it.
fn default_for<'a>(
    defaults: &'a BTreeMap<String, serde_yaml_ng::Value>,
    path: &str,
) -> Option<&'a serde_yaml_ng::Value> {
    canonical_path(defaults, path).and_then(|p| defaults.get(&p))
}

/// Is this file key addressable at all — a field under some accepted spelling,
/// or the contents of a free-form map?
fn is_known_key(defaults: &BTreeMap<String, serde_yaml_ng::Value>, path: &str) -> bool {
    canonical_path(defaults, path).is_some()
        || OPAQUE_MAPS.contains(&path)
        || OPAQUE_MAPS.iter().any(|m| path.starts_with(&format!("{m}.")))
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
        .filter(|k| !is_known_key(&defaults, k))
        .collect();

    assert!(
        unknown.is_empty(),
        "\n{} contains {} key(s) that are NOT fields of Settings:\n{}\n\n\
         `deny_unknown_fields` is on, so at startup these are a HARD LOAD FAILURE unless the \
         key is listed in `load_seal::RETIRED_KEYS` (which names it at WARN and ignores it). \
         Before this guard existed a misspelled `trailing_enabeld:` parsed, saved, and the \
         raw-YAML editor reported 'saved (verbatim)'.\n\n\
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

/// The end-to-end contract, asked of the component that actually enforces it.
///
/// The key-by-key check above compares against `Settings::default()` serialised
/// and therefore cannot see two things the real loader does: `RETIRED_KEYS`
/// (a key with no field that is deliberately accepted, named at WARN, and
/// ignored) and `#[serde(alias)]`. That is the right check for the REPO
/// profile, which we control and keep clean. It is the wrong check for the
/// operator's live store, which legitimately carries retired keys from every
/// release he has ever run.
///
/// So the live store is judged by the only question that matters: **does the
/// app open?** This cannot drift from the loader, because it IS the loader.
fn assert_the_loader_accepts(path: &Path, remedy: &str) {
    if let Err(err) = Settings::from_yaml(path) {
        panic!(
            "\n{} DOES NOT LOAD.\n\n{err:?}\n\n\
             This is a startup failure, not a lint: the app will not open with this file. A key \
             that is neither a field nor a `RETIRED_KEYS` entry is refused by \
             `deny_unknown_fields` — which is the intended behaviour for a typo, and a \
             regression for a key some earlier release wrote.\n\n{remedy}\n",
            path.display(),
        );
    }
}

#[test]
fn repo_config_contains_no_key_that_is_not_a_settings_field() {
    let _guard = lock_root_config();
    assert_no_unknown_keys(
        &root_config(),
        "Remedy: delete the key from config.yaml, or restore the field it names.",
    );
}

/// The operator's store must still OPEN THE APP.
///
/// Deliberately not `assert_no_unknown_keys`: his file carries 55 keys with no
/// field, and every one of them is a `RETIRED_KEYS` entry that the loader names
/// at WARN and ignores — which is the whole point of that table. Asserting
/// "no key without a field" here would demand he delete lines the loader is
/// explicitly designed to tolerate, and the printed remedy would send him to a
/// migration script to fix a file that was never broken.
///
/// What must never happen is a key that is neither. That is a refused load and
/// a dark app, and it is what this asks.
#[test]
fn the_live_store_still_loads() {
    let Some(path) = live_store() else {
        return; // no live store on this machine — nothing to check
    };
    assert_the_loader_accepts(
        &path,
        "Remedy: add the key to `load_seal::RETIRED_KEYS` in config.rs if a past release wrote \
         it (it will then be named at WARN and ignored), or restore the field it names. If it \
         is a typo, run `neoethos-cli config normalize --write`, which backs the file up first, \
         prints every override beside the default it shadows, and restores the backup unless \
         the rewritten file reloads to identical settings. Do NOT hand-edit the live store.",
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
        let Some(want_default) = default_for(&defaults, key) else {
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
    let _guard = lock_root_config();
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
         and `neoethos-cli config normalize` puts it in front of him."
    );
}

#[test]
fn the_repo_config_exists_and_parses() {
    let _guard = lock_root_config();
    let p = root_config();
    assert!(p.exists(), "repo config missing: {}", p.display());
    let leaves = leaves_of_file(&p);
    assert!(!leaves.is_empty(), "{} parsed to nothing", p.display());

    // Valid YAML is not the bar. `the_repo_profile_carries_only_its_overrides`
    // REWRITES this file, and a developer reaches it with
    // `$env:CONFIG_FILE = 'config.yaml'` — the one supported escape hatch. A
    // rewrite that produced a file the loader refuses would break that hatch,
    // and YAML-parses-fine is exactly the check that would not notice.
    assert_the_loader_accepts(
        &p,
        "The repo profile must load through the same seal as any other config. If the collapse \
         produced this, the collapse is wrong — do not hand-patch the file.",
    );
}

// ---------------------------------------------------------------------------
// 5. THE COLLAPSE. The repo profile carries OVERRIDES ONLY.
// ---------------------------------------------------------------------------
//
// Wave 1 gave the project one LOAD PATH. This is the one FILE.
//
// The repo `config.yaml` restated 383 values, of which the overwhelming
// majority were simply the code default written out a second time. Every one of
// those was a place a `Default` change could be contradicted by a file nobody
// re-read — which is exactly how `prefilter_top_k` shipped at 50 in one file
// and 240 in another for eight months. The seed file was solved by generating
// it (`generated_seed_is_current.rs`). This is the other half: the repo profile
// is reduced to the keys that actually DISAGREE with the code, so a value can
// only appear in a file when it means something.
//
// # Why this test may DELETE lines from config.yaml but may never CHANGE one
//
// A key is dropped only when its value is byte-for-byte the compiled default
// (numerically, to 1e-12). Dropping it therefore cannot change any effective
// value — the loader supplies the identical number from `Default`. A key whose
// value DIFFERS is kept verbatim, whatever it is, registered or not. This test
// never rewrites a value, never reconciles a disagreement, and never touches
// the operator's live store.

/// Reasons attached to keys the repo profile keeps but which are not on the
/// [`PINNED`] list. [`ROOT_REGISTERED`] carries the reason for the pinned ones;
/// this carries the rest, so that the annotations B wrote into the 935-line
/// file survive the collapse in the one place that is checked.
const ROOT_NOTES: &[(&str, &str)] = &[
    (
        "system.trading_mode",
        "The profile this file exists to describe. NOT merged with models.discovery_mode — that \
         merge was overturned by the refuters and is never to be executed.",
    ),
];

const PROFILE_HEADER: &str = "\
# ============================================================================
# DEVELOPER EXPERIMENT PROFILE — OVERRIDES ONLY. NOT the config a run reads.
# ============================================================================
#
# Generated by crates/neoethos-core/tests/shipped_config_matches_defaults.rs
# (`the_repo_profile_carries_only_its_overrides`). Every key BELOW differs from
# `neoethos_core::config::Settings::default()`. Every key NOT below is the code
# default, deliberately absent so that it cannot contradict one.
#
# THE SCHEME (docs/config-single-source-of-truth.md):
#
#   * The Rust `Default` impls are the SINGLE SOURCE OF DEFAULT VALUES.
#   * desktop/src-tauri/resources/config.yaml is GENERATED from those defaults
#     and fails the build if it drifts.
#   * The operator's store (%LOCALAPPDATA%\\neoethos\\config.yaml) is the ONLY
#     file a run reads, and carries HIS overrides only. Convert it with
#     `neoethos-cli config normalize --write` — never by hand.
#   * THIS FILE is read by NOTHING unless you point $CONFIG_FILE at it.
#
# HOW TO RUN A LOCAL EXPERIMENT WITH A DIFFERENT SETTING:
#
#     Copy-Item config.yaml my-experiment.yaml     # edit the copy
#     $env:CONFIG_FILE = 'my-experiment.yaml'
#     cargo run -p neoethos-cli -- discover
#
# $CONFIG_FILE is the FIRST branch of the resolution order and the ONLY
# supported way to point a run at a different file. The bare relative
# \"config.yaml\" fallback is DELETED: until 2026-08-10 the same binary read a
# different config depending on the directory it was started from, so simply
# being in the repo root silently turned the OOS export gate off. A run now
# logs, by name, which file it opened — or that it opened none.
#
# TO ADD AN OVERRIDE: add the key here. If it is on the PINNED list in the
# generating test you must also register the value AND the reason in
# ROOT_REGISTERED, or the test fails naming your key. To REMOVE an override,
# delete the line; the code default takes over.
# ============================================================================
";

/// Insert `value` at a dotted path into a YAML mapping, creating intermediate
/// mappings. Insertion order is the caller's, so a sorted key list yields a
/// deterministic, section-grouped file.
fn insert_dotted(root: &mut serde_yaml_ng::Mapping, dotted: &str, value: serde_yaml_ng::Value) {
    let mut parts: Vec<&str> = dotted.split('.').collect();
    let leaf = parts.pop().expect("a dotted path has at least one segment");
    let mut node: &mut serde_yaml_ng::Mapping = root;
    for part in parts {
        let key = serde_yaml_ng::Value::String(part.to_string());
        if !node.contains_key(&key) {
            node.insert(
                key.clone(),
                serde_yaml_ng::Value::Mapping(serde_yaml_ng::Mapping::new()),
            );
        }
        node = node
            .get_mut(&key)
            .and_then(|v| v.as_mapping_mut())
            .expect("intermediate config nodes are mappings");
    }
    node.insert(serde_yaml_ng::Value::String(leaf.to_string()), value);
}

fn reason_for(path: &str) -> Option<&'static str> {
    ROOT_REGISTERED
        .iter()
        .find(|(p, _, _)| *p == path)
        .map(|(_, _, why)| *why)
        .or_else(|| {
            ROOT_NOTES
                .iter()
                .find(|(p, _)| *p == path)
                .map(|(_, why)| *why)
        })
}

/// Wrap `text` into `# ` comment lines under a `#   <path>` heading.
fn comment_block(path: &str, text: &str) -> String {
    let mut out = format!("#   {path}\n");
    let mut line = String::from("#     ");
    for word in text.split_whitespace() {
        if line.len() + word.len() + 1 > 78 {
            out.push_str(line.trim_end());
            out.push('\n');
            line = String::from("#     ");
        }
        line.push_str(word);
        line.push(' ');
    }
    out.push_str(line.trim_end());
    out.push('\n');
    out
}

#[test]
fn the_repo_profile_carries_only_its_overrides() {
    let _guard = lock_root_config();
    let path = root_config();
    let defaults = default_leaves();
    let file = leaves_of_file(&path);

    // A key that is not a field at all is NOT something this test may act on:
    // it cannot be compared to a default, so it can be neither dropped as
    // redundant nor kept as an override. Refuse, and leave the file alone.
    let unknown: Vec<&String> = file
        .keys()
        .filter(|k| !is_known_key(&defaults, k))
        .collect();
    assert!(
        unknown.is_empty(),
        "{} carries {} key(s) that are not fields of Settings, so the profile cannot be \
         collapsed without guessing what they meant. NOTHING HAS BEEN WRITTEN. Delete them (or \
         restore their fields) and re-run:\n{}",
        path.display(),
        unknown.len(),
        unknown.iter().map(|k| format!("  - {k}")).collect::<Vec<_>>().join("\n"),
    );

    let mut kept: Vec<(&String, &serde_yaml_ng::Value)> = Vec::new();
    let mut dropped: Vec<String> = Vec::new();
    for (key, value) in &file {
        // Alias-aware: a snake_case spelling of a camelCase field is the SAME
        // key, and dropping it as "unknown" would delete a live override.
        match default_for(&defaults, key) {
            // Identical to the compiled default: the line says nothing the code
            // does not already say, and removing it cannot change a value.
            Some(d) if same(value, d) => {
                dropped.push(format!("  DROP  {key}: {value:?}  (== the code default)"))
            }
            _ => kept.push((key, value)),
        }
    }

    let mut body = serde_yaml_ng::Mapping::new();
    for &(key, value) in &kept {
        insert_dotted(&mut body, key.as_str(), value.clone());
    }
    let rendered = serde_yaml_ng::to_string(&serde_yaml_ng::Value::Mapping(body))
        .expect("the retained overrides must serialise");
    let rendered = rendered.strip_prefix("---\n").unwrap_or(&rendered);

    let mut reasons = String::from(
        "#\n# WHY EACH KEY BELOW DIVERGES FROM THE CODE DEFAULT\n\
         # (unannotated keys are ordinary developer tuning; the ones with a\n\
         #  reason are decisions that neither side may move alone)\n#\n",
    );
    for &(key, _) in &kept {
        if let Some(why) = reason_for(key.as_str()) {
            reasons.push_str(&comment_block(key, why));
            reasons.push_str("#\n");
        }
    }
    reasons.push_str(
        "# ============================================================================\n",
    );

    let want = format!("{PROFILE_HEADER}{reasons}{rendered}");
    let have = std::fs::read_to_string(&path).unwrap_or_default();
    if have == want {
        return;
    }

    std::fs::write(&path, &want).unwrap_or_else(|e| {
        panic!(
            "the repo profile at {} is not overrides-only AND could not be rewritten: {e}",
            path.display()
        )
    });

    panic!(
        "\n\
         {} was not overrides-only, so it HAS BEEN COLLAPSED and REWRITTEN.\n\
         \n\
         {} key(s) kept (they differ from the code default), {} key(s) dropped.\n\
         Every dropped key held EXACTLY the compiled default, so no effective value moved —\n\
         the loader supplies the identical number from `Settings::default()`. Review `git diff`\n\
         and commit.\n\
         \n\
         {}\n\
         \n\
         This test fails on rewrite by design, the same way generated_seed_is_current.rs does:\n\
         a config file shrinking is something a human reads once. Re-run to confirm green.\n",
        path.display(),
        kept.len(),
        dropped.len(),
        if dropped.is_empty() {
            "  (no key dropped — only the header, ordering or a value's rendering changed)".to_string()
        } else {
            dropped.join("\n")
        },
    );
}

/// A note whose key is not a field guards nothing and reads as coverage.
#[test]
fn every_annotated_path_is_a_real_field() {
    let defaults = default_leaves();
    let dead: Vec<&str> = ROOT_NOTES
        .iter()
        .map(|(p, _)| *p)
        .filter(|p| !defaults.contains_key(*p))
        .collect();
    assert!(
        dead.is_empty(),
        "these ROOT_NOTES paths are not fields of Settings — the field was renamed or deleted \
         and the note was left behind: {dead:?}"
    );
}
