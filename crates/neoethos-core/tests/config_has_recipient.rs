//! Every operator-settable value must reach something that changes behaviour.
//!
//! # Why this test exists
//!
//! A census on 2026-08-04 walked all 408 fields of `Settings` and asked, for
//! each one, "what reads this?". 51 had no answer. They were not dead code in
//! the usual sense — they parsed, validated, persisted, round-tripped through
//! `GET /settings`, and several had a control in the Settings UI. They simply
//! had **no final recipient**. Setting one and watching nothing change was the
//! entire experience.
//!
//! Three shapes were found, worst last:
//!
//! * **PHANTOM** — no consumer anywhere; belongs to a feature never built.
//!   45 of them, e.g. every `perplexity_*` knob but the enable flag, all four
//!   `auto_rescore_*`, `smc_freshness_limit`, `vortex_memory_map`. Deleted.
//! * **UNWIRED** — a consumer mechanism exists but nothing connects the config
//!   to it. `challenge_phase` is the specimen: `PropFirmPhaseRiskDefaults::
//!   for_preset(preset, phase)` takes exactly that string and only tests call
//!   it. Kept, documented, and listed in [`UNWIRED`] below — because wiring
//!   them would change live sizing and must be a deliberate act.
//! * **STORED** — validated, persisted, displayed, then ignored. The purest
//!   defect, because it is indistinguishable from working. `ui_locale` offered
//!   an English/Ελληνικά picker in an app with no i18n layer;
//!   `news_calendar_source` accepted any provider id while the fetcher
//!   hardcoded ForexFactory's URL. Both fixed.
//!
//! # What this test enforces
//!
//! 1. Every `Settings` field is read somewhere outside `config.rs`.
//! 2. The two ledgers below only ever shrink — a new entry needs a code change
//!    here, which is the point: adding a knob nobody reads must be loud.
//! 3. Every field `POST /settings` accepts is actually applied to `Settings`.
//! 4. No UI control is backed by a field in the ledgers — the UI may only
//!    expose knobs that reach a live consumer.
//!
//! Without this, a census that found 51 in six months finds 60 in the next.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Fields kept ON PURPOSE despite having no reader: the mechanism they were
/// written for exists, but connecting it would change trading behaviour, so it
/// must be a deliberate, measured act — not a side effect of tidying.
///
/// **This list may shrink. It must never grow.** Each entry carries the exact
/// mechanism it should reach, so wiring it is a decision, not an excavation.
const UNWIRED: &[(&str, &str)] = &[
    (
        "challenge_phase",
        "domain::prop_firm::PropFirmPhaseRiskDefaults::for_preset(preset, phase) \
         takes this string; only its own #[cfg(test)] block calls it",
    ),
    (
        "recovery_mode_enabled",
        "domain::risk::RiskManager::update_recovery_state flips recovery_mode \
         from drawdown alone and consults no toggle; RiskManager has no \
         production constructor at all",
    ),
    (
        "rl_population_size",
        "evolution::neat_impl::NeatTrainer::population_size is hardcoded (96 \
         default / 24 floor); this field's default of 5 would collapse it",
    ),
];

/// Fields whose only reader sits inside `#[cfg(test)]` scaffolding. They are
/// not phantoms — the code that reads them is real — but no production run
/// touches it, so an operator changing them sees nothing.
///
/// **This list may shrink. It must never grow.**
const TEST_FIXTURE_ONLY: &[(&str, &str)] = &[
    (
        "perplexity_enabled",
        "last reader (the legacy egui AppState::new test fixture) was deleted \
         in the 2026-08-08 dead-code purge; with it, core NewsFilter lost its \
         only non-core-test constructor — DELETE-or-WIRE candidate",
    ),
    (
        "news_lookahead_minutes",
        "same — reader deleted with the AppState fixture (2026-08-08 purge)",
    ),
    (
        "news_kill_window_min",
        "same — reader deleted with the AppState fixture (2026-08-08 purge)",
    ),
];

fn workspace_root() -> PathBuf {
    // crates/neoethos-core -> crates -> <workspace root>
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("neoethos-core must live two levels below the workspace root")
        .to_path_buf()
}

fn config_rs() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("config.rs")
}

/// `pub some_field: Type,` inside any `pub struct ... {}` in config.rs.
fn declared_settings_fields(src: &str) -> BTreeMap<String, String> {
    let mut fields = BTreeMap::new();
    let mut current: Option<String> = None;
    for line in src.lines() {
        if let Some(rest) = line.strip_prefix("pub struct ") {
            let name = rest
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .next()
                .unwrap_or_default();
            if !name.is_empty() {
                current = Some(name.to_string());
            }
            continue;
        }
        if line.starts_with('}') {
            current = None;
            continue;
        }
        let Some(owner) = current.as_ref() else {
            continue;
        };
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix("pub ") else {
            continue;
        };
        let Some((name, _ty)) = rest.split_once(':') else {
            continue;
        };
        let name = name.trim();
        if name.is_empty()
            || !name
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        {
            continue;
        }
        fields.insert(name.to_string(), owner.clone());
    }
    fields
}

fn collect_sources(root: &Path, out: &mut Vec<PathBuf>) {
    const SKIP: &[&str] = &["target", ".git", "node_modules", "vendor", "dist"];
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if path.is_dir() {
            if !SKIP.contains(&name.as_str()) {
                collect_sources(&path, out);
            }
        } else if matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("rs") | Some("ts") | Some("tsx")
        ) {
            out.push(path);
        }
    }
}

/// Field names that appear as a `.field` access somewhere other than config.rs.
fn fields_read_in_workspace(candidates: &BTreeSet<String>) -> BTreeSet<String> {
    let root = workspace_root();
    let skip = std::fs::canonicalize(config_rs()).ok();
    let mut files = Vec::new();
    collect_sources(&root, &mut files);

    let mut seen = BTreeSet::new();
    for file in files {
        if let (Some(skip), Ok(canon)) = (skip.as_ref(), std::fs::canonicalize(&file)) {
            if &canon == skip {
                continue;
            }
        }
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        let bytes = text.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            if bytes[i] != b'.' {
                i += 1;
                continue;
            }
            // Skip whitespace between the dot and the identifier so that
            // `settings\n    .field` counts as a read.
            let mut j = i + 1;
            while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                j += 1;
            }
            let start = j;
            while j < bytes.len()
                && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_')
            {
                j += 1;
            }
            if j > start {
                let ident = &text[start..j];
                if candidates.contains(ident) {
                    seen.insert(ident.to_string());
                }
            }
            i = if j > i { j } else { i + 1 };
        }
    }
    seen
}

#[test]
fn every_settings_field_reaches_a_consumer() {
    let src = std::fs::read_to_string(config_rs()).expect("config.rs must be readable");
    let declared = declared_settings_fields(&src);
    assert!(
        declared.len() > 250,
        "parsed only {} config fields — the struct parser is broken, not the config",
        declared.len()
    );

    let names: BTreeSet<String> = declared.keys().cloned().collect();
    let read = fields_read_in_workspace(&names);

    let ledger: BTreeMap<&str, &str> = UNWIRED
        .iter()
        .chain(TEST_FIXTURE_ONLY.iter())
        .copied()
        .collect();

    let orphans: Vec<(&String, &String)> = declared
        .iter()
        .filter(|(name, _)| !read.contains(*name) && !ledger.contains_key(name.as_str()))
        .collect();

    assert!(
        orphans.is_empty(),
        "{} config field(s) have NO FINAL RECIPIENT — the operator can set them \
         and nothing reads them:\n{}\n\n\
         Pick one:\n\
         • delete the field (and its config.yaml key + any UI control), or\n\
         • wire it to the code that should honour it, or\n\
         • if a real mechanism exists but connecting it would change trading \
         behaviour, add it to UNWIRED in this file WITH the mechanism named.\n\
         Do not add it to the ledger to silence this — a value the operator \
         can set that changes nothing is the defect this test exists to stop.",
        orphans.len(),
        orphans
            .iter()
            .map(|(n, owner)| format!("  - {owner}::{n}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn the_no_recipient_ledgers_only_ever_shrink() {
    // Frozen at the 2026-08-04 census. Lowering these is progress; raising
    // either one means a knob that does nothing was added on purpose.
    assert!(
        UNWIRED.len() <= 3,
        "UNWIRED grew to {} — wire the new entry or delete it, don't record it",
        UNWIRED.len()
    );
    assert!(
        TEST_FIXTURE_ONLY.len() <= 3,
        "TEST_FIXTURE_ONLY grew to {} — a config field whose only reader is a \
         test fixture is not configuration",
        TEST_FIXTURE_ONLY.len()
    );

    let src = std::fs::read_to_string(config_rs()).expect("config.rs must be readable");
    let declared = declared_settings_fields(&src);
    for (field, mechanism) in UNWIRED.iter().chain(TEST_FIXTURE_ONLY.iter()) {
        assert!(
            declared.contains_key(*field),
            "`{field}` is in a no-recipient ledger but no longer exists in \
             Settings — drop the stale ledger entry"
        );
        assert!(
            !mechanism.trim().is_empty(),
            "`{field}` must name the mechanism it fails to reach"
        );
    }
}

// ─────────────────────────── the UI side ───────────────────────────

fn settings_endpoint_src() -> String {
    let path = workspace_root()
        .join("crates")
        .join("neoethos-app")
        .join("src")
        .join("server")
        .join("settings.rs");
    std::fs::read_to_string(path).expect("server/settings.rs must be readable")
}

fn advanced_screen_src() -> String {
    let path = workspace_root()
        .join("desktop")
        .join("src")
        .join("screens")
        .join("Advanced.tsx");
    std::fs::read_to_string(path).expect("Advanced.tsx must be readable")
}

/// Field names of `struct SettingsUpdateDto` — what `POST /settings` accepts.
fn update_dto_fields(src: &str) -> BTreeSet<String> {
    let start = src
        .find("pub struct SettingsUpdateDto {")
        .expect("SettingsUpdateDto must exist");
    let body = &src[start..];
    let end = body.find("\n}").expect("SettingsUpdateDto must be closed");
    body[..end]
        .lines()
        .filter_map(|l| l.trim_start().strip_prefix("pub "))
        .filter_map(|l| l.split_once(':'))
        .map(|(n, _)| n.trim().to_string())
        .filter(|n| !n.is_empty())
        .collect()
}

fn camel_to_snake(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_uppercase() {
            out.push('_');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

#[test]
fn every_ui_control_has_a_field_behind_it() {
    let ui = advanced_screen_src();
    let dto = update_dto_fields(&settings_endpoint_src());

    // `{ key: "searchPopulation", label: ... }` entries in the GROUPS catalog.
    let mut orphan_controls = Vec::new();
    for (idx, _) in ui.match_indices("key: \"") {
        let rest = &ui[idx + 6..];
        let Some(end) = rest.find('"') else { continue };
        let key = &rest[..end];
        let snake = camel_to_snake(key);
        if !dto.contains(&snake) {
            orphan_controls.push(key.to_string());
        }
    }

    assert!(
        orphan_controls.is_empty(),
        "Advanced.tsx offers control(s) that POST /settings does not accept — \
         the operator edits them, presses Save, gets \"✓ Settings saved.\", and \
         the value is dropped by serde: {orphan_controls:?}"
    );
}

#[test]
fn every_accepted_field_is_actually_applied() {
    let src = settings_endpoint_src();
    let dto = update_dto_fields(&src);
    let handler_start = src
        .find("pub async fn update_settings(")
        .expect("update_settings handler must exist");
    let handler = &src[handler_start..];

    let unapplied: Vec<&String> = dto
        .iter()
        .filter(|f| !handler.contains(&format!("payload.{f}")))
        .collect();

    assert!(
        unapplied.is_empty(),
        "POST /settings declares field(s) it never applies to Settings: \
         {unapplied:?}. Accepting a value and dropping it is worse than \
         rejecting it — the caller is told it was saved."
    );
}

/// Fields whose YAML value is a free-form map — their nested keys are data
/// (model names, timeframe names), not config field names, so the seed scan
/// must not treat them as such.
const MAP_VALUED_FIELDS: &[&str] = &[
    "hpo_trials_by_model",
    "max_epochs_by_model",
    "model_param_overrides",
    "prop_search_max_rows_by_tf",
];

#[test]
fn the_bundled_seed_config_declares_nothing_settings_would_drop() {
    // `desktop/src-tauri/resources/config.yaml` is what the installer copies
    // into the user data dir on first run. `Settings` does NOT derive
    // `deny_unknown_fields`, so a key here that no longer exists on the struct
    // is read, ignored, and never mentioned — a shipped default with no final
    // recipient, seeded onto every fresh install.
    let seed_path = workspace_root()
        .join("desktop")
        .join("src-tauri")
        .join("resources")
        .join("config.yaml");
    let seed = std::fs::read_to_string(&seed_path).expect("bundled seed config must be readable");

    let src = std::fs::read_to_string(config_rs()).expect("config.rs must be readable");
    let declared = declared_settings_fields(&src);

    let mut unknown = Vec::new();
    let mut skip_indent: Option<usize> = None;
    for (lineno, line) in seed.lines().enumerate() {
        let indent = line.len() - line.trim_start().len();
        if let Some(level) = skip_indent {
            if line.trim().is_empty() || indent > level {
                continue;
            }
            skip_indent = None;
        }
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') || trimmed.starts_with("- ") {
            continue;
        }
        let Some((key, _)) = trimmed.split_once(':') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty()
            || !key
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        {
            continue;
        }
        if MAP_VALUED_FIELDS.contains(&key) {
            skip_indent = Some(indent);
            continue;
        }
        if !declared.contains_key(key) {
            unknown.push(format!("  line {}: {key}", lineno + 1));
        }
    }

    assert!(
        unknown.is_empty(),
        "the bundled seed config.yaml declares {} key(s) that no longer exist \
         on Settings:\n{}\n\nserde ignores unknown keys, so every fresh install \
         would ship these and silently drop them. Delete them from \
         desktop/src-tauri/resources/config.yaml.",
        unknown.len(),
        unknown.join("\n")
    );
}

#[test]
fn no_ui_control_is_backed_by_a_no_recipient_field() {
    // A control may only exist for a knob that reaches a live consumer. This
    // is the check `ui_locale` failed: a Language picker, fully plumbed
    // through the DTO, backed by a field with no reader in the workspace.
    let ui = advanced_screen_src();
    let ledger: BTreeSet<&str> = UNWIRED
        .iter()
        .chain(TEST_FIXTURE_ONLY.iter())
        .map(|(f, _)| *f)
        .collect();

    let mut offenders = Vec::new();
    for (idx, _) in ui.match_indices("key: \"") {
        let rest = &ui[idx + 6..];
        let Some(end) = rest.find('"') else { continue };
        let snake = camel_to_snake(&rest[..end]);
        if ledger.contains(snake.as_str()) {
            offenders.push(snake);
        }
    }

    assert!(
        offenders.is_empty(),
        "UI control(s) are backed by fields with no live consumer: {offenders:?}. \
         Wire the field or remove the control — do not ship a knob that moves \
         nothing."
    );
}
