//! The desktop seed is a GENERATED PROJECTION of `Settings::default()`.
//!
//! # What this closes
//!
//! `desktop/src-tauri/resources/config.yaml` used to be an independently
//! editable file with 233 keys, against the repo root's 383. Seed minus root
//! was 150 keys; root minus seed was zero. Five of the keys they did share
//! disagreed, and **all five were on the money path** — `discovery_mode`,
//! `daily_drawdown_limit`, `total_drawdown_limit`, `prop_firm_rules`,
//! `account_currency`. Two carried f32-widened fingerprints
//! (`0.10000000149011612`, `0.20000000298023224`), which is the signature of a
//! value that was written by code, not chosen by a person.
//!
//! Reconciling those five by hand would have fixed the symptom and left the
//! mechanism — a second hand-editable source of default values — completely
//! intact. So the seed stops being a source. It is generated here, and any
//! `Default` change that is not mirrored into it **fails the build**.
//!
//! # Why this test writes the file instead of only complaining
//!
//! The file is an artifact with no independent authority: there is exactly one
//! correct content for it, and this test can compute it. Refusing to write it
//! would mean transcribing 390 fields by hand — which is the error this whole
//! exercise exists to stop. So it regenerates and then **fails anyway**,
//! printing every key whose value changed. The failure is the point: a shipped
//! default moving is something a human has to read and commit, and `git diff`
//! is the review surface.
//!
//! A green run therefore means "the seed on disk already equalled the
//! defaults". A red run means "it did not, here is exactly how, and the file
//! has been corrected for you to inspect".

use std::path::{Path, PathBuf};

use neoethos_core::config::Settings;

/// Written as the first line of the generated body. `scripts/migrate_live_config.ps1`
/// refuses to use the seed as a defaults source unless it finds this marker,
/// so a stale hand-written seed can never be mistaken for the real defaults.
pub const GENERATED_MARKER: &str = "# neoethos-generated-defaults: v1";

const HEADER: &str = "\
# ============================================================================
# GENERATED FILE — DO NOT EDIT BY HAND.
# ============================================================================
#
# Produced from `neoethos_core::config::Settings::default()` by
#   crates/neoethos-core/tests/generated_seed_is_current.rs
#
# Every value below IS the code default. This file has no opinion of its own
# and cannot drift: a `Default` change that is not mirrored here fails the
# build, and the failure names each key whose shipped value moved.
#
# TO CHANGE A SHIPPED DEFAULT: change the `Default` impl in
# crates/neoethos-core/src/config.rs, then re-run
#   cargo test -p neoethos-core --test generated_seed_is_current
# and commit the regenerated file. There is no other supported edit.
#
# The operator's live store (%LOCALAPPDATA%\\neoethos\\config.yaml) is the only
# file a run actually reads. It carries OVERRIDES ONLY. Migrate it with
# scripts/migrate_live_config.ps1 — never by hand, never unattended.
# ============================================================================
";

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = <root>/crates/neoethos-core
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root is two levels above the crate manifest")
        .to_path_buf()
}

/// `desktop_seed_is_the_serialised_defaults` REWRITES the seed while
/// `the_seed_on_disk_declares_itself_generated` reads it, on parallel threads
/// of the same binary. Without this a reader can observe a half-written file
/// and fail for a reason unrelated to what it guards. Poisoning is ignored: a
/// panicking assert must not turn the rest of the suite into lock errors.
static SEED_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock_seed() -> std::sync::MutexGuard<'static, ()> {
    SEED_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn seed_path() -> PathBuf {
    repo_root()
        .join("desktop")
        .join("src-tauri")
        .join("resources")
        .join("config.yaml")
}

/// The exact bytes the seed must contain.
fn expected_seed() -> String {
    let body = serde_yaml_ng::to_string(&Settings::default()).expect(
        "Settings::default() must serialise to YAML. If this fails, a field's type is not \
         Serialize — which also means Settings::save() cannot write it, so the failure is real \
         and not a test artifact.",
    );
    // serde_yaml_ng may or may not prefix a document marker; the generated file
    // is a plain mapping either way.
    let body = body.strip_prefix("---\n").unwrap_or(&body);
    format!("{HEADER}{GENERATED_MARKER}\n{body}")
}

/// Flatten a YAML value to `dotted.path -> rendered value` so a diff can name
/// keys rather than line numbers. Sequences and free-form maps are compared
/// whole, as single leaves.
fn flatten(prefix: &str, v: &serde_yaml_ng::Value, out: &mut Vec<(String, String)>) {
    match v {
        serde_yaml_ng::Value::Mapping(m) if !m.is_empty() => {
            for (k, val) in m {
                let key = k.as_str().map(str::to_owned).unwrap_or_else(|| format!("{k:?}"));
                let path = if prefix.is_empty() { key } else { format!("{prefix}.{key}") };
                flatten(&path, val, out);
            }
        }
        other => out.push((prefix.to_string(), format!("{other:?}"))),
    }
}

fn leaves_of(text: &str) -> Option<Vec<(String, String)>> {
    let doc: serde_yaml_ng::Value = serde_yaml_ng::from_str(text).ok()?;
    let mut out = Vec::new();
    flatten("", &doc, &mut out);
    out.sort();
    Some(out)
}

/// Report which keys change, in the language of "what a fresh install stops
/// doing and starts doing".
fn describe_change(old: &str, new: &str) -> String {
    let (Some(a), Some(b)) = (leaves_of(old), leaves_of(new)) else {
        return "  (the previous file could not be parsed as YAML — full replacement)".to_string();
    };
    let mut lines = Vec::new();
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() || j < b.len() {
        match (a.get(i), b.get(j)) {
            (Some((ka, va)), Some((kb, vb))) if ka == kb => {
                if va != vb {
                    lines.push(format!("  CHANGED  {ka}: {va} -> {vb}"));
                }
                i += 1;
                j += 1;
            }
            (Some((ka, va)), Some((kb, _))) if ka < kb => {
                lines.push(format!(
                    "  REMOVED  {ka} (was {va}) — no longer a field of Settings, or no longer \
                     serialised"
                ));
                i += 1;
            }
            (Some(_), Some((kb, vb))) => {
                lines.push(format!("  ADDED    {kb}: {vb} — was taking this default silently"));
                j += 1;
            }
            (Some((ka, va)), None) => {
                lines.push(format!("  REMOVED  {ka} (was {va})"));
                i += 1;
            }
            (None, Some((kb, vb))) => {
                lines.push(format!("  ADDED    {kb}: {vb}"));
                j += 1;
            }
            (None, None) => break,
        }
    }
    if lines.is_empty() {
        "  (no key changed — only formatting or the header)".to_string()
    } else {
        lines.join("\n")
    }
}

#[test]
fn desktop_seed_is_the_serialised_defaults() {
    let _guard = lock_seed();
    let path = seed_path();
    let want = expected_seed();
    let have = std::fs::read_to_string(&path).unwrap_or_default();

    if have == want {
        return;
    }

    let report = describe_change(&have, &want);
    std::fs::write(&path, &want).unwrap_or_else(|e| {
        panic!(
            "the desktop seed at {} has drifted from Settings::default() AND could not be \
             rewritten: {e}",
            path.display()
        )
    });

    panic!(
        "\n\
         The desktop seed did not equal Settings::default(), so it HAS BEEN REGENERATED.\n\
         File: {}\n\
         \n\
         Review `git diff` on that file and commit it. Every line below is a change to what a\n\
         FRESH DESKTOP INSTALL does, because the seed is what a fresh install starts from:\n\
         \n\
         {report}\n\
         \n\
         This test fails on rewrite by design. A shipped default moving is not something that\n\
         should pass silently — it is exactly how this project shipped prefilter_top_k at 50 in\n\
         one file and 240 in another for eight months. Re-run the test to confirm green.\n",
        path.display()
    );
}

/// The generator is only worth anything if the marker it writes is actually
/// present, because that marker is what `scripts/migrate_live_config.ps1` keys
/// off before it will trust this file as a defaults source. A migration that
/// silently used a stale hand-written seed as "the defaults" would write wrong
/// values into the one file a run reads.
#[test]
fn the_generated_body_carries_the_marker_the_migration_tool_requires() {
    let generated = expected_seed();
    assert!(
        generated.contains(GENERATED_MARKER),
        "the generated seed must contain the line `{GENERATED_MARKER}` — \
         scripts/migrate_live_config.ps1 refuses to read defaults from a file without it"
    );
    // And it must be a real marker line, not a substring of prose.
    assert!(
        generated
            .lines()
            .any(|l| l.trim_end() == GENERATED_MARKER),
        "the marker must be a line of its own"
    );
}

/// The marker must be on the FILE, not just in the string this test can
/// compute. `desktop_seed_is_the_serialised_defaults` proves it by byte
/// equality, but only after it has already rewritten the file — so on the run
/// where a hand-written seed is still on disk, this is the check that names the
/// actual problem ("this file was written by a person") instead of drowning it
/// in a whole-file diff.
#[test]
fn the_seed_on_disk_declares_itself_generated() {
    let _guard = lock_seed();
    let path = seed_path();
    let Ok(text) = std::fs::read_to_string(&path) else {
        // An absent seed is a valid end state of the collapse — the compiled
        // defaults are the defaults, so a fresh install needs no file at all.
        // `desktop_seed_is_the_serialised_defaults` creates it either way.
        return;
    };
    assert!(
        text.lines().any(|l| l.trim_end() == GENERATED_MARKER),
        "{} does not carry `{GENERATED_MARKER}`, so it was written by hand. It is a GENERATED \
         projection of Settings::default(): the defaults live in \
         crates/neoethos-core/src/config.rs and nowhere else. Change the `Default` impl and \
         re-run `cargo test -p neoethos-core --test generated_seed_is_current`. \
         scripts/migrate_live_config.ps1 also refuses to read defaults from a file without this \
         marker, so a hand-edited seed cannot leak values into the operator's live store.",
        path.display()
    );
}

/// `Settings::default()` must round-trip through the generated file. If it does
/// not, the seed is unloadable and every fresh install fails at startup — a
/// failure this test is far cheaper to hit than an installer is.
#[test]
fn the_generated_seed_loads_back_as_settings() {
    let text = expected_seed();
    let parsed: Result<Settings, _> = serde_yaml_ng::from_str(&text);
    assert!(
        parsed.is_ok(),
        "the generated seed does not deserialise back into Settings: {:?}. \
         A field that serialises but does not deserialise (or a `deny_unknown_fields` struct \
         whose serialised form contains a key it rejects) makes the shipped seed unloadable.",
        parsed.err()
    );
}
