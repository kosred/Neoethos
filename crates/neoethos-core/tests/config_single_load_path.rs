//! The loop-closing test for the single resolution point.
//!
//! Two directions, so neither can rot:
//!
//! * **file → field**: every field on `Settings` must survive a write/read
//!   round trip, so a knob cannot exist in Rust and be unreachable from a
//!   config file.
//! * **field → file**: every key in every config file this project actually
//!   ships — and the operator's own live store — must land on a field, or the
//!   load fails loudly. That is the direction that used to rot silently: 150
//!   keys in the repo file were absent from the desktop seed, and a misspelled
//!   key saved and reported success.
//!
//! Plus the guards that keep the seal a seal. The mechanism is compile-time
//! (a private field + a hand-written `Deserialize`), which by definition
//! cannot be exercised from a test that compiles — so the tests here assert
//! that the mechanism is still *present in the source*, and the behaviour it
//! buys is asserted directly.
//!
//! `cargo test -p neoethos-core --test config_single_load_path`

use neoethos_core::config::{user_config_path, ConfigSource, Settings};
use std::path::{Path, PathBuf};

// ── helpers ─────────────────────────────────────────────────────────────────

/// Write `yaml` to a uniquely-named scratch file and load it.
///
/// Deliberately does NOT touch `$CONFIG_FILE`: the test binary runs its tests
/// in parallel threads of one process, and an env var is process-global. A
/// test that mutated it would make the other tests read someone else's file —
/// which is the exact defect this suite exists to prevent.
fn load_yaml(tag: &str, yaml: &str) -> anyhow::Result<Settings> {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "neoethos-config-seal-{}-{}-{:?}.yaml",
        std::process::id(),
        tag,
        std::thread::current().id()
    ));
    std::fs::write(&path, yaml).expect("write scratch config");
    let out = Settings::from_yaml(&path);
    let _ = std::fs::remove_file(&path);
    out
}

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = <repo>/crates/neoethos-core
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("repo root above crates/neoethos-core")
        .to_path_buf()
}

fn config_rs() -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/config.rs"))
        .expect("read config.rs")
}

// ── direction 1: every field is reachable from a file ───────────────────────

#[test]
fn every_field_survives_a_file_round_trip() {
    let defaults = Settings::default();
    let yaml = serde_yaml_ng::to_string(&defaults).expect("serialize defaults");
    let reloaded = load_yaml("roundtrip", &yaml).expect("a serialized Settings must load back");

    let a = serde_yaml_ng::to_string(&defaults).unwrap();
    let b = serde_yaml_ng::to_string(&reloaded).unwrap();
    assert_eq!(
        a, b,
        "a field that does not survive write -> read is a knob that exists in Rust and cannot be \
         set from a config file"
    );
}

#[test]
fn an_empty_file_is_a_valid_all_defaults_document() {
    let loaded = load_yaml("empty", "").expect("an empty config must mean 'all defaults'");
    assert_eq!(
        serde_yaml_ng::to_string(&loaded).unwrap(),
        serde_yaml_ng::to_string(&Settings::default()).unwrap()
    );
}

// ── direction 2: every key in a real file lands on a field ──────────────────

#[test]
fn the_two_repo_config_files_load() {
    for rel in ["config.yaml", "desktop/src-tauri/resources/config.yaml"] {
        let path = repo_root().join(rel);
        if !path.exists() {
            // The desktop seed is being collapsed into a generated artifact;
            // its absence is a valid end state, its being unloadable is not.
            continue;
        }
        let loaded = Settings::from_yaml(&path);
        assert!(
            loaded.is_ok(),
            "shipped config {} must load: {:?}",
            path.display(),
            loaded.err()
        );
    }
}

/// The one that matters. The operator's live store is the ONLY file a real run
/// reads, it is hand-tuned, and it is dated 2026-07-31 — long before any of
/// today's keys existed. Making it unloadable is worse than leaving the defect.
#[test]
fn the_operators_live_store_still_loads() {
    let path = user_config_path();
    if !path.exists() {
        eprintln!("no live store at {} — nothing to guard", path.display());
        return;
    }
    let loaded = Settings::from_yaml(&path);
    assert!(
        loaded.is_ok(),
        "the operator's live store {} MUST still load. If this fails, a key in his file is \
         neither a field nor listed in RETIRED_KEYS — add it to RETIRED_KEYS rather than \
         expecting him to edit the file: {:?}",
        path.display(),
        loaded.err()
    );
}

// ── the unknown-key regime ──────────────────────────────────────────────────

#[test]
fn a_misspelled_key_is_refused_and_named() {
    // The canonical case from the UI-lie list: this used to parse, save, and
    // report "config.yaml saved (verbatim)" through the raw-YAML editor — the
    // only route to 364 of the 390 knobs.
    let err = load_yaml("typo", "risk:\n  trailing_enabeld: true\n")
        .expect_err("a misspelled key must be REFUSED, not silently ignored");
    let text = format!("{err:#}");
    assert!(
        text.contains("trailing_enabeld"),
        "the refusal must NAME the key. got: {text}"
    );
    assert!(
        text.contains("RETIRED_KEYS"),
        "the refusal must tell the operator what to do. got: {text}"
    );
}

#[test]
fn an_unknown_top_level_section_is_refused() {
    let err = load_yaml("toplevel", "not_a_section:\n  a: 1\n")
        .expect_err("an unknown top-level key must be refused");
    assert!(format!("{err:#}").contains("not_a_section"));
}

#[test]
fn a_retired_key_is_accepted_and_ignored() {
    // Four tombstones are live in the operator's store right now. They must
    // stay loadable — named at WARN, never fatal.
    let yaml = "models:\n  export_onnx: true\nnews:\n  news_kill_window_min: 30\n  \
                perplexity_enabled: true\nsecrets_file: keys.txt\n";
    let loaded = load_yaml("retired", yaml);
    assert!(
        loaded.is_ok(),
        "a RETIRED key must be warned about and ignored, never fatal: {:?}",
        loaded.err()
    );
}

#[test]
fn a_hardware_derived_key_in_a_file_is_ignored() {
    // `n_jobs: 11` is a detector output that `Settings::save` pickled back in
    // as an input. A file must not be able to freeze it.
    let loaded = load_yaml("derived", "system:\n  n_jobs: 1\n  num_gpus: 7\n")
        .expect("derived keys are ignored, not fatal");
    assert_ne!(
        loaded.system.num_gpus, 7,
        "a hardware-derived field must NOT be settable from a file"
    );
}

// ── the prop-firm preset ordering bug ───────────────────────────────────────

#[test]
fn selecting_a_preset_actually_seeds_its_numbers() {
    // `#[serde(default)]` on RiskConfig builds Default FIRST with
    // PropFirmPreset::default() == Ftmo, so `preset: the5ers` used to arrive
    // AFTER the six fields it is documented to seed. You got The5%ers' name
    // with FTMO's drawdown, lot and target numbers.
    let ftmo = Settings::default().risk.daily_drawdown_limit;
    let loaded = load_yaml("preset", "risk:\n  preset: the5ers\n").expect("load");
    assert_ne!(
        loaded.risk.daily_drawdown_limit, ftmo,
        "selecting The5%ers must move the drawdown numbers off FTMO's"
    );
    assert!(
        loaded.risk.daily_drawdown_limit < ftmo,
        "The5%ers is the tighter firm; its daily stop must be below FTMO's"
    );
}

#[test]
fn an_explicit_value_beats_the_preset_seed() {
    // A preset is documented as a seed, not a lock. The operator's typed value
    // is his word — it is reported loudly when it is the looser of the two, and
    // it is never silently replaced.
    let loaded = load_yaml(
        "explicit",
        "risk:\n  preset: the5ers\n  daily_drawdown_limit: 0.011\n",
    )
    .expect("load");
    assert_eq!(
        loaded.risk.daily_drawdown_limit, 0.011,
        "an explicitly typed limit must survive the preset re-derivation"
    );
}

#[test]
fn the_default_preset_is_stable_without_a_preset_key() {
    let implicit = load_yaml("implicit", "risk:\n  initial_balance: 10000.0\n").expect("load");
    let defaults = Settings::default();
    assert_eq!(
        implicit.risk.daily_drawdown_limit,
        defaults.risk.daily_drawdown_limit
    );
    assert_eq!(implicit.risk.max_lot_size, defaults.risk.max_lot_size);
    assert_eq!(
        implicit.risk.max_trades_per_day,
        defaults.risk.max_trades_per_day
    );
}

// ── provenance: which file did this process read? ───────────────────────────

#[test]
fn a_loaded_settings_knows_which_file_it_came_from() {
    let loaded = load_yaml("prov", "system:\n  symbol: EURUSD\n").expect("load");
    assert_eq!(loaded.provenance().source(), ConfigSource::ExplicitPath);
    assert!(
        loaded.provenance().path().is_some(),
        "a file-backed Settings must carry the path it opened — nothing in the workspace could \
         answer 'which of the four config files did this process read?' before 2026-08-10"
    );
    assert_eq!(
        Settings::default().provenance().source(),
        ConfigSource::CompiledDefaults,
        "a defaulted Settings must say so, so a `unwrap_or_default()` at a call site is visible"
    );
}

// ── the seal itself ─────────────────────────────────────────────────────────

/// The enforcement is a compile error, so this asserts the mechanism is still
/// there. If someone "simplifies" the hand-written `Deserialize` back to a
/// derive, or drops the private field, the seal is gone and nothing else in
/// this file would notice.
#[test]
fn the_construction_seal_is_still_in_place() {
    let src = config_rs();

    assert!(
        src.contains("mod load_seal"),
        "Settings must live behind the private loader module"
    );
    assert!(
        src.contains("provenance: ConfigProvenance"),
        "Settings must carry the PRIVATE provenance field — that is what makes a `Settings {{ .. }}` \
         literal outside the loader a compile error"
    );
    assert!(
        src.contains("impl<'de> Deserialize<'de> for Settings"),
        "Deserialize for Settings must be hand-written — it IS the loader. A derive would be a \
         second load path that every from_str in the workspace could reach"
    );

    // A derived Deserialize on the public struct would silently restore the
    // bypass. `SettingsWire` is the only thing allowed to derive it.
    let derive_line_before_settings = src
        .split("pub struct Settings {")
        .next()
        .map(|head| head.rsplit('\n').take(6).collect::<String>())
        .unwrap_or_default();
    assert!(
        !derive_line_before_settings.contains("Deserialize"),
        "`pub struct Settings` must NOT derive Deserialize"
    );
}

#[test]
fn every_config_struct_denies_unknown_fields() {
    let src = config_rs();
    let struct_count = src.matches("\npub struct ").count();
    let denies = src.matches("deny_unknown_fields").count();
    assert!(
        denies >= struct_count,
        "every config struct must set deny_unknown_fields ({denies} attributes for {struct_count} \
         structs). Without it a misspelled key inside a section parses, saves, and reports success."
    );
}

/// The env-override layer is gone; it must not come back by accident. 27 names
/// were reachable only through `load_with_env`, which had exactly one reference
/// in the workspace: its own definition.
#[test]
fn the_env_override_layer_is_gone() {
    let src = config_rs();
    for banned in [
        "apply_overrides_from_lookup",
        "fn load_with_env",
        "NEOETHOS_BOT_SYMBOL",
        "NEOETHOS_BOT_DEVICE",
    ] {
        assert!(
            !src.contains(banned),
            "`{banned}` is back in config.rs. One config, no env: a value that can arrive from \
             two places has no single resolution point."
        );
    }
    // `$CONFIG_FILE` is the ONE remaining env read, and it is the documented
    // step 1 of the resolution order rather than a per-knob override.
    assert_eq!(
        src.matches("std::env::var(").count(),
        1,
        "config.rs must read exactly one env var: CONFIG_FILE, the first branch of the resolution \
         order"
    );
}
