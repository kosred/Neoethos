//! THE RATCHET: this crate must not read the environment for a knob.
//!
//! 2026-08-10, the env→config wave. `crates/neoethos-search` carried ~60
//! `NEOETHOS_*` / `FOREX_*` / `RAYON_*` names across nine files. Every one of
//! them either became a typed field on the single `Settings` or was derived
//! from the probed hardware. The failure mode this closes is not the read
//! itself — it is that a value could reach a money decision without appearing
//! in any config file, any artifact, or any log, so two runs of "the same
//! configuration" were not the same run.
//!
//! Migration alone reproduces the defect: the last consolidation pass migrated
//! six boundaries to `from_settings` and left every `from_env` sibling alive
//! next to it, and they were still there eleven weeks later. So this test
//! closes the loop from the other side — it greps this crate's own sources and
//! FAILS when a new `env::var` appears outside `#[cfg(test)]` and outside the
//! small, named allowlist below.
//!
//! If you are here because this test just failed: the answer is a config field
//! on `neoethos_core::Settings` reached through an installed
//! `*RuntimeOverrides` struct, or a value derived from
//! `neoethos_core::system::HardwareProbe` / `available_memory_bytes()`. It is
//! not a new entry in the allowlist unless the read decides nothing.

use std::path::{Path, PathBuf};

/// Reads that are allowed to remain, each because it CANNOT change what a run
/// computes. `(path suffix, why)` — the path is matched with `ends_with` on a
/// forward-slash-normalised relative path.
const ALLOWED: &[(&str, &str)] = &[(
    "src/execution_profile.rs",
    "RECORDER + the retired-env reporter. `raw_env` writes ambient values into the run \
         profile and nothing branches on the result; `report_retired_env_vars` exists \
         precisely to shout that a retired name is set and ignored.",
)];

fn is_env_read(line: &str) -> bool {
    let trimmed = line.trim_start();
    if trimmed.starts_with("//") || trimmed.starts_with("*") {
        return false;
    }
    // Strip a trailing line comment so a mention inside one does not count.
    let code = match trimmed.find("//") {
        Some(i) => &trimmed[..i],
        None => trimmed,
    };
    code.contains("env::var(") || code.contains("env::var_os(") || code.contains("env::vars(")
}

fn retired_env_alias(line: &str) -> Option<&'static str> {
    let trimmed = line.trim_start();
    if trimmed.starts_with("//") || trimmed.starts_with('*') {
        return None;
    }
    let code = match trimmed.find("//") {
        Some(i) => &trimmed[..i],
        None => trimmed,
    };
    let tokens: Vec<&str> = code
        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
        .filter(|token| !token.is_empty())
        .collect();

    if tokens
        .iter()
        .any(|token| token.starts_with("install_") && token.ends_with("_from_env"))
    {
        return Some("retired install_*_from_env alias");
    }
    if tokens
        .windows(2)
        .any(|pair| pair == ["SmcSearchConfig", "from_env"])
    {
        return Some("retired SmcSearchConfig::from_env alias");
    }
    if tokens
        .windows(2)
        .any(|pair| pair == ["SeenSignatureMemory", "from_env"])
    {
        return Some("retired SeenSignatureMemory::from_env alias");
    }
    if tokens.windows(2).any(|pair| pair == ["fn", "from_env"]) {
        return Some("retired from_env constructor definition");
    }
    None
}

#[test]
fn retired_env_alias_detector_covers_definitions_reexports_and_calls() {
    for forbidden in [
        "pub fn install_smc_search_config_from_env() {}",
        "pub use smc::install_smc_search_config_from_env;",
        "pub fn from_env() -> Self { Self::current() }",
        "let cfg = SmcSearchConfig :: from_env();",
        "let seen = SeenSignatureMemory::from_env();",
    ] {
        assert!(
            retired_env_alias(forbidden).is_some(),
            "detector missed forbidden production surface: {forbidden}"
        );
    }
}

/// Line indices (0-based) that sit inside a `#[cfg(test)]` item.
///
/// Deliberately simple: find `#[cfg(test)]`, walk forward to the first `{`,
/// then to its matching `}` counting braces. That covers `mod tests { .. }`
/// and `fn ..() { .. }`, which is every shape this crate uses. Braces inside
/// string literals would fool it; there are none in the test modules here, and
/// a false POSITIVE (skipping too much) is the safe direction for a ratchet
/// that must never block a legitimate build — a false negative just means one
/// more line to justify.
fn cfg_test_lines(src: &str) -> Vec<bool> {
    let lines: Vec<&str> = src.lines().collect();
    let mut skip = vec![false; lines.len()];
    let mut i = 0usize;
    while i < lines.len() {
        if !lines[i].trim_start().starts_with("#[cfg(test)]") {
            i += 1;
            continue;
        }
        // Walk to the opening brace of the item that follows. A `;` first
        // means the item is a declaration like `#[cfg(test)] mod foo;` — the
        // whole of `foo.rs` is then test-only, which the filename rule below
        // handles.
        let mut j = i;
        let mut declaration_only = false;
        while j < lines.len() && !lines[j].contains('{') {
            skip[j] = true;
            if lines[j].trim_end().ends_with(';') {
                declaration_only = true;
                break;
            }
            j += 1;
        }
        if declaration_only || j >= lines.len() {
            i = j + 1;
            continue;
        }
        let mut depth = 0i32;
        while j < lines.len() {
            skip[j] = true;
            for ch in lines[j].chars() {
                match ch {
                    '{' => depth += 1,
                    '}' => depth -= 1,
                    _ => {}
                }
            }
            if depth <= 0 {
                break;
            }
            j += 1;
        }
        i = j + 1;
    }
    skip
}

fn rust_sources(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn no_production_env_reads_in_neoethos_search() {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src = crate_root.join("src");
    let mut files = Vec::new();
    rust_sources(&src, &mut files);
    assert!(
        !files.is_empty(),
        "found no .rs files under {} — the ratchet is not scanning anything",
        src.display()
    );

    let mut offences: Vec<String> = Vec::new();
    for file in &files {
        let rel = file
            .strip_prefix(&crate_root)
            .unwrap_or(file)
            .to_string_lossy()
            .replace('\\', "/");
        if ALLOWED.iter().any(|(suffix, _)| rel.ends_with(suffix)) {
            continue;
        }
        // Whole-file test modules. This crate declares them at the parent with
        // `#[cfg(all(test, ..))] mod device_tests;`, so the file itself carries
        // no marker to find — the convention is the name.
        let name = file
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        if name == "tests.rs" || name.ends_with("_tests.rs") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(file) else {
            continue;
        };
        let skip = cfg_test_lines(&text);
        for (idx, line) in text.lines().enumerate() {
            if skip.get(idx).copied().unwrap_or(false) {
                continue;
            }
            if is_env_read(line) {
                offences.push(format!("{rel}:{}: {}", idx + 1, line.trim()));
            }
        }
    }

    assert!(
        offences.is_empty(),
        "neoethos-search reads the environment on a production path again \
         ({} site(s)).\n\n{}\n\nONE CONFIG, NO ENV. Put the value on \
         `neoethos_core::Settings` and reach it through an installed \
         `*RuntimeOverrides` struct, or derive it from the hardware probe. If the \
         read genuinely decides nothing, add it to ALLOWED in this file WITH the \
         reason — that list is the record of what was argued, not a place to hide \
         a knob.",
        offences.len(),
        offences.join("\n")
    );
}

/// The allowlist is a liability, so it is capped. Growing it is a decision
/// someone has to make deliberately, in this file, in front of this number.
#[test]
fn the_allowlist_stays_small() {
    assert!(
        ALLOWED.len() <= 2,
        "the env allowlist has grown to {} entries; every one of them is a place a \
         value can reach a run without appearing in config",
        ALLOWED.len()
    );
    for (path, why) in ALLOWED {
        assert!(
            why.len() > 40,
            "allowlist entry {path} has no real justification: {why:?}"
        );
    }
}

/// Once the typed settings/runtime-override boundary replaced the legacy env
/// surface, keeping compatibility aliases became actively unsafe: each
/// `install_*_from_env` shim installed DEFAULTS into a `OnceLock`, so an early
/// call could permanently win over the real settings. Keep definitions,
/// re-exports, and calls out of production source instead of preserving dead
/// spellings indefinitely.
#[test]
fn no_retired_env_aliases_in_production_search_surface() {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src = crate_root.join("src");
    let mut files = Vec::new();
    rust_sources(&src, &mut files);
    assert!(
        !files.is_empty(),
        "found no .rs files under {} — the ratchet is not scanning anything",
        src.display()
    );

    let mut offences = Vec::new();
    for file in &files {
        let name = file.file_name().unwrap_or_default().to_string_lossy();
        if name == "tests.rs" || name.ends_with("_tests.rs") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(file) else {
            continue;
        };
        let skip = cfg_test_lines(&text);
        let rel = file
            .strip_prefix(&crate_root)
            .unwrap_or(file)
            .to_string_lossy()
            .replace('\\', "/");
        for (idx, line) in text.lines().enumerate() {
            if skip.get(idx).copied().unwrap_or(false) {
                continue;
            }
            if let Some(reason) = retired_env_alias(line) {
                offences.push(format!("{rel}:{}: {reason}: {}", idx + 1, line.trim()));
            }
        }
    }

    assert!(
        offences.is_empty(),
        "retired env aliases remain on the production search surface ({} site(s)).\n\n{}\n\n\
         Delete the alias definition/re-export/call. Production callers must use the typed \
         `current()` or runtime-override boundary; no DEFAULT installer may initialize a \
         `OnceLock` before settings.",
        offences.len(),
        offences.join("\n")
    );
}
