//! THE RATCHET: this crate must not read the environment for a knob.
//!
//! See the twin in `crates/neoethos-search/tests/env_surface_is_empty.rs` for
//! the full reasoning. In this crate the two reads that mattered were
//! `NEOETHOS_FEATURE_CUBE_MODE` (whose `ram` arm returned BEFORE the free-RAM
//! check — a hidden fallback dressed as a choice, and the never-OOM invariant
//! inverted) and `NEOETHOS_REQUIRE_GPU` in
//! `resolved_indicator_compute_policy`, which was the ONLY way to reach
//! `RequireGpu` because the Settings seam it documented had zero callers.
//!
//! Both are now derived or config-installed. This test is what stops the next
//! one appearing.

use std::path::{Path, PathBuf};

/// Reads allowed to remain because they cannot change what a run computes.
const ALLOWED: &[(&str, &str)] = &[(
    "src/lib.rs",
    "The retired-env reporter itself. It reads the retired names for the sole purpose of \
     shouting at ERROR that they were found and ignored; nothing branches on the value.",
)];

fn is_env_read(line: &str) -> bool {
    let trimmed = line.trim_start();
    if trimmed.starts_with("//") || trimmed.starts_with("*") {
        return false;
    }
    let code = match trimmed.find("//") {
        Some(i) => &trimmed[..i],
        None => trimmed,
    };
    code.contains("env::var(") || code.contains("env::var_os(") || code.contains("env::vars(")
}

/// Line indices inside a `#[cfg(test)]` item. Same deliberately-simple brace
/// walk as the search-crate twin; over-skipping is the safe direction.
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
fn no_production_env_reads_in_neoethos_data() {
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
        "neoethos-data reads the environment on a production path again ({} site(s)).\n\n{}\n\n\
         ONE CONFIG, NO ENV. Derive it from the hardware probe, or install it from \
         `Settings` through a typed seam like `hpc_ta::set_indicator_compute_policy`.",
        offences.len(),
        offences.join("\n")
    );
}
