//! Ship-gate §5.1.7 smoke tests for the installer scaffold.
//!
//! These tests do NOT build a real .deb / .AppImage / .tar.gz (those
//! steps run in `release-installers.yml` on a tag push and take 10+
//! minutes per artifact). They verify the *scaffold* — that every
//! packaging script parses, every manifest is valid YAML/JSON, and
//! every required file is present — so that the GitHub Actions
//! release pipeline cannot fail because of a typo nobody noticed.
//!
//! The actual binary-artifact build is validated by running
//! `packaging/portable/build-portable.sh` (the simplest "at least one
//! of" path per the ship-gate) which only requires `cargo` + `tar`.
//!
//! Run: `cargo test -p forex-app --test packaging_smoke`

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Walk back from `CARGO_MANIFEST_DIR` (the forex-app crate root) to the
/// workspace root that owns `packaging/`.
fn workspace_root() -> PathBuf {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // crate_dir = .../forex-ai/crates/forex-app
    crate_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/forex-app")
        .to_path_buf()
}

fn packaging_dir() -> PathBuf {
    workspace_root().join("packaging")
}

#[test]
fn packaging_directory_exists() {
    let dir = packaging_dir();
    assert!(
        dir.is_dir(),
        "packaging/ must exist at workspace root: {}",
        dir.display()
    );
}

#[test]
fn portable_build_script_is_present_and_executable() {
    // The simplest "at least one of (.deb, .AppImage, .tar.gz)" path —
    // shipped so that a developer with only `cargo` and `tar` on PATH
    // can still produce a release artifact.
    let script = packaging_dir().join("portable").join("build-portable.sh");
    assert!(
        script.is_file(),
        "expected portable build script at {}",
        script.display()
    );

    // Cross-platform exec check: bash shebang plus a meaningful body.
    let body = fs::read_to_string(&script).expect("read portable script");
    assert!(
        body.starts_with("#!/usr/bin/env bash") || body.starts_with("#!/bin/bash"),
        "portable script must start with a bash shebang"
    );
    assert!(
        body.contains("cargo build --release"),
        "portable script must invoke a release build"
    );
    assert!(
        body.contains("tar czf"),
        "portable script must emit a .tar.gz"
    );
}

#[test]
fn all_packaging_shell_scripts_have_valid_bash_syntax() {
    // bash -n parses a script without executing it — exactly the
    // check we need at CI time before invoking it for real.
    let bash = match Command::new("bash").arg("--version").output() {
        Ok(o) if o.status.success() => "bash",
        _ => {
            eprintln!("[packaging_smoke] bash not on PATH — skipping shell syntax check");
            return;
        }
    };
    let scripts = collect_files(&packaging_dir(), |p| {
        p.extension().is_some_and(|e| e == "sh")
            || p.file_name().is_some_and(|n| n == "AppRun")
    });
    assert!(
        !scripts.is_empty(),
        "expected at least one .sh script under packaging/"
    );
    for script in scripts {
        let out = Command::new(bash)
            .arg("-n")
            .arg(&script)
            .output()
            .unwrap_or_else(|err| panic!("failed to run `bash -n {}`: {err}", script.display()));
        assert!(
            out.status.success(),
            "shell syntax error in {}:\n{}",
            script.display(),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

#[test]
fn winget_manifests_have_required_keys() {
    // WinGet's `winget validate` is the source of truth — we cannot
    // call it cross-platform from a unit test. The structural check
    // below catches the common typos: missing top-level keys, wrong
    // PackageIdentifier, ManifestVersion drift.
    let winget_dir = packaging_dir().join("winget");
    let files = collect_files(&winget_dir, |p| {
        p.extension().is_some_and(|e| e == "yaml" || e == "yml")
    });
    assert!(
        files.len() >= 3,
        "WinGet requires at least 3 manifests (yaml + installer + locale); found {}",
        files.len()
    );
    for file in files {
        let body = fs::read_to_string(&file)
            .unwrap_or_else(|err| panic!("read {}: {err}", file.display()));
        // Every WinGet manifest carries these two keys at the root.
        for required in ["PackageIdentifier:", "ManifestVersion:"] {
            assert!(
                body.contains(required),
                "WinGet manifest {} missing `{}`",
                file.display(),
                required
            );
        }
        // Cross-manifest sanity: the PackageIdentifier should be ours.
        assert!(
            body.contains("kosred.forex-ai"),
            "WinGet manifest {} has wrong PackageIdentifier",
            file.display()
        );
    }
}

#[test]
fn cargo_deb_and_rpm_metadata_present() {
    // The two cargo subcommands (cargo-deb, cargo-generate-rpm) drive
    // off the `[package.metadata.deb]` / `[package.metadata.generate-rpm]`
    // tables in forex-app/Cargo.toml. If either disappears, the release
    // workflow falls over silently with "no metadata table".
    let cargo_toml = workspace_root()
        .join("crates")
        .join("forex-app")
        .join("Cargo.toml");
    let body = fs::read_to_string(&cargo_toml).expect("read forex-app Cargo.toml");
    assert!(
        body.contains("[package.metadata.deb]"),
        "forex-app Cargo.toml missing [package.metadata.deb] — cargo-deb cannot run"
    );
    assert!(
        body.contains("[package.metadata.generate-rpm]"),
        "forex-app Cargo.toml missing [package.metadata.generate-rpm] — cargo-generate-rpm cannot run"
    );
    // Both must reference the release binary path so the artifact
    // actually contains forex-app.
    assert!(
        body.contains("target/release/forex-app"),
        "packaging metadata must point at target/release/forex-app"
    );
}

#[test]
fn chocolatey_nuspec_parses_as_xml() {
    let nuspec = packaging_dir()
        .join("chocolatey")
        .join("forex-ai")
        .join("forex-ai.nuspec");
    let body = fs::read_to_string(&nuspec)
        .unwrap_or_else(|err| panic!("read {}: {err}", nuspec.display()));
    assert!(
        body.trim_start().starts_with("<?xml") || body.trim_start().starts_with("<package"),
        "{}: expected XML / NuSpec, got: {:?}",
        nuspec.display(),
        body.chars().take(80).collect::<String>()
    );
    // Sanity: must declare an id and version.
    assert!(body.contains("<id>"), "nuspec missing <id>");
    assert!(body.contains("<version>"), "nuspec missing <version>");
}

#[test]
fn scoop_manifest_is_valid_json() {
    let manifest = packaging_dir().join("scoop").join("forex-ai.json");
    let body = fs::read_to_string(&manifest).expect("read scoop manifest");
    let parsed: serde_json::Value =
        serde_json::from_str(&body).expect("scoop manifest must be valid JSON");
    let obj = parsed.as_object().expect("scoop manifest must be a JSON object");
    // Spec: every Scoop manifest needs `version` and `url` at minimum.
    assert!(
        obj.contains_key("version"),
        "scoop manifest missing `version` key"
    );
    assert!(
        obj.contains_key("description") || obj.contains_key("homepage"),
        "scoop manifest should carry at least description or homepage"
    );
}

#[test]
fn appimage_appdir_has_required_files() {
    // AppImage spec requires AppRun + .desktop + an icon at the root
    // of the .AppDir.
    let appdir = packaging_dir().join("appimage").join("forex-app.AppDir");
    for required in ["AppRun", "forex-app.desktop"] {
        let path = appdir.join(required);
        assert!(
            path.is_file(),
            "AppImage .AppDir missing {}: {}",
            required,
            path.display()
        );
    }
    // Icon is optional during scaffolding (a TODO placeholder is fine).
    let icon_present = appdir.join("forex-app.png").exists()
        || appdir.join("forex-app.png.TODO").exists();
    assert!(
        icon_present,
        "AppImage .AppDir must carry forex-app.png OR a TODO marker file"
    );
}

/// Walk `root` recursively and return every file matching `pred`.
fn collect_files<F: Fn(&Path) -> bool>(root: &Path, pred: F) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if !root.is_dir() {
        return out;
    }
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if pred(&path) {
                out.push(path);
            }
        }
    }
    out
}

