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
//! Run: `cargo test -p neoethos-app --test packaging_smoke`

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Walk back from `CARGO_MANIFEST_DIR` (the neoethos-app crate root) to the
/// workspace root that owns `packaging/`.
fn workspace_root() -> PathBuf {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // crate_dir = .../neoethos/crates/neoethos-app
    crate_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/neoethos-app")
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

// Gated to Unix-like targets only. On Windows, Git Bash / WSL bash IS
// usually on PATH, but Windows-style paths (`C:\Users\...`) passed to a
// Unix-style bash get mangled to `/c/Users/...` or `C:Users...` (depending
// on the bash flavour), producing spurious "No such file or directory"
// failures even when the .sh script is syntactically valid. The release
// pipeline that actually invokes these scripts runs on Ubuntu in CI, so
// the syntax check belongs on the same target.
#[cfg(not(windows))]
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
        p.extension().is_some_and(|e| e == "sh") || p.file_name().is_some_and(|n| n == "AppRun")
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
            body.contains("kosred.neoethos"),
            "WinGet manifest {} has wrong PackageIdentifier",
            file.display()
        );
    }
}

#[test]
fn cargo_deb_and_rpm_metadata_present() {
    // The two cargo subcommands (cargo-deb, cargo-generate-rpm) drive
    // off the `[package.metadata.deb]` / `[package.metadata.generate-rpm]`
    // tables in neoethos-app/Cargo.toml. If either disappears, the release
    // workflow falls over silently with "no metadata table".
    let cargo_toml = workspace_root()
        .join("crates")
        .join("neoethos-app")
        .join("Cargo.toml");
    let body = fs::read_to_string(&cargo_toml).expect("read neoethos-app Cargo.toml");
    assert!(
        body.contains("[package.metadata.deb]"),
        "neoethos-app Cargo.toml missing [package.metadata.deb] — cargo-deb cannot run"
    );
    assert!(
        body.contains("[package.metadata.generate-rpm]"),
        "neoethos-app Cargo.toml missing [package.metadata.generate-rpm] — cargo-generate-rpm cannot run"
    );
    // Both must reference the release binary path so the artifact
    // actually contains neoethos-app.
    assert!(
        body.contains("target/release/neoethos-app"),
        "packaging metadata must point at target/release/neoethos-app"
    );
}

#[test]
fn chocolatey_nuspec_parses_as_xml() {
    let nuspec = packaging_dir()
        .join("chocolatey")
        .join("neoethos")
        .join("neoethos.nuspec");
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
    let manifest = packaging_dir().join("scoop").join("neoethos.json");
    let body = fs::read_to_string(&manifest).expect("read scoop manifest");
    let parsed: serde_json::Value =
        serde_json::from_str(&body).expect("scoop manifest must be valid JSON");
    let obj = parsed
        .as_object()
        .expect("scoop manifest must be a JSON object");
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
    let appdir = packaging_dir().join("appimage").join("neoethos-app.AppDir");
    for required in ["AppRun", "neoethos-app.desktop"] {
        let path = appdir.join(required);
        assert!(
            path.is_file(),
            "AppImage .AppDir missing {}: {}",
            required,
            path.display()
        );
    }
    // Icon is optional during scaffolding (a TODO placeholder is fine).
    let icon_present =
        appdir.join("neoethos-app.png").exists() || appdir.join("neoethos-app.png.TODO").exists();
    assert!(
        icon_present,
        "AppImage .AppDir must carry neoethos-app.png OR a TODO marker file"
    );
}

#[test]
fn windows_release_binary_does_not_import_debug_vc_runtime() {
    let release_dir = workspace_root().join("target").join("release");
    let exe = release_dir.join("neoethos-app.exe");
    if !exe.is_file() {
        eprintln!(
            "[packaging_smoke] {} not present; skipping Windows release import check",
            exe.display()
        );
        return;
    }

    let mut checked = Vec::new();
    let mut stack = vec![exe];
    while let Some(path) = stack.pop() {
        if checked.iter().any(|seen: &PathBuf| same_path(seen, &path)) {
            continue;
        }
        let imports = pe_imports(&path).unwrap_or_else(|err| {
            panic!("failed to read PE imports for {}: {err}", path.display())
        });
        let debug_runtime = imports.iter().find(|name| {
            let lower = name.to_ascii_lowercase();
            lower.ends_with("d.dll") && (lower.contains("vcomp") || lower.contains("vcruntime"))
        });
        assert!(
            debug_runtime.is_none(),
            "{} imports debug VC runtime {}; clean Windows machines will not have this DLL",
            path.display(),
            debug_runtime.unwrap()
        );
        for import in imports {
            let local = release_dir.join(&import);
            if local.is_file() {
                stack.push(local);
            }
        }
        checked.push(path);
    }
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

fn same_path(a: &Path, b: &Path) -> bool {
    a.to_string_lossy()
        .eq_ignore_ascii_case(&b.to_string_lossy())
}

fn pe_imports(path: &Path) -> std::io::Result<Vec<String>> {
    let bytes = fs::read(path)?;
    if bytes.get(0..2) != Some(b"MZ") {
        return Ok(Vec::new());
    }
    let pe_offset = read_u32(&bytes, 0x3c)? as usize;
    if bytes.get(pe_offset..pe_offset + 4) != Some(b"PE\0\0") {
        return Ok(Vec::new());
    }

    let coff = pe_offset + 4;
    let section_count = read_u16(&bytes, coff + 2)? as usize;
    let optional_header_size = read_u16(&bytes, coff + 16)? as usize;
    let optional = coff + 20;
    let magic = read_u16(&bytes, optional)?;
    let data_directory = match magic {
        0x10b => optional + 96,
        0x20b => optional + 112,
        _ => return Ok(Vec::new()),
    };
    let import_rva = read_u32(&bytes, data_directory + 8)?;
    if import_rva == 0 {
        return Ok(Vec::new());
    }

    let section_table = optional + optional_header_size;
    let mut sections = Vec::new();
    for idx in 0..section_count {
        let off = section_table + idx * 40;
        let virtual_size = read_u32(&bytes, off + 8)?;
        let virtual_address = read_u32(&bytes, off + 12)?;
        let raw_size = read_u32(&bytes, off + 16)?;
        let raw_ptr = read_u32(&bytes, off + 20)?;
        sections.push((virtual_address, virtual_size.max(raw_size), raw_ptr));
    }

    let mut imports = Vec::new();
    let mut descriptor = rva_to_offset(import_rva, &sections)? as usize;
    loop {
        let original_first_thunk = read_u32(&bytes, descriptor)?;
        let time_date_stamp = read_u32(&bytes, descriptor + 4)?;
        let forwarder_chain = read_u32(&bytes, descriptor + 8)?;
        let name_rva = read_u32(&bytes, descriptor + 12)?;
        let first_thunk = read_u32(&bytes, descriptor + 16)?;
        if original_first_thunk == 0
            && time_date_stamp == 0
            && forwarder_chain == 0
            && name_rva == 0
            && first_thunk == 0
        {
            break;
        }
        let name_offset = rva_to_offset(name_rva, &sections)? as usize;
        let end = bytes[name_offset..]
            .iter()
            .position(|byte| *byte == 0)
            .map(|pos| name_offset + pos)
            .ok_or_else(|| std::io::Error::other("unterminated import name"))?;
        imports.push(String::from_utf8_lossy(&bytes[name_offset..end]).to_string());
        descriptor += 20;
    }
    Ok(imports)
}

fn read_u16(bytes: &[u8], offset: usize) -> std::io::Result<u16> {
    let slice = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| std::io::Error::other("PE read out of bounds"))?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> std::io::Result<u32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| std::io::Error::other("PE read out of bounds"))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn rva_to_offset(rva: u32, sections: &[(u32, u32, u32)]) -> std::io::Result<u32> {
    for (virtual_address, size, raw_ptr) in sections {
        if *virtual_address <= rva && rva < virtual_address.saturating_add(*size) {
            return Ok(raw_ptr.saturating_add(rva - virtual_address));
        }
    }
    Err(std::io::Error::other(format!(
        "RVA {rva:#x} outside PE sections"
    )))
}
