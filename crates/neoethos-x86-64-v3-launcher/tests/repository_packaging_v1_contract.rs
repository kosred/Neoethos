use std::fs;
use std::path::{Path, PathBuf};

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .unwrap()
        .to_path_buf()
}

fn source(path: &str) -> String {
    fs::read_to_string(repository_root().join(path)).unwrap()
}

#[test]
fn launcher_is_a_workspace_member_but_not_a_default_application_payload() {
    let manifest = source("Cargo.toml");
    let member = "\"crates/neoethos-x86-64-v3-launcher\"";

    assert_eq!(manifest.matches(member).count(), 1);
    let default_members = manifest
        .split("default-members = [")
        .nth(1)
        .unwrap()
        .split(']')
        .next()
        .unwrap();
    assert!(!default_members.contains(member));
}

#[test]
fn repository_default_builds_are_baseline_and_never_globally_v3() {
    let config = source(".cargo/config.toml");

    assert!(!config.contains("target-cpu=x86-64-v3"));
    assert!(!config.contains("target_feature=+avx2"));
    assert!(!config.contains("[target.x86_64-pc-windows-msvc]"));
    assert!(!config.contains("[target.x86_64-unknown-linux-gnu]"));
}

#[test]
fn paid_gpu_test_harnesses_do_not_execute_unlaunched_v3_test_binaries() {
    let bootstrap = source("scripts/gpu-bench/remote_bootstrap.sh");

    assert!(!bootstrap.contains("target-cpu=x86-64-v3"));
    assert!(bootstrap.contains("-C link-arg=-fuse-ld=bfd"));
}

#[test]
fn windows_gpu_portable_bundle_stages_baseline_launcher_and_private_v3_payload() {
    let workflow = source(".github/workflows/release.yml");

    for required in [
        "$env:CARGO_TARGET_DIR = \"target\\x86-64-v3-payload\"",
        "$env:RUSTFLAGS = \"-C target-cpu=x86-64-v3\"",
        "$env:CARGO_TARGET_DIR = \"target\\x86-64-baseline-launcher\"",
        "$env:RUSTFLAGS = \"-C target-cpu=x86-64\"",
        "cargo build -p neoethos-x86-64-v3-launcher --release",
        "target\\x86-64-baseline-launcher\\release\\neoethos-x86-64-v3-launcher.exe",
        "target\\x86-64-v3-payload\\release\\neoethos-app.exe",
        "$outDir\\NeoEthos.exe",
        "$outDir\\NeoEthos.x86-64-v3.exe",
    ] {
        assert!(
            workflow.contains(required),
            "missing Windows token {required}"
        );
    }
}

#[test]
fn linux_gpu_portable_bundle_stages_baseline_launcher_and_private_v3_payload() {
    let workflow = source(".github/workflows/release.yml");

    for required in [
        "CARGO_TARGET_DIR=target/x86-64-v3-payload RUSTFLAGS='-C target-cpu=x86-64-v3'",
        "CARGO_TARGET_DIR=target/x86-64-baseline-launcher RUSTFLAGS='-C target-cpu=x86-64'",
        "cargo build -p neoethos-x86-64-v3-launcher --release",
        "target/x86-64-baseline-launcher/release/neoethos-x86-64-v3-launcher",
        "target/x86-64-v3-payload/release/neoethos-app",
        "$OUTDIR/NeoEthos",
        "$OUTDIR/NeoEthos.x86-64-v3",
    ] {
        assert!(
            workflow.contains(required),
            "missing Linux token {required}"
        );
    }
}

#[test]
fn windows_public_testing_archive_stages_both_app_and_cli_launch_pairs() {
    let workflow = source(".github/workflows/release-installers.yml");

    for required in [
        "target\\x86-64-baseline-launcher\\release\\neoethos-x86-64-v3-launcher.exe",
        "target\\x86-64-v3-payload\\release\\neoethos-app.exe",
        "target\\x86-64-v3-payload\\release\\neoethos-cli.exe",
        "out\\neoethos-app.exe",
        "out\\neoethos-app.x86-64-v3.exe",
        "out\\neoethos-cli.exe",
        "out\\neoethos-cli.x86-64-v3.exe",
    ] {
        assert!(
            workflow.contains(required),
            "missing archive token {required}"
        );
    }
}

#[test]
fn appimage_public_entry_is_the_baseline_launcher_with_a_private_payload() {
    let build = source("packaging/appimage/build.sh");
    let app_run = source("packaging/appimage/neoethos-app.AppDir/AppRun");

    for required in [
        "CARGO_TARGET_DIR=target/x86-64-v3-payload RUSTFLAGS='-C target-cpu=x86-64-v3'",
        "CARGO_TARGET_DIR=target/x86-64-baseline-launcher RUSTFLAGS='-C target-cpu=x86-64'",
        "target/x86-64-baseline-launcher/release/neoethos-x86-64-v3-launcher",
        "target/x86-64-v3-payload/release/neoethos-app",
        "${APPDIR}/usr/bin/neoethos-app",
        "${APPDIR}/usr/bin/neoethos-app.x86-64-v3",
    ] {
        assert!(
            build.contains(required),
            "missing AppImage token {required}"
        );
    }
    assert!(app_run.contains("exec \"${APPDIR}/usr/bin/neoethos-app\" \"$@\""));
    assert!(!app_run.contains("neoethos-app.x86-64-v3\" \"$@\""));
}

#[test]
fn every_packaging_source_that_compiles_v3_also_names_the_launcher_and_private_payload() {
    for path in [
        ".github/workflows/release.yml",
        ".github/workflows/release-installers.yml",
        "packaging/appimage/build.sh",
    ] {
        let packaging_source = source(path);
        if packaging_source.contains("target-cpu=x86-64-v3") {
            assert!(packaging_source.contains("neoethos-x86-64-v3-launcher"));
            assert!(packaging_source.contains(".x86-64-v3"));
            assert!(packaging_source.contains("target-cpu=x86-64"));
        }
    }
}
