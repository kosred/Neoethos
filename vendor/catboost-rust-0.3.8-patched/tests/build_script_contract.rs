#[path = "../build_support.rs"]
mod build_support;

use std::path::{Path, PathBuf};

#[test]
fn resolves_profile_output_without_assuming_target_directory_name() {
    let out_dir = PathBuf::from("workspace")
        .join("cache-models-cuda")
        .join("debug")
        .join("build")
        .join("catboost-rust-deadbeef")
        .join("out");

    let resolved = build_support::cargo_profile_output_dir(&out_dir)
        .expect("a canonical Cargo OUT_DIR must resolve");

    assert_eq!(
        resolved,
        Path::new("workspace")
            .join("cache-models-cuda")
            .join("debug")
    );
}

#[test]
fn resolves_custom_named_profile_without_collapsing_it_to_release() {
    let out_dir =
        Path::new("workspace/cache-models-cuda/release-lto/build/catboost-rust-deadbeef/out");

    let resolved = build_support::cargo_profile_output_dir(out_dir)
        .expect("a custom Cargo profile OUT_DIR must resolve structurally");

    assert_eq!(
        resolved,
        Path::new("workspace/cache-models-cuda/release-lto")
    );
}

#[test]
fn refuses_non_cargo_output_layouts() {
    let non_cargo = Path::new("workspace/cache-models-cuda/debug/catboost/out");
    assert!(build_support::cargo_profile_output_dir(non_cargo).is_err());
}

#[test]
fn production_build_script_uses_the_layout_helper_and_emits_no_routine_warnings() {
    let source = include_str!("../build.rs");

    assert!(source.contains("build_support::cargo_profile_output_dir"));
    assert!(!source.contains("ends_with(\"target\")"));
    assert!(!source.contains("cargo:warning="));
    assert!(!source.contains("cargo::warning="));
}

#[test]
fn selected_runtime_staging_rejects_missing_empty_and_partial_copies() {
    let source = include_str!("../build.rs");
    let stage = source
        .split_once("fn stage_selected_runtime(")
        .expect("missing selected-runtime staging helper")
        .1;

    for required in [
        "source.is_file()",
        "source_len == 0",
        "fs::copy",
        "copied_bytes != source_len",
    ] {
        assert!(stage.contains(required), "staging must contain {required}");
    }
}

#[test]
fn unsupported_windows_arm64_fails_before_linking_without_an_import_library() {
    let source = include_str!("../build.rs");

    assert!(!source.contains("(\"windows\", \"aarch64\") => ("));
    assert!(source.contains("CatBoost v1.2.x does not publish a Windows aarch64 import library"));
}

#[test]
fn explicit_test_registration_survives_upstreams_disabled_autotest_discovery() {
    let manifest = include_str!("../Cargo.toml");

    assert!(manifest.contains("[[test]]"));
    assert!(manifest.contains("name = \"build_script_contract\""));
    assert!(manifest.contains("path = \"tests/build_script_contract.rs\""));
}
