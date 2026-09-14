#[path = "../build_support.rs"]
mod build_support;

use std::path::{Path, PathBuf};
use std::{fs, panic};

struct RuntimeFixture {
    root: PathBuf,
    source: PathBuf,
    destination: PathBuf,
}

impl RuntimeFixture {
    fn new(source: Option<&[u8]>, destination: Option<&[u8]>) -> Self {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "neoethos-catboost-staging-{}-{nonce}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        fs::create_dir(&root).expect("create a new, fixture-owned directory");
        let fixture = Self {
            source: root.join("selected.dll"),
            destination: root.join("staged.dll"),
            root,
        };
        if let Some(bytes) = source {
            fs::write(&fixture.source, bytes).unwrap();
        }
        if let Some(bytes) = destination {
            fs::write(&fixture.destination, bytes).unwrap();
        }
        fixture
    }
}

impl Drop for RuntimeFixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).expect("remove only this fixture's new directory");
    }
}

#[test]
fn stages_a_selected_runtime_when_destination_is_missing() {
    let fixture = RuntimeFixture::new(Some(b"selected-runtime"), None);
    build_support::stage_selected_runtime(&fixture.source, &fixture.destination);
    assert_eq!(fs::read(&fixture.destination).unwrap(), b"selected-runtime");
}

#[test]
fn replaces_same_length_runtime_when_a_later_chunk_differs() {
    let selected = vec![0x17_u8; 3 * 64 * 1024 + 5];
    let mut stale = selected.clone();
    stale[2 * 64 * 1024 + 1] = 0x19;
    let fixture = RuntimeFixture::new(Some(&selected), Some(&stale));
    build_support::stage_selected_runtime(&fixture.source, &fixture.destination);
    assert_eq!(fs::read(&fixture.destination).unwrap(), selected);
}

#[test]
fn missing_or_empty_source_never_reuses_an_existing_runtime() {
    for selected in [None, Some([].as_slice())] {
        let fixture = RuntimeFixture::new(selected, Some(b"existing-runtime"));
        assert!(
            panic::catch_unwind(|| {
                build_support::stage_selected_runtime(&fixture.source, &fixture.destination);
            })
            .is_err()
        );
        assert_eq!(fs::read(&fixture.destination).unwrap(), b"existing-runtime");
    }
}

#[test]
fn identical_source_and_destination_never_truncate_the_runtime() {
    let fixture = RuntimeFixture::new(Some(b"selected-runtime"), None);
    build_support::stage_selected_runtime(&fixture.source, &fixture.source);
    assert_eq!(fs::read(&fixture.source).unwrap(), b"selected-runtime");
}

#[cfg(windows)]
fn lock_destination_against_replacement(path: &Path) -> fs::File {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_SHARE_READ: u32 = 0x0000_0001;
    let file = fs::OpenOptions::new()
        .read(true)
        .share_mode(FILE_SHARE_READ)
        .open(path)
        .unwrap();
    let write_error = fs::OpenOptions::new().write(true).open(path).unwrap_err();
    assert_eq!(write_error.raw_os_error(), Some(32));
    file
}

#[cfg(windows)]
#[test]
fn identical_locked_windows_runtime_needs_no_replacement() {
    let fixture = RuntimeFixture::new(Some(b"selected-runtime"), Some(b"selected-runtime"));
    let _locked = lock_destination_against_replacement(&fixture.destination);
    build_support::stage_selected_runtime(&fixture.source, &fixture.destination);
    assert_eq!(fs::read(&fixture.destination).unwrap(), b"selected-runtime");
}

#[cfg(windows)]
#[test]
fn different_locked_windows_runtime_still_fails_closed() {
    let fixture = RuntimeFixture::new(Some(b"selected-runtime"), Some(b"previous-runtime"));
    let _locked = lock_destination_against_replacement(&fixture.destination);
    assert!(
        panic::catch_unwind(|| {
            build_support::stage_selected_runtime(&fixture.source, &fixture.destination);
        })
        .is_err()
    );
    assert_eq!(fs::read(&fixture.destination).unwrap(), b"previous-runtime");
}

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
    let source = include_str!("../build_support.rs");
    assert!(
        include_str!("../build.rs")
            .contains("build_support::stage_selected_runtime(&lib_source_path, &lib_dest_path)")
    );
    let stage = source
        .split_once("fn stage_selected_runtime(")
        .expect("missing selected-runtime staging helper")
        .1;

    for required in [
        "source.is_file()",
        "source_len == 0",
        "fs::copy",
        "copied_bytes != source_len",
        "same_runtime_contents(source, destination, source_len)",
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
