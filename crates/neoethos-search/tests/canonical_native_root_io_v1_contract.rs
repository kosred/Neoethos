#![cfg(target_os = "linux")]
use neoethos_core::Settings;
use neoethos_search::{
    CanonicalNativeDiscoveryRequestErrorV1 as Error, CanonicalResearchContractArtifactRefV1,
    SealedCanonicalRootV1, load_canonical_research_contract_artifact_v1,
};
use sha2::{Digest, Sha256};
use std::ffi::CString;
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use tempfile::TempDir;
const LIMIT: usize = 8 * 1024 * 1024;
fn settings(root: &Path) -> Settings {
    let mut settings = Settings::default();
    settings.system.data_dir = root.to_path_buf();
    settings
}
fn exact_sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
type LoadResult = Result<(), Error>;
fn load(root: &Path, relative: &str, sha: String) -> LoadResult {
    let sealed = SealedCanonicalRootV1::from_startup_settings(&settings(root))?;
    let reference = CanonicalResearchContractArtifactRefV1::checked_new(relative, sha)?;
    load_canonical_research_contract_artifact_v1(&sealed, reference).map(|_| ())
}
fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let source = source.split_once(start).unwrap().1;
    source.split_once(end).unwrap().0
}
#[test]
fn component_and_final_symlinks_are_never_followed() {
    let root = TempDir::new().unwrap();
    let outside = TempDir::new().unwrap();
    let root_alias_parent = TempDir::new().unwrap();
    let root_alias = root_alias_parent.path().join("root-link");
    symlink(root.path(), &root_alias).unwrap();
    let error = SealedCanonicalRootV1::from_startup_settings(&settings(&root_alias))
        .err()
        .unwrap();
    assert!(matches!(error, Error::CanonicalRootUnavailable(_)));
    let bytes = b"outside artifact";
    fs::write(outside.path().join("contract.json"), bytes).unwrap();
    fs::create_dir(root.path().join("inside-dir")).unwrap();
    fs::write(root.path().join("inside-dir/contract.json"), bytes).unwrap();
    fs::write(root.path().join("inside.json"), bytes).unwrap();
    for (target, link) in [
        (outside.path().to_owned(), "component"),
        (outside.path().join("contract.json"), "final.json"),
        (PathBuf::from("inside-dir"), "component-in"),
        (PathBuf::from("inside.json"), "final-in"),
    ] {
        symlink(target, root.path().join(link)).unwrap();
    }
    for relative in
        "component/contract.json|final.json|component-in/contract.json|final-in".split('|')
    {
        let error = load(root.path(), relative, exact_sha(bytes)).unwrap_err();
        assert!(matches!(error, Error::UnsafeLink | Error::EscapeOrMount));
    }
    let mount = load(Path::new("/"), "proc/cpuinfo", "0".repeat(64));
    assert_eq!(mount.unwrap_err(), Error::EscapeOrMount);
}
#[test]
fn directory_fifo_and_sparse_overflow_are_rejected_without_blocking() {
    let root = TempDir::new().unwrap();
    fs::create_dir(root.path().join("directory")).unwrap();
    let fifo = root.path().join("fifo");
    let fifo_c = CString::new(fifo.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(fifo_c.as_ptr(), 0o600) }, 0);
    assert_eq!(
        load(root.path(), "directory", "0".repeat(64)).unwrap_err(),
        Error::NonRegularArtifact
    );
    let root_path = root.path().to_owned();
    let (tx, rx) = mpsc::channel();
    let worker = thread::spawn(move || {
        tx.send(load(&root_path, "fifo", "0".repeat(64))).unwrap();
    });
    let fifo_result = rx.recv_timeout(Duration::from_secs(2)).unwrap();
    worker.join().unwrap();
    assert_eq!(fifo_result.unwrap_err(), Error::NonRegularArtifact);
    let exact = vec![b' '; LIMIT];
    fs::write(root.path().join("exact.json"), &exact).unwrap();
    let error = load(root.path(), "exact.json", exact_sha(&exact)).unwrap_err();
    assert!(matches!(error, Error::ContractDecode(_)));
    let sparse = fs::File::create(root.path().join("oversized.json")).unwrap();
    sparse.set_len(LIMIT as u64 + 2).unwrap();
    let error = load(root.path(), "oversized.json", "0".repeat(64)).unwrap_err();
    assert!(matches!(
        error,
        Error::ArtifactTooLarge { maximum, observed }
            if maximum == LIMIT as u64 && observed == maximum + 2
    ));
}
#[test]
fn retained_root_detects_rename_and_path_replacement_before_load() {
    let root = TempDir::new().unwrap();
    fs::write(root.path().join("artifact.json"), b"inside").unwrap();
    let sealed = SealedCanonicalRootV1::from_startup_settings(&settings(root.path())).unwrap();
    let original = root.path().to_owned();
    let moved = original.with_extension("moved");
    fs::rename(&original, &moved).unwrap();
    fs::create_dir(&original).unwrap();
    let result = load_canonical_research_contract_artifact_v1(
        &sealed,
        CanonicalResearchContractArtifactRefV1::checked_new("artifact.json", exact_sha(b"inside"))
            .unwrap(),
    );
    fs::remove_dir(&original).unwrap();
    fs::rename(moved, original).unwrap();
    assert!(matches!(result, Err(Error::RaceDetected)));
}
#[test]
fn non_linux_stubs_return_before_input_inspection() {
    let root = section(
        include_str!("../src/canonical_native_root_io_v1.rs"),
        "#[cfg(not(target_os = \"linux\"))]\nimpl SealedCanonicalRootV1 {",
        "#[cfg(target_os = \"linux\")]\nmod linux",
    );
    let loader = section(
        include_str!("../src/canonical_native_discovery_request_v1.rs"),
        "pub fn load_canonical_research_contract_artifact_v1",
        "#[cfg(target_os = \"linux\")]",
    );
    for block in [root, loader] {
        assert!(block.contains("UnsupportedPlatform"));
        for access in "data_dir|Path::|relative_path|reference.validate|read_canonical".split('|') {
            assert!(!block.contains(access));
        }
    }
}
