use std::fs::{self, File};
use std::io::Write;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::symlink;
use std::path::Path;
use std::sync::{Arc, Barrier};
use std::thread;

use neoethos_core::Settings;
use tempfile::TempDir;

use super::{
    BoundedCanonicalArtifactWriterV1, CanonicalArtifactFinalStateV1,
    CanonicalArtifactPublishDispositionV1, CanonicalArtifactPublishErrorKindV1,
    CanonicalArtifactPublishGateRejectionV1, CanonicalArtifactPublishStateV1,
    CanonicalArtifactTemporaryStateV1, MAX_CANONICAL_ATOMIC_PUBLISH_BYTES_V1,
    SealedCanonicalRootV1, VerifiedDirectoryV1, artifact_identity,
    cleanup_temporary_name_inode_safe_v1, finish_new_install_v1, fstat,
    publish_canonical_artifact_create_new_v1,
    publish_canonical_artifact_create_new_with_pre_link_test_hook_v1, root_identity,
};

const RESULT: &str = "research/native-discovery/v1/cngr1-test.json";

fn settings(root: &Path) -> Settings {
    let mut settings = Settings::default();
    settings.system.data_dir = root.to_path_buf();
    settings
}

fn sealed(root: &TempDir) -> SealedCanonicalRootV1 {
    SealedCanonicalRootV1::from_startup_settings(&settings(root.path())).unwrap()
}

fn publish(
    root: &SealedCanonicalRootV1,
    bytes: &[u8],
) -> Result<super::CanonicalArtifactPublishReceiptV1, super::CanonicalArtifactPublishErrorV1> {
    publish_canonical_artifact_create_new_v1(
        root,
        RESULT,
        |writer| writer.write_all(bytes),
        || Ok(()),
    )
}

fn temp_names(parent: &Path) -> Vec<String> {
    fs::read_dir(parent)
        .map(|entries| {
            entries
                .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                .filter(|name| name.starts_with(".neoethos-canonical-tmp-"))
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn bounded_writer_rejects_the_whole_overflowing_write() {
    assert_eq!(MAX_CANONICAL_ATOMIC_PUBLISH_BYTES_V1, 512 * 1024 * 1024);
    let mut sink = Vec::new();
    let (written, attempted) = {
        let mut writer = BoundedCanonicalArtifactWriterV1::new(&mut sink, 4);
        assert_eq!(writer.write(b"abc").unwrap(), 3);
        let error = writer.write(b"de").unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::FileTooLarge);
        (writer.bytes_written(), writer.overflow_attempted_bytes())
    };
    assert_eq!(sink, b"abc");
    assert_eq!(written, 3);
    assert_eq!(attempted, Some(5));
}

#[test]
fn concurrent_identical_publishers_install_once_and_accept_the_exact_winner() {
    let root = TempDir::new().unwrap();
    let root_path = root.path().to_path_buf();
    let barrier = Arc::new(Barrier::new(2));
    let workers: Vec<_> = (0..2)
        .map(|_| {
            let root_path = root_path.clone();
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                let sealed =
                    SealedCanonicalRootV1::from_startup_settings(&settings(&root_path)).unwrap();
                barrier.wait();
                publish(&sealed, b"identical")
            })
        })
        .collect();
    let outcomes: Vec<_> = workers
        .into_iter()
        .map(|worker| worker.join().unwrap().unwrap().disposition())
        .collect();

    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| **outcome == CanonicalArtifactPublishDispositionV1::Installed)
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| {
                **outcome == CanonicalArtifactPublishDispositionV1::ExistingIdentical
            })
            .count(),
        1
    );
    assert_eq!(fs::read(root.path().join(RESULT)).unwrap(), b"identical");
    assert!(temp_names(&root.path().join("research/native-discovery/v1")).is_empty());
}

#[test]
fn concurrent_different_publishers_never_replace_the_first_complete_object() {
    let root = TempDir::new().unwrap();
    let root_path = root.path().to_path_buf();
    let barrier = Arc::new(Barrier::new(2));
    let workers: Vec<_> = [b"first".as_slice(), b"other".as_slice()]
        .into_iter()
        .map(|bytes| {
            let root_path = root_path.clone();
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                let sealed =
                    SealedCanonicalRootV1::from_startup_settings(&settings(&root_path)).unwrap();
                barrier.wait();
                publish(&sealed, bytes)
            })
        })
        .collect();
    let outcomes: Vec<_> = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .collect();

    assert_eq!(outcomes.iter().filter(|outcome| outcome.is_ok()).count(), 1);
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| matches!(
                outcome,
                Err(error)
                    if matches!(
                        error.kind(),
                        CanonicalArtifactPublishErrorKindV1::ExistingContentMismatch
                    )
            ))
            .count(),
        1
    );
    let installed = fs::read(root.path().join(RESULT)).unwrap();
    assert!(installed == b"first" || installed == b"other");
    assert!(temp_names(&root.path().join("research/native-discovery/v1")).is_empty());
}

#[test]
fn publisher_creates_components_and_installs_durable_exact_bytes() {
    let root = TempDir::new().unwrap();
    let receipt = publish(&sealed(&root), b"{\"ok\":true}").unwrap();

    assert_eq!(
        receipt.disposition(),
        CanonicalArtifactPublishDispositionV1::Installed
    );
    assert_eq!(receipt.bytes_written(), 11);
    assert_eq!(receipt.relative_path(), RESULT);
    assert_eq!(
        fs::read(root.path().join(RESULT)).unwrap(),
        b"{\"ok\":true}"
    );
    assert!(temp_names(&root.path().join("research/native-discovery/v1")).is_empty());
}

#[test]
fn every_output_directory_component_rejects_a_symlink_escape() {
    for component in ["research", "native-discovery", "v1"] {
        let root = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let mut parent = root.path().to_path_buf();
        for current in ["research", "native-discovery", "v1"] {
            if current == component {
                symlink(outside.path(), parent.join(current)).unwrap();
                break;
            }
            parent.push(current);
            fs::create_dir(&parent).unwrap();
        }

        let error = publish(&sealed(&root), b"never outside").unwrap_err();
        assert!(matches!(
            error.kind(),
            CanonicalArtifactPublishErrorKindV1::UnsafeLink
                | CanonicalArtifactPublishErrorKindV1::EscapeOrMount
        ));
        assert!(!outside.path().join("cngr1-test.json").exists());
    }
}

#[test]
fn an_existing_winner_is_accepted_only_when_exactly_byte_identical() {
    let root = TempDir::new().unwrap();
    let sealed = sealed(&root);
    let first = publish(&sealed, b"winner").unwrap();
    assert_eq!(
        first.disposition(),
        CanonicalArtifactPublishDispositionV1::Installed
    );

    let identical = publish(&sealed, b"winner").unwrap();
    assert_eq!(
        identical.disposition(),
        CanonicalArtifactPublishDispositionV1::ExistingIdentical
    );

    let mismatch = publish(&sealed, b"loser!").unwrap_err();
    assert_eq!(
        mismatch.state(),
        CanonicalArtifactPublishStateV1::new(
            CanonicalArtifactFinalStateV1::ExistingDifferent,
            CanonicalArtifactTemporaryStateV1::RemovedDurable,
        )
    );
    assert_eq!(
        mismatch.state().temporary_state(),
        CanonicalArtifactTemporaryStateV1::RemovedDurable
    );
    assert!(matches!(
        mismatch.kind(),
        CanonicalArtifactPublishErrorKindV1::ExistingContentMismatch
    ));
    assert_eq!(fs::read(root.path().join(RESULT)).unwrap(), b"winner");
}

#[test]
fn rejected_preinstall_gate_leaves_no_final_or_temporary_name() {
    let root = TempDir::new().unwrap();
    let sealed = sealed(&root);
    let error = publish_canonical_artifact_create_new_v1(
        &sealed,
        RESULT,
        |writer| writer.write_all(b"staged"),
        || Err(CanonicalArtifactPublishGateRejectionV1::Cancelled),
    )
    .unwrap_err();

    assert!(matches!(
        error.kind(),
        CanonicalArtifactPublishErrorKindV1::PreInstallRejected(
            CanonicalArtifactPublishGateRejectionV1::Cancelled
        )
    ));
    assert_eq!(
        error.state(),
        CanonicalArtifactPublishStateV1::new(
            CanonicalArtifactFinalStateV1::NotInstalled,
            CanonicalArtifactTemporaryStateV1::RemovedDurable,
        )
    );
    assert!(!root.path().join(RESULT).exists());
    assert!(temp_names(&root.path().join("research/native-discovery/v1")).is_empty());
}

#[test]
fn gate_cannot_mutate_staged_bytes_before_a_new_install() {
    let root = TempDir::new().unwrap();
    let sealed = sealed(&root);
    let parent = root.path().join("research/native-discovery/v1");
    let error = publish_canonical_artifact_create_new_v1(
        &sealed,
        RESULT,
        |writer| writer.write_all(b"good!!"),
        || {
            let names = temp_names(&parent);
            assert_eq!(names.len(), 1);
            fs::write(parent.join(&names[0]), b"evil!!").unwrap();
            Ok(())
        },
    )
    .unwrap_err();

    assert!(matches!(
        error.kind(),
        CanonicalArtifactPublishErrorKindV1::RaceDetected
    ));
    assert!(!root.path().join(RESULT).exists());
    assert!(temp_names(&parent).is_empty());
}

#[test]
fn gate_cannot_mutate_a_loser_into_an_identical_existing_winner() {
    let root = TempDir::new().unwrap();
    let sealed = sealed(&root);
    let parent = root.path().join("research/native-discovery/v1");
    publish(&sealed, b"winner").unwrap();

    let error = publish_canonical_artifact_create_new_v1(
        &sealed,
        RESULT,
        |writer| writer.write_all(b"loser!"),
        || {
            let names = temp_names(&parent);
            assert_eq!(names.len(), 1);
            fs::write(parent.join(&names[0]), b"winner").unwrap();
            Ok(())
        },
    )
    .unwrap_err();

    assert!(matches!(
        error.kind(),
        CanonicalArtifactPublishErrorKindV1::RaceDetected
    ));
    assert_eq!(fs::read(root.path().join(RESULT)).unwrap(), b"winner");
    assert!(temp_names(&parent).is_empty());
}

#[test]
fn pre_link_name_replacement_never_installs_the_replacement_inode() {
    let root = TempDir::new().unwrap();
    let sealed = sealed(&root);
    let parent = root.path().join("research/native-discovery/v1");
    let error = publish_canonical_artifact_create_new_with_pre_link_test_hook_v1(
        &sealed,
        RESULT,
        |writer| writer.write_all(b"good!!"),
        || Ok(()),
        || {
            let names = temp_names(&parent);
            assert_eq!(names.len(), 1);
            let temporary = parent.join(&names[0]);
            fs::remove_file(&temporary).unwrap();
            fs::write(temporary, b"evil!!").unwrap();
        },
    )
    .unwrap_err();

    assert!(matches!(
        error.kind(),
        CanonicalArtifactPublishErrorKindV1::RaceDetected
    ));
    assert!(!root.path().join(RESULT).exists());
}

#[test]
fn a_root_rename_after_the_gate_cleans_the_staged_inode_before_failing() {
    let root = TempDir::new().unwrap();
    let sealed = sealed(&root);
    let original = root.path().to_path_buf();
    let moved = original.with_extension("moved-during-publish");
    let error = publish_canonical_artifact_create_new_v1(
        &sealed,
        RESULT,
        |writer| writer.write_all(b"staged"),
        || {
            fs::rename(&original, &moved).unwrap();
            Ok(())
        },
    )
    .unwrap_err();

    assert!(matches!(
        error.kind(),
        CanonicalArtifactPublishErrorKindV1::RaceDetected
    ));
    assert_eq!(
        error.state(),
        CanonicalArtifactPublishStateV1::new(
            CanonicalArtifactFinalStateV1::NotInstalled,
            CanonicalArtifactTemporaryStateV1::RemovedDurable,
        )
    );
    let moved_parent = moved.join("research/native-discovery/v1");
    assert!(!moved_parent.join("cngr1-test.json").exists());
    assert!(temp_names(&moved_parent).is_empty());
    fs::rename(moved, original).unwrap();
}

#[test]
fn an_unsafe_existing_final_name_is_rejected_after_staged_cleanup() {
    let root = TempDir::new().unwrap();
    let outside = TempDir::new().unwrap();
    let parent = root.path().join("research/native-discovery/v1");
    fs::create_dir_all(&parent).unwrap();
    fs::write(outside.path().join("outside.json"), b"outside").unwrap();
    symlink(
        outside.path().join("outside.json"),
        root.path().join(RESULT),
    )
    .unwrap();

    let error = publish(&sealed(&root), b"staged").unwrap_err();
    assert!(matches!(
        error.kind(),
        CanonicalArtifactPublishErrorKindV1::UnsafeLink
            | CanonicalArtifactPublishErrorKindV1::EscapeOrMount
    ));
    assert!(temp_names(&parent).is_empty());
    assert_eq!(
        fs::read(outside.path().join("outside.json")).unwrap(),
        b"outside"
    );
}

#[test]
fn post_link_directory_sync_failure_reports_both_partial_states() {
    let root = TempDir::new().unwrap();
    let sealed = sealed(&root);
    let root_c = std::ffi::CString::new(root.path().as_os_str().as_bytes()).unwrap();
    let raw_fd = unsafe {
        libc::open(
            root_c.as_ptr(),
            libc::O_PATH | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    assert!(raw_fd >= 0);
    let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
    let identity = root_identity(&fstat(fd.as_raw_fd()).unwrap());
    let parent = VerifiedDirectoryV1 {
        fd,
        identity,
        expected_path: root.path().to_path_buf(),
    };
    fs::write(root.path().join("temporary"), b"linked").unwrap();
    fs::hard_link(root.path().join("temporary"), root.path().join("final")).unwrap();
    let temporary_file = File::open(root.path().join("temporary")).unwrap();
    let linked_identity = artifact_identity(&fstat(temporary_file.as_raw_fd()).unwrap());
    let staged_identity = super::ArtifactIdentity {
        links: 1,
        ..linked_identity
    };
    let snapshot = super::ExactStagedSnapshotV1 {
        file: temporary_file.try_clone().unwrap(),
        identity: staged_identity,
        name: Some(std::ffi::CString::new("temporary").unwrap()),
    };

    let error = finish_new_install_v1(
        &sealed,
        &parent,
        "final",
        &std::ffi::CString::new("final").unwrap(),
        &std::ffi::CString::new("temporary").unwrap(),
        staged_identity,
        &snapshot,
        6,
    )
    .unwrap_err();
    assert_eq!(
        error.state(),
        CanonicalArtifactPublishStateV1::new(
            CanonicalArtifactFinalStateV1::InstalledSyncPending,
            CanonicalArtifactTemporaryStateV1::Present,
        )
    );
    assert!(matches!(
        error.kind(),
        CanonicalArtifactPublishErrorKindV1::Io {
            operation: "fsync installed artifact parent",
            ..
        }
    ));
}

#[test]
fn cleanup_refuses_to_unlink_a_replaced_temporary_inode() {
    let root = TempDir::new().unwrap();
    let parent = root.path().join("parent");
    fs::create_dir(&parent).unwrap();
    let parent_file = File::open(&parent).unwrap();
    let name = std::ffi::CString::new("temporary").unwrap();
    let temporary = parent.join("temporary");
    fs::write(&temporary, b"ours").unwrap();
    let ours = File::open(&temporary).unwrap();
    let expected = artifact_identity(&fstat(ours.as_raw_fd()).unwrap());

    fs::remove_file(&temporary).unwrap();
    fs::write(&temporary, b"replacement").unwrap();
    let state = CanonicalArtifactPublishStateV1::new(
        CanonicalArtifactFinalStateV1::InstalledDurable,
        CanonicalArtifactTemporaryStateV1::Present,
    );
    let error =
        cleanup_temporary_name_inode_safe_v1(parent_file.as_raw_fd(), &name, expected, state)
            .unwrap_err();

    assert!(matches!(
        error.kind(),
        CanonicalArtifactPublishErrorKindV1::RaceDetected
    ));
    assert_eq!(error.state(), state);
    assert_eq!(fs::read(temporary).unwrap(), b"replacement");
}

#[test]
fn relative_target_rejects_absolute_and_dot_components() {
    let root = TempDir::new().unwrap();
    let sealed = sealed(&root);
    for path in [
        "",
        "/absolute.json",
        "../escape.json",
        "a/../escape.json",
        "./x.json",
    ] {
        let error = publish_canonical_artifact_create_new_v1(
            &sealed,
            path,
            |writer| writer.write_all(b"x"),
            || Ok(()),
        )
        .unwrap_err();
        assert!(matches!(
            error.kind(),
            CanonicalArtifactPublishErrorKindV1::InvalidRelativePath
        ));
    }
}
