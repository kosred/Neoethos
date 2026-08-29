use super::sealed_writer_v1_tests::with_fully_populated_sealed_result_v1;
use super::*;
use crate::canonical_native_generation_zero_publication_v1::{
    CanonicalNativeGenerationZeroPublicationErrorKindV1,
    CanonicalNativeGenerationZeroPublicationFinalStateV1,
    CanonicalNativeGenerationZeroPublicationGateRejectionV1,
    CanonicalNativeGenerationZeroPublicationTemporaryStateV1,
    publish_canonical_native_generation_zero_research_result_v1,
};
use crate::canonical_native_root_io_v1::SealedCanonicalRootV1;
use neoethos_core::Settings;
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::fs;
use tempfile::TempDir;

fn sealed_root_v1(root: &TempDir) -> SealedCanonicalRootV1 {
    let mut settings = Settings::default();
    settings.system.data_dir = root.path().to_owned();
    SealedCanonicalRootV1::from_startup_settings(&settings).unwrap()
}

fn expected_relative_path_v1(evidence_identity_sha256: &str) -> String {
    format!("research/native-discovery/v1/cngr1-{evidence_identity_sha256}.json")
}

#[test]
fn installed_publication_returns_exact_bounded_owned_summary() {
    with_fully_populated_sealed_result_v1(|view, seal| {
        let root_dir = TempDir::new().unwrap();
        let root = sealed_root_v1(&root_dir);
        let gate_calls = Cell::new(0_u32);
        let receipt =
            publish_canonical_native_generation_zero_research_result_v1(&root, view, seal, || {
                gate_calls.set(gate_calls.get() + 1);
                Ok(())
            })
            .unwrap();

        let expected_relative = expected_relative_path_v1(view.evidence_identity_sha256());
        assert_eq!(receipt.relative_path(), expected_relative);
        assert_eq!(receipt.byte_count(), seal.byte_count());
        assert_eq!(receipt.file_sha256(), seal.sha256());
        assert!(!receipt.reused_identical());
        assert_eq!(
            receipt.final_state(),
            CanonicalNativeGenerationZeroPublicationFinalStateV1::InstalledDurable
        );
        assert_eq!(
            receipt.temporary_state(),
            CanonicalNativeGenerationZeroPublicationTemporaryStateV1::RemovedDurable
        );
        assert_eq!(
            receipt.evidence_identity_sha256(),
            view.evidence_identity_sha256()
        );
        assert_eq!(
            receipt.financial_input_receipt_identity_sha256(),
            view.financial_input_receipt_identity_sha256()
        );
        assert_eq!(
            receipt.native_input_receipt_identity_sha256(),
            view.milestone().native_input_receipt_identity_sha256()
        );
        assert_eq!(
            receipt.population_sizing_receipt_identity_sha256(),
            view.milestone().population_sizing_receipt_identity_sha256()
        );
        assert_eq!(
            receipt.resolved_population(),
            view.milestone().resolved_population()
        );
        assert_eq!(receipt.term_cap(), view.milestone().term_cap());
        assert_eq!(
            receipt.selected_device_ordinal(),
            view.milestone().selected_device_ordinal()
        );
        assert_eq!(receipt.engine(), view.milestone().engine());
        let counters = view.milestone().residency_counters();
        assert_eq!(receipt.parent_h2d_bytes(), counters.parent_upload_bytes());
        assert_eq!(
            receipt.adaptive_h2d_bytes(),
            counters.adaptive_upload_bytes()
        );
        assert_eq!(receipt.metric_rows(), counters.metric_rows_readback_rows());
        assert_eq!(
            receipt.metric_bytes(),
            counters.metric_rows_readback_bytes()
        );
        assert!(receipt.consumer_completion_confirmed());
        assert!(!receipt.replay_identity_sealed());
        assert_eq!(gate_calls.get(), 1);

        let bytes = fs::read(root_dir.path().join(receipt.relative_path())).unwrap();
        assert_eq!(bytes.len() as u64, receipt.byte_count());
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            receipt.file_sha256()
        );
    });
}

#[test]
fn identical_content_address_is_reused_without_replacement() {
    with_fully_populated_sealed_result_v1(|view, seal| {
        let root_dir = TempDir::new().unwrap();
        let root = sealed_root_v1(&root_dir);
        let installed =
            publish_canonical_native_generation_zero_research_result_v1(&root, view, seal, || {
                Ok(())
            })
            .unwrap();
        let installed_bytes = fs::read(root_dir.path().join(installed.relative_path())).unwrap();
        let reused =
            publish_canonical_native_generation_zero_research_result_v1(&root, view, seal, || {
                Ok(())
            })
            .unwrap();

        assert!(reused.reused_identical());
        assert_eq!(
            reused.final_state(),
            CanonicalNativeGenerationZeroPublicationFinalStateV1::ExistingIdentical
        );
        assert_eq!(installed.relative_path(), reused.relative_path());
        assert_eq!(installed.file_sha256(), reused.file_sha256());
        assert_eq!(
            fs::read(root_dir.path().join(reused.relative_path())).unwrap(),
            installed_bytes
        );
    });
}

#[test]
fn rejected_pre_install_gate_reports_not_installed_and_durable_cleanup() {
    with_fully_populated_sealed_result_v1(|view, seal| {
        let root_dir = TempDir::new().unwrap();
        let root = sealed_root_v1(&root_dir);
        let error =
            publish_canonical_native_generation_zero_research_result_v1(&root, view, seal, || {
                Err(CanonicalNativeGenerationZeroPublicationGateRejectionV1::Cancelled)
            })
            .unwrap_err();

        assert_eq!(
            error.kind(),
            &CanonicalNativeGenerationZeroPublicationErrorKindV1::PreInstallRejected(
                CanonicalNativeGenerationZeroPublicationGateRejectionV1::Cancelled,
            )
        );
        assert_eq!(
            error.final_state(),
            CanonicalNativeGenerationZeroPublicationFinalStateV1::NotInstalled
        );
        assert_eq!(
            error.temporary_state(),
            CanonicalNativeGenerationZeroPublicationTemporaryStateV1::RemovedDurable
        );
        assert!(
            !root_dir
                .path()
                .join(expected_relative_path_v1(view.evidence_identity_sha256()))
                .exists()
        );
    });
}

#[test]
fn nonidentical_existing_content_address_is_preserved_and_rejected() {
    with_fully_populated_sealed_result_v1(|view, seal| {
        let root_dir = TempDir::new().unwrap();
        let relative = expected_relative_path_v1(view.evidence_identity_sha256());
        let path = root_dir.path().join(&relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"not-the-sealed-result").unwrap();
        let root = sealed_root_v1(&root_dir);
        let error =
            publish_canonical_native_generation_zero_research_result_v1(&root, view, seal, || {
                Ok(())
            })
            .unwrap_err();

        assert_eq!(
            error.kind(),
            &CanonicalNativeGenerationZeroPublicationErrorKindV1::ExistingContentMismatch
        );
        assert_eq!(
            error.final_state(),
            CanonicalNativeGenerationZeroPublicationFinalStateV1::ExistingDifferent
        );
        assert_eq!(
            error.temporary_state(),
            CanonicalNativeGenerationZeroPublicationTemporaryStateV1::RemovedDurable
        );
        assert_eq!(fs::read(path).unwrap(), b"not-the-sealed-result");
    });
}

#[test]
fn mismatched_precomputed_seal_fails_before_install() {
    with_fully_populated_sealed_result_v1(|view, seal| {
        let root_dir = TempDir::new().unwrap();
        let root = sealed_root_v1(&root_dir);
        let forged = CanonicalNativeGenerationZeroCompactJsonSealV1 {
            byte_count: seal.byte_count(),
            sha256: "0".repeat(64),
        };
        let error = publish_canonical_native_generation_zero_research_result_v1(
            &root,
            view,
            &forged,
            || Ok(()),
        )
        .unwrap_err();

        assert!(matches!(
            error.kind(),
            CanonicalNativeGenerationZeroPublicationErrorKindV1::SealMismatch { .. }
        ));
        assert_eq!(
            error.final_state(),
            CanonicalNativeGenerationZeroPublicationFinalStateV1::NotInstalled
        );
        assert_eq!(
            error.temporary_state(),
            CanonicalNativeGenerationZeroPublicationTemporaryStateV1::RemovedDurable
        );
        assert!(
            !root_dir
                .path()
                .join(expected_relative_path_v1(view.evidence_identity_sha256()))
                .exists()
        );
    });
}

#[test]
fn adapter_source_streams_the_borrowed_view_and_has_no_public_or_population_copy_seam() {
    const SOURCE: &str = include_str!("canonical_native_generation_zero_publication_v1.rs");
    const LIB: &str = include_str!("lib.rs");

    for required in [
        "publish_canonical_artifact_create_new_v1(",
        "write_canonical_native_generation_zero_research_result_v1(",
        "research/native-discovery/v1/cngr1-",
        "expected_seal.byte_count()",
        "actual_seal != *expected_seal",
    ] {
        assert!(
            SOURCE.contains(required),
            "missing adapter source marker `{required}`"
        );
    }
    for forbidden in [
        "SearchResult",
        ".search_result()",
        ".genes",
        ".metrics",
        ".to_vec()",
        "collect::<Vec",
        "serde_json::",
        "std::fs::",
        "File::create",
        "pub fn publish_canonical_native_generation_zero",
    ] {
        assert!(
            !SOURCE.contains(forbidden),
            "forbidden adapter seam `{forbidden}`"
        );
    }
    assert!(LIB.contains(
        "#[cfg(all(feature = \"gpu-cuda\", target_os = \"linux\"))]\nmod canonical_native_generation_zero_publication_v1;"
    ));
    assert!(!LIB.contains("pub use canonical_native_generation_zero_publication_v1"));
}
