#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Chunk 3 is the first production caller of the sealed result publisher"
    )
)]

use std::fmt;
use std::io;

use crate::canonical_native_generation_zero_result_v1::{
    CanonicalNativeGenerationZeroCompactJsonSealV1,
    CanonicalNativeGenerationZeroResearchResultViewV1,
    write_canonical_native_generation_zero_research_result_v1,
};
use crate::canonical_native_root_io_v1::{
    CanonicalArtifactFinalStateV1, CanonicalArtifactPublishDispositionV1,
    CanonicalArtifactPublishErrorKindV1, CanonicalArtifactPublishGateRejectionV1,
    CanonicalArtifactTemporaryStateV1, SealedCanonicalRootV1,
    publish_canonical_artifact_create_new_v1,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CanonicalNativeGenerationZeroPublicationGateRejectionV1 {
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CanonicalNativeGenerationZeroPublicationFinalStateV1 {
    NotInstalled,
    InstalledSyncPending,
    InstalledDurable,
    ExistingIdentical,
    ExistingDifferent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CanonicalNativeGenerationZeroPublicationTemporaryStateV1 {
    NotCreated,
    Present,
    RemovedSyncPending,
    RemovedDurable,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CanonicalNativeGenerationZeroPublicationErrorKindV1 {
    InvalidEvidenceIdentity,
    ResultWrite(String),
    SealMismatch {
        expected_byte_count: u64,
        actual_byte_count: u64,
        expected_sha256: String,
        actual_sha256: String,
    },
    ReceiptInvariant(String),
    InvalidRelativePath,
    SecureResolutionUnavailable(String),
    UnsafeLink,
    EscapeOrMount,
    RaceDetected,
    NonRegularArtifact,
    ArtifactTooLarge {
        maximum: u64,
        attempted: u64,
    },
    PreInstallRejected(CanonicalNativeGenerationZeroPublicationGateRejectionV1),
    ExistingContentMismatch,
    Io {
        operation: &'static str,
        message: String,
    },
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct CanonicalNativeGenerationZeroPublicationErrorV1 {
    kind: CanonicalNativeGenerationZeroPublicationErrorKindV1,
    final_state: CanonicalNativeGenerationZeroPublicationFinalStateV1,
    temporary_state: CanonicalNativeGenerationZeroPublicationTemporaryStateV1,
}

impl CanonicalNativeGenerationZeroPublicationErrorV1 {
    pub(crate) const fn kind(&self) -> &CanonicalNativeGenerationZeroPublicationErrorKindV1 {
        &self.kind
    }

    pub(crate) const fn final_state(&self) -> CanonicalNativeGenerationZeroPublicationFinalStateV1 {
        self.final_state
    }

    pub(crate) const fn temporary_state(
        &self,
    ) -> CanonicalNativeGenerationZeroPublicationTemporaryStateV1 {
        self.temporary_state
    }
}

impl fmt::Display for CanonicalNativeGenerationZeroPublicationErrorV1 {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            output,
            "canonical native Generation-zero publication {:?}/{:?}: {:?}",
            self.final_state, self.temporary_state, self.kind
        )
    }
}

impl std::error::Error for CanonicalNativeGenerationZeroPublicationErrorV1 {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CanonicalNativeGenerationZeroPublicationReceiptV1 {
    relative_path: String,
    byte_count: u64,
    file_sha256: String,
    reused_identical: bool,
    final_state: CanonicalNativeGenerationZeroPublicationFinalStateV1,
    temporary_state: CanonicalNativeGenerationZeroPublicationTemporaryStateV1,
    evidence_identity_sha256: String,
    financial_input_receipt_identity_sha256: String,
    native_input_receipt_identity_sha256: String,
    population_sizing_receipt_identity_sha256: String,
    resolved_population: usize,
    term_cap: usize,
    selected_device_ordinal: u32,
    engine: String,
    parent_h2d_bytes: u64,
    adaptive_h2d_bytes: u64,
    metric_rows: u64,
    metric_bytes: u64,
    consumer_completion_confirmed: bool,
    replay_identity_sealed: bool,
}

impl CanonicalNativeGenerationZeroPublicationReceiptV1 {
    pub(crate) fn relative_path(&self) -> &str {
        &self.relative_path
    }

    pub(crate) const fn byte_count(&self) -> u64 {
        self.byte_count
    }

    pub(crate) fn file_sha256(&self) -> &str {
        &self.file_sha256
    }

    pub(crate) const fn reused_identical(&self) -> bool {
        self.reused_identical
    }

    pub(crate) const fn final_state(&self) -> CanonicalNativeGenerationZeroPublicationFinalStateV1 {
        self.final_state
    }

    pub(crate) const fn temporary_state(
        &self,
    ) -> CanonicalNativeGenerationZeroPublicationTemporaryStateV1 {
        self.temporary_state
    }

    pub(crate) fn evidence_identity_sha256(&self) -> &str {
        &self.evidence_identity_sha256
    }

    pub(crate) fn financial_input_receipt_identity_sha256(&self) -> &str {
        &self.financial_input_receipt_identity_sha256
    }

    pub(crate) fn native_input_receipt_identity_sha256(&self) -> &str {
        &self.native_input_receipt_identity_sha256
    }

    pub(crate) fn population_sizing_receipt_identity_sha256(&self) -> &str {
        &self.population_sizing_receipt_identity_sha256
    }

    pub(crate) const fn resolved_population(&self) -> usize {
        self.resolved_population
    }

    pub(crate) const fn term_cap(&self) -> usize {
        self.term_cap
    }

    pub(crate) const fn selected_device_ordinal(&self) -> u32 {
        self.selected_device_ordinal
    }

    pub(crate) fn engine(&self) -> &str {
        &self.engine
    }

    pub(crate) const fn parent_h2d_bytes(&self) -> u64 {
        self.parent_h2d_bytes
    }

    pub(crate) const fn adaptive_h2d_bytes(&self) -> u64 {
        self.adaptive_h2d_bytes
    }

    pub(crate) const fn metric_rows(&self) -> u64 {
        self.metric_rows
    }

    pub(crate) const fn metric_bytes(&self) -> u64 {
        self.metric_bytes
    }

    pub(crate) const fn consumer_completion_confirmed(&self) -> bool {
        self.consumer_completion_confirmed
    }

    pub(crate) const fn replay_identity_sealed(&self) -> bool {
        self.replay_identity_sealed
    }
}

struct SealMismatchV1 {
    expected_byte_count: u64,
    actual_byte_count: u64,
    expected_sha256: String,
    actual_sha256: String,
}

pub(crate) fn publish_canonical_native_generation_zero_research_result_v1(
    root: &SealedCanonicalRootV1,
    view: &CanonicalNativeGenerationZeroResearchResultViewV1<'_>,
    expected_seal: &CanonicalNativeGenerationZeroCompactJsonSealV1,
    pre_install_gate: impl FnOnce()
        -> Result<(), CanonicalNativeGenerationZeroPublicationGateRejectionV1>,
) -> Result<
    CanonicalNativeGenerationZeroPublicationReceiptV1,
    CanonicalNativeGenerationZeroPublicationErrorV1,
> {
    let evidence_identity = view.evidence_identity_sha256();
    if !is_canonical_lower_hex_sha256_v1(evidence_identity) {
        return Err(publication_error_v1(
            CanonicalNativeGenerationZeroPublicationErrorKindV1::InvalidEvidenceIdentity,
            CanonicalNativeGenerationZeroPublicationFinalStateV1::NotInstalled,
            CanonicalNativeGenerationZeroPublicationTemporaryStateV1::NotCreated,
        ));
    }
    let relative_path = format!("research/native-discovery/v1/cngr1-{evidence_identity}.json");
    let mut result_write_error = None;
    let mut seal_mismatch = None;
    let mut emitted_seal = None;
    let low_level = publish_canonical_artifact_create_new_v1(
        root,
        &relative_path,
        |writer| {
            let actual_seal = match write_canonical_native_generation_zero_research_result_v1(
                view,
                writer,
                expected_seal.byte_count(),
            ) {
                Ok(seal) => seal,
                Err(error) => {
                    result_write_error = Some(error.to_string());
                    return Err(io::Error::other(
                        "sealed Generation-zero result writer rejected the view",
                    ));
                }
            };
            if actual_seal != *expected_seal {
                seal_mismatch = Some(SealMismatchV1 {
                    expected_byte_count: expected_seal.byte_count(),
                    actual_byte_count: actual_seal.byte_count(),
                    expected_sha256: expected_seal.sha256().to_owned(),
                    actual_sha256: actual_seal.sha256().to_owned(),
                });
                return Err(io::Error::other(
                    "sealed Generation-zero result bytes mismatched the precomputed seal",
                ));
            }
            emitted_seal = Some(actual_seal);
            Ok(())
        },
        || match pre_install_gate() {
            Ok(()) => Ok(()),
            Err(CanonicalNativeGenerationZeroPublicationGateRejectionV1::Cancelled) => {
                Err(CanonicalArtifactPublishGateRejectionV1::Cancelled)
            }
        },
    );

    let low_level_receipt = match low_level {
        Ok(receipt) => receipt,
        Err(error) => {
            let final_state = map_final_state_v1(error.state().final_state());
            let temporary_state = map_temporary_state_v1(error.state().temporary_state());
            let kind = if let Some(mismatch) = seal_mismatch {
                CanonicalNativeGenerationZeroPublicationErrorKindV1::SealMismatch {
                    expected_byte_count: mismatch.expected_byte_count,
                    actual_byte_count: mismatch.actual_byte_count,
                    expected_sha256: mismatch.expected_sha256,
                    actual_sha256: mismatch.actual_sha256,
                }
            } else if let Some(message) = result_write_error {
                CanonicalNativeGenerationZeroPublicationErrorKindV1::ResultWrite(message)
            } else {
                map_low_level_error_kind_v1(error.kind())
            };
            return Err(publication_error_v1(kind, final_state, temporary_state));
        }
    };
    let actual_seal = emitted_seal.ok_or_else(|| {
        success_invariant_error_v1(
            "low-level publisher returned success before the result writer produced a seal",
            low_level_receipt.disposition(),
        )
    })?;
    if low_level_receipt.relative_path() != relative_path
        || low_level_receipt.bytes_written() != actual_seal.byte_count()
    {
        return Err(success_invariant_error_v1(
            "low-level publication receipt disagreed with the sealed path or byte count",
            low_level_receipt.disposition(),
        ));
    }

    let (reused_identical, final_state) = match low_level_receipt.disposition() {
        CanonicalArtifactPublishDispositionV1::Installed => (
            false,
            CanonicalNativeGenerationZeroPublicationFinalStateV1::InstalledDurable,
        ),
        CanonicalArtifactPublishDispositionV1::ExistingIdentical => (
            true,
            CanonicalNativeGenerationZeroPublicationFinalStateV1::ExistingIdentical,
        ),
    };
    let milestone = view.milestone();
    let counters = milestone.residency_counters();
    Ok(CanonicalNativeGenerationZeroPublicationReceiptV1 {
        relative_path: low_level_receipt.relative_path().to_owned(),
        byte_count: actual_seal.byte_count(),
        file_sha256: actual_seal.sha256().to_owned(),
        reused_identical,
        final_state,
        temporary_state: CanonicalNativeGenerationZeroPublicationTemporaryStateV1::RemovedDurable,
        evidence_identity_sha256: evidence_identity.to_owned(),
        financial_input_receipt_identity_sha256: view
            .financial_input_receipt_identity_sha256()
            .to_owned(),
        native_input_receipt_identity_sha256: milestone
            .native_input_receipt_identity_sha256()
            .to_owned(),
        population_sizing_receipt_identity_sha256: milestone
            .population_sizing_receipt_identity_sha256()
            .to_owned(),
        resolved_population: milestone.resolved_population(),
        term_cap: milestone.term_cap(),
        selected_device_ordinal: milestone.selected_device_ordinal(),
        engine: milestone.engine().to_owned(),
        parent_h2d_bytes: counters.parent_upload_bytes(),
        adaptive_h2d_bytes: counters.adaptive_upload_bytes(),
        metric_rows: counters.metric_rows_readback_rows(),
        metric_bytes: counters.metric_rows_readback_bytes(),
        consumer_completion_confirmed: milestone.consumer_completion_confirmed(),
        replay_identity_sealed: milestone.replay_identity_sealed(),
    })
}

fn is_canonical_lower_hex_sha256_v1(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn success_invariant_error_v1(
    detail: &str,
    disposition: CanonicalArtifactPublishDispositionV1,
) -> CanonicalNativeGenerationZeroPublicationErrorV1 {
    let final_state = match disposition {
        CanonicalArtifactPublishDispositionV1::Installed => {
            CanonicalNativeGenerationZeroPublicationFinalStateV1::InstalledDurable
        }
        CanonicalArtifactPublishDispositionV1::ExistingIdentical => {
            CanonicalNativeGenerationZeroPublicationFinalStateV1::ExistingIdentical
        }
    };
    publication_error_v1(
        CanonicalNativeGenerationZeroPublicationErrorKindV1::ReceiptInvariant(detail.to_owned()),
        final_state,
        CanonicalNativeGenerationZeroPublicationTemporaryStateV1::RemovedDurable,
    )
}

fn publication_error_v1(
    kind: CanonicalNativeGenerationZeroPublicationErrorKindV1,
    final_state: CanonicalNativeGenerationZeroPublicationFinalStateV1,
    temporary_state: CanonicalNativeGenerationZeroPublicationTemporaryStateV1,
) -> CanonicalNativeGenerationZeroPublicationErrorV1 {
    CanonicalNativeGenerationZeroPublicationErrorV1 {
        kind,
        final_state,
        temporary_state,
    }
}

fn map_final_state_v1(
    state: CanonicalArtifactFinalStateV1,
) -> CanonicalNativeGenerationZeroPublicationFinalStateV1 {
    match state {
        CanonicalArtifactFinalStateV1::NotInstalled => {
            CanonicalNativeGenerationZeroPublicationFinalStateV1::NotInstalled
        }
        CanonicalArtifactFinalStateV1::InstalledSyncPending => {
            CanonicalNativeGenerationZeroPublicationFinalStateV1::InstalledSyncPending
        }
        CanonicalArtifactFinalStateV1::InstalledDurable => {
            CanonicalNativeGenerationZeroPublicationFinalStateV1::InstalledDurable
        }
        CanonicalArtifactFinalStateV1::ExistingIdentical => {
            CanonicalNativeGenerationZeroPublicationFinalStateV1::ExistingIdentical
        }
        CanonicalArtifactFinalStateV1::ExistingDifferent => {
            CanonicalNativeGenerationZeroPublicationFinalStateV1::ExistingDifferent
        }
    }
}

fn map_temporary_state_v1(
    state: CanonicalArtifactTemporaryStateV1,
) -> CanonicalNativeGenerationZeroPublicationTemporaryStateV1 {
    match state {
        CanonicalArtifactTemporaryStateV1::NotCreated => {
            CanonicalNativeGenerationZeroPublicationTemporaryStateV1::NotCreated
        }
        CanonicalArtifactTemporaryStateV1::Present => {
            CanonicalNativeGenerationZeroPublicationTemporaryStateV1::Present
        }
        CanonicalArtifactTemporaryStateV1::RemovedSyncPending => {
            CanonicalNativeGenerationZeroPublicationTemporaryStateV1::RemovedSyncPending
        }
        CanonicalArtifactTemporaryStateV1::RemovedDurable => {
            CanonicalNativeGenerationZeroPublicationTemporaryStateV1::RemovedDurable
        }
    }
}

fn map_low_level_error_kind_v1(
    kind: &CanonicalArtifactPublishErrorKindV1,
) -> CanonicalNativeGenerationZeroPublicationErrorKindV1 {
    match kind {
        CanonicalArtifactPublishErrorKindV1::InvalidRelativePath => {
            CanonicalNativeGenerationZeroPublicationErrorKindV1::InvalidRelativePath
        }
        CanonicalArtifactPublishErrorKindV1::SecureResolutionUnavailable(reason) => {
            CanonicalNativeGenerationZeroPublicationErrorKindV1::SecureResolutionUnavailable(
                reason.clone(),
            )
        }
        CanonicalArtifactPublishErrorKindV1::UnsafeLink => {
            CanonicalNativeGenerationZeroPublicationErrorKindV1::UnsafeLink
        }
        CanonicalArtifactPublishErrorKindV1::EscapeOrMount => {
            CanonicalNativeGenerationZeroPublicationErrorKindV1::EscapeOrMount
        }
        CanonicalArtifactPublishErrorKindV1::RaceDetected => {
            CanonicalNativeGenerationZeroPublicationErrorKindV1::RaceDetected
        }
        CanonicalArtifactPublishErrorKindV1::NonRegularArtifact => {
            CanonicalNativeGenerationZeroPublicationErrorKindV1::NonRegularArtifact
        }
        CanonicalArtifactPublishErrorKindV1::ArtifactTooLarge { maximum, attempted } => {
            CanonicalNativeGenerationZeroPublicationErrorKindV1::ArtifactTooLarge {
                maximum: *maximum,
                attempted: *attempted,
            }
        }
        CanonicalArtifactPublishErrorKindV1::PreInstallRejected(
            CanonicalArtifactPublishGateRejectionV1::Cancelled,
        ) => CanonicalNativeGenerationZeroPublicationErrorKindV1::PreInstallRejected(
            CanonicalNativeGenerationZeroPublicationGateRejectionV1::Cancelled,
        ),
        CanonicalArtifactPublishErrorKindV1::ExistingContentMismatch => {
            CanonicalNativeGenerationZeroPublicationErrorKindV1::ExistingContentMismatch
        }
        CanonicalArtifactPublishErrorKindV1::Io { operation, message } => {
            CanonicalNativeGenerationZeroPublicationErrorKindV1::Io {
                operation: *operation,
                message: message.clone(),
            }
        }
    }
}
