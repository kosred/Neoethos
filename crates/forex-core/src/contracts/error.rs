use std::fmt;

use super::{ArtifactKind, BackendKind, RuntimeMode};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactContractError {
    MissingProvenanceField(&'static str),
    UnsupportedSchemaVersion {
        actual: u16,
        expected: u16,
    },
    BackendAssignmentMismatch {
        backend_kind: BackendKind,
        assignment_backend: BackendKind,
    },
    CanonicalArtifactHasDegradedReason,
    MissingDegradedReason,
    WrongArtifactKind {
        actual: ArtifactKind,
        expected: ArtifactKind,
    },
    LiveRejectedArtifactKind(ArtifactKind),
    LiveRejectedRuntimeMode {
        mode: RuntimeMode,
        backend: BackendKind,
    },
    LiveRejectedMismatch {
        field: &'static str,
        actual: String,
        expected: String,
    },
    LiveRejectedStaleArtifact {
        age_seconds: i64,
        max_age_seconds: i64,
    },
    TemporalPolicyViolation {
        field: &'static str,
        reason: String,
    },
    TemporalPolicyMismatch {
        field: &'static str,
        actual: String,
        expected: String,
    },
}

impl fmt::Display for ArtifactContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingProvenanceField(field) => {
                write!(f, "artifact provenance is missing required field `{field}`")
            }
            Self::UnsupportedSchemaVersion { actual, expected } => write!(
                f,
                "artifact schema version {actual} is unsupported; expected {expected}"
            ),
            Self::BackendAssignmentMismatch {
                backend_kind,
                assignment_backend,
            } => write!(
                f,
                "artifact backend kind {backend_kind:?} does not match device assignment backend {assignment_backend:?}"
            ),
            Self::CanonicalArtifactHasDegradedReason => {
                write!(
                    f,
                    "canonical artifact cannot carry a degraded runtime reason"
                )
            }
            Self::MissingDegradedReason => {
                write!(
                    f,
                    "non-canonical artifact must record a degraded runtime reason"
                )
            }
            Self::WrongArtifactKind { actual, expected } => {
                write!(f, "wrong artifact kind {actual:?}; expected {expected:?}")
            }
            Self::LiveRejectedArtifactKind(kind) => {
                write!(
                    f,
                    "artifact kind {kind:?} is not eligible for live execution"
                )
            }
            Self::LiveRejectedRuntimeMode { mode, backend } => write!(
                f,
                "runtime mode {mode:?} with backend {backend:?} is not live-safe"
            ),
            Self::LiveRejectedMismatch {
                field,
                actual,
                expected,
            } => write!(
                f,
                "live contract mismatch for {field}: actual `{actual}` expected `{expected}`"
            ),
            Self::LiveRejectedStaleArtifact {
                age_seconds,
                max_age_seconds,
            } => write!(
                f,
                "live artifact is stale: age {age_seconds}s exceeds max {max_age_seconds}s"
            ),
            Self::TemporalPolicyViolation { field, reason } => {
                write!(
                    f,
                    "temporal feature contract violation for {field}: {reason}"
                )
            }
            Self::TemporalPolicyMismatch {
                field,
                actual,
                expected,
            } => write!(
                f,
                "temporal feature contract mismatch for {field}: actual `{actual}` expected `{expected}`"
            ),
        }
    }
}

impl std::error::Error for ArtifactContractError {}
