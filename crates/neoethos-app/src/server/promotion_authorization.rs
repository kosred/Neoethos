//! Model-free gate between discovery evidence and filesystem model copying.

use std::path::{Path, PathBuf};

pub(crate) const REQUIRED_COMPOSITE_PROMOTION_AUTHORITY_KIND_V3: &str =
    "neoethos.search-promotion-summary.v3";

#[derive(Debug)]
pub(crate) enum PromotionAuthorizationError {
    UnsafePathLeaf { field: &'static str, value: String },
    MissingModelTargets { path: PathBuf },
    UnsupportedSchema { found: Option<u64> },
    MalformedModelTargets { reason: String },
    RequestedIdentityMismatch { reason: String },
    ReceiptDigestMismatch { expected: String, found: String },
    MissingPromotionSummary { path: PathBuf },
    InvalidPromotionSummary { reason: String },
    PromotionSummaryMismatch,
    UnsupportedEvidenceSchema { found: String },
    InvalidCompositeEvidenceScope,
    MissingHeldOutEvidence { kind: &'static str },
    FailedHeldOutEvidence { kind: &'static str },
    MissingTrainedArtifacts { path: PathBuf },
    CopyFailed { reason: String },
}

impl std::fmt::Display for PromotionAuthorizationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsafePathLeaf { field, value } => {
                write!(
                    formatter,
                    "promotion {field} `{value}` is not one safe path leaf"
                )
            }
            Self::MissingModelTargets { path } => {
                write!(
                    formatter,
                    "model_targets v3 is missing at {}",
                    path.display()
                )
            }
            Self::UnsupportedSchema { found } => write!(
                formatter,
                "model_targets schema {found:?} cannot authorize promotion"
            ),
            Self::MalformedModelTargets { reason } => {
                write!(formatter, "model_targets v3 is malformed: {reason}")
            }
            Self::RequestedIdentityMismatch { reason } => {
                write!(formatter, "model_targets identity mismatch: {reason}")
            }
            Self::ReceiptDigestMismatch { expected, found } => write!(
                formatter,
                "model_targets receipt digest {found} does not match recomputed {expected}"
            ),
            Self::MissingPromotionSummary { path } => write!(
                formatter,
                "canonical promotion summary is missing at {}",
                path.display()
            ),
            Self::InvalidPromotionSummary { reason } => {
                write!(
                    formatter,
                    "canonical promotion summary is invalid: {reason}"
                )
            }
            Self::PromotionSummaryMismatch => write!(
                formatter,
                "canonical promotion summary differs from the v3 target binding"
            ),
            Self::UnsupportedEvidenceSchema { found } => write!(
                formatter,
                "promotion evidence `{found}` is diagnostic-only; exact composite v3 is required"
            ),
            Self::InvalidCompositeEvidenceScope => write!(
                formatter,
                "promotion evidence lacks an exact composite in-sample/holdout/forward/prop scope"
            ),
            Self::MissingHeldOutEvidence { kind } => {
                write!(
                    formatter,
                    "required exact promotion evidence `{kind}` is missing"
                )
            }
            Self::FailedHeldOutEvidence { kind } => {
                write!(
                    formatter,
                    "required exact promotion evidence `{kind}` failed"
                )
            }
            Self::MissingTrainedArtifacts { path } => write!(
                formatter,
                "trained model artifacts are missing at {}",
                path.display()
            ),
            Self::CopyFailed { reason } => {
                write!(formatter, "authorized model copy failed: {reason}")
            }
        }
    }
}

impl std::error::Error for PromotionAuthorizationError {}

#[derive(Debug, Clone)]
pub(crate) struct ValidatedPromotionPath {
    symbol: String,
    base_tf: String,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CompositeAuthorityChecksV3 {
    pub(crate) exact_receipt_config_sidecar: bool,
    pub(crate) exact_composite_scope: bool,
    pub(crate) required_evidence_complete: bool,
    pub(crate) required_evidence_passed: bool,
}

#[derive(Debug)]
pub(crate) struct PromotionCopyPermit {
    path: ValidatedPromotionPath,
}

#[derive(Debug)]
pub(crate) struct AuthorizedModelCopy {
    pub(crate) destination: PathBuf,
    pub(crate) files_copied: usize,
}

pub(crate) fn validate_promotion_path_leafs(
    symbol: &str,
    base_tf: &str,
) -> Result<ValidatedPromotionPath, PromotionAuthorizationError> {
    validate_path_leaf("symbol", symbol)?;
    validate_path_leaf("base timeframe", base_tf)?;
    const DIRECT_CANONICAL_TIMEFRAMES: &[&str] = &[
        "M1", "M2", "M3", "M4", "M5", "M10", "M15", "M30", "H1", "H4", "H12", "D1", "W1", "MN1",
    ];
    if !DIRECT_CANONICAL_TIMEFRAMES.contains(&base_tf) {
        return Err(PromotionAuthorizationError::UnsafePathLeaf {
            field: "base timeframe",
            value: base_tf.to_owned(),
        });
    }
    Ok(ValidatedPromotionPath {
        symbol: symbol.to_owned(),
        base_tf: base_tf.to_owned(),
    })
}

fn validate_path_leaf(field: &'static str, value: &str) -> Result<(), PromotionAuthorizationError> {
    let path = Path::new(value);
    let mut components = path.components();
    let is_one_normal_component = matches!(
        (components.next(), components.next()),
        (Some(std::path::Component::Normal(component)), None) if component == value
    );
    let contains_portable_forbidden_character = value.chars().any(|character| {
        character.is_control()
            || matches!(
                character,
                '/' | '\\' | ':' | '<' | '>' | '"' | '|' | '?' | '*'
            )
    });
    let windows_stem = value
        .split('.')
        .next()
        .unwrap_or_default()
        .to_ascii_uppercase();
    let is_windows_device_name = matches!(
        windows_stem.as_str(),
        "CON"
            | "PRN"
            | "AUX"
            | "NUL"
            | "COM1"
            | "COM2"
            | "COM3"
            | "COM4"
            | "COM5"
            | "COM6"
            | "COM7"
            | "COM8"
            | "COM9"
            | "LPT1"
            | "LPT2"
            | "LPT3"
            | "LPT4"
            | "LPT5"
            | "LPT6"
            | "LPT7"
            | "LPT8"
            | "LPT9"
    );
    if value.is_empty()
        || value.trim() != value
        || matches!(value, "." | "..")
        || value.ends_with('.')
        || contains_portable_forbidden_character
        || is_windows_device_name
        || path.is_absolute()
        || !is_one_normal_component
    {
        return Err(PromotionAuthorizationError::UnsafePathLeaf {
            field,
            value: value.to_owned(),
        });
    }
    Ok(())
}

pub(crate) fn authorize_exact_composite_promotion_v3(
    path: ValidatedPromotionPath,
    artifact_kind: &str,
    checks: CompositeAuthorityChecksV3,
) -> Result<PromotionCopyPermit, PromotionAuthorizationError> {
    if artifact_kind != REQUIRED_COMPOSITE_PROMOTION_AUTHORITY_KIND_V3 {
        return Err(PromotionAuthorizationError::UnsupportedEvidenceSchema {
            found: artifact_kind.to_owned(),
        });
    }
    if !checks.exact_receipt_config_sidecar {
        return Err(PromotionAuthorizationError::PromotionSummaryMismatch);
    }
    if !checks.exact_composite_scope {
        return Err(PromotionAuthorizationError::InvalidCompositeEvidenceScope);
    }
    if !checks.required_evidence_complete {
        return Err(PromotionAuthorizationError::MissingHeldOutEvidence {
            kind: "composite_v3",
        });
    }
    if !checks.required_evidence_passed {
        return Err(PromotionAuthorizationError::FailedHeldOutEvidence {
            kind: "composite_v3",
        });
    }
    Ok(PromotionCopyPermit { path })
}

pub(crate) fn copy_model_tree_if_authorized(
    authorization: Result<PromotionCopyPermit, PromotionAuthorizationError>,
    models_root: &Path,
    live_models_root: &Path,
) -> Result<AuthorizedModelCopy, PromotionAuthorizationError> {
    let permit = authorization?;
    let source = models_root
        .join(&permit.path.symbol)
        .join(&permit.path.base_tf);
    if !source.is_dir() {
        return Err(PromotionAuthorizationError::MissingTrainedArtifacts { path: source });
    }
    let destination = live_models_root
        .join(&permit.path.symbol)
        .join(&permit.path.base_tf);
    let files_copied = copy_dir_recursive(&source, &destination).map_err(|error| {
        PromotionAuthorizationError::CopyFailed {
            reason: error.to_string(),
        }
    })?;
    Ok(AuthorizedModelCopy {
        destination,
        files_copied,
    })
}

fn copy_dir_recursive(source: &Path, destination: &Path) -> std::io::Result<usize> {
    std::fs::create_dir_all(destination)?;
    let mut copied = 0;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let from = entry.path();
        let to = destination.join(entry.file_name());
        if from.is_dir() {
            copied += copy_dir_recursive(&from, &to)?;
        } else {
            std::fs::copy(from, to)?;
            copied += 1;
        }
    }
    Ok(copied)
}
