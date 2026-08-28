//! Runtime discovery for verified canonical Vortex generations only.
//!
//! Explicit user-source discovery lives in `import_discover`. A CSV, JSON,
//! Parquet, Arrow, loose Vortex file, or retired `symbol=/timeframe=` tree can
//! appear in this report only as rejected metadata; it is never runnable.

use crate::core::dataset_manifest::{read_current_manifest, read_current_manifest_metadata};
use anyhow::Result;
use neoethos_dataset_contracts::CanonicalDatasetIdentity;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const MAX_WALK_DEPTH: usize = 1;
pub const MAX_FILE_SIZE_BYTES: u64 = 64 * 1024 * 1024 * 1024;

/// Runtime storage has exactly one format. Source-format vocabulary belongs to
/// the explicit import boundary and cannot leak into discovery/load decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataFormat {
    Vortex,
}

impl DataFormat {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Vortex => "Vortex",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFileEntry {
    pub path: PathBuf,
    /// Opaque, reversible identity used by exact load/update requests.
    pub dataset_identity: String,
    /// Current content-addressed generation used as the next publication's
    /// compare-and-swap base.
    pub generation: String,
    pub manifest_binding_sha256: String,
    pub symbol: Option<String>,
    pub timeframe: Option<String>,
    pub format: DataFormat,
    pub size_bytes: u64,
    pub verification: DataVerificationStatus,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DataVerificationStatus {
    /// Manifest/path/provenance metadata was validated, but the generation
    /// bytes were deliberately not hashed or decoded.
    ManifestOnly,
    /// The generation hash, row count, and timestamp endpoints were checked.
    GenerationVerified,
}

impl DataVerificationStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ManifestOnly => "manifest_only",
            Self::GenerationVerified => "generation_verified",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SkipReason {
    ImportRequired(String),
    RetiredLayout(String),
    InvalidCanonicalIdentity(String),
    UnverifiedGeneration(String),
    Unreadable(String),
}

impl SkipReason {
    pub const fn category(&self) -> &'static str {
        match self {
            Self::ImportRequired(_) => "import_required",
            Self::RetiredLayout(_) => "retired_layout",
            Self::InvalidCanonicalIdentity(_) => "invalid_canonical_identity",
            Self::UnverifiedGeneration(_) => "unverified_generation",
            Self::Unreadable(_) => "unreadable",
        }
    }

    pub fn detail(&self) -> &str {
        match self {
            Self::ImportRequired(detail)
            | Self::RetiredLayout(detail)
            | Self::InvalidCanonicalIdentity(detail)
            | Self::UnverifiedGeneration(detail)
            | Self::Unreadable(detail) => detail,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkippedFile {
    pub path: PathBuf,
    pub reason: SkipReason,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetDiscovery {
    pub root: PathBuf,
    pub entries: Vec<DataFileEntry>,
    pub skipped: Vec<SkippedFile>,
}

impl DatasetDiscovery {
    pub fn scan(root: impl AsRef<Path>) -> Result<Self> {
        Self::scan_with_verification(root, true)
    }

    /// Bounded metadata inventory for frequently refreshed status/UI views.
    /// Entries are explicitly labelled `manifest_only`; this never authorizes
    /// them for runtime use.
    pub fn scan_metadata(root: impl AsRef<Path>) -> Result<Self> {
        Self::scan_with_verification(root, false)
    }

    fn scan_with_verification(root: impl AsRef<Path>, verify_data: bool) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let mut entries = Vec::new();
        let mut skipped = Vec::new();
        if !root.exists() {
            return Ok(Self {
                root,
                entries,
                skipped,
            });
        }
        for directory in std::fs::read_dir(&root)? {
            let directory = match directory {
                Ok(directory) => directory,
                Err(error) => {
                    skipped.push(SkippedFile {
                        path: root.clone(),
                        reason: SkipReason::Unreadable(error.to_string()),
                    });
                    continue;
                }
            };
            let file_type = match directory.file_type() {
                Ok(file_type) => file_type,
                Err(error) => {
                    skipped.push(SkippedFile {
                        path: directory.path(),
                        reason: SkipReason::Unreadable(error.to_string()),
                    });
                    continue;
                }
            };
            if !file_type.is_dir() {
                skipped.push(SkippedFile {
                    path: directory.path(),
                    reason: SkipReason::ImportRequired(
                        "loose source files require the explicit import command before runtime use"
                            .to_owned(),
                    ),
                });
                continue;
            }
            let name = directory.file_name().to_string_lossy().into_owned();
            if name.starts_with("symbol=") {
                skipped.push(SkippedFile {
                    path: directory.path(),
                    reason: SkipReason::RetiredLayout(
                        "symbol=/timeframe= requires explicit offline migration".to_owned(),
                    ),
                });
                continue;
            }
            if !name.starts_with("d1-") {
                continue;
            }
            let identity = match CanonicalDatasetIdentity::from_path_component(&name) {
                Ok(identity) => identity,
                Err(error) => {
                    skipped.push(SkippedFile {
                        path: directory.path(),
                        reason: SkipReason::InvalidCanonicalIdentity(error.to_string()),
                    });
                    continue;
                }
            };
            let manifest_result = if verify_data {
                read_current_manifest(&root, &identity)
            } else {
                read_current_manifest_metadata(&root, &identity)
            };
            let manifest = match manifest_result {
                Ok(manifest) => manifest,
                Err(error) => {
                    skipped.push(SkippedFile {
                        path: directory.path(),
                        reason: SkipReason::UnverifiedGeneration(format!("{error:#}")),
                    });
                    continue;
                }
            };
            let path = manifest.generation_path();
            let size_bytes = match std::fs::metadata(&path) {
                Ok(metadata) => metadata.len(),
                Err(error) => {
                    skipped.push(SkippedFile {
                        path,
                        reason: SkipReason::Unreadable(error.to_string()),
                    });
                    continue;
                }
            };
            entries.push(DataFileEntry {
                path,
                dataset_identity: identity.to_path_component(),
                generation: manifest.generation_id().to_owned(),
                manifest_binding_sha256: manifest.manifest_binding_sha256().to_owned(),
                symbol: Some(identity.symbol_name().to_owned()),
                timeframe: Some(identity.timeframe().as_str().to_owned()),
                format: DataFormat::Vortex,
                size_bytes,
                verification: if verify_data {
                    DataVerificationStatus::GenerationVerified
                } else {
                    DataVerificationStatus::ManifestOnly
                },
            });
        }
        entries.sort_by(|left, right| {
            left.symbol
                .cmp(&right.symbol)
                .then_with(|| left.timeframe.cmp(&right.timeframe))
                .then_with(|| left.path.cmp(&right.path))
        });
        skipped.sort_by(|left, right| left.path.cmp(&right.path));
        Ok(Self {
            root,
            entries,
            skipped,
        })
    }

    pub fn symbols(&self) -> Vec<String> {
        let mut values = self
            .entries
            .iter()
            .filter_map(|entry| entry.symbol.clone())
            .collect::<Vec<_>>();
        values.sort();
        values.dedup();
        values
    }

    pub fn timeframes(&self) -> Vec<String> {
        let mut values = self
            .entries
            .iter()
            .filter_map(|entry| entry.timeframe.clone())
            .collect::<Vec<_>>();
        values.sort();
        values.dedup();
        values
    }

    pub fn format_counts(&self) -> Vec<(DataFormat, usize)> {
        if self.entries.is_empty() {
            Vec::new()
        } else {
            vec![(DataFormat::Vortex, self.entries.len())]
        }
    }

    pub fn skip_counts_by_category(&self) -> Vec<(String, usize)> {
        let mut counts = std::collections::BTreeMap::new();
        for skipped in &self.skipped {
            *counts
                .entry(skipped.reason.category().to_owned())
                .or_insert(0) += 1;
        }
        counts.into_iter().collect()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
