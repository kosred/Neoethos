//! Metadata-only discovery for explicit user imports.
//!
//! Source-format vocabulary belongs here and in `import_service`; runtime
//! discovery never interprets CSV/JSON/Parquet/Arrow paths as runnable data.

use crate::core::import_provenance::ImportSourceFormat;
use anyhow::Result;
use neoethos_dataset_contracts::CanonicalTimeframe;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

pub const MAX_IMPORT_WALK_DEPTH: usize = 4;
pub const MAX_IMPORT_SOURCE_BYTES: u64 = 64 * 1024 * 1024 * 1024;

const SILENT_EXTENSIONS: &[&str] = &[
    "md",
    "txt",
    "rst",
    "log",
    "yml",
    "yaml",
    "toml",
    "lock",
    "gitignore",
    "gitattributes",
    "sha256",
    "sha1",
    "asc",
    "zip",
    "gz",
    "tar",
    "bz2",
    "xz",
    "7z",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportSourceEntry {
    pub path: PathBuf,
    pub symbol: Option<String>,
    pub timeframe: Option<String>,
    pub format: ImportSourceFormat,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImportSkipReason {
    UnknownExtension(String),
    UnsupportedTimeframe(String),
    TooLarge(u64),
    Unreadable(String),
}

impl ImportSkipReason {
    pub const fn category(&self) -> &'static str {
        match self {
            Self::UnknownExtension(_) => "unknown_extension",
            Self::UnsupportedTimeframe(_) => "unsupported_timeframe",
            Self::TooLarge(_) => "too_large",
            Self::Unreadable(_) => "unreadable",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkippedImportSource {
    pub path: PathBuf,
    pub reason: ImportSkipReason,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportDiscovery {
    pub root: PathBuf,
    pub entries: Vec<ImportSourceEntry>,
    pub skipped: Vec<SkippedImportSource>,
}

impl ImportDiscovery {
    pub fn scan(root: impl AsRef<Path>) -> Result<Self> {
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

        for walked in WalkDir::new(&root)
            .max_depth(MAX_IMPORT_WALK_DEPTH)
            .follow_links(false)
            .into_iter()
        {
            let walked = match walked {
                Ok(walked) => walked,
                Err(error) => {
                    skipped.push(SkippedImportSource {
                        path: error
                            .path()
                            .map(Path::to_path_buf)
                            .unwrap_or_else(|| root.clone()),
                        reason: ImportSkipReason::Unreadable(error.to_string()),
                    });
                    continue;
                }
            };
            if !walked.file_type().is_file() {
                continue;
            }
            let path = walked.path().to_path_buf();
            let extension = path
                .extension()
                .and_then(|value| value.to_str())
                .map(str::to_ascii_lowercase)
                .unwrap_or_default();
            if SILENT_EXTENSIONS.contains(&extension.as_str()) {
                continue;
            }
            let size_bytes = match walked.metadata() {
                Ok(metadata) => metadata.len(),
                Err(error) => {
                    skipped.push(SkippedImportSource {
                        path,
                        reason: ImportSkipReason::Unreadable(error.to_string()),
                    });
                    continue;
                }
            };
            if size_bytes > MAX_IMPORT_SOURCE_BYTES {
                skipped.push(SkippedImportSource {
                    path,
                    reason: ImportSkipReason::TooLarge(size_bytes),
                });
                continue;
            }
            let Some(format) = ImportSourceFormat::from_extension(&extension) else {
                skipped.push(SkippedImportSource {
                    path,
                    reason: ImportSkipReason::UnknownExtension(extension),
                });
                continue;
            };
            let (symbol, timeframe) = infer_symbol_timeframe(&path, &root);
            if let Some(label) = timeframe.as_deref()
                && label.parse::<CanonicalTimeframe>().is_err()
            {
                skipped.push(SkippedImportSource {
                    path,
                    reason: ImportSkipReason::UnsupportedTimeframe(label.to_owned()),
                });
                continue;
            }
            entries.push(ImportSourceEntry {
                path,
                symbol,
                timeframe,
                format,
                size_bytes,
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

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
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
        values.sort_by_key(|label| {
            label
                .parse::<CanonicalTimeframe>()
                .map(CanonicalTimeframe::ctrader_protocol_code)
                .unwrap_or(i32::MAX)
        });
        values.dedup();
        values
    }

    pub fn format_counts(&self) -> Vec<(ImportSourceFormat, usize)> {
        let mut counts = BTreeMap::new();
        for entry in &self.entries {
            *counts.entry(entry.format).or_insert(0) += 1;
        }
        counts.into_iter().collect()
    }

    pub fn skip_counts_by_category(&self) -> Vec<(String, usize)> {
        let mut counts = BTreeMap::new();
        for entry in &self.skipped {
            *counts
                .entry(entry.reason.category().to_owned())
                .or_insert(0) += 1;
        }
        counts.into_iter().collect()
    }
}

fn infer_symbol_timeframe(path: &Path, root: &Path) -> (Option<String>, Option<String>) {
    let relative = path.strip_prefix(root).unwrap_or(path);
    let components = relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let mut symbol = components.iter().find_map(|component| {
        component
            .strip_prefix("symbol=")
            .map(|value| value.trim().to_ascii_uppercase())
            .filter(|value| !value.is_empty())
    });
    let mut timeframe = components.iter().find_map(|component| {
        component
            .strip_prefix("timeframe=")
            .map(|value| value.trim().to_ascii_uppercase())
            .filter(|value| !value.is_empty())
    });

    if let Some(stem) = path.file_stem().and_then(|value| value.to_str()) {
        let tokens = stem
            .split(['_', '-', '.'])
            .filter(|token| !token.is_empty())
            .map(str::to_ascii_uppercase)
            .collect::<Vec<_>>();
        if timeframe.is_none() {
            timeframe = tokens
                .iter()
                .find(|token| token.parse::<CanonicalTimeframe>().is_ok())
                .cloned();
        }
        if symbol.is_none() {
            let timeframe_index = tokens
                .iter()
                .position(|token| token.parse::<CanonicalTimeframe>().is_ok());
            symbol = timeframe_index
                .and_then(|index| index.checked_sub(1))
                .and_then(|index| tokens.get(index).cloned())
                .or_else(|| tokens.first().cloned())
                .filter(|value| value.parse::<CanonicalTimeframe>().is_err());
        }
    }

    if timeframe.is_none() || symbol.is_none() {
        for window in components.windows(2) {
            let first = window[0].trim().to_ascii_uppercase();
            let second = window[1].trim().to_ascii_uppercase();
            if second.parse::<CanonicalTimeframe>().is_ok() {
                symbol.get_or_insert(first);
                timeframe.get_or_insert(second);
            }
        }
    }
    (symbol, timeframe)
}
