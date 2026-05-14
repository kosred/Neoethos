//! Filesystem-level dataset discovery.
//!
//! Operator concern (2026-05-14, verbatim Greek):
//!   "Ένα θέμα του UI και του cli είναι ότι δεν βοηθά τον χρήστη να
//!    πλοηγηθεί στα αρχεία και να δώσει πιθανό φάκελο που υπάρχουν
//!    υποφακέλους με τα δεδομένα."
//! Translation: UI and CLI don't help the user navigate the filesystem
//! or pick a folder that contains subfolders with data.
//!
//! This module gives both the egui UI and the CLI a single entry-point
//! that takes a `&Path` to a folder and walks it recursively (capped at
//! [`MAX_WALK_DEPTH`]), classifying every file it finds:
//!
//! - by extension (`.vortex`, `.parquet`, `.feather`/`.arrow`,
//!   `.csv`, `.tsv`, `.json`/`.jsonl`) — see [`DataFormat`],
//! - by symbol + timeframe parsed from the filename / parent folders
//!   (supports `EURUSD_M5.csv`, `EURUSD/M5.csv`, `EURUSD/M5/2024.parquet`,
//!   Hive-style `symbol=EURUSD/timeframe=H1/*.parquet`),
//! - rejecting any timeframe label NOT in
//!   [`forex_core::CANONICAL_TIMEFRAMES`] (e.g. `EURUSD_H2.csv` lands in
//!   `skipped` with [`SkipReason::UnsupportedTimeframe`] instead of being
//!   silently loaded).
//!
//! The returned [`DatasetDiscovery`] is a *report only* — this module
//! never touches Vortex conversion. That belongs to
//! `universal_importer.rs` (a parallel agent owns that work).
//!
//! ## Constants
//!
//! All knobs that govern the walk live at the top of this file so they
//! are reviewable in one place — no magic numbers buried in loops.

use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

/// Maximum recursion depth for [`DatasetDiscovery::scan`].
///
/// A typical layout is `<root>/symbol=EURUSD/timeframe=M5/year=2024/*.parquet`
/// which is 4 components deep, so cap at 4 to keep the walk cheap on
/// large mounted drives while still catching the common patterns.
pub const MAX_WALK_DEPTH: usize = 4;

/// Files larger than this are classified as
/// [`SkipReason::TooLarge`] without being read. The discovery report is
/// metadata-only, so anything bigger than this is almost certainly a
/// bulk archive that should be staged separately.
///
/// 4 GiB — large enough to admit a full year of M1 Vortex per symbol,
/// small enough to exclude raw broker dumps the user did not intend to
/// import.
pub const MAX_FILE_SIZE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Extensions we silently ignore (docs, archives, git artifacts).
/// These don't appear in `skipped` because they're not data candidates
/// in the first place — including them would spam the UI summary.
const SILENTLY_IGNORED_EXTENSIONS: &[&str] = &[
    "md", "txt", "rst", "log", "yml", "yaml", "toml", "lock",
    "gitignore", "gitattributes", "sha256", "sha1", "asc",
    "zip", "gz", "tar", "bz2", "xz", "7z",
];

/// Classification of a single discovered data file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataFormat {
    Vortex,
    Parquet,
    Arrow,
    Csv,
    Tsv,
    Json,
    JsonLines,
}

impl DataFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            DataFormat::Vortex => "Vortex",
            DataFormat::Parquet => "Parquet",
            DataFormat::Arrow => "Arrow",
            DataFormat::Csv => "Csv",
            DataFormat::Tsv => "Tsv",
            DataFormat::Json => "Json",
            DataFormat::JsonLines => "JsonLines",
        }
    }

    /// Classify by extension. Returns `None` for non-data files.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "vortex" | "vtx" => Some(DataFormat::Vortex),
            "parquet" | "pq" => Some(DataFormat::Parquet),
            "feather" | "arrow" | "ipc" => Some(DataFormat::Arrow),
            "csv" => Some(DataFormat::Csv),
            "tsv" | "tab" => Some(DataFormat::Tsv),
            "json" => Some(DataFormat::Json),
            "jsonl" | "ndjson" => Some(DataFormat::JsonLines),
            _ => None,
        }
    }
}

/// A single file that survived classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFileEntry {
    pub path: PathBuf,
    pub symbol: Option<String>,
    pub timeframe: Option<String>,
    pub format: DataFormat,
    pub size_bytes: u64,
}

/// Why a file was excluded from the entry list.
///
/// The `UnsupportedTimeframe(label)` variant carries the offending
/// label (e.g. `"H2"`) so the UI/CLI can surface it instead of the
/// vague "unrecognised file" message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SkipReason {
    UnknownExtension(String),
    UnsupportedTimeframe(String),
    TooLarge(u64),
    Unreadable(String),
}

impl SkipReason {
    pub fn category(&self) -> &'static str {
        match self {
            SkipReason::UnknownExtension(_) => "unknown_extension",
            SkipReason::UnsupportedTimeframe(_) => "unsupported_timeframe",
            SkipReason::TooLarge(_) => "too_large",
            SkipReason::Unreadable(_) => "unreadable",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkippedFile {
    pub path: PathBuf,
    pub reason: SkipReason,
}

/// Aggregate report of [`DatasetDiscovery::scan`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetDiscovery {
    pub root: PathBuf,
    pub entries: Vec<DataFileEntry>,
    pub skipped: Vec<SkippedFile>,
}

impl DatasetDiscovery {
    /// Walk `root` to depth [`MAX_WALK_DEPTH`] and classify every regular
    /// file. Returns an empty (but successful) report if `root` does
    /// not exist or has no readable entries — the caller is expected
    /// to surface "empty dataset" explicitly rather than silently
    /// falling back to anything else.
    pub fn scan(root: impl AsRef<Path>) -> Result<Self> {
        let root_buf = root.as_ref().to_path_buf();
        let mut entries: Vec<DataFileEntry> = Vec::new();
        let mut skipped: Vec<SkippedFile> = Vec::new();

        if !root_buf.exists() {
            return Ok(Self {
                root: root_buf,
                entries,
                skipped,
            });
        }

        for dir_entry in WalkDir::new(&root_buf)
            .max_depth(MAX_WALK_DEPTH)
            .follow_links(false)
            .into_iter()
        {
            let dir_entry = match dir_entry {
                Ok(d) => d,
                Err(err) => {
                    // walkdir gives us the offending path even on error.
                    let path = err
                        .path()
                        .map(|p| p.to_path_buf())
                        .unwrap_or_else(|| root_buf.clone());
                    skipped.push(SkippedFile {
                        path,
                        reason: SkipReason::Unreadable(err.to_string()),
                    });
                    continue;
                }
            };

            if !dir_entry.file_type().is_file() {
                continue;
            }

            let path = dir_entry.path().to_path_buf();
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_default();

            // Silently ignore docs/archives/git artifacts — they are not
            // candidates for data and would spam the skipped list.
            if SILENTLY_IGNORED_EXTENSIONS
                .iter()
                .any(|e| *e == ext.as_str())
            {
                continue;
            }

            // Size check before format check — a 10 GiB file is "too
            // large" regardless of whether its extension is one we
            // recognise.
            let size_bytes = dir_entry
                .metadata()
                .map(|m| m.len())
                .unwrap_or(0);
            if size_bytes > MAX_FILE_SIZE_BYTES {
                skipped.push(SkippedFile {
                    path,
                    reason: SkipReason::TooLarge(size_bytes),
                });
                continue;
            }

            let format = match DataFormat::from_extension(&ext) {
                Some(fmt) => fmt,
                None => {
                    skipped.push(SkippedFile {
                        path,
                        reason: SkipReason::UnknownExtension(ext),
                    });
                    continue;
                }
            };

            let (symbol, timeframe_raw) =
                infer_symbol_and_timeframe(&path, &root_buf);

            // Reject filenames that imply a NON-canonical timeframe
            // (e.g. H2) so they don't sneak into training. The
            // canonical list lives in forex-core and is the single
            // source of truth.
            if let Some(tf_label) = timeframe_raw.as_ref() {
                if !forex_core::is_canonical_timeframe(tf_label) {
                    skipped.push(SkippedFile {
                        path,
                        reason: SkipReason::UnsupportedTimeframe(
                            tf_label.clone(),
                        ),
                    });
                    continue;
                }
            }

            entries.push(DataFileEntry {
                path,
                symbol,
                timeframe: timeframe_raw,
                format,
                size_bytes,
            });
        }

        // Deterministic output: sort entries by (symbol, timeframe, path)
        // so two scans of the same tree produce identical summaries.
        entries.sort_by(|a, b| {
            a.symbol
                .cmp(&b.symbol)
                .then_with(|| a.timeframe.cmp(&b.timeframe))
                .then_with(|| a.path.cmp(&b.path))
        });
        skipped.sort_by(|a, b| a.path.cmp(&b.path));

        Ok(Self {
            root: root_buf,
            entries,
            skipped,
        })
    }

    /// Distinct symbols (sorted, uppercase) found in `entries`.
    pub fn symbols(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .entries
            .iter()
            .filter_map(|e| e.symbol.clone())
            .collect();
        out.sort();
        out.dedup();
        out
    }

    /// Distinct timeframes (sorted by canonical resolution order).
    pub fn timeframes(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .entries
            .iter()
            .filter_map(|e| e.timeframe.clone())
            .collect();
        out.sort();
        out.dedup();
        // Stable sort by canonical-list position; non-canonical (None
        // already filtered out, but defensive) fall to the end.
        out.sort_by_key(|tf| {
            forex_core::CANONICAL_TIMEFRAMES
                .iter()
                .position(|c| *c == tf.as_str())
                .unwrap_or(usize::MAX)
        });
        out
    }

    /// Count entries per format, in declaration order of `DataFormat`.
    pub fn format_counts(&self) -> Vec<(DataFormat, usize)> {
        let all = [
            DataFormat::Vortex,
            DataFormat::Parquet,
            DataFormat::Arrow,
            DataFormat::Csv,
            DataFormat::Tsv,
            DataFormat::Json,
            DataFormat::JsonLines,
        ];
        all.iter()
            .map(|fmt| {
                (
                    *fmt,
                    self.entries
                        .iter()
                        .filter(|e| e.format == *fmt)
                        .count(),
                )
            })
            .filter(|(_, n)| *n > 0)
            .collect()
    }

    /// Count skipped files grouped by `SkipReason::category()`.
    /// Returned as a sorted Vec for stable formatting.
    pub fn skip_counts_by_category(&self) -> Vec<(String, usize)> {
        use std::collections::BTreeMap;
        let mut map: BTreeMap<String, usize> = BTreeMap::new();
        for s in &self.skipped {
            *map.entry(s.reason.category().to_string()).or_insert(0) += 1;
        }
        map.into_iter().collect()
    }

    /// True iff no data candidates were found.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ─── Symbol / timeframe inference ──────────────────────────────────────
//
// Reuses the logic patterns from `universal_importer.rs` so a file
// classified as `(EURUSD, M5)` by discovery is the same `(EURUSD, M5)`
// the importer would assign — keeping the two views consistent.

fn infer_symbol_and_timeframe(
    path: &Path,
    root: &Path,
) -> (Option<String>, Option<String>) {
    let rel = path.strip_prefix(root).unwrap_or(path);
    let mut symbol: Option<String> = None;
    let mut timeframe: Option<String> = None;

    // 1) Hive-style components on the path: symbol=EURUSD / timeframe=M5
    for component in rel.components() {
        let s = component.as_os_str().to_string_lossy().to_string();
        if let Some(rest) = s.strip_prefix("symbol=") {
            symbol = Some(rest.to_ascii_uppercase());
            continue;
        }
        if let Some(rest) = s.strip_prefix("timeframe=") {
            timeframe = Some(rest.to_ascii_uppercase());
            continue;
        }
        if symbol.is_none() && looks_like_symbol(&s) {
            symbol = Some(canonical_symbol(&s));
            continue;
        }
        if timeframe.is_none() && looks_like_timeframe_token(&s) {
            timeframe = Some(s.to_ascii_uppercase());
            continue;
        }
    }

    // 2) Filename stem tokens — handle "EURUSD_M5", "EURUSD-M5-2024",
    // and bare "M5" when the parent already gave us a symbol.
    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
        for token in stem.split(['_', '-', '.', ' ']) {
            let upper = token.to_ascii_uppercase();
            if symbol.is_none() && looks_like_symbol(&upper) {
                symbol = Some(canonical_symbol(&upper));
            }
            if timeframe.is_none() && looks_like_timeframe_token(&upper) {
                timeframe = Some(upper);
            }
        }
    }

    (symbol, timeframe)
}

fn canonical_symbol(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_uppercase()
}

fn looks_like_symbol(s: &str) -> bool {
    let canon = canonical_symbol(s);
    // 6-letter forex pair (EURUSD, GBPUSD, …)
    let is_forex_pair = canon.len() == 6 && canon.chars().all(|c| c.is_ascii_alphabetic());
    // Common 6–8 char extended instruments (gold, silver, crypto, indices)
    let is_extended = matches!(canon.len(), 6..=8)
        && (canon.starts_with("XAU")
            || canon.starts_with("XAG")
            || canon.starts_with("BTC")
            || canon.starts_with("ETH")
            || canon.starts_with("LTC")
            || canon.starts_with("SPX")
            || canon.starts_with("US30")
            || canon.starts_with("NAS"));
    is_forex_pair || is_extended
}

/// Loose timeframe-shaped token detector. Returns true for ANY token
/// that *looks* like a timeframe (so we capture H2 → and then reject
/// it as `UnsupportedTimeframe`). We do NOT call
/// `is_canonical_timeframe` here because that would silently swallow
/// non-canonical labels.
fn looks_like_timeframe_token(s: &str) -> bool {
    let upper = s.trim().to_ascii_uppercase();
    if upper.is_empty() || upper.len() > 4 {
        return false;
    }
    // Special-case the only multi-letter prefix in the canonical
    // set: "MN1" (monthly). Everything else is a single-letter
    // prefix (M/H/D/W) followed by digits.
    if upper == "MN1" {
        return true;
    }
    let mut chars = upper.chars();
    let head = chars.next();
    let tail_is_digits = chars.clone().all(|c| c.is_ascii_digit());
    let tail_has_digits = chars.clone().any(|c| c.is_ascii_digit());
    matches!(head, Some('M') | Some('H') | Some('D') | Some('W'))
        && tail_is_digits
        && tail_has_digits
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "forex_discover_{}_{}_{}",
            label,
            std::process::id(),
            nonce
        ))
    }

    fn touch(path: &Path, bytes: &[u8]) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let mut f = fs::File::create(path).unwrap();
        f.write_all(bytes).unwrap();
    }

    #[test]
    fn classifies_extensions() {
        assert_eq!(DataFormat::from_extension("csv"), Some(DataFormat::Csv));
        assert_eq!(DataFormat::from_extension("CSV"), Some(DataFormat::Csv));
        assert_eq!(
            DataFormat::from_extension("parquet"),
            Some(DataFormat::Parquet)
        );
        assert_eq!(
            DataFormat::from_extension("vortex"),
            Some(DataFormat::Vortex)
        );
        assert_eq!(
            DataFormat::from_extension("arrow"),
            Some(DataFormat::Arrow)
        );
        assert_eq!(
            DataFormat::from_extension("feather"),
            Some(DataFormat::Arrow)
        );
        assert_eq!(
            DataFormat::from_extension("jsonl"),
            Some(DataFormat::JsonLines)
        );
        assert_eq!(DataFormat::from_extension("xlsx"), None);
        assert_eq!(DataFormat::from_extension(""), None);
    }

    #[test]
    fn scan_picks_up_flat_csv_layout() {
        let root = unique_root("flat_csv");
        touch(&root.join("EURUSD_M5.csv"), b"time,o,h,l,c\n");
        touch(&root.join("GBPUSD_H1.csv"), b"time,o,h,l,c\n");

        let report = DatasetDiscovery::scan(&root).unwrap();
        assert_eq!(report.entries.len(), 2);
        assert_eq!(
            report.symbols(),
            vec!["EURUSD".to_string(), "GBPUSD".to_string()]
        );
        // Canonical order: M5 before H1.
        assert_eq!(
            report.timeframes(),
            vec!["M5".to_string(), "H1".to_string()]
        );
        assert!(report.skipped.is_empty());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn scan_picks_up_hive_partitioned_layout() {
        let root = unique_root("hive");
        touch(
            &root.join("symbol=EURUSD/timeframe=M5/2024.parquet"),
            b"",
        );
        touch(
            &root.join("symbol=EURUSD/timeframe=H1/2024.parquet"),
            b"",
        );

        let report = DatasetDiscovery::scan(&root).unwrap();
        assert_eq!(report.entries.len(), 2);
        assert_eq!(report.symbols(), vec!["EURUSD".to_string()]);
        assert!(
            report
                .entries
                .iter()
                .all(|e| e.format == DataFormat::Parquet)
        );
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn scan_rejects_non_canonical_h2_timeframe() {
        let root = unique_root("h2_skip");
        touch(&root.join("EURUSD_H2.csv"), b"");
        let report = DatasetDiscovery::scan(&root).unwrap();
        assert!(report.entries.is_empty(), "H2 must not be loaded");
        assert_eq!(report.skipped.len(), 1);
        match &report.skipped[0].reason {
            SkipReason::UnsupportedTimeframe(label) => {
                assert_eq!(label, "H2");
            }
            other => panic!("expected UnsupportedTimeframe(H2), got {:?}", other),
        }
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn scan_records_unknown_extension() {
        let root = unique_root("unknown_ext");
        touch(&root.join("rates.xlsx"), b"");
        let report = DatasetDiscovery::scan(&root).unwrap();
        assert!(report.entries.is_empty());
        assert_eq!(report.skipped.len(), 1);
        assert!(matches!(
            report.skipped[0].reason,
            SkipReason::UnknownExtension(ref e) if e == "xlsx"
        ));
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn scan_ignores_docs_and_archives_silently() {
        let root = unique_root("ignored");
        touch(&root.join("README.md"), b"hi");
        touch(&root.join("LICENSE.txt"), b"x");
        touch(&root.join("dump.tar.gz"), b"x");
        let report = DatasetDiscovery::scan(&root).unwrap();
        assert!(report.entries.is_empty());
        assert!(
            report.skipped.is_empty(),
            "docs and archives must be silent — got {:?}",
            report.skipped
        );
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn scan_missing_root_is_empty_not_error() {
        let root = unique_root("missing");
        let report = DatasetDiscovery::scan(&root).unwrap();
        assert!(report.is_empty());
    }

    #[test]
    fn timeframe_token_detector_admits_h2_but_rejects_random() {
        // H2 must be detected as timeframe-shaped (so we can skip it
        // with the right reason) — NOT silently rejected.
        assert!(looks_like_timeframe_token("H2"));
        assert!(looks_like_timeframe_token("M5"));
        assert!(looks_like_timeframe_token("MN1"));
        assert!(!looks_like_timeframe_token("HELLO"));
        assert!(!looks_like_timeframe_token(""));
    }

    #[test]
    fn format_counts_only_includes_present_formats() {
        let root = unique_root("counts");
        touch(&root.join("EURUSD_M1.csv"), b"");
        touch(&root.join("EURUSD_M1.parquet"), b"");
        touch(&root.join("EURUSD_M1.vortex"), b"");
        let report = DatasetDiscovery::scan(&root).unwrap();
        let counts = report.format_counts();
        // Each present format must appear; absent ones must not.
        let formats: Vec<DataFormat> = counts.iter().map(|(f, _)| *f).collect();
        assert!(formats.contains(&DataFormat::Csv));
        assert!(formats.contains(&DataFormat::Parquet));
        assert!(formats.contains(&DataFormat::Vortex));
        assert!(!formats.contains(&DataFormat::Json));
        fs::remove_dir_all(&root).ok();
    }
}
