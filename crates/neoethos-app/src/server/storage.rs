//! `/storage/paths` — configured locations and bounded, read-only metadata.
//! Counts describe filesystem entries, not strategies or datasets. This is a
//! best-effort snapshot, not an atomic or race-hard filesystem authority.

use std::path::{Path, PathBuf};

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

use neoethos_core::Settings;

use super::errors::{actionable_error, internal_panic};
use super::state::AppApiState;

const SCAN_ENTRY_LIMIT: usize = 200_000;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StorageEntry {
    pub key: String,
    pub label: String,
    pub path: String,
    pub exists: bool,
    pub is_dir: bool,
    pub size_bytes: u64,
    /// Observed immediate children, or one for a regular-file root.
    pub item_count: usize,
    /// Newest observed metadata timestamp; partial scans need not find the newest.
    pub last_modified_ms: Option<i64>,
    /// data | models | journal | logs | config | secret | cache
    pub kind: String,
    pub scan_status: &'static str,
    pub scan_error: Option<String>,
}

impl StorageEntry {
    fn scan_failed(&mut self, status: &'static str, error: impl ToString) {
        self.scan_status = status;
        self.scan_error.get_or_insert_with(|| error.to_string());
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StoragePathsDto {
    pub entries: Vec<StorageEntry>,
}

fn mtime_ms(meta: &std::fs::Metadata) -> Option<i64> {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .and_then(|d| i64::try_from(d.as_millis()).ok())
}

fn is_link(meta: &std::fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        // FILE_ATTRIBUTE_REPARSE_POINT also covers directory junctions.
        if is_link_attributes(false, meta.file_attributes()) {
            return true;
        }
    }
    is_link_attributes(meta.file_type().is_symlink(), 0)
}

fn is_link_attributes(symlink: bool, attributes: u32) -> bool {
    symlink || attributes & 0x400 != 0
}

/// One global entry budget, including iterator errors. Query metadata afresh
/// (Windows DirEntry metadata can be cached), and recheck directories before
/// descent. Ancestor links/replacement are not a race-hard confinement boundary.
fn stats(
    path: &Path,
    result: &mut StorageEntry,
    limit: usize,
    metadata: impl Fn(&Path) -> std::io::Result<std::fs::Metadata>,
) {
    let meta = match metadata(path) {
        Ok(meta) => meta,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            result.scan_status = "missing";
            return;
        }
        Err(error) => {
            result.scan_failed("unavailable", error);
            return;
        }
    };
    result.exists = true;
    result.is_dir = meta.is_dir();
    result.last_modified_ms = mtime_ms(&meta);
    if is_link(&meta) || (!meta.is_dir() && !meta.is_file()) {
        result.scan_failed(
            "unavailable",
            "Root is a link, reparse point or unsupported file type; not scanned.",
        );
        return;
    }
    if meta.is_file() {
        result.size_bytes = meta.len();
        result.item_count = 1;
        return;
    }

    let mut stack = vec![path.to_path_buf()];
    let mut visited = 0usize;
    'walk: while let Some(dir) = stack.pop() {
        if dir != path && visited == limit {
            result.scan_failed(
                "partial",
                format!("Stopped at the global {limit}-entry scan limit."),
            );
            break;
        }
        let entries = metadata(&dir).and_then(|meta| {
            if is_link(&meta) || !meta.is_dir() {
                return Err(std::io::Error::other(
                    "Directory changed or is a link/reparse point; not scanned.",
                ));
            }
            std::fs::read_dir(&dir)
        });
        let entries = match entries {
            Ok(entries) => entries,
            Err(error) => {
                result.scan_failed(
                    if dir == path {
                        "unavailable"
                    } else {
                        "partial"
                    },
                    error,
                );
                continue;
            }
        };
        for item in entries {
            if visited == limit {
                result.scan_failed(
                    "partial",
                    format!("Stopped at the global {limit}-entry scan limit."),
                );
                break 'walk;
            }
            visited += 1;
            let item = match item {
                Ok(item) => item,
                Err(error) => {
                    result.scan_failed("partial", error);
                    continue;
                }
            };
            if dir == path {
                result.item_count += 1; // bounded by visited
            }
            let child = item.path();
            let meta = match metadata(&child) {
                Ok(meta) => meta,
                Err(error) => {
                    result.scan_failed("partial", error);
                    continue;
                }
            };
            if is_link(&meta) || (!meta.is_dir() && !meta.is_file()) {
                result.scan_failed(
                    "partial",
                    "Links, reparse points or unsupported file types were not scanned.",
                );
                continue;
            }
            if let Some(time) = mtime_ms(&meta) {
                result.last_modified_ms =
                    Some(result.last_modified_ms.map_or(time, |old| old.max(time)));
            }
            if meta.is_dir() {
                stack.push(child);
            } else if let Some(size) = result.size_bytes.checked_add(meta.len()) {
                result.size_bytes = size;
            } else {
                result.scan_failed("partial", "Byte total overflowed; scan stopped.");
                break 'walk;
            }
        }
    }
}

fn entry(key: &str, label: &str, kind: &str, path: PathBuf) -> anyhow::Result<StorageEntry> {
    // Unlike canonicalize this works for missing paths and preserves verbatim
    // Windows path semantics without stripping their prefix.
    let path = std::path::absolute(path)?;
    let mut result = StorageEntry {
        key: key.to_owned(),
        label: label.to_owned(),
        path: path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Storage path is not valid Unicode"))?
            .to_owned(),
        exists: false,
        is_dir: false,
        size_bytes: 0,
        item_count: 0,
        last_modified_ms: None,
        kind: kind.to_owned(),
        scan_status: "complete",
        scan_error: None,
    };
    stats(&path, &mut result, SCAN_ENTRY_LIMIT, |path| {
        std::fs::symlink_metadata(path)
    });
    Ok(result)
}

fn scan_paths(config_path: PathBuf) -> anyhow::Result<StoragePathsDto> {
    // A broken/missing explicit config is not permission to display default paths.
    let settings = Settings::from_yaml(&config_path)?;
    let data_dir = settings.system.data_dir;
    let cache_dir = settings.system.cache_dir;
    let credentials = neoethos_core::broker_config::credentials_file_path()?;
    let entries = vec![
        entry(
            "config",
            "Engine config (config.yaml)",
            "config",
            config_path,
        )?,
        entry("data", "Market data (Vortex)", "data", data_dir.clone())?,
        entry(
            "models",
            "Trained models",
            "models",
            PathBuf::from("models"),
        )?,
        entry("cache", "Engine cache", "cache", cache_dir)?,
        entry(
            "journal",
            "Trade journal",
            "journal",
            data_dir.join("journal"),
        )?,
        entry(
            "logs",
            "Logs",
            "logs",
            neoethos_core::logging::default_log_dir(),
        )?,
        entry("credentials", "Broker credentials", "secret", credentials)?,
    ];
    Ok(StoragePathsDto { entries })
}

pub async fn paths(State(state): State<AppApiState>) -> Response {
    let config_path = state.config_path().to_path_buf();
    match tokio::task::spawn_blocking(move || scan_paths(config_path)).await {
        Ok(Ok(dto)) => Json(dto).into_response(),
        Ok(Err(error)) => actionable_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Storage paths could not be resolved. Check the configured settings file and path overrides.",
            &error,
        ),
        Err(error) => internal_panic("Reading storage paths", error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::{Error, ErrorKind, Write};

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir()
                .join(format!("neoethos-storage-{}-{nonce}", std::process::id()));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn blank() -> StorageEntry {
        StorageEntry {
            key: "test".into(),
            label: "Test".into(),
            path: String::new(),
            exists: false,
            is_dir: false,
            size_bytes: 0,
            item_count: 0,
            last_modified_ms: None,
            kind: "cache".into(),
            scan_status: "complete",
            scan_error: None,
        }
    }

    #[test]
    fn missing_paths_are_absolute_and_not_measured_empty_directories() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = PathBuf::from(format!(
            "neoethos-absent-storage-{}-{nonce}",
            std::process::id()
        ));
        assert!(!path.exists());
        let report = entry("missing", "Missing", "cache", path).unwrap();
        assert!(Path::new(&report.path).is_absolute());
        assert!(!report.exists);
        assert_eq!(report.scan_status, "missing");
        assert!(report.scan_error.is_none());
        let wire = serde_json::to_value(report).unwrap();
        assert_eq!(wire["scanStatus"], "missing");
        assert_eq!(wire["scanError"], serde_json::Value::Null);
    }

    #[test]
    fn counts_nested_bytes_top_level_items_and_open_file_metadata() {
        let root = Fixture::new();
        fs::create_dir(root.0.join("nested")).unwrap();
        fs::write(root.0.join("nested/a"), [0u8; 3]).unwrap();
        let mut open = fs::File::create(root.0.join("active.log")).unwrap();
        open.write_all(&[1u8; 5]).unwrap();
        open.flush().unwrap(); // keep the writer open during the metadata read
        let report = entry("root", "Root", "logs", root.0.clone()).unwrap();
        assert_eq!(report.scan_status, "complete");
        assert_eq!((report.size_bytes, report.item_count), (8, 2));
        assert!(report.last_modified_ms.is_some());
        let file = entry("file", "File", "logs", root.0.join("active.log")).unwrap();
        assert_eq!((file.size_bytes, file.item_count), (5, 1));
        assert!(!file.is_dir);
    }

    #[test]
    fn zero_one_and_n_budgets_stop_the_whole_walk_before_queued_subtrees() {
        let root = Fixture::new();
        for name in ["a", "b", "c"] {
            fs::create_dir(root.0.join(name)).unwrap();
            fs::write(root.0.join(name).join("payload"), [1u8; 7]).unwrap();
        }
        for limit in [0, 1, 2, 3, 4, 6, 7] {
            let queries = std::cell::RefCell::new(Vec::new());
            let mut report = blank();
            stats(&root.0, &mut report, limit, |path| {
                queries.borrow_mut().push(path.to_path_buf());
                fs::symlink_metadata(path)
            });
            assert!(report.item_count <= limit.min(3));
            if limit <= 3 {
                assert_eq!(report.size_bytes, 0);
                assert_eq!(queries.borrow().len(), 2 + limit);
            }
            if limit < 6 {
                assert_eq!(report.scan_status, "partial");
                assert!(report.scan_error.unwrap().contains("global"));
            } else {
                assert_eq!(report.scan_status, "complete");
                assert_eq!((report.item_count, report.size_bytes), (3, 21));
            }
        }
        let empty = Fixture::new();
        let mut report = blank();
        stats(&empty.0, &mut report, 0, |path| fs::symlink_metadata(path));
        assert_eq!(report.scan_status, "complete");
    }

    #[test]
    fn injected_unreadable_metadata_is_not_missing_or_complete() {
        let root = Fixture::new();
        let mut unavailable = blank();
        stats(&root.0, &mut unavailable, 10, |_| {
            Err(Error::new(ErrorKind::PermissionDenied, "root denied"))
        });
        assert_eq!(unavailable.scan_status, "unavailable");
        assert_eq!(unavailable.scan_error.as_deref(), Some("root denied"));

        // The metadata snapshot says directory, but read_dir observes a file:
        // deterministic replacement/read failure, with no permission assumptions.
        let changed = root.0.join("changed");
        fs::write(&changed, []).unwrap();
        let mut read_failed = blank();
        stats(&changed, &mut read_failed, 10, |_| {
            fs::symlink_metadata(&root.0)
        });
        assert_eq!(read_failed.scan_status, "unavailable");
        assert!(read_failed.scan_error.is_some());

        fs::write(root.0.join("denied"), [0u8; 9]).unwrap();
        fs::write(root.0.join("visible"), [0u8; 4]).unwrap();
        let mut partial = blank();
        stats(&root.0, &mut partial, 10, |path| {
            if path.file_name().is_some_and(|name| name == "denied") {
                Err(Error::new(ErrorKind::PermissionDenied, "child denied"))
            } else {
                fs::symlink_metadata(path)
            }
        });
        assert_eq!(partial.scan_status, "partial");
        assert_eq!((partial.size_bytes, partial.item_count), (4, 3));
        assert_eq!(partial.scan_error.as_deref(), Some("child denied"));
    }

    #[test]
    fn byte_overflow_preserves_observed_total_and_reports_partial() {
        let root = Fixture::new();
        fs::write(root.0.join("file"), [1u8]).unwrap();
        let mut report = blank();
        report.size_bytes = u64::MAX; // exercise overflow without a huge fixture
        stats(&root.0, &mut report, 10, |path| fs::symlink_metadata(path));
        assert_eq!(report.size_bytes, u64::MAX);
        assert_eq!(report.scan_status, "partial");
        assert!(report.scan_error.unwrap().contains("overflowed"));
    }

    #[test]
    fn link_and_windows_junction_attribute_are_both_refused() {
        assert!(is_link_attributes(true, 0));
        assert!(is_link_attributes(false, 0x400));
        assert!(is_link_attributes(false, 0x410));
        assert!(!is_link_attributes(false, 0x10));
    }

    #[cfg(unix)]
    #[test]
    fn root_and_nested_symlinks_are_not_followed() {
        let root = Fixture::new();
        fs::create_dir(root.0.join("target")).unwrap();
        fs::write(root.0.join("target/payload"), [0u8; 32]).unwrap();
        std::os::unix::fs::symlink(root.0.join("target"), root.0.join("link")).unwrap();
        let link = entry("link", "Link", "cache", root.0.join("link")).unwrap();
        assert_eq!(link.scan_status, "unavailable");
        assert_eq!(link.size_bytes, 0);
        let parent = entry("root", "Root", "cache", root.0.clone()).unwrap();
        assert_eq!(parent.scan_status, "partial");
        assert_eq!(parent.size_bytes, 32);
    }

    #[test]
    fn broken_and_missing_config_never_substitute_default_storage_paths() {
        let root = Fixture::new();
        let config = root.0.join("config.yaml");
        fs::write(&config, "system: [").unwrap();
        let before = fs::read(&config).unwrap();
        let error = scan_paths(config.clone()).unwrap_err();
        assert!(error.to_string().contains("not valid YAML"));
        assert_eq!(fs::read(config).unwrap(), before);
        assert_eq!(fs::read_dir(&root.0).unwrap().count(), 1);
        assert!(scan_paths(root.0.join("missing.yaml")).is_err());
    }
}
