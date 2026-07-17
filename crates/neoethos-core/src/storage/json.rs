use anyhow::{Context, Result};
use serde::{Serialize, de::DeserializeOwned};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsonBackupWriteConfig {
    pub artifact_label: &'static str,
    pub temp_extension: &'static str,
    pub backup_extension: &'static str,
}

pub fn write_json_atomic<T: Serialize + ?Sized>(path: impl AsRef<Path>, value: &T) -> Result<()> {
    let mut json = serde_json::to_vec_pretty(value).context("serialize artifact")?;
    json.push(b'\n'); // terminate json artifact
    write_bytes_atomic(path, &json)
}

/// Atomically write raw bytes to `path` (audit M07): serialize into a UNIQUE
/// hidden temp file in the SAME directory, fsync it, then atomically rename
/// it over the target. A crash at any point leaves either the previous file
/// intact or the new file complete — never a truncated/partial file. Same-
/// directory temp guarantees the rename is a same-filesystem atomic op; the
/// per-call-unique temp name means concurrent writers never clobber each
/// other's staging file. Use for any canonical on-disk state (config.yaml,
/// symbol metadata, …) where a half-written file would be corruption.
pub fn write_bytes_atomic(path: impl AsRef<Path>, bytes: &[u8]) -> Result<()> {
    let path = path.as_ref();
    // M07 writer lock: serialize same-target writers within this process so
    // two threads saving the same file can't interleave their temp-write +
    // rename sequences (each write stays all-or-nothing regardless, but
    // without the lock the LOSER's rename could land after the winner's,
    // reordering updates non-deterministically). Keyed by the path as given;
    // callers use canonical config/artifact paths so aliasing is not a
    // concern in practice. Cross-PROCESS coordination remains last-writer-
    // wins via the atomic rename (unchanged semantics, never a torn file).
    static WRITER_LOCKS: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<PathBuf, std::sync::Arc<std::sync::Mutex<()>>>>,
    > = std::sync::OnceLock::new();
    let lock = {
        let registry = WRITER_LOCKS.get_or_init(Default::default);
        let mut map = registry.lock().unwrap_or_else(|e| e.into_inner());
        map.entry(path.to_path_buf()).or_default().clone()
    };
    let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("create artifact directory {}", parent.display()))?;
    let tmp_path = temporary_path(path);
    {
        let mut tmp = File::create(&tmp_path)
            .with_context(|| format!("create temp artifact {}", tmp_path.display()))?;
        tmp.write_all(bytes)
            .with_context(|| format!("write temp artifact {}", tmp_path.display()))?;
        tmp.sync_all()
            .with_context(|| format!("fsync temp artifact {}", tmp_path.display()))?;
    }
    rename_with_windows_retry(&tmp_path, path).with_context(|| {
        format!(
            "atomically rename {} to {}",
            tmp_path.display(),
            path.display()
        )
    })?;
    if let Ok(dir) = File::open(parent) {
        // Directory fsync is best-effort: some filesystems (tmpfs, NFS, FAT)
        // legitimately return EINVAL. The atomic rename above is what
        // guarantees crash safety; the dir-sync is belt-and-braces.
        // Log at debug so a real syscall failure on a real FS is still
        // observable.
        if let Err(err) = dir.sync_all() {
            tracing::debug!(
                target: "neoethos_core::storage::json",
                dir = %parent.display(),
                error = %err,
                "fsync(parent_dir) failed; rename was atomic so this is non-fatal"
            );
        }
    }
    Ok(())
}

pub fn write_json_with_backup<T: Serialize + ?Sized>(
    path: impl AsRef<Path>,
    value: &T,
    config: JsonBackupWriteConfig,
) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "create {} artifact directory {}",
                config.artifact_label,
                parent.display()
            )
        })?;
    }

    let temp_path = path.with_extension(config.temp_extension);
    let backup_path = path.with_extension(config.backup_extension);
    let payload = serde_json::to_vec_pretty(value)
        .with_context(|| format!("serialize {}", config.artifact_label))?;
    if temp_path.exists() {
        fs::remove_file(&temp_path).with_context(|| {
            format!(
                "remove stale staged {} {}",
                config.artifact_label,
                temp_path.display()
            )
        })?;
    }
    if backup_path.exists() {
        fs::remove_file(&backup_path).with_context(|| {
            format!(
                "remove stale backup {} {}",
                config.artifact_label,
                backup_path.display()
            )
        })?;
    }
    {
        let mut temp = match File::create(&temp_path) {
            Ok(file) => file,
            // Defensive: under heavy parallel training, a concurrent
            // stage/cleanup of the SAME artifact dir can remove our parent dir
            // between the create_dir_all above and this File::create (ENOENT) —
            // this surfaced as `online_pa`/`online_hoeffding` "No such file or
            // directory" failures. Re-create the parent and retry once before
            // giving up, so a transient race doesn't sink the model.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent).with_context(|| {
                        format!(
                            "re-create {} artifact directory {}",
                            config.artifact_label,
                            parent.display()
                        )
                    })?;
                }
                File::create(&temp_path).with_context(|| {
                    format!(
                        "create staged {} {} (after parent re-create)",
                        config.artifact_label,
                        temp_path.display()
                    )
                })?
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "create staged {} {}",
                        config.artifact_label,
                        temp_path.display()
                    )
                });
            }
        };
        temp.write_all(&payload).with_context(|| {
            format!(
                "write staged {} to {}",
                config.artifact_label,
                temp_path.display()
            )
        })?;
        temp.sync_all().with_context(|| {
            format!(
                "fsync staged {} {}",
                config.artifact_label,
                temp_path.display()
            )
        })?;
    }
    if path.exists() {
        fs::rename(path, &backup_path).with_context(|| {
            format!(
                "backup current {} {}",
                config.artifact_label,
                path.display()
            )
        })?;
    }
    if let Err(error) = fs::rename(&temp_path, path) {
        // Best-effort rollback. Failure of either rollback step is logged
        // but does NOT mask the original write error — the caller still
        // sees a `bail!` so they treat the artifact as not-written.
        if backup_path.exists() {
            if let Err(rb_err) = fs::rename(&backup_path, path) {
                tracing::error!(
                    target: "neoethos_core::storage::json",
                    artifact = config.artifact_label,
                    backup = %backup_path.display(),
                    path = %path.display(),
                    error = %rb_err,
                    "failed to restore backup after write failure"
                );
            }
        } else if temp_path.exists() {
            if let Err(rb_err) = fs::remove_file(&temp_path) {
                tracing::warn!(
                    target: "neoethos_core::storage::json",
                    artifact = config.artifact_label,
                    temp = %temp_path.display(),
                    error = %rb_err,
                    "failed to remove staged temp file after write failure"
                );
            }
        }
        anyhow::bail!(
            "write {} to {} failed: {}",
            config.artifact_label,
            path.display(),
            error
        );
    }
    if backup_path.exists() {
        fs::remove_file(&backup_path).with_context(|| {
            format!(
                "remove backup {} {}",
                config.artifact_label,
                backup_path.display()
            )
        })?;
    }
    Ok(())
}

/// Configuration for [`write_dir_with_backup`]. Mirrors
/// [`JsonBackupWriteConfig`] but for **directory-level** atomic
/// replacement (e.g. multi-file model artifacts: `model.json` +
/// `metadata.json` + `weights.bin` in the same dir).
///
/// `temp_extension` and `backup_extension` are appended to the
/// target dir path via `Path::with_extension`. Example:
/// target `/models/bayesian/eurusd_m1` with `temp_extension =
/// "tmp_bayesian_artifact"` resolves to
/// `/models/bayesian/eurusd_m1.tmp_bayesian_artifact`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirBackupWriteConfig {
    /// Human-readable label used in error and tracing messages.
    /// E.g. `"meta-model artifact"` or `"bayesian artifact"`.
    pub artifact_label: &'static str,
    /// Extension appended to the staged-temp directory. Convention:
    /// `tmp_<family>_artifact` (e.g. `tmp_bayesian_artifact`).
    pub temp_extension: &'static str,
    /// Extension appended to the backup-of-previous directory.
    /// Convention: `bak_<family>_artifact`.
    pub backup_extension: &'static str,
}

/// Atomically replace a target DIRECTORY by running `writer` against
/// a staged-temp dir and renaming it into place. Mirrors
/// [`write_json_with_backup`] semantics but at directory granularity,
/// for multi-file artifacts (model + metadata + weights, etc.).
///
/// GROUP E consolidation (operator directive 2026-05-25): replaces 4
/// duplicate implementations across `neoethos-models` —
/// `ensemble.rs`, `statistical/bayesian_impl.rs`,
/// `statistical/linear_impl.rs`, `training_orchestrator.rs` — each
/// of which hand-rolled ~80-100 LOC of identical staged-tmp +
/// backup + atomic-rename + rollback logic.
///
/// Contract:
/// 1. Compute `staged_path = target.with_extension(temp_extension)`
///    and `backup_path = target.with_extension(backup_extension)`.
/// 2. Delete any stale staged_path from a previous interrupted run.
/// 3. Create staged_path (empty dir).
/// 4. Run `writer(&staged_path)` — caller writes its files here.
/// 5. If writer errored → clean up staged_path and propagate the error.
/// 6. Delete any stale backup_path.
/// 7. If target exists → rename it to backup_path.
/// 8. Rename staged_path → target. If THIS fails, try to restore
///    backup_path → target (log a structured error on restore failure
///    so the operator sees the inconsistency).
/// 9. Delete backup_path on success (or leave it if restore was needed).
///
/// Use this whenever a model expert needs to atomically replace its
/// on-disk artifact directory in a way that survives a crash mid-write.
pub fn write_dir_with_backup<F>(
    path: impl AsRef<Path>,
    config: DirBackupWriteConfig,
    writer: F,
) -> Result<()>
where
    F: FnOnce(&Path) -> Result<()>,
{
    let path = path.as_ref();
    let staged_path = path.with_extension(config.temp_extension);
    let backup_path = path.with_extension(config.backup_extension);

    // Step 1-2: clean up any stale staged dir from a prior interrupted run.
    cleanup_dir_if_present(&staged_path, config.artifact_label, "staged")?;

    // Step 3: create the staged dir.
    fs::create_dir_all(&staged_path).with_context(|| {
        format!(
            "create staged {} dir {}",
            config.artifact_label,
            staged_path.display()
        )
    })?;

    // Step 4-5: run the writer; clean up staged on error.
    if let Err(error) = writer(&staged_path) {
        let _ = cleanup_dir_if_present(&staged_path, config.artifact_label, "staged");
        return Err(error);
    }

    // Step 6: clean up any stale backup from a prior interrupted run.
    cleanup_dir_if_present(&backup_path, config.artifact_label, "backup")?;

    // Step 7: if target exists, move it to backup.
    if path.exists() {
        fs::rename(path, &backup_path).with_context(|| {
            format!(
                "move previous {} into backup {}",
                config.artifact_label,
                backup_path.display()
            )
        })?;
    }

    // Step 8: rename staged → target. On failure, attempt backup restore.
    if let Err(error) = fs::rename(&staged_path, path) {
        if backup_path.exists() {
            if let Err(restore_err) = fs::rename(&backup_path, path) {
                tracing::error!(
                    target: "neoethos_core::storage::dir",
                    artifact = config.artifact_label,
                    backup = %backup_path.display(),
                    target = %path.display(),
                    error = %restore_err,
                    "failed to restore backup after staged-rename failure; \
                     artifact directory may be in an inconsistent state"
                );
            }
        }
        anyhow::bail!(
            "rename staged {} into {} failed: {}",
            config.artifact_label,
            path.display(),
            error
        );
    }

    // Step 9: clean up the backup on success.
    cleanup_dir_if_present(&backup_path, config.artifact_label, "backup")?;
    Ok(())
}

fn cleanup_dir_if_present(path: &Path, artifact_label: &str, role: &str) -> Result<()> {
    if path.exists() {
        fs::remove_dir_all(path).with_context(|| {
            format!(
                "remove {role} {} dir {}",
                artifact_label,
                path.display()
            )
        })?;
    }
    Ok(())
}

pub fn read_json<T: DeserializeOwned>(path: impl AsRef<Path>, artifact_label: &str) -> Result<T> {
    let path = path.as_ref();
    let payload = fs::read(path)
        .with_context(|| format!("read {artifact_label} artifact {}", path.display()))?;
    serde_json::from_slice(&payload)
        .with_context(|| format!("parse {artifact_label} artifact {}", path.display()))
}

/// `fs::rename` with a bounded retry on Windows transient failures (M07).
///
/// Rust's `std::fs::rename` DOES replace an existing destination on Windows
/// (`MoveFileExW`/`FileRenameInfoEx` with replace-existing — verified by the
/// replace test in this module on Windows). What CAN happen is a TRANSIENT
/// sharing violation / access-denied when another process holds the
/// destination open without `FILE_SHARE_DELETE` at that instant — antivirus
/// scanners and concurrent readers are the classic culprits. Those clear in
/// milliseconds, so retry briefly instead of failing a config/metadata save
/// over a scanner's 20 ms window. Non-transient errors fail immediately.
fn rename_with_windows_retry(from: &Path, to: &Path) -> std::io::Result<()> {
    const ATTEMPTS: u32 = 10;
    const BACKOFF: std::time::Duration = std::time::Duration::from_millis(20);
    let mut last_err = None;
    for attempt in 0..ATTEMPTS {
        match fs::rename(from, to) {
            Ok(()) => return Ok(()),
            Err(err) => {
                // ERROR_ACCESS_DENIED (5) / ERROR_SHARING_VIOLATION (32) are
                // the transient Windows cases; PermissionDenied covers the
                // mapped kind on other platforms. Anything else is real.
                let transient = matches!(err.raw_os_error(), Some(5) | Some(32))
                    || err.kind() == std::io::ErrorKind::PermissionDenied;
                if !transient || attempt + 1 == ATTEMPTS {
                    return Err(err);
                }
                last_err = Some(err);
                std::thread::sleep(BACKOFF);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| std::io::Error::other("rename retry exhausted")))
}

pub fn temporary_path(path: &Path) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact.json");
    // pid + per-call sequence → unique even for concurrent writers to the
    // same target within one process (M07), and unique across processes.
    let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
    path.with_file_name(format!(".{file_name}.tmp-{}-{seq}", std::process::id()))
}

pub fn stable_json_hash<T: Serialize + ?Sized>(value: &T) -> Result<String> {
    let bytes = serde_json::to_vec(value).context("serialize value for stable hash")?;
    Ok(format!("fnv64:{:016x}", crate::utils::fnv1a64(&bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct SampleArtifact {
        name: String,
        value: usize,
    }

    fn unique_test_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("current time should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "neoethos_core_json_io_{name}_{}_{}",
            std::process::id(),
            nanos
        ))
    }

    #[test]
    fn atomic_json_write_round_trips_and_uses_hidden_temp_path() {
        let dir = unique_test_dir("atomic");
        let path = dir.join("artifact.json");
        let artifact = SampleArtifact {
            name: "alpha".to_string(),
            value: 7,
        };

        write_json_atomic(&path, &artifact).expect("write atomic json");
        let reloaded: SampleArtifact = read_json(&path, "sample").expect("read atomic json");

        assert_eq!(reloaded, artifact);
        assert!(
            temporary_path(&path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .starts_with(".artifact.json.tmp-")
        );
        std::fs::remove_dir_all(&dir).expect("cleanup atomic json dir");
    }

    #[test]
    fn write_bytes_atomic_round_trips_replaces_and_leaves_no_temp() {
        let dir = unique_test_dir("bytes");
        let path = dir.join("state.yaml");

        write_bytes_atomic(&path, b"first: 1\n").expect("write first");
        assert_eq!(std::fs::read(&path).unwrap(), b"first: 1\n");

        // Replacing overwrites atomically and leaves no staging file behind.
        write_bytes_atomic(&path, b"second: 2\n").expect("replace");
        assert_eq!(std::fs::read(&path).unwrap(), b"second: 2\n");
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp-"))
            .collect();
        assert!(leftovers.is_empty(), "no temp file may linger: {leftovers:?}");

        std::fs::remove_dir_all(&dir).expect("cleanup bytes dir");
    }

    #[test]
    fn concurrent_writers_to_same_path_never_corrupt() {
        // M07 hardening: 8 threads × 25 writes each hammering ONE target.
        // Every observed final state must be ONE COMPLETE payload (never a
        // torn/interleaved file), no temp files may linger, and the writes
        // must serialize (the per-path writer lock) so replace-over-open
        // windows on Windows are also exercised.
        let dir = unique_test_dir("concurrent");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.json");
        let path_ref = &path;

        std::thread::scope(|scope| {
            for writer in 0..8u32 {
                scope.spawn(move || {
                    for i in 0..25u32 {
                        // Distinct, self-consistent payload per write: the
                        // marker appears at both ends so a torn mix of two
                        // writers is detectable.
                        let marker = format!("w{writer}-i{i}");
                        let body = format!("{marker}|{}|{marker}\n", "x".repeat(512));
                        write_bytes_atomic(path_ref, body.as_bytes())
                            .expect("concurrent atomic write must succeed");
                    }
                });
            }
        });

        let content = std::fs::read_to_string(&path).expect("final file readable");
        let parts: Vec<&str> = content.trim_end().split('|').collect();
        assert_eq!(parts.len(), 3, "payload structure intact: {content:?}");
        assert_eq!(parts[0], parts[2], "head/tail markers must match (no torn write)");
        assert_eq!(parts[1].len(), 512, "body length intact");

        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp-"))
            .collect();
        assert!(leftovers.is_empty(), "no temp files may linger: {leftovers:?}");
        std::fs::remove_dir_all(&dir).expect("cleanup concurrent dir");
    }

    #[test]
    fn temporary_path_is_unique_per_call() {
        // M07: two staging paths for the same target must differ so concurrent
        // writers never clobber each other's temp file.
        let target = Path::new("/some/dir/config.yaml");
        let a = temporary_path(target);
        let b = temporary_path(target);
        assert_ne!(a, b, "each call must yield a unique temp path");
        assert_eq!(a.parent(), target.parent(), "temp must stay in the same dir");
    }

    #[test]
    fn backup_json_write_replaces_existing_file_and_removes_staging_files() {
        let dir = unique_test_dir("backup");
        let path = dir.join("artifact.json");
        let first = SampleArtifact {
            name: "first".to_string(),
            value: 1,
        };
        let second = SampleArtifact {
            name: "second".to_string(),
            value: 2,
        };
        let config = JsonBackupWriteConfig {
            artifact_label: "sample artifact",
            temp_extension: "tmp_sample",
            backup_extension: "bak_sample",
        };

        write_json_with_backup(&path, &first, config).expect("write first payload");
        write_json_with_backup(&path, &second, config).expect("replace payload");
        let reloaded: SampleArtifact = read_json(&path, "sample").expect("read replaced json");

        assert_eq!(reloaded, second);
        assert!(!path.with_extension("tmp_sample").exists());
        assert!(!path.with_extension("bak_sample").exists());
        std::fs::remove_dir_all(&dir).expect("cleanup backup json dir");
    }

    #[test]
    fn stable_json_hash_uses_canonical_fnv64_prefix() {
        let artifact = SampleArtifact {
            name: "alpha".to_string(),
            value: 7,
        };

        let first = stable_json_hash(&artifact).expect("hash first");
        let second = stable_json_hash(&artifact).expect("hash second");

        assert_eq!(first, second);
        assert!(first.starts_with("fnv64:"));
    }
}
