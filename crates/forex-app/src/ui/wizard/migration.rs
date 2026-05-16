//! Portable → installed migration detector + copy runtime.
//!
//! Spec: `installer_wizard_ux_spec.md` §6 "Migration from portable".
//! forex-ai pre-0.5 was portable (all state under `~/.forex-ai/`); the
//! installer-aware wizard sniffs for that directory at Step 2 entry
//! and surfaces a migration prompt.
//!
//! Detection lives in [`detect_portable_install`] / [`describe_root`].
//! The per-file copy machinery is in [`migrate_portable_install`] —
//! atomic write + content-hash verify + skip the regen-able cache.
//!
//! ## Hashing
//!
//! Spec §6 names SHA-256 explicitly for the post-write verify. The
//! `sha2` workspace dep landed in Phase 2D (see
//! `autonomy_risk::compute_quiz_answer_hash` for the canonical
//! reference impl), so this module uses `sha2::Sha256` + `hex::encode`
//! to produce a `sha256:<lowercase-hex>` content digest. The digest
//! is recomputed on the freshly-written destination bytes; a mismatch
//! aborts the migration so a torn write cannot land silently.

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use forex_core::storage::json::temporary_path;
use sha2::{Digest, Sha256};

/// Files that, if present in a candidate directory, qualify it as a
/// legacy portable install. Spec §6 enumerates these verbatim.
pub const WIZARD_PORTABLE_SENTINEL_FILES: &[&str] = &[
    "config.yaml",
    "broker_credentials.toml",
];

/// Directories that count as legacy payloads.
pub const WIZARD_PORTABLE_SENTINEL_DIRS: &[&str] = &["checkpoints", "data", "history"];

/// Where to look for the legacy portable install. Spec §6 lists three
/// canonical roots; we probe all of them and stop at the first that
/// matches.
pub fn portable_candidate_roots() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(home) = dirs::home_dir() {
        roots.push(home.join(".forex-ai"));
        roots.push(home.join("forex-ai"));
    }
    // Windows-style `%USERPROFILE%/.forex-ai` is covered by `home_dir`
    // on Windows.
    roots
}

#[derive(Debug, Clone, Default)]
pub struct PortableMigrationReport {
    pub root: PathBuf,
    pub has_config_yaml: bool,
    pub has_broker_credentials: bool,
    pub has_checkpoints: bool,
    pub has_data: bool,
    pub has_history: bool,
}

impl PortableMigrationReport {
    pub fn summary_lines(&self) -> Vec<String> {
        let mut out = vec![format!("Source: {}", self.root.display())];
        if self.has_config_yaml {
            out.push("  • config.yaml".to_string());
        }
        if self.has_broker_credentials {
            out.push("  • broker_credentials.toml".to_string());
        }
        if self.has_checkpoints {
            out.push("  • checkpoints/".to_string());
        }
        if self.has_data {
            out.push("  • data/".to_string());
        }
        if self.has_history {
            out.push("  • history/".to_string());
        }
        out
    }

    /// Has anything migrate-worthy been detected at all?
    pub fn is_actionable(&self) -> bool {
        self.has_config_yaml
            || self.has_broker_credentials
            || self.has_checkpoints
            || self.has_data
            || self.has_history
    }
}

/// Walk the candidate roots and return the first that contains at
/// least one sentinel file or dir. Returns `None` if no legacy
/// install is detected.
pub fn detect_portable_install() -> Option<PortableMigrationReport> {
    for root in portable_candidate_roots() {
        let report = describe_root(&root);
        if report.is_actionable() {
            return Some(report);
        }
    }
    None
}

/// Describe a single candidate root. Public so tests can inject a
/// scratch directory rather than touching the real `$HOME`.
pub fn describe_root(root: &Path) -> PortableMigrationReport {
    PortableMigrationReport {
        root: root.to_path_buf(),
        has_config_yaml: root.join("config.yaml").is_file(),
        has_broker_credentials: root.join("broker_credentials.toml").is_file(),
        has_checkpoints: root.join("checkpoints").is_dir(),
        has_data: root.join("data").is_dir(),
        has_history: root.join("history").is_dir(),
    }
}

/// Directories inside the portable root whose contents the migration
/// runtime deliberately skips. Spec §6 — "the cache will regen".
pub const WIZARD_MIGRATION_SKIP_DIRS: &[&str] = &["cache"];

/// Per-file outcome reported back to the wizard summary screen so a
/// partial failure can be surfaced verbatim (operator no-silent-
/// fallback rule).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigratedEntry {
    /// Source path inside the portable root (relative to the root).
    pub relative_path: PathBuf,
    /// Bytes successfully copied + fsynced.
    pub bytes: u64,
    /// SHA-256 content digest, expressed as `sha256:<lowercase-hex>`.
    /// The same string is recomputed on the destination bytes after
    /// the atomic rename — a mismatch aborts the migration.
    pub content_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PortableMigrationOutcome {
    pub copied: Vec<MigratedEntry>,
    /// Files we skipped because the destination already had identical
    /// content (idempotent re-run).
    pub skipped_identical: Vec<PathBuf>,
    /// Skipped paths whose parent is in `WIZARD_MIGRATION_SKIP_DIRS`.
    pub skipped_cache: Vec<PathBuf>,
}

impl PortableMigrationOutcome {
    pub fn total_bytes_copied(&self) -> u64 {
        self.copied.iter().map(|e| e.bytes).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.copied.is_empty()
            && self.skipped_identical.is_empty()
            && self.skipped_cache.is_empty()
    }
}

/// Copy a portable install into the installer-managed config dir.
///
/// - Reads each file under `source_root` (recursively).
/// - Skips entries whose first path component is in
///   `WIZARD_MIGRATION_SKIP_DIRS` (the cache is regen-able).
/// - Computes a content hash before the write.
/// - Atomic-writes via a sibling temp file + rename.
/// - Re-reads the destination and re-hashes; mismatch is a hard
///   error (rolled back by the temp-file discipline upstream).
///
/// If `dest_root` already contains a byte-identical file at the
/// same relative path, the copy is skipped (`skipped_identical`) so
/// a re-run is a no-op and operators can resume a partial migration
/// safely.
pub fn migrate_portable_install(
    source_root: &Path,
    dest_root: &Path,
) -> Result<PortableMigrationOutcome> {
    let mut outcome = PortableMigrationOutcome::default();

    if !source_root.is_dir() {
        return Ok(outcome); // no-op — caller already validated via `describe_root`
    }
    fs::create_dir_all(dest_root)
        .with_context(|| format!("create dest root {}", dest_root.display()))?;

    let mut stack: Vec<PathBuf> = vec![source_root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let read = fs::read_dir(&dir)
            .with_context(|| format!("read_dir {}", dir.display()))?;
        for entry in read {
            let entry = entry.with_context(|| format!("dir entry under {}", dir.display()))?;
            let path = entry.path();
            let file_type = entry
                .file_type()
                .with_context(|| format!("stat {}", path.display()))?;

            // Compute relative path from `source_root` so the
            // destination layout mirrors the source.
            let rel = path
                .strip_prefix(source_root)
                .with_context(|| format!("strip prefix for {}", path.display()))?
                .to_path_buf();

            if is_in_skip_dir(&rel) {
                if file_type.is_file() {
                    outcome.skipped_cache.push(rel);
                } else if file_type.is_dir() {
                    collect_skipped_files(&path, source_root, &mut outcome.skipped_cache)?;
                }
                continue;
            }

            if file_type.is_dir() {
                stack.push(path);
                continue;
            }
            if !file_type.is_file() {
                // Symlinks, sockets, etc. — spec §6 covers regular
                // files only. Skip silently; the visible payloads
                // are config/credentials/checkpoints/history/data.
                continue;
            }

            let dest_path = dest_root.join(&rel);
            if let Some(parent) = dest_path.parent() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("create parent {}", parent.display()))?;
            }

            // Hash + copy + verify.
            let (bytes, source_hash) = read_and_hash(&path)
                .with_context(|| format!("hash source {}", path.display()))?;

            // Idempotency check — if the destination already holds
            // the same bytes we skip without re-writing.
            if dest_path.is_file() {
                let (_, dest_hash) = read_and_hash(&dest_path)
                    .with_context(|| format!("hash dest {}", dest_path.display()))?;
                if dest_hash == source_hash {
                    outcome.skipped_identical.push(rel);
                    continue;
                }
            }

            atomic_copy(&path, &dest_path)
                .with_context(|| format!("atomic copy {}", dest_path.display()))?;

            let (written_bytes, written_hash) = read_and_hash(&dest_path)
                .with_context(|| format!("verify dest {}", dest_path.display()))?;
            if written_hash != source_hash || written_bytes != bytes {
                anyhow::bail!(
                    "post-copy sha256 mismatch at {} (source {} != dest {})",
                    dest_path.display(),
                    source_hash,
                    written_hash,
                );
            }

            outcome.copied.push(MigratedEntry {
                relative_path: rel,
                bytes,
                content_hash: written_hash,
            });
        }
    }

    Ok(outcome)
}

fn collect_skipped_files(
    dir: &Path,
    source_root: &Path,
    skipped: &mut Vec<PathBuf>,
) -> Result<()> {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let read = fs::read_dir(&current)
            .with_context(|| format!("read skipped dir {}", current.display()))?;
        for entry in read {
            let entry = entry.with_context(|| format!("dir entry under {}", current.display()))?;
            let path = entry.path();
            let file_type = entry
                .file_type()
                .with_context(|| format!("stat {}", path.display()))?;
            if file_type.is_dir() {
                stack.push(path);
            } else if file_type.is_file() {
                let rel = path
                    .strip_prefix(source_root)
                    .with_context(|| format!("strip prefix for {}", path.display()))?
                    .to_path_buf();
                skipped.push(rel);
            }
        }
    }
    Ok(())
}

/// True if the relative path's first component is in
/// `WIZARD_MIGRATION_SKIP_DIRS`.
fn is_in_skip_dir(rel: &Path) -> bool {
    rel.components()
        .next()
        .map(|c| c.as_os_str())
        .and_then(|s| s.to_str())
        .map(|first| WIZARD_MIGRATION_SKIP_DIRS.contains(&first))
        .unwrap_or(false)
}

/// Buffered read + SHA-256 content digest. Returns
/// `(bytes_read, "sha256:<lowercase-hex>")`. Mirrors
/// `autonomy_risk::compute_quiz_answer_hash` (Phase 2D reference)
/// for the hasher choice + lowercase-hex encoding.
fn read_and_hash(path: &Path) -> io::Result<(u64, String)> {
    let mut file = fs::File::open(path)?;
    let mut buf = Vec::with_capacity(8192);
    file.read_to_end(&mut buf)?;
    let mut hasher = Sha256::new();
    hasher.update(&buf);
    let digest = hex::encode(hasher.finalize());
    Ok((buf.len() as u64, format!("sha256:{}", digest)))
}

/// Copy `source → dest` via a sibling temp file + atomic rename +
/// fsync. Mirrors the discipline of
/// `forex_core::storage::json::write_json_atomic` so the operator's
/// no-torn-write policy holds for portable migration too.
fn atomic_copy(source: &Path, dest: &Path) -> Result<()> {
    let payload = fs::read(source)
        .with_context(|| format!("read source {}", source.display()))?;

    let tmp = temporary_path(dest);
    {
        use std::io::Write;
        let mut handle = fs::File::create(&tmp)
            .with_context(|| format!("create temp {}", tmp.display()))?;
        handle
            .write_all(&payload)
            .with_context(|| format!("write temp {}", tmp.display()))?;
        handle
            .sync_all()
            .with_context(|| format!("fsync temp {}", tmp.display()))?;
    }
    fs::rename(&tmp, dest).with_context(|| {
        format!(
            "atomic rename {} → {}",
            tmp.display(),
            dest.display()
        )
    })?;
    // Best-effort directory fsync to match the json-atomic writer.
    if let Some(parent) = dest.parent() {
        if let Ok(d) = fs::File::open(parent) {
            // Same rationale as `write_json_atomic`: rename was
            // atomic, dir-fsync is belt-and-braces. We log via the
            // standard `tracing` target so a real ENOTSUP shows up.
            if let Err(err) = d.sync_all() {
                tracing::debug!(
                    target: "forex_app::ui::wizard::migration",
                    dir = %parent.display(),
                    error = %err,
                    "dir fsync after migration copy failed (non-fatal)"
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp_dir(label: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("forex-ai-wizard-migration-{}-{}", label, nanos));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn describe_empty_root_is_not_actionable() {
        let root = tmp_dir("empty");
        let report = describe_root(&root);
        assert!(!report.is_actionable());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn describe_root_detects_config_yaml_only() {
        let root = tmp_dir("config-only");
        fs::write(root.join("config.yaml"), b"placeholder").unwrap();
        let report = describe_root(&root);
        assert!(report.has_config_yaml);
        assert!(report.is_actionable());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn describe_root_detects_dirs() {
        let root = tmp_dir("dirs");
        fs::create_dir_all(root.join("history")).unwrap();
        fs::create_dir_all(root.join("checkpoints")).unwrap();
        let report = describe_root(&root);
        assert!(report.has_history);
        assert!(report.has_checkpoints);
        assert!(!report.has_data);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn portable_sentinel_files_constants_are_non_empty() {
        assert!(!WIZARD_PORTABLE_SENTINEL_FILES.is_empty());
        assert!(!WIZARD_PORTABLE_SENTINEL_DIRS.is_empty());
    }

    #[test]
    fn skip_dir_predicate_matches_first_component_only() {
        assert!(is_in_skip_dir(Path::new("cache/old.bin")));
        assert!(is_in_skip_dir(Path::new("cache")));
        assert!(!is_in_skip_dir(Path::new("data/cache/old.bin")));
        assert!(!is_in_skip_dir(Path::new("config.yaml")));
    }

    /// `migration_skip_on_no_portable_install` — empty `~/.forex-ai/`,
    /// no-op (operator brief required test).
    #[test]
    fn migration_skip_on_no_portable_install() {
        let src = tmp_dir("skip-src");
        let dest = tmp_dir("skip-dest");
        // Empty source — no files exist.
        let outcome = migrate_portable_install(&src, &dest).expect("noop migration");
        assert!(outcome.is_empty());
        let _ = fs::remove_dir_all(&src);
        let _ = fs::remove_dir_all(&dest);
    }

    #[test]
    fn migration_returns_empty_outcome_when_source_root_missing() {
        let dest = tmp_dir("nonexistent-src-dest");
        let bogus = std::env::temp_dir().join("forex-ai-this-does-not-exist-and-never-will");
        let outcome = migrate_portable_install(&bogus, &dest).expect("missing root is a no-op");
        assert!(outcome.is_empty());
        let _ = fs::remove_dir_all(&dest);
    }

    /// `migration_copies_files_with_sha256_verify` — populate the
    /// source tempdir with portable payloads, assert copy + post-write
    /// SHA-256 round-trip + cache directory skipped (operator brief).
    #[test]
    fn migration_copies_files_with_sha256_verify() {
        let src = tmp_dir("copy-src");
        let dest = tmp_dir("copy-dest");

        // Populate a realistic-ish portable tree.
        fs::write(src.join("config.yaml"), b"system:\n  symbol: EURUSD\n").unwrap();
        fs::write(src.join("broker_credentials.toml"), b"[ctrader]\nclient_id = \"x\"\n").unwrap();
        fs::create_dir_all(src.join("history")).unwrap();
        fs::write(src.join("history/EURUSD_M1.parquet"), b"PAR1XXX").unwrap();
        fs::create_dir_all(src.join("checkpoints")).unwrap();
        fs::write(src.join("checkpoints/model.bin"), b"checkpoint-bytes").unwrap();
        // Cache file that MUST be skipped (regen-able).
        fs::create_dir_all(src.join("cache")).unwrap();
        fs::write(src.join("cache/transient.bin"), b"throwaway").unwrap();

        let outcome = migrate_portable_install(&src, &dest).expect("migrate");

        // Cache directory's payload must be in skipped_cache, not copied.
        assert_eq!(outcome.skipped_cache.len(), 1);
        assert_eq!(outcome.skipped_cache[0], PathBuf::from("cache/transient.bin"));
        assert!(!dest.join("cache/transient.bin").exists());

        // Every non-cache file must be copied + present + hashed.
        let expected_copied: std::collections::HashSet<PathBuf> = [
            PathBuf::from("config.yaml"),
            PathBuf::from("broker_credentials.toml"),
            PathBuf::from("history/EURUSD_M1.parquet"),
            PathBuf::from("checkpoints/model.bin"),
        ]
        .into_iter()
        .collect();
        let actual_copied: std::collections::HashSet<PathBuf> = outcome
            .copied
            .iter()
            .map(|e| e.relative_path.clone())
            .collect();
        assert_eq!(actual_copied, expected_copied);

        // Round-trip integrity: every destination file's bytes match
        // the source file's bytes.
        for entry in &outcome.copied {
            let source_bytes = fs::read(src.join(&entry.relative_path)).unwrap();
            let dest_bytes = fs::read(dest.join(&entry.relative_path)).unwrap();
            assert_eq!(source_bytes, dest_bytes, "{} bytes match", entry.relative_path.display());
            assert_eq!(entry.bytes as usize, source_bytes.len());
            assert!(
                entry.content_hash.starts_with("sha256:"),
                "expected sha256: prefix, got {}",
                entry.content_hash
            );
        }

        // Re-run is idempotent — second pass copies nothing, skips
        // every file as identical.
        let outcome2 = migrate_portable_install(&src, &dest).expect("idempotent re-run");
        assert!(outcome2.copied.is_empty());
        assert_eq!(outcome2.skipped_identical.len(), 4);

        let _ = fs::remove_dir_all(&src);
        let _ = fs::remove_dir_all(&dest);
    }

    #[test]
    fn migration_overwrites_when_source_differs() {
        let src = tmp_dir("overwrite-src");
        let dest = tmp_dir("overwrite-dest");
        fs::write(src.join("config.yaml"), b"new-content").unwrap();
        fs::write(dest.join("config.yaml"), b"old-content").unwrap();

        let outcome = migrate_portable_install(&src, &dest).expect("migrate");
        assert_eq!(outcome.copied.len(), 1);
        assert_eq!(fs::read(dest.join("config.yaml")).unwrap(), b"new-content");
        let _ = fs::remove_dir_all(&src);
        let _ = fs::remove_dir_all(&dest);
    }

    /// Pin that the migration runtime hashes with real SHA-256
    /// (not the old fnv64 placeholder). Independently computes the
    /// SHA-256 digest of a 64-byte payload and asserts the migrator
    /// agrees byte-for-byte on the lowercase-hex encoding.
    #[test]
    fn migration_uses_real_sha256_not_fnv64() {
        use sha2::{Digest, Sha256};

        let src = tmp_dir("sha256-src");
        let dest = tmp_dir("sha256-dest");

        // 64 deterministic bytes — values chosen to exercise the full
        // 0..=63 range so a length- or byte-order bug would show.
        let payload: Vec<u8> = (0u8..64).collect();
        fs::write(src.join("config.yaml"), &payload).unwrap();

        let outcome = migrate_portable_install(&src, &dest).expect("migrate");
        assert_eq!(outcome.copied.len(), 1);
        let entry = &outcome.copied[0];
        assert_eq!(entry.relative_path, PathBuf::from("config.yaml"));
        assert_eq!(entry.bytes, 64);

        // Independently compute SHA-256(payload).
        let mut hasher = Sha256::new();
        hasher.update(&payload);
        let expected = format!("sha256:{}", hex::encode(hasher.finalize()));

        assert_eq!(
            entry.content_hash, expected,
            "migration digest must equal independently-computed SHA-256"
        );
        // Guard against any regression to the old fnv64 prefix.
        assert!(!entry.content_hash.starts_with("fnv64:"));
        assert!(entry.content_hash.starts_with("sha256:"));
        // sha256 hex is exactly 64 chars; with the "sha256:" prefix
        // the full string is 7 + 64 = 71 chars.
        assert_eq!(entry.content_hash.len(), 7 + 64);

        let _ = fs::remove_dir_all(&src);
        let _ = fs::remove_dir_all(&dest);
    }
}
