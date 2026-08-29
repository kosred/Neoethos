//! Cross-process liveness for run-scoped Vortex feature scratch data.
//!
//! PID files and timestamps are diagnostic only: they cannot distinguish a
//! long-running process from a crashed process or PID reuse. An operating-
//! system file lock outside the run directory is the sole liveness authority.
//! Process death releases the lock; cleanup never follows a symlink/reparse
//! entry or removes a run while its lock is held.

use std::fs::{self, File, OpenOptions, TryLockError};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};

const RUN_PREFIX: &str = "run-";
const LEASE_DIRECTORY: &str = ".feature-run-leases";

#[derive(Debug)]
pub struct FeatureRunLease {
    scratch_root: PathBuf,
    run_id: String,
    run_dir: PathBuf,
    lock_path: PathBuf,
    lock_file: File,
    remove_on_drop: bool,
}

impl FeatureRunLease {
    /// Create a new run directory and take its exclusive cross-process lease.
    pub fn create(scratch_root: impl AsRef<Path>, run_id: &str) -> Result<Self> {
        let (scratch_root, lock_path, run_dir) = prepared_paths(scratch_root.as_ref(), run_id)?;
        let lock_file = acquire_exclusive(&lock_path)?;
        if run_dir.exists() {
            bail!(
                "feature run directory already exists for run id `{run_id}`: {}; sweep or recover it explicitly",
                run_dir.display()
            );
        }
        fs::create_dir(&run_dir)
            .with_context(|| format!("failed to create feature run {}", run_dir.display()))?;
        verify_contained_directory(&scratch_root, &run_dir, "feature run")?;
        Ok(Self {
            scratch_root,
            run_id: run_id.to_owned(),
            run_dir,
            lock_path,
            lock_file,
            remove_on_drop: true,
        })
    }

    /// Open an existing run only after acquiring its exclusive lease.
    ///
    /// This is intended for explicit recovery; an active owner makes the call
    /// fail immediately instead of blocking or sharing mutable scratch state.
    pub fn open_existing(scratch_root: impl AsRef<Path>, run_id: &str) -> Result<Self> {
        let (scratch_root, lock_path, run_dir) = prepared_paths(scratch_root.as_ref(), run_id)?;
        let lock_file = acquire_exclusive(&lock_path)?;
        ensure!(
            run_dir.is_dir(),
            "feature run `{run_id}` does not exist at {}",
            run_dir.display()
        );
        verify_contained_directory(&scratch_root, &run_dir, "feature run")?;
        Ok(Self {
            scratch_root,
            run_id: run_id.to_owned(),
            run_dir,
            lock_path,
            lock_file,
            remove_on_drop: true,
        })
    }

    pub fn scratch_root(&self) -> &Path {
        &self.scratch_root
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn run_dir(&self) -> &Path {
        &self.run_dir
    }

    pub fn lock_path(&self) -> &Path {
        &self.lock_path
    }
}

impl Drop for FeatureRunLease {
    fn drop(&mut self) {
        // Keep the exclusive OS lock held while removing the run. A new owner
        // therefore cannot observe a half-deleted directory.
        if self.remove_on_drop {
            match fs::remove_dir_all(&self.run_dir) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => tracing::warn!(
                    target: "neoethos_data::feature_run_lease",
                    run_id = %self.run_id,
                    path = %self.run_dir.display(),
                    error = %error,
                    "failed to remove Vortex feature run during RAII cleanup"
                ),
            }
        }
        if let Err(error) = self.lock_file.unlock() {
            tracing::warn!(
                target: "neoethos_data::feature_run_lease",
                run_id = %self.run_id,
                path = %self.lock_path.display(),
                error = %error,
                "failed to release Vortex feature run lease"
            );
        }
    }
}

/// Remove only contained run directories whose OS lease can be acquired
/// exclusively without waiting. Age, PID, and diagnostic files are ignored.
pub fn sweep_orphan_feature_runs(scratch_root: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let scratch_root = scratch_root.as_ref();
    if !scratch_root.exists() {
        return Ok(Vec::new());
    }
    let scratch_root = canonical_scratch_root(scratch_root)?;
    let mut removed = Vec::new();
    for entry in fs::read_dir(&scratch_root)
        .with_context(|| format!("failed to enumerate {}", scratch_root.display()))?
    {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(run_id) = name.strip_prefix(RUN_PREFIX) else {
            continue;
        };
        if file_type.is_symlink() {
            bail!(
                "refusing to inspect symlinked feature run {}",
                entry.path().display()
            );
        }
        if !file_type.is_dir() {
            continue;
        }
        validate_run_id(run_id)?;
        let run_dir = entry.path();
        verify_contained_directory(&scratch_root, &run_dir, "feature run candidate")?;
        let lock_path = lease_path(&scratch_root, run_id)?;
        let lock_file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&lock_path)
            .with_context(|| format!("failed to open feature lease {}", lock_path.display()))?;
        match lock_file.try_lock() {
            Ok(()) => {
                fs::remove_dir_all(&run_dir).with_context(|| {
                    format!("failed to remove orphan feature run {}", run_dir.display())
                })?;
                lock_file.unlock().with_context(|| {
                    format!("failed to unlock feature lease {}", lock_path.display())
                })?;
                removed.push(run_dir);
            }
            Err(TryLockError::WouldBlock) => {}
            Err(TryLockError::Error(error)) => {
                return Err(error).with_context(|| {
                    format!("failed to inspect feature lease {}", lock_path.display())
                });
            }
        }
    }
    removed.sort();
    Ok(removed)
}

fn prepared_paths(scratch_root: &Path, run_id: &str) -> Result<(PathBuf, PathBuf, PathBuf)> {
    validate_run_id(run_id)?;
    fs::create_dir_all(scratch_root)
        .with_context(|| format!("failed to create scratch root {}", scratch_root.display()))?;
    let scratch_root = canonical_scratch_root(scratch_root)?;
    let lock_path = lease_path(&scratch_root, run_id)?;
    let run_dir = scratch_root.join(format!("{RUN_PREFIX}{run_id}"));
    ensure!(
        run_dir.parent() == Some(scratch_root.as_path()),
        "feature run path escaped its scratch root"
    );
    Ok((scratch_root, lock_path, run_dir))
}

fn canonical_scratch_root(scratch_root: &Path) -> Result<PathBuf> {
    let metadata = fs::symlink_metadata(scratch_root)
        .with_context(|| format!("failed to inspect scratch root {}", scratch_root.display()))?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "feature scratch root must be a real directory, not a symlink: {}",
        scratch_root.display()
    );
    scratch_root.canonicalize().with_context(|| {
        format!(
            "failed to canonicalize scratch root {}",
            scratch_root.display()
        )
    })
}

fn lease_path(scratch_root: &Path, run_id: &str) -> Result<PathBuf> {
    let lease_directory = scratch_root.join(LEASE_DIRECTORY);
    fs::create_dir_all(&lease_directory).with_context(|| {
        format!(
            "failed to create feature lease root {}",
            lease_directory.display()
        )
    })?;
    verify_contained_directory(scratch_root, &lease_directory, "feature lease root")?;
    Ok(lease_directory.join(format!("{run_id}.lease")))
}

fn acquire_exclusive(lock_path: &Path) -> Result<File> {
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(lock_path)
        .with_context(|| format!("failed to open feature lease {}", lock_path.display()))?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(TryLockError::WouldBlock) => bail!(
            "feature run is already active under lease {}",
            lock_path.display()
        ),
        Err(TryLockError::Error(error)) => Err(error)
            .with_context(|| format!("failed to lock feature lease {}", lock_path.display())),
    }
}

fn validate_run_id(run_id: &str) -> Result<()> {
    ensure!(
        !run_id.is_empty()
            && run_id.len() <= 128
            && run_id != "."
            && run_id != ".."
            && run_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')),
        "unsafe feature run id `{run_id}`"
    );
    Ok(())
}

fn verify_contained_directory(root: &Path, path: &Path, label: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect {label} {}", path.display()))?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "{label} must be a real contained directory: {}",
        path.display()
    );
    let canonical = path
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {label} {}", path.display()))?;
    ensure!(
        canonical.parent() == Some(root),
        "{label} escaped scratch root: {}",
        canonical.display()
    );
    Ok(())
}
