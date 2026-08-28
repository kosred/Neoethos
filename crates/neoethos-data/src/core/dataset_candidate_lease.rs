//! Cross-process liveness for unpublished Vortex candidates.
//!
//! Age and PID files cannot distinguish a slow writer from a crashed writer
//! (and PIDs are reusable). The operating-system file lock is the sole
//! liveness authority: process death releases it automatically.

use anyhow::{Context, Result};
use std::fs::{self, File, OpenOptions, TryLockError};
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct DatasetCandidateLease {
    candidate_path: PathBuf,
    lock_path: PathBuf,
    file: File,
}

impl DatasetCandidateLease {
    pub fn acquire(candidate_path: impl AsRef<Path>) -> Result<Self> {
        let candidate_path = candidate_path.as_ref().to_path_buf();
        let lock_path = candidate_lock_path(&candidate_path)?;
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&lock_path)
            .with_context(|| format!("failed to open candidate lease {}", lock_path.display()))?;
        file.lock()
            .with_context(|| format!("failed to lock candidate lease {}", lock_path.display()))?;
        Ok(Self {
            candidate_path,
            lock_path,
            file,
        })
    }

    pub fn candidate_path(&self) -> &Path {
        &self.candidate_path
    }

    pub fn lock_path(&self) -> &Path {
        &self.lock_path
    }
}

impl Drop for DatasetCandidateLease {
    fn drop(&mut self) {
        if let Err(error) = self.file.unlock() {
            tracing::warn!(
                target: "neoethos_data::dataset_candidate_lease",
                path = %self.lock_path.display(),
                error = %error,
                "failed to release dataset candidate lease"
            );
        }
    }
}

pub fn collect_orphan_candidates(directory: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let directory = directory.as_ref();
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let mut removed = Vec::new();
    for entry in fs::read_dir(directory)
        .with_context(|| format!("failed to enumerate candidate root {}", directory.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if entry.file_type()?.is_symlink()
            || !(file_name.starts_with("candidate-") || file_name.starts_with(".candidate-"))
            || path.extension().and_then(|extension| extension.to_str()) != Some("vortex")
        {
            continue;
        }
        let lock_path = candidate_lock_path(&path)?;
        let lock_file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&lock_path)
            .with_context(|| format!("failed to open candidate lease {}", lock_path.display()))?;
        match lock_file.try_lock() {
            Ok(()) => {
                fs::remove_file(&path).with_context(|| {
                    format!("failed to remove orphan candidate {}", path.display())
                })?;
                lock_file.unlock().with_context(|| {
                    format!("failed to unlock candidate lease {}", lock_path.display())
                })?;
                drop(lock_file);
                match fs::remove_file(&lock_path) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => {
                        return Err(error).with_context(|| {
                            format!("failed to remove candidate lease {}", lock_path.display())
                        });
                    }
                }
                removed.push(path);
            }
            Err(TryLockError::WouldBlock) => {}
            Err(TryLockError::Error(error)) => {
                return Err(error).with_context(|| {
                    format!("failed to inspect candidate lease {}", lock_path.display())
                });
            }
        }
    }
    Ok(removed)
}

fn candidate_lock_path(candidate_path: &Path) -> Result<PathBuf> {
    let file_name = candidate_path
        .file_name()
        .and_then(|name| name.to_str())
        .context("candidate path must have a UTF-8 file name")?;
    Ok(candidate_path.with_file_name(format!("{file_name}.lease")))
}
