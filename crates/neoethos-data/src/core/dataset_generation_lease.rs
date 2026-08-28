//! Reader pins for immutable Vortex generations.

use anyhow::{Context, Result, bail};
use std::fs::{self, File, OpenOptions, TryLockError};
use std::path::{Path, PathBuf};
use vortex_array::ArrayRef;

#[derive(Debug)]
pub struct DatasetGenerationLease {
    generation_path: PathBuf,
    expected_sha256: String,
    lock_path: PathBuf,
    file: File,
}

impl DatasetGenerationLease {
    pub(crate) fn acquire(
        dataset_root: &Path,
        generation_id: &str,
        generation_path: PathBuf,
        expected_sha256: String,
    ) -> Result<Self> {
        let lock_path = generation_lock_path(dataset_root, generation_id)?;
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&lock_path)
            .with_context(|| format!("failed to open generation lease {}", lock_path.display()))?;
        file.lock_shared()
            .with_context(|| format!("failed to lock generation lease {}", lock_path.display()))?;
        Ok(Self {
            generation_path,
            expected_sha256,
            lock_path,
            file,
        })
    }

    pub fn path(&self) -> &Path {
        &self.generation_path
    }

    pub fn reopen_verified(&self) -> Result<ArrayRef> {
        let actual = crate::core::dataset_manifest::sha256_file(&self.generation_path)?;
        if actual != self.expected_sha256 {
            bail!(
                "generation hash mismatch for {}: expected {}, got {}",
                self.generation_path.display(),
                self.expected_sha256,
                actual
            );
        }
        crate::core::vortex_io::read_vortex_array(&self.generation_path)
    }
}

impl Drop for DatasetGenerationLease {
    fn drop(&mut self) {
        if let Err(error) = self.file.unlock() {
            tracing::warn!(
                target: "neoethos_data::dataset_generation_lease",
                path = %self.lock_path.display(),
                error = %error,
                "failed to release dataset generation lease"
            );
        }
    }
}

pub(crate) struct ExclusiveGenerationLease {
    pub lock_path: PathBuf,
    file: File,
}

impl ExclusiveGenerationLease {
    pub fn unlock(self) -> Result<()> {
        self.file
            .unlock()
            .with_context(|| format!("failed to unlock {}", self.lock_path.display()))
    }
}

pub(crate) fn try_acquire_exclusive(
    dataset_root: &Path,
    generation_id: &str,
) -> Result<Option<ExclusiveGenerationLease>> {
    let lock_path = generation_lock_path(dataset_root, generation_id)?;
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(&lock_path)
        .with_context(|| format!("failed to open generation lease {}", lock_path.display()))?;
    match file.try_lock() {
        Ok(()) => Ok(Some(ExclusiveGenerationLease { lock_path, file })),
        Err(TryLockError::WouldBlock) => Ok(None),
        Err(TryLockError::Error(error)) => Err(error)
            .with_context(|| format!("failed to inspect generation lease {}", lock_path.display())),
    }
}

pub(crate) fn remove_lock_file(lock_path: &Path) -> Result<()> {
    match fs::remove_file(lock_path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error)
            .with_context(|| format!("failed to remove generation lease {}", lock_path.display())),
    }
}

fn generation_lock_path(dataset_root: &Path, generation_id: &str) -> Result<PathBuf> {
    if generation_id.is_empty()
        || generation_id == "."
        || generation_id == ".."
        || generation_id.contains(['/', '\\', ':'])
    {
        bail!("unsafe generation id {generation_id:?}");
    }
    let lock_root = dataset_root.join(".generation-leases");
    fs::create_dir_all(&lock_root).with_context(|| {
        format!(
            "failed to create generation lease directory {}",
            lock_root.display()
        )
    })?;
    Ok(lock_root.join(format!("{generation_id}.lease")))
}
