//! Memory-mapped, feature-major feature store — `[n_features × n_samples]`
//! `f32`, where each feature's full time series is one **contiguous** mmap row.
//!
//! ## Why this exists
//!
//! The discovery used to materialise the multi-resolution feature set as a
//! dense in-RAM `ndarray::Array2<f32>` in `[samples × features]` layout
//! (`FeatureFrame.data`). For full M1 data that is **~100 GB**
//! (5.27M samples × ~4700 features × 4 bytes). On top of that, the GA hot
//! path needs the *transposed* `[features × samples]` layout
//! (`indicators.row(idx)`, idx = gene-selected feature), so
//! `search_engine::transpose_features` made a **second full copy** — another
//! ~100 GB → ~200 GB peak. That exceeds the Windows commit limit and OOMs
//! even when physical RAM is free.
//!
//! ## The fix
//!
//! Build features **directly** in `[features × samples]` (feature-major) into
//! a memory-mapped file. This:
//!   - removes the dense `[samples × features]` matrix, and
//!   - removes the transpose-copy (the layout IS already what the GA wants),
//!
//! and lets the OS page in **only the feature rows the population actually
//! references** (the GA's working set), so RAM stays bounded regardless of
//! total data size. The store is written one feature series at a time
//! (RAM-bounded during the build) and mmap'd read-only for the GA.
//!
//! Layout: row `i` (feature `i`) occupies bytes
//! `[i*n_samples*4 .. (i+1)*n_samples*4)`. The whole file viewed as `&[f32]`
//! is the `[n_features × n_samples]` matrix in C (row-major) order.

use anyhow::{Context, Result, ensure};
use memmap2::Mmap;
use ndarray::ArrayView2;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const F32_BYTES: usize = std::mem::size_of::<f32>();

/// Reinterpret an `&[f32]` as raw bytes. Always sound: `f32` is plain-old-data
/// (every bit pattern is a valid byte sequence) and `u8` has alignment 1.
#[inline]
fn f32_as_bytes(s: &[f32]) -> &[u8] {
    // SAFETY: POD source, byte (align-1) destination, same lifetime.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

/// Reinterpret raw bytes as `&[f32]`. Sound when `b` is 4-byte aligned and its
/// length is a multiple of 4 — guaranteed here because the only callers pass a
/// page-aligned mmap whose length is `n_features * n_samples * 4`.
#[inline]
fn bytes_as_f32(b: &[u8]) -> &[f32] {
    debug_assert_eq!(b.len() % F32_BYTES, 0, "mmap byte length not a multiple of 4");
    debug_assert_eq!(
        b.as_ptr() as usize % std::mem::align_of::<f32>(),
        0,
        "mmap not 4-byte aligned"
    );
    // SAFETY: alignment + length invariants asserted above; same lifetime as `b`.
    unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len() / F32_BYTES) }
}

/// Writer: pre-sizes the backing file and writes feature rows by index.
///
/// Typical use: `create(path, n_features, n_samples)`, then `write_feature`
/// for each feature index `0..n_features` (in any order — the build computes
/// per-TF and writes each TF's features at their global row offsets), then
/// `finish()` to flush.
pub struct FeatureStoreWriter {
    file: BufWriter<std::fs::File>,
    n_samples: usize,
    n_written: usize,
}

impl FeatureStoreWriter {
    /// Create an empty append-mode writer. Stream features in with
    /// [`Self::append_feature`] (one full `n_samples` series each); the feature
    /// count is the number of appends and is returned by [`Self::finish`] — no
    /// upfront total is needed, which is exactly what the per-TF build wants.
    pub fn create(path: &Path, n_samples: usize) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .with_context(|| format!("create feature store {}", path.display()))?;
        Ok(Self {
            file: BufWriter::with_capacity(1 << 20, file),
            n_samples,
            n_written: 0,
        })
    }

    #[inline]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }
    #[inline]
    pub fn n_features(&self) -> usize {
        self.n_written
    }

    /// Append one feature's full time series (`n_samples` values) as the next
    /// feature-major row. Streaming (RAM-bounded) — no whole-matrix buffer.
    pub fn append_feature(&mut self, series: &[f32]) -> Result<()> {
        ensure!(
            series.len() == self.n_samples,
            "feature series len {} != n_samples {}",
            series.len(),
            self.n_samples
        );
        self.file
            .write_all(f32_as_bytes(series))
            .context("append feature to store")?;
        self.n_written += 1;
        Ok(())
    }

    /// Flush and return the number of features written.
    pub fn finish(mut self) -> Result<usize> {
        self.file.flush().context("flush feature store")?;
        Ok(self.n_written)
    }
}

/// Read-only, mmap'd feature store — `[n_features × n_samples]` `f32`,
/// feature-major. `feature_row` is zero-copy.
///
/// When `delete_on_drop` is set, the backing file is removed when the store
/// drops (the mmap is unmapped first, as Windows refuses to delete a mapped
/// file). This is the discovery path: a per-(symbol, timeframe) store can be
/// ~13 GB for full M1 data, so leaving 14 symbols' stores on disk would
/// accumulate ~180 GB. RAII cleanup ties each store's lifetime to the
/// `FeatureFrame` (held in an `Arc`, so the file is removed exactly once when
/// the last reference drops — typically at the end of that discovery run).
#[derive(Debug)]
pub struct FeatureStore {
    /// `Option` so `Drop` can unmap (take) before deleting the file.
    mmap: Option<Mmap>,
    n_features: usize,
    n_samples: usize,
    path: PathBuf,
    delete_on_drop: bool,
}

impl FeatureStore {
    pub fn open(
        path: &Path,
        n_features: usize,
        n_samples: usize,
        delete_on_drop: bool,
    ) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .open(path)
            .with_context(|| format!("open feature store {}", path.display()))?;
        // SAFETY: read-only mapping of a file we sized during the build.
        let mmap = unsafe { Mmap::map(&file) }.context("mmap feature store")?;
        let expect = n_features
            .checked_mul(n_samples)
            .and_then(|v| v.checked_mul(F32_BYTES))
            .context("feature store size overflow")?;
        ensure!(
            mmap.len() >= expect,
            "feature store {} too small: {} bytes < expected {}",
            path.display(),
            mmap.len(),
            expect
        );
        Ok(Self {
            mmap: Some(mmap),
            n_features,
            n_samples,
            path: path.to_path_buf(),
            delete_on_drop,
        })
    }

    /// Bytes of the live mmap. Panics only if called after `Drop` took it,
    /// which cannot happen through a live `&self`.
    #[inline]
    fn bytes(&self) -> &[f32] {
        bytes_as_f32(
            self.mmap
                .as_ref()
                .expect("feature store mmap is live for the store's lifetime"),
        )
    }

    #[inline]
    pub fn n_features(&self) -> usize {
        self.n_features
    }
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Feature `idx`'s contiguous time series — a zero-copy slice into the
    /// mmap. Touching it pages in only this feature's row (~`n_samples*4`
    /// bytes), which is exactly the GA's per-gene access.
    #[inline]
    pub fn feature_row(&self, idx: usize) -> &[f32] {
        debug_assert!(idx < self.n_features);
        let all: &[f32] = self.bytes();
        let start = idx * self.n_samples;
        &all[start..start + self.n_samples]
    }

    /// `[n_features × n_samples]` view over the whole mmap — drop-in for the
    /// GA eval's `indicators` `ArrayView2` (so `view.row(idx)` is feature
    /// `idx`'s series, contiguous, OS-paged on demand).
    pub fn as_view(&self) -> ArrayView2<'_, f32> {
        let all: &[f32] = self.bytes();
        ArrayView2::from_shape(
            (self.n_features, self.n_samples),
            &all[..self.n_features * self.n_samples],
        )
        .expect("feature store dimensions match the mmap length")
    }
}

impl Drop for FeatureStore {
    fn drop(&mut self) {
        if self.delete_on_drop {
            // Capture size before unmap so we can report exactly how much disk
            // is reclaimed when this TF's cube is released (operator visibility
            // — the build streams 12+ GB cubes that were previously invisible).
            let freed_mb = std::fs::metadata(&self.path)
                .map(|m| m.len() / (1 << 20))
                .unwrap_or(0);
            // Unmap BEFORE deleting — Windows denies removal of a mapped file.
            self.mmap = None;
            match std::fs::remove_file(&self.path) {
                Ok(()) => {
                    tracing::info!(
                        target: "neoethos_data::feature_store",
                        path = %self.path.display(),
                        freed_mb,
                        "released feature cube — disk reclaimed"
                    );
                }
                // Best-effort: a leftover temp file is harmless (next run
                // truncate-creates over it); only warn so disk creep is visible.
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                    tracing::warn!(
                        target: "neoethos_data::feature_store",
                        path = %self.path.display(),
                        error = %e,
                        "failed to remove feature store backing file on drop"
                    );
                }
                Err(_) => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read_roundtrips_feature_major() {
        let dir = std::env::temp_dir().join("neoethos_fs_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("roundtrip.fstore");
        let (nf, ns) = (3usize, 5usize);
        {
            let mut w = FeatureStoreWriter::create(&path, ns).unwrap();
            w.append_feature(&[0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
            w.append_feature(&[10.0, 11.0, 12.0, 13.0, 14.0]).unwrap();
            w.append_feature(&[20.0, 21.0, 22.0, 23.0, 24.0]).unwrap();
            assert_eq!(w.finish().unwrap(), nf);
        }
        let s = FeatureStore::open(&path, nf, ns, false).unwrap();
        assert_eq!(s.n_features(), nf);
        assert_eq!(s.n_samples(), ns);
        assert_eq!(s.feature_row(1), &[10.0, 11.0, 12.0, 13.0, 14.0]);
        let v = s.as_view();
        assert_eq!(v.dim(), (nf, ns));
        assert_eq!(v.row(2).to_vec(), vec![20.0, 21.0, 22.0, 23.0, 24.0]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn delete_on_drop_removes_backing_file() {
        let dir = std::env::temp_dir().join("neoethos_fs_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("delete_on_drop.fstore");
        {
            let mut w = FeatureStoreWriter::create(&path, 2).unwrap();
            w.append_feature(&[1.0, 2.0]).unwrap();
            assert_eq!(w.finish().unwrap(), 1);
        }
        assert!(path.exists(), "store file should exist after writing");
        {
            let s = FeatureStore::open(&path, 1, 2, true).unwrap();
            assert_eq!(s.feature_row(0), &[1.0, 2.0]);
        } // store drops here → file removed
        assert!(
            !path.exists(),
            "delete_on_drop store must remove its backing file on drop"
        );
    }
}
