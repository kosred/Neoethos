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
use memmap2::{Mmap, MmapMut};
use ndarray::ArrayView2;
use std::fs::OpenOptions;
use std::path::Path;

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
    mmap: MmapMut,
    n_features: usize,
    n_samples: usize,
}

impl FeatureStoreWriter {
    pub fn create(path: &Path, n_features: usize, n_samples: usize) -> Result<Self> {
        let len = n_features
            .checked_mul(n_samples)
            .and_then(|v| v.checked_mul(F32_BYTES))
            .context("feature store size overflow (n_features * n_samples * 4)")?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .with_context(|| format!("create feature store {}", path.display()))?;
        file.set_len(len as u64)
            .with_context(|| format!("set_len {} on feature store", len))?;
        // SAFETY: we own the file, it is sized to `len`, and no other process
        // maps it concurrently during the build.
        let mmap = unsafe { MmapMut::map_mut(&file) }.context("mmap_mut feature store")?;
        Ok(Self {
            mmap,
            n_features,
            n_samples,
        })
    }

    #[inline]
    pub fn n_features(&self) -> usize {
        self.n_features
    }
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Write feature `idx`'s full series (`n_samples` values) at its row offset.
    pub fn write_feature(&mut self, idx: usize, series: &[f32]) -> Result<()> {
        ensure!(
            idx < self.n_features,
            "feature idx {idx} >= n_features {}",
            self.n_features
        );
        ensure!(
            series.len() == self.n_samples,
            "feature series len {} != n_samples {}",
            series.len(),
            self.n_samples
        );
        let start = idx * self.n_samples * F32_BYTES;
        let bytes: &[u8] = f32_as_bytes(series);
        self.mmap[start..start + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    /// Flush all writes to disk.
    pub fn finish(self) -> Result<()> {
        self.mmap.flush().context("flush feature store")?;
        Ok(())
    }
}

/// Read-only, mmap'd feature store — `[n_features × n_samples]` `f32`,
/// feature-major. Cheap to clone-by-reopen; `feature_row` is zero-copy.
pub struct FeatureStore {
    mmap: Mmap,
    n_features: usize,
    n_samples: usize,
}

impl FeatureStore {
    pub fn open(path: &Path, n_features: usize, n_samples: usize) -> Result<Self> {
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
            mmap,
            n_features,
            n_samples,
        })
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
        let all: &[f32] = bytes_as_f32(&self.mmap);
        let start = idx * self.n_samples;
        &all[start..start + self.n_samples]
    }

    /// `[n_features × n_samples]` view over the whole mmap — drop-in for the
    /// GA eval's `indicators` `ArrayView2` (so `view.row(idx)` is feature
    /// `idx`'s series, contiguous, OS-paged on demand).
    pub fn as_view(&self) -> ArrayView2<'_, f32> {
        let all: &[f32] = bytes_as_f32(&self.mmap);
        ArrayView2::from_shape(
            (self.n_features, self.n_samples),
            &all[..self.n_features * self.n_samples],
        )
        .expect("feature store dimensions match the mmap length")
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
            let mut w = FeatureStoreWriter::create(&path, nf, ns).unwrap();
            w.write_feature(0, &[0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
            w.write_feature(2, &[20.0, 21.0, 22.0, 23.0, 24.0]).unwrap();
            w.write_feature(1, &[10.0, 11.0, 12.0, 13.0, 14.0]).unwrap();
            w.finish().unwrap();
        }
        let s = FeatureStore::open(&path, nf, ns).unwrap();
        assert_eq!(s.n_features(), nf);
        assert_eq!(s.n_samples(), ns);
        assert_eq!(s.feature_row(1), &[10.0, 11.0, 12.0, 13.0, 14.0]);
        let v = s.as_view();
        assert_eq!(v.dim(), (nf, ns));
        assert_eq!(v.row(2).to_vec(), vec![20.0, 21.0, 22.0, 23.0, 24.0]);
        let _ = std::fs::remove_file(&path);
    }
}
