use super::super::FeatureFrame;
use super::discover::DataFormat;
use super::to_vortex::{IngestionSchema, cache_dir_for, cache_path_for, convert_to_vortex};
use super::vortex_io::{read_vortex_array, write_vortex_array};
use anyhow::{Context, Result, bail};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::{PrimitiveArray, StructArray};
use vortex_array::dtype::{FieldName, NativePType};

pub struct FeatureCache {
    pub dir: PathBuf,
    pub ttl_minutes: u64,
    pub enabled: bool,
}

impl FeatureCache {
    pub fn new(dir: &str, ttl_minutes: u64, enabled: bool) -> Self {
        Self {
            dir: PathBuf::from(dir),
            ttl_minutes,
            enabled,
        }
    }

    fn is_fresh(&self, path: &Path) -> bool {
        let Ok(meta) = std::fs::metadata(path) else {
            return false;
        };
        let Ok(mod_time) = meta.modified() else {
            return false;
        };
        let Ok(elapsed) = SystemTime::now().duration_since(mod_time) else {
            return false;
        };
        elapsed.as_secs() <= self.ttl_minutes * 60
    }

    pub fn load(&self, key: &str) -> Result<Option<FeatureFrame>> {
        if !self.enabled {
            return Ok(None);
        }
        let path = self.dir.join(format!("{key}.vortex"));
        if !path.exists() || !self.is_fresh(&path) {
            return Ok(None);
        }

        match read_vortex_array(&path).and_then(vortex_to_feature_frame) {
            Ok(frame) => Ok(Some(frame)),
            Err(err) => {
                // Cache corruption: log so we can correlate frequency with
                // upstream writer bugs, then delete and re-derive. Don't
                // bubble up; the caller treats `None` as cache-miss.
                tracing::warn!(
                    target: "neoethos_data::loader",
                    path = %path.display(),
                    error = %err,
                    "feature cache entry failed to decode; deleting and re-deriving"
                );
                if let Err(rm_err) = fs::remove_file(&path) {
                    tracing::warn!(
                        target: "neoethos_data::loader",
                        path = %path.display(),
                        error = %rm_err,
                        "feature cache: failed to delete corrupt entry"
                    );
                }
                Ok(None)
            }
        }
    }

    pub fn store(&self, key: &str, frame: &FeatureFrame) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        std::fs::create_dir_all(&self.dir)?;
        let path = self.dir.join(format!("{key}.vortex"));
        let array = feature_frame_to_vortex(frame)?;
        write_vortex_array(&path, array)?;
        Ok(())
    }
}

pub fn feature_frame_to_vortex(frame: &FeatureFrame) -> Result<vortex_array::ArrayRef> {
    let n_rows = frame.data.nrows();
    if frame.timestamps.len() != n_rows {
        bail!("feature frame timestamp/data row mismatch");
    }
    if frame.names.len() != frame.data.ncols() {
        bail!("feature frame name/data column mismatch");
    }

    let mut names: Vec<FieldName> = Vec::with_capacity(frame.names.len() + 1);
    let mut arrays = Vec::with_capacity(frame.names.len() + 1);

    names.push("timestamp".into());
    arrays.push(PrimitiveArray::from_iter(frame.timestamps.iter().copied()).into_array());

    for (idx, name) in frame.names.iter().enumerate() {
        let column = frame.data.column(idx).iter().copied().collect::<Vec<_>>();
        names.push(name.clone().into());
        arrays.push(PrimitiveArray::from_iter(column).into_array());
    }

    Ok(StructArray::try_new(
        names.into(),
        arrays,
        n_rows,
        vortex_array::validity::Validity::NonNullable,
    )
    .context("failed to build feature vortex struct array")?
    .into_array())
}

pub fn vortex_to_feature_frame(array: vortex_array::ArrayRef) -> Result<FeatureFrame> {
    let struct_array = array.to_struct();

    let timestamp_field = struct_array
        .unmasked_field_by_name("timestamp")
        .context("missing timestamp field")?;
    let timestamps = extract_non_null_primitive_vec::<i64>(timestamp_field, "timestamp")?;
    let n_rows = timestamps.len();

    let mut names = Vec::with_capacity(struct_array.names().len().saturating_sub(1));
    let mut columns = Vec::with_capacity(struct_array.names().len().saturating_sub(1));

    for name in struct_array.names().iter() {
        let field_name = name.to_string();
        if field_name == "timestamp" {
            continue;
        }
        let field = struct_array
            .unmasked_field_by_name(&field_name)
            .with_context(|| format!("missing feature field {field_name}"))?;
        let values = extract_non_null_primitive_vec::<f32>(field, &field_name)?;
        if values.len() != n_rows {
            bail!("feature field {field_name} row mismatch");
        }
        names.push(field_name);
        columns.push(values);
    }

    let n_cols = names.len();
    let mut data = ndarray::Array2::<f32>::zeros((n_rows, n_cols));
    for (col_idx, values) in columns.iter().enumerate() {
        for (row_idx, value) in values.iter().copied().enumerate() {
            data[(row_idx, col_idx)] = value;
        }
    }

    Ok(FeatureFrame {
        timestamps,
        names,
        data,
    })
}

fn extract_non_null_primitive_vec<T: NativePType>(
    array: &vortex_array::ArrayRef,
    label: &str,
) -> Result<Vec<T>> {
    if !array
        .all_valid()
        .with_context(|| format!("failed to inspect {label} validity"))?
    {
        bail!("{label} contains nulls");
    }

    Ok(array.to_primitive().as_slice::<T>().to_vec())
}

// ─── Auto-conversion entry point ───────────────────────────────────────

/// Resolve a user-supplied path to a Vortex file, converting if needed.
///
/// Behaviour:
/// - If `source` is already `.vortex`, returns the path as-is.
/// - Otherwise: detects the source format via
///   [`DataFormat::from_extension`], computes a deterministic cache
///   filename keyed on (canonical source path + mtime + size), and
///   either reuses a fresh cached Vortex or invokes
///   [`convert_to_vortex`] to produce one.
///
/// The cache lives at `<source_parent>/.forex-vortex-cache/` (per
/// [`super::to_vortex::VORTEX_CACHE_DIR_NAME`]). When the source file
/// changes (mtime or size), the hash changes and a fresh cache entry
/// is produced; the old one is left in place — operators can prune
/// `.forex-vortex-cache/` at will, it is regenerated on demand.
///
/// Returns the Vortex path that downstream code should now open via
/// [`crate::load_vortex`].
pub fn resolve_path_to_vortex(
    source: &Path,
    schema_hint: Option<&IngestionSchema>,
) -> Result<PathBuf> {
    if !source.exists() {
        bail!(
            "resolve_path_to_vortex: source missing: {}",
            source.display()
        );
    }

    let format =
        DataFormat::from_extension(source.extension().and_then(|e| e.to_str()).unwrap_or(""))
            .with_context(|| {
                format!(
                    "resolve_path_to_vortex: unsupported extension on {}",
                    source.display()
                )
            })?;

    // Fast path: already Vortex on disk.
    if format == DataFormat::Vortex {
        tracing::info!(
            target: "neoethos_data::loader",
            source = %source.display(),
            "resolve_path_to_vortex: already vortex, no conversion"
        );
        return Ok(source.to_path_buf());
    }

    let cache_path = cache_path_for(source).with_context(|| {
        format!(
            "resolve_path_to_vortex: compute cache path for {}",
            source.display()
        )
    })?;

    // Cache hit only if cache file is newer-or-equal to source mtime
    // (the filename hash already encodes mtime+size, so existence is
    // a sufficient check; the mtime check is defensive against the
    // unlikely case where someone touched the cache file).
    if cache_path.exists() && cache_is_fresh(source, &cache_path) {
        tracing::info!(
            target: "neoethos_data::loader",
            source = %source.display(),
            cache = %cache_path.display(),
            "resolve_path_to_vortex: cache hit"
        );
        return Ok(cache_path);
    }

    tracing::info!(
        target: "neoethos_data::loader",
        source = %source.display(),
        format = format.as_str(),
        cache = %cache_path.display(),
        "resolve_path_to_vortex: cache miss, converting"
    );

    let cache_dir = cache_dir_for(source);
    if !cache_dir.exists() {
        fs::create_dir_all(&cache_dir).with_context(|| {
            format!(
                "resolve_path_to_vortex: create cache dir {}",
                cache_dir.display()
            )
        })?;
    }

    convert_to_vortex(source, format, &cache_path, schema_hint).with_context(|| {
        format!(
            "resolve_path_to_vortex: convert {} -> {}",
            source.display(),
            cache_path.display()
        )
    })?;

    tracing::info!(
        target: "neoethos_data::loader",
        source = %source.display(),
        cache = %cache_path.display(),
        "resolve_path_to_vortex: conversion complete"
    );
    Ok(cache_path)
}

fn cache_is_fresh(source: &Path, cache: &Path) -> bool {
    let Ok(src_meta) = fs::metadata(source) else {
        return false;
    };
    let Ok(cache_meta) = fs::metadata(cache) else {
        return false;
    };
    let Ok(src_mtime) = src_meta.modified() else {
        return false;
    };
    let Ok(cache_mtime) = cache_meta.modified() else {
        return false;
    };
    // Cache is fresh iff it was modified at or after the source.
    cache_mtime >= src_mtime
}
