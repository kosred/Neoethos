// REMOVED 2026-08-02: `FeatureCache`, plus the `feature_frame_to_vortex` /
// `vortex_to_feature_frame` codec that existed only to serve it.
//
// `FeatureCache::load` and `::store` had ZERO callers. Meanwhile FOUR sites
// constructed one — neoethos-cli/src/main.rs (three) and
// app_services/discovery.rs — and passed it into a parameter that
// `prepare_multitimeframe_features_with_options` had literally named
// `_cache`. Four call sites believed features were being cached; the
// underscore says the author knew they were not.
//
// It was not wired up, because as designed it could not be wired up safely:
//
//   FRESHNESS BY WALL CLOCK. `is_fresh` compared the entry's mtime against a
//   60-minute TTL and nothing else. Not the source data, not the timeframe
//   set, not the feature profile, not `normalize_features`, not the
//   indicator registry version. Two runs with identical inputs an hour apart
//   would see different features, and — the case that matters — the
//   re-import that fixed the 14 240 corrupt zero-price bars would have been
//   shadowed by stale cached features for an hour after it landed.
//
//   PEAK MEMORY AS A FUNCTION OF USER PARAMS. `store` calls
//   `feature_frame_to_vortex`, which materialises every column into a Vec
//   and then a StructArray. The frame it would be handed is often a
//   `FeatureData::Mmap` — the streaming sink that exists precisely so the
//   ~13 GB M1 cube never lands in RAM at once. Caching it would have
//   re-materialised the whole thing, which is the NEVER-OOM invariant
//   inverted.
//
// A correctly-keyed feature-cube cache is still worth having. It must key on
// the INPUTS (source path + mtime + size per timeframe, base/higher TF set,
// FeatureBuildOptions, indicator registry version) rather than on the clock,
// and stream rather than materialise. `resolve_path_to_vortex` below is the
// model: same crate, same file, keyed on canonical path + mtime + size, and
// actually called.

use super::discover::DataFormat;
use super::to_vortex::{IngestionSchema, cache_dir_for, cache_path_for, convert_to_vortex};
use anyhow::{Context, Result, bail};
use std::fs;
use std::path::{Path, PathBuf};

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
