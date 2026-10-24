//! Receipt-keyed research-only feature-frame cache.
//!
//! Repeat canonical-research runs over the same dataset rebuild the exact
//! feature cube (~90 minutes on the frozen EURUSD M15+H1,H4 corpus). This
//! module persists a completed frame under a key derived from the input
//! identity - the per-timeframe dataset generation/manifest/Vortex digests,
//! the exact feature-build recipe (including the streaming working-set batch)
//! and the resolved vector-ta math authority - so an identical input digest
//! skips the build. Any digest change produces a different key and therefore a
//! miss. Correctness is NOT trusted to the key alone: the CLI seals each
//! stored frame with its canonical search receipt and refuses a cache hit
//! whose recomputed receipt differs from the sealed one.
//!
//! RESEARCH-ONLY: the cache is never active unless the operator sets
//! NEOETHOS_FEATURE_FRAME_CACHE_DIR. Disk usage is bounded by
//! NEOETHOS_FEATURE_FRAME_CACHE_MAX_BYTES (default 32 GiB); the oldest
//! entries are evicted when the bound is exceeded.

use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;

use neoethos_feature_contracts::SourceArtifactBindingV1;

use crate::core::features::{
    FeatureBuildOptions, FeatureCellValidity, FeatureColumnF64, FeatureFrame, FeatureFrameRowIds,
};
use crate::core::hpc_ta::{SweepBatch, resolved_canonical_feature_execution_authority_v1};
use crate::core::normalization::SearchNormalizationFittedStateV1;
use neoethos_feature_contracts::{DatasetFeatureArtifactProvenanceV1, FeaturePlanV1};

/// Environment variable naming the cache directory. Unset (or empty) disables
/// the cache entirely; the production feature build path is then unchanged.
pub const ENV_FEATURE_FRAME_CACHE_DIR: &str = "NEOETHOS_FEATURE_FRAME_CACHE_DIR";
/// Environment variable naming the total disk budget in bytes.
pub const ENV_FEATURE_FRAME_CACHE_MAX_BYTES: &str = "NEOETHOS_FEATURE_FRAME_CACHE_MAX_BYTES";
/// Default disk budget: 32 GiB (approximately three frozen EURUSD cube entries).
pub const DEFAULT_FEATURE_FRAME_CACHE_MAX_BYTES: u64 = 32 * 1024 * 1024 * 1024;

const CACHE_KEY_DOMAIN_V1: &[u8] = b"neoethos.feature-frame-cache-key.v1\0";
const CACHE_META_SCHEMA_V1: &str = "neoethos.feature-frame-cache-entry.v1";
const CACHE_META_VERSION_V1: u32 = 1;
const CACHE_PAYLOAD_MAGIC_V1: &[u8] = b"NEOTFFC1";

/// Research-only feature-frame cache configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureFrameCacheConfig {
    pub dir: PathBuf,
    pub max_total_bytes: u64,
}

/// Resolve the cache configuration from the environment. None disables the
/// cache and keeps the production build path byte-identical.
pub fn feature_frame_cache_config_from_env() -> Option<FeatureFrameCacheConfig> {
    let dir = std::env::var_os(ENV_FEATURE_FRAME_CACHE_DIR)?;
    if dir.is_empty() {
        return None;
    }
    let dir = PathBuf::from(dir);
    let max_total_bytes = std::env::var(ENV_FEATURE_FRAME_CACHE_MAX_BYTES)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_FEATURE_FRAME_CACHE_MAX_BYTES);
    Some(FeatureFrameCacheConfig { dir, max_total_bytes })
}

fn validate_cache_key(key: &str) -> Result<()> {
    ensure!(
        key.len() == 64 && key.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "feature-frame cache key must be 64 lowercase hex characters, got {key:?}"
    );
    Ok(())
}

/// Compute the input-receipt cache key for one feature build.
///
/// The key covers every digest that identifies the input: the symbol, every
/// timeframe's exact dataset identity/manifest/Vortex generation (as
/// SourceArtifactBindingV1 values), the exact feature-build options, the
/// streaming working-set batch (when present), and the resolved vector-ta math
/// authority. The frame content is a deterministic function of exactly these
/// inputs, so an equal key is an equal frame; any digest change yields a
/// different key (a miss) and the old entry is eventually evicted.
pub fn feature_frame_cache_input_key(
    symbol: &str,
    bindings: &[SourceArtifactBindingV1],
    options: &FeatureBuildOptions,
    batch: Option<&SweepBatch>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(CACHE_KEY_DOMAIN_V1);
    hasher.update((symbol.len() as u64).to_le_bytes());
    hasher.update(symbol.as_bytes());
    let mut sorted: Vec<&SourceArtifactBindingV1> = bindings.iter().collect();
    sorted.sort_by(|left, right| left.source_node_id().cmp(right.source_node_id()));
    for binding in sorted {
        let node = binding.source_node_id();
        hasher.update((node.len() as u64).to_le_bytes());
        hasher.update(node.as_bytes());
        hasher.update(binding.dataset_identity().canonical_bytes());
        hasher.update(binding.manifest_schema_id().as_bytes());
        hasher.update(binding.manifest_hash());
        hasher.update(binding.generation_id().as_bytes());
        hasher.update(binding.vortex_hash());
        for segment in binding.segments() {
            hasher.update(segment.row_start().to_le_bytes());
            hasher.update(segment.row_end().to_le_bytes());
            hasher.update(segment.timestamp_start_ms().to_le_bytes());
            hasher.update(segment.timestamp_end_ms().to_le_bytes());
        }
    }
    hasher.update(
        serde_json::to_vec(options).expect("feature build options must serialize canonically"),
    );
    if let Some(batch) = batch {
        hasher.update(serde_json::to_vec(batch).expect("sweep batch must serialize canonically"));
    }
    let authority = resolved_canonical_feature_execution_authority_v1();
    hasher.update(format!("{:?}", authority.policy).as_bytes());
    hasher.update(format!("{:?}", authority.selected_lane).as_bytes());
    hasher.update(authority.vector_ta_math_authority.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheEntryMetaV1 {
    schema: String,
    version: u32,
    key: String,
    plan_canonical_bytes: Vec<u8>,
    provenance_canonical_bytes: Vec<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    normalization_fitted_state: Option<SearchNormalizationFittedStateV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    feature_build_options: Option<FeatureBuildOptions>,
    /// Contiguous row-id origin; absent when explicit row ids are stored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    row_id_origin: Option<usize>,
    /// Explicit row ids, u64 little-endian packed; absent for contiguous.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    row_ids_packed: Option<Vec<u8>>,
    timestamps_count: u64,
    column_count: u64,
    payload_bytes: u64,
}

fn meta_path(cfg: &FeatureFrameCacheConfig, key: &str) -> PathBuf {
    cfg.dir.join(key).join("meta.json")
}

fn payload_path(cfg: &FeatureFrameCacheConfig, key: &str) -> PathBuf {
    cfg.dir.join(key).join("payload.bin")
}

fn entry_dir(cfg: &FeatureFrameCacheConfig, key: &str) -> PathBuf {
    cfg.dir.join(key)
}

fn read_entry_meta(cfg: &FeatureFrameCacheConfig, key: &str) -> Result<CacheEntryMetaV1> {
    let meta: CacheEntryMetaV1 = serde_json::from_slice(
        &std::fs::read(meta_path(cfg, key)).context("feature-frame cache entry meta missing")?,
    )
    .context("feature-frame cache entry meta malformed")?;
    ensure!(
        meta.schema == CACHE_META_SCHEMA_V1,
        "unsupported feature-frame cache entry schema"
    );
    ensure!(
        meta.version == CACHE_META_VERSION_V1,
        "unsupported feature-frame cache entry version"
    );
    ensure!(meta.key == key, "feature-frame cache entry key mismatch");
    Ok(meta)
}

fn row_ids_from_meta(meta: &CacheEntryMetaV1, rows: usize) -> Result<FeatureFrameRowIds> {
    match (&meta.row_id_origin, &meta.row_ids_packed) {
        (Some(origin), None) => Ok(FeatureFrameRowIds::Contiguous { origin: *origin }),
        (None, Some(packed)) => {
            ensure!(
                packed.len() == rows * 8,
                "feature-frame cache explicit row ids have {} bytes for {rows} rows",
                packed.len()
            );
            let ids = packed
                .chunks_exact(8)
                .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("8-byte chunk")))
                .collect::<Vec<_>>();
            Ok(FeatureFrameRowIds::Explicit(Arc::new(ids)))
        }
        (None, None) => Ok(FeatureFrameRowIds::Contiguous { origin: 0 }),
        (Some(_), Some(_)) => bail!("feature-frame cache entry carries both row-id encodings"),
    }
}

/// Load a cached feature frame for the key. Ok(None) means no entry exists;
/// a present but malformed/tampered entry is an error, never a silent rebuild.
pub fn feature_frame_cache_load(
    key: &str,
    cfg: &FeatureFrameCacheConfig,
) -> Result<Option<FeatureFrame>> {
    validate_cache_key(key)?;
    if !entry_dir(cfg, key).is_dir() {
        return Ok(None);
    }
    let meta = read_entry_meta(cfg, key)?;
    let plan = FeaturePlanV1::from_canonical_bytes(&meta.plan_canonical_bytes)
        .map_err(|error| anyhow::anyhow!("feature-frame cache plan bytes: {error}"))?;
    let provenance = DatasetFeatureArtifactProvenanceV1::from_canonical_bytes(
        &plan,
        &meta.provenance_canonical_bytes,
    )
    .map_err(|error| anyhow::anyhow!("feature-frame cache provenance bytes: {error}"))?;

    let mut reader = BufReader::new(
        File::open(payload_path(cfg, key)).context("feature-frame cache payload missing")?,
    );
    let mut magic = [0u8; CACHE_PAYLOAD_MAGIC_V1.len()];
    reader
        .read_exact(&mut magic)
        .context("feature-frame cache payload header")?;
    ensure!(
        magic == CACHE_PAYLOAD_MAGIC_V1,
        "feature-frame cache payload magic mismatch"
    );
    fn read_u64(reader: &mut BufReader<File>) -> Result<u64> {
        let mut bytes = [0u8; 8];
        reader.read_exact(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }
    let timestamps_count = read_u64(&mut reader)?;
    let column_count = read_u64(&mut reader)?;
    ensure!(
        timestamps_count == meta.timestamps_count,
        "feature-frame cache payload timestamp count mismatch"
    );
    ensure!(
        column_count == meta.column_count,
        "feature-frame cache payload column count mismatch"
    );
    let rows = usize::try_from(timestamps_count).context("cached frame row count overflow")?;
    let columns = usize::try_from(column_count).context("cached frame column count overflow")?;
    let mut timestamps = Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut bytes = [0u8; 8];
        reader.read_exact(&mut bytes)?;
        timestamps.push(i64::from_le_bytes(bytes));
    }
    let mut frame_columns = Vec::with_capacity(columns);
    for _ in 0..columns {
        let mut name_len = [0u8; 4];
        reader.read_exact(&mut name_len)?;
        let name_len = u32::from_le_bytes(name_len) as usize;
        let mut name_bytes = vec![0u8; name_len];
        reader.read_exact(&mut name_bytes)?;
        let name =
            String::from_utf8(name_bytes).context("feature-frame cache column name utf8")?;
        let mut values = Vec::with_capacity(rows);
        for _ in 0..rows {
            let mut bytes = [0u8; 8];
            reader.read_exact(&mut bytes)?;
            values.push(f64::from_bits(u64::from_le_bytes(bytes)));
        }
        let mut validity = Vec::with_capacity(rows);
        for _ in 0..rows {
            let mut byte = [0u8; 1];
            reader.read_exact(&mut byte)?;
            let code = byte[0];
            validity.push(
                FeatureCellValidity::from_code(code)
                    .with_context(|| format!("feature-frame cache validity code {code}"))?,
            );
        }
        frame_columns.push(
            FeatureColumnF64::new(name, values, validity)
                .context("feature-frame cache column construction")?,
        );
    }
    let row_ids = row_ids_from_meta(&meta, rows)?;
    let frame = FeatureFrame::from_cache_payload(
        timestamps,
        frame_columns,
        Arc::new(plan),
        Arc::new(provenance),
        row_ids,
        meta.normalization_fitted_state.map(Arc::new),
        meta.feature_build_options.map(Arc::new),
    )
    .context("feature-frame cache reconstruction")?;
    Ok(Some(frame))
}

fn pack_row_ids(ids: &[u64]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(ids.len() * 8);
    for id in ids {
        packed.extend_from_slice(&id.to_le_bytes());
    }
    packed
}

/// Persist the frame under the key and enforce the disk budget. Returns the
/// payload bytes written for this entry. The write is staged in a temporary
/// directory and promoted with one rename; an existing entry of the same key
/// (concurrent writer) is left untouched.
pub fn feature_frame_cache_store(
    key: &str,
    frame: &FeatureFrame,
    cfg: &FeatureFrameCacheConfig,
) -> Result<u64> {
    validate_cache_key(key)?;
    std::fs::create_dir_all(&cfg.dir).context("feature-frame cache directory")?;
    let stage = cfg.dir.join(format!(
        ".tmp-{key}-{}-{:?}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    std::fs::create_dir_all(&stage).context("feature-frame cache staging directory")?;
    let cleanup_stage = || {
        let _ = std::fs::remove_dir_all(&stage);
    };
    let write_result = (|| -> Result<u64> {
        let rows = frame.n_samples();
        let columns = frame.n_features();
        let mut writer = BufWriter::new(File::create(stage.join("payload.bin"))?);
        writer.write_all(CACHE_PAYLOAD_MAGIC_V1)?;
        writer.write_all(&(rows as u64).to_le_bytes())?;
        writer.write_all(&(columns as u64).to_le_bytes())?;
        for timestamp in &frame.timestamps {
            writer.write_all(&timestamp.to_le_bytes())?;
        }
        let indices = (0..columns).collect::<Vec<_>>();
        let mut row_ids = None;
        for chunk in indices.chunks(64) {
            let projection = frame.project_columns(chunk, 0..rows)?;
            if row_ids.is_none() {
                row_ids = Some(projection.row_ids.clone());
            }
            for (index, column) in chunk.iter().zip(&projection.columns) {
                ensure!(
                    column.name == frame.names[*index],
                    "feature-frame cache projection renamed column {}",
                    *index
                );
                ensure!(
                    column.values.len() == rows && column.validity.len() == rows,
                    "feature-frame cache projection truncated column {}",
                    *index
                );
                let name_len = u32::try_from(column.name.len())
                    .context("feature-frame cache column name length")?;
                writer.write_all(&name_len.to_le_bytes())?;
                writer.write_all(column.name.as_bytes())?;
                for value in &column.values {
                    writer.write_all(&value.to_bits().to_le_bytes())?;
                }
                for cell in &column.validity {
                    writer.write_all(&[cell.code()])?;
                }
            }
        }
        writer.flush()?;
        let payload_bytes = writer
            .get_ref()
            .metadata()
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        drop(writer);

        let row_ids = row_ids.context("feature-frame cache projection produced no row ids")?;
        let contiguous = row_ids
            .first()
            .map(|origin| {
                row_ids
                    .iter()
                    .enumerate()
                    .all(|(row, id)| *id == *origin + row as u64)
            })
            .unwrap_or(false);
        let (row_id_origin, row_ids_packed) = if contiguous {
            (row_ids.first().map(|origin| *origin as usize), None)
        } else {
            (None, Some(pack_row_ids(&row_ids)))
        };
        let meta = CacheEntryMetaV1 {
            schema: CACHE_META_SCHEMA_V1.to_owned(),
            version: CACHE_META_VERSION_V1,
            key: key.to_owned(),
            plan_canonical_bytes: frame.plan().canonical_bytes().to_vec(),
            provenance_canonical_bytes: frame.provenance().canonical_bytes().to_vec(),
            normalization_fitted_state: frame.normalization_fitted_state().cloned(),
            feature_build_options: frame.feature_build_options().cloned(),
            row_id_origin,
            row_ids_packed,
            timestamps_count: rows as u64,
            column_count: columns as u64,
            payload_bytes,
        };
        std::fs::write(
            stage.join("meta.json"),
            serde_json::to_vec_pretty(&meta).context("feature-frame cache meta serialization")?,
        )?;
        Ok(payload_bytes)
    })();
    let payload_bytes = match write_result {
        Ok(payload_bytes) => payload_bytes,
        Err(error) => {
            cleanup_stage();
            return Err(error);
        }
    };
    let target = entry_dir(cfg, key);
    match std::fs::rename(&stage, &target) {
        Ok(()) => {}
        Err(_) if target.is_dir() => {
            cleanup_stage();
        }
        Err(error) => {
            cleanup_stage();
            return Err(error).context("promote feature-frame cache entry");
        }
    }
    evict_over_budget(cfg, key)?;
    Ok(payload_bytes)
}

fn evict_over_budget(cfg: &FeatureFrameCacheConfig, keep_key: &str) -> Result<()> {
    let mut entries: Vec<(std::time::SystemTime, String, u64)> = Vec::new();
    for dir_entry in std::fs::read_dir(&cfg.dir).context("feature-frame cache scan")? {
        let dir_entry = dir_entry.context("feature-frame cache entry")?;
        let name = dir_entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if name.starts_with('.') {
            let _ = std::fs::remove_dir_all(dir_entry.path());
            continue;
        }
        let Ok(meta) = read_entry_meta(cfg, name) else {
            continue;
        };
        let Ok(modified) = dir_entry.metadata().and_then(|m| m.modified()) else {
            continue;
        };
        entries.push((modified, name.to_owned(), meta.payload_bytes));
    }
    entries.sort_by_key(|(modified, name, _)| (*modified, name.clone()));
    let mut total: u64 = entries.iter().map(|(_, _, bytes)| *bytes).sum();
    let mut evicted_any = false;
    while total > cfg.max_total_bytes {
        let Some(index) = entries.iter().position(|(_, name, _)| name != keep_key) else {
            break;
        };
        let (_, name, bytes) = entries.remove(index);
        let path = entry_dir(cfg, &name);
        if path.is_dir() {
            std::fs::remove_dir_all(&path)
                .with_context(|| format!("evict feature-frame cache entry {name}"))?;
        }
        total = total.saturating_sub(bytes);
        evicted_any = true;
    }
    if evicted_any {
        tracing::info!(
            target: "neoethos_data::feature_frame_cache",
            total_bytes = total,
            max_total_bytes = cfg.max_total_bytes,
            "feature-frame cache evicted entries to honor the disk budget"
        );
    }
    Ok(())
}

/// Per-timeframe input identity digests of a loaded symbol dataset, as exact
/// source bindings. The key builder hashes these, so a cache hit requires the
/// byte-identical dataset generations/manifests.
pub fn symbol_dataset_cache_bindings(
    dataset: &crate::SymbolDataset,
) -> Result<Vec<SourceArtifactBindingV1>> {
    let mut bindings = Vec::with_capacity(dataset.source_artifacts.len());
    let mut timeframes = dataset.source_artifacts.keys().cloned().collect::<Vec<_>>();
    timeframes.sort();
    for timeframe in timeframes {
        let artifact = dataset
            .source_artifacts
            .get(&timeframe)
            .context("symbol dataset artifact missing")?;
        bindings.push(artifact.source_binding(timeframe)?);
    }
    Ok(bindings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{
        ctrader_sample_ohlcv, ctrader_test_feature_frame_from_columns,
        ctrader_test_feature_frame_from_columns_with_options,
        ctrader_test_feature_frame_with_normalization,
    };
    use neoethos_dataset_contracts::{
        BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe,
    };
    use neoethos_feature_contracts::SourceSegmentV1;
    use tempfile::TempDir;

    fn test_binding(node: &str, hash_byte: u8) -> SourceArtifactBindingV1 {
        let identity = CanonicalDatasetIdentity::external(
            "test-source-namespace",
            "EURUSD",
            CanonicalTimeframe::M1,
            BarTimestampConvention::BarOpen,
        )
        .expect("test dataset identity");
        SourceArtifactBindingV1::new(
            node,
            identity,
            "neoethos.test-cache-manifest.v1",
            [hash_byte; 32],
            format!("generation-{hash_byte}"),
            [hash_byte; 32],
            BarTimestampConvention::BarOpen,
            vec![SourceSegmentV1::new(0, 100, 1_600_000_000_000, 1_600_100_000_000)
                .expect("test segment")],
        )
        .expect("test source binding")
    }

    fn test_bindings(hash_byte: u8) -> Vec<SourceArtifactBindingV1> {
        vec![
            test_binding("node:m15", hash_byte),
            test_binding("node:h1", hash_byte),
        ]
    }

    fn test_options() -> FeatureBuildOptions {
        FeatureBuildOptions {
            higher_tfs: vec!["H1".to_owned()],
            ..FeatureBuildOptions::default()
        }
    }

    fn test_config(temp: &TempDir, max_total_bytes: u64) -> FeatureFrameCacheConfig {
        FeatureFrameCacheConfig {
            dir: temp.path().to_path_buf(),
            max_total_bytes,
        }
    }

    fn test_frame() -> FeatureFrame {
        let ohlcv = ctrader_sample_ohlcv();
        let timestamps = ohlcv.timestamp.clone().expect("fixture timestamps");
        let n = timestamps.len();
        let a = FeatureColumnF64::new(
            "a",
            (0..n).map(|row| row as f64 * 0.5).collect(),
            vec![FeatureCellValidity::Valid; n],
        )
        .expect("column a");
        let mut b_values = (0..n).map(|row| row as f64 * -0.25).collect::<Vec<_>>();
        let mut b_validity = vec![FeatureCellValidity::Valid; n];
        b_validity[1] = FeatureCellValidity::Warmup;
        b_values[1] = f64::NAN;
        let b = FeatureColumnF64::new("b", b_values, b_validity).expect("column b");
        let mut c_values = vec![0.0_f64; n];
        c_values[0] = f64::MIN_POSITIVE;
        let mut c_validity = vec![FeatureCellValidity::Valid; n];
        c_validity[2] = FeatureCellValidity::NonFinite;
        c_values[2] = f64::NAN;
        let c = FeatureColumnF64::new("c", c_values, c_validity).expect("column c");
        ctrader_test_feature_frame_from_columns(timestamps, vec![a, b, c]).expect("test frame")
    }

    fn assert_frames_equivalent(left: &FeatureFrame, right: &FeatureFrame) {
        assert_eq!(left.n_samples(), right.n_samples(), "row count");
        assert_eq!(left.n_features(), right.n_features(), "column count");
        assert_eq!(left.timestamps, right.timestamps, "timestamps");
        assert_eq!(left.names, right.names, "names");
        assert_eq!(left.plan_identity(), right.plan_identity(), "plan identity");
        assert_eq!(
            left.provenance_identity(),
            right.provenance_identity(),
            "provenance identity"
        );
        for column in 0..left.n_features() {
            for row in 0..left.n_samples() {
                let left_cell = left.cell(row, column).expect("left cell");
                let right_cell = right.cell(row, column).expect("right cell");
                assert_eq!(
                    left_cell.value.to_bits(),
                    right_cell.value.to_bits(),
                    "value bits row {row} column {column}"
                );
                assert_eq!(
                    left_cell.validity, right_cell.validity,
                    "validity row {row} column {column}"
                );
            }
        }
        assert_eq!(
            serde_json::to_value(left.normalization_fitted_state()).expect("left norm state"),
            serde_json::to_value(right.normalization_fitted_state()).expect("right norm state"),
            "normalization fitted state"
        );
        assert_eq!(
            serde_json::to_value(left.feature_build_options()).expect("left options"),
            serde_json::to_value(right.feature_build_options()).expect("right options"),
            "feature build options"
        );
        let indices = (0..left.n_features()).collect::<Vec<_>>();
        let left_ids = left
            .project_columns(&indices, 0..left.n_samples())
            .expect("left projection")
            .row_ids
            .clone();
        let right_ids = right
            .project_columns(&indices, 0..right.n_samples())
            .expect("right projection")
            .row_ids
            .clone();
        assert_eq!(left_ids, right_ids, "row ids");
    }

    #[test]
    fn key_is_stable_and_sensitive_to_every_digest_change() {
        let options = test_options();
        let base = feature_frame_cache_input_key("EURUSD", &test_bindings(7), &options, None);
        assert_eq!(
            base,
            feature_frame_cache_input_key("EURUSD", &test_bindings(7), &options, None),
            "key must be deterministic"
        );
        let mut reversed = test_bindings(7);
        reversed.reverse();
        assert_eq!(
            base,
            feature_frame_cache_input_key("EURUSD", &reversed, &options, None),
            "key must not depend on binding order"
        );
        assert_ne!(
            base,
            feature_frame_cache_input_key("EURUSD", &test_bindings(8), &options, None),
            "a different dataset digest (manifest/Vortex hash) must change the key"
        );
        let renamed = vec![
            test_binding("node:m15", 7),
            SourceArtifactBindingV1::new(
                "node:h1",
                test_binding("node:h1", 7).dataset_identity().clone(),
                "neoethos.test-cache-manifest.v1",
                *test_binding("node:h1", 7).manifest_hash(),
                "generation-different".to_owned(),
                *test_binding("node:h1", 7).vortex_hash(),
                BarTimestampConvention::BarOpen,
                test_binding("node:h1", 7).segments().to_vec(),
            )
            .expect("renamed generation binding"),
        ];
        assert_ne!(
            base,
            feature_frame_cache_input_key("EURUSD", &renamed, &options, None),
            "a different generation id must change the key"
        );
        let mut other_options = options.clone();
        other_options.quantitative_semantics =
            crate::core::features::QuantitativeFeatureSemantics::RollingValidityV5;
        assert_ne!(
            base,
            feature_frame_cache_input_key("EURUSD", &test_bindings(7), &other_options, None),
            "a different build recipe must change the key"
        );
        let batch = SweepBatch {
            cursor: 0,
            next_cursor: 1,
            base_indicator_ids: vec!["rsi"],
            replace_base_vocabulary: false,
            pairs: vec![crate::core::hpc_ta::SweepPair { id: "rsi", period: 14 }],
            planned_columns: 1,
            space_len: 1,
            exhausted: false,
        };
        assert_ne!(
            base,
            feature_frame_cache_input_key("EURUSD", &test_bindings(7), &options, Some(&batch)),
            "a streaming batch must change the key"
        );
        assert_ne!(
            base,
            feature_frame_cache_input_key("GBPUSD", &test_bindings(7), &options, None),
            "the symbol must change the key"
        );
    }

    #[test]
    fn small_frame_round_trips_bit_exactly() {
        let temp = TempDir::new().expect("temp dir");
        let cfg = test_config(&temp, DEFAULT_FEATURE_FRAME_CACHE_MAX_BYTES);
        let raw = test_frame();
        let normalized =
            ctrader_test_feature_frame_with_normalization(&raw, 0..(raw.n_samples().min(80)), None)
                .expect("normalized fixture");
        let columns = normalized
            .names
            .iter()
            .enumerate()
            .map(|(index, name)| {
                let cells = (0..normalized.n_samples())
                    .map(|row| normalized.cell(row, index).expect("cell"))
                    .collect::<Vec<_>>();
                FeatureColumnF64::new(
                    name.clone(),
                    cells.iter().map(|cell| cell.value).collect(),
                    cells.iter().map(|cell| cell.validity).collect(),
                )
                .expect("column")
            })
            .collect::<Vec<_>>();
        let frame = ctrader_test_feature_frame_from_columns_with_options(
            normalized.timestamps.clone(),
            columns,
            FeatureBuildOptions {
                normalization_training_rows: Some(0..raw.n_samples().min(80)),
                higher_tfs: vec!["H1".to_owned()],
                ..FeatureBuildOptions::default()
            },
        )
        .expect("frame with options");
        let key =
            feature_frame_cache_input_key("EURUSD", &test_bindings(7), &test_options(), None);
        let stored_bytes = feature_frame_cache_store(&key, &frame, &cfg).expect("store");
        assert!(stored_bytes > 0, "payload must be non-empty");
        let loaded = feature_frame_cache_load(&key, &cfg).expect("load").expect("entry");
        assert_frames_equivalent(&frame, &loaded);
    }

    #[test]
    fn load_rejects_corrupt_or_tampered_entries() {
        let temp = TempDir::new().expect("temp dir");
        let cfg = test_config(&temp, DEFAULT_FEATURE_FRAME_CACHE_MAX_BYTES);
        let frame = test_frame();
        let key = feature_frame_cache_input_key("EURUSD", &test_bindings(7), &test_options(), None);
        feature_frame_cache_store(&key, &frame, &cfg).expect("store");

        let missing = feature_frame_cache_input_key("EURUSD", &test_bindings(9), &test_options(), None);
        assert!(
            feature_frame_cache_load(&missing, &cfg).expect("load").is_none(),
            "unknown key is a plain miss"
        );

        let payload = payload_path(&cfg, &key);
        let full = std::fs::read(&payload).expect("payload read");
        std::fs::write(&payload, &full[..full.len() / 2]).expect("truncate payload");
        assert!(
            feature_frame_cache_load(&key, &cfg).is_err(),
            "truncated payload must refuse"
        );

        std::fs::write(&payload, &full).expect("restore payload");
        std::fs::write(meta_path(&cfg, &key), b"{not json").expect("corrupt meta");
        assert!(
            feature_frame_cache_load(&key, &cfg).is_err(),
            "corrupt meta must refuse"
        );

        let meta = CacheEntryMetaV1 {
            schema: CACHE_META_SCHEMA_V1.to_owned(),
            version: CACHE_META_VERSION_V1,
            key: "0".repeat(64),
            plan_canonical_bytes: frame.plan().canonical_bytes().to_vec(),
            provenance_canonical_bytes: frame.provenance().canonical_bytes().to_vec(),
            normalization_fitted_state: None,
            feature_build_options: None,
            row_id_origin: Some(0),
            row_ids_packed: None,
            timestamps_count: frame.n_samples() as u64,
            column_count: frame.n_features() as u64,
            payload_bytes: full.len() as u64,
        };
        std::fs::write(
            meta_path(&cfg, &key),
            serde_json::to_vec_pretty(&meta).expect("meta serialize"),
        )
        .expect("write meta");
        assert!(
            feature_frame_cache_load(&key, &cfg).is_err(),
            "key mismatch must refuse"
        );
    }

    #[test]
    fn store_bounds_disk_and_evicts_oldest_entry() {
        let temp = TempDir::new().expect("temp dir");
        let generous = test_config(&temp, DEFAULT_FEATURE_FRAME_CACHE_MAX_BYTES);
        let frame = test_frame();
        let key_a = feature_frame_cache_input_key("EURUSD", &test_bindings(7), &test_options(), None);
        let key_b = feature_frame_cache_input_key("EURUSD", &test_bindings(8), &test_options(), None);
        let bytes_a = feature_frame_cache_store(&key_a, &frame, &generous).expect("store A");
        std::thread::sleep(std::time::Duration::from_millis(30));
        let bounded = test_config(&temp, bytes_a);
        feature_frame_cache_store(&key_b, &frame, &bounded).expect("store B");
        assert!(
            feature_frame_cache_load(&key_a, &bounded)
                .expect("load A")
                .is_none(),
            "oldest entry must be evicted under the disk budget"
        );
        let loaded_b = feature_frame_cache_load(&key_b, &bounded)
            .expect("load B")
            .expect("entry B present");
        assert_frames_equivalent(&frame, &loaded_b);
    }
}
