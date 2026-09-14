//! Run-scoped f64 Vortex feature storage with explicit per-cell validity.
//!
//! This is the sole run-scoped persisted shared-feature store. The public
//! [`crate::core::features::FeatureFrame`] retains f64 values and explicit
//! validity in memory and across Vortex projection/window reads.

use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use vortex_array::arrays::{PrimitiveArray, StructArray};
use vortex_array::dtype::{DType, FieldName, FieldNames, Nullability, PType};
use vortex_array::validity::Validity;
use vortex_array::{ArrayRef, IntoArray, ToCanonical};
use vortex_buffer::Buffer;

use crate::FeatureBuildControl;
use crate::core::dataset_manifest::sha256_file;
use crate::core::feature_run_lease::FeatureRunLease;
use crate::core::features::{FeatureCellValidity, FeatureColumnF64};
use crate::core::timestamps::validate_canonical_millisecond_timestamps;
use crate::core::vortex_io::{
    read_vortex_file_metadata, read_vortex_projection_range, write_vortex_chunks_fallible_guarded,
};

const FILE_NAME: &str = "features.vortex";
const TIMESTAMP_FIELD: &str = "timestamp_ms";
const ROW_ID_FIELD: &str = "__neoethos_row_id";
const VALIDITY_PREFIX: &str = "__neoethos_validity__";
const SCHEMA_DOMAIN: &[u8] = b"neoethos.vortex-feature-store.schema.v1\0";
const IDENTITY_DOMAIN: &[u8] = b"neoethos.vortex-feature-store.identity.v1\0";
const DEFAULT_CHUNK_ROWS: usize = 8_192;
const DEFAULT_CACHE_BYTES: usize = 64 * 1024 * 1024;
const SCRATCH_DISK_RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
// A wide TF may have thousands of physical fields. Bound the INPUT batch as
// well as the writer's buffered output; neither bound removes a feature/row.
const MAX_FEATURE_CHUNK_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VortexFeatureStoreOptions {
    pub chunk_rows: usize,
    pub decoded_cache_bytes: usize,
}

impl Default for VortexFeatureStoreOptions {
    fn default() -> Self {
        Self {
            chunk_rows: DEFAULT_CHUNK_ROWS,
            decoded_cache_bytes: DEFAULT_CACHE_BYTES,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VortexFeatureBatch {
    pub timestamps: Vec<i64>,
    pub row_ids: Vec<u64>,
    pub columns: Vec<FeatureColumnF64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecodedCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub resident_bytes: usize,
    pub entries: usize,
}

#[derive(Debug)]
pub struct VortexFeatureStore {
    lease: Arc<FeatureRunLease>,
    path: PathBuf,
    names: Vec<String>,
    n_samples: usize,
    file_sha256: String,
    schema_sha256: [u8; 32],
    identity_sha256: [u8; 32],
    cache: Mutex<DecodedChunkCache>,
}

/// One logical feature schema backed by independently persisted Vortex shards.
///
/// Each shard owns a contiguous range of the global feature order. Projection
/// partitions arbitrary caller order across those ranges, reads only the
/// requested fields from each shard, and restores the exact caller order in
/// the returned batch. Keeping shards independent lets a multi-timeframe
/// builder persist and release one full aligned block before computing the
/// next one.
#[derive(Debug)]
pub struct VortexFeatureStoreSet {
    stores: Vec<Arc<VortexFeatureStore>>,
    offsets: Vec<usize>,
    names: Vec<String>,
    n_samples: usize,
}

impl VortexFeatureStoreSet {
    pub fn new(stores: Vec<Arc<VortexFeatureStore>>) -> Result<Self> {
        ensure!(!stores.is_empty(), "Vortex feature store set needs shards");
        let n_samples = stores[0].n_samples();
        let identity_sha256 = stores[0].identity_sha256;
        let mut offsets = Vec::with_capacity(stores.len() + 1);
        let mut names = Vec::new();
        offsets.push(0);
        for store in &stores {
            ensure!(
                store.n_samples() == n_samples,
                "Vortex feature shard row count mismatch: {} != {n_samples}",
                store.n_samples()
            );
            ensure!(
                store.identity_sha256 == identity_sha256,
                "Vortex feature shard identity mismatch"
            );
            names.extend(store.names().iter().cloned());
            offsets.push(names.len());
        }
        validate_names(&names)?;
        Ok(Self {
            stores,
            offsets,
            names,
            n_samples,
        })
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub const fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub fn shard_count(&self) -> usize {
        self.stores.len()
    }

    pub(crate) fn matches_row_identity(
        &self,
        timestamps: &[i64],
        row_origin: usize,
    ) -> Result<bool> {
        self.stores[0].matches_row_identity(timestamps, row_origin)
    }

    pub fn project(
        &self,
        column_indices: &[usize],
        row_range: Range<usize>,
    ) -> Result<Arc<VortexFeatureBatch>> {
        ensure!(
            !column_indices.is_empty(),
            "Vortex feature projection must select at least one column"
        );
        validate_range(&row_range, self.n_samples)?;
        let mut unique = HashSet::with_capacity(column_indices.len());
        let mut per_shard = vec![Vec::<(usize, usize)>::new(); self.stores.len()];
        for (output_index, &global_index) in column_indices.iter().enumerate() {
            ensure!(
                global_index < self.names.len(),
                "feature column {global_index} is outside 0..{}",
                self.names.len()
            );
            ensure!(
                unique.insert(global_index),
                "duplicate feature column {global_index}"
            );
            let shard_index = self
                .offsets
                .partition_point(|&offset| offset <= global_index)
                .saturating_sub(1)
                .min(self.stores.len() - 1);
            per_shard[shard_index].push((output_index, global_index - self.offsets[shard_index]));
        }

        let mut timestamps = None;
        let mut row_ids = None;
        let mut columns = vec![None; column_indices.len()];
        for (store, requested) in self.stores.iter().zip(per_shard) {
            if requested.is_empty() {
                continue;
            }
            let local_indices = requested
                .iter()
                .map(|(_, local_index)| *local_index)
                .collect::<Vec<_>>();
            let batch = store.project(&local_indices, row_range.clone())?;
            if let Some(expected) = timestamps.as_ref() {
                ensure!(
                    expected == &batch.timestamps,
                    "Vortex feature shards returned different timestamps"
                );
            } else {
                timestamps = Some(batch.timestamps.clone());
            }
            if let Some(expected) = row_ids.as_ref() {
                ensure!(
                    expected == &batch.row_ids,
                    "Vortex feature shards returned different row IDs"
                );
            } else {
                row_ids = Some(batch.row_ids.clone());
            }
            for ((output_index, _), column) in requested.iter().zip(&batch.columns) {
                columns[*output_index] = Some(column.clone());
            }
        }

        Ok(Arc::new(VortexFeatureBatch {
            timestamps: timestamps.context("Vortex feature store set produced no timestamps")?,
            row_ids: row_ids.context("Vortex feature store set produced no row IDs")?,
            columns: columns
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .context("Vortex feature store set left a projected column unresolved")?,
        }))
    }
}

impl VortexFeatureStore {
    pub fn create(
        lease: Arc<FeatureRunLease>,
        timestamps: &[i64],
        columns: &[FeatureColumnF64],
        options: VortexFeatureStoreOptions,
    ) -> Result<Arc<Self>> {
        Self::create_with_control(
            lease,
            timestamps,
            columns,
            options,
            &FeatureBuildControl::default(),
        )
    }

    pub(crate) fn create_with_control(
        lease: Arc<FeatureRunLease>,
        timestamps: &[i64],
        columns: &[FeatureColumnF64],
        options: VortexFeatureStoreOptions,
        control: &FeatureBuildControl,
    ) -> Result<Arc<Self>> {
        control.checkpoint()?;
        validate_options(options)?;
        validate_source(timestamps, columns)?;
        let path = lease.run_dir().join(FILE_NAME);
        ensure!(
            !path.exists(),
            "refusing to replace an existing immutable Vortex feature store {}",
            path.display()
        );

        let chunk_rows = bounded_feature_chunk_rows(columns.len(), options.chunk_rows)?;
        let ranges = (0..timestamps.len())
            .step_by(chunk_rows)
            .map(|start| start..(start + chunk_rows).min(timestamps.len()));
        let chunks = ranges.map(|range| {
            control.report(
                "vortex_write_rows",
                format!("{} columns", columns.len()),
                range.start,
                timestamps.len(),
            )?;
            build_chunk(timestamps, columns, range, control)
        });
        let disk_allowance = Mutex::new(0_u64);
        let before_write = |bytes: usize| -> std::io::Result<()> {
            control.checkpoint().map_err(std::io::Error::other)?;
            // Live free space on the actual scratch volume, not an assumed
            // compression ratio or the data-root drive. This is a guard, not
            // an OS reservation against unrelated applications' allocations.
            // Refresh after at most 32 MiB of encoded writes, not one OS call
            // per tiny encoding buffer. Each byte still spends this allowance.
            let mut allowance = disk_allowance
                .lock()
                .map_err(|_| std::io::Error::other("Vortex disk allowance poisoned"))?;
            if *allowance == 0 || bytes as u64 > *allowance {
                let available = crate::core::source_snapshot::available_disk_bytes(lease.run_dir())
                    .map_err(std::io::Error::other)?;
                require_scratch_disk_headroom(available, bytes as u64)
                    .map_err(std::io::Error::other)?;
                *allowance = available
                    .saturating_sub(SCRATCH_DISK_RESERVE_BYTES)
                    .min(MAX_FEATURE_CHUNK_BYTES as u64);
            }
            *allowance = allowance.saturating_sub(bytes as u64);
            Ok(())
        };
        let stats = write_vortex_chunks_fallible_guarded(&path, chunks, u64::MAX, before_write)
            .with_context(|| format!("write Vortex feature store {}", path.display()))?;
        control.report(
            "vortex_write_rows",
            format!("{} columns", columns.len()),
            timestamps.len(),
            timestamps.len(),
        )?;
        ensure!(
            stats.row_count == timestamps.len() as u64,
            "Vortex feature writer reported {} rows for {} input rows",
            stats.row_count,
            timestamps.len()
        );

        Self::open_with_control(
            lease,
            columns.iter().map(|column| column.name.clone()).collect(),
            options.decoded_cache_bytes,
            control,
        )
    }

    pub fn open(
        lease: Arc<FeatureRunLease>,
        expected_names: Vec<String>,
        decoded_cache_bytes: usize,
    ) -> Result<Arc<Self>> {
        Self::open_with_control(
            lease,
            expected_names,
            decoded_cache_bytes,
            &FeatureBuildControl::default(),
        )
    }

    fn open_with_control(
        lease: Arc<FeatureRunLease>,
        expected_names: Vec<String>,
        decoded_cache_bytes: usize,
        control: &FeatureBuildControl,
    ) -> Result<Arc<Self>> {
        control.report("vortex_verify", "file hash and row identity", 0, 1)?;
        validate_names(&expected_names)?;
        let path = lease.run_dir().join(FILE_NAME);
        ensure!(
            path.is_file(),
            "completed Vortex feature store is missing: {}",
            path.display()
        );
        let metadata = read_vortex_file_metadata(&path)
            .with_context(|| format!("read Vortex feature metadata {}", path.display()))?;
        validate_physical_schema(metadata.dtype(), &expected_names)?;
        let n_samples = usize::try_from(metadata.row_count())
            .context("Vortex feature row count does not fit usize")?;
        ensure!(n_samples > 0, "Vortex feature store must not be empty");
        let file_sha256 = sha256_file(&path)?;
        control.checkpoint()?;
        let schema_sha256 = schema_hash(&expected_names);
        let identity_sha256 = identity_hash_from_file(&path, n_samples, control)?;
        control.report("vortex_verify", "file hash and row identity", 1, 1)?;
        Ok(Arc::new(Self {
            lease,
            path,
            names: expected_names,
            n_samples,
            file_sha256,
            schema_sha256,
            identity_sha256,
            cache: Mutex::new(DecodedChunkCache::new(decoded_cache_bytes)),
        }))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn lease(&self) -> &Arc<FeatureRunLease> {
        &self.lease
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub const fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub(crate) fn matches_row_identity(
        &self,
        timestamps: &[i64],
        row_origin: usize,
    ) -> Result<bool> {
        Ok(self.identity_sha256 == identity_hash_from_timestamps(timestamps, row_origin)?)
    }

    pub fn cache_stats(&self) -> DecodedCacheStats {
        self.cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .stats()
    }

    pub fn project(
        &self,
        column_indices: &[usize],
        row_range: Range<usize>,
    ) -> Result<Arc<VortexFeatureBatch>> {
        self.validate_projection(column_indices, &row_range)?;
        let key = CacheKey {
            file_sha256: self.file_sha256.clone(),
            schema_sha256: self.schema_sha256,
            columns: column_indices.to_vec(),
            start: row_range.start,
            end: row_range.end,
        };
        {
            let mut cache = self
                .cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(batch) = cache.get(&key) {
                return Ok(batch);
            }
        }

        let decoded = Arc::new(self.read_projection_uncached(column_indices, row_range)?);
        let weight = decoded_weight(&decoded)?;
        self.cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(key, Arc::clone(&decoded), weight);
        Ok(decoded)
    }

    pub fn window(self: &Arc<Self>, row_range: Range<usize>) -> Result<VortexFeatureWindow> {
        validate_range(&row_range, self.n_samples)?;
        Ok(VortexFeatureWindow {
            store: Arc::clone(self),
            absolute_range: row_range,
        })
    }

    fn validate_projection(
        &self,
        column_indices: &[usize],
        row_range: &Range<usize>,
    ) -> Result<()> {
        ensure!(
            !column_indices.is_empty(),
            "Vortex feature projection must select at least one column"
        );
        validate_range(row_range, self.n_samples)?;
        let mut unique = HashSet::with_capacity(column_indices.len());
        for &column in column_indices {
            ensure!(
                column < self.names.len(),
                "feature column {column} is outside 0..{}",
                self.names.len()
            );
            ensure!(unique.insert(column), "duplicate feature column {column}");
        }
        Ok(())
    }

    fn read_projection_uncached(
        &self,
        column_indices: &[usize],
        row_range: Range<usize>,
    ) -> Result<VortexFeatureBatch> {
        let mut physical_fields = Vec::with_capacity(2 + column_indices.len() * 2);
        physical_fields.push(TIMESTAMP_FIELD.to_owned());
        physical_fields.push(ROW_ID_FIELD.to_owned());
        for &column in column_indices {
            physical_fields.push(self.names[column].clone());
            physical_fields.push(validity_field(&self.names[column]));
        }
        let field_refs = physical_fields
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let start = u64::try_from(row_range.start).context("row range start does not fit u64")?;
        let end = u64::try_from(row_range.end).context("row range end does not fit u64")?;
        let array = read_vortex_projection_range(&self.path, &field_refs, start..end)?;
        let structure = array.to_struct();
        let timestamps = extract_non_null::<i64>(
            structure.unmasked_field_by_name(TIMESTAMP_FIELD)?,
            TIMESTAMP_FIELD,
        )?;
        let row_ids = extract_non_null::<u64>(
            structure.unmasked_field_by_name(ROW_ID_FIELD)?,
            ROW_ID_FIELD,
        )?;
        ensure!(
            timestamps.len() == row_range.len() && row_ids.len() == row_range.len(),
            "projected identity column length mismatch"
        );
        for (local_row, &row_id) in row_ids.iter().enumerate() {
            let expected = u64::try_from(row_range.start + local_row)
                .context("expected row id does not fit u64")?;
            ensure!(
                row_id == expected,
                "Vortex row identity mismatch at projected row {local_row}: expected {expected}, got {row_id}"
            );
        }

        let mut columns = Vec::with_capacity(column_indices.len());
        for &column_index in column_indices {
            let name = &self.names[column_index];
            let value_array = structure.unmasked_field_by_name(name)?;
            ensure!(
                matches!(
                    value_array.dtype(),
                    DType::Primitive(PType::F64, Nullability::Nullable)
                ),
                "feature `{name}` must be nullable f64, got {}",
                value_array.dtype()
            );
            let mut values = value_array.to_primitive().as_slice::<f64>().to_vec();
            let reason_name = validity_field(name);
            let reason_codes = extract_non_null::<u8>(
                structure.unmasked_field_by_name(&reason_name)?,
                &reason_name,
            )?;
            ensure!(
                values.len() == reason_codes.len(),
                "feature `{name}` value/reason length mismatch"
            );
            // ArrayRef::is_valid(row) asks a chunked array to reconstruct its
            // complete validity expression on EVERY cell. Decode that bitmap
            // once per projected column, then check each reason against it.
            // This preserves all null/reason checks and exact f64 values.
            let physical_validity = value_array
                .validity_mask()
                .with_context(|| format!("decode feature `{name}` validity bitmap"))?;
            ensure!(
                physical_validity.len() == values.len(),
                "feature `{name}` value/bitmap length mismatch"
            );
            let mut validity = Vec::with_capacity(reason_codes.len());
            for (row, code) in reason_codes.into_iter().enumerate() {
                let reason = FeatureCellValidity::from_code(code).with_context(|| {
                    format!("feature `{name}` row {row} has unknown validity code {code}")
                })?;
                let physical_valid = physical_validity.value(row);
                ensure!(
                    physical_valid == reason.is_valid(),
                    "feature `{name}` row {row} null bitmap disagrees with validity reason {reason:?}"
                );
                if !reason.is_valid() {
                    values[row] = f64::NAN;
                }
                validity.push(reason);
            }
            columns.push(FeatureColumnF64::new(name.clone(), values, validity)?);
        }

        Ok(VortexFeatureBatch {
            timestamps,
            row_ids,
            columns,
        })
    }
}

#[derive(Debug, Clone)]
pub struct VortexFeatureWindow {
    store: Arc<VortexFeatureStore>,
    absolute_range: Range<usize>,
}

impl VortexFeatureWindow {
    pub fn len(&self) -> usize {
        self.absolute_range.len()
    }

    pub fn is_empty(&self) -> bool {
        self.absolute_range.is_empty()
    }

    pub fn names(&self) -> &[String] {
        self.store.names()
    }

    pub fn absolute_range(&self) -> Range<usize> {
        self.absolute_range.clone()
    }

    pub fn project(&self, column_indices: &[usize]) -> Result<Arc<VortexFeatureBatch>> {
        self.store
            .project(column_indices, self.absolute_range.clone())
    }

    pub fn window(&self, relative_range: Range<usize>) -> Result<Self> {
        validate_range(&relative_range, self.len())?;
        let start = self
            .absolute_range
            .start
            .checked_add(relative_range.start)
            .context("nested Vortex feature window start overflow")?;
        let end = self
            .absolute_range
            .start
            .checked_add(relative_range.end)
            .context("nested Vortex feature window end overflow")?;
        Ok(Self {
            store: Arc::clone(&self.store),
            absolute_range: start..end,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    file_sha256: String,
    schema_sha256: [u8; 32],
    columns: Vec<usize>,
    start: usize,
    end: usize,
}

#[derive(Debug)]
struct CacheEntry {
    batch: Arc<VortexFeatureBatch>,
    weight: usize,
}

#[derive(Debug)]
struct DecodedChunkCache {
    capacity: usize,
    resident: usize,
    entries: HashMap<CacheKey, CacheEntry>,
    order: VecDeque<CacheKey>,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl DecodedChunkCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            resident: 0,
            entries: HashMap::new(),
            order: VecDeque::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    fn get(&mut self, key: &CacheKey) -> Option<Arc<VortexFeatureBatch>> {
        let batch = self.entries.get(key).map(|entry| Arc::clone(&entry.batch));
        if batch.is_some() {
            self.hits = self.hits.saturating_add(1);
            if let Some(position) = self.order.iter().position(|candidate| candidate == key) {
                self.order.remove(position);
            }
            self.order.push_back(key.clone());
        } else {
            self.misses = self.misses.saturating_add(1);
        }
        batch
    }

    fn insert(&mut self, key: CacheKey, batch: Arc<VortexFeatureBatch>, weight: usize) {
        if let Some(previous) = self.entries.remove(&key) {
            self.resident = self.resident.saturating_sub(previous.weight);
            self.order.retain(|candidate| candidate != &key);
        }
        if self.capacity == 0 || weight > self.capacity {
            return;
        }
        while self.resident.saturating_add(weight) > self.capacity {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            if let Some(entry) = self.entries.remove(&oldest) {
                self.resident = self.resident.saturating_sub(entry.weight);
                self.evictions = self.evictions.saturating_add(1);
            }
        }
        self.resident = self.resident.saturating_add(weight);
        self.order.push_back(key.clone());
        self.entries.insert(key, CacheEntry { batch, weight });
    }

    fn stats(&self) -> DecodedCacheStats {
        DecodedCacheStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            resident_bytes: self.resident,
            entries: self.entries.len(),
        }
    }
}

fn bounded_feature_chunk_rows(columns: usize, requested_rows: usize) -> Result<usize> {
    let bytes_per_row = columns
        .checked_mul(10)
        .and_then(|bytes| bytes.checked_add(16))
        .context("Vortex feature chunk width overflow")?;
    let rows = requested_rows.min(MAX_FEATURE_CHUNK_BYTES / bytes_per_row);
    ensure!(rows > 0, "Vortex feature schema cannot fit one bounded row");
    Ok(rows)
}

fn require_scratch_disk_headroom(available: u64, next_write: u64) -> Result<()> {
    let required = SCRATCH_DISK_RESERVE_BYTES
        .checked_add(next_write)
        .context("Vortex scratch disk requirement overflow")?;
    ensure!(
        available >= required,
        "Vortex scratch disk exhausted: available_bytes={available}, next_write_bytes={next_write}, reserved_free_bytes={SCRATCH_DISK_RESERVE_BYTES}; stopped before consuming the system reserve; free space or use a bounded streaming search, no input history or indicators were removed"
    );
    Ok(())
}

fn build_chunk(
    timestamps: &[i64],
    columns: &[FeatureColumnF64],
    range: Range<usize>,
    control: &FeatureBuildControl,
) -> Result<ArrayRef> {
    let len = range.len();
    let mut names = Vec::with_capacity(2 + columns.len() * 2);
    let mut arrays = Vec::with_capacity(2 + columns.len() * 2);
    names.push(FieldName::from(TIMESTAMP_FIELD));
    arrays.push(
        PrimitiveArray::new(
            Buffer::copy_from(&timestamps[range.clone()]),
            Validity::NonNullable,
        )
        .into_array(),
    );
    names.push(FieldName::from(ROW_ID_FIELD));
    let row_ids = range
        .clone()
        .map(|row| u64::try_from(row).expect("usize row id must fit u64"))
        .collect::<Vec<_>>();
    arrays.push(PrimitiveArray::new(Buffer::from(row_ids), Validity::NonNullable).into_array());

    for column in columns {
        control.checkpoint()?;
        names.push(FieldName::from(column.name.as_str()));
        let physical_values = range
            .clone()
            .map(|row| {
                if column.validity[row].is_valid() {
                    column.values[row]
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let validity =
            Validity::from_iter(range.clone().map(|row| column.validity[row].is_valid()));
        arrays.push(PrimitiveArray::new(Buffer::from(physical_values), validity).into_array());
        names.push(FieldName::from(validity_field(&column.name)));
        let reasons = range
            .clone()
            .map(|row| column.validity[row].code())
            .collect::<Vec<_>>();
        arrays.push(PrimitiveArray::new(Buffer::from(reasons), Validity::NonNullable).into_array());
    }

    Ok(
        StructArray::try_new(FieldNames::from(names), arrays, len, Validity::NonNullable)?
            .into_array(),
    )
}

fn validate_source(timestamps: &[i64], columns: &[FeatureColumnF64]) -> Result<()> {
    validate_canonical_millisecond_timestamps(timestamps)?;
    ensure!(!columns.is_empty(), "Vortex feature store needs columns");
    validate_names(
        &columns
            .iter()
            .map(|column| column.name.clone())
            .collect::<Vec<_>>(),
    )?;
    for column in columns {
        ensure!(
            column.len() == timestamps.len(),
            "feature `{}` has {} rows but timestamp grid has {}",
            column.name,
            column.len(),
            timestamps.len()
        );
    }
    Ok(())
}

fn validate_names(names: &[String]) -> Result<()> {
    ensure!(!names.is_empty(), "Vortex feature schema must not be empty");
    let mut unique = HashSet::with_capacity(names.len());
    for name in names {
        ensure!(
            !name.is_empty() && name.len() <= 1_024,
            "invalid Vortex feature name length for `{name}`"
        );
        ensure!(
            name != TIMESTAMP_FIELD && name != ROW_ID_FIELD && !name.starts_with(VALIDITY_PREFIX),
            "feature name `{name}` collides with reserved Vortex metadata"
        );
        ensure!(
            unique.insert(name),
            "duplicate Vortex feature name `{name}`"
        );
    }
    Ok(())
}

fn validate_options(options: VortexFeatureStoreOptions) -> Result<()> {
    ensure!(options.chunk_rows > 0, "Vortex chunk rows must be positive");
    Ok(())
}

fn validate_range(range: &Range<usize>, len: usize) -> Result<()> {
    ensure!(
        range.start <= range.end && range.end <= len,
        "Vortex feature row range {}..{} is outside 0..{len}",
        range.start,
        range.end
    );
    Ok(())
}

fn validity_field(name: &str) -> String {
    format!("{VALIDITY_PREFIX}{name}")
}

fn validate_physical_schema(dtype: &DType, names: &[String]) -> Result<()> {
    let structure = dtype
        .as_struct_fields_opt()
        .context("Vortex feature file root must be a struct")?;
    let expected_names = std::iter::once(TIMESTAMP_FIELD.to_owned())
        .chain(std::iter::once(ROW_ID_FIELD.to_owned()))
        .chain(
            names
                .iter()
                .flat_map(|name| [name.clone(), validity_field(name)]),
        )
        .collect::<Vec<_>>();
    let actual_names = structure
        .names()
        .iter()
        .map(|name| name.as_ref().to_owned())
        .collect::<Vec<_>>();
    ensure!(
        actual_names == expected_names,
        "Vortex feature schema names/order mismatch: expected {expected_names:?}, got {actual_names:?}"
    );
    let expected_dtypes = std::iter::once(DType::Primitive(PType::I64, Nullability::NonNullable))
        .chain(std::iter::once(DType::Primitive(
            PType::U64,
            Nullability::NonNullable,
        )))
        .chain(names.iter().flat_map(|_| {
            [
                DType::Primitive(PType::F64, Nullability::Nullable),
                DType::Primitive(PType::U8, Nullability::NonNullable),
            ]
        }))
        .collect::<Vec<_>>();
    let actual_dtypes = structure.fields().collect::<Vec<_>>();
    ensure!(
        actual_dtypes == expected_dtypes,
        "Vortex feature physical dtype mismatch: expected {expected_dtypes:?}, got {actual_dtypes:?}"
    );
    Ok(())
}

fn schema_hash(names: &[String]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(SCHEMA_DOMAIN);
    hasher.update((names.len() as u32).to_be_bytes());
    for name in names {
        hasher.update((name.len() as u32).to_be_bytes());
        hasher.update(name.as_bytes());
    }
    hasher.finalize().into()
}

fn identity_hash_from_file(
    path: &Path,
    n_samples: usize,
    control: &FeatureBuildControl,
) -> Result<[u8; 32]> {
    let mut hasher = new_identity_hasher(n_samples)?;
    let mut previous_timestamp = None;
    for start in (0..n_samples).step_by(DEFAULT_CHUNK_ROWS) {
        control.checkpoint()?;
        let end = (start + DEFAULT_CHUNK_ROWS).min(n_samples);
        let start_u64 = u64::try_from(start).context("identity range start does not fit u64")?;
        let end_u64 = u64::try_from(end).context("identity range end does not fit u64")?;
        let array = read_vortex_projection_range(
            path,
            &[TIMESTAMP_FIELD, ROW_ID_FIELD],
            start_u64..end_u64,
        )?;
        let structure = array.to_struct();
        let timestamps = extract_non_null::<i64>(
            structure.unmasked_field_by_name(TIMESTAMP_FIELD)?,
            TIMESTAMP_FIELD,
        )?;
        let row_ids = extract_non_null::<u64>(
            structure.unmasked_field_by_name(ROW_ID_FIELD)?,
            ROW_ID_FIELD,
        )?;
        ensure!(
            timestamps.len() == end - start && row_ids.len() == end - start,
            "Vortex identity projection length mismatch"
        );
        validate_canonical_millisecond_timestamps(&timestamps)?;
        if let (Some(previous), Some(&first)) = (previous_timestamp, timestamps.first()) {
            ensure!(
                first > previous,
                "Vortex timestamps must remain strictly increasing across identity chunks"
            );
        }
        for (local_row, (&timestamp, &row_id)) in timestamps.iter().zip(&row_ids).enumerate() {
            let expected = u64::try_from(start + local_row)
                .context("expected Vortex identity row id does not fit u64")?;
            ensure!(
                row_id == expected,
                "Vortex row identity mismatch at row {}: expected {expected}, got {row_id}",
                start + local_row
            );
            update_identity_hash(&mut hasher, timestamp, row_id);
        }
        previous_timestamp = timestamps.last().copied();
    }
    Ok(hasher.finalize().into())
}

fn identity_hash_from_timestamps(timestamps: &[i64], row_origin: usize) -> Result<[u8; 32]> {
    validate_canonical_millisecond_timestamps(timestamps)?;
    let mut hasher = new_identity_hasher(timestamps.len())?;
    for (offset, &timestamp) in timestamps.iter().enumerate() {
        let row = row_origin
            .checked_add(offset)
            .context("Vortex identity row id overflow")?;
        let row_id = u64::try_from(row).context("Vortex identity row id does not fit u64")?;
        update_identity_hash(&mut hasher, timestamp, row_id);
    }
    Ok(hasher.finalize().into())
}

fn new_identity_hasher(n_samples: usize) -> Result<Sha256> {
    let mut hasher = Sha256::new();
    hasher.update(IDENTITY_DOMAIN);
    hasher.update(
        u64::try_from(n_samples)
            .context("Vortex identity row count does not fit u64")?
            .to_be_bytes(),
    );
    Ok(hasher)
}

fn update_identity_hash(hasher: &mut Sha256, timestamp: i64, row_id: u64) {
    hasher.update(timestamp.to_be_bytes());
    hasher.update(row_id.to_be_bytes());
}

fn extract_non_null<T: vortex_array::dtype::NativePType>(
    array: &ArrayRef,
    label: &str,
) -> Result<Vec<T>> {
    ensure!(
        array
            .all_valid()
            .with_context(|| format!("inspect {label} validity"))?,
        "Vortex metadata field `{label}` contains nulls"
    );
    Ok(array.to_primitive().as_slice::<T>().to_vec())
}

fn decoded_weight(batch: &VortexFeatureBatch) -> Result<usize> {
    let rows = batch.timestamps.len();
    let identities = rows
        .checked_mul(std::mem::size_of::<i64>() + std::mem::size_of::<u64>())
        .context("decoded identity byte count overflow")?;
    batch.columns.iter().try_fold(identities, |total, column| {
        let values = rows
            .checked_mul(std::mem::size_of::<f64>())
            .context("decoded f64 byte count overflow")?;
        let validity = rows
            .checked_mul(std::mem::size_of::<FeatureCellValidity>())
            .context("decoded validity byte count overflow")?;
        total
            .checked_add(values)
            .and_then(|value| value.checked_add(validity))
            .and_then(|value| value.checked_add(column.name.len()))
            .context("decoded Vortex cache weight overflow")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bulk_chunked_validity_matches_every_scalar_bit_with_bounded_timing_evidence() -> Result<()> {
        use vortex_array::arrays::ChunkedArray;
        const ROWS: usize = 8192;
        const CHUNK_ROWS: usize = 128;
        for mode in ["all_valid", "all_invalid", "mixed"] {
            let expected = (0..ROWS)
                .map(|row| match mode {
                    "all_valid" => true,
                    "all_invalid" => false,
                    _ => row % 11 < 5,
                })
                .collect::<Vec<_>>();
            let chunks = (0..ROWS)
                .step_by(CHUNK_ROWS)
                .map(|start| {
                    PrimitiveArray::new(
                        Buffer::from(vec![1.0_f64; CHUNK_ROWS]),
                        Validity::from_iter(expected[start..start + CHUNK_ROWS].iter().copied()),
                    )
                    .into_array()
                })
                .collect();
            let array =
                ChunkedArray::try_new(chunks, DType::Primitive(PType::F64, Nullability::Nullable))?
                    .into_array();
            let start = std::time::Instant::now();
            let scalar = (0..ROWS)
                .map(|row| array.is_valid(row).map_err(anyhow::Error::new))
                .collect::<Result<Vec<_>>>()?;
            let scalar_micros = start.elapsed().as_micros();
            let start = std::time::Instant::now();
            let mask = array.validity_mask()?;
            let bulk = (0..ROWS).map(|row| mask.value(row)).collect::<Vec<_>>();
            let bulk_micros = start.elapsed().as_micros();
            assert_eq!(scalar, expected);
            assert_eq!(bulk, expected);
            // Record measured work, but don't turn scheduler noise into a
            // flaky timing assertion or claim this is an end-to-end benchmark.
            eprintln!(
                "VORTEX_VALIDITY_COMPARE mode={mode} rows={ROWS} chunks={} scalar_us={scalar_micros} bulk_us={bulk_micros}",
                ROWS / CHUNK_ROWS
            );
        }
        Ok(())
    }

    #[test]
    fn bulk_projection_preserves_all_reason_codes_and_rejects_bitmap_disagreement() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let timestamps = (0..128)
            .map(|row| 1_704_067_200_000_i64 + row * 60_000)
            .collect::<Vec<_>>();
        let reasons = (0..128)
            .map(|row| FeatureCellValidity::from_code((row % 10) as u8).unwrap())
            .collect::<Vec<_>>();
        let column = FeatureColumnF64::new("signal", vec![-0.0; 128], reasons)?;
        let lease = Arc::new(FeatureRunLease::create(temp.path(), "bulk-validity")?);
        let store = VortexFeatureStore::create(
            lease,
            &timestamps,
            &[column.clone()],
            VortexFeatureStoreOptions {
                chunk_rows: 8,
                decoded_cache_bytes: 0,
            },
        )?;
        let projected = store.project(&[0], 3..125)?;
        assert_eq!(projected.columns[0].validity, column.validity[3..125]);
        assert_eq!(
            projected.columns[0]
                .values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            column.values[3..125]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        drop(store);

        let corrupt_lease = Arc::new(FeatureRunLease::create(temp.path(), "bitmap-disagreement")?);
        let names = FieldNames::from(
            [
                TIMESTAMP_FIELD,
                ROW_ID_FIELD,
                "signal",
                "__neoethos_validity__signal",
            ]
            .into_iter()
            .map(FieldName::from)
            .collect::<Vec<_>>(),
        );
        let arrays = vec![
            PrimitiveArray::new(Buffer::from(timestamps.clone()), Validity::NonNullable)
                .into_array(),
            PrimitiveArray::new(
                Buffer::from((0..128_u64).collect::<Vec<_>>()),
                Validity::NonNullable,
            )
            .into_array(),
            PrimitiveArray::new(Buffer::from(vec![1.0_f64; 128]), Validity::AllInvalid)
                .into_array(),
            PrimitiveArray::new(
                Buffer::from(vec![FeatureCellValidity::Valid.code(); 128]),
                Validity::NonNullable,
            )
            .into_array(),
        ];
        let chunk = StructArray::try_new(names, arrays, 128, Validity::NonNullable)?.into_array();
        write_vortex_chunks_fallible_guarded(
            &corrupt_lease.run_dir().join(FILE_NAME),
            std::iter::once(Ok(chunk)),
            u64::MAX,
            |_| Ok(()),
        )?;
        let store = VortexFeatureStore::open(corrupt_lease, vec!["signal".to_owned()], 0)?;
        let error = store
            .project(&[0], 0..128)
            .expect_err("bitmap/reason disagreement must remain rejected");
        assert!(
            error.to_string().contains("null bitmap disagrees"),
            "{error:#}"
        );
        Ok(())
    }

    #[test]
    fn wide_feature_chunks_bound_bytes_without_removing_rows_or_columns() -> Result<()> {
        for columns in [1, 100, 1011, 9099] {
            let rows = bounded_feature_chunk_rows(columns, 8192)?;
            assert!(rows > 0 && rows <= 8192);
            assert!(rows * (columns * 10 + 16) <= MAX_FEATURE_CHUNK_BYTES);
        }
        assert!(bounded_feature_chunk_rows(usize::MAX, 8192).is_err());
        assert!(bounded_feature_chunk_rows(1, 0).is_err());
        Ok(())
    }

    #[test]
    fn scratch_disk_guard_preserves_the_system_reserve_and_names_the_cause() {
        assert!(require_scratch_disk_headroom(SCRATCH_DISK_RESERVE_BYTES + 8, 8).is_ok());
        let error = require_scratch_disk_headroom(SCRATCH_DISK_RESERVE_BYTES + 7, 8).unwrap_err();
        assert!(error.to_string().contains("available_bytes="));
        assert!(error.to_string().contains("reserved_free_bytes="));
        assert!(require_scratch_disk_headroom(u64::MAX, u64::MAX).is_err());
    }

    #[test]
    fn stop_mid_vortex_write_publishes_nothing_and_releases_scratch() -> Result<()> {
        use std::sync::atomic::{AtomicBool, Ordering};
        let temp = tempfile::tempdir()?;
        let lease = Arc::new(FeatureRunLease::create(temp.path(), "cancel-writer")?);
        let run_dir = lease.run_dir().to_path_buf();
        let timestamps = (0..8)
            .map(|i| 1_704_067_200_000 + i * 60_000)
            .collect::<Vec<_>>();
        let column =
            FeatureColumnF64::new("signal", vec![1.; 8], vec![FeatureCellValidity::Valid; 8])?;
        let flag = Arc::new(AtomicBool::new(false));
        let signal = Arc::clone(&flag);
        let control = FeatureBuildControl::new(flag).with_observer(move |event| {
            // The first two real chunks have returned from writer.push.
            if event.stage == "vortex_write_rows" && event.completed == 4 {
                signal.store(true, Ordering::Release);
            }
        });
        let error = VortexFeatureStore::create_with_control(
            Arc::clone(&lease),
            &timestamps,
            &[column],
            VortexFeatureStoreOptions {
                chunk_rows: 2,
                decoded_cache_bytes: 0,
            },
            &control,
        )
        .expect_err("a stopped writer cannot publish a store");
        assert!(crate::FeatureBuildCancelled::matches(&error), "{error:#}");
        assert!(
            std::fs::read_dir(&run_dir)?.next().is_none(),
            "staged file leaked"
        );
        drop(lease);
        assert!(
            !run_dir.exists(),
            "scratch lease retained after workers returned"
        );
        Ok(())
    }

    fn cache_key() -> CacheKey {
        CacheKey {
            file_sha256: "file".to_owned(),
            schema_sha256: [7; 32],
            columns: vec![0],
            start: 0,
            end: 1,
        }
    }

    fn batch() -> Arc<VortexFeatureBatch> {
        Arc::new(VortexFeatureBatch {
            timestamps: vec![1_704_067_200_000],
            row_ids: vec![0],
            columns: vec![
                FeatureColumnF64::new("feature", vec![1.0], vec![FeatureCellValidity::Valid])
                    .expect("valid cache fixture"),
            ],
        })
    }

    #[test]
    fn duplicate_cache_insert_replaces_accounting_instead_of_double_counting() {
        let mut cache = DecodedChunkCache::new(1_024);
        let key = cache_key();
        let batch = batch();
        let weight = decoded_weight(&batch).expect("fixture weight");

        cache.insert(key.clone(), Arc::clone(&batch), weight);
        cache.insert(key, batch, weight);

        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.resident_bytes, weight);
        assert_eq!(cache.order.len(), 1);
    }

    #[test]
    fn source_identity_hash_binds_timestamp_bits_and_row_origin() {
        let timestamps = [1_704_067_200_000_i64, 1_704_067_260_000_i64];
        let shifted = [1_704_067_201_000_i64, 1_704_067_261_000_i64];

        let identity = identity_hash_from_timestamps(&timestamps, 0).expect("valid identity");
        assert_ne!(
            identity,
            identity_hash_from_timestamps(&shifted, 0).expect("valid shifted identity")
        );
        assert_ne!(
            identity,
            identity_hash_from_timestamps(&timestamps, 1).expect("valid offset identity")
        );
    }
}
