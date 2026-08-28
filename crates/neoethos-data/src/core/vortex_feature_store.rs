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

use crate::core::dataset_manifest::sha256_file;
use crate::core::feature_run_lease::FeatureRunLease;
use crate::core::features::{FeatureCellValidity, FeatureColumnF64};
use crate::core::timestamps::validate_canonical_millisecond_timestamps;
use crate::core::vortex_io::{
    read_vortex_file_metadata, read_vortex_projection_range, write_vortex_chunks_fallible,
};

const FILE_NAME: &str = "features.vortex";
const TIMESTAMP_FIELD: &str = "timestamp_ms";
const ROW_ID_FIELD: &str = "__neoethos_row_id";
const VALIDITY_PREFIX: &str = "__neoethos_validity__";
const SCHEMA_DOMAIN: &[u8] = b"neoethos.vortex-feature-store.schema.v1\0";
const DEFAULT_CHUNK_ROWS: usize = 8_192;
const DEFAULT_CACHE_BYTES: usize = 64 * 1024 * 1024;

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
    cache: Mutex<DecodedChunkCache>,
}

impl VortexFeatureStore {
    pub fn create(
        lease: Arc<FeatureRunLease>,
        timestamps: &[i64],
        columns: &[FeatureColumnF64],
        options: VortexFeatureStoreOptions,
    ) -> Result<Arc<Self>> {
        validate_options(options)?;
        validate_source(timestamps, columns)?;
        let path = lease.run_dir().join(FILE_NAME);
        ensure!(
            !path.exists(),
            "refusing to replace an existing immutable Vortex feature store {}",
            path.display()
        );

        let ranges = (0..timestamps.len())
            .step_by(options.chunk_rows)
            .map(|start| start..(start + options.chunk_rows).min(timestamps.len()));
        let chunks = ranges.map(|range| build_chunk(timestamps, columns, range));
        let stats = write_vortex_chunks_fallible(&path, chunks)
            .with_context(|| format!("write Vortex feature store {}", path.display()))?;
        ensure!(
            stats.row_count == timestamps.len() as u64,
            "Vortex feature writer reported {} rows for {} input rows",
            stats.row_count,
            timestamps.len()
        );

        Self::open(
            lease,
            columns.iter().map(|column| column.name.clone()).collect(),
            options.decoded_cache_bytes,
        )
    }

    pub fn open(
        lease: Arc<FeatureRunLease>,
        expected_names: Vec<String>,
        decoded_cache_bytes: usize,
    ) -> Result<Arc<Self>> {
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
        let schema_sha256 = schema_hash(&expected_names);
        Ok(Arc::new(Self {
            lease,
            path,
            names: expected_names,
            n_samples,
            file_sha256,
            schema_sha256,
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
            let mut validity = Vec::with_capacity(reason_codes.len());
            for (row, code) in reason_codes.into_iter().enumerate() {
                let reason = FeatureCellValidity::from_code(code).with_context(|| {
                    format!("feature `{name}` row {row} has unknown validity code {code}")
                })?;
                let physical_valid = value_array
                    .is_valid(row)
                    .with_context(|| format!("inspect feature `{name}` row {row} validity"))?;
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

fn build_chunk(
    timestamps: &[i64],
    columns: &[FeatureColumnF64],
    range: Range<usize>,
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
}
