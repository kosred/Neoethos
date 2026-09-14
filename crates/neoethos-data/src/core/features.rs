use crate::core::feature_registry::{
    FeatureColumnMetadata, feature_metadata_for_names, validate_feature_names,
};
use anyhow::Result;
use ndarray::Array2;
use neoethos_feature_contracts::{
    DatasetFeatureArtifactProvenanceIdentityV1, DatasetFeatureArtifactProvenanceV1,
    FeaturePlanIdentityV1, FeaturePlanV1,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::ops::Range;
use std::str::FromStr;
use std::sync::Arc;

/// Per-cell validity carried independently from the f64 payload.
///
/// Invalid cells use a canonical NaN payload as a second line of defence, but
/// consumers must gate on this typed reason. In particular, a real numeric
/// `0.0` with [`FeatureCellValidity::Valid`] is not interchangeable with
/// warmup, a missing bar, a zero denominator, or a degenerate feature.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureCellValidity {
    Valid = 0,
    Warmup = 1,
    MissingInput = 2,
    Gap = 3,
    Stale = 4,
    ZeroDenominator = 5,
    Degenerate = 6,
    NonFinite = 7,
    ComputeFailure = 8,
    AlignmentMissing = 9,
}

/// Version 4 supports an explicit per-row close-availability schedule and
/// bounded freshness for calendar timeframes. Fixed timeframes use open + exact
/// period; calendar timeframes become available at the next direct broker
/// bar-open and expire after that just-closed bar's observed open-to-open span.
/// No 24-hour/7-day/30-day duration is invented.
pub const HIGHER_TIMEFRAME_ALIGNMENT_SEMANTIC_VERSION: u32 = 4;

impl FeatureCellValidity {
    #[inline]
    pub const fn is_valid(self) -> bool {
        matches!(self, Self::Valid)
    }

    #[inline]
    pub const fn code(self) -> u8 {
        self as u8
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Valid => "valid",
            Self::Warmup => "warmup",
            Self::MissingInput => "missing_input",
            Self::Gap => "gap",
            Self::Stale => "stale",
            Self::ZeroDenominator => "zero_denominator",
            Self::Degenerate => "degenerate",
            Self::NonFinite => "non_finite",
            Self::ComputeFailure => "compute_failure",
            Self::AlignmentMissing => "alignment_missing",
        }
    }

    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Valid),
            1 => Some(Self::Warmup),
            2 => Some(Self::MissingInput),
            3 => Some(Self::Gap),
            4 => Some(Self::Stale),
            5 => Some(Self::ZeroDenominator),
            6 => Some(Self::Degenerate),
            7 => Some(Self::NonFinite),
            8 => Some(Self::ComputeFailure),
            9 => Some(Self::AlignmentMissing),
            _ => None,
        }
    }
}

/// Internal scalar f64 feature column used while Tasks 5B-9 migrate the public
/// `FeatureFrame`/Vortex/model contracts atomically.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureColumnF64 {
    pub name: String,
    pub values: Vec<f64>,
    pub validity: Vec<FeatureCellValidity>,
}

impl FeatureColumnF64 {
    pub fn new(
        name: impl Into<String>,
        mut values: Vec<f64>,
        validity: Vec<FeatureCellValidity>,
    ) -> Result<Self> {
        let name = name.into();
        anyhow::ensure!(!name.is_empty(), "feature column name must not be empty");
        anyhow::ensure!(
            values.len() == validity.len(),
            "feature column `{name}` has {} values but {} validity entries",
            values.len(),
            validity.len()
        );

        for (row, (value, validity)) in values.iter_mut().zip(&validity).enumerate() {
            if validity.is_valid() {
                anyhow::ensure!(
                    value.is_finite(),
                    "feature column `{name}` row {row} is marked valid with non-finite value {value}"
                );
            } else {
                *value = f64::NAN;
            }
        }

        Ok(Self {
            name,
            values,
            validity,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn invalidate(&mut self, row: usize, reason: FeatureCellValidity) -> Result<()> {
        anyhow::ensure!(
            !reason.is_valid(),
            "invalidate requires an explicit invalidity reason"
        );
        let validity = self
            .validity
            .get_mut(row)
            .ok_or_else(|| anyhow::anyhow!("feature row {row} is out of bounds"))?;
        let value = self
            .values
            .get_mut(row)
            .ok_or_else(|| anyhow::anyhow!("feature row {row} is out of bounds"))?;
        *validity = reason;
        *value = f64::NAN;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FeatureProfile {
    #[default]
    Standard,
    Full,
    HPC,
    Adaptive,
}

impl FromStr for FeatureProfile {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "standard" => Ok(Self::Standard),
            "full" => Ok(Self::Full),
            "hpc" => Ok(Self::HPC),
            "adaptive" => Ok(Self::Adaptive),
            _ => Err(format!("unknown feature profile: {}", s)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureBuildOptions {
    pub profile: FeatureProfile,
    pub prefix_base_features: bool,
    pub higher_tfs: Vec<String>,
    /// Exact in-sample rows used to fit normalization. `None` is valid only
    /// when normalization is disabled; no production path may infer an 80%
    /// split from the full series.
    pub normalization_training_rows: Option<Range<usize>>,
    /// Project away columns with no valid cell in the exact normalization
    /// training range before fitting. This is opt-in because model training
    /// needs a leak-free usable schema, while discovery keeps its separately
    /// versioned feature projection policy.
    #[serde(default)]
    pub drop_columns_without_normalization_training_support: bool,
    /// Exact Classic selection captured by a streaming producer. None retains
    /// the legacy adaptive-prefix recipe; replay never invents a missing batch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub classic_ta_working_set: Option<super::hpc_ta::SweepBatch>,
}

impl Default for FeatureBuildOptions {
    fn default() -> Self {
        Self {
            profile: FeatureProfile::Standard,
            prefix_base_features: false,
            higher_tfs: Vec::new(),
            normalization_training_rows: None,
            drop_columns_without_normalization_training_support: false,
            classic_ta_working_set: None,
        }
    }
}

/// Backing storage for an f64 feature frame. Persisted scratch data has one
/// format only: Vortex; superseded mmap variants are deliberately absent.
#[derive(Debug, Clone)]
pub enum FeatureData {
    InMemory(Vec<FeatureColumnF64>),
    Vortex(Arc<crate::core::vortex_feature_store::VortexFeatureStore>),
    VortexSet(Arc<crate::core::vortex_feature_store::VortexFeatureStoreSet>),
    VortexWindow(crate::core::vortex_feature_store::VortexFeatureWindow),
    /// Lazy row/column view over an existing frame. This keeps one physical
    /// backing (RAM columns or Vortex) while preserving the exact f64 values,
    /// validity reasons, source-generation leases, and artifact provenance.
    View(FeatureFrameView),
}

#[derive(Debug, Clone)]
pub struct FeatureFrameView {
    parent: Arc<FeatureFrame>,
    column_indices: Vec<usize>,
    row_range: Range<usize>,
    normalization: Option<Vec<crate::core::normalization::RobustNormalizationFitF64>>,
    row_indices: Option<Arc<Vec<usize>>>,
}

/// Immutable source-row receipts carried by a feature frame. Ordinary frames
/// and contiguous windows stay allocation-free; arbitrary row selections keep
/// the exact source IDs instead of inventing a new contiguous range.
#[derive(Debug, Clone)]
enum FeatureFrameRowIds {
    Contiguous { origin: usize },
    Explicit(Arc<Vec<u64>>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureCellF64 {
    pub value: f64,
    pub validity: FeatureCellValidity,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureDenseMatrixF64 {
    pub values: Array2<f64>,
    pub validity: Array2<FeatureCellValidity>,
}

fn dense_window_destination_layout(rows: usize, columns: usize) -> Result<(usize, u64)> {
    let cells = rows
        .checked_mul(columns)
        .ok_or_else(|| anyhow::anyhow!("dense feature destination cell count overflowed"))?;
    let value_bytes = cells
        .checked_mul(std::mem::size_of::<f64>())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| anyhow::anyhow!("dense feature destination value capacity overflowed"))?;
    let bytes = cells
        .checked_mul(std::mem::size_of::<FeatureCellValidity>())
        .and_then(|validity_bytes| validity_bytes.checked_add(value_bytes))
        .ok_or_else(|| anyhow::anyhow!("dense feature destination byte count overflowed"))?;
    Ok((cells, u64::try_from(bytes)?))
}

/// RAM-bounded projection schedule for consumers that scan a complete feature
/// frame. Vortex can project many columns in one physical scan, but asking it
/// for the complete cube at once defeats the scratch-store boundary. This plan
/// converts current allocation headroom and the frame's actual row count into
/// both a column batch width and a maximum number of concurrent batches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveFeatureProjectionPlan {
    pub columns_per_batch: usize,
    pub concurrent_batches: usize,
    pub budget_bytes: u64,
    pub estimated_bytes_per_batch: u64,
}

const FEATURE_PROJECTION_MAX_BUDGET_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const FEATURE_PROJECTION_BYTES_PER_ROW_FIXED: u64 = 16; // timestamp + source row id
const FEATURE_PROJECTION_BYTES_PER_CELL: u64 = 16; // f64 + validity + decode headroom
const FEATURE_PROJECTION_BATCH_OVERHEAD_BYTES: u64 = 1024 * 1024;

/// Resolve a fallible projection plan from measured allocation headroom.
///
/// Kept public so Search can use the same calculation for hashing, statistics
/// and prefilter scoring. Production callers should pass
/// [`neoethos_core::allocation_headroom_bytes`]; the explicit argument makes
/// the boundary deterministic and directly testable. A nonempty request is
/// refused if even one complete column cannot fit the estimated budget. Zero
/// headroom does not authorize a fallback allocation or a reduced dataset.
pub fn adaptive_feature_projection_plan_for_available_memory(
    row_count: usize,
    column_count: usize,
    requested_concurrency: usize,
    available_memory_bytes: u64,
) -> Result<AdaptiveFeatureProjectionPlan> {
    if column_count == 0 {
        return Ok(AdaptiveFeatureProjectionPlan {
            columns_per_batch: 0,
            concurrent_batches: 0,
            budget_bytes: 0,
            estimated_bytes_per_batch: 0,
        });
    }

    // Retain two thirds of current headroom for resident OHLCV, search
    // state, allocator fragmentation and the operating system. The cap keeps
    // one scan from becoming a multi-gigabyte latency spike on large hosts.
    let budget_bytes = (available_memory_bytes / 3).min(FEATURE_PROJECTION_MAX_BUDGET_BYTES);
    let rows = u64::try_from(row_count)?;
    let fixed_bytes = rows
        .checked_mul(FEATURE_PROJECTION_BYTES_PER_ROW_FIXED)
        .and_then(|bytes| bytes.checked_add(FEATURE_PROJECTION_BATCH_OVERHEAD_BYTES))
        .ok_or_else(|| anyhow::anyhow!("feature projection fixed-byte estimate overflowed"))?;
    let bytes_per_column = rows
        .checked_mul(FEATURE_PROJECTION_BYTES_PER_CELL)
        .ok_or_else(|| anyhow::anyhow!("feature projection column-byte estimate overflowed"))?
        .max(1);
    let minimum_batch_bytes = fixed_bytes
        .checked_add(bytes_per_column)
        .ok_or_else(|| anyhow::anyhow!("feature projection batch-byte estimate overflowed"))?;
    anyhow::ensure!(
        minimum_batch_bytes <= budget_bytes,
        "feature projection admission refused: one complete {row_count}-row column needs an estimated {minimum_batch_bytes} bytes, but the projection budget is {budget_bytes} bytes from {available_memory_bytes} bytes of measured allocation headroom; no rows or columns were omitted"
    );
    let requested_concurrency = requested_concurrency.max(1).min(column_count);
    let affordable_concurrency =
        usize::try_from(budget_bytes / minimum_batch_bytes).unwrap_or(usize::MAX);
    let concurrent_batches = requested_concurrency.min(affordable_concurrency);
    let per_batch_budget = budget_bytes / concurrent_batches as u64;
    // Spare RAM must not collapse a wide independent workload into one batch
    // and leave the admitted workers idle. Bound batch width by useful work as
    // well as bytes; this changes scheduling, never column coverage/order.
    let columns_per_batch = ((per_batch_budget - fixed_bytes) / bytes_per_column)
        .min(column_count.div_ceil(concurrent_batches) as u64) as usize;
    let estimated_bytes_per_batch = fixed_bytes + bytes_per_column * columns_per_batch as u64;

    Ok(AdaptiveFeatureProjectionPlan {
        columns_per_batch,
        concurrent_batches,
        budget_bytes,
        estimated_bytes_per_batch,
    })
}

/// Resolve a projection plan from live allocation headroom and the actual
/// frame dimensions. Failure propagates before projection or worker dispatch;
/// callers must not replace it with a one-column or fixed-memory fallback.
/// The estimate is a snapshot, not a process-wide byte reservation.
pub fn adaptive_feature_projection_plan(
    frame: &FeatureFrame,
    requested_concurrency: usize,
) -> Result<AdaptiveFeatureProjectionPlan> {
    adaptive_feature_projection_plan_for_available_memory(
        frame.n_samples(),
        frame.n_features(),
        requested_concurrency,
        neoethos_core::allocation_headroom_bytes(),
    )
}

#[derive(Debug, Clone)]
pub struct FeatureFrame {
    pub timestamps: Vec<i64>,
    pub names: Vec<String>,
    pub data: FeatureData,
    plan: Arc<FeaturePlanV1>,
    provenance: Arc<DatasetFeatureArtifactProvenanceV1>,
    source_generation_leases:
        Arc<Vec<Arc<crate::core::dataset_generation_lease::DatasetGenerationLease>>>,
    row_ids: FeatureFrameRowIds,
    normalization_fitted_state:
        Option<Arc<crate::core::normalization::SearchNormalizationFittedStateV1>>,
    feature_build_options: Option<Arc<FeatureBuildOptions>>,
}

/// A sealed column projection of one immutable shared source. Construct once
/// before causal inference; each row window reuses its plan/provenance without
/// retaining another full timestamp vector or materializing feature values.
pub struct BoundFeatureColumnProjection {
    source: Arc<FeatureFrame>,
    column_indices: Vec<usize>,
    plan: Arc<FeaturePlanV1>,
    provenance: Arc<DatasetFeatureArtifactProvenanceV1>,
}

impl BoundFeatureColumnProjection {
    pub fn row_window(&self, rows: Range<usize>) -> Result<FeatureFrame> {
        anyhow::ensure!(
            rows.start < rows.end && rows.end <= self.source.n_samples(),
            "bound column window must be nonempty and within its source"
        );
        let timestamps = self.source.timestamps[rows.clone()].to_vec();
        crate::core::timestamps::validate_canonical_millisecond_timestamps(&timestamps)?;
        let frame = FeatureFrame {
            timestamps,
            names: self.plan.final_outputs().to_vec(),
            data: FeatureData::View(FeatureFrameView {
                parent: Arc::clone(&self.source),
                column_indices: self.column_indices.clone(),
                row_range: rows.clone(),
                normalization: None,
                row_indices: None,
            }),
            plan: Arc::clone(&self.plan),
            provenance: Arc::clone(&self.provenance),
            source_generation_leases: Arc::clone(&self.source.source_generation_leases),
            row_ids: self.source.row_ids_for_window(rows)?,
            normalization_fitted_state: self.source.normalization_fitted_state.clone(),
            feature_build_options: self.source.feature_build_options.clone(),
        };
        frame.validate_backing()?;
        Ok(frame)
    }
}

impl FeatureFrame {
    pub fn bind_column_projection(
        self: &Arc<Self>,
        column_indices: &[usize],
    ) -> Result<BoundFeatureColumnProjection> {
        anyhow::ensure!(
            self.names == self.plan.final_outputs(),
            "bound feature source schema differs from its immutable plan"
        );
        self.validate_backing()?;
        self.validate_projection(column_indices, &(0..self.n_samples()))?;
        let identity = column_indices.iter().copied().eq(0..self.n_features());
        let (plan, provenance) = if identity {
            (Arc::clone(&self.plan), Arc::clone(&self.provenance))
        } else {
            let names = column_indices
                .iter()
                .map(|&index| self.names[index].clone())
                .collect();
            let plan = Arc::new(FeaturePlanV1::new(self.plan.nodes().to_vec(), names)?);
            let provenance = Arc::new(DatasetFeatureArtifactProvenanceV1::new(
                &plan,
                self.provenance.bindings().to_vec(),
            )?);
            (plan, provenance)
        };
        Ok(BoundFeatureColumnProjection {
            source: Arc::clone(self),
            column_indices: column_indices.to_vec(),
            plan,
            provenance,
        })
    }

    pub fn from_columns(
        timestamps: Vec<i64>,
        columns: Vec<FeatureColumnF64>,
        plan: FeaturePlanV1,
        provenance: DatasetFeatureArtifactProvenanceV1,
    ) -> Result<Self> {
        let names = columns
            .iter()
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        Self::build(
            timestamps,
            names,
            FeatureData::InMemory(columns),
            plan,
            provenance,
            Vec::new(),
            0,
        )
    }

    pub(crate) fn from_canonical_columns(
        timestamps: Vec<i64>,
        columns: Vec<FeatureColumnF64>,
        plan: FeaturePlanV1,
        provenance: DatasetFeatureArtifactProvenanceV1,
        source_generation_leases: Vec<
            Arc<crate::core::dataset_generation_lease::DatasetGenerationLease>,
        >,
    ) -> Result<Self> {
        let names = columns
            .iter()
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        anyhow::ensure!(
            !source_generation_leases.is_empty(),
            "canonical feature frame requires at least one pinned source generation"
        );
        Self::build(
            timestamps,
            names,
            FeatureData::InMemory(columns),
            plan,
            provenance,
            source_generation_leases,
            0,
        )
    }

    pub fn from_vortex(
        timestamps: Vec<i64>,
        store: Arc<crate::core::vortex_feature_store::VortexFeatureStore>,
        plan: FeaturePlanV1,
        provenance: DatasetFeatureArtifactProvenanceV1,
    ) -> Result<Self> {
        let names = store.names().to_vec();
        Self::build(
            timestamps,
            names,
            FeatureData::Vortex(store),
            plan,
            provenance,
            Vec::new(),
            0,
        )
    }

    pub(crate) fn from_canonical_vortex_set(
        timestamps: Vec<i64>,
        stores: Arc<crate::core::vortex_feature_store::VortexFeatureStoreSet>,
        plan: FeaturePlanV1,
        provenance: DatasetFeatureArtifactProvenanceV1,
        source_generation_leases: Vec<
            Arc<crate::core::dataset_generation_lease::DatasetGenerationLease>,
        >,
    ) -> Result<Self> {
        anyhow::ensure!(
            !source_generation_leases.is_empty(),
            "canonical Vortex feature frame requires pinned source generations"
        );
        let names = stores.names().to_vec();
        Self::build(
            timestamps,
            names,
            FeatureData::VortexSet(stores),
            plan,
            provenance,
            source_generation_leases,
            0,
        )
    }

    fn build(
        timestamps: Vec<i64>,
        names: Vec<String>,
        data: FeatureData,
        plan: FeaturePlanV1,
        provenance: DatasetFeatureArtifactProvenanceV1,
        source_generation_leases: Vec<
            Arc<crate::core::dataset_generation_lease::DatasetGenerationLease>,
        >,
        row_origin: usize,
    ) -> Result<Self> {
        Self::build_with_authority(
            timestamps,
            names,
            data,
            Arc::new(plan),
            Arc::new(provenance),
            Arc::new(source_generation_leases),
            FeatureFrameRowIds::Contiguous { origin: row_origin },
            None,
            None,
        )
    }

    fn build_with_authority(
        timestamps: Vec<i64>,
        names: Vec<String>,
        data: FeatureData,
        plan: Arc<FeaturePlanV1>,
        provenance: Arc<DatasetFeatureArtifactProvenanceV1>,
        source_generation_leases: Arc<
            Vec<Arc<crate::core::dataset_generation_lease::DatasetGenerationLease>>,
        >,
        row_ids: FeatureFrameRowIds,
        normalization_fitted_state: Option<
            Arc<crate::core::normalization::SearchNormalizationFittedStateV1>,
        >,
        feature_build_options: Option<Arc<FeatureBuildOptions>>,
    ) -> Result<Self> {
        crate::core::timestamps::validate_canonical_millisecond_timestamps(&timestamps)?;
        anyhow::ensure!(!names.is_empty(), "feature frame must contain columns");
        anyhow::ensure!(
            names == plan.final_outputs(),
            "feature frame names/order do not match FeaturePlan final outputs"
        );
        DatasetFeatureArtifactProvenanceV1::from_canonical_bytes(
            &plan,
            provenance.canonical_bytes(),
        )
        .map_err(|error| anyhow::anyhow!("feature provenance does not match plan: {error}"))?;
        let frame = Self {
            timestamps,
            names,
            data,
            plan,
            provenance,
            source_generation_leases,
            row_ids,
            normalization_fitted_state,
            feature_build_options,
        };
        frame.validate_backing()?;
        Ok(frame)
    }

    fn validate_backing(&self) -> Result<()> {
        let rows = self.timestamps.len();
        match &self.row_ids {
            FeatureFrameRowIds::Contiguous { origin } => {
                let end = origin
                    .checked_add(rows)
                    .ok_or_else(|| anyhow::anyhow!("feature row receipt range overflow"))?;
                u64::try_from(*origin)
                    .map_err(|_| anyhow::anyhow!("feature row receipt origin does not fit u64"))?;
                u64::try_from(end.saturating_sub(1)).map_err(|_| {
                    anyhow::anyhow!("feature row receipt endpoint does not fit u64")
                })?;
            }
            FeatureFrameRowIds::Explicit(row_ids) => {
                anyhow::ensure!(
                    row_ids.len() == rows,
                    "feature row receipt count mismatch: {} IDs for {rows} rows",
                    row_ids.len()
                );
                anyhow::ensure!(
                    row_ids.windows(2).all(|pair| pair[0] < pair[1]),
                    "feature row receipts must be strictly increasing"
                );
            }
        }
        match &self.data {
            FeatureData::InMemory(columns) => {
                anyhow::ensure!(
                    columns.len() == self.names.len(),
                    "feature column count mismatch"
                );
                for (index, column) in columns.iter().enumerate() {
                    anyhow::ensure!(
                        column.name == self.names[index],
                        "feature column {index} name/order mismatch"
                    );
                    anyhow::ensure!(
                        column.len() == rows,
                        "feature column `{}` has {} rows; frame has {rows}",
                        column.name,
                        column.len()
                    );
                }
            }
            FeatureData::Vortex(store) => {
                anyhow::ensure!(
                    store.n_samples() == rows,
                    "Vortex feature row count mismatch"
                );
                anyhow::ensure!(
                    store.names() == self.names,
                    "Vortex feature schema mismatch"
                );
                let FeatureFrameRowIds::Contiguous { origin } = &self.row_ids else {
                    anyhow::bail!("Vortex feature backing requires contiguous row identities");
                };
                anyhow::ensure!(
                    store.matches_row_identity(&self.timestamps, *origin)?,
                    "Vortex feature timestamp/row identity mismatch"
                );
            }
            FeatureData::VortexSet(stores) => {
                anyhow::ensure!(
                    stores.n_samples() == rows,
                    "Vortex feature-set row count mismatch"
                );
                anyhow::ensure!(
                    stores.names() == self.names,
                    "Vortex feature-set schema mismatch"
                );
                let FeatureFrameRowIds::Contiguous { origin } = &self.row_ids else {
                    anyhow::bail!("Vortex feature-set backing requires contiguous row identities");
                };
                anyhow::ensure!(
                    stores.matches_row_identity(&self.timestamps, *origin)?,
                    "Vortex feature-set timestamp/row identity mismatch"
                );
            }
            FeatureData::VortexWindow(window) => {
                anyhow::ensure!(
                    window.len() == rows,
                    "Vortex feature-window row count mismatch"
                );
                anyhow::ensure!(
                    window.names() == self.names,
                    "Vortex feature-window schema mismatch"
                );
            }
            FeatureData::View(view) => {
                if let Some(indices) = &view.row_indices {
                    anyhow::ensure!(
                        indices.len() == rows
                            && indices.windows(2).all(|pair| pair[0] < pair[1])
                            && indices
                                .last()
                                .is_some_and(|last| *last < view.parent.n_samples()),
                        "indexed feature view row receipt mismatch"
                    );
                }
                if let Some(fits) = &view.normalization {
                    anyhow::ensure!(
                        fits.len() == self.names.len(),
                        "normalized view fit count mismatch"
                    );
                    anyhow::ensure!(
                        view.parent.normalization_fitted_state().is_none(),
                        "normalized view cannot normalize its parent twice"
                    );
                }
                anyhow::ensure!(
                    view.row_range.start <= view.row_range.end
                        && view.row_range.end <= view.parent.n_samples(),
                    "feature view row range {:?} is outside 0..{}",
                    view.row_range,
                    view.parent.n_samples()
                );
                anyhow::ensure!(
                    view.row_range.end - view.row_range.start == rows,
                    "feature view row count mismatch"
                );
                anyhow::ensure!(
                    view.column_indices.len() == self.names.len(),
                    "feature view column count mismatch"
                );
                let mut unique = HashSet::with_capacity(view.column_indices.len());
                for (logical, &physical) in view.column_indices.iter().enumerate() {
                    anyhow::ensure!(
                        physical < view.parent.n_features(),
                        "feature view column {physical} is out of bounds"
                    );
                    anyhow::ensure!(
                        unique.insert(physical),
                        "duplicate feature view column {physical}"
                    );
                    anyhow::ensure!(
                        self.names[logical] == view.parent.names[physical],
                        "feature view column name/order mismatch"
                    );
                }
            }
        }
        Ok(())
    }

    pub fn column_metadata(&self) -> Result<Vec<FeatureColumnMetadata>> {
        feature_metadata_for_names(&self.names)
    }

    pub fn validate_registry(&self) -> Result<()> {
        validate_feature_names(&self.names)
    }

    pub fn plan_identity(&self) -> FeaturePlanIdentityV1 {
        self.plan.identity()
    }

    pub fn provenance_identity(&self) -> DatasetFeatureArtifactProvenanceIdentityV1 {
        self.provenance.identity()
    }

    pub fn plan(&self) -> &FeaturePlanV1 {
        &self.plan
    }

    pub fn provenance(&self) -> &DatasetFeatureArtifactProvenanceV1 {
        &self.provenance
    }

    /// Original train-only normalization parameters, shared unchanged across
    /// column projections and row windows. Their hash remains bound to the
    /// full normalization node even when the final output selects a subset.
    pub fn normalization_fitted_state(
        &self,
    ) -> Option<&crate::core::normalization::SearchNormalizationFittedStateV1> {
        self.normalization_fitted_state.as_deref()
    }

    pub fn feature_build_options(&self) -> Option<&FeatureBuildOptions> {
        self.feature_build_options.as_deref()
    }

    /// Estimate only from explicit raw training rows, one admitted column
    /// batch at a time. The full backing remains shared for later fold/live
    /// transformations; no inverse of the clipped values is ever attempted.
    pub fn fit_normalization(
        &self,
        training_rows: Range<usize>,
        drop_unsupported: bool,
        control: &crate::FeatureBuildControl,
    ) -> Result<crate::SearchNormalizationFittedStateV1> {
        use rayon::prelude::*;
        anyhow::ensure!(
            self.normalization_fitted_state.is_none(),
            "normalization fit requires raw features"
        );
        anyhow::ensure!(
            training_rows.start < training_rows.end && training_rows.end <= self.n_samples(),
            "normalization training range is outside raw frame"
        );
        let mut names = Vec::new();
        let mut fits = Vec::new();
        let mut start = 0;
        while start < self.n_features() {
            control.checkpoint()?;
            // Fitter scratch plus raw projection and its mutation. Admission
            // uses current headroom and the caller's installed CPU lease.
            let per_worker = (training_rows.len() as u64).saturating_mul(64);
            let budget = crate::higher_timeframe_parallel_budget_bytes(
                false,
                neoethos_core::allocation_headroom_bytes(),
                0,
                0,
            );
            let workers = rayon::current_num_threads()
                .min(self.n_features() - start)
                .min(usize::try_from(budget / per_worker.max(1)).unwrap_or(usize::MAX));
            anyhow::ensure!(
                workers > 0,
                "model normalization training scratch exceeds available RAM; no rows or vocabulary were removed"
            );
            let end = start + workers;
            let batch = (start..end)
                .into_par_iter()
                .map(|index| -> Result<_> {
                    control.checkpoint()?;
                    let projected = self.project_columns(&[index], training_rows.clone())?;
                    let mut column = projected.columns[0].clone();
                    if !column.validity.iter().any(|validity| validity.is_valid())
                        && drop_unsupported
                    {
                        return Ok(None);
                    }
                    let rows = column.len();
                    let mut fit = crate::core::normalization::normalize_search_feature_column_f64(
                        &mut column,
                        0..rows,
                    )?;
                    fit.training_rows = training_rows.clone();
                    control.checkpoint()?;
                    Ok(Some((self.names[index].clone(), fit)))
                })
                .collect::<Result<Vec<_>>>()?;
            for (name, fit) in batch.into_iter().flatten() {
                names.push(name);
                fits.push(fit);
            }
            start = end;
        }
        crate::SearchNormalizationFittedStateV1::new(names, fits)
    }

    /// Lazy immutable transform over an existing raw RAM/Vortex frame. This
    /// MODEL/fold path retains the raw graph and binds its exact fitted state;
    /// the existing Search producer's plan and arithmetic remain unchanged.
    pub fn with_fitted_normalization(
        self: &Arc<Self>,
        state: &crate::SearchNormalizationFittedStateV1,
    ) -> Result<Self> {
        use neoethos_feature_contracts::{FeatureNodeV1, FeatureOperationTagV1, FeatureOutputV1};
        anyhow::ensure!(
            self.normalization_fitted_state.is_none()
                && !self
                    .plan
                    .nodes()
                    .iter()
                    .any(|node| node.operation() == FeatureOperationTagV1::Normalization),
            "frozen normalization requires a raw frame; double normalization is forbidden"
        );
        state.validate()?;
        let raw_indices = self
            .names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.as_str(), index))
            .collect::<std::collections::HashMap<_, _>>();
        let columns = state
            .column_names()
            .iter()
            .map(|name| {
                raw_indices.get(name.as_str()).copied().ok_or_else(|| {
                    anyhow::anyhow!("raw model input is missing fitted column `{name}`")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut nodes = Vec::with_capacity(self.plan.nodes().len() + 1);
        let mut inputs = Vec::new();
        for node in self.plan.nodes() {
            let has_output = node
                .outputs()
                .iter()
                .any(|output| raw_indices.contains_key(output.name()));
            if has_output {
                inputs.push(node.id().to_owned());
                nodes.push(
                    node.with_output_names(
                        node.outputs()
                            .iter()
                            .map(|output| {
                                if raw_indices.contains_key(output.name()) {
                                    format!("model-input:raw:{}", output.name())
                                } else {
                                    output.name().to_owned()
                                }
                            })
                            .collect(),
                    )?,
                );
            } else {
                nodes.push(node.clone());
            }
        }
        let source_hash = crate::semantic_source_hash(&[include_bytes!("normalization.rs")]);
        nodes.push(FeatureNodeV1::transform(
            "normalization:robust-f64",
            FeatureOperationTagV1::Normalization,
            crate::SEARCH_NORMALIZATION_POLICY_VERSION,
            inputs,
            state
                .column_names()
                .iter()
                .map(|name| {
                    FeatureOutputV1::f64(name.clone(), crate::SEARCH_NORMALIZATION_POLICY_VERSION)
                })
                .collect::<std::result::Result<Vec<_>, _>>()?,
            Vec::new(),
            source_hash,
            source_hash,
            Some(state.fitted_state_hash()?),
        )?);
        let plan = FeaturePlanV1::new(nodes, state.column_names().to_vec())?;
        state.validate_plan(&plan)?;
        let provenance =
            DatasetFeatureArtifactProvenanceV1::new(&plan, self.provenance.bindings().to_vec())?;
        let mut options = self.feature_build_options().cloned().unwrap_or_default();
        options.normalization_training_rows = Some(state.training_rows()?);
        Self::build_with_authority(
            self.timestamps.clone(),
            state.column_names().to_vec(),
            FeatureData::View(FeatureFrameView {
                parent: Arc::clone(self),
                column_indices: columns,
                row_range: 0..self.n_samples(),
                normalization: Some(state.fits().to_vec()),
                row_indices: None,
            }),
            Arc::new(plan),
            Arc::new(provenance),
            Arc::clone(&self.source_generation_leases),
            self.row_ids.clone(),
            Some(Arc::new(state.clone())),
            Some(Arc::new(options)),
        )
    }

    /// A whole-frame view with a genuinely shared in-memory backing. Unlike
    /// cloning an owned `InMemory(Vec<...>)`, this allocates no feature values.
    pub fn shared_view(self: &Arc<Self>) -> Result<Self> {
        Self::build_with_authority(
            self.timestamps.clone(),
            self.names.clone(),
            FeatureData::View(FeatureFrameView {
                parent: Arc::clone(self),
                column_indices: (0..self.n_features()).collect(),
                row_range: 0..self.n_samples(),
                normalization: None,
                row_indices: None,
            }),
            Arc::clone(&self.plan),
            Arc::clone(&self.provenance),
            Arc::clone(&self.source_generation_leases),
            self.row_ids.clone(),
            self.normalization_fitted_state.clone(),
            self.feature_build_options.clone(),
        )
    }

    /// Derive a bounded row view without cloning source values/timestamps or
    /// re-hashing its unchanged private plan/provenance. Those immutable Arcs
    /// were validated when the parent was constructed; public schema and the
    /// new row/backing invariants are still checked here.
    pub fn shared_row_window(self: &Arc<Self>, rows: Range<usize>) -> Result<Self> {
        anyhow::ensure!(
            rows.start < rows.end && rows.end <= self.n_samples(),
            "shared feature window must be nonempty and within its source"
        );
        anyhow::ensure!(
            self.names == self.plan.final_outputs(),
            "shared feature source schema differs from its immutable plan"
        );
        let timestamps = self.timestamps[rows.clone()].to_vec();
        crate::core::timestamps::validate_canonical_millisecond_timestamps(&timestamps)?;
        let frame = Self {
            timestamps,
            names: self.names.clone(),
            data: FeatureData::View(FeatureFrameView {
                parent: Arc::clone(self),
                column_indices: (0..self.n_features()).collect(),
                row_range: rows.clone(),
                normalization: None,
                row_indices: None,
            }),
            plan: Arc::clone(&self.plan),
            provenance: Arc::clone(&self.provenance),
            source_generation_leases: Arc::clone(&self.source_generation_leases),
            row_ids: self.row_ids_for_window(rows)?,
            normalization_fitted_state: self.normalization_fitted_state.clone(),
            feature_build_options: self.feature_build_options.clone(),
        };
        frame.validate_backing()?;
        Ok(frame)
    }

    pub(crate) fn with_feature_build_options(mut self, options: FeatureBuildOptions) -> Self {
        self.feature_build_options = Some(Arc::new(options));
        self
    }

    /// Indexed views share raw values and materialize only requested batches.
    pub fn shared_select_rows(self: &Arc<Self>, indices: &[usize]) -> Result<Self> {
        anyhow::ensure!(
            !indices.is_empty()
                && indices.windows(2).all(|pair| pair[0] < pair[1])
                && indices.last().is_some_and(|last| *last < self.n_samples()),
            "shared feature rows must be nonempty, ordered, unique and within the source"
        );
        let row_ids = indices
            .iter()
            .map(|&row| match &self.row_ids {
                FeatureFrameRowIds::Contiguous { origin } => origin
                    .checked_add(row)
                    .and_then(|row| u64::try_from(row).ok())
                    .ok_or_else(|| anyhow::anyhow!("source row identity overflow")),
                FeatureFrameRowIds::Explicit(ids) => Ok(ids[row]),
            })
            .collect::<Result<Vec<_>>>()?;
        Self::build_with_authority(
            indices.iter().map(|&row| self.timestamps[row]).collect(),
            self.names.clone(),
            FeatureData::View(FeatureFrameView {
                parent: Arc::clone(self),
                column_indices: (0..self.n_features()).collect(),
                row_range: 0..indices.len(),
                normalization: None,
                row_indices: Some(Arc::new(indices.to_vec())),
            }),
            Arc::clone(&self.plan),
            Arc::clone(&self.provenance),
            Arc::clone(&self.source_generation_leases),
            FeatureFrameRowIds::Explicit(Arc::new(row_ids)),
            self.normalization_fitted_state.clone(),
            self.feature_build_options.clone(),
        )
    }

    pub(crate) fn with_normalization_fitted_state(
        mut self,
        state: crate::core::normalization::SearchNormalizationFittedStateV1,
    ) -> Result<Self> {
        anyhow::ensure!(
            self.normalization_fitted_state.is_none(),
            "feature frame already has a fitted normalization state"
        );
        state.validate_plan(&self.plan)?;
        self.normalization_fitted_state = Some(Arc::new(state));
        Ok(self)
    }

    pub fn ensure_semantically_compatible(&self, other: &Self) -> Result<()> {
        anyhow::ensure!(
            self.plan_identity() == other.plan_identity(),
            "FeaturePlanIdentity mismatch"
        );
        Ok(())
    }

    pub fn ensure_same_artifact(&self, other: &Self) -> Result<()> {
        self.ensure_semantically_compatible(other)?;
        anyhow::ensure!(
            self.provenance_identity() == other.provenance_identity(),
            "DatasetFeatureArtifactProvenance mismatch"
        );
        Ok(())
    }

    #[inline]
    pub fn n_samples(&self) -> usize {
        self.timestamps.len()
    }

    #[inline]
    pub fn n_features(&self) -> usize {
        self.names.len()
    }

    fn row_ids_for_range(&self, row_range: Range<usize>) -> Result<Vec<u64>> {
        match &self.row_ids {
            FeatureFrameRowIds::Contiguous { origin } => row_range
                .map(|row| {
                    let source_row = origin
                        .checked_add(row)
                        .ok_or_else(|| anyhow::anyhow!("feature row receipt overflow"))?;
                    u64::try_from(source_row)
                        .map_err(|_| anyhow::anyhow!("feature row receipt does not fit u64"))
                })
                .collect(),
            FeatureFrameRowIds::Explicit(row_ids) => Ok(row_ids[row_range].to_vec()),
        }
    }

    fn row_ids_for_window(&self, row_range: Range<usize>) -> Result<FeatureFrameRowIds> {
        match &self.row_ids {
            FeatureFrameRowIds::Contiguous { origin } => Ok(FeatureFrameRowIds::Contiguous {
                origin: origin
                    .checked_add(row_range.start)
                    .ok_or_else(|| anyhow::anyhow!("feature row receipt window overflow"))?,
            }),
            FeatureFrameRowIds::Explicit(row_ids) => Ok(FeatureFrameRowIds::Explicit(Arc::new(
                row_ids[row_range].to_vec(),
            ))),
        }
    }

    pub fn project_columns(
        &self,
        column_indices: &[usize],
        row_range: Range<usize>,
    ) -> Result<Arc<crate::core::vortex_feature_store::VortexFeatureBatch>> {
        self.project_columns_mode(column_indices, row_range, true)
    }

    fn project_columns_mode(
        &self,
        column_indices: &[usize],
        row_range: Range<usize>,
        apply_normalization: bool,
    ) -> Result<Arc<crate::core::vortex_feature_store::VortexFeatureBatch>> {
        self.validate_projection(column_indices, &row_range)?;
        anyhow::ensure!(
            apply_normalization
                || self.normalization_fitted_state.is_none()
                || matches!(self.data, FeatureData::View(_)),
            "raw model input is unavailable for materialized normalized features; inverse normalization is forbidden"
        );
        match &self.data {
            FeatureData::InMemory(columns) => {
                let selected = column_indices
                    .iter()
                    .map(|&column| {
                        FeatureColumnF64::new(
                            columns[column].name.clone(),
                            columns[column].values[row_range.clone()].to_vec(),
                            columns[column].validity[row_range.clone()].to_vec(),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                let row_ids = self.row_ids_for_range(row_range.clone())?;
                Ok(Arc::new(
                    crate::core::vortex_feature_store::VortexFeatureBatch {
                        timestamps: self.timestamps[row_range].to_vec(),
                        row_ids,
                        columns: selected,
                    },
                ))
            }
            FeatureData::Vortex(store) => store.project(column_indices, row_range),
            FeatureData::VortexSet(stores) => stores.project(column_indices, row_range),
            FeatureData::VortexWindow(window) => window.window(row_range)?.project(column_indices),
            FeatureData::View(view) => {
                let physical_columns = column_indices
                    .iter()
                    .map(|&column| view.column_indices[column])
                    .collect::<Vec<_>>();
                let physical_range = (view.row_range.start + row_range.start)
                    ..(view.row_range.start + row_range.end);
                let batch = if let Some(indices) = &view.row_indices {
                    let selected = &indices[row_range];
                    let mut output = crate::core::vortex_feature_store::VortexFeatureBatch {
                        timestamps: Vec::with_capacity(selected.len()),
                        row_ids: Vec::with_capacity(selected.len()),
                        columns: physical_columns
                            .iter()
                            .map(|&column| {
                                FeatureColumnF64::new(
                                    view.parent.names[column].clone(),
                                    Vec::with_capacity(selected.len()),
                                    Vec::with_capacity(selected.len()),
                                )
                            })
                            .collect::<Result<Vec<_>>>()?,
                    };
                    let mut offset = 0;
                    while offset < selected.len() {
                        let start = selected[offset];
                        let mut end = offset + 1;
                        while end < selected.len() && selected[end] == selected[end - 1] + 1 {
                            end += 1;
                        }
                        let source = view.parent.project_columns_mode(
                            &physical_columns,
                            start..(selected[end - 1] + 1),
                            apply_normalization,
                        )?;
                        output.timestamps.extend_from_slice(&source.timestamps);
                        output.row_ids.extend_from_slice(&source.row_ids);
                        for (destination, source) in output.columns.iter_mut().zip(&source.columns)
                        {
                            destination.values.extend_from_slice(&source.values);
                            destination.validity.extend_from_slice(&source.validity);
                        }
                        offset = end;
                    }
                    Arc::new(output)
                } else {
                    view.parent.project_columns_mode(
                        &physical_columns,
                        physical_range,
                        apply_normalization,
                    )?
                };
                if let Some(fits) = &view.normalization
                    && apply_normalization
                {
                    let mut transformed =
                        Arc::try_unwrap(batch).unwrap_or_else(|shared| (*shared).clone());
                    for (&logical, column) in column_indices.iter().zip(&mut transformed.columns) {
                        crate::core::normalization::apply_search_normalization_fit(
                            column,
                            &fits[logical],
                        )?;
                    }
                    Ok(Arc::new(transformed))
                } else {
                    Ok(batch)
                }
            }
        }
    }

    /// Select and reorder logical feature columns without copying their
    /// physical values. The selected output order becomes part of a new
    /// `FeaturePlanIdentity`; concrete dataset provenance stays identical.
    pub fn select_columns(&self, column_indices: &[usize]) -> Result<Self> {
        self.validate_projection(column_indices, &(0..self.n_samples()))?;
        // Adapters receive bounded already-projected views from a bound
        // inference context. Their identity projection must not rebuild the
        // complete graph. Only a View can be cheaply cloned; validate its
        // public schema/backing before reusing the immutable sealed authority.
        if matches!(&self.data, FeatureData::View(_))
            && column_indices.iter().copied().eq(0..self.n_features())
        {
            anyhow::ensure!(
                self.names == self.plan.final_outputs(),
                "identity projection schema differs from its immutable plan"
            );
            crate::core::timestamps::validate_canonical_millisecond_timestamps(&self.timestamps)?;
            self.validate_backing()?;
            return Ok(self.clone());
        }
        let names = column_indices
            .iter()
            .map(|&column| self.names[column].clone())
            .collect::<Vec<_>>();
        let plan = FeaturePlanV1::new(self.plan.nodes().to_vec(), names.clone())
            .map_err(|error| anyhow::anyhow!("invalid projected feature plan: {error}"))?;
        let provenance =
            DatasetFeatureArtifactProvenanceV1::new(&plan, self.provenance.bindings().to_vec())
                .map_err(|error| {
                    anyhow::anyhow!("invalid projected feature provenance: {error}")
                })?;
        Self::build_with_authority(
            self.timestamps.clone(),
            names,
            FeatureData::View(FeatureFrameView {
                parent: Arc::new(self.clone()),
                column_indices: column_indices.to_vec(),
                row_range: 0..self.n_samples(),
                normalization: None,
                row_indices: None,
            }),
            Arc::new(plan),
            Arc::new(provenance),
            Arc::clone(&self.source_generation_leases),
            self.row_ids.clone(),
            self.normalization_fitted_state.clone(),
            self.feature_build_options.clone(),
        )
    }

    /// Select strictly increasing source rows while preserving the frame's
    /// exact schema, timestamps, semantic/provenance identities, generation
    /// leases, validity reasons, and source-row receipt IDs.
    pub fn select_rows(&self, row_indices: &[usize]) -> Result<Self> {
        anyhow::ensure!(
            !row_indices.is_empty(),
            "feature row selection must not be empty"
        );
        for &row in row_indices {
            anyhow::ensure!(
                row < self.n_samples(),
                "feature row {row} is outside 0..{}",
                self.n_samples()
            );
        }
        for pair in row_indices.windows(2) {
            anyhow::ensure!(
                pair[0] < pair[1],
                "feature row selection must be strictly increasing without duplicates"
            );
        }

        let column_indices = (0..self.n_features()).collect::<Vec<_>>();
        let mut selected_timestamps = Vec::with_capacity(row_indices.len());
        let mut selected_row_ids = Vec::with_capacity(row_indices.len());
        let mut selected_columns = self
            .names
            .iter()
            .map(|_| {
                (
                    Vec::with_capacity(row_indices.len()),
                    Vec::with_capacity(row_indices.len()),
                )
            })
            .collect::<Vec<_>>();

        let mut append_run = |start: usize, end: usize| -> Result<()> {
            let batch = self.project_columns(&column_indices, start..end)?;
            anyhow::ensure!(
                batch.timestamps.as_slice() == &self.timestamps[start..end],
                "feature row selection timestamp receipt mismatch for {start}..{end}"
            );
            anyhow::ensure!(
                batch.columns.len() == self.n_features(),
                "feature row selection column receipt mismatch"
            );
            selected_timestamps.extend_from_slice(&batch.timestamps);
            selected_row_ids.extend_from_slice(&batch.row_ids);
            for (column, (values, validity)) in
                batch.columns.iter().zip(selected_columns.iter_mut())
            {
                values.extend_from_slice(&column.values);
                validity.extend_from_slice(&column.validity);
            }
            Ok(())
        };

        let mut run_start = row_indices[0];
        let mut previous = row_indices[0];
        for &row in &row_indices[1..] {
            if previous.checked_add(1) != Some(row) {
                append_run(run_start, previous + 1)?;
                run_start = row;
            }
            previous = row;
        }
        append_run(run_start, previous + 1)?;

        let columns = self
            .names
            .iter()
            .cloned()
            .zip(selected_columns)
            .map(|(name, (values, validity))| FeatureColumnF64::new(name, values, validity))
            .collect::<Result<Vec<_>>>()?;
        Self::build_with_authority(
            selected_timestamps,
            self.names.clone(),
            FeatureData::InMemory(columns),
            Arc::clone(&self.plan),
            Arc::clone(&self.provenance),
            Arc::clone(&self.source_generation_leases),
            FeatureFrameRowIds::Explicit(Arc::new(selected_row_ids)),
            self.normalization_fitted_state.clone(),
            self.feature_build_options.clone(),
        )
    }

    fn validate_projection(&self, columns: &[usize], range: &Range<usize>) -> Result<()> {
        anyhow::ensure!(!columns.is_empty(), "feature projection needs columns");
        anyhow::ensure!(
            range.start <= range.end && range.end <= self.n_samples(),
            "feature row range {:?} is outside 0..{}",
            range,
            self.n_samples()
        );
        let mut unique = HashSet::with_capacity(columns.len());
        for &column in columns {
            anyhow::ensure!(
                column < self.n_features(),
                "feature column {column} is out of bounds"
            );
            anyhow::ensure!(unique.insert(column), "duplicate feature column {column}");
        }
        Ok(())
    }

    pub fn feature_column(&self, index: usize) -> Result<Arc<FeatureColumnF64>> {
        let batch = self.project_columns(&[index], 0..self.n_samples())?;
        Ok(Arc::new(batch.columns[0].clone()))
    }

    /// Price-only adapters get the original raw column through retained
    /// model view lineage, preserving every row projection. No inverse of
    /// clipped/materialized normalization is available or attempted.
    pub fn raw_model_column(&self, name: &str) -> Result<Arc<FeatureColumnF64>> {
        let index = self
            .names
            .iter()
            .position(|candidate| candidate == name)
            .ok_or_else(|| anyhow::anyhow!("raw model feature `{name}` is absent"))?;
        let batch = self.project_columns_mode(&[index], 0..self.n_samples(), false)?;
        Ok(Arc::new(batch.columns[0].clone()))
    }

    /// Resolve a base feature only from the recorded prefix recipe and exact
    /// source identity. No lowest-timeframe or name-substring guessing.
    pub fn model_base_feature_name(&self, unprefixed: &str) -> Result<String> {
        let name = if let Some(options) = self
            .feature_build_options()
            .filter(|options| options.prefix_base_features)
        {
            let higher = options
                .higher_tfs
                .iter()
                .map(|tf| tf.parse::<crate::CanonicalTimeframe>())
                .collect::<std::result::Result<std::collections::BTreeSet<_>, _>>()?;
            let bases = self
                .provenance
                .bindings()
                .iter()
                .map(|binding| binding.dataset_identity())
                .filter(|identity| !higher.contains(&identity.timeframe()))
                .map(|identity| (identity.to_path_component(), identity.timeframe()))
                .collect::<std::collections::BTreeMap<_, _>>();
            anyhow::ensure!(
                bases.len() == 1,
                "model base-column alias requires one exact recipe-bound source identity"
            );
            format!(
                "{}_{}",
                bases.values().next().expect("one base").as_str(),
                unprefixed
            )
        } else {
            unprefixed.to_owned()
        };
        anyhow::ensure!(
            self.names.contains(&name),
            "model base feature `{name}` is absent"
        );
        Ok(name)
    }

    pub fn cell(&self, sample: usize, feature: usize) -> Result<FeatureCellF64> {
        let batch = self.project_columns(&[feature], sample..sample.saturating_add(1))?;
        Ok(FeatureCellF64 {
            value: batch.columns[0].values[0],
            validity: batch.columns[0].validity[0],
        })
    }

    pub fn row_is_eligible(&self, sample: usize, required_features: &[usize]) -> Result<bool> {
        let batch = self.project_columns(required_features, sample..sample.saturating_add(1))?;
        Ok(batch
            .columns
            .iter()
            .all(|column| column.validity[0].is_valid()))
    }

    pub fn dense_window(&self, start: usize, end: usize) -> Result<FeatureDenseMatrixF64> {
        self.dense_window_with_headroom(start, end, neoethos_core::allocation_headroom_bytes)
    }

    fn dense_window_with_headroom(
        &self,
        start: usize,
        end: usize,
        mut allocation_headroom: impl FnMut() -> u64,
    ) -> Result<FeatureDenseMatrixF64> {
        use rayon::prelude::*;

        let columns = (0..self.n_features()).collect::<Vec<_>>();
        self.validate_projection(&columns, &(start..end))?;
        let rows = end - start;
        let (cells, destination_bytes) = dense_window_destination_layout(rows, columns.len())?;
        if rows == 0 {
            return Ok(FeatureDenseMatrixF64 {
                values: Array2::from_shape_vec((0, columns.len()), Vec::new())?,
                validity: Array2::from_shape_vec((0, columns.len()), Vec::new())?,
            });
        }

        // Reserve the unavoidable destination once before admitting transient
        // projections. This is a headroom estimate plus fallible reservations,
        // not a process-wide reservation or a guarantee of physical RAM.
        let available = allocation_headroom();
        let projection_headroom = available.checked_sub(destination_bytes).ok_or_else(|| {
            anyhow::anyhow!(
                "dense feature destination admission refused: {rows} rows by {} columns need {destination_bytes} bytes, exceeding {available} bytes of measured allocation headroom; no rows or columns were omitted",
                columns.len()
            )
        })?;
        let workers = rayon::current_num_threads();
        let mut plan = adaptive_feature_projection_plan_for_available_memory(
            rows,
            columns.len(),
            workers,
            projection_headroom,
        )?;
        let mut values = Vec::new();
        values.try_reserve_exact(cells)?;
        let mut validity = Vec::new();
        validity.try_reserve_exact(cells)?;
        values.resize(cells, f64::NAN);
        validity.resize(cells, FeatureCellValidity::AlignmentMissing);

        let mut first_column = 0;
        while first_column < columns.len() {
            let wave_end = first_column
                .saturating_add(
                    plan.columns_per_batch
                        .saturating_mul(plan.concurrent_batches),
                )
                .min(columns.len());
            let projected = columns[first_column..wave_end]
                .par_chunks(plan.columns_per_batch)
                .enumerate()
                .map(|(batch_index, selected)| {
                    self.project_columns(selected, start..end)
                        .map(|batch| (first_column + batch_index * plan.columns_per_batch, batch))
                })
                .collect::<Result<Vec<_>>>()?;
            values
                .par_chunks_mut(columns.len())
                .zip(validity.par_chunks_mut(columns.len()))
                .enumerate()
                .for_each(|(row, (values, validity))| {
                    for (offset, batch) in &projected {
                        for (local, column) in batch.columns.iter().enumerate() {
                            values[offset + local] = column.values[row];
                            validity[offset + local] = column.validity[row];
                        }
                    }
                });
            drop(projected);
            first_column = wave_end;
            if first_column < columns.len() {
                // The destination is now resident: do not subtract it again.
                // Fresh headroom also observes retained decoded-cache growth;
                // dropping a projection wave does not empty backing caches.
                plan = adaptive_feature_projection_plan_for_available_memory(
                    rows,
                    columns.len() - first_column,
                    workers,
                    allocation_headroom(),
                )?;
            }
        }
        Ok(FeatureDenseMatrixF64 {
            values: Array2::from_shape_vec((rows, columns.len()), values)?,
            validity: Array2::from_shape_vec((rows, columns.len()), validity)?,
        })
    }

    pub fn row_slice(&self, start: usize, end: usize) -> Result<Self> {
        let start = start.min(self.n_samples());
        let end = end.min(self.n_samples()).max(start);
        let batch =
            self.project_columns(&(0..self.n_features()).collect::<Vec<_>>(), start..end)?;
        Self::build_with_authority(
            batch.timestamps.clone(),
            self.names.clone(),
            FeatureData::InMemory(batch.columns.clone()),
            Arc::clone(&self.plan),
            Arc::clone(&self.provenance),
            Arc::clone(&self.source_generation_leases),
            FeatureFrameRowIds::Explicit(Arc::new(batch.row_ids.clone())),
            self.normalization_fitted_state.clone(),
            self.feature_build_options.clone(),
        )
    }

    pub fn row_window(&self, start: usize, end: usize) -> Result<Self> {
        let start = start.min(self.n_samples());
        let end = end.min(self.n_samples()).max(start);
        Self::build_with_authority(
            self.timestamps[start..end].to_vec(),
            self.names.clone(),
            FeatureData::View(FeatureFrameView {
                parent: Arc::new(self.clone()),
                column_indices: (0..self.n_features()).collect(),
                row_range: start..end,
                normalization: None,
                row_indices: None,
            }),
            Arc::clone(&self.plan),
            Arc::clone(&self.provenance),
            Arc::clone(&self.source_generation_leases),
            self.row_ids_for_window(start..end)?,
            self.normalization_fitted_state.clone(),
            self.feature_build_options.clone(),
        )
    }

    #[inline]
    pub fn n_values(&self) -> usize {
        self.n_samples() * self.n_features()
    }

    pub fn to_dense_samples_major(&self) -> Result<FeatureDenseMatrixF64> {
        self.dense_window(0, self.n_samples())
    }
}

/// Causally align typed f64 feature columns onto a canonical millisecond base
/// grid while retaining the exact reason a cell is unavailable.
///
/// `availability_lag_ms` is normally one complete higher-timeframe period for
/// open-stamped bars. A feature row cannot be observed before
/// `feature_timestamp + availability_lag_ms`. Forward-filled observations are
/// invalidated as [`FeatureCellValidity::Stale`] after `max_age_ms`; rows that
/// have not yet become available are [`FeatureCellValidity::AlignmentMissing`].
pub fn align_feature_columns_by_ms(
    base_ms: &[i64],
    feature_ms: &[i64],
    feature_columns: &[FeatureColumnF64],
    forward_fill: bool,
    max_age_ms: Option<i64>,
    availability_lag_ms: i64,
) -> Result<Vec<FeatureColumnF64>> {
    anyhow::ensure!(
        availability_lag_ms >= 0,
        "feature availability lag must be non-negative"
    );
    let available_at = feature_ms
        .iter()
        .enumerate()
        .map(|(row, timestamp)| {
            timestamp
                .checked_add(availability_lag_ms)
                .map(Some)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "feature availability timestamp overflow at row {row}: {timestamp} + {availability_lag_ms}"
                    )
                })
        })
        .collect::<Result<Vec<_>>>()?;
    align_feature_columns_at_explicit_availability_ms(
        base_ms,
        feature_ms,
        &available_at,
        feature_columns,
        forward_fill,
        max_age_ms,
    )
}

/// Causally align a direct calendar-timeframe source without inventing a fixed
/// duration.
///
/// Source row `N` becomes observable only at the actually observed open of row
/// `N + 1`. Its freshness window is then bounded by the observed
/// `open[N]..open[N + 1]` span. The final source row has no evidenced close and
/// is therefore never exposed. This deliberately fails stale rather than
/// forward-filling the last known D1/W1/MN1 feature forever when newer direct
/// broker rows are missing.
pub fn align_calendar_feature_columns_by_observed_next_open_ms(
    base_ms: &[i64],
    feature_open_ms: &[i64],
    feature_columns: &[FeatureColumnF64],
    forward_fill: bool,
) -> Result<Vec<FeatureColumnF64>> {
    let mut available_at_ms = feature_open_ms
        .iter()
        .skip(1)
        .copied()
        .map(Some)
        .collect::<Vec<_>>();
    if !feature_open_ms.is_empty() {
        available_at_ms.push(None);
    }
    align_feature_columns_at_explicit_availability_ms_with_freshness(
        base_ms,
        feature_open_ms,
        &available_at_ms,
        feature_columns,
        forward_fill,
        AlignmentFreshness::ObservedSourceSpan,
    )
}

/// Align source rows using exact per-row availability timestamps.
///
/// `None` means that a row has no evidenced close yet. Once a `None` appears,
/// every later row must also be unavailable. This is how a direct calendar
/// series keeps its final (possibly still-forming) bar out of backtests without
/// guessing a fixed duration.
pub fn align_feature_columns_at_explicit_availability_ms(
    base_ms: &[i64],
    feature_open_ms: &[i64],
    available_at_ms: &[Option<i64>],
    feature_columns: &[FeatureColumnF64],
    forward_fill: bool,
    max_age_ms: Option<i64>,
) -> Result<Vec<FeatureColumnF64>> {
    align_feature_columns_at_explicit_availability_ms_with_freshness(
        base_ms,
        feature_open_ms,
        available_at_ms,
        feature_columns,
        forward_fill,
        AlignmentFreshness::Fixed(max_age_ms),
    )
}

#[derive(Clone, Copy)]
enum AlignmentFreshness {
    Fixed(Option<i64>),
    ObservedSourceSpan,
}

fn align_feature_columns_at_explicit_availability_ms_with_freshness(
    base_ms: &[i64],
    feature_open_ms: &[i64],
    available_at_ms: &[Option<i64>],
    feature_columns: &[FeatureColumnF64],
    forward_fill: bool,
    freshness: AlignmentFreshness,
) -> Result<Vec<FeatureColumnF64>> {
    use crate::core::timestamps::validate_canonical_millisecond_timestamps;

    if let AlignmentFreshness::Fixed(Some(max_age_ms)) = freshness {
        anyhow::ensure!(max_age_ms >= 0, "feature max age must be non-negative");
    }
    validate_canonical_millisecond_timestamps(base_ms)
        .map_err(|error| anyhow::anyhow!("invalid base alignment timestamps: {error}"))?;
    validate_canonical_millisecond_timestamps(feature_open_ms)
        .map_err(|error| anyhow::anyhow!("invalid feature alignment timestamps: {error}"))?;
    anyhow::ensure!(
        available_at_ms.len() == feature_open_ms.len(),
        "feature availability schedule has {} rows but the source timestamp grid has {}",
        available_at_ms.len(),
        feature_open_ms.len()
    );

    let mut previous_available = None;
    let mut unavailable_tail_started = false;
    for (row, (&open_ms, available_ms)) in feature_open_ms.iter().zip(available_at_ms).enumerate() {
        match available_ms {
            Some(available_ms) => {
                anyhow::ensure!(
                    !unavailable_tail_started,
                    "feature availability row {row} resumes after an unevidenced row"
                );
                anyhow::ensure!(
                    *available_ms >= open_ms,
                    "feature row {row} is available before its bar-open timestamp"
                );
                if let Some(previous) = previous_available {
                    anyhow::ensure!(
                        *available_ms > previous,
                        "feature availability timestamps are duplicate or descending at row {row}"
                    );
                }
                previous_available = Some(*available_ms);
            }
            None => unavailable_tail_started = true,
        }
    }

    let mut names = HashSet::with_capacity(feature_columns.len());
    for column in feature_columns {
        anyhow::ensure!(
            column.len() == feature_open_ms.len(),
            "feature column `{}` has {} rows but the timestamp grid has {}",
            column.name,
            column.len(),
            feature_open_ms.len()
        );
        anyhow::ensure!(
            names.insert(column.name.as_str()),
            "duplicate aligned feature column `{}`",
            column.name
        );
    }

    let mut output_values = feature_columns
        .iter()
        .map(|_| vec![f64::NAN; base_ms.len()])
        .collect::<Vec<_>>();
    let mut output_validity = feature_columns
        .iter()
        .map(|_| vec![FeatureCellValidity::AlignmentMissing; base_ms.len()])
        .collect::<Vec<_>>();

    let mut feature_cursor = 0usize;
    let mut last_available_row = None;
    for (base_row, &base_timestamp) in base_ms.iter().enumerate() {
        while feature_cursor < available_at_ms.len() {
            match available_at_ms[feature_cursor] {
                Some(available_ms) if available_ms <= base_timestamp => {
                    last_available_row = Some(feature_cursor);
                    feature_cursor += 1;
                }
                Some(_) | None => break,
            }
        }
        let Some(feature_row) = last_available_row else {
            continue;
        };
        let available_ms = available_at_ms[feature_row]
            .expect("last available row always has an evidenced timestamp");
        let age = base_timestamp.checked_sub(available_ms).ok_or_else(|| {
            anyhow::anyhow!(
                "feature age overflow at base row {base_row}: {base_timestamp} - {available_ms}"
            )
        })?;
        if age != 0 && !forward_fill {
            continue;
        }
        let max_age_ms = match freshness {
            AlignmentFreshness::Fixed(max_age_ms) => max_age_ms,
            AlignmentFreshness::ObservedSourceSpan => Some(
                available_ms
                    .checked_sub(feature_open_ms[feature_row])
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "observed calendar span overflow at source row {feature_row}: \
                             {available_ms} - {}",
                            feature_open_ms[feature_row]
                        )
                    })?,
            ),
        };
        if max_age_ms.is_some_and(|max_age| age > max_age) {
            for validity in &mut output_validity {
                validity[base_row] = FeatureCellValidity::Stale;
            }
            continue;
        }
        for (column_index, source) in feature_columns.iter().enumerate() {
            let reason = source.validity[feature_row];
            output_validity[column_index][base_row] = reason;
            if reason.is_valid() {
                output_values[column_index][base_row] = source.values[feature_row];
            }
        }
    }

    feature_columns
        .iter()
        .zip(output_values.into_iter().zip(output_validity))
        .map(|(source, (values, validity))| {
            FeatureColumnF64::new(source.name.clone(), values, validity)
        })
        .collect()
}

#[cfg(test)]
mod fitted_model_view_tests {
    use super::*;

    fn dense_test_frame() -> Arc<FeatureFrame> {
        Arc::new(
            crate::test_fixtures::ctrader_test_feature_frame_from_columns(
                crate::test_fixtures::canonical_test_timestamps(17),
                (0..7)
                    .map(|column| {
                        FeatureColumnF64::new(
                            format!("dense_{column}"),
                            (0..17)
                                .map(|row| match row {
                                    0 => -0.0,
                                    1 => 0.0,
                                    _ => (row * 7 + column) as f64 / 13.0,
                                })
                                .collect(),
                            (0..17)
                                .map(|row| {
                                    if row < 10 {
                                        FeatureCellValidity::Valid
                                    } else {
                                        FeatureCellValidity::from_code(
                                            (1 + (row + column) % 9) as u8,
                                        )
                                        .unwrap()
                                    }
                                })
                                .collect(),
                        )
                    })
                    .collect::<Result<Vec<_>>>()
                    .unwrap(),
            )
            .unwrap(),
        )
    }

    fn assert_dense_waves_match_projection(frame: &FeatureFrame, start: usize, end: usize) {
        let columns = (0..frame.n_features()).collect::<Vec<_>>();
        let expected = frame.project_columns(&columns, start..end).unwrap();
        let rows = end - start;
        let (_, destination_bytes) =
            dense_window_destination_layout(rows, frame.n_features()).unwrap();
        let minimum = FEATURE_PROJECTION_BATCH_OVERHEAD_BYTES
            + rows as u64
                * (FEATURE_PROJECTION_BYTES_PER_ROW_FIXED + FEATURE_PROJECTION_BYTES_PER_CELL);
        // Exactly two one-column projections per wave; seven columns force an
        // uneven final wave. Later snapshots already exclude the destination.
        let mut snapshots = 0;
        let dense = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| {
                frame.dense_window_with_headroom(start, end, || {
                    snapshots += 1;
                    6 * minimum + if snapshots == 1 { destination_bytes } else { 0 }
                })
            })
            .unwrap();
        assert_eq!(snapshots, frame.n_features().div_ceil(2));
        assert_eq!(dense.values.dim(), (rows, frame.n_features()));
        assert!(dense.values.is_standard_layout());
        assert!(dense.validity.is_standard_layout());
        for row in 0..rows {
            for (column, source) in expected.columns.iter().enumerate() {
                assert_eq!(
                    dense.values[(row, column)].to_bits(),
                    source.values[row].to_bits()
                );
                assert_eq!(dense.validity[(row, column)], source.validity[row]);
            }
        }
    }

    #[test]
    fn dense_window_waves_preserve_exact_cells_and_row_major_order() {
        let frame = dense_test_frame();
        assert_dense_waves_match_projection(&frame, 0, 17);
        assert_dense_waves_match_projection(&frame, 3, 16);
        let dense = frame.dense_window(0, 17).unwrap();
        assert_eq!(dense.values[(0, 0)].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(dense.values[(1, 0)].to_bits(), 0.0_f64.to_bits());
        for code in 1..10 {
            assert!(dense.validity.iter().any(|reason| *reason as u8 == code));
        }
        for (value, reason) in dense.values.iter().zip(&dense.validity) {
            if !reason.is_valid() {
                assert_eq!(value.to_bits(), f64::NAN.to_bits());
            }
        }
    }

    #[test]
    fn dense_window_admission_checks_destination_ranges_and_later_headroom() {
        assert_eq!(dense_window_destination_layout(17, 7).unwrap(), (119, 1071));
        assert!(dense_window_destination_layout(usize::MAX, 2).is_err());
        assert!(dense_window_destination_layout(isize::MAX as usize, 1).is_err());
        let frame = dense_test_frame();
        for (start, end) in [(5, 4), (0, 18), (18, 18)] {
            assert!(
                frame
                    .dense_window_with_headroom(start, end, || {
                        panic!("invalid ranges must be refused before allocation planning")
                    })
                    .is_err()
            );
        }
        for index in [0, 5, 17] {
            let empty = frame
                .dense_window_with_headroom(index, index, || {
                    panic!("empty windows need no destination or projection allocation")
                })
                .unwrap();
            assert_eq!(empty.values.dim(), (0, 7));
            assert_eq!(empty.validity.dim(), (0, 7));
        }
        let (_, destination_bytes) = dense_window_destination_layout(17, 7).unwrap();
        let error = frame
            .dense_window_with_headroom(0, 17, || destination_bytes - 1)
            .unwrap_err();
        assert!(error.to_string().contains("destination admission refused"));
        let minimum = FEATURE_PROJECTION_BATCH_OVERHEAD_BYTES
            + 17 * (FEATURE_PROJECTION_BYTES_PER_ROW_FIXED + FEATURE_PROJECTION_BYTES_PER_CELL);
        let error = frame
            .dense_window_with_headroom(0, 17, || destination_bytes + 3 * minimum - 1)
            .unwrap_err();
        assert!(error.to_string().contains("projection admission refused"));
        let mut snapshots = 0;
        let error = frame
            .dense_window_with_headroom(0, 17, || {
                snapshots += 1;
                if snapshots == 1 {
                    destination_bytes + 3 * minimum
                } else {
                    0
                }
            })
            .unwrap_err();
        assert_eq!(snapshots, 2);
        assert!(error.to_string().contains("projection admission refused"));
    }

    #[test]
    fn dense_window_vortex_waves_preserve_normalized_indexed_cells() -> Result<()> {
        use crate::core::feature_run_lease::FeatureRunLease;
        use crate::core::vortex_feature_store::{VortexFeatureStore, VortexFeatureStoreOptions};

        let raw = dense_test_frame();
        let FeatureData::InMemory(columns) = &raw.data else {
            unreachable!()
        };
        let temp = tempfile::tempdir()?;
        let lease = Arc::new(FeatureRunLease::create(
            temp.path(),
            "dense-wave-regression",
        )?);
        let store = VortexFeatureStore::create(
            lease,
            &raw.timestamps,
            columns,
            VortexFeatureStoreOptions {
                chunk_rows: 8,
                decoded_cache_bytes: 0,
            },
        )?;
        let vortex = Arc::new(FeatureFrame::from_vortex(
            raw.timestamps.clone(),
            store,
            (*raw.plan).clone(),
            (*raw.provenance).clone(),
        )?);
        assert_dense_waves_match_projection(&vortex, 0, 17);
        assert_eq!(vortex.dense_window(17, 17)?.values.dim(), (0, 7));
        let fit = raw.fit_normalization(0..10, false, &crate::FeatureBuildControl::default())?;
        let normalized = Arc::new(vortex.with_fitted_normalization(&fit)?);
        let reordered = Arc::new(normalized.select_columns(&[6, 0, 5, 1, 4, 2, 3])?);
        let indexed = reordered.shared_select_rows(&[1, 3, 4, 7, 10, 16])?;
        assert_dense_waves_match_projection(&indexed, 0, 6);
        assert_dense_waves_match_projection(&indexed, 1, 5);
        assert_eq!(
            indexed.normalization_fitted_state(),
            reordered.normalization_fitted_state()
        );
        assert_eq!(
            indexed.project_columns(&[0], 0..6)?.row_ids,
            vec![1, 3, 4, 7, 10, 16]
        );
        Ok(())
    }

    #[test]
    fn legacy_feature_options_keep_exact_wire_shape_without_a_working_set() {
        let legacy = br#"{"profile":"Standard","prefix_base_features":false,"higher_tfs":[],"normalization_training_rows":null,"drop_columns_without_normalization_training_support":false}"#;
        let options: FeatureBuildOptions = serde_json::from_slice(legacy).unwrap();
        assert_eq!(options, FeatureBuildOptions::default());
        assert_eq!(serde_json::to_vec(&options).unwrap(), legacy);
    }

    fn raw_frame(offset: f64) -> Arc<FeatureFrame> {
        Arc::new(
            crate::test_fixtures::ctrader_test_feature_frame_from_columns(
                crate::test_fixtures::canonical_test_timestamps(20),
                vec![
                    FeatureColumnF64::new(
                        "signal",
                        (0..20)
                            .map(|row| row as f64 + if row >= 10 { offset } else { 0.0 })
                            .collect(),
                        vec![FeatureCellValidity::Valid; 20],
                    )
                    .unwrap(),
                    FeatureColumnF64::new(
                        "future_only",
                        (0..20)
                            .map(|row| if row < 10 { f64::NAN } else { row as f64 })
                            .collect(),
                        (0..20)
                            .map(|row| {
                                if row < 10 {
                                    FeatureCellValidity::Warmup
                                } else {
                                    FeatureCellValidity::Valid
                                }
                            })
                            .collect(),
                    )
                    .unwrap(),
                ],
            )
            .unwrap(),
        )
    }

    #[test]
    fn shared_causal_window_retains_exact_authority_and_only_shares_parent_values() {
        let raw = raw_frame(0.0);
        let window = raw.shared_row_window(3..7).unwrap();
        let FeatureData::View(view) = &window.data else {
            panic!("must remain a view")
        };
        assert!(Arc::ptr_eq(&view.parent, &raw));
        assert!(Arc::ptr_eq(&window.plan, &raw.plan));
        assert!(Arc::ptr_eq(&window.provenance, &raw.provenance));
        assert_eq!(window.timestamps, raw.timestamps[3..7]);
        let projected = window.project_columns(&[0], 0..4).unwrap();
        assert_eq!(projected.row_ids, vec![3, 4, 5, 6]);
        assert_eq!(projected.columns[0].values, vec![3.0, 4.0, 5.0, 6.0]);
        assert!(raw.shared_row_window(3..3).is_err());
        assert!(raw.shared_row_window(3..21).is_err());
        let mut changed_schema = raw.shared_view().unwrap();
        changed_schema.names[0] = "not_the_plan".into();
        assert!(Arc::new(changed_schema).shared_row_window(3..7).is_err());
    }

    #[test]
    fn bound_column_windows_reuse_one_sealed_projection_and_preserve_raw_model_rows() {
        let raw = raw_frame(0.0);
        let fit = raw
            .fit_normalization(0..20, false, &crate::FeatureBuildControl::default())
            .unwrap();
        let normalized = Arc::new(raw.with_fitted_normalization(&fit).unwrap());
        for source in [&raw, &normalized] {
            let bound = source.bind_column_projection(&[1, 0]).unwrap();
            assert!(Arc::ptr_eq(&bound.source, source));
            let first = bound.row_window(3..7).unwrap();
            let second = bound.row_window(7..11).unwrap();
            assert!(Arc::ptr_eq(&first.plan, &second.plan));
            assert!(Arc::ptr_eq(&first.provenance, &second.provenance));
            let original = source
                .shared_row_window(3..7)
                .unwrap()
                .select_columns(&[1, 0])
                .unwrap();
            assert_eq!(first.plan_identity(), original.plan_identity());
            assert_eq!(first.provenance_identity(), original.provenance_identity());
            assert_eq!(
                first.normalization_fitted_state(),
                original.normalization_fitted_state()
            );
            let actual = first.project_columns(&[0, 1], 0..4).unwrap();
            let expected = original.project_columns(&[0, 1], 0..4).unwrap();
            assert_eq!(actual.timestamps, expected.timestamps);
            assert_eq!(actual.row_ids, expected.row_ids);
            for (actual, expected) in actual.columns.iter().zip(&expected.columns) {
                assert_eq!(actual.name, expected.name);
                assert_eq!(actual.validity, expected.validity);
                assert_eq!(
                    actual
                        .values
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .values
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
            }
            assert_eq!(
                first.raw_model_column("signal").unwrap().values,
                vec![3.0, 4.0, 5.0, 6.0]
            );
            let identity = first.select_columns(&[0, 1]).unwrap();
            assert!(Arc::ptr_eq(&first.plan, &identity.plan));
            let FeatureData::View(view) = &identity.data else {
                panic!("must remain shared")
            };
            assert!(Arc::ptr_eq(&view.parent, source));
            assert!(bound.row_window(1..21).is_err());
            assert!(bound.row_window(3..3).is_err());
            let mut drift = first.clone();
            drift.names.swap(0, 1);
            assert!(drift.select_columns(&[0, 1]).is_err());
        }
        assert!(raw.bind_column_projection(&[0, 0]).is_err());
        assert!(raw.bind_column_projection(&[2]).is_err());
        let mut drift = raw.shared_view().unwrap();
        drift.names[0] = "wrong_schema".into();
        assert!(Arc::new(drift).bind_column_projection(&[0]).is_err());
    }

    #[test]
    fn frozen_model_view_shares_raw_backing_and_never_fits_on_validation() {
        let raw = raw_frame(0.0);
        let altered = raw_frame(1.0e9);
        let control = crate::FeatureBuildControl::default();
        let fit = raw.fit_normalization(0..10, true, &control).unwrap();
        assert_eq!(
            fit,
            altered.fit_normalization(0..10, true, &control).unwrap()
        );
        assert_eq!(fit.column_names(), &["signal".to_owned()]);
        let transformed = Arc::new(raw.with_fitted_normalization(&fit).unwrap());
        let FeatureData::View(view) = &transformed.data else {
            panic!("must be lazy")
        };
        assert!(Arc::ptr_eq(&view.parent, &raw));
        assert_eq!(raw.cell(15, 0).unwrap().value, 15.0);
        let selected = transformed.shared_select_rows(&[1, 3, 8, 15]).unwrap();
        let batch = selected.project_columns(&[0], 1..4).unwrap();
        assert_eq!(batch.row_ids, vec![3, 8, 15]);
        assert_eq!(
            selected.raw_model_column("signal").unwrap().values,
            vec![1.0, 3.0, 8.0, 15.0]
        );
        for (local, original) in [3, 8, 15].into_iter().enumerate() {
            assert_eq!(
                batch.columns[0].values[local].to_bits(),
                transformed.cell(original, 0).unwrap().value.to_bits()
            );
        }
        assert!(transformed.with_fitted_normalization(&fit).is_err());
        let materialized = transformed.row_slice(0, 10).unwrap();
        assert!(materialized.raw_model_column("signal").is_err());
        let live = Arc::new(altered.shared_select_rows(&[18, 19]).unwrap());
        let live = live.with_fitted_normalization(&fit).unwrap();
        assert_eq!(
            live.cell(0, 0).unwrap().value,
            crate::core::normalization::Z_CLIP_F64
        );
        assert_eq!(live.normalization_fitted_state(), Some(&fit));
    }

    #[test]
    fn raw_shared_views_preserve_exact_rows_without_copying_the_cube() {
        let raw = raw_frame(0.0);
        let shared = raw.shared_view().unwrap();
        let FeatureData::View(view) = &shared.data else {
            panic!("must be shared")
        };
        assert!(Arc::ptr_eq(&view.parent, &raw));
        assert_eq!(shared.plan_identity(), raw.plan_identity());
        assert!(shared.normalization_fitted_state().is_none());
        let selected = raw.shared_select_rows(&[2, 4, 6]).unwrap();
        assert_eq!(
            selected.project_columns(&[0], 0..3).unwrap().row_ids,
            vec![2, 4, 6]
        );
        assert!(raw.shared_select_rows(&[2, 2]).is_err());
        assert!(raw.shared_select_rows(&[19, 20]).is_err());
    }

    #[test]
    fn model_base_alias_uses_explicit_recipe_and_keeps_raw_prices_after_projection() {
        let raw = crate::test_fixtures::ctrader_test_feature_frame_from_columns(
            crate::test_fixtures::canonical_test_timestamps(12),
            vec![
                FeatureColumnF64::new(
                    "M1_quant_close",
                    (0..12).map(|row| 1.0 + row as f64 * 0.01).collect(),
                    vec![FeatureCellValidity::Valid; 12],
                )
                .unwrap(),
            ],
        )
        .unwrap()
        .with_feature_build_options(FeatureBuildOptions {
            prefix_base_features: true,
            normalization_training_rows: Some(0..6),
            ..Default::default()
        });
        let raw = Arc::new(raw);
        let fit = raw
            .fit_normalization(0..6, true, &crate::FeatureBuildControl::default())
            .unwrap();
        let normalized = raw.with_fitted_normalization(&fit).unwrap();
        let projected = normalized
            .select_columns(&[0])
            .unwrap()
            .row_window(1, 4)
            .unwrap();
        let name = projected.model_base_feature_name("quant_close").unwrap();
        assert_eq!(name, "M1_quant_close");
        assert_eq!(
            projected.raw_model_column(&name).unwrap().values,
            vec![1.01, 1.02, 1.03]
        );
        assert!(
            projected.raw_model_column("quant_close").is_err(),
            "raw reader must not guess an alias"
        );
    }
}

#[cfg(test)]
mod align_tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn adaptive_projection_plan_scales_with_rows_ram_and_concurrency() -> Result<()> {
        let small = adaptive_feature_projection_plan_for_available_memory(
            10_000,
            800,
            1,
            16 * 1024 * 1024 * 1024,
        )?;
        let dense = adaptive_feature_projection_plan_for_available_memory(
            1_000_000,
            800,
            1,
            16 * 1024 * 1024 * 1024,
        )?;
        let parallel = adaptive_feature_projection_plan_for_available_memory(
            1_000_000,
            800,
            11,
            16 * 1024 * 1024 * 1024,
        )?;

        assert!(small.columns_per_batch > dense.columns_per_batch);
        assert!(parallel.concurrent_batches > 1);
        assert!(parallel.columns_per_batch < dense.columns_per_batch);
        assert!(
            parallel
                .estimated_bytes_per_batch
                .saturating_mul(parallel.concurrent_batches as u64)
                <= parallel.budget_bytes
        );
        Ok(())
    }

    #[test]
    fn adaptive_projection_plan_refuses_zero_and_insufficient_headroom() {
        for available in [0, 1, 64 * 1024 * 1024] {
            let error = adaptive_feature_projection_plan_for_available_memory(
                1_000_000, 779, 11, available,
            )
            .expect_err("no invented 64 MiB or oversized one-column fallback");
            assert!(error.to_string().contains("admission refused"));
            assert!(
                error
                    .to_string()
                    .contains("no rows or columns were omitted")
            );
        }
    }

    #[test]
    fn adaptive_projection_plan_admits_the_exact_single_column_boundary() -> Result<()> {
        // Independent arithmetic: timestamps/row IDs + f64/validity/decode
        // allowance, plus the existing one-MiB overhead. One third is admitted.
        let minimum = 1_000 * (16 + 16) + 1_048_576;
        let plan =
            adaptive_feature_projection_plan_for_available_memory(1_000, 7, 11, 3 * minimum)?;
        assert_eq!(plan.columns_per_batch, 1);
        assert_eq!(plan.concurrent_batches, 1);
        assert_eq!(plan.budget_bytes, minimum);
        assert_eq!(plan.estimated_bytes_per_batch, minimum);
        assert!(
            adaptive_feature_projection_plan_for_available_memory(1_000, 7, 11, 3 * minimum - 1)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn adaptive_projection_plan_keeps_small_columns_parallel_when_ram_is_plentiful() -> Result<()> {
        let plan = adaptive_feature_projection_plan_for_available_memory(
            10_000,
            800,
            11,
            16 * 1024 * 1024 * 1024,
        )?;
        assert_eq!(plan.concurrent_batches, 11);
        assert_eq!(plan.columns_per_batch, 73);
        assert_eq!(800usize.div_ceil(plan.columns_per_batch), 11);
        Ok(())
    }

    #[test]
    fn adaptive_projection_plan_covers_every_column_within_each_admitted_wave() -> Result<()> {
        for rows in [1_000, 25_000, 2_000_000] {
            for columns in [1, 7, 800] {
                for workers in [0, 1, 3, 11] {
                    for available in [10 << 20, 64 << 20, 2 << 30, 32u64 << 30] {
                        let minimum = rows as u64 * 32 + 1_048_576;
                        let budget = (available / 3).min(2_147_483_648);
                        let result = adaptive_feature_projection_plan_for_available_memory(
                            rows, columns, workers, available,
                        );
                        if budget < minimum {
                            assert!(result.is_err());
                            continue;
                        }
                        let plan = result?;
                        assert!(plan.columns_per_batch > 0);
                        assert!(plan.concurrent_batches > 0);
                        assert!(plan.concurrent_batches <= workers.max(1).min(columns));
                        assert!(
                            plan.estimated_bytes_per_batch * plan.concurrent_batches as u64
                                <= budget
                        );
                        let ids = (0..columns).collect::<Vec<_>>();
                        let mut seen = Vec::new();
                        for wave in ids.chunks(plan.columns_per_batch * plan.concurrent_batches) {
                            let mut peak = 0;
                            for batch in wave.chunks(plan.columns_per_batch) {
                                peak += rows as u64 * 16 * (1 + batch.len() as u64) + 1_048_576;
                                seen.extend_from_slice(batch);
                            }
                            assert!(peak <= budget);
                        }
                        assert_eq!(seen, ids);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn adaptive_projection_plan_refuses_overflow_and_preserves_empty_work() -> Result<()> {
        assert!(
            adaptive_feature_projection_plan_for_available_memory(usize::MAX, 1, 1, u64::MAX)
                .is_err()
        );
        let empty = adaptive_feature_projection_plan_for_available_memory(usize::MAX, 0, 11, 0)?;
        assert_eq!(empty.columns_per_batch, 0);
        assert_eq!(empty.concurrent_batches, 0);
        assert_eq!(empty.budget_bytes, 0);
        assert_eq!(empty.estimated_bytes_per_batch, 0);
        Ok(())
    }

    fn ms_grid(start_min: i64, step_min: i64, n: usize) -> Vec<i64> {
        const START_MS: i64 = 1_700_000_000_000;
        (0..n as i64)
            .map(|i| START_MS + (start_min + i * step_min) * 60_000)
            .collect()
    }

    fn align_test_matrix(
        base_ms: &[i64],
        feature_ms: &[i64],
        feature_data: &Array2<f64>,
        forward_fill: bool,
        max_age_ms: Option<i64>,
        availability_lag_ms: i64,
    ) -> Result<Array2<f64>> {
        anyhow::ensure!(
            feature_data.nrows() == feature_ms.len(),
            "test feature matrix row mismatch"
        );
        let columns = (0..feature_data.ncols())
            .map(|column| {
                FeatureColumnF64::new(
                    format!("test_{column}"),
                    feature_data.column(column).iter().copied().collect(),
                    vec![FeatureCellValidity::Valid; feature_ms.len()],
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let aligned = align_feature_columns_by_ms(
            base_ms,
            feature_ms,
            &columns,
            forward_fill,
            max_age_ms,
            availability_lag_ms,
        )?;
        let mut matrix = Array2::from_elem((base_ms.len(), aligned.len()), f64::NAN);
        for (column, values) in aligned.iter().enumerate() {
            for (row, value) in values.values.iter().copied().enumerate() {
                matrix[(row, column)] = value;
            }
        }
        Ok(matrix)
    }

    #[test]
    fn calendar_alignment_uses_observed_closes_and_expires_without_an_invented_period() {
        const HOUR_MS: i64 = 60 * 60 * 1_000;
        const START_MS: i64 = 1_700_000_000_000;
        let base_ms = [0, 12, 22, 23, 24, 46, 47, 48, 71, 72]
            .into_iter()
            .map(|hour| START_MS + hour * HOUR_MS)
            .collect::<Vec<_>>();
        let feature_open_ms = vec![START_MS, START_MS + 23 * HOUR_MS, START_MS + 47 * HOUR_MS];
        let source = FeatureColumnF64::new(
            "D1_truth",
            vec![10.0, 20.0, 30.0],
            vec![FeatureCellValidity::Valid; 3],
        )
        .expect("valid calendar source column");

        let aligned = align_calendar_feature_columns_by_observed_next_open_ms(
            &base_ms,
            &feature_open_ms,
            &[source],
            true,
        )
        .expect("align by broker-observed next opens");

        assert_eq!(aligned.len(), 1);
        for row in 0..3 {
            assert_eq!(
                aligned[0].validity[row],
                FeatureCellValidity::AlignmentMissing
            );
            assert!(aligned[0].values[row].is_nan());
        }
        for row in 3..6 {
            assert_eq!(aligned[0].validity[row], FeatureCellValidity::Valid);
            assert_eq!(aligned[0].values[row], 10.0);
        }
        for row in 6..9 {
            assert_eq!(aligned[0].validity[row], FeatureCellValidity::Valid);
            assert_eq!(aligned[0].values[row], 20.0);
        }
        assert_eq!(aligned[0].validity[9], FeatureCellValidity::Stale);
        assert!(aligned[0].values[9].is_nan());
        assert!(
            !aligned[0].values.contains(&30.0),
            "the last direct calendar bar has no evidenced close and must stay invisible"
        );
    }

    #[test]
    fn align_unbounded_forward_fills_to_end() {
        // Legacy behaviour preserved when max_age = None (lag 0 — note this
        // legacy mode hands t=0..4 the CONTAINING M5 bucket, i.e. lookahead;
        // production HTF alignment passes the period as lag since D02).
        let base_ns = ms_grid(0, 1, 10); // M1 × 10 bars
        let feat_ns = ms_grid(0, 5, 2); // M5 × 2 bars: t=0, t=5
        let feat_data = array![[1.0_f64], [2.0_f64]];
        let aligned = align_test_matrix(&base_ns, &feat_ns, &feat_data, true, None, 0)
            .expect("align f64 test matrix");
        // Without max_age, every base bar past t=5 keeps value 2.0.
        assert_eq!(aligned[(0, 0)], 1.0); // t=0
        assert_eq!(aligned[(4, 0)], 1.0); // t=4 (before first M5 close at 5)
        assert_eq!(aligned[(5, 0)], 2.0); // t=5
        assert_eq!(aligned[(9, 0)], 2.0); // t=9 — frozen, what F-308 calls the bug
    }

    #[test]
    fn align_close_availability_never_reads_the_forming_bar() {
        // Audit D02: with lag = the higher-TF period, a base bar may only
        // read higher-TF bars that have CLOSED at or before its stamp.
        let base_ns = ms_grid(0, 1, 12); // M1 × 12: t=0..11
        let feat_ns = ms_grid(0, 5, 2); //  M5 × 2: opens t=0 (closes 5), t=5 (closes 10)
        let feat_data = array![[1.0_f64], [2.0_f64]];
        let lag = 5 * 60_000_i64; // one M5 period
        let max_age = Some(10 * 60_000_i64); // 2× period, from close
        let aligned = align_test_matrix(&base_ns, &feat_ns, &feat_data, true, max_age, lag)
            .expect("align f64 test matrix");
        // t=0..4: bar[0] is still FORMING (closes at t=5) — its final values
        // must be invisible. The old alignment leaked 1.0 here.
        for i in 0..5 {
            assert!(
                aligned[(i, 0)].is_nan(),
                "t={i}: forming-bar leak — got {}",
                aligned[(i, 0)]
            );
        }
        // t=5..9: bar[0] closed at t=5 → its values become available; bar[1]
        // is forming (closes t=10) and must stay invisible.
        for i in 5..10 {
            assert_eq!(aligned[(i, 0)], 1.0, "t={i}");
        }
        // t=10,11: bar[1] closed at t=10.
        assert_eq!(aligned[(10, 0)], 2.0);
        assert_eq!(aligned[(11, 0)], 2.0);
    }

    #[test]
    fn align_close_availability_staleness_measured_from_close() {
        // One M5 bar opening t=0 (closes t=5), max_age = 3 min FROM CLOSE:
        // available t=5..8, stale (NaN) from t=9.
        let base_ns = ms_grid(0, 1, 12);
        let feat_ns = ms_grid(0, 5, 1);
        let feat_data = array![[7.0_f64]];
        let lag = 5 * 60_000_i64;
        let max_age = Some(3 * 60_000_i64);
        let aligned = align_test_matrix(&base_ns, &feat_ns, &feat_data, true, max_age, lag)
            .expect("align f64 test matrix");
        for i in 0..5 {
            assert!(aligned[(i, 0)].is_nan(), "t={i}: not yet closed");
        }
        for i in 5..9 {
            assert_eq!(aligned[(i, 0)], 7.0, "t={i}: fresh after close");
        }
        for i in 9..12 {
            assert!(
                aligned[(i, 0)].is_nan(),
                "t={i}: stale past max_age from close"
            );
        }
    }

    #[test]
    fn align_max_age_caps_stale_forward_fill() {
        // F-308 fix: max_age = 3 minutes (in ns) drops values past 3 min lag.
        let base_ns = ms_grid(0, 1, 10);
        let feat_ns = ms_grid(0, 5, 2);
        let feat_data = array![[1.0_f64], [2.0_f64]];
        let max_age_ns = Some(3_i64 * 60_000);
        let aligned = align_test_matrix(&base_ns, &feat_ns, &feat_data, true, max_age_ns, 0)
            .expect("align f64 test matrix");
        // t=0 → exact, 1.0
        assert_eq!(aligned[(0, 0)], 1.0);
        // t=1,2,3 → within 3min of t=0, still ffill to 1.0
        assert_eq!(aligned[(3, 0)], 1.0);
        // t=4 → 4 min after t=0, EXCEEDS max_age → NaN
        assert!(
            aligned[(4, 0)].is_nan(),
            "expected NaN at t=4, got {}",
            aligned[(4, 0)]
        );
        // t=5 → exact match on second feat row, value 2.0
        assert_eq!(aligned[(5, 0)], 2.0);
        // t=6,7,8 → within 3min of t=5, ffill 2.0
        assert_eq!(aligned[(8, 0)], 2.0);
        // t=9 → 4 min after t=5, exceeds → NaN. This is what kills the
        // frozen-constant downstream propagation in the F-308 scenario.
        assert!(
            aligned[(9, 0)].is_nan(),
            "expected NaN at t=9, got {}",
            aligned[(9, 0)]
        );
    }

    #[test]
    fn align_max_age_zero_preserves_exact_matches() {
        // Edge case: max_age = 0 forbids any forward-fill, only exact ts hits.
        let base_ns = ms_grid(0, 1, 5);
        let feat_ns = ms_grid(0, 5, 1); // single feat row at t=0
        let feat_data = array![[42.0_f64]];
        let aligned = align_test_matrix(&base_ns, &feat_ns, &feat_data, true, Some(0), 0)
            .expect("align f64 test matrix");
        assert_eq!(aligned[(0, 0)], 42.0); // exact match
        for i in 1..5 {
            assert!(aligned[(i, 0)].is_nan(), "expected NaN at i={i}");
        }
    }

    #[test]
    fn align_max_age_with_ffill_false_is_consistent() {
        // When ffill is false, max_age has no effect — only exact matches.
        let base_ns = ms_grid(0, 1, 5);
        let feat_ns = ms_grid(0, 5, 1);
        let feat_data = array![[7.0_f64]];
        let aligned = align_test_matrix(&base_ns, &feat_ns, &feat_data, false, Some(i64::MAX), 0)
            .expect("align f64 test matrix");
        assert_eq!(aligned[(0, 0)], 7.0);
        for i in 1..5 {
            assert!(aligned[(i, 0)].is_nan());
        }
    }

    #[test]
    fn align_empty_feature_grid_fails_closed() {
        let base_ns = ms_grid(0, 1, 5);
        let feat_ns: Vec<i64> = Vec::new();
        let feat_data: Array2<f64> = Array2::zeros((0, 2));
        let error = align_test_matrix(&base_ns, &feat_ns, &feat_data, true, Some(60_000), 0)
            .expect_err("an empty direct feature timestamp grid is not canonical");
        assert!(format!("{error:#}").contains("must not be empty"));
    }

    #[test]
    fn align_higher_tf_ends_before_base_last_creates_nan_tail() {
        // The F-308 production scenario: base = M1 × 100 fresh bars,
        // higher TF = D1 with only 1 bar at t=0. Without max_age the
        // entire 100-bar base would have constant D1 values. With
        // max_age = 2 × D1_period = 2 days, all but the first ~2*1440 min
        // of base bars become NaN.
        let base_ns = ms_grid(0, 1, 100); // M1 × 100 = 100 min span
        let feat_ns = ms_grid(0, 1440, 1); // single D1 bar at t=0
        let feat_data = array![[99.0_f64]];
        let max_age_ns = Some(2_i64 * 1440 * 60_000);
        let aligned = align_test_matrix(&base_ns, &feat_ns, &feat_data, true, max_age_ns, 0)
            .expect("align f64 test matrix");
        // All 100 base bars are within 2 days of t=0, so ALL get 99.0.
        for i in 0..100 {
            assert_eq!(aligned[(i, 0)], 99.0);
        }
        // Now tighten max_age to 50 minutes — only first 51 base bars
        // (t=0..50) survive; rest become NaN.
        let max_age_ns = Some(50_i64 * 60_000);
        let aligned = align_test_matrix(&base_ns, &feat_ns, &feat_data, true, max_age_ns, 0)
            .expect("align f64 test matrix");
        for i in 0..=50 {
            assert_eq!(aligned[(i, 0)], 99.0, "i={i}");
        }
        for i in 51..100 {
            assert!(
                aligned[(i, 0)].is_nan(),
                "expected NaN at i={i}, got {}",
                aligned[(i, 0)]
            );
        }
    }
}
