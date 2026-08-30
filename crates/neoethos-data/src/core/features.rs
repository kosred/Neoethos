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

/// Version 3 supports an explicit per-row close-availability schedule. Fixed
/// timeframes use open + exact period; calendar timeframes use the next direct
/// broker bar-open and never invent 24-hour/7-day/30-day durations.
pub const HIGHER_TIMEFRAME_ALIGNMENT_SEMANTIC_VERSION: u32 = 3;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureBuildOptions {
    pub profile: FeatureProfile,
    pub include_smc: bool,
    pub include_hpc_ta: bool,
    pub include_regime: bool,
    pub include_quant: bool,
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
}

impl Default for FeatureBuildOptions {
    fn default() -> Self {
        Self {
            profile: FeatureProfile::Standard,
            include_smc: true,
            include_hpc_ta: true,
            include_regime: true,
            include_quant: true,
            prefix_base_features: false,
            higher_tfs: Vec::new(),
            normalization_training_rows: None,
            drop_columns_without_normalization_training_support: false,
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
}

impl FeatureFrame {
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
        self.validate_projection(column_indices, &row_range)?;
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
                view.parent
                    .project_columns(&physical_columns, physical_range)
            }
        }
    }

    /// Select and reorder logical feature columns without copying their
    /// physical values. The selected output order becomes part of a new
    /// `FeaturePlanIdentity`; concrete dataset provenance stays identical.
    pub fn select_columns(&self, column_indices: &[usize]) -> Result<Self> {
        self.validate_projection(column_indices, &(0..self.n_samples()))?;
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
            }),
            Arc::new(plan),
            Arc::new(provenance),
            Arc::clone(&self.source_generation_leases),
            self.row_ids.clone(),
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
        let columns = (0..self.n_features()).collect::<Vec<_>>();
        let batch = self.project_columns(&columns, start..end)?;
        let rows = end.saturating_sub(start);
        let mut values = Array2::from_elem((rows, self.n_features()), f64::NAN);
        let mut validity = Array2::from_elem(
            (rows, self.n_features()),
            FeatureCellValidity::AlignmentMissing,
        );
        for (column_index, column) in batch.columns.iter().enumerate() {
            for row in 0..rows {
                values[(row, column_index)] = column.values[row];
                validity[(row, column_index)] = column.validity[row];
            }
        }
        Ok(FeatureDenseMatrixF64 { values, validity })
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
            }),
            Arc::clone(&self.plan),
            Arc::clone(&self.provenance),
            Arc::clone(&self.source_generation_leases),
            self.row_ids_for_window(start..end)?,
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
    use crate::core::timestamps::validate_canonical_millisecond_timestamps;

    if let Some(max_age_ms) = max_age_ms {
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
mod align_tests {
    use super::*;
    use ndarray::{Array2, array};

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
    fn calendar_alignment_uses_the_next_direct_bar_open_without_a_fixed_period() {
        const HOUR_MS: i64 = 60 * 60 * 1_000;
        const START_MS: i64 = 1_700_000_000_000;
        let base_ms = [0, 12, 22, 23, 24, 46, 47, 48]
            .into_iter()
            .map(|hour| START_MS + hour * HOUR_MS)
            .collect::<Vec<_>>();
        let feature_open_ms = vec![START_MS, START_MS + 23 * HOUR_MS, START_MS + 47 * HOUR_MS];
        let available_at_ms = vec![
            Some(START_MS + 23 * HOUR_MS),
            Some(START_MS + 47 * HOUR_MS),
            None,
        ];
        let source = FeatureColumnF64::new(
            "D1_truth",
            vec![10.0, 20.0, 30.0],
            vec![FeatureCellValidity::Valid; 3],
        )
        .expect("valid calendar source column");

        let aligned = align_feature_columns_at_explicit_availability_ms(
            &base_ms,
            &feature_open_ms,
            &available_at_ms,
            &[source],
            true,
            None,
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
        for row in 6..8 {
            assert_eq!(aligned[0].validity[row], FeatureCellValidity::Valid);
            assert_eq!(aligned[0].values[row], 20.0);
        }
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
