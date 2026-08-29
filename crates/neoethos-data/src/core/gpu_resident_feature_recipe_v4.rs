//! Data-owned local producer drafts and one-shot global column-schema sealing.
//!
//! Producers describe their local outputs and owner-derived memory before any
//! global ordinal exists. Data assigns global ordinals exactly once in the
//! canonical CPU schema order. This module deliberately cannot mint runtime
//! admission, a fitted feature plan, or source provenance.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error as StdError;
use std::fmt;

use anyhow::{Context, Result as AnyResult, ensure};
use neoethos_dataset_contracts::CanonicalTimeframe;
use neoethos_feature_contracts::{
    DatasetFeatureArtifactProvenanceV1, FeatureNodeV1, FeatureOperationTagV1, FeatureOutputV1,
    FeatureParameterV1, FeaturePlanV1,
};
use neoethos_gpu_contracts::resident_feature_store_v3::{
    ResidentFeatureContractErrorV3, ResidentFeatureProducerV3, ResidentFeatureRouteV3,
    ResidentFeatureStageV3, ResidentProducerCapabilityManifestV3, ResidentProducerCapabilityV3,
};
use neoethos_gpu_cuda::resident_robust_normalization_v2::resident_robust_normalization_disabled_fit_sha256_v2;
use sha2::{Digest, Sha256};

use super::features::FeatureProfile;
use super::gpu_resident_robust_normalization_v2::SealedCanonicalRobustNormalizationSplitV2;
use super::pinned_canonical_series_v1::{
    MaterializedPinnedResidentCanonicalSourcesV1, PinnedResidentCanonicalSourceDescriptorV1,
};

pub(crate) const RESIDENT_FEATURE_SCHEMA_VERSION_V4: u32 = 4;
pub(crate) const MAX_RESIDENT_PRODUCER_BATCH_COLUMNS_V4: usize = 64;

/// Canonical feature-column order. This is intentionally independent of the
/// capability-manifest order and the runtime implementation inventory.
pub(crate) const RESIDENT_COLUMN_SCHEMA_ORDER_V4: [ResidentFeatureProducerV3; 7] = [
    ResidentFeatureProducerV3::Smc,
    ResidentFeatureProducerV3::ClassicTa,
    ResidentFeatureProducerV3::Quant,
    ResidentFeatureProducerV3::Session,
    ResidentFeatureProducerV3::Regime,
    ResidentFeatureProducerV3::Footprint,
    ResidentFeatureProducerV3::HigherTimeframeAlignment,
];

/// Post-column transforms/infrastructure. These entries never own a schema
/// span and never enter producer batch-to-route coverage.
pub(crate) const RESIDENT_TRANSFORM_ORDER_V4: [ResidentFeatureProducerV3; 3] = [
    ResidentFeatureProducerV3::RobustNormalization,
    ResidentFeatureProducerV3::CanonicalContentSha256,
    ResidentFeatureProducerV3::FeatureMajorToBarMajor,
];

#[derive(Debug)]
pub(crate) enum ResidentFeatureRecipeErrorV4 {
    EmptyField(&'static str),
    ZeroSemanticVersion,
    InvalidParameter(String),
    DuplicateParameter(String),
    DuplicateFeatureName(String),
    DuplicateCapability(ResidentFeatureProducerV3),
    CapabilityProducerMismatch {
        draft: ResidentFeatureProducerV3,
        capability: ResidentFeatureProducerV3,
    },
    TransformCapabilityProducerMismatch {
        expected: ResidentFeatureProducerV3,
        actual: ResidentFeatureProducerV3,
    },
    MissingCapability(ResidentFeatureProducerV3),
    ProducerOrderMismatch {
        index: usize,
        expected: ResidentFeatureProducerV3,
        actual: ResidentFeatureProducerV3,
    },
    MissingColumnProducers {
        missing: Vec<ResidentFeatureProducerV3>,
    },
    InvalidBatchExtent {
        producer: ResidentFeatureProducerV3,
        local_first_column: usize,
        column_count: usize,
    },
    BatchCoverageMismatch {
        producer: ResidentFeatureProducerV3,
        expected: usize,
        actual: usize,
    },
    WorkspacePreflightMismatch(&'static str),
    SourceMaterialization(String),
    ExtentOverflow(&'static str),
    Contract(ResidentFeatureContractErrorV3),
}

impl fmt::Display for ResidentFeatureRecipeErrorV4 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(formatter, "resident recipe-v4 {field} is empty"),
            Self::ZeroSemanticVersion => {
                formatter.write_str("resident recipe-v4 semantic version must be positive")
            }
            Self::InvalidParameter(name) => {
                write!(
                    formatter,
                    "resident recipe-v4 parameter `{name}` is invalid"
                )
            }
            Self::DuplicateParameter(name) => {
                write!(formatter, "resident recipe-v4 repeats parameter `{name}`")
            }
            Self::DuplicateFeatureName(name) => {
                write!(formatter, "resident recipe-v4 repeats feature `{name}`")
            }
            Self::DuplicateCapability(producer) => write!(
                formatter,
                "resident recipe-v4 repeats the {} capability",
                producer.as_str()
            ),
            Self::CapabilityProducerMismatch { draft, capability } => write!(
                formatter,
                "resident {} draft carries a {} capability",
                draft.as_str(),
                capability.as_str()
            ),
            Self::TransformCapabilityProducerMismatch { expected, actual } => write!(
                formatter,
                "resident transform capability is {}, expected {}",
                actual.as_str(),
                expected.as_str()
            ),
            Self::MissingCapability(producer) => write!(
                formatter,
                "resident recipe-v4 is missing the {} capability",
                producer.as_str()
            ),
            Self::ProducerOrderMismatch {
                index,
                expected,
                actual,
            } => write!(
                formatter,
                "resident schema-v4 producer {index} is {}, expected {}",
                actual.as_str(),
                expected.as_str()
            ),
            Self::MissingColumnProducers { missing } => {
                write!(
                    formatter,
                    "resident schema-v4 is missing producers {missing:?}"
                )
            }
            Self::InvalidBatchExtent {
                producer,
                local_first_column,
                column_count,
            } => write!(
                formatter,
                "resident {} batch {local_first_column}+{column_count} is invalid",
                producer.as_str()
            ),
            Self::BatchCoverageMismatch {
                producer,
                expected,
                actual,
            } => write!(
                formatter,
                "resident {} batch coverage ended at {actual}, expected {expected}",
                producer.as_str()
            ),
            Self::WorkspacePreflightMismatch(field) => {
                write!(
                    formatter,
                    "resident recipe-v4 workspace preflight mismatched {field}"
                )
            }
            Self::SourceMaterialization(detail) => {
                write!(
                    formatter,
                    "resident recipe-v4 source materialization failed: {detail}"
                )
            }
            Self::ExtentOverflow(field) => {
                write!(formatter, "resident recipe-v4 {field} overflowed")
            }
            Self::Contract(error) => error.fmt(formatter),
        }
    }
}

/// Move-only continuation between metadata-only pinning and the complete
/// seven-producer schema seal. It owns the exact source leases and canonical
/// normalization split while crate-owned producer factories append local
/// drafts. It is not a materialization admission or a feature-plan identity.
#[derive(Debug)]
pub(crate) struct PreparedResidentFeatureRecipeAssemblyV4 {
    sources: MaterializedPinnedResidentCanonicalSourcesV1,
    base_timeframe: CanonicalTimeframe,
    profile: FeatureProfile,
    row_count: usize,
    budget_rows: usize,
    producer_capabilities: ResidentProducerCapabilityManifestV3,
    robust_normalization_split: SealedCanonicalRobustNormalizationSplitV2,
    column_schema: ResidentColumnSchemaAssemblerV4,
}

impl PreparedResidentFeatureRecipeAssemblyV4 {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_workspace_preflight(
        source_descriptor: PinnedResidentCanonicalSourceDescriptorV1,
        base_timeframe: CanonicalTimeframe,
        profile: FeatureProfile,
        row_count: usize,
        budget_rows: usize,
        producer_capabilities: ResidentProducerCapabilityManifestV3,
        robust_normalization_split: SealedCanonicalRobustNormalizationSplitV2,
    ) -> Result<Self, ResidentFeatureRecipeErrorV4> {
        if row_count == 0
            || budget_rows < row_count
            || source_descriptor.generation_count() == 0
            || source_descriptor.source_binding(base_timeframe).is_err()
            || producer_capabilities.capabilities().len() != ResidentFeatureProducerV3::ALL.len()
            || !robust_normalization_split.is_intact_for_row_count(row_count)
        {
            return Err(ResidentFeatureRecipeErrorV4::WorkspacePreflightMismatch(
                "source, extents, capabilities, or normalization split",
            ));
        }
        let sources = source_descriptor
            .into_materialized_resident_sources_v1(base_timeframe)
            .map_err(|error| {
                ResidentFeatureRecipeErrorV4::SourceMaterialization(error.to_string())
            })?;
        if sources.base().frame().len() != row_count {
            return Err(ResidentFeatureRecipeErrorV4::WorkspacePreflightMismatch(
                "materialized base row count",
            ));
        }
        Ok(Self {
            sources,
            base_timeframe,
            profile,
            row_count,
            budget_rows,
            producer_capabilities,
            robust_normalization_split,
            column_schema: ResidentColumnSchemaAssemblerV4::default(),
        })
    }

    pub(crate) const fn base_timeframe(&self) -> CanonicalTimeframe {
        self.base_timeframe
    }

    pub(crate) const fn row_count(&self) -> usize {
        self.row_count
    }

    pub(crate) const fn budget_rows(&self) -> usize {
        self.budget_rows
    }

    pub(crate) const fn profile(&self) -> FeatureProfile {
        self.profile
    }

    pub(crate) const fn resident_sources(&self) -> &MaterializedPinnedResidentCanonicalSourcesV1 {
        &self.sources
    }

    pub(crate) fn append_owner_draft(
        &mut self,
        draft: ResidentProducerDraftV4,
    ) -> Result<(), ResidentFeatureRecipeErrorV4> {
        self.column_schema.append(draft)
    }

    pub(crate) fn seal(
        self,
        transform_capabilities: ResidentTransformCapabilityDraftV4,
    ) -> Result<PreparedResidentFeatureRecipeV4, ResidentFeatureRecipeErrorV4> {
        let schema = self.column_schema.seal(transform_capabilities)?;
        if schema.capability_manifest() != &self.producer_capabilities {
            return Err(ResidentFeatureRecipeErrorV4::WorkspacePreflightMismatch(
                "producer capability manifest",
            ));
        }
        Ok(PreparedResidentFeatureRecipeV4 {
            sources: self.sources,
            base_timeframe: self.base_timeframe,
            profile: self.profile,
            row_count: self.row_count,
            budget_rows: self.budget_rows,
            robust_normalization_split: self.robust_normalization_split,
            schema,
        })
    }
}

/// Complete pre-device recipe continuation. Exact identities and admission are
/// deliberately derived by Data's later materializer; this token carries no
/// caller-supplied digest and cannot exist until all ten real capabilities and
/// all seven column drafts agree.
#[derive(Debug)]
pub(crate) struct PreparedResidentFeatureRecipeV4 {
    sources: MaterializedPinnedResidentCanonicalSourcesV1,
    base_timeframe: CanonicalTimeframe,
    profile: FeatureProfile,
    row_count: usize,
    budget_rows: usize,
    robust_normalization_split: SealedCanonicalRobustNormalizationSplitV2,
    schema: SealedResidentColumnSchemaV4,
}

impl PreparedResidentFeatureRecipeV4 {
    pub(crate) fn into_materialization_v4(
        self,
    ) -> AnyResult<PreparedResidentFeatureMaterializationV4> {
        let Self {
            sources,
            base_timeframe,
            profile,
            row_count,
            budget_rows,
            robust_normalization_split,
            schema,
        } = self;
        let normalization_enabled = robust_normalization_split.enabled();
        ensure!(
            sources.base().frame().len() == row_count,
            "materialized resident base row count drifted after recipe seal"
        );
        let SealedResidentColumnSchemaV4 {
            routes,
            plan_route_templates,
            producer_batches,
            capability_manifest,
        } = schema;
        let normalization_capability = capability_manifest
            .capabilities()
            .iter()
            .find(|capability| {
                capability.producer() == ResidentFeatureProducerV3::RobustNormalization
            })
            .context("resident recipe omitted robust-normalization capability")?;
        let feature_identity = ResidentFeatureIdentityTemplateV4::seal(
            sources,
            base_timeframe,
            profile,
            row_count,
            budget_rows,
            plan_route_templates,
            normalization_enabled,
            normalization_capability.implementation_sha256(),
            normalization_capability.exact_math_authority().to_owned(),
            &routes,
            &capability_manifest,
        )?;
        Ok(PreparedResidentFeatureMaterializationV4 {
            feature_identity,
            robust_normalization_split,
            planned_routes: routes,
            producer_batches,
            capability_manifest,
        })
    }
}

impl StdError for ResidentFeatureRecipeErrorV4 {}

impl From<ResidentFeatureContractErrorV3> for ResidentFeatureRecipeErrorV4 {
    fn from(error: ResidentFeatureContractErrorV3) -> Self {
        Self::Contract(error)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ResidentCanonicalParameterValueV4 {
    Bool(bool),
    U64(u64),
    I64(i64),
    F64Bits(u64),
    Text(String),
    Hash([u8; 32]),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResidentCanonicalParameterV4 {
    name: String,
    value: ResidentCanonicalParameterValueV4,
}

impl ResidentCanonicalParameterV4 {
    pub(crate) fn from_typed_value(
        name: impl Into<String>,
        value: ResidentCanonicalParameterValueV4,
    ) -> Result<Self, ResidentFeatureRecipeErrorV4> {
        let name = name.into();
        if semantic_text_is_blank(&name) {
            return Err(ResidentFeatureRecipeErrorV4::EmptyField(
                "canonical parameter name",
            ));
        }
        match &value {
            ResidentCanonicalParameterValueV4::F64Bits(bits)
                if !f64::from_bits(*bits).is_finite() || *bits == (-0.0_f64).to_bits() =>
            {
                return Err(ResidentFeatureRecipeErrorV4::InvalidParameter(name));
            }
            ResidentCanonicalParameterValueV4::Text(text) if semantic_text_is_blank(text) => {
                return Err(ResidentFeatureRecipeErrorV4::InvalidParameter(name));
            }
            ResidentCanonicalParameterValueV4::Hash(hash) if hash.iter().all(|byte| *byte == 0) => {
                return Err(ResidentFeatureRecipeErrorV4::InvalidParameter(name));
            }
            _ => {}
        }
        Ok(Self { name, value })
    }

    fn into_feature_parameter_v1(self) -> AnyResult<FeatureParameterV1> {
        let parameter = match self.value {
            ResidentCanonicalParameterValueV4::Bool(value) => {
                FeatureParameterV1::bool(self.name, value)?
            }
            ResidentCanonicalParameterValueV4::U64(value) => {
                FeatureParameterV1::u64(self.name, value)?
            }
            ResidentCanonicalParameterValueV4::I64(value) => {
                FeatureParameterV1::i64(self.name, value)?
            }
            ResidentCanonicalParameterValueV4::F64Bits(bits) => {
                FeatureParameterV1::f64(self.name, f64::from_bits(bits))?
            }
            ResidentCanonicalParameterValueV4::Text(value) => {
                FeatureParameterV1::text(self.name, value)?
            }
            ResidentCanonicalParameterValueV4::Hash(value) => {
                FeatureParameterV1::hash(self.name, value)?
            }
        };
        Ok(parameter)
    }
}

/// A producer-local route description. It deliberately has no global ordinal,
/// route id, parameter hash, or route receipt.
#[derive(Debug)]
pub(crate) struct ResidentRouteDraftV4 {
    feature_name: String,
    indicator_id: Option<String>,
    output_id: Option<String>,
    stage: ResidentFeatureStageV3,
    swept_period: Option<u64>,
    typed_parameters: Vec<ResidentCanonicalParameterV4>,
    route_domain: &'static str,
}

impl ResidentRouteDraftV4 {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_typed_parts(
        feature_name: impl Into<String>,
        indicator_id: Option<impl Into<String>>,
        output_id: Option<impl Into<String>>,
        stage: ResidentFeatureStageV3,
        swept_period: Option<u64>,
        mut typed_parameters: Vec<ResidentCanonicalParameterV4>,
        route_domain: &'static str,
    ) -> Result<Self, ResidentFeatureRecipeErrorV4> {
        let feature_name = feature_name.into();
        let indicator_id = indicator_id.map(Into::into);
        let output_id = output_id.map(Into::into);
        if semantic_text_is_blank(&feature_name) {
            return Err(ResidentFeatureRecipeErrorV4::EmptyField("feature name"));
        }
        if semantic_text_is_blank(route_domain)
            || indicator_id.as_deref().is_some_and(semantic_text_is_blank)
            || output_id.as_deref().is_some_and(semantic_text_is_blank)
        {
            return Err(ResidentFeatureRecipeErrorV4::EmptyField(
                "route semantic identity",
            ));
        }
        typed_parameters.sort_by(|left, right| left.name.cmp(&right.name));
        let mut parameter_names = BTreeSet::new();
        for parameter in &typed_parameters {
            if !parameter_names.insert(parameter.name.as_str()) {
                return Err(ResidentFeatureRecipeErrorV4::DuplicateParameter(
                    parameter.name.clone(),
                ));
            }
        }
        Ok(Self {
            feature_name,
            indicator_id,
            output_id,
            stage,
            swept_period,
            typed_parameters,
            route_domain,
        })
    }

    pub(crate) fn canonical_parameter_tuple_sha256_v4(
        &self,
    ) -> Result<[u8; 32], ResidentFeatureRecipeErrorV4> {
        derive_parameter_tuple_sha256_v4(&self.typed_parameters)
    }
}

/// Exact producer-local memory supplied by the owner that defines the same
/// allocations. Data never infers additional retained or scratch bytes.
#[derive(Debug)]
pub(crate) struct ResidentProducerBatchDraftV4 {
    local_first_column: usize,
    column_count: usize,
    additional_retained_bytes: u64,
    scratch_bytes: u64,
}

impl ResidentProducerBatchDraftV4 {
    pub(crate) fn from_owner_preflight(
        local_first_column: usize,
        column_count: usize,
        additional_retained_bytes: u64,
        scratch_bytes: u64,
    ) -> Self {
        Self {
            local_first_column,
            column_count,
            additional_retained_bytes,
            scratch_bytes,
        }
    }
}

#[derive(Debug)]
pub(crate) struct ResidentProducerDraftV4 {
    producer: ResidentFeatureProducerV3,
    semantic_version: u32,
    routes: Vec<ResidentRouteDraftV4>,
    batches: Vec<ResidentProducerBatchDraftV4>,
    capability: ResidentProducerCapabilityV3,
}

impl ResidentProducerDraftV4 {
    pub(crate) fn from_owner_preflight(
        producer: ResidentFeatureProducerV3,
        semantic_version: u32,
        routes: Vec<ResidentRouteDraftV4>,
        batches: Vec<ResidentProducerBatchDraftV4>,
        capability: ResidentProducerCapabilityV3,
    ) -> Result<Self, ResidentFeatureRecipeErrorV4> {
        if semantic_version == 0 {
            return Err(ResidentFeatureRecipeErrorV4::ZeroSemanticVersion);
        }
        if capability.producer() != producer {
            return Err(ResidentFeatureRecipeErrorV4::CapabilityProducerMismatch {
                draft: producer,
                capability: capability.producer(),
            });
        }
        if routes.is_empty() || batches.is_empty() {
            return Err(ResidentFeatureRecipeErrorV4::EmptyField(
                "producer routes/batches",
            ));
        }
        let mut names = BTreeSet::new();
        for route in &routes {
            if !names.insert(route.feature_name.as_str()) {
                return Err(ResidentFeatureRecipeErrorV4::DuplicateFeatureName(
                    route.feature_name.clone(),
                ));
            }
        }
        let mut next_local_column = 0_usize;
        for batch in &batches {
            let local_first_column = batch.local_first_column;
            let column_count = batch.column_count;
            if column_count == 0
                || column_count > MAX_RESIDENT_PRODUCER_BATCH_COLUMNS_V4
                || local_first_column != next_local_column
            {
                return Err(ResidentFeatureRecipeErrorV4::InvalidBatchExtent {
                    producer,
                    local_first_column,
                    column_count,
                });
            }
            next_local_column = local_first_column.checked_add(column_count).ok_or(
                ResidentFeatureRecipeErrorV4::ExtentOverflow("producer-local batch range"),
            )?;
        }
        if next_local_column != routes.len() {
            return Err(ResidentFeatureRecipeErrorV4::BatchCoverageMismatch {
                producer,
                expected: routes.len(),
                actual: next_local_column,
            });
        }
        Ok(Self {
            producer,
            semantic_version,
            routes,
            batches,
            capability,
        })
    }
}

/// Move-only transform capability draft. Transform capabilities are required
/// for the complete ten-entry capability manifest but never own columns or
/// producer batches.
#[derive(Debug)]
pub(crate) struct ResidentTransformCapabilityDraftV4 {
    capabilities: [ResidentProducerCapabilityV3; 3],
}

impl ResidentTransformCapabilityDraftV4 {
    pub(crate) fn from_owner_capabilities(
        robust_normalization: ResidentProducerCapabilityV3,
        canonical_content_sha256: ResidentProducerCapabilityV3,
        feature_major_to_bar_major: ResidentProducerCapabilityV3,
    ) -> Result<Self, ResidentFeatureRecipeErrorV4> {
        let capabilities = [
            robust_normalization,
            canonical_content_sha256,
            feature_major_to_bar_major,
        ];
        for (capability, expected) in capabilities.iter().zip(RESIDENT_TRANSFORM_ORDER_V4) {
            if capability.producer() != expected {
                return Err(
                    ResidentFeatureRecipeErrorV4::TransformCapabilityProducerMismatch {
                        expected,
                        actual: capability.producer(),
                    },
                );
            }
        }
        Ok(Self { capabilities })
    }
}

#[derive(Debug)]
pub(crate) struct ResolvedResidentProducerBatchMemoryV4 {
    producer: ResidentFeatureProducerV3,
    first_column: usize,
    column_count: usize,
    additional_retained_bytes: u64,
    scratch_bytes: u64,
}

impl ResolvedResidentProducerBatchMemoryV4 {
    pub(crate) const fn producer(&self) -> ResidentFeatureProducerV3 {
        self.producer
    }

    pub(crate) const fn first_column(&self) -> usize {
        self.first_column
    }

    pub(crate) const fn column_count(&self) -> usize {
        self.column_count
    }

    pub(crate) const fn additional_retained_bytes(&self) -> u64 {
        self.additional_retained_bytes
    }

    pub(crate) const fn scratch_bytes(&self) -> u64 {
        self.scratch_bytes
    }
}

#[derive(Debug)]
pub(crate) struct ResidentFeaturePlanRouteTemplateV4 {
    producer: ResidentFeatureProducerV3,
    semantic_version: u32,
    feature_name: String,
    route_node_id: String,
    typed_parameters: Vec<ResidentCanonicalParameterV4>,
    route_domain: String,
    implementation_sha256: [u8; 32],
    exact_math_authority: String,
    route_receipt_sha256: [u8; 32],
}

#[derive(Debug)]
pub(crate) struct SealedResidentColumnSchemaV4 {
    routes: Vec<ResidentFeatureRouteV3>,
    plan_route_templates: Vec<ResidentFeaturePlanRouteTemplateV4>,
    producer_batches: Vec<ResolvedResidentProducerBatchMemoryV4>,
    capability_manifest: ResidentProducerCapabilityManifestV3,
}

impl SealedResidentColumnSchemaV4 {
    pub(crate) fn routes(&self) -> &[ResidentFeatureRouteV3] {
        &self.routes
    }

    pub(crate) const fn capability_manifest(&self) -> &ResidentProducerCapabilityManifestV3 {
        &self.capability_manifest
    }
}

/// Complete pre-device recipe after every pinned generation has been decoded
/// exactly once. Identity hashes here are admission templates, not final
/// FeaturePlan or artifact-provenance identities.
#[derive(Debug)]
pub(crate) struct PreparedResidentFeatureMaterializationV4 {
    feature_identity: ResidentFeatureIdentityTemplateV4,
    robust_normalization_split: SealedCanonicalRobustNormalizationSplitV2,
    planned_routes: Vec<ResidentFeatureRouteV3>,
    producer_batches: Vec<ResolvedResidentProducerBatchMemoryV4>,
    capability_manifest: ResidentProducerCapabilityManifestV3,
}

impl PreparedResidentFeatureMaterializationV4 {
    #[allow(clippy::type_complexity)]
    pub(crate) fn into_parts(
        self,
    ) -> (
        ResidentFeatureIdentityTemplateV4,
        SealedCanonicalRobustNormalizationSplitV2,
        Vec<ResidentFeatureRouteV3>,
        Vec<ResolvedResidentProducerBatchMemoryV4>,
        ResidentProducerCapabilityManifestV3,
    ) {
        (
            self.feature_identity,
            self.robust_normalization_split,
            self.planned_routes,
            self.producer_batches,
            self.capability_manifest,
        )
    }
}

/// Move-only inputs required to construct the final FeaturePlan after runtime
/// normalization has produced and validated its fitted-state digest.
#[derive(Debug)]
pub(crate) struct ResidentFeatureIdentityTemplateV4 {
    sources: MaterializedPinnedResidentCanonicalSourcesV1,
    routes: Vec<ResidentFeaturePlanRouteTemplateV4>,
    normalization_enabled: bool,
    normalization_implementation_sha256: [u8; 32],
    normalization_exact_math_authority: String,
    dataset_recipe_sha256: [u8; 32],
    feature_plan_schema_sha256: [u8; 32],
    route_plan_sha256: [u8; 32],
}

impl ResidentFeatureIdentityTemplateV4 {
    #[allow(clippy::too_many_arguments)]
    fn seal(
        sources: MaterializedPinnedResidentCanonicalSourcesV1,
        base_timeframe: CanonicalTimeframe,
        profile: FeatureProfile,
        row_count: usize,
        budget_rows: usize,
        routes: Vec<ResidentFeaturePlanRouteTemplateV4>,
        normalization_enabled: bool,
        normalization_implementation_sha256: [u8; 32],
        normalization_exact_math_authority: String,
        planned_routes: &[ResidentFeatureRouteV3],
        capabilities: &ResidentProducerCapabilityManifestV3,
    ) -> AnyResult<Self> {
        ensure!(
            !routes.is_empty() && routes.len() == planned_routes.len(),
            "resident identity template route census drifted"
        );
        let dataset_recipe_sha256 = derive_dataset_recipe_sha256_v4(
            &sources,
            base_timeframe,
            profile,
            row_count,
            budget_rows,
        )?;
        let feature_plan_schema_sha256 = derive_feature_plan_schema_sha256_v4(
            &routes,
            planned_routes,
            capabilities,
            normalization_enabled,
        )?;
        let route_plan_sha256 = derive_route_plan_sha256_v4(planned_routes)?;
        Ok(Self {
            sources,
            routes,
            normalization_enabled,
            normalization_implementation_sha256,
            normalization_exact_math_authority,
            dataset_recipe_sha256,
            feature_plan_schema_sha256,
            route_plan_sha256,
        })
    }

    pub(crate) const fn dataset_recipe_sha256(&self) -> [u8; 32] {
        self.dataset_recipe_sha256
    }

    pub(crate) const fn feature_plan_schema_sha256(&self) -> [u8; 32] {
        self.feature_plan_schema_sha256
    }

    pub(crate) const fn route_plan_sha256(&self) -> [u8; 32] {
        self.route_plan_sha256
    }

    pub(crate) const fn resident_sources(&self) -> &MaterializedPinnedResidentCanonicalSourcesV1 {
        &self.sources
    }

    pub(crate) fn finalize_after_normalization_v4(
        self,
        normalization_fit_sha256: [u8; 32],
    ) -> AnyResult<FinalizedResidentFeatureIdentityV4> {
        ensure!(
            normalization_fit_sha256 != [0; 32],
            "runtime normalization fit digest is zero"
        );
        let mut nodes = Vec::new();
        let mut bindings = Vec::new();
        let mut source_node_ids = Vec::new();
        for source in self.sources.all_sources() {
            let binding = source.binding();
            let source_node_id = binding.source_node_id().to_owned();
            let source_token = binding.dataset_identity().to_path_component();
            let mut outputs = ["open", "high", "low", "close"]
                .into_iter()
                .map(|field| FeatureOutputV1::f64(format!("physical:{source_token}:{field}"), 1))
                .collect::<Result<Vec<_>, _>>()?;
            if source.frame().ohlcv().volume.is_some() {
                outputs.push(FeatureOutputV1::f64(
                    format!("physical:{source_token}:volume"),
                    1,
                )?);
            }
            nodes.push(FeatureNodeV1::source(
                source_node_id.clone(),
                binding.dataset_identity().clone(),
                "neoethos.ohlcv.f64-ms.v1",
                1,
                outputs,
                source.source_binding_sha256(),
            )?);
            source_node_ids.push(source_node_id);
            bindings.push(binding.clone());
        }
        let base_source_node_id = self.sources.base().binding().source_node_id().to_owned();
        let all_source_node_ids = source_node_ids.clone();
        let mut route_node_ids = Vec::with_capacity(self.routes.len());
        let mut final_outputs = Vec::with_capacity(self.routes.len());
        for route in self.routes {
            let output_name = if self.normalization_enabled {
                format!("pre-normalize:{}", route.feature_name)
            } else {
                route.feature_name.clone()
            };
            let inputs = if route.producer == ResidentFeatureProducerV3::HigherTimeframeAlignment {
                all_source_node_ids.clone()
            } else {
                vec![base_source_node_id.clone()]
            };
            let parameters = route
                .typed_parameters
                .into_iter()
                .map(ResidentCanonicalParameterV4::into_feature_parameter_v1)
                .collect::<AnyResult<Vec<_>>>()?;
            let semantic_source_hash = derive_route_semantic_source_sha256_v4(
                &route.route_domain,
                &route.exact_math_authority,
                route.route_receipt_sha256,
            )?;
            let operation = if route.producer == ResidentFeatureProducerV3::HigherTimeframeAlignment
            {
                FeatureOperationTagV1::HigherTimeframeAlignment
            } else {
                FeatureOperationTagV1::Indicator
            };
            nodes.push(FeatureNodeV1::transform(
                route.route_node_id.clone(),
                operation,
                route.semantic_version,
                inputs,
                vec![FeatureOutputV1::f64(output_name, route.semantic_version)?],
                parameters,
                route.implementation_sha256,
                semantic_source_hash,
                None,
            )?);
            route_node_ids.push(route.route_node_id);
            final_outputs.push(route.feature_name);
        }
        if self.normalization_enabled {
            let semantic_source_hash = derive_route_semantic_source_sha256_v4(
                "neoethos.data.resident-robust-normalization.v2",
                &self.normalization_exact_math_authority,
                self.feature_plan_schema_sha256,
            )?;
            let outputs = final_outputs
                .iter()
                .map(|name| FeatureOutputV1::f64(name.clone(), 2))
                .collect::<Result<Vec<_>, _>>()?;
            nodes.push(FeatureNodeV1::transform(
                "normalization:resident-robust-f64-v2",
                FeatureOperationTagV1::Normalization,
                2,
                route_node_ids,
                outputs,
                vec![FeatureParameterV1::hash(
                    "pre_fit_feature_plan_schema_sha256",
                    self.feature_plan_schema_sha256,
                )?],
                self.normalization_implementation_sha256,
                semantic_source_hash,
                Some(normalization_fit_sha256),
            )?);
        } else {
            ensure!(
                normalization_fit_sha256 == canonical_disabled_normalization_fit_sha256_v4(),
                "normalization node is forbidden when disabled"
            );
        }
        let feature_plan = FeaturePlanV1::new(nodes, final_outputs)?;
        let source_provenance = DatasetFeatureArtifactProvenanceV1::new(&feature_plan, bindings)?;
        Ok(FinalizedResidentFeatureIdentityV4 {
            feature_plan,
            source_provenance,
            resident_sources: self.sources,
        })
    }
}

#[derive(Debug)]
pub(crate) struct FinalizedResidentFeatureIdentityV4 {
    feature_plan: FeaturePlanV1,
    source_provenance: DatasetFeatureArtifactProvenanceV1,
    resident_sources: MaterializedPinnedResidentCanonicalSourcesV1,
}

impl FinalizedResidentFeatureIdentityV4 {
    pub(crate) fn into_parts(
        self,
    ) -> (
        FeaturePlanV1,
        DatasetFeatureArtifactProvenanceV1,
        MaterializedPinnedResidentCanonicalSourcesV1,
    ) {
        (
            self.feature_plan,
            self.source_provenance,
            self.resident_sources,
        )
    }
}

fn canonical_disabled_normalization_fit_sha256_v4() -> [u8; 32] {
    resident_robust_normalization_disabled_fit_sha256_v2()
}

#[derive(Debug, Default)]
pub(crate) struct ResidentColumnSchemaAssemblerV4 {
    drafts: Vec<ResidentProducerDraftV4>,
    feature_names: BTreeSet<String>,
    capabilities_by_producer: BTreeSet<ResidentFeatureProducerV3>,
}

impl ResidentColumnSchemaAssemblerV4 {
    pub(crate) fn append(
        &mut self,
        draft: ResidentProducerDraftV4,
    ) -> Result<(), ResidentFeatureRecipeErrorV4> {
        let index = self.drafts.len();
        let expected = RESIDENT_COLUMN_SCHEMA_ORDER_V4.get(index).copied().ok_or(
            ResidentFeatureRecipeErrorV4::ProducerOrderMismatch {
                index,
                expected: ResidentFeatureProducerV3::HigherTimeframeAlignment,
                actual: draft.producer,
            },
        )?;
        if draft.producer != expected {
            return Err(ResidentFeatureRecipeErrorV4::ProducerOrderMismatch {
                index,
                expected,
                actual: draft.producer,
            });
        }
        if self.capabilities_by_producer.contains(&draft.producer) {
            return Err(ResidentFeatureRecipeErrorV4::DuplicateCapability(
                draft.producer,
            ));
        }
        for route in &draft.routes {
            if self.feature_names.contains(&route.feature_name) {
                return Err(ResidentFeatureRecipeErrorV4::DuplicateFeatureName(
                    route.feature_name.clone(),
                ));
            }
        }
        self.capabilities_by_producer.insert(draft.producer);
        self.feature_names
            .extend(draft.routes.iter().map(|route| route.feature_name.clone()));
        self.drafts.push(draft);
        Ok(())
    }

    pub(crate) fn seal(
        self,
        transform_capabilities: ResidentTransformCapabilityDraftV4,
    ) -> Result<SealedResidentColumnSchemaV4, ResidentFeatureRecipeErrorV4> {
        if self.drafts.len() != RESIDENT_COLUMN_SCHEMA_ORDER_V4.len() {
            let present = self
                .drafts
                .iter()
                .map(|draft| draft.producer)
                .collect::<BTreeSet<_>>();
            let missing = RESIDENT_COLUMN_SCHEMA_ORDER_V4
                .iter()
                .copied()
                .filter(|producer| !present.contains(producer))
                .collect();
            return Err(ResidentFeatureRecipeErrorV4::MissingColumnProducers { missing });
        }
        let mut routes = Vec::new();
        let mut plan_route_templates = Vec::new();
        let mut producer_batches = Vec::new();
        let mut capabilities_by_producer = BTreeMap::new();
        let mut global_first_column = 0_usize;
        for draft in self.drafts {
            let ResidentProducerDraftV4 {
                producer,
                semantic_version,
                routes: local_routes,
                batches,
                capability,
            } = draft;
            let producer_first_column = global_first_column;
            let implementation_sha256 = capability.implementation_sha256();
            let exact_math_authority = capability.exact_math_authority().to_owned();
            for route in local_routes {
                let global_column = u64::try_from(global_first_column)
                    .map_err(|_| ResidentFeatureRecipeErrorV4::ExtentOverflow("global ordinal"))?;
                let parameter_tuple_sha256 =
                    derive_parameter_tuple_sha256_v4(&route.typed_parameters)?;
                let route_receipt = derive_route_receipt_sha256_v4(
                    global_column,
                    producer,
                    semantic_version,
                    &route,
                    parameter_tuple_sha256,
                    implementation_sha256,
                    &exact_math_authority,
                )?;
                let route_id =
                    derive_route_id_v4(global_column, producer, &route.feature_name, route_receipt);
                plan_route_templates.push(ResidentFeaturePlanRouteTemplateV4 {
                    producer,
                    semantic_version,
                    feature_name: route.feature_name.clone(),
                    route_node_id: route_id.clone(),
                    typed_parameters: route.typed_parameters.clone(),
                    route_domain: route.route_domain.to_owned(),
                    implementation_sha256,
                    exact_math_authority: exact_math_authority.clone(),
                    route_receipt_sha256: route_receipt,
                });
                routes.push(ResidentFeatureRouteV3::new(
                    global_column,
                    route.feature_name,
                    producer,
                    route.indicator_id,
                    route.output_id,
                    route.stage,
                    route.swept_period,
                    parameter_tuple_sha256,
                    route_id,
                    route_receipt,
                )?);
                global_first_column = global_first_column.checked_add(1).ok_or(
                    ResidentFeatureRecipeErrorV4::ExtentOverflow("global feature count"),
                )?;
            }
            for batch in batches {
                producer_batches.push(ResolvedResidentProducerBatchMemoryV4 {
                    producer,
                    first_column: producer_first_column
                        .checked_add(batch.local_first_column)
                        .ok_or(ResidentFeatureRecipeErrorV4::ExtentOverflow(
                            "global producer batch start",
                        ))?,
                    column_count: batch.column_count,
                    additional_retained_bytes: batch.additional_retained_bytes,
                    scratch_bytes: batch.scratch_bytes,
                });
            }
            if capabilities_by_producer
                .insert(producer, capability)
                .is_some()
            {
                return Err(ResidentFeatureRecipeErrorV4::DuplicateCapability(producer));
            }
        }
        for capability in transform_capabilities.capabilities {
            let producer = capability.producer();
            if capabilities_by_producer
                .insert(producer, capability)
                .is_some()
            {
                return Err(ResidentFeatureRecipeErrorV4::DuplicateCapability(producer));
            }
        }
        let mut ordered_capabilities = Vec::with_capacity(ResidentFeatureProducerV3::ALL.len());
        for producer in ResidentFeatureProducerV3::ALL {
            ordered_capabilities.push(
                capabilities_by_producer
                    .remove(&producer)
                    .ok_or(ResidentFeatureRecipeErrorV4::MissingCapability(producer))?,
            );
        }
        let capability_manifest = ResidentProducerCapabilityManifestV3::seal(ordered_capabilities)?;
        Ok(SealedResidentColumnSchemaV4 {
            routes,
            plan_route_templates,
            producer_batches,
            capability_manifest,
        })
    }
}

fn derive_dataset_recipe_sha256_v4(
    sources: &MaterializedPinnedResidentCanonicalSourcesV1,
    base_timeframe: CanonicalTimeframe,
    profile: FeatureProfile,
    row_count: usize,
    budget_rows: usize,
) -> AnyResult<[u8; 32]> {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.data.resident-dataset-recipe.v4\0");
    let receipt = sources.receipt().to_json_bytes()?;
    update_bytes(&mut hasher, &receipt, "canonical series receipt")?;
    update_bytes(
        &mut hasher,
        base_timeframe.to_string().as_bytes(),
        "base timeframe",
    )?;
    hasher.update([feature_profile_tag_v4(profile)]);
    update_usize(&mut hasher, row_count, "dataset recipe rows")?;
    update_usize(&mut hasher, budget_rows, "dataset recipe budget rows")?;
    update_usize(&mut hasher, sources.source_count(), "dataset source count")?;
    for source in sources.all_sources() {
        hasher.update(source.source_binding_sha256());
    }
    Ok(hasher.finalize().into())
}

fn derive_feature_plan_schema_sha256_v4(
    route_templates: &[ResidentFeaturePlanRouteTemplateV4],
    planned_routes: &[ResidentFeatureRouteV3],
    capabilities: &ResidentProducerCapabilityManifestV3,
    normalization_enabled: bool,
) -> AnyResult<[u8; 32]> {
    ensure!(
        route_templates.len() == planned_routes.len(),
        "feature-plan schema route/template count mismatch"
    );
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.data.resident-feature-plan-schema.v4\0");
    hasher.update(RESIDENT_FEATURE_SCHEMA_VERSION_V4.to_le_bytes());
    hasher.update([u8::from(normalization_enabled)]);
    update_usize(&mut hasher, planned_routes.len(), "schema route count")?;
    for (template, route) in route_templates.iter().zip(planned_routes) {
        ensure!(
            template.feature_name == route.feature_name()
                && template.producer == route.producer()
                && template.route_receipt_sha256 == route.route_receipt_sha256(),
            "feature-plan schema route/template identity mismatch"
        );
        hasher.update(route.ordinal().to_le_bytes());
        hasher.update([route.producer() as u8]);
        hasher.update(template.semantic_version.to_le_bytes());
        hasher.update(route.canonical_parameter_tuple_sha256());
        hasher.update(route.route_receipt_sha256());
        hasher.update(template.implementation_sha256);
        update_bytes(
            &mut hasher,
            template.route_domain.as_bytes(),
            "feature-plan route domain",
        )?;
        update_bytes(
            &mut hasher,
            template.exact_math_authority.as_bytes(),
            "feature-plan exact math authority",
        )?;
    }
    update_usize(
        &mut hasher,
        capabilities.capabilities().len(),
        "capability count",
    )?;
    for capability in capabilities.capabilities() {
        hasher.update([capability.producer() as u8]);
        update_bytes(
            &mut hasher,
            capability.implementation_id().as_bytes(),
            "capability implementation id",
        )?;
        hasher.update(capability.implementation_sha256());
        update_bytes(
            &mut hasher,
            capability.exact_math_authority().as_bytes(),
            "capability exact math authority",
        )?;
    }
    Ok(hasher.finalize().into())
}

fn derive_route_plan_sha256_v4(routes: &[ResidentFeatureRouteV3]) -> AnyResult<[u8; 32]> {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.data.resident-route-plan.v4\0");
    update_usize(&mut hasher, routes.len(), "route-plan count")?;
    for route in routes {
        hasher.update(route.ordinal().to_le_bytes());
        hasher.update([route.producer() as u8]);
        hasher.update(route.route_receipt_sha256());
    }
    Ok(hasher.finalize().into())
}

fn derive_route_semantic_source_sha256_v4(
    route_domain: &str,
    exact_math_authority: &str,
    route_receipt_sha256: [u8; 32],
) -> AnyResult<[u8; 32]> {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.data.resident-route-semantic-source.v4\0");
    update_bytes(
        &mut hasher,
        route_domain.as_bytes(),
        "route semantic domain",
    )?;
    update_bytes(
        &mut hasher,
        exact_math_authority.as_bytes(),
        "route exact math authority",
    )?;
    hasher.update(route_receipt_sha256);
    Ok(hasher.finalize().into())
}

fn feature_profile_tag_v4(profile: FeatureProfile) -> u8 {
    match profile {
        FeatureProfile::Standard => 0,
        FeatureProfile::Full => 1,
        FeatureProfile::HPC => 2,
        FeatureProfile::Adaptive => 3,
    }
}

fn derive_parameter_tuple_sha256_v4(
    typed_parameters: &[ResidentCanonicalParameterV4],
) -> Result<[u8; 32], ResidentFeatureRecipeErrorV4> {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.data.resident-parameter-tuple.v4\0");
    update_usize(&mut hasher, typed_parameters.len(), "parameter count")?;
    for parameter in typed_parameters {
        update_bytes(&mut hasher, parameter.name.as_bytes(), "parameter name")?;
        match &parameter.value {
            ResidentCanonicalParameterValueV4::Bool(value) => {
                hasher.update([0, u8::from(*value)]);
            }
            ResidentCanonicalParameterValueV4::U64(value) => {
                hasher.update([1]);
                hasher.update(value.to_le_bytes());
            }
            ResidentCanonicalParameterValueV4::I64(value) => {
                hasher.update([2]);
                hasher.update(value.to_le_bytes());
            }
            ResidentCanonicalParameterValueV4::F64Bits(value) => {
                hasher.update([3]);
                hasher.update(value.to_le_bytes());
            }
            ResidentCanonicalParameterValueV4::Text(value) => {
                hasher.update([4]);
                update_bytes(&mut hasher, value.as_bytes(), "parameter text")?;
            }
            ResidentCanonicalParameterValueV4::Hash(value) => {
                hasher.update([5]);
                hasher.update(value);
            }
        }
    }
    Ok(hasher.finalize().into())
}

fn semantic_text_is_blank(value: &str) -> bool {
    value.trim().is_empty()
}

fn derive_route_id_v4(
    global_column: u64,
    producer: ResidentFeatureProducerV3,
    feature_name: &str,
    route_receipt_sha256: [u8; 32],
) -> String {
    format!(
        "neoethos.data.resident-feature-schema.v4:{}:{global_column}:{feature_name}:{}",
        producer.as_str(),
        sha256_hex_v4(route_receipt_sha256)
    )
}

fn sha256_hex_v4(hash: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(64);
    for byte in hash {
        encoded.push(HEX[usize::from(byte >> 4)] as char);
        encoded.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    encoded
}

#[allow(clippy::too_many_arguments)]
fn derive_route_receipt_sha256_v4(
    global_column: u64,
    producer: ResidentFeatureProducerV3,
    semantic_version: u32,
    route: &ResidentRouteDraftV4,
    parameter_hash: [u8; 32],
    implementation_hash: [u8; 32],
    exact_math_authority: &str,
) -> Result<[u8; 32], ResidentFeatureRecipeErrorV4> {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.data.resident-feature-route-receipt.v4\0");
    hasher.update(RESIDENT_FEATURE_SCHEMA_VERSION_V4.to_le_bytes());
    hasher.update(global_column.to_le_bytes());
    hasher.update([producer as u8]);
    hasher.update(semantic_version.to_le_bytes());
    update_bytes(&mut hasher, route.route_domain.as_bytes(), "route domain")?;
    update_bytes(&mut hasher, route.feature_name.as_bytes(), "feature name")?;
    update_optional_text(&mut hasher, route.indicator_id.as_deref(), "indicator id")?;
    update_optional_text(&mut hasher, route.output_id.as_deref(), "output id")?;
    hasher.update([stage_tag(route.stage)]);
    match route.swept_period {
        Some(period) => {
            hasher.update([1]);
            hasher.update(period.to_le_bytes());
        }
        None => hasher.update([0]),
    }
    hasher.update(parameter_hash);
    hasher.update(implementation_hash);
    update_bytes(
        &mut hasher,
        exact_math_authority.as_bytes(),
        "exact math authority",
    )?;
    Ok(hasher.finalize().into())
}

fn stage_tag(stage: ResidentFeatureStageV3) -> u8 {
    match stage {
        ResidentFeatureStageV3::Base => 0,
        ResidentFeatureStageV3::Historical => 1,
        ResidentFeatureStageV3::Extended => 2,
        ResidentFeatureStageV3::Derived => 3,
        ResidentFeatureStageV3::HigherTimeframeAligned => 4,
        ResidentFeatureStageV3::Normalized => 5,
    }
}

fn update_optional_text(
    hasher: &mut Sha256,
    value: Option<&str>,
    field: &'static str,
) -> Result<(), ResidentFeatureRecipeErrorV4> {
    match value {
        Some(value) => {
            hasher.update([1]);
            update_bytes(hasher, value.as_bytes(), field)
        }
        None => {
            hasher.update([0]);
            Ok(())
        }
    }
}

fn update_bytes(
    hasher: &mut Sha256,
    bytes: &[u8],
    field: &'static str,
) -> Result<(), ResidentFeatureRecipeErrorV4> {
    update_usize(hasher, bytes.len(), field)?;
    hasher.update(bytes);
    Ok(())
}

fn update_usize(
    hasher: &mut Sha256,
    value: usize,
    field: &'static str,
) -> Result<(), ResidentFeatureRecipeErrorV4> {
    let value =
        u64::try_from(value).map_err(|_| ResidentFeatureRecipeErrorV4::ExtentOverflow(field))?;
    hasher.update(value.to_le_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capability(producer: ResidentFeatureProducerV3) -> ResidentProducerCapabilityV3 {
        ResidentProducerCapabilityV3::new(
            producer,
            format!("recipe-v4-test-{}", producer.as_str()),
            [producer as u8 + 1; 32],
            "recipe-v4 synthetic exact-math authority",
        )
        .expect("synthetic capability")
    }

    fn route(name: &str) -> ResidentRouteDraftV4 {
        ResidentRouteDraftV4::from_typed_parts(
            name,
            Some("recipe_v4_test"),
            Some(name),
            ResidentFeatureStageV3::Derived,
            None,
            vec![
                ResidentCanonicalParameterV4::from_typed_value(
                    "window",
                    ResidentCanonicalParameterValueV4::U64(1),
                )
                .expect("typed parameter"),
            ],
            "neoethos.data.recipe-v4.synthetic-route",
        )
        .expect("synthetic route")
    }

    fn one_column_draft(producer: ResidentFeatureProducerV3) -> ResidentProducerDraftV4 {
        ResidentProducerDraftV4::from_owner_preflight(
            producer,
            1,
            vec![route(&format!("{}_column", producer.as_str()))],
            vec![ResidentProducerBatchDraftV4::from_owner_preflight(
                0, 1, 0, 0,
            )],
            capability(producer),
        )
        .expect("one-column producer draft")
    }

    fn transform_capabilities() -> ResidentTransformCapabilityDraftV4 {
        ResidentTransformCapabilityDraftV4::from_owner_capabilities(
            capability(ResidentFeatureProducerV3::RobustNormalization),
            capability(ResidentFeatureProducerV3::CanonicalContentSha256),
            capability(ResidentFeatureProducerV3::FeatureMajorToBarMajor),
        )
        .expect("three exact transform capabilities")
    }

    #[test]
    fn valid_seven_producer_seal_assigns_monotonic_global_routes_and_batches() {
        let mut assembler = ResidentColumnSchemaAssemblerV4::default();
        for producer in RESIDENT_COLUMN_SCHEMA_ORDER_V4 {
            assembler
                .append(one_column_draft(producer))
                .expect("schema-ordered producer");
        }
        let sealed = assembler
            .seal(transform_capabilities())
            .expect("complete column schema");
        assert_eq!(sealed.routes.len(), RESIDENT_COLUMN_SCHEMA_ORDER_V4.len());
        assert_eq!(
            sealed.producer_batches.len(),
            RESIDENT_COLUMN_SCHEMA_ORDER_V4.len()
        );
        for (global_column, (route, batch)) in sealed
            .routes
            .iter()
            .zip(&sealed.producer_batches)
            .enumerate()
        {
            assert_eq!(route.ordinal(), global_column as u64);
            assert_eq!(batch.first_column(), global_column);
            assert_eq!(batch.column_count(), 1);
            assert_eq!(
                batch.producer(),
                RESIDENT_COLUMN_SCHEMA_ORDER_V4[global_column]
            );
        }
    }

    #[test]
    fn producer_reordering_and_transform_pseudo_columns_fail_closed() {
        let mut reordered = ResidentColumnSchemaAssemblerV4::default();
        assert!(matches!(
            reordered.append(one_column_draft(ResidentFeatureProducerV3::ClassicTa)),
            Err(ResidentFeatureRecipeErrorV4::ProducerOrderMismatch { .. })
        ));

        let mut transform = ResidentColumnSchemaAssemblerV4::default();
        assert!(matches!(
            transform.append(one_column_draft(
                ResidentFeatureProducerV3::RobustNormalization
            )),
            Err(ResidentFeatureRecipeErrorV4::ProducerOrderMismatch { .. })
        ));
    }

    #[test]
    fn missing_producers_and_batch_gaps_or_overlaps_fail_closed() {
        let mut incomplete = ResidentColumnSchemaAssemblerV4::default();
        incomplete
            .append(one_column_draft(ResidentFeatureProducerV3::Smc))
            .expect("first producer");
        assert!(matches!(
            incomplete.seal(transform_capabilities()),
            Err(ResidentFeatureRecipeErrorV4::MissingColumnProducers { .. })
        ));

        for batches in [
            vec![ResidentProducerBatchDraftV4::from_owner_preflight(
                1, 1, 0, 0,
            )],
            vec![
                ResidentProducerBatchDraftV4::from_owner_preflight(0, 1, 0, 0),
                ResidentProducerBatchDraftV4::from_owner_preflight(0, 1, 0, 0),
            ],
        ] {
            assert!(matches!(
                ResidentProducerDraftV4::from_owner_preflight(
                    ResidentFeatureProducerV3::Smc,
                    1,
                    vec![route("smc_a"), route("smc_b")],
                    batches,
                    capability(ResidentFeatureProducerV3::Smc),
                ),
                Err(ResidentFeatureRecipeErrorV4::InvalidBatchExtent { .. })
            ));
        }
    }

    #[test]
    fn guessed_zero_hash_parameter_is_rejected_before_draft() {
        assert!(matches!(
            ResidentCanonicalParameterV4::from_typed_value(
                "source_binding_sha256",
                ResidentCanonicalParameterValueV4::Hash([0; 32]),
            ),
            Err(ResidentFeatureRecipeErrorV4::InvalidParameter(name))
                if name == "source_binding_sha256"
        ));
    }

    #[test]
    fn route_identity_changes_when_full_semantic_fragment_changes() {
        let first = derive_route_id_v4(7, ResidentFeatureProducerV3::Smc, "smc_route", [0x11; 32]);
        let second = derive_route_id_v4(7, ResidentFeatureProducerV3::Smc, "smc_route", [0x22; 32]);
        assert_ne!(first, second);
        assert!(first.ends_with(&"11".repeat(32)));
        assert!(second.ends_with(&"22".repeat(32)));
    }

    #[test]
    fn ten_entry_capability_manifest_uses_contract_order() {
        let mut assembler = ResidentColumnSchemaAssemblerV4::default();
        for producer in RESIDENT_COLUMN_SCHEMA_ORDER_V4 {
            assembler
                .append(one_column_draft(producer))
                .expect("schema-ordered producer");
        }
        let sealed = assembler
            .seal(transform_capabilities())
            .expect("complete recipe capability manifest");
        assert_eq!(
            sealed
                .capability_manifest()
                .capabilities()
                .iter()
                .map(ResidentProducerCapabilityV3::producer)
                .collect::<Vec<_>>(),
            ResidentFeatureProducerV3::ALL
        );
    }

    #[test]
    fn missing_or_mislabelled_transform_capability_fails_closed() {
        assert!(matches!(
            ResidentTransformCapabilityDraftV4::from_owner_capabilities(
                capability(ResidentFeatureProducerV3::CanonicalContentSha256),
                capability(ResidentFeatureProducerV3::CanonicalContentSha256),
                capability(ResidentFeatureProducerV3::FeatureMajorToBarMajor),
            ),
            Err(
                ResidentFeatureRecipeErrorV4::TransformCapabilityProducerMismatch {
                    expected: ResidentFeatureProducerV3::RobustNormalization,
                    actual: ResidentFeatureProducerV3::CanonicalContentSha256,
                }
            )
        ));
    }

    #[test]
    fn whitespace_only_semantic_fields_fail_closed() {
        assert!(matches!(
            ResidentCanonicalParameterV4::from_typed_value(
                "   ",
                ResidentCanonicalParameterValueV4::U64(1),
            ),
            Err(ResidentFeatureRecipeErrorV4::EmptyField(
                "canonical parameter name"
            ))
        ));
        assert!(matches!(
            ResidentCanonicalParameterV4::from_typed_value(
                "label",
                ResidentCanonicalParameterValueV4::Text(" \t ".to_owned()),
            ),
            Err(ResidentFeatureRecipeErrorV4::InvalidParameter(name)) if name == "label"
        ));

        for (feature_name, indicator_id, output_id, route_domain) in [
            ("   ", Some("indicator"), Some("output"), "route-domain"),
            ("feature", Some("   "), Some("output"), "route-domain"),
            ("feature", Some("indicator"), Some("   "), "route-domain"),
            ("feature", Some("indicator"), Some("output"), " \t "),
        ] {
            assert!(matches!(
                ResidentRouteDraftV4::from_typed_parts(
                    feature_name,
                    indicator_id,
                    output_id,
                    ResidentFeatureStageV3::Derived,
                    None,
                    Vec::new(),
                    route_domain,
                ),
                Err(ResidentFeatureRecipeErrorV4::EmptyField(_))
            ));
        }
    }
}
