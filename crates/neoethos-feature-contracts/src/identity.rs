use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::error::Error;
use std::fmt;

use neoethos_dataset_contracts::{BarTimestampConvention, CanonicalDatasetIdentity};
use sha2::{Digest, Sha256};

const FEATURE_PLAN_DOMAIN_V1: &[u8] = b"neoethos.feature-plan.identity\0";
const FEATURE_PLAN_VERSION_V1: u16 = 1;
const ARTIFACT_PROVENANCE_DOMAIN_V1: &[u8] = b"neoethos.dataset-feature-artifact-provenance.v1\0";
const ARTIFACT_PROVENANCE_VERSION_V1: u16 = 1;
const MAX_TEXT_BYTES: usize = 4 * 1024;
const MAX_CANONICAL_ITEMS: usize = 1_000_000;
const MAX_CANONICAL_BYTES: usize = 64 * 1024 * 1024;
const MAX_NESTED_CANONICAL_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeatureContractError {
    detail: String,
}

impl FeatureContractError {
    pub(crate) fn new(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
        }
    }
}

impl fmt::Display for FeatureContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.detail)
    }
}

impl Error for FeatureContractError {}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FeatureOperationTagV1 {
    Source = 1,
    Indicator = 2,
    HigherTimeframeAlignment = 4,
    Normalization = 5,
    CrossPairAlignment = 6,
    Derived = 7,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeatureOutputV1 {
    name: String,
    validity_semantic_version: u32,
}

impl FeatureOutputV1 {
    pub fn f64(
        name: impl Into<String>,
        validity_semantic_version: u32,
    ) -> Result<Self, FeatureContractError> {
        let name = name.into();
        validate_text("feature output name", &name)?;
        require_positive_version("feature-output validity", validity_semantic_version)?;
        Ok(Self {
            name,
            validity_semantic_version,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn validity_semantic_version(&self) -> u32 {
        self.validity_semantic_version
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FeatureParameterValueV1 {
    Bool(bool),
    U64(u64),
    I64(i64),
    F64Bits(u64),
    Text(String),
    Hash([u8; 32]),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeatureParameterV1 {
    name: String,
    value: FeatureParameterValueV1,
}

impl FeatureParameterV1 {
    pub fn f64(name: impl Into<String>, value: f64) -> Result<Self, FeatureContractError> {
        if !value.is_finite() {
            return Err(FeatureContractError::new(
                "feature-plan f64 parameter must be finite",
            ));
        }
        if value.to_bits() == (-0.0_f64).to_bits() {
            return Err(FeatureContractError::new(
                "feature-plan f64 parameter must not be negative zero",
            ));
        }
        Self::new(name, FeatureParameterValueV1::F64Bits(value.to_bits()))
    }

    pub fn bool(name: impl Into<String>, value: bool) -> Result<Self, FeatureContractError> {
        Self::new(name, FeatureParameterValueV1::Bool(value))
    }

    pub fn u64(name: impl Into<String>, value: u64) -> Result<Self, FeatureContractError> {
        Self::new(name, FeatureParameterValueV1::U64(value))
    }

    pub fn i64(name: impl Into<String>, value: i64) -> Result<Self, FeatureContractError> {
        Self::new(name, FeatureParameterValueV1::I64(value))
    }

    pub fn text(
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<Self, FeatureContractError> {
        let value = value.into();
        validate_text("feature-plan text parameter", &value)?;
        Self::new(name, FeatureParameterValueV1::Text(value))
    }

    pub fn hash(name: impl Into<String>, value: [u8; 32]) -> Result<Self, FeatureContractError> {
        Self::new(name, FeatureParameterValueV1::Hash(value))
    }

    fn new(
        name: impl Into<String>,
        value: FeatureParameterValueV1,
    ) -> Result<Self, FeatureContractError> {
        let name = name.into();
        validate_text("feature parameter name", &name)?;
        Ok(Self { name, value })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SourceContractV1 {
    dataset_identity: CanonicalDatasetIdentity,
    physical_schema_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeatureNodeV1 {
    id: String,
    operation: FeatureOperationTagV1,
    semantic_version: u32,
    inputs: Vec<String>,
    outputs: Vec<FeatureOutputV1>,
    parameters: Vec<FeatureParameterV1>,
    formula_manifest_hash: [u8; 32],
    semantic_source_hash: [u8; 32],
    fitted_state_hash: Option<[u8; 32]>,
    source: Option<SourceContractV1>,
}

impl FeatureNodeV1 {
    pub fn source(
        id: impl Into<String>,
        dataset_identity: CanonicalDatasetIdentity,
        physical_schema_id: impl Into<String>,
        semantic_version: u32,
        outputs: Vec<FeatureOutputV1>,
        semantic_source_hash: [u8; 32],
    ) -> Result<Self, FeatureContractError> {
        let physical_schema_id = physical_schema_id.into();
        validate_text("source physical schema id", &physical_schema_id)?;
        Self::build(
            id,
            FeatureOperationTagV1::Source,
            semantic_version,
            Vec::new(),
            outputs,
            Vec::new(),
            [0; 32],
            semantic_source_hash,
            None,
            Some(SourceContractV1 {
                dataset_identity,
                physical_schema_id,
            }),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn transform(
        id: impl Into<String>,
        operation: FeatureOperationTagV1,
        semantic_version: u32,
        inputs: Vec<String>,
        outputs: Vec<FeatureOutputV1>,
        parameters: Vec<FeatureParameterV1>,
        formula_manifest_hash: [u8; 32],
        semantic_source_hash: [u8; 32],
        fitted_state_hash: Option<[u8; 32]>,
    ) -> Result<Self, FeatureContractError> {
        if operation == FeatureOperationTagV1::Source {
            return Err(FeatureContractError::new(
                "a transform cannot use the source operation tag",
            ));
        }
        Self::build(
            id,
            operation,
            semantic_version,
            inputs,
            outputs,
            parameters,
            formula_manifest_hash,
            semantic_source_hash,
            fitted_state_hash,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        id: impl Into<String>,
        operation: FeatureOperationTagV1,
        semantic_version: u32,
        inputs: Vec<String>,
        outputs: Vec<FeatureOutputV1>,
        parameters: Vec<FeatureParameterV1>,
        formula_manifest_hash: [u8; 32],
        semantic_source_hash: [u8; 32],
        fitted_state_hash: Option<[u8; 32]>,
        source: Option<SourceContractV1>,
    ) -> Result<Self, FeatureContractError> {
        let id = id.into();
        validate_text("feature node id", &id)?;
        require_positive_version("feature-node semantic", semantic_version)?;
        if outputs.is_empty() {
            return Err(FeatureContractError::new(format!(
                "feature node `{id}` must declare at least one output"
            )));
        }
        validate_unique_texts("feature-node input", inputs.iter().map(String::as_str))?;
        validate_unique_texts(
            "feature-node output",
            outputs.iter().map(|output| output.name.as_str()),
        )?;
        validate_unique_texts(
            "feature-node parameter",
            parameters.iter().map(|parameter| parameter.name.as_str()),
        )?;
        Ok(Self {
            id,
            operation,
            semantic_version,
            inputs,
            outputs,
            parameters,
            formula_manifest_hash,
            semantic_source_hash,
            fitted_state_hash,
            source,
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub const fn operation(&self) -> FeatureOperationTagV1 {
        self.operation
    }

    pub const fn formula_manifest_hash(&self) -> [u8; 32] {
        self.formula_manifest_hash
    }

    pub const fn semantic_source_hash(&self) -> [u8; 32] {
        self.semantic_source_hash
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FeaturePlanIdentityV1([u8; 32]);

impl FeaturePlanIdentityV1 {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        hex(&self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeaturePlanV1 {
    nodes: Vec<FeatureNodeV1>,
    final_outputs: Vec<String>,
    canonical_bytes: Vec<u8>,
    identity: FeaturePlanIdentityV1,
}

impl FeaturePlanV1 {
    pub fn new(
        nodes: Vec<FeatureNodeV1>,
        final_outputs: Vec<String>,
    ) -> Result<Self, FeatureContractError> {
        if nodes.is_empty() {
            return Err(FeatureContractError::new(
                "feature plan must contain at least one node",
            ));
        }
        let nodes = canonical_topological_order(nodes)?;
        validate_plan_outputs(&nodes)?;
        validate_final_outputs(&nodes, &final_outputs)?;
        let canonical_bytes = encode_feature_plan(&nodes, &final_outputs)?;
        let identity = FeaturePlanIdentityV1(Sha256::digest(&canonical_bytes).into());
        Ok(Self {
            nodes,
            final_outputs,
            canonical_bytes,
            identity,
        })
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, FeatureContractError> {
        let mut cursor = Cursor::new(bytes)?;
        cursor.require_exact(FEATURE_PLAN_DOMAIN_V1, "feature-plan domain")?;
        let version = cursor.read_u16("feature-plan version")?;
        if version != FEATURE_PLAN_VERSION_V1 {
            return Err(FeatureContractError::new(format!(
                "unsupported feature-plan version {version}"
            )));
        }
        let node_count = cursor.read_count("feature nodes")?;
        if node_count == 0 {
            return Err(FeatureContractError::new(
                "feature plan must contain at least one node",
            ));
        }
        let mut nodes = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            let id = cursor.read_text("feature node id")?;
            let operation = operation_from_tag(cursor.read_u8("feature operation tag")?)?;
            let semantic_version = cursor.read_u32("feature semantic version")?;
            let source = match cursor.read_u8("source-contract presence")? {
                0 => None,
                1 => {
                    let dataset_bytes = cursor.read_length_prefixed("dataset identity")?;
                    let dataset_identity = CanonicalDatasetIdentity::from_canonical_bytes(
                        dataset_bytes,
                    )
                    .map_err(|error| {
                        FeatureContractError::new(format!(
                            "invalid source dataset identity: {error}"
                        ))
                    })?;
                    let physical_schema_id = cursor.read_text("source physical schema id")?;
                    Some(SourceContractV1 {
                        dataset_identity,
                        physical_schema_id,
                    })
                }
                tag => {
                    return Err(FeatureContractError::new(format!(
                        "invalid source-contract presence tag {tag}"
                    )));
                }
            };
            let inputs = cursor.read_text_vec("feature-node inputs")?;
            let output_count = cursor.read_count("feature-node outputs")?;
            let mut outputs = Vec::with_capacity(output_count);
            for _ in 0..output_count {
                let name = cursor.read_text("feature output name")?;
                let dtype = cursor.read_u8("feature output dtype")?;
                if dtype != 1 {
                    return Err(FeatureContractError::new(format!(
                        "unsupported shared feature dtype tag {dtype}"
                    )));
                }
                let validity_version = cursor.read_u32("feature validity version")?;
                outputs.push(FeatureOutputV1::f64(name, validity_version)?);
            }
            let parameter_count = cursor.read_count("feature-node parameters")?;
            let mut parameters = Vec::with_capacity(parameter_count);
            for _ in 0..parameter_count {
                let name = cursor.read_text("feature parameter name")?;
                let value = match cursor.read_u8("feature parameter value tag")? {
                    1 => match cursor.read_u8("bool parameter")? {
                        0 => FeatureParameterValueV1::Bool(false),
                        1 => FeatureParameterValueV1::Bool(true),
                        tag => {
                            return Err(FeatureContractError::new(format!(
                                "invalid bool parameter tag {tag}"
                            )));
                        }
                    },
                    2 => FeatureParameterValueV1::U64(cursor.read_u64("u64 parameter")?),
                    3 => FeatureParameterValueV1::I64(cursor.read_i64("i64 parameter")?),
                    4 => {
                        let bits = cursor.read_u64("f64 parameter bits")?;
                        let value = f64::from_bits(bits);
                        FeatureParameterV1::f64(name.clone(), value)?.value
                    }
                    5 => FeatureParameterValueV1::Text(cursor.read_text("text parameter")?),
                    6 => FeatureParameterValueV1::Hash(cursor.read_hash("hash parameter")?),
                    tag => {
                        return Err(FeatureContractError::new(format!(
                            "unknown feature parameter value tag {tag}"
                        )));
                    }
                };
                parameters.push(FeatureParameterV1::new(name, value)?);
            }
            let formula_manifest_hash = cursor.read_hash("formula manifest hash")?;
            let semantic_source_hash = cursor.read_hash("semantic source hash")?;
            let fitted_state_hash = match cursor.read_u8("fitted-state presence")? {
                0 => None,
                1 => Some(cursor.read_hash("fitted-state hash")?),
                tag => {
                    return Err(FeatureContractError::new(format!(
                        "invalid fitted-state presence tag {tag}"
                    )));
                }
            };
            let node = match source {
                Some(source) => FeatureNodeV1::source(
                    id,
                    source.dataset_identity,
                    source.physical_schema_id,
                    semantic_version,
                    outputs,
                    semantic_source_hash,
                )?,
                None => FeatureNodeV1::transform(
                    id,
                    operation,
                    semantic_version,
                    inputs,
                    outputs,
                    parameters,
                    formula_manifest_hash,
                    semantic_source_hash,
                    fitted_state_hash,
                )?,
            };
            nodes.push(node);
        }
        let final_outputs = cursor.read_text_vec("feature-plan final outputs")?;
        cursor.require_empty("feature plan")?;
        let plan = Self::new(nodes, final_outputs)?;
        if plan.canonical_bytes != bytes {
            return Err(FeatureContractError::new(
                "feature-plan bytes are not canonical",
            ));
        }
        Ok(plan)
    }

    pub fn nodes(&self) -> &[FeatureNodeV1] {
        &self.nodes
    }

    pub fn final_outputs(&self) -> &[String] {
        &self.final_outputs
    }

    pub const fn identity(&self) -> FeaturePlanIdentityV1 {
        self.identity
    }

    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }

    fn source_contracts(&self) -> BTreeMap<&str, &SourceContractV1> {
        self.nodes
            .iter()
            .filter_map(|node| {
                node.source
                    .as_ref()
                    .map(|source| (node.id.as_str(), source))
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceSegmentV1 {
    row_start: u64,
    row_end: u64,
    timestamp_start_ms: i64,
    timestamp_end_ms: i64,
}

impl SourceSegmentV1 {
    pub fn new(
        row_start: u64,
        row_end: u64,
        timestamp_start_ms: i64,
        timestamp_end_ms: i64,
    ) -> Result<Self, FeatureContractError> {
        if row_start >= row_end {
            return Err(FeatureContractError::new(
                "source segment row range must be non-empty and increasing",
            ));
        }
        if timestamp_start_ms > timestamp_end_ms {
            return Err(FeatureContractError::new(
                "source segment timestamp bounds must be increasing",
            ));
        }
        Ok(Self {
            row_start,
            row_end,
            timestamp_start_ms,
            timestamp_end_ms,
        })
    }

    pub const fn row_start(&self) -> u64 {
        self.row_start
    }

    pub const fn row_end(&self) -> u64 {
        self.row_end
    }

    pub const fn timestamp_start_ms(&self) -> i64 {
        self.timestamp_start_ms
    }

    pub const fn timestamp_end_ms(&self) -> i64 {
        self.timestamp_end_ms
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceArtifactBindingV1 {
    source_node_id: String,
    dataset_identity: CanonicalDatasetIdentity,
    manifest_schema_id: String,
    manifest_hash: [u8; 32],
    generation_id: String,
    vortex_hash: [u8; 32],
    bar_timestamp_convention: BarTimestampConvention,
    segments: Vec<SourceSegmentV1>,
}

impl SourceArtifactBindingV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source_node_id: impl Into<String>,
        dataset_identity: CanonicalDatasetIdentity,
        manifest_schema_id: impl Into<String>,
        manifest_hash: [u8; 32],
        generation_id: impl Into<String>,
        vortex_hash: [u8; 32],
        bar_timestamp_convention: BarTimestampConvention,
        mut segments: Vec<SourceSegmentV1>,
    ) -> Result<Self, FeatureContractError> {
        let source_node_id = source_node_id.into();
        let manifest_schema_id = manifest_schema_id.into();
        let generation_id = generation_id.into();
        validate_text("provenance source node id", &source_node_id)?;
        validate_text("provenance manifest schema id", &manifest_schema_id)?;
        validate_opaque_component("provenance generation id", &generation_id)?;
        if segments.is_empty() {
            return Err(FeatureContractError::new(
                "source artifact binding must contain at least one consumed segment",
            ));
        }
        segments.sort_by_key(|segment| (segment.row_start, segment.row_end));
        for pair in segments.windows(2) {
            if pair[0].row_end > pair[1].row_start {
                return Err(FeatureContractError::new(
                    "source artifact row segments overlap",
                ));
            }
            if pair[0].timestamp_end_ms >= pair[1].timestamp_start_ms {
                return Err(FeatureContractError::new(
                    "source artifact timestamp segments overlap or are out of order",
                ));
            }
        }
        if dataset_identity.bar_timestamp_convention() != bar_timestamp_convention {
            return Err(FeatureContractError::new(
                "source binding bar convention disagrees with its dataset identity",
            ));
        }
        Ok(Self {
            source_node_id,
            dataset_identity,
            manifest_schema_id,
            manifest_hash,
            generation_id,
            vortex_hash,
            bar_timestamp_convention,
            segments,
        })
    }

    pub fn source_node_id(&self) -> &str {
        &self.source_node_id
    }

    pub const fn dataset_identity(&self) -> &CanonicalDatasetIdentity {
        &self.dataset_identity
    }

    pub fn manifest_schema_id(&self) -> &str {
        &self.manifest_schema_id
    }

    pub const fn manifest_hash(&self) -> &[u8; 32] {
        &self.manifest_hash
    }

    pub fn generation_id(&self) -> &str {
        &self.generation_id
    }

    pub const fn vortex_hash(&self) -> &[u8; 32] {
        &self.vortex_hash
    }

    pub const fn bar_timestamp_convention(&self) -> BarTimestampConvention {
        self.bar_timestamp_convention
    }

    pub fn segments(&self) -> &[SourceSegmentV1] {
        &self.segments
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DatasetFeatureArtifactProvenanceIdentityV1([u8; 32]);

impl DatasetFeatureArtifactProvenanceIdentityV1 {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        hex(&self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DatasetFeatureArtifactProvenanceV1 {
    bindings: Vec<SourceArtifactBindingV1>,
    canonical_bytes: Vec<u8>,
    identity: DatasetFeatureArtifactProvenanceIdentityV1,
}

impl DatasetFeatureArtifactProvenanceV1 {
    pub fn new(
        plan: &FeaturePlanV1,
        mut bindings: Vec<SourceArtifactBindingV1>,
    ) -> Result<Self, FeatureContractError> {
        let expected = plan.source_contracts();
        if expected.is_empty() {
            return Err(FeatureContractError::new(
                "feature plan has no source nodes for artifact provenance",
            ));
        }
        bindings.sort_by(|left, right| left.source_node_id.cmp(&right.source_node_id));
        let mut seen = HashSet::with_capacity(bindings.len());
        for binding in &bindings {
            if !seen.insert(binding.source_node_id.as_str()) {
                return Err(FeatureContractError::new(format!(
                    "duplicate provenance binding for `{}`",
                    binding.source_node_id
                )));
            }
            let source = expected
                .get(binding.source_node_id.as_str())
                .ok_or_else(|| {
                    FeatureContractError::new(format!(
                        "unknown provenance source node `{}`",
                        binding.source_node_id
                    ))
                })?;
            if source.dataset_identity != binding.dataset_identity {
                return Err(FeatureContractError::new(format!(
                    "provenance dataset identity does not match source node `{}`",
                    binding.source_node_id
                )));
            }
            if binding.bar_timestamp_convention
                != source.dataset_identity.bar_timestamp_convention()
            {
                return Err(FeatureContractError::new(format!(
                    "provenance bar convention does not match source node `{}`",
                    binding.source_node_id
                )));
            }
        }
        let missing = expected
            .keys()
            .copied()
            .filter(|source| !seen.contains(source))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(FeatureContractError::new(format!(
                "missing provenance source bindings: {}",
                missing.join(", ")
            )));
        }
        let canonical_bytes = encode_artifact_provenance(&bindings)?;
        let identity =
            DatasetFeatureArtifactProvenanceIdentityV1(Sha256::digest(&canonical_bytes).into());
        Ok(Self {
            bindings,
            canonical_bytes,
            identity,
        })
    }

    pub fn from_canonical_bytes(
        plan: &FeaturePlanV1,
        bytes: &[u8],
    ) -> Result<Self, FeatureContractError> {
        let mut cursor = Cursor::new(bytes)?;
        cursor.require_exact(
            ARTIFACT_PROVENANCE_DOMAIN_V1,
            "dataset-feature provenance domain",
        )?;
        let version = cursor.read_u16("dataset-feature provenance version")?;
        if version != ARTIFACT_PROVENANCE_VERSION_V1 {
            return Err(FeatureContractError::new(format!(
                "unsupported dataset-feature provenance version {version}"
            )));
        }
        let binding_count = cursor.read_count("provenance bindings")?;
        let mut bindings = Vec::with_capacity(binding_count);
        for _ in 0..binding_count {
            let source_node_id = cursor.read_text("provenance source node id")?;
            let dataset_bytes = cursor.read_length_prefixed("provenance dataset identity")?;
            let dataset_identity = CanonicalDatasetIdentity::from_canonical_bytes(dataset_bytes)
                .map_err(|error| {
                    FeatureContractError::new(format!(
                        "invalid provenance dataset identity: {error}"
                    ))
                })?;
            let manifest_schema_id = cursor.read_text("provenance manifest schema id")?;
            let manifest_hash = cursor.read_hash("provenance manifest hash")?;
            let generation_id = cursor.read_text("provenance generation id")?;
            let vortex_hash = cursor.read_hash("provenance Vortex hash")?;
            let convention = BarTimestampConvention::from_identity_tag(
                cursor.read_u8("provenance bar convention")?,
            )
            .map_err(|error| {
                FeatureContractError::new(format!("invalid provenance bar convention: {error}"))
            })?;
            let segment_count = cursor.read_count("provenance segments")?;
            let mut segments = Vec::with_capacity(segment_count);
            for _ in 0..segment_count {
                segments.push(SourceSegmentV1::new(
                    cursor.read_u64("segment row start")?,
                    cursor.read_u64("segment row end")?,
                    cursor.read_i64("segment timestamp start")?,
                    cursor.read_i64("segment timestamp end")?,
                )?);
            }
            bindings.push(SourceArtifactBindingV1::new(
                source_node_id,
                dataset_identity,
                manifest_schema_id,
                manifest_hash,
                generation_id,
                vortex_hash,
                convention,
                segments,
            )?);
        }
        cursor.require_empty("dataset-feature provenance")?;
        let provenance = Self::new(plan, bindings)?;
        if provenance.canonical_bytes != bytes {
            return Err(FeatureContractError::new(
                "dataset-feature provenance bytes are not canonical",
            ));
        }
        Ok(provenance)
    }

    pub fn bindings(&self) -> &[SourceArtifactBindingV1] {
        &self.bindings
    }

    pub const fn identity(&self) -> DatasetFeatureArtifactProvenanceIdentityV1 {
        self.identity
    }

    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }
}

fn canonical_topological_order(
    nodes: Vec<FeatureNodeV1>,
) -> Result<Vec<FeatureNodeV1>, FeatureContractError> {
    let mut by_id = BTreeMap::new();
    for node in nodes {
        if by_id.insert(node.id.clone(), node).is_some() {
            return Err(FeatureContractError::new(
                "feature plan contains duplicate node ids",
            ));
        }
    }
    let mut indegree = BTreeMap::<String, usize>::new();
    let mut outgoing = BTreeMap::<String, Vec<String>>::new();
    for (id, node) in &by_id {
        indegree.insert(id.clone(), node.inputs.len());
        for input in &node.inputs {
            if !by_id.contains_key(input) {
                return Err(FeatureContractError::new(format!(
                    "feature node `{id}` references missing input `{input}`"
                )));
            }
            outgoing.entry(input.clone()).or_default().push(id.clone());
        }
    }
    for dependants in outgoing.values_mut() {
        dependants.sort();
    }
    let mut ready = indegree
        .iter()
        .filter_map(|(id, &count)| (count == 0).then_some(id.clone()))
        .collect::<BTreeSet<_>>();
    let mut ordered_ids = Vec::with_capacity(by_id.len());
    while let Some(id) = ready.pop_first() {
        ordered_ids.push(id.clone());
        if let Some(dependants) = outgoing.get(&id) {
            for dependant in dependants {
                let count = indegree
                    .get_mut(dependant)
                    .expect("outgoing node exists in indegree map");
                *count -= 1;
                if *count == 0 {
                    ready.insert(dependant.clone());
                }
            }
        }
    }
    if ordered_ids.len() != by_id.len() {
        return Err(FeatureContractError::new(
            "feature plan contains a dependency cycle",
        ));
    }
    Ok(ordered_ids
        .into_iter()
        .map(|id| by_id.remove(&id).expect("topological id exists"))
        .collect())
}

fn validate_plan_outputs(nodes: &[FeatureNodeV1]) -> Result<(), FeatureContractError> {
    let mut outputs = HashSet::new();
    for node in nodes {
        for output in &node.outputs {
            if !outputs.insert(output.name.as_str()) {
                return Err(FeatureContractError::new(format!(
                    "feature plan output `{}` is produced more than once",
                    output.name
                )));
            }
        }
    }
    Ok(())
}

fn validate_final_outputs(
    nodes: &[FeatureNodeV1],
    final_outputs: &[String],
) -> Result<(), FeatureContractError> {
    if final_outputs.is_empty() {
        return Err(FeatureContractError::new(
            "feature plan must declare at least one ordered final output",
        ));
    }
    let available = nodes
        .iter()
        .flat_map(|node| node.outputs.iter().map(|output| output.name.as_str()))
        .collect::<HashSet<_>>();
    let mut seen = HashSet::with_capacity(final_outputs.len());
    for output in final_outputs {
        validate_text("feature-plan final output", output)?;
        if !available.contains(output.as_str()) {
            return Err(FeatureContractError::new(format!(
                "feature-plan final output `{output}` has no producing node"
            )));
        }
        if !seen.insert(output.as_str()) {
            return Err(FeatureContractError::new(format!(
                "feature-plan final output `{output}` is duplicated"
            )));
        }
    }
    Ok(())
}

fn encode_feature_plan(
    nodes: &[FeatureNodeV1],
    final_outputs: &[String],
) -> Result<Vec<u8>, FeatureContractError> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(FEATURE_PLAN_DOMAIN_V1);
    bytes.extend_from_slice(&FEATURE_PLAN_VERSION_V1.to_be_bytes());
    push_count(&mut bytes, nodes.len(), "feature nodes")?;
    for node in nodes {
        push_text(&mut bytes, &node.id)?;
        bytes.push(node.operation as u8);
        bytes.extend_from_slice(&node.semantic_version.to_be_bytes());
        match &node.source {
            Some(source) => {
                bytes.push(1);
                push_bytes(&mut bytes, &source.dataset_identity.canonical_bytes())?;
                push_text(&mut bytes, &source.physical_schema_id)?;
            }
            None => bytes.push(0),
        }
        push_count(&mut bytes, node.inputs.len(), "node inputs")?;
        for input in &node.inputs {
            push_text(&mut bytes, input)?;
        }
        push_count(&mut bytes, node.outputs.len(), "node outputs")?;
        for output in &node.outputs {
            push_text(&mut bytes, &output.name)?;
            bytes.push(1); // canonical shared physical dtype: f64
            bytes.extend_from_slice(&output.validity_semantic_version.to_be_bytes());
        }
        push_count(&mut bytes, node.parameters.len(), "node parameters")?;
        for parameter in &node.parameters {
            push_text(&mut bytes, &parameter.name)?;
            match &parameter.value {
                FeatureParameterValueV1::Bool(value) => {
                    bytes.push(1);
                    bytes.push(u8::from(*value));
                }
                FeatureParameterValueV1::U64(value) => {
                    bytes.push(2);
                    bytes.extend_from_slice(&value.to_be_bytes());
                }
                FeatureParameterValueV1::I64(value) => {
                    bytes.push(3);
                    bytes.extend_from_slice(&value.to_be_bytes());
                }
                FeatureParameterValueV1::F64Bits(bits) => {
                    bytes.push(4);
                    bytes.extend_from_slice(&bits.to_be_bytes());
                }
                FeatureParameterValueV1::Text(value) => {
                    bytes.push(5);
                    push_text(&mut bytes, value)?;
                }
                FeatureParameterValueV1::Hash(value) => {
                    bytes.push(6);
                    bytes.extend_from_slice(value);
                }
            }
        }
        bytes.extend_from_slice(&node.formula_manifest_hash);
        bytes.extend_from_slice(&node.semantic_source_hash);
        match node.fitted_state_hash {
            Some(hash) => {
                bytes.push(1);
                bytes.extend_from_slice(&hash);
            }
            None => bytes.push(0),
        }
    }
    push_count(
        &mut bytes,
        final_outputs.len(),
        "feature-plan final outputs",
    )?;
    for output in final_outputs {
        push_text(&mut bytes, output)?;
    }
    Ok(bytes)
}

fn encode_artifact_provenance(
    bindings: &[SourceArtifactBindingV1],
) -> Result<Vec<u8>, FeatureContractError> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(ARTIFACT_PROVENANCE_DOMAIN_V1);
    bytes.extend_from_slice(&ARTIFACT_PROVENANCE_VERSION_V1.to_be_bytes());
    push_count(&mut bytes, bindings.len(), "provenance bindings")?;
    for binding in bindings {
        push_text(&mut bytes, &binding.source_node_id)?;
        push_bytes(&mut bytes, &binding.dataset_identity.canonical_bytes())?;
        push_text(&mut bytes, &binding.manifest_schema_id)?;
        bytes.extend_from_slice(&binding.manifest_hash);
        push_text(&mut bytes, &binding.generation_id)?;
        bytes.extend_from_slice(&binding.vortex_hash);
        bytes.push(binding.bar_timestamp_convention.identity_tag());
        push_count(&mut bytes, binding.segments.len(), "provenance segments")?;
        for segment in &binding.segments {
            bytes.extend_from_slice(&segment.row_start.to_be_bytes());
            bytes.extend_from_slice(&segment.row_end.to_be_bytes());
            bytes.extend_from_slice(&segment.timestamp_start_ms.to_be_bytes());
            bytes.extend_from_slice(&segment.timestamp_end_ms.to_be_bytes());
        }
    }
    Ok(bytes)
}

fn validate_text(field: &str, value: &str) -> Result<(), FeatureContractError> {
    if value.is_empty() {
        return Err(FeatureContractError::new(format!(
            "{field} must not be empty"
        )));
    }
    if value.len() > MAX_TEXT_BYTES {
        return Err(FeatureContractError::new(format!(
            "{field} exceeds {MAX_TEXT_BYTES} bytes"
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(FeatureContractError::new(format!(
            "{field} contains a control character"
        )));
    }
    Ok(())
}

fn validate_unique_texts<'a>(
    field: &str,
    values: impl IntoIterator<Item = &'a str>,
) -> Result<(), FeatureContractError> {
    let mut unique = HashSet::new();
    for value in values {
        validate_text(field, value)?;
        if !unique.insert(value) {
            return Err(FeatureContractError::new(format!(
                "duplicate {field} `{value}`"
            )));
        }
    }
    Ok(())
}

fn validate_opaque_component(field: &str, value: &str) -> Result<(), FeatureContractError> {
    validate_text(field, value)?;
    if matches!(value, "." | "..")
        || value.contains('/')
        || value.contains('\\')
        || value.contains(':')
    {
        return Err(FeatureContractError::new(format!(
            "{field} must be one opaque path component"
        )));
    }
    Ok(())
}

fn require_positive_version(field: &str, version: u32) -> Result<(), FeatureContractError> {
    if version == 0 {
        return Err(FeatureContractError::new(format!(
            "{field} version must be positive"
        )));
    }
    Ok(())
}

fn push_count(bytes: &mut Vec<u8>, count: usize, field: &str) -> Result<(), FeatureContractError> {
    let count = u32::try_from(count)
        .map_err(|_| FeatureContractError::new(format!("{field} count exceeds u32")))?;
    bytes.extend_from_slice(&count.to_be_bytes());
    Ok(())
}

fn push_text(bytes: &mut Vec<u8>, value: &str) -> Result<(), FeatureContractError> {
    push_bytes(bytes, value.as_bytes())
}

fn push_bytes(bytes: &mut Vec<u8>, value: &[u8]) -> Result<(), FeatureContractError> {
    let length = u32::try_from(value.len())
        .map_err(|_| FeatureContractError::new("canonical field exceeds u32 bytes"))?;
    bytes.extend_from_slice(&length.to_be_bytes());
    bytes.extend_from_slice(value);
    Ok(())
}

fn operation_from_tag(tag: u8) -> Result<FeatureOperationTagV1, FeatureContractError> {
    match tag {
        1 => Ok(FeatureOperationTagV1::Source),
        2 => Ok(FeatureOperationTagV1::Indicator),
        3 => Err(FeatureContractError::new(
            "retired timeframe-resample artifacts are unsupported; use direct broker/source generations",
        )),
        4 => Ok(FeatureOperationTagV1::HigherTimeframeAlignment),
        5 => Ok(FeatureOperationTagV1::Normalization),
        6 => Ok(FeatureOperationTagV1::CrossPairAlignment),
        7 => Ok(FeatureOperationTagV1::Derived),
        _ => Err(FeatureContractError::new(format!(
            "unknown feature operation tag {tag}"
        ))),
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Result<Self, FeatureContractError> {
        if bytes.len() > MAX_CANONICAL_BYTES {
            return Err(FeatureContractError::new(format!(
                "canonical payload exceeds {MAX_CANONICAL_BYTES} bytes"
            )));
        }
        Ok(Self { bytes, offset: 0 })
    }

    fn require_exact(&mut self, expected: &[u8], field: &str) -> Result<(), FeatureContractError> {
        let actual = self.take(expected.len(), field)?;
        if actual != expected {
            return Err(FeatureContractError::new(format!("invalid {field}")));
        }
        Ok(())
    }

    fn require_empty(&self, field: &str) -> Result<(), FeatureContractError> {
        if self.offset != self.bytes.len() {
            return Err(FeatureContractError::new(format!(
                "{field} contains {} trailing bytes",
                self.bytes.len() - self.offset
            )));
        }
        Ok(())
    }

    fn read_u8(&mut self, field: &str) -> Result<u8, FeatureContractError> {
        Ok(self.take(1, field)?[0])
    }

    fn read_u16(&mut self, field: &str) -> Result<u16, FeatureContractError> {
        let bytes: [u8; 2] = self
            .take(2, field)?
            .try_into()
            .expect("cursor returned the requested u16 width");
        Ok(u16::from_be_bytes(bytes))
    }

    fn read_u32(&mut self, field: &str) -> Result<u32, FeatureContractError> {
        let bytes: [u8; 4] = self
            .take(4, field)?
            .try_into()
            .expect("cursor returned the requested u32 width");
        Ok(u32::from_be_bytes(bytes))
    }

    fn read_u64(&mut self, field: &str) -> Result<u64, FeatureContractError> {
        let bytes: [u8; 8] = self
            .take(8, field)?
            .try_into()
            .expect("cursor returned the requested u64 width");
        Ok(u64::from_be_bytes(bytes))
    }

    fn read_i64(&mut self, field: &str) -> Result<i64, FeatureContractError> {
        let bytes: [u8; 8] = self
            .take(8, field)?
            .try_into()
            .expect("cursor returned the requested i64 width");
        Ok(i64::from_be_bytes(bytes))
    }

    fn read_hash(&mut self, field: &str) -> Result<[u8; 32], FeatureContractError> {
        Ok(self
            .take(32, field)?
            .try_into()
            .expect("cursor returned the requested hash width"))
    }

    fn read_count(&mut self, field: &str) -> Result<usize, FeatureContractError> {
        let count = usize::try_from(self.read_u32(field)?)
            .map_err(|_| FeatureContractError::new(format!("{field} count does not fit usize")))?;
        if count > MAX_CANONICAL_ITEMS {
            return Err(FeatureContractError::new(format!(
                "{field} count exceeds {MAX_CANONICAL_ITEMS}"
            )));
        }
        Ok(count)
    }

    fn read_length_prefixed(&mut self, field: &str) -> Result<&'a [u8], FeatureContractError> {
        let length = usize::try_from(self.read_u32(field)?)
            .map_err(|_| FeatureContractError::new(format!("{field} length does not fit usize")))?;
        if length > MAX_NESTED_CANONICAL_BYTES {
            return Err(FeatureContractError::new(format!(
                "{field} exceeds {MAX_NESTED_CANONICAL_BYTES} bytes"
            )));
        }
        self.take(length, field)
    }

    fn read_text(&mut self, field: &str) -> Result<String, FeatureContractError> {
        let length = usize::try_from(self.read_u32(field)?)
            .map_err(|_| FeatureContractError::new(format!("{field} length does not fit usize")))?;
        if length > MAX_TEXT_BYTES {
            return Err(FeatureContractError::new(format!(
                "{field} exceeds {MAX_TEXT_BYTES} bytes"
            )));
        }
        let bytes = self.take(length, field)?;
        let value = std::str::from_utf8(bytes)
            .map_err(|_| FeatureContractError::new(format!("{field} is not valid UTF-8")))?;
        validate_text(field, value)?;
        Ok(value.to_owned())
    }

    fn read_text_vec(&mut self, field: &str) -> Result<Vec<String>, FeatureContractError> {
        let count = self.read_count(field)?;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(self.read_text(field)?);
        }
        Ok(values)
    }

    fn take(&mut self, length: usize, field: &str) -> Result<&'a [u8], FeatureContractError> {
        let end = self.offset.checked_add(length).ok_or_else(|| {
            FeatureContractError::new(format!("{field} length overflows canonical payload"))
        })?;
        if end > self.bytes.len() {
            return Err(FeatureContractError::new(format!(
                "canonical payload ended while reading {field}"
            )));
        }
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(DIGITS[(byte >> 4) as usize] as char);
        encoded.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retired_resample_operation_tag_fails_closed() {
        let error = operation_from_tag(3).expect_err("resample artifacts must stay retired");
        assert!(error.to_string().contains("direct broker/source"));
    }
}
