//! Versioned control-plane contracts for an exact CUDA-resident feature store.
//!
//! These serializable values never confer access to CUDA allocations, streams,
//! contexts or events. `neoethos-data` mints the opaque runtime authority only
//! after real device, route, producer-capability and working-set validation.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

pub const GPU_ONLY_RESIDENT_ADMISSION_SCHEMA_V3: &str = "neoethos.gpu-only-resident-admission.v3";
pub const SEALED_RESIDENT_FEATURE_STORE_SCHEMA_V3: &str =
    "neoethos.sealed-resident-feature-store.v3";
pub const CANONICAL_MERKLE_CHUNK_ROWS_V3: usize = 4096;
pub const CANONICAL_FEATURE_MERKLE_LEAF_DOMAIN_V3: &[u8] =
    b"neoethos.canonical-feature-content.merkle.leaf.v3\0";
pub const CANONICAL_FEATURE_MERKLE_NODE_DOMAIN_V3: &[u8] =
    b"neoethos.canonical-feature-content.merkle.node.v3\0";
pub const CANONICAL_FEATURE_CONTENT_HASH_DOMAIN_V3: &[u8] =
    b"neoethos.canonical-feature-content.merkle.root.v3\0";
pub const READY_EVENT_INTEROP_ABI_V3: &str =
    "cuda-driver-runtime.same-primary-context.cuEventRecord-cuStreamWaitEvent.v3";
pub const PORTABLE_CUDA_SHA256_AUTHORITY_V3: &str =
    "neoethos.cuda.in-tree.parallel-merkle-sha256.exact-bits-validity.v3";
pub const RESIDENT_VALIDITY_SCHEMA_V3: &str =
    "neoethos.resident-validity.lossless-u4-logical-u8.v3";

const SHA256_BYTES: usize = 32;
const CUDA_UUID_BYTES: usize = 16;
const F64_BYTES: u64 = 8;
const I64_BYTES: u64 = 8;
const PRODUCER_VALIDITY_BYTES: u64 = 1;
const MERKLE_DIGEST_BYTES: u64 = 32;
const VALIDITY_ERROR_FLAG_BYTES: u64 = 4;
const VALIDITY_ATOMIC_ALIGNMENT_BYTES: u64 = 4;
const MAX_VALIDITY_CODE_V3: u8 = 9;
const SMC_SLOTS_V3: u64 = 11;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidentFeatureContractErrorV3 {
    EmptyField {
        field: &'static str,
    },
    ZeroHash {
        field: &'static str,
    },
    MissingProducerCapabilities {
        missing: Vec<ResidentFeatureProducerV3>,
    },
    DuplicateProducerCapability {
        producer: ResidentFeatureProducerV3,
    },
    ProducerCapabilityOrderMismatch {
        index: usize,
        expected: ResidentFeatureProducerV3,
        actual: ResidentFeatureProducerV3,
    },
    ArithmeticOverflow {
        field: &'static str,
    },
    WorkingSetExceedsDevice {
        required_bytes: u64,
        available_bytes: u64,
    },
    RouteOrderMismatch {
        index: usize,
        actual: u64,
    },
    DuplicateFeatureName {
        name: String,
    },
    InvalidStagePeriod {
        feature_name: String,
    },
    LayoutMismatch {
        field: &'static str,
        expected: u64,
        actual: u64,
    },
    InvalidValidityCode {
        index: usize,
        code: u8,
    },
    InvalidPackedValidity {
        reason: &'static str,
    },
    NativeSassTargetMismatch {
        expected: String,
        actual: String,
    },
    AdmissionIdentityMismatch,
    DeviceIdentityMismatch,
    PrimaryContextMismatch,
    OrderedSchemaMismatch,
    Sha256AuthorityMissingPortablePath,
}

impl fmt::Display for ResidentFeatureContractErrorV3 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField { field } => write!(formatter, "{field} must not be empty"),
            Self::ZeroHash { field } => write!(formatter, "{field} must not be an all-zero hash"),
            Self::MissingProducerCapabilities { missing } => write!(
                formatter,
                "strict resident admission is missing producer capabilities: {missing:?}"
            ),
            Self::DuplicateProducerCapability { producer } => {
                write!(formatter, "duplicate resident capability for {producer:?}")
            }
            Self::ProducerCapabilityOrderMismatch {
                index,
                expected,
                actual,
            } => write!(
                formatter,
                "resident capability {index} is {actual:?}, expected {expected:?}"
            ),
            Self::ArithmeticOverflow { field } => {
                write!(
                    formatter,
                    "resident feature arithmetic overflowed at {field}"
                )
            }
            Self::WorkingSetExceedsDevice {
                required_bytes,
                available_bytes,
            } => write!(
                formatter,
                "resident working set needs {required_bytes} bytes, only {available_bytes} are available"
            ),
            Self::RouteOrderMismatch { index, actual } => write!(
                formatter,
                "resident route at vector index {index} declares ordinal {actual}"
            ),
            Self::DuplicateFeatureName { name } => {
                write!(formatter, "duplicate resident feature name `{name}`")
            }
            Self::InvalidStagePeriod { feature_name } => write!(
                formatter,
                "resident route `{feature_name}` has an invalid stage/period pair"
            ),
            Self::LayoutMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "resident layout {field} is {actual}, expected {expected}"
            ),
            Self::InvalidValidityCode { index, code } => {
                write!(
                    formatter,
                    "validity code {code} at logical cell {index} exceeds 9"
                )
            }
            Self::InvalidPackedValidity { reason } => {
                write!(formatter, "invalid packed resident validity: {reason}")
            }
            Self::NativeSassTargetMismatch { expected, actual } => write!(
                formatter,
                "native SASS target `{actual}` does not match selected device `{expected}`"
            ),
            Self::AdmissionIdentityMismatch => {
                formatter.write_str("phase-two admission identity does not match phase one")
            }
            Self::DeviceIdentityMismatch => {
                formatter.write_str("phase-two CUDA device/build identity drifted")
            }
            Self::PrimaryContextMismatch => {
                formatter.write_str("producer event is not from the admitted primary context")
            }
            Self::OrderedSchemaMismatch => {
                formatter.write_str("phase-two ordered feature schema drifted")
            }
            Self::Sha256AuthorityMissingPortablePath => formatter.write_str(
                "canonical resident SHA-256 lacks the mandatory portable in-tree CUDA path",
            ),
        }
    }
}

impl std::error::Error for ResidentFeatureContractErrorV3 {}

fn require_text(field: &'static str, value: &str) -> Result<(), ResidentFeatureContractErrorV3> {
    if value.trim().is_empty() {
        Err(ResidentFeatureContractErrorV3::EmptyField { field })
    } else {
        Ok(())
    }
}

fn require_hash(
    field: &'static str,
    value: &[u8; SHA256_BYTES],
) -> Result<(), ResidentFeatureContractErrorV3> {
    if value.iter().all(|byte| *byte == 0) {
        Err(ResidentFeatureContractErrorV3::ZeroHash { field })
    } else {
        Ok(())
    }
}

fn checked_mul(
    left: u64,
    right: u64,
    field: &'static str,
) -> Result<u64, ResidentFeatureContractErrorV3> {
    left.checked_mul(right)
        .ok_or(ResidentFeatureContractErrorV3::ArithmeticOverflow { field })
}

fn checked_add(
    left: u64,
    right: u64,
    field: &'static str,
) -> Result<u64, ResidentFeatureContractErrorV3> {
    left.checked_add(right)
        .ok_or(ResidentFeatureContractErrorV3::ArithmeticOverflow { field })
}

fn checked_sum(
    values: impl IntoIterator<Item = u64>,
    field: &'static str,
) -> Result<u64, ResidentFeatureContractErrorV3> {
    values
        .into_iter()
        .try_fold(0_u64, |sum, value| checked_add(sum, value, field))
}

fn packed_validity_logical_bytes(cells: u64) -> u64 {
    cells / 2 + cells % 2
}

fn align_up_u64(
    value: u64,
    alignment: u64,
    field: &'static str,
) -> Result<u64, ResidentFeatureContractErrorV3> {
    let with_padding = checked_add(value, alignment - 1, field)?;
    Ok((with_padding / alignment) * alignment)
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum ResidentFeatureProducerV3 {
    ClassicTa = 0,
    Smc = 1,
    Quant = 2,
    Session = 3,
    Regime = 4,
    Footprint = 5,
    HigherTimeframeAlignment = 6,
    RobustNormalization = 7,
    CanonicalContentSha256 = 8,
    FeatureMajorToBarMajor = 9,
}

impl ResidentFeatureProducerV3 {
    pub const ALL: [Self; 10] = [
        Self::ClassicTa,
        Self::Smc,
        Self::Quant,
        Self::Session,
        Self::Regime,
        Self::Footprint,
        Self::HigherTimeframeAlignment,
        Self::RobustNormalization,
        Self::CanonicalContentSha256,
        Self::FeatureMajorToBarMajor,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ClassicTa => "classic_ta",
            Self::Smc => "smc",
            Self::Quant => "quant",
            Self::Session => "session",
            Self::Regime => "regime",
            Self::Footprint => "footprint",
            Self::HigherTimeframeAlignment => "higher_timeframe_alignment",
            Self::RobustNormalization => "robust_normalization",
            Self::CanonicalContentSha256 => "canonical_content_sha256",
            Self::FeatureMajorToBarMajor => "feature_major_to_bar_major",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResidentProducerCapabilityV3 {
    producer: ResidentFeatureProducerV3,
    implementation_id: String,
    implementation_sha256: [u8; SHA256_BYTES],
    exact_math_authority: String,
}

impl ResidentProducerCapabilityV3 {
    pub fn new(
        producer: ResidentFeatureProducerV3,
        implementation_id: impl Into<String>,
        implementation_sha256: [u8; SHA256_BYTES],
        exact_math_authority: impl Into<String>,
    ) -> Result<Self, ResidentFeatureContractErrorV3> {
        let implementation_id = implementation_id.into();
        let exact_math_authority = exact_math_authority.into();
        require_text("producer implementation id", &implementation_id)?;
        require_hash("producer implementation sha256", &implementation_sha256)?;
        require_text("producer exact math authority", &exact_math_authority)?;
        Ok(Self {
            producer,
            implementation_id,
            implementation_sha256,
            exact_math_authority,
        })
    }

    pub const fn producer(&self) -> ResidentFeatureProducerV3 {
        self.producer
    }

    pub fn implementation_id(&self) -> &str {
        &self.implementation_id
    }

    pub const fn implementation_sha256(&self) -> [u8; SHA256_BYTES] {
        self.implementation_sha256
    }

    pub fn exact_math_authority(&self) -> &str {
        &self.exact_math_authority
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResidentProducerCapabilityManifestV3 {
    capabilities: Vec<ResidentProducerCapabilityV3>,
}

impl ResidentProducerCapabilityManifestV3 {
    pub fn seal(
        capabilities: Vec<ResidentProducerCapabilityV3>,
    ) -> Result<Self, ResidentFeatureContractErrorV3> {
        let mut seen = BTreeSet::new();
        for capability in &capabilities {
            if !seen.insert(capability.producer()) {
                return Err(
                    ResidentFeatureContractErrorV3::DuplicateProducerCapability {
                        producer: capability.producer(),
                    },
                );
            }
        }
        let missing = ResidentFeatureProducerV3::ALL
            .iter()
            .copied()
            .filter(|producer| !seen.contains(producer))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(ResidentFeatureContractErrorV3::MissingProducerCapabilities { missing });
        }
        for (index, (actual, expected)) in capabilities
            .iter()
            .map(ResidentProducerCapabilityV3::producer)
            .zip(ResidentFeatureProducerV3::ALL)
            .enumerate()
        {
            if actual != expected {
                return Err(
                    ResidentFeatureContractErrorV3::ProducerCapabilityOrderMismatch {
                        index,
                        expected,
                        actual,
                    },
                );
            }
        }
        Ok(Self { capabilities })
    }

    pub fn capabilities(&self) -> &[ResidentProducerCapabilityV3] {
        &self.capabilities
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CudaPrimaryContextBuildIdentityV3 {
    ordinal: u32,
    device_uuid: [u8; CUDA_UUID_BYTES],
    compute_capability_major: u16,
    compute_capability_minor: u16,
    primary_context_process_token: [u8; SHA256_BYTES],
    driver_version: String,
    runtime_version: String,
    nvcc_version: String,
    native_sass_target: String,
    vector_ta_build_sha256: [u8; SHA256_BYTES],
    gpu_cuda_build_sha256: [u8; SHA256_BYTES],
    exact_math_authority: String,
}

impl CudaPrimaryContextBuildIdentityV3 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ordinal: u32,
        device_uuid: [u8; CUDA_UUID_BYTES],
        compute_capability_major: u16,
        compute_capability_minor: u16,
        primary_context_process_token: [u8; SHA256_BYTES],
        driver_version: impl Into<String>,
        runtime_version: impl Into<String>,
        nvcc_version: impl Into<String>,
        native_sass_target: impl Into<String>,
        vector_ta_build_sha256: [u8; SHA256_BYTES],
        gpu_cuda_build_sha256: [u8; SHA256_BYTES],
        exact_math_authority: impl Into<String>,
    ) -> Result<Self, ResidentFeatureContractErrorV3> {
        if device_uuid.iter().all(|byte| *byte == 0) {
            return Err(ResidentFeatureContractErrorV3::ZeroHash {
                field: "CUDA device UUID",
            });
        }
        if compute_capability_major == 0 {
            return Err(ResidentFeatureContractErrorV3::EmptyField {
                field: "CUDA compute capability",
            });
        }
        require_hash(
            "primary context process token",
            &primary_context_process_token,
        )?;
        require_hash("vector-ta CUDA build sha256", &vector_ta_build_sha256)?;
        require_hash("neoethos-gpu-cuda build sha256", &gpu_cuda_build_sha256)?;
        let driver_version = driver_version.into();
        let runtime_version = runtime_version.into();
        let nvcc_version = nvcc_version.into();
        let native_sass_target = native_sass_target.into();
        let exact_math_authority = exact_math_authority.into();
        require_text("CUDA driver version", &driver_version)?;
        require_text("CUDA runtime version", &runtime_version)?;
        require_text("NVCC version", &nvcc_version)?;
        require_text("native SASS target", &native_sass_target)?;
        require_text("CUDA exact math authority", &exact_math_authority)?;
        let expected_sass_target = format!(
            "sm_{}{}",
            compute_capability_major, compute_capability_minor
        );
        if native_sass_target != expected_sass_target {
            return Err(ResidentFeatureContractErrorV3::NativeSassTargetMismatch {
                expected: expected_sass_target,
                actual: native_sass_target,
            });
        }
        Ok(Self {
            ordinal,
            device_uuid,
            compute_capability_major,
            compute_capability_minor,
            primary_context_process_token,
            driver_version,
            runtime_version,
            nvcc_version,
            native_sass_target,
            vector_ta_build_sha256,
            gpu_cuda_build_sha256,
            exact_math_authority,
        })
    }

    pub const fn ordinal(&self) -> u32 {
        self.ordinal
    }

    pub const fn primary_context_process_token(&self) -> [u8; SHA256_BYTES] {
        self.primary_context_process_token
    }

    pub fn native_sass_target(&self) -> &str {
        &self.native_sass_target
    }

    pub const fn vector_ta_build_sha256(&self) -> [u8; SHA256_BYTES] {
        self.vector_ta_build_sha256
    }

    pub fn nvcc_version(&self) -> &str {
        &self.nvcc_version
    }

    pub fn exact_math_authority(&self) -> &str {
        &self.exact_math_authority
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ResidentFeatureStageV3 {
    Base,
    Historical,
    Extended,
    Derived,
    HigherTimeframeAligned,
    Normalized,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResidentFeatureRouteV3 {
    ordinal: u64,
    feature_name: String,
    producer: ResidentFeatureProducerV3,
    indicator_id: Option<String>,
    output_id: Option<String>,
    stage: ResidentFeatureStageV3,
    swept_period: Option<u64>,
    canonical_parameter_tuple_sha256: [u8; SHA256_BYTES],
    route_id: String,
    route_receipt_sha256: [u8; SHA256_BYTES],
}

impl ResidentFeatureRouteV3 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ordinal: u64,
        feature_name: impl Into<String>,
        producer: ResidentFeatureProducerV3,
        indicator_id: Option<impl Into<String>>,
        output_id: Option<impl Into<String>>,
        stage: ResidentFeatureStageV3,
        swept_period: Option<u64>,
        canonical_parameter_tuple_sha256: [u8; SHA256_BYTES],
        route_id: impl Into<String>,
        route_receipt_sha256: [u8; SHA256_BYTES],
    ) -> Result<Self, ResidentFeatureContractErrorV3> {
        let feature_name = feature_name.into();
        let indicator_id = indicator_id.map(Into::into);
        let output_id = output_id.map(Into::into);
        let route_id = route_id.into();
        require_text("resident feature name", &feature_name)?;
        require_text("resident route id", &route_id)?;
        require_hash(
            "canonical parameter tuple sha256",
            &canonical_parameter_tuple_sha256,
        )?;
        require_hash("resident route receipt sha256", &route_receipt_sha256)?;
        if indicator_id.as_deref().is_some_and(str::is_empty)
            || output_id.as_deref().is_some_and(str::is_empty)
        {
            return Err(ResidentFeatureContractErrorV3::EmptyField {
                field: "indicator/output identity",
            });
        }
        let period_is_valid = match stage {
            ResidentFeatureStageV3::Historical | ResidentFeatureStageV3::Extended => {
                swept_period.is_some_and(|period| period > 0)
            }
            ResidentFeatureStageV3::Base
            | ResidentFeatureStageV3::Derived
            | ResidentFeatureStageV3::HigherTimeframeAligned
            | ResidentFeatureStageV3::Normalized => swept_period.is_none(),
        };
        if !period_is_valid {
            return Err(ResidentFeatureContractErrorV3::InvalidStagePeriod { feature_name });
        }
        Ok(Self {
            ordinal,
            feature_name,
            producer,
            indicator_id,
            output_id,
            stage,
            swept_period,
            canonical_parameter_tuple_sha256,
            route_id,
            route_receipt_sha256,
        })
    }

    pub const fn ordinal(&self) -> u64 {
        self.ordinal
    }

    pub fn feature_name(&self) -> &str {
        &self.feature_name
    }

    pub const fn producer(&self) -> ResidentFeatureProducerV3 {
        self.producer
    }

    pub const fn swept_period(&self) -> Option<u64> {
        self.swept_period
    }

    pub const fn canonical_parameter_tuple_sha256(&self) -> [u8; SHA256_BYTES] {
        self.canonical_parameter_tuple_sha256
    }

    pub const fn route_receipt_sha256(&self) -> [u8; SHA256_BYTES] {
        self.route_receipt_sha256
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ResidentWorkingSetRequestV3 {
    pub row_count: usize,
    pub column_count: usize,
    pub max_live_producer_bytes: u64,
    pub max_live_producer_scratch_bytes: u64,
    pub normalization_scratch_bytes: u64,
    pub fit_metadata_bytes: u64,
    pub pointer_and_schema_metadata_bytes: u64,
    pub device_free_bytes_snapshot: u64,
    pub allocator_context_reserve_bytes: u64,
    pub reserve_policy_id: String,
}

impl ResidentWorkingSetRequestV3 {
    pub fn seal(self) -> Result<ResidentWorkingSetBoundV3, ResidentFeatureContractErrorV3> {
        ResidentWorkingSetBoundV3::seal(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResidentWorkingSetBoundV3 {
    row_count: u64,
    column_count: u64,
    final_bar_major_value_bytes: u64,
    packed_validity_logical_bytes: u64,
    packed_validity_allocated_bytes: u64,
    parent_ohlcv_bytes: u64,
    parent_clock_bytes: u64,
    parent_smc_bytes: u64,
    parent_dataset_bytes: u64,
    canonical_root_bytes: u64,
    active_view_indices_bytes: u64,
    lazy_view_indices_capacity_bytes: u64,
    max_live_producer_bytes: u64,
    max_live_producer_scratch_bytes: u64,
    normalization_scratch_bytes: u64,
    fit_metadata_bytes: u64,
    pointer_and_schema_metadata_bytes: u64,
    merkle_leaf_count: u64,
    merkle_scratch_bytes: u64,
    compact_hash_and_error_bytes: u64,
    full_feature_major_staging_bytes: u64,
    steady_device_bytes: u64,
    peak_device_bytes: u64,
    device_free_bytes_snapshot: u64,
    allocator_context_reserve_bytes: u64,
    reserve_policy_id: String,
    available_device_bytes: u64,
}

impl ResidentWorkingSetBoundV3 {
    fn seal(request: ResidentWorkingSetRequestV3) -> Result<Self, ResidentFeatureContractErrorV3> {
        if request.row_count == 0 || request.column_count == 0 {
            return Err(ResidentFeatureContractErrorV3::EmptyField {
                field: "resident rows and columns",
            });
        }
        require_text(
            "allocator/context reserve policy id",
            &request.reserve_policy_id,
        )?;
        let row_count = u64::try_from(request.row_count).map_err(|_| {
            ResidentFeatureContractErrorV3::ArithmeticOverflow {
                field: "resident row count",
            }
        })?;
        let column_count = u64::try_from(request.column_count).map_err(|_| {
            ResidentFeatureContractErrorV3::ArithmeticOverflow {
                field: "resident column count",
            }
        })?;
        let cells = checked_mul(row_count, column_count, "resident cell count")?;
        let final_bar_major_value_bytes = checked_mul(cells, F64_BYTES, "final bar-major values")?;
        let packed_validity_logical_bytes = packed_validity_logical_bytes(cells);
        let packed_validity_allocated_bytes = align_up_u64(
            packed_validity_logical_bytes,
            VALIDITY_ATOMIC_ALIGNMENT_BYTES,
            "packed validity allocation",
        )?;
        let parent_ohlcv_bytes = checked_mul(
            checked_mul(row_count, 5, "parent OHLCV arrays")?,
            F64_BYTES,
            "parent OHLCV bytes",
        )?;
        let parent_clock_bytes = checked_mul(
            checked_mul(row_count, 3, "parent clock arrays")?,
            I64_BYTES,
            "parent clock bytes",
        )?;
        let parent_smc_bytes = checked_mul(row_count, SMC_SLOTS_V3, "parent SMC bytes")?;
        let parent_dataset_bytes = checked_sum(
            [parent_ohlcv_bytes, parent_clock_bytes, parent_smc_bytes],
            "parent resident dataset bytes",
        )?;
        let canonical_root_bytes = MERKLE_DIGEST_BYTES;
        let active_view_indices_bytes = 0;
        let lazy_view_indices_capacity_bytes = 0;
        let chunk_rows = u64::try_from(CANONICAL_MERKLE_CHUNK_ROWS_V3).map_err(|_| {
            ResidentFeatureContractErrorV3::ArithmeticOverflow {
                field: "Merkle chunk rows",
            }
        })?;
        let timestamp_chunk_count = row_count / chunk_rows + u64::from(row_count % chunk_rows != 0);
        let producer_count = checked_add(column_count, 1, "Merkle producer count")?;
        let merkle_leaf_count =
            checked_mul(timestamp_chunk_count, producer_count, "Merkle leaf count")?;
        let one_merkle_level_bytes = checked_mul(
            merkle_leaf_count,
            MERKLE_DIGEST_BYTES,
            "one Merkle scratch level",
        )?;
        let merkle_scratch_bytes =
            checked_mul(one_merkle_level_bytes, 2, "two Merkle scratch levels")?;
        let compact_hash_and_error_bytes = VALIDITY_ERROR_FLAG_BYTES;
        let steady_device_bytes = checked_sum(
            [
                final_bar_major_value_bytes,
                packed_validity_allocated_bytes,
                parent_dataset_bytes,
                canonical_root_bytes,
                request.fit_metadata_bytes,
            ],
            "resident steady bytes",
        )?;
        let peak_device_bytes = checked_sum(
            [
                steady_device_bytes,
                request.max_live_producer_bytes,
                request.max_live_producer_scratch_bytes,
                request.normalization_scratch_bytes,
                request.pointer_and_schema_metadata_bytes,
                merkle_scratch_bytes,
                compact_hash_and_error_bytes,
            ],
            "resident peak bytes",
        )?;
        let available_device_bytes = request
            .device_free_bytes_snapshot
            .checked_sub(request.allocator_context_reserve_bytes)
            .ok_or(ResidentFeatureContractErrorV3::WorkingSetExceedsDevice {
                required_bytes: peak_device_bytes,
                available_bytes: 0,
            })?;
        if peak_device_bytes > available_device_bytes {
            return Err(ResidentFeatureContractErrorV3::WorkingSetExceedsDevice {
                required_bytes: peak_device_bytes,
                available_bytes: available_device_bytes,
            });
        }
        Ok(Self {
            row_count,
            column_count,
            final_bar_major_value_bytes,
            packed_validity_logical_bytes,
            packed_validity_allocated_bytes,
            parent_ohlcv_bytes,
            parent_clock_bytes,
            parent_smc_bytes,
            parent_dataset_bytes,
            canonical_root_bytes,
            active_view_indices_bytes,
            lazy_view_indices_capacity_bytes,
            max_live_producer_bytes: request.max_live_producer_bytes,
            max_live_producer_scratch_bytes: request.max_live_producer_scratch_bytes,
            normalization_scratch_bytes: request.normalization_scratch_bytes,
            fit_metadata_bytes: request.fit_metadata_bytes,
            pointer_and_schema_metadata_bytes: request.pointer_and_schema_metadata_bytes,
            merkle_leaf_count,
            merkle_scratch_bytes,
            compact_hash_and_error_bytes,
            full_feature_major_staging_bytes: 0,
            steady_device_bytes,
            peak_device_bytes,
            device_free_bytes_snapshot: request.device_free_bytes_snapshot,
            allocator_context_reserve_bytes: request.allocator_context_reserve_bytes,
            reserve_policy_id: request.reserve_policy_id,
            available_device_bytes,
        })
    }

    pub const fn row_count(&self) -> u64 {
        self.row_count
    }
    pub const fn column_count(&self) -> u64 {
        self.column_count
    }
    pub const fn final_bar_major_value_bytes(&self) -> u64 {
        self.final_bar_major_value_bytes
    }
    pub const fn packed_validity_logical_bytes(&self) -> u64 {
        self.packed_validity_logical_bytes
    }
    pub const fn packed_validity_allocated_bytes(&self) -> u64 {
        self.packed_validity_allocated_bytes
    }
    pub const fn parent_ohlcv_bytes(&self) -> u64 {
        self.parent_ohlcv_bytes
    }
    pub const fn parent_clock_bytes(&self) -> u64 {
        self.parent_clock_bytes
    }
    pub const fn parent_smc_bytes(&self) -> u64 {
        self.parent_smc_bytes
    }
    pub const fn parent_dataset_bytes(&self) -> u64 {
        self.parent_dataset_bytes
    }
    pub const fn canonical_root_bytes(&self) -> u64 {
        self.canonical_root_bytes
    }
    pub const fn active_view_indices_bytes(&self) -> u64 {
        self.active_view_indices_bytes
    }
    pub const fn lazy_view_indices_capacity_bytes(&self) -> u64 {
        self.lazy_view_indices_capacity_bytes
    }
    pub const fn max_live_producer_bytes(&self) -> u64 {
        self.max_live_producer_bytes
    }
    pub const fn max_live_producer_scratch_bytes(&self) -> u64 {
        self.max_live_producer_scratch_bytes
    }
    pub const fn normalization_scratch_bytes(&self) -> u64 {
        self.normalization_scratch_bytes
    }
    pub const fn fit_metadata_bytes(&self) -> u64 {
        self.fit_metadata_bytes
    }
    pub const fn pointer_and_schema_metadata_bytes(&self) -> u64 {
        self.pointer_and_schema_metadata_bytes
    }
    pub const fn merkle_leaf_count(&self) -> u64 {
        self.merkle_leaf_count
    }
    pub const fn merkle_scratch_bytes(&self) -> u64 {
        self.merkle_scratch_bytes
    }
    pub const fn compact_hash_and_error_bytes(&self) -> u64 {
        self.compact_hash_and_error_bytes
    }
    pub const fn full_feature_major_staging_bytes(&self) -> u64 {
        self.full_feature_major_staging_bytes
    }
    pub const fn steady_device_bytes(&self) -> u64 {
        self.steady_device_bytes
    }
    pub const fn peak_device_bytes(&self) -> u64 {
        self.peak_device_bytes
    }
    pub const fn remaining_peak_after_parent_bytes(&self) -> u64 {
        // `parent_dataset_bytes` is already allocated and retained when the
        // runtime assembler takes its second same-context free-memory snapshot.
        self.peak_device_bytes - self.parent_dataset_bytes
    }
    pub const fn device_free_bytes_snapshot(&self) -> u64 {
        self.device_free_bytes_snapshot
    }
    pub const fn allocator_context_reserve_bytes(&self) -> u64 {
        self.allocator_context_reserve_bytes
    }
    pub fn reserve_policy_id(&self) -> &str {
        &self.reserve_policy_id
    }
    pub const fn available_device_bytes(&self) -> u64 {
        self.available_device_bytes
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuOnlyResidentAdmissionRequestV3 {
    pub dataset_recipe_sha256: [u8; SHA256_BYTES],
    pub feature_plan_schema_sha256: [u8; SHA256_BYTES],
    pub route_plan_sha256: [u8; SHA256_BYTES],
    pub admission_identity_sha256: [u8; SHA256_BYTES],
    pub planned_routes: Vec<ResidentFeatureRouteV3>,
    pub capabilities: ResidentProducerCapabilityManifestV3,
    pub device: CudaPrimaryContextBuildIdentityV3,
    pub working_set: ResidentWorkingSetBoundV3,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GpuOnlyResidentAdmissionV3 {
    schema: &'static str,
    dataset_recipe_sha256: [u8; SHA256_BYTES],
    feature_plan_schema_sha256: [u8; SHA256_BYTES],
    route_plan_sha256: [u8; SHA256_BYTES],
    admission_identity_sha256: [u8; SHA256_BYTES],
    planned_routes: Vec<ResidentFeatureRouteV3>,
    capabilities: ResidentProducerCapabilityManifestV3,
    device: CudaPrimaryContextBuildIdentityV3,
    working_set: ResidentWorkingSetBoundV3,
}

impl GpuOnlyResidentAdmissionV3 {
    pub fn seal(
        request: GpuOnlyResidentAdmissionRequestV3,
    ) -> Result<Self, ResidentFeatureContractErrorV3> {
        require_hash("dataset recipe sha256", &request.dataset_recipe_sha256)?;
        require_hash(
            "feature-plan schema sha256",
            &request.feature_plan_schema_sha256,
        )?;
        require_hash("route-plan sha256", &request.route_plan_sha256)?;
        require_hash(
            "admission identity sha256",
            &request.admission_identity_sha256,
        )?;
        if request.planned_routes.len()
            != usize::try_from(request.working_set.column_count).unwrap_or(usize::MAX)
        {
            return Err(ResidentFeatureContractErrorV3::LayoutMismatch {
                field: "phase-one route count",
                expected: request.working_set.column_count,
                actual: u64::try_from(request.planned_routes.len()).unwrap_or(u64::MAX),
            });
        }
        let mut names = BTreeSet::new();
        for (index, route) in request.planned_routes.iter().enumerate() {
            let expected = u64::try_from(index).map_err(|_| {
                ResidentFeatureContractErrorV3::ArithmeticOverflow {
                    field: "resident route ordinal",
                }
            })?;
            if route.ordinal() != expected {
                return Err(ResidentFeatureContractErrorV3::RouteOrderMismatch {
                    index,
                    actual: route.ordinal(),
                });
            }
            if !names.insert(route.feature_name().to_owned()) {
                return Err(ResidentFeatureContractErrorV3::DuplicateFeatureName {
                    name: route.feature_name().to_owned(),
                });
            }
            if !request
                .capabilities
                .capabilities()
                .iter()
                .any(|capability| capability.producer() == route.producer())
            {
                return Err(
                    ResidentFeatureContractErrorV3::MissingProducerCapabilities {
                        missing: vec![route.producer()],
                    },
                );
            }
        }
        Ok(Self {
            schema: GPU_ONLY_RESIDENT_ADMISSION_SCHEMA_V3,
            dataset_recipe_sha256: request.dataset_recipe_sha256,
            feature_plan_schema_sha256: request.feature_plan_schema_sha256,
            route_plan_sha256: request.route_plan_sha256,
            admission_identity_sha256: request.admission_identity_sha256,
            planned_routes: request.planned_routes,
            capabilities: request.capabilities,
            device: request.device,
            working_set: request.working_set,
        })
    }

    pub fn planned_routes(&self) -> &[ResidentFeatureRouteV3] {
        &self.planned_routes
    }
    pub const fn device(&self) -> &CudaPrimaryContextBuildIdentityV3 {
        &self.device
    }
    pub const fn working_set(&self) -> &ResidentWorkingSetBoundV3 {
        &self.working_set
    }
    pub const fn admission_identity_sha256(&self) -> [u8; SHA256_BYTES] {
        self.admission_identity_sha256
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ResidentValidityEncodingV3 {
    LosslessU4LogicalU8Sha256,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResidentFeatureLayoutRequestV3 {
    pub row_count: usize,
    pub column_count: usize,
    pub canonical_content_merkle_sha256: [u8; SHA256_BYTES],
    pub source_column_count: u64,
    pub producer_batch_count: u64,
    pub validity_initialization_count: u64,
    pub value_layout_launch_count: u64,
    pub validity_boundary_launch_count: u64,
    pub layout_transform_value_bytes: u64,
    pub layout_transform_logical_validity_bytes: u64,
    pub full_feature_major_staging_bytes: u64,
    pub max_live_producer_bytes: u64,
    pub max_live_producer_scratch_bytes: u64,
    pub pre_materialization_free_bytes_snapshot: u64,
    pub post_parent_free_bytes_snapshot: u64,
    pub retained_parent_dataset_bytes: u64,
    pub remaining_peak_after_parent_bytes: u64,
    pub allocator_context_reserve_bytes: u64,
    pub reserve_policy_id: String,
}

impl ResidentFeatureLayoutRequestV3 {
    pub fn seal(self) -> Result<ResidentFeatureLayoutV3, ResidentFeatureContractErrorV3> {
        ResidentFeatureLayoutV3::seal(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResidentFeatureLayoutV3 {
    row_count: u64,
    column_count: u64,
    canonical_content_merkle_sha256: [u8; SHA256_BYTES],
    validity_schema: &'static str,
    validity_encoding: ResidentValidityEncodingV3,
    validity_max_code: u8,
    validity_low_nibble_is_earlier_cell: bool,
    validity_odd_padding_nibble: u8,
    source_column_count: u64,
    producer_batch_count: u64,
    validity_initialization_count: u64,
    value_layout_launch_count: u64,
    validity_boundary_launch_count: u64,
    layout_transform_value_bytes: u64,
    layout_transform_logical_validity_bytes: u64,
    resident_packed_validity_logical_bytes: u64,
    resident_packed_validity_allocated_bytes: u64,
    full_feature_major_staging_bytes: u64,
    max_live_producer_bytes: u64,
    max_live_producer_scratch_bytes: u64,
    pre_materialization_free_bytes_snapshot: u64,
    post_parent_free_bytes_snapshot: u64,
    retained_parent_dataset_bytes: u64,
    remaining_peak_after_parent_bytes: u64,
    allocator_context_reserve_bytes: u64,
    reserve_policy_id: String,
}

impl ResidentFeatureLayoutV3 {
    fn seal(
        request: ResidentFeatureLayoutRequestV3,
    ) -> Result<Self, ResidentFeatureContractErrorV3> {
        require_hash(
            "canonical feature content Merkle sha256",
            &request.canonical_content_merkle_sha256,
        )?;
        if request.row_count == 0 || request.column_count == 0 {
            return Err(ResidentFeatureContractErrorV3::EmptyField {
                field: "resident layout rows and columns",
            });
        }
        let row_count = u64::try_from(request.row_count).map_err(|_| {
            ResidentFeatureContractErrorV3::ArithmeticOverflow {
                field: "resident layout row count",
            }
        })?;
        let column_count = u64::try_from(request.column_count).map_err(|_| {
            ResidentFeatureContractErrorV3::ArithmeticOverflow {
                field: "resident layout column count",
            }
        })?;
        let cells = checked_mul(row_count, column_count, "resident layout cell count")?;
        let expected_value_bytes = checked_mul(cells, F64_BYTES, "layout f64 bytes")?;
        let expected_logical_validity_bytes = checked_mul(
            cells,
            PRODUCER_VALIDITY_BYTES,
            "layout logical validity bytes",
        )?;
        let packed_logical_bytes = packed_validity_logical_bytes(cells);
        let packed_allocated_bytes = align_up_u64(
            packed_logical_bytes,
            VALIDITY_ATOMIC_ALIGNMENT_BYTES,
            "layout packed validity allocation",
        )?;
        for (field, expected, actual) in [
            (
                "source column count",
                column_count,
                request.source_column_count,
            ),
            (
                "layout-transform value bytes",
                expected_value_bytes,
                request.layout_transform_value_bytes,
            ),
            (
                "layout-transform logical validity bytes",
                expected_logical_validity_bytes,
                request.layout_transform_logical_validity_bytes,
            ),
            (
                "full feature-major staging bytes",
                0,
                request.full_feature_major_staging_bytes,
            ),
        ] {
            if expected != actual {
                return Err(ResidentFeatureContractErrorV3::LayoutMismatch {
                    field,
                    expected,
                    actual,
                });
            }
        }
        if request.producer_batch_count == 0 || request.producer_batch_count > column_count {
            return Err(ResidentFeatureContractErrorV3::LayoutMismatch {
                field: "producer batch count",
                expected: column_count,
                actual: request.producer_batch_count,
            });
        }
        require_text(
            "resident layout allocator/context reserve policy id",
            &request.reserve_policy_id,
        )?;
        if request.pre_materialization_free_bytes_snapshot == 0
            || request.post_parent_free_bytes_snapshot == 0
            || request.retained_parent_dataset_bytes == 0
            || request.remaining_peak_after_parent_bytes == 0
        {
            return Err(ResidentFeatureContractErrorV3::EmptyField {
                field: "resident two-phase free-memory evidence",
            });
        }
        let post_parent_available = request
            .post_parent_free_bytes_snapshot
            .checked_sub(request.allocator_context_reserve_bytes)
            .ok_or(ResidentFeatureContractErrorV3::WorkingSetExceedsDevice {
                required_bytes: request.remaining_peak_after_parent_bytes,
                available_bytes: 0,
            })?;
        if request.remaining_peak_after_parent_bytes > post_parent_available {
            return Err(ResidentFeatureContractErrorV3::WorkingSetExceedsDevice {
                required_bytes: request.remaining_peak_after_parent_bytes,
                available_bytes: post_parent_available,
            });
        }
        if request.validity_initialization_count != 1 {
            return Err(ResidentFeatureContractErrorV3::LayoutMismatch {
                field: "validity initialization count",
                expected: 1,
                actual: request.validity_initialization_count,
            });
        }
        for (field, actual) in [
            (
                "value layout launch count",
                request.value_layout_launch_count,
            ),
            (
                "validity boundary launch count",
                request.validity_boundary_launch_count,
            ),
        ] {
            if actual != request.producer_batch_count {
                return Err(ResidentFeatureContractErrorV3::LayoutMismatch {
                    field,
                    expected: request.producer_batch_count,
                    actual,
                });
            }
        }
        Ok(Self {
            row_count,
            column_count,
            canonical_content_merkle_sha256: request.canonical_content_merkle_sha256,
            validity_schema: RESIDENT_VALIDITY_SCHEMA_V3,
            validity_encoding: ResidentValidityEncodingV3::LosslessU4LogicalU8Sha256,
            validity_max_code: MAX_VALIDITY_CODE_V3,
            validity_low_nibble_is_earlier_cell: true,
            validity_odd_padding_nibble: 0,
            source_column_count: request.source_column_count,
            producer_batch_count: request.producer_batch_count,
            validity_initialization_count: request.validity_initialization_count,
            value_layout_launch_count: request.value_layout_launch_count,
            validity_boundary_launch_count: request.validity_boundary_launch_count,
            layout_transform_value_bytes: request.layout_transform_value_bytes,
            layout_transform_logical_validity_bytes: request
                .layout_transform_logical_validity_bytes,
            resident_packed_validity_logical_bytes: packed_logical_bytes,
            resident_packed_validity_allocated_bytes: packed_allocated_bytes,
            full_feature_major_staging_bytes: request.full_feature_major_staging_bytes,
            max_live_producer_bytes: request.max_live_producer_bytes,
            max_live_producer_scratch_bytes: request.max_live_producer_scratch_bytes,
            pre_materialization_free_bytes_snapshot: request
                .pre_materialization_free_bytes_snapshot,
            post_parent_free_bytes_snapshot: request.post_parent_free_bytes_snapshot,
            retained_parent_dataset_bytes: request.retained_parent_dataset_bytes,
            remaining_peak_after_parent_bytes: request.remaining_peak_after_parent_bytes,
            allocator_context_reserve_bytes: request.allocator_context_reserve_bytes,
            reserve_policy_id: request.reserve_policy_id,
        })
    }

    pub const fn canonical_content_merkle_sha256(&self) -> [u8; SHA256_BYTES] {
        self.canonical_content_merkle_sha256
    }
    pub const fn row_count(&self) -> u64 {
        self.row_count
    }
    pub const fn column_count(&self) -> u64 {
        self.column_count
    }
    pub const fn validity_encoding(&self) -> ResidentValidityEncodingV3 {
        self.validity_encoding
    }
    pub const fn source_column_count(&self) -> u64 {
        self.source_column_count
    }
    pub const fn producer_batch_count(&self) -> u64 {
        self.producer_batch_count
    }
    pub const fn validity_initialization_count(&self) -> u64 {
        self.validity_initialization_count
    }
    pub const fn layout_transform_launch_count(&self) -> u64 {
        self.value_layout_launch_count + self.validity_boundary_launch_count
    }
    pub const fn full_feature_major_staging_bytes(&self) -> u64 {
        self.full_feature_major_staging_bytes
    }
    pub const fn pre_materialization_free_bytes_snapshot(&self) -> u64 {
        self.pre_materialization_free_bytes_snapshot
    }
    pub const fn post_parent_free_bytes_snapshot(&self) -> u64 {
        self.post_parent_free_bytes_snapshot
    }
    pub const fn retained_parent_dataset_bytes(&self) -> u64 {
        self.retained_parent_dataset_bytes
    }
    pub const fn remaining_peak_after_parent_bytes(&self) -> u64 {
        self.remaining_peak_after_parent_bytes
    }
    pub const fn allocator_context_reserve_bytes(&self) -> u64 {
        self.allocator_context_reserve_bytes
    }
    pub fn reserve_policy_id(&self) -> &str {
        &self.reserve_policy_id
    }
}

pub const RESIDENT_PARENT_DATASET_LAYOUT_AUTHORITY_V4: &str =
    "neoethos.resident-parent-dataset-layout.v4";

/// Compact identity and exact resident extents for the immutable native
/// evaluator parent. These nine arrays remain device-resident with the final
/// feature store; Search never reconstructs or reuploads them. V4 adds the
/// canonical open and volume arrays required by the admitted Classic/vector-ta
/// graph; the old seven-array V3 layout is not an active alias.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResidentParentDatasetLayoutV4 {
    authority: &'static str,
    row_count: u64,
    open_sha256: [u8; SHA256_BYTES],
    high_sha256: [u8; SHA256_BYTES],
    low_sha256: [u8; SHA256_BYTES],
    close_sha256: [u8; SHA256_BYTES],
    volume_sha256: [u8; SHA256_BYTES],
    timestamps_sha256: [u8; SHA256_BYTES],
    months_sha256: [u8; SHA256_BYTES],
    days_sha256: [u8; SHA256_BYTES],
    smc_rows_sha256: [u8; SHA256_BYTES],
    ohlcv_bytes: u64,
    clock_bytes: u64,
    smc_bytes: u64,
}

impl ResidentParentDatasetLayoutV4 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        row_count: usize,
        open_sha256: [u8; SHA256_BYTES],
        high_sha256: [u8; SHA256_BYTES],
        low_sha256: [u8; SHA256_BYTES],
        close_sha256: [u8; SHA256_BYTES],
        volume_sha256: [u8; SHA256_BYTES],
        timestamps_sha256: [u8; SHA256_BYTES],
        months_sha256: [u8; SHA256_BYTES],
        days_sha256: [u8; SHA256_BYTES],
        smc_rows_sha256: [u8; SHA256_BYTES],
    ) -> Result<Self, ResidentFeatureContractErrorV3> {
        if row_count == 0 {
            return Err(ResidentFeatureContractErrorV3::EmptyField {
                field: "resident parent row count",
            });
        }
        for (field, hash) in [
            ("resident open sha256", &open_sha256),
            ("resident high sha256", &high_sha256),
            ("resident low sha256", &low_sha256),
            ("resident close sha256", &close_sha256),
            ("resident volume sha256", &volume_sha256),
            ("resident timestamps sha256", &timestamps_sha256),
            ("resident months sha256", &months_sha256),
            ("resident days sha256", &days_sha256),
            ("resident SMC rows sha256", &smc_rows_sha256),
        ] {
            require_hash(field, hash)?;
        }
        let row_count = u64::try_from(row_count).map_err(|_| {
            ResidentFeatureContractErrorV3::ArithmeticOverflow {
                field: "resident parent row count",
            }
        })?;
        let ohlcv_bytes = checked_mul(
            checked_mul(row_count, 5, "resident parent OHLCV arrays")?,
            F64_BYTES,
            "resident parent OHLCV bytes",
        )?;
        let clock_bytes = checked_mul(
            checked_mul(row_count, 3, "resident parent clock arrays")?,
            I64_BYTES,
            "resident parent clock bytes",
        )?;
        let smc_bytes = checked_mul(row_count, SMC_SLOTS_V3, "resident parent SMC bytes")?;
        Ok(Self {
            authority: RESIDENT_PARENT_DATASET_LAYOUT_AUTHORITY_V4,
            row_count,
            open_sha256,
            high_sha256,
            low_sha256,
            close_sha256,
            volume_sha256,
            timestamps_sha256,
            months_sha256,
            days_sha256,
            smc_rows_sha256,
            ohlcv_bytes,
            clock_bytes,
            smc_bytes,
        })
    }

    pub const fn row_count(&self) -> u64 {
        self.row_count
    }
    pub const fn authority(&self) -> &'static str {
        self.authority
    }
    pub const fn ohlcv_bytes(&self) -> u64 {
        self.ohlcv_bytes
    }
    pub const fn clock_bytes(&self) -> u64 {
        self.clock_bytes
    }
    pub const fn smc_bytes(&self) -> u64 {
        self.smc_bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResidentReadyEventV3 {
    device_ordinal: u32,
    primary_context_process_token: [u8; SHA256_BYTES],
    producer_stream_process_token: [u8; SHA256_BYTES],
    ready_event_process_token: [u8; SHA256_BYTES],
    producer_record_sequence: u64,
    interop_abi: &'static str,
    host_synchronize_count: u64,
}

impl ResidentReadyEventV3 {
    pub fn new(
        device_ordinal: u32,
        primary_context_process_token: [u8; SHA256_BYTES],
        producer_stream_process_token: [u8; SHA256_BYTES],
        ready_event_process_token: [u8; SHA256_BYTES],
        producer_record_sequence: u64,
    ) -> Result<Self, ResidentFeatureContractErrorV3> {
        require_hash(
            "ready-event primary context process token",
            &primary_context_process_token,
        )?;
        require_hash(
            "ready-event producer stream process token",
            &producer_stream_process_token,
        )?;
        require_hash("ready-event process token", &ready_event_process_token)?;
        if producer_record_sequence == 0 {
            return Err(ResidentFeatureContractErrorV3::EmptyField {
                field: "ready-event producer record sequence",
            });
        }
        Ok(Self {
            device_ordinal,
            primary_context_process_token,
            producer_stream_process_token,
            ready_event_process_token,
            producer_record_sequence,
            interop_abi: READY_EVENT_INTEROP_ABI_V3,
            host_synchronize_count: 0,
        })
    }

    pub const fn interop_abi(&self) -> &'static str {
        self.interop_abi
    }
    pub const fn host_synchronize_count(&self) -> u64 {
        self.host_synchronize_count
    }
    pub const fn recorded_after_final_incremental_layout_normalization_and_merkle(&self) -> bool {
        true
    }
    pub const fn consumer_must_wait_before_first_read(&self) -> bool {
        true
    }
    pub const fn retains_store_until_consumer_completion(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CuPqcHostCompilerV3 {
    Gcc,
    Clang,
    Msvc,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CuPqcSupportProbeV3 {
    target_os: String,
    target_arch: String,
    host_compiler: CuPqcHostCompilerV3,
    cuda_major: u16,
    cuda_minor: u16,
    sm: u16,
    package_present: bool,
    cpp17_enabled: bool,
    device_lto_enabled: bool,
    redistribution_terms_accepted: bool,
}

impl CuPqcSupportProbeV3 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        target_os: impl Into<String>,
        target_arch: impl Into<String>,
        host_compiler: CuPqcHostCompilerV3,
        cuda_major: u16,
        cuda_minor: u16,
        sm: u16,
        package_present: bool,
        cpp17_enabled: bool,
        device_lto_enabled: bool,
        redistribution_terms_accepted: bool,
    ) -> Self {
        Self {
            target_os: target_os.into(),
            target_arch: target_arch.into(),
            host_compiler,
            cuda_major,
            cuda_minor,
            sm,
            package_present,
            cpp17_enabled,
            device_lto_enabled,
            redistribution_terms_accepted,
        }
    }

    /// Current cuPQC 0.4.1 support remains optional. Windows/MSVC and any
    /// unlisted SM require the portable in-tree implementation.
    pub fn optional_acceleration_supported(&self) -> bool {
        let supported_os = self.target_os == "linux";
        let supported_arch = matches!(self.target_arch.as_str(), "x86_64" | "aarch64");
        let supported_compiler = matches!(
            self.host_compiler,
            CuPqcHostCompilerV3::Gcc | CuPqcHostCompilerV3::Clang
        );
        let supported_cuda = (self.cuda_major, self.cuda_minor) >= (12, 8);
        let supported_sm = matches!(self.sm, 70 | 75 | 80 | 86 | 87 | 89 | 90);
        supported_os
            && supported_arch
            && supported_compiler
            && supported_cuda
            && supported_sm
            && self.package_present
            && self.cpp17_enabled
            && self.device_lto_enabled
            && self.redistribution_terms_accepted
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CanonicalCudaSha256AuthorityV3 {
    portable_implementation_id: &'static str,
    optional_cupqc_acceleration: Option<CuPqcSupportProbeV3>,
}

impl CanonicalCudaSha256AuthorityV3 {
    pub const fn portable_in_tree() -> Self {
        Self {
            portable_implementation_id: PORTABLE_CUDA_SHA256_AUTHORITY_V3,
            optional_cupqc_acceleration: None,
        }
    }

    pub fn with_optional_cupqc(mut self, probe: CuPqcSupportProbeV3) -> Self {
        if probe.optional_acceleration_supported() {
            self.optional_cupqc_acceleration = Some(probe);
        }
        self
    }

    pub const fn portable_path_is_mandatory(&self) -> bool {
        !self.portable_implementation_id.is_empty()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SealedResidentFeatureStoreRequestV3 {
    pub admission_identity_sha256: [u8; SHA256_BYTES],
    pub final_feature_plan_v3_sha256: [u8; SHA256_BYTES],
    pub normalization_fit_sha256: [u8; SHA256_BYTES],
    pub source_provenance_sha256: [u8; SHA256_BYTES],
    pub ordered_feature_names: Vec<String>,
    pub layout: ResidentFeatureLayoutV3,
    pub parent_dataset: ResidentParentDatasetLayoutV4,
    pub ready_event: ResidentReadyEventV3,
    pub sha256_authority: CanonicalCudaSha256AuthorityV3,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SealedResidentFeatureStoreV3 {
    schema: &'static str,
    admission_identity_sha256: [u8; SHA256_BYTES],
    final_feature_plan_v3_sha256: [u8; SHA256_BYTES],
    normalization_fit_sha256: [u8; SHA256_BYTES],
    source_provenance_sha256: [u8; SHA256_BYTES],
    ordered_feature_names: Vec<String>,
    device: CudaPrimaryContextBuildIdentityV3,
    layout: ResidentFeatureLayoutV3,
    parent_dataset: ResidentParentDatasetLayoutV4,
    ready_event: ResidentReadyEventV3,
    sha256_authority: CanonicalCudaSha256AuthorityV3,
}

impl SealedResidentFeatureStoreV3 {
    pub fn seal(
        admission: &GpuOnlyResidentAdmissionV3,
        request: SealedResidentFeatureStoreRequestV3,
    ) -> Result<Self, ResidentFeatureContractErrorV3> {
        if request.admission_identity_sha256 != admission.admission_identity_sha256 {
            return Err(ResidentFeatureContractErrorV3::AdmissionIdentityMismatch);
        }
        require_hash(
            "final FeaturePlanV3 sha256",
            &request.final_feature_plan_v3_sha256,
        )?;
        require_hash(
            "normalization fit sha256",
            &request.normalization_fit_sha256,
        )?;
        require_hash(
            "source provenance sha256",
            &request.source_provenance_sha256,
        )?;
        if request.ready_event.device_ordinal != admission.device.ordinal
            || request.layout.row_count != admission.working_set.row_count
            || request.layout.column_count != admission.working_set.column_count
            || request.layout.resident_packed_validity_allocated_bytes
                != admission.working_set.packed_validity_allocated_bytes
            || request.layout.max_live_producer_bytes
                != admission.working_set.max_live_producer_bytes
            || request.layout.max_live_producer_scratch_bytes
                != admission.working_set.max_live_producer_scratch_bytes
            || request.layout.pre_materialization_free_bytes_snapshot
                != admission.working_set.device_free_bytes_snapshot
            || request.layout.remaining_peak_after_parent_bytes
                != admission.working_set.remaining_peak_after_parent_bytes()
            || request.layout.retained_parent_dataset_bytes
                != admission.working_set.parent_dataset_bytes
            || request.layout.allocator_context_reserve_bytes
                != admission.working_set.allocator_context_reserve_bytes
            || request.layout.reserve_policy_id != admission.working_set.reserve_policy_id
            || request.parent_dataset.row_count != admission.working_set.row_count
            || request.parent_dataset.authority != RESIDENT_PARENT_DATASET_LAYOUT_AUTHORITY_V4
            || request.parent_dataset.ohlcv_bytes != admission.working_set.parent_ohlcv_bytes
            || request.parent_dataset.clock_bytes != admission.working_set.parent_clock_bytes
            || request.parent_dataset.smc_bytes != admission.working_set.parent_smc_bytes
        {
            return Err(ResidentFeatureContractErrorV3::DeviceIdentityMismatch);
        }
        if request.ready_event.primary_context_process_token
            != admission.device.primary_context_process_token
        {
            return Err(ResidentFeatureContractErrorV3::PrimaryContextMismatch);
        }
        let admitted_names = admission
            .planned_routes
            .iter()
            .map(|route| route.feature_name.as_str());
        if admitted_names.ne(request.ordered_feature_names.iter().map(String::as_str)) {
            return Err(ResidentFeatureContractErrorV3::OrderedSchemaMismatch);
        }
        if !request.sha256_authority.portable_path_is_mandatory() {
            return Err(ResidentFeatureContractErrorV3::Sha256AuthorityMissingPortablePath);
        }
        Ok(Self {
            schema: SEALED_RESIDENT_FEATURE_STORE_SCHEMA_V3,
            admission_identity_sha256: request.admission_identity_sha256,
            final_feature_plan_v3_sha256: request.final_feature_plan_v3_sha256,
            normalization_fit_sha256: request.normalization_fit_sha256,
            source_provenance_sha256: request.source_provenance_sha256,
            ordered_feature_names: request.ordered_feature_names,
            device: admission.device.clone(),
            layout: request.layout,
            parent_dataset: request.parent_dataset,
            ready_event: request.ready_event,
            sha256_authority: request.sha256_authority,
        })
    }

    pub const fn layout(&self) -> &ResidentFeatureLayoutV3 {
        &self.layout
    }
    pub const fn ready_event(&self) -> &ResidentReadyEventV3 {
        &self.ready_event
    }
    pub const fn canonical_feature_content_merkle_sha256(&self) -> [u8; SHA256_BYTES] {
        self.layout.canonical_content_merkle_sha256
    }
}

/// Packs exact producer-owned logical validity codes. The earlier logical cell
/// occupies the low nibble; an odd final high nibble is canonical zero.
pub fn pack_logical_validity_u4_v3(
    logical_validity: &[u8],
) -> Result<Vec<u8>, ResidentFeatureContractErrorV3> {
    let mut packed = vec![0_u8; logical_validity.len().div_ceil(2)];
    for (index, code) in logical_validity.iter().copied().enumerate() {
        if code > MAX_VALIDITY_CODE_V3 {
            return Err(ResidentFeatureContractErrorV3::InvalidValidityCode { index, code });
        }
        if index & 1 == 0 {
            packed[index / 2] = code;
        } else {
            packed[index / 2] |= code << 4;
        }
    }
    Ok(packed)
}

fn logical_validity_code(
    packed_validity_u4: &[u8],
    cell: usize,
) -> Result<u8, ResidentFeatureContractErrorV3> {
    let byte = packed_validity_u4[cell / 2];
    let code = if cell & 1 == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    };
    if code > MAX_VALIDITY_CODE_V3 {
        Err(ResidentFeatureContractErrorV3::InvalidValidityCode { index: cell, code })
    } else {
        Ok(code)
    }
}

fn update_u64_le(hasher: &mut Sha256, value: u64) {
    hasher.update(value.to_le_bytes());
}

fn finish_sha256(hasher: Sha256) -> [u8; SHA256_BYTES] {
    hasher.finalize().into()
}

/// Exact CPU oracle for the device V3 tree. Values are raw `f64::to_bits`
/// words in bar-major order. Validity is physically u4 but hashed as the exact
/// producer-owned logical u8 byte for every cell.
pub fn canonical_feature_merkle_sha256_host_oracle_v3(
    timestamps: &[i64],
    ordered_feature_names: &[String],
    bar_major_value_bits: &[u64],
    packed_validity_u4: &[u8],
) -> Result<[u8; SHA256_BYTES], ResidentFeatureContractErrorV3> {
    if timestamps.is_empty() || ordered_feature_names.is_empty() {
        return Err(ResidentFeatureContractErrorV3::EmptyField {
            field: "V3 Merkle rows and columns",
        });
    }
    let rows = timestamps.len();
    let columns = ordered_feature_names.len();
    let cells =
        rows.checked_mul(columns)
            .ok_or(ResidentFeatureContractErrorV3::ArithmeticOverflow {
                field: "V3 Merkle cells",
            })?;
    if bar_major_value_bits.len() != cells {
        return Err(ResidentFeatureContractErrorV3::LayoutMismatch {
            field: "V3 Merkle value cells",
            expected: u64::try_from(cells).unwrap_or(u64::MAX),
            actual: u64::try_from(bar_major_value_bits.len()).unwrap_or(u64::MAX),
        });
    }
    let expected_packed = cells.div_ceil(2);
    if packed_validity_u4.len() != expected_packed {
        return Err(ResidentFeatureContractErrorV3::InvalidPackedValidity {
            reason: "packed byte extent does not equal ceil(logical cells / 2)",
        });
    }
    if cells & 1 == 1
        && packed_validity_u4
            .last()
            .is_some_and(|byte| byte & 0xf0 != 0)
    {
        return Err(ResidentFeatureContractErrorV3::InvalidPackedValidity {
            reason: "odd-cell high padding nibble is not zero",
        });
    }
    let mut unique_names = BTreeSet::new();
    for name in ordered_feature_names {
        require_text("V3 Merkle feature name", name)?;
        if !unique_names.insert(name.as_str()) {
            return Err(ResidentFeatureContractErrorV3::DuplicateFeatureName {
                name: name.clone(),
            });
        }
    }
    for cell in 0..cells {
        let _ = logical_validity_code(packed_validity_u4, cell)?;
    }

    let timestamp_chunk_count = rows.div_ceil(CANONICAL_MERKLE_CHUNK_ROWS_V3);
    let leaf_count = timestamp_chunk_count
        .checked_mul(columns.checked_add(1).ok_or(
            ResidentFeatureContractErrorV3::ArithmeticOverflow {
                field: "V3 Merkle producer count",
            },
        )?)
        .ok_or(ResidentFeatureContractErrorV3::ArithmeticOverflow {
            field: "V3 Merkle leaf count",
        })?;
    let mut leaves = Vec::with_capacity(leaf_count);

    for chunk in 0..timestamp_chunk_count {
        let row_start = chunk * CANONICAL_MERKLE_CHUNK_ROWS_V3;
        let row_end = rows.min(row_start + CANONICAL_MERKLE_CHUNK_ROWS_V3);
        let mut hasher = Sha256::new();
        hasher.update(CANONICAL_FEATURE_MERKLE_LEAF_DOMAIN_V3);
        hasher.update([0_u8]);
        update_u64_le(&mut hasher, chunk as u64);
        update_u64_le(&mut hasher, row_start as u64);
        update_u64_le(&mut hasher, (row_end - row_start) as u64);
        for timestamp in &timestamps[row_start..row_end] {
            update_u64_le(&mut hasher, *timestamp as u64);
        }
        leaves.push(finish_sha256(hasher));
    }

    for chunk in 0..timestamp_chunk_count {
        let row_start = chunk * CANONICAL_MERKLE_CHUNK_ROWS_V3;
        let row_end = rows.min(row_start + CANONICAL_MERKLE_CHUNK_ROWS_V3);
        for (column, name) in ordered_feature_names.iter().enumerate() {
            let leaf_ordinal = timestamp_chunk_count + chunk * columns + column;
            let mut hasher = Sha256::new();
            hasher.update(CANONICAL_FEATURE_MERKLE_LEAF_DOMAIN_V3);
            hasher.update([1_u8]);
            update_u64_le(&mut hasher, leaf_ordinal as u64);
            update_u64_le(&mut hasher, row_start as u64);
            update_u64_le(&mut hasher, (row_end - row_start) as u64);
            update_u64_le(&mut hasher, column as u64);
            update_u64_le(&mut hasher, name.len() as u64);
            hasher.update(name.as_bytes());
            for row in row_start..row_end {
                let cell = row * columns + column;
                update_u64_le(&mut hasher, bar_major_value_bits[cell]);
                hasher.update([logical_validity_code(packed_validity_u4, cell)?]);
            }
            leaves.push(finish_sha256(hasher));
        }
    }

    let mut level = 0_u64;
    while leaves.len() > 1 {
        let mut parents = Vec::with_capacity(leaves.len().div_ceil(2));
        for (node, children) in leaves.chunks(2).enumerate() {
            let mut hasher = Sha256::new();
            hasher.update(CANONICAL_FEATURE_MERKLE_NODE_DOMAIN_V3);
            update_u64_le(&mut hasher, level);
            update_u64_le(&mut hasher, node as u64);
            hasher.update([children.len() as u8]);
            for child in children {
                hasher.update(child);
            }
            parents.push(finish_sha256(hasher));
        }
        leaves = parents;
        level += 1;
    }

    let tree_root = leaves
        .pop()
        .expect("nonempty rows and columns always create at least one V3 leaf");
    let mut hasher = Sha256::new();
    hasher.update(CANONICAL_FEATURE_CONTENT_HASH_DOMAIN_V3);
    update_u64_le(&mut hasher, rows as u64);
    update_u64_le(&mut hasher, columns as u64);
    update_u64_le(&mut hasher, CANONICAL_MERKLE_CHUNK_ROWS_V3 as u64);
    update_u64_le(&mut hasher, timestamp_chunk_count as u64);
    update_u64_le(&mut hasher, leaf_count as u64);
    hasher.update(tree_root);
    Ok(finish_sha256(hasher))
}
