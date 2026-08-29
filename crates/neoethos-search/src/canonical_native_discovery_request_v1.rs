use std::fmt;
use std::path::Path;

#[cfg(all(feature = "gpu-cuda", target_os = "linux"))]
use std::collections::HashSet;

use serde::{Deserialize, Deserializer, Serialize};
#[cfg(target_os = "linux")]
use sha2::{Digest, Sha256};

use crate::canonical_native_root_io_v1::SealedCanonicalRootV1;
#[cfg(target_os = "linux")]
use crate::canonical_native_root_io_v1::read_canonical_artifact_exact_v1;
#[cfg(all(feature = "gpu-cuda", target_os = "linux"))]
use crate::canonical_native_runtime_authority_v1::seal_generation_zero_runtime_authority_v1;
use crate::canonical_native_runtime_authority_v1::{
    CanonicalNativeGenerationZeroRuntimeAuthorityV1, CanonicalNativeRuntimeInstallReceiptV1,
};
use crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3;

pub const CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_SCHEMA_V1: &str =
    "neoethos.canonical-research-contract-artifact-ref.v1";
pub const CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_VERSION_V1: u16 = 1;
pub const MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1: u64 = 512 * 1024 * 1024;
pub const MAX_CANONICAL_NATIVE_GEN0_CONFIGURED_POPULATION_V1: usize = 1_000_000;
pub const MAX_CANONICAL_NATIVE_GEN0_RESOLVED_POPULATION_V1: usize = 1_000_000;
pub const MAX_CANONICAL_NATIVE_GEN0_TERMS_V1: usize = 4_096;
pub const MAX_CANONICAL_NATIVE_GEN0_STRING_BYTES_V1: usize = 64 * 1024;
pub const MAX_CANONICAL_NATIVE_GEN0_VECTOR_ELEMENTS_V1: usize = 1_000_000;
pub const MAX_CANONICAL_NATIVE_GEN0_SOURCE_COUNT_V1: usize =
    neoethos_data::CANONICAL_TIMEFRAMES.len();

#[derive(Debug, PartialEq, Eq)]
pub enum CanonicalNativeDiscoveryRequestErrorV1 {
    InvalidArtifactReference(String),
    UnsupportedPlatform,
    CanonicalRootUnavailable(String),
    SecureResolutionUnavailable(String),
    UnsafeLink,
    EscapeOrMount,
    RaceDetected,
    NonRegularArtifact,
    ArtifactTooLarge { maximum: u64, observed: u64 },
    ArtifactIo(String),
    ArtifactHashMismatch { expected: String, actual: String },
    ContractDecode(String),
    ContractValidation(String),
    InvalidGenerationZeroOverrides(String),
    RuntimeAuthority(String),
    MigrationEnabled,
    ContractSettingsMismatch(String),
    UnsupportedGenerationZeroPolicy { policy: &'static str },
    DatasetSeries(String),
    ExactDatasetGenerationConflict(neoethos_data::ExactDatasetGenerationConflict),
    RequestLimitExceeded { limit: &'static str },
}

impl fmt::Display for CanonicalNativeDiscoveryRequestErrorV1 {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArtifactReference(reason) => write!(output, "invalid reference: {reason}"),
            Self::UnsupportedPlatform => output.write_str("unsupported platform"),
            Self::CanonicalRootUnavailable(reason) => write!(output, "root unavailable: {reason}"),
            Self::SecureResolutionUnavailable(reason) => {
                write!(output, "secure resolution unavailable: {reason}")
            }
            Self::UnsafeLink => output.write_str("unsafe link"),
            Self::EscapeOrMount => output.write_str("root escape or mount crossing"),
            Self::RaceDetected => output.write_str("artifact/root race detected"),
            Self::NonRegularArtifact => output.write_str("artifact is not regular"),
            Self::ArtifactTooLarge { maximum, observed } => {
                write!(output, "artifact too large: {observed}>{maximum}")
            }
            Self::ArtifactIo(reason) => write!(output, "artifact I/O: {reason}"),
            Self::ArtifactHashMismatch { expected, actual } => {
                write!(output, "artifact hash mismatch: {actual}!={expected}")
            }
            Self::ContractDecode(reason) => write!(output, "contract decode: {reason}"),
            Self::ContractValidation(reason) => write!(output, "contract invalid: {reason}"),
            Self::InvalidGenerationZeroOverrides(reason) => {
                write!(output, "invalid Generation-zero overrides: {reason}")
            }
            Self::RuntimeAuthority(reason) => write!(output, "runtime authority: {reason}"),
            Self::MigrationEnabled => {
                output.write_str("federation migration must already be disabled")
            }
            Self::ContractSettingsMismatch(reason) => {
                write!(output, "contract/startup Settings mismatch: {reason}")
            }
            Self::UnsupportedGenerationZeroPolicy { policy } => {
                write!(output, "unsupported Generation-zero policy: {policy}")
            }
            Self::DatasetSeries(reason) => write!(output, "exact dataset series: {reason}"),
            Self::ExactDatasetGenerationConflict(conflict) => conflict.fmt(output),
            Self::RequestLimitExceeded { limit } => {
                write!(output, "canonical native request exceeds {limit}")
            }
        }
    }
}

impl std::error::Error for CanonicalNativeDiscoveryRequestErrorV1 {}

#[derive(Clone, Debug, Default, Serialize)]
pub struct CanonicalNativeGenerationZeroOverridesV1 {
    population: Option<usize>,
    population_auto: Option<bool>,
    max_indicators: Option<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GenerationZeroOverridesWireV1 {
    #[serde(default)]
    population: Option<usize>,
    #[serde(default)]
    population_auto: Option<bool>,
    #[serde(default)]
    max_indicators: Option<usize>,
}

impl CanonicalNativeGenerationZeroOverridesV1 {
    pub fn checked_new(
        population: Option<usize>,
        population_auto: Option<bool>,
        max_indicators: Option<usize>,
    ) -> Result<Self, CanonicalNativeDiscoveryRequestErrorV1> {
        let overrides = Self {
            population,
            population_auto,
            max_indicators,
        };
        overrides.validate()?;
        Ok(overrides)
    }

    pub const fn population(&self) -> Option<usize> {
        self.population
    }

    pub const fn population_auto(&self) -> Option<bool> {
        self.population_auto
    }

    pub const fn max_indicators(&self) -> Option<usize> {
        self.max_indicators
    }

    fn validate(&self) -> Result<(), CanonicalNativeDiscoveryRequestErrorV1> {
        if self.population.is_some_and(|value| {
            !(10..=MAX_CANONICAL_NATIVE_GEN0_CONFIGURED_POPULATION_V1).contains(&value)
        }) {
            return Err(invalid_overrides("population is outside the named V1 cap"));
        }
        if self
            .max_indicators
            .is_some_and(|value| value > MAX_CANONICAL_NATIVE_GEN0_TERMS_V1)
        {
            return Err(invalid_overrides(
                "max_indicators is outside the named V1 term cap",
            ));
        }
        Ok(())
    }
}

impl<'de> Deserialize<'de> for CanonicalNativeGenerationZeroOverridesV1 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = GenerationZeroOverridesWireV1::deserialize(deserializer)?;
        Self::checked_new(wire.population, wire.population_auto, wire.max_indicators)
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalNativeExecutionScopeV1 {
    GenerationZeroOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalNativeCostBandStatusV1 {
    UnusedGenerationZero,
}

pub struct CanonicalNativeGenerationZeroScopeV1 {
    execution_scope: CanonicalNativeExecutionScopeV1,
    raw_legacy_generations_unused_full_search: usize,
    clamped_legacy_generations_unused_full_search: usize,
    cost_band_status: CanonicalNativeCostBandStatusV1,
    cost_band_pips_unused_generation_zero: Option<(f64, f64)>,
    identity_sha256: String,
}

impl CanonicalNativeGenerationZeroScopeV1 {
    #[cfg(all(feature = "gpu-cuda", target_os = "linux"))]
    fn seal(raw: usize, clamped: usize, cost_band: Option<(f64, f64)>) -> Self {
        let mut digest = Sha256::new();
        digest.update(b"neoethos.canonical-native.gen0-scope.v1\0");
        digest.update((raw as u64).to_le_bytes());
        digest.update((clamped as u64).to_le_bytes());
        for value in cost_band.into_iter().flat_map(|pair| [pair.0, pair.1]) {
            digest.update(value.to_bits().to_le_bytes());
        }
        Self {
            execution_scope: CanonicalNativeExecutionScopeV1::GenerationZeroOnly,
            raw_legacy_generations_unused_full_search: raw,
            clamped_legacy_generations_unused_full_search: clamped,
            cost_band_status: CanonicalNativeCostBandStatusV1::UnusedGenerationZero,
            cost_band_pips_unused_generation_zero: cost_band,
            identity_sha256: format!("{:x}", digest.finalize()),
        }
    }

    pub const fn execution_scope(&self) -> CanonicalNativeExecutionScopeV1 {
        self.execution_scope
    }
    pub const fn raw_legacy_generations_unused_full_search(&self) -> usize {
        self.raw_legacy_generations_unused_full_search
    }
    pub const fn clamped_legacy_generations_unused_full_search(&self) -> usize {
        self.clamped_legacy_generations_unused_full_search
    }
    pub const fn cost_band_status(&self) -> CanonicalNativeCostBandStatusV1 {
        self.cost_band_status
    }
    pub const fn cost_band_pips_unused_generation_zero(&self) -> Option<(f64, f64)> {
        self.cost_band_pips_unused_generation_zero
    }
    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }
}

pub struct CanonicalNativeGenerationZeroLimitsV1 {
    configured_population_cap: usize,
    resolved_population_cap: usize,
    term_cap: usize,
    string_bytes_cap: usize,
    vector_elements_cap: usize,
    source_count_cap: usize,
    result_bytes_cap: u64,
}

impl CanonicalNativeGenerationZeroLimitsV1 {
    #[cfg(all(feature = "gpu-cuda", target_os = "linux"))]
    fn checked(
        population: usize,
        configured_terms: usize,
        loaded: &LoadedCanonicalResearchContractV1,
        config: &crate::DiscoveryConfig,
    ) -> Result<Self, CanonicalNativeDiscoveryRequestErrorV1> {
        let projection = loaded.source_projection();
        if population > MAX_CANONICAL_NATIVE_GEN0_CONFIGURED_POPULATION_V1 {
            return Err(limit("configured_population_cap"));
        }
        if configured_terms > MAX_CANONICAL_NATIVE_GEN0_TERMS_V1 {
            return Err(limit("term_cap"));
        }
        validate_string_cap_v1(loaded.relative_path())?;
        validate_contract_owned_caps_v1(loaded.contract())?;
        validate_config_owned_request_caps_v1(config)?;
        validate_source_shape_caps_v1(
            projection.bindings().len(),
            projection
                .bindings()
                .iter()
                .map(|binding| binding.segments().len()),
        )?;
        for binding in projection.bindings() {
            for value in [binding.manifest_schema_id(), binding.generation_id()] {
                validate_string_cap_v1(value)?;
            }
        }
        Ok(Self {
            configured_population_cap: MAX_CANONICAL_NATIVE_GEN0_CONFIGURED_POPULATION_V1,
            resolved_population_cap: MAX_CANONICAL_NATIVE_GEN0_RESOLVED_POPULATION_V1,
            term_cap: MAX_CANONICAL_NATIVE_GEN0_TERMS_V1,
            string_bytes_cap: MAX_CANONICAL_NATIVE_GEN0_STRING_BYTES_V1,
            vector_elements_cap: MAX_CANONICAL_NATIVE_GEN0_VECTOR_ELEMENTS_V1,
            source_count_cap: MAX_CANONICAL_NATIVE_GEN0_SOURCE_COUNT_V1,
            result_bytes_cap: MAX_CANONICAL_NATIVE_GEN0_RESULT_BYTES_V1,
        })
    }

    pub const fn configured_population_cap(&self) -> usize {
        self.configured_population_cap
    }
    pub const fn resolved_population_cap(&self) -> usize {
        self.resolved_population_cap
    }
    pub const fn term_cap(&self) -> usize {
        self.term_cap
    }
    pub const fn string_bytes_cap(&self) -> usize {
        self.string_bytes_cap
    }
    pub const fn vector_elements_cap(&self) -> usize {
        self.vector_elements_cap
    }
    pub const fn source_count_cap(&self) -> usize {
        self.source_count_cap
    }
    pub const fn result_bytes_cap(&self) -> u64 {
        self.result_bytes_cap
    }
}

#[derive(Debug, Serialize)]
pub struct CanonicalResearchContractArtifactRefV1 {
    schema: String,
    version: u16,
    relative_path: String,
    expected_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactRefWireV1 {
    schema: String,
    version: u16,
    relative_path: String,
    expected_sha256: String,
}

impl CanonicalResearchContractArtifactRefV1 {
    pub fn checked_new(
        relative_path: impl Into<String>,
        expected_sha256: impl Into<String>,
    ) -> Result<Self, CanonicalNativeDiscoveryRequestErrorV1> {
        let reference = Self {
            schema: CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_SCHEMA_V1.to_owned(),
            version: CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_VERSION_V1,
            relative_path: relative_path.into(),
            expected_sha256: expected_sha256.into(),
        };
        reference.validate()?;
        Ok(reference)
    }

    pub fn relative_path(&self) -> &str {
        &self.relative_path
    }

    pub fn expected_sha256(&self) -> &str {
        &self.expected_sha256
    }

    fn validate(&self) -> Result<(), CanonicalNativeDiscoveryRequestErrorV1> {
        validate_string_cap_v1(&self.relative_path)?;
        if self.schema != CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_SCHEMA_V1
            || self.version != CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_VERSION_V1
        {
            return Err(invalid_reference("unsupported schema or version"));
        }
        validate_relative_path(&self.relative_path)?;
        if self.expected_sha256.len() != 64
            || !self
                .expected_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(invalid_reference(
                "expected SHA-256 is not 64 lowercase hex",
            ));
        }
        Ok(())
    }
}

impl<'de> Deserialize<'de> for CanonicalResearchContractArtifactRefV1 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = ArtifactRefWireV1::deserialize(deserializer)?;
        if wire.schema != CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_SCHEMA_V1
            || wire.version != CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_VERSION_V1
        {
            return Err(serde::de::Error::custom(
                "unsupported artifact-reference schema/version",
            ));
        }
        Self::checked_new(wire.relative_path, wire.expected_sha256)
            .map_err(serde::de::Error::custom)
    }
}

pub struct LoadedCanonicalResearchContractV1 {
    contract: CanonicalTrendbarResearchExecutionContractV3,
    relative_path: String,
    exact_artifact_sha256: String,
    byte_len: u64,
    contract_identity_sha256: String,
    #[cfg(feature = "gpu-cuda")]
    source_projection: neoethos_data::CanonicalPinnedSourceProjectionV1,
}

impl LoadedCanonicalResearchContractV1 {
    pub const fn contract(&self) -> &CanonicalTrendbarResearchExecutionContractV3 {
        &self.contract
    }

    pub fn relative_path(&self) -> &str {
        &self.relative_path
    }

    pub fn exact_artifact_sha256(&self) -> &str {
        &self.exact_artifact_sha256
    }

    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    pub fn contract_identity_sha256(&self) -> &str {
        &self.contract_identity_sha256
    }

    #[cfg(feature = "gpu-cuda")]
    pub const fn source_projection(&self) -> &neoethos_data::CanonicalPinnedSourceProjectionV1 {
        &self.source_projection
    }
}

#[cfg(all(test, target_os = "linux"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CanonicalArtifactLoadBoundaryV1 {
    AfterReferenceValidation,
    AfterDescriptorAcquisition,
}

pub fn load_canonical_research_contract_artifact_v1(
    canonical_root: &SealedCanonicalRootV1,
    reference: CanonicalResearchContractArtifactRefV1,
) -> Result<LoadedCanonicalResearchContractV1, CanonicalNativeDiscoveryRequestErrorV1> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (canonical_root, reference);
        return Err(CanonicalNativeDiscoveryRequestErrorV1::UnsupportedPlatform);
    }
    #[cfg(target_os = "linux")]
    {
        reference.validate()?;
        let bytes = read_canonical_artifact_exact_v1(canonical_root, reference.relative_path())?;
        finish_loaded_contract_v1(reference, bytes)
    }
}

#[cfg(target_os = "linux")]
fn finish_loaded_contract_v1(
    reference: CanonicalResearchContractArtifactRefV1,
    bytes: Vec<u8>,
) -> Result<LoadedCanonicalResearchContractV1, CanonicalNativeDiscoveryRequestErrorV1> {
    let actual_sha256 = format!("{:x}", Sha256::digest(&bytes));
    if actual_sha256 != reference.expected_sha256 {
        return Err(
            CanonicalNativeDiscoveryRequestErrorV1::ArtifactHashMismatch {
                expected: reference.expected_sha256,
                actual: actual_sha256,
            },
        );
    }
    let contract: CanonicalTrendbarResearchExecutionContractV3 = serde_json::from_slice(&bytes)
        .map_err(|error| {
            CanonicalNativeDiscoveryRequestErrorV1::ContractDecode(error.to_string())
        })?;
    validate_contract_owned_caps_v1(&contract)?;
    contract.validate().map_err(contract_validation)?;
    contract
        .validate_against_receipt(contract.input_receipt())
        .map_err(contract_validation)?;
    let contract_identity_sha256 = contract.identity_sha256().map_err(contract_validation)?;
    #[cfg(feature = "gpu-cuda")]
    let source_projection =
        crate::resident_population_auto_sizing_receipt_v2::canonical_pinned_source_projection_from_search_receipt_v1(
            contract.input_receipt(),
        )
        .map_err(|error| CanonicalNativeDiscoveryRequestErrorV1::ContractValidation(error.to_string()))?;
    Ok(LoadedCanonicalResearchContractV1 {
        contract,
        relative_path: reference.relative_path,
        exact_artifact_sha256: actual_sha256,
        byte_len: bytes.len() as u64,
        contract_identity_sha256,
        #[cfg(feature = "gpu-cuda")]
        source_projection,
    })
}

#[cfg(all(test, target_os = "linux"))]
fn load_canonical_research_contract_with_test_hook_v1(
    canonical_root: &SealedCanonicalRootV1,
    reference: CanonicalResearchContractArtifactRefV1,
    mut hook: impl FnMut(CanonicalArtifactLoadBoundaryV1),
) -> Result<LoadedCanonicalResearchContractV1, CanonicalNativeDiscoveryRequestErrorV1> {
    reference.validate()?;
    hook(CanonicalArtifactLoadBoundaryV1::AfterReferenceValidation);
    let bytes =
        crate::canonical_native_root_io_v1::read_canonical_artifact_exact_with_test_hook_v1(
            canonical_root,
            reference.relative_path(),
            (
                || hook(CanonicalArtifactLoadBoundaryV1::AfterDescriptorAcquisition),
                || {},
            ),
        )?;
    finish_loaded_contract_v1(reference, bytes)
}

pub struct CanonicalNativeDiscoveryRequestV1 {
    loaded_contract: LoadedCanonicalResearchContractV1,
    startup_settings_sha256: String,
    runtime_install_receipt: CanonicalNativeRuntimeInstallReceiptV1,
    runtime_authority: CanonicalNativeGenerationZeroRuntimeAuthorityV1,
    canonical_root: SealedCanonicalRootV1,
    exact_series: neoethos_data::CanonicalDatasetSeriesReceiptV1,
    config: crate::DiscoveryConfig,
    scope: CanonicalNativeGenerationZeroScopeV1,
    limits: CanonicalNativeGenerationZeroLimitsV1,
    feature_profile: neoethos_data::FeatureProfile,
}

impl CanonicalNativeDiscoveryRequestV1 {
    pub const fn loaded_contract(&self) -> &LoadedCanonicalResearchContractV1 {
        &self.loaded_contract
    }
    pub fn startup_settings_sha256(&self) -> &str {
        &self.startup_settings_sha256
    }
    pub const fn runtime_install_receipt(&self) -> &CanonicalNativeRuntimeInstallReceiptV1 {
        &self.runtime_install_receipt
    }
    pub const fn runtime_authority(&self) -> &CanonicalNativeGenerationZeroRuntimeAuthorityV1 {
        &self.runtime_authority
    }
    pub const fn canonical_root(&self) -> &SealedCanonicalRootV1 {
        &self.canonical_root
    }
    pub const fn exact_series(&self) -> &neoethos_data::CanonicalDatasetSeriesReceiptV1 {
        &self.exact_series
    }
    pub const fn config(&self) -> &crate::DiscoveryConfig {
        &self.config
    }
    pub const fn scope(&self) -> &CanonicalNativeGenerationZeroScopeV1 {
        &self.scope
    }
    pub const fn limits(&self) -> &CanonicalNativeGenerationZeroLimitsV1 {
        &self.limits
    }
    pub const fn feature_profile(&self) -> neoethos_data::FeatureProfile {
        self.feature_profile
    }

    /// The executor calls this at the last boundary before native Data
    /// preflight. It is deliberately separate from resolution so a federation
    /// toggle in between cannot inherit stale authority.
    pub fn revalidate_before_native_preflight_v1(
        &self,
        startup_settings: &neoethos_core::Settings,
    ) -> Result<(), CanonicalNativeDiscoveryRequestErrorV1> {
        self.runtime_authority
            .validate_current(startup_settings, &self.runtime_install_receipt)
    }
}

#[cfg(feature = "gpu-cuda")]
pub fn resolve_canonical_native_discovery_request_v1(
    startup_settings: &neoethos_core::Settings,
    runtime_install_receipt: &CanonicalNativeRuntimeInstallReceiptV1,
    contract_ref: CanonicalResearchContractArtifactRefV1,
    overrides: CanonicalNativeGenerationZeroOverridesV1,
) -> Result<CanonicalNativeDiscoveryRequestV1, CanonicalNativeDiscoveryRequestErrorV1> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (
            startup_settings,
            runtime_install_receipt,
            contract_ref,
            overrides,
        );
        return Err(CanonicalNativeDiscoveryRequestErrorV1::UnsupportedPlatform);
    }
    #[cfg(target_os = "linux")]
    {
        overrides.validate()?;
        runtime_install_receipt.validate_current(startup_settings)?;
        let runtime_authority =
            seal_generation_zero_runtime_authority_v1(startup_settings, runtime_install_receipt)?;
        validate_settings_owned_request_caps_v1(startup_settings)?;
        let canonical_root = SealedCanonicalRootV1::from_startup_settings(startup_settings)?;
        let loaded_contract =
            load_canonical_research_contract_artifact_v1(&canonical_root, contract_ref)?;
        let mut config = crate::DiscoveryConfig::try_from_settings_for_canonical_trendbar_research(
            startup_settings,
            loaded_contract.contract(),
        )
        .map_err(|error| {
            CanonicalNativeDiscoveryRequestErrorV1::ContractSettingsMismatch(error.to_string())
        })?;

        config.population = overrides.population().unwrap_or(config.population);
        config.population_auto = overrides
            .population_auto()
            .unwrap_or(config.population_auto);
        let configured_max_indicators = overrides
            .max_indicators()
            .unwrap_or(startup_settings.models.prop_search_max_indicators);
        // Preserve zero as a bounded native sentinel. Chunk 3 resolves it from
        // exact prepared feature count F before sizing; usize::MAX never enters
        // native sizing or result evidence.
        config.max_indicators = configured_max_indicators;
        let projection = loaded_contract.source_projection();
        config.timeframe_label = projection.base_timeframe().as_str().to_owned();
        config.higher_timeframes = projection
            .bindings()
            .iter()
            .filter(|binding| binding.dataset_identity() != projection.anchor_dataset_identity())
            .map(|binding| binding.dataset_identity().timeframe().as_str().to_owned())
            .collect();
        validate_generation_zero_policy(startup_settings, &config, loaded_contract.contract())?;

        let limits = CanonicalNativeGenerationZeroLimitsV1::checked(
            config.population,
            configured_max_indicators,
            &loaded_contract,
            &config,
        )?;
        let exact_series = exact_series_from_projection(loaded_contract.source_projection())?;
        match neoethos_data::pin_exact_canonical_series_v1(
            startup_settings.system.data_dir.as_path(),
            exact_series.clone(),
        ) {
            Ok(pin) => drop(pin),
            Err(error) => {
                if let Some(conflict) =
                    error.downcast_ref::<neoethos_data::ExactDatasetGenerationConflict>()
                {
                    return Err(
                        CanonicalNativeDiscoveryRequestErrorV1::ExactDatasetGenerationConflict(
                            conflict.clone(),
                        ),
                    );
                }
                return Err(CanonicalNativeDiscoveryRequestErrorV1::DatasetSeries(
                    error.to_string(),
                ));
            }
        }
        let scope = CanonicalNativeGenerationZeroScopeV1::seal(
            startup_settings.models.prop_search_generations,
            config.generations,
            config.cost_band_pips,
        );
        Ok(CanonicalNativeDiscoveryRequestV1 {
            loaded_contract,
            startup_settings_sha256: runtime_install_receipt.startup_settings_sha256().to_owned(),
            runtime_install_receipt: runtime_install_receipt.clone(),
            runtime_authority,
            canonical_root,
            exact_series,
            config,
            scope,
            limits,
            feature_profile: neoethos_data::FeatureProfile::Standard,
        })
    }
}

#[cfg(all(feature = "gpu-cuda", target_os = "linux"))]
fn exact_series_from_projection(
    projection: &neoethos_data::CanonicalPinnedSourceProjectionV1,
) -> Result<neoethos_data::CanonicalDatasetSeriesReceiptV1, CanonicalNativeDiscoveryRequestErrorV1>
{
    let mut seen = HashSet::with_capacity(projection.bindings().len());
    let mut anchor = None;
    let mut selected = Vec::with_capacity(projection.bindings().len());
    for binding in projection.bindings() {
        if !seen.insert(binding.dataset_identity().timeframe()) {
            return Err(CanonicalNativeDiscoveryRequestErrorV1::DatasetSeries(
                "projection repeats a timeframe binding".to_owned(),
            ));
        }
        let generation = neoethos_data::SelectedDatasetGenerationV1::new(
            binding.dataset_identity().clone(),
            binding.generation_id(),
            binding
                .manifest_sha256()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
        )
        .map_err(|error| {
            CanonicalNativeDiscoveryRequestErrorV1::DatasetSeries(error.to_string())
        })?;
        if binding.dataset_identity() == projection.anchor_dataset_identity() {
            if anchor.replace(generation.clone()).is_some() {
                return Err(CanonicalNativeDiscoveryRequestErrorV1::DatasetSeries(
                    "projection contains more than one anchor binding".to_owned(),
                ));
            }
        }
        selected.push(generation);
    }
    let anchor = anchor.ok_or_else(|| {
        CanonicalNativeDiscoveryRequestErrorV1::DatasetSeries(
            "projection contains no anchor binding".to_owned(),
        )
    })?;
    neoethos_data::CanonicalDatasetSeriesReceiptV1::new(anchor, selected)
        .map_err(|error| CanonicalNativeDiscoveryRequestErrorV1::DatasetSeries(error.to_string()))
}

#[cfg(all(feature = "gpu-cuda", target_os = "linux"))]
fn validate_generation_zero_policy(
    settings: &neoethos_core::Settings,
    config: &crate::DiscoveryConfig,
    contract: &CanonicalTrendbarResearchExecutionContractV3,
) -> Result<(), CanonicalNativeDiscoveryRequestErrorV1> {
    let unsupported = if settings.risk.backtest_spread_pips_asian.is_some()
        || settings.risk.backtest_spread_pips_overlap.is_some()
        || settings.risk.backtest_spread_pips_late_ny.is_some()
        || config.session_spread_pips.is_some()
    {
        Some("session_spread_curve")
    } else if config.adaptive_thresholds {
        Some("adaptive_thresholds")
    } else if settings.models.gene_stop_bounds.atr_scaled {
        Some("atr_scaled_gene_bounds")
    } else if config.runtime_overrides.min_history_years != 0 {
        Some("minimum_history")
    } else if config.discovery_ledger_enabled {
        Some("discovery_ledger")
    } else if config.max_rows != 0
        || config
            .max_rows_by_timeframe
            .values()
            .any(|value| *value != 0)
    {
        Some("row_cap")
    } else if config.runtime_overrides.prefilter_top_k != 0 {
        Some("feature_prefilter")
    } else {
        None
    };
    if let Some(policy) = unsupported {
        return Err(
            CanonicalNativeDiscoveryRequestErrorV1::UnsupportedGenerationZeroPolicy { policy },
        );
    }
    let payoff_inputs = crate::payoff_inputs_for_config(config, contract.pip_value_per_lot());
    crate::assert_payoff_floor_reachable(config.target_profile.min_payoff_ratio, &payoff_inputs)
        .map_err(
            |_| CanonicalNativeDiscoveryRequestErrorV1::UnsupportedGenerationZeroPolicy {
                policy: "unreachable_payoff_floor",
            },
        )?;
    Ok(())
}

fn validate_relative_path(path: &str) -> Result<(), CanonicalNativeDiscoveryRequestErrorV1> {
    let bytes = path.as_bytes();
    if path.is_empty()
        || Path::new(path).is_absolute()
        || path.starts_with('/')
        || path.ends_with('/')
        || path.contains("//")
        || path.contains('\\')
        || path.chars().any(char::is_control)
        || (bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':')
        || path
            .split('/')
            .any(|component| component.is_empty() || component == "." || component == "..")
    {
        return Err(invalid_reference(
            "relative path is not canonical normal components",
        ));
    }
    Ok(())
}

fn validate_string_cap_v1(value: &str) -> Result<(), CanonicalNativeDiscoveryRequestErrorV1> {
    if value.len() > MAX_CANONICAL_NATIVE_GEN0_STRING_BYTES_V1 {
        Err(limit("string_bytes_cap"))
    } else {
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn validate_source_shape_caps_v1(
    source_count: usize,
    segment_counts: impl IntoIterator<Item = usize>,
) -> Result<(), CanonicalNativeDiscoveryRequestErrorV1> {
    if source_count > MAX_CANONICAL_NATIVE_GEN0_SOURCE_COUNT_V1 {
        return Err(limit("source_count_cap"));
    }
    let mut total_segments = 0_usize;
    for segment_count in segment_counts {
        total_segments = total_segments
            .checked_add(segment_count)
            .ok_or_else(|| limit("vector_elements_cap"))?;
        if total_segments > MAX_CANONICAL_NATIVE_GEN0_VECTOR_ELEMENTS_V1 {
            return Err(limit("vector_elements_cap"));
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn validate_contract_owned_caps_v1(
    contract: &CanonicalTrendbarResearchExecutionContractV3,
) -> Result<(), CanonicalNativeDiscoveryRequestErrorV1> {
    let receipt = contract.input_receipt();
    validate_source_shape_caps_v1(
        receipt.source_bindings().len(),
        receipt
            .source_bindings()
            .iter()
            .map(|binding| binding.segments().len()),
    )?;
    for value in [
        contract.input_receipt_sha256(),
        contract.symbol(),
        contract.account_currency(),
        contract.assumption_source_id(),
        contract.assumption_source_sha256(),
        receipt.anchor_dataset_identity(),
        receipt.feature_plan_identity(),
        receipt.feature_provenance_identity(),
        receipt.feature_content_sha256(),
        receipt.feature_execution().vector_ta_math_authority(),
    ] {
        validate_string_cap_v1(value)?;
    }
    for binding in receipt.source_bindings() {
        for value in [
            binding.source_node_id(),
            binding.dataset_identity(),
            binding.manifest_schema_id(),
            binding.manifest_sha256(),
            binding.generation_id(),
            binding.vortex_sha256(),
            binding.bar_timestamp_convention(),
        ] {
            validate_string_cap_v1(value)?;
        }
    }
    Ok(())
}

#[cfg(all(feature = "gpu-cuda", target_os = "linux"))]
fn validate_settings_owned_request_caps_v1(
    settings: &neoethos_core::Settings,
) -> Result<(), CanonicalNativeDiscoveryRequestErrorV1> {
    for value in [
        settings.system.base_timeframe.as_str(),
        settings.system.symbol.as_str(),
        settings.system.account_currency.as_str(),
        settings.models.discovery_ledger.cache_dir.as_str(),
    ] {
        validate_string_cap_v1(value)?;
    }
    if settings.system.higher_timeframes.len() > MAX_CANONICAL_NATIVE_GEN0_SOURCE_COUNT_V1
        || settings.models.prop_search_max_rows_by_tf.len()
            > MAX_CANONICAL_NATIVE_GEN0_SOURCE_COUNT_V1
    {
        return Err(limit("source_count_cap"));
    }
    for value in settings
        .system
        .higher_timeframes
        .iter()
        .map(String::as_str)
        .chain(
            settings
                .models
                .prop_search_max_rows_by_tf
                .keys()
                .map(String::as_str),
        )
    {
        validate_string_cap_v1(value)?;
    }
    Ok(())
}

#[cfg(all(feature = "gpu-cuda", target_os = "linux"))]
fn validate_config_owned_request_caps_v1(
    config: &crate::DiscoveryConfig,
) -> Result<(), CanonicalNativeDiscoveryRequestErrorV1> {
    for value in [
        config.timeframe_label.as_str(),
        config.evaluation_symbol.as_str(),
        config.evaluation_account_currency.as_str(),
        config.discovery_ledger_cache_dir.as_str(),
    ] {
        validate_string_cap_v1(value)?;
    }
    if config.higher_timeframes.len() > MAX_CANONICAL_NATIVE_GEN0_SOURCE_COUNT_V1
        || config.max_rows_by_timeframe.len() > MAX_CANONICAL_NATIVE_GEN0_SOURCE_COUNT_V1
    {
        return Err(limit("source_count_cap"));
    }
    for value in config
        .higher_timeframes
        .iter()
        .map(String::as_str)
        .chain(config.max_rows_by_timeframe.keys().map(String::as_str))
    {
        validate_string_cap_v1(value)?;
    }
    Ok(())
}

fn invalid_reference(reason: &str) -> CanonicalNativeDiscoveryRequestErrorV1 {
    CanonicalNativeDiscoveryRequestErrorV1::InvalidArtifactReference(reason.to_owned())
}
fn invalid_overrides(reason: &str) -> CanonicalNativeDiscoveryRequestErrorV1 {
    CanonicalNativeDiscoveryRequestErrorV1::InvalidGenerationZeroOverrides(reason.to_owned())
}
fn limit(limit: &'static str) -> CanonicalNativeDiscoveryRequestErrorV1 {
    CanonicalNativeDiscoveryRequestErrorV1::RequestLimitExceeded { limit }
}
#[cfg(target_os = "linux")]
fn contract_validation(error: anyhow::Error) -> CanonicalNativeDiscoveryRequestErrorV1 {
    CanonicalNativeDiscoveryRequestErrorV1::ContractValidation(error.to_string())
}
#[cfg(all(test, target_os = "linux"))]
mod deterministic_race_tests {
    use super::*;
    use CanonicalArtifactLoadBoundaryV1::{
        AfterDescriptorAcquisition as PostOpen, AfterReferenceValidation as PreOpen,
    };
    use CanonicalNativeDiscoveryRequestErrorV1::{
        ArtifactTooLarge, EscapeOrMount, RaceDetected, UnsafeLink,
    };
    use neoethos_core::Settings;
    use std::ffi::CString;
    use std::fs;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::symlink;
    use std::path::{Path, PathBuf};
    use tempfile::TempDir;
    const LIMIT: usize = 8 << 20;
    fn sealed(root: &TempDir) -> SealedCanonicalRootV1 {
        let mut settings = Settings::default();
        settings.system.data_dir = root.path().to_owned();
        SealedCanonicalRootV1::from_startup_settings(&settings).unwrap()
    }
    fn exchange(left: &Path, right: &Path) {
        let left = CString::new(left.as_os_str().as_bytes()).unwrap();
        let right = CString::new(right.as_os_str().as_bytes()).unwrap();
        let cwd = libc::AT_FDCWD;
        let flags = libc::RENAME_EXCHANGE;
        let result = unsafe { libc::renameat2(cwd, left.as_ptr(), cwd, right.as_ptr(), flags) };
        assert_eq!(result, 0);
    }
    fn race_case(component: bool) -> (TempDir, TempDir, &'static str, PathBuf, PathBuf) {
        let root = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let (relative, live_name, spare_name) = if component {
            ("slot/artifact.json", "slot", "spare")
        } else {
            ("artifact.json", "artifact.json", "spare.json")
        };
        let live = root.path().join(live_name);
        let spare = root.path().join(spare_name);
        let outside_file = outside.path().join("artifact.json");
        fs::write(&outside_file, b"outside artifact").unwrap();
        if component {
            fs::create_dir(&live).unwrap();
            fs::write(live.join("artifact.json"), b"inside artifact").unwrap();
            symlink(outside.path(), &spare).unwrap();
        } else {
            fs::write(&live, b"inside artifact").unwrap();
            symlink(outside_file, &spare).unwrap();
        }
        (root, outside, relative, live, spare)
    }
    fn race_error(
        component: bool,
        selected: CanonicalArtifactLoadBoundaryV1,
        inside_root: bool,
    ) -> CanonicalNativeDiscoveryRequestErrorV1 {
        let (root, outside, relative, live, spare) = race_case(component);
        let outside_peer = if component {
            outside.path().to_owned()
        } else {
            outside.path().join("artifact.json")
        };
        let peer = if selected == PreOpen {
            &spare
        } else {
            &outside_peer
        };
        load_canonical_research_contract_with_test_hook_v1(
            &sealed(&root),
            CanonicalResearchContractArtifactRefV1::checked_new(
                relative,
                format!("{:x}", Sha256::digest(b"inside artifact")),
            )
            .unwrap(),
            |boundary| {
                if boundary == selected {
                    if inside_root {
                        fs::rename(&live, &spare).unwrap();
                    } else {
                        exchange(&live, peer);
                    }
                }
            },
        )
        .err()
        .unwrap()
    }
    #[test]
    fn exchange_and_inside_rename_are_rejected_at_reviewed_boundaries() {
        for component in [true, false] {
            for selected in [PreOpen, PostOpen] {
                let error = race_error(component, selected, false);
                let pre_open = selected == PreOpen;
                assert!(error == EscapeOrMount || pre_open && error == UnsafeLink);
            }
        }
        assert_eq!(race_error(false, PostOpen, true), RaceDetected);
    }
    #[test]
    fn post_initial_stat_growth_is_rejected_by_the_bounded_read() {
        let root = TempDir::new().unwrap();
        let path = root.path().join("exact-max.json");
        fs::write(&path, vec![b' '; LIMIT]).unwrap();
        let result =
            crate::canonical_native_root_io_v1::read_canonical_artifact_exact_with_test_hook_v1(
                &sealed(&root),
                "exact-max.json",
                (
                    || {},
                    || {
                        let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
                        file.set_len((2 * LIMIT) as u64).unwrap();
                    },
                ),
            );
        assert!(matches!(
            result,
            Err(ArtifactTooLarge { maximum, observed }) if observed == maximum + 1
        ));
    }

    #[test]
    fn oversized_relative_path_hits_the_named_cap_before_the_loader_callback() {
        let root = TempDir::new().unwrap();
        let reference = CanonicalResearchContractArtifactRefV1 {
            schema: CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_SCHEMA_V1.to_owned(),
            version: CANONICAL_RESEARCH_CONTRACT_ARTIFACT_REF_VERSION_V1,
            relative_path: "a".repeat(MAX_CANONICAL_NATIVE_GEN0_STRING_BYTES_V1 + 1),
            expected_sha256: "0".repeat(64),
        };
        let mut callback_called = false;
        let result =
            load_canonical_research_contract_with_test_hook_v1(&sealed(&root), reference, |_| {
                callback_called = true
            });
        assert!(matches!(
            result,
            Err(
                CanonicalNativeDiscoveryRequestErrorV1::RequestLimitExceeded {
                    limit: "string_bytes_cap"
                }
            )
        ));
        assert!(!callback_called);
    }

    #[test]
    fn borrowed_source_shape_census_rejects_oversized_counts_with_named_caps() {
        assert!(matches!(
            validate_source_shape_caps_v1(
                MAX_CANONICAL_NATIVE_GEN0_SOURCE_COUNT_V1 + 1,
                std::iter::empty(),
            ),
            Err(
                CanonicalNativeDiscoveryRequestErrorV1::RequestLimitExceeded {
                    limit: "source_count_cap"
                }
            )
        ));
        assert!(matches!(
            validate_source_shape_caps_v1(
                1,
                std::iter::once(MAX_CANONICAL_NATIVE_GEN0_VECTOR_ELEMENTS_V1 + 1),
            ),
            Err(
                CanonicalNativeDiscoveryRequestErrorV1::RequestLimitExceeded {
                    limit: "vector_elements_cap"
                }
            )
        ));
        assert!(matches!(
            validate_source_shape_caps_v1(
                1,
                [MAX_CANONICAL_NATIVE_GEN0_VECTOR_ELEMENTS_V1, usize::MAX],
            ),
            Err(
                CanonicalNativeDiscoveryRequestErrorV1::RequestLimitExceeded {
                    limit: "vector_elements_cap"
                }
            )
        ));
    }
}
