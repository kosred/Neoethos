//! Exact, move-only handoff from a locked research finalist into candidate training.
//!
//! This boundary deliberately stops before combined OOS, promotion, or live deployment.

mod config;
mod install;

use config::resolve_promotion_candidate_training_config_with_plan_v1;
pub use config::{
    PromotionCandidateTrainingConfigIdentityV1,
    resolve_promotion_candidate_training_config_identity_v1,
};

use crate::{ModelTrainingProgress, TrainingOrchestrator};
use neoethos_core::Settings;
use neoethos_data::{CanonicalDatasetSeriesReceiptV1, CanonicalTimeframe};
use neoethos_execution_budget::CpuLease;
use neoethos_search::{
    CanonicalSearchInputReceiptV2, CanonicalTrendbarResearchExecutionContractV3,
    canonical_locked_portfolio_identity_sha256_v1,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::path::Path;

pub(crate) use install::install_promotion_candidate_model_tree_v1;

const HANDOFF_SCHEMA_V1: &str = "neoethos.promotion-candidate-training-handoff.v1";
const LOCKED_PORTFOLIO_SCHEMA_V1: &str = "neoethos.locked-promotion-portfolio-envelope.v1";
const BROKER_IDENTITY_SCHEMA_V1: &str = "neoethos.promotion-candidate-broker-authority.v1";
const MANIFEST_SCHEMA_V1: &str = "neoethos.promotion-candidate-training-manifest.v1";
const SCHEMA_VERSION_V1: u16 = 1;
const HANDOFF_IDENTITY_DOMAIN_V1: &[u8] = b"neoethos.promotion-candidate-training-handoff.v1\0";
const LOCKED_PORTFOLIO_IDENTITY_DOMAIN_V1: &[u8] = b"neoethos.locked-final-portfolio.v1\0";

pub const MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1: usize = 8 * 1024 * 1024;
pub const MAX_PROMOTION_CANDIDATE_MODEL_TREE_BYTES_V1: u64 = 64 * 1024 * 1024 * 1024;
pub const MAX_PROMOTION_CANDIDATE_MODEL_FILE_COUNT_V1: usize = 100_000;
pub const PROMOTION_CANDIDATE_TRAINING_EVIDENCE_FILE_V1: &str =
    "promotion-candidate-training-evidence-v1.json";
const MAX_PLANNED_MODELS_V1: usize = 256;
const MAX_MODEL_NAME_BYTES_V1: usize = 128;
const MAX_REFUSAL_DETAIL_BYTES_V1: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromotionCandidateTrainingRefusalCodeV1 {
    InvalidHandoff,
    HandoffTooLarge,
    InputReceiptMismatch,
    OosCutoffLeakage,
    RuntimeConfigMismatch,
    ModelConfigMismatch,
    ModelInventoryIncomplete,
    ModelTrainingFailed,
    TrainingFailed,
    CandidateRootInvalid,
    StagingFailed,
    ArtifactTreeInvalid,
    ArtifactTreeLimitExceeded,
    AtomicInstallUnsupported,
    AtomicInstallFailed,
    CandidateIdentityCollision,
    InstalledTreeChanged,
    EvidenceEncodingFailed,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionCandidateTrainingRefusalV1 {
    code: PromotionCandidateTrainingRefusalCodeV1,
    detail: String,
}

impl PromotionCandidateTrainingRefusalV1 {
    pub const fn code(&self) -> PromotionCandidateTrainingRefusalCodeV1 {
        self.code
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for PromotionCandidateTrainingRefusalV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}: {}", self.code, self.detail)
    }
}

impl Error for PromotionCandidateTrainingRefusalV1 {}

pub(crate) fn refusal_v1(
    code: PromotionCandidateTrainingRefusalCodeV1,
    detail: impl Into<String>,
) -> PromotionCandidateTrainingRefusalV1 {
    let mut detail = detail.into();
    if detail.len() > MAX_REFUSAL_DETAIL_BYTES_V1 {
        let mut boundary = MAX_REFUSAL_DETAIL_BYTES_V1;
        while !detail.is_char_boundary(boundary) {
            boundary -= 1;
        }
        detail.truncate(boundary);
    }
    PromotionCandidateTrainingRefusalV1 { code, detail }
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionCandidateBrokerAuthorityIdentityV1 {
    schema: String,
    version: u16,
    identity_sha256: String,
}

impl PromotionCandidateBrokerAuthorityIdentityV1 {
    pub fn checked_new(
        identity_sha256: String,
    ) -> Result<Self, PromotionCandidateTrainingRefusalV1> {
        let value = Self {
            schema: BROKER_IDENTITY_SCHEMA_V1.to_owned(),
            version: SCHEMA_VERSION_V1,
            identity_sha256,
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<(), PromotionCandidateTrainingRefusalV1> {
        if self.schema != BROKER_IDENTITY_SCHEMA_V1 || self.version != SCHEMA_VERSION_V1 {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff,
                "unsupported broker-authority identity schema/version",
            ));
        }
        validate_sha256_v1(&self.identity_sha256, "broker-authority identity")
    }

    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionCandidateLockedPortfolioV1 {
    schema: String,
    version: u16,
    canonical_json: String,
    identity_sha256: String,
}

impl PromotionCandidateLockedPortfolioV1 {
    pub fn from_serializable<T: Serialize>(
        portfolio: &T,
    ) -> Result<Self, PromotionCandidateTrainingRefusalV1> {
        let bytes = serde_json::to_vec(portfolio).map_err(|error| {
            refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff,
                format!("encode locked finalist portfolio: {error}"),
            )
        })?;
        if bytes.len() > MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1 {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::HandoffTooLarge,
                format!("locked finalist portfolio is {} bytes", bytes.len()),
            ));
        }
        let identity_sha256 =
            canonical_locked_portfolio_identity_sha256_v1(portfolio).map_err(|error| {
                refusal_v1(
                    PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff,
                    format!("hash locked finalist portfolio: {error}"),
                )
            })?;
        let value = Self {
            schema: LOCKED_PORTFOLIO_SCHEMA_V1.to_owned(),
            version: SCHEMA_VERSION_V1,
            canonical_json: String::from_utf8(bytes).map_err(|error| {
                refusal_v1(
                    PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff,
                    format!("locked finalist portfolio JSON is not UTF-8: {error}"),
                )
            })?,
            identity_sha256,
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<(), PromotionCandidateTrainingRefusalV1> {
        if self.schema != LOCKED_PORTFOLIO_SCHEMA_V1 || self.version != SCHEMA_VERSION_V1 {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff,
                "unsupported locked-portfolio envelope schema/version",
            ));
        }
        if self.canonical_json.len() > MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1 {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::HandoffTooLarge,
                "locked finalist portfolio exceeds the handoff byte cap",
            ));
        }
        serde_json::from_str::<serde_json::Value>(&self.canonical_json).map_err(|error| {
            refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff,
                format!("locked finalist portfolio is not JSON: {error}"),
            )
        })?;
        let actual = domain_sha256_v1(
            LOCKED_PORTFOLIO_IDENTITY_DOMAIN_V1,
            self.canonical_json.as_bytes(),
        );
        if actual != self.identity_sha256 {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff,
                "locked finalist portfolio bytes disagree with their identity",
            ));
        }
        Ok(())
    }

    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    pub fn canonical_json_bytes(&self) -> &[u8] {
        self.canonical_json.as_bytes()
    }

    pub fn deserialize_exact<T>(&self) -> Result<T, PromotionCandidateTrainingRefusalV1>
    where
        T: DeserializeOwned + Serialize,
    {
        self.validate()?;
        let value: T = serde_json::from_slice(self.canonical_json_bytes()).map_err(|error| {
            refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff,
                format!("decode typed locked finalist portfolio: {error}"),
            )
        })?;
        let identity = canonical_locked_portfolio_identity_sha256_v1(&value).map_err(|error| {
            refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff,
                format!("rehash typed locked finalist portfolio: {error}"),
            )
        })?;
        if identity != self.identity_sha256 {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff,
                "typed locked finalist portfolio does not reproduce the sealed identity",
            ));
        }
        Ok(value)
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionCandidateTrainingHandoffV1 {
    schema: String,
    version: u16,
    canonical_series: CanonicalDatasetSeriesReceiptV1,
    #[serde(
        serialize_with = "serialize_timeframe_v1",
        deserialize_with = "deserialize_timeframe_v1"
    )]
    base_timeframe: CanonicalTimeframe,
    search_input_receipt: CanonicalSearchInputReceiptV2,
    screening_contract: CanonicalTrendbarResearchExecutionContractV3,
    locked_portfolio: PromotionCandidateLockedPortfolioV1,
    oos_cutoff_ms: i64,
    purge_bars: usize,
    broker_authority: PromotionCandidateBrokerAuthorityIdentityV1,
    training_config: PromotionCandidateTrainingConfigIdentityV1,
}

impl PromotionCandidateTrainingHandoffV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn checked_new(
        canonical_series: CanonicalDatasetSeriesReceiptV1,
        base_timeframe: CanonicalTimeframe,
        search_input_receipt: CanonicalSearchInputReceiptV2,
        screening_contract: CanonicalTrendbarResearchExecutionContractV3,
        locked_portfolio: PromotionCandidateLockedPortfolioV1,
        oos_cutoff_ms: i64,
        purge_bars: usize,
        broker_authority: PromotionCandidateBrokerAuthorityIdentityV1,
        training_config: PromotionCandidateTrainingConfigIdentityV1,
    ) -> Result<Self, PromotionCandidateTrainingRefusalV1> {
        let value = Self {
            schema: HANDOFF_SCHEMA_V1.to_owned(),
            version: SCHEMA_VERSION_V1,
            canonical_series,
            base_timeframe,
            search_input_receipt,
            screening_contract,
            locked_portfolio,
            oos_cutoff_ms,
            purge_bars,
            broker_authority,
            training_config,
        };
        value.validate()?;
        Ok(value)
    }

    pub(crate) fn validate(&self) -> Result<(), PromotionCandidateTrainingRefusalV1> {
        if self.schema != HANDOFF_SCHEMA_V1 || self.version != SCHEMA_VERSION_V1 {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff,
                "unsupported promotion-candidate handoff schema/version",
            ));
        }
        self.canonical_series.validate().map_err(|error| {
            refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InputReceiptMismatch,
                format!("validate selected canonical series: {error}"),
            )
        })?;
        let anchor = self.search_input_receipt.validate().map_err(|error| {
            refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InputReceiptMismatch,
                format!("validate canonical search receipt: {error}"),
            )
        })?;
        self.screening_contract
            .validate_against_receipt(&self.search_input_receipt)
            .map_err(|error| {
                refusal_v1(
                    PromotionCandidateTrainingRefusalCodeV1::InputReceiptMismatch,
                    format!("validate screening/search binding: {error}"),
                )
            })?;
        if self.canonical_series.anchor().identity() != &anchor
            || anchor.timeframe() != self.base_timeframe
        {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InputReceiptMismatch,
                "selected series/base timeframe differs from the search anchor",
            ));
        }
        validate_series_against_search_v1(&self.canonical_series, &self.search_input_receipt)?;
        if self.oos_cutoff_ms <= 0 {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff,
                "OOS cutoff must be a positive epoch millisecond",
            ));
        }
        for binding in self.search_input_receipt.source_bindings() {
            for segment in binding.segments() {
                if segment.timestamp_end_ms() >= self.oos_cutoff_ms {
                    return Err(refusal_v1(
                        PromotionCandidateTrainingRefusalCodeV1::OosCutoffLeakage,
                        format!(
                            "search source {} reaches {} at/after OOS cutoff {}",
                            binding.source_node_id(),
                            segment.timestamp_end_ms(),
                            self.oos_cutoff_ms
                        ),
                    ));
                }
            }
        }
        if self.purge_bars > 1_000_000 {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff,
                "purge bars must not exceed 1,000,000",
            ));
        }
        self.locked_portfolio.validate()?;
        self.broker_authority.validate()?;
        self.training_config.validate()?;
        let bytes = serde_json::to_vec(self).map_err(|error| {
            refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff,
                format!("encode promotion-candidate handoff: {error}"),
            )
        })?;
        if bytes.len() > MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1 {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::HandoffTooLarge,
                format!("promotion-candidate handoff is {} bytes", bytes.len()),
            ));
        }
        Ok(())
    }

    pub fn identity_sha256(&self) -> Result<String, PromotionCandidateTrainingRefusalV1> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(|error| {
            refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff,
                format!("encode promotion-candidate handoff identity: {error}"),
            )
        })?;
        Ok(domain_sha256_v1(HANDOFF_IDENTITY_DOMAIN_V1, &bytes))
    }

    pub fn validate_against_config_identity_v1(
        &self,
        actual: &PromotionCandidateTrainingConfigIdentityV1,
    ) -> Result<(), PromotionCandidateTrainingRefusalV1> {
        self.training_config.validate()?;
        actual.validate()?;
        if self.training_config.runtime_config_sha256 != actual.runtime_config_sha256 {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::RuntimeConfigMismatch,
                "runtime execution plan changed after handoff sealing",
            ));
        }
        if self.training_config.model_config_sha256 != actual.model_config_sha256
            || self.training_config.planned_models != actual.planned_models
        {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::ModelConfigMismatch,
                "effective model configuration changed after handoff sealing",
            ));
        }
        Ok(())
    }

    pub fn validate_against_settings_v1(
        &self,
        settings: &Settings,
    ) -> Result<(), PromotionCandidateTrainingRefusalV1> {
        self.validate()?;
        if self.purge_bars != settings.models.label_horizon_bars {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::ModelConfigMismatch,
                format!(
                    "handoff purge {} differs from configured label horizon {}",
                    self.purge_bars, settings.models.label_horizon_bars
                ),
            ));
        }
        let sealed_plan = self
            .training_config
            .sealed_hardware_plan_v1()
            .ok_or_else(|| {
                refusal_v1(
                    PromotionCandidateTrainingRefusalCodeV1::RuntimeConfigMismatch,
                    "promotion training handoff lacks its exact sealed hardware plan",
                )
            })?;
        let actual =
            resolve_promotion_candidate_training_config_with_plan_v1(settings, sealed_plan)?;
        self.validate_against_config_identity_v1(&actual)
    }

    pub const fn canonical_series(&self) -> &CanonicalDatasetSeriesReceiptV1 {
        &self.canonical_series
    }

    pub const fn base_timeframe(&self) -> CanonicalTimeframe {
        self.base_timeframe
    }

    pub const fn search_input_receipt(&self) -> &CanonicalSearchInputReceiptV2 {
        &self.search_input_receipt
    }

    pub const fn screening_contract(&self) -> &CanonicalTrendbarResearchExecutionContractV3 {
        &self.screening_contract
    }

    pub const fn locked_portfolio(&self) -> &PromotionCandidateLockedPortfolioV1 {
        &self.locked_portfolio
    }

    pub const fn oos_cutoff_ms(&self) -> i64 {
        self.oos_cutoff_ms
    }

    pub const fn purge_bars(&self) -> usize {
        self.purge_bars
    }

    pub const fn broker_authority(&self) -> &PromotionCandidateBrokerAuthorityIdentityV1 {
        &self.broker_authority
    }

    pub const fn training_config(&self) -> &PromotionCandidateTrainingConfigIdentityV1 {
        &self.training_config
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionCandidateModelArtifactV1 {
    model_name: String,
    relative_dir: String,
    tree_sha256: String,
    file_count: u64,
    total_bytes: u64,
}

impl PromotionCandidateModelArtifactV1 {
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn relative_dir(&self) -> &str {
        &self.relative_dir
    }

    pub fn tree_sha256(&self) -> &str {
        &self.tree_sha256
    }

    pub const fn file_count(&self) -> u64 {
        self.file_count
    }

    pub const fn total_bytes(&self) -> u64 {
        self.total_bytes
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionCandidateTrainingManifestV1 {
    schema: String,
    version: u16,
    handoff_identity_sha256: String,
    locked_portfolio_identity_sha256: String,
    runtime_config_sha256: String,
    model_config_sha256: String,
    candidate_relative_dir: String,
    candidate_tree_sha256: String,
    evidence_sha256: String,
    model_artifacts: Vec<PromotionCandidateModelArtifactV1>,
    total_file_count: u64,
    total_bytes: u64,
}

impl PromotionCandidateTrainingManifestV1 {
    pub(crate) fn new_v1(
        handoff_identity_sha256: String,
        locked_portfolio_identity_sha256: String,
        runtime_config_sha256: String,
        model_config_sha256: String,
        candidate_tree_sha256: String,
        evidence_sha256: String,
        model_artifacts: Vec<PromotionCandidateModelArtifactV1>,
        total_file_count: u64,
        total_bytes: u64,
    ) -> Self {
        Self {
            schema: MANIFEST_SCHEMA_V1.to_owned(),
            version: SCHEMA_VERSION_V1,
            candidate_relative_dir: candidate_tree_sha256.clone(),
            candidate_tree_sha256,
            handoff_identity_sha256,
            locked_portfolio_identity_sha256,
            runtime_config_sha256,
            model_config_sha256,
            evidence_sha256,
            model_artifacts,
            total_file_count,
            total_bytes,
        }
    }

    pub(crate) fn validate_shape_v1(&self) -> Result<(), PromotionCandidateTrainingRefusalV1> {
        if self.schema != MANIFEST_SCHEMA_V1
            || self.version != SCHEMA_VERSION_V1
            || self.candidate_relative_dir != self.candidate_tree_sha256
        {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InstalledTreeChanged,
                "unsupported or inconsistent candidate manifest",
            ));
        }
        for (value, label) in [
            (&self.handoff_identity_sha256, "handoff"),
            (&self.locked_portfolio_identity_sha256, "locked portfolio"),
            (&self.runtime_config_sha256, "runtime config"),
            (&self.model_config_sha256, "model config"),
            (&self.candidate_tree_sha256, "candidate tree"),
            (&self.evidence_sha256, "evidence"),
        ] {
            validate_sha256_v1(value, label)?;
        }
        if self.model_artifacts.is_empty()
            || self.model_artifacts.len() > MAX_PLANNED_MODELS_V1
            || self.total_file_count == 0
            || self.total_bytes > MAX_PROMOTION_CANDIDATE_MODEL_TREE_BYTES_V1
        {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InstalledTreeChanged,
                "candidate manifest counts are outside their bounds",
            ));
        }
        Ok(())
    }

    pub fn candidate_relative_dir(&self) -> &str {
        &self.candidate_relative_dir
    }

    pub fn candidate_tree_sha256(&self) -> &str {
        &self.candidate_tree_sha256
    }

    pub fn model_artifacts(&self) -> &[PromotionCandidateModelArtifactV1] {
        &self.model_artifacts
    }

    pub fn verify_installed(
        &self,
        candidate_root: &Path,
    ) -> Result<(), PromotionCandidateTrainingRefusalV1> {
        install::verify_installed_manifest_v1(candidate_root, self)
    }

    pub fn reopen_handoff(
        &self,
        candidate_root: &Path,
    ) -> Result<PromotionCandidateTrainingHandoffV1, PromotionCandidateTrainingRefusalV1> {
        install::reopen_installed_handoff_v1(candidate_root, self)
    }
}

#[derive(Debug)]
pub enum PromotionCandidateTrainingTerminalV1 {
    Installed(PromotionCandidateTrainingManifestV1),
    ExistingIdentical(PromotionCandidateTrainingManifestV1),
    Refused(PromotionCandidateTrainingRefusalV1),
}

pub fn train_and_deploy_promotion_candidate_v1<R>(
    settings: &Settings,
    handoff: PromotionCandidateTrainingHandoffV1,
    candidate_root: &Path,
    data_root: &Path,
    lease: &CpuLease,
    progress_fn: R,
) -> PromotionCandidateTrainingTerminalV1
where
    R: Fn(ModelTrainingProgress) + Send + Sync + Clone + 'static,
{
    if let Err(error) = handoff.validate_against_settings_v1(settings) {
        return PromotionCandidateTrainingTerminalV1::Refused(error);
    }
    let sealed_hardware_plan_v1 = handoff
        .training_config()
        .sealed_hardware_plan_v1()
        .expect("validated promotion handoff has a sealed hardware plan")
        .clone();
    let staging = match install::create_staging_root_v1(candidate_root, data_root) {
        Ok(path) => path,
        Err(error) => return PromotionCandidateTrainingTerminalV1::Refused(error),
    };
    let orchestrator = TrainingOrchestrator::new(settings.clone(), staging.clone())
        .with_data_root(data_root)
        .with_oos_lock_from_ms(handoff.oos_cutoff_ms())
        .with_sealed_hardware_plan_v1(sealed_hardware_plan_v1);
    let summary = orchestrator.train_canonical_series_receipt_with_progress(
        handoff.canonical_series(),
        handoff.base_timeframe(),
        handoff.search_input_receipt(),
        handoff.screening_contract(),
        lease,
        progress_fn,
    );
    match summary {
        Ok(summary) => {
            install_promotion_candidate_model_tree_v1(candidate_root, &staging, handoff, &summary)
        }
        Err(error) => {
            install::cleanup_staging_root_v1(candidate_root, &staging);
            PromotionCandidateTrainingTerminalV1::Refused(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::TrainingFailed,
                format!("exact receipt-bound candidate training failed: {error}"),
            ))
        }
    }
}

fn validate_series_against_search_v1(
    series: &CanonicalDatasetSeriesReceiptV1,
    receipt: &CanonicalSearchInputReceiptV2,
) -> Result<(), PromotionCandidateTrainingRefusalV1> {
    let selected = series
        .direct_timeframes()
        .iter()
        .map(|value| (value.identity().to_path_component(), value))
        .collect::<BTreeMap<_, _>>();
    if selected.len() != receipt.source_bindings().len() {
        return Err(refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::InputReceiptMismatch,
            "selected series and canonical search receipt have different source counts",
        ));
    }
    let mut seen = BTreeSet::new();
    for binding in receipt.source_bindings() {
        let Some(expected) = selected.get(binding.dataset_identity()) else {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InputReceiptMismatch,
                format!(
                    "search source {} is absent from the selected series",
                    binding.dataset_identity()
                ),
            ));
        };
        let vortex_sha256 = generation_sha256_v1(expected.generation_id()).ok_or_else(|| {
            refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InputReceiptMismatch,
                "selected generation id is not canonical g1 SHA-256 Vortex",
            )
        })?;
        if expected.generation_id() != binding.generation_id()
            || expected.manifest_binding_sha256() != binding.manifest_sha256()
            || vortex_sha256 != binding.vortex_sha256()
            || !seen.insert(binding.dataset_identity())
        {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::InputReceiptMismatch,
                format!(
                    "search source {} differs from its exact selected generation",
                    binding.dataset_identity()
                ),
            ));
        }
    }
    Ok(())
}

fn serialize_timeframe_v1<S>(
    timeframe: &CanonicalTimeframe,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(timeframe.as_str())
}

fn deserialize_timeframe_v1<'de, D>(deserializer: D) -> Result<CanonicalTimeframe, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    value.parse().map_err(serde::de::Error::custom)
}

fn generation_sha256_v1(generation_id: &str) -> Option<&str> {
    generation_id
        .strip_prefix("g1-")
        .and_then(|value| value.strip_suffix(".vortex"))
        .filter(|value| is_sha256_v1(value))
}

fn validate_model_names_v1(names: &[String]) -> Result<(), PromotionCandidateTrainingRefusalV1> {
    if names.is_empty() || names.len() > MAX_PLANNED_MODELS_V1 {
        return Err(refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::ModelConfigMismatch,
            "configured model inventory is empty or exceeds 256 entries",
        ));
    }
    let mut seen = BTreeSet::new();
    for name in names {
        if name.len() > MAX_MODEL_NAME_BYTES_V1
            || name.is_empty()
            || name == "."
            || name == ".."
            || name.chars().any(char::is_control)
            || name.contains(['/', '\\'])
            || !seen.insert(name)
        {
            return Err(refusal_v1(
                PromotionCandidateTrainingRefusalCodeV1::ModelConfigMismatch,
                format!("invalid or repeated configured model name {name:?}"),
            ));
        }
    }
    Ok(())
}

fn validate_sha256_v1(value: &str, label: &str) -> Result<(), PromotionCandidateTrainingRefusalV1> {
    if is_sha256_v1(value) {
        Ok(())
    } else {
        Err(refusal_v1(
            PromotionCandidateTrainingRefusalCodeV1::InvalidHandoff,
            format!("{label} is not canonical lowercase SHA-256"),
        ))
    }
}

fn is_sha256_v1(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hash_json_v1<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
    code: PromotionCandidateTrainingRefusalCodeV1,
) -> Result<String, PromotionCandidateTrainingRefusalV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        refusal_v1(
            code,
            format!("encode exact config identity material: {error}"),
        )
    })?;
    Ok(domain_sha256_v1(domain, &bytes))
}

pub(crate) fn domain_sha256_v1(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}
