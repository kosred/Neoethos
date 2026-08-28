//! Broker-independent historical research in gross reference-risk units.
//!
//! This lane deliberately has no financial execution surface. It consumes an
//! exact canonical search input, evaluates price-native barrier geometry on the
//! CPU, and produces an artifact that is structurally ineligible for promotion.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use neoethos_core::execution::BudgetedCpuExecutor;
use neoethos_core::execution_budget::CpuLeaseTransfer;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::data_selection::{
    CanonicalSearchArtifactScopeV2, CanonicalSearchRunInputV2, CanonicalSearchWindowRoleV1,
};

pub const HISTORICAL_RESEARCH_SCHEMA_VERSION: u16 = 2;
pub const HISTORICAL_RESEARCH_EXECUTION_CONTRACT_SCHEMA_VERSION: u16 = 1;
pub const HISTORICAL_CANDIDATE_SCAN_SCHEMA_VERSION: u16 = 2;
pub const HISTORICAL_CANDIDATE_SIGNAL_GENERATOR_ID: &str =
    "neoethos.search-engine.signals-for-gene.linear-threshold.v1";
pub const HISTORICAL_CANDIDATE_RANKING_POLICY_ID: &str = "gross-r-expectancy-desc_drawdown-asc_win-rate-desc_payoff-desc_trade-count-desc_candidate-identity-asc.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoricalResearchBackendV1 {
    CpuOnly,
    Auto,
    GpuOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoricalResearchSignalV1 {
    Flat,
    Long,
    Short,
}

#[derive(Debug, Clone, Copy)]
pub enum HistoricalResearchGeometryV1<'a> {
    PriceNativeVolatilityDistance {
        distance_semantic_id: &'a str,
        distance_by_signal_bar: &'a [f64],
        stop_multiple: f64,
        target_multiple: f64,
    },
    FixedPips {
        stop_pips: f64,
        target_pips: f64,
    },
}

#[derive(Debug)]
pub struct HistoricalResearchRequestV2<'request, 'data> {
    pub input: &'request CanonicalSearchRunInputV2<'data>,
    pub backend: HistoricalResearchBackendV1,
    pub signal_semantic_id: &'request str,
    pub signals: &'request [HistoricalResearchSignalV1],
    pub geometry: HistoricalResearchGeometryV1<'request>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoricalResearchArtifactClassV1 {
    ResearchOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoricalResearchPromotionEligibilityV1 {
    NotPromotionEligible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoricalResearchAccountingV1 {
    GrossReferenceR,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoricalResearchIntrabarAmbiguityV1 {
    StopBeforeTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoricalResearchSignalTimingV1 {
    PriorClosedBar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoricalResearchEntryReferenceV1 {
    NextBarOpen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoricalResearchPriceBasisV1 {
    CanonicalReferenceOhlc,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HistoricalResearchSeriesBindingV1 {
    semantic_id: String,
    value_count: u64,
    values_sha256: String,
}

impl HistoricalResearchSeriesBindingV1 {
    pub fn semantic_id(&self) -> &str {
        &self.semantic_id
    }

    pub const fn value_count(&self) -> u64 {
        self.value_count
    }

    pub fn values_sha256(&self) -> &str {
        &self.values_sha256
    }

    fn validate(&self, label: &str) -> Result<(), HistoricalResearchError> {
        validate_semantic_id(label, &self.semantic_id)?;
        if self.value_count < 2 {
            return Err(invalid_contract(format!(
                "{label} value count {} is below two",
                self.value_count
            )));
        }
        validate_sha256_hex(label, &self.values_sha256)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HistoricalResearchPriceNativeVolatilityDistanceV1 {
    distance_source: HistoricalResearchSeriesBindingV1,
    stop_multiple: f64,
    target_multiple: f64,
}

impl HistoricalResearchPriceNativeVolatilityDistanceV1 {
    pub const fn distance_source(&self) -> &HistoricalResearchSeriesBindingV1 {
        &self.distance_source
    }

    pub const fn stop_multiple(&self) -> f64 {
        self.stop_multiple
    }

    pub const fn target_multiple(&self) -> f64 {
        self.target_multiple
    }

    fn validate(&self) -> Result<(), HistoricalResearchError> {
        self.distance_source.validate("distance source")?;
        require_finite_scalar("stop_multiple", self.stop_multiple)?;
        require_finite_scalar("target_multiple", self.target_multiple)?;
        require_positive_scalar("stop_multiple", self.stop_multiple)?;
        require_positive_scalar("target_multiple", self.target_multiple)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HistoricalResearchExecutionContractV1 {
    schema_version: u16,
    signal_timing: HistoricalResearchSignalTimingV1,
    entry_reference: HistoricalResearchEntryReferenceV1,
    price_basis: HistoricalResearchPriceBasisV1,
    signal_source: HistoricalResearchSeriesBindingV1,
    geometry: HistoricalResearchPriceNativeVolatilityDistanceV1,
}

impl HistoricalResearchExecutionContractV1 {
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub const fn signal_timing(&self) -> HistoricalResearchSignalTimingV1 {
        self.signal_timing
    }

    pub const fn entry_reference(&self) -> HistoricalResearchEntryReferenceV1 {
        self.entry_reference
    }

    pub const fn price_basis(&self) -> HistoricalResearchPriceBasisV1 {
        self.price_basis
    }

    pub const fn signal_source(&self) -> &HistoricalResearchSeriesBindingV1 {
        &self.signal_source
    }

    pub const fn geometry(&self) -> &HistoricalResearchPriceNativeVolatilityDistanceV1 {
        &self.geometry
    }

    pub fn validate(&self) -> Result<(), HistoricalResearchError> {
        if self.schema_version != HISTORICAL_RESEARCH_EXECUTION_CONTRACT_SCHEMA_VERSION {
            return Err(invalid_contract(format!(
                "unsupported execution-contract schema {}; expected {}",
                self.schema_version, HISTORICAL_RESEARCH_EXECUTION_CONTRACT_SCHEMA_VERSION
            )));
        }
        self.signal_source.validate("signal source")?;
        self.geometry.validate()?;
        if self.signal_source.value_count != self.geometry.distance_source.value_count {
            return Err(invalid_contract(
                "signal and distance bindings have different value counts",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoricalResearchMetricsV1 {
    gross_r_expectancy: Option<f64>,
    gross_r_max_drawdown: f64,
    gross_r_win_rate: Option<f64>,
    gross_r_payoff: Option<f64>,
    trade_count: u64,
}

impl HistoricalResearchMetricsV1 {
    pub const fn gross_r_expectancy(&self) -> Option<f64> {
        self.gross_r_expectancy
    }

    pub const fn gross_r_max_drawdown(&self) -> f64 {
        self.gross_r_max_drawdown
    }

    pub const fn gross_r_win_rate(&self) -> Option<f64> {
        self.gross_r_win_rate
    }

    pub const fn gross_r_payoff(&self) -> Option<f64> {
        self.gross_r_payoff
    }

    pub const fn trade_count(&self) -> u64 {
        self.trade_count
    }

    fn validate(&self) -> Result<(), HistoricalResearchError> {
        for (field, value) in [
            ("gross_r_expectancy", self.gross_r_expectancy),
            ("gross_r_win_rate", self.gross_r_win_rate),
            ("gross_r_payoff", self.gross_r_payoff),
        ] {
            if value.is_some_and(|value| !value.is_finite()) {
                return Err(invalid_artifact(format!("{field} is not finite")));
            }
        }
        if !self.gross_r_max_drawdown.is_finite() || self.gross_r_max_drawdown < 0.0 {
            return Err(invalid_artifact(
                "gross_r_max_drawdown is not finite and non-negative",
            ));
        }
        if self
            .gross_r_win_rate
            .is_some_and(|value| !(0.0..=1.0).contains(&value))
        {
            return Err(invalid_artifact("gross_r_win_rate is outside 0..=1"));
        }
        if self.gross_r_payoff.is_some_and(|value| value <= 0.0) {
            return Err(invalid_artifact("gross_r_payoff is not positive"));
        }
        if self.trade_count == 0
            && (self.gross_r_expectancy.is_some()
                || self.gross_r_win_rate.is_some()
                || self.gross_r_payoff.is_some())
        {
            return Err(invalid_artifact(
                "zero-trade metrics contain defined sample statistics",
            ));
        }
        if self.trade_count > 0
            && (self.gross_r_expectancy.is_none() || self.gross_r_win_rate.is_none())
        {
            return Err(invalid_artifact(
                "non-empty metrics omit expectancy or win rate",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HistoricalResearchArtifactV2 {
    schema_version: u16,
    artifact_class: HistoricalResearchArtifactClassV1,
    promotion_eligibility: HistoricalResearchPromotionEligibilityV1,
    backend: HistoricalResearchBackendV1,
    accounting: HistoricalResearchAccountingV1,
    intrabar_ambiguity: HistoricalResearchIntrabarAmbiguityV1,
    scope: CanonicalSearchArtifactScopeV2,
    execution_contract: HistoricalResearchExecutionContractV1,
    evidence_identity_sha256: String,
    metrics: HistoricalResearchMetricsV1,
}

impl HistoricalResearchArtifactV2 {
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub const fn artifact_class(&self) -> HistoricalResearchArtifactClassV1 {
        self.artifact_class
    }

    pub const fn promotion_eligibility(&self) -> HistoricalResearchPromotionEligibilityV1 {
        self.promotion_eligibility
    }

    pub const fn backend(&self) -> HistoricalResearchBackendV1 {
        self.backend
    }

    pub const fn accounting(&self) -> HistoricalResearchAccountingV1 {
        self.accounting
    }

    pub const fn intrabar_ambiguity(&self) -> HistoricalResearchIntrabarAmbiguityV1 {
        self.intrabar_ambiguity
    }

    pub const fn scope(&self) -> &CanonicalSearchArtifactScopeV2 {
        &self.scope
    }

    pub const fn execution_contract(&self) -> &HistoricalResearchExecutionContractV1 {
        &self.execution_contract
    }

    pub fn evidence_identity_sha256(&self) -> &str {
        &self.evidence_identity_sha256
    }

    pub const fn metrics(&self) -> &HistoricalResearchMetricsV1 {
        &self.metrics
    }

    pub fn validate(&self) -> Result<(), HistoricalResearchError> {
        if self.schema_version != HISTORICAL_RESEARCH_SCHEMA_VERSION {
            return Err(invalid_artifact(format!(
                "unsupported artifact schema {}; expected {}",
                self.schema_version, HISTORICAL_RESEARCH_SCHEMA_VERSION
            )));
        }
        if self.artifact_class != HistoricalResearchArtifactClassV1::ResearchOnly
            || self.promotion_eligibility
                != HistoricalResearchPromotionEligibilityV1::NotPromotionEligible
            || self.backend != HistoricalResearchBackendV1::CpuOnly
            || self.accounting != HistoricalResearchAccountingV1::GrossReferenceR
            || self.intrabar_ambiguity != HistoricalResearchIntrabarAmbiguityV1::StopBeforeTarget
        {
            return Err(invalid_artifact(
                "artifact classification or execution policy is not the v2 research contract",
            ));
        }
        self.scope
            .validate()
            .map_err(|error| HistoricalResearchError::InvalidCanonicalScope {
                detail: error.to_string(),
            })?;
        self.execution_contract.validate()?;
        self.metrics.validate()?;
        validate_sha256_hex("evidence identity", &self.evidence_identity_sha256)
            .map_err(|error| invalid_artifact(error.to_string()))?;
        let expected = evidence_identity_sha256(&self.scope, &self.execution_contract)?;
        if self.evidence_identity_sha256 != expected {
            return Err(invalid_artifact(
                "evidence identity does not match scope and execution contract",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoricalCandidateScanKindV1 {
    ExplicitOrderedCandidateScan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoricalCandidateFailurePolicyV1 {
    FailEntireScan,
    RetainFailedCandidate,
}

#[derive(Debug, Clone, Copy)]
pub struct HistoricalCandidateDistanceSourceV1<'a> {
    pub receipt_sha256: &'a str,
    pub semantic_id: &'a str,
    pub values: &'a [f64],
}

#[derive(Debug)]
pub struct HistoricalCandidateScanRequestV2<'request, 'data> {
    pub input: &'request CanonicalSearchRunInputV2<'data>,
    pub backend: HistoricalResearchBackendV1,
    pub candidates: &'request [crate::genetic::Gene],
    pub failure_policy: HistoricalCandidateFailurePolicyV1,
    pub distance_source: HistoricalCandidateDistanceSourceV1<'request>,
    pub stop_multiple: f64,
    pub target_multiple: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoricalCandidateFailureStageV1 {
    UnsupportedSignalMode,
    FeatureValidity,
    SignalGeneration,
    ResearchEvaluation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HistoricalCandidateFailureV1 {
    stage: HistoricalCandidateFailureStageV1,
    detail: String,
}

impl HistoricalCandidateFailureV1 {
    pub const fn stage(&self) -> HistoricalCandidateFailureStageV1 {
        self.stage
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoricalCandidateResultStatusV1 {
    Evaluated,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HistoricalCandidateResultV2 {
    input_ordinal: u64,
    candidate_identity_sha256: String,
    status: HistoricalCandidateResultStatusV1,
    signal_identity_sha256: Option<String>,
    artifact: Option<HistoricalResearchArtifactV2>,
    failure: Option<HistoricalCandidateFailureV1>,
}

impl HistoricalCandidateResultV2 {
    pub const fn input_ordinal(&self) -> u64 {
        self.input_ordinal
    }

    pub fn candidate_identity_sha256(&self) -> &str {
        &self.candidate_identity_sha256
    }

    pub const fn status(&self) -> HistoricalCandidateResultStatusV1 {
        self.status
    }

    pub fn signal_identity_sha256(&self) -> Option<&str> {
        self.signal_identity_sha256.as_deref()
    }

    pub const fn artifact(&self) -> Option<&HistoricalResearchArtifactV2> {
        self.artifact.as_ref()
    }

    pub const fn failure(&self) -> Option<&HistoricalCandidateFailureV1> {
        self.failure.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HistoricalCandidateRankV1 {
    rank: u64,
    input_ordinal: u64,
    candidate_identity_sha256: String,
}

impl HistoricalCandidateRankV1 {
    pub const fn rank(&self) -> u64 {
        self.rank
    }

    pub const fn input_ordinal(&self) -> u64 {
        self.input_ordinal
    }

    pub fn candidate_identity_sha256(&self) -> &str {
        &self.candidate_identity_sha256
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HistoricalCandidateScanContractV2 {
    schema_version: u16,
    search_kind: HistoricalCandidateScanKindV1,
    artifact_class: HistoricalResearchArtifactClassV1,
    promotion_eligibility: HistoricalResearchPromotionEligibilityV1,
    backend: HistoricalResearchBackendV1,
    failure_policy: HistoricalCandidateFailurePolicyV1,
    signal_generator_id: String,
    ranking_policy_id: String,
    scope: CanonicalSearchArtifactScopeV2,
    distance_source: HistoricalResearchSeriesBindingV1,
    stop_multiple: f64,
    target_multiple: f64,
    ordered_candidate_identities: Vec<String>,
}

impl HistoricalCandidateScanContractV2 {
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub const fn search_kind(&self) -> HistoricalCandidateScanKindV1 {
        self.search_kind
    }

    pub fn signal_generator_id(&self) -> &str {
        &self.signal_generator_id
    }

    pub fn ranking_policy_id(&self) -> &str {
        &self.ranking_policy_id
    }

    pub const fn scope(&self) -> &CanonicalSearchArtifactScopeV2 {
        &self.scope
    }

    pub const fn distance_source(&self) -> &HistoricalResearchSeriesBindingV1 {
        &self.distance_source
    }

    pub fn ordered_candidate_identities(&self) -> &[String] {
        &self.ordered_candidate_identities
    }

    fn validate(&self) -> Result<(), HistoricalCandidateScanError> {
        if self.schema_version != HISTORICAL_CANDIDATE_SCAN_SCHEMA_VERSION
            || self.search_kind != HistoricalCandidateScanKindV1::ExplicitOrderedCandidateScan
            || self.artifact_class != HistoricalResearchArtifactClassV1::ResearchOnly
            || self.promotion_eligibility
                != HistoricalResearchPromotionEligibilityV1::NotPromotionEligible
            || self.backend != HistoricalResearchBackendV1::CpuOnly
            || self.signal_generator_id != HISTORICAL_CANDIDATE_SIGNAL_GENERATOR_ID
            || self.ranking_policy_id != HISTORICAL_CANDIDATE_RANKING_POLICY_ID
        {
            return Err(invalid_scan_contract(
                "schema, classification, backend, or signal generator is not the v2 contract",
            ));
        }
        self.scope
            .validate()
            .map_err(|error| invalid_scan_contract(format!("scope: {error}")))?;
        self.distance_source
            .validate("candidate-scan distance source")
            .map_err(|error| invalid_scan_contract(error.to_string()))?;
        require_finite_scalar("stop_multiple", self.stop_multiple)
            .and_then(|_| require_positive_scalar("stop_multiple", self.stop_multiple))
            .map_err(|error| invalid_scan_contract(error.to_string()))?;
        require_finite_scalar("target_multiple", self.target_multiple)
            .and_then(|_| require_positive_scalar("target_multiple", self.target_multiple))
            .map_err(|error| invalid_scan_contract(error.to_string()))?;
        if self.ordered_candidate_identities.is_empty() {
            return Err(invalid_scan_contract(
                "ordered candidate identities are empty",
            ));
        }
        let mut unique = BTreeSet::new();
        for identity in &self.ordered_candidate_identities {
            validate_sha256_hex("candidate identity", identity)
                .map_err(|error| invalid_scan_contract(error.to_string()))?;
            if !unique.insert(identity) {
                return Err(invalid_scan_contract(
                    "ordered candidate identities contain a duplicate",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HistoricalCandidateScanResultV2 {
    contract: HistoricalCandidateScanContractV2,
    search_identity_sha256: String,
    results: Vec<HistoricalCandidateResultV2>,
    ranking: Vec<HistoricalCandidateRankV1>,
}

impl HistoricalCandidateScanResultV2 {
    pub const fn search_kind(&self) -> HistoricalCandidateScanKindV1 {
        self.contract.search_kind
    }

    pub const fn artifact_class(&self) -> HistoricalResearchArtifactClassV1 {
        self.contract.artifact_class
    }

    pub const fn promotion_eligibility(&self) -> HistoricalResearchPromotionEligibilityV1 {
        self.contract.promotion_eligibility
    }

    pub const fn backend(&self) -> HistoricalResearchBackendV1 {
        self.contract.backend
    }

    pub const fn contract(&self) -> &HistoricalCandidateScanContractV2 {
        &self.contract
    }

    pub fn search_identity_sha256(&self) -> &str {
        &self.search_identity_sha256
    }

    pub fn results(&self) -> &[HistoricalCandidateResultV2] {
        &self.results
    }

    pub fn ranking(&self) -> &[HistoricalCandidateRankV1] {
        &self.ranking
    }

    pub fn best_candidate_identity_sha256(&self) -> Option<&str> {
        self.ranking
            .first()
            .map(HistoricalCandidateRankV1::candidate_identity_sha256)
    }

    pub fn validate(&self) -> Result<(), HistoricalCandidateScanError> {
        self.contract.validate()?;
        validate_sha256_hex("candidate-scan identity", &self.search_identity_sha256)
            .map_err(|error| invalid_scan_contract(error.to_string()))?;
        let expected_identity = candidate_scan_identity_sha256(&self.contract)?;
        if self.search_identity_sha256 != expected_identity {
            return Err(invalid_scan_contract(
                "search identity does not match the exact scan contract",
            ));
        }
        if self.results.len() != self.contract.ordered_candidate_identities.len() {
            return Err(invalid_scan_contract(
                "candidate result count differs from the ordered candidate set",
            ));
        }
        for (index, result) in self.results.iter().enumerate() {
            if result.input_ordinal != index as u64
                || result.candidate_identity_sha256
                    != self.contract.ordered_candidate_identities[index]
            {
                return Err(invalid_scan_contract(
                    "candidate result order or identity differs from the scan contract",
                ));
            }
            match result.status {
                HistoricalCandidateResultStatusV1::Evaluated
                    if result.artifact.is_some()
                        && result.failure.is_none()
                        && result.signal_identity_sha256.is_some() => {}
                HistoricalCandidateResultStatusV1::Failed
                    if result.artifact.is_none()
                        && result.failure.is_some()
                        && result.signal_identity_sha256.is_none() => {}
                _ => {
                    return Err(invalid_scan_contract(
                        "candidate status disagrees with its artifact/failure fields",
                    ));
                }
            }
        }
        let evaluated = self
            .results
            .iter()
            .filter(|result| result.status == HistoricalCandidateResultStatusV1::Evaluated)
            .map(|result| result.candidate_identity_sha256.as_str())
            .collect::<BTreeSet<_>>();
        let ranked = self
            .ranking
            .iter()
            .map(|entry| entry.candidate_identity_sha256.as_str())
            .collect::<BTreeSet<_>>();
        if evaluated != ranked || self.ranking.len() != ranked.len() {
            return Err(invalid_scan_contract(
                "ranking does not contain every evaluated candidate exactly once",
            ));
        }
        for (index, entry) in self.ranking.iter().enumerate() {
            if entry.rank != index as u64 + 1 {
                return Err(invalid_scan_contract(
                    "ranking positions are not contiguous",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HistoricalResearchError {
    UnsupportedBackend {
        requested: HistoricalResearchBackendV1,
    },
    UnsupportedGeometry,
    TooFewRows {
        actual: usize,
    },
    ShapeMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFinite {
        field: &'static str,
        index: Option<usize>,
    },
    NonPositive {
        field: &'static str,
        index: Option<usize>,
    },
    InvalidCandle {
        index: usize,
    },
    InvalidCanonicalScope {
        detail: String,
    },
    InvalidExecutionContract {
        detail: String,
    },
    InvalidArtifact {
        detail: String,
    },
}

impl fmt::Display for HistoricalResearchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedBackend { requested } => {
                write!(
                    formatter,
                    "historical research requires CpuOnly; got {requested:?}"
                )
            }
            Self::UnsupportedGeometry => formatter.write_str(
                "historical research requires price-native volatility distance geometry",
            ),
            Self::TooFewRows { actual } => {
                write!(
                    formatter,
                    "historical research requires at least two rows; got {actual}"
                )
            }
            Self::ShapeMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "historical research {field} length is {actual}; expected {expected}"
            ),
            Self::NonFinite { field, index } => match index {
                Some(index) => write!(
                    formatter,
                    "historical research {field}[{index}] is not finite"
                ),
                None => write!(formatter, "historical research {field} is not finite"),
            },
            Self::NonPositive { field, index } => match index {
                Some(index) => write!(
                    formatter,
                    "historical research {field}[{index}] is not positive"
                ),
                None => write!(formatter, "historical research {field} is not positive"),
            },
            Self::InvalidCandle { index } => write!(
                formatter,
                "historical research OHLC geometry is invalid at row {index}"
            ),
            Self::InvalidCanonicalScope { detail } => {
                write!(
                    formatter,
                    "historical research canonical scope is invalid: {detail}"
                )
            }
            Self::InvalidExecutionContract { detail } => {
                write!(
                    formatter,
                    "historical research execution contract is invalid: {detail}"
                )
            }
            Self::InvalidArtifact { detail } => {
                write!(
                    formatter,
                    "historical research artifact is invalid: {detail}"
                )
            }
        }
    }
}

impl Error for HistoricalResearchError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HistoricalCandidateScanError {
    UnsupportedBackend {
        requested: HistoricalResearchBackendV1,
    },
    EmptyCandidateSet,
    DistanceReceiptMismatch {
        expected: String,
        received: String,
    },
    InvalidDistanceSource {
        detail: String,
    },
    InvalidCandidateIdentity {
        candidate_index: usize,
        detail: String,
    },
    DuplicateCandidateIdentity {
        first_index: usize,
        duplicate_index: usize,
        identity_sha256: String,
    },
    CandidateFailed {
        candidate_index: usize,
        candidate_identity_sha256: String,
        stage: HistoricalCandidateFailureStageV1,
        detail: String,
    },
    CpuExecution {
        detail: String,
    },
    InvalidScanContract {
        detail: String,
    },
}

impl fmt::Display for HistoricalCandidateScanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedBackend { requested } => {
                write!(
                    formatter,
                    "candidate scan requires CpuOnly; got {requested:?}"
                )
            }
            Self::EmptyCandidateSet => formatter.write_str("candidate scan set is empty"),
            Self::DistanceReceiptMismatch { expected, received } => write!(
                formatter,
                "candidate distance receipt {received} does not match input receipt {expected}"
            ),
            Self::InvalidDistanceSource { detail } => {
                write!(formatter, "candidate distance source is invalid: {detail}")
            }
            Self::InvalidCandidateIdentity {
                candidate_index,
                detail,
            } => write!(
                formatter,
                "candidate {candidate_index} has no valid exact identity: {detail}"
            ),
            Self::DuplicateCandidateIdentity {
                first_index,
                duplicate_index,
                identity_sha256,
            } => write!(
                formatter,
                "candidates {first_index} and {duplicate_index} share identity {identity_sha256}"
            ),
            Self::CandidateFailed {
                candidate_index,
                candidate_identity_sha256,
                stage,
                detail,
            } => write!(
                formatter,
                "candidate {candidate_index} ({candidate_identity_sha256}) failed at {stage:?}: {detail}"
            ),
            Self::CpuExecution { detail } => {
                write!(formatter, "candidate scan CPU execution failed: {detail}")
            }
            Self::InvalidScanContract { detail } => {
                write!(formatter, "candidate scan contract is invalid: {detail}")
            }
        }
    }
}

impl Error for HistoricalCandidateScanError {}

pub fn run_historical_research_v2(
    request: HistoricalResearchRequestV2<'_, '_>,
) -> Result<HistoricalResearchArtifactV2, HistoricalResearchError> {
    if request.backend != HistoricalResearchBackendV1::CpuOnly {
        return Err(HistoricalResearchError::UnsupportedBackend {
            requested: request.backend,
        });
    }

    let (distance_semantic_id, distances, stop_multiple, target_multiple) = match request.geometry {
        HistoricalResearchGeometryV1::PriceNativeVolatilityDistance {
            distance_semantic_id,
            distance_by_signal_bar,
            stop_multiple,
            target_multiple,
        } => (
            distance_semantic_id,
            distance_by_signal_bar,
            stop_multiple,
            target_multiple,
        ),
        HistoricalResearchGeometryV1::FixedPips { .. } => {
            return Err(HistoricalResearchError::UnsupportedGeometry);
        }
    };

    validate_semantic_id("signal source", request.signal_semantic_id)?;
    validate_semantic_id("distance source", distance_semantic_id)?;

    let ohlcv = request.input.ohlcv();
    let rows = ohlcv.len();
    if rows < 2 {
        return Err(HistoricalResearchError::TooFewRows { actual: rows });
    }
    require_shape("signals", request.signals.len(), rows)?;
    require_shape("distance_by_signal_bar", distances.len(), rows)?;
    require_shape("open", ohlcv.open.len(), rows)?;
    require_shape("high", ohlcv.high.len(), rows)?;
    require_shape("low", ohlcv.low.len(), rows)?;

    require_finite_scalar("stop_multiple", stop_multiple)?;
    require_finite_scalar("target_multiple", target_multiple)?;
    require_positive_scalar("stop_multiple", stop_multiple)?;
    require_positive_scalar("target_multiple", target_multiple)?;

    for row in 0..rows {
        let values = [
            ("open", ohlcv.open[row]),
            ("high", ohlcv.high[row]),
            ("low", ohlcv.low[row]),
            ("close", ohlcv.close[row]),
            ("distance_by_signal_bar", distances[row]),
        ];
        for (field, value) in values {
            if !value.is_finite() {
                return Err(HistoricalResearchError::NonFinite {
                    field,
                    index: Some(row),
                });
            }
        }
        if distances[row] <= 0.0 {
            return Err(HistoricalResearchError::NonPositive {
                field: "distance_by_signal_bar",
                index: Some(row),
            });
        }
        if ohlcv.low[row] > ohlcv.high[row]
            || ohlcv.open[row] < ohlcv.low[row]
            || ohlcv.open[row] > ohlcv.high[row]
            || ohlcv.close[row] < ohlcv.low[row]
            || ohlcv.close[row] > ohlcv.high[row]
        {
            return Err(HistoricalResearchError::InvalidCandle { index: row });
        }
    }

    let execution_contract = HistoricalResearchExecutionContractV1 {
        schema_version: HISTORICAL_RESEARCH_EXECUTION_CONTRACT_SCHEMA_VERSION,
        signal_timing: HistoricalResearchSignalTimingV1::PriorClosedBar,
        entry_reference: HistoricalResearchEntryReferenceV1::NextBarOpen,
        price_basis: HistoricalResearchPriceBasisV1::CanonicalReferenceOhlc,
        signal_source: HistoricalResearchSeriesBindingV1 {
            semantic_id: request.signal_semantic_id.to_owned(),
            value_count: rows as u64,
            values_sha256: signal_values_sha256(request.signals),
        },
        geometry: HistoricalResearchPriceNativeVolatilityDistanceV1 {
            distance_source: HistoricalResearchSeriesBindingV1 {
                semantic_id: distance_semantic_id.to_owned(),
                value_count: rows as u64,
                values_sha256: distance_values_sha256(distances),
            },
            stop_multiple,
            target_multiple,
        },
    };
    execution_contract.validate()?;

    let scope = CanonicalSearchArtifactScopeV2::from_run_input(
        CanonicalSearchWindowRoleV1::DiscoveryInput,
        request.input,
    )
    .map_err(|error| HistoricalResearchError::InvalidCanonicalScope {
        detail: error.to_string(),
    })?;

    let mut outcomes = Vec::new();
    let mut signal_bar = 0_usize;
    while signal_bar + 1 < rows {
        let direction = match request.signals[signal_bar] {
            HistoricalResearchSignalV1::Flat => {
                signal_bar += 1;
                continue;
            }
            HistoricalResearchSignalV1::Long => 1.0,
            HistoricalResearchSignalV1::Short => -1.0,
        };

        let stop_distance = distances[signal_bar] * stop_multiple;
        let target_distance = distances[signal_bar] * target_multiple;
        if !stop_distance.is_finite() {
            return Err(HistoricalResearchError::NonFinite {
                field: "resolved_stop_distance",
                index: Some(signal_bar),
            });
        }
        if !target_distance.is_finite() {
            return Err(HistoricalResearchError::NonFinite {
                field: "resolved_target_distance",
                index: Some(signal_bar),
            });
        }
        if stop_distance <= 0.0 {
            return Err(HistoricalResearchError::NonPositive {
                field: "resolved_stop_distance",
                index: Some(signal_bar),
            });
        }
        if target_distance <= 0.0 {
            return Err(HistoricalResearchError::NonPositive {
                field: "resolved_target_distance",
                index: Some(signal_bar),
            });
        }

        let entry_bar = signal_bar + 1;
        let entry = ohlcv.open[entry_bar];
        let stop = entry - direction * stop_distance;
        let target = entry + direction * target_distance;
        if !stop.is_finite() || !target.is_finite() {
            return Err(HistoricalResearchError::NonFinite {
                field: "resolved_barrier",
                index: Some(signal_bar),
            });
        }
        let stop_is_strictly_beyond_entry = if direction > 0.0 {
            stop < entry
        } else {
            stop > entry
        };
        if !stop_is_strictly_beyond_entry {
            return Err(HistoricalResearchError::NonPositive {
                field: "resolved_stop_barrier_distance",
                index: Some(signal_bar),
            });
        }
        let target_is_strictly_beyond_entry = if direction > 0.0 {
            target > entry
        } else {
            target < entry
        };
        if !target_is_strictly_beyond_entry {
            return Err(HistoricalResearchError::NonPositive {
                field: "resolved_target_barrier_distance",
                index: Some(signal_bar),
            });
        }

        let mut exit_bar = rows - 1;
        let mut outcome = None;
        for bar in entry_bar..rows {
            let stop_hit = if direction > 0.0 {
                ohlcv.low[bar] <= stop
            } else {
                ohlcv.high[bar] >= stop
            };
            let target_hit = if direction > 0.0 {
                ohlcv.high[bar] >= target
            } else {
                ohlcv.low[bar] <= target
            };
            if stop_hit {
                outcome = Some(-1.0);
                exit_bar = bar;
                break;
            }
            if target_hit {
                outcome = Some(target_distance / stop_distance);
                exit_bar = bar;
                break;
            }
        }

        let gross_r =
            outcome.unwrap_or_else(|| direction * (ohlcv.close[rows - 1] - entry) / stop_distance);
        if !gross_r.is_finite() {
            return Err(HistoricalResearchError::NonFinite {
                field: "gross_r_outcome",
                index: Some(signal_bar),
            });
        }
        outcomes.push(gross_r);
        signal_bar = exit_bar;
    }

    let evidence_identity_sha256 = evidence_identity_sha256(&scope, &execution_contract)?;
    let artifact = HistoricalResearchArtifactV2 {
        schema_version: HISTORICAL_RESEARCH_SCHEMA_VERSION,
        artifact_class: HistoricalResearchArtifactClassV1::ResearchOnly,
        promotion_eligibility: HistoricalResearchPromotionEligibilityV1::NotPromotionEligible,
        backend: HistoricalResearchBackendV1::CpuOnly,
        accounting: HistoricalResearchAccountingV1::GrossReferenceR,
        intrabar_ambiguity: HistoricalResearchIntrabarAmbiguityV1::StopBeforeTarget,
        scope,
        execution_contract,
        evidence_identity_sha256,
        metrics: metrics_from_outcomes(&outcomes),
    };
    artifact.validate()?;
    Ok(artifact)
}

pub fn scan_historical_candidates_v2(
    request: HistoricalCandidateScanRequestV2<'_, '_>,
    executor: &BudgetedCpuExecutor,
    transfer: CpuLeaseTransfer,
) -> Result<HistoricalCandidateScanResultV2, HistoricalCandidateScanError> {
    if request.backend != HistoricalResearchBackendV1::CpuOnly {
        return Err(HistoricalCandidateScanError::UnsupportedBackend {
            requested: request.backend,
        });
    }
    if request.candidates.is_empty() {
        return Err(HistoricalCandidateScanError::EmptyCandidateSet);
    }

    let input_receipt_sha256 = request
        .input
        .receipt()
        .identity_sha256()
        .map_err(|error| invalid_scan_contract(format!("input receipt: {error}")))?;
    if request.distance_source.receipt_sha256 != input_receipt_sha256 {
        return Err(HistoricalCandidateScanError::DistanceReceiptMismatch {
            expected: input_receipt_sha256,
            received: request.distance_source.receipt_sha256.to_owned(),
        });
    }
    validate_semantic_id(
        "candidate distance source",
        request.distance_source.semantic_id,
    )
    .map_err(
        |error| HistoricalCandidateScanError::InvalidDistanceSource {
            detail: error.to_string(),
        },
    )?;
    let rows = request.input.ohlcv().len();
    if request.distance_source.values.len() != rows {
        return Err(HistoricalCandidateScanError::InvalidDistanceSource {
            detail: format!(
                "distance length is {}; input has {rows} rows",
                request.distance_source.values.len()
            ),
        });
    }
    for (index, value) in request.distance_source.values.iter().copied().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(HistoricalCandidateScanError::InvalidDistanceSource {
                detail: format!("distance[{index}] is not finite and positive"),
            });
        }
    }
    for (field, value) in [
        ("stop_multiple", request.stop_multiple),
        ("target_multiple", request.target_multiple),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(HistoricalCandidateScanError::InvalidDistanceSource {
                detail: format!("{field} is not finite and positive"),
            });
        }
    }

    let mut ordered_candidate_identities = Vec::with_capacity(request.candidates.len());
    let mut first_by_identity = BTreeMap::<String, usize>::new();
    for (candidate_index, candidate) in request.candidates.iter().enumerate() {
        let identity =
            historical_candidate_signal_identity_sha256(candidate).map_err(|detail| {
                HistoricalCandidateScanError::InvalidCandidateIdentity {
                    candidate_index,
                    detail,
                }
            })?;
        if let Some(&first_index) = first_by_identity.get(&identity) {
            return Err(HistoricalCandidateScanError::DuplicateCandidateIdentity {
                first_index,
                duplicate_index: candidate_index,
                identity_sha256: identity,
            });
        }
        first_by_identity.insert(identity.clone(), candidate_index);
        ordered_candidate_identities.push(identity);
    }

    let scope = CanonicalSearchArtifactScopeV2::from_run_input(
        CanonicalSearchWindowRoleV1::DiscoveryInput,
        request.input,
    )
    .map_err(|error| invalid_scan_contract(format!("input scope: {error}")))?;
    let contract = HistoricalCandidateScanContractV2 {
        schema_version: HISTORICAL_CANDIDATE_SCAN_SCHEMA_VERSION,
        search_kind: HistoricalCandidateScanKindV1::ExplicitOrderedCandidateScan,
        artifact_class: HistoricalResearchArtifactClassV1::ResearchOnly,
        promotion_eligibility: HistoricalResearchPromotionEligibilityV1::NotPromotionEligible,
        backend: HistoricalResearchBackendV1::CpuOnly,
        failure_policy: request.failure_policy,
        signal_generator_id: HISTORICAL_CANDIDATE_SIGNAL_GENERATOR_ID.to_owned(),
        ranking_policy_id: HISTORICAL_CANDIDATE_RANKING_POLICY_ID.to_owned(),
        scope,
        distance_source: HistoricalResearchSeriesBindingV1 {
            semantic_id: request.distance_source.semantic_id.to_owned(),
            value_count: rows as u64,
            values_sha256: distance_values_sha256(request.distance_source.values),
        },
        stop_multiple: request.stop_multiple,
        target_multiple: request.target_multiple,
        ordered_candidate_identities,
    };
    contract.validate()?;
    let search_identity_sha256 = candidate_scan_identity_sha256(&contract)?;

    // Indexed Rayon preserves input ordinal in the collected vector. Failure
    // policy is intentionally applied only after the join and in that exact
    // order, so worker width/scheduling cannot select a different first error
    // or perturb the serialized evidence.
    let evaluated = executor
        .execute(transfer, || {
            request
                .candidates
                .par_iter()
                .enumerate()
                .map(|(candidate_index, candidate)| {
                    evaluate_historical_candidate(
                        &request,
                        candidate_index,
                        candidate,
                        &contract.ordered_candidate_identities[candidate_index],
                    )
                })
                .collect::<Vec<_>>()
        })
        .map_err(|error| HistoricalCandidateScanError::CpuExecution {
            detail: error.to_string(),
        })?;

    let mut results = Vec::with_capacity(request.candidates.len());
    for (candidate_index, evaluation) in evaluated.into_iter().enumerate() {
        match evaluation {
            Ok(result) => results.push(result),
            Err(failure) => retain_or_fail_candidate(
                request.failure_policy,
                candidate_index,
                contract.ordered_candidate_identities[candidate_index].clone(),
                failure,
                &mut results,
            )?,
        }
    }

    let mut ranked_indices = results
        .iter()
        .enumerate()
        .filter_map(|(index, result)| {
            (result.status == HistoricalCandidateResultStatusV1::Evaluated).then_some(index)
        })
        .collect::<Vec<_>>();
    ranked_indices
        .sort_by(|left, right| compare_candidate_results(&results[*left], &results[*right]));
    let ranking = ranked_indices
        .into_iter()
        .enumerate()
        .map(|(rank, result_index)| HistoricalCandidateRankV1 {
            rank: rank as u64 + 1,
            input_ordinal: results[result_index].input_ordinal,
            candidate_identity_sha256: results[result_index].candidate_identity_sha256.clone(),
        })
        .collect();
    let output = HistoricalCandidateScanResultV2 {
        contract,
        search_identity_sha256,
        results,
        ranking,
    };
    output.validate()?;
    Ok(output)
}

/// Stable identity of the exact signal rule consumed by the historical scan.
///
/// Candidate generators use this same domain owner to deduplicate before a
/// scan; they must not reproduce or approximate the hashing contract.
pub fn historical_candidate_signal_identity_sha256(
    candidate: &crate::genetic::Gene,
) -> Result<String, String> {
    if candidate.indices.is_empty() {
        return Err("signal rule has no feature indices".to_owned());
    }
    if candidate.indices.len() != candidate.weights.len() {
        return Err(format!(
            "signal rule has {} indices but {} weights",
            candidate.indices.len(),
            candidate.weights.len()
        ));
    }
    if !candidate.long_threshold.is_finite() || !candidate.short_threshold.is_finite() {
        return Err("signal thresholds are not finite".to_owned());
    }
    if candidate.long_threshold <= candidate.short_threshold {
        return Err("long threshold is not strictly above short threshold".to_owned());
    }

    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.historical-candidate.signal-rule.v1\0");
    hasher.update((candidate.indices.len() as u64).to_le_bytes());
    for (position, (&index, &weight)) in
        candidate.indices.iter().zip(&candidate.weights).enumerate()
    {
        if !weight.is_finite() {
            return Err(format!("weight[{position}] is not finite"));
        }
        hasher.update((index as u64).to_le_bytes());
        hasher.update(canonical_f64_bits(weight).to_le_bytes());
    }
    hasher.update(canonical_f64_bits(candidate.long_threshold).to_le_bytes());
    hasher.update(canonical_f64_bits(candidate.short_threshold).to_le_bytes());
    for flag in candidate_signal_flags(candidate) {
        hasher.update([u8::from(flag)]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn canonical_f64_bits(value: f64) -> u64 {
    (if value == 0.0 { 0.0_f64 } else { value }).to_bits()
}

fn candidate_signal_flags(candidate: &crate::genetic::Gene) -> [bool; 11] {
    [
        candidate.use_ob,
        candidate.use_fvg,
        candidate.use_liq_sweep,
        candidate.mtf_confirmation,
        candidate.use_premium_discount,
        candidate.use_inducement,
        candidate.use_bos,
        candidate.use_choch,
        candidate.use_eqh,
        candidate.use_eql,
        candidate.use_displacement,
    ]
}

fn generate_candidate_signals(
    input: &CanonicalSearchRunInputV2<'_>,
    candidate: &crate::genetic::Gene,
) -> Result<Vec<HistoricalResearchSignalV1>, HistoricalCandidateFailureV1> {
    if candidate_signal_flags(candidate)
        .into_iter()
        .any(|flag| flag)
    {
        return Err(HistoricalCandidateFailureV1 {
            stage: HistoricalCandidateFailureStageV1::UnsupportedSignalMode,
            detail: "signals_for_gene does not apply structural gates; flag-bearing candidates are refused"
                .to_owned(),
        });
    }
    let mut checked_features = BTreeSet::new();
    for &feature_index in &candidate.indices {
        if feature_index >= input.features().n_features() {
            return Err(HistoricalCandidateFailureV1 {
                stage: HistoricalCandidateFailureStageV1::FeatureValidity,
                detail: format!(
                    "feature index {feature_index} is outside the {}-column exact frame",
                    input.features().n_features()
                ),
            });
        }
        if !checked_features.insert(feature_index) {
            continue;
        }
        let column = input
            .features()
            .feature_column(feature_index)
            .map_err(|error| HistoricalCandidateFailureV1 {
                stage: HistoricalCandidateFailureStageV1::FeatureValidity,
                detail: format!("project feature {feature_index}: {error}"),
            })?;
        for row in 0..column.values.len() {
            // Typed warmup/gap cells are deliberately ineligible inputs. The
            // production signal generator maps them to Flat and never reads
            // their numeric payload. A cell explicitly marked Valid, however,
            // must carry a finite value or the candidate fails closed.
            if column.validity[row].is_valid() && !column.values[row].is_finite() {
                return Err(HistoricalCandidateFailureV1 {
                    stage: HistoricalCandidateFailureStageV1::FeatureValidity,
                    detail: format!(
                        "feature {feature_index} row {row} is marked valid but is not finite"
                    ),
                });
            }
        }
    }

    let raw = crate::genetic::signals_for_gene(input.features(), candidate).map_err(|error| {
        HistoricalCandidateFailureV1 {
            stage: HistoricalCandidateFailureStageV1::SignalGeneration,
            detail: error.to_string(),
        }
    })?;
    if raw.len() != input.ohlcv().len() {
        return Err(HistoricalCandidateFailureV1 {
            stage: HistoricalCandidateFailureStageV1::SignalGeneration,
            detail: format!(
                "signal generator returned {} rows for {} input bars",
                raw.len(),
                input.ohlcv().len()
            ),
        });
    }
    raw.into_iter()
        .enumerate()
        .map(|(row, signal)| match signal {
            -1 => Ok(HistoricalResearchSignalV1::Short),
            0 => Ok(HistoricalResearchSignalV1::Flat),
            1 => Ok(HistoricalResearchSignalV1::Long),
            _ => Err(HistoricalCandidateFailureV1 {
                stage: HistoricalCandidateFailureStageV1::SignalGeneration,
                detail: format!("signal row {row} has unsupported value {signal}"),
            }),
        })
        .collect()
}

fn evaluate_historical_candidate(
    request: &HistoricalCandidateScanRequestV2<'_, '_>,
    candidate_index: usize,
    candidate: &crate::genetic::Gene,
    candidate_identity_sha256: &str,
) -> Result<HistoricalCandidateResultV2, HistoricalCandidateFailureV1> {
    let signals = generate_candidate_signals(request.input, candidate)?;
    let signal_semantic_id =
        format!("{HISTORICAL_CANDIDATE_SIGNAL_GENERATOR_ID}:{candidate_identity_sha256}");
    let artifact = run_historical_research_v2(HistoricalResearchRequestV2 {
        input: request.input,
        backend: HistoricalResearchBackendV1::CpuOnly,
        signal_semantic_id: &signal_semantic_id,
        signals: &signals,
        geometry: HistoricalResearchGeometryV1::PriceNativeVolatilityDistance {
            distance_semantic_id: request.distance_source.semantic_id,
            distance_by_signal_bar: request.distance_source.values,
            stop_multiple: request.stop_multiple,
            target_multiple: request.target_multiple,
        },
    })
    .map_err(|error| HistoricalCandidateFailureV1 {
        stage: HistoricalCandidateFailureStageV1::ResearchEvaluation,
        detail: error.to_string(),
    })?;
    let signal_identity_sha256 = artifact
        .execution_contract()
        .signal_source()
        .values_sha256()
        .to_owned();
    Ok(HistoricalCandidateResultV2 {
        input_ordinal: candidate_index as u64,
        candidate_identity_sha256: candidate_identity_sha256.to_owned(),
        status: HistoricalCandidateResultStatusV1::Evaluated,
        signal_identity_sha256: Some(signal_identity_sha256),
        artifact: Some(artifact),
        failure: None,
    })
}

fn retain_or_fail_candidate(
    policy: HistoricalCandidateFailurePolicyV1,
    candidate_index: usize,
    candidate_identity_sha256: String,
    failure: HistoricalCandidateFailureV1,
    results: &mut Vec<HistoricalCandidateResultV2>,
) -> Result<(), HistoricalCandidateScanError> {
    match policy {
        HistoricalCandidateFailurePolicyV1::FailEntireScan => {
            Err(HistoricalCandidateScanError::CandidateFailed {
                candidate_index,
                candidate_identity_sha256,
                stage: failure.stage,
                detail: failure.detail,
            })
        }
        HistoricalCandidateFailurePolicyV1::RetainFailedCandidate => {
            results.push(HistoricalCandidateResultV2 {
                input_ordinal: candidate_index as u64,
                candidate_identity_sha256,
                status: HistoricalCandidateResultStatusV1::Failed,
                signal_identity_sha256: None,
                artifact: None,
                failure: Some(failure),
            });
            Ok(())
        }
    }
}

fn compare_candidate_results(
    left: &HistoricalCandidateResultV2,
    right: &HistoricalCandidateResultV2,
) -> Ordering {
    let left_metrics = left
        .artifact
        .as_ref()
        .expect("only evaluated results are ranked")
        .metrics();
    let right_metrics = right
        .artifact
        .as_ref()
        .expect("only evaluated results are ranked")
        .metrics();
    compare_optional_f64_desc(
        left_metrics.gross_r_expectancy(),
        right_metrics.gross_r_expectancy(),
    )
    .then_with(|| {
        left_metrics
            .gross_r_max_drawdown()
            .total_cmp(&right_metrics.gross_r_max_drawdown())
    })
    .then_with(|| {
        compare_optional_f64_desc(
            left_metrics.gross_r_win_rate(),
            right_metrics.gross_r_win_rate(),
        )
    })
    .then_with(|| {
        compare_optional_f64_desc(
            left_metrics.gross_r_payoff(),
            right_metrics.gross_r_payoff(),
        )
    })
    .then_with(|| right_metrics.trade_count().cmp(&left_metrics.trade_count()))
    .then_with(|| {
        left.candidate_identity_sha256
            .cmp(&right.candidate_identity_sha256)
    })
    .then_with(|| left.input_ordinal.cmp(&right.input_ordinal))
}

fn compare_optional_f64_desc(left: Option<f64>, right: Option<f64>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => right.total_cmp(&left),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn candidate_scan_identity_sha256(
    contract: &HistoricalCandidateScanContractV2,
) -> Result<String, HistoricalCandidateScanError> {
    let bytes = serde_json::to_vec(contract)
        .map_err(|error| invalid_scan_contract(format!("serialize scan identity: {error}")))?;
    Ok(sha256_hex(&bytes))
}

fn invalid_scan_contract(detail: impl Into<String>) -> HistoricalCandidateScanError {
    HistoricalCandidateScanError::InvalidScanContract {
        detail: detail.into(),
    }
}

#[derive(Serialize)]
struct HistoricalResearchEvidenceIdentityMaterialV2<'a> {
    schema_version: u16,
    artifact_class: HistoricalResearchArtifactClassV1,
    promotion_eligibility: HistoricalResearchPromotionEligibilityV1,
    backend: HistoricalResearchBackendV1,
    accounting: HistoricalResearchAccountingV1,
    intrabar_ambiguity: HistoricalResearchIntrabarAmbiguityV1,
    scope: &'a CanonicalSearchArtifactScopeV2,
    execution_contract: &'a HistoricalResearchExecutionContractV1,
}

fn evidence_identity_sha256(
    scope: &CanonicalSearchArtifactScopeV2,
    execution_contract: &HistoricalResearchExecutionContractV1,
) -> Result<String, HistoricalResearchError> {
    let material = HistoricalResearchEvidenceIdentityMaterialV2 {
        schema_version: HISTORICAL_RESEARCH_SCHEMA_VERSION,
        artifact_class: HistoricalResearchArtifactClassV1::ResearchOnly,
        promotion_eligibility: HistoricalResearchPromotionEligibilityV1::NotPromotionEligible,
        backend: HistoricalResearchBackendV1::CpuOnly,
        accounting: HistoricalResearchAccountingV1::GrossReferenceR,
        intrabar_ambiguity: HistoricalResearchIntrabarAmbiguityV1::StopBeforeTarget,
        scope,
        execution_contract,
    };
    let bytes = serde_json::to_vec(&material)
        .map_err(|error| invalid_artifact(format!("serialize evidence identity: {error}")))?;
    Ok(sha256_hex(&bytes))
}

fn signal_values_sha256(signals: &[HistoricalResearchSignalV1]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.historical-research.signal-values.v1\0");
    hasher.update((signals.len() as u64).to_le_bytes());
    for signal in signals {
        hasher.update([match signal {
            HistoricalResearchSignalV1::Flat => 0_u8,
            HistoricalResearchSignalV1::Long => 1_u8,
            HistoricalResearchSignalV1::Short => 2_u8,
        }]);
    }
    format!("{:x}", hasher.finalize())
}

fn distance_values_sha256(distances: &[f64]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"neoethos.historical-research.distance-values.f64-le.v1\0");
    hasher.update((distances.len() as u64).to_le_bytes());
    for distance in distances {
        hasher.update(distance.to_bits().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_semantic_id(label: &str, semantic_id: &str) -> Result<(), HistoricalResearchError> {
    if semantic_id.is_empty()
        || semantic_id.len() > 256
        || semantic_id.trim() != semantic_id
        || semantic_id.chars().any(char::is_control)
    {
        return Err(invalid_contract(format!(
            "{label} semantic id is empty, padded, too long, or contains control characters"
        )));
    }
    Ok(())
}

fn validate_sha256_hex(label: &str, value: &str) -> Result<(), HistoricalResearchError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid_contract(format!(
            "{label} SHA-256 is not 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn invalid_contract(detail: impl Into<String>) -> HistoricalResearchError {
    HistoricalResearchError::InvalidExecutionContract {
        detail: detail.into(),
    }
}

fn invalid_artifact(detail: impl Into<String>) -> HistoricalResearchError {
    HistoricalResearchError::InvalidArtifact {
        detail: detail.into(),
    }
}

fn require_shape(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), HistoricalResearchError> {
    if actual != expected {
        return Err(HistoricalResearchError::ShapeMismatch {
            field,
            expected,
            actual,
        });
    }
    Ok(())
}

fn require_finite_scalar(field: &'static str, value: f64) -> Result<(), HistoricalResearchError> {
    if !value.is_finite() {
        return Err(HistoricalResearchError::NonFinite { field, index: None });
    }
    Ok(())
}

fn require_positive_scalar(field: &'static str, value: f64) -> Result<(), HistoricalResearchError> {
    if value <= 0.0 {
        return Err(HistoricalResearchError::NonPositive { field, index: None });
    }
    Ok(())
}

fn metrics_from_outcomes(outcomes: &[f64]) -> HistoricalResearchMetricsV1 {
    let trade_count = outcomes.len() as u64;
    let gross_r_expectancy =
        (!outcomes.is_empty()).then(|| outcomes.iter().sum::<f64>() / outcomes.len() as f64);
    let wins = outcomes
        .iter()
        .copied()
        .filter(|outcome| *outcome > 0.0)
        .collect::<Vec<_>>();
    let losses = outcomes
        .iter()
        .copied()
        .filter(|outcome| *outcome < 0.0)
        .collect::<Vec<_>>();
    let gross_r_win_rate =
        (!outcomes.is_empty()).then(|| wins.len() as f64 / outcomes.len() as f64);
    let gross_r_payoff = (!wins.is_empty() && !losses.is_empty()).then(|| {
        let average_win = wins.iter().sum::<f64>() / wins.len() as f64;
        let average_loss = losses.iter().map(|loss| loss.abs()).sum::<f64>() / losses.len() as f64;
        average_win / average_loss
    });

    let mut cumulative = 0.0_f64;
    let mut peak = 0.0_f64;
    let mut max_drawdown = 0.0_f64;
    for outcome in outcomes {
        cumulative += outcome;
        peak = peak.max(cumulative);
        max_drawdown = max_drawdown.max(peak - cumulative);
    }

    HistoricalResearchMetricsV1 {
        gross_r_expectancy,
        gross_r_max_drawdown: max_drawdown,
        gross_r_win_rate,
        gross_r_payoff,
        trade_count,
    }
}
