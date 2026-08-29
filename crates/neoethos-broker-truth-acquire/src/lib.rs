//! Fail-closed one-shot broker-truth acquisition preflight.
//!
//! This crate accepts only explicit non-secret paths and bindings. It verifies
//! and pins the exact canonical Search generations, freezes immutable reviewed
//! inputs, and prepares existing evidence-only capture contracts. It does not
//! connect to cTrader, load credentials, publish evidence, validate semantic
//! trust, or authorize an evaluator.

#![forbid(unsafe_code)]

use std::collections::HashSet;
use std::error::Error;
use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::path::{Component, Path, PathBuf};

use clap::{ArgAction, Parser, ValueEnum};
use neoethos_broker_history::broker_truth_capture::{
    BrokerFinancialTruthCaptureRequestV2, ExactConversionLegCaptureRequestV2,
    ExactConversionRouteCaptureRequestV2, ExactQuoteInstrumentV2,
};
use neoethos_broker_truth::{
    BrokerFinancialTruthBindingV1, BrokerTruthAcquisitionArtifactRoleV1,
    BrokerTruthAcquisitionArtifactSourceV1, BrokerTruthAcquisitionArtifactV1,
    BrokerTruthAcquisitionAuthorityManifestV1, BrokerTruthReviewedSynchronizationBindingV1,
    EvidenceWindowV1, ReviewedQuoteReplayRuleIdentityV2,
};
use neoethos_data::core::dataset_generation_lease::DatasetGenerationLease;
use neoethos_data::{
    CTraderEnvironment, CanonicalDatasetIdentity, SelectedDatasetGenerationV1,
    open_exact_dataset_generation,
};
use neoethos_search::{
    CanonicalSearchArtifactScopeV2, CanonicalSearchInputReceiptV2, CanonicalSearchWindowRoleV1,
};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256};

mod acquisition_orchestration_v1;
mod finalist_quote_replay_acquisition_v1;

pub use acquisition_orchestration_v1::{
    BrokerTruthAcquisitionOrchestrationErrorCodeV1, BrokerTruthAcquisitionOrchestrationErrorV1,
    BrokerTruthAcquisitionOutcomeV1, execute_prepared_acquisition_v1,
};
pub use finalist_quote_replay_acquisition_v1::{
    BrokerTruthPromotionEligibilityV1, BrokerTruthSemanticStatusV1,
    FinalistQuoteReplayAcquisitionErrorCodeV1, FinalistQuoteReplayAcquisitionErrorV1,
    FinalistQuoteReplayAcquisitionInputV1, FinalistQuoteReplayAcquisitionOutcomeV1,
    FinalistQuoteReplayAcquisitionRequestV1, FinalistQuoteReplayArtifactClassV1,
    FinalistQuoteReplayRestartPolicyV1, MAX_FINALIST_QUOTE_REPLAY_WINDOW_MS_V1,
    acquire_finalist_quote_replay_v1,
};

const ACQUISITION_SCHEMA_VERSION_V1: u16 = 1;
const MAX_FROZEN_INPUT_BYTES_V1: u64 = 64 * 1024 * 1024;

const RECEIPT_FILE_V1: &str = "canonical-search-input-receipt.json";
const SCOPE_FILE_V1: &str = "canonical-search-artifact-scope.json";
const ROOT_VERIFICATION_FILE_V1: &str = "canonical-root-verification.json";
const WINDOW_BINDING_FILE_V1: &str = "canonical-scope-window-binding.json";
const CAPTURE_PLAN_FILE_V1: &str = "broker-truth-capture-plan.json";
const REVIEW_RECORD_FILE_V1: &str = "quote-replay-review-record.json";
const PROTOCOL_EVIDENCE_FILE_V1: &str = "ctrader-protocol-evidence.json";
const TRUST_ROOT_FILE_V1: &str = "quote-review-trust-root.pub";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrokerTruthAcquisitionPreflightErrorCodeV1 {
    InvalidArguments,
    UnsafePath,
    ArtifactDigestMismatch,
    CanonicalReceiptInvalid,
    CanonicalScopeInvalid,
    CanonicalAuthorityMismatch,
    ExactRootOpenFailed,
    WindowMismatch,
    CapturePlanMismatch,
    SynchronizationMismatch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrokerTruthAcquisitionPreflightErrorV1 {
    code: BrokerTruthAcquisitionPreflightErrorCodeV1,
    detail: &'static str,
}

impl BrokerTruthAcquisitionPreflightErrorV1 {
    pub const fn code(&self) -> BrokerTruthAcquisitionPreflightErrorCodeV1 {
        self.code
    }

    pub const fn detail(&self) -> &'static str {
        self.detail
    }
}

impl fmt::Display for BrokerTruthAcquisitionPreflightErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.detail)
    }
}

impl Error for BrokerTruthAcquisitionPreflightErrorV1 {}

fn preflight_error(
    code: BrokerTruthAcquisitionPreflightErrorCodeV1,
    detail: &'static str,
) -> BrokerTruthAcquisitionPreflightErrorV1 {
    BrokerTruthAcquisitionPreflightErrorV1 { code, detail }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Deserialize)]
#[serde(rename_all = "lowercase")]
enum AcquisitionEnvironmentArgumentV1 {
    Demo,
    Live,
}

impl AcquisitionEnvironmentArgumentV1 {
    const fn canonical(self) -> CTraderEnvironment {
        match self {
            Self::Demo => CTraderEnvironment::Demo,
            Self::Live => CTraderEnvironment::Live,
        }
    }

    const fn endpoint_host(self) -> &'static str {
        match self {
            Self::Demo => "demo.ctraderapi.com",
            Self::Live => "live.ctraderapi.com",
        }
    }
}

#[derive(Parser)]
#[command(name = "broker-truth-acquire", disable_help_subcommand = true)]
pub struct BrokerTruthAcquisitionArgsV1 {
    #[arg(long)]
    data_root: PathBuf,
    #[arg(long)]
    canonical_receipt: PathBuf,
    #[arg(long)]
    canonical_receipt_sha256: String,
    #[arg(long)]
    canonical_scope: PathBuf,
    #[arg(long)]
    canonical_scope_sha256: String,
    #[arg(long)]
    canonical_root_verification: PathBuf,
    #[arg(long)]
    canonical_root_verification_sha256: String,
    #[arg(long)]
    canonical_scope_window_binding: PathBuf,
    #[arg(long)]
    canonical_scope_window_binding_sha256: String,
    #[arg(long)]
    capture_plan: PathBuf,
    #[arg(long)]
    capture_plan_sha256: String,
    #[arg(long)]
    review_record: PathBuf,
    #[arg(long)]
    review_record_sha256: String,
    #[arg(long)]
    protocol_evidence: PathBuf,
    #[arg(long)]
    protocol_evidence_sha256: String,
    #[arg(long)]
    trust_root: PathBuf,
    #[arg(long)]
    trust_root_sha256: String,
    #[arg(long, required = true, action = ArgAction::Append)]
    quote_observations: Vec<PathBuf>,
    #[arg(long, required = true, action = ArgAction::Append)]
    quote_observations_sha256: Vec<String>,
    #[arg(long, required = true, action = ArgAction::Append)]
    reviewed_replay_rules: Vec<PathBuf>,
    #[arg(long, required = true, action = ArgAction::Append)]
    reviewed_replay_rules_sha256: Vec<String>,
    #[arg(long)]
    environment: AcquisitionEnvironmentArgumentV1,
    #[arg(long)]
    account_id: i64,
    #[arg(long)]
    from_ms: i64,
    #[arg(long)]
    to_ms_exclusive: i64,
    #[arg(long)]
    work_parent: PathBuf,
    #[arg(long)]
    store_root: PathBuf,
}

impl BrokerTruthAcquisitionArgsV1 {
    pub fn try_parse_from<I, T>(
        arguments: I,
    ) -> Result<Self, BrokerTruthAcquisitionPreflightErrorV1>
    where
        I: IntoIterator<Item = T>,
        T: Into<OsString> + Clone,
    {
        let parsed = <Self as Parser>::try_parse_from(arguments).map_err(|_| {
            preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments,
                "broker-truth acquisition arguments are incomplete or invalid",
            )
        })?;
        validate_argument_shape(&parsed)?;
        Ok(parsed)
    }
}

fn validate_argument_shape(
    args: &BrokerTruthAcquisitionArgsV1,
) -> Result<(), BrokerTruthAcquisitionPreflightErrorV1> {
    let count = args.quote_observations.len();
    if count == 0
        || args.quote_observations_sha256.len() != count
        || args.reviewed_replay_rules.len() != count
        || args.reviewed_replay_rules_sha256.len() != count
        || args.account_id <= 0
    {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments,
            "broker-truth acquisition arguments do not form exact evidence pairs",
        ));
    }
    Ok(())
}

pub struct PreparedBrokerTruthAcquisitionV1 {
    environment: CTraderEnvironment,
    account_id: i64,
    evidence_window: EvidenceWindowV1,
    capture_request: BrokerFinancialTruthCaptureRequestV2,
    authority_manifest: BrokerTruthAcquisitionAuthorityManifestV1,
    artifact_sources: Vec<BrokerTruthAcquisitionArtifactSourceV1>,
    generation_leases: Vec<DatasetGenerationLease>,
    reviewed_synchronization_count: usize,
    work_parent: PathBuf,
    store_root: PathBuf,
}

impl PreparedBrokerTruthAcquisitionV1 {
    pub const fn environment(&self) -> CTraderEnvironment {
        self.environment
    }

    pub const fn account_id(&self) -> i64 {
        self.account_id
    }

    pub const fn evidence_window(&self) -> EvidenceWindowV1 {
        self.evidence_window
    }

    pub const fn capture_request(&self) -> &BrokerFinancialTruthCaptureRequestV2 {
        &self.capture_request
    }

    pub const fn authority_manifest(&self) -> &BrokerTruthAcquisitionAuthorityManifestV1 {
        &self.authority_manifest
    }

    pub fn artifact_sources(&self) -> &[BrokerTruthAcquisitionArtifactSourceV1] {
        &self.artifact_sources
    }

    pub fn opened_generation_count(&self) -> usize {
        self.generation_leases.len()
    }

    pub const fn reviewed_synchronization_count(&self) -> usize {
        self.reviewed_synchronization_count
    }

    pub fn work_parent(&self) -> &Path {
        &self.work_parent
    }

    pub fn store_root(&self) -> &Path {
        &self.store_root
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalRootVerificationReceiptWireV1 {
    schema_version: u16,
    canonical_search_input_receipt_identity_sha256: String,
    opened_generations: Vec<OpenedGenerationWireV1>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenedGenerationWireV1 {
    source_node_id: String,
    selected_generation: SelectedDatasetGenerationV1,
    manifest_schema_id: String,
    vortex_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalScopeWindowBindingWireV1 {
    schema_version: u16,
    canonical_search_input_receipt_identity_sha256: String,
    canonical_search_artifact_scope_identity_sha256: String,
    role: CanonicalSearchWindowRoleV1,
    row_start: u64,
    row_end: u64,
    timestamp_start_ms: i64,
    timestamp_end_ms: i64,
    evidence_window: EvidenceWindowV1,
    window_policy_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactInstrumentWireV1 {
    symbol_id: i64,
    symbol_name: String,
    base_asset_id: i64,
    base_asset_name: String,
    quote_asset_id: i64,
    quote_asset_name: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactAccountAssetWireV1 {
    asset_id: i64,
    asset_name: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactConversionLegWireV1 {
    from_asset_id: i64,
    from_asset_name: String,
    to_asset_id: i64,
    to_asset_name: String,
    instrument: ExactInstrumentWireV1,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactConversionRouteWireV1 {
    purpose: String,
    from_asset_id: i64,
    from_asset_name: String,
    to_asset_id: i64,
    to_asset_name: String,
    legs: Vec<ExactConversionLegWireV1>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureSynchronizationWireV1 {
    ordinal: u32,
    account_id: i64,
    instrument: ExactInstrumentWireV1,
    window: EvidenceWindowV1,
    quote_observations_sha256: String,
    reviewed_replay_rules_sha256: String,
    review_identity_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BrokerTruthCapturePlanWireV1 {
    schema_version: u16,
    canonical_search_input_receipt_identity_sha256: String,
    canonical_search_artifact_scope_identity_sha256: String,
    canonical_root_verification_sha256: String,
    canonical_scope_window_binding_sha256: String,
    review_record_sha256: String,
    protocol_evidence_sha256: String,
    trust_root_sha256: String,
    environment: AcquisitionEnvironmentArgumentV1,
    server: String,
    account_id: i64,
    window: EvidenceWindowV1,
    primary_instrument: ExactInstrumentWireV1,
    account_asset: ExactAccountAssetWireV1,
    conversion_routes: Vec<ExactConversionRouteWireV1>,
    synchronizations: Vec<CaptureSynchronizationWireV1>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReviewSynchronizationWireV1 {
    ordinal: u32,
    instrument: ExactInstrumentWireV1,
    quote_observations_sha256: String,
    reviewed_replay_rules_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QuoteReplayReviewRecordWireV1 {
    schema_version: u16,
    canonical_search_input_receipt_identity_sha256: String,
    canonical_search_artifact_scope_identity_sha256: String,
    canonical_scope_window_binding_sha256: String,
    trust_root_sha256: String,
    protocol_evidence_sha256: String,
    window_policy_id: String,
    environment: AcquisitionEnvironmentArgumentV1,
    server: String,
    account_id: i64,
    window: EvidenceWindowV1,
    synchronizations: Vec<ReviewSynchronizationWireV1>,
}

struct FrozenArtifactV1 {
    source_path: PathBuf,
    bytes: Vec<u8>,
    sha256: String,
}

struct FrozenInputsV1 {
    receipt: FrozenArtifactV1,
    scope: FrozenArtifactV1,
    root_verification: FrozenArtifactV1,
    window_binding: FrozenArtifactV1,
    capture_plan: FrozenArtifactV1,
    review_record: FrozenArtifactV1,
    protocol_evidence: FrozenArtifactV1,
    trust_root: FrozenArtifactV1,
    quote_observations: Vec<FrozenArtifactV1>,
    reviewed_replay_rules: Vec<FrozenArtifactV1>,
}

fn validate_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn path_is_lexically_normal_absolute(path: &Path) -> bool {
    path.is_absolute()
        && path.components().all(|component| {
            matches!(
                component,
                Component::Prefix(_) | Component::RootDir | Component::Normal(_)
            )
        })
}

fn metadata_is_link_or_reparse(metadata: &fs::Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;

        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0400;
        metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
    }
    #[cfg(not(windows))]
    {
        false
    }
}

fn validate_existing_path_chain(path: &Path) -> Result<(), BrokerTruthAcquisitionPreflightErrorV1> {
    if !path_is_lexically_normal_absolute(path) {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath,
            "broker-truth acquisition requires normalized absolute paths",
        ));
    }
    let mut cursor = Some(path);
    while let Some(entry) = cursor {
        let metadata = fs::symlink_metadata(entry).map_err(|_| {
            preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath,
                "broker-truth acquisition path cannot be inspected",
            )
        })?;
        if metadata_is_link_or_reparse(&metadata) {
            return Err(preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath,
                "broker-truth acquisition path contains a link or reparse point",
            ));
        }
        cursor = entry.parent();
    }
    Ok(())
}

fn validate_directory_path(path: &Path) -> Result<(), BrokerTruthAcquisitionPreflightErrorV1> {
    validate_existing_path_chain(path)?;
    let metadata = fs::symlink_metadata(path).map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath,
            "broker-truth acquisition directory cannot be inspected",
        )
    })?;
    if !metadata.is_dir() || metadata_is_link_or_reparse(&metadata) {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath,
            "broker-truth acquisition directory is not a regular directory",
        ));
    }
    Ok(())
}

fn validate_explicit_roots(
    data_root: &Path,
    work_parent: &Path,
    store_root: &Path,
) -> Result<(), BrokerTruthAcquisitionPreflightErrorV1> {
    for path in [data_root, work_parent, store_root] {
        validate_directory_path(path)?;
    }
    for (left, right) in [
        (data_root, work_parent),
        (data_root, store_root),
        (work_parent, store_root),
    ] {
        if left == right || left.starts_with(right) || right.starts_with(left) {
            return Err(preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath,
                "broker-truth acquisition roots must be explicit and disjoint",
            ));
        }
    }
    Ok(())
}

fn read_frozen_artifact(
    path: &Path,
    expected_sha256: &str,
) -> Result<FrozenArtifactV1, BrokerTruthAcquisitionPreflightErrorV1> {
    if !validate_sha256(expected_sha256) {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::ArtifactDigestMismatch,
            "frozen acquisition artifact SHA-256 is not canonical",
        ));
    }
    validate_existing_path_chain(path)?;
    let before = fs::symlink_metadata(path).map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath,
            "frozen acquisition artifact cannot be inspected",
        )
    })?;
    if !before.is_file()
        || metadata_is_link_or_reparse(&before)
        || before.len() == 0
        || before.len() > MAX_FROZEN_INPUT_BYTES_V1
    {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath,
            "frozen acquisition artifact is not a bounded regular file",
        ));
    }
    let bytes = fs::read(path).map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath,
            "frozen acquisition artifact cannot be read",
        )
    })?;
    let after = fs::symlink_metadata(path).map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath,
            "frozen acquisition artifact cannot be re-inspected",
        )
    })?;
    if metadata_is_link_or_reparse(&after)
        || after.len() != before.len()
        || u64::try_from(bytes.len()).ok() != Some(before.len())
    {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath,
            "frozen acquisition artifact changed while being read",
        ));
    }
    let actual_sha256 = sha256_bytes(&bytes);
    if actual_sha256 != expected_sha256 {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::ArtifactDigestMismatch,
            "frozen acquisition artifact digest does not match its explicit argument",
        ));
    }
    Ok(FrozenArtifactV1 {
        source_path: path.to_path_buf(),
        bytes,
        sha256: actual_sha256,
    })
}

fn all_artifact_paths(args: &BrokerTruthAcquisitionArgsV1) -> Vec<&Path> {
    let mut paths = vec![
        args.canonical_receipt.as_path(),
        args.canonical_scope.as_path(),
        args.canonical_root_verification.as_path(),
        args.canonical_scope_window_binding.as_path(),
        args.capture_plan.as_path(),
        args.review_record.as_path(),
        args.protocol_evidence.as_path(),
        args.trust_root.as_path(),
    ];
    paths.extend(args.quote_observations.iter().map(PathBuf::as_path));
    paths.extend(args.reviewed_replay_rules.iter().map(PathBuf::as_path));
    paths
}

fn validate_artifact_paths(
    args: &BrokerTruthAcquisitionArgsV1,
) -> Result<(), BrokerTruthAcquisitionPreflightErrorV1> {
    let paths = all_artifact_paths(args);
    let mut unique = HashSet::with_capacity(paths.len());
    for path in paths {
        if !unique.insert(path.to_path_buf())
            || path.starts_with(&args.data_root)
            || path.starts_with(&args.work_parent)
            || path.starts_with(&args.store_root)
        {
            return Err(preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath,
                "frozen acquisition artifact paths must be distinct from writable roots",
            ));
        }
    }
    Ok(())
}

fn freeze_inputs(
    args: &BrokerTruthAcquisitionArgsV1,
) -> Result<FrozenInputsV1, BrokerTruthAcquisitionPreflightErrorV1> {
    validate_artifact_paths(args)?;
    let quote_observations = args
        .quote_observations
        .iter()
        .zip(&args.quote_observations_sha256)
        .map(|(path, digest)| read_frozen_artifact(path, digest))
        .collect::<Result<Vec<_>, _>>()?;
    let reviewed_replay_rules = args
        .reviewed_replay_rules
        .iter()
        .zip(&args.reviewed_replay_rules_sha256)
        .map(|(path, digest)| read_frozen_artifact(path, digest))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(FrozenInputsV1 {
        receipt: read_frozen_artifact(&args.canonical_receipt, &args.canonical_receipt_sha256)?,
        scope: read_frozen_artifact(&args.canonical_scope, &args.canonical_scope_sha256)?,
        root_verification: read_frozen_artifact(
            &args.canonical_root_verification,
            &args.canonical_root_verification_sha256,
        )?,
        window_binding: read_frozen_artifact(
            &args.canonical_scope_window_binding,
            &args.canonical_scope_window_binding_sha256,
        )?,
        capture_plan: read_frozen_artifact(&args.capture_plan, &args.capture_plan_sha256)?,
        review_record: read_frozen_artifact(&args.review_record, &args.review_record_sha256)?,
        protocol_evidence: read_frozen_artifact(
            &args.protocol_evidence,
            &args.protocol_evidence_sha256,
        )?,
        trust_root: read_frozen_artifact(&args.trust_root, &args.trust_root_sha256)?,
        quote_observations,
        reviewed_replay_rules,
    })
}

fn decode_strict_json<T: DeserializeOwned>(
    artifact: &FrozenArtifactV1,
    code: BrokerTruthAcquisitionPreflightErrorCodeV1,
    detail: &'static str,
) -> Result<T, BrokerTruthAcquisitionPreflightErrorV1> {
    serde_json::from_slice(&artifact.bytes).map_err(|_| preflight_error(code, detail))
}

fn exact_window(
    window: EvidenceWindowV1,
) -> Result<EvidenceWindowV1, BrokerTruthAcquisitionPreflightErrorV1> {
    EvidenceWindowV1::new(
        window.from_unix_ms_inclusive(),
        window.to_unix_ms_exclusive(),
    )
    .map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::WindowMismatch,
            "broker-truth acquisition window is not a valid half-open interval",
        )
    })
}

fn validate_window_policy_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value == value.trim()
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_' | b'.')
        })
}

fn validate_scope_window_binding(
    binding: &CanonicalScopeWindowBindingWireV1,
    receipt_identity_sha256: &str,
    scope_identity_sha256: &str,
    scope: &CanonicalSearchArtifactScopeV2,
    requested_window: EvidenceWindowV1,
) -> Result<(), BrokerTruthAcquisitionPreflightErrorV1> {
    let scope_window = scope.evaluated_window();
    if binding.schema_version != ACQUISITION_SCHEMA_VERSION_V1
        || binding.canonical_search_input_receipt_identity_sha256 != receipt_identity_sha256
        || binding.canonical_search_artifact_scope_identity_sha256 != scope_identity_sha256
    {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalAuthorityMismatch,
            "scope-window binding does not name the exact canonical receipt and scope",
        ));
    }
    if binding.role != scope_window.role()
        || binding.row_start != scope_window.row_start()
        || binding.row_end != scope_window.row_end()
        || binding.timestamp_start_ms != scope_window.timestamp_start_ms()
        || binding.timestamp_end_ms != scope_window.timestamp_end_ms()
        || exact_window(binding.evidence_window)? != binding.evidence_window
        || binding.evidence_window != requested_window
        || binding.evidence_window.from_unix_ms_inclusive() != binding.timestamp_start_ms
        || binding.evidence_window.to_unix_ms_exclusive() <= binding.timestamp_end_ms
        || !validate_window_policy_id(&binding.window_policy_id)
    {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::WindowMismatch,
            "scope-window binding does not exactly bind the reviewed half-open window",
        ));
    }
    Ok(())
}

fn open_exact_receipt_generations(
    data_root: &Path,
    receipt: &CanonicalSearchInputReceiptV2,
    receipt_identity_sha256: &str,
    verification: &CanonicalRootVerificationReceiptWireV1,
) -> Result<Vec<DatasetGenerationLease>, BrokerTruthAcquisitionPreflightErrorV1> {
    if verification.schema_version != ACQUISITION_SCHEMA_VERSION_V1
        || verification.canonical_search_input_receipt_identity_sha256 != receipt_identity_sha256
        || verification.opened_generations.len() != receipt.source_bindings().len()
    {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalAuthorityMismatch,
            "root-verification receipt does not exactly cover the canonical receipt",
        ));
    }

    let mut leases = Vec::with_capacity(receipt.source_bindings().len());
    for (binding, opened) in receipt
        .source_bindings()
        .iter()
        .zip(&verification.opened_generations)
    {
        let identity = CanonicalDatasetIdentity::from_path_component(binding.dataset_identity())
            .map_err(|_| {
                preflight_error(
                    BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalReceiptInvalid,
                    "canonical receipt contains an invalid dataset identity",
                )
            })?;
        let selected = SelectedDatasetGenerationV1::new(
            identity,
            binding.generation_id(),
            binding.manifest_sha256(),
        )
        .map_err(|_| {
            preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalReceiptInvalid,
                "canonical receipt contains an invalid selected generation",
            )
        })?;
        if opened.source_node_id != binding.source_node_id()
            || opened.selected_generation != selected
            || opened.manifest_schema_id != binding.manifest_schema_id()
            || opened.vortex_sha256 != binding.vortex_sha256()
        {
            return Err(preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalAuthorityMismatch,
                "root-verification entry differs from its exact receipt source binding",
            ));
        }

        let (manifest, lease) =
            open_exact_dataset_generation(data_root, &selected).map_err(|_| {
                preflight_error(
                    BrokerTruthAcquisitionPreflightErrorCodeV1::ExactRootOpenFailed,
                    "exact receipt-selected dataset generation could not be opened and pinned",
                )
            })?;
        if manifest.identity() != selected.identity()
            || manifest.generation_id() != selected.generation_id()
            || manifest.manifest_binding_sha256() != selected.manifest_binding_sha256()
            || manifest.schema_id() != opened.manifest_schema_id
            || manifest.vortex_sha256() != opened.vortex_sha256
        {
            return Err(preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::ExactRootOpenFailed,
                "opened dataset generation differs from the exact root-verification receipt",
            ));
        }
        let timestamp_range = manifest.timestamp_range();
        if binding.segments().iter().any(|segment| {
            segment.row_end() > manifest.row_count()
                || segment.timestamp_start_ms() < timestamp_range.start_ms()
                || segment.timestamp_end_ms() > timestamp_range.end_ms()
        }) {
            return Err(preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::ExactRootOpenFailed,
                "canonical receipt segment lies outside its exact opened generation",
            ));
        }
        leases.push(lease);
    }
    Ok(leases)
}

fn exact_capture_instrument(
    instrument: &ExactInstrumentWireV1,
) -> Result<ExactQuoteInstrumentV2, BrokerTruthAcquisitionPreflightErrorV1> {
    ExactQuoteInstrumentV2::new(
        instrument.symbol_id,
        instrument.symbol_name.clone(),
        instrument.base_asset_id,
        instrument.base_asset_name.clone(),
        instrument.quote_asset_id,
        instrument.quote_asset_name.clone(),
    )
    .map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CapturePlanMismatch,
            "capture plan contains an invalid exact quoted instrument",
        )
    })
}

fn exact_capture_routes(
    routes: &[ExactConversionRouteWireV1],
) -> Result<Vec<ExactConversionRouteCaptureRequestV2>, BrokerTruthAcquisitionPreflightErrorV1> {
    routes
        .iter()
        .map(|route| {
            let legs = route
                .legs
                .iter()
                .map(|leg| {
                    let instrument = exact_capture_instrument(&leg.instrument)?;
                    ExactConversionLegCaptureRequestV2::new(
                        leg.from_asset_id,
                        leg.from_asset_name.clone(),
                        leg.to_asset_id,
                        leg.to_asset_name.clone(),
                        instrument,
                    )
                    .map_err(|_| {
                        preflight_error(
                            BrokerTruthAcquisitionPreflightErrorCodeV1::CapturePlanMismatch,
                            "capture plan contains an invalid exact conversion leg",
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            ExactConversionRouteCaptureRequestV2::new(
                route.purpose.clone(),
                route.from_asset_id,
                route.from_asset_name.clone(),
                route.to_asset_id,
                route.to_asset_name.clone(),
                legs,
            )
            .map_err(|_| {
                preflight_error(
                    BrokerTruthAcquisitionPreflightErrorCodeV1::CapturePlanMismatch,
                    "capture plan contains an invalid exact conversion route",
                )
            })
        })
        .collect()
}

fn validate_plan_and_review_headers(
    args: &BrokerTruthAcquisitionArgsV1,
    frozen: &FrozenInputsV1,
    plan: &BrokerTruthCapturePlanWireV1,
    review: &QuoteReplayReviewRecordWireV1,
    window_binding: &CanonicalScopeWindowBindingWireV1,
    receipt_identity_sha256: &str,
    scope_identity_sha256: &str,
    requested_window: EvidenceWindowV1,
) -> Result<(), BrokerTruthAcquisitionPreflightErrorV1> {
    if plan.schema_version != ACQUISITION_SCHEMA_VERSION_V1
        || plan.canonical_search_input_receipt_identity_sha256 != receipt_identity_sha256
        || plan.canonical_search_artifact_scope_identity_sha256 != scope_identity_sha256
        || plan.canonical_root_verification_sha256 != frozen.root_verification.sha256
        || plan.canonical_scope_window_binding_sha256 != frozen.window_binding.sha256
        || plan.review_record_sha256 != frozen.review_record.sha256
        || plan.protocol_evidence_sha256 != frozen.protocol_evidence.sha256
        || plan.trust_root_sha256 != frozen.trust_root.sha256
        || plan.environment != args.environment
        || plan.server != args.environment.endpoint_host()
        || plan.account_id != args.account_id
    {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CapturePlanMismatch,
            "capture plan does not exactly bind the explicit acquisition authority",
        ));
    }
    if exact_window(plan.window)? != plan.window || plan.window != requested_window {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::WindowMismatch,
            "capture plan window differs from the explicit half-open window",
        ));
    }
    if review.schema_version != ACQUISITION_SCHEMA_VERSION_V1
        || review.canonical_search_input_receipt_identity_sha256 != receipt_identity_sha256
        || review.canonical_search_artifact_scope_identity_sha256 != scope_identity_sha256
        || review.canonical_scope_window_binding_sha256 != frozen.window_binding.sha256
        || review.trust_root_sha256 != frozen.trust_root.sha256
        || review.protocol_evidence_sha256 != frozen.protocol_evidence.sha256
        || review.window_policy_id != window_binding.window_policy_id
        || review.environment != args.environment
        || review.server != args.environment.endpoint_host()
        || review.account_id != args.account_id
    {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalAuthorityMismatch,
            "review record does not exactly bind the frozen acquisition authority",
        ));
    }
    if exact_window(review.window)? != review.window || review.window != requested_window {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::WindowMismatch,
            "review record window differs from the explicit half-open window",
        ));
    }
    Ok(())
}

fn build_capture_request(
    args: &BrokerTruthAcquisitionArgsV1,
    anchor: &CanonicalDatasetIdentity,
    receipt_identity_sha256: &str,
    requested_window: EvidenceWindowV1,
    plan: &BrokerTruthCapturePlanWireV1,
) -> Result<BrokerFinancialTruthCaptureRequestV2, BrokerTruthAcquisitionPreflightErrorV1> {
    let exact_anchor = CanonicalDatasetIdentity::ctrader(
        args.environment.canonical(),
        args.environment.endpoint_host(),
        args.account_id,
        plan.primary_instrument.symbol_id,
        plan.primary_instrument.symbol_name.clone(),
        anchor.timeframe(),
        anchor.bar_timestamp_convention(),
    )
    .map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CapturePlanMismatch,
            "capture plan cannot form an exact cTrader anchor identity",
        )
    })?;
    if &exact_anchor != anchor {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CapturePlanMismatch,
            "capture plan environment, account, server, or primary instrument differs from the canonical anchor",
        ));
    }

    let primary = exact_capture_instrument(&plan.primary_instrument)?;
    let binding = BrokerFinancialTruthBindingV1::new(
        anchor,
        receipt_identity_sha256,
        requested_window,
        plan.primary_instrument.base_asset_id,
        plan.primary_instrument.base_asset_name.clone(),
        plan.primary_instrument.quote_asset_id,
        plan.primary_instrument.quote_asset_name.clone(),
        plan.account_asset.asset_id,
        plan.account_asset.asset_name.clone(),
    )
    .map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CapturePlanMismatch,
            "capture plan cannot form the exact broker-financial binding",
        )
    })?;
    let conversion_routes = exact_capture_routes(&plan.conversion_routes)?;
    BrokerFinancialTruthCaptureRequestV2::new(args.account_id, binding, primary, conversion_routes)
        .map_err(|_| {
            preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::CapturePlanMismatch,
                "capture plan cannot form the exact broker-financial capture request",
            )
        })
}

fn planned_instruments(plan: &BrokerTruthCapturePlanWireV1) -> Vec<&ExactInstrumentWireV1> {
    std::iter::once(&plan.primary_instrument)
        .chain(
            plan.conversion_routes
                .iter()
                .flat_map(|route| route.legs.iter().map(|leg| &leg.instrument)),
        )
        .collect()
}

fn build_reviewed_synchronizations(
    args: &BrokerTruthAcquisitionArgsV1,
    frozen: &FrozenInputsV1,
    plan: &BrokerTruthCapturePlanWireV1,
    review: &QuoteReplayReviewRecordWireV1,
    requested_window: EvidenceWindowV1,
) -> Result<Vec<BrokerTruthReviewedSynchronizationBindingV1>, BrokerTruthAcquisitionPreflightErrorV1>
{
    let instruments = planned_instruments(plan);
    let count = instruments.len();
    if count == 0
        || plan.synchronizations.len() != count
        || review.synchronizations.len() != count
        || frozen.quote_observations.len() != count
        || frozen.reviewed_replay_rules.len() != count
    {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::SynchronizationMismatch,
            "capture plan, review, and explicit evidence do not contain the same synchronization set",
        ));
    }

    let mut bindings = Vec::with_capacity(count);
    for (index, instrument) in instruments.into_iter().enumerate() {
        let ordinal = u32::try_from(index).map_err(|_| {
            preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::SynchronizationMismatch,
                "reviewed synchronization count exceeds its versioned ordinal range",
            )
        })?;
        let planned = &plan.synchronizations[index];
        let reviewed = &review.synchronizations[index];
        let observations = &frozen.quote_observations[index];
        let rules = &frozen.reviewed_replay_rules[index];
        if planned.ordinal != ordinal
            || reviewed.ordinal != ordinal
            || planned.account_id != args.account_id
            || &planned.instrument != instrument
            || &reviewed.instrument != instrument
            || exact_window(planned.window)? != planned.window
            || planned.window != requested_window
            || planned.quote_observations_sha256 != observations.sha256
            || reviewed.quote_observations_sha256 != observations.sha256
            || planned.reviewed_replay_rules_sha256 != rules.sha256
            || reviewed.reviewed_replay_rules_sha256 != rules.sha256
        {
            return Err(preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::SynchronizationMismatch,
                "reviewed synchronization order or exact account/instrument/window evidence differs",
            ));
        }
        let review_identity = ReviewedQuoteReplayRuleIdentityV2::new(
            frozen.review_record.sha256.clone(),
            frozen.protocol_evidence.sha256.clone(),
            observations.sha256.clone(),
        )
        .map_err(|_| {
            preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::SynchronizationMismatch,
                "reviewed synchronization identity inputs are invalid",
            )
        })?;
        if planned.review_identity_sha256 != review_identity.identity_sha256() {
            return Err(preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::SynchronizationMismatch,
                "capture plan review identity differs from the exact frozen review evidence",
            ));
        }
        bindings.push(
            BrokerTruthReviewedSynchronizationBindingV1::new(
                ordinal,
                args.account_id,
                instrument.symbol_id,
                requested_window,
                review_identity,
                rules.sha256.clone(),
            )
            .map_err(|_| {
                preflight_error(
                    BrokerTruthAcquisitionPreflightErrorCodeV1::SynchronizationMismatch,
                    "reviewed synchronization cannot form a strict authority binding",
                )
            })?,
        );
    }
    Ok(bindings)
}

fn artifact_and_source(
    role: BrokerTruthAcquisitionArtifactRoleV1,
    relative_path: String,
    frozen: &FrozenArtifactV1,
) -> Result<
    (
        BrokerTruthAcquisitionArtifactV1,
        BrokerTruthAcquisitionArtifactSourceV1,
    ),
    BrokerTruthAcquisitionPreflightErrorV1,
> {
    let byte_len = u64::try_from(frozen.bytes.len()).map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalAuthorityMismatch,
            "frozen acquisition artifact length exceeds the authority contract",
        )
    })?;
    let artifact = BrokerTruthAcquisitionArtifactV1::new(
        role,
        relative_path.clone(),
        frozen.sha256.clone(),
        byte_len,
    )
    .map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalAuthorityMismatch,
            "frozen acquisition artifact cannot form an authority descriptor",
        )
    })?;
    let source =
        BrokerTruthAcquisitionArtifactSourceV1::new(relative_path, frozen.source_path.clone())
            .map_err(|_| {
                preflight_error(
                    BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalAuthorityMismatch,
                    "frozen acquisition artifact cannot form an immutable source binding",
                )
            })?;
    Ok((artifact, source))
}

fn build_artifacts(
    frozen: &FrozenInputsV1,
) -> Result<
    (
        Vec<BrokerTruthAcquisitionArtifactV1>,
        Vec<BrokerTruthAcquisitionArtifactSourceV1>,
    ),
    BrokerTruthAcquisitionPreflightErrorV1,
> {
    let fixed = [
        (
            BrokerTruthAcquisitionArtifactRoleV1::CanonicalSearchInputReceipt,
            RECEIPT_FILE_V1,
            &frozen.receipt,
        ),
        (
            BrokerTruthAcquisitionArtifactRoleV1::CanonicalSearchArtifactScope,
            SCOPE_FILE_V1,
            &frozen.scope,
        ),
        (
            BrokerTruthAcquisitionArtifactRoleV1::CanonicalRootVerificationReceipt,
            ROOT_VERIFICATION_FILE_V1,
            &frozen.root_verification,
        ),
        (
            BrokerTruthAcquisitionArtifactRoleV1::CanonicalScopeWindowBinding,
            WINDOW_BINDING_FILE_V1,
            &frozen.window_binding,
        ),
        (
            BrokerTruthAcquisitionArtifactRoleV1::CapturePlan,
            CAPTURE_PLAN_FILE_V1,
            &frozen.capture_plan,
        ),
        (
            BrokerTruthAcquisitionArtifactRoleV1::ReviewRecord,
            REVIEW_RECORD_FILE_V1,
            &frozen.review_record,
        ),
        (
            BrokerTruthAcquisitionArtifactRoleV1::ProtocolEvidence,
            PROTOCOL_EVIDENCE_FILE_V1,
            &frozen.protocol_evidence,
        ),
        (
            BrokerTruthAcquisitionArtifactRoleV1::TrustRoot,
            TRUST_ROOT_FILE_V1,
            &frozen.trust_root,
        ),
    ];
    let mut artifacts = Vec::with_capacity(8 + 2 * frozen.quote_observations.len());
    let mut sources = Vec::with_capacity(artifacts.capacity());
    for (role, relative_path, artifact) in fixed {
        let (descriptor, source) = artifact_and_source(role, relative_path.to_owned(), artifact)?;
        artifacts.push(descriptor);
        sources.push(source);
    }
    for (index, (observations, rules)) in frozen
        .quote_observations
        .iter()
        .zip(&frozen.reviewed_replay_rules)
        .enumerate()
    {
        let ordinal = u32::try_from(index).map_err(|_| {
            preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalAuthorityMismatch,
                "acquisition artifact count exceeds its versioned ordinal range",
            )
        })?;
        let (descriptor, source) = artifact_and_source(
            BrokerTruthAcquisitionArtifactRoleV1::QuoteSessionObservations { ordinal },
            format!("quote-session-observations-{ordinal:03}.vortex"),
            observations,
        )?;
        artifacts.push(descriptor);
        sources.push(source);
        let (descriptor, source) = artifact_and_source(
            BrokerTruthAcquisitionArtifactRoleV1::ReviewedQuoteReplayRules { ordinal },
            format!("reviewed-quote-replay-rules-{ordinal:03}.vortex"),
            rules,
        )?;
        artifacts.push(descriptor);
        sources.push(source);
    }
    Ok((artifacts, sources))
}

/// Strictly freezes and binds the non-secret acquisition inputs without
/// connecting to a broker, publishing evidence, or creating semantic authority.
pub fn prepare_acquisition_v1(
    args: BrokerTruthAcquisitionArgsV1,
) -> Result<PreparedBrokerTruthAcquisitionV1, BrokerTruthAcquisitionPreflightErrorV1> {
    validate_argument_shape(&args)?;
    validate_explicit_roots(&args.data_root, &args.work_parent, &args.store_root)?;
    let requested_window =
        EvidenceWindowV1::new(args.from_ms, args.to_ms_exclusive).map_err(|_| {
            preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::WindowMismatch,
                "explicit acquisition window is not a valid half-open interval",
            )
        })?;
    let frozen = freeze_inputs(&args)?;

    let receipt =
        CanonicalSearchInputReceiptV2::from_json_bytes(&frozen.receipt.bytes).map_err(|_| {
            preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalReceiptInvalid,
                "canonical Search input receipt is invalid",
            )
        })?;
    let anchor = receipt.validate().map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalReceiptInvalid,
            "canonical Search input receipt cannot establish an exact anchor",
        )
    })?;
    let receipt_identity_sha256 = receipt.identity_sha256().map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalReceiptInvalid,
            "canonical Search input receipt identity cannot be computed",
        )
    })?;

    let scope =
        CanonicalSearchArtifactScopeV2::from_json_bytes(&frozen.scope.bytes).map_err(|_| {
            preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalScopeInvalid,
                "canonical Search artifact scope is invalid",
            )
        })?;
    scope.validate_against_receipt(&receipt).map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalScopeInvalid,
            "canonical Search artifact scope differs from the explicit receipt",
        )
    })?;
    let scope_identity_sha256 = scope.identity_sha256().map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalScopeInvalid,
            "canonical Search artifact scope identity cannot be computed",
        )
    })?;

    let root_verification: CanonicalRootVerificationReceiptWireV1 = decode_strict_json(
        &frozen.root_verification,
        BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalAuthorityMismatch,
        "canonical root-verification receipt is invalid",
    )?;
    let window_binding: CanonicalScopeWindowBindingWireV1 = decode_strict_json(
        &frozen.window_binding,
        BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalAuthorityMismatch,
        "canonical scope-window binding is invalid",
    )?;
    validate_scope_window_binding(
        &window_binding,
        &receipt_identity_sha256,
        &scope_identity_sha256,
        &scope,
        requested_window,
    )?;

    let plan: BrokerTruthCapturePlanWireV1 = decode_strict_json(
        &frozen.capture_plan,
        BrokerTruthAcquisitionPreflightErrorCodeV1::CapturePlanMismatch,
        "broker-truth capture plan is invalid",
    )?;
    let review: QuoteReplayReviewRecordWireV1 = decode_strict_json(
        &frozen.review_record,
        BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalAuthorityMismatch,
        "quote-replay review record is invalid",
    )?;
    validate_plan_and_review_headers(
        &args,
        &frozen,
        &plan,
        &review,
        &window_binding,
        &receipt_identity_sha256,
        &scope_identity_sha256,
        requested_window,
    )?;

    let generation_leases = open_exact_receipt_generations(
        &args.data_root,
        &receipt,
        &receipt_identity_sha256,
        &root_verification,
    )?;
    let capture_request = build_capture_request(
        &args,
        &anchor,
        &receipt_identity_sha256,
        requested_window,
        &plan,
    )?;
    let reviewed_synchronizations =
        build_reviewed_synchronizations(&args, &frozen, &plan, &review, requested_window)?;
    let reviewed_synchronization_count = reviewed_synchronizations.len();
    let (artifacts, artifact_sources) = build_artifacts(&frozen)?;
    let authority_manifest = BrokerTruthAcquisitionAuthorityManifestV1::new(
        receipt_identity_sha256,
        scope_identity_sha256,
        frozen.root_verification.sha256,
        frozen.window_binding.sha256,
        frozen.capture_plan.sha256,
        frozen.trust_root.sha256,
        artifacts,
        reviewed_synchronizations,
    )
    .map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalAuthorityMismatch,
            "frozen acquisition inputs cannot form an evidence-only authority manifest",
        )
    })?;

    Ok(PreparedBrokerTruthAcquisitionV1 {
        environment: args.environment.canonical(),
        account_id: args.account_id,
        evidence_window: requested_window,
        capture_request,
        authority_manifest,
        artifact_sources,
        generation_leases,
        reviewed_synchronization_count,
        work_parent: args.work_parent,
        store_root: args.store_root,
    })
}

#[cfg(test)]
mod acquisition_orchestration_v1_tests;
