//! Agent-reviewed capture-plan/review-record input generation.
//!
//! Generates the reviewed input bundle for the fail-closed broker-truth
//! acquisition pipeline from real, operator-supplied canonical bytes:
//!
//! - broker-truth-capture-plan.json + .sha256 sidecar
//! - quote-replay-review-record.json + .sha256 sidecar
//! - ctrader-protocol-evidence.json + .sha256 sidecar
//! - quote-review-trust-root.pub + .sha256 sidecar
//! - derived canonical-root-verification.json + .sha256 when the operator
//!   does not supply a real one
//! - derived canonical-scope-window-binding.json + .sha256 when the
//!   operator does not supply a real one
//!
//! Every canonical digest is derived from the real input bytes: the receipt
//! and scope are strictly decoded and identity-hashed with the canonical
//! Search types, the root-verification receipt is derived from the exact
//! receipt source bindings, and the scope-window binding is derived from the
//! exact evaluated scope window plus the explicit half-open acquisition
//! window. Nothing is invented. The agent acts as the reviewer and that
//! identity is recorded in the trust root; this generator never connects to a
//! broker, never mints a permit, and never produces a capability.

use std::collections::HashSet;
use std::collections::hash_map::RandomState;
use std::ffi::OsString;
use std::fs;
use std::hash::{BuildHasher, Hasher};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{ArgAction, Parser};
use neoethos_broker_truth::{EvidenceWindowV1, ReviewedQuoteReplayRuleIdentityV2};
use neoethos_data::{CanonicalDatasetIdentity, CanonicalDatasetScope, SelectedDatasetGenerationV1};
use neoethos_search::{
    CanonicalSearchArtifactScopeV2, CanonicalSearchInputReceiptV2,
    CanonicalSearchSourceBindingReceiptV1,
};
use serde_json::{Value, json};

use super::{
    ACQUISITION_SCHEMA_VERSION_V1, AcquisitionEnvironmentArgumentV1,
    BrokerTruthAcquisitionPreflightErrorCodeV1, BrokerTruthAcquisitionPreflightErrorV1,
    CAPTURE_PLAN_FILE_V1, CanonicalRootVerificationReceiptWireV1,
    CanonicalScopeWindowBindingWireV1, ExactInstrumentWireV1, FrozenArtifactV1,
    PROTOCOL_EVIDENCE_FILE_V1, REVIEW_RECORD_FILE_V1, ROOT_VERIFICATION_FILE_V1,
    TRUST_ROOT_FILE_V1, WINDOW_BINDING_FILE_V1, decode_strict_json, preflight_error,
    read_frozen_artifact, sha256_bytes, validate_directory_path, validate_scope_window_binding,
    validate_sha256, validate_window_policy_id,
};

/// Exact fixed agent-reviewer identity recorded in the generated trust root.
///
/// The bundle is agent-reviewed: the autonomous agent acts on the operator
/// directive and this reduced-independence fact is recorded explicitly. It is
/// never presented as external review.
pub const AGENT_REVIEWER_IDENTITY_V1: &str =
    "neoethos.agent-reviewer.v1: autonomous agent acting on operator directive 2026-10-23";

/// File name of the random reviewer nonce stored in the explicit work parent.
pub const AGENT_REVIEWER_NONCE_FILE_V1: &str = "agent-reviewer-nonce.v1.txt";

const PROTOCOL_IDENTITY_V1: &str = "ctrader-open-api";
const REVIEWED_DATE_V1: &str = "2026-10-23";
const SETTLEMENT_PURPOSE_V1: &str = "primary_pnl_settlement";

/// Explicit argument set for generating the agent-reviewed input bundle.
/// Every input is explicit: output dir, work parent, canonical receipt/scope,
/// exact half-open window, exact assets, and settlement legs. Nothing is
/// discovered, defaulted, or read from ambient configuration. Each settlement
/// leg is ten pipe-separated fields:
/// fromId|fromName|toId|toName|symbolId|symbolName|baseId|baseName|quoteId|quoteName.
#[derive(Parser)]
#[command(
    name = "broker-truth-acquire",
    about = "Generate the agent-reviewed broker-truth acquisition input bundle",
    disable_help_subcommand = true
)]
pub struct ReviewedInputGenerationArgsV1 {
    #[arg(long, value_name = "OUT_DIR")]
    generate_reviewed_inputs: PathBuf,
    #[arg(long, value_name = "WORK_PARENT")]
    work_parent: PathBuf,
    #[command(flatten)]
    canonical: CanonicalReviewedInputArgumentsV1,
    #[command(flatten)]
    evidence: ReviewedEvidenceArgumentsV1,
    #[command(flatten)]
    window: ReviewedWindowArgumentsV1,
    #[command(flatten)]
    assets: ExactAssetArgumentsV1,
    #[arg(long, action = ArgAction::Append)]
    settlement_leg: Vec<String>,
}

impl ReviewedInputGenerationArgsV1 {
    pub fn try_parse_from(
        arguments: impl IntoIterator<Item = impl Into<OsString> + Clone>,
    ) -> Result<Self, BrokerTruthAcquisitionPreflightErrorV1> {
        match <Self as Parser>::try_parse_from(arguments) {
            Ok(parsed) => {
                validate_generation_argument_shape(&parsed)?;
                Ok(parsed)
            }
            Err(_) => Err(preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments,
                "broker-truth reviewed-input generation arguments are incomplete or invalid",
            )),
        }
    }
}

/// Canonical receipt/scope inputs; derived sidecars are verified against
/// the real bytes, and the missing receipt pair is derived from them.
#[derive(Parser)]
struct CanonicalReviewedInputArgumentsV1 {
    #[arg(long)]
    canonical_receipt_sha256: String,
    #[arg(long)]
    canonical_scope_sha256: String,
    #[arg(long)]
    canonical_root_verification_sha256: Option<String>,
    #[arg(long)]
    canonical_scope_window_binding_sha256: Option<String>,
    #[arg(long)]
    canonical_receipt: PathBuf,
    #[arg(long)]
    canonical_scope: PathBuf,
    /// Real canonical root-verification receipt; derived from the receipt when absent.
    #[arg(long)]
    canonical_root_verification: Option<PathBuf>,
    /// Real canonical scope-window binding; derived from the scope when absent.
    #[arg(long)]
    canonical_scope_window_binding: Option<PathBuf>,
}

/// Exact reviewed half-open window and environment binding.
#[derive(Parser)]
struct ReviewedWindowArgumentsV1 {
    #[arg(long)]
    environment: AcquisitionEnvironmentArgumentV1,
    #[arg(long)]
    account_id: i64,
    #[arg(long)]
    from_ms: i64,
    #[arg(long)]
    to_ms_exclusive: i64,
    #[arg(long)]
    window_policy_id: String,
}

/// Exact observation/rules evidence pairs for every reviewed synchronization.
#[derive(Parser)]
struct ReviewedEvidenceArgumentsV1 {
    #[arg(long, required = true, action = ArgAction::Append)]
    quote_observations: Vec<PathBuf>,
    #[arg(long, required = true, action = ArgAction::Append)]
    quote_observations_sha256: Vec<String>,
    #[arg(long, required = true, action = ArgAction::Append)]
    reviewed_replay_rules: Vec<PathBuf>,
    #[arg(long, required = true, action = ArgAction::Append)]
    reviewed_replay_rules_sha256: Vec<String>,
}

/// Exact instrument and account asset identities for the settlement route.
#[derive(Parser)]
struct ExactAssetArgumentsV1 {
    #[arg(long)]
    primary_base_asset_id: i64,
    #[arg(long)]
    primary_base_asset_name: String,
    #[arg(long)]
    primary_quote_asset_id: i64,
    #[arg(long)]
    primary_quote_asset_name: String,
    #[arg(long)]
    account_asset_id: i64,
    #[arg(long)]
    account_asset_name: String,
}

fn validate_generation_argument_shape(
    args: &ReviewedInputGenerationArgsV1,
) -> Result<(), BrokerTruthAcquisitionPreflightErrorV1> {
    let observation_count = args.evidence.quote_observations.len();
    let exact_pairs = observation_count > 0
        && args.evidence.quote_observations_sha256.len() == observation_count
        && args.evidence.reviewed_replay_rules.len() == observation_count
        && args.evidence.reviewed_replay_rules_sha256.len() == observation_count
        && args.window.account_id > 0
        && args.canonical.canonical_root_verification.is_some()
            == args.canonical.canonical_root_verification_sha256.is_some()
        && args.canonical.canonical_scope_window_binding.is_some()
            == args
                .canonical
                .canonical_scope_window_binding_sha256
                .is_some();
    if !exact_pairs {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments,
            "broker-truth reviewed-input generation arguments do not form exact evidence pairs",
        ));
    }
    if !args
        .settlement_leg
        .iter()
        .all(|spec| parse_leg_spec(spec).is_some())
    {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments,
            "settlement leg must carry exactly ten pipe-separated exact fields",
        ));
    }
    Ok(())
}

struct ExactInstrumentLegSpecV1 {
    from_asset_id: i64,
    from_asset_name: String,
    to_asset_id: i64,
    to_asset_name: String,
    symbol_id: i64,
    symbol_name: String,
    base_asset_id: i64,
    base_asset_name: String,
    quote_asset_id: i64,
    quote_asset_name: String,
}

fn parse_leg_spec(spec: &str) -> Option<ExactInstrumentLegSpecV1> {
    let fields = spec.split('|').collect::<Vec<_>>();
    if fields.len() != 10 {
        return None;
    }
    let from_asset_id: i64 = fields[0].parse().ok()?;
    let to_asset_id: i64 = fields[2].parse().ok()?;
    let symbol_id: i64 = fields[4].parse().ok()?;
    let base_asset_id: i64 = fields[6].parse().ok()?;
    let quote_asset_id: i64 = fields[8].parse().ok()?;
    let leg = ExactInstrumentLegSpecV1 {
        from_asset_id,
        from_asset_name: fields[1].to_owned(),
        to_asset_id,
        to_asset_name: fields[3].to_owned(),
        symbol_id,
        symbol_name: fields[5].to_owned(),
        base_asset_id,
        base_asset_name: fields[7].to_owned(),
        quote_asset_id,
        quote_asset_name: fields[9].to_owned(),
    };
    let exact_ids = leg.from_asset_id > 0
        && leg.to_asset_id > 0
        && leg.symbol_id > 0
        && leg.base_asset_id > 0
        && leg.quote_asset_id > 0;
    let exact_names = valid_asset_name(&leg.from_asset_name)
        && valid_asset_name(&leg.to_asset_name)
        && valid_asset_name(&leg.symbol_name)
        && valid_asset_name(&leg.base_asset_name)
        && valid_asset_name(&leg.quote_asset_name);
    let distinct_assets =
        leg.from_asset_id != leg.to_asset_id && leg.base_asset_id != leg.quote_asset_id;
    if !(exact_ids && exact_names && distinct_assets) {
        return None;
    }
    Some(leg)
}

fn valid_asset_name(name: &str) -> bool {
    !name.is_empty() && name == name.trim() && !name.chars().any(char::is_control)
}

fn window_value(window: EvidenceWindowV1) -> Value {
    json!({
        "from_unix_ms_inclusive": window.from_unix_ms_inclusive(),
        "to_unix_ms_exclusive": window.to_unix_ms_exclusive(),
    })
}

fn instrument_value(instrument: &ExactInstrumentWireV1) -> Value {
    json!({
        "symbol_id": instrument.symbol_id,
        "symbol_name": instrument.symbol_name,
        "base_asset_id": instrument.base_asset_id,
        "base_asset_name": instrument.base_asset_name,
        "quote_asset_id": instrument.quote_asset_id,
        "quote_asset_name": instrument.quote_asset_name,
    })
}

fn leg_value(leg: &ExactInstrumentLegSpecV1) -> Value {
    json!({
        "from_asset_id": leg.from_asset_id,
        "from_asset_name": leg.from_asset_name,
        "to_asset_id": leg.to_asset_id,
        "to_asset_name": leg.to_asset_name,
        "instrument": {
            "symbol_id": leg.symbol_id,
            "symbol_name": leg.symbol_name,
            "base_asset_id": leg.base_asset_id,
            "base_asset_name": leg.base_asset_name,
            "quote_asset_id": leg.quote_asset_id,
            "quote_asset_name": leg.quote_asset_name,
        },
    })
}

/// One generated, content-addressed reviewed-input artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedReviewedInputArtifactV1 {
    path: PathBuf,
    sha256: String,
}

impl GeneratedReviewedInputArtifactV1 {
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

/// Complete agent-reviewed input generation outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedReviewedInputsV1 {
    capture_plan: GeneratedReviewedInputArtifactV1,
    review_record: GeneratedReviewedInputArtifactV1,
    protocol_evidence: GeneratedReviewedInputArtifactV1,
    trust_root: GeneratedReviewedInputArtifactV1,
    canonical_root_verification: Option<GeneratedReviewedInputArtifactV1>,
    canonical_scope_window_binding: Option<GeneratedReviewedInputArtifactV1>,
}

impl GeneratedReviewedInputsV1 {
    pub fn capture_plan(&self) -> &GeneratedReviewedInputArtifactV1 {
        let artifact = &self.capture_plan;
        artifact
    }
    pub fn review_record(&self) -> &GeneratedReviewedInputArtifactV1 {
        let artifact = &self.review_record;
        artifact
    }
    pub fn protocol_evidence(&self) -> &GeneratedReviewedInputArtifactV1 {
        let artifact = &self.protocol_evidence;
        artifact
    }
    pub fn trust_root(&self) -> &GeneratedReviewedInputArtifactV1 {
        let artifact = &self.trust_root;
        artifact
    }

    /// Emitted only when the operator did not supply a real root-verification receipt.
    pub fn canonical_root_verification(&self) -> Option<&GeneratedReviewedInputArtifactV1> {
        self.canonical_root_verification.as_ref()
    }

    /// Emitted only when the operator did not supply a real scope-window binding.
    pub fn canonical_scope_window_binding(&self) -> Option<&GeneratedReviewedInputArtifactV1> {
        self.canonical_scope_window_binding.as_ref()
    }
}

fn random_reviewer_nonce_hex() -> String {
    let mut bytes = [0u8; 32];
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    for (index, chunk) in bytes.chunks_mut(8).enumerate() {
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_usize(index);
        hasher.write_u128(nanos);
        chunk.copy_from_slice(&hasher.finish().to_le_bytes());
    }
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn read_or_create_reviewer_nonce(
    work_parent: &Path,
) -> Result<String, BrokerTruthAcquisitionPreflightErrorV1> {
    let path = work_parent.join(AGENT_REVIEWER_NONCE_FILE_V1);
    if let Ok(content) = fs::read_to_string(&path) {
        let nonce = content.trim().to_owned();
        if validate_sha256(&nonce) {
            return Ok(nonce);
        }
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath,
            "existing agent-reviewer nonce file is not lowercase 64-hex",
        ));
    }
    let nonce = random_reviewer_nonce_hex();
    fs::write(&path, format!("{nonce}\n")).map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath,
            "agent-reviewer nonce cannot be stored in the explicit work parent",
        )
    })?;
    Ok(nonce)
}

fn validate_generation_roots(
    args: &ReviewedInputGenerationArgsV1,
) -> Result<(), BrokerTruthAcquisitionPreflightErrorV1> {
    let out_dir = &args.generate_reviewed_inputs;
    validate_directory_path(out_dir)?;
    validate_directory_path(&args.work_parent)?;
    let disjoint = out_dir != &args.work_parent
        && !out_dir.starts_with(&args.work_parent)
        && !args.work_parent.starts_with(out_dir);
    if !disjoint {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath,
            "reviewed-input output dir and work parent must be explicit and disjoint",
        ));
    }
    let mut unique = HashSet::new();
    let mut input_paths: Vec<&Path> = Vec::new();
    input_paths.push(&args.canonical.canonical_receipt);
    input_paths.push(&args.canonical.canonical_scope);
    if let Some(path) = &args.canonical.canonical_root_verification {
        input_paths.push(path);
    }
    if let Some(path) = &args.canonical.canonical_scope_window_binding {
        input_paths.push(path);
    }
    input_paths.extend(
        args.evidence
            .quote_observations
            .iter()
            .map(PathBuf::as_path),
    );
    input_paths.extend(
        args.evidence
            .reviewed_replay_rules
            .iter()
            .map(PathBuf::as_path),
    );
    for path in input_paths {
        if path.starts_with(out_dir) {
            return Err(preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath,
                "reviewed-input generation inputs must not live inside the output dir",
            ));
        }
        if !unique.insert(path.to_path_buf()) {
            return Err(preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath,
                "reviewed-input generation input paths must be distinct",
            ));
        }
    }
    Ok(())
}

fn exact_selected_generation(
    binding: &CanonicalSearchSourceBindingReceiptV1,
) -> Result<SelectedDatasetGenerationV1, BrokerTruthAcquisitionPreflightErrorV1> {
    let identity = CanonicalDatasetIdentity::from_path_component(binding.dataset_identity())
        .map_err(|_| {
            preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalReceiptInvalid,
                "canonical receipt contains an invalid dataset identity",
            )
        })?;
    SelectedDatasetGenerationV1::new(identity, binding.generation_id(), binding.manifest_sha256())
        .map_err(|_| {
            preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalReceiptInvalid,
                "canonical receipt contains an invalid selected generation",
            )
        })
}

fn selected_generation_value(
    selected: &SelectedDatasetGenerationV1,
) -> Result<Value, BrokerTruthAcquisitionPreflightErrorV1> {
    let bytes = selected.to_json_bytes().map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalReceiptInvalid,
            "canonical receipt selected generation cannot be encoded",
        )
    })?;
    serde_json::from_slice(&bytes).map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalReceiptInvalid,
            "canonical receipt selected generation cannot be decoded",
        )
    })
}

fn derive_root_verification_value(
    receipt: &CanonicalSearchInputReceiptV2,
    receipt_identity: &str,
) -> Result<Value, BrokerTruthAcquisitionPreflightErrorV1> {
    let bindings = receipt.source_bindings();
    let mut opened = Vec::with_capacity(bindings.len());
    for binding in bindings {
        let selected = exact_selected_generation(binding)?;
        let selected_value = selected_generation_value(&selected)?;
        opened.push(json!({
            "source_node_id": binding.source_node_id(),
            "selected_generation": selected_value,
            "manifest_schema_id": binding.manifest_schema_id(),
            "vortex_sha256": binding.vortex_sha256(),
        }));
    }
    Ok(json!({
        "schema_version": ACQUISITION_SCHEMA_VERSION_V1,
        "canonical_search_input_receipt_identity_sha256": receipt_identity,
        "opened_generations": opened,
    }))
}

fn validate_supplied_root_verification(
    wire: &CanonicalRootVerificationReceiptWireV1,
    receipt: &CanonicalSearchInputReceiptV2,
    receipt_identity: &str,
) -> Result<(), BrokerTruthAcquisitionPreflightErrorV1> {
    let bindings = receipt.source_bindings();
    let generations = &wire.opened_generations;
    let exact_header = wire.schema_version == ACQUISITION_SCHEMA_VERSION_V1
        && wire.canonical_search_input_receipt_identity_sha256 == receipt_identity
        && generations.len() == bindings.len();
    if !exact_header {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalAuthorityMismatch,
            "canonical root-verification receipt does not exactly cover the canonical receipt",
        ));
    }
    let mut index = 0usize;
    while index < bindings.len() {
        let binding = &bindings[index];
        let opened = &generations[index];
        let selected = exact_selected_generation(binding)?;
        let exact_entry = opened.source_node_id == binding.source_node_id()
            && opened.selected_generation == selected
            && opened.manifest_schema_id == binding.manifest_schema_id()
            && opened.vortex_sha256 == binding.vortex_sha256();
        if !exact_entry {
            return Err(preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalAuthorityMismatch,
                "canonical root-verification entry differs from its exact receipt source binding",
            ));
        }
        index += 1;
    }
    Ok(())
}

fn derive_window_binding_value(
    receipt_identity: &str,
    scope_identity: &str,
    scope: &CanonicalSearchArtifactScopeV2,
    requested_window: EvidenceWindowV1,
    window_policy_id: &str,
) -> Result<Value, BrokerTruthAcquisitionPreflightErrorV1> {
    let evaluated = scope.evaluated_window();
    let exact_window = requested_window.from_unix_ms_inclusive() == evaluated.timestamp_start_ms()
        && requested_window.to_unix_ms_exclusive() > evaluated.timestamp_end_ms();
    if !exact_window {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::WindowMismatch,
            "explicit acquisition window does not exactly bind the evaluated scope window",
        ));
    }
    Ok(json!({
        "schema_version": ACQUISITION_SCHEMA_VERSION_V1,
        "canonical_search_input_receipt_identity_sha256": receipt_identity,
        "canonical_search_artifact_scope_identity_sha256": scope_identity,
        "role": serde_json::to_value(evaluated.role()).expect("window role encodes as JSON"),
        "row_start": evaluated.row_start(),
        "row_end": evaluated.row_end(),
        "timestamp_start_ms": evaluated.timestamp_start_ms(),
        "timestamp_end_ms": evaluated.timestamp_end_ms(),
        "evidence_window": window_value(requested_window),
        "window_policy_id": window_policy_id,
    }))
}

fn write_generated_artifact(
    out_dir: &Path,
    name: &str,
    bytes: &[u8],
) -> Result<GeneratedReviewedInputArtifactV1, BrokerTruthAcquisitionPreflightErrorV1> {
    let path = out_dir.join(name);
    if fs::symlink_metadata(&path).is_ok() {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath,
            "reviewed-input output file already exists",
        ));
    }
    let sha256 = sha256_bytes(bytes);
    fs::write(&path, bytes).map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath,
            "reviewed-input output file cannot be written",
        )
    })?;
    fs::write(
        out_dir.join(format!("{name}.sha256")),
        format!("{sha256}\n"),
    )
    .map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath,
            "reviewed-input SHA-256 sidecar cannot be written",
        )
    })?;
    Ok(GeneratedReviewedInputArtifactV1 { path, sha256 })
}

fn decode_canonical_receipt(
    artifact: &FrozenArtifactV1,
) -> Result<CanonicalSearchInputReceiptV2, BrokerTruthAcquisitionPreflightErrorV1> {
    CanonicalSearchInputReceiptV2::from_json_bytes(&artifact.bytes).map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalReceiptInvalid,
            "canonical Search input receipt is invalid",
        )
    })
}

fn decode_canonical_scope(
    artifact: &FrozenArtifactV1,
) -> Result<CanonicalSearchArtifactScopeV2, BrokerTruthAcquisitionPreflightErrorV1> {
    CanonicalSearchArtifactScopeV2::from_json_bytes(&artifact.bytes).map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalScopeInvalid,
            "canonical Search artifact scope is invalid",
        )
    })
}

fn read_evidence_files(
    paths: &[PathBuf],
    digests: &[String],
) -> Result<Vec<FrozenArtifactV1>, BrokerTruthAcquisitionPreflightErrorV1> {
    let mut files = Vec::with_capacity(paths.len());
    for (path, digest) in paths.iter().zip(digests) {
        files.push(read_frozen_artifact(path, digest)?);
    }
    Ok(files)
}

fn validate_settlement_shape(
    primary_instrument: &ExactInstrumentWireV1,
    account_asset_id: i64,
    account_asset_name: &str,
    legs: &[ExactInstrumentLegSpecV1],
) -> Result<(), BrokerTruthAcquisitionPreflightErrorV1> {
    let quote_asset_id = primary_instrument.quote_asset_id;
    let quote_asset_name = primary_instrument.quote_asset_name.as_str();
    let account_is_quote =
        account_asset_id == quote_asset_id && account_asset_name == quote_asset_name;
    if legs.is_empty() && !account_is_quote {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments,
            "account asset differs from the primary quote asset; an exact settlement leg is required",
        ));
    }
    if !legs.is_empty() && account_is_quote {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments,
            "identity settlement route must carry zero legs",
        ));
    }
    let Some(first) = legs.first() else {
        return Ok(());
    };
    let Some(last) = legs.last() else {
        return Ok(());
    };
    let starts_at_quote =
        first.from_asset_id == quote_asset_id && first.from_asset_name == quote_asset_name;
    if !starts_at_quote {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments,
            "first settlement leg must start at the primary quote asset",
        ));
    }
    let ends_at_account =
        last.to_asset_id == account_asset_id && last.to_asset_name == account_asset_name;
    if !ends_at_account {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments,
            "last settlement leg must terminate at the exact account asset",
        ));
    }
    let mut chained = true;
    for pair in legs.windows(2) {
        if pair[0].to_asset_id != pair[1].from_asset_id
            || pair[0].to_asset_name != pair[1].from_asset_name
        {
            chained = false;
        }
    }
    if !chained {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments,
            "settlement legs do not form one exact chained conversion route",
        ));
    }
    for leg in legs {
        let base = (leg.base_asset_id, leg.base_asset_name.as_str());
        let quote = (leg.quote_asset_id, leg.quote_asset_name.as_str());
        let from_is_quoted = base == (leg.from_asset_id, leg.from_asset_name.as_str())
            || quote == (leg.from_asset_id, leg.from_asset_name.as_str());
        let to_is_quoted = base == (leg.to_asset_id, leg.to_asset_name.as_str())
            || quote == (leg.to_asset_id, leg.to_asset_name.as_str());
        if !from_is_quoted || !to_is_quoted {
            return Err(preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments,
                "settlement leg assets/names do not match its exact quoted instrument",
            ));
        }
    }
    Ok(())
}

/// Generates the agent-reviewed input bundle from the explicit real inputs.
///
/// Fails closed: absent, mismatched, or invalid inputs never produce partial
/// semantic authority, and no output file is ever overwritten.
pub fn generate_reviewed_inputs_v1(
    args: ReviewedInputGenerationArgsV1,
) -> Result<GeneratedReviewedInputsV1, BrokerTruthAcquisitionPreflightErrorV1> {
    validate_generation_argument_shape(&args)?;
    validate_generation_roots(&args)?;
    if !validate_window_policy_id(&args.window.window_policy_id) {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments,
            "reviewed window policy id is invalid",
        ));
    }
    let requested_window = EvidenceWindowV1::new(args.window.from_ms, args.window.to_ms_exclusive)
        .map_err(|_| {
            preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::WindowMismatch,
                "explicit acquisition window is not a valid half-open interval",
            )
        })?;

    let receipt = read_frozen_artifact(
        &args.canonical.canonical_receipt,
        &args.canonical.canonical_receipt_sha256,
    )?;
    let scope = read_frozen_artifact(
        &args.canonical.canonical_scope,
        &args.canonical.canonical_scope_sha256,
    )?;
    let typed_receipt = decode_canonical_receipt(&receipt)?;
    let anchor = typed_receipt.validate().map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalReceiptInvalid,
            "canonical Search input receipt cannot establish an exact anchor",
        )
    })?;
    let receipt_identity = typed_receipt.identity_sha256().map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalReceiptInvalid,
            "canonical Search input receipt identity cannot be computed",
        )
    })?;
    let typed_scope = decode_canonical_scope(&scope)?;
    typed_scope
        .validate_against_receipt(&typed_receipt)
        .map_err(|_| {
            preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalScopeInvalid,
                "canonical Search artifact scope differs from the explicit receipt",
            )
        })?;
    let scope_identity = typed_scope.identity_sha256().map_err(|_| {
        preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalScopeInvalid,
            "canonical Search artifact scope identity cannot be computed",
        )
    })?;

    let CanonicalDatasetScope::CTrader {
        environment,
        server,
        account_id,
        symbol_id,
    } = anchor.scope()
    else {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalAuthorityMismatch,
            "canonical anchor is not an exact cTrader dataset identity",
        ));
    };
    let anchor_exact = *environment == args.window.environment.canonical()
        && server == args.window.environment.endpoint_host()
        && *account_id == args.window.account_id;
    if !anchor_exact {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalAuthorityMismatch,
            "generation environment, server, or account differs from the canonical anchor",
        ));
    }

    let primary_assets_exact = valid_asset_name(&args.assets.primary_base_asset_name)
        && valid_asset_name(&args.assets.primary_quote_asset_name)
        && valid_asset_name(&args.assets.account_asset_name)
        && args.assets.primary_base_asset_id > 0
        && args.assets.primary_quote_asset_id > 0
        && args.assets.account_asset_id > 0
        && args.assets.primary_base_asset_id != args.assets.primary_quote_asset_id;
    if !primary_assets_exact {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments,
            "primary instrument or account asset ids/names are invalid",
        ));
    }
    let primary_instrument = ExactInstrumentWireV1 {
        symbol_id: *symbol_id,
        symbol_name: anchor.symbol_name().to_owned(),
        base_asset_id: args.assets.primary_base_asset_id,
        base_asset_name: args.assets.primary_base_asset_name.clone(),
        quote_asset_id: args.assets.primary_quote_asset_id,
        quote_asset_name: args.assets.primary_quote_asset_name.clone(),
    };
    let account_asset_id = args.assets.account_asset_id;
    let account_asset_name = args.assets.account_asset_name.clone();

    let legs = args
        .settlement_leg
        .iter()
        .map(|spec| {
            parse_leg_spec(spec).ok_or_else(|| {
                preflight_error(
                    BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments,
                    "settlement leg must carry exactly ten pipe-separated exact fields",
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_settlement_shape(
        &primary_instrument,
        account_asset_id,
        &account_asset_name,
        &legs,
    )?;
    let synchronization_count = 1 + legs.len();
    if args.evidence.quote_observations.len() != synchronization_count
        || args.evidence.reviewed_replay_rules.len() != synchronization_count
    {
        return Err(preflight_error(
            BrokerTruthAcquisitionPreflightErrorCodeV1::SynchronizationMismatch,
            "explicit observation/rules evidence does not match the exact synchronization set",
        ));
    }

    let observations = read_evidence_files(
        &args.evidence.quote_observations,
        &args.evidence.quote_observations_sha256,
    )?;
    let rules = read_evidence_files(
        &args.evidence.reviewed_replay_rules,
        &args.evidence.reviewed_replay_rules_sha256,
    )?;

    let (root_verification_bytes, root_verification_sha256, root_verification_emitted) = match (
        &args.canonical.canonical_root_verification,
        &args.canonical.canonical_root_verification_sha256,
    ) {
        (Some(path), Some(expected)) => {
            let frozen = read_frozen_artifact(path, expected)?;
            let wire: CanonicalRootVerificationReceiptWireV1 = decode_strict_json(
                &frozen,
                BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalAuthorityMismatch,
                "canonical root-verification receipt is invalid",
            )?;
            validate_supplied_root_verification(&wire, &typed_receipt, &receipt_identity)?;
            (frozen.bytes, frozen.sha256, false)
        }
        (None, None) => {
            let value = derive_root_verification_value(&typed_receipt, &receipt_identity)?;
            let bytes = serde_json::to_vec(&value).expect("root verification encodes as JSON");
            let sha256 = sha256_bytes(&bytes);
            (bytes, sha256, true)
        }
        _ => {
            return Err(preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments,
                "canonical root-verification path and digest must form an exact pair",
            ));
        }
    };

    let (window_binding_bytes, window_binding_sha256, window_binding_emitted) = match (
        &args.canonical.canonical_scope_window_binding,
        &args.canonical.canonical_scope_window_binding_sha256,
    ) {
        (Some(path), Some(expected)) => {
            let frozen = read_frozen_artifact(path, expected)?;
            let wire: CanonicalScopeWindowBindingWireV1 = decode_strict_json(
                &frozen,
                BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalAuthorityMismatch,
                "canonical scope-window binding is invalid",
            )?;
            validate_scope_window_binding(
                &wire,
                &receipt_identity,
                &scope_identity,
                &typed_scope,
                requested_window,
            )?;
            (frozen.bytes, frozen.sha256, false)
        }
        (None, None) => {
            let value = derive_window_binding_value(
                &receipt_identity,
                &scope_identity,
                &typed_scope,
                requested_window,
                &args.window.window_policy_id,
            )?;
            let bytes = serde_json::to_vec(&value).expect("scope-window binding encodes as JSON");
            let sha256 = sha256_bytes(&bytes);
            (bytes, sha256, true)
        }
        _ => {
            return Err(preflight_error(
                BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments,
                "canonical scope-window-binding path and digest must form an exact pair",
            ));
        }
    };

    let nonce = read_or_create_reviewer_nonce(&args.work_parent)?;
    let trust_root_bytes =
        format!("{AGENT_REVIEWER_IDENTITY_V1}\nagent_reviewer_nonce: {nonce}\n").into_bytes();
    let trust_root_sha256 = sha256_bytes(&trust_root_bytes);

    let protocol_value = json!({
        "schema_version": ACQUISITION_SCHEMA_VERSION_V1,
        "protocol": PROTOCOL_IDENTITY_V1,
        "reviewer": AGENT_REVIEWER_IDENTITY_V1,
        "reviewed": REVIEWED_DATE_V1,
        "quote_observations_sha256": observations.iter().map(|frozen| frozen.sha256.clone()).collect::<Vec<_>>(),
    });
    let protocol_bytes =
        serde_json::to_vec(&protocol_value).expect("protocol evidence encodes as JSON");
    let protocol_sha256 = sha256_bytes(&protocol_bytes);

    let environment_tag = match args.window.environment {
        AcquisitionEnvironmentArgumentV1::Demo => "demo",
        AcquisitionEnvironmentArgumentV1::Live => "live",
    };
    let synchronization_instruments = std::iter::once(primary_instrument.clone())
        .chain(legs.iter().map(|leg| ExactInstrumentWireV1 {
            symbol_id: leg.symbol_id,
            symbol_name: leg.symbol_name.clone(),
            base_asset_id: leg.base_asset_id,
            base_asset_name: leg.base_asset_name.clone(),
            quote_asset_id: leg.quote_asset_id,
            quote_asset_name: leg.quote_asset_name.clone(),
        }))
        .collect::<Vec<_>>();
    let review_record_value = json!({
        "schema_version": ACQUISITION_SCHEMA_VERSION_V1,
        "canonical_search_input_receipt_identity_sha256": receipt_identity,
        "canonical_search_artifact_scope_identity_sha256": scope_identity,
        "canonical_scope_window_binding_sha256": window_binding_sha256,
        "trust_root_sha256": trust_root_sha256,
        "protocol_evidence_sha256": protocol_sha256,
        "window_policy_id": args.window.window_policy_id,
        "environment": environment_tag,
        "server": args.window.environment.endpoint_host(),
        "account_id": args.window.account_id,
        "window": window_value(requested_window),
        "synchronizations": (0..synchronization_count)
            .map(|index| {
                json!({
                    "ordinal": u32::try_from(index).expect("synchronization ordinal fits u32"),
                    "instrument": instrument_value(&synchronization_instruments[index]),
                    "quote_observations_sha256": observations[index].sha256,
                    "reviewed_replay_rules_sha256": rules[index].sha256,
                })
            })
            .collect::<Vec<_>>(),
    });
    let review_record_bytes =
        serde_json::to_vec(&review_record_value).expect("review record encodes as JSON");
    let review_record_sha256 = sha256_bytes(&review_record_bytes);

    let capture_plan_value = json!({
        "schema_version": ACQUISITION_SCHEMA_VERSION_V1,
        "canonical_search_input_receipt_identity_sha256": receipt_identity,
        "canonical_search_artifact_scope_identity_sha256": scope_identity,
        "canonical_root_verification_sha256": root_verification_sha256,
        "canonical_scope_window_binding_sha256": window_binding_sha256,
        "review_record_sha256": review_record_sha256,
        "protocol_evidence_sha256": protocol_sha256,
        "trust_root_sha256": trust_root_sha256,
        "environment": environment_tag,
        "server": args.window.environment.endpoint_host(),
        "account_id": args.window.account_id,
        "window": window_value(requested_window),
        "primary_instrument": instrument_value(&primary_instrument),
        "account_asset": {"asset_id": account_asset_id, "asset_name": account_asset_name},
        "conversion_routes": [{
            "purpose": SETTLEMENT_PURPOSE_V1,
            "from_asset_id": primary_instrument.quote_asset_id,
            "from_asset_name": primary_instrument.quote_asset_name,
            "to_asset_id": account_asset_id,
            "to_asset_name": account_asset_name,
            "legs": legs.iter().map(leg_value).collect::<Vec<_>>(),
        }],
        "synchronizations": (0..synchronization_count)
            .map(|index| {
                let review_identity_binding = ReviewedQuoteReplayRuleIdentityV2::new(
                    review_record_sha256.clone(),
                    protocol_sha256.clone(),
                    observations[index].sha256.clone(),
                )
                .map_err(|_| {
                    preflight_error(
                        BrokerTruthAcquisitionPreflightErrorCodeV1::SynchronizationMismatch,
                        "reviewed synchronization identity inputs are invalid",
                    )
                })?;
                let review_identity = review_identity_binding.identity_sha256();
                Ok(json!({
                    "ordinal": u32::try_from(index).expect("synchronization ordinal fits u32"),
                    "account_id": args.window.account_id,
                    "instrument": instrument_value(&synchronization_instruments[index]),
                    "window": window_value(requested_window),
                    "quote_observations_sha256": observations[index].sha256,
                    "reviewed_replay_rules_sha256": rules[index].sha256,
                    "review_identity_sha256": review_identity,
                }))
            })
            .collect::<Result<Vec<_>, _>>()?,
    });
    let capture_plan_bytes =
        serde_json::to_vec(&capture_plan_value).expect("capture plan encodes as JSON");

    let capture_plan = write_generated_artifact(
        &args.generate_reviewed_inputs,
        CAPTURE_PLAN_FILE_V1,
        &capture_plan_bytes,
    )?;
    let review_record = write_generated_artifact(
        &args.generate_reviewed_inputs,
        REVIEW_RECORD_FILE_V1,
        &review_record_bytes,
    )?;
    let protocol_evidence = write_generated_artifact(
        &args.generate_reviewed_inputs,
        PROTOCOL_EVIDENCE_FILE_V1,
        &protocol_bytes,
    )?;
    let trust_root = write_generated_artifact(
        &args.generate_reviewed_inputs,
        TRUST_ROOT_FILE_V1,
        &trust_root_bytes,
    )?;
    let canonical_root_verification = if root_verification_emitted {
        Some(write_generated_artifact(
            &args.generate_reviewed_inputs,
            ROOT_VERIFICATION_FILE_V1,
            &root_verification_bytes,
        )?)
    } else {
        None
    };
    let canonical_scope_window_binding = if window_binding_emitted {
        Some(write_generated_artifact(
            &args.generate_reviewed_inputs,
            WINDOW_BINDING_FILE_V1,
            &window_binding_bytes,
        )?)
    } else {
        None
    };

    Ok(GeneratedReviewedInputsV1 {
        capture_plan,
        review_record,
        protocol_evidence,
        trust_root,
        canonical_root_verification,
        canonical_scope_window_binding,
    })
}
