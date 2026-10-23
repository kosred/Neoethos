//! Contract for the agent-reviewed capture-plan / review-record generator.
//!
//! The acquisition CLI must offer a reviewed-input generator that emits REAL,
//! schema-exact files (capture plan, review record, protocol evidence, trust
//! root, and the per-synchronization reviewed quote-session evidence that the
//! history ingress loads), content-addressed with lowercase SHA-256 sidecars,
//! deriving every canonical digest from the real operator-supplied input
//! bytes. The agent acts as reviewer and that identity is recorded in the
//! trust root; the generator never produces a permit or capability.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

use neoethos_broker_history::bootstrap_writer::{
    BrokerTrendbarStreamRequest, publish_broker_trendbar_chunks,
};
use neoethos_broker_history::broker_truth_capture::ExactQuoteInstrumentV2;
use neoethos_broker_history::{
    LoadedReviewedCTraderQuoteSynchronizationV2, ReviewedCTraderQuoteSynchronizationSourceV2,
    load_reviewed_ctrader_quote_synchronizations_v2,
};
use neoethos_broker_truth::{
    BrokerTruthReviewedSynchronizationBindingV1, EvidenceWindowV1,
    ReviewedQuoteReplayRuleIdentityV2,
};
use neoethos_broker_truth_acquire::{
    AGENT_REVIEWER_IDENTITY_V1, AGENT_REVIEWER_NONCE_FILE_V1,
    BrokerTruthAcquisitionPreflightErrorCodeV1, BrokerTruthAcquisitionPreflightErrorV1,
    GeneratedReviewedInputArtifactV1, GeneratedReviewedInputsV1,
    ReviewedInputGenerationArgsV1, generate_reviewed_inputs_v1,
};
use neoethos_data::core::vortex_io::write_vortex_array;
use neoethos_data::{
    BarTimestampConvention, CTraderEnvironment, CanonicalDatasetIdentity, CanonicalOhlcvChunk,
    CanonicalTimeframe, CanonicalVolumeChunk, SelectedDatasetGenerationV1,
};
use neoethos_search::{
    CanonicalSearchArtifactScopeV2, CanonicalSearchInputReceiptV2, CanonicalSearchWindowRoleV1,
};
use vortex_array::IntoArray;
use vortex_array::arrays::VarBinArray;

const SERVER: &str = "demo.ctraderapi.com";
const WINDOW_POLICY_ID: &str = "reviewed-half-open-m1-v1";
const FROM_MS: i64 = 1_700_000_040_000;
const LAST_BAR_MS: i64 = FROM_MS + 120_000;
const TO_MS_EXCLUSIVE: i64 = FROM_MS + 180_000;
const ACCOUNT_ID: i64 = 7_001;
const PRIMARY_SYMBOL_ID: i64 = 14;
const CONVERSION_SYMBOL_ID: i64 = 15;
const USD_ASSET_ID: i64 = 2;
const EUR_ASSET_ID: i64 = 1;
const JPY_ASSET_ID: i64 = 3;
const TICK_ARCHIVE_SCHEMA_V1: &str = "neoethos.unreviewed-ctrader-tick-archive.v1";
const TICK_RESPONSE_PAYLOAD_TYPE: u32 = 2_146;
const BID_SIDE: i32 = 1;
const ASK_SIDE: i32 = 2;

// ---------------------------------------------------------------------------
// Strict mirrors of the private acquisition wire schemas. Every field name
// must match the wire type in crates/neoethos-broker-truth-acquire/src/lib.rs
// exactly: these mirrors reject unknown fields, so a drift fails the test.

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MirrorWindow {
    from_unix_ms_inclusive: i64,
    to_unix_ms_exclusive: i64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MirrorInstrument {
    symbol_id: i64,
    symbol_name: String,
    base_asset_id: i64,
    base_asset_name: String,
    quote_asset_id: i64,
    quote_asset_name: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MirrorAccountAsset {
    asset_id: i64,
    asset_name: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MirrorReviewSync {
    ordinal: u32,
    reviewed_replay_rules_sha256: String,
    instrument: MirrorInstrument,
    quote_observations_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MirrorReviewRecord {
    synchronizations: Vec<MirrorReviewSync>,
    schema_version: u16,
    protocol_evidence_sha256: String,
    trust_root_sha256: String,
    canonical_scope_window_binding_sha256: String,
    canonical_search_artifact_scope_identity_sha256: String,
    canonical_search_input_receipt_identity_sha256: String,
    window_policy_id: String,
    environment: String,
    server: String,
    account_id: i64,
    window: MirrorWindow,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MirrorPlanSync {
    window: MirrorWindow,
    ordinal: u32,
    review_identity_sha256: String,
    account_id: i64,
    instrument: MirrorInstrument,
    reviewed_replay_rules_sha256: String,
    quote_observations_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MirrorLeg {
    from_asset_id: i64,
    from_asset_name: String,
    to_asset_id: i64,
    to_asset_name: String,
    instrument: MirrorInstrument,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MirrorRoute {
    purpose: String,
    from_asset_id: i64,
    from_asset_name: String,
    to_asset_id: i64,
    to_asset_name: String,
    legs: Vec<MirrorLeg>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MirrorPlan {
    schema_version: u16,
    environment: String,
    server: String,
    account_id: i64,
    window: MirrorWindow,
    primary_instrument: MirrorInstrument,
    account_asset: MirrorAccountAsset,
    conversion_routes: Vec<MirrorRoute>,
    synchronizations: Vec<MirrorPlanSync>,
    canonical_search_input_receipt_identity_sha256: String,
    canonical_search_artifact_scope_identity_sha256: String,
    canonical_root_verification_sha256: String,
    canonical_scope_window_binding_sha256: String,
    review_record_sha256: String,
    protocol_evidence_sha256: String,
    trust_root_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MirrorOpenedGeneration {
    source_node_id: String,
    selected_generation: Value,
    manifest_schema_id: String,
    vortex_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MirrorRootVerification {
    opened_generations: Vec<MirrorOpenedGeneration>,
    schema_version: u16,
    canonical_search_input_receipt_identity_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MirrorWindowBinding {
    role: String,
    row_start: u64,
    row_end: u64,
    timestamp_start_ms: i64,
    timestamp_end_ms: i64,
    evidence_window: MirrorWindow,
    window_policy_id: String,
    canonical_search_input_receipt_identity_sha256: String,
    canonical_search_artifact_scope_identity_sha256: String,
    schema_version: u16,
}

// ---------------------------------------------------------------------------

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(64);
    for byte in digest {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

fn is_lowercase_hex64(value: &str) -> bool {
    if value.len() != 64 {
        return false;
    }
    let mut only_hex = true;
    for byte in value.bytes() {
        only_hex &= byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte);
    }
    only_hex
}

fn fixture_window() -> EvidenceWindowV1 {
    EvidenceWindowV1::new(FROM_MS, TO_MS_EXCLUSIVE).expect("valid half-open fixture window")
}

fn primary_identity() -> CanonicalDatasetIdentity {
    CanonicalDatasetIdentity::ctrader(
        CTraderEnvironment::Demo,
        SERVER,
        ACCOUNT_ID,
        PRIMARY_SYMBOL_ID,
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("valid exact cTrader dataset identity")
}

/// One archived cTrader tick page in the real tick-archive page format. The
/// fixture is synthetic but structurally identical to the archive pages the
/// generator reads and validates.
struct ProbePageContent {
    sequence: u64,
    side: i32,
    client_msg_id: String,
    envelope: Value,
    raw_response_sha256: Option<String>,
    schema: &'static str,
}

impl ProbePageContent {
    fn bid() -> Self {
        Self::new(0, BID_SIDE)
    }

    fn ask() -> Self {
        Self::new(1, ASK_SIDE)
    }

    fn new(sequence: u64, side: i32) -> Self {
        let client_msg_id = format!("reviewed-fixture-{sequence}-side-{side}");
        Self {
            sequence,
            side,
            client_msg_id: client_msg_id.clone(),
            envelope: json!({
                "clientMsgId": client_msg_id,
                "payloadType": TICK_RESPONSE_PAYLOAD_TYPE,
                "payload": {
                    "ctidTraderAccountId": ACCOUNT_ID,
                    "hasMore": false,
                    "tickData": [
                        {"tick": 100 + sequence, "timestamp": FROM_MS + 1_000 + i64::try_from(sequence).expect("sequence fits i64")},
                    ],
                },
            }),
            raw_response_sha256: None,
            schema: TICK_ARCHIVE_SCHEMA_V1,
        }
    }
}

/// Mirrors tick_archive.rs raw_fragments: bounded UTF-8 fragments of the
/// serialized page record, one VarBin vortex column.
fn fragment_array(raw: &str) -> vortex_array::ArrayRef {
    let mut fragments = Vec::new();
    let mut rest = raw;
    while !rest.is_empty() {
        let mut end = rest.len().min(1024);
        while !rest.is_char_boundary(end) {
            end -= 1;
        }
        fragments.push(&rest[..end]);
        rest = &rest[end..];
    }
    VarBinArray::from(fragments).into_array()
}

fn write_probe_page(root: &Path, name: &str, content: &ProbePageContent) -> (PathBuf, String) {
    let raw_response_json = serde_json::to_string(&content.envelope).expect("envelope JSON");
    let raw_response_sha256 = content
        .raw_response_sha256
        .clone()
        .unwrap_or_else(|| sha256(raw_response_json.as_bytes()));
    let record = json!({
        "schema": content.schema,
        "plan_sha256": "cd".repeat(32),
        "sequence": content.sequence,
        "cursor": {
            "chunk_from_ms": FROM_MS,
            "chunk_to_ms_exclusive": TO_MS_EXCLUSIVE,
            "page_to_ms_exclusive": TO_MS_EXCLUSIVE,
            "side": content.side,
        },
        "client_msg_id": content.client_msg_id,
        "captured_at_ms": 1_800_000_000_000_i64,
        "raw_response_sha256": raw_response_sha256,
        "raw_response_json": raw_response_json,
    });
    let serialized = serde_json::to_string(&record).expect("page record JSON");
    let path = root.join(name);
    write_vortex_array(&path, fragment_array(&serialized)).expect("write probe page fixture");
    let bytes = fs::read(&path).expect("read probe page fixture");
    (path, sha256(&bytes))
}

struct FixturePublication {
    selected_value: Value,
    manifest_schema_id: String,
    manifest_vortex_sha256: String,
    manifest_binding_sha256: String,
    generation_id: String,
}

fn publish_fixture_generation(
    configured_root: &Path,
    identity: &CanonicalDatasetIdentity,
) -> FixturePublication {
    let publication = publish_broker_trendbar_chunks(BrokerTrendbarStreamRequest {
        retrieved_unix_ms: 1_800_000_000_000,
        configured_root,
        identity,
        expected_generation: None,
        requested_from_ms: FROM_MS,
        requested_to_ms: TO_MS_EXCLUSIVE,
        returned_from_ms: FROM_MS,
        returned_to_ms: LAST_BAR_MS,
        row_count: 3,
        chunks: vec![Ok(CanonicalOhlcvChunk {
            volume: CanonicalVolumeChunk::Int64(vec![10, 11, 12]),
            close: vec![1.11, 1.12, 1.13],
            low: vec![1.09, 1.10, 1.11],
            high: vec![1.12, 1.13, 1.14],
            open: vec![1.10, 1.11, 1.12],
            timestamp_ms: vec![FROM_MS, FROM_MS + 60_000, LAST_BAR_MS],
        })],
    })
    .expect("publish exact canonical cTrader generation");
    let manifest = publication.manifest();
    let selected = SelectedDatasetGenerationV1::from_manifest(manifest)
        .expect("select exact canonical generation");
    let selected_value: Value = serde_json::from_slice(
        &selected
            .to_json_bytes()
            .expect("selected generation receipt JSON"),
    )
    .expect("selected generation JSON value");
    FixturePublication {
        selected_value,
        manifest_schema_id: manifest.schema_id().to_owned(),
        manifest_vortex_sha256: manifest.vortex_sha256().to_owned(),
        manifest_binding_sha256: manifest.manifest_binding_sha256().to_owned(),
        generation_id: manifest.generation_id().to_owned(),
    }
}

struct Fixture {
    _temp: TempDir,
    args: Vec<String>,
    observations: Vec<(PathBuf, String)>,
    manifest_vortex_sha256: String,
    manifest_schema_id: String,
    selected_value: Value,
    scope_identity: String,
    receipt_identity: String,
    work_parent: PathBuf,
    out_dir: PathBuf,
    input_root: PathBuf,
}

impl Fixture {
    fn new(account_asset_id: i64, account_asset_name: &str, legs: &[&str]) -> Self {
        let pages = [ProbePageContent::bid(), ProbePageContent::ask()];
        Self::with_pages(account_asset_id, account_asset_name, legs, &pages)
    }

    fn with_pages(
        account_asset_id: i64,
        account_asset_name: &str,
        legs: &[&str],
        pages: &[ProbePageContent; 2],
    ) -> Self {
        let temp = tempfile::tempdir().expect("temporary generator fixture");
        let data_root = temp.path().join("canonical-data");
        let input_root = temp.path().join("immutable-inputs");
        let work_parent = temp.path().join("capture-work");
        let out_dir = temp.path().join("reviewed-inputs");
        for directory in [&input_root, &work_parent, &out_dir] {
            fs::create_dir_all(directory).expect("create generator fixture directory");
        }

        let identity = primary_identity();
        let publication = publish_fixture_generation(&data_root, &identity);
        let selected_value = publication.selected_value;
        let manifest_schema_id = publication.manifest_schema_id;
        let manifest_vortex_sha256 = publication.manifest_vortex_sha256;
        let manifest_binding_sha256 = publication.manifest_binding_sha256;
        let generation_id = publication.generation_id;

        let receipt_value = json!({
            "feature_content_sha256": "33".repeat(32),
            "feature_provenance_identity": "22".repeat(32),
            "feature_plan_identity": "11".repeat(32),
            "anchor_dataset_identity": identity.to_path_component(),
            "schema_version": 2,
            "source_bindings": [{
                "generation_id": generation_id,
                "segments": [{
                    "timestamp_end_ms": LAST_BAR_MS,
                    "timestamp_start_ms": FROM_MS,
                    "row_end": 3,
                    "row_start": 0
                }],
                "vortex_sha256": manifest_vortex_sha256,
                "manifest_sha256": manifest_binding_sha256,
                "manifest_schema_id": manifest_schema_id,
                "dataset_identity": identity.to_path_component(),
                "source_node_id": "anchor",
                "bar_timestamp_convention": identity.bar_timestamp_convention().to_string()
            }],
            "feature_execution": {
                "selected_lane": "cpu_scalar",
                "vector_ta_math_authority": "neoethos.vector-ta.cpu-f64-exact-bits.v1",
                "compute_policy": "auto",
                "schema_version": 1
            }
        });
        let typed_receipt = CanonicalSearchInputReceiptV2::from_json_bytes(
            &serde_json::to_vec(&receipt_value).expect("encode receipt fixture"),
        )
        .expect("strict canonical receipt fixture");
        let receipt_identity = typed_receipt
            .identity_sha256()
            .expect("canonical receipt identity");
        let receipt_bytes = typed_receipt
            .to_json_bytes()
            .expect("canonical receipt JSON");
        let receipt_sha256 = sha256(&receipt_bytes);
        let receipt_path = input_root.join("canonical-search-input-receipt.json");
        fs::write(&receipt_path, &receipt_bytes).expect("write canonical receipt file");

        let typed_scope = CanonicalSearchArtifactScopeV2::for_entire_receipt(
            CanonicalSearchWindowRoleV1::Holdout,
            typed_receipt.clone(),
        )
        .expect("canonical holdout scope fixture");
        let scope_identity = typed_scope
            .identity_sha256()
            .expect("canonical scope identity");
        let scope_bytes = typed_scope.to_json_bytes().expect("canonical scope JSON");
        let scope_sha256 = sha256(&scope_bytes);
        let scope_path = input_root.join("canonical-search-artifact-scope.json");
        fs::write(&scope_path, &scope_bytes).expect("write canonical scope file");

        let mut observations = Vec::with_capacity(pages.len());
        for (index, page) in pages.iter().enumerate() {
            let name = format!("probe-page-{index:03}.vortex");
            observations.push(write_probe_page(&input_root, &name, page));
        }

        let mut args = vec![
            "broker-truth-acquire".to_owned(),
            "--generate-reviewed-inputs".to_owned(),
            out_dir.to_string_lossy().into_owned(),
            "--work-parent".to_owned(),
            work_parent.to_string_lossy().into_owned(),
            "--canonical-receipt".to_owned(),
            receipt_path.to_string_lossy().into_owned(),
            "--canonical-receipt-sha256".to_owned(),
            receipt_sha256.clone(),
            "--canonical-scope".to_owned(),
            scope_path.to_string_lossy().into_owned(),
            "--canonical-scope-sha256".to_owned(),
            scope_sha256.clone(),
            "--environment".to_owned(),
            "demo".to_owned(),
            "--account-id".to_owned(),
            ACCOUNT_ID.to_string(),
            "--from-ms".to_owned(),
            FROM_MS.to_string(),
            "--to-ms-exclusive".to_owned(),
            TO_MS_EXCLUSIVE.to_string(),
            "--window-policy-id".to_owned(),
            WINDOW_POLICY_ID.to_owned(),
            "--primary-base-asset-id".to_owned(),
            EUR_ASSET_ID.to_string(),
            "--primary-base-asset-name".to_owned(),
            "EUR".to_owned(),
            "--primary-quote-asset-id".to_owned(),
            USD_ASSET_ID.to_string(),
            "--primary-quote-asset-name".to_owned(),
            "USD".to_owned(),
            "--account-asset-id".to_owned(),
            account_asset_id.to_string(),
            "--account-asset-name".to_owned(),
            account_asset_name.to_owned(),
        ];
        for leg in legs {
            args.push("--settlement-leg".to_owned());
            args.push((*leg).to_owned());
        }
        for (path, digest) in &observations {
            args.push("--quote-observations".to_owned());
            args.push(path.to_string_lossy().into_owned());
            args.push("--quote-observations-sha256".to_owned());
            args.push(digest.clone());
        }

        Self {
            _temp: temp,
            input_root,
            out_dir,
            work_parent,
            receipt_identity,
            scope_identity,
            selected_value,
            manifest_schema_id,
            manifest_vortex_sha256,
            observations,
            args,
        }
    }

    fn generate(
        &self,
    ) -> Result<GeneratedReviewedInputsV1, BrokerTruthAcquisitionPreflightErrorV1> {
        let args = ReviewedInputGenerationArgsV1::try_parse_from(self.args.clone())
            .expect("valid reviewed-input generation arguments");
        generate_reviewed_inputs_v1(args)
    }
}

fn generation_parse_error(args: Vec<String>) -> BrokerTruthAcquisitionPreflightErrorCodeV1 {
    match ReviewedInputGenerationArgsV1::try_parse_from(args) {
        Ok(_) => panic!("invalid generation arguments unexpectedly parsed"),
        Err(error) => error.code(),
    }
}

fn with_flag_value(args: &[String], flag: &str, value: String) -> Vec<String> {
    let mut changed = args.to_vec();
    let index = changed
        .iter()
        .position(|argument| argument == flag)
        .expect("fixture contains required flag");
    changed[index + 1] = value;
    changed
}

fn expect_generation_error(
    args: Vec<String>,
    expected: BrokerTruthAcquisitionPreflightErrorCodeV1,
) {
    let parsed = ReviewedInputGenerationArgsV1::try_parse_from(args)
        .expect("syntactically valid generation arguments");
    let error =
        generate_reviewed_inputs_v1(parsed).expect_err("invalid generation input must be refused");
    assert_eq!(error.code(), expected);
}

fn read_json_mirror<T: for<'de> Deserialize<'de>>(path: &Path) -> T {
    serde_json::from_slice(&fs::read(path).expect("read generated artifact"))
        .expect("generated artifact decodes into the strict wire mirror")
}

fn assert_sidecar(path: &Path, sha256: &str) {
    let sidecar = PathBuf::from(format!("{}.sha256", path.display()));
    let content = fs::read_to_string(&sidecar).expect("generated SHA-256 sidecar");
    assert_eq!(
        content.trim(),
        sha256,
        "sidecar must carry the lowercase digest"
    );
}

fn assert_no_permit_content(bytes: &[u8]) {
    let lowered = String::from_utf8_lossy(bytes).to_lowercase();
    assert!(
        !lowered.contains("permit") && !lowered.contains("capability"),
        "generated artifact must never produce a permit or capability"
    );
}

fn assert_exact_instrument(
    instrument: &MirrorInstrument,
    symbol_id: i64,
    symbol_name: &str,
    base_id: i64,
    base_name: &str,
    quote_id: i64,
    quote_name: &str,
) {
    assert_eq!(instrument.symbol_id, symbol_id);
    assert_eq!(instrument.symbol_name, symbol_name);
    assert_eq!(instrument.base_asset_id, base_id);
    assert_eq!(instrument.base_asset_name, base_name);
    assert_eq!(instrument.quote_asset_id, quote_id);
    assert_eq!(instrument.quote_asset_name, quote_name);
}

fn assert_exact_window(window: &MirrorWindow) {
    assert_eq!(window.from_unix_ms_inclusive, FROM_MS);
    assert_eq!(window.to_unix_ms_exclusive, TO_MS_EXCLUSIVE);
}

fn expect_review_identity(
    plan_sync: &MirrorPlanSync,
    review_record_sha256: &str,
    protocol_sha256: &str,
    observation_sha256: &str,
) {
    let binding = ReviewedQuoteReplayRuleIdentityV2::new(
        review_record_sha256.to_owned(),
        protocol_sha256.to_owned(),
        observation_sha256.to_owned(),
    )
    .expect("reviewed replay-rule identity");
    assert_eq!(plan_sync.review_identity_sha256, binding.identity_sha256());
}

fn assert_evidence_artifact(artifact: &GeneratedReviewedInputArtifactV1) {
    let bytes = fs::read(artifact.path()).expect("read generated evidence artifact");
    assert_eq!(sha256(&bytes), artifact.sha256().to_owned());
    assert_sidecar(artifact.path(), artifact.sha256());
}

/// Loads one emitted synchronization through the REAL history ingress
/// (load_reviewed_ctrader_quote_synchronizations_v2) with the exact binding
/// the acquisition orchestrator constructs. This is the acceptance proof that
/// the emitted evidence is schema-exact and ingress-valid.
fn load_emitted_synchronization(
    ordinal: u32,
    symbol_id: i64,
    symbol_name: &str,
    base_id: i64,
    base_name: &str,
    quote_id: i64,
    quote_name: &str,
    generated: &GeneratedReviewedInputsV1,
) -> LoadedReviewedCTraderQuoteSynchronizationV2 {
    let ordinal_index = usize::try_from(ordinal).expect("ordinal fits fixture index");
    let review_record_sha256 = generated.review_record().sha256().to_owned();
    let protocol_sha256 = generated.protocol_evidence().sha256().to_owned();
    let observations_sha256 = generated.quote_observations()[ordinal_index]
        .sha256()
        .to_owned();
    let rules_sha256 = generated.reviewed_replay_rules()[ordinal_index]
        .sha256()
        .to_owned();
    let review_identity = ReviewedQuoteReplayRuleIdentityV2::new(
        review_record_sha256,
        protocol_sha256,
        observations_sha256,
    )
    .expect("reviewed replay-rule identity");
    let binding = BrokerTruthReviewedSynchronizationBindingV1::new(
        ordinal,
        ACCOUNT_ID,
        symbol_id,
        fixture_window(),
        review_identity,
        rules_sha256,
    )
    .expect("exact reviewed synchronization binding");
    let instrument = ExactQuoteInstrumentV2::new(
        symbol_id, symbol_name, base_id, base_name, quote_id, quote_name,
    )
    .expect("exact fixture instrument");
    let source = ReviewedCTraderQuoteSynchronizationSourceV2::new(
        binding,
        instrument,
        generated.quote_observations()[ordinal_index].path().to_path_buf(),
        generated.reviewed_replay_rules()[ordinal_index].path().to_path_buf(),
    )
    .expect("explicit evidence source");
    let loaded = load_reviewed_ctrader_quote_synchronizations_v2(vec![source])
        .expect("history ingress must accept the emitted reviewed evidence");
    assert_eq!(loaded.len(), 1);
    loaded.into_iter().next().expect("one loaded synchronization")
}

#[test]
fn generator_emits_schema_exact_content_addressed_reviewed_bundle() {
    let fixture = Fixture::new(USD_ASSET_ID, "USD", &[]);
    let generated = fixture.generate().expect("agent-reviewed input generation");

    let capture_plan = generated.capture_plan();
    let review_record = generated.review_record();
    let protocol_evidence = generated.protocol_evidence();
    let trust_root = generated.trust_root();
    let root_verification = generated
        .canonical_root_verification()
        .expect("derived canonical root verification");
    let window_binding = generated
        .canonical_scope_window_binding()
        .expect("derived canonical scope-window binding");
    assert_eq!(generated.quote_observations().len(), 1);
    assert_eq!(generated.reviewed_replay_rules().len(), 1);
    for artifact in generated
        .quote_observations()
        .iter()
        .chain(generated.reviewed_replay_rules())
    {
        assert_evidence_artifact(artifact);
    }

    for (label, digest) in [
        ("capture plan", capture_plan.sha256()),
        ("review record", review_record.sha256()),
        ("protocol evidence", protocol_evidence.sha256()),
        ("trust root", trust_root.sha256()),
        ("root verification", root_verification.sha256()),
        ("window binding", window_binding.sha256()),
    ] {
        assert!(
            is_lowercase_hex64(digest),
            "{label} digest is not lowercase 64-hex: {digest}"
        );
    }
    for (label, artifact) in [
        ("capture plan", capture_plan),
        ("review record", review_record),
        ("protocol evidence", protocol_evidence),
        ("trust root", trust_root),
        ("root verification", root_verification),
        ("window binding", window_binding),
    ] {
        let file_bytes =
            fs::read(artifact.path()).unwrap_or_else(|error| panic!("read {label}: {error}"));
        assert_eq!(
            sha256(&file_bytes),
            artifact.sha256().to_owned(),
            "{label} digest must equal its file bytes"
        );
        assert_sidecar(artifact.path(), artifact.sha256());
    }

    let review: MirrorReviewRecord = read_json_mirror(review_record.path());
    assert_eq!(review.schema_version, 1);
    assert_eq!(review.environment, "demo");
    assert_eq!(review.server, SERVER);
    assert_eq!(review.account_id, ACCOUNT_ID);
    assert_exact_window(&review.window);
    assert_eq!(review.window_policy_id, WINDOW_POLICY_ID);
    assert_eq!(
        review.canonical_search_input_receipt_identity_sha256,
        fixture.receipt_identity
    );
    assert_eq!(
        review.canonical_search_artifact_scope_identity_sha256,
        fixture.scope_identity
    );
    assert_eq!(
        review.canonical_scope_window_binding_sha256,
        window_binding.sha256().to_owned()
    );
    assert_eq!(review.trust_root_sha256, trust_root.sha256().to_owned());
    assert_eq!(
        review.protocol_evidence_sha256,
        protocol_evidence.sha256().to_owned()
    );
    assert_eq!(review.synchronizations.len(), 1);
    let review_sync = &review.synchronizations[0];
    assert_eq!(review_sync.ordinal, 0);
    assert_exact_instrument(
        &review_sync.instrument,
        PRIMARY_SYMBOL_ID,
        "EURUSD",
        EUR_ASSET_ID,
        "EUR",
        USD_ASSET_ID,
        "USD",
    );
    assert_eq!(
        review_sync.quote_observations_sha256,
        generated.quote_observations()[0].sha256().to_owned()
    );
    assert_eq!(
        review_sync.reviewed_replay_rules_sha256,
        generated.reviewed_replay_rules()[0].sha256().to_owned()
    );

    let plan: MirrorPlan = read_json_mirror(capture_plan.path());
    assert_eq!(
        (
            plan.schema_version,
            plan.environment.as_str(),
            plan.server.as_str(),
            plan.account_id,
            plan.window.from_unix_ms_inclusive,
            plan.window.to_unix_ms_exclusive,
            plan.primary_instrument.symbol_id,
            plan.primary_instrument.symbol_name.as_str(),
            plan.primary_instrument.base_asset_id,
            plan.primary_instrument.base_asset_name.as_str(),
            plan.primary_instrument.quote_asset_id,
            plan.primary_instrument.quote_asset_name.as_str(),
        ),
        (
            1,
            "demo",
            SERVER,
            ACCOUNT_ID,
            FROM_MS,
            TO_MS_EXCLUSIVE,
            PRIMARY_SYMBOL_ID,
            "EURUSD",
            EUR_ASSET_ID,
            "EUR",
            USD_ASSET_ID,
            "USD",
        )
    );
    assert_eq!(
        (
            plan.account_asset.asset_id,
            plan.account_asset.asset_name.as_str()
        ),
        (USD_ASSET_ID, "USD")
    );
    assert_eq!(
        (
            plan.canonical_search_input_receipt_identity_sha256.as_str(),
            plan.canonical_search_artifact_scope_identity_sha256
                .as_str(),
            plan.canonical_root_verification_sha256.as_str(),
            plan.canonical_scope_window_binding_sha256.as_str(),
            plan.review_record_sha256.as_str(),
            plan.protocol_evidence_sha256.as_str(),
            plan.trust_root_sha256.as_str(),
        ),
        (
            fixture.receipt_identity.as_str(),
            fixture.scope_identity.as_str(),
            root_verification.sha256(),
            window_binding.sha256(),
            review_record.sha256(),
            protocol_evidence.sha256(),
            trust_root.sha256(),
        )
    );
    assert_eq!(plan.conversion_routes.len(), 1);
    let route = &plan.conversion_routes[0];
    assert_eq!(route.purpose, "primary_pnl_settlement");
    assert_eq!(route.from_asset_id, USD_ASSET_ID);
    assert_eq!(route.from_asset_name, "USD");
    assert_eq!(route.to_asset_id, USD_ASSET_ID);
    assert_eq!(route.to_asset_name, "USD");
    assert!(
        route.legs.is_empty(),
        "identity settlement carries zero legs"
    );
    assert_eq!(plan.synchronizations.len(), 1);
    let plan_sync = &plan.synchronizations[0];
    assert_eq!(plan_sync.ordinal, 0);
    assert_eq!(plan_sync.account_id, ACCOUNT_ID);
    assert_exact_window(&plan_sync.window);
    assert_eq!(
        plan_sync.quote_observations_sha256,
        generated.quote_observations()[0].sha256().to_owned()
    );
    assert_eq!(
        plan_sync.reviewed_replay_rules_sha256,
        generated.reviewed_replay_rules()[0].sha256().to_owned()
    );
    expect_review_identity(
        plan_sync,
        review_record.sha256(),
        protocol_evidence.sha256(),
        generated.quote_observations()[0].sha256(),
    );

    let root: MirrorRootVerification = read_json_mirror(root_verification.path());
    assert_eq!(root.schema_version, 1);
    assert_eq!(
        root.canonical_search_input_receipt_identity_sha256,
        fixture.receipt_identity
    );
    assert_eq!(root.opened_generations.len(), 1);
    assert_eq!(root.opened_generations[0].source_node_id, "anchor");
    assert_eq!(
        root.opened_generations[0].selected_generation,
        fixture.selected_value
    );
    assert_eq!(
        root.opened_generations[0].manifest_schema_id,
        fixture.manifest_schema_id
    );
    assert_eq!(
        root.opened_generations[0].vortex_sha256,
        fixture.manifest_vortex_sha256
    );

    let binding: MirrorWindowBinding = read_json_mirror(window_binding.path());
    assert_eq!(binding.schema_version, 1);
    assert_eq!(binding.role, "holdout");
    assert_eq!(binding.row_start, 0);
    assert_eq!(binding.row_end, 3);
    assert_eq!(binding.timestamp_start_ms, FROM_MS);
    assert_eq!(binding.timestamp_end_ms, LAST_BAR_MS);
    assert_exact_window(&binding.evidence_window);
    assert_eq!(binding.window_policy_id, WINDOW_POLICY_ID);
    assert_eq!(
        binding.canonical_search_input_receipt_identity_sha256,
        fixture.receipt_identity
    );
    assert_eq!(
        binding.canonical_search_artifact_scope_identity_sha256,
        fixture.scope_identity
    );

    let trust_root_bytes = fs::read(trust_root.path()).expect("read trust root");
    let trust_root_text = String::from_utf8_lossy(&trust_root_bytes);
    assert!(
        trust_root_text.contains(AGENT_REVIEWER_IDENTITY_V1),
        "trust root must record the exact agent-reviewer identity"
    );
    let nonce_path = fixture.work_parent.join(AGENT_REVIEWER_NONCE_FILE_V1);
    let nonce = fs::read_to_string(&nonce_path).expect("agent-reviewer nonce file in work dir");
    let nonce = nonce.trim();
    assert!(is_lowercase_hex64(nonce), "nonce must be lowercase 64-hex");
    assert!(
        trust_root_text.contains(nonce),
        "trust root must bind the stored agent-reviewer nonce"
    );

    let protocol_bytes = fs::read(protocol_evidence.path()).expect("read protocol evidence");
    let protocol: Value = serde_json::from_slice(&protocol_bytes).expect("protocol evidence JSON");
    assert_eq!(protocol["schema_version"], 1);
    assert_eq!(protocol["protocol"], "ctrader-open-api");
    assert_eq!(
        protocol["quote_observations_sha256"],
        json!([generated.quote_observations()[0].sha256().to_owned()])
    );
    assert_eq!(protocol["reviewer"], AGENT_REVIEWER_IDENTITY_V1);

    for (label, path) in [
        ("capture plan", capture_plan.path()),
        ("review record", review_record.path()),
        ("protocol evidence", protocol_evidence.path()),
        ("trust root", trust_root.path()),
        ("root verification", root_verification.path()),
        ("window binding", window_binding.path()),
    ] {
        assert_no_permit_content(
            &fs::read(path).unwrap_or_else(|error| panic!("read {label}: {error}")),
        );
    }

    // The real acceptance proof: the emitted observations + replay rules load
    // through the unchanged history ingress with the exact acquisition binding.
    let loaded = load_emitted_synchronization(
        0,
        PRIMARY_SYMBOL_ID,
        "EURUSD",
        EUR_ASSET_ID,
        "EUR",
        USD_ASSET_ID,
        "USD",
        &generated,
    );
    assert_eq!(loaded.ordinal(), 0);
    assert_eq!(loaded.account_id(), ACCOUNT_ID);
    assert_eq!(loaded.symbol_id(), PRIMARY_SYMBOL_ID);
    assert_eq!(loaded.window(), fixture_window());
    assert_eq!(loaded.raw_observation_count(), 2);
    assert_eq!(loaded.decoded_replay_rule_count(), 1);
    assert_eq!(
        loaded.review_identity_sha256(),
        ReviewedQuoteReplayRuleIdentityV2::new(
            review_record.sha256().to_owned(),
            protocol_evidence.sha256().to_owned(),
            generated.quote_observations()[0].sha256().to_owned(),
        )
        .expect("identity")
        .identity_sha256()
    );
}

#[test]
fn generator_emits_exact_settlement_conversion_route_and_second_synchronization() {
    let leg_spec = format!(
        "{USD_ASSET_ID}|USD|{EUR_ASSET_ID}|EUR|{PRIMARY_SYMBOL_ID}|EURUSD|{EUR_ASSET_ID}|EUR|{USD_ASSET_ID}|USD"
    );
    let legs = [leg_spec.as_str()];
    // Order independence: the ask page is supplied before the bid page and the
    // generator must still bind sides from the archived cursor, not CLI order.
    let pages = [ProbePageContent::ask(), ProbePageContent::bid()];
    let fixture = Fixture::with_pages(EUR_ASSET_ID, "EUR", &legs, &pages);
    let generated = fixture.generate().expect("agent-reviewed input generation");
    let capture_plan = generated.capture_plan();
    let review_record = generated.review_record();
    let protocol_evidence = generated.protocol_evidence();
    assert_eq!(generated.quote_observations().len(), 2);
    assert_eq!(generated.reviewed_replay_rules().len(), 2);
    for artifact in generated
        .quote_observations()
        .iter()
        .chain(generated.reviewed_replay_rules())
    {
        assert_evidence_artifact(artifact);
    }

    let plan: MirrorPlan = read_json_mirror(capture_plan.path());
    assert_eq!(plan.account_asset.asset_id, EUR_ASSET_ID);
    assert_eq!(plan.account_asset.asset_name, "EUR");
    assert_eq!(plan.conversion_routes.len(), 1);
    let route = &plan.conversion_routes[0];
    assert_eq!(route.purpose, "primary_pnl_settlement");
    assert_eq!(route.from_asset_id, USD_ASSET_ID);
    assert_eq!(route.from_asset_name, "USD");
    assert_eq!(route.to_asset_id, EUR_ASSET_ID);
    assert_eq!(route.to_asset_name, "EUR");
    assert_eq!(route.legs.len(), 1);
    let leg = &route.legs[0];
    assert_eq!(leg.from_asset_id, USD_ASSET_ID);
    assert_eq!(leg.from_asset_name, "USD");
    assert_eq!(leg.to_asset_id, EUR_ASSET_ID);
    assert_eq!(leg.to_asset_name, "EUR");
    assert_exact_instrument(
        &leg.instrument,
        PRIMARY_SYMBOL_ID,
        "EURUSD",
        EUR_ASSET_ID,
        "EUR",
        USD_ASSET_ID,
        "USD",
    );
    assert_eq!(plan.synchronizations.len(), 2);
    assert_eq!(
        plan.synchronizations[0].instrument.symbol_id,
        PRIMARY_SYMBOL_ID
    );
    let second_sync = &plan.synchronizations[1];
    assert_eq!(second_sync.ordinal, 1);
    assert_eq!(second_sync.account_id, ACCOUNT_ID);
    assert_exact_instrument(
        &second_sync.instrument,
        PRIMARY_SYMBOL_ID,
        "EURUSD",
        EUR_ASSET_ID,
        "EUR",
        USD_ASSET_ID,
        "USD",
    );
    assert_exact_window(&second_sync.window);
    assert_eq!(
        second_sync.quote_observations_sha256,
        generated.quote_observations()[1].sha256().to_owned()
    );
    assert_eq!(
        second_sync.reviewed_replay_rules_sha256,
        generated.reviewed_replay_rules()[1].sha256().to_owned()
    );
    expect_review_identity(
        second_sync,
        review_record.sha256(),
        protocol_evidence.sha256(),
        generated.quote_observations()[1].sha256(),
    );

    let review: MirrorReviewRecord = read_json_mirror(review_record.path());
    assert_eq!(review.synchronizations.len(), 2);
    let second_review_sync = &review.synchronizations[1];
    assert_eq!(second_review_sync.ordinal, 1);
    assert_eq!(
        second_review_sync.instrument.symbol_id,
        PRIMARY_SYMBOL_ID
    );
    assert_eq!(
        second_review_sync.quote_observations_sha256,
        generated.quote_observations()[1].sha256().to_owned()
    );
    assert_eq!(
        second_review_sync.reviewed_replay_rules_sha256,
        generated.reviewed_replay_rules()[1].sha256().to_owned()
    );

    // Both synchronizations must load through the real ingress in caller order.
    let mut sources = Vec::new();
    for ordinal in 0..2_u32 {
        let ordinal_index = usize::try_from(ordinal).expect("ordinal index");
        let review_identity = ReviewedQuoteReplayRuleIdentityV2::new(
            review_record.sha256().to_owned(),
            protocol_evidence.sha256().to_owned(),
            generated.quote_observations()[ordinal_index].sha256().to_owned(),
        )
        .expect("reviewed replay-rule identity");
        let binding = BrokerTruthReviewedSynchronizationBindingV1::new(
            ordinal,
            ACCOUNT_ID,
            PRIMARY_SYMBOL_ID,
            fixture_window(),
            review_identity,
            generated.reviewed_replay_rules()[ordinal_index].sha256().to_owned(),
        )
        .expect("exact reviewed synchronization binding");
        let instrument = ExactQuoteInstrumentV2::new(
            PRIMARY_SYMBOL_ID,
            "EURUSD",
            EUR_ASSET_ID,
            "EUR",
            USD_ASSET_ID,
            "USD",
        )
        .expect("exact fixture instrument");
        sources.push(
            ReviewedCTraderQuoteSynchronizationSourceV2::new(
                binding,
                instrument,
                generated.quote_observations()[ordinal_index].path().to_path_buf(),
                generated.reviewed_replay_rules()[ordinal_index]
                    .path()
                    .to_path_buf(),
            )
            .expect("explicit evidence source"),
        );
    }
    let loaded = load_reviewed_ctrader_quote_synchronizations_v2(sources)
        .expect("history ingress must accept both emitted synchronizations");
    assert_eq!(loaded.len(), 2);
    for (ordinal, synchronization) in loaded.iter().enumerate() {
        assert_eq!(synchronization.ordinal(), ordinal as u32);
        assert_eq!(synchronization.account_id(), ACCOUNT_ID);
        assert_eq!(synchronization.symbol_id(), PRIMARY_SYMBOL_ID);
        assert_eq!(synchronization.window(), fixture_window());
        assert_eq!(synchronization.raw_observation_count(), 2);
        assert_eq!(synchronization.decoded_replay_rule_count(), 1);
    }
}

#[test]
fn generator_refuses_multi_instrument_synchronizations_without_per_instrument_pages() {
    let leg_spec = format!(
        "{USD_ASSET_ID}|USD|{JPY_ASSET_ID}|JPY|{CONVERSION_SYMBOL_ID}|USDJPY|{USD_ASSET_ID}|USD|{JPY_ASSET_ID}|JPY"
    );
    let legs = [leg_spec.as_str()];
    let fixture = Fixture::new(JPY_ASSET_ID, "JPY", &legs);
    expect_generation_error(
        fixture.args,
        BrokerTruthAcquisitionPreflightErrorCodeV1::SynchronizationMismatch,
    );
}

#[test]
fn generator_refuses_tampered_or_invalid_quote_observation_pages() {
    let cases: [(&str, fn(&mut [ProbePageContent; 2]), BrokerTruthAcquisitionPreflightErrorCodeV1); 7] = [
        (
            "raw digest mismatch",
            |pages| {
                pages[0].raw_response_sha256 = Some("0".repeat(64));
            },
            BrokerTruthAcquisitionPreflightErrorCodeV1::ArtifactDigestMismatch,
        ),
        (
            "duplicate bid sides",
            |pages| {
                pages[1].sequence = 1;
                pages[1].side = BID_SIDE;
                pages[1].client_msg_id = "reviewed-fixture-1-side-1".to_owned();
                pages[1].envelope = json!({
                    "clientMsgId": "reviewed-fixture-1-side-1",
                    "payloadType": TICK_RESPONSE_PAYLOAD_TYPE,
                    "payload": {"ctidTraderAccountId": ACCOUNT_ID, "hasMore": false, "tickData": []},
                });
            },
            BrokerTruthAcquisitionPreflightErrorCodeV1::SynchronizationMismatch,
        ),
        (
            "wrong payload type",
            |pages| {
                pages[0].envelope = json!({
                    "clientMsgId": pages[0].client_msg_id,
                    "payloadType": 9_999,
                    "payload": {"ctidTraderAccountId": ACCOUNT_ID, "hasMore": false, "tickData": []},
                });
            },
            BrokerTruthAcquisitionPreflightErrorCodeV1::SynchronizationMismatch,
        ),
        (
            "detached envelope client message id",
            |pages| {
                pages[0].envelope = json!({
                    "clientMsgId": "another-client-message-id",
                    "payloadType": TICK_RESPONSE_PAYLOAD_TYPE,
                    "payload": {"ctidTraderAccountId": ACCOUNT_ID, "hasMore": false, "tickData": []},
                });
            },
            BrokerTruthAcquisitionPreflightErrorCodeV1::SynchronizationMismatch,
        ),
        (
            "detached envelope account",
            |pages| {
                pages[0].envelope = json!({
                    "clientMsgId": pages[0].client_msg_id,
                    "payloadType": TICK_RESPONSE_PAYLOAD_TYPE,
                    "payload": {"ctidTraderAccountId": ACCOUNT_ID + 1, "hasMore": false, "tickData": []},
                });
            },
            BrokerTruthAcquisitionPreflightErrorCodeV1::SynchronizationMismatch,
        ),
        (
            "invalid archived side",
            |pages| {
                pages[0].side = 3;
            },
            BrokerTruthAcquisitionPreflightErrorCodeV1::SynchronizationMismatch,
        ),
        (
            "unknown archive schema",
            |pages| {
                pages[0].schema = "not-the-tick-archive-schema";
            },
            BrokerTruthAcquisitionPreflightErrorCodeV1::SynchronizationMismatch,
        ),
    ];
    for (_label, tamper, expected) in cases {
        let mut pages = [ProbePageContent::bid(), ProbePageContent::ask()];
        tamper(&mut pages);
        let fixture = Fixture::with_pages(USD_ASSET_ID, "USD", &[], &pages);
        expect_generation_error(fixture.args, expected);
    }
}

#[test]
fn generator_fails_closed_on_digest_mismatch_missing_path_and_bad_window() {
    let fixture = Fixture::new(USD_ASSET_ID, "USD", &[]);
    let wrong_digest = "0".repeat(64);
    expect_generation_error(
        with_flag_value(
            &fixture.args,
            "--canonical-receipt-sha256",
            wrong_digest.clone(),
        ),
        BrokerTruthAcquisitionPreflightErrorCodeV1::ArtifactDigestMismatch,
    );
    expect_generation_error(
        with_flag_value(&fixture.args, "--quote-observations-sha256", wrong_digest),
        BrokerTruthAcquisitionPreflightErrorCodeV1::ArtifactDigestMismatch,
    );
    let missing_receipt = fixture
        .input_root
        .join("missing-canonical-search-input-receipt.json")
        .to_string_lossy()
        .into_owned();
    expect_generation_error(
        with_flag_value(&fixture.args, "--canonical-receipt", missing_receipt),
        BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath,
    );
    expect_generation_error(
        with_flag_value(&fixture.args, "--from-ms", (FROM_MS + 60_000).to_string()),
        BrokerTruthAcquisitionPreflightErrorCodeV1::WindowMismatch,
    );
    expect_generation_error(
        with_flag_value(&fixture.args, "--account-id", (ACCOUNT_ID + 1).to_string()),
        BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalAuthorityMismatch,
    );
    expect_generation_error(
        with_flag_value(&fixture.args, "--environment", "live".to_owned()),
        BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalAuthorityMismatch,
    );

    // Pre-existing output file is refused instead of overwritten.
    let fixture = Fixture::new(USD_ASSET_ID, "USD", &[]);
    let preexisting = fixture.out_dir.join("broker-truth-capture-plan.json");
    fs::write(&preexisting, b"pre-existing bytes").expect("write pre-existing output file");
    let error = fixture
        .generate()
        .expect_err("existing output must be refused");
    assert_eq!(
        error.code(),
        BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath
    );

    // Pre-existing evidence output is also refused instead of overwritten.
    let fixture = Fixture::new(USD_ASSET_ID, "USD", &[]);
    let preexisting = fixture
        .out_dir
        .join("reviewed-quote-replay-rules-000.vortex");
    fs::write(&preexisting, b"pre-existing bytes").expect("write pre-existing evidence output");
    let error = fixture
        .generate()
        .expect_err("existing evidence output must be refused");
    assert_eq!(
        error.code(),
        BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath
    );
}

#[test]
fn generator_rejects_malformed_arguments_secret_flags_and_relative_paths() {
    let fixture = Fixture::new(USD_ASSET_ID, "USD", &[]);

    let secret_flags = vec![
        "--client-secret",
        "--access-token",
        "--refresh-token",
        "--credentials-path",
        "--api-key",
    ];
    for secret_flag in secret_flags {
        let mut args = fixture.args.clone();
        args.push(secret_flag.to_owned());
        args.push("do-not-read-or-print-me".to_owned());
        assert_eq!(
            generation_parse_error(args),
            BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments,
            "secret-bearing flag {secret_flag} must not exist"
        );
    }

    let mut args = fixture.args.clone();
    args.push("--discover-sibling-authority".to_owned());
    assert_eq!(
        generation_parse_error(args),
        BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments
    );

    let mut args = fixture.args.clone();
    args[2] = "relative-reviewed-inputs".to_owned();
    let parsed = ReviewedInputGenerationArgsV1::try_parse_from(args)
        .expect("syntactically valid generation arguments");
    let error = generate_reviewed_inputs_v1(parsed).expect_err("relative out dir must be refused");
    assert_eq!(
        error.code(),
        BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath
    );

    let mut missing = fixture.args.clone();
    let index = missing
        .iter()
        .position(|argument| argument == "--canonical-receipt")
        .expect("fixture receipt path flag");
    missing.drain(index..=index + 1);
    assert_eq!(
        generation_parse_error(missing),
        BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments
    );

    let mut malformed = fixture.args.clone();
    malformed.push("--settlement-leg".to_owned());
    malformed.push("2|USD".to_owned());
    assert_eq!(
        generation_parse_error(malformed),
        BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments
    );

    let mut count = fixture.args.clone();
    count.extend([
        "--quote-observations".to_owned(),
        fixture.observations[0].0.to_string_lossy().into_owned(),
    ]);
    assert_eq!(
        generation_parse_error(count),
        BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments,
        "unbalanced observation evidence pairs must be refused"
    );
}

#[test]
fn generator_source_has_no_permit_capability_or_ambient_authority_path() {
    let source = include_str!("../src/reviewed_input_generation_v1.rs");
    assert!(
        source.contains(AGENT_REVIEWER_IDENTITY_V1),
        "generator source must carry the exact agent-reviewer identity"
    );
    for forbidden in [
        "std::env",
        "ProductionCTraderOpenApiSession",
        "capture_production_broker_financial_truth_v2",
        "BrokerFinancialTruthPermit",
        "install_broker_financial_truth",
        "current_broker_financial_truth_capability_v1",
        "read_current_manifest",
        "open_current",
        "load_exact_production_broker_truth_credentials_v2",
        "read_dir(",
        "glob(",
        "Default::default",
        "fallback",
    ] {
        assert!(
            !source.contains(forbidden),
            "generator source must not contain forbidden authority path {forbidden}"
        );
    }
}
