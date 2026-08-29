use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use neoethos_broker_history::bootstrap_writer::{
    BrokerTrendbarStreamRequest, publish_broker_trendbar_chunks,
};
use neoethos_broker_history::broker_truth_capture::ExactQuoteInstrumentV2;
use neoethos_broker_history::{BrokerEnvironment, ProductionBrokerTruthCancellationV2};
use neoethos_broker_truth::{
    BrokerFinancialTruthArtifactSourceV1, BrokerFinancialTruthBindingV1,
    BrokerFinancialTruthBundleManifestV2, BrokerFinancialTruthBundleReceiptV2,
    BrokerFinancialTruthBundleStoreV1, BrokerFinancialTruthVortexSchemaV1,
    BrokerTruthAcquisitionPromotionEligibilityV1, BrokerTruthAcquisitionSemanticStatusV1,
    BrokerTruthAcquisitionStoreV1, EvidenceWindowV1, ExactBrokerRequestChunkV2,
    ExactBrokerRequestPageV2, ExactCapturedEvidencePairV1, ExactConversionRouteEvidenceV2,
    ExactDealReconciliationEvidenceV2, ExactQuoteSideEvidenceV2, ExactSymbolContractEvidenceV2,
    ImmutableVortexArtifactV1, QuoteSideV1, ReviewedQuoteReplayRuleEvidenceV2,
    ReviewedQuoteReplayRuleIdentityV2, SynchronizedBidAskEvidenceV2,
};
use neoethos_data::core::dataset_manifest::PublishResult;
use neoethos_data::{
    BarTimestampConvention, CTraderEnvironment, CanonicalDatasetIdentity, CanonicalOhlcvChunk,
    CanonicalTimeframe, CanonicalVolumeChunk, SelectedDatasetGenerationV1,
};
use neoethos_search::{
    CanonicalSearchArtifactScopeV2, CanonicalSearchInputReceiptV2, CanonicalSearchWindowRoleV1,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use vortex_array::IntoArray;
use vortex_array::arrays::{PrimitiveArray, StructArray, VarBinArray};

use crate::acquisition_orchestration_v1::{
    BrokerTruthCaptureInvocationV1, BrokerTruthCaptureRunnerFailureV1, BrokerTruthCaptureRunnerV1,
    execute_prepared_acquisition_with_runner_v1,
};
use crate::{
    BrokerTruthAcquisitionArgsV1, BrokerTruthAcquisitionOrchestrationErrorCodeV1,
    BrokerTruthAcquisitionOrchestrationErrorV1, BrokerTruthAcquisitionOutcomeV1,
    PreparedBrokerTruthAcquisitionV1, execute_prepared_acquisition_v1, prepare_acquisition_v1,
};

const ACCOUNT_ID: i64 = 7_001;
const SYMBOL_ID: i64 = 14;
const EUR_ASSET_ID: i64 = 1;
const USD_ASSET_ID: i64 = 2;
const FROM_MS: i64 = 1_700_000_040_000;
const LAST_BAR_MS: i64 = FROM_MS + 120_000;
const TO_MS_EXCLUSIVE: i64 = FROM_MS + 180_000;
const PAYLOAD_TYPE: u32 = 2_146;
const WINDOW_POLICY_ID: &str = "reviewed-half-open-m1-v1";
const SERVER: &str = "demo.ctraderapi.com";

#[derive(Clone)]
struct FrozenInput {
    path: PathBuf,
    sha256: String,
}

struct Fixture {
    _temp: TempDir,
    prepared: PreparedBrokerTruthAcquisitionV1,
    store_root: PathBuf,
    bft2_source_root: PathBuf,
    observations_path: PathBuf,
    trust_root_path: PathBuf,
    binding: BrokerFinancialTruthBindingV1,
    instrument: ExactQuoteInstrumentV2,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn digest(byte: u8) -> String {
    format!("{byte:02x}").repeat(32)
}

fn write_bytes(directory: &Path, name: &str, bytes: &[u8]) -> FrozenInput {
    let path = directory.join(name);
    fs::write(&path, bytes).expect("write frozen orchestration input");
    FrozenInput {
        path,
        sha256: sha256(bytes),
    }
}

fn write_json(directory: &Path, name: &str, value: &Value) -> FrozenInput {
    write_bytes(
        directory,
        name,
        &serde_json::to_vec(value).expect("encode frozen orchestration JSON"),
    )
}

fn window() -> EvidenceWindowV1 {
    EvidenceWindowV1::new(FROM_MS, TO_MS_EXCLUSIVE).expect("valid half-open fixture window")
}

fn window_json() -> Value {
    json!({
        "from_unix_ms_inclusive": FROM_MS,
        "to_unix_ms_exclusive": TO_MS_EXCLUSIVE
    })
}

fn primary_identity() -> CanonicalDatasetIdentity {
    CanonicalDatasetIdentity::ctrader(
        CTraderEnvironment::Demo,
        SERVER,
        ACCOUNT_ID,
        SYMBOL_ID,
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("valid exact cTrader dataset identity")
}

fn exact_instrument_json() -> Value {
    json!({
        "symbol_id": SYMBOL_ID,
        "symbol_name": "EURUSD",
        "base_asset_id": EUR_ASSET_ID,
        "base_asset_name": "EUR",
        "quote_asset_id": USD_ASSET_ID,
        "quote_asset_name": "USD"
    })
}

fn publish_primary_generation(
    data_root: &Path,
    identity: &CanonicalDatasetIdentity,
) -> PublishResult {
    publish_broker_trendbar_chunks(BrokerTrendbarStreamRequest {
        configured_root: data_root,
        identity,
        expected_generation: None,
        requested_from_ms: FROM_MS,
        requested_to_ms: TO_MS_EXCLUSIVE,
        retrieved_unix_ms: 1_800_000_000_000,
        returned_from_ms: FROM_MS,
        returned_to_ms: LAST_BAR_MS,
        row_count: 3,
        chunks: vec![Ok(CanonicalOhlcvChunk {
            timestamp_ms: vec![FROM_MS, FROM_MS + 60_000, LAST_BAR_MS],
            open: vec![1.10, 1.11, 1.12],
            high: vec![1.12, 1.13, 1.14],
            low: vec![1.09, 1.10, 1.11],
            close: vec![1.11, 1.12, 1.13],
            volume: CanonicalVolumeChunk::Int64(vec![10, 11, 12]),
        })],
    })
    .expect("publish exact canonical cTrader generation")
}

fn side_code(side: QuoteSideV1) -> u8 {
    match side {
        QuoteSideV1::Bid => 0,
        QuoteSideV1::Ask => 1,
    }
}

fn reviewed_evidence_array(
    rows: &[(u64, u8, Option<QuoteSideV1>, String, String)],
) -> vortex_array::ArrayRef {
    let fields: Vec<(&str, vortex_array::ArrayRef)> = vec![
        (
            "sequence",
            PrimitiveArray::from_iter(rows.iter().map(|row| row.0)).into_array(),
        ),
        (
            "account_id",
            PrimitiveArray::from_iter(rows.iter().map(|_| ACCOUNT_ID)).into_array(),
        ),
        (
            "evidence_kind",
            PrimitiveArray::from_iter(rows.iter().map(|row| row.1)).into_array(),
        ),
        (
            "has_symbol_id",
            PrimitiveArray::from_iter(rows.iter().map(|_| 1_u8)).into_array(),
        ),
        (
            "symbol_id",
            PrimitiveArray::from_iter(rows.iter().map(|_| SYMBOL_ID)).into_array(),
        ),
        (
            "has_quote_side",
            PrimitiveArray::from_iter(rows.iter().map(|row| u8::from(row.2.is_some())))
                .into_array(),
        ),
        (
            "quote_side",
            PrimitiveArray::from_iter(rows.iter().map(|row| row.2.map_or(0, side_code)))
                .into_array(),
        ),
        (
            "has_requested_window",
            PrimitiveArray::from_iter(rows.iter().map(|_| 1_u8)).into_array(),
        ),
        (
            "requested_from_unix_ms_inclusive",
            PrimitiveArray::from_iter(rows.iter().map(|_| FROM_MS)).into_array(),
        ),
        (
            "requested_to_unix_ms_exclusive",
            PrimitiveArray::from_iter(rows.iter().map(|_| TO_MS_EXCLUSIVE)).into_array(),
        ),
        (
            "client_msg_id",
            VarBinArray::from(rows.iter().map(|row| row.3.as_str()).collect::<Vec<_>>())
                .into_array(),
        ),
        (
            "payload_type",
            PrimitiveArray::from_iter(rows.iter().map(|_| PAYLOAD_TYPE)).into_array(),
        ),
        (
            "payload_json",
            VarBinArray::from(rows.iter().map(|row| row.4.as_str()).collect::<Vec<_>>())
                .into_array(),
        ),
    ];
    StructArray::from_fields(&fields)
        .expect("exact reviewed evidence struct")
        .into_array()
}

fn write_reviewed_pair(input_root: &Path) -> (FrozenInput, FrozenInput) {
    let bid_client_id = "task5b-reviewed-sync-0".to_owned();
    let ask_client_id = "task5b-reviewed-sync-1".to_owned();
    let observations = [
        (
            0,
            0,
            Some(QuoteSideV1::Bid),
            bid_client_id.clone(),
            json!({
                "clientMsgId": bid_client_id,
                "payload": {
                    "ctidTraderAccountId": ACCOUNT_ID,
                    "hasMore": false,
                    "tickData": []
                },
                "payloadType": PAYLOAD_TYPE
            })
            .to_string(),
        ),
        (
            1,
            0,
            Some(QuoteSideV1::Ask),
            ask_client_id.clone(),
            json!({
                "clientMsgId": ask_client_id,
                "payload": {
                    "ctidTraderAccountId": ACCOUNT_ID,
                    "hasMore": false,
                    "tickData": []
                },
                "payloadType": PAYLOAD_TYPE
            })
            .to_string(),
        ),
    ];
    let rules = [(
        0,
        1,
        None,
        "task5b-reviewed-sync-0".to_owned(),
        json!({
            "accountId": ACCOUNT_ID,
            "askClientMsgId": "task5b-reviewed-sync-1",
            "bidClientMsgId": "task5b-reviewed-sync-0",
            "symbolId": SYMBOL_ID,
            "window": {
                "fromUnixMsInclusive": FROM_MS,
                "toUnixMsExclusive": TO_MS_EXCLUSIVE
            }
        })
        .to_string(),
    )];
    let observations_path = input_root.join("quote-session-observations-000.vortex");
    let rules_path = input_root.join("reviewed-quote-replay-rules-000.vortex");
    neoethos_data::core::vortex_io::write_vortex_array(
        &observations_path,
        reviewed_evidence_array(&observations),
    )
    .expect("write exact reviewed observation Vortex");
    neoethos_data::core::vortex_io::write_vortex_array(
        &rules_path,
        reviewed_evidence_array(&rules),
    )
    .expect("write exact reviewed rule Vortex");
    let observation_bytes = fs::read(&observations_path).expect("read exact observation Vortex");
    let rules_bytes = fs::read(&rules_path).expect("read exact rule Vortex");
    (
        FrozenInput {
            path: observations_path,
            sha256: sha256(&observation_bytes),
        },
        FrozenInput {
            path: rules_path,
            sha256: sha256(&rules_bytes),
        },
    )
}

impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().expect("temporary Task5b fixture");
        let data_root = temp.path().join("canonical-data");
        let input_root = temp.path().join("immutable-inputs");
        let work_parent = temp.path().join("capture-work");
        let store_root = temp.path().join("broker-truth-store");
        let bft2_source_root = temp.path().join("bft2-sources");
        fs::create_dir_all(&input_root).expect("create immutable input root");
        fs::create_dir_all(&work_parent).expect("create explicit work parent");
        fs::create_dir_all(&store_root).expect("create explicit store root");
        fs::create_dir_all(&bft2_source_root).expect("create BFT2 source root");

        let identity = primary_identity();
        let publication = publish_primary_generation(&data_root, &identity);
        let manifest = publication.manifest();
        let selected = SelectedDatasetGenerationV1::from_manifest(manifest)
            .expect("select exact canonical generation");

        let receipt_value = json!({
            "schema_version": 2,
            "anchor_dataset_identity": identity.to_path_component(),
            "feature_plan_identity": "11".repeat(32),
            "feature_provenance_identity": "22".repeat(32),
            "feature_content_sha256": "33".repeat(32),
            "feature_execution": {
                "schema_version": 1,
                "compute_policy": "auto",
                "vector_ta_math_authority": "neoethos.vector-ta.cpu-f64-exact-bits.v1",
                "selected_lane": "cpu_scalar"
            },
            "source_bindings": [{
                "source_node_id": "anchor",
                "dataset_identity": identity.to_path_component(),
                "manifest_schema_id": manifest.schema_id(),
                "manifest_sha256": manifest.manifest_binding_sha256(),
                "generation_id": manifest.generation_id(),
                "vortex_sha256": manifest.vortex_sha256(),
                "bar_timestamp_convention": identity.bar_timestamp_convention().to_string(),
                "segments": [{
                    "row_start": 0,
                    "row_end": 3,
                    "timestamp_start_ms": FROM_MS,
                    "timestamp_end_ms": LAST_BAR_MS
                }]
            }]
        });
        let typed_receipt = CanonicalSearchInputReceiptV2::from_json_bytes(
            &serde_json::to_vec(&receipt_value).expect("encode receipt fixture"),
        )
        .expect("strict canonical receipt fixture");
        let receipt_identity = typed_receipt
            .identity_sha256()
            .expect("canonical receipt identity");
        let receipt = write_bytes(
            &input_root,
            "canonical-search-input-receipt.json",
            &typed_receipt
                .to_json_bytes()
                .expect("canonical receipt JSON"),
        );

        let typed_scope = CanonicalSearchArtifactScopeV2::for_entire_receipt(
            CanonicalSearchWindowRoleV1::Holdout,
            typed_receipt.clone(),
        )
        .expect("canonical holdout scope fixture");
        let scope_identity = typed_scope
            .identity_sha256()
            .expect("canonical scope identity");
        let scope = write_bytes(
            &input_root,
            "canonical-search-artifact-scope.json",
            &typed_scope.to_json_bytes().expect("canonical scope JSON"),
        );
        let selected_value: Value = serde_json::from_slice(
            &selected
                .to_json_bytes()
                .expect("selected generation receipt JSON"),
        )
        .expect("selected generation JSON value");
        let root_verification = write_json(
            &input_root,
            "canonical-root-verification.json",
            &json!({
                "schema_version": 1,
                "canonical_search_input_receipt_identity_sha256": receipt_identity,
                "opened_generations": [{
                    "source_node_id": "anchor",
                    "selected_generation": selected_value,
                    "manifest_schema_id": manifest.schema_id(),
                    "vortex_sha256": manifest.vortex_sha256()
                }]
            }),
        );
        let window_binding = write_json(
            &input_root,
            "canonical-scope-window-binding.json",
            &json!({
                "schema_version": 1,
                "canonical_search_input_receipt_identity_sha256": receipt_identity,
                "canonical_search_artifact_scope_identity_sha256": scope_identity,
                "role": "holdout",
                "row_start": 0,
                "row_end": 3,
                "timestamp_start_ms": FROM_MS,
                "timestamp_end_ms": LAST_BAR_MS,
                "evidence_window": window_json(),
                "window_policy_id": WINDOW_POLICY_ID
            }),
        );
        let protocol = write_bytes(
            &input_root,
            "ctrader-protocol-evidence.json",
            br#"{"schema_version":1,"protocol":"ctrader-open-api","reviewed":"2026-08-23"}"#,
        );
        let trust_root = write_bytes(
            &input_root,
            "quote-review-trust-root.pub",
            b"uninstalled-test-review-key-v1\n",
        );
        let (observations, rules) = write_reviewed_pair(&input_root);
        let primary = exact_instrument_json();
        let review_record = write_json(
            &input_root,
            "quote-replay-review-record.json",
            &json!({
                "schema_version": 1,
                "canonical_search_input_receipt_identity_sha256": receipt_identity,
                "canonical_search_artifact_scope_identity_sha256": scope_identity,
                "canonical_scope_window_binding_sha256": window_binding.sha256,
                "trust_root_sha256": trust_root.sha256,
                "protocol_evidence_sha256": protocol.sha256,
                "window_policy_id": WINDOW_POLICY_ID,
                "environment": "demo",
                "server": SERVER,
                "account_id": ACCOUNT_ID,
                "window": window_json(),
                "synchronizations": [{
                    "ordinal": 0,
                    "instrument": primary,
                    "quote_observations_sha256": observations.sha256,
                    "reviewed_replay_rules_sha256": rules.sha256
                }]
            }),
        );
        let review_identity = ReviewedQuoteReplayRuleIdentityV2::new(
            review_record.sha256.clone(),
            protocol.sha256.clone(),
            observations.sha256.clone(),
        )
        .expect("exact reviewed replay-rule identity");
        let capture_plan = write_json(
            &input_root,
            "broker-truth-capture-plan.json",
            &json!({
                "schema_version": 1,
                "canonical_search_input_receipt_identity_sha256": receipt_identity,
                "canonical_search_artifact_scope_identity_sha256": scope_identity,
                "canonical_root_verification_sha256": root_verification.sha256,
                "canonical_scope_window_binding_sha256": window_binding.sha256,
                "review_record_sha256": review_record.sha256,
                "protocol_evidence_sha256": protocol.sha256,
                "trust_root_sha256": trust_root.sha256,
                "environment": "demo",
                "server": SERVER,
                "account_id": ACCOUNT_ID,
                "window": window_json(),
                "primary_instrument": primary,
                "account_asset": {
                    "asset_id": USD_ASSET_ID,
                    "asset_name": "USD"
                },
                "conversion_routes": [{
                    "purpose": "primary_pnl_settlement",
                    "from_asset_id": USD_ASSET_ID,
                    "from_asset_name": "USD",
                    "to_asset_id": USD_ASSET_ID,
                    "to_asset_name": "USD",
                    "legs": []
                }],
                "synchronizations": [{
                    "ordinal": 0,
                    "account_id": ACCOUNT_ID,
                    "instrument": primary,
                    "window": window_json(),
                    "quote_observations_sha256": observations.sha256,
                    "reviewed_replay_rules_sha256": rules.sha256,
                    "review_identity_sha256": review_identity.identity_sha256()
                }]
            }),
        );
        let args = vec![
            "broker-truth-acquire".to_owned(),
            "--data-root".to_owned(),
            data_root.to_string_lossy().into_owned(),
            "--canonical-receipt".to_owned(),
            receipt.path.to_string_lossy().into_owned(),
            "--canonical-receipt-sha256".to_owned(),
            receipt.sha256,
            "--canonical-scope".to_owned(),
            scope.path.to_string_lossy().into_owned(),
            "--canonical-scope-sha256".to_owned(),
            scope.sha256,
            "--canonical-root-verification".to_owned(),
            root_verification.path.to_string_lossy().into_owned(),
            "--canonical-root-verification-sha256".to_owned(),
            root_verification.sha256,
            "--canonical-scope-window-binding".to_owned(),
            window_binding.path.to_string_lossy().into_owned(),
            "--canonical-scope-window-binding-sha256".to_owned(),
            window_binding.sha256,
            "--capture-plan".to_owned(),
            capture_plan.path.to_string_lossy().into_owned(),
            "--capture-plan-sha256".to_owned(),
            capture_plan.sha256,
            "--review-record".to_owned(),
            review_record.path.to_string_lossy().into_owned(),
            "--review-record-sha256".to_owned(),
            review_record.sha256,
            "--protocol-evidence".to_owned(),
            protocol.path.to_string_lossy().into_owned(),
            "--protocol-evidence-sha256".to_owned(),
            protocol.sha256,
            "--trust-root".to_owned(),
            trust_root.path.to_string_lossy().into_owned(),
            "--trust-root-sha256".to_owned(),
            trust_root.sha256,
            "--quote-observations".to_owned(),
            observations.path.to_string_lossy().into_owned(),
            "--quote-observations-sha256".to_owned(),
            observations.sha256,
            "--reviewed-replay-rules".to_owned(),
            rules.path.to_string_lossy().into_owned(),
            "--reviewed-replay-rules-sha256".to_owned(),
            rules.sha256,
            "--environment".to_owned(),
            "demo".to_owned(),
            "--account-id".to_owned(),
            ACCOUNT_ID.to_string(),
            "--from-ms".to_owned(),
            FROM_MS.to_string(),
            "--to-ms-exclusive".to_owned(),
            TO_MS_EXCLUSIVE.to_string(),
            "--work-parent".to_owned(),
            work_parent.to_string_lossy().into_owned(),
            "--store-root".to_owned(),
            store_root.to_string_lossy().into_owned(),
        ];
        let parsed = BrokerTruthAcquisitionArgsV1::try_parse_from(args)
            .expect("parse exact Task5b preflight arguments");
        let prepared = prepare_acquisition_v1(parsed).expect("prepare exact Task5b authority");
        let binding = prepared.capture_request().binding().clone();
        let instrument = prepared.capture_request().primary_instrument().clone();

        Self {
            _temp: temp,
            prepared,
            store_root,
            bft2_source_root,
            observations_path: observations.path,
            trust_root_path: trust_root.path,
            binding,
            instrument,
        }
    }
}

fn write_bft2_artifact(
    source_root: &Path,
    sources: &mut Vec<BrokerFinancialTruthArtifactSourceV1>,
    relative_path: &str,
    schema: BrokerFinancialTruthVortexSchemaV1,
) -> ImmutableVortexArtifactV1 {
    let source_path = source_root.join(relative_path);
    fs::write(
        &source_path,
        format!("opaque integrity-only Task5b BFT2 fixture: {relative_path}"),
    )
    .expect("write integrity-only BFT2 artifact");
    let artifact = ImmutableVortexArtifactV1::from_file(relative_path, schema, 1, &source_path)
        .expect("describe integrity-only BFT2 artifact");
    sources.push(
        BrokerFinancialTruthArtifactSourceV1::new(relative_path, source_path)
            .expect("bind exact BFT2 source"),
    );
    artifact
}

fn bft2_quote_side(
    source_root: &Path,
    sources: &mut Vec<BrokerFinancialTruthArtifactSourceV1>,
    binding: &BrokerFinancialTruthBindingV1,
    instrument: &ExactQuoteInstrumentV2,
    side: QuoteSideV1,
) -> ExactQuoteSideEvidenceV2 {
    let label = match side {
        QuoteSideV1::Bid => "bid",
        QuoteSideV1::Ask => "ask",
    };
    let request_window = binding.evaluated_window();
    let page = ExactBrokerRequestPageV2::new(
        0,
        0,
        format!("primary-{label}-page"),
        request_window,
        Some(FROM_MS + 30_000),
        Some(FROM_MS + 30_000),
        1,
        false,
        None,
    )
    .expect("terminal exact quote page");
    let chunk = ExactBrokerRequestChunkV2::new(0, request_window, vec![page])
        .expect("exact quote request chunk");
    let raw = write_bft2_artifact(
        source_root,
        sources,
        &format!("primary-{label}-pages-raw.vortex"),
        BrokerFinancialTruthVortexSchemaV1::CTraderTickRequestPagesRawV2,
    );
    let decoded = write_bft2_artifact(
        source_root,
        sources,
        &format!("primary-{label}-ticks-decoded.vortex"),
        BrokerFinancialTruthVortexSchemaV1::CTraderTicksDecodedV2,
    );
    ExactQuoteSideEvidenceV2::new(
        side,
        instrument.symbol_id(),
        instrument.symbol_name(),
        instrument.base_asset_id(),
        instrument.quote_asset_id(),
        request_window,
        vec![chunk],
        raw,
        decoded,
    )
    .expect("exact quote-side evidence")
}

fn publish_integrity_only_bft2(
    store_root: &Path,
    source_root: &Path,
    binding: &BrokerFinancialTruthBindingV1,
    instrument: &ExactQuoteInstrumentV2,
) -> BrokerFinancialTruthBundleReceiptV2 {
    let mut sources = Vec::new();
    let bid = bft2_quote_side(
        source_root,
        &mut sources,
        binding,
        instrument,
        QuoteSideV1::Bid,
    );
    let ask = bft2_quote_side(
        source_root,
        &mut sources,
        binding,
        instrument,
        QuoteSideV1::Ask,
    );
    let observations = write_bft2_artifact(
        source_root,
        &mut sources,
        "quote-session-observations-raw.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderQuoteSessionObservationsRawV2,
    );
    let rules = write_bft2_artifact(
        source_root,
        &mut sources,
        "reviewed-quote-replay-rules-decoded.vortex",
        BrokerFinancialTruthVortexSchemaV1::CTraderReviewedQuoteReplayRulesDecodedV2,
    );
    let review_identity =
        ReviewedQuoteReplayRuleIdentityV2::new(digest(0xa1), digest(0xa2), observations.sha256())
            .expect("integrity-only BFT2 review identity");
    let replay = ReviewedQuoteReplayRuleEvidenceV2::new(review_identity, observations, rules)
        .expect("integrity-only replay evidence");
    let primary =
        SynchronizedBidAskEvidenceV2::new(bid, ask, replay).expect("synchronized primary quotes");
    let symbol_contracts = ExactSymbolContractEvidenceV2::new(
        write_bft2_artifact(
            source_root,
            &mut sources,
            "light-symbol-responses-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderLightSymbolResponsesRawV2,
        ),
        write_bft2_artifact(
            source_root,
            &mut sources,
            "full-symbol-responses-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderSymbolResponsesRawV2,
        ),
        write_bft2_artifact(
            source_root,
            &mut sources,
            "account-asset-responses-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderAccountAssetResponsesRawV2,
        ),
        write_bft2_artifact(
            source_root,
            &mut sources,
            "trader-account-responses-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderTraderAccountResponsesRawV2,
        ),
        write_bft2_artifact(
            source_root,
            &mut sources,
            "symbol-money-contracts-decoded.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderSymbolMoneyContractsDecodedV2,
        ),
    )
    .expect("exact symbol contracts");
    let pnl = ExactCapturedEvidencePairV1::new(
        write_bft2_artifact(
            source_root,
            &mut sources,
            "position-unrealized-pnl-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderUnrealizedPnlResponsesRawV2,
        ),
        write_bft2_artifact(
            source_root,
            &mut sources,
            "position-unrealized-pnl-decoded.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderUnrealizedPnlDecodedV2,
        ),
    );
    let deal_page = ExactBrokerRequestPageV2::new(
        0,
        0,
        "deal-page",
        binding.evaluated_window(),
        None,
        None,
        0,
        false,
        Some(100),
    )
    .expect("terminal empty DealList page");
    let deal_chunk = ExactBrokerRequestChunkV2::new(0, binding.evaluated_window(), vec![deal_page])
        .expect("exact DealList chunk");
    let close_deal = ExactDealReconciliationEvidenceV2::new(
        binding.evaluated_window(),
        deal_chunk,
        write_bft2_artifact(
            source_root,
            &mut sources,
            "reconcile-responses-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderReconcileResponsesRawV2,
        ),
        write_bft2_artifact(
            source_root,
            &mut sources,
            "deal-pages-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderDealPagesRawV2,
        ),
        write_bft2_artifact(
            source_root,
            &mut sources,
            "close-deal-reconciliation-decoded.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderCloseDealReconciliationDecodedV2,
        ),
    )
    .expect("exact close/deal evidence");
    let settlement = ExactConversionRouteEvidenceV2::new(
        "primary_pnl_settlement",
        binding.primary_quote_asset_id(),
        binding.primary_quote_asset_name(),
        binding.account_asset_id(),
        binding.account_asset_name(),
        Vec::new(),
    )
    .expect("identity settlement route");
    let manifest = BrokerFinancialTruthBundleManifestV2::new(
        binding.clone(),
        primary,
        vec![settlement],
        symbol_contracts,
        pnl,
        close_deal,
    )
    .expect("integrity-only BFT2 manifest");
    BrokerFinancialTruthBundleStoreV1::new(store_root)
        .publish_v2(&manifest, &sources)
        .expect("publish real immutable integrity-only BFT2 fixture")
}

enum SpyResult {
    Published(BrokerFinancialTruthBundleReceiptV2),
    Failed,
}

struct OfflineRunnerSpy {
    expected_store_root: PathBuf,
    expected_cancellation_address: usize,
    expected_binding: BrokerFinancialTruthBindingV1,
    result: SpyResult,
    calls: usize,
    saw_published_authority: bool,
}

impl OfflineRunnerSpy {
    fn new(
        expected_store_root: PathBuf,
        cancellation: &ProductionBrokerTruthCancellationV2,
        expected_binding: BrokerFinancialTruthBindingV1,
        result: SpyResult,
    ) -> Self {
        Self {
            expected_store_root,
            expected_cancellation_address: cancellation as *const _ as usize,
            expected_binding,
            result,
            calls: 0,
            saw_published_authority: false,
        }
    }
}

impl BrokerTruthCaptureRunnerV1 for OfflineRunnerSpy {
    fn capture(
        &mut self,
        invocation: BrokerTruthCaptureInvocationV1,
        cancellation: &ProductionBrokerTruthCancellationV2,
    ) -> Result<BrokerFinancialTruthBundleReceiptV2, BrokerTruthCaptureRunnerFailureV1> {
        self.calls += 1;
        assert_eq!(
            cancellation as *const _ as usize, self.expected_cancellation_address,
            "orchestration must pass the same lexical cancellation object"
        );
        assert_eq!(invocation.environment(), BrokerEnvironment::Demo);
        assert_eq!(invocation.account_id(), ACCOUNT_ID);
        assert_eq!(invocation.window(), window());
        assert_eq!(invocation.reviewed_synchronization_count(), 1);
        assert_eq!(invocation.binding(), &self.expected_binding);
        assert_eq!(invocation.store_root(), self.expected_store_root.as_path());
        self.saw_published_authority =
            BrokerTruthAcquisitionStoreV1::new(&self.expected_store_root)
                .open_authority(invocation.authority_receipt())
                .is_ok();
        match &self.result {
            SpyResult::Published(receipt) => Ok(receipt.clone()),
            SpyResult::Failed => Err(BrokerTruthCaptureRunnerFailureV1::opaque()),
        }
    }
}

fn published_ids(root: &Path, prefix: &str) -> Vec<String> {
    let mut ids = fs::read_dir(root)
        .expect("inspect explicit test store root")
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| name.starts_with(prefix))
        .collect::<Vec<_>>();
    ids.sort();
    ids
}

fn assert_sanitized_error(
    error: &BrokerTruthAcquisitionOrchestrationErrorV1,
    forbidden_paths: &[&Path],
) {
    for path in forbidden_paths {
        assert!(
            !error.detail().contains(&path.to_string_lossy().to_string()),
            "orchestration errors must not disclose input paths"
        );
    }
    assert!(!error.detail().contains("clientMsgId"));
    assert!(!error.detail().contains("tickData"));
}

#[test]
fn exact_ingress_authority_runner_bft2_and_link_are_ordered_and_evidence_only() {
    let Fixture {
        _temp,
        prepared,
        store_root,
        bft2_source_root,
        observations_path: _,
        trust_root_path: _,
        binding,
        instrument,
    } = Fixture::new();
    let broker_receipt =
        publish_integrity_only_bft2(&store_root, &bft2_source_root, &binding, &instrument);
    let cancellation = ProductionBrokerTruthCancellationV2::new();
    let mut runner = OfflineRunnerSpy::new(
        store_root.clone(),
        &cancellation,
        binding.clone(),
        SpyResult::Published(broker_receipt.clone()),
    );

    let outcome = execute_prepared_acquisition_with_runner_v1(prepared, &cancellation, &mut runner)
        .expect("orchestrate exact immutable evidence");

    assert_eq!(runner.calls, 1);
    assert!(runner.saw_published_authority);
    assert_eq!(outcome.broker_truth_receipt(), &broker_receipt);
    assert_eq!(
        outcome.semantic_status(),
        BrokerTruthAcquisitionSemanticStatusV1::UnvalidatedEvidenceOnly
    );
    assert_eq!(
        outcome.promotion_eligibility(),
        BrokerTruthAcquisitionPromotionEligibilityV1::NotPromotionEligible
    );
    let store = BrokerTruthAcquisitionStoreV1::new(&store_root);
    let reopened = store
        .open_link(outcome.link_receipt())
        .expect("reopen exact immutable acquisition link");
    assert_eq!(
        reopened.manifest().authority_receipt(),
        outcome.authority_receipt()
    );
    assert_eq!(
        reopened.manifest().broker_truth_receipt(),
        outcome.broker_truth_receipt()
    );
    assert_eq!(reopened.manifest().binding(), &binding);
    assert_eq!(published_ids(&store_root, "bfta1-").len(), 1);
    assert_eq!(published_ids(&store_root, "bft2-").len(), 1);
    assert_eq!(published_ids(&store_root, "bftl1-").len(), 1);
    assert!(!store_root.join("current").exists());
    assert!(!store_root.join("default").exists());
}

#[test]
fn reviewed_ingress_tamper_or_prepared_source_reordering_fails_before_authority_and_runner() {
    let Fixture {
        _temp,
        prepared,
        store_root,
        bft2_source_root: _,
        observations_path,
        trust_root_path: _,
        binding,
        instrument: _,
    } = Fixture::new();
    OpenOptions::new()
        .append(true)
        .open(&observations_path)
        .expect("open reviewed observations for deliberate tamper")
        .write_all(b"post-preflight-tamper")
        .expect("tamper reviewed observations");
    let cancellation = ProductionBrokerTruthCancellationV2::new();
    let mut runner = OfflineRunnerSpy::new(
        store_root.clone(),
        &cancellation,
        binding,
        SpyResult::Failed,
    );
    let error = execute_prepared_acquisition_with_runner_v1(prepared, &cancellation, &mut runner)
        .expect_err("reviewed Vortex tamper must fail before publication");
    assert_eq!(
        error.code(),
        BrokerTruthAcquisitionOrchestrationErrorCodeV1::ReviewedSynchronizationInvalid
    );
    assert_sanitized_error(&error, &[&observations_path]);
    assert_eq!(runner.calls, 0);
    assert!(published_ids(&store_root, "bfta1-").is_empty());

    let Fixture {
        _temp,
        mut prepared,
        store_root,
        bft2_source_root: _,
        observations_path,
        trust_root_path: _,
        binding,
        instrument: _,
    } = Fixture::new();
    prepared.artifact_sources.swap(8, 9);
    let cancellation = ProductionBrokerTruthCancellationV2::new();
    let mut runner = OfflineRunnerSpy::new(
        store_root.clone(),
        &cancellation,
        binding,
        SpyResult::Failed,
    );
    let error = execute_prepared_acquisition_with_runner_v1(prepared, &cancellation, &mut runner)
        .expect_err("prepared artifact/source order mismatch must fail closed");
    assert_eq!(
        error.code(),
        BrokerTruthAcquisitionOrchestrationErrorCodeV1::ReviewedSynchronizationInvalid
    );
    assert_sanitized_error(&error, &[&observations_path]);
    assert_eq!(runner.calls, 0);
    assert!(published_ids(&store_root, "bfta1-").is_empty());
}

#[test]
fn authority_tamper_and_runner_failure_never_publish_a_link() {
    let Fixture {
        _temp,
        prepared,
        store_root,
        bft2_source_root: _,
        observations_path,
        trust_root_path,
        binding,
        instrument: _,
    } = Fixture::new();
    OpenOptions::new()
        .append(true)
        .open(&trust_root_path)
        .expect("open trust bytes for deliberate tamper")
        .write_all(b"tamper")
        .expect("tamper trust bytes");
    let cancellation = ProductionBrokerTruthCancellationV2::new();
    let mut runner = OfflineRunnerSpy::new(
        store_root.clone(),
        &cancellation,
        binding,
        SpyResult::Failed,
    );
    let error = execute_prepared_acquisition_with_runner_v1(prepared, &cancellation, &mut runner)
        .expect_err("authority source tamper must fail before runner");
    assert_eq!(
        error.code(),
        BrokerTruthAcquisitionOrchestrationErrorCodeV1::AuthorityPublicationFailed
    );
    assert_sanitized_error(&error, &[&trust_root_path, &observations_path]);
    assert_eq!(runner.calls, 0);
    assert!(published_ids(&store_root, "bfta1-").is_empty());
    assert!(published_ids(&store_root, "bftl1-").is_empty());

    let Fixture {
        _temp,
        prepared,
        store_root,
        bft2_source_root: _,
        observations_path,
        trust_root_path,
        binding,
        instrument: _,
    } = Fixture::new();
    let cancellation = ProductionBrokerTruthCancellationV2::new();
    let mut runner = OfflineRunnerSpy::new(
        store_root.clone(),
        &cancellation,
        binding,
        SpyResult::Failed,
    );
    let error = execute_prepared_acquisition_with_runner_v1(prepared, &cancellation, &mut runner)
        .expect_err("capture failure must not become a linked outcome");
    assert_eq!(
        error.code(),
        BrokerTruthAcquisitionOrchestrationErrorCodeV1::CaptureFailed
    );
    assert_sanitized_error(&error, &[&trust_root_path, &observations_path]);
    assert_eq!(runner.calls, 1);
    assert!(runner.saw_published_authority);
    assert_eq!(published_ids(&store_root, "bfta1-").len(), 1);
    assert!(published_ids(&store_root, "bftl1-").is_empty());
}

#[test]
fn production_request_identity_mismatch_fails_before_runner() {
    let Fixture {
        _temp,
        mut prepared,
        store_root,
        bft2_source_root: _,
        observations_path,
        trust_root_path,
        binding,
        instrument: _,
    } = Fixture::new();
    prepared.account_id += 1;
    let cancellation = ProductionBrokerTruthCancellationV2::new();
    let mut runner = OfflineRunnerSpy::new(
        store_root.clone(),
        &cancellation,
        binding,
        SpyResult::Failed,
    );
    let error = execute_prepared_acquisition_with_runner_v1(prepared, &cancellation, &mut runner)
        .expect_err("mutated prepared account identity must fail before runner");
    assert_eq!(
        error.code(),
        BrokerTruthAcquisitionOrchestrationErrorCodeV1::CaptureRequestInvalid
    );
    assert_sanitized_error(&error, &[&trust_root_path, &observations_path]);
    assert_eq!(runner.calls, 0);
    assert_eq!(published_ids(&store_root, "bfta1-").len(), 1);
    assert!(published_ids(&store_root, "bftl1-").is_empty());
}

#[test]
fn unbacked_runner_receipt_cannot_create_evidence_status_or_link() {
    let Fixture {
        _temp,
        prepared,
        store_root,
        bft2_source_root: _,
        observations_path,
        trust_root_path,
        binding,
        instrument: _,
    } = Fixture::new();
    let missing_sha = digest(0x61);
    let unbacked_receipt = BrokerFinancialTruthBundleReceiptV2::from_json_bytes(
        format!(r#"{{"bundle_id":"bft2-{missing_sha}","manifest_sha256":"{missing_sha}"}}"#)
            .as_bytes(),
    )
    .expect("syntactically exact but unbacked receipt");
    let cancellation = ProductionBrokerTruthCancellationV2::new();
    let mut runner = OfflineRunnerSpy::new(
        store_root.clone(),
        &cancellation,
        binding,
        SpyResult::Published(unbacked_receipt),
    );
    let error = execute_prepared_acquisition_with_runner_v1(prepared, &cancellation, &mut runner)
        .expect_err("an unbacked fake receipt must fail the real link store barrier");
    assert_eq!(
        error.code(),
        BrokerTruthAcquisitionOrchestrationErrorCodeV1::LinkPublicationFailed
    );
    assert_sanitized_error(&error, &[&trust_root_path, &observations_path]);
    assert_eq!(runner.calls, 1);
    assert!(runner.saw_published_authority);
    assert_eq!(published_ids(&store_root, "bfta1-").len(), 1);
    assert!(published_ids(&store_root, "bftl1-").is_empty());
}

#[test]
fn public_entrypoint_is_concrete_and_runner_injection_remains_crate_private() {
    let _entrypoint: fn(
        PreparedBrokerTruthAcquisitionV1,
        &ProductionBrokerTruthCancellationV2,
    ) -> Result<
        BrokerTruthAcquisitionOutcomeV1,
        BrokerTruthAcquisitionOrchestrationErrorV1,
    > = execute_prepared_acquisition_v1;

    let production = fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/acquisition_orchestration_v1.rs"
    ))
    .expect("read orchestration production source");
    for required in [
        "load_reviewed_ctrader_quote_synchronizations_v2",
        "ReviewedCTraderQuoteSynchronizationSourceV2::new",
        ".publish_authority(",
        "ProductionBrokerTruthCaptureRequestV2::new",
        "capture_production_broker_financial_truth_v2",
        ".publish_link(",
        "UnvalidatedEvidenceOnly",
        "NotPromotionEligible",
        "pub(super) trait BrokerTruthCaptureRunnerV1",
    ] {
        assert!(
            production.contains(required),
            "orchestration source must contain {required}"
        );
    }
    for forbidden in [
        "client_secret",
        "access_token",
        "refresh_token",
        "credentials_path",
        "load_exact_production_broker_truth_credentials_v2",
        "ProductionCTraderBrokerTruthAuthenticationWireV2",
        "TcpStream",
        "connect_async",
        "std::env",
        "current_dir",
        "read_dir",
        "Default::default",
        "unwrap_or",
        "fallback",
        "capability",
        "permit",
    ] {
        assert!(
            !production.contains(forbidden),
            "bridge orchestration must not contain {forbidden}"
        );
    }
    let library = fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/lib.rs"))
        .expect("read bridge root source");
    assert!(!library.contains("BrokerTruthCaptureRunnerV1"));
    assert!(!library.contains("execute_prepared_acquisition_with_runner_v1"));
}
