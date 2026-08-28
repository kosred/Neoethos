use std::fs;
use std::path::{Path, PathBuf};

use neoethos_broker_history::bootstrap_writer::{
    BrokerTrendbarStreamRequest, publish_broker_trendbar_chunks,
};
use neoethos_broker_truth::{
    BrokerTruthAcquisitionArtifactRoleV1, BrokerTruthAcquisitionPromotionEligibilityV1,
    BrokerTruthAcquisitionSemanticStatusV1, EvidenceWindowV1, ReviewedQuoteReplayRuleIdentityV2,
};
use neoethos_broker_truth_acquire::{
    BrokerTruthAcquisitionArgsV1, BrokerTruthAcquisitionPreflightErrorCodeV1,
    prepare_acquisition_v1,
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

const ACCOUNT_ID: i64 = 7_001;
const PRIMARY_SYMBOL_ID: i64 = 14;
const CONVERSION_SYMBOL_ID: i64 = 15;
const EUR_ASSET_ID: i64 = 1;
const USD_ASSET_ID: i64 = 2;
const JPY_ASSET_ID: i64 = 3;
const FROM_MS: i64 = 1_700_000_040_000;
const LAST_BAR_MS: i64 = FROM_MS + 120_000;
const TO_MS_EXCLUSIVE: i64 = FROM_MS + 180_000;
const WINDOW_POLICY_ID: &str = "reviewed-half-open-m1-v1";
const SERVER: &str = "demo.ctraderapi.com";

#[derive(Clone)]
struct FrozenInput {
    path: PathBuf,
    sha256: String,
}

struct Fixture {
    _temp: TempDir,
    data_root: PathBuf,
    identity: CanonicalDatasetIdentity,
    selected: SelectedDatasetGenerationV1,
    args: Vec<String>,
    receipt: FrozenInput,
    capture_plan: FrozenInput,
    trust_root: FrozenInput,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn write_bytes(directory: &Path, name: &str, bytes: &[u8]) -> FrozenInput {
    let path = directory.join(name);
    fs::write(&path, bytes).expect("write frozen preflight input");
    FrozenInput {
        path,
        sha256: sha256(bytes),
    }
}

fn write_json(directory: &Path, name: &str, value: &Value) -> FrozenInput {
    write_bytes(
        directory,
        name,
        &serde_json::to_vec(value).expect("encode frozen JSON input"),
    )
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

fn publish_primary_generation(
    data_root: &Path,
    identity: &CanonicalDatasetIdentity,
    expected_generation: Option<&str>,
    close_offset: f64,
) -> PublishResult {
    let chunk = CanonicalOhlcvChunk {
        timestamp_ms: vec![FROM_MS, FROM_MS + 60_000, LAST_BAR_MS],
        open: vec![1.10, 1.11, 1.12],
        high: vec![1.12, 1.13, 1.14],
        low: vec![1.09, 1.10, 1.11],
        close: vec![1.11, 1.12, 1.13 + close_offset],
        volume: CanonicalVolumeChunk::Int64(vec![10, 11, 12]),
    };
    publish_broker_trendbar_chunks(BrokerTrendbarStreamRequest {
        configured_root: data_root,
        identity,
        expected_generation,
        requested_from_ms: FROM_MS,
        requested_to_ms: TO_MS_EXCLUSIVE,
        retrieved_unix_ms: 1_800_000_000_000,
        returned_from_ms: FROM_MS,
        returned_to_ms: LAST_BAR_MS,
        row_count: 3,
        chunks: vec![Ok(chunk)],
    })
    .expect("publish exact canonical cTrader generation")
}

fn exact_instrument(
    symbol_id: i64,
    symbol_name: &str,
    base_asset_id: i64,
    base_asset_name: &str,
    quote_asset_id: i64,
    quote_asset_name: &str,
) -> Value {
    json!({
        "symbol_id": symbol_id,
        "symbol_name": symbol_name,
        "base_asset_id": base_asset_id,
        "base_asset_name": base_asset_name,
        "quote_asset_id": quote_asset_id,
        "quote_asset_name": quote_asset_name
    })
}

fn window() -> Value {
    json!({
        "from_unix_ms_inclusive": FROM_MS,
        "to_unix_ms_exclusive": TO_MS_EXCLUSIVE
    })
}

fn replace_flag_value(args: &mut [String], flag: &str, value: impl Into<String>) {
    let index = args
        .iter()
        .position(|argument| argument == flag)
        .expect("fixture contains required flag");
    args[index + 1] = value.into();
}

fn remove_all_flag_values(args: &mut Vec<String>, flag: &str) {
    let mut removed = false;
    while let Some(index) = args.iter().position(|argument| argument == flag) {
        args.drain(index..=index + 1);
        removed = true;
    }
    assert!(removed, "fixture contains required flag {flag}");
}

fn parse_error_code(args: Vec<String>) -> BrokerTruthAcquisitionPreflightErrorCodeV1 {
    match BrokerTruthAcquisitionArgsV1::try_parse_from(args) {
        Ok(_) => panic!("invalid acquisition arguments unexpectedly parsed"),
        Err(error) => error.code(),
    }
}

fn preflight_error_code(args: Vec<String>) -> BrokerTruthAcquisitionPreflightErrorCodeV1 {
    let parsed = BrokerTruthAcquisitionArgsV1::try_parse_from(args)
        .expect("mutated fixture remains syntactically valid");
    match prepare_acquisition_v1(parsed) {
        Ok(_) => panic!("invalid acquisition evidence unexpectedly passed preflight"),
        Err(error) => error.code(),
    }
}

impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().expect("temporary acquisition fixture");
        let data_root = temp.path().join("canonical-data");
        let input_root = temp.path().join("immutable-inputs");
        let work_parent = temp.path().join("capture-work");
        let store_root = temp.path().join("broker-truth-store");
        fs::create_dir_all(&input_root).expect("create immutable input root");
        fs::create_dir_all(&work_parent).expect("create explicit work parent");
        fs::create_dir_all(&store_root).expect("create explicit store root");

        let identity = primary_identity();
        let publication = publish_primary_generation(&data_root, &identity, None, 0.0);
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
                "evidence_window": window(),
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
        let primary_observations = write_bytes(
            &input_root,
            "quote-session-observations-000.vortex",
            b"immutable-primary-observation-fixture-v1",
        );
        let primary_rules = write_bytes(
            &input_root,
            "reviewed-quote-replay-rules-000.vortex",
            b"immutable-primary-reviewed-rules-fixture-v1",
        );
        let conversion_observations = write_bytes(
            &input_root,
            "quote-session-observations-001.vortex",
            b"immutable-conversion-observation-fixture-v1",
        );
        let conversion_rules = write_bytes(
            &input_root,
            "reviewed-quote-replay-rules-001.vortex",
            b"immutable-conversion-reviewed-rules-fixture-v1",
        );

        let primary = exact_instrument(
            PRIMARY_SYMBOL_ID,
            "EURUSD",
            EUR_ASSET_ID,
            "EUR",
            USD_ASSET_ID,
            "USD",
        );
        let conversion = exact_instrument(
            CONVERSION_SYMBOL_ID,
            "USDJPY",
            USD_ASSET_ID,
            "USD",
            JPY_ASSET_ID,
            "JPY",
        );
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
                "window": window(),
                "synchronizations": [
                    {
                        "ordinal": 0,
                        "instrument": primary,
                        "quote_observations_sha256": primary_observations.sha256,
                        "reviewed_replay_rules_sha256": primary_rules.sha256
                    },
                    {
                        "ordinal": 1,
                        "instrument": conversion,
                        "quote_observations_sha256": conversion_observations.sha256,
                        "reviewed_replay_rules_sha256": conversion_rules.sha256
                    }
                ]
            }),
        );
        let primary_review_identity = ReviewedQuoteReplayRuleIdentityV2::new(
            review_record.sha256.clone(),
            protocol.sha256.clone(),
            primary_observations.sha256.clone(),
        )
        .expect("primary reviewed replay-rule identity");
        let conversion_review_identity = ReviewedQuoteReplayRuleIdentityV2::new(
            review_record.sha256.clone(),
            protocol.sha256.clone(),
            conversion_observations.sha256.clone(),
        )
        .expect("conversion reviewed replay-rule identity");

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
                "window": window(),
                "primary_instrument": primary,
                "account_asset": {"asset_id": JPY_ASSET_ID, "asset_name": "JPY"},
                "conversion_routes": [{
                    "purpose": "primary_pnl_settlement",
                    "from_asset_id": USD_ASSET_ID,
                    "from_asset_name": "USD",
                    "to_asset_id": JPY_ASSET_ID,
                    "to_asset_name": "JPY",
                    "legs": [{
                        "from_asset_id": USD_ASSET_ID,
                        "from_asset_name": "USD",
                        "to_asset_id": JPY_ASSET_ID,
                        "to_asset_name": "JPY",
                        "instrument": conversion
                    }]
                }],
                "synchronizations": [
                    {
                        "ordinal": 0,
                        "account_id": ACCOUNT_ID,
                        "instrument": primary,
                        "window": window(),
                        "quote_observations_sha256": primary_observations.sha256,
                        "reviewed_replay_rules_sha256": primary_rules.sha256,
                        "review_identity_sha256": primary_review_identity.identity_sha256()
                    },
                    {
                        "ordinal": 1,
                        "account_id": ACCOUNT_ID,
                        "instrument": conversion,
                        "window": window(),
                        "quote_observations_sha256": conversion_observations.sha256,
                        "reviewed_replay_rules_sha256": conversion_rules.sha256,
                        "review_identity_sha256": conversion_review_identity.identity_sha256()
                    }
                ]
            }),
        );

        let args = vec![
            "broker-truth-acquire".to_owned(),
            "--data-root".to_owned(),
            data_root.to_string_lossy().into_owned(),
            "--canonical-receipt".to_owned(),
            receipt.path.to_string_lossy().into_owned(),
            "--canonical-receipt-sha256".to_owned(),
            receipt.sha256.clone(),
            "--canonical-scope".to_owned(),
            scope.path.to_string_lossy().into_owned(),
            "--canonical-scope-sha256".to_owned(),
            scope.sha256.clone(),
            "--canonical-root-verification".to_owned(),
            root_verification.path.to_string_lossy().into_owned(),
            "--canonical-root-verification-sha256".to_owned(),
            root_verification.sha256.clone(),
            "--canonical-scope-window-binding".to_owned(),
            window_binding.path.to_string_lossy().into_owned(),
            "--canonical-scope-window-binding-sha256".to_owned(),
            window_binding.sha256.clone(),
            "--capture-plan".to_owned(),
            capture_plan.path.to_string_lossy().into_owned(),
            "--capture-plan-sha256".to_owned(),
            capture_plan.sha256.clone(),
            "--review-record".to_owned(),
            review_record.path.to_string_lossy().into_owned(),
            "--review-record-sha256".to_owned(),
            review_record.sha256.clone(),
            "--protocol-evidence".to_owned(),
            protocol.path.to_string_lossy().into_owned(),
            "--protocol-evidence-sha256".to_owned(),
            protocol.sha256.clone(),
            "--trust-root".to_owned(),
            trust_root.path.to_string_lossy().into_owned(),
            "--trust-root-sha256".to_owned(),
            trust_root.sha256.clone(),
            "--quote-observations".to_owned(),
            primary_observations.path.to_string_lossy().into_owned(),
            "--quote-observations-sha256".to_owned(),
            primary_observations.sha256.clone(),
            "--reviewed-replay-rules".to_owned(),
            primary_rules.path.to_string_lossy().into_owned(),
            "--reviewed-replay-rules-sha256".to_owned(),
            primary_rules.sha256.clone(),
            "--quote-observations".to_owned(),
            conversion_observations.path.to_string_lossy().into_owned(),
            "--quote-observations-sha256".to_owned(),
            conversion_observations.sha256.clone(),
            "--reviewed-replay-rules".to_owned(),
            conversion_rules.path.to_string_lossy().into_owned(),
            "--reviewed-replay-rules-sha256".to_owned(),
            conversion_rules.sha256.clone(),
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

        Self {
            _temp: temp,
            data_root,
            identity,
            selected,
            args,
            receipt,
            capture_plan,
            trust_root,
        }
    }
}

#[test]
fn every_authority_input_is_explicit_and_secret_inputs_are_not_accepted() {
    let fixture = Fixture::new();
    for required in [
        "--data-root",
        "--canonical-receipt",
        "--canonical-receipt-sha256",
        "--canonical-scope",
        "--canonical-scope-sha256",
        "--canonical-root-verification",
        "--canonical-root-verification-sha256",
        "--canonical-scope-window-binding",
        "--canonical-scope-window-binding-sha256",
        "--capture-plan",
        "--capture-plan-sha256",
        "--review-record",
        "--review-record-sha256",
        "--protocol-evidence",
        "--protocol-evidence-sha256",
        "--trust-root",
        "--trust-root-sha256",
        "--quote-observations",
        "--quote-observations-sha256",
        "--reviewed-replay-rules",
        "--reviewed-replay-rules-sha256",
        "--environment",
        "--account-id",
        "--from-ms",
        "--to-ms-exclusive",
        "--work-parent",
        "--store-root",
    ] {
        let mut args = fixture.args.clone();
        remove_all_flag_values(&mut args, required);
        assert_eq!(
            parse_error_code(args),
            BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments,
            "missing {required} must be refused"
        );
    }

    for forbidden in [
        "--client-secret",
        "--access-token",
        "--refresh-token",
        "--credentials-path",
        "--api-key",
    ] {
        let mut args = fixture.args.clone();
        args.extend([forbidden.to_owned(), "do-not-read-or-print-me".to_owned()]);
        assert_eq!(
            parse_error_code(args),
            BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments,
            "secret-bearing flag {forbidden} must not exist"
        );
    }

    let mut duplicate = fixture.args.clone();
    duplicate.extend(["--account-id".to_owned(), ACCOUNT_ID.to_string()]);
    assert_eq!(
        parse_error_code(duplicate),
        BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments
    );
    let mut unknown = fixture.args.clone();
    unknown.push("--discover-sibling-authority".to_owned());
    assert_eq!(
        parse_error_code(unknown),
        BrokerTruthAcquisitionPreflightErrorCodeV1::InvalidArguments
    );
}

#[test]
fn exact_receipt_scope_root_window_plan_and_review_set_prepare_evidence_only_authority() {
    let fixture = Fixture::new();
    let args = BrokerTruthAcquisitionArgsV1::try_parse_from(fixture.args.clone())
        .expect("complete explicit acquisition arguments");
    let prepared = prepare_acquisition_v1(args).expect("strict acquisition preflight");

    assert_eq!(prepared.environment(), CTraderEnvironment::Demo);
    assert_eq!(prepared.account_id(), ACCOUNT_ID);
    assert_eq!(
        prepared.evidence_window(),
        EvidenceWindowV1::new(FROM_MS, TO_MS_EXCLUSIVE).expect("same half-open window")
    );
    assert_eq!(
        prepared
            .capture_request()
            .binding()
            .canonical_dataset_identity(),
        &fixture.identity
    );
    assert_eq!(
        prepared.capture_request().primary_instrument().symbol_id(),
        PRIMARY_SYMBOL_ID
    );
    assert_eq!(prepared.capture_request().conversion_routes().len(), 1);
    assert_eq!(
        prepared.capture_request().conversion_routes()[0].legs()[0]
            .instrument()
            .symbol_id(),
        CONVERSION_SYMBOL_ID
    );
    assert_eq!(prepared.opened_generation_count(), 1);
    assert_eq!(prepared.reviewed_synchronization_count(), 2);
    assert_eq!(prepared.artifact_sources().len(), 12);
    assert_eq!(
        prepared.authority_manifest().semantic_status(),
        BrokerTruthAcquisitionSemanticStatusV1::UnvalidatedEvidenceOnly
    );
    assert_eq!(
        prepared.authority_manifest().promotion_eligibility(),
        BrokerTruthAcquisitionPromotionEligibilityV1::NotPromotionEligible
    );
    assert_eq!(
        prepared
            .authority_manifest()
            .reviewed_synchronizations()
            .iter()
            .map(|binding| (binding.ordinal(), binding.symbol_id()))
            .collect::<Vec<_>>(),
        vec![(0, PRIMARY_SYMBOL_ID), (1, CONVERSION_SYMBOL_ID)]
    );
    assert_eq!(
        prepared.authority_manifest().artifacts()[0].role(),
        BrokerTruthAcquisitionArtifactRoleV1::CanonicalSearchInputReceipt
    );
    assert_eq!(
        prepared.work_parent(),
        fixture._temp.path().join("capture-work")
    );
    assert_eq!(
        prepared.store_root(),
        fixture._temp.path().join("broker-truth-store")
    );
}

#[test]
fn tamper_path_window_and_synchronization_mismatches_fail_closed() {
    let mut fixture = Fixture::new();
    fs::write(&fixture.trust_root.path, b"tampered trust bytes")
        .expect("tamper immutable trust fixture");
    assert_eq!(
        preflight_error_code(fixture.args.clone()),
        BrokerTruthAcquisitionPreflightErrorCodeV1::ArtifactDigestMismatch
    );

    fixture = Fixture::new();
    let aliased_receipt = fixture
        .receipt
        .path
        .parent()
        .expect("receipt parent")
        .join("..")
        .join("immutable-inputs")
        .join("canonical-search-input-receipt.json");
    replace_flag_value(
        &mut fixture.args,
        "--canonical-receipt",
        aliased_receipt.to_string_lossy(),
    );
    assert_eq!(
        preflight_error_code(fixture.args.clone()),
        BrokerTruthAcquisitionPreflightErrorCodeV1::UnsafePath
    );

    fixture = Fixture::new();
    replace_flag_value(
        &mut fixture.args,
        "--to-ms-exclusive",
        (TO_MS_EXCLUSIVE + 60_000).to_string(),
    );
    assert_eq!(
        preflight_error_code(fixture.args.clone()),
        BrokerTruthAcquisitionPreflightErrorCodeV1::WindowMismatch
    );

    fixture = Fixture::new();
    let mut plan: Value =
        serde_json::from_slice(&fs::read(&fixture.capture_plan.path).expect("read capture plan"))
            .expect("decode capture plan");
    plan["synchronizations"]
        .as_array_mut()
        .expect("synchronization array")
        .swap(0, 1);
    let changed = serde_json::to_vec(&plan).expect("encode reordered capture plan");
    fs::write(&fixture.capture_plan.path, &changed).expect("write reordered capture plan");
    replace_flag_value(&mut fixture.args, "--capture-plan-sha256", sha256(&changed));
    assert_eq!(
        preflight_error_code(fixture.args),
        BrokerTruthAcquisitionPreflightErrorCodeV1::SynchronizationMismatch
    );
}

#[test]
fn strict_receipt_decode_and_exact_generation_open_never_fall_back_to_current() {
    let mut fixture = Fixture::new();
    let mut receipt: Value =
        serde_json::from_slice(&fs::read(&fixture.receipt.path).expect("read canonical receipt"))
            .expect("decode canonical receipt fixture");
    receipt["legacy_symbol_fallback"] = Value::String("EURUSD".to_owned());
    let changed = serde_json::to_vec(&receipt).expect("encode unknown-field receipt");
    fs::write(&fixture.receipt.path, &changed).expect("write unknown-field receipt");
    replace_flag_value(
        &mut fixture.args,
        "--canonical-receipt-sha256",
        sha256(&changed),
    );
    assert_eq!(
        preflight_error_code(fixture.args),
        BrokerTruthAcquisitionPreflightErrorCodeV1::CanonicalReceiptInvalid
    );

    let fixture = Fixture::new();
    let replacement = publish_primary_generation(
        &fixture.data_root,
        &fixture.identity,
        Some(fixture.selected.generation_id()),
        0.01,
    );
    assert_ne!(replacement.generation(), fixture.selected.generation_id());
    assert_eq!(
        preflight_error_code(fixture.args),
        BrokerTruthAcquisitionPreflightErrorCodeV1::ExactRootOpenFailed,
        "an advanced current pointer must not promote the new generation or reopen by symbol"
    );
}

#[test]
fn preflight_source_has_no_session_secret_ambient_authority_or_permit_path() {
    let source = include_str!("../src/lib.rs");
    let manifest = include_str!("../Cargo.toml");
    for forbidden in [
        "std::env",
        "var_os(",
        "load_exact_production_broker_truth_credentials_v2",
        "ProductionCTraderOpenApiSession",
        "CTraderBrokerTruthSameSessionV2",
        "capture_production_broker_financial_truth_v2",
        "current_broker_financial_truth_capability_v1",
        "install_broker_financial_truth",
        "BrokerFinancialTruthPermit",
    ] {
        assert!(
            !source.contains(forbidden),
            "preflight source must not contain forbidden authority/session path {forbidden}"
        );
    }
    for forbidden_dependency in ["keyring", "reqwest", "tokio"] {
        assert!(
            !manifest.contains(forbidden_dependency),
            "preflight crate must not directly depend on {forbidden_dependency}"
        );
    }
    assert!(
        source.contains("symlink_metadata"),
        "preflight must inspect artifact path entries without following symlinks/reparse aliases"
    );
    assert!(
        !source.contains("read_current_manifest")
            && !source.contains("load_canonical_timeframe(")
            && !source.contains("open_current"),
        "preflight must have no current-generation or legacy scalar/OHLC fallback"
    );
}
