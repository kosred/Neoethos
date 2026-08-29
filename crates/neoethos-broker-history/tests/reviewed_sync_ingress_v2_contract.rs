use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use neoethos_broker_history::broker_truth_capture::ExactQuoteInstrumentV2;
use neoethos_broker_history::broker_truth_ctrader::ReviewedCTraderQuoteSynchronizationV2;
use neoethos_broker_history::{
    ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2,
    ReviewedCTraderQuoteSynchronizationSourceV2, load_reviewed_ctrader_quote_synchronizations_v2,
};
use neoethos_broker_truth::{
    BrokerFinancialTruthVortexSchemaV1, BrokerTruthReviewedSynchronizationBindingV1,
    EvidenceWindowV1, ImmutableVortexArtifactV1, QuoteSideV1, ReviewedQuoteReplayRuleIdentityV2,
};
use serde_json::{Value, json};
use tempfile::TempDir;
use vortex_array::IntoArray;
use vortex_array::arrays::{PrimitiveArray, StructArray, VarBinArray};

const ACCOUNT_ID: i64 = 7_001;
const SYMBOL_ID: i64 = 14;
const WINDOW_FROM: i64 = 1_700_000_040_000;
const WINDOW_TO: i64 = WINDOW_FROM + 180_000;
const PAYLOAD_TYPE: u32 = 2_146;
const REVIEW_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const PROTOCOL_SHA: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const FALSE_OBSERVATION_SHA: &str =
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LayoutTamper {
    None,
    ExtraField,
    MissingField,
    NullableField,
    WrongDtype,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SemanticTamper {
    None,
    IdentityDigest,
    Sequence,
    Account,
    Symbol,
    Window,
    MissingSymbolAuthority,
    ReversedSides,
    DuplicateSide,
    MissingObservation,
    ExtraObservation,
    ExtraReplayRule,
    WrongKind,
    RawDecodedLink,
}

#[derive(Clone)]
struct EvidenceRow {
    sequence: u64,
    account_id: i64,
    kind: u8,
    symbol_id: Option<i64>,
    quote_side: Option<QuoteSideV1>,
    requested_window: Option<EvidenceWindowV1>,
    client_msg_id: String,
    payload_type: u32,
    payload_json: String,
}

struct WrittenPair {
    observations_path: PathBuf,
    replay_rules_path: PathBuf,
    binding: BrokerTruthReviewedSynchronizationBindingV1,
    expected_review_identity_sha256: String,
}

impl WrittenPair {
    fn into_source(
        self,
    ) -> Result<
        ReviewedCTraderQuoteSynchronizationSourceV2,
        neoethos_broker_history::ReviewedCTraderQuoteSynchronizationIngressErrorV2,
    > {
        ReviewedCTraderQuoteSynchronizationSourceV2::new(
            self.binding,
            instrument(),
            self.observations_path,
            self.replay_rules_path,
        )
    }
}

fn window() -> EvidenceWindowV1 {
    EvidenceWindowV1::new(WINDOW_FROM, WINDOW_TO).expect("valid half-open fixture window")
}

fn instrument() -> ExactQuoteInstrumentV2 {
    ExactQuoteInstrumentV2::new(SYMBOL_ID, "EURUSD", 1, "EUR", 2, "USD")
        .expect("valid exact fixture instrument")
}

fn side_code(side: QuoteSideV1) -> u8 {
    match side {
        QuoteSideV1::Bid => 0,
        QuoteSideV1::Ask => 1,
    }
}

fn canonical_json(value: Value) -> String {
    serde_json::to_string(&value).expect("canonical fixture JSON")
}

fn observation_rows(prefix: &str) -> Vec<EvidenceRow> {
    [QuoteSideV1::Bid, QuoteSideV1::Ask]
        .into_iter()
        .enumerate()
        .map(|(sequence, side)| {
            let client_msg_id = format!("{prefix}-{}", side_code(side));
            EvidenceRow {
                sequence: sequence as u64,
                account_id: ACCOUNT_ID,
                kind: 0,
                symbol_id: Some(SYMBOL_ID),
                quote_side: Some(side),
                requested_window: Some(window()),
                client_msg_id: client_msg_id.clone(),
                payload_type: PAYLOAD_TYPE,
                payload_json: json!({
                    "clientMsgId": client_msg_id,
                    "payload": {
                        "ctidTraderAccountId": ACCOUNT_ID,
                        "hasMore": false,
                        "tickData": []
                    },
                    "payloadType": PAYLOAD_TYPE
                })
                .to_string(),
            }
        })
        .collect()
}

fn replay_rule_rows(prefix: &str) -> Vec<EvidenceRow> {
    vec![EvidenceRow {
        sequence: 0,
        account_id: ACCOUNT_ID,
        kind: 1,
        symbol_id: Some(SYMBOL_ID),
        quote_side: None,
        requested_window: Some(window()),
        client_msg_id: format!("{prefix}-0"),
        payload_type: PAYLOAD_TYPE,
        payload_json: canonical_json(json!({
            "accountId": ACCOUNT_ID,
            "askClientMsgId": format!("{prefix}-1"),
            "bidClientMsgId": format!("{prefix}-0"),
            "symbolId": SYMBOL_ID,
            "window": {
                "fromUnixMsInclusive": WINDOW_FROM,
                "toUnixMsExclusive": WINDOW_TO
            }
        })),
    }]
}

fn apply_semantic_tamper(
    tamper: SemanticTamper,
    observations: &mut Vec<EvidenceRow>,
    replay_rules: &mut Vec<EvidenceRow>,
) {
    match tamper {
        SemanticTamper::None | SemanticTamper::IdentityDigest => {}
        SemanticTamper::Sequence => observations[1].sequence = 7,
        SemanticTamper::Account => observations[0].account_id += 1,
        SemanticTamper::Symbol => replay_rules[0].symbol_id = Some(SYMBOL_ID + 1),
        SemanticTamper::Window => {
            replay_rules[0].requested_window = Some(
                EvidenceWindowV1::new(WINDOW_FROM + 1, WINDOW_TO)
                    .expect("valid deliberately mismatched window"),
            );
        }
        SemanticTamper::MissingSymbolAuthority => observations[0].symbol_id = None,
        SemanticTamper::ReversedSides => {
            observations[0].quote_side = Some(QuoteSideV1::Ask);
            observations[1].quote_side = Some(QuoteSideV1::Bid);
        }
        SemanticTamper::DuplicateSide => {
            observations[1].quote_side = Some(QuoteSideV1::Bid);
        }
        SemanticTamper::MissingObservation => {
            observations.pop();
        }
        SemanticTamper::ExtraObservation => {
            let mut extra = observations[1].clone();
            extra.sequence = 2;
            extra.client_msg_id.push_str("-extra");
            observations.push(extra);
        }
        SemanticTamper::ExtraReplayRule => {
            let mut extra = replay_rules[0].clone();
            extra.sequence = 1;
            replay_rules.push(extra);
        }
        SemanticTamper::WrongKind => replay_rules[0].kind = 0,
        SemanticTamper::RawDecodedLink => {
            replay_rules[0].client_msg_id = "unretained-client-msg-id".to_owned();
        }
    }
}

fn evidence_array(rows: &[EvidenceRow], layout_tamper: LayoutTamper) -> vortex_array::ArrayRef {
    let account_ids = match layout_tamper {
        LayoutTamper::NullableField => PrimitiveArray::from_option_iter(
            rows.iter()
                .enumerate()
                .map(|(index, row)| (index != 0).then_some(row.account_id)),
        )
        .into_array(),
        LayoutTamper::WrongDtype => PrimitiveArray::from_iter(
            rows.iter()
                .map(|row| u64::try_from(row.account_id).expect("positive fixture account")),
        )
        .into_array(),
        _ => PrimitiveArray::from_iter(rows.iter().map(|row| row.account_id)).into_array(),
    };
    let mut fields: Vec<(&str, vortex_array::ArrayRef)> = vec![
        (
            "sequence",
            PrimitiveArray::from_iter(rows.iter().map(|row| row.sequence)).into_array(),
        ),
        ("account_id", account_ids),
        (
            "evidence_kind",
            PrimitiveArray::from_iter(rows.iter().map(|row| row.kind)).into_array(),
        ),
        (
            "has_symbol_id",
            PrimitiveArray::from_iter(rows.iter().map(|row| u8::from(row.symbol_id.is_some())))
                .into_array(),
        ),
        (
            "symbol_id",
            PrimitiveArray::from_iter(rows.iter().map(|row| row.symbol_id.unwrap_or(0)))
                .into_array(),
        ),
        (
            "has_quote_side",
            PrimitiveArray::from_iter(rows.iter().map(|row| u8::from(row.quote_side.is_some())))
                .into_array(),
        ),
        (
            "quote_side",
            PrimitiveArray::from_iter(rows.iter().map(|row| row.quote_side.map_or(0, side_code)))
                .into_array(),
        ),
        (
            "has_requested_window",
            PrimitiveArray::from_iter(
                rows.iter()
                    .map(|row| u8::from(row.requested_window.is_some())),
            )
            .into_array(),
        ),
        (
            "requested_from_unix_ms_inclusive",
            PrimitiveArray::from_iter(rows.iter().map(|row| {
                row.requested_window
                    .map_or(0, EvidenceWindowV1::from_unix_ms_inclusive)
            }))
            .into_array(),
        ),
        (
            "requested_to_unix_ms_exclusive",
            PrimitiveArray::from_iter(rows.iter().map(|row| {
                row.requested_window
                    .map_or(0, EvidenceWindowV1::to_unix_ms_exclusive)
            }))
            .into_array(),
        ),
        (
            "client_msg_id",
            VarBinArray::from(
                rows.iter()
                    .map(|row| row.client_msg_id.as_str())
                    .collect::<Vec<_>>(),
            )
            .into_array(),
        ),
        (
            "payload_type",
            PrimitiveArray::from_iter(rows.iter().map(|row| row.payload_type)).into_array(),
        ),
    ];
    if layout_tamper != LayoutTamper::MissingField {
        fields.push((
            "payload_json",
            VarBinArray::from(
                rows.iter()
                    .map(|row| row.payload_json.as_str())
                    .collect::<Vec<_>>(),
            )
            .into_array(),
        ));
    }
    if layout_tamper == LayoutTamper::ExtraField {
        fields.push((
            "unexpected",
            PrimitiveArray::from_iter(rows.iter().map(|_| 1_u8)).into_array(),
        ));
    }
    StructArray::from_fields(&fields)
        .expect("structural evidence fixture")
        .into_array()
}

fn write_vortex(path: &Path, array: vortex_array::ArrayRef) {
    neoethos_data::core::vortex_io::write_vortex_array(path, array)
        .expect("write exact Vortex fixture");
}

fn file_sha256(
    relative_path: &str,
    path: &Path,
    schema: BrokerFinancialTruthVortexSchemaV1,
    row_count: usize,
) -> String {
    ImmutableVortexArtifactV1::from_file(relative_path, schema, row_count as u64, path)
        .expect("hash immutable fixture")
        .sha256()
        .to_owned()
}

fn write_pair(
    root: &Path,
    file_prefix: &str,
    ordinal: u32,
    semantic_tamper: SemanticTamper,
    observations_layout: LayoutTamper,
    rules_layout: LayoutTamper,
) -> WrittenPair {
    let message_prefix = format!("reviewed-sync-{file_prefix}");
    let mut observations = observation_rows(&message_prefix);
    let mut replay_rules = replay_rule_rows(&message_prefix);
    apply_semantic_tamper(semantic_tamper, &mut observations, &mut replay_rules);

    let observations_name = format!("{file_prefix}-quote-observations.vortex");
    let replay_rules_name = format!("{file_prefix}-reviewed-replay-rules.vortex");
    let observations_path = root.join(&observations_name);
    let replay_rules_path = root.join(&replay_rules_name);
    write_vortex(
        &observations_path,
        evidence_array(&observations, observations_layout),
    );
    write_vortex(
        &replay_rules_path,
        evidence_array(&replay_rules, rules_layout),
    );

    let observations_sha256 = file_sha256(
        &observations_name,
        &observations_path,
        BrokerFinancialTruthVortexSchemaV1::CTraderQuoteSessionObservationsRawV2,
        observations.len(),
    );
    let replay_rules_sha256 = file_sha256(
        &replay_rules_name,
        &replay_rules_path,
        BrokerFinancialTruthVortexSchemaV1::CTraderReviewedQuoteReplayRulesDecodedV2,
        replay_rules.len(),
    );
    let identity_observations_sha256 = if semantic_tamper == SemanticTamper::IdentityDigest {
        FALSE_OBSERVATION_SHA
    } else {
        &observations_sha256
    };
    let review_identity = ReviewedQuoteReplayRuleIdentityV2::new(
        REVIEW_SHA,
        PROTOCOL_SHA,
        identity_observations_sha256,
    )
    .expect("exact reviewed-rule identity");
    let expected_review_identity_sha256 = review_identity.identity_sha256().to_owned();
    let binding = BrokerTruthReviewedSynchronizationBindingV1::new(
        ordinal,
        ACCOUNT_ID,
        SYMBOL_ID,
        window(),
        review_identity,
        replay_rules_sha256,
    )
    .expect("exact reviewed synchronization binding");
    WrittenPair {
        observations_path,
        replay_rules_path,
        binding,
        expected_review_identity_sha256,
    }
}

fn write_opaque_pair(root: &Path) -> WrittenPair {
    let observations_name = "opaque-quote-observations.vortex";
    let replay_rules_name = "opaque-reviewed-replay-rules.vortex";
    let observations_path = root.join(observations_name);
    let replay_rules_path = root.join(replay_rules_name);
    fs::write(&observations_path, b"opaque Task4 quote observation bytes")
        .expect("write opaque observations");
    fs::write(&replay_rules_path, b"opaque Task4 replay-rule bytes")
        .expect("write opaque replay rules");
    let observations_sha256 = file_sha256(
        observations_name,
        &observations_path,
        BrokerFinancialTruthVortexSchemaV1::CTraderQuoteSessionObservationsRawV2,
        2,
    );
    let replay_rules_sha256 = file_sha256(
        replay_rules_name,
        &replay_rules_path,
        BrokerFinancialTruthVortexSchemaV1::CTraderReviewedQuoteReplayRulesDecodedV2,
        1,
    );
    let review_identity =
        ReviewedQuoteReplayRuleIdentityV2::new(REVIEW_SHA, PROTOCOL_SHA, observations_sha256)
            .expect("opaque bytes still have an exact identity, but no structural meaning");
    let expected_review_identity_sha256 = review_identity.identity_sha256().to_owned();
    let binding = BrokerTruthReviewedSynchronizationBindingV1::new(
        0,
        ACCOUNT_ID,
        SYMBOL_ID,
        window(),
        review_identity,
        replay_rules_sha256,
    )
    .expect("opaque evidence-only binding");
    WrittenPair {
        observations_path,
        replay_rules_path,
        binding,
        expected_review_identity_sha256,
    }
}

fn assert_load_error(
    source: ReviewedCTraderQuoteSynchronizationSourceV2,
    expected: ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2,
) {
    let error = match load_reviewed_ctrader_quote_synchronizations_v2(vec![source]) {
        Ok(_) => panic!("invalid reviewed synchronization evidence was accepted"),
        Err(error) => error,
    };
    assert_eq!(error.code(), expected);
}

#[test]
fn exact_pairs_decode_in_declared_order_to_concrete_evidence_only_synchronizations() {
    let root = TempDir::new().expect("fixture root");
    let first = write_pair(
        root.path(),
        "first",
        0,
        SemanticTamper::None,
        LayoutTamper::None,
        LayoutTamper::None,
    );
    let second = write_pair(
        root.path(),
        "second",
        1,
        SemanticTamper::None,
        LayoutTamper::None,
        LayoutTamper::None,
    );
    let expected_identities = [
        first.expected_review_identity_sha256.clone(),
        second.expected_review_identity_sha256.clone(),
    ];
    let loaded = load_reviewed_ctrader_quote_synchronizations_v2(vec![
        first.into_source().expect("first explicit source"),
        second.into_source().expect("second explicit source"),
    ])
    .expect("exact reviewed Vortex evidence must load");

    assert_eq!(loaded.len(), 2);
    for (ordinal, synchronization) in loaded.iter().enumerate() {
        assert_eq!(synchronization.ordinal(), ordinal as u32);
        assert_eq!(synchronization.account_id(), ACCOUNT_ID);
        assert_eq!(synchronization.symbol_id(), SYMBOL_ID);
        assert_eq!(synchronization.window(), window());
        assert_eq!(synchronization.raw_observation_count(), 2);
        assert_eq!(synchronization.decoded_replay_rule_count(), 1);
        assert_eq!(
            synchronization.review_identity_sha256(),
            expected_identities[ordinal]
        );
    }
    let concrete: Vec<ReviewedCTraderQuoteSynchronizationV2> = loaded
        .into_iter()
        .map(|synchronization| synchronization.into_synchronization())
        .collect();
    assert_eq!(concrete.len(), 2);

    let reversed_first = write_pair(
        root.path(),
        "reversed-first",
        0,
        SemanticTamper::None,
        LayoutTamper::None,
        LayoutTamper::None,
    );
    let reversed_second = write_pair(
        root.path(),
        "reversed-second",
        1,
        SemanticTamper::None,
        LayoutTamper::None,
        LayoutTamper::None,
    );
    let error = match load_reviewed_ctrader_quote_synchronizations_v2(vec![
        reversed_second
            .into_source()
            .expect("second explicit source"),
        reversed_first.into_source().expect("first explicit source"),
    ]) {
        Ok(_) => panic!("loader reordered noncontiguous authority behind the caller's back"),
        Err(error) => error,
    };
    assert_eq!(
        error.code(),
        ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch
    );
}

#[test]
fn opaque_or_post_binding_tampered_bytes_fail_before_structural_semantics() {
    let root = TempDir::new().expect("fixture root");
    let opaque = write_opaque_pair(root.path());
    assert_load_error(
        opaque.into_source().expect("explicit opaque source paths"),
        ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::VortexReadFailed,
    );

    for (name, tamper_rules) in [
        ("post-binding-observations-tamper", false),
        ("post-binding-rules-tamper", true),
    ] {
        let tampered = write_pair(
            root.path(),
            name,
            0,
            SemanticTamper::None,
            LayoutTamper::None,
            LayoutTamper::None,
        );
        let path = if tamper_rules {
            tampered.replay_rules_path.clone()
        } else {
            tampered.observations_path.clone()
        };
        let source = tampered.into_source().expect("explicit exact source");
        OpenOptions::new()
            .append(true)
            .open(path)
            .expect("open bound evidence file")
            .write_all(b"post-binding-tamper")
            .expect("tamper after binding");
        assert_load_error(
            source,
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::ArtifactDigestMismatch,
        );
    }
}

#[test]
fn extra_missing_nullable_or_wrong_dtype_columns_fail_exact_schema() {
    let root = TempDir::new().expect("fixture root");
    for (name, observations_tamper, rules_tamper) in [
        (
            "observations-extra-field",
            LayoutTamper::ExtraField,
            LayoutTamper::None,
        ),
        (
            "observations-missing-field",
            LayoutTamper::MissingField,
            LayoutTamper::None,
        ),
        (
            "observations-nullable-field",
            LayoutTamper::NullableField,
            LayoutTamper::None,
        ),
        (
            "observations-wrong-dtype",
            LayoutTamper::WrongDtype,
            LayoutTamper::None,
        ),
        (
            "rules-extra-field",
            LayoutTamper::None,
            LayoutTamper::ExtraField,
        ),
        (
            "rules-missing-field",
            LayoutTamper::None,
            LayoutTamper::MissingField,
        ),
        (
            "rules-nullable-field",
            LayoutTamper::None,
            LayoutTamper::NullableField,
        ),
        (
            "rules-wrong-dtype",
            LayoutTamper::None,
            LayoutTamper::WrongDtype,
        ),
    ] {
        let pair = write_pair(
            root.path(),
            name,
            0,
            SemanticTamper::None,
            observations_tamper,
            rules_tamper,
        );
        assert_load_error(
            pair.into_source().expect("explicit schema-tamper source"),
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::VortexSchemaMismatch,
        );
    }
}

#[test]
fn identity_account_symbol_window_side_count_and_linkage_fail_closed() {
    let root = TempDir::new().expect("fixture root");
    for (name, tamper, expected) in [
        (
            "identity",
            SemanticTamper::IdentityDigest,
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::ArtifactDigestMismatch,
        ),
        (
            "sequence",
            SemanticTamper::Sequence,
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
        ),
        (
            "account",
            SemanticTamper::Account,
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
        ),
        (
            "symbol",
            SemanticTamper::Symbol,
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
        ),
        (
            "window",
            SemanticTamper::Window,
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
        ),
        (
            "missing-symbol",
            SemanticTamper::MissingSymbolAuthority,
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
        ),
        (
            "side-order",
            SemanticTamper::ReversedSides,
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
        ),
        (
            "duplicate-side",
            SemanticTamper::DuplicateSide,
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
        ),
        (
            "missing-observation",
            SemanticTamper::MissingObservation,
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
        ),
        (
            "extra-observation",
            SemanticTamper::ExtraObservation,
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
        ),
        (
            "extra-rule",
            SemanticTamper::ExtraReplayRule,
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
        ),
        (
            "wrong-kind",
            SemanticTamper::WrongKind,
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::SynchronizationMismatch,
        ),
        (
            "raw-decoded-link",
            SemanticTamper::RawDecodedLink,
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::RawDecodedMismatch,
        ),
    ] {
        let pair = write_pair(
            root.path(),
            name,
            0,
            tamper,
            LayoutTamper::None,
            LayoutTamper::None,
        );
        assert_load_error(
            pair.into_source().expect("explicit semantic-tamper source"),
            expected,
        );
    }
}

#[test]
fn relative_and_mutable_selector_paths_are_refused_without_discovery() {
    let root = TempDir::new().expect("fixture root");
    let explicit = write_pair(
        root.path(),
        "relative-path",
        0,
        SemanticTamper::None,
        LayoutTamper::None,
        LayoutTamper::None,
    );
    let relative_error = match ReviewedCTraderQuoteSynchronizationSourceV2::new(
        explicit.binding,
        instrument(),
        PathBuf::from("relative-observations.vortex"),
        explicit.replay_rules_path,
    ) {
        Ok(_) => panic!("relative evidence path was accepted"),
        Err(error) => error,
    };
    assert_eq!(
        relative_error.code(),
        ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::UnsafePath
    );

    for selector in ["current", "default"] {
        let selector_root = root.path().join(selector);
        fs::create_dir(&selector_root).expect("create mutable-selector fixture directory");
        let pair = write_pair(
            &selector_root,
            "selector-path",
            0,
            SemanticTamper::None,
            LayoutTamper::None,
            LayoutTamper::None,
        );
        let error = match pair.into_source() {
            Ok(_) => panic!("mutable/default selector path was accepted"),
            Err(error) => error,
        };
        assert_eq!(
            error.code(),
            ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2::UnsafePath
        );
    }
}

#[test]
fn history_ingress_source_constructs_only_concrete_evidence_objects() {
    let source_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("reviewed_sync_ingress_v2.rs");
    let source = fs::read_to_string(source_path).expect("history-owned Task5a production source");
    for required in [
        "CapturedBrokerEvidenceRowV2::new",
        "CapturedBrokerEvidencePairV2::new",
        "CapturedQuoteSynchronizationV2::new",
        "ReviewedCTraderQuoteSynchronizationV2::new",
    ] {
        assert!(
            source.contains(required),
            "missing concrete evidence construction {required}"
        );
    }
    for forbidden in [
        "BrokerFinancialTruthCapabilityV1",
        "install_broker_financial_truth",
        "permit",
        "std::env",
        "current_dir(",
        "read_dir(",
        "glob(",
        "Default::default",
        "unwrap_or",
        "fallback",
    ] {
        assert!(
            !source.contains(forbidden),
            "Task5a ingress contains prohibited authority/discovery token {forbidden}"
        );
    }
}
