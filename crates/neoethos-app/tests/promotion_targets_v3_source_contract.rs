//! Focused, model-free source contract for the promotion authorization hand-off.
//!
//! Run directly with `rustc --test` so the unrelated neoethos-models build wall
//! cannot turn this fail-closed app boundary into an untestable promise.

const DISCOVERY: &str = include_str!("../src/app_services/discovery.rs");
const STRATEGY_LAB: &str = include_str!("../src/server/strategy_lab.rs");
const AUTHORIZATION: &str = include_str!("../src/server/promotion_authorization.rs");

fn assert_contains_all(source: &str, required: &[&str]) {
    for needle in required {
        assert!(
            source.contains(needle),
            "missing required source contract: {needle}"
        );
    }
}

#[test]
fn model_targets_v3_embeds_the_exact_search_and_promotion_authority() {
    assert_contains_all(
        DISCOVERY,
        &[
            "pub const MODEL_TARGETS_SCHEMA_VERSION: u32 = 3;",
            "pub search_input_receipt: CanonicalSearchInputReceiptV2",
            "pub search_input_receipt_sha256: String",
            "pub search_config_hash: String",
            "pub promotion_summary_authority: StoredPromotionSummaryAuthorityV3",
            "pub envelope: CanonicalSearchArtifactEnvelopeV2<PromotionSummaryAuthorityPayloadV3>",
            "#[serde(deny_unknown_fields)]",
        ],
    );
}

#[test]
fn v3_portfolio_entries_do_not_keep_permissive_v1_defaults() {
    let entry_start = DISCOVERY
        .find("pub struct ModelTargetEntry")
        .expect("model target entry");
    let entry_end = DISCOVERY[entry_start..]
        .find("pub const MODEL_TARGETS_SCHEMA_VERSION")
        .map(|offset| entry_start + offset)
        .expect("schema constant after entry");
    let entry = &DISCOVERY[entry_start..entry_end];
    assert!(
        !entry.contains("#[serde(default)]"),
        "v3 must reject missing promotion metrics instead of filling permissive v1 defaults"
    );
}

#[test]
fn desktop_research_output_keeps_its_classification_and_never_writes_model_targets() {
    assert_contains_all(
        DISCOVERY,
        &[
            "run_canonical_trendbar_research_discovery_with_holdout_and_progress(",
            "run_prepared_canonical_trendbar_research_with_holdout_and_progress_v3(",
            "save_discovery_research_result(",
            "result: &neoethos_search::CanonicalTrendbarResearchDiscoveryResultV3",
            "result.execution_contract().assumption_source_sha256()",
            "NotPromotionEligible",
            "write_json_atomic_compact(&path, result)",
        ],
    );
    for removed in [
        "write_model_targets_for_discovery(",
        "save_portfolio_json(",
        "save_promotion_summary_json(",
    ] {
        assert!(
            !DISCOVERY.contains(removed),
            "research output leaked into {removed}"
        );
    }
    // A separate, schema-validated bar-OOS portfolio is allowed. It is not a
    // ModelTargetsV3 promotion permit and may not erase the research envelope.
    let worker = DISCOVERY
        .split_once("pub fn start_discovery_job(")
        .unwrap()
        .1
        .split_once("\n/// On-disk contract between Discovery output")
        .unwrap()
        .0;
    let research = worker
        .find("let output_path = save_discovery_research_result(")
        .unwrap();
    let nonempty = worker.find("ensure_non_empty_portfolio(").unwrap();
    let save = worker.find("save_live_portfolio_json(").unwrap();
    let reload = worker.find("load_live_portfolio_json(").unwrap();
    let handoff = worker
        .find("PromotionCandidateTrainingHandoffV1::from_discovery_portfolio(")
        .unwrap();
    assert!(research < nonempty && nonempty < save && save < reload && reload < handoff);
    let artifact = include_str!("../../neoethos-search/src/live_portfolio.rs");
    let writer = artifact
        .split_once("pub fn save_live_portfolio_json(")
        .unwrap()
        .1
        .split_once("/// Load a live portfolio artifact")
        .unwrap()
        .0;
    assert!(
        writer
            .find("LivePortfolioArtifact::from_discovery(")
            .unwrap()
            < writer.find("artifact.validate()?").unwrap()
    );
    assert!(
        writer.find("artifact.validate()?").unwrap() < writer.find("write_json_atomic(").unwrap()
    );
    let typed = include_str!("../src/server/engines_control/typed_execution_v1.rs");
    assert!(typed.contains("if intent.training_after_success {"));
    let binding = typed
        .split_once("let selected = handoff::load(")
        .unwrap()
        .1
        .split_once("let settings = handoff::settings_for_series(")
        .unwrap()
        .0;
    assert_contains_all(
        binding,
        &[
            "validate_follow_on_series_v1(",
            "expected_series.as_ref()",
            "selected.canonical_series()",
        ],
    );
    let discovery_worker = typed
        .split_once("fn spawn_discovery_worker_v1(")
        .unwrap()
        .1
        .split_once("fn spawn_training_worker_v1(")
        .unwrap()
        .0;
    assert!(discovery_worker.contains("let mut lease = lease;"));
    assert!(discovery_worker.contains("continue_discovery_training_v1("));
    assert!(discovery_worker.contains("Some(expected_series)"));
    assert!(!discovery_worker.contains("start_training_job("));
    assert!(!discovery_worker.contains("start_typed_training_execution_v1("));
    let continuation = typed
        .split_once("async fn continue_discovery_training_v1<")
        .unwrap()
        .1
        .split_once("async fn prepare_discovery_request_v1(")
        .unwrap()
        .0;
    assert_contains_all(
        continuation,
        &[
            "discovery_snapshot.state != JobState::Succeeded",
            "handoffs.next().is_some()",
            "handoff::handoff_path(",
            "TypedTrainingSelectionPolicyV1::DiscoveryHandoff",
            "transition_discovery_to_training_v1()",
        ],
    );
    for forbidden in [
        "TypedTrainingSelectionPolicyV1::Exact",
        "copy_model_tree",
        "promote_if_gated",
    ] {
        assert!(
            !continuation.contains(forbidden),
            "candidate continuation must not mint live authority: {forbidden}"
        );
    }
}

#[test]
fn loader_has_typed_fail_closed_schema_and_exact_binding_checks() {
    assert_contains_all(
        STRATEGY_LAB,
        &[
            "schema_version != MODEL_TARGETS_SCHEMA_VERSION",
            "actual_authority != file.promotion_summary_authority.envelope",
            ".validate_against(",
            ".identity_sha256()",
            "StatusCode::PRECONDITION_FAILED",
        ],
    );
    assert_contains_all(
        AUTHORIZATION,
        &[
            "pub(crate) enum PromotionAuthorizationError",
            "UnsupportedSchema",
            "ReceiptDigestMismatch",
            "PromotionSummaryMismatch",
            "UnsupportedEvidenceSchema",
            "MissingHeldOutEvidence",
            "FailedHeldOutEvidence",
        ],
    );
}

#[test]
fn current_v3_summary_still_cannot_mint_a_copy_permit_without_composite_scope() {
    assert_contains_all(
        STRATEGY_LAB,
        &[
            "authorize_exact_composite_promotion_v3",
            "REQUIRED_COMPOSITE_PROMOTION_AUTHORITY_KIND_V3",
            "CompositeAuthorityChecksV3",
            "exact_composite_scope: false",
            "required_evidence_complete: false",
            "required_evidence_passed: false",
        ],
    );
    assert!(
        !STRATEGY_LAB.contains("require_passing_promotion_evidence(actual_authority.payload())"),
        "a non-composite payload must not authorize live copy"
    );
}

#[test]
fn path_leaves_are_validated_before_any_model_target_or_live_path_is_built() {
    assert_contains_all(
        STRATEGY_LAB,
        &[
            "validate_promotion_path_leafs",
            ".parse::<CanonicalTimeframe>()",
            "copy_model_tree_if_authorized",
        ],
    );
    let authorize = STRATEGY_LAB
        .split_once("fn authorize_model_targets_for_promotion(")
        .expect("opaque promotion authorizer")
        .1
        .split_once("fn read_model_targets_for_promotion(")
        .expect("permit-free reader follows authorizer")
        .0;
    let validate = authorize.find("validate_promotion_path_leafs").unwrap();
    let load = authorize
        .find("read_model_targets_for_promotion(data_root")
        .unwrap();
    let permit = authorize
        .find("authorize_exact_composite_promotion_v3(")
        .unwrap();
    assert!(
        validate < load && load < permit,
        "POST validates, reads, then authorizes"
    );
    assert_contains_all(
        authorize,
        &[
            "exact_receipt_config_sidecar: true",
            "exact_composite_scope: false",
            "required_evidence_complete: false",
            "required_evidence_passed: false",
        ],
    );

    let reader = STRATEGY_LAB
        .split_once("fn read_model_targets_for_promotion(")
        .expect("shared exact reader")
        .1
        .split_once("// ─── POST /strategy_lab/promote")
        .expect("end of reader")
        .0;
    let validate = reader.find("validate_promotion_path_leafs").unwrap();
    let read_path = reader.find("model_targets_path_for").unwrap();
    assert!(
        validate < read_path,
        "GET and POST share validation before any artifact path"
    );
    assert!(!reader.contains("authorize_exact_composite_promotion_v3"));

    let diagnostic = STRATEGY_LAB
        .split_once("fn evaluate_promotion_for(")
        .expect("GET diagnostic entry")
        .1
        .split_once("fn evaluate_authorized_promotion_for(")
        .expect("separate POST authorizer")
        .0;
    assert!(diagnostic.contains("read_model_targets_for_promotion("));
    for forbidden in [
        "current_broker_financial_truth_capability_v1",
        "authorize_model_targets_for_promotion(",
        "authorize_exact_composite_promotion_v3(",
        "copy_model_tree_if_authorized(",
    ] {
        assert!(
            !diagnostic.contains(forbidden),
            "GET cannot invoke {forbidden}"
        );
    }
}

#[test]
fn opaque_authorization_precedes_the_only_live_copy_call() {
    assert_contains_all(
        STRATEGY_LAB,
        &[
            "authorize_exact_composite_promotion_v3",
            "copy_model_tree_if_authorized",
        ],
    );

    let promote = STRATEGY_LAB
        .find("fn promote_if_gated")
        .expect("authoritative promotion function");
    let body = &STRATEGY_LAB[promote..];
    let authorization = body
        .find("evaluate_authorized_promotion_for")
        .expect("promotion authorization/evaluation");
    let copy = body
        .find("copy_model_tree_if_authorized")
        .expect("permit-gated artifact copy");
    assert!(authorization < copy);
}
