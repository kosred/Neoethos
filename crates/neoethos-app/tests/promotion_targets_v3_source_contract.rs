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
fn writer_copies_the_saved_discovery_result_authority_without_reconstruction() {
    assert_contains_all(
        DISCOVERY,
        &[
            "save_promotion_summary_json(&summary_path, result)",
            "CanonicalSearchArtifactEnvelopeV2::<PromotionSummaryAuthorityPayloadV3>::from_json_bytes",
            "search_input_receipt: result.search_input_receipt.clone()",
            "search_input_receipt_sha256: result.search_input_receipt_sha256()?",
            "search_config_hash: result.search_config_hash.clone()",
            "promotion_summary_authority: StoredPromotionSummaryAuthorityV3",
        ],
    );

    let save = DISCOVERY
        .find("save_promotion_summary_json(&summary_path, result)")
        .expect("canonical promotion summary save");
    let targets_write = DISCOVERY[save..]
        .find("write_json_atomic(&path, &file)")
        .map(|offset| save + offset)
        .expect("model_targets atomic write after summary authority");
    assert!(
        save < targets_write,
        "authority must exist before model_targets v3 is published"
    );
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
        .find("fn authorize_model_targets_for_promotion")
        .expect("promotion loader");
    let body = &STRATEGY_LAB[authorize..];
    let validate = body
        .find("validate_promotion_path_leafs")
        .expect("safe leaf validation");
    let read_path = body
        .find("model_targets_path_for")
        .expect("model_targets path construction");
    assert!(
        validate < read_path,
        "path leaves must be proven before the first artifact path"
    );
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
