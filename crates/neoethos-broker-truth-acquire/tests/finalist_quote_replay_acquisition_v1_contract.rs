//! RED-only contract for bounded quote acquisition at the two finalist replay seams.
//!
//! Canonical cTrader trendbars remain the sole feature/search/training dataset.
//! Historical Bid/Ask ticks may be captured only for the already-locked outer
//! holdout/OOS execution replay, including explicit seed and exit padding.

use std::fs;
use std::path::{Path, PathBuf};

use neoethos_broker_history::ProductionBrokerTruthCancellationV2;
use neoethos_broker_truth_acquire::{
    FinalistQuoteReplayAcquisitionErrorCodeV1, FinalistQuoteReplayAcquisitionErrorV1,
    FinalistQuoteReplayAcquisitionOutcomeV1, FinalistQuoteReplayAcquisitionRequestV1,
    acquire_finalist_quote_replay_v1,
};

const PRODUCTION_RELATIVE_PATH: &str = "src/finalist_quote_replay_acquisition_v1.rs";

fn crate_root() -> PathBuf {
    option_env!("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-broker-truth-acquire"))
}

fn repository_root() -> PathBuf {
    crate_root()
        .parent()
        .and_then(Path::parent)
        .expect("acquisition crate must live under <repository>/crates")
        .to_path_buf()
}

fn read_crate(relative: &str) -> String {
    let path = crate_root().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read required source {}: {error}", path.display()))
}

fn read_repository(relative: &str) -> String {
    let path = repository_root().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read required source {}: {error}", path.display()))
}

fn production_source() -> String {
    let path = crate_root().join(PRODUCTION_RELATIVE_PATH);
    fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "RED: missing bounded finalist quote acquisition {}: {error}",
            path.display()
        )
    })
}

fn function_body<'a>(source: &'a str, marker: &str) -> &'a str {
    let start = source
        .find(marker)
        .unwrap_or_else(|| panic!("missing function marker {marker:?}"));
    let open = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .expect("function has an opening brace");
    let mut depth = 0_u32;
    for (offset, byte) in source.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("function {marker:?} has no closing brace")
}

fn require_tokens(source: &str, tokens: &[&str]) {
    for token in tokens {
        assert!(
            source.contains(token),
            "missing finalist quote-acquisition contract token `{token}`"
        );
    }
}

#[test]
fn public_surface_is_versioned_one_shot_and_error_typed() {
    let entrypoint: fn(
        FinalistQuoteReplayAcquisitionRequestV1,
        &ProductionBrokerTruthCancellationV2,
    ) -> Result<
        FinalistQuoteReplayAcquisitionOutcomeV1,
        FinalistQuoteReplayAcquisitionErrorV1,
    > = acquire_finalist_quote_replay_v1;
    assert_ne!(entrypoint as usize, 0);
    assert!(std::mem::size_of::<FinalistQuoteReplayAcquisitionErrorCodeV1>() > 0);

    let source = production_source();
    let library = read_crate("src/lib.rs");
    require_tokens(
        &source,
        &[
            "pub struct FinalistQuoteReplayAcquisitionRequestV1",
            "pub struct FinalistQuoteReplayAcquisitionOutcomeV1",
            "pub enum FinalistQuoteReplayAcquisitionErrorCodeV1",
            "pub fn acquire_finalist_quote_replay_v1(",
            "FinalistQuoteReplayRestartPolicyV1::RestartWholeBoundedCaptureOnce",
        ],
    );
    require_tokens(
        &library,
        &[
            "mod finalist_quote_replay_acquisition_v1;",
            "FinalistQuoteReplayAcquisitionRequestV1",
            "FinalistQuoteReplayAcquisitionOutcomeV1",
            "acquire_finalist_quote_replay_v1",
        ],
    );
}

#[test]
fn locked_finalist_scope_and_padding_are_exact_not_inferred() {
    let source = production_source();
    require_tokens(
        &source,
        &[
            "CanonicalSearchArtifactScopeV2",
            "LockedFinalistOosReplayScopeV1",
            "canonical_search_input_receipt_sha256",
            "canonical_signal_plan_sha256",
            "portfolio_identity_sha256",
            "search_config_hash",
            "holdout_scope_identity_sha256",
            "locked_evaluation_window",
            "required_quote_coverage_window",
            "seed_padding_ms",
            "exit_padding_ms",
            "FinalistScopeMismatch",
            "PaddingMismatch",
            "MAX_FINALIST_QUOTE_REPLAY_WINDOW_MS_V1",
        ],
    );
    for forbidden in [
        "CanonicalSearchArtifactScopeV2::for_entire_receipt",
        "read_current_manifest",
        "current_generation",
        "latest_generation",
        "unwrap_or_default",
    ] {
        assert!(
            !source.contains(forbidden),
            "finalist scope is inferred or defaulted through `{forbidden}`"
        );
    }
}

#[test]
fn capture_is_same_session_v2_chunked_paged_and_restarts_whole_window_only() {
    let source = production_source();
    let acquire = function_body(&source, "pub fn acquire_finalist_quote_replay_v1(");
    require_tokens(
        acquire,
        &[
            "ProductionBrokerTruthCaptureRequestV2",
            "capture_production_broker_financial_truth_v2",
            "CTraderBrokerTruthAdapterV2",
            "capture_and_publish_broker_financial_truth_v2",
            "MAX_CTRADER_TICK_REQUEST_SPAN_MS_V2",
            "response_has_more",
            "FinalistQuoteReplayRestartPolicyV1::RestartWholeBoundedCaptureOnce",
            "restart_whole_bounded_capture",
            "SameSessionCaptureRequired",
            "IncompletePageCoverage",
        ],
    );
    for forbidden in [
        "resume_partial_capture",
        "adopt_partial_bundle",
        "append_partial_pages",
        "cross_session_pages",
        "publish_v1(",
    ] {
        assert!(
            !acquire.contains(forbidden),
            "bounded V2 capture contains forbidden partial/cross-session route `{forbidden}`"
        );
    }
}

#[test]
fn actual_bft2_manifest_is_bound_only_after_capture_then_link_is_reopened() {
    let source = production_source();
    let acquire = function_body(&source, "pub fn acquire_finalist_quote_replay_v1(");
    require_tokens(
        acquire,
        &[
            "BrokerTruthAcquisitionStoreV1::new",
            ".open_authority(",
            "BrokerFinancialTruthBundleStoreV1::new",
            ".open_exact_v2(",
            "broker_truth_receipt.manifest_sha256()",
            "QuoteValidatedResearchReplayBindingV1::new",
            ".publish_link(",
            ".open_link(",
            "actual_quote_evidence_manifest_sha256",
            "TwoPhaseManifestBindingMismatch",
        ],
    );
    let capture = acquire
        .find("capture_production_broker_financial_truth_v2")
        .expect("same-session BFT2 capture");
    let actual_manifest = acquire
        .find("broker_truth_receipt.manifest_sha256()")
        .expect("post-capture actual BFT2 manifest identity");
    let replay_binding = acquire
        .find("QuoteValidatedResearchReplayBindingV1::new")
        .expect("post-capture replay binding");
    let link = acquire
        .find(".publish_link(")
        .expect("post-binding immutable acquisition link");
    assert!(
        capture < actual_manifest && actual_manifest < replay_binding && replay_binding < link,
        "actual BFT2 manifest must be learned after capture and bound before link publication"
    );
}

#[test]
fn zero_rows_incomplete_pages_or_tamper_are_evidence_errors_never_no_fill() {
    let source = production_source();
    require_tokens(
        &source,
        &[
            "inspect_untrusted_broker_financial_truth_bundle_v2",
            "event_count() == 0",
            "response_has_more()",
            "ZeroRowQuoteCoverage",
            "IncompletePageCoverage",
            "ArtifactDigestMismatch",
            "CoverageWindowMismatch",
            "CaptureEvidenceInvalid",
        ],
    );
    for forbidden in [
        "NoEligibleQuoteWithinEntryWait",
        "EntryUnavailable",
        "ExactZeroRowQuoteWindowProofV1::new",
        "empty_is_no_fill",
    ] {
        assert!(
            !source.contains(forbidden),
            "missing acquisition evidence is misclassified as replay outcome via `{forbidden}`"
        );
    }
}

#[test]
fn output_is_research_only_with_merge_none_and_never_authorizes_promotion() {
    let source = production_source();
    require_tokens(
        &source,
        &[
            "QuoteValidatedResearchReplayPolicyV1::new",
            "reviewed_same_timestamp_merge_rule: None",
            "FinalistQuoteReplayArtifactClassV1::ResearchOnly",
            "BrokerTruthSemanticStatusV1::UnvalidatedEvidenceOnly",
            "BrokerTruthPromotionEligibilityV1::NotPromotionEligible",
            "QuoteValidatedResearchReplayBindingV1",
            "BrokerTruthAcquisitionLinkReceiptV1",
        ],
    );
    for forbidden in [
        "SameTimestampCrossSideOrderV1::BidBeforeAsk",
        "SameTimestampCrossSideOrderV1::AskBeforeBid",
        "BrokerTruthPromotionEligibilityV1::PromotionEligible",
        "BrokerFinancialTruthCapabilityV1",
        "BrokerFinancialTruthPermitV1",
        "install_broker_financial_truth",
        "permit_issued: true",
    ] {
        assert!(
            !source.contains(forbidden),
            "acquisition creates caller-selected ordering or promotion authority via `{forbidden}`"
        );
    }
}

#[test]
fn quote_acquisition_cannot_enter_ga_cpcv_features_or_bulk_trendbar_research() {
    let source = production_source();
    for forbidden in [
        "FeatureFrame",
        "run_discovery_cycle",
        "CombinatorialPurgedCV",
        "Cpcv",
        "prepare_multitimeframe_features",
        "resample",
        "indicator",
    ] {
        assert!(
            !source.contains(forbidden),
            "finalist quote acquisition leaks into research/features through `{forbidden}`"
        );
    }

    let discovery = read_repository("crates/neoethos-search/src/discovery.rs");
    let numerical = function_body(
        &discovery,
        "fn run_discovery_cycle_values_with_progress<F>(",
    );
    let genetic = read_repository("crates/neoethos-search/src/genetic/search_engine.rs");
    let canonical_trendbars =
        read_repository("crates/neoethos-search/src/canonical_trendbar_research.rs");
    for (name, consumer) in [
        ("numerical discovery", numerical),
        ("genetic search", genetic.as_str()),
        ("canonical trendbar research", canonical_trendbars.as_str()),
    ] {
        for forbidden in [
            "acquire_finalist_quote_replay_v1",
            "FinalistQuoteReplayAcquisitionRequestV1",
            "CTraderBrokerTruthAdapterV2",
        ] {
            assert!(
                !consumer.contains(forbidden),
                "{name} improperly consumes finalist quote acquisition via `{forbidden}`"
            );
        }
    }
}
