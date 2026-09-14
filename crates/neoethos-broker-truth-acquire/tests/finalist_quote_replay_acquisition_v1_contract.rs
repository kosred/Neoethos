//! Source contracts for bounded quote acquisition and reviewed replay consumers.
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
fn captured_finalist_has_an_independently_reviewed_quote_replay_consumer() {
    let source = production_source();
    let single = function_body(&source, "pub fn replay_reviewed_decision_v1(");
    require_tokens(single, &["self.replay_reviewed_decision_sequence_v1("]);
    let replay = function_body(&source, "pub fn replay_reviewed_decision_sequence_v1(");
    let open = function_body(&source, "pub fn open_reviewed_quote_snapshot_v1(");
    require_tokens(
        replay,
        &[
            "reviewed: ReviewedBrokerFinancialTruthEvidenceV2",
            "self.replay_binding.clone()",
            "self.replay_policy.clone()",
            "QuoteValidatedResearchReplayPlanV1::validate_ordered_sequence(&plans)",
            "self.open_reviewed_quote_snapshot_v1(store, reviewed)",
            "replay_sealed_quote_validated_decision_sequence_v1(&plans, &evidence)",
        ],
    );
    assert!(
        replay.find("QuoteValidatedResearchReplayPlanV1::validate_ordered_sequence(&plans)")
            < replay.find("self.open_reviewed_quote_snapshot_v1(store, reviewed)")
    );
    require_tokens(
        open,
        &[
            "validate_reviewed_broker_financial_truth_authority_v2(",
            "&self.acquisition_link_receipt",
            "into_sealed_historical_bid_ask_quote_replay_evidence_v2(",
            "&self.replay_binding",
        ],
    );
    assert!(
        open.find("validate_reviewed_broker_financial_truth_authority_v2(")
            < open.find("into_sealed_historical_bid_ask_quote_replay_evidence_v2(")
    );
    assert!(
        replay.find("self.open_reviewed_quote_snapshot_v1(store, reviewed)")
            < replay.find("replay_sealed_quote_validated_decision_sequence_v1(")
    );
    for forbidden in [
        "ReviewedBrokerFinancialTruthEvidenceV2::checked_new",
        "current_broker_financial_truth",
        "capture_production_broker_financial_truth_v2",
        "open_sealed_historical_bid_ask_quote_replay_evidence_v1",
    ] {
        assert!(
            !replay.contains(forbidden) && !single.contains(forbidden) && !open.contains(forbidden),
            "replay fabricates review or recaptures evidence: {forbidden}"
        );
    }
}

#[test]
fn canonical_signals_reuse_the_position_engine_and_the_single_reviewed_snapshot() {
    let source = production_source();
    let replay = function_body(&source, "pub fn replay_reviewed_signal_lane_v1(");
    require_tokens(
        replay,
        &[
            "lane.validate(&self.replay_binding, &self.replay_policy)",
            "self.open_reviewed_quote_snapshot_v1(store, reviewed)",
            "neoethos_trader::data_replay::replay_canonical_signal_quote_lane_v1(",
            "&self.replay_binding",
            "&self.replay_policy",
            "&evidence",
        ],
    );
    assert!(replay.find("lane.validate(") < replay.find("self.open_reviewed_quote_snapshot_v1("));
    let driver = read_repository("crates/neoethos-trader/src/quote_signal_replay.rs");
    require_tokens(
        &driver,
        &[
            "DecisionEngine::new(DecisionConfig::gene_parity(lane.pip_size))",
            "trailing.next_stop_price(",
            "preview_sealed_quote_validated_research_entry_v1(",
            "replay_sealed_quote_validated_research_v1(",
            "ClosedCanonicalBarTimeExitV1::new(",
            "entry.bid_extrema_before(end)",
        ],
    );
    let position = read_repository("crates/neoethos-trader/src/position.rs");
    require_tokens(&position, &["next_stop_price("]);
    let locked = function_body(
        &driver,
        "pub fn replay_locked_canonical_signal_portfolio_v3(",
    );
    require_tokens(
        locked,
        &[
            "cpu.require_current_pool()",
            "locked.validate_replay_binding(binding, policy)?",
            "evidence.validate_replay_context_v1(binding, policy)?",
            "QuoteBars::Canonical",
            ".par_iter()",
            "confidences: Some(&locked.ordered_confidences()[index])",
            "locked.entry_eligible(index, row)",
            "locked.entry_stop_target_pips(index, row)",
            "entry_eligibility: Some(&eligible)",
            "entry_brackets: Some(&brackets)",
        ],
    );
    for forbidden in ["Vec<LiveBar>", "Position::new("] {
        assert!(
            !driver.contains(forbidden),
            "replay rebuilt owned historical bars or a fake position"
        );
    }
    for forbidden in [
        "MockExecutionAdapter",
        "EngineStats",
        "current_broker_financial_truth_capability_v1",
        "signals_for_gene",
    ] {
        assert!(
            !driver.contains(forbidden),
            "signal replay has an unrelated fallback or duplicate signal path: {forbidden}"
        );
    }
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
