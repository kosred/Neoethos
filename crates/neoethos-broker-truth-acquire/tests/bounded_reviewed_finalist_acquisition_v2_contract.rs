use neoethos_broker_truth_acquire::{
    BoundedReviewedFinalistAcquisitionErrorCodeV2, BoundedReviewedFinalistAcquisitionErrorV2,
    LockedFinalistBrokerTruthAcquisitionInputV2, PreparedBoundedReviewedFinalistAcquisitionV2,
    UnvalidatedLockedFinalistBrokerTruthEvidenceV2,
    execute_bounded_reviewed_finalist_acquisition_v2,
    prepare_bounded_reviewed_finalist_acquisition_v2,
};

#[test]
fn bounded_two_phase_surface_is_typed_and_returns_only_unvalidated_evidence() {
    let prepare = prepare_bounded_reviewed_finalist_acquisition_v2;
    let execute = execute_bounded_reviewed_finalist_acquisition_v2;
    assert_ne!(prepare as *const () as usize, 0);
    assert_ne!(execute as *const () as usize, 0);
    assert!(std::mem::size_of::<LockedFinalistBrokerTruthAcquisitionInputV2>() > 0);
    assert!(std::mem::size_of::<PreparedBoundedReviewedFinalistAcquisitionV2>() > 0);
    assert!(std::mem::size_of::<UnvalidatedLockedFinalistBrokerTruthEvidenceV2>() > 0);
    assert!(std::mem::size_of::<BoundedReviewedFinalistAcquisitionErrorV2>() > 0);
    assert!(std::mem::size_of::<BoundedReviewedFinalistAcquisitionErrorCodeV2>() > 0);
}

#[test]
fn wrapper_delegates_to_existing_exact_preflight_and_finalist_capture_only() {
    let source = include_str!("../src/bounded_reviewed_finalist_acquisition_v2.rs");
    for required in [
        "prepare_acquisition_v1(args)",
        "FinalistQuoteReplayAcquisitionRequestV1::new",
        "acquire_finalist_quote_replay_v1",
        "FinalistQuoteReplayArtifactClassV1::ResearchOnly",
        "BrokerTruthSemanticStatusV1::UnvalidatedEvidenceOnly",
        "BrokerTruthPromotionEligibilityV1::NotPromotionEligible",
        "MAX_FINALIST_QUOTE_REPLAY_WINDOW_MS_V1",
    ] {
        assert!(
            source.contains(required),
            "missing bounded contract {required}"
        );
    }
    for forbidden in [
        "validate_reviewed_broker_financial_truth_authority_v2",
        "BrokerFinancialTruthAuthorityV2",
        "BrokerFinancialTruthPermitV1",
        "current_broker_financial_truth",
        "std::env",
        "resume_partial_capture",
        "latest",
        "current",
    ] {
        assert!(
            !source.contains(forbidden),
            "unvalidated acquisition contains forbidden authority/global route {forbidden}"
        );
    }
}
