use neoethos_broker_truth::{
    BrokerFinancialOperationV1, BrokerFinancialTruthContractErrorCodeV1,
    BrokerTruthAcquisitionArtifactRoleV1, BrokerTruthAcquisitionArtifactV1,
    BrokerTruthAcquisitionAuthorityManifestV1, BrokerTruthAcquisitionPromotionEligibilityV1,
    BrokerTruthAcquisitionSemanticStatusV1, BrokerTruthReviewedSynchronizationBindingV1,
    EvidenceWindowV1, ReviewedQuoteReplayRuleIdentityV2,
    current_broker_financial_truth_capability_v1,
};

const WINDOW_FROM_MS: i64 = 1_700_000_000_000;
const WINDOW_TO_MS: i64 = WINDOW_FROM_MS + 60_000;

fn digest(byte: u8) -> String {
    format!("{byte:02x}").repeat(32)
}

fn artifact(
    role: BrokerTruthAcquisitionArtifactRoleV1,
    relative_path: &str,
    digest_byte: u8,
) -> BrokerTruthAcquisitionArtifactV1 {
    BrokerTruthAcquisitionArtifactV1::new(role, relative_path, digest(digest_byte), 17)
        .expect("valid immutable acquisition artifact")
}

fn complete_manifest() -> BrokerTruthAcquisitionAuthorityManifestV1 {
    let review_identity =
        ReviewedQuoteReplayRuleIdentityV2::new(digest(0x41), digest(0x42), digest(0x44))
            .expect("review identity");
    let synchronization = BrokerTruthReviewedSynchronizationBindingV1::new(
        0,
        7,
        42,
        EvidenceWindowV1::new(WINDOW_FROM_MS, WINDOW_TO_MS).expect("evidence window"),
        review_identity,
        digest(0x45),
    )
    .expect("reviewed synchronization binding");
    let artifacts = vec![
        artifact(
            BrokerTruthAcquisitionArtifactRoleV1::CanonicalSearchInputReceipt,
            "canonical-search-input-receipt.json",
            0x31,
        ),
        artifact(
            BrokerTruthAcquisitionArtifactRoleV1::CanonicalSearchArtifactScope,
            "canonical-search-artifact-scope.json",
            0x32,
        ),
        artifact(
            BrokerTruthAcquisitionArtifactRoleV1::CanonicalRootVerificationReceipt,
            "canonical-root-verification.json",
            0x34,
        ),
        artifact(
            BrokerTruthAcquisitionArtifactRoleV1::CanonicalScopeWindowBinding,
            "canonical-scope-window-binding.json",
            0x35,
        ),
        artifact(
            BrokerTruthAcquisitionArtifactRoleV1::CapturePlan,
            "broker-truth-capture-plan.json",
            0x33,
        ),
        artifact(
            BrokerTruthAcquisitionArtifactRoleV1::ReviewRecord,
            "quote-replay-review-record.json",
            0x41,
        ),
        artifact(
            BrokerTruthAcquisitionArtifactRoleV1::ProtocolEvidence,
            "ctrader-protocol-evidence.bin",
            0x42,
        ),
        artifact(
            BrokerTruthAcquisitionArtifactRoleV1::TrustRoot,
            "quote-review-trust-root.pub",
            0x43,
        ),
        artifact(
            BrokerTruthAcquisitionArtifactRoleV1::QuoteSessionObservations { ordinal: 0 },
            "quote-session-observations-000.vortex",
            0x44,
        ),
        artifact(
            BrokerTruthAcquisitionArtifactRoleV1::ReviewedQuoteReplayRules { ordinal: 0 },
            "reviewed-quote-replay-rules-000.vortex",
            0x45,
        ),
    ];

    BrokerTruthAcquisitionAuthorityManifestV1::new(
        digest(0x51),
        digest(0x52),
        digest(0x34),
        digest(0x35),
        digest(0x33),
        digest(0x43),
        artifacts,
        vec![synchronization],
    )
    .expect("complete acquisition authority")
}

#[test]
fn complete_authority_is_permanently_evidence_only_and_content_bound() {
    let manifest = complete_manifest();

    assert_eq!(
        manifest.semantic_status(),
        BrokerTruthAcquisitionSemanticStatusV1::UnvalidatedEvidenceOnly
    );
    assert_eq!(
        manifest.promotion_eligibility(),
        BrokerTruthAcquisitionPromotionEligibilityV1::NotPromotionEligible
    );
    assert_eq!(
        manifest.canonical_search_input_receipt_sha256(),
        digest(0x51)
    );
    assert_eq!(
        manifest.canonical_search_artifact_scope_sha256(),
        digest(0x52)
    );
    assert_eq!(manifest.canonical_root_verification_sha256(), digest(0x34));
    assert_eq!(
        manifest.canonical_scope_window_binding_sha256(),
        digest(0x35)
    );
    assert_eq!(manifest.capture_plan_sha256(), digest(0x33));
    assert_eq!(manifest.expected_trust_root_sha256(), digest(0x43));
    assert_eq!(manifest.reviewed_synchronizations().len(), 1);
    assert_eq!(manifest.reviewed_synchronizations()[0].ordinal(), 0);
    assert_eq!(manifest.reviewed_synchronizations()[0].account_id(), 7);
    assert_eq!(manifest.reviewed_synchronizations()[0].symbol_id(), 42);
    assert_eq!(
        manifest.reviewed_synchronizations()[0].window(),
        EvidenceWindowV1::new(WINDOW_FROM_MS, WINDOW_TO_MS).expect("same evidence window")
    );

    let canonical = manifest
        .canonical_json_bytes()
        .expect("canonical authority JSON");
    let reopened = BrokerTruthAcquisitionAuthorityManifestV1::from_json_bytes(&canonical)
        .expect("strict authority reopen");
    assert_eq!(reopened, manifest);
    assert_eq!(
        reopened.identity_sha256().expect("authority identity"),
        manifest.identity_sha256().expect("same authority identity")
    );

    current_broker_financial_truth_capability_v1()
        .require(BrokerFinancialOperationV1::HistoricalEvaluation)
        .expect_err("acquisition authority must not install or mint a financial permit");
}

#[test]
fn trust_or_artifact_digest_tampering_is_refused_on_strict_reopen() {
    let manifest = complete_manifest();
    let mut value: serde_json::Value = serde_json::from_slice(
        &manifest
            .canonical_json_bytes()
            .expect("canonical authority JSON"),
    )
    .expect("authority JSON value");
    value["expected_trust_root_sha256"] = serde_json::Value::String(digest(0x7f));

    let error = BrokerTruthAcquisitionAuthorityManifestV1::from_json_bytes(
        &serde_json::to_vec(&value).expect("tampered JSON"),
    )
    .expect_err("trust-root digest detached from the frozen artifact must fail");
    assert_eq!(
        error.code(),
        BrokerFinancialTruthContractErrorCodeV1::InvalidManifest
    );

    let mut value: serde_json::Value = serde_json::from_slice(
        &manifest
            .canonical_json_bytes()
            .expect("canonical authority JSON"),
    )
    .expect("authority JSON value");
    value["artifacts"][8]["sha256"] = serde_json::Value::String(digest(0x7e));
    let error = BrokerTruthAcquisitionAuthorityManifestV1::from_json_bytes(
        &serde_json::to_vec(&value).expect("tampered JSON"),
    )
    .expect_err("observation bytes detached from the review identity must fail");
    assert_eq!(
        error.code(),
        BrokerFinancialTruthContractErrorCodeV1::InvalidManifest
    );
}

#[test]
fn missing_or_noncontiguous_reviewed_synchronization_pairs_are_refused() {
    let manifest = complete_manifest();
    let artifacts = manifest
        .artifacts()
        .iter()
        .filter(|artifact| {
            artifact.role()
                != BrokerTruthAcquisitionArtifactRoleV1::ReviewedQuoteReplayRules { ordinal: 0 }
        })
        .cloned()
        .collect();
    let error = BrokerTruthAcquisitionAuthorityManifestV1::new(
        manifest.canonical_search_input_receipt_sha256(),
        manifest.canonical_search_artifact_scope_sha256(),
        manifest.canonical_root_verification_sha256(),
        manifest.canonical_scope_window_binding_sha256(),
        manifest.capture_plan_sha256(),
        manifest.expected_trust_root_sha256(),
        artifacts,
        manifest.reviewed_synchronizations().to_vec(),
    )
    .expect_err("every reviewed synchronization needs an exact rules artifact");
    assert_eq!(
        error.code(),
        BrokerFinancialTruthContractErrorCodeV1::MissingEvidence
    );

    let review_identity =
        ReviewedQuoteReplayRuleIdentityV2::new(digest(0x41), digest(0x42), digest(0x44))
            .expect("review identity");
    let noncontiguous = BrokerTruthReviewedSynchronizationBindingV1::new(
        1,
        7,
        42,
        EvidenceWindowV1::new(WINDOW_FROM_MS, WINDOW_TO_MS).expect("evidence window"),
        review_identity,
        digest(0x45),
    )
    .expect("individually shaped binding");
    let error = BrokerTruthAcquisitionAuthorityManifestV1::new(
        manifest.canonical_search_input_receipt_sha256(),
        manifest.canonical_search_artifact_scope_sha256(),
        manifest.canonical_root_verification_sha256(),
        manifest.canonical_scope_window_binding_sha256(),
        manifest.capture_plan_sha256(),
        manifest.expected_trust_root_sha256(),
        manifest.artifacts().to_vec(),
        vec![noncontiguous],
    )
    .expect_err("reviewed synchronization ordinals must start at zero and be contiguous");
    assert_eq!(
        error.code(),
        BrokerFinancialTruthContractErrorCodeV1::InvalidManifest
    );
}
