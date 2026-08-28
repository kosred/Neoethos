use super::*;
use chrono::{Duration, Utc};

fn canonical_provenance(artifact_kind: ArtifactKind) -> ArtifactProvenance {
    ArtifactProvenance::new(
        artifact_kind,
        "feature-schema-v1",
        "dataset-a",
        "symbols-eurusd",
        "m1-m5",
        "timestamp-policy-v1",
        "feature-availability-v1",
        "label-policy-v1",
        "training-config-v1",
        "search-config-v1",
        "runtime-config-v1",
        "risk-config-v1",
        DeterminismPolicy::Deterministic { seed: 42 },
        "hardware-profile-1",
        DeviceAssignment::cpu(),
        BackendKind::NativeCpu,
        RuntimeMode::Canonical,
        None,
        "abc1234",
    )
    .expect("fixture provenance should be valid")
}

fn canonical_live_provenance() -> ArtifactProvenance {
    canonical_provenance(ArtifactKind::LiveReadyStrategy)
}

#[test]
fn artifact_refuses_missing_provenance() {
    let mut provenance = canonical_live_provenance();
    provenance.feature_schema_hash.clear();

    let error = ArtifactEnvelope::new(provenance, ()).expect_err("empty hash must fail");

    assert_eq!(
        error,
        ArtifactContractError::MissingProvenanceField("feature_schema_hash")
    );
}

#[test]
fn artifact_refuses_wrong_kind() {
    let envelope = ArtifactEnvelope::new(canonical_live_provenance(), ())
        .expect("fixture envelope should be valid");

    let error = envelope
        .require_kind(ArtifactKind::SearchCheckpoint)
        .expect_err("live-ready artifact must not masquerade as a checkpoint");

    assert_eq!(
        error,
        ArtifactContractError::WrongArtifactKind {
            actual: ArtifactKind::LiveReadyStrategy,
            expected: ArtifactKind::SearchCheckpoint,
        }
    );
}

#[test]
fn typed_contract_accepts_matching_checkpoint_artifact() {
    let checkpoint = SearchCheckpointArtifact::new(
        canonical_provenance(ArtifactKind::SearchCheckpoint),
        "search cursor",
    )
    .expect("checkpoint provenance should satisfy checkpoint contract");

    assert_eq!(checkpoint.contract_kind(), ArtifactKind::SearchCheckpoint);
    assert_eq!(checkpoint.contract_name(), "search_checkpoint_artifact");
}

#[test]
fn typed_contract_refuses_checkpoint_as_training_model() {
    let error = TrainingModelArtifact::new(
        canonical_provenance(ArtifactKind::SearchCheckpoint),
        "model payload",
    )
    .expect_err("checkpoint artifact must not satisfy training model contract");

    assert_eq!(
        error,
        ArtifactContractError::WrongArtifactKind {
            actual: ArtifactKind::SearchCheckpoint,
            expected: ArtifactKind::TrainingModel,
        }
    );
}

#[test]
fn typed_live_contract_enforces_live_readiness() {
    let provenance = ArtifactProvenance::new(
        ArtifactKind::LiveReadyStrategy,
        "feature-schema-v1",
        "dataset-a",
        "symbols-eurusd",
        "m1-m5",
        "timestamp-policy-v1",
        "feature-availability-v1",
        "label-policy-v1",
        "training-config-v1",
        "search-config-v1",
        "runtime-config-v1",
        "risk-config-v1",
        DeterminismPolicy::Deterministic { seed: 42 },
        "hardware-profile-1",
        DeviceAssignment {
            backend: BackendKind::CpuReference,
            device: "cpu".to_string(),
            device_ids: Vec::new(),
        },
        BackendKind::CpuReference,
        RuntimeMode::Degraded,
        Some(RuntimeDegradedReason::new(
            "cuda_unavailable",
            "CUDA was requested but unavailable",
        )),
        "abc1234",
    )
    .expect("degraded provenance is structurally valid");

    let error = LiveReadyStrategyArtifact::new(provenance, "strategy payload")
        .expect_err("typed live-ready contract must reject degraded runtime mode");

    assert_eq!(
        error,
        ArtifactContractError::LiveRejectedRuntimeMode {
            mode: RuntimeMode::Degraded,
            backend: BackendKind::CpuReference,
        }
    );
}

#[test]
fn live_rejects_degraded_artifact() {
    let provenance = ArtifactProvenance::new(
        ArtifactKind::LiveReadyStrategy,
        "feature-schema-v1",
        "dataset-a",
        "symbols-eurusd",
        "m1-m5",
        "timestamp-policy-v1",
        "feature-availability-v1",
        "label-policy-v1",
        "training-config-v1",
        "search-config-v1",
        "runtime-config-v1",
        "risk-config-v1",
        DeterminismPolicy::Deterministic { seed: 42 },
        "hardware-profile-1",
        DeviceAssignment {
            backend: BackendKind::CpuReference,
            device: "cpu".to_string(),
            device_ids: Vec::new(),
        },
        BackendKind::CpuReference,
        RuntimeMode::Degraded,
        Some(RuntimeDegradedReason::new(
            "cuda_unavailable",
            "CUDA was requested but unavailable",
        )),
        "abc1234",
    )
    .expect("degraded provenance is valid but not live-safe");
    let envelope = ArtifactEnvelope::new(provenance, ()).expect("envelope is structurally valid");

    let error = envelope
        .require_live_ready()
        .expect_err("degraded artifact cannot pass live gate");

    assert_eq!(
        error,
        ArtifactContractError::LiveRejectedRuntimeMode {
            mode: RuntimeMode::Degraded,
            backend: BackendKind::CpuReference,
        }
    );
}

#[test]
fn runtime_safety_report_marks_canonical_native_artifact_live_safe() {
    let report = canonical_live_provenance().runtime_safety_report();

    assert!(report.live_safe);
    assert!(report.issues.is_empty());
    assert!(report.degraded_reason.is_none());
    assert!(report.backend_assignment_matches);
}

#[test]
fn runtime_safety_report_exposes_degraded_backend_metadata() {
    let provenance = ArtifactProvenance::new(
        ArtifactKind::LiveReadyStrategy,
        "feature-schema-v1",
        "dataset-a",
        "symbols-eurusd",
        "m1-m5",
        "timestamp-policy-v1",
        "feature-availability-v1",
        "label-policy-v1",
        "training-config-v1",
        "search-config-v1",
        "runtime-config-v1",
        "risk-config-v1",
        DeterminismPolicy::Deterministic { seed: 42 },
        "hardware-profile-1",
        DeviceAssignment {
            backend: BackendKind::CpuReference,
            device: "cpu".to_string(),
            device_ids: Vec::new(),
        },
        BackendKind::CpuReference,
        RuntimeMode::Degraded,
        Some(RuntimeDegradedReason::new(
            "cuda_unavailable",
            "CUDA was requested but unavailable",
        )),
        "abc1234",
    )
    .expect("degraded provenance is structurally valid");

    let report = provenance.runtime_safety_report();

    assert!(!report.live_safe);
    assert_eq!(
        report
            .degraded_reason
            .as_ref()
            .map(|reason| reason.code.as_str()),
        Some("cuda_unavailable")
    );
    assert!(report.has_issue(RuntimeSafetyIssue::NonCanonicalRuntimeMode));
    assert!(report.has_issue(RuntimeSafetyIssue::DegradedBackend));
}

#[test]
fn runtime_safety_report_exposes_backend_assignment_mismatch() {
    let mut provenance = canonical_live_provenance();
    provenance.backend_kind = BackendKind::CpuReference;

    let report = provenance.runtime_safety_report();

    assert!(!report.live_safe);
    assert!(!report.backend_assignment_matches);
    assert!(report.has_issue(RuntimeSafetyIssue::BackendAssignmentMismatch));
    assert!(report.has_issue(RuntimeSafetyIssue::DegradedBackend));
}

#[test]
fn live_accepts_canonical_live_ready_artifact() {
    let envelope = ArtifactEnvelope::new(canonical_live_provenance(), ())
        .expect("fixture envelope should be valid");

    envelope
        .require_live_ready()
        .expect("canonical live-ready artifact should pass live gate");
}

fn canonical_live_execution_contract() -> LiveExecutionContract {
    LiveExecutionContract::new(
        "feature-schema-v1",
        "timestamp-policy-v1",
        "feature-availability-v1",
        "symbols-eurusd",
        "runtime-config-v1",
        "risk-config-v1",
    )
}

#[test]
fn live_contract_accepts_matching_live_ready_artifact() {
    let envelope = LiveReadyStrategyArtifact::new(canonical_live_provenance(), "strategy")
        .expect("live-ready fixture should satisfy typed contract");
    let contract =
        canonical_live_execution_contract().with_required_backend(BackendKind::NativeCpu);

    envelope
        .require_live_execution_contract(&contract)
        .expect("matching live contract should pass");
}

#[test]
fn live_contract_rejects_checkpoint_and_diagnostic_artifacts() {
    let contract = canonical_live_execution_contract();
    let checkpoint =
        ArtifactEnvelope::new(canonical_provenance(ArtifactKind::SearchCheckpoint), ())
            .expect("checkpoint is structurally valid");
    let checkpoint_error = checkpoint
        .require_live_execution_contract(&contract)
        .expect_err("search checkpoints must not pass the live gate");
    assert_eq!(
        checkpoint_error,
        ArtifactContractError::LiveRejectedArtifactKind(ArtifactKind::SearchCheckpoint)
    );

    let diagnostic = ArtifactProvenance::new(
        ArtifactKind::LiveReadyStrategy,
        "feature-schema-v1",
        "dataset-a",
        "symbols-eurusd",
        "m1-m5",
        "timestamp-policy-v1",
        "feature-availability-v1",
        "label-policy-v1",
        "training-config-v1",
        "search-config-v1",
        "runtime-config-v1",
        "risk-config-v1",
        DeterminismPolicy::Deterministic { seed: 42 },
        "hardware-profile-1",
        DeviceAssignment {
            backend: BackendKind::NativeCpu,
            device: "cpu".to_string(),
            device_ids: Vec::new(),
        },
        BackendKind::NativeCpu,
        RuntimeMode::DiagnosticOnly,
        Some(RuntimeDegradedReason::new(
            "diagnostic_only",
            "artifact was produced by a diagnostic run",
        )),
        "abc1234",
    )
    .expect("diagnostic artifact is structurally valid");
    let diagnostic = ArtifactEnvelope::new(diagnostic, ()).expect("diagnostic envelope");
    let diagnostic_error = diagnostic
        .require_live_execution_contract(&contract)
        .expect_err("diagnostic artifacts must not pass live contract");
    assert_eq!(
        diagnostic_error,
        ArtifactContractError::LiveRejectedRuntimeMode {
            mode: RuntimeMode::DiagnosticOnly,
            backend: BackendKind::NativeCpu,
        }
    );
}

#[test]
fn live_contract_rejects_feature_timestamp_runtime_and_risk_mismatches() {
    let envelope = ArtifactEnvelope::new(canonical_live_provenance(), ())
        .expect("fixture envelope should be valid");

    let feature_error = LiveExecutionContract::new(
        "feature-schema-v2",
        "timestamp-policy-v1",
        "feature-availability-v1",
        "symbols-eurusd",
        "runtime-config-v1",
        "risk-config-v1",
    )
    .validate_provenance(&envelope.provenance)
    .expect_err("feature schema mismatch must fail");
    assert_eq!(
        feature_error,
        ArtifactContractError::LiveRejectedMismatch {
            field: "feature_schema_hash",
            actual: "feature-schema-v1".to_string(),
            expected: "feature-schema-v2".to_string(),
        }
    );

    let timestamp_error = LiveExecutionContract::new(
        "feature-schema-v1",
        "timestamp-policy-v2",
        "feature-availability-v1",
        "symbols-eurusd",
        "runtime-config-v1",
        "risk-config-v1",
    )
    .validate_provenance(&envelope.provenance)
    .expect_err("timestamp policy mismatch must fail");
    assert!(matches!(
        timestamp_error,
        ArtifactContractError::LiveRejectedMismatch {
            field: "timestamp_policy_hash",
            ..
        }
    ));

    let risk_error = LiveExecutionContract::new(
        "feature-schema-v1",
        "timestamp-policy-v1",
        "feature-availability-v1",
        "symbols-eurusd",
        "runtime-config-v2",
        "risk-config-v2",
    )
    .validate_provenance(&envelope.provenance)
    .expect_err("runtime/cost mismatch must fail before risk mismatch");
    assert!(matches!(
        risk_error,
        ArtifactContractError::LiveRejectedMismatch {
            field: "runtime_config_hash",
            ..
        }
    ));
}

#[test]
fn live_contract_rejects_stale_artifact_and_backend_mismatch() {
    let mut provenance = canonical_live_provenance();
    let now = Utc::now();
    provenance.created_at = now - chrono::Duration::seconds(120);
    let contract = canonical_live_execution_contract().with_max_artifact_age_seconds(60);

    let stale_error = contract
        .validate_provenance_at(&provenance, now)
        .expect_err("stale live artifact must fail");
    assert_eq!(
        stale_error,
        ArtifactContractError::LiveRejectedStaleArtifact {
            age_seconds: 120,
            max_age_seconds: 60,
        }
    );

    let backend_error = canonical_live_execution_contract()
        .with_required_backend(BackendKind::NativeCuda)
        .validate_provenance(&canonical_live_provenance())
        .expect_err("backend mismatch must fail");
    assert_eq!(
        backend_error,
        ArtifactContractError::LiveRejectedMismatch {
            field: "backend_kind",
            actual: "NativeCpu".to_string(),
            expected: "NativeCuda".to_string(),
        }
    );
}

fn temporal_contract_fixture() -> TemporalFeatureContract {
    TemporalFeatureContract::strict_live(
        "UTC",
        "alignment-policy-v1",
        "label-policy-v1",
        "walk-forward-policy-v1",
        "live-readiness-policy-v1",
    )
    .expect("strict temporal contract should be valid")
}

#[test]
fn temporal_contract_hashes_canonical_timestamp_and_feature_policy() {
    let contract = temporal_contract_fixture();

    assert_eq!(
        contract.timestamp_policy,
        TimestampPolicy::new(
            TimestampUnit::Milliseconds,
            CandleTimestampPolicy::OpenTime,
            "UTC",
        )
    );
    assert_eq!(
        contract.feature_availability_policy.multi_timeframe,
        MultiTimeframeAvailabilityPolicy::ClosedHigherTimeframeOnly
    );
    assert!(!contract.feature_availability_policy.allow_lookahead);
    assert!(contract.timestamp_policy_hash().starts_with("fnv64:"));
    assert!(
        contract
            .feature_availability_policy_hash()
            .starts_with("fnv64:")
    );
    assert_eq!(
        contract.timestamp_policy_hash(),
        temporal_contract_fixture().timestamp_policy_hash()
    );
}

#[test]
fn temporal_contract_rejects_lookahead_and_partial_mtf() {
    let lookahead_error = TemporalFeatureContract::new(
        TimestampPolicy::new(
            TimestampUnit::Milliseconds,
            CandleTimestampPolicy::OpenTime,
            "UTC",
        ),
        FeatureAvailabilityPolicy {
            multi_timeframe: MultiTimeframeAvailabilityPolicy::ClosedHigherTimeframeOnly,
            embargo_bars: 0,
            allow_lookahead: true,
            alignment_policy_hash: "alignment-policy-v1".to_string(),
        },
        "label-policy-v1",
        "walk-forward-policy-v1",
        "live-readiness-policy-v1",
    )
    .expect_err("lookahead must be rejected");

    assert!(matches!(
        lookahead_error,
        ArtifactContractError::TemporalPolicyViolation {
            field: "feature_availability_policy.allow_lookahead",
            ..
        }
    ));

    let partial_mtf_error = TemporalFeatureContract::new(
        TimestampPolicy::new(
            TimestampUnit::Milliseconds,
            CandleTimestampPolicy::OpenTime,
            "UTC",
        ),
        FeatureAvailabilityPolicy {
            multi_timeframe: MultiTimeframeAvailabilityPolicy::CurrentPartialAllowed,
            embargo_bars: 0,
            allow_lookahead: false,
            alignment_policy_hash: "alignment-policy-v1".to_string(),
        },
        "label-policy-v1",
        "walk-forward-policy-v1",
        "live-readiness-policy-v1",
    )
    .expect_err("partial higher-timeframe features must be rejected");

    assert!(matches!(
        partial_mtf_error,
        ArtifactContractError::TemporalPolicyViolation {
            field: "feature_availability_policy.multi_timeframe",
            ..
        }
    ));
}

#[test]
fn temporal_contract_validates_artifact_provenance_hashes() {
    let contract = temporal_contract_fixture();
    let mut provenance = canonical_live_provenance();
    provenance.timestamp_policy_hash = contract.timestamp_policy_hash();
    provenance.feature_availability_policy_hash = contract.feature_availability_policy_hash();
    provenance.label_policy_hash = contract.label_policy_hash.clone();

    contract
        .validate_provenance(&provenance)
        .expect("matching temporal provenance should be accepted");

    provenance.timestamp_policy_hash = "other-timestamp-policy".to_string();
    let error = contract
        .validate_provenance(&provenance)
        .expect_err("changed timestamp policy must be rejected");

    assert!(matches!(
        error,
        ArtifactContractError::TemporalPolicyMismatch {
            field: "timestamp_policy_hash",
            ..
        }
    ));
}

#[test]
fn temporal_scope_hashes_validate_contract_drift() {
    let contract = temporal_contract_fixture();
    let mut hashes = TemporalScopeHashes::from_contract(&contract);

    hashes
        .validate_contract(&contract)
        .expect("matching temporal hashes should validate");

    hashes.timestamp_policy_hash = "different-timestamp-policy".to_string();
    let error = hashes
        .validate_contract(&contract)
        .expect_err("changed timestamp policy hash must fail");

    assert!(matches!(
        error,
        ArtifactContractError::TemporalPolicyMismatch {
            field: "timestamp_policy_hash",
            ..
        }
    ));
}

#[test]
fn live_validation_evidence_default_is_neutral_pass_only_for_walkforward_and_cpcv() {
    let evidence = LiveValidationEvidence::default();
    assert!(!evidence.walkforward_passed);
    assert!(!evidence.cpcv_passed);
    assert_eq!(evidence.forward_test_passed, None);
    assert_eq!(evidence.prop_firm_passed, None);
    assert!(evidence.live_sim_runtime_model_hash.is_none());

    let passed = LiveValidationEvidence::passed_all();
    assert!(passed.walkforward_passed);
    assert!(passed.cpcv_passed);
    assert_eq!(passed.forward_test_passed, Some(true));
    assert_eq!(passed.prop_firm_passed, Some(true));
}

#[test]
fn live_contract_validate_evidence_accepts_when_no_gates_required() {
    let contract = canonical_live_execution_contract();
    let evidence = LiveValidationEvidence::default();
    contract
        .validate_evidence(&evidence)
        .expect("default contract has no required gates");
}

#[test]
fn live_contract_rejects_failed_walkforward_and_cpcv_gates() {
    let contract = canonical_live_execution_contract()
        .with_required_walkforward_pass()
        .with_required_cpcv_pass();
    let mut evidence = LiveValidationEvidence::passed_all();
    evidence.walkforward_passed = false;
    let err = contract
        .validate_evidence(&evidence)
        .expect_err("failed walkforward gate must reject");
    assert_eq!(
        err,
        ArtifactContractError::LiveRejectedFailedEvidenceGate {
            gate: "walkforward",
        }
    );

    let mut evidence = LiveValidationEvidence::passed_all();
    evidence.cpcv_passed = false;
    let err = contract
        .validate_evidence(&evidence)
        .expect_err("failed cpcv gate must reject");
    assert_eq!(
        err,
        ArtifactContractError::LiveRejectedFailedEvidenceGate { gate: "cpcv" }
    );
}

#[test]
fn live_contract_rejects_missing_forward_test_evidence_when_required() {
    let contract = canonical_live_execution_contract().with_required_forward_test_pass();

    let mut evidence = LiveValidationEvidence::passed_all();
    evidence.forward_test_passed = None;
    let err = contract
        .validate_evidence(&evidence)
        .expect_err("missing forward-test evidence must reject");
    assert_eq!(
        err,
        ArtifactContractError::LiveRejectedMissingEvidence {
            gate: "forward_test",
        }
    );

    let mut evidence = LiveValidationEvidence::passed_all();
    evidence.forward_test_passed = Some(false);
    let err = contract
        .validate_evidence(&evidence)
        .expect_err("failed forward-test evidence must reject");
    assert_eq!(
        err,
        ArtifactContractError::LiveRejectedFailedEvidenceGate {
            gate: "forward_test",
        }
    );
}

#[test]
fn live_contract_rejects_missing_or_failed_prop_firm_evidence() {
    let contract = canonical_live_execution_contract().with_required_prop_firm_pass();

    let mut evidence = LiveValidationEvidence::passed_all();
    evidence.prop_firm_passed = None;
    let err = contract
        .validate_evidence(&evidence)
        .expect_err("missing prop-firm evidence must reject");
    assert_eq!(
        err,
        ArtifactContractError::LiveRejectedMissingEvidence { gate: "prop_firm" }
    );

    let mut evidence = LiveValidationEvidence::passed_all();
    evidence.prop_firm_passed = Some(false);
    let err = contract
        .validate_evidence(&evidence)
        .expect_err("failed prop-firm evidence must reject");
    assert_eq!(
        err,
        ArtifactContractError::LiveRejectedFailedEvidenceGate { gate: "prop_firm" }
    );
}

#[test]
fn live_contract_rejects_live_sim_runtime_hash_mismatch() {
    let contract = canonical_live_execution_contract()
        .with_required_live_sim_runtime_model_hash("runtime-model-v1");

    let mut evidence = LiveValidationEvidence::passed_all();
    evidence.live_sim_runtime_model_hash = None;
    let err = contract
        .validate_evidence(&evidence)
        .expect_err("missing live-sim runtime hash must reject");
    assert_eq!(
        err,
        ArtifactContractError::LiveRejectedMissingEvidence {
            gate: "live_sim_runtime_model",
        }
    );

    let mut evidence = LiveValidationEvidence::passed_all();
    evidence.live_sim_runtime_model_hash = Some("runtime-model-v2".to_string());
    let err = contract
        .validate_evidence(&evidence)
        .expect_err("mismatched live-sim runtime hash must reject");
    assert!(matches!(
        err,
        ArtifactContractError::LiveRejectedMismatch {
            field: "live_sim_runtime_model_hash",
            ..
        }
    ));

    let mut evidence = LiveValidationEvidence::passed_all();
    evidence.live_sim_runtime_model_hash = Some("runtime-model-v1".to_string());
    contract
        .validate_evidence(&evidence)
        .expect("matching live-sim runtime hash must accept");
}

fn complete_validation_evidence() -> ValidationEvidenceManifest {
    ValidationEvidenceManifest::new(
        "canonical-backtest-fnv",
        "walkforward-fnv",
        "forward-test-fnv",
        "live-execution-sim-fnv",
        "prop-firm-risk-fnv",
    )
    .expect("fixture validation evidence should be complete")
}

#[test]
fn validation_evidence_manifest_requires_all_promotion_artifacts() {
    let error = ValidationEvidenceManifest::new(
        "canonical-backtest-fnv",
        "walkforward-fnv",
        "",
        "live-execution-sim-fnv",
        "prop-firm-risk-fnv",
    )
    .expect_err("missing forward-test evidence must fail");

    assert_eq!(
        error,
        ArtifactContractError::MissingValidationEvidence("forward_test_validation_hash")
    );
}

#[test]
fn validation_evidence_manifest_reports_structured_evidence_status() {
    let mut evidence = complete_validation_evidence();
    evidence.prop_firm_risk_validation_hash.clear();

    assert_eq!(
        evidence.missing_kinds(),
        vec![ValidationEvidenceKind::PropFirmRisk]
    );

    let checks = evidence.evidence_checks();
    assert_eq!(checks.len(), ValidationEvidenceKind::ALL.len());
    assert!(
        checks
            .iter()
            .any(|check| check.kind == ValidationEvidenceKind::PropFirmRisk && !check.present)
    );
    assert_eq!(
        evidence.hash_for(ValidationEvidenceKind::ForwardTest),
        Some("forward-test-fnv")
    );
}

#[test]
fn live_promotion_gate_accepts_canonical_deterministic_artifact_with_complete_evidence() {
    let envelope = ArtifactEnvelope::new(canonical_live_provenance(), ())
        .expect("fixture envelope should be structurally valid");
    let gate =
        LivePromotionGate::new(canonical_live_execution_contract()).require_deterministic(true);

    gate.validate(&envelope, &complete_validation_evidence())
        .expect("complete deterministic live artifact should pass promotion gate");
}

#[test]
fn crate_root_exports_live_contract_types() {
    let _kind = crate::ValidationEvidenceKind::PropFirmRisk;
    let _issue = crate::RuntimeSafetyIssue::DegradedBackend;
    let _status = crate::PromotionReadinessStatus::Passed;
    let _evidence = crate::LiveValidationEvidence::default();
    let gate: crate::LivePromotionGate =
        LivePromotionGate::new(canonical_live_execution_contract());

    assert!(gate.require_deterministic);
}

#[test]
fn live_promotion_gate_rejects_stale_artifact_with_injected_clock() {
    let now = Utc::now();
    let mut provenance = canonical_live_provenance();
    provenance.created_at = now - Duration::seconds(61);
    let envelope =
        ArtifactEnvelope::new(provenance, ()).expect("stale artifact is structurally valid");
    let gate = LivePromotionGate::new(
        canonical_live_execution_contract().with_max_artifact_age_seconds(60),
    );

    let error = gate
        .validate_at(&envelope, &complete_validation_evidence(), now)
        .expect_err("promotion gate must reject stale artifacts at the supplied clock");

    assert_eq!(
        error,
        ArtifactContractError::LiveRejectedStaleArtifact {
            age_seconds: 61,
            max_age_seconds: 60,
        }
    );
}

#[test]
fn live_promotion_gate_rejects_best_effort_when_deterministic_required() {
    let mut provenance = canonical_live_provenance();
    provenance.determinism_policy = DeterminismPolicy::BestEffort;
    let envelope =
        ArtifactEnvelope::new(provenance, ()).expect("best-effort artifact is structurally valid");
    let gate =
        LivePromotionGate::new(canonical_live_execution_contract()).require_deterministic(true);

    let error = gate
        .validate(&envelope, &complete_validation_evidence())
        .expect_err("deterministic promotion gate must reject best-effort artifacts");

    assert_eq!(
        error,
        ArtifactContractError::PromotionRejectedDeterminism {
            actual: DeterminismPolicy::BestEffort,
        }
    );
}

#[test]
fn live_promotion_gate_rejects_degraded_runtime_even_with_complete_evidence() {
    let provenance = ArtifactProvenance::new(
        ArtifactKind::LiveReadyStrategy,
        "feature-schema-v1",
        "dataset-a",
        "symbols-eurusd",
        "m1-m5",
        "timestamp-policy-v1",
        "feature-availability-v1",
        "label-policy-v1",
        "training-config-v1",
        "search-config-v1",
        "runtime-config-v1",
        "risk-config-v1",
        DeterminismPolicy::Deterministic { seed: 42 },
        "hardware-profile-1",
        DeviceAssignment {
            backend: BackendKind::CpuReference,
            device: "cpu".to_string(),
            device_ids: Vec::new(),
        },
        BackendKind::CpuReference,
        RuntimeMode::Degraded,
        Some(RuntimeDegradedReason::new(
            "cuda_unavailable",
            "CUDA was requested but unavailable",
        )),
        "abc1234",
    )
    .expect("degraded provenance is structurally valid");
    let envelope = ArtifactEnvelope::new(provenance, ()).expect("degraded envelope");
    let gate =
        LivePromotionGate::new(canonical_live_execution_contract()).require_deterministic(true);

    let error = gate
        .validate(&envelope, &complete_validation_evidence())
        .expect_err("degraded runtime must not pass promotion gate");

    assert_eq!(
        error,
        ArtifactContractError::LiveRejectedRuntimeMode {
            mode: RuntimeMode::Degraded,
            backend: BackendKind::CpuReference,
        }
    );
}

#[test]
fn live_promotion_readiness_report_collects_operator_reasons_without_failing() {
    let mut provenance = canonical_live_provenance();
    provenance.determinism_policy = DeterminismPolicy::BestEffort;
    let envelope =
        ArtifactEnvelope::new(provenance, ()).expect("best-effort artifact is structurally valid");
    let gate =
        LivePromotionGate::new(canonical_live_execution_contract()).require_deterministic(true);

    let report = gate.readiness_report(&envelope, None);

    assert!(!report.ready);
    assert!(!report.validation_evidence_complete);
    assert!(!report.determinism_requirement_passed);
    assert!(
        report
            .rejection_reasons
            .iter()
            .any(|reason| reason.contains("validation evidence"))
    );
    assert!(
        report
            .rejection_reasons
            .iter()
            .any(|reason| reason.contains("deterministic"))
    );
    assert!(report.checks.iter().any(|check| check.kind
        == PromotionReadinessCheckKind::ValidationEvidence
        && check.status == PromotionReadinessStatus::Failed));
    assert!(report.checks.iter().any(|check| check.kind
        == PromotionReadinessCheckKind::DeterminismRequirement
        && check.status == PromotionReadinessStatus::Failed));
    assert_eq!(
        report.evidence_checks.len(),
        ValidationEvidenceKind::ALL.len()
    );
    assert!(report.evidence_checks.iter().all(|check| !check.present));
}

#[test]
fn live_promotion_readiness_report_includes_runtime_safety_snapshot() {
    let provenance = ArtifactProvenance::new(
        ArtifactKind::LiveReadyStrategy,
        "feature-schema-v1",
        "dataset-a",
        "symbols-eurusd",
        "m1-m5",
        "timestamp-policy-v1",
        "feature-availability-v1",
        "label-policy-v1",
        "training-config-v1",
        "search-config-v1",
        "runtime-config-v1",
        "risk-config-v1",
        DeterminismPolicy::Deterministic { seed: 42 },
        "hardware-profile-1",
        DeviceAssignment {
            backend: BackendKind::CpuReference,
            device: "cpu".to_string(),
            device_ids: Vec::new(),
        },
        BackendKind::CpuReference,
        RuntimeMode::Degraded,
        Some(RuntimeDegradedReason::new(
            "cuda_unavailable",
            "CUDA was requested but unavailable",
        )),
        "abc1234",
    )
    .expect("degraded provenance is structurally valid");
    let envelope = ArtifactEnvelope::new(provenance, ()).expect("degraded envelope");
    let gate = LivePromotionGate::new(canonical_live_execution_contract());

    let report = gate.readiness_report(&envelope, Some(&complete_validation_evidence()));

    assert!(!report.ready);
    assert!(!report.runtime_safety.live_safe);
    assert!(
        report
            .runtime_safety
            .has_issue(RuntimeSafetyIssue::DegradedBackend)
    );
    assert!(report.checks.iter().any(|check| check.kind
        == PromotionReadinessCheckKind::RuntimeSafety
        && check.status == PromotionReadinessStatus::Failed));
}
