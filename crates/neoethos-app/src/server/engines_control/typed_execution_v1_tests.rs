use super::*;

fn successful_discovery_with_handoff() -> JobSnapshot {
    let mut snapshot = JobSnapshot::new(JobKind::Discovery);
    snapshot.state = JobState::Succeeded;
    snapshot.report.highlights = vec![("training_handoff".to_owned(), "a".repeat(64))];
    snapshot
}

fn follow_on_series_fixture(generation: char) -> CanonicalDatasetSeriesReceiptV1 {
    let identity = neoethos_data::CanonicalDatasetIdentity::external(
        "typed-lifecycle-fixture",
        "EURUSD",
        CanonicalTimeframe::M5,
        neoethos_data::BarTimestampConvention::BarOpen,
    )
    .unwrap();
    let selected = SelectedDatasetGenerationV1::new(
        identity,
        format!("g1-{}.vortex", generation.to_string().repeat(64)),
        "f".repeat(64),
    )
    .unwrap();
    CanonicalDatasetSeriesReceiptV1::new(selected.clone(), vec![selected]).unwrap()
}

#[test]
fn automatic_training_binds_the_complete_exact_discovery_series() {
    let original = follow_on_series_fixture('1');
    let different_generation = follow_on_series_fixture('2');
    validate_follow_on_series_v1(Some(&original), &original).unwrap();
    assert!(validate_follow_on_series_v1(Some(&original), &different_generation).is_err());
    // An explicit, standalone candidate request has no prior Discovery in this
    // process; its complete durable handoff still validates in the real loader.
    validate_follow_on_series_v1(None, &different_generation).unwrap();
}

#[tokio::test]
async fn successful_discovery_continues_only_its_handoff_under_the_same_lease() {
    let _isolation = crate::PROCESS_EXECUTION_TEST_LOCK.lock().await;
    let mut lease =
        try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Discovery).unwrap();
    let token = lease.token();
    let cancel = CancellationFlag::new();
    let (snapshot_tx, snapshots) = watch::channel(TypedLegacyExecutionSnapshotV1::new(
        token,
        ProcessExecutionKindV1::Discovery,
        queued_snapshot_v1(JobKind::Discovery),
    ));
    let terminal = continue_discovery_training_v1(
        successful_discovery_with_handoff(),
        &cancel,
        &mut lease,
        &snapshot_tx,
        |intent| async move {
            assert_eq!(
                intent.selection,
                TypedTrainingSelectionPolicyV1::DiscoveryHandoff {
                    identity_sha256: "a".repeat(64),
                }
            );
            assert_eq!(
                neoethos_search::active_process_execution_kind_v1(),
                Some(ProcessExecutionKindV1::Training)
            );
            assert!(
                try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Migration).is_err()
            );
            let snapshot = snapshots.borrow().clone();
            assert_eq!(snapshot.lease_token, token);
            assert_eq!(snapshot.lease_kind, ProcessExecutionKindV1::Training);
            assert_eq!(snapshot.job_snapshot.kind, JobKind::Training);
            let mut trained = JobSnapshot::new(JobKind::Training);
            trained.state = JobState::Succeeded;
            Ok(trained)
        },
    )
    .await
    .unwrap();
    assert_eq!(lease.token(), token);
    assert_eq!(lease.kind(), ProcessExecutionKindV1::Training);
    assert!(matches!(
        terminal_from_snapshot_v1(terminal, token, JobKind::Training),
        TypedLegacyExecutionTerminalV1::Succeeded {
            completed_kind: JobKind::Training,
            ..
        }
    ));
}

#[tokio::test]
async fn multiple_batch_handoffs_require_selection_without_losing_completed_research() {
    let _isolation = crate::PROCESS_EXECUTION_TEST_LOCK.lock().await;
    for retain_last_highlight in [false, true] {
        let mut lease =
            try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Discovery).unwrap();
        let token = lease.token();
        let mut completed = successful_discovery_with_handoff();
        if !retain_last_highlight {
            completed.report.highlights.clear();
        }
        completed.report.counters = vec![
            ("working_set_completed_batches".to_owned(), 3),
            ("working_set_saved_results".to_owned(), 3),
            ("working_set_training_handoffs".to_owned(), 2),
        ];
        let (tx, _rx) = watch::channel(TypedLegacyExecutionSnapshotV1::new(
            token,
            ProcessExecutionKindV1::Discovery,
            completed.clone(),
        ));
        let terminal = continue_discovery_training_v1(
            completed.clone(),
            &CancellationFlag::new(),
            &mut lease,
            &tx,
            |_| async { panic!("multiple published batches must not select the last result") },
        )
        .await
        .unwrap_err();
        assert_eq!(terminal.kind, JobKind::Discovery);
        assert_eq!(terminal.state, JobState::Failed);
        assert_eq!(terminal.report.counters, completed.report.counters);
        assert_eq!(terminal.report.highlights, completed.report.highlights);
        assert!(
            terminal
                .report
                .summary
                .contains("published 2 batch handoffs")
        );
        assert!(
            terminal
                .report
                .summary
                .contains("select a published result in Training")
        );
        assert_eq!(lease.token(), token);
        assert_eq!(lease.kind(), ProcessExecutionKindV1::Discovery);
    }
}

#[tokio::test]
async fn failed_cancelled_or_unbound_discovery_never_enters_training() {
    let _isolation = crate::PROCESS_EXECUTION_TEST_LOCK.lock().await;
    let mut cases = Vec::new();
    for state in [
        JobState::Queued,
        JobState::Running,
        JobState::Degraded,
        JobState::Failed,
        JobState::Cancelled,
    ] {
        let mut snapshot = successful_discovery_with_handoff();
        snapshot.state = state;
        cases.push(snapshot);
    }
    let mut missing = successful_discovery_with_handoff();
    missing.report.highlights.clear();
    cases.push(missing);
    let mut duplicate = successful_discovery_with_handoff();
    duplicate
        .report
        .highlights
        .push(("training_handoff".to_owned(), "a".repeat(64)));
    cases.push(duplicate);
    for invalid in ["../escape".to_owned(), "A".repeat(64), "a".repeat(63)] {
        let mut malformed = successful_discovery_with_handoff();
        malformed.report.highlights[0].1 = invalid;
        cases.push(malformed);
    }
    let mut wrong_phase = successful_discovery_with_handoff();
    wrong_phase.kind = JobKind::Training;
    cases.push(wrong_phase);
    for snapshot in cases {
        let mut lease =
            try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Discovery).unwrap();
        let (tx, _rx) = watch::channel(TypedLegacyExecutionSnapshotV1::new(
            lease.token(),
            ProcessExecutionKindV1::Discovery,
            snapshot.clone(),
        ));
        let result = continue_discovery_training_v1(
            snapshot,
            &CancellationFlag::new(),
            &mut lease,
            &tx,
            |_| async { panic!("rejected Discovery must not invoke a training starter") },
        )
        .await;
        assert!(result.is_err());
        assert_eq!(lease.kind(), ProcessExecutionKindV1::Discovery);
    }
}

#[tokio::test]
async fn automatic_training_rejects_symbol_only_fallback_before_any_io() {
    let error = prepare_training_request_v1(
        &AppApiState::new(),
        TypedTrainingExecutionIntentV1 {
            selection: TypedTrainingSelectionPolicyV1::Exact {
                symbol: "EURUSD".to_owned(),
                base_timeframe: CanonicalTimeframe::M5,
            },
        },
        &CancellationFlag::new(),
        Some(follow_on_series_fixture('1')),
    )
    .await
    .unwrap_err();
    assert!(
        error
            .detail()
            .contains("cannot use a symbol-only selection")
    );
}

#[tokio::test]
async fn training_event_drain_propagates_the_pipeline_cancellation_to_the_child() {
    let _isolation = crate::PROCESS_EXECUTION_TEST_LOCK.lock().await;
    let lease = try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Training).unwrap();
    let root_cancel = CancellationFlag::new();
    let child_cancel = CancellationFlag::new();
    root_cancel.request();
    let (snapshot_tx, snapshots) = watch::channel(TypedLegacyExecutionSnapshotV1::new(
        lease.token(),
        ProcessExecutionKindV1::Training,
        queued_snapshot_v1(JobKind::Training),
    ));
    let (events_tx, mut events_rx) = mpsc::channel(1);
    events_tx
        .send(ServiceEvent::TrainingUpdated(cancelled_snapshot_v1(
            JobKind::Training,
        )))
        .await
        .unwrap();
    let result = drain_job_events_v1(
        &AppApiState::new(),
        JobKind::Training,
        &root_cancel,
        &child_cancel,
        &mut events_rx,
        &snapshot_tx,
        lease.token(),
        ProcessExecutionKindV1::Training,
    )
    .await
    .unwrap_err();
    assert!(child_cancel.is_requested());
    assert_eq!(result.state, JobState::Cancelled);
    assert_eq!(snapshots.borrow().lease_token, lease.token());
    assert_eq!(snapshots.borrow().job_snapshot.kind, JobKind::Training);
}

#[tokio::test]
async fn cancellation_between_discovery_and_training_keeps_the_discovery_phase() {
    let _isolation = crate::PROCESS_EXECUTION_TEST_LOCK.lock().await;
    let mut lease =
        try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Discovery).unwrap();
    let cancel = CancellationFlag::new();
    cancel.request();
    let (tx, _rx) = watch::channel(TypedLegacyExecutionSnapshotV1::new(
        lease.token(),
        ProcessExecutionKindV1::Discovery,
        queued_snapshot_v1(JobKind::Discovery),
    ));
    let terminal = continue_discovery_training_v1(
        successful_discovery_with_handoff(),
        &cancel,
        &mut lease,
        &tx,
        |_| async { panic!("cancelled chain must not enter Training") },
    )
    .await
    .unwrap_err();
    assert_eq!(terminal.kind, JobKind::Discovery);
    assert_eq!(terminal.state, JobState::Cancelled);
    assert_eq!(lease.kind(), ProcessExecutionKindV1::Discovery);
}

#[tokio::test]
async fn cancellation_after_lease_transition_stops_training_before_settings_or_handoff_io() {
    let _isolation = crate::PROCESS_EXECUTION_TEST_LOCK.lock().await;
    let mut lease =
        try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Discovery).unwrap();
    let token = lease.token();
    let cancel = CancellationFlag::new();
    let (tx, _rx) = watch::channel(TypedLegacyExecutionSnapshotV1::new(
        token,
        ProcessExecutionKindV1::Discovery,
        queued_snapshot_v1(JobKind::Discovery),
    ));
    let state = AppApiState::new();
    state
        .install_engine(JobKind::Training, CancellationFlag::new(), token + 1)
        .await;
    let mut previous = JobSnapshot::new(JobKind::Training);
    previous.state = JobState::Succeeded;
    previous.report.counters = vec![("models_trained".to_owned(), 15)];
    state
        .update_engine_snapshot(JobKind::Training, &previous, token + 1)
        .await;
    let terminal = continue_discovery_training_v1(
        successful_discovery_with_handoff(),
        &cancel,
        &mut lease,
        &tx,
        |intent| {
            cancel.request();
            run_training_intent_v1(
                &state,
                intent,
                &cancel,
                &tx,
                token,
                None,
                Some(follow_on_series_fixture('1')),
            )
        },
    )
    .await
    .unwrap_err();
    assert_eq!(terminal.kind, JobKind::Training);
    assert_eq!(terminal.state, JobState::Cancelled);
    persist_terminal_state_v1(
        &state,
        &terminal_from_snapshot_v1(terminal.clone(), token, JobKind::Training),
        JobKind::Training,
        token,
    )
    .await;
    assert_eq!(
        state.engine_state(JobKind::Training).await,
        EngineRunState::Cancelled
    );
    assert!(state.engine_progress(JobKind::Training).await.2.is_empty());
    assert_eq!(lease.token(), token);
    assert_eq!(lease.kind(), ProcessExecutionKindV1::Training);
}

#[tokio::test]
async fn chained_training_failure_is_not_reported_as_successful_discovery() {
    let _isolation = crate::PROCESS_EXECUTION_TEST_LOCK.lock().await;
    let mut lease =
        try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Discovery).unwrap();
    let token = lease.token();
    let (tx, _rx) = watch::channel(TypedLegacyExecutionSnapshotV1::new(
        token,
        ProcessExecutionKindV1::Discovery,
        queued_snapshot_v1(JobKind::Discovery),
    ));
    let snapshot = continue_discovery_training_v1(
        successful_discovery_with_handoff(),
        &CancellationFlag::new(),
        &mut lease,
        &tx,
        |_| async {
            Err(failed_snapshot_v1(
                JobKind::Training,
                "candidate validation rejected",
            ))
        },
    )
    .await
    .unwrap_err();
    let terminal = terminal_from_snapshot_v1(snapshot, token, JobKind::Training);
    let (kind, snapshot) = terminal_kind_and_snapshot_v1(&terminal, JobKind::Discovery);
    assert_eq!(kind, JobKind::Training);
    assert_eq!(
        snapshot.unwrap().report.summary,
        "candidate validation rejected"
    );
    assert!(matches!(
        terminal,
        TypedLegacyExecutionTerminalV1::Failed { .. }
    ));
}

#[tokio::test]
async fn chained_training_panic_uses_the_latest_phase_and_joins_before_releasing_the_handle() {
    let _isolation = crate::PROCESS_EXECUTION_TEST_LOCK.lock().await;
    let mut lease =
        try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Discovery).unwrap();
    let token = lease.token();
    let cancel = CancellationFlag::new();
    let worker_cancel = cancel.clone();
    let (snapshot_tx, snapshots) = watch::channel(TypedLegacyExecutionSnapshotV1::new(
        token,
        ProcessExecutionKindV1::Discovery,
        queued_snapshot_v1(JobKind::Discovery),
    ));
    let (admission_tx, admission) = oneshot::channel();
    let (terminal_tx, terminal) = oneshot::channel();
    let worker = tokio::spawn(async move {
        let _admission_tx = admission_tx;
        let _terminal_tx = terminal_tx;
        let _ = continue_discovery_training_v1(
            successful_discovery_with_handoff(),
            &worker_cancel,
            &mut lease,
            &snapshot_tx,
            |_| async { panic!("synthetic chained Training panic") },
        )
        .await;
    });
    let handle = TypedLegacyExecutionJobHandleV1 {
        lease_token: token,
        initial_kind: JobKind::Discovery,
        cancel,
        snapshots,
        admission,
        terminal,
        worker,
    };
    let terminal = tokio::time::timeout(std::time::Duration::from_secs(5), handle.await_terminal())
        .await
        .unwrap();
    let TypedLegacyExecutionTerminalV1::WorkerPanicked {
        lease_token,
        job_kind,
        detail,
    } = terminal
    else {
        panic!("panicking continuation must return its joined panic");
    };
    assert_eq!(lease_token, token);
    assert_eq!(job_kind, JobKind::Training);
    assert!(detail.contains("synthetic chained Training panic"));
    let replacement =
        try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Migration).unwrap();
    assert_ne!(replacement.token(), token);
}

#[test]
fn configured_m5_discovery_selects_the_higher_ladder_without_changing_model_context() {
    let mut settings = Settings::default();
    settings.system.multi_resolution_timeframes = [
        "M1", "M3", "M5", "M15", "M30", "H1", "H12", "H4", "D1", "W1", "MN1",
    ]
    .map(str::to_owned)
    .to_vec();
    let model_context = settings.system.resolve_higher_timeframes("M5");
    assert!(model_context.iter().any(|tf| tf == "M1"));
    assert!(model_context.iter().any(|tf| tf == "M3"));
    let resolved = TypedHigherTimeframePolicyV1::Configured
        .resolve_for_discovery(&settings, CanonicalTimeframe::M5)
        .unwrap();
    assert_eq!(
        resolved,
        ["M15", "M30", "H1", "H12", "H4", "D1", "W1", "MN1"]
    );
    assert_eq!(
        settings.system.resolve_higher_timeframes("M5"),
        model_context
    );
}

#[test]
fn configured_discovery_ladder_tracks_the_effective_base_including_the_largest_tf() {
    let settings = Settings::default();
    assert_eq!(
        TypedHigherTimeframePolicyV1::Configured
            .resolve_for_discovery(&settings, CanonicalTimeframe::H1)
            .unwrap(),
        ["H4", "H12", "D1", "W1", "MN1"]
    );
    assert!(
        TypedHigherTimeframePolicyV1::Configured
            .resolve_for_discovery(&settings, CanonicalTimeframe::MN1)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn configured_discovery_honours_the_active_list_and_normalizes_labels() {
    let mut settings = Settings::default();
    settings.system.higher_timeframes = [" h4 ", "d1", "m1", " m5 ", " "]
        .map(str::to_owned)
        .to_vec();
    settings.system.multi_resolution_timeframes = vec!["unknown".to_owned()];
    settings.system.multi_resolution_enabled = false;
    assert_eq!(
        TypedHigherTimeframePolicyV1::Configured
            .resolve_for_discovery(&settings, CanonicalTimeframe::M5)
            .unwrap(),
        ["H4", "D1"]
    );
    settings.system.multi_resolution_enabled = true;
    settings.system.multi_resolution_timeframes.clear();
    assert_eq!(
        TypedHigherTimeframePolicyV1::Configured
            .resolve_for_discovery(&settings, CanonicalTimeframe::M5)
            .unwrap(),
        ["H4", "D1"]
    );
}

#[test]
fn configured_discovery_rejects_unknown_and_duplicate_active_timeframes() {
    let mut settings = Settings::default();
    for multi_resolution_enabled in [true, false] {
        settings.system.multi_resolution_enabled = multi_resolution_enabled;
        for (values, expected) in [
            (["H1", "invalid"], "invalid configured Discovery timeframe"),
            (["H1", " h1 "], "duplicate higher timeframe H1"),
        ] {
            settings.system.multi_resolution_timeframes = values.map(str::to_owned).to_vec();
            settings.system.higher_timeframes = settings.system.multi_resolution_timeframes.clone();
            let error = TypedHigherTimeframePolicyV1::Configured
                .resolve_for_discovery(&settings, CanonicalTimeframe::M5)
                .unwrap_err();
            assert!(matches!(
                error,
                TypedLegacyExecutionAdmissionErrorV1::BadRequest(_)
            ));
            assert!(error.detail().contains(expected), "{error}");
        }
    }
}

#[test]
fn exact_discovery_timeframes_are_not_filtered_or_replaced_by_configuration() {
    let settings = Settings::default();
    assert_eq!(
        TypedHigherTimeframePolicyV1::Exact(vec![CanonicalTimeframe::D1, CanonicalTimeframe::H1])
            .resolve_for_discovery(&settings, CanonicalTimeframe::M5)
            .unwrap(),
        ["D1", "H1"]
    );
    assert!(
        TypedHigherTimeframePolicyV1::Exact(Vec::new())
            .resolve_for_discovery(&settings, CanonicalTimeframe::M5)
            .unwrap()
            .is_empty()
    );
    for timeframe in [CanonicalTimeframe::M1, CanonicalTimeframe::M5] {
        let error = TypedHigherTimeframePolicyV1::Exact(vec![timeframe])
            .resolve_for_discovery(&settings, CanonicalTimeframe::M5)
            .unwrap_err();
        assert!(error.detail().contains("must be strictly above base M5"));
    }
    let error = TypedHigherTimeframePolicyV1::Exact(vec![CanonicalTimeframe::H1; 2])
        .resolve_for_discovery(&settings, CanonicalTimeframe::M5)
        .unwrap_err();
    assert_eq!(error.detail(), "duplicate higher timeframe H1");
}

#[tokio::test]
async fn discovery_worker_progress_reaches_desktop_status_without_a_stale_summary_or_fake_zero() {
    let _isolation = crate::PROCESS_EXECUTION_TEST_LOCK.lock().await;
    let state = AppApiState::new();
    let lease = try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Discovery).unwrap();
    let token = lease.token();
    let cancel = CancellationFlag::new();
    state
        .install_engine(JobKind::Discovery, cancel.clone(), token)
        .await;
    let initial = JobSnapshot::new(JobKind::Discovery);
    let (snapshots_tx, mut snapshots_rx) = watch::channel(TypedLegacyExecutionSnapshotV1::new(
        token,
        ProcessExecutionKindV1::Discovery,
        initial.clone(),
    ));
    let (events_tx, mut events_rx) = mpsc::channel(2);
    let worker_state = state.clone();
    let worker = tokio::spawn(async move {
        let _lease = lease;
        drain_job_events_v1(
            &worker_state,
            JobKind::Discovery,
            &cancel,
            &CancellationFlag::new(),
            &mut events_rx,
            &snapshots_tx,
            token,
            ProcessExecutionKindV1::Discovery,
        )
        .await
    });

    let mut snapshot = initial;
    snapshot.state = JobState::Running;
    snapshot.report.summary = "outdated preparation summary".to_owned();
    snapshot.progress.stage = "screening".to_owned();
    snapshot.progress.message = "Evaluating candidates; worker active".to_owned();
    snapshot.progress.percent = None;
    snapshot.report.counters = vec![("candidates_evaluated".to_owned(), 1_250)];
    events_tx
        .send(ServiceEvent::DiscoveryUpdated(snapshot.clone()))
        .await
        .unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(5), snapshots_rx.changed())
        .await
        .unwrap()
        .unwrap();
    let status = crate::server::system_status::engines(axum::extract::State(state.clone()))
        .await
        .unwrap()
        .0;
    assert_eq!(status.discovery, "Running");
    assert_eq!(status.discovery_summary, snapshot.progress.message);
    assert_eq!(status.discovery_stage, "screening");
    assert!(status.discovery_percent.is_none());
    assert_eq!(status.discovery_counters[0].value, 1_250);
    assert!(!status.discovery_start_available);

    snapshot.state = JobState::Succeeded;
    snapshot.report.summary = "Research artifact saved; no trading authority".to_owned();
    snapshot.report.counters = vec![
        ("candidates_evaluated".to_owned(), 1_300),
        ("walkforward_tested".to_owned(), 37),
    ];
    // The last live message must not replace the terminal report.
    events_tx
        .send(ServiceEvent::DiscoveryUpdated(snapshot.clone()))
        .await
        .unwrap();
    let terminal = tokio::time::timeout(std::time::Duration::from_secs(5), worker)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(terminal, snapshot);
    let status = crate::server::system_status::engines(axum::extract::State(state))
        .await
        .unwrap()
        .0;
    assert_eq!(status.discovery_summary, snapshot.report.summary);
    assert!(status.discovery_percent.is_none());
    assert_eq!(status.discovery_counters.len(), 2);
    assert_eq!(status.discovery_counters[0].value, 1_300);
    assert_eq!(status.discovery_counters[1].name, "walkforward_tested");
    assert_eq!(status.discovery_counters[1].value, 37);
    assert!(status.discovery_start_available);
    assert!(!status.historical_evaluation_available);
}

#[test]
fn generation_policy_distinguishes_exact_api_from_validation_floor() {
    let mut exact = DiscoveryConfig::default();
    exact.generations = 20;
    TypedDiscoveryOverridesV1::checked_new(
        None,
        Some(TypedDiscoveryGenerationOverrideV1::Exact(7)),
        None,
        None,
        None,
        None,
    )
    .expect("exact override")
    .apply(&mut exact);
    assert_eq!(exact.generations, 7);

    let mut floor = DiscoveryConfig::default();
    floor.generations = 20;
    TypedDiscoveryOverridesV1::checked_new(
        None,
        Some(TypedDiscoveryGenerationOverrideV1::Floor(30)),
        None,
        None,
        None,
        None,
    )
    .expect("floor override")
    .apply(&mut floor);
    assert_eq!(floor.generations, 30);
}

#[tokio::test]
async fn panicking_training_worker_preserves_token_and_updates_training_slot() {
    let _isolation = crate::PROCESS_EXECUTION_TEST_LOCK.lock().await;
    let lease = try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Training)
        .expect("training lease");
    let lease_token = lease.token();
    let cancel = CancellationFlag::new();
    let initial = TypedLegacyExecutionSnapshotV1::new(
        lease_token,
        ProcessExecutionKindV1::Training,
        queued_snapshot_v1(JobKind::Training),
    );
    let (snapshot_tx, snapshots) = watch::channel(initial);
    let (admission_tx, admission) = oneshot::channel();
    let (terminal_tx, terminal) = oneshot::channel();
    let worker = tokio::spawn(async move {
        let _lease = lease;
        let _snapshot_tx = snapshot_tx;
        let _admission_tx = admission_tx;
        let _terminal_tx = terminal_tx;
        panic!("synthetic training worker panic");
    });
    let handle = TypedLegacyExecutionJobHandleV1 {
        lease_token,
        initial_kind: JobKind::Training,
        cancel: cancel.clone(),
        snapshots,
        admission,
        terminal,
        worker,
    };
    let state = AppApiState::new();
    state
        .install_engine(JobKind::Training, cancel, lease_token)
        .await;
    detach_typed_legacy_execution_observer_v1(state.clone(), handle);

    for _ in 0..100 {
        if state.engine_state(JobKind::Training).await == EngineRunState::Failed {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert_eq!(
        state.engine_state(JobKind::Training).await,
        EngineRunState::Failed
    );
    let replacement = try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Migration)
        .expect("observer must await worker lease release");
    assert_ne!(replacement.token(), lease_token);
}

#[test]
fn cancelled_preparation_is_terminal_cancelled_not_failed() {
    let cancel = CancellationFlag::new();
    cancel.request();
    let snapshot =
        preparation_error_snapshot_v1(JobKind::Discovery, &cancel, "cancelled before Settings");
    assert_eq!(snapshot.state, JobState::Cancelled);
}

#[tokio::test(flavor = "current_thread")]
async fn strategy_only_uses_the_training_lease_and_cancels_before_handoff_io() {
    let _isolation = crate::PROCESS_EXECUTION_TEST_LOCK.lock().await;
    let handle = start_typed_training_execution_v1(
        AppApiState::new(),
        TypedTrainingExecutionIntentV1 {
            selection: TypedTrainingSelectionPolicyV1::StrategyResearchHandoff {
                identity_sha256: "a".repeat(64),
            },
        },
    )
    .expect("start the strategy-only typed worker");
    let token = handle.lease_token;
    // The worker has not been polled. This request must not read a user profile,
    // load a model, consume a final scope, or enter another execution lane.
    handle.cancel();
    assert_eq!(
        neoethos_search::active_process_execution_kind_v1(),
        Some(ProcessExecutionKindV1::Training)
    );
    assert!(try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Discovery).is_err());
    let terminal = tokio::time::timeout(std::time::Duration::from_secs(5), handle.await_terminal())
        .await
        .expect("cancelled strategy worker must join");
    let TypedLegacyExecutionTerminalV1::Cancelled {
        final_snapshot,
        lease_token,
    } = terminal
    else {
        panic!("strategy cancellation must not become success or an admission error");
    };
    assert_eq!(lease_token, token);
    assert_eq!(final_snapshot.kind, JobKind::Training);
    assert_eq!(final_snapshot.state, JobState::Cancelled);
    let replacement = try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Discovery)
        .expect("joined strategy worker must release the shared lease");
    assert_ne!(replacement.token(), token);
}

#[tokio::test(flavor = "current_thread")]
async fn cancelled_discovery_keeps_its_process_lease_until_the_owned_worker_returns() {
    let _isolation = crate::PROCESS_EXECUTION_TEST_LOCK.lock().await;
    let handle = start_typed_discovery_execution_v1(
        AppApiState::new(),
        TypedDiscoveryExecutionIntentV1 {
            symbol: "EURUSD".to_owned(),
            base_timeframe: CanonicalTimeframe::M1,
            higher_timeframes: TypedHigherTimeframePolicyV1::Exact(Vec::new()),
            overrides: TypedDiscoveryOverridesV1::default(),
            settings_gate: TypedDiscoverySettingsGateV1::None,
            dataset_policy: TypedDiscoveryDatasetPolicyV1::Current,
            training_after_success: false,
        },
    )
    .expect("start the actual typed worker");
    let lease_token = handle.lease_token;
    // No await has occurred: cancellation precedes Settings/dataset/broker IO.
    assert!(!handle.is_finished());
    handle.cancel();
    assert!(
        try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Migration).is_err(),
        "the spawned owner must retain the lease even before its first poll"
    );
    let terminal = tokio::time::timeout(std::time::Duration::from_secs(5), handle.await_terminal())
        .await
        .expect("cancelled worker must terminate");
    let TypedLegacyExecutionTerminalV1::Cancelled {
        final_snapshot,
        lease_token: returned_token,
    } = terminal
    else {
        panic!("pre-cancelled Discovery did not return a cancelled terminal");
    };
    assert_eq!(returned_token, lease_token);
    assert_eq!(final_snapshot.state, JobState::Cancelled);
    let replacement = try_acquire_process_execution_lease_v1(ProcessExecutionKindV1::Migration)
        .expect("await_terminal must await the worker and release the lease");
    assert_ne!(replacement.token(), lease_token);
}
