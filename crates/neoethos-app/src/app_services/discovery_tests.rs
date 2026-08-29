use super::*;

use crate::app_services::{
    ServiceEvent,
    jobs::{JobKind, JobSnapshot, JobState},
};
use neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1;
use neoethos_data::{
    BarTimestampConvention, CanonicalOhlcvPublishRequest, CanonicalVolumeRef, Ohlcv,
    publish_canonical_ohlcv_generation,
};
use neoethos_search::Gene;
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;

fn unique_test_root(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "neoethos-app-{label}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos()
    ))
}

fn sample_search_input_receipt() -> neoethos_search::CanonicalSearchInputReceiptV2 {
    let features = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
    let anchor = features.provenance().bindings()[0]
        .dataset_identity()
        .clone();
    neoethos_search::CanonicalSearchInputReceiptV2::from_feature_frame(&anchor, &features)
        .expect("canonical search test receipt")
}

fn sample_discovery_authority() -> (
    neoethos_search::CanonicalSearchInputReceiptV2,
    neoethos_search::CanonicalSearchArtifactScopeV2,
) {
    let receipt = sample_search_input_receipt();
    let scope = neoethos_search::CanonicalSearchArtifactScopeV2::for_entire_receipt(
        neoethos_search::CanonicalSearchWindowRoleV1::DiscoveryInput,
        receipt.clone(),
    )
    .expect("canonical full DiscoveryInput test scope");
    (receipt, scope)
}

fn publish_two_bar_fixture(
    root: &Path,
    identity: &CanonicalDatasetIdentity,
    close: f64,
    expected_generation: Option<&str>,
) -> SelectedDatasetGenerationV1 {
    let step_ms = identity
        .timeframe()
        .fixed_duration_ms()
        .expect("test fixture uses a fixed-duration timeframe");
    let seed = 1_700_000_000_000_i64;
    let start_ms = seed - seed.rem_euclid(step_ms);
    let ohlcv = Ohlcv {
        timestamp: Some(vec![start_ms, start_ms + step_ms]),
        open: vec![close, close],
        high: vec![close, close],
        low: vec![close, close],
        close: vec![close, close],
        volume: None,
    };
    let provenance = ProducerProvenanceEnvelopeV1::new(
        "neoethos.app-exact-selection-test.v1",
        identity.canonical_bytes(),
    )
    .expect("test provenance");
    let published = publish_canonical_ohlcv_generation(CanonicalOhlcvPublishRequest {
        configured_root: root,
        identity,
        expected_generation,
        provenance: &provenance,
        ohlcv: &ohlcv,
        volume: CanonicalVolumeRef::Absent,
        rows_per_chunk: 2,
    })
    .expect("publish canonical test fixture");
    SelectedDatasetGenerationV1::from_manifest(published.manifest())
        .expect("selected test generation")
}

fn request_for(base: CanonicalTimeframe, higher_tfs: Vec<String>) -> (PathBuf, DiscoveryRequest) {
    let root = unique_test_root("pinned-request");
    std::fs::create_dir_all(&root).expect("create pinned request root");
    let base_identity = CanonicalDatasetIdentity::external(
        "embedded-ctrader-fixture-unverified",
        neoethos_data::test_fixtures::ctrader_sample_symbol(),
        base,
        BarTimestampConvention::BarOpen,
    )
    .expect("valid exact fixture identity");
    let anchor = publish_two_bar_fixture(&root, &base_identity, 1.125, None);
    for label in &higher_tfs {
        let timeframe = label
            .parse::<CanonicalTimeframe>()
            .expect("canonical test higher timeframe");
        let identity = identity_for_timeframe(&base_identity, timeframe)
            .expect("same-series higher timeframe identity");
        publish_two_bar_fixture(&root, &identity, 1.125, None);
    }
    let pinned =
        pin_discovery_input(&root, anchor, &higher_tfs).expect("pin exact test generations");
    let request = DiscoveryRequest {
        data_root: root.clone(),
        pinned_input: Arc::new(pinned),
        higher_tfs,
        config: neoethos_search::DiscoveryConfig::default(),
        prop_firm_rules: PropFirmRiskRules::default(),
    };
    (root, request)
}

struct TestRequest {
    root: PathBuf,
    request: Option<DiscoveryRequest>,
}

impl std::ops::Deref for TestRequest {
    type Target = DiscoveryRequest;

    fn deref(&self) -> &Self::Target {
        self.request.as_ref().expect("test request is live")
    }
}

impl std::ops::DerefMut for TestRequest {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.request.as_mut().expect("test request is live")
    }
}

impl Drop for TestRequest {
    fn drop(&mut self) {
        drop(self.request.take());
        std::fs::remove_dir_all(&self.root).expect("remove sample discovery test root");
    }
}

fn sample_request() -> TestRequest {
    let (root, request) = request_for(CanonicalTimeframe::M1, Vec::new());
    TestRequest {
        root,
        request: Some(request),
    }
}

#[test]
fn invalid_request_fails_before_launch() {
    let mut request = sample_request();
    request.data_root = PathBuf::new();

    let err = request
        .validate()
        .expect_err("expected invalid request to fail");
    assert!(err.to_string().contains("data root"));
}

#[test]
fn duplicate_higher_timeframes_fail_instead_of_hashing_a_different_request() {
    let mut request = sample_request();
    request.higher_tfs = vec!["M5".to_owned(), "m5".to_owned()];

    let error = request
        .validate()
        .expect_err("case-normalized duplicate timeframe must fail closed");
    assert!(error.to_string().contains("duplicate higher timeframe M5"));
}

#[test]
fn a_higher_timeframe_must_be_strictly_above_the_selected_base() {
    let mut request = sample_request();
    request.higher_tfs = vec!["M1".to_owned()];

    let error = request
        .validate()
        .expect_err("base timeframe cannot also be a higher timeframe");
    assert!(error.to_string().contains("strictly above base M1"));
}

#[test]
fn request_symbol_and_base_timeframe_are_derived_from_the_pinned_receipt() {
    let request = sample_request();

    assert_eq!(request.symbol(), "EURUSD");
    assert_eq!(request.base_tf(), "M1");
    assert_eq!(
        request.dataset_identity(),
        request.pinned_input.receipt().anchor().identity(),
        "request selectors must come only from the pinned receipt"
    );
}

#[test]
fn discovery_requires_only_base_and_explicit_higher_timeframes() {
    let (h4_root, h4_request) = request_for(CanonicalTimeframe::H4, Vec::new());
    assert_eq!(
        required_direct_timeframes(&h4_request).expect("base-only direct set"),
        vec![CanonicalTimeframe::H4]
    );
    drop(h4_request);
    std::fs::remove_dir_all(h4_root).expect("remove H4 test root");

    let (m15_root, m15_request) = request_for(CanonicalTimeframe::M15, vec!["H1".to_owned()]);
    assert_eq!(
        required_direct_timeframes(&m15_request).expect("base plus explicit higher direct set"),
        vec![CanonicalTimeframe::M15, CanonicalTimeframe::H1]
    );
    drop(m15_request);
    std::fs::remove_dir_all(m15_root).expect("remove M15/H1 test root");
}

#[test]
fn pinned_input_ignores_another_legitimate_source_and_survives_pointer_advance() {
    let root = unique_test_root("exact-discovery");
    std::fs::create_dir_all(&root).expect("create exact-selection test root");

    let selected = CanonicalDatasetIdentity::external(
        "selected-source",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("selected identity");
    let other = CanonicalDatasetIdentity::external(
        "other-source",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("other identity");
    let generation_one = publish_two_bar_fixture(&root, &selected, 1.125, None);
    publish_two_bar_fixture(&root, &other, 9.875, None);

    let pinned =
        pin_discovery_input(&root, generation_one.clone(), &[]).expect("pin selected generation");

    publish_two_bar_fixture(
        &root,
        &selected,
        1.250,
        Some(generation_one.generation_id()),
    );
    #[cfg(not(feature = "gpu-nvidia"))]
    {
        let dataset = pinned
            .take_pinned_series_v1()
            .expect("move exact pinned series")
            .into_cpu_dataset_without_native_adapter_v1()
            .expect("decode the reader-leased generation after pointer advance");
        assert_eq!(dataset.frames["M1"].close, vec![1.125, 1.125]);
        assert_eq!(dataset.source_artifacts["M1"].identity(), &selected);
    }
    #[cfg(feature = "gpu-nvidia")]
    assert_eq!(
        pinned.receipt().anchor().generation_id(),
        generation_one.generation_id(),
        "the CUDA build retains the exact leased generation until a sealed route selects its factory"
    );

    let stale = pin_discovery_input(&root, generation_one, &[])
        .expect_err("a new run with the stale receipt must conflict");
    assert!(
        stale
            .downcast_ref::<neoethos_data::ExactDatasetGenerationConflict>()
            .is_some(),
        "stale receipt must remain a typed conflict: {stale:#}"
    );

    drop(pinned);
    std::fs::remove_dir_all(&root).expect("remove exact-selection test root");
}

#[test]
fn background_identity_resolution_rejects_ambiguity_and_lists_every_candidate() {
    let first = CanonicalDatasetIdentity::external(
        "source-a",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("first identity");
    let second = CanonicalDatasetIdentity::external(
        "source-b",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("second identity");

    let error =
        select_unique_background_identity(vec![second.clone(), first.clone()], "EURUSD", "M1")
            .expect_err("background selection must not pick first when two series match");
    let message = error.to_string();
    assert!(message.contains(&first.to_path_component()));
    assert!(message.contains(&second.to_path_component()));
}

#[test]
fn background_identity_resolution_rejects_zero_exact_matches_and_lists_known_series() {
    let m5 = CanonicalDatasetIdentity::external(
        "source-a",
        "EURUSD",
        CanonicalTimeframe::M5,
        BarTimestampConvention::BarOpen,
    )
    .expect("M5 identity");
    let h1 = CanonicalDatasetIdentity::external(
        "source-b",
        "EURUSD",
        CanonicalTimeframe::H1,
        BarTimestampConvention::BarOpen,
    )
    .expect("H1 identity");

    let error = select_unique_background_identity(vec![h1.clone(), m5.clone()], "EURUSD", "M1")
        .expect_err("background selection must not choose another timeframe");
    let message = error.to_string();
    assert!(message.contains(&m5.to_path_component()));
    assert!(message.contains(&h1.to_path_component()));
}

#[test]
fn background_identity_resolution_returns_the_only_exact_match() {
    let selected = CanonicalDatasetIdentity::external(
        "source-a",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("selected identity");
    let other_timeframe =
        identity_for_timeframe(&selected, CanonicalTimeframe::H1).expect("same series H1 identity");

    let resolved =
        select_unique_background_identity(vec![other_timeframe, selected.clone()], "eurusd", "m1")
            .expect("one exact background identity");
    assert_eq!(resolved, selected);
}

#[test]
fn target_timeframe_identity_preserves_the_exact_source_or_broker_scope() {
    let request = sample_request();
    let external = request.dataset_identity().clone();
    let external_h1 = identity_for_timeframe(&external, CanonicalTimeframe::H1)
        .expect("derive exact external H1 identity");
    assert_eq!(external_h1.scope(), external.scope());
    assert_eq!(external_h1.symbol_name(), external.symbol_name());
    assert_eq!(external_h1.timeframe(), CanonicalTimeframe::H1);

    let broker = CanonicalDatasetIdentity::ctrader(
        neoethos_data::CTraderEnvironment::Demo,
        "demo.ctraderapi.com",
        42,
        1,
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("broker identity");
    let broker_h4 = identity_for_timeframe(&broker, CanonicalTimeframe::H4)
        .expect("derive exact broker H4 identity");
    assert_eq!(broker_h4.scope(), broker.scope());
    assert_eq!(broker_h4.symbol_name(), broker.symbol_name());
    assert_eq!(broker_h4.timeframe(), CanonicalTimeframe::H4);
}

#[test]
fn required_direct_timeframe_set_contains_only_base_and_explicit_higher_frames() {
    let mut request = sample_request();
    request.higher_tfs = vec!["M5".to_owned(), "H1".to_owned(), "H4".to_owned()];

    let required = required_direct_timeframes(&request).expect("canonical direct timeframe set");
    assert_eq!(
        required,
        vec![
            CanonicalTimeframe::M1,
            CanonicalTimeframe::M5,
            CanonicalTimeframe::H1,
            CanonicalTimeframe::H4,
        ]
    );
}

#[test]
fn cancellation_request_maps_to_cancelled_snapshot() {
    let snapshot = cancelled_snapshot(JobKind::Discovery, "operator cancelled discovery");

    assert_eq!(snapshot.state, JobState::Cancelled);
    assert_eq!(snapshot.report.summary, "operator cancelled discovery");
}

#[test]
fn empty_portfolio_failure_maps_to_failed_snapshot() {
    let snapshot = failed_snapshot(
        JobKind::Discovery,
        anyhow::anyhow!("Discovery produced an empty portfolio for EURUSD M1 (candidates=4)"),
    );

    assert_eq!(snapshot.state, JobState::Failed);
    assert_eq!(snapshot.report.errors.len(), 1);
    assert!(snapshot.report.errors[0].contains("empty portfolio"));
}

#[test]
fn success_snapshot_carries_candidate_and_portfolio_counters() {
    let best = Gene {
        strategy_id: "alpha-1".to_string(),
        fitness: 1450.0,
        sharpe_ratio: 1.82,
        win_rate: 0.64,
        ..Gene::default()
    };

    let second = Gene {
        strategy_id: "alpha-2".to_string(),
        fitness: 1200.0,
        sharpe_ratio: 1.55,
        win_rate: 0.59,
        ..Gene::default()
    };

    let (search_input_receipt, selection_scope) = sample_discovery_authority();
    let result = DiscoveryResult {
        search_input_receipt,
        selection_scope,
        holdout_scope: None,
        search_config_hash: "fnv64:0123456789abcdef".to_string(),
        cost_band_by_strategy: Vec::new(),
        portfolio: vec![best.clone(), second],
        candidates: vec![best, Gene::default(), Gene::default()],
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: Vec::new(),
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: Vec::new(),
        prop_firm_validation_artifacts: Vec::new(),
        funnel_profile: None,

        effective_smc_gate_threshold: f64::NAN,
    };

    let snapshot = completed_snapshot(JobSnapshot::new(JobKind::Discovery), &result);

    assert_eq!(snapshot.state, JobState::Succeeded);
    assert_eq!(
        snapshot.report.counters,
        vec![
            ("candidates".to_string(), 3),
            ("portfolio".to_string(), 2),
            ("rejected".to_string(), 1),
            ("quality_scored".to_string(), 0),
            ("trade_logs".to_string(), 0),
        ]
    );
    assert!(
        snapshot
            .report
            .highlights
            .iter()
            .any(|(name, value)| { name == "best_strategy" && value == "alpha-1" })
    );
    assert!(
        snapshot
            .report
            .highlights
            .iter()
            .any(|(name, value)| { name == "best_sharpe" && value == "1.82" })
    );
    assert!(
        snapshot
            .report
            .entries
            .iter()
            .any(|entry| entry.contains("alpha-1") && entry.contains("win_rate=0.64"))
    );
    assert!(
        snapshot
            .report
            .events
            .iter()
            .any(|event| event.message.contains("completed discovery"))
    );
}

// #211: the `completed_snapshot` highlight emits `best_oos_sharpe`
// taken from `forward_test_validation_artifacts`, distinct from
// `best_sharpe` (which is in-sample stage-1). Both columns end up in
// the validation CSV so a big IS-OOS gap can be spotted at-a-glance.
#[test]
fn success_snapshot_emits_best_oos_sharpe_from_forward_test_artifacts() {
    use neoethos_search::{
        BacktestMetrics, CanonicalSearchArtifactScopeV2, CanonicalSearchWindowRoleV1,
        ForwardTestSummary, ForwardTestValidationArtifactFile,
    };

    let best = Gene {
        strategy_id: "alpha-1".to_string(),
        fitness: 1450.0,
        // In-sample stage-1 Sharpe — the GA optimized for this.
        sharpe_ratio: 5.50,
        win_rate: 0.64,
        ..Gene::default()
    };

    let lo_oos_metrics = BacktestMetrics {
        net_profit: 0.0,
        sharpe: 1.20,
        peak_equity: 0.0,
        max_drawdown: 0.0,
        win_rate: 0.0,
        profit_factor: 0.0,
        expectancy: 0.0,
        monthly_target_hit_rate: 0.0,
        trade_count: 0,
        consistency: 0.0,
        max_daily_drawdown: 0.0,
    };
    let hi_oos_metrics = BacktestMetrics {
        sharpe: 1.55,
        ..lo_oos_metrics
    };

    let (search_input_receipt, selection_scope) = sample_discovery_authority();
    let holdout_scope = CanonicalSearchArtifactScopeV2::for_entire_receipt(
        CanonicalSearchWindowRoleV1::Holdout,
        search_input_receipt.clone(),
    )
    .expect("canonical full holdout test scope");
    let holdout_bars = (holdout_scope.evaluated_window().row_end()
        - holdout_scope.evaluated_window().row_start()) as usize;
    let search_config_hash = "fnv64:0123456789abcdef";

    let forward_artifacts = vec![
        ForwardTestValidationArtifactFile::new(
            holdout_scope.clone(),
            search_config_hash,
            &best,
            ForwardTestSummary {
                bars: holdout_bars,
                metrics: lo_oos_metrics,
                span_days: 1.0,
            },
        )
        .expect("lower-Sharpe canonical forward-test artifact"),
        ForwardTestValidationArtifactFile::new(
            holdout_scope.clone(),
            search_config_hash,
            &best,
            ForwardTestSummary {
                bars: holdout_bars,
                metrics: hi_oos_metrics,
                span_days: 1.0,
            },
        )
        .expect("higher-Sharpe canonical forward-test artifact"),
    ];

    let result = DiscoveryResult {
        search_input_receipt,
        selection_scope,
        holdout_scope: Some(holdout_scope),
        search_config_hash: search_config_hash.to_string(),
        cost_band_by_strategy: Vec::new(),
        portfolio: vec![best.clone()],
        candidates: vec![best],
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: Vec::new(),
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: forward_artifacts,
        prop_firm_validation_artifacts: Vec::new(),
        funnel_profile: None,

        effective_smc_gate_threshold: f64::NAN,
    };

    let snapshot = completed_snapshot(JobSnapshot::new(JobKind::Discovery), &result);

    // In-sample Sharpe still emitted (unchanged from prior contract).
    assert!(
        snapshot
            .report
            .highlights
            .iter()
            .any(|(name, value)| { name == "best_sharpe" && value == "5.50" }),
        "best_sharpe (in-sample) must still be present"
    );
    // New OOS highlight picks the MAX Sharpe across the forward-test
    // tail artifacts — 1.55 wins over 1.20.
    assert!(
        snapshot
            .report
            .highlights
            .iter()
            .any(|(name, value)| { name == "best_oos_sharpe" && value == "1.5500" }),
        "best_oos_sharpe must be the max forward-test sharpe (1.55)"
    );
}

#[test]
fn success_snapshot_omits_best_oos_sharpe_when_forward_test_artifacts_empty() {
    // Backward compatibility: when no forward-test artifacts are
    // produced (tail too short, or `compute_discovery_forward_test_artifacts`
    // failed) the highlight is simply absent. The validation harness
    // treats absence as `None` and falls back to in-sample reporting.
    let best = Gene {
        strategy_id: "alpha-1".to_string(),
        sharpe_ratio: 1.82,
        ..Gene::default()
    };
    let (search_input_receipt, selection_scope) = sample_discovery_authority();
    let result = DiscoveryResult {
        search_input_receipt,
        selection_scope,
        holdout_scope: None,
        search_config_hash: "fnv64:0123456789abcdef".to_string(),
        cost_band_by_strategy: Vec::new(),
        portfolio: vec![best.clone()],
        candidates: vec![best],
        quality_metrics: Vec::new(),
        logged_trades: Vec::new(),
        effective_feature_names: Vec::new(),
        validation_gates: DiscoveryValidationGates::pending(),
        canonical_backtest_artifacts: Vec::new(),
        walkforward_validation_artifacts: Vec::new(),
        forward_test_validation_artifacts: Vec::new(),
        prop_firm_validation_artifacts: Vec::new(),
        funnel_profile: None,

        effective_smc_gate_threshold: f64::NAN,
    };
    let snapshot = completed_snapshot(JobSnapshot::new(JobKind::Discovery), &result);
    assert!(
        !snapshot
            .report
            .highlights
            .iter()
            .any(|(name, _)| name == "best_oos_sharpe"),
        "best_oos_sharpe must be absent when no forward-test artifacts exist"
    );
    // best_sharpe (in-sample) is still emitted.
    assert!(
        snapshot
            .report
            .highlights
            .iter()
            .any(|(name, _)| name == "best_sharpe")
    );
}

#[tokio::test]
async fn start_discovery_job_emits_initial_snapshot_with_requested_targets() {
    let higher_tfs = vec!["M5".to_string(), "M15".to_string(), "H1".to_string()];
    let (root, mut request) = request_for(CanonicalTimeframe::M1, higher_tfs);
    request.config.population = 96;
    request.config.generations = 7;
    request.config.candidate_count = 144;
    request.config.portfolio_size = 24;
    let (tx, mut rx) = mpsc::channel(10000);

    let handle = start_discovery_job(request.clone(), tx).expect("job should start");
    let event = rx.recv().await.expect("expected initial discovery event");
    let ServiceEvent::DiscoveryUpdated(snapshot) = event else {
        panic!("expected discovery update event");
    };

    assert_eq!(snapshot.state, JobState::Running);
    assert_eq!(snapshot.progress.stage, "using_pinned_data");
    assert_eq!(
        snapshot.report.counters,
        vec![
            ("target_candidates".to_string(), 144),
            ("target_portfolio".to_string(), 24),
            ("generations".to_string(), 7),
            ("population".to_string(), 96),
        ]
    );
    assert!(
        snapshot
            .report
            .highlights
            .iter()
            .any(|(name, value)| name == "symbol" && value == "EURUSD")
    );
    assert!(
        snapshot
            .report
            .highlights
            .iter()
            .any(|(name, value)| name == "higher_tfs" && value == "M5, M15, H1")
    );
    assert!(snapshot.report.events.iter().any(|event| {
        event.message.contains("planned discovery")
            && event.message.contains("candidate_count=144")
            && event.message.contains("portfolio_size=24")
    }));
    assert_eq!(
        snapshot.report.log_path,
        Some(canonical_log_path().display().to_string())
    );

    handle.cancel.request();
    drop(request);
    loop {
        let next = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("discovery cancellation timed out");
        let Some(ServiceEvent::DiscoveryUpdated(update)) = next else {
            break;
        };
        if matches!(
            update.state,
            JobState::Cancelled | JobState::Failed | JobState::Succeeded | JobState::Degraded
        ) {
            break;
        }
    }
    drop(handle);
    std::fs::remove_dir_all(root).expect("remove start-job test root");
}

#[test]
fn backend_portfolio_milestone_updates_discovery_snapshot_with_live_counts() {
    let request = sample_request();
    let mut snapshot = JobSnapshot::new(JobKind::Discovery);
    snapshot.state = JobState::Running;
    snapshot.progress = JobProgress {
        percent: Some(0.75),
        stage: "running_discovery".to_string(),
        message: "evaluating strategy candidates for EURUSD".to_string(),
    };
    snapshot.report = JobReport {
        counters: requested_discovery_counters(&request),
        highlights: requested_discovery_highlights(&request),
        log_path: Some(canonical_log_path().display().to_string()),
        ..JobReport::default()
    };

    apply_backend_discovery_event(
        &mut snapshot,
        &neoethos_search::DiscoveryProgress::PortfolioSelected {
            portfolio_size: 12,
            rejected_by_correlation: 5,
            target_portfolio: 24,
        },
    );

    assert_eq!(snapshot.state, JobState::Running);
    assert_eq!(snapshot.progress.stage, "portfolio_construction");
    assert!(snapshot.progress.percent.expect("percent should exist") >= 0.9);
    assert!(
        snapshot
            .report
            .counters
            .iter()
            .any(|(name, value)| name == "portfolio" && *value == 12)
    );
    assert!(
        snapshot
            .report
            .counters
            .iter()
            .any(|(name, value)| name == "rejected_by_correlation" && *value == 5)
    );
    assert!(
        snapshot
            .report
            .events
            .iter()
            .any(|event| event.message.contains("portfolio selection"))
    );
    assert!(
        snapshot
            .report
            .entries
            .iter()
            .any(|entry| entry.contains("portfolio | accepted=12"))
    );
}
