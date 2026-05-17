use super::*;


use crate::app_services::{
    ServiceEvent,
    jobs::{JobKind, JobSnapshot, JobState},
};
use forex_search::Gene;
use std::path::PathBuf;
use tokio::sync::mpsc;

fn sample_request() -> DiscoveryRequest {
    DiscoveryRequest {
        data_root: PathBuf::from("data"),
        symbol: "EURUSD".to_string(),
        base_tf: "M1".to_string(),
        higher_tfs: vec!["M5".to_string(), "M15".to_string()],
        config: forex_search::DiscoveryConfig::default(),
        prop_firm_rules: PropFirmRiskRules::default(),
    }
}

#[test]
fn invalid_request_fails_before_launch() {
    let mut request = sample_request();
    request.symbol.clear();

    let err = request
        .validate()
        .expect_err("expected invalid request to fail");
    assert!(err.to_string().contains("symbol"));
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

    let result = DiscoveryResult {
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

#[tokio::test]
async fn start_discovery_job_emits_initial_snapshot_with_requested_targets() {
    let mut request = sample_request();
    request.higher_tfs = vec!["M5".to_string(), "M15".to_string(), "H1".to_string()];
    request.config.population = 96;
    request.config.generations = 7;
    request.config.candidate_count = 144;
    request.config.portfolio_size = 24;
    let (tx, mut rx) = mpsc::channel(10000);

    let _handle = start_discovery_job(request.clone(), tx).expect("job should start");
    let event = rx.recv().await.expect("expected initial discovery event");
    let ServiceEvent::DiscoveryUpdated(snapshot) = event else {
        panic!("expected discovery update event");
    };

    assert_eq!(snapshot.state, JobState::Running);
    assert_eq!(snapshot.progress.stage, "loading_data");
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
        &forex_search::DiscoveryProgress::PortfolioSelected {
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

// ── Multi-symbol discovery fan-out (audit gap #1) ─────────────────────

#[test]
fn multi_symbol_request_validate_rejects_empty_symbol_list() {
    let req = MultiSymbolDiscoveryRequest {
        data_root: PathBuf::from("./data"),
        symbols: Vec::new(),
        base_tf: "M5".to_string(),
        higher_tfs: vec!["M15".to_string()],
        config: forex_search::DiscoveryConfig::default(),
        prop_firm_rules: PropFirmRiskRules::default(),
    };
    let err = req.validate().expect_err("empty symbols must reject");
    assert!(
        err.to_string().contains("at least one symbol"),
        "wrong error message: {err}"
    );
}

#[test]
fn multi_symbol_request_validate_rejects_empty_string_in_list() {
    let req = MultiSymbolDiscoveryRequest {
        data_root: PathBuf::from("./data"),
        symbols: vec!["EURUSD".to_string(), "  ".to_string()],
        base_tf: "M5".to_string(),
        higher_tfs: vec!["M15".to_string()],
        config: forex_search::DiscoveryConfig::default(),
        prop_firm_rules: PropFirmRiskRules::default(),
    };
    let err = req.validate().expect_err("whitespace symbol must reject");
    assert!(
        err.to_string().contains("empty symbol"),
        "wrong error message: {err}"
    );
}

#[test]
fn multi_symbol_request_validate_rejects_empty_base_tf() {
    let req = MultiSymbolDiscoveryRequest {
        data_root: PathBuf::from("./data"),
        symbols: vec!["EURUSD".to_string()],
        base_tf: "".to_string(),
        higher_tfs: vec![],
        config: forex_search::DiscoveryConfig::default(),
        prop_firm_rules: PropFirmRiskRules::default(),
    };
    assert!(req.validate().is_err());
}

#[test]
fn multi_symbol_request_validate_rejects_empty_data_root() {
    let req = MultiSymbolDiscoveryRequest {
        data_root: PathBuf::new(),
        symbols: vec!["EURUSD".to_string()],
        base_tf: "M5".to_string(),
        higher_tfs: vec![],
        config: forex_search::DiscoveryConfig::default(),
        prop_firm_rules: PropFirmRiskRules::default(),
    };
    assert!(req.validate().is_err());
}

#[test]
fn multi_symbol_into_single_symbol_requests_produces_one_per_symbol() {
    let symbols = vec!["EURUSD".to_string(), "GBPUSD".to_string(), "XAUUSD".to_string()];
    let req = MultiSymbolDiscoveryRequest {
        data_root: PathBuf::from("./data"),
        symbols: symbols.clone(),
        base_tf: "M5".to_string(),
        higher_tfs: vec!["M15".to_string(), "H1".to_string()],
        config: forex_search::DiscoveryConfig::default(),
        prop_firm_rules: PropFirmRiskRules::default(),
    };
    let singles = req.into_single_symbol_requests();
    assert_eq!(singles.len(), 3, "must produce one request per symbol");
    // Order must match input order so the UI can map result back.
    assert_eq!(singles[0].symbol, "EURUSD");
    assert_eq!(singles[1].symbol, "GBPUSD");
    assert_eq!(singles[2].symbol, "XAUUSD");
    // Shared config preserved across all clones.
    for req in &singles {
        assert_eq!(req.base_tf, "M5");
        assert_eq!(req.higher_tfs, vec!["M15".to_string(), "H1".to_string()]);
        assert_eq!(req.data_root, PathBuf::from("./data"));
    }
}

#[test]
fn multi_symbol_each_single_request_passes_its_own_validate() {
    let req = MultiSymbolDiscoveryRequest {
        data_root: PathBuf::from("./data"),
        symbols: vec!["EURUSD".to_string(), "GBPUSD".to_string()],
        base_tf: "M5".to_string(),
        higher_tfs: vec![],
        config: forex_search::DiscoveryConfig::default(),
        prop_firm_rules: PropFirmRiskRules::default(),
    };
    for single in req.into_single_symbol_requests() {
        assert!(
            single.validate().is_ok(),
            "fan-out child failed its own validate: {:?}",
            single
        );
    }
}
