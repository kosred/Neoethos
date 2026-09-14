use super::*;
use axum::body::{Body, to_bytes};
use axum::http::Request;
use axum::routing::get;
use neoethos_models::{
    MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1, PromotionCandidateTrainingManifestV1,
};
use tower::ServiceExt;

struct TestRoot(std::path::PathBuf);

impl TestRoot {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let sequence = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "neoethos-intelligence-research-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn intelligence_inventory_preserves_valid_handoffs_and_reports_corrupt_neighbors_read_only() {
    let root = TestRoot::new();
    let data = root.0.join("data");
    let models = root.0.join("models");
    // Genuine sealed handoff only: no models, prices, training or final use.
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/combined-research-candidate-v1/training-handoff.json");
    let bytes = std::fs::read(fixture).unwrap();
    let selected: neoethos_models::PromotionCandidateTrainingHandoffV1 =
        serde_json::from_slice(&bytes).unwrap();
    let identity = handoff::publish(&data, &selected).unwrap();
    let portfolio = selected
        .locked_portfolio()
        .deserialize_live_portfolio()
        .unwrap();
    let directory = data.join("discovery_targets");
    let corrupt = "e".repeat(64);
    let foreign = "f".repeat(64);
    assert_ne!(identity, corrupt);
    assert_ne!(identity, foreign);
    for (name, contents) in [
        (corrupt.clone(), b"{broken".to_vec()),
        (foreign.clone(), bytes),
        ("notes".to_owned(), b"{}".to_vec()),
    ] {
        std::fs::write(
            directory.join(format!("{name}.training-handoff.json")),
            contents,
        )
        .unwrap();
    }
    let before: std::collections::BTreeMap<_, _> = std::fs::read_dir(&directory)
        .unwrap()
        .map(|entry| {
            let path = entry.unwrap().path();
            let bytes = std::fs::read(&path).unwrap();
            (path, bytes)
        })
        .collect();
    for models_exist in [false, true] {
        if models_exist {
            std::fs::create_dir(&models).unwrap();
        }
        let dto = scan_intelligence_at(&data, &models).unwrap();
        assert_eq!(dto.models_dir_exists, models_exist);
        assert_eq!(dto.training_handoffs.len(), 1);
        assert_eq!(dto.training_handoffs[0].identity, identity);
        assert_eq!(dto.discovery_targets.len(), portfolio.genes.len());
        for ((target, gene), evidence) in dto
            .discovery_targets
            .iter()
            .zip(&portfolio.genes)
            .zip(&portfolio.sizing_evidence)
        {
            assert_eq!(target.strategy_id, gene.strategy_id);
            assert_eq!(target.sharpe, Some(evidence.oos_metrics().sharpe));
            assert_eq!(target.win_rate, Some(evidence.oos_metrics().win_rate));
        }
        let unavailable = &dto.training_handoff_unavailable;
        assert_eq!(unavailable.len(), 3);
        assert_eq!(
            unavailable
                .iter()
                .map(|row| row.identity.as_str())
                .collect::<Vec<_>>(),
            vec![corrupt.as_str(), foreign.as_str(), "notes"]
        );
        assert!(unavailable.iter().all(|row| !row.reason.is_empty()));
        assert!(unavailable[1].reason.contains("identity changed"));
        assert!(unavailable[2].reason.contains("canonical lowercase"));
        let wire = serde_json::to_value(&dto).unwrap();
        assert_eq!(
            wire["trainingHandoffUnavailable"].as_array().unwrap().len(),
            3
        );
        assert!(wire["trainingHandoffs"][0].get("strategies").is_none());
    }
    // Inventory diagnostics never relax strict selection for start/report reads.
    for invalid in [corrupt.as_str(), foreign.as_str(), "notes"] {
        assert!(handoff::load(&data, invalid).is_err());
    }
    assert_eq!(
        handoff::load(&data, &identity)
            .unwrap()
            .identity_sha256()
            .unwrap(),
        identity
    );
    assert_eq!(std::fs::read_dir(&directory).unwrap().count(), before.len());
    for (path, bytes) in before {
        assert_eq!(std::fs::read(path).unwrap(), bytes);
    }
    assert_eq!(std::fs::read_dir(&models).unwrap().count(), 0);
}

#[test]
fn intelligence_inventory_distinguishes_missing_directory_from_unreadable_inventory() {
    let root = TestRoot::new();
    let data = root.0.join("data");
    let models = root.0.join("models");
    let empty = scan_intelligence_at(&data, &models).unwrap();
    assert!(empty.training_handoffs.is_empty());
    assert!(empty.training_handoff_unavailable.is_empty());
    assert_eq!(std::fs::read_dir(&root.0).unwrap().count(), 0);
    std::fs::create_dir(&data).unwrap();
    let wrong_kind = data.join("discovery_targets");
    std::fs::write(&wrong_kind, b"not a directory").unwrap();
    assert!(scan_intelligence_at(&data, &models).is_err());
    assert_eq!(std::fs::read(wrong_kind).unwrap(), b"not a directory");
}

#[tokio::test]
async fn research_http_requires_one_canonical_selector_before_configuration_or_disk_access() {
    // Use the real HTTP handler/extractor, without the unrelated process-global
    // auth-token fixture or any config override. Every request must fail before
    // Settings is read or a blocking candidate verification is scheduled.
    let router = axum::Router::new()
        .route("/intelligence/research", get(research))
        .with_state(AppApiState::new());
    let identity = "a".repeat(64);
    for query in [
        String::new(),
        "?training_handoff=".to_owned(),
        "?training_handoff=..%2Fconfig.yaml".to_owned(),
        format!("?training_handoff={}", "A".repeat(64)),
        format!("?training_handoff={}", "a".repeat(63)),
        format!("?training_handoff={identity}&path=outside"),
        format!("?training_handoff={identity}&training_handoff={identity}"),
    ] {
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/intelligence/research{query}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{query}");
        let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(value["error"].as_str().is_some(), "{value}");
        assert!(value["detail"].as_str().is_some(), "{value}");
        assert!(!value["error"].as_str().unwrap().contains("configuration"));
    }
}

#[test]
fn research_missing_selection_is_not_a_successful_empty_report_and_creates_nothing() {
    let root = TestRoot::new();
    let error = read_selected_research(
        &root.0.join("data"),
        &root.0.join("candidates"),
        &"a".repeat(64),
    )
    .unwrap_err();
    assert_eq!(error.status, StatusCode::NOT_FOUND);
    assert!(error.message.contains("handoff"));
    assert_eq!(std::fs::read_dir(&root.0).unwrap().count(), 0);
}

#[test]
fn research_invalid_saved_selection_is_not_mistaken_for_an_untrained_candidate() {
    let root = TestRoot::new();
    let identity = "a".repeat(64);
    let path = handoff::handoff_path(&root.0, &identity).unwrap();
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, b"{}").unwrap();
    let error = read_selected_research(&root.0, &root.0.join("candidates"), &identity).unwrap_err();
    assert_eq!(error.status, StatusCode::BAD_REQUEST);
    assert_eq!(std::fs::read(&path).unwrap(), b"{}");
    assert!(!root.0.join("candidates").exists());
}

#[test]
fn research_checked_context_is_request_local_and_new_reads_revalidate_selection() {
    let root = TestRoot::new();
    let identity =
        crate::app_services::training::install_saved_strategy_research_test_fixture(&root.0)
            .unwrap();
    let data = root.0.join("data");
    let candidates = root.0.join("models/candidates");
    let expected = read_selected_research(&data, &candidates, &identity).unwrap();
    let context = load_saved_final_research_context(&data, &identity).unwrap();
    let path = handoff::handoff_path(&data, &identity).unwrap();
    let original = std::fs::read(&path).unwrap();

    // The in-flight request owns its checked snapshot; no second handoff read
    // is needed. A separate request must not reuse that projection or hide the
    // newly corrupt/missing selection behind an empty successful response.
    std::fs::write(&path, b"{}").unwrap();
    let same_request =
        read_saved_final_research_reports_with_context(&data, &candidates, context).unwrap();
    assert_eq!(
        serde_json::to_value(same_request).unwrap(),
        serde_json::to_value(expected).unwrap()
    );
    let invalid = read_selected_research(&data, &candidates, &identity).unwrap_err();
    assert_eq!(invalid.status, StatusCode::BAD_REQUEST);
    assert!(invalid.message.contains("handoff"));
    std::fs::remove_file(&path).unwrap();
    let missing = read_selected_research(&data, &candidates, &identity).unwrap_err();
    assert_eq!(missing.status, StatusCode::NOT_FOUND);
    std::fs::write(&path, original).unwrap();

    // Checked selection does not cache report bytes or bypass journal/hash
    // verification. Corruption after selection remains unavailable evidence.
    let context = load_saved_final_research_context(&data, &identity).unwrap();
    let report = std::fs::read_dir(&candidates)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    std::fs::write(&report, b"{}").unwrap();
    let corrupt_report =
        read_saved_final_research_reports_with_context(&data, &candidates, context).unwrap();
    assert!(corrupt_report.reports.is_empty());
    assert_eq!(corrupt_report.unavailable.len(), 1);
    assert!(
        corrupt_report.unavailable[0]
            .reason
            .contains("semantic SHA-256 mismatch")
    );
}

#[test]
fn research_reads_only_the_exact_manifest_and_bounds_regular_file_bytes() {
    let root = TestRoot::new();
    let identity =
        crate::app_services::training::install_saved_research_test_fixture(&root.0).unwrap();
    let candidates = root.0.join("models/candidates");
    let data = root.0.join("data");
    let other = candidates.join(format!("{}.manifest.json", "b".repeat(64)));
    std::fs::write(&other, b"{}").unwrap();
    let exact = candidates.join(format!("{identity}.manifest.json"));
    std::fs::remove_file(&exact).unwrap();
    let absent = read_selected_research(&data, &candidates, &identity).unwrap();
    assert_eq!(absent.status, "completed_results");
    assert!(absent.reports.is_empty());
    assert_eq!(absent.unavailable.len(), 2);

    std::fs::write(&exact, b"{}").unwrap();
    let malformed = read_selected_research(&data, &candidates, &identity).unwrap();
    assert!(malformed.reports.is_empty());
    assert_eq!(malformed.unavailable.len(), 2);
    assert_eq!(std::fs::read(&exact).unwrap(), b"{}");

    std::fs::File::create(&exact)
        .unwrap()
        .set_len(MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1 as u64 + 1)
        .unwrap();
    let oversized = read_selected_research(&data, &candidates, &identity).unwrap();
    assert!(oversized.reports.is_empty());
    assert_eq!(oversized.unavailable.len(), 2);
    assert!(
        oversized
            .unavailable
            .iter()
            .all(|row| row.reason.contains("exact byte bound"))
    );

    std::fs::remove_file(&exact).unwrap();
    std::fs::create_dir(&exact).unwrap();
    let directory = read_selected_research(&data, &candidates, &identity).unwrap();
    assert!(directory.reports.is_empty());
    assert_eq!(directory.unavailable.len(), 2);
    assert!(
        directory
            .unavailable
            .iter()
            .all(|row| row.reason.contains("not a regular file"))
    );
    assert_eq!(std::fs::read(&other).unwrap(), b"{}");
}

#[test]
fn research_error_classification_keeps_io_failures_distinct_from_bad_evidence() {
    let denied = anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::PermissionDenied))
        .context("read selected candidate manifest");
    let denied = candidate_read_error(denied);
    assert_eq!(denied.status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(format!("{:#}", denied.source).contains("read selected candidate manifest"));
    let invalid = candidate_read_error(anyhow::anyhow!("saved report identity changed"));
    assert_eq!(invalid.status, StatusCode::BAD_REQUEST);
}

const HTTP_FIXTURE_CHILD: &str = "NEOETHOS_SAVED_RESEARCH_HTTP_TEST_CHILD";
const HTTP_FIXTURE_MARKER: &str = "http-test-owned-root";
const HTTP_FIXTURE_COMPLETED: &str = "http-test-completed";

#[test]
fn research_http_reads_real_candidate_in_isolated_process() {
    if std::env::var_os(HTTP_FIXTURE_CHILD).is_some() {
        let root = std::env::current_dir().unwrap();
        assert_eq!(
            std::fs::read(root.join(HTTP_FIXTURE_MARKER)).unwrap(),
            b"owned isolated HTTP fixture"
        );
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(exercise_real_candidate_http(&root));
        return;
    }

    // A separate test process keeps the production relative models/candidates
    // path and the real route/config resolver, without touching the parent
    // harness's CWD, settings singleton, API token or repository model store.
    let root = TestRoot::new();
    let parent_cwd = std::env::current_dir().unwrap();
    std::fs::write(
        root.0.join(HTTP_FIXTURE_MARKER),
        b"owned isolated HTTP fixture",
    )
    .unwrap();
    let stdout = root.0.join("child.stdout.log");
    let stderr = root.0.join("child.stderr.log");
    let module = module_path!().split_once("::").unwrap().1;
    let test_name = format!("{module}::research_http_reads_real_candidate_in_isolated_process");
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &test_name, "--nocapture", "--test-threads=1"])
        .env(HTTP_FIXTURE_CHILD, "1")
        .current_dir(&root.0)
        .stdout(std::fs::File::create(&stdout).unwrap())
        .stderr(std::fs::File::create(&stderr).unwrap())
        .spawn()
        .expect("spawn the same compiled HTTP test harness in its owned directory");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {}
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("could not poll the owned HTTP fixture child: {error}");
            }
        }
        if std::time::Instant::now() >= deadline {
            // It may have exited between try_wait and kill; still join it.
            let _ = child.kill();
            child.wait().expect("join timed-out HTTP fixture child");
            break None;
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    };
    let out = std::fs::read_to_string(&stdout).unwrap();
    let err = std::fs::read_to_string(&stderr).unwrap();
    // Write to the real handles, not println/eprintln: libtest captures those
    // macros and otherwise discards successful child diagnostics. The outer
    // harness's final verdict remains the only count for this parent test.
    {
        use std::io::Write;
        let mut output = std::io::stdout().lock();
        writeln!(output, "BEGIN SAVED-RESEARCH HTTP CHILD STDOUT").unwrap();
        output.write_all(out.as_bytes()).unwrap();
        writeln!(output, "\nEND SAVED-RESEARCH HTTP CHILD STDOUT").unwrap();
        let mut errors = std::io::stderr().lock();
        writeln!(errors, "BEGIN SAVED-RESEARCH HTTP CHILD STDERR").unwrap();
        errors.write_all(err.as_bytes()).unwrap();
        writeln!(errors, "\nEND SAVED-RESEARCH HTTP CHILD STDERR").unwrap();
    }
    assert!(
        status.is_some_and(|status| status.success()),
        "real HTTP fixture child failed or timed out: {status:?}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
    );
    // A misspelled --exact filter must not masquerade as successful coverage.
    let completed =
        std::fs::read_to_string(root.0.join(HTTP_FIXTURE_COMPLETED)).unwrap_or_else(|error| {
            panic!("HTTP child did not reach its assertions: {error}\n{out}\n{err}")
        });
    handoff::handoff_path(&root.0, &completed)
        .expect("child completed a canonical selected handoff");
    assert_eq!(std::env::current_dir().unwrap(), parent_cwd);
}

async fn exercise_real_candidate_http(root: &Path) {
    let identity =
        crate::app_services::training::install_saved_strategy_research_test_fixture(root).expect(
            "publish genuine handoff and synthetic strategy research without installing any models",
        );
    let config_path = root.join("config.yaml");
    let config = serde_json::json!({"system": {"data_dir": root.join("data")}});
    std::fs::write(&config_path, serde_yaml_ng::to_string(&config).unwrap()).unwrap();
    crate::server::state::install_config_path(&config_path);
    let router = crate::server::router(AppApiState::new());

    let before = research_fixture_fingerprints(root);
    let (status, strategy) = request_real_research(&router, &identity).await;
    assert_eq!(status, StatusCode::OK, "{strategy}");
    assert_eq!(strategy["reports"].as_array().unwrap().len(), 1);
    assert!(strategy["unavailable"].as_array().unwrap().is_empty());
    let report = &strategy["reports"][0];
    assert_eq!(report["evaluationMode"], "strategy_only");
    assert_eq!(report["geneOnly"]["netProfit"], -20.0);
    assert_eq!(report["promotionEligible"], false);
    for field in [
        "combined",
        "blendMode",
        "blendGateFloor",
        "blendVetoBelow",
        "modelHistoryRows",
        "invalidModelSignalRows",
        "configuredLiveMlGate",
    ] {
        assert!(report.get(field).unwrap().is_null(), "{field}: {report}");
    }
    assert_eq!(
        std::fs::read_dir(root.join("models/candidates"))
            .unwrap()
            .count(),
        1,
        "only the strategy report exists: no model tree, input or manifest"
    );
    assert_eq!(before, research_fixture_fingerprints(root));
    assert_eq!(
        crate::app_services::training::install_saved_research_test_fixture(root).unwrap(),
        identity
    );

    let before = research_fixture_fingerprints(root);
    let (status, value) = request_real_research(&router, &identity).await;
    assert_eq!(status, StatusCode::OK, "{value}");
    assert_eq!(value["trainingHandoff"], identity);
    assert_eq!(value["status"], "completed_results");
    assert!(
        value["unavailable"].as_array().unwrap().is_empty(),
        "{value}"
    );
    let reports = value["reports"].as_array().unwrap();
    assert_eq!(
        reports.len(),
        3,
        "all three attempts across both modes, not a latest/best-only report"
    );
    let mut profits = Vec::new();
    let mut first = 0;
    let mut reused = 0;
    for report in reports {
        assert_eq!(report["accountCurrency"], "USD");
        assert_eq!(report["promotionEligible"], false);
        if report["evaluationMode"] == "train_models" {
            assert_eq!(
                report["geneOnly"]["netProfit"],
                report["combined"]["netProfit"]
            );
            profits.push(report["combined"]["netProfit"].as_f64().unwrap());
        } else {
            assert_eq!(report["evaluationMode"], "strategy_only");
            assert!(report["combined"].is_null());
        }
        assert!(report["rowEnd"].as_u64().unwrap() > report["rowStart"].as_u64().unwrap());
        assert_eq!(
            report["historicalExposure"],
            "unknown_before_this_local_journal_not_never_ever_seen_evidence"
        );
        match report["holdoutUse"].as_str().unwrap() {
            "first_recorded_local_use_of_reserved_final_scope" => first += 1,
            "reused_reserved_final_scope_research_only" => reused += 1,
            other => panic!("unrecognized final-window use {other}"),
        }
    }
    profits.sort_by(f64::total_cmp);
    assert_eq!(profits, vec![-20.0, 25.0], "retain the losing attempt too");
    assert_eq!((first, reused), (1, 2));
    assert_ne!(reports[0]["reportId"], reports[1]["reportId"]);
    assert_eq!(
        research_fixture_fingerprints(root),
        before,
        "GET must not modify candidate, journal, reports, handoff or config"
    );

    let candidate_root = root.join("models").join("candidates");
    let manifest_path = candidate_root.join(format!("{identity}.manifest.json"));
    let manifest_bytes = std::fs::read(&manifest_path).unwrap();
    let manifest: PromotionCandidateTrainingManifestV1 =
        serde_json::from_slice(&manifest_bytes).unwrap();
    let parked_manifest = root.join("parked.manifest.json");
    std::fs::rename(&manifest_path, &parked_manifest).unwrap();
    let (status, empty) = request_real_research(&router, &identity).await;
    assert_eq!(status, StatusCode::OK, "{empty}");
    assert_eq!(empty["status"], "completed_results");
    assert_eq!(empty["trainingHandoff"], identity);
    assert_eq!(empty["reports"].as_array().unwrap().len(), 1);
    assert_eq!(empty["reports"][0]["evaluationMode"], "strategy_only");
    assert_eq!(empty["unavailable"].as_array().unwrap().len(), 2);
    std::fs::rename(&parked_manifest, &manifest_path).unwrap();

    // Corruption after a successful read must not reuse a cached proof or yield
    // the old result. Change only this child-owned copy of installation evidence.
    let evidence = candidate_root
        .join(manifest.candidate_relative_dir())
        .join(neoethos_models::PROMOTION_CANDIDATE_TRAINING_EVIDENCE_FILE_V1);
    use std::io::Write;
    std::fs::OpenOptions::new()
        .append(true)
        .open(evidence)
        .unwrap()
        .write_all(b" ")
        .unwrap();
    let (status, corrupted) = request_real_research(&router, &identity).await;
    assert_eq!(status, StatusCode::OK, "{corrupted}");
    assert_eq!(corrupted["reports"].as_array().unwrap().len(), 1);
    assert_eq!(corrupted["reports"][0]["evaluationMode"], "strategy_only");
    assert_eq!(corrupted["unavailable"].as_array().unwrap().len(), 2);
    assert!(
        corrupted["unavailable"]
            .as_array()
            .unwrap()
            .iter()
            .all(|row| row["reason"]
                .as_str()
                .unwrap()
                .contains("installed candidate tree hash/count/bytes changed"))
    );
    std::fs::write(root.join(HTTP_FIXTURE_COMPLETED), identity).unwrap();
}

async fn request_real_research(
    router: &axum::Router,
    identity: &str,
) -> (StatusCode, serde_json::Value) {
    let response = router
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!(
                    "/intelligence/research?training_handoff={identity}"
                ))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    assert_eq!(
        response.headers()[axum::http::header::CACHE_CONTROL],
        "no-store"
    );
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    (status, serde_json::from_slice(&body).unwrap())
}

fn research_fixture_fingerprints(
    root: &Path,
) -> std::collections::BTreeMap<std::path::PathBuf, [u8; 32]> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut files = std::collections::BTreeMap::new();
    let mut pending = vec![
        root.join("models"),
        root.join("data"),
        root.join("config.yaml"),
    ];
    while let Some(path) = pending.pop() {
        let metadata = std::fs::symlink_metadata(&path).unwrap();
        assert!(
            !metadata.file_type().is_symlink(),
            "no external fixture links"
        );
        if metadata.is_dir() {
            pending.extend(
                std::fs::read_dir(path)
                    .unwrap()
                    .map(|entry| entry.unwrap().path()),
            );
        } else {
            assert!(metadata.is_file());
            let mut file = std::fs::File::open(&path).unwrap();
            let mut hash = Sha256::new();
            let mut buffer = [0u8; 32 * 1024];
            loop {
                let count = file.read(&mut buffer).unwrap();
                if count == 0 {
                    break;
                }
                hash.update(&buffer[..count]);
            }
            files.insert(
                path.strip_prefix(root).unwrap().to_owned(),
                hash.finalize().into(),
            );
        }
    }
    files
}
