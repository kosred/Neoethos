//! GET readiness is a diagnostic read, not a promotion capability.
//! Fixtures are owned temporary files; no broker, model inference or global config mutation.
use super::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

struct Fixture {
    root: PathBuf,
    config: PathBuf,
    targets: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "neoethos-promotion-diagnostic-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::create_dir(&root).expect("fresh owned fixture directory");
        let config = root.join("config.yaml");
        let mut settings = Settings::default();
        settings.system.data_dir = root.join("data");
        settings.system.symbol = "USDJPY".to_owned();
        settings.system.base_timeframe = "H1".to_owned();
        settings.models.promotion_gate.enabled = false;
        settings.models.promotion_gate.min_sharpe = 2.5;
        settings.save(&config).unwrap();
        let targets = model_targets_path_for(&settings.system.data_dir, "USDJPY", "H1");
        Self {
            root,
            config,
            targets,
        }
    }

    async fn status(&self) -> Response {
        promotion_status_for_config(
            PromotionQuery {
                symbol: None,
                base_tf: None,
            },
            self.config.clone(),
        )
        .await
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        // Only this test's newly created, uniquely named temp directory.
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}

async fn json(response: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn assert_hold(value: &serde_json::Value) {
    assert_eq!(value["decision"]["promoted"], false);
    assert!(value["aggregate"].is_null(), "no fabricated metrics");
    assert_eq!(value["decision"]["criteria"], serde_json::json!([]));
    let summary = value["decision"]["summary"].as_str().unwrap();
    for missing in [
        "Missing candidate selection",
        "Independent final OOS evidence is unverified",
        "Exact composite",
        "ResearchOnly V2/V3 results are not permits",
        "Metric thresholds were not evaluated",
    ] {
        assert!(summary.contains(missing), "{summary}");
    }
    assert!(!summary.contains("BROKER_FINANCIAL_TRUTH_UNAVAILABLE_V1"));
}

#[tokio::test(flavor = "current_thread")]
async fn missing_targets_return_hold_with_config_defaults_and_no_writes() {
    let fixture = Fixture::new();
    let before = std::fs::read(&fixture.config).unwrap();
    let response = fixture.status().await;
    assert_eq!(response.status(), StatusCode::OK);
    let value = json(response).await;
    assert_hold(&value);
    assert_eq!(value["symbol"], "USDJPY");
    assert_eq!(value["baseTf"], "H1");
    assert_eq!(value["portfolioSize"], 0);
    assert_eq!(value["config"]["enabled"], false);
    assert_eq!(value["config"]["minSharpe"], 2.5);
    let summary = value["decision"]["summary"].as_str().unwrap();
    assert!(summary.contains("model_targets v3 is missing"), "{summary}");
    assert!(summary.contains("not proof that no research candidates exist"));
    assert_eq!(std::fs::read(&fixture.config).unwrap(), before);
    assert!(!fixture.targets.exists());
    assert!(!fixture.root.join("data").exists());
    assert!(!fixture.root.join("live_models").exists());
    assert_eq!(std::fs::read_dir(&fixture.root).unwrap().count(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn malformed_and_legacy_targets_are_hold_not_heuristic_metrics() {
    let fixture = Fixture::new();
    std::fs::create_dir_all(fixture.targets.parent().unwrap()).unwrap();
    for (bytes, expected) in [
        (b"{".as_slice(), "model_targets v3 is malformed"),
        (
            br#"{"schema_version":1,"portfolio":[{"sharpe_ratio":999.0}]}"#.as_slice(),
            "schema Some(1) cannot authorize promotion",
        ),
    ] {
        std::fs::write(&fixture.targets, bytes).unwrap();
        let response = fixture.status().await;
        assert_eq!(response.status(), StatusCode::OK);
        let value = json(response).await;
        assert_hold(&value);
        assert_eq!(value["portfolioSize"], 0);
        assert!(
            value["decision"]["summary"]
                .as_str()
                .unwrap()
                .contains(expected)
        );
        assert_eq!(std::fs::read(&fixture.targets).unwrap(), bytes);
    }
}

#[tokio::test(flavor = "current_thread")]
async fn unsafe_query_is_bad_request_and_unreadable_config_is_server_error() {
    let fixture = Fixture::new();
    let response = promotion_status_for_config(
        PromotionQuery {
            symbol: Some("../EURUSD".to_owned()),
            base_tf: Some("M5".to_owned()),
        },
        fixture.config.clone(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(!fixture.root.join("data").exists());

    let response = promotion_status_for_config(
        PromotionQuery {
            symbol: None,
            base_tf: None,
        },
        fixture.root.join("missing-config.yaml"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert!(
        !json(response)
            .await
            .to_string()
            .contains("BROKER_FINANCIAL_TRUTH_UNAVAILABLE_V1")
    );
}

#[test]
fn validated_count_cannot_enable_promotion_even_when_metric_gate_is_disabled() {
    // Presentation seam only: this is not a fixture claiming valid composite evidence.
    let config = PromotionGateConfig {
        enabled: false,
        ..PromotionGateConfig::default()
    };
    let dto =
        promotion_readiness_response("EURUSD".to_owned(), "M5".to_owned(), config.clone(), Ok(4));
    let value = serde_json::to_value(dto).unwrap();
    assert_hold(&value);
    assert_eq!(value["portfolioSize"], 4);
    assert_eq!(value["config"], serde_json::to_value(config).unwrap());
    assert!(
        value["decision"]["summary"]
            .as_str()
            .unwrap()
            .contains("diagnostic only")
    );
}

#[test]
fn exact_evidence_errors_remain_distinguishable_without_inventing_metric_failures() {
    for (error, expected) in [
        (
            PromotionAuthorizationError::RequestedIdentityMismatch {
                reason: "wrong series".into(),
            },
            "identity mismatch: wrong series",
        ),
        (
            PromotionAuthorizationError::PromotionSummaryMismatch,
            "differs from the v3 target binding",
        ),
        (
            PromotionAuthorizationError::MissingHeldOutEvidence { kind: "final_oos" },
            "evidence `final_oos` is missing",
        ),
        (
            PromotionAuthorizationError::FailedHeldOutEvidence { kind: "final_oos" },
            "evidence `final_oos` failed",
        ),
        (
            PromotionAuthorizationError::InvalidCompositeEvidenceScope,
            "lacks an exact composite",
        ),
    ] {
        let value = serde_json::to_value(promotion_readiness_response(
            "EURUSD".into(),
            "M5".into(),
            PromotionGateConfig::default(),
            Err(error),
        ))
        .unwrap();
        assert_hold(&value);
        assert!(
            value["decision"]["summary"]
                .as_str()
                .unwrap()
                .contains(expected)
        );
    }
}
