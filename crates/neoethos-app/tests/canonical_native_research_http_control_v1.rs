use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use neoethos_app::server::router;
use neoethos_app::server::state::AppApiState;
use serde_json::{Value, json};
use tower::ServiceExt;

const LOWER_HEX_SHA256: &str = "0000000000000000000000000000000000000000000000000000000000000000";

async fn response_json(response: axum::response::Response) -> Value {
    let bytes = to_bytes(response.into_body(), 64 * 1024)
        .await
        .expect("bounded response body");
    serde_json::from_slice(&bytes).expect("JSON response")
}

#[tokio::test]
async fn native_start_route_is_fail_closed_without_startup_authority() {
    let response = router(AppApiState::new())
        .oneshot(
            Request::post("/engines/native-research/start")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&json!({
                        "contractArtifact": {
                            "relativePath": "contracts/research.json",
                            "expectedSha256": LOWER_HEX_SHA256,
                        },
                        "population": 200,
                        "populationAuto": true,
                        "maxIndicators": 0,
                    }))
                    .expect("request JSON"),
                ))
                .expect("request"),
        )
        .await
        .expect("router response");
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        response_json(response).await,
        json!({
            "started": false,
            "kind": "canonical_native_research",
            "errorCode": "native_runtime_authority_unavailable",
            "detail": "canonical native startup authority is not installed",
            "requestedKind": null,
            "activeKind": null,
        })
    );
}

#[tokio::test]
async fn native_cancel_route_requires_an_exact_active_lease_token() {
    let response = router(AppApiState::new())
        .oneshot(
            Request::post("/engines/native-research/cancel")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"leaseToken":"42"}"#))
                .expect("request"),
        )
        .await
        .expect("router response");
    assert_eq!(response.status(), StatusCode::CONFLICT);
    assert_eq!(
        response_json(response).await,
        json!({
            "cancellationRequested": false,
            "kind": "canonical_native_research",
            "leaseToken": "42",
            "state": "Idle",
            "errorCode": "native_research_not_running",
        })
    );
}

#[tokio::test]
async fn native_cancel_rejects_non_decimal_or_zero_tokens_before_state_access() {
    for token in ["0", "-1", "18446744073709551616", "not-a-token"] {
        let response = router(AppApiState::new())
            .oneshot(
                Request::post("/engines/native-research/cancel")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({ "leaseToken": token })).expect("request JSON"),
                    ))
                    .expect("request"),
            )
            .await
            .expect("router response");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response_json(response).await,
            json!({
                "cancellationRequested": false,
                "kind": "canonical_native_research",
                "leaseToken": token,
                "state": "Invalid",
                "errorCode": "invalid_native_research_lease_token",
            })
        );
    }
}

#[test]
fn route_source_registers_only_the_native_lane_and_existing_status_path() {
    let source = include_str!("../src/server/mod.rs");
    assert!(source.contains(
        ".route(\n            \"/engines/native-research/start\",\n            post(engines_control::canonical_native_research_start),\n        )"
    ));
    assert!(source.contains(
        ".route(\n            \"/engines/native-research/cancel\",\n            post(engines_control::canonical_native_research_cancel),\n        )"
    ));
    assert!(source.contains(".route(\"/engines/status\", get(system_status::engines))"));
    assert!(!source.contains("/system/status"));
}
