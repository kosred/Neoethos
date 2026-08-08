//! Tier-1 test 6.1.3 (design doc): demo-guard behavior against a mock axum
//! backend on an ephemeral port. Covers, exhaustively and off the real
//! system: the Live refusal (exact message), the fail-closed refusals, the
//! backend-down message, and the happy path with `risky:false` and the
//! lots→wire-unit conversion asserted by the mock.

use std::sync::{Arc, Mutex};

use axum::Json;
use axum::extract::State;
use axum::routing::{get, post};
use serde_json::{Value, json};

use neoethos_mcp::backend::Backend;
use neoethos_mcp::params::*;

/// What the mock backend serves + what it captured.
#[derive(Clone)]
struct MockState {
    environment: &'static str,
    connected: bool,
    /// `Value::Null` models the cTrader "unknown" (is_live absent).
    is_live: Value,
    /// Bodies captured by the trade-route handlers, keyed by route.
    captured: Arc<Mutex<Vec<(String, Value)>>>,
}

impl MockState {
    fn demo() -> Self {
        Self {
            environment: "Demo",
            connected: true,
            is_live: json!(false),
            captured: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

fn mock_router(state: MockState) -> axum::Router {
    async fn broker_status(State(s): State<MockState>) -> Json<Value> {
        Json(json!({
            "adapter": "cTrader",
            "environment": s.environment,
            "accountId": "12345",
            "connected": s.connected,
            "clientIdPrefix": "abc_…",
        }))
    }
    async fn broker_accounts(State(s): State<MockState>) -> Json<Value> {
        Json(json!({
            "environment": s.environment,
            "permissionScope": "trader",
            "accountCount": 1,
            "accounts": [{
                "accountId": "12345",
                "brokerTitle": "MockBroker",
                "accountName": "mock",
                "traderLogin": 1,
                "isLive": s.is_live,
                "enabledForExecution": true,
            }],
        }))
    }
    async fn account_snapshot(State(_s): State<MockState>) -> Json<Value> {
        Json(json!({
            "balance": 10000.0,
            "equity": 10000.0,
            "freeMargin": 9900.0,
            "usedMargin": 100.0,
            "currency": "EUR",
            "fetchedAtUnixMs": 0,
            "positions": [{
                "positionId": 42,
                "volumeUnits": 1_000_000,
                "symbol": "EURUSD",
                "side": "BUY",
                "volume": 0.10,
                "openTimestampMs": 0,
                "pnlPips": 0.0,
                "pnlUsd": 0.0,
                "entryPrice": 1.1,
                "stopLoss": 1.09,
                "takeProfit": 1.12,
                "volumeLots": 0.10,
            }],
        }))
    }
    fn capture(
        route: &'static str,
    ) -> impl Fn(
        State<MockState>,
        Json<Value>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Json<Value>> + Send>>
    + Clone {
        move |State(s): State<MockState>, Json(body): Json<Value>| {
            Box::pin(async move {
                s.captured
                    .lock()
                    .expect("captured lock")
                    .push((route.to_string(), body));
                Json(json!({ "status": "Filled", "message": "mock accepted" }))
            })
        }
    }
    axum::Router::new()
        .route("/broker/status", get(broker_status))
        .route("/broker/accounts", get(broker_accounts))
        .route("/account/snapshot", get(account_snapshot))
        .route("/orders", post(capture("/orders")))
        .route("/positions/close", post(capture("/positions/close")))
        .route("/autonomous/stop", post(capture("/autonomous/stop")))
        .with_state(state)
}

/// Bind the mock on an ephemeral loopback port and serve it in the
/// background; returns the base URL.
async fn spawn_mock(state: MockState) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    let app = mock_router(state);
    tokio::spawn(async move {
        // The server lives for the whole test process; an error here would
        // surface as request failures in the assertions below.
        let _ = axum::serve(listener, app).await;
    });
    format!("http://{addr}")
}

fn open_params() -> OpenPositionParams {
    serde_json::from_value(json!({
        "symbol": "EURUSD",
        "side": "buy",
        "volume_lots": 0.01,
        "stop_loss_pips": 20.0,
        "take_profit_pips": 20.0,
    }))
    .expect("valid params")
}

// ── Live environment → refused with the EXACT design message ──────────────

#[tokio::test]
async fn live_environment_is_refused_with_the_exact_message() {
    let state = MockState {
        environment: "Live",
        is_live: json!(true),
        ..MockState::demo()
    };
    let base = spawn_mock(state.clone()).await;
    let backend = Backend::new(&base, None).expect("backend");

    let err = backend
        .op_open_position(open_params())
        .await
        .expect_err("a Live environment must refuse");
    assert_eq!(
        err.0,
        "DEMO GUARD REFUSED: the active broker environment is 'Live' (account 12345). This \
         control plane operates demo accounts only — no trade-affecting tool will execute \
         against a live environment, ever. Switch the active account to Demo in the desktop \
         app, then retry."
    );
    assert!(
        state.captured.lock().expect("lock").is_empty(),
        "no trade route may be touched when the guard refuses"
    );
}

// ── Fail-closed refusals ──────────────────────────────────────────────────

#[tokio::test]
async fn unknown_is_live_fails_closed() {
    let state = MockState {
        is_live: Value::Null, // broker did not say — absence of proof = live
        ..MockState::demo()
    };
    let base = spawn_mock(state.clone()).await;
    let backend = Backend::new(&base, None).expect("backend");

    let err = backend
        .op_open_position(open_params())
        .await
        .expect_err("unknown is_live must refuse (fail-closed)");
    assert_eq!(
        err.0,
        "DEMO GUARD REFUSED (fail-closed): cannot positively verify the active account is a \
         demo account (broker connected=true, environment=Demo, is_live=None). \
         Trade-affecting tools stay locked until /broker/status reports connected=true + \
         environment=Demo and /broker/accounts reports is_live=false for the active account."
    );
    assert!(state.captured.lock().expect("lock").is_empty());
}

#[tokio::test]
async fn disconnected_broker_fails_closed() {
    let state = MockState {
        connected: false,
        ..MockState::demo()
    };
    let base = spawn_mock(state.clone()).await;
    let backend = Backend::new(&base, None).expect("backend");

    let err = backend
        .op_open_position(open_params())
        .await
        .expect_err("a disconnected broker must refuse");
    assert!(
        err.0.starts_with("DEMO GUARD REFUSED (fail-closed):"),
        "unexpected message: {}",
        err.0
    );
    assert!(err.0.contains("connected=false"), "message: {}", err.0);
    assert!(state.captured.lock().expect("lock").is_empty());
}

#[tokio::test]
async fn is_live_true_on_a_demo_environment_fails_closed() {
    // Contradictory state (env says Demo, account says live) must refuse.
    let state = MockState {
        is_live: json!(true),
        ..MockState::demo()
    };
    let base = spawn_mock(state.clone()).await;
    let backend = Backend::new(&base, None).expect("backend");

    let err = backend
        .op_open_position(open_params())
        .await
        .expect_err("is_live=true must refuse");
    assert!(err.0.contains("is_live=Some(true)"), "message: {}", err.0);
    assert!(state.captured.lock().expect("lock").is_empty());
}

// ── Backend down → the fixed unreachable message ──────────────────────────

#[tokio::test]
async fn backend_down_returns_the_unreachable_message() {
    // Bind then immediately drop a listener: the port is real but nothing
    // serves it — connection refused.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    drop(listener);
    let base = format!("http://{addr}");
    let backend = Backend::new(&base, None).expect("backend");

    let expected_core = format!(
        "NeoEthos backend is not reachable at {base} (connection refused). Start the desktop \
         app or run 'neoethos-app --server', then retry. This control plane never falls back \
         to acting on its own."
    );

    // A read tool carries the message directly…
    let err = backend.op_engine_status().await.expect_err("must fail");
    assert!(
        err.0.starts_with(&expected_core),
        "read-path message drifted: {}",
        err.0
    );

    // …and a trade tool fails closed at the guard, still carrying it.
    let err = backend
        .op_open_position(open_params())
        .await
        .expect_err("must fail");
    assert!(
        err.0.starts_with("DEMO GUARD REFUSED (fail-closed):") && err.0.contains(&expected_core),
        "guard-path message drifted: {}",
        err.0
    );
}

// ── Happy path: forwarded with risky:false + wire-unit conversion ─────────

#[tokio::test]
async fn open_position_forwards_with_risky_false_and_required_stop_loss() {
    let state = MockState::demo();
    let base = spawn_mock(state.clone()).await;
    let backend = Backend::new(&base, None).expect("backend");

    let result = backend
        .op_open_position(open_params())
        .await
        .expect("must forward");
    assert_eq!(result["status"], "Filled");

    let captured = state.captured.lock().expect("lock").clone();
    assert_eq!(captured.len(), 1, "exactly one order must reach the broker");
    let (route, body) = &captured[0];
    assert_eq!(route, "/orders");
    assert_eq!(body["risky"], json!(false), "risky must be hardcoded false");
    assert_eq!(body["symbol"], "EURUSD");
    assert_eq!(body["side"], "buy");
    assert_eq!(body["volumeLots"], json!(0.01));
    assert_eq!(
        body["stopLossPips"],
        json!(20.0),
        "SL is schema-required and must be sent"
    );
    assert_eq!(body["takeProfitPips"], json!(20.0));
}

#[tokio::test]
async fn close_position_converts_lots_to_wire_units() {
    let state = MockState::demo();
    let base = spawn_mock(state.clone()).await;
    let backend = Backend::new(&base, None).expect("backend");

    // Partial close: 0.05 of a 0.10-lot position with 1_000_000 wire units
    // must convert to exactly 500_000 wire units.
    backend
        .op_close_position(
            serde_json::from_value(json!({ "position_id": 42, "volume_lots": 0.05 }))
                .expect("params"),
        )
        .await
        .expect("partial close forwards");

    // Full close: omitted volume_lots closes the whole snapshot volume.
    backend
        .op_close_position(serde_json::from_value(json!({ "position_id": 42 })).expect("params"))
        .await
        .expect("full close forwards");

    let captured = state.captured.lock().expect("lock").clone();
    assert_eq!(captured.len(), 2);
    assert_eq!(captured[0].0, "/positions/close");
    assert_eq!(captured[0].1["positionId"], json!(42));
    assert_eq!(
        captured[0].1["volume"],
        json!(500_000),
        "0.05 lots → 500_000 wire units"
    );
    assert_eq!(
        captured[1].1["volume"],
        json!(1_000_000),
        "full close → full wire volume"
    );
}

#[tokio::test]
async fn close_position_refuses_more_than_the_open_volume() {
    let state = MockState::demo();
    let base = spawn_mock(state.clone()).await;
    let backend = Backend::new(&base, None).expect("backend");

    let err = backend
        .op_close_position(
            serde_json::from_value(json!({ "position_id": 42, "volume_lots": 0.20 }))
                .expect("params"),
        )
        .await
        .expect_err("over-close must refuse");
    assert!(err.0.contains("exceeds"), "message: {}", err.0);
    assert!(
        state.captured.lock().expect("lock").is_empty(),
        "the broker must not see an over-sized close"
    );
}

// ── The one deliberate guard exemption ────────────────────────────────────

#[tokio::test]
async fn autopilot_stop_works_even_when_the_guard_would_refuse() {
    // Live environment: every trade tool refuses — but stop must still work
    // (halts engines, no broker order; design §2).
    let state = MockState {
        environment: "Live",
        is_live: json!(true),
        ..MockState::demo()
    };
    let base = spawn_mock(state.clone()).await;
    let backend = Backend::new(&base, None).expect("backend");

    backend
        .op_autopilot_stop()
        .await
        .expect("autopilot_stop is guard-exempt by design");
    let captured = state.captured.lock().expect("lock").clone();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].0, "/autonomous/stop");
}
