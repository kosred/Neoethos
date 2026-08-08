//! Safety regression: an `action_id` that carries URL metacharacters must be
//! refused BEFORE any request is built, so the unguarded `reject_pending_action`
//! tool can never be steered onto the guarded `/actions/{id}/confirm` route.
//!
//! The Backend points at a port nothing listens on. If validation did NOT
//! short-circuit, the call would reach the transport layer and fail with the
//! "not reachable" message; instead it must fail with the charset-rejection
//! message and touch no socket.

use neoethos_mcp::backend::Backend;
use neoethos_mcp::params::{ConfirmPendingActionParams, RejectPendingActionParams};

fn dead_backend() -> Backend {
    // 127.0.0.1:1 — loopback so construction is allowed, port 1 so any real
    // request refuses fast. Validation must win before that.
    Backend::new("http://127.0.0.1:1", None).expect("loopback backend builds")
}

const MALICIOUS: &[&str] = &[
    "realId/confirm?x=",   // the exact reject→confirm path-injection payload
    "realId/confirm",      // bare traversal onto the guarded route
    "id with spaces",      // whitespace
    "id%2Fconfirm",        // percent-encoded slash
    "id#frag",             // fragment
    "id?q=1",              // query
    "../../confirm",       // dot-dot traversal
];

#[tokio::test]
async fn reject_refuses_action_ids_that_could_redirect_the_route() {
    let backend = dead_backend();
    for bad in MALICIOUS {
        let err = backend
            .op_reject_pending_action(RejectPendingActionParams {
                action_id: (*bad).to_string(),
                reason: None,
            })
            .await
            .expect_err("a metacharacter-bearing action_id must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("must contain only"),
            "reject with action_id {bad:?} should fail on charset validation, got: {msg}"
        );
        assert!(
            !msg.contains("not reachable"),
            "validation must short-circuit before any request for {bad:?}, got: {msg}"
        );
    }
}

#[tokio::test]
async fn confirm_refuses_action_ids_that_could_redirect_the_route() {
    let backend = dead_backend();
    for bad in MALICIOUS {
        let err = backend
            .op_confirm_pending_action(ConfirmPendingActionParams {
                action_id: (*bad).to_string(),
                volume_units_override: None,
            })
            .await
            .expect_err("a metacharacter-bearing action_id must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("must contain only"),
            "confirm with action_id {bad:?} should fail on charset validation, got: {msg}"
        );
        // Must refuse before ensure_demo() even runs — i.e. before any socket.
        assert!(
            !msg.contains("not reachable"),
            "validation must short-circuit before the demo probe for {bad:?}, got: {msg}"
        );
    }
}

#[tokio::test]
async fn a_clean_action_id_passes_validation_and_only_then_hits_transport() {
    let backend = dead_backend();
    // A well-formed id survives validation, so the failure now comes from the
    // dead port — proving the clean path is NOT over-rejected.
    let err = backend
        .op_reject_pending_action(RejectPendingActionParams {
            action_id: "pa_12345-abc".to_string(),
            reason: Some("not now".to_string()),
        })
        .await
        .expect_err("port 1 refuses the connection");
    let msg = err.to_string();
    assert!(
        !msg.contains("must contain only"),
        "a clean action_id must pass charset validation, got: {msg}"
    );
    assert!(
        msg.contains("not reachable"),
        "a clean action_id should reach transport and fail there, got: {msg}"
    );
}
