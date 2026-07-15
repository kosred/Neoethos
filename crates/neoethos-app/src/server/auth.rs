//! Per-launch bearer-token authentication for the local API (audit S01).
//!
//! The loopback API previously had NO authentication: its security rested
//! entirely on binding to `127.0.0.1`. Combined with the (now fixed) permissive
//! CORS, any web page could drive the trading endpoints through the operator's
//! browser; and binding to a non-loopback interface exposed the unauthenticated
//! trade-capable API to the whole network.
//!
//! This module adds a bearer token. Enforcement is ACTIVE only when a token is
//! installed for the launch:
//!   * `serve()` REFUSES to bind a non-loopback address unless the operator
//!     enabled auth via `NEOETHOS_API_TOKEN` (fail-closed — closes the network
//!     exposure hole decisively).
//!   * an operator-supplied `NEOETHOS_API_TOKEN` activates enforcement in every
//!     mode (headless or desktop).
//!   * the default loopback bind installs NO token, so the existing desktop UI
//!     (which does not yet send a bearer) keeps working; browser cross-origin
//!     access is already blocked by the CORS allowlist.
//!
//! The webview-side change that keeps the token out of JavaScript (proxying
//! every UI call through a Tauri command) is a separate, larger follow-up.

use axum::body::Body;
use axum::extract::Request;
use axum::http::{StatusCode, header::AUTHORIZATION};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use std::sync::RwLock;

/// The token enforced for this launch. `None` = permissive (no auth).
static API_TOKEN: RwLock<Option<String>> = RwLock::new(None);

/// Install the token to enforce for this launch. Called once at startup.
pub fn set_api_token(token: String) {
    *API_TOKEN.write().unwrap_or_else(|e| e.into_inner()) = Some(token);
}

/// The token currently enforced, if any.
pub fn current_api_token() -> Option<String> {
    API_TOKEN.read().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Generate a fresh 256-bit token (hex-encoded, 64 chars), never logged.
pub fn generate_token() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    hex::encode(bytes)
}

/// Operator-supplied token via `NEOETHOS_API_TOKEN` (any non-empty value),
/// trimmed. `None` when unset or blank.
pub fn configured_token() -> Option<String> {
    std::env::var("NEOETHOS_API_TOKEN")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Constant-time string comparison so a wrong token can't be recovered by
/// timing the response.
fn constant_time_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Middleware: when a token is installed, require `Authorization: Bearer
/// <token>` on every route except `/healthz`; otherwise allow (the permissive
/// loopback default). `/healthz` is always open so liveness probes work.
pub async fn require_token(req: Request<Body>, next: Next) -> Response {
    if req.uri().path() == "/healthz" {
        return next.run(req).await;
    }
    let Some(expected) = current_api_token() else {
        return next.run(req).await; // no token installed → permissive
    };
    let presented = req
        .headers()
        .get(AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(str::trim);
    match presented {
        Some(tok) if constant_time_eq(tok, &expected) => next.run(req).await,
        _ => (
            StatusCode::UNAUTHORIZED,
            "missing or invalid API token (Authorization: Bearer <token>)",
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_tokens_are_256_bit_hex_and_unique() {
        let a = generate_token();
        let b = generate_token();
        assert_eq!(a.len(), 64, "32 bytes hex = 64 chars");
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b, "each launch token must differ");
    }

    #[test]
    fn constant_time_eq_matches_only_identical_strings() {
        let t = generate_token();
        assert!(constant_time_eq(&t, &t));
        assert!(!constant_time_eq(&t, &generate_token()));
        assert!(!constant_time_eq("abc", "abcd")); // length mismatch
        assert!(!constant_time_eq("abc", "abd"));
    }

    #[test]
    fn set_and_current_token_round_trip() {
        set_api_token("unit-test-token".to_string());
        assert_eq!(current_api_token().as_deref(), Some("unit-test-token"));
        // Restore permissive default so this doesn't leak into other tests
        // that share the process global.
        *API_TOKEN.write().unwrap() = None;
    }
}
