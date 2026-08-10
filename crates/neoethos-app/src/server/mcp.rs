//! MCP sidecar management endpoints (task #33 — full MCP support).
//!
//! The isolated `neoethos-mcp` sidecar (own workspace, rmcp client)
//! connects to configured MCP servers (cTrader remote, MT5, filesystem,
//! web search, …) and exposes their tools on `127.0.0.1:7431`
//! (`/health`, `/tools`, `/call`). The Supervisor already consumes it
//! (mcp_tools / mcp_call actions, approval-gated). These endpoints give
//! the UI what it needs to manage the setup without hand-editing files:
//!
//! - `GET  /mcp/config` — current `mcp_servers.json` (or a starter
//!   template when none exists yet).
//! - `PUT  /mcp/config` — validate + write `mcp_servers.json`. The
//!   sidecar reads its config at startup, so changes apply on the next
//!   app restart (stated in the response).
//! - `GET  /mcp/status` — proxy the sidecar's `/health` + `/tools` so
//!   the UI can show connected servers + available tools.
//!
//! SECURITY: tool CALLS are not exposed here. Tools are invoked only
//! through the Supervisor's action framework, where trade-affecting
//! actions require operator approval — a third-party MCP server (e.g. an
//! MT5 bridge) must never place orders without a human click.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

const CONFIG_FILE: &str = "mcp_servers.json";

/// Starter template shown when no config exists yet. Mirrors
/// `mcp/mcp_servers.example.json` (kept small — the UI links the docs).
const TEMPLATE: &str = r#"{
  "port": 7431,
  "servers": [
    {
      "name": "ctrader",
      "transport": "http",
      "url": "https://mcp.spotware.com/mcp"
    }
  ]
}
"#;

/// The port the sidecar listens on when `mcp_servers.json` does not say.
/// Matches [`TEMPLATE`] and `mcp/mcp_servers.example.json` — those three must
/// move together.
const DEFAULT_SIDECAR_PORT: u16 = 7431;

/// Local MCP sidecar base URL — **derived from the sidecar's own config file,
/// not from the environment.**
///
/// # What changed and why (2026-08-10, config consolidation)
///
/// This used to read `NEOETHOS_MCP_URL`. Two call sites read it independently
/// (here and `app_services::supervisor`), so a typo in one shell export could
/// point the Supervisor at one process and the `/mcp/status` card at another,
/// and the UI would report a healthy sidecar the Supervisor could not reach.
///
/// There is no new config key, because there does not need to be one: the
/// sidecar's port is ALREADY stated, exactly once, in `mcp_servers.json` —
/// the file this module writes at `PUT /mcp/config` and the file the sidecar
/// itself reads at startup. Reading the port from there means the two
/// processes cannot disagree about it: there is one number, in one file, that
/// both of them obey.
///
/// The sidecar is always local (it is spawned next to the app), so the host
/// is fixed at `127.0.0.1`. Making it configurable would re-open the same
/// two-places-one-fact hole for no operator benefit.
///
/// If `NEOETHOS_MCP_URL` is still exported, `app_services::retired_env` says
/// so by name at startup. Nothing here reads it.
pub(crate) fn sidecar_url() -> String {
    format!("http://127.0.0.1:{}", sidecar_port())
}

/// The `port` field of `mcp_servers.json`, or [`DEFAULT_SIDECAR_PORT`].
///
/// A missing FILE is the ordinary "no MCP configured yet" state and is not
/// worth a warning. A file that EXISTS but whose `port` is absent, not a
/// number, or out of range is a disagreement between what the operator wrote
/// and what this process will dial — non-negotiable #3 says that substitution
/// gets named, with both values.
fn sidecar_port() -> u16 {
    let Ok(raw) = std::fs::read_to_string(CONFIG_FILE) else {
        return DEFAULT_SIDECAR_PORT;
    };
    let parsed: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(err) => {
            tracing::warn!(
                target: "neoethos_app::server::mcp",
                path = CONFIG_FILE,
                error = %err,
                fallback_port = DEFAULT_SIDECAR_PORT,
                "mcp_servers.json exists but is not valid JSON — dialling the \
                 sidecar on the default port instead. The sidecar reads the same \
                 file, so it may not be listening there."
            );
            return DEFAULT_SIDECAR_PORT;
        }
    };
    match parsed.get("port") {
        None => DEFAULT_SIDECAR_PORT,
        Some(v) => match v.as_u64().and_then(|n| u16::try_from(n).ok()).filter(|n| *n > 0) {
            Some(port) => port,
            None => {
                tracing::warn!(
                    target: "neoethos_app::server::mcp",
                    path = CONFIG_FILE,
                    configured_port = %v,
                    fallback_port = DEFAULT_SIDECAR_PORT,
                    "mcp_servers.json `port` is not a valid TCP port — dialling \
                     the default instead. Fix the file (Settings → MCP) so this \
                     process and the sidecar agree on one number."
                );
                DEFAULT_SIDECAR_PORT
            }
        },
    }
}

// ─── GET /mcp/config ───────────────────────────────────────────────────────

pub async fn config_get() -> Json<serde_json::Value> {
    match std::fs::read_to_string(CONFIG_FILE) {
        Ok(content) => Json(serde_json::json!({
            "exists": true,
            "path": CONFIG_FILE,
            "content": content,
        })),
        Err(_) => Json(serde_json::json!({
            "exists": false,
            "path": CONFIG_FILE,
            "content": TEMPLATE,
        })),
    }
}

// ─── PUT /mcp/config ───────────────────────────────────────────────────────

#[derive(Debug, serde::Deserialize)]
pub struct McpConfigBody {
    pub content: String,
}

pub async fn config_put(Json(body): Json<McpConfigBody>) -> Response {
    // Validate it is JSON with the shape the sidecar expects (an object
    // with a `servers` array) BEFORE writing — a syntax error here would
    // otherwise brick the sidecar silently on the next start.
    let parsed: serde_json::Value = match serde_json::from_str(&body.content) {
        Ok(v) => v,
        Err(err) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("not valid JSON: {err}"),
                })),
            )
                .into_response();
        }
    };
    if !parsed
        .get("servers")
        .map(|s| s.is_array())
        .unwrap_or(false)
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "config must be an object with a `servers` array \
                          (see mcp/mcp_servers.example.json)",
            })),
        )
            .into_response();
    }
    if let Err(err) = std::fs::write(CONFIG_FILE, &body.content) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("write {CONFIG_FILE} failed: {err}"),
            })),
        )
            .into_response();
    }
    Json(serde_json::json!({
        "saved": true,
        "path": CONFIG_FILE,
        "note": "The MCP sidecar reads its config at startup — restart the app to apply.",
    }))
    .into_response()
}

// ─── GET /mcp/status ───────────────────────────────────────────────────────

pub async fn status() -> Json<serde_json::Value> {
    let base = sidecar_url();
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
    {
        Ok(c) => c,
        Err(err) => {
            return Json(serde_json::json!({
                "reachable": false,
                "error": format!("http client: {err}"),
            }));
        }
    };
    let health: Option<serde_json::Value> = match client.get(format!("{base}/health")).send().await
    {
        Ok(r) => r.json().await.ok(),
        Err(_) => None,
    };
    let Some(health) = health else {
        return Json(serde_json::json!({
            "reachable": false,
            "url": base,
            // W10b (2026-08-10): name BOTH binaries. `neoethos-mcp` is the
            // outbound sidecar from the `mcp/` workspace; `neoethos-control-plane`
            // is the inbound control plane from `crates/neoethos-mcp`. Until
            // today both crates produced an executable called `neoethos-mcp`,
            // so an operator could verify the wrong process and believe this
            // message was wrong.
            "note": "MCP sidecar not running. The process wanted is `neoethos-mcp` \
                     (the OUTBOUND sidecar from the `mcp/` workspace), installed \
                     next to the app — NOT `neoethos-control-plane` (the INBOUND \
                     control plane from `crates/neoethos-mcp`), which is a \
                     different program. Changes to mcp_servers.json, including \
                     its `port`, apply on the next app start.",
        }));
    };
    let tools: serde_json::Value = match client.get(format!("{base}/tools")).send().await {
        Ok(r) => r.json().await.unwrap_or(serde_json::json!({"tools": []})),
        Err(_) => serde_json::json!({"tools": []}),
    };
    Json(serde_json::json!({
        "reachable": true,
        "url": base,
        "health": health,
        "tools": tools.get("tools").cloned().unwrap_or(serde_json::json!([])),
    }))
}
