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

/// Local MCP sidecar base URL (overridable via `NEOETHOS_MCP_URL`).
fn sidecar_url() -> String {
    std::env::var("NEOETHOS_MCP_URL").unwrap_or_else(|_| "http://127.0.0.1:7431".to_string())
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
            "note": "MCP sidecar not running. It starts with the app when \
                     neoethos-mcp.exe is installed next to it; changes to \
                     mcp_servers.json apply on the next app start.",
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
