//! NeoEthos MCP sidecar — bridges the app's Codex/Supervisor to MCP tools.
//!
//! Reads a small JSON config listing MCP servers (cTrader remote over HTTP,
//! filesystem/web over a spawned command), connects to each with the official
//! `rmcp` SDK, and exposes ONE local HTTP API the app calls:
//!   GET  /health        → which servers connected
//!   GET  /tools         → every tool across all servers
//!   POST /call {server,tool,args} → invoke a tool, return its result
//!
//! ISOLATION: separate process + own Cargo.lock. `rmcp`'s tree never touches
//! the trading engine's pinned stack; the app talks to this over localhost.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use rmcp::ServiceExt;
use rmcp::model::CallToolRequestParams;
use rmcp::service::{Peer, RoleClient};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
struct ServerCfg {
    name: String,
    /// "http" (remote, needs `url`) or "stdio" (spawn `command` + `args`).
    transport: String,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct Config {
    #[serde(default)]
    servers: Vec<ServerCfg>,
    /// Local HTTP bind port the app calls (default 7431).
    #[serde(default)]
    port: Option<u16>,
}

/// Connected servers, name → transport-erased MCP peer.
#[derive(Clone)]
struct AppState {
    peers: Arc<HashMap<String, Peer<RoleClient>>>,
}

fn config_path() -> PathBuf {
    std::env::args()
        .skip(1)
        .zip(std::env::args().skip(2))
        .find(|(f, _)| f == "--config")
        .map(|(_, v)| PathBuf::from(v))
        .unwrap_or_else(|| PathBuf::from("mcp_servers.json"))
}

async fn connect(cfg: &ServerCfg) -> Result<Peer<RoleClient>> {
    match cfg.transport.as_str() {
        "http" => {
            let url = cfg.url.clone().context("http server needs a url")?;
            let transport =
                rmcp::transport::StreamableHttpClientTransport::from_uri(url);
            let service = ().serve(transport).await.context("serve http client")?;
            Ok(service.peer().clone())
        }
        "stdio" => {
            let command = cfg.command.clone().context("stdio server needs a command")?;
            let mut cmd = tokio::process::Command::new(command);
            cmd.args(&cfg.args);
            let transport = rmcp::transport::TokioChildProcess::new(cmd)
                .context("spawn stdio server")?;
            let service = ().serve(transport).await.context("serve stdio client")?;
            Ok(service.peer().clone())
        }
        other => anyhow::bail!("unknown transport '{other}'"),
    }
}

async fn health(State(st): State<AppState>) -> Json<serde_json::Value> {
    let names: Vec<&String> = st.peers.keys().collect();
    Json(serde_json::json!({ "servers": names, "count": names.len() }))
}

async fn list_tools(State(st): State<AppState>) -> Json<serde_json::Value> {
    let mut out = Vec::new();
    for (name, peer) in st.peers.iter() {
        match peer.list_tools(Default::default()).await {
            Ok(res) => {
                for t in res.tools {
                    out.push(serde_json::json!({
                        "server": name,
                        "name": t.name,
                        "description": t.description,
                    }));
                }
            }
            Err(e) => tracing::warn!(%name, "list_tools failed: {e}"),
        }
    }
    Json(serde_json::json!({ "tools": out }))
}

#[derive(Deserialize)]
struct CallBody {
    server: String,
    tool: String,
    #[serde(default)]
    args: serde_json::Value,
}

#[derive(Serialize)]
struct CallResult {
    ok: bool,
    result: serde_json::Value,
    error: Option<String>,
}

async fn call_tool(State(st): State<AppState>, Json(body): Json<CallBody>) -> Json<CallResult> {
    let Some(peer) = st.peers.get(&body.server) else {
        return Json(CallResult { ok: false, result: serde_json::Value::Null, error: Some(format!("no server '{}'", body.server)) });
    };
    // CallToolRequestParams is #[non_exhaustive] — build it via serde instead
    // of a struct literal.
    let param: CallToolRequestParams = match serde_json::from_value(serde_json::json!({
        "name": body.tool,
        "arguments": body.args.as_object().cloned(),
    })) {
        Ok(p) => p,
        Err(e) => {
            return Json(CallResult { ok: false, result: serde_json::Value::Null, error: Some(format!("bad params: {e}")) });
        }
    };
    match peer.call_tool(param).await {
        Ok(res) => Json(CallResult {
            ok: true,
            result: serde_json::to_value(res).unwrap_or(serde_json::Value::Null),
            error: None,
        }),
        Err(e) => Json(CallResult { ok: false, result: serde_json::Value::Null, error: Some(e.to_string()) }),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "neoethos_mcp=info,rmcp=warn".into()),
        )
        .init();

    let path = config_path();
    let cfg: Config = match std::fs::read_to_string(&path) {
        Ok(s) => serde_json::from_str(&s).context("parse mcp config")?,
        Err(_) => {
            tracing::warn!("no MCP config at {} — starting with no servers", path.display());
            Config { servers: Vec::new(), port: None }
        }
    };

    let mut peers = HashMap::new();
    for s in &cfg.servers {
        match connect(s).await {
            Ok(peer) => {
                tracing::info!(name = %s.name, transport = %s.transport, "connected to MCP server");
                peers.insert(s.name.clone(), peer);
            }
            Err(e) => tracing::warn!(name = %s.name, "MCP connect failed (skipping): {e:#}"),
        }
    }

    let state = AppState { peers: Arc::new(peers) };
    let port = cfg.port.unwrap_or(7431);
    let app = Router::new()
        .route("/health", get(health))
        .route("/tools", get(list_tools))
        .route("/call", post(call_tool))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await
        .with_context(|| format!("bind 127.0.0.1:{port}"))?;
    tracing::info!("NeoEthos MCP sidecar on http://127.0.0.1:{port} — /health /tools /call");
    axum::serve(listener, app).await.context("serve")?;
    Ok(())
}
