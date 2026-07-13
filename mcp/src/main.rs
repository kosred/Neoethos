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

use std::path::PathBuf;

use anyhow::{Context, Result};
use neoethos_mcp::{AppState, Config, router};

fn config_path() -> PathBuf {
    std::env::args()
        .skip(1)
        .zip(std::env::args().skip(2))
        .find(|(f, _)| f == "--config")
        .map(|(_, v)| PathBuf::from(v))
        .unwrap_or_else(|| PathBuf::from("mcp_servers.json"))
}

async fn serve_api(state: AppState, port: u16) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
        .await
        .with_context(|| format!("bind 127.0.0.1:{port}"))?;
    tracing::info!(
        "NeoEthos MCP sidecar on http://127.0.0.1:{port} — /health /tools /call"
    );
    axum::serve(listener, router(state))
        .with_graceful_shutdown(async {
            match tokio::signal::ctrl_c().await {
                Ok(()) => tracing::info!("shutdown signal received"),
                Err(error) => tracing::error!(%error, "failed to listen for shutdown signal"),
            }
        })
        .await
        .context("serve")
}

#[tokio::main]
async fn main() -> Result<()> {
    let env_filter = if std::env::var_os("RUST_LOG").is_some() {
        tracing_subscriber::EnvFilter::try_from_default_env().context("parse RUST_LOG")?
    } else {
        tracing_subscriber::EnvFilter::new("neoethos_mcp=info,rmcp=warn")
    };
    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .init();

    let path = config_path();
    let cfg: Config = match std::fs::read_to_string(&path) {
        Ok(s) => serde_json::from_str(&s).context("parse mcp config")?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            tracing::warn!("no MCP config at {} — starting with no servers", path.display());
            Config::default()
        }
        Err(error) => {
            return Err(error).with_context(|| format!("read MCP config {}", path.display()));
        }
    };

    let state = AppState::connect_configured(&cfg.servers).await;
    let port = cfg.port.unwrap_or(7431);
    let serve_result = serve_api(state.clone(), port).await;
    let shutdown_result = state.shutdown_all().await;

    serve_result?;
    shutdown_result.context("shut down MCP services")
}
