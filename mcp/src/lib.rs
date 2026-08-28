//! Runtime lifecycle and local HTTP API for the NeoEthos MCP sidecar.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use rmcp::ServiceExt;
use rmcp::model::CallToolRequestParams;
use rmcp::service::{Peer, RoleClient, RunningService};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

#[derive(Debug, Clone, Deserialize)]
pub struct ServerCfg {
    pub name: String,
    /// `http` (remote, needs `url`) or `stdio` (spawn `command` + `args`).
    pub transport: String,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    /// Optional child-process environment. This is ignored by HTTP transports.
    #[serde(default)]
    pub env: HashMap<String, String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub servers: Vec<ServerCfg>,
    /// Local HTTP bind port the app calls (default 7431).
    #[serde(default)]
    pub port: Option<u16>,
}

/// An MCP client connection whose service task remains owned until shutdown.
#[derive(Debug)]
pub struct ConnectedService {
    peer: Peer<RoleClient>,
    service: Mutex<Option<RunningService<RoleClient, ()>>>,
}

impl ConnectedService {
    fn new(service: RunningService<RoleClient, ()>) -> Self {
        Self {
            peer: service.peer().clone(),
            service: Mutex::new(Some(service)),
        }
    }

    pub fn peer(&self) -> &Peer<RoleClient> {
        &self.peer
    }

    pub async fn is_closed(&self) -> bool {
        self.service
            .lock()
            .await
            .as_ref()
            .is_none_or(RunningService::is_closed)
    }

    pub async fn shutdown(&self) -> Result<()> {
        let service = self.service.lock().await.take();
        if let Some(service) = service {
            let reason = service.cancel().await.context("cancel MCP service")?;
            tracing::debug!(?reason, "MCP service stopped");
        }
        Ok(())
    }
}

pub async fn connect(cfg: &ServerCfg) -> Result<ConnectedService> {
    let service = match cfg.transport.as_str() {
        "http" => {
            let url = cfg.url.clone().context("http server needs a url")?;
            let transport = rmcp::transport::StreamableHttpClientTransport::from_uri(url);
            ().serve(transport).await.context("serve http client")?
        }
        "stdio" => {
            let command = cfg
                .command
                .clone()
                .context("stdio server needs a command")?;
            let mut cmd = tokio::process::Command::new(command);
            cmd.args(&cfg.args).envs(&cfg.env);
            let transport =
                rmcp::transport::TokioChildProcess::new(cmd).context("spawn stdio server")?;
            ().serve(transport).await.context("serve stdio client")?
        }
        other => anyhow::bail!("unknown transport '{other}'"),
    };
    Ok(ConnectedService::new(service))
}

#[derive(Debug, Clone, Serialize)]
pub struct ServerHealth {
    pub name: String,
    pub connected: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct HealthSnapshot {
    pub servers: Vec<ServerHealth>,
    pub count: usize,
}

/// Connected servers, name to owned MCP service task.
#[derive(Debug, Clone)]
pub struct AppState {
    services: Arc<HashMap<String, Arc<ConnectedService>>>,
}

impl AppState {
    pub fn new(services: HashMap<String, Arc<ConnectedService>>) -> Self {
        Self {
            services: Arc::new(services),
        }
    }

    /// Connect every configured service. Failed connections are logged and
    /// skipped so one unavailable integration does not disable the sidecar.
    pub async fn connect_configured(configs: &[ServerCfg]) -> Self {
        let mut services = HashMap::new();
        for cfg in configs {
            match connect(cfg).await {
                Ok(service) => {
                    tracing::info!(
                        name = %cfg.name,
                        transport = %cfg.transport,
                        "connected to MCP server"
                    );
                    let service = Arc::new(service);
                    if let Some(previous) = services.insert(cfg.name.clone(), service) {
                        tracing::warn!(name = %cfg.name, "duplicate MCP server name replaced");
                        if let Err(error) = previous.shutdown().await {
                            tracing::error!(
                                name = %cfg.name,
                                %error,
                                "failed to shut down replaced MCP service"
                            );
                        }
                    }
                }
                Err(error) => {
                    tracing::warn!(name = %cfg.name, "MCP connect failed (skipping): {error:#}");
                }
            }
        }
        Self::new(services)
    }

    pub async fn health_snapshot(&self) -> HealthSnapshot {
        let mut servers = Vec::with_capacity(self.services.len());
        for (name, service) in self.services.iter() {
            servers.push(ServerHealth {
                name: name.clone(),
                connected: !service.is_closed().await,
            });
        }
        servers.sort_unstable_by(|left, right| left.name.cmp(&right.name));
        let count = servers.iter().filter(|server| server.connected).count();
        HealthSnapshot { servers, count }
    }

    pub async fn shutdown_all(&self) -> Result<()> {
        let mut failures = Vec::new();
        for (name, service) in self.services.iter() {
            if let Err(error) = service.shutdown().await {
                failures.push(format!("{name}: {error:#}"));
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            anyhow::bail!("failed to shut down MCP services: {}", failures.join("; "))
        }
    }
}

async fn health(State(state): State<AppState>) -> Json<HealthSnapshot> {
    Json(state.health_snapshot().await)
}

async fn list_tools(State(state): State<AppState>) -> Json<serde_json::Value> {
    let mut tools = Vec::new();
    for (name, service) in state.services.iter() {
        match service.peer().list_tools(Default::default()).await {
            Ok(result) => {
                for tool in result.tools {
                    tools.push(serde_json::json!({
                        "server": name,
                        "name": tool.name,
                        "description": tool.description,
                    }));
                }
            }
            Err(error) => tracing::warn!(%name, "list_tools failed: {error}"),
        }
    }
    Json(serde_json::json!({ "tools": tools }))
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

async fn call_tool(
    State(state): State<AppState>,
    Json(body): Json<CallBody>,
) -> Json<CallResult> {
    let Some(service) = state.services.get(&body.server) else {
        return Json(CallResult {
            ok: false,
            result: serde_json::Value::Null,
            error: Some(format!("no server '{}'", body.server)),
        });
    };
    // CallToolRequestParams is non-exhaustive, so deserialize the wire shape
    // instead of constructing it with a struct literal.
    let params: CallToolRequestParams = match serde_json::from_value(serde_json::json!({
        "name": body.tool,
        "arguments": body.args.as_object().cloned(),
    })) {
        Ok(params) => params,
        Err(error) => {
            return Json(CallResult {
                ok: false,
                result: serde_json::Value::Null,
                error: Some(format!("bad params: {error}")),
            });
        }
    };
    // Shutdown-deadlock defense-in-depth (2026-07-16): a tool call may not
    // outlive this bound even if the upstream MCP server hangs forever.
    // Without it, a hung call kept the axum graceful-shutdown drain waiting
    // (the primary fix cancels services AT the shutdown signal; this cap
    // guarantees the handler itself always terminates too). 300s is generous
    // for any real tool.
    const TOOL_CALL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);
    let call = tokio::time::timeout(TOOL_CALL_TIMEOUT, service.peer().call_tool(params));
    match call.await {
        Err(_elapsed) => {
            return Json(CallResult {
                ok: false,
                result: serde_json::Value::Null,
                error: Some(format!(
                    "tool call '{}' on '{}' timed out after {}s (upstream MCP server \
                     unresponsive)",
                    body.tool,
                    body.server,
                    TOOL_CALL_TIMEOUT.as_secs()
                )),
            });
        }
        Ok(Ok(result)) => match serde_json::to_value(result) {
            Ok(result) => Json(CallResult {
                ok: true,
                result,
                error: None,
            }),
            Err(error) => Json(CallResult {
                ok: false,
                result: serde_json::Value::Null,
                error: Some(format!("serialize tool result: {error}")),
            }),
        },
        Ok(Err(error)) => Json(CallResult {
            ok: false,
            result: serde_json::Value::Null,
            error: Some(error.to_string()),
        }),
    }
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/tools", get(list_tools))
        .route("/call", post(call_tool))
        .with_state(state)
}
