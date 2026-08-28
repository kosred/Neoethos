use std::collections::HashMap;
use std::ffi::OsStr;
use std::sync::Arc;

use anyhow::{Context, Result};
use neoethos_mcp::{AppState, ConnectedService, ServerCfg, connect};
use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::{CallToolRequestParams, ServerCapabilities, ServerInfo};
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};
use rmcp::{ServerHandler, ServiceExt, schemars, tool, tool_handler, tool_router};
use tokio::sync::oneshot;

const STDIO_CHILD_ENV: &str = "NEOETHOS_MCP_TEST_STDIO_CHILD";

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct EchoRequest {
    message: String,
}

#[derive(Debug, Clone)]
struct EchoServer {
    tool_router: ToolRouter<Self>,
}

impl EchoServer {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl EchoServer {
    #[tool(description = "Return the supplied message unchanged")]
    fn echo(&self, Parameters(request): Parameters<EchoRequest>) -> String {
        request.message
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for EchoServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
    }
}

struct HttpFixture {
    url: String,
    stop: oneshot::Sender<()>,
    task: tokio::task::JoinHandle<std::io::Result<()>>,
}

impl HttpFixture {
    async fn start() -> Result<Self> {
        let service: StreamableHttpService<EchoServer, LocalSessionManager> =
            StreamableHttpService::new(
                || Ok(EchoServer::new()),
                Default::default(),
                StreamableHttpServerConfig::default().with_sse_keep_alive(None),
            );
        let router = axum::Router::new().nest_service("/mcp", service);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .context("bind HTTP MCP fixture")?;
        let address = listener.local_addr().context("read HTTP fixture address")?;
        let (stop, stopped) = oneshot::channel();
        let task = tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async move {
                    stopped.await.expect("HTTP fixture stop sender must remain live");
                })
                .await
        });
        Ok(Self {
            url: format!("http://{address}/mcp"),
            stop,
            task,
        })
    }

    async fn shutdown(self) -> Result<()> {
        self.stop
            .send(())
            .map_err(|()| anyhow::anyhow!("HTTP MCP fixture stopped before shutdown"))?;
        self.task
            .await
            .context("join HTTP MCP fixture")?
            .context("serve HTTP MCP fixture")
    }
}

async fn connect_http(name: &str, url: String) -> Result<Arc<ConnectedService>> {
    let config = ServerCfg {
        name: name.to_owned(),
        transport: "http".to_owned(),
        url: Some(url),
        command: None,
        args: Vec::new(),
        env: HashMap::new(),
    };
    Ok(Arc::new(connect(&config).await?))
}

async fn connect_stdio(name: &str) -> Result<Arc<ConnectedService>> {
    let executable = std::env::current_exe().context("resolve lifecycle test executable")?;
    let config = ServerCfg {
        name: name.to_owned(),
        transport: "stdio".to_owned(),
        url: None,
        command: Some(executable.to_string_lossy().into_owned()),
        args: vec![
            "--exact".to_owned(),
            "stdio_echo_server_child".to_owned(),
            "--quiet".to_owned(),
            "--test-threads=1".to_owned(),
        ],
        env: HashMap::from([(STDIO_CHILD_ENV.to_owned(), "1".to_owned())]),
    };
    Ok(Arc::new(connect(&config).await?))
}

fn state_with(services: impl IntoIterator<Item = (String, Arc<ConnectedService>)>) -> AppState {
    AppState::new(services.into_iter().collect())
}

async fn assert_echo(service: &ConnectedService, message: &str) -> Result<()> {
    let arguments = serde_json::from_value(serde_json::json!({ "message": message }))
        .context("build echo arguments")?;
    let result = service
        .peer()
        .call_tool(CallToolRequestParams::new("echo").with_arguments(arguments))
        .await
        .context("call echo tool")?;
    let value = serde_json::to_value(result).context("serialize echo response")?;
    assert_eq!(value["content"][0]["text"], message);
    Ok(())
}

#[tokio::test]
async fn lifecycle_and_approval_http_holder_keeps_tool_callable() -> Result<()> {
    let fixture = HttpFixture::start().await?;
    let service = connect_http("http-echo", fixture.url.clone()).await?;
    let state = state_with([("http-echo".to_owned(), Arc::clone(&service))]);

    assert_echo(&service, "after HTTP connect returned").await?;

    state.shutdown_all().await?;
    fixture.shutdown().await
}

#[tokio::test]
async fn lifecycle_and_approval_stdio_holder_keeps_tool_callable() -> Result<()> {
    let service = connect_stdio("stdio-echo").await?;
    let state = state_with([("stdio-echo".to_owned(), Arc::clone(&service))]);

    assert_echo(&service, "after stdio connect returned").await?;

    state.shutdown_all().await
}

#[tokio::test]
async fn lifecycle_and_approval_shutdown_closes_transport_and_health() -> Result<()> {
    let fixture = HttpFixture::start().await?;
    let http = connect_http("http-echo", fixture.url.clone()).await?;
    let stdio = connect_stdio("stdio-echo").await?;
    let state = state_with([
        ("http-echo".to_owned(), Arc::clone(&http)),
        ("stdio-echo".to_owned(), Arc::clone(&stdio)),
    ]);

    let before = state.health_snapshot().await;
    assert_eq!(before.count, 2);
    assert!(before.servers.iter().all(|server| server.connected));

    state.shutdown_all().await?;

    assert!(http.is_closed().await);
    assert!(stdio.is_closed().await);
    let after = state.health_snapshot().await;
    assert_eq!(after.count, 0);
    assert!(after.servers.iter().all(|server| !server.connected));

    fixture.shutdown().await
}

#[test]
fn stdio_echo_server_child() {
    if std::env::var_os(STDIO_CHILD_ENV).as_deref() != Some(OsStr::new("1")) {
        return;
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build stdio MCP fixture runtime");
    runtime.block_on(async {
        let service = EchoServer::new()
            .serve(rmcp::transport::io::stdio())
            .await
            .expect("start stdio MCP fixture");
        service
            .waiting()
            .await
            .expect("wait for stdio MCP fixture shutdown");
    });
}
