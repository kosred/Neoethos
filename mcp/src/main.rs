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
use neoethos_execution_budget::{
    StartupEvent, StartupRuntimeKind, StartupTrace, detected_request_with_parent,
    format_startup_diagnostics, install_process_budget, parse_parent_cpu_assignment,
};
use neoethos_mcp::{AppState, Config, router};

struct Args {
    config_path: PathBuf,
    startup_diagnostics: bool,
}

async fn serve_api(state: AppState, port: u16) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
        .await
        .with_context(|| format!("bind 127.0.0.1:{port}"))?;
    tracing::info!(
        "NeoEthos MCP sidecar on http://127.0.0.1:{port} — /health /tools /call"
    );
    // Shutdown-deadlock fix (2026-07-16): axum's graceful shutdown waits for
    // in-flight requests to finish AFTER this future resolves — but an
    // in-flight `/call` blocked on a hung MCP server only finishes once its
    // service is cancelled, and the old code cancelled services AFTER serve
    // returned. Circular wait: serve → /call → service → (after serve).
    // Cancel the services HERE, at the signal, before the future resolves —
    // pending peer calls then error out immediately and the drain completes.
    // `shutdown_all` is idempotent (Option::take per service), so the
    // belt-and-braces second call in `main` stays safe.
    let shutdown_state = state.clone();
    axum::serve(listener, router(state))
        .with_graceful_shutdown(async move {
            match tokio::signal::ctrl_c().await {
                Ok(()) => tracing::info!("shutdown signal received"),
                Err(error) => tracing::error!(%error, "failed to listen for shutdown signal"),
            }
            if let Err(error) = shutdown_state.shutdown_all().await {
                tracing::warn!(%error, "MCP service shutdown reported errors (continuing drain)");
            }
        })
        .await
        .context("serve")
}

fn main() -> Result<()> {
    let raw_args: Vec<String> = std::env::args().collect();
    let Some(args) = parse_args(&raw_args)? else {
        return Ok(());
    };
    let (cfg, config_warning) = load_config(&args.config_path)?;
    let mut startup_trace = StartupTrace::default();
    startup_trace.record(StartupEvent::ConfigurationLoaded)?;
    let parent_cpu_assignment = parse_parent_cpu_assignment(&raw_args)?;
    startup_trace.record(StartupEvent::ParentCpuCapParsed)?;
    let request = detected_request_with_parent(parent_cpu_assignment);
    neoethos_execution_budget::resolve_execution_budget(request.clone())?;
    startup_trace.record(StartupEvent::CpuBudgetResolved)?;
    let installed = install_process_budget(request)?;
    startup_trace.record(StartupEvent::CpuBudgetInstalled)?;

    let env_filter = if std::env::var_os("RUST_LOG").is_some() {
        tracing_subscriber::EnvFilter::try_from_default_env().context("parse RUST_LOG")?
    } else {
        tracing_subscriber::EnvFilter::new("neoethos_mcp=info,rmcp=warn")
    };
    tracing_subscriber::fmt().with_env_filter(env_filter).init();
    if let Some(warning) = config_warning {
        tracing::warn!("{warning}");
    }

    let runtime_workers = installed.resolved().effective_worker_limit.get();
    let managed_runtime = build_managed_runtime(runtime_workers)?;
    startup_trace.record(StartupEvent::TokioRuntimeBuilt)?;
    if args.startup_diagnostics {
        eprintln!(
            "{}",
            format_startup_diagnostics(
                "neoethos-mcp",
                installed,
                StartupRuntimeKind::Tokio,
                Some(runtime_workers),
                &startup_trace,
            )
        );
        return Ok(());
    }
    startup_trace.record(StartupEvent::ApplicationBuilderStarted)?;
    tracing::info!(
        target: "neoethos_mcp::startup",
        "{}",
        format_startup_diagnostics(
            "neoethos-mcp",
            installed,
            StartupRuntimeKind::Tokio,
            Some(runtime_workers),
            &startup_trace,
        )
    );
    managed_runtime.block_on(async_main(cfg))
}

async fn async_main(cfg: Config) -> Result<()> {
    let state = AppState::connect_configured(&cfg.servers).await;
    let port = cfg.port.unwrap_or(7431);
    let serve_result = serve_api(state.clone(), port).await;
    let shutdown_result = state.shutdown_all().await;

    serve_result?;
    shutdown_result.context("shut down MCP services")
}

fn build_managed_runtime(worker_threads: usize) -> std::io::Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
        .enable_all()
        .build()
}

fn load_config(path: &PathBuf) -> Result<(Config, Option<String>)> {
    match std::fs::read_to_string(path) {
        Ok(serialized) => Ok((
            serde_json::from_str(&serialized).context("parse mcp config")?,
            None,
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok((
            Config::default(),
            Some(format!(
                "no MCP config at {} — starting with no servers",
                path.display()
            )),
        )),
        Err(error) => Err(error).with_context(|| format!("read MCP config {}", path.display())),
    }
}

fn parse_args(raw_args: &[String]) -> Result<Option<Args>> {
    let mut config_path = PathBuf::from("mcp_servers.json");
    let mut startup_diagnostics = false;
    let mut index = 1;
    while index < raw_args.len() {
        match raw_args[index].as_str() {
            "--config" => {
                index += 1;
                config_path = raw_args
                    .get(index)
                    .map(PathBuf::from)
                    .ok_or_else(|| anyhow::anyhow!("--config requires a path"))?;
            }
            "--cpu-threads" => {
                index += 1;
                raw_args
                    .get(index)
                    .ok_or_else(|| anyhow::anyhow!("--cpu-threads requires a value"))?;
            }
            value if value.starts_with("--cpu-threads=") => {}
            "--startup-diagnostics" => startup_diagnostics = true,
            "-h" | "--help" => {
                eprintln!(
                    "neoethos-mcp [--config <path>] [--cpu-threads <positive integer>] \
                     [--startup-diagnostics]"
                );
                return Ok(None);
            }
            unknown => anyhow::bail!("unknown argument `{unknown}`"),
        }
        index += 1;
    }
    Ok(Some(Args {
        config_path,
        startup_diagnostics,
    }))
}
