//! `neoethos-control-plane` — the Codex control plane binary.
//!
//! stdio MCP server. stdout carries JSON-RPC exclusively; all logging goes
//! to stderr (a CI test spawns this binary and asserts stdout stays
//! protocol-clean). Configuration is CLI flags only — no env vars, per the
//! one-config directive:
//!
//!   neoethos-control-plane [--base-url http://127.0.0.1:7423] [--token <bearer>]
//!
//! The binary was called `neoethos-mcp` until 2026-08-10, which collided with
//! the unrelated outbound sidecar built from the top-level `mcp/` workspace —
//! the one the installer places next to the desktop app and spawns by name.

use neoethos_execution_budget::{
    StartupEvent, StartupRuntimeKind, StartupTrace, detected_request_with_parent,
    format_startup_diagnostics, install_process_budget, parse_parent_cpu_assignment,
};
use rmcp::ServiceExt;
use rmcp::transport::stdio;

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:7423";

const USAGE: &str = "neoethos-control-plane — NeoEthos Codex control plane (MCP server over stdio)\n\
                     \n\
                     USAGE:\n\
                     \x20 neoethos-control-plane [--base-url <url>] [--token <bearer>]\n\
                     \n\
                     OPTIONS:\n\
                     \x20 --base-url <url>   Backend API base URL (default http://127.0.0.1:7423)\n\
                     \x20 --token <bearer>   Forwarded as Authorization: Bearer when the operator\n\
                     \x20                    enabled NEOETHOS_API_TOKEN on the backend\n\
                     \x20 --allow-remote     Permit a non-loopback --base-url (default: refuse).\n\
                     \x20                    The demo guard trusts the backend it talks to and the\n\
                     \x20                    bearer token is sent to it, so a remote host is opt-in\n\
                     \x20 --cpu-threads <n>  Parent-assigned positive CPU worker cap\n\
                     \x20 --startup-diagnostics\n\
                     \x20                    Build the managed runtime, report its budget, and exit\n\
                     \x20 -h, --help         Print this help (to stderr) and exit";

struct Args {
    base_url: String,
    token: Option<String>,
    startup_diagnostics: bool,
}

fn parse_args(raw_args: &[String]) -> anyhow::Result<Option<Args>> {
    let mut base_url = DEFAULT_BASE_URL.to_string();
    let mut token: Option<String> = None;
    let mut allow_remote = false;
    let mut startup_diagnostics = false;
    let mut args = raw_args.iter().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--base-url" => {
                base_url = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--base-url requires a value"))?
                    .clone();
            }
            "--token" => {
                token = Some(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("--token requires a value"))?
                        .clone(),
                );
            }
            "--allow-remote" => {
                allow_remote = true;
            }
            "--cpu-threads" => {
                args.next()
                    .ok_or_else(|| anyhow::anyhow!("--cpu-threads requires a value"))?;
            }
            value if value.starts_with("--cpu-threads=") => {}
            "--startup-diagnostics" => startup_diagnostics = true,
            "-h" | "--help" => {
                // Help goes to STDERR — stdout is reserved for JSON-RPC.
                eprintln!("{USAGE}");
                return Ok(None);
            }
            other => {
                anyhow::bail!("unknown argument '{other}'\n\n{USAGE}");
            }
        }
    }
    neoethos_mcp::enforce_loopback(&base_url, allow_remote)?;
    Ok(Some(Args {
        base_url,
        token,
        startup_diagnostics,
    }))
}

fn main() -> anyhow::Result<()> {
    let raw_args: Vec<String> = std::env::args().collect();
    let Some(args) = parse_args(&raw_args)? else {
        return Ok(());
    };
    let mut startup_trace = StartupTrace::default();
    let parent_cpu_assignment = parse_parent_cpu_assignment(&raw_args)?;
    startup_trace.record(StartupEvent::ParentCpuCapParsed)?;
    let request = detected_request_with_parent(parent_cpu_assignment);
    neoethos_execution_budget::resolve_execution_budget(request.clone())?;
    startup_trace.record(StartupEvent::CpuBudgetResolved)?;
    let installed = install_process_budget(request)?;
    startup_trace.record(StartupEvent::CpuBudgetInstalled)?;

    // Logging is stderr-only because stdout is the JSON-RPC channel. It is
    // initialized only after the immutable process budget exists.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let runtime_workers = installed.resolved().effective_worker_limit.get();
    let managed_runtime = build_managed_runtime(runtime_workers)?;
    startup_trace.record(StartupEvent::TokioRuntimeBuilt)?;
    if args.startup_diagnostics {
        eprintln!(
            "{}",
            format_startup_diagnostics(
                "neoethos-control-plane",
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
            "neoethos-control-plane",
            installed,
            StartupRuntimeKind::Tokio,
            Some(runtime_workers),
            &startup_trace,
        )
    );
    managed_runtime.block_on(async_main(args))
}

fn build_managed_runtime(worker_threads: usize) -> std::io::Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
        .enable_all()
        .build()
}

async fn async_main(args: Args) -> anyhow::Result<()> {
    let plane = neoethos_mcp::ControlPlane::new(&args.base_url, args.token)?;
    tracing::info!(
        target: "neoethos_mcp",
        base_url = %plane.backend().base_url(),
        "NeoEthos Codex control plane starting on stdio (demo-guarded trading surface)"
    );

    let service = plane
        .serve(stdio())
        .await
        .map_err(|e| anyhow::anyhow!("MCP initialization over stdio failed: {e}"))?;
    service
        .waiting()
        .await
        .map_err(|e| anyhow::anyhow!("MCP service terminated with an error: {e}"))?;
    Ok(())
}
