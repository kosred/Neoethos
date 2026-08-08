//! NeoEthos Codex control plane — an MCP *server* (stdio transport) that
//! wraps the localhost backend HTTP API so OpenAI Codex CLI (or any
//! MCP-speaking agent runtime) can operate the whole system: account state,
//! market data, journal analytics, discovery/training jobs, autopilot,
//! settings, and DEMO-ONLY trade execution.
//!
//! Design of record: `docs/codex-control-plane.md` (commit c6431311).
//!
//! Hard rules enforced here:
//! - **Demo-only, fail-closed**: every trade-affecting tool re-verifies the
//!   active broker account is a demo account on every call ([`backend`]).
//! - **Loud failure, no fallback**: a backend that is not running produces a
//!   fixed actionable error on every call; nothing retries or degrades.
//! - **stdout is protocol-only**: logging goes to stderr; this crate
//!   contains no `println!`.

pub mod backend;
pub mod ops;
pub mod params;
pub mod server;

pub use backend::{Backend, DemoProof, GUARDED_TOOLS, TRADE_ROUTES, ToolError};
pub use server::ControlPlane;

/// Refuse a non-loopback backend host unless the operator explicitly opted in.
///
/// The demo guard authorizes trades by trusting the answers of the very backend
/// the base URL names, and `--token` is forwarded to it on every request. A
/// remote (or MITM) host could therefore both harvest the token and answer
/// `environment:"Demo"` to unlock the trade routes. "localhost only" is the
/// design's stated posture, so a non-loopback target must be a deliberate, loud
/// choice — `--allow-remote` — never a silent default.
pub fn enforce_loopback(base_url: &str, allow_remote: bool) -> anyhow::Result<()> {
    let host = reqwest::Url::parse(base_url)
        .map_err(|e| anyhow::anyhow!("--base-url is not a valid URL ({base_url:?}): {e}"))?
        .host_str()
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("--base-url {base_url:?} has no host"))?;
    // `Url::host_str` returns an IPv6 literal bracketed (`[::1]`); strip the
    // brackets before parsing so IPv6 loopback is recognised, not refused.
    let host_for_ip = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(&host);
    let is_loopback = host == "localhost"
        || host_for_ip
            .parse::<std::net::IpAddr>()
            .map(|ip| ip.is_loopback())
            .unwrap_or(false);
    if is_loopback {
        return Ok(());
    }
    if allow_remote {
        tracing::warn!(
            target: "neoethos_mcp",
            host = %host,
            "--base-url points at a NON-LOOPBACK host and --allow-remote was given: the demo \
             guard now trusts a remote backend and the bearer token is sent to it. This defeats \
             the 'localhost only' posture — proceed only if you fully control that host."
        );
        return Ok(());
    }
    anyhow::bail!(
        "--base-url {base_url:?} points at non-loopback host {host:?}. This control plane binds \
         to a localhost backend by design: it forwards --token to that host and trusts its demo \
         answers to unlock trading. Pass --allow-remote to override only if you fully control the \
         host and network path."
    )
}
