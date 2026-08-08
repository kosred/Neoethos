//! `neoethos-mcp` — the Codex control plane binary.
//!
//! stdio MCP server. stdout carries JSON-RPC exclusively; all logging goes
//! to stderr (a CI test spawns this binary and asserts stdout stays
//! protocol-clean). Configuration is CLI flags only — no env vars, per the
//! one-config directive:
//!
//!   neoethos-mcp [--base-url http://127.0.0.1:7423] [--token <bearer>]

use rmcp::ServiceExt;
use rmcp::transport::stdio;

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:7423";

const USAGE: &str = "neoethos-mcp — NeoEthos Codex control plane (MCP server over stdio)\n\
                     \n\
                     USAGE:\n\
                     \x20 neoethos-mcp [--base-url <url>] [--token <bearer>]\n\
                     \n\
                     OPTIONS:\n\
                     \x20 --base-url <url>   Backend API base URL (default http://127.0.0.1:7423)\n\
                     \x20 --token <bearer>   Forwarded as Authorization: Bearer when the operator\n\
                     \x20                    enabled NEOETHOS_API_TOKEN on the backend\n\
                     \x20 -h, --help         Print this help (to stderr) and exit";

fn parse_args() -> anyhow::Result<Option<(String, Option<String>)>> {
    let mut base_url = DEFAULT_BASE_URL.to_string();
    let mut token: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--base-url" => {
                base_url = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--base-url requires a value"))?;
            }
            "--token" => {
                token = Some(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("--token requires a value"))?,
                );
            }
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
    Ok(Some((base_url, token)))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Logging FIRST, to stderr ONLY (stdout is the JSON-RPC channel).
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let Some((base_url, token)) = parse_args()? else {
        return Ok(()); // --help
    };

    let plane = neoethos_mcp::ControlPlane::new(&base_url, token)?;
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
