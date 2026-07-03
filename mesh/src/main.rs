//! NeoEthos Mesh — P2P sidecar, **Phase A: identity + local bridge**.
//!
//! What this binary does in the DEFAULT build (honestly, and only this):
//!   1. Loads or creates a STABLE cryptographic identity — an ed25519 key
//!      persisted to `<data-dir>/identity.key`. Its public key is this node's
//!      permanent address on the mesh (the same key iroh will present once P2P
//!      is enabled, since iroh uses the same ed25519 curve).
//!   2. Verifies it can reach the LOCAL NeoEthos app's HTTP API
//!      (`/federation/*`) — the bridge every later phase builds on.
//!
//! With `--features p2p` it ALSO comes online over iroh (QUIC + TLS 1.3 +
//! default relay network) so the node is reachable by other NeoEthos nodes
//! anywhere, through NAT. That feature is gated only because iroh 0.95's
//! dependency tree currently fails to build on stable rustc (an upstream
//! pre-release bug in `ed25519-dalek`); the code below is complete and ready.
//!
//! What NO build does yet (Phases B–F, see README.md): gossip peer discovery,
//! the claim/accept/result work protocol, artifact blobs. Distributed
//! discovery over a trusted group works TODAY over HTTP (Advanced →
//! Federation); this sidecar is the road to doing it P2P, serverless.
//!
//! ISOLATION: separate process, own Cargo.lock. Nothing here can touch the
//! trading engine — it only speaks to the app over localhost HTTP.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use ed25519_dalek::SigningKey;
use rand::RngCore;

/// ALPN for the NeoEthos mesh protocol. Bumped when the wire protocol changes.
#[cfg(feature = "p2p")]
const ALPN: &[u8] = b"neoethos/mesh/0";

struct Args {
    data_dir: PathBuf,
    app_url: String,
}

fn parse_args() -> Args {
    let mut data_dir = PathBuf::from("mesh-data");
    let mut app_url = "http://127.0.0.1:8080".to_string();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--data-dir" => {
                if let Some(v) = it.next() {
                    data_dir = PathBuf::from(v);
                }
            }
            "--app-url" => {
                if let Some(v) = it.next() {
                    app_url = v.trim_end_matches('/').to_string();
                }
            }
            "-h" | "--help" => {
                eprintln!(
                    "neoethos-mesh (Phase A: identity + local bridge)\n\
                     \n\
                     USAGE:\n  neoethos-mesh [--data-dir <dir>] [--app-url <http://127.0.0.1:PORT>]\n\
                     \n\
                     --data-dir  where the persistent identity.key lives (default: ./mesh-data)\n\
                     --app-url   the LOCAL NeoEthos app HTTP API to bridge to (default: http://127.0.0.1:8080)\n\
                     \n\
                     Build with `--features p2p` to also come online over the\n\
                     iroh relay network (see mesh/README.md). This sidecar NEVER\n\
                     touches the trading engine directly — only its HTTP API."
                );
                std::process::exit(0);
            }
            other => eprintln!("(ignoring unknown arg: {other})"),
        }
    }
    Args { data_dir, app_url }
}

/// Load the persistent 32-byte identity seed, or create + save one on first
/// run. Returns the ed25519 signing key derived from it.
fn load_or_create_identity(data_dir: &PathBuf) -> Result<SigningKey> {
    std::fs::create_dir_all(data_dir).with_context(|| format!("create {}", data_dir.display()))?;
    let path = data_dir.join("identity.key");
    let seed: [u8; 32] = match std::fs::read(&path) {
        Ok(b) if b.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&b);
            arr
        }
        _ => {
            let mut arr = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut arr);
            std::fs::write(&path, arr).with_context(|| format!("write {}", path.display()))?;
            tracing::info!("created a new mesh identity at {}", path.display());
            arr
        }
    };
    Ok(SigningKey::from_bytes(&seed))
}

/// Best-effort check that the local NeoEthos app is reachable over HTTP —
/// the bridge all later phases depend on. Never fatal.
async fn ping_local_app(app_url: &str) -> Result<String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;
    let resp = client
        .get(format!("{app_url}/federation/status"))
        .send()
        .await?
        .error_for_status()?;
    let v: serde_json::Value = resp.json().await?;
    let queued = v.get("jobsQueued").and_then(|x| x.as_u64()).unwrap_or(0);
    Ok(format!("reachable — {queued} federated job(s) queued locally"))
}

/// Bring the node online over iroh (relay network). Complete + ready; compiled
/// only under `--features p2p` until iroh's tree builds on stable rustc.
#[cfg(feature = "p2p")]
async fn go_online(seed: &[u8; 32]) -> Result<iroh::Endpoint> {
    let secret_key = iroh::SecretKey::from_bytes(seed);
    let endpoint = iroh::Endpoint::builder()
        .secret_key(secret_key)
        .alpns(vec![ALPN.to_vec()])
        .bind()
        .await
        .context("failed to bind the iroh endpoint (network/relay issue)")?;
    tracing::info!("online — reachable by other NeoEthos nodes via the iroh relay network");
    Ok(endpoint)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "neoethos_mesh=info,iroh=warn".into()),
        )
        .init();

    let args = parse_args();
    tracing::info!("NeoEthos Mesh — Phase A (identity + local bridge)");

    // 1. Stable identity. The public key is this node's permanent mesh address.
    let signing_key = load_or_create_identity(&args.data_dir)?;
    let node_id = hex::encode(signing_key.verifying_key().to_bytes());
    tracing::info!(%node_id, "this node's permanent mesh identity (ed25519 public key)");

    // 2. (feature p2p) Come online over iroh.
    #[cfg(feature = "p2p")]
    let _endpoint = {
        let seed = signing_key.to_bytes();
        go_online(&seed).await?
    };
    #[cfg(not(feature = "p2p"))]
    tracing::info!(
        "relay connectivity is available in a `--features p2p` build (see mesh/README.md); \
         this default build provides the stable identity + local bridge"
    );

    // 3. Verify the local-app HTTP bridge (fail-soft).
    match ping_local_app(&args.app_url).await {
        Ok(msg) => tracing::info!(app_url = %args.app_url, "local NeoEthos app {msg}"),
        Err(e) => tracing::warn!(
            app_url = %args.app_url,
            "local NeoEthos app not reachable ({e}); start the app or pass --app-url with its port"
        ),
    }

    tracing::info!(
        "Phase A ready. Peer discovery + the work protocol (Phases B–F) are the next \
         bricks — see mesh/README.md. Press Ctrl-C to stop."
    );
    tokio::signal::ctrl_c().await.ok();
    tracing::info!("shutting down");
    #[cfg(feature = "p2p")]
    _endpoint.close().await;
    Ok(())
}
