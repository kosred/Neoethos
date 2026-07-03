//! NeoEthos Mesh — the **fully-automatic** P2P sidecar.
//!
//! No Tailscale. No port-forwarding. No coordinator URL. No human in the loop.
//! A node just starts and joins the swarm:
//!   1. Stable identity — an ed25519 key persisted to `<data-dir>/identity.key`;
//!      its public key is this node's permanent mesh address.
//!   2. Automatic connectivity — iroh's default relay network does NAT
//!      hole-punching, so the node is reachable anywhere with zero config.
//!   3. Automatic peer discovery — every NeoEthos node subscribes to the SAME
//!      gossip rendezvous topic and announces itself, so nodes find each other
//!      with no manual setup (bootstrap seeds via `NEOETHOS_MESH_SEEDS`).
//!   4. Bridge — talks to the local NeoEthos app's `/federation/*` HTTP API;
//!      work distribution reuses that audited path, so the mesh is pure
//!      transport and every result still passes the local gates.
//!
//! ISOLATION: separate process, its own Cargo.lock. Nothing here can touch the
//! trading engine — it only speaks to the app over localhost HTTP.

use std::path::PathBuf;
use std::str::FromStr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use bytes::Bytes;
use futures_lite::StreamExt;
use iroh::endpoint::presets;
use iroh::{Endpoint, EndpointId, SecretKey};
use iroh_gossip::api::Event;
use iroh_gossip::net::Gossip;
use iroh_gossip::proto::TopicId;
use serde::{Deserialize, Serialize};

/// ALPN for direct NeoEthos mesh connections (work protocol, later phases).
const ALPN: &[u8] = b"neoethos/mesh/0";

/// The fixed rendezvous topic. EVERY NeoEthos node subscribes to this exact
/// topic, which is how they discover each other automatically. 32 bytes.
const ANNOUNCE_TOPIC: [u8; 32] = *b"neoethos-mesh-announce-v1-000000";

/// A node's periodic "I'm here + what I can do" broadcast.
#[derive(Debug, Serialize, Deserialize)]
struct Announce {
    node_id: String,
    cpu_cores: u32,
    work_types: Vec<String>,
    app_online: bool,
    ts: u64,
}

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
                    "neoethos-mesh — fully-automatic P2P discovery sidecar\n\
                     \n\
                     USAGE:\n  neoethos-mesh [--data-dir <dir>] [--app-url <http://127.0.0.1:PORT>]\n\
                     \n\
                     --data-dir  where the persistent identity.key lives (default: ./mesh-data)\n\
                     --app-url   the LOCAL NeoEthos app HTTP API (default: http://127.0.0.1:8080)\n\
                     \n\
                     ENV:\n  NEOETHOS_MESH_SEEDS  comma-separated bootstrap node ids (optional)\n\
                     \n\
                     No network configuration is needed — the node joins the swarm\n\
                     automatically via the iroh relay network. It NEVER touches the\n\
                     trading engine directly, only its HTTP API."
                );
                std::process::exit(0);
            }
            other => eprintln!("(ignoring unknown arg: {other})"),
        }
    }
    Args { data_dir, app_url }
}

/// Load the persistent 32-byte identity seed, or create + save one on first run.
fn load_or_create_secret(data_dir: &PathBuf) -> Result<SecretKey> {
    std::fs::create_dir_all(data_dir).with_context(|| format!("create {}", data_dir.display()))?;
    let path = data_dir.join("identity.key");
    let seed: [u8; 32] = match std::fs::read(&path) {
        Ok(b) if b.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&b);
            arr
        }
        _ => {
            use rand::RngCore;
            let mut arr = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut arr);
            std::fs::write(&path, arr).with_context(|| format!("write {}", path.display()))?;
            tracing::info!("created a new mesh identity at {}", path.display());
            arr
        }
    };
    Ok(SecretKey::from_bytes(&seed))
}

/// Bootstrap seed node ids from `NEOETHOS_MESH_SEEDS` (comma-separated). Empty
/// is fine — the first node in a swarm simply waits for others to find it.
fn seed_nodes() -> Vec<EndpointId> {
    std::env::var("NEOETHOS_MESH_SEEDS")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .filter_map(|s| match EndpointId::from_str(s) {
            Ok(id) => Some(id),
            Err(e) => {
                tracing::warn!("ignoring bad seed node id '{s}': {e}");
                None
            }
        })
        .collect()
}

/// Best-effort check that the local NeoEthos app is reachable over HTTP.
async fn app_online(app_url: &str) -> bool {
    let Ok(client) = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
    else {
        return false;
    };
    client
        .get(format!("{app_url}/federation/status"))
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .is_ok()
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "neoethos_mesh=info,iroh=warn,iroh_gossip=warn".into()),
        )
        .init();

    let args = parse_args();
    tracing::info!("NeoEthos Mesh — fully-automatic P2P discovery");

    // 1. Stable identity.
    let secret_key = load_or_create_secret(&args.data_dir)?;
    let my_id = secret_key.public();
    tracing::info!(node_id = %my_id, "this node's permanent mesh address");

    // 2. Automatic connectivity — iroh's default relay network (NAT
    //    hole-punching) + n0 discovery. Zero network config.
    // presets::N0 = default relay network + n0 discovery, fully automatic.
    let endpoint = Endpoint::builder(presets::N0)
        .secret_key(secret_key)
        .alpns(vec![ALPN.to_vec()])
        .bind()
        .await
        .context("failed to bind the iroh endpoint")?;
    tracing::info!("online via the iroh relay network — reachable anywhere, no config needed");

    // 3. Automatic peer discovery — gossip on the shared rendezvous topic.
    let gossip = Gossip::builder().spawn(endpoint.clone());
    let _router = iroh::protocol::Router::builder(endpoint.clone())
        .accept(iroh_gossip::ALPN, gossip.clone())
        .spawn();

    let topic_id = TopicId::from_bytes(ANNOUNCE_TOPIC);
    let seeds = seed_nodes();
    if seeds.is_empty() {
        tracing::info!("no bootstrap seeds set — waiting for peers to find us (set NEOETHOS_MESH_SEEDS to join faster)");
    } else {
        tracing::info!(count = seeds.len(), "bootstrapping discovery from seed nodes");
    }
    let topic = gossip
        .subscribe(topic_id, seeds)
        .await
        .context("failed to subscribe to the rendezvous topic")?;
    let (sender, mut receiver) = topic.split();

    // Announce loop — broadcast our presence + capabilities every 60s.
    let cpu_cores = std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(1);
    let app_url = args.app_url.clone();
    let my_id_str = my_id.to_string();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(60));
        loop {
            tick.tick().await;
            let ann = Announce {
                node_id: my_id_str.clone(),
                cpu_cores,
                work_types: vec!["discovery".into()],
                app_online: app_online(&app_url).await,
                ts: now_secs(),
            };
            if let Ok(json) = serde_json::to_vec(&ann) {
                if let Err(e) = sender.broadcast(Bytes::from(json)).await {
                    tracing::debug!("announce broadcast failed: {e}");
                }
            }
        }
    });

    // Discovery loop — learn about peers as they announce.
    tokio::spawn(async move {
        while let Some(event) = receiver.next().await {
            match event {
                Ok(Event::Received(msg)) => {
                    if let Ok(ann) = serde_json::from_slice::<Announce>(&msg.content) {
                        tracing::info!(
                            peer = %ann.node_id, cores = ann.cpu_cores, app = ann.app_online,
                            "discovered a NeoEthos peer"
                        );
                    }
                }
                Ok(Event::NeighborUp(id)) => tracing::info!(peer = %id, "peer connected"),
                Ok(Event::NeighborDown(id)) => tracing::info!(peer = %id, "peer left"),
                Ok(_) => {}
                Err(e) => tracing::debug!("gossip stream error: {e}"),
            }
        }
    });

    tracing::info!(
        "swarm joined. Peer discovery is automatic; the work protocol bridges to \
         the app's /federation/* API (see mesh/README.md). Press Ctrl-C to stop."
    );
    tokio::signal::ctrl_c().await.ok();
    tracing::info!("shutting down");
    endpoint.close().await;
    Ok(())
}
