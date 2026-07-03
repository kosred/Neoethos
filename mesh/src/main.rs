//! NeoEthos Mesh — fully-automatic P2P sidecar with WORK DISTRIBUTION.
//!
//! No Tailscale, no port-forwarding, no coordinator URL, no human in the loop.
//! A node starts, joins the swarm, and — when peers publish work — actually
//! distributes and runs it:
//!   1. Stable ed25519 identity (persisted); public key = mesh address.
//!   2. Automatic connectivity via iroh's default relay network (presets::N0).
//!   3. Automatic peer discovery via iroh-gossip on a shared rendezvous topic;
//!      each node announces {capabilities, has_work}.
//!   4. WORK PROTOCOL over iroh QUIC streams (ALPN `neoethos/mesh/0`):
//!      a worker seeing a coordinator with queued work claims a job, runs it
//!      on its OWN local NeoEthos engine, and submits the result back — where
//!      it passes every local gate before it can mean anything.
//!
//! Roles are automatic and simultaneous: any node with queued jobs
//! (/federation/status) is a coordinator; any node with the engine idle is a
//! worker. Discovery is fully distributed; training runs on the worker
//! (model-weight return over iroh-blobs is the documented next step).
//!
//! ISOLATION: separate process, own Cargo.lock. Nothing here can touch the
//! trading engine — it only speaks to the app over localhost HTTP.

use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use bytes::Bytes;
use futures_lite::StreamExt;
use iroh::endpoint::presets;
use iroh::protocol::{AcceptError, ProtocolHandler};
use iroh::{Endpoint, EndpointId, SecretKey};
use iroh_gossip::api::Event;
use iroh_gossip::net::Gossip;
use iroh_gossip::proto::TopicId;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

/// Direct-connection ALPN for the work protocol.
const MESH_ALPN: &[u8] = b"neoethos/mesh/0";
/// Shared rendezvous topic — every node subscribes to it for discovery.
const ANNOUNCE_TOPIC: [u8; 32] = *b"neoethos-mesh-announce-v1-000000";
/// A peer is forgotten if we haven't heard an announce in this long.
const PEER_TTL: Duration = Duration::from_secs(180);

// ── Wire types ───────────────────────────────────────────────────────────────

/// A node's periodic "I'm here + what I can do + do I have work" broadcast.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Announce {
    node_id: String,
    cpu_cores: u32,
    ram_mb: u64,
    gpu: bool,
    work_types: Vec<String>,
    app_online: bool,
    /// True when this node's local app has queued federated jobs.
    has_work: bool,
    ts: u64,
}

/// This machine's real hardware, read once from the local app's `/hardware`
/// (falls back to core count if the app is offline). This is what the node
/// contributes to the swarm's total capacity.
#[derive(Debug, Clone, Copy)]
struct HwCaps {
    cpu_cores: u32,
    ram_mb: u64,
    gpu: bool,
}

/// One unit of work (mirrors the app's FedJob; work_type: discovery|training).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FedJob {
    symbol: String,
    #[serde(rename = "baseTf")]
    base_tf: String,
    #[serde(default = "disc")]
    work_type: String,
}
fn disc() -> String {
    "discovery".into()
}

/// Request a worker sends to a coordinator over a QUIC bi-stream.
#[derive(Debug, Serialize, Deserialize)]
enum MeshReq {
    GetJob { worker: String },
    Submit {
        symbol: String,
        base_tf: String,
        portfolio_json: String,
        trades_json: Option<String>,
    },
}

/// Coordinator's reply.
#[derive(Debug, Serialize, Deserialize)]
enum MeshResp {
    Job(Option<FedJob>),
    SubmitAck { ok: bool, msg: String },
    Err(String),
}

#[derive(Debug, Clone)]
struct PeerInfo {
    cpu_cores: u32,
    ram_mb: u64,
    gpu: bool,
    app_online: bool,
    has_work: bool,
    last_seen: Instant,
}

type Peers = Arc<Mutex<HashMap<EndpointId, PeerInfo>>>;

// ── Config / args ─────────────────────────────────────────────────────────────

struct Args {
    data_dir: PathBuf,
    app_url: String,
}

fn resolve_app_url(explicit: Option<String>) -> String {
    if let Some(u) = explicit {
        return u.trim_end_matches('/').to_string();
    }
    // The desktop app writes its ephemeral port here on startup; the headless
    // CLI uses the fixed 7423. Prefer the file, fall back to 7423.
    if let Ok(p) = std::fs::read_to_string(std::env::temp_dir().join("neoethos_api_port")) {
        if let Ok(port) = p.trim().parse::<u16>() {
            if port != 0 {
                return format!("http://127.0.0.1:{port}");
            }
        }
    }
    "http://127.0.0.1:7423".to_string()
}

fn parse_args() -> Args {
    let mut data_dir = PathBuf::from("mesh-data");
    let mut app_url_explicit: Option<String> = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--data-dir" => data_dir = it.next().map(PathBuf::from).unwrap_or(data_dir),
            "--app-url" => app_url_explicit = it.next(),
            "-h" | "--help" => {
                eprintln!(
                    "neoethos-mesh — fully-automatic P2P work-distribution sidecar\n\n\
                     USAGE:\n  neoethos-mesh [--data-dir <dir>] [--app-url <http://127.0.0.1:PORT>]\n\n\
                     --data-dir  where identity.key lives (default: ./mesh-data)\n\
                     --app-url   local NeoEthos app API. Auto-detected from the\n\
                                 desktop app's port file, else http://127.0.0.1:7423\n\n\
                     ENV: NEOETHOS_MESH_SEEDS  comma-separated bootstrap node ids\n\n\
                     Joins the swarm automatically; never touches the trading engine\n\
                     directly, only its HTTP API."
                );
                std::process::exit(0);
            }
            other => eprintln!("(ignoring unknown arg: {other})"),
        }
    }
    Args { data_dir, app_url: resolve_app_url(app_url_explicit) }
}

// ── Identity ──────────────────────────────────────────────────────────────────

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

fn seed_nodes() -> Vec<EndpointId> {
    std::env::var("NEOETHOS_MESH_SEEDS")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .filter_map(|s| EndpointId::from_str(s).ok())
        .collect()
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

// ── Local-app HTTP helpers (the bridge) ──────────────────────────────────────

#[derive(Debug, Clone)]
struct AppClient {
    http: reqwest::Client,
    base: String,
}

impl AppClient {
    fn new(base: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("reqwest client");
        Self { http, base }
    }

    /// This machine's real hardware from the app's `/hardware`. Falls back to
    /// the local logical core count (RAM 0, no GPU) when the app is offline.
    async fn hardware(&self) -> HwCaps {
        let fallback = HwCaps {
            cpu_cores: std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(1),
            ram_mb: 0,
            gpu: false,
        };
        let Ok(r) = self.http.get(format!("{}/hardware", self.base)).send().await else {
            return fallback;
        };
        let Ok(v) = r.json::<serde_json::Value>().await else {
            return fallback;
        };
        HwCaps {
            cpu_cores: v.pointer("/cpu/coresLogical").and_then(|x| x.as_u64())
                .map(|c| c as u32).unwrap_or(fallback.cpu_cores),
            ram_mb: v.pointer("/ram/totalMb").and_then(|x| x.as_u64()).unwrap_or(0),
            gpu: v.pointer("/gpu/available").and_then(|x| x.as_bool()).unwrap_or(false),
        }
    }

    /// (app_online, has_queued_work)
    async fn status(&self) -> (bool, bool) {
        match self.http.get(format!("{}/federation/status", self.base)).send().await {
            Ok(r) if r.status().is_success() => {
                let v: serde_json::Value = r.json().await.unwrap_or_default();
                let queued = v.get("jobsQueued").and_then(|x| x.as_u64()).unwrap_or(0);
                (true, queued > 0)
            }
            _ => (false, false),
        }
    }

    /// Lease the next local federated job (coordinator side helper).
    async fn next_job(&self, worker: &str) -> Option<FedJob> {
        let r = self
            .http
            .get(format!("{}/federation/job?worker={worker}", self.base))
            .send()
            .await
            .ok()?;
        if r.status().is_success() {
            r.json::<FedJob>().await.ok()
        } else {
            None
        }
    }

    /// Hand a result to the local coordinator's gates (coordinator side helper).
    async fn submit(
        &self,
        worker: &str,
        symbol: &str,
        base_tf: &str,
        portfolio_json: &str,
        trades_json: Option<&str>,
    ) -> (bool, String) {
        let body = serde_json::json!({
            "worker": worker, "symbol": symbol, "baseTf": base_tf,
            "portfolioJson": portfolio_json, "tradesJson": trades_json,
        });
        match self.http.post(format!("{}/federation/submit", self.base)).json(&body).send().await {
            Ok(r) => {
                let ok = r.status().is_success();
                let v: serde_json::Value = r.json().await.unwrap_or_default();
                let msg = v.get("saved").and_then(|x| x.as_str()).map(String::from)
                    .or_else(|| v.get("error").and_then(|x| x.as_str()).map(String::from))
                    .unwrap_or_default();
                (ok, msg)
            }
            Err(e) => (false, e.to_string()),
        }
    }

    /// Start a local engine run for a claimed job (worker side).
    async fn start_engine(&self, work_type: &str, symbol: &str, base_tf: &str) -> bool {
        let path = if work_type == "training" {
            "/engines/training/start"
        } else {
            "/engines/discovery/start"
        };
        let body = serde_json::json!({ "symbol": symbol, "base_tf": base_tf });
        self.http.post(format!("{}{path}", self.base)).json(&body).send().await
            .map(|r| r.status().is_success()).unwrap_or(false)
    }

    /// Poll the engine state string for a work type ("Running"/"Idle"/…).
    async fn engine_state(&self, work_type: &str) -> String {
        let key = if work_type == "training" { "training" } else { "discovery" };
        let Ok(r) = self.http.get(format!("{}/engines/status", self.base)).send().await else {
            return "Unknown".into();
        };
        let Ok(v) = r.json::<serde_json::Value>().await else {
            return "Unknown".into();
        };
        v.get(key).and_then(|x| x.as_str()).map(String::from).unwrap_or_else(|| "Unknown".into())
    }

    /// Newest discovery portfolio (path + trades) for a combo, produced after
    /// `since_ms`. Returns (portfolio_json, trades_json).
    async fn collect_portfolio(
        &self,
        symbol: &str,
        base_tf: &str,
        since_ms: i64,
    ) -> Option<(String, Option<String>)> {
        let r = self.http.get(format!("{}/portfolios/list", self.base)).send().await.ok()?;
        let v: serde_json::Value = r.json().await.ok()?;
        let list = v.get("portfolios").and_then(|x| x.as_array())?;
        let mut best: Option<(i64, String)> = None;
        for e in list {
            let sym = e.get("symbol").and_then(|x| x.as_str()).unwrap_or("");
            let tf = e.get("baseTf").and_then(|x| x.as_str()).unwrap_or("");
            let modt = e.get("modifiedMs").and_then(|x| x.as_i64()).unwrap_or(0);
            let path = e.get("path").and_then(|x| x.as_str()).unwrap_or("");
            if sym.eq_ignore_ascii_case(symbol)
                && tf.eq_ignore_ascii_case(base_tf)
                && modt >= since_ms
                && !path.is_empty()
                && best.as_ref().map(|(m, _)| modt > *m).unwrap_or(true)
            {
                best = Some((modt, path.to_string()));
            }
        }
        let (_, path) = best?;
        let portfolio_json = std::fs::read_to_string(&path).ok()?;
        let trades_path = path.replace(".live_portfolio.json", ".trades.json");
        let trades_json = std::fs::read_to_string(&trades_path).ok();
        Some((portfolio_json, trades_json))
    }
}

// ── Coordinator: QUIC protocol handler ───────────────────────────────────────

#[derive(Debug, Clone)]
struct MeshProto {
    app: AppClient,
}

impl ProtocolHandler for MeshProto {
    async fn accept(&self, connection: iroh::endpoint::Connection) -> Result<(), AcceptError> {
        let remote = connection.remote_id();
        // One request-response per bi-stream; loop while the peer opens streams.
        loop {
            let (mut send, mut recv) = match connection.accept_bi().await {
                Ok(s) => s,
                Err(_) => break,
            };
            let req_bytes = recv.read_to_end(1 << 20).await.map_err(AcceptError::from_err)?;
            let resp = match serde_json::from_slice::<MeshReq>(&req_bytes) {
                Ok(MeshReq::GetJob { worker }) => {
                    let job = self.app.next_job(&worker).await;
                    if job.is_some() {
                        tracing::info!(peer = %remote, "leased a job to a peer worker");
                    }
                    MeshResp::Job(job)
                }
                Ok(MeshReq::Submit { symbol, base_tf, portfolio_json, trades_json }) => {
                    let (ok, msg) = self
                        .app
                        .submit(&remote.to_string(), &symbol, &base_tf, &portfolio_json, trades_json.as_deref())
                        .await;
                    tracing::info!(peer = %remote, %symbol, ok, "received a result from a peer worker");
                    MeshResp::SubmitAck { ok, msg }
                }
                Err(e) => MeshResp::Err(format!("bad request: {e}")),
            };
            let out = serde_json::to_vec(&resp).unwrap_or_default();
            send.write_all(&out).await.map_err(AcceptError::from_err)?;
            send.finish().map_err(AcceptError::from_err)?;
        }
        connection.closed().await;
        Ok(())
    }
}

// ── Worker: run one job from a coordinator peer ──────────────────────────────

async fn rpc(
    endpoint: &Endpoint,
    coordinator: EndpointId,
    req: &MeshReq,
) -> Result<MeshResp> {
    let conn = endpoint.connect(coordinator, MESH_ALPN).await.context("connect")?;
    let (mut send, mut recv) = conn.open_bi().await.context("open_bi")?;
    send.write_all(&serde_json::to_vec(req)?).await?;
    send.finish()?;
    let bytes = recv.read_to_end(1 << 20).await?;
    conn.close(0u32.into(), b"done");
    Ok(serde_json::from_slice(&bytes)?)
}

/// Try to run exactly one job from `coordinator`. Returns true if work was done.
async fn run_one_job(
    endpoint: &Endpoint,
    app: &AppClient,
    my_id: &str,
    coordinator: EndpointId,
) -> Result<bool> {
    let job = match rpc(endpoint, coordinator, &MeshReq::GetJob { worker: my_id.into() }).await? {
        MeshResp::Job(Some(j)) => j,
        _ => return Ok(false),
    };
    tracing::info!(%job.symbol, %job.base_tf, wt = %job.work_type, coord = %coordinator, "claimed a job");

    let start_ms = now_secs() as i64 * 1000;
    if !app.start_engine(&job.work_type, &job.symbol, &job.base_tf).await {
        tracing::warn!("local engine refused to start the job; giving it back");
        return Ok(false);
    }

    // Wait for the engine to start, then finish (bounded).
    let mut waited = 0u64;
    let mut ever_ran = false;
    loop {
        tokio::time::sleep(Duration::from_secs(10)).await;
        waited += 10;
        let st = app.engine_state(&job.work_type).await;
        if st == "Running" {
            ever_ran = true;
        } else if ever_ran || waited > 120 {
            break;
        }
        if waited > 24 * 3600 {
            break; // hard cap
        }
    }

    if job.work_type == "training" {
        // Training runs locally; returning the model weights to the coordinator
        // needs iroh-blobs (documented next step). The compute offload is real.
        tracing::info!("training job ran locally; model-weight return is the iroh-blobs follow-up");
        return Ok(true);
    }

    // Discovery: collect the produced portfolio and submit it to the coordinator.
    match app.collect_portfolio(&job.symbol, &job.base_tf, start_ms).await {
        Some((pf, trades)) => {
            let resp = rpc(
                endpoint,
                coordinator,
                &MeshReq::Submit {
                    symbol: job.symbol.clone(),
                    base_tf: job.base_tf.clone(),
                    portfolio_json: pf,
                    trades_json: trades,
                },
            )
            .await?;
            match resp {
                MeshResp::SubmitAck { ok, msg } => {
                    tracing::info!(ok, %msg, "submitted result to coordinator")
                }
                other => tracing::warn!("unexpected submit response: {other:?}"),
            }
        }
        None => tracing::warn!("no portfolio artifact found after the run (empty result?)"),
    }
    Ok(true)
}

// ── main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "neoethos_mesh=info,iroh=warn,iroh_gossip=warn".into()),
        )
        .init();

    let args = parse_args();
    tracing::info!(app = %args.app_url, "NeoEthos Mesh — automatic P2P work distribution");

    let secret_key = load_or_create_secret(&args.data_dir)?;
    let my_id = secret_key.public();
    let my_id_str = my_id.to_string();
    tracing::info!(node_id = %my_id, "this node's permanent mesh address");

    let app = AppClient::new(args.app_url.clone());

    // Automatic connectivity (relay + n0 discovery) with the work-protocol ALPN.
    let endpoint = Endpoint::builder(presets::N0)
        .secret_key(secret_key)
        .alpns(vec![MESH_ALPN.to_vec()])
        .bind()
        .await
        .context("failed to bind the iroh endpoint")?;
    tracing::info!("online via the iroh relay network — reachable anywhere, no config needed");

    // Gossip discovery + the coordinator protocol handler, both on the Router.
    let gossip = Gossip::builder().spawn(endpoint.clone());
    let _router = iroh::protocol::Router::builder(endpoint.clone())
        .accept(iroh_gossip::ALPN, gossip.clone())
        .accept(MESH_ALPN, MeshProto { app: app.clone() })
        .spawn();

    let topic = gossip
        .subscribe(TopicId::from_bytes(ANNOUNCE_TOPIC), seed_nodes())
        .await
        .context("subscribe to rendezvous topic")?;
    let (sender, mut receiver) = topic.split();

    let peers: Peers = Arc::new(Mutex::new(HashMap::new()));

    // Announce loop — broadcast this node's REAL hardware so the swarm knows
    // its total capacity (cores / RAM / GPUs), plus liveness + has_work.
    {
        let app = app.clone();
        let id = my_id_str.clone();
        tokio::spawn(async move {
            // 20s keeps discovery responsive when a new peer joins between
            // ticks; the payload is a few hundred bytes, so bandwidth is a
            // non-issue. (Re-announce-on-NeighborUp is the Phase-D refinement.)
            let mut tick = tokio::time::interval(Duration::from_secs(20));
            loop {
                tick.tick().await;
                let (online, has_work) = app.status().await;
                let hw = app.hardware().await;
                let ann = Announce {
                    node_id: id.clone(),
                    cpu_cores: hw.cpu_cores,
                    ram_mb: hw.ram_mb,
                    gpu: hw.gpu,
                    work_types: vec!["discovery".into()],
                    app_online: online,
                    has_work,
                    ts: now_secs(),
                };
                if let Ok(json) = serde_json::to_vec(&ann) {
                    let _ = sender.broadcast(Bytes::from(json)).await;
                }
            }
        });
    }

    // Discovery loop — maintain the peer table.
    {
        let peers = peers.clone();
        tokio::spawn(async move {
            while let Some(event) = receiver.next().await {
                if let Ok(Event::Received(msg)) = event {
                    if let Ok(a) = serde_json::from_slice::<Announce>(&msg.content) {
                        if let Ok(id) = EndpointId::from_str(&a.node_id) {
                            peers.lock().await.insert(
                                id,
                                PeerInfo {
                                    cpu_cores: a.cpu_cores,
                                    ram_mb: a.ram_mb,
                                    gpu: a.gpu,
                                    app_online: a.app_online,
                                    has_work: a.has_work,
                                    last_seen: Instant::now(),
                                },
                            );
                            tracing::info!(peer = %a.node_id, cores = a.cpu_cores, ram_mb = a.ram_mb, gpu = a.gpu, has_work = a.has_work, "peer announce");
                        }
                    }
                }
            }
        });
    }

    // Swarm-capacity summary — the whole point: the swarm reported as ONE
    // machine. Aggregates this node + every live peer, logs it, and writes a
    // status file the app/UI can surface ("your swarm = N cores, M GB, K GPUs").
    {
        let peers = peers.clone();
        let app = app.clone();
        let me = my_id_str.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(30));
            loop {
                tick.tick().await;
                let mine = app.hardware().await;
                let (mut nodes, mut cores, mut ram_mb, mut gpus) = (1u64, mine.cpu_cores as u64, mine.ram_mb, mine.gpu as u64);
                {
                    let mut p = peers.lock().await;
                    p.retain(|_, v| v.last_seen.elapsed() < PEER_TTL);
                    for (_, v) in p.iter() {
                        nodes += 1;
                        cores += v.cpu_cores as u64;
                        ram_mb += v.ram_mb;
                        gpus += v.gpu as u64;
                    }
                }
                tracing::info!(
                    nodes, total_cores = cores, total_ram_gb = ram_mb / 1024, total_gpus = gpus,
                    "SWARM CAPACITY — the network as one machine"
                );
                let snapshot = serde_json::json!({
                    "nodes": nodes,
                    "totalCores": cores,
                    "totalRamGb": ram_mb as f64 / 1024.0,
                    "totalGpus": gpus,
                    "self": { "nodeId": me, "cores": mine.cpu_cores, "ramMb": mine.ram_mb, "gpu": mine.gpu },
                    "ts": now_secs(),
                });
                let _ = std::fs::write(
                    std::env::temp_dir().join("neoethos_mesh_swarm.json"),
                    snapshot.to_string(),
                );
            }
        });
    }

    // Worker loop — if the local engine is idle and a peer has work, run one job.
    {
        let peers = peers.clone();
        let endpoint = endpoint.clone();
        let app = app.clone();
        let me = my_id;
        let my_id_str2 = my_id_str.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(20)).await;
                // Prune stale peers.
                let coordinator = {
                    let mut p = peers.lock().await;
                    p.retain(|_, v| v.last_seen.elapsed() < PEER_TTL);
                    p.iter()
                        .find(|(id, v)| **id != me && v.app_online && v.has_work)
                        .map(|(id, _)| *id)
                };
                // Only take work when our own engine is idle (don't oversubscribe).
                let disc = app.engine_state("discovery").await;
                if disc == "Running" {
                    continue;
                }
                if let Some(coord) = coordinator {
                    match run_one_job(&endpoint, &app, &my_id_str2, coord).await {
                        Ok(true) => tracing::info!("completed a distributed job"),
                        Ok(false) => {}
                        Err(e) => tracing::debug!("job attempt failed: {e}"),
                    }
                }
            }
        });
    }

    tracing::info!("swarm joined — discovery work is distributed automatically. Ctrl-C to stop.");
    tokio::signal::ctrl_c().await.ok();
    tracing::info!("shutting down");
    endpoint.close().await;
    Ok(())
}
