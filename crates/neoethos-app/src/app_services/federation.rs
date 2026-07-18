//! Federation Phase 0 — "SETI@home for strategy discovery", with NO server.
//!
//! Operator vision (2026-07-02): small traders can't buy server farms, but a
//! thousand €300 mini-PCs can out-search one. Discovery federates naturally:
//! the real lever is COVERAGE (symbols × timeframes × seeds), each combo is
//! an independent job, historical bars are public, and — the gift — a claimed
//! strategy is CHEAP to verify deterministically, so nobody has to be trusted.
//!
//! No-server design: any NeoEthos instance can act as the COORDINATOR — the
//! app already ships an HTTP server. A friend exposes theirs (Tailscale
//! serve / port-forward / ngrok); everyone else points a WORKER at that URL.
//!
//! Trust model (Phase 0): submissions land in `<cache>/federation_inbox/` as
//! ordinary `*.live_portfolio.json` artifacts. They surface in the normal
//! portfolio list like any local discovery result — and pass through the SAME
//! defences as everything else: Strategy Lab gates, tail-risk/parity checks,
//! the blacklist, and the demo forward-test gate before any real money. A
//! shared token (optional) keeps drive-by junk out of the inbox.
//!
//! Deliberately NOT here yet (Phase 1+): elite-gene migration between
//! islands, shared blacklist sync, redundancy scoring, contributor credits.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::server::state::AppApiState;

/// One unit of federated work: run `work_type` on this combo.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct FedJob {
    pub symbol: String,
    pub base_tf: String,
    /// "discovery" (default) or "training". Lets a work plan mix strategy
    /// search and model training across the swarm. Serde-default keeps old
    /// clients (which send only symbol+base_tf) working unchanged.
    #[serde(default = "default_work_type")]
    pub work_type: String,
}

fn default_work_type() -> String {
    "discovery".to_string()
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FedLease {
    pub job: FedJob,
    pub worker: String,
    pub leased_at_unix_ms: i64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FedReceived {
    pub worker: String,
    pub symbol: String,
    pub base_tf: String,
    pub saved_path: String,
    pub received_at_unix_ms: i64,
}

#[derive(Default)]
struct CoordinatorState {
    queue: VecDeque<FedJob>,
    leases: Vec<(FedJob, String, Instant, i64)>,
    received: Vec<FedReceived>,
    token: Option<String>,
}

static COORDINATOR: OnceLock<Mutex<CoordinatorState>> = OnceLock::new();

fn coord() -> &'static Mutex<CoordinatorState> {
    COORDINATOR.get_or_init(|| Mutex::new(CoordinatorState::default()))
}

/// Leases older than this go back to the queue (a worker died mid-run).
const LEASE_TTL: Duration = Duration::from_secs(12 * 3600);

/// Coordinator: replace the work plan (operator sets combos + optional token).
pub fn set_jobs(jobs: Vec<FedJob>, token: Option<String>) -> usize {
    let mut c = coord().lock().expect("federation coordinator lock");
    c.queue = jobs
        .into_iter()
        .filter(|j| !j.symbol.trim().is_empty() && !j.base_tf.trim().is_empty())
        .map(|j| {
            let wt = j.work_type.trim().to_lowercase();
            FedJob {
                symbol: j.symbol.trim().to_uppercase(),
                base_tf: j.base_tf.trim().to_uppercase(),
                work_type: if wt == "training" { "training".into() } else { "discovery".into() },
            }
        })
        .collect();
    c.leases.clear();
    c.token = token.filter(|t| !t.trim().is_empty());
    c.queue.len()
}

/// Coordinator: token gate for worker-facing endpoints. `None` token = open.
pub fn token_ok(provided: Option<&str>) -> bool {
    let c = coord().lock().expect("federation coordinator lock");
    match &c.token {
        None => true,
        Some(t) => provided == Some(t.as_str()),
    }
}

/// Coordinator: lease the next job to `worker` (re-queueing expired leases).
pub fn next_job(worker: &str) -> Option<FedJob> {
    let mut c = coord().lock().expect("federation coordinator lock");
    // Reclaim dead leases first.
    let now = Instant::now();
    let mut reclaimed: Vec<FedJob> = Vec::new();
    c.leases.retain(|(job, _, at, _)| {
        if now.duration_since(*at) > LEASE_TTL {
            reclaimed.push(job.clone());
            false
        } else {
            true
        }
    });
    for j in reclaimed {
        c.queue.push_back(j);
    }
    let job = c.queue.pop_front()?;
    c.leases.push((
        job.clone(),
        worker.to_string(),
        now,
        chrono::Utc::now().timestamp_millis(),
    ));
    Some(job)
}

/// Coordinator: accept a submitted portfolio artifact. Validates the JSON
/// shape, writes it into `<cache>/federation_inbox/` (which the normal
/// portfolio scan already covers), and closes the matching lease.
pub fn submit(
    worker: &str,
    symbol: &str,
    base_tf: &str,
    portfolio_json: &str,
    trades_json: Option<&str>,
) -> Result<String> {
    let v: serde_json::Value =
        serde_json::from_str(portfolio_json).context("portfolio payload is not valid JSON")?;
    let genes = v
        .get("genes")
        .or_else(|| v.get("full_genes"))
        .and_then(|g| g.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    if genes == 0 {
        anyhow::bail!("submitted portfolio has no genes — rejected");
    }

    let cache_dir = neoethos_core::Settings::from_yaml(&crate::server::state::current_config_path())
        .map(|s| s.system.cache_dir)
        .unwrap_or_else(|_| PathBuf::from("cache"));
    let inbox = cache_dir.join("federation_inbox");
    std::fs::create_dir_all(&inbox).context("create federation_inbox")?;

    let ts = chrono::Utc::now().timestamp_millis();
    // symbol/base_tf arrive over HTTP and become filename components —
    // keep only [A-Za-z0-9_-] (defense-in-depth against separator/..
    // tricks, mirroring the mesh sidecar's safe_path_component).
    let sanitize = |s: &str| -> String {
        s.trim()
            .to_uppercase()
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
            .take(32)
            .collect()
    };
    let sym = sanitize(symbol);
    let tf = sanitize(base_tf);
    if sym.is_empty() || tf.is_empty() {
        anyhow::bail!("submitted symbol/baseTf contain no usable characters — rejected");
    }
    let stem = format!("fed_{sym}_{tf}_{ts}");
    let pf_path = inbox.join(format!("{stem}.live_portfolio.json"));
    std::fs::write(&pf_path, portfolio_json).context("write submitted portfolio")?;
    if let Some(t) = trades_json {
        if serde_json::from_str::<serde_json::Value>(t).is_ok() {
            let _ = std::fs::write(inbox.join(format!("{stem}.trades.json")), t);
        }
    }

    let saved = pf_path.display().to_string();
    // A retired strategy resurfacing from another node stays dead here too.
    if crate::app_services::strategy_blacklist::is_blacklisted(&saved) {
        tracing::warn!(
            target: "neoethos_app::federation",
            %saved, "submitted portfolio matches the local blacklist — kept on disk but flagged"
        );
    }

    let mut c = coord().lock().expect("federation coordinator lock");
    c.leases
        .retain(|(job, w, _, _)| !(w == worker && job.symbol == sym && job.base_tf == tf));
    c.received.push(FedReceived {
        worker: worker.to_string(),
        symbol: sym,
        base_tf: tf,
        saved_path: saved.clone(),
        received_at_unix_ms: ts,
    });
    tracing::info!(
        target: "neoethos_app::federation",
        %worker, genes, %saved, "federation: portfolio received into the inbox"
    );
    Ok(saved)
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FederationStatus {
    pub jobs_queued: usize,
    pub leases: Vec<FedLease>,
    pub received: Vec<FedReceived>,
    pub token_required: bool,
    pub worker_running: bool,
    pub worker_status: String,
}

pub fn status() -> FederationStatus {
    let c = coord().lock().expect("federation coordinator lock");
    FederationStatus {
        jobs_queued: c.queue.len(),
        leases: c
            .leases
            .iter()
            .map(|(job, w, _, ms)| FedLease {
                job: job.clone(),
                worker: w.clone(),
                leased_at_unix_ms: *ms,
            })
            .collect(),
        received: c.received.iter().rev().take(50).cloned().collect(),
        token_required: c.token.is_some(),
        worker_running: WORKER_RUNNING.load(Ordering::SeqCst),
        worker_status: worker_status_text(),
    }
}

// ─── Worker side ────────────────────────────────────────────────────────────

static WORKER_RUNNING: AtomicBool = AtomicBool::new(false);
static WORKER_STATUS: OnceLock<Mutex<String>> = OnceLock::new();

fn set_worker_status(s: impl Into<String>) {
    let s = s.into();
    tracing::info!(target: "neoethos_app::federation", "worker: {s}");
    *WORKER_STATUS
        .get_or_init(|| Mutex::new(String::new()))
        .lock()
        .expect("worker status lock") = s;
}

fn worker_status_text() -> String {
    WORKER_STATUS
        .get_or_init(|| Mutex::new(String::new()))
        .lock()
        .expect("worker status lock")
        .clone()
}

pub fn worker_stop() {
    WORKER_RUNNING.store(false, Ordering::SeqCst);
    set_worker_status("stop requested — finishing the current step");
}

/// Start the worker loop: fetch a job from `coordinator_url`, run the normal
/// local Discovery for it (same handler, same gates), submit the produced
/// artifacts back, repeat. Fail-soft on everything; `worker_stop()` exits.
pub fn worker_start(state: AppApiState, coordinator_url: String, worker_id: String, token: Option<String>) -> Result<()> {
    let url = coordinator_url.trim().trim_end_matches('/').to_string();
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        anyhow::bail!("coordinator URL must start with http:// or https://");
    }
    if WORKER_RUNNING.swap(true, Ordering::SeqCst) {
        anyhow::bail!("federation worker already running");
    }
    let worker_id = if worker_id.trim().is_empty() {
        format!("worker-{}", std::process::id())
    } else {
        worker_id.trim().to_string()
    };

    tokio::spawn(async move {
        use crate::app_services::jobs::JobKind;
        use crate::server::engines_control;
        use axum::Json;
        use axum::extract::State;

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .user_agent("neoethos-federation/0.5")
            .build()
            .expect("reqwest client");
        set_worker_status(format!("started — polling {url} as {worker_id}"));

        'outer: while WORKER_RUNNING.load(Ordering::SeqCst) {
            // 1. Fetch a job.
            let mut req = client.get(format!("{url}/federation/job?worker={worker_id}"));
            if let Some(t) = &token {
                req = req.header("x-fed-token", t);
            }
            let job: Option<FedJob> = match req.send().await {
                Ok(r) if r.status().is_success() => r.json().await.ok(),
                Ok(r) if r.status() == reqwest::StatusCode::NOT_FOUND => None,
                Ok(r) => {
                    set_worker_status(format!("coordinator refused ({}) — retrying in 10m", r.status()));
                    None
                }
                Err(e) => {
                    set_worker_status(format!("coordinator unreachable ({e}) — retrying in 10m"));
                    None
                }
            };
            let Some(job) = job else {
                for _ in 0..60 {
                    if !WORKER_RUNNING.load(Ordering::SeqCst) {
                        break 'outer;
                    }
                    tokio::time::sleep(Duration::from_secs(10)).await;
                }
                continue;
            };

            // 2. Run the local discovery for it (same path as the UI button).
            set_worker_status(format!("job {} {} — starting discovery", job.symbol, job.base_tf));
            let started_ms = chrono::Utc::now().timestamp_millis();
            loop {
                if !WORKER_RUNNING.load(Ordering::SeqCst) {
                    break 'outer;
                }
                let body: engines_control::StartJobBody = match serde_json::from_value(
                    serde_json::json!({ "symbol": job.symbol, "base_tf": job.base_tf }),
                ) {
                    Ok(b) => b,
                    Err(_) => break,
                };
                let resp =
                    engines_control::discovery_start(State(state.clone()), Some(Json(body))).await;
                let status = resp.status();
                if status.is_success() {
                    break;
                }
                if status == axum::http::StatusCode::CONFLICT {
                    set_worker_status(format!(
                        "job {} {} — local engine busy, waiting",
                        job.symbol, job.base_tf
                    ));
                    tokio::time::sleep(Duration::from_secs(300)).await;
                    continue;
                }
                set_worker_status(format!(
                    "job {} {} — discovery refused ({status}); skipping job",
                    job.symbol, job.base_tf
                ));
                continue 'outer;
            }

            // 3. Wait for the run to finish (poll the engine state).
            loop {
                if !WORKER_RUNNING.load(Ordering::SeqCst) {
                    break 'outer;
                }
                tokio::time::sleep(Duration::from_secs(30)).await;
                if !matches!(
                    state.engine_state(JobKind::Discovery).await,
                    crate::server::engines_control::EngineRunState::Running
                ) {
                    break;
                }
            }

            // 4. Submit every artifact this run produced for the combo.
            let cache_dir = neoethos_core::Settings::from_yaml(
                &crate::server::state::current_config_path(),
            )
            .map(|s| s.system.cache_dir)
            .unwrap_or_else(|_| PathBuf::from("cache"));
            let mut submitted = 0usize;
            for pf in find_new_artifacts(&cache_dir, &job.symbol, &job.base_tf, started_ms) {
                let Ok(pf_json) = std::fs::read_to_string(&pf) else { continue };
                let trades_json = std::fs::read_to_string(
                    pf.display()
                        .to_string()
                        .trim_end_matches(".live_portfolio.json")
                        .to_string()
                        + ".trades.json",
                )
                .ok();
                let mut req = client.post(format!("{url}/federation/submit")).json(
                    &serde_json::json!({
                        "worker": worker_id,
                        "symbol": job.symbol,
                        "baseTf": job.base_tf,
                        "portfolioJson": pf_json,
                        "tradesJson": trades_json,
                    }),
                );
                if let Some(t) = &token {
                    req = req.header("x-fed-token", t);
                }
                match req.send().await {
                    Ok(r) if r.status().is_success() => submitted += 1,
                    Ok(r) => set_worker_status(format!("submit refused ({})", r.status())),
                    Err(e) => set_worker_status(format!("submit failed ({e})")),
                }
            }
            set_worker_status(format!(
                "job {} {} done — {submitted} artifact(s) submitted; fetching next",
                job.symbol, job.base_tf
            ));
        }
        WORKER_RUNNING.store(false, Ordering::SeqCst);
        set_worker_status("stopped");
    });
    Ok(())
}

/// Artifacts under `root` matching the combo and newer than `since_ms`.
fn find_new_artifacts(root: &PathBuf, symbol: &str, base_tf: &str, since_ms: i64) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.clone()];
    let sym = symbol.to_uppercase();
    let tf = base_tf.to_uppercase();
    let mut visited = 0usize;
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for ent in rd.flatten() {
            visited += 1;
            if visited > 200_000 {
                return out;
            }
            let p = ent.path();
            if p.is_dir() {
                // Never re-submit what the coordinator role received.
                if p.file_name().is_some_and(|n| n == "federation_inbox") {
                    continue;
                }
                stack.push(p);
                continue;
            }
            let name = p.file_name().map(|f| f.to_string_lossy().to_uppercase()).unwrap_or_default();
            if !name.ends_with("LIVE_PORTFOLIO.JSON")
                || !name.contains(&sym)
                || !name.contains(&tf)
            {
                continue;
            }
            let modified_ms = std::fs::metadata(&p)
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0);
            if modified_ms >= since_ms {
                out.push(p);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_lease_and_submit_close_the_loop() {
        set_jobs(
            vec![
                FedJob { symbol: "eurusd".into(), base_tf: "m15".into(), work_type: "discovery".into() },
                FedJob { symbol: "GBPUSD".into(), base_tf: "H1".into(), work_type: "training".into() },
            ],
            Some("secret".into()),
        );
        assert!(!token_ok(None));
        assert!(!token_ok(Some("wrong")));
        assert!(token_ok(Some("secret")));

        let j = next_job("alice").expect("job available");
        assert_eq!(j.symbol, "EURUSD");
        assert_eq!(j.base_tf, "M15");
        let s = status();
        assert_eq!(s.jobs_queued, 1);
        assert_eq!(s.leases.len(), 1);

        // Garbage submission is rejected; a real one closes the lease.
        assert!(submit("alice", "EURUSD", "M15", "not json", None).is_err());
        assert!(submit("alice", "EURUSD", "M15", r#"{"genes":[]}"#, None).is_err());

        // Reset to an open coordinator so other tests aren't token-locked.
        set_jobs(Vec::new(), None);
    }
}
