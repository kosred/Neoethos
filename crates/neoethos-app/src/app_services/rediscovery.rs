//! Retirement → re-discovery trigger — the third leg of the Symbiotic-GP
//! retraining loop (Loginov & Heywood; operator directive 2026-07-02).
//!
//! Auto-cull already closes the NEGATIVE half of the feedback loop: a live
//! strategy that hits the loss criteria is stopped and its fingerprint is
//! permanently blacklisted (never selectable, never re-discovered). This
//! module closes the POSITIVE half: the retirement leaves a coverage gap on
//! that (symbol, base_tf) — so queue a fresh Discovery run to refill it.
//!
//! Design: the live-engine loop only PUSHES a request into a process-global
//! queue (it has no access to `AppApiState`); a watcher spawned at server
//! startup drains the queue through the SAME `engines_control::discovery_start`
//! handler the UI and the Supervisor use — every validation, preflight and
//! single-engine gate applies. Gated by `system.auto_rediscover_on_cull`
//! (Settings toggle, default ON). Fail-soft everywhere: a full engine queue
//! retries on the next tick, a permanent failure (e.g. no data) drops the
//! request with a WARN instead of looping forever.

use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

use axum::Json;
use axum::extract::State;

use crate::server::state::AppApiState;

/// (symbol, base_tf) pairs waiting for a rediscovery slot.
static QUEUE: OnceLock<Mutex<VecDeque<(String, String)>>> = OnceLock::new();

fn queue() -> &'static Mutex<VecDeque<(String, String)>> {
    QUEUE.get_or_init(|| Mutex::new(VecDeque::new()))
}

/// Called from the live-engine auto-cull path after a strategy is retired.
/// Cheap, non-blocking, deduplicated — safe from any thread.
pub fn request(symbol: String, base_tf: String) {
    let Ok(mut q) = queue().lock() else { return };
    if q.iter().any(|(s, t)| *s == symbol && *t == base_tf) {
        return; // already queued — one run refills the gap for all culls on the combo
    }
    tracing::info!(
        target: "neoethos_app::rediscovery",
        %symbol, %base_tf,
        "auto-cull retirement → queueing rediscovery for the gap"
    );
    q.push_back((symbol, base_tf));
}

/// Spawn the queue drainer. One instance per process, started alongside the
/// supervisor heartbeat.
pub fn spawn(state: AppApiState) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(60));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            let next = {
                let Ok(q) = queue().lock() else { continue };
                q.front().cloned()
            };
            let Some((symbol, base_tf)) = next else { continue };

            // Settings gate — read fresh each tick so the toggle applies live.
            let enabled = neoethos_core::Settings::from_yaml(
                &crate::server::state::current_config_path(),
            )
            .map(|s| s.system.auto_rediscover_on_cull)
            .unwrap_or(true);
            if !enabled {
                tracing::info!(
                    target: "neoethos_app::rediscovery",
                    %symbol, %base_tf,
                    "auto_rediscover_on_cull is OFF in Settings — dropping queued rediscovery"
                );
                if let Ok(mut q) = queue().lock() {
                    q.pop_front();
                }
                continue;
            }

            let body: crate::server::engines_control::StartJobBody =
                match serde_json::from_value(serde_json::json!({
                    "symbol": symbol,
                    "base_tf": base_tf,
                })) {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::warn!(
                            target: "neoethos_app::rediscovery",
                            error = %e, "malformed rediscovery request — dropping"
                        );
                        if let Ok(mut q) = queue().lock() {
                            q.pop_front();
                        }
                        continue;
                    }
                };
            let resp = crate::server::engines_control::discovery_start(
                State(state.clone()),
                Some(Json(body)),
            )
            .await;
            let status = resp.status();
            if status.is_success() {
                tracing::info!(
                    target: "neoethos_app::rediscovery",
                    %symbol, %base_tf,
                    "rediscovery started — refilling the slot the retired strategy left"
                );
                if let Ok(mut q) = queue().lock() {
                    q.pop_front();
                }
            } else if status == axum::http::StatusCode::CONFLICT {
                // Discovery (or its training auto-chain) is busy — keep the
                // request queued and retry on a later tick.
                tracing::debug!(
                    target: "neoethos_app::rediscovery",
                    %symbol, %base_tf, "discovery engine busy — will retry"
                );
            } else {
                // Permanent-looking failure (no data for the combo, bad config…):
                // drop instead of retry-looping; the WARN tells the operator why.
                tracing::warn!(
                    target: "neoethos_app::rediscovery",
                    %symbol, %base_tf, status = %status,
                    "rediscovery start failed — dropping request (fix the cause and \
                     start Discovery manually if still wanted)"
                );
                if let Ok(mut q) = queue().lock() {
                    q.pop_front();
                }
            }
        }
    });
}
