//! Retirement → re-discovery trigger — the third leg of the Symbiotic-GP
//! retraining loop (Loginov & Heywood; operator directive 2026-07-02).
//!
//! Auto-cull already closes the NEGATIVE half of the feedback loop: a live
//! strategy that hits the loss criteria is stopped and its fingerprint is
//! permanently blacklisted. This module closes the POSITIVE half: the
//! retirement leaves a coverage gap on that (symbol, base_tf) — so queue a
//! fresh Discovery run to refill it.
//!
//! # What "blacklisted" means — closed 2026-08-10 (#219)
//!
//! This header once said the retired strategy is *"never selectable, never
//! re-discovered"* while only the first half was true. Both halves hold now:
//!
//! * **Selection is guarded.** `server::autonomous` and `app_services::federation`
//!   both call [`strategy_blacklist::is_blacklisted`] before a portfolio can go
//!   live, and `server::portfolios` hides retired ones from the listing.
//! * **Identity is the gene, not the file** ([`strategy_blacklist`], #218), so a
//!   re-discovered artifact describing the SAME rule is caught at selection even
//!   though its bytes differ.
//! * **Discovery now consults it too.** The identity moved down to
//!   `neoethos_core::strategy_identity`, which `neoethos-search` can see, and
//!   `neoethos_search::live_portfolio` drops any RETIRED RULE from the artifact
//!   the trader consumes — per gene, so a culled rule bundled with different
//!   company no longer slips through on an artifact-level hash. The set is read
//!   from this same `strategy_blacklist.json` at search startup
//!   (`install_search_runtime_overrides_from_settings`).
//!
//! What is still true and is NOT a defect: the GA can still SPEND time
//! re-deriving a retired rule inside a run — the gate is at promotion, not at
//! mutation — and it says so loudly when it happens.
//!
//! [`strategy_blacklist`]: crate::app_services::strategy_blacklist
//! [`strategy_blacklist::is_blacklisted`]: crate::app_services::strategy_blacklist::is_blacklisted
//!
//! Design: the live-engine loop only PUSHES a request into a process-global
//! queue (it has no access to `AppApiState`); a watcher spawned at server
//! startup drains the queue through the shared typed Discovery start — every
//! validation, preflight and process-wide execution gate applies without a
//! JSON/HTTP round trip. Gated by `system.auto_rediscover_on_cull`
//! (Settings toggle, default ON). Fail-soft everywhere: a full engine queue
//! retries on the next tick, a permanent failure (e.g. no data) drops the
//! request with a WARN instead of looping forever.

use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

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
            let Some((symbol, base_tf)) = next else {
                continue;
            };

            let base_timeframe = match base_tf
                .trim()
                .to_uppercase()
                .parse::<neoethos_data::CanonicalTimeframe>()
            {
                Ok(timeframe) => timeframe,
                Err(error) => {
                    tracing::warn!(
                        target: "neoethos_app::rediscovery",
                        %symbol,
                        %base_tf,
                        error = %error,
                        "rediscovery base timeframe is invalid — dropping request"
                    );
                    if let Ok(mut q) = queue().lock() {
                        q.pop_front();
                    }
                    continue;
                }
            };
            let start = crate::server::engines_control::start_typed_discovery_execution_v1(
                state.clone(),
                crate::server::engines_control::TypedDiscoveryExecutionIntentV1 {
                    symbol: symbol.clone(),
                    base_timeframe,
                    higher_timeframes:
                        crate::server::engines_control::TypedHigherTimeframePolicyV1::Configured,
                    overrides:
                        crate::server::engines_control::TypedDiscoveryOverridesV1::default(),
                    settings_gate: crate::server::engines_control::TypedDiscoverySettingsGateV1::RequireAutoRediscoveryEnabled,
                    dataset_policy: crate::server::engines_control::TypedDiscoveryDatasetPolicyV1::Current,
                    training_after_success: true,
                },
            );
            match start {
                Ok(handle) => {
                    crate::server::engines_control::detach_typed_legacy_execution_observer_v1(
                        state.clone(),
                        handle,
                    );
                    tracing::info!(
                        target: "neoethos_app::rediscovery",
                        %symbol, %base_tf,
                        "rediscovery started — refilling the slot the retired strategy left"
                    );
                    if let Ok(mut q) = queue().lock() {
                        q.pop_front();
                    }
                }
                Err(crate::server::engines_control::TypedLegacyExecutionStartErrorV1::Busy(
                    busy,
                )) => {
                    tracing::debug!(
                        target: "neoethos_app::rediscovery",
                        %symbol,
                        %base_tf,
                        requested = %busy.requested(),
                        active = %busy.active(),
                        "discovery engine busy — will retry"
                    );
                }
                Err(error) => {
                    tracing::warn!(
                        target: "neoethos_app::rediscovery",
                        %symbol,
                        %base_tf,
                        error = %error,
                        "rediscovery start failed — dropping request"
                    );
                    if let Ok(mut q) = queue().lock() {
                        q.pop_front();
                    }
                }
            }
        }
    });
}
