//! Margin-call watchdog — the reader the broker's margin-call feed never had.
//!
//! # What this closes (#238, audit §2.11)
//!
//! `ctrader_messages::build_margin_call_list_request` shipped in the
//! 2026-06-10 API-completeness pass and, until today, had **zero callers**
//! outside its own unit test. Every breaker in `live_trading` keys off the
//! account BALANCE and REALISED P&L (`live_trading.rs:1510`, `:1532`). An
//! **unrealised** margin emergency — equity falling toward the used margin
//! while every position is still open, which is how an account is actually
//! liquidated — reached the operator only if he happened to be looking at
//! cTrader's own platform at the time.
//!
//! # What was already there, and what was not
//!
//! Re-verified before writing this module, because the audit's framing was
//! only half right:
//!
//! * The `ProtoOAMarginCallTriggerEvent` push **already arrives** on the spot
//!   streamer's socket and is already recognised
//!   (`live_spots_streamer.rs:617-625`). It logs `warn!` and **continues**.
//!   So the event was never invisible — it was simply never *acted on*.
//! * The `ProtoOAMarginCallListReq` **poll** genuinely had no caller, and with
//!   it no way to know the account's margin level between push events (or when
//!   the streamer is not running at all — it is best-effort at startup and
//!   silently absent when credentials are missing, `main.rs:428-440`).
//!
//! This module supplies the missing half: an independent poll that computes the
//! margin level itself and does not depend on any push event arriving, plus the
//! halt those events should route into.
//!
//! # The halt
//!
//! Two things happen when a margin call is detected, and they are deliberately
//! different in scope:
//!
//! 1. **A process-wide sticky halt** ([`active_halt`]) that
//!    `broker_api::prepare_new_order` consults, so **no route** — autopilot,
//!    manual `POST /orders`, `POST /orders/pending`, or the MCP sidecar, all of
//!    which funnel through that one function — can open a new position.
//!    Closing, cancelling and amending stops remain permitted: those reduce
//!    exposure.
//! 2. **The persisted Risky-Mode kill switch**
//!    (`risky_mode_persistence::record_kill_switch_trip`), which is the
//!    `HardwareConnLoss`-class halt the audit asked for. It survives an app
//!    restart, is rendered by the Risk screen (`server/risk.rs:231`) and is
//!    consulted by `live_trading.rs:1599`.
//!
//! **Known limitation, stated rather than hidden:** the persisted halt in (2)
//! is read by `live_trading` only under `trading_mode_risky`
//! (`live_trading.rs:1597`). In prop-firm mode it does not by itself stop the
//! engine. That is why (1) exists and is enforced at the broker boundary
//! instead of inside one engine's loop — (1) is mode-independent.
//!
//! # Clearing it
//!
//! The sticky halt is process-local and clears on backend restart, or via
//! [`clear_halt`]. It is deliberately NOT persisted: a false positive must
//! never be able to lock the operator out of his own account across restarts.
//! The 24 h Risky-Mode cooldown that (2) starts is persisted and auto-re-arms
//! through the existing bridge poll.
//!
//! # Fail-closed, and no silent drops
//!
//! * A margin-call *response we cannot parse* halts. "We asked and could not
//!   understand the answer" is not "everything is fine".
//! * A run of [`MAX_CONSECUTIVE_POLL_FAILURES`] consecutive poll failures halts
//!   — that is the connection-loss condition the `HardwareConnLoss` tier is
//!   named after.
//! * Every poll that fails, times out, or returns a payload with rows this
//!   process could not use is COUNTED and logged with its reason. The counters
//!   are readable via [`poll_counters`].
//! * A broker that reports **no** configured threshold is logged loudly and
//!   does NOT halt: with nothing to compare against there is no breach to
//!   detect, and halting on "the broker configured no margin call" would fire
//!   permanently on accounts where that is simply true.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use crate::app_services::broker_api::{MarginStatus, fetch_margin_status_blocking};

/// How often to ask the broker. Deliberately not a config knob yet: this is a
/// safety watchdog and the cost is one WSS session per interval. 60 s is short
/// enough that a deteriorating account is caught long before liquidation on any
/// realistic timeframe this system trades (M3 and up) and long enough that it
/// is invisible next to the bridge's own polling.
pub const POLL_INTERVAL: Duration = Duration::from_secs(60);

/// Consecutive failed polls that escalate to a halt. Three at 60 s means the
/// broker has been unreachable for ~3 minutes with (possibly) open positions.
/// Fewer would fire on ordinary network blips; more would be a lie about what
/// "watchdog" means.
pub const MAX_CONSECUTIVE_POLL_FAILURES: u64 = 3;

/// Why trading is halted. Carried so the refusal message names the cause
/// instead of saying "halted".
#[derive(Debug, Clone, PartialEq)]
pub enum HaltReason {
    /// The broker's configured margin-call threshold has been reached.
    MarginCall {
        margin_level_pct: f64,
        threshold_pct: f64,
        equity: f64,
        used_margin: f64,
    },
    /// The broker answered, but this process could not understand the answer.
    /// Fail-closed: an unparseable margin-call reply is treated as a call.
    UnparseableResponse { detail: String },
    /// The broker could not be reached for [`MAX_CONSECUTIVE_POLL_FAILURES`]
    /// consecutive polls. This is the `HardwareConnLoss` condition.
    BrokerUnreachable {
        consecutive_failures: u64,
        last_error: String,
    },
    /// Routed in from the streaming socket's `ProtoOAMarginCallTriggerEvent`
    /// or `ProtoOAAccountDisconnectEvent`. See [`record_push_event_halt`].
    PushEvent { detail: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Halt {
    pub reason: HaltReason,
    pub at_unix_ms: i64,
}

impl Halt {
    /// One-line human description used verbatim in the order-refusal error the
    /// operator sees in the UI.
    pub fn describe(&self) -> String {
        match &self.reason {
            HaltReason::MarginCall {
                margin_level_pct,
                threshold_pct,
                equity,
                used_margin,
            } => format!(
                "the broker reports this account is at MARGIN CALL — margin level \
                 {margin_level_pct:.1}% has reached the configured threshold \
                 {threshold_pct:.1}% (equity {equity:.2} against {used_margin:.2} used margin)"
            ),
            HaltReason::UnparseableResponse { detail } => format!(
                "the broker's margin-call status could not be read, so it is UNKNOWN \
                 whether this account is in margin call, and it is treated as if it is \
                 ({detail})"
            ),
            HaltReason::BrokerUnreachable {
                consecutive_failures,
                last_error,
            } => format!(
                "the broker has been unreachable for {consecutive_failures} consecutive \
                 margin-status polls, so the account's margin level is UNKNOWN \
                 (last error: {last_error})"
            ),
            HaltReason::PushEvent { detail } => {
                format!("the broker pushed a live risk event: {detail}")
            }
        }
    }
}

static HALTED: AtomicBool = AtomicBool::new(false);

fn halt_slot() -> &'static Mutex<Option<Halt>> {
    static SLOT: OnceLock<Mutex<Option<Halt>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

// ── No silent drops: every poll outcome is counted ─────────────────────────
static POLLS_ATTEMPTED: AtomicU64 = AtomicU64::new(0);
static POLLS_FAILED: AtomicU64 = AtomicU64::new(0);
static POLLS_UNUSABLE_ROWS: AtomicU64 = AtomicU64::new(0);
static POLLS_NO_THRESHOLD: AtomicU64 = AtomicU64::new(0);
static POLLS_MISSING_USED_MARGIN: AtomicU64 = AtomicU64::new(0);
static CONSECUTIVE_FAILURES: AtomicU64 = AtomicU64::new(0);

/// Snapshot of the watchdog's own bookkeeping. Every discard on this path is
/// in here; nothing is dropped without landing in one of these counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PollCounters {
    pub attempted: u64,
    pub failed: u64,
    /// Margin-call rows the broker sent that could not be turned into a usable
    /// threshold (summed across polls).
    pub unusable_threshold_rows: u64,
    /// Polls where the broker returned no usable threshold at all.
    pub polls_with_no_threshold: u64,
    /// Open positions the broker returned without a `usedMargin` field
    /// (summed across polls). Each one makes the computed margin level
    /// OPTIMISTIC.
    pub positions_missing_used_margin: u64,
    pub consecutive_failures: u64,
}

pub fn poll_counters() -> PollCounters {
    PollCounters {
        attempted: POLLS_ATTEMPTED.load(Ordering::Relaxed),
        failed: POLLS_FAILED.load(Ordering::Relaxed),
        unusable_threshold_rows: POLLS_UNUSABLE_ROWS.load(Ordering::Relaxed),
        polls_with_no_threshold: POLLS_NO_THRESHOLD.load(Ordering::Relaxed),
        positions_missing_used_margin: POLLS_MISSING_USED_MARGIN.load(Ordering::Relaxed),
        consecutive_failures: CONSECUTIVE_FAILURES.load(Ordering::Relaxed),
    }
}

/// The active halt, or `None`. Hot path — `broker_api::prepare_new_order`
/// calls this before every order, so the common case is one relaxed atomic
/// load and no lock.
pub fn active_halt() -> Option<Halt> {
    if !HALTED.load(Ordering::Acquire) {
        return None;
    }
    halt_slot().lock().ok().and_then(|g| g.clone())
}

/// Clear the sticky halt. Intended for an operator-initiated "I have reduced
/// exposure, resume" action; a backend restart does the same thing because the
/// flag is process-local by design.
pub fn clear_halt() {
    let previous = halt_slot().lock().ok().and_then(|mut g| g.take());
    HALTED.store(false, Ordering::Release);
    CONSECUTIVE_FAILURES.store(0, Ordering::Relaxed);
    tracing::warn!(
        target: "neoethos_app::margin_call",
        cleared = ?previous.map(|h| h.describe()),
        "margin-call halt CLEARED — new positions may be opened again"
    );
}

fn now_unix_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Arm the halt.
///
/// The stored reason is ALWAYS refreshed so `describe()` reports the current
/// numbers, but the `error!` line fires only when the KIND of halt changes (or
/// on the first arm). Comparison is by enum discriminant, not by value: a
/// margin level drifting 97.4% → 97.3% is the same emergency, and a watchdog
/// that screams at `error` every 60 s teaches the operator to filter it out —
/// at which point the next real one is invisible. The still-active case is
/// logged at `warn`.
fn arm(reason: HaltReason) {
    let halt = Halt {
        reason: reason.clone(),
        at_unix_ms: now_unix_ms(),
    };
    let already_same = match halt_slot().lock() {
        Ok(mut slot) => {
            let same = slot.as_ref().is_some_and(|h| {
                std::mem::discriminant(&h.reason) == std::mem::discriminant(&reason)
            });
            *slot = Some(halt.clone());
            same
        }
        Err(_) => {
            // A poisoned mutex must not be able to swallow a margin call.
            // The atomic below still halts trading; say so.
            tracing::error!(
                target: "neoethos_app::margin_call",
                "margin-call halt slot mutex is poisoned — trading is HALTED anyway, \
                 but the reason could not be recorded"
            );
            false
        }
    };
    HALTED.store(true, Ordering::Release);

    if already_same {
        tracing::warn!(
            target: "neoethos_app::margin_call",
            reason = %halt.describe(),
            "margin-call halt still active"
        );
        return;
    }

    tracing::error!(
        target: "neoethos_app::margin_call",
        reason = %halt.describe(),
        "TRADING HALTED — no new position will be opened by any route (autopilot, \
         manual orders, pending orders or the MCP sidecar). Closing positions, \
         cancelling resting orders and amending stops are still allowed. Restart the \
         backend once the account is healthy to clear this."
    );

    // Durable half: start the persisted 24 h Risky-Mode cooldown so the halt
    // survives a restart, appears on the Risk screen and is read by
    // `live_trading.rs:1599`. This is the `HardwareConnLoss`-class halt.
    if let Err(e) = crate::app_services::risky_mode_persistence::record_kill_switch_trip() {
        tracing::error!(
            target: "neoethos_app::margin_call",
            error = %e,
            "margin-call halt is ACTIVE in this process, but the persisted 24h \
             kill-switch cooldown could NOT be written — the halt will not survive \
             an app restart"
        );
    }
}

/// Route a broker PUSH event into the halt.
///
/// **This is the entry point `live_spots_streamer` should call.** That module
/// already receives and recognises both relevant events and today only logs
/// them:
///   * `live_spots_streamer.rs:617` — `ProtoOAMarginCallTriggerEvent` (2172)
///   * `live_spots_streamer.rs:580` — `ProtoOAAccountDisconnectEvent` (2164),
///     which is where `ctrader_messages::parse_account_disconnect_event`
///     belongs.
/// Both are one line each. This module does not own that file, so the poller
/// below reaches the same halt independently and the wiring is not blocked on
/// that edit landing.
pub fn record_push_event_halt(detail: impl Into<String>) {
    arm(HaltReason::PushEvent {
        detail: detail.into(),
    });
}

/// Run one poll. Public so a future `POST /risk/margin-check` route can force
/// one without waiting for the interval.
///
/// Returns the status when the broker answered and was understood. Errors are
/// already counted and logged before they are returned.
pub fn poll_once() -> anyhow::Result<MarginStatus> {
    // A fresh install with no broker configured has no positions and no
    // account to protect. Halting there would refuse orders on a machine that
    // has never traded, for a reason the operator could do nothing about. This
    // is NOT counted as a failed poll — there was nothing to ask.
    if !crate::app_services::broker_api::broker_credentials_configured() {
        tracing::debug!(
            target: "neoethos_app::margin_call",
            "no cTrader account configured — margin watchdog idle (nothing to watch)"
        );
        return Err(anyhow::anyhow!(
            "no cTrader account configured; margin-status poll skipped"
        ));
    }

    POLLS_ATTEMPTED.fetch_add(1, Ordering::Relaxed);

    let status = match fetch_margin_status_blocking() {
        Ok(s) => s,
        Err(e) => {
            POLLS_FAILED.fetch_add(1, Ordering::Relaxed);
            let detail = e.to_string();

            // "The broker replied and we could not read the reply" is NOT the
            // same failure as "we could not reach the broker", and it does not
            // get a failure budget: a wire-format change does not heal on the
            // next poll, and treating it as transient is a fail-OPEN.
            if detail.contains(crate::app_services::broker_api::MARGIN_STATUS_UNREADABLE_SENTINEL) {
                tracing::error!(
                    target: "neoethos_app::margin_call",
                    error = %detail,
                    "the broker's margin-status reply could not be read — halting on the \
                     FIRST occurrence rather than waiting out a failure budget"
                );
                arm(HaltReason::UnparseableResponse {
                    detail: detail.clone(),
                });
                return Err(e);
            }

            let consecutive = CONSECUTIVE_FAILURES.fetch_add(1, Ordering::Relaxed) + 1;
            tracing::warn!(
                target: "neoethos_app::margin_call",
                consecutive,
                budget = MAX_CONSECUTIVE_POLL_FAILURES,
                error = %detail,
                "margin-status poll FAILED — the account's margin level is unknown \
                 for this interval"
            );
            if consecutive >= MAX_CONSECUTIVE_POLL_FAILURES {
                arm(HaltReason::BrokerUnreachable {
                    consecutive_failures: consecutive,
                    last_error: detail,
                });
            }
            return Err(e);
        }
    };

    CONSECUTIVE_FAILURES.store(0, Ordering::Relaxed);

    // NO SILENT DROPS — account for everything the broker sent that this
    // process could not fully use, before deciding anything.
    if status.thresholds.unusable_rows > 0 {
        POLLS_UNUSABLE_ROWS.fetch_add(status.thresholds.unusable_rows as u64, Ordering::Relaxed);
        tracing::warn!(
            target: "neoethos_app::margin_call",
            unusable_rows = status.thresholds.unusable_rows,
            reasons = ?status.thresholds.unusable_reasons,
            usable_rows = status.thresholds.thresholds.len(),
            "the broker returned margin-call rows this build could not use — the \
             margin check runs against the rows it DID understand"
        );
    }
    if status.positions_missing_used_margin > 0 {
        POLLS_MISSING_USED_MARGIN.fetch_add(
            status.positions_missing_used_margin as u64,
            Ordering::Relaxed,
        );
        tracing::warn!(
            target: "neoethos_app::margin_call",
            positions_missing_used_margin = status.positions_missing_used_margin,
            open_positions = status.open_position_count,
            used_margin = status.used_margin,
            "the broker omitted usedMargin on one or more open positions — the \
             computed margin level is OPTIMISTIC by that much"
        );
    }

    if status.is_margin_call() {
        // Both are `Some` whenever `is_margin_call()` holds — that is how
        // `breached_threshold_pct` is derived — but do not `unwrap`: an
        // unwrap on the margin-call path would turn a wire change into a
        // panic in a watchdog thread, which silently stops the watchdog.
        arm(HaltReason::MarginCall {
            margin_level_pct: status.margin_level_pct.unwrap_or(f64::NAN),
            threshold_pct: status.breached_threshold_pct.unwrap_or(f64::NAN),
            equity: status.equity,
            used_margin: status.used_margin,
        });
        return Ok(status);
    }

    match (status.margin_level_pct, status.thresholds.tightest_threshold_pct()) {
        (Some(level), Some(threshold)) => {
            tracing::debug!(
                target: "neoethos_app::margin_call",
                margin_level_pct = level,
                tightest_threshold_pct = threshold,
                equity = status.equity,
                used_margin = status.used_margin,
                open_positions = status.open_position_count,
                environment = status.environment_label,
                "margin level healthy"
            );
        }
        (Some(level), None) => {
            // Positions are open and the broker configured no threshold we can
            // read. Do NOT halt — see the module header. Say it every time, at
            // warn, because it means this watchdog is not actually watching.
            POLLS_NO_THRESHOLD.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                target: "neoethos_app::margin_call",
                margin_level_pct = level,
                open_positions = status.open_position_count,
                "the broker returned NO usable margin-call threshold for this account, \
                 so the margin level cannot be compared against anything — this \
                 watchdog cannot protect the account until it does"
            );
        }
        (None, _) => {
            // No used margin ⇒ no open position ⇒ no margin level and no call.
            tracing::debug!(
                target: "neoethos_app::margin_call",
                open_positions = status.open_position_count,
                "no used margin — nothing to compare"
            );
        }
    }

    Ok(status)
}

/// Spawn the watchdog. Best-effort at startup exactly like the spot streamer:
/// if credentials are missing the first polls fail, are counted, and — after
/// the failure budget — arm the halt, which is the fail-closed answer for "we
/// cannot see the account".
///
/// Call once from the binary's startup path. Runs on its own OS thread because
/// `fetch_margin_status_blocking` is synchronous WSS.
pub fn spawn() {
    if SPAWNED.swap(true, Ordering::SeqCst) {
        tracing::warn!(
            target: "neoethos_app::margin_call",
            "margin-call watchdog spawn requested twice; ignoring the second"
        );
        return;
    }
    start_thread();
}

static SPAWNED: AtomicBool = AtomicBool::new(false);

/// Silently-idempotent start.
///
/// **Why this exists, and why it is a stopgap.** The watchdog belongs in the
/// binary's startup path next to `spread_stats::spawn()` and
/// `rediscovery::spawn()` (`main.rs:415-421`) — that is one line, and it is
/// the right fix. `main.rs` was outside the scope of the change that added
/// this module, so [`prepare_new_order`](crate::app_services::broker_api)
/// calls this before the first order it prepares, which guarantees the
/// watchdog is running in any process that actually opens a position.
///
/// The gap that leaves, stated rather than hidden: a position opened from
/// cTrader's own web/mobile platform on the same account, in a session where
/// this backend never submits an order, is not watched until it does. Moving
/// the call to `main.rs` closes that gap completely.
pub fn ensure_spawned() {
    if SPAWNED.swap(true, Ordering::SeqCst) {
        return;
    }
    start_thread();
}

fn start_thread() {
    let spawned = std::thread::Builder::new()
        .name("margin-call-watchdog".into())
        .spawn(|| {
            tracing::info!(
                target: "neoethos_app::margin_call",
                interval_secs = POLL_INTERVAL.as_secs(),
                failure_budget = MAX_CONSECUTIVE_POLL_FAILURES,
                "margin-call watchdog started"
            );
            loop {
                let _ = poll_once();
                std::thread::sleep(POLL_INTERVAL);
            }
        });
    if let Err(e) = spawned {
        // Fail LOUD, not closed: refusing every order because a thread would
        // not spawn is a worse failure than trading without the watchdog, and
        // the operator can see this line. Stated here so the choice is not a
        // silent one.
        tracing::error!(
            target: "neoethos_app::margin_call",
            error = %e,
            "could not spawn the margin-call watchdog — NOTHING is monitoring this \
             account's margin level. Restart the backend."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halt_describes_a_margin_call_with_both_numbers() {
        let h = Halt {
            reason: HaltReason::MarginCall {
                margin_level_pct: 97.5,
                threshold_pct: 100.0,
                equity: 487.25,
                used_margin: 499.74,
            },
            at_unix_ms: 0,
        };
        let text = h.describe();
        assert!(text.contains("97.5%"), "{text}");
        assert!(text.contains("100.0%"), "{text}");
        assert!(text.to_uppercase().contains("MARGIN CALL"), "{text}");
    }

    #[test]
    fn unreachable_halt_names_the_failure_count_and_the_last_error() {
        let h = Halt {
            reason: HaltReason::BrokerUnreachable {
                consecutive_failures: 3,
                last_error: "socket closed unexpectedly".to_string(),
            },
            at_unix_ms: 0,
        };
        let text = h.describe();
        assert!(text.contains('3'), "{text}");
        assert!(text.contains("socket closed unexpectedly"), "{text}");
    }

    /// The fail-closed direction is the whole point: "we could not read the
    /// answer" must read as a halt, never as an all-clear.
    #[test]
    fn unparseable_response_describes_itself_as_unknown_and_treated_as_a_call() {
        let h = Halt {
            reason: HaltReason::UnparseableResponse {
                detail: "payload shape changed".to_string(),
            },
            at_unix_ms: 0,
        };
        let text = h.describe();
        assert!(text.contains("UNKNOWN"), "{text}");
        assert!(text.contains("treated as if it is"), "{text}");
    }
}
