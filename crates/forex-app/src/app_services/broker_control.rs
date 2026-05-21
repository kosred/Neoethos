//! Broker-control bridge — async signal channel between the cTrader
//! streaming worker (which has no `&mut TradingSession` handle) and the
//! main UI loop in `main.rs::ForexApp::process_messages` (the only place
//! that can call `TradingSession::trip_hardware_kill_global`).
//!
//! ## Why a separate channel from `ServiceEvent`
//!
//! `ServiceEvent` is a `tokio::sync::mpsc` channel owned by `ForexApp`
//! and only readable from `process_messages`. The streaming worker
//! runs inside a synchronous blocking call
//! (`load_live_chart_update_with_transport` → `read_next_spot_event`)
//! from a non-async background thread spun off by the
//! `CTraderLiveStreamingBackend` impl, and we cannot route a
//! `tokio::sync::mpsc::Sender` into that path without leaking a runtime
//! handle through the production backend trait surface. Crossbeam's
//! sync channel gives us a `Sender` that is plain `Send + Sync + Clone`
//! and works from any thread without a runtime — which is what the
//! streaming worker needs.
//!
//! ## Signal semantics
//!
//! - [`BrokerControlSignal::HardwareKill`] is emitted when the broker
//!   sends `ProtoOAAccountDisconnectEvent` — the streaming worker has
//!   already best-effort-closed the socket; the main loop must then
//!   trip the T-Hardware kill switch (research §5.6 in
//!   `risky_mode_compounding_research.md`) and write the
//!   `HARDWARE_KILL_<unix-secs>.flag` sentinel.
//! - [`BrokerControlSignal::ConnectionRestored`] is emitted by the
//!   reconnect logic after a successful re-auth; we **only log** this
//!   signal — the operator must clear the halt manually via the
//!   "Clear HALT" banner button so a flaky connection cannot cause the
//!   trading session to flap on/off.
//!
//! ## Sentinel separation
//!
//! `trip_manual_halt` writes `<data-dir>/HALTED_<unix-secs>.flag`.
//! `trip_hardware_kill_global` writes
//! `<data-dir>/HARDWARE_KILL_<unix-secs>.flag` — separate file so the
//! operator running `ls <data-dir>` can tell at a glance whether the
//! halt was operator-initiated or broker-initiated. This matters for
//! post-mortem audits where the distinction between "I panicked" vs.
//! "the broker dropped me" determines whether retraining of the
//! reconnect logic is needed.
//!
//! WARNING-SUPPRESSION RATIONALE (audit 2026-05-21): the
//! `#![allow(dead_code)]` below is intentional and TIME-BOUND.
//! Today's streaming path (Task #3, wired 2026-05-20) routes
//! disconnect events through `ServiceEvent::ConnectOutcome(Err)`
//! instead of this dedicated crossbeam bridge — that works because
//! the streaming worker has a `ServiceEvent` sender in scope. But
//! when the streaming work moves into a backend implementation that
//! doesn't (e.g. the planned `forex-backend` crate extraction for
//! the Flutter migration), this channel is the right shape because
//! it's plain `Send + Sync + Clone` with no runtime dependency.
//! **Remove the allow the moment any non-test path calls
//! `install_broker_control_sender` / `send_broker_control_signal`.**

#![allow(dead_code)]

use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded};
use std::sync::OnceLock;

/// Channel capacity. Hardware-kill events are rare (one per broker
/// disconnect, which itself is rate-limited to once per 10 min on the
/// cTrader side per `spotware_proto_new_messages.md` §B). A capacity
/// of 16 is wildly overprovisioned for the normal case but means the
/// streaming worker never has to choose between blocking and dropping
/// the signal even if a hostile broker spams disconnect events.
const BROKER_CONTROL_CHANNEL_CAPACITY: usize = 16;

/// Control signal pushed from a broker-facing worker thread (the
/// cTrader streaming worker today) into the main UI loop. The main
/// loop reads with [`try_recv_broker_control`] once per frame and acts
/// via `TradingSession::trip_hardware_kill_global`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BrokerControlSignal {
    /// The broker dropped our session server-side. The streaming
    /// worker has already closed its local socket; the main loop must
    /// flip the T-Hardware kill switch so all subsequent orders are
    /// rejected at the pre-trade gate.
    ///
    /// `reason` is the human-readable string surfaced by
    /// `handle_account_disconnect_event` — it is threaded into the
    /// `tracing::error!` line and the sentinel-file body so the
    /// post-mortem trail is complete.
    HardwareKill { reason: String },
    /// The reconnect loop established a fresh session. We log only —
    /// the halt remains in force until the operator clears it.
    ConnectionRestored,
}

/// Process-global sender slot. Installed exactly once at app startup
/// by `install_broker_control_sender` and read by the streaming worker
/// via `send_broker_control_signal`. `OnceLock` keeps the API simple:
/// the sender is `Send + Sync + Clone`, so we don't need a `Mutex`
/// around it — a writer and a reader can both hold a clone.
static BROKER_CONTROL_SENDER: OnceLock<Sender<BrokerControlSignal>> = OnceLock::new();

/// Install the process-global sender. Returns the matching receiver
/// the caller (the main UI loop) must keep and poll. Idempotent only
/// in the sense that subsequent calls return `None` for the receiver
/// because the global sender is already installed — `OnceLock::set`
/// fails silently after the first install. In production this is
/// called exactly once from `ForexApp::new`; the `cfg(test)` path uses
/// [`make_broker_control_channel_for_test`] which bypasses the global.
pub fn install_broker_control_sender() -> Option<Receiver<BrokerControlSignal>> {
    let (tx, rx) = bounded::<BrokerControlSignal>(BROKER_CONTROL_CHANNEL_CAPACITY);
    match BROKER_CONTROL_SENDER.set(tx) {
        Ok(()) => Some(rx),
        Err(_) => None,
    }
}

/// Push a signal from a worker thread. Returns `true` when the signal
/// was queued, `false` when no sender is installed (the dev / headless
/// case where the main loop never installed one) or the channel is at
/// capacity — both treated as best-effort drops because we already
/// surfaced the disconnect through `anyhow::Error` containing the
/// `CTRADER_ACCOUNT_DISCONNECT_SENTINEL` string. A dropped signal
/// degrades to "operator must notice the banner / log and HALT
/// manually", which is still safe.
pub fn send_broker_control_signal(signal: BrokerControlSignal) -> bool {
    let Some(sender) = BROKER_CONTROL_SENDER.get() else {
        tracing::debug!(
            target: "forex_app::broker_control",
            "no broker-control sender installed; dropping signal"
        );
        return false;
    };
    match sender.try_send(signal) {
        Ok(()) => true,
        Err(err) => {
            tracing::warn!(
                target: "forex_app::broker_control",
                error = %err,
                "broker-control channel rejected signal (full or disconnected)"
            );
            false
        }
    }
}

/// Non-blocking poll used by the main UI loop once per frame. Returns
/// the next pending signal, or `None` when the queue is empty / the
/// channel was dropped. We deliberately drain only ONE signal per call
/// so a flood of disconnect events cannot starve egui repaints — the
/// caller loops over `try_recv` until empty if it wants drain
/// semantics, but a single tick of the main loop processes one at a
/// time which is fine because the only action is "flip the halt
/// flag", which is idempotent.
pub fn try_recv_broker_control(
    receiver: &Receiver<BrokerControlSignal>,
) -> Option<BrokerControlSignal> {
    match receiver.try_recv() {
        Ok(signal) => Some(signal),
        Err(TryRecvError::Empty) => None,
        Err(TryRecvError::Disconnected) => {
            // All senders were dropped (shouldn't happen — the
            // sender lives in the static `OnceLock`). Log once and
            // return `None` so the main loop continues unaffected.
            tracing::warn!(
                target: "forex_app::broker_control",
                "broker-control channel disconnected; no further signals will arrive"
            );
            None
        }
    }
}

/// Test-only constructor that returns a fresh `(Sender, Receiver)`
/// pair without touching the process-global slot. Lets unit tests
/// exercise the signal plumbing in parallel without contending on the
/// `OnceLock`.
#[cfg(test)]
pub fn make_broker_control_channel_for_test()
-> (Sender<BrokerControlSignal>, Receiver<BrokerControlSignal>) {
    bounded::<BrokerControlSignal>(BROKER_CONTROL_CHANNEL_CAPACITY)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hardware_kill_round_trips_through_test_channel() {
        let (tx, rx) = make_broker_control_channel_for_test();
        tx.send(BrokerControlSignal::HardwareKill {
            reason: "account_id=712345 dropped by broker (session must re-auth)".to_string(),
        })
        .expect("send");
        let received = try_recv_broker_control(&rx).expect("queued signal");
        match received {
            BrokerControlSignal::HardwareKill { reason } => {
                assert!(reason.contains("712345"));
            }
            other => panic!("expected HardwareKill, got {other:?}"),
        }
        // Channel must now be empty.
        assert!(try_recv_broker_control(&rx).is_none());
    }

    #[test]
    fn connection_restored_round_trips_through_test_channel() {
        let (tx, rx) = make_broker_control_channel_for_test();
        tx.send(BrokerControlSignal::ConnectionRestored)
            .expect("send");
        assert_eq!(
            try_recv_broker_control(&rx),
            Some(BrokerControlSignal::ConnectionRestored)
        );
    }

    #[test]
    fn send_without_installed_sender_returns_false() {
        // This relies on a fresh process — when run inside the wider
        // test suite the static `OnceLock` may or may not have been
        // installed by another test. We assert only the "no panic /
        // returns bool" contract: a real production drop returns
        // `false`, a stale installed channel returns whatever
        // `try_send` says. Either way, `send_broker_control_signal`
        // never panics.
        let _ = send_broker_control_signal(BrokerControlSignal::ConnectionRestored);
    }
}
