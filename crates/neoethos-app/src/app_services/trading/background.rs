//! Shared background-task spawner used by every "fire-and-forget"
//! worker in `TradingSession` (`start_connect`, `start_ctrader_chart_fetch`,
//! `start_ctrader_bootstrap_batch`, future inference producer, …).
//!
//! Two problems this module solves at once:
//!
//! 1. **Panic visibility (Task #2 in the V0.4 audit).** Before this helper
//!    landed, every callsite did `std::thread::spawn(move || work())` and
//!    `reap_finished_background_tasks` joined the handle with
//!    `let _ = handle.join();` — silently swallowing any panic payload. A
//!    panicked worker left the UI's job snapshot stuck at `Running` forever
//!    because no `ServiceEvent::*Failed` was ever emitted. We now wrap the
//!    worker closure in `std::panic::catch_unwind` and, on panic, emit a
//!    [`ServiceEvent::BackgroundTaskPanic`] so the UI can surface a clear
//!    error to the operator instead of the dreaded spinning state.
//!
//! 2. **De-duplicating the spawn boilerplate (Task #11).** The
//!    `std::thread::Builder::new().name(...).spawn(move || { ... }).expect(...)`
//!    incantation was previously copy-pasted across four callsites in
//!    `session.rs`. One helper, one consistent thread-name prefix, one
//!    panic-handling policy.
//!
//! The closure is responsible for sending its own success/failure
//! [`ServiceEvent`]s — this helper only intervenes on panic.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::thread::{Builder, JoinHandle};

use tokio::sync::mpsc::Sender;

use crate::app_services::ServiceEvent;

/// Spawn `work` on a named OS thread. If `work` panics, the panic is
/// captured and translated into a [`ServiceEvent::BackgroundTaskPanic`]
/// on `tx` so the UI thread can show a clear error instead of a
/// permanently-spinning job snapshot.
///
/// `task` is a `&'static str` used both as the OS-level thread name (with
/// a `forex-bg-` prefix so the operator can identify our threads in a
/// debugger or `ps`) and as the diagnostic label in the emitted event.
///
/// # Why panics don't propagate to `handle.join()`
///
/// We catch the panic inside the worker closure so the `JoinHandle`
/// always completes with `Ok(())`. This means existing
/// `reap_finished_background_tasks()` logic (which uses
/// `let _ = handle.join()`) does not need to change — panics are
/// surfaced before the handle is reaped, via `ServiceEvent`.
pub(crate) fn spawn_background_task<F>(
    task: &'static str,
    tx: Sender<ServiceEvent>,
    work: F,
) -> JoinHandle<()>
where
    F: FnOnce() + Send + 'static,
{
    Builder::new()
        .name(format!("forex-bg-{task}"))
        .spawn(move || {
            // `AssertUnwindSafe` is sound here because the closure does
            // not hold any references that would be observable after the
            // panic — each background worker owns its own request/state
            // by move. If a future worker needs to share mutable state
            // across the panic boundary, refactor that state behind a
            // `Mutex<PoisonRecovery>` or similar before adding it here.
            let result = catch_unwind(AssertUnwindSafe(work));
            if let Err(panic_payload) = result {
                let message = panic_message(&panic_payload);
                tracing::error!(
                    target: "neoethos_app::background",
                    task = task,
                    panic = %message,
                    "background task panicked; surfacing to UI via ServiceEvent::BackgroundTaskPanic"
                );
                let _ = tx.blocking_send(ServiceEvent::BackgroundTaskPanic {
                    task: task.to_string(),
                    message,
                });
            }
        })
        .expect("OS refused to spawn background thread (out of file descriptors / process limit?)")
}

/// Extract a human-readable message from a panic payload.
///
/// `std::panic::catch_unwind` returns `Box<dyn Any + Send>` carrying
/// whatever was passed to `panic!`. The two common forms produced by
/// `panic!("...")` and `panic!("{}", x)` are `String` and `&'static str`;
/// anything else (e.g. a user type passed to `panic_any`) falls back to
/// a generic placeholder so the operator at least sees that a panic
/// happened.
fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else {
        "<non-string panic payload>".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::sync::mpsc;

    #[test]
    fn successful_work_produces_no_event() {
        let (tx, mut rx) = mpsc::channel::<ServiceEvent>(4);
        let handle = spawn_background_task("test_ok", tx, || {
            // Intentionally trivial — the contract is "no panic, no event".
        });
        handle.join().expect("join clean thread");
        // The receiver should be empty: closure completed normally and
        // emitted nothing of its own, so no BackgroundTaskPanic was sent.
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            rx.try_recv().is_err(),
            "no event should be emitted on clean completion"
        );
    }

    #[test]
    fn panic_with_string_payload_is_surfaced() {
        let (tx, mut rx) = mpsc::channel::<ServiceEvent>(4);
        let handle = spawn_background_task("test_panic_string", tx, || {
            panic!("synthetic test panic: {}", 42);
        });
        handle.join().expect("join panicked thread (panic was caught)");
        // The wrapper must have emitted a BackgroundTaskPanic with our
        // task name and the formatted panic message.
        std::thread::sleep(Duration::from_millis(50));
        match rx.try_recv() {
            Ok(ServiceEvent::BackgroundTaskPanic { task, message }) => {
                assert_eq!(task, "test_panic_string");
                assert!(
                    message.contains("synthetic test panic"),
                    "panic message should contain the original payload, got: {message}"
                );
            }
            other => panic!("expected BackgroundTaskPanic, got {other:?}"),
        }
    }

    #[test]
    fn panic_with_static_str_payload_is_surfaced() {
        let (tx, mut rx) = mpsc::channel::<ServiceEvent>(4);
        let handle = spawn_background_task("test_panic_str", tx, || {
            panic!("literal string panic");
        });
        handle.join().expect("join panicked thread");
        std::thread::sleep(Duration::from_millis(50));
        match rx.try_recv() {
            Ok(ServiceEvent::BackgroundTaskPanic { task, message }) => {
                assert_eq!(task, "test_panic_str");
                assert_eq!(message, "literal string panic");
            }
            other => panic!("expected BackgroundTaskPanic, got {other:?}"),
        }
    }
}
