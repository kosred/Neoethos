//! Wall-clock helpers shared across the workspace (#152).
//!
//! Every site that needed "current Unix ms" rolled its own
//! `SystemTime::now().duration_since(UNIX_EPOCH)` chain with
//! slightly different fallback handling (some panic, some
//! `.unwrap_or(0)`, some `.unwrap_or_else(|_| Duration::from_secs(0))`).
//! That made it hard to do two important things later:
//!
//! 1. **Freeze the clock in tests** — you can't mock 5
//!    independently-written `SystemTime::now()` call sites.
//! 2. **Audit time sources** for the timezone-mixing bugs the
//!    deep audit flagged (#164).
//!
//! So this module owns the canonical "now in Unix ms (UTC)"
//! helper. Future iterations can swap the body for a
//! mockable `Clock` trait without re-touching every call site.

use std::time::{SystemTime, UNIX_EPOCH};

/// Returns the number of milliseconds since the Unix epoch (UTC).
///
/// Returns `0` if the system clock is set before 1970, which is
/// the same fallback the previous five hand-rolled implementations
/// used. A system clock that wrong would break a lot more than a
/// timestamp field — the conservative fallback keeps every callsite
/// observable in logs rather than panicking and losing the
/// surrounding context.
pub fn now_unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_unix_ms_is_reasonable() {
        let ms = now_unix_ms();
        // Reasonable bounds: after 2020-01-01 (1577836800000)
        // and before 2099-01-01 (4070908800000). Wider than the
        // test will live; if this assertion fails the system
        // clock is genuinely off, not the function.
        assert!(ms > 1_577_836_800_000, "got {ms}");
        assert!(ms < 4_070_908_800_000, "got {ms}");
    }

    #[test]
    fn two_calls_return_monotonic_or_equal_values() {
        let a = now_unix_ms();
        let b = now_unix_ms();
        assert!(b >= a, "now_unix_ms went backwards: {a} → {b}");
    }
}
