//! `client_order_id` composition + OAuth token refresh-window constant.
//!
//! Carved out of trading.rs so the atomic counter that guarantees
//! `client_order_id` uniqueness lives next to the constants that control
//! the OAuth refresh window. Both are used by the idempotent retry path
//! and by the new-order builder.
//!
//! PRESERVED FIX (do not change without auditor sign-off):
//! - Batch 10 (docs research on `Ordering` semantics): the atomic
//!   `fetch_add` in `next_client_order_seq` uses `Ordering::Relaxed`.
//!   `Relaxed` is sufficient because atomic ops are linearizable, and
//!   `Ordering` only synchronizes *other* memory accesses around the
//!   atomic — there are none here. The standard library uses `Relaxed`
//!   for similar counters.

use std::time::{SystemTime, UNIX_EPOCH};

/// How close to the bundle expiry we proactively refresh the token (seconds).
pub(super) const CTRADER_TOKEN_REFRESH_WINDOW_SECS: i64 = 300;

pub(super) fn current_unix_seconds() -> anyhow::Result<i64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| anyhow::anyhow!("system clock is before unix epoch"))?
        .as_secs() as i64)
}

/// Process-local monotonic counter for `client_order_id` uniqueness.
/// Two distinct orders within the same wall-clock second would otherwise
/// produce the same `client_order_id` (e.g. a scaling-in strategy firing
/// two market orders 50ms apart) and the broker's clientOrderId-based
/// dedup would collapse them. Pairing the second-resolution timestamp
/// with this monotonic counter guarantees uniqueness for distinct orders
/// AND keeps the same id stable across retries of a single order, which
/// is what the retry-with-backoff path relies on for safe replay.
pub(super) fn next_client_order_seq() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    // Relaxed is sufficient — atomic ops are linearizable; Ordering only synchronizes other memory accesses (std lib uses Relaxed for similar counters).
    COUNTER.fetch_add(1, Ordering::Relaxed)
}
