//! Shared exponential-backoff sleep used by every cTrader-facing
//! retry loop (execution, streaming, account session). Previously
//! duplicated byte-for-byte across `ctrader_execution.rs` and
//! `ctrader_streaming.rs`; one file is enough.
//!
//! The contract: `attempt == 0` is a no-op (caller's first try).
//! For `attempt >= 1` the sleep is `base * 2^(attempt-1) + jitter_ms`,
//! clamped to 5,000 ms so a runaway retry loop cannot lock a thread
//! for minutes. The `2^N` shift is bounded at `2^5 = 32` so the
//! intermediate factor cannot overflow even if the operator sets a
//! very high `max_attempts`.

use std::time::Duration;

const MAX_DELAY_MS: u64 = 5_000;
const MAX_FACTOR_SHIFT: u32 = 5;

/// Block the current thread for an exponentially-growing duration
/// keyed off the failed-attempt counter. See module-level docs for
/// the full contract.
pub fn backoff_sleep(attempt: u32, base_ms: u64) {
    if attempt == 0 {
        return;
    }
    let factor = 1u64 << (attempt - 1).min(MAX_FACTOR_SHIFT);
    let jitter = jitter_ms();
    let delay_ms = base_ms
        .saturating_mul(factor)
        .saturating_add(jitter)
        .min(MAX_DELAY_MS);
    std::thread::sleep(Duration::from_millis(delay_ms));
}

fn jitter_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| (d.subsec_nanos() % 100) as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attempt_zero_is_a_no_op_with_no_sleep() {
        let start = std::time::Instant::now();
        backoff_sleep(0, 100);
        assert!(start.elapsed().as_millis() < 5);
    }

    #[test]
    fn factor_shift_is_capped_so_max_delay_is_5_seconds() {
        // attempt 100 would otherwise shift by 99, overflow, and sleep
        // for centuries. The cap at MAX_FACTOR_SHIFT keeps it sane.
        let factor = 1u64 << (100u32 - 1).min(MAX_FACTOR_SHIFT);
        assert_eq!(factor, 32);
        // base 1000 * 32 = 32_000, clamped to 5_000.
        let clamped = (1000u64.saturating_mul(factor)).min(MAX_DELAY_MS);
        assert_eq!(clamped, MAX_DELAY_MS);
    }
}
