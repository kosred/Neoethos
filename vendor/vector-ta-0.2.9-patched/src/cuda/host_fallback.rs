//! The one place in the CUDA lane where "we computed this on the host" is
//! recorded.
//!
//! # Why this exists
//!
//! Eleven CUDA wrappers in this crate used to do the following: call
//! `Module::get_function` on a kernel symbol so that symbol resolution
//! succeeded, THROW THE RESULTING FUNCTION AWAY, compute the whole indicator on
//! the host through `Kernel::ScalarBatch`, and then `DeviceBuffer::from_slice`
//! the host answer so the caller received a device pointer and could not tell
//! the difference. Nine of those wrappers pointed at a one-line empty kernel
//! (`extern "C" __global__ void possible_rsi_batch_f64() {}`); the other two —
//! `pattern_recognition_wrapper.rs` and `rogers_satchell_volatility_wrapper.rs`
//! — had no kernel at all.
//!
//! That is not a fallback. A fallback is visible. That was a disguise.
//!
//! # What is and is not a fallback
//!
//! * **No card present** — computing on the CPU is the CORRECT answer, not a
//!   fallback. It is also not reachable through this module: a `Cuda*` wrapper
//!   only exists once `Context::new` has succeeded on a real device, so every
//!   call site below already has a card. The no-card path lives entirely
//!   outside `src/cuda/` in the crate's ordinary `*_with_kernel` entry points,
//!   and nothing here touches it.
//! * **Card present and a kernel exists** — the kernel MUST run. A failed
//!   launch is an `Err` naming the indicator. It is never quietly recomputed on
//!   the host. Nothing routes through this module.
//! * **Card present and no kernel exists** — the kernel gets WRITTEN. Until it
//!   is, a host computation here is debt, and [`record`] is how that debt stays
//!   countable instead of invisible.
//!
//! # The target value of this counter is zero
//!
//! [`total`] returning 0 after a full run is the goal state, not a nice-to-have.
//! A non-zero reading names indicators that still owe a kernel. It is a
//! TRANSITIONAL instrument: never cite it as a reason not to write a kernel,
//! and never present a counted fallback as "handled".

#![cfg(feature = "cuda")]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

static TOTAL: AtomicU64 = AtomicU64::new(0);

fn table() -> &'static Mutex<BTreeMap<&'static str, u64>> {
    static TABLE: OnceLock<Mutex<BTreeMap<&'static str, u64>>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Record that `indicator` was computed on the HOST inside a CUDA wrapper while
/// a device was present.
///
/// `indicator` is the crate's indicator id (`"pattern_recognition"`), not the
/// wrapper file name and not a kernel symbol, so a reading can be matched
/// against `all_indicators.rs` without a translation step.
///
/// Cheap enough to call unconditionally: one atomic increment plus one
/// uncontended mutex, against an indicator computation measured in
/// milliseconds.
pub fn record(indicator: &'static str) {
    TOTAL.fetch_add(1, Ordering::Relaxed);
    if let Ok(mut map) = table().lock() {
        *map.entry(indicator).or_insert(0) += 1;
    }
}

/// Total host fallbacks recorded since process start (or since [`reset`]).
///
/// Zero is the goal state. Anything else is a list of kernels still owed.
pub fn total() -> u64 {
    TOTAL.load(Ordering::Relaxed)
}

/// Per-indicator counts, ascending by indicator id.
///
/// Returns an empty vector rather than panicking if the table mutex was
/// poisoned — a poisoned counter must not take down a trading process.
pub fn snapshot() -> Vec<(&'static str, u64)> {
    match table().lock() {
        Ok(map) => map.iter().map(|(name, count)| (*name, *count)).collect(),
        Err(_) => Vec::new(),
    }
}

/// One line per indicator that still fell back, or `None` when nothing did.
///
/// Shaped for a log line at the end of a run, which is the only way a counter
/// like this stays honest: a number nobody prints is the same as no number.
pub fn report() -> Option<String> {
    let rows = snapshot();
    if rows.is_empty() {
        return None;
    }
    let mut out = String::from("CUDA host fallbacks (kernels still owed):");
    for (name, count) in rows {
        out.push_str(&format!("\n  {name}: {count}"));
    }
    Some(out)
}

/// Clear the counters. For tests and for a caller that reports per run.
pub fn reset() {
    TOTAL.store(0, Ordering::Relaxed);
    if let Ok(mut map) = table().lock() {
        map.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_and_reports_by_indicator() {
        reset();
        record("pattern_recognition");
        record("pattern_recognition");
        record("possible_rsi");
        assert_eq!(total(), 3);
        assert_eq!(
            snapshot(),
            vec![("pattern_recognition", 2), ("possible_rsi", 1)]
        );
        let text = report().expect("a fallback was recorded");
        assert!(text.contains("pattern_recognition: 2"));
        assert!(text.contains("possible_rsi: 1"));
        reset();
        assert_eq!(total(), 0);
        assert!(report().is_none());
    }
}
