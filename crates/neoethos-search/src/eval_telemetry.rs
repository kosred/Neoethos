//! Which evaluator ran, how often, and for how long.
//!
//! Two hours went into asking why a discovery kept the card idle, and the
//! answer turned out to be that the generation loop never calls the function
//! whose GPU lane was being changed. Nothing in the run said so: the evaluators
//! are silent, so "this one is hot" and "this one is never called" look
//! identical from the outside.
//!
//! Every candidate evaluation entry point reports itself here. The first call
//! to each is logged immediately — that alone answers "is this path even
//! used?" — and the totals are printed at the end of the run, which answers
//! "where did the time go?" without a profiler or another ten-hour rerun.
//!
//! The cost is one atomic increment and one `Instant::now()` per call, against
//! evaluations that take microseconds at minimum. Turning it off is not worth
//! the risk of being blind again.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Default, Clone, Copy)]
struct Tally {
    calls: u64,
    nanos: u128,
    /// Candidates (or genes) seen, so a slow path can be told apart from a
    /// merely popular one.
    items: u64,
}

/// `name @ caller`, leaked once per distinct pair.
///
/// The registry is keyed by `&'static str` and the pairs are bounded by the
/// number of call sites, so leaking them is cheaper and simpler than making the
/// whole map own its keys.
fn interned(name: &'static str, caller: &'static str) -> &'static str {
    static TABLE: OnceLock<Mutex<BTreeMap<(&'static str, &'static str), &'static str>>> =
        OnceLock::new();
    let table = TABLE.get_or_init(|| Mutex::new(BTreeMap::new()));
    let Ok(mut table) = table.lock() else {
        return name;
    };
    table
        .entry((name, caller))
        .or_insert_with(|| Box::leak(format!("{name} @ {caller}").into_boxed_str()))
}

fn registry() -> &'static Mutex<BTreeMap<&'static str, Tally>> {
    static REGISTRY: OnceLock<Mutex<BTreeMap<&'static str, Tally>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Time one evaluation and attribute it to `name`.
///
/// Returns whatever the body returns, so it wraps an existing call without
/// changing its shape.
pub fn measure<T>(name: &'static str, items: usize, body: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let out = body();
    record(name, items, started.elapsed());
    out
}

/// Attribute an already-measured duration to `name`.
pub fn record(name: &'static str, items: usize, elapsed: Duration) {
    let mut first_call = false;
    // Split by caller when one claimed this region, so a shared evaluator does
    // not report five different jobs as one line.
    let caller = current_caller();
    let key: &'static str = if caller.is_empty() {
        name
    } else {
        interned(name, caller)
    };
    if let Ok(mut registry) = registry().lock() {
        let tally = registry.entry(key).or_default();
        first_call = tally.calls == 0;
        tally.calls += 1;
        tally.nanos += elapsed.as_nanos();
        tally.items += items as u64;
    }
    if first_call {
        // The single most useful line: it distinguishes "cold" from "never
        // called", which no amount of staring at the code does reliably.
        tracing::info!(
            target: "neoethos_search::eval_telemetry",
            evaluator = name,
            items,
            elapsed_ms = elapsed.as_millis() as u64,
            "evaluator entered for the first time"
        );
    }
}

/// Log every evaluator's totals, busiest first.
///
/// Call once a work unit finishes. Silent when nothing was recorded, so a run
/// that never evaluated anything does not print an empty table and imply it
/// measured something.
pub fn log_summary(context: &str) {
    let Ok(registry) = registry().lock() else {
        return;
    };
    if registry.is_empty() {
        return;
    }
    let mut rows: Vec<(&'static str, Tally)> =
        registry.iter().map(|(name, tally)| (*name, *tally)).collect();
    rows.sort_by(|a, b| b.1.nanos.cmp(&a.1.nanos));
    let total: u128 = rows.iter().map(|(_, tally)| tally.nanos).sum();
    for (name, tally) in rows {
        let seconds = tally.nanos as f64 / 1e9;
        tracing::info!(
            target: "neoethos_search::eval_telemetry",
            context,
            evaluator = name,
            calls = tally.calls,
            items = tally.items,
            seconds = format!("{seconds:.1}"),
            share_pct = format!("{:.1}", if total > 0 { 100.0 * tally.nanos as f64 / total as f64 } else { 0.0 }),
            "evaluator totals"
        );
    }
}

/// Which caller is on the stack, for the evaluators several code paths share.
///
/// `fast_evaluate_strategy_core` is called from five places and reported 210 500
/// times for 340 seconds — 44 % of a real run — as one number, which says the
/// per-gene screen is expensive and nothing about which screen. Optimising the
/// wrong one is a day spent for nothing, and yesterday that happened three
/// times over.
///
/// The caller sets this around the region it owns; the evaluator appends it to
/// its own name. Thread-local because the screens run under rayon, and a global
/// would attribute one thread's work to another's caller.
pub struct CallerScope;

thread_local! {
    static CALLER: std::cell::RefCell<Option<&'static str>> = const { std::cell::RefCell::new(None) };
}

impl CallerScope {
    /// Attribute everything measured on this thread until the guard drops.
    pub fn enter(name: &'static str) -> Self {
        CALLER.with(|c| *c.borrow_mut() = Some(name));
        Self
    }
}

impl Drop for CallerScope {
    fn drop(&mut self) {
        CALLER.with(|c| *c.borrow_mut() = None);
    }
}

/// The active caller, or `""` when nothing claimed the work.
pub fn current_caller() -> &'static str {
    CALLER.with(|c| c.borrow().unwrap_or(""))
}

/// Forget everything recorded so far, so one work unit's totals cannot be read
/// as another's.
pub fn reset() {
    if let Ok(mut registry) = registry().lock() {
        registry.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn totals_accumulate_per_evaluator_and_reset_clears_them() {
        reset();
        measure("test::alpha", 10, || std::thread::sleep(Duration::from_millis(2)));
        measure("test::alpha", 5, || ());
        measure("test::beta", 1, || ());
        {
            let registry = registry().lock().expect("registry");
            let alpha = registry.get("test::alpha").copied().expect("alpha recorded");
            assert_eq!(alpha.calls, 2);
            assert_eq!(alpha.items, 15);
            assert!(alpha.nanos > 0);
            assert_eq!(registry.get("test::beta").expect("beta recorded").calls, 1);
        }
        reset();
        assert!(registry().lock().expect("registry").is_empty());
    }

    /// A path that is never taken must leave no entry at all — an evaluator
    /// missing from the table is the finding, not an omission.
    #[test]
    fn an_unused_evaluator_leaves_no_trace() {
        reset();
        measure("test::used", 1, || ());
        let registry = registry().lock().expect("registry");
        assert!(registry.contains_key("test::used"));
        assert!(!registry.contains_key("test::never_called"));
    }
}
