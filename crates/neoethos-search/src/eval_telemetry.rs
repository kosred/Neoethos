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
    /// When this evaluator was first entered and last left, as offsets from the
    /// first record of the run.
    ///
    /// `nanos` is summed across threads, so a rayon-parallel evaluator can report
    /// more seconds than the run took. Reading those as wall time is how a stage
    /// that costs seconds gets mistaken for the bottleneck: one measurement had
    /// 413.6 s and 340.0 s inside a 452.4 s run, and only one of them was
    /// sequential. The span says which.
    first_ns: u128,
    last_ns: u128,
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

/// When the run's first measurement happened, so spans share an origin.
fn run_start() -> &'static Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now)
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
        let now = run_start().elapsed().as_nanos();
        if first_call {
            tally.first_ns = now.saturating_sub(elapsed.as_nanos());
        }
        tally.last_ns = now;
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
    let mut rows: Vec<(&'static str, Tally)> = registry
        .iter()
        .map(|(name, tally)| (*name, *tally))
        .collect();
    rows.sort_by(|a, b| b.1.nanos.cmp(&a.1.nanos));
    let total: u128 = rows.iter().map(|(_, tally)| tally.nanos).sum();
    for (name, tally) in rows {
        let seconds = tally.nanos as f64 / 1e9;
        // The span is the wall-clock window this evaluator occupied. Compared
        // against `seconds` it says whether the work was spread over threads: a
        // parallel stage sums to far more than its span, a sequential one to
        // about the same. Deciding what to optimise needs the span; deciding
        // where CPU went needs the sum.
        let span_seconds = tally.last_ns.saturating_sub(tally.first_ns) as f64 / 1e9;
        let parallelism = if span_seconds > 0.001 {
            seconds / span_seconds
        } else {
            1.0
        };
        tracing::info!(
            target: "neoethos_search::eval_telemetry",
            context,
            evaluator = name,
            calls = tally.calls,
            items = tally.items,
            seconds = format!("{seconds:.1}"),
            wall_span_s = format!("{span_seconds:.1}"),
            effective_threads = format!("{parallelism:.1}"),
            share_pct = format!("{:.1}", if total > 0 { 100.0 * tally.nanos as f64 / total as f64 } else { 0.0 }),
            "evaluator totals — share_pct is of SUMMED time; compare seconds against wall_span_s"
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

/// Process-wide, not thread-local.
///
/// Thread-local was the first attempt and measured nothing: these screens run
/// under rayon, the guard is taken on the calling thread, and the workers that
/// do the evaluating never see it. A global works because the phases it labels
/// are sequential — the quality screen finishes before validation starts — so
/// there is no window where two of them are active at once. If that ever stops
/// being true the totals will blur rather than lie, and the fix is to pass the
/// label down explicitly.
static CALLER: std::sync::atomic::AtomicPtr<u8> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

impl CallerScope {
    /// Attribute everything measured until the guard drops.
    pub fn enter(name: &'static str) -> Self {
        CALLER.store(
            name.as_ptr() as *mut u8,
            std::sync::atomic::Ordering::Relaxed,
        );
        LEN.store(name.len(), std::sync::atomic::Ordering::Relaxed);
        Self
    }
}

static LEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl Drop for CallerScope {
    fn drop(&mut self) {
        CALLER.store(std::ptr::null_mut(), std::sync::atomic::Ordering::Relaxed);
        LEN.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

/// The active caller, or `""` when nothing claimed the work.
pub fn current_caller() -> &'static str {
    let ptr = CALLER.load(std::sync::atomic::Ordering::Relaxed);
    if ptr.is_null() {
        return "";
    }
    let len = LEN.load(std::sync::atomic::Ordering::Relaxed);
    // Safe: the pointer only ever comes from a `&'static str` handed to
    // `enter`, and the length is stored with it.
    unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len)) }
}

/// Forget everything recorded so far, so one work unit's totals cannot be read
/// as another's.
pub fn reset() {
    if let Ok(mut registry) = registry().lock() {
        registry.clear();
    }
    if let Ok(mut devices) = device_registry().lock() {
        devices.clear();
    }
}

// ── Device axis (task #35, 2026-08-09) ────────────────────────────────────
//
// The name@caller registry above answers "which evaluator ran, for how long",
// but NOT "did it run on the GPU or silently fall back to the CPU". A run that
// spent 100% on a 3090 and one that silently ran the whole GA on the CPU
// produced byte-identical end-of-run output — the exact indistinguishability
// that let `prop_search_device: cpu` waste rented cards for eight months.
//
// This sibling registry attributes WALL time (never the summed rayon-thread
// nanos the `Tally::nanos` note above warns about — a 128-way CPU eval would
// otherwise swamp the single-stream GPU number and read as ~1% GPU on a genuine
// GPU run) to a device, counts card-present CPU fallbacks, and prints one line
// next to the goal report so the two can never look alike again.

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    #[default]
    Cpu,
    Gpu,
}

#[derive(Default, Clone, Copy)]
struct DeviceMetrics {
    gpu_nanos: u128,
    cpu_nanos: u128,
    /// CPU fallbacks that happened WHILE a usable CUDA card was present — the
    /// bad kind. A card-less host records 0 here by construction.
    cpu_fallback_calls: u64,

    // ── Per-launch detail (2026-08-09) ────────────────────────────────────
    //
    // `gpu_nanos` says the card was the device. It does NOT say how much of
    // that time the card was actually computing. A measured 3090 run reported
    // 452.4 s of "GPU" wall for 0.30 s of device stage timing — 1.3 % — and
    // nothing in the run's own output could have told the two apart. The
    // fields below split one launch into the two halves that decide that
    // question, and count the launches so the split cannot be read as a rate
    // when it is a per-call constant.
    /// Device launches: one per submission that actually reached the engine.
    launches: u64,
    /// Launches that had to tear down the resident session and re-upload the
    /// dataset first. Each one frees and re-allocates the whole workspace, and
    /// every `cudaFree` is an implicit device-wide sync.
    session_rebuilds: u64,
    /// Launches that were leaves of a recursive population split. `calls=7`
    /// at the outer entry point says nothing about these — a single outer call
    /// can fan out into dozens.
    split_leaves: u64,
    /// Host work before the first kernel is submitted: hashing the dataset
    /// key, sizing the session, staging and uploading genes and scenarios.
    host_prep_nanos: u128,
    /// Submit → wait → readback. The only part of a launch the card is in.
    device_nanos: u128,
    /// What the engine itself counted, rather than what the host assumed.
    kernel_submissions: u64,
    synchronization_events: u64,
}

fn device_registry() -> &'static Mutex<BTreeMap<&'static str, DeviceMetrics>> {
    static REGISTRY: OnceLock<Mutex<BTreeMap<&'static str, DeviceMetrics>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Is a usable CUDA card + the native f64 prototype-B lane actually present?
/// The ONLY true hardware check — a mere feature/env flag is not enough. False
/// on every build without prototype B compiled in (Vulkan/ROCm/CPU), so those
/// hosts report "no card" as the normal, expected state.
fn card_present() -> bool {
    #[cfg(feature = "gpu-b-adapter")]
    {
        crate::gpu_native::prototype_b_population_eval::prototype_b_available()
    }
    #[cfg(not(feature = "gpu-b-adapter"))]
    {
        false
    }
}

/// Attribute one population-eval's WALL duration to the device it ran on.
pub fn record_device(lane: &'static str, device: Device, wall: Duration) {
    if let Ok(mut registry) = device_registry().lock() {
        let metrics = registry.entry(lane).or_default();
        match device {
            Device::Gpu => metrics.gpu_nanos += wall.as_nanos(),
            Device::Cpu => metrics.cpu_nanos += wall.as_nanos(),
        }
    }
}

/// One device launch, split into the parts that decide whether the card was
/// working or waiting.
///
/// Filled by the engine adapter and submitted once per launch. Every field is
/// something the launch already knew and threw away: the host has both clocks,
/// and the engine returns `NeoPopulationCounters` from `evaluate` — which the
/// call site discarded into `_counters` one line after receiving it.
#[derive(Default, Clone, Copy)]
pub struct LaunchRecord {
    /// Everything before the first submission: dataset identity, session
    /// sizing, gene and scenario staging and upload.
    pub host_prep: Duration,
    /// Submit, wait, read back.
    pub device: Duration,
    /// From the engine's own counters, not inferred.
    pub kernel_submissions: u64,
    pub synchronization_events: u64,
    /// The resident session was torn down and the dataset re-uploaded.
    pub session_rebuilt: bool,
    /// This launch was a leaf of a recursive population split.
    pub split_leaf: bool,
}

/// Record one device launch against `lane`.
pub fn record_launch(lane: &'static str, record: LaunchRecord) {
    if let Ok(mut registry) = device_registry().lock() {
        let metrics = registry.entry(lane).or_default();
        metrics.launches += 1;
        metrics.host_prep_nanos += record.host_prep.as_nanos();
        metrics.device_nanos += record.device.as_nanos();
        metrics.kernel_submissions += record.kernel_submissions;
        metrics.synchronization_events += record.synchronization_events;
        if record.session_rebuilt {
            metrics.session_rebuilds += 1;
        }
        if record.split_leaf {
            metrics.split_leaves += 1;
        }
    }
}

// ── Which lane a launch belongs to ───────────────────────────────────────────
//
// The engine adapter is shared: the GA's `evaluate_population_core` and the
// validation tail's `validation_backtest_population` call the SAME prototype-B
// entry point with identical argument lists. That is exactly the property that
// let the two be confused for months — "the kernel got 8.7x faster" applied to
// whichever of them the reader had in mind. So the caller names its lane and the
// launch is filed under it, instead of both collapsing into one row.
//
// Thread-local, unlike `CallerScope` above: the lane is entered on the thread
// that performs the launch and the launch happens on that same thread (a direct
// call in the GA arm, inside `catch_unwind` in the validation arm). A global
// would misfile launches whenever two lanes overlap on different threads.
thread_local! {
    static LANE: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(None) };
}

/// Name the lane every launch on this thread belongs to until the guard drops.
pub struct LaneScope(Option<&'static str>);

impl LaneScope {
    pub fn enter(lane: &'static str) -> Self {
        Self(LANE.with(|slot| slot.replace(Some(lane))))
    }
}

impl Drop for LaneScope {
    fn drop(&mut self) {
        LANE.with(|slot| slot.set(self.0));
    }
}

/// The lane launches on this thread belong to.
///
/// The default names the engine rather than a caller, so a launch from a path
/// that forgot to declare its lane is still attributed to something true — and
/// is visibly missing a lane name rather than silently folded into one.
pub fn current_lane() -> &'static str {
    LANE.with(|slot| slot.get())
        .unwrap_or("prototype_b (lane undeclared)")
}

// ── Recursive population splits ──────────────────────────────────────────────
//
// `split_and_evaluate` halves a population and re-enters the evaluator for each
// side, which can recurse again. The outer entry point is what telemetry counted,
// so a `calls=7` line described 7 outer entries covering an unknown number of real
// launches — the repo has twice named a bottleneck from numbers of that shape.
//
// Thread-local because the split recursion never leaves the thread that started
// it, while the quality screen may drive several of them at once under rayon.
thread_local! {
    static SPLIT_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Mark everything launched under this guard as a split leaf.
pub struct SplitScope;

impl SplitScope {
    pub fn enter() -> Self {
        SPLIT_DEPTH.with(|depth| depth.set(depth.get().saturating_add(1)));
        Self
    }
}

impl Drop for SplitScope {
    fn drop(&mut self) {
        SPLIT_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

/// Whether the current launch is inside a recursive split.
pub fn inside_split() -> bool {
    SPLIT_DEPTH.with(|depth| depth.get()) > 0
}

/// Count a CPU fallback that happened while a card was present. No-op on a
/// card-less host, so "0 fallbacks" always means "nothing bad happened".
pub fn note_cpu_fallback(lane: &'static str) {
    if !card_present() {
        return;
    }
    if let Ok(mut registry) = device_registry().lock() {
        registry.entry(lane).or_default().cpu_fallback_calls += 1;
    }
}

/// Print the GPU-vs-CPU split next to the goal report.
///
/// Unlike `log_summary`, this prints EVEN WHEN EMPTY: a run that recorded no
/// device rows still states whether a card was present, so "0% GPU" can always
/// be read as either "no card (normal)" or "card present, everything fell back
/// (bad)" — never left as an inference.
pub fn device_summary() {
    let card = card_present();
    let rows: Vec<(&'static str, DeviceMetrics)> = device_registry()
        .lock()
        .map(|registry| registry.iter().map(|(lane, m)| (*lane, *m)).collect())
        .unwrap_or_default();

    let mut gpu_total: u128 = 0;
    let mut cpu_total: u128 = 0;
    let mut fallbacks: u64 = 0;
    let mut launches_total: u64 = 0;
    let mut host_prep_total: u128 = 0;
    let mut device_total: u128 = 0;
    for (lane, m) in &rows {
        gpu_total += m.gpu_nanos;
        cpu_total += m.cpu_nanos;
        fallbacks += m.cpu_fallback_calls;
        launches_total += m.launches;
        host_prep_total += m.host_prep_nanos;
        device_total += m.device_nanos;
        let denom = (m.gpu_nanos + m.cpu_nanos) as f64;
        let gpu_pct = if denom > 0.0 {
            100.0 * m.gpu_nanos as f64 / denom
        } else {
            0.0
        };
        tracing::info!(
            target: "neoethos_search::eval_telemetry",
            lane,
            gpu_pct = format!("{gpu_pct:.1}"),
            gpu_s = format!("{:.1}", m.gpu_nanos as f64 / 1e9),
            cpu_s = format!("{:.1}", m.cpu_nanos as f64 / 1e9),
            cpu_fallbacks = m.cpu_fallback_calls,
            "population-eval device split (per lane)"
        );
        // Second line rather than more fields on the first: this one answers a
        // different question ("was the card working or waiting?") and is empty
        // for lanes that never launched, which would otherwise pad every row
        // with zeros and hide the row that has the answer.
        if m.launches == 0 {
            continue;
        }
        let host_prep_s = m.host_prep_nanos as f64 / 1e9;
        let device_s = m.device_nanos as f64 / 1e9;
        let accounted = host_prep_s + device_s;
        tracing::info!(
            target: "neoethos_search::eval_telemetry",
            lane,
            launches = m.launches,
            session_rebuilds = m.session_rebuilds,
            split_leaves = m.split_leaves,
            kernel_submissions = m.kernel_submissions,
            sync_events = m.synchronization_events,
            host_prep_ms = format!("{:.1}", host_prep_s * 1e3),
            device_ms = format!("{:.1}", device_s * 1e3),
            device_pct = format!(
                "{:.1}",
                if accounted > 0.0 { 100.0 * device_s / accounted } else { 0.0 }
            ),
            host_prep_ms_per_launch = format!(
                "{:.1}", host_prep_s * 1e3 / m.launches.max(1) as f64
            ),
            device_ms_per_launch = format!(
                "{:.1}", device_s * 1e3 / m.launches.max(1) as f64
            ),
            "launch anatomy — device_pct is of host_prep+device, NOT of wall; a low \
             value means the card waited on the host, not that the work was small"
        );
    }

    let denom = (gpu_total + cpu_total) as f64;
    let gpu_pct = if denom > 0.0 {
        100.0 * gpu_total as f64 / denom
    } else {
        0.0
    };

    // The launch totals ride on BOTH summary branches: a run that stayed 100 %
    // on the card still has to say how much of that was the card computing.
    // "gpu_pct = 100" and "device_pct = 1.3" are both true at once, and only
    // printing the first is how a starved card reads as a healthy one.
    let accounted_total = (host_prep_total + device_total) as f64;
    let device_pct_total = if accounted_total > 0.0 {
        100.0 * device_total as f64 / accounted_total
    } else {
        0.0
    };

    if card && (gpu_pct < 95.0 || fallbacks > 0) {
        // The card was there and the population eval did NOT stay on it. This is
        // the exact failure the operator asked to be impossible to miss: a low
        // average GPU load means the card is starved, and any fallback at all
        // means work silently leaked to the CPU.
        tracing::warn!(
            target: "neoethos_search::eval_telemetry",
            card_present = card,
            gpu_pct = format!("{gpu_pct:.1}"),
            cpu_fallbacks = fallbacks,
            launches = launches_total,
            host_prep_s = format!("{:.1}", host_prep_total as f64 / 1e9),
            device_s = format!("{:.1}", device_total as f64 / 1e9),
            device_pct = format!("{device_pct_total:.1}"),
            "population eval did NOT stay on the GPU — a CUDA card is present but \
             most eval WALL time ran on the CPU. Low average GPU load means the \
             card is starved; investigate before trusting the run's throughput."
        );
    } else {
        tracing::info!(
            target: "neoethos_search::eval_telemetry",
            card_present = card,
            gpu_pct = format!("{gpu_pct:.1}"),
            cpu_fallbacks = fallbacks,
            launches = launches_total,
            host_prep_s = format!("{:.1}", host_prep_total as f64 / 1e9),
            device_s = format!("{:.1}", device_total as f64 / 1e9),
            device_pct = format!("{device_pct_total:.1}"),
            "population-eval device summary (100% GPU on a card is the goal; \
             0% with no card present is normal). device_pct is how much of a \
             launch the card actually computed — gpu_pct=100 with device_pct in \
             the single digits is a starved card, not a healthy run."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both tests drive the SAME process-global registry and `reset()` it, so
    /// running them concurrently races: one test's `reset()` can wipe the other's
    /// just-recorded entry before its assert. Serialize them on a shared lock so
    /// the outcome is deterministic under `cargo test`'s default threading (this
    /// was a latent flake that surfaced when unrelated timing shifted).
    static TELEMETRY_TEST_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn totals_accumulate_per_evaluator_and_reset_clears_them() {
        let _guard = TELEMETRY_TEST_GUARD
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        reset();
        measure("test::alpha", 10, || {
            std::thread::sleep(Duration::from_millis(2))
        });
        measure("test::alpha", 5, || ());
        measure("test::beta", 1, || ());
        {
            let registry = registry().lock().expect("registry");
            let alpha = registry
                .get("test::alpha")
                .copied()
                .expect("alpha recorded");
            assert_eq!(alpha.calls, 2);
            assert_eq!(alpha.items, 15);
            assert!(alpha.nanos > 0);
            assert_eq!(registry.get("test::beta").expect("beta recorded").calls, 1);
        }
        reset();
        assert!(registry().lock().expect("registry").is_empty());
    }

    /// Launches, rebuilds, split leaves and the host/device split accumulate
    /// per lane, and a split leaf is counted as BOTH a launch and a leaf.
    ///
    /// The number this pins is the one the old telemetry could not produce:
    /// `calls` counted outer entries, so seven outer calls that fanned out into
    /// thirty real launches reported seven. A summary that cannot tell those
    /// apart is how the repo named the wrong bottleneck twice.
    #[test]
    fn launches_split_leaves_and_the_host_device_split_accumulate() {
        let _guard = TELEMETRY_TEST_GUARD
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        reset();
        record_launch(
            "test::lane",
            LaunchRecord {
                host_prep: Duration::from_millis(30),
                device: Duration::from_millis(10),
                kernel_submissions: 3,
                synchronization_events: 4,
                session_rebuilt: true,
                split_leaf: false,
            },
        );
        record_launch(
            "test::lane",
            LaunchRecord {
                host_prep: Duration::from_millis(10),
                device: Duration::from_millis(10),
                kernel_submissions: 3,
                synchronization_events: 4,
                session_rebuilt: false,
                split_leaf: true,
            },
        );
        let metrics = device_registry()
            .lock()
            .expect("device registry")
            .get("test::lane")
            .copied()
            .expect("lane recorded");
        assert_eq!(metrics.launches, 2, "both launches counted");
        assert_eq!(metrics.session_rebuilds, 1);
        assert_eq!(metrics.split_leaves, 1, "a split leaf is still a launch");
        assert_eq!(metrics.kernel_submissions, 6);
        assert_eq!(metrics.synchronization_events, 8);
        assert_eq!(
            metrics.host_prep_nanos,
            Duration::from_millis(40).as_nanos()
        );
        assert_eq!(metrics.device_nanos, Duration::from_millis(20).as_nanos());
        reset();
        assert!(
            device_registry()
                .lock()
                .expect("device registry")
                .is_empty()
        );
    }

    /// A launch files itself under the lane its caller declared, and the two
    /// lanes that share the prototype-B adapter must not collapse into one row.
    #[test]
    fn the_lane_scope_separates_the_two_callers_that_share_the_engine() {
        assert_eq!(current_lane(), "prototype_b (lane undeclared)");
        {
            let _outer = LaneScope::enter("population_eval");
            assert_eq!(current_lane(), "population_eval");
            {
                let _inner = LaneScope::enter("validation_eval");
                assert_eq!(current_lane(), "validation_eval");
            }
            // Restores the enclosing lane rather than clearing it — the
            // adapter is re-entrant through `split_and_evaluate`.
            assert_eq!(current_lane(), "population_eval");
        }
        assert_eq!(current_lane(), "prototype_b (lane undeclared)");
    }

    /// Split depth nests, so a leaf several halvings deep is still a leaf.
    #[test]
    fn split_depth_nests_and_unwinds() {
        assert!(!inside_split());
        {
            let _first = SplitScope::enter();
            assert!(inside_split());
            {
                let _second = SplitScope::enter();
                assert!(inside_split());
            }
            assert!(inside_split(), "the outer split is still in progress");
        }
        assert!(!inside_split());
    }

    /// A path that is never taken must leave no entry at all — an evaluator
    /// missing from the table is the finding, not an omission.
    #[test]
    fn an_unused_evaluator_leaves_no_trace() {
        let _guard = TELEMETRY_TEST_GUARD
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        reset();
        measure("test::used", 1, || ());
        let registry = registry().lock().expect("registry");
        assert!(registry.contains_key("test::used"));
        assert!(!registry.contains_key("test::never_called"));
    }
}
