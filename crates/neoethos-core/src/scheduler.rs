//! Admission-control math for the multi-GPU / hybrid scheduler (Stage 0).
//!
//! These are *pure* functions: given the detected [`HardwareProfile`] and a
//! single combo's workload dimensions, they compute
//!   1. how many combos of this footprint may run concurrently without
//!      exceeding host RAM, and
//!   2. whether a single combo's full GA population fits the one GPU device
//!      that the worker process actually uses.
//!
//! The planner reports whether the unchunked workload fits the usable RAM / VRAM
//! budgets. A false `fits_on_gpu` is an explicit requirement for the worker to
//! use its bounded single-device chunking path or fall back to CPU; it is never
//! permission to pretend the work was split across installed cards.
//!
//! Decision rule (the operator's design, locked):
//!   - **Heavy** combo (footprint monopolises RAM) → run alone on one card and
//!     chunk its population inside the worker when the full population does
//!     not fit. CubeCL multi-device clients are not safe in this codebase yet.
//!   - **Light** combo → pack several concurrently across the cards = throughput.

use crate::system::HardwareProfile;

const BYTES_PER_GB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Default monthly-bucket capacity (matches the search backtest kernel default
/// `month_capacity = 240`). Small relative to the per-sample buffers, but
/// counted so the estimate never under-shoots.
pub const DEFAULT_MONTH_CAPACITY: usize = 240;

/// Per-combo workload dimensions that drive the memory estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComboShape {
    /// Number of bars in the timeframe's series — the dominant term in both the
    /// host feature cube and the per-gene GPU backtest buffer.
    pub series_rows: usize,
    /// GA population size (genes evaluated per generation).
    pub population: usize,
    /// Number of feature columns in the expanded (host) feature cube.
    pub feature_count: usize,
    /// Monthly-bucket capacity for the backtest kernel's per-gene monthly
    /// buffers.
    pub month_capacity: usize,
}

impl ComboShape {
    /// Build a shape with the default monthly-bucket capacity.
    pub fn new(series_rows: usize, population: usize, feature_count: usize) -> Self {
        Self {
            series_rows,
            population,
            feature_count,
            month_capacity: DEFAULT_MONTH_CAPACITY,
        }
    }
}

/// Tunable safety margins + byte-cost constants. Defaults are deliberately
/// conservative — we would rather under-pack than OOM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdmissionPolicy {
    /// Fraction of a card's VRAM the planner may fill (headroom for the
    /// wgpu/CUDA context, kernel scratch, and fragmentation).
    pub vram_usable_fraction: f64,
    /// Fraction of available host RAM the planner may fill across **all**
    /// concurrent combos.
    pub ram_usable_fraction: f64,
    /// Bytes per (gene, sample) in the GPU backtest buffer: an `i32` signal plus
    /// an `f32` confidence = 8 B (see `cubecl_eval.rs`).
    pub bytes_per_gene_sample: usize,
    /// Bytes per (row, feature) cell in the host feature cube (`f32`).
    pub bytes_per_feature_cell: usize,
    /// Multiplier on the raw cube estimate for intermediate copies /
    /// fragmentation.
    pub ram_overhead_factor: f64,
    /// A combo is "heavy" when its RAM footprint is `>=` this fraction of usable
    /// RAM: it then monopolises the host RAM budget (concurrency 1).
    pub heavy_ram_fraction: f64,
}

impl Default for AdmissionPolicy {
    fn default() -> Self {
        Self {
            vram_usable_fraction: 0.80,
            ram_usable_fraction: 0.75,
            bytes_per_gene_sample: 8, // i32 signal + f32 confidence
            bytes_per_feature_cell: 4, // f32
            ram_overhead_factor: 2.0, // raw cube + working copies
            heavy_ram_fraction: 0.50,
        }
    }
}

/// How a combo should be scheduled relative to the rest of the box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComboClass {
    /// Footprint monopolises host RAM: run alone on one GPU or the CPU lane.
    Heavy,
    /// Footprint is small: pack several concurrently to fill the box.
    Light,
}

/// Result of admitting a single combo against the current hardware.
#[derive(Debug, Clone, PartialEq)]
pub struct ComboAdmissionPlan {
    pub class: ComboClass,
    /// How many combos of this footprint fit in usable RAM *and* on the cards at
    /// once (always `>= 1`).
    pub max_concurrency: usize,
    /// GPU devices assigned to this worker (`0` => CPU-only / no GPU). This is
    /// currently at most one because a child process pins to one device.
    pub cards_per_combo: usize,
    /// Full population assigned to the selected card (`0` on the CPU lane).
    /// When it exceeds the per-device capacity, the evaluator must chunk it on
    /// that same card; it must never imply unimplemented cross-card sharding.
    pub genes_per_card: usize,
    /// CPU threads to grant each concurrent worker (cores split across the
    /// active workers, never below 1).
    pub cpu_threads_per_worker: usize,
    /// Estimated host RAM for one combo, GB.
    pub est_ram_per_combo_gb: f64,
    /// Estimated VRAM used per card given `genes_per_card`, GB.
    pub est_vram_per_card_gb: f64,
    /// Whether the full population fits on one selected GPU at the row count.
    /// `false` => the worker must chunk on that device or use the CPU lane.
    pub fits_on_gpu: bool,
    /// Fail-loud explanations (warnings / decisions) for logging.
    pub notes: Vec<String>,
}

/// Estimated host RAM for one combo's expanded feature cube, in GB.
pub fn est_ram_per_combo_gb(shape: ComboShape, policy: &AdmissionPolicy) -> f64 {
    let cells = shape.series_rows as f64 * shape.feature_count.max(1) as f64;
    cells * policy.bytes_per_feature_cell as f64 * policy.ram_overhead_factor / BYTES_PER_GB
}

/// Bytes of VRAM a single gene needs on a card for the backtest kernel.
pub fn vram_per_gene_bytes(shape: ComboShape, policy: &AdmissionPolicy) -> usize {
    // Sample buffers (signals + confidence) dominate; monthly buffers are small.
    shape
        .series_rows
        .saturating_mul(policy.bytes_per_gene_sample)
        .saturating_add(shape.month_capacity.saturating_mul(8))
}

/// Max genes that fit on one card with `device_vram_gb` total VRAM.
pub fn genes_per_card(device_vram_gb: f64, shape: ComboShape, policy: &AdmissionPolicy) -> usize {
    let usable = (device_vram_gb * policy.vram_usable_fraction * BYTES_PER_GB).max(0.0);
    let per_gene = vram_per_gene_bytes(shape, policy).max(1) as f64;
    (usable / per_gene).floor().max(0.0) as usize
}

/// Conservative VRAM (GB) assumed for a GPU that exists but doesn't report its
/// memory (common on consumer Vulkan/wgpu adapters, which surface 0). The
/// worker's bounded pool + gene-chunking path can stream oversized populations,
/// so the scheduler still needs the card COUNT to dispatch one combo per card.
/// The assumed value is used only to expose that chunking is required; it must
/// not be treated as measured dedicated VRAM.
const ASSUMED_VRAM_GB: f64 = 8.0;

/// Per-GPU usable VRAM in GB, one entry per detected card.
///
/// Prefer real reported VRAM (NVIDIA via nvidia-smi). When NO card reports its
/// VRAM (consumer wgpu/Vulkan adapters surface 0) but GPUs DO exist, fall back
/// to counting the devices with [`ASSUMED_VRAM_GB`] each — otherwise a consumer
/// multi-GPU box would be seen as "0 cards" and run everything sequentially.
/// The worker-side bounded pool and chunking keep the actual allocation bounded;
/// this fallback is not a claim about dedicated memory capacity.
fn gpu_device_vrams(hw: &HardwareProfile) -> Vec<f64> {
    let from_list: Vec<f64> = hw.gpu_mem_gb.iter().copied().filter(|m| *m > 0.0).collect();
    if !from_list.is_empty() {
        return from_list;
    }
    let from_accel: Vec<f64> = hw
        .accelerator_devices
        .iter()
        .filter(|d| d.memory_gb > 0.0)
        .map(|d| d.memory_gb)
        .collect();
    if !from_accel.is_empty() {
        return from_accel;
    }
    // No card reports VRAM. Count the devices that exist (by accelerator list or
    // num_gpus) and assume a conservative size — never-OOM handles the fitting.
    let device_count = hw.accelerator_devices.len().max(hw.num_gpus);
    vec![ASSUMED_VRAM_GB; device_count]
}

fn cpu_threads_per_worker(cores: usize, active_workers: usize) -> usize {
    (cores / active_workers.max(1)).max(1)
}

/// Plan how a single combo should be admitted against `hw`, guaranteeing the
/// returned single-device workload description exposes when chunking is needed
/// to stay inside usable RAM / VRAM budgets.
pub fn plan_combo(
    shape: ComboShape,
    hw: &HardwareProfile,
    policy: &AdmissionPolicy,
) -> ComboAdmissionPlan {
    let mut notes = Vec::new();

    // --- RAM side: footprint, classification, RAM-bound concurrency ---------
    let ram_per_combo = est_ram_per_combo_gb(shape, policy);
    let usable_ram = (hw.available_ram_gb * policy.ram_usable_fraction).max(0.0);
    let max_concurrency_ram = if ram_per_combo <= f64::EPSILON {
        notes.push("degenerate RAM estimate (0); defaulting concurrency to 1".to_string());
        1
    } else {
        ((usable_ram / ram_per_combo).floor() as usize).max(1)
    };
    let heavy = ram_per_combo >= usable_ram * policy.heavy_ram_fraction;
    let class = if heavy { ComboClass::Heavy } else { ComboClass::Light };
    if ram_per_combo > usable_ram {
        notes.push(format!(
            "combo RAM est {:.1}GB exceeds usable RAM {:.1}GB — runs alone and may page; consider chunking rows",
            ram_per_combo, usable_ram
        ));
    }

    // --- GPU side: count cards, size one-device work -------------------------
    let gpu_vrams = gpu_device_vrams(hw);
    let num_gpus = gpu_vrams.len();
    let per_gene_gb = vram_per_gene_bytes(shape, policy) as f64 / BYTES_PER_GB;

    if num_gpus == 0 {
        let active = if heavy { 1 } else { max_concurrency_ram };
        notes.push("no usable GPU detected — combo runs on the CPU lane".to_string());
        return ComboAdmissionPlan {
            class,
            max_concurrency: active,
            cards_per_combo: 0,
            genes_per_card: 0,
            cpu_threads_per_worker: cpu_threads_per_worker(hw.cpu_cores, active),
            est_ram_per_combo_gb: ram_per_combo,
            est_vram_per_card_gb: 0.0,
            fits_on_gpu: false,
            notes,
        };
    }

    // A worker is pinned to one card. Size the full assigned population against
    // the smallest selectable card so every logical slot has a safe contract.
    let min_vram = gpu_vrams.iter().copied().fold(f64::INFINITY, f64::min);
    let per_card_cap = genes_per_card(min_vram, shape, policy);

    if per_card_cap == 0 {
        notes.push(format!(
            "even one gene needs {:.1}GB VRAM ({} rows) — exceeds usable VRAM on a {:.0}GB card; chunk rows or use CPU",
            per_gene_gb, shape.series_rows, min_vram
        ));
        let active = if heavy { 1 } else { max_concurrency_ram };
        return ComboAdmissionPlan {
            class,
            max_concurrency: active,
            cards_per_combo: 0,
            genes_per_card: 0,
            cpu_threads_per_worker: cpu_threads_per_worker(hw.cpu_cores, active),
            est_ram_per_combo_gb: ram_per_combo,
            est_vram_per_card_gb: 0.0,
            fits_on_gpu: false,
            notes,
        };
    }

    let cards_per_combo = 1;
    let max_concurrency = if heavy {
        1
    } else {
        max_concurrency_ram.min(num_gpus).max(1)
    };
    let genes = shape.population;
    let fits_on_gpu = genes <= per_card_cap;
    if !fits_on_gpu {
        notes.push(format!(
            "population {} must run on one card but exceeds its cap of {} genes — chunk on one card or use CPU",
            shape.population, per_card_cap
        ));
    }
    let est_vram_per_card_gb = genes as f64 * per_gene_gb;

    let cpu_threads_per_worker = cpu_threads_per_worker(hw.cpu_cores, max_concurrency);

    ComboAdmissionPlan {
        class,
        max_concurrency,
        cards_per_combo,
        genes_per_card: genes,
        cpu_threads_per_worker,
        est_ram_per_combo_gb: ram_per_combo,
        est_vram_per_card_gb,
        fits_on_gpu,
        notes,
    }
}

// ============================================================================
// Stage 1: the work scheduler state machine.
//
// A pure, hardware-free state machine that the CLI `schedule` command drives.
// It owns the queue + a logical device pool + RAM accounting + the heavy-vs-
// light dispatch policy; the CLI maps logical card slots to real device ids
// (NEOETHOS_BOT_SEARCH_EVAL_{WGPU,CUDA}_DEVICE) and does the actual
// `Command::spawn`. Keeping the policy here makes it unit-testable with no GPU.
//
// Logical card slots are `0..usable_card_count`. The CLI builds the list of
// usable device ids (excluding 0-VRAM adapters) once and indexes into it, so a
// slot here maps to a concrete device id there.
// ============================================================================

use std::collections::VecDeque;

/// One combo to schedule, with its precomputed admission plan.
#[derive(Debug, Clone, PartialEq)]
pub struct ComboItem {
    /// Stable identifier, e.g. `"EURUSD/M1"`.
    pub id: String,
    pub shape: ComboShape,
    pub plan: ComboAdmissionPlan,
}

impl ComboItem {
    pub fn new(id: impl Into<String>, shape: ComboShape, plan: ComboAdmissionPlan) -> Self {
        Self { id: id.into(), shape, plan }
    }
}

/// A dispatch decision handed to a worker.
#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub id: String,
    /// Logical card slots assigned (empty => CPU lane).
    pub card_ids: Vec<usize>,
    pub genes_per_card: usize,
    pub cpu_threads: usize,
    pub class: ComboClass,
}

#[derive(Debug, Clone)]
struct RunningItem {
    item: ComboItem,
    card_ids: Vec<usize>,
    ram_gb: f64,
}

/// Greedy, FIFO, hardware-aware scheduler.
///
/// Multi-card scaling is **combo-level**: each combo runs on ONE card and many
/// combos run concurrently, one per card, bounded by free cards + usable RAM.
/// Intra-combo GPU sharding is disabled because CubeCL WGPU cannot safely drive
/// the project's multiple device contexts concurrently in one process. The
/// admission plan and dispatch contract therefore both assign one card.
/// On a many-card, big-RAM box this fills every card with a different combo.
///
/// Invariants:
///   - in-flight committed RAM never exceeds usable RAM (except a single combo
///     that alone needs more than the whole budget, which runs by itself);
///   - logical card slots are never double-assigned.
#[derive(Debug)]
pub struct WorkScheduler {
    pending: VecDeque<ComboItem>,
    running: Vec<RunningItem>,
    free_cards: Vec<usize>,
    total_cards: usize,
    usable_ram_gb: f64,
    committed_ram_gb: f64,
    cpu_cores: usize,
}

impl WorkScheduler {
    /// Build a scheduler. Combos are ordered **heavy-first** (biggest RAM
    /// footprint first) so the deepest timeframes start earliest.
    pub fn new(mut combos: Vec<ComboItem>, hw: &HardwareProfile, policy: &AdmissionPolicy) -> Self {
        combos.sort_by_key(|c| match c.plan.class {
            ComboClass::Heavy => 0u8,
            ComboClass::Light => 1u8,
        });
        let total_cards = gpu_device_vrams(hw).len();
        Self {
            pending: combos.into(),
            running: Vec::new(),
            free_cards: (0..total_cards).collect(),
            total_cards,
            usable_ram_gb: (hw.available_ram_gb * policy.ram_usable_fraction).max(0.0),
            committed_ram_gb: 0.0,
            cpu_cores: hw.cpu_cores.max(1),
        }
    }

    pub fn total_cards(&self) -> usize {
        self.total_cards
    }
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }
    pub fn running_len(&self) -> usize {
        self.running.len()
    }
    pub fn free_cards(&self) -> usize {
        self.free_cards.len()
    }
    pub fn is_done(&self) -> bool {
        self.pending.is_empty() && self.running.is_empty()
    }

    /// Dispatch as many queued combos as currently fit. Returns the newly
    /// started assignments (empty when nothing can start yet — wait for a
    /// [`Self::complete`] call).
    pub fn poll(&mut self) -> Vec<Assignment> {
        let mut out = Vec::new();
        while let Some(front) = self.pending.front() {
            let ram = front.plan.est_ram_per_combo_gb;
            // One card per GPU combo (sharding disabled). A combo whose plan
            // requests 0 cards — no GPU present, or requeued to CPU after a
            // failure — runs on the CPU lane. So GPU concurrency is bounded by
            // free cards, CPU-lane concurrency purely by RAM.
            let need_cards = if self.total_cards > 0 && front.plan.cards_per_combo > 0 {
                1
            } else {
                0
            };
            // Card budget: stop once every card is busy.
            if need_cards > self.free_cards.len() {
                break;
            }
            // RAM budget: the first in-flight combo may exceed the whole budget
            // (it runs by itself); otherwise the in-flight sum must stay under
            // usable RAM. FIFO-strict: stop at the first combo that doesn't fit.
            let ram_ok =
                self.running.is_empty() || self.committed_ram_gb + ram <= self.usable_ram_gb + 1e-9;
            if !ram_ok {
                break;
            }

            // Commit it.
            let item = self.pending.pop_front().expect("front exists");
            let card_ids: Vec<usize> = if need_cards > 0 {
                self.free_cards.drain(0..need_cards).collect()
            } else {
                Vec::new()
            };
            self.committed_ram_gb += ram;
            let class = item.plan.class;
            let genes = item.plan.genes_per_card;
            out.push(Assignment {
                id: item.id.clone(),
                card_ids: card_ids.clone(),
                genes_per_card: genes,
                cpu_threads: 0, // set fairly below
                class,
            });
            self.running.push(RunningItem { item, card_ids, ram_gb: ram });
        }
        // Fairly split CPU cores across ALL in-flight workers (advisory budget;
        // avoids over-subscribing the cores when several combos run concurrently).
        if !out.is_empty() {
            let active = self.running.len().max(1);
            let cpu = (self.cpu_cores / active).max(1);
            for a in out.iter_mut() {
                a.cpu_threads = cpu;
            }
        }
        out
    }

    /// Mark a dispatched combo finished; returns its cards + RAM to the pool.
    pub fn complete(&mut self, id: &str) {
        if let Some(pos) = self.running.iter().position(|r| r.item.id == id) {
            let done = self.running.remove(pos);
            self.free_cards.extend(done.card_ids);
            self.free_cards.sort_unstable();
            self.committed_ram_gb = (self.committed_ram_gb - done.ram_gb).max(0.0);
        }
    }

    /// A dispatched combo failed (crash / GPU OOM). Free its resources and
    /// requeue it for the **CPU lane** (no cards) at the back of the queue —
    /// the belt-and-suspenders fallback behind the never-OOM math.
    pub fn fail_and_requeue_cpu(&mut self, id: &str) {
        if let Some(pos) = self.running.iter().position(|r| r.item.id == id) {
            let done = self.running.remove(pos);
            self.free_cards.extend(done.card_ids);
            self.free_cards.sort_unstable();
            self.committed_ram_gb = (self.committed_ram_gb - done.ram_gb).max(0.0);
            let mut item = done.item;
            item.plan.cards_per_combo = 0;
            item.plan.genes_per_card = 0;
            item.plan.fits_on_gpu = false;
            item.plan
                .notes
                .push("requeued to CPU lane after GPU failure/OOM".to_string());
            self.pending.push_back(item);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::HardwareProfile;

    fn hw(cores: usize, ram_gb: f64, cards: &[f64]) -> HardwareProfile {
        HardwareProfile {
            schema_version: crate::schema_version::SchemaVersion::new(1),
            cpu_cores: cores,
            total_ram_gb: ram_gb,
            available_ram_gb: ram_gb,
            gpu_names: cards.iter().map(|_| "test-gpu".to_string()).collect(),
            num_gpus: cards.len(),
            gpu_mem_gb: cards.to_vec(),
            accelerator_devices: Vec::new(),
            timestamp: "test".to_string(),
            platform_label: "test".to_string(),
        }
    }

    #[test]
    fn cpu_only_box_runs_on_cpu_lane() {
        let p = AdmissionPolicy::default();
        let shape = ComboShape::new(5_000_000, 4000, 2000);
        let plan = plan_combo(shape, &hw(64, 128.0, &[]), &p);
        assert_eq!(plan.cards_per_combo, 0);
        assert_eq!(plan.genes_per_card, 0);
        assert!(!plan.fits_on_gpu);
        assert!(plan.cpu_threads_per_worker >= 1);
    }

    #[test]
    fn heavy_m1_uses_one_card_and_requires_chunking() {
        let p = AdmissionPolicy::default();
        // ~74GB cube => heavy on a 116GB box. The installed card count must not
        // make the single-device worker appear to fit without chunking.
        let shape = ComboShape::new(5_000_000, 4000, 2000);
        let plan = plan_combo(shape, &hw(60, 116.0, &[48.0; 8]), &p);
        assert_eq!(plan.class, ComboClass::Heavy);
        assert_eq!(plan.max_concurrency, 1, "heavy monopolises the box");
        assert_eq!(plan.cards_per_combo, 1);
        assert_eq!(plan.genes_per_card, 4000);
        assert!(!plan.fits_on_gpu);
        assert!(plan.est_vram_per_card_gb > 48.0 * p.vram_usable_fraction);
        // Heavy => the single worker gets all cores.
        assert_eq!(plan.cpu_threads_per_worker, 60);
    }

    #[test]
    fn single_device_worker_plan_does_not_divide_population_across_installed_cards() {
        let p = AdmissionPolicy::default();
        // The runtime pins each combo to exactly one card. A population that
        // would fit only after division by eight must therefore be rejected,
        // not advertised as an eight-way shard that the child never executes.
        let shape = ComboShape::new(5_000_000, 4000, 2000);
        let plan = plan_combo(shape, &hw(60, 116.0, &[48.0; 8]), &p);

        assert_eq!(plan.cards_per_combo, 1);
        assert_eq!(plan.genes_per_card, shape.population);
        assert!(!plan.fits_on_gpu);
        assert!(plan.notes.iter().any(|note| note.contains("one card")));
    }

    #[test]
    fn light_h4_packs_concurrently_one_card_each() {
        let p = AdmissionPolicy::default();
        // Tiny series => light; should pack one combo per card.
        let shape = ComboShape::new(30_000, 4000, 2000);
        let plan = plan_combo(shape, &hw(64, 116.0, &[48.0; 8]), &p);
        assert_eq!(plan.class, ComboClass::Light);
        assert_eq!(plan.cards_per_combo, 1);
        assert_eq!(plan.max_concurrency, 8, "8 cards => 8 light combos at once");
        assert!(plan.fits_on_gpu);
        // Cores split across the 8 active workers.
        assert_eq!(plan.cpu_threads_per_worker, 8);
    }

    #[test]
    fn single_card_heavy_uses_that_one_card() {
        let p = AdmissionPolicy::default();
        let shape = ComboShape::new(800_000, 4000, 2000);
        let plan = plan_combo(shape, &hw(60, 116.0, &[48.0]), &p);
        assert_eq!(plan.cards_per_combo, 1);
        assert!(plan.fits_on_gpu);
        assert!(plan.est_vram_per_card_gb <= 48.0 * p.vram_usable_fraction);
    }

    #[test]
    fn never_exceeds_vram_budget_across_shapes() {
        let p = AdmissionPolicy::default();
        let cards = [48.0; 4];
        for &rows in &[10_000usize, 250_000, 1_000_000, 3_000_000] {
            for &pop in &[256usize, 2000, 8000] {
                let shape = ComboShape::new(rows, pop, 1500);
                let plan = plan_combo(shape, &hw(48, 256.0, &cards), &p);
                if plan.fits_on_gpu {
                    assert!(
                        plan.est_vram_per_card_gb <= 48.0 * p.vram_usable_fraction + 1e-6,
                        "rows={rows} pop={pop} vram/card={} exceeds budget",
                        plan.est_vram_per_card_gb
                    );
                }
            }
        }
    }

    #[test]
    fn genes_per_card_respects_cap() {
        let p = AdmissionPolicy::default();
        let shape = ComboShape::new(1_000_000, 4000, 1000);
        // 48GB * 0.8 = 38.4GB usable; per gene = 1e6*8 = 8MB => ~4800 genes/card.
        let cap = genes_per_card(48.0, shape, &p);
        assert!(cap > 0);
        let used = cap as f64 * vram_per_gene_bytes(shape, &p) as f64 / BYTES_PER_GB;
        assert!(used <= 48.0 * p.vram_usable_fraction + 1e-9);
    }

    #[test]
    fn population_too_big_to_fit_is_flagged_not_silently_truncated() {
        let p = AdmissionPolicy::default();
        // Huge rows so each card holds very few genes; small card count.
        let shape = ComboShape::new(2_000_000_000, 100_000, 100);
        let plan = plan_combo(shape, &hw(32, 256.0, &[48.0, 48.0]), &p);
        // Either a single gene doesn't fit (cards=0) or the split doesn't fit;
        // in both cases fits_on_gpu must be false and a note must explain why.
        assert!(!plan.fits_on_gpu);
        assert!(!plan.notes.is_empty());
    }

    // --- WorkScheduler (Stage 1) ------------------------------------------

    fn mk_item(id: &str, class: ComboClass, cards: usize, ram: f64) -> ComboItem {
        ComboItem {
            id: id.to_string(),
            shape: ComboShape::new(1000, 100, 10),
            plan: ComboAdmissionPlan {
                class,
                max_concurrency: 1,
                cards_per_combo: cards,
                genes_per_card: if cards > 0 { 50 } else { 0 },
                cpu_threads_per_worker: 4,
                est_ram_per_combo_gb: ram,
                est_vram_per_card_gb: 1.0,
                fits_on_gpu: cards > 0,
                notes: vec![],
            },
        }
    }

    #[test]
    fn all_light_combos_fill_every_card_at_once() {
        let combos: Vec<_> = (0..8)
            .map(|i| mk_item(&format!("L{i}"), ComboClass::Light, 1, 1.0))
            .collect();
        let mut sched = WorkScheduler::new(combos, &hw(64, 256.0, &[48.0; 8]), &AdmissionPolicy::default());
        let started = sched.poll();
        assert_eq!(started.len(), 8, "8 cards => 8 light combos concurrently");
        // Distinct, non-overlapping card slots covering 0..8.
        let mut cards: Vec<usize> = started.iter().flat_map(|a| a.card_ids.clone()).collect();
        cards.sort_unstable();
        assert_eq!(cards, (0..8).collect::<Vec<_>>());
        assert_eq!(sched.free_cards(), 0);
    }

    #[test]
    fn consumer_wgpu_zero_vram_box_counts_cards_not_serialize() {
        // Consumer Vulkan/wgpu adapters report 0 VRAM but the GPUs exist
        // (num_gpus). never-OOM makes every combo fit any card, so the scheduler
        // must COUNT them — otherwise a 2-GPU consumer box would run everything
        // on "0 cards" sequentially (the bug this fixes).
        let combos = vec![
            mk_item("A", ComboClass::Light, 1, 1.0),
            mk_item("B", ComboClass::Light, 1, 1.0),
        ];
        let mut sched = WorkScheduler::new(
            combos,
            &hw(8, 32.0, &[0.0, 0.0]),
            &AdmissionPolicy::default(),
        );
        assert_eq!(sched.total_cards(), 2, "two 0-VRAM GPUs must count as 2 cards");
        let started = sched.poll();
        assert_eq!(started.len(), 2, "both combos dispatched across the 2 cards");
        assert_eq!(sched.free_cards(), 0);
    }

    #[test]
    fn heavy_does_not_block_lights_combo_level_concurrency() {
        let combos = vec![
            mk_item("Llate", ComboClass::Light, 1, 1.0),
            mk_item("H", ComboClass::Heavy, 8, 70.0),
            mk_item("Lother", ComboClass::Light, 1, 1.0),
        ];
        let mut sched = WorkScheduler::new(combos, &hw(60, 116.0, &[48.0; 8]), &AdmissionPolicy::default());
        // No heavy-exclusivity: the heavy (sorted first) runs on ONE card while
        // the two lights run concurrently on other cards (70+1+1 = 72 <= 87 usable).
        let started = sched.poll();
        assert_eq!(started.len(), 3, "heavy + 2 lights run concurrently");
        assert_eq!(started[0].id, "H", "heavy-first ordering");
        assert!(started.iter().all(|a| a.card_ids.len() == 1), "one card per combo");
        assert_eq!(started[0].cpu_threads, 20, "60 cores / 3 in-flight workers");
        assert_eq!(sched.free_cards(), 5);
        assert_eq!(sched.running_len(), 3);
    }

    #[test]
    fn ram_budget_caps_concurrency_even_with_free_cards() {
        // usable RAM = 134 * 0.75 = 100.5; each combo wants 40 => only 2 fit.
        let combos: Vec<_> = (0..8)
            .map(|i| mk_item(&format!("L{i}"), ComboClass::Light, 1, 40.0))
            .collect();
        let mut sched = WorkScheduler::new(combos, &hw(64, 134.0, &[48.0; 8]), &AdmissionPolicy::default());
        let started = sched.poll();
        assert_eq!(started.len(), 2, "RAM-bound to 2 despite 8 free cards");
        assert_eq!(sched.free_cards(), 6);
    }

    #[test]
    fn cards_and_ram_are_returned_on_completion() {
        let combos = vec![
            mk_item("A", ComboClass::Light, 2, 10.0),
            mk_item("B", ComboClass::Light, 2, 10.0),
        ];
        let mut sched = WorkScheduler::new(combos, &hw(32, 128.0, &[48.0; 4]), &AdmissionPolicy::default());
        let started = sched.poll();
        assert_eq!(started.len(), 2);
        // One card per combo now (sharding disabled): 2 of 4 cards used.
        assert_eq!(sched.free_cards(), 2);
        sched.complete("A");
        assert_eq!(sched.free_cards(), 3);
        sched.complete("B");
        assert_eq!(sched.free_cards(), 4);
        assert!(sched.is_done());
    }

    #[test]
    fn cpu_only_box_runs_combo_on_cpu_lane() {
        let combos = vec![mk_item("H", ComboClass::Heavy, 0, 200.0)];
        let mut sched = WorkScheduler::new(combos, &hw(96, 256.0, &[]), &AdmissionPolicy::default());
        assert_eq!(sched.total_cards(), 0);
        let started = sched.poll();
        assert_eq!(started.len(), 1);
        assert!(started[0].card_ids.is_empty(), "CPU lane => no cards");
        assert_eq!(started[0].cpu_threads, 96);
    }

    #[test]
    fn oom_requeue_moves_combo_to_cpu_lane() {
        let combos = vec![mk_item("G", ComboClass::Light, 2, 10.0)];
        let mut sched = WorkScheduler::new(combos, &hw(32, 128.0, &[48.0; 4]), &AdmissionPolicy::default());
        let started = sched.poll();
        assert_eq!(started.len(), 1);
        assert_eq!(started[0].card_ids.len(), 1, "one card per combo");
        // Simulate a GPU failure: requeue to the CPU lane (plan cards -> 0).
        sched.fail_and_requeue_cpu("G");
        assert_eq!(sched.free_cards(), 4, "card freed after failure");
        let retry = sched.poll();
        assert_eq!(retry.len(), 1);
        assert!(retry[0].card_ids.is_empty(), "retried on CPU lane, no cards");
        assert_eq!(sched.pending_len(), 0);
    }
}
