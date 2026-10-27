use super::smc_indicators::{
    SmcSearchConfig, enforce_min_structural_smc_flags, randomize_smc_flags,
};
use super::strategy_gene::Gene;
use neoethos_core::utils::fnv1a64_update;
use rand::Rng;
use rand::seq::index::sample;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{HashSet, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;

// Phase 63 lifted the FNV-1a constants into `neoethos_core::utils::hashing`
// so the seen-signature ledger and the contract-policy hashes produce
// byte-for-byte identical output. `FNV_OFFSET_BASIS` is kept here only
// because the gene-signature hash chains it explicitly through
// `fnv1a_update` calls that pre-date the extraction.
const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParentSelectionPolicy {
    Uniform,
    RankWeighted,
    Softmax,
    Tournament,
}

impl ParentSelectionPolicy {
    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "uniform" => Self::Uniform,
            "rank" | "rank_weighted" | "rank-weighted" => Self::RankWeighted,
            "softmax" | "boltzmann" => Self::Softmax,
            "tournament" => Self::Tournament,
            _ => Self::RankWeighted,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurvivorSelectionPolicy {
    Elitist,
    RankWeighted,
    Tournament,
    Generational,
}

impl SurvivorSelectionPolicy {
    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "elitist" | "elite" => Self::Elitist,
            "rank" | "rank_weighted" | "rank-weighted" => Self::RankWeighted,
            "tournament" => Self::Tournament,
            "generational" | "none" | "non_elitist" | "non-elitist" => Self::Generational,
            _ => Self::RankWeighted,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EvolutionSearchPolicy {
    pub survivor_fraction: f64,
    pub immigrant_fraction: f64,
    pub parent_selection: ParentSelectionPolicy,
    pub survivor_selection: SurvivorSelectionPolicy,
    pub selection_temperature: f64,
    pub tournament_size: usize,
}

impl EvolutionSearchPolicy {
    pub fn new(
        survivor_fraction: f64,
        immigrant_fraction: f64,
        parent_selection: ParentSelectionPolicy,
        survivor_selection: SurvivorSelectionPolicy,
        selection_temperature: f64,
        tournament_size: usize,
    ) -> Self {
        Self {
            survivor_fraction: survivor_fraction.clamp(0.0, 0.95),
            immigrant_fraction: immigrant_fraction.clamp(0.0, 0.95),
            parent_selection,
            survivor_selection,
            selection_temperature: selection_temperature.max(1e-3),
            tournament_size: tournament_size.max(2),
        }
    }
}

impl Default for EvolutionSearchPolicy {
    fn default() -> Self {
        Self::new(
            0.10,
            0.25,
            ParentSelectionPolicy::RankWeighted,
            SurvivorSelectionPolicy::RankWeighted,
            0.75,
            4,
        )
    }
}

fn fnv1a_update(hash: u64, bytes: &[u8]) -> u64 {
    fnv1a64_update(hash, bytes)
}

fn quantize_f64(value: f64, scale: f64) -> i64 {
    (value * scale).round() as i64
}

fn rank_weights(candidate_indices: &[usize]) -> Vec<f64> {
    let total = candidate_indices.len().max(1) as f64;
    candidate_indices
        .iter()
        .enumerate()
        .map(|(rank, _)| (total - rank as f64).max(1.0))
        .collect()
}

fn softmax_weights(
    scores: &[f64],
    candidate_indices: &[usize],
    selection_temperature: f64,
) -> Vec<f64> {
    let temperature = selection_temperature.max(1e-6);
    let eligible: Vec<f64> = candidate_indices
        .iter()
        .filter_map(|idx| scores.get(*idx))
        .copied()
        .filter(|score| score.is_finite())
        .collect();
    if eligible.is_empty() {
        // Explicit all-invalid fallback: no finite eligible genome exists.
        // This is degenerate uniform fallback, never healthy selection and
        // never promotion of an invalid genome above a finite one.
        return vec![1.0; candidate_indices.len()];
    }
    let max_score = eligible.into_iter().fold(f64::NEG_INFINITY, f64::max);

    candidate_indices
        .iter()
        .map(|idx| {
            let score = scores[*idx];
            if !score.is_finite() {
                return 0.0;
            }
            let centered = (score - max_score) / temperature;
            if centered.is_finite() {
                centered.exp().max(1e-12)
            } else {
                0.0
            }
        })
        .collect()
}

fn draw_weighted_offset(weights: &[f64], rng: &mut impl Rng) -> usize {
    let total = weights
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .sum::<f64>();
    if total <= 0.0 {
        return 0;
    }

    let mut target = rng.random_range(0.0..total);
    for (idx, weight) in weights.iter().enumerate() {
        let normalized = if weight.is_finite() && *weight > 0.0 {
            *weight
        } else {
            0.0
        };
        if target <= normalized {
            return idx;
        }
        target -= normalized;
    }
    weights.len().saturating_sub(1)
}

// For n <= 2^26 every integer in the triangular sum is exactly representable
// in f64. Binary-searching that sum therefore preserves the old ordered
// subtraction's inclusive boundary, total, and single RNG draw. Above this
// bound retain the general floating-point reduction rather than approximate it.
fn rank_offset_for_target(count: usize, target: f64) -> usize {
    debug_assert!(count > 0 && count <= (1 << 26));
    let n = count as u64;
    let (mut low, mut high) = (0, count - 1);
    while low < high {
        let middle = low + (high - low) / 2;
        let terms = middle as u64 + 1;
        let cumulative = (terms * (2 * n - terms + 1) / 2) as f64;
        if target <= cumulative {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    low
}

pub fn select_parent_index(
    scores: &[f64],
    candidate_indices: &[usize],
    selection_policy: ParentSelectionPolicy,
    tournament_size: usize,
    selection_temperature: f64,
    rng: &mut impl Rng,
) -> usize {
    if candidate_indices.is_empty() {
        return 0;
    }

    match selection_policy {
        ParentSelectionPolicy::Uniform => {
            candidate_indices[rng.random_range(0..candidate_indices.len())]
        }
        ParentSelectionPolicy::Tournament => {
            let rounds = tournament_size.max(2).min(candidate_indices.len());
            let mut winner = candidate_indices[rng.random_range(0..candidate_indices.len())];
            let mut winner_score = scores.get(winner).copied().unwrap_or(f64::NEG_INFINITY);
            for _ in 1..rounds {
                let candidate = candidate_indices[rng.random_range(0..candidate_indices.len())];
                let candidate_score = scores.get(candidate).copied().unwrap_or(f64::NEG_INFINITY);
                if candidate_score > winner_score {
                    winner = candidate;
                    winner_score = candidate_score;
                }
            }
            winner
        }
        ParentSelectionPolicy::RankWeighted => {
            let count = candidate_indices.len();
            if count <= (1 << 26) {
                let n = count as u64;
                let total = (n * (n + 1) / 2) as f64;
                let target = rng.random_range(0.0..total);
                return candidate_indices[rank_offset_for_target(count, target)];
            }
            let weights = rank_weights(candidate_indices);
            candidate_indices[draw_weighted_offset(&weights, rng)]
        }
        ParentSelectionPolicy::Softmax => {
            let weights = softmax_weights(scores, candidate_indices, selection_temperature);
            candidate_indices[draw_weighted_offset(&weights, rng)]
        }
    }
}

pub fn select_survivor_indices(
    scores: &[f64],
    survivor_count: usize,
    survivor_policy: SurvivorSelectionPolicy,
    selection_temperature: f64,
    tournament_size: usize,
    rng: &mut impl Rng,
) -> Vec<usize> {
    let requested = survivor_count.min(scores.len());
    if requested == 0 {
        return Vec::new();
    }

    match survivor_policy {
        SurvivorSelectionPolicy::Elitist => (0..requested).collect(),
        SurvivorSelectionPolicy::Generational => Vec::new(),
        SurvivorSelectionPolicy::RankWeighted | SurvivorSelectionPolicy::Tournament => {
            let parent_policy = match survivor_policy {
                SurvivorSelectionPolicy::RankWeighted => ParentSelectionPolicy::RankWeighted,
                SurvivorSelectionPolicy::Tournament => ParentSelectionPolicy::Tournament,
                SurvivorSelectionPolicy::Elitist | SurvivorSelectionPolicy::Generational => {
                    ParentSelectionPolicy::Uniform
                }
            };

            let mut available: Vec<usize> = (0..scores.len()).collect();
            let mut selected = Vec::with_capacity(requested);
            while !available.is_empty() && selected.len() < requested {
                let idx = select_parent_index(
                    scores,
                    &available,
                    parent_policy,
                    tournament_size,
                    selection_temperature,
                    rng,
                );
                selected.push(idx);
                // IDs start unique/sorted and removal preserves that order.
                // Do not re-filter the entire population for each survivor.
                let offset = available
                    .binary_search(&idx)
                    .expect("selected available parent");
                available.remove(offset);
            }
            selected.sort_unstable();
            selected
        }
    }
}

pub fn gene_signature_hash(gene: &Gene) -> u64 {
    let mut h = FNV_OFFSET_BASIS;
    h = fnv1a_update(h, &(gene.indices.len() as u64).to_le_bytes());
    for idx in &gene.indices {
        h = fnv1a_update(h, &(*idx as u64).to_le_bytes());
    }
    h = fnv1a_update(h, &(gene.weights.len() as u64).to_le_bytes());
    for w in &gene.weights {
        h = fnv1a_update(h, &quantize_f64(*w, 10_000.0).to_le_bytes());
    }
    h = fnv1a_update(
        h,
        &quantize_f64(gene.long_threshold, 1_000_000.0).to_le_bytes(),
    );
    h = fnv1a_update(
        h,
        &quantize_f64(gene.short_threshold, 1_000_000.0).to_le_bytes(),
    );
    h = fnv1a_update(h, &[gene.use_ob as u8]);
    h = fnv1a_update(h, &[gene.use_fvg as u8]);
    h = fnv1a_update(h, &[gene.use_liq_sweep as u8]);
    h = fnv1a_update(h, &[gene.mtf_confirmation as u8]);
    h = fnv1a_update(h, &[gene.use_premium_discount as u8]);
    h = fnv1a_update(h, &[gene.use_inducement as u8]);
    h = fnv1a_update(h, &[gene.use_bos as u8]);
    h = fnv1a_update(h, &[gene.use_choch as u8]);
    h = fnv1a_update(h, &[gene.use_eqh as u8]);
    h = fnv1a_update(h, &[gene.use_eql as u8]);
    h = fnv1a_update(h, &[gene.use_displacement as u8]);
    h = fnv1a_update(h, &quantize_f64(gene.tp_pips, 100.0).to_le_bytes());
    h = fnv1a_update(h, &quantize_f64(gene.sl_pips, 100.0).to_le_bytes());
    h = fnv1a_update(h, &quantize_f64(gene.stop_vol_mult, 100.0).to_le_bytes());
    h
}

#[derive(Debug, Default)]
pub struct SeenSignatureMemory {
    pub all: HashSet<u64>,
    pub order: VecDeque<u64>,
    pub pending: Vec<u64>,
    pub file_path: Option<PathBuf>,
    pub flush_every: usize,
    pub max_entries: usize,
}

/// Typed runtime knobs that previously lived only in
/// `NEOETHOS_BOT_PROP_SEEN_*` env vars. The seen-signature memory consults
/// the cached overrides each time it is constructed, but the env vars
/// themselves are read at most once per process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SeenSignatureMemoryRuntimeOverrides {
    pub flush_every: usize,
    pub load_max: usize,
    pub max_entries: usize,
    pub file_path: Option<PathBuf>,
}

impl Default for SeenSignatureMemoryRuntimeOverrides {
    fn default() -> Self {
        Self {
            flush_every: 4096,
            load_max: 3_000_000,
            max_entries: 3_000_000,
            file_path: None,
        }
    }
}

/// The cap `max_entries: 0` derives to, from the machine rather than from a
/// number nobody could pick.
///
/// WHY THIS REPLACED `usize::MAX` (2026-08-10). `0` used to mean UNBOUNDED: a
/// `HashSet<u64>` plus a `VecDeque<u64>` growing for the whole run — and `0` is
/// exactly what someone types for "no limit", so the sentinel meant what it
/// looked like right up until the box swapped. Peak memory is a function of the
/// hardware, never of a user parameter.
///
/// Sizing: one signature costs 8 B in the deque plus a hashbrown slot (8 B key
/// + 1 B control, ~1.14× for the 87.5 % load factor) ≈ 27 B; we budget 32 B.
/// The dedup memory gets 2 % of available RAM, clamped to [1 M, 64 M] entries —
/// at 32 B that is 32 MB to 2 GB. The old flat default was 3 M entries (~96 MB),
/// which lands inside the band on any box with ≥ 5 GB free, so this is not a
/// quiet behaviour change on ordinary hardware.
fn derived_seen_max_entries() -> usize {
    const BYTES_PER_ENTRY: u64 = 32;
    const FLOOR: usize = 1_000_000;
    const CEILING: usize = 64_000_000;
    let available = neoethos_core::system::available_memory_bytes();
    if available == 0 {
        // The probe failed. Say so and take the floor — a failed probe must not
        // look like a generous answer.
        tracing::warn!(
            target: "neoethos_search::seen_memory",
            entries = FLOOR,
            "available-memory probe returned 0; seen-signature cap falls back to the floor"
        );
        return FLOOR;
    }
    let budget = (available / 50).max(1); // 2 %
    ((budget / BYTES_PER_ENTRY) as usize).clamp(FLOOR, CEILING)
}

impl SeenSignatureMemoryRuntimeOverrides {
    // `from_env()` DELETED 2026-08-10 with `NEOETHOS_BOT_PROP_SEEN_FLUSH_EVERY`,
    // `_LOAD_MAX`, `_MAX_ENTRIES` and `_FILE`. The last of those seeded the
    // dedup memory from a file ACROSS RUNS, so it changed which candidates a
    // run could generate — cross-run state, set by export, recorded nowhere.
    // All four are typed on `models.seen_signature_runtime`.

    /// Config-driven constructor. `max_entries == 0` now means **DERIVE from
    /// available RAM** (see `derived_seen_max_entries`) rather than
    /// UNBOUNDED.
    ///
    /// ⚠ WHAT THIS CHANGES. Eviction is FIFO, so a cap that BITES re-admits
    /// previously-seen genes and changes what the run explores — it is not a
    /// pure memory knob. The effective cap is therefore logged whenever it is
    /// derived, and the first eviction is logged too (`insert_in_memory`), so a
    /// run that started recycling signatures says so instead of quietly
    /// exploring differently.
    pub fn from_settings(s: &neoethos_core::Settings) -> Self {
        let c = &s.models.seen_signature_runtime;
        let max_entries = if c.max_entries == 0 {
            let derived = derived_seen_max_entries();
            tracing::info!(
                target: "neoethos_search::seen_memory",
                configured = 0,
                effective = derived,
                "models.seen_signature_runtime.max_entries is 0 = DERIVE; the seen-signature \
                 cap is sized from available RAM. Eviction is FIFO, so if this cap binds the \
                 run will re-admit genes it has already tried."
            );
            derived
        } else {
            c.max_entries.max(1)
        };
        Self {
            flush_every: c.flush_every.max(1),
            load_max: c.load_max,
            max_entries,
            file_path: c
                .file_path
                .as_ref()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .map(PathBuf::from),
        }
    }
}

static SEEN_SIGNATURE_MEMORY_RUNTIME_OVERRIDES: OnceLock<SeenSignatureMemoryRuntimeOverrides> =
    OnceLock::new();

// One exact discovery-ledger hand-off to the next GA constructed on this
// execution thread. The default configuration deliberately has no global seen
// file, so the caller's validated receipt/config-bound hashes need an in-memory
// carrier. Thread-local + one-shot semantics prevent hashes from one concurrent
// discovery or an aborted earlier run from leaking into another search.
thread_local! {
    static STAGED_SEEN_SIGNATURE_HASHES: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn clear_staged_seen_signature_hashes_on_this_thread() {
    STAGED_SEEN_SIGNATURE_HASHES.with(|staged| staged.borrow_mut().clear());
}

pub(crate) fn stage_seen_signature_hashes_for_next_current_on_this_thread(hashes: &[u64]) {
    STAGED_SEEN_SIGNATURE_HASHES.with(|staged| {
        let mut staged = staged.borrow_mut();
        staged.clear();
        staged.extend_from_slice(hashes);
    });
}

fn take_staged_seen_signature_hashes_on_this_thread() -> Vec<u64> {
    STAGED_SEEN_SIGNATURE_HASHES.with(|staged| std::mem::take(&mut *staged.borrow_mut()))
}

/// Install process-wide seen-signature-memory overrides. Returns
/// `Err(existing)` when overrides were already installed earlier (the
/// first install wins).
pub fn install_seen_signature_memory_runtime_overrides(
    overrides: SeenSignatureMemoryRuntimeOverrides,
) -> Result<(), SeenSignatureMemoryRuntimeOverrides> {
    SEEN_SIGNATURE_MEMORY_RUNTIME_OVERRIDES.set(overrides)
}

/// Config-driven install — reads the seen-signature knobs from the single
/// `Settings` instead of the environment. Idempotent.
pub fn install_seen_signature_memory_runtime_overrides_from_settings(s: &neoethos_core::Settings) {
    let _ = SEEN_SIGNATURE_MEMORY_RUNTIME_OVERRIDES
        .set(SeenSignatureMemoryRuntimeOverrides::from_settings(s));
}

/// Returns the currently installed seen-signature-memory overrides, or
/// the deterministic defaults when no install has happened.
pub fn current_seen_signature_memory_runtime_overrides() -> SeenSignatureMemoryRuntimeOverrides {
    SEEN_SIGNATURE_MEMORY_RUNTIME_OVERRIDES
        .get()
        .cloned()
        .unwrap_or_default()
}

impl SeenSignatureMemory {
    /// Build the dedup memory from the installed runtime overrides, seeding it
    /// from the persisted signature file when one is configured.
    pub fn current() -> Self {
        let overrides = current_seen_signature_memory_runtime_overrides();
        let flush_every = overrides.flush_every.max(1);
        let load_max = overrides.load_max;
        let max_entries = overrides.max_entries;
        let file_path = overrides.file_path;

        let mut memory = Self {
            all: HashSet::new(),
            order: VecDeque::new(),
            pending: Vec::new(),
            file_path,
            flush_every,
            max_entries,
        };
        if let Some(path) = memory.file_path.clone() {
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            if let Ok(buf) = fs::read(&path) {
                if buf.len() >= 8 && buf.len() % 8 == 0 {
                    for chunk in buf.chunks_exact(8) {
                        if load_max > 0 && memory.all.len() >= load_max {
                            break;
                        }
                        let mut arr = [0_u8; 8];
                        arr.copy_from_slice(chunk);
                        memory.insert_in_memory(u64::from_le_bytes(arr));
                    }
                } else if let Ok(text) = String::from_utf8(buf) {
                    for line in text.lines() {
                        if load_max > 0 && memory.all.len() >= load_max {
                            break;
                        }
                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }
                        if let Ok(v) = u64::from_str_radix(line.trim_start_matches("0x"), 16) {
                            memory.insert_in_memory(v);
                        } else if let Ok(v) = line.parse::<u64>() {
                            memory.insert_in_memory(v);
                        }
                    }
                }
            }
        }
        for signature in take_staged_seen_signature_hashes_on_this_thread() {
            memory.insert_in_memory(signature);
        }
        memory
    }

    pub fn insert_in_memory(&mut self, sig: u64) -> bool {
        if !self.all.insert(sig) {
            return false;
        }
        self.order.push_back(sig);
        if self.max_entries != usize::MAX {
            while self.all.len() > self.max_entries {
                if let Some(old) = self.order.pop_front() {
                    self.all.remove(&old);
                    // FIRST EVICTION (2026-08-10). Past this point the memory
                    // is no longer a dedup guarantee: eviction is FIFO, so the
                    // oldest signatures become admissible again and the GA can
                    // re-generate genes it already evaluated. That changes what
                    // the run EXPLORES, not just what it stores, so it is said
                    // out loud exactly once instead of being a silent property
                    // of a memory number.
                    static EVICTED: std::sync::Once = std::sync::Once::new();
                    let cap = self.max_entries;
                    EVICTED.call_once(|| {
                        tracing::warn!(
                            target: "neoethos_search::seen_memory",
                            max_entries = cap,
                            "seen-signature cap reached — evicting oldest signatures FIFO. \
                             Previously-tried genes are now re-admissible, so this run will \
                             explore differently from one with a larger cap."
                        );
                    });
                } else {
                    break;
                }
            }
        }
        true
    }

    pub fn insert_hash(&mut self, sig: u64) -> bool {
        if !self.insert_in_memory(sig) {
            return false;
        }
        if self.file_path.is_some() {
            self.pending.push(sig);
            if self.pending.len() >= self.flush_every {
                self.flush();
            }
        }
        true
    }

    pub fn insert_gene(&mut self, gene: &Gene) -> bool {
        self.insert_hash(gene_signature_hash(gene))
    }

    pub fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let path = match &self.file_path {
            Some(p) => p.clone(),
            None => {
                self.pending.clear();
                return;
            }
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) {
            let mut bytes = Vec::with_capacity(self.pending.len() * 8);
            for v in &self.pending {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            // Only clear `pending` if both the write AND the flush
            // succeeded. Otherwise the data is still buffered and will
            // be retried on the next call. Without the flush check,
            // an OS-level buffer-write failure (disk full, broken
            // file handle) would silently drop data because we'd
            // clear pending even though nothing reached disk.
            if file.write_all(&bytes).is_ok() && file.flush().is_ok() {
                self.pending.clear();
            }
        }
    }
}

fn random_coarse_weight(rng: &mut impl Rng) -> f64 {
    // 2026-07-02 (search-space expansion, SCORING_VERSION 5 changelog): weights
    // may now be NEGATIVE — "exit/veto when this indicator disagrees" is a real
    // strategy pattern the GA previously could not express (the seed templates
    // carry -0.20/-0.25 contrarian terms, but every mutation used to replace
    // them with a positive level, so inherited negatives only ever decayed).
    // Both signal paths (CPU `combined += w*v`, GPU kernel `weight * indicator`)
    // and `Gene::normalize` (retains by |w|) are sign-agnostic — verified before
    // this change. Negative draws are the 1/3 minority: the feature library is
    // direction-aligned by design, so contrarian terms are the exception the
    // search may reach, not the default it must fight through.
    let levels = [0.2, 0.4, 0.6, 0.8, 1.0];
    let w = levels[rng.random_range(0..levels.len())];
    if rng.random_bool(1.0 / 3.0) { -w } else { w }
}

/// Static fallback threshold ladder. Calibrated for z-score-normalised
/// features per the 2026-05-26 narrow-ladder fix (F-273). Used when
/// the adaptive ladder hasn't been installed for this process.
const STATIC_THRESHOLD_LADDER: [f64; 6] = [0.10, 0.20, 0.35, 0.50, 0.70, 0.90];

/// Relative tolerance for treating adjacent adaptive-ladder slots as the same
/// randomized choice. Duplicate/near-duplicate slots silently overweight one
/// threshold; such a ladder must use the static fallback instead.
const ADAPTIVE_THRESHOLD_LADDER_DISTINCT_REL_TOLERANCE: f64 = 1e-9;

/// F-277 (2026-05-28): process-wide adaptive threshold ladder derived
/// from the actual feature-magnitude profile of the current discovery
/// dataset. Installed once at the start of `run_discovery_cycle_with_progress`
/// via `install_adaptive_threshold_ladder` when the operator opts in
/// (`NEOETHOS_BOT_PROP_ADAPTIVE_THRESHOLDS=1`). When None, gene init falls
/// back to the static z-score ladder above.
///
/// Why: the static ladder is calibrated for the assumption that
/// normalised features have unit-ish variance, but the cTrader vector_ta
/// pipeline's normalisation step is per-indicator (not per-symbol-per-TF),
/// so a metal-pair feature cube can have very different magnitudes
/// from a JPY-cross cube. The static ladder works for the majority of
/// FX majors but mis-calibrates extreme-volatility cases (XAGUSD on M1,
/// BTCUSD on H1). The adaptive ladder reads the actual cube and picks
/// thresholds at the dataset's own percentile points.
/// Audit D06 (2026-07-13): this was a `OnceLock` (first-write-wins-forever).
/// In a multi-symbol batch sweep — the orchestrator runs many (symbol, TF)
/// combos in ONE process, sequentially — only the FIRST symbol's ladder was
/// installed and every later symbol silently inherited it (an XAGUSD ladder
/// applied to EURUSD, etc.). It is now a replaceable cell: each discovery run
/// installs its OWN dataset's ladder (or clears back to the static one when
/// adaptive thresholds are off), so no ladder leaks across symbols. Discovery
/// is single-instance and runs sequentially, so a per-run replace is correct;
/// the `RwLock` still allows the GA's many reader threads to read concurrently
/// within a run.
static ADAPTIVE_THRESHOLD_LADDER: std::sync::RwLock<Option<[f64; 6]>> =
    std::sync::RwLock::new(None);

/// Install (REPLACE) the adaptive threshold ladder derived from the current
/// run's feature cube. Unlike the old first-write-wins semantics, each call
/// overwrites, so a later symbol's run gets its own ladder. Values should be
/// sorted ascending and finite + non-negative (the derivation clamps/sorts).
pub fn install_adaptive_threshold_ladder(ladder: [f64; 6]) {
    let mut guard = ADAPTIVE_THRESHOLD_LADDER
        .write()
        .unwrap_or_else(|e| e.into_inner());
    *guard = Some(ladder);
}

/// Clear the adaptive ladder back to the static fallback. Called at the start
/// of a run that does NOT use adaptive thresholds (or whose derivation was
/// degenerate) so the PREVIOUS symbol's ladder cannot leak into it.
pub fn clear_adaptive_threshold_ladder() {
    let mut guard = ADAPTIVE_THRESHOLD_LADDER
        .write()
        .unwrap_or_else(|e| e.into_inner());
    *guard = None;
}

/// Read the currently-installed threshold ladder. Returns the adaptive ladder
/// when installed, else the static fallback.
pub fn current_threshold_ladder() -> [f64; 6] {
    ADAPTIVE_THRESHOLD_LADDER
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .unwrap_or(STATIC_THRESHOLD_LADDER)
}

/// Per-RUN dataset scale for the gene stop/target band: the median ATR of the
/// bars being searched, in pips. `None` = no scale installed, use the absolute
/// pip band.
///
/// A replaceable cell rather than a `OnceLock` for exactly the reason audit D06
/// gave for the threshold ladder above: the batch orchestrator runs many
/// (symbol, timeframe) combos in ONE process, sequentially, and a first-write-
/// wins cell would apply the first combo's M5 scale to every later H1 and H4 run
/// — the very cross-timeframe confusion this whole change exists to remove.
/// Discovery is single-instance and its combos are sequential, so a per-run
/// replace is correct; the `RwLock` still lets the GA's reader threads read
/// concurrently within a run.
static GENE_STOP_ATR_PIPS: std::sync::RwLock<Option<f64>> = std::sync::RwLock::new(None);

/// Install (REPLACE) this run's ATR scale, in pips. Non-finite or non-positive
/// values clear the scale instead of installing a band that would collapse to a
/// point — a zero ATR would otherwise produce `sl ∈ [0, 0]`.
pub fn install_gene_stop_atr_scale(atr_pips: f64) {
    let mut guard = GENE_STOP_ATR_PIPS
        .write()
        .unwrap_or_else(|e| e.into_inner());
    *guard = if atr_pips.is_finite() && atr_pips > 0.0 {
        Some(atr_pips)
    } else {
        None
    };
}

/// Clear back to the absolute pip band. Called at the start of a run that does
/// NOT scale by ATR, so the PREVIOUS combo's scale cannot leak into it.
pub fn clear_gene_stop_atr_scale() {
    let mut guard = GENE_STOP_ATR_PIPS
        .write()
        .unwrap_or_else(|e| e.into_inner());
    *guard = None;
}

/// The stop/target band, resolved to pips for the dataset currently being
/// searched.
///
/// WHAT CHANGED AND WHAT IT PERMITS (2026-08-09). The band used to be four
/// literals: `sl.clamp(6, 20)`, `tp.clamp(12, 45)`, reward:risk sampled in
/// `[1.5, 2.5]`. Those are M5 pip counts. On H1 (ATR ≈ 12 pips) the whole stop
/// band sits inside one bar's range; on H4 (ATR ≈ 30 pips) it sits below it. So
/// every "just search a higher timeframe" suggestion was void — the higher
/// timeframes could not be expressed by any gene the GA was able to write.
///
/// The band also excluded the reward:risk its own payoff floor required. With
/// barriers only, payoff is `(tp - c)/(sl + c)`, so a 2.0 floor needs
/// `tp >= 2·sl + 3·c`; at the charged c = 2.89 pips a 20-pip stop needs
/// `tp >= 48.67`, outside the 45-pip ceiling, and no draw sampled above 2.5 R.
///
/// WHAT IT DOES NOT BUY. Nothing, in expectation. Measured across the exit-
/// geometry sweep, expectancy stayed at -4.15 pips per trade while payoff moved
/// 0.91 → 2.53. Widening the band changes which shapes are REACHABLE. Whether
/// any of them pays is decided by the expectancy gate, not here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedGeneStopBounds {
    pub sl_min_pips: f64,
    pub sl_max_pips: f64,
    pub tp_min_pips: f64,
    pub tp_max_pips: f64,
    pub rr_min: f64,
    pub rr_max: f64,
    /// The ATR (pips) the band was derived from, or `None` when the absolute
    /// fallback band is in force. Reported so a run's log can say which.
    pub atr_pips: Option<f64>,
}

impl ResolvedGeneStopBounds {
    /// A stop/target pair in the middle of the band — what the old
    /// `(15.0, 30.0)` literal expressed on M5, stated as a position in the band
    /// instead of as two pip counts that only meant something on one timeframe.
    pub fn mid_pair(&self) -> (f64, f64) {
        let sl = 0.5 * (self.sl_min_pips + self.sl_max_pips);
        let rr = 0.5 * (self.rr_min + self.rr_max);
        (sl, (sl * rr).clamp(self.tp_min_pips, self.tp_max_pips))
    }
}

/// Resolve the configured multiples against this run's installed ATR scale.
pub fn current_gene_stop_bounds() -> ResolvedGeneStopBounds {
    let cfg = super::runtime_overrides::current_gene_stop_bounds_overrides();
    let atr = *GENE_STOP_ATR_PIPS.read().unwrap_or_else(|e| e.into_inner());
    let atr = if cfg.atr_scaled { atr } else { None };
    match atr {
        Some(atr_pips) => {
            let sl_min = (cfg.sl_min_atr * atr_pips).max(1e-6);
            let sl_max = (cfg.sl_max_atr * atr_pips).max(sl_min);
            ResolvedGeneStopBounds {
                sl_min_pips: sl_min,
                sl_max_pips: sl_max,
                // The target band is the stop band times the reward:risk band, so
                // every (sl, rr) the initialiser can draw lands INSIDE the clamp.
                // The old ceiling was independent of the stop band, which is how
                // a legal draw (sl 20 × rr 2.5 = 50) got silently clipped to 45
                // and became a 2.25 R gene the search never asked for.
                tp_min_pips: sl_min * cfg.rr_min,
                tp_max_pips: sl_max * cfg.rr_max,
                rr_min: cfg.rr_min,
                rr_max: cfg.rr_max,
                atr_pips: Some(atr_pips),
            }
        }
        None => ResolvedGeneStopBounds {
            sl_min_pips: cfg.sl_min_pips,
            sl_max_pips: cfg.sl_max_pips,
            tp_min_pips: cfg.tp_min_pips,
            tp_max_pips: cfg.tp_max_pips,
            rr_min: cfg.rr_min,
            rr_max: cfg.rr_max,
            atr_pips: None,
        },
    }
}

fn random_coarse_threshold(rng: &mut impl Rng) -> f64 {
    // F-273 narrow ladder by default; F-277 adaptive ladder when the
    // operator opts in via `install_adaptive_threshold_ladder` from
    // `run_discovery_cycle_with_progress`.
    let levels = current_threshold_ladder();
    levels[rng.random_range(0..levels.len())]
}

/// F-277 (2026-05-28): derive an adaptive threshold ladder from the
/// per-column magnitude profile of a feature cube. The ladder maps
/// to typical-signal-strength percentile points so the GA's random
/// init produces genes that fire at meaningful rates regardless of
/// the dataset's absolute scale.
///
/// Algorithm:
/// 1. For each column, collect `|value|` of finite rows (typically
///    the column's mean-deviated magnitude).
/// 2. Pool across columns into one sample (each column contributes
///    its median |value|).
/// 3. Compute 6 percentile points: [p10, p25, p50, p75, p90, p99].
/// 4. Clamp each to `[1e-4, 10.0]` to guard pathological zero-variance
///    or numerically explosive columns.
/// 5. Sort ascending (already is by construction).
///
/// Returns `None` when the cube is empty or has zero finite values.
pub fn derive_adaptive_threshold_ladder_from_features(
    features: &neoethos_data::FeatureFrame,
) -> anyhow::Result<Option<[f64; 6]>> {
    let n_cols = features.n_features();
    let n_rows = features.n_samples();
    if n_cols == 0 || n_rows == 0 {
        return Ok(None);
    }

    // Per-column median |value|. Project RAM-sized multi-column batches and
    // score independent batches in parallel. The former implementation issued
    // one full Vortex scan per column and fully sorted every column, making this
    // supposedly cheap pre-search calibration take hours on dense M5/M1 data.
    // `select_nth_unstable_by` returns the exact same rank (`len / 2`) as that
    // full sort without paying O(n log n); no approximate quantile is used.
    let projection_plan =
        neoethos_data::adaptive_feature_projection_plan(features, rayon::current_num_threads())?;
    tracing::info!(
        target: "neoethos_search::adaptive_thresholds",
        rows = n_rows,
        columns = n_cols,
        columns_per_projection = projection_plan.columns_per_batch,
        concurrent_projections = projection_plan.concurrent_batches,
        projection_budget_bytes = projection_plan.budget_bytes,
        "deriving the exact adaptive threshold ladder in bounded parallel projections"
    );
    derive_adaptive_threshold_ladder_with_projection_schedule(
        features,
        projection_plan.columns_per_batch,
        projection_plan.concurrent_batches,
    )
}

// Keep the mathematical reduction independent of the live RAM probe so tests
// can prove that different admitted schedules do not change the search space.
// Production always enters through the adaptive planner above.
/// Median abs value using the SAME convention as robust normalization:
/// average the two central values for an even valid count, odd middle otherwise.
fn adaptive_column_median_abs(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let middle = values.len() / 2;
    let median = if values.len() % 2 == 1 {
        let (_, &mut median, _) = values.select_nth_unstable_by(middle, f64::total_cmp);
        median
    } else {
        let (lower, &mut upper, _) = values.select_nth_unstable_by(middle, f64::total_cmp);
        let lower_middle = lower
            .iter()
            .copied()
            .max_by(f64::total_cmp)
            .expect("even-length sample has a lower middle");
        lower_middle * 0.5 + upper * 0.5
    };
    (median.is_finite() && median >= 0.0).then_some(median)
}

fn derive_adaptive_threshold_ladder_with_projection_schedule(
    features: &neoethos_data::FeatureFrame,
    columns_per_batch: usize,
    concurrent_batches: usize,
) -> anyhow::Result<Option<[f64; 6]>> {
    let n_cols = features.n_features();
    let n_rows = features.n_samples();
    if n_cols == 0 || n_rows == 0 {
        return Ok(None);
    }
    let column_indices = (0..n_cols).collect::<Vec<_>>();
    let mut per_col_median_abs: Vec<f64> = Vec::with_capacity(n_cols);
    let columns_per_batch = columns_per_batch.max(1);
    let wave_columns = columns_per_batch
        .saturating_mul(concurrent_batches.max(1))
        .max(1);
    for wave in column_indices.chunks(wave_columns) {
        let wave_medians = wave
            .par_chunks(columns_per_batch)
            .map(|indices| -> anyhow::Result<Vec<f64>> {
                let projection = features.project_columns(indices, 0..n_rows)?;
                anyhow::ensure!(
                    projection.columns.len() == indices.len(),
                    "adaptive-threshold projection returned {} columns for {} indices",
                    projection.columns.len(),
                    indices.len()
                );
                let mut medians = Vec::with_capacity(indices.len());
                for column in &projection.columns {
                    let abs_values = column
                        .values
                        .iter()
                        .zip(&column.validity)
                        .filter_map(|(value, validity)| validity.is_valid().then_some(value.abs()))
                        .collect::<Vec<_>>();
                    if let Some(median) = adaptive_column_median_abs(abs_values) {
                        medians.push(median);
                    }
                }
                Ok(medians)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        for medians in wave_medians {
            per_col_median_abs.extend(medians);
        }
    }

    if per_col_median_abs.is_empty() {
        return Ok(None);
    }

    // Pool: percentile points on the per-column-median sample.
    per_col_median_abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pct = |p: f64| -> f64 {
        let idx = ((p * (per_col_median_abs.len() as f64 - 1.0)).round() as usize)
            .min(per_col_median_abs.len() - 1);
        per_col_median_abs[idx].clamp(1e-4, 10.0)
    };

    let ladder = [
        pct(0.10),
        pct(0.25),
        pct(0.50),
        pct(0.75),
        pct(0.90),
        pct(0.99),
    ];
    // A randomized six-slot grid is useful only when every adjacent slot is a
    // distinct choice. Normalization can standardize the per-column median
    // statistic to the same value across genuinely different columns, and
    // even/odd median conventions can also create tiny artificial spreads.
    // Reject both equal and near-equal slots; the discovery owner then uses the
    // existing static fallback rather than silently weighting one threshold.
    for i in 1..ladder.len() {
        let scale = ladder[i].abs().max(ladder[i - 1].abs()).max(1.0);
        if ladder[i] < ladder[i - 1]
            || ladder[i] - ladder[i - 1] <= ADAPTIVE_THRESHOLD_LADDER_DISTINCT_REL_TOLERANCE * scale
        {
            return Ok(None);
        }
    }
    Ok(Some(ladder))
}

/// Reset every derived/financial metric on a Gene that was inherited from a
/// parent during crossover/mutation but is no longer accurate for the child.
pub fn reset_gene_metrics(gene: &mut Gene) {
    gene.fitness = 0.0;
    gene.sharpe_ratio = 0.0;
    gene.win_rate = 0.0;
    gene.max_drawdown = 0.0;
    gene.profit_factor = 0.0;
    gene.expectancy = 0.0;
    gene.trades_count = 0;
    gene.slice_pass_rate = 0.0;
    gene.consistency = 0.0;
}

pub fn new_random_gene(
    n_indicators: usize,
    max_indicators: usize,
    generation: usize,
    smc_cfg: &SmcSearchConfig,
    rng: &mut impl Rng,
) -> Gene {
    let min_indicators = 1;
    let max_indicators = max_indicators.clamp(min_indicators, n_indicators.max(1));
    let count = rng.random_range(min_indicators..=max_indicators);
    let sample = sample(rng, n_indicators.max(1), count);
    let indices: Vec<usize> = sample.iter().collect();
    let weights: Vec<f64> = (0..count).map(|_| random_coarse_weight(rng)).collect();
    let long_threshold = random_coarse_threshold(rng);
    let short_threshold = -random_coarse_threshold(rng);
    // SL/TP in pips, drawn from the band resolved for THIS dataset.
    //
    // History. Operator directive (2026-07-23) replaced a 100-pip ceiling with
    // the literals `sl ∈ [6, 20]`, `tp ∈ [12, 45]`, `rr ∈ [1.5, 2.5]`, to stop
    // the GA pinning TP at a target live trading never reached. That fixed the
    // symptom and created two new defects, both measured 2026-08-09:
    //   1. Those are M5 pip counts. H1 (ATR ≈ 12 pips) and H4 (ATR ≈ 30 pips)
    //      were literally inexpressible — the entire stop band sat inside one
    //      bar's range.
    //   2. `rr <= 2.5` with a 45-pip ceiling cannot reach the reward:risk a 2.0
    //      payoff floor demands (`tp >= 2·sl + 3·c`; 48.67 at sl = 20, c = 2.89).
    //      The gate and the search space contradicted each other.
    // `current_gene_stop_bounds()` is now the single owner of the band, in ATR
    // units, so it means the same thing on every timeframe. It falls back to the
    // exact literals above when no ATR scale is installed, so the
    // `neoethos-models` GA and every test fixture are unchanged.
    //
    // This buys no expected profit — see `ResolvedGeneStopBounds` for the
    // measurement. It makes shapes reachable; the expectancy gate decides which
    // of them survive.
    let bounds = current_gene_stop_bounds();
    let (sl_pips, tp_pips) = if rng.random_bool(0.2) {
        bounds.mid_pair()
    } else {
        let sl: f64 = rng.random_range(bounds.sl_min_pips..=bounds.sl_max_pips);
        let rr: f64 = rng.random_range(bounds.rr_min..=bounds.rr_max);
        let tp = (sl * rr).clamp(bounds.tp_min_pips, bounds.tp_max_pips);
        (sl, tp)
    };
    // Adaptive stops: seed a searchable volatility multiplier so the absolute
    // stop scales at entry. The sl/tp pair above still owns the dimensionless
    // reward:risk ratio, so every random gene can express a different exit.
    // Off → 0.0 = fixed-stop, byte-identical to before.
    let stop_vol_mult = if crate::stop_target::adaptive_stops_enabled() {
        rng.random_range(0.5..=3.0)
    } else {
        0.0
    };
    let strategy_id = format!("gene_{}_{}", rng.random_range(0..1_000_000u64), generation);
    let mut gene = Gene {
        indices,
        weights,
        long_threshold,
        short_threshold,
        fitness: 0.0,
        sharpe_ratio: 0.0,
        win_rate: 0.0,
        max_drawdown: 0.0,
        profit_factor: 0.0,
        expectancy: 0.0,
        trades_count: 0,
        generation,
        strategy_id,
        use_ob: false,
        use_fvg: false,
        use_liq_sweep: false,
        mtf_confirmation: true,
        use_premium_discount: false,
        use_inducement: false,
        use_bos: false,
        use_choch: false,
        use_eqh: false,
        use_eql: false,
        use_displacement: false,
        tp_pips,
        sl_pips,
        slice_pass_rate: 0.0,
        consistency: 0.0,
        stop_vol_mult,
    };
    randomize_smc_flags(&mut gene, smc_cfg, rng);
    enforce_min_structural_smc_flags(&mut gene, smc_cfg, rng);
    gene.normalize(n_indicators, 1);
    gene
}

pub fn generate_random_genes(
    n_genes: usize,
    n_indicators: usize,
    max_indicators: usize,
    generation: usize,
    smc_cfg: &SmcSearchConfig,
    rng: &mut impl Rng,
) -> Vec<Gene> {
    (0..n_genes)
        .map(|_| new_random_gene(n_indicators, max_indicators, generation, smc_cfg, rng))
        .collect()
}

pub fn unique_candidate_or_retry(
    mut candidate: Gene,
    seen: &mut SeenSignatureMemory,
    n_indicators: usize,
    max_indicators: usize,
    generation: usize,
    max_attempts: usize,
    smc_cfg: &SmcSearchConfig,
    rng: &mut impl Rng,
) -> Gene {
    candidate.normalize(n_indicators, 1);
    let accepted = seen.insert_gene(&candidate);
    #[cfg(test)]
    if !accepted {
        super::search_engine::record_retry_counter(
            super::search_engine::RetryDiagnosticCounterV1::DuplicateAttempt,
        );
    }
    if accepted {
        return candidate;
    }
    let attempts = max_attempts.max(1);
    for _ in 0..attempts {
        let mut probe = new_random_gene(n_indicators, max_indicators, generation, smc_cfg, rng);
        #[cfg(test)]
        super::search_engine::record_retry_counter(
            super::search_engine::RetryDiagnosticCounterV1::NewbornDraw,
        );
        probe.normalize(n_indicators, 1);
        if seen.insert_gene(&probe) {
            #[cfg(test)]
            super::search_engine::record_retry_counter(
                super::search_engine::RetryDiagnosticCounterV1::SuccessfulReplacement,
            );
            return probe;
        }
    }
    #[cfg(test)]
    super::search_engine::record_retry_counter(
        super::search_engine::RetryDiagnosticCounterV1::Exhaustion,
    );
    candidate
}

pub fn crossover(a: &Gene, b: &Gene, generation: usize, rng: &mut impl Rng) -> Gene {
    // Note: callers must pass the same `rng` they use elsewhere in the same
    // search; using a fresh `rand::rng()` here would break the deterministic
    // seed introduced for CPU/GPU parity (see search_engine::build_search_rng).
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    let split_a = a.indices.len() / 2;
    let split_b = b.indices.len() / 2;
    indices.extend_from_slice(&a.indices[..split_a]);
    indices.extend_from_slice(&b.indices[split_b..]);
    weights.extend_from_slice(&a.weights[..split_a]);
    weights.extend_from_slice(&b.weights[split_b..]);
    if indices.is_empty() {
        indices.push(*a.indices.first().unwrap_or(&0));
        weights.push(*a.weights.first().unwrap_or(&1.0));
    }
    let mut child = a.clone();
    child.indices = indices;
    child.weights = weights;
    child.strategy_id = format!("gene_{}_{}", rng.random_range(0..1_000_000u64), generation);
    child.generation = generation;
    reset_gene_metrics(&mut child);

    child.long_threshold = if rng.random_bool(0.5) {
        a.long_threshold
    } else {
        b.long_threshold
    };
    child.short_threshold = if rng.random_bool(0.5) {
        a.short_threshold
    } else {
        b.short_threshold
    };
    child.use_ob = if rng.random_bool(0.5) {
        a.use_ob
    } else {
        b.use_ob
    };
    child.use_fvg = if rng.random_bool(0.5) {
        a.use_fvg
    } else {
        b.use_fvg
    };
    child.use_liq_sweep = if rng.random_bool(0.5) {
        a.use_liq_sweep
    } else {
        b.use_liq_sweep
    };
    child.mtf_confirmation = if rng.random_bool(0.5) {
        a.mtf_confirmation
    } else {
        b.mtf_confirmation
    };
    child.use_premium_discount = if rng.random_bool(0.5) {
        a.use_premium_discount
    } else {
        b.use_premium_discount
    };
    child.use_inducement = if rng.random_bool(0.5) {
        a.use_inducement
    } else {
        b.use_inducement
    };
    child.use_bos = if rng.random_bool(0.5) {
        a.use_bos
    } else {
        b.use_bos
    };
    child.use_choch = if rng.random_bool(0.5) {
        a.use_choch
    } else {
        b.use_choch
    };
    child.use_eqh = if rng.random_bool(0.5) {
        a.use_eqh
    } else {
        b.use_eqh
    };
    child.use_eql = if rng.random_bool(0.5) {
        a.use_eql
    } else {
        b.use_eql
    };
    child.use_displacement = if rng.random_bool(0.5) {
        a.use_displacement
    } else {
        b.use_displacement
    };
    // Stop and target are one exit-geometry trait. Inheriting their halves from
    // different parents can manufacture an out-of-band reward:risk ratio that
    // neither parent was evaluated with.
    (child.sl_pips, child.tp_pips) = if rng.random_bool(0.5) {
        (a.sl_pips, a.tp_pips)
    } else {
        (b.sl_pips, b.tp_pips)
    };
    // Adaptive-stop multiplier is inherited like the fixed stops so the GA can
    // recombine it (0.0 on both parents => stays fixed-stop).
    child.stop_vol_mult = if rng.random_bool(0.5) {
        a.stop_vol_mult
    } else {
        b.stop_vol_mult
    };

    child
}

pub fn mutate(
    gene: &Gene,
    n_indicators: usize,
    max_indicators: usize,
    generation: usize,
    smc_cfg: &SmcSearchConfig,
    stagnant_generations: usize,
    rng: &mut impl Rng,
) -> Gene {
    let mut mutated = gene.clone();

    // Adaptive mutation rate based on stagnation
    let (num_mutations, intensity) = if stagnant_generations > 10 {
        (3, 1.5_f64) // Heavy exploration
    } else if stagnant_generations > 5 {
        (2, 1.2_f64) // Moderate exploration
    } else if stagnant_generations == 0 {
        (1, 0.5_f64) // Exploitation (improvement streak)
    } else {
        (1, 1.0_f64) // Normal
    };

    for _ in 0..num_mutations {
        match rng.random_range(0..4) {
            0 => {
                // 2026-06-28 anti-stagnation: EXPLOIT (intensity < 1.0) keeps tweaking
                // weights of the current indicators; but when STAGNANT (intensity high)
                // bias hard toward REPLACING indicators. The old 50/50 gate + 0.3
                // per-indicator swap meant ~10% of mutations ever changed an indicator,
                // so the GA stayed stuck optimising the SAME indicators ("doesn't change
                // indicators, tiny sample"). Now higher intensity → more full re-samples
                // + a higher per-indicator swap chance.
                let tweak_weights_prob = (0.5 / intensity as f64).clamp(0.0, 1.0);
                if !mutated.indices.is_empty()
                    && (intensity < 1.0 || rng.random_bool(tweak_weights_prob))
                {
                    let idx = rng.random_range(0..mutated.indices.len());
                    if rng.random_bool((0.6 * intensity as f64).clamp(0.0, 1.0)) {
                        mutated.indices[idx] = rng.random_range(0..n_indicators.max(1));
                    }
                    // Sign-flip is the cheapest contrarian move: same indicator,
                    // same magnitude, inverted contribution. Kept as a 25% branch
                    // so plain magnitude resampling still dominates.
                    if rng.random_bool(0.25) {
                        mutated.weights[idx] = -mutated.weights[idx];
                    } else {
                        mutated.weights[idx] = random_coarse_weight(rng);
                    }
                } else {
                    let min_indicators = 1;
                    let max_indicators = max_indicators.clamp(min_indicators, n_indicators.max(1));
                    let count = rng.random_range(min_indicators..=max_indicators);
                    let sample = sample(rng, n_indicators.max(1), count);
                    mutated.indices = sample.iter().collect();
                    mutated.weights = (0..count).map(|_| random_coarse_weight(rng)).collect();
                }
            }
            1 => {
                mutated.long_threshold = random_coarse_threshold(rng);
                mutated.short_threshold = -random_coarse_threshold(rng);
            }
            2 => {
                let range = 0.2 * intensity as f64;
                if mutated.stop_vol_mult > 0.0 {
                    // Adaptive-stop gene: both active exit traits evolve. The
                    // multiplier controls stop distance; tp/sl controls payoff.
                    mutated.stop_vol_mult = (mutated.stop_vol_mult
                        * rng.random_range((1.0 - range)..(1.0 + range)))
                    .clamp(0.3, 4.0);
                    let bounds = current_gene_stop_bounds();
                    let stop = if mutated.sl_pips.is_finite() && mutated.sl_pips > 0.0 {
                        mutated
                            .sl_pips
                            .clamp(bounds.sl_min_pips, bounds.sl_max_pips)
                    } else {
                        bounds.mid_pair().0
                    };
                    let current_reward_risk = crate::stop_target::effective_adaptive_reward_risk(
                        mutated.sl_pips,
                        mutated.tp_pips,
                        0.5 * (bounds.rr_min + bounds.rr_max),
                    );
                    let reward_risk = (current_reward_risk
                        * rng.random_range((1.0 - range)..(1.0 + range)))
                    .clamp(bounds.rr_min, bounds.rr_max);
                    mutated.sl_pips = stop;
                    mutated.tp_pips = stop * reward_risk;
                } else {
                    // Fixed-stop gene: mutate inside the SAME band the
                    // initialiser draws from, resolved for this dataset. Using
                    // the literals here while the initialiser used ATR units
                    // would let mutation drag every gene back to the M5 band one
                    // generation after birth — which is how a per-timeframe fix
                    // silently becomes an M5-only fix.
                    let bounds = current_gene_stop_bounds();
                    mutated.tp_pips = (mutated.tp_pips
                        * rng.random_range((1.0 - range)..(1.0 + range)))
                    .clamp(bounds.tp_min_pips, bounds.tp_max_pips);
                    mutated.sl_pips = (mutated.sl_pips
                        * rng.random_range((1.0 - range)..(1.0 + range)))
                    .clamp(bounds.sl_min_pips, bounds.sl_max_pips);
                }
            }
            _ => {
                // In exploitation mode, reduce the chance of randomly flipping SMC flags
                if intensity >= 1.0 || rng.random_bool(0.3) {
                    randomize_smc_flags(&mut mutated, smc_cfg, rng);
                }
            }
        }
    }

    if rng.random_bool(0.25 * intensity as f64) {
        enforce_min_structural_smc_flags(&mut mutated, smc_cfg, rng);
    }
    mutated.strategy_id = format!("gene_{}_{}", rng.random_range(0..1_000_000u64), generation);
    mutated.generation = generation;
    reset_gene_metrics(&mut mutated);
    if mutated.has_total_weight_cancellation(n_indicators) {
        // A finite crossover child can algebraically cancel every term. Treat
        // that as a degenerate offspring requiring explicit replacement, using
        // the existing seeded newborn mechanism; normalize fixed +1 fallback
        // would silently turn a canceled inherited term into a strong active one.
        return new_random_gene(n_indicators, max_indicators, generation, smc_cfg, rng);
    }
    mutated.normalize(n_indicators, 1);
    mutated
}

// **Scoring Phase B COMPLETE (2026-05-25 verbose-build pass)**: the
// previous `score_from_metrics` deprecated shim was deleted. The only
// in-crate caller (`apply_metrics` below) calls
// `crate::scoring::ga_fitness` directly. External callers reaching for
// the old name now get a compile error pointing at the canonical
// function — which is what we want.

pub fn apply_metrics(
    genes: &mut [Gene],
    metrics: &[[f64; 11]],
    config: &super::strategy_gene::EvaluationConfig,
    evaluation_span_days: f64,
    monthly_returns: Option<&[Vec<f64>]>,
) {
    for (index, (gene, m)) in genes.iter_mut().zip(metrics.iter()).enumerate() {
        gene.fitness =
            if matches!(
                config.fitness_table,
                crate::scoring::FitnessTable::MonthlyUpsideV1
                    | crate::scoring::FitnessTable::MonthlyUpsideDdAverseV1
                    | crate::scoring::FitnessTable::MonthlyUpsideDdBudgetV1
            ) {
                // Opt-in B9 / B9-dd-averse (docs section 14.7): the explicit
                // selector wins over the legacy boolean chain and reads the
                // evaluator's completed-month return vector.
                // PopulationEvaluationV1::apply refuses missing/misaligned
                // transport; a legitimately empty per-gene series still takes
                // the scorer's -100.0 guard.
                let months = monthly_returns
                    .and_then(|series| series.get(index))
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                if config.fitness_table == crate::scoring::FitnessTable::MonthlyUpsideDdAverseV1 {
                    crate::scoring::ga_fitness_monthly_upside_dd_averse_v1(months, m)
                } else if config.fitness_table
                    == crate::scoring::FitnessTable::MonthlyUpsideDdBudgetV1
                {
                    crate::scoring::ga_fitness_monthly_upside_dd_budget_v1(months, m)
                } else {
                    crate::scoring::ga_fitness_monthly_upside_v1(months, m)
                }
            } else if let Some(goal) = config.growth_goal {
                crate::scoring::ga_fitness_goal(m, config.initial_equity, evaluation_span_days, goal)
            } else if config.growth_objective {
                crate::scoring::ga_fitness_growth(m)
            } else {
                crate::scoring::ga_fitness(m)
            };
        gene.sharpe_ratio = m[1];
        gene.max_drawdown = m[3];
        gene.win_rate = m[4];
        gene.profit_factor = m[5];
        gene.expectancy = m[6];
        gene.trades_count = m[8].max(0.0) as usize;
        gene.slice_pass_rate = 1.0;
        gene.consistency = m[9];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The band must mean the same thing on every timeframe, and the absolute
    /// band must still be exactly the pre-2026-08-09 literals when no scale is
    /// installed — otherwise this "fix" silently changes the `neoethos-models`
    /// GA and every test fixture that never installs one.
    ///
    /// These tests mutate process-wide state, so they run in ONE test function.
    /// Split into three, cargo's thread-per-test would interleave the installs
    /// and the assertions would depend on scheduling.
    #[test]
    fn the_stop_band_scales_with_the_dataset_and_falls_back_to_the_old_literals() {
        clear_gene_stop_atr_scale();
        let absolute = current_gene_stop_bounds();
        assert_eq!(absolute.atr_pips, None);
        assert_eq!(absolute.sl_min_pips, 6.0, "the pre-2026-08-09 stop floor");
        assert_eq!(
            absolute.sl_max_pips, 20.0,
            "the pre-2026-08-09 stop ceiling"
        );
        assert_eq!(absolute.tp_min_pips, 12.0);
        assert_eq!(absolute.tp_max_pips, 45.0);

        // EURUSD M5: ATR ≈ 5 pips. The band the literals were tuned for.
        install_gene_stop_atr_scale(5.0);
        let m5 = current_gene_stop_bounds();
        assert_eq!(m5.atr_pips, Some(5.0));
        assert!(
            (m5.sl_min_pips - 5.0).abs() < 1e-9 && (m5.sl_max_pips - 20.0).abs() < 1e-9,
            "M5 must land on roughly the old band, got [{}, {}]",
            m5.sl_min_pips,
            m5.sl_max_pips
        );

        // EURUSD H1: ATR ≈ 12 pips. Under the old literals the ENTIRE stop band
        // ([6, 20]) sat inside one bar's range, so an H1 gene could not express
        // a stop the market could not reach by accident. This is the defect that
        // made every "search a higher timeframe" suggestion void.
        install_gene_stop_atr_scale(12.0);
        let h1 = current_gene_stop_bounds();
        assert!(
            h1.sl_min_pips >= 12.0,
            "an H1 stop floor must be at least one bar's range, got {}",
            h1.sl_min_pips
        );
        assert!(h1.sl_max_pips > 20.0, "H1 must reach past the old ceiling");

        // H4: ATR ≈ 30 pips. The old ceiling was below the old FLOOR here.
        install_gene_stop_atr_scale(30.0);
        let h4 = current_gene_stop_bounds();
        assert!(h4.sl_min_pips > 20.0 && h4.tp_max_pips > 45.0);

        // The reward:risk a 2.0 payoff floor demands must be INSIDE the band.
        // Payoff with barriers only is (tp - c)/(sl + c); at c = 2.89 pips a
        // 20-pip stop needs tp >= 48.67, i.e. rr >= 2.43. The old rr ceiling was
        // 2.50 with a 45-pip tp clamp, so the draw that satisfied it was clipped
        // away before it could be evaluated.
        install_gene_stop_atr_scale(5.0);
        let band = current_gene_stop_bounds();
        let c = 2.89_f64;
        let sl = band.sl_max_pips;
        let needed_tp = 2.0 * sl + 3.0 * c;
        assert!(
            band.rr_max * sl >= needed_tp && band.tp_max_pips >= needed_tp,
            "a 2.0 payoff floor needs tp >= {needed_tp:.2} at sl {sl:.2}; band allows \
             rr_max*sl = {:.2}, tp_max = {:.2}",
            band.rr_max * sl,
            band.tp_max_pips
        );

        // A degenerate scale must CLEAR, never install a band that collapses to
        // a point — `random_range(0.0..=0.0)` is not a search.
        install_gene_stop_atr_scale(0.0);
        assert_eq!(current_gene_stop_bounds().atr_pips, None);
        install_gene_stop_atr_scale(f64::NAN);
        assert_eq!(current_gene_stop_bounds().atr_pips, None);

        // Every gene the initialiser and the mutator can produce must sit inside
        // the resolved band — a mutator using different bounds from the
        // initialiser drags the population back to the M5 band one generation
        // after birth.
        install_gene_stop_atr_scale(30.0);
        let band = current_gene_stop_bounds();
        let smc = crate::genetic::SmcSearchConfig::default();
        let mut rng = rand::rng();
        let mut distinct_adaptive_ratios = std::collections::BTreeSet::new();
        let mut adaptive_ratio_mutated = false;
        for _ in 0..200 {
            let gene = new_random_gene(8, 4, 0, &smc, &mut rng);
            let gene_ratio = gene.tp_pips / gene.sl_pips;
            assert!(gene_ratio >= band.rr_min - 1e-9 && gene_ratio <= band.rr_max + 1e-9);
            distinct_adaptive_ratios.insert((gene_ratio * 1_000_000.0).round() as i64);
            assert!(
                gene.sl_pips >= band.sl_min_pips - 1e-9 && gene.sl_pips <= band.sl_max_pips + 1e-9,
                "initialised sl {} outside [{}, {}]",
                gene.sl_pips,
                band.sl_min_pips,
                band.sl_max_pips
            );
            let mutated = mutate(&gene, 8, 4, 1, &smc, 0, &mut rng);
            let mutated_ratio = mutated.tp_pips / mutated.sl_pips;
            assert!(
                mutated_ratio >= band.rr_min - 1e-9 && mutated_ratio <= band.rr_max + 1e-9,
                "mutated adaptive ratio {mutated_ratio} outside [{}, {}]",
                band.rr_min,
                band.rr_max
            );
            adaptive_ratio_mutated |= (mutated_ratio - gene_ratio).abs() > 1e-12;
            assert!(
                mutated.sl_pips >= band.sl_min_pips - 1e-9
                    && mutated.sl_pips <= band.sl_max_pips + 1e-9,
                "mutated sl {} outside [{}, {}]",
                mutated.sl_pips,
                band.sl_min_pips,
                band.sl_max_pips
            );
        }
        assert!(
            distinct_adaptive_ratios.len() > 20,
            "random population collapsed to only {} adaptive exit ratios",
            distinct_adaptive_ratios.len()
        );
        assert!(adaptive_ratio_mutated, "adaptive ratio never mutated");

        use rand::{SeedableRng, rngs::StdRng};
        let names = crate::genetic::seed_templates::multi_tf_test_feature_names();
        let seed_count = names.len();
        let seeds = crate::genetic::seed_templates::seed_professional_templates(
            50,
            &names,
            seed_count,
            &mut StdRng::seed_from_u64(13),
        );
        assert!(
            !seeds.is_empty(),
            "full multi-TF schema must resolve templates"
        );
        for gene in &seeds {
            assert!(
                gene.sl_pips >= band.sl_min_pips - 1e-9 && gene.sl_pips <= band.sl_max_pips + 1e-9,
                "template sl {} outside H4 band [{}, {}]",
                gene.sl_pips,
                band.sl_min_pips,
                band.sl_max_pips
            );
            assert!(
                gene.tp_pips >= band.tp_min_pips - 1e-9 && gene.tp_pips <= band.tp_max_pips + 1e-9,
                "template tp {} outside H4 band [{}, {}]",
                gene.tp_pips,
                band.tp_min_pips,
                band.tp_max_pips
            );
            let ratio = gene.tp_pips / gene.sl_pips;
            assert!(
                ratio >= band.rr_min - 1e-9 && ratio <= band.rr_max + 1e-9,
                "template rr {ratio} outside [{}, {}]",
                band.rr_min,
                band.rr_max
            );
        }

        // Leave the process as we found it so no later test inherits an H4 scale.
        clear_gene_stop_atr_scale();
    }

    #[test]
    fn rank_weights_follow_candidate_order() {
        let weights = rank_weights(&[10, 20, 30, 40]);
        assert_eq!(weights, vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn rank_draw_matches_legacy_choices_and_rng_state_including_sparse_candidate_ids() {
        use rand::{SeedableRng, rngs::StdRng};
        for count in [0, 1, 2, 7, 200, 2_107, 21_069] {
            let candidates = (0..count).map(|i| 3 * (count - i) + 7).collect::<Vec<_>>();
            let weights = rank_weights(&candidates);
            let mut old_rng = StdRng::seed_from_u64(731);
            let mut new_rng = old_rng.clone();
            for _ in 0..1_024 {
                let expected = if count == 0 {
                    0
                } else {
                    candidates[draw_weighted_offset(&weights, &mut old_rng)]
                };
                let actual = select_parent_index(
                    &[],
                    &candidates,
                    ParentSelectionPolicy::RankWeighted,
                    4,
                    0.75,
                    &mut new_rng,
                );
                assert_eq!(actual, expected, "population={count}");
            }
            assert_eq!(
                new_rng.random::<u64>(),
                old_rng.random::<u64>(),
                "RNG draw count"
            );
        }
    }

    #[test]
    fn rank_cumulative_boundaries_keep_the_earlier_candidate() {
        for count in [1usize, 2, 11, 21_069, 1 << 26] {
            assert_eq!(rank_offset_for_target(count, 0.0), 0);
            let n = count as u64;
            for offset in [0, count / 2, count - 1] {
                let terms = offset as u64 + 1;
                let boundary = (terms * (2 * n - terms + 1) / 2) as f64;
                assert_eq!(rank_offset_for_target(count, boundary), offset);
                assert_eq!(rank_offset_for_target(count, boundary.next_down()), offset);
                if offset + 1 < count {
                    assert_eq!(
                        rank_offset_for_target(count, boundary.next_up()),
                        offset + 1
                    );
                }
            }
        }
    }

    #[test]
    fn survivor_removal_preserves_rank_and_tournament_draw_sequences() {
        use rand::{SeedableRng, rngs::StdRng};
        let scores = (0..257).map(|i| 1.0 / (i + 1) as f64).collect::<Vec<_>>();
        for policy in [
            SurvivorSelectionPolicy::RankWeighted,
            SurvivorSelectionPolicy::Tournament,
        ] {
            for count in [0, 1, 27, 257, 300] {
                let mut old_rng = StdRng::seed_from_u64(51);
                let mut new_rng = old_rng.clone();
                let mut available = (0..scores.len()).collect::<Vec<_>>();
                let mut expected = Vec::new();
                while !available.is_empty() && expected.len() < count {
                    let idx = if policy == SurvivorSelectionPolicy::RankWeighted {
                        available[draw_weighted_offset(&rank_weights(&available), &mut old_rng)]
                    } else {
                        select_parent_index(
                            &scores,
                            &available,
                            ParentSelectionPolicy::Tournament,
                            4,
                            0.75,
                            &mut old_rng,
                        )
                    };
                    expected.push(idx);
                    available.retain(|candidate| *candidate != idx);
                }
                expected.sort_unstable();
                assert_eq!(
                    select_survivor_indices(&scores, count, policy, 0.75, 4, &mut new_rng),
                    expected
                );
                assert_eq!(new_rng.random::<u64>(), old_rng.random::<u64>());
            }
        }
    }

    #[test]
    #[ignore = "bounded bookkeeping benchmark; run explicitly, not on every test invocation"]
    fn benchmark_rank_parent_selection_at_m30_population() {
        use rand::{SeedableRng, rngs::StdRng};
        use std::{hint::black_box, time::Instant};
        let candidates = (0..21_069).collect::<Vec<_>>();
        let mut old_rng = StdRng::seed_from_u64(19);
        let mut new_rng = old_rng.clone();
        let started = Instant::now();
        let expected = (0..10_000)
            .map(|_| {
                // Include the old per-draw allocation/reduction, not a cached oracle.
                black_box(draw_weighted_offset(
                    &rank_weights(black_box(&candidates)),
                    &mut old_rng,
                ))
            })
            .collect::<Vec<_>>();
        let old_elapsed = started.elapsed();
        let started = Instant::now();
        let actual = (0..10_000)
            .map(|_| {
                black_box(select_parent_index(
                    &[],
                    black_box(&candidates),
                    ParentSelectionPolicy::RankWeighted,
                    4,
                    0.75,
                    &mut new_rng,
                ))
            })
            .collect::<Vec<_>>();
        let new_elapsed = started.elapsed();
        assert_eq!(actual, expected);
        assert_eq!(new_rng.random::<u64>(), old_rng.random::<u64>());
        eprintln!(
            "GA rank benchmark population=21069 draws=10000 legacy_s={:.6} optimized_s={:.6} exact_choices=true debug_assertions={}",
            old_elapsed.as_secs_f64(),
            new_elapsed.as_secs_f64(),
            cfg!(debug_assertions)
        );
    }

    #[test]
    fn zero_weight_fallback_is_deterministic() {
        let mut rng = rand::rng();
        assert_eq!(draw_weighted_offset(&[0.0, f64::NAN, -1.0], &mut rng), 0);
    }

    #[test]
    fn reset_gene_metrics_clears_parent_scores() {
        let mut gene = Gene {
            fitness: 3.2,
            sharpe_ratio: 1.4,
            win_rate: 0.7,
            max_drawdown: 0.05,
            profit_factor: 1.8,
            expectancy: 12.0,
            trades_count: 42,
            slice_pass_rate: 0.9,
            consistency: 0.6,
            ..Default::default()
        };

        reset_gene_metrics(&mut gene);

        assert_eq!(gene.fitness, 0.0);
        assert_eq!(gene.sharpe_ratio, 0.0);
        assert_eq!(gene.win_rate, 0.0);
        assert_eq!(gene.max_drawdown, 0.0);
        assert_eq!(gene.profit_factor, 0.0);
        assert_eq!(gene.expectancy, 0.0);
        assert_eq!(gene.trades_count, 0);
        assert_eq!(gene.slice_pass_rate, 0.0);
        assert_eq!(gene.consistency, 0.0);
    }

    #[test]
    fn seen_signature_memory_runtime_overrides_defaults_match_legacy_env_defaults() {
        let defaults = SeenSignatureMemoryRuntimeOverrides::default();
        assert_eq!(defaults.flush_every, 4096);
        assert_eq!(defaults.load_max, 3_000_000);
        assert_eq!(defaults.max_entries, 3_000_000);
        assert!(defaults.file_path.is_none());
    }

    #[test]
    fn seen_signature_from_settings_default_matches_env_default() {
        // Behavior-preservation gate (config-consolidation S2f): a fresh
        // `Settings` reproduces the engine seen-signature defaults exactly.
        let s = neoethos_core::Settings::default();
        assert_eq!(
            SeenSignatureMemoryRuntimeOverrides::from_settings(&s),
            SeenSignatureMemoryRuntimeOverrides::default()
        );
    }

    /// `0` used to mean UNBOUNDED — a `HashSet` + `VecDeque` growing for the
    /// whole run, on the sentinel someone types for "no limit". It now means
    /// DERIVE, and the derived value must be a real, bounded cap.
    #[test]
    fn zero_max_entries_derives_a_bounded_cap_instead_of_unbounded() {
        let mut s = neoethos_core::Settings::default();
        s.models.seen_signature_runtime.max_entries = 0;
        let resolved = SeenSignatureMemoryRuntimeOverrides::from_settings(&s);
        assert_ne!(
            resolved.max_entries,
            usize::MAX,
            "0 must derive from RAM, never resolve to unbounded"
        );
        assert!(
            resolved.max_entries >= 1_000_000,
            "the derived cap must not fall below the floor: {}",
            resolved.max_entries
        );
        assert!(
            resolved.max_entries <= 64_000_000,
            "the derived cap must not exceed the ceiling: {}",
            resolved.max_entries
        );
    }

    #[test]
    fn current_seen_signature_memory_runtime_overrides_returns_legal_values() {
        let observed = current_seen_signature_memory_runtime_overrides();
        assert!(observed.flush_every >= 1);
        assert!(observed.max_entries >= 1);
    }

    // ─── F-277 adaptive threshold ladder tests ────────────────────────

    // Empty feature frames are structurally invalid. Every ladder fixture now
    // uses the same f64/validity/provenance contract as production.
    fn frame_of(a: ndarray::Array2<f64>) -> neoethos_data::FeatureFrame {
        let names = (0..a.ncols())
            .map(|column| format!("test_feature_{column}"))
            .collect();
        neoethos_data::test_fixtures::ctrader_test_feature_frame_from_matrix(
            neoethos_data::test_fixtures::canonical_test_timestamps(a.nrows()),
            names,
            a,
        )
        .expect("valid f64 adaptive-threshold fixture")
    }

    #[test]
    fn derive_ladder_returns_none_for_all_nan_cube() {
        let mut data: ndarray::Array2<f64> = ndarray::Array2::zeros((10, 3));
        data.fill(f64::NAN);
        assert!(
            derive_adaptive_threshold_ladder_from_features(&frame_of(data))
                .expect("all-NaN scan must not fail")
                .is_none()
        );
    }

    fn diverse_raw_ladder_frame() -> neoethos_data::FeatureFrame {
        let n = 100usize;
        let scales = [
            0.001_f64, 0.002, 0.004, 0.007, 0.010, 0.020, 0.040, 0.080, 0.100, 0.120, 0.140, 0.160,
        ];
        let mut data: ndarray::Array2<f64> = ndarray::Array2::zeros((n, scales.len()));
        for r in 0..n {
            for (j, scale) in scales.iter().enumerate() {
                data[(r, j)] = (r as f64 + 1.0) * scale;
            }
        }
        frame_of(data)
    }

    fn normalized_frame_of(a: ndarray::Array2<f64>) -> std::sync::Arc<neoethos_data::FeatureFrame> {
        let raw = std::sync::Arc::new(frame_of(a));
        let n = raw.n_samples();
        let fit = raw
            .fit_normalization(0..n, false, &neoethos_data::FeatureBuildControl::default())
            .expect("robust normalization fit");
        std::sync::Arc::new(
            raw.with_fitted_normalization(&fit)
                .expect("materialize robust normalization"),
        )
    }

    #[test]
    fn derive_ladder_produces_monotonic_ascending_values() {
        // A RAW fixture needs enough distinct per-column medians to supply six
        // distinct quantiles. This is not a normalized-collapse fixture.
        let frame = diverse_raw_ladder_frame();
        let ladder = derive_adaptive_threshold_ladder_from_features(&frame)
            .expect("adaptive-threshold scan must not fail")
            .expect("diverse raw cube must produce ladder");
        for i in 1..ladder.len() {
            assert!(
                ladder[i] > ladder[i - 1],
                "ladder[{i}]={} must be distinctly above ladder[{}]={}",
                ladder[i],
                i - 1,
                ladder[i - 1]
            );
        }
    }

    #[test]
    fn derive_ladder_rejects_mad_normalized_odd_collapse() {
        let n = 63usize;
        let mut data: ndarray::Array2<f64> = ndarray::Array2::zeros((n, 8));
        let scales = [0.002_f64, 0.05, 0.7, 4.0, 0.001, 0.02];
        for r in 0..n {
            let centered = r as f64 - 31.0;
            for (j, scale) in scales.iter().enumerate() {
                data[(r, j)] = centered * scale;
            }
            data[(r, 6)] = if r % 17 == 0 { 1.0 } else { 0.0 };
            data[(r, 7)] = if r % 29 == 0 { 1.0 } else { 0.0 };
        }
        let frame = normalized_frame_of(data);
        assert!(
            derive_adaptive_threshold_ladder_from_features(&frame)
                .expect("adaptive scan")
                .is_none(),
            "normalized diverse columns collapse to one median statistic; fallback is required"
        );
    }

    #[test]
    fn adaptive_column_median_matches_robust_mean_of_central_pair() {
        assert_eq!(adaptive_column_median_abs(vec![3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(
            adaptive_column_median_abs(vec![4.0, 1.0, 3.0, 2.0]),
            Some(2.5)
        );
    }

    #[test]
    fn adaptive_column_median_handles_even_extremes_without_overflow() {
        let median =
            adaptive_column_median_abs(vec![f64::MAX, f64::MAX]).expect("finite extreme median");
        assert!(
            median.is_finite(),
            "overflow-safe half-sum must stay finite"
        );
        let median = adaptive_column_median_abs(vec![f64::MAX, f64::MAX / 2.0, 1.0, 2.0])
            .expect("finite mixed-extreme median");
        assert!(
            median.is_finite(),
            "overflow-safe half-sum must stay finite"
        );
    }

    #[test]
    fn derive_ladder_rejects_asymmetric_even_shape_collapse() {
        let n = 64usize;
        let shapes: [fn(f64) -> f64; 8] = [
            |x| x,
            |x| x * x,
            |x| x * x * x,
            |x| x.sqrt(),
            |x| x.ln(),
            |x| (x / 20.0).exp(),
            |x| 1.0 / x,
            |x| (x / 5.0).sin(),
        ];
        let mut data: ndarray::Array2<f64> = ndarray::Array2::zeros((n, shapes.len()));
        for r in 0..n {
            let x = (r + 1) as f64;
            for (j, shape) in shapes.iter().enumerate() {
                data[(r, j)] = shape(x);
            }
        }
        let frame = normalized_frame_of(data);
        assert!(
            derive_adaptive_threshold_ladder_from_features(&frame)
                .expect("adaptive scan")
                .is_none(),
            "asymmetric even normalized shapes must collapse under the corrected median convention"
        );
    }

    #[test]
    fn derive_ladder_rejects_asymmetric_even_normalized_collapse() {
        let n = 64usize;
        let mut data: ndarray::Array2<f64> = ndarray::Array2::zeros((n, 8));
        let scales = [0.003_f64, 0.02, 0.3, 2.0, 0.001, 0.05, 0.9, 4.0];
        for (j, scale) in scales.iter().enumerate() {
            let mut values = Vec::with_capacity(n);
            for r in 0..n / 2 {
                values.push(-((r + 1) as f64) * scale);
            }
            for r in 0..n / 2 {
                let positive = if r == n / 2 - 1 {
                    100.0
                } else {
                    (r + 1) as f64
                };
                values.push(positive * scale);
            }
            for (r, value) in values.into_iter().enumerate() {
                data[(r, j)] = value;
            }
        }
        let frame = normalized_frame_of(data);
        assert!(
            derive_adaptive_threshold_ladder_from_features(&frame)
                .expect("adaptive scan")
                .is_none(),
            "asymmetric even normalized columns must fall back, not retain artificial spread"
        );
    }

    #[test]
    fn derive_ladder_rejects_continuous_sparse_mixture() {
        let n = 64usize;
        let mut data: ndarray::Array2<f64> = ndarray::Array2::zeros((n, 8));
        let scales = [0.002_f64, 0.05, 0.7, 4.0];
        for r in 0..n {
            let centered = r as f64 - 31.5;
            for (j, scale) in scales.iter().enumerate() {
                data[(r, j)] = centered * scale;
            }
            for (j, modulus) in [(4usize, 7usize), (5, 11), (6, 13), (7, 17)] {
                data[(r, j)] = if r % modulus == 0 { 1.0 } else { 0.0 };
            }
        }
        let frame = normalized_frame_of(data);
        assert!(
            derive_adaptive_threshold_ladder_from_features(&frame)
                .expect("adaptive scan")
                .is_none(),
            "four continuous + four sparse normalized columns must fall back"
        );
    }

    #[test]
    fn derive_ladder_rejects_sparse_only_normalized_columns() {
        let n = 64usize;
        let mut data: ndarray::Array2<f64> = ndarray::Array2::zeros((n, 6));
        for r in 0..n {
            for j in 0..6 {
                data[(r, j)] = if r % (7 + j * 2) == 0 { 1.0 } else { 0.0 };
            }
        }
        let frame = normalized_frame_of(data);
        assert!(
            derive_adaptive_threshold_ladder_from_features(&frame)
                .expect("adaptive scan")
                .is_none(),
            "sparse-only normalized cube must fall back explicitly"
        );
        clear_adaptive_threshold_ladder();
        assert_eq!(current_threshold_ladder(), STATIC_THRESHOLD_LADDER);
    }

    fn ladder_vortex_copy(
        features: &neoethos_data::FeatureFrame,
        root: &std::path::Path,
    ) -> neoethos_data::FeatureFrame {
        use neoethos_data::core::feature_run_lease::FeatureRunLease;
        use neoethos_data::core::vortex_feature_store::{
            VortexFeatureStore, VortexFeatureStoreOptions,
        };

        let neoethos_data::FeatureData::InMemory(columns) = &features.data else {
            panic!("ladder fixture must start in RAM")
        };
        let lease = std::sync::Arc::new(
            FeatureRunLease::create(root, "adaptive-ladder-parity").expect("isolated Vortex lease"),
        );
        let store = VortexFeatureStore::create(
            lease,
            &features.timestamps,
            columns,
            VortexFeatureStoreOptions::default(),
        )
        .expect("persist exact ladder fixture");
        neoethos_data::FeatureFrame::from_vortex(
            features.timestamps.clone(),
            store,
            features.plan().clone(),
            features.provenance().clone(),
        )
        .expect("Vortex ladder frame")
    }

    #[test]
    fn derive_ladder_matches_hand_calculated_median_and_percentile_ranks() {
        use FeatureCellValidity::{Gap, Valid, Warmup};
        use neoethos_data::{FeatureCellValidity, FeatureColumnF64};

        let central = f64::from_bits(0.8_f64.to_bits() + 1);
        let medians = [
            0.0,
            f64::MIN_POSITIVE,
            0.001,
            0.1,
            central,
            1.5,
            3.0,
            9.0,
            20.0,
        ];
        let mut columns = medians
            .into_iter()
            .enumerate()
            .map(|(index, median)| {
                // Five valid magnitudes: [0.25m, 0.5m, m, 2m, 3m]. The odd
                // middle is exactly m under either median convention.
                FeatureColumnF64::new(
                    format!("known_median_{index}"),
                    vec![
                        median * 0.25,
                        median * 0.5,
                        median,
                        median * 2.0,
                        median * 3.0,
                        f64::MAX,
                        -99.0,
                    ],
                    vec![Valid, Valid, Valid, Valid, Valid, Warmup, Gap],
                )
                .expect("known median column")
            })
            .collect::<Vec<_>>();
        columns.push(
            FeatureColumnF64::new("all_invalid", vec![f64::NAN; 7], vec![Warmup; 7])
                .expect("all-invalid column"),
        );
        let ram = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
            neoethos_data::test_fixtures::canonical_test_timestamps(7),
            columns,
        )
        .expect("analytic ladder fixture");
        let temporary = tempfile::tempdir().expect("isolated analytic Vortex root");
        let vortex = ladder_vortex_copy(&ram, temporary.path());
        // Nine contributing columns: round(p * 8) selects 1, 2, 4, 6, 7, 8.
        // The first and last selected values exercise both explicit clamps.
        let expected = [1e-4_f64, 0.001, central, 3.0, 9.0, 10.0].map(f64::to_bits);
        for frame in [&ram, &vortex] {
            let actual = derive_adaptive_threshold_ladder_from_features(frame)
                .expect("analytic ladder scan")
                .expect("known medians produce a ladder");
            assert_eq!(actual.map(f64::to_bits), expected);
        }
    }

    fn independent_threshold_ladder(frame: &neoethos_data::FeatureFrame) -> Option<[u64; 6]> {
        use neoethos_data::FeatureData;
        let FeatureData::InMemory(raw_columns) = &frame.data else {
            unreachable!()
        };
        let mut medians = raw_columns
            .iter()
            .filter_map(|column| {
                let mut magnitudes = column
                    .values
                    .iter()
                    .zip(&column.validity)
                    .filter(|(value, validity)| value.is_finite() && validity.is_valid())
                    .map(|(value, _)| value.abs())
                    .collect::<Vec<_>>();
                if magnitudes.is_empty() {
                    return None;
                }
                magnitudes.sort_by(f64::total_cmp);
                let middle = magnitudes.len() / 2;
                if magnitudes.len() % 2 == 0 {
                    // Overflow-safe mathematical average, matching the norm.
                    Some(magnitudes[middle - 1] * 0.5 + magnitudes[middle] * 0.5)
                } else {
                    magnitudes.get(middle).copied()
                }
            })
            .collect::<Vec<_>>();
        if medians.is_empty() {
            return None;
        }
        medians.sort_by(f64::total_cmp);
        let pct = |percentile: f64| -> f64 {
            let index = (percentile * (medians.len() - 1) as f64).round() as usize;
            medians[index].clamp(1e-4, 10.0)
        };
        let ladder = [
            pct(0.10),
            pct(0.25),
            pct(0.50),
            pct(0.75),
            pct(0.90),
            pct(0.99),
        ];
        // Independent copy of the documented fallback contract: the six
        // randomized slots are useful only while adjacent values differ.
        let tolerance = 1e-9_f64;
        let distinct = ladder.windows(2).all(|pair| {
            let scale = pair[0].abs().max(pair[1].abs()).max(1.0);
            pair[1] > pair[0] && pair[1] - pair[0] > tolerance * scale
        });
        distinct.then(|| ladder.map(f64::to_bits))
    }

    fn assert_adaptive_ladder_schedule_parity(
        ram: &neoethos_data::FeatureFrame,
        expected: Option<[u64; 6]>,
    ) {
        let temporary = tempfile::tempdir().expect("isolated parity Vortex root");
        let vortex = ladder_vortex_copy(ram, temporary.path());
        for workers in [1, 2, 5] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .expect("isolated parity worker pool");
            for frame in [ram, &vortex] {
                let actual = pool
                    .install(|| derive_adaptive_threshold_ladder_from_features(frame))
                    .expect("live-planned ladder")
                    .map(|ladder| ladder.map(f64::to_bits));
                assert_eq!(actual, expected, "workers={workers}");
                for (batch_columns, concurrent_batches) in [
                    (1, 1),
                    (1, 3),
                    (2, 3),
                    (5, 2),
                    (19, 1),
                    (usize::MAX, usize::MAX),
                ] {
                    let actual = pool
                        .install(|| {
                            derive_adaptive_threshold_ladder_with_projection_schedule(
                                frame,
                                batch_columns,
                                concurrent_batches,
                            )
                        })
                        .expect("scheduled ladder")
                        .map(|ladder| ladder.map(f64::to_bits));
                    assert_eq!(
                        actual, expected,
                        "workers={workers}, batch_columns={batch_columns}, concurrent_batches={concurrent_batches}",
                    );
                }
            }
        }
    }

    #[test]
    fn derive_ladder_edge_fixture_is_bit_exact_across_ram_vortex_workers_and_schedules() {
        use FeatureCellValidity::{Gap, Valid, Warmup};
        use neoethos_data::{FeatureCellValidity, FeatureColumnF64};

        let rows = 2_053;
        let columns = (0..19)
            .map(|column| {
                let values = (0..rows)
                    .map(|row| {
                        let sign = if row % 2 == 0 { -1.0 } else { 1.0 };
                        sign * match column % 5 {
                            0 => 0.0,
                            1 => f64::MIN_POSITIVE * (row % 7 + 1) as f64,
                            2 => f64::from_bits(1.0_f64.to_bits() + (row % 31) as u64),
                            3 => f64::MAX / (row % 17 + 1) as f64,
                            _ => ((row * 37 + column * 13) % 509) as f64 / 64.0,
                        }
                    })
                    .collect();
                let validity = (0..rows)
                    .map(|row| {
                        if column == 18 || row % 17 == 0 {
                            Warmup
                        } else if row % 23 == 0 {
                            Gap
                        } else {
                            Valid
                        }
                    })
                    .collect();
                FeatureColumnF64::new(format!("mixed_δοκιμή_{column}"), values, validity)
                    .expect("mixed precision/validity column")
            })
            .collect();
        let ram = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
            neoethos_data::test_fixtures::canonical_test_timestamps(rows),
            columns,
        )
        .expect("mixed ladder fixture");
        let expected = independent_threshold_ladder(&ram);
        assert_adaptive_ladder_schedule_parity(&ram, expected);
    }

    #[test]
    fn derive_ladder_diverse_raw_is_bit_exact_across_ram_vortex_workers_and_schedules() {
        let ram = diverse_raw_ladder_frame();
        let expected = independent_threshold_ladder(&ram);
        assert!(
            expected.is_some(),
            "diverse RAW fixture must retain a Some ladder"
        );
        assert_adaptive_ladder_schedule_parity(&ram, expected);
    }

    #[test]
    fn adaptive_ladder_replaces_per_run_and_clears_to_static() {
        // Audit D06: a later run must REPLACE the ladder (not be blocked by a
        // first-write-wins OnceLock), and clearing must revert to the static
        // fallback so no symbol's ladder leaks into a later run.
        let ladder_a = [0.01_f64, 0.02, 0.03, 0.04, 0.05, 0.06];
        let ladder_b = [0.5_f64, 0.6, 0.7, 0.8, 0.9, 1.0];

        install_adaptive_threshold_ladder(ladder_a);
        assert_eq!(current_threshold_ladder(), ladder_a);

        // A second symbol's run replaces the first — NOT ignored.
        install_adaptive_threshold_ladder(ladder_b);
        assert_eq!(
            current_threshold_ladder(),
            ladder_b,
            "later run must replace the earlier ladder (D06)"
        );

        // A run with adaptive off clears back to the static fallback.
        clear_adaptive_threshold_ladder();
        assert_eq!(current_threshold_ladder(), STATIC_THRESHOLD_LADDER);
    }
}

#[cfg(test)]
mod generator_audit_tests {
    use super::*;
    use crate::genetic::smc_indicators::smc_structural_flag_count;
    use rand::{SeedableRng, rngs::StdRng};

    fn audit_gene(indices: Vec<usize>) -> Gene {
        let (sl_pips, tp_pips) = current_gene_stop_bounds().mid_pair();
        let mut gene = Gene {
            weights: vec![0.4; indices.len()],
            indices,
            long_threshold: 0.2,
            short_threshold: -0.2,
            sl_pips,
            tp_pips,
            stop_vol_mult: 0.0,
            ..Gene::default()
        };
        gene.normalize(8, 1);
        gene
    }

    #[test]
    fn crossover_inherits_one_indicator_only_from_b() {
        let a = audit_gene(vec![0]);
        let b = audit_gene(vec![1]);
        let child = crossover(&a, &b, 1, &mut StdRng::seed_from_u64(7));
        assert_eq!(child.indices, vec![1]);
        let child_reversed = crossover(&b, &a, 1, &mut StdRng::seed_from_u64(7));
        assert_eq!(child_reversed.indices, vec![0]);
    }

    #[test]
    fn crossover_fixed_midpoint_is_parent_order_sensitive() {
        let a = audit_gene(vec![0, 1]);
        let b = audit_gene(vec![2, 3]);
        let child = crossover(&a, &b, 1, &mut StdRng::seed_from_u64(11));
        assert_eq!(child.indices, vec![0, 3]);
        let swapped = crossover(&b, &a, 1, &mut StdRng::seed_from_u64(11));
        assert_eq!(swapped.indices, vec![2, 1]);
    }

    #[test]
    fn mutation_in_improvement_mode_keeps_arity_and_measures_locus_changes() {
        let cfg = SmcSearchConfig::default();
        let parent = audit_gene(vec![0, 1, 2]);
        let mut rng = StdRng::seed_from_u64(1234);
        let mut changed = 0usize;
        let mut neutral = 0usize;
        let mut reduced = 0usize;
        let mut unchanged_arity = 0usize;
        let mut neutral_indices_other_traits_changed = 0usize;
        let mut whole_genome_neutral = 0usize;
        let samples = 2_000usize;
        for _ in 0..samples {
            let child = mutate(&parent, 8, 4, 1, &cfg, 0, &mut rng);
            assert!(
                child.indices.len() <= parent.indices.len(),
                "improvement mode never grows arity"
            );
            if child.indices.len() < parent.indices.len() {
                reduced += 1;
            } else {
                unchanged_arity += 1;
            }
            if child.indices != parent.indices {
                changed += 1;
            } else {
                neutral += 1;
            }
            if child.indices == parent.indices {
                let mut left = parent.clone();
                let mut right = child.clone();
                right.strategy_id = left.strategy_id.clone();
                right.generation = left.generation;
                reset_gene_metrics(&mut left);
                reset_gene_metrics(&mut right);
                let same_other = left == right;
                if same_other {
                    whole_genome_neutral += 1;
                } else {
                    neutral_indices_other_traits_changed += 1;
                }
            }
        }
        eprintln!(
            "improvement-mode mutation: inherited={neutral} changed={changed} reduced_arity={reduced} same_arity={unchanged_arity} neutral_indices_other_traits_changed={neutral_indices_other_traits_changed} whole_genome_neutral={whole_genome_neutral} samples={samples}"
        );
        assert!(changed > 0);
        assert!(neutral > 0);
        assert_eq!(changed + neutral, samples);
        assert_eq!(reduced + unchanged_arity, samples);
        assert!(neutral_indices_other_traits_changed > 0);
        assert!(whole_genome_neutral < neutral);
    }

    #[test]
    fn random_newborn_always_satisfies_configured_min_structural_flags() {
        let cfg = SmcSearchConfig::default();
        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..500 {
            let gene = new_random_gene(6, 4, 0, &cfg, &mut rng);
            assert!(
                smc_structural_flag_count(&gene) >= cfg.min_flags,
                "new_random_gene enforces min_flags on every newborn"
            );
        }
    }

    fn operator_traits_equal(left: &Gene, right: &Gene) -> bool {
        if left.indices != right.indices
            || left.weights != right.weights
            || left.long_threshold != right.long_threshold
            || left.short_threshold != right.short_threshold
            || left.sl_pips != right.sl_pips
            || left.tp_pips != right.tp_pips
            || left.stop_vol_mult != right.stop_vol_mult
        {
            return false;
        }
        let flags = |gene: &Gene| {
            [
                gene.use_ob,
                gene.use_fvg,
                gene.use_liq_sweep,
                gene.use_premium_discount,
                gene.use_inducement,
                gene.mtf_confirmation,
                gene.use_bos,
                gene.use_choch,
                gene.use_eqh,
                gene.use_eql,
                gene.use_displacement,
            ]
        };
        flags(left) == flags(right)
    }

    #[test]
    fn duplicate_retry_forced_duplicate_sample_has_no_unchanged_genomes() {
        let cfg = SmcSearchConfig::default();
        let parent = audit_gene(vec![0]);
        let mut seen = SeenSignatureMemory::current();
        assert!(seen.insert_gene(&parent));
        let mut rng = StdRng::seed_from_u64(77);
        let mut unchanged_genome = 0usize;
        let samples = 200usize;
        for _ in 0..samples {
            let replacement =
                unique_candidate_or_retry(parent.clone(), &mut seen, 8, 4, 1, 20, &cfg, &mut rng);
            if operator_traits_equal(&replacement, &parent) {
                unchanged_genome += 1;
            }
        }
        eprintln!(
            "duplicate retry forced-duplicate sample: unchanged_genome={unchanged_genome} samples={samples}"
        );
        assert_eq!(
            unchanged_genome, 0,
            "forced duplicate retry sample produced no unchanged genomes; source branch draws wholesale newborns, this counter alone cannot distinguish a local mutation"
        );
    }

    fn cancellation_parent_genes() -> (Gene, Gene) {
        let mut a = Gene {
            indices: vec![140, 202],
            weights: vec![0.2, -0.4],
            long_threshold: 0.2,
            short_threshold: -0.2,
            sl_pips: 20.0,
            tp_pips: 40.0,
            ..Gene::default()
        };
        a.normalize(323, 1);
        let mut b = Gene {
            indices: vec![97, 140],
            weights: vec![0.4, -0.2],
            long_threshold: 0.2,
            short_threshold: -0.2,
            sl_pips: 20.0,
            tp_pips: 40.0,
            ..Gene::default()
        };
        b.normalize(323, 1);
        (a, b)
    }

    #[test]
    fn finite_total_cancellation_is_replaced_not_injected_as_plus_one() {
        use rand::{SeedableRng, rngs::StdRng};
        let (a, b) = cancellation_parent_genes();
        let child = crossover(&a, &b, 1, &mut StdRng::seed_from_u64(1));
        assert_eq!(child.indices, vec![140, 140]);
        assert_eq!(child.weights, vec![0.2, -0.2]);
        assert!(child.has_total_weight_cancellation(323));

        // Reproduce mutate's exact first RNG draw type/shape. This seed selects
        // a branch that rewrites no index or weight, so the pre-repair final
        // normalize would have converted the canceled term to [140] weight +1.0.
        let mut branch_rng = StdRng::seed_from_u64(0);
        let branch: i32 = branch_rng.random_range(0..4);
        assert!(
            branch == 1 || branch == 3,
            "expected a threshold/SMC branch with no index/weight rewrite, got {branch}"
        );
        let mut old_style = child.clone();
        old_style.normalize(323, 1);
        assert_eq!(old_style.indices, vec![140]);
        assert_eq!(old_style.weights, vec![1.0]);
        let cfg = SmcSearchConfig::default();
        let mut rng = StdRng::seed_from_u64(0);
        let replacement = mutate(&child, 323, 5, 1, &cfg, 1, &mut rng);
        assert!(!replacement.has_total_weight_cancellation(323));
        assert!(replacement.weights.iter().all(|weight| *weight != 0.0));
        assert_ne!(replacement.indices, vec![140]);
        assert_ne!(replacement.weights, vec![1.0]);

        let mut second_rng = StdRng::seed_from_u64(0);
        let repeated = mutate(&child, 323, 5, 1, &cfg, 1, &mut second_rng);
        assert_eq!(replacement.indices, repeated.indices);
        assert_eq!(replacement.weights, repeated.weights);
        assert_eq!(replacement.long_threshold, repeated.long_threshold);
        assert_eq!(replacement.short_threshold, repeated.short_threshold);
        assert_eq!(replacement.sl_pips, repeated.sl_pips);
        assert_eq!(replacement.tp_pips, repeated.tp_pips);

        // Preserve signed noncanceling and partial-cancellation offspring.
        let mut partial = Gene {
            indices: vec![140, 140, 231],
            weights: vec![0.2, -0.2, -0.8],
            ..Gene::default()
        };
        partial.normalize(323, 1);
        assert_eq!(partial.indices, vec![231]);
        assert_eq!(partial.weights, vec![-0.8]);
        let mut signed = Gene {
            indices: vec![140],
            weights: vec![-1.0],
            ..Gene::default()
        };
        signed.normalize(323, 1);
        assert_eq!(signed.indices, vec![140]);
        assert_eq!(signed.weights, vec![-1.0]);

        // Malformed-input repair is intentionally unchanged.
        let mut malformed = Gene {
            indices: vec![140],
            weights: vec![f64::NAN],
            ..Gene::default()
        };
        malformed.normalize(323, 1);
        assert_eq!(malformed.weights, vec![1.0]);
    }

    #[test]
    fn softmax_never_promotes_invalid_genome_above_finite_candidates() {
        let weights = softmax_weights(&[2.0, f64::NEG_INFINITY, 1.0], &[0, 1, 2], 1.0);
        let total: f64 = weights.iter().sum();
        assert!((weights[0] - 1.0).abs() < 1e-12);
        assert_eq!(weights[1], 0.0, "invalid candidate weight is zero");
        assert!((weights[2] - (-1.0f64).exp()).abs() < 1e-12);
        assert!(weights[1] / total < 1e-12);
        assert!((weights[0] / total - 1.0 / (1.0 + (-1.0f64).exp())).abs() < 1e-12);
        assert_eq!(
            softmax_weights(&[f64::NEG_INFINITY, f64::NEG_INFINITY], &[0, 1], 1.0),
            vec![1.0, 1.0],
            "all-invalid remains an explicit uniform fallback"
        );
    }

    /// The opt-in MonthlyUpsideV1 selector must route fitness through the
    /// evaluator's completed-month return vector (neo-671 B9, docs §14.7), and
    /// the default selector must keep the canonical dispatch untouched.
    #[test]
    fn monthly_upside_selector_routes_fitness_and_default_dispatch_is_unchanged() {
        use crate::scoring::FitnessTable;

        let metrics = vec![
            [10.0, 1.5, 10_100.0, 0.02, 0.60, 1.8, 0.5, 0.5, 100.0, 0.7, 0.01],
            [20.0, 1.2, 20_100.0, 0.05, 0.78, 2.1, 0.4, 0.2, 50.0, 0.6, 0.02],
        ];
        let months = vec![vec![0.02; 10], vec![0.04, 0.04, 0.04, 0.04, -0.13]];

        // Default selector: the canonical growth-goal / growth / ga_fitness chain.
        let mut genes = vec![Gene::default(), Gene::default()];
        let default_cfg = crate::genetic::EvaluationConfig::default();
        apply_metrics(&mut genes, &metrics, &default_cfg, 1.0, None);
        assert_eq!(genes[0].fitness, crate::scoring::ga_fitness(&metrics[0]));
        assert_eq!(genes[1].fitness, crate::scoring::ga_fitness(&metrics[1]));

        // The selector routes to ga_fitness_monthly_upside_v1 with the vector.
        let mut selected_cfg = crate::genetic::EvaluationConfig::default();
        selected_cfg.fitness_table = FitnessTable::MonthlyUpsideV1;
        apply_metrics(&mut genes, &metrics, &selected_cfg, 1.0, Some(&months));
        assert_eq!(
            genes[0].fitness,
            crate::scoring::ga_fitness_monthly_upside_v1(&months[0], &metrics[0])
        );
        assert_eq!(
            genes[1].fitness,
            crate::scoring::ga_fitness_monthly_upside_v1(&months[1], &metrics[1])
        );
        assert_ne!(
            genes[0].fitness,
            crate::scoring::ga_fitness(&metrics[0]),
            "the MonthlyUpsideV1 selector must not fall through to the canonical scorer"
        );

        // The dd-averse selector routes to the SAME monthly vector through its
        // own scorer: identical when both candidates are under the drawdown
        // thresholds, penalized when one is not.
        let mut selected_cfg = crate::genetic::EvaluationConfig::default();
        selected_cfg.fitness_table = FitnessTable::MonthlyUpsideDdAverseV1;
        let clean = metrics.clone();
        apply_metrics(&mut genes, &clean, &selected_cfg, 1.0, Some(&months));
        assert_eq!(
            genes[0].fitness,
            crate::scoring::ga_fitness_monthly_upside_dd_averse_v1(&months[0], &clean[0])
        );
        let mut risky = clean.clone();
        risky[0][3] = 0.30; // 30 % drawdown, far above the 10 % free band
        apply_metrics(&mut genes, &risky, &selected_cfg, 1.0, Some(&months));
        assert!(
            genes[0].fitness < crate::scoring::ga_fitness_monthly_upside_v1(&months[0], &risky[0]),
            "the dd-averse selector must penalize drawdown relative to B9"
        );

        // The dd-budget selector multiplies the profit term by the drawdown
        // budget factor, so the same 30 % drawdown candidate turns negative.
        let mut budget_cfg = crate::genetic::EvaluationConfig::default();
        budget_cfg.fitness_table = FitnessTable::MonthlyUpsideDdBudgetV1;
        apply_metrics(&mut genes, &risky, &budget_cfg, 1.0, Some(&months));
        assert_eq!(
            genes[0].fitness,
            crate::scoring::ga_fitness_monthly_upside_dd_budget_v1(&months[0], &risky[0])
        );
        assert!(
            genes[0].fitness < 0.0,
            "a 30 % drawdown candidate must score negative under the 15 % budget"
        );
    }

}
