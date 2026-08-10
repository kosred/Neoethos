use super::smc_indicators::{
    SmcSearchConfig, enforce_min_structural_smc_flags, randomize_smc_flags,
};
use super::strategy_gene::Gene;
use neoethos_core::utils::fnv1a64_update;
use rand::Rng;
use rand::seq::index::sample;
use serde::{Deserialize, Serialize};
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

fn quantize_f32(value: f32, scale: f32) -> i64 {
    ((value as f64) * (scale as f64)).round() as i64
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
    let max_score = candidate_indices
        .iter()
        .filter_map(|idx| scores.get(*idx))
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_score.is_finite() {
        return vec![1.0; candidate_indices.len()];
    }

    candidate_indices
        .iter()
        .map(|idx| {
            let centered = (scores[*idx] - max_score) / temperature;
            if centered.is_finite() {
                centered.exp().max(1e-12)
            } else {
                1.0
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
                available.retain(|candidate| *candidate != idx);
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
        h = fnv1a_update(h, &quantize_f32(*w, 10_000.0).to_le_bytes());
    }
    h = fnv1a_update(
        h,
        &quantize_f32(gene.long_threshold, 1_000_000.0).to_le_bytes(),
    );
    h = fnv1a_update(
        h,
        &quantize_f32(gene.short_threshold, 1_000_000.0).to_le_bytes(),
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
    /// available RAM** (see [`derived_seen_max_entries`]) rather than
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

/// Install process-wide seen-signature-memory overrides. Returns
/// `Err(existing)` when overrides were already installed earlier (the
/// first install wins).
pub fn install_seen_signature_memory_runtime_overrides(
    overrides: SeenSignatureMemoryRuntimeOverrides,
) -> Result<(), SeenSignatureMemoryRuntimeOverrides> {
    SEEN_SIGNATURE_MEMORY_RUNTIME_OVERRIDES.set(overrides)
}

/// RETIRED 2026-08-10 — installs the typed defaults and reads no environment.
/// Kept only because `genetic/mod.rs` and `lib.rs` re-export it.
pub fn install_seen_signature_memory_runtime_overrides_from_env() {
    tracing::error!(
        target: "neoethos_search::retired_env",
        "install_seen_signature_memory_runtime_overrides_from_env() is RETIRED and installs \
         typed DEFAULTS — the NEOETHOS_BOT_PROP_SEEN_* layer no longer exists. Call \
         install_seen_signature_memory_runtime_overrides_from_settings(&settings)."
    );
    let _ = SEEN_SIGNATURE_MEMORY_RUNTIME_OVERRIDES
        .set(SeenSignatureMemoryRuntimeOverrides::default());
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
    /// Deprecated spelling of [`Self::current`] — it has read no environment
    /// since the typed boundary landed, and none at all since 2026-08-10.
    /// Retained because `discovery.rs`, `genetic/search_engine.rs` and
    /// `discovery_ledger.rs` call it.
    pub fn from_env() -> Self {
        Self::current()
    }

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

fn random_coarse_weight(rng: &mut impl Rng) -> f32 {
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
const STATIC_THRESHOLD_LADDER: [f32; 6] = [0.10, 0.20, 0.35, 0.50, 0.70, 0.90];

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
static ADAPTIVE_THRESHOLD_LADDER: std::sync::RwLock<Option<[f32; 6]>> =
    std::sync::RwLock::new(None);

/// Install (REPLACE) the adaptive threshold ladder derived from the current
/// run's feature cube. Unlike the old first-write-wins semantics, each call
/// overwrites, so a later symbol's run gets its own ladder. Values should be
/// sorted ascending and finite + non-negative (the derivation clamps/sorts).
pub fn install_adaptive_threshold_ladder(ladder: [f32; 6]) {
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
pub fn current_threshold_ladder() -> [f32; 6] {
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
    let mut guard = GENE_STOP_ATR_PIPS.write().unwrap_or_else(|e| e.into_inner());
    *guard = if atr_pips.is_finite() && atr_pips > 0.0 {
        Some(atr_pips)
    } else {
        None
    };
}

/// Clear back to the absolute pip band. Called at the start of a run that does
/// NOT scale by ATR, so the PREVIOUS combo's scale cannot leak into it.
pub fn clear_gene_stop_atr_scale() {
    let mut guard = GENE_STOP_ATR_PIPS.write().unwrap_or_else(|e| e.into_inner());
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

fn random_coarse_threshold(rng: &mut impl Rng) -> f32 {
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
) -> Option<[f32; 6]> {
    let n_cols = features.n_features();
    let n_rows = features.n_samples();
    if n_cols == 0 || n_rows == 0 {
        return None;
    }

    // Per-column median |value|. `feature_column` yields one contiguous series
    // per call — for the mmap backing this is a single feature-major row read,
    // so the scan is sequential rather than strided across the whole matrix.
    let mut per_col_median_abs: Vec<f32> = Vec::with_capacity(n_cols);
    for c in 0..n_cols {
        let mut abs_vals: Vec<f32> = features
            .feature_column(c)
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .map(|v| v.abs())
            .collect();
        if abs_vals.is_empty() {
            continue;
        }
        abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = abs_vals[abs_vals.len() / 2];
        if median.is_finite() && median >= 0.0 {
            per_col_median_abs.push(median);
        }
    }

    if per_col_median_abs.is_empty() {
        return None;
    }

    // Pool: percentile points on the per-column-median sample.
    per_col_median_abs
        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pct = |p: f64| -> f32 {
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
    // Final safety: ensure strictly non-decreasing (clamp+pct may
    // collapse two adjacent points in degenerate inputs).
    for i in 1..ladder.len() {
        if ladder[i] < ladder[i - 1] {
            // Degenerate distribution — bail to static fallback.
            return None;
        }
    }
    Some(ladder)
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
    let weights: Vec<f32> = (0..count).map(|_| random_coarse_weight(rng)).collect();
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
    // Adaptive stops (Stage 2c): when enabled, seed a searchable volatility
    // multiplier so the stop scales with volatility at entry (the fixed sl/tp
    // above are then ignored in favour of `mult × vol-distance`). Off → 0.0 =
    // fixed-stop, byte-identical to before.
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
    if seen.insert_gene(&candidate) {
        return candidate;
    }
    let attempts = max_attempts.max(1);
    for _ in 0..attempts {
        let mut probe = new_random_gene(n_indicators, max_indicators, generation, smc_cfg, rng);
        probe.normalize(n_indicators, 1);
        if seen.insert_gene(&probe) {
            return probe;
        }
    }
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
    child.tp_pips = if rng.random_bool(0.5) {
        a.tp_pips
    } else {
        b.tp_pips
    };
    child.sl_pips = if rng.random_bool(0.5) {
        a.sl_pips
    } else {
        b.sl_pips
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
        (3, 1.5_f32) // Heavy exploration
    } else if stagnant_generations > 5 {
        (2, 1.2_f32) // Moderate exploration
    } else if stagnant_generations == 0 {
        (1, 0.5_f32) // Exploitation (improvement streak)
    } else {
        (1, 1.0_f32) // Normal
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
                    // Adaptive-stop gene: perturb the volatility multiplier (the
                    // active stop knob) instead of the unused fixed pips. Clamped
                    // to a sane band so the stop stays a small multiple of the
                    // bar's vol distance.
                    mutated.stop_vol_mult = (mutated.stop_vol_mult
                        * rng.random_range((1.0 - range)..(1.0 + range)))
                    .clamp(0.3, 4.0);
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
    mutated.normalize(n_indicators, 1);
    mutated
}

// **Scoring Phase B COMPLETE (2026-05-25 verbose-build pass)**: the
// previous `score_from_metrics` deprecated shim was deleted. The only
// in-crate caller (`apply_metrics` below) calls
// `crate::scoring::ga_fitness` directly. External callers reaching for
// the old name now get a compile error pointing at the canonical
// function — which is what we want.

pub fn apply_metrics(genes: &mut [Gene], metrics: &[[f64; 11]], growth_objective: bool) {
    for (gene, m) in genes.iter_mut().zip(metrics.iter()) {
        // Scoring Phase B (2026-05-25): call the canonical
        // `crate::scoring::ga_fitness` directly rather than the
        // local `#[deprecated]` `score_from_metrics` shim.
        // scoring_version 5 (2026-07-02): Risky discovery evolves under the
        // Kelly log-growth objective; everything else keeps the v4 formula.
        gene.fitness = if growth_objective {
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
        assert_eq!(absolute.sl_max_pips, 20.0, "the pre-2026-08-09 stop ceiling");
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
        for _ in 0..200 {
            let gene = new_random_gene(8, 4, 0, &smc, &mut rng);
            assert!(
                gene.sl_pips >= band.sl_min_pips - 1e-9
                    && gene.sl_pips <= band.sl_max_pips + 1e-9,
                "initialised sl {} outside [{}, {}]",
                gene.sl_pips,
                band.sl_min_pips,
                band.sl_max_pips
            );
            let mutated = mutate(&gene, 8, 4, 1, &smc, 0, &mut rng);
            assert!(
                mutated.sl_pips >= band.sl_min_pips - 1e-9
                    && mutated.sl_pips <= band.sl_max_pips + 1e-9,
                "mutated sl {} outside [{}, {}]",
                mutated.sl_pips,
                band.sl_min_pips,
                band.sl_max_pips
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

    // The ladder fn moved from a raw `Array2` to the out-of-core-aware
    // `FeatureFrame`; wrap each test cube (timestamps/names are irrelevant to
    // the per-column magnitude math).
    fn frame_of(a: ndarray::Array2<f32>) -> neoethos_data::FeatureFrame {
        neoethos_data::FeatureFrame::from_array(Vec::new(), Vec::new(), a)
    }

    #[test]
    fn derive_ladder_returns_none_for_empty_cube() {
        let empty: ndarray::Array2<f32> = ndarray::Array2::zeros((0, 0));
        assert!(derive_adaptive_threshold_ladder_from_features(&frame_of(empty)).is_none());

        let no_rows: ndarray::Array2<f32> = ndarray::Array2::zeros((0, 5));
        assert!(derive_adaptive_threshold_ladder_from_features(&frame_of(no_rows)).is_none());

        let no_cols: ndarray::Array2<f32> = ndarray::Array2::zeros((100, 0));
        assert!(derive_adaptive_threshold_ladder_from_features(&frame_of(no_cols)).is_none());
    }

    #[test]
    fn derive_ladder_returns_none_for_all_nan_cube() {
        let mut data: ndarray::Array2<f32> = ndarray::Array2::zeros((10, 3));
        data.fill(f32::NAN);
        assert!(derive_adaptive_threshold_ladder_from_features(&frame_of(data)).is_none());
    }

    #[test]
    fn derive_ladder_produces_monotonic_ascending_values() {
        // Sample with varied magnitudes per column.
        let mut data: ndarray::Array2<f32> = ndarray::Array2::zeros((100, 4));
        for r in 0..100 {
            data[(r, 0)] = (r as f32) * 0.01; // 0..1.0 — small magnitudes
            data[(r, 1)] = (r as f32) * 0.05; // 0..5.0 — medium
            data[(r, 2)] = (r as f32) * 0.10; // 0..10.0 — large (will clamp)
            data[(r, 3)] = -(r as f32) * 0.02; // negative side
        }
        let ladder = derive_adaptive_threshold_ladder_from_features(&frame_of(data))
            .expect("non-degenerate cube must produce ladder");
        // Monotone ascending
        for i in 1..ladder.len() {
            assert!(
                ladder[i] >= ladder[i - 1],
                "ladder[{i}]={} < ladder[{}]={}",
                ladder[i],
                i - 1,
                ladder[i - 1]
            );
        }
        // All clamped to safe range
        for v in ladder.iter() {
            assert!(v.is_finite());
            assert!(*v >= 1e-4);
            assert!(*v <= 10.0);
        }
    }

    #[test]
    fn adaptive_ladder_replaces_per_run_and_clears_to_static() {
        // Audit D06: a later run must REPLACE the ladder (not be blocked by a
        // first-write-wins OnceLock), and clearing must revert to the static
        // fallback so no symbol's ladder leaks into a later run.
        let ladder_a = [0.01_f32, 0.02, 0.03, 0.04, 0.05, 0.06];
        let ladder_b = [0.5_f32, 0.6, 0.7, 0.8, 0.9, 1.0];

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
