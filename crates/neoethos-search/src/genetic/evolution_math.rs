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
/// `FOREX_BOT_PROP_SEEN_*` env vars. The seen-signature memory consults
/// the cached overrides each time it is constructed, but the env vars
/// themselves are read at most once per process.
#[derive(Debug, Clone, PartialEq, Eq)]
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

impl SeenSignatureMemoryRuntimeOverrides {
    /// One-shot read of the legacy `FOREX_BOT_PROP_SEEN_*` env vars.
    pub fn from_env() -> Self {
        let flush_every = std::env::var("FOREX_BOT_PROP_SEEN_FLUSH_EVERY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4096)
            .max(1);
        let load_max = std::env::var("FOREX_BOT_PROP_SEEN_LOAD_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(3_000_000);
        let max_entries_raw = std::env::var("FOREX_BOT_PROP_SEEN_MAX_ENTRIES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(load_max);
        let max_entries = if max_entries_raw == 0 {
            usize::MAX
        } else {
            max_entries_raw.max(1)
        };
        let file_path = std::env::var("FOREX_BOT_PROP_SEEN_FILE")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        Self {
            flush_every,
            load_max,
            max_entries,
            file_path,
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

/// Convenience wrapper that resolves the legacy `FOREX_BOT_PROP_SEEN_*`
/// env vars once and installs them. Idempotent.
pub fn install_seen_signature_memory_runtime_overrides_from_env() {
    let _ = SEEN_SIGNATURE_MEMORY_RUNTIME_OVERRIDES
        .set(SeenSignatureMemoryRuntimeOverrides::from_env());
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
    pub fn from_env() -> Self {
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
    let levels = [0.2, 0.4, 0.6, 0.8, 1.0];
    levels[rng.random_range(0..levels.len())]
}

fn random_coarse_threshold(rng: &mut impl Rng) -> f32 {
    // **2026-05-26 operator directive (dual-mode product, taskdoc #273)**:
    // the previous `[0.30, 0.45, 0.60, 0.80, 1.00, 1.20]` ladder was
    // calibrated against an older feature pipeline; empirical scoring on
    // the current z-score frames shows the *combined* signal magnitude
    // for a multi-indicator gene sits around 0.05-0.15 (≈ 0.5σ), with a
    // 95th-percentile near 0.30-0.50. The old ladder's *minimum*
    // threshold of 0.30 was at-or-above the 95th percentile, meaning
    // most random initial genes never fired — even one indicator with a
    // strong signal couldn't push the weighted combo past 0.30.
    //
    // The narrower ladder `[0.10, 0.20, 0.35, 0.50, 0.70, 0.90]` keeps the
    // top end (for picky genes that fire only on extreme moves) but
    // adds a working low-end (genes that fire often, even on mild
    // moves). Activity + downstream filters (min_trades, prop-firm
    // window-pass) prevent over-trading from winning.
    //
    // Doctrine: F-058 (2026-05-25) hard-picked the normalised ladder; this
    // change narrows that same ladder to match the empirical feature
    // magnitudes, not the assumption of unit-variance noise.
    //
    // **F-058 history (kept for context)**: the dual-convention switch via
    // `FOREX_BOT_NORMALIZE_FEATURES` was a leftover from audit-era
    // debugging when the feature pipeline emitted both raw-indicator
    // and z-score frames. The current feature pipeline (post-#212
    // vector_ta normalisation) ALWAYS emits z-score-normalised
    // features — so the "raw" threshold ladder is unreachable in
    // practice. If the operator ever switches back to a raw-feature
    // pipeline, re-introduce the env switch — but DO NOT default-on it
    // again because the dual-convention was a silent runtime trap.
    let levels = [0.10, 0.20, 0.35, 0.50, 0.70, 0.90];
    levels[rng.random_range(0..levels.len())]
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
    let (sl_pips, tp_pips) = if rng.random_bool(0.2) {
        (15.0, 30.0)
    } else {
        let sl: f64 = rng.random_range(5.0..=50.0);
        let rr: f64 = rng.random_range(1.5..=3.0);
        let tp = (sl * rr).clamp(10.0, 100.0);
        (sl, tp)
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
                // In exploitation mode, prefer tweaking weights over full indicator replacement
                if !mutated.indices.is_empty() && (intensity < 1.0 || rng.random_bool(0.5)) {
                    let idx = rng.random_range(0..mutated.indices.len());
                    if rng.random_bool(0.3 * intensity as f64) {
                        mutated.indices[idx] = rng.random_range(0..n_indicators.max(1));
                    }
                    mutated.weights[idx] = random_coarse_weight(rng);
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
                mutated.tp_pips = (mutated.tp_pips
                    * rng.random_range((1.0 - range)..(1.0 + range)))
                .clamp(10.0, 100.0);
                mutated.sl_pips = (mutated.sl_pips
                    * rng.random_range((1.0 - range)..(1.0 + range)))
                .clamp(5.0, 50.0);
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

pub fn apply_metrics(genes: &mut [Gene], metrics: &[[f64; 11]]) {
    for (gene, m) in genes.iter_mut().zip(metrics.iter()) {
        // Scoring Phase B (2026-05-25): call the canonical
        // `crate::scoring::ga_fitness` directly rather than the
        // local `#[deprecated]` `score_from_metrics` shim. Behaviour
        // is byte-for-byte identical (pinned by Phase-A test).
        gene.fitness = crate::scoring::ga_fitness(m);
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
    fn current_seen_signature_memory_runtime_overrides_returns_legal_values() {
        let observed = current_seen_signature_memory_runtime_overrides();
        assert!(observed.flush_every >= 1);
        assert!(observed.max_entries >= 1);
    }
}
