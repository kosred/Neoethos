use super::smc_indicators::{
    SmcSearchConfig, enforce_min_structural_smc_flags, randomize_smc_flags,
};
use super::strategy_gene::Gene;
use rand::Rng;
use rand::seq::index::sample;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;

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

fn fnv1a_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for b in bytes {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
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

impl SeenSignatureMemory {
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
        let max_entries = std::env::var("FOREX_BOT_PROP_SEEN_MAX_ENTRIES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(load_max);
        let max_entries = if max_entries == 0 {
            usize::MAX
        } else {
            max_entries.max(1)
        };
        let file_raw = std::env::var("FOREX_BOT_PROP_SEEN_FILE")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let file_path = file_raw.map(PathBuf::from);

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
            if file.write_all(&bytes).is_ok() {
                let _ = file.flush();
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
    let levels = [0.15, 0.25, 0.35, 0.45, 0.55];
    levels[rng.random_range(0..levels.len())]
}

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

pub fn score_from_metrics(metrics: &[f64; 11]) -> f64 {
    let sharpe = metrics[1];
    let max_dd = metrics[3];
    let win_rate = metrics[4];
    let profit_factor = metrics[5];
    let trades = metrics[8];
    let consistency = metrics[9];

    if !sharpe.is_finite() || trades < 1.0 {
        return f64::NEG_INFINITY;
    }

    // Confidence factor: more trades → higher confidence (caps at ~100 trades)
    let trades_confidence = (trades.sqrt() / 10.0).min(1.0);

    // Sharpe weighted by confidence (primary signal)
    let sharpe_component = sharpe * trades_confidence * 0.40;

    // Consistency: reward smooth monthly returns
    let consistency_component = consistency.clamp(0.0, 1.0) * 0.25;

    // Drawdown: proportional penalty, not cliff-based
    let dd_penalty = (max_dd * 15.0).min(5.0);

    // Profit factor: smooth reward above 1.0, smooth penalty below
    let pf_component = if profit_factor >= 1.0 {
        ((profit_factor - 1.0) * 0.5).min(1.5) * 0.20
    } else {
        -(1.0 / profit_factor.max(0.1)) * 0.30
    };

    // Win rate bonus (minor, since PF already captures edge quality)
    let wr_component = ((win_rate - 0.45) * 2.0).clamp(0.0, 0.5) * 0.10;

    sharpe_component + consistency_component + pf_component + wr_component - dd_penalty
}

pub fn apply_metrics(genes: &mut [Gene], metrics: &[[f64; 11]]) {
    for (gene, m) in genes.iter_mut().zip(metrics.iter()) {
        gene.fitness = score_from_metrics(m);
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
}
