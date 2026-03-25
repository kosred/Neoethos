use super::strategy_gene::Gene;
use super::smc_indicators::{SmcSearchConfig, randomize_smc_flags, enforce_min_structural_smc_flags};
use rand::Rng;
use rand::seq::index::sample;
use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;
use std::fs::{self, OpenOptions};
use std::io::Write;

const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;

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

pub fn gene_signature_hash(gene: &Gene) -> u64 {
    let mut h = FNV_OFFSET_BASIS;
    h = fnv1a_update(h, &(gene.indices.len() as u64).to_le_bytes());
    for idx in &gene.indices { h = fnv1a_update(h, &(*idx as u64).to_le_bytes()); }
    h = fnv1a_update(h, &(gene.weights.len() as u64).to_le_bytes());
    for w in &gene.weights { h = fnv1a_update(h, &quantize_f32(*w, 10_000.0).to_le_bytes()); }
    h = fnv1a_update(h, &quantize_f32(gene.long_threshold, 1_000_000.0).to_le_bytes());
    h = fnv1a_update(h, &quantize_f32(gene.short_threshold, 1_000_000.0).to_le_bytes());
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
        let flush_every = std::env::var("FOREX_BOT_PROP_SEEN_FLUSH_EVERY").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(4096).max(1);
        let load_max = std::env::var("FOREX_BOT_PROP_SEEN_LOAD_MAX").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(3_000_000);
        let max_entries = std::env::var("FOREX_BOT_PROP_SEEN_MAX_ENTRIES").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(load_max);
        let max_entries = if max_entries == 0 { usize::MAX } else { max_entries.max(1) };
        let file_raw = std::env::var("FOREX_BOT_PROP_SEEN_FILE").ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        let file_path = file_raw.map(PathBuf::from);

        let mut memory = Self { all: HashSet::new(), order: VecDeque::new(), pending: Vec::new(), file_path, flush_every, max_entries };
        if let Some(path) = memory.file_path.clone() {
            if let Some(parent) = path.parent() { let _ = fs::create_dir_all(parent); }
            if let Ok(buf) = fs::read(&path) {
                if buf.len() >= 8 && buf.len() % 8 == 0 {
                    for chunk in buf.chunks_exact(8) {
                        if load_max > 0 && memory.all.len() >= load_max { break; }
                        let mut arr = [0_u8; 8];
                        arr.copy_from_slice(chunk);
                        memory.insert_in_memory(u64::from_le_bytes(arr));
                    }
                } else if let Ok(text) = String::from_utf8(buf) {
                    for line in text.lines() {
                        if load_max > 0 && memory.all.len() >= load_max { break; }
                        let line = line.trim();
                        if line.is_empty() { continue; }
                        if let Ok(v) = u64::from_str_radix(line.trim_start_matches("0x"), 16) { memory.insert_in_memory(v); }
                        else if let Ok(v) = line.parse::<u64>() { memory.insert_in_memory(v); }
                    }
                }
            }
        }
        memory
    }

    pub fn insert_in_memory(&mut self, sig: u64) -> bool {
        if !self.all.insert(sig) { return false; }
        self.order.push_back(sig);
        if self.max_entries != usize::MAX {
            while self.all.len() > self.max_entries {
                if let Some(old) = self.order.pop_front() { self.all.remove(&old); } else { break; }
            }
        }
        true
    }

    pub fn insert_hash(&mut self, sig: u64) -> bool {
        if !self.insert_in_memory(sig) { return false; }
        if self.file_path.is_some() {
            self.pending.push(sig);
            if self.pending.len() >= self.flush_every { self.flush(); }
        }
        true
    }

    pub fn insert_gene(&mut self, gene: &Gene) -> bool {
        self.insert_hash(gene_signature_hash(gene))
    }

    pub fn flush(&mut self) {
        if self.pending.is_empty() { return; }
        let path = match &self.file_path { Some(p) => p.clone(), None => { self.pending.clear(); return; } };
        if let Some(parent) = path.parent() { let _ = fs::create_dir_all(parent); }
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) {
            let mut bytes = Vec::with_capacity(self.pending.len() * 8);
            for v in &self.pending { bytes.extend_from_slice(&v.to_le_bytes()); }
            if file.write_all(&bytes).is_ok() { let _ = file.flush(); self.pending.clear(); }
        }
    }
}

pub fn new_random_gene(n_indicators: usize, max_indicators: usize, generation: usize, smc_cfg: &SmcSearchConfig) -> Gene {
    let mut rng = rand::rng();
    let min_indicators = 1;
    let max_indicators = max_indicators.clamp(min_indicators, n_indicators.max(1));
    let count = rng.random_range(min_indicators..=max_indicators);
    let sample = sample(&mut rng, n_indicators.max(1), count);
    let indices: Vec<usize> = sample.iter().collect();
    let weights: Vec<f32> = (0..count).map(|_| rng.random_range(0.1..1.0)).collect();
    let long_threshold = rng.random_range(0.15..0.55);
    let short_threshold = -rng.random_range(0.15..0.55);
    let (sl_pips, tp_pips) = if rng.random_bool(0.2) { (15.0, 30.0) } else {
        let sl: f64 = rng.random_range(5.0..=50.0);
        let rr: f64 = rng.random_range(1.5..=3.0);
        let tp = (sl * rr).clamp(10.0, 100.0);
        (sl, tp)
    };
    let strategy_id = format!("gene_{}_{}", rng.random_range(0..1_000_000u64), generation);
    let mut gene = Gene {
        indices, weights, long_threshold, short_threshold,
        fitness: 0.0, sharpe_ratio: 0.0, win_rate: 0.0, max_drawdown: 0.0, profit_factor: 0.0,
        expectancy: 0.0, trades_count: 0, generation, strategy_id,
        use_ob: false, use_fvg: false, use_liq_sweep: false, mtf_confirmation: true,
        use_premium_discount: false, use_inducement: false, use_bos: false, use_choch: false,
        use_eqh: false, use_eql: false, use_displacement: false,
        tp_pips, sl_pips, slice_pass_rate: 0.0, consistency: 0.0,
    };
    randomize_smc_flags(&mut gene, smc_cfg, &mut rng);
    enforce_min_structural_smc_flags(&mut gene, smc_cfg, &mut rng);
    gene
}

pub fn generate_random_genes(n_genes: usize, n_indicators: usize, max_indicators: usize, generation: usize, smc_cfg: &SmcSearchConfig) -> Vec<Gene> {
    (0..n_genes).map(|_| new_random_gene(n_indicators, max_indicators, generation, smc_cfg)).collect()
}

pub fn unique_candidate_or_retry(mut candidate: Gene, seen: &mut SeenSignatureMemory, n_indicators: usize, max_indicators: usize, generation: usize, max_attempts: usize, smc_cfg: &SmcSearchConfig) -> Gene {
    candidate.normalize(n_indicators, 1);
    if seen.insert_gene(&candidate) { return candidate; }
    let attempts = max_attempts.max(1);
    for _ in 0..attempts {
        let mut probe = new_random_gene(n_indicators, max_indicators, generation, smc_cfg);
        probe.normalize(n_indicators, 1);
        if seen.insert_gene(&probe) { return probe; }
    }
    candidate
}

pub fn crossover(a: &Gene, b: &Gene, generation: usize) -> Gene {
    let mut rng = rand::rng();
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
    child.fitness = 0.0;
    
    child.long_threshold = if rng.random_bool(0.5) { a.long_threshold } else { b.long_threshold };
    child.short_threshold = if rng.random_bool(0.5) { a.short_threshold } else { b.short_threshold };
    child.use_ob = if rng.random_bool(0.5) { a.use_ob } else { b.use_ob };
    child.use_fvg = if rng.random_bool(0.5) { a.use_fvg } else { b.use_fvg };
    child.use_liq_sweep = if rng.random_bool(0.5) { a.use_liq_sweep } else { b.use_liq_sweep };
    child.mtf_confirmation = if rng.random_bool(0.5) { a.mtf_confirmation } else { b.mtf_confirmation };
    child.use_premium_discount = if rng.random_bool(0.5) { a.use_premium_discount } else { b.use_premium_discount };
    child.use_inducement = if rng.random_bool(0.5) { a.use_inducement } else { b.use_inducement };
    child.use_bos = if rng.random_bool(0.5) { a.use_bos } else { b.use_bos };
    child.use_choch = if rng.random_bool(0.5) { a.use_choch } else { b.use_choch };
    child.use_eqh = if rng.random_bool(0.5) { a.use_eqh } else { b.use_eqh };
    child.use_eql = if rng.random_bool(0.5) { a.use_eql } else { b.use_eql };
    child.use_displacement = if rng.random_bool(0.5) { a.use_displacement } else { b.use_displacement };
    child.tp_pips = if rng.random_bool(0.5) { a.tp_pips } else { b.tp_pips };
    child.sl_pips = if rng.random_bool(0.5) { a.sl_pips } else { b.sl_pips };
    
    child
}

pub fn mutate(gene: &Gene, n_indicators: usize, max_indicators: usize, generation: usize, smc_cfg: &SmcSearchConfig) -> Gene {
    let mut rng = rand::rng();
    let mut mutated = gene.clone();
    match rng.random_range(0..4) {
        0 => {
            if !mutated.indices.is_empty() && rng.random_bool(0.5) {
                let idx = rng.random_range(0..mutated.indices.len());
                mutated.indices[idx] = rng.random_range(0..n_indicators.max(1));
                mutated.weights[idx] = rng.random_range(0.1..1.0);
            } else {
                let min_indicators = 1;
                let max_indicators = max_indicators.clamp(min_indicators, n_indicators.max(1));
                let count = rng.random_range(min_indicators..=max_indicators);
                let sample = sample(&mut rng, n_indicators.max(1), count);
                mutated.indices = sample.iter().collect();
                mutated.weights = (0..count).map(|_| rng.random_range(0.1..1.0)).collect();
            }
        },
        1 => {
            mutated.long_threshold = (mutated.long_threshold * rng.random_range(0.7..1.3)).clamp(0.08, 0.8);
            mutated.short_threshold = (mutated.short_threshold * rng.random_range(0.7..1.3)).clamp(-0.8, -0.08);
        },
        2 => {
            mutated.tp_pips = (mutated.tp_pips * rng.random_range(0.8..1.2)).clamp(10.0, 100.0);
            mutated.sl_pips = (mutated.sl_pips * rng.random_range(0.8..1.2)).clamp(5.0, 50.0);
        },
        _ => randomize_smc_flags(&mut mutated, smc_cfg, &mut rng),
    }
    if rng.random_bool(0.25) { enforce_min_structural_smc_flags(&mut mutated, smc_cfg, &mut rng); }
    mutated.strategy_id = format!("gene_{}_{}", rng.random_range(0..1_000_000u64), generation);
    mutated.generation = generation;
    mutated
}

pub fn score_from_metrics(metrics: &[f64; 11]) -> f64 {
    let net_profit = metrics[0];
    let sharpe = metrics[1];
    let max_dd = metrics[3];
    let profit_factor = metrics[5];
    let consistency = metrics[9];
    let dd_cap = 0.07;
    let pfloor = 1.0;
    let dd_penalty = 10.0 * (max_dd - dd_cap).max(0.0);
    let pf_penalty = if profit_factor <= pfloor { 5.0 } else { 0.0 };
    sharpe + (net_profit / 10_000.0) + (consistency * 0.5) - dd_penalty - pf_penalty
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
