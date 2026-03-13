use crate::stop_target::{infer_stop_target_pips, StopTargetSettings};
use anyhow::{bail, Result};
use chrono::{Datelike, TimeZone, Utc};
use forex_data::{FeatureFrame, Ohlcv};
use ndarray::Array2;
use rand::seq::index::sample;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

fn clamp01(v: f64) -> f64 {
    v.max(0.0).min(1.0)
}

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

fn gene_signature_hash(gene: &Gene) -> u64 {
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
struct SeenSignatureMemory {
    all: HashSet<u64>,
    order: VecDeque<u64>,
    pending: Vec<u64>,
    file_path: Option<PathBuf>,
    flush_every: usize,
    max_entries: usize,
}

impl SeenSignatureMemory {
    fn from_env() -> Self {
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

    fn insert_in_memory(&mut self, sig: u64) -> bool {
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

    fn insert_hash(&mut self, sig: u64) -> bool {
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

    fn insert_gene(&mut self, gene: &Gene) -> bool {
        self.insert_hash(gene_signature_hash(gene))
    }

    fn flush(&mut self) {
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

fn unique_candidate_or_retry(
    mut candidate: Gene,
    seen: &mut SeenSignatureMemory,
    n_indicators: usize,
    max_indicators: usize,
    generation: usize,
    max_attempts: usize,
    smc_cfg: &SmcSearchConfig,
) -> Gene {
    candidate.normalize(n_indicators, 1);
    if seen.insert_gene(&candidate) {
        return candidate;
    }
    let attempts = max_attempts.max(1);
    for _ in 0..attempts {
        let mut probe = new_random_gene(n_indicators, max_indicators, generation, smc_cfg);
        probe.normalize(n_indicators, 1);
        if seen.insert_gene(&probe) {
            return probe;
        }
    }
    candidate
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gene {
    pub indices: Vec<usize>,
    pub weights: Vec<f32>,
    pub long_threshold: f32,
    pub short_threshold: f32,
    pub fitness: f64,
    pub sharpe_ratio: f64,
    pub win_rate: f64,
    pub max_drawdown: f64,
    pub profit_factor: f64,
    pub expectancy: f64,
    pub trades_count: usize,
    pub generation: usize,
    pub strategy_id: String,
    pub use_ob: bool,
    pub use_fvg: bool,
    pub use_liq_sweep: bool,
    pub mtf_confirmation: bool,
    pub use_premium_discount: bool,
    pub use_inducement: bool,
    #[serde(default)]
    pub use_bos: bool,
    #[serde(default)]
    pub use_choch: bool,
    #[serde(default)]
    pub use_eqh: bool,
    #[serde(default)]
    pub use_eql: bool,
    #[serde(default)]
    pub use_displacement: bool,
    pub tp_pips: f64,
    pub sl_pips: f64,
    pub slice_pass_rate: f64,
}

impl Gene {
    fn normalize(&mut self, n_indicators: usize, min_indicators: usize) {
        if self.indices.is_empty() {
            self.indices.push(0);
        }
        if self.weights.len() != self.indices.len() {
            self.weights = vec![1.0; self.indices.len()];
        }
        let n_indicators = n_indicators.max(1);
        for idx in &mut self.indices {
            if *idx >= n_indicators {
                *idx %= n_indicators;
            }
        }
        let min_indicators = min_indicators.min(n_indicators.max(1));
        if self.indices.len() < min_indicators {
            let mut rng = rand::rng();
            let mut seen = std::collections::HashSet::new();
            for idx in &self.indices {
                seen.insert(*idx);
            }
            while self.indices.len() < min_indicators {
                let idx = rng.random_range(0..n_indicators.max(1));
                if seen.insert(idx) {
                    self.indices.push(idx);
                    self.weights.push(rng.random_range(0.1..1.0));
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub genes: Vec<Gene>,
    pub metrics: Vec<[f64; 11]>,
}

#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    pub max_hold_bars: usize,
    pub trailing_enabled: bool,
    pub trailing_atr_multiplier: f64,
    pub trailing_be_trigger_r: f64,
    pub pip_value: f64,
    pub spread_pips: f64,
    pub commission_per_trade: f64,
    pub pip_value_per_lot: f64,
    pub smc_gate_threshold: f32,
    pub smc_weight_ob: f32,
    pub smc_weight_fvg: f32,
    pub smc_weight_liq: f32,
    pub smc_weight_mtf: f32,
    pub smc_weight_premium: f32,
    pub smc_weight_inducement: f32,
    pub smc_weight_bos: f32,
    pub smc_weight_choch: f32,
    pub smc_weight_eqh: f32,
    pub smc_weight_eql: f32,
    pub smc_weight_displacement: f32,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            max_hold_bars: 0,
            trailing_enabled: false,
            trailing_atr_multiplier: 1.0,
            trailing_be_trigger_r: 1.0,
            pip_value: env_f64("FOREX_BOT_PROP_PIP_VALUE", 0.0001),
            spread_pips: env_f64("FOREX_BOT_PROP_SPREAD_PIPS", 1.5),
            commission_per_trade: env_f64("FOREX_BOT_PROP_COMMISSION", 0.0),
            pip_value_per_lot: env_f64("FOREX_BOT_PROP_PIP_VALUE_PER_LOT", 10.0),
            smc_gate_threshold: env_f32("FOREX_BOT_PROP_SMC_GATE", 0.75),
            smc_weight_ob: env_f32("FOREX_BOT_PROP_SMC_W_OB", 1.0),
            smc_weight_fvg: env_f32("FOREX_BOT_PROP_SMC_W_FVG", 1.0),
            smc_weight_liq: env_f32("FOREX_BOT_PROP_SMC_W_LIQ", 1.0),
            smc_weight_mtf: env_f32("FOREX_BOT_PROP_SMC_W_MTF", 1.0),
            smc_weight_premium: env_f32("FOREX_BOT_PROP_SMC_W_PREMIUM", 1.0),
            smc_weight_inducement: env_f32("FOREX_BOT_PROP_SMC_W_INDUCEMENT", 1.0),
            smc_weight_bos: env_f32("FOREX_BOT_PROP_SMC_W_BOS", 1.0),
            smc_weight_choch: env_f32("FOREX_BOT_PROP_SMC_W_CHOCH", 1.0),
            smc_weight_eqh: env_f32("FOREX_BOT_PROP_SMC_W_EQH", 1.0),
            smc_weight_eql: env_f32("FOREX_BOT_PROP_SMC_W_EQL", 1.0),
            smc_weight_displacement: env_f32("FOREX_BOT_PROP_SMC_W_DISPLACEMENT", 1.0),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SmcSearchConfig {
    force_ratio: f64,
    min_flags: usize,
    p_ob: f64,
    p_fvg: f64,
    p_liq: f64,
    p_premium: f64,
    p_inducement: f64,
    p_mtf: f64,
    p_bos: f64,
    p_choch: f64,
    p_eqh: f64,
    p_eql: f64,
    p_displacement: f64,
}

impl SmcSearchConfig {
    fn from_env() -> Self {
        let default_p = clamp01(env_f64("FOREX_BOT_PROP_SMC_ENABLE_P", 0.50));
        let mut cfg = Self {
            force_ratio: clamp01(env_f64("FOREX_BOT_PROP_SMC_FORCE_RATIO", 0.65)),
            min_flags: env_usize("FOREX_BOT_PROP_SMC_MIN_FLAGS", 1),
            p_ob: clamp01(env_f64("FOREX_BOT_PROP_SMC_P_OB", default_p)),
            p_fvg: clamp01(env_f64("FOREX_BOT_PROP_SMC_P_FVG", default_p)),
            p_liq: clamp01(env_f64("FOREX_BOT_PROP_SMC_P_LIQ", default_p)),
            p_premium: clamp01(env_f64("FOREX_BOT_PROP_SMC_P_PREMIUM", default_p)),
            p_inducement: clamp01(env_f64("FOREX_BOT_PROP_SMC_P_INDUCEMENT", default_p)),
            p_mtf: clamp01(env_f64("FOREX_BOT_PROP_SMC_P_MTF", 0.85)),
            p_bos: clamp01(env_f64("FOREX_BOT_PROP_SMC_P_BOS", default_p)),
            p_choch: clamp01(env_f64("FOREX_BOT_PROP_SMC_P_CHOCH", default_p)),
            p_eqh: clamp01(env_f64("FOREX_BOT_PROP_SMC_P_EQH", default_p)),
            p_eql: clamp01(env_f64("FOREX_BOT_PROP_SMC_P_EQL", default_p)),
            p_displacement: clamp01(env_f64("FOREX_BOT_PROP_SMC_P_DISPLACEMENT", default_p)),
        };
        if !env_bool("FOREX_BOT_PROP_SMC_FORCE_ENABLED", true) {
            cfg.force_ratio = 0.0;
            cfg.min_flags = 0;
        }
        cfg
    }
}

fn smc_structural_flag_count(gene: &Gene) -> usize {
    let mut n = 0usize;
    if gene.use_ob {
        n += 1;
    }
    if gene.use_fvg {
        n += 1;
    }
    if gene.use_liq_sweep {
        n += 1;
    }
    if gene.use_premium_discount {
        n += 1;
    }
    if gene.use_inducement {
        n += 1;
    }
    if gene.use_bos {
        n += 1;
    }
    if gene.use_choch {
        n += 1;
    }
    if gene.use_eqh {
        n += 1;
    }
    if gene.use_eql {
        n += 1;
    }
    if gene.use_displacement {
        n += 1;
    }
    n
}

fn randomize_smc_flags(gene: &mut Gene, cfg: &SmcSearchConfig, rng: &mut impl Rng) {
    gene.use_ob = rng.random_bool(cfg.p_ob);
    gene.use_fvg = rng.random_bool(cfg.p_fvg);
    gene.use_liq_sweep = rng.random_bool(cfg.p_liq);
    gene.use_premium_discount = rng.random_bool(cfg.p_premium);
    gene.use_inducement = rng.random_bool(cfg.p_inducement);
    gene.mtf_confirmation = rng.random_bool(cfg.p_mtf);
    gene.use_bos = rng.random_bool(cfg.p_bos);
    gene.use_choch = rng.random_bool(cfg.p_choch);
    gene.use_eqh = rng.random_bool(cfg.p_eqh);
    gene.use_eql = rng.random_bool(cfg.p_eql);
    gene.use_displacement = rng.random_bool(cfg.p_displacement);
}

fn enforce_min_structural_smc_flags(gene: &mut Gene, cfg: &SmcSearchConfig, rng: &mut impl Rng) {
    let need = cfg.min_flags.min(10);
    if need == 0 {
        return;
    }
    while smc_structural_flag_count(gene) < need {
        match rng.random_range(0..10) {
            0 => gene.use_ob = true,
            1 => gene.use_fvg = true,
            2 => gene.use_liq_sweep = true,
            3 => gene.use_premium_discount = true,
            4 => gene.use_inducement = true,
            5 => gene.use_bos = true,
            6 => gene.use_choch = true,
            7 => gene.use_eqh = true,
            8 => gene.use_eql = true,
            _ => gene.use_displacement = true,
        }
    }
    if !gene.mtf_confirmation && rng.random_bool(cfg.p_mtf.max(0.5)) {
        gene.mtf_confirmation = true;
    }
}

fn enforce_population_smc_ratio(genes: &mut [Gene], cfg: &SmcSearchConfig) {
    if genes.is_empty() {
        return;
    }
    let target = ((genes.len() as f64) * cfg.force_ratio).ceil() as usize;
    if target == 0 {
        return;
    }
    let mut active = genes
        .iter()
        .filter(|g| smc_structural_flag_count(g) > 0)
        .count();
    if active >= target {
        return;
    }
    let mut rng = rand::rng();
    for gene in genes.iter_mut() {
        if active >= target {
            break;
        }
        if smc_structural_flag_count(gene) > 0 {
            continue;
        }
        enforce_min_structural_smc_flags(gene, cfg, &mut rng);
        active += 1;
    }
}

pub fn month_day_indices(timestamps: &[i64]) -> (Vec<i64>, Vec<i64>) {
    let mut months = Vec::with_capacity(timestamps.len());
    let mut days = Vec::with_capacity(timestamps.len());
    for ts in timestamps {
        let dt = Utc.timestamp_millis_opt(*ts).single();
        if let Some(dt) = dt {
            let month_key = (dt.year() as i64) * 12 + dt.month() as i64;
            let day_key = (dt.year() as i64) * 10000 + (dt.month() as i64) * 100 + dt.day() as i64;
            months.push(month_key);
            days.push(day_key);
        } else {
            months.push(0);
            days.push(0);
        }
    }
    (months, days)
}

fn build_gene_arrays(genes: &[Gene]) -> (Vec<i32>, Vec<i32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut offsets = Vec::with_capacity(genes.len() + 1);
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    let mut long_thr = Vec::with_capacity(genes.len());
    let mut short_thr = Vec::with_capacity(genes.len());
    offsets.push(0);
    for gene in genes {
        long_thr.push(gene.long_threshold);
        short_thr.push(gene.short_threshold);
        for (idx, weight) in gene.indices.iter().zip(gene.weights.iter()) {
            indices.push(*idx as i32);
            weights.push(*weight);
        }
        offsets.push(indices.len() as i32);
    }
    (offsets, indices, weights, long_thr, short_thr)
}

fn transpose_features(frame: &FeatureFrame) -> Array2<f32> {
    frame.data.t().to_owned()
}

#[derive(Debug, Clone, Copy, Default)]
struct SmcColumns {
    ob: Option<usize>,
    fvg: Option<usize>,
    liq: Option<usize>,
    trend: Option<usize>,
    premium: Option<usize>,
    inducement: Option<usize>,
    bos: Option<usize>,
    choch: Option<usize>,
    eqh: Option<usize>,
    eql: Option<usize>,
    displacement: Option<usize>,
}

fn normalize_feature_name(name: &str) -> String {
    name.to_ascii_lowercase()
        .replace('-', "_")
        .replace(' ', "_")
}

fn find_feature_column(names: &[String], aliases: &[&str]) -> Option<usize> {
    let normalized_aliases: Vec<String> =
        aliases.iter().map(|a| normalize_feature_name(a)).collect();
    for (idx, raw) in names.iter().enumerate() {
        let norm = normalize_feature_name(raw);
        if normalized_aliases
            .iter()
            .any(|a| norm == *a || norm.contains(a))
        {
            return Some(idx);
        }
    }
    None
}

fn quantize_dir(value: f32) -> i8 {
    if value > 1e-9 {
        1
    } else if value < -1e-9 {
        -1
    } else {
        0
    }
}

fn quantize_binary(value: f32) -> i8 {
    if value > 1e-9 {
        1
    } else {
        0
    }
}

fn detect_smc_columns(names: &[String]) -> SmcColumns {
    SmcColumns {
        ob: find_feature_column(names, &["smc_ob", "order_block", "ob"]),
        fvg: find_feature_column(names, &["smc_fvg", "fair_value_gap", "fvg"]),
        liq: find_feature_column(names, &["smc_liq", "liquidity_sweep", "liq_sweep", "liq"]),
        trend: find_feature_column(names, &["smc_trend", "trend", "market_trend"]),
        premium: find_feature_column(names, &["smc_premium", "premium_discount", "premium"]),
        inducement: find_feature_column(names, &["smc_inducement", "inducement"]),
        bos: find_feature_column(names, &["smc_bos", "bos", "break_of_structure"]),
        choch: find_feature_column(names, &["smc_choch", "choch", "change_of_character"]),
        eqh: find_feature_column(names, &["smc_eqh", "eqh", "equal_highs"]),
        eql: find_feature_column(names, &["smc_eql", "eql", "equal_lows"]),
        displacement: find_feature_column(
            names,
            &["smc_displacement", "displacement", "impulse_displacement"],
        ),
    }
}

fn derive_smc_arrays(
    ohlcv: &Ohlcv,
) -> (
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
) {
    let n = ohlcv.close.len();
    let mut ob = vec![0_i8; n];
    let mut fvg = vec![0_i8; n];
    let mut liq = vec![0_i8; n];
    let mut trend = vec![0_i8; n];
    let mut premium = vec![0_i8; n];
    let mut inducement = vec![0_i8; n];
    let mut bos = vec![0_i8; n];
    let mut choch = vec![0_i8; n];
    let mut eqh = vec![0_i8; n];
    let mut eql = vec![0_i8; n];
    let mut displacement = vec![0_i8; n];
    if n == 0 {
        return (
            ob,
            fvg,
            liq,
            trend,
            premium,
            inducement,
            bos,
            choch,
            eqh,
            eql,
            displacement,
        );
    }

    let lookback = 12usize;
    let eq_lookback = 20usize;
    let displacement_lookback = 20usize;
    for i in 0..n {
        // Trend proxy: 12-bar momentum sign.
        if i >= lookback {
            let d = ohlcv.close[i] - ohlcv.close[i - lookback];
            trend[i] = if d > 0.0 {
                1
            } else if d < 0.0 {
                -1
            } else {
                0
            };
        } else if i > 0 {
            let d = ohlcv.close[i] - ohlcv.close[i - 1];
            trend[i] = if d > 0.0 {
                1
            } else if d < 0.0 {
                -1
            } else {
                0
            };
        }

        // Premium/discount proxy inside current candle range:
        // close in discount half -> +1 (long context), premium half -> -1 (short context)
        let mid = (ohlcv.high[i] + ohlcv.low[i]) * 0.5;
        premium[i] = if ohlcv.close[i] <= mid { 1 } else { -1 };

        if i >= 1 {
            // Order-block style proxy via engulfing behavior.
            let bull = ohlcv.close[i] > ohlcv.open[i]
                && ohlcv.close[i - 1] < ohlcv.open[i - 1]
                && ohlcv.close[i] >= ohlcv.high[i - 1];
            let bear = ohlcv.close[i] < ohlcv.open[i]
                && ohlcv.close[i - 1] > ohlcv.open[i - 1]
                && ohlcv.close[i] <= ohlcv.low[i - 1];
            ob[i] = if bull {
                1
            } else if bear {
                -1
            } else {
                0
            };

            // Inducement proxy: wick imbalance relative to body.
            let body = (ohlcv.close[i] - ohlcv.open[i]).abs();
            let upper = ohlcv.high[i] - ohlcv.open[i].max(ohlcv.close[i]);
            let lower = ohlcv.open[i].min(ohlcv.close[i]) - ohlcv.low[i];
            if body > 1e-12 && ((upper / body) > 2.0 || (lower / body) > 2.0) {
                inducement[i] = 1;
            }
        }

        if i >= 2 {
            // Fair-value-gap proxy.
            if ohlcv.low[i] > ohlcv.high[i - 2] {
                fvg[i] = 1;
            } else if ohlcv.high[i] < ohlcv.low[i - 2] {
                fvg[i] = -1;
            }
        }

        if i >= 3 {
            // Liquidity sweep proxy over previous 3 bars.
            let prev_low = ohlcv.low[(i - 3)..i]
                .iter()
                .fold(f64::INFINITY, |a, b| a.min(*b));
            let prev_high = ohlcv.high[(i - 3)..i]
                .iter()
                .fold(f64::NEG_INFINITY, |a, b| a.max(*b));
            if ohlcv.low[i] < prev_low && ohlcv.close[i] > prev_low {
                liq[i] = 1;
            } else if ohlcv.high[i] > prev_high && ohlcv.close[i] < prev_high {
                liq[i] = -1;
            }
        }

        if i >= lookback {
            let prev_low = ohlcv.low[(i - lookback)..i]
                .iter()
                .fold(f64::INFINITY, |a, b| a.min(*b));
            let prev_high = ohlcv.high[(i - lookback)..i]
                .iter()
                .fold(f64::NEG_INFINITY, |a, b| a.max(*b));
            if ohlcv.close[i] > prev_high {
                bos[i] = 1;
            } else if ohlcv.close[i] < prev_low {
                bos[i] = -1;
            }
        }

        if i >= 1 && trend[i] != 0 && trend[i - 1] != 0 && trend[i] != trend[i - 1] {
            choch[i] = trend[i];
        }

        if i >= eq_lookback {
            let lb = i - eq_lookback;
            let mut range_sum = 0.0;
            for j in lb..=i {
                range_sum += (ohlcv.high[j] - ohlcv.low[j]).abs();
            }
            let avg_range = range_sum / ((eq_lookback as f64) + 1.0);
            let tol = (avg_range * 0.1).max(1e-9);
            for j in lb..i {
                if (ohlcv.high[i] - ohlcv.high[j]).abs() <= tol {
                    eqh[i] = -1;
                    break;
                }
            }
            for j in lb..i {
                if (ohlcv.low[i] - ohlcv.low[j]).abs() <= tol {
                    eql[i] = 1;
                    break;
                }
            }
        }

        if i >= displacement_lookback {
            let body = (ohlcv.close[i] - ohlcv.open[i]).abs();
            let mut avg_body = 0.0;
            for j in (i - displacement_lookback)..i {
                avg_body += (ohlcv.close[j] - ohlcv.open[j]).abs();
            }
            avg_body /= displacement_lookback as f64;
            if avg_body > 1e-12 && body >= (1.8 * avg_body) {
                displacement[i] = if ohlcv.close[i] > ohlcv.open[i] {
                    1
                } else if ohlcv.close[i] < ohlcv.open[i] {
                    -1
                } else {
                    0
                };
            }
        }
    }

    (
        ob,
        fvg,
        liq,
        trend,
        premium,
        inducement,
        bos,
        choch,
        eqh,
        eql,
        displacement,
    )
}

fn build_smc_arrays(
    frame: &FeatureFrame,
    ohlcv: &Ohlcv,
) -> (
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
    Vec<i8>,
) {
    let n = frame.data.nrows();
    let cols = detect_smc_columns(&frame.names);
    let (
        mut ob,
        mut fvg,
        mut liq,
        mut trend,
        mut premium,
        mut inducement,
        mut bos,
        mut choch,
        mut eqh,
        mut eql,
        mut displacement,
    ) = derive_smc_arrays(ohlcv);

    let apply_dir_col = |target: &mut Vec<i8>, col_opt: Option<usize>| {
        if let Some(col) = col_opt {
            if col < frame.data.ncols() {
                for i in 0..n {
                    target[i] = quantize_dir(frame.data[(i, col)]);
                }
            }
        }
    };
    let apply_binary_col = |target: &mut Vec<i8>, col_opt: Option<usize>| {
        if let Some(col) = col_opt {
            if col < frame.data.ncols() {
                for i in 0..n {
                    target[i] = quantize_binary(frame.data[(i, col)]);
                }
            }
        }
    };
    let apply_eqh_col = |target: &mut Vec<i8>, col_opt: Option<usize>| {
        if let Some(col) = col_opt {
            if col < frame.data.ncols() {
                for i in 0..n {
                    let v = frame.data[(i, col)];
                    let q = quantize_dir(v);
                    if q != 0 {
                        target[i] = q;
                    } else if quantize_binary(v) != 0 {
                        target[i] = -1;
                    } else {
                        target[i] = 0;
                    }
                }
            }
        }
    };
    let apply_eql_col = |target: &mut Vec<i8>, col_opt: Option<usize>| {
        if let Some(col) = col_opt {
            if col < frame.data.ncols() {
                for i in 0..n {
                    let v = frame.data[(i, col)];
                    let q = quantize_dir(v);
                    if q != 0 {
                        target[i] = q;
                    } else if quantize_binary(v) != 0 {
                        target[i] = 1;
                    } else {
                        target[i] = 0;
                    }
                }
            }
        }
    };
    let apply_dir_fill_zeros = |target: &mut Vec<i8>, col_opt: Option<usize>| {
        if let Some(col) = col_opt {
            if col < frame.data.ncols() {
                for i in 0..n {
                    if target[i] == 0 {
                        target[i] = quantize_dir(frame.data[(i, col)]);
                    }
                }
            }
        }
    };
    let apply_eq_levels = |target: &mut Vec<i8>, eqh_col: Option<usize>, eql_col: Option<usize>| {
        if let Some(col) = eqh_col {
            if col < frame.data.ncols() {
                for i in 0..n {
                    if quantize_binary(frame.data[(i, col)]) != 0 {
                        target[i] = -1;
                    }
                }
            }
        }
        if let Some(col) = eql_col {
            if col < frame.data.ncols() {
                for i in 0..n {
                    if quantize_binary(frame.data[(i, col)]) != 0 {
                        target[i] = 1;
                    }
                }
            }
        }
    };

    apply_dir_col(&mut ob, cols.ob);
    apply_dir_col(&mut fvg, cols.fvg);
    apply_dir_col(&mut liq, cols.liq);
    apply_dir_col(&mut trend, cols.trend);
    apply_dir_col(&mut premium, cols.premium);
    apply_binary_col(&mut inducement, cols.inducement);
    apply_dir_col(&mut bos, cols.bos);
    apply_dir_col(&mut choch, cols.choch);
    apply_eqh_col(&mut eqh, cols.eqh);
    apply_eql_col(&mut eql, cols.eql);
    apply_dir_col(&mut displacement, cols.displacement);
    apply_dir_fill_zeros(&mut ob, cols.bos);
    apply_dir_fill_zeros(&mut ob, cols.choch);
    apply_eq_levels(&mut liq, cols.eqh, cols.eql);
    apply_dir_fill_zeros(&mut trend, cols.bos);
    apply_dir_fill_zeros(&mut trend, cols.choch);
    apply_dir_fill_zeros(&mut trend, cols.displacement);
    if let Some(col) = cols.displacement {
        if col < frame.data.ncols() {
            for i in 0..n {
                if quantize_dir(frame.data[(i, col)]) != 0 {
                    inducement[i] = 1;
                }
            }
        }
    }
    for i in 0..n {
        if displacement[i] != 0 {
            inducement[i] = 1;
        }
    }

    (
        ob,
        fvg,
        liq,
        trend,
        premium,
        inducement,
        bos,
        choch,
        eqh,
        eql,
        displacement,
    )
}

fn new_random_gene(
    n_indicators: usize,
    max_indicators: usize,
    generation: usize,
    smc_cfg: &SmcSearchConfig,
) -> Gene {
    let mut rng = rand::rng();
    let min_indicators = 1.min(n_indicators.max(1));
    let max_indicators = max_indicators.max(min_indicators).min(n_indicators.max(1));
    let count = rng.random_range(min_indicators..=max_indicators);
    let sample = sample(&mut rng, n_indicators.max(1), count);
    let indices: Vec<usize> = sample.iter().collect();
    let weights: Vec<f32> = (0..count).map(|_| rng.random_range(0.1..1.0)).collect();
    let long_threshold = rng.random_range(0.15..0.55);
    let short_threshold = -rng.random_range(0.15..0.55);
    let (sl_pips, tp_pips) = if rng.random_bool(0.2) {
        (0.0, 0.0) // auto-infer later
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
    };
    randomize_smc_flags(&mut gene, smc_cfg, &mut rng);
    enforce_min_structural_smc_flags(&mut gene, smc_cfg, &mut rng);
    gene
}

fn generate_random_genes(
    n_genes: usize,
    n_indicators: usize,
    max_indicators: usize,
    generation: usize,
    smc_cfg: &SmcSearchConfig,
) -> Vec<Gene> {
    (0..n_genes)
        .map(|_| new_random_gene(n_indicators, max_indicators, generation, smc_cfg))
        .collect()
}

pub fn signals_for_gene(features: &FeatureFrame, gene: &Gene) -> Vec<i8> {
    let n_samples = features.data.nrows();
    let mut combined = vec![0.0_f32; n_samples];
    for (idx, weight) in gene.indices.iter().zip(gene.weights.iter()) {
        if *idx >= features.data.ncols() {
            continue;
        }
        let col = features.data.column(*idx);
        for (i, v) in col.iter().enumerate() {
            combined[i] += *weight * *v;
        }
    }
    let mut signals = vec![0_i8; n_samples];
    for i in 0..n_samples {
        let v = combined[i];
        if v >= gene.long_threshold {
            signals[i] = 1;
        } else if v <= gene.short_threshold {
            signals[i] = -1;
        }
    }
    signals
}

pub fn evaluate_genes(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    genes: &[Gene],
    config: &EvaluationConfig,
) -> Result<Vec<[f64; 11]>> {
    if features.data.nrows() == 0 || features.data.ncols() == 0 {
        bail!("empty feature matrix");
    }
    let n_samples = features.data.nrows();
    if ohlcv.close.len() != n_samples {
        bail!("ohlcv length does not match feature rows");
    }

    let indicators = transpose_features(features);
    let (offsets, indices, weights, long_thr, short_thr) = build_gene_arrays(genes);
    let (sl_pips, tp_pips) = resolve_stop_target_arrays(genes, ohlcv, config);
    let (months, days) = month_day_indices(&features.timestamps);
    let (
        ob_arr,
        fvg_arr,
        liq_arr,
        trend_arr,
        premium_arr,
        inducement_arr,
        bos_arr,
        choch_arr,
        eqh_arr,
        eql_arr,
        displacement_arr,
    ) = build_smc_arrays(features, ohlcv);
    let use_ob: Vec<i8> = genes.iter().map(|g| if g.use_ob { 1 } else { 0 }).collect();
    let use_fvg: Vec<i8> = genes
        .iter()
        .map(|g| if g.use_fvg { 1 } else { 0 })
        .collect();
    let use_liq: Vec<i8> = genes
        .iter()
        .map(|g| if g.use_liq_sweep { 1 } else { 0 })
        .collect();
    let use_mtf: Vec<i8> = genes
        .iter()
        .map(|g| if g.mtf_confirmation { 1 } else { 0 })
        .collect();
    let use_premium: Vec<i8> = genes
        .iter()
        .map(|g| if g.use_premium_discount { 1 } else { 0 })
        .collect();
    let use_inducement: Vec<i8> = genes
        .iter()
        .map(|g| if g.use_inducement { 1 } else { 0 })
        .collect();
    let use_bos: Vec<i8> = genes
        .iter()
        .map(|g| if g.use_bos { 1 } else { 0 })
        .collect();
    let use_choch: Vec<i8> = genes
        .iter()
        .map(|g| if g.use_choch { 1 } else { 0 })
        .collect();
    let use_eqh: Vec<i8> = genes
        .iter()
        .map(|g| if g.use_eqh { 1 } else { 0 })
        .collect();
    let use_eql: Vec<i8> = genes
        .iter()
        .map(|g| if g.use_eql { 1 } else { 0 })
        .collect();
    let use_displacement: Vec<i8> = genes
        .iter()
        .map(|g| if g.use_displacement { 1 } else { 0 })
        .collect();

    let metrics = crate::eval::evaluate_population_core(
        &ohlcv.close,
        &ohlcv.high,
        &ohlcv.low,
        indicators.view(),
        &offsets,
        &indices,
        &weights,
        &long_thr,
        &short_thr,
        &months,
        &days,
        &sl_pips,
        &tp_pips,
        &ob_arr,
        &fvg_arr,
        &liq_arr,
        &trend_arr,
        &premium_arr,
        &inducement_arr,
        &bos_arr,
        &choch_arr,
        &eqh_arr,
        &eql_arr,
        &displacement_arr,
        &use_ob,
        &use_fvg,
        &use_liq,
        &use_mtf,
        &use_premium,
        &use_inducement,
        &use_bos,
        &use_choch,
        &use_eqh,
        &use_eql,
        &use_displacement,
        config.smc_gate_threshold,
        config.smc_weight_ob,
        config.smc_weight_fvg,
        config.smc_weight_liq,
        config.smc_weight_mtf,
        config.smc_weight_premium,
        config.smc_weight_inducement,
        config.smc_weight_bos,
        config.smc_weight_choch,
        config.smc_weight_eqh,
        config.smc_weight_eql,
        config.smc_weight_displacement,
        config.max_hold_bars,
        config.trailing_enabled,
        config.trailing_atr_multiplier,
        config.trailing_be_trigger_r,
        config.pip_value,
        config.spread_pips,
        config.commission_per_trade,
        config.pip_value_per_lot,
    )
    .map_err(|e| anyhow::anyhow!(e))?;

    Ok(metrics)
}

fn resolve_stop_target_arrays(
    genes: &[Gene],
    ohlcv: &Ohlcv,
    config: &EvaluationConfig,
) -> (Vec<f64>, Vec<f64>) {
    let pip_size = if config.pip_value.is_finite() && config.pip_value > 0.0 {
        config.pip_value
    } else {
        0.0001
    };
    let default = infer_stop_target_pips(
        &ohlcv.open,
        &ohlcv.high,
        &ohlcv.low,
        &ohlcv.close,
        &StopTargetSettings::default(),
        pip_size,
        0,
    );
    let (default_sl, default_tp) = default
        .map(|(sl, tp, _rr)| (sl, tp))
        .unwrap_or((20.0, 40.0));

    let mut sl_pips = Vec::with_capacity(genes.len());
    let mut tp_pips = Vec::with_capacity(genes.len());
    for gene in genes {
        let sl = if gene.sl_pips.is_finite() && gene.sl_pips > 0.0 {
            gene.sl_pips
        } else {
            default_sl
        };
        let tp = if gene.tp_pips.is_finite() && gene.tp_pips > 0.0 {
            gene.tp_pips
        } else {
            default_tp
        };
        sl_pips.push(sl);
        tp_pips.push(tp);
    }
    (sl_pips, tp_pips)
}

fn score_from_metrics(metrics: &[f64; 11]) -> f64 {
    let net_profit = metrics[0];
    let sharpe = metrics[1];
    let max_dd = metrics[3];
    let profit_factor = metrics[5];
    let dd_cap = 0.07;
    let pfloor = 1.0;
    let dd_penalty = 10.0 * (max_dd - dd_cap).max(0.0);
    let pf_penalty = if profit_factor <= pfloor { 5.0 } else { 0.0 };
    sharpe + (net_profit / 10_000.0) - dd_penalty - pf_penalty
}

fn apply_metrics(genes: &mut [Gene], metrics: &[[f64; 11]]) {
    for (gene, m) in genes.iter_mut().zip(metrics.iter()) {
        gene.fitness = score_from_metrics(m);
        gene.sharpe_ratio = m[1];
        gene.max_drawdown = m[3];
        gene.win_rate = m[4];
        gene.profit_factor = m[5];
        gene.expectancy = m[6];
        gene.trades_count = m[8].max(0.0) as usize;
        gene.slice_pass_rate = 1.0;
    }
}

fn crossover(a: &Gene, b: &Gene, generation: usize) -> Gene {
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
    let long_threshold = if rng.random_bool(0.5) {
        a.long_threshold
    } else {
        b.long_threshold
    };
    let short_threshold = if rng.random_bool(0.5) {
        a.short_threshold
    } else {
        b.short_threshold
    };
    let use_ob = if rng.random_bool(0.5) {
        a.use_ob
    } else {
        b.use_ob
    };
    let use_fvg = if rng.random_bool(0.5) {
        a.use_fvg
    } else {
        b.use_fvg
    };
    let use_liq_sweep = if rng.random_bool(0.5) {
        a.use_liq_sweep
    } else {
        b.use_liq_sweep
    };
    let mtf_confirmation = if rng.random_bool(0.5) {
        a.mtf_confirmation
    } else {
        b.mtf_confirmation
    };
    let use_premium_discount = if rng.random_bool(0.5) {
        a.use_premium_discount
    } else {
        b.use_premium_discount
    };
    let use_inducement = if rng.random_bool(0.5) {
        a.use_inducement
    } else {
        b.use_inducement
    };
    let use_bos = if rng.random_bool(0.5) {
        a.use_bos
    } else {
        b.use_bos
    };
    let use_choch = if rng.random_bool(0.5) {
        a.use_choch
    } else {
        b.use_choch
    };
    let use_eqh = if rng.random_bool(0.5) {
        a.use_eqh
    } else {
        b.use_eqh
    };
    let use_eql = if rng.random_bool(0.5) {
        a.use_eql
    } else {
        b.use_eql
    };
    let use_displacement = if rng.random_bool(0.5) {
        a.use_displacement
    } else {
        b.use_displacement
    };
    let tp_pips = if rng.random_bool(0.5) {
        a.tp_pips
    } else {
        b.tp_pips
    };
    let sl_pips = if rng.random_bool(0.5) {
        a.sl_pips
    } else {
        b.sl_pips
    };
    let strategy_id = format!("gene_{}_{}", rng.random_range(0..1_000_000u64), generation);
    Gene {
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
        use_ob,
        use_fvg,
        use_liq_sweep,
        mtf_confirmation,
        use_premium_discount,
        use_inducement,
        use_bos,
        use_choch,
        use_eqh,
        use_eql,
        use_displacement,
        tp_pips,
        sl_pips,
        slice_pass_rate: 0.0,
    }
}

fn mutate(
    gene: &Gene,
    n_indicators: usize,
    max_indicators: usize,
    generation: usize,
    smc_cfg: &SmcSearchConfig,
) -> Gene {
    let mut rng = rand::rng();
    let mut mutated = gene.clone();
    let mutation_type = rng.random_range(0..4);
    match mutation_type {
        0 => {
            if !mutated.indices.is_empty() && rng.random_bool(0.5) {
                let idx = rng.random_range(0..mutated.indices.len());
                mutated.indices[idx] = rng.random_range(0..n_indicators.max(1));
                mutated.weights[idx] = rng.random_range(0.1..1.0);
            } else {
                let min_indicators = 1.min(n_indicators.max(1));
                let max_indicators = max_indicators.max(min_indicators).min(n_indicators.max(1));
                let count = rng.random_range(min_indicators..=max_indicators);
                let sample = sample(&mut rng, n_indicators.max(1), count);
                mutated.indices = sample.iter().collect();
                mutated.weights = (0..count).map(|_| rng.random_range(0.1..1.0)).collect();
            }
        }
        1 => {
            mutated.long_threshold =
                (mutated.long_threshold * rng.random_range(0.7..1.3)).clamp(0.08, 0.8);
            mutated.short_threshold =
                (mutated.short_threshold * rng.random_range(0.7..1.3)).clamp(-0.8, -0.08);
        }
        2 => {
            mutated.tp_pips = (mutated.tp_pips * rng.random_range(0.8..1.2)).clamp(10.0, 100.0);
            mutated.sl_pips = (mutated.sl_pips * rng.random_range(0.8..1.2)).clamp(5.0, 50.0);
        }
        _ => {
            randomize_smc_flags(&mut mutated, smc_cfg, &mut rng);
        }
    }
    if rng.random_bool(0.25) {
        enforce_min_structural_smc_flags(&mut mutated, smc_cfg, &mut rng);
    }
    mutated.strategy_id = format!("gene_{}_{}", rng.random_range(0..1_000_000u64), generation);
    mutated.generation = generation;
    mutated
}

pub fn random_search(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    n_genes: usize,
    max_indicators: usize,
) -> Result<SearchResult> {
    let n_indicators = features.data.ncols();
    let smc_cfg = SmcSearchConfig::from_env();
    let mut genes = generate_random_genes(n_genes, n_indicators, max_indicators, 0, &smc_cfg);
    enforce_population_smc_ratio(&mut genes, &smc_cfg);
    for gene in genes.iter_mut() {
        gene.normalize(n_indicators, 1);
    }
    let metrics = evaluate_genes(features, ohlcv, &genes, &EvaluationConfig::default())?;
    Ok(SearchResult { genes, metrics })
}

pub fn evolve_search(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    population: usize,
    generations: usize,
    max_indicators: usize,
) -> Result<SearchResult> {
    if population == 0 {
        bail!("population must be > 0");
    }
    let n_indicators = features.data.ncols();
    let smc_cfg = SmcSearchConfig::from_env();
    let gate_start = env_f32(
        "FOREX_BOT_PROP_SMC_GATE_START",
        env_f32("FOREX_BOT_PROP_SMC_GATE", 0.75),
    );
    let gate_end = env_f32("FOREX_BOT_PROP_SMC_GATE_END", 0.35);
    let gate_curve = env_f32("FOREX_BOT_PROP_SMC_GATE_CURVE", 1.0).max(0.1);
    let gate_stagnation_step = env_f32("FOREX_BOT_PROP_SMC_GATE_STAGNATION_STEP", 0.03).max(0.0);
    let gate_lo = gate_start.min(gate_end);
    let gate_hi = gate_start.max(gate_end);
    let mut eval_cfg = EvaluationConfig::default();
    eval_cfg.smc_gate_threshold = gate_start.max(gate_lo).min(gate_hi);

    let seen_retry_attempts = std::env::var("FOREX_BOT_PROP_SEEN_RETRY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(16)
        .max(1);
    let mut seen_memory = SeenSignatureMemory::from_env();
    let mut genes = generate_random_genes(population, n_indicators, max_indicators, 0, &smc_cfg);
    enforce_population_smc_ratio(&mut genes, &smc_cfg);
    genes = genes
        .into_iter()
        .map(|gene| {
            unique_candidate_or_retry(
                gene,
                &mut seen_memory,
                n_indicators,
                max_indicators,
                0,
                seen_retry_attempts,
                &smc_cfg,
            )
        })
        .collect();
    let mut best_metrics = Vec::new();
    let mut profitable_archive: Vec<(Gene, [f64; 11])> = Vec::new();
    let mut seen_strategy_ids: HashSet<String> = HashSet::new();
    let archive_mode = std::env::var("FOREX_BOT_PROP_ARCHIVE_MODE")
        .unwrap_or_else(|_| "net".to_string())
        .to_ascii_lowercase();
    let archive_min_net = std::env::var("FOREX_BOT_PROP_ARCHIVE_MIN_NET")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    let archive_min_pf = std::env::var("FOREX_BOT_PROP_ARCHIVE_MIN_PF")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(1.0);
    let archive_min_sharpe = std::env::var("FOREX_BOT_PROP_ARCHIVE_MIN_SHARPE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    let archive_cap = std::env::var("FOREX_BOT_PROP_ARCHIVE_CAP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| {
            population
                .saturating_mul(generations.max(1))
                .max(population)
        })
        .max(population.max(1));
    let base_immigrant_ratio = std::env::var("FOREX_BOT_PROP_RANDOM_IMMIGRANTS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .map(|v| v.clamp(0.0, 0.95))
        .unwrap_or(0.25);
    let stagnation_patience = std::env::var("FOREX_BOT_PROP_STAGNATION_GENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2)
        .max(1);
    let mut best_score_seen = f64::NEG_INFINITY;
    let mut stagnant_gens = 0usize;

    if generations == 0 {
        let metrics = evaluate_genes(features, ohlcv, &genes, &eval_cfg)?;
        apply_metrics(&mut genes, &metrics);
        seen_memory.flush();
        return Ok(SearchResult { genes, metrics });
    }

    for gen in 0..generations {
        let progress = if generations <= 1 {
            1.0_f32
        } else {
            (gen as f32) / ((generations - 1) as f32)
        };
        let shaped = progress.powf(gate_curve);
        let mut gate_now = gate_start + (gate_end - gate_start) * shaped;
        if stagnant_gens >= stagnation_patience {
            gate_now -= gate_stagnation_step * (stagnant_gens as f32);
        }
        eval_cfg.smc_gate_threshold = gate_now.max(gate_lo).min(gate_hi);

        let metrics = evaluate_genes(features, ohlcv, &genes, &eval_cfg)?;
        apply_metrics(&mut genes, &metrics);

        let mut scored: Vec<(f64, Gene, [f64; 11])> = genes
            .iter()
            .cloned()
            .zip(metrics.into_iter())
            .map(|(g, m)| (g.fitness, g, m))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let top_score = scored.first().map(|x| x.0).unwrap_or(f64::NEG_INFINITY);
        if top_score.is_finite() && top_score > best_score_seen + 1e-12 {
            best_score_seen = top_score;
            stagnant_gens = 0;
        } else {
            stagnant_gens = stagnant_gens.saturating_add(1);
        }

        for (_score, gene, m) in scored.iter() {
            if profitable_archive.len() >= archive_cap {
                break;
            }
            let net_profit = m[0];
            let sharpe = m[1];
            let profit_factor = m[5];
            let trades = m[8];
            if !net_profit.is_finite()
                || !sharpe.is_finite()
                || !profit_factor.is_finite()
                || !trades.is_finite()
            {
                continue;
            }
            let keep = match archive_mode.as_str() {
                "active" => trades > 0.0,
                "pf" | "profit_factor" => trades > 0.0 && profit_factor > archive_min_pf,
                "sharpe" => trades > 0.0 && sharpe > archive_min_sharpe,
                _ => trades > 0.0 && net_profit > archive_min_net,
            };
            if !keep {
                continue;
            }
            let sid = if gene.strategy_id.is_empty() {
                format!(
                    "{:?}|{:?}|{:.3}|{:.3}",
                    gene.indices, gene.weights, gene.long_threshold, gene.short_threshold
                )
            } else {
                gene.strategy_id.clone()
            };
            if !seen_strategy_ids.insert(sid) {
                continue;
            }
            profitable_archive.push((gene.clone(), *m));
        }

        let elite_count = (population.max(2) as f32 * 0.2) as usize;
        let elite_count = elite_count.max(2).min(scored.len());
        let elites: Vec<Gene> = scored
            .iter()
            .take(elite_count)
            .map(|(_, g, _)| g.clone())
            .collect();
        best_metrics = scored
            .iter()
            .take(elite_count)
            .map(|(_, _, m)| *m)
            .collect();

        if gen + 1 == generations {
            seen_memory.flush();
            if !profitable_archive.is_empty() {
                profitable_archive.sort_by(|a, b| {
                    b.1[0]
                        .partial_cmp(&a.1[0])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let genes = profitable_archive.iter().map(|(g, _)| g.clone()).collect();
                let metrics = profitable_archive.iter().map(|(_, m)| *m).collect();
                return Ok(SearchResult { genes, metrics });
            }
            return Ok(SearchResult {
                genes: elites,
                metrics: best_metrics,
            });
        }

        let mut next = Vec::with_capacity(population);
        next.extend(elites.clone());
        let mut immigrant_ratio = base_immigrant_ratio;
        if stagnant_gens >= stagnation_patience {
            immigrant_ratio = immigrant_ratio.max(0.5);
        }
        let max_new_slots = population.saturating_sub(next.len());
        let immigrant_count = ((population as f64) * immigrant_ratio).round() as usize;
        let immigrant_count = immigrant_count.min(max_new_slots);
        for _ in 0..immigrant_count {
            let immigrant = unique_candidate_or_retry(
                new_random_gene(n_indicators, max_indicators, gen + 1, &smc_cfg),
                &mut seen_memory,
                n_indicators,
                max_indicators,
                gen + 1,
                seen_retry_attempts,
                &smc_cfg,
            );
            next.push(immigrant);
        }
        let mut rng = rand::rng();
        let parent_pool_len = (elite_count.saturating_mul(3))
            .min(scored.len())
            .max(elite_count);
        while next.len() < population {
            let a = &scored[rng.random_range(0..parent_pool_len)].1;
            let b = &scored[rng.random_range(0..parent_pool_len)].1;
            let child = unique_candidate_or_retry(
                mutate(
                    &crossover(a, b, gen + 1),
                    n_indicators,
                    max_indicators,
                    gen + 1,
                    &smc_cfg,
                ),
                &mut seen_memory,
                n_indicators,
                max_indicators,
                gen + 1,
                seen_retry_attempts,
                &smc_cfg,
            );
            next.push(child);
        }
        enforce_population_smc_ratio(&mut next, &smc_cfg);
        genes = next;
        seen_memory.flush();
    }

    seen_memory.flush();
    Ok(SearchResult {
        genes,
        metrics: best_metrics,
    })
}
