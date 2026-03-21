use super::strategy_gene::Gene;
use forex_data::{FeatureFrame, Ohlcv};
use rand::Rng;

#[derive(Debug, Clone, Copy)]
pub struct SmcSearchConfig {
    pub force_ratio: f64,
    pub min_flags: usize,
    pub p_ob: f64,
    pub p_fvg: f64,
    pub p_liq: f64,
    pub p_premium: f64,
    pub p_inducement: f64,
    pub p_mtf: f64,
    pub p_bos: f64,
    pub p_choch: f64,
    pub p_eqh: f64,
    pub p_eql: f64,
    pub p_displacement: f64,
}

impl SmcSearchConfig {
    pub fn from_env() -> Self {
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

        let default_p = env_f64("FOREX_BOT_PROP_SMC_ENABLE_P", 0.50).clamp(0.0, 1.0);
        let mut cfg = Self {
            force_ratio: env_f64("FOREX_BOT_PROP_SMC_FORCE_RATIO", 0.65).clamp(0.0, 1.0),
            min_flags: env_usize("FOREX_BOT_PROP_SMC_MIN_FLAGS", 1),
            p_ob: env_f64("FOREX_BOT_PROP_SMC_P_OB", default_p).clamp(0.0, 1.0),
            p_fvg: env_f64("FOREX_BOT_PROP_SMC_P_FVG", default_p).clamp(0.0, 1.0),
            p_liq: env_f64("FOREX_BOT_PROP_SMC_P_LIQ", default_p).clamp(0.0, 1.0),
            p_premium: env_f64("FOREX_BOT_PROP_SMC_P_PREMIUM", default_p).clamp(0.0, 1.0),
            p_inducement: env_f64("FOREX_BOT_PROP_SMC_P_INDUCEMENT", default_p).clamp(0.0, 1.0),
            p_mtf: env_f64("FOREX_BOT_PROP_SMC_P_MTF", 0.85).clamp(0.0, 1.0),
            p_bos: env_f64("FOREX_BOT_PROP_SMC_P_BOS", default_p).clamp(0.0, 1.0),
            p_choch: env_f64("FOREX_BOT_PROP_SMC_P_CHOCH", default_p).clamp(0.0, 1.0),
            p_eqh: env_f64("FOREX_BOT_PROP_SMC_P_EQH", default_p).clamp(0.0, 1.0),
            p_eql: env_f64("FOREX_BOT_PROP_SMC_P_EQL", default_p).clamp(0.0, 1.0),
            p_displacement: env_f64("FOREX_BOT_PROP_SMC_P_DISPLACEMENT", default_p).clamp(0.0, 1.0),
        };
        if !env_bool("FOREX_BOT_PROP_SMC_FORCE_ENABLED", true) {
            cfg.force_ratio = 0.0;
            cfg.min_flags = 0;
        }
        cfg
    }
}

pub fn randomize_smc_flags(gene: &mut Gene, cfg: &SmcSearchConfig, rng: &mut impl Rng) {
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

pub fn smc_structural_flag_count(gene: &Gene) -> usize {
    let mut n = 0usize;
    if gene.use_ob { n += 1; }
    if gene.use_fvg { n += 1; }
    if gene.use_liq_sweep { n += 1; }
    if gene.use_premium_discount { n += 1; }
    if gene.use_inducement { n += 1; }
    if gene.use_bos { n += 1; }
    if gene.use_choch { n += 1; }
    if gene.use_eqh { n += 1; }
    if gene.use_eql { n += 1; }
    if gene.use_displacement { n += 1; }
    n
}

pub fn enforce_min_structural_smc_flags(gene: &mut Gene, cfg: &SmcSearchConfig, rng: &mut impl Rng) {
    let need = cfg.min_flags.min(10);
    if need == 0 { return; }
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

pub fn enforce_population_smc_ratio(genes: &mut [Gene], cfg: &SmcSearchConfig) {
    if genes.is_empty() { return; }
    let target = ((genes.len() as f64) * cfg.force_ratio).ceil() as usize;
    if target == 0 { return; }
    let mut active = genes.iter().filter(|g| smc_structural_flag_count(g) > 0).count();
    if active >= target { return; }
    let mut rng = rand::rng();
    for gene in genes.iter_mut() {
        if active >= target { break; }
        if smc_structural_flag_count(gene) > 0 { continue; }
        enforce_min_structural_smc_flags(gene, cfg, &mut rng);
        active += 1;
    }
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

pub type SmcSignalTuple = (
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
);

fn normalize_feature_name(name: &str) -> String {
    name.to_ascii_lowercase().replace(['-', ' '], "_")
}

fn find_feature_column(names: &[String], aliases: &[&str]) -> Option<usize> {
    let normalized_aliases: Vec<String> = aliases.iter().map(|a| normalize_feature_name(a)).collect();
    for (idx, raw) in names.iter().enumerate() {
        let norm = normalize_feature_name(raw);
        if normalized_aliases.iter().any(|a| norm == *a || norm.contains(a)) {
            return Some(idx);
        }
    }
    None
}

fn quantize_dir(value: f32) -> i8 {
    if value > 1e-9 { 1 } else if value < -1e-9 { -1 } else { 0 }
}

fn quantize_binary(value: f32) -> i8 {
    if value > 1e-9 { 1 } else { 0 }
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
        displacement: find_feature_column(names, &["smc_displacement", "displacement", "impulse_displacement"]),
    }
}

pub fn derive_smc_arrays(ohlcv: &Ohlcv) -> SmcSignalTuple {
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
    
    if n == 0 { return (ob, fvg, liq, trend, premium, inducement, bos, choch, eqh, eql, displacement); }

    let lookback = 12usize;
    let eq_lookback = 20usize;
    let displacement_lookback = 20usize;
    
    for i in 0..n {
        if i >= lookback {
            let d = ohlcv.close[i] - ohlcv.close[i - lookback];
            trend[i] = if d > 0.0 { 1 } else if d < 0.0 { -1 } else { 0 };
        } else if i > 0 {
            let d = ohlcv.close[i] - ohlcv.close[i - 1];
            trend[i] = if d > 0.0 { 1 } else if d < 0.0 { -1 } else { 0 };
        }

        let mid = (ohlcv.high[i] + ohlcv.low[i]) * 0.5;
        premium[i] = if ohlcv.close[i] <= mid { 1 } else { -1 };

        if i >= 1 {
            let bull = ohlcv.close[i] > ohlcv.open[i] && ohlcv.close[i-1] < ohlcv.open[i-1] && ohlcv.close[i] >= ohlcv.high[i-1];
            let bear = ohlcv.close[i] < ohlcv.open[i] && ohlcv.close[i-1] > ohlcv.open[i-1] && ohlcv.close[i] <= ohlcv.low[i-1];
            ob[i] = if bull { 1 } else if bear { -1 } else { 0 };

            let body = (ohlcv.close[i] - ohlcv.open[i]).abs();
            let upper = ohlcv.high[i] - ohlcv.open[i].max(ohlcv.close[i]);
            let lower = ohlcv.open[i].min(ohlcv.close[i]) - ohlcv.low[i];
            if body > 1e-12 && ((upper / body) > 2.0 || (lower / body) > 2.0) {
                inducement[i] = 1;
            }
        }

        if i >= 2 {
            if ohlcv.low[i] > ohlcv.high[i - 2] { fvg[i] = 1; }
            else if ohlcv.high[i] < ohlcv.low[i - 2] { fvg[i] = -1; }
        }

        if i >= 3 {
            let prev_low = ohlcv.low[(i-3)..i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let prev_high = ohlcv.high[(i-3)..i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            if ohlcv.low[i] < prev_low && ohlcv.close[i] > prev_low { liq[i] = 1; }
            else if ohlcv.high[i] > prev_high && ohlcv.close[i] < prev_high { liq[i] = -1; }
        }

        if i >= lookback {
            let prev_low = ohlcv.low[(i-lookback)..i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let prev_high = ohlcv.high[(i-lookback)..i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            if ohlcv.close[i] > prev_high { bos[i] = 1; }
            else if ohlcv.close[i] < prev_low { bos[i] = -1; }
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
                if (ohlcv.high[i] - ohlcv.high[j]).abs() <= tol { eqh[i] = -1; break; }
            }
            for j in lb..i {
                if (ohlcv.low[i] - ohlcv.low[j]).abs() <= tol { eql[i] = 1; break; }
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
                displacement[i] = if ohlcv.close[i] > ohlcv.open[i] { 1 } else if ohlcv.close[i] < ohlcv.open[i] { -1 } else { 0 };
            }
        }
    }

    (ob, fvg, liq, trend, premium, inducement, bos, choch, eqh, eql, displacement)
}

pub fn build_smc_arrays(frame: &FeatureFrame, ohlcv: &Ohlcv) -> SmcSignalTuple {
    let n = frame.data.nrows();
    let cols = detect_smc_columns(&frame.names);
    let (mut ob, mut fvg, mut liq, mut trend, mut premium, mut inducement, mut bos, mut choch, mut eqh, mut eql, mut displacement) = derive_smc_arrays(ohlcv);

    let apply_dir_col = |target: &mut Vec<i8>, col_opt: Option<usize>| {
        if let Some(col) = col_opt {
            if col < frame.data.ncols() {
                for (i, slot) in target.iter_mut().enumerate().take(n) {
                    *slot = quantize_dir(frame.data[(i, col)]);
                }
            }
        }
    };
    let apply_binary_col = |target: &mut Vec<i8>, col_opt: Option<usize>| {
        if let Some(col) = col_opt {
            if col < frame.data.ncols() {
                for (i, slot) in target.iter_mut().enumerate().take(n) {
                    *slot = quantize_binary(frame.data[(i, col)]);
                }
            }
        }
    };
    let apply_eqh_col = |target: &mut Vec<i8>, col_opt: Option<usize>| {
        if let Some(col) = col_opt {
            if col < frame.data.ncols() {
                for (i, slot) in target.iter_mut().enumerate().take(n) {
                    let v = frame.data[(i, col)];
                    let q = quantize_dir(v);
                    *slot = if q != 0 { q } else if quantize_binary(v) != 0 { -1 } else { 0 };
                }
            }
        }
    };
    let apply_eql_col = |target: &mut Vec<i8>, col_opt: Option<usize>| {
        if let Some(col) = col_opt {
            if col < frame.data.ncols() {
                for (i, slot) in target.iter_mut().enumerate().take(n) {
                    let v = frame.data[(i, col)];
                    let q = quantize_dir(v);
                    *slot = if q != 0 { q } else if quantize_binary(v) != 0 { 1 } else { 0 };
                }
            }
        }
    };
    let apply_dir_fill_zeros = |target: &mut Vec<i8>, col_opt: Option<usize>| {
        if let Some(col) = col_opt {
            if col < frame.data.ncols() {
                for (i, slot) in target.iter_mut().enumerate().take(n) {
                    if *slot == 0 {
                        *slot = quantize_dir(frame.data[(i, col)]);
                    }
                }
            }
        }
    };
    let apply_eq_levels = |target: &mut Vec<i8>, eqh_col: Option<usize>, eql_col: Option<usize>| {
        if let Some(col) = eqh_col {
            if col < frame.data.ncols() {
                for (i, slot) in target.iter_mut().enumerate().take(n) {
                    if quantize_binary(frame.data[(i, col)]) != 0 {
                        *slot = -1;
                    }
                }
            }
        }
        if let Some(col) = eql_col {
            if col < frame.data.ncols() {
                for (i, slot) in target.iter_mut().enumerate().take(n) {
                    if quantize_binary(frame.data[(i, col)]) != 0 {
                        *slot = 1;
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
            for (i, slot) in inducement.iter_mut().enumerate().take(n) {
                if quantize_dir(frame.data[(i, col)]) != 0 {
                    *slot = 1;
                }
            }
        }
    }
    for (disp, slot) in displacement.iter().zip(inducement.iter_mut()) {
        if *disp != 0 {
            *slot = 1;
        }
    }

    (ob, fvg, liq, trend, premium, inducement, bos, choch, eqh, eql, displacement)
}
