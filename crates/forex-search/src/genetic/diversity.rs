use super::strategy_gene::Gene;
use std::collections::HashMap;

pub type EvalMetrics = [f64; 11];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DiversityKey {
    pub indicator_count_bin: u8,
    pub smc_mask: u16,
    pub rr_bin: i16,
    pub trade_bin: u8,
    pub pf_bin: u8,
    pub dd_bin: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct DiversityArchiveConfig {
    pub max_total: usize,
    pub per_bucket_cap: usize,
    pub min_archive_score: f64,
}

impl DiversityArchiveConfig {
    pub fn from_env(default_max_total: usize) -> Self {
        let max_total = std::env::var("FOREX_BOT_PROP_DIVERSE_ARCHIVE_CAP")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(default_max_total.max(1));
        let per_bucket_cap = std::env::var("FOREX_BOT_PROP_DIVERSE_BUCKET_CAP")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(8);
        let min_archive_score = std::env::var("FOREX_BOT_PROP_DIVERSE_MIN_SCORE")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite())
            .unwrap_or(f64::NEG_INFINITY);

        Self {
            max_total,
            per_bucket_cap,
            min_archive_score,
        }
    }
}

fn finite_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() { value } else { fallback }
}

fn clamp_bin(value: f64, step: f64, max_bin: u8) -> u8 {
    if !value.is_finite() || step <= 0.0 {
        return 0;
    }
    ((value / step).floor() as i64).clamp(0, max_bin as i64) as u8
}

pub fn smc_mask(gene: &Gene) -> u16 {
    let flags = [
        gene.use_ob,
        gene.use_fvg,
        gene.use_liq_sweep,
        gene.mtf_confirmation,
        gene.use_premium_discount,
        gene.use_inducement,
        gene.use_bos,
        gene.use_choch,
        gene.use_eqh,
        gene.use_eql,
        gene.use_displacement,
    ];

    flags.iter().enumerate().fold(
        0_u16,
        |mask, (idx, flag)| {
            if *flag { mask | (1_u16 << idx) } else { mask }
        },
    )
}

pub fn diversity_key(gene: &Gene, metrics: &EvalMetrics) -> DiversityKey {
    let rr = if gene.sl_pips.is_finite() && gene.sl_pips > 0.0 {
        gene.tp_pips / gene.sl_pips
    } else {
        0.0
    };
    let trade_count = finite_or(metrics[8], 0.0).max(0.0);
    let profit_factor = finite_or(metrics[5], 0.0).max(0.0);
    let max_dd = finite_or(metrics[3], 1.0).max(0.0);

    DiversityKey {
        indicator_count_bin: gene.indices.len().clamp(1, u8::MAX as usize) as u8,
        smc_mask: smc_mask(gene),
        rr_bin: (rr * 10.0).round().clamp(-100.0, 100.0) as i16,
        trade_bin: clamp_bin(trade_count, 50.0, 20),
        pf_bin: clamp_bin(profit_factor, 0.25, 40),
        dd_bin: clamp_bin(max_dd, 0.01, 50),
    }
}

pub fn archive_quality_score(metrics: &EvalMetrics) -> f64 {
    let net = finite_or(metrics[0], 0.0);
    let sharpe = finite_or(metrics[1], 0.0);
    let max_dd = finite_or(metrics[3], 1.0).max(0.0);
    let win_rate = finite_or(metrics[4], 0.0).clamp(0.0, 1.0);
    let profit_factor = finite_or(metrics[5], 0.0).max(0.0);
    let expectancy = finite_or(metrics[6], 0.0);
    let trades = finite_or(metrics[8], 0.0).max(0.0);
    let consistency = finite_or(metrics[9], 0.0).clamp(0.0, 1.0);

    let trade_confidence = (trades.sqrt() / 12.0).min(1.0);
    let net_component = (net / 10_000.0).clamp(-5.0, 5.0) * 0.25;
    let sharpe_component = sharpe.clamp(-3.0, 5.0) * 0.25 * trade_confidence;
    let pf_component = ((profit_factor - 1.0) * 0.75).clamp(-2.0, 3.0) * 0.20;
    let consistency_component = consistency * 0.20;
    let win_component = ((win_rate - 0.45) * 2.0).clamp(0.0, 0.8) * 0.10;
    let expectancy_component = (expectancy / 100.0).clamp(-2.0, 2.0) * 0.10;
    let dd_penalty = (max_dd * 12.0).min(4.0);

    net_component
        + sharpe_component
        + pf_component
        + consistency_component
        + win_component
        + expectancy_component
        - dd_penalty
}

pub fn select_diverse_archive(
    mut archive: Vec<(Gene, EvalMetrics, usize)>,
    config: DiversityArchiveConfig,
) -> Vec<(Gene, EvalMetrics, usize)> {
    archive.sort_by(|a, b| {
        archive_quality_score(&b.1)
            .partial_cmp(&archive_quality_score(&a.1))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2.cmp(&b.2))
    });

    let mut bucket_counts: HashMap<DiversityKey, usize> = HashMap::new();
    let mut selected = Vec::with_capacity(config.max_total.min(archive.len()));

    for (gene, metrics, seq) in archive {
        let score = archive_quality_score(&metrics);
        if score < config.min_archive_score {
            continue;
        }
        let key = diversity_key(&gene, &metrics);
        let count = bucket_counts.entry(key).or_insert(0);
        if *count >= config.per_bucket_cap {
            continue;
        }
        *count += 1;
        selected.push((gene, metrics, seq));
        if selected.len() >= config.max_total {
            break;
        }
    }

    selected.sort_by(|a, b| {
        b.1[0]
            .partial_cmp(&a.1[0])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2.cmp(&b.2))
    });
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_metrics(net: f64, pf: f64, dd: f64, trades: f64) -> EvalMetrics {
        [
            net,
            1.0,
            100_000.0 + net,
            dd,
            0.55,
            pf,
            net / trades.max(1.0),
            0.0,
            trades,
            0.6,
            0.0,
        ]
    }

    fn test_gene(id: &str, use_ob: bool, tp: f64, sl: f64) -> Gene {
        Gene {
            strategy_id: id.to_string(),
            indices: vec![0, 1],
            weights: vec![1.0, 0.5],
            long_threshold: 0.25,
            short_threshold: -0.25,
            use_ob,
            tp_pips: tp,
            sl_pips: sl,
            ..Default::default()
        }
    }

    #[test]
    fn diversity_selection_limits_bucket_dominance() {
        let mut archive = Vec::new();
        for i in 0..10 {
            archive.push((
                test_gene(&format!("same_{i}"), true, 40.0, 20.0),
                test_metrics(10_000.0 - i as f64, 1.8, 0.04, 120.0),
                i,
            ));
        }
        archive.push((
            test_gene("different", false, 25.0, 20.0),
            test_metrics(5_000.0, 1.5, 0.03, 90.0),
            99,
        ));

        let selected = select_diverse_archive(
            archive,
            DiversityArchiveConfig {
                max_total: 20,
                per_bucket_cap: 3,
                min_archive_score: f64::NEG_INFINITY,
            },
        );

        assert_eq!(selected.len(), 4);
        assert!(
            selected
                .iter()
                .any(|(gene, _, _)| gene.strategy_id == "different")
        );
    }
}
