use super::strategy_gene::Gene;

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

// `DiversityArchiveConfig` + `select_diverse_archive` + `archive_quality_score` —
// DELETED 2026-05-26 (operator directive: dual-mode product). Three structs/fns
// with zero callers outside their own tests. Diversity gating now happens at
// two narrower seams:
//   - `seen_signature_memory` (genetic::evolution_math) deduplicates the
//     working population by gene signature.
//   - `correlation` pruning at `discovery.rs::finalize_candidates_with_progress`
//     enforces portfolio-level diversity on the FINAL strategies.
// The `DiversityKey` + `diversity_key` + `smc_mask` helpers below are kept —
// they remain useful for telemetry/funnel diagnostics even though no live
// archive selection consumes them right now.

use neoethos_core::utils::finite_or;

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
    fn diversity_key_distinguishes_indicator_bins() {
        // Sanity: two genes that differ in indicator count produce different keys.
        let one = test_gene("one", true, 40.0, 20.0);
        let mut two = test_gene("two", true, 40.0, 20.0);
        two.indices.push(2);
        two.weights.push(0.25);
        let metrics_a = test_metrics(1_000.0, 1.5, 0.03, 50.0);
        let metrics_b = test_metrics(1_000.0, 1.5, 0.03, 50.0);
        assert_ne!(diversity_key(&one, &metrics_a), diversity_key(&two, &metrics_b));
    }
}
