//! Decode terminal CUDA control data into the existing Search candidate union.
//! No feature matrix is downloaded and no backtest or validation is rerun here.

use anyhow::{Context, Result, ensure};
use neoethos_gpu_cuda::resident_archive_output_v3::{
    ResidentArchiveCandidateV3, ResidentSearchTerminalCandidatesV3,
};

use crate::genetic::{EvaluationConfig, Gene, SearchResult, apply_metrics};

struct TerminalGeneViewV3<'a> {
    identity: u64,
    generation: u32,
    smc_flags: u32,
    long_threshold: f64,
    short_threshold: f64,
    target_pips: f64,
    stop_pips: f64,
    stop_vol_multiplier: f64,
    indices: &'a [u64],
    weights: &'a [f64],
    metrics: &'a [f64; 11],
}

impl<'a> From<&'a ResidentArchiveCandidateV3<'a>> for TerminalGeneViewV3<'a> {
    fn from(candidate: &'a ResidentArchiveCandidateV3<'a>) -> Self {
        Self {
            identity: candidate.gene_identity(),
            generation: candidate.generation(),
            smc_flags: candidate.smc_flags(),
            long_threshold: candidate.long_threshold(),
            short_threshold: candidate.short_threshold(),
            target_pips: candidate.target_pips(),
            stop_pips: candidate.stop_pips(),
            stop_vol_multiplier: candidate.stop_vol_multiplier(),
            indices: candidate.indices(),
            weights: candidate.weights(),
            metrics: &candidate.metric_row().values,
        }
    }
}

fn decode_terminal_gene_v3(
    run_identity: u64,
    candidate: TerminalGeneViewV3<'_>,
    config: &EvaluationConfig,
    span_days: f64,
) -> Result<(Gene, [f64; 11])> {
    ensure!(
        candidate.indices.len() == candidate.weights.len()
            && !candidate.indices.is_empty()
            && candidate.smc_flags & !0x7ff == 0,
        "terminal native gene has an invalid term or SMC extent"
    );
    let trades = candidate.metrics[8];
    ensure!(
        trades.is_finite() && trades >= 0.0 && trades.fract() == 0.0 && trades < usize::MAX as f64,
        "terminal native trade count cannot be represented exactly by Search"
    );
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(candidate.indices.len())
        .context("reserve terminal native gene indices")?;
    for &index in candidate.indices {
        indices.push(usize::try_from(index).context("terminal feature index exceeds usize")?);
    }
    let mut weights = Vec::new();
    weights
        .try_reserve_exact(candidate.weights.len())
        .context("reserve terminal native gene weights")?;
    weights.extend_from_slice(candidate.weights);
    let flag = |bit: u32| candidate.smc_flags & (1_u32 << bit) != 0;
    let mut gene = Gene {
        indices,
        weights,
        long_threshold: candidate.long_threshold,
        short_threshold: candidate.short_threshold,
        tp_pips: candidate.target_pips,
        sl_pips: candidate.stop_pips,
        stop_vol_mult: candidate.stop_vol_multiplier,
        generation: usize::try_from(candidate.generation)
            .context("terminal native generation exceeds usize")?,
        strategy_id: format!("native_{run_identity:016x}_{:016x}", candidate.identity),
        use_ob: flag(0),
        use_fvg: flag(1),
        use_liq_sweep: flag(2),
        mtf_confirmation: flag(3),
        use_premium_discount: flag(4),
        use_inducement: flag(5),
        use_bos: flag(6),
        use_choch: flag(7),
        use_eqh: flag(8),
        use_eql: flag(9),
        use_displacement: flag(10),
        ..Gene::default()
    };
    // The GPU has already simulated this row. This shared function merely
    // fills the same derived Gene fields as the CPU path. Never normalize or
    // repair a transported genome, and never substitute a missing metric.
    apply_metrics(
        std::slice::from_mut(&mut gene),
        std::slice::from_ref(candidate.metrics),
        config,
        span_days,
    );
    Ok((gene, *candidate.metrics))
}

pub(super) fn search_result_from_terminal_v3(
    terminal: ResidentSearchTerminalCandidatesV3,
    config: &EvaluationConfig,
    span_days: f64,
    effective_smc_gate_threshold: f64,
    evaluation_slots: u64,
) -> Result<SearchResult> {
    ensure!(
        span_days.is_finite() && span_days > 0.0,
        "terminal Search conversion requires the actual positive Stage1 span"
    );
    ensure!(
        effective_smc_gate_threshold.is_finite() && effective_smc_gate_threshold >= 0.0,
        "terminal Search conversion requires the measured final SMC gate"
    );
    ensure!(
        config.initial_equity.is_finite() && config.initial_equity > 0.0,
        "terminal Search conversion requires actual positive account equity"
    );
    if let Some(goal) = config.growth_goal {
        ensure!(
            config.growth_objective,
            "terminal goal is detached from Risky mode"
        );
        goal.validate().map_err(anyhow::Error::msg)?;
    }
    let (archive, population) = terminal.into_parts();
    archive
        .len()
        .checked_add(population.len())
        .context("terminal candidate union overflow")?;
    let mut archived = Vec::new();
    archived
        .try_reserve_exact(archive.len())
        .context("reserve complete native archive union")?;
    for (sequence, candidate) in archive.candidates().enumerate() {
        let (gene, metrics) = decode_terminal_gene_v3(
            archive.run_identity(),
            (&candidate).into(),
            config,
            span_days,
        )?;
        archived.push((gene, metrics, sequence));
    }
    let mut scored = Vec::new();
    scored
        .try_reserve_exact(population.len())
        .context("reserve complete evaluated native population")?;
    for (ordinal, candidate) in population.candidates().enumerate() {
        let (gene, metrics) = decode_terminal_gene_v3(
            population.run_identity(),
            (&candidate).into(),
            config,
            span_days,
        )?;
        scored.push((gene.fitness, ordinal, gene, metrics));
    }
    Ok(crate::genetic::search_engine::finish_evaluated_generation(
        archived,
        scored,
        effective_smc_gate_threshold,
        evaluation_slots,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view<'a>(id: u64, metrics: &'a [f64; 11]) -> TerminalGeneViewV3<'a> {
        TerminalGeneViewV3 {
            identity: id,
            generation: 2,
            smc_flags: 0x555,
            long_threshold: 0.75,
            short_threshold: -0.25,
            target_pips: 0.75,
            stop_pips: 0.25,
            stop_vol_multiplier: 1.5,
            indices: &[1923, 3],
            weights: &[-0.0, -0.125],
            metrics,
        }
    }

    fn config() -> EvaluationConfig {
        EvaluationConfig {
            initial_equity: 12_345.25,
            growth_objective: true,
            growth_goal: Some(crate::scoring::RiskyGrowthGoal {
                start_balance: 100.0,
                target_balance: 50_000.0,
                horizon_days: 180.0,
            }),
            ..EvaluationConfig::default()
        }
    }

    #[test]
    fn terminal_gene_preserves_genome_units_flags_and_shared_goal_metrics() {
        let metrics = [
            125.5, 1.2, 12_470.75, 0.1, 0.75, 2.0, 12.55, 0.5, 10.0, 0.6, 0.2,
        ];
        let config = config();
        let (gene, returned) =
            decode_terminal_gene_v3(77, view(900, &metrics), &config, 3.125).unwrap();
        assert_eq!(gene.strategy_id, "native_000000000000004d_0000000000000384");
        assert_eq!(gene.generation, 2);
        assert_eq!(gene.indices, [1923, 3]);
        assert_eq!(
            gene.weights.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            [-0.0f64, -0.125].map(f64::to_bits)
        );
        assert_eq!(
            (gene.sl_pips, gene.tp_pips, gene.stop_vol_mult),
            (0.25, 0.75, 1.5)
        );
        assert_eq!((gene.long_threshold, gene.short_threshold), (0.75, -0.25));
        assert_eq!(
            [
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
                gene.use_displacement
            ],
            [
                true, false, true, false, true, false, true, false, true, false, true
            ]
        );
        assert_eq!(returned.map(f64::to_bits), metrics.map(f64::to_bits));
        assert_eq!(gene.trades_count, 10);
        assert_eq!(
            gene.fitness.to_bits(),
            crate::scoring::ga_fitness_goal(
                &metrics,
                config.initial_equity,
                3.125,
                config.growth_goal.unwrap()
            )
            .to_bits()
        );
        assert_eq!(
            (
                gene.sharpe_ratio,
                gene.max_drawdown,
                gene.win_rate,
                gene.profit_factor,
                gene.expectancy,
                gene.consistency
            ),
            (1.2, 0.1, 0.75, 2.0, 12.55, 0.6)
        );
    }

    #[test]
    fn terminal_union_uses_last_metrics_and_keeps_nonarchive_and_rejected_candidates() {
        let config = config();
        let old = [1.0, 1.0, 12_346.25, 0.1, 0.5, 1.1, 1.0, 0.5, 1.0, 0.5, 0.1];
        let current = [2.0, 1.5, 12_347.25, 0.1, 0.5, 1.2, 2.0, 0.5, 1.0, 0.5, 0.1];
        let rejected = [
            -13_000.0,
            f64::NEG_INFINITY,
            12_345.25,
            1.2,
            0.0,
            0.0,
            -13_000.0,
            0.0,
            1.0,
            0.0,
            1.2,
        ];
        let (a, am) = decode_terminal_gene_v3(77, view(1, &old), &config, 3.125).unwrap();
        let (b, bm) = decode_terminal_gene_v3(77, view(2, &current), &config, 3.125).unwrap();
        let mut rejected_view = view(3, &rejected);
        rejected_view.indices = &[3, 1923];
        let (c, cm) = decode_terminal_gene_v3(77, rejected_view, &config, 3.125).unwrap();
        assert_eq!(c.fitness, f64::NEG_INFINITY);
        let result = crate::genetic::search_engine::finish_evaluated_generation(
            vec![(a, am, 0)],
            vec![(b.fitness, 0, b, bm), (c.fitness, 1, c, cm)],
            0.35,
            6,
        );
        assert_eq!(result.genes.len(), 2);
        assert_eq!(
            result.genes[0].strategy_id,
            "native_000000000000004d_0000000000000002"
        );
        assert_eq!(result.metrics[0], current);
        assert_eq!(result.metrics[1][1], f64::NEG_INFINITY);
        assert_eq!(result.effective_smc_gate_threshold, 0.35);
    }

    #[test]
    fn terminal_gene_refuses_nonintegral_or_unrepresentable_trade_counts() {
        for count in [-1.0, 0.5, f64::NAN, f64::INFINITY, usize::MAX as f64] {
            let mut metrics = [0.0; 11];
            metrics[8] = count;
            assert!(decode_terminal_gene_v3(77, view(1, &metrics), &config(), 1.0).is_err());
        }
    }
}
