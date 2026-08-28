#[allow(unused_imports)]
use super::*;
use crate::genetic::{EvaluationConfig, Gene};

pub(super) fn metric_row_v1() -> [f64; 11] {
    [100.0, 1.2, 110.0, 0.05, 0.6, 1.5, 2.0, 0.8, 20.0, 0.9, 0.02]
}

pub(super) fn gene_for_metrics_v1(metrics: &[f64; 11], growth_objective: bool) -> Gene {
    Gene {
        indices: vec![0, 2],
        weights: vec![0.25, -0.5],
        long_threshold: 0.4,
        short_threshold: -0.4,
        fitness: if growth_objective {
            crate::scoring::ga_fitness_growth(metrics)
        } else {
            crate::scoring::ga_fitness(metrics)
        },
        sharpe_ratio: metrics[1],
        win_rate: metrics[4],
        max_drawdown: metrics[3],
        profit_factor: metrics[5],
        expectancy: metrics[6],
        trades_count: metrics[8].max(0.0) as usize,
        generation: 0,
        strategy_id: "gen0-strategy".to_owned(),
        tp_pips: 40.0,
        sl_pips: 20.0,
        slice_pass_rate: 1.0,
        consistency: metrics[9],
        stop_vol_mult: 0.0,
        ..Gene::default()
    }
}

pub(super) fn valid_population_v1(growth_objective: bool) -> (Vec<Gene>, Vec<[f64; 11]>) {
    let metrics = vec![metric_row_v1()];
    let genes = vec![gene_for_metrics_v1(&metrics[0], growth_objective)];
    (genes, metrics)
}

pub(super) fn valid_evaluation_config_v1(growth_objective: bool) -> EvaluationConfig {
    let mut config = EvaluationConfig::for_symbol("EURUSD", "USD", Some(1.1), Some(1.2), Some(7.0));
    config.growth_objective = growth_objective;
    config
}

#[test]
fn payload_rejects_every_nonfinite_gene_metric_and_gate_before_serde() {
    let (genes, metrics) = valid_population_v1(false);
    validate_population_payload_v1(&genes, &metrics, 0.5, 1, 5, 5, false)
        .expect("valid direct population payload");

    let gene_mutations: &[fn(&mut Gene, f64)] = &[
        |gene, invalid| gene.long_threshold = invalid,
        |gene, invalid| gene.short_threshold = invalid,
        |gene, invalid| gene.fitness = invalid,
        |gene, invalid| gene.sharpe_ratio = invalid,
        |gene, invalid| gene.win_rate = invalid,
        |gene, invalid| gene.max_drawdown = invalid,
        |gene, invalid| gene.profit_factor = invalid,
        |gene, invalid| gene.expectancy = invalid,
        |gene, invalid| gene.tp_pips = invalid,
        |gene, invalid| gene.sl_pips = invalid,
        |gene, invalid| gene.slice_pass_rate = invalid,
        |gene, invalid| gene.consistency = invalid,
        |gene, invalid| gene.stop_vol_mult = invalid,
        |gene, invalid| gene.weights[0] = invalid,
        |gene, invalid| gene.weights[1] = invalid,
    ];
    assert_eq!(gene_mutations.len(), 15);
    for mutate in gene_mutations {
        for nonfinite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut invalid = genes.clone();
            mutate(&mut invalid[0], nonfinite);
            assert!(
                validate_population_payload_v1(&invalid, &metrics, 0.5, 1, 5, 5, false).is_err()
            );
        }
    }
    for index in 0..11 {
        let mut invalid = metrics.clone();
        invalid[0][index] = if index % 2 == 0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
        assert!(validate_population_payload_v1(&genes, &invalid, 0.5, 1, 5, 5, false).is_err());
    }
    for gate in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(validate_population_payload_v1(&genes, &metrics, gate, 1, 5, 5, false).is_err());
    }
}

#[test]
fn payload_requires_generation_zero_strict_terms_and_exact_cardinality() {
    let (genes, metrics) = valid_population_v1(false);
    let invalid_genes: Vec<Vec<Gene>> = [
        {
            let mut value = genes.clone();
            value[0].generation = 1;
            value
        },
        {
            let mut value = genes.clone();
            value[0].indices = vec![2, 1];
            value
        },
        {
            let mut value = genes.clone();
            value[0].indices = vec![1, 1];
            value
        },
        {
            let mut value = genes.clone();
            value[0].indices = vec![0, 5];
            value
        },
        {
            let mut value = genes.clone();
            value[0].weights.pop();
            value
        },
        {
            let mut value = genes.clone();
            value[0].indices = vec![0, 1, 2, 3, 4, 5];
            value[0].weights = vec![1.0; 6];
            value
        },
    ]
    .into_iter()
    .collect();
    for invalid in invalid_genes {
        assert!(validate_population_payload_v1(&invalid, &metrics, 0.5, 1, 5, 5, false).is_err());
    }
    assert!(validate_population_payload_v1(&[], &metrics, 0.5, 1, 5, 5, false).is_err());
    assert!(validate_population_payload_v1(&genes, &[], 0.5, 1, 5, 5, false).is_err());
    assert!(validate_population_payload_v1(&genes, &metrics, 0.5, 2, 5, 5, false).is_err());
}

#[test]
fn canonical_genome_validation_is_strict_without_overconstraining_population_order() {
    let (genes, metrics) = valid_population_v1(false);
    let mutations: &[fn(&mut Gene)] = &[
        |gene| gene.indices.clear(),
        |gene| gene.weights.clear(),
        |gene| gene.weights[0] = 1.0e-6,
        |gene| gene.weights[0] = 5.000_001,
        |gene| gene.weights[0] = -5.000_001,
        |gene| gene.long_threshold = gene.short_threshold,
        |gene| gene.sl_pips = 0.0,
        |gene| gene.tp_pips = -1.0,
        |gene| gene.stop_vol_mult = -1.0,
    ];
    for mutate in mutations {
        let mut invalid = genes.clone();
        mutate(&mut invalid[0]);
        assert!(validate_population_payload_v1(&invalid, &metrics, 0.5, 1, 5, 5, false).is_err());
    }

    for weight in [-5.0, -0.25, 1.000_001e-6, 5.0] {
        let mut boundary = genes.clone();
        boundary[0].weights[0] = weight;
        boundary[0].tp_pips = 50.0;
        validate_population_payload_v1(&boundary, &metrics, 0.5, 1, 5, 5, false)
            .expect("canonical negative/boundary weight and seed TP are valid");
    }

    let duplicate_genes = vec![genes[0].clone(), genes[0].clone()];
    let duplicate_metrics = vec![metrics[0], metrics[0]];
    validate_population_payload_v1(&duplicate_genes, &duplicate_metrics, 0.5, 2, 5, 5, false)
        .expect("duplicate genomes and strategy IDs preserve evaluated order");
}

#[test]
fn trade_count_metric_must_be_finite_nonnegative_and_integral_before_cast() {
    let (genes, metrics) = valid_population_v1(false);
    for invalid_trade_count in [-1.0, 1.5, f64::MAX, f64::NAN, f64::INFINITY] {
        let mut invalid = metrics.clone();
        invalid[0][8] = invalid_trade_count;
        assert!(validate_population_payload_v1(&genes, &invalid, 0.5, 1, 5, 5, false).is_err());
    }
}

#[test]
fn derived_gene_fields_are_bit_exact_for_both_scoring_modes() {
    for growth_objective in [false, true] {
        let (genes, metrics) = valid_population_v1(growth_objective);
        validate_population_payload_v1(&genes, &metrics, 0.5, 1, 5, 5, growth_objective)
            .expect("derived fields match canonical apply_metrics");
        let mut wrong = genes.clone();
        wrong[0].fitness = if growth_objective {
            crate::scoring::ga_fitness(&metrics[0])
        } else {
            crate::scoring::ga_fitness_growth(&metrics[0])
        };
        assert!(
            validate_population_payload_v1(&wrong, &metrics, 0.5, 1, 5, 5, growth_objective,)
                .is_err()
        );
        for mutate in [
            |gene: &mut Gene| gene.sharpe_ratio = 9.0,
            |gene: &mut Gene| gene.max_drawdown = 9.0,
            |gene: &mut Gene| gene.win_rate = 9.0,
            |gene: &mut Gene| gene.profit_factor = 9.0,
            |gene: &mut Gene| gene.expectancy = 9.0,
            |gene: &mut Gene| gene.trades_count += 1,
            |gene: &mut Gene| gene.consistency = 9.0,
            |gene: &mut Gene| gene.slice_pass_rate = 0.5,
        ] {
            let mut wrong = genes.clone();
            mutate(&mut wrong[0]);
            assert!(
                validate_population_payload_v1(&wrong, &metrics, 0.5, 1, 5, 5, growth_objective,)
                    .is_err()
            );
        }
    }
}

#[test]
fn evaluation_snapshot_binds_versioned_objective_and_request_mode() {
    let mut config = valid_evaluation_config_v1(false);
    let standard =
        CanonicalNativeGenerationZeroEvaluationEvidenceV1::checked_from_evaluation_config_v1(
            &config,
            crate::discovery::DiscoveryMode::PropFirm,
        )
        .unwrap();
    let repeated =
        CanonicalNativeGenerationZeroEvaluationEvidenceV1::checked_from_evaluation_config_v1(
            &config,
            crate::discovery::DiscoveryMode::PropFirm,
        )
        .unwrap();
    assert_eq!(standard.identity_sha256(), repeated.identity_sha256());
    assert!(!standard.growth_objective());
    assert_eq!(
        standard.scoring_objective(),
        CanonicalNativeGenerationZeroScoringObjectiveV1::PropConsistencyV4
    );
    assert!(
        CanonicalNativeGenerationZeroEvaluationEvidenceV1::checked_from_evaluation_config_v1(
            &config,
            crate::discovery::DiscoveryMode::Risky,
        )
        .is_err()
    );

    let mutations: &[fn(&mut EvaluationConfig)] = &[
        |value| value.symbol = "GBPUSD".to_owned(),
        |value| value.account_currency = "EUR".to_owned(),
        |value| value.max_hold_bars += 1,
        |value| value.trailing_enabled = !value.trailing_enabled,
        |value| {
            value.trailing_atr_multiplier =
                f64::from_bits(value.trailing_atr_multiplier.to_bits() + 1);
        },
        |value| {
            value.trailing_be_trigger_r = f64::from_bits(value.trailing_be_trigger_r.to_bits() + 1);
        },
        |value| {
            value.trailing_min_lock_pips =
                f64::from_bits(value.trailing_min_lock_pips.to_bits() + 1);
        },
        |value| value.pip_value = f64::from_bits(value.pip_value.to_bits() + 1),
        |value| value.spread_pips = f64::from_bits(value.spread_pips.to_bits() + 1),
        |value| {
            value.commission_per_trade = f64::from_bits(value.commission_per_trade.to_bits() + 1);
        },
        |value| {
            value.pip_value_per_lot = f64::from_bits(value.pip_value_per_lot.to_bits() + 1);
        },
        |value| {
            value.swap_long_pips_per_day =
                f64::from_bits(value.swap_long_pips_per_day.to_bits() + 1);
        },
        |value| {
            value.swap_short_pips_per_day =
                f64::from_bits(value.swap_short_pips_per_day.to_bits() + 1);
        },
        |value| {
            value.pnl_conversion_fee_rate =
                f64::from_bits(value.pnl_conversion_fee_rate.to_bits() + 1);
        },
        |value| {
            value.smc_gate_threshold = f64::from_bits(value.smc_gate_threshold.to_bits() + 1);
        },
        |value| value.smc_weight_ob = f64::from_bits(value.smc_weight_ob.to_bits() + 1),
        |value| value.smc_weight_fvg = f64::from_bits(value.smc_weight_fvg.to_bits() + 1),
        |value| value.smc_weight_liq = f64::from_bits(value.smc_weight_liq.to_bits() + 1),
        |value| value.smc_weight_mtf = f64::from_bits(value.smc_weight_mtf.to_bits() + 1),
        |value| {
            value.smc_weight_premium = f64::from_bits(value.smc_weight_premium.to_bits() + 1);
        },
        |value| {
            value.smc_weight_inducement = f64::from_bits(value.smc_weight_inducement.to_bits() + 1);
        },
        |value| value.smc_weight_bos = f64::from_bits(value.smc_weight_bos.to_bits() + 1),
        |value| value.smc_weight_choch = f64::from_bits(value.smc_weight_choch.to_bits() + 1),
        |value| value.smc_weight_eqh = f64::from_bits(value.smc_weight_eqh.to_bits() + 1),
        |value| value.smc_weight_eql = f64::from_bits(value.smc_weight_eql.to_bits() + 1),
        |value| {
            value.smc_weight_displacement =
                f64::from_bits(value.smc_weight_displacement.to_bits() + 1);
        },
    ];
    assert_eq!(mutations.len(), 26);
    for mutate in mutations {
        let mut changed = config.clone();
        mutate(&mut changed);
        let changed =
            CanonicalNativeGenerationZeroEvaluationEvidenceV1::checked_from_evaluation_config_v1(
                &changed,
                crate::discovery::DiscoveryMode::PropFirm,
            )
            .unwrap();
        assert_ne!(standard.identity_sha256(), changed.identity_sha256());
    }

    config.growth_objective = true;
    let growth =
        CanonicalNativeGenerationZeroEvaluationEvidenceV1::checked_from_evaluation_config_v1(
            &config,
            crate::discovery::DiscoveryMode::Risky,
        )
        .unwrap();
    assert!(growth.growth_objective());
    assert_eq!(
        growth.scoring_objective(),
        CanonicalNativeGenerationZeroScoringObjectiveV1::RiskyKellyGrowthV5
    );
    assert_ne!(standard.identity_sha256(), growth.identity_sha256());
}

#[test]
fn evaluation_snapshot_census_and_all_f64_inputs_are_fail_closed() {
    let source = include_str!("genetic/strategy_gene.rs");
    let body = source
        .split_once("pub struct EvaluationConfig {")
        .unwrap()
        .1
        .split_once("\n}")
        .unwrap()
        .0;
    let field_lines: Vec<_> = body
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("pub ") && line.ends_with(','))
        .collect();
    assert_eq!(field_lines.len(), 27);
    assert_eq!(body.matches(": String,").count(), 2);
    assert_eq!(body.matches(": usize,").count(), 1);
    assert_eq!(body.matches(": bool,").count(), 2);
    assert_eq!(body.matches(": f64,").count(), 22);

    let setters: &[fn(&mut EvaluationConfig, f64)] = &[
        |value, invalid| value.trailing_atr_multiplier = invalid,
        |value, invalid| value.trailing_be_trigger_r = invalid,
        |value, invalid| value.trailing_min_lock_pips = invalid,
        |value, invalid| value.pip_value = invalid,
        |value, invalid| value.spread_pips = invalid,
        |value, invalid| value.commission_per_trade = invalid,
        |value, invalid| value.pip_value_per_lot = invalid,
        |value, invalid| value.swap_long_pips_per_day = invalid,
        |value, invalid| value.swap_short_pips_per_day = invalid,
        |value, invalid| value.pnl_conversion_fee_rate = invalid,
        |value, invalid| value.smc_gate_threshold = invalid,
        |value, invalid| value.smc_weight_ob = invalid,
        |value, invalid| value.smc_weight_fvg = invalid,
        |value, invalid| value.smc_weight_liq = invalid,
        |value, invalid| value.smc_weight_mtf = invalid,
        |value, invalid| value.smc_weight_premium = invalid,
        |value, invalid| value.smc_weight_inducement = invalid,
        |value, invalid| value.smc_weight_bos = invalid,
        |value, invalid| value.smc_weight_choch = invalid,
        |value, invalid| value.smc_weight_eqh = invalid,
        |value, invalid| value.smc_weight_eql = invalid,
        |value, invalid| value.smc_weight_displacement = invalid,
    ];
    assert_eq!(setters.len(), 22);
    for set in setters {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut config =
                EvaluationConfig::for_symbol("EURUSD", "USD", Some(1.1), Some(1.2), Some(7.0));
            set(&mut config, invalid);
            assert!(
                CanonicalNativeGenerationZeroEvaluationEvidenceV1::checked_from_evaluation_config_v1(
                    &config,
                    crate::discovery::DiscoveryMode::PropFirm,
                )
                .is_err()
            );
        }
    }
}
