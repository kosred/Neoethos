//! Isolated genuine-host-GA smoke: actual cTrader EURUSD M1 fixture,
//! actual Rust indicator-to-signal-to-backtest evaluator, 400 x 5 GA.
//! NO full discovery funnel, no external broker receipt, no final OOS.
//! Run alone with feature experimental-active-percentile.
use super::*;
use super::super::runtime_overrides::{
    install_genetic_search_runtime_overrides, GeneticSearchRuntimeOverrides
};

#[test]
fn ga_400_x_5_entire_real_rust_generation_loop_with_captured_ctrader_bars() -> Result<()> {
    let features = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
    let prices = neoethos_data::test_fixtures::ctrader_sample_ohlcv();
    assert_eq!(features.timestamps.len(), prices.close.len());
    let mut knobs=GeneticSearchRuntimeOverrides::default();
    knobs.seed=Some(20261026);
    knobs.novelty_weight=0.20;
    knobs.novelty_neighbors=15;
    knobs.convergence_patience=0;
    install_genetic_search_runtime_overrides(knobs)
        .map_err(|_| anyhow!("GA smoke test requires exclusive runtime overrides"))?;
    // Explicit *research fixture* economic settings: no guessed defaults.
    // A broker-authoritative run MUST get actual swap/fees/price from its
    // verified receipt; the 100 captured-bar test only verifies execution.
    let mut evaluation=EvaluationConfig::for_symbol(
        "EURUSD","USD",Some(1.10),Some(1.2),Some(7.0)
    );
    evaluation.initial_equity=10_000.0;
    evaluation.spread_pips=1.2;
    evaluation.commission_per_trade=7.0;
    evaluation.pip_value_per_lot=10.0;
    evaluation.pip_value=0.0001;
    evaluation.swap_long_pips_per_day=0.0;
    evaluation.swap_short_pips_per_day=0.0;
    evaluation.pnl_conversion_fee_rate=0.0;
    evaluation.max_hold_bars=10;
    evaluation.growth_objective=true;
    evaluation.growth_goal=Some(crate::scoring::RiskyGrowthGoal{
        start_balance:100.0,target_balance:50_000.0,horizon_days:180.0
    });
    let started=Instant::now();
    let mut steps=Vec::new();
    let output=evolve_search_with_progress_and_limits(
        &features, &prices, 400,5,2,None,Some(evaluation),
        |generation,gens,best,stagnation,archive|{
            assert_eq!(gens,5);
            steps.push((generation,best,stagnation,archive));
            eprintln!("SMOKE_GA_PROGRESS generation={generation}/{gens} score={best:.8} stagnant={stagnation} archive={archive}");
        },
    )?;
    assert_eq!(steps.len(),5,"all 5 real GA generations must execute");
    assert_eq!(steps.iter().map(|x|x.0).collect::<Vec<_>>(),vec![1,2,3,4,5]);
    assert_eq!(output.genes.len(),output.metrics.len());
    assert!(!output.genes.is_empty(),"handoff must retain evaluated candidates");
    assert!(output.metrics.iter().all(|m|m.len()==11));
    let winning=output.metrics.iter().filter(|m|m[0].is_finite()&&m[0]>0.0&&m[8]>0.0).count();
    let active=output.metrics.iter().filter(|m|m[8]>0.0).count();
    eprintln!("SMOKE_GA_COMPLETED population=400 generations=5 evaluation_slots_at_least=2000 final_rows={} active={} positive_net={} cTrader_fixture_rows={} secs={:.3} active_percentile={} broker_authoritative_oos=false",
        output.genes.len(),active,winning,prices.close.len(),started.elapsed().as_secs_f64(),cfg!(feature="experimental-active-percentile"));
    Ok(())
}
