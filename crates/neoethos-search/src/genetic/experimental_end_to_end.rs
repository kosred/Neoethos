//! Genuine NeoEthos GA 400x5 END-TO-END GENETIC/SELECTION/ARCHIVE smoke.
//! Evaluation rows are explicitly DETERMINISTIC TEST DOUBLES, not market P&L.
//! The sealed market-evaluator route remains forbidden without receipts.
//! Invariant: the selected real fitness functions are called via apply_metrics,
//! and ALL FIVE COMPLETE generations plus final archive replay must run.
use super::*;
use super::super::runtime_overrides::{
    install_genetic_search_runtime_overrides, GeneticSearchRuntimeOverrides
};

#[test]
fn ga_400_x_5_complete_real_genetic_loop_with_test_evaluator() -> Result<()> {
    let names = (0..240).map(|n|format!("feature_{n}")).collect::<Vec<_>>();
    let policy=ExactSearchSizingPolicyV1::new(false,12)?;
    let mut knobs=GeneticSearchRuntimeOverrides::default();
    knobs.seed=Some(20261026);
    knobs.novelty_weight=0.20;
    knobs.novelty_neighbors=15;
    knobs.convergence_patience=0;
    install_genetic_search_runtime_overrides(knobs)
        .map_err(|_| anyhow!("GA smoke requires exclusive runtime overrides"))?;
    let goal=crate::scoring::RiskyGrowthGoal{
        start_balance:100.0,target_balance:50_000.0,horizon_days:180.0
    };
    let evaluation=EvaluationConfig {
        initial_equity:10_000.0,
        growth_objective:true,
        growth_goal:Some(goal),
        ..EvaluationConfig::default()
    };
    let started=Instant::now();
    let mut calls=Vec::new();
    let mut steps=Vec::new();
    let result=evolve_search_with_generation_evaluator_v1(
        &names,180.0,400,5,12,None,Some(evaluation),&policy,
        |genes,_config|{
            calls.push(genes.len());
            Ok(genes.iter().map(|gene|{
                // Deliberately artificial evaluator ONLY to run the
                // real native fitness/novelty/reproduction/archiving loop.
                let signature=gene.indices.iter().fold(0_usize,|a,&b|a.wrapping_add(b));
                let active=signature%5!=0;
                let edge=gene.weights.iter().take(8).copied().sum::<f64>().tanh();
                let trades=if active { 40.0+(signature%90) as f64 }else{0.0};
                let net=if active {edge*2_000.0}else{0.0};
                let dd=if active {0.01+edge.abs()*0.06}else{0.0};
                let sharpe=if active {0.5+edge}else{0.0};
                let wr=if active {(0.45+edge*0.10).clamp(0.0,1.0)}else{0.0};
                let pf=if active {1.0+edge*0.30}else{0.0};
                [net,sharpe,10_000.0+net,dd,wr,pf,net/trades.max(1.0),
                 0.5,trades,0.7,0.004]
            }).collect::<Vec<[f64;11]>>())
        },
        |generation,generations,best,stagnant,archive|{
            steps.push((generation,archive));
            eprintln!("GENETIC_SMOKE_PROGRESS generation={generation}/{generations} best_selection_score={best:.8} stagnant={stagnant} archive={archive}");
        },
        || false,
    )?;
    assert_eq!(steps.len(),5,"must complete all 5 generations");
    assert_eq!(steps.iter().map(|x|x.0).collect::<Vec<_>>(),vec![1,2,3,4,5]);
    assert!(calls.len()>=5,"final replay may add evaluation batches");
    assert_eq!(&calls[..5],&[400,400,400,400,400]);
    assert!(!result.genes.is_empty(),"final archive/population handoff cannot be empty");
    assert_eq!(result.genes.len(),result.metrics.len());
    let active=result.metrics.iter().filter(|m|m[8]>0.0).count();
    let positive=result.metrics.iter().filter(|m|m[8]>0.0 && m[0]>0.0).count();
    assert!(active>0,"sample must exercise valid fitness candidates");
    assert!(positive>0,"sample must exercise profitable archive admission");
    eprintln!("GENETIC_SMOKE_COMPLETED pop=400 generations=5 generation_eval_slots=2000 total_eval_calls={} final_candidates={} active={} positive_synthetic_metric={} secs={:.3} active_percentile={} real_market_evaluation=false broker_route_guard_preserved=true",
        calls.len(),result.genes.len(),active,positive,started.elapsed().as_secs_f64(),
        cfg!(feature="experimental-active-percentile"));
    Ok(())
}
