//! Experimental exact-scoring sensitivity of *actual* NeoEthos GA objectives.
//!
//! This does not change scoring, use broker data, discover strategies, or test
//! final OOS. Counterfactual metrics change one field at a time specifically to
//! measure derivatives; they are NOT valid executed trading ledgers.
use neoethos_gpu_contracts::resident_search_scoring_v2::{
    RiskyGrowthGoal, score_prop_firm_ga_fitness_v4,
    score_risky_ga_fitness_growth_v5, score_risky_ga_fitness_goal_v6,
};

fn fixture() -> [f64; 11] {
    // Shared reference for isolated input sensitivities, not a real backtest.
    // net, Sharpe, peak, max_dd, wr, PF, exp, monthly hit, trades,
    // consistency, max_daily_dd.
    [20.0, 1.2, 0.0, 0.02, 0.48, 1.20, 0.2, 0.50, 100.0, 0.65, 0.005]
}
fn variant(base: [f64;11], slot:usize, value:f64)->[f64;11] {
    let mut m=base;
    m[slot]=value;
    m
}
fn print_scores(label: &str, m: &[f64;11], goal:RiskyGrowthGoal, initial:f64) {
    println!("SCORE_ROW\t{label}\tprop_v4={:.10}\trisky_v5={:.10}\trisky_goal_v6={:.10}",
        score_prop_firm_ga_fitness_v4(m),
        score_risky_ga_fitness_growth_v5(m),
        score_risky_ga_fitness_goal_v6(m,initial,180.,goal));
}
fn counterfactual_prop(
    base:[f64;11], multiplier:f64, dd_weight:f64, daily_weight:f64,
)->f64 {
    // Preserve all non-net/non-drawdown terms by stripping just these from
    // the real score, and adding configurable terms. Research-only formula.
    let raw=score_prop_firm_ga_fitness_v4(&base);
    let activity=0.3+0.7*(base[8]/30.0).clamp(0.,1.);
    let old_net=(base[0]/20_000.).clamp(-2.,2.)*0.15*activity;
    let old_dd=(base[3].max(0.)*15.).min(5.);
    let old_daily=base[10].clamp(0.,1.)*10.;
    raw -old_net + old_dd + old_daily
        + old_net*multiplier
        - (base[3].max(0.)*dd_weight).min(5.)
        -base[10].clamp(0.,1.)*daily_weight
}
fn main(){
    let goal=RiskyGrowthGoal{start_balance:100.,target_balance:50_000.,horizon_days:180.};
    let base=fixture();
    print_scores("baseline_net20_dd2pct_daily0.5pct", &base, goal,10_000.);
    for (slot,name,values) in [
        (0, "net_usd",vec![-1000.,-100.,0.,20.,100.,1000.,5000.,20000.]),
        (3, "max_dd_ratio",vec![0.,0.005,0.01,0.02,0.05,0.10,0.20]),
        (10, "worst_day_dd_ratio",vec![0.,0.002,0.005,0.01,0.02,0.04]),
        (5, "profit_factor",vec![0.80,0.95,0.99,0.9999,1.0,1.0001,1.01,1.2,1.8,3.]),
        (4, "win_rate",vec![0.30,0.40,0.45,0.48,0.50,0.60]),
        (7, "monthly_hit",vec![0.,0.25,0.5,0.75,1.]),
        (8, "trades",vec![1.,10.,30.,100.,500.]),
    ] {
        for val in values {
            let m=variant(base,slot,val);
            println!("SENSITIVITY\tvariable={name}\tvalue={val:.5}\tprop_v4={:.10}\trisky_v5={:.10}\trisky_goal_v6={:.10}",
                score_prop_firm_ga_fitness_v4(&m),
                score_risky_ga_fitness_growth_v5(&m),
                score_risky_ga_fitness_goal_v6(&m,10_000.,180.,goal));
        }
    }
    let net20=score_prop_firm_ga_fitness_v4(&variant(base,0,20.));
    let net120=score_prop_firm_ga_fitness_v4(&variant(base,0,120.));
    let dd1=score_prop_firm_ga_fitness_v4(&variant(base,3,base[3]+.01));
    let daily1=score_prop_firm_ga_fitness_v4(&variant(base,10,base[10]+.01));
    let pf_below=score_prop_firm_ga_fitness_v4(&variant(base,5,0.9999));
    let pf_above=score_prop_firm_ga_fitness_v4(&variant(base,5,1.0001));
    let net_effect=net120-net20;
    let dd_effect=net20-dd1;
    let daily_effect=net20-daily1;
    println!("ISOLATED_MAGNITUDE\tgain_100_usd={net_effect:.10}\tpenalty_1pp_overall_dd={dd_effect:.10}\tpenalty_1pp_daily_dd={daily_effect:.10}\tprofit_factor_jump_at_1={:.10}",
        pf_above-pf_below);
    println!("ISOLATED_RATIO\t1pp_dd_over_100usd_reward={:.1}\t1pp_daily_over_100usd_reward={:.1}", dd_effect/net_effect,daily_effect/net_effect);
    assert!((net_effect-0.00075).abs()<1e-10);
    assert!((dd_effect-0.15).abs()<1e-10);
    assert!((daily_effect-0.10).abs()<1e-10);
    assert!(pf_above-pf_below>0.249,"large gap just below PF1.0 is real");
    let risky_a=variant(base,0,20.);
    let risky_b=variant(base,0,2000.);
    assert_eq!(score_risky_ga_fitness_growth_v5(&risky_a),
               score_risky_ga_fitness_growth_v5(&risky_b),
               "v5 ignores positive-net deltas if PF/WR/trades same");
    assert!(score_risky_ga_fitness_goal_v6(&risky_b,10_000.,180.,goal)
           >score_risky_ga_fitness_goal_v6(&risky_a,10_000.,180.,goal),
           "v6 does reward larger realized profits");
    for net_gain_multiplier in [1.0,5.0,10.0,20.0] {
        for (dd_weight,daily_weight) in [(15.,10.),(8.,5.),(4.,2.)] {
            let calm=variant(variant(base,0,100.),3,0.02);
            let growth=variant(variant(base,0,2000.),3,0.05);
            let score_a=counterfactual_prop(calm,net_gain_multiplier,dd_weight,daily_weight);
            let score_b=counterfactual_prop(growth,net_gain_multiplier,dd_weight,daily_weight);
            println!("RESEARCH_ABLATION\tprofit_multiplier={net_gain_multiplier:.0}\tDD_weight={dd_weight:.0}\tdaily_weight={daily_weight:.0}\tcalm_score={score_a:.9}\tgrowth_score={score_b:.9}\twinner={}",
                if score_a>=score_b {"calm"} else {"growth"});
        }
    }
    println!("SCORING_SENSITIVITY_PASS real_v4_exact=1 real_v5_exact=1 real_v6_exact=1 counterfactual_production_change=0 MARKET_DATA=NONE");
}
