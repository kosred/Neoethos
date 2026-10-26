//! Experimental selection-pressure ablation using ACTUAL NeoEthos Rust
//! resident GA fitness functions. Pure synthetic fitness-metric fixtures;
//! NO historical market bars, NO production scoring edits, NO trading.
//!
//! Compare scoring/sorting alternatives over identical 400-candidate
//! populations and reproducible seed batches. All candidate metric rows
//! are created ONCE per seed/profile, then only selection changes.
use neoethos_gpu_contracts::resident_search_scoring_v2::{
    score_prop_firm_ga_fitness_v4, score_risky_ga_fitness_goal_v6, RiskyGrowthGoal,
};
use std::cmp::Ordering;

const POP: usize=400;
const TOP: usize=20;
const SEEDS: usize=64;
const NOVELTY_WEIGHT: f64=0.20;

#[derive(Clone)]
struct Candidate {
    fitness:f64,
    novelty:f64,
    net:f64,
    trades:f64,
    quality:f64,
}
struct Rand(u64);
impl Rand {
    fn new(seed:u64)->Self {Self(seed)}
    fn u(&mut self)->f64 {
        self.0^=self.0<<13;
        self.0^=self.0>>7;
        self.0^=self.0<<17;
        (self.0>>11) as f64/(1u64<<53) as f64
    }
}

const METHODS:[&str;6]=[
    "baseline_all_finite_minmax",
    "raw_only",
    "raw_plus_0p2_novelty",
    "active_minmax",
    "active_percentile",
    "active_winsorized_5_95",
];

fn fixture(seed:u64, nontrader_fraction:f64, mode:&str)->Vec<Candidate>{
    let mut rng=Rand::new(seed);
    let goal=RiskyGrowthGoal{start_balance:100.,target_balance:50_000.,horizon_days:180.};
    (0..POP).map(|_| {
        let no_trade=rng.u()<nontrader_fraction;
        let q=2.*rng.u()-1.;
        let noise=2.*rng.u()-1.;
        let novelty=rng.u();
        let trades=if no_trade {0.} else {15.+(rng.u()*185.).floor()};
        let net=if no_trade {0.} else {10_000.*(0.06+q*0.25+noise*0.01)};
        let sharpe=if no_trade {0.} else {0.7+q*0.8};
        let max_dd=if no_trade {0.} else {0.005+(1.-q)*0.035+rng.u()*0.008};
        let pf=if no_trade {0.} else {1.15+q*0.3};
        let wr=if no_trade {0.} else {(0.49+q*.10).clamp(0.,1.)};
        let monthly=if no_trade {0.} else {(0.5+q*.3).clamp(0.,1.)};
        let consistency=if no_trade {0.} else {(0.5+q*.2).clamp(0.,1.)};
        let daily_dd=if no_trade {0.} else {0.002+(1.-q)*0.005};
        let m=[net,sharpe,0.,max_dd,wr,pf,0.,monthly,trades,consistency,daily_dd];
        let score=if mode=="Risky-v6" {
            score_risky_ga_fitness_goal_v6(&m,10_000.,180.,goal)
        }else{
            score_prop_firm_ga_fitness_v4(&m)
        };
        assert!(score.is_finite(),"synthetic scoring must be finite");
        if no_trade { assert_eq!(score,-100.); }
        Candidate{fitness:score,novelty,net,trades,quality:q}
    }).collect()
}

fn quantile(sorted:&[f64],p:f64)->f64 {
    if sorted.is_empty() {return 0.0}
    let f=p*(sorted.len()-1) as f64;
    let a=f.floor() as usize;let b=f.ceil() as usize;
    sorted[a]+(sorted[b]-sorted[a])*(f-a as f64)
}

fn calculate_order(candidates:&[Candidate], method:&str,weight:f64)->Vec<usize>{
    let active:Vec<usize>=(0..candidates.len()).filter(|&i|candidates[i].trades>0. && candidates[i].fitness.is_finite()).collect();
    let mut rankvec=vec![-1e9f64;candidates.len()];
    let bound_indices:Vec<usize>=if method=="baseline_all_finite_minmax" {
        (0..candidates.len()).filter(|&i|candidates[i].fitness.is_finite()).collect()
    }else{active.clone()};
    let mut sorted:Vec<f64>=bound_indices.iter().map(|&i|candidates[i].fitness).collect();
    sorted.sort_by(|a,b|a.total_cmp(b));
    let lo=if method=="active_winsorized_5_95" {quantile(&sorted,0.05)} else {*sorted.first().unwrap_or(&0.)};
    let hi=if method=="active_winsorized_5_95" {quantile(&sorted,0.95)} else {*sorted.last().unwrap_or(&0.)};
    let range=(hi-lo).max(1e-9);
    let novmax=candidates.iter().map(|x|x.novelty).fold(0.,f64::max).max(1e-9);
    if method=="active_percentile" {
        let mut members=active.clone();
        members.sort_by(|a,b|candidates[*a].fitness.total_cmp(&candidates[*b].fitness).then_with(||a.cmp(b)));
        let mut start=0;
        while start<members.len(){
            let mut end=start+1;
            while end<members.len() && candidates[members[start]].fitness==candidates[members[end]].fitness {end+=1}
            let midpoint=(start+end-1) as f64/2.;
            let percentile=if members.len()==1 {.5} else {midpoint/(members.len()-1) as f64};
            for i in start..end {rankvec[members[i]]=percentile}
            start=end;
        }
    }
    let mut scored=Vec::with_capacity(candidates.len());
    for (i,x) in candidates.iter().enumerate(){
        let nov=x.novelty/novmax;
        let s=match method {
            "raw_only"=>x.fitness,
            "raw_plus_0p2_novelty"=>x.fitness+weight*nov,
            "baseline_all_finite_minmax"=>{
                let norm=(x.fitness-lo)/range;
                (1.-weight)*norm+weight*nov
            },
            "active_minmax"|"active_winsorized_5_95"=>{
                if x.trades<=0. || !x.fitness.is_finite(){-1e9}
                else {
                    let norm=(x.fitness.clamp(lo,hi)-lo)/range;
                    (1.-weight)*norm+weight*nov
                }
            },
            "active_percentile"=>{
                if x.trades<=0. || !x.fitness.is_finite(){-1e9}
                else {(1.-weight)*rankvec[i]+weight*nov}
            },
            _=>panic!("unknown comparison arm"),
        };
        scored.push((i,s));
    }
    scored.sort_by(|(ia,sa),(ib,sb)|sb.total_cmp(sa).then_with(||ia.cmp(ib)));
    scored.into_iter().map(|(i,_)|i).collect()
}

fn verify_invariants(){
    // No novelty means affine normalization of all finite scores is order preserving.
    let pop=fixture(42,0.6,"Risky-v6");
    let raw=calculate_order(&pop,"raw_only",0.);
    let base=calculate_order(&pop,"baseline_all_finite_minmax",0.);
    assert_eq!(raw,base,"minmax without novelty must preserve ranks");
    // If baseline includes -100, novelty may move a lower-profit genome above.
    let score=calculate_order(&pop,"baseline_all_finite_minmax",NOVELTY_WEIGHT);
    assert_ne!(score,raw,"mixture with novelty can invert pure-fitness order");
    // Eligibility handling: each active-only method must never select a no-trader
    // while at least TOP active genomes exist.
    for method in ["active_minmax","active_percentile","active_winsorized_5_95"] {
        let v=calculate_order(&pop,method,NOVELTY_WEIGHT);
        assert!(v[..TOP].iter().all(|&i|pop[i].trades>0.));
    }
    // Scaling raw scoring by 100x must not change pure fitness ranking,
    // but the raw+novelty mixture IS sensitive to units.
    let mut scaled=pop.clone();
    for x in &mut scaled {x.fitness*=100.}
    assert_eq!(calculate_order(&pop,"raw_only",0.),calculate_order(&scaled,"raw_only",0.));
    let unscaled=calculate_order(&pop,"raw_plus_0p2_novelty",NOVELTY_WEIGHT);
    let scaled_raw=calculate_order(&scaled,"raw_plus_0p2_novelty",NOVELTY_WEIGHT);
    assert_ne!(unscaled,scaled_raw,"raw + novelty depends arbitrarily on units");
    println!("RANK_INVARIANTS_PASS no_novelty_minmax_preserves_order=1 raw_scale_changes_mixed_order=1");
}

fn run(){
    verify_invariants();
    println!("NORMALIZATION_ABLATION_BEGIN population={POP} seeds={SEEDS} top={TOP} novelty_weight={NOVELTY_WEIGHT}");
    println!("NOTE synthetic correlated metrics; EXACT Rust Risky V6 & PropFirm V4 scorers; no historic trades/edge certification");
    for mode in ["Risky-v6","PropFirm-v4"]{
        for fraction in [0.,0.05,0.25,0.60,0.90]{
            let mut sums=vec![(0.,0.,0.,0.,0.,0.,0.);METHODS.len()];
            let mut changed_counts=vec![0usize;METHODS.len()];
            for seed in 0..SEEDS {
                let pop=fixture(0xDEADBEEF_u64.wrapping_add(seed as u64*7801),fraction,mode);
                let reference=calculate_order(&pop,"raw_only",0.);
                for (j,method) in METHODS.iter().enumerate(){
                    let order=calculate_order(&pop,method,NOVELTY_WEIGHT);
                    let top=&order[..TOP];
                    let (mut avg_fit,mut avg_net,mut avg_q,mut avg_nov,mut active,mut positive,mut overlap)=(0.,0.,0.,0.,0.,0.,0.);
                    for &i in top{
                        avg_fit+=pop[i].fitness/TOP as f64;
                        avg_net+=pop[i].net/(10_000.*TOP as f64);
                        avg_q+=pop[i].quality/TOP as f64;
                        avg_nov+=pop[i].novelty/TOP as f64;
                        if pop[i].trades>0. {active+=1.}
                        if pop[i].net>0. {positive+=1.}
                        if reference[..TOP].contains(&i){overlap+=1.}
                    }
                    let (ref mut f,ref mut n,ref mut q,ref mut v,ref mut a,ref mut p,ref mut o)=sums[j];
                    *f+=avg_fit;*n+=avg_net;*q+=avg_q;*v+=avg_nov;*a+=active;*p+=positive;*o+=overlap;
                    if order[..TOP] != reference[..TOP] {changed_counts[j]+=1;}
                }
            }
            for (j,name) in METHODS.iter().enumerate(){
                let (fit,net,quality,novelty,active,positive,overlap)=sums[j];
                let n=SEEDS as f64;
                println!("ABLAT mode={mode} no_trade_fraction={fraction:.2} method={name} mean_raw_fit_top20={:.6} mean_relative_net_top20={:.6} mean_quality_top20={:.6} mean_novelty_top20={:.6} active_top20={:.2} netpositive_top20={:.2} overlap_raw_top20={:.2} changed_vs_raw_trials={}/{}",
                    fit/n,net/n,quality/n,novelty/n,active/n,positive/n,overlap/n,changed_counts[j],SEEDS);
            }
        }
    }
    println!("NORMALIZATION_ABLATION_PASS real_scoring=1 synthetic_selection_only=1 final_oos_touched=0 production_changed=0");
}
fn main(){run();}
