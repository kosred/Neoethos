//! Read-only research capacity probe for the REAL NeoEthos genetic operators.
//!
//! Verifies POP=400 x GEN=1000 can generate, recombine, mutate and preserve
//! bounded candidate/archive metadata on a GitHub CPU runner. It DOES NOT
//! price any historical trade, execute the selection funnel or claim GPU
//! throughput or that the complete 1000-generation search will finish.
//! Financial evaluation requires sealed broker-truth research contracts.
//! This test must never bypass the historical evaluation authority.

use std::collections::{HashSet, VecDeque};
use std::time::Instant;
use rand::{rngs::StdRng, SeedableRng};
use neoethos_search::genetic::{
    SmcSearchConfig, GeneticSearchRuntimeOverrides, crossover, generate_random_genes,
    gene_signature_hash, mutate, new_random_gene,
};

#[test]
fn capacity_400_population_1000_generations_genuine_genetic_operators() {
    const POP: usize = 400;
    const GEN: usize = 1000;
    const FEATURES: usize = 240;
    const MAX_TERMS: usize = 12;
    let mut rng = StdRng::seed_from_u64(0x4E_45_4F_45_54_48_4F_53);
    let smc = SmcSearchConfig::default();
    let overrides = GeneticSearchRuntimeOverrides::default();
    assert_eq!(overrides.checked_effective_archive_cap(POP, GEN), Some(50_000));
    let archive_cap=overrides.checked_effective_archive_cap(POP,GEN).unwrap();

    let started=Instant::now();
    let mut population=generate_random_genes(POP, FEATURES, MAX_TERMS, 0, &smc, &mut rng);
    assert_eq!(population.len(), POP);
    let mut recent_archive=VecDeque::with_capacity(archive_cap);
    let mut unique_hashes=HashSet::with_capacity(archive_cap);
    let mut total_children=0usize;
    let mut structural_faults=0usize;

    for generation in 1..=GEN {
        let mut next=Vec::with_capacity(POP);
        for i in 0..POP {
            // 25% independently sampled immigrants, remaining true
            // NeoEthos crossover + mutation; no mocked strategy templates.
            let child=if i%4==0 {
                new_random_gene(FEATURES,MAX_TERMS,generation,&smc,&mut rng)
            } else {
                let pa=&population[(i*17+generation)%POP];
                let pb=&population[(i*31+generation/2)%POP];
                let crossed=crossover(pa,pb,generation,&mut rng);
                mutate(&crossed,FEATURES,MAX_TERMS,generation,&smc,generation%20,&mut rng)
            };
            if child.indices.len()!=child.weights.len()
                || child.indices.len()>MAX_TERMS || child.indices.is_empty()
                || child.indices.iter().any(|&x|x>=FEATURES)
            {
                structural_faults+=1;
            }
            let signature=gene_signature_hash(&child);
            if unique_hashes.insert(signature) {
                // A worst-case *metadata* archive stress: retain full Gene,
                // no actual GPU device buffers/indicator cube/trade ledgers.
                recent_archive.push_back((signature,child.clone()));
                if recent_archive.len()>archive_cap {
                    if let Some((old,_))=recent_archive.pop_front() {
                        unique_hashes.remove(&old);
                    }
                }
            }
            next.push(child);
            total_children+=1;
        }
        population=next;
        if generation%100==0 || generation==1 {
            let rss=std::fs::read_to_string("/proc/self/status").unwrap_or_default()
                .lines().filter(|l|l.starts_with("VmHWM:") || l.starts_with("VmRSS:"))
                .collect::<Vec<_>>().join("; ");
            println!("GA_CAPACITY PROGRESS generation={generation}/{GEN} population={POP} created={total_children} unique_recent={} archive={} elapsed_ms={} {}",
                unique_hashes.len(),recent_archive.len(),started.elapsed().as_millis(),rss);
        }
    }
    assert_eq!(total_children,POP*GEN);
    assert_eq!(structural_faults,0,"genetic operator emitted structurally invalid genomes");
    assert_eq!(population.len(),POP);
    assert!(recent_archive.len()<=archive_cap);
    println!("GA_CAPACITY RESULT PASS pop={POP} generations={GEN} operations={total_children} archive_cap={archive_cap} retained_genomes={} genetic_only_ms={} NOTE=NO_MARKET_EVALUATION",
        recent_archive.len(),started.elapsed().as_millis());
}
