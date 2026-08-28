use crate::resident_generation_v1::{
    GeneticOperatorIdentityV1, checked_philox_counter_mapping_v1,
    checked_philox_rejection_draw_index_v1, philox4x32_10_reference_v1,
};
use crate::resident_scoring_v2::ResidentScoringObjectiveV2;
use crate::resident_search_v2::{
    ResidentSearchAdvancePendingV2, ResidentSearchFixtureGeneV2, ResidentSearchFixturePlanV2,
    ResidentSearchGenerationFixtureSnapshotV2, ResidentSearchRunV2, ResidentSearchStateV2,
    ResidentSearchTryCompleteV2, ResidentSearchV2Error,
};
use crate::{
    NeoPopulationSettings, PopulationDatasetView, PopulationSession, SMC_SLOTS, ScenarioDescriptor,
};
use neoethos_gpu_contracts::ABI_VERSION;
use neoethos_gpu_contracts::resident_search_scoring_v2::{
    score_prop_firm_ga_fitness_v4, score_risky_ga_fitness_growth_v5,
};

const BARS: usize = 96;
const FEATURES: usize = 8;
const POPULATION: usize = 8;
const MAX_TERMS: usize = 3;
const SEARCH_SEED: u64 = 0x1234_5678_9abc_def0;
const RUN_IDENTITY: [u8; 32] = [0x62; 32];
const SCORING_CPU_ORACLE_TOLERANCE_V2: f64 = 2.0e-11;
const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;

#[derive(Debug, Clone)]
struct CpuGeneV2 {
    gene_identity: u64,
    content_hash: u64,
    term_count: u32,
    smc_flags: u32,
    long_threshold: f64,
    short_threshold: f64,
    target_pips: f64,
    stop_pips: f64,
    stop_vol_multiplier: f64,
    generation: u32,
    term_indices: [u64; MAX_TERMS],
    term_weights: [f64; MAX_TERMS],
}

#[derive(Debug)]
struct CpuGenerationV2 {
    initial_genes: Vec<CpuGeneV2>,
    final_genes: Vec<CpuGeneV2>,
    ranked_population_ordinals: Vec<u64>,
    parent_a: Vec<u64>,
    parent_b: Vec<u64>,
    selected_survivors: Vec<u64>,
    sorted_dedup_flags: Vec<u8>,
    candidate_valid_flags: Vec<u8>,
    selected_count: u32,
    dedup_run_count: u32,
}

fn fixture_session() -> Result<PopulationSession, Box<dyn std::error::Error>> {
    let close = (0..BARS)
        .map(|bar| 1.08 + bar as f64 * 0.000_01 + (bar % 5) as f64 * 0.000_003)
        .collect::<Vec<_>>();
    let high = close
        .iter()
        .enumerate()
        .map(|(bar, value)| value + 0.000_08 + (bar % 3) as f64 * 0.000_001)
        .collect::<Vec<_>>();
    let low = close
        .iter()
        .enumerate()
        .map(|(bar, value)| value - 0.000_07 - (bar % 4) as f64 * 0.000_001)
        .collect::<Vec<_>>();
    let indicators = (0..FEATURES * BARS)
        .map(|index| match index % 4 {
            0 => 0.35,
            1 => -0.2,
            2 => 0.1,
            _ => -0.05,
        })
        .collect::<Vec<_>>();
    let months = vec![202_401_i64; BARS];
    let days = (0..BARS)
        .map(|bar| 1_i64 + i64::try_from(bar / 24).expect("day fits i64"))
        .collect::<Vec<_>>();
    let timestamps = (0..BARS)
        .map(|bar| 1_704_067_200_000_i64 + bar as i64 * 300_000)
        .collect::<Vec<_>>();
    let smc_rows = vec![0_i8; BARS * SMC_SLOTS];
    let mut session = PopulationSession::create(0, 1)?;
    session.upload_dataset(PopulationDatasetView {
        close: &close,
        high: &high,
        low: &low,
        indicators: &indicators,
        feature_count: FEATURES,
        months: &months,
        days: &days,
        timestamps: &timestamps,
        smc_rows: &smc_rows,
        adaptive_base_pips: None,
    })?;
    Ok(session)
}

fn settings() -> NeoPopulationSettings {
    NeoPopulationSettings {
        abi_version: ABI_VERSION,
        max_hold_bars: 8,
        min_hold_bars: 1,
        max_trades_per_day: 3,
        month_capacity: 12,
        gap_threshold_ms: 600_000,
        initial_equity: 100_000.0,
        pip_value: 0.000_1,
        spread_pips: 0.8,
        commission_per_trade: 5.0,
        pip_value_per_lot: 10.0,
        risk_per_trade_min: 0.005,
        risk_per_trade_max: 0.01,
        high_quality_confidence: 0.75,
        spread_pips_asian: 0.8,
        spread_pips_overlap: 0.8,
        spread_pips_late_ny: 0.8,
        ..NeoPopulationSettings::default()
    }
}

fn begin_run(
    objective: ResidentScoringObjectiveV2,
    novelty_weight: f64,
) -> Result<ResidentSearchRunV2, ResidentSearchV2Error> {
    let session = fixture_session().map_err(|_| {
        ResidentSearchV2Error::InvalidPlan("failed to construct CUDA fixture session")
    })?;
    let plan = ResidentSearchFixturePlanV2::new_with_scoring_objective_v2(
        POPULATION, FEATURES, objective,
    )?;
    session.begin_resident_search_scoring_fixture_v2(
        plan,
        [1.0; SMC_SLOTS],
        true,
        objective,
        novelty_weight,
    )
}

fn upload_full_population_scenarios(
    run: &mut ResidentSearchRunV2,
) -> Result<(), ResidentSearchV2Error> {
    let scenarios = (0..POPULATION)
        .map(|population_ordinal| ScenarioDescriptor {
            base_candidate_id: population_ordinal as u64,
            scenario_id: 10_000 + population_ordinal as u64,
            window_offset: 0,
            window_len: BARS as u32,
            ..ScenarioDescriptor::default()
        })
        .collect::<Vec<_>>();
    run.upload_resident_scenarios_v2(&scenarios)
}

fn ordered_f64_score(value: f64) -> u64 {
    assert!(value.is_finite());
    let canonical = if value == 0.0 { 0.0 } else { value };
    let bits = canonical.to_bits();
    let key = if bits >> 63 == 0 {
        bits ^ (1_u64 << 63)
    } else {
        !bits
    };
    key.max(1)
}

fn philox_draw(
    generation: usize,
    identity: u64,
    operator: GeneticOperatorIdentityV1,
    draw_index: u64,
) -> [u32; 4] {
    let address = checked_philox_counter_mapping_v1(
        SEARCH_SEED,
        &RUN_IDENTITY,
        generation,
        identity,
        operator,
        draw_index,
    )
    .expect("fixture Philox address is bounded");
    philox4x32_10_reference_v1(address.counter(), address.key())
}

fn uniform_below(
    generation: usize,
    identity: u64,
    operator: GeneticOperatorIdentityV1,
    decision_slot: u32,
    upper: u64,
) -> u64 {
    if upper <= 1 {
        return 0;
    }
    let limit = u64::MAX - (u64::MAX % upper);
    for attempt in 0..=u32::MAX {
        let draw = philox_draw(
            generation,
            identity,
            operator,
            checked_philox_rejection_draw_index_v1(decision_slot, attempt),
        );
        let value = (u64::from(draw[1]) << 32) | u64::from(draw[0]);
        if value < limit {
            return value % upper;
        }
    }
    unreachable!("a fixture rejection sample must terminate")
}

fn normalize_gene(gene: &mut CpuGeneV2) {
    let mut write = 0;
    for term in 0..usize::try_from(gene.term_count).expect("term count fits usize") {
        let index = gene.term_indices[term];
        let weight = gene.term_weights[term].clamp(-5.0, 5.0);
        if index < FEATURES as u64 && weight.is_finite() && weight.abs() > 1.0e-6 {
            gene.term_indices[write] = index;
            gene.term_weights[write] = weight;
            write += 1;
        }
    }
    if write == 0 {
        gene.term_indices[0] = gene.gene_identity % FEATURES as u64;
        gene.term_weights[0] = if gene.gene_identity & 1 == 0 {
            1.0
        } else {
            -1.0
        };
        write = 1;
    }
    gene.term_count = write as u32;
    for term in write..MAX_TERMS {
        gene.term_indices[term] = 0;
        gene.term_weights[term] = 0.0;
    }
    gene.long_threshold = gene.long_threshold.clamp(0.0, 1.0);
    gene.short_threshold = gene.short_threshold.clamp(0.0, 1.0);
    gene.target_pips = gene.target_pips.clamp(0.0, 1_000_000.0);
    gene.stop_pips = gene.stop_pips.clamp(0.0, 1_000_000.0);
    gene.stop_vol_multiplier = gene.stop_vol_multiplier.clamp(0.0, 1_000_000.0);
    gene.smc_flags &= (1_u32 << SMC_SLOTS) - 1;
}

fn hash_mix(mut hash: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn content_hash(gene: &CpuGeneV2) -> u64 {
    let mut hash = FNV_OFFSET;
    for value in [
        u64::from(gene.term_count),
        u64::from(gene.smc_flags),
        gene.long_threshold.to_bits(),
        gene.short_threshold.to_bits(),
        gene.target_pips.to_bits(),
        gene.stop_pips.to_bits(),
        gene.stop_vol_multiplier.to_bits(),
    ] {
        hash = hash_mix(hash, value);
    }
    for term in 0..MAX_TERMS {
        hash = hash_mix(hash, gene.term_indices[term]);
        hash = hash_mix(hash, gene.term_weights[term].to_bits());
    }
    hash
}

fn initialize_gene(candidate: usize) -> CpuGeneV2 {
    let identity = candidate as u64;
    let mut gene = CpuGeneV2 {
        gene_identity: identity,
        content_hash: 0,
        term_count: 1 + uniform_below(
            0,
            identity,
            GeneticOperatorIdentityV1::InitializeTermCount,
            0,
            MAX_TERMS as u64,
        ) as u32,
        smc_flags: 0,
        long_threshold: 0.05
            * (uniform_below(
                0,
                identity,
                GeneticOperatorIdentityV1::InitializeThreshold,
                0,
                6,
            ) as f64
                + 1.0),
        short_threshold: 0.05
            * (uniform_below(
                0,
                identity,
                GeneticOperatorIdentityV1::InitializeThreshold,
                1,
                6,
            ) as f64
                + 1.0),
        target_pips: uniform_below(
            0,
            identity,
            GeneticOperatorIdentityV1::InitializeStopGeometry,
            0,
            6,
        ) as f64
            + 1.0,
        stop_pips: uniform_below(
            0,
            identity,
            GeneticOperatorIdentityV1::InitializeStopGeometry,
            1,
            6,
        ) as f64
            + 1.0,
        stop_vol_multiplier: uniform_below(
            0,
            identity,
            GeneticOperatorIdentityV1::InitializeStopGeometry,
            2,
            6,
        ) as f64
            + 1.0,
        generation: 0,
        term_indices: [0; MAX_TERMS],
        term_weights: [0.0; MAX_TERMS],
    };
    for term in 0..MAX_TERMS {
        gene.term_indices[term] = uniform_below(
            0,
            identity,
            GeneticOperatorIdentityV1::InitializeIndicator,
            term as u32,
            FEATURES as u64,
        );
        let magnitude = 1.0
            + uniform_below(
                0,
                identity,
                GeneticOperatorIdentityV1::InitializeWeightLevel,
                term as u32,
                5,
            ) as f64;
        gene.term_weights[term] = if philox_draw(
            0,
            identity,
            GeneticOperatorIdentityV1::InitializeWeightSign,
            term as u64,
        )[0] & 1
            == 0
        {
            magnitude
        } else {
            -magnitude
        };
    }
    normalize_gene(&mut gene);
    gene.content_hash = content_hash(&gene);
    gene
}

fn rank_from_weighted_draw(draw: u64) -> usize {
    let mut low = 0_u64;
    let mut high = POPULATION as u64;
    while low < high {
        let middle = low + (high - low) / 2;
        let items = middle + 1;
        let prefix = items * (2 * POPULATION as u64 - middle) / 2;
        if draw < prefix {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    usize::try_from(low.min(POPULATION as u64 - 1)).expect("rank fits usize")
}

fn cpu_generation(scores: &[f64]) -> CpuGenerationV2 {
    let initial_genes = (0..POPULATION).map(initialize_gene).collect::<Vec<_>>();
    let mut ranked_population_ordinals = (0..POPULATION as u64).collect::<Vec<_>>();
    ranked_population_ordinals.sort_by(|left, right| {
        ordered_f64_score(scores[*right as usize])
            .cmp(&ordered_f64_score(scores[*left as usize]))
            .then_with(|| {
                initial_genes[*left as usize]
                    .gene_identity
                    .cmp(&initial_genes[*right as usize].gene_identity)
            })
            .then_with(|| left.cmp(right))
    });

    let rank_weight_total = POPULATION as u64 * (POPULATION as u64 + 1) / 2;
    let mut parent_a = vec![0; POPULATION];
    let mut parent_b = vec![0; POPULATION];
    for child in 0..POPULATION {
        let rank_a = rank_from_weighted_draw(uniform_below(
            0,
            child as u64,
            GeneticOperatorIdentityV1::ParentA,
            0,
            rank_weight_total,
        ));
        let rank_b = rank_from_weighted_draw(uniform_below(
            0,
            child as u64,
            GeneticOperatorIdentityV1::ParentB,
            0,
            rank_weight_total,
        ));
        parent_a[child] = ranked_population_ordinals[rank_a];
        parent_b[child] = ranked_population_ordinals[rank_b];
    }

    let survivor_draw = uniform_below(
        0,
        0,
        GeneticOperatorIdentityV1::Survivor,
        0,
        rank_weight_total,
    );
    let selected_survivors =
        vec![ranked_population_ordinals[rank_from_weighted_draw(survivor_draw)]];
    let mut final_genes = Vec::with_capacity(POPULATION);
    for destination in 0..POPULATION {
        if destination == 0 {
            let mut survivor = initial_genes[selected_survivors[0] as usize].clone();
            survivor.generation = 1;
            final_genes.push(survivor);
            continue;
        }
        let mut child = initial_genes[parent_a[destination] as usize].clone();
        let right = &initial_genes[parent_b[destination] as usize];
        child.gene_identity = (1_u64 << 32) ^ destination as u64;
        child.generation = 1;
        let scalar = philox_draw(
            1,
            child.gene_identity,
            GeneticOperatorIdentityV1::CrossoverScalar,
            0,
        );
        if scalar[0] & 1 != 0 {
            child.long_threshold = right.long_threshold;
        }
        if scalar[1] & 1 != 0 {
            child.short_threshold = right.short_threshold;
        }
        if scalar[2] & 1 != 0 {
            child.target_pips = right.target_pips;
        }
        if scalar[3] & 1 != 0 {
            child.stop_pips = right.stop_pips;
        }
        for term in 0..MAX_TERMS {
            if philox_draw(
                1,
                child.gene_identity,
                GeneticOperatorIdentityV1::CrossoverScalar,
                term as u64 + 1,
            )[0] & 1
                != 0
            {
                child.term_indices[term] = right.term_indices[term];
                child.term_weights[term] = right.term_weights[term];
            }
        }
        let mutation = philox_draw(
            1,
            child.gene_identity,
            GeneticOperatorIdentityV1::MutationKind,
            0,
        );
        assert!(u64::from(mutation[0]) < 1_u64 << 32);
        let term = uniform_below(
            1,
            child.gene_identity,
            GeneticOperatorIdentityV1::MutationValue,
            0,
            MAX_TERMS as u64,
        ) as usize;
        child.term_indices[term] = uniform_below(
            1,
            child.gene_identity,
            GeneticOperatorIdentityV1::MutationValue,
            1,
            FEATURES as u64,
        );
        let magnitude = 1.0
            + uniform_below(
                1,
                child.gene_identity,
                GeneticOperatorIdentityV1::MutationValue,
                2,
                5,
            ) as f64;
        child.term_weights[term] = if philox_draw(
            1,
            child.gene_identity,
            GeneticOperatorIdentityV1::MutationValue,
            3_u64 << 32,
        )[0] & 1
            == 0
        {
            magnitude
        } else {
            -magnitude
        };
        normalize_gene(&mut child);
        child.content_hash = content_hash(&child);
        final_genes.push(child);
    }

    let mut by_hash = (0..POPULATION).collect::<Vec<_>>();
    by_hash.sort_by_key(|candidate| final_genes[*candidate].content_hash);
    let mut sorted_dedup_flags = vec![1; POPULATION];
    let mut candidate_valid_flags = vec![1; POPULATION];
    for position in 1..POPULATION {
        let prior = by_hash[position - 1];
        let candidate = by_hash[position];
        if final_genes[prior].content_hash == final_genes[candidate].content_hash {
            let equal = genes_equal(&final_genes[prior], &final_genes[candidate]);
            assert!(
                !equal,
                "clean fixture unexpectedly produced duplicate content"
            );
            sorted_dedup_flags[position] = 0;
            candidate_valid_flags[candidate] = 0;
        }
    }
    let dedup_run_count = by_hash
        .iter()
        .enumerate()
        .filter(|(index, candidate)| {
            *index == 0
                || final_genes[**candidate].content_hash
                    != final_genes[by_hash[*index - 1]].content_hash
        })
        .count() as u32;
    let selected_count =
        u32::try_from(sorted_dedup_flags.iter().filter(|flag| **flag != 0).count())
            .expect("fixture population fits u32");
    CpuGenerationV2 {
        initial_genes,
        final_genes,
        ranked_population_ordinals,
        parent_a,
        parent_b,
        selected_survivors,
        sorted_dedup_flags,
        candidate_valid_flags,
        selected_count,
        dedup_run_count,
    }
}

fn genes_equal(left: &CpuGeneV2, right: &CpuGeneV2) -> bool {
    left.term_count == right.term_count
        && left.smc_flags == right.smc_flags
        && left.long_threshold.to_bits() == right.long_threshold.to_bits()
        && left.short_threshold.to_bits() == right.short_threshold.to_bits()
        && left.target_pips.to_bits() == right.target_pips.to_bits()
        && left.stop_pips.to_bits() == right.stop_pips.to_bits()
        && left.stop_vol_multiplier.to_bits() == right.stop_vol_multiplier.to_bits()
        && left.term_indices == right.term_indices
        && left
            .term_weights
            .iter()
            .zip(right.term_weights.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

fn assert_gene_exact(actual: &ResidentSearchFixtureGeneV2, expected: &CpuGeneV2) {
    assert_eq!(actual.gene_identity, expected.gene_identity);
    assert_eq!(actual.content_hash, expected.content_hash);
    assert_eq!(actual.term_count, expected.term_count);
    assert_eq!(actual.smc_flags, expected.smc_flags);
    assert_eq!(
        actual.long_threshold.to_bits(),
        expected.long_threshold.to_bits()
    );
    assert_eq!(
        actual.short_threshold.to_bits(),
        expected.short_threshold.to_bits()
    );
    assert_eq!(actual.target_pips.to_bits(), expected.target_pips.to_bits());
    assert_eq!(actual.stop_pips.to_bits(), expected.stop_pips.to_bits());
    assert_eq!(
        actual.stop_vol_multiplier.to_bits(),
        expected.stop_vol_multiplier.to_bits()
    );
    assert_eq!(actual.generation, expected.generation);
    assert_eq!(actual.term_indices, expected.term_indices);
    assert!(
        actual
            .term_weights
            .iter()
            .zip(expected.term_weights.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}

fn assert_score_oracle(
    snapshot: &ResidentSearchGenerationFixtureSnapshotV2,
    objective: ResidentScoringObjectiveV2,
) {
    for (row, device_score) in snapshot.metric_rows.iter().zip(&snapshot.fitness_scores) {
        assert!(row.values.iter().all(|value| value.is_finite()));
        let cpu_score = match objective {
            ResidentScoringObjectiveV2::PropFirmV4 => score_prop_firm_ga_fitness_v4(&row.values),
            ResidentScoringObjectiveV2::RiskyGrowthV5 => {
                score_risky_ga_fitness_growth_v5(&row.values)
            }
        };
        let scale = cpu_score.abs().max(device_score.abs()).max(1.0);
        assert!(
            (cpu_score - device_score).abs() <= SCORING_CPU_ORACLE_TOLERANCE_V2 * scale,
            "CPU={cpu_score:?} device={device_score:?} metrics={:?}",
            row.values
        );
    }
    for (score, key) in snapshot.fitness_scores.iter().zip(&snapshot.decision_keys) {
        assert_eq!(*key, ordered_f64_score(*score));
    }
}

fn assert_full_generation_oracle(snapshot: &ResidentSearchGenerationFixtureSnapshotV2) {
    let cpu = cpu_generation(&snapshot.fitness_scores);
    assert_eq!(
        snapshot.ranked_population_ordinals,
        cpu.ranked_population_ordinals
    );
    assert_eq!(snapshot.parent_a, cpu.parent_a);
    assert_eq!(snapshot.parent_b, cpu.parent_b);
    assert_eq!(snapshot.selected_survivors, cpu.selected_survivors);
    assert_eq!(snapshot.sorted_dedup_flags, cpu.sorted_dedup_flags);
    assert_eq!(snapshot.candidate_valid_flags, cpu.candidate_valid_flags);
    assert_eq!(snapshot.selected_count, cpu.selected_count);
    assert_eq!(snapshot.dedup_run_count, cpu.dedup_run_count);
    for (actual, expected) in snapshot.initial_genes.iter().zip(&cpu.initial_genes) {
        assert_gene_exact(actual, expected);
    }
    for (actual, expected) in snapshot.final_genes.iter().zip(&cpu.final_genes) {
        assert_gene_exact(actual, expected);
    }
}

fn complete_pending(
    mut pending: ResidentSearchAdvancePendingV2,
) -> Result<ResidentSearchRunV2, ResidentSearchV2Error> {
    for _ in 0..100_000 {
        match pending.try_complete_one_generation_v2()? {
            ResidentSearchTryCompleteV2::NotReady(owner) => {
                pending = owner;
                std::thread::yield_now();
            }
            ResidentSearchTryCompleteV2::Complete(run) => return Ok(run),
        }
    }
    panic!("bounded completion polling exhausted")
}

fn assert_clean_advance(snapshot: &ResidentSearchGenerationFixtureSnapshotV2) {
    assert_eq!(snapshot.scoring_device_fault, 0);
    assert_eq!(snapshot.generation_device_fault, 0);
    assert_eq!(snapshot.gene_hash_collision_fault, 0);
    assert_eq!(snapshot.control_fault_word, 0);
    assert_eq!(snapshot.stop_requested, 0);
    assert_eq!(snapshot.current_store_index, 1);
    assert_eq!(snapshot.generation_index, 1);
    assert_eq!(snapshot.store_epoch, 2);
    assert_eq!(snapshot.initial_genes.len(), POPULATION);
    assert_eq!(snapshot.final_genes.len(), POPULATION);
    assert!(
        snapshot
            .final_genes
            .iter()
            .all(|gene| gene.generation == 1 && gene.term_count > 0)
    );
    assert_eq!(snapshot.population_counters.gene_upload_bytes, 0);
    assert_eq!(snapshot.population_counters.full_readback_bytes, 0);
    assert!(snapshot.terminal_synchronization_count > 0);
    assert!(snapshot.terminal_readback_count > 0);
    assert!(snapshot.terminal_readback_bytes > 0);
}

#[test]
fn resident_search_scores_and_advances_exactly_one_generation_on_real_cuda()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(
        std::env::var("NEOETHOS_REQUIRE_GPU").ok().as_deref(),
        Some("1"),
        "NEOETHOS_REQUIRE_GPU=1 is mandatory for this real-card oracle"
    );
    assert_eq!(ordered_f64_score(-0.0_f64), ordered_f64_score(0.0));

    for invalid_novelty in [-0.0_f64, 0.25, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let result = begin_run(ResidentScoringObjectiveV2::PropFirmV4, invalid_novelty);
        assert!(matches!(result, Err(ResidentSearchV2Error::InvalidPlan(_))));
    }

    for objective in [
        ResidentScoringObjectiveV2::PropFirmV4,
        ResidentScoringObjectiveV2::RiskyGrowthV5,
    ] {
        let mut run = begin_run(objective, 0.0)?;
        let admission = run
            .combined_admission_summary_fixture_v2()
            .ok_or("missing combined admission")?;
        assert_eq!(admission.free_memory_snapshot_count(), 1);
        assert_eq!(admission.generation_allocation_count(), 1);
        assert_eq!(admission.scoring_allocation_count(), 1);
        assert_eq!(admission.terminal_host_allocation_count(), 1);
        assert!(admission.terminal_host_receipt_bytes() > 0);
        assert!(admission.full_discovery_reserve_bytes() > 0);
        assert!(
            admission.total_device_bytes()
                <= admission
                    .same_context_free_bytes()
                    .checked_sub(admission.full_discovery_reserve_bytes())
                    .ok_or("reserve exceeds free snapshot")?
        );
        assert!(admission.runtime_identity_is_exact());
        assert!(admission.sealed_before_first_allocation());
        upload_full_population_scenarios(&mut run)?;
        let pending = run.advance_one_full_population_generation_v2(&settings())?;
        assert_eq!(pending.state_v2(), ResidentSearchStateV2::AdvancePending);
        assert_eq!(
            pending.committed_gene_view_summary_v2().generation_index(),
            0
        );
        let mut run = complete_pending(pending)?;
        assert_eq!(run.state_v2(), ResidentSearchStateV2::AdvancedOnce);
        let ready = run
            .ready_event_summary_v2()
            .ok_or("missing committed ready receipt")?;
        assert_eq!(ready.generation_index(), 1);
        assert_eq!(ready.intermediate_host_wait_count(), 0);
        assert_eq!(ready.intermediate_readback_count(), 0);
        let terminal = run
            .terminal_receipt_summary_v2()
            .ok_or("missing terminal receipt")?;
        assert_eq!(terminal.generation_index(), 1);
        assert_eq!(terminal.store_epoch(), 2);
        assert_eq!(terminal.current_store_index(), 1);
        assert_eq!(terminal.device_fault_word(), 0);
        assert_eq!(terminal.compact_async_d2h_count(), 1);
        assert!(terminal.compact_async_d2h_bytes() > 0);
        assert!(terminal.completion_event_query_count() > 0);
        assert_eq!(terminal.completion_stream_synchronize_count(), 0);
        let snapshot = run.terminal_fixture_snapshot_v2()?;
        eprintln!(
            "resident-search-v2 terminal objective={objective:?} generation={} store_epoch={} store={} fault={} compact_d2h_count={} compact_d2h_bytes={} event_queries={} stream_syncs={} gene_upload_bytes={} full_readback_bytes={}",
            terminal.generation_index(),
            terminal.store_epoch(),
            terminal.current_store_index(),
            terminal.device_fault_word(),
            terminal.compact_async_d2h_count(),
            terminal.compact_async_d2h_bytes(),
            terminal.completion_event_query_count(),
            terminal.completion_stream_synchronize_count(),
            snapshot.population_counters.gene_upload_bytes,
            snapshot.population_counters.full_readback_bytes,
        );
        assert_eq!(snapshot.scoring_objective, objective as u32);
        assert_score_oracle(&snapshot, objective);
        assert_full_generation_oracle(&snapshot);
        assert_clean_advance(&snapshot);
        let second = run.advance_one_full_population_generation_v2(&settings());
        assert!(matches!(
            second,
            Err(ResidentSearchV2Error::OneGenerationAdvanceAlreadyEnqueued)
        ));
    }

    let identities = [50_u64, 10, 10, 40, 30, 20, 60, 0];
    let mut reordered = begin_run(ResidentScoringObjectiveV2::PropFirmV4, 0.0)?;
    for (ordinal, identity) in identities.into_iter().enumerate() {
        reordered.set_gene_identity_fixture_v2(ordinal as u64, identity)?;
    }
    reordered.set_scoring_metric_mode_fixture_v2(3)?;
    upload_full_population_scenarios(&mut reordered)?;
    let pending = reordered.advance_one_full_population_generation_v2(&settings())?;
    let mut reordered = complete_pending(pending)?;
    let snapshot = reordered.terminal_fixture_snapshot_v2()?;
    let mut expected = (0..POPULATION as u64).collect::<Vec<_>>();
    expected.sort_by(|left, right| {
        ordered_f64_score(snapshot.fitness_scores[*right as usize])
            .cmp(&ordered_f64_score(snapshot.fitness_scores[*left as usize]))
            .then_with(|| identities[*left as usize].cmp(&identities[*right as usize]))
            .then_with(|| left.cmp(right))
    });
    assert_ne!(expected, (0..POPULATION as u64).collect::<Vec<_>>());
    assert_eq!(snapshot.ranked_population_ordinals, expected);
    drop(reordered.close_fixture_v2()?);

    let nonfinite_values = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::NEG_INFINITY,
    ];
    let cleanup_before = ResidentSearchRunV2::pending_drop_audit_fixture_v2()?;
    for (metric_slot, nonfinite) in nonfinite_values.into_iter().enumerate() {
        let mut run = begin_run(ResidentScoringObjectiveV2::PropFirmV4, 0.0)?;
        run.set_scoring_metric_fault_fixture_v2(metric_slot as u32, nonfinite)?;
        upload_full_population_scenarios(&mut run)?;
        let pending = run.advance_one_full_population_generation_v2(&settings())?;
        let fault = match complete_pending(pending) {
            Ok(_) => panic!("nonfinite metric must fault transaction"),
            Err(error) => error,
        };
        match fault {
            ResidentSearchV2Error::DeviceTerminalFault(receipt) => {
                assert_ne!(receipt.scoring_device_fault(), 0);
                assert_ne!(receipt.generation_device_fault(), 0);
                assert_ne!(receipt.control_fault_word(), 0);
                assert_eq!(receipt.stop_requested(), 1);
                assert_eq!(receipt.generation_index(), 0);
                assert_eq!(receipt.store_epoch(), 1);
                assert_eq!(receipt.current_store_index(), 0);
                assert_eq!(receipt.compact_async_d2h_count(), 1);
            }
            other => panic!("unexpected terminal fault: {other:?}"),
        }
    }

    let mut duplicate = begin_run(ResidentScoringObjectiveV2::PropFirmV4, 0.0)?;
    duplicate.set_duplicate_final_gene_content_fixture_v2(0, 1)?;
    upload_full_population_scenarios(&mut duplicate)?;
    let pending = duplicate.advance_one_full_population_generation_v2(&settings())?;
    let duplicate_fault = match complete_pending(pending) {
        Ok(_) => panic!("duplicate full-gene content must fail closed"),
        Err(error) => error,
    };
    assert!(matches!(
        duplicate_fault,
        ResidentSearchV2Error::DeviceTerminalFault(_)
    ));
    let cleanup_after = ResidentSearchRunV2::pending_drop_audit_fixture_v2()?;
    assert_eq!(
        cleanup_after.terminal_fault_cleanup_count()
            - cleanup_before.terminal_fault_cleanup_count(),
        12,
        "all eleven metric faults plus the duplicate fault must clean owners"
    );
    assert_eq!(
        cleanup_after.terminal_session_destroy_count()
            - cleanup_before.terminal_session_destroy_count(),
        12,
        "every Ready+fault path must destroy its terminal-proven session"
    );

    let mut dropped = begin_run(ResidentScoringObjectiveV2::PropFirmV4, 0.0)?;
    upload_full_population_scenarios(&mut dropped)?;
    let pending = dropped.advance_one_full_population_generation_v2(&settings())?;
    drop(pending);
    let audit = ResidentSearchRunV2::pending_drop_audit_fixture_v2()?;
    assert!(audit.poisoned_pending_drop_count() > 0);
    assert_eq!(audit.reused_in_flight_session_count(), 0);

    let mut fresh = begin_run(ResidentScoringObjectiveV2::PropFirmV4, 0.0)?;
    upload_full_population_scenarios(&mut fresh)?;
    let pending = fresh.advance_one_full_population_generation_v2(&settings())?;
    let fresh = complete_pending(pending)?;
    drop(fresh.close_fixture_v2()?);

    // Archive/kNN novelty, persistent evolution and final promotion remain out
    // of scope; all five production readiness bits stay fail-closed.
    Ok(())
}
