use super::{
    RESIDENT_ALLOCATOR_CONTEXT_RESERVE_POLICY_V3, ResidentFeatureColumnBindingV3,
    ResidentFeatureStoreCudaErrorV3, ResidentFeatureStoreSearchStartErrorV2,
    ResidentPopulationSessionV3,
};
use crate::full_discovery_workspace_plan_v1::seal_test_full_discovery_run_device_v3;
use crate::population::{
    CudaPopulationError, PopulationEvaluationViewV1, PopulationGeneView, PopulationTimestampModeV1,
    ResidentAdaptiveBaseRequestV1, STATUS_ADAPTIVE_BASE_DEGENERATE,
    STATUS_STRICT_RESIDENT_POISONED,
};
use crate::resident_generation_v1::{
    ParentSelectionPolicyV1, ResidentGenerationPlanAuthorityInputV1,
    SealedResidentGenerationPlanV1, SurvivorSelectionPolicyV1,
    discovery_generation_semantics_sha256_v1, resident_metric_semantics_sha256_v2,
    seal_resident_generation_plan_v1,
};
use crate::resident_scoring_v2::{
    ResidentScoringObjectiveV2, novelty_disabled_semantics_sha256_v2, rank_semantics_sha256_v2,
    scoring_semantics_sha256_v2,
};
#[cfg(feature = "cuda-device-fixtures")]
use crate::resident_search_v2::resident_search_v2_production_readiness;
use crate::resident_smc_v3::{
    RESIDENT_SMC_COLUMN_NAMES_V3, begin_resident_smc_store_v3, prepare_resident_smc_parent_v3,
};
use crate::{
    GeneDescriptor, NeoPopulationSettings, SMC_SLOTS, ScenarioDescriptor,
    acquire_discovery_run_device_admission_v1,
};
use neoethos_gpu_contracts::ABI_VERSION;
use neoethos_gpu_contracts::resident_feature_store_v3::ResidentWorkingSetRequestV3;
use std::sync::Arc;

const ROWS: usize = 160;
const CANDIDATE_ID: u64 = 7;
const SCENARIO_ID: u64 = 11;

fn fixture_search_plan() -> SealedResidentGenerationPlanV1 {
    seal_resident_generation_plan_v1(ResidentGenerationPlanAuthorityInputV1 {
        parent_selection: ParentSelectionPolicyV1::RankWeighted,
        survivor_selection: SurvivorSelectionPolicyV1::RankWeighted,
        max_terms_per_gene: 3,
        minimum_terms_per_gene: 1,
        logical_population_count: 1,
        retained_evaluation_capacity: 1,
        feature_count: RESIDENT_SMC_COLUMN_NAMES_V3.len(),
        generation_count: 1,
        survivor_count: 1,
        immigrant_count: 0,
        search_seed: 0x9d2c_a877_61e4_05b3,
        mutation_intensity_q32: 0,
        threshold_ladder_bits: std::array::from_fn(|index| {
            (0.05_f64 * (index as f64 + 1.0)).to_bits()
        }),
        stop_bounds_bits: std::array::from_fn(|index| (index as f64 + 1.0).to_bits()),
        smc_probability_q32: [0; SMC_SLOTS],
        generation_semantics_sha256: discovery_generation_semantics_sha256_v1(),
        run_identity_sha256: [0x71; 32],
        strategy_gene_schema_sha256: [0x72; 32],
        rank_semantics_sha256: rank_semantics_sha256_v2(),
        metric_semantics_sha256: resident_metric_semantics_sha256_v2(),
        scoring_semantics_sha256: scoring_semantics_sha256_v2(
            ResidentScoringObjectiveV2::PropFirmV4,
        ),
        novelty_semantics_sha256: novelty_disabled_semantics_sha256_v2(),
        scenario_order_semantics_sha256: [0x77; 32],
        cuda_build_manifest_sha256: [0x78; 32],
        rng_mapping_sha256: [0x79; 32],
    })
    .expect("fixture Search plan is sealed")
}

fn fixture_bindings() -> Vec<ResidentFeatureColumnBindingV3> {
    RESIDENT_SMC_COLUMN_NAMES_V3
        .iter()
        .enumerate()
        .map(|(ordinal, name)| ResidentFeatureColumnBindingV3 {
            ordinal,
            feature_name: (*name).to_owned(),
            canonical_parameter_tuple_sha256: [(ordinal + 1) as u8; 32],
            route_receipt_sha256: [(ordinal + 65) as u8; 32],
        })
        .collect()
}

fn fixture_ohlcv() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<i64>) {
    let mut open = Vec::with_capacity(ROWS);
    let mut high = Vec::with_capacity(ROWS);
    let mut low = Vec::with_capacity(ROWS);
    let mut close = Vec::with_capacity(ROWS);
    let mut volume = Vec::with_capacity(ROWS);
    let mut timestamps = Vec::with_capacity(ROWS);
    for row in 0..ROWS {
        let base = 1.08 + row as f64 * 0.000_01;
        let delta = match row % 4 {
            0 => 0.000_03,
            1 => -0.000_02,
            2 => 0.000_01,
            _ => -0.000_04,
        };
        let row_close = base + delta;
        open.push(base);
        high.push(base.max(row_close) + 0.000_07 + (row % 3) as f64 * 0.000_001);
        low.push(base.min(row_close) - 0.000_06 - (row % 5) as f64 * 0.000_001);
        close.push(row_close);
        volume.push(1_000.0 + row as f64 * 0.25);
        timestamps.push(1_704_067_200_000 + row as i64 * 300_000);
    }
    (open, high, low, close, volume, timestamps)
}

fn exact_working_set(
    run_device: &super::GpuOnlyRunDeviceAdmissionV3,
    bindings: &[ResidentFeatureColumnBindingV3],
    row_count: usize,
    retained_feature_device_bytes: usize,
) -> Result<
    neoethos_gpu_contracts::resident_feature_store_v3::ResidentWorkingSetBoundV3,
    Box<dyn std::error::Error>,
> {
    let pointer_table_bytes = bindings
        .len()
        .checked_mul(4 * std::mem::size_of::<u64>())
        .ok_or("pointer table byte overflow")?;
    let name_offset_bytes = bindings
        .len()
        .checked_add(1)
        .and_then(|count| count.checked_mul(std::mem::size_of::<u64>()))
        .ok_or("name offset byte overflow")?;
    let name_bytes = bindings.iter().try_fold(0_usize, |sum, binding| {
        sum.checked_add(binding.feature_name.len())
            .ok_or("feature name byte overflow")
    })?;
    Ok(ResidentWorkingSetRequestV3 {
        row_count,
        column_count: bindings.len(),
        max_live_producer_bytes: retained_feature_device_bytes as u64,
        max_live_producer_scratch_bytes: 0,
        normalization_scratch_bytes: 0,
        fit_metadata_bytes: 0,
        pointer_and_schema_metadata_bytes: pointer_table_bytes
            .checked_add(name_offset_bytes)
            .and_then(|bytes| bytes.checked_add(name_bytes))
            .ok_or("pointer/schema metadata byte overflow")?
            as u64,
        device_free_bytes_snapshot: run_device.phase_one_free_bytes_snapshot(),
        allocator_context_reserve_bytes: run_device.allocator_context_reserve_bytes(),
        reserve_policy_id: RESIDENT_ALLOCATOR_CONTEXT_RESERVE_POLICY_V3.to_owned(),
    }
    .seal()?)
}

fn wait_for_batch_retirement(
    assembler: &mut super::ResidentFeatureStoreAssemblerV3,
) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
    while !assembler.try_retire_completed_batch()? {
        std::thread::yield_now();
    }
    Ok(())
}

fn adaptive_failure_session(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    timestamps: &[i64],
) -> Result<ResidentPopulationSessionV3, Box<dyn std::error::Error>> {
    let admission = acquire_discovery_run_device_admission_v1()?;
    let run_device = seal_test_full_discovery_run_device_v3(admission, 4 * 1024 * 1024, 1024)?;
    let context = Arc::clone(run_device.primary_context_for_resident_producer_v3());
    let stream = Arc::clone(run_device.run_stream_for_resident_producer_v3());
    let ordinal = run_device.device_identity().ordinal();
    let bindings = fixture_bindings();
    let materialization = prepare_resident_smc_parent_v3(
        &run_device,
        open,
        high,
        low,
        close,
        volume,
        timestamps,
        bindings.clone(),
    )?;
    let working_set = exact_working_set(
        &run_device,
        &bindings,
        close.len(),
        materialization.receipt().retained_feature_device_bytes,
    )?;
    let (mut assembler, pending) =
        begin_resident_smc_store_v3(run_device, bindings, &working_set, materialization)?;
    pending.append_to(&mut assembler)?;
    wait_for_batch_retirement(&mut assembler)?;
    let owner = assembler.seal()?;
    loop {
        match owner.compact_hashes_if_ready() {
            Ok(_) => break,
            Err(ResidentFeatureStoreCudaErrorV3::NotReady) => std::thread::yield_now(),
            Err(error) => return Err(error.into()),
        }
    }
    let resident_import = owner.import_on_consumer_stream(context, stream, ordinal)?;
    Ok(resident_import.consume_into_population_session_v3()?)
}

/// Real Data+population admission for the compact Search fixture. Unlike the
/// older FullDiscovery test owner, this carries genuine checked population
/// limits, the selected device/build identity and the retired Data allocation.
#[cfg(feature = "cuda-device-fixtures")]
fn compact_three_generation_session_v3(
    population: usize,
    evaluation_capacity: usize,
) -> Result<ResidentPopulationSessionV3, Box<dyn std::error::Error>> {
    use crate::data_population_workspace_plan_v1::{
        DataPopulationWorkspacePreflightRequestV1, bind_data_population_gpu_workspace_plan_v1,
        native_cuda_data_population_preflight_facts_v1, seal_data_population_gpu_workspace_plan_v1,
    };
    use crate::run_device_admission_v1::SealedDiscoveryRunDeviceAdmissionV1;
    use neoethos_gpu_contracts::resident_feature_store_v3::ResidentWorkingSetExtentRequestV3;

    let admission = acquire_discovery_run_device_admission_v1()?;
    let SealedDiscoveryRunDeviceAdmissionV1::NativeCuda(native) = &admission else {
        return Err("compact three-generation fixture requires an actual CUDA device".into());
    };
    let facts = native_cuda_data_population_preflight_facts_v1(native);
    let bindings = fixture_bindings();
    let producer = crate::resident_smc_v3::preflight_resident_smc_memory_v4(ROWS)?;
    assert_eq!(producer.feature_column_count(), bindings.len());
    let feature_bytes = ROWS
        .checked_mul(bindings.len())
        .and_then(|cells| cells.checked_mul(9))
        .ok_or("SMC fixture value/validity extent overflow")?;
    let names = bindings.iter().try_fold(0_usize, |total, binding| {
        total
            .checked_add(binding.feature_name.len())
            .ok_or("SMC fixture names overflow")
    })?;
    let schema_bytes = bindings
        .len()
        .checked_mul(4 * 8)
        .and_then(|bytes| bytes.checked_add((bindings.len() + 1) * 8))
        .and_then(|bytes| bytes.checked_add(names))
        .ok_or("SMC fixture schema overflow")?;
    let extent_request = ResidentWorkingSetExtentRequestV3 {
        row_count: ROWS,
        column_count: bindings.len(),
        max_live_producer_bytes: feature_bytes as u64,
        max_live_producer_scratch_bytes: producer.scratch_bytes() as u64,
        normalization_scratch_bytes: 0,
        fit_metadata_bytes: 0,
        pointer_and_schema_metadata_bytes: schema_bytes as u64,
    };
    let plan =
        seal_data_population_gpu_workspace_plan_v1(DataPopulationWorkspacePreflightRequestV1 {
            native_admission_facts: facts,
            data_extent: extent_request.clone().seal()?,
            max_ordered_index_count: 0,
            max_adaptive_row_count: 0,
            gene_plan: crate::PopulationGeneStorePlanV1::checked_from_gene_extents_v1(
                population, population,
            )?,
            metrics_plan: crate::PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(
                evaluation_capacity,
                12,
            )?,
            classic_ta_capability:
                crate::resident_classic_ta_v3::resident_classic_ta_capability_v3()?,
        })?;
    let run_device = bind_data_population_gpu_workspace_plan_v1(admission, plan)?
        .into_gpu_only_run_device_admission_v3();
    let context = Arc::clone(run_device.primary_context_for_resident_producer_v3());
    let stream = Arc::clone(run_device.run_stream_for_resident_producer_v3());
    let ordinal = run_device.device_identity().ordinal();
    let working_set = ResidentWorkingSetRequestV3 {
        row_count: extent_request.row_count,
        column_count: extent_request.column_count,
        max_live_producer_bytes: extent_request.max_live_producer_bytes,
        max_live_producer_scratch_bytes: extent_request.max_live_producer_scratch_bytes,
        normalization_scratch_bytes: extent_request.normalization_scratch_bytes,
        fit_metadata_bytes: extent_request.fit_metadata_bytes,
        pointer_and_schema_metadata_bytes: extent_request.pointer_and_schema_metadata_bytes,
        device_free_bytes_snapshot: run_device.phase_one_free_bytes_snapshot(),
        allocator_context_reserve_bytes: run_device.allocator_context_reserve_bytes(),
        reserve_policy_id: RESIDENT_ALLOCATOR_CONTEXT_RESERVE_POLICY_V3.to_owned(),
    }
    .seal()?;
    let (open, high, low, close, volume, timestamps) = fixture_ohlcv();
    let materialization = prepare_resident_smc_parent_v3(
        &run_device,
        &open,
        &high,
        &low,
        &close,
        &volume,
        &timestamps,
        bindings.clone(),
    )?;
    assert_eq!(
        materialization.receipt().retained_feature_device_bytes,
        feature_bytes
    );
    assert_eq!(
        materialization.receipt().transient_device_bytes,
        producer.scratch_bytes()
    );
    let (mut assembler, pending) =
        begin_resident_smc_store_v3(run_device, bindings, &working_set, materialization)?;
    pending.append_to(&mut assembler)?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while !assembler.try_retire_completed_batch()? {
        if std::time::Instant::now() >= deadline {
            return Err("SMC retirement timed out".into());
        }
        std::thread::yield_now();
    }
    let owner = assembler.seal()?;
    loop {
        match owner.compact_hashes_if_ready() {
            Ok(_) => break,
            Err(ResidentFeatureStoreCudaErrorV3::NotReady) => {
                if std::time::Instant::now() >= deadline {
                    return Err("SMC content seal timed out".into());
                }
                std::thread::yield_now();
            }
            Err(error) => return Err(error.into()),
        }
    }
    let session = owner
        .import_on_consumer_stream(context, stream, ordinal)?
        .consume_into_population_session_v3()?;
    assert_eq!(
        session
            .data_population_limits()
            .expect("real population limits")
            .max_candidate_count(),
        population as u64
    );
    assert_eq!(
        session
            .data_population_limits()
            .expect("real population limits")
            .max_concurrent_scenario_count(),
        evaluation_capacity as u64
    );
    Ok(session)
}

#[cfg(feature = "cuda-device-fixtures")]
fn run_compact_generation_fixture_v3(
    losing_costs: bool,
    generations: usize,
    evaluation_capacity: usize,
    archive_capacity: u64,
) -> Result<
    crate::resident_archive_output_v3::ResidentSearchTerminalCandidatesV3,
    Box<dyn std::error::Error>,
> {
    use crate::resident_generation_v1::{
        ResidentAdaptiveGenerationInputsV3, ResidentGenerationTemplateV3,
        discovery_adaptive_generation_semantics_sha256_v3,
        seal_adaptive_resident_generation_plan_v3,
    };
    use crate::resident_search_slice2_v3::{
        ResidentSearchExecutionInputsV3, ResidentSearchExecutionPlanV3, ResidentSearchTryCompleteV3,
    };
    use neoethos_gpu_contracts::resident_search_scoring_v2::RiskyGrowthGoal;

    const POPULATION: usize = 12;
    let session = compact_three_generation_session_v3(POPULATION, evaluation_capacity)?;
    let (_, _, _, _, _, timestamps) = fixture_ohlcv();
    let plan = seal_adaptive_resident_generation_plan_v3(
        ResidentGenerationPlanAuthorityInputV1 {
            parent_selection: ParentSelectionPolicyV1::RankWeighted,
            survivor_selection: SurvivorSelectionPolicyV1::Elitist,
            max_terms_per_gene: 1,
            minimum_terms_per_gene: 1,
            logical_population_count: POPULATION,
            retained_evaluation_capacity: evaluation_capacity,
            feature_count: RESIDENT_SMC_COLUMN_NAMES_V3.len(),
            generation_count: generations,
            survivor_count: 9,
            immigrant_count: 0,
            search_seed: 0x9173_c025_dbea_486f,
            mutation_intensity_q32: 1_u64 << 32,
            threshold_ladder_bits: std::array::from_fn(|i| (0.05 * (i + 1) as f64).to_bits()),
            stop_bounds_bits: [100.0_f64, 101.0, 100.0, 101.0, 1.0, 2.0].map(f64::to_bits),
            smc_probability_q32: [0; SMC_SLOTS],
            generation_semantics_sha256: discovery_adaptive_generation_semantics_sha256_v3(),
            run_identity_sha256: [if losing_costs { 0xa1 } else { 0xa2 }; 32],
            strategy_gene_schema_sha256: [0xa3; 32],
            rank_semantics_sha256: rank_semantics_sha256_v2(),
            metric_semantics_sha256: resident_metric_semantics_sha256_v2(),
            scoring_semantics_sha256: scoring_semantics_sha256_v2(
                ResidentScoringObjectiveV2::RiskyGrowthGoalV6,
            ),
            novelty_semantics_sha256: novelty_disabled_semantics_sha256_v2(),
            scenario_order_semantics_sha256: [0xa4; 32],
            cuda_build_manifest_sha256: session.device_identity().gpu_cuda_build_sha256(),
            rng_mapping_sha256: [0xa5; 32],
        },
        ResidentAdaptiveGenerationInputsV3 {
            tournament_size: 3,
            min_structural_smc_flags: 0,
            adaptive_stops_enabled: false,
            seen_capacity: 128,
            seen_retry_attempts: 4,
            seed_template_count: 1,
            soft_stagnation_patience: 100,
            survivor_fraction: 0.75,
            immigrant_fraction: 0.0,
            selection_temperature: 1.0,
            minimum_improvement: 0.0,
            gate_start: 0.0,
            gate_end: 0.0,
            gate_curve: 1.0,
            gate_stagnation_step: 0.0,
            smc_force_ratio: 0.0,
            initial_seen_hashes: Vec::new(),
            templates: vec![ResidentGenerationTemplateV3 {
                feature_indices: vec![0],
                weights: vec![0.0],
                smc_flags: 0,
                long_threshold: -1.0,
                short_threshold: -2.0,
                target_pips: 100.0,
                stop_pips: 100.0,
                stop_vol_multiplier: 0.0,
            }],
        },
    )
    .map_err(|error| format!("seal adaptive generation fixture: {error:?}"))?;
    let settings = NeoPopulationSettings {
        abi_version: ABI_VERSION,
        initial_equity: 100.0,
        pip_value: 0.0001,
        pip_value_per_lot: 10.0,
        commission_per_trade: if losing_costs { 1_000.0 } else { 0.0 },
        min_hold_bars: 1,
        max_hold_bars: 1,
        month_capacity: 12,
        gap_threshold_ms: 600_000,
        risk_per_trade_min: 0.0,
        risk_per_trade_max: 0.01,
        high_quality_confidence: 0.75,
        ..NeoPopulationSettings::default()
    };
    let execution = ResidentSearchExecutionPlanV3::for_compact_session_v3(
        &session,
        plan,
        ResidentSearchExecutionInputsV3 {
            settings,
            scenarios: (0..POPULATION)
                .map(|i| ScenarioDescriptor {
                    base_candidate_id: i as u64,
                    scenario_id: i as u64,
                    window_len: ROWS as u32,
                    ..ScenarioDescriptor::default()
                })
                .collect(),
            smc_weights: [1.0; SMC_SLOTS],
            smc_gate_disabled: true,
            growth_objective: true,
            growth_goal: Some(RiskyGrowthGoal {
                start_balance: 100.0,
                target_balance: 500.0,
                horizon_days: 180.0,
            }),
            stage1_row_start: 0,
            stage1_row_end: ROWS as u64,
            first_timestamp_ms: timestamps[0],
            last_timestamp_ms: timestamps[ROWS - 1],
            novelty_weight: 0.35,
            archive_capacity,
            adaptive_base_request: None,
            archive_policy: Some(
                crate::resident_search_slice2_v3::ResidentSearchArchivePolicyV3 {
                    mode: 0,
                    neighbors: 15,
                    min_net: 0.0,
                    min_pf: 1.0,
                    min_sharpe: 0.0,
                },
            ),
        },
    )?;
    let mut chain = session.begin_resident_search_slice2_v3(execution);
    for _ in 0..generations {
        chain = chain
            .enqueue_score_and_rank_v3()
            .map_err(|e| format!("rank: {e:?}"))?
            .enqueue_stage_archive_from_rank_v3()
            .map_err(|e| format!("archive: {e:?}"))?
            .enqueue_evolve_and_publish_v3()
            .map_err(|e| format!("evolve: {e:?}"))?;
    }
    let mut pending = chain
        .enqueue_terminal_seal_v3()
        .map_err(|e| format!("terminal: {e:?}"))?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        match pending
            .try_complete_v3()
            .map_err(|e| format!("completion: {e:?}"))?
        {
            ResidentSearchTryCompleteV3::Complete(receipt) => {
                return Ok(receipt.into_terminal_candidates_v3());
            }
            ResidentSearchTryCompleteV3::NotReady(next) => {
                if std::time::Instant::now() >= deadline {
                    return Err("three-generation terminal timed out".into());
                }
                pending = next;
                std::thread::yield_now();
            }
        }
    }
}

#[test]
#[cfg(feature = "cuda-device-fixtures")]
fn compact_slice2_three_generations_export_archive_and_last_evaluated_population_on_device()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(
        std::env::var("NEOETHOS_REQUIRE_GPU").as_deref(),
        Ok("1"),
        "this test requires real CUDA, not a host-only pass"
    );
    // P=12/C=5 exercises two full chunks and a short final chunk; admission
    // still owns all twelve genes and only five evaluation workspaces.
    let output = run_compact_generation_fixture_v3(false, 3, 5, 36)?;
    let full = run_compact_generation_fixture_v3(false, 3, 12, 36)?;
    let one = run_compact_generation_fixture_v3(false, 3, 5, 1)?;
    let archive = output.archive();
    let population = output.population();
    assert_eq!(archive.terminal_generation(), 3);
    assert_eq!(population.evaluated_generation(), 2);
    assert_eq!(population.len(), 12);
    assert_eq!(population.host_copy_count(), 4);
    assert_eq!(population.host_copy_bytes(), 12 * (176 + 16));
    assert!(
        !archive.is_empty(),
        "actual profitable template must reach the archive"
    );
    assert_eq!(archive.host_copy_count(), 5);
    assert_eq!(archive.host_copy_bytes(), archive.len() as u64 * 440);
    // With 9 of 12 preserved per generation, at least 6 original genes remain
    // after two reproduction steps. Their birth generation must not be reset.
    assert!(
        population
            .candidates()
            .filter(|gene| gene.generation() == 0)
            .count()
            >= 6
    );
    assert!(population.candidates().any(|gene| gene.generation() == 2));
    assert!(population.candidates().all(|gene| gene.generation() <= 2));
    for gene in population.candidates() {
        assert_eq!(gene.metric_row().candidate_id, gene.gene_identity());
        assert!(
            gene.metric_row()
                .values
                .iter()
                .all(|value| value.is_finite())
        );
    }
    assert!(
        archive
            .candidates()
            .any(|gene| gene.generation() == 0 && gene.metric_row().values[0] > 0.0)
    );
    assert_eq!(population.len(), full.population().len());
    assert_eq!(archive.len(), full.archive().len());
    // Capacity is genuinely reached. Retention must not affect population-only
    // novelty or reproduction, and must retain a complete best-net observation.
    // This does not claim that this market fixture forces a hash collision or
    // eviction; the deterministic real-device index fixture below covers those.
    assert_eq!(one.archive().len(), 1);
    assert_eq!(one.archive().host_copy_count(), 5);
    assert_eq!(one.archive().host_copy_bytes(), 440);
    assert_eq!(one.population().len(), population.len());
    assert_eq!(one.population().host_copy_count(), 4);
    assert_eq!(
        one.population().host_copy_bytes(),
        population.host_copy_bytes()
    );
    let one_winner = one.archive().candidates().next().unwrap();
    let best_net = archive
        .candidates()
        .map(|gene| gene.metric_row().values[0])
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(best_net.is_finite());
    assert_eq!(one_winner.metric_row().values[0], best_net);
    let reference_winner = archive
        .candidates()
        .find(|gene| {
            gene.gene_identity() == one_winner.gene_identity()
                && gene.content_hash() == one_winner.content_hash()
                && gene.metric_row().values.map(f64::to_bits)
                    == one_winner.metric_row().values.map(f64::to_bits)
        })
        .expect("capacity-one winner must be one actual whole observation, not mixed metadata");
    for (chunked, single) in population
        .candidates()
        .zip(full.population().candidates())
        .chain(archive.candidates().zip(full.archive().candidates()))
        .chain(one.population().candidates().zip(population.candidates()))
        .chain(std::iter::once((one_winner, reference_winner)))
    {
        assert_eq!(chunked.gene_identity(), single.gene_identity());
        assert_eq!(chunked.content_hash(), single.content_hash());
        assert_eq!(chunked.generation(), single.generation());
        assert_eq!(chunked.smc_flags(), single.smc_flags());
        assert_eq!(chunked.indices(), single.indices());
        assert_eq!(
            chunked
                .weights()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            single
                .weights()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            [
                chunked.long_threshold(),
                chunked.short_threshold(),
                chunked.target_pips(),
                chunked.stop_pips(),
                chunked.stop_vol_multiplier()
            ]
            .map(f64::to_bits),
            [
                single.long_threshold(),
                single.short_threshold(),
                single.target_pips(),
                single.stop_pips(),
                single.stop_vol_multiplier()
            ]
            .map(f64::to_bits)
        );
        assert_eq!(
            chunked.metric_row().candidate_id,
            single.metric_row().candidate_id
        );
        assert_eq!(
            chunked.metric_row().scenario_id,
            single.metric_row().scenario_id
        );
        assert_eq!(
            chunked.metric_row().values.map(f64::to_bits),
            single.metric_row().values.map(f64::to_bits)
        );
    }
    Ok(())
}

#[test]
#[cfg(feature = "cuda-device-fixtures")]
fn compact_slice2_indexed_archive_collision_eviction_and_reinsertion_on_device()
-> Result<(), Box<dyn std::error::Error>> {
    use crate::run_device_admission_v1::SealedDiscoveryRunDeviceAdmissionV1;
    unsafe extern "C" {
        fn fixture_check_adaptive_archive_index_v3(device: u32, passed_checks: *mut u64) -> i32;
    }
    assert_eq!(
        std::env::var("NEOETHOS_REQUIRE_GPU").as_deref(),
        Ok("1"),
        "this test requires the actual CUDA index/heap kernel, not a host-only pass"
    );
    let admission = acquire_discovery_run_device_admission_v1()?;
    let SealedDiscoveryRunDeviceAdmissionV1::NativeCuda(native) = &admission else {
        return Err("indexed archive fixture requires an actual CUDA device".into());
    };
    let mut passed_checks = 0_u64;
    // This explicitly synthetic algorithm fixture calls the production device
    // helpers. It creates no run, metric receipt, trade or promotion evidence.
    let status =
        unsafe { fixture_check_adaptive_archive_index_v3(native.ordinal, &mut passed_checks) };
    assert_eq!(
        status, 0,
        "actual index/heap fixture failed with mask {passed_checks:#x}"
    );
    assert_eq!(
        passed_checks, 0xff,
        "all eight collision, erase/reinsert, heap/tie, improvement and corruption checks must run"
    );
    Ok(())
}

#[test]
#[cfg(feature = "cuda-device-fixtures")]
fn compact_slice2_three_generations_keep_economic_rejections_out_of_batch_faults_on_device()
-> Result<(), Box<dyn std::error::Error>> {
    use neoethos_gpu_contracts::resident_search_scoring_v2::{
        ResidentScoringOutcomeV2, RiskyGrowthGoal, checked_resident_goal_score_v6,
    };
    assert_eq!(
        std::env::var("NEOETHOS_REQUIRE_GPU").as_deref(),
        Ok("1"),
        "this test requires real CUDA, not a host-only pass"
    );
    // Observe the genuine initial rejected template before selection can
    // legitimately remove it. The same costs/genome then run three generations;
    // no claim is made that every final candidate must still trade or reject.
    let initial = run_compact_generation_fixture_v3(true, 1, 5, 36)?;
    let output = run_compact_generation_fixture_v3(true, 3, 5, 36)?;
    assert_eq!(output.population().evaluated_generation(), 2);
    assert_eq!(output.population().len(), 12);
    assert_eq!(output.population().host_copy_count(), 4);
    assert!(output.archive().is_empty());
    assert_eq!(output.archive().host_copy_count(), 0);
    let (_, _, _, _, _, timestamps) = fixture_ohlcv();
    let span_days = (timestamps[ROWS - 1] - timestamps[0]) as f64 / 86_400_000.0;
    let goal = RiskyGrowthGoal {
        start_balance: 100.0,
        target_balance: 500.0,
        horizon_days: 180.0,
    };
    assert!(output.population().candidates().all(|gene| !matches!(
        checked_resident_goal_score_v6(&gene.metric_row().values, 100.0, span_days, goal),
        ResidentScoringOutcomeV2::Fault(_)
    )));
    assert!(
        initial.population().candidates().any(|gene| {
            let metrics = gene.metric_row().values;
            metrics[0] < -100.0
                && matches!(
                    checked_resident_goal_score_v6(&metrics, 100.0, span_days, goal),
                    ResidentScoringOutcomeV2::EconomicReject(_)
                )
        }),
        "a genuine cost-induced equity loss must survive as a rejected candidate, not abort the run"
    );
    Ok(())
}

fn assert_adaptive_evaluation_fails_closed(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    timestamps: &[i64],
    pip_size: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut session = adaptive_failure_session(open, high, low, close, volume, timestamps)?;
    let view = PopulationEvaluationViewV1::full(ROWS, PopulationTimestampModeV1::Canonical, None)?;
    let request = ResidentAdaptiveBaseRequestV1::checked_canonical_v1(&view, pip_size, 1, 0)?;
    session.bind_evaluation_view_with_resident_adaptive_base_v1(view, request)?;

    let descriptors = [GeneDescriptor {
        candidate_id: CANDIDATE_ID,
        term_offset: 0,
        term_count: 1,
        long_threshold: 1.0e300,
        short_threshold: -1.0e300,
        stop_ticks: 100,
        target_ticks: 200,
        stop_vol_multiplier: 1.0,
        flags: 0,
        reserved: 0,
    }];
    let offsets = [0_i32, 1];
    let indices = [0_i32];
    let weights = [0.0_f64];
    let stop_pips = [10.0_f64];
    let target_pips = [20.0_f64];
    let stop_vol_multipliers = [1.0_f64];
    let smc_flags = [0_i8; SMC_SLOTS];
    let smc_weights = [0.0_f64; SMC_SLOTS];
    session.upload_genes(PopulationGeneView {
        descriptors: &descriptors,
        offsets: &offsets,
        indices: &indices,
        weights: &weights,
        stop_pips: &stop_pips,
        target_pips: &target_pips,
        stop_vol_multipliers: &stop_vol_multipliers,
        smc_flags: &smc_flags,
        smc_weights: &smc_weights,
        gate_threshold: 1.0e300,
        smc_gate_disabled: true,
    })?;
    session.upload_scenarios(&[ScenarioDescriptor {
        base_candidate_id: 0,
        scenario_id: SCENARIO_ID,
        window_offset: 0,
        window_len: ROWS as u32,
        ..ScenarioDescriptor::default()
    }])?;
    let settings = NeoPopulationSettings {
        abi_version: ABI_VERSION,
        max_hold_bars: 8,
        min_hold_bars: 1,
        max_trades_per_day: 1,
        month_capacity: 12,
        gap_threshold_ms: 600_000,
        initial_equity: 100_000.0,
        pip_value: 0.000_1,
        spread_pips: 1.0,
        commission_per_trade: 7.0,
        pip_value_per_lot: 10.0,
        risk_per_trade_min: 0.005,
        risk_per_trade_max: 0.01,
        high_quality_confidence: 0.75,
        spread_pips_asian: 1.0,
        spread_pips_overlap: 1.0,
        spread_pips_late_ny: 1.0,
        ..NeoPopulationSettings::default()
    };
    let result = session
        .enqueue_metrics_only_v1(&settings)?
        .consume_terminal_compact_result_v1();
    match result {
        Err(CudaPopulationError::Native { status, .. }) => {
            assert_eq!(status, STATUS_ADAPTIVE_BASE_DEGENERATE);
            Ok(())
        }
        Err(error) => Err(format!("unexpected adaptive failure: {error}").into()),
        Ok(_) => Err("degenerate resident adaptive base silently produced metrics".into()),
    }
}

#[test]
fn resident_adaptive_constant_candles_fail_closed_before_metrics_are_accepted()
-> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("NEOETHOS_REQUIRE_GPU").is_none() {
        eprintln!("skipping required-card fixture because NEOETHOS_REQUIRE_GPU is absent");
        return Ok(());
    }
    let open = vec![1.08_f64; ROWS];
    let high = vec![1.08_f64; ROWS];
    let low = vec![1.08_f64; ROWS];
    let close = vec![1.08_f64; ROWS];
    let volume = vec![1_000.0_f64; ROWS];
    let timestamps = (0..ROWS)
        .map(|row| 1_704_067_200_000 + row as i64 * 300_000)
        .collect::<Vec<_>>();
    assert_adaptive_evaluation_fails_closed(
        &open,
        &high,
        &low,
        &close,
        &volume,
        &timestamps,
        0.000_1,
    )
}

#[test]
fn resident_population_subpip_cash_risk_and_unavailable_entries_match_closed_form()
-> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("NEOETHOS_REQUIRE_GPU").is_none() {
        eprintln!("skipping required-card fixture because NEOETHOS_REQUIRE_GPU is absent");
        return Ok(());
    }
    // One causal entry, then a both-hit bar resolved stop-first. This is a
    // device regression against a cash-risk formula, not whole-pipeline parity.
    let open = vec![1.0; ROWS];
    let close = vec![1.0; ROWS];
    let high = vec![1.000_1; ROWS];
    let low = vec![0.999_9; ROWS];
    let volume = vec![1_000.0; ROWS];
    let timestamps = (0..ROWS)
        .map(|row| 1_704_067_200_000 + row as i64 * 300_000)
        .collect::<Vec<_>>();
    for (risk, adaptive_base, multiplier, expected_trades) in [
        (0.01, None, 0.0, 1.0),
        (0.0, None, 0.0, 0.0),
        (0.01, Some(0.0), 1.0, 0.0),
        (0.01, Some(f64::MAX), 2.0, 0.0),
    ] {
        let mut session =
            adaptive_failure_session(&open, &high, &low, &close, &volume, &timestamps)?;
        session.bind_evaluation_view_v1(PopulationEvaluationViewV1::full(
            ROWS,
            PopulationTimestampModeV1::Canonical,
            adaptive_base.map(|base| Arc::<[f64]>::from(vec![base; ROWS])),
        )?)?;
        let descriptors = [GeneDescriptor {
            candidate_id: CANDIDATE_ID,
            term_offset: 0,
            term_count: 1,
            long_threshold: -1.0,
            short_threshold: -2.0,
            stop_ticks: 250,
            target_ticks: 500,
            stop_vol_multiplier: multiplier,
            flags: 0,
            reserved: 0,
        }];
        session.upload_genes(PopulationGeneView {
            descriptors: &descriptors,
            offsets: &[0, 1],
            indices: &[0],
            weights: &[0.0],
            stop_pips: &[0.25],
            target_pips: &[0.5],
            stop_vol_multipliers: &[multiplier],
            smc_flags: &[0; SMC_SLOTS],
            smc_weights: &[0.0; SMC_SLOTS],
            gate_threshold: 0.0,
            smc_gate_disabled: true,
        })?;
        session.upload_scenarios(&[ScenarioDescriptor {
            base_candidate_id: 0,
            scenario_id: SCENARIO_ID,
            window_offset: 0,
            window_len: 4,
            ..ScenarioDescriptor::default()
        }])?;
        let settings = NeoPopulationSettings {
            abi_version: ABI_VERSION,
            flags: neoethos_gpu_contracts::POPULATION_SETTINGS_FLAG_RISK_BASED_SIZING,
            min_hold_bars: 1,
            max_trades_per_day: 1,
            month_capacity: 12,
            initial_equity: 10_000.0,
            pip_value: 0.000_1,
            pip_value_per_lot: 10.0,
            risk_per_trade_min: risk,
            risk_per_trade_max: risk,
            high_quality_confidence: 0.75,
            adaptive_rr: 2.0,
            ..NeoPopulationSettings::default()
        };
        let result = session
            .enqueue_metrics_only_v1(&settings)?
            .consume_terminal_compact_result_v1()?;
        let metrics = result.metric_row();
        assert_eq!(metrics.candidate_id, CANDIDATE_ID);
        assert_eq!(metrics.scenario_id, SCENARIO_ID);
        assert_eq!(metrics.values[8], expected_trades);
        // 1% of 10000 / (0.25 pips * 10 currency/pip/lot) = 40 lots.
        // No costs/carry; a full stop loses exactly 100 account-currency units.
        let expected_net = -risk * settings.initial_equity * expected_trades;
        assert!((metrics.values[0] - expected_net).abs() <= 1.0e-7);
        assert!((metrics.values[6] - expected_net).abs() <= 1.0e-7);
        assert_eq!(result.terminal_readback_bytes(), 104);
        let counters = session.read_residency_counters_v1()?;
        assert_eq!(counters.parent_upload_count(), 0);
        assert_eq!(counters.diagnostic_readback_count(), 0);
        let lease = session.record_consumer_completion()?;
        while !lease.completion_is_ready()? {
            std::thread::yield_now();
        }
        drop(lease);
    }
    Ok(())
}

#[cfg(feature = "cuda-device-fixtures")]
fn read_adaptive_fixture_view(
    session: &mut ResidentPopulationSessionV3,
    view: PopulationEvaluationViewV1,
    tail_step: usize,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let request =
        ResidentAdaptiveBaseRequestV1::checked_canonical_v1(&view, 0.000_1, tail_step, 0)?;
    session.bind_evaluation_view_with_resident_adaptive_base_v1(view, request)?;
    Ok(session.copy_resident_adaptive_base_fixture_v1()?)
}

#[cfg(feature = "cuda-device-fixtures")]
fn retire_adaptive_fixture(
    session: ResidentPopulationSessionV3,
) -> Result<(), Box<dyn std::error::Error>> {
    let lease = session.record_consumer_completion()?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while !lease.completion_is_ready()? {
        if std::time::Instant::now() >= deadline {
            return Err("adaptive fixture retirement timed out".into());
        }
        std::thread::yield_now();
    }
    Ok(())
}

#[cfg(feature = "cuda-device-fixtures")]
fn assert_same_adaptive_cells(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        if expected.is_nan() {
            assert!(
                actual.is_nan(),
                "row {row} became tradable before its history existed"
            );
        } else {
            assert!(expected.is_finite() && *expected > 0.0);
            assert_eq!(actual.to_bits(), expected.to_bits(), "adaptive row {row}");
        }
    }
}

#[cfg(feature = "cuda-device-fixtures")]
#[test]
fn resident_adaptive_ordered_views_recompute_on_the_gathered_sequence_and_are_prefix_causal()
-> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(
        std::env::var("NEOETHOS_REQUIRE_GPU").as_deref(),
        Ok("1"),
        "requires actual CUDA kernels, never a skipped host-only pass"
    );
    let (open, high, low, close, volume, timestamps) = fixture_ohlcv();
    // Two disjoint chronological blocks. Local rolling windows and adjacent
    // returns must use this gathered sequence, not precomputed parent stops.
    let ordered: Arc<[u64]> = (0_u64..60).chain(80..140).collect::<Vec<_>>().into();
    let prefix: Arc<[u64]> = ordered[..110].to_vec().into();
    let make_ordered = |indices: Arc<[u64]>| {
        PopulationEvaluationViewV1::ordered_indices(
            ROWS,
            indices,
            PopulationTimestampModeV1::DisabledIndexDelta,
            None,
        )
    };
    let gather = |values: &[f64]| {
        ordered
            .iter()
            .map(|row| values[*row as usize])
            .collect::<Vec<_>>()
    };
    let gathered_timestamps = ordered
        .iter()
        .map(|row| timestamps[*row as usize])
        .collect::<Vec<_>>();
    for tail_step in [1, 7] {
        let mut session =
            adaptive_failure_session(&open, &high, &low, &close, &volume, &timestamps)?;
        let parent_values = read_adaptive_fixture_view(
            &mut session,
            PopulationEvaluationViewV1::full(
                ROWS,
                PopulationTimestampModeV1::DisabledIndexDelta,
                None,
            )?,
            tail_step,
        )?;
        let prefix_values = read_adaptive_fixture_view(
            &mut session,
            make_ordered(Arc::clone(&prefix))?,
            tail_step,
        )?;
        let ordered_values = read_adaptive_fixture_view(
            &mut session,
            make_ordered(Arc::clone(&ordered))?,
            tail_step,
        )?;
        assert!(ordered_values[..100].iter().all(|value| value.is_nan()));
        assert!(
            ordered_values[100..]
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
        );
        assert_same_adaptive_cells(&ordered_values[..prefix.len()], &prefix_values);
        assert!(
            (100..ordered.len()).any(|row| {
                ordered_values[row].to_bits() != parent_values[ordered[row] as usize].to_bits()
            }),
            "negative control must detect incorrectly gathering the parent stop series"
        );

        // Rebind before any readback/completion barrier. Each map's native
        // staging must outlive its async copy, even when the Rust view changes.
        // The pending map has no external Arc owner and differs from the final
        // map. Replacing it must not leave a DMA borrowing released Rust data.
        let pending_view = make_ordered((1_u64..111).collect::<Vec<_>>().into())?;
        let pending_request = ResidentAdaptiveBaseRequestV1::checked_canonical_v1(
            &pending_view,
            0.000_1,
            tail_step,
            0,
        )?;
        session
            .bind_evaluation_view_with_resident_adaptive_base_v1(pending_view, pending_request)?;
        let rebound = read_adaptive_fixture_view(
            &mut session,
            make_ordered(Arc::clone(&ordered))?,
            tail_step,
        )?;
        assert_same_adaptive_cells(&rebound, &ordered_values);
        let counters = session.read_residency_counters_v1()?;
        assert_eq!(counters.full_binding_count(), 1);
        assert_eq!(counters.ordered_binding_count(), 4);
        assert_eq!(
            counters.ordered_index_upload_bytes(),
            ((2 * prefix.len() + 2 * ordered.len()) * 8) as u64
        );
        assert_eq!(counters.adaptive_upload_bytes(), 0);
        assert_eq!(counters.parent_upload_count(), 0);
        assert_eq!(counters.parent_upload_bytes(), 0);
        assert_eq!(counters.stream_creation_count(), 0);
        retire_adaptive_fixture(session)?;

        // Independent *device layout* control: a new fixture parent physically
        // contains only the gathered HLC. This is not a CPU fallback or a claim
        // of complete CPU/GPU parity; fixed CPU checkpoints are checked above.
        let mut gathered_session = adaptive_failure_session(
            &gather(&open),
            &gather(&high),
            &gather(&low),
            &gather(&close),
            &gather(&volume),
            &gathered_timestamps,
        )?;
        let gathered_values = read_adaptive_fixture_view(
            &mut gathered_session,
            PopulationEvaluationViewV1::full(
                ordered.len(),
                PopulationTimestampModeV1::DisabledIndexDelta,
                None,
            )?,
            tail_step,
        )?;
        assert_same_adaptive_cells(&ordered_values, &gathered_values);
        retire_adaptive_fixture(gathered_session)?;
    }
    Ok(())
}

#[test]
fn resident_adaptive_tiny_positive_pip_overflow_fails_closed_before_metrics_are_accepted()
-> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("NEOETHOS_REQUIRE_GPU").is_none() {
        eprintln!("skipping required-card fixture because NEOETHOS_REQUIRE_GPU is absent");
        return Ok(());
    }
    let (open, high, low, close, volume, timestamps) = fixture_ohlcv();
    assert_adaptive_evaluation_fails_closed(
        &open,
        &high,
        &low,
        &close,
        &volume,
        &timestamps,
        f64::from_bits(1),
    )
}

#[test]
fn resident_adaptive_checked_bind_validates_the_current_token_and_poison_rejects_uploads()
-> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("NEOETHOS_REQUIRE_GPU").is_none() {
        eprintln!("skipping required-card fixture because NEOETHOS_REQUIRE_GPU is absent");
        return Ok(());
    }
    let (open, high, low, close, volume, timestamps) = fixture_ohlcv();

    let mut session_a = adaptive_failure_session(&open, &high, &low, &close, &volume, &timestamps)?;
    let view_a =
        PopulationEvaluationViewV1::full(ROWS, PopulationTimestampModeV1::Canonical, None)?;
    let request_a = ResidentAdaptiveBaseRequestV1::checked_canonical_v1(&view_a, 0.000_1, 1, 0)?;
    let rejected = session_a.bind_evaluation_view_with_resident_adaptive_base_checked_v1(
        view_a,
        request_a,
        |_current_token| {
            Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                "receipt rejected the exact current token".into(),
            ))
        },
    );
    assert!(matches!(
        rejected,
        Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(message))
            if message == "receipt rejected the exact current token"
    ));
    let rejected_upload = session_a.upload_scenarios(&[ScenarioDescriptor {
        base_candidate_id: 0,
        scenario_id: SCENARIO_ID,
        window_offset: 0,
        window_len: ROWS as u32,
        ..ScenarioDescriptor::default()
    }]);
    assert!(matches!(
        rejected_upload,
        Err(ResidentFeatureStoreCudaErrorV3::Population(
            CudaPopulationError::Native {
                status: STATUS_STRICT_RESIDENT_POISONED,
                ..
            }
        ))
    ));
    let lease_a = session_a.record_consumer_completion()?;
    while !lease_a.completion_is_ready()? {
        std::thread::yield_now();
    }
    drop(lease_a);

    let mut session_b = adaptive_failure_session(&open, &high, &low, &close, &volume, &timestamps)?;
    let view_b =
        PopulationEvaluationViewV1::full(ROWS, PopulationTimestampModeV1::Canonical, None)?;
    let request_b = ResidentAdaptiveBaseRequestV1::checked_canonical_v1(&view_b, 0.000_1, 1, 0)?;
    let mut validator_calls = 0_u32;
    let accepted = session_b.bind_evaluation_view_with_resident_adaptive_base_checked_v1(
        view_b,
        request_b,
        |current_token| {
            validator_calls += 1;
            assert_eq!(
                current_token.request_identity_sha256(),
                request_b.identity_sha256(),
            );
            Ok(())
        },
    )?;
    assert_eq!(validator_calls, 1);
    assert_eq!(
        accepted.request_identity_sha256(),
        request_b.identity_sha256()
    );
    assert_ne!(accepted.token_identity_sha256(), [0; 32]);
    let lease_b = session_b.record_consumer_completion()?;
    while !lease_b.completion_is_ready()? {
        std::thread::yield_now();
    }
    drop(lease_b);

    let mut session_c = adaptive_failure_session(&open, &high, &low, &close, &volume, &timestamps)?;
    let view_c =
        PopulationEvaluationViewV1::full(ROWS, PopulationTimestampModeV1::Canonical, None)?;
    let request_c = ResidentAdaptiveBaseRequestV1::checked_canonical_v1(&view_c, 0.000_1, 1, 0)?;
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = session_c.bind_evaluation_view_with_resident_adaptive_base_checked_v1(
            view_c,
            request_c,
            |_current_token| -> Result<(), ResidentFeatureStoreCudaErrorV3> {
                panic!("validator panic fixture")
            },
        );
    }));
    assert!(unwind.is_err());
    let upload_after_unwind = session_c.upload_scenarios(&[ScenarioDescriptor {
        base_candidate_id: 0,
        scenario_id: SCENARIO_ID,
        window_offset: 0,
        window_len: ROWS as u32,
        ..ScenarioDescriptor::default()
    }]);
    assert!(matches!(
        upload_after_unwind,
        Err(ResidentFeatureStoreCudaErrorV3::Population(
            CudaPopulationError::Native {
                status: STATUS_STRICT_RESIDENT_POISONED,
                ..
            }
        ))
    ));
    let lease_c = session_c.record_consumer_completion()?;
    while !lease_c.completion_is_ready()? {
        std::thread::yield_now();
    }
    drop(lease_c);
    Ok(())
}

#[cfg(feature = "cuda-device-fixtures")]
#[test]
fn resident_store_v3_terminal_metrics_only_path_is_one_session_and_leak_free()
-> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("NEOETHOS_REQUIRE_GPU").is_none() {
        eprintln!("skipping required-card fixture because NEOETHOS_REQUIRE_GPU is absent");
        return Ok(());
    }

    let admission = acquire_discovery_run_device_admission_v1()?;
    let probes = admission.probe_counters();
    assert_eq!(probes.physical_inventory_probe_count(), 1);
    assert_eq!(probes.cuda_enumeration_count(), 1);
    assert_eq!(probes.primary_context_acquisition_count(), 1);
    assert_eq!(probes.run_stream_creation_count(), 1);

    let run_device = seal_test_full_discovery_run_device_v3(admission, 4 * 1024 * 1024, 1024)?;
    let context = Arc::clone(run_device.primary_context_for_resident_producer_v3());
    let stream = Arc::clone(run_device.run_stream_for_resident_producer_v3());
    let ordinal = run_device.device_identity().ordinal();
    let bindings = fixture_bindings();
    let (open, high, low, close, volume, timestamps) = fixture_ohlcv();
    let materialization = prepare_resident_smc_parent_v3(
        &run_device,
        &open,
        &high,
        &low,
        &close,
        &volume,
        &timestamps,
        bindings.clone(),
    )?;
    let smc_receipt = materialization.receipt();
    assert_eq!(smc_receipt.row_count, ROWS);
    assert_eq!(smc_receipt.feature_column_count, bindings.len());
    assert_eq!(smc_receipt.parent_smc_slot_count, SMC_SLOTS);
    assert_eq!(smc_receipt.producer_ready_event_count, 1);
    assert_eq!(smc_receipt.compact_control_plane_d2h_bytes, 100);
    let working_set = exact_working_set(
        &run_device,
        &bindings,
        close.len(),
        smc_receipt.retained_feature_device_bytes,
    )?;

    let (mut assembler, pending) =
        begin_resident_smc_store_v3(run_device, bindings.clone(), &working_set, materialization)?;
    pending.append_to(&mut assembler)?;
    wait_for_batch_retirement(&mut assembler)?;
    let owner = assembler.seal()?;
    let hashes = loop {
        match owner.compact_hashes_if_ready() {
            Ok(hashes) => break hashes,
            Err(ResidentFeatureStoreCudaErrorV3::NotReady) => std::thread::yield_now(),
            Err(error) => return Err(error.into()),
        }
    };
    let layout = owner.layout_evidence(&hashes);
    assert_eq!(layout.rows, ROWS);
    assert_eq!(layout.columns, bindings.len());
    assert_eq!(layout.producer_batch_count, 1);
    assert_eq!(layout.full_feature_major_staging_bytes, 0);

    let resident_import = owner.import_on_consumer_stream(context, stream, ordinal)?;
    let mut session = resident_import.consume_into_population_session_v3()?;
    session.bind_evaluation_view_v1(PopulationEvaluationViewV1::full(
        ROWS,
        PopulationTimestampModeV1::Canonical,
        None,
    )?)?;
    session.bind_evaluation_view_v1(PopulationEvaluationViewV1::contiguous_range(
        ROWS,
        8,
        56,
        PopulationTimestampModeV1::Canonical,
        None,
    )?)?;
    let ordered: Arc<[u64]> = (0_u64..ROWS as u64).step_by(2).collect::<Vec<_>>().into();
    let ordered_rows = ordered.len();
    session.bind_evaluation_view_v1(PopulationEvaluationViewV1::ordered_indices(
        ROWS,
        Arc::clone(&ordered),
        PopulationTimestampModeV1::Canonical,
        None,
    )?)?;
    let adaptive_view =
        PopulationEvaluationViewV1::full(ROWS, PopulationTimestampModeV1::Canonical, None)?;
    let adaptive_request =
        ResidentAdaptiveBaseRequestV1::checked_canonical_v1(&adaptive_view, 0.000_1, 1, 0)?;
    session.bind_evaluation_view_with_resident_adaptive_base_v1(adaptive_view, adaptive_request)?;
    let adaptive_base = session.copy_resident_adaptive_base_fixture_v1()?;
    assert_eq!(adaptive_base.len(), ROWS);
    // The former whole-vector digest blessed fabricated positive warm-up
    // stops. Keep the independently recorded ready CPU checkpoints, and test
    // causal unavailability explicitly instead of regenerating that digest.
    assert!(adaptive_base[..100].iter().all(|value| value.is_nan()));
    assert!(
        adaptive_base[100..]
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
    );
    let checkpoint_bits = [100_usize, 101, 159].map(|index| adaptive_base[index].to_bits());
    assert_eq!(
        checkpoint_bits,
        [
            0x4001_0fae_b68c_45f7,
            0x4001_055c_7dca_6cbb,
            0x4001_0569_3d50_cb03,
        ],
    );

    let descriptors = [GeneDescriptor {
        candidate_id: CANDIDATE_ID,
        term_offset: 0,
        term_count: 1,
        long_threshold: 1.0e300,
        short_threshold: -1.0e300,
        stop_ticks: 100,
        target_ticks: 200,
        stop_vol_multiplier: 0.0,
        flags: 0,
        reserved: 0,
    }];
    let offsets = [0_i32, 1];
    let indices = [0_i32];
    let weights = [0.0_f64];
    let stop_pips = [10.0_f64];
    let target_pips = [20.0_f64];
    let stop_vol_multipliers = [0.0_f64];
    let smc_flags = [0_i8; SMC_SLOTS];
    let smc_weights = [0.0_f64; SMC_SLOTS];
    session.upload_genes(PopulationGeneView {
        descriptors: &descriptors,
        offsets: &offsets,
        indices: &indices,
        weights: &weights,
        stop_pips: &stop_pips,
        target_pips: &target_pips,
        stop_vol_multipliers: &stop_vol_multipliers,
        smc_flags: &smc_flags,
        smc_weights: &smc_weights,
        gate_threshold: 1.0e300,
        smc_gate_disabled: true,
    })?;
    session.upload_scenarios(&[ScenarioDescriptor {
        base_candidate_id: 0,
        scenario_id: SCENARIO_ID,
        window_offset: 0,
        window_len: ordered_rows as u32,
        ..ScenarioDescriptor::default()
    }])?;
    let settings = NeoPopulationSettings {
        abi_version: ABI_VERSION,
        max_hold_bars: 8,
        min_hold_bars: 1,
        max_trades_per_day: 1,
        month_capacity: 12,
        gap_threshold_ms: 600_000,
        initial_equity: 100_000.0,
        pip_value: 0.000_1,
        spread_pips: 1.0,
        commission_per_trade: 7.0,
        pip_value_per_lot: 10.0,
        risk_per_trade_min: 0.005,
        risk_per_trade_max: 0.01,
        high_quality_confidence: 0.75,
        spread_pips_asian: 1.0,
        spread_pips_overlap: 1.0,
        spread_pips_late_ny: 1.0,
        ..NeoPopulationSettings::default()
    };
    let terminal = session
        .enqueue_metrics_only_v1(&settings)?
        .consume_terminal_compact_result_v1()?;
    assert_eq!(terminal.metric_row().candidate_id, CANDIDATE_ID);
    assert_eq!(terminal.metric_row().scenario_id, SCENARIO_ID);
    assert_eq!(
        terminal.metric_row().values,
        [0.0, 0.0, 100_000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(terminal.scenario_count(), 1);
    assert_eq!(terminal.terminal_synchronization_count(), 1);
    assert_eq!(terminal.terminal_readback_count(), 1);
    assert_eq!(terminal.terminal_readback_rows(), 1);
    assert_eq!(terminal.terminal_readback_bytes(), 104);
    assert_ne!(terminal.receipt_identity_sha256(), [0; 32]);

    // A completed strict launch leaves scenario arrays and a metrics-only
    // workspace resident. Uploading the next generation's genes must retire
    // those transients before allocating the new unsplittable gene store; the
    // following multi-scenario launch proves the session remains usable after
    // that exact transition.
    session.upload_genes(PopulationGeneView {
        descriptors: &descriptors,
        offsets: &offsets,
        indices: &indices,
        weights: &weights,
        stop_pips: &stop_pips,
        target_pips: &target_pips,
        stop_vol_multipliers: &stop_vol_multipliers,
        smc_flags: &smc_flags,
        smc_weights: &smc_weights,
        gate_threshold: 1.0e300,
        smc_gate_disabled: true,
    })?;

    let multi_scenarios = [
        ScenarioDescriptor {
            base_candidate_id: 0,
            scenario_id: SCENARIO_ID + 1,
            window_offset: 0,
            window_len: ordered_rows as u32,
            ..ScenarioDescriptor::default()
        },
        ScenarioDescriptor {
            base_candidate_id: 0,
            scenario_id: SCENARIO_ID + 2,
            window_offset: 0,
            window_len: ordered_rows as u32,
            ..ScenarioDescriptor::default()
        },
        ScenarioDescriptor {
            base_candidate_id: 0,
            scenario_id: SCENARIO_ID + 3,
            window_offset: 0,
            window_len: ordered_rows as u32,
            ..ScenarioDescriptor::default()
        },
    ];
    session.upload_scenarios(&multi_scenarios)?;
    let host_metrics = session
        .enqueue_metrics_only_v1(&settings)?
        .consume_host_metrics_v1()?;
    assert_eq!(host_metrics.scenario_count(), 3);
    assert_eq!(host_metrics.terminal_synchronization_count(), 1);
    assert_eq!(host_metrics.terminal_readback_count(), 1);
    assert_eq!(host_metrics.terminal_readback_rows(), 3);
    assert_eq!(host_metrics.terminal_readback_bytes(), 3 * 104);
    assert_ne!(host_metrics.receipt_identity_sha256(), [0; 32]);
    assert_eq!(host_metrics.counters().synchronization_events, 1);
    assert_eq!(host_metrics.counters().full_readback_bytes, 3 * 104);
    let metric_rows = host_metrics.into_metric_rows();
    assert_eq!(metric_rows.len(), multi_scenarios.len());
    for (row, scenario) in metric_rows.iter().zip(multi_scenarios) {
        assert_eq!(row.candidate_id, CANDIDATE_ID);
        assert_eq!(row.scenario_id, scenario.scenario_id);
        assert_eq!(
            row.values,
            [0.0, 0.0, 100_000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    let counters = session.read_residency_counters_v1()?;
    assert_eq!(counters.parent_upload_count(), 0);
    assert_eq!(counters.parent_upload_bytes(), 0);
    assert_eq!(counters.view_binding_count(), 4);
    assert_eq!(counters.full_binding_count(), 2);
    assert_eq!(counters.range_binding_count(), 1);
    assert_eq!(counters.ordered_binding_count(), 1);
    assert_eq!(
        counters.ordered_index_upload_bytes(),
        (ordered_rows * std::mem::size_of::<u64>()) as u64
    );
    assert_eq!(counters.adaptive_upload_bytes(), 0);
    assert_eq!(counters.stream_creation_count(), 0);
    assert_eq!(counters.explicit_synchronization_count(), 2);
    assert_eq!(counters.metric_rows_readback_count(), 1);
    assert_eq!(counters.metric_rows_readback_rows(), 3);
    assert_eq!(counters.metric_rows_readback_bytes(), 3 * 104);
    assert_eq!(counters.diagnostic_readback_count(), 1);
    assert_eq!(counters.diagnostic_readback_rows(), ROWS as u64);
    assert_eq!(
        counters.diagnostic_readback_bytes(),
        (ROWS * std::mem::size_of::<f64>()) as u64
    );
    assert_eq!(counters.accepted_trade_total_readback_count(), 0);

    let lease = session.record_consumer_completion()?;
    while !lease.completion_is_ready()? {
        std::thread::yield_now();
    }
    assert_eq!(lease.rows(), ROWS);
    assert_eq!(lease.columns(), bindings.len());
    drop(lease);
    drop(owner);
    Ok(())
}

#[cfg(feature = "cuda-device-fixtures")]
#[test]
fn resident_store_v3_moves_into_search_v2_and_enqueues_on_real_cuda()
-> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("NEOETHOS_REQUIRE_GPU").is_none() {
        eprintln!("skipping required-card fixture because NEOETHOS_REQUIRE_GPU is absent");
        return Ok(());
    }

    let readiness = resident_search_v2_production_readiness();
    assert!(readiness.device_owned_search_control());
    assert!(readiness.native_bridge_production_sealed());
    assert!(!readiness.terminal_cleanup_lease());
    assert!(!readiness.exact_generation_semantics());
    assert!(!readiness.production_ready());

    let (open, high, low, close, volume, timestamps) = fixture_ohlcv();
    let mut session = adaptive_failure_session(&open, &high, &low, &close, &volume, &timestamps)?;
    session.bind_evaluation_view_v1(PopulationEvaluationViewV1::full(
        ROWS,
        PopulationTimestampModeV1::Canonical,
        None,
    )?)?;

    let plan = fixture_search_plan();

    let mut search = session.consume_into_resident_search_run_v2(plan, [1.0; SMC_SLOTS], true)?;
    search.upload_resident_scenarios_v2(&[ScenarioDescriptor {
        base_candidate_id: 0,
        scenario_id: SCENARIO_ID,
        window_offset: 0,
        window_len: ROWS as u32,
        ..ScenarioDescriptor::default()
    }])?;
    let settings = NeoPopulationSettings {
        abi_version: ABI_VERSION,
        max_hold_bars: 8,
        min_hold_bars: 1,
        max_trades_per_day: 1,
        month_capacity: 12,
        gap_threshold_ms: 600_000,
        initial_equity: 100_000.0,
        pip_value: 0.000_1,
        spread_pips: 1.0,
        commission_per_trade: 7.0,
        pip_value_per_lot: 10.0,
        risk_per_trade_min: 0.005,
        risk_per_trade_max: 0.01,
        high_quality_confidence: 0.75,
        spread_pips_asian: 1.0,
        spread_pips_overlap: 1.0,
        spread_pips_late_ny: 1.0,
        ..NeoPopulationSettings::default()
    };
    let host_metrics = search
        .enqueue_resident_gene_metrics_fixture_v2(&settings)?
        .consume_host_metrics_v1()?;
    assert_eq!(host_metrics.scenario_count(), 1);
    assert_eq!(host_metrics.terminal_synchronization_count(), 1);
    assert_eq!(host_metrics.terminal_readback_count(), 1);
    assert_eq!(host_metrics.terminal_readback_rows(), 1);
    assert_eq!(host_metrics.terminal_readback_bytes(), 104);
    assert_ne!(host_metrics.receipt_identity_sha256(), [0; 32]);
    let counters = host_metrics.counters();
    assert_eq!(counters.gene_upload_bytes, 0);
    assert!(counters.scenario_upload_bytes > 0);
    assert_eq!(counters.synchronization_events, 1);
    assert_eq!(counters.full_readback_bytes, 104);
    let rows = host_metrics.metric_rows();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].candidate_id, 0);
    assert_eq!(rows[0].scenario_id, SCENARIO_ID);
    assert!(rows[0].values.iter().all(|value| value.is_finite()));

    let lease = search.record_consumer_completion()?;
    while !lease.completion_is_ready()? {
        std::thread::yield_now();
    }
    assert_eq!(lease.rows(), ROWS);
    assert_eq!(lease.columns(), RESIDENT_SMC_COLUMN_NAMES_V3.len());
    drop(lease);
    Ok(())
}

#[test]
fn resident_store_v3_search_start_failure_returns_event_owned_recovery_carrier()
-> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("NEOETHOS_REQUIRE_GPU").is_none() {
        eprintln!("skipping required-card fixture because NEOETHOS_REQUIRE_GPU is absent");
        return Ok(());
    }
    let (open, high, low, close, volume, timestamps) = fixture_ohlcv();
    let session = adaptive_failure_session(&open, &high, &low, &close, &volume, &timestamps)?;
    let error = match session.consume_into_resident_search_run_v2(
        fixture_search_plan(),
        [0.0; SMC_SLOTS],
        true,
    ) {
        Ok(_) => return Err("zero SMC weights unexpectedly started Search".into()),
        Err(error) => error,
    };
    assert!(matches!(
        &error,
        ResidentFeatureStoreSearchStartErrorV2::Search { .. }
    ));
    let lease = error
        .into_cleanup_lease()
        .ok_or("Search start failure did not return its recovery lease")?;
    while !lease.completion_is_ready()? {
        std::thread::yield_now();
    }
    assert_eq!(lease.rows(), ROWS);
    assert_eq!(lease.columns(), RESIDENT_SMC_COLUMN_NAMES_V3.len());
    drop(lease);
    Ok(())
}
