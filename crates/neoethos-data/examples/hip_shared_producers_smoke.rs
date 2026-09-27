//! Real-device Session-v2 + SMC-v3 smoke on ONE uploaded OHLCV owner.
//! Exact comparison uses existing production Rust CPU oracles, not independent
//! mathematical truth. A selected mixed-family pack also checks the existing
//! canonical Merkle oracle and the shared adaptive resident generation chain.
//! No device => failure; no whole-pipeline parity, financial admission or OOS claim.

#[path = "support/hip_producer_smoke_v1.rs"]
mod support;
use anyhow::{Context as _, Result, ensure};
use chrono::{Datelike, TimeZone, Utc};
use neoethos_data::core::dataset_manifest::{
    DatasetTimestampRange, ProducerProvenanceEnvelopeV1, PublishRequest, publish_vortex_generation,
};
use neoethos_data::core::features::FeatureColumnF64;
use neoethos_data::core::gpu_hip_feature_store_v1::{
    PreparedHipCanonicalFeatureStoreV1, ResidentHipCanonicalFeatureStoreV1,
    ResidentHipCanonicalPopulationV1,
};
use neoethos_data::core::gpu_hip_ohlcv_v1::PreparedHipOhlcvV1;
use neoethos_data::core::gpu_hip_session_v1::{HIP_SESSION_COLUMN_NAMES_V1, ResidentHipSessionV1};
use neoethos_data::core::gpu_hip_smc_v1::{
    HIP_SMC_COLUMN_NAMES_V1, HipSmcMemoryPlanV1, ResidentHipSmcV1,
};
use neoethos_data::core::normalization::{
    SearchNormalizationFittedStateV1, normalize_search_feature_column_f64,
};
use neoethos_data::core::session_features::compute_session_feature_columns_f64;
use neoethos_data::core::smc::compute_smc_feature_columns_f64;
use neoethos_data::{CanonicalOhlcvFrame, Ohlcv};
use neoethos_dataset_contracts::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe,
};
use neoethos_gpu_contracts::ABI_VERSION;
use neoethos_gpu_contracts::device::{GeneDescriptor, NeoPopulationSettings, ScenarioDescriptor};
use neoethos_gpu_contracts::resident_feature_store_v3::{
    canonical_feature_merkle_sha256_host_oracle_v3, pack_logical_validity_u4_v3,
};
use neoethos_gpu_contracts::resident_search_scoring_v2::{
    ResidentScoringOutcomeV2, RiskyGrowthGoal, checked_resident_goal_score_v6,
};
use neoethos_gpu_cuda::hip_runtime_v1::{HipRunLeaseV1, hip_native_build_manifest_v1};
use neoethos_gpu_cuda::resident_search_slice2_v3::{
    HipResidentSearchTryCompleteV3, ResidentSearchArchivePolicyV3, ResidentSearchExecutionInputsV3,
};
use neoethos_gpu_cuda::{
    ParentSelectionPolicyV1, PopulationEvaluationViewV1, PopulationGeneView,
    PopulationTimestampModeV1, ResidentAdaptiveGenerationInputsV3,
    ResidentGenerationPlanAuthorityInputV1, ResidentGenerationTemplateV3,
    ResidentScoringObjectiveV2, SMC_SLOTS, SurvivorSelectionPolicyV1,
    discovery_adaptive_generation_semantics_sha256_v3, novelty_disabled_semantics_sha256_v2,
    rank_semantics_sha256_v2, resident_metric_semantics_sha256_v2, scoring_semantics_sha256_v2,
    seal_adaptive_resident_generation_plan_v3,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use support::{compare_terminal, hex};

const SCHEMA: &str = "neoethos.data.hip-shared-producers-smoke.v1";
const ROWS: usize = 129;
// Deliberately interleaved, non-prefix selection. This is a fixture recipe,
// never a production default or a claim that only these families exist.
const SELECTED: [(bool, usize); 5] = [(false, 22), (true, 1), (false, 0), (true, 45), (false, 5)];
// All have valid training support even in the deliberate zero-volume fixture.
// The raw recipe above intentionally also exercises unsupported VWAP cells.
const NORMALIZED_SELECTED: [(bool, usize); 5] =
    [(false, 13), (true, 1), (false, 17), (true, 12), (true, 45)];
const FIXTURE_TRAINING_END: usize = 103; // floor(129 * 0.8), never the holdout.
const FIXTURE_ALLOCATOR_RESERVE: u64 = 1 << 20;
const GA_POPULATION: usize = 12;
const GA_CHUNK: usize = 5;
const GA_GENERATIONS: usize = 3;
const GA_ARCHIVE_CAPACITY: u64 = 36;
const GA_MONTH_CAPACITY: u32 = 2;
const GA_SEED: u64 = 0x9173_c025_dbea_486f;

fn fixture_scenarios() -> Vec<ScenarioDescriptor> {
    // Canonical full-P order, not C scenarios or a truncated logical population.
    (0..GA_POPULATION)
        .map(|candidate| ScenarioDescriptor {
            base_candidate_id: candidate as u64,
            scenario_id: candidate as u64,
            window_len: FIXTURE_TRAINING_END as u32,
            ..Default::default()
        })
        .collect()
}

fn fixture_goal() -> RiskyGrowthGoal {
    RiskyGrowthGoal {
        start_balance: 100.0,
        target_balance: 500.0,
        horizon_days: 180.0,
    }
}

fn canonical_fixture(id: &str, input: &Ohlcv) -> Result<(tempfile::TempDir, CanonicalOhlcvFrame)> {
    let root = tempfile::tempdir()?;
    let identity = CanonicalDatasetIdentity::external(
        "hip-shared-smoke",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )?;
    let provenance = ProducerProvenanceEnvelopeV1::new(
        "neoethos.hip-shared-smoke.synthetic.v1",
        format!("synthetic fixture {id}; not market, strategy, or OOS evidence").into_bytes(),
    )?;
    let times = input.timestamp.as_ref().context("fixture clock absent")?;
    ensure!(!times.is_empty(), "fixture must contain rows");
    publish_vortex_generation(PublishRequest {
        configured_root: root.path(),
        identity: &identity,
        expected_generation: None,
        timestamp_range: DatasetTimestampRange::new(times[0], times[times.len() - 1])?,
        provenance: &provenance,
        chunks: neoethos_data::ohlcv_to_vortex_chunks(input, 37)?,
    })?;
    let frame = neoethos_data::load_canonical_timeframe(root.path(), &identity)?;
    Ok((root, frame))
}

fn selected_cpu_root(
    input: &Ohlcv,
    session: &[FeatureColumnF64],
    smc: &[FeatureColumnF64],
    selection: &[(bool, usize)],
    normalize: bool,
) -> Result<([u8; 32], Option<SearchNormalizationFittedStateV1>)> {
    let mut columns = selection
        .iter()
        .map(|&(is_smc, index)| {
            (if is_smc { smc } else { session })
                .get(index)
                .cloned()
                .context("CPU selected index out of range")
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        columns
            .iter()
            .all(|c| c.values.len() == input.len() && c.validity.len() == input.len()),
        "CPU selected column extent drift"
    );
    let names = columns
        .iter()
        .map(|column| column.name.clone())
        .collect::<Vec<_>>();
    let fitted = if normalize {
        ensure!(
            input.len() == ROWS,
            "normalization fixture geometry changed"
        );
        let fits = columns
            .iter_mut()
            .map(|column| normalize_search_feature_column_f64(column, 0..FIXTURE_TRAINING_END))
            .collect::<Result<Vec<_>>>()?;
        Some(SearchNormalizationFittedStateV1::new(names.clone(), fits)?)
    } else {
        None
    };
    let mut bits = Vec::with_capacity(input.len() * columns.len());
    let mut validity = Vec::with_capacity(bits.capacity());
    for row in 0..input.len() {
        for column in &columns {
            bits.push(column.values[row].to_bits());
            validity.push(column.validity[row].code());
        }
    }
    let root = canonical_feature_merkle_sha256_host_oracle_v3(
        input.timestamp.as_deref().context("fixture clock absent")?,
        &names,
        &bits,
        &pack_logical_validity_u4_v3(&validity)?,
    )?;
    Ok((root, fitted))
}

fn fixture(flat: bool) -> Ohlcv {
    const MINUTES: [i64; 16] = [
        0, 1, 419, 420, 421, 479, 480, 590, 600, 719, 720, 790, 959, 960, 1260, 1439,
    ];
    let mut result = Ohlcv {
        timestamp: Some(Vec::new()),
        open: Vec::new(),
        high: Vec::new(),
        low: Vec::new(),
        close: Vec::new(),
        volume: Some(Vec::new()),
    };
    for row in 0..ROWS {
        result
            .timestamp
            .as_mut()
            .unwrap()
            .push(1_704_067_200_000 + ((row / 16) as i64 * 1440 + MINUTES[row % 16]) * 60_000);
        let open = if flat {
            8.0
        } else {
            8.0 + (row % 17) as f64 / 4.0
        };
        let close = open
            + if flat {
                0.0
            } else if row % 2 == 0 {
                0.125
            } else {
                -0.125
            };
        result.open.push(open);
        result.close.push(close);
        result
            .high
            .push(if flat { open } else { open.max(close) + 0.0625 });
        result
            .low
            .push(if flat { open } else { open.min(close) - 0.0625 });
        result
            .volume
            .as_mut()
            .unwrap()
            .push(if flat || row % 5 == 0 {
                0.0
            } else {
                1.0 + (row % 3) as f64
            });
    }
    result
}

fn check_parent_terminal(
    input: &Ohlcv,
    months: &[u8],
    days: &[u8],
    slots: &[u8],
    hashes: &[[u8; 32]; 3],
) -> Result<()> {
    ensure!(
        months.len() == input.len() * 8
            && days.len() == input.len() * 8
            && slots.len() == input.len() * 11,
        "SMC terminal parent shape mismatch"
    );
    for (i, bytes) in [months, days, slots].into_iter().enumerate() {
        ensure!(
            <[u8; 32]>::from(Sha256::digest(bytes)) == hashes[i],
            "SMC device-generated parent hash differs from terminal bytes at lane {i}"
        );
    }
    for (row, timestamp) in input
        .timestamp
        .as_ref()
        .context("fixture clock absent")?
        .iter()
        .enumerate()
    {
        let date = Utc
            .timestamp_millis_opt(*timestamp)
            .single()
            .context("fixture calendar invalid")?;
        let month = i64::from(date.year()) * 12 + i64::from(date.month());
        let day =
            i64::from(date.year()) * 10_000 + i64::from(date.month()) * 100 + i64::from(date.day());
        ensure!(
            i64::from_le_bytes(months[row * 8..row * 8 + 8].try_into()?) == month,
            "SMC month identity differs at row {row}"
        );
        ensure!(
            i64::from_le_bytes(days[row * 8..row * 8 + 8].try_into()?) == day,
            "SMC day identity differs at row {row}"
        );
    }
    ensure!(
        slots.iter().all(|byte| (-1..=1).contains(&(*byte as i8))),
        "SMC slot outside signed ternary domain"
    );
    Ok(())
}

fn check_resident_generations(
    parent: &mut ResidentHipCanonicalPopulationV1<'_, '_, '_>,
    canonical: &ResidentHipCanonicalFeatureStoreV1<'_, '_>,
    input: &Ohlcv,
    id: &str,
    flat: bool,
) -> Result<()> {
    let names = canonical.physical_plan().ordered_names();
    let fvg = names
        .iter()
        .position(|name| name == "smc_fvg")
        .context("GA FVG gate absent")?;
    let timestamps = input.timestamp.as_ref().context("GA clock absent")?;
    ensure!(timestamps.len() == ROWS, "GA parent extent changed");
    let first = timestamps[0];
    let last = timestamps[FIXTURE_TRAINING_END - 1];
    let span_days = (last - first) as f64 / 86_400_000.0;
    ensure!(span_days > 0.0, "GA evaluation time span is empty");
    let manifest = hip_native_build_manifest_v1().context("actual HIP native manifest absent")?;
    let build_hash: [u8; 32] = Sha256::digest(manifest.as_bytes()).into();
    // These are explicitly diagnostic identities, not invented Data/Search
    // financial authority. The physical parent retains actual source/fit/lease
    // authority; the common sealer additionally binds every plan/policy field.
    let request_bytes = serde_json::to_vec(&json!({
        "schema":"neoethos.hip-shared-smoke.adaptive-request.v1", "id":id,
        "assembly":hex(&parent.assembly_identity_sha256()), "names":names,
        "population":GA_POPULATION, "chunk":GA_CHUNK, "generations":GA_GENERATIONS,
        "seed":GA_SEED, "view":[0,FIXTURE_TRAINING_END], "timestamps":[first,last],
        "novelty_weight":0.35, "archive_mode":"active", "archive_capacity":GA_ARCHIVE_CAPACITY,
        "scope":"synthetic-physical-integration-only"
    }))?;
    let plan = seal_adaptive_resident_generation_plan_v3(
        ResidentGenerationPlanAuthorityInputV1 {
            parent_selection: ParentSelectionPolicyV1::RankWeighted,
            survivor_selection: SurvivorSelectionPolicyV1::Elitist,
            max_terms_per_gene: 1,
            minimum_terms_per_gene: 1,
            logical_population_count: GA_POPULATION,
            retained_evaluation_capacity: GA_CHUNK,
            feature_count: names.len(),
            generation_count: GA_GENERATIONS,
            survivor_count: 9,
            immigrant_count: 0,
            search_seed: GA_SEED,
            mutation_intensity_q32: 1u64 << 32,
            threshold_ladder_bits: [0.05_f64, 0.10, 0.15, 0.20, 0.25, 0.30].map(f64::to_bits),
            stop_bounds_bits: [100.0_f64, 101.0, 100.0, 101.0, 1.0, 2.0].map(f64::to_bits),
            smc_probability_q32: [0; SMC_SLOTS],
            generation_semantics_sha256: discovery_adaptive_generation_semantics_sha256_v3(),
            run_identity_sha256: Sha256::digest(&request_bytes).into(),
            strategy_gene_schema_sha256: *canonical.feature_plan().identity().as_bytes(),
            rank_semantics_sha256: rank_semantics_sha256_v2(),
            metric_semantics_sha256: resident_metric_semantics_sha256_v2(),
            scoring_semantics_sha256: scoring_semantics_sha256_v2(
                ResidentScoringObjectiveV2::RiskyGrowthGoalV6,
            ),
            // The raw scorer does not blend; the shared archive path performs
            // the configured population-neighbor novelty pass at weight 0.35.
            novelty_semantics_sha256: novelty_disabled_semantics_sha256_v2(),
            scenario_order_semantics_sha256: Sha256::digest(
                b"diagnostic-full-P-base-and-scenario-ordinal-v1",
            )
            .into(),
            hip_native_build_manifest_sha256: build_hash,
            rng_mapping_sha256: Sha256::digest(
                b"diagnostic-shared-native-philox-adaptive-algorithm1-not-CPU-RNG-parity",
            )
            .into(),
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
            gate_start: 0.125,
            gate_end: 0.375,
            gate_curve: 1.0,
            gate_stagnation_step: 0.0,
            smc_force_ratio: 0.0,
            templates: vec![ResidentGenerationTemplateV3 {
                feature_indices: vec![fvg as u64],
                weights: vec![0.0],
                smc_flags: 0,
                long_threshold: -1.0,
                short_threshold: -2.0,
                target_pips: 100.0,
                stop_pips: 100.0,
                stop_vol_multiplier: 0.0,
            }],
            initial_seen_hashes: vec![],
        },
    )
    .map_err(|error| anyhow::anyhow!("diagnostic adaptive generation plan: {error:?}"))?;
    let settings = NeoPopulationSettings {
        abi_version: ABI_VERSION,
        initial_equity: 100.0,
        pip_value: 0.01,
        pip_value_per_lot: 1.0,
        max_hold_bars: 1,
        min_hold_bars: 1,
        max_trades_per_day: 1,
        month_capacity: GA_MONTH_CAPACITY,
        gap_threshold_ms: 600_000,
        risk_per_trade_min: 0.0,
        risk_per_trade_max: 0.01,
        high_quality_confidence: 0.75,
        ..Default::default()
    };
    let identity = parent.runtime_identity().clone();
    let mut chain = parent.begin_resident_search_slice2_v3(
        plan,
        ResidentSearchExecutionInputsV3 {
            settings,
            scenarios: fixture_scenarios(),
            smc_weights: [1.0; SMC_SLOTS],
            smc_gate_disabled: true,
            growth_objective: true,
            growth_goal: Some(fixture_goal()),
            stage1_row_start: 0,
            stage1_row_end: FIXTURE_TRAINING_END as u64,
            first_timestamp_ms: first,
            last_timestamp_ms: last,
            novelty_weight: 0.35,
            archive_capacity: GA_ARCHIVE_CAPACITY,
            adaptive_base_request: None,
            archive_policy: Some(ResidentSearchArchivePolicyV3 {
                mode: 1,
                neighbors: 15,
                min_net: 0.0,
                min_pf: 1.0,
                min_sharpe: 0.0,
            }),
        },
    )?;
    for generation in 0..GA_GENERATIONS {
        chain = chain
            .enqueue_score_and_rank_v3()?
            .enqueue_stage_archive_from_rank_v3()?
            .enqueue_evolve_and_publish_v3()?;
        let (next, checkpoint) = chain.checkpoint_v3()?;
        ensure!(
            checkpoint.evaluated_generation() == generation as u64
                && checkpoint.evaluated_generations() == (generation + 1) as u64
                && checkpoint.evaluation_slots() == ((generation + 1) * GA_POPULATION) as u64
                && checkpoint.evaluated_gate() == 0.125 * (generation + 1) as f64,
            "device checkpoint omitted candidates/generations or changed gate schedule"
        );
        println!(
            "{}",
            json!({"type":"resident-generation-checkpoint", "schema":SCHEMA,"id":id,
            "evaluated_generation":checkpoint.evaluated_generation(),"evaluation_slots":checkpoint.evaluation_slots(),
            "gate":checkpoint.evaluated_gate(),"stagnant_generations":checkpoint.stagnant_generations(),
            "control_copy_count":checkpoint.control_copy_count(),"control_copy_bytes":checkpoint.control_copy_bytes(),
            "initial_control_upload_count":checkpoint.initial_upload_count(),"initial_control_upload_bytes":checkpoint.initial_upload_bytes()})
        );
        chain = next;
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let mut pending = chain.enqueue_terminal_seal_v3()?;
    let (output, checkpoint) = loop {
        match pending.try_complete_v3()? {
            HipResidentSearchTryCompleteV3::NotReady(next) => {
                ensure!(
                    std::time::Instant::now() < deadline,
                    "HIP GA terminal timed out; no successful device claim"
                );
                pending = next;
                std::thread::yield_now();
            }
            HipResidentSearchTryCompleteV3::Complete(output, checkpoint) => {
                break (output, checkpoint);
            }
        }
    };
    ensure!(
        parent.runtime_identity() == &identity,
        "GA did not restore the same HIP parent lease"
    );
    let checkpoint = checkpoint.context("terminal lost actual adaptive checkpoint")?;
    let population = output.population();
    let archive = output.archive();
    ensure!(
        population.len() == GA_POPULATION
            && population.evaluated_generation() == 2
            && archive.terminal_generation() == 3
            && checkpoint.evaluation_slots() == 36,
        "HIP terminal omitted the complete last evaluated population"
    );
    ensure!(
        population.host_copy_count() == 4
            && population.host_copy_bytes() == 12 * 192
            && !archive.is_empty()
            && archive.host_copy_count() == 5
            && archive.host_copy_bytes() == archive.len() as u64 * 440,
        "HIP terminal changed compact transfer counts or lost the active template archive"
    );
    ensure!(
        population
            .candidates()
            .filter(|gene| gene.generation() == 0)
            .count()
            >= 6
            && population.candidates().any(|gene| gene.generation() == 2),
        "HIP survivor ancestry or actual reproduction was lost"
    );
    let scenario_ids = population
        .candidates()
        .map(|gene| gene.metric_row().scenario_id)
        .collect::<std::collections::BTreeSet<_>>();
    ensure!(
        scenario_ids == (0..GA_POPULATION as u64).collect(),
        "terminal scenario identities do not cover all P"
    );
    for gene in population.candidates().chain(archive.candidates()) {
        ensure!(
            gene.metric_row().candidate_id == gene.gene_identity()
                && gene.generation() <= 2
                && gene.indices().len() == 1
                && gene.weights().len() == 1
                && gene.indices()[0] < names.len() as u64
                && !matches!(
                    checked_resident_goal_score_v6(
                        &gene.metric_row().values,
                        100.0,
                        span_days,
                        fixture_goal()
                    ),
                    ResidentScoringOutcomeV2::Fault(_)
                ),
            "HIP terminal whole-observation/metric validation failed"
        );
        if flat {
            ensure!(
                gene.metric_row().values[0] == 0.0,
                "flat zero-cost GA violates independent zero-PnL arithmetic"
            );
        }
    }
    println!(
        "{}",
        json!({"type":"resident-adaptive-ga","schema":SCHEMA,"id":id,
        "population":population.len(),"chunk":GA_CHUNK,"generations":GA_GENERATIONS,
        "evaluation_slots":checkpoint.evaluation_slots(),"archive_members":archive.len(),
        "population_copy_count":population.host_copy_count(),"population_copy_bytes":population.host_copy_bytes(),
        "archive_copy_count":archive.host_copy_count(),"archive_copy_bytes":archive.host_copy_bytes(),
        "last_evaluated_generation":population.evaluated_generation(),"lease_id":identity.lease_id(),
        "stream_id":identity.stream_id(),"native_build_manifest_sha256":hex(&build_hash),
        "candidate_ids":population.candidates().map(|gene| gene.gene_identity()).collect::<Vec<_>>(),
        "metric_value_bits":population.candidates().map(|gene| gene.metric_row().values.map(f64::to_bits)).collect::<Vec<_>>(),
        "known_flat_zero_cost_arithmetic_checked":flat,"cpu_cuda_hip_ga_parity":false,
        "financial_search_admission":false,"trading_or_oos_proof":false,"device_executed":true})
    );
    Ok(())
}

fn check_resident_population(
    population: &mut ResidentHipCanonicalPopulationV1<'_, '_, '_>,
    canonical: &ResidentHipCanonicalFeatureStoreV1<'_, '_>,
    id: &str,
    flat: bool,
) -> Result<()> {
    let fvg = canonical
        .physical_plan()
        .ordered_names()
        .iter()
        .position(|name| name == "smc_fvg")
        .context("fixture lacks its FVG gate")?;
    let descriptors = [
        GeneDescriptor {
            candidate_id: 501,
            term_count: 1,
            long_threshold: f64::MAX,
            short_threshold: -f64::MAX,
            ..Default::default()
        },
        GeneDescriptor {
            candidate_id: 602,
            term_offset: 1,
            term_count: 1,
            long_threshold: -0.5,
            short_threshold: -1.0,
            ..Default::default()
        },
    ];
    let indices = [i32::try_from(fvg)?; 2];
    let genes = PopulationGeneView {
        descriptors: &descriptors,
        offsets: &[0, 1, 2],
        indices: &indices,
        weights: &[1.0; 2],
        stop_pips: &[10.0; 2],
        target_pips: &[10.0; 2],
        stop_vol_multipliers: &[0.0; 2],
        smc_flags: &[0; 2 * SMC_SLOTS],
        smc_weights: &[1.0; SMC_SLOTS],
        gate_threshold: 0.0,
        smc_gate_disabled: true,
    };
    let scenarios = [(1, 9003, 2_000_000), (0, 9001, 0), (1, 9002, 0)].map(
        |(base_candidate_id, scenario_id, commission_micros)| ScenarioDescriptor {
            base_candidate_id,
            scenario_id,
            commission_micros,
            window_len: FIXTURE_TRAINING_END as u32,
            ..Default::default()
        },
    );
    // Synthetic diagnostic economics, never a production risk configuration.
    let settings = NeoPopulationSettings {
        abi_version: ABI_VERSION,
        initial_equity: 100.0,
        pip_value: 0.01,
        pip_value_per_lot: 1.0,
        max_hold_bars: 1,
        min_hold_bars: 1,
        max_trades_per_day: 1,
        month_capacity: 2,
        gap_threshold_ms: 600_000,
        ..Default::default()
    };
    let view = PopulationEvaluationViewV1::contiguous_range(
        ROWS,
        0,
        FIXTURE_TRAINING_END,
        PopulationTimestampModeV1::Canonical,
        None,
    )?;
    ensure!(
        population.feature_plan() == canonical.feature_plan()
            && population.provenance() == canonical.provenance()
            && population.normalization_fitted_state() == canonical.normalization_fitted_state(),
        "Data population guard lost source/recipe/fit metadata"
    );
    let result = population.evaluate_metrics_v1(view, genes, &scenarios, &settings)?;
    ensure!(
        result.runtime_identity() == canonical.runtime_identity(),
        "HIP population changed lease"
    );
    let metrics = result.metrics();
    let rows = metrics.metric_rows();
    ensure!(
        rows.len() == scenarios.len(),
        "HIP omitted cohort scenarios"
    );
    for (row, expected) in rows.iter().zip([(602, 9003), (501, 9001), (602, 9002)]) {
        ensure!(
            (row.candidate_id, row.scenario_id) == expected
                && row.values.iter().all(|value| value.is_finite()),
            "HIP cohort identity/value drift"
        );
    }
    ensure!(
        rows[1].values[0] == 0.0 && rows[1].values[8] == 0.0,
        "neutral gene unexpectedly traded"
    );
    if flat {
        // Independent arithmetic: flat prices earn zero before costs; each
        // charged fill loses exactly two units. Require actual trading too.
        ensure!(
            rows[0].values[8] > 0.0
                && rows[2].values[8] == rows[0].values[8]
                && rows[0].values[0] == -2.0 * rows[0].values[8]
                && rows[2].values[0] == 0.0,
            "flat HIP population violates known fill/commission arithmetic"
        );
    }
    ensure!(
        metrics.counters().dataset_upload_bytes == 0
            && metrics.terminal_synchronization_count() == 1
            && metrics.terminal_readback_count() == 1
            && metrics.terminal_readback_bytes() == 3 * 104,
        "HIP resident population copied its parent or changed bounded metrics transfer"
    );
    println!(
        "{}",
        json!({"type":"resident-population","schema":SCHEMA,"id":id,
        "view_start":0,"view_end":FIXTURE_TRAINING_END,"scenarios":rows.len(),
        "candidate_ids":rows.iter().map(|row| row.candidate_id).collect::<Vec<_>>(),
        "net_results":rows.iter().map(|row| row.values[0]).collect::<Vec<_>>(),
        "trades":rows.iter().map(|row| row.values[8]).collect::<Vec<_>>(),
        "dataset_reupload_bytes":metrics.counters().dataset_upload_bytes,
        "terminal_metric_bytes":metrics.terminal_readback_bytes(),
        "physical_binding_sha256":hex(&result.physical_binding_sha256()),
        "known_flat_cost_arithmetic_checked":flat,"population_cpu_parity":false,
        "search_admission":false,"trading_or_oos_proof":false})
    );
    Ok(())
}

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        (args.len() == 3 || args.len() == 5) && args[1] == "--device",
        "usage: hip_shared_producers_smoke --device <HIP ordinal> [--normalization none|enabled]"
    );
    let normalize = if args.len() == 5 {
        ensure!(args[3] == "--normalization", "unexpected fixture option");
        match args[4].as_str() {
            "none" => false,
            "enabled" => true,
            _ => anyhow::bail!("normalization fixture option must be none or enabled"),
        }
    } else {
        false
    };
    let ordinal: u32 = args[2]
        .parse()
        .context("HIP ordinal must be a nonnegative u32")?;
    run_with_options(ordinal, normalize)
}

fn run_with_options(ordinal: u32, normalize: bool) -> Result<()> {
    let normalization_label = if normalize {
        "explicit-diagnostic-policy-v3"
    } else {
        "explicit-diagnostic-None"
    };
    let selection = if normalize {
        &NORMALIZED_SELECTED
    } else {
        &SELECTED
    };
    let lease = HipRunLeaseV1::acquire(ordinal)
        .context("actual HIP device required; no skip or fallback")?;
    // Explicit diagnostic-only configuration in this standalone process. The
    // production assembler still reads and honors the installed startup mode.
    neoethos_data::install_data_runtime_overrides(normalize);
    ensure!(
        neoethos_data::current_data_runtime_overrides().normalize_features == normalize,
        "fixture expected its explicit startup normalization installation"
    );
    println!(
        "{}",
        json!({"type":"metadata","schema":SCHEMA,"backend":"amd-hip",
        "scope":"shared-producers-pack-adaptive-GA-and-resident-cohort-smoke-only","cpu_oracle":"existing-production-Session-v2-and-SMC-v3",
        "source":"temporary-canonical-Vortex-synthetic-generation","normalization":normalization_label,
        "independent_mathematical_proof":false,"trading_or_oos_proof":false,
        "device_ordinal":lease.identity().device_ordinal(),"device_uuid":hex(&lease.identity().device_uuid()),
        "architecture":lease.identity().architecture(),"runtime_version":lease.identity().runtime_version(),
        "driver_version":lease.identity().driver_version(),"lease_id":lease.identity().lease_id(),
        "stream_id":lease.identity().stream_id()})
    );
    let mut mismatches = 0;
    let mut merkle_mismatches = 0;
    for (id, flat) in [
        ("gapped-calendar-fvg-history", false),
        ("flat-zero-volume", true),
    ] {
        let (_fixture_root, frame) = canonical_fixture(id, &fixture(flat))?;
        let input = frame.ohlcv();
        let prepared = PreparedHipOhlcvV1::preflight_canonical(&frame)?;
        let smc_plan = HipSmcMemoryPlanV1::checked(input.len())?;
        let cpu_session = compute_session_feature_columns_f64(&input)?;
        let cpu_smc = compute_smc_feature_columns_f64(&input)?;
        let shared = prepared.upload(&lease)?;
        let session = ResidentHipSessionV1::materialize_on(&shared)?;
        let smc = ResidentHipSmcV1::materialize_on(&shared)?;
        ensure!(
            shared.same_upload_as(session.inputs()) && shared.same_upload_as(smc.inputs()),
            "producers did not retain the same actual uploaded input owner"
        );
        let selected = selection
            .iter()
            .map(|&(is_smc, index)| {
                if is_smc {
                    smc.column_v1(index)
                } else {
                    session.column_v1(index)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        // Explicit diagnostic reserve, not a production budget estimate.
        let canonical = PreparedHipCanonicalFeatureStoreV1::preflight(
            &smc,
            selected,
            FIXTURE_ALLOCATOR_RESERVE,
        )?
        .materialize()?;
        let (cpu_root, cpu_fitted) =
            selected_cpu_root(input, &cpu_session, &cpu_smc, selection, normalize)?;
        ensure!(
            canonical.normalization_fitted_state() == cpu_fitted.as_ref(),
            "actual HIP fit words differ from the CPU policy-v3 fitted state"
        );
        if let Some(fit) = canonical.normalization_fitted_state() {
            ensure!(
                fit.training_rows()? == (0..FIXTURE_TRAINING_END),
                "HIP canonical fixture fit touched a different scope"
            );
            fit.validate_plan(canonical.feature_plan())?;
        }
        let hip_root = canonical.canonical_content_merkle_sha256();
        let merkle_matches = cpu_root == hip_root;
        merkle_mismatches += usize::from(!merkle_matches);
        ensure!(
            canonical.provenance().bindings().len() == 1
                && canonical.provenance().bindings()[0].segments()[0].row_start() == 0
                && canonical.provenance().bindings()[0].segments()[0].row_end() == ROWS as u64,
            "canonical fixture source segment drift"
        );
        println!(
            "{}",
            json!({"type":"canonical-pack","schema":SCHEMA,"id":id,
            "rows":input.len(),"columns":canonical.physical_plan().columns(),
            "ordered_names":canonical.physical_plan().ordered_names(),
            "cpu_merkle_sha256":hex(&cpu_root),"hip_merkle_sha256":hex(&hip_root),
            "merkle_matches":merkle_matches,"normalization":normalization_label,
            "fit_words_match_cpu":cpu_fitted.is_some(),
            "normalization_training_end":if normalize { Some(FIXTURE_TRAINING_END) } else { None },
            "feature_plan_sha256":hex(canonical.feature_plan().identity().as_bytes()),
            "source_provenance_sha256":hex(canonical.provenance().identity().as_bytes()),
            "assembly_identity_sha256":hex(&canonical.assembly_identity_sha256()),
            "pack_readback_count":canonical.physical_receipt().readback_count(),
            "pack_readback_bytes":canonical.physical_receipt().readback_bytes(),
            "additional_feature_d2h_bytes":0,"search_admission":false})
        );
        // Exact future evaluator workspace is charged before binding Search.
        // This fixture uses fixed-pip stops (no resident adaptive-base buffer),
        // independently of its enabled feature-normalization policy.
        let mut population =
            canonical.bind_population_for_search_v3(GA_CHUNK, GA_MONTH_CAPACITY, 0)?;
        // GA starts on the pristine parent. Terminal completion restores that
        // exact core before ordinary cohort uploads replace its gene/scenarios.
        check_resident_generations(&mut population, &canonical, input, id, flat)?;
        check_resident_population(&mut population, &canonical, id, flat)?;
        population.try_close()?;
        // SMC's checked completion fence is after Session on this same stream.
        // Only this terminal validation section reads feature arrays back.
        let session_diagnostic = session.read_terminal_diagnostic(ROWS * 23 * 9)?;
        let smc_diagnostic = smc.read_terminal_diagnostic(ROWS * 441)?;
        let session_values = session_diagnostic.values;
        let session_validity = session_diagnostic.validity;
        let smc_values = smc_diagnostic.values;
        let smc_validity = smc_diagnostic.validity;
        let months = smc_diagnostic.months;
        let days = smc_diagnostic.days;
        let slots = smc_diagnostic.parent_slots;
        check_parent_terminal(
            &input,
            &months,
            &days,
            &slots,
            smc.identity().generated_parent_sha256(),
        )?;
        for (family, cpu, names, values, validity, artifact) in [
            (
                "Session-v2",
                &cpu_session[..],
                &HIP_SESSION_COLUMN_NAMES_V1[..],
                &session_values[..],
                &session_validity[..],
                session.identity().artifact_sha256(),
            ),
            (
                "SMC-v3",
                &cpu_smc[..],
                &HIP_SMC_COLUMN_NAMES_V1[..],
                &smc_values[..],
                &smc_validity[..],
                smc.identity().artifact_sha256(),
            ),
        ] {
            let result = compare_terminal(cpu, names, input.len(), values, validity)?;
            mismatches += result.value_mismatches + result.validity_mismatches;
            println!(
                "{}",
                json!({"type":"fixture","schema":SCHEMA,"id":format!("{id}/{family}"),
                "family":family,"rows":input.len(),"columns":names.len(),"cells":input.len()*names.len(),
                "input_sha256":hex(&prepared.identity().input_sha256()),"artifact_sha256":hex(&artifact),
                "value_bit_mismatches":result.value_mismatches,"validity_mismatches":result.validity_mismatches,
                "cpu_values_sha256":result.cpu_value_sha256,"hip_values_sha256":hex(&Sha256::digest(values)),
                "cpu_validity_sha256":result.cpu_validity_sha256,"hip_validity_sha256":hex(&Sha256::digest(validity)),
                "first_mismatch":result.first_mismatch,"terminal_feature_d2h_bytes":values.len()+validity.len()})
            );
        }
        println!(
            "{}",
            json!({"type":"shared-input","schema":SCHEMA,"id":id,"same_upload":true,
            "logical_input_upload_bytes":prepared.memory_plan().device_bytes(),
            "logical_incremental_smc_device_bytes":smc_plan.incremental_device_bytes(),
            "smc_seal_d2h_bytes":smc_plan.sealing_d2h_bytes(),"parent_terminal_d2h_bytes":months.len()+days.len()+slots.len(),
            "parent_hashes_match_readback":true,"calendar_matches_chrono":true,
            "parent_slot_cpu_parity":false})
        );
        // Close output handles first, then the last shared input handle. Observe
        // every explicit release error even if an earlier family release fails.
        let packed = canonical.try_close();
        let a = session.try_close();
        let b = smc.try_close();
        let c = shared.try_close();
        packed?;
        a?;
        b?;
        c?;
    }
    lease
        .try_close()
        .map_err(|e| anyhow::anyhow!("HIP lease cleanup failed: {e}"))?;
    println!(
        "{}",
        json!({"type":"summary","schema":SCHEMA,"fixtures":4,"cells":2*ROWS*(23+46),
        "mismatches":mismatches,"canonical_pack_fixtures":2,"merkle_mismatches":merkle_mismatches,
        "resident_population_fixtures":2,"resident_population_scenarios":6,
        "adaptive_ga_fixtures":2,"adaptive_ga_population":GA_POPULATION,"adaptive_ga_chunk":GA_CHUNK,
        "adaptive_ga_generations":GA_GENERATIONS,"adaptive_ga_evaluation_slots":2*GA_POPULATION*GA_GENERATIONS,
        "device_executed":true,"cleanup_completed":true,"passed":mismatches==0 && merkle_mismatches==0,
        "whole_pipeline_parity":false})
    );
    ensure!(
        mismatches == 0 && merkle_mismatches == 0,
        "shared HIP producer smoke found {mismatches} exact cells and {merkle_mismatches} Merkle mismatches"
    );
    Ok(())
}
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!(
                "{}",
                json!({"type":"error","schema":SCHEMA,"message":format!("{error:#}"),
            "passed":false,"gpu_fallback":false})
            );
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Explicitly selected actual-device integration, not one of the host-only
    /// controls. The same compiled example test binary can be reused on a GPU.
    #[test]
    #[ignore = "requires an actual AMD HIP device; no skip or CPU fallback"]
    fn actual_hip_shared_producers_pack_ga_and_population() -> Result<()> {
        ensure!(
            std::env::var("NEOETHOS_REQUIRE_GPU").as_deref() == Ok("1"),
            "this ignored fixture requires NEOETHOS_REQUIRE_GPU=1"
        );
        let ordinal = std::env::var("NEOETHOS_HIP_DEVICE")
            .unwrap_or_else(|_| "0".to_owned())
            .parse::<u32>()
            .context("NEOETHOS_HIP_DEVICE must be a nonnegative ordinal")?;
        // Enabled policy3 is intentional; never disable normalization just to
        // obtain a device pass. The ordinary executable retains both options.
        run_with_options(ordinal, true)
    }

    #[test]
    fn canonical_selected_recipe_merkle_controls_are_host_only() -> Result<()> {
        let (_root, frame) = canonical_fixture("host-contract", &fixture(false))?;
        let input = frame.ohlcv();
        let prepared = PreparedHipOhlcvV1::preflight_canonical(&frame)?;
        assert_eq!(prepared.memory_plan().row_count(), ROWS);
        let session = compute_session_feature_columns_f64(input)?;
        let smc = compute_smc_feature_columns_f64(input)?;
        let expected = selected_cpu_root(input, &session, &smc, &SELECTED, false)?.0;
        let mut reversed = SELECTED;
        reversed.reverse();
        assert_ne!(
            expected,
            selected_cpu_root(input, &session, &smc, &reversed, false)?.0
        );
        assert!(selected_cpu_root(input, &session, &smc, &[], false).is_err());
        assert!(selected_cpu_root(input, &session, &smc, &[(false, 23)], false).is_err());
        assert!(
            selected_cpu_root(input, &session, &smc, &[SELECTED[0], SELECTED[0]], false).is_err()
        );
        assert_eq!(frame.artifact().row_count(), ROWS as u64);
        Ok(())
    }

    #[test]
    fn enabled_selected_cpu_oracle_retains_gate_identity_and_original_training_scope() -> Result<()>
    {
        for flat in [false, true] {
            let input = fixture(flat);
            let session = compute_session_feature_columns_f64(&input)?;
            let smc = compute_smc_feature_columns_f64(&input)?;
            let (_, fit) = selected_cpu_root(&input, &session, &smc, &NORMALIZED_SELECTED, true)?;
            let fit = fit.context("enabled CPU fit missing")?;
            assert_eq!(fit.training_rows()?, 0..FIXTURE_TRAINING_END);
            for index in [1, 3] {
                assert_eq!(fit.fits()[index].median, 0.0);
                assert_eq!(fit.fits()[index].scale, 1.0);
                assert!(!fit.fits()[index].degenerate);
            }
        }
        Ok(())
    }

    #[test]
    fn shared_fixture_oracles_and_exact_comparator_negative_controls_are_host_only() {
        let scenarios = fixture_scenarios();
        assert!(GA_CHUNK < GA_POPULATION && GA_POPULATION % GA_CHUNK != 0);
        assert_eq!(scenarios.len(), GA_POPULATION);
        for (ordinal, scenario) in scenarios.iter().enumerate() {
            assert_eq!(
                (scenario.base_candidate_id, scenario.scenario_id),
                (ordinal as u64, ordinal as u64)
            );
            assert_eq!(scenario.window_offset, 0);
            assert_eq!(scenario.window_len, FIXTURE_TRAINING_END as u32);
        }
        for flat in [false, true] {
            let input = fixture(flat);
            PreparedHipOhlcvV1::preflight(&input).unwrap();
            for (cpu, names) in [
                (
                    compute_session_feature_columns_f64(&input).unwrap(),
                    &HIP_SESSION_COLUMN_NAMES_V1[..],
                ),
                (
                    compute_smc_feature_columns_f64(&input).unwrap(),
                    &HIP_SMC_COLUMN_NAMES_V1[..],
                ),
            ] {
                let mut values: Vec<u8> = cpu
                    .iter()
                    .flat_map(|c| c.values.iter().flat_map(|v| v.to_bits().to_le_bytes()))
                    .collect();
                let mut validity: Vec<u8> = cpu
                    .iter()
                    .flat_map(|c| c.validity.iter().map(|v| v.code()))
                    .collect();
                let good = compare_terminal(&cpu, names, ROWS, &values, &validity).unwrap();
                assert_eq!((good.value_mismatches, good.validity_mismatches), (0, 0));
                values[0] ^= 1;
                let bad = compare_terminal(&cpu, names, ROWS, &values, &validity).unwrap();
                assert_eq!((bad.value_mismatches, bad.validity_mismatches), (1, 0));
                values[0] ^= 1;
                validity[0] ^= 1;
                let bad = compare_terminal(&cpu, names, ROWS, &values, &validity).unwrap();
                assert_eq!((bad.value_mismatches, bad.validity_mismatches), (0, 1));
            }
        }
    }
}
