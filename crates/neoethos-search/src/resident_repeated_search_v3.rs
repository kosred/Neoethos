//! Actual compact V5 -> resident adaptive generations -> complete terminal census.
//! This boundary returns Search evidence, never validation or promotion authority.

use super::*;
use crate::genetic::evolution_math::{
    ParentSelectionPolicy, SeenSignatureMemory, SurvivorSelectionPolicy,
};
use crate::genetic::search_engine::{
    evaluation_backtest_settings, resident_generation_population_settings_v1,
};
use neoethos_gpu_cuda::resident_search_slice2_v3::{
    ResidentSearchArchivePolicyV3, ResidentSearchExecutionInputsV3, ResidentSearchExecutionPlanV3,
    ResidentSearchRejectedAuthorityV3, ResidentSearchTryCompleteV3,
};
use neoethos_gpu_cuda::{
    ParentSelectionPolicyV1, ResidentAdaptiveCheckpointV3, ResidentAdaptiveGenerationInputsV3,
    ResidentGenerationPlanAuthorityInputV1, ResidentGenerationTemplateV3,
    ResidentScoringObjectiveV2, SurvivorSelectionPolicyV1,
    discovery_adaptive_generation_semantics_sha256_v3, novelty_disabled_semantics_sha256_v2,
    rank_semantics_sha256_v2, resident_metric_semantics_sha256_v2, scoring_semantics_sha256_v2,
    seal_adaptive_resident_generation_plan_v3,
};
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};
use std::time::{Duration, Instant};

/// All archived observations plus the last evaluated population, reduced by the
/// same exact-behavior union as CPU Search. It is not a completed Discovery run.
pub struct ResidentRepeatedSearchResultV3 {
    pub(crate) receipt: CanonicalGpuResidentSearchInputReceiptV3,
    pub(crate) search_result: crate::genetic::SearchResult,
    pub(crate) checkpoint: ResidentAdaptiveCheckpointV3,
    pub(crate) seed: u64,
    pub(crate) stage1_time_scope: ResidentSelectionStage1TimeScopeV2,
    pub(crate) scope: CanonicalGpuResidentSearchArtifactScopeV3,
    screening_scope: ResidentFeatureScreeningScopeV2,
    validation_parent:
        crate::exact_resident_dataset_authority_v1::SealedExactResidentCompactParentV3,
    evaluation_config: crate::genetic::EvaluationConfig,
    population_sizing_receipt: ResidentPopulationAutoSizingReceiptV2,
    runtime_snapshot: crate::genetic::search_engine::ResidentGenerationZeroRuntimeSnapshotV1,
    financial_contract:
        crate::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractV3,
    session: Option<neoethos_gpu_cuda::resident_feature_store_v3::ResidentPopulationSessionV3>,
}

impl ResidentRepeatedSearchResultV3 {
    pub fn search_result(&self) -> &crate::genetic::SearchResult {
        &self.search_result
    }
    pub fn checkpoint(&self) -> &ResidentAdaptiveCheckpointV3 {
        &self.checkpoint
    }
    pub const fn resolved_seed(&self) -> u64 {
        self.seed
    }
    pub fn native_receipt(&self) -> &CanonicalGpuResidentSearchInputReceiptV3 {
        &self.receipt
    }
    pub fn stage1_timestamp_bounds_ms(&self) -> (i64, i64) {
        self.stage1_time_scope.timestamp_bounds_ms()
    }
    pub fn resident_scope(&self) -> &CanonicalGpuResidentSearchArtifactScopeV3 {
        &self.scope
    }

    /// Evaluate only selected-data scopes on the genuine retained Data owner.
    /// Returns strict metric evidence, not a validation pass or a trade ledger.
    /// Calibration and final holdout require their own later authority producer.
    pub fn evaluate_selection_population_v3(
        &mut self,
        genes: &[crate::genetic::Gene],
        view: &crate::ExactResidentDatasetViewV1,
        timestamp_mode: neoethos_gpu_cuda::PopulationTimestampModeV1,
        cancelled: impl Fn() -> bool,
    ) -> Result<
        crate::population_execution_evidence_v1::retained_compact_v3::ExactResidentCompactMetricsV3,
    > {
        let selection = self
            .screening_scope
            .selection_range_v3()
            .map_err(anyhow::Error::msg)?;
        use crate::exact_resident_dataset_authority_v1::ExactResidentDatasetViewRequestV1 as Request;
        let view = match view {
            crate::ExactResidentDatasetViewV1::Full { row_count } => {
                ensure!(
                    u64::try_from(*row_count)? == self.receipt.row_count(),
                    "full resident view row count differs from its parent"
                );
                Request::Full
            }
            crate::ExactResidentDatasetViewV1::ContiguousRange { start, end } => {
                Request::ContiguousRange {
                    start: *start,
                    end: *end,
                }
            }
            crate::ExactResidentDatasetViewV1::OrderedIndices { indices } => {
                Request::OrderedIndices(indices)
            }
        };
        let timestamp_mode = match timestamp_mode {
            neoethos_gpu_cuda::PopulationTimestampModeV1::Canonical => crate::population_execution_evidence_v1::ExactPopulationTimestampModeV1::Canonical,
            neoethos_gpu_cuda::PopulationTimestampModeV1::DisabledIndexDelta => crate::population_execution_evidence_v1::ExactPopulationTimestampModeV1::DisabledIndexDelta,
        };
        self.financial_contract
            .validate_evaluation_costs(&self.evaluation_config)?;
        let session = self
            .session
            .as_mut()
            .context("resident validation owner already consumed")?;
        crate::population_execution_evidence_v1::retained_compact_v3::evaluate_genes_v3(
            &self.validation_parent,
            session,
            selection,
            genes,
            &self.evaluation_config,
            &self.population_sizing_receipt,
            &self.runtime_snapshot,
            view,
            timestamp_mode,
            cancelled,
        )
    }
    pub fn with_resident_population_session<Output>(
        &mut self,
        consumer: impl FnOnce(
            &mut neoethos_gpu_cuda::resident_feature_store_v3::ResidentPopulationSessionV3,
        ) -> Result<Output>,
    ) -> Result<Output> {
        self.scope.validate()?;
        let session = self
            .session
            .as_mut()
            .context("resident validation owner already consumed")?;
        let identity = |session: &neoethos_gpu_cuda::resident_feature_store_v3::ResidentPopulationSessionV3| {
            (session.rows(), session.columns(), session.admission_identity_sha256(),
             session.canonical_content_merkle(), session.data_transient_retirement_process_token(),
             session.device_identity().primary_context_process_token(),
             session.data_population_limits().map(|limits| limits.workspace_plan_identity_sha256()))
        };
        let expected = identity(session);
        ensure!(
            expected.0 as u64 == self.receipt.row_count()
                && expected.1 as u64 == self.receipt.column_count()
                && expected.4 != [0; 32]
                && expected.6.is_some(),
            "retained Data owner differs from the completed Search parent"
        );
        let outcome = consumer(session);
        ensure!(
            identity(session) == expected,
            "retained Data owner drifted during validation"
        );
        outcome
    }
}

fn complete_compact_session(
    session: neoethos_gpu_cuda::resident_feature_store_v3::ResidentPopulationSessionV3,
) -> Result<()> {
    let lease = session.record_consumer_completion()?;
    while !lease.completion_is_ready()? {
        std::thread::yield_now();
    }
    drop(lease);
    Ok(())
}

impl Drop for ResidentRepeatedSearchResultV3 {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            if let Err(error) = complete_compact_session(session) {
                tracing::error!(%error, "resident post-Search Data completion failed; uncertain native lifetime retained");
            }
        }
    }
}

fn rejection<A>(rejected: ResidentSearchRejectedAuthorityV3<A>) -> anyhow::Error {
    let (error, authority) = rejected.into_parts_v3();
    // The typed authority retains/leaks uncertain CUDA work according to its
    // existing Drop policy; never substitute an unconditional native free.
    drop(authority);
    anyhow::anyhow!("resident Search transition rejected: {error}")
}

fn probability_q32(value: f64) -> Result<u64> {
    ensure!(
        value.is_finite() && (0.0..=1.0).contains(&value),
        "invalid SMC probability"
    );
    Ok((value * (1_u64 << 32) as f64).round() as u64)
}

fn template(gene: crate::genetic::Gene) -> Result<ResidentGenerationTemplateV3> {
    let mut feature_indices = Vec::new();
    feature_indices.try_reserve_exact(gene.indices.len())?;
    for index in gene.indices {
        feature_indices.push(u64::try_from(index)?);
    }
    let flags = [
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
        gene.use_displacement,
    ]
    .into_iter()
    .enumerate()
    .fold(0_u32, |bits, (index, enabled)| {
        bits | (u32::from(enabled) << index)
    });
    Ok(ResidentGenerationTemplateV3 {
        feature_indices,
        weights: gene.weights,
        smc_flags: flags,
        long_threshold: gene.long_threshold,
        short_threshold: gene.short_threshold,
        target_pips: gene.tp_pips,
        stop_pips: gene.sl_pips,
        stop_vol_multiplier: gene.stop_vol_mult,
    })
}

fn convergence_reached(
    elapsed: Duration,
    since_improvement: Duration,
    stagnant_generations: u64,
    max_runtime: Option<Duration>,
    patience: usize,
    fraction: f64,
) -> bool {
    let Some(budget) = max_runtime else {
        return false;
    };
    let floor = budget.mul_f64(fraction);
    patience > 0
        && elapsed >= floor
        && (stagnant_generations >= patience as u64 || since_improvement >= floor)
}

/// Execute the actual admitted compact population on CUDA. Only one-time seed
/// controls and bounded generation checkpoints cross the host boundary before
/// terminal export. Unsupported configured policies fail before native Search
/// work; neither population nor candidate census is reduced to make them fit.
pub fn run_prepared_canonical_trendbar_research_resident_search_v5<F, C>(
    prepared: PreparedCanonicalDiscoveryRunInputV5,
    mut progress: F,
    cancelled: C,
) -> Result<ResidentRepeatedSearchResultV3>
where
    F: FnMut(DiscoveryProgress),
    C: Fn() -> bool,
{
    let PreparedCanonicalDiscoveryRunInputV5 { native } = prepared;
    let PreparedNativeCudaCanonicalDiscoveryRunInputV5 {
        receipt,
        feature_names,
        sealed_store,
        population_sizing_receipt: sizing,
        financial_contract,
        mut evaluation_config,
        runtime_snapshot: snapshot,
        stage1_time_scope,
        screening_scope,
        discovery_config: config,
    } = native;
    ensure!(
        !cancelled(),
        "__DISCOVERY_CANCELLED__ before resident Search binding"
    );
    snapshot.validate_current("before repeated resident Search binding")?;
    snapshot.validate_against_receipt_v2(&sizing)?;
    stage1_time_scope
        .validate_binding_v2(
            sealed_store.contract().layout().row_count(),
            sizing.stage1_row_start() as u64,
            sizing.stage1_row_end() as u64,
            sealed_store.pinned_source_projection_v1().identity_sha256(),
        )
        .map_err(anyhow::Error::msg)?;
    sizing.validate_financial_authority_against_pinned_source_projection_v2(
        &financial_contract,
        sealed_store.pinned_source_projection_v1(),
    )?;
    let recomputed =
        evaluation_config_from_canonical_trendbar_contract_v2(&config, &financial_contract)?;
    ensure!(
        evaluation_config.symbol == recomputed.symbol
            && evaluation_config.account_currency == recomputed.account_currency
            && evaluation_config.growth_objective == recomputed.growth_objective
            && evaluation_config.growth_goal == recomputed.growth_goal
            && evaluation_config.kill_zones_enabled == recomputed.kill_zones_enabled
            && evaluation_config.session_spread_pips.is_none()
            && [
                evaluation_config.initial_equity,
                evaluation_config.pip_value,
                evaluation_config.pip_value_per_lot,
                evaluation_config.spread_pips,
                evaluation_config.commission_per_trade,
                evaluation_config.swap_long_pips_per_day,
                evaluation_config.swap_short_pips_per_day,
                evaluation_config.pnl_conversion_fee_rate,
                evaluation_config.risk_per_trade_min,
                evaluation_config.risk_per_trade_max,
                evaluation_config.high_quality_confidence
            ]
            .map(f64::to_bits)
                == [
                    recomputed.initial_equity,
                    recomputed.pip_value,
                    recomputed.pip_value_per_lot,
                    recomputed.spread_pips,
                    recomputed.commission_per_trade,
                    recomputed.swap_long_pips_per_day,
                    recomputed.swap_short_pips_per_day,
                    recomputed.pnl_conversion_fee_rate,
                    recomputed.risk_per_trade_min,
                    recomputed.risk_per_trade_max,
                    recomputed.high_quality_confidence
                ]
                .map(f64::to_bits),
        "carried resident evaluation differs from its financial/config authority"
    );
    let runtime = snapshot.genetic_search();
    ensure!(
        snapshot.seen_memory().file_path.is_none(),
        "native adaptive Search does not yet preserve configured seen-signature append-file persistence"
    );
    ensure!(
        !snapshot.migration_enabled(),
        "native adaptive Search does not support configured migration"
    );
    ensure!(
        config.generations > 0 && config.max_hours.is_finite(),
        "invalid resident generation/time budget"
    );
    let max_runtime = if config.max_hours > 0.0 {
        Some(
            Duration::try_from_secs_f64(config.max_hours * 3600.0)
                .context("invalid Search time budget")?,
        )
    } else {
        None
    };
    let population = sizing.resolved_population();
    let archive_capacity = runtime
        .checked_effective_archive_cap(population, config.generations)
        .context("resident archive capacity overflow")?;
    let seed = runtime.seed.unwrap_or_else(|| rand::rng().random());
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let bank_count = (population / 4).min(50).max((population / 10).min(50));
    let mut templates = Vec::new();
    templates.try_reserve_exact(bank_count)?;
    for gene in crate::genetic::seed_templates::seed_professional_templates(
        bank_count,
        &feature_names,
        feature_names.len(),
        &mut rng,
    ) {
        templates.push(template(gene)?);
    }
    let seed_template_count = templates.len().min((population / 10).min(50));
    // Consume the existing one-shot staged ledger memory exactly once, after
    // checking the frozen installed snapshot. Never re-load per generation.
    let seen = SeenSignatureMemory::current();
    ensure!(
        seen.file_path.is_none() && seen.max_entries == snapshot.seen_memory().max_entries,
        "seen memory changed after immutable snapshot validation"
    );
    let mut initial_seen_hashes = Vec::new();
    initial_seen_hashes.try_reserve_exact(seen.order.len())?;
    initial_seen_hashes.extend(seen.order.iter().copied());
    let selection = runtime.resolved_selection();
    let gate = runtime.resolved_smc_gate();
    evaluation_config.smc_gate_threshold = gate.start;
    let smc = snapshot.smc_search();
    let bounds = snapshot.gene_stop_bounds();
    let mut backtest = evaluation_backtest_settings(&evaluation_config)?;
    backtest.adaptive_base_pips = None;
    backtest.adaptive_rr = sizing.adaptive_rr();
    let settings = resident_generation_population_settings_v1(&backtest)?;
    let objective = if evaluation_config.growth_goal.is_some() {
        ResidentScoringObjectiveV2::RiskyGrowthGoalV6
    } else if evaluation_config.growth_objective {
        ResidentScoringObjectiveV2::RiskyGrowthV5
    } else {
        ResidentScoringObjectiveV2::PropFirmV4
    };
    let mut run_hash = Sha256::new();
    run_hash.update(b"neoethos.canonical-resident-adaptive-search.v3\0");
    run_hash.update(receipt.identity_sha256()?.as_bytes());
    run_hash.update(sizing.identity_sha256().as_bytes());
    run_hash.update(financial_contract.identity_sha256()?.as_bytes());
    // Bind the entire resolved request once, including the configured mode,
    // risk/goal, generation/time budget, and validation policy. Runtime knobs
    // and native settings below are separate inputs, not a complete config.
    run_hash.update(
        crate::canonical_discovery_config_digest_v1::canonical_discovery_config_digest_v1(&config)
            .map_err(|error| {
                anyhow::anyhow!("encode resolved resident Search config: {error:?}")
            })?,
    );
    run_hash.update(serde_json::to_vec(runtime)?);
    run_hash.update(serde_json::to_vec(&settings)?);
    let smc_weights = [
        evaluation_config.smc_weight_ob,
        evaluation_config.smc_weight_fvg,
        evaluation_config.smc_weight_liq,
        evaluation_config.smc_weight_mtf,
        evaluation_config.smc_weight_premium,
        evaluation_config.smc_weight_inducement,
        evaluation_config.smc_weight_bos,
        evaluation_config.smc_weight_choch,
        evaluation_config.smc_weight_eqh,
        evaluation_config.smc_weight_eql,
        evaluation_config.smc_weight_displacement,
    ];
    for weight in smc_weights {
        run_hash.update(weight.to_bits().to_le_bytes());
    }
    run_hash.update([u8::from(snapshot.smc_gate_disabled())]);
    run_hash.update(seed.to_le_bytes());
    for value in [
        evaluation_config.initial_equity,
        evaluation_config.risk_per_trade_min,
        evaluation_config.risk_per_trade_max,
        evaluation_config.high_quality_confidence,
    ] {
        run_hash.update(value.to_bits().to_le_bytes());
    }
    if let Some(goal) = evaluation_config.growth_goal {
        for value in [goal.start_balance, goal.target_balance, goal.horizon_days] {
            run_hash.update(value.to_bits().to_le_bytes());
        }
    }
    let (first_timestamp_ms, last_timestamp_ms) = stage1_time_scope.timestamp_bounds_ms();
    run_hash.update(first_timestamp_ms.to_le_bytes());
    run_hash.update(last_timestamp_ms.to_le_bytes());
    let run_identity_sha256: [u8; 32] = run_hash.finalize().into();
    let scope = CanonicalGpuResidentSearchArtifactScopeV3::for_entire_receipt(
        CanonicalSearchWindowRoleV1::DiscoveryInput,
        receipt.clone(),
    )?;
    let mut strict = bind_strict_resident_feature_store_v3_run_input(sealed_store, &scope)?;
    let plan = strict.with_resident_population_session_v3(|session| {
        let limits = session
            .data_population_limits()
            .context("compact owner lacks Data limits")?;
        sizing.validate_against_execution_limits_v2(
            session.device_identity().ordinal(),
            session.pre_materialization_free_bytes_snapshot(),
            session.rows(),
            session.columns(),
            limits,
        )?;
        let semantics = discovery_adaptive_generation_semantics_sha256_v3();
        let survivor_count =
            ((population as f64 * selection.survivor_fraction).round() as usize).min(population);
        let immigrant_count = ((population as f64 * selection.immigrant_ratio).round() as usize)
            .min(population - survivor_count);
        let base = ResidentGenerationPlanAuthorityInputV1 {
            parent_selection: match selection.parent {
                ParentSelectionPolicy::Uniform => ParentSelectionPolicyV1::Uniform,
                ParentSelectionPolicy::RankWeighted => ParentSelectionPolicyV1::RankWeighted,
                ParentSelectionPolicy::Softmax => ParentSelectionPolicyV1::Softmax,
                ParentSelectionPolicy::Tournament => ParentSelectionPolicyV1::Tournament,
            },
            survivor_selection: match selection.survivor {
                SurvivorSelectionPolicy::Elitist => SurvivorSelectionPolicyV1::Elitist,
                SurvivorSelectionPolicy::RankWeighted => SurvivorSelectionPolicyV1::RankWeighted,
                SurvivorSelectionPolicy::Tournament => SurvivorSelectionPolicyV1::Tournament,
                SurvivorSelectionPolicy::Generational => SurvivorSelectionPolicyV1::Generational,
            },
            max_terms_per_gene: sizing.term_cap(),
            minimum_terms_per_gene: 1,
            logical_population_count: population,
            retained_evaluation_capacity: usize::try_from(
                limits
                    .max_concurrent_scenario_count()
                    .min(population as u64),
            )?,
            feature_count: session.columns(),
            generation_count: config.generations,
            survivor_count,
            immigrant_count,
            search_seed: seed,
            // V1 probability is unused by the identity-bound adaptive algorithm;
            // adaptive mutation count/intensity is derived on device from stagnation.
            mutation_intensity_q32: 0,
            threshold_ladder_bits: snapshot.threshold_ladder().map(f64::to_bits),
            stop_bounds_bits: [
                bounds.sl_min_pips,
                bounds.sl_max_pips,
                bounds.tp_min_pips,
                bounds.tp_max_pips,
                bounds.rr_min,
                bounds.rr_max,
            ]
            .map(f64::to_bits),
            smc_probability_q32: [
                smc.p_ob,
                smc.p_fvg,
                smc.p_liq,
                smc.p_mtf,
                smc.p_premium,
                smc.p_inducement,
                smc.p_bos,
                smc.p_choch,
                smc.p_eqh,
                smc.p_eql,
                smc.p_displacement,
            ]
            .map(probability_q32)
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .expect("eleven SMC fields"),
            generation_semantics_sha256: semantics,
            run_identity_sha256,
            strategy_gene_schema_sha256: Sha256::digest(include_bytes!("genetic/strategy_gene.rs"))
                .into(),
            rank_semantics_sha256: rank_semantics_sha256_v2(),
            metric_semantics_sha256: resident_metric_semantics_sha256_v2(),
            scoring_semantics_sha256: scoring_semantics_sha256_v2(objective),
            novelty_semantics_sha256: novelty_disabled_semantics_sha256_v2(),
            scenario_order_semantics_sha256: Sha256::digest(
                b"resident-base-scenario.v3;ordinal=candidate;window=exact-Stage1;no-perturbation",
            )
            .into(),
            cuda_build_manifest_sha256: session.device_identity().gpu_cuda_build_sha256(),
            rng_mapping_sha256: semantics,
        };
        let generation = seal_adaptive_resident_generation_plan_v3(
            base,
            ResidentAdaptiveGenerationInputsV3 {
                tournament_size: runtime.effective_tournament_size(population),
                min_structural_smc_flags: u32::try_from(smc.min_flags)?,
                adaptive_stops_enabled: sizing.adaptive_base_effective_for_stage1(),
                seen_capacity: seen.max_entries,
                seen_retry_attempts: runtime.effective_seen_retry_attempts(),
                seed_template_count,
                soft_stagnation_patience: runtime.effective_stagnation_patience(),
                survivor_fraction: selection.survivor_fraction,
                immigrant_fraction: selection.immigrant_ratio,
                selection_temperature: selection.temperature,
                minimum_improvement: runtime.effective_min_improvement(),
                gate_start: gate.start,
                gate_end: gate.end,
                gate_curve: gate.curve,
                gate_stagnation_step: gate.stagnation_step,
                smc_force_ratio: smc.force_ratio,
                templates,
                initial_seen_hashes,
            },
        )
        .map_err(|error| anyhow::anyhow!("seal adaptive native generation: {error:?}"))?;
        let rows = sizing.stage1_row_end() - sizing.stage1_row_start();
        let mut scenarios = Vec::new();
        scenarios.try_reserve_exact(population)?;
        for ordinal in 0..population {
            scenarios.push(neoethos_gpu_contracts::device::ScenarioDescriptor {
                base_candidate_id: ordinal as u64,
                scenario_id: ordinal as u64,
                window_len: u32::try_from(rows)?,
                ..neoethos_gpu_contracts::device::ScenarioDescriptor::default()
            });
        }
        ResidentSearchExecutionPlanV3::for_compact_session_v3(
            session,
            generation,
            ResidentSearchExecutionInputsV3 {
                settings,
                scenarios,
                smc_weights,
                smc_gate_disabled: snapshot.smc_gate_disabled(),
                growth_objective: evaluation_config.growth_objective,
                growth_goal: evaluation_config.growth_goal,
                stage1_row_start: sizing.stage1_row_start() as u64,
                stage1_row_end: sizing.stage1_row_end() as u64,
                first_timestamp_ms,
                last_timestamp_ms,
                novelty_weight: runtime.novelty_weight,
                archive_capacity: archive_capacity as u64,
                archive_policy: Some(ResidentSearchArchivePolicyV3 {
                    mode: match runtime.archive_scoring.mode.as_str() {
                        "active" => 1,
                        "pf" | "profit_factor" => 2,
                        "sharpe" => 3,
                        _ => 0,
                    },
                    // Zero-k is only meaningful with disabled novelty on CPU. The
                    // native zero-weight route does not evaluate neighbor scores.
                    neighbors: u32::try_from(
                        runtime
                            .novelty_neighbors
                            .max(usize::from(runtime.novelty_weight == 0.0)),
                    )?,
                    min_net: runtime.archive_scoring.min_net,
                    min_pf: runtime.archive_scoring.min_pf,
                    min_sharpe: runtime.archive_scoring.min_sharpe,
                }),
                adaptive_base_request: sizing
                    .resident_adaptive_view_and_request_v2()?
                    .map(|(_, request)| request),
            },
        )
        .and_then(ResidentSearchExecutionPlanV3::retain_compact_session_for_validation_v3)
        .map_err(anyhow::Error::msg)
    });
    let plan = match plan {
        Ok(plan) if !cancelled() => plan,
        result => {
            let lease = record_resident_feature_store_consumer_completion_v3(strict)?;
            drop(retain_resident_completion_until_ready_v1(lease)?);
            return Err(result.err().unwrap_or_else(|| {
                anyhow::anyhow!("__DISCOVERY_CANCELLED__ before resident Search launch")
            }));
        }
    };
    let (_, session) = strict.into_resident_population_session_v3()?;
    let mut chain = session.begin_resident_search_slice2_v3(plan);
    progress(DiscoveryProgress::SearchStarted {
        population,
        generations: config.generations,
        max_indicators: sizing.term_cap(),
    });
    let started = Instant::now();
    let mut last_improvement = started;
    let mut best = f64::NEG_INFINITY;
    let was_cancelled = loop {
        chain = chain
            .enqueue_score_and_rank_v3()
            .map_err(rejection)?
            .enqueue_stage_archive_from_rank_v3()
            .map_err(rejection)?
            .enqueue_evolve_and_publish_v3()
            .map_err(rejection)?;
        let (next, checkpoint) = chain.checkpoint_v3().map_err(rejection)?;
        chain = next;
        if checkpoint.best_score() > best + runtime.effective_min_improvement() {
            best = checkpoint.best_score();
            last_improvement = Instant::now();
        }
        progress(DiscoveryProgress::StageAdvanced {
            stage: "resident_cuda_generation",
            detail: format!(
                "completed {}/{} generations; {} actual evaluations; best {}; stagnant {}",
                checkpoint.evaluated_generations(),
                config.generations,
                checkpoint.evaluation_slots(),
                checkpoint.best_score(),
                checkpoint.stagnant_generations()
            ),
        });
        let was_cancelled = cancelled();
        if was_cancelled
            || checkpoint.evaluated_generations() >= config.generations as u64
            || max_runtime.is_some_and(|limit| started.elapsed() >= limit)
            || convergence_reached(
                started.elapsed(),
                last_improvement.elapsed(),
                checkpoint.stagnant_generations(),
                max_runtime,
                runtime.effective_convergence_patience(),
                runtime.effective_convergence_min_elapsed_fraction(),
            )
        {
            break was_cancelled;
        }
    };
    let mut pending = chain.enqueue_terminal_seal_v3().map_err(rejection)?;
    let terminal = loop {
        match pending
            .try_complete_v3()
            .map_err(|error| anyhow::anyhow!("resident terminal completion: {error}"))?
        {
            ResidentSearchTryCompleteV3::NotReady(next) => {
                pending = next;
                std::thread::yield_now();
            }
            ResidentSearchTryCompleteV3::Complete(terminal) => break terminal,
        }
    };
    let (session, candidates, checkpoint) = terminal
        .into_retained_compact_parts_v3()
        .map_err(|_| anyhow::anyhow!("terminal Search did not restore its retained Data owner"))?;
    let outcome = (|| {
        ensure!(
            !was_cancelled && !cancelled(),
            "__DISCOVERY_CANCELLED__ resident Search completed cleanup without publishing partial candidates"
        );
        let checkpoint = checkpoint.context("terminal Search omitted adaptive control evidence")?;
        let search_result = super::terminal_adapter::search_result_from_terminal_v3(
            candidates,
            &evaluation_config,
            stage1_time_scope.evaluation_span_days(),
            checkpoint.evaluated_gate(),
            checkpoint.evaluation_slots(),
        )?;
        snapshot.validate_current("after repeated resident Search completion")?;
        let validation_parent =
            crate::exact_resident_dataset_authority_v1::seal_exact_resident_compact_parent_v3(
                &scope, &session,
            )?;
        screening_scope
            .selection_range_v3()
            .map_err(anyhow::Error::msg)?;
        ensure!(
            screening_scope.parent_row_count() == receipt.row_count(),
            "retained validation selection scope differs from its actual parent"
        );
        Ok((search_result, checkpoint, validation_parent))
    })();
    match outcome {
        Ok((search_result, checkpoint, validation_parent)) => {
            evaluation_config.smc_gate_threshold = checkpoint.evaluated_gate();
            Ok(ResidentRepeatedSearchResultV3 {
                receipt,
                search_result,
                checkpoint,
                seed,
                stage1_time_scope,
                scope,
                screening_scope,
                validation_parent,
                evaluation_config,
                population_sizing_receipt: sizing,
                runtime_snapshot: snapshot,
                financial_contract,
                session: Some(session),
            })
        }
        Err(error) => {
            complete_compact_session(session)?;
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolved_template_mapping_preserves_every_genome_field_without_repair() {
        let gene = crate::genetic::Gene {
            indices: vec![2, 19, 71],
            weights: vec![-0.25, 0.5, 0.25],
            long_threshold: 0.375,
            short_threshold: -0.625,
            tp_pips: 50.0,
            sl_pips: 0.25,
            stop_vol_mult: 1.75,
            use_ob: true,
            use_fvg: false,
            use_liq_sweep: true,
            mtf_confirmation: false,
            use_premium_discount: true,
            use_inducement: false,
            use_bos: true,
            use_choch: false,
            use_eqh: true,
            use_eql: false,
            use_displacement: true,
            ..Default::default()
        };
        let mapped = template(gene.clone()).unwrap();
        assert_eq!(mapped.feature_indices, vec![2, 19, 71]);
        assert_eq!(
            mapped
                .weights
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            gene.weights.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(mapped.smc_flags, 0b10101010101);
        assert_eq!(
            [
                mapped.long_threshold,
                mapped.short_threshold,
                mapped.target_pips,
                mapped.stop_pips,
                mapped.stop_vol_multiplier
            ]
            .map(f64::to_bits),
            [
                gene.long_threshold,
                gene.short_threshold,
                gene.tp_pips,
                gene.sl_pips,
                gene.stop_vol_mult
            ]
            .map(f64::to_bits)
        );
        // The adapter must not sort, normalize, clamp to random-generation
        // bounds, or turn invalid input into an apparently valid template.
        let invalid = template(crate::genetic::Gene {
            indices: vec![4, 1],
            weights: vec![f64::NAN, -2.0],
            ..Default::default()
        })
        .unwrap();
        assert_eq!(invalid.feature_indices, vec![4, 1]);
        assert!(invalid.weights[0].is_nan());
        assert_eq!(invalid.weights[1], -2.0);
        // The immutable checked native plan, not this lossless mapper, rejects it.
    }

    #[test]
    fn adaptive_probability_and_convergence_keep_cpu_boundaries() {
        assert_eq!(probability_q32(0.0).unwrap(), 0);
        assert_eq!(probability_q32(1.0).unwrap(), 1_u64 << 32);
        for p in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
            assert!(probability_q32(p).is_err());
        }
        assert!(!convergence_reached(
            Duration::from_secs(100),
            Duration::from_secs(100),
            1000,
            None,
            10,
            0.5
        ));
        let budget = Some(Duration::from_secs(100));
        assert!(!convergence_reached(
            Duration::from_secs(49),
            Duration::from_secs(49),
            1000,
            budget,
            10,
            0.5
        ));
        assert!(convergence_reached(
            Duration::from_secs(50),
            Duration::from_secs(1),
            10,
            budget,
            10,
            0.5
        ));
        assert!(convergence_reached(
            Duration::from_secs(50),
            Duration::from_secs(50),
            1,
            budget,
            10,
            0.5
        ));
        assert!(!convergence_reached(
            Duration::from_secs(100),
            Duration::from_secs(100),
            1000,
            budget,
            0,
            0.5
        ));
    }
}
