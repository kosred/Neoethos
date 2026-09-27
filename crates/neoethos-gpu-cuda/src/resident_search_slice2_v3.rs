#[cfg(feature = "hip-native-kernels")]
use crate::hip_runtime_v1::feature_store_v1::{HipPopulationParentV1, HipSearchParentIdentityV3};
#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
use crate::resident_archive_output_v3::{
    ResidentArchiveTerminalOutputV3, ResidentSearchTerminalCandidatesV3,
};
#[cfg(feature = "cuda")]
use crate::resident_feature_store_v3::{
    ResidentFeatureStoreConsumerLeaseV3, ResidentPopulationSessionV3,
};
#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
use crate::resident_generation_v1::{ResidentAdaptiveCheckpointV3, SealedResidentGenerationPlanV1};
#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
use crate::resident_scoring_v2::{ResidentScoringGoalContextV2, ResidentScoringObjectiveV2};
#[cfg(feature = "cuda")]
use crate::resident_search_slice2_admission_v2::ResidentSearchCompactInputIdentityV3;
#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
use crate::resident_search_slice2_admission_v2::ResidentSearchSlice2InputIdentityV3;
#[cfg(feature = "cuda")]
type SelectedFeatureStoreCompletionV3 = ResidentFeatureStoreConsumerLeaseV3;
#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
use crate::resident_search_v2::{
    ResidentSearchSlice2NativeErrorV3, ResidentSearchSlice2NativeOwnerV3,
    ResidentSearchSlice2NativeTryCompleteV3, ResidentSearchSlice2RequestV3,
};
#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
use crate::{
    NeoPopulationSettings, PopulationEvaluationViewV1, PopulationTimestampModeV1,
    ResidentAdaptiveBaseRequestV1, ScenarioDescriptor,
};

#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
struct ResidentSearchStartAuthorityV3 {
    request: ResidentSearchSlice2RequestV3,
    plan: SealedResidentGenerationPlanV1,
    smc_weights: [f64; 11],
    smc_gate_disabled: bool,
    settings: NeoPopulationSettings,
    scenarios: Box<[ScenarioDescriptor]>,
    objective: ResidentScoringObjectiveV2,
    novelty_weight: f64,
    goal_context: Option<ResidentScoringGoalContextV2>,
    stage1_view: PopulationEvaluationViewV1,
    adaptive_base_request: Option<ResidentAdaptiveBaseRequestV1>,
    retain_compact_session: bool,
}

#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
struct ResidentSearchAuthorityStateV3 {
    session: Option<ResidentSearchPopulationOwnerV3>,
    execution_plan: Option<ResidentSearchExecutionPlanV3>,
    native: Option<ResidentSearchSlice2NativeOwnerV3>,
    settings: Option<NeoPopulationSettings>,
    scenario_ids: Vec<u64>,
    terminal_candidates: Option<ResidentSearchTerminalCandidatesV3>,
    adaptive_checkpoint: Option<ResidentAdaptiveCheckpointV3>,
    retain_compact_session: bool,
    #[cfg(feature = "cuda")]
    completion: Option<SelectedFeatureStoreCompletionV3>,
    poisoned: bool,
}

#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
#[derive(Debug)]
enum ResidentSearchRejectedTransitionV3 {
    Native(ResidentSearchSlice2NativeErrorV3),
    MissingAuthority,
    PopulationLifetime,
    InputIdentity(&'static str),
    #[cfg(feature = "cuda")]
    ViewBinding(crate::resident_feature_store_v3::ResidentFeatureStoreCudaErrorV3),
    #[cfg(feature = "hip-native-kernels")]
    HipViewBinding(crate::CudaPopulationError),
}

/// Preserve the actual Data owner: compact materialization does not create a
/// trim-prefilter allocation, map or receipt. The native Search chain retains
/// the compact owner's terminal deallocation obligations.
#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
enum ResidentSearchPopulationOwnerV3 {
    #[cfg(feature = "cuda")]
    Compact(ResidentPopulationSessionV3),
    #[cfg(feature = "hip-native-kernels")]
    Hip {
        core: Option<crate::PopulationSession>,
        identity: HipSearchParentIdentityV3,
    },
}

#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
impl ResidentSearchPopulationOwnerV3 {
    fn validate_plan_owner_v3(
        &self,
        input: &ResidentSearchSlice2InputIdentityV3,
    ) -> Result<(), &'static str> {
        match (self, input) {
            #[cfg(feature = "cuda")]
            (Self::Compact(owner), ResidentSearchSlice2InputIdentityV3::Compact(identity)) => {
                identity.matches_session_v3(owner)
            }
            #[cfg(feature = "hip-native-kernels")]
            (
                Self::Hip { core, identity },
                ResidentSearchSlice2InputIdentityV3::Hip { parent, .. },
            ) => {
                if identity == parent
                    && core
                        .as_ref()
                        .is_some_and(|core| identity.matches_core_v3(core))
                {
                    Ok(())
                } else {
                    Err("HIP Search core differs from its retained physical parent")
                }
            }
        }
    }

    fn bind_stage1_view_v3(
        &mut self,
        view: PopulationEvaluationViewV1,
        adaptive: Option<ResidentAdaptiveBaseRequestV1>,
    ) -> Result<(), ResidentSearchRejectedTransitionV3> {
        #[cfg(feature = "hip-native-kernels")]
        {
            let Self::Hip { core, .. } = self;
            let core = core
                .as_mut()
                .ok_or(ResidentSearchRejectedTransitionV3::PopulationLifetime)?;
            match adaptive {
                None => core
                    .bind_evaluation_view_v1(view)
                    .map_err(ResidentSearchRejectedTransitionV3::HipViewBinding),
                Some(request) => core
                    .bind_evaluation_view_with_resident_adaptive_base_v1(view, request)
                    .map(|_| ())
                    .map_err(ResidentSearchRejectedTransitionV3::HipViewBinding),
            }
        }
        #[cfg(feature = "cuda")]
        {
            let Self::Compact(owner) = self;
            match adaptive {
            None => owner.bind_evaluation_view_v1(view)
                .map_err(ResidentSearchRejectedTransitionV3::ViewBinding),
            Some(request) => owner.bind_evaluation_view_with_resident_adaptive_base_checked_v1(
                view, request, |token| {
                    if token.request_identity_sha256() != request.identity_sha256()
                        || token.resident_session_identity_sha256() == [0; 32]
                        || token.view_identity_sha256() == [0; 32]
                        || token.token_identity_sha256() == [0; 32]
                    {
                        return Err(crate::resident_feature_store_v3::ResidentFeatureStoreCudaErrorV3::InvalidInput(
                            "resident adaptive token is detached from the bound Stage1 request".into()));
                    }
                    Ok(())
                },
            ).map(|_| ()).map_err(ResidentSearchRejectedTransitionV3::ViewBinding),
        }
        }
    }

    fn take_population_session_for_slice2_v3(
        &mut self,
    ) -> Result<crate::PopulationSession, ResidentSearchRejectedTransitionV3> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Compact(owner) => owner
                .take_population_session_for_slice2_v3()
                .map_err(|_| ResidentSearchRejectedTransitionV3::PopulationLifetime),
            #[cfg(feature = "hip-native-kernels")]
            Self::Hip { core, .. } => core
                .take()
                .ok_or(ResidentSearchRejectedTransitionV3::PopulationLifetime),
        }
    }

    #[cfg(feature = "cuda")]
    fn complete_resident_search_slice2_v3(
        self,
        population: crate::PopulationSession,
    ) -> Result<ResidentFeatureStoreConsumerLeaseV3, ResidentSearchRejectedTransitionV3> {
        match self {
            Self::Compact(mut owner) => {
                if let Err(population) = owner.restore_population_session_from_slice2_v3(population)
                {
                    // An unexpected owner mismatch cannot authorize destruction
                    // of the detached session or its still-retained Data owner.
                    std::mem::forget(population);
                    return Err(ResidentSearchRejectedTransitionV3::PopulationLifetime);
                }
                owner
                    .record_consumer_completion()
                    .map_err(|_| ResidentSearchRejectedTransitionV3::PopulationLifetime)
            }
        }
    }
}

/// Immutable requested execution inputs. This is not a memory reservation or
/// performance calibration: native admission is measured at execution start.
pub struct ResidentSearchExecutionPlanV3 {
    #[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
    inner: Option<ResidentSearchStartAuthorityV3>,
    #[cfg(not(any(feature = "cuda", feature = "hip-native-kernels")))]
    #[allow(dead_code)]
    inner: core::convert::Infallible,
}

/// Ordinary requested inputs, not CUDA admission or evidence of CPU policy parity.
/// Search must derive these from its immutable config and pinned Stage1 endpoints.
/// In particular, a goal horizon is not an evaluation timestamp span.
#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
#[derive(Clone, Copy, Debug)]
pub struct ResidentSearchArchivePolicyV3 {
    /// 0: net, 1: active, 2: profit factor, 3: Sharpe; strict CPU thresholds.
    pub mode: u32,
    pub neighbors: u32,
    pub min_net: f64,
    pub min_pf: f64,
    pub min_sharpe: f64,
}

#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
impl ResidentSearchArchivePolicyV3 {
    pub(crate) fn validate_v3(&self) -> Result<(), &'static str> {
        if self.mode > 3
            || self.neighbors == 0
            || ![self.min_net, self.min_pf, self.min_sharpe]
                .into_iter()
                .all(f64::is_finite)
        {
            return Err("invalid native archive policy or neighborhood");
        }
        Ok(())
    }
}

#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
pub struct ResidentSearchExecutionInputsV3 {
    pub settings: NeoPopulationSettings,
    pub scenarios: Vec<ScenarioDescriptor>,
    pub smc_weights: [f64; 11],
    pub smc_gate_disabled: bool,
    pub growth_objective: bool,
    pub growth_goal: Option<neoethos_gpu_contracts::resident_search_scoring_v2::RiskyGrowthGoal>,
    pub stage1_row_start: u64,
    pub stage1_row_end: u64,
    pub first_timestamp_ms: i64,
    pub last_timestamp_ms: i64,
    pub novelty_weight: f64,
    pub archive_capacity: u64,
    pub archive_policy: Option<ResidentSearchArchivePolicyV3>,
    /// Exact request already admitted by Data's population sizing receipt.
    /// None means the existing fixed-pip path, not a host adaptive series.
    pub adaptive_base_request: Option<ResidentAdaptiveBaseRequestV1>,
}

#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
impl ResidentSearchExecutionInputsV3 {
    fn stage1_view_v3(
        &self,
        parent_rows: usize,
    ) -> Result<PopulationEvaluationViewV1, &'static str> {
        let start =
            usize::try_from(self.stage1_row_start).map_err(|_| "Stage1 start exceeds usize")?;
        let end = usize::try_from(self.stage1_row_end).map_err(|_| "Stage1 end exceeds usize")?;
        let view = if start == 0 && end == parent_rows {
            PopulationEvaluationViewV1::full(
                parent_rows,
                PopulationTimestampModeV1::Canonical,
                None,
            )
        } else {
            PopulationEvaluationViewV1::contiguous_range(
                parent_rows,
                start,
                end,
                PopulationTimestampModeV1::Canonical,
                None,
            )
        }
        .map_err(|_| "cannot bind canonical Stage1 view")?;
        if let Some(request) = self.adaptive_base_request {
            let rebuilt = ResidentAdaptiveBaseRequestV1::checked_canonical_v1(
                &view,
                self.settings.pip_value,
                usize::try_from(request.tail_step())
                    .map_err(|_| "adaptive tail step exceeds usize")?,
                usize::try_from(request.tail_max_bars())
                    .map_err(|_| "adaptive tail cap exceeds usize")?,
            )
            .map_err(|_| "cannot reconstruct admitted Stage1 adaptive request")?;
            if rebuilt != request {
                return Err("resident adaptive request differs from the canonical Stage1 view");
            }
        }
        Ok(view)
    }

    fn validate_v3(
        &self,
        population: u64,
        parent_rows: u64,
    ) -> Result<
        (
            ResidentScoringObjectiveV2,
            Option<ResidentScoringGoalContextV2>,
        ),
        &'static str,
    > {
        if let Some(policy) = self.archive_policy {
            policy.validate_v3()?;
        }
        let rows = self
            .stage1_row_end
            .checked_sub(self.stage1_row_start)
            .filter(|rows| *rows >= 2 && self.stage1_row_end <= parent_rows)
            .ok_or("resident execution requires an exact nonempty multi-row Stage1 interval")?;
        let elapsed = self
            .last_timestamp_ms
            .checked_sub(self.first_timestamp_ms)
            .filter(|elapsed| *elapsed > 0)
            .ok_or(
                "resident execution requires increasing representable pinned Stage1 timestamps",
            )?;
        if self.adaptive_base_request.is_some_and(|request| {
            request.parent_row_count() != parent_rows
                || request.view_start() != self.stage1_row_start
                || request.view_row_count() != rows
                || request.pip_size().to_bits() != self.settings.pip_value.to_bits()
        }) {
            return Err("resident adaptive request differs from the exact Stage1 view or pip size");
        }
        let settings = &self.settings;
        if settings.abi_version != neoethos_gpu_contracts::ABI_VERSION
            || settings._trailing_pad != 0
            || settings.trailing_enabled > 1
            || settings.month_capacity == 0
            || ![
                settings.initial_equity,
                settings.pip_value,
                settings.spread_pips,
                settings.commission_per_trade,
                settings.pip_value_per_lot,
                settings.swap_long_pips_per_day,
                settings.swap_short_pips_per_day,
                settings.pnl_conversion_fee_rate,
                settings.risk_per_trade_min,
                settings.risk_per_trade_max,
                settings.high_quality_confidence,
                settings.adaptive_rr,
                settings.trailing_atr_multiplier,
                settings.trailing_be_trigger_r,
                settings.trailing_min_lock_pips,
                settings.spread_pips_asian,
                settings.spread_pips_overlap,
                settings.spread_pips_late_ny,
            ]
            .iter()
            .all(|value| value.is_finite())
            || settings.initial_equity <= 0.0
            || settings.pip_value <= 0.0
            || settings.pip_value_per_lot <= 0.0
            || settings.risk_per_trade_min < 0.0
            || settings.risk_per_trade_max < settings.risk_per_trade_min
            || settings.risk_per_trade_max > 1.0
            || settings.high_quality_confidence <= 0.0
            || !self.novelty_weight.is_finite()
            || !(0.0..=1.0).contains(&self.novelty_weight)
            || self
                .smc_weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight < 0.0)
            || self.smc_weights.iter().all(|weight| *weight == 0.0)
            || self.archive_capacity == 0
        {
            return Err("resident execution settings, SMC weights or archive request are invalid");
        }
        if population == 0 || u64::try_from(self.scenarios.len()).ok() != Some(population) {
            return Err("resident execution must evaluate every admitted candidate exactly once");
        }
        for (ordinal, scenario) in self.scenarios.iter().enumerate() {
            if scenario.base_candidate_id != ordinal as u64
                || scenario.scenario_id != ordinal as u64
                || scenario.window_offset != 0
                || u64::from(scenario.window_len) != rows
                || scenario.scenario_type != 0
                || scenario.rng_counter != 0
                || scenario.spread_ticks != neoethos_gpu_contracts::device::NO_TICK_OVERRIDE
                || scenario.slippage_ticks != 0
                || scenario.commission_micros != neoethos_gpu_contracts::device::NO_MICRO_OVERRIDE
                || scenario.perturbation_offset != 0
                || scenario.perturbation_count != 0
                || scenario.reserved != 0
            {
                return Err(
                    "resident generation requires ordered full-population base scenarios with the sealed costs and exact Stage1 view",
                );
            }
        }
        match (self.growth_objective, self.growth_goal) {
            (false, None) => Ok((ResidentScoringObjectiveV2::PropFirmV4, None)),
            (true, None) => Ok((ResidentScoringObjectiveV2::RiskyGrowthV5, None)),
            (true, Some(goal)) => {
                goal.validate()?;
                Ok((
                    ResidentScoringObjectiveV2::RiskyGrowthGoalV6,
                    Some(ResidentScoringGoalContextV2 {
                        initial_equity: settings.initial_equity,
                        span_days: elapsed as f64 / 86_400_000.0,
                        goal,
                    }),
                ))
            }
            (false, Some(_)) => Err("resident growth goal requires the growth objective"),
        }
    }
}

#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
impl ResidentSearchExecutionPlanV3 {
    /// Keep the genuine compact Data parent for GPU validation after Search.
    /// The eventual consumer must explicitly complete its returned session.
    #[cfg(feature = "cuda")]
    pub fn retain_compact_session_for_validation_v3(mut self) -> Result<Self, &'static str> {
        let start = self.inner.as_mut().ok_or("missing resident Search plan")?;
        start.retain_compact_session = true;
        Ok(self)
    }

    /// Pure host binding to a real compact Data owner. This does not launch,
    /// reserve memory, simplify adaptive policies, or certify device parity.
    #[cfg(feature = "cuda")]
    pub fn for_compact_session_v3(
        session: &ResidentPopulationSessionV3,
        plan: SealedResidentGenerationPlanV1,
        inputs: ResidentSearchExecutionInputsV3,
    ) -> Result<Self, &'static str> {
        let identity = ResidentSearchCompactInputIdentityV3::from_session_v3(session)?;
        let input = ResidentSearchSlice2InputIdentityV3::Compact(identity);
        input.validate_plan_v3(&plan)?;
        let (objective, goal_context) =
            inputs.validate_v3(plan.logical_population_count_v1(), session.rows() as u64)?;
        let limits = session
            .data_population_limits()
            .ok_or("compact Search requires sealed population limits")?;
        if u64::from(inputs.settings.month_capacity) != limits.month_capacity()
            || plan.retained_evaluation_capacity_v1() == 0
            || plan.retained_evaluation_capacity_v1() > limits.max_concurrent_scenario_count()
            || plan.retained_evaluation_capacity_v1() > plan.logical_population_count_v1()
            || plan.scoring_semantics_sha256_v1()
                != crate::resident_scoring_v2::scoring_semantics_sha256_v2(objective)
            || plan.rank_semantics_sha256_v1()
                != crate::resident_scoring_v2::rank_semantics_sha256_v2()
            || plan.novelty_semantics_sha256_v1()
                != crate::resident_scoring_v2::novelty_disabled_semantics_sha256_v2()
        {
            return Err(
                "resident execution differs from admitted scenario capacity or current base-scoring semantics",
            );
        }
        let stage1_view = inputs.stage1_view_v3(session.rows())?;
        if inputs.adaptive_base_request.is_some()
            && inputs.stage1_row_end - inputs.stage1_row_start > limits.max_adaptive_row_count()
        {
            return Err("Stage1 adaptive base exceeds the sealed Data workspace");
        }
        Ok(Self {
            inner: Some(ResidentSearchStartAuthorityV3 {
                request: ResidentSearchSlice2RequestV3 {
                    input,
                    archive_capacity: inputs.archive_capacity,
                    archive_policy: inputs.archive_policy,
                },
                plan,
                smc_weights: inputs.smc_weights,
                smc_gate_disabled: inputs.smc_gate_disabled,
                settings: inputs.settings,
                scenarios: inputs.scenarios.into_boxed_slice(),
                objective,
                // Base scorer stays raw; the archive setter binds this actual
                // requested weight exactly once before the first rank enqueue.
                novelty_weight: inputs.novelty_weight,
                goal_context,
                stage1_view,
                adaptive_base_request: inputs.adaptive_base_request,
                retain_compact_session: false,
            }),
        })
    }
}

pub struct ResidentSearchGenerationChainV3 {
    #[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
    inner: ResidentSearchAuthorityStateV3,
    #[cfg(not(any(feature = "cuda", feature = "hip-native-kernels")))]
    inner: core::convert::Infallible,
}

pub struct ResidentSearchRankEnqueuedV3 {
    #[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
    inner: ResidentSearchAuthorityStateV3,
    #[cfg(not(any(feature = "cuda", feature = "hip-native-kernels")))]
    inner: core::convert::Infallible,
}

pub struct ResidentSearchArchiveStagedV3 {
    #[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
    inner: ResidentSearchAuthorityStateV3,
    #[cfg(not(any(feature = "cuda", feature = "hip-native-kernels")))]
    inner: core::convert::Infallible,
}

pub struct ResidentSearchTerminalPendingV3 {
    #[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
    inner: ResidentSearchAuthorityStateV3,
    #[cfg(not(any(feature = "cuda", feature = "hip-native-kernels")))]
    inner: core::convert::Infallible,
}

pub struct ResidentSearchTerminalReceiptV3 {
    #[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
    inner: ResidentSearchAuthorityStateV3,
    #[cfg(not(any(feature = "cuda", feature = "hip-native-kernels")))]
    #[allow(dead_code)]
    inner: core::convert::Infallible,
}

pub enum ResidentSearchTryCompleteV3 {
    NotReady(ResidentSearchTerminalPendingV3),
    Complete(ResidentSearchTerminalReceiptV3),
}

pub struct ResidentSearchTransitionErrorV3 {
    #[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
    inner: ResidentSearchRejectedTransitionV3,
    #[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
    retained_terminal_authority: Option<ResidentSearchAuthorityStateV3>,
    #[cfg(not(any(feature = "cuda", feature = "hip-native-kernels")))]
    #[allow(dead_code)]
    inner: core::convert::Infallible,
}

pub struct ResidentSearchRejectedAuthorityV3<A> {
    error: ResidentSearchTransitionErrorV3,
    authority: A,
}

impl core::fmt::Debug for ResidentSearchTransitionErrorV3 {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        #[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
        {
            core::fmt::Debug::fmt(&self.inner, formatter)
        }
        #[cfg(not(any(feature = "cuda", feature = "hip-native-kernels")))]
        {
            match self.inner {}
        }
    }
}

impl core::fmt::Display for ResidentSearchTransitionErrorV3 {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        #[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
        {
            match &self.inner {
                ResidentSearchRejectedTransitionV3::Native(error) => {
                    core::fmt::Display::fmt(error, formatter)
                }
                ResidentSearchRejectedTransitionV3::MissingAuthority => formatter
                    .write_str("resident Search transition is missing its execution authority"),
                ResidentSearchRejectedTransitionV3::PopulationLifetime => formatter.write_str(
                    "resident Search could not retain or release its population lifetime",
                ),
                ResidentSearchRejectedTransitionV3::InputIdentity(detail) => {
                    formatter.write_str(detail)
                }
                #[cfg(feature = "cuda")]
                ResidentSearchRejectedTransitionV3::ViewBinding(error) => {
                    core::fmt::Display::fmt(error, formatter)
                }
                #[cfg(feature = "hip-native-kernels")]
                ResidentSearchRejectedTransitionV3::HipViewBinding(error) => {
                    core::fmt::Display::fmt(error, formatter)
                }
            }
        }
        #[cfg(not(any(feature = "cuda", feature = "hip-native-kernels")))]
        {
            match self.inner {}
        }
    }
}

impl std::error::Error for ResidentSearchTransitionErrorV3 {}

impl<A> core::fmt::Debug for ResidentSearchRejectedAuthorityV3<A> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ResidentSearchRejectedAuthorityV3")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
impl ResidentSearchTerminalReceiptV3 {
    /// Transfer the same restored Data owner, not a reconstructed host parent.
    /// Only an explicitly retained compact run can produce these parts.
    #[cfg(feature = "cuda")]
    pub fn into_retained_compact_parts_v3(
        mut self,
    ) -> Result<
        (
            ResidentPopulationSessionV3,
            ResidentSearchTerminalCandidatesV3,
            Option<ResidentAdaptiveCheckpointV3>,
        ),
        Self,
    > {
        if self.inner.poisoned
            || self.inner.native.is_some()
            || self.inner.completion.is_some()
            || !self.inner.retain_compact_session
            || self.inner.terminal_candidates.is_none()
            || !matches!(
                self.inner.session,
                Some(ResidentSearchPopulationOwnerV3::Compact(_))
            )
        {
            return Err(self);
        }
        let Some(ResidentSearchPopulationOwnerV3::Compact(session)) = self.inner.session.take()
        else {
            unreachable!("checked retained compact owner");
        };
        let candidates = self
            .inner
            .terminal_candidates
            .take()
            .expect("checked terminal candidates");
        Ok((session, candidates, self.inner.adaptive_checkpoint.take()))
    }

    /// The last device-produced control checkpoint, if requested by this run.
    pub fn adaptive_checkpoint_v3(&self) -> Option<&ResidentAdaptiveCheckpointV3> {
        self.inner.adaptive_checkpoint.as_ref()
    }

    /// All committed archive members, not the complete Search candidate census.
    pub fn archive_output_v3(&self) -> &ResidentArchiveTerminalOutputV3 {
        self.terminal_candidates_v3().archive()
    }

    /// Consumes terminal lifetime authority only after its native work completed.
    pub fn into_archive_output_v3(mut self) -> ResidentArchiveTerminalOutputV3 {
        self.inner
            .terminal_candidates
            .take()
            .expect("completed receipt retains both candidate sources")
            .into_parts()
            .0
    }

    /// Complete compact terminal inputs for Search's existing archive/final
    /// evaluated population union. No unevaluated offspring is included.
    pub fn terminal_candidates_v3(&self) -> &ResidentSearchTerminalCandidatesV3 {
        self.inner
            .terminal_candidates
            .as_ref()
            .expect("completed receipt retains both candidate sources")
    }

    pub fn into_terminal_candidates_v3(mut self) -> ResidentSearchTerminalCandidatesV3 {
        self.inner
            .terminal_candidates
            .take()
            .expect("completed receipt retains both candidate sources")
    }
}

impl ResidentSearchGenerationChainV3 {
    /// Wait for the completed generation and read only bounded control state.
    /// This does not copy the resident population or its per-candidate metrics.
    #[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
    pub fn checkpoint_v3(
        self,
    ) -> Result<(Self, ResidentAdaptiveCheckpointV3), ResidentSearchRejectedAuthorityV3<Self>> {
        let mut state = self.inner;
        if state.poisoned || state.native.is_none() {
            return Err(reject_resident_search_transition_v3(
                Self { inner: state },
                ResidentSearchRejectedTransitionV3::MissingAuthority,
            ));
        }
        match state
            .native
            .as_mut()
            .expect("checked native owner")
            .copy_adaptive_checkpoint_v3()
        {
            Ok(checkpoint) => {
                state.adaptive_checkpoint = Some(checkpoint);
                Ok((Self { inner: state }, checkpoint))
            }
            Err(error) => {
                state.poisoned = true;
                Err(reject_resident_search_transition_v3(
                    Self { inner: state },
                    ResidentSearchRejectedTransitionV3::Native(error),
                ))
            }
        }
    }

    pub fn enqueue_score_and_rank_v3(
        self,
    ) -> Result<ResidentSearchRankEnqueuedV3, ResidentSearchRejectedAuthorityV3<Self>> {
        #[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
        {
            let mut state = self.inner;
            if state.poisoned {
                return Err(reject_resident_search_transition_v3(
                    ResidentSearchGenerationChainV3 { inner: state },
                    ResidentSearchRejectedTransitionV3::MissingAuthority,
                ));
            }
            if state.native.is_none() {
                let Some(mut execution_plan) = state.execution_plan.take() else {
                    return Err(reject_resident_search_transition_v3(
                        ResidentSearchGenerationChainV3 { inner: state },
                        ResidentSearchRejectedTransitionV3::MissingAuthority,
                    ));
                };
                let Some(start) = execution_plan.inner.take() else {
                    return Err(reject_resident_search_transition_v3(
                        ResidentSearchGenerationChainV3 { inner: state },
                        ResidentSearchRejectedTransitionV3::MissingAuthority,
                    ));
                };
                let Some(session) = state.session.as_mut() else {
                    return Err(reject_resident_search_transition_v3(
                        ResidentSearchGenerationChainV3 { inner: state },
                        ResidentSearchRejectedTransitionV3::MissingAuthority,
                    ));
                };
                if let Err(error) = session.validate_plan_owner_v3(&start.request.input) {
                    // No native allocation or detach has occurred. Preserve
                    // both the requested plan and original Data owner normally.
                    execution_plan.inner = Some(start);
                    state.execution_plan = Some(execution_plan);
                    return Err(reject_resident_search_transition_v3(
                        ResidentSearchGenerationChainV3 { inner: state },
                        ResidentSearchRejectedTransitionV3::InputIdentity(error),
                    ));
                }
                // Rebind the real session immediately before detach. A caller
                // cannot prebind/rebind another interval and retain this plan's
                // goal span while evaluating the wrong parent rows.
                if let Err(error) =
                    session.bind_stage1_view_v3(start.stage1_view, start.adaptive_base_request)
                {
                    state.poisoned = true;
                    return Err(reject_resident_search_transition_v3(
                        ResidentSearchGenerationChainV3 { inner: state },
                        error,
                    ));
                }
                let population = match session.take_population_session_for_slice2_v3() {
                    Ok(population) => population,
                    Err(_) => {
                        state.poisoned = true;
                        return Err(reject_resident_search_transition_v3(
                            ResidentSearchGenerationChainV3 { inner: state },
                            ResidentSearchRejectedTransitionV3::PopulationLifetime,
                        ));
                    }
                };
                let mut native = match population.begin_resident_search_slice2_native_v3(
                    start.plan,
                    start.smc_weights,
                    start.smc_gate_disabled,
                    start.objective,
                    start.novelty_weight,
                    start.goal_context,
                    start.request,
                ) {
                    Ok(native) => native,
                    Err(error) => {
                        state.poisoned = true;
                        return Err(reject_resident_search_transition_v3(
                            ResidentSearchGenerationChainV3 { inner: state },
                            ResidentSearchRejectedTransitionV3::Native(error),
                        ));
                    }
                };
                if let Err(error) = native.upload_resident_scenarios_v3(&start.scenarios) {
                    state.native = Some(native);
                    state.poisoned = true;
                    return Err(reject_resident_search_transition_v3(
                        ResidentSearchGenerationChainV3 { inner: state },
                        ResidentSearchRejectedTransitionV3::Native(error),
                    ));
                }
                state.settings = Some(start.settings);
                state.retain_compact_session = start.retain_compact_session;
                state.scenario_ids = start
                    .scenarios
                    .iter()
                    .map(|scenario| scenario.scenario_id)
                    .collect();
                state.native = Some(native);
            }
            let native = state.native.take().expect("validated Slice2 native owner");
            let settings = state.settings.expect("validated Slice2 settings");
            match native.enqueue_score_and_rank_v3(&settings) {
                Ok(native) => {
                    state.native = Some(native);
                    state.adaptive_checkpoint = None;
                    Ok(ResidentSearchRankEnqueuedV3 { inner: state })
                }
                Err(rejected) => {
                    let (error, native) = rejected.into_parts_v3();
                    state.native = Some(native);
                    state.poisoned = true;
                    Err(reject_resident_search_transition_v3(
                        ResidentSearchGenerationChainV3 { inner: state },
                        ResidentSearchRejectedTransitionV3::Native(error),
                    ))
                }
            }
        }
        #[cfg(not(any(feature = "cuda", feature = "hip-native-kernels")))]
        {
            match self.inner {}
        }
    }

    pub fn enqueue_terminal_seal_v3(
        self,
    ) -> Result<ResidentSearchTerminalPendingV3, ResidentSearchRejectedAuthorityV3<Self>> {
        #[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
        {
            let mut state = self.inner;
            let Some(native) = state.native.take() else {
                return Err(reject_resident_search_transition_v3(
                    ResidentSearchGenerationChainV3 { inner: state },
                    ResidentSearchRejectedTransitionV3::MissingAuthority,
                ));
            };
            match native.enqueue_terminal_seal_v3() {
                Ok(native) => {
                    state.native = Some(native);
                    Ok(ResidentSearchTerminalPendingV3 { inner: state })
                }
                Err(rejected) => {
                    let (error, native) = rejected.into_parts_v3();
                    state.native = Some(native);
                    state.poisoned = true;
                    Err(reject_resident_search_transition_v3(
                        ResidentSearchGenerationChainV3 { inner: state },
                        ResidentSearchRejectedTransitionV3::Native(error),
                    ))
                }
            }
        }
        #[cfg(not(any(feature = "cuda", feature = "hip-native-kernels")))]
        {
            match self.inner {}
        }
    }
}

impl ResidentSearchRankEnqueuedV3 {
    pub fn enqueue_stage_archive_from_rank_v3(
        self,
    ) -> Result<ResidentSearchArchiveStagedV3, ResidentSearchRejectedAuthorityV3<Self>> {
        #[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
        {
            let mut state = self.inner;
            let Some(native) = state.native.take() else {
                return Err(reject_resident_search_transition_v3(
                    ResidentSearchRankEnqueuedV3 { inner: state },
                    ResidentSearchRejectedTransitionV3::MissingAuthority,
                ));
            };
            match native.enqueue_stage_archive_from_rank_v3() {
                Ok(native) => {
                    state.native = Some(native);
                    Ok(ResidentSearchArchiveStagedV3 { inner: state })
                }
                Err(rejected) => {
                    let (error, native) = rejected.into_parts_v3();
                    state.native = Some(native);
                    state.poisoned = true;
                    Err(reject_resident_search_transition_v3(
                        ResidentSearchRankEnqueuedV3 { inner: state },
                        ResidentSearchRejectedTransitionV3::Native(error),
                    ))
                }
            }
        }
        #[cfg(not(any(feature = "cuda", feature = "hip-native-kernels")))]
        {
            match self.inner {}
        }
    }
}

impl ResidentSearchArchiveStagedV3 {
    pub fn enqueue_evolve_and_publish_v3(
        self,
    ) -> Result<ResidentSearchGenerationChainV3, ResidentSearchRejectedAuthorityV3<Self>> {
        #[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
        {
            let mut state = self.inner;
            let Some(native) = state.native.take() else {
                return Err(reject_resident_search_transition_v3(
                    ResidentSearchArchiveStagedV3 { inner: state },
                    ResidentSearchRejectedTransitionV3::MissingAuthority,
                ));
            };
            match native.enqueue_evolve_and_publish_v3() {
                Ok(native) => {
                    state.native = Some(native);
                    Ok(ResidentSearchGenerationChainV3 { inner: state })
                }
                Err(rejected) => {
                    let (error, native) = rejected.into_parts_v3();
                    state.native = Some(native);
                    state.poisoned = true;
                    Err(reject_resident_search_transition_v3(
                        ResidentSearchArchiveStagedV3 { inner: state },
                        ResidentSearchRejectedTransitionV3::Native(error),
                    ))
                }
            }
        }
        #[cfg(not(any(feature = "cuda", feature = "hip-native-kernels")))]
        {
            match self.inner {}
        }
    }

    #[cfg(not(any(feature = "cuda", feature = "hip-native-kernels")))]
    #[allow(dead_code)]
    pub(crate) fn from_ranked_v3(
        ranked: ResidentSearchRankEnqueuedV3,
    ) -> ResidentSearchArchiveStagedV3 {
        match ranked.inner {}
    }
}

#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
impl Drop for ResidentSearchExecutionPlanV3 {
    fn drop(&mut self) {
        let _ = &self.inner;
    }
}

#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
impl Drop for ResidentSearchTerminalReceiptV3 {
    fn drop(&mut self) {
        let _ = &self.inner;
    }
}

#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
impl Drop for ResidentSearchTransitionErrorV3 {
    fn drop(&mut self) {
        if let ResidentSearchRejectedTransitionV3::Native(error) = &self.inner {
            let _ = error;
        }
        let _ = &self.retained_terminal_authority;
    }
}

impl ResidentSearchTerminalPendingV3 {
    pub fn try_complete_v3(
        self,
    ) -> Result<ResidentSearchTryCompleteV3, ResidentSearchTransitionErrorV3> {
        #[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
        {
            let mut state = self.inner;
            let Some(native) = state.native.take() else {
                return Err(ResidentSearchTransitionErrorV3 {
                    inner: ResidentSearchRejectedTransitionV3::MissingAuthority,
                    retained_terminal_authority: Some(state),
                });
            };
            match native.try_complete_terminal_v3() {
                Ok(ResidentSearchSlice2NativeTryCompleteV3::NotReady(native)) => {
                    state.native = Some(native);
                    Ok(ResidentSearchTryCompleteV3::NotReady(
                        ResidentSearchTerminalPendingV3 { inner: state },
                    ))
                }
                Ok(ResidentSearchSlice2NativeTryCompleteV3::Complete(mut native)) => {
                    let copied = (|| {
                        let archive = native.copy_terminal_archive_v3(&state.scenario_ids)?;
                        let population = native.copy_terminal_population_v3(&state.scenario_ids)?;
                        if state
                            .adaptive_checkpoint
                            .as_ref()
                            .is_some_and(|checkpoint| {
                                checkpoint.run_identity() != population.run_identity()
                                    || checkpoint.evaluated_generation()
                                        != population.evaluated_generation()
                            })
                        {
                            return Err(crate::resident_archive_output_v3::ResidentArchiveOutputErrorV3::Authority(
                                "terminal population differs from last adaptive checkpoint").into());
                        }
                        ResidentSearchTerminalCandidatesV3::seal(archive, population)
                            .map_err(ResidentSearchSlice2NativeErrorV3::from)
                    })();
                    let output = match copied {
                        Ok(output) => output,
                        Err(error @ ResidentSearchSlice2NativeErrorV3::ArchiveOutput(_)) => {
                            // The terminal event already proved all GPU work
                            // complete. A host allocation/validation rejection
                            // discards output, not that proof. Release the exact
                            // graph rather than turning a host error into a VRAM leak.
                            let failure = match native.release_terminal_v3() {
                                Ok(population) => {
                                    match complete_population_lifetime_v3(&mut state, population) {
                                        Ok(()) => ResidentSearchRejectedTransitionV3::Native(error),
                                        Err(cleanup_error) => cleanup_error,
                                    }
                                }
                                Err(rejected) => {
                                    let (cleanup_error, native) = rejected.into_parts_v3();
                                    state.native = Some(native);
                                    ResidentSearchRejectedTransitionV3::Native(cleanup_error)
                                }
                            };
                            state.poisoned = true;
                            return Err(ResidentSearchTransitionErrorV3 {
                                inner: failure,
                                retained_terminal_authority: Some(state),
                            });
                        }
                        Err(error) => {
                            state.native = Some(native);
                            state.poisoned = true;
                            return Err(ResidentSearchTransitionErrorV3 {
                                inner: ResidentSearchRejectedTransitionV3::Native(error),
                                retained_terminal_authority: Some(state),
                            });
                        }
                    };
                    state.terminal_candidates = Some(output);
                    let population = match native.release_terminal_v3() {
                        Ok(population) => population,
                        Err(rejected) => {
                            let (error, native) = rejected.into_parts_v3();
                            state.native = Some(native);
                            state.poisoned = true;
                            return Err(ResidentSearchTransitionErrorV3 {
                                inner: ResidentSearchRejectedTransitionV3::Native(error),
                                retained_terminal_authority: Some(state),
                            });
                        }
                    };
                    match complete_population_lifetime_v3(&mut state, population) {
                        Ok(()) => Ok(ResidentSearchTryCompleteV3::Complete(
                            ResidentSearchTerminalReceiptV3 { inner: state },
                        )),
                        Err(error) => {
                            state.poisoned = true;
                            Err(ResidentSearchTransitionErrorV3 {
                                inner: error,
                                retained_terminal_authority: Some(state),
                            })
                        }
                    }
                }
                Err(rejected) => {
                    let (error, native) = rejected.into_parts_v3();
                    state.native = Some(native);
                    state.poisoned = true;
                    Err(ResidentSearchTransitionErrorV3 {
                        inner: ResidentSearchRejectedTransitionV3::Native(error),
                        retained_terminal_authority: Some(state),
                    })
                }
            }
        }
        #[cfg(not(any(feature = "cuda", feature = "hip-native-kernels")))]
        {
            match self.inner {}
        }
    }
}

#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
fn complete_population_lifetime_v3(
    state: &mut ResidentSearchAuthorityStateV3,
    population: crate::PopulationSession,
) -> Result<(), ResidentSearchRejectedTransitionV3> {
    let session = state
        .session
        .take()
        .ok_or(ResidentSearchRejectedTransitionV3::MissingAuthority)?;
    #[cfg(feature = "hip-native-kernels")]
    {
        let ResidentSearchPopulationOwnerV3::Hip { identity, .. } = session;
        if !identity.matches_core_v3(&population) {
            std::mem::forget(population);
            return Err(ResidentSearchRejectedTransitionV3::PopulationLifetime);
        }
        state.session = Some(ResidentSearchPopulationOwnerV3::Hip {
            core: Some(population),
            identity,
        });
        Ok(())
    }
    #[cfg(feature = "cuda")]
    {
        if state.retain_compact_session && state.terminal_candidates.is_some() {
            let ResidentSearchPopulationOwnerV3::Compact(mut owner) = session;
            if let Err(population) = owner.restore_population_session_from_slice2_v3(population) {
                std::mem::forget(population);
                return Err(ResidentSearchRejectedTransitionV3::PopulationLifetime);
            }
            state.session = Some(ResidentSearchPopulationOwnerV3::Compact(owner));
            return Ok(());
        }
        let completion = session.complete_resident_search_slice2_v3(population)?;
        state.completion = Some(completion);
        Ok(())
    }
}

impl<A> ResidentSearchRejectedAuthorityV3<A> {
    pub fn into_parts_v3(self) -> (ResidentSearchTransitionErrorV3, A) {
        #[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
        let _ = &self.error.inner;
        (self.error, self.authority)
    }
}

#[cfg(feature = "cuda")]
impl ResidentPopulationSessionV3 {
    /// Consume the real compact Data owner into the resident generation
    /// chain. The independently checked execution plan is
    /// still mandatory; this does not mint plan or device-readiness evidence.
    pub fn begin_resident_search_slice2_v3(
        self,
        execution_plan: ResidentSearchExecutionPlanV3,
    ) -> ResidentSearchGenerationChainV3 {
        start_resident_population_search_slice2_v3(
            ResidentSearchPopulationOwnerV3::Compact(self),
            execution_plan,
        )
    }
}

#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
fn start_resident_population_search_slice2_v3(
    session: ResidentSearchPopulationOwnerV3,
    execution_plan: ResidentSearchExecutionPlanV3,
) -> ResidentSearchGenerationChainV3 {
    ResidentSearchGenerationChainV3 {
        inner: ResidentSearchAuthorityStateV3 {
            session: Some(session),
            execution_plan: Some(execution_plan),
            native: None,
            settings: None,
            scenario_ids: Vec::new(),
            terminal_candidates: None,
            adaptive_checkpoint: None,
            retain_compact_session: false,
            #[cfg(feature = "cuda")]
            completion: None,
            poisoned: false,
        },
    }
}

#[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
fn reject_resident_search_transition_v3<A>(
    authority: A,
    inner: ResidentSearchRejectedTransitionV3,
) -> ResidentSearchRejectedAuthorityV3<A> {
    ResidentSearchRejectedAuthorityV3 {
        error: ResidentSearchTransitionErrorV3 {
            inner,
            retained_terminal_authority: None,
        },
        authority,
    }
}

/// Error without an escapable native state. A post-detach failure quarantines
/// the actual lease and deliberately retains all unfinished native authority.
#[cfg(feature = "hip-native-kernels")]
#[derive(Debug, thiserror::Error)]
#[error("{detail}")]
pub struct HipResidentSearchErrorV3 {
    detail: String,
}
#[cfg(feature = "hip-native-kernels")]
impl HipResidentSearchErrorV3 {
    fn detail(error: impl core::fmt::Display) -> Self {
        Self {
            detail: error.to_string(),
        }
    }
}

/// Lifetime envelope over the existing common stage; it is not another GA loop.
/// No accessor can yield its common stage, session, or state-bearing error.
#[cfg(feature = "hip-native-kernels")]
#[must_use]
pub struct HipResidentSearchBoundV3<'search, 'parent, 'timestamp, 'lease, Stage> {
    inner: Option<Stage>,
    parent: Option<&'search mut HipPopulationParentV1<'parent, 'timestamp, 'lease>>,
    identity: HipSearchParentIdentityV3,
}
#[cfg(feature = "hip-native-kernels")]
impl<Stage> core::fmt::Debug for HipResidentSearchBoundV3<'_, '_, '_, '_, Stage> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HipResidentSearchBoundV3")
            .field("physical_binding", &self.identity.physical_binding_sha256)
            .finish_non_exhaustive()
    }
}
#[cfg(feature = "hip-native-kernels")]
impl<Stage> Drop for HipResidentSearchBoundV3<'_, '_, '_, '_, Stage> {
    fn drop(&mut self) {
        // A transition can unwind after taking inner but before returning a new
        // stage. The still-held parent, not inner presence, owns that obligation.
        if let Some(parent) = self.parent.take() {
            parent.quarantine_search_v3();
        }
        if let Some(stage) = self.inner.take() {
            // Never drop a state-bearing native owner after its lease is unsafe.
            std::mem::forget(stage);
        }
    }
}
#[cfg(feature = "hip-native-kernels")]
impl<'s, 'p, 't, 'l, Stage> HipResidentSearchBoundV3<'s, 'p, 't, 'l, Stage> {
    fn wrap<Next>(&mut self, stage: Next) -> HipResidentSearchBoundV3<'s, 'p, 't, 'l, Next> {
        HipResidentSearchBoundV3 {
            inner: Some(stage),
            parent: self.parent.take(),
            identity: self.identity.clone(),
        }
    }
    fn fail(&mut self, error: ResidentSearchTransitionErrorV3) -> HipResidentSearchErrorV3 {
        if let Some(parent) = self.parent.as_deref_mut() {
            parent.quarantine_search_v3();
        }
        let detail = HipResidentSearchErrorV3::detail(&error);
        // Terminal errors may themselves retain the complete native graph.
        std::mem::forget(error);
        detail
    }
    fn transition<Next>(
        mut self,
        operation: impl FnOnce(Stage) -> Result<Next, ResidentSearchRejectedAuthorityV3<Stage>>,
    ) -> Result<HipResidentSearchBoundV3<'s, 'p, 't, 'l, Next>, HipResidentSearchErrorV3> {
        let stage = self
            .inner
            .take()
            .expect("HIP envelope retains its exact stage");
        match operation(stage) {
            Ok(next) => Ok(self.wrap(next)),
            Err(rejected) => {
                let (error, stage) = rejected.into_parts_v3();
                let error = self.fail(error);
                std::mem::forget(stage);
                Err(error)
            }
        }
    }
}
#[cfg(feature = "hip-native-kernels")]
pub enum HipResidentSearchTryCompleteV3<'s, 'p, 't, 'l> {
    NotReady(HipResidentSearchBoundV3<'s, 'p, 't, 'l, ResidentSearchTerminalPendingV3>),
    /// Both sources were copied and sealed, then the same parent was restored.
    Complete(
        ResidentSearchTerminalCandidatesV3,
        Option<ResidentAdaptiveCheckpointV3>,
    ),
}
#[cfg(feature = "hip-native-kernels")]
impl<'s, 'p, 't, 'l> HipResidentSearchBoundV3<'s, 'p, 't, 'l, ResidentSearchGenerationChainV3> {
    pub fn enqueue_score_and_rank_v3(
        self,
    ) -> Result<
        HipResidentSearchBoundV3<'s, 'p, 't, 'l, ResidentSearchRankEnqueuedV3>,
        HipResidentSearchErrorV3,
    > {
        self.transition(ResidentSearchGenerationChainV3::enqueue_score_and_rank_v3)
    }
    pub fn enqueue_terminal_seal_v3(
        self,
    ) -> Result<
        HipResidentSearchBoundV3<'s, 'p, 't, 'l, ResidentSearchTerminalPendingV3>,
        HipResidentSearchErrorV3,
    > {
        self.transition(ResidentSearchGenerationChainV3::enqueue_terminal_seal_v3)
    }
    pub fn checkpoint_v3(
        self,
    ) -> Result<(Self, ResidentAdaptiveCheckpointV3), HipResidentSearchErrorV3> {
        let mut result = self.transition(ResidentSearchGenerationChainV3::checkpoint_v3)?;
        let (stage, checkpoint) = result
            .inner
            .take()
            .expect("checkpoint transition returned its stage");
        Ok((result.wrap(stage), checkpoint))
    }
}
#[cfg(feature = "hip-native-kernels")]
impl<'s, 'p, 't, 'l> HipResidentSearchBoundV3<'s, 'p, 't, 'l, ResidentSearchRankEnqueuedV3> {
    pub fn enqueue_stage_archive_from_rank_v3(
        self,
    ) -> Result<
        HipResidentSearchBoundV3<'s, 'p, 't, 'l, ResidentSearchArchiveStagedV3>,
        HipResidentSearchErrorV3,
    > {
        self.transition(ResidentSearchRankEnqueuedV3::enqueue_stage_archive_from_rank_v3)
    }
}
#[cfg(feature = "hip-native-kernels")]
impl<'s, 'p, 't, 'l> HipResidentSearchBoundV3<'s, 'p, 't, 'l, ResidentSearchArchiveStagedV3> {
    pub fn enqueue_evolve_and_publish_v3(
        self,
    ) -> Result<
        HipResidentSearchBoundV3<'s, 'p, 't, 'l, ResidentSearchGenerationChainV3>,
        HipResidentSearchErrorV3,
    > {
        self.transition(ResidentSearchArchiveStagedV3::enqueue_evolve_and_publish_v3)
    }
}
#[cfg(feature = "hip-native-kernels")]
impl<'s, 'p, 't, 'l> HipResidentSearchBoundV3<'s, 'p, 't, 'l, ResidentSearchTerminalPendingV3> {
    pub fn try_complete_v3(
        mut self,
    ) -> Result<HipResidentSearchTryCompleteV3<'s, 'p, 't, 'l>, HipResidentSearchErrorV3> {
        let pending = self
            .inner
            .take()
            .expect("HIP envelope retains pending terminal");
        match pending.try_complete_v3() {
            Ok(ResidentSearchTryCompleteV3::NotReady(pending)) => {
                Ok(HipResidentSearchTryCompleteV3::NotReady(self.wrap(pending)))
            }
            Ok(ResidentSearchTryCompleteV3::Complete(mut receipt)) => {
                let Some(ResidentSearchPopulationOwnerV3::Hip {
                    core: Some(core),
                    identity,
                }) = receipt.inner.session.take()
                else {
                    let error = self.fail(ResidentSearchTransitionErrorV3 {
                        inner: ResidentSearchRejectedTransitionV3::PopulationLifetime,
                        retained_terminal_authority: None,
                    });
                    std::mem::forget(receipt);
                    return Err(error);
                };
                if identity != self.identity {
                    if let Some(parent) = self.parent.as_deref_mut() {
                        parent.quarantine_search_v3();
                    }
                    std::mem::forget(core);
                    std::mem::forget(receipt);
                    return Err(HipResidentSearchErrorV3::detail(
                        "HIP terminal identity changed",
                    ));
                }
                let parent = self
                    .parent
                    .as_deref_mut()
                    .expect("HIP terminal retains parent borrow");
                if let Err(error) = parent.restore_search_core_v3(core, &self.identity) {
                    std::mem::forget(receipt);
                    return Err(HipResidentSearchErrorV3::detail(error));
                }
                // The parent is proved restored; disarm before any later unwind.
                self.parent.take();
                let candidates = receipt
                    .inner
                    .terminal_candidates
                    .take()
                    .expect("sealed terminal candidates");
                Ok(HipResidentSearchTryCompleteV3::Complete(
                    candidates,
                    receipt.inner.adaptive_checkpoint.take(),
                ))
            }
            Err(error) => Err(self.fail(error)),
        }
    }
}
#[cfg(feature = "hip-native-kernels")]
impl<'p, 't, 'l> HipPopulationParentV1<'p, 't, 'l> {
    /// Execute the existing native GA on this physical parent. Higher-level Data
    /// and Search retain responsibility for genuine selected/holdout/config scope.
    pub fn begin_resident_search_slice2_v3<'s>(
        &'s mut self,
        plan: SealedResidentGenerationPlanV1,
        inputs: ResidentSearchExecutionInputsV3,
    ) -> Result<
        HipResidentSearchBoundV3<'s, 'p, 't, 'l, ResidentSearchGenerationChainV3>,
        HipResidentSearchErrorV3,
    > {
        use sha2::{Digest, Sha256};
        let identity = self
            .capture_search_identity_v3()
            .map_err(HipResidentSearchErrorV3::detail)?;
        let (objective, goal_context) = inputs
            .validate_v3(plan.logical_population_count_v1(), identity.rows as u64)
            .map_err(HipResidentSearchErrorV3::detail)?;
        let stage1_view = inputs
            .stage1_view_v3(identity.rows)
            .map_err(HipResidentSearchErrorV3::detail)?;
        let budget = identity.search_budget.ok_or_else(|| HipResidentSearchErrorV3::detail(
            "HIP GA requires the explicit pre-bind evaluation budget, not a generic allocator reserve"))?;
        let adaptive_rows = inputs
            .adaptive_base_request
            .map_or(0, |request| request.view_row_count());
        if !budget.covers_v3(
            plan.retained_evaluation_capacity_v1(),
            inputs.settings.month_capacity,
            adaptive_rows,
        ) {
            return Err(HipResidentSearchErrorV3::detail(
                "HIP Search C/month/adaptive view exceeds its explicit pre-bind budget",
            ));
        }
        if plan.feature_count_v1() != identity.features as u64
            || plan.native_build_manifest_sha256_v1() != identity.native_build_sha256
            || plan.retained_evaluation_capacity_v1() == 0
            || plan.retained_evaluation_capacity_v1() > plan.logical_population_count_v1()
            || !(1..=16).contains(&plan.max_terms_per_gene_v1())
            || inputs.archive_capacity > 65_535
            || inputs.settings.month_capacity > i32::MAX as u32
            || plan.scoring_semantics_sha256_v1()
                != crate::resident_scoring_v2::scoring_semantics_sha256_v2(objective)
            || plan.rank_semantics_sha256_v1()
                != crate::resident_scoring_v2::rank_semantics_sha256_v2()
            || plan.novelty_semantics_sha256_v1()
                != crate::resident_scoring_v2::novelty_disabled_semantics_sha256_v2()
        {
            return Err(HipResidentSearchErrorV3::detail(
                "HIP Search plan differs from its physical parent or requested semantics",
            ));
        }
        let mut scope = Sha256::new();
        scope.update(b"neoethos.hip-search.requested-physical-scope.v1");
        scope.update(identity.physical_binding_sha256);
        scope.update(plan.plan_identity_sha256_v1());
        scope.update(inputs.stage1_row_start.to_le_bytes());
        scope.update(inputs.stage1_row_end.to_le_bytes());
        scope.update(inputs.first_timestamp_ms.to_le_bytes());
        scope.update(inputs.last_timestamp_ms.to_le_bytes());
        scope.update(crate::population::hash_population_settings_identity_v1(
            &inputs.settings,
        ));
        let input = ResidentSearchSlice2InputIdentityV3::Hip {
            parent: identity.clone(),
            requested_plan_sha256: plan.plan_identity_sha256_v1(),
            requested_scope_sha256: scope.finalize().into(),
        };
        input
            .validate_plan_v3(&plan)
            .map_err(HipResidentSearchErrorV3::detail)?;
        let execution_plan = ResidentSearchExecutionPlanV3 {
            inner: Some(ResidentSearchStartAuthorityV3 {
                request: ResidentSearchSlice2RequestV3 {
                    input,
                    archive_capacity: inputs.archive_capacity,
                    archive_policy: inputs.archive_policy,
                },
                plan,
                smc_weights: inputs.smc_weights,
                smc_gate_disabled: inputs.smc_gate_disabled,
                settings: inputs.settings,
                scenarios: inputs.scenarios.into_boxed_slice(),
                objective,
                novelty_weight: inputs.novelty_weight,
                goal_context,
                stage1_view,
                adaptive_base_request: inputs.adaptive_base_request,
                retain_compact_session: false,
            }),
        };
        // Every ordinary input/semantic check above precedes detachment.
        let core = self
            .take_search_core_v3(&identity, adaptive_rows)
            .map_err(HipResidentSearchErrorV3::detail)?;
        let stage = start_resident_population_search_slice2_v3(
            ResidentSearchPopulationOwnerV3::Hip {
                core: Some(core),
                identity: identity.clone(),
            },
            execution_plan,
        );
        Ok(HipResidentSearchBoundV3 {
            inner: Some(stage),
            parent: Some(self),
            identity,
        })
    }
}

#[cfg(test)]
mod tests {
    #[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
    fn execution_inputs_fixture_v3() -> super::ResidentSearchExecutionInputsV3 {
        use neoethos_gpu_contracts::resident_search_scoring_v2::RiskyGrowthGoal;
        super::ResidentSearchExecutionInputsV3 {
            settings: crate::NeoPopulationSettings {
                abi_version: neoethos_gpu_contracts::ABI_VERSION,
                initial_equity: 12_345.25,
                pip_value: 0.0001,
                pip_value_per_lot: 7.4,
                spread_pips: 2.5,
                commission_per_trade: 7.7,
                swap_long_pips_per_day: -2.4,
                swap_short_pips_per_day: -0.1,
                risk_per_trade_min: 0.01,
                risk_per_trade_max: 0.30,
                high_quality_confidence: 0.8,
                month_capacity: 480,
                ..crate::NeoPopulationSettings::default()
            },
            scenarios: (0..2)
                .map(|candidate| crate::ScenarioDescriptor {
                    base_candidate_id: candidate,
                    scenario_id: candidate,
                    window_len: 3,
                    ..crate::ScenarioDescriptor::default()
                })
                .collect(),
            smc_weights: [1.0; 11],
            smc_gate_disabled: false,
            growth_objective: true,
            growth_goal: Some(RiskyGrowthGoal {
                start_balance: 100.0,
                target_balance: 50_000.0,
                horizon_days: 180.0,
            }),
            stage1_row_start: 5,
            stage1_row_end: 8,
            first_timestamp_ms: 86_400_000,
            last_timestamp_ms: 345_630_000,
            novelty_weight: 0.35,
            archive_capacity: 100,
            archive_policy: None,
            adaptive_base_request: None,
        }
    }

    #[test]
    #[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
    fn compact_execution_inputs_keep_actual_equity_calendar_span_and_requested_novelty() {
        let mut inputs = execution_inputs_fixture_v3();
        for weight in [0.0, 0.2, 0.35, 1.0] {
            inputs.novelty_weight = weight;
            let (objective, context) = inputs.validate_v3(2, 10).unwrap();
            assert_eq!(
                objective,
                super::ResidentScoringObjectiveV2::RiskyGrowthGoalV6
            );
            let context = context.unwrap();
            assert_eq!(context.initial_equity.to_bits(), 12_345.25_f64.to_bits());
            assert_eq!(
                context.span_days.to_bits(),
                (259_230_000.0_f64 / 86_400_000.0).to_bits()
            );
            assert_eq!(context.goal, inputs.growth_goal.unwrap());
            assert_eq!(inputs.novelty_weight.to_bits(), weight.to_bits());
        }
    }

    #[test]
    #[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
    fn compact_execution_inputs_reject_partial_census_detached_span_costs_and_invalid_policy() {
        let mut inputs = execution_inputs_fixture_v3();
        inputs.scenarios.pop();
        assert!(inputs.validate_v3(2, 10).is_err());
        let mut inputs = execution_inputs_fixture_v3();
        inputs.scenarios[1].base_candidate_id = 0;
        assert!(inputs.validate_v3(2, 10).is_err());
        let mut inputs = execution_inputs_fixture_v3();
        inputs.scenarios[0].spread_ticks = 0;
        assert!(inputs.validate_v3(2, 10).is_err());
        let mut inputs = execution_inputs_fixture_v3();
        inputs.last_timestamp_ms = inputs.first_timestamp_ms;
        assert!(inputs.validate_v3(2, 10).is_err());
        assert!(execution_inputs_fixture_v3().validate_v3(2, 7).is_err());
        for weight in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
            let mut inputs = execution_inputs_fixture_v3();
            inputs.novelty_weight = weight;
            assert!(inputs.validate_v3(2, 10).is_err());
        }
        let mut inputs = execution_inputs_fixture_v3();
        inputs.settings.swap_long_pips_per_day = f64::NAN;
        assert!(inputs.validate_v3(2, 10).is_err());
        let mut inputs = execution_inputs_fixture_v3();
        inputs.growth_objective = false;
        assert!(inputs.validate_v3(2, 10).is_err());
        inputs.growth_goal = None;
        assert_eq!(
            inputs.validate_v3(2, 10).unwrap(),
            (super::ResidentScoringObjectiveV2::PropFirmV4, None)
        );
    }

    #[test]
    #[cfg(any(feature = "cuda", feature = "hip-native-kernels"))]
    fn compact_execution_stage1_view_matches_exact_parent_and_adaptive_request() {
        let mut inputs = execution_inputs_fixture_v3();
        let range = inputs.stage1_view_v3(10).unwrap();
        assert_eq!(range.row_count(), 3);
        assert_eq!(range.range(), Some(5..8));
        inputs.stage1_row_start = 0;
        inputs.stage1_row_end = 120;
        let full = inputs.stage1_view_v3(120).unwrap();
        let request = super::ResidentAdaptiveBaseRequestV1::checked_canonical_v1(
            &full,
            inputs.settings.pip_value,
            1,
            120,
        )
        .unwrap();
        inputs.adaptive_base_request = Some(request);
        let rebound = inputs.stage1_view_v3(120).unwrap();
        assert_eq!(rebound.kind(), full.kind());
        assert_eq!(rebound.row_count(), full.row_count());
        inputs.stage1_row_start = 1;
        assert!(inputs.stage1_view_v3(120).is_err());
        inputs.stage1_row_start = 0;
        inputs.settings.pip_value *= 2.0;
        assert!(inputs.stage1_view_v3(120).is_err());
    }

    #[test]
    fn compact_execution_owner_identity_is_rechecked_before_detach_without_poisoning() {
        let source = include_str!("resident_search_slice2_v3.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let start = source
            .split("let Some(session) = state.session.as_mut() else {")
            .nth(1)
            .unwrap();
        let guard = start
            .find("session.validate_plan_owner_v3(&start.request.input)")
            .unwrap();
        let bind = start.find("session.bind_stage1_view_v3(").unwrap();
        let detach = start
            .find("session.take_population_session_for_slice2_v3()")
            .unwrap();
        assert!(guard < bind && bind < detach);
        let rejection = &start[guard..bind];
        assert!(rejection.contains("execution_plan.inner = Some(start);"));
        assert!(rejection.contains("state.execution_plan = Some(execution_plan);"));
        assert!(!rejection.contains("state.poisoned = true"));
        let admission =
            include_str!("resident_search_slice2_admission_v2.rs").replace(char::is_whitespace, "");
        for actual in [
            "session.admission_identity_sha256()",
            "session.canonical_content_merkle()",
            "session.data_transient_retirement_process_token()",
            "session.device_identity().clone()",
            "session.data_population_limits()",
            "Self::from_session_v3(session)?",
        ] {
            assert!(
                admission.contains(actual),
                "missing actual owner binding {actual}"
            );
        }
    }

    #[test]
    fn cuda_authority_state_is_private_move_only_and_native_wired() {
        let source = include_str!("resident_search_slice2_v3.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let trim_source = include_str!("resident_trim_prefilter_v1.rs");

        assert_eq!(source.matches(concat!("pub ", "struct ")).count(), 12);
        assert_eq!(source.matches(concat!("pub ", "enum ")).count(), 2);
        let hip_drop = source
            .split("impl<Stage> Drop for HipResidentSearchBoundV3")
            .nth(1)
            .unwrap()
            .split("#[cfg(feature = \"hip-native-kernels\")]")
            .next()
            .unwrap();
        let quarantines_taken_stage = |body: &str| {
            let parent = body.find("if let Some(parent) = self.parent.take()");
            let inner = body.find("if let Some(stage) = self.inner.take()");
            matches!((parent, inner), (Some(parent), Some(inner)) if parent < inner)
                && body.contains("parent.quarantine_search_v3();")
        };
        assert!(quarantines_taken_stage(hip_drop));
        assert!(!quarantines_taken_stage(
            &hip_drop.replace("self.parent.take()", "self.parent.as_deref_mut()")
        ));
        let restored = source
            .split("if let Err(error) = parent.restore_search_core_v3(core, &self.identity)")
            .nth(1)
            .unwrap();
        assert!(
            restored.find("self.parent.take();").unwrap()
                < restored.find("let candidates = receipt").unwrap()
        );
        assert_eq!(
            source
                .matches(concat!("inner: core::convert::", "Infallible,"))
                .count(),
            7
        );
        assert!(source.contains("struct ResidentSearchAuthorityStateV3 {"));
        assert!(source.contains("session: Option<ResidentSearchPopulationOwnerV3>,"));
        assert!(source.contains("native: Option<ResidentSearchSlice2NativeOwnerV3>,"));
        assert!(source.contains("settings: Option<NeoPopulationSettings>,"));
        assert!(source.contains("fn start_resident_population_search_slice2_v3("));
        for native_transition in [
            "native.enqueue_score_and_rank_v3(&settings)",
            "native.enqueue_stage_archive_from_rank_v3()",
            "native.enqueue_evolve_and_publish_v3()",
            "native.enqueue_terminal_seal_v3()",
            "native.try_complete_terminal_v3()",
            "native.copy_terminal_archive_v3(&state.scenario_ids)",
            "native.copy_terminal_population_v3(&state.scenario_ids)",
            "native.release_terminal_v3()",
        ] {
            assert!(
                source.contains(native_transition),
                "missing {native_transition}"
            );
        }
        assert!(source.contains("ResidentSearchSlice2NativeTryCompleteV3::NotReady(native)"));
        assert!(!source.contains(concat!(
            "ResidentSearchRejectedTransitionV3::",
            "ScoreAndRank"
        )));
        assert!(!source.contains(concat!(
            "Ok(ResidentSearchTryCompleteV3::",
            "NotReady(self))"
        )));
        assert!(source.contains("pub fn begin_resident_search_slice2_v3("));
        assert!(!trim_source.contains("pub fn begin_resident_search_slice2_v3("));
    }

    #[test]
    fn slice2_start_retains_requested_inputs_without_fabricating_runtime_calibration() {
        let source = include_str!("resident_search_slice2_v3.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap()
            .replace("\r\n", "\n");
        let trim_source = include_str!("resident_trim_prefilter_v1.rs").replace("\r\n", "\n");

        let execution_plan = concat!(
            "pub ",
            "struct ResidentSearchExecutionPlanV3",
            " {\n    #[cfg(any(feature = \"cuda\", feature = \"hip-native-kernels\"))]\n    inner: Option<ResidentSearchStartAuthorityV3>,"
        );
        let private_start = "fn start_resident_population_search_slice2_v3(\n    session: ResidentSearchPopulationOwnerV3,\n    execution_plan: ResidentSearchExecutionPlanV3,\n) -> ResidentSearchGenerationChainV3";
        let public_start = "pub fn begin_resident_search_slice2_v3(\n        self,\n        execution_plan: ResidentSearchExecutionPlanV3,\n    ) -> ResidentSearchGenerationChainV3";

        assert!(source.contains(execution_plan));
        assert!(source.contains(private_start));
        assert!(source.contains(public_start));
        for retained in [
            "plan: SealedResidentGenerationPlanV1,",
            "smc_weights: [f64; 11],",
            "smc_gate_disabled: bool,",
            "settings: NeoPopulationSettings,",
            "scenarios: Box<[ScenarioDescriptor]>,",
            "request: ResidentSearchSlice2RequestV3,",
            "objective: ResidentScoringObjectiveV2,",
            "goal_context: Option<ResidentScoringGoalContextV2>,",
        ] {
            assert!(
                source.contains(retained),
                "missing retained start authority {retained}"
            );
        }
        assert!(source.contains("start_resident_population_search_slice2_v3(\n            ResidentSearchPopulationOwnerV3::Compact(self),\n            execution_plan,\n        )"));
        assert!(!trim_source.contains("start_resident_search_slice2_v3("));

        assert!(!source.contains(
            "session: ResidentSearchPopulationOwnerV3,\n) -> ResidentSearchGenerationChainV3"
        ));
        assert!(!source.contains("pub fn begin_resident_search_slice2_v3(\n        self,\n    )"));
        assert!(!source.contains("seal_resident_archive_knn_calibration_receipt_v2("));
        for forbidden in [
            concat!("Clone for ResidentSearch", "ExecutionPlanV3"),
            concat!("Copy for ResidentSearch", "ExecutionPlanV3"),
            concat!("Default for ResidentSearch", "ExecutionPlanV3"),
            concat!("mint_", "calibration"),
            concat!("fixture_", "calibration"),
            concat!("calibration_", "binding(&self)"),
        ] {
            assert!(
                !source.contains(forbidden),
                "forbidden fabrication seam: {forbidden}"
            );
        }
    }

    #[test]
    fn slice2_compact_owner_uses_the_shared_chain_without_fabricating_trim() {
        let source = include_str!("resident_search_slice2_v3.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let compact_entry = source
            .split("impl ResidentPopulationSessionV3 {")
            .nth(1)
            .unwrap()
            .split("fn start_resident_population_search_slice2_v3(")
            .next()
            .unwrap();
        assert!(compact_entry.contains("execution_plan: ResidentSearchExecutionPlanV3"));
        assert!(compact_entry.contains("ResidentSearchPopulationOwnerV3::Compact(self)"));
        assert!(compact_entry.contains("start_resident_population_search_slice2_v3("));
        assert!(!compact_entry.contains("ResidentTrimmedPopulationSessionV1"));
        assert!(!source.contains("Trimmed("));
        let admission = include_str!("resident_search_slice2_admission_v2.rs");
        assert!(!admission.contains("Trimmed {"));
        let trim = include_str!("resident_trim_prefilter_v1.rs");
        assert!(trim.contains("pub struct ResidentTrimmedPopulationSessionV1 {"));
        assert!(trim.contains("pub fn consume_into_population_session_v3("));
        assert!(trim.contains("pub fn seal_resident_trim_prefilter_device_views_v1("));
        assert!(trim.contains("impl Drop for ResidentTrimmedPopulationSessionV1 {"));
        assert!(!trim.contains("begin_resident_search_slice2_v3("));
        assert!(!trim.contains("take_population_session_for_slice2_v3("));
        let compact_completion = source
            .split("Self::Compact(mut owner) => {")
            .nth(1)
            .unwrap()
            .split("/// Immutable requested execution inputs.")
            .next()
            .unwrap();
        let restore = compact_completion
            .find("owner.restore_population_session_from_slice2_v3(population)")
            .unwrap();
        let complete = compact_completion
            .find(".record_consumer_completion()")
            .unwrap();
        assert!(restore < complete);
        assert!(compact_completion.contains("std::mem::forget(population)"));
        assert!(!compact_completion.contains("trim_prefilter_release"));
        // The compact entry does not duplicate the evaluator or add a readback.
        assert_eq!(
            source
                .matches("native.enqueue_score_and_rank_v3(&settings)")
                .count(),
            1
        );
        assert_eq!(
            source
                .matches("native.enqueue_evolve_and_publish_v3()")
                .count(),
            1
        );
    }

    #[test]
    fn slice2_terminal_retains_both_evaluated_sources_before_releasing_native_graph() {
        let source = include_str!("resident_search_slice2_v3.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let complete = source
            .split("Complete(mut native)) => {")
            .nth(1)
            .expect("completion arm");
        let copy = complete
            .find("native.copy_terminal_archive_v3(&state.scenario_ids)")
            .unwrap();
        let retain = complete
            .find("state.terminal_candidates = Some(output);")
            .unwrap();
        let population = complete
            .find("native.copy_terminal_population_v3(&state.scenario_ids)")
            .unwrap();
        let seal = complete
            .find("ResidentSearchTerminalCandidatesV3::seal(archive, population)")
            .unwrap();
        let release = retain
            + complete[retain..]
                .find("native.release_terminal_v3()")
                .unwrap();
        let finish = release
            + complete[release..]
                .find("complete_population_lifetime_v3(&mut state, population)")
                .unwrap();
        assert!(
            copy < population
                && population < seal
                && seal < retain
                && retain < release
                && release < finish
        );
        assert!(source.contains("start.objective,"));
        assert!(source.contains("start.goal_context,"));
        let terminal_keeps_requested_objective =
            |body: &str| !body.contains(concat!("ResidentScoringObjectiveV2::", "PropFirmV4"));
        assert!(terminal_keeps_requested_objective(complete));
        assert!(!terminal_keeps_requested_objective(&format!(
            "{complete}\nResidentScoringObjectiveV2::PropFirmV4"
        )));
    }

    #[test]
    fn slice2_host_output_rejection_releases_proven_graph_but_native_copy_failure_does_not() {
        let source = include_str!("resident_search_slice2_v3.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let host_rejection = source
            .split("Err(error @ ResidentSearchSlice2NativeErrorV3::ArchiveOutput(_)) => {")
            .nth(1)
            .unwrap();
        let (host_cleanup, native_rejection) =
            host_rejection.split_once("Err(error) => {").unwrap();
        assert!(host_cleanup.contains("native.release_terminal_v3()"));
        assert!(host_cleanup.contains("complete_population_lifetime_v3(&mut state, population)"));
        assert!(host_cleanup.contains("state.native = Some(native);"));
        let native_rejection = native_rejection
            .split("state.terminal_candidates = Some(output);")
            .next()
            .unwrap();
        assert!(native_rejection.contains("state.native = Some(native);"));
        assert!(!native_rejection.contains("native.release_terminal_v3()"));

        let native = include_str!("resident_search_v2.rs");
        let release = native
            .split("pub(crate) fn release_terminal_v3(")
            .nth(1)
            .unwrap()
            .split("impl Drop for ResidentSearchSlice2NativeOwnerV3")
            .next()
            .unwrap();
        assert!(
            release.contains("self.state != ResidentSearchSlice2NativeStateV3::TerminalComplete")
        );
        assert!(!release.contains("!self.archive_exported"));
        let copy = native
            .split("pub(crate) fn copy_terminal_archive_v3(")
            .nth(1)
            .unwrap()
            .split("pub(crate) fn release_terminal_v3(")
            .next()
            .unwrap();
        assert!(copy.contains("self.state = ResidentSearchSlice2NativeStateV3::Poisoned;"));
    }
}
