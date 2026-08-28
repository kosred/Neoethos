use super::super::Ohlcv;
use crate::core::all_indicators::ALL_INDICATORS;
use crate::core::feature_budget::{VocabularyBudget, admit_indicators};
use crate::core::features::{FeatureCellValidity, FeatureColumnF64};
use crate::core::indicator_ledger::{
    DropReason, IndicatorLedger, expected_non_producing, has_finite_variation, output_ids_for,
    planned_output_count, series_fingerprint,
};
use crate::core::timestamps::validate_canonical_millisecond_timestamps;
use rayon::prelude::*;
use std::collections::HashSet;
use std::panic::{AssertUnwindSafe, catch_unwind};
use vector_ta::indicators::dispatch::{
    IndicatorComputeOutput, IndicatorComputeRequest, IndicatorDataRef, IndicatorSeries, ParamKV,
    ParamValue, compute_cpu,
};
use vector_ta::indicators::registry::indicator_data_requirements;
use vector_ta::utilities::data_loader::Candles;
use vector_ta::utilities::enums::Kernel;
use vector_ta::utilities::helpers::detect_best_kernel;

#[cfg(all(test, feature = "gpu-cuda-device-fixtures"))]
#[path = "gpu_resident_classic_ta_v3_device_tests.rs"]
mod gpu_resident_classic_ta_v3_device_tests;

/// Fewest distinct indicator ids the base pass must produce a column for
/// before the feature build is considered sane, on a frame long enough to warm
/// them up.
///
/// MEASURED, not aspirational: over real EURUSD M5 bars (identical at 20k and
/// 200k rows) 324 of the 342 ids produce at least one column once the output
/// ids are enumerated and the accept test keys off the value count. 280 leaves
/// room for vector-ta variation while making the regression this constant
/// exists to catch — the collapse to ONE producing id (`ttm_trend`) that this
/// codebase ran on for sixteen months — a hard, named failure.
pub const MIN_PRODUCING_INDICATOR_IDS: usize = 280;

/// Fewest base-pass columns, same measurement: the repaired pass yields ~674.
pub const MIN_BASE_VOCABULARY_COLUMNS: usize = 400;

/// Frames shorter than this do not enforce [`MIN_PRODUCING_INDICATOR_IDS`].
///
/// On a 100-bar fixture most indicators legitimately fail their warmup, so a
/// low count is the data's property and not a regression. The census is still
/// logged at every length — only the hard error is length-gated, and the log
/// line says which mode it ran in.
pub const VOCABULARY_FLOOR_MIN_ROWS: usize = 5_000;

/// Largest indicator-period this module ever asks vector-ta to compute
/// in its multi-period sweep. Used by [`max_indicator_warmup`] so the
/// genetic search can pre-flight gene admission against the data slice
/// length and skip indicators that would panic the kernel
/// (`warm prefix exceeds row width`, vector-ta v0.2.9 #212).
pub const MAX_MULTI_PERIOD_LOOKBACK: usize = 200;

/// Maximum warmup periods (in bars) that the indicator stack can
/// produce on a frame with `n_rows` bars. Returns the largest period
/// from the multi-period sweep that still fits, or 0 if the frame is
/// too short to compute any of the parameterized indicators safely.
///
/// Used by the validation harness and pre-flight guards to refuse
/// evaluation on slices smaller than the indicator's warmup. The
/// thresholds match the `alt_periods` array in `compute_classic_ta_columns`.
pub fn max_indicator_warmup(n_rows: usize) -> usize {
    const ALT_PERIODS: &[usize] = &[7, 21, 50, 100, 200];
    ALT_PERIODS
        .iter()
        .rev()
        .find(|&&p| p < n_rows)
        .copied()
        .unwrap_or(0)
}

/// Computes ALL 340+ Technical Indicators automatically using VectorTA's Dispatch Engine.
/// Multi-output indicators are automatically decomposed into separate named columns.
///
/// Each indicator call is wrapped in `std::panic::catch_unwind` because
/// vector-ta v0.2.9 panics on a small subset of period/data combinations
/// (e.g. EURUSD M5 hits `warm prefix exceeds row width` at
/// `utilities/helpers.rs:159`, #212). The wrapping converts a panic into
/// a silently-skipped column rather than tearing down the worker thread,
/// which on the rayon-driven discovery hot path would otherwise abort
/// the whole TF run with no fallback path. The pre-flight
/// [`max_indicator_warmup`] helper still gates the multi-period sweep
/// so the common case never reaches the kernel boundary.
pub fn compute_classic_ta_columns(ohlcv: &Ohlcv) -> anyhow::Result<Vec<(String, Vec<f64>)>> {
    compute_classic_ta_columns_with_policy(ohlcv, resolved_indicator_compute_policy())
}

/// Process-wide policy set once by the binary that owns the operator's
/// Settings. The first read freezes the default `Auto` policy so a later
/// settings install cannot relabel feature bits that were already computed.
static POLICY_OVERRIDE: std::sync::OnceLock<IndicatorComputePolicy> = std::sync::OnceLock::new();

/// Bind the indicator lane policy for this process. Idempotent-or-refuse: the
/// second call fails with the value already in force rather than silently
/// keeping one of the two, because "which lane ran" is exactly the question
/// this whole module exists to answer unambiguously.
///
/// This is the seam the operator's Settings plugs into. It is deliberately a
/// single point, so there is one place to read to know what a run did.
pub fn set_indicator_compute_policy(
    policy: IndicatorComputePolicy,
) -> Result<(), IndicatorComputePolicy> {
    match POLICY_OVERRIDE.set(policy) {
        Ok(()) => Ok(()),
        Err(_) => Err(*POLICY_OVERRIDE
            .get()
            .expect("just failed to set, so it is set")),
    }
}

/// The exclusive policy a caller that does not name one gets.
///
/// The operator's resolved Settings are installed once through
/// [`set_indicator_compute_policy`]. Without an explicit selection, the first
/// read installs `Auto` as the immutable process policy. That ordering is part
/// of receipt identity: a later `GpuOnly` install must fail instead of labeling
/// already-computed CPU bits as CUDA output. `Auto` chooses one complete
/// supported lane; it never authorizes a partial CPU/CUDA plan. The retired
/// `NEOETHOS_REQUIRE_GPU` environment variable is still reported as an error by
/// [`crate::report_retired_env_vars`].
pub fn resolved_indicator_compute_policy() -> IndicatorComputePolicy {
    crate::report_retired_env_vars();
    *POLICY_OVERRIDE.get_or_init(|| IndicatorComputePolicy::Auto)
}

/// Which exclusive device may compute the complete production feature plan.
///
/// A value never authorizes splitting one plan between host and device. Until
/// the full resident CUDA graph is connected, [`Self::GpuOnly`] fails before
/// any feature computation and [`Self::Auto`] resolves to [`Self::CpuOnly`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndicatorComputePolicy {
    /// Select one complete supported lane. There is currently no complete GPU
    /// plan, so this selects CPU-only; it never creates a mixed run.
    Auto,
    /// CPU only. This is the parity reference — f64 end to end.
    CpuOnly,
    /// GPU only. The entire requested graph must be resident and supported,
    /// otherwise the request is rejected before CPU or CUDA work begins.
    GpuOnly,
}

/// Exact vector-ta arithmetic lane selected for the canonical feature build.
///
/// This is an identity authority, not a performance hint. SIMD implementations
/// are permitted to have different f64 bit patterns, so persisted search
/// receipts must distinguish every lane that `Kernel::Auto` can actually
/// select. The AVX-512 name pins the complete target-feature union required by
/// vector-ta's current implementation rather than the misleading shorthand
/// `avx512f` alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedCanonicalFeatureMathLaneV1 {
    CpuScalar,
    CpuAvx2Fma,
    CpuAvx512F64Avx2FmaDqVlBw,
    GpuCudaF64Strict,
}

/// Runtime authority captured alongside a canonical search feature frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedCanonicalFeatureExecutionAuthorityV1 {
    pub policy: IndicatorComputePolicy,
    pub selected_lane: ResolvedCanonicalFeatureMathLaneV1,
    pub vector_ta_math_authority: &'static str,
}

pub const VECTOR_TA_CPU_F64_MATH_AUTHORITY_V1: &str = "neoethos.vector-ta.cpu-f64-exact-bits.v1";
pub const VECTOR_TA_CUDA_F64_MATH_AUTHORITY_V1: &str = "neoethos.vector-ta.cuda-f64-exact-bits.v1";

/// Families proven non-bit-exact by the required-card integrated M15/M30
/// census. Standard/HPC/Adaptive remain usable only by excluding each family
/// atomically until its CPU/CUDA value and validity authority is repaired.
/// Full keeps the complete graph and therefore continues to fail closed.
#[cfg(feature = "gpu-cuda")]
const GPU_ONLY_PARITY_DEFERRED_INDICATORS_V3: &[&str] = &[
    "geometric_bias_oscillator",
    "gopalakrishnan_range_index",
    "historical_volatility",
    "ift_rsi",
    "l1_ehlers_phasor",
    "maaq",
    "natr",
    "nma",
    "pfe",
    "premier_rsi_oscillator",
    "pwma",
    "sgf",
    "sinwma",
    "squeeze_index",
    "sqwma",
    "supersmoother_3_pole",
    "trendflex",
    "ttm_trend",
    "ultosc",
    "uma",
    "vidya",
    "volatility_adjusted_ma",
    "vpwma",
    "wave_smoother",
    "wma",
];

/// Resolve the same process policy and CPU dispatcher used by production.
///
/// `GpuOnly` records the strict CUDA authority. That policy still has to pass
/// the complete pre-launch CUDA graph check before any feature computation;
/// this function never turns an unavailable GPU request into a CPU receipt.
pub fn resolved_canonical_feature_execution_authority_v1()
-> ResolvedCanonicalFeatureExecutionAuthorityV1 {
    let policy = resolved_indicator_compute_policy();
    let (selected_lane, vector_ta_math_authority) = match policy {
        IndicatorComputePolicy::Auto | IndicatorComputePolicy::CpuOnly => {
            let selected_lane = match detect_best_kernel() {
                Kernel::Scalar => ResolvedCanonicalFeatureMathLaneV1::CpuScalar,
                Kernel::Avx2 => ResolvedCanonicalFeatureMathLaneV1::CpuAvx2Fma,
                Kernel::Avx512 => ResolvedCanonicalFeatureMathLaneV1::CpuAvx512F64Avx2FmaDqVlBw,
                unexpected => panic!(
                    "vector-ta single-series Auto resolved unsupported canonical lane {unexpected:?}"
                ),
            };
            (selected_lane, VECTOR_TA_CPU_F64_MATH_AUTHORITY_V1)
        }
        IndicatorComputePolicy::GpuOnly => (
            ResolvedCanonicalFeatureMathLaneV1::GpuCudaF64Strict,
            VECTOR_TA_CUDA_F64_MATH_AUTHORITY_V1,
        ),
    };
    ResolvedCanonicalFeatureExecutionAuthorityV1 {
        policy,
        selected_lane,
        vector_ta_math_authority,
    }
}

#[derive(Debug, Clone, Copy)]
struct ClassicInputAvailability {
    timestamps: bool,
    volume: bool,
}

impl ClassicInputAvailability {
    fn from_ohlcv(ohlcv: &Ohlcv) -> Self {
        Self {
            timestamps: ohlcv.timestamp.is_some(),
            volume: ohlcv.volume.is_some(),
        }
    }

    #[cfg(all(feature = "gpu-cuda", test))]
    fn all_present() -> Self {
        Self {
            timestamps: true,
            volume: true,
        }
    }

    fn missing_for(self, indicator_id: &str) -> Option<&'static str> {
        let requirements = indicator_data_requirements(indicator_id);
        match (
            requirements.requires_timestamps && !self.timestamps,
            requirements.requires_volume && !self.volume,
        ) {
            (true, true) => Some("timestamps and volume are absent"),
            (true, false) => Some("timestamps are absent"),
            (false, true) => Some("volume is absent"),
            (false, false) => None,
        }
    }
}

/// The exact vocabulary admission decision used by one classic-TA execution.
///
/// This is captured while the production run still owns the budget decision.
/// Callers must not reconstruct it after the columns have been allocated:
/// `VocabularyBudget` is derived from currently available memory, so a second
/// probe can describe a different machine state and report a false admission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClassicTaExecutionReport {
    pub budget_rows: usize,
    pub available_bytes_at_admission: u64,
    pub max_columns: usize,
    pub admitted_indicator_ids: Vec<&'static str>,
    pub budget_deferred_indicator_ids: Vec<&'static str>,
    /// Exact canonical-order families excluded from an explicitly versioned
    /// GPU-only routeable subset because at least one requested output lacks a
    /// complete resident f64 contract. Full-profile admission never uses this
    /// escape hatch and remains atomic/fail-closed.
    pub capability_deferred_indicator_ids: Vec<&'static str>,
    pub capability_deferred_output_count: usize,
    /// Columns requested by the complete base vocabulary, before admission.
    pub planned_base_columns: usize,
    /// Columns represented by the admitted base IDs.
    pub admitted_base_columns: usize,
    pub historical_sweep_reserved_columns: usize,
    pub historical_sweep_produced_columns: usize,
    pub extended_mode: &'static str,
    pub extended_admitted_indicator_ids: Vec<&'static str>,
    pub extended_budget_deferred_indicator_ids: Vec<&'static str>,
    pub extended_budget_columns: usize,
    pub extended_planned_columns: usize,
    pub produced_columns: usize,
}

/// Classic-TA values and the ledger/admission facts from the same execution.
///
/// The large f64 columns are owned exactly once. Existing production callers
/// use the value-only wrappers below; audit/parity callers consume this type so
/// their evidence cannot drift from the execution it describes.
#[derive(Debug)]
pub struct ClassicTaComputation {
    pub columns: Vec<(String, Vec<f64>)>,
    pub report: ClassicTaExecutionReport,
    pub ledger: IndicatorLedger,
}

/// One allocation-free Classic/vector-ta admission decision, shared by the
/// CPU reference and the strict CUDA planner.
///
/// The budget probe happens exactly once.  In particular, `GpuOnly` must not
/// rebuild this object after opening a CUDA context: available RAM can change
/// between probes, which would let the receipt describe a different admitted
/// vocabulary from the one that actually ran.
#[derive(Clone, Debug)]
pub(crate) struct ClassicTaAdmissionPlan {
    pub(crate) budget_rows: usize,
    pub(crate) budget: VocabularyBudget,
    pub(crate) base_budget: VocabularyBudget,
    pub(crate) sweep_reserved: usize,
    pub(crate) admitted_indicator_ids: Vec<&'static str>,
    pub(crate) budget_deferred_indicator_ids: Vec<&'static str>,
    pub(crate) capability_deferred_indicator_ids: Vec<&'static str>,
    pub(crate) capability_deferred_output_count: usize,
    pub(crate) gpu_route_mode: &'static str,
    pub(crate) historical_indicator_ids: Vec<&'static str>,
    pub(crate) planned_base_columns: usize,
    pub(crate) admitted_base_columns: usize,
    pub(crate) working_set: Option<std::sync::Arc<SweepBatch>>,
    pub(crate) extended_groups: Vec<(&'static str, Vec<usize>)>,
    pub(crate) extended_budget_deferred_indicator_ids: Vec<&'static str>,
    pub(crate) extended_mode: &'static str,
    pub(crate) extended_budget_columns: usize,
    pub(crate) extended_planned_columns: usize,
}

/// One run-wide Classic/vector-ta admission decision.
///
/// Construction probes available RAM exactly once and, for `GpuOnly`, resolves
/// the complete admitted CUDA graph before any feature producer or CUDA
/// context exists.  Every direct timeframe in a multi-timeframe cube borrows
/// this same value, so later allocations cannot silently narrow the admitted
/// vocabulary.
#[derive(Clone, Debug)]
pub struct ClassicTaRunPlan {
    policy: IndicatorComputePolicy,
    admission: ClassicTaAdmissionPlan,
    #[cfg(feature = "gpu-cuda")]
    resident_cuda_launches: Option<Vec<crate::core::classic_cuda_plan::ResolvedClassicCudaLaunch>>,
}

/// Immutable projection of the one already-probed and already-resolved
/// Classic TA run plan for the strict resident producer. This is deliberately
/// crate-private: it carries no device authority and cannot be rebuilt by
/// App/Search from an ordinal, a free-memory number or a caller recipe.
#[cfg(feature = "gpu-cuda")]
#[derive(Clone, Debug)]
pub(crate) struct ClassicTaResidentPlanProjectionV3 {
    pub(crate) budget_rows: usize,
    pub(crate) available_bytes_at_admission: u64,
    pub(crate) admitted_indicator_ids: Vec<&'static str>,
    pub(crate) capability_deferred_indicator_ids: Vec<&'static str>,
    pub(crate) capability_deferred_output_count: usize,
    pub(crate) gpu_route_mode: &'static str,
    pub(crate) extended_groups: Vec<(&'static str, Vec<usize>)>,
    pub(crate) working_set: Option<std::sync::Arc<SweepBatch>>,
    pub(crate) launches: Vec<crate::core::classic_cuda_plan::ResolvedClassicCudaLaunch>,
}

impl ClassicTaRunPlan {
    pub const fn policy(&self) -> IndicatorComputePolicy {
        self.policy
    }

    pub fn admission_report(&self) -> ClassicTaExecutionReport {
        self.admission.execution_report(0, 0)
    }

    #[cfg(feature = "gpu-cuda")]
    pub(crate) fn resident_admission_projection_v3(
        &self,
    ) -> anyhow::Result<ClassicTaResidentPlanProjectionV3> {
        if self.policy != IndicatorComputePolicy::GpuOnly {
            anyhow::bail!("resident Classic TA projection requires the frozen GpuOnly run plan")
        }
        let launches = self.resident_cuda_launches.clone().ok_or_else(|| {
            anyhow::anyhow!("GpuOnly run plan omitted its pre-context resolved Classic CUDA graph")
        })?;
        Ok(ClassicTaResidentPlanProjectionV3 {
            budget_rows: self.admission.budget_rows,
            available_bytes_at_admission: self.admission.budget.available_bytes,
            admitted_indicator_ids: self.admission.admitted_indicator_ids.clone(),
            capability_deferred_indicator_ids: self
                .admission
                .capability_deferred_indicator_ids
                .clone(),
            capability_deferred_output_count: self.admission.capability_deferred_output_count,
            gpu_route_mode: self.admission.gpu_route_mode,
            extended_groups: self.admission.extended_groups.clone(),
            working_set: self.admission.working_set.clone(),
            launches,
        })
    }
}

impl ClassicTaAdmissionPlan {
    pub(crate) fn execution_report(
        &self,
        historical_sweep_produced_columns: usize,
        produced_columns: usize,
    ) -> ClassicTaExecutionReport {
        ClassicTaExecutionReport {
            budget_rows: self.budget_rows,
            available_bytes_at_admission: self.budget.available_bytes,
            max_columns: self.budget.max_columns,
            admitted_indicator_ids: self.admitted_indicator_ids.clone(),
            budget_deferred_indicator_ids: self.budget_deferred_indicator_ids.clone(),
            capability_deferred_indicator_ids: self.capability_deferred_indicator_ids.clone(),
            capability_deferred_output_count: self.capability_deferred_output_count,
            planned_base_columns: self.planned_base_columns,
            admitted_base_columns: self.admitted_base_columns,
            historical_sweep_reserved_columns: self.sweep_reserved,
            historical_sweep_produced_columns,
            extended_mode: self.extended_mode,
            extended_admitted_indicator_ids: self
                .extended_groups
                .iter()
                .map(|(id, _)| *id)
                .collect(),
            extended_budget_deferred_indicator_ids: self
                .extended_budget_deferred_indicator_ids
                .clone(),
            extended_budget_columns: self.extended_budget_columns,
            extended_planned_columns: self.extended_planned_columns,
            produced_columns,
        }
    }

    /// Start the exact ledger with the two admission-only drop classes.  Both
    /// execution lanes add compute outcomes to this same shape.
    pub(crate) fn admission_ledger(&self) -> IndicatorLedger {
        let mut ledger = IndicatorLedger::new();
        for id in &self.budget_deferred_indicator_ids {
            ledger.dropped(
                id,
                id,
                DropReason::OverBudget,
                format!(
                    "base-vocabulary budget full at {} columns ({} of {} total reserved for the \
                     period sweep; sized at {} bars, {:.2} GB free)",
                    self.base_budget.max_columns,
                    self.sweep_reserved,
                    self.budget.max_columns,
                    self.budget_rows,
                    self.budget.available_bytes as f64 / 1e9
                ),
            );
        }
        for id in &self.extended_budget_deferred_indicator_ids {
            ledger.dropped(
                id,
                id,
                DropReason::OverBudget,
                format!(
                    "extended period sweep: only {} columns of the {} the machine affords were \
                     still unspent after the base vocabulary and the historical sweep",
                    self.extended_budget_columns, self.budget.max_columns
                ),
            );
        }
        for id in &self.capability_deferred_indicator_ids {
            ledger.dropped(
                id,
                id,
                DropReason::UnsupportedCapability,
                format!(
                    "{id} is excluded by {} because at least one canonical output lacks a complete resident f64 route; the Full profile remains fail-closed",
                    self.gpu_route_mode
                ),
            );
        }
        ledger
    }
}

/// The multi-period sweep, with an explicit lane policy.
///
/// # The device lane is f64 end to end
///
/// This doc used to say the opposite, and said it first, on the public API:
/// that vector-ta's device layer was f32-only, that routing a column through
/// the card therefore CHANGED ITS VALUE, and that the divergence was a
/// deliberate measured trade-off announced at WARN. That is no longer true and
/// the inverted version was actively dangerous, because a reader who believed
/// it would conclude the CPU fallback was the SAFE outcome.
///
/// What is true now: the lane uploads f64, runs f64 kernels compiled with
/// `-prec-div=true -prec-sqrt=true -fmad=false -ftz=false` and never
/// `--use_fast_math`, and returns f64 with no narrowing anywhere. The claim is
/// not made by this comment — it is measured by
/// `gpu_cpu_indicator_sweep_parity`, whose tolerance is `(1e-12, 1e-12)`,
/// tight enough that an f32 lane sneaking back in (relative 1.19e-7) fails
/// instantly.
///
/// The column SET, NAMES and ORDER are identical in every policy.
///
/// # `Auto` is one complete CPU lane, never a failed-GPU fallback
///
/// `Auto` currently resolves directly to the complete CPU reference. A caller
/// that requires the card must install [`IndicatorComputePolicy::GpuOnly`]; an
/// incomplete route or failed launch is then an error for the whole Classic
/// plan, never a per-indicator warning followed by CPU numbers. Callers that do
/// not name a policy get [`resolved_indicator_compute_policy`].
pub fn compute_classic_ta_columns_with_policy(
    ohlcv: &Ohlcv,
    policy: IndicatorComputePolicy,
) -> anyhow::Result<Vec<(String, Vec<f64>)>> {
    Ok(compute_classic_ta_columns_with_policy_report(ohlcv, policy)?.columns)
}

/// Execute with an explicit lane policy and return the exact admission ledger
/// captured by that same run.
pub fn compute_classic_ta_columns_with_policy_report(
    ohlcv: &Ohlcv,
    policy: IndicatorComputePolicy,
) -> anyhow::Result<ClassicTaComputation> {
    let rows = ohlcv.len();
    compute_classic_ta_columns_sized_report(ohlcv, policy, rows)
}

/// Compute Classic/vector-ta columns with an explicit f64 value/validity
/// contract.
///
/// The legacy tuple route temporarily remains as an in-worktree parity bridge
/// while Tasks 5B-9 migrate every consumer atomically.  This boundary is the
/// only route allowed to enter the shared f64 feature plan: missing optional
/// series never become numeric zeroes, leading NaNs are warmup, post-warmup
/// non-finite cells stay invalid, and a preflight placeholder retains the exact
/// reason captured by the same execution's [`IndicatorLedger`].
pub fn compute_classic_ta_feature_columns_f64(
    ohlcv: &Ohlcv,
    policy: IndicatorComputePolicy,
    budget_rows: usize,
) -> anyhow::Result<Vec<FeatureColumnF64>> {
    let run_plan = prepare_classic_ta_run_plan(budget_rows.max(ohlcv.len()), policy)?;
    compute_classic_ta_feature_columns_f64_with_run_plan(ohlcv, &run_plan)
}

/// Compute one frame through an already captured run-wide admission plan.
pub fn compute_classic_ta_feature_columns_f64_with_run_plan(
    ohlcv: &Ohlcv,
    run_plan: &ClassicTaRunPlan,
) -> anyhow::Result<Vec<FeatureColumnF64>> {
    validate_classic_ta_input(ohlcv)?;
    let ClassicTaComputation {
        columns,
        report: _,
        ledger,
    } = compute_classic_ta_columns_sized_report_with_run_plan(ohlcv, run_plan)?;

    columns
        .into_iter()
        .map(|(name, values)| {
            let indicator_id = classic_indicator_id_for_column(&name).ok_or_else(|| {
                anyhow::anyhow!(
                    "classic/vector-ta column `{name}` has no canonical indicator identity"
                )
            })?;
            let requirements = indicator_data_requirements(indicator_id);
            let missing_required_input = (requirements.requires_timestamps
                && ohlcv.timestamp.is_none())
                || (requirements.requires_volume && ohlcv.volume.is_none());

            let validity = if missing_required_input {
                vec![FeatureCellValidity::MissingInput; values.len()]
            } else {
                classify_classic_ta_validity(&name, &values, &ledger)
            };
            FeatureColumnF64::new(name, values, validity)
        })
        .collect()
}

fn validate_classic_ta_input(ohlcv: &Ohlcv) -> anyhow::Result<()> {
    let n = ohlcv.close.len();
    anyhow::ensure!(
        ohlcv.open.len() == n && ohlcv.high.len() == n && ohlcv.low.len() == n,
        "classic/vector-ta OHLC length mismatch: open={} high={} low={} close={n}",
        ohlcv.open.len(),
        ohlcv.high.len(),
        ohlcv.low.len()
    );
    if n == 0 {
        return Ok(());
    }

    for row in 0..n {
        let (open, high, low, close) = (
            ohlcv.open[row],
            ohlcv.high[row],
            ohlcv.low[row],
            ohlcv.close[row],
        );
        anyhow::ensure!(
            open.is_finite() && high.is_finite() && low.is_finite() && close.is_finite(),
            "classic/vector-ta OHLC row {row} contains a non-finite price"
        );
        anyhow::ensure!(
            open > 0.0 && high > 0.0 && low > 0.0 && close > 0.0,
            "classic/vector-ta OHLC row {row} contains a non-positive price"
        );
        anyhow::ensure!(
            low <= open.min(close) && high >= open.max(close) && high >= low,
            "classic/vector-ta OHLC row {row} violates low <= open/close <= high"
        );
    }

    if let Some(timestamps) = &ohlcv.timestamp {
        anyhow::ensure!(
            timestamps.len() == n,
            "classic/vector-ta timestamp length {} does not match {n} OHLC rows",
            timestamps.len()
        );
        validate_canonical_millisecond_timestamps(timestamps)?;
    }
    if let Some(volume) = &ohlcv.volume {
        anyhow::ensure!(
            volume.len() == n,
            "classic/vector-ta volume length {} does not match {n} OHLC rows",
            volume.len()
        );
        for (row, value) in volume.iter().copied().enumerate() {
            anyhow::ensure!(
                value.is_finite() && value >= 0.0,
                "classic/vector-ta volume row {row} must be finite and non-negative"
            );
        }
    }
    Ok(())
}

fn classic_indicator_id_for_column(name: &str) -> Option<&'static str> {
    ALL_INDICATORS
        .iter()
        .copied()
        .filter(|id| {
            name == *id
                || name
                    .strip_prefix(*id)
                    .is_some_and(|suffix| suffix.starts_with('_'))
        })
        .max_by_key(|id| id.len())
}

fn classify_classic_ta_validity(
    name: &str,
    values: &[f64],
    ledger: &IndicatorLedger,
) -> Vec<FeatureCellValidity> {
    if values.iter().all(|value| value.is_nan()) {
        let reason = if ledger
            .drop_reasons_for_column(name)
            .contains(&DropReason::PreflightWarmup)
        {
            FeatureCellValidity::Warmup
        } else if ledger
            .drop_reasons_for_column(name)
            .contains(&DropReason::MissingRequiredInput)
        {
            FeatureCellValidity::MissingInput
        } else {
            FeatureCellValidity::ComputeFailure
        };
        return vec![reason; values.len()];
    }

    let mut observed_finite = false;
    values
        .iter()
        .map(|value| {
            if value.is_finite() {
                observed_finite = true;
                FeatureCellValidity::Valid
            } else if !observed_finite && value.is_nan() {
                FeatureCellValidity::Warmup
            } else {
                FeatureCellValidity::NonFinite
            }
        })
        .collect()
}

/// The real entry point: same as [`compute_classic_ta_columns_with_policy`] but
/// with the vocabulary budget sized against `budget_rows` rather than against
/// THIS frame.
///
/// # Why the budget must not be sized from the frame
///
/// The vocabulary budget converts free RAM into a maximum column count, and
/// `admit_indicators` then takes the prefix of `ALL_INDICATORS` that fits. If
/// the budget is sized from each frame's own row count, the admitted ID SET
/// becomes a function of the frame — and every timeframe has a different row
/// count. Measured on the operator's box (20.6 GB free): base M5 at 1,054,320
/// bars admits 269 ids, H1 at 70,288 admits all 342, H4 at 17,572 admits all
/// 342. The per-timeframe cube widths then differ by ~140 columns,
/// `lib.rs::try_assemble_cube_in_ram` refuses to assemble (its width invariant
/// is doing exactly its job), and every run on a box under roughly 40 GB free
/// silently falls through to the slower streaming disk path.
///
/// So the CALLER sizes one budget from the run's widest frame — the base
/// timeframe — and passes that row count to every timeframe. Conservative by
/// construction: the higher timeframes are charged the base frame's per-column
/// price, so the plan can only ever over-reserve, never under-reserve.
///
/// `budget_rows` must be >= the widest frame this run will build. Passing the
/// frame's own length is correct only for a single-frame build.
pub fn compute_classic_ta_columns_sized(
    ohlcv: &Ohlcv,
    policy: IndicatorComputePolicy,
    budget_rows: usize,
) -> anyhow::Result<Vec<(String, Vec<f64>)>> {
    Ok(compute_classic_ta_columns_sized_report(ohlcv, policy, budget_rows)?.columns)
}

/// Resolve the one budget/admission/working-set decision both execution lanes
/// consume. Registry/table inspection only; no feature or device allocation.
fn build_classic_ta_admission_plan(n: usize, budget_rows: usize) -> ClassicTaAdmissionPlan {
    let budget_rows = budget_rows.max(n);
    let budget = VocabularyBudget::for_run(budget_rows);
    let sweep_reserved = planned_sweep_columns();
    let base_budget = budget.reserve(sweep_reserved);
    let all_ids: Vec<&'static str> = ALL_INDICATORS.to_vec();
    let (admitted_indicator_ids, budget_deferred_indicator_ids, planned_base_columns) =
        admit_indicators(&all_ids, &base_budget);
    let admitted_base_columns = admitted_indicator_ids
        .iter()
        .map(|id| planned_output_count(id))
        .sum::<usize>();

    // The extension is a planning fact, not a consequence of how many columns
    // one particular frame happened to produce.  Resolve it before either lane
    // allocates Candles/device buffers so CPU and CUDA consume the same request.
    let planned_so_far = admitted_base_columns + sweep_reserved;
    let unspent = budget.max_columns.saturating_sub(planned_so_far);
    let working_set = current_extended_sweep_working_set();
    let (
        extended_groups,
        extended_budget_deferred_indicator_ids,
        extended_mode,
        extended_budget_columns,
    ) = match working_set.as_deref() {
        Some(batch) => {
            if batch.planned_columns > unspent {
                tracing::warn!(
                    target: "neoethos_data::hpc_ta",
                    cursor = batch.cursor,
                    batch_columns = batch.planned_columns,
                    unspent_columns = unspent,
                    max_columns = budget.max_columns,
                    "the installed streaming working set is WIDER than this machine still \
                     affords after the base vocabulary and the historical sweep. Building it \
                     anyway — the caller sized the batch and refusing here would silently \
                     change which parameter region the run explored — but expect memory \
                     pressure. Size the batch from VocabularyBudget minus the resident plan."
                );
            }
            (
                batch.grouped_by_id(),
                Vec::new(),
                "streaming_batch",
                batch.planned_columns,
            )
        }
        None => {
            let extended_budget = unspent.min(planned_so_far);
            let (plan, deferred) = extended_sweep_plan(extended_budget);
            let groups = plan
                .into_iter()
                .map(|id| (id, extended_sweep_periods(id)))
                .collect();
            (groups, deferred, "budget_prefix", extended_budget)
        }
    };
    let extended_planned_columns = extended_groups
        .iter()
        .map(|(id, periods)| planned_output_count(id) * periods.len())
        .sum();

    ClassicTaAdmissionPlan {
        budget_rows,
        budget,
        base_budget,
        sweep_reserved,
        admitted_indicator_ids,
        budget_deferred_indicator_ids,
        capability_deferred_indicator_ids: Vec::new(),
        capability_deferred_output_count: 0,
        gpu_route_mode: "complete_graph_v1",
        historical_indicator_ids: MULTI_PERIOD_IDS.to_vec(),
        planned_base_columns,
        admitted_base_columns,
        working_set,
        extended_groups,
        extended_budget_deferred_indicator_ids,
        extended_mode,
        extended_budget_columns,
        extended_planned_columns,
    }
}

/// Capture the exact allocation-free Classic/vector-ta graph for a run.
///
/// `budget_rows` is the widest independently downloaded direct timeframe.
/// `GpuOnly` resolves every admitted output here, before feature computation;
/// an unsupported output therefore returns the complete ordered manifest
/// without creating a CUDA context or silently substituting CPU/f32 work.
pub fn prepare_classic_ta_run_plan(
    budget_rows: usize,
    policy: IndicatorComputePolicy,
) -> anyhow::Result<ClassicTaRunPlan> {
    let admission = build_classic_ta_admission_plan(budget_rows, budget_rows);
    #[cfg(feature = "gpu-cuda")]
    let mut resident_cuda_launches = None;
    if policy == IndicatorComputePolicy::GpuOnly && budget_rows > 0 {
        #[cfg(not(feature = "gpu-cuda"))]
        anyhow::bail!(
            "GpuOnly is unavailable: neoethos-data was built without the gpu-cuda feature. \
             Strict GPU execution never substitutes CpuOnly."
        );

        #[cfg(feature = "gpu-cuda")]
        {
            let plan = crate::core::classic_cuda_plan::build_exact_classic_cuda_plan(
                budget_rows,
                &admission.admitted_indicator_ids,
                &MULTI_PERIOD_IDS,
                &admission.extended_groups,
            )?;
            let resolved = crate::core::classic_cuda_plan::resolve_gpu_only_classic_plan(&plan)
                .map_err(|error| {
                    anyhow::anyhow!(
                        "GpuOnly run admission rows={} available_bytes={} max_columns={} \
                         admitted_base_ids={:?} extended_ids={:?}: {error:#}",
                        admission.budget_rows,
                        admission.budget.available_bytes,
                        admission.budget.max_columns,
                        admission.admitted_indicator_ids,
                        admission
                            .extended_groups
                            .iter()
                            .map(|(id, _)| *id)
                            .collect::<Vec<_>>(),
                    )
                })?;
            resident_cuda_launches = Some(resolved);
        }
    }
    Ok(ClassicTaRunPlan {
        policy,
        admission,
        #[cfg(feature = "gpu-cuda")]
        resident_cuda_launches,
    })
}

/// Seal the explicitly versioned exact-routeable Classic subset used by
/// non-Full GPU-only milestone/search profiles. Every family with any missing
/// output contract or missing exact pre-device allocation authority is removed
/// atomically, recorded by canonical id and exact deferred output count, and
/// bound into the resident recipe identity. The Full profile keeps using
/// [`prepare_classic_ta_run_plan`] and therefore still refuses the same complete
/// gap manifest without excluding anything.
#[cfg(feature = "gpu-cuda")]
pub(crate) fn prepare_classic_ta_gpu_exact_parity_run_plan_v3(
    budget_rows: usize,
) -> anyhow::Result<ClassicTaRunPlan> {
    let mut admission = build_classic_ta_admission_plan(budget_rows, budget_rows);
    anyhow::ensure!(
        admission.working_set.is_none(),
        "gpu_only_exact_parity_subset_v3 refuses an installed extended sweep working set"
    );
    let full_plan = crate::core::classic_cuda_plan::build_exact_classic_cuda_plan(
        budget_rows,
        &admission.admitted_indicator_ids,
        &admission.historical_indicator_ids,
        &admission.extended_groups,
    )?;
    let gaps = crate::core::classic_cuda_plan::preflight_exact_classic_cuda_plan(&full_plan)
        .err()
        .unwrap_or_default();
    let mut deferred = gaps
        .iter()
        .map(|gap| gap.indicator_id)
        .collect::<HashSet<_>>();
    deferred.extend(GPU_ONLY_PARITY_DEFERRED_INDICATORS_V3.iter().copied());

    // Routeability alone is insufficient: the run-device carrier may only be
    // consumed after every launch has an exact allocation receipt. Today that
    // authority is complete for primary single-output f64 sweeps; named owners
    // stay fail-closed until their retained parameter/scratch sizing is sealed.
    let candidate_admitted_indicator_ids = admission
        .admitted_indicator_ids
        .iter()
        .copied()
        .filter(|indicator_id| !deferred.contains(indicator_id))
        .collect::<Vec<_>>();
    let candidate_historical_indicator_ids = admission
        .historical_indicator_ids
        .iter()
        .copied()
        .filter(|indicator_id| !deferred.contains(indicator_id))
        .collect::<Vec<_>>();
    let candidate_extended_groups = admission
        .extended_groups
        .iter()
        .filter(|(indicator_id, _)| !deferred.contains(indicator_id))
        .cloned()
        .collect::<Vec<_>>();
    let candidate_plan = crate::core::classic_cuda_plan::build_exact_classic_cuda_plan(
        budget_rows,
        &candidate_admitted_indicator_ids,
        &candidate_historical_indicator_ids,
        &candidate_extended_groups,
    )?;
    let candidate_launches =
        crate::core::classic_cuda_plan::resolve_gpu_only_classic_plan(&candidate_plan)?;
    let mut allocation_deferred = HashSet::new();
    let mut node_cursor = 0usize;
    for launch in &candidate_launches {
        let output_count = launch.output_count();
        let node_end = node_cursor.checked_add(output_count).ok_or_else(|| {
            anyhow::anyhow!("Classic exact-routeable allocation census output cursor overflowed")
        })?;
        let launch_nodes = candidate_plan
            .nodes
            .get(node_cursor..node_end)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Classic exact-routeable allocation census exceeded the candidate graph"
                )
            })?;
        let indicator_id = launch_nodes
            .first()
            .ok_or_else(|| {
                anyhow::anyhow!("Classic exact-routeable allocation census found an empty launch")
            })?
            .indicator_id;
        anyhow::ensure!(
            launch_nodes
                .iter()
                .all(|node| node.indicator_id == indicator_id),
            "Classic exact-routeable launch crossed indicator-family ownership"
        );
        if !matches!(
            launch,
            crate::core::classic_cuda_plan::ResolvedClassicCudaLaunch::Primary(_)
        ) {
            allocation_deferred.insert(indicator_id);
        }
        node_cursor = node_end;
    }
    anyhow::ensure!(
        node_cursor == candidate_plan.nodes.len(),
        "Classic exact-routeable allocation census did not consume the candidate graph"
    );
    deferred.extend(allocation_deferred);

    admission.capability_deferred_indicator_ids = ALL_INDICATORS
        .iter()
        .copied()
        .filter(|indicator_id| deferred.contains(indicator_id))
        .collect();
    admission.capability_deferred_output_count = full_plan
        .nodes
        .iter()
        .filter(|node| deferred.contains(node.indicator_id))
        .count();
    admission.gpu_route_mode = "gpu_only_exact_parity_subset_v3";
    admission
        .admitted_indicator_ids
        .retain(|indicator_id| !deferred.contains(indicator_id));
    admission.admitted_base_columns = admission
        .admitted_indicator_ids
        .iter()
        .map(|indicator_id| planned_output_count(indicator_id))
        .sum();
    admission.historical_indicator_ids = MULTI_PERIOD_IDS
        .iter()
        .copied()
        .filter(|indicator_id| !deferred.contains(indicator_id))
        .collect();
    admission.sweep_reserved = admission
        .historical_indicator_ids
        .iter()
        .map(|indicator_id| planned_output_count(indicator_id) * ALT_PERIODS.len())
        .sum();
    admission
        .extended_groups
        .retain(|(indicator_id, _)| !deferred.contains(indicator_id));
    admission.extended_budget_columns = admission
        .extended_groups
        .iter()
        .map(|(indicator_id, periods)| planned_output_count(indicator_id) * periods.len())
        .sum();
    admission.extended_planned_columns = admission.extended_budget_columns;
    admission.extended_mode = "gpu_only_exact_parity_subset_v3";

    let routeable_plan = crate::core::classic_cuda_plan::build_exact_classic_cuda_plan(
        budget_rows,
        &admission.admitted_indicator_ids,
        &admission.historical_indicator_ids,
        &admission.extended_groups,
    )?;
    let resident_cuda_launches =
        crate::core::classic_cuda_plan::resolve_gpu_only_classic_plan(&routeable_plan)?;
    anyhow::ensure!(
        !resident_cuda_launches.is_empty()
            && !admission.capability_deferred_indicator_ids.is_empty()
            && admission.capability_deferred_output_count
                == full_plan
                    .nodes
                    .iter()
                    .filter(|node| deferred.contains(node.indicator_id))
                    .count(),
        "gpu_only_exact_parity_subset_v3 produced an invalid debt receipt"
    );
    Ok(ClassicTaRunPlan {
        policy: IndicatorComputePolicy::GpuOnly,
        admission,
        resident_cuda_launches: Some(resident_cuda_launches),
    })
}

/// Required-card fixture oracle for the exact versioned Classic subset.
/// Admission is resolved through the same GpuOnly plan as production, then
/// only the execution policy is switched to CPU so the fixture compares the
/// identical ordered route graph without launching a second GPU authority.
#[cfg(feature = "gpu-cuda-device-fixtures")]
pub fn compute_classic_ta_gpu_exact_parity_feature_columns_for_device_fixture_v3(
    ohlcv: &Ohlcv,
    budget_rows: usize,
) -> anyhow::Result<Vec<FeatureColumnF64>> {
    let mut run_plan =
        prepare_classic_ta_gpu_exact_parity_run_plan_v3(budget_rows.max(ohlcv.len()))?;
    run_plan.policy = IndicatorComputePolicy::CpuOnly;
    run_plan.resident_cuda_launches = None;
    compute_classic_ta_feature_columns_f64_with_run_plan(ohlcv, &run_plan)
}

/// Sized form of [`compute_classic_ta_columns_with_policy_report`].
pub fn compute_classic_ta_columns_sized_report(
    ohlcv: &Ohlcv,
    policy: IndicatorComputePolicy,
    budget_rows: usize,
) -> anyhow::Result<ClassicTaComputation> {
    let run_plan = prepare_classic_ta_run_plan(budget_rows.max(ohlcv.len()), policy)?;
    compute_classic_ta_columns_sized_report_with_run_plan(ohlcv, &run_plan)
}

/// Sized execution through one previously captured run-wide admission plan.
pub fn compute_classic_ta_columns_sized_report_with_run_plan(
    ohlcv: &Ohlcv,
    run_plan: &ClassicTaRunPlan,
) -> anyhow::Result<ClassicTaComputation> {
    let policy = run_plan.policy;
    if policy == IndicatorComputePolicy::GpuOnly {
        #[cfg(not(feature = "gpu-cuda"))]
        anyhow::bail!(
            "GpuOnly is unavailable: neoethos-data was built without the gpu-cuda feature. \
             Strict GPU execution never substitutes CpuOnly."
        );
    }
    let n = ohlcv.len();
    if n == 0 {
        return Ok(ClassicTaComputation {
            columns: Vec::new(),
            report: ClassicTaExecutionReport {
                budget_rows: 0,
                available_bytes_at_admission: 0,
                max_columns: 0,
                admitted_indicator_ids: Vec::new(),
                budget_deferred_indicator_ids: Vec::new(),
                capability_deferred_indicator_ids: Vec::new(),
                capability_deferred_output_count: 0,
                planned_base_columns: 0,
                admitted_base_columns: 0,
                historical_sweep_reserved_columns: 0,
                historical_sweep_produced_columns: 0,
                extended_mode: "empty_frame",
                extended_admitted_indicator_ids: Vec::new(),
                extended_budget_deferred_indicator_ids: Vec::new(),
                extended_budget_columns: 0,
                extended_planned_columns: 0,
                produced_columns: 0,
            },
            ledger: IndicatorLedger::new(),
        });
    }
    anyhow::ensure!(
        n <= run_plan.admission.budget_rows,
        "classic/vector-ta frame has {n} rows but the frozen run plan was sized for only {}",
        run_plan.admission.budget_rows
    );
    let admission = run_plan.admission.clone();

    #[cfg(feature = "gpu-cuda")]
    if policy == IndicatorComputePolicy::GpuOnly {
        let plan = crate::core::classic_cuda_plan::build_exact_classic_cuda_plan(
            n,
            &admission.admitted_indicator_ids,
            &admission.historical_indicator_ids,
            &admission.extended_groups,
        )?;
        return crate::core::classic_cuda_plan::execute_gpu_only_classic_plan(
            ohlcv, plan, admission,
        );
    }

    let budget_rows = admission.budget_rows;
    let input_availability = ClassicInputAvailability::from_ohlcv(ohlcv);

    // 1. Pack data into VectorTA Candles struct (once; shared read-only
    //    across the rayon workers below — `Candles` holds plain Vecs so it
    //    is `Sync`).
    let timestamps = ohlcv.timestamp.clone().unwrap_or_else(|| vec![0i64; n]);
    let volume = ohlcv.volume.clone().unwrap_or_else(|| vec![0.0; n]);

    let candles = Candles::new(
        timestamps,
        ohlcv.open.clone(),
        ohlcv.high.clone(),
        ohlcv.low.clone(),
        ohlcv.close.clone(),
        volume,
    );

    // 2. SIZE THE VOCABULARY BEFORE ALLOCATING ANY OF IT.
    //
    //    Repairing the dispatch takes the base pass from 1 produced column to
    //    ~674, and at 843,456 bars an f64 column is 6.75 MB — so the staging
    //    peak goes from 6.7 MB to 4.5 GB per timeframe. `VocabularyBudget`
    //    turns FREE RAM (never a user parameter, never a constant) into a
    //    maximum column count, and `admit_indicators` takes a deterministic
    //    prefix of `ALL_INDICATORS` that fits it. Planning is registry lookups
    //    only, so the decision is made before a single Vec is allocated.
    //
    //    THE SWEEP IS RESERVED FIRST. The period sweep used to be staged
    //    entirely OUTSIDE this budget, so `max_columns` was never the peak: at
    //    the M5 store's 1,054,320 bars its 120 columns are another 1.01 GB of
    //    f64 staging that nothing accounted for, and it mattered most exactly
    //    where the budget binds. The statically valid critical sweep has first
    //    claim, and the base pass takes what is left.
    let budget = &admission.budget;
    let base_budget = &admission.base_budget;
    let admitted = &admission.admitted_indicator_ids;
    let planned_columns = admission.planned_base_columns;
    let base_admitted_plan = admission.admitted_base_columns;

    // 3. Dispatch to every admitted indicator — PARALLEL across indicators.
    //    Each indicator is an independent pure function of the shared
    //    `candles`, so this is a rayon `par_iter` whose results are collected
    //    BY POSITION, preserving `ALL_INDICATORS` order exactly. The feature
    //    build runs ONCE up-front (not inside the GA candidate `par_iter`), so
    //    this does not nest with the discovery hot path.
    //
    //    ── THE TWO FIXES, AND WHY BOTH WERE NEEDED ────────────────────────
    //
    //    (a) OUTPUT IDS ARE ENUMERATED. `output_id: None` against a
    //        multi-output indicator is answered by vector-ta with
    //        `InvalidParam { key: "output_id" }` — 92 of the 342 ids failed on
    //        exactly this, and it is why every multi-output indicator (macd,
    //        bollinger_bands, stoch, keltner, supertrend, …) produced nothing
    //        anywhere in the system. `compute_single_indicator` in this same
    //        file has always done it correctly for the chart endpoint; this
    //        loop never did. `output_ids_for` is now the ONE place that
    //        resolves it, for both call sites.
    //
    //    (b) THE ACCEPT TEST KEYS OFF THE VALUE COUNT, NOT rows/cols.
    //        vector-ta reports a 1-D series as `rows=1 x cols=n`, so the old
    //        `if out_cols <= 1` was FALSE for every single-output f64
    //        indicator on any frame with more than one bar. Those fell into
    //        the multi-output branch, which then required `rows >= n` — i.e.
    //        `1 >= 200000` — and dropped them with no else and no log. That
    //        alone accounted for 232 of the 342: series that were CORRECT,
    //        FULL-LENGTH, and thrown away. `flatten_indicator_series` (used by
    //        the chart path, where this lesson was learned and never carried
    //        back up) keys off `v.len()` and is now used here too.
    //
    //    Everything a worker discards is recorded in its own `IndicatorLedger`
    //    with a reason and the dispatch message. Nothing on this path is
    //    allowed to vanish.
    let per_id: Vec<(Vec<(String, Vec<f64>)>, IndicatorLedger)> = admitted
        .par_iter()
        .map(|&id| {
            let mut out: Vec<(String, Vec<f64>)> = Vec::new();
            let mut ledger = IndicatorLedger::new();
            dispatch_indicator_outputs(
                &candles,
                id,
                id,
                &[],
                n,
                Kernel::Auto,
                input_availability,
                &mut out,
                &mut ledger,
            );
            (out, ledger)
        })
        .collect();

    let mut ledger = admission.admission_ledger();
    let mut cols: Vec<(String, Vec<f64>)> = Vec::new();
    for (mut produced, led) in per_id {
        cols.append(&mut produced);
        ledger.merge(led);
    }
    // Admission is a pre-compute planning fact. `cols.len()` is the number
    // that survived dispatch and may be smaller because of unknown outputs,
    // unsupported capabilities or kernel errors; passing it here falsely
    // labels those defects as RAM deferrals. The indicator ledger below owns
    // production failures, while this line reports only the actual budget
    // decision.
    base_budget.log("base-vocabulary", planned_columns, base_admitted_plan);

    // 3. Multi-period variants for the most critical indicators. Appended
    //    after the base columns to preserve the original ordering exactly.
    //
    //    ROUTING: this body is the CPU reference only. `GpuOnly` returned above
    //    through the exact typed plan before Candles existed, so no individual
    //    id may switch lane here. Multi-output ids are dispatched by their
    //    canonical `output_ids_for` identities on CPU; their strict CUDA
    //    counterparts must all preflight and launch inside one resident engine
    //    or the whole GPU request fails before work.
    //
    //    ORDER IS LOAD-BEARING: column order feeds `effective_feature_names`
    //    and every discovery artifact. The per-indicator results are collected
    //    into a slot indexed by position in `MULTI_PERIOD_IDS` and then
    //    concatenated in that order, so the emitted order is byte-identical to
    //    the pure-CPU path regardless of which lane produced which column.
    let (multi_cols, multi_ledger) = compute_multi_period_columns(
        ohlcv,
        &candles,
        n,
        policy,
        &admission.historical_indicator_ids,
    );
    let sweep_actual = multi_cols.len();
    cols.extend(multi_cols);
    ledger.merge(multi_ledger);

    // 4b. THE EXTENDED SWEEP — the answer to "a period is not a detail".
    //
    //     Repairing the dispatch bought 695 base columns, and every one of them
    //     is computed at ONE vector-ta default. That is not what was asked for:
    //     RSI(4) is a scalping oscillator and RSI(54) is a regime filter, and a
    //     vocabulary that can only reach one of them is 300 arbitrary points in
    //     a space nobody chose. `period_plan` already knows how to drive each
    //     indicator's real window parameters — it was simply unreachable for
    //     anything outside the hardcoded sixteen.
    //
    //     So every id whose window is drivable (`Key` or `Ratio`) and that is
    //     NOT already in `MULTI_PERIOD_IDS` is swept too, in `ALL_INDICATORS`
    //     order, for as many (id, period) pairs as the machine can still
    //     afford after the base pass and the historical sweep have been paid
    //     for. On a box with no headroom this admits nothing and the emitted
    //     column set is exactly what it was — the extension can only ever use
    //     memory that was going to go unused.
    //
    //     This is a WORKING-SET step toward the streaming design in
    //     `docs/streaming-parameter-search.md`, not the design itself: it still
    //     materialises one cube. What it establishes is the part that design
    //     needs and did not have — a hardware-sized, deterministic, repeatable
    //     admission over the (indicator, period) space.
    //     BOUNDED BY THE VOCABULARY IT EXTENDS, NOT ONLY BY RAM.
    //
    //     Memory is not the only resource the never-OOM reasoning has to cover:
    //     a run that does not finish is no more useful than one that OOMs. The
    //     extension's memory cost is bounded by `budget`, but its TIME cost is
    //     not bounded by anything — and the two diverge exactly where the frame
    //     is short and the box is large, because the per-column price then
    //     collapses while the per-column WORK does not. Measured: at 6,000 bars
    //     with 18.8 GB free the budget allows 4,096 columns, so the extension
    //     admits every sweepable id and the pass goes from ~9 s to ~12 min.
    //
    //     So the extension may never plan more columns than the vocabulary it
    //     extends. That is a ratio to a measured quantity rather than a
    //     constant — it scales with the machine through `base_admitted_plan` —
    //     and it is deliberately a stopgap: the real answer is not to
    //     materialise the extension at all but to stream it, which is
    //     `docs/streaming-parameter-search.md`.
    //
    //     SIZED FROM THE PLAN, NEVER FROM WHAT THIS FRAME HAPPENED TO PRODUCE.
    //     Using `cols.len()` here would make the admitted extension a function
    //     of the frame, which is the same defect one level up: two timeframes
    //     would extend by different amounts and the cube widths would diverge.
    //
    //     STREAMING OVERRIDE (2026-08-10). When a working set is installed by
    //     `install_extended_sweep_working_set`, the extension is EXACTLY that
    //     batch's (indicator, period) pairs and the budget-capped prefix is not
    //     consulted. With no working set installed — the default, and every
    //     existing caller — this block is byte-identical to what it was: same
    //     plan function, same statically valid/distinct period points per id,
    //     same emission order,
    //     same deferral ledger. That identity is the parity property, and it is
    //     structural rather than tested-by-luck: both lanes run through the SAME
    //     `sweep_one_id_ledgered`.
    //
    //     THE STOPGAP STAYS, AND HERE IS WHY IT MUST. The measurement that
    //     motivated it — "~9 s to ~12 min at 6,000 bars with 4,096 columns
    //     allowed" — DID NOT REPRODUCE at HEAD: the production pass at 6,000
    //     bars with `max_columns` 4,096 takes 1.36 s and emits 1,795 columns,
    //     and the cap binds at every bar count measured (1,795 produced against
    //     4,096 / 4,096 / 3,244 allowed at 6k / 20k / 200k). But the cost that
    //     the cap bounds is per-column-COMPUTED, not per-column-HELD — 342 ids
    //     take 1.10 s at 6k bars and 10.47 s at 60k — so STREAMING DOES NOT
    //     REMOVE IT. Ten batches of a tenth of the space cost what one pass over
    //     the whole space costs, plus ten prefilters instead of one. Removing
    //     the cap here would therefore hand back an unbounded-in-time extension
    //     with nothing having been fixed. What WOULD remove it is a cheaper
    //     dispatch, and the target is named: at 6,000 bars the slowest TEN of
    //     342 ids are 79.3% of the base pass and the slowest ten of the sweep
    //     are 86.2% of it (`goertzel_cycle_composite_wave` 2.17 s,
    //     `smooth_theil_sen` 1.12 s, `volume_adjusted_ma` 0.79 s, each for five
    //     periods).
    let spent = cols.len();
    let working_set = &admission.working_set;
    let ext_groups = &admission.extended_groups;
    let ext_mode = admission.extended_mode;
    if !ext_groups.is_empty() {
        let ext: Vec<(Vec<(String, Vec<f64>)>, IndicatorLedger)> = ext_groups
            .par_iter()
            .map(|(id, periods)| {
                sweep_one_id_ledgered(&candles, *id, periods, n, Kernel::Auto, input_availability)
            })
            .collect();
        for (mut c, l) in ext {
            cols.append(&mut c);
            ledger.merge(l);
        }
    }
    tracing::info!(
        target: "neoethos_data::hpc_ta",
        rows = n,
        budget_rows,
        max_columns = budget.max_columns,
        base_columns = spent - sweep_actual,
        sweep_columns = sweep_actual,
        extended_mode = ext_mode,
        extended_cursor = working_set.as_deref().map(|b| b.cursor).unwrap_or(0),
        extended_space_len = working_set.as_deref().map(|b| b.space_len).unwrap_or(0),
        extended_ids = ext_groups.len(),
        extended_pairs = ext_groups.iter().map(|(_, p)| p.len()).sum::<usize>(),
        extended_deferred = admission.extended_budget_deferred_indicator_ids.len(),
        extended_columns = cols.len() - spent,
        total_columns = cols.len(),
        "indicator vocabulary composition (base pass at library defaults + period sweeps)"
    );

    // 5. CENSUS. Fingerprint every column exactly once and report the two
    //    quality facts the old code could not have known:
    //
    //      * DUPLICATES — a column bit-identical to an earlier one on this
    //        frame. Formula-proven aliases and ignored/saturated sweep points
    //        are removed statically before dispatch; remaining matches are
    //        reported as possible corpus coincidences, never dropped from one
    //        frame. The schema therefore depends on the production contract,
    //        not on market values.
    //      * DEGENERATE — no finite variation on this frame. Ballast for any
    //        correlation-ranked prefilter. Also kept, for the same reason.
    //
    //      * A DUPLICATE NAME, by contrast, is a HARD ERROR. Two columns called
    //        the same thing is not a quality problem, it is a correctness one:
    //        every downstream projection is by name, so the second silently
    //        shadows the first in any name→index map. The extended sweep emits
    //        `<id>_<period>` names into the same namespace as the base pass's
    //        `<id>_<output>` names, so this is now a reachable collision rather
    //        than a theoretical one.
    {
        let mut seen: HashSet<u64> = HashSet::with_capacity(cols.len());
        let mut names: HashSet<&str> = HashSet::with_capacity(cols.len());
        let mut collisions: Vec<&str> = Vec::new();
        let mut infinite_columns: Vec<(&str, usize, usize, f64)> = Vec::new();
        for (name, values) in &cols {
            if !names.insert(name.as_str()) {
                collisions.push(name.as_str());
            }
            let mut infinite_count = 0usize;
            let mut first_infinite: Option<(usize, f64)> = None;
            for (row, &value) in values.iter().enumerate() {
                if value.is_infinite() {
                    infinite_count += 1;
                    first_infinite.get_or_insert((row, value));
                }
            }
            if let Some((row, value)) = first_infinite {
                infinite_columns.push((name.as_str(), infinite_count, row, value));
            }
            if !seen.insert(series_fingerprint(values)) {
                ledger.duplicate_column(name);
            }
            if !has_finite_variation(values) {
                ledger.degenerate_column(name);
            }
        }
        if !collisions.is_empty() {
            anyhow::bail!(
                "the indicator pass emitted {} DUPLICATE COLUMN NAME(S): {:?}. Column names are \
                 the only key every downstream projection has, so a collision silently shadows a \
                 real feature. Rename the sweep suffix for the id(s) involved.",
                collisions.len(),
                collisions.iter().take(20).collect::<Vec<_>>()
            );
        }
        if !infinite_columns.is_empty() {
            anyhow::bail!(
                "the indicator pass emitted infinity in {} column(s): {:?}. NaN is the explicit \
                 validity representation for warmup/gaps; +/-infinity is never a valid market \
                 feature. Repair the independently reviewed formula or exclude the indicator \
                 statically from the production vocabulary — never clamp, zero-fill, or drop it \
                 based on this frame.",
                infinite_columns.len(),
                infinite_columns.iter().take(20).collect::<Vec<_>>()
            );
        }
    }

    ledger.log_summary("classic-ta", n);

    // 6. THE FLOOR. This is the whole point of the ledger: a regression from
    //    ~800 columns back to 66 is not a log line anyone has to notice, it is
    //    a hard error that names the drop bucket which grew.
    //
    //    Length-gated, and honestly so: on a short frame most indicators fail
    //    their warmup legitimately, so enforcing there would make the guard
    //    fire on fixtures instead of on regressions.
    //
    //    CLAMPED BY WHAT THE MACHINE AFFORDED. The floor and the budget were
    //    added in the same change and were never exercised together: at the M5
    //    store's real depth on the operator's own box the budget admits 269 ids
    //    and this floor demands 280, so the feature build hard-errored and
    //    discovery could not start. "This machine cannot afford the vocabulary"
    //    and "the dispatch regressed" are different incidents; the budget's own
    //    truncation is already a WARN, and this floor is only about the second.
    //    See `IndicatorLedger::enforce_floor`.
    if n >= VOCABULARY_FLOOR_MIN_ROWS {
        ledger.enforce_floor(
            "classic-ta",
            n,
            MIN_PRODUCING_INDICATOR_IDS,
            MIN_BASE_VOCABULARY_COLUMNS,
            admitted.len(),
            base_admitted_plan,
        )?;
    } else {
        tracing::info!(
            target: "neoethos_data::hpc_ta",
            rows = n,
            floor_min_rows = VOCABULARY_FLOOR_MIN_ROWS,
            producing_ids = ledger.producing_ids(),
            columns = cols.len(),
            "frame is shorter than the vocabulary floor's minimum row count — the census above \
             was recorded but the hard floor was NOT enforced for this frame"
        );
    }

    let report = admission.execution_report(sweep_actual, cols.len());

    Ok(ClassicTaComputation {
        columns: cols,
        report,
        ledger,
    })
}

/// Dispatch ONE indicator across its statically admitted production outputs,
/// appending each produced column and recording every discard with a reason.
///
/// This is the single place the two dispatch mistakes are fixed, shared by the
/// base pass and the multi-period sweep so they cannot drift apart again:
///
///   * `output_ids_for` supplies the `output_id` a multi-output indicator
///     requires, resolved from vector-ta's registry (or, for the five
///     multi-output ids that have no registry entry, from the override table
///     in `core::indicator_ledger` harvested from the dispatcher source);
///   * `flatten_indicator_series` keys acceptance off the VALUE COUNT, because
///     vector-ta reports a 1-D series as `rows=1 x cols=n`.
///
/// `column_prefix` is the base column name — `"rsi"` for the base pass,
/// `"rsi_21"` for the sweep. Every `Some(output_id)` suffixes that semantic id
/// (`"macd_21_signal"`), even when static filtering leaves only one named
/// output, so the name never changes when a redundant sibling is removed.
fn dispatch_indicator_outputs(
    candles: &Candles,
    id: &'static str,
    column_prefix: &str,
    params: &[ParamKV],
    n: usize,
    kernel: Kernel,
    input_availability: ClassicInputAvailability,
    out: &mut Vec<(String, Vec<f64>)>,
    ledger: &mut IndicatorLedger,
) {
    if let Some(detail) = input_availability.missing_for(id) {
        let first = out.len();
        push_absent_columns(id, column_prefix, n, out);
        for (name, _) in &out[first..] {
            ledger.dropped(
                id,
                name,
                DropReason::MissingRequiredInput,
                format!("{detail}; no scalar kernel was launched"),
            );
        }
        return;
    }

    let outputs = output_ids_for(id);
    let excluded = expected_non_producing(id);

    for out_id in outputs {
        let name = match out_id {
            Some(o) => format!("{column_prefix}_{o}"),
            None => column_prefix.to_string(),
        };
        let data_ref = IndicatorDataRef::Candles {
            candles,
            source: None,
        };
        let req = IndicatorComputeRequest {
            indicator_id: id,
            output_id: out_id,
            data: data_ref,
            params,
            kernel,
        };
        // #212: a small subset of indicator/data combinations in vector-ta
        // v0.2.9 panic instead of returning Err. Catch it per-output so one bad
        // column never tears down the frame — but COUNT it, which the previous
        // handler did only as an un-aggregated warn.
        let computed = catch_unwind(AssertUnwindSafe(|| compute_cpu(req)));
        // ONE place decides what a discard does to the COLUMN SET.
        //
        // A capability failure (no dispatch arm, no such output, unsupported
        // kernel) removes the column, and that absence is what the vocabulary
        // floor detects a regression by. A FRAME failure (warmup, data length,
        // compute) keeps the name and fills it with NaN, because otherwise the
        // emitted width would be a function of the frame — and every timeframe
        // has a different one, so the cube could not be assembled. See
        // `DropReason::is_frame_dependent`.
        let mut discard: Option<(DropReason, String)> = None;
        match computed {
            Err(_) => {
                discard = Some((
                    DropReason::KernelPanic,
                    "vector-ta kernel panicked (issue #212)".to_string(),
                ));
            }
            Ok(Err(e)) => {
                discard = Some((DropReason::from_dispatch(&e), e.to_string()));
            }
            Ok(Ok(output)) => {
                // A PATTERN MATRIX IS NOT A LONG SERIES.
                //
                // `normalize_indicator_len` correctly refuses `62 * n` values
                // for an n-bar frame — taking the head would return pattern 0's
                // flags under the whole indicator's name. But refusing is not
                // the same as having nothing to emit: the library hands back
                // `pattern_ids`, one name per row, so the matrix decomposes
                // exactly. Handled BEFORE the flatten so the refusal stays the
                // rule for everything that has no such metadata.
                if let Some(matrix) = pattern_matrix_columns(&output, &name, n) {
                    if excluded.is_some() {
                        ledger.stale_exclusion(id);
                    }
                    for col in matrix {
                        ledger.produced(id);
                        out.push(col);
                    }
                    continue;
                }
                match flatten_indicator_series(output.series, n) {
                    Ok((values, raw_len)) => {
                        if raw_len > n {
                            // The tail was dropped. A discard is a discard even when
                            // the column survives it.
                            ledger.dropped(
                                id,
                                &name,
                                DropReason::Truncated,
                                format!("kernel returned {raw_len} values for {n} bars; head kept"),
                            );
                        }
                        if excluded.is_some() {
                            // The exclusion table said this id cannot produce. It
                            // did. That means the table is stale, which is a thing
                            // to fix, not to shrug at.
                            ledger.stale_exclusion(id);
                        }
                        ledger.produced(id);
                        out.push((name.clone(), values));
                    }
                    Err(e) => {
                        discard = Some((DropReason::ShortSeries, e.to_string()));
                    }
                }
            }
        }
        if let Some((reason, detail)) = discard {
            ledger.dropped(id, &name, reason, detail);
            if reason.is_frame_dependent() {
                out.push((name, vec![f64::NAN; n]));
            }
        }
    }
}

/// Emit the columns an indicator WOULD have produced, filled with NaN, using
/// exactly the names `dispatch_indicator_outputs` would have used.
///
/// Used only where the skip is a function of the FRAME rather than of the
/// indicator's capability — today that is the `#212` warmup pre-flight guard,
/// which reads `n` directly. Those skips must not change the column set,
/// because every timeframe has a different `n` and the cube's width invariant
/// would then fail on every multi-timeframe run.
///
/// NaN, never zero: a zero is a real number the GA can threshold against, and
/// "this frame cannot support this period" is an absence, not a reading. It is
/// also what the warmup prefix of every windowed indicator already looks like,
/// so nothing downstream meets a new shape.
fn push_absent_columns(
    id: &'static str,
    column_prefix: &str,
    n: usize,
    out: &mut Vec<(String, Vec<f64>)>,
) {
    let outputs = output_ids_for(id);
    for out_id in outputs {
        let name = match out_id {
            Some(o) => format!("{column_prefix}_{o}"),
            None => column_prefix.to_string(),
        };
        out.push((name, vec![f64::NAN; n]));
    }
}

/// Decompose a pattern MATRIX output into one named signed column per pattern,
/// or `None` when this output is not a matrix.
///
/// `pattern_recognition` is the only id in vector-ta 0.2.9 that returns one:
/// `rows = PATTERN_RUNNERS.len()` and `cols = bars`, laid out PATTERN-MAJOR as
/// `values_i8[row * cols .. row * cols + cols]`. One row is one pattern's
/// series across all bars. The signed `-100/-80/0/80/100` values preserve both
/// direction and pattern strength through the production feature boundary.
///
/// Every one of the three shape facts is CHECKED rather than trusted — the row
/// count against the id list, the column count against the frame, and the value
/// count against their product. A mismatch returns `None`, and the caller then
/// takes the normal path, where `normalize_indicator_len` refuses the flattened
/// multi-series with a named hard error. Guessing an orientation here would
/// produce one column per pattern with silent mis-attribution, which is strictly worse than
/// the drop it replaces.
fn pattern_matrix_columns(
    output: &IndicatorComputeOutput,
    column_prefix: &str,
    n: usize,
) -> Option<Vec<(String, Vec<f64>)>> {
    let ids = output.pattern_ids.as_ref()?;
    if ids.is_empty() || output.rows != ids.len() || output.cols != n {
        return None;
    }
    let IndicatorSeries::I32(values) = &output.series else {
        return None;
    };
    if values.len() != output.rows.checked_mul(output.cols)? {
        return None;
    }
    if !values
        .iter()
        .all(|value| matches!(*value, -100 | -80 | 0 | 80 | 100))
    {
        return None;
    }
    Some(
        ids.iter()
            .enumerate()
            .map(|(row, pattern)| {
                let start = row * n;
                let series = values[start..start + n]
                    .iter()
                    .map(|&value| value as f64)
                    .collect();
                (format!("{column_prefix}_{pattern}"), series)
            })
            .collect(),
    )
}

/// The 16 indicators that declare a real, production-relevant period sweep, in
/// emission order. No-window indicators are deliberately absent: inventing a
/// `period` for OBV or VWAP only creates five aliases of their base feature.
pub const MULTI_PERIOD_IDS: [&str; 16] = [
    "rsi",
    "ema",
    "sma",
    "atr",
    "adx",
    "cci",
    "stoch",
    "macd",
    "bollinger_bands",
    "keltner",
    "supertrend",
    "willr",
    "roc",
    "mom",
    "tsi",
    "mfi",
];

/// The periods swept for each of [`MULTI_PERIOD_IDS`].
pub const ALT_PERIODS: [usize; 5] = [7, 21, 50, 100, 200];

/// Columns the HISTORICAL period sweep will stage, planned from the registry
/// before anything is allocated.
///
/// Registry lookups only — no compute — so this can be subtracted from the
/// machine's budget BEFORE the base pass is admitted. That ordering is the
/// point: the statically valid critical sweep has first claim on memory, so a
/// constrained machine does not silently lose the parameterised vocabulary.
///
/// It over-counts on short frames — the `#212` pre-flight guard skips periods
/// whose warmup exceeds the frame — and that is the correct direction for a
/// budget: a plan may over-reserve, never under-reserve.
pub fn planned_sweep_columns() -> usize {
    MULTI_PERIOD_IDS
        .iter()
        .map(|id| planned_output_count(id) * ALT_PERIODS.len())
        .sum()
}

/// Columns the RESIDENT part of a pass will stage on this machine: the admitted
/// base vocabulary plus the historical period sweep, i.e. everything that is
/// present in every batch and is not the streaming extension.
///
/// This is the same arithmetic `compute_classic_ta_columns_sized` performs, in
/// ONE place, so a streaming loop that sizes its batch from
/// `max_columns - resident` cannot drift from what the pass actually spends.
/// Registry lookups only; nothing is allocated.
pub fn planned_resident_columns(budget_rows: usize) -> usize {
    let budget = VocabularyBudget::for_run(budget_rows);
    let sweep_reserved = planned_sweep_columns();
    let base_budget = budget.reserve(sweep_reserved);
    let all_ids: Vec<&'static str> = ALL_INDICATORS.to_vec();
    let (admitted, _deferred, _planned) = admit_indicators(&all_ids, &base_budget);
    let base_plan: usize = admitted.iter().map(|id| planned_output_count(id)).sum();
    base_plan + sweep_reserved
}

/// How wide one streaming batch may be on this machine: what
/// [`VocabularyBudget`] affords at `budget_rows`, minus the resident plan.
///
/// A function of the hardware and the widest frame, never of a user parameter —
/// the never-OOM invariant. Zero means the machine cannot afford ANY streaming
/// extension, which the caller must treat as "do not stream", not as "stream a
/// batch of nothing".
pub fn streaming_batch_columns(budget_rows: usize) -> usize {
    VocabularyBudget::for_run(budget_rows)
        .max_columns
        .saturating_sub(planned_resident_columns(budget_rows))
}

/// Is this id's window drivable by the sweep, i.e. would sweeping it produce
/// genuinely DIFFERENT columns rather than five copies of one?
///
/// Three conditions, all of them load-bearing:
///
///   * it must be REGISTERED — an unregistered id's `period_plan` falls back to
///     the `"period"` naming convention, which the by-name dispatch arms may or
///     may not read. Sweeping on a guess is how a vocabulary fills with
///     duplicates wearing distinct names;
///   * its plan must be [`PeriodPlan::Key`], [`PeriodPlan::Ratio`], or
///     [`PeriodPlan::RegistryRatio`], never `NoWindow` — no-window indicators
///     such as OBV and VWAP are not sweepable by construction;
///   * it must not be on `EXPECTED_NON_PRODUCING`. Those ids cannot produce even
///     once; sweeping them would multiply a known, named failure by five. They
///     are still ATTEMPTED once per frame by the base pass, so the exclusion
///     table cannot rot behind this filter.
///
/// [`MULTI_PERIOD_IDS`] is excluded by the caller, not here, because those ids
/// are swept already and re-sweeping them would emit duplicate column NAMES —
/// a hard error in `compute_classic_ta_columns_sized`.
fn is_extended_sweepable(id: &'static str) -> bool {
    if expected_non_producing(id).is_some() {
        return false;
    }
    if vector_ta::indicators::registry::get_indicator(id).is_none() {
        return false;
    }
    matches!(
        period_plan(id),
        PeriodPlan::Key(_) | PeriodPlan::Ratio(_) | PeriodPlan::RegistryRatio(_)
    ) && !extended_sweep_periods(id).is_empty()
}

/// Which ids the EXTENDED period sweep can afford, in `ALL_INDICATORS` order.
///
/// Returns `(admitted, deferred)`. This is the same admission mechanism as
/// `feature_budget::admit_indicators`, applied to the (indicator, period) space
/// instead of the indicator space: a deterministic, repeatable prefix sized from
/// what the machine has left after the base vocabulary and the historical sweep
/// have been paid for.
///
/// `budget_columns` of zero admits nothing and defers everything — the emitted
/// column set is then exactly what it was before the extension existed, which is
/// the property that makes this safe to turn on: it can only ever spend memory
/// that was going to go unused.
///
/// Determinism matters beyond reproducibility here. This function is the
/// prototype of the streaming design's advance through the parameter space
/// (`docs/streaming-parameter-search.md`): a batch selector that never repeats
/// itself and never depends on the frame is what lets a later change swap the
/// working set between generations without the column layout becoming a
/// function of scheduling.
pub fn extended_sweep_plan(budget_columns: usize) -> (Vec<&'static str>, Vec<&'static str>) {
    let mut admitted: Vec<&'static str> = Vec::new();
    let mut deferred: Vec<&'static str> = Vec::new();
    let mut used = 0usize;
    for &id in ALL_INDICATORS {
        if MULTI_PERIOD_IDS.contains(&id) || !is_extended_sweepable(id) {
            continue;
        }
        let want = planned_output_count(id) * extended_sweep_periods(id).len();
        if used + want <= budget_columns {
            used += want;
            admitted.push(id);
        } else {
            deferred.push(id);
        }
    }
    (admitted, deferred)
}

// ─────────────────────────────────────────────────────────────────────────────
// THE ADVANCE — a streaming working set over the (indicator, period) space.
//
// WHAT THIS IS FOR, STATED HONESTLY, BECAUSE THE DESIGN DOC IS WRONG ABOUT IT.
//
// `docs/streaming-parameter-search.md` §1 justifies streaming by the cost of
// MATERIALISING the cube. Measured at HEAD, that justification does not hold:
// the production feature pass at 6,000 real EURUSD M5 bars with the budget
// allowing 4,096 columns takes **1.36 s** and emits 1,795 columns, and at
// 200,000 bars **26.76 s**. Time is linear in BARS and linear in COLUMNS
// COMPUTED; it is NOT a function of columns HELD. So per unit of (indicator,
// period) space explored, streaming costs exactly what materialising costs —
// and a streamed run re-pays the prefilter (5.6% of a run) once per batch.
//
// Streaming buys exactly two things, and neither is the feature build:
//
//   1. the MEMORY bound, which `VocabularyBudget` already gives; and
//   2. — the only real prize — never paying the DOWNSTREAM stages for a batch
//      that is knowably doomed. On the run the doc cites
//      (`docs/measurements/3090-47260276/card-run-valid.log`) the quality
//      screen was 88,971 ms = **50.4% of wall time** and took 174 candidates
//      in and 0 out. That is what the early-reject predicate in
//      `neoethos_search::discovery` exists to skip.
//
// Build the loop to save feature-build time and it will measure as a
// REGRESSION. Build it to skip the quality screen and it is worth 4-6x more
// (indicator, period) regions examined per hour on the same hardware.
//
// WHY `extended_sweep_plan` COULD NOT BE THE ADVANCE, despite its own doc
// comment claiming it was the prototype of one: it is a PREFIX selector that
// always restarts at index 0, it takes no offset, and it enumerates IDS (all
// every statically valid/distinct `ALT_PERIODS` point atomically) rather than
// (indicator, period) pairs, so a
// batch boundary could only ever fall between ids. The three functions below
// are the advance it described but was not.
// ─────────────────────────────────────────────────────────────────────────────

/// One point in the sweep space: an indicator and the window it is evaluated at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SweepPair {
    pub id: &'static str,
    pub period: usize,
}

/// A working set: the slice of the (indicator, period) space one batch covers.
///
/// TWO ORDERS, AND THEY ARE DELIBERATELY DIFFERENT.
///
/// * **Selection** walks period-OUTER / id-inner, so a batch mixes timescales
///   rather than being 32 flavours of one window. That is the order
///   `docs/streaming-parameter-search.md` §3.2 specifies and it is the one the
///   cursor indexes into.
/// * **Emission** is id-outer (`ALL_INDICATORS` order) / period-inner —
///   *whatever the batch selected*. This is not a detail: emission order feeds
///   `effective_feature_names` and every discovery artifact, and the parity
///   requirement is that a batch covering the WHOLE space emits a column list
///   byte-identical to today's non-streaming pass. Sorting the selected pairs
///   back into emission order is what makes that true by construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SweepBatch {
    /// Cursor this batch began at, in SELECTION order. One integer — this is
    /// the whole resumable state of a streaming run.
    pub cursor: usize,
    /// Cursor the next batch begins at. Always `> cursor` unless the space is
    /// exhausted or the budget admits nothing.
    pub next_cursor: usize,
    /// The batch's pairs, in EMISSION order.
    pub pairs: Vec<SweepPair>,
    /// Columns this batch plans to stage, from registry lookups only.
    pub planned_columns: usize,
    /// Total pairs in the space, so a log line can say "batch 7 of 1,620".
    pub space_len: usize,
    /// True when this batch reached the end of the space. The loop decides what
    /// that means (stop, or wrap to cursor 0 for a second pass); wrapping is
    /// never done here, because a silent wrap is a repeat and the whole point
    /// of the cursor is that a pair appears in exactly one batch per sweep.
    pub exhausted: bool,
}

impl SweepBatch {
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// True when this batch is the entire space starting from zero — the
    /// degenerate case the parity test pins.
    pub fn covers_whole_space(&self) -> bool {
        self.cursor == 0 && self.pairs.len() == self.space_len && self.space_len > 0
    }

    /// The batch regrouped for dispatch: `(id, periods)` in emission order,
    /// periods ascending within each id. This is the shape
    /// `sweep_one_id_ledgered` already takes, so the streaming lane and the
    /// prefix lane run through the SAME sweep code with the same naming, the
    /// same `#212` pre-flight guard and the same ledger.
    pub fn grouped_by_id(&self) -> Vec<(&'static str, Vec<usize>)> {
        let mut out: Vec<(&'static str, Vec<usize>)> = Vec::new();
        for pair in &self.pairs {
            match out.last_mut() {
                Some((id, periods)) if *id == pair.id => periods.push(pair.period),
                _ => out.push((pair.id, vec![pair.period])),
            }
        }
        out
    }
}

/// Position of an id in `ALL_INDICATORS`, for the emission sort. Linear scan
/// over a 342-entry static — called once per batch, not per column.
fn all_indicators_rank(id: &str) -> usize {
    ALL_INDICATORS
        .iter()
        .position(|&candidate| candidate == id)
        .unwrap_or(usize::MAX)
}

/// Every (indicator, period) pair the extended sweep can reach, in SELECTION
/// order (period-outer, id-inner).
///
/// Four properties, all of which the streaming loop needs and none of which are
/// free:
///
/// * **no repeats** — a pair appears exactly once, so batch *k* and batch *j*
///   are disjoint;
/// * **deterministic** — a pure function of `ALL_INDICATORS`, `ALT_PERIODS` and
///   the vector-ta registry, so batch *k* is the same set on every machine and
///   every run and a result is reproducible from `(seed, cursor)` alone;
/// * **frame-independent** — nothing here reads the frame, so every timeframe
///   in a run gets the SAME batch and the per-TF cube widths stay equal (which
///   is what `lib.rs::try_assemble_cube_in_ram`'s width invariant requires);
/// * **exclusion-stable** — `MULTI_PERIOD_IDS` are excluded exactly as
///   `extended_sweep_plan` excludes them, so a streamed column can never
///   collide by NAME with a historical sweep column (a collision is a hard
///   error in `compute_classic_ta_columns_sized`, by design).
pub fn extended_sweep_space() -> Vec<SweepPair> {
    let mut space = Vec::new();
    for &period in ALT_PERIODS.iter() {
        for &id in ALL_INDICATORS {
            if MULTI_PERIOD_IDS.contains(&id) || !is_extended_sweepable(id) {
                continue;
            }
            if sweep_point_is_distinct_and_valid(id, period) {
                space.push(SweepPair { id, period });
            }
        }
    }
    space
}

/// Size of the space `extended_sweep_space` enumerates.
pub fn extended_sweep_space_len() -> usize {
    extended_sweep_space().len()
}

/// Batch *k*: the pairs from `cursor` onward that fit `budget_columns`.
///
/// `budget_columns` comes from [`VocabularyBudget`] — free RAM divided by the
/// widest frame's per-column price, minus what the resident families and the
/// base vocabulary already claimed. It is never a constant and never a user
/// parameter, which is the never-OOM invariant applied rather than circumvented.
///
/// ONE PAIR IS ALWAYS TAKEN when the budget is non-zero and the cursor is
/// inside the space. Without that, an id whose planned output count alone
/// exceeds the budget would produce an empty non-advancing batch forever — a
/// loop that makes no progress and says nothing, which is the silent-stall
/// shape of the silent-drop defect. Taking it anyway over-spends by at most one
/// indicator's outputs and is reported by the budget log.
pub fn extended_sweep_batch(cursor: usize, budget_columns: usize) -> SweepBatch {
    let space = extended_sweep_space();
    let space_len = space.len();
    if cursor >= space_len || budget_columns == 0 {
        return SweepBatch {
            cursor,
            next_cursor: cursor,
            pairs: Vec::new(),
            planned_columns: 0,
            space_len,
            exhausted: cursor >= space_len,
        };
    }
    let mut selected: Vec<SweepPair> = Vec::new();
    let mut used = 0usize;
    let mut idx = cursor;
    while idx < space_len {
        let pair = space[idx];
        let want = planned_output_count(pair.id);
        if !selected.is_empty() && used + want > budget_columns {
            break;
        }
        used += want;
        selected.push(pair);
        idx += 1;
    }
    // Emission order: id-outer in `ALL_INDICATORS` order, period ascending
    // within each id. `sort_by_key` is stable, but the key is total so
    // stability is not load-bearing — the order is a function of the SET alone,
    // never of which cursor produced it.
    selected.sort_by_key(|pair| (all_indicators_rank(pair.id), pair.period));
    SweepBatch {
        cursor,
        next_cursor: idx,
        pairs: selected,
        planned_columns: used,
        space_len,
        exhausted: idx >= space_len,
    }
}

/// The working set installed for this process, or `None` for the historical
/// budget-prefix behaviour.
///
/// A process-level seam, exactly like [`set_indicator_compute_policy`] above and
/// for the same reason: the cube build reaches `compute_classic_ta_columns_sized`
/// through six frames of `rayon::join` in `lib.rs`, and threading a batch
/// parameter through all of them would touch call sites in crates this change
/// does not own. It is not ambient state in the dangerous sense — the batch is a
/// pure function of `(cursor, budget_columns)`, it is logged by cursor and width
/// on every pass, and `with_extended_sweep_working_set` in `lib.rs` scopes the
/// install so it cannot leak past the build it was made for.
static EXTENDED_SWEEP_WORKING_SET: std::sync::RwLock<Option<std::sync::Arc<SweepBatch>>> =
    std::sync::RwLock::new(None);

/// Install (or clear) the working set. Returns the PREVIOUS value so a scoped
/// helper can restore it — including when the build between install and restore
/// unwinds.
///
/// Lock poisoning is recovered rather than propagated: a poisoned lock here
/// means some other thread panicked while holding it, and refusing to read the
/// working set would silently change which columns the run builds. The value is
/// a plain `Option<Arc<_>>` with no invariant a panic could have broken.
pub fn install_extended_sweep_working_set(
    batch: Option<std::sync::Arc<SweepBatch>>,
) -> Option<std::sync::Arc<SweepBatch>> {
    let mut guard = EXTENDED_SWEEP_WORKING_SET
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    std::mem::replace(&mut *guard, batch)
}

/// The working set in force, or `None` when the pass should take the historical
/// budget-capped prefix.
pub fn current_extended_sweep_working_set() -> Option<std::sync::Arc<SweepBatch>> {
    EXTENDED_SWEEP_WORKING_SET
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

/// CPU sweep for ONE indicator across `periods`. This is the parity reference
/// — f64 end to end — and is the only implementation of the column naming and
/// the #212 pre-flight guard, so the GPU lane cannot drift from it by
/// accident.
// Only the real-card parity test calls the un-ledgered shape. Production has
// no per-indicator CPU fallback: `GpuOnly` is rejected before work until its
// complete resident graph is available, while `CpuOnly` aggregates one census
// for the whole sweep via `cpu_multi_period_all`.
#[cfg(all(feature = "gpu-cuda", test))]
fn cpu_multi_period_columns(
    candles: &Candles,
    ind_id: &str,
    periods: &[usize],
    n: usize,
    kernel: Kernel,
) -> Vec<(String, Vec<f64>)> {
    let (cols, ledger) = cpu_multi_period_columns_ledgered(
        candles,
        ind_id,
        periods,
        n,
        kernel,
        ClassicInputAvailability::all_present(),
    );
    // This wrapper exists so the GPU lane's per-indicator CPU fallback and the
    // parity test keep their original signature. It must still be incapable of
    // dropping silently, so anything the sweep discarded is reported here.
    if ledger.dropped_columns() > 0 {
        ledger.log_summary("multi-period-sweep", n);
    }
    cols
}

/// How a period maps onto ONE indicator's actual parameter names.
///
/// A period is not a detail of an indicator: RSI(4) is a scalping oscillator
/// and RSI(54) is a regime filter, and the search has to be able to reach both.
/// But "sweep everything at `period = P`" is wrong in a specific, measurable
/// way — vector-ta reads parameters BY KEY, and an indicator that does not
/// declare `period` simply ignores the key and returns its default series. Five
/// swept "periods" then produce five identical columns wearing legitimate
/// names, which the GA would happily fit as if they were distinct evidence.
///
/// So the sweep asks the registry what the indicator's window parameters
/// actually are:
///
///   * [`PeriodPlan::Key`] — the indicator declares one window key
///     (`period`, `length`, …). Sweep it directly. This covers rsi, ema, sma,
///     atr, adx, cci, willr, roc, mom, tsi, mfi AND the three multi-output ids
///     bollinger_bands / keltner / supertrend, which all declare `period` and
///     produced nothing only because their `output_id` was never supplied.
///   * [`PeriodPlan::Ratio`] — the indicator declares a coupled TUPLE of
///     windows whose RELATIVE sizes are its identity: macd is
///     (fast 12, slow 26, signal 9) and stoch is (fastk 14, slowk 3, slowd 3).
///     Setting all three to the same P would not be "macd at period P", it
///     would be a different indicator. The whole tuple is scaled by
///     `P / anchor`, preserving the shape, so macd at 7 vs at 200 really are
///     the fast and slow versions of the same phenomenon.
///   * [`PeriodPlan::NoWindow`] — the indicator declares no window at all
///     (OBV, VWAP). It is not sweepable: no synthetic compatibility parameter
///     is dispatched and no alias columns are added to the production schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PeriodPlan {
    Key(&'static str),
    Ratio(&'static [(&'static str, i64)]),
    /// A coupled window tuple read directly from vector-ta's registry. The
    /// largest declared default is the anchor and every member is scaled by
    /// the same rational factor, so adding a new registered indicator cannot
    /// silently fall back to an ignored synthetic `period` key.
    RegistryRatio(&'static str),
    NoWindow,
}

/// `(indicator, (param key, default value))` for the coupled-window
/// indicators, read from vector-ta's registry param tables. The FIRST entry is
/// the anchor the swept period replaces; the rest keep their ratio to it.
const COUPLED_WINDOWS: &[(&str, &[(&str, i64)])] = &[
    // registry.rs PARAM_MACD: fast 12, slow 26, signal 9. Anchored on `slow`,
    // the window that sets the indicator's timescale.
    (
        "macd",
        &[
            ("slow_period", 26),
            ("fast_period", 12),
            ("signal_period", 9),
        ],
    ),
    // registry.rs PARAM_STOCH: fastk 14, slowk 3, slowd 3. Anchored on fastk.
    (
        "stoch",
        &[
            ("fastk_period", 14),
            ("slowk_period", 3),
            ("slowd_period", 3),
        ],
    ),
    // registry.rs PARAM_TSI (line 3030): long_period 25, short_period 13. tsi
    // declares NEITHER `period` nor `length`, so it used to fall through to
    // `NoWindow` and its five swept "periods" were five bit-identical copies —
    // MEASURED on 60,000 real EURUSD M5 bars as the duplicate group
    // [tsi, tsi_7, tsi_21, tsi_50, tsi_100, tsi_200]. Anchored on long_period,
    // the window that sets the timescale, with short_period scaled to keep the
    // 25:13 ratio that IS the indicator.
    ("tsi", &[("long_period", 25), ("short_period", 13)]),
];

/// Window-parameter keys the sweep knows how to drive directly.
const RECOGNISED_WINDOW_KEYS: [&str; 4] = ["period", "length", "lookback_period", "lookback"];

/// Resolve the sweep plan for one indicator from vector-ta's declared params.
///
/// `NoWindow` means "this indicator declares no window-shaped parameter at
/// all". It must NEVER mean "it declares one under a name we do not recognise"
/// — that was the tsi defect, and it is silent by construction: the swept key
/// is simply ignored by the kernel and five identical columns come back wearing
/// five different names. So an unmatched window-shaped key is a WARN naming the
/// id and the key, not a shrug.
fn period_plan(ind_id: &'static str) -> PeriodPlan {
    if let Some((_, keys)) = COUPLED_WINDOWS.iter().find(|(k, _)| *k == ind_id) {
        return PeriodPlan::Ratio(keys);
    }
    if let Some(info) = vector_ta::indicators::registry::get_indicator(ind_id) {
        for key in RECOGNISED_WINDOW_KEYS {
            if info.params.iter().any(|p| p.key == key) {
                return PeriodPlan::Key(key);
            }
        }
        let unmatched = unmatched_window_keys(ind_id);
        if !unmatched.is_empty() {
            match registry_window_defaults(ind_id) {
                Ok(defaults) if !defaults.is_empty() => {
                    return PeriodPlan::RegistryRatio(ind_id);
                }
                Ok(_) => unreachable!("unmatched keys were non-empty"),
                Err(error) => {
                    tracing::warn!(
                        target: "neoethos_data::hpc_ta",
                        indicator = ind_id,
                        keys = %unmatched.join(","),
                        error,
                        "indicator has a window tuple that cannot be scaled from the vector-ta \
                         registry; refusing to invent defaults"
                    );
                }
            }
        }
        return PeriodPlan::NoWindow;
    }
    // Unregistered: `period` is the convention the by-name dispatch arms use.
    PeriodPlan::Key("period")
}

/// Integer parameters whose NAME looks like a window but which none of
/// [`RECOGNISED_WINDOW_KEYS`] matched — the tsi class.
///
/// Separated from [`period_plan`] so a test can assert the class is empty
/// across the whole swept set without capturing log output.
fn unmatched_window_keys(ind_id: &str) -> Vec<&'static str> {
    if COUPLED_WINDOWS.iter().any(|(k, _)| *k == ind_id) {
        return Vec::new();
    }
    let Some(info) = vector_ta::indicators::registry::get_indicator(ind_id) else {
        return Vec::new();
    };
    if info
        .params
        .iter()
        .any(|p| RECOGNISED_WINDOW_KEYS.contains(&p.key))
    {
        return Vec::new();
    }
    info.params
        .iter()
        .filter(|p| {
            matches!(
                p.kind,
                vector_ta::indicators::registry::IndicatorParamKind::Int
            ) && (p.key.contains("period") || p.key.contains("length"))
        })
        .map(|p| p.key)
        .collect()
}

/// Registry window keys paired with their authoritative positive integer
/// defaults. An incomplete tuple is an error: scaling only the members whose
/// defaults happened to be readable changes the indicator's identity.
fn registry_window_defaults(ind_id: &str) -> std::result::Result<Vec<(&'static str, i64)>, String> {
    use vector_ta::indicators::registry::ParamValueStatic;

    let info = vector_ta::indicators::registry::get_indicator(ind_id)
        .ok_or_else(|| format!("{ind_id} is absent from the vector-ta registry"))?;
    let keys = unmatched_window_keys(ind_id);
    let mut defaults = Vec::with_capacity(keys.len());
    for key in keys {
        let param = info
            .params
            .iter()
            .find(|param| param.key == key)
            .ok_or_else(|| format!("registry key {key} disappeared while resolving {ind_id}"))?;
        let default = match param.default {
            Some(ParamValueStatic::Int(value)) if value > 0 => value,
            Some(other) => {
                return Err(format!(
                    "{ind_id}.{key} needs a positive Int default, found {other:?}"
                ));
            }
            None => return Err(format!("{ind_id}.{key} has no registry default")),
        };
        defaults.push((key, default));
    }
    Ok(defaults)
}

fn scale_window_tuple(
    keys: &[(&'static str, i64)],
    anchor_default: i64,
    period: usize,
) -> Vec<ParamKV<'static>> {
    assert!(anchor_default > 0, "window anchor must be positive");
    let target = i128::try_from(period).expect("swept period exceeds i128");
    let anchor = i128::from(anchor_default);
    keys.iter()
        .map(|&(key, default)| {
            assert!(default > 0, "{key} default must be positive");
            // All operands are positive. Adding anchor/2 implements the same
            // half-up rounding as the previous f64 `.round()` expression,
            // without architecture-dependent floating conversion.
            let scaled = ((i128::from(default) * target + anchor / 2) / anchor).max(1);
            ParamKV {
                key,
                value: ParamValue::Int(
                    i64::try_from(scaled).expect("scaled window exceeds the dispatch i64 ABI"),
                ),
            }
        })
        .collect()
}

/// Build the parameter list for one indicator at one swept period.
fn sweep_params(plan: PeriodPlan, period: usize) -> Vec<ParamKV<'static>> {
    match plan {
        PeriodPlan::Key(key) => vec![ParamKV {
            key,
            value: ParamValue::Int(period as i64),
        }],
        PeriodPlan::Ratio(keys) => scale_window_tuple(keys, keys[0].1, period),
        PeriodPlan::RegistryRatio(ind_id) => {
            let keys = registry_window_defaults(ind_id).unwrap_or_else(|error| {
                panic!("period_plan admitted an invalid registry tuple for {ind_id}: {error}")
            });
            let anchor_default = keys
                .iter()
                .map(|(_, default)| *default)
                .max()
                .expect("RegistryRatio cannot contain an empty tuple");
            scale_window_tuple(&keys, anchor_default, period)
        }
        PeriodPlan::NoWindow => Vec::new(),
    }
}

/// CUDA's generic f64 ABI accepts one integer timescale.  Resolve the exact
/// anchor represented by the CPU request without inventing a default.
///
/// `None` is the base pass (`params: &[]`), so its anchor is read from the same
/// registry/default tuple [`period_plan`] uses.  A no-window formula receives
/// the inert value `1`; preflight separately proves that its registered kernel
/// is period-invariant before that value is allowed to launch.
#[cfg(feature = "gpu-cuda")]
pub(super) fn classic_cuda_period_anchor(
    indicator_id: &'static str,
    swept_period: Option<usize>,
) -> std::result::Result<usize, String> {
    if let Some(period) = swept_period {
        return (period > 0)
            .then_some(period)
            .ok_or_else(|| format!("{indicator_id}: swept period must be positive"));
    }

    use vector_ta::indicators::registry::ParamValueStatic;
    match period_plan(indicator_id) {
        PeriodPlan::Key(key) => {
            let info =
                vector_ta::indicators::registry::get_indicator(indicator_id).ok_or_else(|| {
                    format!(
                        "{indicator_id}: base CPU dispatch is unregistered, so its `{key}` default \
                         cannot be proven from the canonical registry"
                    )
                })?;
            let param = info
                .params
                .iter()
                .find(|param| param.key == key)
                .ok_or_else(|| format!("{indicator_id}: registry no longer declares `{key}`"))?;
            match param.default {
                Some(ParamValueStatic::Int(value)) if value > 0 => usize::try_from(value)
                    .map_err(|_| format!("{indicator_id}.{key}: default {value} exceeds usize")),
                other => Err(format!(
                    "{indicator_id}.{key}: expected a positive integer default, found {other:?}"
                )),
            }
        }
        PeriodPlan::Ratio(keys) => usize::try_from(keys[0].1).map_err(|_| {
            format!(
                "{indicator_id}.{}: default {} exceeds usize",
                keys[0].0, keys[0].1
            )
        }),
        PeriodPlan::RegistryRatio(id) => registry_window_defaults(id)?
            .into_iter()
            .map(|(_, default)| default)
            .max()
            .ok_or_else(|| format!("{indicator_id}: registry window tuple is empty"))
            .and_then(|default| {
                usize::try_from(default)
                    .map_err(|_| format!("{indicator_id}: default {default} exceeds usize"))
            }),
        PeriodPlan::NoWindow => Ok(1),
    }
}

#[cfg(feature = "gpu-cuda")]
pub(super) fn classic_cuda_base_has_no_window(indicator_id: &'static str) -> bool {
    matches!(period_plan(indicator_id), PeriodPlan::NoWindow)
}

/// Exact integer overrides the CPU sweep sends for an admitted point.  The
/// planner records them in its gap manifest so a CUDA route cannot claim only
/// the column name while silently interpreting a different coupled tuple.
#[cfg(feature = "gpu-cuda")]
pub(super) fn classic_cuda_sweep_params(
    indicator_id: &'static str,
    period: usize,
) -> std::result::Result<Vec<(&'static str, i64)>, String> {
    sweep_params(period_plan(indicator_id), period)
        .into_iter()
        .map(|param| match param.value {
            ParamValue::Int(value) => Ok((param.key, value)),
            other => Err(format!(
                "{indicator_id}.{}: CUDA sweep needs an integer override, found {other:?}",
                param.key
            )),
        })
        .collect()
}

/// Formula-proven parameter points that must not enter the production sweep.
///
/// These are static properties of the implementation, never conclusions drawn
/// from one market frame. Keeping the reason beside the point makes saturation
/// and integer-scaling collisions reviewable instead of silently shaving the
/// schema at runtime.
pub const SWEEP_POINT_EXCLUSIONS: &[(&str, usize, &str)] = &[
    (
        "cycle_channel_oscillator",
        7,
        "ratio scaling makes the internal short delay zero, so fast and slow both read the undelayed source",
    ),
    (
        "ehlers_itrend",
        100,
        "the implementation clamps its adaptive MESA period to 50, making period 100 identical to period 50",
    ),
    (
        "ehlers_itrend",
        200,
        "the implementation clamps its adaptive MESA period to 50, making period 200 identical to period 50",
    ),
];

/// Formula-level reason why an extended sweep point is deliberately absent.
pub fn sweep_point_exclusion(indicator_id: &str, period: usize) -> Option<&'static str> {
    SWEEP_POINT_EXCLUSIONS
        .iter()
        .find(|(candidate_id, candidate_period, _)| {
            *candidate_id == indicator_id && *candidate_period == period
        })
        .map(|(_, _, reason)| *reason)
}

/// Whether one extended-sweep point is both accepted by the registry contract
/// and semantically different from the base vocabulary's default call.
///
/// This is deliberately static: the emitted feature schema may depend on the
/// indicator registry and requested parameter point, never on the values in a
/// particular market frame. It prevents three expensive false features:
///
/// * a point equal to every overridden registry default (the base pass already
///   computed it);
/// * an integer outside its declared min/max bounds;
/// * a coupled tuple whose rounding reverses or collapses a strict ordering
///   present in the authoritative defaults (for example short < long);
/// * a formula-proven saturation or integer-scaling collision named in
///   [`SWEEP_POINT_EXCLUSIONS`].
fn sweep_point_is_distinct_and_valid(ind_id: &'static str, period: usize) -> bool {
    use std::cmp::Ordering;
    use vector_ta::indicators::registry::{IndicatorParamKind, ParamValueStatic};

    if sweep_point_exclusion(ind_id, period).is_some() {
        return false;
    }

    let Some(info) = vector_ta::indicators::registry::get_indicator(ind_id) else {
        return false;
    };
    let params = sweep_params(period_plan(ind_id), period);
    if params.is_empty() {
        return false;
    }

    let mut generated_with_defaults = Vec::with_capacity(params.len());
    let mut all_at_default = true;
    for param in &params {
        let ParamValue::Int(value) = param.value else {
            return false;
        };
        let Some(declared) = info
            .params
            .iter()
            .find(|candidate| candidate.key == param.key)
        else {
            return false;
        };
        if !matches!(declared.kind, IndicatorParamKind::Int) {
            return false;
        }
        if declared.min.is_some_and(|minimum| (value as f64) < minimum)
            || declared.max.is_some_and(|maximum| (value as f64) > maximum)
        {
            return false;
        }
        let Some(ParamValueStatic::Int(default)) = declared.default else {
            return false;
        };
        all_at_default &= value == default;
        generated_with_defaults.push((default, value));
    }
    if all_at_default {
        return false;
    }

    for left in 0..generated_with_defaults.len() {
        for right in (left + 1)..generated_with_defaults.len() {
            let (left_default, left_value) = generated_with_defaults[left];
            let (right_default, right_value) = generated_with_defaults[right];
            match left_default.cmp(&right_default) {
                Ordering::Less if left_value >= right_value => return false,
                Ordering::Greater if left_value <= right_value => return false,
                _ => {}
            }
        }
    }
    true
}

/// Extended points for one id after static contract validation and removal of
/// the base-default duplicate. Every caller uses this exact list, so memory
/// planning, streaming selection, and actual dispatch cannot disagree.
fn extended_sweep_periods(ind_id: &'static str) -> Vec<usize> {
    ALT_PERIODS
        .iter()
        .copied()
        .filter(|period| sweep_point_is_distinct_and_valid(ind_id, *period))
        .collect()
}

/// The sweep, with its outcomes counted.
///
/// Every discard the old body made silently is now a reason:
///   * the `#212` pre-flight skip is `PreflightWarmup` (was a bare `continue`);
///   * a dispatch `Err` is its own variant (was `if let Ok(...)` with no else —
///     the branch that swallowed the five multi-output ids' `output_id`
///     complaint 25 times per frame);
///   * a wrong-length series is `ShortSeries` (was the `_ => {}` catch-all).
fn cpu_multi_period_columns_ledgered(
    candles: &Candles,
    ind_id: &str,
    periods: &[usize],
    n: usize,
    kernel: Kernel,
    input_availability: ClassicInputAvailability,
) -> (Vec<(String, Vec<f64>)>, IndicatorLedger) {
    let mut out: Vec<(String, Vec<f64>)> = Vec::new();
    let mut ledger = IndicatorLedger::new();
    // `dispatch_indicator_outputs` needs a 'static id, which every caller has:
    // the sweep is driven by `MULTI_PERIOD_IDS` and the GPU spec table, both
    // `&'static str` tables.
    let Some(static_id) = MULTI_PERIOD_IDS.iter().copied().find(|s| *s == ind_id) else {
        // Not one of the swept ids — the caller is asking for something this
        // function was never given a table entry for. Loud, not silent.
        ledger.dropped(
            "unknown-sweep-id",
            ind_id,
            DropReason::UnknownIndicator,
            "id is not in hpc_ta::MULTI_PERIOD_IDS, so the sweep has no plan for it",
        );
        return (out, ledger);
    };
    let _ = &mut out;
    let _ = &mut ledger;
    sweep_one_id_ledgered(candles, static_id, periods, n, kernel, input_availability)
}

/// Sweep ONE `'static` indicator id across `periods`.
///
/// Shared by the historical sixteen (through
/// [`cpu_multi_period_columns_ledgered`], which keeps the id-table guard the
/// GPU lane relies on) and by the extended sweep, so the two cannot drift on
/// naming, on the `#212` pre-flight guard, or on what they record.
fn sweep_one_id_ledgered(
    candles: &Candles,
    static_id: &'static str,
    periods: &[usize],
    n: usize,
    kernel: Kernel,
    input_availability: ClassicInputAvailability,
) -> (Vec<(String, Vec<f64>)>, IndicatorLedger) {
    let mut out: Vec<(String, Vec<f64>)> = Vec::new();
    let mut ledger = IndicatorLedger::new();
    let plan = period_plan(static_id);
    for &period in periods {
        // #212: pre-flight check — if the period is larger than the data
        // length, vector-ta's `warm_prefix` exceeds the row width and the
        // kernel panics at `helpers.rs:159` instead of returning Err. Skip the
        // call entirely for these cases. The 1.25× safety margin matches the
        // kernel's typical `first_valid_idx + period` formula plus a small
        // headroom for indicators with extra warmup beyond the period itself.
        if (period as f64) * 1.25 >= n as f64 {
            // THE COLUMN SET MUST NOT DEPEND ON THE FRAME.
            //
            // This guard skipped the period entirely, so a short frame emitted
            // FEWER columns than a long one — and every timeframe has a
            // different length. That made the emitted width a function of the
            // frame, which is precisely what `lib.rs::try_assemble_cube_in_ram`
            // refuses to assemble, so a run would fall through to the slower
            // streaming disk path with nothing but a debug line to say why.
            // This becomes especially damaging once the extended sweep reaches
            // hundreds of points. Caught by
            // `cube_assembly_tests::ram_and_disk_cubes_are_identical`.
            //
            // So the columns are still EMITTED, at full frame length, filled
            // with NaN — the same value the warmup prefix of every windowed
            // indicator already carries, and the honest one here: this frame
            // cannot support this period, so the reading does not exist. It is
            // NOT zero, which would be a real number the GA could threshold
            // against. The skip is still counted under `PreflightWarmup`, and
            // the column shows up in the degenerate census as carrying no
            // information.
            let column = format!("{static_id}_{period}");
            ledger.dropped(
                static_id,
                &column,
                DropReason::PreflightWarmup,
                format!(
                    "period {period} * 1.25 >= {n} bars (#212 pre-flight guard); column emitted \
                     as all-NaN to keep the column set independent of the frame length"
                ),
            );
            push_absent_columns(static_id, &column, n, &mut out);
            continue;
        }
        let params = sweep_params(plan, period);
        dispatch_indicator_outputs(
            candles,
            static_id,
            &format!("{static_id}_{period}"),
            &params,
            n,
            kernel,
            input_availability,
            &mut out,
            &mut ledger,
        );
    }
    (out, ledger)
}

/// Pure-CPU multi-period sweep across all of [`MULTI_PERIOD_IDS`], parallel
/// across indicators. Column order is by position in `MULTI_PERIOD_IDS` —
/// `flat_map_iter` + `collect` on an indexed `par_iter` preserves it.
fn cpu_multi_period_all(
    candles: &Candles,
    n: usize,
    input_availability: ClassicInputAvailability,
    indicator_ids: &[&'static str],
) -> (Vec<(String, Vec<f64>)>, IndicatorLedger) {
    let per_id: Vec<(Vec<(String, Vec<f64>)>, IndicatorLedger)> = indicator_ids
        .par_iter()
        .map(|&ind_id| {
            cpu_multi_period_columns_ledgered(
                candles,
                ind_id,
                &ALT_PERIODS,
                n,
                Kernel::Auto,
                input_availability,
            )
        })
        .collect();
    let mut cols = Vec::new();
    let mut ledger = IndicatorLedger::new();
    for (mut c, l) in per_id {
        cols.append(&mut c);
        ledger.merge(l);
    }
    // ONE summary for the whole sweep rather than sixteen — but never zero.
    ledger.log_summary("multi-period-sweep", n);
    (cols, ledger)
}

// --- exclusive lane selection ---------------------------------------------
//
// This helper is CPU-only. `GpuOnly` returns through `classic_cuda_plan` before
// Candles are allocated and materializes f64 only after every admitted output
// has passed the exact CUDA preflight. It may never enter this helper.
fn compute_multi_period_columns(
    ohlcv: &Ohlcv,
    candles: &Candles,
    n: usize,
    policy: IndicatorComputePolicy,
    indicator_ids: &[&'static str],
) -> (Vec<(String, Vec<f64>)>, IndicatorLedger) {
    use crate::core::indicator_telemetry::{IndicatorLane, IndicatorRunSummary, record};
    use std::time::Instant;

    let lane = match policy {
        IndicatorComputePolicy::CpuOnly => IndicatorLane::CpuByPolicy,
        IndicatorComputePolicy::Auto => IndicatorLane::CpuNoFeature,
        IndicatorComputePolicy::GpuOnly => {
            unreachable!("GpuOnly must return through the exact CUDA executor before this helper")
        }
    };
    let started = Instant::now();
    let (columns, ledger) = cpu_multi_period_all(
        candles,
        n,
        ClassicInputAvailability::from_ohlcv(ohlcv),
        indicator_ids,
    );
    record(IndicatorRunSummary {
        gpu_indicators: Vec::new(),
        cpu_indicators: indicator_ids.iter().map(|id| (*id, lane.clone())).collect(),
        cpu_time: started.elapsed(),
        ..Default::default()
    });
    (columns, ledger)
}

/// One series returned by `compute_single_indicator` — multi-output
/// indicators (Bollinger Bands, MACD, Stochastic) decompose into
/// several of these.
#[derive(Debug, Clone)]
pub struct IndicatorLine {
    /// Human-readable line name. Single-output indicators use the
    /// indicator id (e.g. `"sma"`); multi-output ones suffix with
    /// the column index (`"bollinger_bands_line0"`, `"…_line1"`,
    /// `"…_line2"` for lower/middle/upper).
    pub name: String,
    /// Series aligned with the input ohlcv length. NaN-padding at
    /// the start is preserved (so e.g. SMA(20)[0..19] = NaN), which
    /// the chart renders as a gap before the line begins.
    pub values: Vec<f64>,
}

/// Compute a single indicator on demand — the interactive Chart
/// screen calls this through the `/indicators` HTTP endpoint
/// whenever the user adds an indicator to the overlay. Cheap enough
/// to recompute on every pan; vector_ta dispatches to CPU SIMD or
/// GPU kernels under the hood.
///
/// `params` is a key→f64 map. Conventional keys per indicator:
///   * `sma`/`ema`/`rsi`/`atr`/`adx`: `period`
///   * `bollinger_bands`: `period`, `std_dev`
///   * `macd`: `fast`, `slow`, `signal`
///   * `stoch`: `k_period`, `k_slow`, `d_period`
/// Unrecognised keys are silently ignored. Empty map = library defaults.
///
/// Returns the row count + lines on success, anyhow error if the
/// indicator id is unknown or the kernel rejects the input.
pub fn compute_single_indicator(
    ohlcv: &Ohlcv,
    indicator_id: &str,
    params: &std::collections::HashMap<String, f64>,
) -> anyhow::Result<Vec<IndicatorLine>> {
    let n = ohlcv.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Pack the ohlcv slice into a Candles instance for the dispatch
    // API. Timestamps and volume are nice-to-have but not required
    // by most indicators — we fill zeros when missing.
    let timestamps = ohlcv.timestamp.clone().unwrap_or_else(|| vec![0i64; n]);
    let volume = ohlcv.volume.clone().unwrap_or_else(|| vec![0.0; n]);
    let candles = Candles::new(
        timestamps,
        ohlcv.open.clone(),
        ohlcv.high.clone(),
        ohlcv.low.clone(),
        ohlcv.close.clone(),
        volume,
    );
    let data_ref = IndicatorDataRef::Candles {
        candles: &candles,
        source: None,
    };

    // Translate the f64 param map into vector_ta's ParamKV array.
    // vector_ta accepts ints for period-like params and floats for
    // multipliers (e.g. Bollinger Bands' std_dev); we route based on
    // whether the value has a fractional part.
    let mut kv: Vec<vector_ta::indicators::dispatch::ParamKV> = Vec::with_capacity(params.len());
    for (k, v) in params {
        // Leak the &'static str via Box::leak so the dispatch API
        // can hold a 'static reference. Param map is tiny (≤ 5
        // entries) and lives for the call, so the leak is bounded
        // by the call site — acceptable trade-off for the simpler
        // wire shape.
        let key: &'static str = Box::leak(k.clone().into_boxed_str());
        let value = if v.fract() == 0.0 && v.abs() <= i64::MAX as f64 {
            vector_ta::indicators::dispatch::ParamValue::Int(*v as i64)
        } else {
            vector_ta::indicators::dispatch::ParamValue::Float(*v)
        };
        kv.push(vector_ta::indicators::dispatch::ParamKV { key, value });
    }

    // Look up the indicator's declared outputs. Multi-output indicators
    // (MACD, Bollinger Bands, Stochastic, …) REQUIRE an explicit
    // `output_id` per series in vector_ta — dispatching them with
    // `output_id: None` fails with "output_id is required for
    // multi-output indicators". Single-output indicators use `None`
    // (the library's default output).
    let output_ids: Vec<Option<&'static str>> = vector_ta::indicators::registry::list_indicators()
        .iter()
        .find(|i| i.id == indicator_id)
        .map(|info| {
            if info.outputs.len() <= 1 {
                vec![None]
            } else {
                info.outputs.iter().map(|o| Some(o.id)).collect()
            }
        })
        .unwrap_or_else(|| vec![None]);

    let mut lines = Vec::with_capacity(output_ids.len());
    for out_id in output_ids {
        let req = IndicatorComputeRequest {
            indicator_id,
            output_id: out_id,
            data: data_ref,
            params: &kv,
            kernel: Kernel::Auto,
        };
        let output = compute_cpu(req).map_err(|e| {
            anyhow::anyhow!("vector_ta dispatch failed ({indicator_id}/{out_id:?}): {e:?}")
        })?;
        let (values, raw_len) = flatten_indicator_series(output.series, n)?;
        if raw_len > n {
            tracing::warn!(
                target: "neoethos_data::hpc_ta",
                indicator = indicator_id,
                output = ?out_id,
                raw_len,
                bars = n,
                "chart indicator returned more values than the frame has bars; the head was kept"
            );
        }
        // Single-output → bare indicator id (e.g. "sma"); multi-output →
        // "<id>_<output>" (e.g. "macd_signal") so the chart legend can
        // split on '_' and show the per-line label.
        let name = match out_id {
            Some(id) => format!("{indicator_id}_{id}"),
            None => indicator_id.to_string(),
        };
        lines.push(IndicatorLine { name, values });
    }

    Ok(lines)
}

/// Flatten a vector_ta series for ONE output into exactly `n` values.
/// vector_ta reports a 1-D series as rows=1 × cols=n, so we key off the
/// value count, not the rows/cols metadata — the previous `rows == n`
/// assumption rejected every single-output series with a "shape
/// mismatch" error.
///
/// Returns `(values, raw_value_count)`. The raw count is handed back so the
/// caller can LEDGER a truncation instead of performing one silently — this
/// function used to be the last uncounted discard on the repaired path.
fn flatten_indicator_series(
    series: IndicatorSeries,
    n: usize,
) -> anyhow::Result<(Vec<f64>, usize)> {
    match series {
        IndicatorSeries::F64(v) => normalize_indicator_len(v, n),
        IndicatorSeries::I32(v) => {
            normalize_indicator_len(v.into_iter().map(|x| x as f64).collect(), n)
        }
        IndicatorSeries::Bool(v) => normalize_indicator_len(
            v.into_iter().map(|x| if x { 1.0 } else { 0.0 }).collect(),
            n,
        ),
    }
}

/// Fit one output's raw values onto the frame, or refuse.
///
/// Three outcomes, no fourth:
///
///   * exactly `n` — the normal case;
///   * an exact MULTIPLE of `n` — that is a FLATTENED MULTI-SERIES, not a long
///     single series. Taking the head would return the first output's values
///     under whichever `output_id` was requested: silent MIS-ATTRIBUTION, which
///     is worse than a silent loss because the column looks legitimate and
///     carries the wrong meaning. Hard error, named;
///   * longer for any other reason — head-aligned truncation, which is real and
///     is reported back to the caller so it lands in the ledger under
///     [`DropReason::Truncated`] rather than happening quietly.
fn normalize_indicator_len(v: Vec<f64>, n: usize) -> anyhow::Result<(Vec<f64>, usize)> {
    let raw = v.len();
    if raw == n {
        Ok((v, raw))
    } else if raw > n {
        if n > 0 && raw % n == 0 {
            anyhow::bail!(
                "indicator returned {raw} values for a {n}-bar frame — an exact {}x multiple, \
                 i.e. a FLATTENED MULTI-SERIES. Taking the head would silently return one \
                 output's values under another output's name; there is no correct silent \
                 handling, so this is refused.",
                raw / n
            );
        }
        // Bar-aligned from the start; take the leading n (warmup padding
        // lives at the head and stays aligned with candle index 0).
        Ok((v.into_iter().take(n).collect(), raw))
    } else {
        anyhow::bail!("indicator returned {} values, expected ≥{}", raw, n)
    }
}

#[cfg(test)]
mod streaming_advance_tests {
    use super::*;
    use std::collections::HashSet;

    /// PARITY, AND IT COMES FIRST.
    ///
    /// The degenerate case: one batch whose working set is the WHOLE space,
    /// starting at cursor 0. Its emitted `(id, periods)` grouping must be
    /// exactly what the non-streaming pass builds — the budget-prefix plan with
    /// every id at every statically valid, non-default `ALT_PERIODS` point, in
    /// `ALL_INDICATORS` order.
    ///
    /// Asserted on the (id, period) LIST, not on a count: the extension emits
    /// `<id>_<period>` names into the same namespace as the base pass and a
    /// duplicate NAME is a hard error in `compute_classic_ta_columns_sized`, so
    /// a width-only assertion would pass while the column set changed.
    #[test]
    fn whole_space_batch_is_byte_identical_to_the_non_streaming_plan() {
        let space_len = extended_sweep_space_len();
        assert!(space_len > 0, "the sweep space must not be empty");
        let batch = extended_sweep_batch(0, usize::MAX);
        assert!(batch.covers_whole_space());
        assert!(batch.exhausted);
        assert_eq!(batch.next_cursor, space_len);

        // What the non-streaming lane would dispatch, at an unbounded budget.
        let (plan, deferred) = extended_sweep_plan(usize::MAX);
        assert!(deferred.is_empty(), "an unbounded budget defers nothing");
        let expected: Vec<(&'static str, Vec<usize>)> = plan
            .into_iter()
            .map(|id| (id, extended_sweep_periods(id)))
            .collect();

        assert_eq!(
            batch.grouped_by_id(),
            expected,
            "the whole-space batch must dispatch exactly the ids, periods and ORDER the \
             non-streaming pass dispatches — column order feeds effective_feature_names"
        );
    }

    /// The advance never repeats and never loses a pair: the disjoint union of
    /// every batch is exactly the space, in selection order.
    #[test]
    fn successive_batches_partition_the_space_exactly_once() {
        let space = extended_sweep_space();
        let mut cursor = 0usize;
        let mut seen: Vec<SweepPair> = Vec::new();
        let mut guard = 0usize;
        loop {
            let batch = extended_sweep_batch(cursor, 40);
            if batch.is_empty() {
                break;
            }
            assert!(
                batch.next_cursor > cursor,
                "a non-empty batch must advance the cursor, or the loop stalls silently"
            );
            seen.extend(batch.pairs.iter().copied());
            cursor = batch.next_cursor;
            guard += 1;
            assert!(guard < 100_000, "advance failed to terminate");
            if batch.exhausted {
                break;
            }
        }
        assert_eq!(cursor, space.len());
        assert_eq!(
            seen.len(),
            space.len(),
            "every pair must appear exactly once"
        );
        let unique: HashSet<SweepPair> = seen.iter().copied().collect();
        assert_eq!(unique.len(), space.len(), "no pair may appear twice");
        let space_set: HashSet<SweepPair> = space.iter().copied().collect();
        assert_eq!(unique, space_set, "the union of the batches IS the space");
    }

    /// Batch *k* is the same set on every call — the property that makes a
    /// result reproducible from `(seed, cursor)` alone.
    #[test]
    fn a_batch_is_a_pure_function_of_cursor_and_budget() {
        for cursor in [0usize, 1, 37, 200] {
            let a = extended_sweep_batch(cursor, 64);
            let b = extended_sweep_batch(cursor, 64);
            assert_eq!(a, b, "batch at cursor {cursor} is not deterministic");
        }
    }

    /// Selection is period-outer so a batch MIXES timescales; emission is
    /// id-outer so the column layout never becomes a function of scheduling.
    #[test]
    fn selection_mixes_timescales_and_emission_is_id_ordered() {
        let space = extended_sweep_space();
        // Selection: the first `ALT_PERIODS[0]`-period run precedes any
        // `ALT_PERIODS[1]` pair.
        let first_period = space[0].period;
        assert_eq!(first_period, ALT_PERIODS[0]);
        // Emission within one batch: non-decreasing ALL_INDICATORS rank, and
        // ascending period within an id.
        let batch = extended_sweep_batch(0, 512);
        let mut last = (0usize, 0usize);
        for pair in &batch.pairs {
            let key = (all_indicators_rank(pair.id), pair.period);
            assert!(key > last || last == (0, 0), "emission order is not sorted");
            last = key;
        }
    }

    /// A budget smaller than one id's outputs must still advance — a batch that
    /// makes no progress is a stall with no log line, which is the silent-drop
    /// defect one level up.
    #[test]
    fn a_budget_too_small_for_one_indicator_still_advances() {
        let batch = extended_sweep_batch(0, 1);
        assert!(!batch.is_empty());
        assert_eq!(batch.next_cursor, 1);
    }

    /// A zero budget admits nothing and does NOT advance, and an exhausted
    /// cursor reports itself rather than wrapping. Wrapping is the loop's
    /// decision, never a silent one here.
    #[test]
    fn zero_budget_and_exhausted_cursor_are_reported_not_papered_over() {
        let empty = extended_sweep_batch(0, 0);
        assert!(empty.is_empty());
        assert_eq!(empty.next_cursor, 0);
        assert!(!empty.exhausted);

        let len = extended_sweep_space_len();
        let past_end = extended_sweep_batch(len, usize::MAX);
        assert!(past_end.is_empty());
        assert!(past_end.exhausted);
        assert_eq!(past_end.next_cursor, len);
    }

    /// The space can never collide by NAME with the historical sweep, because a
    /// duplicate column name is a hard error in the pass that emits both.
    #[test]
    fn the_space_excludes_the_historical_sweep_ids() {
        for pair in extended_sweep_space() {
            assert!(
                !MULTI_PERIOD_IDS.contains(&pair.id),
                "{} is swept by the historical sweep; streaming it too emits duplicate \
                 column NAMES, which is a hard error",
                pair.id
            );
        }
    }

    /// Install/restore is exact, so a scoped build cannot leak its working set
    /// into the next one.
    #[test]
    fn installing_a_working_set_returns_the_previous_one() {
        let batch = std::sync::Arc::new(extended_sweep_batch(0, 8));
        let previous = install_extended_sweep_working_set(Some(batch.clone()));
        assert_eq!(
            current_extended_sweep_working_set().as_deref(),
            Some(&*batch)
        );
        let restored = install_extended_sweep_working_set(previous);
        assert_eq!(restored.as_deref(), Some(&*batch));
        assert!(current_extended_sweep_working_set().is_none());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn semantic_pattern_columns(
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> std::collections::BTreeMap<String, Vec<f64>> {
        let output = compute_cpu(IndicatorComputeRequest {
            indicator_id: "pattern_recognition",
            output_id: Some("matrix"),
            data: IndicatorDataRef::Ohlc {
                open,
                high,
                low,
                close,
            },
            params: &[],
            kernel: Kernel::Scalar,
        })
        .expect("native vector-ta CPU pattern dispatch must succeed");
        assert!(
            matches!(&output.series, IndicatorSeries::I32(_)),
            "production pattern dispatch must preserve signed magnitude"
        );
        pattern_matrix_columns(&output, "pattern_recognition", close.len())
            .expect("hpc decomposition must accept the signed pattern matrix")
            .into_iter()
            .collect()
    }

    fn baseline_pattern_ohlc() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        (
            vec![100.0; 192],
            vec![102.0; 192],
            vec![99.0; 192],
            vec![101.0; 192],
        )
    }

    #[test]
    fn semantic_uniqueness_cpu_dispatch_to_hpc_preserves_pattern_magnitude_and_sign() {
        let (mut open, mut high, mut low, mut close) = baseline_pattern_ohlc();
        open[190] = 105.0;
        high[190] = 105.05;
        low[190] = 100.95;
        close[190] = 101.0;
        open[191] = 106.0;
        high[191] = 109.05;
        low[191] = 105.95;
        close[191] = 109.0;
        let kicking = semantic_pattern_columns(&open, &high, &low, &close);
        assert_eq!(kicking["pattern_recognition_cdlkicking"][191], 100.0);
        assert_eq!(
            kicking["pattern_recognition_cdlkickingbylength"][191],
            -100.0
        );

        // A shared endpoint is strength 80. Run both first-candle directions
        // so the complete native {-100,-80,0,80,100} magnitude contract is
        // exercised through dispatch and hpc decomposition, not just directly.
        let (mut open, mut high, mut low, mut close) = baseline_pattern_ohlc();
        open[190] = 100.0;
        high[190] = 104.5;
        low[190] = 99.5;
        close[190] = 104.0;
        open[191] = 104.0;
        high[191] = 104.2;
        low[191] = 103.3;
        close[191] = 103.5;
        let bearish = semantic_pattern_columns(&open, &high, &low, &close);
        assert_eq!(bearish["pattern_recognition_cdlharami"][191], -80.0);
        assert_eq!(bearish["pattern_recognition_cdlharami"][0], 0.0);

        open[190] = 104.0;
        close[190] = 100.0;
        open[191] = 100.0;
        high[191] = 100.7;
        low[191] = 99.8;
        close[191] = 100.5;
        let bullish = semantic_pattern_columns(&open, &high, &low, &close);
        assert_eq!(bullish["pattern_recognition_cdlharami"][191], 80.0);
    }

    // #212: pre-flight check helper used by the validation harness and
    // gene admission gate to refuse computation on slices smaller than
    // the indicator's warmup. These assertions document the contract
    // and trap regressions if `alt_periods` ever changes.
    #[test]
    fn max_indicator_warmup_returns_zero_for_tiny_frames() {
        assert_eq!(max_indicator_warmup(0), 0);
        assert_eq!(max_indicator_warmup(5), 0);
        assert_eq!(max_indicator_warmup(7), 0);
    }

    #[test]
    fn max_indicator_warmup_returns_largest_fitting_period() {
        // n=8 fits 7 (smallest alt_period) but not 21.
        assert_eq!(max_indicator_warmup(8), 7);
        assert_eq!(max_indicator_warmup(22), 21);
        assert_eq!(max_indicator_warmup(51), 50);
        assert_eq!(max_indicator_warmup(101), 100);
        assert_eq!(max_indicator_warmup(201), 200);
    }

    #[test]
    fn max_indicator_warmup_caps_at_largest_period() {
        // Even for huge frames the helper does not exceed the
        // configured `MAX_MULTI_PERIOD_LOOKBACK`.
        assert_eq!(max_indicator_warmup(10_000), 200);
        assert_eq!(max_indicator_warmup(1_000_000), MAX_MULTI_PERIOD_LOOKBACK);
    }

    // Pre-flight gate documented in `compute_classic_ta_columns`: the
    // multi-period sweep skips any period whose `*1.25` safety margin
    // exceeds the frame length. This test pins the contract so a refactor
    // can't silently drop the guard and reintroduce the #212 panic.
    #[test]
    fn pre_flight_gate_skips_periods_larger_than_frame() {
        // For a 30-row frame: 7 fits (7*1.25=8.75 < 30), 21 fits
        // (26.25 < 30), 50 does NOT fit (62.5 ≥ 30).
        let n: usize = 30;
        let acceptable: Vec<usize> = [7usize, 21, 50, 100, 200]
            .into_iter()
            .filter(|p| (*p as f64) * 1.25 < n as f64)
            .collect();
        assert_eq!(acceptable, vec![7, 21]);
    }

    /// The two constants that used to be inline literals must still describe
    /// the same sweep, otherwise `max_indicator_warmup` and the GPU lane's
    /// skip guard disagree with what is actually computed.
    #[test]
    fn alt_periods_matches_the_warmup_helper() {
        assert_eq!(ALT_PERIODS, [7, 21, 50, 100, 200]);
        assert_eq!(
            *ALT_PERIODS.last().unwrap(),
            MAX_MULTI_PERIOD_LOOKBACK,
            "MAX_MULTI_PERIOD_LOOKBACK must be the largest swept period"
        );
        assert_eq!(MULTI_PERIOD_IDS.len(), 16);
        assert!(!MULTI_PERIOD_IDS.contains(&"obv"));
        assert!(!MULTI_PERIOD_IDS.contains(&"vwap"));
    }

    /// Automatic selection currently resolves the whole request to CpuOnly
    /// because the complete GPU graph is not promoted. It must therefore
    /// preserve the exact CpuOnly schema rather than running a partial CUDA
    /// sweep and reassembling host columns.
    #[test]
    fn lane_policy_does_not_change_column_names_or_order() {
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let cpu = compute_classic_ta_columns_with_policy(&ohlcv, IndicatorComputePolicy::CpuOnly)
            .unwrap();
        let auto =
            compute_classic_ta_columns_with_policy(&ohlcv, IndicatorComputePolicy::Auto).unwrap();
        let cpu_names: Vec<&str> = cpu.iter().map(|(n, _)| n.as_str()).collect();
        let auto_names: Vec<&str> = auto.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            cpu_names, auto_names,
            "the lane policy changed the feature-frame column layout — every stored artifact \
            would be invalidated"
        );
    }

    /// Strict GPU selection is a whole-graph execution boundary. It must
    /// reject the still-incomplete resident feature plan before the CPU base
    /// pass starts, never execute the old CPU -> partial CUDA -> CPU route.
    #[test]
    fn strict_gpu_policy_rejects_the_incomplete_graph_before_work() {
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let result =
            compute_classic_ta_columns_with_policy(&ohlcv, IndicatorComputePolicy::GpuOnly);
        let error = result.expect_err("strict GPU mode must reject an incomplete resident graph");
        let message = format!("{error:#}");
        eprintln!("NEOETHOS_GPU_ONLY_REJECTION={message}");
        assert!(
            message.contains("GpuOnly preflight rejected before any CPU or CUDA work"),
            "unexpected strict-GPU rejection: {message}"
        );
        #[cfg(feature = "gpu-cuda")]
        assert!(
            message.contains("classic-TA output route(s) are incomplete")
                && message.contains("No CPU segment")
                && message.contains("missing_"),
            "unexpected strict-GPU rejection: {message}"
        );
        #[cfg(not(feature = "gpu-cuda"))]
        assert!(
            message.contains("built without the gpu-cuda feature")
                && message.contains("never substitutes CpuOnly"),
            "unexpected strict-GPU rejection: {message}"
        );
    }

    #[test]
    fn execution_report_accounts_for_the_exact_production_admission() {
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let run =
            compute_classic_ta_columns_with_policy_report(&ohlcv, IndicatorComputePolicy::CpuOnly)
                .unwrap();
        let report = &run.report;

        assert_eq!(report.produced_columns, run.columns.len());
        assert_eq!(report.budget_rows, ohlcv.len());
        assert_eq!(
            report.admitted_indicator_ids.len() + report.budget_deferred_indicator_ids.len(),
            ALL_INDICATORS.len(),
            "every base indicator must be classified by the execution that actually ran"
        );
        assert_eq!(
            report.planned_base_columns,
            ALL_INDICATORS
                .iter()
                .map(|id| planned_output_count(id))
                .sum::<usize>()
        );
        assert_eq!(
            report.admitted_base_columns,
            report
                .admitted_indicator_ids
                .iter()
                .map(|id| planned_output_count(id))
                .sum::<usize>()
        );
        assert_eq!(
            report.extended_planned_columns,
            report
                .extended_admitted_indicator_ids
                .iter()
                .map(|id| planned_output_count(id) * extended_sweep_periods(id).len())
                .sum::<usize>()
        );
        assert!(
            report
                .admitted_indicator_ids
                .iter()
                .all(|id| !report.budget_deferred_indicator_ids.contains(id)),
            "an indicator cannot be both admitted and deferred"
        );
        assert_eq!(
            run.ledger.duplicate_count(),
            report.produced_columns - {
                let unique = run
                    .columns
                    .iter()
                    .map(|(_, values)| series_fingerprint(values))
                    .collect::<HashSet<_>>();
                unique.len()
            }
        );
    }

    // =======================================================================
    // The 341-silent-drop regression guards.
    //
    // These run on the embedded ~100-bar cTrader fixture, which is BELOW
    // `VOCABULARY_FLOOR_MIN_ROWS`, so the hard floor does not fire here — most
    // indicators legitimately fail their warmup at 100 bars. What these assert
    // is the SHAPE of the fix, which is length-independent: that the vocabulary
    // is no longer one column, that multi-output indicators are dispatched with
    // an output id, and that every column is full length.
    // =======================================================================

    /// The headline regression: the base pass used to yield exactly ONE column
    /// (`ttm_trend`) out of 342 ids. Anything close to that again is the bug
    /// returning.
    #[test]
    fn the_base_vocabulary_is_no_longer_a_single_column() {
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let cols = compute_classic_ta_columns_with_policy(&ohlcv, IndicatorComputePolicy::CpuOnly)
            .unwrap();
        let names: Vec<&str> = cols.iter().map(|(n, _)| n.as_str()).collect();
        // The sweep alone contributes 65 on a long frame, but at ~100 bars only
        // the 7/21/50 periods clear the #212 pre-flight guard. The base pass is
        // what this asserts: it must contribute far more than the single
        // `ttm_trend` it produced for sixteen months.
        let swept: Vec<&&str> = names
            .iter()
            .filter(|n| ALT_PERIODS.iter().any(|p| n.contains(&format!("_{p}"))))
            .collect();
        let base_columns = names.len() - swept.len();
        assert!(
            base_columns > 100,
            "the base ALL_INDICATORS pass produced {base_columns} columns — it used to produce \
             exactly 1 (ttm_trend) and the repaired pass produces hundreds. Column sample: {:?}",
            &names[..names.len().min(12)]
        );
    }

    /// `output_id: None` against a multi-output indicator is the cause of 92 of
    /// the 342 failures, and is precisely why the five multi-output ids in the
    /// period sweep produced nothing. Bollinger Bands must now emit its
    /// declared lines, named by output rather than by position.
    #[test]
    fn multi_output_indicators_now_emit_one_named_column_per_output() {
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let cols = compute_classic_ta_columns_with_policy(&ohlcv, IndicatorComputePolicy::CpuOnly)
            .unwrap();
        let bb: Vec<&String> = cols
            .iter()
            .map(|(n, _)| n)
            .filter(|n| n.starts_with("bollinger_bands"))
            .collect();
        assert!(
            bb.len() >= 3,
            "bollinger_bands is multi-output and must produce one column per declared output; \
             got {bb:?}"
        );
        assert!(
            !bb.iter().any(|n| n.contains("_line")),
            "columns must be named by OUTPUT ID, not by position: {bb:?}"
        );
    }

    /// A short column would be zero-padded by the cube copy in `lib.rs`, and a
    /// padded zero is indistinguishable from a real reading. Nothing here may
    /// emit one.
    #[test]
    fn every_emitted_column_is_exactly_frame_length() {
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let n = ohlcv.len();
        let cols = compute_classic_ta_columns_with_policy(&ohlcv, IndicatorComputePolicy::CpuOnly)
            .unwrap();
        assert!(!cols.is_empty());
        for (name, v) in &cols {
            assert_eq!(
                v.len(),
                n,
                "column '{name}' is {} values, expected {n}",
                v.len()
            );
        }
    }

    /// THE COLUMN SET MUST NOT DEPEND ON THE FRAME LENGTH.
    ///
    /// Every timeframe in a run is a different length, and
    /// `lib.rs::try_assemble_cube_in_ram` refuses to assemble a cube whose
    /// per-timeframe widths differ — it then falls through to the slower
    /// streaming disk path, with a debug line as the only symptom. Two
    /// mechanisms made the width frame-dependent and both were live: the `#212`
    /// warmup pre-flight guard (structurally — it reads `n`), and any dispatch
    /// failure that is a property of the data rather than of the indicator.
    /// Both now emit an all-NaN column under the right name instead of nothing.
    ///
    /// 60 bars vs the full fixture is chosen so the pre-flight guard fires on
    /// one side and not the other (period 50 needs 62.5 bars).
    #[test]
    fn the_column_set_is_independent_of_the_frame_length() {
        let full = crate::test_fixtures::ctrader_sample_ohlcv();
        assert!(
            full.len() > 60,
            "fixture too short to truncate meaningfully"
        );
        let short = Ohlcv {
            timestamp: full.timestamp.as_ref().map(|t| t[..60].to_vec()),
            open: full.open[..60].to_vec(),
            high: full.high[..60].to_vec(),
            low: full.low[..60].to_vec(),
            close: full.close[..60].to_vec(),
            volume: full.volume.as_ref().map(|v| v[..60].to_vec()),
        };
        let a =
            compute_classic_ta_columns_with_policy(&full, IndicatorComputePolicy::CpuOnly).unwrap();
        // Sized against the LONGER frame, exactly as a multi-timeframe build
        // does — the budget must not vary per timeframe either.
        let b =
            compute_classic_ta_columns_sized(&short, IndicatorComputePolicy::CpuOnly, full.len())
                .unwrap();
        let names_a: Vec<&str> = a.iter().map(|(n, _)| n.as_str()).collect();
        let names_b: Vec<&str> = b.iter().map(|(n, _)| n.as_str()).collect();
        let set_a: HashSet<&str> = names_a.iter().copied().collect();
        let set_b: HashSet<&str> = names_b.iter().copied().collect();
        let mut only_full: Vec<&str> = set_a.difference(&set_b).copied().collect();
        let mut only_short: Vec<&str> = set_b.difference(&set_a).copied().collect();
        only_full.sort_unstable();
        only_short.sort_unstable();
        assert_eq!(
            names_a.len(),
            names_b.len(),
            "a {}-bar frame emitted {} columns and a 60-bar frame emitted {} — the cube's width \
             invariant would refuse to assemble and the run would silently take the disk path; \
             only_full={only_full:?}; only_short={only_short:?}",
            full.len(),
            names_a.len(),
            names_b.len()
        );
        assert_eq!(
            names_a, names_b,
            "column names/order differ between frame lengths"
        );
        for (name, v) in &b {
            assert_eq!(v.len(), 60, "column '{name}' is not frame length");
        }
    }

    /// Column names must be unique — a duplicate name silently shadows a real
    /// feature wherever the cube is projected by name.
    #[test]
    fn column_names_are_unique() {
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let cols = compute_classic_ta_columns_with_policy(&ohlcv, IndicatorComputePolicy::CpuOnly)
            .unwrap();
        let mut seen = std::collections::HashSet::new();
        for (name, _) in &cols {
            assert!(seen.insert(name.clone()), "duplicate column name '{name}'");
        }
    }

    /// The sweep must ask each indicator for the window keys it actually
    /// declares. Setting `period` on MACD would be ignored and produce five
    /// identical columns wearing five different names.
    #[test]
    fn the_period_sweep_uses_each_indicators_real_window_parameters() {
        assert_eq!(period_plan("rsi"), PeriodPlan::Key("period"));
        // MACD declares fast/slow/signal, never `period`.
        match period_plan("macd") {
            PeriodPlan::Ratio(keys) => {
                assert_eq!(keys[0].0, "slow_period");
                let p = sweep_params(PeriodPlan::Ratio(keys), 52);
                // Anchored on slow=26, so 52 doubles the whole tuple:
                // fast 12->24, signal 9->18. The SHAPE is preserved, which is
                // what makes "macd at 52" a real indicator rather than noise.
                let get = |k: &str| {
                    p.iter()
                        .find(|kv| kv.key == k)
                        .map(|kv| match kv.value {
                            ParamValue::Int(i) => i,
                            _ => panic!("period params must be Int"),
                        })
                        .unwrap_or_else(|| panic!("missing {k}"))
                };
                assert_eq!(get("slow_period"), 52);
                assert_eq!(get("fast_period"), 24);
                assert_eq!(get("signal_period"), 18);
            }
            other => panic!("macd must use a coupled-window plan, got {other:?}"),
        }
        // stoch is the other coupled one.
        assert!(matches!(period_plan("stoch"), PeriodPlan::Ratio(_)));
        assert_eq!(
            period_plan("alligator"),
            PeriodPlan::RegistryRatio("alligator")
        );
        // OBV declares no window. It stays as one base feature and a sweep may
        // not invent ignored parameters or compatibility aliases for it.
        assert_eq!(period_plan("obv"), PeriodPlan::NoWindow);
        assert!(sweep_params(PeriodPlan::NoWindow, 21).is_empty());
    }

    /// A registry-declared integer window must never fall through to
    /// `NoWindow`: that makes vector-ta ignore the synthetic `period` key and
    /// emits several bit-identical features under different names.
    #[test]
    fn every_windowed_extended_indicator_has_a_scalable_period_plan() {
        let unresolved = ALL_INDICATORS
            .iter()
            .copied()
            .filter(|id| !MULTI_PERIOD_IDS.contains(id))
            .filter(|id| !unmatched_window_keys(id).is_empty())
            .filter(|id| matches!(period_plan(id), PeriodPlan::NoWindow))
            .map(|id| (id, unmatched_window_keys(id)))
            .collect::<Vec<_>>();
        assert!(
            unresolved.is_empty(),
            "windowed indicators still routed through NoWindow: {unresolved:#?}"
        );
    }

    /// The vector-ta registry is the source of truth for automatically scaled
    /// tuples. At the anchor default the generated params must reproduce every
    /// registry default exactly; at every search period they remain positive
    /// and complete.
    #[test]
    fn registry_window_tuple_scaling_preserves_authoritative_defaults() {
        use vector_ta::indicators::registry::ParamValueStatic;

        for &id in ALL_INDICATORS {
            let keys = unmatched_window_keys(id);
            if keys.is_empty() {
                continue;
            }
            let defaults = registry_window_defaults(id)
                .unwrap_or_else(|error| panic!("invalid registry tuple for {id}: {error}"));
            assert_eq!(defaults.len(), keys.len(), "{id}: incomplete window tuple");
            let anchor = defaults
                .iter()
                .map(|(_, default)| *default)
                .max()
                .expect("non-empty tuple has an anchor");
            let at_default = sweep_params(
                PeriodPlan::RegistryRatio(id),
                usize::try_from(anchor).expect("positive default fits usize"),
            );
            assert_eq!(at_default.len(), defaults.len(), "{id}: param count drift");
            for &(key, default) in &defaults {
                let generated = at_default
                    .iter()
                    .find(|param| param.key == key)
                    .unwrap_or_else(|| panic!("{id}: generated tuple omitted {key}"));
                assert_eq!(
                    generated.value,
                    ParamValue::Int(default),
                    "{id}.{key}: anchor scaling changed the registry default"
                );

                let info = vector_ta::indicators::registry::get_indicator(id)
                    .expect("ALL_INDICATORS entry must be registered");
                let declared = info
                    .params
                    .iter()
                    .find(|param| param.key == key)
                    .expect("key came from this registry entry");
                assert_eq!(declared.default, Some(ParamValueStatic::Int(default)));
                if let Some(minimum) = declared.min {
                    assert!(
                        (default as f64) >= minimum,
                        "{id}.{key}: default {default} < registry min {minimum}"
                    );
                }
                if let Some(maximum) = declared.max {
                    assert!(
                        (default as f64) <= maximum,
                        "{id}.{key}: default {default} > registry max {maximum}"
                    );
                }
            }
            for &period in &ALT_PERIODS {
                let generated = sweep_params(PeriodPlan::RegistryRatio(id), period);
                assert_eq!(generated.len(), defaults.len(), "{id}@{period}");
                for param in generated {
                    match param.value {
                        ParamValue::Int(value) => {
                            assert!(value >= 1, "{id}.{} scaled to {value}", param.key)
                        }
                        _ => panic!("{id}.{} did not generate an Int", param.key),
                    }
                }
            }
        }
    }

    /// The base vocabulary already evaluates every indicator at its registry
    /// defaults. Re-emitting that exact parameter tuple under a `_50`/`_100`
    /// suffix is dead work and a false extra feature. Likewise, a point that
    /// violates the registry's declared bounds must be rejected by the static
    /// sweep plan, not attempted later and converted into an all-NaN column.
    #[test]
    fn extended_space_excludes_default_equivalent_and_invalid_parameter_points() {
        let space = extended_sweep_space();

        assert!(
            !space.contains(&SweepPair {
                id: "atr_percentile",
                period: 50,
            }),
            "atr_percentile@50 reproduces its exact registry-default tuple"
        );
        assert!(
            !space.contains(&SweepPair {
                id: "halftrend",
                period: 100,
            }),
            "halftrend@100 reproduces its exact registry-default tuple"
        );
        assert!(
            !space.contains(&SweepPair {
                id: "geometric_bias_oscillator",
                period: 7,
            }),
            "geometric_bias_oscillator.length has registry minimum 10"
        );
        assert!(
            !space.contains(&SweepPair {
                id: "ehlers_autocorrelation_periodogram",
                period: 7,
            }),
            "the scaled min_period at anchor 7 falls below its registry minimum"
        );

        for pair in [
            SweepPair {
                id: "atr_percentile",
                period: 21,
            },
            SweepPair {
                id: "halftrend",
                period: 50,
            },
            SweepPair {
                id: "geometric_bias_oscillator",
                period: 21,
            },
            SweepPair {
                id: "ehlers_autocorrelation_periodogram",
                period: 21,
            },
        ] {
            assert!(space.contains(&pair), "valid point was lost: {pair:?}");
        }
    }

    #[test]
    fn formula_proven_sweep_collisions_are_named_unique_and_absent() {
        let mut keys = std::collections::BTreeSet::new();
        for (id, period, reason) in SWEEP_POINT_EXCLUSIONS {
            assert!(ALL_INDICATORS.contains(id), "unknown excluded id {id}");
            assert!(ALT_PERIODS.contains(period), "{id}@{period} is not swept");
            assert!(
                keys.insert((*id, *period)),
                "duplicate sweep exclusion for {id}@{period}"
            );
            assert!(
                reason.len() > 20,
                "{id}@{period} has no formula-level exclusion reason"
            );
            assert_eq!(
                sweep_point_exclusion(id, *period),
                Some(*reason),
                "exclusion lookup drifted"
            );
            assert!(
                !sweep_point_is_distinct_and_valid(id, *period),
                "{id}@{period} remains dispatchable despite its formula collision"
            );
        }

        for (id, period) in [
            ("cycle_channel_oscillator", 21),
            ("ehlers_itrend", 50),
            ("half_causal_estimator", 21),
            ("half_causal_estimator", 50),
        ] {
            assert!(
                sweep_point_is_distinct_and_valid(id, period),
                "neighbouring valid point was over-excluded: {id}@{period}"
            );
        }
        let hce_21 = sweep_params(PeriodPlan::RegistryRatio("half_causal_estimator"), 21);
        let hce_21_value = |key| {
            hce_21
                .iter()
                .find(|param| param.key == key)
                .map(|param| match param.value {
                    ParamValue::Int(value) => value,
                    _ => panic!("HCE RegistryRatio values must remain integers"),
                })
                .unwrap_or_else(|| panic!("HCE@21 omitted {key}"))
        };
        assert_eq!(hce_21_value("data_period"), 5);
        assert_eq!(hce_21_value("filter_length"), 21);

        assert_eq!(
            period_plan("ehlers_pma"),
            PeriodPlan::NoWindow,
            "Ehlers PMA has no period parameter in its formula"
        );
        assert!(
            !is_extended_sweepable("ehlers_pma"),
            "a parameterless formula must not enter the extended period sweep"
        );
    }

    /// The ratio scaling must never produce a zero or negative window.
    #[test]
    fn coupled_window_scaling_never_underflows_to_zero() {
        for (_, keys) in COUPLED_WINDOWS {
            for &period in &ALT_PERIODS {
                for kv in sweep_params(PeriodPlan::Ratio(keys), period) {
                    match kv.value {
                        ParamValue::Int(i) => {
                            assert!(i >= 1, "{} scaled to {i} at period {period}", kv.key)
                        }
                        _ => panic!("expected Int"),
                    }
                }
            }
        }
    }

    /// The floor must be reachable: it is measured against the real recovery
    /// (324 ids / ~674 base columns), not set above what the library can give.
    #[test]
    fn the_vocabulary_floor_is_below_the_measured_recovery() {
        assert!(MIN_PRODUCING_INDICATOR_IDS < ALL_INDICATORS.len());
        assert!(
            MIN_PRODUCING_INDICATOR_IDS >= 200,
            "a floor low enough to pass the 1-column state would not be a floor at all"
        );
        assert!(MIN_BASE_VOCABULARY_COLUMNS >= 400);
    }

    // =======================================================================
    // Task #22 — CPU-vs-CUDA parity for the multi-period indicator sweep.
    //
    // Built only `--features gpu-cuda`, i.e. only on a box with the CUDA
    // toolkit. Runs against the REAL embedded EURUSD M1 sample, not a
    // synthetic ramp: `crates/neoethos-data/test_fixtures/eurusd_m1_100bars.json`
    // is captured cTrader data, so the price scale, the tick granularity and
    // the high/low geometry are the ones the kernels will actually see.
    //
    // ── THE TOLERANCE, AND WHY IT IS NOW NEARLY ZERO ──────────────────────
    //
    // This block used to argue for a LOOSE tolerance, because bit equality was
    // impossible for two reasons that were both true at the time:
    //
    //   1. vector-ta's device layer was f32 end to end
    //      (`CudaDeviceVectorF32`, `upload_f32`,
    //      `IndicatorCudaSeries::HostF32`) while every CPU indicator returns
    //      f64 — ~7 significant decimal digits against ~16; and
    //   2. vector-ta's build.rs passed `--use_fast_math` unless
    //      `CUDA_FAST_MATH=0` was exported, enabling flush-to-zero and
    //      approximate div/sqrt/rcp.
    //
    // BOTH ARE FIXED (2026-08-09):
    //
    //   1. `device_types_f64.rs` adds the f64 device vocabulary,
    //      `upload_ohlcv_f64` uploads without narrowing, and
    //      `kernels/cuda/neoethos_f64_kernels.cu` holds a real f64 kernel for
    //      EVERY id in `GPU_SWEEP_SPECS` — asserted, not counted, by
    //      `gpu_indicators::tests::every_spec_has_an_f64_kernel_with_a_matching_input_contract`;
    //   2. that file is listed in vector-ta build.rs's `F64_LANE_SOURCES`, so
    //      it is compiled with `-prec-div=true -prec-sqrt=true -fmad=false
    //      -ftz=false` and NEVER with `--use_fast_math` — the flag is no
    //      longer something an operator can get wrong.
    //
    // The kernels were written against this crate's own f64 CPU
    // implementations, preserving the accumulation ORDER (one thread per
    // column for everything recursive) and the exact `fma` placement (CUDA
    // `fma()` wherever the CPU uses `f64::mul_add`, plain `a*b+c` wherever it
    // does not, with `-fmad=false` stopping the compiler from fusing behind
    // our back). So the target is BIT EQUALITY, and the remaining tolerance is
    // a small safety margin against the parts of that claim that only a real
    // card can settle:
    //
    //   * whether nvcc reassociated anything we did not intend;
    //   * whether the device's `fma`, division and `fabs` agree with the
    //     host's on every input in the fixture (they should: all are
    //     IEEE-754 correctly rounded in double, and `-prec-div` forces the
    //     correctly-rounded divide).
    //
    // 1e-12 relative is roughly 4500 ulp at double precision — wide enough
    // that a single unintended reassociation still passes and gets REPORTED by
    // the printed worst-case, narrow enough that an f32 lane sneaking back in
    // (1.19e-7) fails instantly, which is the regression this guards.
    //
    // WORKFLOW: run this on the card, read the printed per-indicator worst
    // case, and if it is 0.0 across the board, tighten to exact equality.
    // Until then these numbers are a bound, not a measurement.
    //
    // What is NOT tolerated, at all:
    //   * a different NaN mask — the warmup boundary is structural, and a
    //     shifted warmup means the two lanes are computing different windows;
    //   * a different column set, name or order;
    //   * an infinity on either side.
    // =======================================================================

    /// `(absolute floor, relative ceiling)` for one indicator id.
    ///
    /// Uniform now: the lane is f64 on both sides, so there is no longer a
    /// reason for the recursive indicators to get a looser budget than the
    /// windowed ones. If the card shows otherwise, split it again WITH the
    /// measured numbers rather than with an error model.
    #[cfg(feature = "gpu-cuda")]
    fn parity_tolerance(_id: &str) -> (f64, f64) {
        // Absolute floor for values legitimately near zero (`mom`, `roc` and
        // the CCI numerator all cross zero), then a relative ceiling that is
        // five orders of magnitude tighter than f32 epsilon.
        (1e-12, 1e-12)
    }

    #[cfg(feature = "gpu-cuda")]
    #[test]
    fn gpu_cpu_indicator_sweep_parity() {
        use crate::core::gpu_indicators::{GPU_SWEEP_SPECS, GpuIndicatorEngine};

        let require_gpu = std::env::var("NEOETHOS_REQUIRE_GPU")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false);

        // REAL data. 100 bars clears the 7/21/50 periods under the 1.25x
        // pre-flight guard; 100 and 200 launch on neither lane but remain
        // present as all-NaN schema placeholders. The full-length check is the
        // CLI feature build on the box.
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let n = ohlcv.len();
        assert!(n >= 64, "fixture too short to sweep anything: {n} bars");

        let candles = {
            let timestamps = ohlcv.timestamp.clone().unwrap_or_else(|| vec![0i64; n]);
            let volume = ohlcv.volume.clone().unwrap_or_else(|| vec![0.0; n]);
            Candles::new(
                timestamps,
                ohlcv.open.clone(),
                ohlcv.high.clone(),
                ohlcv.low.clone(),
                ohlcv.close.clone(),
                volume,
            )
        };

        let engine = match GpuIndicatorEngine::new(&ohlcv, 0) {
            Ok(e) => e,
            Err(e) => {
                if require_gpu {
                    panic!("NEOETHOS_REQUIRE_GPU set but the CUDA lane would not open: {e:?}");
                }
                eprintln!(
                    "gpu_cpu_indicator_sweep_parity: SKIPPED — no usable CUDA lane: {e:?}\n\
                     (set NEOETHOS_REQUIRE_GPU=1 to make this a hard failure)"
                );
                return;
            }
        };

        eprintln!(
            "parity vs {} ({}), native cubin archs {:?}, source {}, precision {}, \
             {n} real EURUSD M1 bars",
            engine.device_name(),
            engine.device_arch(),
            vector_ta::cuda::module_loader::COMPILED_ARCHS,
            crate::core::indicator_telemetry::VECTOR_TA_ARCH_SOURCE,
            engine.precision(),
        );

        let mut worst_overall = 0.0f64;
        let mut mismatch_count = 0usize;
        let mut mismatch_examples = Vec::new();
        const MAX_MISMATCH_EXAMPLES: usize = 256;

        let mut record_mismatch = |message: String| {
            mismatch_count += 1;
            if mismatch_examples.len() < MAX_MISMATCH_EXAMPLES {
                mismatch_examples.push(message);
            }
        };
        for spec in GPU_SWEEP_SPECS {
            let gpu = engine
                .sweep_columns(spec, &ALT_PERIODS)
                .unwrap_or_else(|e| panic!("device sweep failed for {}: {e:?}", spec.id));
            // `Kernel::Scalar`, NOT `Kernel::Auto`, and this is load-bearing.
            //
            // Every f64 kernel in `neoethos_f64_kernels.cu` was written against
            // the crate's `*_scalar` implementation, bar for bar, in its exact
            // accumulation order. `neoethos-data` enables vector-ta's
            // `nightly-avx`, so on x86_64 `Auto` resolves to Avx2/Avx512 for
            // several of these ids — a DIFFERENT function body from the one the
            // kernel was transcribed from, even where the two agree today.
            //
            // They do agree today, and that is measured rather than assumed:
            // `tests/f64_lane_cpu_reference.rs` runs both kernels over every
            // claimed id on clean AND gapped bars and demands BIT equality. It
            // used to fail for two ids — `vwap` at index 136 and `wilders` at
            // its seed bar, 1 ULP each — and both were withheld from
            // `F64_KERNELS` over it. vector-ta has since fixed BOTH at the
            // source (`vwap_row_scalar_pv` deleted; the wilders warm-up seed
            // association unified on the 4-wide scalar tree), so
            // `WITHHELD_PENDING_CPU_SELF_CONSISTENCY` is `&[]` and `vwap` is
            // now claimed by `GPU_SWEEP_SPECS`.
            //
            // Pinning Scalar here SURVIVES that fix and is still the right
            // oracle: it is the reference the kernels were written against, so
            // a future AVX reassociation fails the card-less test by name
            // instead of failing this one against a CORRECT kernel and burning
            // a rented card on a phantom. The sweep that runs in PRODUCTION
            // still uses `Auto`; only the oracle pins Scalar.
            let cpu = cpu_multi_period_columns(&candles, spec.id, &ALT_PERIODS, n, Kernel::Scalar);

            // Structural equality first — a mismatch here is a bug, never a
            // tolerance question.
            let gpu_names: Vec<&str> = gpu.iter().map(|(k, _)| k.as_str()).collect();
            let cpu_names: Vec<&str> = cpu.iter().map(|(k, _)| k.as_str()).collect();
            if gpu_names != cpu_names {
                record_mismatch(format!(
                    "{}: column set/name/order differs between lanes: gpu={gpu_names:?} cpu={cpu_names:?}",
                    spec.id
                ));
                continue;
            }

            let (abs_tol, rel_tol) = parity_tolerance(spec.id);
            let mut worst = 0.0f64;
            let mut worst_at = String::new();

            for ((name, gcol), (_, ccol)) in gpu.iter().zip(cpu.iter()) {
                if gcol.len() != ccol.len() {
                    record_mismatch(format!(
                        "{name}: length differs between lanes: gpu={} cpu={}",
                        gcol.len(),
                        ccol.len()
                    ));
                    continue;
                }
                for (j, (&g, &c)) in gcol.iter().zip(ccol.iter()).enumerate() {
                    if g.is_nan() != c.is_nan() {
                        record_mismatch(format!(
                            "{name}[{j}]: NaN mask differs (gpu={g} cpu={c}) — the warmup boundary \
                             is structural, so the two lanes are computing different windows"
                        ));
                        continue;
                    }
                    if c.is_nan() {
                        continue;
                    }
                    if !g.is_finite() {
                        record_mismatch(format!(
                            "{name}[{j}]: device produced {g} where the CPU produced a finite {c}"
                        ));
                        continue;
                    }
                    let delta = (g - c).abs();
                    let allowed = abs_tol + rel_tol * c.abs();
                    // Track the normalised overshoot so "worst" is comparable
                    // across indicators with different price scales.
                    let ratio = delta / allowed.max(f64::MIN_POSITIVE);
                    if ratio > worst {
                        worst = ratio;
                        worst_at = format!("{name}[{j}] gpu={g} cpu={c} |d|={delta}");
                    }
                    if delta > allowed {
                        record_mismatch(format!(
                            "{name}[{j}]: |gpu-cpu| = {delta} > {allowed} \
                             (abs_tol={abs_tol}, rel_tol={rel_tol}) — gpu={g} cpu={c}"
                        ));
                    }
                }
            }
            worst_overall = worst_overall.max(worst);
            eprintln!(
                "  {:<6} worst = {:.4} of budget (abs {:.0e} + rel {:.0e})   {}",
                spec.id, worst, abs_tol, rel_tol, worst_at
            );
        }

        engine
            .synchronize()
            .expect("synchronize after parity sweep");
        assert!(
            mismatch_count == 0,
            "GPU/CPU f64 parity found {mismatch_count} mismatches; first {}:\n{}",
            mismatch_examples.len(),
            mismatch_examples.join("\n")
        );
        eprintln!(
            "parity worst across all indicators: {worst_overall:.4} of budget. \
             0.0 means the two lanes are BIT-IDENTICAL on this frame. That is now a legitimate \
             target rather than a trap, because the CPU side of this comparison is pinned to \
             Kernel::Scalar — the reference every kernel was written against — instead of \
             Kernel::Auto, which resolves to AVX and is not bit-identical to Scalar for every \
             indicator in this crate. Anything above 0 names the ONE cell where a rounding \
             differs and must be explained before acceptance; do not widen the tolerance to \
             make it pass."
        );
    }

    /// The engine must never hand back a working-looking handle on a device
    /// whose arch cannot load the kernels — and when it refuses, the message
    /// must name BOTH arches so the operator can act without reading source.
    #[cfg(feature = "gpu-cuda")]
    #[test]
    fn arch_mismatch_error_names_both_arches() {
        use crate::core::gpu_indicators::GpuIndicatorEngine;
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        match GpuIndicatorEngine::new(&ohlcv, 0) {
            Ok(engine) => {
                // A lane that opened has already proven a real module load and
                // kernel launch, so the arches are compatible by construction.
                // Assert the reported strings are populated, not placeholders.
                assert!(engine.device_arch().starts_with("sm_"));
                assert!(
                    !crate::core::indicator_telemetry::VECTOR_TA_NATIVE_ARCHS.is_empty(),
                    "a live GPU lane must report at least one exact native cubin architecture"
                );
            }
            Err(e) => {
                let msg = format!("{e:#}");
                assert!(
                    msg.contains("compute capability") || msg.contains("cuda_available"),
                    "refusal message must name the device state, got: {msg}"
                );
            }
        }
    }
}
