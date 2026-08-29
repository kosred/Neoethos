use anyhow::{Context, Result};
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, warn};

// ── Training cancellation (process-global) ──────────────────────────────────
// Training runs one-at-a-time (engines_control refuses a concurrent job), so a
// single global flag is safe. The app installs it before a run and clears it
// after; the per-model loop polls it so Stop halts training after the current
// model instead of only at coarse phase boundaries.
static TRAINING_CANCEL: std::sync::OnceLock<
    std::sync::Mutex<Option<Arc<std::sync::atomic::AtomicBool>>>,
> = std::sync::OnceLock::new();

fn training_cancel_slot() -> &'static std::sync::Mutex<Option<Arc<std::sync::atomic::AtomicBool>>> {
    TRAINING_CANCEL.get_or_init(|| std::sync::Mutex::new(None))
}

/// Install (or clear with `None`) the cancellation flag the training loop polls
/// between models. Set before a training run, clear after.
pub fn set_training_cancel(flag: Option<Arc<std::sync::atomic::AtomicBool>>) {
    if let Ok(mut slot) = training_cancel_slot().lock() {
        *slot = flag;
    }
}

fn training_cancel_requested() -> bool {
    training_cancel_slot()
        .lock()
        .ok()
        .and_then(|slot| {
            slot.as_ref()
                .map(|f| f.load(std::sync::atomic::Ordering::Relaxed))
        })
        .unwrap_or(false)
}

use crate::base::ExpertModel;
use crate::burn_models::{active_burn_backend_name, normalize_burn_device_policy};
use crate::ensemble::{
    CalibrationMethod, ConformalPredictionExpert, MetaBlender, MetaDecisionStack,
    ProbabilityCalibrationExpert,
};
use crate::exit_agent::ExitAgent;
use crate::forecasting::hmm_regime::{HmmRegimeConfig, RegimeHmmExpert};
use crate::parallel_trainer::{
    ModelConfig, ModelTrainingFailure, ModelTrainingProgress, ModelType, TrainingPayload,
    train_models_parallel_with_progress,
};
use crate::runtime::capabilities::{CapabilityState, ModelFamily};
use crate::runtime::dispatch::{DispatchPlan, build_dispatch_plan};
use crate::runtime::hpo::{
    OPTIMIZATION_REPORT_FILE_NAME, OptimizationReport, OptimizationTrialRecord, ValidationMetrics,
    evaluate_prediction_quality, time_series_holdout_split, write_optimization_report,
};
use crate::runtime::profile::{
    TRAINING_RUNTIME_PROFILE_FILE_NAME, TrainingRuntimeProfile, write_training_runtime_profile,
};
use crate::runtime::training_artifact::{
    write_model_runtime_artifact_contract_sidecar, write_training_model_artifact_contract_sidecar,
};
use crate::soft_actor_critic::SoftActorCritic;
use crate::tree_models::config::ParamValue;
use crate::tree_models::{CatBoostExpert, LightGBMExpert, SklearsTreeExpert, XGBoostExpert};
use crate::{
    BayesianLogitExpert, ElasticNetExpert, GeneticStrategyExpert, IsolationForestExpert, KANExpert,
    LogisticExpert, MLPExpert, NBeatsExpert, NBeatsxNfExpert, NeatExpert, NeuroEvoExpert,
    OnlineHoeffdingExpert, OnlinePassiveAggressiveExpert, PatchTSTExpert, SwarmForecaster,
    TabNetExpert, TiDEExpert, TiDENfExpert, TimesNetExpert, TradingReinforcementLearner,
    TransformerExpert,
};
use neoethos_core::system::HardwareProbe;
use neoethos_core::{AcceleratorBackend, HardwareExecutionPlan, Settings, WorkloadKind};
use neoethos_data::{
    CanonicalDatasetSeriesReceiptV1, CanonicalTimeframe, FeatureBuildOptions, FeatureFrame, Ohlcv,
    SymbolDataset, load_exact_dataset_series_receipt, load_symbol_dataset,
    prepare_multitimeframe_features_with_options,
};
use neoethos_execution_budget::CpuLease;
use neoethos_search::genetic::{ParentSelectionPolicy, SurvivorSelectionPolicy};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingRunSummary {
    pub planned_models: Vec<String>,
    pub completed_models: Vec<String>,
    pub failed_models: Vec<ModelTrainingFailure>,
}

#[derive(Debug, Clone, Copy)]
enum TrainingLabelEconomics {
    BrokerFinancialTruth,
    CanonicalTrendbarScreeningV2 {
        pip_size: f64,
        round_trip_cost_pips: f64,
    },
}

pub struct TrainingOrchestrator {
    pub settings: neoethos_core::Settings,
    pub models_dir: PathBuf,
    /// v0.5 ML-integration Stage 4 — leak-free OOS-locked retrain. When `Some(ms)`,
    /// each symbol's training frame + labels are truncated to rows with
    /// `timestamp < ms`, minus a `label_horizon_bars` purge (the triple-barrier
    /// label looks forward), BEFORE the warmup drop + train/val split. The
    /// resulting experts have seen ZERO bars at/after the cutoff, so a blend that
    /// uses them can be validated on `[ms, end)` without look-ahead. Callers MUST
    /// point `models_dir` at a SEPARATE root (e.g. `models_oos_locked/`) so the
    /// production `models/` artifacts are never overwritten with leak-locked ones.
    pub oos_lock_from_ms: Option<i64>,
    /// Explicit override for the historical-bars root, for callers that resolve
    /// it themselves (the CLI's `--data-root`).
    ///
    /// **2026-08-10 — this replaces `NEOETHOS_BOT_DATA_ROOT`.** The orchestrator
    /// used to read that variable directly and only fall back to
    /// `system.data_dir`. `neoethos-cli` set it with `std::env::set_var` before
    /// calling into this crate, so the argument travelled between two parts of
    /// the SAME process through the environment — which also meant any leftover
    /// export in the operator's shell silently redirected training to a
    /// different bar store than the config named. `None` means
    /// `system.data_dir`, which is the configured answer.
    pub data_root_override: Option<PathBuf>,
}

const LEGACY_RLLIB_MIGRATION_ERROR_V1: &str = "legacy RLlib migration boundary v1: `use_rllib_agent`, `auto_enable_rllib`, and non-zero `rllib_num_workers` are unsupported because this build has no Ray runtime; set the flags to false, set `rllib_num_workers: 0`, and use `use_rl_agent: true` for the native rlkit DQN backend";

/// Keep legacy configuration keys readable, but never reinterpret a request
/// for Ray/RLlib as permission to run a different implementation. This guard
/// runs before dispatch materialization, so neither model selection nor any
/// backend/tool probing can occur for an unsupported request.
fn reject_legacy_rllib_request_v1(settings: &Settings) -> Result<()> {
    if settings.models.use_rllib_agent
        || settings.models.auto_enable_rllib
        || settings.models.rllib_num_workers != 0
    {
        anyhow::bail!(LEGACY_RLLIB_MIGRATION_ERROR_V1);
    }
    Ok(())
}

fn reject_legacy_rllib_model_params_v1(
    model_name: &str,
    params: &HashMap<String, String>,
) -> Result<()> {
    if canonical_model_name(model_name) != "dqn" {
        return Ok(());
    }

    let backend_requests_rllib = params
        .get("backend")
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("rllib"));
    let workers_request_rllib = params
        .get("rllib_num_workers")
        .is_some_and(|value| value.trim() != "0");
    let legacy_substitution_marker =
        params.contains_key("__rllib_requested") || params.contains_key("auto_rllib");
    if backend_requests_rllib || workers_request_rllib || legacy_substitution_marker {
        anyhow::bail!(LEGACY_RLLIB_MIGRATION_ERROR_V1);
    }
    Ok(())
}

fn is_supported_orchestrator_burn_device_policy(policy: &str) -> bool {
    matches!(policy, "auto" | "cpu" | "gpu") || policy.starts_with("gpu:")
}

fn model_requires_cuda_in_full_nvidia_run(name: &str, family: ModelFamily) -> bool {
    let canonical = canonical_model_name(name);
    crate::registry::CUDA_CAPABLE_MODEL_NAMES.contains(&canonical)
        || matches!(family, ModelFamily::Deep | ModelFamily::Exit)
        || canonical == "sac"
}

fn pin_cpu_only_model_device(
    name: &str,
    family: ModelFamily,
    params: &mut HashMap<String, String>,
) {
    if model_requires_cuda_in_full_nvidia_run(name, family) {
        return;
    }
    params.insert("device".to_string(), "cpu".to_string());
    params.insert("__planned_backend".to_string(), "cpu".to_string());
    params.insert("__planned_device".to_string(), "cpu".to_string());
}

const DENSE_TRAINING_MIN_ROWS: usize = 500;
const DENSE_TRAINING_REQUIRED_COLUMNS: [&str; 2] = ["quant_log_return", "quant_log_volatility"];

#[derive(Debug)]
struct DenseTrainingProjection {
    frame: FeatureFrame,
    row_start: usize,
    dropped_columns: usize,
}

fn trailing_valid_run(validity: &[neoethos_data::FeatureCellValidity]) -> usize {
    validity
        .iter()
        .rev()
        .take_while(|cell| cell.is_valid())
        .count()
}

/// Select the largest deterministic dense suffix available to strict model
/// adapters. The score maximizes retained valid cells, then retained rows,
/// while the two versioned HMM inputs and at least 500 rows are mandatory.
/// This is a typed column/row projection over the purged in-sample frame: it
/// never changes a cell value or validity reason and never reads OOS rows.
fn project_dense_training_suffix(frame: &FeatureFrame) -> Result<DenseTrainingProjection> {
    anyhow::ensure!(
        frame.n_samples() >= DENSE_TRAINING_MIN_ROWS,
        "training feature frame has {} rows; dense model training requires at least {}",
        frame.n_samples(),
        DENSE_TRAINING_MIN_ROWS
    );

    let mut trailing_runs = Vec::with_capacity(frame.n_features());
    for feature in 0..frame.n_features() {
        let column = frame
            .feature_column(feature)
            .with_context(|| format!("inspect validity for feature `{}`", frame.names[feature]))?;
        trailing_runs.push(trailing_valid_run(&column.validity));
    }

    let required_run_cap = DENSE_TRAINING_REQUIRED_COLUMNS
        .iter()
        .map(|required| {
            let feature = frame
                .names
                .iter()
                .position(|name| name == required)
                .with_context(|| format!("dense training is missing required feature `{required}`"))?;
            let run = trailing_runs[feature];
            anyhow::ensure!(
                run >= DENSE_TRAINING_MIN_ROWS,
                "dense training required feature `{required}` has only {run} trailing valid rows; need at least {DENSE_TRAINING_MIN_ROWS}"
            );
            Ok(run)
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .min()
        .context("dense training required-column set is empty")?;

    let mut candidate_rows = trailing_runs
        .iter()
        .copied()
        .filter(|run| *run >= DENSE_TRAINING_MIN_ROWS && *run <= required_run_cap)
        .collect::<Vec<_>>();
    candidate_rows.sort_unstable();
    candidate_rows.dedup();

    let mut best: Option<(usize, usize, usize)> = None;
    for rows in candidate_rows {
        let columns = trailing_runs.iter().filter(|run| **run >= rows).count();
        let cells = rows.checked_mul(columns).ok_or_else(|| {
            anyhow::anyhow!("dense training suffix score overflow: {rows} rows x {columns} columns")
        })?;
        if best.is_none_or(|current| (cells, rows, columns) > current) {
            best = Some((cells, rows, columns));
        }
    }
    let (_, retained_rows, _) = best.context("training frame has no dense valid suffix")?;
    let kept_columns = trailing_runs
        .iter()
        .enumerate()
        .filter_map(|(feature, run)| (*run >= retained_rows).then_some(feature))
        .collect::<Vec<_>>();
    anyhow::ensure!(
        !kept_columns.is_empty(),
        "dense training suffix retained no feature columns"
    );
    for required in DENSE_TRAINING_REQUIRED_COLUMNS {
        anyhow::ensure!(
            kept_columns
                .iter()
                .any(|feature| frame.names[*feature] == required),
            "dense training suffix dropped required feature `{required}`"
        );
    }

    let dropped_columns = frame.n_features().saturating_sub(kept_columns.len());
    let row_start = frame.n_samples() - retained_rows;
    let projected_columns = frame.select_columns(&kept_columns)?;
    let projected = projected_columns.row_window(row_start, frame.n_samples())?;
    for feature in 0..projected.n_features() {
        let column = projected.feature_column(feature)?;
        anyhow::ensure!(
            column.validity.iter().all(|cell| cell.is_valid()),
            "dense training projection retained an invalid cell in feature `{}`",
            column.name
        );
    }

    Ok(DenseTrainingProjection {
        frame: projected,
        row_start,
        dropped_columns,
    })
}

/// Write a hardware-plan value into a model's params, RECORDING the fact when
/// it replaced a different value that config or a per-model override had put
/// there.
///
/// Audit #177/#178/#180/#182/#183/#184/#186/#188 — the "config knob whose
/// reader is overwritten" family. The plan winning is correct and intentional
/// (it is what keeps the never-OOM invariant); the plan winning **silently** is
/// the defect. `models.train_batch_size` is the measured case: 13 readers, all
/// replaced here, and nothing told the operator his value never arrived.
fn plan_insert(
    params: &mut HashMap<String, String>,
    overridden: &mut Vec<String>,
    key: &str,
    value: String,
) {
    if let Some(prev) = params.get(key)
        && prev != &value
    {
        overridden.push(format!("{key}: configured={prev} → effective={value}"));
    }
    params.insert(key.to_string(), value);
}

fn burn_policy_from_workload_device(device: &str) -> String {
    let normalized = device.trim().to_ascii_lowercase();
    if normalized.is_empty() || normalized == "cpu" {
        return "cpu".to_string();
    }
    if let Some((_, index)) = normalized.split_once(':') {
        if index == "all" {
            return "gpu".to_string();
        }
        if index.parse::<usize>().is_ok() {
            return format!("gpu:{index}");
        }
    }
    "gpu".to_string()
}

impl TrainingOrchestrator {
    pub fn new(settings: neoethos_core::Settings, models_dir: PathBuf) -> Self {
        Self {
            settings,
            models_dir,
            oos_lock_from_ms: None,
            data_root_override: None,
        }
    }

    /// Builder: point training at an explicit historical-bars root instead of
    /// `system.data_dir`. This is the replacement for the CLI setting
    /// `NEOETHOS_BOT_DATA_ROOT` in its own process. See
    /// [`Self::data_root_override`].
    pub fn with_data_root(mut self, data_root: impl Into<PathBuf>) -> Self {
        self.data_root_override = Some(data_root.into());
        self
    }

    /// The historical-bars root this run will read: the explicit override if
    /// one was given, else `system.data_dir`. There is no third source.
    pub fn data_root(&self) -> String {
        match self.data_root_override.as_ref() {
            Some(root) => root.to_string_lossy().to_string(),
            None => self.settings.system.data_dir.to_string_lossy().to_string(),
        }
    }

    /// Builder: lock training to the in-sample window `timestamp < oos_from_ms`
    /// (minus the triple-barrier purge) so the trained experts are leak-free for
    /// an OOS blend validation on `[oos_from_ms, end)`. See [`Self::oos_lock_from_ms`].
    pub fn with_oos_lock_from_ms(mut self, oos_from_ms: i64) -> Self {
        self.oos_lock_from_ms = Some(oos_from_ms);
        self
    }

    fn preferred_burn_device_policy(&self) -> String {
        let gpu_pref = self
            .settings
            .system
            .enable_gpu_preference
            .trim()
            .to_ascii_lowercase();
        let system_device = self.settings.system.device.trim().to_ascii_lowercase();
        let normalized_system_device = normalize_burn_device_policy(&system_device);
        match gpu_pref.as_str() {
            "false" | "cpu" => "cpu".to_string(),
            "true" | "gpu" => {
                if system_device.is_empty() || system_device == "cpu" {
                    "gpu".to_string()
                } else if is_supported_orchestrator_burn_device_policy(&normalized_system_device) {
                    normalized_system_device
                } else {
                    "gpu".to_string()
                }
            }
            _ => {
                if is_supported_orchestrator_burn_device_policy(&normalized_system_device) {
                    normalized_system_device
                } else {
                    "auto".to_string()
                }
            }
        }
    }

    fn full_nvidia_device_policy_for_config(&self, config: &ModelConfig) -> Result<String> {
        let canonical = canonical_model_name(&config.name);
        match canonical {
            "meta_blender" | "probability_calibrator" | "conformal_gate" | "meta_stack" => {
                Ok(self.settings.models.tree_runtime.device.clone())
            }
            "elasticnet" | "logistic" => Ok(config
                .params
                .get("device")
                .cloned()
                .unwrap_or_else(|| self.settings.models.statistical_device.clone())),
            _ => config.params.get("device").cloned().with_context(|| {
                format!(
                    "full NVIDIA training model `{}` has no explicit device policy",
                    config.name
                )
            }),
        }
    }

    /// Prove that one configured model has a compiled GPU implementation and
    /// is pinned to the exact CUDA device admitted by the full-run hardware
    /// plan. A full-GPU run never treats an explicit CPU plan as an acceptable
    /// substitute: unsupported models stop before feature construction,
    /// Search, or training spends accelerator time.
    fn validate_nvidia_model_config_v1(&self, config: &ModelConfig) -> Result<()> {
        let canonical = canonical_model_name(&config.name);
        anyhow::ensure!(
            crate::registry::supports_nvidia_cuda_for_model(canonical, config.capability_family),
            "full NVIDIA model `{}` has no compiled NVIDIA CUDA implementation in this build",
            config.name
        );

        let policy = self.full_nvidia_device_policy_for_config(config)?;
        let parsed = crate::common::parse_cuda_device_policy(&policy).with_context(|| {
            format!(
                "full NVIDIA model `{}` has invalid device policy `{policy}`",
                config.name
            )
        })?;
        anyhow::ensure!(
            policy.trim().contains(':')
                && matches!(parsed, crate::common::CudaDevicePolicy::Gpu { ordinal: 0 }),
            "full NVIDIA model `{}` must request exact CUDA ordinal 0, got `{policy}`",
            config.name
        );
        Ok(())
    }

    /// Validate the complete configured model expansion and its exact CUDA-only
    /// routing before a canonical full run spends time building features or
    /// searching. This is deliberately stricter than ordinary CPU-capable
    /// training: every selected model must have a compiled NVIDIA CUDA
    /// implementation and target exact CUDA ordinal zero. There is no CPU
    /// substitution.
    pub fn preflight_full_nvidia_cuda_training(&self) -> Result<Vec<String>> {
        let dispatch_plan = self.create_dispatch_plan()?;
        self.validate_dispatch_plan(&dispatch_plan)?;
        let hardware_plan = self.hardware_execution_plan();
        anyhow::ensure!(
            hardware_plan.gpu_enabled
                && hardware_plan.primary_backend == AcceleratorBackend::Cuda
                && hardware_plan
                    .profile
                    .accelerator_devices
                    .iter()
                    .any(|device| { device.id == 0 && device.backend == AcceleratorBackend::Cuda }),
            "full NVIDIA training requires a detected CUDA device at ordinal 0"
        );
        for workload_kind in [
            WorkloadKind::StrategySearch,
            WorkloadKind::TreeTraining,
            WorkloadKind::DeepTraining,
            WorkloadKind::RlTraining,
        ] {
            let workload = hardware_plan.workload(workload_kind).with_context(|| {
                format!("full NVIDIA training lacks the {workload_kind:?} hardware plan")
            })?;
            anyhow::ensure!(
                workload.backend == AcceleratorBackend::Cuda && workload.device_ids.contains(&0),
                "full NVIDIA training workload {workload_kind:?} is not pinned to CUDA ordinal 0: backend={:?}, device={}, ordinals={:?}",
                workload.backend,
                workload.device,
                workload.device_ids
            );
        }

        let configs =
            self.build_training_configs_with_hardware_plan(&dispatch_plan, &hardware_plan)?;
        let required_voters = crate::ensemble_inference::DEFAULT_BOOTSTRAP_EXPERT_NAMES
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let planned_canonical = configs
            .iter()
            .map(|config| canonical_model_name(&config.name))
            .collect::<BTreeSet<_>>();
        let missing_voters = required_voters
            .difference(&planned_canonical)
            .copied()
            .collect::<Vec<_>>();
        let unconsumed_models = planned_canonical
            .difference(&required_voters)
            .copied()
            .collect::<Vec<_>>();
        anyhow::ensure!(
            missing_voters.is_empty(),
            "full NVIDIA training dispatch is missing production ensemble voters: {}",
            missing_voters.join(", ")
        );
        anyhow::ensure!(
            unconsumed_models.is_empty(),
            "full NVIDIA training dispatch contains models with no production ensemble consumer: {}",
            unconsumed_models.join(", ")
        );
        let mut planned_models = Vec::with_capacity(configs.len());
        for config in configs {
            self.validate_nvidia_model_config_v1(&config)?;
            planned_models.push(config.name);
        }
        Ok(planned_models)
    }

    /// Validate only the exact model set enabled for one explicitly scoped
    /// NVIDIA training run. Unlike the full-ensemble preflight, this does not
    /// invent missing voters that are outside the configured plan; every model
    /// that is present still has to resolve to its real CUDA implementation on
    /// ordinal zero. An explicit CPU-only policy is rejected, not substituted.
    pub fn preflight_configured_nvidia_training(&self) -> Result<Vec<String>> {
        let dispatch_plan = self.create_dispatch_plan()?;
        self.validate_dispatch_plan(&dispatch_plan)?;
        let hardware_plan = self.hardware_execution_plan();
        anyhow::ensure!(
            hardware_plan.gpu_enabled
                && hardware_plan.primary_backend == AcceleratorBackend::Cuda
                && hardware_plan
                    .profile
                    .accelerator_devices
                    .iter()
                    .any(|device| { device.id == 0 && device.backend == AcceleratorBackend::Cuda }),
            "configured NVIDIA training requires a detected CUDA device at ordinal 0"
        );
        let configs =
            self.build_training_configs_with_hardware_plan(&dispatch_plan, &hardware_plan)?;
        anyhow::ensure!(
            !configs.is_empty(),
            "configured NVIDIA training resolved an empty model plan"
        );

        let mut planned_models = Vec::with_capacity(configs.len());
        for config in configs {
            self.validate_nvidia_model_config_v1(&config)?;
            planned_models.push(config.name);
        }
        Ok(planned_models)
    }

    pub fn train_symbol(&self, symbol: &str, base_tf: &str, lease: &CpuLease) -> Result<()> {
        let summary = self.train_symbol_with_progress(symbol, base_tf, lease, |_| {})?;
        if !summary.failed_models.is_empty() {
            anyhow::bail!(
                "Training failed for [{}]; successful models: [{}]",
                summary
                    .failed_models
                    .iter()
                    .map(|failure| failure.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
                summary.completed_models.join(", ")
            );
        }

        Ok(())
    }

    pub fn train_symbol_with_progress<R>(
        &self,
        symbol: &str,
        base_tf: &str,
        lease: &CpuLease,
        progress_fn: R,
    ) -> Result<TrainingRunSummary>
    where
        R: Fn(ModelTrainingProgress) + Send + Sync + Clone + 'static,
    {
        let data_root = self.data_root();
        let dataset = load_symbol_dataset(&data_root, symbol)?;
        self.train_dataset_with_progress(
            symbol,
            base_tf,
            dataset,
            TrainingLabelEconomics::BrokerFinancialTruth,
            lease,
            progress_fn,
        )
    }

    /// Train the complete configured model surface from one explicitly
    /// selected immutable canonical series. Every direct timeframe is reopened
    /// by generation id + manifest-binding hash before feature computation.
    pub fn train_canonical_series_with_progress<R>(
        &self,
        series: &CanonicalDatasetSeriesReceiptV1,
        base_tf: CanonicalTimeframe,
        search_input: neoethos_search::data_selection::CanonicalSearchInput,
        screening_contract: &neoethos_search::CanonicalTrendbarResearchExecutionContractV3,
        lease: &CpuLease,
        progress_fn: R,
    ) -> Result<TrainingRunSummary>
    where
        R: Fn(ModelTrainingProgress) + Send + Sync + Clone + 'static,
    {
        series.validate()?;
        let (pip_size, round_trip_cost_pips) = {
            let search_input = search_input.as_run_input()?;
            screening_contract.validate_against_input(&search_input)?;
            (
                screening_contract.pip_size(),
                screening_contract.screening_round_trip_cost_pips(),
            )
        };
        drop(search_input);
        let symbol = series.anchor().identity().symbol_name();
        let required_timeframes = self.selected_feature_timeframes(base_tf.as_str())?;
        let selected_timeframes = series
            .direct_timeframes()
            .iter()
            .map(|selected| selected.identity().timeframe())
            .collect::<BTreeSet<_>>();
        anyhow::ensure!(
            selected_timeframes.contains(&base_tf),
            "canonical training series for {symbol} does not contain base timeframe {base_tf}"
        );
        for required in &required_timeframes {
            let required = required.parse::<CanonicalTimeframe>().with_context(|| {
                format!("training requested non-canonical feature timeframe {required}")
            })?;
            anyhow::ensure!(
                selected_timeframes.contains(&required),
                "canonical training series for {symbol} does not contain required direct timeframe {required}"
            );
        }
        let dataset = load_exact_dataset_series_receipt(self.data_root(), series)?;
        self.train_dataset_with_progress(
            symbol,
            base_tf.as_str(),
            dataset,
            TrainingLabelEconomics::CanonicalTrendbarScreeningV2 {
                pip_size,
                round_trip_cost_pips,
            },
            lease,
            progress_fn,
        )
    }

    /// Train from an immutable canonical series and an independently persisted
    /// canonical-search input receipt. This is the exact two-phase hand-off for
    /// a completed historical search whose feature frame has already been
    /// released; it preserves receipt-bound screening economics without
    /// requiring live broker financial truth or replaying Discovery.
    pub fn train_canonical_series_receipt_with_progress<R>(
        &self,
        series: &CanonicalDatasetSeriesReceiptV1,
        base_tf: CanonicalTimeframe,
        input_receipt: &neoethos_search::CanonicalSearchInputReceiptV2,
        screening_contract: &neoethos_search::CanonicalTrendbarResearchExecutionContractV3,
        lease: &CpuLease,
        progress_fn: R,
    ) -> Result<TrainingRunSummary>
    where
        R: Fn(ModelTrainingProgress) + Send + Sync + Clone + 'static,
    {
        series.validate()?;
        screening_contract.validate_against_receipt(input_receipt)?;
        let pip_size = screening_contract.pip_size();
        let round_trip_cost_pips = screening_contract.screening_round_trip_cost_pips();
        let symbol = series.anchor().identity().symbol_name();
        let required_timeframes = self.selected_feature_timeframes(base_tf.as_str())?;
        let selected_timeframes = series
            .direct_timeframes()
            .iter()
            .map(|selected| selected.identity().timeframe())
            .collect::<BTreeSet<_>>();
        anyhow::ensure!(
            selected_timeframes.contains(&base_tf),
            "canonical training series for {symbol} does not contain base timeframe {base_tf}"
        );
        for required in &required_timeframes {
            let required = required.parse::<CanonicalTimeframe>().with_context(|| {
                format!("training requested non-canonical feature timeframe {required}")
            })?;
            anyhow::ensure!(
                selected_timeframes.contains(&required),
                "canonical training series for {symbol} does not contain required direct timeframe {required}"
            );
        }
        let dataset = load_exact_dataset_series_receipt(self.data_root(), series)?;
        self.train_dataset_with_progress(
            symbol,
            base_tf.as_str(),
            dataset,
            TrainingLabelEconomics::CanonicalTrendbarScreeningV2 {
                pip_size,
                round_trip_cost_pips,
            },
            lease,
            progress_fn,
        )
    }

    pub fn train_canonical_series(
        &self,
        series: &CanonicalDatasetSeriesReceiptV1,
        base_tf: CanonicalTimeframe,
        search_input: neoethos_search::data_selection::CanonicalSearchInput,
        screening_contract: &neoethos_search::CanonicalTrendbarResearchExecutionContractV3,
        lease: &CpuLease,
    ) -> Result<()> {
        let summary = self.train_canonical_series_with_progress(
            series,
            base_tf,
            search_input,
            screening_contract,
            lease,
            |_| {},
        )?;
        if !summary.failed_models.is_empty() {
            anyhow::bail!(
                "Training failed for [{}]; successful models: [{}]",
                summary
                    .failed_models
                    .iter()
                    .map(|failure| failure.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
                summary.completed_models.join(", ")
            );
        }
        Ok(())
    }

    fn train_dataset_with_progress<R>(
        &self,
        symbol: &str,
        base_tf: &str,
        dataset: SymbolDataset,
        label_economics: TrainingLabelEconomics,
        lease: &CpuLease,
        progress_fn: R,
    ) -> Result<TrainingRunSummary>
    where
        R: Fn(ModelTrainingProgress) + Send + Sync + Clone + 'static,
    {
        info!("Starting Pure-Rust training for symbol: {}", symbol);

        let dispatch_plan = self.create_dispatch_plan()?;
        self.validate_dispatch_plan(&dispatch_plan)?;
        let configs = self.build_training_configs(&dispatch_plan)?;
        let planned_models: Vec<String> =
            configs.iter().map(|config| config.name.clone()).collect();

        let higher_tfs = self.selected_feature_timeframes(base_tf)?;
        let base_ohlcv = dataset.frames.get(base_tf).context("base tf missing")?;
        let oos_training_boundary = if let Some(cutoff) = self.oos_lock_from_ms {
            let timestamps = base_ohlcv
                .timestamp
                .as_ref()
                .context("OOS-lock requires base-tf timestamps")?;
            let in_sample = timestamps.iter().take_while(|&&t| t < cutoff).count();
            // Purge the last `label_horizon_bars` IS rows whose triple-barrier
            // label looks forward across the cutoff. Normalization is fitted
            // to this same purged prefix, never to the OOS suffix or purge rows.
            let purge = self.settings.models.label_horizon_bars;
            let keep = in_sample.saturating_sub(purge);
            if keep < 256 {
                anyhow::bail!(
                    "OOS-lock: only {keep} in-sample rows before {cutoff} (after a {purge}-bar \
                     purge) for {symbol}/{base_tf} — too few to train leak-free; pick a later cutoff"
                );
            }
            Some((cutoff, in_sample, purge, keep))
        } else {
            None
        };
        let opts = FeatureBuildOptions {
            higher_tfs: higher_tfs.clone(),
            prefix_base_features: self.settings.system.multi_resolution_prefix_base,
            normalization_training_rows: oos_training_boundary.map(|(_, _, _, keep)| 0..keep),
            drop_columns_without_normalization_training_support: true,
            ..FeatureBuildOptions::default()
        };
        let mut frame = lease
            .scope(|| prepare_multitimeframe_features_with_options(&dataset, base_tf, &opts))?;
        let mut labels = match label_economics {
            TrainingLabelEconomics::BrokerFinancialTruth => {
                self.derive_labels(base_ohlcv, symbol)?
            }
            TrainingLabelEconomics::CanonicalTrendbarScreeningV2 {
                pip_size,
                round_trip_cost_pips,
            } => self.derive_labels_with_screening_costs(
                base_ohlcv,
                symbol,
                pip_size,
                round_trip_cost_pips,
            )?,
        };
        if frame.n_samples() != labels.len() {
            anyhow::bail!(
                "training frame/label mismatch for {symbol}/{base_tf}: {} rows vs {} labels",
                frame.n_samples(),
                labels.len()
            );
        }
        let mut source_row_indices = (0..frame.n_samples()).collect::<Vec<_>>();

        // Stage 4 leak-free OOS lock: truncate to rows STRICTLY before the cutoff,
        // minus a triple-barrier purge, BEFORE the warmup drop + split. The frame
        // rows are 1:1 with `base_ohlcv` (and `labels`) here, so truncate the typed
        // frame, labels, and canonical source-row receipts in lock-step.
        if let Some((cutoff, in_sample, purge, keep)) = oos_training_boundary {
            let timestamps = base_ohlcv
                .timestamp
                .as_ref()
                .context("OOS-lock requires base-tf timestamps")?;
            if timestamps.as_slice() != frame.timestamps.as_slice() {
                anyhow::bail!(
                    "OOS-lock: base timestamp/feature receipt mismatch ({} ts vs {} rows)",
                    timestamps.len(),
                    frame.n_samples()
                );
            }
            info!(
                "OOS-lock: {symbol}/{base_tf} truncated to {keep} in-sample rows (< {cutoff}, \
                 purged {purge} look-ahead rows of {in_sample}); experts will not see >= cutoff"
            );
            frame = frame.row_window(0, keep)?;
            labels.truncate(keep);
            source_row_indices.truncate(keep);
        }

        // Dense model adapters intentionally fail on typed invalid cells. A
        // raw intersection across thousands of partially-supported features is
        // empty even on a long frame, so select one exact in-sample suffix and
        // the columns that are valid throughout it. No values are rewritten.
        let rows_before_dense_projection = frame.n_samples();
        let dense_projection = project_dense_training_suffix(&frame)?;
        let dense_row_start = dense_projection.row_start;
        let dense_dropped_columns = dense_projection.dropped_columns;
        frame = dense_projection.frame;
        labels = labels[dense_row_start..].to_vec();
        source_row_indices = source_row_indices[dense_row_start..].to_vec();
        info!(
            "Projected strict dense training suffix: kept {} / {} rows and {} columns; dropped {} partially-supported columns without changing cell values",
            frame.n_samples(),
            rows_before_dense_projection,
            frame.n_features(),
            dense_dropped_columns
        );

        let (filtered_frame, filtered_labels, filtered_source_rows) =
            if self.settings.models.filter_to_base_signal {
                self.apply_base_signal_filter(&frame, &labels, &source_row_indices)?
            } else {
                (frame, labels, source_row_indices)
            };
        let (budgeted_frame, budgeted_labels, budgeted_source_rows, row_budget_applied) = self
            .apply_training_row_budget(&filtered_frame, &filtered_labels, &filtered_source_rows)?;
        // HMM training consumes the exact versioned feature-plan values after
        // the same OOS, validity, signal-filter and row-budget boundaries as
        // every other model. It never reconstructs observations from bare
        // OHLCV or from a feature-selected frame that may omit its columns.
        let hmm_observations: std::sync::Arc<std::result::Result<ndarray::Array2<f64>, String>> =
            std::sync::Arc::new(
                RegimeHmmExpert::training_observations_from_feature_frame(&budgeted_frame, lease)
                    .map_err(|error| error.to_string()),
            );
        let selected_frame = self.apply_feature_selection(
            &budgeted_frame,
            &budgeted_labels,
            &budgeted_source_rows,
            base_ohlcv,
            lease,
        )?;
        let payload = Arc::new(TrainingPayload::from_frame_with_source_rows(
            selected_frame,
            budgeted_labels,
            budgeted_source_rows,
        )?);
        let models_dir = self.models_dir.clone();
        let settings = self.settings.clone();
        let symbol = symbol.to_string();
        let base_tf = base_tf.to_string();
        let trained = train_models_parallel_with_progress(
            configs,
            payload,
            lease,
            progress_fn,
            move |config, payload, model_lease| {
                info!(
                    "Training model instance: {} ({:?})",
                    config.name, config.model_type
                );
                train_model_dispatch(
                    &models_dir,
                    &settings,
                    &symbol,
                    &base_tf,
                    row_budget_applied,
                    config,
                    payload,
                    model_lease,
                    &hmm_observations,
                )
            },
        )?;

        info!(
            "Successfully trained models: {:?}",
            trained.successful_models
        );
        Ok(TrainingRunSummary {
            planned_models,
            completed_models: trained.successful_models,
            failed_models: trained.failed_models,
        })
    }

    fn create_dispatch_plan(&self) -> Result<DispatchPlan> {
        reject_legacy_rllib_request_v1(&self.settings)?;
        let mut requested_models = self.settings.models.ml_models.clone();
        requested_models.extend(self.settings.models.phase5_core_models.clone());

        if self.settings.models.regime_router_enabled {
            requested_models.extend(self.settings.models.regime_trend_models.clone());
            requested_models.extend(self.settings.models.regime_range_models.clone());
            requested_models.extend(self.settings.models.regime_neutral_models.clone());
        }

        if self.settings.models.phase5_filter_meta_blender {
            requested_models.push("meta_blender".to_string());
        }
        if self.settings.models.calibration_enabled {
            requested_models.push("probability_calibrator".to_string());
            if self.settings.risk.conformal_enabled {
                requested_models.push("conformal_gate".to_string());
            }
            if self.settings.models.phase5_filter_meta_blender {
                requested_models.push("meta_stack".to_string());
            }
        }
        if self.settings.models.use_sac_agent {
            // `use_sac_agent` now drives the REAL discrete Soft
            // Actor-Critic entry/direction policy (see
            // `crate::soft_actor_critic`), which participates in the
            // soft-voting ensemble like the DQN entry voter. It used to
            // silently alias to the DQN-backed `exit_agent`; that alias
            // is gone. The exit-side agent is not auto-requested here:
            // it has no production consumer (F-318). Operators can still
            // request `exit_agent` explicitly through `models.ml_models`
            // when developing a future exit-side pipeline.
            requested_models.push("sac".to_string());
        }
        if self.settings.models.use_rl_agent {
            requested_models.push("dqn".to_string());
        }
        if self.settings.models.use_neuroevolution {
            requested_models.push("neuro_evo".to_string());
            requested_models.push("neat".to_string());
        }
        // `prop_search_*` configures the real discovery/search cycle. Do not
        // also train the separate `GeneticStrategyExpert` implicitly: its
        // artifact has no production loader and discovery already performs
        // the strategy search. The expert remains available when explicitly
        // listed in `models.ml_models` for direct development work.
        // hmm_regime — the soft regime gate the ensemble's role-aware
        // combiner expects. Loader + adapter shipped 2026-05-25 but the
        // training request was never added, so every install reported it
        // permanently "missing" (operator audit 2026-07-11). Train it
        // whenever ANY training is happening (Baum-Welch on a 2-column
        // matrix is seconds of CPU); an entirely-empty request still
        // fails loud below — that's a misconfiguration, not a run.
        if !requested_models.is_empty() {
            requested_models.push("hmm_regime".to_string());
        }
        requested_models = self.expand_requested_models(requested_models);
        if self.settings.models.regime_router_enabled
            && self.settings.models.regime_router_min_models > 0
        {
            let routed_models = requested_models
                .iter()
                .filter(|name| {
                    configured_contains_model(&self.settings.models.regime_trend_models, name)
                        || configured_contains_model(
                            &self.settings.models.regime_range_models,
                            name,
                        )
                        || configured_contains_model(
                            &self.settings.models.regime_neutral_models,
                            name,
                        )
                })
                .count();
            if routed_models < self.settings.models.regime_router_min_models {
                anyhow::bail!(
                    "regime router requires at least {} routed models but only {} are configured",
                    self.settings.models.regime_router_min_models,
                    routed_models
                );
            }
        }
        build_dispatch_plan(&requested_models)
    }

    fn expand_requested_models(&self, requested_models: Vec<String>) -> Vec<String> {
        let mut expanded = Vec::new();

        for name in requested_models {
            // Operator Stop: halt after the current model rather than only at
            // coarse phase boundaries.
            if training_cancel_requested() {
                warn!(
                    target: "neoethos_models::training",
                    "training cancelled by operator — stopping after the current model"
                );
                break;
            }
            let trimmed = name.trim();
            if trimmed.is_empty() {
                continue;
            }

            if trimmed.eq_ignore_ascii_case("transformer") {
                if !self.settings.models.enable_transformer_expert {
                    continue;
                }

                let replica_count = self.settings.models.num_transformers.max(1);
                if replica_count == 1 {
                    expanded.push("transformer".to_string());
                } else {
                    for replica_idx in 1..=replica_count {
                        expanded.push(format!("transformer_{replica_idx:02}"));
                    }
                }
                continue;
            }

            expanded.push(trimmed.to_string());
        }

        expanded
    }

    fn validate_dispatch_plan(&self, dispatch_plan: &DispatchPlan) -> Result<()> {
        let blocked: Vec<String> = dispatch_plan
            .entries
            .iter()
            .filter(|entry| entry.state == CapabilityState::Planned)
            .map(|entry| format!("{} ({}, {})", entry.name, entry.family, entry.state))
            .collect();

        if !blocked.is_empty() {
            anyhow::bail!(
                "configured models resolve to planned capabilities and cannot enter the training dispatch: {}",
                blocked.join(", ")
            );
        }

        Ok(())
    }

    fn build_training_configs(&self, dispatch_plan: &DispatchPlan) -> Result<Vec<ModelConfig>> {
        let hardware_plan = self.hardware_execution_plan();
        self.build_training_configs_with_hardware_plan(dispatch_plan, &hardware_plan)
    }

    fn build_training_configs_with_hardware_plan(
        &self,
        dispatch_plan: &DispatchPlan,
        hardware_plan: &HardwareExecutionPlan,
    ) -> Result<Vec<ModelConfig>> {
        dispatch_plan
            .entries
            .iter()
            .map(|entry| {
                let mut params = self.default_model_params(&entry.name);
                self.inject_runtime_model_params(&entry.name, &mut params);
                self.apply_model_param_overrides(&entry.name, &mut params);
                reject_legacy_rllib_model_params_v1(&entry.name, &params)?;
                // Resource limits and accelerator provenance are authoritative.
                // Apply them last so stale settings or per-model overrides cannot
                // oversubscribe the detected machine or route work to another device.
                self.apply_hardware_plan_params(
                    &entry.name,
                    entry.family,
                    hardware_plan,
                    &mut params,
                );
                pin_cpu_only_model_device(&entry.name, entry.family, &mut params);
                Ok(ModelConfig {
                    name: entry.name.clone(),
                    model_type: self.map_model_type(&entry.name)?,
                    capability_family: entry.family,
                    capability_state: entry.state,
                    params,
                })
            })
            .collect()
    }

    fn hardware_execution_plan(&self) -> HardwareExecutionPlan {
        let mut probe = HardwareProbe::new();
        let profile = probe.detect();
        HardwareExecutionPlan::from_settings_and_profile(&self.settings, profile)
    }

    fn apply_hardware_plan_params(
        &self,
        name: &str,
        family: ModelFamily,
        plan: &HardwareExecutionPlan,
        params: &mut HashMap<String, String>,
    ) {
        let workload = match family {
            ModelFamily::Tree => Some(WorkloadKind::TreeTraining),
            ModelFamily::Deep | ModelFamily::Exit => Some(WorkloadKind::DeepTraining),
            ModelFamily::Rl => Some(WorkloadKind::RlTraining),
            ModelFamily::Evolutionary if canonical_model_name(name) == "genetic" => {
                Some(WorkloadKind::StrategySearch)
            }
            _ => None,
        };
        let Some(workload) = workload.and_then(|kind| plan.workload(kind)) else {
            return;
        };
        params.insert(
            "__planned_backend".to_string(),
            workload.backend.as_str().to_string(),
        );
        params.insert("__planned_device".to_string(), workload.device.clone());
        params.insert(
            "__planned_precision".to_string(),
            workload.precision.as_str().to_string(),
        );
        params.insert(
            "__planned_cpu_threads".to_string(),
            workload.cpu_threads.to_string(),
        );
        params.insert(
            "__planned_batch_size".to_string(),
            workload.batch_size.to_string(),
        );
        params.insert(
            "__planned_memory_budget_gb".to_string(),
            format!("{:.6}", workload.memory_budget_gb),
        );
        // ── AUDIT #177/#178/#180/#182/#183/#184/#186/#188 — the "knob whose
        // reader is overwritten" family, made VISIBLE ────────────────────
        //
        // Everything below deliberately WINS over config and per-model
        // overrides: the machine's real capacity is authoritative, and that is
        // the correct precedence (it is what keeps the never-OOM invariant).
        // The defect was never the precedence — it was the SILENCE. A knob the
        // operator sets in `config.yaml` (`models.train_batch_size` is the
        // measured case: 13 readers, all replaced here) is simply gone by the
        // time training runs, with nothing anywhere saying it was replaced or
        // by what. He tunes a number, the number does nothing, and no log line
        // contradicts him.
        //
        // Nothing about WHICH value wins changes here. Only that a replaced
        // value is now named, with its configured and effective figures, once
        // per model. If a knob you set never appears in this line, the plan did
        // not touch it — and its reader is a different problem.
        let mut overridden: Vec<String> = Vec::new();

        plan_insert(
            params,
            &mut overridden,
            "cpu_threads",
            workload.cpu_threads.to_string(),
        );
        if workload.batch_size > 0 {
            plan_insert(
                params,
                &mut overridden,
                "batch_size",
                workload.batch_size.to_string(),
            );
        }
        plan_insert(
            params,
            &mut overridden,
            "memory_budget_gb",
            format!("{:.6}", workload.memory_budget_gb),
        );

        match family {
            ModelFamily::Tree | ModelFamily::Rl => {
                plan_insert(params, &mut overridden, "device", workload.device.clone());
            }
            ModelFamily::Deep | ModelFamily::Exit => {
                plan_insert(
                    params,
                    &mut overridden,
                    "device",
                    burn_policy_from_workload_device(&workload.device),
                );
                plan_insert(
                    params,
                    &mut overridden,
                    "training_precision",
                    workload.precision.as_str().to_string(),
                );
            }
            ModelFamily::Evolutionary if canonical_model_name(name) == "genetic" => {
                plan_insert(params, &mut overridden, "device", workload.device.clone());
            }
            _ => {}
        }

        if !overridden.is_empty() {
            tracing::warn!(
                target: "neoethos_models::training",
                model = %name,
                overrides = %overridden.join("; "),
                "hardware execution plan REPLACED configured training parameters — the machine's \
                 measured capacity is authoritative (never-OOM), so these config values do not \
                 reach training"
            );
        }
    }

    fn epochs_from_seconds(seconds: u64, default_epochs: usize) -> usize {
        let derived = (seconds / 30).clamp(16, 400) as usize;
        derived.max(default_epochs)
    }

    fn max_epochs_for_model(&self, name: &str, default_epochs: usize) -> usize {
        if let Some(explicit) = self.settings.models.max_epochs_by_model.get(name) {
            return (*explicit).max(1);
        }

        let seconds = match name {
            "transformer" => self.settings.models.transformer_train_seconds,
            "nbeats" | "nbeatsx_nf" => self.settings.models.nbeats_train_seconds,
            "tide" | "tide_nf" => self.settings.models.tide_train_seconds,
            "tabnet" => self.settings.models.tabnet_train_seconds,
            "kan" => self.settings.models.kan_train_seconds,
            "mlp" => self.settings.models.mlp_train_seconds,
            _ => 0,
        };

        if seconds == 0 {
            default_epochs.max(1)
        } else {
            Self::epochs_from_seconds(seconds, default_epochs)
        }
    }

    fn min_calibration_rows(&self) -> usize {
        self.settings.models.calibration_min_rows.max(32)
    }

    fn effective_label_horizon_bars(&self) -> usize {
        if self.settings.models.label_horizon_bars > 0 {
            self.settings.models.label_horizon_bars
        } else {
            self.settings.risk.meta_label_max_hold_bars.max(1)
        }
    }

    fn holdout_pct(&self) -> f64 {
        let configured = self.settings.models.train_holdout_pct;
        if configured.is_finite() && (0.0..0.5).contains(&configured) {
            configured.max(0.05)
        } else {
            (1.0 - self.settings.models.global_train_ratio).clamp(0.05, 0.40)
        }
    }

    fn hpo_trials_for_model(&self, name: &str) -> usize {
        let canonical = canonical_model_name(name);
        self.settings
            .models
            .hpo_trials_by_model
            .get(canonical)
            .copied()
            .unwrap_or(self.settings.models.hpo_trials)
            .max(1)
    }

    fn inject_runtime_model_params(&self, name: &str, params: &mut HashMap<String, String>) {
        params.insert(
            "__hpo_backend".to_string(),
            self.settings.models.hpo_backend.clone(),
        );
        params.insert(
            "__hpo_trials".to_string(),
            self.hpo_trials_for_model(name).to_string(),
        );
        params.insert(
            "__hpo_max_rows".to_string(),
            self.settings.models.hpo_max_rows.to_string(),
        );
        params.insert(
            "__holdout_pct".to_string(),
            format!("{:.6}", self.holdout_pct()),
        );
        params.insert(
            "__conf_threshold".to_string(),
            format!("{:.6}", self.settings.models.prop_conf_threshold),
        );
        params.insert(
            "__metric_weight".to_string(),
            format!("{:.6}", self.settings.models.prop_metric_weight),
        );
        params.insert(
            "__accuracy_weight".to_string(),
            format!("{:.6}", self.settings.models.prop_accuracy_weight),
        );
        params.insert(
            "__embargo_minutes".to_string(),
            self.settings.models.embargo_minutes.to_string(),
        );
    }

    fn apply_model_param_overrides(&self, name: &str, params: &mut HashMap<String, String>) {
        if let Some(exact) = self.settings.models.model_param_overrides.get(name) {
            params.extend(
                exact
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone())),
            );
        }

        let canonical = canonical_model_name(name);
        if canonical != name
            && let Some(canonical_overrides) =
                self.settings.models.model_param_overrides.get(canonical)
        {
            for (key, value) in canonical_overrides {
                params.entry(key.clone()).or_insert_with(|| value.clone());
            }
        }
    }

    fn transformer_hidden_dim(&self) -> usize {
        log_knob_twin_disagreement(
            "models.transformer_d_model",
            self.settings.models.transformer_d_model,
            "models.transformer_hidden_dim",
            self.settings.models.transformer_hidden_dim,
        );
        self.settings
            .models
            .transformer_d_model
            .max(self.settings.models.transformer_hidden_dim)
            .max(16)
    }

    fn transformer_heads(&self) -> usize {
        log_knob_twin_disagreement(
            "models.transformer_n_heads",
            self.settings.models.transformer_n_heads,
            "models.transformer_heads",
            self.settings.models.transformer_heads,
        );
        self.settings
            .models
            .transformer_n_heads
            .max(self.settings.models.transformer_heads)
            .max(1)
    }

    fn transformer_layers(&self) -> usize {
        log_knob_twin_disagreement(
            "models.transformer_n_layers",
            self.settings.models.transformer_n_layers,
            "models.transformer_layers",
            self.settings.models.transformer_layers,
        );
        self.settings
            .models
            .transformer_n_layers
            .max(self.settings.models.transformer_layers)
            .max(1)
    }

    fn effective_training_row_budget(&self) -> Option<usize> {
        // Three row caps collapsed by `.min()`. `.min()` is the safe direction
        // — the tightest cap binds — but an operator who RAISED one of them has
        // not raised the budget, and nothing said which one held the line.
        let caps = [
            (
                "models.global_max_rows",
                self.settings.models.global_max_rows,
            ),
            (
                "models.global_max_rows_per_symbol",
                self.settings.models.global_max_rows_per_symbol,
            ),
            (
                "system.max_training_rows_per_tf",
                self.settings.system.max_training_rows_per_tf,
            ),
        ];
        let active: Vec<(&str, usize)> = caps.into_iter().filter(|(_, value)| *value > 0).collect();
        let winner: Option<(&str, usize)> = active.iter().copied().min_by_key(|(_, value)| *value);
        if let Some((winning_field, winning_value)) = winner
            && active.iter().any(|(_, value)| *value != winning_value)
        {
            let all = active
                .iter()
                .map(|(field, value)| format!("{field}={value}"))
                .collect::<Vec<_>>()
                .join(", ");
            tracing::warn!(
                target: "neoethos_models::config",
                binding_field = winning_field,
                binding_value = winning_value,
                candidates = %all,
                "training row budget: {all} — the tightest cap binds \
                 ({winning_field}={winning_value}). Raising any of the others \
                 alone will not raise the budget."
            );
        }
        winner.map(|(_, value)| value)
    }

    fn apply_training_row_budget(
        &self,
        frame: &FeatureFrame,
        labels: &[i32],
        source_row_indices: &[usize],
    ) -> Result<(FeatureFrame, Vec<i32>, Vec<usize>, Option<usize>)> {
        if frame.n_samples() != labels.len() {
            anyhow::bail!(
                "training row-budget mismatch: {} rows vs {} labels",
                frame.n_samples(),
                labels.len()
            );
        }
        if source_row_indices.len() != labels.len() {
            anyhow::bail!(
                "training row-budget source receipt mismatch: {} rows vs {} receipts",
                labels.len(),
                source_row_indices.len()
            );
        }

        let Some(cap) = self.effective_training_row_budget() else {
            return Ok((
                frame.clone(),
                labels.to_vec(),
                source_row_indices.to_vec(),
                None,
            ));
        };

        if frame.n_samples() <= cap {
            return Ok((
                frame.clone(),
                labels.to_vec(),
                source_row_indices.to_vec(),
                None,
            ));
        }

        let start = frame.n_samples().saturating_sub(cap);
        let budgeted_frame = frame.row_window(start, frame.n_samples())?;
        let budgeted_labels = labels[start..].to_vec();
        let budgeted_source_rows = source_row_indices[start..].to_vec();

        info!(
            "Applied training row budget: kept {} / {} rows",
            budgeted_frame.n_samples(),
            frame.n_samples()
        );
        Ok((
            budgeted_frame,
            budgeted_labels,
            budgeted_source_rows,
            Some(cap),
        ))
    }

    fn selected_feature_timeframes(&self, base_tf: &str) -> Result<Vec<String>> {
        let base = base_tf
            .parse::<CanonicalTimeframe>()
            .with_context(|| format!("training base timeframe {base_tf} is not canonical"))?;
        let source = self.settings.system.resolve_higher_timeframes(base_tf);

        // Use the same shared config resolver as discovery/CLI. With
        // multi-resolution disabled it excludes lower/equal timeframes; with it
        // enabled it preserves the explicit direct-timeframe list minus the base.
        let mut selected = Vec::new();
        for timeframe in &source {
            let canonical = timeframe.parse::<CanonicalTimeframe>().with_context(|| {
                format!(
                    "training feature timeframe {timeframe} is not a direct canonical timeframe"
                )
            })?;
            if canonical == base {
                continue;
            }
            if selected
                .iter()
                .any(|existing: &String| existing == canonical.as_str())
            {
                continue;
            }
            selected.push(canonical.as_str().to_owned());
        }

        Ok(selected)
    }

    fn conformal_alpha(&self) -> f32 {
        self.settings.risk.conformal_alpha.clamp(1e-4, 0.99) as f32
    }

    fn conformal_min_prediction_set(&self) -> usize {
        self.settings.risk.conformal_abstain_min_set_size.max(1)
    }

    fn fit_l1_ranked_features(
        &self,
        frame: &FeatureFrame,
        labels: &[i32],
        lease: &CpuLease,
    ) -> Result<Vec<(String, f64)>> {
        let alpha = 1.0 / self.settings.models.l1_feature_selection_c.max(1e-3);
        let mut selector = ElasticNetExpert::new(alpha, 1.0);
        selector.learning_rate = 0.03;
        selector.epochs = 400;
        selector.fit(frame, labels, lease)?;
        selector.ranked_feature_importance()
    }

    /// Sample from any frame (not necessarily the full dataset).
    /// Uses the most-recent rows within the provided frame up to the configured limit.
    fn recent_sample_from(
        &self,
        frame: &FeatureFrame,
        labels: &[i32],
    ) -> Result<(FeatureFrame, Vec<i32>, usize)> {
        let sample_limit = self.settings.models.l1_feature_selection_sample_limit.max(
            self.settings
                .models
                .l1_feature_selection_max_features
                .max(1),
        );
        let total_rows = frame.n_samples();
        let sample_rows = total_rows.min(sample_limit.max(1));
        let start = total_rows.saturating_sub(sample_rows);
        let sampled = frame.row_window(start, total_rows)?;
        let sampled_labels = labels[start..].to_vec();
        Ok((sampled, sampled_labels, start))
    }

    fn take_row_subset(
        &self,
        frame: &FeatureFrame,
        labels: &[i32],
        indices: &[usize],
    ) -> Result<(FeatureFrame, Vec<i32>)> {
        if indices.is_empty() {
            anyhow::bail!("row subset indices must not be empty");
        }
        if frame.n_samples() != labels.len() {
            anyhow::bail!(
                "row subset frame/label mismatch: {} rows vs {} labels",
                frame.n_samples(),
                labels.len()
            );
        }
        for &row in indices {
            anyhow::ensure!(
                row < frame.n_samples(),
                "row subset index {row} is outside 0..{}",
                frame.n_samples()
            );
        }
        let subset = frame.select_rows(indices)?;
        let subset_labels = indices.iter().map(|&row| labels[row]).collect();
        Ok((subset, subset_labels))
    }

    fn derive_regime_buckets(
        &self,
        base_ohlcv: &Ohlcv,
        source_row_indices: &[usize],
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let mut trend = Vec::new();
        let mut range = Vec::new();
        let mut neutral = Vec::new();

        for (local_idx, &global_idx) in source_row_indices.iter().enumerate() {
            if global_idx >= base_ohlcv.close.len()
                || global_idx >= base_ohlcv.open.len()
                || global_idx >= base_ohlcv.high.len()
                || global_idx >= base_ohlcv.low.len()
            {
                break;
            }

            let open = base_ohlcv.open[global_idx];
            let close = base_ohlcv.close[global_idx];
            let high = base_ohlcv.high[global_idx];
            let low = base_ohlcv.low[global_idx];
            let range_span = (high - low).abs().max(1e-9);
            let body_ratio = ((close - open).abs() / range_span) as f32;

            let lookback = global_idx.saturating_sub(16);
            let anchor = base_ohlcv.close[lookback];
            let drift_ratio = ((close - anchor).abs() / close.abs().max(1e-9)) as f32;

            if body_ratio >= 0.60 || drift_ratio >= 0.0035 {
                trend.push(local_idx);
            } else if body_ratio <= 0.25 && drift_ratio <= 0.0012 {
                range.push(local_idx);
            } else {
                neutral.push(local_idx);
            }
        }

        (trend, range, neutral)
    }

    fn apply_feature_selection(
        &self,
        frame: &FeatureFrame,
        labels: &[i32],
        source_row_indices: &[usize],
        base_ohlcv: &Ohlcv,
        lease: &CpuLease,
    ) -> Result<FeatureFrame> {
        if !self.settings.models.l1_feature_selection_enabled {
            return Ok(frame.clone());
        }
        if frame.n_samples() != labels.len() || labels.len() != source_row_indices.len() {
            anyhow::bail!(
                "feature selection receipt mismatch: {} rows, {} labels, {} source rows",
                frame.n_samples(),
                labels.len(),
                source_row_indices.len()
            );
        }

        let min_features = self
            .settings
            .models
            .l1_feature_selection_min_features
            .max(1);
        let max_features = self
            .settings
            .models
            .l1_feature_selection_max_features
            .max(min_features);
        if frame.n_features() <= min_features {
            return Ok(frame.clone());
        }

        // TR-1 fix: feature selection must run only on train-split (first 80%) to
        // prevent the L1 selector from seeing validation/test rows and overfitting.
        let total_rows = frame.n_samples();
        let train_end = (total_rows * 4 / 5).max(1);
        let train_frame = frame.row_window(0, train_end)?;
        let train_labels: Vec<i32> = labels[..train_end.min(labels.len())].to_vec();

        // Sample up to the configured limit from within the train portion only
        let (sampled_frame, sampled_labels, sample_start) =
            self.recent_sample_from(&train_frame, &train_labels)?;
        let sampled_source_rows = &source_row_indices
            [sample_start..sample_start.saturating_add(sampled_frame.n_samples())];
        let mut score_map = HashMap::<String, f64>::new();

        for (name, score) in self.fit_l1_ranked_features(&sampled_frame, &sampled_labels, lease)? {
            *score_map.entry(name).or_insert(0.0) += score.max(0.0);
        }

        // Source-row receipts survive every warmup/filter/budget operation, so
        // regime buckets are mapped to the exact originating OHLCV bars rather
        // than guessed from the filtered frame's local position.
        if self.settings.models.l1_feature_selection_per_regime {
            let (trend_rows, range_rows, neutral_rows) =
                self.derive_regime_buckets(base_ohlcv, sampled_source_rows);

            for rows in [trend_rows, range_rows, neutral_rows] {
                if rows.len() < min_features.max(32) {
                    continue;
                }
                let (subset_frame, subset_labels) =
                    self.take_row_subset(&sampled_frame, &sampled_labels, &rows)?;
                for (name, score) in
                    self.fit_l1_ranked_features(&subset_frame, &subset_labels, lease)?
                {
                    *score_map.entry(name).or_insert(0.0) += score.max(0.0) * 0.75;
                }
            }
        }

        let mut ranked = score_map.into_iter().collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.1.total_cmp(&left.1));

        let positive = ranked
            .iter()
            .filter(|(_, score)| *score > 1e-6)
            .count()
            .max(min_features)
            .min(max_features)
            .min(frame.n_features());

        let selected_names = ranked
            .into_iter()
            .take(positive)
            .map(|(name, _)| name)
            .collect::<Vec<_>>();

        if selected_names.is_empty() || selected_names.len() >= frame.n_features() {
            return Ok(frame.clone());
        }

        let selected_columns = selected_names
            .iter()
            .map(|name| {
                frame
                    .names
                    .iter()
                    .position(|candidate| candidate == name)
                    .with_context(|| format!("selected feature `{name}` missing from frame"))
            })
            .collect::<Result<Vec<_>>>()?;

        info!(
            "Applied L1 feature selection: kept {} / {} features",
            selected_columns.len(),
            frame.n_features()
        );
        frame.select_columns(&selected_columns)
    }

    fn apply_base_signal_filter(
        &self,
        frame: &FeatureFrame,
        labels: &[i32],
        source_row_indices: &[usize],
    ) -> Result<(FeatureFrame, Vec<i32>, Vec<usize>)> {
        if frame.n_samples() != labels.len() {
            anyhow::bail!(
                "base-signal filter row/label mismatch: {} rows vs {} labels",
                frame.n_samples(),
                labels.len()
            );
        }
        if source_row_indices.len() != labels.len() {
            anyhow::bail!(
                "base-signal filter source receipt mismatch: {} rows vs {} receipts",
                labels.len(),
                source_row_indices.len()
            );
        }

        let active_rows = labels.iter().filter(|label| **label != 0).count();
        if active_rows < 64 || active_rows == labels.len() {
            return Ok((frame.clone(), labels.to_vec(), source_row_indices.to_vec()));
        }

        let selected_rows = labels
            .iter()
            .enumerate()
            .filter_map(|(row, label)| (*label != 0).then_some(row))
            .collect::<Vec<_>>();
        let filtered_frame = frame.select_rows(&selected_rows)?;
        let filtered_labels = selected_rows
            .iter()
            .map(|&row| labels[row])
            .collect::<Vec<_>>();
        let filtered_source_rows = selected_rows
            .iter()
            .map(|&row| source_row_indices[row])
            .collect::<Vec<_>>();

        if filtered_frame.n_samples() != filtered_labels.len() || filtered_labels.is_empty() {
            anyhow::bail!(
                "base-signal filter produced inconsistent payload: {} rows vs {} labels",
                filtered_frame.n_samples(),
                filtered_labels.len()
            );
        }

        info!(
            "Applied base-signal directional filter: kept {} / {} rows",
            filtered_frame.n_samples(),
            frame.n_samples()
        );
        Ok((filtered_frame, filtered_labels, filtered_source_rows))
    }

    fn default_model_params(&self, name: &str) -> HashMap<String, String> {
        let canonical = canonical_model_name(name);
        // v0.5 ML-integration Stage 1(a): the gradient boosters seed from
        // REGULARIZED, bar-scalable defaults (shallow depth, column + row
        // subsampling, L1/L2, leaf-size floors) instead of the legacy
        // full-depth / full-data / no-shrinkage values that memorize thin-TF
        // (D1/W1/MN) triple-barrier targets. The per-(symbol,TF) bar-count
        // scaling of `n_estimators` / `num_iterations` / `iterations` /
        // `min_data_in_leaf` happens later in `train_model_dispatch` where
        // `payload.frame.n_samples()` is known. Set
        // `models.regularized_model_defaults = false` to restore the legacy
        // unregularized seeds for a controlled before/after OOS comparison.
        //
        // CRITICAL (verified): this seed map is the REAL production source for
        // the named boosters — `parse_tree_params` forwards every key here into
        // the expert's `config.params`. Editing only each booster's
        // `default_params()` would be a silent no-op because the orchestrator
        // always passes `Some(parse_tree_params(...))`.
        let reg = self.settings.models.regularized_model_defaults;
        let device = self.settings.models.tree_device_preference.clone();
        match canonical {
            "xgboost_rf" if reg => HashMap::from([
                ("variant".to_string(), "rf".to_string()),
                ("num_parallel_tree".to_string(), "64".to_string()),
                // rf is bagging — keep slightly deeper than the boosters for
                // member diversity, but still subsample rows + columns + L2.
                ("max_depth".to_string(), "5".to_string()),
                ("subsample".to_string(), "0.8".to_string()),
                ("colsample_bytree".to_string(), "0.8".to_string()),
                ("colsample_bynode".to_string(), "0.8".to_string()),
                ("min_child_weight".to_string(), "10".to_string()),
                ("reg_lambda".to_string(), "5.0".to_string()),
                ("reg_alpha".to_string(), "0.5".to_string()),
                ("gamma".to_string(), "0.5".to_string()),
                ("device".to_string(), device),
            ]),
            "xgboost_rf" => HashMap::from([
                ("variant".to_string(), "rf".to_string()),
                ("num_parallel_tree".to_string(), "64".to_string()),
                ("subsample".to_string(), "0.8".to_string()),
                ("colsample_bynode".to_string(), "0.8".to_string()),
                ("device".to_string(), device),
            ]),
            "xgboost_dart" if reg => HashMap::from([
                ("variant".to_string(), "dart".to_string()),
                ("rate_drop".to_string(), "0.1".to_string()),
                ("skip_drop".to_string(), "0.5".to_string()),
                ("max_depth".to_string(), "4".to_string()),
                ("learning_rate".to_string(), "0.03".to_string()),
                ("subsample".to_string(), "0.8".to_string()),
                ("colsample_bytree".to_string(), "0.8".to_string()),
                ("colsample_bylevel".to_string(), "0.8".to_string()),
                ("colsample_bynode".to_string(), "0.8".to_string()),
                ("min_child_weight".to_string(), "10".to_string()),
                ("reg_lambda".to_string(), "5.0".to_string()),
                ("reg_alpha".to_string(), "0.5".to_string()),
                ("gamma".to_string(), "0.5".to_string()),
                ("device".to_string(), device),
            ]),
            "xgboost_dart" => HashMap::from([
                ("variant".to_string(), "dart".to_string()),
                ("rate_drop".to_string(), "0.1".to_string()),
                ("skip_drop".to_string(), "0.5".to_string()),
                ("device".to_string(), device),
            ]),
            "catboost_alt" if reg => HashMap::from([
                ("variant".to_string(), "alt".to_string()),
                // "alt" variant — keep a touch more depth for ensemble
                // diversity than the primary catboost, still strongly L2'd.
                ("depth".to_string(), "6".to_string()),
                ("l2_leaf_reg".to_string(), "6.0".to_string()),
                ("device".to_string(), device),
            ]),
            "catboost_alt" => HashMap::from([
                ("variant".to_string(), "alt".to_string()),
                ("depth".to_string(), "10".to_string()),
                ("l2_leaf_reg".to_string(), "5.0".to_string()),
                ("device".to_string(), device),
            ]),
            "lightgbm" if reg => HashMap::from([
                ("device".to_string(), device),
                ("num_iterations".to_string(), "400".to_string()),
                ("learning_rate".to_string(), "0.03".to_string()),
                ("max_depth".to_string(), "4".to_string()),
                ("num_leaves".to_string(), "15".to_string()),
                ("min_data_in_bin".to_string(), "3".to_string()),
                // baseline; bar-scaled to max(20, bars/200) in train_model_dispatch
                ("min_data_in_leaf".to_string(), "50".to_string()),
                ("feature_fraction".to_string(), "0.8".to_string()),
                ("bagging_fraction".to_string(), "0.8".to_string()),
                ("bagging_freq".to_string(), "1".to_string()),
                ("min_gain_to_split".to_string(), "0.01".to_string()),
                ("lambda_l1".to_string(), "0.5".to_string()),
                ("lambda_l2".to_string(), "5.0".to_string()),
            ]),
            "lightgbm" => HashMap::from([
                ("device".to_string(), device),
                ("num_iterations".to_string(), "400".to_string()),
                ("learning_rate".to_string(), "0.05".to_string()),
                ("max_depth".to_string(), "8".to_string()),
                ("num_leaves".to_string(), "31".to_string()),
            ]),
            "xgboost" if reg => HashMap::from([
                ("device".to_string(), device),
                // baseline; bar-scaled to 200/400/800 in train_model_dispatch
                ("n_estimators".to_string(), "400".to_string()),
                ("max_depth".to_string(), "4".to_string()),
                ("learning_rate".to_string(), "0.03".to_string()),
                ("subsample".to_string(), "0.8".to_string()),
                ("colsample_bytree".to_string(), "0.8".to_string()),
                ("colsample_bylevel".to_string(), "0.8".to_string()),
                ("colsample_bynode".to_string(), "0.8".to_string()),
                ("min_child_weight".to_string(), "10".to_string()),
                ("reg_lambda".to_string(), "5.0".to_string()),
                ("reg_alpha".to_string(), "0.5".to_string()),
                ("gamma".to_string(), "0.5".to_string()),
            ]),
            "xgboost" => HashMap::from([
                ("device".to_string(), device),
                ("n_estimators".to_string(), "800".to_string()),
                ("max_depth".to_string(), "8".to_string()),
                ("learning_rate".to_string(), "0.05".to_string()),
            ]),
            "catboost" if reg => HashMap::from([
                ("device".to_string(), device),
                // baseline; bar-scaled to 200/400/800 in train_model_dispatch
                ("iterations".to_string(), "400".to_string()),
                ("depth".to_string(), "4".to_string()),
                ("learning_rate".to_string(), "0.03".to_string()),
                ("l2_leaf_reg".to_string(), "6.0".to_string()),
            ]),
            "catboost" => HashMap::from([
                ("device".to_string(), device),
                ("iterations".to_string(), "500".to_string()),
                ("depth".to_string(), "8".to_string()),
                ("learning_rate".to_string(), "0.05".to_string()),
            ]),
            "mlp" => HashMap::from([
                ("device".to_string(), self.preferred_burn_device_policy()),
                (
                    "hidden_dim".to_string(),
                    self.settings.models.nf_hidden_dim.to_string(),
                ),
                ("n_layers".to_string(), "3".to_string()),
                ("dropout".to_string(), "0.10".to_string()),
                (
                    "batch_size".to_string(),
                    self.settings.models.train_batch_size.max(8).to_string(),
                ),
                (
                    "max_epochs".to_string(),
                    self.max_epochs_for_model("mlp", 100).to_string(),
                ),
            ]),
            "transformer" => {
                let replica_idx = transformer_replica_index(name).unwrap_or(1);
                let replica_offset = replica_idx.saturating_sub(1);
                let dropout = (self.settings.models.transformer_dropout
                    + replica_offset as f64 * 0.01)
                    .clamp(0.0, 0.35);
                let hidden_dim = self.transformer_hidden_dim();
                let n_heads = self.transformer_heads();
                let n_layers = self.transformer_layers();
                HashMap::from([
                    ("hidden_dim".to_string(), hidden_dim.to_string()),
                    ("n_heads".to_string(), n_heads.to_string()),
                    ("n_layers".to_string(), n_layers.to_string()),
                    (
                        "token_count".to_string(),
                        self.settings
                            .models
                            .transformer_seq_len
                            .clamp(2, 64)
                            .to_string(),
                    ),
                    ("dim_ff".to_string(), (hidden_dim * 2).to_string()),
                    ("dropout".to_string(), format!("{dropout:.4}")),
                    (
                        "batch_size".to_string(),
                        self.settings.models.train_batch_size.max(8).to_string(),
                    ),
                    (
                        "max_epochs".to_string(),
                        self.max_epochs_for_model("transformer", 120).to_string(),
                    ),
                    (
                        "seed".to_string(),
                        (42_u64 + replica_offset as u64 * 97).to_string(),
                    ),
                    ("ensemble_member".to_string(), replica_idx.to_string()),
                    ("device".to_string(), self.preferred_burn_device_policy()),
                ])
            }
            "nbeats" => HashMap::from([
                ("device".to_string(), self.preferred_burn_device_policy()),
                (
                    "hidden_dim".to_string(),
                    self.settings.models.nbeats_hidden_dim.to_string(),
                ),
                ("n_blocks".to_string(), "4".to_string()),
                ("dropout".to_string(), "0.05".to_string()),
                (
                    "batch_size".to_string(),
                    self.settings.models.train_batch_size.max(8).to_string(),
                ),
                (
                    "max_epochs".to_string(),
                    self.max_epochs_for_model("nbeats", 100).to_string(),
                ),
            ]),
            "tide" => HashMap::from([
                ("device".to_string(), self.preferred_burn_device_policy()),
                (
                    "hidden_dim".to_string(),
                    self.settings.models.tide_hidden_dim.to_string(),
                ),
                ("dropout".to_string(), "0.05".to_string()),
                (
                    "batch_size".to_string(),
                    self.settings.models.train_batch_size.max(8).to_string(),
                ),
                (
                    "max_epochs".to_string(),
                    self.max_epochs_for_model("tide", 100).to_string(),
                ),
            ]),
            "tabnet" => HashMap::from([
                ("device".to_string(), self.preferred_burn_device_policy()),
                (
                    "hidden_dim".to_string(),
                    self.settings.models.tabnet_hidden_dim.to_string(),
                ),
                ("n_steps".to_string(), "5".to_string()),
                ("relaxation_factor".to_string(), "1.5".to_string()),
                (
                    "batch_size".to_string(),
                    self.settings.models.train_batch_size.max(8).to_string(),
                ),
                (
                    "max_epochs".to_string(),
                    self.max_epochs_for_model("tabnet", 120).to_string(),
                ),
            ]),
            "kan" => HashMap::from([
                ("device".to_string(), self.preferred_burn_device_policy()),
                (
                    "hidden_dim".to_string(),
                    self.settings.models.kan_hidden_dim.to_string(),
                ),
                ("n_layers".to_string(), "3".to_string()),
                (
                    "grid_size".to_string(),
                    self.settings.models.kan_grid_size.to_string(),
                ),
                ("dropout".to_string(), "0.05".to_string()),
                (
                    "batch_size".to_string(),
                    self.settings.models.train_batch_size.max(8).to_string(),
                ),
                (
                    "max_epochs".to_string(),
                    self.max_epochs_for_model("kan", 100).to_string(),
                ),
            ]),
            "patchtst" => HashMap::from([
                ("device".to_string(), self.preferred_burn_device_policy()),
                (
                    "hidden_dim".to_string(),
                    self.transformer_hidden_dim().to_string(),
                ),
                ("n_heads".to_string(), self.transformer_heads().to_string()),
                (
                    "n_layers".to_string(),
                    self.transformer_layers().to_string(),
                ),
                (
                    "dim_ff".to_string(),
                    (self.transformer_hidden_dim() * 2).to_string(),
                ),
                ("dropout".to_string(), "0.10".to_string()),
                (
                    "batch_size".to_string(),
                    self.settings.models.train_batch_size.max(8).to_string(),
                ),
                (
                    "max_epochs".to_string(),
                    self.max_epochs_for_model("patchtst", 120).to_string(),
                ),
            ]),
            "timesnet" => HashMap::from([
                ("device".to_string(), self.preferred_burn_device_policy()),
                (
                    "hidden_dim".to_string(),
                    self.settings.models.nf_hidden_dim.to_string(),
                ),
                ("dropout".to_string(), "0.05".to_string()),
                (
                    "batch_size".to_string(),
                    self.settings.models.train_batch_size.max(8).to_string(),
                ),
                (
                    "max_epochs".to_string(),
                    self.max_epochs_for_model("timesnet", 100).to_string(),
                ),
            ]),
            "nbeatsx_nf" => HashMap::from([
                ("device".to_string(), self.preferred_burn_device_policy()),
                (
                    "hidden_dim".to_string(),
                    self.settings.models.nbeats_hidden_dim.to_string(),
                ),
                ("n_blocks".to_string(), "4".to_string()),
                (
                    "batch_size".to_string(),
                    self.settings.models.train_batch_size.max(8).to_string(),
                ),
                (
                    "max_epochs".to_string(),
                    self.max_epochs_for_model("nbeatsx_nf", 100).to_string(),
                ),
            ]),
            "tide_nf" => HashMap::from([
                ("device".to_string(), self.preferred_burn_device_policy()),
                (
                    "hidden_dim".to_string(),
                    self.settings.models.tide_hidden_dim.to_string(),
                ),
                ("dropout".to_string(), "0.05".to_string()),
                (
                    "batch_size".to_string(),
                    self.settings.models.train_batch_size.max(8).to_string(),
                ),
                (
                    "max_epochs".to_string(),
                    self.max_epochs_for_model("tide_nf", 100).to_string(),
                ),
            ]),
            "meta_blender" => HashMap::from([("backend".to_string(), "xgboost".to_string())]),
            "probability_calibrator" => HashMap::from([
                (
                    "method".to_string(),
                    self.settings.models.calibration_method.clone(),
                ),
                (
                    "min_rows".to_string(),
                    self.min_calibration_rows().to_string(),
                ),
            ]),
            "conformal_gate" => HashMap::from([
                (
                    "method".to_string(),
                    self.settings.models.calibration_method.clone(),
                ),
                (
                    "alpha".to_string(),
                    format!("{:.6}", self.conformal_alpha()),
                ),
                (
                    "min_prediction_set".to_string(),
                    self.conformal_min_prediction_set().to_string(),
                ),
                (
                    "min_rows".to_string(),
                    self.min_calibration_rows().to_string(),
                ),
            ]),
            "meta_stack" => HashMap::from([
                (
                    "method".to_string(),
                    self.settings.models.calibration_method.clone(),
                ),
                (
                    "alpha".to_string(),
                    format!("{:.6}", self.conformal_alpha()),
                ),
                (
                    "min_prediction_set".to_string(),
                    self.conformal_min_prediction_set().to_string(),
                ),
                (
                    "min_rows".to_string(),
                    self.min_calibration_rows().to_string(),
                ),
            ]),
            "exit_agent" => HashMap::from([
                ("device".to_string(), self.preferred_burn_device_policy()),
                (
                    "hidden_dim".to_string(),
                    self.settings.models.exit_agent_hidden_dim.to_string(),
                ),
                (
                    "gamma".to_string(),
                    format!("{:.6}", self.settings.models.exit_agent_gamma),
                ),
                (
                    "epsilon".to_string(),
                    format!("{:.6}", self.settings.models.exit_agent_epsilon),
                ),
                (
                    "epsilon_min".to_string(),
                    format!("{:.6}", self.settings.models.exit_agent_epsilon_min),
                ),
                (
                    "epsilon_decay".to_string(),
                    format!("{:.6}", self.settings.models.exit_agent_epsilon_decay),
                ),
                (
                    "memory_capacity".to_string(),
                    self.settings.models.exit_agent_memory_capacity.to_string(),
                ),
                (
                    "reward_horizon".to_string(),
                    if self.settings.models.exit_agent_reward_horizon == 0 {
                        self.settings
                            .risk
                            .triple_barrier_max_bars
                            .clamp(6, 64)
                            .to_string()
                    } else {
                        self.settings.models.exit_agent_reward_horizon.to_string()
                    },
                ),
                (
                    "warmup_steps".to_string(),
                    if self.settings.models.exit_agent_warmup_steps == 0 {
                        Self::epochs_from_seconds(self.settings.models.rl_train_seconds, 96)
                            .to_string()
                    } else {
                        self.settings.models.exit_agent_warmup_steps.to_string()
                    },
                ),
            ]),
            // Soft Actor-Critic (discrete) — RL entry/direction policy.
            // Reuses the shared RL hyperparameter knobs (gamma / learning
            // rate / horizon / batch / epochs). `tau` and
            // `target_entropy_scale` are SAC-specific and use faithful
            // defaults from Christodoulou (2019).
            "sac" => HashMap::from([
                ("device".to_string(), self.preferred_burn_device_policy()),
                (
                    "hidden_dim".to_string(),
                    self.settings
                        .models
                        .rl_network_arch
                        .first()
                        .copied()
                        .unwrap_or(256)
                        .max(8)
                        .to_string(),
                ),
                (
                    "gamma".to_string(),
                    format!("{:.6}", self.settings.models.rl_gamma),
                ),
                ("tau".to_string(), "0.010000".to_string()),
                (
                    "learning_rate".to_string(),
                    format!("{:.6}", self.settings.models.rl_learning_rate),
                ),
                ("target_entropy_scale".to_string(), "0.980000".to_string()),
                (
                    "epochs".to_string(),
                    Self::epochs_from_seconds(self.settings.models.rl_train_seconds, 32)
                        .to_string(),
                ),
                (
                    "batch_size".to_string(),
                    self.settings.models.train_batch_size.max(32).to_string(),
                ),
                (
                    "reward_horizon".to_string(),
                    if self.settings.models.rl_reward_horizon == 0 {
                        self.settings
                            .risk
                            .triple_barrier_max_bars
                            .clamp(6, 64)
                            .to_string()
                    } else {
                        self.settings.models.rl_reward_horizon.to_string()
                    },
                ),
                (
                    "episode_len".to_string(),
                    if self.settings.models.rl_episode_len == 0 {
                        self.settings
                            .models
                            .transformer_seq_len
                            .clamp(24, 256)
                            .to_string()
                    } else {
                        self.settings.models.rl_episode_len.to_string()
                    },
                ),
            ]),
            "online_pa" => HashMap::from([
                ("c".to_string(), "1.0".to_string()),
                (
                    "epochs".to_string(),
                    Self::epochs_from_seconds(self.settings.models.rl_train_seconds / 4, 8)
                        .to_string(),
                ),
            ]),
            "online_hoeffding" => HashMap::from([
                ("n_steps".to_string(), "24".to_string()),
                ("learning_rate".to_string(), "0.05".to_string()),
                ("feature_subsample_rate".to_string(), "0.8".to_string()),
                ("max_depth".to_string(), "5".to_string()),
                ("n_bins".to_string(), "32".to_string()),
                ("grace_period".to_string(), "32".to_string()),
                (
                    "drift_detector".to_string(),
                    if self.settings.risk.feature_drift_threshold <= 0.20 {
                        "adwin".to_string()
                    } else {
                        "page_hinkley".to_string()
                    },
                ),
                (
                    "drift_delta".to_string(),
                    format!(
                        "{:.6}",
                        (self.settings.risk.feature_drift_threshold / 100.0).clamp(0.0005, 0.02)
                    ),
                ),
            ]),
            "isolation_forest" => HashMap::from([
                ("n_trees".to_string(), "128".to_string()),
                ("sample_size".to_string(), "256".to_string()),
            ]),
            "genetic" => HashMap::from([
                (
                    "population".to_string(),
                    self.settings
                        .models
                        .prop_search_population
                        .max(16)
                        .to_string(),
                ),
                (
                    "generations".to_string(),
                    self.settings
                        .models
                        .prop_search_generations
                        .max(1)
                        .to_string(),
                ),
                (
                    "max_indicators".to_string(),
                    if self.settings.models.prop_search_max_indicators == 0 {
                        "64".to_string()
                    } else {
                        self.settings.models.prop_search_max_indicators.to_string()
                    },
                ),
                (
                    "portfolio_size".to_string(),
                    self.settings
                        .models
                        .prop_search_portfolio_size
                        .clamp(4, self.settings.models.prop_search_population.max(4))
                        .to_string(),
                ),
                (
                    "parent_selection".to_string(),
                    self.settings.models.prop_search_parent_selection.clone(),
                ),
                (
                    "survivor_selection".to_string(),
                    self.settings.models.prop_search_survivor_selection.clone(),
                ),
                (
                    "survivor_fraction".to_string(),
                    format!("{:.6}", self.settings.models.prop_search_survivor_fraction),
                ),
                (
                    "immigrant_fraction".to_string(),
                    format!("{:.6}", self.settings.models.prop_search_immigrant_fraction),
                ),
                (
                    "selection_temperature".to_string(),
                    format!(
                        "{:.6}",
                        self.settings.models.prop_search_selection_temperature
                    ),
                ),
                (
                    "device".to_string(),
                    self.settings.models.prop_search_device.clone(),
                ),
                (
                    "checkpoint".to_string(),
                    self.settings
                        .models
                        .prop_search_checkpoint
                        .to_string_lossy()
                        .to_string(),
                ),
                (
                    "async".to_string(),
                    self.settings.models.prop_search_async.to_string(),
                ),
                (
                    "async_wait".to_string(),
                    self.settings.models.prop_search_async_wait.to_string(),
                ),
                (
                    "tournament_size".to_string(),
                    if self.settings.models.prop_search_tournament_size == 0 {
                        (self.settings.models.prop_search_population.max(16) / 10)
                            .max(3)
                            .to_string()
                    } else {
                        self.settings
                            .models
                            .prop_search_tournament_size
                            .max(3)
                            .to_string()
                    },
                ),
            ]),
            "dqn" => {
                // rlkit is the explicit native DQN backend. Legacy RLlib/Ray
                // requests are rejected before dispatch and never arrive here.
                HashMap::from([
                    ("backend".to_string(), "rlkit".to_string()),
                    (
                        "epochs".to_string(),
                        Self::epochs_from_seconds(self.settings.models.rl_train_seconds, 48)
                            .to_string(),
                    ),
                    (
                        "max_steps".to_string(),
                        (self.settings.models.rl_timesteps / 10_000)
                            .clamp(128, 4096)
                            .to_string(),
                    ),
                    (
                        "batch_size".to_string(),
                        self.settings.models.train_batch_size.max(32).to_string(),
                    ),
                    (
                        "state_bins".to_string(),
                        self.settings.models.rl_state_bins.to_string(),
                    ),
                    (
                        "state_encoding".to_string(),
                        self.settings.models.rl_state_encoding.clone(),
                    ),
                    (
                        "update_interval".to_string(),
                        if self.settings.models.rl_update_interval == 0 {
                            self.settings
                                .models
                                .rl_parallel_envs
                                .clamp(8, 128)
                                .to_string()
                        } else {
                            self.settings.models.rl_update_interval.max(1).to_string()
                        },
                    ),
                    (
                        "update_freq".to_string(),
                        if self.settings.models.rl_update_freq == 0 {
                            self.settings
                                .models
                                .rl_eval_episodes
                                .clamp(1, 16)
                                .to_string()
                        } else {
                            self.settings.models.rl_update_freq.max(1).to_string()
                        },
                    ),
                    (
                        "learning_rate".to_string(),
                        format!("{:.6}", self.settings.models.rl_learning_rate),
                    ),
                    (
                        "gamma".to_string(),
                        format!("{:.6}", self.settings.models.rl_gamma),
                    ),
                    (
                        "epsilon_start".to_string(),
                        format!("{:.6}", self.settings.models.rl_epsilon_start),
                    ),
                    (
                        "epsilon_end".to_string(),
                        format!("{:.6}", self.settings.models.rl_epsilon_end),
                    ),
                    (
                        "epsilon_decay".to_string(),
                        format!("{:.6}", self.settings.models.rl_epsilon_decay),
                    ),
                    (
                        "buffer_capacity".to_string(),
                        if self.settings.models.rl_buffer_capacity == 0 {
                            (self.settings.models.train_batch_size.max(32) * 1024).to_string()
                        } else {
                            self.settings.models.rl_buffer_capacity.to_string()
                        },
                    ),
                    (
                        "parallel_envs".to_string(),
                        self.settings.models.rl_parallel_envs.max(1).to_string(),
                    ),
                    (
                        "eval_episodes".to_string(),
                        self.settings.models.rl_eval_episodes.max(1).to_string(),
                    ),
                    (
                        "reward_horizon".to_string(),
                        if self.settings.models.rl_reward_horizon == 0 {
                            self.settings
                                .risk
                                .triple_barrier_max_bars
                                .clamp(6, 64)
                                .to_string()
                        } else {
                            self.settings.models.rl_reward_horizon.to_string()
                        },
                    ),
                    (
                        "episode_len".to_string(),
                        if self.settings.models.rl_episode_len == 0 {
                            self.settings
                                .models
                                .transformer_seq_len
                                .clamp(24, 256)
                                .to_string()
                        } else {
                            self.settings.models.rl_episode_len.to_string()
                        },
                    ),
                    (
                        "ray_tune_max_concurrency".to_string(),
                        self.settings
                            .models
                            .ray_tune_max_concurrency
                            .max(1)
                            .to_string(),
                    ),
                    ("device".to_string(), self.settings.system.device.clone()),
                    (
                        "hidden_dims".to_string(),
                        self.settings
                            .models
                            .rl_network_arch
                            .iter()
                            .map(|value| value.to_string())
                            .collect::<Vec<_>>()
                            .join(","),
                    ),
                ])
            }
            "swarm_forecaster" => HashMap::from([
                (
                    "memory_limit_mb".to_string(),
                    format!("{:.2}", self.settings.models.swarm_memory_limit_mb),
                ),
                (
                    "horizon".to_string(),
                    if self.settings.models.swarm_horizon == 0 {
                        self.settings
                            .risk
                            .triple_barrier_max_bars
                            .clamp(8, 64)
                            .to_string()
                    } else {
                        self.settings.models.swarm_horizon.to_string()
                    },
                ),
                (
                    "frequency".to_string(),
                    self.settings.models.swarm_frequency.clone(),
                ),
                (
                    "strategy".to_string(),
                    self.settings.models.swarm_strategy.clone(),
                ),
                (
                    "accuracy_target".to_string(),
                    format!(
                        "{:.4}",
                        self.settings.models.prop_conf_threshold.clamp(0.5, 0.99)
                    ),
                ),
                (
                    "latency_ms".to_string(),
                    if self.settings.models.swarm_latency_ms == 0 {
                        self.settings
                            .system
                            .poll_interval_seconds
                            .saturating_mul(1000)
                            .to_string()
                    } else {
                        self.settings.models.swarm_latency_ms.to_string()
                    },
                ),
                (
                    "online_learning".to_string(),
                    self.settings.models.swarm_online_learning.to_string(),
                ),
                (
                    "interpretability_needed".to_string(),
                    self.settings
                        .models
                        .swarm_interpretability_needed
                        .to_string(),
                ),
            ]),
            "neuro_evo" => HashMap::from([
                ("backend".to_string(), "crfmnes_cpu".to_string()),
                ("device".to_string(), self.settings.system.device.clone()),
                (
                    "hidden_dim".to_string(),
                    self.settings.models.evo_hidden_size.to_string(),
                ),
                (
                    "sigma".to_string(),
                    format!("{:.6}", self.settings.models.evo_sigma),
                ),
                (
                    "generations".to_string(),
                    Self::epochs_from_seconds(self.settings.models.evo_train_seconds, 24)
                        .to_string(),
                ),
                (
                    "population".to_string(),
                    self.settings.models.evo_population.max(16).to_string(),
                ),
                (
                    "islands".to_string(),
                    self.settings.models.evo_islands.max(1).to_string(),
                ),
            ]),
            "neat" => HashMap::from([
                (
                    "population".to_string(),
                    self.settings.models.evo_population.max(48).to_string(),
                ),
                (
                    "generations".to_string(),
                    Self::epochs_from_seconds(self.settings.models.evo_train_seconds, 48)
                        .to_string(),
                ),
                (
                    "mutation_rate".to_string(),
                    format!(
                        "{:.6}",
                        (self.settings.models.evo_sigma * 2.4).clamp(0.2, 1.2)
                    ),
                ),
                ("species_elitism".to_string(), "0".to_string()),
                (
                    "compatibility_threshold".to_string(),
                    format!(
                        "{:.6}",
                        (1.5 + self.settings.models.evo_sigma * 4.0).clamp(1.5, 4.0)
                    ),
                ),
                ("immigrant_fraction".to_string(), "0.100000".to_string()),
                ("seed".to_string(), "42".to_string()),
                ("device".to_string(), self.settings.system.device.clone()),
            ]),
            _ => HashMap::new(),
        }
    }

    fn map_model_type(&self, name: &str) -> Result<ModelType> {
        match canonical_model_name(name) {
            "lightgbm" => Ok(ModelType::LightGBM),
            "xgboost" | "xgboost_rf" | "xgboost_dart" => Ok(ModelType::XGBoost),
            "catboost" | "catboost_alt" => Ok(ModelType::CatBoost),
            "sklears_tree" => Ok(ModelType::SklearsTree),
            "mlp" => Ok(ModelType::MLP),
            "nbeats" => Ok(ModelType::NBeats),
            "nbeatsx_nf" => Ok(ModelType::NBeatsxNf),
            "tide" => Ok(ModelType::TiDE),
            "tide_nf" => Ok(ModelType::TiDENf),
            "tabnet" => Ok(ModelType::TabNet),
            "kan" => Ok(ModelType::KAN),
            "transformer" => Ok(ModelType::Transformer),
            "patchtst" => Ok(ModelType::PatchTST),
            "timesnet" => Ok(ModelType::TimesNet),
            "elasticnet" => Ok(ModelType::ElasticNet),
            "logistic" => Ok(ModelType::Logistic),
            "bayes_logit" => Ok(ModelType::BayesianLogit),
            "meta_blender" => Ok(ModelType::MetaBlender),
            "probability_calibrator" => Ok(ModelType::ProbabilityCalibrator),
            "conformal_gate" => Ok(ModelType::ConformalGate),
            "meta_stack" => Ok(ModelType::MetaStack),
            "exit_agent" => Ok(ModelType::ExitAgent),
            "sac" => Ok(ModelType::SacAgent),
            "online_pa" => Ok(ModelType::OnlinePassiveAggressive),
            "online_hoeffding" => Ok(ModelType::OnlineHoeffding),
            "isolation_forest" => Ok(ModelType::IsolationForest),
            "dqn" => Ok(ModelType::Dqn),
            "swarm_forecaster" => Ok(ModelType::SwarmForecaster),
            "genetic" => Ok(ModelType::Genetic),
            "neuro_evo" => Ok(ModelType::NeuroEvo),
            "neat" => Ok(ModelType::Neat),
            "hmm_regime" => Ok(ModelType::HmmRegime),
            other => anyhow::bail!(
                "Model `{other}` does not have a concrete ModelType mapping in the orchestrator"
            ),
        }
    }

    /// Derive per-bar training labels from a barrier race on future bars.
    ///
    /// The bracket geometry is a config choice — `models.label_geometry`,
    /// resolved by [`resolve_label_geometry`] and recorded in every training
    /// profile/artifact:
    ///
    /// * `symmetric` (default): target distance == stop distance, and the
    ///   round-trip cost pushes BOTH barriers outward, so the label is a fair
    ///   direction race net of costs. This matches how the horizon fallback
    ///   below has always charged the cost. Real EURUSD M15 bars measure a
    ///   ~0.49/0.51 class prior.
    /// * `asymmetric`: the pre-2026-08-08 long-bracket geometry (40-pip
    ///   target vs 20-pip stop via the `risk.meta_label_fixed_*` floors, rr
    ///   floored at `risk.min_risk_reward`, both barriers shifted UP by the
    ///   cost). It manufactures a ~66/34 prior on M15 — kept only for
    ///   comparison runs against old artifacts.
    ///
    /// # KNOWN LIMITATION — label semantics vs the Sell mapping
    ///
    /// Do not rediscover this: in `asymmetric` mode `-1` means "a LONG got
    /// stopped out", NOT "a short would have won" — the stop is 20 pips away
    /// while a short's own target would be 40 pips away, and both barriers
    /// are shifted by a LONG's costs. Yet the artifact maps `-1` to Sell, so
    /// every asymmetric Sell signal was trained on "longs lost here", a
    /// strictly weaker claim. `symmetric` mode narrows this defect — the race
    /// is equidistant and cost is charged both ways, so `-1` at least means
    /// "the down-move cleared a short's distance-plus-cost" — but it does NOT
    /// eliminate it:
    ///
    /// * there is still no abstain class: `0` conflates "no barrier reached"
    ///   with "genuinely flat", and models treat it as a third direction
    ///   rather than "stay out";
    /// * a single label still serves both directions; a correct label set
    ///   would derive LONG and SHORT outcomes independently (each side with
    ///   its own entry price, spread charge, and bracket), plus an explicit
    ///   abstain when neither side clears its costs, and the artifact would
    ///   map model outputs to {Buy, Sell, Abstain} trained on exactly those
    ///   three outcomes.
    ///
    /// Changing the label cardinality/semantics is a training-format break
    /// across every consumer (recorded 2026-08-08, deliberately deferred);
    /// the geometry fix above is what could land safely tonight.
    fn derive_labels(&self, ohlcv: &Ohlcv, symbol: &str) -> Result<Vec<i32>> {
        let broker_truth = neoethos_core::current_broker_financial_truth_capability_v1()
            .require(neoethos_core::BrokerFinancialOperationV1::HistoricalEvaluation)
            .map_err(anyhow::Error::new)?;
        let pip_size = broker_truth
            .exact_pip_size_v1(symbol)
            .map_err(anyhow::Error::new)?;
        self.derive_labels_unchecked_test_oracle(
            ohlcv,
            symbol,
            pip_size,
            self.configured_label_round_trip_cost_pips()?,
        )
    }

    fn derive_labels_with_screening_costs(
        &self,
        ohlcv: &Ohlcv,
        symbol: &str,
        pip_size: f64,
        round_trip_cost_pips: f64,
    ) -> Result<Vec<i32>> {
        anyhow::ensure!(
            pip_size.is_finite() && pip_size > 0.0,
            "canonical training screening labels require a positive broker pip size"
        );
        anyhow::ensure!(
            round_trip_cost_pips.is_finite() && round_trip_cost_pips >= 0.0,
            "canonical training labels require non-negative V2 screening costs in pips"
        );
        self.derive_labels_unchecked_test_oracle(ohlcv, symbol, pip_size, round_trip_cost_pips)
    }

    fn configured_label_round_trip_cost_pips(&self) -> Result<f64> {
        let commission_account_per_lot = self.settings.risk.commission_per_lot;
        anyhow::ensure!(
            commission_account_per_lot.is_finite() && commission_account_per_lot >= 0.0,
            "settings-only training requires a finite, non-negative account-currency commission assumption"
        );
        anyhow::ensure!(
            commission_account_per_lot == 0.0,
            "settings-only training cannot convert account-currency commission into pips; use the versioned canonical screening-cost contract"
        );
        let full_spread_pips = self.settings.risk.backtest_spread_pips;
        let slippage_pips_per_fill = self.settings.risk.slippage_pips;
        anyhow::ensure!(
            full_spread_pips.is_finite() && full_spread_pips >= 0.0,
            "settings-only training requires a finite, non-negative full-spread assumption"
        );
        anyhow::ensure!(
            slippage_pips_per_fill.is_finite() && slippage_pips_per_fill >= 0.0,
            "settings-only training requires a finite, non-negative per-fill slippage assumption"
        );
        let round_trip_cost_pips = full_spread_pips + 2.0 * slippage_pips_per_fill;
        Ok(round_trip_cost_pips)
    }

    /// Pure label-geometry oracle. This is deliberately private: production
    /// training may reach it through [`Self::derive_labels`] after the typed
    /// broker-truth capability check, or through the receipt-bound canonical
    /// research entry after its exact broker pip size is validated. Unit tests
    /// use the same body to retain adversarial formula coverage without
    /// weakening either boundary.
    fn derive_labels_unchecked_test_oracle(
        &self,
        ohlcv: &Ohlcv,
        symbol: &str,
        pip_size: f64,
        round_trip_cost_pips: f64,
    ) -> Result<Vec<i32>> {
        anyhow::ensure!(
            !symbol.trim().is_empty()
                && symbol.len() <= 128
                && !symbol.chars().any(char::is_control),
            "derive_labels requires one bounded symbol identity"
        );
        let n = ohlcv.close.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if ohlcv.open.len() != n {
            anyhow::bail!(
                "derive_labels requires aligned OHLCV series: open={} close={}",
                ohlcv.open.len(),
                n
            );
        }
        if ohlcv.high.len() != n {
            anyhow::bail!(
                "derive_labels requires aligned OHLCV series: high={} close={}",
                ohlcv.high.len(),
                n
            );
        }
        if ohlcv.low.len() != n {
            anyhow::bail!(
                "derive_labels requires aligned OHLCV series: low={} close={}",
                ohlcv.low.len(),
                n
            );
        }

        let mut labels = vec![0; n];
        let hold_bars = self.effective_label_horizon_bars();
        let atr_period = self.settings.risk.atr_period.max(2);
        let min_distance = self.settings.risk.meta_label_min_dist.max(1e-6);
        // What a round trip costs, in price.
        //
        // The barriers sat at `close +/- distance` and charged nothing, so a
        // label said "price moved" where the gene side asks "did the trade make
        // money". Canonical discovery charges a V2 screening envelope: full
        // spread once, per-fill slippage twice, and per-fill account-currency
        // commission twice after conversion to pips. The receipt-bound
        // training entry supplies that same complete screening number. It is
        // not historical quote replay. Before this boundary, the two halves of
        // the system were optimising different objectives, and on EURUSD M15
        // every directional model collapsed to a constant: three unrelated
        // architectures reporting accuracy identical to sixteen significant
        // figures, which is what a correct answer to a question about costless
        // price movement looks like.
        //
        // A long enters at the ask and leaves at the bid, so it is behind by the
        // full spread from the first tick: profit needs `move > cost`. Both
        // barriers therefore shift UP by the cost — the target gets further
        // away and the stop gets nearer, which is what the trade actually faces.
        if !(pip_size.is_finite() && pip_size > 0.0) {
            anyhow::bail!("derive_labels requires an exact positive broker pip size");
        }
        if !(round_trip_cost_pips.is_finite() && round_trip_cost_pips >= 0.0) {
            anyhow::bail!("derive_labels requires non-negative screening round-trip cost pips");
        }
        let round_trip_cost = round_trip_cost_pips * pip_size;
        if !round_trip_cost.is_finite() {
            anyhow::bail!("derive_labels round-trip cost overflows price space");
        }
        // 💰 MONEY PATH. Two fields, one number, silently `.max()`ed: lowering
        // either one alone does NOT lower the stop distance, and the label set
        // this produces is what every model learns to trade.
        log_money_knob_twin_disagreement(
            "models.label_stop_atr_multiplier",
            self.settings.models.label_stop_atr_multiplier,
            "risk.atr_stop_multiplier",
            self.settings.risk.atr_stop_multiplier,
            "label stop distance (ATR multiples)",
        );
        let atr_stop_multiplier = self
            .settings
            .models
            .label_stop_atr_multiplier
            .max(self.settings.risk.atr_stop_multiplier)
            .max(0.1);
        let geometry = resolve_label_geometry(&self.settings)?;
        let base_rr = geometry.rr_floor;
        let fixed_sl = geometry.fixed_stop;
        let fixed_tp = geometry.fixed_target;
        let true_ranges = compute_true_ranges(ohlcv);
        let use_triple_barrier = self.settings.models.label_use_triple_barrier;
        let neutral_band_fraction = self
            .settings
            .models
            .label_neutral_band_atr_fraction
            .clamp(0.05, 1.0);

        for (i, slot) in labels.iter_mut().enumerate() {
            let entry = ohlcv.close[i];
            if !entry.is_finite() || entry.abs() <= f64::EPSILON || i + 1 >= n {
                continue;
            }

            let atr = trailing_average(&true_ranges, i, atr_period).max(min_distance);
            let stop_distance = fixed_sl.max(atr * atr_stop_multiplier).max(min_distance);
            let take_profit_distance = fixed_tp.max(stop_distance * base_rr).max(min_distance);
            let horizon_end = (i + hold_bars).min(n - 1);
            if use_triple_barrier {
                // Where the cost sits depends on what the label claims.
                //
                // Symmetric mode is a direction race: EACH side must clear
                // its own distance PLUS the round trip, so both barriers move
                // outward — the same both-ways charge the horizon fallback
                // below has always applied. Asymmetric (legacy) mode is a
                // long bracket: the long is behind by the spread from the
                // first tick, so both barriers shift UP — which also means
                // its `-1` is "the long got stopped", see the doc comment.
                let (upper_barrier, lower_barrier) = match geometry.mode {
                    LabelGeometryMode::Symmetric => (
                        entry + take_profit_distance + round_trip_cost,
                        entry - stop_distance - round_trip_cost,
                    ),
                    LabelGeometryMode::Asymmetric => (
                        entry + take_profit_distance + round_trip_cost,
                        entry - stop_distance + round_trip_cost,
                    ),
                };

                for forward_idx in (i + 1)..=horizon_end {
                    let high = ohlcv.high[forward_idx];
                    let low = ohlcv.low[forward_idx];
                    let close = ohlcv.close[forward_idx];
                    let tp_hit = high.is_finite() && high >= upper_barrier;
                    let sl_hit = low.is_finite() && low <= lower_barrier;

                    if tp_hit && sl_hit {
                        let upper_dist = (close - upper_barrier).abs();
                        let lower_dist = (close - lower_barrier).abs();
                        *slot = if upper_dist < lower_dist {
                            1
                        } else if lower_dist < upper_dist {
                            -1
                        } else if close >= entry {
                            1
                        } else {
                            -1
                        };
                        break;
                    }

                    if tp_hit {
                        *slot = 1;
                        break;
                    }

                    if sl_hit {
                        *slot = -1;
                        break;
                    }
                }
            }

            if *slot != 0 {
                continue;
            }

            // The horizon fallback pays the round trip too.
            //
            // It compared the raw move against the band, so a drift smaller
            // than the spread was labelled a direction. That is the second
            // cost-free path into the same labels: shifting the barriers alone
            // left this one still answering "did price move" while the gene
            // side asks "did the trade clear its costs".
            //
            // Charged symmetrically — a short pays the same round trip a long
            // does — which widens the neutral zone by twice the cost rather
            // than sliding it in one direction.
            let terminal_close = ohlcv.close[horizon_end];
            let terminal_move = terminal_close - entry;
            let neutral_band = min_distance.max(atr * neutral_band_fraction);
            if terminal_move - round_trip_cost >= neutral_band {
                *slot = 1;
            } else if -terminal_move - round_trip_cost >= neutral_band {
                *slot = -1;
            }
        }

        Ok(labels)
    }

    #[cfg(test)]
    fn derive_labels_test_oracle(
        &self,
        ohlcv: &Ohlcv,
        symbol: &str,
        pip_size: f64,
    ) -> Result<Vec<i32>> {
        let round_trip_cost_pips = self.settings.risk.backtest_spread_pips.max(0.0)
            + 2.0 * self.settings.risk.slippage_pips.max(0.0);
        self.derive_labels_unchecked_test_oracle(ohlcv, symbol, pip_size, round_trip_cost_pips)
    }
}

/// Which bracket geometry `models.label_geometry` resolved to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LabelGeometryMode {
    Symmetric,
    Asymmetric,
}

impl LabelGeometryMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            LabelGeometryMode::Symmetric => "symmetric",
            LabelGeometryMode::Asymmetric => "asymmetric",
        }
    }
}

/// The label bracket geometry after every floor and clamp has been applied —
/// the numbers `derive_labels` actually races, and the numbers the training
/// profile records so a model can name the labels it was trained on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ResolvedLabelGeometry {
    pub mode: LabelGeometryMode,
    /// Fixed floor for the stop-side distance, in price.
    pub fixed_stop: f64,
    /// Fixed floor for the target-side distance, in price. Equals
    /// `fixed_stop` in symmetric mode.
    pub fixed_target: f64,
    /// Multiplier flooring the target at `rr_floor ×` the per-bar stop.
    /// Exactly 1.0 in symmetric mode — no rr floor pushes the target out.
    pub rr_floor: f64,
}

/// Resolve `models.label_geometry` + the `risk.meta_label_*` floors into the
/// bracket `derive_labels` uses. Fails loudly on an unknown geometry name —
/// a typo must stop training, not silently fall back to either geometry.
pub(crate) fn resolve_label_geometry(
    settings: &neoethos_core::Settings,
) -> Result<ResolvedLabelGeometry> {
    let raw = settings.models.label_geometry.trim();
    let mode = match raw.to_ascii_lowercase().as_str() {
        "symmetric" => LabelGeometryMode::Symmetric,
        "asymmetric" => LabelGeometryMode::Asymmetric,
        other => anyhow::bail!(
            "models.label_geometry = `{other}` is not a label geometry; valid values are \
             `symmetric` (default — equal bracket distances, cost charged both ways) and \
             `asymmetric` (legacy 2:1 long bracket, kept for comparison runs)"
        ),
    };
    let min_distance = settings.risk.meta_label_min_dist.max(1e-6);
    let fixed_stop = settings.risk.meta_label_fixed_sl.max(min_distance);
    let (fixed_target, rr_floor) = match mode {
        LabelGeometryMode::Symmetric => (fixed_stop, 1.0),
        LabelGeometryMode::Asymmetric => {
            // 💰 MONEY PATH, and mode-gated: this `.max()` only runs in
            // `asymmetric` geometry, so an operator who lowered one of the two
            // fields sees the change take effect or not depending on
            // `models.label_geometry`.
            log_money_knob_twin_disagreement(
                "models.label_take_profit_rr",
                settings.models.label_take_profit_rr,
                "risk.min_risk_reward",
                settings.risk.min_risk_reward,
                "asymmetric label reward:risk floor",
            );
            (
                settings.risk.meta_label_fixed_tp.max(min_distance),
                settings
                    .models
                    .label_take_profit_rr
                    .max(settings.risk.min_risk_reward)
                    .max(1.0),
            )
        }
    };
    Ok(ResolvedLabelGeometry {
        mode,
        fixed_stop,
        fixed_target,
        rr_floor,
    })
}

fn compute_true_ranges(ohlcv: &Ohlcv) -> Vec<f64> {
    let mut true_ranges = Vec::with_capacity(ohlcv.close.len());
    let mut previous_close: Option<f64> = None;

    for idx in 0..ohlcv.close.len() {
        let high = ohlcv
            .high
            .get(idx)
            .copied()
            .unwrap_or_else(|| ohlcv.close[idx]);
        let low = ohlcv
            .low
            .get(idx)
            .copied()
            .unwrap_or_else(|| ohlcv.close[idx]);
        let close = ohlcv.close[idx];
        let range = match previous_close {
            Some(prev_close) => {
                let intrabar = (high - low).abs();
                let gap_high = (high - prev_close).abs();
                let gap_low = (low - prev_close).abs();
                intrabar.max(gap_high).max(gap_low)
            }
            None => (high - low).abs(),
        };
        true_ranges.push(if range.is_finite() { range } else { 0.0 });
        previous_close = Some(close);
    }

    true_ranges
}

fn trailing_average(values: &[f64], end_idx: usize, window: usize) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let window = window.max(1);
    let start_idx = end_idx.saturating_add(1).saturating_sub(window);
    let slice = &values[start_idx..=end_idx.min(values.len() - 1)];
    if slice.is_empty() {
        0.0
    } else {
        slice.iter().copied().sum::<f64>() / slice.len() as f64
    }
}

fn canonical_model_name(name: &str) -> &str {
    if name
        .strip_prefix("transformer_")
        .is_some_and(|suffix| !suffix.is_empty() && suffix.chars().all(|ch| ch.is_ascii_digit()))
    {
        "transformer"
    } else {
        name
    }
}

fn transformer_replica_index(name: &str) -> Option<usize> {
    name.strip_prefix("transformer_")
        .and_then(|suffix| suffix.parse::<usize>().ok())
        .filter(|idx| *idx > 0)
}

fn configured_contains_model(configured: &[String], candidate: &str) -> bool {
    let canonical = canonical_model_name(candidate);
    configured
        .iter()
        .any(|name| name == candidate || canonical_model_name(name) == canonical)
}

fn parse_tree_params(params: &HashMap<String, String>) -> HashMap<String, ParamValue> {
    params
        .iter()
        .map(|(key, value)| {
            let parsed = match value.trim().to_ascii_lowercase().as_str() {
                "true" => ParamValue::Bool(true),
                "false" => ParamValue::Bool(false),
                _ => {
                    if let Ok(parsed) = value.parse::<i32>() {
                        ParamValue::Int(parsed)
                    } else if let Ok(parsed) = value.parse::<f64>() {
                        ParamValue::Float(parsed)
                    } else {
                        ParamValue::String(value.clone())
                    }
                }
            };
            (key.clone(), parsed)
        })
        .collect()
}

fn parse_f32_param(params: &HashMap<String, String>, key: &str, default: f32) -> f32 {
    params
        .get(key)
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default)
}

fn parse_f64_param(params: &HashMap<String, String>, key: &str, default: f64) -> f64 {
    params
        .get(key)
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default)
}

fn parse_usize_param(params: &HashMap<String, String>, key: &str, default: usize) -> usize {
    params
        .get(key)
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn parse_u64_param(params: &HashMap<String, String>, key: &str, default: u64) -> u64 {
    params
        .get(key)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn parse_bool_param(params: &HashMap<String, String>, key: &str, default: bool) -> bool {
    params
        .get(key)
        .map(|value| match value.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => true,
            "false" | "0" | "no" | "off" => false,
            _ => default,
        })
        .unwrap_or(default)
}

fn parse_string_param(params: &HashMap<String, String>, key: &str) -> Option<String> {
    params
        .get(key)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn model_params_only(params: &HashMap<String, String>) -> HashMap<String, String> {
    params
        .iter()
        .filter(|(key, _)| !key.starts_with("__"))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn hpo_backend_from_params(params: &HashMap<String, String>) -> String {
    params
        .get("__hpo_backend")
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "grid".to_string())
}

fn hpo_trials_from_params(params: &HashMap<String, String>) -> usize {
    parse_usize_param(params, "__hpo_trials", 1).max(1)
}

fn hpo_max_rows_from_params(params: &HashMap<String, String>) -> usize {
    parse_usize_param(params, "__hpo_max_rows", 0)
}

fn embargo_minutes_from_params(params: &HashMap<String, String>) -> usize {
    parse_usize_param(params, "__embargo_minutes", 0)
}

fn holdout_pct_from_params(params: &HashMap<String, String>) -> f64 {
    parse_f64_param(params, "__holdout_pct", 0.20).clamp(0.05, 0.45)
}

fn confidence_threshold_from_params(params: &HashMap<String, String>) -> f64 {
    parse_f64_param(params, "__conf_threshold", 0.55).clamp(0.0, 1.0)
}

fn metric_weight_from_params(params: &HashMap<String, String>) -> f64 {
    parse_f64_param(params, "__metric_weight", 1.0).max(0.0)
}

fn accuracy_weight_from_params(params: &HashMap<String, String>) -> f64 {
    parse_f64_param(params, "__accuracy_weight", 0.10).clamp(0.0, 1.0)
}

fn parse_parent_selection_policy(params: &HashMap<String, String>) -> ParentSelectionPolicy {
    ParentSelectionPolicy::parse(
        parse_string_param(params, "parent_selection")
            .as_deref()
            .unwrap_or("rank"),
    )
}

fn parse_survivor_selection_policy(params: &HashMap<String, String>) -> SurvivorSelectionPolicy {
    SurvivorSelectionPolicy::parse(
        parse_string_param(params, "survivor_selection")
            .as_deref()
            .unwrap_or("rank"),
    )
}

fn training_runtime_profile(
    settings: &neoethos_core::Settings,
    config: &ModelConfig,
    symbol: &str,
    base_tf: &str,
    payload: &TrainingPayload,
    row_budget_applied: Option<usize>,
    higher_timeframes: Vec<String>,
    label_geometry: &ResolvedLabelGeometry,
) -> TrainingRuntimeProfile {
    let effective_label_horizon_bars = if settings.models.label_horizon_bars > 0 {
        settings.models.label_horizon_bars
    } else {
        settings.risk.meta_label_max_hold_bars.max(1)
    };
    let requested_backend = parse_string_param(&config.params, "backend");
    let requested_device = parse_string_param(&config.params, "device");
    let planned_backend = parse_string_param(&config.params, "__planned_backend");
    let planned_device = parse_string_param(&config.params, "__planned_device");
    let planned_precision = parse_string_param(&config.params, "__planned_precision");
    let planned_cpu_threads = config
        .params
        .get("__planned_cpu_threads")
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0);
    let planned_batch_size = config
        .params
        .get("__planned_batch_size")
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0);
    let planned_memory_budget_gb = config
        .params
        .get("__planned_memory_budget_gb")
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0);
    let mut notes = Vec::new();
    if settings.models.enable_ddp || settings.models.enable_fsdp {
        notes.push(
            format!(
                "distributed deep-learning flags are recorded for runtime traceability; active burn backend in this build is `{}` and training still executes as a single-process local run",
                active_burn_backend_name()
            ),
        );
    }
    if parse_bool_param(&config.params, "async", false) {
        notes.push(
            "async discovery/search hint recorded; current model-training dispatch still executes synchronously inside the orchestrator".to_string(),
        );
    }
    if parse_string_param(&config.params, "checkpoint").is_some() {
        notes.push(
            "checkpoint path recorded for future search persistence; current model artifact path remains the primary persisted output".to_string(),
        );
    }

    TrainingRuntimeProfile {
        model_name: config.name.clone(),
        capability_family: config.capability_family,
        capability_state: config.capability_state,
        symbol: symbol.to_string(),
        base_timeframe: base_tf.to_string(),
        feature_count: payload.frame.n_features(),
        dataset_rows: payload.frame.n_samples(),
        row_budget_applied,
        label_horizon_bars: settings.models.label_horizon_bars,
        effective_label_horizon_bars,
        meta_label_max_hold_bars: settings.risk.meta_label_max_hold_bars.max(1),
        label_use_triple_barrier: settings.models.label_use_triple_barrier,
        label_geometry: label_geometry.mode.as_str().to_string(),
        label_fixed_stop: label_geometry.fixed_stop,
        label_fixed_target: label_geometry.fixed_target,
        label_rr_floor: label_geometry.rr_floor,
        higher_timeframes,
        multi_resolution_enabled: settings.system.multi_resolution_enabled,
        base_features_prefixed: settings.system.multi_resolution_prefix_base,
        base_signal_filter_enabled: settings.models.filter_to_base_signal,
        l1_feature_selection_enabled: settings.models.l1_feature_selection_enabled,
        requested_backend,
        requested_device,
        planned_backend,
        planned_device,
        planned_precision,
        planned_cpu_threads,
        planned_batch_size,
        planned_memory_budget_gb,
        checkpoint_path: parse_string_param(&config.params, "checkpoint").map(PathBuf::from),
        async_requested: parse_bool_param(&config.params, "async", false),
        async_wait_requested: parse_bool_param(&config.params, "async_wait", false),
        train_years: parse_usize_param(&config.params, "train_years", 0),
        val_years: parse_usize_param(&config.params, "val_years", 0),
        requested_hpo_backend: hpo_backend_from_params(&config.params),
        requested_hpo_trials: hpo_trials_from_params(&config.params),
        holdout_pct: holdout_pct_from_params(&config.params),
        embargo_minutes: embargo_minutes_from_params(&config.params),
        rllib_requested: false,
        rllib_num_workers: 0,
        ray_tune_max_concurrency: parse_usize_param(
            &config.params,
            "ray_tune_max_concurrency",
            settings.models.ray_tune_max_concurrency.max(1),
        ),
        ddp_enabled: settings.models.enable_ddp,
        fsdp_enabled: settings.models.enable_fsdp,
        ddp_world_size: settings.models.ddp_world_size.max(1),
        symbol_hash_buckets: settings.models.symbol_hash_buckets.max(1),
        notes,
    }
}

fn training_profile_higher_timeframes(
    settings: &neoethos_core::Settings,
    base_tf: &str,
) -> Vec<String> {
    settings.system.resolve_higher_timeframes(base_tf)
}

fn write_training_profile_sidecar(
    artifact_dir: &std::path::Path,
    settings: &neoethos_core::Settings,
    config: &ModelConfig,
    symbol: &str,
    base_tf: &str,
    payload: &TrainingPayload,
    row_budget_applied: Option<usize>,
) -> Result<TrainingRuntimeProfile> {
    // Resolves the same geometry `derive_labels` used at the top of the run —
    // an invalid geometry name has already failed the run loudly before any
    // model trains, so this is re-derivation for the record, not a re-check.
    let label_geometry = resolve_label_geometry(settings)?;
    let profile = training_runtime_profile(
        settings,
        config,
        symbol,
        base_tf,
        payload,
        row_budget_applied,
        training_profile_higher_timeframes(settings, base_tf),
        &label_geometry,
    );
    write_training_runtime_profile(
        &artifact_dir.join(TRAINING_RUNTIME_PROFILE_FILE_NAME),
        &profile,
    )?;
    Ok(profile)
}

fn timeframe_to_minutes(base_tf: &str) -> usize {
    let upper = base_tf.trim().to_ascii_uppercase();
    if let Some(minutes) = upper
        .strip_prefix('M')
        .and_then(|value| value.parse::<usize>().ok())
    {
        return minutes.max(1);
    }
    if let Some(hours) = upper
        .strip_prefix('H')
        .and_then(|value| value.parse::<usize>().ok())
    {
        return (hours.max(1)) * 60;
    }
    if let Some(days) = upper
        .strip_prefix('D')
        .and_then(|value| value.parse::<usize>().ok())
    {
        return (days.max(1)) * 1_440;
    }
    if let Some(weeks) = upper
        .strip_prefix('W')
        .and_then(|value| value.parse::<usize>().ok())
    {
        return (weeks.max(1)) * 10_080;
    }
    if let Some(months) = upper
        .strip_prefix("MN")
        .and_then(|value| value.parse::<usize>().ok())
    {
        return (months.max(1)) * 43_200;
    }
    1
}

/// The hard train→val leak floor, in bars of the base timeframe. Not
/// configurable in either direction: `models.embargo_minutes` widens the gap,
/// nothing narrows it below this.
const MIN_EMBARGO_BARS: usize = 20;

// ---------------------------------------------------------------------------
// Duplicate-knob disagreement reporting (W2-12)
// ---------------------------------------------------------------------------
//
// Six settings in this file are pairs of fields collapsed by `.max()` or
// `.min()`. The collapse itself is NOT changed here — changing which number
// binds is a behaviour change that needs its own measurement, and for the two
// money-path pairs it needs the operator's decision about which name survives.
//
// What changes is that the disagreement can no longer be silent. A silent
// `.max()` means RAISING EITHER field raises the effective value: an operator
// who lowered one has not lowered the setting, and no log line told him. The
// transformer trio agree at 256 today by luck, so today these lines are quiet.
//
// Once one name in each pair is deleted, delete the matching call.

/// Log at WARN when two integer fields that get `.max()`ed together disagree,
/// naming both and the winner.
fn log_knob_twin_disagreement(field_a: &str, value_a: usize, field_b: &str, value_b: usize) {
    if value_a == value_b {
        return;
    }
    let (winner, winning_value) = if value_a >= value_b {
        (field_a, value_a)
    } else {
        (field_b, value_b)
    };
    tracing::warn!(
        target: "neoethos_models::config",
        field_a,
        value_a,
        field_b,
        value_b,
        binding_field = winner,
        binding_value = winning_value,
        "duplicate knob: {field_a}={value_a} and {field_b}={value_b} are merged \
         with .max(); {winner}={winning_value} binds. Lowering only one of them \
         does not lower the effective value."
    );
}

/// As [`log_knob_twin_disagreement`], for float pairs on the money path. Louder
/// wording because these two decide the stop distance and the reward:risk floor
/// that every label — and therefore every trained model — is built from.
fn log_money_knob_twin_disagreement(
    field_a: &str,
    value_a: f64,
    field_b: &str,
    value_b: f64,
    what: &str,
) {
    if (value_a - value_b).abs() < f64::EPSILON {
        return;
    }
    let (winner, winning_value) = if value_a >= value_b {
        (field_a, value_a)
    } else {
        (field_b, value_b)
    };
    tracing::warn!(
        target: "neoethos_models::config",
        field_a,
        value_a,
        field_b,
        value_b,
        binding_field = winner,
        binding_value = winning_value,
        decides = what,
        "MONEY-PATH duplicate knob: {field_a}={value_a} and {field_b}={value_b} \
         both claim to set the {what}; they are merged with .max(), so \
         {winner}={winning_value} binds. Lowering only one of them changes \
         nothing."
    );
}

fn embargo_rows_for_timeframe(base_tf: &str, embargo_minutes: usize) -> usize {
    let tf_minutes = timeframe_to_minutes(base_tf).max(1);
    let raw = if embargo_minutes == 0 {
        0
    } else {
        ((embargo_minutes as f64) / (tf_minutes as f64)).ceil() as usize
    };
    // M4: enforce a hard minimum so a misconfigured `embargo_minutes = 0`
    // can't allow train→val with no time gap on intraday models. 20 bars on
    // the base timeframe is the floor (~20 minutes on M1, ~5h on M15) — long
    // enough to bracket the typical label horizon for sub-hour forecasters.
    //
    // 2026-08-10: `NEOETHOS_BOT_PROP_MIN_EMBARGO_BARS` could LOWER this floor,
    // including to zero, from a shell — a leak guard that an untracked export
    // could switch off. A safety floor is not an operator preference; the way
    // to widen the gap is `models.embargo_minutes`, which is config, is in the
    // artifact, and can only make the embargo longer than this floor, never
    // shorter. Deleted; reported at startup if still set.
    raw.max(MIN_EMBARGO_BARS)
}

fn halton(mut index: usize, base: usize) -> f64 {
    let mut fraction = 1.0_f64;
    let mut result = 0.0_f64;
    let base = base.max(2);
    while index > 0 {
        fraction /= base as f64;
        result += fraction * (index % base) as f64;
        index /= base;
    }
    result
}

fn sample_choice(
    values: &[&str],
    trial_idx: usize,
    trials: usize,
    backend: &str,
    base: usize,
) -> String {
    if values.is_empty() {
        return String::new();
    }
    let index = if backend == "grid" {
        if trials <= 1 {
            0
        } else {
            trial_idx % values.len()
        }
    } else {
        let frac = halton(trial_idx + 1, base);
        ((frac * values.len() as f64).floor() as usize).min(values.len().saturating_sub(1))
    };
    values[index].to_string()
}

fn sample_f64(
    min: f64,
    max: f64,
    trial_idx: usize,
    trials: usize,
    backend: &str,
    base: usize,
) -> f64 {
    if (max - min).abs() < f64::EPSILON {
        return min;
    }
    let fraction = if backend == "grid" {
        if trials <= 1 {
            0.5
        } else {
            trial_idx as f64 / (trials.saturating_sub(1).max(1)) as f64
        }
    } else {
        halton(trial_idx + 1, base)
    };
    min + ((max - min) * fraction.clamp(0.0, 1.0))
}

fn sample_usize(
    values: &[usize],
    trial_idx: usize,
    trials: usize,
    backend: &str,
    base: usize,
) -> usize {
    if values.is_empty() {
        return 0;
    }
    let index = if backend == "grid" {
        if trials <= 1 {
            0
        } else {
            trial_idx % values.len()
        }
    } else {
        let frac = halton(trial_idx + 1, base);
        ((frac * values.len() as f64).floor() as usize).min(values.len().saturating_sub(1))
    };
    values[index]
}

fn calibration_method_from_params(params: &HashMap<String, String>) -> CalibrationMethod {
    match params
        .get("method")
        .map(|value| value.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("identity") => CalibrationMethod::Identity,
        Some("temperature") => CalibrationMethod::Temperature,
        _ => CalibrationMethod::Platt,
    }
}

fn model_artifact_dir(
    models_dir: &std::path::Path,
    symbol: &str,
    base_tf: &str,
    model_name: &str,
) -> PathBuf {
    models_dir.join(symbol).join(base_tf).join(model_name)
}

/// GROUP E remediation 2026-05-25: 5 hand-rolled functions replaced
/// with a single delegation to the canonical `write_dir_with_backup`
/// helper in `neoethos-core::storage::json`. Saves ~60 LOC of duplicate
/// staged-tmp+backup logic (this file is one of 4 — final one).
/// Existing tests `with_staged_training_artifact_dir_promotes_complete_directory`
/// and `with_staged_training_artifact_dir_cleans_up_failed_stage` exercise
/// the same function name and continue to pin the consolidated semantics.
fn with_staged_training_artifact_dir<F>(path: &Path, writer: F) -> Result<()>
where
    F: FnOnce(&Path) -> Result<()>,
{
    neoethos_core::storage::json::write_dir_with_backup(
        path,
        neoethos_core::storage::json::DirBackupWriteConfig {
            artifact_label: "training artifact",
            temp_extension: "tmp_training_dispatch",
            backup_extension: "bak_training_dispatch",
        },
        writer,
    )
}

#[allow(clippy::too_many_arguments)]
fn persist_training_artifacts<F>(
    artifact_dir: &Path,
    settings: &neoethos_core::Settings,
    config: &ModelConfig,
    symbol: &str,
    base_tf: &str,
    payload: &TrainingPayload,
    row_budget_applied: Option<usize>,
    optimization_report: Option<&OptimizationReport>,
    save_model: F,
) -> Result<()>
where
    F: FnOnce(&Path) -> Result<()>,
{
    with_staged_training_artifact_dir(artifact_dir, |staged_dir| {
        save_model(staged_dir)?;
        if let Some(report) = optimization_report {
            write_optimization_report(&staged_dir.join(OPTIMIZATION_REPORT_FILE_NAME), report)?;
        }
        let profile = write_training_profile_sidecar(
            staged_dir,
            settings,
            config,
            symbol,
            base_tf,
            payload,
            row_budget_applied,
        )?;
        write_training_model_artifact_contract_sidecar(
            staged_dir, settings, config, payload, &profile,
        )?;
        write_model_runtime_artifact_contract_sidecar(
            staged_dir, settings, config, payload, &profile,
        )?;
        Ok(())
    })
}

fn supports_hpo(model_type: ModelType) -> bool {
    matches!(
        model_type,
        ModelType::LightGBM
            | ModelType::XGBoost
            | ModelType::CatBoost
            | ModelType::MLP
            | ModelType::NBeats
            | ModelType::NBeatsxNf
            | ModelType::TiDE
            | ModelType::TiDENf
            | ModelType::TabNet
            | ModelType::KAN
            | ModelType::Transformer
            | ModelType::PatchTST
            | ModelType::TimesNet
            | ModelType::ElasticNet
            | ModelType::Logistic
            | ModelType::OnlinePassiveAggressive
            | ModelType::OnlineHoeffding
            | ModelType::IsolationForest
            | ModelType::NeuroEvo
            | ModelType::Neat
    )
}

fn uses_shared_expert_dispatch(model_type: ModelType) -> bool {
    matches!(
        model_type,
        ModelType::LightGBM
            | ModelType::XGBoost
            | ModelType::CatBoost
            | ModelType::MLP
            | ModelType::NBeats
            | ModelType::NBeatsxNf
            | ModelType::TiDE
            | ModelType::TiDENf
            | ModelType::TabNet
            | ModelType::KAN
            | ModelType::Transformer
            | ModelType::PatchTST
            | ModelType::TimesNet
            | ModelType::ElasticNet
            | ModelType::Logistic
            | ModelType::BayesianLogit
            | ModelType::MetaBlender
            | ModelType::ProbabilityCalibrator
            | ModelType::ConformalGate
            | ModelType::MetaStack
            | ModelType::OnlinePassiveAggressive
            | ModelType::OnlineHoeffding
            | ModelType::IsolationForest
            | ModelType::NeuroEvo
            | ModelType::Neat
    )
}

/// Phase D1.4.1 — per-family seed offsets for the tree experts.
/// LightGBM honours `seed`, XGBoost honours `seed`, CatBoost
/// honours `random_seed`. The native boosters read these keys
/// from the params HashMap that gets threaded through
/// `parse_tree_params()`; we just need to make sure each tree
/// family receives a DISTINCT seed value instead of inheriting
/// whatever the operator's global config had (or nothing).
///
/// Offsets are prime-spaced from the operator's `seed_base` (or
/// 42 when unset) so any operator-supplied shift can't cause
/// adjacent families to collide.
fn inject_tree_seed(
    params: &HashMap<String, String>,
    seed_key: &str,
    family_offset: u64,
) -> HashMap<String, String> {
    let mut p = params.clone();
    if !p.contains_key(seed_key) {
        // Operator hasn't pinned a seed — inject our family offset.
        // Operators wanting determinism across runs can set the
        // model's seed in config.yaml directly; that override
        // takes precedence (the `!contains_key` guard above).
        p.insert(seed_key.to_string(), (42 + family_offset).to_string());
    }
    p
}

fn build_expert_model(
    config: &ModelConfig,
    input_dim: usize,
    params: &HashMap<String, String>,
) -> Result<Box<dyn ExpertModel>> {
    match config.model_type {
        ModelType::LightGBM => {
            // Per-family seed = 42 + 1
            let seeded = inject_tree_seed(params, "seed", 1);
            Ok(Box::new(LightGBMExpert::new(
                0,
                Some(parse_tree_params(&seeded)),
            )))
        }
        ModelType::XGBoost => {
            // Per-family seed = 42 + 2 (gbtree variant).
            // The xgboost_rf and xgboost_dart variants will get
            // their own offsets when the orchestrator routes the
            // booster variant — they currently all funnel through
            // ModelType::XGBoost so they share the same offset for
            // now. A future commit can split them via the
            // booster_variant config field.
            let seeded = inject_tree_seed(params, "seed", 2);
            Ok(Box::new(XGBoostExpert::new(
                0,
                Some(parse_tree_params(&seeded)),
            )))
        }
        ModelType::CatBoost => {
            // CatBoost uses `random_seed` not `seed`. Offset = 42 + 5
            // (catboost) vs +6 (catboost_alt). The orchestrator
            // currently treats both as ModelType::CatBoost; the
            // alt-variant gets its own offset in a follow-up.
            let seeded = inject_tree_seed(params, "random_seed", 5);
            Ok(Box::new(CatBoostExpert::new_with_params(
                0,
                parse_tree_params(&seeded),
            )))
        }
        ModelType::ElasticNet => {
            let mut model = ElasticNetExpert::new(
                parse_f64_param(params, "alpha", 0.1),
                parse_f64_param(params, "l1_ratio", 0.5),
            );
            model.learning_rate = parse_f64_param(params, "lr", model.learning_rate);
            model.epochs = parse_usize_param(params, "epochs", model.epochs);
            Ok(Box::new(model))
        }
        ModelType::Logistic => {
            let mut model = LogisticExpert::new();
            model.alpha = parse_f64_param(params, "alpha", model.alpha);
            model.learning_rate = parse_f64_param(params, "lr", model.learning_rate);
            model.epochs = parse_usize_param(params, "epochs", model.epochs);
            Ok(Box::new(model))
        }
        ModelType::BayesianLogit => Ok(Box::new(BayesianLogitExpert::new())),
        ModelType::MetaBlender => Ok(Box::new(MetaBlender::new())),
        ModelType::ProbabilityCalibrator => {
            let mut model =
                ProbabilityCalibrationExpert::new(calibration_method_from_params(params));
            model.min_fit_rows = parse_usize_param(params, "min_rows", model.min_fit_rows);
            Ok(Box::new(model))
        }
        ModelType::ConformalGate => {
            let mut model = ConformalPredictionExpert::new(
                calibration_method_from_params(params),
                parse_f64_param(params, "alpha", 0.10),
            );
            model.min_prediction_set =
                parse_usize_param(params, "min_prediction_set", model.min_prediction_set);
            model.min_fit_rows = parse_usize_param(params, "min_rows", model.min_fit_rows);
            Ok(Box::new(model))
        }
        ModelType::MetaStack => {
            let mut model = MetaDecisionStack::new(
                calibration_method_from_params(params),
                parse_f64_param(params, "alpha", 0.10),
            );
            model.min_prediction_set =
                parse_usize_param(params, "min_prediction_set", model.min_prediction_set);
            model.min_fit_rows = parse_usize_param(params, "min_rows", model.min_fit_rows);
            Ok(Box::new(model))
        }
        ModelType::OnlinePassiveAggressive => Ok(Box::new(OnlinePassiveAggressiveExpert::new(
            parse_f32_param(params, "c", 1.0),
            parse_usize_param(params, "epochs", 4),
        ))),
        ModelType::OnlineHoeffding => {
            Ok(Box::new(OnlineHoeffdingExpert::new(Some(params.clone()))))
        }
        ModelType::IsolationForest => {
            let mut model = IsolationForestExpert::new(
                parse_usize_param(params, "n_trees", 128),
                parse_usize_param(params, "sample_size", 256),
            );
            if let Some(extension_level) = params
                .get("extension_level")
                .and_then(|value| value.parse::<usize>().ok())
            {
                model.extension_level = extension_level;
            }
            if let Some(max_tree_depth) = params
                .get("max_tree_depth")
                .and_then(|value| value.parse::<usize>().ok())
            {
                model.max_tree_depth = Some(max_tree_depth);
            }
            Ok(Box::new(model))
        }
        ModelType::NeuroEvo => Ok(Box::new(
            NeuroEvoExpert::with_config(
                input_dim.max(1),
                parse_usize_param(params, "hidden_dim", 32),
                parse_f64_param(params, "sigma", 0.25),
                parse_usize_param(params, "generations", 24),
            )
            .with_device_policy(
                parse_string_param(params, "device").unwrap_or_else(|| "auto".to_string()),
            )
            .with_search_topology(
                parse_usize_param(params, "population", 16),
                parse_usize_param(params, "islands", 1),
            ),
        )),
        ModelType::Neat => Ok(Box::new(
            NeatExpert::with_config(
                input_dim.max(1),
                parse_usize_param(params, "population", 96),
                parse_usize_param(params, "generations", 48),
            )
            .with_device_policy(
                parse_string_param(params, "device").unwrap_or_else(|| "auto".to_string()),
            )
            .with_search_params(
                parse_f32_param(params, "mutation_rate", 0.85),
                parse_usize_param(params, "species_elitism", 0),
                parse_f32_param(params, "compatibility_threshold", 2.5),
                parse_f32_param(params, "immigrant_fraction", 0.1),
                parse_u64_param(params, "seed", 42),
            ),
        )),
        // Phase D1.4 — per-family seed offsets so the 10 deep
        // experts converge to DIFFERENT solutions rather than
        // identical-by-seed-42 ones. The seed plumbing through
        // BurnDeepExpert was already there; we just stopped passing
        // 42 everywhere. Per-family offsets are stable across
        // training runs (deterministic) but distinct across
        // families. If the operator wants randomized seeds they
        // can override per model via the `seed` key in the params
        // HashMap (BurnDeepExpert's new() takes seed as its
        // constructor arg; the orchestrator's value here is the
        // default).
        //
        // Why these specific offsets: each prime-spaced from 42
        // so even if the operator's settings injects a seed_base
        // additive shift, no two adapters collide. Architectural
        // diversity (MLP vs Transformer vs NBEATS etc.) still
        // dominates as the primary diversifier — the seed
        // offsets are a secondary diversifier per the 2026
        // ensemble-learning research the operator surfaced.
        ModelType::MLP => Ok(Box::new(MLPExpert::new(42, Some(params.clone())))),
        ModelType::NBeats => Ok(Box::new(NBeatsExpert::new(43, Some(params.clone())))),
        ModelType::NBeatsxNf => Ok(Box::new(NBeatsxNfExpert::new(47, Some(params.clone())))),
        ModelType::TiDE => Ok(Box::new(TiDEExpert::new(53, Some(params.clone())))),
        ModelType::TiDENf => Ok(Box::new(TiDENfExpert::new(59, Some(params.clone())))),
        ModelType::TabNet => Ok(Box::new(TabNetExpert::new(61, Some(params.clone())))),
        ModelType::KAN => Ok(Box::new(KANExpert::new(67, Some(params.clone())))),
        ModelType::Transformer => Ok(Box::new(TransformerExpert::new(71, Some(params.clone())))),
        ModelType::PatchTST => Ok(Box::new(PatchTSTExpert::new(73, Some(params.clone())))),
        ModelType::TimesNet => Ok(Box::new(TimesNetExpert::new(79, Some(params.clone())))),
        other => anyhow::bail!(
            "model `{}` ({:?}) does not support the shared ExpertModel training path",
            config.name,
            other
        ),
    }
}

fn generate_hpo_candidate_params(
    config: &ModelConfig,
    base_params: &HashMap<String, String>,
    trial_idx: usize,
    trials: usize,
    backend: &str,
) -> HashMap<String, String> {
    let mut params = base_params.clone();
    let canonical = canonical_model_name(&config.name);

    match canonical {
        "lightgbm" => {
            params.insert(
                "learning_rate".to_string(),
                format!(
                    "{:.5}",
                    sample_f64(0.02, 0.12, trial_idx, trials, backend, 2)
                ),
            );
            params.insert(
                "num_iterations".to_string(),
                sample_usize(&[240, 400, 560, 720, 960], trial_idx, trials, backend, 3).to_string(),
            );
            params.insert(
                "max_depth".to_string(),
                sample_choice(&["4", "6", "8", "10", "12"], trial_idx, trials, backend, 5),
            );
            params.insert(
                "num_leaves".to_string(),
                sample_choice(&["15", "31", "63", "127"], trial_idx, trials, backend, 7),
            );
            params.insert(
                "min_data_in_leaf".to_string(),
                sample_choice(&["16", "32", "64", "128"], trial_idx, trials, backend, 11),
            );
            params.insert(
                "feature_fraction".to_string(),
                format!(
                    "{:.4}",
                    sample_f64(0.60, 1.00, trial_idx, trials, backend, 13)
                ),
            );
            params.insert(
                "bagging_fraction".to_string(),
                format!(
                    "{:.4}",
                    sample_f64(0.60, 1.00, trial_idx, trials, backend, 17)
                ),
            );
            params.insert(
                "lambda_l2".to_string(),
                format!(
                    "{:.4}",
                    sample_f64(0.0, 8.0, trial_idx, trials, backend, 19)
                ),
            );
        }
        "xgboost" => {
            params.insert(
                "learning_rate".to_string(),
                format!(
                    "{:.5}",
                    sample_f64(0.02, 0.12, trial_idx, trials, backend, 2)
                ),
            );
            params.insert(
                "n_estimators".to_string(),
                sample_usize(&[320, 480, 640, 800, 960], trial_idx, trials, backend, 3).to_string(),
            );
            params.insert(
                "max_depth".to_string(),
                sample_choice(&["4", "6", "8", "10"], trial_idx, trials, backend, 5),
            );
            params.insert(
                "min_child_weight".to_string(),
                sample_choice(&["1", "2", "4", "8"], trial_idx, trials, backend, 7),
            );
            params.insert(
                "subsample".to_string(),
                format!(
                    "{:.4}",
                    sample_f64(0.60, 1.00, trial_idx, trials, backend, 11)
                ),
            );
            params.insert(
                "colsample_bytree".to_string(),
                format!(
                    "{:.4}",
                    sample_f64(0.60, 1.00, trial_idx, trials, backend, 13)
                ),
            );
            params.insert(
                "reg_lambda".to_string(),
                format!(
                    "{:.4}",
                    sample_f64(0.5, 8.0, trial_idx, trials, backend, 17)
                ),
            );
            if canonical_model_name(&config.name) == "xgboost" && config.name.contains("dart") {
                params.insert(
                    "rate_drop".to_string(),
                    format!(
                        "{:.4}",
                        sample_f64(0.05, 0.25, trial_idx, trials, backend, 19)
                    ),
                );
                params.insert(
                    "skip_drop".to_string(),
                    format!(
                        "{:.4}",
                        sample_f64(0.10, 0.70, trial_idx, trials, backend, 23)
                    ),
                );
            }
            if config.name.contains("rf") {
                params.insert(
                    "num_parallel_tree".to_string(),
                    sample_usize(&[32, 64, 96, 128], trial_idx, trials, backend, 29).to_string(),
                );
            }
        }
        "catboost" => {
            params.insert(
                "iterations".to_string(),
                sample_usize(&[320, 480, 640, 800, 960], trial_idx, trials, backend, 2).to_string(),
            );
            params.insert(
                "depth".to_string(),
                sample_choice(&["4", "6", "8", "10"], trial_idx, trials, backend, 3),
            );
            params.insert(
                "learning_rate".to_string(),
                format!(
                    "{:.5}",
                    sample_f64(0.02, 0.12, trial_idx, trials, backend, 5)
                ),
            );
            params.insert(
                "l2_leaf_reg".to_string(),
                format!(
                    "{:.4}",
                    sample_f64(1.0, 10.0, trial_idx, trials, backend, 7)
                ),
            );
            params.insert(
                "random_strength".to_string(),
                format!(
                    "{:.4}",
                    sample_f64(0.0, 3.0, trial_idx, trials, backend, 11)
                ),
            );
            params.insert(
                "bagging_temperature".to_string(),
                format!(
                    "{:.4}",
                    sample_f64(0.0, 2.0, trial_idx, trials, backend, 13)
                ),
            );
        }
        "mlp" | "transformer" | "patchtst" | "timesnet" | "nbeats" | "nbeatsx_nf" | "tide"
        | "tide_nf" | "tabnet" | "kan" => {
            params.insert(
                "hidden_dim".to_string(),
                sample_usize(&[128, 192, 256, 384, 512], trial_idx, trials, backend, 2).to_string(),
            );
            params.insert(
                "dropout".to_string(),
                format!(
                    "{:.4}",
                    sample_f64(0.02, 0.25, trial_idx, trials, backend, 3)
                ),
            );
            params.insert(
                "lr".to_string(),
                format!(
                    "{:.6}",
                    sample_f64(0.0001, 0.0030, trial_idx, trials, backend, 5)
                ),
            );
            params.insert(
                "batch_size".to_string(),
                sample_usize(&[16, 32, 64, 128], trial_idx, trials, backend, 7).to_string(),
            );
            params.insert(
                "max_epochs".to_string(),
                sample_usize(&[48, 72, 96, 128, 160], trial_idx, trials, backend, 11).to_string(),
            );
            if matches!(canonical, "transformer" | "patchtst") {
                params.insert(
                    "n_heads".to_string(),
                    sample_choice(&["4", "6", "8", "12"], trial_idx, trials, backend, 13),
                );
                params.insert(
                    "n_layers".to_string(),
                    sample_choice(&["2", "3", "4", "6"], trial_idx, trials, backend, 17),
                );
                if canonical == "transformer" {
                    params.insert(
                        "token_count".to_string(),
                        sample_choice(&["8", "16", "32", "64"], trial_idx, trials, backend, 41),
                    );
                }
            }
            if matches!(canonical, "nbeats" | "nbeatsx_nf") {
                params.insert(
                    "n_blocks".to_string(),
                    sample_choice(&["2", "3", "4", "5"], trial_idx, trials, backend, 19),
                );
            }
            if canonical == "tabnet" {
                params.insert(
                    "n_steps".to_string(),
                    sample_choice(&["3", "4", "5", "6"], trial_idx, trials, backend, 23),
                );
                params.insert(
                    "relaxation_factor".to_string(),
                    format!(
                        "{:.4}",
                        sample_f64(1.2, 2.0, trial_idx, trials, backend, 29)
                    ),
                );
            }
            if canonical == "kan" {
                params.insert(
                    "n_layers".to_string(),
                    sample_choice(&["2", "3", "4"], trial_idx, trials, backend, 31),
                );
                params.insert(
                    "grid_size".to_string(),
                    sample_choice(&["7", "9", "13", "17"], trial_idx, trials, backend, 37),
                );
            }
        }
        "elasticnet" => {
            params.insert(
                "alpha".to_string(),
                format!(
                    "{:.6}",
                    sample_f64(0.0005, 0.5, trial_idx, trials, backend, 2)
                ),
            );
            params.insert(
                "l1_ratio".to_string(),
                format!(
                    "{:.5}",
                    sample_f64(0.05, 0.95, trial_idx, trials, backend, 3)
                ),
            );
            params.insert(
                "lr".to_string(),
                format!(
                    "{:.6}",
                    sample_f64(0.001, 0.05, trial_idx, trials, backend, 5)
                ),
            );
            params.insert(
                "epochs".to_string(),
                sample_usize(&[200, 400, 800, 1200], trial_idx, trials, backend, 7).to_string(),
            );
        }
        "logistic" => {
            params.insert(
                "alpha".to_string(),
                format!(
                    "{:.6}",
                    sample_f64(0.0005, 0.25, trial_idx, trials, backend, 2)
                ),
            );
            params.insert(
                "lr".to_string(),
                format!(
                    "{:.6}",
                    sample_f64(0.001, 0.05, trial_idx, trials, backend, 5)
                ),
            );
            params.insert(
                "epochs".to_string(),
                sample_usize(&[150, 250, 500, 900], trial_idx, trials, backend, 7).to_string(),
            );
        }
        "online_pa" => {
            params.insert(
                "c".to_string(),
                format!("{:.5}", sample_f64(0.1, 5.0, trial_idx, trials, backend, 2)),
            );
            params.insert(
                "epochs".to_string(),
                sample_usize(&[2, 4, 6, 8, 12], trial_idx, trials, backend, 3).to_string(),
            );
        }
        "online_hoeffding" => {
            params.insert(
                "learning_rate".to_string(),
                format!(
                    "{:.6}",
                    sample_f64(0.005, 0.20, trial_idx, trials, backend, 2)
                ),
            );
            params.insert(
                "max_depth".to_string(),
                sample_choice(&["3", "5", "7", "9"], trial_idx, trials, backend, 3),
            );
            params.insert(
                "n_bins".to_string(),
                sample_choice(&["16", "32", "48", "64"], trial_idx, trials, backend, 5),
            );
            params.insert(
                "grace_period".to_string(),
                sample_choice(&["16", "32", "48", "64"], trial_idx, trials, backend, 7),
            );
            params.insert(
                "feature_subsample_rate".to_string(),
                format!(
                    "{:.4}",
                    sample_f64(0.50, 1.00, trial_idx, trials, backend, 11)
                ),
            );
        }
        "isolation_forest" => {
            params.insert(
                "n_trees".to_string(),
                sample_usize(&[64, 96, 128, 192, 256], trial_idx, trials, backend, 2).to_string(),
            );
            params.insert(
                "sample_size".to_string(),
                sample_usize(&[128, 192, 256, 384, 512], trial_idx, trials, backend, 3).to_string(),
            );
            params.insert(
                "extension_level".to_string(),
                sample_choice(&["0", "1", "2"], trial_idx, trials, backend, 5),
            );
            params.insert(
                "max_tree_depth".to_string(),
                sample_choice(&["8", "12", "16", "24"], trial_idx, trials, backend, 7),
            );
        }
        "neuro_evo" => {
            params.insert(
                "hidden_dim".to_string(),
                sample_usize(&[16, 32, 64, 96, 128], trial_idx, trials, backend, 2).to_string(),
            );
            params.insert(
                "sigma".to_string(),
                format!(
                    "{:.5}",
                    sample_f64(0.05, 0.45, trial_idx, trials, backend, 3)
                ),
            );
            params.insert(
                "generations".to_string(),
                sample_usize(&[16, 24, 32, 48, 64], trial_idx, trials, backend, 5).to_string(),
            );
        }
        "neat" => {
            params.insert(
                "population".to_string(),
                sample_usize(&[48, 64, 96, 128, 160], trial_idx, trials, backend, 2).to_string(),
            );
            params.insert(
                "generations".to_string(),
                sample_usize(&[24, 36, 48, 64, 96], trial_idx, trials, backend, 3).to_string(),
            );
            params.insert(
                "mutation_rate".to_string(),
                format!(
                    "{:.5}",
                    sample_f64(0.35, 1.05, trial_idx, trials, backend, 5)
                ),
            );
            params.insert(
                "compatibility_threshold".to_string(),
                format!("{:.5}", sample_f64(1.5, 4.0, trial_idx, trials, backend, 7)),
            );
        }
        _ => {}
    }

    enforce_planned_resource_caps(&mut params);
    params
}

fn enforce_planned_resource_caps(params: &mut HashMap<String, String>) {
    for (key, planned_key) in [
        ("cpu_threads", "__planned_cpu_threads"),
        ("batch_size", "__planned_batch_size"),
    ] {
        let Some(planned) = params
            .get(planned_key)
            .and_then(|value| value.parse::<usize>().ok())
        else {
            continue;
        };
        let effective = params
            .get(key)
            .and_then(|value| value.parse::<usize>().ok())
            .map_or(planned, |requested| requested.min(planned));
        params.insert(key.to_string(), effective.to_string());
    }

    let Some(planned_memory_gb) = params
        .get("__planned_memory_budget_gb")
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
    else {
        return;
    };
    let effective_memory_gb = params
        .get("memory_budget_gb")
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .map_or(planned_memory_gb, |requested| {
            requested.min(planned_memory_gb)
        });
    params.insert(
        "memory_budget_gb".to_string(),
        format!("{effective_memory_gb:.6}"),
    );
}

fn select_hpo_dataset(
    payload: &TrainingPayload,
    max_rows: usize,
) -> Result<(FeatureFrame, Vec<i32>, Option<usize>)> {
    if payload.frame.n_samples() != payload.labels.len() {
        anyhow::bail!(
            "HPO payload mismatch: {} rows vs {} labels",
            payload.frame.n_samples(),
            payload.labels.len()
        );
    }
    if payload.source_row_indices.len() != payload.labels.len() {
        anyhow::bail!(
            "HPO payload source receipt mismatch: {} rows vs {} receipts",
            payload.labels.len(),
            payload.source_row_indices.len()
        );
    }
    if max_rows == 0 || payload.frame.n_samples() <= max_rows {
        return Ok((
            payload.frame.as_ref().clone(),
            payload.labels.as_ref().clone(),
            None,
        ));
    }

    let start = payload.frame.n_samples().saturating_sub(max_rows);
    let frame = payload.frame.row_window(start, payload.frame.n_samples())?;
    let labels = payload.labels[start..].to_vec();
    Ok((frame, labels, Some(max_rows)))
}

/// Row-gather helper for the non-contiguous CombinatorialPurgedCV index sets
/// (the purged train fold is a union of groups with holes from purge/embargo,
/// so a contiguous `slice` cannot express it).
fn take_frame_rows(frame: &FeatureFrame, idx: &[usize]) -> Result<FeatureFrame> {
    frame.select_rows(idx)
}

/// Stage 1(c): score each HPO candidate with CombinatorialPurgedCV instead of a
/// single time-series holdout. Each candidate is fit + evaluated on every purged
/// path; the candidate score is `mean - stdev` of the per-path objective, so a
/// candidate that only generalizes to one lucky window is penalized — the direct
/// overfit signal. Reuses `neoethos_search::CombinatorialPurgedCV` (the SAME
/// purge+embargo machinery the gene side uses) rather than reinventing it.
/// Returns `Ok(None)` when no valid purged paths can be formed or every
/// candidate fails, so the caller falls back to the single-holdout path.
#[allow(clippy::too_many_arguments)]
fn optimize_model_config_cpcv(
    config: &ModelConfig,
    hpo_frame: &FeatureFrame,
    hpo_labels: &[i32],
    lease: &CpuLease,
    backend: &str,
    trials_requested: usize,
    base_params: &HashMap<String, String>,
    confidence_threshold: f64,
    metric_weight: f64,
    accuracy_weight: f64,
    row_budget_applied: Option<usize>,
    hpo_rows_applied: Option<usize>,
) -> Result<Option<(HashMap<String, String>, OptimizationReport)>> {
    const N_SPLITS: usize = 6;
    const N_TEST_GROUPS: usize = 2;
    const EMBARGO_PCT: f64 = 0.02;
    const PURGE_PCT: f64 = 0.01;

    let n = hpo_frame.n_samples();
    let cv = neoethos_search::CombinatorialPurgedCV::new(
        N_SPLITS,
        N_TEST_GROUPS,
        EMBARGO_PCT,
        PURGE_PCT,
    );
    let splits = cv.split(n);
    if splits.is_empty() {
        return Ok(None);
    }
    let n_paths = splits.len();

    let mut best_params = base_params.clone();
    let mut best_score = f64::NEG_INFINITY; // mean-minus-stdev aggregate
    let mut best_metrics: Option<ValidationMetrics> = None;
    let mut best_trial_index = 0usize;
    let mut trials: Vec<OptimizationTrialRecord> = Vec::new();

    for trial_idx in 0..trials_requested {
        let candidate_params = if trial_idx == 0 {
            base_params.clone()
        } else {
            generate_hpo_candidate_params(config, base_params, trial_idx, trials_requested, backend)
        };

        let mut fold_scores: Vec<f64> = Vec::with_capacity(n_paths);
        let mut repr_metrics: Option<ValidationMetrics> = None;
        let mut fold_error: Option<String> = None;

        for (train_idx, test_idx) in &splits {
            let train_frame = match take_frame_rows(hpo_frame, train_idx) {
                Ok(frame) => frame,
                Err(error) => {
                    fold_error = Some(error.to_string());
                    break;
                }
            };
            let test_frame = match take_frame_rows(hpo_frame, test_idx) {
                Ok(frame) => frame,
                Err(error) => {
                    fold_error = Some(error.to_string());
                    break;
                }
            };
            let train_labels: Vec<i32> = train_idx.iter().map(|&i| hpo_labels[i]).collect();
            let test_labels: Vec<i32> = test_idx.iter().map(|&i| hpo_labels[i]).collect();

            let mut model =
                match build_expert_model(config, hpo_frame.n_features(), &candidate_params) {
                    Ok(model) => model,
                    Err(error) => {
                        fold_error = Some(error.to_string());
                        break;
                    }
                };
            match model
                .fit_with_validation(
                    &train_frame,
                    &train_labels,
                    Some(&test_frame),
                    Some(&test_labels),
                    lease,
                )
                .and_then(|_| model.predict_proba(&test_frame, lease))
            {
                Ok(probabilities) => {
                    let metrics = evaluate_prediction_quality(
                        &probabilities,
                        &test_labels,
                        confidence_threshold,
                        metric_weight,
                        accuracy_weight,
                    )?;
                    fold_scores.push(metrics.objective_score);
                    if repr_metrics.is_none() {
                        repr_metrics = Some(metrics);
                    }
                }
                Err(error) => {
                    fold_error = Some(error.to_string());
                    break;
                }
            }
        }

        // Audit B09 (2026-07-13): require EVERY planned fold to complete. The
        // fold loop `break`s on the first failure, but any folds that had
        // already succeeded left a NON-empty `fold_scores` that was then
        // scored — so a candidate that failed a later (typically harder) fold
        // could still win on its lucky early folds, with metrics taken from
        // just the first fold. Incomplete CPCV coverage is now a rejection,
        // not a partial score.
        if fold_scores.len() != splits.len() {
            trials.push(OptimizationTrialRecord {
                index: trial_idx,
                backend: backend.to_string(),
                params: candidate_params,
                metrics: None,
                error: fold_error.or_else(|| {
                    Some(format!(
                        "CPCV incomplete: only {}/{} folds completed",
                        fold_scores.len(),
                        splits.len()
                    ))
                }),
                selected: false,
            });
            continue;
        }

        let mean = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
        let variance =
            fold_scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / fold_scores.len() as f64;
        let aggregate = mean - variance.sqrt();

        if aggregate > best_score {
            best_score = aggregate;
            best_params = candidate_params.clone();
            best_metrics = repr_metrics.clone();
            best_trial_index = trial_idx;
        }
        trials.push(OptimizationTrialRecord {
            index: trial_idx,
            backend: backend.to_string(),
            params: candidate_params,
            metrics: repr_metrics,
            error: None,
            // Marked authoritatively after the loop so exactly one is selected.
            selected: false,
        });
    }

    let trials_completed = trials
        .iter()
        .filter(|trial| trial.metrics.is_some())
        .count();
    if trials_completed == 0 {
        // Every candidate failed under CPCV — let the caller fall back to the
        // single-holdout path instead of returning an unusable report.
        return Ok(None);
    }
    if let Some(selected_trial) = trials.get_mut(best_trial_index) {
        selected_trial.selected = true;
    }

    let report = OptimizationReport {
        model_name: config.name.clone(),
        capability_family: config.capability_family,
        capability_state: config.capability_state,
        backend: backend.to_string(),
        trials_requested,
        trials_completed,
        // CPCV does not use a single holdout fraction; 0.0 is valid ([0,1)).
        holdout_pct: 0.0,
        train_rows: n,
        // Approximate per-path test size for the report (n/S * test groups).
        val_rows: (n / N_SPLITS) * N_TEST_GROUPS,
        selected_trial_index: best_trial_index,
        selected_params: best_params.clone(),
        selected_metrics: best_metrics,
        row_budget_applied,
        hpo_rows_applied,
        notes: vec![format!(
            "CombinatorialPurgedCV HPO: {N_SPLITS} splits, {N_TEST_GROUPS} test groups, \
             {n_paths} purged paths; candidate score = mean-minus-stdev of objective across paths"
        )],
        trials,
    };

    Ok(Some((best_params, report)))
}

fn optimize_model_config(
    config: &ModelConfig,
    payload: &TrainingPayload,
    lease: &CpuLease,
    base_tf: &str,
    row_budget_applied: Option<usize>,
) -> Result<(HashMap<String, String>, OptimizationReport)> {
    let backend = hpo_backend_from_params(&config.params);
    let trials_requested = if supports_hpo(config.model_type) {
        hpo_trials_from_params(&config.params)
    } else {
        1
    };
    let base_params = model_params_only(&config.params);
    let holdout_pct = holdout_pct_from_params(&config.params);
    let confidence_threshold = confidence_threshold_from_params(&config.params);
    let metric_weight = metric_weight_from_params(&config.params);
    let accuracy_weight = accuracy_weight_from_params(&config.params);
    let embargo_rows =
        embargo_rows_for_timeframe(base_tf, embargo_minutes_from_params(&config.params));
    let (hpo_frame, hpo_labels, hpo_rows_applied) =
        select_hpo_dataset(payload, hpo_max_rows_from_params(&config.params))?;

    // Stage 1(c): CombinatorialPurgedCV HPO scoring for the heavy boosters.
    // `apply_overfit_overrides` sets `__ml_cpcv` only for thick-enough heavy
    // boosters; we additionally require trials > 1 (a single trial cannot
    // benefit from cross-path variance penalization).
    if parse_bool_param(&config.params, "__ml_cpcv", false) && trials_requested > 1 {
        if let Some(result) = optimize_model_config_cpcv(
            config,
            &hpo_frame,
            &hpo_labels,
            lease,
            &backend,
            trials_requested,
            &base_params,
            confidence_threshold,
            metric_weight,
            accuracy_weight,
            row_budget_applied,
            hpo_rows_applied,
        )? {
            return Ok(result);
        }
        // Fall through to the single-holdout path when CPCV could not form
        // valid purged paths (e.g. dataset still too small after gating).
    }

    let Some((train_frame, train_labels, val_frame, val_labels)) =
        time_series_holdout_split(&hpo_frame, &hpo_labels, holdout_pct, embargo_rows, 256, 64)?
    else {
        // Audit B10: the no-HPO small-data path used to emit `trials: vec![]`
        // with no selected trial — a structurally INVALID report
        // (validate_optimization_report requires >=1 trial and exactly one
        // selected). Emit a VALID single base-params trial instead, marked
        // selected, so the report round-trips and honestly records that HPO
        // was skipped for lack of data.
        let base_trial = OptimizationTrialRecord {
            index: 0,
            backend: backend.clone(),
            params: base_params.clone(),
            metrics: None,
            error: Some(
                "dataset too small for holdout-driven HPO; base params used without validation"
                    .to_string(),
            ),
            selected: true,
        };
        let report = OptimizationReport {
            model_name: config.name.clone(),
            capability_family: config.capability_family,
            capability_state: config.capability_state,
            backend,
            trials_requested,
            trials_completed: 0,
            holdout_pct,
            train_rows: hpo_frame.n_samples(),
            val_rows: 0,
            selected_trial_index: 0,
            selected_params: base_params.clone(),
            selected_metrics: None,
            row_budget_applied,
            hpo_rows_applied,
            notes: vec!["dataset too small for holdout-driven HPO; using base params".to_string()],
            trials: vec![base_trial],
        };
        return Ok((base_params, report));
    };

    let mut best_params = base_params.clone();
    let mut best_metrics = None;
    let mut best_score = f64::NEG_INFINITY;
    let mut best_trial_index = 0_usize;
    let mut trials = Vec::new();

    for trial_idx in 0..trials_requested {
        let candidate_params = if trial_idx == 0 {
            base_params.clone()
        } else {
            generate_hpo_candidate_params(
                config,
                &base_params,
                trial_idx,
                trials_requested,
                &backend,
            )
        };

        let mut model =
            match build_expert_model(config, payload.frame.n_features(), &candidate_params) {
                Ok(model) => model,
                Err(error) => {
                    trials.push(OptimizationTrialRecord {
                        index: trial_idx,
                        backend: backend.clone(),
                        params: candidate_params,
                        metrics: None,
                        error: Some(error.to_string()),
                        selected: false,
                    });
                    continue;
                }
            };

        // M5/M6/M7: forward the explicit HPO val frame through to models
        // that support val-based early stopping (Burn deep learners,
        // gradient boosters, anomaly detectors). Models that do not opt in
        // fall back to `fit` via the default trait impl, so existing
        // models keep working without changes.
        match model
            .fit_with_validation(
                &train_frame,
                &train_labels,
                Some(&val_frame),
                Some(&val_labels),
                lease,
            )
            .and_then(|_| model.predict_proba(&val_frame, lease))
        {
            Ok(probabilities) => {
                let metrics = evaluate_prediction_quality(
                    &probabilities,
                    &val_labels,
                    confidence_threshold,
                    metric_weight,
                    accuracy_weight,
                )?;
                if metrics.objective_score > best_score {
                    best_score = metrics.objective_score;
                    best_metrics = Some(metrics.clone());
                    best_params = candidate_params.clone();
                    best_trial_index = trial_idx;
                }
                trials.push(OptimizationTrialRecord {
                    index: trial_idx,
                    backend: backend.clone(),
                    params: candidate_params,
                    metrics: Some(metrics),
                    error: None,
                    // Marked authoritatively after the loop so EXACTLY ONE trial
                    // is selected (validate_optimization_report requires it).
                    // Previously every improving trial was marked, so a
                    // monotonically-improving HPO run failed validation and the
                    // model (e.g. nbeats) was lost.
                    selected: false,
                });
            }
            Err(error) => {
                trials.push(OptimizationTrialRecord {
                    index: trial_idx,
                    backend: backend.clone(),
                    params: candidate_params,
                    metrics: None,
                    error: Some(error.to_string()),
                    selected: false,
                });
            }
        }
    }

    if let Some(selected_trial) = trials.get_mut(best_trial_index) {
        selected_trial.selected = true;
    }

    let trials_completed = trials
        .iter()
        .filter(|trial| trial.metrics.is_some())
        .count();
    let notes = if trials_completed == 0 {
        vec!["all HPO trials failed; falling back to base params".to_string()]
    } else {
        Vec::new()
    };

    let report = OptimizationReport {
        model_name: config.name.clone(),
        capability_family: config.capability_family,
        capability_state: config.capability_state,
        backend,
        trials_requested,
        trials_completed,
        holdout_pct,
        train_rows: train_frame.n_samples(),
        val_rows: val_frame.n_samples(),
        selected_trial_index: best_trial_index,
        selected_params: best_params.clone(),
        selected_metrics: best_metrics,
        row_budget_applied,
        hpo_rows_applied,
        notes,
        trials,
    };

    Ok((best_params, report))
}

/// v0.5 ML-integration Stage 1(b): per-(symbol,TF) overfit gate + tree-budget
/// scaling for the heavy gradient boosters, applied at train time where the
/// real bar count (`payload.frame.n_samples()`) is known. No-op when
/// `regularized_model_defaults` is off or the model is not one of the heavy
/// boosters (xgboost/xgboost_rf/xgboost_dart/lightgbm/catboost/catboost_alt).
///
/// - Tree budget scales with bars: <3000 -> 200, 3000-10000 -> 400, >10000 -> 800.
/// - Below `heavy_booster_min_bars` (default 4000; D1 ~2700 falls below) the
///   booster is forced onto a shrunk preset (shallow depth, few trees, few
///   leaves) AND per-bar HPO is disabled (`__hpo_trials=1`) — a thin holdout
///   cannot reliably select 5+ hyperparameters. Below an absolute floor (800)
///   an even tinier preset is used. The model still trains and votes (we do not
///   silently drop a voter); it is simply low-variance.
fn apply_overfit_overrides(
    settings: &neoethos_core::Settings,
    config: &ModelConfig,
    bars: usize,
) -> ModelConfig {
    let canonical = canonical_model_name(&config.name);
    let is_heavy = matches!(
        canonical,
        "xgboost" | "xgboost_rf" | "xgboost_dart" | "lightgbm" | "catboost" | "catboost_alt"
    );
    if !is_heavy {
        return config.clone();
    }

    let mut params = config.params.clone();
    let min_bars = settings.models.heavy_booster_min_bars;
    let thin = min_bars > 0 && bars < min_bars;
    let very_thin = thin && bars < 800;

    if settings.models.regularized_model_defaults {
        // Bar-scaled tree budget (early stopping on the val frame still caps it).
        let trees = if thin {
            if very_thin { 80 } else { 150 }
        } else if bars < 3000 {
            200
        } else if bars <= 10_000 {
            400
        } else {
            800
        };
        let depth: Option<i64> = if thin {
            Some(if very_thin { 2 } else { 3 })
        } else {
            None
        };
        let leaves: Option<i64> = if thin {
            Some(if very_thin { 4 } else { 7 })
        } else {
            None
        };

        match canonical {
            "xgboost" | "xgboost_rf" | "xgboost_dart" => {
                params.insert("n_estimators".into(), trees.to_string());
                if let Some(d) = depth {
                    params.insert("max_depth".into(), d.to_string());
                }
            }
            "lightgbm" => {
                params.insert("num_iterations".into(), trees.to_string());
                // Leaf-size floor scales with data (min 20) — single-row leaves
                // are the classic LightGBM overfit on thin TFs.
                let min_leaf = (bars / 200).max(20);
                params.insert("min_data_in_leaf".into(), min_leaf.to_string());
                if let Some(d) = depth {
                    params.insert("max_depth".into(), d.to_string());
                }
                if let Some(l) = leaves {
                    params.insert("num_leaves".into(), l.to_string());
                }
            }
            "catboost" | "catboost_alt" => {
                params.insert("iterations".into(), trees.to_string());
                if let Some(d) = depth {
                    params.insert("depth".into(), d.to_string());
                }
            }
            _ => {}
        }

        if thin {
            // A thin holdout cannot select 5+ hyperparameters — disable per-bar HPO.
            params.insert("__hpo_trials".into(), "1".into());
            tracing::info!(
                target: "neoethos_models::training",
                model = %config.name,
                bars,
                heavy_booster_min_bars = min_bars,
                trees,
                "thin-data overfit gate: shrunk booster preset + HPO disabled"
            );
        }
    }

    // Stage 1(c): enable CombinatorialPurgedCV HPO scoring for the heavy
    // boosters when the data is thick enough (purged 15-path CV is wasteful and
    // unstable on thin data, which is also forced to trials=1 above). The
    // `optimize_model_config` reader additionally requires trials > 1.
    if settings.models.ml_cpcv_enabled && !thin {
        params.insert("__ml_cpcv".into(), "1".into());
    }

    ModelConfig {
        name: config.name.clone(),
        model_type: config.model_type,
        capability_family: config.capability_family,
        capability_state: config.capability_state,
        params,
    }
}

#[allow(clippy::too_many_arguments)]
fn train_model_dispatch(
    models_dir: &std::path::Path,
    settings: &neoethos_core::Settings,
    symbol: &str,
    base_tf: &str,
    row_budget_applied: Option<usize>,
    config: &ModelConfig,
    payload: &TrainingPayload,
    lease: &CpuLease,
    // Pre-built exact `(quant_log_return, quant_log_volatility)` observations
    // from the versioned feature frame before generic feature selection.
    // `Err(reason)` is surfaced loudly only if hmm_regime is dispatched.
    hmm_observations: &std::result::Result<ndarray::Array2<f64>, String>,
) -> Result<()> {
    let artifact_dir = model_artifact_dir(models_dir, symbol, base_tf, &config.name);

    if uses_shared_expert_dispatch(config.model_type) {
        // Stage 1(b): apply the bar-count overfit gate BEFORE HPO so the
        // selection respects the gated budget.
        let gated = apply_overfit_overrides(settings, config, payload.frame.n_samples());
        let config = &gated;
        let (selected_params, optimization_report) =
            optimize_model_config(config, payload, lease, base_tf, row_budget_applied)?;
        let effective_config = ModelConfig {
            name: config.name.clone(),
            model_type: config.model_type,
            capability_family: config.capability_family,
            capability_state: config.capability_state,
            params: selected_params.clone(),
        };
        let mut model = build_expert_model(
            &effective_config,
            payload.frame.n_features(),
            &selected_params,
        )?;
        // M3: After HPO selects best params on a train/val split, the standard
        // ML practice is to refit on the FULL dataset with those params for
        // production deployment. We do that here, but we also stamp the
        // optimisation report's split metadata onto the saved artifact so a
        // reviewer can see that the val rows were not held out from the
        // deployed model. Without this, `default_training_summary` records
        // `train_rows = dataset_rows, val_rows = 0`, hiding the leakage that
        // any "val score" reported afterwards is upper-biased.
        tracing::info!(
            target: "neoethos_models::training",
            model = %config.name,
            full_refit_rows = payload.frame.n_samples(),
            hpo_train_rows = optimization_report.train_rows,
            hpo_val_rows = optimization_report.val_rows,
            "post-HPO full-data refit (val rows were used for HPO selection only)"
        );
        model.fit(payload.frame.as_ref(), payload.labels.as_ref(), lease)?;
        persist_training_artifacts(
            &artifact_dir,
            settings,
            &effective_config,
            symbol,
            base_tf,
            payload,
            row_budget_applied,
            Some(&optimization_report),
            |staged_dir| model.save(staged_dir),
        )?;
        return Ok(());
    }

    match config.model_type {
        ModelType::SklearsTree => {
            let mut model = SklearsTreeExpert::new();
            model.fit(payload.frame.as_ref(), payload.labels.as_ref(), lease)?;
            persist_training_artifacts(
                &artifact_dir,
                settings,
                config,
                symbol,
                base_tf,
                payload,
                row_budget_applied,
                None,
                |staged_dir| model.save(staged_dir),
            )?;
            Ok(())
        }
        ModelType::ExitAgent => {
            let mut model = ExitAgent::with_hidden_dim(
                payload.frame.n_features().max(1),
                parse_usize_param(&config.params, "hidden_dim", 64),
            )
            .with_device_policy(
                parse_string_param(&config.params, "device").unwrap_or_else(|| "auto".to_string()),
            )?
            .with_gamma(parse_f32_param(&config.params, "gamma", 0.99))
            .with_epsilon(parse_f32_param(&config.params, "epsilon", 0.20))
            .with_exploration_schedule(
                parse_f32_param(&config.params, "epsilon_min", 0.05),
                parse_f32_param(&config.params, "epsilon_decay", 0.999),
            )
            .with_memory_capacity(parse_usize_param(&config.params, "memory_capacity", 10_000))
            .with_reward_horizon(parse_usize_param(&config.params, "reward_horizon", 0))
            .with_warmup_steps(parse_usize_param(&config.params, "warmup_steps", 0));
            model.fit_from_frame(payload.frame.as_ref(), payload.labels.as_ref(), lease)?;
            persist_training_artifacts(
                &artifact_dir,
                settings,
                config,
                symbol,
                base_tf,
                payload,
                row_budget_applied,
                None,
                |staged_dir| model.save(staged_dir),
            )?;
            Ok(())
        }
        ModelType::SacAgent => {
            let mut model = SoftActorCritic::with_hidden_dim(
                payload.frame.n_features().max(1),
                parse_usize_param(&config.params, "hidden_dim", 256),
            )
            .with_device_policy(
                parse_string_param(&config.params, "device").unwrap_or_else(|| "auto".to_string()),
            )?
            .with_gamma(parse_f32_param(&config.params, "gamma", 0.99))
            .with_tau(parse_f32_param(&config.params, "tau", 0.01))
            .with_learning_rate(parse_f64_param(&config.params, "learning_rate", 3e-4))
            .with_target_entropy_scale(parse_f32_param(
                &config.params,
                "target_entropy_scale",
                0.98,
            ))
            .with_train_schedule(
                parse_usize_param(&config.params, "epochs", 32),
                parse_usize_param(&config.params, "batch_size", 64),
            )
            .with_episode_layout(
                parse_usize_param(&config.params, "reward_horizon", 0),
                parse_usize_param(&config.params, "episode_len", 0),
            );
            model.train_on_frame(payload.frame.as_ref(), payload.labels.as_ref(), lease)?;
            persist_training_artifacts(
                &artifact_dir,
                settings,
                config,
                symbol,
                base_tf,
                payload,
                row_budget_applied,
                None,
                |staged_dir| model.save(staged_dir),
            )?;
            Ok(())
        }
        ModelType::Dqn => {
            let mut model = TradingReinforcementLearner::new()
                .with_state_bins(parse_usize_param(&config.params, "state_bins", 255) as u16)
                .with_encoding_name(
                    parse_string_param(&config.params, "state_encoding")
                        .as_deref()
                        .unwrap_or("normalized"),
                )
                .with_train_schedule(
                    parse_usize_param(&config.params, "epochs", 48),
                    parse_usize_param(&config.params, "max_steps", 512),
                    parse_usize_param(&config.params, "batch_size", 64),
                )
                .with_update_schedule(
                    parse_usize_param(&config.params, "update_interval", 32),
                    parse_usize_param(&config.params, "update_freq", 4),
                )
                .with_optimizer(
                    parse_f64_param(&config.params, "learning_rate", 1e-3),
                    parse_f32_param(&config.params, "gamma", 0.99),
                )
                .with_exploration_schedule(
                    parse_f32_param(&config.params, "epsilon_start", 1.0),
                    parse_f32_param(&config.params, "epsilon_end", 0.02),
                    parse_f32_param(&config.params, "epsilon_decay", 0.995),
                )
                .with_buffer_capacity(parse_usize_param(&config.params, "buffer_capacity", 50_000))
                .with_runtime_hints(
                    parse_string_param(&config.params, "backend")
                        .unwrap_or_else(|| "rlkit".to_string()),
                    parse_string_param(&config.params, "device")
                        .unwrap_or_else(|| "auto".to_string()),
                    parse_usize_param(&config.params, "parallel_envs", 1),
                    parse_usize_param(&config.params, "eval_episodes", 8),
                    0,
                    parse_usize_param(&config.params, "ray_tune_max_concurrency", 1),
                )
                .with_episode_layout(
                    parse_usize_param(&config.params, "reward_horizon", 0),
                    parse_usize_param(&config.params, "episode_len", 0),
                );
            if let Some(hidden_dims) = config.params.get("hidden_dims") {
                let parsed = hidden_dims
                    .split(',')
                    .filter_map(|value| value.trim().parse::<usize>().ok())
                    .filter(|value| *value > 0)
                    .collect::<Vec<_>>();
                if !parsed.is_empty() {
                    model = model.with_hidden_dims(parsed);
                }
            }
            model.train_on_frame(payload.frame.as_ref(), payload.labels.as_ref(), lease)?;
            persist_training_artifacts(
                &artifact_dir,
                settings,
                config,
                symbol,
                base_tf,
                payload,
                row_budget_applied,
                None,
                |staged_dir| model.save(staged_dir),
            )?;
            Ok(())
        }
        ModelType::SwarmForecaster => {
            let mut model =
                SwarmForecaster::new(parse_f64_param(&config.params, "memory_limit_mb", 256.0));
            if let Some(horizon) = config
                .params
                .get("horizon")
                .and_then(|value| value.parse::<usize>().ok())
            {
                model.config.horizon = horizon.max(1);
            }
            if let Some(frequency) = parse_string_param(&config.params, "frequency") {
                model.config.frequency = frequency;
            }
            if let Some(strategy) = parse_string_param(&config.params, "strategy") {
                model.config.strategy = match strategy.trim().to_ascii_lowercase().as_str() {
                    "simple" | "simple_average" => {
                        crate::forecasting::swarm_impl::SwarmEnsembleStrategy::SimpleAverage
                    }
                    "weighted" | "weighted_average" => {
                        crate::forecasting::swarm_impl::SwarmEnsembleStrategy::WeightedAverage
                    }
                    "median" => crate::forecasting::swarm_impl::SwarmEnsembleStrategy::Median,
                    "trimmed" | "trimmed_mean" => {
                        crate::forecasting::swarm_impl::SwarmEnsembleStrategy::TrimmedMean
                    }
                    _ => {
                        crate::forecasting::swarm_impl::SwarmEnsembleStrategy::BayesianModelAveraging
                    }
                };
            }
            model.config.accuracy_target = parse_f32_param(
                &config.params,
                "accuracy_target",
                model.config.accuracy_target,
            );
            model.config.latency_requirement_ms = parse_f32_param(
                &config.params,
                "latency_ms",
                model.config.latency_requirement_ms,
            );
            model.config.online_learning = parse_bool_param(
                &config.params,
                "online_learning",
                model.config.online_learning,
            );
            model.config.interpretability_needed = parse_bool_param(
                &config.params,
                "interpretability_needed",
                model.config.interpretability_needed,
            );
            model.fit_from_frame(payload.frame.as_ref(), symbol, lease)?;
            persist_training_artifacts(
                &artifact_dir,
                settings,
                config,
                symbol,
                base_tf,
                payload,
                row_budget_applied,
                None,
                |staged_dir| model.save(staged_dir),
            )?;
            Ok(())
        }
        ModelType::HmmRegime => {
            // Baum-Welch EM over the pre-built (log_return, log_volatility)
            // matrix — unsupervised, seconds of CPU, honours the same OOS
            // cutoff as the supervised experts (sliced upstream).
            let observations = match hmm_observations {
                Ok(obs) => obs,
                Err(reason) => anyhow::bail!(
                    "hmm_regime: could not derive (log_return, log_volatility) \
                     observations from the base OHLCV: {reason}"
                ),
            };
            let model = lease.scope(|| {
                RegimeHmmExpert::train(
                    observations,
                    vec![
                        "quant_log_return".to_string(),
                        "quant_log_volatility".to_string(),
                    ],
                    HmmRegimeConfig::default(),
                )
            })?;
            persist_training_artifacts(
                &artifact_dir,
                settings,
                config,
                symbol,
                base_tf,
                payload,
                row_budget_applied,
                None,
                |staged_dir| model.save_to_path(staged_dir),
            )?;
            Ok(())
        }
        ModelType::Genetic => {
            let mut model = GeneticStrategyExpert::new(
                parse_usize_param(&config.params, "population", 64),
                parse_usize_param(&config.params, "generations", 12),
                parse_usize_param(&config.params, "max_indicators", 8),
            )?
            .with_portfolio_size(parse_usize_param(&config.params, "portfolio_size", 12))
            .with_search_policy(
                parse_parent_selection_policy(&config.params),
                parse_survivor_selection_policy(&config.params),
                parse_f64_param(&config.params, "survivor_fraction", 0.10),
                parse_f64_param(&config.params, "immigrant_fraction", 0.18),
                parse_f64_param(&config.params, "selection_temperature", 0.75),
                parse_usize_param(&config.params, "tournament_size", 3),
            );
            model.fit(
                payload.frame.as_ref(),
                payload.labels.as_ref(),
                Some(symbol),
                lease,
            )?;
            persist_training_artifacts(
                &artifact_dir,
                settings,
                config,
                symbol,
                base_tf,
                payload,
                row_budget_applied,
                None,
                |staged_dir| model.save(staged_dir),
            )?;
            Ok(())
        }
        other => anyhow::bail!(
            "training dispatch contract drift for model `{}` ({:?}); registry/orchestrator mapping is inconsistent",
            config.name,
            other
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::get_model_capability;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn orchestrator_with_models(models: &[&str]) -> TrainingOrchestrator {
        let mut settings = neoethos_core::Settings::default();
        settings.models.ml_models = models.iter().map(|name| (*name).to_string()).collect();
        settings.models.phase5_core_models.clear();
        settings.models.phase5_filter_meta_blender = false;
        settings.models.calibration_enabled = false;
        settings.models.use_rl_agent = false;
        settings.models.use_sac_agent = false;
        settings.models.use_rllib_agent = false;
        settings.models.use_neuroevolution = false;
        settings.risk.conformal_enabled = false;
        TrainingOrchestrator::new(settings, PathBuf::from("models"))
    }

    fn unique_test_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("current time should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "neoethos_training_orchestrator_{name}_{}_{}",
            std::process::id(),
            nanos
        ))
    }

    fn test_training_payload(
        matrix: ndarray::Array2<f64>,
        feature_names: Vec<String>,
        labels: Vec<i32>,
    ) -> TrainingPayload {
        let timestamps = neoethos_data::test_fixtures::canonical_test_timestamps(matrix.nrows());
        let frame = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_matrix(
            timestamps,
            feature_names,
            matrix,
        )
        .expect("build typed test feature frame");
        TrainingPayload::from_frame(frame, labels).expect("build typed training payload")
    }

    fn validity_test_column(
        name: &str,
        rows: usize,
        valid_from: usize,
        invalid_every: Option<usize>,
    ) -> neoethos_data::FeatureColumnF64 {
        let mut values = Vec::with_capacity(rows);
        let mut validity = Vec::with_capacity(rows);
        for row in 0..rows {
            let valid = row >= valid_from
                && invalid_every.is_none_or(|stride| (row - valid_from) % stride != 0);
            if valid {
                values.push(row as f64);
                validity.push(neoethos_data::FeatureCellValidity::Valid);
            } else {
                values.push(f64::NAN);
                validity.push(neoethos_data::FeatureCellValidity::Warmup);
            }
        }
        neoethos_data::FeatureColumnF64::new(name, values, validity)
            .expect("build typed validity test column")
    }

    #[test]
    fn dense_training_suffix_maximizes_valid_cells_and_preserves_required_hmm_inputs() {
        let rows = 1_000;
        let timestamps = neoethos_data::test_fixtures::canonical_test_timestamps(rows);
        let frame = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
            timestamps,
            vec![
                validity_test_column("quant_log_return", rows, 1, None),
                validity_test_column("quant_log_volatility", rows, 2, None),
                validity_test_column("dense_signal", rows, 100, None),
                validity_test_column("intermittent_signal", rows, 0, Some(2)),
                validity_test_column("never_valid", rows, rows, None),
            ],
        )
        .expect("build dense suffix fixture");

        let projected =
            project_dense_training_suffix(&frame).expect("project exact dense training suffix");
        assert_eq!(projected.row_start, 100);
        assert_eq!(projected.dropped_columns, 2);
        assert_eq!(projected.frame.n_samples(), 900);
        assert_eq!(
            projected.frame.names,
            vec!["quant_log_return", "quant_log_volatility", "dense_signal"]
        );
        for feature in 0..projected.frame.n_features() {
            assert!(
                projected
                    .frame
                    .feature_column(feature)
                    .expect("projected column")
                    .validity
                    .iter()
                    .all(|cell| cell.is_valid())
            );
        }
    }

    #[test]
    fn dense_training_suffix_fails_when_a_required_hmm_input_has_too_little_support() {
        let rows = 1_000;
        let timestamps = neoethos_data::test_fixtures::canonical_test_timestamps(rows);
        let frame = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
            timestamps,
            vec![
                validity_test_column("quant_log_return", rows, 1, None),
                validity_test_column("quant_log_volatility", rows, 501, None),
                validity_test_column("dense_signal", rows, 0, None),
            ],
        )
        .expect("build insufficient-support fixture");

        let error = project_dense_training_suffix(&frame)
            .expect_err("required HMM support below 500 rows must fail closed");
        assert!(
            error
                .to_string()
                .contains("quant_log_volatility` has only 499 trailing valid rows")
        );
    }

    fn persist_sample_tree_training_artifacts(name: &str) -> PathBuf {
        let settings = neoethos_core::Settings::default();
        let config = ModelConfig {
            name: "lightgbm".to_string(),
            model_type: ModelType::LightGBM,
            capability_family: crate::runtime::capabilities::ModelFamily::Tree,
            capability_state: CapabilityState::Verified,
            params: HashMap::from([
                ("__planned_backend".to_string(), "cpu".to_string()),
                ("__planned_device".to_string(), "cpu".to_string()),
            ]),
        };
        let payload = test_training_payload(
            ndarray::arr2(&[[1.0, 0.1], [1.1, 0.2], [1.2, 0.3], [1.3, 0.4]]),
            vec!["return_1".to_string(), "volatility_3".to_string()],
            vec![0, 1, 0, 1],
        );
        let artifact_dir = unique_test_dir(name);

        persist_training_artifacts(
            &artifact_dir,
            &settings,
            &config,
            "EURUSD",
            "M15",
            &payload,
            Some(4),
            None,
            |staged_dir| {
                std::fs::write(staged_dir.join("model.bin"), b"model")
                    .context("write model marker")?;
                Ok(())
            },
        )
        .expect("persist training artifacts");

        artifact_dir
    }

    #[test]
    fn create_dispatch_plan_rejects_empty_model_config() {
        let orchestrator = orchestrator_with_models(&[]);
        let err = orchestrator
            .create_dispatch_plan()
            .expect_err("expected empty-config error");
        assert!(err.to_string().contains("no model names"));
    }

    // v0.5 ML-integration Stage 1(a) guard: the regularized seed map is the
    // REAL production source for the named boosters (verdict #3). If these keys
    // ever regress out of `default_model_params`, the booster silently trains
    // with the legacy unregularized values and a before/after OOS shows no
    // effect — this test fails loudly instead.
    #[test]
    fn regularized_seed_maps_contain_reg_keys() {
        let orch = orchestrator_with_models(&["xgboost"]);
        assert!(orch.settings.models.regularized_model_defaults);

        let xgb = orch.default_model_params("xgboost");
        assert_eq!(xgb.get("max_depth").map(String::as_str), Some("4"));
        assert_eq!(xgb.get("subsample").map(String::as_str), Some("0.8"));
        assert_eq!(xgb.get("colsample_bytree").map(String::as_str), Some("0.8"));
        assert_eq!(xgb.get("min_child_weight").map(String::as_str), Some("10"));
        assert_eq!(xgb.get("reg_lambda").map(String::as_str), Some("5.0"));
        assert_eq!(xgb.get("reg_alpha").map(String::as_str), Some("0.5"));
        assert_eq!(xgb.get("gamma").map(String::as_str), Some("0.5"));

        let lgbm = orch.default_model_params("lightgbm");
        assert_eq!(lgbm.get("num_leaves").map(String::as_str), Some("15"));
        assert_eq!(lgbm.get("max_depth").map(String::as_str), Some("4"));
        assert_eq!(
            lgbm.get("feature_fraction").map(String::as_str),
            Some("0.8")
        );
        assert_eq!(
            lgbm.get("bagging_fraction").map(String::as_str),
            Some("0.8")
        );
        assert_eq!(lgbm.get("lambda_l2").map(String::as_str), Some("5.0"));
        assert!(lgbm.contains_key("min_data_in_leaf"));

        let cat = orch.default_model_params("catboost");
        assert_eq!(cat.get("depth").map(String::as_str), Some("4"));
        assert_eq!(cat.get("l2_leaf_reg").map(String::as_str), Some("6.0"));
    }

    #[test]
    fn legacy_seed_maps_restore_unregularized_defaults() {
        let mut orch = orchestrator_with_models(&["xgboost"]);
        orch.settings.models.regularized_model_defaults = false;

        let xgb = orch.default_model_params("xgboost");
        assert_eq!(xgb.get("max_depth").map(String::as_str), Some("8"));
        assert_eq!(xgb.get("n_estimators").map(String::as_str), Some("800"));
        // Regularizers are NOT seeded in legacy mode (the booster's neutral
        // inline fallbacks then apply == legacy behaviour).
        assert!(!xgb.contains_key("subsample"));
        assert!(!xgb.contains_key("reg_lambda"));

        let lgbm = orch.default_model_params("lightgbm");
        assert_eq!(lgbm.get("num_leaves").map(String::as_str), Some("31"));
        assert!(!lgbm.contains_key("feature_fraction"));
    }

    #[test]
    fn overfit_gate_shrinks_and_disables_hpo_on_thin_data() {
        let orch = orchestrator_with_models(&["xgboost"]);
        let base = ModelConfig {
            name: "xgboost".to_string(),
            model_type: ModelType::XGBoost,
            capability_family: crate::runtime::capabilities::ModelFamily::Tree,
            capability_state: CapabilityState::Verified,
            params: orch.default_model_params("xgboost"),
        };

        // Thin (D1-like): below the 4000-bar gate -> shrunk preset + HPO off,
        // and NO CPCV meta flag (15-path CV is wasteful on thin data).
        let thin = apply_overfit_overrides(&orch.settings, &base, 2700);
        assert_eq!(thin.params.get("max_depth").map(String::as_str), Some("3"));
        assert_eq!(
            thin.params.get("n_estimators").map(String::as_str),
            Some("150")
        );
        assert_eq!(
            thin.params.get("__hpo_trials").map(String::as_str),
            Some("1")
        );
        assert!(!thin.params.contains_key("__ml_cpcv"));

        // Thick: bar-scaled budget, no shrink, CPCV enabled.
        let thick = apply_overfit_overrides(&orch.settings, &base, 50_000);
        assert_eq!(
            thick.params.get("n_estimators").map(String::as_str),
            Some("800")
        );
        assert_eq!(thick.params.get("max_depth").map(String::as_str), Some("4"));
        assert!(!thick.params.contains_key("__hpo_trials"));
        assert_eq!(thick.params.get("__ml_cpcv").map(String::as_str), Some("1"));
    }

    #[test]
    fn build_dispatch_plan_orders_and_deduplicates_enabled_models() {
        let orchestrator = orchestrator_with_models(&["mlp", "lightgbm", "patchtst", "lightgbm"]);
        let plan = orchestrator
            .create_dispatch_plan()
            .expect("dispatch plan should build");

        let names: Vec<_> = plan
            .entries
            .iter()
            .map(|entry| entry.name.as_str())
            .collect();
        // hmm_regime joins every non-empty plan (regime gate — wired 2026-07-11).
        assert_eq!(names, vec!["hmm_regime", "lightgbm", "mlp", "patchtst"]);
    }

    #[test]
    fn default_configured_inventory_resolves_to_capabilities_and_deterministic_plan() {
        let settings = neoethos_core::Settings::default();
        let orchestrator = TrainingOrchestrator::new(settings.clone(), PathBuf::from("models"));

        for name in &settings.models.ml_models {
            let capability = get_model_capability(name)
                .unwrap_or_else(|| panic!("missing capability for configured model {name}"));
            assert_eq!(capability.name, *name);
        }

        let first_plan = orchestrator
            .create_dispatch_plan()
            .expect("dispatch plan should build from default settings");
        let second_plan = orchestrator
            .create_dispatch_plan()
            .expect("dispatch plan should be deterministic");

        let first_names: Vec<_> = first_plan
            .entries
            .iter()
            .map(|entry| entry.name.as_str())
            .collect();
        let second_names: Vec<_> = second_plan
            .entries
            .iter()
            .map(|entry| entry.name.as_str())
            .collect();

        assert_eq!(first_names, second_names);
        assert_eq!(first_plan.entries.len(), second_plan.entries.len());
        assert!(
            !first_plan.entries.is_empty(),
            "default dispatch plan should not be empty"
        );

        for entry in &first_plan.entries {
            let capability = get_model_capability(&entry.name)
                .unwrap_or_else(|| panic!("missing capability for dispatch entry {}", entry.name));
            assert_eq!(entry.family, capability.family);
            assert_eq!(entry.state, capability.state);
        }
    }

    #[test]
    fn validate_dispatch_plan_accepts_implemented_capabilities() {
        let orchestrator = orchestrator_with_models(&["xgboost", "mlp"]);
        let plan = orchestrator
            .create_dispatch_plan()
            .expect("dispatch plan should build");

        orchestrator
            .validate_dispatch_plan(&plan)
            .expect("implemented capabilities should be runnable");
    }

    #[test]
    fn build_training_configs_maps_tree_variants_to_base_trainers_with_params() {
        let orchestrator = orchestrator_with_models(&["catboost_alt"]);
        let mut plan = orchestrator
            .create_dispatch_plan()
            .expect("dispatch plan should build");
        for entry in &mut plan.entries {
            entry.state = CapabilityState::Verified;
        }

        let configs = orchestrator
            .build_training_configs(&plan)
            .expect("tree variants should map to concrete base trainers");
        // catboost_alt + the always-on hmm_regime regime gate.
        assert_eq!(configs.len(), 2);
        let cat = configs
            .iter()
            .find(|c| c.name == "catboost_alt")
            .expect("catboost_alt config present");
        assert_eq!(cat.model_type, ModelType::CatBoost);
        assert_eq!(cat.params.get("variant"), Some(&"alt".to_string()));
    }

    #[test]
    fn hardware_plan_params_reach_the_deep_trainer_contract() {
        use neoethos_core::system::{
            AcceleratorBackend, AcceleratorDevice, AcceleratorDeviceClass,
            HARDWARE_PROFILE_SCHEMA_VERSION, HardwareProfile, TrainingPrecision,
        };

        let mut orchestrator = orchestrator_with_models(&["mlp"]);
        orchestrator.settings.system.enable_gpu_preference = "cuda".to_string();
        orchestrator.settings.system.hardware.cpu_budget = Some(5);
        let profile = HardwareProfile {
            schema_version: HARDWARE_PROFILE_SCHEMA_VERSION,
            cpu_cores: 12,
            total_ram_gb: 32.0,
            available_ram_gb: 20.0,
            gpu_names: vec!["Test GPU".to_string()],
            num_gpus: 1,
            gpu_mem_gb: vec![8.0],
            accelerator_devices: vec![AcceleratorDevice {
                id: 0,
                name: "Test GPU".to_string(),
                backend: AcceleratorBackend::Cuda,
                device_class: AcceleratorDeviceClass::DiscreteGpu,
                backend_index: 0,
                memory_gb: 8.0,
                supported_precisions: vec![TrainingPrecision::Fp32],
                compute_capability: Some((8, 0)),
                source: "test".to_string(),
            }],
            timestamp: "test".to_string(),
            platform_label: "test".to_string(),
        };
        let plan =
            HardwareExecutionPlan::from_settings_and_profile(&orchestrator.settings, profile);
        let workload = plan
            .workload(WorkloadKind::DeepTraining)
            .expect("deep-training workload should exist");
        let mut params = HashMap::from([("batch_size".to_string(), "17".to_string())]);

        orchestrator.apply_hardware_plan_params("mlp", ModelFamily::Deep, &plan, &mut params);

        assert_eq!(
            params.get("__planned_cpu_threads"),
            Some(&workload.cpu_threads.to_string())
        );
        assert_eq!(
            params.get("__planned_batch_size"),
            Some(&workload.batch_size.to_string())
        );
        assert_eq!(
            params.get("__planned_memory_budget_gb"),
            Some(&format!("{:.6}", workload.memory_budget_gb))
        );
        assert_eq!(params.get("cpu_threads"), Some(&"5".to_string()));
        assert_eq!(
            params.get("batch_size"),
            Some(&workload.batch_size.to_string())
        );
        assert_eq!(
            params.get("memory_budget_gb"),
            Some(&format!("{:.6}", workload.memory_budget_gb))
        );
    }

    #[test]
    fn model_overrides_cannot_escape_the_resolved_hardware_plan() {
        let mut orchestrator = orchestrator_with_models(&["mlp"]);
        orchestrator.settings.models.model_param_overrides.insert(
            "mlp".to_string(),
            HashMap::from([
                ("cpu_threads".to_string(), usize::MAX.to_string()),
                ("batch_size".to_string(), usize::MAX.to_string()),
                ("memory_budget_gb".to_string(), f64::MAX.to_string()),
                ("device".to_string(), "definitely-invalid".to_string()),
                ("training_precision".to_string(), "invalid".to_string()),
            ]),
        );
        let dispatch_plan = orchestrator
            .create_dispatch_plan()
            .expect("dispatch plan should build");

        let configs = orchestrator
            .build_training_configs(&dispatch_plan)
            .expect("training configs should build");
        let mlp = configs
            .iter()
            .find(|config| config.name == "mlp")
            .expect("mlp config should exist");

        assert_eq!(
            mlp.params.get("cpu_threads"),
            mlp.params.get("__planned_cpu_threads")
        );
        assert_eq!(
            mlp.params.get("batch_size"),
            mlp.params.get("__planned_batch_size")
        );
        assert_eq!(
            mlp.params.get("memory_budget_gb"),
            mlp.params.get("__planned_memory_budget_gb")
        );
        assert_ne!(
            mlp.params.get("device").map(String::as_str),
            Some("definitely-invalid")
        );
        assert_ne!(
            mlp.params.get("training_precision").map(String::as_str),
            Some("invalid")
        );
    }

    #[test]
    fn hpo_candidates_cannot_exceed_the_planned_batch_size() {
        let base_params = HashMap::from([
            ("batch_size".to_string(), "32".to_string()),
            ("__planned_batch_size".to_string(), "32".to_string()),
        ]);
        let config = ModelConfig {
            name: "mlp".to_string(),
            model_type: ModelType::MLP,
            capability_family: ModelFamily::Deep,
            capability_state: CapabilityState::Verified,
            params: base_params.clone(),
        };

        for trial_idx in 1..16 {
            let candidate =
                generate_hpo_candidate_params(&config, &base_params, trial_idx, 16, "tpe");
            let batch_size = candidate["batch_size"]
                .parse::<usize>()
                .expect("batch size should be numeric");
            assert!(
                batch_size <= 32,
                "trial {trial_idx} escaped the planned batch-size cap: {batch_size}"
            );
        }
    }

    #[test]
    fn build_training_configs_maps_configured_models_to_concrete_model_types() {
        let orchestrator =
            orchestrator_with_models(&["lightgbm", "online_pa", "patchtst", "genetic"]);
        let plan = orchestrator
            .create_dispatch_plan()
            .expect("dispatch plan should build");

        let configs = orchestrator
            .build_training_configs(&plan)
            .expect("configured models should map to concrete model types");
        let pairs = configs
            .iter()
            .map(|config| (config.name.as_str(), config.model_type))
            .collect::<Vec<_>>();

        assert_eq!(
            pairs,
            vec![
                ("genetic", ModelType::Genetic),
                ("hmm_regime", ModelType::HmmRegime),
                ("lightgbm", ModelType::LightGBM),
                ("online_pa", ModelType::OnlinePassiveAggressive),
                ("patchtst", ModelType::PatchTST),
            ]
        );
    }

    #[test]
    fn build_training_configs_maps_neat_to_concrete_model_type() {
        let orchestrator = orchestrator_with_models(&["neat"]);
        let plan = orchestrator
            .create_dispatch_plan()
            .expect("dispatch plan should build");

        let configs = orchestrator
            .build_training_configs(&plan)
            .expect("neat should map to a concrete model type");
        // neat + the always-on hmm_regime regime gate.
        assert_eq!(configs.len(), 2);
        let neat = configs
            .iter()
            .find(|c| c.name == "neat")
            .expect("neat config present");
        assert_eq!(neat.model_type, ModelType::Neat);
    }

    #[test]
    fn derive_labels_uses_meta_label_max_hold_bars_when_model_horizon_is_zero() {
        let mut orchestrator = orchestrator_with_models(&["lightgbm"]);
        orchestrator.settings.models.label_use_triple_barrier = false;
        orchestrator.settings.models.label_horizon_bars = 0;
        orchestrator.settings.risk.meta_label_max_hold_bars = 3;
        orchestrator.settings.risk.triple_barrier_max_bars = 1;
        orchestrator.settings.risk.vol_horizon_bars = 1;
        orchestrator.settings.risk.meta_label_min_dist = 0.2;

        let ohlcv = Ohlcv {
            timestamp: Some(vec![0, 1, 2, 3]),
            open: vec![100.0, 100.1, 100.2, 100.3],
            high: vec![100.0, 100.1, 100.2, 100.3],
            low: vec![100.0, 100.1, 100.2, 100.3],
            close: vec![100.0, 100.1, 100.2, 100.3],
            volume: None,
        };

        let labels = orchestrator
            .derive_labels_test_oracle(&ohlcv, "EURUSD", 0.0001)
            .expect("aligned OHLCV should derive labels");
        assert_eq!(labels[0], 1);
    }

    #[test]
    fn derive_labels_rejects_an_empty_symbol_identity_before_geometry_work() {
        let orchestrator = orchestrator_with_models(&["lightgbm"]);
        let ohlcv = Ohlcv {
            timestamp: Some(Vec::new()),
            open: Vec::new(),
            high: Vec::new(),
            low: Vec::new(),
            close: Vec::new(),
            volume: None,
        };

        let error = orchestrator
            .derive_labels_test_oracle(&ohlcv, " ", 0.0001)
            .expect_err("empty symbol identity must fail closed");
        assert!(error.to_string().contains("bounded symbol identity"));
    }

    /// A move that clears the target but not the target plus the spread.
    ///
    /// The barriers used to sit at `close +/- distance` and charge nothing, so
    /// this rose exactly to the target and was labelled a win — a win the gene
    /// side, which charges 2.0 pips on every trade, would have booked as a loss.
    /// Training on that label is what taught the ensemble to answer a question
    /// nobody asks.
    ///
    /// Sizes are explicit rather than derived so the arithmetic is checkable:
    /// EURUSD pip is 0.0001, spread 1.5 + two 0.5-pip slippage fills = 2.5
    /// pips = 0.00025 in price, and the target sits 0.0010 above entry. The
    /// series peaks at +0.0011 — past the bare target, short of target-plus-cost.
    #[test]
    fn a_move_that_does_not_clear_the_spread_is_not_a_win() {
        let mut orchestrator = orchestrator_with_models(&["lightgbm"]);
        orchestrator.settings.models.label_use_triple_barrier = true;
        orchestrator.settings.models.label_horizon_bars = 4;
        orchestrator.settings.risk.meta_label_min_dist = 0.0010;
        orchestrator.settings.risk.meta_label_fixed_sl = 0.0010;
        orchestrator.settings.risk.meta_label_fixed_tp = 0.0010;
        orchestrator.settings.risk.backtest_spread_pips = 1.5;
        orchestrator.settings.risk.slippage_pips = 0.5;

        let entry = 1.1000;
        // Rises to +0.0011 and stays there: past the bare target, short of the
        // target once the round trip is paid.
        let close = vec![
            entry,
            entry + 0.0005,
            entry + 0.0011,
            entry + 0.0011,
            entry + 0.0011,
        ];
        let ohlcv = Ohlcv {
            timestamp: Some(vec![0, 1, 2, 3, 4]),
            open: close.clone(),
            high: close.clone(),
            low: close.clone(),
            close: close.clone(),
            volume: None,
        };

        let labels = orchestrator
            .derive_labels_test_oracle(&ohlcv, "EURUSD", 0.0001)
            .expect("aligned OHLCV should derive labels");
        assert_ne!(
            labels[0], 1,
            "a move of 11 pips against a 10-pip target and a 2.5-pip round trip is not a             win — the label must charge both slippage fills"
        );

        // Guard the guard: with the cost removed the same series IS a win, so
        // this fixture genuinely exercises the shift rather than failing for
        // some unrelated reason.
        orchestrator.settings.risk.backtest_spread_pips = 0.0;
        orchestrator.settings.risk.slippage_pips = 0.0;
        let costless = orchestrator
            .derive_labels_test_oracle(&ohlcv, "EURUSD", 0.0001)
            .expect("aligned OHLCV should derive labels");
        assert_eq!(
            costless[0], 1,
            "without a spread this move does clear the target — if it does not, the              fixture is wrong and the assertion above proves nothing"
        );
    }

    /// 8 192 real EURUSD M15 bars, 2024-07-01 → 2024-10-28 UTC, exported from
    /// the operator's parquet store (corrupt zero-price bars filtered at
    /// export). The slice was chosen for near-zero drift (+43 pips over four
    /// months) so the class prior measures the GEOMETRY, not the period's
    /// trend.
    fn real_eurusd_m15_fixture() -> Ohlcv {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/eurusd_m15_real.csv"
        );
        let raw = std::fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("real-bar fixture {path} must be readable: {error}"));
        let mut timestamp = Vec::new();
        let mut open = Vec::new();
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut close = Vec::new();
        for (line_no, line) in raw.lines().enumerate().skip(1) {
            let mut fields = line.split(',');
            let mut next = |name: &str| {
                fields
                    .next()
                    .unwrap_or_else(|| panic!("fixture line {line_no} is missing {name}"))
            };
            timestamp.push(
                next("timestamp_ms")
                    .parse::<i64>()
                    .unwrap_or_else(|e| panic!("fixture line {line_no} timestamp: {e}")),
            );
            for (name, series) in [
                ("open", &mut open),
                ("high", &mut high),
                ("low", &mut low),
                ("close", &mut close),
            ] {
                series.push(
                    next(name)
                        .parse::<f64>()
                        .unwrap_or_else(|e| panic!("fixture line {line_no} {name}: {e}")),
                );
            }
        }
        assert_eq!(close.len(), 8192, "fixture must carry all 8 192 bars");
        Ohlcv {
            timestamp: Some(timestamp),
            open,
            high,
            low,
            close,
            volume: None,
        }
    }

    /// THE class-balance guard. The default label geometry must produce a
    /// coin-flip class prior on real bars.
    ///
    /// The asymmetric bracket (40-pip target vs 20-pip stop, rr floored at
    /// 2.0) MANUFACTURED a ~66/34 prior on M15 — the floors bind on 88.4 % of
    /// bars, so the nearer stop simply gets hit first more often. 14 models
    /// recorded validation accuracy bit-identical to that prior: constant
    /// predictors faithfully answering a rigged question. Symmetric geometry
    /// on these same bars measures ~0.49 — a coin flip, which is what an
    /// unbiased label looks like on noise.
    ///
    /// If this fails after a label change, the geometry has drifted back to
    /// manufacturing a directional prior. Do NOT widen the band — fix the
    /// geometry.
    #[test]
    fn default_label_geometry_yields_a_coin_flip_class_prior_on_real_bars() {
        let ohlcv = real_eurusd_m15_fixture();
        let share_of_longs = |orchestrator: &TrainingOrchestrator| {
            let labels = orchestrator
                .derive_labels_test_oracle(&ohlcv, "EURUSD", 0.0001)
                .expect("real fixture bars must derive labels");
            let longs = labels.iter().filter(|&&label| label == 1).count();
            let shorts = labels.iter().filter(|&&label| label == -1).count();
            assert!(
                longs + shorts > 4000,
                "fixture must produce a meaningful number of directional labels, got {}",
                longs + shorts
            );
            longs as f64 / (longs + shorts) as f64
        };

        // DEFAULT config — nothing set, exactly what a fresh operator gets.
        let orchestrator = orchestrator_with_models(&["lightgbm"]);
        assert_eq!(
            orchestrator.settings.models.label_geometry, "symmetric",
            "the DEFAULT geometry must be symmetric"
        );
        let symmetric_share = share_of_longs(&orchestrator);
        assert!(
            (0.45..=0.55).contains(&symmetric_share),
            "default (symmetric) geometry must yield a coin-flip class prior on real bars; \
             measured +1 share {symmetric_share:.4} outside 50±5 % — the label is \
             manufacturing a directional prior again"
        );

        // Guard the guard: the legacy asymmetric geometry on the SAME bars
        // must show the manufactured skew, proving this fixture genuinely
        // discriminates geometry rather than passing vacuously.
        let mut legacy = orchestrator_with_models(&["lightgbm"]);
        legacy.settings.models.label_geometry = "asymmetric".to_string();
        let asymmetric_share = share_of_longs(&legacy);
        assert!(
            asymmetric_share < 0.45,
            "legacy asymmetric geometry should manufacture a skewed prior on these bars \
             (historically ~0.30-0.37); measured {asymmetric_share:.4} — if this is now \
             balanced the fixture no longer exercises the geometry switch"
        );
    }

    /// Identical bars, opposite verdicts — the geometry switch, isolated.
    ///
    /// The series dips 19 pips (through the legacy 18-pip cost-shifted stop,
    /// NOT through the symmetric 22-pip lower barrier) then rallies 25 pips
    /// (through the symmetric 22-pip upper barrier, far short of the legacy
    /// 42-pip target). Asymmetric labels it -1 — "the long got stopped";
    /// symmetric labels it +1 — the up-move won the fair race.
    #[test]
    fn symmetric_and_asymmetric_geometry_disagree_by_construction() {
        let entry = 1.1000;
        let ohlcv = Ohlcv {
            timestamp: Some(vec![0, 1, 2, 3]),
            open: vec![entry, entry, 1.0985, 1.1024],
            high: vec![entry, entry, 1.1025, 1.1024],
            low: vec![entry, 1.0981, 1.0985, 1.1024],
            close: vec![entry, 1.0985, 1.1024, 1.1024],
            volume: None,
        };

        let orchestrator = orchestrator_with_models(&["lightgbm"]);
        let symmetric = orchestrator
            .derive_labels_test_oracle(&ohlcv, "EURUSD", 0.0001)
            .expect("labels derive");
        assert_eq!(
            symmetric[0], 1,
            "symmetric race: +25 pips clears the 20-pip distance plus 2-pip cost, and the \
             19-pip dip never reaches the 22-pip lower barrier"
        );

        let mut legacy = orchestrator_with_models(&["lightgbm"]);
        legacy.settings.models.label_geometry = "asymmetric".to_string();
        let asymmetric = legacy
            .derive_labels_test_oracle(&ohlcv, "EURUSD", 0.0001)
            .expect("labels derive");
        assert_eq!(
            asymmetric[0], -1,
            "legacy bracket: the 19-pip dip goes through the cost-shifted 18-pip stop before \
             the rally, and the 42-pip target is never reached"
        );
    }

    #[test]
    fn unknown_label_geometry_fails_loudly() {
        let mut orchestrator = orchestrator_with_models(&["lightgbm"]);
        orchestrator.settings.models.label_geometry = "symetric".to_string();
        let ohlcv = Ohlcv {
            timestamp: Some(vec![0, 1]),
            open: vec![1.0, 1.0],
            high: vec![1.0, 1.0],
            low: vec![1.0, 1.0],
            close: vec![1.0, 1.0],
            volume: None,
        };
        let err = orchestrator
            .derive_labels_test_oracle(&ohlcv, "EURUSD", 0.0001)
            .expect_err("a typo in label_geometry must stop training, not pick a side silently");
        let message = err.to_string();
        assert!(
            message.contains("symetric") && message.contains("symmetric"),
            "error must quote the bad value and name the valid ones, got: {message}"
        );
    }

    #[test]
    fn preferred_burn_device_policy_rejects_unknown_gpu_token() {
        let mut orchestrator = orchestrator_with_models(&["mlp"]);
        orchestrator.settings.system.enable_gpu_preference = "gpu".to_string();
        orchestrator.settings.system.device = "mystery_gpu".to_string();

        assert_eq!(orchestrator.preferred_burn_device_policy(), "gpu");
    }

    #[test]
    fn full_nvidia_model_validation_rejects_a_model_without_a_compiled_gpu_route() {
        let orchestrator = orchestrator_with_models(&["online_hoeffding"]);
        let config = ModelConfig {
            name: "online_hoeffding".to_string(),
            model_type: ModelType::OnlineHoeffding,
            capability_family: ModelFamily::Adaptive,
            capability_state: CapabilityState::Verified,
            params: HashMap::from([("device".to_string(), "cpu".to_string())]),
        };

        let error = orchestrator
            .validate_nvidia_model_config_v1(&config)
            .expect_err("full NVIDIA admission must reject every model without a GPU route");
        let message = error.to_string();
        assert!(
            message.contains("online_hoeffding")
                && message.contains("no compiled NVIDIA CUDA implementation"),
            "full-GPU refusal must name the exact missing model and capability: {message}"
        );
    }

    #[cfg(feature = "statistical-gpu")]
    #[test]
    fn full_nvidia_model_validation_rejects_cpu_policy_even_when_gpu_code_is_compiled() {
        let orchestrator = orchestrator_with_models(&["logistic"]);
        let config = ModelConfig {
            name: "logistic".to_string(),
            model_type: ModelType::Logistic,
            capability_family: ModelFamily::Meta,
            capability_state: CapabilityState::Verified,
            params: HashMap::from([("device".to_string(), "cpu".to_string())]),
        };

        let error = orchestrator
            .validate_nvidia_model_config_v1(&config)
            .expect_err("compiled GPU code must not excuse an explicit CPU policy");
        assert!(
            error.to_string().contains("exact CUDA ordinal 0"),
            "CPU policy must fail as a device mismatch, not silently fall back: {error:#}"
        );
    }

    #[test]
    fn derive_labels_returns_error_on_ohlcv_mismatch() {
        let orchestrator = orchestrator_with_models(&["lightgbm"]);
        let ohlcv = Ohlcv {
            timestamp: None,
            open: vec![1.0, 2.0],
            high: vec![1.0],
            low: vec![1.0, 2.0],
            close: vec![1.0, 2.0],
            volume: None,
        };

        let err = orchestrator
            .derive_labels_test_oracle(&ohlcv, "EURUSD", 0.0001)
            .expect_err("misaligned OHLCV should fail");
        assert!(err.to_string().contains("aligned OHLCV series"));
    }

    #[test]
    fn production_label_derivation_refuses_before_formula_or_input_validation() {
        let orchestrator = orchestrator_with_models(&["lightgbm"]);
        let malformed = Ohlcv {
            timestamp: None,
            open: vec![1.0, 2.0],
            high: vec![1.0],
            low: vec![1.0, 2.0],
            close: vec![1.0, 2.0],
            volume: None,
        };

        let error = orchestrator
            .derive_labels(&malformed, "UNKNOWN-SYMBOL")
            .expect_err("production labels must require broker-gated evidence");
        assert!(
            error
                .to_string()
                .contains("BROKER_FINANCIAL_TRUTH_UNAVAILABLE_V1"),
            "formula/input work ran before the broker-truth boundary: {error:#}"
        );
    }

    #[test]
    fn training_runtime_profile_records_capability_metadata_and_effective_label_horizon() {
        let mut orchestrator = orchestrator_with_models(&["lightgbm"]);
        orchestrator.settings.models.label_horizon_bars = 0;
        orchestrator.settings.risk.meta_label_max_hold_bars = 3;
        orchestrator.settings.models.label_use_triple_barrier = false;

        let plan = orchestrator
            .create_dispatch_plan()
            .expect("dispatch plan should build");
        let configs = orchestrator
            .build_training_configs(&plan)
            .expect("training configs should build");
        // Pick lightgbm explicitly — the always-on hmm_regime sorts first.
        let config = configs
            .iter()
            .find(|c| c.name == "lightgbm")
            .expect("lightgbm config present");
        let payload = test_training_payload(
            ndarray::Array2::<f64>::zeros((4, 1)),
            vec!["feature_0".to_string()],
            vec![0, 0, 0, 0],
        );

        let geometry = resolve_label_geometry(&orchestrator.settings)
            .expect("default settings resolve a label geometry");
        let profile = training_runtime_profile(
            &orchestrator.settings,
            config,
            "EURUSD",
            "M1",
            &payload,
            Some(4),
            vec!["H1".to_string()],
            &geometry,
        );

        assert_eq!(
            profile.capability_family,
            crate::runtime::capabilities::ModelFamily::Tree
        );
        assert_eq!(profile.capability_state, CapabilityState::Verified);
        assert_eq!(profile.label_horizon_bars, 0);
        assert_eq!(profile.effective_label_horizon_bars, 3);
        assert_eq!(profile.meta_label_max_hold_bars, 3);
        assert!(!profile.label_use_triple_barrier);
        // The artifact names the labels it was trained on: default geometry is
        // symmetric — equal fixed floors, no rr push.
        assert_eq!(profile.label_geometry, "symmetric");
        assert_eq!(profile.label_fixed_stop, 0.0020);
        assert_eq!(profile.label_fixed_target, 0.0020);
        assert_eq!(profile.label_rr_floor, 1.0);
        assert!(profile.planned_cpu_threads.is_some());
        assert!(profile.planned_batch_size.is_some());
        assert!(profile.planned_memory_budget_gb.is_some());
    }

    #[test]
    fn every_legacy_rllib_request_fails_before_dispatch() {
        let mut explicit = orchestrator_with_models(&["dqn"]);
        explicit.settings.models.use_rllib_agent = true;

        let mut automatic = orchestrator_with_models(&["dqn"]);
        automatic.settings.models.auto_enable_rllib = true;

        let mut worker_request = orchestrator_with_models(&["dqn"]);
        worker_request.settings.models.rllib_num_workers = 2;

        for orchestrator in [explicit, automatic, worker_request] {
            let error = orchestrator
                .create_dispatch_plan()
                .expect_err("legacy RLlib configuration must fail before dispatch");
            assert_eq!(error.to_string(), LEGACY_RLLIB_MIGRATION_ERROR_V1);
        }
    }

    #[test]
    fn dqn_uses_explicit_native_rlkit_without_rllib_degradation() {
        let orchestrator = orchestrator_with_models(&["dqn"]);
        let params = orchestrator.default_model_params("dqn");
        assert_eq!(params.get("backend").map(String::as_str), Some("rlkit"));

        let config = ModelConfig {
            name: "dqn".to_string(),
            model_type: ModelType::Dqn,
            capability_family: crate::runtime::capabilities::ModelFamily::Rl,
            capability_state: CapabilityState::Implemented,
            params,
        };
        let payload = test_training_payload(
            ndarray::Array2::<f64>::zeros((4, 1)),
            vec!["feature_0".to_string()],
            vec![0, 0, 0, 0],
        );

        let geometry = resolve_label_geometry(&orchestrator.settings)
            .expect("default settings resolve a label geometry");
        let profile = training_runtime_profile(
            &orchestrator.settings,
            &config,
            "EURUSD",
            "M1",
            &payload,
            Some(4),
            Vec::new(),
            &geometry,
        );

        assert_eq!(profile.requested_backend.as_deref(), Some("rlkit"));
        assert!(!profile.rllib_requested);
        assert_eq!(profile.rllib_num_workers, 0);
        assert!(
            profile.notes.iter().all(|note| !note.contains("rllib")),
            "native rlkit must not carry an RLlib degradation note: {:?}",
            profile.notes
        );
    }

    #[test]
    fn persist_training_artifacts_writes_training_model_artifact_contract() {
        let artifact_dir = persist_sample_tree_training_artifacts("training_model_contract");

        let sidecar_path = artifact_dir.join("training_model_artifact.json");
        assert!(
            sidecar_path.is_file(),
            "training-model contract sidecar should be written"
        );
        let sidecar: neoethos_core::TrainingModelArtifact<TrainingRuntimeProfile> =
            serde_json::from_slice(
                &std::fs::read(&sidecar_path).expect("read training model contract sidecar"),
            )
            .expect("deserialize training model contract sidecar");

        assert_eq!(
            sidecar.contract_kind(),
            neoethos_core::ArtifactKind::TrainingModel
        );
        assert_eq!(
            sidecar.provenance.artifact_kind,
            neoethos_core::ArtifactKind::TrainingModel
        );
        assert_eq!(sidecar.payload.model_name, "lightgbm");
        assert_eq!(sidecar.payload.symbol, "EURUSD");
        assert_eq!(sidecar.payload.base_timeframe, "M15");
        assert_eq!(sidecar.payload.feature_count, 2);
        assert_eq!(sidecar.payload.dataset_rows, 4);
        assert_eq!(
            sidecar.provenance.backend_kind,
            sidecar.provenance.device_assignment.backend
        );
        assert!(
            !sidecar.provenance.training_config_hash.trim().is_empty(),
            "training config hash should be populated"
        );
        assert!(
            !sidecar.provenance.dataset_fingerprint.trim().is_empty(),
            "dataset fingerprint should be populated"
        );

        if artifact_dir.exists() {
            std::fs::remove_dir_all(&artifact_dir).expect("cleanup training contract dir");
        }
    }

    #[test]
    fn persist_training_artifacts_writes_model_runtime_artifact_contract() {
        let artifact_dir = persist_sample_tree_training_artifacts("model_runtime_contract");

        let sidecar_path = artifact_dir.join("model_runtime_artifact.json");
        assert!(
            sidecar_path.is_file(),
            "model-runtime contract sidecar should be written"
        );
        let sidecar: neoethos_core::ModelRuntimeArtifact<TrainingRuntimeProfile> =
            serde_json::from_slice(
                &std::fs::read(&sidecar_path).expect("read model runtime contract sidecar"),
            )
            .expect("deserialize model runtime contract sidecar");

        assert_eq!(
            sidecar.contract_kind(),
            neoethos_core::ArtifactKind::ModelRuntime
        );
        assert_eq!(
            sidecar.provenance.artifact_kind,
            neoethos_core::ArtifactKind::ModelRuntime
        );
        assert_eq!(sidecar.payload.model_name, "lightgbm");
        assert_eq!(sidecar.payload.symbol, "EURUSD");
        assert_eq!(sidecar.payload.base_timeframe, "M15");
        assert_eq!(sidecar.payload.feature_count, 2);
        assert_eq!(sidecar.payload.dataset_rows, 4);
        assert_eq!(
            sidecar.provenance.backend_kind,
            sidecar.provenance.device_assignment.backend
        );
        assert!(
            !sidecar.provenance.runtime_config_hash.trim().is_empty(),
            "runtime config hash should be populated"
        );

        if artifact_dir.exists() {
            std::fs::remove_dir_all(&artifact_dir).expect("cleanup model runtime contract dir");
        }
    }

    #[test]
    fn with_staged_training_artifact_dir_promotes_complete_directory() {
        let target_dir = unique_test_dir("staged_promote");
        let marker_name = "marker.txt";
        with_staged_training_artifact_dir(&target_dir, |staged_dir| {
            std::fs::write(staged_dir.join(marker_name), b"ok")
                .context("write staged training marker")?;
            Ok(())
        })
        .expect("staged training artifact directory should promote");

        assert!(target_dir.join(marker_name).is_file());
        // Pre-2026-05-26 this called a local `staged_training_artifact_dir`
        // helper. After GROUP-E consolidation (#223) the helper was inlined
        // into `write_dir_with_backup` (json.rs:230) which computes
        // `target.with_extension(temp_extension)`. Re-derive the same path
        // here so the post-condition still pins the implementation.
        assert!(
            !target_dir.with_extension("tmp_training_dispatch").exists(),
            "staged dir must be cleaned up after promote"
        );

        if target_dir.exists() {
            std::fs::remove_dir_all(&target_dir).expect("cleanup promoted target dir");
        }
    }

    #[test]
    fn with_staged_training_artifact_dir_cleans_up_failed_stage() {
        let target_dir = unique_test_dir("staged_cleanup");
        // See sibling test for why this is `with_extension(...)` rather than
        // a helper call.
        let staged_dir = target_dir.with_extension("tmp_training_dispatch");

        let err = with_staged_training_artifact_dir(&target_dir, |_staged_dir| {
            anyhow::bail!("synthetic staging failure")
        })
        .expect_err("failing staged write must bubble up");
        assert!(err.to_string().contains("synthetic staging failure"));
        assert!(
            !staged_dir.exists(),
            "failed staged dir should be cleaned up"
        );
        assert!(
            !target_dir.exists(),
            "target dir should not be created on failure"
        );
    }
}
