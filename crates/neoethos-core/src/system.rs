use crate::config::Settings;
use crate::contracts::{DeviceAssignment, RuntimeDegradedReason};
use neoethos_execution_budget::{
    BudgetCap, BudgetCapProvenance, CapacityDetection, CoordinationScope, ExecutionBudgetRequest,
    LogicalThreadCount, ResolutionError, ResolvedExecutionBudget, WorkerLimit,
    installed_process_budget, resolve_execution_budget,
};
use serde::{Deserialize, Serialize};
#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
use std::process::Command;
use sysinfo::{ProcessesToUpdate, System, get_current_pid};

// NO `use std::env` (2026-08-09). After `AutoTuner` was deleted this file reads
// and writes zero environment variables — the whole hardware decision now comes
// from the probe and `Settings`, and nothing else. Adding an `env::var` here
// re-opens the second resolution path this wave exists to close; add a field to
// `system.hardware` instead.

mod backends;
pub use backends::AcceleratorBackend;
use backends::{choose_primary_backend, normalize_accelerator_preference};

/// Current schema version of `hardware_profile.json`. Per D4
/// versioning policy: bumped only when fields are removed /
/// renamed / type-changed in a way `#[serde(default)]` can't
/// bridge. Adding new optional fields stays at v1.
pub const HARDWARE_PROFILE_SCHEMA_VERSION: crate::schema_version::SchemaVersion =
    crate::schema_version::SchemaVersion::new(1);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareProfile {
    /// On-disk schema version. Defaults to v1 (the pre-versioning
    /// shape) for files written by older builds.
    #[serde(default = "crate::schema_version::default_v1")]
    pub schema_version: crate::schema_version::SchemaVersion,
    /// Host inventory captured with this profile. It is not the process worker
    /// ceiling; affinity/cgroup-aware capacity comes from available_parallelism.
    pub cpu_cores: usize,
    pub total_ram_gb: f64,
    pub available_ram_gb: f64,
    pub gpu_names: Vec<String>,
    pub num_gpus: usize,
    pub gpu_mem_gb: Vec<f64>,
    #[serde(default)]
    pub accelerator_devices: Vec<AcceleratorDevice>,
    pub timestamp: String,
    pub platform_label: String,
}

impl crate::schema_version::HasSchemaVersion for HardwareProfile {
    const CURRENT: crate::schema_version::SchemaVersion = HARDWARE_PROFILE_SCHEMA_VERSION;
    fn schema_version(&self) -> crate::schema_version::SchemaVersion {
        self.schema_version
    }
}

pub struct HardwareProbe {
    sys: System,
    #[allow(dead_code)] // consumed by backend-specific probe features
    runtime_overrides: HardwareRuntimeOverrides,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct HardwareRuntimeOverrides {
    pub training_precision: Option<TrainingPrecision>,
    pub cuda_precisions: Option<Vec<TrainingPrecision>>,
    pub rocm_precisions: Option<Vec<TrainingPrecision>>,
    pub wgpu_precisions: Option<Vec<TrainingPrecision>>,
    pub wgpu_device_names: Vec<String>,
}

impl HardwareRuntimeOverrides {
    // REMOVED 2026-08-03: `from_env()`. Zero callers — `from_settings` below is
    // what production installs (system.rs:311 and :651). It read six env vars
    // that therefore did nothing: NEOETHOS_BOT_CPU_BUDGET,
    // NEOETHOS_BOT_TRAIN_PRECISION (plus a FOREX_TRAIN_PRECISION alias),
    // NEOETHOS_BOT_CUDA_PRECISIONS, _ROCM_PRECISIONS, _WGPU_PRECISIONS and
    // _WGPU_DEVICES. Every one has a config field on `system.hardware`, so
    // nothing is lost — but anyone who set one and watched precision not change
    // was fighting a function with no caller.
    //
    // CPU width is no longer installed through this compatibility struct.
    // `ExecutionBudgetInputs` keeps the persistent, legacy, and parent caps
    // distinct and resolves them once before runtime initialization.

    /// Config-driven constructor. A `hardware_from_settings_default_matches_default`
    /// test guarantees a fresh `Settings` reproduces [`Self::default`].
    pub fn from_settings(s: &crate::config::Settings) -> Self {
        let c = &s.system.hardware;
        Self {
            training_precision: c.training_precision,
            cuda_precisions: c.cuda_precisions.clone(),
            rocm_precisions: c.rocm_precisions.clone(),
            wgpu_precisions: c.wgpu_precisions.clone(),
            wgpu_device_names: c.wgpu_device_names.clone(),
        }
    }

    pub fn precision_override(
        &self,
        backend: AcceleratorBackend,
    ) -> Option<Vec<TrainingPrecision>> {
        match backend {
            AcceleratorBackend::Cuda => self.cuda_precisions.clone(),
            AcceleratorBackend::Rocm => self.rocm_precisions.clone(),
            AcceleratorBackend::Wgpu
            | AcceleratorBackend::Vulkan
            | AcceleratorBackend::Metal
            | AcceleratorBackend::Dx12 => self.wgpu_precisions.clone(),
            AcceleratorBackend::Cpu => None,
        }
    }
}

/// Typed, non-mutating inputs to the single process CPU-capacity resolver.
///
/// `Settings` remains persistent operator intent. A parent assignment is a
/// separate ephemeral cap and is never written back into either the canonical
/// or legacy settings field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionBudgetInputs {
    request: ExecutionBudgetRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionBudgetInputError {
    key: &'static str,
}

impl std::fmt::Display for ExecutionBudgetInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "`{}` must be greater than zero", self.key)
    }
}

impl std::error::Error for ExecutionBudgetInputError {}

impl ExecutionBudgetInputs {
    pub fn from_settings_and_parent(
        settings: &Settings,
        parent_assignment: Option<usize>,
        coordination_scope: CoordinationScope,
    ) -> Result<Self, ExecutionBudgetInputError> {
        Self::from_settings_parent_and_detection(
            settings,
            parent_assignment,
            CapacityDetection::detect(),
            coordination_scope,
        )
    }

    /// Deterministic form for tests and callers that already performed the
    /// process-capacity preflight.
    pub fn from_settings_parent_and_detection(
        settings: &Settings,
        parent_assignment: Option<usize>,
        detection: CapacityDetection,
        coordination_scope: CoordinationScope,
    ) -> Result<Self, ExecutionBudgetInputError> {
        let persistent_limit = cap_from_value(
            "system.hardware.cpu_budget",
            settings.system.hardware.cpu_budget,
            BudgetCapProvenance::PersistentSetting,
        )?;
        let legacy_persistent_limit = cap_from_value(
            "models.backtest_runtime.rayon_threads",
            settings.models.backtest_runtime.rayon_threads,
            BudgetCapProvenance::LegacyPersistentSetting,
        )?;
        let parent_limit = cap_from_value(
            "--cpu-threads",
            parent_assignment,
            BudgetCapProvenance::ParentAssignment,
        )?;

        Ok(Self {
            request: ExecutionBudgetRequest {
                host_logical_threads: None,
                detection,
                persistent_limit,
                legacy_persistent_limit,
                parent_limit,
                coordination_scope,
            },
        })
    }

    pub fn with_host_logical_threads(
        mut self,
        host_logical_threads: usize,
    ) -> Result<Self, ExecutionBudgetInputError> {
        self.request.host_logical_threads =
            Some(LogicalThreadCount::new(host_logical_threads).map_err(|_| {
                ExecutionBudgetInputError {
                    key: "hardware_profile.cpu_cores",
                }
            })?);
        Ok(self)
    }

    pub fn request(&self) -> &ExecutionBudgetRequest {
        &self.request
    }

    pub fn resolve(self) -> Result<ResolvedExecutionBudget, ResolutionError> {
        resolve_execution_budget(self.request)
    }
}

fn cap_from_value(
    key: &'static str,
    value: Option<usize>,
    provenance: BudgetCapProvenance,
) -> Result<Option<BudgetCap>, ExecutionBudgetInputError> {
    value
        .map(|value| {
            WorkerLimit::new(value)
                .map(|limit| BudgetCap::new(limit, provenance))
                .map_err(|_| ExecutionBudgetInputError { key })
        })
        .transpose()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrainingPrecision {
    Fp32,
    Fp16,
    Bf16,
    Fp8,
    Bf4,
}

impl TrainingPrecision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
            Self::Bf4 => "bf4",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcceleratorDevice {
    /// Stable profile-local identifier. Backend runtimes should use
    /// `backend_index`, which follows their device-class-specific indexing.
    pub id: usize,
    pub name: String,
    pub backend: AcceleratorBackend,
    #[serde(default)]
    pub device_class: AcceleratorDeviceClass,
    #[serde(default)]
    pub backend_index: usize,
    pub memory_gb: f64,
    pub supported_precisions: Vec<TrainingPrecision>,
    pub compute_capability: Option<(i64, i64)>,
    pub source: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AcceleratorDeviceClass {
    DiscreteGpu,
    IntegratedGpu,
    VirtualGpu,
    #[default]
    Other,
}

impl AcceleratorDevice {
    pub fn device_string(&self) -> String {
        if let Some(selector) = self.cubecl_wgpu_selector() {
            format!("{}:{selector}", self.backend.as_str())
        } else {
            format!("{}:{}", self.backend.as_str(), self.backend_index)
        }
    }

    pub fn supports_precision(&self, precision: TrainingPrecision) -> bool {
        self.supported_precisions.contains(&precision)
    }

    pub fn cubecl_wgpu_selector(&self) -> Option<String> {
        if !self.backend.is_wgpu_family() {
            return None;
        }
        let kind = match self.device_class {
            AcceleratorDeviceClass::DiscreteGpu => "discrete",
            AcceleratorDeviceClass::IntegratedGpu => "integrated",
            AcceleratorDeviceClass::VirtualGpu => "virtual",
            AcceleratorDeviceClass::Other => "default",
        };
        Some(format!("{kind}:{}", self.backend_index))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkloadKind {
    DataIngestion,
    FeatureEngineering,
    StrategySearch,
    TreeTraining,
    DeepTraining,
    RlTraining,
    Inference,
    Ui,
}

/// CPU work a concrete job can usefully execute in parallel.
///
/// This is demand, not another capacity authority. The granted width is the
/// smaller of useful parallel units and the already-resolved process limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkloadDemand {
    pub workload: WorkloadKind,
    pub parallel_units: usize,
    pub owns_cpu_worker_pool: bool,
}

impl WorkloadDemand {
    pub const fn for_parallel_units(workload: WorkloadKind, parallel_units: usize) -> Self {
        Self {
            workload,
            parallel_units,
            owns_cpu_worker_pool: true,
        }
    }

    pub const fn lightweight_control(workload: WorkloadKind) -> Self {
        Self {
            workload,
            parallel_units: 0,
            owns_cpu_worker_pool: false,
        }
    }

    pub fn granted_workers(self, budget: &ResolvedExecutionBudget) -> usize {
        if !self.owns_cpu_worker_pool {
            return 0;
        }
        self.parallel_units.min(budget.effective_worker_limit.get())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuBudget {
    pub threads: usize,
}

impl CpuBudget {
    pub fn new(threads: usize) -> Self {
        // Zero is meaningful for orchestration-only lanes such as UI control:
        // they own no private CPU worker pool. CPU-heavy jobs are admitted via
        // the positive WorkerLimit in neoethos-execution-budget.
        Self { threads }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuBudget {
    pub device_ids: Vec<usize>,
    pub memory_budget_gb: f64,
}

impl GpuBudget {
    pub fn new(device_ids: Vec<usize>, memory_budget_gb: f64) -> Self {
        Self {
            device_ids,
            memory_budget_gb: memory_budget_gb.max(0.0),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrecisionPolicy {
    pub precision: TrainingPrecision,
    pub mixed_precision_allowed: bool,
}

impl PrecisionPolicy {
    pub fn from_precision(precision: TrainingPrecision) -> Self {
        Self {
            precision,
            mixed_precision_allowed: !matches!(precision, TrainingPrecision::Fp32),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedWorkloadAssignment {
    pub workload: WorkloadKind,
    pub hardware_profile_id: String,
    pub device_assignment: DeviceAssignment,
    pub cpu_budget: CpuBudget,
    pub gpu_budget: Option<GpuBudget>,
    pub precision_policy: PrecisionPolicy,
    pub batch_size: usize,
    pub runtime_degraded_reason: Option<RuntimeDegradedReason>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkloadExecutionPlan {
    pub workload: WorkloadKind,
    pub backend: AcceleratorBackend,
    pub device: String,
    pub device_ids: Vec<usize>,
    pub precision: TrainingPrecision,
    /// Useful parallel units declared before admission.
    pub requested_cpu_threads: usize,
    /// Granted workers, never above the installed process capacity.
    pub cpu_threads: usize,
    pub batch_size: usize,
    pub memory_budget_gb: f64,
    pub notes: Vec<String>,
}

impl WorkloadExecutionPlan {
    pub fn device_assignment(&self) -> DeviceAssignment {
        DeviceAssignment {
            backend: self.backend.backend_kind(),
            device: self.device.clone(),
            device_ids: self.device_ids.clone(),
        }
    }

    pub fn cpu_budget(&self) -> CpuBudget {
        CpuBudget::new(self.cpu_threads)
    }

    pub fn gpu_budget(&self) -> Option<GpuBudget> {
        self.backend
            .is_gpu()
            .then(|| GpuBudget::new(self.device_ids.clone(), self.memory_budget_gb))
    }

    pub fn precision_policy(&self) -> PrecisionPolicy {
        PrecisionPolicy::from_precision(self.precision)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CpuCapacityDiagnostics {
    /// Legacy host inventory only; never used as the process ceiling.
    pub host_logical_threads: Option<usize>,
    /// OS/cgroup/affinity-aware capacity visible to this process.
    pub effective_logical_threads: usize,
    pub reserved_logical_threads: usize,
    pub installed_worker_limit: usize,
    pub coordination_scope: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareExecutionPlan {
    pub profile: HardwareProfile,
    pub cpu_capacity: CpuCapacityDiagnostics,
    pub gpu_enabled: bool,
    pub primary_backend: AcceleratorBackend,
    pub preferred_precision: TrainingPrecision,
    pub workloads: Vec<WorkloadExecutionPlan>,
    pub warnings: Vec<String>,
}

impl HardwareExecutionPlan {
    pub fn from_settings_and_profile(settings: &Settings, profile: HardwareProfile) -> Self {
        Self::from_settings_profile_and_overrides(
            settings,
            profile,
            &HardwareRuntimeOverrides::from_settings(settings),
        )
    }

    pub fn from_settings_profile_and_overrides(
        settings: &Settings,
        profile: HardwareProfile,
        runtime_overrides: &HardwareRuntimeOverrides,
    ) -> Self {
        let resolved_budget = installed_process_budget()
            .map(|installed| installed.resolved().clone())
            .unwrap_or_else(|| {
                ExecutionBudgetInputs::from_settings_and_parent(
                    settings,
                    None,
                    CoordinationScope::ProcessLocal,
                )
                .and_then(|inputs| inputs.with_host_logical_threads(profile.cpu_cores.max(1)))
                .unwrap_or_else(|error| panic!("invalid CPU execution budget input: {error}"))
                .resolve()
                .unwrap_or_else(|error| panic!("invalid CPU execution budget request: {error}"))
            });
        Self::from_settings_profile_overrides_and_budget(
            settings,
            profile,
            runtime_overrides,
            &resolved_budget,
        )
    }

    /// Deterministic planner entry used when startup has already installed or
    /// tests have explicitly supplied the resolved process capacity.
    pub fn from_settings_profile_overrides_and_budget(
        settings: &Settings,
        profile: HardwareProfile,
        runtime_overrides: &HardwareRuntimeOverrides,
        resolved_budget: &ResolvedExecutionBudget,
    ) -> Self {
        let preference = normalize_accelerator_preference(&settings.system.enable_gpu_preference);
        let cuda_devices = profile.devices_for_backend(AcceleratorBackend::Cuda);
        let has_gpu = !profile.accelerator_devices.is_empty();
        let gpu_allowed = !matches!(preference.as_str(), "cpu" | "off");
        let gpu_forced = matches!(
            preference.as_str(),
            "gpu" | "cuda" | "rocm" | "wgpu" | "vulkan" | "metal" | "dx12"
        );
        let primary_backend = choose_primary_backend(&preference, &profile);
        let gpu_enabled = has_gpu && gpu_allowed && primary_backend.is_gpu();
        let backend_devices = profile.devices_for_planned_backend(primary_backend);
        let preferred_precision =
            choose_training_precision(&profile, primary_backend, runtime_overrides);
        let mut warnings = Vec::new();
        if gpu_forced && !has_gpu {
            warnings.push(
                "GPU was requested but no accelerator device was detected; using CPU plans."
                    .to_string(),
            );
        }
        if gpu_enabled && preference == "cuda" && cuda_devices.is_empty() {
            warnings.push(
                "CUDA was requested but no CUDA device was detected; CUDA-only search/RL/tree paths will use CPU fallback."
                    .to_string(),
            );
        }
        if primary_backend == AcceleratorBackend::Rocm {
            warnings.push(
                "ROCm deep planning applies to Burn/deep workloads; current search/RL native tensor paths still require an implemented ROCm runtime and therefore use CPU fallback."
                    .to_string(),
            );
        }

        let cpu_budget = resolved_budget.effective_worker_limit.get();
        let host_memory_budget_gb = profile.available_ram_gb.max(1.0);
        let device_ids: Vec<usize> = if gpu_enabled {
            backend_devices.iter().map(|device| device.id).collect()
        } else {
            Vec::new()
        };
        let primary_device = if gpu_enabled {
            backend_devices
                .first()
                .map(|device| device.device_string())
                .unwrap_or_else(|| "none".to_string())
        } else {
            "cpu".to_string()
        };
        let min_vram_gb = min_gpu_memory_gb(&profile);
        let train_batch_size = training_batch_size(gpu_enabled, min_vram_gb);
        let infer_batch_size = inference_batch_size(gpu_enabled, min_vram_gb);

        let search_gpu_requested = !matches!(
            settings
                .models
                .prop_search_device
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "cpu" | "off" | "false"
        );
        let tree_gpu_requested = !matches!(
            settings
                .models
                .tree_device_preference
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "cpu" | "off" | "false"
        );
        let search_gpu_enabled = gpu_enabled
            && search_gpu_requested
            && (primary_backend == AcceleratorBackend::Cuda || primary_backend.is_wgpu_family());
        let search_device = if !search_gpu_enabled {
            "cpu".to_string()
        } else if primary_backend == AcceleratorBackend::Cuda {
            "cuda:all".to_string()
        } else {
            backend_devices
                .first()
                .map(|device| device.device_string())
                .unwrap_or_else(|| "cpu".to_string())
        };
        let search_device_ids = if search_gpu_enabled {
            backend_devices.iter().map(|device| device.id).collect()
        } else {
            Vec::new()
        };
        let search_backend = if search_gpu_enabled {
            primary_backend
        } else {
            AcceleratorBackend::Cpu
        };
        let tree_gpu_enabled = gpu_enabled && tree_gpu_requested && !cuda_devices.is_empty();
        let tree_backend = if tree_gpu_enabled {
            AcceleratorBackend::Cuda
        } else {
            AcceleratorBackend::Cpu
        };
        let rl_gpu_enabled = gpu_enabled && !cuda_devices.is_empty();
        let rl_backend = if rl_gpu_enabled {
            AcceleratorBackend::Cuda
        } else {
            AcceleratorBackend::Cpu
        };

        let mut workloads = Vec::new();
        workloads.push(WorkloadExecutionPlan {
            workload: WorkloadKind::DataIngestion,
            backend: AcceleratorBackend::Cpu,
            device: "cpu".to_string(),
            device_ids: Vec::new(),
            precision: choose_training_precision(
                &profile,
                if gpu_enabled && search_gpu_requested && !cuda_devices.is_empty() {
                    AcceleratorBackend::Cuda
                } else {
                    AcceleratorBackend::Cpu
                },
                runtime_overrides,
            ),
            requested_cpu_threads: cpu_budget,
            cpu_threads: WorkloadDemand::for_parallel_units(
                WorkloadKind::DataIngestion,
                cpu_budget,
            )
            .granted_workers(resolved_budget),
            batch_size: 0,
            memory_budget_gb: host_memory_budget_gb * 0.20,
            notes: vec![
                "Vortex/cTrader I/O stays CPU-bound and isolated from UI/inference threads."
                    .to_string(),
            ],
        });
        workloads.push(WorkloadExecutionPlan {
            workload: WorkloadKind::FeatureEngineering,
            backend: AcceleratorBackend::Cpu,
            device: "cpu".to_string(),
            device_ids: Vec::new(),
            precision: TrainingPrecision::Fp32,
            requested_cpu_threads: cpu_budget,
            cpu_threads: cpu_budget,
            batch_size: 0,
            memory_budget_gb: host_memory_budget_gb * 0.35,
            notes: vec![
                "ICT/SMC remains feature engineering only; model decisions stay autonomous."
                    .to_string(),
            ],
        });
        workloads.push(WorkloadExecutionPlan {
            workload: WorkloadKind::StrategySearch,
            backend: search_backend,
            device: search_device,
            device_ids: search_device_ids,
            precision: TrainingPrecision::Fp32,
            requested_cpu_threads: cpu_budget,
            cpu_threads: cpu_budget,
            batch_size: if search_gpu_enabled {
                train_batch_size
            } else {
                0
            },
            memory_budget_gb: planned_memory_budget_gb(
                &profile,
                search_backend,
                0.45,
                0.80,
            ),
            notes: vec!["Search evaluation uses the compiled CubeCL CUDA or WGPU runtime for GA offspring generation, signal synthesis, and the stateful backtest loop; price-normalized backtest arithmetic remains FP32 for pip-safe parity. ROCm stays an explicit CPU fallback until its runtime path is implemented.".to_string()],
        });
        workloads.push(WorkloadExecutionPlan {
            workload: WorkloadKind::TreeTraining,
            backend: tree_backend,
            device: if tree_gpu_enabled {
                "cuda:0".to_string()
            } else {
                "cpu".to_string()
            },
            device_ids: if tree_gpu_enabled {
                vec![0]
            } else {
                Vec::new()
            },
            precision: TrainingPrecision::Fp32,
            requested_cpu_threads: cpu_budget,
            cpu_threads: cpu_budget,
            batch_size: if tree_gpu_enabled { train_batch_size } else { 64 },
            memory_budget_gb: planned_memory_budget_gb(&profile, tree_backend, 0.35, 0.70),
            notes: vec!["Tree GPU support depends on each native backend feature; fallback must stay explicit in metadata.".to_string()],
        });
        workloads.push(WorkloadExecutionPlan {
            workload: WorkloadKind::DeepTraining,
            backend: primary_backend,
            device: primary_device.clone(),
            device_ids: device_ids.clone(),
            precision: preferred_precision,
            requested_cpu_threads: cpu_budget,
            cpu_threads: cpu_budget,
            batch_size: train_batch_size,
            memory_budget_gb: planned_memory_budget_gb(&profile, primary_backend, 0.55, 0.80),
            notes: vec![format!(
                "Burn/deep training should use planner policy with effective precision {}.",
                preferred_precision.as_str()
            )],
        });
        workloads.push(WorkloadExecutionPlan {
            workload: WorkloadKind::RlTraining,
            backend: rl_backend,
            device: if rl_gpu_enabled {
                "cuda:0".to_string()
            } else {
                "cpu".to_string()
            },
            device_ids: if rl_gpu_enabled { vec![0] } else { Vec::new() },
            precision: TrainingPrecision::Fp32,
            requested_cpu_threads: cpu_budget,
            cpu_threads: cpu_budget,
            batch_size: if rl_gpu_enabled { train_batch_size } else { 64 },
            memory_budget_gb: planned_memory_budget_gb(&profile, rl_backend, 0.35, 0.70),
            notes: vec![
                "RL CUDA remains feature-gated; unavailable CUDA must degrade explicitly to CPU."
                    .to_string(),
            ],
        });
        workloads.push(WorkloadExecutionPlan {
            workload: WorkloadKind::Inference,
            backend: primary_backend,
            device: primary_device,
            device_ids,
            precision: preferred_precision,
            requested_cpu_threads: cpu_budget,
            cpu_threads: WorkloadDemand::for_parallel_units(WorkloadKind::Inference, cpu_budget)
                .granted_workers(resolved_budget),
            batch_size: infer_batch_size,
            memory_budget_gb: planned_memory_budget_gb(&profile, primary_backend, 0.20, 0.50),
            notes: vec![
                "Inference uses smaller reserved budget so live execution and UI stay responsive."
                    .to_string(),
            ],
        });
        workloads.push(WorkloadExecutionPlan {
            workload: WorkloadKind::Ui,
            backend: AcceleratorBackend::Cpu,
            device: "cpu".to_string(),
            device_ids: Vec::new(),
            precision: TrainingPrecision::Fp32,
            requested_cpu_threads: 0,
            cpu_threads: WorkloadDemand::lightweight_control(WorkloadKind::Ui)
                .granted_workers(resolved_budget),
            batch_size: 0,
            memory_budget_gb: host_memory_budget_gb * 0.05,
            notes: vec!["UI stays message-channel driven and owns no private CPU worker pool; any CPU-heavy UI request enters the shared admission broker.".to_string()],
        });

        // ── Name every configured value this plan is about to override ──────
        //
        // Until 2026-08-09 the ONLY code that logged `warnings` was
        // `AutoTuner::apply`, which had zero callers — so "GPU was requested but
        // no accelerator device was detected" had never been printed in a real
        // run. A downgrade nobody is told about is a failure wearing the costume
        // of a choice.
        //
        // Nothing below changes a value. Each entry names the field, the
        // operator's setting, the computed result and the reason, so a silent
        // substitution becomes an impossible one.
        let mut overrides: Vec<String> = Vec::new();
        if gpu_forced && !gpu_enabled {
            overrides.push(format!(
                "system.enable_gpu_preference = {preference:?} (yours) -> GPU DISABLED (computed): {}",
                if has_gpu {
                    "devices were detected but no GPU backend could be chosen for them"
                } else {
                    "no accelerator device was detected"
                }
            ));
        }
        if search_gpu_requested && !search_gpu_enabled {
            overrides.push(format!(
                "models.prop_search_device = {:?} (yours) -> strategy search planned on \"cpu\" (computed): {}",
                settings.models.prop_search_device.trim(),
                if !has_gpu {
                    "no accelerator device was detected"
                } else if !gpu_allowed {
                    "system.enable_gpu_preference forbids the GPU"
                } else if !primary_backend.is_gpu() {
                    "the chosen primary backend is CPU"
                } else {
                    "the primary backend has no search runtime — search needs CUDA or a wgpu-family backend"
                }
            ));
        }
        if tree_gpu_requested && !tree_gpu_enabled {
            overrides.push(format!(
                "models.tree_device_preference = {:?} (yours) -> tree training planned on \"cpu\" (computed): {}",
                settings.models.tree_device_preference.trim(),
                if !gpu_enabled {
                    "no usable accelerator"
                } else {
                    "tree GPU support requires a CUDA device and none was detected"
                }
            ));
        }
        if settings.models.train_batch_size != train_batch_size {
            overrides.push(format!(
                "models.train_batch_size = {} (yours) -> {} (computed from {} and a smallest-VRAM figure of {:.1} GB). \
                 The training orchestrator applies planned params LAST, so the configured value never reaches the model",
                settings.models.train_batch_size,
                train_batch_size,
                if gpu_enabled { "an accelerator being present" } else { "CPU-only execution" },
                min_vram_gb
            ));
        }

        let plan = Self {
            profile,
            cpu_capacity: CpuCapacityDiagnostics {
                host_logical_threads: resolved_budget
                    .host_logical_threads
                    .map(LogicalThreadCount::get),
                effective_logical_threads: resolved_budget.effective_logical_threads.get(),
                reserved_logical_threads: resolved_budget.reserved_logical_threads,
                installed_worker_limit: resolved_budget.effective_worker_limit.get(),
                coordination_scope: format!("{:?}", resolved_budget.coordination_scope),
            },
            gpu_enabled,
            primary_backend,
            preferred_precision,
            workloads,
            warnings,
        };
        plan.announce(&overrides);
        plan
    }

    /// Say out loud what the machine decided: the probe, every workload
    /// assignment, every planner warning, and every configured value the plan
    /// overrides.
    ///
    /// Announced once per DISTINCT outcome. The plan is rebuilt for each
    /// training run, so an unchanged machine stays quiet; a plan that differs
    /// from the last one announced always prints, so a machine that changes
    /// underneath you is never silent.
    fn announce(&self, overrides: &[String]) {
        let fingerprint = format!(
            "{}|{}|{}|{}|{}|{}",
            self.profile.stable_id(),
            self.gpu_enabled,
            self.primary_backend.as_str(),
            self.preferred_precision.as_str(),
            self.workloads
                .iter()
                .map(|plan| format!(
                    "{:?}:{}:{}:{}:{}",
                    plan.workload,
                    plan.device,
                    plan.batch_size,
                    plan.requested_cpu_threads,
                    plan.cpu_threads
                ))
                .collect::<Vec<_>>()
                .join(","),
            overrides.join(";"),
        );

        static LAST_ANNOUNCED: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
        {
            let mut last = LAST_ANNOUNCED
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if last.as_deref() == Some(fingerprint.as_str()) {
                return;
            }
            *last = Some(fingerprint);
        }

        tracing::info!(
            target: "neoethos_core::system",
            host_logical_threads = self.profile.cpu_cores as u64,
            effective_logical_threads = self.cpu_capacity.effective_logical_threads as u64,
            reserved_logical_threads = self.cpu_capacity.reserved_logical_threads as u64,
            installed_worker_limit = self.cpu_capacity.installed_worker_limit as u64,
            coordination_scope = %self.cpu_capacity.coordination_scope,
            available_ram_gb = self.profile.available_ram_gb,
            detected_gpus = self.profile.num_gpus as u64,
            backend = self.primary_backend.as_str(),
            precision = self.preferred_precision.as_str(),
            gpu_enabled = self.gpu_enabled,
            "hardware execution plan resolved from the probe — these values are COMPUTED, not configured"
        );
        for plan in &self.workloads {
            tracing::info!(
                target: "neoethos_core::system",
                workload = ?plan.workload,
                backend = plan.backend.as_str(),
                device = %plan.device,
                requested_cpu_workers = plan.requested_cpu_threads as u64,
                granted_cpu_workers = plan.cpu_threads as u64,
                batch_size = plan.batch_size as u64,
                memory_budget_gb = plan.memory_budget_gb,
                "hardware plan workload assignment"
            );
        }
        for warning in &self.warnings {
            tracing::warn!(target: "neoethos_core::system", "hardware planner: {warning}");
        }
        for line in overrides {
            tracing::warn!(
                target: "neoethos_core::system",
                "hardware plan OVERRIDES a value you set: {line}"
            );
        }
    }

    pub fn workload(&self, kind: WorkloadKind) -> Option<&WorkloadExecutionPlan> {
        self.workloads.iter().find(|plan| plan.workload == kind)
    }

    pub fn profile_id(&self) -> String {
        self.profile.stable_id()
    }

    pub fn workload_assignment(&self, kind: WorkloadKind) -> Option<ResolvedWorkloadAssignment> {
        let hardware_profile_id = self.profile_id();
        self.workload(kind)
            .map(|plan| plan.resolved_assignment(hardware_profile_id))
    }

    pub fn workload_assignments(&self) -> Vec<ResolvedWorkloadAssignment> {
        let hardware_profile_id = self.profile_id();
        self.workloads
            .iter()
            .map(|plan| plan.resolved_assignment(hardware_profile_id.clone()))
            .collect()
    }
}

impl WorkloadExecutionPlan {
    pub fn resolved_assignment(
        &self,
        hardware_profile_id: impl Into<String>,
    ) -> ResolvedWorkloadAssignment {
        ResolvedWorkloadAssignment {
            workload: self.workload,
            hardware_profile_id: hardware_profile_id.into(),
            device_assignment: self.device_assignment(),
            cpu_budget: self.cpu_budget(),
            gpu_budget: self.gpu_budget(),
            precision_policy: self.precision_policy(),
            batch_size: self.batch_size,
            runtime_degraded_reason: self.runtime_degraded_reason(),
            notes: self.notes.clone(),
        }
    }

    fn runtime_degraded_reason(&self) -> Option<RuntimeDegradedReason> {
        if self.backend.is_gpu() {
            return None;
        }
        let requested_gpu_fallback = self.notes.iter().any(|note| {
            let note = note.to_ascii_lowercase();
            note.contains("fallback") || note.contains("degrade") || note.contains("unavailable")
        });
        requested_gpu_fallback.then(|| {
            RuntimeDegradedReason::new(
                "gpu_path_unavailable",
                "Scheduler resolved this workload to CPU while notes indicate a GPU path is unavailable or falling back.",
            )
        })
    }
}

impl Default for HardwareProbe {
    fn default() -> Self {
        Self::new()
    }
}

static HARDWARE_RUNTIME_OVERRIDES: std::sync::OnceLock<HardwareRuntimeOverrides> =
    std::sync::OnceLock::new();

/// Install the process-wide hardware runtime overrides from `Settings` (call
/// once at startup, before any `HardwareProbe::new`). The first install wins.
pub fn install_hardware_runtime_overrides_from_settings(s: &crate::config::Settings) {
    let _ = HARDWARE_RUNTIME_OVERRIDES.set(HardwareRuntimeOverrides::from_settings(s));
}

/// Current hardware runtime overrides (defaults if never installed — e.g. in
/// unit tests — preserving the historical env-absent behaviour).
pub fn current_hardware_runtime_overrides() -> &'static HardwareRuntimeOverrides {
    HARDWARE_RUNTIME_OVERRIDES.get_or_init(HardwareRuntimeOverrides::default)
}

/// Clamp host RAM figures to this process's cgroup limit when one exists.
///
/// `System::total_memory()` and `available_memory()` report the HOST's RAM.
/// Inside a container they are not what this process may actually use — and
/// every rented vast.ai box is a Docker container, so on exactly the hardware
/// the NEVER-OOM invariant was written for ("peak memory is a function of the
/// AVAILABLE hardware, never of user parameters") the numbers feeding it were
/// the wrong machine's.
///
/// `cgroup_limits()` returns `None` off Linux and on an unconstrained host, and
/// a limit is only applied when it is non-zero and actually smaller than the
/// host figure, so this is a no-op everywhere it should be.
///
/// One function, because the decision must be made in one place: two readers of
/// the same setting drifting apart is how `apply_mode_overrides` and the search
/// runtime overrides each ended up meaning something different from what they
/// claimed.
/// The whole decision, in a form a test can reach. A cgroup limit of 0 means
/// "unset", and a limit at or above the host figure means "unconstrained" —
/// both must yield the host value untouched, or an ordinary machine starts
/// sizing itself against a phantom limit.
fn tighter_of(host: u64, limit: u64) -> u64 {
    if limit > 0 && limit < host {
        limit
    } else {
        host
    }
}

fn clamp_memory_figures_to_reported_cgroup(
    host_total: u64,
    host_available: u64,
    limit_total: u64,
    limit_available: u64,
) -> (u64, u64) {
    (
        tighter_of(host_total, limit_total),
        host_available.min(limit_available),
    )
}

fn preferred_cgroup_memory_limits(
    process: Option<(u64, u64)>,
    root: Option<(u64, u64)>,
) -> Option<(u64, u64)> {
    process.or(root)
}

fn current_process_cgroup_memory_limits(sys: &System) -> Option<(u64, u64)> {
    let process = get_current_pid()
        .ok()
        .and_then(|pid| sys.process(pid))
        .and_then(|process| process.cgroup_limits())
        .map(|limits| (limits.total_memory, limits.free_memory));
    let root = sys
        .cgroup_limits()
        .map(|limits| (limits.total_memory, limits.free_memory));
    preferred_cgroup_memory_limits(process, root)
}

fn clamp_to_cgroup(sys: &System, host_total: u64, host_available: u64) -> (u64, u64) {
    let Some((limit_total, limit_available)) = current_process_cgroup_memory_limits(sys) else {
        return (host_total, host_available);
    };

    let (total, available) = clamp_memory_figures_to_reported_cgroup(
        host_total,
        host_available,
        limit_total,
        limit_available,
    );

    if total < host_total || available < host_available {
        static ANNOUNCED: std::sync::Once = std::sync::Once::new();
        ANNOUNCED.call_once(|| {
            tracing::info!(
                target: "neoethos_core::system",
                host_total_mb = host_total / 1024 / 1024,
                cgroup_total_mb = total / 1024 / 1024,
                host_available_mb = host_available / 1024 / 1024,
                cgroup_available_mb = available / 1024 / 1024,
                "running under a cgroup memory limit; sizing against the container, not the host"
            );
        });
    }

    (total, available)
}

/// Currently-available RAM in bytes (a fresh point-in-time probe), honouring a
/// container memory limit when one is in force.
///
/// Cheap enough to call before each feature-cube build so the builder can
/// decide RAM-resident vs disk-mmap assembly based on the machine's actual
/// free memory at that moment. Returns 0 if the probe fails, which callers
/// treat as "unknown → take the safe (disk) path".
pub fn available_memory_bytes() -> u64 {
    let mut sys = System::new();
    sys.refresh_memory();
    if let Ok(pid) = get_current_pid() {
        sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
    }
    clamp_to_cgroup(&sys, sys.total_memory(), sys.available_memory()).1
}

/// Total RAM in bytes available to this process. Pairs with
/// [`available_memory_bytes`] so callers (and the UI resource strip) can show
/// a "X of Y GB free" readout — and inside a container both report the
/// container's budget rather than the host's.
pub fn total_memory_bytes() -> u64 {
    let mut sys = System::new();
    sys.refresh_memory();
    if let Ok(pid) = get_current_pid() {
        sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
    }
    clamp_to_cgroup(&sys, sys.total_memory(), sys.available_memory()).0
}

impl HardwareProbe {
    pub fn new() -> Self {
        Self::with_runtime_overrides(current_hardware_runtime_overrides().clone())
    }

    pub fn with_runtime_overrides(runtime_overrides: HardwareRuntimeOverrides) -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();
        Self {
            sys,
            runtime_overrides,
        }
    }

    pub fn detect(&mut self) -> HardwareProfile {
        self.sys.refresh_all();

        let cpu_cores = self.sys.cpus().len().max(1);
        // Same clamp as the free functions above — the probe that drives every
        // sizing decision must not see a different machine from them.
        let (total_bytes, available_bytes) = clamp_to_cgroup(
            &self.sys,
            self.sys.total_memory(),
            self.sys.available_memory(),
        );
        let total_ram_gb = total_bytes as f64 / 1024.0 / 1024.0 / 1024.0;
        let available_ram_gb = available_bytes as f64 / 1024.0 / 1024.0 / 1024.0;

        let accelerator_devices = self.detect_accelerator_devices();
        let gpu_names = accelerator_devices
            .iter()
            .map(|device| device.name.clone())
            .collect::<Vec<_>>();
        let gpu_mem_gb = accelerator_devices
            .iter()
            .map(|device| device.memory_gb)
            .collect::<Vec<_>>();
        let num_gpus = gpu_names.len();

        let platform_label = format!(
            "{} {}",
            System::name().unwrap_or_default(),
            System::os_version().unwrap_or_default()
        );

        HardwareProfile {
            schema_version: HARDWARE_PROFILE_SCHEMA_VERSION,
            cpu_cores,
            total_ram_gb,
            available_ram_gb,
            gpu_names,
            num_gpus,
            gpu_mem_gb,
            accelerator_devices,
            timestamp: chrono::Utc::now().to_rfc3339(),
            platform_label,
        }
    }

    fn detect_accelerator_devices(&self) -> Vec<AcceleratorDevice> {
        #[allow(unused_mut)] // CPU-only builds compile every backend block out.
        let mut devices = Vec::new();
        #[cfg(feature = "gpu-cuda")]
        devices.extend(self.detect_nvidia_accelerators());
        #[cfg(feature = "gpu-rocm")]
        devices.extend(self.detect_rocm_accelerators(devices.len()));
        #[cfg(feature = "gpu-wgpu")]
        {
            let detected = self.detect_wgpu_accelerators();
            if detected.is_empty() {
                devices.extend(self.detect_wgpu_hint_accelerators(devices.len()));
            } else {
                devices.extend(detected);
            }
        }
        devices
    }

    #[cfg(feature = "gpu-wgpu")]
    fn detect_wgpu_accelerators(&self) -> Vec<AcceleratorDevice> {
        let Some(infos) = probe_wgpu_adapter_infos() else {
            return Vec::new();
        };
        let precision_override = self
            .runtime_overrides
            .precision_override(AcceleratorBackend::Wgpu);
        normalize_wgpu_adapter_infos(&infos, precision_override.as_deref())
    }

    #[cfg(feature = "gpu-cuda")]
    fn detect_nvidia_accelerators(&self) -> Vec<AcceleratorDevice> {
        let (names, mems) = self.detect_gpus_nvidia_smi();
        let compute_caps = self.detect_nvidia_compute_caps();
        names
            .into_iter()
            .enumerate()
            .map(|(idx, name)| {
                let compute_capability = compute_caps.get(idx).copied().flatten();
                let mut supported_precisions =
                    vec![TrainingPrecision::Fp32, TrainingPrecision::Fp16];
                if compute_capability
                    .map(|(major, _minor)| major >= 8)
                    .unwrap_or(false)
                {
                    supported_precisions.push(TrainingPrecision::Bf16);
                }
                if compute_capability
                    .map(|(major, minor)| major > 8 || (major == 8 && minor >= 9))
                    .unwrap_or(false)
                {
                    supported_precisions.push(TrainingPrecision::Fp8);
                }
                AcceleratorDevice {
                    id: idx,
                    name,
                    backend: AcceleratorBackend::Cuda,
                    device_class: AcceleratorDeviceClass::DiscreteGpu,
                    backend_index: idx,
                    memory_gb: mems.get(idx).copied().unwrap_or(0.0),
                    supported_precisions,
                    compute_capability,
                    source: "nvidia-smi".to_string(),
                }
            })
            .collect()
    }

    #[cfg(feature = "gpu-cuda")]
    fn detect_gpus_nvidia_smi(&self) -> (Vec<String>, Vec<f64>) {
        let mut names = Vec::new();
        let mut mems = Vec::new();

        let smi_candidates = if cfg!(target_os = "windows") {
            vec![
                "nvidia-smi",
                r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
                r"C:\Windows\System32\nvidia-smi.exe",
            ]
        } else {
            vec!["nvidia-smi"]
        };

        for cmd in smi_candidates {
            if let Ok(output) = Command::new(cmd)
                .args(["--query-gpu=name", "--format=csv,noheader"])
                .output()
                && output.status.success()
            {
                let out_str = String::from_utf8_lossy(&output.stdout);
                for line in out_str.lines() {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        names.push(trimmed.to_string());
                    }
                }
                if !names.is_empty() {
                    if let Ok(mem_out) = Command::new(cmd)
                        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
                        .output()
                    {
                        let mem_str = String::from_utf8_lossy(&mem_out.stdout);
                        for line in mem_str.lines() {
                            if let Ok(mb) = line.trim().parse::<f64>() {
                                mems.push(mb / 1024.0);
                            }
                        }
                    }
                    return (names, mems);
                }
            }
        }

        (vec![], vec![])
    }

    #[cfg(feature = "gpu-cuda")]
    fn detect_nvidia_compute_caps(&self) -> Vec<Option<(i64, i64)>> {
        let smi_candidates = if cfg!(target_os = "windows") {
            vec![
                "nvidia-smi",
                r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
                r"C:\Windows\System32\nvidia-smi.exe",
            ]
        } else {
            vec!["nvidia-smi"]
        };

        for cmd in smi_candidates {
            let mut command = Command::new(cmd);
            command.args(["--query-gpu=compute_cap", "--format=csv,noheader"]);
            // GROUP H remediation: 2s timeout per F-890.
            let Some(output) = run_hw_probe_with_timeout(command) else {
                continue;
            };
            if !output.status.success() {
                continue;
            }
            let stdout = String::from_utf8_lossy(&output.stdout);
            let caps = stdout
                .lines()
                .map(|line| parse_compute_capability(line.trim()))
                .collect::<Vec<_>>();
            if !caps.is_empty() {
                return caps;
            }
        }

        Vec::new()
    }

    #[cfg(feature = "gpu-rocm")]
    fn detect_rocm_accelerators(&self, id_offset: usize) -> Vec<AcceleratorDevice> {
        // GROUP H remediation: 2s timeout (operator directive 2026-05-25).
        let rocminfo = run_hw_probe_with_timeout(Command::new("rocminfo"));
        if let Some(output) = rocminfo
            && output.status.success()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let names = stdout
                .lines()
                .filter_map(|line| {
                    line.split_once("Marketing Name:")
                        .map(|(_, value)| value.trim().to_string())
                })
                .filter(|name| !name.is_empty())
                .collect::<Vec<_>>();
            if !names.is_empty() {
                return names
                    .into_iter()
                    .enumerate()
                    .map(|(idx, name)| AcceleratorDevice {
                        id: id_offset + idx,
                        name,
                        backend: AcceleratorBackend::Rocm,
                        device_class: AcceleratorDeviceClass::DiscreteGpu,
                        backend_index: idx,
                        memory_gb: 0.0,
                        supported_precisions: self
                            .runtime_overrides
                            .precision_override(AcceleratorBackend::Rocm)
                            .unwrap_or_else(|| {
                                vec![TrainingPrecision::Fp32, TrainingPrecision::Fp16]
                            }),
                        compute_capability: None,
                        source: "rocminfo".to_string(),
                    })
                    .collect();
            }
        }

        Vec::new()
    }

    #[cfg(feature = "gpu-wgpu")]
    fn detect_wgpu_hint_accelerators(&self, id_offset: usize) -> Vec<AcceleratorDevice> {
        self.runtime_overrides
            .wgpu_device_names
            .iter()
            .enumerate()
            .map(|(idx, name)| AcceleratorDevice {
                id: id_offset + idx,
                name: name.clone(),
                backend: AcceleratorBackend::Wgpu,
                device_class: AcceleratorDeviceClass::Other,
                backend_index: idx,
                memory_gb: 0.0,
                supported_precisions: self
                    .runtime_overrides
                    .precision_override(AcceleratorBackend::Wgpu)
                    .unwrap_or_else(|| vec![TrainingPrecision::Fp32]),
                compute_capability: None,
                source: "hardware_runtime_overrides.wgpu_device_names".to_string(),
            })
            .collect()
    }
}

#[cfg(feature = "gpu-wgpu")]
fn wgpu_probe_backends() -> wgpu::Backends {
    #[cfg(target_os = "macos")]
    {
        wgpu::Backends::METAL
    }
    #[cfg(target_family = "wasm")]
    {
        wgpu::Backends::BROWSER_WEBGPU
    }
    #[cfg(all(not(target_os = "macos"), not(target_family = "wasm")))]
    {
        // CubeCL's AutoGraphicsApi selects Vulkan on Windows and Linux. Probe
        // the same backend so reported ordinals match WgpuDevice selection.
        wgpu::Backends::VULKAN
    }
}

#[cfg(feature = "gpu-wgpu")]
fn probe_wgpu_adapter_infos() -> Option<Vec<wgpu::AdapterInfo>> {
    const WGPU_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let backends = wgpu_probe_backends();
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends,
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            });
            pollster::block_on(instance.enumerate_adapters(backends))
                .into_iter()
                .map(|adapter| adapter.get_info())
                .collect::<Vec<_>>()
        }))
        .ok();
        let _ = tx.send(result);
    });

    match rx.recv_timeout(WGPU_PROBE_TIMEOUT) {
        Ok(Some(infos)) => Some(infos),
        Ok(None) => {
            tracing::warn!(
                target: "neoethos_core::system",
                "WGPU adapter enumeration panicked; treating WGPU as unavailable"
            );
            None
        }
        Err(_) => {
            tracing::warn!(
                target: "neoethos_core::system",
                timeout_ms = WGPU_PROBE_TIMEOUT.as_millis() as u64,
                "WGPU adapter enumeration timed out; treating WGPU as unavailable"
            );
            None
        }
    }
}

#[cfg(feature = "gpu-wgpu")]
fn normalize_wgpu_adapter_infos(
    infos: &[wgpu::AdapterInfo],
    precision_override: Option<&[TrainingPrecision]>,
) -> Vec<AcceleratorDevice> {
    let mut discrete_index = 0usize;
    let mut integrated_index = 0usize;
    let mut virtual_index = 0usize;
    let mut other_index = 0usize;
    let mut devices = Vec::new();

    for info in infos {
        let backend = match info.backend {
            wgpu::Backend::Vulkan => AcceleratorBackend::Vulkan,
            wgpu::Backend::Metal => AcceleratorBackend::Metal,
            wgpu::Backend::Dx12 => AcceleratorBackend::Dx12,
            wgpu::Backend::Gl | wgpu::Backend::BrowserWebGpu => AcceleratorBackend::Wgpu,
            wgpu::Backend::Noop => continue,
        };
        let (device_class, backend_index) = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => {
                let index = discrete_index;
                discrete_index += 1;
                (AcceleratorDeviceClass::DiscreteGpu, index)
            }
            wgpu::DeviceType::IntegratedGpu => {
                let index = integrated_index;
                integrated_index += 1;
                (AcceleratorDeviceClass::IntegratedGpu, index)
            }
            wgpu::DeviceType::VirtualGpu => {
                let index = virtual_index;
                virtual_index += 1;
                (AcceleratorDeviceClass::VirtualGpu, index)
            }
            wgpu::DeviceType::Other => {
                let index = other_index;
                other_index += 1;
                (AcceleratorDeviceClass::Other, index)
            }
            wgpu::DeviceType::Cpu => continue,
        };
        let id = devices.len();
        devices.push(AcceleratorDevice {
            id,
            name: info.name.clone(),
            backend,
            device_class,
            backend_index,
            // wgpu does not expose reliable dedicated VRAM. In particular,
            // Windows reports shared-memory iGPU values inconsistently.
            memory_gb: 0.0,
            supported_precisions: precision_override
                .map(<[TrainingPrecision]>::to_vec)
                .unwrap_or_else(|| vec![TrainingPrecision::Fp32]),
            compute_capability: None,
            source: format!(
                "wgpu:{:?}:vendor={:#06x}:device={:#06x}:driver={}",
                info.backend, info.vendor, info.device, info.driver
            ),
        });
    }
    devices
}

/// GROUP H remediation (operator directive 2026-05-25, F-890):
/// run an external hardware-probe subprocess (`nvidia-smi`,
/// `rocminfo`, `rocm-smi`) with a hard 2-second timeout. On a healthy
/// host they answer in <100 ms; on a broken-NVML or zombie-rocm-smi
/// install they can otherwise hang the entire backend's startup
/// path. We spawn on a separate thread and accept that the
/// subprocess may continue running in the background — the main
/// process is unblocked which is what matters.
#[cfg(any(feature = "gpu-cuda", feature = "gpu-rocm"))]
fn run_hw_probe_with_timeout(mut cmd: Command) -> Option<std::process::Output> {
    const HW_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(cmd.output());
    });
    match rx.recv_timeout(HW_PROBE_TIMEOUT) {
        Ok(Ok(output)) => Some(output),
        Ok(Err(err)) => {
            tracing::debug!(
                target: "neoethos_core::system",
                error = %err,
                "hardware-probe subprocess failed to spawn"
            );
            None
        }
        Err(_) => {
            tracing::warn!(
                target: "neoethos_core::system",
                timeout_ms = HW_PROBE_TIMEOUT.as_millis() as u64,
                "hardware-probe subprocess timed out; treating as not-available"
            );
            None
        }
    }
}

impl HardwareProfile {
    pub fn stable_id(&self) -> String {
        let device_fingerprint = self
            .accelerator_devices
            .iter()
            .map(|device| {
                format!(
                    "{}:{}:{:?}:{}:{:.3}:{:?}:{:?}",
                    device.backend.as_str(),
                    device.name,
                    device.device_class,
                    device.backend_index,
                    device.memory_gb,
                    device.supported_precisions,
                    device.compute_capability
                )
            })
            .collect::<Vec<_>>()
            .join("|");
        stable_hex_hash(&format!(
            "cpu={};ram={:.3};platform={};devices={}",
            self.cpu_cores, self.total_ram_gb, self.platform_label, device_fingerprint
        ))
    }

    pub fn devices_for_backend(&self, backend: AcceleratorBackend) -> Vec<&AcceleratorDevice> {
        self.accelerator_devices
            .iter()
            .filter(|device| device.backend == backend)
            .collect()
    }

    pub fn wgpu_native_devices(&self) -> Vec<&AcceleratorDevice> {
        self.accelerator_devices
            .iter()
            .filter(|device| device.backend.is_wgpu_family())
            .collect()
    }

    pub fn wgpu_capable_devices(&self) -> Vec<&AcceleratorDevice> {
        self.accelerator_devices
            .iter()
            .filter(|device| device.backend.is_wgpu_family())
            .collect()
    }

    pub fn devices_for_planned_backend(
        &self,
        backend: AcceleratorBackend,
    ) -> Vec<&AcceleratorDevice> {
        if backend.is_wgpu_family() {
            self.wgpu_capable_devices()
        } else {
            self.devices_for_backend(backend)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// REMOVED 2026-08-09 (D2): `AutoTuner`, `AutoTuneHints` and
// `apply_thread_env_defaults` — ~145 lines, zero callers.
//
// WHY IT WENT rather than being wired:
//
//   1. It had never run. `grep -rn AutoTuner` returned exactly two lines in the
//      whole workspace, both declarations. No construction, no `.apply()`, not
//      even a test.
//
//   2. Its successor exists AND is called. `HardwareExecutionPlan` (above) is
//      built by the training orchestrator and computes the identical quantities
//      from the identical probe — `training_batch_size`, `inference_batch_size`,
//      `planned_memory_budget_gb`, the per-workload device — and hands them to
//      the consumer as PARAMS, applied last so no stale setting can survive.
//      `AutoTuner::apply` did the opposite: it wrote nine derived values BACK
//      INTO `Settings`, and `Settings::save` then serialises the whole struct.
//      That is the mechanism that pickles detector output into the operator's
//      YAML as if he had typed it. Wiring it would have re-frozen the same lie
//      on a different machine, and would additionally have overwritten
//      `system.device`, `models.prop_search_device` and
//      `models.tree_device_preference` — three values the operator sets on
//      purpose.
//
//   3. It let the hardware decide how hard to search. `hpo_trials = if gpu { 50 }
//      else { 20 }` is not a capacity decision: a different trial count returns a
//      different answer, not the same answer in a different footprint. The
//      capacity/preference line is that a knob may be derived only when a machine
//      with twice the RAM would want a different number AND ONLY because of the
//      hardware. `hpo_trials`, `prop_search_population`, `_generations`,
//      `_max_hours` and `_max_rows` all fail that test and stay settable.
//
// ⚠ THE OMP/MKL/OPENBLAS THREAD CLAMP WENT WITH IT, DELIBERATELY.
// `apply_thread_env_defaults` was the only code in the workspace that set
// `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS`, and its ONLY
// caller was `AutoTuner::apply`, which had none. So those variables were NOT
// being set in any run, ever. Re-homing the call would therefore not have
// preserved behaviour — it would have clamped BLAS/OpenMP for the FIRST TIME,
// inside the subsystem already identified as the thread-oversubscription
// bottleneck, in a change presented as a no-op. If a thread clamp is wanted it
// must land on its own, with its own before/after measurement, and be tracked as
// its own item. It is recorded as one; it is not silently smuggled in here.
//
// The surviving batch/memory derivation helpers are below. CPU capacity has no
// legacy helper: `ExecutionBudgetInputs` is the only resolver, including for
// retired-key diagnostics.
// ─────────────────────────────────────────────────────────────────────────────

fn stable_hex_hash(value: &str) -> String {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001B3;

    let mut hash = FNV_OFFSET;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{hash:016x}")
}

fn choose_training_precision(
    profile: &HardwareProfile,
    backend: AcceleratorBackend,
    runtime_overrides: &HardwareRuntimeOverrides,
) -> TrainingPrecision {
    let requested = runtime_overrides.training_precision;
    let devices = profile.devices_for_planned_backend(backend);
    let supported_by_all = |precision| {
        !devices.is_empty()
            && devices
                .iter()
                .all(|device| device.supports_precision(precision))
    };

    match requested {
        Some(TrainingPrecision::Bf16) if supported_by_all(TrainingPrecision::Bf16) => {
            TrainingPrecision::Bf16
        }
        Some(TrainingPrecision::Fp32) => TrainingPrecision::Fp32,
        Some(_) => TrainingPrecision::Fp32,
        None if supported_by_all(TrainingPrecision::Bf16) => TrainingPrecision::Bf16,
        None => TrainingPrecision::Fp32,
    }
}

// REMOVED 2026-08-03 with `HardwareRuntimeOverrides::from_env`, which was their
// only caller: `parse_env_usize`, `parse_env_precisions` and
// `parse_training_precision`. The compiler named all three the moment the dead
// parent went — dead code hides behind a dead caller, so a deletion should
// always be followed by re-reading the warnings rather than stopping at the
// first thing removed.
//
// Note for whoever migrates precision to config: `parse_training_precision` was
// DUPLICATED in neoethos-search/src/cubecl_eval.rs:1735, which is still live.
// Two independent parsers for one string vocabulary, in two crates, is the
// familiar shape — the copy that survives is now the only one, so any future
// vocabulary change has exactly one place to land.

#[cfg(feature = "gpu-cuda")]
fn parse_compute_capability(value: &str) -> Option<(i64, i64)> {
    let mut parts = value.split('.');
    let major = parts.next()?.trim().parse::<i64>().ok()?;
    let minor = parts.next().unwrap_or("0").trim().parse::<i64>().ok()?;
    Some((major, minor))
}

/// Smallest usable VRAM figure across the detected accelerators, in GB.
///
/// `pub` since 2026-08-09 so a caller reporting a retired, now-derived config
/// key can name the number this machine computes rather than a guess.
pub fn min_gpu_memory_gb(profile: &HardwareProfile) -> f64 {
    let min_vram = profile
        .gpu_mem_gb
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .fold(f64::INFINITY, f64::min);
    if min_vram.is_finite() { min_vram } else { 0.0 }
}

/// Memory allowance for a workload — CAPACITY, and the never-OOM invariant in
/// one function: the figure is a fraction of what the machine reports, never a
/// number a user typed.
pub fn planned_memory_budget_gb(
    profile: &HardwareProfile,
    backend: AcceleratorBackend,
    host_fraction: f64,
    gpu_fraction: f64,
) -> f64 {
    if !backend.is_gpu() {
        return profile.available_ram_gb.max(1.0) * host_fraction.clamp(0.0, 1.0);
    }

    let devices = profile.devices_for_planned_backend(backend);
    let min_dedicated_gb = devices
        .iter()
        .map(|device| device.memory_gb)
        .filter(|value| value.is_finite() && *value > 0.0)
        .fold(f64::INFINITY, f64::min);
    let capacity_gb = if min_dedicated_gb.is_finite() {
        min_dedicated_gb
    } else if backend.is_wgpu_family() && !devices.is_empty() {
        // wgpu intentionally does not report dedicated VRAM. This is an
        // allocation allowance derived from currently available shared host
        // memory, not a claim about physical VRAM capacity.
        (profile.available_ram_gb.max(1.0) * 0.25).clamp(1.0, 8.0)
    } else {
        0.0
    };

    capacity_gb * gpu_fraction.clamp(0.0, 1.0)
}

/// Deep-training batch size — CAPACITY. Same inputs, same answer, different
/// footprint: a bigger card does more rows per step, it does not change which
/// model comes out. Derived, never configured.
pub fn training_batch_size(enable_gpu: bool, min_vram_gb: f64) -> usize {
    if !enable_gpu {
        return 64;
    }
    if min_vram_gb >= 40.0 {
        2048
    } else if min_vram_gb >= 20.0 {
        1024
    } else if min_vram_gb >= 12.0 {
        512
    } else {
        256
    }
}

/// Inference batch size — CAPACITY, same argument as [`training_batch_size`].
pub fn inference_batch_size(enable_gpu: bool, min_vram_gb: f64) -> usize {
    if !enable_gpu {
        return 128;
    }
    if min_vram_gb >= 40.0 {
        8192
    } else if min_vram_gb >= 20.0 {
        4096
    } else if min_vram_gb >= 12.0 {
        2048
    } else {
        1024
    }
}

/// A config key that used to be settable and is now computed from the machine.
///
/// A stale copy of one of these in the operator's file must not be reported as a
/// nameless "unknown field": he set it on purpose once, and he is owed the
/// reason it stopped being his to set plus the number the machine computes
/// instead. `Settings::load` consults this table when it rejects a key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetiredDerivedKey {
    /// Dotted config path as it appears in the YAML.
    pub key: &'static str,
    /// What the machine computes it from, in words the operator can check.
    pub derived_from: &'static str,
    /// The knob that still IS his, when one exists.
    pub set_instead: Option<&'static str>,
}

/// Every key retired by the 2026-08-09 derive pass (D2 ⟷ D4).
///
/// These are CAPACITY knobs: a machine with twice the RAM wants a different
/// number, and only because of the hardware. Deliberately ABSENT from this table
/// — they change the answer, not the footprint, and stay settable:
/// `hpo_trials`, `prop_search_population`,
/// `prop_search_generations`, `prop_search_max_hours`, `prop_search_max_rows`,
/// `cpcv_max_rows`, `l1_feature_selection_sample_limit`, `global_max_rows`.
pub const RETIRED_DERIVED_KEYS: &[RetiredDerivedKey] = &[
    RetiredDerivedKey {
        key: "system.n_jobs",
        derived_from: "effective available_parallelism() minus the fixed two-thread stability reserve, then narrowed by system.hardware.cpu_budget",
        set_instead: Some("system.hardware.cpu_budget"),
    },
    RetiredDerivedKey {
        key: "system.num_gpus",
        derived_from: "the accelerator probe (HardwareProfile::num_gpus)",
        set_instead: Some("system.enable_gpu_preference"),
    },
    RetiredDerivedKey {
        key: "models.inference_batch_size",
        derived_from: "the Inference workload plan: GPU presence and the smallest detected VRAM",
        set_instead: None,
    },
];

/// Look a retired key up by its full dotted path or by its bare leaf name, so a
/// loader that only has `"n_jobs"` from a serde error still finds it.
pub fn retired_derived_key(key: &str) -> Option<&'static RetiredDerivedKey> {
    let needle = key.trim();
    RETIRED_DERIVED_KEYS.iter().find(|entry| {
        entry.key == needle
            || entry
                .key
                .rsplit('.')
                .next()
                .is_some_and(|leaf| leaf == needle)
    })
}

impl RetiredDerivedKey {
    /// What this machine computes for the key right now, when it can be had
    /// without spinning up a full accelerator probe.
    ///
    /// `None` means "needs the probe" — report the derivation rather than invent
    /// a number, because a wrong number in an error message is worse than none.
    pub fn computed_value(&self, settings: &Settings) -> Option<String> {
        match self.key {
            "system.n_jobs" => {
                let resolved = ExecutionBudgetInputs::from_settings_and_parent(
                    settings,
                    None,
                    CoordinationScope::ProcessLocal,
                )
                .ok()?
                .resolve()
                .ok()?;
                Some(resolved.effective_worker_limit.get().to_string())
            }
            _ => None,
        }
    }

    /// The whole message, in one place, so every caller says the same thing.
    pub fn ignored_message(&self, settings: &Settings) -> String {
        let computed = match self.computed_value(settings) {
            Some(value) => format!("this machine computes {value}"),
            None => "the value is computed per run from the hardware plan".to_string(),
        };
        let instead = match self.set_instead {
            Some(knob) => format!(" If you meant to constrain it, set `{knob}`."),
            None => String::new(),
        };
        format!(
            "`{}` is ignored because it is derived, not configured: {}; {}. \
             Remove the key from your config.{}",
            self.key, self.derived_from, computed, instead
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::BackendKind;

    #[test]
    fn hardware_from_settings_default_matches_default() {
        // A fresh Settings reproduces the env-absent HardwareRuntimeOverrides
        // (all None / empty), so the env -> config migration is
        // behaviour-preserving for default operators.
        let s = crate::config::Settings::default();
        assert_eq!(
            HardwareRuntimeOverrides::from_settings(&s),
            HardwareRuntimeOverrides::default()
        );
    }

    #[test]
    fn parent_cpu_assignment_is_a_separate_cap_and_does_not_mutate_settings() {
        let mut settings = crate::config::Settings::default();
        settings.system.hardware.cpu_budget = Some(12);
        settings.models.backtest_runtime.rayon_threads = Some(12);
        let original_canonical = settings.system.hardware.cpu_budget;
        let original_legacy = settings.models.backtest_runtime.rayon_threads;

        let resolved = ExecutionBudgetInputs::from_settings_parent_and_detection(
            &settings,
            Some(3),
            CapacityDetection::supplied(LogicalThreadCount::new(64).expect("positive")),
            CoordinationScope::ManagedProcessTree,
        )
        .expect("positive caps")
        .resolve()
        .expect("valid provenance");

        assert_eq!(resolved.effective_worker_limit.get(), 3);
        assert_eq!(settings.system.hardware.cpu_budget, original_canonical);
        assert_eq!(
            settings.models.backtest_runtime.rayon_threads,
            original_legacy
        );
    }

    #[cfg(feature = "gpu-wgpu")]
    fn wgpu_adapter_info(
        name: &str,
        backend: wgpu::Backend,
        device_type: wgpu::DeviceType,
        vendor: u32,
        device: u32,
    ) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: name.to_string(),
            vendor,
            device,
            device_type,
            device_pci_bus_id: String::new(),
            driver: "test-driver".to_string(),
            driver_info: "1.0".to_string(),
            backend,
            subgroup_min_size: 32,
            subgroup_max_size: 64,
            transient_saves_memory: false,
        }
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn wgpu_adapter_info_maps_integrated_gpu_without_fake_vram() {
        let infos = vec![
            wgpu_adapter_info(
                "AMD Radeon Graphics",
                wgpu::Backend::Vulkan,
                wgpu::DeviceType::IntegratedGpu,
                0x1002,
                0x1638,
            ),
            wgpu_adapter_info(
                "Microsoft Basic Render Driver",
                wgpu::Backend::Vulkan,
                wgpu::DeviceType::Cpu,
                0,
                0,
            ),
        ];

        let devices = normalize_wgpu_adapter_infos(&infos, None);

        assert_eq!(devices.len(), 1, "software adapters are not accelerators");
        assert_eq!(devices[0].backend, AcceleratorBackend::Vulkan);
        assert_eq!(
            devices[0].device_class,
            AcceleratorDeviceClass::IntegratedGpu
        );
        assert_eq!(devices[0].backend_index, 0);
        assert_eq!(devices[0].memory_gb, 0.0, "shared memory is not fake VRAM");
        assert!(devices[0].source.contains("wgpu"));
    }

    fn profile(gpus: usize, vram_gb: f64) -> HardwareProfile {
        HardwareProfile {
            schema_version: HARDWARE_PROFILE_SCHEMA_VERSION,
            cpu_cores: 64,
            total_ram_gb: 256.0,
            available_ram_gb: 192.0,
            gpu_names: (0..gpus).map(|idx| format!("GPU {idx}")).collect(),
            num_gpus: gpus,
            gpu_mem_gb: vec![vram_gb; gpus],
            accelerator_devices: (0..gpus)
                .map(|idx| AcceleratorDevice {
                    id: idx,
                    name: format!("GPU {idx}"),
                    backend: AcceleratorBackend::Cuda,
                    device_class: AcceleratorDeviceClass::DiscreteGpu,
                    backend_index: idx,
                    memory_gb: vram_gb,
                    supported_precisions: vec![
                        TrainingPrecision::Fp32,
                        TrainingPrecision::Fp16,
                        TrainingPrecision::Bf16,
                    ],
                    compute_capability: Some((8, 0)),
                    source: "test".to_string(),
                })
                .collect(),
            timestamp: "test".to_string(),
            platform_label: "test".to_string(),
        }
    }

    #[test]
    fn hardware_plan_assigns_gpu_search_and_keeps_ui_cpu_bound() {
        let mut settings = Settings::default();
        settings.system.enable_gpu_preference = "cuda".to_string();
        settings.models.prop_search_device = "auto".to_string();
        let plan = HardwareExecutionPlan::from_settings_and_profile(&settings, profile(2, 24.0));

        assert!(plan.gpu_enabled);
        assert_eq!(plan.primary_backend, AcceleratorBackend::Cuda);
        assert_eq!(
            plan.workload(WorkloadKind::StrategySearch).unwrap().device,
            "cuda:all"
        );
        assert_eq!(
            plan.workload(WorkloadKind::Ui).unwrap().backend,
            AcceleratorBackend::Cpu
        );
    }

    #[test]
    fn hardware_plan_uses_process_capacity_not_legacy_profile_inventory() {
        let settings = Settings::default();
        let resolved = ExecutionBudgetInputs::from_settings_parent_and_detection(
            &settings,
            None,
            CapacityDetection::supplied(LogicalThreadCount::new(12).expect("positive")),
            CoordinationScope::ProcessLocal,
        )
        .expect("default settings")
        .with_host_logical_threads(64)
        .expect("positive inventory")
        .resolve()
        .expect("valid provenance");
        let plan = HardwareExecutionPlan::from_settings_profile_overrides_and_budget(
            &settings,
            profile(0, 0.0),
            &HardwareRuntimeOverrides::from_settings(&settings),
            &resolved,
        );

        assert_eq!(plan.profile.cpu_cores, 64, "profile is inventory only");
        assert_eq!(plan.cpu_capacity.effective_logical_threads, 12);
        assert_eq!(plan.cpu_capacity.installed_worker_limit, 10);
        assert_eq!(
            plan.workload(WorkloadKind::DataIngestion)
                .expect("data workload")
                .cpu_threads,
            10
        );
        assert_eq!(
            plan.workload(WorkloadKind::Inference)
                .expect("inference workload")
                .cpu_threads,
            10
        );
        assert_eq!(
            plan.workload(WorkloadKind::Ui)
                .expect("UI control lane")
                .cpu_threads,
            0,
            "UI control owns no private worker pool"
        );
    }

    #[test]
    fn gpu_memory_budget_is_derived_from_device_memory_not_host_ram() {
        let mut settings = Settings::default();
        settings.system.enable_gpu_preference = "cuda".to_string();
        settings.models.prop_search_device = "auto".to_string();
        let plan = HardwareExecutionPlan::from_settings_and_profile(&settings, profile(1, 8.0));

        for kind in [
            WorkloadKind::StrategySearch,
            WorkloadKind::DeepTraining,
            WorkloadKind::Inference,
        ] {
            let gpu_budget = plan
                .workload_assignment(kind)
                .and_then(|assignment| assignment.gpu_budget)
                .expect("GPU workload should expose a GPU budget");
            assert!(
                gpu_budget.memory_budget_gb <= 8.0,
                "{kind:?} advertised {:.1} GB against an 8 GB device",
                gpu_budget.memory_budget_gb
            );
        }
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn hardware_plan_assigns_integrated_wgpu_to_strategy_search() {
        let mut settings = Settings::default();
        settings.system.enable_gpu_preference = "gpu".to_string();
        settings.models.prop_search_device = "auto".to_string();
        let profile = HardwareProfile {
            schema_version: HARDWARE_PROFILE_SCHEMA_VERSION,
            cpu_cores: 12,
            total_ram_gb: 32.0,
            available_ram_gb: 20.0,
            gpu_names: vec!["AMD Radeon Graphics".to_string()],
            num_gpus: 1,
            gpu_mem_gb: vec![0.0],
            accelerator_devices: vec![AcceleratorDevice {
                id: 0,
                name: "AMD Radeon Graphics".to_string(),
                backend: AcceleratorBackend::Vulkan,
                device_class: AcceleratorDeviceClass::IntegratedGpu,
                backend_index: 0,
                memory_gb: 0.0,
                supported_precisions: vec![TrainingPrecision::Fp32],
                compute_capability: None,
                source: "test-wgpu".to_string(),
            }],
            timestamp: "test".to_string(),
            platform_label: "test".to_string(),
        };

        let plan = HardwareExecutionPlan::from_settings_and_profile(&settings, profile);
        let search = plan
            .workload(WorkloadKind::StrategySearch)
            .expect("strategy-search plan should exist");

        assert_eq!(search.backend, AcceleratorBackend::Wgpu);
        assert_eq!(search.device, "vulkan:integrated:0");
        assert_eq!(search.device_ids, vec![0]);
        assert_eq!(search.memory_budget_gb, 4.0);
        assert!(
            search.memory_budget_gb < 20.0,
            "shared-memory allowance must remain below available host RAM"
        );
        assert!(search.runtime_degraded_reason().is_none());
    }

    #[test]
    fn hardware_plan_falls_back_to_cpu_when_gpu_requested_but_missing() {
        let mut settings = Settings::default();
        settings.system.enable_gpu_preference = "gpu".to_string();
        let plan = HardwareExecutionPlan::from_settings_and_profile(&settings, profile(0, 0.0));

        assert!(!plan.gpu_enabled);
        assert_eq!(plan.primary_backend, AcceleratorBackend::Cpu);
        assert!(!plan.warnings.is_empty());
    }

    #[test]
    fn canonical_cpu_setting_and_precision_override_resolve_without_env() {
        let mut settings = Settings::default();
        settings.system.enable_gpu_preference = "cuda".to_string();
        settings.system.hardware.cpu_budget = Some(4);
        let runtime_overrides = HardwareRuntimeOverrides {
            training_precision: Some(TrainingPrecision::Bf16),
            ..HardwareRuntimeOverrides::default()
        };

        let plan = HardwareExecutionPlan::from_settings_profile_and_overrides(
            &settings,
            profile(1, 24.0),
            &runtime_overrides,
        );

        assert_eq!(plan.preferred_precision, TrainingPrecision::Bf16);
        assert_eq!(
            plan.workload(WorkloadKind::DeepTraining)
                .expect("deep training workload should exist")
                .cpu_threads,
            4
        );
        assert_eq!(
            plan.workload_assignment(WorkloadKind::DeepTraining)
                .expect("deep training assignment should exist")
                .precision_policy
                .precision,
            TrainingPrecision::Bf16
        );
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn hardware_probe_consumes_typed_wgpu_overrides() {
        let runtime_overrides = HardwareRuntimeOverrides {
            wgpu_precisions: Some(vec![TrainingPrecision::Fp32, TrainingPrecision::Fp16]),
            wgpu_device_names: vec!["wgpu-test-device".to_string()],
            ..HardwareRuntimeOverrides::default()
        };
        let probe = HardwareProbe::with_runtime_overrides(runtime_overrides);

        let devices = probe.detect_wgpu_hint_accelerators(10);

        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].id, 10);
        assert_eq!(devices[0].name, "wgpu-test-device");
        assert_eq!(devices[0].backend, AcceleratorBackend::Wgpu);
        assert_eq!(
            devices[0].supported_precisions,
            vec![TrainingPrecision::Fp32, TrainingPrecision::Fp16]
        );
        assert_eq!(
            devices[0].source,
            "hardware_runtime_overrides.wgpu_device_names"
        );
    }

    #[test]
    fn hardware_profile_stable_id_ignores_ephemeral_probe_state() {
        let mut first = profile(1, 24.0);
        let mut second = first.clone();
        first.timestamp = "2026-05-06T00:00:00Z".to_string();
        second.timestamp = "2026-05-06T01:00:00Z".to_string();
        first.available_ram_gb = 190.0;
        second.available_ram_gb = 128.0;

        assert_eq!(first.stable_id(), second.stable_id());
    }

    #[test]
    fn workload_assignment_exposes_scheduler_owned_budgets_and_device() {
        let mut settings = Settings::default();
        settings.system.enable_gpu_preference = "cuda".to_string();
        settings.models.prop_search_device = "auto".to_string();
        let plan = HardwareExecutionPlan::from_settings_and_profile(&settings, profile(2, 24.0));

        let assignment = plan
            .workload_assignment(WorkloadKind::StrategySearch)
            .expect("search workload assignment should exist");

        assert_eq!(assignment.hardware_profile_id, plan.profile_id());
        assert_eq!(
            assignment.device_assignment.backend,
            BackendKind::NativeCuda
        );
        assert_eq!(assignment.device_assignment.device, "cuda:all");
        assert_eq!(assignment.device_assignment.device_ids, vec![0, 1]);
        assert!(assignment.cpu_budget.threads > 0);
        assert_eq!(
            assignment.gpu_budget.as_ref().unwrap().device_ids,
            vec![0, 1]
        );
        assert_eq!(
            assignment.precision_policy.precision,
            TrainingPrecision::Fp32
        );
        assert!(assignment.runtime_degraded_reason.is_none());
    }

    #[test]
    fn workload_assignment_records_cpu_degradation_when_gpu_path_falls_back() {
        let mut settings = Settings::default();
        settings.system.enable_gpu_preference = "rocm".to_string();
        let profile = HardwareProfile {
            schema_version: HARDWARE_PROFILE_SCHEMA_VERSION,
            cpu_cores: 64,
            total_ram_gb: 256.0,
            available_ram_gb: 192.0,
            gpu_names: vec!["AMD GPU".to_string()],
            num_gpus: 1,
            gpu_mem_gb: vec![24.0],
            accelerator_devices: vec![AcceleratorDevice {
                id: 0,
                name: "AMD GPU".to_string(),
                backend: AcceleratorBackend::Rocm,
                device_class: AcceleratorDeviceClass::DiscreteGpu,
                backend_index: 0,
                memory_gb: 24.0,
                supported_precisions: vec![TrainingPrecision::Fp32, TrainingPrecision::Fp16],
                compute_capability: None,
                source: "test".to_string(),
            }],
            timestamp: "test".to_string(),
            platform_label: "test".to_string(),
        };
        let plan = HardwareExecutionPlan::from_settings_and_profile(&settings, profile);

        let assignment = plan
            .workload_assignment(WorkloadKind::StrategySearch)
            .expect("search workload assignment should exist");

        assert_eq!(assignment.device_assignment.backend, BackendKind::NativeCpu);
        assert_eq!(assignment.device_assignment.device, "cpu");
        assert_eq!(
            assignment.runtime_degraded_reason.unwrap().code,
            "gpu_path_unavailable"
        );
    }

    #[test]
    fn hardware_plan_keeps_rocm_as_primary_backend_when_only_rocm_is_available() {
        let mut settings = Settings::default();
        settings.system.enable_gpu_preference = "rocm".to_string();
        let profile = HardwareProfile {
            schema_version: HARDWARE_PROFILE_SCHEMA_VERSION,
            cpu_cores: 64,
            total_ram_gb: 256.0,
            available_ram_gb: 192.0,
            gpu_names: vec!["AMD GPU".to_string()],
            num_gpus: 1,
            gpu_mem_gb: vec![24.0],
            accelerator_devices: vec![AcceleratorDevice {
                id: 0,
                name: "AMD GPU".to_string(),
                backend: AcceleratorBackend::Rocm,
                device_class: AcceleratorDeviceClass::DiscreteGpu,
                backend_index: 0,
                memory_gb: 24.0,
                supported_precisions: vec![TrainingPrecision::Fp32, TrainingPrecision::Fp16],
                compute_capability: None,
                source: "test".to_string(),
            }],
            timestamp: "test".to_string(),
            platform_label: "test".to_string(),
        };

        let plan = HardwareExecutionPlan::from_settings_and_profile(&settings, profile);

        assert!(plan.gpu_enabled);
        assert_eq!(plan.primary_backend, AcceleratorBackend::Rocm);
        assert_eq!(
            plan.workload(WorkloadKind::DeepTraining).unwrap().device,
            "rocm:0"
        );
        assert_eq!(
            plan.workload(WorkloadKind::StrategySearch).unwrap().backend,
            AcceleratorBackend::Cpu
        );
    }

    /// A container limit must win, and a non-limit must not.
    ///
    /// This existed as an inline closure and had no test, which is how the
    /// system spent every rented vast.ai run sizing itself against the HOST's
    /// RAM instead of the container's — `available_memory()` answers for the
    /// machine, not for the process. Invert the comparison here and the bug
    /// comes back silently, with the run merely meaning something weaker than
    /// it claims.
    #[test]
    fn a_cgroup_limit_tightens_but_an_absent_one_does_not() {
        const HOST: u64 = 64 * 1024 * 1024 * 1024;
        const CONTAINER: u64 = 12 * 1024 * 1024 * 1024;

        // The case that matters: a real container budget below the host.
        assert_eq!(tighter_of(HOST, CONTAINER), CONTAINER);

        // 0 means the cgroup did not report a limit — not "no memory".
        assert_eq!(tighter_of(HOST, 0), HOST);

        // An unconstrained cgroup reports the host figure, or more.
        assert_eq!(tighter_of(HOST, HOST), HOST);
        assert_eq!(tighter_of(HOST, HOST * 2), HOST);

        // And it never invents memory the host does not have.
        assert!(tighter_of(HOST, CONTAINER) <= HOST);
    }

    #[test]
    fn reported_zero_cgroup_headroom_clamps_available_but_not_total_memory() {
        const HOST_TOTAL: u64 = 64 * 1024 * 1024 * 1024;
        const HOST_AVAILABLE: u64 = 32 * 1024 * 1024 * 1024;

        assert_eq!(
            clamp_memory_figures_to_reported_cgroup(HOST_TOTAL, HOST_AVAILABLE, 0, 0),
            (HOST_TOTAL, 0)
        );
    }

    #[test]
    fn current_process_cgroup_wins_and_root_is_only_a_fallback() {
        let process = (128_u64, 32_u64);
        let root = (256_u64, 200_u64);

        assert_eq!(
            preferred_cgroup_memory_limits(Some(process), Some(root)),
            Some(process)
        );
        assert_eq!(preferred_cgroup_memory_limits(None, Some(root)), Some(root));
        assert_eq!(preferred_cgroup_memory_limits(None, None), None);
    }
}
