use crate::config::Settings;
use serde::{Deserialize, Serialize};
use std::env;
use std::process::Command;
use sysinfo::System;
use tracing::{info, warn};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareProfile {
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

pub struct HardwareProbe {
    sys: System,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcceleratorBackend {
    Cpu,
    Cuda,
    Rocm,
    Wgpu,
    Vulkan,
    Metal,
    Dx12,
}

impl AcceleratorBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
            Self::Wgpu => "wgpu",
            Self::Vulkan => "vulkan",
            Self::Metal => "metal",
            Self::Dx12 => "dx12",
        }
    }

    pub fn is_gpu(self) -> bool {
        !matches!(self, Self::Cpu)
    }

    pub fn is_cuda_native(self) -> bool {
        matches!(self, Self::Cuda)
    }

    pub fn is_wgpu_family(self) -> bool {
        matches!(self, Self::Wgpu | Self::Vulkan | Self::Metal | Self::Dx12)
    }
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
    pub id: usize,
    pub name: String,
    pub backend: AcceleratorBackend,
    pub memory_gb: f64,
    pub supported_precisions: Vec<TrainingPrecision>,
    pub compute_capability: Option<(i64, i64)>,
    pub source: String,
}

impl AcceleratorDevice {
    pub fn device_string(&self) -> String {
        format!("{}:{}", self.backend.as_str(), self.id)
    }

    pub fn supports_precision(&self, precision: TrainingPrecision) -> bool {
        self.supported_precisions.contains(&precision)
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkloadExecutionPlan {
    pub workload: WorkloadKind,
    pub backend: AcceleratorBackend,
    pub device: String,
    pub device_ids: Vec<usize>,
    pub precision: TrainingPrecision,
    pub cpu_threads: usize,
    pub batch_size: usize,
    pub memory_budget_gb: f64,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareExecutionPlan {
    pub profile: HardwareProfile,
    pub gpu_enabled: bool,
    pub primary_backend: AcceleratorBackend,
    pub preferred_precision: TrainingPrecision,
    pub workloads: Vec<WorkloadExecutionPlan>,
    pub warnings: Vec<String>,
}

impl HardwareExecutionPlan {
    pub fn from_settings_and_profile(settings: &Settings, profile: HardwareProfile) -> Self {
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
        let preferred_precision = choose_training_precision(&profile, primary_backend);
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
        if primary_backend.is_wgpu_family() || primary_backend == AcceleratorBackend::Rocm {
            warnings.push(
                "Non-CUDA deep planning applies to Burn/deep workloads; current search/RL native tensor paths remain CUDA-only unless explicitly refactored."
                    .to_string(),
            );
        }

        let cpu_budget = resolve_cpu_budget_from_env(profile.cpu_cores.max(1));
        let memory_budget_gb = profile.available_ram_gb.max(1.0);
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

        let mut workloads = Vec::new();
        workloads.push(WorkloadExecutionPlan {
            workload: WorkloadKind::DataIngestion,
            backend: AcceleratorBackend::Cpu,
            device: "cpu".to_string(),
            device_ids: Vec::new(),
            precision: choose_training_precision(
                profile,
                if gpu_enabled && search_gpu_requested && !cuda_devices.is_empty() {
                    AcceleratorBackend::Cuda
                } else {
                    AcceleratorBackend::Cpu
                },
            ),
            cpu_threads: cpu_budget.clamp(1, 8),
            batch_size: 0,
            memory_budget_gb: memory_budget_gb * 0.20,
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
            cpu_threads: cpu_budget,
            batch_size: 0,
            memory_budget_gb: memory_budget_gb * 0.35,
            notes: vec![
                "ICT/SMC remains feature engineering only; model decisions stay autonomous."
                    .to_string(),
            ],
        });
        workloads.push(WorkloadExecutionPlan {
            workload: WorkloadKind::StrategySearch,
            backend: if gpu_enabled && search_gpu_requested && !cuda_devices.is_empty() {
                AcceleratorBackend::Cuda
            } else {
                AcceleratorBackend::Cpu
            },
            device: if gpu_enabled && search_gpu_requested && !cuda_devices.is_empty() {
                "cuda:all".to_string()
            } else {
                "cpu".to_string()
            },
            device_ids: if gpu_enabled && search_gpu_requested && !cuda_devices.is_empty() {
                cuda_devices.iter().map(|device| device.id).collect()
            } else {
                Vec::new()
            },
            precision: TrainingPrecision::Fp32,
            cpu_threads: cpu_budget,
            batch_size: if gpu_enabled { train_batch_size } else { 0 },
            memory_budget_gb: memory_budget_gb * 0.45,
            notes: vec!["Search evaluation now uses CUDA kernels for GA offspring generation, generic signal synthesis, and the per-gene stateful backtest loop; signal synthesis can follow the requested training precision, while the price-normalized backtest kernel stays FP32 for pip-safe arithmetic; WGPU/ROCm/Metal discovery still reports CPU fallback until native evaluator kernels exist there.".to_string()],
        });
        workloads.push(WorkloadExecutionPlan {
            workload: WorkloadKind::TreeTraining,
            backend: if gpu_enabled && tree_gpu_requested && !cuda_devices.is_empty() {
                AcceleratorBackend::Cuda
            } else {
                AcceleratorBackend::Cpu
            },
            device: if gpu_enabled && tree_gpu_requested && !cuda_devices.is_empty() {
                "cuda:0".to_string()
            } else {
                "cpu".to_string()
            },
            device_ids: if gpu_enabled && tree_gpu_requested && !cuda_devices.is_empty() {
                vec![0]
            } else {
                Vec::new()
            },
            precision: TrainingPrecision::Fp32,
            cpu_threads: cpu_budget,
            batch_size: train_batch_size,
            memory_budget_gb: memory_budget_gb * 0.35,
            notes: vec!["Tree GPU support depends on each native backend feature; fallback must stay explicit in metadata.".to_string()],
        });
        workloads.push(WorkloadExecutionPlan {
            workload: WorkloadKind::DeepTraining,
            backend: primary_backend,
            device: primary_device.clone(),
            device_ids: device_ids.clone(),
            precision: preferred_precision,
            cpu_threads: cpu_budget,
            batch_size: train_batch_size,
            memory_budget_gb: memory_budget_gb * 0.55,
            notes: vec![format!(
                "Burn/deep training should use planner policy with effective precision {}.",
                preferred_precision.as_str()
            )],
        });
        workloads.push(WorkloadExecutionPlan {
            workload: WorkloadKind::RlTraining,
            backend: if gpu_enabled && !cuda_devices.is_empty() {
                AcceleratorBackend::Cuda
            } else {
                AcceleratorBackend::Cpu
            },
            device: if gpu_enabled && !cuda_devices.is_empty() {
                "cuda:0".to_string()
            } else {
                "cpu".to_string()
            },
            device_ids: if gpu_enabled && !cuda_devices.is_empty() {
                vec![0]
            } else {
                Vec::new()
            },
            precision: TrainingPrecision::Fp32,
            cpu_threads: cpu_budget,
            batch_size: train_batch_size,
            memory_budget_gb: memory_budget_gb * 0.35,
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
            cpu_threads: cpu_budget.clamp(1, 16),
            batch_size: infer_batch_size,
            memory_budget_gb: memory_budget_gb * 0.20,
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
            cpu_threads: 2.min(cpu_budget).max(1),
            batch_size: 0,
            memory_budget_gb: memory_budget_gb * 0.05,
            notes: vec!["UI stays message-channel driven and never owns ML/GPU work.".to_string()],
        });

        Self {
            profile,
            gpu_enabled,
            primary_backend,
            preferred_precision,
            workloads,
            warnings,
        }
    }

    pub fn workload(&self, kind: WorkloadKind) -> Option<&WorkloadExecutionPlan> {
        self.workloads.iter().find(|plan| plan.workload == kind)
    }
}

impl Default for HardwareProbe {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareProbe {
    pub fn new() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();
        Self { sys }
    }

    pub fn detect(&mut self) -> HardwareProfile {
        self.sys.refresh_all();

        let cpu_cores = self.sys.cpus().len().max(1);
        let total_ram_gb = self.sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
        let available_ram_gb = self.sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0;

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
        let mut devices = self.detect_nvidia_accelerators();
        devices.extend(self.detect_rocm_accelerators(devices.len()));
        devices.extend(self.detect_wgpu_hint_accelerators(devices.len()));
        devices
    }

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
                    memory_gb: mems.get(idx).copied().unwrap_or(0.0),
                    supported_precisions,
                    compute_capability,
                    source: "nvidia-smi".to_string(),
                }
            })
            .collect()
    }

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
            {
                if output.status.success() {
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
        }

        (vec![], vec![])
    }

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
            let Ok(output) = Command::new(cmd)
                .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
                .output()
            else {
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

    fn detect_rocm_accelerators(&self, id_offset: usize) -> Vec<AcceleratorDevice> {
        let rocminfo = Command::new("rocminfo").output().ok();
        if let Some(output) = rocminfo {
            if output.status.success() {
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
                            memory_gb: 0.0,
                            supported_precisions: env_precision_override("rocm").unwrap_or_else(
                                || vec![TrainingPrecision::Fp32, TrainingPrecision::Fp16],
                            ),
                            compute_capability: None,
                            source: "rocminfo".to_string(),
                        })
                        .collect();
                }
            }
        }

        Vec::new()
    }

    fn detect_wgpu_hint_accelerators(&self, id_offset: usize) -> Vec<AcceleratorDevice> {
        let Ok(raw) = env::var("FOREX_BOT_WGPU_DEVICES") else {
            return Vec::new();
        };
        raw.split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .enumerate()
            .map(|(idx, name)| AcceleratorDevice {
                id: id_offset + idx,
                name: name.to_string(),
                backend: AcceleratorBackend::Wgpu,
                memory_gb: 0.0,
                supported_precisions: env_precision_override("wgpu")
                    .unwrap_or_else(|| vec![TrainingPrecision::Fp32]),
                compute_capability: None,
                source: "FOREX_BOT_WGPU_DEVICES".to_string(),
            })
            .collect()
    }
}

impl HardwareProfile {
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
            .filter(|device| {
                device.backend.is_wgpu_family() || device.backend == AcceleratorBackend::Rocm
            })
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

pub struct AutoTuner<'a> {
    settings: &'a mut Settings,
    profile: HardwareProfile,
}

#[derive(Debug, Clone)]
pub struct AutoTuneHints {
    pub enable_gpu: bool,
    pub num_gpus: usize,
    pub device: String,
    pub prop_search_device: String,
    pub tree_device_preference: String,
    pub n_jobs: usize,
    pub train_batch_size: usize,
    pub inference_batch_size: usize,
    pub hpo_trials: usize,
    pub adaptive_training_budget: f64,
    pub feature_workers: usize,
    pub is_hpc: bool,
    pub execution_plan: HardwareExecutionPlan,
}

impl<'a> AutoTuner<'a> {
    pub fn new(settings: &'a mut Settings, profile: HardwareProfile) -> Self {
        Self { settings, profile }
    }

    pub fn apply(&mut self) -> AutoTuneHints {
        let hints = self.evaluate_hints();

        self.settings.system.enable_gpu = hints.enable_gpu;
        self.settings.system.num_gpus = hints.num_gpus;
        self.settings.system.device = hints.device.clone();
        self.settings.system.n_jobs = hints.n_jobs;
        self.settings.models.prop_search_device = hints.prop_search_device.clone();
        self.settings.models.tree_device_preference = hints.tree_device_preference.clone();

        self.settings.models.train_batch_size = hints.train_batch_size;
        self.settings.models.inference_batch_size = hints.inference_batch_size;
        self.settings.models.hpo_trials = hints.hpo_trials;

        self.apply_thread_env_defaults(&hints);

        info!(
            "Auto-Tuner Applied: GPU={} Device={}",
            hints.enable_gpu, hints.device
        );
        for warning in &hints.execution_plan.warnings {
            warn!("Hardware planner warning: {}", warning);
        }

        hints
    }

    fn evaluate_hints(&self) -> AutoTuneHints {
        let plan =
            HardwareExecutionPlan::from_settings_and_profile(self.settings, self.profile.clone());
        let cpu_cores = self.profile.cpu_cores.max(1);
        let ram_gb = self.profile.available_ram_gb;
        let cpu_budget = resolve_cpu_budget_from_env(cpu_cores);
        let (
            train_device,
            train_batch_size,
            inference_batch_size,
            prop_search_device,
            tree_device_preference,
        ) = {
            let train_plan = plan
                .workload(WorkloadKind::DeepTraining)
                .expect("hardware planner must include deep-training plan");
            let infer_plan = plan
                .workload(WorkloadKind::Inference)
                .expect("hardware planner must include inference plan");
            let search_plan = plan
                .workload(WorkloadKind::StrategySearch)
                .expect("hardware planner must include search plan");
            let tree_plan = plan
                .workload(WorkloadKind::TreeTraining)
                .expect("hardware planner must include tree-training plan");
            (
                train_plan.device.clone(),
                train_plan.batch_size,
                infer_plan.batch_size,
                search_plan.device.clone(),
                tree_plan.device.clone(),
            )
        };

        // Feature workers logic
        let per_worker_gb = 2.0;
        let ram_based_workers = (ram_gb / per_worker_gb) as usize;
        let feature_workers = 1.max(cpu_budget.min(ram_based_workers));

        AutoTuneHints {
            enable_gpu: plan.gpu_enabled,
            num_gpus: if plan.gpu_enabled {
                self.profile.num_gpus
            } else {
                0
            },
            device: train_device,
            prop_search_device,
            tree_device_preference,
            n_jobs: cpu_budget,
            train_batch_size,
            inference_batch_size,
            hpo_trials: if plan.gpu_enabled { 50 } else { 20 },
            adaptive_training_budget: if plan.gpu_enabled { 3600.0 } else { 1800.0 },
            feature_workers,
            is_hpc: ram_gb > 64.0 && cpu_cores >= 32,
            execution_plan: plan,
        }
    }

    fn resolve_cpu_budget(&self, total_cores: usize) -> usize {
        resolve_cpu_budget_from_env(total_cores)
    }

    fn apply_thread_env_defaults(&self, hints: &AutoTuneHints) {
        let n_threads = hints.n_jobs.max(1).to_string();
        unsafe {
            env::set_var("OMP_NUM_THREADS", &n_threads);
            env::set_var("MKL_NUM_THREADS", &n_threads);
            env::set_var("OPENBLAS_NUM_THREADS", &n_threads);
        }
    }
}

fn normalize_accelerator_preference(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => "auto".to_string(),
        "cpu" | "false" | "0" | "no" | "off" => "cpu".to_string(),
        "gpu" | "true" | "1" | "yes" | "on" | "nvidia" | "amd" => "gpu".to_string(),
        "cuda" | "cuda:0" => "cuda".to_string(),
        "rocm" | "hip" | "rocm:0" | "hip:0" => "rocm".to_string(),
        "wgpu" | "wgpu:0" | "wgpu_vulkan" => "wgpu".to_string(),
        "vulkan" | "vulkan:0" => "vulkan".to_string(),
        "metal" | "metal:0" => "metal".to_string(),
        "dx12" | "directx12" | "d3d12" | "dx12:0" => "dx12".to_string(),
        other => other.to_string(),
    }
}

fn choose_primary_backend(preference: &str, profile: &HardwareProfile) -> AcceleratorBackend {
    if profile.accelerator_devices.is_empty() || preference == "cpu" || preference == "off" {
        return AcceleratorBackend::Cpu;
    }

    let has_cuda = !profile
        .devices_for_backend(AcceleratorBackend::Cuda)
        .is_empty();
    let has_rocm = !profile
        .devices_for_backend(AcceleratorBackend::Rocm)
        .is_empty();
    let has_wgpu = !profile.wgpu_native_devices().is_empty();

    match preference {
        "cuda" => {
            if has_cuda {
                AcceleratorBackend::Cuda
            } else {
                AcceleratorBackend::Cpu
            }
        }
        "rocm" => {
            if has_rocm {
                AcceleratorBackend::Rocm
            } else {
                AcceleratorBackend::Cpu
            }
        }
        "wgpu" => {
            if has_wgpu {
                AcceleratorBackend::Wgpu
            } else {
                AcceleratorBackend::Cpu
            }
        }
        "vulkan" => {
            if has_wgpu {
                AcceleratorBackend::Vulkan
            } else {
                AcceleratorBackend::Cpu
            }
        }
        "metal" => {
            if has_wgpu {
                AcceleratorBackend::Metal
            } else {
                AcceleratorBackend::Cpu
            }
        }
        "dx12" => {
            if has_wgpu {
                AcceleratorBackend::Dx12
            } else {
                AcceleratorBackend::Cpu
            }
        }
        "gpu" | "auto" => {
            if has_cuda {
                AcceleratorBackend::Cuda
            } else if has_rocm {
                AcceleratorBackend::Rocm
            } else if has_wgpu {
                AcceleratorBackend::Wgpu
            } else {
                AcceleratorBackend::Cpu
            }
        }
        _ if has_cuda => AcceleratorBackend::Cuda,
        _ if has_wgpu => AcceleratorBackend::Wgpu,
        _ if has_rocm => AcceleratorBackend::Rocm,
        _ => AcceleratorBackend::Cpu,
    }
}

fn choose_training_precision(
    profile: &HardwareProfile,
    backend: AcceleratorBackend,
) -> TrainingPrecision {
    let requested = requested_training_precision_from_env();
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

fn requested_training_precision_from_env() -> Option<TrainingPrecision> {
    ["FOREX_BOT_TRAIN_PRECISION", "FOREX_TRAIN_PRECISION"]
        .iter()
        .find_map(|key| env::var(key).ok())
        .and_then(|value| parse_training_precision(&value))
}

fn parse_training_precision(value: &str) -> Option<TrainingPrecision> {
    match value.trim().to_ascii_lowercase().as_str() {
        "fp32" | "f32" | "float32" => Some(TrainingPrecision::Fp32),
        "fp16" | "f16" | "float16" | "half" => Some(TrainingPrecision::Fp16),
        "bf16" | "bfloat16" => Some(TrainingPrecision::Bf16),
        "fp8" | "float8" => Some(TrainingPrecision::Fp8),
        "bf4" => Some(TrainingPrecision::Bf4),
        "auto" | "" => None,
        _ => None,
    }
}

fn env_precision_override(backend: &str) -> Option<Vec<TrainingPrecision>> {
    let key = format!("FOREX_BOT_{}_PRECISIONS", backend.to_ascii_uppercase());
    let Ok(raw) = env::var(key) else {
        return None;
    };
    let values = raw
        .split(',')
        .filter_map(parse_training_precision)
        .collect::<Vec<_>>();
    (!values.is_empty()).then_some(values)
}

fn parse_compute_capability(value: &str) -> Option<(i64, i64)> {
    let mut parts = value.split('.');
    let major = parts.next()?.trim().parse::<i64>().ok()?;
    let minor = parts.next().unwrap_or("0").trim().parse::<i64>().ok()?;
    Some((major, minor))
}

fn min_gpu_memory_gb(profile: &HardwareProfile) -> f64 {
    let min_vram = profile
        .gpu_mem_gb
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .fold(f64::INFINITY, f64::min);
    if min_vram.is_finite() { min_vram } else { 0.0 }
}

fn training_batch_size(enable_gpu: bool, min_vram_gb: f64) -> usize {
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

fn inference_batch_size(enable_gpu: bool, min_vram_gb: f64) -> usize {
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

fn resolve_cpu_budget_from_env(total_cores: usize) -> usize {
    if let Ok(val) = env::var("FOREX_BOT_CPU_BUDGET") {
        if let Ok(n) = val.parse::<usize>() {
            return n.min(total_cores).max(1);
        }
    }
    total_cores.saturating_sub(1).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(gpus: usize, vram_gb: f64) -> HardwareProfile {
        HardwareProfile {
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
    fn hardware_plan_falls_back_to_cpu_when_gpu_requested_but_missing() {
        let mut settings = Settings::default();
        settings.system.enable_gpu_preference = "gpu".to_string();
        let plan = HardwareExecutionPlan::from_settings_and_profile(&settings, profile(0, 0.0));

        assert!(!plan.gpu_enabled);
        assert_eq!(plan.primary_backend, AcceleratorBackend::Cpu);
        assert!(!plan.warnings.is_empty());
    }

    #[test]
    fn hardware_plan_keeps_rocm_as_primary_backend_when_only_rocm_is_available() {
        let mut settings = Settings::default();
        settings.system.enable_gpu_preference = "rocm".to_string();
        let profile = HardwareProfile {
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
}
