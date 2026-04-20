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
    Wgpu,
}

impl AcceleratorBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Wgpu => "wgpu",
        }
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
    pub workloads: Vec<WorkloadExecutionPlan>,
    pub warnings: Vec<String>,
}

impl HardwareExecutionPlan {
    pub fn from_settings_and_profile(settings: &Settings, profile: HardwareProfile) -> Self {
        let preference = normalize_accelerator_preference(&settings.system.enable_gpu_preference);
        let has_gpu = profile.num_gpus > 0;
        let gpu_allowed = !matches!(preference.as_str(), "cpu" | "off");
        let gpu_forced = matches!(preference.as_str(), "gpu" | "cuda" | "wgpu");
        let gpu_enabled = has_gpu && gpu_allowed;
        let primary_backend = if gpu_enabled {
            match preference.as_str() {
                "wgpu" => AcceleratorBackend::Wgpu,
                _ => AcceleratorBackend::Cuda,
            }
        } else {
            AcceleratorBackend::Cpu
        };
        let mut warnings = Vec::new();
        if gpu_forced && !has_gpu {
            warnings.push(
                "GPU was requested but no NVIDIA GPU was detected by nvidia-smi; using CPU plans."
                    .to_string(),
            );
        }
        if matches!(primary_backend, AcceleratorBackend::Wgpu) {
            warnings.push(
                "WGPU is planned only for Burn/deep workloads; tensor/search CUDA paths remain separate."
                    .to_string(),
            );
        }

        let cpu_budget = resolve_cpu_budget_from_env(profile.cpu_cores.max(1));
        let memory_budget_gb = profile.available_ram_gb.max(1.0);
        let device_ids: Vec<usize> = if gpu_enabled {
            (0..profile.num_gpus).collect()
        } else {
            Vec::new()
        };
        let primary_device = if gpu_enabled {
            format!("{}:0", primary_backend.as_str())
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
            backend: if gpu_enabled && search_gpu_requested {
                AcceleratorBackend::Cuda
            } else {
                AcceleratorBackend::Cpu
            },
            device: if gpu_enabled && search_gpu_requested {
                "cuda:all".to_string()
            } else {
                "cpu".to_string()
            },
            device_ids: if gpu_enabled && search_gpu_requested {
                device_ids.clone()
            } else {
                Vec::new()
            },
            cpu_threads: cpu_budget,
            batch_size: if gpu_enabled { train_batch_size } else { 0 },
            memory_budget_gb: memory_budget_gb * 0.45,
            notes: vec!["Search should use the forex-search GPU feature when enabled; otherwise it must report CPU fallback.".to_string()],
        });
        workloads.push(WorkloadExecutionPlan {
            workload: WorkloadKind::TreeTraining,
            backend: if gpu_enabled && tree_gpu_requested {
                AcceleratorBackend::Cuda
            } else {
                AcceleratorBackend::Cpu
            },
            device: if gpu_enabled && tree_gpu_requested {
                "cuda:0".to_string()
            } else {
                "cpu".to_string()
            },
            device_ids: if gpu_enabled && tree_gpu_requested {
                vec![0]
            } else {
                Vec::new()
            },
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
            cpu_threads: cpu_budget,
            batch_size: train_batch_size,
            memory_budget_gb: memory_budget_gb * 0.55,
            notes: vec!["Burn/RL tensor training receives the planner device policy instead of ad-hoc per-model defaults.".to_string()],
        });
        workloads.push(WorkloadExecutionPlan {
            workload: WorkloadKind::RlTraining,
            backend: if gpu_enabled {
                AcceleratorBackend::Cuda
            } else {
                AcceleratorBackend::Cpu
            },
            device: if gpu_enabled {
                "cuda:0".to_string()
            } else {
                "cpu".to_string()
            },
            device_ids: if gpu_enabled { vec![0] } else { Vec::new() },
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
            cpu_threads: 2.min(cpu_budget).max(1),
            batch_size: 0,
            memory_budget_gb: memory_budget_gb * 0.05,
            notes: vec!["UI stays message-channel driven and never owns ML/GPU work.".to_string()],
        });

        Self {
            profile,
            gpu_enabled,
            primary_backend,
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

        let (gpu_names, gpu_mem_gb) = self.detect_gpus_nvidia_smi();
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
            timestamp: chrono::Utc::now().to_rfc3339(),
            platform_label,
        }
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
        "gpu" | "true" | "1" | "yes" | "on" | "nvidia" => "gpu".to_string(),
        "cuda" | "cuda:0" => "cuda".to_string(),
        "wgpu" | "vulkan" | "wgpu_vulkan" => "wgpu".to_string(),
        other => other.to_string(),
    }
}

fn min_gpu_memory_gb(profile: &HardwareProfile) -> f64 {
    let min_vram = profile
        .gpu_mem_gb
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .fold(f64::INFINITY, f64::min);
    if min_vram.is_finite() {
        min_vram
    } else {
        0.0
    }
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
}
