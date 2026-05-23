// Hardware / accelerator capability detection (CPU features, CUDA devices, Metal).
// REMOVES: legacy multiprocessing, GIL-related threading workarounds

use std::env;
#[cfg(feature = "tch")]
use std::time::Instant;
use tracing::{debug, info, warn};

use neoethos_core::system::HardwareProbe;
use neoethos_core::{
    AcceleratorBackend, AcceleratorDevice, BackendKind, ResolvedWorkloadAssignment,
    TrainingPrecision,
};
#[cfg(feature = "tch")]
use tch::{Cuda, Device, Kind, Tensor};

type GpuDetection = (usize, Vec<String>, Vec<f64>, Vec<(i64, i64)>);

// ============================================================================
// HARDWARE INFO STRUCTURE
// ============================================================================

#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub cpu_cores: usize,
    pub cpu_cores_usable: usize, // cores - 1 for OS stability
    pub gpu_count: usize,
    pub gpu_names: Vec<String>,
    pub gpu_memory_gb: Vec<f64>,
    pub compute_capabilities: Vec<(i64, i64)>,
    pub accelerator_devices: Vec<AcceleratorDevice>,
    pub os_name: String,
}

impl HardwareInfo {
    /// Auto-detect all hardware: CPUs, GPUs, RAM, OS
    /// This is what legacy needed multiprocessing to emulate
    pub fn detect() -> Self {
        let cpu_cores = num_cpus::get();
        let cpu_cores_usable = cpu_cores.saturating_sub(1).max(1); // Reserve 1 for OS

        let mut core_probe = HardwareProbe::new();
        let core_profile = core_probe.detect();
        let accelerator_devices = core_profile.accelerator_devices;
        let (gpu_count, gpu_names, gpu_memory_gb, compute_capabilities) =
            if accelerator_devices.is_empty() {
                Self::detect_gpus()
            } else {
                let names = accelerator_devices
                    .iter()
                    .map(|device| device.name.clone())
                    .collect::<Vec<_>>();
                let memory = accelerator_devices
                    .iter()
                    .map(|device| device.memory_gb)
                    .collect::<Vec<_>>();
                let capabilities = accelerator_devices
                    .iter()
                    .map(|device| device.compute_capability.unwrap_or((0, 0)))
                    .collect::<Vec<_>>();
                (accelerator_devices.len(), names, memory, capabilities)
            };

        let os_name = env::consts::OS.to_string();

        info!(
            "Hardware detected: {} CPUs ({} usable), {} GPUs, OS: {}",
            cpu_cores, cpu_cores_usable, gpu_count, os_name
        );

        for (i, name) in gpu_names.iter().enumerate() {
            info!(
                "  GPU {}: {} ({:.1} GB, SM {}.{})",
                i,
                name,
                gpu_memory_gb.get(i).unwrap_or(&0.0),
                compute_capabilities.get(i).map(|c| c.0).unwrap_or(0),
                compute_capabilities.get(i).map(|c| c.1).unwrap_or(0),
            );
        }

        Self {
            cpu_cores,
            cpu_cores_usable,
            gpu_count,
            gpu_names,
            gpu_memory_gb,
            compute_capabilities,
            accelerator_devices,
            os_name,
        }
    }

    /// Detect GPUs using tch (PyTorch bindings)
    /// legacy lines 61-73
    fn detect_gpus() -> GpuDetection {
        #[cfg(feature = "tch")]
        {
            if !Cuda::is_available() {
                return (0, vec![], vec![], vec![]);
            }

            let count = Cuda::device_count();
            let mut names = Vec::new();
            let mut memory = Vec::new();
            let mut capabilities = Vec::new();

            for i in 0..count {
                // Get device properties
                match Cuda::get_device_name(i as i64) {
                    Ok(name) => names.push(name),
                    Err(_) => names.push(format!("GPU {}", i)),
                }

                // Get memory (in GB)
                // tch doesn't expose total memory directly, estimate from cudaGetDeviceProperties
                // For now, set to 0.0 and it can be queried separately if needed
                memory.push(0.0);

                // Get compute capability
                capabilities.push(Cuda::get_device_capability(i as i64));
            }

            (count as usize, names, memory, capabilities)
        }

        #[cfg(not(feature = "tch"))]
        {
            (0, vec![], vec![], vec![])
        }
    }

    /// Check if GPU supports bfloat16 (Ampere+ = SM 8.0+)
    /// legacy lines 76-84
    pub fn gpu_supports_bf16(&self, gpu_idx: usize) -> bool {
        if let Some(device) = self.accelerator_devices.get(gpu_idx) {
            return device.supports_precision(TrainingPrecision::Bf16);
        }
        if gpu_idx >= self.compute_capabilities.len() {
            return false;
        }
        let (major, _minor) = self.compute_capabilities[gpu_idx];
        major >= 8
    }

    /// Check if GPU supports FP8 (Ada/Hopper/Blackwell = SM 8.9+)
    /// legacy lines 99-109
    pub fn gpu_supports_fp8(&self, gpu_idx: usize) -> bool {
        if let Some(device) = self.accelerator_devices.get(gpu_idx) {
            return device.supports_precision(TrainingPrecision::Fp8);
        }
        if gpu_idx >= self.compute_capabilities.len() {
            return false;
        }
        let (major, minor) = self.compute_capabilities[gpu_idx];
        (major > 8) || (major == 8 && minor >= 9)
    }
}

// ============================================================================
// DEVICE BENCHMARKING
// ============================================================================

/// Simple benchmarking to determine relative CPU vs GPU performance
/// legacy lines 18-58
pub struct DeviceBenchmark;

impl DeviceBenchmark {
    /// Measure time for a matrix multiplication
    /// legacy lines 22-42
    #[cfg(feature = "tch")]
    pub fn benchmark_matmul(device: Device, size: i64) -> f64 {
        let result = std::panic::catch_unwind(|| {
            let x = Tensor::randn(&[size, size], (Kind::Float, device));
            let y = Tensor::randn(&[size, size], (Kind::Float, device));

            // Synchronize if GPU
            if device.is_cuda() {
                Cuda::synchronize();
            }

            let start = Instant::now();
            let _z = x.matmul(&y);

            if device.is_cuda() {
                Cuda::synchronize();
            }

            start.elapsed().as_secs_f64()
        });

        result.unwrap_or(f64::INFINITY)
    }

    #[cfg(not(feature = "tch"))]
    pub fn benchmark_matmul(_device: &str, _size: i64) -> f64 {
        f64::INFINITY
    }

    /// Estimate CPU->GPU transfer latency
    /// legacy lines 44-58
    #[cfg(feature = "tch")]
    pub fn estimate_transfer_overhead(device: Device, size_mb: usize) -> f64 {
        if !device.is_cuda() {
            return 0.0;
        }

        let result = std::panic::catch_unwind(|| {
            let elements = (size_mb * 1024 * 1024) / 4; // float32
            let x = Tensor::randn(&[elements as i64], (Kind::Float, Device::Cpu));

            Cuda::synchronize();
            let start = Instant::now();
            let _y = x.to(device);
            Cuda::synchronize();

            start.elapsed().as_secs_f64()
        });

        result.unwrap_or(f64::INFINITY)
    }

    #[cfg(not(feature = "tch"))]
    pub fn estimate_transfer_overhead(_device: &str, _size_mb: usize) -> f64 {
        0.0
    }
}

// ============================================================================
// DEVICE SELECTION
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DevicePreference {
    Auto,
    Cpu,
    Gpu(usize), // GPU index
    AllGpus,
}

/// Select device based on request and availability
/// legacy lines 173-194
pub fn select_device(requested: DevicePreference) -> Vec<String> {
    let mut probe = HardwareProbe::new();
    let profile = probe.detect();
    let accelerator_devices = profile.accelerator_devices;
    let first_cuda = accelerator_devices
        .iter()
        .find(|device| device.backend == AcceleratorBackend::Cuda)
        .map(AcceleratorDevice::device_string);
    let first_accelerator = accelerator_devices
        .first()
        .map(AcceleratorDevice::device_string);

    match requested {
        DevicePreference::Cpu => vec!["cpu".to_string()],
        DevicePreference::Gpu(idx) => accelerator_devices
            .iter()
            .find(|device| device.id == idx)
            .map(AcceleratorDevice::device_string)
            .map(|device| vec![device])
            .unwrap_or_else(|| vec!["cpu".to_string()]),
        DevicePreference::AllGpus => {
            let devices = accelerator_devices
                .iter()
                .map(AcceleratorDevice::device_string)
                .collect::<Vec<_>>();
            if devices.is_empty() {
                vec!["cpu".to_string()]
            } else {
                devices
            }
        }
        DevicePreference::Auto => first_cuda
            .or(first_accelerator)
            .map(|device| vec![device])
            .unwrap_or_else(|| vec!["cpu".to_string()]),
    }
}

/// Select devices from a scheduler-owned workload assignment.
pub fn select_device_from_assignment(assignment: &ResolvedWorkloadAssignment) -> Vec<String> {
    match assignment.device_assignment.backend {
        BackendKind::NativeCpu
        | BackendKind::BurnCpu
        | BackendKind::CpuReference
        | BackendKind::LocalSurrogateFallback
        | BackendKind::Unavailable => vec!["cpu".to_string()],
        _ if !assignment.device_assignment.device_ids.is_empty() => assignment
            .device_assignment
            .device_ids
            .iter()
            .map(|id| {
                format!(
                    "{}:{}",
                    assignment.device_assignment.backend.as_device_prefix(),
                    id
                )
            })
            .collect(),
        _ => vec![assignment.device_assignment.device.clone()],
    }
}

trait BackendDevicePrefix {
    fn as_device_prefix(self) -> &'static str;
}

impl BackendDevicePrefix for BackendKind {
    fn as_device_prefix(self) -> &'static str {
        match self {
            BackendKind::NativeCuda | BackendKind::CudaKernel => "cuda",
            BackendKind::BurnWgpu => "wgpu",
            BackendKind::NativeTreeGpu => "gpu",
            BackendKind::NativeCpu
            | BackendKind::BurnCpu
            | BackendKind::NativeTreeCpu
            | BackendKind::CpuReference
            | BackendKind::LocalSurrogateFallback
            | BackendKind::ExternalRuntime
            | BackendKind::Unavailable => "cpu",
        }
    }
}

/// Get list of available GPU device strings
/// legacy lines 70-73
pub fn get_available_gpus() -> Vec<String> {
    let mut probe = HardwareProbe::new();
    probe
        .detect()
        .accelerator_devices
        .into_iter()
        .map(|device| device.device_string())
        .collect()
}

// ============================================================================
// TORCH BACKEND TUNING
// ============================================================================

/// Enable fast math on supported GPUs (TF32 for Ampere/Hopper)
/// legacy lines 130-141
#[cfg(feature = "tch")]
pub fn tune_torch_backend(device: Device) {
    // Note: tch-rs doesn't expose all backend tuning options that legacy PyTorch has
    // These would need to be set via environment variables or C++ FFI

    if device.is_cuda() {
        // Enable TF32 for Ampere/Hopper
        // In legacy: torch.backends.cuda.matmul.allow_tf32 = True
        // In Rust: Set via env var before program start
        env::set_var("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1");

        // Enable cuDNN benchmark mode
        // In legacy: torch.backends.cudnn.benchmark = True
        // In Rust: Set via env var
        env::set_var("TORCH_CUDNN_BENCHMARK", "1");

        debug!("Torch backend tuning enabled for CUDA");
    }
}

#[cfg(not(feature = "tch"))]
pub fn tune_torch_backend(_device: &str) {
    debug!("Torch backend tuning skipped (tch feature disabled)");
}

/// Enable flash attention / memory-efficient SDPA kernels
/// legacy lines 144-170
#[cfg(feature = "tch")]
pub fn enable_flash_attention() {
    // These are PyTorch 2.0+ features
    // tch-rs may not expose them directly yet
    // Set via environment variables
    env::set_var("TORCH_CUDNN_SDPA_ENABLED", "1");
    debug!("Flash attention enabled (if supported by PyTorch version)");
}

#[cfg(not(feature = "tch"))]
pub fn enable_flash_attention() {
    debug!("Flash attention skipped (tch feature disabled)");
}

// ============================================================================
// PARALLEL CONFIGURATION
// ============================================================================

/// Configure rayon thread pool to use all cores minus 1
/// REPLACES legacy's multiprocessing.cpu_count() - no GIL issues!
pub fn configure_rayon_threads(hardware: &HardwareInfo) {
    let threads = hardware.cpu_cores_usable.max(1);

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap_or_else(|e| {
            warn!("Failed to configure rayon threads: {}", e);
        });

    info!("Configured rayon with {} threads (all cores - 1)", threads);
}

/// Get number of parallel jobs for given hardware
/// REPLACES legacy's prefer_gpu_env_jobs (lines 197-204)
/// NO multiprocessing import needed - Rust has no GIL!
pub fn get_parallel_jobs(hardware: &HardwareInfo, requested: Option<usize>) -> usize {
    let max_jobs = hardware.cpu_cores_usable.max(1);

    match requested {
        Some(n) => n.min(max_jobs).max(1),
        None => max_jobs,
    }
}

// ============================================================================
// GPU DISTRIBUTION HELPERS
// ============================================================================

/// Distribute 1-based model ordinals across GPUs using round-robin.
/// HPC CRITICAL: Spread 8 models across 8 A6000 GPUs.
///
/// Model numbering in the public examples/tests is 1-based:
/// model 1 -> GPU 0, model 2 -> GPU 1, ..., model 9 -> GPU 0.
/// A zero ordinal is treated defensively as the first model.
pub fn distribute_gpu_assignment(model_idx: usize, hardware: &HardwareInfo) -> usize {
    if hardware.gpu_count == 0 {
        0
    } else {
        model_idx.saturating_sub(1) % hardware.gpu_count
    }
}

/// Get GPU device string for a model index
pub fn get_gpu_for_model(model_idx: usize, hardware: &HardwareInfo) -> String {
    let gpu_idx = distribute_gpu_assignment(model_idx, hardware);
    format!("cuda:{}", gpu_idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_detection() {
        let hw = HardwareInfo::detect();
        assert!(hw.cpu_cores > 0);
        assert!(hw.cpu_cores_usable >= 1);
        assert!(hw.cpu_cores_usable <= hw.cpu_cores);
        if hw.cpu_cores > 1 {
            assert!(hw.cpu_cores_usable < hw.cpu_cores);
        } else {
            assert_eq!(hw.cpu_cores_usable, 1);
        }
    }

    #[test]
    fn test_gpu_distribution() {
        let hw = HardwareInfo {
            cpu_cores: 250,
            cpu_cores_usable: 249,
            gpu_count: 8,
            gpu_names: vec![],
            gpu_memory_gb: vec![],
            compute_capabilities: vec![],
            accelerator_devices: vec![],
            os_name: "linux".to_string(),
        };

        // Test round-robin distribution
        assert_eq!(distribute_gpu_assignment(1, &hw), 0); // Model 1 → GPU 0
        assert_eq!(distribute_gpu_assignment(2, &hw), 1); // Model 2 → GPU 1
        assert_eq!(distribute_gpu_assignment(8, &hw), 7); // Model 8 → GPU 7
        assert_eq!(distribute_gpu_assignment(9, &hw), 0); // Model 9 → GPU 0 (wraps)
    }

    #[test]
    fn select_device_uses_scheduler_assignment() {
        let assignment = ResolvedWorkloadAssignment {
            workload: neoethos_core::WorkloadKind::DeepTraining,
            hardware_profile_id: "hardware-profile".to_string(),
            device_assignment: neoethos_core::DeviceAssignment {
                backend: BackendKind::NativeCuda,
                device: "cuda:all".to_string(),
                device_ids: vec![0, 2],
            },
            cpu_budget: neoethos_core::CpuBudget::new(8),
            gpu_budget: None,
            precision_policy: neoethos_core::PrecisionPolicy::from_precision(
                TrainingPrecision::Fp32,
            ),
            batch_size: 512,
            runtime_degraded_reason: None,
            notes: Vec::new(),
        };

        assert_eq!(
            select_device_from_assignment(&assignment),
            vec!["cuda:0", "cuda:2"]
        );
    }

    #[test]
    fn select_device_uses_cpu_for_degraded_assignment() {
        let assignment = ResolvedWorkloadAssignment {
            workload: neoethos_core::WorkloadKind::DeepTraining,
            hardware_profile_id: "hardware-profile".to_string(),
            device_assignment: neoethos_core::DeviceAssignment {
                backend: BackendKind::CpuReference,
                device: "cpu".to_string(),
                device_ids: Vec::new(),
            },
            cpu_budget: neoethos_core::CpuBudget::new(8),
            gpu_budget: None,
            precision_policy: neoethos_core::PrecisionPolicy::from_precision(
                TrainingPrecision::Fp32,
            ),
            batch_size: 512,
            runtime_degraded_reason: None,
            notes: Vec::new(),
        };

        assert_eq!(select_device_from_assignment(&assignment), vec!["cpu"]);
    }

    #[test]
    fn test_parallel_jobs() {
        let hw = HardwareInfo {
            cpu_cores: 250,
            cpu_cores_usable: 249,
            gpu_count: 8,
            gpu_names: vec![],
            gpu_memory_gb: vec![],
            compute_capabilities: vec![],
            accelerator_devices: vec![],
            os_name: "linux".to_string(),
        };

        assert_eq!(get_parallel_jobs(&hw, None), 249);
        assert_eq!(get_parallel_jobs(&hw, Some(100)), 100);
        assert_eq!(get_parallel_jobs(&hw, Some(300)), 249); // Capped at usable cores
        assert_eq!(get_parallel_jobs(&hw, Some(0)), 1); // Minimum 1
    }
}
