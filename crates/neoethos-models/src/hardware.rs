// Hardware / accelerator capability detection for the models crate.
//
// SCOPE (2026-08-09, batch D4): this module reports what the accelerators can
// do. It does NOT decide which device a workload runs on.
//
// It used to do both, badly. A second, disconnected device-selection system
// lived here — `DevicePreference`, `select_device`, `select_device_from_assignment`,
// `get_available_gpus`, `distribute_gpu_assignment`, `get_gpu_for_model`,
// `configure_rayon_threads`, `get_parallel_jobs`, plus a `DeviceBenchmark`
// whose two methods returned `f64::INFINITY` / `0.0` in every build that has
// ever shipped. None of it had a single production caller: every reference
// resolved into this file's own `#[cfg(test)]` block. The real device plan
// comes from `neoethos_core::HardwareExecutionPlan` and the real thread budget
// from `tree_models::config::cpu_threads_hint_for`. Two device selectors, one
// connected; the disconnected one is gone.
//
// The `tch` (libtorch) probe went with it — the `tch` Cargo feature was
// enabled by no crate, script or CI job, so `detect_gpus`' CUDA arm was
// cfg-deleted from every build and the surviving arm returned "0 GPUs"
// unconditionally. `HardwareProbe` is now the single source, which is what
// every shipped binary was already using.
//
// The one production consumer of this module is `burn_models.rs:2054`:
// `HardwareInfo::detect()` -> `gpu_supports_bf16()`.

use std::env;
use tracing::info;

use neoethos_core::TrainingPrecision;
use neoethos_core::system::HardwareProbe;

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
    pub accelerator_devices: Vec<neoethos_core::AcceleratorDevice>,
    pub os_name: String,
}

impl HardwareInfo {
    /// Auto-detect CPUs, accelerators and OS via the core `HardwareProbe`.
    ///
    /// `HardwareProbe` is the ONLY accelerator source. When it reports no
    /// devices, this reports no devices — there is no secondary probe and no
    /// silent fallback. (There used to be a `tch`-gated CUDA enumeration here;
    /// the feature was never enabled in any build, so its "fallback" was a
    /// hardcoded zero pretending to be a measurement.)
    pub fn detect() -> Self {
        let cpu_cores = num_cpus::get();
        let cpu_cores_usable = cpu_cores.saturating_sub(1).max(1); // Reserve 1 for OS

        let mut core_probe = HardwareProbe::new();
        let core_profile = core_probe.detect();
        let accelerator_devices = core_profile.accelerator_devices;

        let gpu_names = accelerator_devices
            .iter()
            .map(|device| device.name.clone())
            .collect::<Vec<_>>();
        let gpu_memory_gb = accelerator_devices
            .iter()
            .map(|device| device.memory_gb)
            .collect::<Vec<_>>();
        let compute_capabilities = accelerator_devices
            .iter()
            .map(|device| device.compute_capability.unwrap_or((0, 0)))
            .collect::<Vec<_>>();
        let gpu_count = accelerator_devices.len();

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

    /// Check if GPU supports bfloat16 (Ampere+ = SM 8.0+).
    ///
    /// Production consumer: `burn_models.rs` picks the training dtype from it.
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

    /// Check if GPU supports FP8 (Ada/Hopper/Blackwell = SM 8.9+).
    ///
    /// No consumer yet — the Burn training lane is bf16/fp32 only. Kept
    /// alongside `gpu_supports_bf16` because both read the same probe field
    /// and splitting them would leave a half-answered capability question.
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
    fn accelerator_vectors_stay_index_aligned() {
        // Every consumer indexes gpu_names / gpu_memory_gb / compute_capabilities
        // by the same accelerator ordinal it passes to gpu_supports_bf16.
        let hw = HardwareInfo::detect();
        assert_eq!(hw.gpu_count, hw.accelerator_devices.len());
        assert_eq!(hw.gpu_names.len(), hw.gpu_count);
        assert_eq!(hw.gpu_memory_gb.len(), hw.gpu_count);
        assert_eq!(hw.compute_capabilities.len(), hw.gpu_count);
    }

    #[test]
    fn precision_queries_are_false_beyond_the_device_list() {
        let hw = HardwareInfo::detect();
        let past_end = hw.gpu_count + 8;
        assert!(!hw.gpu_supports_bf16(past_end));
        assert!(!hw.gpu_supports_fp8(past_end));
    }
}
