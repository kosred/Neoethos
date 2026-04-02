//! HPC-specific optimizations for Hyperstack N3-RTX-A6000x8 instances.
//!
//! These optimizations ONLY activate when the specific hardware is detected:
//! - 8× RTX A6000 (48GB each = 384GB VRAM)
//! - 252 CPU cores (AMD EPYC Milan)
//! - 464GB RAM
//!
//! The topology is expected to be:
//! - Socket 0 (NUMA 0): Cores 0-125, RAM 232GB, GPUs 0-3
//! - Socket 1 (NUMA 1): Cores 126-251, RAM 232GB, GPUs 4-7
//! - NVLink: 112GB/s between GPU pairs (0↔1, 2↔3, 4↔5, 6↔7)

use anyhow::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{info, warn};

static HPC_MODE_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Hardware profile for Hyperstack N3 detection
///
/// N3-RTX-A6000x8: 252 physical cores + SMT = 504 logical threads
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperstackN3Profile {
    pub gpu_count: usize,
    pub gpu_min_vram_gb: f64,
    pub cpu_logical_threads: usize, // 504 with SMT
    pub cpu_physical_cores: usize,  // 252 physical
    pub total_ram_gb: f64,
    pub is_numa_dual_socket: bool,
}

impl Default for HyperstackN3Profile {
    fn default() -> Self {
        Self {
            gpu_count: 8,
            gpu_min_vram_gb: 40.0, // Expect at least 40GB per GPU (A6000 has 48GB)
            cpu_logical_threads: 500, // Allow slight variation from 504
            cpu_physical_cores: 250, // Allow slight variation from 252
            total_ram_gb: 450.0,   // Allow slight variation from 464GB
            is_numa_dual_socket: true,
        }
    }
}

/// Detect if running on Hyperstack N3-RTX-A6000x8 hardware
pub fn detect_hyperstack_n3() -> bool {
    // Check if already detected
    if HPC_MODE_ACTIVE.load(Ordering::Relaxed) {
        return true;
    }

    let profile = HyperstackN3Profile::default();

    // Get GPU info
    let gpu_count = tch::Cuda::device_count() as usize;
    if gpu_count < profile.gpu_count {
        return false;
    }

    // Check GPU memory
    let mut min_vram_gb = f64::INFINITY;
    for i in 0..gpu_count {
        if let Ok(props) = tch::Cuda::device_properties(i as i64) {
            let vram_gb = props.total_memory as f64 / (1024.0 * 1024.0 * 1024.0);
            min_vram_gb = min_vram_gb.min(vram_gb);
        }
    }

    if min_vram_gb < profile.gpu_min_vram_gb {
        return false;
    }

    // Check CPU threads (with SMT = 504 logical threads on 252 physical cores)
    let cpu_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);

    if cpu_threads < profile.cpu_logical_threads {
        return false;
    }

    // Check RAM
    let total_ram_gb = detect_total_ram_gb();
    if total_ram_gb < profile.total_ram_gb {
        return false;
    }

    // All checks passed - we're on Hyperstack N3!
    HPC_MODE_ACTIVE.store(true, Ordering::Relaxed);

    info!(
        "🚀 Hyperstack N3 HPC Mode ACTIVATED: {} GPUs @ {:.1}GB+ VRAM, {} logical threads ({} physical cores), {:.1}GB RAM",
        gpu_count,
        min_vram_gb,
        cpu_threads,
        cpu_threads / 2,
        total_ram_gb
    );

    true
}

/// Check if HPC mode is active
pub fn is_hpc_mode() -> bool {
    HPC_MODE_ACTIVE.load(Ordering::Relaxed)
}

/// Force enable HPC mode (for testing)
pub fn force_hpc_mode(enable: bool) {
    HPC_MODE_ACTIVE.store(enable, Ordering::Relaxed);
    if enable {
        info!("HPC mode FORCE ENABLED");
    }
}

/// Detect total system RAM in GB
fn detect_total_ram_gb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    // Format: "MemTotal:       487908384 kB"
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<f64>() {
                            return kb / (1024.0 * 1024.0); // Convert KB to GB
                        }
                    }
                }
            }
        }
    }

    // Fallback: use sysinfo if available
    0.0
}

/// Get GPU-CPU affinity mapping for NUMA optimization
///
/// With 252 physical cores + SMT = 504 logical threads
/// Socket 0 (NUMA 0): Physical 0-125 → Logical 0-125 (primary) + 252-377 (SMT)
/// Socket 1 (NUMA 1): Physical 126-251 → Logical 126-251 (primary) + 378-503 (SMT)
pub fn get_gpu_cpu_affinity(gpu_id: i64) -> Vec<usize> {
    if !is_hpc_mode() {
        return Vec::new();
    }

    // Hyperstack N3 topology with SMT (504 logical threads):
    // Socket 0: GPUs 0-3 → Logical 0-125 (primary) + 252-377 (SMT) = 252 threads
    // Socket 1: GPUs 4-7 → Logical 126-251 (primary) + 378-503 (SMT) = 252 threads
    match gpu_id {
        0..=3 => {
            let mut cores: Vec<usize> = (0..=125).collect(); // Primary threads
            cores.extend(252..=377); // SMT threads
            cores
        }
        4..=7 => {
            let mut cores: Vec<usize> = (126..=251).collect(); // Primary threads
            cores.extend(378..=503); // SMT threads
            cores
        }
        _ => Vec::new(),
    }
}

/// Get NVLink peer status between GPUs
pub fn is_nvlink_pair(gpu_a: i64, gpu_b: i64) -> bool {
    if !is_hpc_mode() {
        return false;
    }

    // NVLink topology on Hyperstack N3:
    // Pairs: (0,1), (2,3), (4,5), (6,7)
    let pairs = [(0, 1), (2, 3), (4, 5), (6, 7)];

    pairs
        .iter()
        .any(|(a, b)| (gpu_a == *a && gpu_b == *b) || (gpu_a == *b && gpu_b == *a))
}

/// Get optimal CPU cores for CPU-bound validation work
///
/// With SMT: Use primary threads for compute, SMT threads for I/O
/// Reserve 24 logical threads (12 physical) for GPU coordination
pub fn get_validation_cpu_cores() -> Vec<usize> {
    if !is_hpc_mode() {
        return Vec::new();
    }

    // Use 480 logical threads (240 physical) for validation
    // Reserve 24 logical threads (12 physical) for GPU coordination
    let mut cores = Vec::with_capacity(480);

    // Socket 0: Primary threads 0-119 (leave 120-125 for GPU 0-3 coordination)
    cores.extend(0..120);
    // Socket 0: SMT threads 252-371 (leave 372-377 for GPU 0-3 coordination)
    cores.extend(252..372);

    // Socket 1: Primary threads 126-245 (leave 246-251 for GPU 4-7 coordination)
    cores.extend(126..246);
    // Socket 1: SMT threads 378-497 (leave 498-503 for GPU 4-7 coordination)
    cores.extend(378..497);

    cores
}

/// Configure thread affinity for current thread (Linux only)
#[cfg(target_os = "linux")]
pub fn set_thread_affinity(cores: &[usize]) -> Result<()> {
    if cores.is_empty() {
        return Ok(());
    }

    use libc::{CPU_SET, CPU_ZERO, cpu_set_t, sched_setaffinity};
    use std::mem;

    unsafe {
        let mut cpuset: cpu_set_t = mem::zeroed();
        CPU_ZERO(&mut cpuset);

        for &core in cores {
            if core < 1024 {
                // CPU_SETSIZE is typically 1024
                CPU_SET(core, &mut cpuset);
            }
        }

        let pid = 0; // Current thread
        let size = mem::size_of::<cpu_set_t>();

        let result = sched_setaffinity(pid, size, &cpuset);
        if result != 0 {
            return Err(anyhow::anyhow!("sched_setaffinity failed"));
        }
    }

    Ok(())
}

#[cfg(not(target_os = "linux"))]
pub fn set_thread_affinity(_cores: &[usize]) -> Result<()> {
    // Thread affinity only supported on Linux
    Ok(())
}

/// Get optimal chunk size for GPU evaluation based on HPC mode
pub fn get_optimal_chunk_size() -> usize {
    if is_hpc_mode() {
        // Larger chunks for A6000 with 48GB VRAM
        8192
    } else {
        // Default chunk size
        2048
    }
}

/// Get optimal population size for HPC mode
pub fn get_optimal_population() -> usize {
    if is_hpc_mode() {
        // Much larger population for 8×A6000
        500_000
    } else {
        24_000
    }
}

/// Print HPC configuration summary
pub fn print_hpc_config() {
    if !is_hpc_mode() {
        return;
    }

    info!("═══════════════════════════════════════════════════════════════");
    info!("🚀 HYPERSTACK N3 HPC CONFIGURATION (SMT Enabled)");
    info!("═══════════════════════════════════════════════════════════════");
    info!("Hardware: 252 physical cores × 2 = 504 logical threads");
    info!("GPU Topology:");
    info!("  Socket 0 (NUMA 0):");
    info!("    GPUs: 0-3");
    info!("    Primary threads: 0-125");
    info!("    SMT threads: 252-377");
    info!("  Socket 1 (NUMA 1):");
    info!("    GPUs: 4-7");
    info!("    Primary threads: 126-251");
    info!("    SMT threads: 378-503");
    info!("  NVLink pairs: (0,1), (2,3), (4,5), (6,7)");
    info!("");
    info!("Optimal Settings:");
    info!("  Population: 500,000 strategies");
    info!("  Chunk size: 8,192");
    info!("  CPU validation threads: 480 (24 reserved for GPU coord)");
    info!("  Thread strategy: Primary threads = compute, SMT = I/O");
    info!("═══════════════════════════════════════════════════════════════");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_cpu_affinity() {
        force_hpc_mode(true);

        let socket0 = get_gpu_cpu_affinity(0);
        assert!(socket0.contains(&0));
        assert!(socket0.contains(&125));
        assert!(!socket0.contains(&126));

        let socket1 = get_gpu_cpu_affinity(4);
        assert!(socket1.contains(&126));
        assert!(socket1.contains(&251));
        assert!(!socket1.contains(&125));

        force_hpc_mode(false);
    }

    #[test]
    fn test_nvlink_pairs() {
        force_hpc_mode(true);

        assert!(is_nvlink_pair(0, 1));
        assert!(is_nvlink_pair(1, 0));
        assert!(is_nvlink_pair(2, 3));
        assert!(is_nvlink_pair(6, 7));
        assert!(!is_nvlink_pair(0, 2));
        assert!(!is_nvlink_pair(3, 4));

        force_hpc_mode(false);
    }
}
