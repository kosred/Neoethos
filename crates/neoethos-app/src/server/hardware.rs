//! `/hardware` — CPU / RAM / GPU snapshot.
//!
//! The Flutter Hardware tab consumes this to render the live system
//! state. Static info (CPU model, core count, total RAM) is captured
//! once at app start; the dynamic numbers (CPU load, RAM used) get
//! refreshed on each request.

use axum::Json;
use axum::extract::State;
use sysinfo::System;

use super::state::AppApiState;

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareDto {
    pub cpu: CpuDto,
    pub ram: RamDto,
    pub gpu: GpuDto,
    /// Whether THIS BINARY was compiled with a GPU lane at all.
    ///
    /// The GPU evaluator lives behind the `gpu-*` cargo features, so a
    /// default build has no GPU code in it — on a machine with a card that
    /// binary runs CPU-only and, before this field existed, said nothing
    /// about it. `gpu.available` answers "is there a card?"; this answers
    /// "can this build use one?". Both must be true for GPU work to happen,
    /// and the UI now says which one is missing instead of leaving the
    /// operator to wonder why a rented card sits idle.
    pub gpu_support: GpuSupportDto,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuSupportDto {
    /// True when a `gpu-*` feature was enabled for this build.
    pub compiled: bool,
    /// Which lane was compiled in: `"cuda"`, `"vulkan"`, `"rocm"`, or
    /// `"none"`. Mirrors the cargo feature actually selected.
    pub backend: String,
    /// Operator-facing one-liner, safe to render verbatim.
    pub detail: String,
}

/// Resolve the compiled-in GPU lane from the cargo features. Kept here (not
/// in the probe) because it is a property of the BINARY, not the machine.
fn gpu_support_dto() -> GpuSupportDto {
    // The vendor features are mutually exclusive in practice; check the most
    // specific first so a multi-feature build reports its primary lane.
    #[cfg(feature = "gpu-nvidia")]
    {
        return GpuSupportDto {
            compiled: true,
            backend: "cuda".to_string(),
            detail: "This build includes the CUDA GPU lane.".to_string(),
        };
    }
    #[cfg(all(feature = "gpu-rocm", not(feature = "gpu-nvidia")))]
    {
        return GpuSupportDto {
            compiled: true,
            backend: "rocm".to_string(),
            detail: "This build includes the ROCm GPU lane.".to_string(),
        };
    }
    #[cfg(all(
        feature = "gpu-vulkan",
        not(feature = "gpu-nvidia"),
        not(feature = "gpu-rocm")
    ))]
    {
        return GpuSupportDto {
            compiled: true,
            backend: "vulkan".to_string(),
            detail: "This build includes the Vulkan/wgpu GPU lane.".to_string(),
        };
    }
    #[cfg(not(any(feature = "gpu-nvidia", feature = "gpu-rocm", feature = "gpu-vulkan")))]
    {
        GpuSupportDto {
            compiled: false,
            backend: "none".to_string(),
            detail: "This build is CPU-only — no GPU lane was compiled in. Even \
                     with a card installed, discovery and training will run on \
                     the CPU. Rebuild with a gpu feature (e.g. \
                     `--features gpu-nvidia`) to use it."
                .to_string(),
        }
    }
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CpuDto {
    pub model: String,
    pub cores_logical: usize,
    pub cores_physical: usize,
    /// 0.0–1.0; average across all logical cores.
    pub load_avg: f32,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RamDto {
    pub total_mb: u64,
    pub used_mb: u64,
    pub available_mb: u64,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuDto {
    /// Best-effort name. The wgpu/Vulkan-backed enumerator that lists
    /// every adapter on the system is tracked separately (#189); until
    /// then we infer iGPU presence from the CPU model string (#188).
    pub name: String,
    /// Whether at least one GPU is detected. Rendered by the Flutter
    /// hardware screen.
    pub available: bool,
    /// One of `"none"`, `"integrated"`, `"discrete"`, `"unknown"`. The
    /// hardware screen uses this to surface a more honest label than
    /// the binary `available`. Defaults to `"unknown"` whenever the
    /// inference is ambiguous — never lies.
    pub kind: String,
}

pub async fn hardware(State(_state): State<AppApiState>) -> Json<HardwareDto> {
    // CPU-load gotcha: `sysinfo` calculates per-CPU usage as the delta
    // between two `refresh_cpu_usage()` calls. A single refresh returns
    // 0.0 (no baseline) or 100.0 (uninitialised counter on Windows). We
    // therefore refresh, sleep through `MINIMUM_CPU_UPDATE_INTERVAL`
    // (~200ms), and refresh again. The sleep is in a blocking task
    // (NOT directly in async) so the tokio reactor stays responsive
    // — `std::thread::sleep` in an async handler would stall every
    // other route for 200ms.
    let dto = tokio::task::spawn_blocking(probe_hardware_blocking)
        .await
        .unwrap_or_else(|e| {
            tracing::error!(
                target: "neoethos_app::server::hardware",
                error = %e,
                "hardware probe task panicked"
            );
            empty_hardware_dto()
        });
    Json(dto)
}

fn probe_hardware_blocking() -> HardwareDto {
    let mut sys = System::new();
    sys.refresh_cpu_usage();
    std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
    sys.refresh_cpu_usage();
    sys.refresh_memory();

    let cpus = sys.cpus();
    let cores_logical = cpus.len();
    let cores_physical = System::physical_core_count().unwrap_or(cores_logical);
    let model = cpus
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let load_avg = if cpus.is_empty() {
        0.0
    } else {
        cpus.iter().map(|c| c.cpu_usage()).sum::<f32>() / cpus.len() as f32 / 100.0
    };

    let total_kb = sys.total_memory();
    let used_kb = sys.used_memory();
    let available_kb = sys.available_memory();

    let gpu = infer_gpu_from_cpu_model(&model);
    HardwareDto {
        cpu: CpuDto {
            model,
            cores_logical,
            cores_physical,
            load_avg,
        },
        ram: RamDto {
            // sysinfo returns bytes since 0.30. Convert to MB (1024² scale).
            total_mb: total_kb / (1024 * 1024),
            used_mb: used_kb / (1024 * 1024),
            available_mb: available_kb / (1024 * 1024),
        },
        gpu,
        gpu_support: gpu_support_dto(),
    }
}

/// #188: best-effort iGPU inference from the CPU model string. Until
/// the real wgpu/Vulkan probe lands (#189), this catches the common
/// case where a Ryzen-U / Intel-Core part has an integrated GPU that
/// `sysinfo` doesn't surface and the old `available: false` placeholder
/// flat-out lied about. Returns `kind = "unknown"` for anything we
/// can't classify with high confidence so the UI never makes up a
/// claim about the user's hardware.
fn infer_gpu_from_cpu_model(cpu_model: &str) -> GpuDto {
    let model_lower = cpu_model.to_lowercase();

    // Ryzen mobile parts (U / H / HS / HX / G suffix) carry Radeon
    // integrated graphics. Desktop "F" parts (Intel) have iGPU
    // DISABLED; everything else in the Intel Core line has one.
    let is_amd_apu = model_lower.contains("ryzen")
        && (model_lower.ends_with('u')
            || model_lower.contains(" u ")
            || model_lower.contains(" h ")
            || model_lower.contains("hs")
            || model_lower.contains("hx")
            || model_lower.contains(" g ")
            || model_lower.ends_with('g'));
    let is_intel_core_with_igpu = model_lower.contains("intel")
        && (model_lower.contains("core") || model_lower.contains("xeon"))
        && !model_lower.contains("-f ")
        && !model_lower.ends_with('f')
        && !model_lower.contains("xeon w");

    if is_amd_apu {
        GpuDto {
            name: format!("Integrated Radeon (inferred from {cpu_model})"),
            available: true,
            kind: "integrated".to_string(),
        }
    } else if is_intel_core_with_igpu {
        GpuDto {
            name: format!("Integrated Intel Graphics (inferred from {cpu_model})"),
            available: true,
            kind: "integrated".to_string(),
        }
    } else {
        GpuDto {
            name: "GPU probe pending (no integrated GPU inferred)".to_string(),
            available: false,
            kind: "unknown".to_string(),
        }
    }
}

fn empty_hardware_dto() -> HardwareDto {
    HardwareDto {
        cpu: CpuDto {
            model: "unknown".to_string(),
            cores_logical: 0,
            cores_physical: 0,
            load_avg: 0.0,
        },
        ram: RamDto {
            total_mb: 0,
            used_mb: 0,
            available_mb: 0,
        },
        gpu: GpuDto {
            name: "probe failed".to_string(),
            available: false,
            kind: "unknown".to_string(),
        },
        // The compiled-in lane is known even when the machine probe fails —
        // it is a property of the binary, not of the hardware.
        gpu_support: gpu_support_dto(),
    }
}
