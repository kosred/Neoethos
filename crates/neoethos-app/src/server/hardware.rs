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
    /// Best-effort name. cTrader desktop reports this for diagnostics;
    /// we ship a placeholder until a real GPU probe lands (the existing
    /// `forex-models::hardware` module knows how to do CUDA discovery
    /// but the dependency edge isn't wired yet — out of scope for the
    /// HTTP-server work).
    pub name: String,
    /// Whether at least one GPU is detected. Mirrors the boolean the
    /// existing `Hardware` egui panel renders.
    pub available: bool,
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
        gpu: GpuDto {
            // Placeholder until a real probe lands. The Flutter side
            // shows this as a hint that GPU detection isn't wired yet,
            // not as a real model claim.
            name: "GPU probe pending".to_string(),
            available: false,
        },
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
        },
    }
}
