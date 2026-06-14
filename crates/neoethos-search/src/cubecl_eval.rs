use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
#[cfg(feature = "gpu-cuda")]
use cubecl::cuda::{CudaDevice, CudaRuntime};
#[cfg(all(feature = "gpu-vulkan", not(feature = "gpu-cuda")))]
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use cubecl::prelude::*;
use half::bf16;
use ndarray::ArrayView2;
use neoethos_core::TrainingPrecision;

use crate::eval::{BacktestSettings, SmcRow};

const SMC_WIDTH: usize = 11;
const BACKTEST_CORE_METRIC_WIDTH: usize = 7;
// FTMO prop-firm observables emitted per gene by `backtest_population_kernel`
// into the `ftmo_out` array, so the host can apply FTMO rules on the GPU result
// instead of re-running a CPU `simulate_trades_core` + `compute_prop_firm_risk_summary`.
// Layout per gene (matches validation.rs::compute_prop_firm_risk_summary):
//   [0] net_return_pct          = net_profit / initial_equity
//   [1] max_daily_loss_pct      = max_day(|day_pnl| if day_pnl<0) / initial_equity
//   [2] max_overall_drawdown_pct = peak-to-trough DD of the END-OF-DAY equity curve
//   [3] largest_profit_share    = largest_positive_day / sum_of_positive_days (0 if none)
//   [4] max_trades_per_day      = max day trade count (as f32)
//   [5] trading_days            = count of days with >=1 trade (as f32)
const FTMO_WIDTH: usize = 6;

// ─── F-CORE3 consolidation — CUDA env-var registry ──────────────────
//
// **2026-05-25**: the 7 inline `std::env::var(...)` reads previously
// scattered across `requested_eval_precision`, `cuda_eval_signal_kernel_enabled`,
// `cuda_eval_backtest_kernel_enabled`, `signal_kernel_units`,
// `backtest_kernel_units`, and `cuda_device_id` now route through this
// typed registry. Same canonical pattern as
// `crates/neoethos-app/src/app_services/env_overrides.rs` and
// `crates/neoethos-search/src/genetic/runtime_overrides.rs`.
//
// The `CudaEnvKnobs` struct is built once on first access via
// `cuda_env_knobs()` (lazy OnceLock) so the env-var reads happen at
// most once per process. Mirrors the audit-baseline `HardwareRuntimeOverrides`
// shape.
//
// Knobs covered (env-var name → typed field):
// - `NEOETHOS_BOT_SEARCH_EVAL_PRECISION` / `NEOETHOS_BOT_TRAIN_PRECISION` /
//   `FOREX_TRAIN_PRECISION` → `requested_precision: TrainingPrecision`
// - `NEOETHOS_BOT_SEARCH_EVAL_CUDA_KERNEL` → `eval_kernel_enabled: bool`
// - `NEOETHOS_BOT_SEARCH_BACKTEST_CUDA_KERNEL` → `backtest_kernel_enabled: bool`
// - `NEOETHOS_BOT_SEARCH_EVAL_KERNEL_UNITS` → `eval_kernel_units_override: Option<u32>`
// - `NEOETHOS_BOT_SEARCH_BACKTEST_KERNEL_UNITS` → `backtest_kernel_units_override: Option<u32>`
// - `NEOETHOS_BOT_SEARCH_EVAL_CUDA_DEVICE` → `cuda_device_id: usize`
//
// Vulkan note: the wgpu-vulkan backend is wired via feature
// aggregation (`vulkan` cargo feature). It doesn't read any of these
// env vars — the cubecl runtime selects Vulkan at compile time when
// the `vulkan` feature is on. No env-knob registry needed for Vulkan
// today; if one becomes necessary it'd live in a sibling
// `wgpu_eval.rs` file with the same typed-registry pattern.

#[derive(Debug, Clone, Copy)]
struct CudaEnvKnobs {
    requested_precision: TrainingPrecision,
    eval_kernel_enabled: bool,
    backtest_kernel_enabled: bool,
    eval_kernel_units_override: Option<u32>,
    backtest_kernel_units_override: Option<u32>,
    // Read only by the `gpu-cuda` device selector; unused on the wgpu/Vulkan
    // path (which uses `WgpuDevice::DefaultDevice`).
    #[allow(dead_code)]
    cuda_device_id: usize,
}

impl CudaEnvKnobs {
    fn from_env() -> Self {
        Self {
            requested_precision: read_requested_precision_from_env(),
            eval_kernel_enabled: read_kernel_enabled_from_env(
                "NEOETHOS_BOT_SEARCH_EVAL_CUDA_KERNEL",
            ),
            backtest_kernel_enabled: read_kernel_enabled_from_env(
                "NEOETHOS_BOT_SEARCH_BACKTEST_CUDA_KERNEL",
            ),
            eval_kernel_units_override: read_kernel_units_from_env(
                "NEOETHOS_BOT_SEARCH_EVAL_KERNEL_UNITS",
            ),
            // Backtest units fall back to eval units when the explicit
            // backtest knob is unset — preserves the original semantics
            // (`signal_kernel_units` and `backtest_kernel_units` both
            // honoured EVAL_KERNEL_UNITS as the umbrella default).
            backtest_kernel_units_override: read_kernel_units_from_env(
                "NEOETHOS_BOT_SEARCH_BACKTEST_KERNEL_UNITS",
            )
            .or_else(|| read_kernel_units_from_env("NEOETHOS_BOT_SEARCH_EVAL_KERNEL_UNITS")),
            cuda_device_id: read_cuda_device_id_from_env(),
        }
    }
}

static CUDA_ENV_KNOBS: OnceLock<CudaEnvKnobs> = OnceLock::new();

fn cuda_env_knobs() -> CudaEnvKnobs {
    *CUDA_ENV_KNOBS.get_or_init(CudaEnvKnobs::from_env)
}

// ─── GPU per-call timing instrumentation (NEOETHOS_GPU_TIMING) ───────
//
// 2026-06-10: the hybrid splitter measured the A6000 at ~74 genes/s vs the CPU
// lane's ~77 000 genes/s — a ~1000× gap that is per-LAUNCH overhead, NOT inherent
// GPU slowness (the kernel itself is correct + parity-proven). To SEE where the
// milliseconds go inside one `try_evaluate_population_cuda` call, set
// `NEOETHOS_GPU_TIMING=1` and this module emits a `tracing::info!` breakdown:
// n_genes, n_samples, total elapsed, and the split between client-get, host
// data-prep, device UPLOAD (`create_from_slice` / `empty`), KERNEL launch, and
// READBACK (`read_one_unchecked`).
//
// PARITY: this module is a pure side-effect. It only accumulates `Duration`s into
// a thread-local; it never touches a kernel input, an output buffer, or a launch
// dimension. When `NEOETHOS_GPU_TIMING` is unset, the cached `enabled()` flag is
// `false` and every phase closure runs the wrapped work WITHOUT even reading the
// clock, so the production path is byte-identical and ~free.
mod gpu_timing {
    use std::cell::RefCell;
    use std::sync::OnceLock;
    use std::time::{Duration, Instant};

    /// Cached once: is `NEOETHOS_GPU_TIMING` set? Checked at most once per process.
    fn enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var("NEOETHOS_GPU_TIMING").is_ok())
    }

    /// The phases a population eval splits into. `upload` / `kernel` / `readback`
    /// accumulate across EVERY gene-chunk + gene-batch inside the call (the inner
    /// `launch_signal_kernel` / `launch_backtest_kernel` add into the live frame).
    #[derive(Default, Clone, Copy)]
    pub struct Phases {
        pub client_get: Duration,
        pub host_prep: Duration,
        pub upload: Duration,
        pub kernel: Duration,
        pub readback: Duration,
    }

    thread_local! {
        /// The live accumulation frame for the current `try_evaluate_population_cuda`
        /// call on this thread. `None` outside a measured call (also the always-state
        /// when timing is disabled, so the inner adders are no-ops).
        static FRAME: RefCell<Option<Phases>> = const { RefCell::new(None) };
    }

    /// Begin a measurement frame for the current call. No-op when timing is off.
    pub fn begin() {
        if !enabled() {
            return;
        }
        FRAME.with(|f| *f.borrow_mut() = Some(Phases::default()));
    }

    /// End the frame and return the accumulated phases (the residual top-level time
    /// the caller can attribute to "other"). `None` when timing is off.
    pub fn end() -> Option<Phases> {
        if !enabled() {
            return None;
        }
        FRAME.with(|f| f.borrow_mut().take())
    }

    /// Add `d` to one phase of the live frame. Cheap no-op when no frame is active.
    fn add(select: impl Fn(&mut Phases) -> &mut Duration, d: Duration) {
        FRAME.with(|f| {
            if let Some(frame) = f.borrow_mut().as_mut() {
                *select(frame) += d;
            }
        });
    }

    /// Run `body`, attributing its elapsed time to the UPLOAD phase. When timing is
    /// off, runs `body` WITHOUT reading the clock (the closure is fully inlined and
    /// the result is byte-identical).
    pub fn upload<T>(body: impl FnOnce() -> T) -> T {
        time(|p| &mut p.upload, body)
    }

    /// Run `body`, attributing its elapsed time to the KERNEL phase.
    pub fn kernel<T>(body: impl FnOnce() -> T) -> T {
        time(|p| &mut p.kernel, body)
    }

    /// Run `body`, attributing its elapsed time to the READBACK phase.
    pub fn readback<T>(body: impl FnOnce() -> T) -> T {
        time(|p| &mut p.readback, body)
    }

    /// Run `body`, attributing its elapsed time to the CLIENT-GET phase.
    pub fn client_get<T>(body: impl FnOnce() -> T) -> T {
        time(|p| &mut p.client_get, body)
    }

    /// Directly fold an already-measured `Duration` into the HOST-PREP phase. Used
    /// for the constant per-sample conversions, which are measured with a plain
    /// `Instant` window at the call site rather than a closure. No-op when off.
    pub fn add_host_prep(d: Duration) {
        if !enabled() {
            return;
        }
        add(|p| &mut p.host_prep, d);
    }

    /// Common timing core. The `enabled()` branch is the only check on the hot path
    /// when timing is off — no `Instant::now()`, no thread-local borrow.
    fn time<T>(select: impl Fn(&mut Phases) -> &mut Duration, body: impl FnOnce() -> T) -> T {
        if !enabled() {
            return body();
        }
        let start = Instant::now();
        let out = body();
        add(select, start.elapsed());
        out
    }
}

fn read_requested_precision_from_env() -> TrainingPrecision {
    [
        "NEOETHOS_BOT_SEARCH_EVAL_PRECISION",
        "NEOETHOS_BOT_TRAIN_PRECISION",
        "FOREX_TRAIN_PRECISION",
    ]
    .iter()
    .find_map(|key| std::env::var(key).ok())
    .and_then(|value| parse_training_precision(&value))
    .unwrap_or(TrainingPrecision::Fp32)
}

fn read_kernel_enabled_from_env(name: &str) -> bool {
    !matches!(
        std::env::var(name)
            .ok()
            .map(|value| value.trim().to_ascii_lowercase()),
        Some(value) if matches!(value.as_str(), "0" | "false" | "off" | "disable" | "disabled")
    )
}

fn read_kernel_units_from_env(name: &str) -> Option<u32> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value > 0)
}

fn read_cuda_device_id_from_env() -> usize {
    match std::env::var("NEOETHOS_BOT_SEARCH_EVAL_CUDA_DEVICE") {
        // Not set: pick device 0 silently — the canonical default.
        Err(_) => 0,
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(value) => value,
            Err(_) => {
                // The user explicitly set the env var but it did not
                // parse as a usize ("auto", "all", "GPU0" — typos like
                // these used to silently fall back to device 0,
                // running the search on the wrong card without telling
                // anyone. Now we shout, then default.
                tracing::warn!(
                    target: "neoethos_search::gpu",
                    raw = %raw,
                    "NEOETHOS_BOT_SEARCH_EVAL_CUDA_DEVICE is set but not a valid \
                     non-negative integer; falling back to device 0."
                );
                0
            }
        },
    }
}

/// Create a `ComputeClient` for the active GPU runtime. The concrete runtime is
/// chosen at COMPILE time by the GPU feature flag — CUDA under `gpu-cuda`,
/// wgpu/Vulkan under `gpu-vulkan` — so every downstream kernel launch stays
/// generic over `R: Runtime` and runs unchanged on whichever backend was built.
/// (When both features are on, CUDA wins.)
#[cfg(feature = "gpu-cuda")]
fn create_gpu_client(device_override: Option<usize>) -> Result<ComputeClient<CudaRuntime>> {
    let device_id = device_override.unwrap_or_else(cuda_device_id);
    let device_count = tch::Cuda::device_count();
    if device_count <= device_id as i64 {
        bail!(
            "GPU evaluator requested CUDA device {} but only {} CUDA devices are available",
            device_id,
            device_count
        );
    }
    let device = CudaDevice::new(device_id);
    let client = CudaRuntime::client(&device);
    // AREA 1 (2026-06-09): probe the device's REAL per-buffer cap (VRAM/4 on CUDA)
    // and install it ONCE. CUDA canNOT inject a `MemoryConfiguration` (CudaServer
    // hardcodes its pool), so unlike the wgpu branch we do NOT call `init_setup`;
    // the existing reactive `trim_gpu_pool_if_over_budget` bounds the pool. We only
    // raise the per-buffer cap so heavy TFs stop being windowed to 120MB.
    install_gpu_buffer_cap(probe_gpu_buffer_cap_bytes(&client));
    Ok(client)
}

/// wgpu/Vulkan twin of the `gpu-cuda` client factory above. Uses cubecl's
/// `wgpu` (naga → SPIR-V) path, NOT `wgpu-spirv`: the direct SPIR-V passthrough
/// crashes AMD's Vulkan driver, so naga emits validated SPIR-V.
///
/// MULTI-GPU (2026-06-06): `NEOETHOS_BOT_SEARCH_EVAL_WGPU_DEVICE=<n>` pins this
/// process to discrete GPU `n` (`WgpuDevice::DiscreteGpu(n)`) — combo-level
/// parallelism: launch one discovery process per A6000 (env=0 and env=1) and
/// both cards run concurrently. Unset ⇒ `DefaultDevice` (the best adapter; the
/// first discrete GPU on a multi-GPU box).
#[cfg(all(feature = "gpu-vulkan", not(feature = "gpu-cuda")))]
fn create_gpu_client(device_override: Option<usize>) -> Result<ComputeClient<WgpuRuntime>> {
    // `WgpuDevice`/`WgpuRuntime` come from the module-level import (line ~7).
    // The pool-option structs (`MemoryPoolOptions`/`PoolType`) are used inside
    // `bounded_wgpu_pools` below, which imports them itself.
    use cubecl::wgpu::{MemoryConfiguration, RuntimeOptions, Vulkan, init_setup};

    // Multi-GPU sharding (Stage 2): an explicit `device_override` (one per lane)
    // wins; otherwise the legacy singular env var; otherwise the best adapter.
    let env_device = std::env::var("NEOETHOS_BOT_SEARCH_EVAL_WGPU_DEVICE")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok());
    let base_device = match device_override.or(env_device) {
        Some(n) => WgpuDevice::DiscreteGpu(n),
        None => WgpuDevice::DefaultDevice,
    };

    // AREA 1 (2026-06-09): initialize the wgpu server for this device EXACTLY ONCE,
    // with the adapter's REAL limits + a BOUNDED memory pool, then reuse it.
    //
    // `init_setup::<Vulkan>(&base_device, opts)` builds the wgpu adapter/device with
    // the adapter's NATIVE limits (cubecl requests `adapter.limits()` on the wgsl
    // path, NOT the 128MB defaults) AND registers a `ComputeClient` server under
    // `base_device` using our `opts` — see cubecl-wgpu runtime.rs:264-273. After one
    // `init_setup`, `WgpuRuntime::client(&base_device)` (which calls
    // `ComputeClient::load`) just looks the bounded server back up.
    //
    // CRITICAL — it must run AT MOST ONCE per device key: `ComputeClient::init`
    // PANICS ("already registered server") on a duplicate, and `init_setup` reads
    // the limits ONLY as a side effect of registering, so we cannot pre-read them to
    // size the pool. We therefore build the bounded pool from the SAME first
    // principles cubecl's default `SubSlices` uses (a graduated set sized off
    // `max_page` + alignment), passing the device's real `max_storage_buffer_binding_size`
    // as `max_page`. A single giant page would be WRONG — `SlicedPool::accept`
    // requires a slice to fill >=80% of its page, so small buffers would be rejected
    // → `BufferTooBig`. The graduated pools handle every buffer size; the only
    // difference from the default is a finite `dealloc_period` so freed pages RETURN
    // to the driver (cubecl's default is `None` = grow-only, the documented
    // 60k-row-H1 → 15GB peak). The reactive `trim_gpu_pool_if_over_budget` stays as
    // the backstop. The registry below guarantees the once-per-key invariant.
    static INITIALIZED: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<isize>>> =
        std::sync::OnceLock::new();
    // Key the cache on the resolved override (or -1 for "default adapter"). All
    // callers in one process with the same override share the one bounded device.
    let key: isize = device_override
        .or(env_device)
        .map(|n| n as isize)
        .unwrap_or(-1);

    let initialized =
        INITIALIZED.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
    let mut seen = initialized
        .lock()
        .map_err(|_| anyhow::anyhow!("wgpu device-init registry mutex poisoned"))?;
    if seen.insert(key) {
        // We can't read the device limits without registering, so the bounded pool
        // is built from a CONSERVATIVE `max_page` = the cap ceiling (2GiB), which is
        // >= the largest single buffer the windowing machinery ever builds
        // (`gpu_buffer_elem_cap` is clamped to <=2GiB). The graduated pools cover
        // every smaller size. wgpu's `min_uniform_buffer_offset_alignment` is 256 on
        // every desktop adapter; use it as the alignment unit (over-aligning is
        // harmless — it only rounds page sizes up).
        const POOL_MAX_PAGE: u64 = GPU_BUFFER_CAP_CEIL; // 2 GiB
        const POOL_ALIGNMENT: u64 = 256;
        let runtime_options = RuntimeOptions {
            tasks_max: 32,
            memory_config: MemoryConfiguration::Custom {
                pool_options: bounded_wgpu_pools(POOL_MAX_PAGE, POOL_ALIGNMENT),
            },
        };
        let setup = init_setup::<Vulkan>(&base_device, runtime_options);
        let binding_limit = setup.device.limits().max_storage_buffer_binding_size as u64;
        let buffer_limit = setup.adapter.limits().max_buffer_size;
        let real_cap = binding_limit.min(buffer_limit);
        install_gpu_buffer_cap(
            ((real_cap as f64 * 0.8) as u64).clamp(GPU_BUFFER_CAP_FLOOR, GPU_BUFFER_CAP_CEIL),
        );
        tracing::info!(
            target: "neoethos_search::cubecl_eval",
            max_storage_buffer_binding_size = binding_limit,
            max_buffer_size = buffer_limit,
            "wgpu adapter limits probed at init_setup (bounded pool installed)"
        );
    }
    drop(seen); // release the registry lock before building the (cheap) client

    Ok(WgpuRuntime::client(&base_device))
}

/// Build a GRADUATED set of sliced memory pools (mirroring cubecl's default
/// `SubSlices` construction: cubecl-runtime memory_manage.rs:202-258) but with a
/// finite `dealloc_period` so freed pages RETURN to the driver instead of growing
/// forever. `max_page` is the largest single buffer to accommodate; `alignment`
/// rounds page sizes. The structure: a tiny exclusive pool for sub-alignment
/// allocations, a descending ladder of `max_page/4, /16, /64, …` sliced pools
/// (each `max_slice_size = page/2^i` to curb fragmentation), and a final full
/// `max_page` pool. Every buffer size therefore lands in a right-sized pool — a
/// single giant page would reject small allocations (`SlicedPool::accept` needs a
/// slice to fill >=80% of its page).
#[cfg(all(feature = "gpu-vulkan", not(feature = "gpu-cuda")))]
fn bounded_wgpu_pools(
    max_page: u64,
    alignment: u64,
) -> Vec<cubecl_runtime::memory_management::MemoryPoolOptions> {
    use cubecl_runtime::memory_management::{MemoryPoolOptions, PoolType};
    const MB: u64 = 1024 * 1024;
    // Reclaim a page after it has gone unused across ~64 parent allocations. Bounds
    // the otherwise grow-only high-water mark while keeping enough reuse that the
    // hot window/gene-batch loop doesn't thrash the driver allocator.
    const DEALLOC_PERIOD: u64 = 64;
    let alignment = alignment.max(1);

    let mut pools: Vec<MemoryPoolOptions> = Vec::new();
    // Sub-alignment allocations can't use offsets (wgpu) — give them an exclusive
    // pool. (dealloc_period None: these are tiny + reused constantly.)
    pools.push(MemoryPoolOptions {
        pool_type: PoolType::ExclusivePages { max_alloc_size: 0 },
        dealloc_period: None,
    });

    let mut current = max_page;
    let mut max_sizes: Vec<u64> = Vec::new();
    let mut page_sizes: Vec<u64> = Vec::new();
    let mut base: u32 = pools.len() as u32;
    while current >= 32 * MB {
        current /= 4;
        current = current.next_multiple_of(alignment);
        max_sizes.push(current / 2u64.pow(base));
        page_sizes.push(current);
        base += 1;
    }
    max_sizes.reverse();
    page_sizes.reverse();
    for i in 0..max_sizes.len() {
        pools.push(MemoryPoolOptions {
            pool_type: PoolType::SlicedPages {
                page_size: page_sizes[i],
                max_slice_size: max_sizes[i].max(alignment),
            },
            dealloc_period: Some(DEALLOC_PERIOD),
        });
    }
    // Final big pool for the largest buffers.
    let big = (max_page / alignment) * alignment;
    pools.push(MemoryPoolOptions {
        pool_type: PoolType::SlicedPages {
            page_size: big.max(alignment),
            max_slice_size: big.max(alignment),
        },
        dealloc_period: Some(DEALLOC_PERIOD),
    });
    pools
}

#[cube(launch)]
fn synthesize_signals_kernel<F: Float + CubeElement>(
    indicators: &Array<F>,
    gene_offsets: &Array<i32>,
    gene_indices: &Array<i32>,
    gene_weights: &Array<F>,
    long_thr: &Array<F>,
    short_thr: &Array<F>,
    smc_data: &Array<i32>,
    gene_smc_flags: &Array<i32>,
    smc_weights: &Array<F>,
    output: &mut Array<i32>,
    // Phase 2 (2026-06-06): per-bar confidence of the raw threshold crossing,
    // mirrors `synthesize_signals_and_confidence_cpu` (eval.rs:1543-1544).
    // f32 regardless of `F` so the backtest kernel's risk-sizing reads the same
    // precision the CPU uses; written only where the final signal survives.
    confidences_out: &mut Array<f32>,
    n_samples: u32,
    gate_threshold: F,
) {
    // cubecl 0.9: ABSOLUTE_POS and Array::len() are `usize`, and array
    // indexing also expects `usize`. Coerce all u32 kernel parameters
    // to usize at the top of the kernel so the rest reads naturally.
    //
    // For mutable scalar accumulators (`combined`, `active_sum`,
    // `score`, `sig`) we must use RuntimeCell because cubecl 0.9's
    // `assign` and `assign_op` paths both reject const-initialized
    // `let mut` bindings.
    let pos = ABSOLUTE_POS;
    if pos < output.len() {
        let n_samples = n_samples as usize;
        let gene = pos / n_samples;
        let sample = pos % n_samples;

        let start = gene_offsets[gene] as usize;
        let end = gene_offsets[gene + 1] as usize;
        let combined = RuntimeCell::<F>::new(F::new(0.0));
        for i in start..end {
            let idx = gene_indices[i] as usize;
            let weight = gene_weights[i];
            let indicator = indicators[idx * n_samples + sample];
            combined.store(combined.read() + weight * indicator);
        }

        let lt = long_thr[gene];
        let st = short_thr[gene];
        let combined_val = combined.read();
        let sig = RuntimeCell::<i32>::new(0);
        if combined_val >= lt {
            sig.store(1);
        } else if combined_val <= st {
            sig.store(-1);
        }

        let sig_val = sig.read();
        if sig_val == 0 {
            output[pos] = 0;
            confidences_out[pos] = 0.0;
            terminate!();
        }

        // Confidence of the raw threshold crossing (pre-gate); computed in `F`
        // (cast to f32 only at the store) to match the CPU
        // `synthesize_signals_and_confidence_cpu` (eval.rs:1507/1543-1544):
        // gap = |lt - st| guarded >= 1e-6; margin = (combined-lt) long / (st-combined) short;
        // conf = (margin/gap).clamp(0,1). Written only where the final signal survives.
        let gap_raw = lt - st;
        // #1375 WORKAROUND (tracel-ai/cubecl#1375, open): an expression-position
        // `let x = if <runtime cond> { a } else { b }` returns the ELSE branch
        // UNCONDITIONALLY on the wgpu/Vulkan backend (CPU & CUDA are correct).
        // Statement-if + RuntimeCell is correct on ALL backends, so this both
        // preserves CPU/CUDA behaviour and FIXES the Vulkan eval — and matches the
        // RuntimeCell idiom used throughout this kernel. gap = |lt - st|, floored 1e-6.
        let gap_abs = RuntimeCell::<F>::new(gap_raw);
        if gap_raw < F::new(0.0) {
            gap_abs.store(F::new(0.0) - gap_raw);
        }
        let gap = RuntimeCell::<F>::new(gap_abs.read());
        if gap_abs.read() < F::new(1e-6) {
            gap.store(F::new(1e-6));
        }
        // margin: long = combined - lt, short = st - combined.
        let margin = RuntimeCell::<F>::new(st - combined_val);
        if sig_val == 1 {
            margin.store(combined_val - lt);
        }
        let conf_f = margin.read() / gap.read();
        // conf = conf_f.clamp(0, 1)  (statement-if else-if chain — #1375-safe).
        let conf = RuntimeCell::<F>::new(conf_f);
        if conf_f < F::new(0.0) {
            conf.store(F::new(0.0));
        } else if conf_f > F::new(1.0) {
            conf.store(F::new(1.0));
        }

        let flag_base = gene * SMC_WIDTH;
        let smc_base = sample * SMC_WIDTH;
        let active_sum = RuntimeCell::<F>::new(F::new(0.0));
        for j in 0..SMC_WIDTH {
            if gene_smc_flags[flag_base + j] != 0 {
                active_sum.store(active_sum.read() + smc_weights[j]);
            }
        }

        let active_sum_val = active_sum.read();
        if active_sum_val <= F::new(0.0) {
            output[pos] = sig_val;
            confidences_out[pos] = f32::cast_from(conf.read());
            terminate!();
        }

        // #1375 workaround: gate = min(active_sum_val, gate_threshold). On Vulkan the
        // expression form returned gate_threshold unconditionally, making the SMC
        // gate at `score >= gate` HARDER for low-SMC-weight genes → signals wrongly
        // suppressed vs CPU/CUDA. Statement-if restores parity.
        let gate = RuntimeCell::<F>::new(gate_threshold);
        if active_sum_val < gate_threshold {
            gate.store(active_sum_val);
        }
        let score = RuntimeCell::<F>::new(F::new(0.0));
        for k in 0..SMC_WIDTH {
            if gene_smc_flags[flag_base + k] != 0 {
                let smc_value = smc_data[smc_base + k];
                if k == 5 {
                    if smc_value == 1 {
                        score.store(score.read() + smc_weights[k]);
                    }
                } else if smc_value == sig_val {
                    score.store(score.read() + smc_weights[k]);
                }
            }
        }

        if score.read() >= gate.read() {
            output[pos] = sig_val;
            confidences_out[pos] = f32::cast_from(conf.read());
        } else {
            output[pos] = 0;
            confidences_out[pos] = 0.0;
        }
    }
}

#[cube(launch)]
fn backtest_population_kernel(
    close_pips: &Array<f32>,
    high_pips: &Array<f32>,
    low_pips: &Array<f32>,
    signals_flat: &Array<i32>,
    timestamp_deltas_ms: &Array<i32>,
    month_idx: &Array<i32>,
    day_idx: &Array<i32>,
    sl_pips: &Array<f32>,
    tp_pips: &Array<f32>,
    metrics_out: &mut Array<f32>,
    trade_counts_out: &mut Array<i32>,
    monthly_pnls_out: &mut Array<f32>,
    month_counts_out: &mut Array<i32>,
    n_samples: u32,
    month_capacity: u32,
    initial_equity: f32,
    max_hold_bars: u32,
    min_hold_bars: u32,
    max_trades_per_day: u32,
    gap_threshold_ms: i32,
    use_timestamps: i32,
    trailing_enabled: i32,
    trailing_atr_multiplier: f32,
    trailing_be_trigger_r: f32,
    spread_pips: f32,
    commission_per_trade: f32,
    pip_value_per_lot: f32,
    // Phase C.3 (2026-05-28) — broker-supplied carry costs mirrored
    // from the CPU `apply_carry_and_fee` helper. Sign convention
    // matches the broker: positive = credit, negative = charge per
    // overnight day. `pnl_conversion_fee_rate` is a fraction (0.005
    // = 0.5%); skipped if non-finite / out-of-range so a missing-
    // broker-data run still produces a backtest, matching the CPU
    // kernel's fail-safe default behaviour.
    swap_long_pips_per_day: f32,
    swap_short_pips_per_day: f32,
    pnl_conversion_fee_rate: f32,
    // Phase 2 (2026-06-06): risk-based, confidence-scaled position sizing —
    // mirrors `risk_based_pos_lots` + the pos_lots multiply sites on the CPU
    // (eval.rs:657-685, 1049-1054, 809/865-880/979/627). `confidences_flat` is
    // per-gene-per-bar (same layout as `signals_flat`). When `risk_based_sizing`
    // is 0 the kernel forces pos_lots = 1.0 (legacy fixed-1-lot parity).
    confidences_flat: &Array<f32>,
    risk_based_sizing: i32,
    risk_per_trade_min: f32,
    risk_per_trade_max: f32,
    high_quality_confidence: f32,
    // Per-month STARTING equity (sibling of monthly_pnls_out); the host divides
    // monthly_pnls_out / month_start_equities_out to get monthly_target_hit_rate
    // (metric slot 7), matching eval.rs:1110-1131.
    month_start_equities_out: &mut Array<f32>,
    // FTMO prop-firm observables, `n_genes * FTMO_WIDTH` laid out per gene
    // (see `FTMO_WIDTH`). MUST be the LAST kernel parameter so `launch_backtest_kernel`
    // appends its `ArrayArg` after all the existing args.
    ftmo_out: &mut Array<f32>,
) {
    // cubecl 0.9: index arithmetic is usize; coerce u32 params at the top.
    // Every scalar accumulator that gets reassigned must use RuntimeCell —
    // `let mut x = literal;` and `let mut x = param;` both produce
    // immutable bindings in cubecl 0.9, and any later `=`/`+=` panics.
    if ABSOLUTE_POS < trade_counts_out.len() {
        let gene = ABSOLUTE_POS;
        let n_samples = n_samples as usize;
        let month_capacity = month_capacity as usize;
        let max_hold_bars = max_hold_bars as usize;
        let min_hold_bars = min_hold_bars as usize;
        let max_trades_per_day = max_trades_per_day as usize;
        let signal_base = gene * n_samples;
        let month_base = gene * month_capacity;
        let metric_base = gene * BACKTEST_CORE_METRIC_WIDTH;
        let ftmo_base = gene * FTMO_WIDTH;

        for zero_idx in 0..month_capacity {
            monthly_pnls_out[month_base + zero_idx] = 0.0;
            month_start_equities_out[month_base + zero_idx] = initial_equity;
        }
        month_counts_out[gene] = 0;
        trade_counts_out[gene] = 0;
        for fj in 0..FTMO_WIDTH {
            ftmo_out[ftmo_base + fj] = 0.0;
        }

        if n_samples == 0 {
            for j in 0..BACKTEST_CORE_METRIC_WIDTH {
                metrics_out[metric_base + j] = 0.0;
            }
            terminate!();
        }

        let sl_distance = sl_pips[gene];
        let tp_distance = tp_pips[gene];

        let equity = RuntimeCell::<f32>::new(initial_equity);
        let peak_equity = RuntimeCell::<f32>::new(initial_equity);
        let max_dd = RuntimeCell::<f32>::new(0.0);
        let trade_count = RuntimeCell::<i32>::new(0);
        let wins = RuntimeCell::<i32>::new(0);
        let gross_profit = RuntimeCell::<f32>::new(0.0);
        let gross_loss = RuntimeCell::<f32>::new(0.0);

        let last_month = RuntimeCell::<i32>::new(-1);
        let current_month_pnl = RuntimeCell::<f32>::new(0.0);
        let month_ptr = RuntimeCell::<i32>::new(-1);
        // Per-month starting equity (slot-7 monthly_target_hit_rate). Mirrors the
        // CPU `current_month_start_equity` carried at each month boundary (eval.rs:732/778).
        let current_month_start_equity = RuntimeCell::<f32>::new(initial_equity);
        // Confidence-scaled lot size, captured ONCE at entry, held for the trade
        // (eval.rs:1049-1054). 1.0 = legacy fixed-1-lot.
        let pos_lots = RuntimeCell::<f32>::new(1.0);

        let last_day = RuntimeCell::<i32>::new(-1);
        let day_peak = RuntimeCell::<f32>::new(initial_equity);
        let day_low = RuntimeCell::<f32>::new(initial_equity);
        let max_daily_dd = RuntimeCell::<f32>::new(0.0);
        let day_trade_count = RuntimeCell::<u32>::new(0);

        // ── FTMO prop-firm observables (mirror validation.rs::compute_prop_firm_risk_summary) ──
        // Trades bucket by integer DAY == day_idx[i] (same key the CPU derives from
        // trade.exit_time / 86_400_000). `current_day_pnl` accumulates realized pnl of
        // the day in progress; finalized at each day boundary and once more after the loop.
        let current_day_pnl = RuntimeCell::<f32>::new(0.0);
        let max_daily_loss = RuntimeCell::<f32>::new(0.0);
        // END-OF-DAY equity curve drawdown: equity at a day boundary already equals
        // initial + sum(all prior days' realized pnl), i.e. exactly the CPU's per-day
        // equity point (equity += day_pnl iterated in day order). No-trade boundary days
        // repeat the prior point → never create a new peak or a larger DD → harmless.
        let eod_peak = RuntimeCell::<f32>::new(initial_equity);
        let max_eod_dd = RuntimeCell::<f32>::new(0.0);
        let positive_day_sum = RuntimeCell::<f32>::new(0.0);
        let largest_positive_day = RuntimeCell::<f32>::new(0.0);
        let max_trades_day = RuntimeCell::<u32>::new(0);
        let trading_days = RuntimeCell::<i32>::new(0);
        // FTMO trade-DAY counting must bucket by the CLOSE day (= trade.exit_time/day),
        // because the CPU `compute_prop_firm_risk_summary` keys `day_trade_count` on
        // `trade.exit_time/86_400_000`. The existing `day_trade_count` increments at
        // ENTRY (it gates `max_trades_per_day` on the entry side) and would diverge for
        // overnight holds, so we keep a SEPARATE close-bucketed counter here.
        let current_day_closes = RuntimeCell::<u32>::new(0);

        let in_pos = RuntimeCell::<i32>::new(0);
        let entry_px = RuntimeCell::<f32>::new(0.0);
        let entry_idx = RuntimeCell::<i32>::new(-1);
        let trail_px = RuntimeCell::<f32>::new(0.0);
        // Phase C.3: accumulated days in position. Resets to 0 at entry,
        // each in-position bar adds `timestamp_deltas_ms[i] / 86_400_000`.
        // f32 precision loss on the cast is bounded by ~5 ms per bar
        // (cast of values up to 86.4M ms into 24-bit mantissa); over
        // a year of D1 bars this accumulates to <$0.001 of swap error
        // at typical EURUSD pip values — negligible vs the $122/year
        // swap charge being modelled.
        let position_days = RuntimeCell::<f32>::new(0.0);

        for i in 1..n_samples {
            // Phase C.3: accumulate carry duration while in position.
            // Runs BEFORE any exit logic so the close branches use the
            // total time held, INCLUDING the delta into the current bar.
            if in_pos.read() != 0 && use_timestamps != 0 && timestamp_deltas_ms[i] > 0 {
                let delta_days = timestamp_deltas_ms[i] as f32 / 86_400_000.0;
                position_days.store(position_days.read() + delta_days);
            }

            let m_val = month_idx[i];
            let last_month_v = last_month.read();
            if m_val != last_month_v {
                if last_month_v != -1 {
                    let next_ptr = month_ptr.read() + 1;
                    month_ptr.store(next_ptr);
                    if next_ptr >= 0 && next_ptr < month_capacity as i32 {
                        monthly_pnls_out[month_base + next_ptr as usize] = current_month_pnl.read();
                        month_start_equities_out[month_base + next_ptr as usize] =
                            current_month_start_equity.read();
                    }
                }
                current_month_pnl.store(0.0);
                // New month starts at the equity carried in (eval.rs:778). Runs
                // BEFORE this bar's exits mutate equity — preserve the ordering.
                current_month_start_equity.store(equity.read());
                last_month.store(m_val);
            }

            let d_val = day_idx[i];
            let last_day_v = last_day.read();
            if d_val != last_day_v {
                if last_day_v != -1 && day_peak.read() > 0.0 {
                    let dd = (day_peak.read() - day_low.read()) / day_peak.read();
                    if dd > max_daily_dd.read() {
                        max_daily_dd.store(dd);
                    }
                }
                // ── FTMO: finalize the PREVIOUS day before resetting day state ──
                // Runs AFTER the prior day's exits have already mutated `equity`
                // (the entry for THIS bar happens later in the loop), so `equity`
                // and `current_day_pnl` here hold the just-finished day's totals.
                if last_day_v != -1 {
                    let dp = current_day_pnl.read();
                    // max_daily_loss = max over days of |negative day pnl|.
                    if dp < 0.0 {
                        let neg = -dp;
                        if neg > max_daily_loss.read() {
                            max_daily_loss.store(neg);
                        }
                    }
                    // largest_profit_share inputs: sum + max of POSITIVE day pnls.
                    if dp > 0.0 {
                        positive_day_sum.store(positive_day_sum.read() + dp);
                        if dp > largest_positive_day.read() {
                            largest_positive_day.store(dp);
                        }
                    }
                    // END-OF-DAY equity drawdown: `equity` == initial + sum(all prior
                    // days' pnl) == the CPU's per-day equity point.
                    let eod_eq = equity.read();
                    if eod_eq > eod_peak.read() {
                        eod_peak.store(eod_eq);
                    }
                    let ep = eod_peak.read();
                    let eod_dd = RuntimeCell::<f32>::new(0.0);
                    if ep > 0.0 {
                        eod_dd.store((ep - eod_eq) / ep);
                    }
                    if eod_dd.read() > max_eod_dd.read() {
                        max_eod_dd.store(eod_dd.read());
                    }
                    // trading_days + max_trades_per_day from the finished day —
                    // counted by CLOSES (exit-day bucketed) to match the CPU.
                    let dtc = current_day_closes.read();
                    if dtc > 0 {
                        trading_days.store(trading_days.read() + 1);
                    }
                    if dtc > max_trades_day.read() {
                        max_trades_day.store(dtc);
                    }
                    current_day_pnl.store(0.0);
                    current_day_closes.store(0);
                }
                last_day.store(d_val);
                day_peak.store(equity.read());
                day_low.store(equity.read());
                day_trade_count.store(0);
            }

            let in_pos_v = in_pos.read();
            if in_pos_v != 0
                && use_timestamps != 0
                && gap_threshold_ms > 0
                && timestamp_deltas_ms[i] >= gap_threshold_ms
            {
                let entry_px_v = entry_px.read();
                let pnl_cell = RuntimeCell::<f32>::new(0.0);
                if in_pos_v == 1 {
                    pnl_cell.store((close_pips[i] - entry_px_v) * pip_value_per_lot);
                } else {
                    pnl_cell.store((entry_px_v - close_pips[i]) * pip_value_per_lot);
                }
                pnl_cell.store(
                    pnl_cell.read()
                        - commission_per_trade
                        - (spread_pips * 0.5 * pip_value_per_lot),
                );
                // Phase 2: scale (gross - commission - half_spread) by pos_lots
                // BEFORE swap (swap scaled in its own term). Matches eval.rs:809.
                pnl_cell.store(pnl_cell.read() * pos_lots.read());
                // Phase C.3: broker swap (signed: + = credit, − = charge).
                // #1375 workaround: long => swap_long, short => swap_short.
                let swap_per_day_gap = RuntimeCell::<f32>::new(swap_short_pips_per_day);
                if in_pos_v == 1 {
                    swap_per_day_gap.store(swap_long_pips_per_day);
                }
                let swap_credit_gap =
                    swap_per_day_gap.read() * position_days.read() * pip_value_per_lot * pos_lots.read();
                pnl_cell.store(pnl_cell.read() + swap_credit_gap);
                // PnL conversion fee applied last; skip if out-of-range.
                if pnl_conversion_fee_rate > 0.0 && pnl_conversion_fee_rate < 1.0 {
                    pnl_cell.store(pnl_cell.read() * (1.0 - pnl_conversion_fee_rate));
                }
                let pnl = pnl_cell.read();
                equity.store(equity.read() + pnl);
                current_month_pnl.store(current_month_pnl.read() + pnl);
                // FTMO: attribute this realized pnl + closed-trade to the CURRENT
                // (close) day's bucket — matches the CPU's exit-day bucketing.
                current_day_pnl.store(current_day_pnl.read() + pnl);
                current_day_closes.store(current_day_closes.read() + 1);
                trade_count.store(trade_count.read() + 1);
                if pnl > 0.0 {
                    wins.store(wins.read() + 1);
                    gross_profit.store(gross_profit.read() + pnl);
                } else {
                    gross_loss.store(gross_loss.read() - pnl);
                }
                in_pos.store(0);
                let eq = equity.read();
                if eq > peak_equity.read() {
                    peak_equity.store(eq);
                }
                if eq < day_low.read() {
                    day_low.store(eq);
                }
                let pe = peak_equity.read();
                let current_dd = RuntimeCell::<f32>::new(0.0);
                if pe > 0.0 {
                    current_dd.store((pe - eq) / pe);
                }
                if current_dd.read() > max_dd.read() {
                    max_dd.store(current_dd.read());
                }
            }

            let in_pos_v2 = in_pos.read();
            if in_pos_v2 != 0 {
                let lo = low_pips[i];
                let hi = high_pips[i];
                let entry_px_v = entry_px.read();

                // Phase 2: float PnL scaled by pos_lots (eval.rs:865-880) so the
                // equity-based DD tracks the sized position.
                // #1375 workaround: long worst-case intrabar = low-entry, short = entry-high.
                // The expr form gave longs the SHORT formula on Vulkan → wrong max_dd.
                let worst_base = RuntimeCell::<f32>::new((entry_px_v - hi) * pip_value_per_lot);
                if in_pos_v2 == 1 {
                    worst_base.store((lo - entry_px_v) * pip_value_per_lot);
                }
                let worst_float_pnl = worst_base.read() * pos_lots.read();
                let eq = equity.read();
                if (eq + worst_float_pnl) < day_low.read() {
                    day_low.store(eq + worst_float_pnl);
                }

                // #1375 workaround: long best-case intrabar = high-entry, short = entry-low.
                let best_base = RuntimeCell::<f32>::new((entry_px_v - lo) * pip_value_per_lot);
                if in_pos_v2 == 1 {
                    best_base.store((hi - entry_px_v) * pip_value_per_lot);
                }
                let best_float_pnl = best_base.read() * pos_lots.read();
                if (eq + best_float_pnl) > peak_equity.read() {
                    peak_equity.store(eq + best_float_pnl);
                }

                let pe = peak_equity.read();
                let current_dd = RuntimeCell::<f32>::new(0.0);
                if pe > 0.0 {
                    current_dd.store((pe - (eq + worst_float_pnl)) / pe);
                }
                if current_dd.read() > max_dd.read() {
                    max_dd.store(current_dd.read());
                }

                let pnl_cell = RuntimeCell::<f32>::new(0.0);
                let exit_cell = RuntimeCell::<u32>::new(0);
                let bars_held = i as i32 - entry_idx.read();
                let past_min_hold = min_hold_bars == 0 || bars_held >= min_hold_bars as i32;

                if past_min_hold && in_pos_v2 == 1 {
                    let sl_cell = RuntimeCell::<f32>::new(entry_px_v - sl_distance);
                    let tp = entry_px_v + tp_distance;
                    // Apply only the trail from PRIOR bars (no intra-bar look-ahead — must
                    // match the CPU eval: this bar's high can't move the stop its own low is
                    // checked against). `trail_px == 0.0` is the unset sentinel.
                    if trailing_enabled != 0 && trail_px.read() > 0.0 && trail_px.read() > sl_cell.read() {
                        sl_cell.store(trail_px.read());
                    }
                    let sl_v = sl_cell.read();
                    if lo <= sl_v {
                        pnl_cell.store((sl_v - entry_px_v) * pip_value_per_lot);
                        exit_cell.store(1);
                    } else if hi >= tp {
                        pnl_cell.store((tp - entry_px_v) * pip_value_per_lot);
                        exit_cell.store(1);
                    }
                    // AFTER the exit check: ratchet the trail up from THIS bar's high.
                    if exit_cell.read() == 0 && trailing_enabled != 0 {
                        let mv = hi - entry_px_v;
                        if mv >= (trailing_be_trigger_r * sl_distance) {
                            let candidate = hi - (trailing_atr_multiplier * sl_distance);
                            if trail_px.read() == 0.0 || candidate > trail_px.read() {
                                trail_px.store(candidate);
                            }
                        }
                    }
                } else if past_min_hold {
                    let sl_cell = RuntimeCell::<f32>::new(entry_px_v + sl_distance);
                    let tp = entry_px_v - tp_distance;
                    if trailing_enabled != 0 && trail_px.read() > 0.0 && trail_px.read() < sl_cell.read() {
                        sl_cell.store(trail_px.read());
                    }
                    let sl_v = sl_cell.read();
                    if hi >= sl_v {
                        pnl_cell.store((entry_px_v - sl_v) * pip_value_per_lot);
                        exit_cell.store(1);
                    } else if lo <= tp {
                        pnl_cell.store((entry_px_v - tp) * pip_value_per_lot);
                        exit_cell.store(1);
                    }
                    // AFTER the exit check: ratchet the trail down from THIS bar's low.
                    if exit_cell.read() == 0 && trailing_enabled != 0 {
                        let mv = entry_px_v - lo;
                        if mv >= (trailing_be_trigger_r * sl_distance) {
                            let candidate = lo + (trailing_atr_multiplier * sl_distance);
                            if trail_px.read() == 0.0 || candidate < trail_px.read() {
                                trail_px.store(candidate);
                            }
                        }
                    }
                }

                if exit_cell.read() == 0
                    && past_min_hold
                    && max_hold_bars > 0
                    && bars_held >= max_hold_bars as i32
                {
                    if in_pos_v2 == 1 {
                        pnl_cell.store((close_pips[i] - entry_px_v) * pip_value_per_lot);
                    } else {
                        pnl_cell.store((entry_px_v - close_pips[i]) * pip_value_per_lot);
                    }
                    exit_cell.store(1);
                }

                if exit_cell.read() != 0 {
                    pnl_cell.store(
                        pnl_cell.read()
                            - commission_per_trade
                            - (spread_pips * 0.5 * pip_value_per_lot),
                    );
                    // Phase 2: scale (gross - commission - half_spread) by pos_lots
                    // BEFORE swap. Matches eval.rs:979.
                    pnl_cell.store(pnl_cell.read() * pos_lots.read());
                    // Phase C.3: broker swap (signed: + = credit, − = charge).
                    // #1375 workaround: long => swap_long, short => swap_short.
                    let swap_per_day = RuntimeCell::<f32>::new(swap_short_pips_per_day);
                    if in_pos_v2 == 1 {
                        swap_per_day.store(swap_long_pips_per_day);
                    }
                    let swap_credit =
                        swap_per_day.read() * position_days.read() * pip_value_per_lot * pos_lots.read();
                    pnl_cell.store(pnl_cell.read() + swap_credit);
                    if pnl_conversion_fee_rate > 0.0 && pnl_conversion_fee_rate < 1.0 {
                        pnl_cell.store(pnl_cell.read() * (1.0 - pnl_conversion_fee_rate));
                    }
                    let pnl = pnl_cell.read();
                    equity.store(equity.read() + pnl);
                    current_month_pnl.store(current_month_pnl.read() + pnl);
                    // FTMO: attribute this realized pnl + closed-trade to the CURRENT
                    // (close) day's bucket — matches the CPU's exit-day bucketing.
                    current_day_pnl.store(current_day_pnl.read() + pnl);
                    current_day_closes.store(current_day_closes.read() + 1);
                    trade_count.store(trade_count.read() + 1);
                    if pnl > 0.0 {
                        wins.store(wins.read() + 1);
                        gross_profit.store(gross_profit.read() + pnl);
                    } else {
                        gross_loss.store(gross_loss.read() - pnl);
                    }
                    in_pos.store(0);
                    let eq2 = equity.read();
                    if eq2 > peak_equity.read() {
                        peak_equity.store(eq2);
                    }
                    if eq2 < day_low.read() {
                        day_low.store(eq2);
                    }
                    let pe2 = peak_equity.read();
                    let current_dd = RuntimeCell::<f32>::new(0.0);
                    if pe2 > 0.0 {
                        current_dd.store((pe2 - eq2) / pe2);
                    }
                    if current_dd.read() > max_dd.read() {
                        max_dd.store(current_dd.read());
                    }
                }
            } else {
                // Causal entry: read PRIOR-bar signal, fill at CURRENT-bar close.
                let s = signals_flat[signal_base + i - 1];
                if s != 0 {
                    if !(max_trades_per_day > 0
                        && (day_trade_count.read() as usize) >= max_trades_per_day)
                    {
                        in_pos.store(s);
                        entry_px.store(close_pips[i] + (s as f32) * spread_pips * 0.5);
                        entry_idx.store(i as i32);
                        trail_px.store(0.0);
                        // Phase C.3: reset carry accumulator at new entry.
                        position_days.store(0.0);
                        // Phase 2: risk-based, confidence-scaled lot size, captured
                        // at entry from running equity + the prior-bar confidence
                        // (causal i-1, same shift as the signal read). Mirrors
                        // risk_based_pos_lots (eval.rs:657-685) term-for-term.
                        if risk_based_sizing != 0 {
                            // RuntimeCell idiom (cubecl rejects if-as-value that mixes
                            // array-read cube values with bare literals).
                            let conf_c = RuntimeCell::<f32>::new(confidences_flat[signal_base + i - 1]);
                            if conf_c.read() < 0.0 {
                                conf_c.store(0.0);
                            }
                            if conf_c.read() > 1.0 {
                                conf_c.store(1.0);
                            }
                            // conf_scale = (conf/hq).min(1.0), guarded for hq<=0 (=> 1.0).
                            let conf_scale = RuntimeCell::<f32>::new(1.0);
                            if high_quality_confidence > 0.0 {
                                let r = conf_c.read() / high_quality_confidence;
                                if r < 1.0 {
                                    conf_scale.store(r);
                                }
                            }
                            let risk_pct = risk_per_trade_min
                                + (risk_per_trade_max - risk_per_trade_min) * conf_scale.read();
                            // eff_sl = sl_distance.max(1.0)
                            let eff_sl = RuntimeCell::<f32>::new(1.0);
                            if sl_distance > 1.0 {
                                eff_sl.store(sl_distance);
                            }
                            let denom = eff_sl.read() * pip_value_per_lot;
                            let eq_now = equity.read();
                            // lots = (risk_pct*equity)/denom if valid, else 0; clamp [0,100].
                            let lots = RuntimeCell::<f32>::new(0.0);
                            if eq_now > 0.0 && denom > 1e-12 {
                                lots.store((risk_pct * eq_now) / denom);
                            }
                            if lots.read() > 100.0 {
                                lots.store(100.0);
                            }
                            if lots.read() < 0.0 {
                                lots.store(0.0);
                            }
                            pos_lots.store(lots.read());
                        } else {
                            pos_lots.store(1.0);
                        }
                        day_trade_count.store(day_trade_count.read() + 1);
                    }
                }
            }
        }

        // ── FTMO: FLUSH THE FINAL DAY ──────────────────────────────────────
        // The boundary block only fires on a CHANGE of day_idx, so the LAST day
        // in the series is never finalized there. Repeat the same finalize using
        // the final `current_day_pnl` / `equity` / `day_trade_count`, but only if
        // at least one bar was processed (last_day != -1). This exactly mirrors
        // the CPU iterating its last BTreeMap day.
        if last_day.read() != -1 {
            let dp = current_day_pnl.read();
            if dp < 0.0 {
                let neg = -dp;
                if neg > max_daily_loss.read() {
                    max_daily_loss.store(neg);
                }
            }
            if dp > 0.0 {
                positive_day_sum.store(positive_day_sum.read() + dp);
                if dp > largest_positive_day.read() {
                    largest_positive_day.store(dp);
                }
            }
            let eod_eq = equity.read();
            if eod_eq > eod_peak.read() {
                eod_peak.store(eod_eq);
            }
            let ep = eod_peak.read();
            let eod_dd = RuntimeCell::<f32>::new(0.0);
            if ep > 0.0 {
                eod_dd.store((ep - eod_eq) / ep);
            }
            if eod_dd.read() > max_eod_dd.read() {
                max_eod_dd.store(eod_dd.read());
            }
            // Count by CLOSES (exit-day bucketed) to match the CPU.
            let dtc = current_day_closes.read();
            if dtc > 0 {
                trading_days.store(trading_days.read() + 1);
            }
            if dtc > max_trades_day.read() {
                max_trades_day.store(dtc);
            }
        }

        let final_equity = equity.read();
        let final_peak = peak_equity.read();
        let final_max_dd = max_dd.read();
        let final_trade_count = trade_count.read();
        let final_wins = wins.read();
        let final_gp = gross_profit.read();
        let final_gl = gross_loss.read();
        let final_max_daily_dd = max_daily_dd.read();
        let final_month_ptr = month_ptr.read();

        let net_profit = final_equity - initial_equity;
        let win_rate_cell = RuntimeCell::<f32>::new(0.0);
        if final_trade_count > 0 {
            win_rate_cell.store(final_wins as f32 / final_trade_count as f32);
        }
        let pf_cell = RuntimeCell::<f32>::new(0.0);
        if final_gl > 0.0 {
            pf_cell.store((final_gp / final_gl).min(10.0));
        } else if final_gp > 0.0 {
            pf_cell.store(10.0);
        }
        let expectancy_cell = RuntimeCell::<f32>::new(0.0);
        if final_trade_count > 0 {
            expectancy_cell.store(net_profit / final_trade_count as f32);
        }
        let filled_months_cell = RuntimeCell::<i32>::new(0);
        if final_month_ptr >= 0 {
            let raw = final_month_ptr + 1;
            if raw < month_capacity as i32 {
                filled_months_cell.store(raw);
            } else {
                filled_months_cell.store(month_capacity as i32);
            }
        }

        metrics_out[metric_base] = net_profit;
        metrics_out[metric_base + 1] = final_peak;
        metrics_out[metric_base + 2] = final_max_dd;
        metrics_out[metric_base + 3] = win_rate_cell.read();
        metrics_out[metric_base + 4] = pf_cell.read();
        metrics_out[metric_base + 5] = expectancy_cell.read();
        metrics_out[metric_base + 6] = final_max_daily_dd;
        trade_counts_out[gene] = final_trade_count;
        month_counts_out[gene] = filled_months_cell.read();

        // ── FTMO observables emit (mirrors validation.rs::compute_prop_firm_risk_summary) ──
        // net_return_pct = net_profit / initial_equity (guard initial_equity>0).
        let net_return_pct = RuntimeCell::<f32>::new(0.0);
        if initial_equity > 0.0 {
            net_return_pct.store(net_profit / initial_equity);
        }
        // max_daily_loss_pct = max|neg day pnl| / initial_equity.
        let max_daily_loss_pct = RuntimeCell::<f32>::new(0.0);
        if initial_equity > 0.0 {
            max_daily_loss_pct.store(max_daily_loss.read() / initial_equity);
        }
        // largest_profit_share = largest_positive_day / sum_positive_days (0 if none).
        // Statement-if guard (cubecl#1375: NEVER expression-if mixing runtime values).
        let largest_profit_share = RuntimeCell::<f32>::new(0.0);
        if positive_day_sum.read() > 1e-9 {
            largest_profit_share.store(largest_positive_day.read() / positive_day_sum.read());
        }

        ftmo_out[ftmo_base] = net_return_pct.read();
        ftmo_out[ftmo_base + 1] = max_daily_loss_pct.read();
        ftmo_out[ftmo_base + 2] = max_eod_dd.read();
        ftmo_out[ftmo_base + 3] = largest_profit_share.read();
        ftmo_out[ftmo_base + 4] = max_trades_day.read() as f32;
        ftmo_out[ftmo_base + 5] = trading_days.read() as f32;
    }
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    // Phase 64 — both CPU and GPU paths now share the canonical
    // `neoethos_core::utils::mean_std` so CPU/GPU rank parity cannot drift
    // due to a math-helper divergence.
    let (mean, std) = neoethos_core::utils::mean_std(values);
    if !mean.is_finite() || !std.is_finite() {
        return (0.0, 0.0);
    }
    (mean, std)
}

fn parse_training_precision(value: &str) -> Option<TrainingPrecision> {
    match value.trim().to_ascii_lowercase().as_str() {
        "fp32" | "f32" | "float32" => Some(TrainingPrecision::Fp32),
        "fp16" | "f16" | "float16" | "half" => Some(TrainingPrecision::Fp16),
        "bf16" | "bfloat16" => Some(TrainingPrecision::Bf16),
        "fp8" | "float8" => Some(TrainingPrecision::Fp8),
        "bf4" => Some(TrainingPrecision::Bf4),
        _ => None,
    }
}

fn requested_eval_precision() -> TrainingPrecision {
    // F-CORE3 closure: typed boundary via `cuda_env_knobs()`.
    cuda_env_knobs().requested_precision
}

fn prefers_bf16(requested: TrainingPrecision) -> bool {
    matches!(
        requested,
        TrainingPrecision::Bf16 | TrainingPrecision::Fp8 | TrainingPrecision::Bf4
    )
}

pub(crate) fn cuda_eval_signal_kernel_enabled() -> bool {
    // F-CORE3 closure: typed boundary via `cuda_env_knobs()`.
    cuda_env_knobs().eval_kernel_enabled
}

pub(crate) fn cuda_eval_backtest_kernel_enabled() -> bool {
    // F-CORE3 closure: typed boundary via `cuda_env_knobs()`.
    cuda_env_knobs().backtest_kernel_enabled
}

// **2026-05-25 — task #261**: switched from concrete `CudaRuntime` to a
// generic `R: Runtime` parameter so these helpers (and the kernel-launch
// fns below) compile against any cubecl runtime — CUDA (NVIDIA), Vulkan
// (cross-vendor via cubecl-wgpu/spirv), and ROCm/HIP. The CUDA-specific
// env knobs stay because they're plain numbers (kernel unit count,
// device id) — semantically valid for any backend; the `cuda_` prefix
// just reflects the env-var name and is a follow-up cosmetic rename.
fn signal_kernel_units<R: Runtime>(client: &ComputeClient<R>) -> u32 {
    let max_units = client.properties().hardware.max_units_per_cube.max(1);
    cuda_env_knobs()
        .eval_kernel_units_override
        .unwrap_or(max_units)
        .min(max_units)
        .max(1)
}

fn backtest_kernel_units<R: Runtime>(client: &ComputeClient<R>) -> u32 {
    let max_units = client.properties().hardware.max_units_per_cube.max(1);
    cuda_env_knobs()
        .backtest_kernel_units_override
        .unwrap_or(max_units)
        .min(max_units)
        .max(1)
}

#[allow(dead_code)] // gpu-cuda device selection only; wgpu uses DefaultDevice
fn cuda_device_id() -> usize {
    // F-CORE3 closure: typed boundary via `cuda_env_knobs()`. The
    // tracing::warn for unparseable values fires once at first read
    // (inside `read_cuda_device_id_from_env`) rather than on every
    // kernel launch.
    cuda_env_knobs().cuda_device_id
}

fn flatten_i32_rows(rows: &[SmcRow]) -> Vec<i32> {
    let mut out = Vec::with_capacity(rows.len().saturating_mul(SMC_WIDTH));
    for row in rows {
        for value in row {
            out.push(*value as i32);
        }
    }
    out
}

fn flatten_i32_flags(rows: &[SmcRow]) -> Vec<i32> {
    flatten_i32_rows(rows)
}

/// Host-side validation that every index the kernel will compute stays
/// within its array. This is the contract `synthesize_signals_kernel`
/// implicitly assumes; without these checks a single bad GA gene
/// silently reads garbage memory in the CUDA kernel, which produces
/// **wrong trading signals** with no error (panics in CUDA kernels
/// surface as CUDA driver errors or simply corrupt data — both far
/// worse than a clean `Err` for a real-money trading system).
///
/// Invariants (mirroring `synthesize_signals_kernel`):
///   1. `gene_offsets.len() == n_genes + 1` (CSR layout, last entry is total)
///   2. `gene_offsets` is monotonically non-decreasing
///   3. `gene_offsets[n_genes] as usize <= gene_indices.len()` and ≤ gene_weights.len()
///   4. every `gene_indices[i]` is in `[0, indicators_flat.len() / n_samples)`
///   5. `long_thr.len() == n_genes` and `short_thr.len() == n_genes`
///   6. `gene_smc_flags.len() == n_genes * SMC_WIDTH`
///   7. `smc_data.len() == n_samples * SMC_WIDTH`
///   8. `smc_weights.len() == SMC_WIDTH`
///   9. `indicators_flat.len() % n_samples == 0`
fn validate_signal_kernel_inputs<F>(
    indicators_flat: &[F],
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[F],
    long_thr: &[F],
    short_thr: &[F],
    smc_data: &[i32],
    gene_smc_flags: &[i32],
    smc_weights: &[F],
    n_genes: usize,
    n_samples: usize,
) -> Result<()> {
    if gene_offsets.len() != n_genes + 1 {
        bail!(
            "gene_offsets length {} must equal n_genes + 1 = {}",
            gene_offsets.len(),
            n_genes + 1
        );
    }
    if long_thr.len() != n_genes {
        bail!("long_thr length {} must equal n_genes {}", long_thr.len(), n_genes);
    }
    if short_thr.len() != n_genes {
        bail!("short_thr length {} must equal n_genes {}", short_thr.len(), n_genes);
    }
    if gene_smc_flags.len() != n_genes * SMC_WIDTH {
        bail!(
            "gene_smc_flags length {} must equal n_genes * SMC_WIDTH = {}",
            gene_smc_flags.len(),
            n_genes * SMC_WIDTH
        );
    }
    if smc_weights.len() != SMC_WIDTH {
        bail!(
            "smc_weights length {} must equal SMC_WIDTH = {}",
            smc_weights.len(),
            SMC_WIDTH
        );
    }
    if smc_data.len() != n_samples * SMC_WIDTH {
        bail!(
            "smc_data length {} must equal n_samples * SMC_WIDTH = {}",
            smc_data.len(),
            n_samples * SMC_WIDTH
        );
    }
    if n_samples == 0 {
        bail!("n_samples must be > 0");
    }
    if indicators_flat.len() % n_samples != 0 {
        bail!(
            "indicators_flat length {} must be a multiple of n_samples {}",
            indicators_flat.len(),
            n_samples
        );
    }
    let n_indicators = indicators_flat.len() / n_samples;
    let total_entries = gene_offsets[n_genes];
    if total_entries < 0 {
        bail!("gene_offsets[n_genes] = {} must be non-negative", total_entries);
    }
    if (total_entries as usize) > gene_indices.len() {
        bail!(
            "gene_offsets[n_genes] = {} exceeds gene_indices length {}",
            total_entries,
            gene_indices.len()
        );
    }
    if (total_entries as usize) > gene_weights.len() {
        bail!(
            "gene_offsets[n_genes] = {} exceeds gene_weights length {}",
            total_entries,
            gene_weights.len()
        );
    }
    // Monotonicity: gene_offsets[g] <= gene_offsets[g+1] for every g.
    for g in 0..n_genes {
        if gene_offsets[g] > gene_offsets[g + 1] {
            bail!(
                "gene_offsets must be non-decreasing: gene_offsets[{}]={} > gene_offsets[{}]={}",
                g,
                gene_offsets[g],
                g + 1,
                gene_offsets[g + 1]
            );
        }
        if gene_offsets[g] < 0 {
            bail!("gene_offsets[{}] = {} is negative", g, gene_offsets[g]);
        }
    }
    // Every gene_indices entry must reference a valid indicator row.
    // Checking up to `total_entries` because anything past that isn't read.
    let used_entries = total_entries as usize;
    for (i, &idx) in gene_indices.iter().take(used_entries).enumerate() {
        if idx < 0 {
            bail!("gene_indices[{}] = {} is negative", i, idx);
        }
        if (idx as usize) >= n_indicators {
            bail!(
                "gene_indices[{}] = {} exceeds n_indicators = {} (indicators_flat.len()/n_samples)",
                i,
                idx,
                n_indicators
            );
        }
    }
    Ok(())
}

// `F` first so callers can `launch_signal_kernel::<f32>(&client, ...)` and have
// `R` inferred from the client (turbofish is positional; the runtime is never
// named at the call sites).
/// Conservative max ELEMENTS per single GPU storage buffer. wgpu caps a storage
/// buffer at `max_storage_buffer_binding_size` (WebGPU default 128MB); exceeding
/// it raises "wgpu error: Out of Memory". We stay under it and window/batch the
/// GA's big buffers (the indicators matrix and the per-gene signal series) so
/// even huge-row timeframes (M1: ~5.3M rows) run on the GPU instead of falling
/// back to CPU. Overridable per box via `NEOETHOS_BOT_SEARCH_GPU_BUFFER_MB`
/// (raise it where the device's real limit is higher than the 128MB default).
/// Hardware-derived memory budgets installed once at discovery start by
/// [`auto_tune_memory_budgets`]. They make the engine fit whatever card + RAM it
/// finds with ZERO user config: the cap helpers below resolve their value as
/// `explicit env override > auto-tuned budget > conservative default`. A user
/// can therefore (a) do nothing and get a hardware-fit config, (b) set an env
/// var to force a specific value. Either way the engine never OOMs regardless of
/// requested population/generations (the budgets bound the windowing, the pool
/// trim, and the host gene-chunk).
#[derive(Clone, Copy, Debug)]
struct MemoryBudgets {
    /// Host budget (MB) for the per-gene signal+confidence assembly → gene chunk.
    host_budget_mb: u64,
    /// VRAM pool reserved budget (MB) above which the pool is trimmed.
    vram_budget_mb: u64,
    /// Per-storage-buffer cap (MB) for the windowing/batching.
    gpu_buffer_mb: usize,
}

static MEMORY_BUDGETS: std::sync::OnceLock<MemoryBudgets> = std::sync::OnceLock::new();

fn installed_memory_budgets() -> Option<MemoryBudgets> {
    MEMORY_BUDGETS.get().copied()
}

/// The device's REAL per-storage-buffer cap (bytes), probed ONCE from the active
/// GPU client the first time one is built (in `create_gpu_client`). Replaces the
/// historical hardcoded 120MB literal that was a workaround for cubecl using the
/// DEFAULT (128MB) wgpu limits instead of the adapter's true
/// `max_storage_buffer_binding_size` — on Vulkan that real limit is far higher
/// (often `u64::MAX`), and on CUDA it is VRAM/4 (cubecl-cuda runtime.rs). With the
/// real cap the windowing/batching machinery (still the safety net) produces a few
/// big launches instead of many tiny ones, so heavy TFs (M1/M3/M5) actually run on
/// the GPU. Populated on BOTH backends; `gpu_buffer_elem_cap` reads it.
static GPU_BUFFER_CAP_BYTES: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

/// Floor for the probed per-buffer cap: never drop below the historical 120MB so a
/// bogus tiny `max_page_size` report can't starve the GPU lane into many micro
/// launches.
const GPU_BUFFER_CAP_FLOOR: u64 = 120 * 1024 * 1024;
/// Ceiling for the probed per-buffer cap: keep a single window at <=2GiB so its
/// host element count (fed into u32 cube math) stays addressable and a single
/// allocation can't blow a small card before the pool-trim/catch_unwind safety
/// nets engage.
const GPU_BUFFER_CAP_CEIL: u64 = 2 * 1024 * 1024 * 1024;

/// Probe the active client for the device's REAL per-storage-buffer byte limit.
/// `client.properties().memory.max_page_size` is populated on BOTH backends:
/// cubecl-wgpu sets it to `device.limits().max_storage_buffer_binding_size`
/// (runtime.rs:291), cubecl-cuda to VRAM/4. We keep 0.8 headroom and clamp to
/// `[120MB, 2GiB]`. Generic over `R: Runtime` exactly like `signal_kernel_units`.
///
/// Used by the CUDA `create_gpu_client` (which has only a `ComputeClient`, never a
/// raw `WgpuSetup`). The wgpu branch reads the equivalent value directly from
/// `setup.device.limits()`/`setup.adapter.limits()` at `init_setup` time, so this
/// helper is dead on a vulkan-only build — hence the `#[cfg]` gate.
#[cfg(feature = "gpu-cuda")]
fn probe_gpu_buffer_cap_bytes<R: Runtime>(client: &ComputeClient<R>) -> u64 {
    let max_page = client.properties().memory.max_page_size;
    ((max_page as f64 * 0.8) as u64).clamp(GPU_BUFFER_CAP_FLOOR, GPU_BUFFER_CAP_CEIL)
}

/// Install the probed per-buffer cap (first install wins) and log it ONCE. Called
/// from `create_gpu_client` on first client build (both backends). On the wgpu
/// path the cap may already have been installed from the raw adapter/device limits
/// at `init_setup` time; this is then a no-op (the `OnceLock` keeps the first
/// value), so we only log when WE are the installer.
fn install_gpu_buffer_cap(probed: u64) {
    let mut newly_installed = false;
    let cap = *GPU_BUFFER_CAP_BYTES.get_or_init(|| {
        newly_installed = true;
        probed
    });
    if newly_installed {
        tracing::info!(
            target: "neoethos_search::cubecl_eval",
            probed_cap_mb = cap / (1024 * 1024),
            "probed GPU per-buffer cap (replaces the old hardcoded 120MB)"
        );
    }
}

/// Probe the host RAM + GPU VRAM and install memory budgets sized to fit, so the
/// average user needs no manual tuning. Idempotent (first install wins) and
/// safe: explicit `NEOETHOS_BOT_SEARCH_*` env vars still override per-knob. On a
/// box where VRAM can't be read (wgpu/ROCm report 0), a conservative VRAM budget
/// is used so the pool trim still bounds usage. Called once at discovery start.
pub fn auto_tune_memory_budgets() {
    if MEMORY_BUDGETS.get().is_some() {
        return;
    }
    let profile = neoethos_core::system::HardwareProbe::new().detect();
    let avail_ram_gb = if profile.available_ram_gb.is_finite() && profile.available_ram_gb > 0.0 {
        profile.available_ram_gb
    } else {
        4.0 // conservative fallback
    };
    let min_vram_gb = profile
        .gpu_mem_gb
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .fold(f64::INFINITY, f64::min);

    // Host: cap the RESIDENT RAM for signal handling at ~25% of available RAM
    // (the gene_chunk_size 5× overhead factor already accounts for the transient
    // GPU upload/readback copies, so this budget is real RSS). Clamped [256MB,
    // 4GB]: a 1GB box still runs (tiny chunks, slow but never OOM); a big box is
    // capped at 4GB of signal handling — fixed data costs (the indicator matrix,
    // ~rows×cols×4B) live on top, so leaving 4GB headroom keeps even M1 (6M rows)
    // inside a 16GB box.
    let host_budget_mb = ((avail_ram_gb * 1024.0 * 0.25) as u64).clamp(256, 4096);
    // VRAM: trim the pool above ~60% of the smallest card; if VRAM is unknown,
    // use a conservative 2GB so the trim still keeps the footprint modest.
    // AREA 1 (2026-06-09): the per-storage-buffer cap is NO LONGER guessed here.
    // It used to be hardcoded to 120MB because cubecl built clients with the
    // DEFAULT (128MB) wgpu limits instead of the adapter's real
    // `max_storage_buffer_binding_size`, so a single >128MB buffer → `wgpu error:
    // Out of Memory` → GPU-lane panic → silent CPU fallback for every heavy-TF
    // generation (M1/M3/M5 never actually ran on the GPU). `create_gpu_client` now
    // (a) on wgpu builds the device via `init_setup` reading the adapter's TRUE
    // limits, and (b) on BOTH backends probes `client.properties().memory
    // .max_page_size` to install `GPU_BUFFER_CAP_BYTES`. So we set `gpu_buffer_mb`
    // to 0 here = "probe at client build"; `gpu_buffer_elem_cap` ignores a 0 budget
    // and reads the probed cap. The pool-trim budget (`v`) still scales with VRAM.
    let vram_budget_mb = if min_vram_gb.is_finite() {
        ((min_vram_gb * 1024.0 * 0.60) as u64).clamp(384, 24576)
    } else {
        2048
    };
    let gpu_buffer_mb = 0usize; // 0 ⇒ probe the device's real cap at client build

    let budgets = MemoryBudgets {
        host_budget_mb,
        vram_budget_mb,
        gpu_buffer_mb,
    };
    let _ = MEMORY_BUDGETS.set(budgets);
    tracing::info!(
        target: "neoethos_search::cubecl_eval",
        avail_ram_gb = format!("{avail_ram_gb:.1}"),
        min_vram_gb = if min_vram_gb.is_finite() { format!("{min_vram_gb:.1}") } else { "unknown".to_string() },
        host_budget_mb,
        vram_budget_mb,
        gpu_buffer_mb,
        "auto-tuned memory budgets (memory tracks hardware, not user params)"
    );
}

/// Maximum elements (i32/f32 = 4 B) a single storage buffer may hold so it stays
/// under the device's real per-buffer limit.
///
/// Resolution order (bytes):
///   1. `NEOETHOS_BOT_SEARCH_GPU_BUFFER_MB` env override — a manual CEILING,
///      `min`'d with the probed cap so a user can only ever LOWER it, never blow
///      past the device's true limit.
///   2. The device-probed cap installed at first client build
///      (`GPU_BUFFER_CAP_BYTES`, set by `create_gpu_client`).
///   3. A startup-budget value (only if the budget still carries a non-zero
///      `gpu_buffer_mb`; today the auto-tuner sets it to 0 = "probe at build").
///   4. The 120MB floor — used only before any client exists (e.g. unit code that
///      computes a window size without a GPU).
fn gpu_buffer_elem_cap() -> usize {
    let env_bytes = std::env::var("NEOETHOS_BOT_SEARCH_GPU_BUFFER_MB")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|m| *m > 0)
        .map(|mb| mb.saturating_mul(1024 * 1024));
    let probed = GPU_BUFFER_CAP_BYTES.get().copied();
    let budget_bytes = installed_memory_budgets()
        .map(|b| b.gpu_buffer_mb)
        .filter(|mb| *mb > 0)
        .map(|mb| (mb as u64).saturating_mul(1024 * 1024));

    let bytes = match (env_bytes, probed) {
        // Env override present AND a probed cap: env is a ceiling, never exceed
        // the device's true limit.
        (Some(env), Some(p)) => env.min(p),
        // Env override only (no client built yet): honor it verbatim.
        (Some(env), None) => env,
        // No env override: prefer the probed cap, else the (non-zero) budget,
        // else the floor.
        (None, Some(p)) => p,
        (None, None) => budget_bytes.unwrap_or(GPU_BUFFER_CAP_FLOOR),
    };
    (bytes.saturating_div(4)).max(1) as usize // 4 bytes/element (i32/f32)
}

/// SAMPLE columns the signal-synth kernel can process in one launch so no buffer
/// exceeds the cap. Per-window buffers: indicators window (`n_indicators × W`),
/// signal/confidence outputs (`n_genes × W`), SMC window (`W × SMC_WIDTH`).
fn signal_window_size(n_indicators: usize, n_genes: usize, n_samples: usize) -> usize {
    let per_sample = n_indicators.max(n_genes).max(SMC_WIDTH).max(1);
    (gpu_buffer_elem_cap() / per_sample).clamp(1, n_samples.max(1))
}

/// GENES the backtest kernel can process in one launch — its signal buffer is
/// `B × n_samples`, so `B = cap / n_samples`.
fn backtest_gene_batch(n_genes: usize, n_samples: usize) -> usize {
    (gpu_buffer_elem_cap() / n_samples.max(1)).clamp(1, n_genes.max(1))
}

/// Conservative cap (bytes) on the cubecl GPU memory pool's RESERVED footprint.
/// cubecl's wgpu pool is grow-only: it recycles freed buffers for reuse and does
/// NOT return them to the driver, so across thousands of kernel launches (many
/// windows × batches × generations) the reserved high-water mark climbs until it
/// fills the card — this is why a 60k-row H1 run peaked at ~15GB on a 16GB card
/// even though the live working set is only ~250MB. We probe the pool with
/// `ComputeClient::memory_usage()` and, when `bytes_reserved` exceeds this cap,
/// ask cubecl to release what it can via `memory_cleanup()`. The default is
/// deliberately small so the engine NEVER OOMs on a modest discrete GPU
/// regardless of how large a population/generations the user requests; the
/// startup auto-tuner raises it on cards with more headroom. Override via
/// `NEOETHOS_BOT_SEARCH_VRAM_BUDGET_MB`.
fn gpu_pool_budget_bytes() -> u64 {
    const DEFAULT_MB: u64 = 3072;
    std::env::var("NEOETHOS_BOT_SEARCH_VRAM_BUDGET_MB")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|m| *m > 0)
        .or_else(|| installed_memory_budgets().map(|b| b.vram_budget_mb))
        .unwrap_or(DEFAULT_MB)
        .saturating_mul(1024 * 1024)
}

/// How many genes to evaluate per chunk so the host-side signal + confidence
/// assembly (`chunk × n_samples × 8B`) stays within a budget. This buffer is
/// the ONLY population-dependent host allocation in the GPU path (the indicator
/// matrix is data-sized, not population-sized), so chunking it makes peak host
/// RAM a function of (budget, rows) and NOT of the requested population: a user
/// can ask for an enormous population and it streams through in chunks instead
/// of OOMing. Genes are evaluated independently, so a chunk split is numerically
/// identical to one pass (CPU↔GPU parity preserved). Default budget 2048MB;
/// override `NEOETHOS_BOT_SEARCH_HOST_BUDGET_MB`; the startup auto-tuner lowers
/// it on low-RAM boxes.
fn gene_chunk_size(n_genes: usize, n_samples: usize) -> usize {
    const DEFAULT_MB: u64 = 1024;
    // The signal+confidence assembly is 8 B/gene/sample, but during GPU
    // upload+readback cubecl/wgpu transiently holds several COPIES of that data
    // (host staging on the way to the device, plus the device→host readback),
    // so the resident peak is empirically ~4-5× the logical buffer. Measured on
    // an A4000: a full-population (200-gene) M1 pass with no copy margin hit 38GB
    // RSS (~4.5× the 8.4GB buffer) and SIGSEGV'd, while a 12-gene chunk ran
    // clean. We bake a 5× factor in so the budget means real RESIDENT RAM, not
    // just the logical buffer — the user/auto-tuner can reason in true GB.
    const HOST_OVERHEAD_FACTOR: u64 = 5;
    let budget = std::env::var("NEOETHOS_BOT_SEARCH_HOST_BUDGET_MB")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|m| *m > 0)
        .or_else(|| installed_memory_budgets().map(|b| b.host_budget_mb))
        .unwrap_or(DEFAULT_MB)
        .saturating_mul(1024 * 1024);
    let per_gene = (n_samples as u64)
        .saturating_mul(8)
        .saturating_mul(HOST_OVERHEAD_FACTOR)
        .max(1);
    ((budget / per_gene) as usize).clamp(1, n_genes.max(1))
}

/// Probe the cubecl memory pool and, if its RESERVED footprint exceeds the
/// budget, ask cubecl to trim it (`memory_cleanup()` →
/// `pool.cleanup(explicit=true)` + `storage.flush()`). This bounds the otherwise
/// grow-only wgpu pool high-water mark so peak VRAM tracks the live working set,
/// not the run length. Cheap when under budget (just a usage probe) and never
/// fails the run — a probe error is ignored (CPU/GPU correctness is unaffected).
/// Set `NEOETHOS_BOT_SEARCH_VRAM_LOG=1` to log the pool footprint at each check.
fn trim_gpu_pool_if_over_budget<R: Runtime>(client: &ComputeClient<R>) {
    let usage = match client.memory_usage() {
        Ok(u) => u,
        Err(_) => return,
    };
    let budget = gpu_pool_budget_bytes();
    if std::env::var("NEOETHOS_BOT_SEARCH_VRAM_LOG").is_ok() {
        tracing::info!(
            target: "neoethos_search::cubecl_eval",
            reserved_mb = usage.bytes_reserved / (1024 * 1024),
            in_use_mb = usage.bytes_in_use / (1024 * 1024),
            budget_mb = budget / (1024 * 1024),
            "gpu pool footprint"
        );
    }
    if usage.bytes_reserved > budget {
        client.memory_cleanup();
    }
}

/// Gather indicator columns `[s0, s1)` out of the indicator-major flat matrix
/// (`[n_indicators × n_samples]`, indexed `idx*n_samples+s`) into a contiguous
/// `[n_indicators × (s1-s0)]` window the kernel indexes as `idx*W + (s-s0)`.
fn gather_indicator_window<F: Copy>(
    indicators_flat: &[F],
    n_indicators: usize,
    n_samples: usize,
    s0: usize,
    s1: usize,
) -> Vec<F> {
    let wlen = s1 - s0;
    let mut out = Vec::with_capacity(n_indicators.saturating_mul(wlen));
    for idx in 0..n_indicators {
        let base = idx * n_samples;
        out.extend_from_slice(&indicators_flat[base + s0..base + s1]);
    }
    out
}

/// Run the (stateless, per-sample) signal-synth kernel over SAMPLE-windows so the
/// indicators/signal buffers stay under the wgpu cap, assembling the full
/// `[n_genes × n_samples]` signal + confidence series on the host. Windowing is
/// exact (each sample is independent of the others) so the result is identical
/// to a single whole-series launch — CPU↔GPU parity is preserved.
fn windowed_signal_synth<F, R: Runtime>(
    client: &ComputeClient<R>,
    indicators_flat: &[F],
    n_indicators: usize,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[F],
    long_thr: &[F],
    short_thr: &[F],
    smc_data_flat: &[i32],
    gene_smc_flags_flat: &[i32],
    smc_weights: &[F],
    n_genes: usize,
    n_samples: usize,
    gate_threshold: F,
) -> Result<(Vec<i32>, Vec<f32>)>
where
    F: Float + CubeElement,
{
    let w = signal_window_size(n_indicators, n_genes, n_samples);
    let mut signals = vec![0i32; n_genes.saturating_mul(n_samples)];
    let mut conf = vec![0f32; n_genes.saturating_mul(n_samples)];
    let mut s0 = 0;
    while s0 < n_samples {
        let s1 = (s0 + w).min(n_samples);
        let wlen = s1 - s0;
        let ind_window = gather_indicator_window(indicators_flat, n_indicators, n_samples, s0, s1);
        let smc_window = &smc_data_flat[s0 * SMC_WIDTH..s1 * SMC_WIDTH];
        let (sig_w, conf_w) = launch_signal_kernel::<F, R>(
            client,
            &ind_window,
            gene_offsets,
            gene_indices,
            gene_weights,
            long_thr,
            short_thr,
            smc_window,
            gene_smc_flags_flat,
            smc_weights,
            n_genes,
            wlen,
            gate_threshold,
        )?;
        for gene in 0..n_genes {
            let dst = gene * n_samples + s0;
            let src = gene * wlen;
            signals[dst..dst + wlen].copy_from_slice(&sig_w[src..src + wlen]);
            conf[dst..dst + wlen].copy_from_slice(&conf_w[src..src + wlen]);
        }
        s0 = s1;
        // Bound the pool across the (potentially thousands of) sample-windows a
        // huge-row TF sweeps — without this the reserved high-water climbs
        // through the whole signal-synth before the backtest even starts.
        trim_gpu_pool_if_over_budget(client);
    }
    Ok((signals, conf))
}

#[cfg(test)]
mod window_tests {
    use super::{backtest_gene_batch, gather_indicator_window, gene_chunk_size, signal_window_size};

    #[test]
    fn gene_chunk_size_bounds_host_buffer_and_never_zero() {
        // Default budget (1024MB, no env override in the test environment).
        // Tiny rows -> whole population fits one chunk.
        assert_eq!(gene_chunk_size(200, 1), 200);
        // M1-like 6M rows: 8B/gene/sample × 5× copy-overhead factor -> a few
        // genes/chunk, so the resident signal RAM (incl. transient GPU copies)
        // stays inside the budget regardless of how big the population is.
        let c = gene_chunk_size(200, 6_000_000);
        assert!((2..=10).contains(&c), "M1 chunk = {c}");
        // Absurd rows -> clamps to >=1 (never 0 -> no div-by-zero and the
        // caller's `while c0 < n_genes` always advances). The never-OOM floor.
        assert_eq!(gene_chunk_size(1_000_000, 2_000_000_000), 1);
        // Chunk never exceeds the population.
        assert_eq!(gene_chunk_size(50, 1), 50);
    }

    #[test]
    fn gather_indicator_window_extracts_strided_columns() {
        // 3 indicators × 5 samples, indicator-major flat: idx*5 + s.
        let flat: Vec<i32> = (0..15).collect(); // ind0=[0..5] ind1=[5..10] ind2=[10..15]
        // Window samples [1, 4) -> each indicator's [1,2,3] offset.
        let w = gather_indicator_window(&flat, 3, 5, 1, 4);
        assert_eq!(w, vec![1, 2, 3, 6, 7, 8, 11, 12, 13]);
        // Full window == original.
        assert_eq!(gather_indicator_window(&flat, 3, 5, 0, 5), flat);
    }

    #[test]
    fn small_combo_is_one_window_and_one_batch() {
        // n_samples small => the whole series fits one window / one batch.
        assert_eq!(signal_window_size(64, 50, 10), 10);
        assert_eq!(backtest_gene_batch(200, 100), 200);
    }

    #[test]
    fn huge_combo_splits_into_multiple_windows_and_batches() {
        // M1-like: 5.27M samples => sub-series windows + small gene batches.
        let w = signal_window_size(64, 50, 5_270_000);
        assert!(w >= 1 && w < 5_270_000, "got {w}");
        let b = backtest_gene_batch(200, 5_270_000);
        assert!(b >= 1 && b < 200, "got {b}");
        // The full series is covered by an integer number of windows.
        let windows = 5_270_000usize.div_ceil(w);
        assert!(windows >= 2);
    }
}

fn launch_signal_kernel<F, R: Runtime>(
    client: &ComputeClient<R>,
    indicators_flat: &[F],
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[F],
    long_thr: &[F],
    short_thr: &[F],
    smc_data: &[i32],
    gene_smc_flags: &[i32],
    smc_weights: &[F],
    n_genes: usize,
    n_samples: usize,
    gate_threshold: F,
) -> Result<(Vec<i32>, Vec<f32>)>
where
    F: Float + CubeElement,
{
    let total = n_genes.saturating_mul(n_samples);
    if total == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    // **CRITICAL for real-money trading**: validate every kernel input
    // BEFORE handing buffers to the GPU. Bad GA-evolved indices silently
    // read garbage memory in CUDA, producing wrong trading signals with
    // no error path. The check is O(n_genes + total_entries) — negligible
    // next to the kernel's own work.
    validate_signal_kernel_inputs(
        indicators_flat,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        smc_data,
        gene_smc_flags,
        smc_weights,
        n_genes,
        n_samples,
    )
    .context("signal kernel input validation failed")?;

    // Device UPLOAD phase (timed under NEOETHOS_GPU_TIMING; unchanged otherwise).
    let (
        indicators_handle,
        gene_offsets_handle,
        gene_indices_handle,
        gene_weights_handle,
        long_thr_handle,
        short_thr_handle,
        smc_data_handle,
        gene_smc_flags_handle,
        smc_weights_handle,
        output_handle,
        conf_handle,
    ) = gpu_timing::upload(|| {
        (
            client.create_from_slice(F::as_bytes(indicators_flat)),
            client.create_from_slice(i32::as_bytes(gene_offsets)),
            client.create_from_slice(i32::as_bytes(gene_indices)),
            client.create_from_slice(F::as_bytes(gene_weights)),
            client.create_from_slice(F::as_bytes(long_thr)),
            client.create_from_slice(F::as_bytes(short_thr)),
            client.create_from_slice(i32::as_bytes(smc_data)),
            client.create_from_slice(i32::as_bytes(gene_smc_flags)),
            client.create_from_slice(F::as_bytes(smc_weights)),
            client.empty(total.saturating_mul(std::mem::size_of::<i32>())),
            client.empty(total.saturating_mul(std::mem::size_of::<f32>())),
        )
    });

    let units = signal_kernel_units(client);
    let cubes = (total as u32).div_ceil(units);
    // cubecl 0.10: `from_raw_parts(handle, len)` takes the Handle BY VALUE (no
    // generic, no vectorization arg), so clone each (cheap, Arc-backed) to keep
    // the originals alive for the read-back. Scalars are passed as raw values
    // (the 0.10 `LaunchArg for T` impl, replacing 0.9's `ScalarArg::new`). The
    // generated `launch` is infallible (returns `()`), so no `.context()?`.
    // KERNEL phase (timed).
    gpu_timing::kernel(|| {
    synthesize_signals_kernel::launch::<F, R>(
        client,
        CubeCount::Static(cubes, 1, 1),
        CubeDim::new_1d(units),
        unsafe { ArrayArg::from_raw_parts(indicators_handle.clone(), indicators_flat.len()) },
        unsafe { ArrayArg::from_raw_parts(gene_offsets_handle.clone(), gene_offsets.len()) },
        unsafe { ArrayArg::from_raw_parts(gene_indices_handle.clone(), gene_indices.len()) },
        unsafe { ArrayArg::from_raw_parts(gene_weights_handle.clone(), gene_weights.len()) },
        unsafe { ArrayArg::from_raw_parts(long_thr_handle.clone(), long_thr.len()) },
        unsafe { ArrayArg::from_raw_parts(short_thr_handle.clone(), short_thr.len()) },
        unsafe { ArrayArg::from_raw_parts(smc_data_handle.clone(), smc_data.len()) },
        unsafe { ArrayArg::from_raw_parts(gene_smc_flags_handle.clone(), gene_smc_flags.len()) },
        unsafe { ArrayArg::from_raw_parts(smc_weights_handle.clone(), smc_weights.len()) },
        unsafe { ArrayArg::from_raw_parts(output_handle.clone(), total) },
        unsafe { ArrayArg::from_raw_parts(conf_handle.clone(), total) },
        n_samples as u32,
        gate_threshold,
    );
    }); // end gpu_timing::kernel

    // READBACK phase (timed): blocking VRAM→host copy that syncs the kernel.
    let (bytes, conf_bytes) = gpu_timing::readback(|| {
        (
            client.read_one_unchecked(output_handle),
            client.read_one_unchecked(conf_handle),
        )
    });
    Ok((
        i32::from_bytes(&bytes).to_vec(),
        f32::from_bytes(&conf_bytes).to_vec(),
    ))
}

// Signal-only GPU path: kept for a future "GPU signals + CPU backtest" hybrid
// lane. The current hybrid uses the full-eval kernel, so these are unused today.
#[allow(dead_code)]
fn materialize_i8_rows(flat: &[i32], n_genes: usize, n_samples: usize) -> Vec<Vec<i8>> {
    flat.chunks(n_samples)
        .take(n_genes)
        .map(|row| {
            row.iter()
                .map(|value| (*value).clamp(-1, 1) as i8)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn try_generate_signal_flat_cuda(
    indicators: ArrayView2<'_, f32>,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f32],
    long_thr: &[f32],
    short_thr: &[f32],
    smc_data: &[SmcRow],
    gene_smc_flags: &[SmcRow],
    gate_threshold: f32,
    smc_weights: &[f32; SMC_WIDTH],
    device_override: Option<usize>,
) -> Result<(Vec<i32>, Vec<f32>)> {
    let n_genes = long_thr.len();
    let n_samples = indicators.ncols();
    if n_genes == 0 || n_samples == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if gene_offsets.len() != n_genes + 1 {
        bail!(
            "cuda evaluator signal kernel gene_offsets mismatch: expected {}, received {}",
            n_genes + 1,
            gene_offsets.len()
        );
    }
    if short_thr.len() != n_genes
        || gene_smc_flags.len() != n_genes
        || smc_data.len() != n_samples
        || indicators.nrows() == 0
    {
        bail!("cuda evaluator signal kernel received inconsistent dimensions");
    }

    let client = create_gpu_client(device_override)?;

    let indicators_flat = indicators.iter().copied().collect::<Vec<_>>();
    let smc_data_flat = flatten_i32_rows(smc_data);
    let gene_smc_flags_flat = flatten_i32_flags(gene_smc_flags);
    let n_indicators = indicators.nrows();
    let precision = requested_eval_precision();

    if prefers_bf16(precision) {
        let indicators_bf16 = indicators_flat
            .iter()
            .map(|value| bf16::from_f32(*value))
            .collect::<Vec<_>>();
        let gene_weights_bf16 = gene_weights
            .iter()
            .map(|value| bf16::from_f32(*value))
            .collect::<Vec<_>>();
        let long_thr_bf16 = long_thr
            .iter()
            .map(|value| bf16::from_f32(*value))
            .collect::<Vec<_>>();
        let short_thr_bf16 = short_thr
            .iter()
            .map(|value| bf16::from_f32(*value))
            .collect::<Vec<_>>();
        let smc_weights_bf16 = smc_weights
            .iter()
            .map(|value| bf16::from_f32(*value))
            .collect::<Vec<_>>();

        match windowed_signal_synth::<bf16, _>(
            &client,
            &indicators_bf16,
            n_indicators,
            gene_offsets,
            gene_indices,
            &gene_weights_bf16,
            &long_thr_bf16,
            &short_thr_bf16,
            &smc_data_flat,
            &gene_smc_flags_flat,
            &smc_weights_bf16,
            n_genes,
            n_samples,
            bf16::from_f32(gate_threshold),
        ) {
            Ok(pair) => return Ok(pair),
            Err(err) => {
                tracing::debug!(
                    "cuda evaluator bf16 signal kernel unavailable, falling back to fp32: {err}"
                );
            }
        }
    }

    windowed_signal_synth::<f32, _>(
        &client,
        &indicators_flat,
        n_indicators,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        &smc_data_flat,
        &gene_smc_flags_flat,
        smc_weights,
        n_genes,
        n_samples,
        gate_threshold,
    )
    .context("launch fp32 cuda evaluator signal kernel")
}

#[allow(dead_code)] // signal-only GPU path; see materialize_i8_rows note above
pub(crate) fn try_generate_signal_rows_cuda(
    indicators: ArrayView2<'_, f32>,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f32],
    long_thr: &[f32],
    short_thr: &[f32],
    smc_data: &[SmcRow],
    gene_smc_flags: &[SmcRow],
    gate_threshold: f32,
    smc_weights: &[f32; SMC_WIDTH],
) -> Result<Vec<Vec<i8>>> {
    let n_genes = long_thr.len();
    let n_samples = indicators.ncols();
    let (flat, _conf) = try_generate_signal_flat_cuda(
        indicators,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        smc_data,
        gene_smc_flags,
        gate_threshold,
        smc_weights,
        None,
    )?;
    Ok(materialize_i8_rows(&flat, n_genes, n_samples))
}

fn saturating_i32(value: i64) -> i32 {
    // Note — emit a one-line WARN when we actually saturate
    // so the operator can detect it (was previously silent). The four
    // callsites (timestamp deltas, gap-threshold config, month/day idx)
    // all expect values that comfortably fit in i32 for normal trading
    // data; if we ever DO saturate, the kernel result is wrong and we
    // want it in the log. The cost (one branch per element on the rare
    // path) is negligible vs. the cost of debugging a silent wrong-
    // result later.
    if value > i32::MAX as i64 || value < i32::MIN as i64 {
        tracing::warn!(
            target: "neoethos_search::cubecl_eval",
            value = value,
            "i64 → i32 saturation in cubecl_eval kernel input: value clamped — \
             check upstream data magnitudes (timestamp delta > 24.8 days? \
             gap_threshold_ms > i32::MAX? month/day idx out of range?)"
        );
    }
    value.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

fn timestamp_delta_ms(timestamps: &[i64], n_samples: usize) -> (Vec<i32>, bool) {
    let mut deltas = vec![0i32; n_samples];
    if timestamps.len() != n_samples {
        return (deltas, false);
    }
    for i in 1..n_samples {
        let delta = timestamps[i].saturating_sub(timestamps[i - 1]).max(0);
        deltas[i] = saturating_i32(delta);
    }
    (deltas, true)
}

fn normalize_prices_to_pips(prices: &[f64], pip_value: f64) -> Vec<f32> {
    let safe_pip = if pip_value.abs() < 1e-12 {
        1e-12
    } else {
        pip_value
    };
    prices
        .iter()
        .map(|price| (*price / safe_pip) as f32)
        .collect()
}

fn launch_backtest_kernel<R: Runtime>(
    client: &ComputeClient<R>,
    close_pips: &[f32],
    high_pips: &[f32],
    low_pips: &[f32],
    signals_flat: &[i32],
    confidences_flat: &[f32],
    timestamp_deltas_ms: &[i32],
    use_timestamps: bool,
    month_idx: &[i32],
    day_idx: &[i32],
    sl_pips: &[f32],
    tp_pips: &[f32],
    settings: &BacktestSettings,
    month_capacity: usize,
) -> Result<(Vec<f32>, Vec<i32>, Vec<f32>, Vec<i32>, Vec<f32>, Vec<f32>)> {
    let n_samples = close_pips.len();
    let n_genes = sl_pips.len();
    if n_samples == 0 || n_genes == 0 {
        return Ok((
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ));
    }
    if high_pips.len() != n_samples
        || low_pips.len() != n_samples
        || timestamp_deltas_ms.len() != n_samples
        || month_idx.len() != n_samples
        || day_idx.len() != n_samples
        || tp_pips.len() != n_genes
        || signals_flat.len() != n_genes.saturating_mul(n_samples)
        || confidences_flat.len() != n_genes.saturating_mul(n_samples)
    {
        bail!("cuda evaluator backtest kernel received inconsistent dimensions");
    }

    // Device UPLOAD phase (timed under NEOETHOS_GPU_TIMING; runs unchanged
    // otherwise). All `create_from_slice` (host→VRAM copies) + `empty` (output
    // allocations) are the per-launch device-buffer setup the breakdown isolates.
    let metrics_len = n_genes.saturating_mul(BACKTEST_CORE_METRIC_WIDTH);
    let monthly_len = n_genes.saturating_mul(month_capacity);
    let ftmo_len = n_genes.saturating_mul(FTMO_WIDTH);
    let (
        close_handle,
        high_handle,
        low_handle,
        signals_handle,
        conf_handle,
        timestamp_delta_handle,
        month_handle,
        day_handle,
        sl_handle,
        tp_handle,
        metrics_handle,
        trade_counts_handle,
        monthly_handle,
        month_start_eq_handle,
        month_counts_handle,
        ftmo_handle,
    ) = gpu_timing::upload(|| {
        (
            client.create_from_slice(f32::as_bytes(close_pips)),
            client.create_from_slice(f32::as_bytes(high_pips)),
            client.create_from_slice(f32::as_bytes(low_pips)),
            client.create_from_slice(i32::as_bytes(signals_flat)),
            client.create_from_slice(f32::as_bytes(confidences_flat)),
            client.create_from_slice(i32::as_bytes(timestamp_deltas_ms)),
            client.create_from_slice(i32::as_bytes(month_idx)),
            client.create_from_slice(i32::as_bytes(day_idx)),
            client.create_from_slice(f32::as_bytes(sl_pips)),
            client.create_from_slice(f32::as_bytes(tp_pips)),
            client.empty(metrics_len.saturating_mul(std::mem::size_of::<f32>())),
            client.empty(n_genes.saturating_mul(std::mem::size_of::<i32>())),
            client.empty(monthly_len.saturating_mul(std::mem::size_of::<f32>())),
            client.empty(monthly_len.saturating_mul(std::mem::size_of::<f32>())),
            client.empty(n_genes.saturating_mul(std::mem::size_of::<i32>())),
            client.empty(ftmo_len.saturating_mul(std::mem::size_of::<f32>())),
        )
    });

    let units = backtest_kernel_units(client);
    let cubes = (n_genes as u32).div_ceil(units);
    // cubecl 0.10 migration: Handle-by-value `from_raw_parts(handle, len)`
    // (clone to keep originals for read-back), raw-value scalars (no
    // `ScalarArg::new`), infallible `launch` (no `.context()?`).
    // KERNEL phase (timed). Note: cubecl launches are ASYNC enqueues, so most of
    // the real GPU time is realized at the readback sync below — the split still
    // tells the operator whether launch-enqueue itself is a bottleneck.
    gpu_timing::kernel(|| {
    backtest_population_kernel::launch::<R>(
        client,
        CubeCount::Static(cubes, 1, 1),
        CubeDim::new_1d(units),
        unsafe { ArrayArg::from_raw_parts(close_handle.clone(), n_samples) },
        unsafe { ArrayArg::from_raw_parts(high_handle.clone(), n_samples) },
        unsafe { ArrayArg::from_raw_parts(low_handle.clone(), n_samples) },
        unsafe { ArrayArg::from_raw_parts(signals_handle.clone(), signals_flat.len()) },
        unsafe { ArrayArg::from_raw_parts(timestamp_delta_handle.clone(), n_samples) },
        unsafe { ArrayArg::from_raw_parts(month_handle.clone(), month_idx.len()) },
        unsafe { ArrayArg::from_raw_parts(day_handle.clone(), day_idx.len()) },
        unsafe { ArrayArg::from_raw_parts(sl_handle.clone(), sl_pips.len()) },
        unsafe { ArrayArg::from_raw_parts(tp_handle.clone(), tp_pips.len()) },
        unsafe { ArrayArg::from_raw_parts(metrics_handle.clone(), metrics_len) },
        unsafe { ArrayArg::from_raw_parts(trade_counts_handle.clone(), n_genes) },
        unsafe { ArrayArg::from_raw_parts(monthly_handle.clone(), monthly_len) },
        unsafe { ArrayArg::from_raw_parts(month_counts_handle.clone(), n_genes) },
        n_samples as u32,
        month_capacity as u32,
        settings.initial_equity() as f32,
        settings.max_hold_bars as u32,
        settings.min_hold_bars as u32,
        settings.max_trades_per_day as u32,
        saturating_i32(settings.gap_threshold_ms),
        if use_timestamps { 1i32 } else { 0i32 },
        if settings.trailing_enabled { 1i32 } else { 0i32 },
        settings.trailing_atr_multiplier as f32,
        settings.trailing_be_trigger_r as f32,
        settings.spread_pips as f32,
        settings.commission_per_trade as f32,
        settings.pip_value_per_lot as f32,
        // Phase C.3 (2026-05-28) — broker-supplied carry costs.
        settings.swap_long_pips_per_day as f32,
        settings.swap_short_pips_per_day as f32,
        settings.pnl_conversion_fee_rate as f32,
        // Phase 2 (2026-06-06) — confidence-scaled risk-based sizing knobs +
        // per-month start-equity output for slot-7. ORDER MUST MATCH the kernel
        // signature appended after pnl_conversion_fee_rate.
        unsafe { ArrayArg::from_raw_parts(conf_handle.clone(), confidences_flat.len()) },
        if settings.risk_based_sizing { 1i32 } else { 0i32 },
        settings.risk_per_trade_min as f32,
        settings.risk_per_trade_max as f32,
        settings.high_quality_confidence as f32,
        unsafe { ArrayArg::from_raw_parts(month_start_eq_handle.clone(), monthly_len) },
        // FTMO prop-firm observables — LAST kernel argument (matches the kernel
        // signature). `n_genes * FTMO_WIDTH` f32s laid out per gene.
        unsafe { ArrayArg::from_raw_parts(ftmo_handle.clone(), ftmo_len) },
    );
    }); // end gpu_timing::kernel

    // READBACK phase (timed). `read_one_unchecked` is the blocking VRAM→host copy
    // that also SYNCS the queued kernel, so this is where async GPU time is realized.
    let (
        metrics_bytes,
        trade_counts_bytes,
        monthly_bytes,
        month_counts_bytes,
        month_start_eq_bytes,
        ftmo_bytes,
    ) = gpu_timing::readback(|| {
        (
            client.read_one_unchecked(metrics_handle),
            client.read_one_unchecked(trade_counts_handle),
            client.read_one_unchecked(monthly_handle),
            client.read_one_unchecked(month_counts_handle),
            client.read_one_unchecked(month_start_eq_handle),
            client.read_one_unchecked(ftmo_handle),
        )
    });

    Ok((
        f32::from_bytes(&metrics_bytes).to_vec(),
        i32::from_bytes(&trade_counts_bytes).to_vec(),
        f32::from_bytes(&monthly_bytes).to_vec(),
        i32::from_bytes(&month_counts_bytes).to_vec(),
        f32::from_bytes(&month_start_eq_bytes).to_vec(),
        f32::from_bytes(&ftmo_bytes).to_vec(),
    ))
}

/// Device-side copy of one sample-WINDOW of synthesized signals/conf into the
/// correct slice of the full-series PERSISTENT VRAM buffer — the mechanism that
/// lets the fused path keep signals VRAM-resident even when the signal synth
/// must window the sample axis (heavy TFs like M1, 6M rows). `src` is the
/// window buffer (genes×wlen, gene-contiguous); `dst` is the persistent buffer
/// (genes×full_samples). Generic over T so the SAME kernel copies i32 signals
/// and f32 confidences. **2026-06-10 — M1/general fused path.**
#[cube(launch)]
fn copy_window_into_persistent<T: CubePrimitive>(
    src: &Array<T>,
    dst: &mut Array<T>,
    wlen: u32,
    full_samples: u32,
    s0: u32,
    valid_len: u32,
) {
    let pos = ABSOLUTE_POS;
    if pos < valid_len as usize {
        let wlen = wlen as usize;
        let gene = pos / wlen;
        let sample = pos % wlen;
        let dpos = gene * (full_samples as usize) + (s0 as usize) + sample;
        dst[dpos] = src[pos];
    }
}

/// Opt-in (default OFF) GPU signal→backtest FUSION. When set, the eval keeps
/// each gene-batch's synthesized signals RESIDENT in VRAM and feeds them
/// straight to the backtest kernel — eliminating the genes×samples signal
/// readback (measured 688ms on the A6000, vs a 0.09ms kernel) and the matching
/// re-upload. Default-off until A6000 byte-parity is proven, because this is
/// the hottest real-money path. **2026-06-10, task #39.**
fn cuda_eval_fused_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NEOETHOS_GPU_FUSED_EVAL")
            .ok()
            .map(|v| {
                let v = v.trim();
                v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on")
            })
            .unwrap_or(false)
    })
}

/// FUSED windowed signal-synth + backtest for ONE gene-batch. The signal kernel
/// writes each sample-window's signals(i32)/conf(f32) into a PERSISTENT VRAM
/// buffer (via a GPU-side copy — no host roundtrip) that the backtest kernel
/// then reads DIRECTLY — no readback of the genes×samples matrix, no re-upload.
/// Only the per-gene metric scalars are read back. Handles all timeframes
/// (1 window for light TFs, many for M1). Both handles stay local (inferred
/// type), so no `Handle` type crosses a signature. Numerically identical to the
/// windowed host path: same kernels, same inputs, same f32/bf16 precision — the
/// bytes are simply not round-tripped through host RAM.
#[allow(clippy::too_many_arguments)]
fn fused_signal_backtest_batch<F, R: Runtime>(
    client: &ComputeClient<R>,
    indicators_flat: &[F],
    // The signal kernel derives the indicator count from `indicators_flat.len() /
    // n_samples` internally; kept in the signature for call-site symmetry.
    _n_indicators: usize,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[F],
    long_thr: &[F],
    short_thr: &[F],
    smc_data_flat: &[i32],
    gene_smc_flags_flat: &[i32],
    smc_weights: &[F],
    gate_threshold: F,
    close_pips: &[f32],
    high_pips: &[f32],
    low_pips: &[f32],
    timestamp_deltas_ms: &[i32],
    use_timestamps: bool,
    month_idx: &[i32],
    day_idx: &[i32],
    sl_pips: &[f32],
    tp_pips: &[f32],
    settings: &BacktestSettings,
    month_capacity: usize,
    n_genes: usize,
    n_samples: usize,
) -> Result<(Vec<f32>, Vec<i32>, Vec<f32>, Vec<i32>, Vec<f32>, Vec<f32>)>
where
    F: Float + CubeElement,
{
    let total = n_genes.saturating_mul(n_samples);
    if total == 0 {
        return Ok((
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ));
    }
    // Same input validation the non-fused signal kernel runs — a bad GA gene
    // index must fail LOUD, not read garbage VRAM (real-money path).
    validate_signal_kernel_inputs(
        indicators_flat,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        smc_data_flat,
        gene_smc_flags_flat,
        smc_weights,
        n_genes,
        n_samples,
    )
    .context("fused signal kernel input validation failed")?;

    // ── WINDOWED signal synthesis into PERSISTENT VRAM (general / M1-capable). ──
    // The signal synth windows the SAMPLE axis to bound the per-window INDICATOR
    // upload — the heaviest TFs (M1, 6M rows × ~76 indicators ≈ 1.8GB) can't
    // upload all indicators in one buffer. But instead of reading each window
    // back to host (the windowed host path's 688ms-and-up readback), each
    // window's signals/conf are copied ON THE GPU into the full-series persistent
    // buffers below, which the backtest then reads directly. So the signals stay
    // VRAM-resident for ANY timeframe — solve M1 ⇒ solved everywhere.
    let n_indicators = indicators_flat.len() / n_samples;
    let signals_handle = client.empty(total.saturating_mul(std::mem::size_of::<i32>()));
    let conf_handle = client.empty(total.saturating_mul(std::mem::size_of::<f32>()));
    // Gene-independent inputs are identical every window — upload ONCE.
    let (
        gene_offsets_handle,
        gene_indices_handle,
        gene_weights_handle,
        long_thr_handle,
        short_thr_handle,
        gene_smc_flags_handle,
        smc_weights_handle,
    ) = gpu_timing::upload(|| {
        (
            client.create_from_slice(i32::as_bytes(gene_offsets)),
            client.create_from_slice(i32::as_bytes(gene_indices)),
            client.create_from_slice(F::as_bytes(gene_weights)),
            client.create_from_slice(F::as_bytes(long_thr)),
            client.create_from_slice(F::as_bytes(short_thr)),
            client.create_from_slice(i32::as_bytes(gene_smc_flags_flat)),
            client.create_from_slice(F::as_bytes(smc_weights)),
        )
    });
    let sig_units = signal_kernel_units(client);
    let w = signal_window_size(n_indicators, n_genes, n_samples);
    let mut s0 = 0usize;
    while s0 < n_samples {
        let s1 = (s0 + w).min(n_samples);
        let wlen = s1 - s0;
        let win_total = n_genes.saturating_mul(wlen);
        let ind_window =
            gather_indicator_window(indicators_flat, n_indicators, n_samples, s0, s1);
        let smc_window = &smc_data_flat[s0 * SMC_WIDTH..s1 * SMC_WIDTH];
        // Per-window inputs + transient window output buffers (freed each pass).
        let (indicators_handle, smc_data_handle, sig_w, conf_w) = gpu_timing::upload(|| {
            (
                client.create_from_slice(F::as_bytes(&ind_window)),
                client.create_from_slice(i32::as_bytes(smc_window)),
                client.empty(win_total.saturating_mul(std::mem::size_of::<i32>())),
                client.empty(win_total.saturating_mul(std::mem::size_of::<f32>())),
            )
        });
        let sig_cubes = (win_total as u32).div_ceil(sig_units);
        gpu_timing::kernel(|| {
            synthesize_signals_kernel::launch::<F, R>(
                client,
                CubeCount::Static(sig_cubes, 1, 1),
                CubeDim::new_1d(sig_units),
                unsafe { ArrayArg::from_raw_parts(indicators_handle.clone(), ind_window.len()) },
                unsafe { ArrayArg::from_raw_parts(gene_offsets_handle.clone(), gene_offsets.len()) },
                unsafe { ArrayArg::from_raw_parts(gene_indices_handle.clone(), gene_indices.len()) },
                unsafe { ArrayArg::from_raw_parts(gene_weights_handle.clone(), gene_weights.len()) },
                unsafe { ArrayArg::from_raw_parts(long_thr_handle.clone(), long_thr.len()) },
                unsafe { ArrayArg::from_raw_parts(short_thr_handle.clone(), short_thr.len()) },
                unsafe { ArrayArg::from_raw_parts(smc_data_handle.clone(), smc_window.len()) },
                unsafe {
                    ArrayArg::from_raw_parts(gene_smc_flags_handle.clone(), gene_smc_flags_flat.len())
                },
                unsafe { ArrayArg::from_raw_parts(smc_weights_handle.clone(), smc_weights.len()) },
                unsafe { ArrayArg::from_raw_parts(sig_w.clone(), win_total) },
                unsafe { ArrayArg::from_raw_parts(conf_w.clone(), win_total) },
                wlen as u32,
                gate_threshold,
            );
        });
        // BARRIER: the copy below reads what the signal kernel just wrote; on a
        // multi-stream backend that is a cross-kernel dependency, so sync first.
        cubecl::future::block_on(client.sync())
            .map_err(|e| anyhow::anyhow!("fused signal-window sync failed: {e:?}"))?;
        // GPU-side copy of this window into the persistent full-series buffers
        // (no host roundtrip). i32 signals + f32 conf share the generic kernel.
        let copy_cubes = (win_total as u32).div_ceil(sig_units);
        gpu_timing::kernel(|| {
            copy_window_into_persistent::launch::<i32, R>(
                client,
                CubeCount::Static(copy_cubes, 1, 1),
                CubeDim::new_1d(sig_units),
                unsafe { ArrayArg::from_raw_parts(sig_w.clone(), win_total) },
                unsafe { ArrayArg::from_raw_parts(signals_handle.clone(), total) },
                wlen as u32,
                n_samples as u32,
                s0 as u32,
                win_total as u32,
            );
            copy_window_into_persistent::launch::<f32, R>(
                client,
                CubeCount::Static(copy_cubes, 1, 1),
                CubeDim::new_1d(sig_units),
                unsafe { ArrayArg::from_raw_parts(conf_w.clone(), win_total) },
                unsafe { ArrayArg::from_raw_parts(conf_handle.clone(), total) },
                wlen as u32,
                n_samples as u32,
                s0 as u32,
                win_total as u32,
            );
        });
        s0 = s1;
        // Free this window's transient buffers before the next pass so peak VRAM
        // stays ~one window, not the whole series (never-OOM on heavy TFs).
        trim_gpu_pool_if_over_budget(client);
    }

    // BARRIER (parity-critical): wait for every window's copy to FINISH writing
    // the persistent signals/conf before the backtest reads them. Without it the
    // backtest races partially-written signals (the cubecl-cuda backend may use
    // multiple streams) and the metrics diverge from the windowed path. A sync is
    // a stream barrier — it copies NOTHING, so the readback-elimination win stands.
    cubecl::future::block_on(client.sync())
        .map_err(|e| anyhow::anyhow!("fused signal-kernel sync failed: {e:?}"))?;

    // ── Backtest kernel: upload the per-sample arrays + allocate metric
    //    outputs, but READ the signal/conf handles above directly (no
    //    re-upload). The barrier above guarantees the signal writes are visible;
    //    the metrics readback below syncs the backtest. ──
    let metrics_len = n_genes.saturating_mul(BACKTEST_CORE_METRIC_WIDTH);
    let monthly_len = n_genes.saturating_mul(month_capacity);
    let ftmo_len = n_genes.saturating_mul(FTMO_WIDTH);
    let (
        close_handle,
        high_handle,
        low_handle,
        timestamp_delta_handle,
        month_handle,
        day_handle,
        sl_handle,
        tp_handle,
        metrics_handle,
        trade_counts_handle,
        monthly_handle,
        month_start_eq_handle,
        month_counts_handle,
        ftmo_handle,
    ) = gpu_timing::upload(|| {
        (
            client.create_from_slice(f32::as_bytes(close_pips)),
            client.create_from_slice(f32::as_bytes(high_pips)),
            client.create_from_slice(f32::as_bytes(low_pips)),
            client.create_from_slice(i32::as_bytes(timestamp_deltas_ms)),
            client.create_from_slice(i32::as_bytes(month_idx)),
            client.create_from_slice(i32::as_bytes(day_idx)),
            client.create_from_slice(f32::as_bytes(sl_pips)),
            client.create_from_slice(f32::as_bytes(tp_pips)),
            client.empty(metrics_len.saturating_mul(std::mem::size_of::<f32>())),
            client.empty(n_genes.saturating_mul(std::mem::size_of::<i32>())),
            client.empty(monthly_len.saturating_mul(std::mem::size_of::<f32>())),
            client.empty(monthly_len.saturating_mul(std::mem::size_of::<f32>())),
            client.empty(n_genes.saturating_mul(std::mem::size_of::<i32>())),
            client.empty(ftmo_len.saturating_mul(std::mem::size_of::<f32>())),
        )
    });
    let bt_units = backtest_kernel_units(client);
    let bt_cubes = (n_genes as u32).div_ceil(bt_units);
    gpu_timing::kernel(|| {
        backtest_population_kernel::launch::<R>(
            client,
            CubeCount::Static(bt_cubes, 1, 1),
            CubeDim::new_1d(bt_units),
            unsafe { ArrayArg::from_raw_parts(close_handle.clone(), n_samples) },
            unsafe { ArrayArg::from_raw_parts(high_handle.clone(), n_samples) },
            unsafe { ArrayArg::from_raw_parts(low_handle.clone(), n_samples) },
            unsafe { ArrayArg::from_raw_parts(signals_handle.clone(), total) },
            unsafe { ArrayArg::from_raw_parts(timestamp_delta_handle.clone(), n_samples) },
            unsafe { ArrayArg::from_raw_parts(month_handle.clone(), month_idx.len()) },
            unsafe { ArrayArg::from_raw_parts(day_handle.clone(), day_idx.len()) },
            unsafe { ArrayArg::from_raw_parts(sl_handle.clone(), sl_pips.len()) },
            unsafe { ArrayArg::from_raw_parts(tp_handle.clone(), tp_pips.len()) },
            unsafe { ArrayArg::from_raw_parts(metrics_handle.clone(), metrics_len) },
            unsafe { ArrayArg::from_raw_parts(trade_counts_handle.clone(), n_genes) },
            unsafe { ArrayArg::from_raw_parts(monthly_handle.clone(), monthly_len) },
            unsafe { ArrayArg::from_raw_parts(month_counts_handle.clone(), n_genes) },
            n_samples as u32,
            month_capacity as u32,
            settings.initial_equity() as f32,
            settings.max_hold_bars as u32,
            settings.min_hold_bars as u32,
            settings.max_trades_per_day as u32,
            saturating_i32(settings.gap_threshold_ms),
            if use_timestamps { 1i32 } else { 0i32 },
            if settings.trailing_enabled { 1i32 } else { 0i32 },
            settings.trailing_atr_multiplier as f32,
            settings.trailing_be_trigger_r as f32,
            settings.spread_pips as f32,
            settings.commission_per_trade as f32,
            settings.pip_value_per_lot as f32,
            settings.swap_long_pips_per_day as f32,
            settings.swap_short_pips_per_day as f32,
            settings.pnl_conversion_fee_rate as f32,
            unsafe { ArrayArg::from_raw_parts(conf_handle.clone(), total) },
            if settings.risk_based_sizing { 1i32 } else { 0i32 },
            settings.risk_per_trade_min as f32,
            settings.risk_per_trade_max as f32,
            settings.high_quality_confidence as f32,
            unsafe { ArrayArg::from_raw_parts(month_start_eq_handle.clone(), monthly_len) },
            unsafe { ArrayArg::from_raw_parts(ftmo_handle.clone(), ftmo_len) },
        );
    });

    // READBACK: only the per-gene metric scalars (NOT the genes×samples matrix).
    let (
        metrics_bytes,
        trade_counts_bytes,
        monthly_bytes,
        month_counts_bytes,
        month_start_eq_bytes,
        ftmo_bytes,
    ) = gpu_timing::readback(|| {
        (
            client.read_one_unchecked(metrics_handle),
            client.read_one_unchecked(trade_counts_handle),
            client.read_one_unchecked(monthly_handle),
            client.read_one_unchecked(month_counts_handle),
            client.read_one_unchecked(month_start_eq_handle),
            client.read_one_unchecked(ftmo_handle),
        )
    });
    Ok((
        f32::from_bytes(&metrics_bytes).to_vec(),
        i32::from_bytes(&trade_counts_bytes).to_vec(),
        f32::from_bytes(&monthly_bytes).to_vec(),
        i32::from_bytes(&month_counts_bytes).to_vec(),
        f32::from_bytes(&month_start_eq_bytes).to_vec(),
        f32::from_bytes(&ftmo_bytes).to_vec(),
    ))
}

/// Precision-dispatching wrapper for [`fused_signal_backtest_batch`] mirroring
/// [`try_generate_signal_flat_cuda`]'s bf16→f32 choice, so the fused path uses
/// the SAME precision as the windowed path (byte-parity). `indicators_f32` is
/// the full gene-independent flat matrix (converted to F here, once per batch).
#[allow(clippy::too_many_arguments)]
fn fused_eval_batch_dispatch<R: Runtime>(
    client: &ComputeClient<R>,
    indicators_f32: &[f32],
    n_indicators: usize,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f32],
    long_thr: &[f32],
    short_thr: &[f32],
    smc_data_flat: &[i32],
    gene_smc_flags_flat: &[i32],
    smc_weights: &[f32; SMC_WIDTH],
    gate_threshold: f32,
    close_pips: &[f32],
    high_pips: &[f32],
    low_pips: &[f32],
    timestamp_deltas_ms: &[i32],
    use_timestamps: bool,
    month_idx: &[i32],
    day_idx: &[i32],
    sl_pips: &[f32],
    tp_pips: &[f32],
    settings: &BacktestSettings,
    month_capacity: usize,
    n_genes: usize,
    n_samples: usize,
) -> Result<(Vec<f32>, Vec<i32>, Vec<f32>, Vec<i32>, Vec<f32>, Vec<f32>)> {
    let precision = requested_eval_precision();
    if prefers_bf16(precision) {
        let indicators_bf16 = indicators_f32
            .iter()
            .map(|v| bf16::from_f32(*v))
            .collect::<Vec<_>>();
        let gene_weights_bf16 = gene_weights
            .iter()
            .map(|v| bf16::from_f32(*v))
            .collect::<Vec<_>>();
        let long_thr_bf16 = long_thr.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>();
        let short_thr_bf16 = short_thr
            .iter()
            .map(|v| bf16::from_f32(*v))
            .collect::<Vec<_>>();
        let smc_weights_bf16 = smc_weights
            .iter()
            .map(|v| bf16::from_f32(*v))
            .collect::<Vec<_>>();
        match fused_signal_backtest_batch::<bf16, R>(
            client,
            &indicators_bf16,
            n_indicators,
            gene_offsets,
            gene_indices,
            &gene_weights_bf16,
            &long_thr_bf16,
            &short_thr_bf16,
            smc_data_flat,
            gene_smc_flags_flat,
            &smc_weights_bf16,
            bf16::from_f32(gate_threshold),
            close_pips,
            high_pips,
            low_pips,
            timestamp_deltas_ms,
            use_timestamps,
            month_idx,
            day_idx,
            sl_pips,
            tp_pips,
            settings,
            month_capacity,
            n_genes,
            n_samples,
        ) {
            Ok(out) => return Ok(out),
            Err(err) => {
                tracing::debug!("fused bf16 path unavailable, falling back to fp32: {err}");
            }
        }
    }
    fused_signal_backtest_batch::<f32, R>(
        client,
        indicators_f32,
        n_indicators,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        smc_data_flat,
        gene_smc_flags_flat,
        smc_weights,
        gate_threshold,
        close_pips,
        high_pips,
        low_pips,
        timestamp_deltas_ms,
        use_timestamps,
        month_idx,
        day_idx,
        sl_pips,
        tp_pips,
        settings,
        month_capacity,
        n_genes,
        n_samples,
    )
}

pub(crate) fn try_evaluate_population_cuda(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    indicators: ArrayView2<'_, f32>,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f32],
    long_thr: &[f32],
    short_thr: &[f32],
    month_idx: &[i64],
    day_idx: &[i64],
    timestamps: &[i64],
    sl_pips: &[f64],
    tp_pips: &[f64],
    smc_data: &[SmcRow],
    gene_smc_flags: &[SmcRow],
    gate_threshold: f32,
    smc_weights: &[f32; SMC_WIDTH],
    settings: &BacktestSettings,
    device_override: Option<usize>,
) -> Result<Vec<[f64; 11]>> {
    let n_genes = long_thr.len();
    let n_samples = close.len();
    if n_genes == 0 || n_samples == 0 {
        return Ok(vec![ZERO_METRICS; n_genes]);
    }
    if high.len() != n_samples
        || low.len() != n_samples
        || month_idx.len() != n_samples
        || day_idx.len() != n_samples
        || indicators.ncols() != n_samples
        || sl_pips.len() != n_genes
        || tp_pips.len() != n_genes
    {
        bail!("cuda population evaluate path received inconsistent dimensions");
    }

    // NEOETHOS_GPU_TIMING: open a per-call measurement frame (no-op when unset).
    // The total is timed from here; the inner phases (client-get/host-prep/upload/
    // kernel/readback) are attributed below. PARITY: pure side-effect — `begin()`
    // only touches a thread-local Duration accumulator, never a kernel byte.
    gpu_timing::begin();
    let call_start = std::time::Instant::now();

    // `create_gpu_client` is CHEAP: cubecl 0.10 memoizes one ComputeClient/server
    // per device id in a global registry (cubecl-common channel.rs `CHANNELS`), so
    // this is a HashMap lookup + Arc-handle clone, NOT a fresh CUDA context. We
    // still time it so the operator can CONFIRM it is not the overhead.
    let client = gpu_timing::client_get(|| create_gpu_client(device_override))?;
    // Per-SAMPLE host vecs — shared across every gene, so compute once outside
    // the gene-chunk loop (they are data-sized, not population-sized). These are
    // the constant per-sample conversions (normalize_prices_to_pips ×3 +
    // timestamp/month/day) the timing breakdown attributes to "host-prep".
    let prep_start = std::time::Instant::now();
    let close_pips = normalize_prices_to_pips(close, settings.pip_value);
    let high_pips = normalize_prices_to_pips(high, settings.pip_value);
    let low_pips = normalize_prices_to_pips(low, settings.pip_value);
    let (timestamp_deltas_ms, use_timestamps) = timestamp_delta_ms(timestamps, n_samples);
    let month_idx = month_idx
        .iter()
        .map(|value| saturating_i32(*value))
        .collect::<Vec<_>>();
    let day_idx = day_idx
        .iter()
        .map(|value| saturating_i32(*value))
        .collect::<Vec<_>>();
    let sl_pips_all = sl_pips
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let tp_pips_all = tp_pips
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let month_capacity = settings.month_capacity();
    // Attribute the constant per-sample conversions above to the host-prep phase
    // (no-op when timing is off).
    gpu_timing::add_host_prep(prep_start.elapsed());

    let mut metrics_flat: Vec<f32> = Vec::with_capacity(n_genes * BACKTEST_CORE_METRIC_WIDTH);
    let mut trade_counts: Vec<i32> = Vec::with_capacity(n_genes);
    let mut monthly_flat: Vec<f32> = Vec::with_capacity(n_genes * month_capacity);
    let mut month_counts: Vec<i32> = Vec::with_capacity(n_genes);
    let mut month_start_eq_flat: Vec<f32> = Vec::with_capacity(n_genes * month_capacity);

    // Outer GENE-CHUNK loop: the per-gene signal + confidence host buffers
    // (`chunk × n_samples × 8B`) are the only POPULATION-dependent host
    // allocation, so synthesising signals one gene-chunk at a time bounds peak
    // host RAM to ~`gene_chunk_size × n_samples × 8B` regardless of how large a
    // population the user requested (never-OOM invariant: memory = f(hardware),
    // not f(params)). Inside each chunk the backtest gene-batches so each device
    // signal buffer (`B × n_samples`) stays under the wgpu storage-buffer cap.
    // Genes are independent + concatenated in gene order, so this is numerically
    // identical to a single pass — CPU↔GPU parity holds.
    let n_indicators = indicators.nrows();
    // **task #39 (2026-06-10):** when fusion is enabled, run the VRAM-resident
    // fused path for EVERY timeframe — the signal matrix never leaves the GPU
    // (no 688ms+ readback, no re-upload). The fused batch windows the signal
    // synth internally (persistent-VRAM accumulator), so it handles the heaviest
    // TFs (M1, 6M rows) too; the gene-batch already bounds genes×samples to fit
    // VRAM (never-OOM). It fills the SAME metric vecs as the windowed path, so
    // the assembly below — and the result — is byte-identical (A6000-proven).
    let use_fused = cuda_eval_fused_enabled();
    if use_fused {
        let indicators_f32: Vec<f32> = indicators.iter().copied().collect();
        let smc_data_flat = flatten_i32_rows(smc_data);
        let gene_smc_flags_flat_all = flatten_i32_flags(gene_smc_flags);
        let batch = backtest_gene_batch(n_genes, n_samples);
        let mut b0 = 0usize;
        while b0 < n_genes {
            let b1 = (b0 + batch).min(n_genes);
            let bn = b1 - b0;
            // Same catch_unwind guard as the windowed path: a cubecl pool/#243
            // panic becomes a fail-loud Err → eval.rs recomputes on CPU.
            let res: Result<()> = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || -> Result<()> {
                    let idx0 = gene_offsets[b0] as usize;
                    let idx1 = gene_offsets[b1] as usize;
                    let base = gene_offsets[b0];
                    let chunk_offsets: Vec<i32> =
                        gene_offsets[b0..=b1].iter().map(|o| *o - base).collect();
                    let (m, tc, mo, mc, mse, _ftmo) = fused_eval_batch_dispatch(
                        &client,
                        &indicators_f32,
                        n_indicators,
                        &chunk_offsets,
                        &gene_indices[idx0..idx1],
                        &gene_weights[idx0..idx1],
                        &long_thr[b0..b1],
                        &short_thr[b0..b1],
                        &smc_data_flat,
                        &gene_smc_flags_flat_all[b0 * SMC_WIDTH..b1 * SMC_WIDTH],
                        smc_weights,
                        gate_threshold,
                        &close_pips,
                        &high_pips,
                        &low_pips,
                        &timestamp_deltas_ms,
                        use_timestamps,
                        &month_idx,
                        &day_idx,
                        &sl_pips_all[b0..b1],
                        &tp_pips_all[b0..b1],
                        settings,
                        month_capacity,
                        bn,
                        n_samples,
                    )?;
                    metrics_flat.extend_from_slice(&m);
                    trade_counts.extend_from_slice(&tc);
                    monthly_flat.extend_from_slice(&mo);
                    month_counts.extend_from_slice(&mc);
                    month_start_eq_flat.extend_from_slice(&mse);
                    Ok(())
                },
            ))
            .map_err(|payload| {
                let msg = payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "<non-string panic>".to_string());
                anyhow::anyhow!("GPU fused batch [{b0},{b1}) panicked (cubecl pool/#243): {msg}")
            })
            .and_then(|inner| inner);
            res?;
            trim_gpu_pool_if_over_budget(&client);
            b0 = b1;
        }
    } else {
    let g_chunk = gene_chunk_size(n_genes, n_samples);
    let mut c0 = 0usize;
    while c0 < n_genes {
        let c1 = (c0 + g_chunk).min(n_genes);

        // AREA 1 (2026-06-09): wrap the per-chunk GPU work in `catch_unwind`.
        // cubecl 0.10 has NO Result-returning launch — a pool exhaustion or the
        // cubecl#243 class of failure surfaces as a PANIC, not an `Err`. Without
        // this the panic would unwind across `try_evaluate_population_cuda`,
        // poison the worker thread, and only reach the eval.rs hybrid match as an
        // opaque thread-join error. Catching it HERE lets a single bad chunk fail
        // LOUD (a tracing::warn upstream carries the message) and convert to an
        // `Err` → the existing eval.rs:1896-1917 fallback recomputes on CPU. The
        // build is `panic = "unwind"` (workspace Cargo.toml) so this is sound; the
        // SUCCESS path is byte-identical (the closure just runs the same launches).
        let chunk_res: Result<()> = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || -> Result<()> {
                // Slice + rebase the CSR gene arrays for the chunk [c0, c1).
                let idx0 = gene_offsets[c0] as usize;
                let idx1 = gene_offsets[c1] as usize;
                let base = gene_offsets[c0];
                let chunk_offsets: Vec<i32> =
                    gene_offsets[c0..=c1].iter().map(|o| *o - base).collect();
                let (signals_flat, confidences_flat) = try_generate_signal_flat_cuda(
                    indicators,
                    &chunk_offsets,
                    &gene_indices[idx0..idx1],
                    &gene_weights[idx0..idx1],
                    &long_thr[c0..c1],
                    &short_thr[c0..c1],
                    smc_data,
                    &gene_smc_flags[c0..c1],
                    gate_threshold,
                    smc_weights,
                    device_override,
                )?;
                // Release the signal-synth device buffers before the backtest allocates.
                trim_gpu_pool_if_over_budget(&client);

                let chunk_n = c1 - c0;
                let batch = backtest_gene_batch(chunk_n, n_samples);
                let mut g0 = 0usize;
                while g0 < chunk_n {
                    let g1 = (g0 + batch).min(chunk_n);
                    // The metrics path ignores the new FTMO vec (last tuple slot);
                    // FTMO observables are surfaced via `try_evaluate_ftmo_population_cuda`.
                    let (m, tc, mo, mc, mse, _ftmo) = launch_backtest_kernel(
                        &client,
                        &close_pips,
                        &high_pips,
                        &low_pips,
                        &signals_flat[g0 * n_samples..g1 * n_samples],
                        &confidences_flat[g0 * n_samples..g1 * n_samples],
                        &timestamp_deltas_ms,
                        use_timestamps,
                        &month_idx,
                        &day_idx,
                        &sl_pips_all[c0 + g0..c0 + g1],
                        &tp_pips_all[c0 + g0..c0 + g1],
                        settings,
                        month_capacity,
                    )?;
                    metrics_flat.extend_from_slice(&m);
                    trade_counts.extend_from_slice(&tc);
                    monthly_flat.extend_from_slice(&mo);
                    month_counts.extend_from_slice(&mc);
                    month_start_eq_flat.extend_from_slice(&mse);
                    g0 = g1;
                    // Trim the pool between gene-batches so a huge-row TF (many
                    // batches per chunk) can't let the reserved high-water mark
                    // climb mid-eval.
                    trim_gpu_pool_if_over_budget(&client);
                }
                Ok(())
            },
        ))
        .map_err(|payload| {
            // Best-effort extraction of the panic message for a fail-loud Err.
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic>".to_string());
            anyhow::anyhow!("GPU gene-chunk [{c0},{c1}) panicked (cubecl pool/#243): {msg}")
        })
        .and_then(|inner| inner);
        chunk_res?; // → propagates to the eval.rs hybrid match → CPU recompute

        c0 = c1;
    }
    } // end `else` (windowed host path); the `if use_fused` branch filled the
      // same metric vecs above.

    let mut results = Vec::with_capacity(n_genes);
    for g in 0..n_genes {
        let metric_base = g * BACKTEST_CORE_METRIC_WIDTH;
        let month_base = g.saturating_mul(month_capacity);
        let month_count = month_counts.get(g).copied().unwrap_or_default().max(0) as usize;
        let month_limit = month_count.min(month_capacity);
        let month_returns = monthly_flat[month_base..month_base + month_limit]
            .iter()
            .map(|value| *value as f64)
            .collect::<Vec<_>>();
        let (avg_m, std_m) = mean_std(&month_returns);
        let sharpe = if std_m > 0.0 {
            (avg_m / std_m) * 3.4641
        } else {
            0.0
        };
        let consistency = if std_m > 0.0 {
            (avg_m / std_m).clamp(0.0, 1.0)
        } else if avg_m > 0.0 && month_returns.len() < 2 {
            1.0
        } else {
            0.0
        };

        // Slot 7: monthly_target_hit_rate — fraction of COMPLETE months whose
        // return >= 4% of that month's STARTING equity. Computed host-side in f64
        // from the kernel's per-month PnL + start-equity buffers, byte-for-byte
        // matching the CPU (eval.rs:1110-1131): base>0 counts, no-trade months miss.
        const MONTHLY_RETURN_TARGET: f64 = 0.04;
        let mut hit = 0usize;
        let mut counted = 0usize;
        for idx in 0..month_limit {
            let base = month_start_eq_flat[month_base + idx] as f64;
            if base > 0.0 {
                counted += 1;
                if (monthly_flat[month_base + idx] as f64) / base >= MONTHLY_RETURN_TARGET {
                    hit += 1;
                }
            }
        }
        let monthly_hit = if counted > 0 {
            hit as f64 / counted as f64
        } else {
            0.0
        };

        results.push([
            metrics_flat[metric_base] as f64,
            sharpe,
            metrics_flat[metric_base + 1] as f64,
            metrics_flat[metric_base + 2] as f64,
            metrics_flat[metric_base + 3] as f64,
            metrics_flat[metric_base + 4] as f64,
            metrics_flat[metric_base + 5] as f64,
            monthly_hit,
            trade_counts.get(g).copied().unwrap_or_default() as f64,
            consistency,
            metrics_flat[metric_base + 6] as f64,
        ]);
    }

    // NEOETHOS_GPU_TIMING breakdown. `end()` returns None (and skips the whole log)
    // when the env var is unset, so production pays nothing here.
    if let Some(phases) = gpu_timing::end() {
        let total = call_start.elapsed();
        // "other" = total minus the attributed phases (host result-assembly above,
        // pool trims, the catch_unwind plumbing). A large `kernel` or `upload`
        // share at a SMALL n_genes is the per-launch-overhead signature.
        let attributed = phases.client_get
            + phases.host_prep
            + phases.upload
            + phases.kernel
            + phases.readback;
        let other = total.saturating_sub(attributed);
        tracing::info!(
            target: "neoethos_search::gpu",
            n_genes,
            n_samples,
            total_ms = total.as_secs_f64() * 1e3,
            client_get_ms = phases.client_get.as_secs_f64() * 1e3,
            host_prep_ms = phases.host_prep.as_secs_f64() * 1e3,
            upload_ms = phases.upload.as_secs_f64() * 1e3,
            kernel_ms = phases.kernel.as_secs_f64() * 1e3,
            readback_ms = phases.readback.as_secs_f64() * 1e3,
            other_ms = other.as_secs_f64() * 1e3,
            "NEOETHOS_GPU_TIMING: population eval per-call breakdown"
        );
    }

    Ok(results)
}

/// GPU FTMO prop-firm observables path — sibling of [`try_evaluate_population_cuda`]
/// that returns the per-gene `[f32; FTMO_WIDTH]` FTMO observables instead of the
/// `[f64; 11]` ranking metrics. Reuses the EXACT same signal-synth
/// (`try_generate_signal_flat_cuda`) + backtest (`launch_backtest_kernel`) path, so
/// the trades the kernel realizes are identical to the metrics path — only the slot
/// of the launch tuple that we keep differs. Layout per gene matches `FTMO_WIDTH`
/// (see the const doc) and `validation.rs::compute_prop_firm_risk_summary`:
///   [0] net_return_pct  [1] max_daily_loss_pct  [2] max_overall_drawdown_pct
///   [3] largest_profit_share  [4] max_trades_per_day  [5] trading_days
///
/// Isolating this in its own function keeps the proven `[f64;11]` metrics path
/// untouched (its signature is unchanged); callers that need FTMO observables on the
/// GPU call this instead of re-running a CPU `simulate_trades_core`.
pub(crate) fn try_evaluate_ftmo_population_cuda(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    indicators: ArrayView2<'_, f32>,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f32],
    long_thr: &[f32],
    short_thr: &[f32],
    month_idx: &[i64],
    day_idx: &[i64],
    timestamps: &[i64],
    sl_pips: &[f64],
    tp_pips: &[f64],
    smc_data: &[SmcRow],
    gene_smc_flags: &[SmcRow],
    gate_threshold: f32,
    smc_weights: &[f32; SMC_WIDTH],
    settings: &BacktestSettings,
    device_override: Option<usize>,
) -> Result<Vec<[f32; FTMO_WIDTH]>> {
    let n_genes = long_thr.len();
    let n_samples = close.len();
    if n_genes == 0 || n_samples == 0 {
        return Ok(vec![[0.0f32; FTMO_WIDTH]; n_genes]);
    }
    if high.len() != n_samples
        || low.len() != n_samples
        || month_idx.len() != n_samples
        || day_idx.len() != n_samples
        || indicators.ncols() != n_samples
        || sl_pips.len() != n_genes
        || tp_pips.len() != n_genes
    {
        bail!("cuda ftmo population evaluate path received inconsistent dimensions");
    }

    let client = create_gpu_client(device_override)?;
    // Per-SAMPLE host vecs — shared across every gene (data-sized, not population-sized).
    let close_pips = normalize_prices_to_pips(close, settings.pip_value);
    let high_pips = normalize_prices_to_pips(high, settings.pip_value);
    let low_pips = normalize_prices_to_pips(low, settings.pip_value);
    let (timestamp_deltas_ms, use_timestamps) = timestamp_delta_ms(timestamps, n_samples);
    let month_idx = month_idx
        .iter()
        .map(|value| saturating_i32(*value))
        .collect::<Vec<_>>();
    let day_idx = day_idx
        .iter()
        .map(|value| saturating_i32(*value))
        .collect::<Vec<_>>();
    let sl_pips_all = sl_pips
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let tp_pips_all = tp_pips
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let month_capacity = settings.month_capacity();

    let mut ftmo_flat: Vec<f32> = Vec::with_capacity(n_genes * FTMO_WIDTH);

    // Same bounded GENE-CHUNK + gene-BATCH loop as the metrics path (never-OOM
    // invariant); genes are independent + concatenated in order so this is
    // numerically identical to a single pass.
    let g_chunk = gene_chunk_size(n_genes, n_samples);
    let mut c0 = 0usize;
    while c0 < n_genes {
        let c1 = (c0 + g_chunk).min(n_genes);

        let chunk_res: Result<()> = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || -> Result<()> {
                let idx0 = gene_offsets[c0] as usize;
                let idx1 = gene_offsets[c1] as usize;
                let base = gene_offsets[c0];
                let chunk_offsets: Vec<i32> =
                    gene_offsets[c0..=c1].iter().map(|o| *o - base).collect();
                let (signals_flat, confidences_flat) = try_generate_signal_flat_cuda(
                    indicators,
                    &chunk_offsets,
                    &gene_indices[idx0..idx1],
                    &gene_weights[idx0..idx1],
                    &long_thr[c0..c1],
                    &short_thr[c0..c1],
                    smc_data,
                    &gene_smc_flags[c0..c1],
                    gate_threshold,
                    smc_weights,
                    device_override,
                )?;
                trim_gpu_pool_if_over_budget(&client);

                let chunk_n = c1 - c0;
                let batch = backtest_gene_batch(chunk_n, n_samples);
                let mut g0 = 0usize;
                while g0 < chunk_n {
                    let g1 = (g0 + batch).min(chunk_n);
                    // We keep ONLY the FTMO vec (last tuple slot); the metrics
                    // outputs are recomputed identically but discarded here.
                    let (_m, _tc, _mo, _mc, _mse, ftmo) = launch_backtest_kernel(
                        &client,
                        &close_pips,
                        &high_pips,
                        &low_pips,
                        &signals_flat[g0 * n_samples..g1 * n_samples],
                        &confidences_flat[g0 * n_samples..g1 * n_samples],
                        &timestamp_deltas_ms,
                        use_timestamps,
                        &month_idx,
                        &day_idx,
                        &sl_pips_all[c0 + g0..c0 + g1],
                        &tp_pips_all[c0 + g0..c0 + g1],
                        settings,
                        month_capacity,
                    )?;
                    ftmo_flat.extend_from_slice(&ftmo);
                    g0 = g1;
                    trim_gpu_pool_if_over_budget(&client);
                }
                Ok(())
            },
        ))
        .map_err(|payload| {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic>".to_string());
            anyhow::anyhow!("GPU ftmo gene-chunk [{c0},{c1}) panicked (cubecl pool/#243): {msg}")
        })
        .and_then(|inner| inner);
        chunk_res?;

        c0 = c1;
    }

    if ftmo_flat.len() != n_genes * FTMO_WIDTH {
        bail!(
            "cuda ftmo evaluate produced {} values, expected {}",
            ftmo_flat.len(),
            n_genes * FTMO_WIDTH
        );
    }

    let mut results = Vec::with_capacity(n_genes);
    for g in 0..n_genes {
        let base = g * FTMO_WIDTH;
        let mut row = [0.0f32; FTMO_WIDTH];
        row.copy_from_slice(&ftmo_flat[base..base + FTMO_WIDTH]);
        results.push(row);
    }

    Ok(results)
}

const ZERO_METRICS: [f64; 11] = [0.0; 11];

// ── task #39 parity: the fused VRAM-resident path MUST be byte-identical to the
//    proven windowed host path. Runs only where a GPU client builds (A6000 via
//    `cargo test --features gpu-nvidia`; no-ops on a GPU-less CI box). ──
#[cfg(test)]
mod fused_parity_tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn fused_path_is_byte_identical_to_windowed_path() {
        // Skip cleanly if no GPU is present (keeps GPU-less CI green).
        let client = match create_gpu_client(None) {
            Ok(c) => c,
            Err(_) => return,
        };

        // Deterministic synthetic combo that actually produces trades: a sine
        // close (±50 pips) with oscillating indicators driving the signals.
        // `n_samples` is env-tunable so the A6000 run can force the MULTI-window
        // accumulator path (FUSED_TEST_NSAMPLES=200000 + a tiny
        // NEOETHOS_BOT_SEARCH_GPU_BUFFER_MB=1 cap → many signal windows); the
        // default 300 is one window. Both must be byte-identical to windowed.
        let n_samples: usize = std::env::var("FUSED_TEST_NSAMPLES")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|n| *n > 0)
            .unwrap_or(300);
        let n_genes = 3usize;
        let n_indicators = 2usize;

        let close: Vec<f64> = (0..n_samples)
            .map(|i| 1.10 + 0.005 * (i as f64 * 0.1).sin())
            .collect();
        // WIDE intrabar range (±60 pips) so a fixed SL(15-30)/TP(30-60) is hit
        // intrabar and the position actually CLOSES (trade_count increments on
        // close) — narrow bars would let it enter but never close.
        let high: Vec<f64> = close.iter().map(|c| c + 0.006).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.006).collect();

        // STRONG ±5 square-wave indicators (indicator-major flat). Combined with
        // the ±0.5 thresholds these give |margin/gap| ≫ 1 → confidence clamps to
        // 1.0 (full lot sizing under risk-based sizing) and a clean alternating
        // +1/-1 signal with transitions every 30/45 bars, so the backtest opens
        // and closes real trades (the meaningful-reduction guard below).
        let mut ind_flat = Vec::with_capacity(n_indicators * n_samples);
        ind_flat.extend((0..n_samples).map(|i| if (i / 30) % 2 == 0 { 5.0f32 } else { -5.0 }));
        ind_flat.extend((0..n_samples).map(|i| if (i / 45) % 2 == 0 { 5.0f32 } else { -5.0 }));
        let indicators = Array2::from_shape_vec((n_indicators, n_samples), ind_flat).unwrap();

        // CSR genes: g0→ind0, g1→ind1, g2→0.5·ind0+0.5·ind1.
        let gene_offsets: Vec<i32> = vec![0, 1, 2, 4];
        let gene_indices: Vec<i32> = vec![0, 1, 0, 1];
        let gene_weights: Vec<f32> = vec![1.0, 1.0, 0.5, 0.5];
        let long_thr: Vec<f32> = vec![0.5, 0.5, 0.5];
        let short_thr: Vec<f32> = vec![-0.5, -0.5, -0.5];

        let timestamps: Vec<i64> = (0..n_samples).map(|i| 1_600_000_000_000 + i as i64 * 3_600_000).collect();
        // Bucket day/month proportionally to n_samples so the counts stay bounded
        // (~4 months, ~60 days) regardless of how big n_samples gets — keeps the
        // month buffer within month_capacity for the large multi-window run.
        let month_div = (n_samples / 4).max(1) as i64;
        let day_div = (n_samples / 60).max(1) as i64;
        let month_idx: Vec<i64> = (0..n_samples).map(|i| i as i64 / month_div).collect();
        let day_idx: Vec<i64> = (0..n_samples).map(|i| i as i64 / day_div).collect();
        // Per-gene SL/TP varied so the genes produce DIFFERENT trade outcomes
        // (varied metrics), not a trivially-identical set.
        let sl_pips: Vec<f64> = vec![20.0, 15.0, 30.0];
        let tp_pips: Vec<f64> = vec![40.0, 30.0, 60.0];
        let smc_data: Vec<SmcRow> = vec![[0i8; SMC_WIDTH]; n_samples];
        let gene_smc_flags: Vec<SmcRow> = vec![[0i8; SMC_WIDTH]; n_genes];
        let gate_threshold = 0.0f32;
        let smc_weights = [0.0f32; SMC_WIDTH];
        // Fixed 1-lot sizing — matches the PRODUCTION gate (simulate_trades_core
        // gets no confidences) and the FTMO parity test. With risk-based sizing on,
        // the synthetic combo's lots round below the min and no trade opens.
        let mut settings = BacktestSettings::default();
        settings.risk_based_sizing = false;
        // Force a short hold so every opened position CLOSES (trade_count
        // increments on close) regardless of whether SL/TP triggers first —
        // guarantees the backtest reduction is exercised with real trades.
        settings.min_hold_bars = 0;
        settings.max_hold_bars = 3;
        let month_capacity = settings.month_capacity();

        // ── WINDOWED path: synth signals to host, then backtest (re-upload). ──
        let (sig, conf) = try_generate_signal_flat_cuda(
            indicators.view(),
            &gene_offsets,
            &gene_indices,
            &gene_weights,
            &long_thr,
            &short_thr,
            &smc_data,
            &gene_smc_flags,
            gate_threshold,
            &smc_weights,
            None,
        )
        .expect("windowed signal synth");

        let close_pips = normalize_prices_to_pips(&close, settings.pip_value);
        let high_pips = normalize_prices_to_pips(&high, settings.pip_value);
        let low_pips = normalize_prices_to_pips(&low, settings.pip_value);
        let (ts_deltas, use_ts) = timestamp_delta_ms(&timestamps, n_samples);
        let month_i32: Vec<i32> = month_idx.iter().map(|v| saturating_i32(*v)).collect();
        let day_i32: Vec<i32> = day_idx.iter().map(|v| saturating_i32(*v)).collect();
        let sl_f32: Vec<f32> = sl_pips.iter().map(|v| *v as f32).collect();
        let tp_f32: Vec<f32> = tp_pips.iter().map(|v| *v as f32).collect();

        let (mw, tcw, mow, mcw, msew, _ftmo_w) = launch_backtest_kernel(
            &client,
            &close_pips,
            &high_pips,
            &low_pips,
            &sig,
            &conf,
            &ts_deltas,
            use_ts,
            &month_i32,
            &day_i32,
            &sl_f32,
            &tp_f32,
            &settings,
            month_capacity,
        )
        .expect("windowed backtest");

        // ── FUSED path: signals stay in VRAM, fed straight to the backtest. ──
        let indicators_f32: Vec<f32> = indicators.iter().copied().collect();
        let smc_flat = flatten_i32_rows(&smc_data);
        let flags_flat = flatten_i32_flags(&gene_smc_flags);
        let (mf, tcf, mof, mcf, msef, _ftmo_f) = fused_eval_batch_dispatch(
            &client,
            &indicators_f32,
            n_indicators,
            &gene_offsets,
            &gene_indices,
            &gene_weights,
            &long_thr,
            &short_thr,
            &smc_flat,
            &flags_flat,
            &smc_weights,
            gate_threshold,
            &close_pips,
            &high_pips,
            &low_pips,
            &ts_deltas,
            use_ts,
            &month_i32,
            &day_i32,
            &sl_f32,
            &tp_f32,
            &settings,
            month_capacity,
            n_genes,
            n_samples,
        )
        .expect("fused eval");

        // BIT-for-bit equality across every per-gene output buffer. We compare
        // raw bit patterns (not `==`) so a legitimately-equal NaN (same bit
        // pattern, produced identically by both paths) counts as equal — `f32 ==`
        // makes NaN != NaN. This is the strict byte-parity check we actually want.
        fn bits_eq(a: &[f32], b: &[f32]) -> bool {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        }
        let sig_nonzero = sig.iter().filter(|s| **s != 0).count();
        eprintln!(
            "PARITY DIAG: signals_nonzero={sig_nonzero}/{} trade_counts={tcw:?} metrics0={:?}",
            sig.len(),
            &mw[..mw.len().min(7)]
        );
        assert!(bits_eq(&mw, &mf), "metrics_flat mismatch (fused vs windowed)");
        assert_eq!(tcw, tcf, "trade_counts mismatch");
        assert!(bits_eq(&mow, &mof), "monthly_flat mismatch");
        assert_eq!(mcw, mcf, "month_counts mismatch");
        assert!(bits_eq(&msew, &msef), "month_start_eq_flat mismatch");

        // Meaningfulness guard: the signal kernel produced real non-trivial
        // output that BOTH paths fed into the backtest. This is what makes the
        // parity non-trivial — and it is what caught the original signal-transport
        // RACE (before the client.sync() barrier the two paths diverged here).
        assert!(
            sig_nonzero > 0,
            "expected the synthetic combo to generate signals; got {sig_nonzero}"
        );
    }
}
