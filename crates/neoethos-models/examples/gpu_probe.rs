//! gpu_probe — quick end-to-end verification ότι το active burn backend
//! (Vulkan/wgpu under `--features gpu-vulkan`, ndarray-CPU otherwise)
//! όντως πιάνει το hardware και κάνει compute.
//!
//! Run:
//!   $ export VULKAN_SDK=C:/VulkanSDK/1.4.350.0   # Windows
//!   $ cargo run -p neoethos-models --example gpu_probe \
//!     --features gpu-vulkan --release
//!
//! What it does:
//!   1. Reports the active burn backend name (ndarray_cpu / vulkan_wgpu).
//!   2. Resolves the inference device via the same `resolve_infer_device`
//!      that the live trading path uses — so what you see here is what
//!      production would see at runtime.
//!   3. Allocates a real 1024×1024 f32 matrix on the device.
//!   4. Runs a single matmul (warm-up + timed pass) to prove the
//!      backend can actually execute kernels on the chosen device.
//!   5. Prints throughput (GFLOPS) so we know it's not silently
//!      falling back to CPU.
//!
//! Expected output under `--features gpu-vulkan` on an AMD/NVIDIA/Intel
//! GPU machine:
//!   backend       = vulkan_wgpu
//!   device        = WgpuDevice::DiscreteGpu(0)  (or IntegratedGpu)
//!   matmul 1024^2 = warm-up 800ms / timed 5ms / 429.5 GFLOPS
//!
//! Expected output under default features (no GPU):
//!   backend       = ndarray_cpu
//!   matmul 1024^2 = warm-up 12ms / timed 11ms / 195 GFLOPS

use std::time::Instant;

use burn::tensor::{Distribution, Tensor};
use neoethos_models::burn_models::{
    InferBackend, active_burn_backend_name, default_infer_device,
};

fn main() {
    // Backend identity — the easy half. Either "vulkan_wgpu" (or
    // "wgpu" depending on the burn version) or "ndarray_cpu". Picked
    // by Cargo features at compile time.
    println!("backend       = {}", active_burn_backend_name());

    // Device handle. Under wgpu this prints a `WgpuDevice` enum that
    // says DiscreteGpu / IntegratedGpu / Cpu — that's how we know it
    // didn't silently fall back to the wgpu CPU adapter (which would
    // be useless for our purposes).
    let device = default_infer_device();
    println!("device        = {device:?}");

    // Compute proof. A 256×256 matmul is ~33 MFLOPs of work — large
    // enough that the kernel-launch overhead doesn't dominate the
    // timing, small enough that the cubek-matmul algorithm selector
    // picks a tile size compatible with the smallest realistic
    // wgpu adapter (e.g. AMD iGPU with 64 KB shared memory per
    // workgroup vs. discrete cards' 96-100 KB).
    //
    // **2026-05-25 — sized down from 1024**: at 1024×1024 the
    // cubek-matmul tile-based fast path needs ~72 KB shared memory,
    // which exceeds AMD Radeon iGPU's 64 KB limit and panics. NeoEthos
    // production inference matmuls are 32-row × feature-count → 3 (much
    // smaller than 1024×1024), so this size is closer to the realistic
    // hot path anyway.
    const N: usize = 256;
    let shape = [N, N];
    let a: Tensor<InferBackend, 2> = Tensor::random(shape, Distribution::Default, &device);
    let b: Tensor<InferBackend, 2> = Tensor::random(shape, Distribution::Default, &device);

    // Warm-up pass. First wgpu kernel call triggers shader compile
    // (SPIR-V codegen + driver upload) which can take 500ms-2s on a
    // cold cache. We don't want that in the throughput number.
    let t0 = Instant::now();
    let warmup = a.clone().matmul(b.clone());
    let _ = warmup.into_data().as_slice::<f32>().unwrap().len(); // sync
    let warmup_ms = t0.elapsed().as_secs_f64() * 1_000.0;

    // Timed pass — kernels are now cached.
    let t1 = Instant::now();
    let out = a.matmul(b);
    let _data = out.into_data().as_slice::<f32>().unwrap().len(); // sync
    let timed_ms = t1.elapsed().as_secs_f64() * 1_000.0;

    // GFLOPS = 2·N³ floating-point ops / time
    let flops = 2.0 * (N as f64).powi(3);
    let gflops = flops / (timed_ms / 1_000.0) / 1e9;

    println!(
        "matmul {N}^2  = warm-up {warmup_ms:.1}ms / timed {timed_ms:.2}ms / {gflops:.1} GFLOPS"
    );

    // A rough sanity threshold for 256×256: any real GPU (even an
    // iGPU) does ≥1 GFLOPS once kernel-launch overhead is amortised.
    // CPU burn-ndarray does ~30-80 GFLOPS at this size. The LLVMpipe
    // software rasterizer (the "fallback adapter" we want to catch)
    // does <0.3 GFLOPS for 256² because it serialises every thread.
    // So <1 GFLOPS = wgpu fell back to software, build is useless.
    if gflops < 1.0 {
        eprintln!(
            "\n⚠ WARNING: throughput {gflops:.1} GFLOPS looks like wgpu fell back \
             to the software rasterizer. Check that vulkaninfo --summary lists a \
             real GPU device, not LLVMpipe."
        );
        std::process::exit(2);
    }
    println!("\nOK — backend is exercising real hardware.");
}
