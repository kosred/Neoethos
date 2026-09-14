//! `gpu_probe` — a fail-closed Burn CUDA compute probe.
//!
//! Run on a host with an NVIDIA driver and CUDA toolkit:
//!
//! ```text
//! cargo run -p neoethos-models --example gpu_probe \
//!   --features burn-cuda-backend --release
//! ```
//!
//! The example resolves CUDA ordinal 0 through the production resolver, checks
//! the recorded backend identity, executes two real matrix multiplications, and
//! reads the result back before reporting throughput. A CPU-only build exits
//! immediately instead of presenting CPU work as a GPU probe.

#[cfg(feature = "burn-cuda-backend")]
use std::time::Instant;

#[cfg(feature = "burn-cuda-backend")]
use burn::tensor::{Distribution, Tensor};
#[cfg(feature = "burn-cuda-backend")]
use neoethos_models::burn_models::{
    InferBackend, active_burn_backend_name, burn_cuda_residency_scope, resolve_infer_device,
};

#[cfg(not(feature = "burn-cuda-backend"))]
fn main() {
    eprintln!(
        "gpu_probe requires the native Burn CUDA backend; rerun with \
         `--features burn-cuda-backend` on an NVIDIA CUDA host"
    );
    std::process::exit(2);
}

#[cfg(feature = "burn-cuda-backend")]
fn main() {
    let _residency = burn_cuda_residency_scope(0);
    let (device, selection) = resolve_infer_device("gpu:0")
        .unwrap_or_else(|error| panic!("resolve CUDA ordinal 0 for gpu_probe: {error:#}"));

    assert_eq!(active_burn_backend_name(), "cuda");
    assert_eq!(selection.execution_backend, "cuda");
    assert_eq!(selection.effective_policy, "gpu:0");
    println!("backend       = {}", active_burn_backend_name());
    println!("device        = {device:?}");

    const N: usize = 1024;
    let shape = [N, N];
    let a: Tensor<InferBackend, 2> = Tensor::random(shape, Distribution::Default, &device);
    let b: Tensor<InferBackend, 2> = Tensor::random(shape, Distribution::Default, &device);

    let warmup_started = Instant::now();
    let warmup = a.clone().matmul(b.clone());
    let warmup_data = warmup.into_data();
    let warmup_values = warmup_data
        .as_slice::<f32>()
        .expect("CUDA warm-up result must be readable as f32");
    assert_eq!(warmup_values.len(), N * N);
    let warmup_ms = warmup_started.elapsed().as_secs_f64() * 1_000.0;

    let timed_started = Instant::now();
    let output = a.matmul(b);
    let output_data = output.into_data();
    let output_values = output_data
        .as_slice::<f32>()
        .expect("CUDA timed result must be readable as f32");
    assert_eq!(output_values.len(), N * N);
    let timed_ms = timed_started.elapsed().as_secs_f64() * 1_000.0;

    let flops = 2.0 * (N as f64).powi(3);
    let gflops = flops / (timed_ms / 1_000.0) / 1e9;
    println!(
        "matmul {N}^2  = warm-up {warmup_ms:.1}ms / timed {timed_ms:.2}ms / {gflops:.1} GFLOPS"
    );
    println!("OK — native Burn CUDA kernels executed on ordinal 0.");
}
