//! Calls the actual ownership API. This does not run any GPU kernel.

use neoethos_gpu_cuda::hip_runtime_v1::HipRunLeaseV1;

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.as_slice() == ["--help"] {
        println!(
            "hip_runtime_probe --ordinal <N>\nReal AMD HIP runtime ownership only; no kernels, Search, allocations, or numerical parity."
        );
        return Ok(());
    }
    if args.len() != 2 || args[0] != "--ordinal" {
        return Err("expected explicit --ordinal <N>; no automatic backend selection".into());
    }
    let ordinal = args[1].parse::<u32>()?;
    let lease = HipRunLeaseV1::acquire(ordinal)?;
    println!("scope=HIP runtime ownership only; device_kernels_executed=false");
    println!("identity={:?}", lease.identity());
    println!("memory={:?}", lease.revalidate()?);
    lease.synchronize()?;
    lease.try_close()?;
    println!("owner_closed=true; numerical_parity_verified=false");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("HIP runtime ownership probe failed: {error}");
        std::process::exit(1);
    }
}
