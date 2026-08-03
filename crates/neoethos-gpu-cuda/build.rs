use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEVICE_SOURCES: [&str; 3] = [
    "native/smoke.cu",
    "native/prototype_b.cu",
    "native/prototype_b_population.cu",
];

fn main() {
    println!("cargo:rerun-if-changed=native/neoethos_gpu_cuda.h");
    println!("cargo:rerun-if-changed=native/layout_asserts.cpp");
    println!("cargo:rerun-if-changed=native/stub.cpp");
    for source in DEVICE_SOURCES {
        println!("cargo:rerun-if-changed={source}");
    }
    println!("cargo:rerun-if-env-changed=CUDACXX");
    println!("cargo:rerun-if-env-changed=NEOETHOS_CUDA_ARCH");

    let cuda_feature = env::var_os("CARGO_FEATURE_CUDA").is_some();

    // Host C++ and CUDA are separate units on purpose. A single `cc::Build`
    // with `.cuda(true)` drives every file through nvcc — including the plain
    // `.cpp` — and hands it gcc-shaped flags (`-ffunction-sections`, `-Wall`,
    // `-gdwarf-4`, ...) that nvcc rejects outright.
    let mut host = cc::Build::new();
    host.cpp(true).std("c++17").include("native");
    host.file("native/layout_asserts.cpp");
    if !cuda_feature {
        host.file("native/stub.cpp");
    }
    host.compile("neoethos_gpu_cuda_abi");

    if cuda_feature {
        compile_device_objects();
    }
}

/// Compile the device translation units by driving nvcc directly.
///
/// cc-rs always passes `--device-c` in CUDA mode, which produces relocatable
/// device code that then *requires* a separate `nvcc -dlink` step. Cargo links
/// the resulting archive with the host linker instead, so the kernels are never
/// registered and every launch fails at runtime (observed as an unusable device
/// function on a real RTX 3060 Ti, not at build time). Whole-program
/// compilation has no such requirement, so the device unit is built here
/// explicitly rather than through cc-rs.
fn compile_device_objects() {
    let nvcc = env::var("CUDACXX").unwrap_or_else(|_| "nvcc".to_string());
    let available = Command::new(&nvcc)
        .arg("--version")
        .status()
        .map(|status| status.success())
        .unwrap_or(false);
    if !available {
        panic!("feature `cuda` requires nvcc; set CUDACXX or install the CUDA toolkit");
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    // PTX for a virtual architecture by default: the driver JITs it for
    // whatever card is actually present, so one build runs on Turing through
    // Hopper. Override for a specific card with NEOETHOS_CUDA_ARCH.
    let arch = env::var("NEOETHOS_CUDA_ARCH")
        .unwrap_or_else(|_| "compute_70,code=compute_70".to_string());
    let debug = env::var("DEBUG").map(|value| value == "true").unwrap_or(false);

    let mut objects = Vec::new();
    for source in DEVICE_SOURCES {
        let stem = Path::new(source)
            .file_stem()
            .expect("device sources have file names")
            .to_string_lossy()
            .into_owned();
        let object = out_dir.join(format!("{stem}.o"));
        let mut command = Command::new(&nvcc);
        command
            .arg("-c")
            .arg(source)
            .arg("-o")
            .arg(&object)
            .arg("-std=c++17")
            .arg(format!("-gencode=arch={arch}"))
            .arg("-Xcompiler=-fPIC")
            // No fused multiply-add contraction. Measured on an RTX A6000 over
            // 20 000 real EURUSD H1 bars: with contraction enabled one
            // candidate in 256 diverged from the canonical CPU result by 0.62 %,
            // because a fused multiply-add flipped a stop/target boundary
            // comparison. Disabling it restores exact parity at 4 096, 20 000
            // and 200 000 bars for about 2 % of throughput. Exactness is the
            // point of this engine; the 2 % is not negotiable against it.
            .arg("-fmad=false")
            .arg("-I")
            .arg("native");
        if debug {
            command.arg("-lineinfo");
        } else {
            command.arg("-O3");
        }
        let status = command
            .status()
            .unwrap_or_else(|error| panic!("failed to run {nvcc} on {source}: {error}"));
        if !status.success() {
            panic!("nvcc failed to compile {source} (status {status})");
        }
        objects.push(object);
    }

    let archive = out_dir.join("libneoethos_gpu_cuda_native.a");
    let _ = std::fs::remove_file(&archive);
    let archiver = env::var("AR").unwrap_or_else(|_| "ar".to_string());
    let status = Command::new(&archiver)
        .arg("rcs")
        .arg(&archive)
        .args(&objects)
        .status()
        .unwrap_or_else(|error| panic!("failed to run {archiver}: {error}"));
    if !status.success() {
        panic!("{archiver} failed to build the device archive (status {status})");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    // `+whole-archive` is load-bearing. nvcc registers each fatbin from a
    // constructor in `.init_array`; nothing in Rust references that constructor
    // by symbol, so a normal archive link keeps the fatbin *data* while
    // dropping the registration. The kernels then fail at launch with "invalid
    // device function" even though the build and the link both succeeded —
    // observed on a real RTX 3060 Ti, where the fatbin sections were present in
    // the test binary but `__cudaRegisterFatBinary` was absent.
    println!("cargo:rustc-link-lib=static:+whole-archive=neoethos_gpu_cuda_native");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    if let Ok(path) = env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={path}/lib64");
    }
    for candidate in ["/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"] {
        if Path::new(candidate).is_dir() {
            println!("cargo:rustc-link-search=native={candidate}");
        }
    }
}
