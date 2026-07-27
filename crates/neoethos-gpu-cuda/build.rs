use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=native/neoethos_gpu_cuda.h");
    println!("cargo:rerun-if-changed=native/layout_asserts.cpp");
    println!("cargo:rerun-if-changed=native/stub.cpp");
    println!("cargo:rerun-if-changed=native/smoke.cu");
    println!("cargo:rerun-if-changed=native/prototype_b.cu");
    println!("cargo:rerun-if-changed=native/prototype_b_population.cu");
    println!("cargo:rerun-if-env-changed=CUDACXX");

    let cuda_feature = env::var_os("CARGO_FEATURE_CUDA").is_some();

    // Host C++ and CUDA are compiled as two separate units on purpose. A single
    // `cc::Build` with `.cuda(true)` drives *every* file through nvcc, including
    // the plain `.cpp`, and forwards host flags such as `-ffunction-sections`
    // verbatim — which nvcc rejects outright ("Unknown option"). Splitting them
    // keeps gcc flags on the gcc unit and lets cc-rs wrap them in `-Xcompiler`
    // for the real device translation units.
    let mut host = cc::Build::new();
    host.cpp(true).std("c++17").include("native");
    host.file("native/layout_asserts.cpp");
    if !cuda_feature {
        host.file("native/stub.cpp");
    }
    host.compile("neoethos_gpu_cuda_abi");

    if cuda_feature {
        let nvcc = env::var("CUDACXX").unwrap_or_else(|_| "nvcc".to_string());
        let available = Command::new(&nvcc)
            .arg("--version")
            .status()
            .map(|status| status.success())
            .unwrap_or(false);
        if !available {
            panic!("feature `cuda` requires nvcc; set CUDACXX or install the CUDA toolkit");
        }
        let mut device = cc::Build::new();
        device
            .cuda(true)
            .cpp(true)
            .std("c++17")
            .include("native")
            .compiler(nvcc)
            .file("native/smoke.cu")
            .file("native/prototype_b.cu")
            .file("native/prototype_b_population.cu");
        device.compile("neoethos_gpu_cuda_native");
        // The CUDA runtime is linked explicitly: the device unit is a static
        // archive, so nothing else pulls in cudart for us.
        println!("cargo:rustc-link-lib=dylib=cudart");
        if let Ok(path) = env::var("CUDA_PATH") {
            println!("cargo:rustc-link-search=native={path}/lib64");
        }
        for candidate in ["/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"] {
            if std::path::Path::new(candidate).is_dir() {
                println!("cargo:rustc-link-search=native={candidate}");
            }
        }
    }
}
