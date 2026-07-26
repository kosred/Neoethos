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
    let mut build = cc::Build::new();
    build.cpp(true).std("c++17").include("native");
    build.file("native/layout_asserts.cpp");

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
        build
            .cuda(true)
            .compiler(nvcc)
            .file("native/smoke.cu")
            .file("native/prototype_b.cu")
            .file("native/prototype_b_population.cu");
    } else {
        build.file("native/stub.cpp");
    }
    build.compile("neoethos_gpu_cuda_native");
}
