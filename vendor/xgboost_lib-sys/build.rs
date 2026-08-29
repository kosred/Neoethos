use bindgen;
use std::env;
use std::path::{Path, PathBuf};

#[path = "../cuda_build_arch.rs"]
mod cuda_build_arch;

const GITHUB_URL: &str =
    "https://github.com/marcomq/rust-xgboost/raw/refs/tags/v3.0.1/xgboost-sys/lib/";

fn main() {
    println!("cargo:rerun-if-changed=../cuda_build_arch.rs");
    println!("cargo:rerun-if-env-changed=NEOETHOS_CUDA_ARCHS");
    let target = env::var("TARGET").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();
    // VENDORED FIX (2026-08-04, the reason this crate is vendored at all).
    // `std::fs::canonicalize` on Windows returns an extended-length path
    // (`\\?\C:\...`). Passing that to CMake as the source directory breaks
    // `file(GLOB)` inside xgboost's CMakeLists — the globs come back empty and
    // configure dies with "No SOURCES given to target: xgboost". Upstream
    // already uses `dunce::canonicalize` for the deps path 29 lines below, so
    // the dependency exists and the author knew the hazard; this one line was
    // missed. `dunce` strips the prefix when the path is representable.
    // Upstream: Marco Mengelkoch, xgboost_lib-sys 3.0.4 (max published).
    let xgb_root = dunce::canonicalize(Path::new("xgboost")).unwrap();

    let wrapper_h = xgb_root.join("include").join("xgboost").join("c_api.h");
    let bindings = bindgen::Builder::default()
        .header(wrapper_h.to_string_lossy())
        .clang_arg(format!("-I{}", xgb_root.join("include").display()))
        .clang_arg(format!(
            "-I{}",
            xgb_root.join("dmlc-core").join("include").display()
        ));

    #[cfg(feature = "cuda")]
    let bindings = bindings.clang_arg("-I/usr/local/cuda/include");
    let bindings = bindings.generate().expect("Unable to generate bindings.");

    let out_path = PathBuf::from(&out_dir);
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings.");

    if target.contains("apple") {
        println!(
            "cargo:rustc-link-search=native={}/opt/libomp/lib",
            &std::env::var("HOMEBREW_PREFIX").unwrap_or("/opt/homebrew".into())
        );
    }

    #[cfg(feature = "use_prebuilt_xgb")]
    {
        if let Ok(xgboost_lib_dir) = std::env::var("XGBOOST_LIB_DIR") {
            println!("cargo:rustc-link-search=native={}", xgboost_lib_dir);
        } else {
            let deps_path =
                dunce::canonicalize(Path::new(&format!("{}/../../../deps", out_dir))).unwrap();
            let deps_path = deps_path.to_string_lossy();
            println!("cargo:rustc-link-search=native={}", deps_path);
            if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                let path = format!("{GITHUB_URL}/mac_arm64");
                if !std::fs::exists(format!("{deps_path}/libxgboost.dylib")).unwrap() {
                    web_copy(
                        &format!("{path}/libxgboost.dylib"),
                        &format!("{deps_path}/libxgboost.dylib"),
                    )
                    .unwrap();
                    web_copy(
                        &format!("{path}/libdmlc.a"),
                        &format!("{deps_path}/libdmlc.a"),
                    )
                    .unwrap();
                }
            } else if cfg!(target_os = "linux") {
                let path = if cfg!(target_arch = "aarch64") {
                    format!("{GITHUB_URL}/linux_arm64")
                } else {
                    format!("{GITHUB_URL}/linux_amd64")
                };
                if !std::fs::exists(format!("{deps_path}/libxgboost.so")).unwrap() {
                    web_copy(
                        &format!("{path}/libxgboost.so"),
                        &format!("{deps_path}/libxgboost.so"),
                    )
                    .unwrap();
                    web_copy(
                        &format!("{path}/libdmlc.a"),
                        &format!("{deps_path}/libdmlc.a"),
                    )
                    .unwrap();
                }
            } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
                let path = format!("{GITHUB_URL}/win_amd64");
                if !std::fs::exists(format!("{deps_path}/xgboost.dll")).unwrap() {
                    web_copy(
                        &format!("{path}/xgboost.dll"),
                        &format!("{deps_path}/xgboost.dll"),
                    )
                    .unwrap();
                    web_copy(
                        &format!("{path}/xgboost.lib"),
                        &format!("{deps_path}/xgboost.lib"),
                    )
                    .unwrap();
                }
            } else {
                if let Ok(homebrew_path) = std::env::var("HOMEBREW_PREFIX") {
                    let xgboost_lib_dir = format!("{}/opt/xgboost/lib", &homebrew_path);
                    println!("cargo:rustc-link-search=native={}", xgboost_lib_dir);
                } else {
                    panic!("Please set $XGBOOST_LIB_DIR")
                }
            }
        }
    }

    #[cfg(feature = "local_build")]
    {
        // compile XGBOOST with cmake and ninja

        // CMake
        let mut dst = cmake::Config::new(&xgb_root);
        dst.generator("Ninja");
        dst.define("CMAKE_BUILD_TYPE", "RelWithDebInfo");

        #[cfg(feature = "cuda")]
        {
            dst.define("USE_CUDA", "ON")
                .define("BUILD_WITH_CUDA", "ON")
                .define("BUILD_WITH_CUDA_CUB", "ON");
            let cuda_architectures = cmake_cuda_architectures();
            println!(
                "cargo:warning=xgboost_lib-sys: CUDA architectures explicitly limited to {cuda_architectures}"
            );
            dst.define("CMAKE_CUDA_ARCHITECTURES", cuda_architectures);
        }

        let dst = dst.build();

        println!("cargo:rustc-link-search=native={}", dst.display());
        println!(
            "cargo:rustc-link-search=native={}",
            dst.join("lib").display()
        );
        println!(
            "cargo:rustc-link-search=native={}",
            dst.join("lib64").display()
        );
        println!("cargo:rustc-link-lib=static=dmlc");
    }

    // link to appropriate C++ lib
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=dylib=omp");
    } else {
        #[cfg(target_os = "linux")]
        {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=stdc++fs");
            println!("cargo:rustc-link-lib=dylib=gomp");
        }
    }

    println!("cargo:rustc-link-lib=dylib=xgboost");

    #[cfg(feature = "cuda")]
    {
        println!("cargo:rustc-link-search={}", "/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=static=cudart_static");
    }
}

#[cfg(all(feature = "local_build", feature = "cuda"))]
fn cmake_cuda_architectures() -> String {
    let architectures = cuda_build_arch::required_cuda_arch_numbers();
    let mut targets = architectures
        .iter()
        .map(|architecture| format!("{architecture}-real"))
        .collect::<Vec<_>>();
    targets.push(format!(
        "{}-virtual",
        architectures
            .last()
            .expect("validated CUDA architecture set is non-empty")
    ));
    targets.join(";")
}

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cfg(feature = "use_prebuilt_xgb")]
fn web_copy(web_src: &str, target: &str) -> Result<()> {
    dbg!(&web_src);
    let resp = reqwest::blocking::get(web_src)?;
    let body = resp.bytes()?;
    std::fs::write(target, &body)?;
    Ok(())
}
