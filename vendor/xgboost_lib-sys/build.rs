use bindgen;
use std::env;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

#[cfg(all(feature = "local_build", feature = "use_prebuilt_xgb"))]
compile_error!(
    "xgboost_lib-sys requires exactly one XGBoost runtime authority; local_build and \
     use_prebuilt_xgb cannot be enabled together"
);

#[cfg(not(any(feature = "local_build", feature = "use_prebuilt_xgb")))]
compile_error!(
    "xgboost_lib-sys requires exactly one XGBoost runtime authority; enable local_build or \
     use_prebuilt_xgb"
);

#[cfg(all(feature = "local_build", feature = "cuda"))]
#[path = "../cuda_build_arch.rs"]
mod cuda_build_arch;

#[cfg(feature = "use_prebuilt_xgb")]
const GITHUB_URL: &str =
    "https://github.com/marcomq/rust-xgboost/raw/refs/tags/v3.0.1/xgboost-sys/lib/";

fn main() {
    let target = env::var("TARGET").unwrap();
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR"));
    // VENDORED FIX (2026-08-04, the reason this crate is vendored at all).
    // `std::fs::canonicalize` on Windows returns an extended-length path
    // (`\\?\C:\...`). Passing that to CMake as the source directory breaks
    // `file(GLOB)` inside xgboost's CMakeLists — the globs come back empty and
    // configure dies with "No SOURCES given to target: xgboost". Upstream
    // already depends on `dunce`; this source-path call was the missed one.
    // `dunce` strips the prefix when the path is representable.
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

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings.");

    if target.contains("apple") {
        println!(
            "cargo:rustc-link-search=native={}/opt/libomp/lib",
            &std::env::var("HOMEBREW_PREFIX").unwrap_or("/opt/homebrew".into())
        );
    }

    #[cfg(feature = "use_prebuilt_xgb")]
    {
        println!("cargo:rerun-if-env-changed=XGBOOST_LIB_DIR");
        let selected_runtime = if let Some(xgboost_lib_dir) = std::env::var_os("XGBOOST_LIB_DIR") {
            let xgboost_lib_dir = PathBuf::from(xgboost_lib_dir);
            println!(
                "cargo:rustc-link-search=native={}",
                xgboost_lib_dir.display()
            );
            xgboost_lib_dir.join(runtime_library_filename(&target))
        } else {
            let deps_path = cargo_profile_output_dir(&out_dir)
                .unwrap_or_else(|error| panic!("xgboost_lib-sys: {error}"))
                .join("deps");
            std::fs::create_dir_all(&deps_path).unwrap_or_else(|error| {
                panic!(
                    "xgboost_lib-sys: failed to create {}: {error}",
                    deps_path.display()
                )
            });
            println!("cargo:rustc-link-search=native={}", deps_path.display());
            if target.contains("apple-darwin") && target.contains("aarch64") {
                let path = format!("{GITHUB_URL}/mac_arm64");
                let runtime = deps_path.join("libxgboost.dylib");
                if !runtime.exists() {
                    web_copy(
                        &format!("{path}/libxgboost.dylib"),
                        &runtime.to_string_lossy(),
                    )
                    .unwrap();
                    web_copy(
                        &format!("{path}/libdmlc.a"),
                        &deps_path.join("libdmlc.a").to_string_lossy(),
                    )
                    .unwrap();
                }
                runtime
            } else if target.contains("linux") {
                let path = if target.contains("aarch64") {
                    format!("{GITHUB_URL}/linux_arm64")
                } else {
                    format!("{GITHUB_URL}/linux_amd64")
                };
                let runtime = deps_path.join("libxgboost.so");
                if !runtime.exists() {
                    web_copy(&format!("{path}/libxgboost.so"), &runtime.to_string_lossy()).unwrap();
                    web_copy(
                        &format!("{path}/libdmlc.a"),
                        &deps_path.join("libdmlc.a").to_string_lossy(),
                    )
                    .unwrap();
                }
                runtime
            } else if target.contains("windows") && target.contains("x86_64") {
                let path = format!("{GITHUB_URL}/win_amd64");
                let runtime = deps_path.join("xgboost.dll");
                if !runtime.exists() {
                    web_copy(&format!("{path}/xgboost.dll"), &runtime.to_string_lossy()).unwrap();
                    web_copy(
                        &format!("{path}/xgboost.lib"),
                        &deps_path.join("xgboost.lib").to_string_lossy(),
                    )
                    .unwrap();
                }
                runtime
            } else if let Some(homebrew_path) = std::env::var_os("HOMEBREW_PREFIX") {
                let xgboost_lib_dir = PathBuf::from(homebrew_path).join("opt/xgboost/lib");
                println!(
                    "cargo:rustc-link-search=native={}",
                    xgboost_lib_dir.display()
                );
                xgboost_lib_dir.join(runtime_library_filename(&target))
            } else {
                panic!("Please set $XGBOOST_LIB_DIR")
            }
        };
        stage_selected_runtime(&out_dir, &selected_runtime, &target);
    }

    #[cfg(feature = "local_build")]
    {
        // compile XGBOOST with cmake and ninja

        // CMake
        let mut config = cmake::Config::new(&xgb_root);
        config
            .generator("Ninja")
            .define("CMAKE_BUILD_TYPE", "RelWithDebInfo");

        #[cfg(feature = "cuda")]
        {
            let architectures = cuda_build_arch::resolve_exact_cuda_architectures()
                .unwrap_or_else(|error| panic!("xgboost_lib-sys: {error}"));
            config
                .define("CMAKE_CUDA_ARCHITECTURES", &architectures.native_only)
                .define("USE_CUDA", "ON");
        }

        let dst = config.build();

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

        let selected_runtime = if target.contains("linux") {
            dst.join("lib").join("libxgboost.so")
        } else if target.contains("windows") {
            dst.join("bin").join("xgboost.dll")
        } else if target.contains("apple-darwin") {
            dst.join("lib").join("libxgboost.dylib")
        } else {
            panic!("xgboost_lib-sys: unsupported local-build target {target}");
        };
        stage_selected_runtime(&out_dir, &selected_runtime, &target);
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

fn runtime_library_filename(target: &str) -> &'static str {
    if target.contains("windows") {
        "xgboost.dll"
    } else if target.contains("linux") {
        "libxgboost.so"
    } else if target.contains("apple-darwin") {
        "libxgboost.dylib"
    } else {
        panic!("xgboost_lib-sys: unsupported runtime target {target}");
    }
}

fn cargo_profile_output_dir(out_dir: &Path) -> Result<PathBuf, String> {
    if out_dir.file_name() != Some(OsStr::new("out")) {
        return Err(format!(
            "OUT_DIR must end in `out`, got `{}`",
            out_dir.display()
        ));
    }
    let package_build_dir = out_dir.parent().ok_or_else(|| {
        format!(
            "OUT_DIR has no package build directory: `{}`",
            out_dir.display()
        )
    })?;
    let cargo_build_dir = package_build_dir.parent().ok_or_else(|| {
        format!(
            "OUT_DIR has no Cargo build directory: `{}`",
            out_dir.display()
        )
    })?;
    if cargo_build_dir.file_name() != Some(OsStr::new("build")) {
        return Err(format!(
            "OUT_DIR is not in Cargo's `<profile>/build/<package>/out` layout: `{}`",
            out_dir.display()
        ));
    }
    cargo_build_dir
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| {
            format!(
                "OUT_DIR has no Cargo profile directory: `{}`",
                out_dir.display()
            )
        })
}

fn stage_selected_runtime(out_dir: &Path, source: &Path, target: &str) {
    let expected_name = runtime_library_filename(target);
    if source.file_name() != Some(OsStr::new(expected_name)) {
        panic!(
            "xgboost_lib-sys: selected runtime {} does not have expected filename {expected_name}",
            source.display()
        );
    }
    if !source.is_file() {
        panic!(
            "xgboost_lib-sys: selected runtime is missing: {}",
            source.display()
        );
    }
    let source_len = source
        .metadata()
        .unwrap_or_else(|error| {
            panic!(
                "xgboost_lib-sys: failed to inspect selected runtime {}: {error}",
                source.display()
            )
        })
        .len();
    if source_len == 0 {
        panic!(
            "xgboost_lib-sys: selected runtime is empty: {}",
            source.display()
        );
    }
    let profile_dir = cargo_profile_output_dir(out_dir)
        .unwrap_or_else(|error| panic!("xgboost_lib-sys: {error}"));
    let destination = profile_dir.join(expected_name);
    let copied_bytes = std::fs::copy(source, &destination).unwrap_or_else(|error| {
        panic!(
            "xgboost_lib-sys: failed to stage selected runtime {} as {}: {error}",
            source.display(),
            destination.display()
        )
    });
    if copied_bytes != source_len {
        panic!(
            "xgboost_lib-sys: staged {copied_bytes} of {source_len} bytes from {} to {}",
            source.display(),
            destination.display()
        );
    }
    eprintln!(
        "INFO xgboost_lib-sys: staged selected runtime {} -> {} ({source_len} bytes)",
        source.display(),
        destination.display()
    );
}

#[cfg(feature = "use_prebuilt_xgb")]
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cfg(feature = "use_prebuilt_xgb")]
fn web_copy(web_src: &str, target: &str) -> Result<()> {
    dbg!(&web_src);
    let resp = reqwest::blocking::get(web_src)?;
    let body = resp.bytes()?;
    std::fs::write(target, &body)?;
    Ok(())
}
