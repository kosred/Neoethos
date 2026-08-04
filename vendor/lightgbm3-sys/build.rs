use cmake::Config;
use std::{
    env,
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug)]
struct DoxygenCallback;

impl bindgen::callbacks::ParseCallbacks for DoxygenCallback {
    fn process_comment(&self, comment: &str) -> Option<String> {
        Some(doxygen_rs::transform(comment))
    }
}

fn main() {
    let target = env::var("TARGET").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();
    let lgbm_root = Path::new(&out_dir).join("lightgbm");

    // copy source code
    if !lgbm_root.exists() {
        copy_dir_recursive(Path::new("lightgbm"), &lgbm_root)
            .unwrap_or_else(|err| panic!("Failed to copy ./lightgbm to {}: {err}", lgbm_root.display()));
    }

    // CMake
    let mut cfg = Config::new(&lgbm_root);
    let cfg = cfg
        .profile("Release")
        .cxxflag("-std=c++14")
        .define("BUILD_STATIC_LIB", "ON");
    #[cfg(target_os = "windows")]
    let cfg = cfg.generator("NMake Makefiles");
    #[cfg(not(feature = "openmp"))]
    let cfg = cfg.define("USE_OPENMP", "OFF");
    #[cfg(feature = "gpu")]
    let cfg = cfg.define("USE_GPU", "1");
    // PATCHED (NeoEthos 2026-08-02): the `cuda` feature is honoured only on
    // Linux. LightGBM's own CUDA block hard-codes GCC driver flags —
    //   set(CMAKE_CUDA_FLAGS "... -Xcompiler=-fPIC -Xcompiler=-Wall")
    // (CMakeLists.txt:221) — so `USE_CUDA=1` cannot configure under MSVC, and
    // upstream documents the CUDA tree learner as Linux-only. Passing the
    // feature through unconditionally would turn every Windows `gpu-cuda`
    // build into a CMake configure failure. Warn instead of failing so the
    // one aggregate feature still builds on both platforms, and say plainly
    // that the resulting library has no CUDA learner.
    let cuda_enabled = cfg!(feature = "cuda") && target.contains("linux");
    if cfg!(feature = "cuda") && !cuda_enabled {
        println!(
            "cargo:warning=lightgbm3-sys: the `cuda` feature was requested but LightGBM's \
             CUDA tree learner is Linux-only upstream (its CMake CUDA flags are GCC-only). \
             Building CPU-only on this target; LightGBMExpert will resolve device_type=cpu."
        );
    }
    if cuda_enabled {
        cfg.define("USE_CUDA", "1");
    }
    let dst = cfg.build();

    // bindgen build
    let mut clang_args = vec!["-x", "c++", "-std=c++14"];
    if target.contains("apple") {
        clang_args.push("-mmacosx-version-min=10.12");
    }
    let bindings = bindgen::Builder::default()
        .header("lightgbm/include/LightGBM/c_api.h")
        .allowlist_file("lightgbm/include/LightGBM/c_api.h")
        .clang_args(&clang_args)
        .clang_arg(format!("-I{}", lgbm_root.join("include").display()))
        .parse_callbacks(Box::new(DoxygenCallback))
        .generate()
        .expect("Unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .unwrap_or_else(|err| panic!("Couldn't write bindings: {err}"));
    // link to appropriate C++ lib
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("linux") {
        println!("cargo:rustc-link-lib=stdc++");
    }
    #[cfg(feature = "openmp")]
    {
        // PATCHED (NeoEthos 2026-08-02): `-fopenmp` is a GCC/Clang *driver*
        // flag. On MSVC the linker is link.exe, which reports it as
        // "LNK4044: unrecognized option '/fopenmp'; ignored" on every link.
        // MSVC does not need it: cmake compiles LightGBM with `/openmp`, and
        // the resulting objects carry a `/DEFAULTLIB:vcomp` directive that
        // pulls the OpenMP runtime in automatically.
        if !target.contains("msvc") {
            println!("cargo:rustc-link-args=-fopenmp");
        }
        if target.contains("apple") {
            println!("cargo:rustc-link-lib=dylib=omp");
            // Link to libomp
            // If it fails to compile in MacOS, try:
            // `brew install libomp`
            // `brew link --force libomp`
            #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
            println!("cargo:rustc-link-search=/usr/local/opt/libomp/lib");
            #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
            println!("cargo:rustc-link-search=/opt/homebrew/opt/libomp/lib");
        } else if target.contains("linux") {
            println!("cargo:rustc-link-lib=dylib=gomp");
        }
    }
    // PATCHED (NeoEthos 2026-08-02): link the CUDA runtime when the CUDA tree
    // learner is actually in the archive.
    //
    // LightGBM's CMake calls `enable_language(CUDA)`, which links cudart into
    // the *cmake* targets. We consume `_lightgbm` as a STATIC archive, so
    // nothing carries that dependency across to the Rust link step and the
    // final binary fails on undefined `cudaMalloc`/`cudaMemcpy`/… . Emitting
    // it here is what makes `USE_CUDA=1` produce a binary that links.
    if cuda_enabled {
        for dir in cuda_library_dirs() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        println!("cargo:rerun-if-env-changed=CUDA_HOME");
    }
    println!("cargo:rustc-link-search={}", out_path.join("lib").display());
    println!("cargo:rustc-link-search=native={}", dst.display());
    if target.contains("windows") {
        println!("cargo:rustc-link-lib=static=lib_lightgbm");
    } else {
        println!("cargo:rustc-link-lib=static=_lightgbm");
    }
}

/// PATCHED (NeoEthos 2026-08-02): candidate directories holding `libcudart.so`,
/// in the order the CUDA toolkit documents them. `CUDA_PATH` is what the
/// official installer sets; `CUDA_HOME` is the convention most CI images use;
/// `/usr/local/cuda` is the default symlink. Every existing candidate is
/// emitted — a missing one is not an error, because the distro package may
/// have put cudart on the default library path already.
fn cuda_library_dirs() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    for key in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
        if let Ok(value) = env::var(key) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                roots.push(PathBuf::from(trimmed));
            }
        }
    }
    roots.push(PathBuf::from("/usr/local/cuda"));

    let mut dirs = Vec::new();
    for root in roots {
        for leaf in ["lib64", "lib", "targets/x86_64-linux/lib"] {
            let candidate = root.join(leaf);
            if candidate.is_dir() && !dirs.contains(&candidate) {
                dirs.push(candidate);
            }
        }
    }
    dirs
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    if !src.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("source directory does not exist: {}", src.display()),
        ));
    }

    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let target_path = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_recursive(&entry.path(), &target_path)?;
        } else if file_type.is_file() {
            fs::copy(entry.path(), target_path)?;
        }
    }
    Ok(())
}
