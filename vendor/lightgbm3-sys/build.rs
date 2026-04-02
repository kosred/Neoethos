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
    #[cfg(feature = "cuda")]
    let cfg = cfg.define("USE_CUDA", "1");
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
        println!("cargo:rustc-link-args=-fopenmp");
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
    println!("cargo:rustc-link-search={}", out_path.join("lib").display());
    println!("cargo:rustc-link-search=native={}", dst.display());
    if target.contains("windows") {
        println!("cargo:rustc-link-lib=static=lib_lightgbm");
    } else {
        println!("cargo:rustc-link-lib=static=_lightgbm");
    }
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
