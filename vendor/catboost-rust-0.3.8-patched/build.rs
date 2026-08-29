extern crate bindgen;

mod build_support;

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

fn get_catboost_version() -> String {
    env::var("CATBOOST_VERSION").unwrap_or_else(|_| "1.2.8".to_string())
}

fn get_platform_info() -> (String, String) {
    let target = env::var("TARGET").unwrap();

    // Determine OS
    let os = if target.contains("apple-darwin") {
        "darwin"
    } else if target.contains("linux") {
        "linux"
    } else if target.contains("windows") {
        "windows"
    } else {
        panic!("Unsupported target: {}", target);
    };

    // Determine architecture
    let arch = if target.contains("x86_64") {
        "x86_64"
    } else if target.contains("aarch64") || target.contains("arm64") {
        "aarch64"
    } else if target.contains("i686") || target.contains("i586") {
        "i686"
    } else {
        panic!("Unsupported architecture for target: {}", target);
    };

    (os.to_string(), arch.to_string())
}

fn stage_selected_runtime(source: &Path, destination: &Path) {
    if !source.is_file() {
        panic!(
            "catboost-rust: selected runtime is missing: {}",
            source.display()
        );
    }
    let source_len = source
        .metadata()
        .unwrap_or_else(|error| {
            panic!(
                "catboost-rust: failed to inspect selected runtime {}: {error}",
                source.display()
            )
        })
        .len();
    if source_len == 0 {
        panic!(
            "catboost-rust: selected runtime is empty: {}",
            source.display()
        );
    }
    let copied_bytes = fs::copy(source, destination).unwrap_or_else(|error| {
        panic!(
            "catboost-rust: failed to stage selected runtime {} as {}: {error}",
            source.display(),
            destination.display()
        )
    });
    if copied_bytes != source_len {
        panic!(
            "catboost-rust: staged {copied_bytes} of {source_len} bytes from {} to {}",
            source.display(),
            destination.display()
        );
    }
    eprintln!(
        "INFO catboost-rust: staged selected runtime {} -> {} ({source_len} bytes)",
        source.display(),
        destination.display()
    );
}

fn download_model_interface_headers(out_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let version = get_catboost_version();

    // Create the model_interface directory
    let model_interface_dir = out_dir.join("libs/model_interface");
    fs::create_dir_all(&model_interface_dir)?;

    // Download the c_api.h file
    let c_api_url = format!(
        "https://raw.githubusercontent.com/catboost/catboost/v{}/catboost/libs/model_interface/c_api.h",
        version
    );

    eprintln!("INFO catboost-rust: downloading c_api.h from {c_api_url}");

    let response = ureq::get(&c_api_url).call()?;
    let status = response.status();
    if !(200..300).contains(&status) {
        return Err(format!("Failed to download c_api.h: HTTP {}", status).into());
    }

    let c_api_path = model_interface_dir.join("c_api.h");
    let mut file = fs::File::create(&c_api_path)?;
    io::copy(&mut response.into_reader(), &mut file)?;

    Ok(())
}

fn download_compiled_library(out_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let (os, arch) = get_platform_info();
    let version = get_catboost_version();

    if os == "windows" && arch == "aarch64" {
        return Err(
            "CatBoost v1.2.x does not publish a Windows aarch64 import library; \
             refusing to download a DLL that Cargo cannot link"
                .into(),
        );
    }

    // Create the library directory early
    let lib_dir = out_dir.join("libs");
    fs::create_dir_all(&lib_dir)?;

    // Parse version to determine URL format
    // v1.0.x - v1.1.x use simple filenames
    // v1.2+ use platform-specific versioned filenames
    let version_parts: Vec<&str> = version.split('.').collect();
    let major: u32 = version_parts
        .first()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let minor: u32 = version_parts
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let use_new_format = major > 1 || (major == 1 && minor >= 2);

    // Determine download URL based on version and platform
    let (lib_filename, download_url) = if use_new_format {
        // v1.2+ format with platform and version in filename
        match (os.as_str(), arch.as_str()) {
            ("linux", "x86_64") => (
                "libcatboostmodel.so".to_string(),
                format!(
                    "https://github.com/catboost/catboost/releases/download/v{}/libcatboostmodel-linux-x86_64-{}.so",
                    version, version
                ),
            ),
            ("linux", "aarch64") => (
                "libcatboostmodel.so".to_string(),
                format!(
                    "https://github.com/catboost/catboost/releases/download/v{}/libcatboostmodel-linux-aarch64-{}.so",
                    version, version
                ),
            ),
            ("darwin", "x86_64") | ("darwin", "aarch64") => (
                "libcatboostmodel.dylib".to_string(),
                format!(
                    "https://github.com/catboost/catboost/releases/download/v{}/libcatboostmodel-darwin-universal2-{}.dylib",
                    version, version
                ),
            ),
            ("windows", "x86_64") => {
                // On Windows, we need to download both the DLL and LIB files
                // First download the DLL
                let dll_url = format!(
                    "https://github.com/catboost/catboost/releases/download/v{}/catboostmodel-windows-x86_64-{}.dll",
                    version, version
                );
                eprintln!("INFO catboost-rust: downloading Windows DLL from {dll_url}");
                let dll_response = ureq::get(&dll_url).call()?;
                if !(200..300).contains(&dll_response.status()) {
                    return Err(
                        format!("Failed to download DLL: HTTP {}", dll_response.status()).into(),
                    );
                }
                let dll_path = lib_dir.join("catboostmodel.dll");
                let mut dll_file = fs::File::create(&dll_path)?;
                io::copy(&mut dll_response.into_reader(), &mut dll_file)?;

                // Then download the LIB file
                let lib_url = format!(
                    "https://github.com/catboost/catboost/releases/download/v{}/catboostmodel-windows-x86_64-{}.lib",
                    version, version
                );
                eprintln!("INFO catboost-rust: downloading Windows LIB from {lib_url}");
                let lib_response = ureq::get(&lib_url).call()?;
                if !(200..300).contains(&lib_response.status()) {
                    return Err(
                        format!("Failed to download LIB: HTTP {}", lib_response.status()).into(),
                    );
                }
                let lib_path = lib_dir.join("catboostmodel.lib");
                let mut lib_file = fs::File::create(&lib_path)?;
                io::copy(&mut lib_response.into_reader(), &mut lib_file)?;

                // Return early for Windows since we've already downloaded both files
                eprintln!(
                    "INFO catboost-rust: downloaded CatBoost library to {}",
                    dll_path.display()
                );
                return Ok(());
            }
            _ => return Err(format!("Unsupported platform: {}-{}", os, arch).into()),
        }
    } else {
        // v1.0.x - v1.1.x format with simple filenames
        match os.as_str() {
            "linux" => (
                "libcatboostmodel.so".to_string(),
                format!(
                    "https://github.com/catboost/catboost/releases/download/v{}/libcatboostmodel.so",
                    version
                ),
            ),
            "darwin" => (
                "libcatboostmodel.dylib".to_string(),
                format!(
                    "https://github.com/catboost/catboost/releases/download/v{}/libcatboostmodel.dylib",
                    version
                ),
            ),
            "windows" => {
                // On Windows, we need to download both the DLL and LIB files
                // First download the DLL
                let dll_url = format!(
                    "https://github.com/catboost/catboost/releases/download/v{}/catboostmodel.dll",
                    version
                );
                eprintln!("INFO catboost-rust: downloading Windows DLL from {dll_url}");
                let dll_response = ureq::get(&dll_url).call()?;
                if !(200..300).contains(&dll_response.status()) {
                    return Err(
                        format!("Failed to download DLL: HTTP {}", dll_response.status()).into(),
                    );
                }
                let dll_path = lib_dir.join("catboostmodel.dll");
                let mut dll_file = fs::File::create(&dll_path)?;
                io::copy(&mut dll_response.into_reader(), &mut dll_file)?;

                // Then download the LIB file
                let lib_url = format!(
                    "https://github.com/catboost/catboost/releases/download/v{}/catboostmodel.lib",
                    version
                );
                eprintln!("INFO catboost-rust: downloading Windows LIB from {lib_url}");
                let lib_response = ureq::get(&lib_url).call()?;
                if !(200..300).contains(&lib_response.status()) {
                    return Err(
                        format!("Failed to download LIB: HTTP {}", lib_response.status()).into(),
                    );
                }
                let lib_path = lib_dir.join("catboostmodel.lib");
                let mut lib_file = fs::File::create(&lib_path)?;
                io::copy(&mut lib_response.into_reader(), &mut lib_file)?;

                // Return early for Windows since we've already downloaded both files
                eprintln!(
                    "INFO catboost-rust: downloaded CatBoost library to {}",
                    dll_path.display()
                );
                return Ok(());
            }
            _ => return Err(format!("Unsupported platform: {}", os).into()),
        }
    };

    eprintln!("INFO catboost-rust: downloading CatBoost v{version} library from {download_url}");

    // Download the library directly into the `libs` directory with its correct name
    let lib_path = lib_dir.join(&lib_filename);
    let mut dest = fs::File::create(&lib_path)?;

    let response = ureq::get(&download_url).call()?;
    let status = response.status();
    if !(200..300).contains(&status) {
        return Err(format!("Failed to download library: HTTP {}", status).into());
    }

    // SIMPLIFIED: No need for extraction, just copy the downloaded content
    io::copy(&mut response.into_reader(), &mut dest)?;

    eprintln!(
        "INFO catboost-rust: downloaded CatBoost library to {}",
        lib_path.display()
    );

    Ok(())
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cb_model_interface_root = out_dir.join("libs/model_interface");

    println!("cargo:rerun-if-env-changed=CATBOOST_VERSION");

    // Declare custom cfg flags for Cargo's check-cfg feature
    println!("cargo::rustc-check-cfg=cfg(catboost_embeddings)");
    println!("cargo::rustc-check-cfg=cfg(catboost_text_count)");
    println!("cargo::rustc-check-cfg=cfg(catboost_staged_prediction)");
    println!("cargo::rustc-check-cfg=cfg(catboost_feature_indices)");
    println!("cargo::rustc-check-cfg=cfg(catboost_zero_copy)");

    // Parse version for feature detection
    let version = get_catboost_version();
    let version_parts: Vec<&str> = version.split('.').collect();
    let major: u32 = version_parts
        .first()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let minor: u32 = version_parts
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let patch: u32 = version_parts
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // Emit cfg flags for version-specific features
    // v1.1.1+: Embedding features support
    if major > 1 || (major == 1 && minor > 1) || (major == 1 && minor == 1 && patch >= 1) {
        println!("cargo:rustc-cfg=catboost_embeddings");
    }

    // v1.2+: Text features count function
    if major > 1 || (major == 1 && minor >= 2) {
        println!("cargo:rustc-cfg=catboost_text_count");
    }

    // v1.2.3+: Staged predictions and feature indices
    if major > 1 || (major == 1 && minor > 2) || (major == 1 && minor == 2 && patch >= 3) {
        println!("cargo:rustc-cfg=catboost_staged_prediction");
        println!("cargo:rustc-cfg=catboost_feature_indices");
    }

    // v1.2.9+: Zero-copy buffer loading
    if major > 1 || (major == 1 && minor > 2) || (major == 1 && minor == 2 && patch >= 9) {
        println!("cargo:rustc-cfg=catboost_zero_copy");
    }

    // Download the model interface headers
    if let Err(e) = download_model_interface_headers(&out_dir) {
        eprintln!("Failed to download model interface headers: {}", e);
        panic!("Cannot proceed without headers");
    }

    // Download the compiled library
    if let Err(e) = download_compiled_library(&out_dir) {
        eprintln!("Failed to download compiled library: {}", e);
        panic!("Cannot proceed without compiled library");
    }

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", cb_model_interface_root.display()))
        .size_t_is_usize(true)
        .generate()
        .expect("Unable to generate bindings.");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings.");

    // 1. Get platform info using your existing function
    let (os, _arch) = get_platform_info();

    // 2. Determine the library filename based on the OS
    let lib_filename = match os.as_str() {
        "windows" => "catboostmodel.dll",
        "darwin" => "libcatboostmodel.dylib", // "darwin" comes from your function
        _ => "libcatboostmodel.so",           // Default to Linux/Unix
    };

    // 3. Copy the library from OUT_DIR/libs to the final target directory
    let lib_source_path = out_dir.join("libs").join(lib_filename);

    // Resolve the final profile output directory from Cargo's authoritative
    // OUT_DIR layout. The upstream implementation searched for an ancestor
    // literally named `target`, which panicked for every valid custom
    // CARGO_TARGET_DIR (for example `target-models-cuda`).
    let target_dir = build_support::cargo_profile_output_dir(&out_dir).unwrap_or_else(|message| {
        panic!("Cannot resolve Cargo profile output directory: {message}")
    });

    let lib_dest_path = target_dir.join(lib_filename);
    stage_selected_runtime(&lib_source_path, &lib_dest_path);

    // On macOS/Linux, change the install name/soname to use @loader_path/$ORIGIN
    // This needs to be done on the source library in OUT_DIR before linking
    if os == "darwin" {
        use std::process::Command;
        let _ = Command::new("install_name_tool")
            .arg("-id")
            .arg(format!("@loader_path/{}", lib_filename))
            .arg(&lib_source_path)
            .status();
        // Also update the copy
        let _ = Command::new("install_name_tool")
            .arg("-id")
            .arg(format!("@loader_path/{}", lib_filename))
            .arg(&lib_dest_path)
            .status();
    } else if os == "linux" {
        use std::process::Command;
        // Use patchelf to set soname to just the library filename on Linux (if available)
        // This is optional - if patchelf is not installed, we just skip it
        let _ = Command::new("patchelf")
            .arg("--set-soname")
            .arg(lib_filename)
            .arg(&lib_source_path)
            .output(); // Use output() to silently ignore if patchelf doesn't exist
        let _ = Command::new("patchelf")
            .arg("--set-soname")
            .arg(lib_filename)
            .arg(&lib_dest_path)
            .output();
    }

    // 4. Set the library search path for the build-time linker
    let lib_search_path = out_dir.join("libs");
    println!(
        "cargo:rustc-link-search=native={}",
        lib_search_path.display()
    );

    // 5. Set the rpath for the run-time linker based on the OS
    match os.as_str() {
        "darwin" => {
            // For macOS, add multiple rpath entries for IDE compatibility
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../..");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../..");
            println!(
                "cargo:rustc-link-arg=-Wl,-rpath,{}",
                lib_search_path.display()
            );
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", target_dir.display());
        }
        "linux" => {
            // For Linux, use $ORIGIN
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../..");
            println!(
                "cargo:rustc-link-arg=-Wl,-rpath,{}",
                lib_search_path.display()
            );
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", target_dir.display());
        }
        _ => {} // No rpath needed for Windows
    }

    println!("cargo:rustc-link-lib=dylib=catboostmodel");
}
