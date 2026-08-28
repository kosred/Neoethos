use std::path::PathBuf;

// REMOVED 2026-08-09 (dead-code purge, batch D2): the protoc/protobuf codegen
// pipeline.
//
// `main()` used to shell out to a vendored `protoc`, generate Rust + upb
// mini-tables from `proto/OpenApi*.proto` (~100 KB) on every clean build, and
// compile the generated C with `cc`. All of it fed exactly one module,
// `app_services/ctrader_openapi.rs`, whose four `include!` blocks were never
// referenced: `grep 'use protobuf\|protobuf::' crates/neoethos-app/src`
// returned zero, and no generated `ProtoOA*` type was ever constructed. The
// cTrader wire format is hand-rolled JSON in `ctrader_messages.rs`.
//
// Deleted with it: `crates/neoethos-app/proto/`, `ctrader_openapi.rs`, and the
// `protobuf`, `protoc-bin-vendored` and `cc` dependencies. `cc` was checked
// first and is NOT needed by `emit_embedded_credentials` — that function is
// pure `std::fs` + `format!`.

fn main() {
    assert_at_most_one_gpu_feature();
    assert_gpu_toolkit_available();
    emit_linux_native_runtime_runpaths();
    emit_embedded_credentials();
}

/// Linux release packages keep each binary's selected native tree-model
/// runtimes in a private `/usr/lib` directory. Portable bundles keep them
/// adjacent to the executable. Encode both deterministic locations in the
/// binary so no environment-variable or build-tree fallback is required.
fn emit_linux_native_runtime_runpaths() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg-bin=neoethos-app=-Wl,-rpath,$ORIGIN");
        println!("cargo:rustc-link-arg-bin=neoethos-app=-Wl,-rpath,$ORIGIN/lib");
        println!("cargo:rustc-link-arg-bin=neoethos-app=-Wl,-rpath,$ORIGIN/../lib/neoethos-app");
    }
}

/// Generates `$OUT_DIR/embedded_credentials.rs` with compile-time cTrader
/// Open API credentials that are baked into the binary for distribution.
///
/// Resolution order (first non-empty value wins for each field):
/// 1. `NEOETHOS_EMBED_CTRADER_CLIENT_ID` / `_CLIENT_SECRET` / `_REDIRECT_URI`
///    environment variables (CI / explicit override).
/// 2. `.local/neoethos/broker_credentials.toml` in the crate root (dev
///    machine fallback — the same file used by the runtime persistence layer).
/// 3. Empty string (build succeeds; embedded fallback is effectively disabled).
fn emit_embedded_credentials() {
    // CARGO_MANIFEST_DIR = <workspace>/crates/neoethos-app  →  workspace root is two levels up.
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir
        .parent() // crates/
        .and_then(|p| p.parent()) // <workspace root>
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| manifest_dir.clone());

    // Tell Cargo when to re-run this step.
    println!("cargo:rerun-if-env-changed=NEOETHOS_EMBED_CTRADER_CLIENT_ID");
    println!("cargo:rerun-if-env-changed=NEOETHOS_EMBED_CTRADER_CLIENT_SECRET");
    println!("cargo:rerun-if-env-changed=NEOETHOS_EMBED_CTRADER_REDIRECT_URI");
    let local_toml = workspace_root.join(".local/neoethos/broker_credentials.toml");
    println!("cargo:rerun-if-changed={}", local_toml.display());

    // --- Step 1: env vars ---
    let mut client_id = std::env::var("NEOETHOS_EMBED_CTRADER_CLIENT_ID")
        .unwrap_or_default()
        .trim()
        .to_string();
    let mut client_secret = std::env::var("NEOETHOS_EMBED_CTRADER_CLIENT_SECRET")
        .unwrap_or_default()
        .trim()
        .to_string();
    let mut redirect_uri = std::env::var("NEOETHOS_EMBED_CTRADER_REDIRECT_URI")
        .unwrap_or_default()
        .trim()
        .to_string();

    // --- Step 2: workspace .local TOML fallback (simple line-by-line key=value scan) ---
    if client_id.is_empty() || client_secret.is_empty() || redirect_uri.is_empty() {
        let toml_path = local_toml;
        if let Ok(contents) = std::fs::read_to_string(&toml_path) {
            for line in contents.lines() {
                let line = line.trim();
                if let Some(rest) = line.strip_prefix("client_id") {
                    if client_id.is_empty() {
                        client_id = extract_toml_string_value(rest);
                    }
                } else if let Some(rest) = line.strip_prefix("client_secret")
                    && client_secret.is_empty()
                {
                    client_secret = extract_toml_string_value(rest);
                } else if let Some(rest) = line.strip_prefix("redirect_uri")
                    && redirect_uri.is_empty()
                {
                    redirect_uri = extract_toml_string_value(rest);
                }
            }
        }
    }

    // --- Emit ---
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let dest = out_dir.join("embedded_credentials.rs");

    let content = format!(
        "pub const EMBEDDED_CTRADER_CLIENT_ID: &str = r#\"{}\"#;\n\
         pub const EMBEDDED_CTRADER_CLIENT_SECRET: &str = r#\"{}\"#;\n\
         pub const EMBEDDED_CTRADER_REDIRECT_URI: &str = r#\"{}\"#;\n",
        client_id, client_secret, redirect_uri
    );

    std::fs::write(&dest, content).expect("failed to write embedded_credentials.rs");

    // L4: previously printed `cargo:warning=Embedded cTrader client_id (N chars) ...`,
    // which surfaced credential length in CI logs. Suppressed; the embed
    // status is still observable via the file written to OUT_DIR. Set
    // `NEOETHOS_BUILD_VERBOSE=1` to re-enable for local debugging.
    let verbose = std::env::var("NEOETHOS_BUILD_VERBOSE")
        .map(|v| matches!(v.trim(), "1" | "true" | "yes"))
        .unwrap_or(false);
    if verbose {
        if !client_id.is_empty() {
            println!(
                "cargo:warning=Embedded cTrader client_id ({} chars) into binary.",
                client_id.len()
            );
        } else {
            println!(
                "cargo:warning=No embedded cTrader credentials found; binary uses empty fallback."
            );
        }
    }
}

/// Extracts the string value from a TOML assignment fragment like ` = "value"`.
/// Returns empty string if the line doesn't look like a quoted assignment.
fn extract_toml_string_value(after_key: &str) -> String {
    // after_key is everything after the key name: ` = "value"` or ` = "value" # comment`
    let after_eq = after_key
        .trim_start()
        .strip_prefix('=')
        .unwrap_or("")
        .trim();
    if let Some(inner) = after_eq.strip_prefix('"') {
        // Find closing quote (ignore escaped quotes for simplicity — our values are simple)
        if let Some(end) = inner.find('"') {
            return inner[..end].to_string();
        }
    }
    String::new()
}

/// #205: GPU backends are mutually exclusive — picking two vendors at
/// once produces duplicate-symbol or wrong-backend links. Fail fast at
/// build.rs with a clear message naming the offending features.
///
/// Alias folding (2026-07-18 deep-audit fix): Cargo sets a feature env for
/// EVERY activated feature, including those pulled in by an alias. So
///   - `gpu` = ["gpu-nvidia"]   → GPU + GPU_NVIDIA both set = ONE vendor
///   - `gpu-apple` = ["gpu-vulkan"] → GPU_APPLE + GPU_VULKAN both set —
///     the OLD check counted these as TWO vendors and PANICKED on every
///     `--features gpu-apple` build even though the alias is by-design
///     (MoltenVK: apple IS the vulkan path). Both aliases now fold into
///     their target vendor before counting.
fn assert_at_most_one_gpu_feature() {
    let nvidia = std::env::var("CARGO_FEATURE_GPU_NVIDIA").is_ok()
        || std::env::var("CARGO_FEATURE_GPU").is_ok();
    let apple = std::env::var("CARGO_FEATURE_GPU_APPLE").is_ok();
    // gpu-apple implies gpu-vulkan (same backend via MoltenVK) — one vendor.
    let vulkan = std::env::var("CARGO_FEATURE_GPU_VULKAN").is_ok() || apple;
    let rocm = std::env::var("CARGO_FEATURE_GPU_ROCM").is_ok();
    let selected: Vec<&str> = [
        ("gpu-nvidia", nvidia),
        ("gpu-vulkan/gpu-apple", vulkan),
        ("gpu-rocm", rocm),
    ]
    .iter()
    .filter_map(|(n, on)| if *on { Some(*n) } else { None })
    .collect();
    if selected.len() > 1 {
        panic!(
            "neoethos-app: multiple GPU backends selected ({}). Pick exactly ONE — \
             a dual build produces duplicate/wrong-backend links and wastes link \
             time + binary size. See crates/neoethos-app/Cargo.toml feature \
             block for descriptions of each option.",
            selected.join(", ")
        );
    }
    // Re-run the check when any GPU feature flips.
    for var in &[
        "CARGO_FEATURE_GPU_NVIDIA",
        "CARGO_FEATURE_GPU_VULKAN",
        "CARGO_FEATURE_GPU_ROCM",
        "CARGO_FEATURE_GPU_APPLE",
        "CARGO_FEATURE_GPU",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }
}

/// #205: pre-check that the selected GPU backend has its toolkit
/// installed on this build machine. Without this, the build fails
/// deep inside `llama-cpp-sys-2/build.rs` with a panic that doesn't
/// name the right SDK to install or where to download it from. The
/// upstream message is correct but easy to miss in 200 lines of
/// CMake spew; we surface it earlier with a clickable URL.
///
/// Detection mirrors the env-var contracts the upstream build
/// scripts check:
///   - CUDA  → CUDA_PATH (set by the official installer on Windows)
///             or libcuda.so present (POSIX heuristic via /usr/local/cuda)
///   - Vulkan → VULKAN_SDK (set by the LunarG installer)
///   - ROCm  → HIP_PATH / ROCM_PATH
///   - Metal → no SDK probe (always present on macOS, never on others)
fn assert_gpu_toolkit_available() {
    let nvidia = std::env::var("CARGO_FEATURE_GPU_NVIDIA").is_ok()
        || std::env::var("CARGO_FEATURE_GPU").is_ok();
    let vulkan = std::env::var("CARGO_FEATURE_GPU_VULKAN").is_ok();
    let rocm = std::env::var("CARGO_FEATURE_GPU_ROCM").is_ok();
    let apple = std::env::var("CARGO_FEATURE_GPU_APPLE").is_ok();

    if nvidia
        && std::env::var("CUDA_PATH").is_err()
        && !std::path::Path::new("/usr/local/cuda").exists()
    {
        panic!(
            "neoethos-app: gpu-nvidia selected but the CUDA toolkit is not on \
             this machine. Install from https://developer.nvidia.com/cuda-downloads \
             then re-run cargo build. (Probed CUDA_PATH env var and /usr/local/cuda.)"
        );
    }
    // 2026-07-18 deep-audit fix: the Vulkan SDK hard-requirement dated from
    // the removed llama backend (ggml compiled Vulkan shaders at BUILD time).
    // The current Vulkan path is cubecl-wgpu + burn-wgpu, which compile WGSL
    // through naga at RUNTIME — NO SDK is needed to build. The old panic
    // blocked perfectly-valid gpu-vulkan builds on machines without the SDK.
    if vulkan && std::env::var("VULKAN_SDK").is_err() {
        println!(
            "cargo:warning=neoethos-app: gpu-vulkan build without VULKAN_SDK — fine: \
             the wgpu path needs no SDK at build time (runtime uses the driver's ICD)."
        );
    }
    if rocm && std::env::var("HIP_PATH").is_err() && std::env::var("ROCM_PATH").is_err() {
        panic!(
            "neoethos-app: gpu-rocm selected but the ROCm toolkit is not on this \
             machine. Install from https://rocm.docs.amd.com/projects/install-on-linux/ \
             then re-run cargo build. (Probed HIP_PATH and ROCM_PATH env vars.)\n\
             Note: ROCm on Windows is experimental — Linux is the supported path."
        );
    }
    // gpu-apple is an alias for gpu-vulkan (MoltenVK) since the Metal/llama
    // backend was removed — it builds anywhere the vulkan path builds, so
    // the old macOS-only panic no longer applies.
    let _ = apple;
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
}

// REMOVED 2026-08-02: `force_link_libtorch_cuda()`.
//
// It emitted, for every `--features gpu`/`gpu-nvidia` build with LIBTORCH set:
//     /INCLUDE:?warp_size@cuda@at@@YAHXZ      (MSVC)
//     -Wl,--no-as-needed -ltorch_cuda         (GNU)
// to stop the linker stripping a library nothing referenced. That was correct
// while `tch` was linked. It is not any more: `tch` is optional in
// neoethos-models and enabled by NO feature, every call site is
// `#[cfg(feature = "tch")]`, and d4df966a dropped the `dep:tch` that
// neoethos-search once had.
//
// So on MSVC `/INCLUDE:` was asking link.exe to resolve a symbol from a
// library that was never on the link line — LNK2001 unresolved external,
// then LNK1120, i.e. the next Windows GPU release build would have failed at
// the link step. Deleting this is a FIX, not a loss of capability: there is
// no tch to keep.
//
// If tch ever comes back, this belongs in tch-rs's own build script, not
// here.
