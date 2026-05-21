use std::io::Write;
use std::path::{Path, PathBuf};

fn main() {
    emit_embedded_credentials();
    force_link_libtorch_cuda();

    let protoc_path = protoc_bin_vendored::protoc_bin_path().unwrap();
    let protoc_dir = protoc_path.parent().unwrap();

    // Add protoc directory to PATH because protobuf-codegen v4.31 hardcodes "protoc" command
    let path = std::env::var_os("PATH").unwrap_or_default();
    let mut paths = std::env::split_paths(&path).collect::<Vec<_>>();
    paths.insert(0, protoc_dir.to_path_buf());
    let new_path = std::env::join_paths(paths).unwrap();
    unsafe {
        std::env::set_var("PATH", new_path);
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let proto_temp_dir = out_dir.join("proto_temp");
    std::fs::create_dir_all(&proto_temp_dir).unwrap();

    // Copy .proto files to temp dir
    let files = [
        "OpenApiCommonModelMessages.proto",
        "OpenApiModelMessages.proto",
        "OpenApiCommonMessages.proto",
        "OpenApiMessages.proto",
    ];
    for file in &files {
        std::fs::copy(format!("proto/{}", file), proto_temp_dir.join(file)).unwrap();
    }

    let gen_dir = out_dir.join("protobuf_generated");
    std::fs::create_dir_all(&gen_dir).unwrap();

    generate_protobuf_sources(&proto_temp_dir, &gen_dir, &files);
    compile_generated_protobuf_sources(&gen_dir, &files);

    // Post-process generated files for Rust 2024 compatibility (unsafe extern blocks)
    fn patch_dir(dir: &std::path::Path) {
        if !dir.exists() {
            return;
        }
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                patch_dir(&path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let content = std::fs::read_to_string(&path).unwrap();
                let patched = content.replace("extern \"C\" {", "unsafe extern \"C\" {");
                std::fs::write(&path, patched).unwrap();
            }
        }
    }
    patch_dir(&gen_dir);
}

fn generate_protobuf_sources(proto_temp_dir: &Path, gen_dir: &Path, files: &[&str]) {
    let crate_mapping_path = gen_dir.join("crate_mapping.txt");
    std::fs::File::create(&crate_mapping_path)
        .and_then(|mut file| file.write_all(b""))
        .expect("failed to create protobuf crate mapping");

    let mut cmd = std::process::Command::new("protoc");
    for file in files {
        cmd.arg(file);
    }
    cmd.arg(format!("--rust_out={}", gen_dir.display()))
        .arg("--rust_opt=experimental-codegen=enabled,kernel=upb")
        .arg(format!("--upb_minitable_out={}", gen_dir.display()))
        .arg(format!("--proto_path={}", proto_temp_dir.display()))
        .arg(format!(
            "--rust_opt=crate_mapping={}",
            crate_mapping_path.display()
        ));

    println!("cargo:rerun-if-changed={}", proto_temp_dir.display());
    let output = cmd.output().expect("failed to run protoc");
    if !output.status.success() {
        panic!(
            "protobuf codegen failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn compile_generated_protobuf_sources(gen_dir: &Path, files: &[&str]) {
    let mut cc_build = cc::Build::new();
    cc_build
        .include(
            std::env::var_os("DEP_UPB_INCLUDE")
                .expect("DEP_UPB_INCLUDE should be set by the protobuf crate"),
        )
        .include(gen_dir);

    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
        cc_build.flag("-std=c99");
    }

    for file in files {
        let c_file = generated_c_file(gen_dir, file);
        if !c_file.exists() {
            panic!(
                "expected generated file {} does not exist",
                c_file.display()
            );
        }
        println!("cargo:rerun-if-changed={}", c_file.display());
        cc_build.file(c_file);
    }

    cc_build.compile(&format!(
        "{}_upb_gen_code",
        std::env::var("CARGO_PKG_NAME").unwrap()
    ));
}

fn generated_c_file(gen_dir: &Path, proto_file: &str) -> PathBuf {
    let mut path = PathBuf::from(proto_file);
    assert!(path.set_extension("upb_minitable.c"));
    gen_dir.join(path)
}

/// Generates `$OUT_DIR/embedded_credentials.rs` with compile-time cTrader
/// Open API credentials that are baked into the binary for distribution.
///
/// Resolution order (first non-empty value wins for each field):
/// 1. `FOREX_AI_EMBED_CTRADER_CLIENT_ID` / `_CLIENT_SECRET` / `_REDIRECT_URI`
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
    println!("cargo:rerun-if-env-changed=FOREX_AI_EMBED_CTRADER_CLIENT_ID");
    println!("cargo:rerun-if-env-changed=FOREX_AI_EMBED_CTRADER_CLIENT_SECRET");
    println!("cargo:rerun-if-env-changed=FOREX_AI_EMBED_CTRADER_REDIRECT_URI");
    let local_toml = workspace_root.join(".local/neoethos/broker_credentials.toml");
    println!("cargo:rerun-if-changed={}", local_toml.display());

    // --- Step 1: env vars ---
    let mut client_id = std::env::var("FOREX_AI_EMBED_CTRADER_CLIENT_ID")
        .unwrap_or_default()
        .trim()
        .to_string();
    let mut client_secret = std::env::var("FOREX_AI_EMBED_CTRADER_CLIENT_SECRET")
        .unwrap_or_default()
        .trim()
        .to_string();
    let mut redirect_uri = std::env::var("FOREX_AI_EMBED_CTRADER_REDIRECT_URI")
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
    // `FOREX_AI_BUILD_VERBOSE=1` to re-enable for local debugging.
    let verbose = std::env::var("FOREX_AI_BUILD_VERBOSE")
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

/// When the `gpu` feature is enabled, force the linker to keep
/// `libtorch_cuda` so `tch::Cuda::device_count()` actually returns the
/// hardware GPU count at runtime. tch-rs only emits a plain
/// `cargo:rustc-link-lib=torch_cuda` which the linker strips because
/// no symbols from it are referenced — the workaround is the standard
/// `--no-as-needed` link arg pair.
fn force_link_libtorch_cuda() {
    if std::env::var("CARGO_FEATURE_GPU").is_err() {
        return;
    }
    if let Ok(libtorch) = std::env::var("LIBTORCH") {
        println!("cargo:rustc-link-arg-bins=-Wl,--no-as-needed");
        println!("cargo:rustc-link-arg-bins=-L{libtorch}/lib");
        println!("cargo:rustc-link-arg-bins=-ltorch_cuda");
        println!("cargo:rustc-link-arg-bins=-Wl,--as-needed");
        println!("cargo:rerun-if-env-changed=LIBTORCH");
    } else {
        println!(
            "cargo:warning=neoethos-app built with `gpu` feature but LIBTORCH env not set; \
             libtorch_cuda will not be force-linked and tch::Cuda::device_count() may return 0"
        );
    }
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_GPU");
}
