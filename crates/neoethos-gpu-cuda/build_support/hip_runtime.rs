//! Host-only AMD HIP ownership adapter. No kernels or GPU-readiness manifest.

use std::env;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[path = "hip_kernels.rs"]
mod hip_kernels;
#[path = "hip_session.rs"]
mod hip_session;

const SOURCES: [&str; 4] = [
    "hip/hip_runtime_owner_v1.cpp",
    "hip/hip_runtime_owner_v1.h",
    "hip/hip_runtime_lifecycle_v1.hpp",
    "hip/hip_runtime_buffers_v1.hpp",
];
const FORBIDDEN_ENV: [&str; 11] = [
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "C_INCLUDE_PATH",
    "LIBRARY_PATH",
    "CCC_OVERRIDE_OPTIONS",
    "CLANG_CONFIG_FILE_USER_DIR",
    "HIPCC_COMPILE_FLAGS_APPEND",
    "HIPCC_LINK_FLAGS_APPEND",
    "HSA_OVERRIDE_GFX_VERSION",
    "PERL5LIB",
    "PERL5OPT",
];

pub fn emit_rerun_contract(enabled: bool) {
    println!("cargo:rerun-if-changed=build_support/hip_runtime.rs");
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    println!("cargo:rerun-if-changed=build_support/hip_session.rs");
    println!("cargo:rerun-if-changed=build_support/hip_kernels.rs");
    // Editing HIP-only native code must not rebuild all CUDA translation units.
    if !enabled {
        return;
    }
    if env::var_os("CARGO_FEATURE_HIP_SESSION_KERNELS").is_some() {
        hip_session::emit_rerun_contract();
    }
    for source in SOURCES {
        println!("cargo:rerun-if-changed={source}");
    }
    for key in FORBIDDEN_ENV
        .into_iter()
        .chain(["ROCM_PATH", "HIP_PLATFORM"])
    {
        println!("cargo:rerun-if-env-changed={key}");
    }
}

pub fn validate_target(host: &str, target: &str) -> Result<(), String> {
    if host == "x86_64-unknown-linux-gnu" && target == host {
        Ok(())
    } else {
        Err(format!(
            "hip-runtime requires a native x86_64 Linux GNU build; host={host}, target={target}"
        ))
    }
}

pub fn compile_arguments(include: &Path, source: &Path, object: &Path) -> Vec<String> {
    vec![
        "--no-default-config".into(),
        "--target=x86_64-unknown-linux-gnu".into(),
        "-x".into(),
        "c++".into(),
        "-std=c++17".into(),
        "-O2".into(),
        "-fPIC".into(),
        "-fvisibility=hidden".into(),
        "-Wall".into(),
        "-Wextra".into(),
        "-Werror".into(),
        "-D__HIP_PLATFORM_AMD__=1".into(),
        "-I".into(),
        include.to_string_lossy().into_owned(),
        "-c".into(),
        source.to_string_lossy().into_owned(),
        "-o".into(),
        object.to_string_lossy().into_owned(),
    ]
}

// Persist and display BOTH complete streams, including successful warnings.
// These host compiler/archive children have a finite deadline and no shell.
fn run(tool: &Path, args: &[String], output: &Path, label: &str) -> Result<String, String> {
    run_with_timeout(tool, args, output, label, Duration::from_secs(120))
}

// A compiler driver can fork cc1/linker children. Give only this invocation a
// fresh process group; timing out the driver alone leaves writers behind.
#[cfg(target_os = "linux")]
fn signal_process_group(child: &Child, signal: i32) -> Result<bool, String> {
    unsafe extern "C" {
        fn kill(pid: i32, signal: i32) -> i32;
    }
    let group = i32::try_from(child.id()).map_err(|e| e.to_string())?;
    if group <= 1 {
        return Err("refusing invalid compiler process group".into());
    }
    // SAFETY: process_group(0) assigned this exact owned child's PID as PGID.
    // No shell, caller-supplied PID, or process-wide signal target is involved.
    if unsafe { kill(-group, signal) } == 0 {
        Ok(true)
    } else {
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(3) {
            // Linux ESRCH: no group remains.
            Ok(false)
        } else {
            Err(format!("compiler process-group signal failed: {error}"))
        }
    }
}

fn stop_child(child: &mut Child) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    let _ = signal_process_group(child, 9)?;
    #[cfg(not(target_os = "linux"))]
    child.kill().map_err(|e| e.to_string())?;
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let reaped = child.try_wait().map_err(|e| e.to_string())?.is_some();
        #[cfg(target_os = "linux")]
        let retired = !signal_process_group(child, 0)?;
        #[cfg(not(target_os = "linux"))]
        let retired = reaped;
        if reaped && retired {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err("compiler group did not retire; logs/artifacts are not closed".into());
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn run_with_timeout(
    tool: &Path,
    args: &[String],
    output: &Path,
    label: &str,
    timeout: Duration,
) -> Result<String, String> {
    let stdout_path = output.join(format!("hip-runtime-{label}.stdout.log"));
    let stderr_path = output.join(format!("hip-runtime-{label}.stderr.log"));
    let stdout = File::create(&stdout_path).map_err(|e| e.to_string())?;
    let stderr = File::create(&stderr_path).map_err(|e| e.to_string())?;
    let mut command = Command::new(tool);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(stderr);
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let mut child = command
        .spawn()
        .map_err(|e| format!("{}: {e}", tool.display()))?;
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                #[cfg(target_os = "linux")]
                if signal_process_group(&child, 0)? {
                    stop_child(&mut child)?;
                    return Err(format!(
                        "HIP runtime {label} driver exited before its children"
                    ));
                }
                break Some(status);
            }
            Ok(None) => (),
            Err(error) => {
                stop_child(&mut child)?;
                return Err(format!("HIP runtime {label} wait failed: {error}"));
            }
        }
        if Instant::now() >= deadline {
            stop_child(&mut child)?;
            break None;
        }
        thread::sleep(Duration::from_millis(20));
    };
    let out = fs::read_to_string(&stdout_path).map_err(|e| e.to_string())?;
    let err = fs::read_to_string(&stderr_path).map_err(|e| e.to_string())?;
    if !out.is_empty() {
        eprint!("{out}");
    }
    if !err.is_empty() {
        eprint!("{err}");
    }
    if !status.is_some_and(|status| status.success()) {
        return Err(format!(
            "HIP runtime {label} failed ({status:?}, deadline={}s); complete logs: {} and {}",
            timeout.as_secs(),
            stdout_path.display(),
            stderr_path.display()
        ));
    }
    Ok(out)
}

pub fn compile() -> Result<(), String> {
    validate_target(
        &env::var("HOST").map_err(|e| e.to_string())?,
        &env::var("TARGET").map_err(|e| e.to_string())?,
    )?;
    for key in FORBIDDEN_ENV {
        if env::var_os(key).is_some_and(|value| !value.is_empty()) {
            return Err(format!("unset implicit HIP runtime build input {key}"));
        }
    }
    if env::var_os("HIP_PLATFORM").is_some_and(|value| value != "amd") {
        return Err("hip-runtime supports AMD HIP only, not a CUDA compatibility backend".into());
    }
    let root =
        PathBuf::from(env::var_os("ROCM_PATH").ok_or("hip-runtime requires explicit ROCM_PATH")?)
            .canonicalize()
            .map_err(|e| format!("ROCM_PATH: {e}"))?;
    let compiler = root.join("llvm/bin/clang++");
    let archiver = root.join("llvm/bin/llvm-ar");
    let include = root.join("include");
    let library = root.join("lib");
    // The API header includes version/platform headers. Track the tree so an
    // in-place SDK update cannot reuse an archive built against older layouts.
    println!("cargo:rerun-if-changed={}", include.join("hip").display());
    for path in [
        &compiler,
        &archiver,
        &include.join("hip/hip_runtime_api.h"),
        &library.join("libamdhip64.so"),
    ] {
        if !path.is_file() {
            return Err(format!("required ROCm file missing: {}", path.display()));
        }
        println!("cargo:rerun-if-changed={}", path.display());
    }
    let output = PathBuf::from(env::var_os("OUT_DIR").ok_or("Cargo OUT_DIR missing")?);
    let source_records = SOURCES
        .into_iter()
        .map(|source| Ok((source, super::artifact_metadata(Path::new(source))?.sha256)))
        .collect::<Result<Vec<_>, String>>()?;
    let compiler_version = run(
        &compiler,
        &["--no-default-config".into(), "--version".into()],
        &output,
        "compiler-version",
    )?;
    let object = output.join("neoethos_hip_runtime_v1.o");
    let session = if env::var_os("CARGO_FEATURE_HIP_SESSION_KERNELS").is_some() {
        Some(hip_session::compile(&root, &output)?)
    } else {
        None
    };
    let mut arguments = compile_arguments(&include, Path::new(SOURCES[0]), &object);
    if let Some((directory, target)) = &session {
        arguments.extend([
            "-I".into(),
            directory.to_string_lossy().into_owned(),
            "-DNEOETHOS_HIP_SESSION_KERNELS_V1=1".into(),
            format!("-DNEOETHOS_HIP_SESSION_TARGET_V1=\"{target}\""),
            "-Dneoethos_resident_session_f64_v2=neoethos_hip_resident_session_f64_v2".into(),
        ]);
    }
    if env::var_os("CARGO_FEATURE_HIP_NATIVE_KERNELS").is_some() {
        arguments.push("-DNEOETHOS_HIP_NATIVE_KERNELS_V1=1".into());
        let (_, target) = session
            .as_ref()
            .ok_or("native HIP kernels require Session archive")?;
        arguments.push(format!("-DNEOETHOS_HIP_NATIVE_TARGET_V1=\"{target}\""));
    }
    run(&compiler, &arguments, &output, "compile")?;
    let archive = output.join("libneoethos_hip_runtime_v1.a");
    run(
        &archiver,
        &[
            "rcs".into(),
            archive.to_string_lossy().into_owned(),
            object.to_string_lossy().into_owned(),
        ],
        &output,
        "archive",
    )?;
    let artifact = super::artifact_metadata(&archive)?;
    let source_pins = source_records
        .into_iter()
        .map(|(source, before)| {
            let pin = super::artifact_metadata(Path::new(source))?;
            if pin.sha256 != before {
                return Err(format!("HIP runtime source changed during build: {source}"));
            }
            Ok(format!(
                "{{\"path\":\"{source}\",\"sha256\":\"{}\"}}",
                pin.sha256
            ))
        })
        .collect::<Result<Vec<_>, String>>()?
        .join(",");
    let manifest = format!(
        "{{\"schema\":\"neoethos.hip-runtime-host-build.v1\",\"backend\":\"amd-hip\",\"scope\":\"host runtime ownership only; no device kernels or Search admission\",\"compiler\":\"{}\",\"artifact_sha256\":\"{}\",\"sources\":[{source_pins}]}}",
        super::json_escape(compiler_version.trim()),
        artifact.sha256
    );
    fs::write(output.join("neoethos_hip_runtime_build_v1.json"), &manifest)
        .map_err(|e| e.to_string())?;
    println!("cargo:rustc-env=NEOETHOS_HIP_RUNTIME_BUILD_MANIFEST_V1={manifest}");
    println!("cargo:rustc-link-search=native={}", output.display());
    println!("cargo:rustc-link-search=native={}", library.display());
    println!("cargo:rustc-link-lib=static=neoethos_hip_runtime_v1");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn compiler_timeout_stops_its_grandchild_not_an_unrelated_group() {
        use std::os::unix::process::CommandExt;
        let output = env::temp_dir().join(format!(
            "neoethos-hip-timeout-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&output).unwrap();
        let mut unrelated = Command::new("/bin/sleep")
            .arg("10")
            .process_group(0)
            .spawn()
            .unwrap();
        let error = run_with_timeout(
            Path::new("/bin/sh"),
            &[
                "-c".into(),
                "printf 'early info\\n'; (sleep 1; printf 'late child\\n') & wait".into(),
            ],
            &output,
            "descendant",
            Duration::from_millis(100),
        )
        .unwrap_err();
        assert!(
            error.contains("deadline") || error.contains("not closed"),
            "{error}"
        );
        assert!(unrelated.try_wait().unwrap().is_none());
        stop_child(&mut unrelated).unwrap();
        thread::sleep(Duration::from_millis(1100));
        let out = fs::read_to_string(output.join("hip-runtime-descendant.stdout.log")).unwrap();
        assert_eq!(out, "early info\n");
        for channel in ["stdout", "stderr"] {
            fs::remove_file(output.join(format!("hip-runtime-descendant.{channel}.log"))).unwrap();
        }
        fs::remove_dir(output).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn compiler_success_and_failure_retain_both_complete_streams() {
        let output = env::temp_dir().join(format!(
            "neoethos-hip-streams-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&output).unwrap();
        for code in [0, 7] {
            let result = run_with_timeout(
                Path::new("/bin/sh"),
                &[
                    "-c".into(),
                    format!("printf 'info\\n'; printf 'warning\\n' >&2; exit {code}"),
                ],
                &output,
                "streams",
                Duration::from_secs(2),
            );
            assert_eq!(result.is_ok(), code == 0);
            assert_eq!(
                fs::read_to_string(output.join("hip-runtime-streams.stdout.log")).unwrap(),
                "info\n"
            );
            assert_eq!(
                fs::read_to_string(output.join("hip-runtime-streams.stderr.log")).unwrap(),
                "warning\n"
            );
        }
        for channel in ["stdout", "stderr"] {
            fs::remove_file(output.join(format!("hip-runtime-streams.{channel}.log"))).unwrap();
        }
        fs::remove_dir(output).unwrap();
    }

    #[test]
    fn hip_runtime_is_native_linux_only_not_a_cross_compile_shortcut() {
        assert!(validate_target("x86_64-unknown-linux-gnu", "x86_64-unknown-linux-gnu").is_ok());
        for (host, target) in [
            ("x86_64-pc-windows-msvc", "x86_64-unknown-linux-gnu"),
            ("x86_64-unknown-linux-gnu", "x86_64-pc-windows-msvc"),
            ("aarch64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"),
        ] {
            assert!(validate_target(host, target).is_err());
        }
    }

    #[test]
    fn hip_runtime_build_is_host_api_only_with_no_implicit_device_target() {
        let args = compile_arguments(
            Path::new("/rocm with spaces/include"),
            Path::new("hip/owner.cpp"),
            Path::new("/out/owner.o"),
        );
        assert_eq!(args[0], "--no-default-config");
        assert!(args.windows(2).any(|pair| pair == ["-x", "c++"]));
        assert!(
            args.windows(2)
                .any(|pair| pair == ["-I", "/rocm with spaces/include"])
        );
        assert!(args.iter().any(|arg| arg == "-D__HIP_PLATFORM_AMD__=1"));
        assert!(args.iter().any(|arg| arg == "-Werror"));
        assert!(!args.iter().any(|arg| arg.contains("offload")
            || arg.contains("sm_")
            || arg.contains("fast-math")));
    }

    #[test]
    fn hip_runtime_sources_do_not_enter_cuda_hipify_translation_inventory() {
        assert!(SOURCES.iter().all(|source| source.starts_with("hip/")));
        assert!(!SOURCES.iter().any(|source| source.starts_with("native/")));
        assert!(FORBIDDEN_ENV.contains(&"HSA_OVERRIDE_GFX_VERSION"));
        assert!(FORBIDDEN_ENV.contains(&"CCC_OVERRIDE_OPTIONS"));
    }
}
