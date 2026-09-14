//! Compile existing producers through the shared, official HIPIFY build path.
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const CUDA_SOURCE: &str = "native/resident_session_v2.cu";
const CUDA_ABI: &str = "native/resident_session_v2_abi.cuh";
const PRECISION: &str = "f64-no-fast-math-no-contract-ieee-denormals";

pub fn emit_rerun_contract() {
    for file in [CUDA_SOURCE, CUDA_ABI] {
        println!("cargo:rerun-if-changed={file}");
    }
    println!("cargo:rerun-if-env-changed=NEOETHOS_HIP_ARCH");
}

fn target(value: &str) -> Result<&str, String> {
    let suffix = value
        .strip_prefix("gfx")
        .ok_or("HIP target must be an explicit gfx architecture")?;
    if !(3..=6).contains(&suffix.len()) || !suffix.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(
            "HIP target must be one exact gfx architecture, without flags or aliases".into(),
        );
    }
    Ok(value)
}

pub fn compile(root: &Path, output: &Path) -> Result<(PathBuf, String), String> {
    let requested = env::var("NEOETHOS_HIP_ARCH")
        .map_err(|_| "HIP kernels require explicit NEOETHOS_HIP_ARCH")?;
    let target = target(&requested)?.to_owned();
    let full = env::var_os("CARGO_FEATURE_HIP_NATIVE_KERNELS").is_some();
    let sources: &[&str] = if full {
        &super::super::DEVICE_SOURCES
    } else {
        &[CUDA_SOURCE]
    };
    let fixtures = env::var_os("CARGO_FEATURE_HIP_DEVICE_FIXTURES").is_some();
    let built = super::hip_kernels::compile(root, output, &target, sources, fixtures)?;
    let source = super::super::artifact_metadata(Path::new(CUDA_SOURCE))?.sha256;
    let abi = super::super::artifact_metadata(Path::new(CUDA_ABI))?.sha256;
    let artifact = &built.artifact_sha256;
    let manifest = format!(
        "{{\"schema\":\"neoethos.hip-session-kernels-build.v1\",\"backend\":\"amd-hip\",\"target\":\"{target}\",\"semantic_version\":2,\"source_sha256\":\"{source}\",\"abi_sha256\":\"{abi}\",\"artifact_sha256\":\"{artifact}\",\"precision\":\"{PRECISION}\",\"device_executed\":false}}"
    );
    fs::write(output.join("neoethos_hip_session_build_v1.json"), &manifest)
        .map_err(|e| e.to_string())?;
    println!("cargo:rustc-env=NEOETHOS_HIP_SESSION_BUILD_MANIFEST_V1={manifest}");
    if full {
        let smc_source =
            super::super::artifact_metadata(Path::new("native/resident_smc_v3.cu"))?.sha256;
        let smc_abi =
            super::super::artifact_metadata(Path::new("hip/hip_runtime_owner_v1.h"))?.sha256;
        let smc_manifest = format!(
            "{{\"schema\":\"neoethos.hip-smc-kernels-build.v1\",\"backend\":\"amd-hip\",\"target\":\"{target}\",\"semantic_version\":3,\"source_sha256\":\"{smc_source}\",\"abi_sha256\":\"{smc_abi}\",\"artifact_sha256\":\"{artifact}\",\"precision\":\"{PRECISION}\",\"device_executed\":false}}"
        );
        fs::write(output.join("neoethos_hip_smc_build_v1.json"), &smc_manifest)
            .map_err(|e| e.to_string())?;
        println!("cargo:rustc-env=NEOETHOS_HIP_SMC_BUILD_MANIFEST_V1={smc_manifest}");
        let compiler = fs::read_to_string(output.join("hip-runtime-compiler-version.stdout.log"))
            .map_err(|e| e.to_string())?;
        let manifest = format!(
            "{{\"schema\":\"neoethos.hip-native-kernels-build.v1\",\"backend\":\"amd-hip\",\"target\":\"{target}\",\"artifact_sha256\":\"{artifact}\",\"precision\":\"{PRECISION}\",\"compiler\":\"{}\",\"sources\":{},\"device_fixtures\":{fixtures},\"device_executed\":false}}",
            super::super::json_escape(compiler.trim()),
            built.sources_json
        );
        fs::write(output.join("neoethos_hip_native_build_v1.json"), &manifest)
            .map_err(|e| e.to_string())?;
        println!("cargo:rustc-env=NEOETHOS_HIP_NATIVE_BUILD_MANIFEST_V1={manifest}");
        super::hip_kernels::embed_manifest_sha256(root, output, &manifest)?;
    }
    Ok((built.native_directory, target))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn session_target_is_exact_not_an_injected_flag() {
        for arch in ["gfx90a", "gfx942", "gfx950", "gfx1100"] {
            assert_eq!(target(arch).unwrap(), arch);
        }
        for arch in [
            "",
            "native",
            "sm_86",
            "gfx",
            "gfx942,gfx950",
            "gfx942 -ffast-math",
            "gfx942:xnack+",
        ] {
            assert!(target(arch).is_err());
        }
    }
}
