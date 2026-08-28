use std::fs;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-models"))
        .parent()
        .and_then(|path| path.parent())
        .expect("models manifest must be below the workspace root")
        .to_path_buf()
}

fn source() -> String {
    let path = workspace_root().join("crates/neoethos-models/src/training_orchestrator.rs");
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read required source {}: {error}", path.display()))
}

fn registry_source() -> String {
    let path = workspace_root().join("crates/neoethos-models/src/registry.rs");
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read required source {}: {error}", path.display()))
}

fn function_body<'a>(source: &'a str, marker: &str) -> &'a str {
    let start = source
        .find(marker)
        .unwrap_or_else(|| panic!("missing function marker {marker:?}"));
    let open = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .expect("function has an opening brace");
    let mut depth = 0_u32;
    for (offset, byte) in source.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("function {marker:?} has no closing brace")
}

#[test]
fn one_authority_rejects_every_model_without_a_compiled_gpu_route() {
    let source = source();
    let validator = function_body(&source, "fn validate_nvidia_model_config_v1(");

    for required in [
        "supports_nvidia_cuda_for_model",
        "canonical_model_name",
        "full_nvidia_device_policy_for_config",
        "CudaDevicePolicy::Gpu { ordinal: 0 }",
        "has no compiled NVIDIA CUDA implementation",
    ] {
        assert!(
            validator.contains(required),
            "strict full-GPU validator is missing `{required}`"
        );
    }
    for forbidden in [
        "CudaDevicePolicy::Cpu",
        "CPU-only",
        "pin_cpu_only_model_device",
    ] {
        assert!(
            !validator.contains(forbidden),
            "strict full-GPU validator still permits CPU substitution through `{forbidden}`"
        );
    }
}

#[test]
fn nvidia_cuda_authority_never_accepts_a_wgpu_only_burn_build() {
    let registry = registry_source();
    let cuda = function_body(&registry, "pub fn supports_nvidia_cuda_for_model(");

    assert!(cuda.contains("feature = \"burn-cuda-backend\""));
    assert!(
        !cuda.contains("feature = \"burn-wgpu-backend\""),
        "NVIDIA CUDA admission must not treat a WGPU-only Burn build as CUDA-capable"
    );
    for required in ["ModelFamily::Deep", "ModelFamily::Exit", "\"sac\""] {
        assert!(
            cuda.contains(required),
            "NVIDIA CUDA authority is missing the Burn route `{required}`"
        );
    }
}

#[test]
fn nvidia_cuda_authority_never_accepts_a_cpu_only_xgboost_build() {
    let registry = registry_source();
    let cuda = function_body(&registry, "pub fn supports_nvidia_cuda_for_model(");
    let xgboost = cuda
        .split("\"xgboost\"")
        .nth(1)
        .expect("NVIDIA CUDA authority is missing XGBoost")
        .split("\"catboost\"")
        .next()
        .expect("CatBoost mapping must follow XGBoost");

    assert!(
        xgboost.contains("feature = \"gpu-cuda\""),
        "XGBoost CUDA admission must require the aggregate that enables xgb/cuda"
    );
    assert!(
        !xgboost.contains("cfg!(feature = \"xgboost\")"),
        "the CPU-only xgboost feature must not satisfy NVIDIA CUDA admission"
    );
}

#[test]
fn xgboost_gpu_preference_reuses_the_same_cuda_authority() {
    let registry = registry_source();
    let prefers = function_body(&registry, "pub fn prefers_gpu_for_model(");
    let xgboost = prefers
        .split("\"xgboost\"")
        .nth(1)
        .expect("GPU preference registry is missing XGBoost")
        .split("\"catboost\"")
        .next()
        .expect("CatBoost preference must follow XGBoost");

    assert!(
        xgboost.contains("supports_nvidia_cuda_for_model(name, family)"),
        "XGBoost preference must reuse the strict CUDA feature authority"
    );
    assert!(
        !xgboost.contains("cfg!(feature = \"xgboost\")"),
        "CPU-only XGBoost must not be preferred as a GPU route"
    );
}

#[test]
fn every_gpu_registry_query_rejects_unknown_or_mismatched_name_family_pairs() {
    let registry = registry_source();
    for marker in [
        "pub fn supports_nvidia_cuda_for_model(",
        "pub fn supports_gpu_for_model(",
        "pub fn prefers_gpu_for_model(",
    ] {
        let body = function_body(&registry, marker);
        assert!(
            body.contains("known_model_family_matches(name, family)"),
            "GPU registry query `{marker}` must reject unknown names and mismatched families"
        );
    }
}

#[test]
fn every_full_nvidia_preflight_validates_every_config_through_the_same_authority() {
    let source = source();
    for marker in [
        "pub fn preflight_full_nvidia_cuda_training(&self)",
        "pub fn preflight_configured_nvidia_training(&self)",
    ] {
        let preflight = function_body(&source, marker);
        assert!(
            preflight.contains("self.validate_nvidia_model_config_v1(&config)?"),
            "{marker} does not validate every configured model through the strict GPU authority"
        );
        for forbidden in [
            "model_requires_cuda_in_full_nvidia_run",
            "CudaDevicePolicy::Cpu",
            "CPU-only model",
            "configured_cuda_models",
        ] {
            assert!(
                !preflight.contains(forbidden),
                "{marker} retains a partial-GPU escape through `{forbidden}`"
            );
        }
    }
}

#[test]
fn ordinary_cpu_planning_remains_separate_from_full_gpu_admission() {
    let source = source();
    let planner = function_body(&source, "fn pin_cpu_only_model_device(");
    let configured = function_body(
        &source,
        "pub fn preflight_configured_nvidia_training(&self)",
    );
    let full = function_body(&source, "pub fn preflight_full_nvidia_cuda_training(&self)");

    assert!(
        planner.contains("\"device\"") && planner.contains("\"cpu\""),
        "ordinary CPU-capable training lost its explicit CPU plan"
    );
    assert!(
        !configured.contains("pin_cpu_only_model_device")
            && !full.contains("pin_cpu_only_model_device"),
        "a full-GPU preflight must never reinterpret ordinary CPU planning as GPU readiness"
    );
}
