use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("models crate must be two levels below the workspace root")
        .to_path_buf()
}

fn read_workspace_file(relative: &str) -> String {
    let path = workspace_root().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read required workspace file {}: {error}", path.display()))
}

#[test]
fn retired_wgpu_vulkan_packages_are_absent_from_the_resolved_workspace() {
    let lock = read_workspace_file("Cargo.lock");

    for retired_package in [
        "burn-wgpu",
        "cubecl-wgpu",
        "wgpu",
        "wgpu-core",
        "wgpu-hal",
        "wgpu-naga-bridge",
    ] {
        assert!(
            !lock.contains(&format!("name = \"{retired_package}\"")),
            "retired package `{retired_package}` returned to Cargo.lock"
        );
    }
}

#[test]
fn cubecl_facade_exposes_cuda_and_future_hip_without_a_wgpu_family() {
    let manifest = read_workspace_file("vendor/cubecl-0.10.0-cuda-hip-only/Cargo.toml");
    let source = read_workspace_file("vendor/cubecl-0.10.0-cuda-hip-only/src/lib.rs");

    for required in [
        "cuda = [\"dep:cubecl-cuda\"]",
        "hip = [\"dep:cubecl-hip\"]",
        "rocm = [\"hip\"]",
        "pub use cubecl_cuda as cuda;",
        "pub use cubecl_hip as hip;",
    ] {
        assert!(
            manifest.contains(required) || source.contains(required),
            "CUDA/HIP-only CubeCL facade is missing `{required}`"
        );
    }

    for retired in [
        "cubecl-wgpu",
        "cubecl_wgpu",
        "wgpu =",
        "vulkan =",
        "webgpu =",
        "metal =",
        "spirv-dump =",
    ] {
        assert!(
            !manifest.contains(retired) && !source.contains(retired),
            "CubeCL facade restored retired backend token `{retired}`"
        );
    }
}

#[test]
fn burn_facade_cannot_reintroduce_optional_wgpu_backend_edges() {
    let workspace_manifest = read_workspace_file("Cargo.toml");
    let models_manifest = read_workspace_file("crates/neoethos-models/Cargo.toml");
    let burn_manifest = read_workspace_file("vendor/burn-0.21.0-neoethos/Cargo.toml");

    assert!(workspace_manifest.contains("burn = { path = \"vendor/burn-0.21.0-neoethos\" }"));
    assert!(
        workspace_manifest.contains("cubecl = { path = \"vendor/cubecl-0.10.0-cuda-hip-only\" }")
    );
    assert!(models_manifest.contains(
        "burn = { version = \"0.21\", default-features = false, features = [\"std\", \"autodiff\"] }"
    ));

    for retired in ["burn-wgpu", "wgpu =", "vulkan =", "webgpu =", "metal ="] {
        assert!(
            !burn_manifest.contains(retired),
            "Burn facade restored retired backend token `{retired}`"
        );
    }
}
