use std::fs;
use std::path::PathBuf;

#[test]
fn gpu_nvidia_discovery_job_reaches_the_real_resident_materializer() {
    let source = fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/app_services/discovery.rs"),
    )
    .expect("read app Discovery production source");

    assert!(
        !source.contains("full native Discovery workspace-plan sealing is not integrated"),
        "the physical-GPU app route still stops before sealing its native workspace"
    );
    assert!(
        source.contains("preflight_gpu_only_feature_workspace_v3"),
        "the app route must consume its exact pinned series into Data's resident preflight"
    );
    assert!(
        source.contains("materialize_gpu_only_feature_store_v3"),
        "the app route must move the admitted CUDA run into Data's resident materializer"
    );
}
