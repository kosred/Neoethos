use std::fs;
use std::path::{Path, PathBuf};

fn repository_root() -> PathBuf {
    if let Some(manifest_dir) = option_env!("CARGO_MANIFEST_DIR") {
        return Path::new(manifest_dir)
            .parent()
            .and_then(Path::parent)
            .expect("neoethos-cli must remain under <repo>/crates")
            .to_path_buf();
    }
    std::env::current_dir().expect("direct source-contract gate must run from repository root")
}

fn packaging_script() -> String {
    let path = repository_root().join("packaging/vps/package-linux-sm120.sh");
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn require_all(source: &str, required: &[&str]) {
    for token in required {
        assert!(source.contains(token), "missing release token {token:?}");
    }
}

#[test]
fn package_only_release_stages_both_linux_binaries_from_one_profile() {
    let script = packaging_script();

    require_all(
        &script,
        &[
            "SOURCE_RELEASE",
            "require_nonempty_file \"${SOURCE_RELEASE}/neoethos-app\"",
            "require_nonempty_file \"${SOURCE_RELEASE}/neoethos-cli\"",
            "install -m 0755 \"${SOURCE_RELEASE}/neoethos-app\"",
            "install -m 0755 \"${SOURCE_RELEASE}/neoethos-cli\"",
        ],
    );
    assert!(
        !script.contains("cargo build"),
        "the paid-card packager must consume the already-built profile"
    );
    assert!(
        !script.contains("x86-64-v3"),
        "the baseline server build must not be mislabeled as a v3 payload"
    );
}

#[test]
fn release_refuses_any_cuda_image_contract_other_than_exact_sm120_sass() {
    let script = packaging_script();

    require_all(
        &script,
        &[
            "neoethos_cuda_build_manifest_v1.json",
            "\"architectures\":[120]",
            "\"sass_targets\":[\"sm_120\"]",
            "\"ptx_targets\":[]",
            "cuda-build-manifest.json",
            "grep -aFq -- \"${CUDA_BUILD_MANIFEST_BYTES}\" \"${SOURCE_RELEASE}/neoethos-app\"",
            "grep -aFq -- \"${CUDA_BUILD_MANIFEST_BYTES}\" \"${SOURCE_RELEASE}/neoethos-cli\"",
        ],
    );
}

#[test]
fn native_and_cuda_runtimes_are_adjacent_and_driver_library_is_forbidden() {
    let script = packaging_script();

    require_all(
        &script,
        &[
            "libxgboost.so",
            "libcatboostmodel.so",
            "libcudart.so",
            "libnvrtc.so",
            "libnvrtc-builtins.so",
            "libcublas.so",
            "libcublasLt.so",
            "libcurand.so",
            "libcusparse.so",
            "libcusolver.so",
            "libnvJitLink.so",
            "libcuda.so",
            "refusing CUDA runtime symlink outside the bundle",
            "rm -f -- \"${runtime_link}\"",
            "ln -- \"${resolved_runtime}\" \"${runtime_link}\"",
            "readelf -d",
            "env -u LD_LIBRARY_PATH ldd",
            "not found",
            "'$ORIGIN'",
        ],
    );
    assert!(
        !script.contains("export LD_LIBRARY_PATH"),
        "the bundle must use executable-relative RUNPATH, not a mutable loader override"
    );
}

#[test]
fn wrappers_pin_config_metadata_and_gpu_requirement_to_bundle_paths() {
    let script = packaging_script();

    require_all(
        &script,
        &[
            "run-neoethos-app.sh",
            "run-neoethos-cli.sh",
            "CONFIG_FILE=\"${BUNDLE_DIR}/config.yaml\"",
            "NEOETHOS_BOT_SYMBOL_METADATA=\"${BUNDLE_DIR}/assets/symbol_metadata/defaults.json\"",
            "unset NEOETHOS_REQUIRE_GPU",
            "prop_search_device: cuda_required",
            "enable_gpu_preference: cuda_required",
            "gpu_only: true",
            "assets/symbol_metadata/defaults.json",
            "exec \"${BUNDLE_DIR}/neoethos-app\" --config \"${CONFIG_FILE}\" \"$@\"",
            "exec \"${BUNDLE_DIR}/neoethos-cli\" \"$@\"",
        ],
    );
    assert!(!script.contains("${CONFIG_FILE:-"));
    assert!(!script.contains("${NEOETHOS_BOT_SYMBOL_METADATA:-"));
    assert!(!script.contains("NEOETHOS_REQUIRE_GPU=1"));
}

#[test]
fn bundle_and_archive_have_manifested_sorted_sha256_evidence() {
    let script = packaging_script();

    require_all(
        &script,
        &[
            "MANIFEST.json",
            "SHA256SUMS",
            "sha256sum",
            "git_sha",
            "cuda_manifest_sha256",
            "gpu_name",
            "driver_version",
            "tar -czf",
            ".tar.gz.sha256",
            "LC_ALL=C sort",
            "python3 -m json.tool",
            "packaging_host_ldd_only",
            ".release-assets",
            "PUBLISH_DIR",
            "mv -- \"${PUBLISH_DIR}\" \"${RELEASE_ASSET_DIR}\"",
        ],
    );

    let archive = script.find("tar -czf").expect("staged archive creation");
    let archive_check = script
        .rfind("sha256sum -c")
        .expect("staged archive checksum verification");
    let publish = script
        .find("mv -- \"${PUBLISH_DIR}\" \"${RELEASE_ASSET_DIR}\"")
        .expect("atomic release-asset directory publication");
    assert!(archive < archive_check && archive_check < publish);
}
