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

fn source(path: &str) -> String {
    fs::read_to_string(repository_root().join(path))
        .unwrap_or_else(|error| panic!("failed to read {path}: {error}"))
}

fn feature_section(manifest: &str) -> String {
    manifest
        .split_once("[features]")
        .expect("xgboost_lib-sys must declare features")
        .1
        .lines()
        .take_while(|line| !line.trim_start().starts_with('['))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn local_build_does_not_implicitly_enable_the_cpu_prebuilt_runtime() {
    for manifest_path in [
        "vendor/xgboost_lib-sys/Cargo.toml",
        "vendor/xgboost_lib-sys/Cargo.toml.orig",
    ] {
        let manifest = source(manifest_path);
        let features = feature_section(&manifest);
        assert!(
            features.lines().any(|line| line.trim() == "default = []"),
            "{manifest_path} must not make the CPU prebuilt runtime an implicit sys-crate default"
        );
        assert!(features.contains("use_prebuilt_xgb"));
        assert!(features.contains("local_build"));
    }
}

#[test]
fn prebuilt_and_local_runtime_authorities_are_mutually_exclusive() {
    let build = source("vendor/xgboost_lib-sys/build.rs");
    assert!(
        build.contains("#[cfg(all(feature = \"local_build\", feature = \"use_prebuilt_xgb\"))]")
    );
    assert!(
        build.contains(
            "#[cfg(not(any(feature = \"local_build\", feature = \"use_prebuilt_xgb\")))]"
        )
    );
    assert!(build.contains("compile_error!"));
    assert!(build.contains("exactly one XGBoost runtime authority"));
}

#[test]
fn selected_runtime_is_staged_fail_closed_into_the_cargo_profile_root() {
    let build = source("vendor/xgboost_lib-sys/build.rs");
    let stage = build
        .split_once("fn stage_selected_runtime(")
        .expect("missing selected-runtime staging helper")
        .1;

    for required in [
        "cargo_profile_output_dir",
        "source.is_file()",
        ".metadata()",
        "std::fs::copy",
        "copied_bytes",
        "source_len",
    ] {
        assert!(stage.contains(required), "staging must contain {required}");
    }
    assert!(!stage.contains("LD_LIBRARY_PATH"));
}

#[test]
fn local_build_stages_only_the_cmake_installed_runtime_for_linux_and_windows() {
    let build = source("vendor/xgboost_lib-sys/build.rs");
    let local = build
        .split_once("#[cfg(feature = \"local_build\")]")
        .expect("missing local-build branch")
        .1
        .split_once("// link to appropriate C++ lib")
        .expect("local-build branch must end before common link directives")
        .0;

    for required in [
        "dst.join(\"lib\").join(\"libxgboost.so\")",
        "dst.join(\"bin\").join(\"xgboost.dll\")",
        "stage_selected_runtime",
    ] {
        assert!(
            local.contains(required),
            "local runtime selection must contain {required}"
        );
    }
    assert!(!local.contains("../../../deps"));
    assert!(!local.contains("use_prebuilt_xgb"));
}

#[test]
fn prebuilt_runtime_is_also_staged_from_its_exact_selected_source() {
    let build = source("vendor/xgboost_lib-sys/build.rs");
    let prebuilt = build
        .split_once("cargo:rerun-if-env-changed=XGBOOST_LIB_DIR")
        .expect("missing prebuilt runtime-selection branch")
        .1
        .split_once("#[cfg(feature = \"local_build\")]")
        .expect("prebuilt branch must end before local build")
        .0;

    assert!(prebuilt.contains("selected_runtime"));
    assert!(prebuilt.contains("stage_selected_runtime"));
    assert!(!prebuilt.contains("LD_LIBRARY_PATH"));
}

#[test]
fn local_build_does_not_import_a_prebuilt_only_filesystem_alias() {
    let build = source("vendor/xgboost_lib-sys/build.rs");
    assert!(
        !build.lines().any(|line| line.trim() == "use std::fs;"),
        "the prebuilt-only filesystem alias is dead under local_build+cuda"
    );
    assert!(build.contains("std::fs::create_dir_all"));
}

#[test]
fn cli_uses_origin_runpath_on_linux_and_windows_executable_directory_semantics() {
    let build = source("crates/neoethos-cli/build.rs");
    assert!(build.contains("CARGO_CFG_TARGET_OS"));
    assert!(build.contains("target_os == \"linux\""));
    assert!(build.contains("cargo:rustc-link-arg-bin=neoethos-cli=-Wl,-rpath,$ORIGIN"));
    assert!(!build.contains("$ORIGIN/deps"));
    assert!(!build.contains("LD_LIBRARY_PATH"));
    assert!(!build.contains("target_os = \"windows\"") || !build.contains("rustc-link-arg"));
}

#[test]
fn release_workflows_consume_only_the_selected_profile_root_xgboost_runtime() {
    let desktop = source(".github/workflows/release-desktop.yml");
    for required in [
        "$xgboostTarget = \"target\\release\\xgboost.dll\"",
        "Test-Path -LiteralPath $xgboostTarget -PathType Leaf",
        "test -s target/release/libxgboost.so",
    ] {
        assert!(
            desktop.contains(required),
            "desktop release must consume profile-root runtime token {required:?}"
        );
    }
    for forbidden in [
        "target\\release\\deps\\xgboost.dll",
        "target/release/deps/libxgboost.so",
    ] {
        assert!(
            !desktop.contains(forbidden),
            "desktop release retains stale dependency-directory runtime path {forbidden:?}"
        );
    }

    let gpu = source(".github/workflows/release.yml");
    for required in [
        "$xgboostTarget = Join-Path $release \"xgboost.dll\"",
        "$catboostTarget = Join-Path $release \"catboostmodel.dll\"",
        "Test-Path -LiteralPath $xgboostTarget -PathType Leaf",
        "Test-Path -LiteralPath $catboostTarget -PathType Leaf",
        "Required XGBoost runtime is missing",
        "Required CatBoost runtime is missing",
    ] {
        assert!(
            gpu.contains(required),
            "GPU release must verify profile-root runtime token {required:?}"
        );
    }
    assert!(
        !gpu.contains("$release\\deps\\xgboost.dll"),
        "GPU release retains the superseded dependency-directory XGBoost runtime path"
    );
}

#[test]
fn lightgbm_is_a_static_payload_while_catboost_staging_is_fail_closed() {
    let lightgbm = source("vendor/lightgbm3-sys/build.rs");
    assert!(lightgbm.contains(".define(\"BUILD_STATIC_LIB\", \"ON\")"));
    assert!(lightgbm.contains("cargo:rustc-link-lib=static="));

    let catboost = source("vendor/catboost-rust-0.3.8-patched/build.rs");
    let stage = catboost
        .split_once("fn stage_selected_runtime(")
        .expect("CatBoost must have one fail-closed selected-runtime staging helper")
        .1;
    for required in [
        "source.is_file()",
        "source_len == 0",
        "fs::copy",
        "copied_bytes != source_len",
    ] {
        assert!(
            stage.contains(required),
            "CatBoost staging must contain {required}"
        );
    }
}

#[test]
fn linux_deb_and_rpm_metadata_install_native_runtimes_in_binary_specific_dirs() {
    let app = source("crates/neoethos-app/Cargo.toml");
    for required in [
        "[\"target/release/libxgboost.so\", \"usr/lib/neoethos-app/\", \"644\"]",
        "[\"target/release/libcatboostmodel.so\", \"usr/lib/neoethos-app/\", \"644\"]",
        "dest = \"/usr/lib/neoethos-app/libxgboost.so\"",
        "dest = \"/usr/lib/neoethos-app/libcatboostmodel.so\"",
    ] {
        assert!(
            app.contains(required),
            "app package metadata lacks {required}"
        );
    }

    let cli = source("crates/neoethos-cli/Cargo.toml");
    for required in [
        "[\"target/release/libxgboost.so\", \"usr/lib/neoethos-cli/\", \"644\"]",
        "[\"target/release/libcatboostmodel.so\", \"usr/lib/neoethos-cli/\", \"644\"]",
        "dest = \"/usr/lib/neoethos-cli/libxgboost.so\"",
        "dest = \"/usr/lib/neoethos-cli/libcatboostmodel.so\"",
    ] {
        assert!(
            cli.contains(required),
            "CLI package metadata lacks {required}"
        );
    }

    let app_build = source("crates/neoethos-app/build.rs");
    assert!(app_build.contains("$ORIGIN/lib"));
    assert!(app_build.contains("$ORIGIN/../lib/neoethos-app"));
    let cli_build = source("crates/neoethos-cli/build.rs");
    assert!(cli_build.contains("$ORIGIN/../lib/neoethos-cli"));
    assert!(!app_build.contains("LD_LIBRARY_PATH"));
    assert!(!cli_build.contains("LD_LIBRARY_PATH"));
}

#[test]
fn appimage_uses_payload_profile_runtimes_without_environment_fallbacks() {
    let build = source("packaging/appimage/build.sh");
    for required in [
        "PAYLOAD_RELEASE=\"${REPO_ROOT}/target/x86-64-v3-payload/release\"",
        "for library in libcatboostmodel.so libxgboost.so; do",
        "test -s \"${PAYLOAD_RELEASE}/${library}\"",
        "\"${APPDIR}/usr/bin/${library}\"",
    ] {
        assert!(
            build.contains(required),
            "AppImage staging lacks {required}"
        );
    }
    assert!(!build.contains("target/release/deps"));

    let app_run = source("packaging/appimage/neoethos-app.AppDir/AppRun");
    assert!(
        !app_run.contains("LD_LIBRARY_PATH"),
        "AppImage must rely on the payload's executable-relative RUNPATH"
    );
}

#[test]
fn installer_workflow_verifies_linux_packages_and_windows_payload_runtimes() {
    let workflow = source(".github/workflows/release-installers.yml");
    for required in [
        "Verify selected profile-root native runtimes",
        "test -s target/release/libxgboost.so",
        "test -s target/release/libcatboostmodel.so",
        "Verify Linux package runtime layouts",
        "dpkg-deb -c",
        "rpm2cpio",
        "$payloadRelease = \"target\\x86-64-v3-payload\\release\"",
        "$xgboostPayload = Join-Path $payloadRelease \"xgboost.dll\"",
        "$catboostPayload = Join-Path $payloadRelease \"catboostmodel.dll\"",
        "Copy-Item -LiteralPath $xgboostPayload -Destination out\\xgboost.dll",
        "Copy-Item -LiteralPath $catboostPayload -Destination out\\catboostmodel.dll",
    ] {
        assert!(
            workflow.contains(required),
            "installer workflow lacks {required}"
        );
    }
    assert!(!workflow.contains("release\\deps\\xgboost.dll"));
    assert!(!workflow.contains("release/deps/libxgboost.so"));
}

#[test]
fn gpu_portables_and_deb_use_the_runtime_selected_for_each_shipped_binary() {
    let workflow = source(".github/workflows/release.yml");
    for required in [
        "RELEASE_FEATURES: \"gpu-nvidia-full\"",
        "NEOETHOS_CUDA_BUILD_MODE: \"cross_release_explicit\"",
        "NEOETHOS_CUDA_ARCHS: \"80\"",
        "CUDA_ARCHS: \"80\"",
        "$payloadRelease = \"target\\x86-64-v3-payload\\release\"",
        "$xgboostPayload = Join-Path $payloadRelease \"xgboost.dll\"",
        "$catboostPayload = Join-Path $payloadRelease \"catboostmodel.dll\"",
        "PAYLOAD_RELEASE=\"target/x86-64-v3-payload/release\"",
        "for library in libcatboostmodel.so libxgboost.so; do",
        "test -s \"$PAYLOAD_RELEASE/$library\"",
        "install -m 0644 \"$PAYLOAD_RELEASE/$library\" \"$OUTDIR/$library\"",
        "cargo deb -p neoethos-app --no-build",
        "Verify GPU deb native runtime layout",
        "grep -Fq '$ORIGIN/lib'",
    ] {
        assert!(workflow.contains(required), "GPU release lacks {required}");
    }
    let portable = workflow
        .split_once("- name: Create portable zip")
        .expect("missing Windows portable archive step")
        .1
        .split_once("- name: Upload Windows artifacts")
        .expect("portable archive step must end before upload")
        .0;
    assert!(!workflow.contains("--formats deb || true"));
    assert!(!portable.contains("-ErrorAction SilentlyContinue"));
    assert!(!workflow.contains("export LD_LIBRARY_PATH="));
    assert!(!workflow.contains("release\\deps\\xgboost.dll"));
    assert!(!workflow.contains("release/deps/libxgboost.so"));

    let app = source("crates/neoethos-app/Cargo.toml");
    assert!(app.contains("gpu-nvidia-full = ["));
    assert!(app.contains("\"neoethos-models/burn-cuda-backend\""));
}
