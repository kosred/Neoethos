#[path = "../../../vendor/cuda_build_arch.rs"]
mod cuda_build_arch;

const XGBOOST_BUILD: &str = include_str!("../../../vendor/xgboost_lib-sys/build.rs");
const LIGHTGBM_BUILD: &str = include_str!("../../../vendor/lightgbm3-sys/build.rs");
const CUDA_BUILD_ARCH: &str = include_str!("../../../vendor/cuda_build_arch.rs");
const LIGHTGBM_CMAKE: &str = include_str!("../../../vendor/lightgbm3-sys/lightgbm/CMakeLists.txt");
const XGBOOST_ALGORITHM: &str =
    include_str!("../../../vendor/xgboost_lib-sys/xgboost/src/common/algorithm.cuh");
const XGBOOST_HISTOGRAM: &str =
    include_str!("../../../vendor/xgboost_lib-sys/xgboost/src/common/hist_util.cuh");
const XGBOOST_SPLITS: &str =
    include_str!("../../../vendor/xgboost_lib-sys/xgboost/src/tree/gpu_hist/evaluate_splits.cu");
const XGBOOST_PARTITIONER: &str =
    include_str!("../../../vendor/xgboost_lib-sys/xgboost/src/tree/gpu_hist/row_partitioner.cuh");
const XGBOOST_RESOURCE: &str =
    include_str!("../../../vendor/xgboost_lib-sys/xgboost/src/common/resource.cu");
const XGBOOST_DRIVER_API: &str =
    include_str!("../../../vendor/xgboost_lib-sys/xgboost/src/common/cuda_dr_utils.cc");

#[test]
fn exact_cuda_architecture_parser_is_plural_sorted_native_only_and_fail_closed() {
    let _production_resolver = cuda_build_arch::resolve_exact_cuda_architectures;

    let parsed = cuda_build_arch::parse_exact_cuda_architectures("sm_89, 8.6;86")
        .expect_err("normalized duplicate architectures must be rejected");
    assert!(parsed.contains("duplicate"));

    let parsed = cuda_build_arch::parse_exact_cuda_architectures("sm_89, 8.6")
        .expect("reviewed exact architecture list");
    assert_eq!(parsed.numeric, "86;89");
    assert_eq!(parsed.native_only, "86-real;89-real");

    for invalid in ["", " ", "sm_x", "9", "8.10", "86-virtual"] {
        assert!(
            cuda_build_arch::parse_exact_cuda_architectures(invalid).is_err(),
            "invalid architecture input must fail closed: {invalid:?}"
        );
    }
}

#[test]
fn host_auto_resolves_the_visible_rtx_3090_to_exact_sm86_native_sass() {
    let resolved = cuda_build_arch::resolve_exact_cuda_architectures_from_inputs(
        cuda_build_arch::ExactCudaArchitectureInputs {
            build_mode: None,
            explicit_architectures: None,
            legacy_cuda_archs: None,
            legacy_cudaarchs: None,
            legacy_cmake_cuda_architectures: None,
            visible_compute_capabilities: Some("8.6\n8.6\n"),
            nvcc_real_architectures: "sm_75\nsm_80\nsm_86\nsm_89\n",
        },
    )
    .expect("the visible RTX 3090 must resolve through host_auto");

    assert_eq!(resolved.numeric, "86");
    assert_eq!(resolved.native_only, "86-real");
    assert_eq!(resolved.resolution_mode, "host_auto");
}

#[test]
fn typed_cross_release_is_the_only_explicit_architecture_authority() {
    let resolved = cuda_build_arch::resolve_exact_cuda_architectures_from_inputs(
        cuda_build_arch::ExactCudaArchitectureInputs {
            build_mode: Some("cross_release_explicit"),
            explicit_architectures: Some("sm_89;8.6"),
            legacy_cuda_archs: None,
            legacy_cudaarchs: None,
            legacy_cmake_cuda_architectures: None,
            visible_compute_capabilities: None,
            nvcc_real_architectures: "sm_75\nsm_86\nsm_89\n",
        },
    )
    .expect("typed cross-release architecture set");

    assert_eq!(resolved.numeric, "86;89");
    assert_eq!(resolved.native_only, "86-real;89-real");
    assert_eq!(resolved.resolution_mode, "cross_release_explicit");
}

#[test]
fn cross_release_accepts_cuda_archs_only_as_an_exact_equality_assertion() {
    let base = || cuda_build_arch::ExactCudaArchitectureInputs {
        build_mode: Some("cross_release_explicit"),
        explicit_architectures: Some("sm_89;8.6"),
        legacy_cuda_archs: Some("86, sm_89"),
        legacy_cudaarchs: None,
        legacy_cmake_cuda_architectures: None,
        visible_compute_capabilities: None,
        nvcc_real_architectures: "sm_75\nsm_86\nsm_89\n",
    };

    let resolved = cuda_build_arch::resolve_exact_cuda_architectures_from_inputs(base())
        .expect("CUDA_ARCHS may mirror, but never replace, the typed exact authority");
    assert_eq!(resolved.numeric, "86;89");
    assert_eq!(resolved.resolution_mode, "cross_release_explicit");

    let mismatch = cuda_build_arch::resolve_exact_cuda_architectures_from_inputs(
        cuda_build_arch::ExactCudaArchitectureInputs {
            legacy_cuda_archs: Some("86"),
            ..base()
        },
    )
    .expect_err("a partial legacy mirror must fail before launching build tools");
    assert!(mismatch.contains("CUDA_ARCHS"));
    assert!(mismatch.contains("does not exactly match"));

    let missing_typed = cuda_build_arch::resolve_exact_cuda_architectures_from_inputs(
        cuda_build_arch::ExactCudaArchitectureInputs {
            explicit_architectures: None,
            legacy_cuda_archs: Some("86;89"),
            ..base()
        },
    )
    .expect_err("CUDA_ARCHS cannot become the authority when the typed set is absent");
    assert!(missing_typed.contains("NEOETHOS_CUDA_ARCHS"));
}

#[test]
fn architecture_resolution_rejects_legacy_environment_and_unproven_targets() {
    let base = || cuda_build_arch::ExactCudaArchitectureInputs {
        build_mode: None,
        explicit_architectures: None,
        legacy_cuda_archs: None,
        legacy_cudaarchs: None,
        legacy_cmake_cuda_architectures: None,
        visible_compute_capabilities: Some("8.6\n"),
        nvcc_real_architectures: "sm_75\nsm_86\n",
    };

    for (name, inputs) in [
        (
            "CUDA_ARCHS",
            cuda_build_arch::ExactCudaArchitectureInputs {
                legacy_cuda_archs: Some("86"),
                ..base()
            },
        ),
        (
            "CUDAARCHS",
            cuda_build_arch::ExactCudaArchitectureInputs {
                legacy_cudaarchs: Some("86"),
                ..base()
            },
        ),
        (
            "CMAKE_CUDA_ARCHITECTURES",
            cuda_build_arch::ExactCudaArchitectureInputs {
                legacy_cmake_cuda_architectures: Some("86"),
                ..base()
            },
        ),
    ] {
        let error = cuda_build_arch::resolve_exact_cuda_architectures_from_inputs(inputs)
            .expect_err("legacy ambient architecture authority must be rejected");
        assert!(error.contains(name), "wrong rejection for {name}: {error}");
    }

    let unsupported = cuda_build_arch::resolve_exact_cuda_architectures_from_inputs(
        cuda_build_arch::ExactCudaArchitectureInputs {
            nvcc_real_architectures: "sm_75\nsm_80\n",
            ..base()
        },
    )
    .expect_err("visible sm_86 must be supported by this exact nvcc");
    assert!(unsupported.contains("sm_86"));

    for inputs in [
        cuda_build_arch::ExactCudaArchitectureInputs {
            visible_compute_capabilities: Some(""),
            ..base()
        },
        cuda_build_arch::ExactCudaArchitectureInputs {
            visible_compute_capabilities: Some("not-a-capability"),
            ..base()
        },
        cuda_build_arch::ExactCudaArchitectureInputs {
            build_mode: Some("host_auto"),
            explicit_architectures: Some("86"),
            ..base()
        },
        cuda_build_arch::ExactCudaArchitectureInputs {
            build_mode: Some("cross_release_explicit"),
            explicit_architectures: None,
            visible_compute_capabilities: None,
            ..base()
        },
    ] {
        assert!(
            cuda_build_arch::resolve_exact_cuda_architectures_from_inputs(inputs).is_err(),
            "ambiguous or incomplete architecture evidence must fail closed"
        );
    }
}

#[test]
fn cuda_tree_builders_consume_one_host_auto_or_typed_cross_release_contract() {
    for (name, source) in [("XGBoost", XGBOOST_BUILD), ("LightGBM", LIGHTGBM_BUILD)] {
        assert!(
            source.contains("resolve_exact_cuda_architectures"),
            "{name} must consume the shared exact CUDA architecture resolver"
        );
        assert!(
            !source.contains("exact_cuda_architectures_from_env"),
            "{name} must not require the rejected legacy CUDA_ARCHS environment"
        );
        assert!(
            source.contains("CMAKE_CUDA_ARCHITECTURES"),
            "{name} must pass the exact native architecture set to CMake"
        );
    }

    for required in [
        "NEOETHOS_CUDA_BUILD_MODE",
        "NEOETHOS_CUDA_ARCHS",
        "NVIDIA_SMI",
        "CUDACXX",
        "--query-gpu=compute_cap",
        "--list-gpu-code",
    ] {
        assert!(
            CUDA_BUILD_ARCH.contains(required),
            "shared CUDA architecture resolver is missing {required}"
        );
    }
    for legacy in ["CUDA_ARCHS", "CUDAARCHS", "CMAKE_CUDA_ARCHITECTURES"] {
        assert!(
            CUDA_BUILD_ARCH.contains(legacy),
            "shared resolver must name and reject legacy authority {legacy}"
        );
    }

    assert!(
        !XGBOOST_BUILD.contains("let mut dst = dst"),
        "the CUDA XGBoost builder must remain warning-clean"
    );
    assert!(
        !XGBOOST_BUILD.contains("BUILD_WITH_CUDA"),
        "the XGBoost builder must not pass removed CMake variables that only emit warnings"
    );
    assert!(
        LIGHTGBM_BUILD.contains("NEOETHOS_EXACT_CUDA_ARCHITECTURES"),
        "LightGBM needs an explicit caller-provided override because upstream rewrites its target list"
    );
    assert!(
        LIGHTGBM_BUILD.contains("fs::remove_dir_all(&lgbm_root)"),
        "a build-script rerun must remove the stale copied LightGBM source tree"
    );
    assert!(
        !LIGHTGBM_BUILD.contains("if !lgbm_root.exists()"),
        "LightGBM source changes must not be ignored merely because OUT_DIR already exists"
    );
    assert!(
        LIGHTGBM_CMAKE.contains("if(DEFINED NEOETHOS_EXACT_CUDA_ARCHITECTURES)"),
        "LightGBM CMake must honor the exact builder-provided architecture set"
    );
    assert!(
        LIGHTGBM_CMAKE.contains("set(CUDA_ARCHS \"${NEOETHOS_EXACT_CUDA_ARCHITECTURES}\")"),
        "LightGBM targets must receive exactly the sealed native-only set"
    );
}

#[test]
fn vendored_xgboost_carries_the_official_cccl_3_backport() {
    for marker in [
        "#if CUB_VERSION >= 300000",
        "cub::SortOrder::Ascending",
        "cub::SortOrder::Descending",
        "kCubSortOrderAscending",
        "cuda::std::plus{}",
    ] {
        assert!(
            XGBOOST_ALGORITHM.contains(marker),
            "XGBoost algorithm.cuh is missing official CCCL compatibility marker {marker:?}"
        );
    }
    assert!(
        XGBOOST_HISTOGRAM.matches("__syncthreads();").count() >= 2,
        "XGBoost histogram synchronization must use the CUDA primitive with CCCL 3"
    );
    assert!(
        XGBOOST_SPLITS.contains("cuda::std::plus{}") && XGBOOST_SPLITS.contains("__syncthreads();"),
        "XGBoost split evaluation is missing the official CCCL 3 compatibility changes"
    );
    assert!(
        XGBOOST_PARTITIONER.contains("cub::NullType, std::uint64_t>::Dispatch")
            && XGBOOST_PARTITIONER.contains("static_cast<std::uint64_t>(total_rows)"),
        "XGBoost scan offsets must use the CCCL 3-compatible unsigned type"
    );
}

#[test]
fn vendored_xgboost_carries_the_official_cuda_13_backport() {
    for marker in [
        "#if (CUDA_VERSION / 1000) >= 13",
        "cudaMemLocation loc;",
        "loc.type = cudaMemLocationTypeDevice;",
        "cudaMemPrefetchAsync(handle_->base_ptr, handle_->base_size, loc, 0,",
    ] {
        assert!(
            XGBOOST_RESOURCE.contains(marker),
            "XGBoost resource.cu is missing official CUDA 13 marker {marker:?}"
        );
    }
    assert!(
        XGBOOST_DRIVER_API.contains("cudaGetDriverEntryPointByVersion")
            && XGBOOST_DRIVER_API.contains("12080, cudaEnablePerThreadDefaultStream"),
        "XGBoost must use the CUDA 13-compatible versioned driver entry-point query"
    );
}
