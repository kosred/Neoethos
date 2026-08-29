#[allow(dead_code)]
#[path = "../build.rs"]
mod cuda_builder;

use cuda_builder::{
    ArchitectureRequest, ArtifactMetadata, CudaArchitecturePlan, NativePlatform,
    architecture_request_from_env, build_nvcc_archive_argv, build_nvcc_argv,
    inspect_artifact_images, render_manifest_v1, validate_external_environment,
};

const NVCC_ARCHS_WITH_BLACKWELL: &str = "compute_75\ncompute_86\ncompute_89\ncompute_90\ncompute_90a\ncompute_100f\ncompute_120\ncompute_120f\n";
const NVCC_CODE_WITH_BLACKWELL: &str =
    "sm_75\nsm_86\nsm_89\nsm_90\nsm_90a\nsm_100f\nsm_120\nsm_120f\n";

#[test]
fn rtx_5090_host_auto_emits_exact_sm120_sass_without_ptx_jit() {
    let plan = CudaArchitecturePlan::host_auto_from_tool_output(
        "12.0\n",
        NVCC_ARCHS_WITH_BLACKWELL,
        NVCC_CODE_WITH_BLACKWELL,
    )
    .expect("CUDA 12.8 supports the detected RTX 5090 capability");

    assert_eq!(plan.resolution_mode(), "host_auto");
    assert_eq!(plan.architectures(), &[120]);
    assert_eq!(
        plan.gencode_args(),
        vec!["--generate-code=arch=compute_120,code=sm_120"]
    );
    assert!(
        plan.gencode_args()
            .iter()
            .all(|arg| !arg.contains("code=compute_"))
    );
}

#[test]
fn superseded_compute70_ptx_fallback_is_absent_from_builder_source() {
    let source = include_str!("../build.rs");

    assert!(!source.contains("compute_70"));
    assert!(!source.contains("code=compute_"));
    assert!(source.contains("used, retain, visibility"));
    assert!(source.contains("/include:neoethos_cuda_build_manifest_v1"));
}

#[test]
fn host_auto_canonicalizes_every_visible_capability() {
    let first = CudaArchitecturePlan::host_auto_from_tool_output(
        "12.0\n8.6\n8.9\n8.6\n",
        NVCC_ARCHS_WITH_BLACKWELL,
        NVCC_CODE_WITH_BLACKWELL,
    )
    .unwrap();
    let reordered = CudaArchitecturePlan::host_auto_from_tool_output(
        "8.9\n12.0\n8.6\n",
        NVCC_ARCHS_WITH_BLACKWELL,
        NVCC_CODE_WITH_BLACKWELL,
    )
    .unwrap();

    assert_eq!(first.architectures(), &[86, 89, 120]);
    assert_eq!(first, reordered);
}

#[test]
fn current_nvcc_family_specific_targets_do_not_hide_numeric_sm120_support() {
    let plan = CudaArchitecturePlan::host_auto_from_tool_output(
        "12.0\n",
        NVCC_ARCHS_WITH_BLACKWELL,
        NVCC_CODE_WITH_BLACKWELL,
    )
    .expect("numeric compute_120/sm_120 remain valid beside 120f targets");

    assert_eq!(plan.architectures(), &[120]);
}

#[test]
fn host_auto_rejects_missing_malformed_and_unsupported_capabilities() {
    let no_gpu = CudaArchitecturePlan::host_auto_from_tool_output(
        "\n",
        NVCC_ARCHS_WITH_BLACKWELL,
        NVCC_CODE_WITH_BLACKWELL,
    )
    .unwrap_err();
    assert!(no_gpu.contains("no visible NVIDIA GPU"), "{no_gpu}");

    let malformed = CudaArchitecturePlan::host_auto_from_tool_output(
        "compute_120\n",
        NVCC_ARCHS_WITH_BLACKWELL,
        NVCC_CODE_WITH_BLACKWELL,
    )
    .unwrap_err();
    assert!(
        malformed.contains("malformed compute capability"),
        "{malformed}"
    );

    let unsupported =
        CudaArchitecturePlan::host_auto_from_tool_output("12.0\n", "compute_90\n", "sm_90\n")
            .unwrap_err();
    assert!(unsupported.contains("compute_120"), "{unsupported}");
    assert!(unsupported.contains("sm_120"), "{unsupported}");
}

#[test]
fn cross_release_is_typed_canonical_and_toolkit_validated() {
    let plan = CudaArchitecturePlan::cross_release_from_tool_output(
        "120;89;120",
        NVCC_ARCHS_WITH_BLACKWELL,
        NVCC_CODE_WITH_BLACKWELL,
    )
    .unwrap();

    assert_eq!(plan.resolution_mode(), "cross_release_explicit");
    assert_eq!(plan.architectures(), &[89, 120]);

    let malformed = CudaArchitecturePlan::cross_release_from_tool_output(
        "compute_120,code=sm_120",
        NVCC_ARCHS_WITH_BLACKWELL,
        NVCC_CODE_WITH_BLACKWELL,
    )
    .unwrap_err();
    assert!(
        malformed.contains("semicolon-separated numeric"),
        "{malformed}"
    );
}

#[test]
fn explicit_architectures_require_the_exact_cross_release_mode() {
    assert_eq!(
        architecture_request_from_env(None, None).unwrap(),
        ArchitectureRequest::HostAuto
    );
    assert_eq!(
        architecture_request_from_env(Some("host_auto"), None).unwrap(),
        ArchitectureRequest::HostAuto
    );
    assert_eq!(
        architecture_request_from_env(Some("cross_release_explicit"), Some("120;89")).unwrap(),
        ArchitectureRequest::CrossRelease("120;89".to_string())
    );

    for (mode, architectures) in [
        (None, Some("120")),
        (Some("host_auto"), Some("120")),
        (Some("cross_release_explicit"), None),
        (Some("cross_release_explicit"), Some("")),
        (Some("manual"), Some("120")),
    ] {
        assert!(
            architecture_request_from_env(mode, architectures).is_err(),
            "mode={mode:?} architectures={architectures:?} must fail closed"
        );
    }
}

#[test]
fn uncontrolled_architecture_and_precision_injection_is_rejected() {
    for (name, value) in [
        ("NEOETHOS_CUDA_ARCH", "compute_70,code=compute_70"),
        ("CUDA_ARCH", "sm_120"),
        ("CUDA_ARCHS", "120"),
        ("NVCC_ARGS", "--use_fast_math"),
        ("NVCC_PREPEND_FLAGS", "-fmad=true"),
        (
            "NVCC_APPEND_FLAGS",
            "--generate-code=arch=compute_70,code=compute_70",
        ),
        ("CUDAFLAGS", "--ftz=true"),
        ("CUDA_FAST_MATH", "1"),
    ] {
        let error = validate_external_environment(&[(name, value)]).unwrap_err();
        assert!(error.contains(name), "{name}: {error}");
    }

    validate_external_environment(&[("CUDA_FAST_MATH", "0")])
        .expect("an explicit fast-math-off assertion preserves the precision contract");
}

#[test]
fn nvcc_argv_is_exact_and_precision_preserving() {
    let plan = CudaArchitecturePlan::host_auto_from_tool_output(
        "12.0\n",
        NVCC_ARCHS_WITH_BLACKWELL,
        NVCC_CODE_WITH_BLACKWELL,
    )
    .unwrap();
    let args = build_nvcc_argv(
        &plan,
        "native/smoke.cu",
        "out/smoke.o",
        false,
        NativePlatform::Unix,
    );

    assert_eq!(
        args,
        vec![
            "-c",
            "native/smoke.cu",
            "-o",
            "out/smoke.o",
            "-std=c++17",
            "--generate-code=arch=compute_120,code=sm_120",
            "--fmad=false",
            "--ftz=false",
            "--prec-div=true",
            "--prec-sqrt=true",
            "-Xcompiler=-fPIC",
            "-I",
            "native",
            "-O3",
        ]
    );
}

#[test]
fn windows_nvcc_argv_and_archive_are_native_not_unix_hard_coded() {
    let plan = CudaArchitecturePlan::host_auto_from_tool_output(
        "12.0\n",
        NVCC_ARCHS_WITH_BLACKWELL,
        NVCC_CODE_WITH_BLACKWELL,
    )
    .unwrap();
    let args = build_nvcc_argv(
        &plan,
        "native/smoke.cu",
        r"out\smoke.obj",
        false,
        NativePlatform::Windows,
    );

    assert!(!args.iter().any(|arg| arg.contains("fPIC")));
    assert_eq!(NativePlatform::Windows.object_extension(), "obj");
    assert_eq!(
        NativePlatform::Windows.device_archive_name(),
        "neoethos_gpu_cuda_native.lib"
    );
    assert_eq!(NativePlatform::Unix.object_extension(), "o");
    assert_eq!(
        NativePlatform::Unix.device_archive_name(),
        "libneoethos_gpu_cuda_native.a"
    );
    assert_eq!(
        build_nvcc_archive_argv(
            &plan,
            r"out\neoethos_gpu_cuda_native.lib",
            &[r"out\smoke.obj".into(), r"out\population.obj".into()],
        ),
        vec![
            "--lib",
            "-o",
            r"out\neoethos_gpu_cuda_native.lib",
            "--generate-code=arch=compute_120,code=sm_120",
            r"out\smoke.obj",
            r"out\population.obj",
        ]
    );
}

#[test]
fn nvcc_archive_argv_reuses_exact_resolved_sass_only_gencode_plan() {
    let plan = CudaArchitecturePlan::host_auto_from_tool_output(
        "12.0\n8.9\n",
        NVCC_ARCHS_WITH_BLACKWELL,
        NVCC_CODE_WITH_BLACKWELL,
    )
    .unwrap();
    let args = build_nvcc_archive_argv(
        &plan,
        "out/libneoethos_gpu_cuda_native.a",
        &["out/smoke.o".into(), "out/prototype_b.o".into()],
    );

    assert_eq!(
        args,
        vec![
            "--lib",
            "-o",
            "out/libneoethos_gpu_cuda_native.a",
            "--generate-code=arch=compute_89,code=sm_89",
            "--generate-code=arch=compute_120,code=sm_120",
            "out/smoke.o",
            "out/prototype_b.o",
        ]
    );
    assert_eq!(
        args.iter()
            .filter(|argument| argument.starts_with("--generate-code="))
            .count(),
        plan.architectures().len()
    );
    assert!(
        !args
            .iter()
            .any(|argument| argument.contains(",code=compute_"))
    );
    assert!(!args.iter().any(|argument| {
        argument.contains("deprecated-gpu-targets") || argument.contains("suppress")
    }));
}

#[test]
fn artifact_inspection_proves_exact_sass_set_and_rejects_any_ptx() {
    let plan = CudaArchitecturePlan::host_auto_from_tool_output(
        "12.0\n8.6\n",
        NVCC_ARCHS_WITH_BLACKWELL,
        NVCC_CODE_WITH_BLACKWELL,
    )
    .unwrap();
    let images = inspect_artifact_images(
        &plan,
        "ELF file 1: smoke.sm_120.cubin\nELF file 2: smoke.sm_86.cubin\n\
         ELF file 3: population.sm_120.cubin\n",
        "",
    )
    .unwrap();

    assert_eq!(images.sass_targets(), &["sm_86", "sm_120"]);
    assert!(images.ptx_targets().is_empty());

    inspect_artifact_images(
        &plan,
        "ELF file 1: smoke.sm_120.cubin\nELF file 2: smoke.sm_86.cubin\n",
        "cuobjdump info : No PTX file found to extract from 'native.a'. \
         You may try with -all option.\n",
    )
    .expect("cuobjdump's canonical no-PTX diagnostic is positive absence evidence");

    let member_listing = concat!(
        "member /tmp/libneoethos_gpu_cuda_native.a:smoke.o:\n\n",
        "member /tmp/libneoethos_gpu_cuda_native.a:prototype_b.o:\n\n",
        "member /tmp/libneoethos_gpu_cuda_native.a:prototype_b_population.o:\n\n",
        "cuobjdump info    : No PTX file found to extract from ",
        "'/tmp/libneoethos_gpu_cuda_native.a'. You may try with -all option.\n",
    );
    inspect_artifact_images(
        &plan,
        "ELF file 1: smoke.sm_120.cubin\nELF file 2: smoke.sm_86.cubin\n",
        member_listing,
    )
    .expect("SASS-only archives may list well-formed members before the no-PTX diagnostic");

    let diagnostic_first_listing = concat!(
        "cuobjdump info    : No PTX file found to extract from ",
        "'/tmp/libneoethos_gpu_cuda_native.a'. You may try with -all option.\n\n",
        "member /tmp/libneoethos_gpu_cuda_native.a:smoke.o:\n",
        "member /tmp/libneoethos_gpu_cuda_native.a:prototype_b.o:\n",
    );
    inspect_artifact_images(
        &plan,
        "ELF file 1: smoke.sm_120.cubin\nELF file 2: smoke.sm_86.cubin\n",
        diagnostic_first_listing,
    )
    .expect("stderr/stdout merging must not make the validated line set order-sensitive");

    inspect_artifact_images(
        &plan,
        "ELF file 1: smoke.sm_120.cubin\nELF file 2: smoke.sm_86.cubin\n",
        concat!(
            "member C:\\cuda out\\neoethos_gpu_cuda_native.lib:smoke.obj:\n",
            "cuobjdump info    : No PTX file found to extract from ",
            "'C:\\cuda out\\neoethos_gpu_cuda_native.lib'. ",
            "You may try with -all option.\n",
        ),
    )
    .expect("member parsing must preserve a Windows drive colon and exact archive path");

    for malformed_listing in [
        "member /tmp/libnative.a:smoke.o:\n",
        "member /tmp/libnative.a:smoke.o:\nunknown cuobjdump output\ncuobjdump info : No PTX file found to extract from '/tmp/libnative.a'.\n",
        "member /tmp/libnative.a:\ncuobjdump info : No PTX file found to extract from '/tmp/libnative.a'.\n",
        "member :smoke.o:\ncuobjdump info : No PTX file found to extract from '/tmp/libnative.a'.\n",
        "member /tmp/other.a:smoke.o:\ncuobjdump info : No PTX file found to extract from '/tmp/libnative.a'.\n",
        "member /tmp/libnative.a:smoke.o:\nmember /tmp/libnative.a:smoke.o:\ncuobjdump info : No PTX file found to extract from '/tmp/libnative.a'.\n",
        "member /tmp/libnative.a:../smoke.o:\ncuobjdump info : No PTX file found to extract from '/tmp/libnative.a'.\n",
        "member /tmp/libnative.a:subdir\\smoke.o:\ncuobjdump info : No PTX file found to extract from '/tmp/libnative.a'.\n",
        "member /tmp/libnative.a:smoke o:\ncuobjdump info : No PTX file found to extract from '/tmp/libnative.a'.\n",
        "cuobjdump info : No PTX file found to extract from '/tmp/libnative.a'.\ncuobjdump info : No PTX file found to extract from '/tmp/libnative.a'.\n",
    ] {
        let error = inspect_artifact_images(
            &plan,
            "ELF file 1: smoke.sm_120.cubin\nELF file 2: smoke.sm_86.cubin\n",
            malformed_listing,
        )
        .unwrap_err();
        assert!(error.contains("unrecognized"), "{error}");
    }

    let missing =
        inspect_artifact_images(&plan, "ELF file 1: smoke.sm_120.cubin\n", "").unwrap_err();
    assert!(missing.contains("requested SASS"), "{missing}");

    let ptx = inspect_artifact_images(
        &plan,
        "ELF file 1: smoke.sm_120.cubin\nELF file 2: smoke.sm_86.cubin\n",
        "PTX file 1: smoke.sm_120.ptx\n",
    )
    .unwrap_err();
    assert!(ptx.contains("PTX"), "{ptx}");
}

#[test]
fn manifest_is_deterministic_and_records_no_ptx_targets() {
    let plan = CudaArchitecturePlan::host_auto_from_tool_output(
        "12.0\n8.6\n",
        NVCC_ARCHS_WITH_BLACKWELL,
        NVCC_CODE_WITH_BLACKWELL,
    )
    .unwrap();
    let artifact = ArtifactMetadata {
        logical_name: "libneoethos_gpu_cuda_native.a".to_string(),
        sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
        byte_len: 42,
    };
    let images = inspect_artifact_images(
        &plan,
        "ELF file 1: smoke.sm_120.cubin\nELF file 2: smoke.sm_86.cubin\n",
        "",
    )
    .unwrap();

    let manifest = render_manifest_v1(
        &plan,
        "Cuda compilation tools, release 12.8, V12.8.93",
        "cuobjdump release 12.8, V12.8.90",
        false,
        &artifact,
        &images,
    );

    assert_eq!(
        manifest,
        concat!(
            "{\"schema\":\"neoethos.cuda-native-build.v1\",",
            "\"resolution_mode\":\"host_auto\",",
            "\"architectures\":[86,120],",
            "\"gencode\":[\"--generate-code=arch=compute_86,code=sm_86\",",
            "\"--generate-code=arch=compute_120,code=sm_120\"],",
            "\"sass_targets\":[\"sm_86\",\"sm_120\"],",
            "\"ptx_targets\":[],",
            "\"precision_flags\":[\"--fmad=false\",\"--ftz=false\",",
            "\"--prec-div=true\",\"--prec-sqrt=true\"],",
            "\"optimization\":\"-O3\",",
            "\"nvcc_version\":\"Cuda compilation tools, release 12.8, V12.8.93\",",
            "\"cuobjdump_version\":\"cuobjdump release 12.8, V12.8.90\",",
            "\"artifact\":{\"logical_name\":\"libneoethos_gpu_cuda_native.a\",",
            "\"sha256\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",",
            "\"byte_len\":42}}"
        )
    );
    assert!(!manifest.contains("compute_86\""));
    assert!(!manifest.contains("compute_120\""));
}
