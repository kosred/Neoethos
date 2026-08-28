use crate::native_sass::{
    ArchRequestSource, CuobjdumpReport, NativeArchInputs, NativeArtifact, NativeSassError,
    native_cubin_filename, plan_native_architectures, select_exact_native_cubin,
    validate_native_manifest, verify_native_cubin,
};
use std::path::Path;

const ELF_CUBIN_FIXTURE: &[u8] = b"\x7fELF\x02\x01\x01\x00native-cubin-fixture";
const INSPECTED_CUBIN_PATH: &str = "/workspace/native/probe_sm120.cubin";

#[test]
fn planner_emits_only_exact_requested_or_visible_architectures() {
    let supported = [80, 86, 89, 90, 100, 120];
    let detected = plan_native_architectures(NativeArchInputs {
        explicit_archs: None,
        detected_archs: &[120, 89, 120],
        nvcc_supported_archs: &supported,
    })
    .expect("every visible architecture is supported by this nvcc fixture");
    assert_eq!(detected.source, ArchRequestSource::DetectedVisibleDevices);
    assert_eq!(detected.architectures, vec![89, 120]);
    assert_eq!(
        detected
            .architectures
            .iter()
            .map(|&arch| native_cubin_filename("adx_kernel", arch))
            .collect::<Vec<_>>(),
        ["adx_kernel_sm89.cubin", "adx_kernel_sm120.cubin"]
    );

    let explicit = plan_native_architectures(NativeArchInputs {
        explicit_archs: Some("sm_120, 8.9, sm_120"),
        detected_archs: &[80],
        nvcc_supported_archs: &supported,
    })
    .expect("the explicit list takes precedence and is canonicalized");
    assert_eq!(explicit.source, ArchRequestSource::ExplicitList);
    assert_eq!(explicit.architectures, vec![89, 120]);

    assert!(matches!(
        plan_native_architectures(NativeArchInputs {
            explicit_archs: None,
            detected_archs: &[],
            nvcc_supported_archs: &supported,
        }),
        Err(NativeSassError::NoTargetArchitectures)
    ));
    assert!(matches!(
        plan_native_architectures(NativeArchInputs {
            explicit_archs: None,
            detected_archs: &[121],
            nvcc_supported_archs: &supported,
        }),
        Err(NativeSassError::UnsupportedArchitectures { missing, .. }) if missing == vec![121]
    ));
}

#[test]
fn manifest_must_be_the_exact_stem_by_architecture_cartesian_product() {
    let records = [
        NativeArtifact::new("adx_kernel", 89, ELF_CUBIN_FIXTURE),
        NativeArtifact::new("adx_kernel", 120, ELF_CUBIN_FIXTURE),
        NativeArtifact::new("vector_ta_native_probe", 89, ELF_CUBIN_FIXTURE),
        NativeArtifact::new("vector_ta_native_probe", 120, ELF_CUBIN_FIXTURE),
    ];
    validate_native_manifest(
        &records,
        &["adx_kernel", "vector_ta_native_probe"],
        &[89, 120],
    )
    .expect("the manifest contains exactly one artifact per stem and architecture");

    assert!(matches!(
        validate_native_manifest(
            &records[..3],
            &["adx_kernel", "vector_ta_native_probe"],
            &[89, 120],
        ),
        Err(NativeSassError::MissingArtifact { stem, arch })
            if stem == "vector_ta_native_probe" && arch == 120
    ));

    let duplicate = [records[0], records[0]];
    assert!(matches!(
        validate_native_manifest(&duplicate, &["adx_kernel"], &[89]),
        Err(NativeSassError::DuplicateArtifact { stem, arch })
            if stem == "adx_kernel" && arch == 89
    ));
}

#[test]
fn verifier_requires_elf_exact_sass_and_zero_embedded_ptx() {
    let clean = CuobjdumpReport {
        list_ptx_succeeded: true,
        list_ptx_stdout: "",
        list_ptx_stderr: "",
        dump_sass_succeeded: true,
        dump_sass_stdout: "Fatbin elf code:\n================\narch = sm_120\ncode for sm_120\n",
        dump_sass_stderr: "",
    };
    let verified = verify_native_cubin(
        120,
        Path::new(INSPECTED_CUBIN_PATH),
        ELF_CUBIN_FIXTURE,
        clean,
    )
    .expect("an ELF cubin with only exact sm_120 SASS is valid");
    assert_eq!(verified.arch, 120);
    assert_eq!(verified.byte_len, ELF_CUBIN_FIXTURE.len());

    let embedded_ptx = CuobjdumpReport {
        list_ptx_stdout: "PTX file    1: embedded.compute_120.ptx",
        ..clean
    };
    assert!(matches!(
        verify_native_cubin(
            120,
            Path::new(INSPECTED_CUBIN_PATH),
            ELF_CUBIN_FIXTURE,
            embedded_ptx,
        ),
        Err(NativeSassError::EmbeddedPtx { arch: 120, .. })
    ));

    let wrong_sass = CuobjdumpReport {
        dump_sass_stdout: "code for sm_89\n",
        ..clean
    };
    assert!(matches!(
        verify_native_cubin(
            120,
            Path::new(INSPECTED_CUBIN_PATH),
            ELF_CUBIN_FIXTURE,
            wrong_sass,
        ),
        Err(NativeSassError::WrongSassArchitecture {
            expected: 120,
            found,
        }) if found == vec![89]
    ));

    assert!(matches!(
        verify_native_cubin(120, Path::new(INSPECTED_CUBIN_PATH), b"not-elf", clean),
        Err(NativeSassError::NotElfCubin { arch: 120 })
    ));
}

#[test]
fn verifier_accepts_only_exact_cuda_12_8_no_ptx_diagnostic_for_inspected_path() {
    let clean = CuobjdumpReport {
        list_ptx_succeeded: true,
        list_ptx_stdout: "",
        list_ptx_stderr: "",
        dump_sass_succeeded: true,
        dump_sass_stdout: "Fatbin elf code:\narch = sm_120\ncode for sm_120\n",
        dump_sass_stderr: "",
    };
    let canonical = format!(
        "cuobjdump info    : No PTX file found to extract from '{}'. You may try with -all option.",
        INSPECTED_CUBIN_PATH
    );

    for diagnostic in [
        canonical.clone(),
        format!("{canonical}\n"),
        format!("{canonical}\r\n"),
    ] {
        let report = CuobjdumpReport {
            list_ptx_stderr: &diagnostic,
            ..clean
        };
        verify_native_cubin(
            120,
            Path::new(INSPECTED_CUBIN_PATH),
            ELF_CUBIN_FIXTURE,
            report,
        )
        .expect("the one canonical CUDA 12.8 no-PTX diagnostic is absence evidence");
    }

    let wrong_path = canonical.replace(INSPECTED_CUBIN_PATH, "/workspace/native/other.cubin");
    let duplicate = format!("{canonical}\n{canonical}\n");
    let extra_member = format!("{canonical}\nPTX file    1: embedded.compute_120.ptx\n");
    let unknown = "cuobjdump info    : no extractable device payload";
    let altered_internal_whitespace = canonical.replace("info    :", "info :");
    let leading_whitespace = format!(" {canonical}");
    let extra_line_ending = format!("{canonical}\n\n");
    let cases = [
        ("wrong inspected path", "", wrong_path.as_str()),
        ("duplicate diagnostic", "", duplicate.as_str()),
        ("extra PTX member", "", extra_member.as_str()),
        ("unknown output", "", unknown),
        (
            "altered internal whitespace",
            "",
            altered_internal_whitespace.as_str(),
        ),
        ("leading whitespace", "", leading_whitespace.as_str()),
        ("extra line ending", "", extra_line_ending.as_str()),
        ("whitespace-only stderr", "", "\n"),
        ("whitespace-only stdout", "\n", ""),
        ("diagnostic on stdout", canonical.as_str(), ""),
    ];
    for (label, stdout, stderr) in cases {
        let report = CuobjdumpReport {
            list_ptx_stdout: stdout,
            list_ptx_stderr: stderr,
            ..clean
        };
        assert!(
            matches!(
                verify_native_cubin(
                    120,
                    Path::new(INSPECTED_CUBIN_PATH),
                    ELF_CUBIN_FIXTURE,
                    report,
                ),
                Err(NativeSassError::EmbeddedPtx { arch: 120, .. })
            ),
            "{label} must fail closed"
        );
    }
}

#[test]
fn runtime_selector_requires_an_exact_device_architecture_match() {
    let sm89 = b"sm89";
    let sm120 = b"sm120";
    let records = [
        NativeArtifact::new("probe", 89, sm89),
        NativeArtifact::new("probe", 120, sm120),
    ];

    assert_eq!(
        select_exact_native_cubin("probe", 12, 0, &records)
            .expect("sm_120 selects only the sm_120 cubin"),
        sm120
    );
    assert!(matches!(
        select_exact_native_cubin("probe", 12, 1, &records),
        Err(NativeSassError::MissingExactArchitecture {
            stem,
            requested: 121,
            available,
        }) if stem == "probe" && available == vec![89, 120]
    ));
    assert!(matches!(
        select_exact_native_cubin("missing", 12, 0, &records),
        Err(NativeSassError::MissingKernel { stem }) if stem == "missing"
    ));
    assert!(matches!(
        select_exact_native_cubin("probe", -1, 0, &records),
        Err(NativeSassError::InvalidDeviceCapability {
            major: -1,
            minor: 0,
        })
    ));
}

#[cfg(feature = "cuda-build-native")]
#[test]
fn native_sass_probe_loads_and_launches_on_real_device() {
    assert!(
        crate::cuda::cuda_available(),
        "the required native-SASS probe did not load and launch on a real CUDA device"
    );
}
