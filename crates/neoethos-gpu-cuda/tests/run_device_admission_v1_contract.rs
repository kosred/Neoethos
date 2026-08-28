use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-gpu-cuda"))
}

fn read(relative: &str) -> String {
    let path = manifest_dir().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read required source {}: {error}", path.display()))
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let (_, tail) = source
        .split_once(start)
        .unwrap_or_else(|| panic!("missing source boundary {start:?}"));
    tail.split_once(end)
        .unwrap_or_else(|| panic!("missing source boundary {end:?} after {start:?}"))
        .0
}

fn require_all(source: &str, required: &[&str]) {
    for token in required {
        assert!(source.contains(token), "missing required token {token:?}");
    }
}

fn assert_exactly_once(source: &str, token: &str) {
    assert_eq!(
        source.matches(token).count(),
        1,
        "expected exactly one occurrence of {token:?}"
    );
}

fn declaration_window<'a>(source: &'a str, declaration: &str) -> &'a str {
    let start = source
        .find(declaration)
        .unwrap_or_else(|| panic!("missing declaration {declaration:?}"));
    let before = source[..start]
        .rfind("\n\n")
        .map(|offset| offset + 2)
        .unwrap_or(0);
    let body_end = source[start..]
        .find("\n}")
        .map(|offset| start + offset + 2)
        .unwrap_or_else(|| panic!("unterminated declaration {declaration:?}"));
    &source[before..body_end]
}

#[test]
fn fallible_cuda_enumeration_contract_remains_a_required_prerequisite() {
    let contract = read("tests/device_enumeration_v1_contract.rs");
    require_all(
        &contract,
        &[
            "neoethos_gpu_cuda_probe_device_count_v1",
            "cudaGetDeviceCount(&count)",
            "NEO_CUDA_DEVICE_PROBE_ADAPTER_UNAVAILABLE",
            "pub fn probe_cuda_device_count_v1() -> Result<u32, CudaDeviceEnumerationErrorV1>",
            "strict_search_route_uses_only_the_fallible_probe_for_cpu_authority",
        ],
    );
    assert!(
        contract.contains("!probe.contains(\"count > 0\")"),
        "the prerequisite must preserve successful exact zero-device enumeration"
    );
}

#[test]
fn crate_exports_one_opaque_no_argument_run_device_acquisition() {
    let lib = read("src/lib.rs");
    require_all(
        &lib,
        &[
            "mod physical_gpu_inventory_v1;",
            "mod run_device_admission_v1;",
            "SealedDiscoveryRunDeviceAdmissionV1",
            "acquire_discovery_run_device_admission_v1",
        ],
    );
    let source = read("src/run_device_admission_v1.rs");
    require_all(
        &source,
        &[
            "SealedDiscoveryRunDeviceAdmissionV1",
            "DiscoveryRunDeviceAdmissionErrorV1",
        ],
    );
    let (_, signature_tail) = source
        .split_once("pub fn acquire_discovery_run_device_admission_v1(")
        .expect("missing public run-device acquisition");
    let (parameters, return_tail) = signature_tail
        .split_once(')')
        .expect("unterminated run-device acquisition parameter list");
    assert!(
        parameters.trim().is_empty(),
        "run-device acquisition accepted caller-controlled parameters"
    );
    assert!(
        return_tail
            .trim_start()
            .starts_with("-> Result<SealedDiscoveryRunDeviceAdmissionV1"),
        "run-device acquisition returned an unexpected authority type"
    );
}

#[test]
fn acquisition_performs_one_inventory_one_cuda_enumeration_one_context_and_one_stream() {
    let source = read("src/run_device_admission_v1.rs");
    let acquisition = section(
        &source,
        "pub fn acquire_discovery_run_device_admission_v1(",
        "\n}",
    );
    for token in [
        "probe_physical_gpu_inventory_v1()",
        "probe_cuda_device_count_v1()",
        "retain_primary_context_once_v1(",
        "create_run_stream_once_v1(",
    ] {
        assert_exactly_once(acquisition, token);
    }
    require_all(
        acquisition,
        &[
            "RunDeviceAcquisitionCountersV1::new()",
            "record_physical_inventory_probe_v1()",
            "record_cuda_enumeration_v1()",
            "record_primary_context_acquisition_v1()",
            "record_run_stream_creation_v1()",
            "seal_exact_once_v1()",
        ],
    );
    for forbidden in [
        "physical_inventory_probe_count: 1",
        "cuda_enumeration_count: 1",
        "primary_context_acquisition_count: 1",
        "run_stream_creation_count: 1",
    ] {
        assert!(
            !acquisition.contains(forbidden),
            "acquisition fabricated a probe counter via {forbidden:?}"
        );
    }
}

#[test]
fn complete_zero_physical_inventory_can_authorize_cpu_without_a_cuda_runtime() {
    let source = read("src/run_device_admission_v1.rs");
    let classifier = section(
        &source,
        "fn classify_discovery_run_device_admission_v1(",
        "\n}",
    );
    require_all(
        classifier,
        &[
            "CompleteNoPhysicalGpu",
            "NativeAdapterUnavailable",
            "RuntimeUnavailable",
            "CpuNoPhysicalGpu",
            "seal_no_physical_gpu_receipt_v1(",
        ],
    );
    assert!(
        !classifier.contains("card_present: bool"),
        "caller-supplied presence boolean authorized CPU"
    );
}

#[test]
fn baseline_zero_and_positive_cuda_enumeration_is_a_typed_contradiction() {
    let source = read("src/run_device_admission_v1.rs");
    let classifier = section(
        &source,
        "fn classify_discovery_run_device_admission_v1(",
        "\n}",
    );
    require_all(
        classifier,
        &[
            "CompleteNoPhysicalGpu",
            "ExactCudaDeviceCount",
            "BaselineCudaContradiction",
        ],
    );
    let zero_at = classifier
        .find("CompleteNoPhysicalGpu")
        .expect("zero physical GPU branch");
    let contradiction_at = classifier
        .find("BaselineCudaContradiction")
        .expect("baseline/CUDA contradiction branch");
    assert!(
        contradiction_at > zero_at,
        "positive CUDA contradiction was not classified from baseline-zero evidence"
    );
}

#[test]
fn any_visible_gpu_without_an_exact_native_backend_or_build_fails_loud() {
    let source = read("src/run_device_admission_v1.rs");
    let classifier = section(
        &source,
        "fn classify_discovery_run_device_admission_v1(",
        "\n}",
    );
    require_all(
        classifier,
        &[
            "CompletePhysicalGpuSet",
            "VisibleGpuWithoutStrictBackend",
            "VisibleGpuBuildIncompatible",
            "CudaEnumerationFailure",
            "NoCompatibleCudaOrdinal",
            "select_lowest_compatible_cuda_ordinal_v1",
        ],
    );
    for forbidden in [
        "CompletePhysicalGpuSet => CpuNoPhysicalGpu",
        "BuildIncompatible => CpuNoPhysicalGpu",
        "CudaEnumerationFailure => CpuNoPhysicalGpu",
    ] {
        assert!(
            !classifier.contains(forbidden),
            "visible GPU incorrectly authorized CPU via {forbidden:?}"
        );
    }
}

#[test]
fn native_admission_binds_exact_hardware_context_build_and_memory_snapshot() {
    let source = read("src/run_device_admission_v1.rs");
    let native = section(
        &source,
        "struct SealedNativeCudaRunDeviceAdmissionV1 {",
        "\n}",
    );
    require_all(
        native,
        &[
            "physical_inventory_identity_sha256:",
            "pci_identity:",
            "device_uuid:",
            "ordinal:",
            "primary_context:",
            "run_stream:",
            "cuda_build_identity:",
            "sass_target:",
            "free_memory_bytes_snapshot:",
        ],
    );
    let stream_at = native
        .find("run_stream:")
        .expect("sealed native admission must retain its stream");
    let context_at = native
        .find("primary_context:")
        .expect("sealed native admission must retain its context");
    assert!(
        stream_at < context_at,
        "CUDA stream must drop before the retained primary context"
    );
}

#[test]
fn run_device_admission_is_move_only_and_has_no_bare_authority_constructor() {
    let source = read("src/run_device_admission_v1.rs");
    for declaration in [
        "pub enum SealedDiscoveryRunDeviceAdmissionV1 {",
        "struct SealedNativeCudaRunDeviceAdmissionV1 {",
    ] {
        let window = declaration_window(&source, declaration);
        for forbidden in ["Clone", "Default", "Deserialize"] {
            assert!(
                !window.contains(forbidden),
                "run-device admission gained reconstructible trait {forbidden:?}"
            );
        }
    }
    for forbidden in [
        "pub fn from_ordinal",
        "pub fn from_reserve",
        "pub fn from_sha256",
        "pub fn native_cuda_unchecked",
        "impl Clone for SealedDiscoveryRunDeviceAdmissionV1",
    ] {
        assert!(
            !source.contains(forbidden),
            "caller could construct run-device authority via {forbidden:?}"
        );
    }
}

#[test]
fn large_native_authority_variants_are_boxed_without_becoming_reconstructible() {
    let source = read("src/run_device_admission_v1.rs");
    let sealed = section(
        &source,
        "pub enum SealedDiscoveryRunDeviceAdmissionV1 {",
        "\n}",
    );
    require_all(
        sealed,
        &["NativeCuda(Box<SealedNativeCudaRunDeviceAdmissionV1>)"],
    );

    let classified_native = section(&source, "struct ClassifiedNativeCudaRunDeviceV1 {", "\n}");
    require_all(classified_native, &["inventory:", "candidate:"]);
    let classified = section(&source, "enum ClassifiedDiscoveryRunDeviceV1 {", "\n}");
    require_all(
        classified,
        &["NativeCuda(Box<ClassifiedNativeCudaRunDeviceV1>)"],
    );
    require_all(
        &source,
        &[
            "ClassifiedDiscoveryRunDeviceV1::NativeCuda(Box::new(",
            "ClassifiedNativeCudaRunDeviceV1 {",
            "SealedDiscoveryRunDeviceAdmissionV1::NativeCuda(Box::new(",
        ],
    );
    for declaration in [
        "struct ClassifiedNativeCudaRunDeviceV1 {",
        "struct SealedNativeCudaRunDeviceAdmissionV1 {",
    ] {
        let window = declaration_window(&source, declaration);
        for forbidden in ["Clone", "Default", "Deserialize"] {
            assert!(
                !window.contains(forbidden),
                "boxed native authority became reconstructible via {forbidden}"
            );
        }
    }
}

#[test]
fn acquisition_has_no_environment_configuration_or_cpu_fallback_authority() {
    let source = read("src/run_device_admission_v1.rs");
    for forbidden in [
        "std::env",
        "CUDA_VISIBLE_DEVICES",
        "HardwareProbe",
        "DevicePreference",
        "allow_cpu",
        "gpu_preferred",
        "RecomputeOnCpu",
        "FallbackDecision",
        "nvidia-smi",
        "wgpu",
    ] {
        assert!(
            !source.contains(forbidden),
            "non-authoritative route input remained via {forbidden:?}"
        );
    }
}
