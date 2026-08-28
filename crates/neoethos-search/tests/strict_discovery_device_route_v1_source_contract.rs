use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-search"))
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
        assert!(
            source.contains(token),
            "strict Discovery device route is missing {token:?}"
        );
    }
}

fn normalized(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

#[test]
fn exact_ordinal_and_no_compatible_gpu_receipt_are_opaque_real_probe_outputs() {
    let route = read("src/strict_discovery_device_route_v1.rs");
    let ordinal = section(&route, "pub struct ExactCudaDeviceOrdinalV1 {", "\n}");
    let no_gpu = section(
        &route,
        "pub struct SealedNoCompatibleGpuProbeReceiptV1 {",
        "\n}",
    );

    require_all(
        ordinal,
        &[
            "selected_ordinal: u32",
            "cuda_device_identity_sha256: String",
            "cuda_build_manifest_sha256: String",
            "probe_receipt_identity_sha256: String",
        ],
    );
    require_all(
        no_gpu,
        &[
            "probe_receipt_identity_sha256: String",
            "native_adapter_compiled: bool",
            "runtime_loaded: bool",
            "reported_device_count: u32",
            "ordinal_observation_manifest_sha256: String",
        ],
    );
    assert!(
        !ordinal.contains("pub "),
        "exact ordinal fields must be private"
    );
    assert!(
        !no_gpu.contains("pub "),
        "no-GPU receipt fields must be private"
    );

    for forbidden in [
        "Deserialize",
        "impl Default for ExactCudaDeviceOrdinalV1",
        "impl Default for SealedNoCompatibleGpuProbeReceiptV1",
        "pub fn new(",
        "pub fn from_",
        "From<",
        "unsafe",
    ] {
        assert!(
            !route.contains(forbidden),
            "caller can mint or deserialize device authority via {forbidden:?}"
        );
    }
}

#[test]
fn public_one_shot_admission_is_opaque_and_has_no_caller_supplied_probe_or_preference() {
    let route = read("src/strict_discovery_device_route_v1.rs");
    let lib = read("src/lib.rs");
    let admission = section(
        &route,
        "pub struct SealedStrictDiscoveryDeviceAdmissionV1 {",
        "\n}",
    );
    require_all(admission, &["route: SealedStrictDiscoveryDeviceRouteV1"]);
    assert!(
        !admission.contains("pub "),
        "one-shot admission fields must be private"
    );

    let acquisition = section(
        &route,
        "pub fn acquire_strict_discovery_device_admission_v1()",
        "-> Result<SealedStrictDiscoveryDeviceAdmissionV1",
    );
    assert!(
        acquisition.trim().is_empty(),
        "strict route acquisition must take no caller-supplied facts"
    );
    require_all(
        &route,
        &[
            "probe_real_strict_discovery_device_route_v1()",
            "neoethos_gpu_cuda::probe_cuda_device_count_v1()",
            "neoethos_gpu_cuda::PopulationSession::create(",
            "i32::try_from(ordinal)",
            "neoethos_gpu_cuda::cuda_build_manifest_v1()",
            "validate_cuda_build_manifest_v1(",
        ],
    );
    require_all(
        &lib,
        &[
            "SealedStrictDiscoveryDeviceAdmissionV1",
            "acquire_strict_discovery_device_admission_v1",
        ],
    );
    for forbidden in [
        "pub fn require_exact_cuda_device_ordinal_v1()",
        "require_exact_cuda_device_ordinal_v1,",
        "impl Clone for SealedStrictDiscoveryDeviceAdmissionV1",
        "Deserialize<'de> for SealedStrictDiscoveryDeviceAdmissionV1",
        "impl Default for SealedStrictDiscoveryDeviceAdmissionV1",
    ] {
        assert!(
            !route.contains(forbidden) && !lib.contains(forbidden),
            "detachable/reconstructible device authority remains via {forbidden:?}"
        );
    }
    for forbidden in [
        "card_present: bool",
        "allow_cpu: bool",
        "device_preference:",
        "requested_ordinal:",
        "std::env",
    ] {
        assert!(
            !acquisition.contains(forbidden),
            "caller/config can influence strict route acquisition via {forbidden:?}"
        );
    }
}

#[test]
fn synthetic_probe_matrix_can_classify_but_cannot_seal_execution_authority() {
    let route = read("src/strict_discovery_device_route_v1.rs");
    let classifier = section(
        &route,
        "pub(crate) fn classify_strict_discovery_probe_observation_v1(",
        "\n}",
    );
    require_all(
        classifier,
        &[
            "UnsealedStrictDiscoveryDeviceRouteV1",
            "native_adapter_compiled",
            "runtime_loaded",
            "reported_device_count",
            "ordinal_outcomes",
            "NativeAdapterNotCompiled",
            "CudaRuntimeUnavailable",
            "VisibleGpuBuildIncompatible",
            "IncompleteCudaProbe",
            "CudaProbeFault",
        ],
    );
    for forbidden in [
        "ExactCudaDeviceOrdinalV1 {",
        "SealedNoCompatibleGpuProbeReceiptV1 {",
        "seal_",
        "Settings",
        "DevicePreference",
        "FallbackPolicy",
    ] {
        assert!(
            !classifier.contains(forbidden),
            "unsealed classifier acquired authority or config input via {forbidden:?}"
        );
    }
}

#[test]
fn cpu_receipt_requires_a_loaded_native_probe_that_enumerated_zero_devices() {
    let route = read("src/strict_discovery_device_route_v1.rs");
    let sealer = section(&route, "fn seal_no_compatible_gpu_probe_receipt_v1(", "\n}");
    require_all(
        sealer,
        &[
            "native_adapter_compiled",
            "runtime_loaded",
            "reported_device_count",
            "NativeAdapterNotCompiled",
            "CudaRuntimeUnavailable",
            "VisibleGpuBuildIncompatible",
            "reported_device_count != 0",
        ],
    );
    for forbidden in [
        "NoCompatibleGpuReasonV1::NoBuildCompatibleCudaOrdinal",
        "reason: NoCompatibleGpuReasonV1::NoBuildCompatibleCudaOrdinal",
    ] {
        assert!(
            !sealer.contains(forbidden),
            "a visible but build-incompatible GPU must not mint a CPU receipt via {forbidden:?}"
        );
    }
}

#[test]
fn exact_route_is_owned_by_the_population_run_and_cannot_be_detached_per_evaluation() {
    let evidence = read("src/population_execution_evidence_v1.rs");
    let native = read("src/population_execution_evidence_v1/native_cuda_resident_v1.rs");
    let discovery = read("src/discovery.rs");

    let run = section(
        &evidence,
        "pub(crate) struct ExactPopulationExecutionRunV1<'a> {",
        "\n}",
    );
    let evaluation = section(
        &evidence,
        "pub(crate) struct ExactPopulationEvaluationV1<'a> {",
        "\n}",
    );
    require_all(
        run,
        &[
            "strict_device_route: SealedStrictDiscoveryDeviceRouteV1",
            "native_residency:",
        ],
    );
    require_all(evaluation, &["strict_device_route:"]);
    require_all(
        &evidence,
        &[
            "admission: SealedStrictDiscoveryDeviceAdmissionV1",
            "admission.into_route_v1()",
            "require_cpu_route_receipt_v1(",
            "require_exact_cuda_device_ordinal_v1(",
        ],
    );
    let begin = section(
        &evidence,
        "pub(crate) fn begin_exact_population_execution_run_v1<'a>(",
        "\n}\n\nimpl ExactPopulationExecutionRunV1",
    );
    for forbidden in [
        "resolve_strict_discovery_device_route_v1()",
        "probe_real_strict_discovery_device_route_v1()",
        "acquire_strict_discovery_device_admission_v1()",
    ] {
        assert!(
            !begin.contains(forbidden),
            "begin must consume the caller's one-shot admission, not re-probe via {forbidden:?}"
        );
    }
    assert_eq!(
        discovery
            .matches("acquire_strict_discovery_device_admission_v1()")
            .count(),
        1,
        "Discovery must acquire one and only one strict route admission"
    );
    let acquire_at = discovery
        .find("acquire_strict_discovery_device_admission_v1()")
        .expect("Discovery strict admission acquisition");
    let begin_at = discovery
        .find("begin_exact_population_execution_run_v1(")
        .expect("Discovery exact population run start");
    assert!(
        acquire_at < begin_at,
        "Discovery must acquire the sealed route before moving it into the run"
    );
    let run_start = &discovery[begin_at..];
    require_all(run_start, &["strict_device_admission"]);
    require_all(
        &native,
        &[
            "evidence.require_exact_cuda_device_ordinal_v1()?",
            "selected_ordinal()",
            "native population session selected CUDA ordinal",
        ],
    );
    assert!(
        !native.contains("device_override.unwrap_or(0)"),
        "native run still accepts an unsealed default/caller ordinal"
    );
}

#[test]
fn backend_strings_and_caller_probe_boole_cannot_authorize_cpu_on_a_card() {
    let backend = read("src/backend.rs");
    for forbidden in [
        "pub struct HardwareProbe",
        "HardwareProbe { card_present",
        "cpu_forced",
        "cpu-forced",
        "GPU_PREFERRED",
        "FallbackPolicy::AllowCpu",
    ] {
        assert!(
            !backend.contains(forbidden),
            "legacy backend authority escape remains via {forbidden:?}"
        );
    }
    require_all(
        &backend,
        &[
            "ExactPopulationEvaluationV1",
            "require_cpu_route_receipt_v1(",
            "require_exact_cuda_device_ordinal_v1(",
            "PopulationEvalEngine::CudaNativeF64",
        ],
    );
}

#[test]
fn selected_gpu_failures_have_no_cpu_recompute_or_mixed_engine_route() {
    let fallback = read("src/gpu_fallback.rs");
    let eval = read("src/eval.rs");

    require_all(
        &fallback,
        &[
            "RetrySameOrdinal",
            "selected_ordinal",
            "next_batch_size",
            "FailLoud",
        ],
    );
    for forbidden in [
        "FallbackToCpu",
        "RecomputeOnCpu",
        "decide_env",
        "require_gpu()",
    ] {
        assert!(
            !fallback.contains(forbidden),
            "strict native failure policy retains {forbidden:?}"
        );
    }
    for forbidden in [
        "RecomputeOnCpu",
        "decide_env(",
        "note_cpu_fallback(",
        "cpu_forced",
        "PopulationEvalEngine::CubeclF64",
    ] {
        assert!(
            !eval.contains(forbidden),
            "GPU-selected evaluation retains mixed/CPU route {forbidden:?}"
        );
    }
    require_all(
        &eval,
        &[
            "require_exact_cuda_device_ordinal_v1()?",
            "require_cpu_route_receipt_v1()?",
            "refusing CPU substitution",
        ],
    );
}

#[test]
fn strict_engine_preflight_never_substitutes_cubecl_for_native_cuda() {
    let engine = read("src/engine_identity.rs");
    let preflight = section(&engine, "pub fn strict_engine_preflight(", "\n}");
    assert!(
        !preflight.contains("Ok(PopulationEvalEngine::CubeclF64)"),
        "GPU-selected Discovery still substitutes a different engine"
    );
    require_all(
        preflight,
        &[
            "PopulationEvalEngine::CudaNativeF64",
            "PrototypeBReadiness::CompiledButUnavailable",
            "PrototypeBReadiness::NotCompiledIn",
            "Err(",
        ],
    );
}

#[test]
fn no_gpu_cpu_execution_consumes_the_sealed_probe_receipt_before_cpu_work() {
    let backend = read("src/backend.rs");
    let dispatch = section(
        &backend,
        "fn evaluate_population_core_with_backend_and_audit_inner(",
        "\n}\n\n/// Unit-only backend oracle",
    );
    let cpu_at = dispatch
        .find("require_cpu_route_receipt_v1(")
        .expect("CPU dispatch must validate the sealed no-compatible-GPU receipt");
    let work_at = dispatch
        .find("crate::eval::validation_backtest_population_cpu(inputs)")
        .expect("CPU dispatch boundary");
    assert!(
        cpu_at < work_at,
        "CPU work began before the real no-compatible-GPU receipt was checked"
    );
    require_all(
        dispatch,
        &[
            "run_with_sealed_no_gpu_receipt(",
            "no_gpu_receipt",
            "CpuStrategyCategory::PopulationEvaluation",
        ],
    );
    assert!(
        !dispatch.contains("cpu_strategy::run("),
        "stale backend fallback/config policy can still block or authorize sealed CPU work"
    );
}

#[test]
fn cpu_receipt_is_an_inhabited_opaque_wrapper_over_feature_gated_authority() {
    let route = read("src/strict_discovery_device_route_v1.rs");
    let compact = normalized(&route);

    let wrapper = section(
        &route,
        "pub(crate) struct SealedCpuDiscoveryRouteReceiptV2 {",
        "\n}",
    );
    require_all(
        wrapper,
        &[
            "_sealed: ()",
            "#[cfg(feature = \"gpu-b-native\")]",
            "kind: SealedCpuDiscoveryRouteReceiptKindV2",
        ],
    );
    for forbidden in ["pub ", "LegacyCudaZero", "PhysicalGpuAbsence"] {
        assert!(
            !wrapper.contains(forbidden),
            "opaque CPU receipt wrapper exposes internal authority through {forbidden:?}",
        );
    }

    require_all(
        &compact,
        &[
            "#[cfg(feature=\"gpu-b-native\")]#[derive(Clone,Debug,PartialEq,Eq)]enumSealedCpuDiscoveryRouteReceiptKindV2{",
            "LegacyCudaZero(SealedNoCompatibleGpuProbeReceiptV1),",
            "#[cfg(feature=\"gpu-cuda\")]PhysicalGpuAbsence{",
            "match&self.kind{",
            "SealedCpuDiscoveryRouteReceiptKindV2::LegacyCudaZero(receipt)=>",
            "#[cfg(not(feature=\"gpu-b-native\"))]{false}",
        ],
    );
    assert_eq!(
        route.matches("LegacyCudaZero(").count(),
        3,
        "the legacy CUDA-zero receipt must have exactly one variant, one match arm and one native constructor",
    );
    for forbidden in [
        "impl Default for SealedCpuDiscoveryRouteReceiptV2",
        "_ =>",
        "todo!",
        "unreachable!",
        "#[allow(",
        "#[expect(",
    ] {
        assert!(
            !route.contains(forbidden),
            "CPU receipt uses a fake/default/suppressed authority route via {forbidden:?}",
        );
    }

    let native_probe = section(
        &route,
        "#[cfg(feature = \"gpu-b-native\")]\nfn probe_real_strict_discovery_device_route_v1()",
        "#[cfg(not(feature = \"gpu-b-native\"))]",
    );
    require_all(
        native_probe,
        &[
            "SealedCpuDiscoveryRouteReceiptV2 {",
            "_sealed: ()",
            "kind: SealedCpuDiscoveryRouteReceiptKindV2::LegacyCudaZero(",
        ],
    );
}
