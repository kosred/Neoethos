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

fn require_cfg_immediately_before(source: &str, item: &str, cfg: &str) {
    let item_at = source
        .find(item)
        .unwrap_or_else(|| panic!("missing release-surface item {item:?}"));
    let mut attributes = Vec::new();
    for line in source[..item_at].trim_end().lines().rev() {
        let line = line.trim();
        if line.starts_with("#[") {
            attributes.push(line);
        } else {
            break;
        }
    }
    assert!(
        attributes.contains(&cfg),
        "{item:?} must be compiled only under {cfg:?}"
    );
}

#[test]
fn default_release_does_not_compile_orphaned_adapter_entrypoints() {
    let eval = read("src/eval.rs");
    let lib = read("src/lib.rs");
    let fallback = read("src/gpu_fallback.rs");

    assert!(
        !eval.contains("fn evaluate_population_core_with_evidence("),
        "the retired duplicate evidence entrypoint has no caller in any feature graph"
    );
    require_cfg_immediately_before(
        &lib,
        "mod gpu_fallback;",
        "#[cfg(any(test, feature = \"gpu-b-adapter\"))]",
    );
    assert!(
        !lib.contains("pub mod gpu_fallback;"),
        "the crate-private native retry implementation is not a public module"
    );
    assert!(fallback.contains("decide_strict_population_failure_v1("));
}

#[test]
fn native_probe_and_retry_internals_follow_their_real_feature_callers() {
    let route = read("src/strict_discovery_device_route_v1.rs");

    for native_only in [
        "const PROBE_HASH_DOMAIN_V1",
        "const DEVICE_HASH_DOMAIN_V1",
        "fn hex_lower(",
        "fn ordinal_observation_manifest_sha256(",
        "fn seal_no_compatible_gpu_probe_receipt_v1(",
    ] {
        require_cfg_immediately_before(&route, native_only, "#[cfg(feature = \"gpu-b-native\")]");
    }
    for native_or_unit_oracle in [
        "pub(crate) enum StrictCudaProbeFailureKindV1",
        "pub(crate) enum CudaOrdinalProbeOutcomeV1",
        "pub(crate) struct StrictDiscoveryProbeObservationV1",
        "pub(crate) enum NoCompatibleGpuReasonV1",
        "pub(crate) enum UnsealedStrictDiscoveryDeviceRouteV1",
        "pub(crate) fn classify_strict_discovery_probe_observation_v1(",
    ] {
        require_cfg_immediately_before(
            &route,
            native_or_unit_oracle,
            "#[cfg(any(test, feature = \"gpu-b-native\"))]",
        );
    }
    for adapter_or_unit_oracle in [
        "pub(crate) enum StrictNativeFailureKindV1",
        "pub(crate) enum StrictNativeFailureActionV1",
        "pub(crate) fn decide_strict_native_failure_v1(",
    ] {
        require_cfg_immediately_before(
            &route,
            adapter_or_unit_oracle,
            "#[cfg(any(test, feature = \"gpu-b-adapter\"))]",
        );
    }
    let route_errors = section(
        &route,
        "pub(crate) enum StrictDiscoveryDeviceRouteErrorCodeV1 {",
        "\n}",
    );
    for native_only_error in [
        "MissingCudaBuildManifest,",
        "DeviceIdentityMismatch,",
        "WrongDeviceRoute,",
    ] {
        require_cfg_immediately_before(
            route_errors,
            native_only_error,
            "#[cfg(feature = \"gpu-b-native\")]",
        );
    }

    let no_native_probe = section(
        &route,
        "#[cfg(not(feature = \"gpu-b-native\"))]\nfn probe_real_strict_discovery_device_route_v1()",
        "\npub fn acquire_strict_discovery_device_admission_v1()",
    );
    assert!(no_native_probe.contains("NativeAdapterNotCompiled"));
    for forbidden in [
        "StrictDiscoveryProbeObservationV1",
        "classify_strict_discovery_probe_observation_v1",
        "seal_no_compatible_gpu_probe_receipt_v1",
        "Sha256",
    ] {
        assert!(
            !no_native_probe.contains(forbidden),
            "the no-adapter build must fail directly, not compile native probe machinery via {forbidden:?}"
        );
    }
    assert!(
        !route.contains("InvalidCudaBuildManifest"),
        "the retired, never-produced strict-route error variant must be removed"
    );
}

#[test]
fn default_release_has_no_fake_sealed_route_constructor_or_dead_code_suppression() {
    let route = read("src/strict_discovery_device_route_v1.rs");
    let sealed_route = section(
        &route,
        "pub(crate) struct SealedStrictDiscoveryDeviceRouteV1 {",
        "\n}",
    );
    assert!(sealed_route.contains("_sealed: ()"));
    require_cfg_immediately_before(
        sealed_route,
        "kind: SealedStrictDiscoveryDeviceRouteKindV1",
        "#[cfg(feature = \"gpu-b-native\")]",
    );
    require_cfg_immediately_before(
        &route,
        "enum SealedStrictDiscoveryDeviceRouteKindV1",
        "#[cfg(feature = \"gpu-b-native\")]",
    );
    let route_kind = section(
        &route,
        "enum SealedStrictDiscoveryDeviceRouteKindV1 {",
        "\n}",
    );
    assert!(route_kind.contains("NativeCuda("));
    assert!(route_kind.contains("CpuNoCompatibleGpu("));

    let native_probe = section(
        &route,
        "#[cfg(feature = \"gpu-b-native\")]\nfn probe_real_strict_discovery_device_route_v1()",
        "#[cfg(not(feature = \"gpu-b-native\"))]",
    );
    assert!(native_probe.contains("kind: SealedStrictDiscoveryDeviceRouteKindV1::NativeCuda("));
    assert!(
        native_probe.contains("kind: SealedStrictDiscoveryDeviceRouteKindV1::CpuNoCompatibleGpu(")
    );
    assert!(native_probe.contains("seal_no_compatible_gpu_probe_receipt_v1("));

    let no_native_probe = section(
        &route,
        "#[cfg(not(feature = \"gpu-b-native\"))]\nfn probe_real_strict_discovery_device_route_v1()",
        "\npub fn acquire_strict_discovery_device_admission_v1()",
    );
    assert!(
        !no_native_probe.contains("SealedStrictDiscoveryDeviceRouteV1 {")
            && !no_native_probe.contains("SealedStrictDiscoveryDeviceRouteKindV1"),
        "a no-adapter build cannot construct an authority-shaped route"
    );

    for forbidden in [
        "allow(dead_code)",
        "expect(dead_code)",
        "fake_no_gpu",
        "synthetic_no_gpu",
    ] {
        assert!(
            !route.contains(forbidden),
            "release warning closure cannot suppress or fake strict authority via {forbidden:?}"
        );
    }
}
