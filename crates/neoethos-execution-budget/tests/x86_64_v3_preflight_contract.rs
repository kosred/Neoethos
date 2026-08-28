use neoethos_execution_budget::{
    DetectedCpuArchitectureV1, X86_64_V3_REQUIREMENTS_V1, X8664V3FeatureSetV1,
    X8664V3PreflightErrorCodeV1, X8664V3RequirementV1, X8664V3SnapshotV1,
    detect_current_x86_64_v3_snapshot_v1, evaluate_x86_64_v3_snapshot_v1,
    require_current_x86_64_v3_v1,
};

fn complete_snapshot() -> X8664V3SnapshotV1 {
    X8664V3SnapshotV1::new(
        DetectedCpuArchitectureV1::X8664,
        X8664V3FeatureSetV1::all_required(),
    )
}

#[test]
fn pins_the_exact_versioned_rust_x86_64_v3_and_xcr0_requirements() {
    let names = X86_64_V3_REQUIREMENTS_V1
        .iter()
        .map(|requirement| requirement.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        names,
        vec![
            "avx",
            "avx2",
            "bmi1",
            "bmi2",
            "cmpxchg16b",
            "f16c",
            "fma",
            "fxsr",
            "lahfsahf",
            "lzcnt",
            "movbe",
            "popcnt",
            "sse",
            "sse2",
            "sse3",
            "sse4.1",
            "sse4.2",
            "ssse3",
            "x87",
            "xsave",
            "xcr0_xmm",
            "xcr0_ymm",
        ]
    );
}

#[test]
fn complete_x86_64_v3_snapshot_is_accepted_without_minting_an_authority_object() {
    assert_eq!(evaluate_x86_64_v3_snapshot_v1(complete_snapshot()), Ok(()));
}

#[test]
fn every_missing_feature_or_xcr0_state_is_refused_fail_closed() {
    for requirement in X86_64_V3_REQUIREMENTS_V1 {
        let snapshot = X8664V3SnapshotV1::new(
            DetectedCpuArchitectureV1::X8664,
            X8664V3FeatureSetV1::all_required().without(requirement),
        );
        let error = evaluate_x86_64_v3_snapshot_v1(snapshot).unwrap_err();

        assert_eq!(
            error.code(),
            X8664V3PreflightErrorCodeV1::MissingRequirements
        );
        assert_eq!(error.missing_requirements(), &[requirement]);
    }
}

#[test]
fn unsupported_architecture_is_refused_even_with_a_caller_supplied_full_set() {
    let snapshot = X8664V3SnapshotV1::new(
        DetectedCpuArchitectureV1::Other,
        X8664V3FeatureSetV1::all_required(),
    );
    let error = evaluate_x86_64_v3_snapshot_v1(snapshot).unwrap_err();

    assert_eq!(
        error.code(),
        X8664V3PreflightErrorCodeV1::UnsupportedArchitecture
    );
    assert_eq!(error.architecture(), DetectedCpuArchitectureV1::Other);
    assert!(error.missing_requirements().is_empty());
}

#[test]
fn diagnostics_are_stable_fail_loud_and_vendor_neutral() {
    let available = X8664V3FeatureSetV1::all_required()
        .without(X8664V3RequirementV1::Avx2)
        .without(X8664V3RequirementV1::Xcr0Ymm);
    let error = evaluate_x86_64_v3_snapshot_v1(X8664V3SnapshotV1::new(
        DetectedCpuArchitectureV1::X8664,
        available,
    ))
    .unwrap_err();
    let diagnostic = error.to_string();

    assert_eq!(
        diagnostic,
        "NEOETHOS_CPU_PREFLIGHT_V1 status=refused required=x86-64-v3 \
         architecture=x86_64 missing=avx2,xcr0_ymm \
         action=install_or_run_on_x86-64-v3"
    );
    assert!(!diagnostic.to_ascii_lowercase().contains("intel"));
    assert!(!diagnostic.to_ascii_lowercase().contains("amd"));
}

#[test]
fn current_host_requirement_uses_the_same_detect_then_evaluate_authority() {
    let detected = detect_current_x86_64_v3_snapshot_v1();
    let evaluated = evaluate_x86_64_v3_snapshot_v1(detected);
    let required = require_current_x86_64_v3_v1();

    assert_eq!(
        required.as_ref().map_err(|error| error.code()),
        evaluated.as_ref().map_err(|error| error.code())
    );
}

#[test]
fn production_detector_is_safe_and_has_no_environment_bypass() {
    let source = include_str!("../src/x86_64_v3.rs");
    let manifest = include_str!("../Cargo.toml");

    assert!(source.contains("is_x86_feature_detected!"));
    assert!(source.contains("raw_cpuid::CpuId"));
    assert!(!source.contains("detect!(\"lahfsahf\""));
    assert!(manifest.contains("raw-cpuid = { version = \"11.6.0\", default-features = false }"));
    for forbidden in [
        "std::env",
        "var_os(",
        "var(",
        "unsafe",
        "__cpuid",
        "_xgetbv",
        "Vendor",
        "Capability",
        "Permit",
    ] {
        assert!(
            !source.contains(forbidden),
            "production preflight must not contain forbidden token {forbidden}"
        );
    }
}

#[test]
fn feature_set_cannot_silently_default_to_v3_compatible() {
    assert_ne!(
        X8664V3FeatureSetV1::empty(),
        X8664V3FeatureSetV1::all_required()
    );
    assert!(
        evaluate_x86_64_v3_snapshot_v1(X8664V3SnapshotV1::new(
            DetectedCpuArchitectureV1::X8664,
            X8664V3FeatureSetV1::empty(),
        ))
        .is_err()
    );
}
