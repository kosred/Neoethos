use neoethos_search::{
    CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_SCHEMA_V1,
    CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_VERSION_V1,
};

#[test]
fn schema_identity_remains_the_only_public_generation_zero_result_surface() {
    assert_eq!(
        CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_SCHEMA_V1,
        "neoethos.canonical-native-generation-zero-research-result.v1"
    );
    assert_eq!(
        CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_VERSION_V1,
        1
    );

    let root = include_str!("../src/lib.rs");
    assert!(root.contains("mod canonical_native_generation_zero_result_v1;"));
    assert!(!root.contains("pub mod canonical_native_generation_zero_result_v1;"));
    let result_exports = root
        .split_once("pub use canonical_native_generation_zero_result_v1::{")
        .expect("result export block")
        .1
        .split_once("};")
        .expect("result export block end")
        .0;
    let exported: Vec<_> = result_exports
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .collect();
    assert_eq!(
        exported,
        [
            "CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_SCHEMA_V1",
            "CANONICAL_NATIVE_GENERATION_ZERO_RESEARCH_RESULT_VERSION_V1",
        ],
        "schema/version must remain the complete public result surface"
    );
}

#[test]
fn raw_fixed_metadata_sizing_authority_is_crate_private() {
    let source = include_str!("../src/canonical_native_generation_zero_result_v1.rs");
    let planner = source
        .split_once("pub(crate) enum CanonicalNativeGenerationZeroResultSizePlanErrorCodeV1")
        .expect("private planner authority start")
        .1
        .split_once("// END CANONICAL_NATIVE_GENERATION_ZERO_SIZE_PLANNER_V1")
        .expect("planner end")
        .0;
    for line in planner.lines() {
        assert!(
            !line.trim_start().starts_with("pub "),
            "planner authority leaked a public item: {line}"
        );
    }
    for required in [
        "pub(crate) struct CanonicalNativeGenerationZeroResultSizePlanErrorV1",
        "pub(crate) struct CanonicalNativeGenerationZeroResultSizePlanV1",
        "pub(crate) fn checked_new(",
    ] {
        assert!(
            planner.contains(required),
            "missing private authority: {required}"
        );
    }
}
