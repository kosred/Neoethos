use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-search"))
}

fn read(relative: &str) -> String {
    fs::read_to_string(manifest_dir().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

fn normalized(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

#[test]
fn cubecl_only_device_override_is_not_compiled_with_the_native_adapter() {
    let eval = normalized(&read("src/eval.rs"));
    assert!(
        eval.contains(
            "#[cfg(not(feature=\"gpu-b-adapter\"))]letdevice_override=eval_gpu_devices().first().copied();"
        ),
        "the CubeCL-only device override must be cfg-scoped at its declaration"
    );
    assert_eq!(
        eval.matches("letdevice_override=").count(),
        1,
        "the release route must not grow a second device-selection value"
    );
}

#[test]
fn obsolete_pre_run_native_submission_ceiling_is_removed() {
    let adapter = read("src/gpu_native/prototype_b_population_eval.rs");
    assert!(
        !adapter.contains("pub(crate) fn submission_ceiling("),
        "sizing without the sealed run ordinal is obsolete; exact rebatching lives in evaluation"
    );
}

#[test]
fn obsolete_blind_batch_constant_and_its_test_are_removed() {
    let adapter = read("src/gpu_native/prototype_b_population_eval.rs");
    for token in [
        "const CONSERVATIVE_BATCH",
        "fn the_blind_batch_is_small_enough_for_any_card",
    ] {
        assert!(
            !adapter.contains(token),
            "obsolete no-ordinal blind sizing remains active: {token}"
        );
    }
}

#[test]
fn host_v1_execution_run_has_no_resident_v3_shape_or_session_state() {
    let evidence = normalized(&read("src/population_execution_evidence_v1.rs"));
    for token in [
        "resident_feature_store_session_v3",
        "parent_row_count",
        "parent_feature_count",
        "ResidentPopulationSessionV3",
    ] {
        assert!(
            !evidence.contains(token),
            "host V1 execution still carries standalone resident V3 state: {token}"
        );
    }
}

#[test]
fn prototype_b_card_presence_helper_follows_its_gpu_only_caller() {
    let eval = normalized(&read("src/eval.rs"));
    assert!(
        eval.contains("#[cfg(feature=\"gpu\")]#[inline]fnprototype_b_card_present()->bool{"),
        "the card-presence helper must compile only with its sole caller in the generic GPU block"
    );
    assert_eq!(
        eval.matches("prototype_b_card_present()").count(),
        2,
        "the helper must have exactly one definition and one production caller"
    );
}
