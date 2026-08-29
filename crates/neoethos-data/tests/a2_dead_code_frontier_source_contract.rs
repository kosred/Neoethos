//! Source-only contract proving the former A2 dead-code frontier is connected.

use std::fs;
use std::path::PathBuf;

fn read(relative: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative);
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

#[test]
fn bound_robust_receipt_is_validated_after_run_device_binding() {
    let source = read("src/core/gpu_resident_feature_store_v3.rs");
    let binding = "robust_normalization.bind_run_device_v2(&run_device)?;";
    let validation = "robust_normalization.validate_working_set(&working_set)?;";

    let binding_offset = source.find(binding).expect("bound Robust receipt");
    let validation_offsets = source
        .match_indices(validation)
        .map(|(offset, _)| offset)
        .collect::<Vec<_>>();
    assert_eq!(
        validation_offsets.len(),
        1,
        "working-set validation must occur exactly once through the bound receipt"
    );
    assert!(
        validation_offsets[0] > binding_offset,
        "the bound receipt, including its binding identity, must validate the working set"
    );
    assert!(source.contains("if self.binding_identity_sha256 == [0; 32]"));
}

#[test]
fn regime_append_is_consumed_without_dead_code_suppression() {
    let regime = read("src/core/gpu_resident_regime_v3.rs");
    let store = read("src/core/gpu_resident_feature_store_v3.rs");
    assert!(regime.contains("pub(crate) fn append_to("));
    assert!(store.contains("regime_input.append_to(&mut assembler, regime_bindings)?"));
    assert!(!regime.contains("dead_code"));
}
