//! Source-only contract for the intentionally narrow A2 dead-code frontier.

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
fn deferred_regime_seams_use_only_item_scoped_expectations() {
    let source = read("src/core/gpu_resident_regime_v3.rs");
    let reason = "the complete ordered A2 ledger still fails closed before run-device acquisition";
    let expected_attribute =
        format!("#[expect(\n        dead_code,\n        reason = \"{reason}\"\n    )]");

    let item = "pub(crate) fn append_to(";
    let item_offset = source
        .find(item)
        .unwrap_or_else(|| panic!("missing {item}"));
    let prefix_start = item_offset.saturating_sub(240);
    let prefix = &source[prefix_start..item_offset];
    assert!(
        prefix.contains(&expected_attribute),
        "{item} must carry the exact item-scoped deferred-frontier expectation"
    );

    assert_eq!(
        source.matches(&expected_attribute).count(),
        1,
        "only the truly dead Regime append seam may expect dead_code"
    );
    assert!(!source.contains("#![allow(dead_code)]"));
    assert!(!source.contains("#[allow(dead_code)]"));
}
