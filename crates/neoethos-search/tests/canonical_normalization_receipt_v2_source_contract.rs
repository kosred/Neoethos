use std::{fs, path::PathBuf};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("neoethos-search must live below the workspace root")
        .to_path_buf()
}

fn source(relative: &str) -> String {
    fs::read_to_string(workspace_root().join(relative)).expect(relative)
}

#[test]
fn search_does_not_mint_or_choose_the_data_normalization_authority() {
    let discovery = source("crates/neoethos-search/src/discovery.rs");
    assert!(!discovery.contains("canonical_discovery_robust_normalization_receipt_v2"));
    assert!(!discovery.contains("CanonicalRobustNormalizationTrainingReceiptV2"));

    let data = source("crates/neoethos-data/src/core/gpu_resident_robust_normalization_v2.rs");
    for required in [
        "pub(crate) struct SealedCanonicalRobustNormalizationSplitV2",
        "training_rows: Range<usize>",
        "row_count: usize",
        "enabled: bool",
        "fn consume",
        "seal_canonical_robust_normalization_split_from_pinned_v2",
        "seal_canonical_robust_normalization_split_from_frame_v2",
        "prepare_resident_robust_normalization_input_v2",
        "sealed_data_runtime_normalization_mode_v2",
        "normalization_scratch_bytes",
        "fit_metadata_bytes",
    ] {
        assert!(data.contains(required), "typed receipt lost `{required}`");
    }
    for forbidden in [
        "pub fn seal(",
        "training_rows: Range<usize>, enabled: bool",
        "row_count: usize, training_rows",
    ] {
        assert!(
            !data.contains(forbidden),
            "Data split accepts caller-selected authority via `{forbidden}`"
        );
    }
    for forbidden in ["impl Clone", "Serialize", "Deserialize", "Default"] {
        assert!(
            !data.contains(forbidden),
            "receipt became caller-mintable via `{forbidden}`"
        );
    }

    let preflight =
        source("crates/neoethos-data/src/core/gpu_only_feature_workspace_preflight_v3.rs");
    let prepared = preflight
        .split_once("pub struct PreparedGpuOnlyFeatureWorkspacePreflightV3 {")
        .expect("phase-zero Data carrier")
        .1
        .split_once("\n}")
        .expect("phase-zero Data carrier body")
        .0;
    assert!(prepared.contains("robust_normalization_split"));
    assert!(preflight.contains("seal_canonical_robust_normalization_split_from_pinned_v2("));
}
