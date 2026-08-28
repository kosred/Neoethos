const FEATURE_FRAME_SOURCE: &str = include_str!("../src/core/features.rs");

#[test]
fn feature_frame_exposes_one_typed_arbitrary_row_selector() {
    assert!(
        FEATURE_FRAME_SOURCE.contains("pub fn select_rows("),
        "FeatureFrame must expose the canonical typed select_rows API"
    );
    assert!(
        FEATURE_FRAME_SOURCE.contains("row_indices: &[usize]"),
        "select_rows must accept explicit typed row indices"
    );
}
