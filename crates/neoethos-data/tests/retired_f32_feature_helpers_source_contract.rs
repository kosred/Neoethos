use std::path::{Path, PathBuf};

fn repository_root() -> PathBuf {
    if let Some(manifest_dir) = option_env!("CARGO_MANIFEST_DIR") {
        return Path::new(manifest_dir)
            .parent()
            .and_then(Path::parent)
            .expect("neoethos-data manifest must be under <repo>/crates")
            .to_path_buf();
    }
    std::env::current_dir().expect("standalone source contract working directory")
}

#[test]
fn superseded_f32_feature_helpers_are_deleted() {
    let root = repository_root();
    for relative in [
        "crates/neoethos-data/src/core/features.rs",
        "crates/neoethos-data/src/core/normalization.rs",
    ] {
        let path = root.join(relative);
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        for retired in [
            "align_features_by_ns",
            "normalize_feature_matrix",
            "normalize_feature_series_in_place",
            "normalize_feature_matrix_copy",
            "NORM_FIT_FRACTION",
            "Array2<f32>",
        ] {
            assert!(
                !source.contains(retired),
                "{} still contains retired f32 helper {retired}",
                path.display()
            );
        }
    }
}
