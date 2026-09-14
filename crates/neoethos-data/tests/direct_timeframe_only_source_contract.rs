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
fn production_accepts_only_direct_timeframe_generations() {
    let root = repository_root();
    assert!(
        !root
            .join("crates/neoethos-data/src/core/resample.rs")
            .exists(),
        "the retired M1-to-higher-timeframe implementation must be deleted"
    );

    let files = [
        "crates/neoethos-data/src/lib.rs",
        "crates/neoethos-data/src/core/canonical_ohlcv.rs",
        "crates/neoethos-data/src/core/direct_timeframes.rs",
        "crates/neoethos-data/src/core/features.rs",
        "crates/neoethos-feature-contracts/src/identity.rs",
        "crates/neoethos-app/src/lib.rs",
        "crates/neoethos-app/src/app_services/discovery.rs",
        "crates/neoethos-search/src/discovery.rs",
        "crates/neoethos-search/src/orchestration.rs",
        "crates/neoethos-cli/src/main.rs",
    ];
    let retired = [
        "resample_ohlcv",
        "ensure_timeframes_with_resample",
        "FixedGridResample",
        "FIXED_GRID_RESAMPLE_SEMANTIC_VERSION",
        "rebuild_stale_higher_tfs",
        "NEOETHOS_BOT_REBUILD_STALE_HIGHER_TFS",
        "Resample = 3",
        "FeatureOperationTagV1::Resample",
        "has no independently evidenced close/alignment rule",
        "calendar base timeframe",
        "align_features_by_ns",
    ];
    for relative in files {
        let path = root.join(relative);
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        for token in retired {
            assert!(
                !source.contains(token),
                "{} still exposes retired timeframe synthesis token {token}",
                path.display()
            );
        }
    }

    // The sealed loader keeps a migration-only tombstone so a configuration
    // written by the older release is named and ignored instead of becoming a
    // hard startup failure. The key is not a field and no runtime source above
    // may consume it.
    let config = std::fs::read_to_string(root.join("crates/neoethos-core/src/config.rs"))
        .expect("sealed config authority must exist");
    assert!(config.contains("path: \"models.data_runtime.rebuild_stale_higher_tfs\""));
    assert!(config.contains("kind: RetiredKind::Deleted"));

    let direct =
        std::fs::read_to_string(root.join("crates/neoethos-data/src/core/direct_timeframes.rs"))
            .expect("direct timeframe contract must exist");
    assert!(direct.contains("require_direct_timeframes"));
    assert!(
        !direct.contains("REQUIRED_DIRECT_TIMEFRAMES"),
        "the data layer must not impose an unrelated fixed timeframe bundle"
    );

    let data = std::fs::read_to_string(root.join("crates/neoethos-data/src/lib.rs"))
        .expect("data runtime source must exist");
    assert!(data.contains("align_calendar_feature_columns_by_observed_next_open_ms"));
    assert!(data.contains("next_direct_bar_open_observed_span_expiry_v1"));
}
