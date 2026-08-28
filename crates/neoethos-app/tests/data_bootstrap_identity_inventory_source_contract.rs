use std::path::{Path, PathBuf};

fn repository_root() -> PathBuf {
    if let Some(manifest_dir) = option_env!("CARGO_MANIFEST_DIR") {
        return Path::new(manifest_dir)
            .parent()
            .and_then(Path::parent)
            .expect("neoethos-app manifest must be under <repo>/crates")
            .to_path_buf();
    }
    std::env::current_dir().expect("standalone source contract working directory")
}

#[test]
fn data_bootstrap_exposes_authoritative_canonical_dataset_receipts() {
    let source = std::fs::read_to_string(
        repository_root().join("crates/neoethos-app/src/server/system_status.rs"),
    )
    .expect("read system status source");

    for required in [
        "DatasetDiscovery::scan_metadata",
        "CanonicalDatasetInventoryDto",
        "dataset_identity",
        "generation",
        "manifest_binding_sha256",
        "source_kind",
        "identity.is_broker_real()",
        "verification",
        "SkippedDatasetInventoryDto",
        "category",
        "detail",
    ] {
        assert!(
            source.contains(required),
            "data bootstrap is missing canonical inventory contract {required}"
        );
    }
    assert!(
        !source.contains("strip_prefix(\"symbol=\")"),
        "data bootstrap must not discover the retired human path layout"
    );
}
