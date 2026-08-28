use neoethos_data::core::dataset_manifest::canonical_dataset_root;
use neoethos_dataset_contracts::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe,
};
use tempfile::tempdir;

#[test]
fn canonical_root_is_one_reversible_contained_component() {
    let root = tempdir().expect("temporary root");
    let identity = CanonicalDatasetIdentity::external(
        "github-snapshot",
        "XAUUSD.r/../CON",
        CanonicalTimeframe::M5,
        BarTimestampConvention::BarOpen,
    )
    .expect("identity");

    let dataset_root = canonical_dataset_root(root.path(), &identity).expect("contained root");
    assert_eq!(dataset_root.parent(), Some(root.path()));
    assert_eq!(
        dataset_root.file_name().and_then(|name| name.to_str()),
        Some(identity.to_path_component().as_str())
    );
}

#[test]
fn canonical_root_rejects_a_preexisting_link() {
    let root = tempdir().expect("temporary root");
    let outside = tempdir().expect("outside root");
    let identity = CanonicalDatasetIdentity::external(
        "fixture",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("identity");
    let linked = root.path().join(identity.to_path_component());

    #[cfg(unix)]
    std::os::unix::fs::symlink(outside.path(), &linked).expect("create symlink fixture");
    #[cfg(windows)]
    std::os::windows::fs::symlink_dir(outside.path(), &linked).expect("create symlink fixture");

    assert!(canonical_dataset_root(root.path(), &identity).is_err());
}
