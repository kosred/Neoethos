use std::fs;
use std::path::PathBuf;

#[test]
fn system_status_measures_the_active_vortex_feature_run_root_only() {
    let status = fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/server/system_status.rs"),
    )
    .expect("read system-status source");
    let accounting = fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/server/feature_store_disk.rs"),
    )
    .expect("read feature-store accounting source");
    let server =
        fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/server/mod.rs"))
            .expect("read server module source");
    let data = fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../neoethos-data/src/lib.rs"),
    )
    .expect("read canonical feature-run root source");
    let source = format!("{status}\n{accounting}\n{server}\n{data}");

    assert!(
        source.contains("neoethos_vortex_feature_runs"),
        "system status must inspect the active Vortex feature-run root"
    );
    assert!(
        source.contains("symlink_metadata"),
        "recursive disk accounting must inspect entries without following symlinks"
    );
    assert!(
        server.contains("mod feature_store_disk;")
            && status.contains("neoethos_data::vortex_feature_run_root()")
            && status.contains("vortex_feature_store_disk_mb("),
        "the status endpoint must call the active Vortex accounting module"
    );
    assert!(
        !accounting.contains("neoethos_vortex_feature_runs"),
        "the app must consume the data crate's canonical scratch-root API instead of duplicating it"
    );
    assert!(
        !source.contains("neoethos_feature_store") && !source.contains(".fstore"),
        "retired .fstore paths must disappear once Vortex is the only feature store"
    );
}
