use std::{fs, path::PathBuf};

fn crate_file(path: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(root.join(path)).expect("read production source")
}

#[test]
fn population_allocation_requires_post_free_stream_retirement_authority() {
    let source = crate_file("src/resident_feature_store_v3.rs");
    assert!(source.contains("struct PendingDataTransientRetirementV1"));
    assert!(source.contains("struct SealedDataTransientRetirementV1"));
    assert!(source.contains("cuStreamSynchronize(Data transient retirement)"));
    assert!(source.contains("bind_population_after_data_transient_retirement_v1"));
    assert!(source.contains("data_transient_retirement_process_token"));

    let release = source
        .find("name_offsets.release_async(stream)?")
        .expect("explicit fallible async free");
    let record = source
        .find("retirement_event.record(&self.producer_stream)?")
        .expect("post-free retirement event");
    assert!(
        release < record,
        "retirement event must follow every queued free"
    );
}

#[test]
fn materializer_rejects_a_plan_prepared_from_a_different_data_extent() {
    let contracts = crate_file("../neoethos-gpu-contracts/src/resident_feature_store_v3.rs");
    assert!(contracts.contains("neoethos.resident-working-set-extent.v3"));
    assert!(contracts.contains("pub fn identity_sha256(&self)"));

    let data = crate_file("../neoethos-data/src/core/gpu_resident_feature_store_v3.rs");
    assert!(data.contains("prepared.workspace_extent.identity_sha256()"));
    assert!(data.contains("limits.data_extent_identity_sha256()"));
    assert!(data.contains(
        "prepared Data recipe extent does not match the exact Data+population stage plan"
    ));
}
