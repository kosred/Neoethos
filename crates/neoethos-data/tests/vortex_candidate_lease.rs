use neoethos_data::core::dataset_candidate_lease::{
    DatasetCandidateLease, collect_orphan_candidates,
};
use tempfile::tempdir;

#[test]
fn candidate_gc_uses_os_lock_liveness_not_age_or_pid() {
    let root = tempdir().expect("temporary root");
    let candidate = root.path().join("candidate-test.vortex");
    std::fs::write(&candidate, b"in progress").expect("candidate fixture");
    let lease = DatasetCandidateLease::acquire(&candidate).expect("candidate writer lease");

    collect_orphan_candidates(root.path()).expect("gc while writer is live");
    assert!(candidate.exists());

    drop(lease);
    collect_orphan_candidates(root.path()).expect("gc after writer release");
    assert!(!candidate.exists());
}
