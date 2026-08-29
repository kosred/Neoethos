use neoethos_core::execution_budget::{
    AuxiliarySlotLimit, AuxiliarySlotRequest, CompositeAdmissionAuthority, CompositeAdmissionGrant,
    CompositeAdmissionRequest, CpuPermitBroker, CpuPermitRequest, WorkerLimit,
};
use std::sync::OnceLock;

// Rust's integration-test harness creates worker threads before invoking each
// `#[test]`. Linux source sealing must initialize earlier so those workers
// inherit the blocked signal set, exactly as production entrypoints do.
#[cfg(target_os = "linux")]
#[ctor::ctor]
fn initialize_linux_source_seal_before_test_harness() {
    neoethos_data::initialize_source_seal_before_runtime()
        .expect("initialize Linux source sealing before the integration-test harness");
}

pub fn import_grant() -> CompositeAdmissionGrant {
    let width = WorkerLimit::new(1).expect("one-worker import test budget");
    import_authority()
        .acquire(CompositeAdmissionRequest::new(
            CpuPermitRequest::local(width),
            AuxiliarySlotRequest::One,
        ))
        .expect("acquire import test resources")
}

fn import_authority() -> &'static CompositeAdmissionAuthority {
    static AUTHORITY: OnceLock<CompositeAdmissionAuthority> = OnceLock::new();
    AUTHORITY.get_or_init(|| {
        let slots = neoethos_data::source_seal_slot_limit();
        let worker_limit = WorkerLimit::new(slots).expect("positive import test worker limit");
        CompositeAdmissionAuthority::new(
            CpuPermitBroker::new(worker_limit),
            AuxiliarySlotLimit::new(slots).expect("positive source-seal slot limit"),
        )
    })
}
