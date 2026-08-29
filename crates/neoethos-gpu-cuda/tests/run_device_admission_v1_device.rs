#![cfg(feature = "cuda")]

use neoethos_gpu_cuda::{
    SealedDiscoveryRunDeviceAdmissionV1, acquire_discovery_run_device_admission_v1,
};

#[test]
fn real_cuda_device_acquisition_is_native_and_every_probe_happens_exactly_once() {
    let admission = acquire_discovery_run_device_admission_v1()
        .expect("a visible compatible CUDA card must produce one sealed native admission");

    assert!(
        matches!(
            &admission,
            SealedDiscoveryRunDeviceAdmissionV1::NativeCuda(_)
        ),
        "a host with a visible CUDA card must never receive CPU admission"
    );

    let counters = admission.probe_counters();
    assert_eq!(counters.physical_inventory_probe_count(), 1);
    assert_eq!(counters.cuda_enumeration_count(), 1);
    assert_eq!(counters.primary_context_acquisition_count(), 1);
    assert_eq!(counters.run_stream_creation_count(), 1);
    assert_ne!(admission.admission_identity_sha256(), [0_u8; 32]);
}
