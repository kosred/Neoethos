use crate::strict_discovery_device_route_v1::{
    CudaOrdinalProbeOutcomeV1, NoCompatibleGpuReasonV1, StrictCudaProbeFailureKindV1,
    StrictDiscoveryProbeObservationV1, StrictNativeFailureActionV1, StrictNativeFailureKindV1,
    UnsealedStrictDiscoveryDeviceRouteV1, classify_strict_discovery_probe_observation_v1,
    decide_strict_native_failure_v1,
};

fn observation(
    native_adapter_compiled: bool,
    runtime_loaded: bool,
    reported_device_count: u32,
    ordinal_outcomes: Vec<CudaOrdinalProbeOutcomeV1>,
) -> StrictDiscoveryProbeObservationV1 {
    StrictDiscoveryProbeObservationV1 {
        native_adapter_compiled,
        runtime_loaded,
        reported_device_count,
        ordinal_outcomes,
    }
}

#[test]
fn no_visible_cuda_ordinal_classifies_cpu_without_minting_authority() {
    let classified =
        classify_strict_discovery_probe_observation_v1(&observation(true, true, 0, vec![])).expect(
            "a complete real native probe with no visible ordinal has a CPU classification",
        );

    assert_eq!(
        classified,
        UnsealedStrictDiscoveryDeviceRouteV1::CpuNoCompatibleGpu {
            reason: NoCompatibleGpuReasonV1::NoVisibleCudaOrdinal,
        }
    );
}

#[test]
fn missing_native_adapter_cannot_claim_that_no_gpu_exists() {
    let error =
        classify_strict_discovery_probe_observation_v1(&observation(false, false, 0, vec![]))
            .expect_err("a build without the native probe cannot prove physical GPU absence");
    assert_eq!(error.code(), "native_adapter_not_compiled");
}

#[test]
fn unavailable_runtime_cannot_claim_that_no_gpu_exists() {
    let error =
        classify_strict_discovery_probe_observation_v1(&observation(true, false, 0, vec![]))
            .expect_err("a driver/runtime fault cannot be transformed into CPU authority");
    assert_eq!(error.code(), "cuda_runtime_unavailable");
}

#[test]
fn visible_but_build_incompatible_ordinals_fail_loud() {
    let error = classify_strict_discovery_probe_observation_v1(&observation(
        true,
        true,
        2,
        vec![
            CudaOrdinalProbeOutcomeV1::BuildIncompatible { ordinal: 0 },
            CudaOrdinalProbeOutcomeV1::BuildIncompatible { ordinal: 1 },
        ],
    ))
    .expect_err("a visible GPU without matching SASS requires a corrected GPU build, not CPU");
    assert_eq!(error.code(), "visible_gpu_build_incompatible");
}

#[test]
fn complete_probe_selects_the_lowest_exact_compatible_ordinal_deterministically() {
    let classified = classify_strict_discovery_probe_observation_v1(&observation(
        true,
        true,
        3,
        vec![
            CudaOrdinalProbeOutcomeV1::Compatible { ordinal: 2 },
            CudaOrdinalProbeOutcomeV1::BuildIncompatible { ordinal: 1 },
            CudaOrdinalProbeOutcomeV1::Compatible { ordinal: 0 },
        ],
    ))
    .expect("a complete probe with compatible native CUDA devices must select one");

    assert_eq!(
        classified,
        UnsealedStrictDiscoveryDeviceRouteV1::NativeCuda {
            selected_ordinal: 0,
        }
    );
}

#[test]
fn incomplete_or_duplicate_ordinal_enumeration_fails_closed() {
    for invalid in [
        observation(
            true,
            true,
            2,
            vec![CudaOrdinalProbeOutcomeV1::Compatible { ordinal: 0 }],
        ),
        observation(
            true,
            true,
            2,
            vec![
                CudaOrdinalProbeOutcomeV1::Compatible { ordinal: 0 },
                CudaOrdinalProbeOutcomeV1::BuildIncompatible { ordinal: 0 },
            ],
        ),
    ] {
        let error = classify_strict_discovery_probe_observation_v1(&invalid)
            .expect_err("partial or duplicated device evidence cannot authorize GPU or CPU");
        assert_eq!(error.code(), "incomplete_cuda_probe");
    }
}

#[test]
fn any_ordinal_probe_fault_refuses_cpu_even_when_another_ordinal_is_compatible() {
    for failure in [
        StrictCudaProbeFailureKindV1::DeviceLost,
        StrictCudaProbeFailureKindV1::UnsupportedRuntime,
        StrictCudaProbeFailureKindV1::UnreadableDeviceIdentity,
    ] {
        let probe = observation(
            true,
            true,
            2,
            vec![
                CudaOrdinalProbeOutcomeV1::Compatible { ordinal: 0 },
                CudaOrdinalProbeOutcomeV1::Fault {
                    ordinal: 1,
                    failure,
                },
            ],
        );
        let error = classify_strict_discovery_probe_observation_v1(&probe)
            .expect_err("a probe fault is not evidence that no compatible GPU exists");
        assert_eq!(error.code(), "cuda_probe_fault");
    }
}

#[test]
fn allocation_pressure_rebatches_only_on_the_selected_ordinal() {
    let action = decide_strict_native_failure_v1(
        StrictNativeFailureKindV1::AllocationPressure,
        3,
        256,
        0,
        4,
    );
    assert_eq!(
        action,
        StrictNativeFailureActionV1::RetrySameOrdinal {
            selected_ordinal: 3,
            next_batch_size: 128,
        }
    );
}

#[test]
fn native_faults_and_exhausted_rebatch_fail_loud_without_cpu_or_another_card() {
    for failure in [
        StrictNativeFailureKindV1::DeviceLost,
        StrictNativeFailureKindV1::Unsupported,
        StrictNativeFailureKindV1::WrongShape,
    ] {
        assert_eq!(
            decide_strict_native_failure_v1(failure, 4, 256, 0, 4),
            StrictNativeFailureActionV1::FailLoud {
                selected_ordinal: 4,
                failure,
            }
        );
    }

    assert_eq!(
        decide_strict_native_failure_v1(StrictNativeFailureKindV1::AllocationPressure, 4, 4, 3, 4,),
        StrictNativeFailureActionV1::FailLoud {
            selected_ordinal: 4,
            failure: StrictNativeFailureKindV1::AllocationPressure,
        }
    );
}
