//! Strict native failure routing for a sealed CUDA ordinal.
//!
//! This module deliberately has no CPU action. Allocation pressure may shrink
//! the physical batch on the same selected ordinal; every other failure, and an
//! exhausted shrink sequence, terminates the work unit.

use crate::strict_discovery_device_route_v1::{
    StrictNativeFailureActionV1, StrictNativeFailureKindV1, decide_strict_native_failure_v1,
};

pub(crate) fn decide_strict_population_failure_v1(
    failure: StrictNativeFailureKindV1,
    selected_ordinal: u32,
    batch_size: usize,
    retry_index: u32,
    max_retries: u32,
) -> StrictNativeFailureActionV1 {
    match decide_strict_native_failure_v1(
        failure,
        selected_ordinal,
        batch_size,
        retry_index,
        max_retries,
    ) {
        StrictNativeFailureActionV1::RetrySameOrdinal {
            selected_ordinal,
            next_batch_size,
        } => StrictNativeFailureActionV1::RetrySameOrdinal {
            selected_ordinal,
            next_batch_size,
        },
        StrictNativeFailureActionV1::FailLoud {
            selected_ordinal,
            failure,
        } => StrictNativeFailureActionV1::FailLoud {
            selected_ordinal,
            failure,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_pressure_rebatches_only_on_the_same_ordinal() {
        assert_eq!(
            decide_strict_population_failure_v1(
                StrictNativeFailureKindV1::AllocationPressure,
                7,
                101,
                0,
                3,
            ),
            StrictNativeFailureActionV1::RetrySameOrdinal {
                selected_ordinal: 7,
                next_batch_size: 51,
            }
        );
    }

    #[test]
    fn non_allocation_faults_and_exhaustion_fail_loud() {
        for failure in [
            StrictNativeFailureKindV1::DeviceLost,
            StrictNativeFailureKindV1::Unsupported,
            StrictNativeFailureKindV1::WrongShape,
        ] {
            assert_eq!(
                decide_strict_population_failure_v1(failure, 2, 128, 0, 4),
                StrictNativeFailureActionV1::FailLoud {
                    selected_ordinal: 2,
                    failure,
                }
            );
        }
        assert_eq!(
            decide_strict_population_failure_v1(
                StrictNativeFailureKindV1::AllocationPressure,
                2,
                4,
                4,
                4,
            ),
            StrictNativeFailureActionV1::FailLoud {
                selected_ordinal: 2,
                failure: StrictNativeFailureKindV1::AllocationPressure,
            }
        );
    }
}
