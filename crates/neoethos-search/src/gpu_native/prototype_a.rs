//! Honest status and measured transfer evidence for the existing fused CubeCL baseline.

use crate::gpu_native::engine::{EngineCapabilities, EngineStatus, TransferSnapshot};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrototypeATelemetry {
    pub gpu_calls: u64,
    pub resident_cache_hits: u64,
    pub resident_cache_misses: u64,
    pub resident_upload_bytes: u64,
    pub streamed_dataset_upload_bytes: u64,
    pub gene_uploads: u64,
    pub gene_upload_bytes: u64,
    pub full_readbacks: u64,
    pub full_readback_bytes: u64,
    pub compact_readbacks: u64,
    pub compact_readback_bytes: u64,
    pub chained_reuploads: u64,
    pub synchronization_events: u64,
}

impl PrototypeATelemetry {
    pub fn transfer_snapshot(self) -> TransferSnapshot {
        TransferSnapshot {
            dataset_uploads: u64::from(self.resident_upload_bytes > 0),
            gene_uploads: self.gene_uploads,
            scenario_uploads: 0,
            full_d2h_readbacks: self.full_readbacks,
            compact_d2h_readbacks: self.compact_readbacks,
            chained_reuploads: self.chained_reuploads,
            synchronization_events: self.synchronization_events,
            h2d_bytes: self
                .resident_upload_bytes
                .saturating_add(self.streamed_dataset_upload_bytes)
                .saturating_add(self.gene_upload_bytes),
            d2h_bytes: self
                .full_readback_bytes
                .saturating_add(self.compact_readback_bytes),
        }
    }

    pub fn satisfies_no_dense_roundtrip(self) -> bool {
        self.full_readbacks == 0 && self.chained_reuploads == 0
    }
}

pub fn prototype_a_status() -> EngineStatus {
    #[cfg(feature = "gpu")]
    {
        EngineStatus::NotBenchmarked
    }
    #[cfg(not(feature = "gpu"))]
    {
        EngineStatus::UnsupportedCapability
    }
}

pub fn prototype_a_capabilities() -> EngineCapabilities {
    EngineCapabilities {
        fixed_stops: true,
        adaptive_stops: true,
        break_even: true,
        trailing: true,
        prop_firm_state: true,
        device_filtering: false,
        compact_readback: true,
    }
}

pub fn reset_prototype_a_telemetry() {
    #[cfg(feature = "gpu")]
    crate::cubecl_eval::reset_cubecl_transfer_telemetry();
}

pub fn prototype_a_telemetry() -> PrototypeATelemetry {
    #[cfg(feature = "gpu")]
    {
        let raw = crate::cubecl_eval::cubecl_transfer_telemetry_snapshot();
        PrototypeATelemetry {
            gpu_calls: raw.gpu_calls,
            resident_cache_hits: raw.resident_cache_hits,
            resident_cache_misses: raw.resident_cache_misses,
            resident_upload_bytes: raw.resident_upload_bytes,
            streamed_dataset_upload_bytes: raw.streamed_dataset_upload_bytes,
            gene_uploads: raw.gene_uploads,
            gene_upload_bytes: raw.gene_upload_bytes,
            full_readbacks: raw.full_readbacks,
            full_readback_bytes: raw.full_readback_bytes,
            compact_readbacks: raw.compact_readbacks,
            compact_readback_bytes: raw.compact_readback_bytes,
            chained_reuploads: raw.chained_reuploads,
            synchronization_events: raw.synchronization_events,
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        PrototypeATelemetry::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_mapping_never_fabricates_scenario_uploads() {
        let telemetry = PrototypeATelemetry {
            resident_upload_bytes: 100,
            gene_uploads: 2,
            gene_upload_bytes: 50,
            compact_readbacks: 2,
            compact_readback_bytes: 20,
            ..PrototypeATelemetry::default()
        };
        let transfers = telemetry.transfer_snapshot();
        assert_eq!(transfers.dataset_uploads, 1);
        assert_eq!(transfers.scenario_uploads, 0);
        assert_eq!(transfers.h2d_bytes, 150);
        assert_eq!(transfers.d2h_bytes, 20);
    }

    #[test]
    fn dense_roundtrip_is_an_explicit_acceptance_failure() {
        let telemetry = PrototypeATelemetry {
            full_readbacks: 1,
            chained_reuploads: 1,
            ..PrototypeATelemetry::default()
        };
        assert!(!telemetry.satisfies_no_dense_roundtrip());
    }
}
