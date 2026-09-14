//! Explicitly test-only HIP population bridge, not canonical Data admission.
//!
//! The fixture uploads and hashes its actual nine input arrays, then borrows
//! their real same-lease keys. These are diagnostic content/request bindings:
//! they are not the production feature recipe, admission seal, or Search proof.
//! All evaluator math, CSR checks, scenario identities, stable boxed tokens and
//! terminal metric validation remain in the existing native/shared path.

use super::*;
use crate::hip_runtime_v1::{
    HipDeviceBufferV1, HipRunLeaseV1, HipRuntimeErrorV1, HipRuntimeIdentityV1,
    hip_native_build_manifest_v1,
};

const HIP_BACKEND_V1: u32 = 2;

#[repr(C)]
struct RawHipResidentFeatureStoreV1 {
    abi_version: u32,
    backend_kind: u32,
    lease_id: u64,
    row_count: u64,
    feature_count: u32,
    smc_slots: u32,
    buffer_keys: [u64; 9],
    allocator_reserve_bytes: u64,
    admission_identity_sha256: [u8; 32],
    canonical_content_merkle: [u8; 32],
    run_stream_process_token: [u8; 32],
}

unsafe extern "C" {
    fn neoethos_hip_population_bind_resident_feature_store_v1(
        parent: *const RawHipResidentFeatureStoreV1,
        status: *mut i32,
    ) -> *mut c_void;
    fn neoethos_hip_native_build_manifest_sha256_v1() -> *const u8;
}

#[derive(Debug, Error)]
enum HipPopulationErrorV1 {
    #[error(transparent)]
    Runtime(#[from] HipRuntimeErrorV1),
    #[error("invalid HIP diagnostic population input: {0}")]
    Invalid(String),
    #[error("native HIP population {operation} failed with status {status}; owner quarantined")]
    Native {
        operation: &'static str,
        status: i32,
    },
}

impl From<CudaPopulationError> for HipPopulationErrorV1 {
    fn from(error: CudaPopulationError) -> Self {
        match error {
            CudaPopulationError::InvalidInput(detail) => Self::Invalid(detail),
            CudaPopulationError::Native {
                operation, status, ..
            } => Self::Native { operation, status },
            CudaPopulationError::RuntimeUnavailable => Self::Native {
                operation: "shared_population_runtime",
                status: STATUS_UNSUPPORTED,
            },
            CudaPopulationError::AsyncFreeOutcomeUnknownDeliberateLeak { operation } => {
                Self::Native {
                    operation,
                    status: STATUS_ASYNC_FREE_OUTCOME_UNKNOWN,
                }
            }
            CudaPopulationError::AsyncAllocationOutcomeUnknownDeliberateLeak { operation } => {
                Self::Native {
                    operation,
                    status: STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN,
                }
            }
        }
    }
}

fn invalid_hip(detail: &str) -> HipPopulationErrorV1 {
    HipPopulationErrorV1::Invalid(detail.to_owned())
}

fn parent_extents(rows: usize, features: usize) -> Result<[usize; 9], HipPopulationErrorV1> {
    if rows == 0 || rows > i32::MAX as usize || features == 0 || features > i32::MAX as usize {
        return Err(invalid_hip(
            "nonzero native signed parent dimensions required",
        ));
    }
    let overflow = || invalid_hip("parent extent overflow");
    let lane = rows.checked_mul(8).ok_or_else(overflow)?;
    let cells = rows.checked_mul(features).ok_or_else(overflow)?;
    let values = cells.checked_mul(8).ok_or_else(overflow)?;
    let validity = (cells / 2 + cells % 2)
        .checked_add(3)
        .ok_or_else(overflow)?
        / 4
        * 4;
    let smc = rows.checked_mul(SMC_SLOTS).ok_or_else(overflow)?;
    Ok([lane, lane, lane, values, validity, lane, lane, lane, smc])
}

fn fixture_content_hash(bytes: &[&[u8]; 9], rows: usize, features: usize) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"neoethos.hip-population.diagnostic-input-bytes.v1");
    hash.update((rows as u64).to_le_bytes());
    hash.update((features as u64).to_le_bytes());
    for (slot, data) in bytes.iter().enumerate() {
        hash.update((slot as u64).to_le_bytes());
        hash.update((data.len() as u64).to_le_bytes());
        hash.update(data);
    }
    hash.finalize().into()
}

// No caller-created key or supplied hash constructor. This diagnostic carrier
// hashes exactly the bytes it submits to the real runtime-owned upload path.
struct HipPopulationDatasetFixtureV1<'lease> {
    lease: &'lease HipRunLeaseV1,
    buffers: [HipDeviceBufferV1<'lease>; 9],
    rows: usize,
    features: usize,
    content_hash: [u8; 32],
}

impl<'lease> HipPopulationDatasetFixtureV1<'lease> {
    fn upload(
        lease: &'lease HipRunLeaseV1,
        rows: usize,
        features: usize,
        bytes: [&[u8]; 9],
    ) -> Result<Self, HipPopulationErrorV1> {
        let sizes = parent_extents(rows, features)?;
        if bytes
            .iter()
            .zip(sizes)
            .any(|(bytes, size)| bytes.len() != size)
        {
            return Err(invalid_hip(
                "all nine fixture arrays must have their exact physical sizes",
            ));
        }
        let content_hash = fixture_content_hash(&bytes, rows, features);
        let mut buffers = Vec::with_capacity(9);
        for data in bytes {
            buffers.push(lease.upload_bytes(data)?);
        }
        let buffers = buffers
            .try_into()
            .map_err(|_| invalid_hip("lost fixture array slot"))?;
        Ok(Self {
            lease,
            buffers,
            rows,
            features,
            content_hash,
        })
    }
}

struct HipPopulationSessionV1<'data, 'lease> {
    core: PopulationSession,
    // Keep both the owner and all buffer objects alive through checked delete.
    parent: &'data HipPopulationDatasetFixtureV1<'lease>,
    identity: HipRuntimeIdentityV1,
    diagnostic_request_hash: [u8; 32],
}

// Deliberately private: the generic host receipt never escapes as CUDA or as a
// production HIP admission receipt. This wrapper carries genuine HIP identity.
struct HipPopulationDiagnosticResultV1 {
    identity: HipRuntimeIdentityV1,
    diagnostic_request_hash: [u8; 32],
    metrics: HostPopulationMetricsReceiptV1,
}

impl<'data, 'lease> HipPopulationSessionV1<'data, 'lease> {
    fn bind_fixture(
        parent: &'data HipPopulationDatasetFixtureV1<'lease>,
        allocator_reserve_bytes: u64,
    ) -> Result<Self, HipPopulationErrorV1> {
        if allocator_reserve_bytes == 0 {
            return Err(invalid_hip("allocator reserve must be nonzero"));
        }
        let sizes = parent_extents(parent.rows, parent.features)?;
        let keys = parent
            .lease
            .checked_population_keys_v1(std::array::from_fn(|i| &parent.buffers[i]), sizes)?;
        if hip_native_build_manifest_v1().is_none() {
            return Err(invalid_hip("missing actual HIP native build manifest"));
        }
        // SAFETY: native exposes a static 32-byte build digest, not a device
        // pointer. It belongs to the linked HIP archive, never the CUDA build.
        let build_ptr = unsafe { neoethos_hip_native_build_manifest_sha256_v1() };
        if build_ptr.is_null() {
            return Err(invalid_hip("null HIP build digest"));
        }
        let mut build_hash = [0; 32];
        unsafe { std::ptr::copy_nonoverlapping(build_ptr, build_hash.as_mut_ptr(), 32) };
        if build_hash == [0; 32] {
            return Err(invalid_hip("empty HIP build digest"));
        }
        let identity = parent.lease.identity().clone();
        let mut hash = Sha256::new();
        hash.update(b"neoethos.hip-population.diagnostic-owned-request.v1");
        hash.update(identity.lease_id().to_le_bytes());
        hash.update(identity.device_ordinal().to_le_bytes());
        hash.update(identity.device_uuid());
        hash.update(identity.stream_id().to_le_bytes());
        hash.update(identity.runtime_version().to_le_bytes());
        hash.update(identity.driver_version().to_le_bytes());
        hash.update(identity.architecture().as_bytes());
        hash.update(build_hash);
        hash.update(parent.content_hash);
        hash.update(allocator_reserve_bytes.to_le_bytes());
        for (key, size) in keys.iter().zip(sizes) {
            hash.update(key.to_le_bytes());
            hash.update((size as u64).to_le_bytes());
        }
        let request: [u8; 32] = hash.finalize().into();
        let mut process = Sha256::new();
        process.update(b"neoethos.hip-population.diagnostic-process-binding.v1");
        process.update(request);
        let process: [u8; 32] = process.finalize().into();
        let raw = RawHipResidentFeatureStoreV1 {
            abi_version: 1,
            backend_kind: HIP_BACKEND_V1,
            lease_id: identity.lease_id(),
            row_count: parent.rows as u64,
            feature_count: parent.features as u32,
            smc_slots: SMC_SLOTS as u32,
            buffer_keys: keys,
            allocator_reserve_bytes,
            // ABI names are fixed; this test-only producer explicitly does not
            // claim these diagnostic bindings are canonical Data admission.
            admission_identity_sha256: request,
            canonical_content_merkle: parent.content_hash,
            run_stream_process_token: process,
        };
        let mut status = STATUS_OK;
        // SAFETY: all keys came from live exact-size same-owner buffers. Native
        // independently pins and validates them; the parent is retained below.
        let handle =
            unsafe { neoethos_hip_population_bind_resident_feature_store_v1(&raw, &mut status) };
        if handle.is_null() || status != STATUS_OK {
            parent.lease.quarantine_after_population_failure_v1();
            return Err(HipPopulationErrorV1::Native {
                operation: "bind_fixture",
                status: if status == STATUS_OK {
                    STATUS_ABI_MISMATCH
                } else {
                    status
                },
            });
        }
        let mut core = PopulationSession::detached_resident_v3();
        core.handle = handle;
        core.device = identity.device_ordinal() as i32;
        core.feature_count = parent.features;
        core.dataset_uploaded = true;
        core.resident_parent_shape_v3 = Some((parent.rows, parent.features));
        core.strict_resident_state = StrictResidentSessionStateV1::StrictIdle;
        core.resident_session_identity_sha256 = Some(request);
        core.native_build_identity_sha256 = Some(build_hash);
        Ok(Self {
            core,
            parent,
            identity,
            diagnostic_request_hash: request,
        })
    }

    fn evaluate(
        &mut self,
        view: PopulationEvaluationViewV1,
        genes: PopulationGeneView<'_>,
        scenarios: &[ScenarioDescriptor],
        settings: &NeoPopulationSettings,
    ) -> Result<HipPopulationDiagnosticResultV1, HipPopulationErrorV1> {
        // Host validation happens before any replaced device authority. The
        // resident parent never accepts an uploaded host adaptive-base series.
        if view.parent_row_count != self.parent.rows || view.adaptive_base_pips.is_some() {
            return Err(invalid_hip(
                "exact resident parent required; host adaptive base is forbidden",
            ));
        }
        let population = genes.validate(self.parent.features)?;
        if scenarios.is_empty()
            || scenarios
                .iter()
                .any(|s| s.base_candidate_id >= population as u64)
        {
            return Err(invalid_hip("scenario must name an uploaded gene"));
        }
        PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(
            scenarios.len(),
            settings.month_capacity,
        )?;
        let result = (|| {
            self.core.bind_evaluation_view_v1(view)?;
            self.core.upload_genes(genes)?;
            self.core.upload_scenarios(scenarios)?;
            self.core
                .enqueue_metrics_only_v1(settings)?
                .consume_host_metrics_v1()
        })();
        match result {
            Ok(metrics) => Ok(HipPopulationDiagnosticResultV1 {
                identity: self.identity.clone(),
                diagnostic_request_hash: self.diagnostic_request_hash,
                metrics,
            }),
            Err(error) => {
                // Includes malformed terminal proof: accepted DMA/work may
                // already exist. Do not reuse the lease or retry native frees.
                self.core.strict_resident_state = StrictResidentSessionStateV1::Poisoned;
                self.parent.lease.quarantine_after_population_failure_v1();
                Err(error.into())
            }
        }
    }

    fn close(&mut self) -> Result<(), HipPopulationErrorV1> {
        if self.core.handle.is_null() {
            return Ok(());
        }
        let handle = std::mem::replace(&mut self.core.handle, std::ptr::null_mut());
        if self.core.strict_resident_state != StrictResidentSessionStateV1::StrictIdle {
            self.parent.lease.quarantine_after_population_failure_v1();
            return Err(HipPopulationErrorV1::Native {
                operation: "close",
                status: STATUS_STRICT_RESIDENT_POISONED,
            });
        }
        // SAFETY: this unique child owns the native handle. The HIP branch
        // proves its lease, synchronizes its owned stream, checks every free,
        // and only then releases borrower pins. Ambiguity retains a tombstone.
        let status = unsafe { neoethos_gpu_cuda_population_destroy_terminal_checked_v2(handle) };
        if status != STATUS_OK {
            self.parent.lease.quarantine_after_population_failure_v1();
            return Err(HipPopulationErrorV1::Native {
                operation: "close",
                status,
            });
        }
        Ok(())
    }
}

impl Drop for HipPopulationSessionV1<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            eprintln!("HIP diagnostic child retained: {error}");
        }
    }
}

#[test]
fn hip_population_bind_abi_and_checked_parent_extents() {
    use std::mem::{align_of, offset_of, size_of};
    assert_eq!(size_of::<RawHipResidentFeatureStoreV1>(), 208);
    assert_eq!(align_of::<RawHipResidentFeatureStoreV1>(), 8);
    assert_eq!(offset_of!(RawHipResidentFeatureStoreV1, lease_id), 8);
    assert_eq!(offset_of!(RawHipResidentFeatureStoreV1, buffer_keys), 32);
    assert_eq!(
        offset_of!(RawHipResidentFeatureStoreV1, allocator_reserve_bytes),
        104
    );
    assert_eq!(
        offset_of!(RawHipResidentFeatureStoreV1, admission_identity_sha256),
        112
    );
    assert_eq!(
        offset_of!(RawHipResidentFeatureStoreV1, canonical_content_merkle),
        144
    );
    assert_eq!(
        offset_of!(RawHipResidentFeatureStoreV1, run_stream_process_token),
        176
    );
    assert_eq!(
        parent_extents(3, 3).unwrap(),
        [24, 24, 24, 72, 8, 24, 24, 24, 33]
    );
    assert_eq!(
        parent_extents(8, 1).unwrap(),
        [64, 64, 64, 64, 4, 64, 64, 64, 88]
    );
    for (rows, features) in [
        (0, 1),
        (1, 0),
        (usize::MAX, 1),
        (1, usize::MAX),
        (i32::MAX as usize, i32::MAX as usize),
    ] {
        assert!(parent_extents(rows, features).is_err());
    }
    let one = [1u8];
    let two = [2u8];
    let mut bytes = [&one[..]; 9];
    let original = fixture_content_hash(&bytes, 1, 1);
    for slot in 0..9 {
        bytes[slot] = &two;
        assert_ne!(fixture_content_hash(&bytes, 1, 1), original);
        bytes[slot] = &one;
    }
    assert_ne!(fixture_content_hash(&bytes, 2, 1), original);
    assert_ne!(fixture_content_hash(&bytes, 1, 2), original);
}

#[test]
fn hip_population_actual_owned_buffers_strict_metrics_and_cleanup() {
    assert_eq!(
        std::env::var("NEOETHOS_REQUIRE_GPU").as_deref(),
        Ok("1"),
        "mandatory device fixture; no driver-less success or silent skip"
    );
    let ordinal = std::env::var("NEOETHOS_HIP_DEVICE")
        .map(|value| value.parse::<u32>().expect("valid HIP ordinal"))
        .unwrap_or(0);
    let lease = HipRunLeaseV1::acquire(ordinal).expect("actual HIP device lease");
    const ROWS: usize = 8;
    let f64_bytes = |value: f64| value.to_ne_bytes().repeat(ROWS);
    let price = f64_bytes(1.0);
    let feature = f64_bytes(0.0);
    let calendar = 1i64.to_ne_bytes().repeat(ROWS);
    let timestamps = (0..ROWS)
        .flat_map(|i| (1_700_000_000_000i64 + i as i64 * 60_000).to_ne_bytes())
        .collect::<Vec<_>>();
    let validity = [0u8; 4];
    let smc = [0u8; ROWS * SMC_SLOTS];
    let data = HipPopulationDatasetFixtureV1::upload(
        &lease,
        ROWS,
        1,
        [
            &price,
            &price,
            &price,
            &feature,
            &validity,
            &calendar,
            &calendar,
            &timestamps,
            &smc,
        ],
    )
    .expect("actual nine same-lease uploads");
    assert!(matches!(
        lease.checked_population_keys_v1([&data.buffers[0]], [1]),
        Err(HipRuntimeErrorV1::InvalidInput(_))
    ));
    let foreign = HipRunLeaseV1::acquire(ordinal).expect("independent real lease");
    assert!(matches!(
        foreign.checked_population_keys_v1([&data.buffers[0]], [64]),
        Err(HipRuntimeErrorV1::InvalidInput(_))
    ));
    foreign.try_close().expect("close independent lease");
    let mut child =
        HipPopulationSessionV1::bind_fixture(&data, 1 << 20).expect("real HIP native bind");
    let descriptors = [
        GeneDescriptor {
            candidate_id: 501,
            term_count: 1,
            long_threshold: 1.0,
            short_threshold: -1.0,
            ..GeneDescriptor::default()
        },
        GeneDescriptor {
            candidate_id: 602,
            term_offset: 1,
            term_count: 1,
            long_threshold: -0.5,
            short_threshold: -1.0,
            ..GeneDescriptor::default()
        },
    ];
    let genes = PopulationGeneView {
        descriptors: &descriptors,
        offsets: &[0, 1, 2],
        indices: &[0, 0],
        weights: &[1.0, 1.0],
        stop_pips: &[10.0, 10.0],
        target_pips: &[10.0, 10.0],
        stop_vol_multipliers: &[0.0, 0.0],
        smc_flags: &[0; 2 * SMC_SLOTS],
        smc_weights: &[1.0; SMC_SLOTS],
        gate_threshold: 0.0,
        smc_gate_disabled: true,
    };
    let scenarios = [
        ScenarioDescriptor {
            base_candidate_id: 1,
            scenario_id: 9003,
            window_len: ROWS as u32,
            commission_micros: 2_000_000,
            ..ScenarioDescriptor::default()
        },
        ScenarioDescriptor {
            base_candidate_id: 0,
            scenario_id: 9001,
            window_len: ROWS as u32,
            ..ScenarioDescriptor::default()
        },
        ScenarioDescriptor {
            base_candidate_id: 1,
            scenario_id: 9002,
            window_len: ROWS as u32,
            ..ScenarioDescriptor::default()
        },
    ];
    let settings = NeoPopulationSettings {
        abi_version: ABI_VERSION,
        initial_equity: 100.0,
        pip_value: 0.01,
        pip_value_per_lot: 1.0,
        max_hold_bars: 1,
        min_hold_bars: 1,
        max_trades_per_day: 1,
        month_capacity: 2,
        gap_threshold_ms: 600_000,
        ..NeoPopulationSettings::default()
    };
    let view =
        PopulationEvaluationViewV1::full(ROWS, PopulationTimestampModeV1::Canonical, None).unwrap();
    let result = child
        .evaluate(view, genes, &scenarios, &settings)
        .expect("actual HIP metrics kernel");
    assert_eq!(result.identity, *lease.identity());
    assert_eq!(
        result.diagnostic_request_hash,
        child.diagnostic_request_hash
    );
    let rows = result.metrics.metric_rows();
    assert_eq!(rows.len(), 3);
    // Independent simple known answers: flat prices, no spread/swap, one
    // permitted daily entry; max-hold closes it next bar. Only the explicit
    // two-unit commission loses money. The neutral gene never crosses a gate.
    for (row, (candidate, scenario, net, trades)) in rows.iter().zip([
        (602, 9003, -2.0f64, 1.0f64),
        (501, 9001, 0.0, 0.0),
        (602, 9002, 0.0, 1.0),
    ]) {
        assert_eq!((row.candidate_id, row.scenario_id), (candidate, scenario));
        assert_eq!(row.values[0].to_bits(), net.to_bits());
        assert_eq!(row.values[8].to_bits(), trades.to_bits());
        assert_eq!(row.values[2].to_bits(), 100.0f64.to_bits());
        assert!(row.values.iter().all(|value| value.is_finite()));
    }
    assert_eq!(result.metrics.terminal_synchronization_count(), 1);
    assert_eq!(result.metrics.terminal_readback_count(), 1);
    assert_eq!(result.metrics.terminal_readback_bytes(), 3 * 104);
    assert_eq!(result.metrics.counters().dataset_upload_bytes, 0);
    assert!(result.metrics.counters().gene_upload_bytes > 0);
    assert!(result.metrics.counters().scenario_upload_bytes > 0);
    child.close().expect("terminal checked child cleanup");
    child.close().expect("no double native cleanup");
    drop(child);
    drop(data);
    lease
        .try_close()
        .expect("same lease closes after child and buffer owners");
}
