//! Data's HIP Session-v2 producer, using the original native mathematical kernel.
//!
//! This is one feature family, not the ten-family resident-store capability or
//! Search admission. It never creates a CUDA identity, loads a CUDA cubin, or
//! computes feature values on the CPU. Input validation matches the existing
//! canonical resident Session-v2 contract, including its retained dual clock.
//! The prepared input borrows immutable OHLCV; the result retains all device
//! inputs and outputs on the caller's genuine HIP lease. No feature D2H occurs.

use anyhow::{Context as _, Result, ensure};
use neoethos_gpu_cuda::hip_runtime_v1::{
    HipDeviceBufferV1, HipRunLeaseV1, HipRuntimeIdentityV1, HipRuntimeMemorySnapshotV1,
    hip_session_build_manifest_v1,
};
#[cfg(test)]
use sha2::{Digest, Sha256};

use crate::Ohlcv;

use super::gpu_hip_ohlcv_v1::{
    CheckedHipProducerBuildV1, PreparedHipOhlcvV1, ResidentHipOhlcvV1, checked_producer_build,
    close_buffers,
};
#[cfg(test)]
use super::gpu_hip_ohlcv_v1::{f64_bytes, i64_bytes};
#[cfg(test)]
use super::timestamps::validate_canonical_millisecond_timestamps;

pub const HIP_SESSION_IDENTITY_SCHEMA_V1: &str = "neoethos.data.hip-session-producer.v1";
pub const HIP_SESSION_COLUMN_NAMES_V1: [&str; 23] = [
    "session_london_open_dist",
    "session_london_high_dist",
    "session_london_low_dist",
    "session_london_range",
    "session_london_vwap_dist",
    "session_ny_open_dist",
    "session_ny_high_dist",
    "session_ny_low_dist",
    "session_ny_range",
    "session_ny_vwap_dist",
    "session_asian_open_dist",
    "session_asian_close_dist",
    "session_asian_range_norm",
    "session_london_ny_overlap",
    "session_vol_ratio",
    "session_prev_close_dist",
    "session_open_gap",
    "daily_range_pct",
    "daily_body_pct",
    "daily_position",
    "daily_high_dist",
    "daily_low_dist",
    "daily_vwap_dist",
];

const SESSION_SOURCE: &[u8] =
    include_bytes!("../../../neoethos-gpu-cuda/native/resident_session_v2.cu");
const SESSION_ABI: &[u8] =
    include_bytes!("../../../neoethos-gpu-cuda/native/resident_session_v2_abi.cuh");
#[cfg(test)]
const SESSION_PRECISION: &str = "f64-no-fast-math-no-contract-ieee-denormals";

/// Logical allocation extents, not allocator overhead or a memory reservation.
/// Uploads additionally require `parent_bytes` of host staging payload if all six
/// copies are pending; host/device allocator overhead is not measured here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HipSessionMemoryPlanV1 {
    rows: usize,
    lane_bytes: usize,
    parent_bytes: usize,
    value_bytes: usize,
    validity_bytes: usize,
    total_device_bytes: usize,
}

impl HipSessionMemoryPlanV1 {
    fn checked(rows: usize) -> Result<Self> {
        ensure!(rows > 0, "HIP Session requires at least one row");
        let lane_bytes = rows.checked_mul(8).context("HIP Session lane overflow")?;
        let parent_bytes = lane_bytes.checked_mul(6).context("HIP parent overflow")?;
        let validity_bytes = rows
            .checked_mul(HIP_SESSION_COLUMN_NAMES_V1.len())
            .context("HIP Session cell-count overflow")?;
        let value_bytes = validity_bytes
            .checked_mul(8)
            .context("HIP Session value-byte overflow")?;
        let total_device_bytes = parent_bytes
            .checked_add(value_bytes)
            .and_then(|bytes| bytes.checked_add(validity_bytes))
            .context("HIP Session total-byte overflow")?;
        ensure!(
            lane_bytes <= isize::MAX as usize && value_bytes <= isize::MAX as usize,
            "HIP Session exceeds addressable Rust slice extent"
        );
        Ok(Self {
            rows,
            lane_bytes,
            parent_bytes,
            value_bytes,
            validity_bytes,
            total_device_bytes,
        })
    }

    pub const fn row_count(&self) -> usize {
        self.rows
    }
    pub const fn parent_lane_bytes(&self) -> usize {
        self.lane_bytes
    }
    pub const fn parent_bytes(&self) -> usize {
        self.parent_bytes
    }
    pub const fn value_bytes(&self) -> usize {
        self.value_bytes
    }
    pub const fn validity_bytes(&self) -> usize {
        self.validity_bytes
    }
    pub const fn total_device_bytes(&self) -> usize {
        self.total_device_bytes
    }

    fn check_available_bytes(&self, free_bytes: u64) -> Result<()> {
        ensure!(
            u64::try_from(self.total_device_bytes).context("HIP byte extent exceeds u64")?
                <= free_bytes,
            "HIP Session logical allocation needs {} bytes; live free-memory snapshot is {free_bytes}",
            self.total_device_bytes
        );
        Ok(())
    }
}

fn checked_manifest(text: &str, architecture: &str) -> Result<CheckedHipProducerBuildV1> {
    checked_producer_build(
        text,
        architecture,
        "neoethos.hip-session-kernels-build.v1",
        2,
        SESSION_SOURCE,
        SESSION_ABI,
    )
}

/// Explicit HIP execution identity. Build presence and submission are not parity
/// evidence; no conversion to the CUDA-only resident admission schema exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HipSessionBuildIdentityV1 {
    runtime: HipRuntimeIdentityV1,
    target: String,
    manifest_sha256: [u8; 32],
    artifact_sha256: [u8; 32],
    input_sha256: [u8; 32],
}

impl HipSessionBuildIdentityV1 {
    pub const fn schema(&self) -> &'static str {
        HIP_SESSION_IDENTITY_SCHEMA_V1
    }
    pub fn runtime(&self) -> &HipRuntimeIdentityV1 {
        &self.runtime
    }
    pub fn target(&self) -> &str {
        &self.target
    }
    pub const fn manifest_sha256(&self) -> [u8; 32] {
        self.manifest_sha256
    }
    pub const fn artifact_sha256(&self) -> [u8; 32] {
        self.artifact_sha256
    }
    pub const fn input_sha256(&self) -> [u8; 32] {
        self.input_sha256
    }
}

/// Immutable, validated input. This does not claim source-dataset provenance or
/// selected/holdout scope authority, and does not fabricate missing lanes.
#[derive(Debug)]
pub struct PreparedHipSessionV1<'input> {
    input: PreparedHipOhlcvV1<'input>,
    memory: HipSessionMemoryPlanV1,
    input_sha256: [u8; 32],
}

impl<'input> PreparedHipSessionV1<'input> {
    pub fn preflight(ohlcv: &'input Ohlcv) -> Result<Self> {
        let input = PreparedHipOhlcvV1::preflight(ohlcv)?;
        input.identity().require_session_clock()?;
        let memory = HipSessionMemoryPlanV1::checked(ohlcv.len())?;
        let input_sha256 = input.identity().session_input_sha256();
        Ok(Self {
            input,
            memory,
            input_sha256,
        })
    }

    pub const fn memory_plan(&self) -> HipSessionMemoryPlanV1 {
        self.memory
    }
    pub const fn input_sha256(&self) -> [u8; 32] {
        self.input_sha256
    }

    /// Submit one existing native Session-v2 kernel on the actual lease. All
    /// eight allocations are owned by the returned batch; no values return to
    /// the CPU. Allocation still may fail after the non-reserving live snapshot.
    pub fn materialize<'lease>(
        &self,
        lease: &'lease HipRunLeaseV1,
    ) -> Result<ResidentHipSessionV1<'lease>> {
        checked_manifest(
            hip_session_build_manifest_v1().context("HIP Session kernel build is absent")?,
            lease.identity().architecture(),
        )?;
        // Check the whole standalone extent before any upload. The shared-input
        // route below instead admits only outputs against remaining live memory.
        let memory = lease.revalidate()?;
        self.memory
            .check_available_bytes(memory.free_memory_bytes())?;
        let input = self.input.upload(lease)?;
        ResidentHipSessionV1::materialize_on(&input)
    }
}

impl<'lease> ResidentHipSessionV1<'lease> {
    /// A selected descriptor carries no public output handle and can only be
    /// consumed by the source-bound HIP canonical assembler.
    #[cfg(feature = "gpu-hip-smc")]
    pub fn column_v1(
        &self,
        index: usize,
    ) -> Result<super::gpu_hip_feature_store_v1::HipResidentFeatureColumnV1<'_, 'lease>> {
        super::gpu_hip_feature_store_v1::HipResidentFeatureColumnV1::from_producer(
            &self.input,
            &self.values,
            &self.validity,
            &HIP_SESSION_COLUMN_NAMES_V1,
            index,
            neoethos_gpu_contracts::resident_feature_store_v3::ResidentFeatureProducerV3::Session,
            2,
            SESSION_SOURCE,
            self.identity.manifest_sha256(),
            self.identity.artifact_sha256(),
        )
    }

    /// Run Session over a genuine existing upload. No input H2D or rehash occurs.
    pub fn materialize_on(input: &ResidentHipOhlcvV1<'lease>) -> Result<Self> {
        input.identity().require_session_clock()?;
        let lease = input.lease();
        let memory = HipSessionMemoryPlanV1::checked(input.memory_plan().row_count())?;
        let manifest_text =
            hip_session_build_manifest_v1().context("HIP Session kernel build is absent")?;
        let manifest = checked_manifest(manifest_text, lease.identity().architecture())?;
        let identity = HipSessionBuildIdentityV1 {
            runtime: lease.identity().clone(),
            target: manifest.target,
            manifest_sha256: manifest.manifest_sha256,
            artifact_sha256: manifest.artifact_sha256,
            input_sha256: input.identity().session_input_sha256(),
        };
        let memory_at_admission = lease.revalidate()?;
        ensure!(
            memory_at_admission.lease_id() == lease.identity().lease_id(),
            "HIP memory owner drift"
        );
        let incremental = memory
            .value_bytes
            .checked_add(memory.validity_bytes)
            .context("HIP Session incremental extent overflow")?;
        super::gpu_hip_ohlcv_v1::check_free_bytes(
            incremental,
            memory_at_admission.free_memory_bytes(),
            "HIP Session outputs",
        )?;
        let values = lease.allocate_bytes(memory.value_bytes)?;
        let validity = lease.allocate_bytes(memory.validity_bytes)?;
        lease.launch_session_f64_v2(memory.rows, input.lanes(), &values, &validity)?;
        Ok(ResidentHipSessionV1 {
            lease,
            input: input.clone(),
            values,
            validity,
            memory,
            memory_at_admission,
            identity,
        })
    }
}

/// Queued, lease-owned feature-major `[23][rows]` f64 values and logical u8
/// validity (0 valid, 1 warmup, 5 zero denominator). Retained parents cannot be
/// dropped before the queued kernel's buffer lifetime. Raw output handles are
/// not exposed: a caller must not overwrite a producer's identity-bound buffers.
pub struct ResidentHipSessionV1<'lease> {
    lease: &'lease HipRunLeaseV1,
    input: ResidentHipOhlcvV1<'lease>,
    values: HipDeviceBufferV1<'lease>,
    validity: HipDeviceBufferV1<'lease>,
    memory: HipSessionMemoryPlanV1,
    memory_at_admission: HipRuntimeMemorySnapshotV1,
    identity: HipSessionBuildIdentityV1,
}

/// Explicit diagnostic-only host copy; this is not a feature-store authority.
pub struct HipSessionDiagnosticV1 {
    pub values: Vec<u8>,
    pub validity: Vec<u8>,
}

impl<'lease> ResidentHipSessionV1<'lease> {
    pub fn identity(&self) -> &HipSessionBuildIdentityV1 {
        &self.identity
    }
    pub const fn memory_plan(&self) -> HipSessionMemoryPlanV1 {
        self.memory
    }
    pub const fn memory_at_admission(&self) -> HipRuntimeMemorySnapshotV1 {
        self.memory_at_admission
    }
    /// Explicit terminal validation boundary. Refuse above the caller's byte
    /// budget BEFORE completion/readback; normal materialization does no D2H.
    pub fn read_terminal_diagnostic(&self, max_bytes: usize) -> Result<HipSessionDiagnosticV1> {
        let required = self
            .memory
            .value_bytes
            .checked_add(self.memory.validity_bytes)
            .context("HIP Session diagnostic extent overflow")?;
        ensure!(
            required <= max_bytes,
            "HIP Session diagnostic exceeds explicit byte budget"
        );
        self.synchronize()?;
        Ok(HipSessionDiagnosticV1 {
            values: self.values.read_bytes()?,
            validity: self.validity.read_bytes()?,
        })
    }
    pub fn retained_parent_bytes(&self) -> usize {
        self.input.memory_plan().device_bytes()
    }
    pub fn inputs(&self) -> &ResidentHipOhlcvV1<'lease> {
        &self.input
    }
    pub fn synchronize(&self) -> Result<()> {
        self.lease
            .synchronize()
            .context("HIP Session completion failed")
    }

    /// Observe every buffer-release error at the explicit terminal boundary.
    /// Native ownership retains uncertain work; neither this method nor the
    /// buffer destructors retry a failed release. The lease is closed separately.
    pub fn try_close(self) -> Result<()> {
        let Self {
            input,
            values,
            validity,
            ..
        } = self;
        let output_result = close_buffers([values, validity]);
        let input_result = input.try_close();
        output_result
            .and(input_result)
            .context("HIP Session buffer cleanup failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_input() -> Ohlcv {
        Ohlcv {
            timestamp: Some(vec![1_700_000_000_000, 1_700_000_060_000]),
            open: vec![2.0, 2.0],
            high: vec![3.0, 3.0],
            low: vec![1.0, 1.0],
            close: vec![2.5, 2.5],
            volume: Some(vec![0.0, 4.0]),
        }
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn manifest() -> serde_json::Value {
        serde_json::json!({
            "schema": "neoethos.hip-session-kernels-build.v1", "backend": "amd-hip",
            "target": "gfx942", "semantic_version": 2,
            "source_sha256": hex(&Sha256::digest(SESSION_SOURCE)),
            "abi_sha256": hex(&Sha256::digest(SESSION_ABI)),
            "artifact_sha256": "ab".repeat(32), "precision": SESSION_PRECISION,
            "device_executed": false,
        })
    }

    #[test]
    fn checked_memory_includes_all_six_parents_and_both_outputs() {
        let plan = HipSessionMemoryPlanV1::checked(100).unwrap();
        assert_eq!(plan.parent_bytes(), 4_800);
        assert_eq!(plan.value_bytes(), 18_400);
        assert_eq!(plan.validity_bytes(), 2_300);
        assert_eq!(plan.total_device_bytes(), 25_500);
        assert!(plan.check_available_bytes(25_499).is_err());
        assert!(plan.check_available_bytes(25_500).is_ok());
        assert!(HipSessionMemoryPlanV1::checked(0).is_err());
        assert!(HipSessionMemoryPlanV1::checked(usize::MAX / 255 + 1).is_err());
        assert!(HipSessionMemoryPlanV1::checked(usize::MAX).is_err());
    }

    #[test]
    fn input_preflight_preserves_canonical_shape_bounds_and_dual_clock() {
        assert!(PreparedHipSessionV1::preflight(&valid_input()).is_ok());
        let mut input = valid_input();
        input.volume = None;
        assert!(PreparedHipSessionV1::preflight(&input).is_err());
        input = valid_input();
        input.timestamp = None;
        assert!(PreparedHipSessionV1::preflight(&input).is_err());
        input = valid_input();
        input.open.pop();
        assert!(PreparedHipSessionV1::preflight(&input).is_err());
        input = valid_input();
        input.timestamp.as_mut().unwrap()[1] = input.timestamp.as_ref().unwrap()[0];
        assert!(PreparedHipSessionV1::preflight(&input).is_err());
        input = valid_input();
        input.timestamp = Some(vec![10_000_000_000_000, 10_000_000_060_000]);
        assert!(
            validate_canonical_millisecond_timestamps(input.timestamp.as_ref().unwrap()).is_ok()
        );
        assert!(PreparedHipSessionV1::preflight(&input).is_err());
        for lane in 0..5 {
            for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
                let mut input = valid_input();
                match lane {
                    0 => input.open[0] = bad,
                    1 => input.high[0] = bad,
                    2 => input.low[0] = bad,
                    3 => input.close[0] = bad,
                    _ => input.volume.as_mut().unwrap()[0] = bad,
                }
                assert!(
                    PreparedHipSessionV1::preflight(&input).is_err(),
                    "lane={lane}, bad={bad}"
                );
            }
        }
        input = valid_input();
        input.high[0] = 2.0;
        assert!(PreparedHipSessionV1::preflight(&input).is_err());
        input = valid_input();
        input.low[0] = 2.25;
        assert!(PreparedHipSessionV1::preflight(&input).is_err());
    }

    #[test]
    fn input_identity_binds_every_lane_and_exact_bits_without_feature_computation() {
        let baseline = PreparedHipSessionV1::preflight(&valid_input())
            .unwrap()
            .input_sha256();
        for lane in 0..6 {
            let mut input = valid_input();
            match lane {
                0 => input.open[0] += 0.125,
                1 => input.high[0] += 0.125,
                2 => input.low[0] -= 0.125,
                3 => input.close[0] += 0.125,
                4 => input.volume.as_mut().unwrap()[0] = -0.0,
                _ => input.timestamp.as_mut().unwrap()[0] -= 1,
            }
            assert_ne!(
                PreparedHipSessionV1::preflight(&input)
                    .unwrap()
                    .input_sha256(),
                baseline
            );
        }
        assert_eq!(
            f64_bytes(&[1.0, -0.0]),
            [1.0f64.to_le_bytes(), (-0.0f64).to_le_bytes()].concat()
        );
        assert_eq!(
            i64_bytes(&[1, -2]),
            [1i64.to_le_bytes(), (-2i64).to_le_bytes()].concat()
        );
    }

    #[test]
    fn manifest_refuses_backend_target_source_precision_and_proof_substitution() {
        assert!(checked_manifest(&manifest().to_string(), "gfx942:sramecc+:xnack-").is_ok());
        assert!(checked_manifest(&manifest().to_string(), "gfx90a").is_err());
        for (field, value) in [
            ("backend", serde_json::json!("cuda")),
            ("target", serde_json::json!("sm_86")),
            ("semantic_version", serde_json::json!(3)),
            ("precision", serde_json::json!("fast-math")),
            ("source_sha256", serde_json::json!("ab".repeat(32))),
            ("abi_sha256", serde_json::json!("ab".repeat(32))),
            ("artifact_sha256", serde_json::json!("00".repeat(32))),
            ("device_executed", serde_json::json!(true)),
        ] {
            let mut changed = manifest();
            changed[field] = value;
            assert!(
                checked_manifest(&changed.to_string(), "gfx942").is_err(),
                "field={field}"
            );
        }
    }

    #[test]
    fn column_schema_matches_cpu_session_metadata_not_a_cuda_capability_receipt() {
        // This host-only assertion compares metadata, not numerical GPU output.
        let names: Vec<_> =
            super::super::session_features::compute_session_feature_columns(&valid_input())
                .into_iter()
                .map(|(name, _)| name)
                .collect();
        assert_eq!(names, HIP_SESSION_COLUMN_NAMES_V1);
    }
}
