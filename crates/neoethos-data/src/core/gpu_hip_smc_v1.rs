//! SMC semantic-v3 on a genuine shared HIP OHLCV upload.
//!
//! Calls the existing native SMC kernel; no CPU feature computation, additional
//! parent upload, or CUDA identity is used. Outputs remain column-major f64 and
//! logical-u8 validity, NOT a packed/bar-major canonical Search feature store.
//! The current causal native kernel is serial; this port claims no speedup.

use super::gpu_hip_ohlcv_v1::{
    CheckedHipProducerBuildV1, HipOhlcvMemoryPlanV1, ResidentHipOhlcvV1, check_free_bytes,
    checked_bytes, checked_producer_build, close_buffers,
};
use anyhow::{Context as _, Result, ensure};
use neoethos_gpu_cuda::hip_runtime_v1::{
    HipDeviceBufferV1, HipRuntimeIdentityV1, HipRuntimeMemorySnapshotV1, hip_smc_build_manifest_v1,
};

pub const HIP_SMC_COLUMN_NAMES_V1: [&str; 46] = [
    "smc_ob",
    "smc_fvg",
    "smc_ifvg",
    "smc_liq_sweep",
    "smc_pd_array",
    "smc_killzone",
    "smc_displacement",
    "smc_breaker_block",
    "smc_mitigation_block",
    "smc_mss",
    "smc_volume_imbalance",
    "smc_bos",
    "smc_eqh",
    "smc_eql",
    "smc_inducement",
    "smc_asian_range",
    "smc_silver_bullet",
    "smc_judas_swing",
    "smc_nwog",
    "smc_ndog",
    "smc_ict_macro",
    "smc_fvg_strength",
    "smc_dealing_range_width",
    "smc_swing_range_pct",
    "smc_ob_strength",
    "smc_trend_bias",
    "smc_unicorn_model",
    "smc_rejection_block",
    "smc_propulsion_block",
    "smc_fib_time_ratio",
    "smc_fib_236",
    "smc_fib_382",
    "smc_fib_500",
    "smc_fib_618",
    "smc_fib_705",
    "smc_fib_786",
    "smc_fib_886",
    "smc_fib_1272",
    "smc_fib_1414",
    "smc_fib_1618",
    "smc_fib_2000",
    "smc_fib_2618",
    "smc_fvg_magnet_dist",
    "smc_fvg_magnet_age",
    "smc_fvg_inside",
    "smc_fvg_open_count",
];
pub const HIP_SMC_PARENT_SLOT_NAMES_V1: [&str; 11] = [
    "ob",
    "fvg",
    "liquidity",
    "trend",
    "premium",
    "inducement",
    "bos",
    "choch",
    "eqh",
    "eql",
    "displacement",
];
const SMC_SOURCE: &[u8] = include_bytes!("../../../neoethos-gpu-cuda/native/resident_smc_v3.cu");
const SMC_ABI: &[u8] = include_bytes!("../../../neoethos-gpu-cuda/hip/hip_runtime_owner_v1.h");

/// Incremental allocations over an already resident 48N-byte OHLCV input.
/// Extents are logical payloads, not allocator overhead or reservations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HipSmcMemoryPlanV1 {
    rows: usize,
    outputs: [usize; 7],
    incremental_bytes: usize,
    parent_bytes: usize,
    standalone_bytes: usize,
}

impl HipSmcMemoryPlanV1 {
    pub fn checked(rows: usize) -> Result<Self> {
        // Existing native FVG/OB birth positions are signed int. This is an
        // exact representability restriction, not a selected-data truncation.
        ensure!(
            rows > 0 && rows - 1 <= i32::MAX as usize,
            "HIP SMC last row must fit the native signed-32-bit birth index"
        );
        let parent_bytes = HipOhlcvMemoryPlanV1::checked(rows)?.device_bytes();
        let outputs = [
            checked_bytes(rows, 46 * 8, "HIP SMC values")?,
            checked_bytes(rows, 46, "HIP SMC validity")?,
            checked_bytes(rows, 8, "HIP SMC months")?,
            checked_bytes(rows, 8, "HIP SMC days")?,
            checked_bytes(rows, 11, "HIP SMC parent slots")?,
            96,
            4,
        ];
        let incremental_bytes = outputs.iter().try_fold(0usize, |sum, n| {
            sum.checked_add(*n)
                .context("HIP SMC incremental extent overflow")
        })?;
        let standalone_bytes = parent_bytes
            .checked_add(incremental_bytes)
            .context("HIP SMC standalone extent overflow")?;
        Ok(Self {
            rows,
            outputs,
            incremental_bytes,
            parent_bytes,
            standalone_bytes,
        })
    }
    pub const fn row_count(&self) -> usize {
        self.rows
    }
    /// values/validity/months/days/SMC11/generated hashes/device error, in ABI order.
    pub const fn output_bytes(&self) -> &[usize; 7] {
        &self.outputs
    }
    pub const fn incremental_device_bytes(&self) -> usize {
        self.incremental_bytes
    }
    pub const fn shared_parent_bytes(&self) -> usize {
        self.parent_bytes
    }
    pub const fn standalone_device_bytes(&self) -> usize {
        self.standalone_bytes
    }
    pub const fn sealing_d2h_bytes(&self) -> usize {
        100
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HipSmcBuildIdentityV1 {
    runtime: HipRuntimeIdentityV1,
    build: CheckedHipProducerBuildV1,
    input_sha256: [u8; 32],
    generated_parent_sha256: [[u8; 32]; 3],
}
impl HipSmcBuildIdentityV1 {
    pub const fn schema(&self) -> &'static str {
        "neoethos.data.hip-smc-producer.v1"
    }
    pub fn runtime(&self) -> &HipRuntimeIdentityV1 {
        &self.runtime
    }
    pub fn target(&self) -> &str {
        &self.build.target
    }
    pub const fn artifact_sha256(&self) -> [u8; 32] {
        self.build.artifact_sha256
    }
    pub const fn manifest_sha256(&self) -> [u8; 32] {
        self.build.manifest_sha256
    }
    pub const fn input_sha256(&self) -> [u8; 32] {
        self.input_sha256
    }
    /// Device-generated SHA256 of months, days, row-major SMC11, in that order.
    pub const fn generated_parent_sha256(&self) -> &[[u8; 32]; 3] {
        &self.generated_parent_sha256
    }
}

fn checked_manifest(text: &str, architecture: &str) -> Result<CheckedHipProducerBuildV1> {
    checked_producer_build(
        text,
        architecture,
        "neoethos.hip-smc-kernels-build.v1",
        3,
        SMC_SOURCE,
        SMC_ABI,
    )
}

fn checked_generated_hashes(bytes: [u8; 96]) -> Result<[[u8; 32]; 3]> {
    let hashes = std::array::from_fn(|i| {
        let mut hash = [0; 32];
        hash.copy_from_slice(&bytes[i * 32..(i + 1) * 32]);
        hash
    });
    ensure!(
        hashes.iter().all(|hash| *hash != [0; 32]),
        "HIP SMC generated parent hash is zero"
    );
    Ok(hashes)
}

/// Successfully completed SMC parent and raw feature batch on the shared lease.
/// Construction requires native error=0 and bounded generated-hash readback.
/// No conversion to CUDA admission or the complete HIP Search carrier exists.
pub struct ResidentHipSmcV1<'lease> {
    input: ResidentHipOhlcvV1<'lease>,
    outputs: [HipDeviceBufferV1<'lease>; 7],
    memory: HipSmcMemoryPlanV1,
    memory_at_admission: HipRuntimeMemorySnapshotV1,
    identity: HipSmcBuildIdentityV1,
}

/// Explicit terminal diagnostic bytes, not canonical Search admission.
pub struct HipSmcDiagnosticV1 {
    pub values: Vec<u8>,
    pub validity: Vec<u8>,
    pub months: Vec<u8>,
    pub days: Vec<u8>,
    pub parent_slots: Vec<u8>,
}

impl<'lease> ResidentHipSmcV1<'lease> {
    /// Select an actual produced column without exposing a mutable device
    /// handle. Canonical assembly preserves this exact name and local index.
    pub fn column_v1(
        &self,
        index: usize,
    ) -> Result<super::gpu_hip_feature_store_v1::HipResidentFeatureColumnV1<'_, 'lease>> {
        super::gpu_hip_feature_store_v1::HipResidentFeatureColumnV1::from_producer(
            &self.input,
            &self.outputs[0],
            &self.outputs[1],
            &HIP_SMC_COLUMN_NAMES_V1,
            index,
            neoethos_gpu_contracts::resident_feature_store_v3::ResidentFeatureProducerV3::Smc,
            3,
            SMC_SOURCE,
            self.identity.manifest_sha256(),
            self.identity.artifact_sha256(),
        )
    }

    pub fn materialize_on(input: &ResidentHipOhlcvV1<'lease>) -> Result<Self> {
        let memory = HipSmcMemoryPlanV1::checked(input.memory_plan().row_count())?;
        let lease = input.lease();
        let build = checked_manifest(
            hip_smc_build_manifest_v1().context("HIP SMC native kernel build is absent")?,
            lease.identity().architecture(),
        )?;
        let memory_at_admission = lease.revalidate()?;
        check_free_bytes(
            memory.incremental_bytes,
            memory_at_admission.free_memory_bytes(),
            "HIP SMC outputs",
        )?;
        let [values, validity, months, days, slots, hashes, error] = memory.outputs;
        let outputs = [
            lease.allocate_bytes(values)?,
            lease.allocate_bytes(validity)?,
            lease.allocate_bytes(months)?,
            lease.allocate_bytes(days)?,
            lease.allocate_bytes(slots)?,
            lease.allocate_bytes(hashes)?,
            lease.allocate_bytes(error)?,
        ];
        let lanes = input.lanes();
        // Native seals only after synchronized error=0, then 96 hash bytes.
        // A failed/partial kernel never marks these allocations initialized.
        let generated = lease.launch_smc_parent_f64_v3(
            memory.rows,
            [lanes[0], lanes[1], lanes[2], lanes[3], lanes[5]],
            std::array::from_fn(|i| &outputs[i]),
        )?;
        let identity = HipSmcBuildIdentityV1 {
            runtime: lease.identity().clone(),
            build,
            input_sha256: input.identity().input_sha256(),
            generated_parent_sha256: checked_generated_hashes(generated)?,
        };
        Ok(Self {
            input: input.clone(),
            outputs,
            memory,
            memory_at_admission,
            identity,
        })
    }
    pub fn inputs(&self) -> &ResidentHipOhlcvV1<'lease> {
        &self.input
    }
    /// Genuine generated months/days/SMC11 in native parent order. Kept private
    /// to Data so an external caller cannot substitute or overwrite a lane.
    pub(crate) fn population_lanes_v1(&self) -> [&HipDeviceBufferV1<'lease>; 3] {
        [&self.outputs[2], &self.outputs[3], &self.outputs[4]]
    }
    pub fn identity(&self) -> &HipSmcBuildIdentityV1 {
        &self.identity
    }
    pub const fn memory_plan(&self) -> HipSmcMemoryPlanV1 {
        self.memory
    }
    pub const fn memory_at_admission(&self) -> HipRuntimeMemorySnapshotV1 {
        self.memory_at_admission
    }
    /// The producer is already sealed; read only at an explicit, byte-bounded
    /// diagnostic boundary. Raw device buffers cannot be overwritten by callers.
    pub fn read_terminal_diagnostic(&self, max_bytes: usize) -> Result<HipSmcDiagnosticV1> {
        let required = self
            .memory
            .incremental_bytes
            .checked_sub(100)
            .context("HIP SMC diagnostic extent underflow")?;
        ensure!(
            required <= max_bytes,
            "HIP SMC diagnostic exceeds explicit byte budget"
        );
        Ok(HipSmcDiagnosticV1 {
            values: self.outputs[0].read_bytes()?,
            validity: self.outputs[1].read_bytes()?,
            months: self.outputs[2].read_bytes()?,
            days: self.outputs[3].read_bytes()?,
            parent_slots: self.outputs[4].read_bytes()?,
        })
    }
    pub fn try_close(self) -> Result<()> {
        let output_result = close_buffers(self.outputs);
        let input_result = self.input.try_close();
        output_result
            .and(input_result)
            .context("HIP SMC buffer cleanup failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    #[test]
    fn smc_metadata_matches_the_existing_cpu_semantic_authority() {
        let input = crate::Ohlcv {
            timestamp: Some(vec![1_700_000_000_000, 1_700_000_060_000]),
            open: vec![2.0; 2],
            high: vec![3.0; 2],
            low: vec![1.0; 2],
            close: vec![2.5; 2],
            volume: Some(vec![1.0; 2]),
        };
        let cpu = super::super::smc::compute_smc_feature_columns_f64(&input).unwrap();
        assert_eq!(super::super::smc::SMC_SEMANTIC_VERSION, 3);
        assert!(
            cpu.iter()
                .map(|c| c.name.as_str())
                .eq(HIP_SMC_COLUMN_NAMES_V1)
        );
    }
    #[test]
    fn exact_smc_incremental_and_shared_combined_extents() {
        let p = HipSmcMemoryPlanV1::checked(100).unwrap();
        assert_eq!(*p.output_bytes(), [36_800, 4600, 800, 800, 1100, 96, 4]);
        assert_eq!(p.incremental_device_bytes(), 44_200);
        assert_eq!(p.standalone_device_bytes(), 49_000);
        assert_eq!(
            p.shared_parent_bytes() + 100 * 23 * 9 + p.incremental_device_bytes(),
            69_700
        );
        assert_eq!(p.sealing_d2h_bytes(), 100);
        assert!(HipSmcMemoryPlanV1::checked(0).is_err());
        assert!(HipSmcMemoryPlanV1::checked(usize::MAX).is_err());
        if usize::BITS >= 64 {
            assert!(HipSmcMemoryPlanV1::checked(i32::MAX as usize + 1).is_ok());
            assert!(HipSmcMemoryPlanV1::checked(i32::MAX as usize + 2).is_err());
        }
    }
    #[test]
    fn generated_hashes_bind_order_and_refuse_partial_zero_output() {
        let mut bytes = [0; 96];
        for i in 0..3 {
            bytes[i * 32..(i + 1) * 32].fill((i + 1) as u8);
        }
        assert_eq!(
            checked_generated_hashes(bytes).unwrap(),
            [[1; 32], [2; 32], [3; 32]]
        );
        for i in 0..3 {
            let mut bad = bytes;
            bad[i * 32..(i + 1) * 32].fill(0);
            assert!(checked_generated_hashes(bad).is_err());
        }
    }
    #[test]
    fn smc_manifest_is_family_source_and_lease_abi_specific() {
        let hex = |bytes: &[u8]| bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
        let m = serde_json::json!({"schema":"neoethos.hip-smc-kernels-build.v1","backend":"amd-hip",
            "target":"gfx942","semantic_version":3,"source_sha256":hex(&Sha256::digest(SMC_SOURCE)),
            "abi_sha256":hex(&Sha256::digest(SMC_ABI)),"artifact_sha256":"ab".repeat(32),
            "precision":"f64-no-fast-math-no-contract-ieee-denormals","device_executed":false});
        assert!(checked_manifest(&m.to_string(), "gfx942:sramecc+:xnack-").is_ok());
        for (field, value) in [
            (
                "schema",
                serde_json::json!("neoethos.hip-session-kernels-build.v1"),
            ),
            ("semantic_version", serde_json::json!(2)),
            ("backend", serde_json::json!("cuda")),
            ("source_sha256", serde_json::json!("01".repeat(32))),
            ("abi_sha256", serde_json::json!("02".repeat(32))),
            ("artifact_sha256", serde_json::json!("00".repeat(32))),
            ("precision", serde_json::json!("fast")),
            ("device_executed", serde_json::json!(true)),
        ] {
            let mut bad = m.clone();
            bad[field] = value;
            assert!(
                checked_manifest(&bad.to_string(), "gfx942").is_err(),
                "{field}"
            );
        }
        assert!(checked_manifest(&m.to_string(), "gfx90a").is_err());
    }
}
