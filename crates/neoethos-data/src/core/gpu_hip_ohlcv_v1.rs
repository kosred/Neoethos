//! One immutable, six-lane HIP input shared by resident feature producers.
//!
//! This validates and hashes supplied data. The canonical-frame entry retains
//! existing dataset provenance; bare OHLCV cannot mint it. Neither entry grants
//! selection/holdout authority or Search admission. Each upload owns native
//! staging on the genuine lease. Cloning the resident handle never uploads or
//! allocates another lane. Producer-specific clock/shape checks remain separate.

use anyhow::{Context as _, Result, ensure};
use neoethos_gpu_cuda::hip_runtime_v1::{
    HipDeviceBufferV1, HipRunLeaseV1, HipRuntimeMemorySnapshotV1,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::rc::Rc;

use super::canonical_ohlcv::{CanonicalDatasetArtifactV1, CanonicalOhlcvFrame};
use super::timestamps::{
    TimestampUnit, infer_timestamp_unit, validate_canonical_millisecond_timestamps,
};
use crate::Ohlcv;
use neoethos_feature_contracts::SourceArtifactBindingV1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HipOhlcvMemoryPlanV1 {
    rows: usize,
    lane_bytes: usize,
    device_bytes: usize,
}

impl HipOhlcvMemoryPlanV1 {
    pub fn checked(rows: usize) -> Result<Self> {
        ensure!(rows > 0, "HIP OHLCV requires at least one row");
        let lane_bytes = checked_bytes(rows, 8, "HIP OHLCV lane")?;
        let device_bytes = lane_bytes
            .checked_mul(6)
            .context("HIP OHLCV total overflow")?;
        Ok(Self {
            rows,
            lane_bytes,
            device_bytes,
        })
    }
    pub const fn row_count(&self) -> usize {
        self.rows
    }
    pub const fn lane_bytes(&self) -> usize {
        self.lane_bytes
    }
    /// Logical payload, not allocator overhead or a reservation.
    pub const fn device_bytes(&self) -> usize {
        self.device_bytes
    }
    /// Maximum pending native upload-staging payload for the six copies.
    pub const fn upload_staging_bytes(&self) -> usize {
        self.device_bytes
    }
}

pub(crate) fn checked_bytes(rows: usize, bytes_per_row: usize, label: &str) -> Result<usize> {
    let bytes = rows
        .checked_mul(bytes_per_row)
        .with_context(|| format!("{label} overflow"))?;
    ensure!(
        bytes <= isize::MAX as usize,
        "{label} exceeds addressable Rust extent"
    );
    Ok(bytes)
}

pub(crate) fn check_free_bytes(required: usize, free: u64, label: &str) -> Result<()> {
    ensure!(
        u64::try_from(required)? <= free,
        "{label} needs {required} logical bytes; live free-memory snapshot is {free}"
    );
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProducerManifestV1 {
    schema: String,
    backend: String,
    target: String,
    semantic_version: u32,
    source_sha256: String,
    abi_sha256: String,
    artifact_sha256: String,
    precision: String,
    device_executed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CheckedHipProducerBuildV1 {
    pub target: String,
    pub manifest_sha256: [u8; 32],
    pub artifact_sha256: [u8; 32],
}

pub(crate) fn decode_hash(text: &str) -> Result<[u8; 32]> {
    ensure!(text.len() == 64, "HIP build hash is not SHA-256");
    let mut bytes = [0u8; 32];
    for (index, pair) in text.as_bytes().chunks_exact(2).enumerate() {
        bytes[index] = u8::from_str_radix(std::str::from_utf8(pair)?, 16)
            .context("HIP build hash is not hexadecimal")?;
    }
    ensure!(bytes != [0; 32], "HIP build hash is zero");
    Ok(bytes)
}

/// Shared validation only: callers supply their real, compile-bound source/ABI
/// and family schema. This cannot turn one family's build into another's.
pub(crate) fn checked_producer_build(
    text: &str,
    architecture: &str,
    schema: &str,
    semantic: u32,
    source: &[u8],
    abi: &[u8],
) -> Result<CheckedHipProducerBuildV1> {
    let m: ProducerManifestV1 =
        serde_json::from_str(text).context("malformed HIP producer manifest")?;
    ensure!(
        m.schema == schema
            && m.backend == "amd-hip"
            && m.semantic_version == semantic
            && m.precision == "f64-no-fast-math-no-contract-ieee-denormals"
            && !m.device_executed,
        "HIP producer schema, backend, semantics, precision, or proof scope differs"
    );
    let suffix = m.target.strip_prefix("gfx").unwrap_or_default();
    ensure!(
        !suffix.is_empty()
            && suffix.bytes().all(|b| b.is_ascii_hexdigit())
            && architecture.split(':').next() == Some(m.target.as_str()),
        "HIP producer artifact target differs from actual AMD device"
    );
    ensure!(
        decode_hash(&m.source_sha256)? == <[u8; 32]>::from(Sha256::digest(source))
            && decode_hash(&m.abi_sha256)? == <[u8; 32]>::from(Sha256::digest(abi)),
        "HIP producer source or ABI differs from this Data producer"
    );
    Ok(CheckedHipProducerBuildV1 {
        target: m.target,
        manifest_sha256: Sha256::digest(text.as_bytes()).into(),
        artifact_sha256: decode_hash(&m.artifact_sha256)?,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HipOhlcvInputIdentityV1 {
    input_sha256: [u8; 32],
    // O/H/L/C/V/timestamps: exact little-endian lane hashes, not feature hashes.
    lane_sha256: [[u8; 32]; 6],
    session_input_sha256: [u8; 32],
    session_clock_compatible: bool,
}

impl HipOhlcvInputIdentityV1 {
    pub const fn input_sha256(&self) -> [u8; 32] {
        self.input_sha256
    }
    pub const fn lane_sha256(&self) -> &[[u8; 32]; 6] {
        &self.lane_sha256
    }
    /// Retained Session-v2 input identity; distinct from the shared lane-hash identity.
    pub const fn session_input_sha256(&self) -> [u8; 32] {
        self.session_input_sha256
    }
    /// Check Session's historical value-clock rule without narrowing other families.
    pub fn require_session_clock(&self) -> Result<()> {
        ensure!(
            self.session_clock_compatible,
            "HIP Session canonical timestamp disagrees with retained value-clock inference"
        );
        Ok(())
    }
}

/// The immutable input borrow prevents mutation between preflight and upload.
#[derive(Debug)]
pub struct PreparedHipOhlcvV1<'input> {
    ohlcv: &'input Ohlcv,
    timestamps: &'input [i64],
    volume: &'input [f64],
    memory: HipOhlcvMemoryPlanV1,
    identity: HipOhlcvInputIdentityV1,
    canonical_source: Option<HipCanonicalInputSourceV1>,
}

/// Only the canonical-frame constructor can supply this source authority. The
/// retained artifact keeps the exact immutable generation leased; the binding
/// records the frame's actual segment, never an invented full-parent extent.
#[derive(Debug, Clone)]
pub(crate) struct HipCanonicalInputSourceV1 {
    artifact: CanonicalDatasetArtifactV1,
    binding: SourceArtifactBindingV1,
}

impl HipCanonicalInputSourceV1 {
    pub(crate) fn binding(&self) -> &SourceArtifactBindingV1 {
        &self.binding
    }

    pub(crate) fn artifact(&self) -> &CanonicalDatasetArtifactV1 {
        &self.artifact
    }
}

impl<'input> PreparedHipOhlcvV1<'input> {
    pub fn preflight(ohlcv: &'input Ohlcv) -> Result<Self> {
        let memory = HipOhlcvMemoryPlanV1::checked(ohlcv.len())?;
        ensure!(
            cfg!(target_endian = "little"),
            "HIP OHLCV requires little-endian host lanes"
        );
        ensure!(
            ohlcv.open.len() == memory.rows
                && ohlcv.high.len() == memory.rows
                && ohlcv.low.len() == memory.rows,
            "HIP OHLCV shape mismatch"
        );
        let timestamps = ohlcv
            .timestamp
            .as_deref()
            .context("HIP OHLCV requires timestamps")?;
        let volume = ohlcv
            .volume
            .as_deref()
            .context("HIP OHLCV requires present volume")?;
        ensure!(
            timestamps.len() == memory.rows && volume.len() == memory.rows,
            "HIP OHLCV timestamp or volume shape mismatch"
        );
        validate_canonical_millisecond_timestamps(timestamps)?;
        for row in 0..memory.rows {
            let [open, high, low, close, volume] = [
                ohlcv.open[row],
                ohlcv.high[row],
                ohlcv.low[row],
                ohlcv.close[row],
                volume[row],
            ];
            ensure!(
                [open, high, low, close, volume]
                    .iter()
                    .all(|v| v.is_finite())
                    && open > 0.0
                    && high > 0.0
                    && low > 0.0
                    && close > 0.0
                    && volume >= 0.0
                    && low <= open.min(close)
                    && high >= open.max(close),
                "HIP OHLCV row {row} violates canonical finite bounds"
            );
        }
        let mut lane_sha256 = [[0; 32]; 6];
        let mut session_hash = Sha256::new();
        // Preserve the already-published Session input identity during extraction.
        session_hash.update(b"neoethos.data.hip-session-input.semantic-v2\0");
        session_hash.update((memory.rows as u64).to_le_bytes());
        let clock_bytes = i64_bytes(timestamps);
        lane_sha256[5] = Sha256::digest(clock_bytes).into();
        session_hash.update(clock_bytes);
        for (index, lane) in [
            &ohlcv.open[..],
            &ohlcv.high[..],
            &ohlcv.low[..],
            &ohlcv.close[..],
            volume,
        ]
        .into_iter()
        .enumerate()
        {
            let bytes = f64_bytes(lane);
            lane_sha256[index] = Sha256::digest(bytes).into();
            session_hash.update(bytes);
        }
        let mut shared_hash = Sha256::new();
        shared_hash.update(b"neoethos.data.hip-ohlcv-input.v1\0");
        shared_hash.update((memory.rows as u64).to_le_bytes());
        for hash in lane_sha256 {
            shared_hash.update(hash);
        }
        Ok(Self {
            ohlcv,
            timestamps,
            volume,
            memory,
            identity: HipOhlcvInputIdentityV1 {
                input_sha256: shared_hash.finalize().into(),
                lane_sha256,
                session_input_sha256: session_hash.finalize().into(),
                session_clock_compatible: infer_timestamp_unit(timestamps)
                    == Some(TimestampUnit::Milliseconds),
            },
            canonical_source: None,
        })
    }

    /// Preflight the exact values of a verified immutable canonical frame. This
    /// does not copy its price arrays or grant a Search selection/holdout view.
    /// The ordinary `preflight(&Ohlcv)` remains diagnostic and unprovenanced.
    pub fn preflight_canonical(frame: &'input CanonicalOhlcvFrame) -> Result<Self> {
        let mut prepared = Self::preflight(frame.ohlcv())?;
        let source_node_id = format!("source:{}", frame.artifact().identity().to_path_component());
        prepared.canonical_source = Some(HipCanonicalInputSourceV1 {
            artifact: frame.artifact().clone(),
            binding: frame.source_binding(source_node_id)?,
        });
        Ok(prepared)
    }
    pub const fn memory_plan(&self) -> HipOhlcvMemoryPlanV1 {
        self.memory
    }
    pub fn identity(&self) -> &HipOhlcvInputIdentityV1 {
        &self.identity
    }

    /// Exactly six H2D uploads. Subsequent producer calls reuse these buffers.
    pub fn upload<'lease>(
        &self,
        lease: &'lease HipRunLeaseV1,
    ) -> Result<ResidentHipOhlcvV1<'lease>> {
        let memory_at_upload = lease.revalidate()?;
        check_free_bytes(
            self.memory.device_bytes,
            memory_at_upload.free_memory_bytes(),
            "HIP OHLCV",
        )?;
        let lanes = [
            lease.upload_bytes(f64_bytes(&self.ohlcv.open))?,
            lease.upload_bytes(f64_bytes(&self.ohlcv.high))?,
            lease.upload_bytes(f64_bytes(&self.ohlcv.low))?,
            lease.upload_bytes(f64_bytes(&self.ohlcv.close))?,
            lease.upload_bytes(f64_bytes(self.volume))?,
            lease.upload_bytes(i64_bytes(self.timestamps))?,
        ];
        Ok(ResidentHipOhlcvV1 {
            inner: Rc::new(HipOhlcvInnerV1 {
                lease,
                lanes,
                memory: self.memory,
                memory_at_upload,
                identity: self.identity.clone(),
                canonical_source: self.canonical_source.clone(),
            }),
        })
    }
}

struct HipOhlcvInnerV1<'lease> {
    lease: &'lease HipRunLeaseV1,
    lanes: [HipDeviceBufferV1<'lease>; 6],
    memory: HipOhlcvMemoryPlanV1,
    memory_at_upload: HipRuntimeMemorySnapshotV1,
    identity: HipOhlcvInputIdentityV1,
    canonical_source: Option<HipCanonicalInputSourceV1>,
}

/// A thread-confined shared handle. The last handle owns physical lane cleanup;
/// pending operations and failed cleanup remain protected by the native registry.
#[derive(Clone)]
pub struct ResidentHipOhlcvV1<'lease> {
    inner: Rc<HipOhlcvInnerV1<'lease>>,
}

impl<'lease> ResidentHipOhlcvV1<'lease> {
    pub fn memory_plan(&self) -> HipOhlcvMemoryPlanV1 {
        self.inner.memory
    }
    pub fn identity(&self) -> &HipOhlcvInputIdentityV1 {
        &self.inner.identity
    }
    pub fn memory_at_upload(&self) -> HipRuntimeMemorySnapshotV1 {
        self.inner.memory_at_upload
    }
    pub fn lease(&self) -> &'lease HipRunLeaseV1 {
        self.inner.lease
    }
    pub(crate) fn lanes(&self) -> [&HipDeviceBufferV1<'lease>; 6] {
        std::array::from_fn(|i| &self.inner.lanes[i])
    }
    pub(crate) fn canonical_source(&self) -> Result<&HipCanonicalInputSourceV1> {
        self.inner.canonical_source.as_ref().context(
            "HIP canonical feature assembly requires a verified CanonicalOhlcvFrame, not bare OHLCV"
        )
    }
    pub fn same_upload_as(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }
    /// Close this shared handle. Other producer handles keep the exact inputs
    /// alive; only the final handle submits physical frees and reports errors.
    pub fn try_close(self) -> Result<()> {
        match Rc::try_unwrap(self.inner) {
            Ok(inner) => close_buffers(inner.lanes),
            Err(shared) => {
                drop(shared);
                Ok(())
            }
        }
    }
}

pub(crate) fn close_buffers<'lease>(
    buffers: impl IntoIterator<Item = HipDeviceBufferV1<'lease>>,
) -> Result<()> {
    let mut first_error = None;
    for buffer in buffers {
        if let Err(error) = buffer.try_close() {
            first_error.get_or_insert(error);
        }
    }
    match first_error {
        Some(error) => Err(error).context("HIP buffer cleanup failed"),
        None => Ok(()),
    }
}

pub(crate) fn f64_bytes(lane: &[f64]) -> &[u8] {
    // SAFETY: initialized padding-free values; borrowed exact extent validated
    // to fit isize before upload. The supported native host is little-endian.
    unsafe { std::slice::from_raw_parts(lane.as_ptr().cast(), std::mem::size_of_val(lane)) }
}
pub(crate) fn i64_bytes(lane: &[i64]) -> &[u8] {
    // SAFETY: as above, for initialized padding-free i64 clock values.
    unsafe { std::slice::from_raw_parts(lane.as_ptr().cast(), std::mem::size_of_val(lane)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn input() -> Ohlcv {
        Ohlcv {
            timestamp: Some(vec![1_700_000_000_000, 1_700_000_060_000]),
            open: vec![2.0; 2],
            high: vec![3.0; 2],
            low: vec![1.0; 2],
            close: vec![2.5; 2],
            volume: Some(vec![0.0, 4.0]),
        }
    }

    #[test]
    fn bare_input_cannot_mint_canonical_source_authority() {
        let input = input();
        assert!(
            PreparedHipOhlcvV1::preflight(&input)
                .unwrap()
                .canonical_source
                .is_none()
        );
    }

    #[test]
    fn canonical_preflight_retains_exact_window_binding_and_real_generation_lease() -> Result<()> {
        use super::super::dataset_manifest::{
            DatasetTimestampRange, ProducerProvenanceEnvelopeV1, PublishRequest,
            publish_vortex_generation,
        };
        use neoethos_dataset_contracts::{
            BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe,
        };
        use std::sync::Arc;
        let root = tempfile::tempdir()?;
        let input = input();
        let identity = CanonicalDatasetIdentity::external(
            "hip-source-test",
            "EURUSD",
            CanonicalTimeframe::M1,
            BarTimestampConvention::BarOpen,
        )?;
        let producer = ProducerProvenanceEnvelopeV1::new(
            "neoethos.hip-source-host-test.v1",
            b"synthetic-ownership-test-not-market-evidence".to_vec(),
        )?;
        publish_vortex_generation(PublishRequest {
            configured_root: root.path(),
            identity: &identity,
            expected_generation: None,
            timestamp_range: DatasetTimestampRange::new(1_700_000_000_000, 1_700_000_060_000)?,
            provenance: &producer,
            chunks: crate::ohlcv_to_vortex_chunks(&input, 2)?,
        })?;
        let full = super::super::canonical_ohlcv::load_canonical_timeframe(root.path(), &identity)?;
        let window = full.row_window(1, 2)?;
        let prepared = PreparedHipOhlcvV1::preflight_canonical(&window)?;
        let source = prepared.canonical_source.as_ref().unwrap();
        let expected = window.source_binding(format!("source:{}", identity.to_path_component()))?;
        assert_eq!(source.binding(), &expected);
        assert_eq!(source.binding().segments()[0].row_start(), 1);
        assert_eq!(source.binding().segments()[0].row_end(), 2);
        assert_eq!(source.artifact().row_count(), 2);
        assert!(Arc::ptr_eq(
            source.artifact().lease(),
            full.artifact().lease()
        ));
        assert_eq!(prepared.ohlcv.close.as_ptr(), window.ohlcv().close.as_ptr());
        assert_eq!(prepared.memory.row_count(), 1);
        // The identical supplied bytes alone still do not grant source authority.
        let bare = PreparedHipOhlcvV1::preflight(window.ohlcv())?;
        assert_eq!(bare.identity(), prepared.identity());
        assert!(bare.canonical_source.is_none());
        Ok(())
    }
    #[test]
    fn shared_input_extent_is_six_lanes_once_and_checked() {
        let p = HipOhlcvMemoryPlanV1::checked(100).unwrap();
        assert_eq!(
            (p.lane_bytes(), p.device_bytes(), p.upload_staging_bytes()),
            (800, 4800, 4800)
        );
        assert!(HipOhlcvMemoryPlanV1::checked(0).is_err());
        assert!(HipOhlcvMemoryPlanV1::checked(usize::MAX).is_err());
        assert!(check_free_bytes(4800, 4799, "test").is_err());
    }
    #[test]
    fn canonical_input_does_not_inherit_sessions_narrower_clock() {
        let mut input = input();
        input.timestamp = Some(vec![10_000_000_000_000, 10_000_000_060_000]);
        let p = PreparedHipOhlcvV1::preflight(&input).unwrap();
        assert!(p.identity().require_session_clock().is_err());
    }
    #[test]
    fn shared_validation_rejects_missing_malformed_nonfinite_and_negative_input() {
        for lane in 0..6 {
            let mut changed = input();
            match lane {
                0 => {
                    changed.open.pop();
                }
                1 => {
                    changed.high.pop();
                }
                2 => {
                    changed.low.pop();
                }
                3 => {
                    changed.close.pop();
                }
                4 => {
                    changed.volume = None;
                }
                _ => {
                    changed.timestamp = None;
                }
            }
            assert!(
                PreparedHipOhlcvV1::preflight(&changed).is_err(),
                "shape lane {lane}"
            );
        }
        for lane in 0..5 {
            for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
                let mut changed = input();
                match lane {
                    0 => changed.open[0] = bad,
                    1 => changed.high[0] = bad,
                    2 => changed.low[0] = bad,
                    3 => changed.close[0] = bad,
                    _ => changed.volume.as_mut().unwrap()[0] = bad,
                }
                assert!(
                    PreparedHipOhlcvV1::preflight(&changed).is_err(),
                    "value lane {lane}"
                );
            }
        }
        let mut changed = input();
        changed.timestamp.as_mut().unwrap()[1] = 1_700_000_000_000_000;
        assert!(PreparedHipOhlcvV1::preflight(&changed).is_err());
        let mut changed = input();
        changed.timestamp.as_mut().unwrap()[1] = 1_700_000_000_000;
        assert!(PreparedHipOhlcvV1::preflight(&changed).is_err());
        let mut changed = input();
        changed.high[0] = 2.25;
        assert!(PreparedHipOhlcvV1::preflight(&changed).is_err());
        let mut changed = input();
        changed.low[0] = 2.25;
        assert!(PreparedHipOhlcvV1::preflight(&changed).is_err());
    }
    #[test]
    fn hashes_bind_every_exact_lane_and_preserve_session_identity() {
        let input = input();
        let p = PreparedHipOhlcvV1::preflight(&input).unwrap();
        let mut old = Sha256::new();
        old.update(b"neoethos.data.hip-session-input.semantic-v2\0");
        old.update(2u64.to_le_bytes());
        for t in input.timestamp.as_ref().unwrap() {
            old.update(t.to_le_bytes());
        }
        for lane in [
            &input.open,
            &input.high,
            &input.low,
            &input.close,
            input.volume.as_ref().unwrap(),
        ] {
            for v in lane {
                old.update(v.to_bits().to_le_bytes());
            }
        }
        assert_eq!(
            p.identity().session_input_sha256(),
            <[u8; 32]>::from(old.finalize())
        );
        assert!(p.identity().lane_sha256().iter().all(|h| *h != [0; 32]));
        let mut changed = input.clone();
        changed.volume.as_mut().unwrap()[0] = -0.0;
        let q = PreparedHipOhlcvV1::preflight(&changed).unwrap();
        assert_ne!(p.identity().input_sha256(), q.identity().input_sha256());
        for i in [0, 1, 2, 3, 5] {
            assert_eq!(p.identity().lane_sha256()[i], q.identity().lane_sha256()[i]);
        }
        assert_ne!(p.identity().lane_sha256()[4], q.identity().lane_sha256()[4]);
    }
}
