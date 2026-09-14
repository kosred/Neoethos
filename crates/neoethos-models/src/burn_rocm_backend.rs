//! Native Burn ROCm lifecycle, not a CUDA identity adapter or CPU fallback.
//!
//! CubeCL owns the HIP context/streams. This owner pins the exact CubeCL/Fusion
//! stream used by a model across calling threads; it is not the separate native
//! Data/Search HIP lease and does not claim to make external device resets safe.

use anyhow::{Context, Result, bail};
use burn::tensor::backend::Backend;
use cubecl::hip::{AmdDevice, HipRuntime};
use cubecl::prelude::{ComputeClient, Runtime};
use cubecl_common::stream_id::StreamId;
use std::sync::Mutex;

pub(crate) struct RocmModelResidency {
    ordinal: usize,
    stream: StreamId,
    client: ComputeClient<HipRuntime>,
    operation: Mutex<()>,
}

impl std::fmt::Debug for RocmModelResidency {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("RocmModelResidency")
            .field("ordinal", &self.ordinal)
            .field("stream", &self.stream)
            .finish_non_exhaustive()
    }
}

impl RocmModelResidency {
    pub(crate) fn new(policy: &str) -> Result<Self> {
        let ordinal = crate::common::parse_rocm_device_ordinal(policy)?;
        // Query the real selected AMD device before creating Burn handles. A
        // missing runtime/card is an error, never ndarray or ordinal-zero fallback.
        let _ = memory_snapshot(ordinal)?;
        let stream = StreamId::current();
        let mut client = HipRuntime::client(&AmdDevice::new(ordinal));
        // SAFETY: retain exactly the same logical stream as the model's Burn
        // operations; executes() restores it for both Fusion and CubeCL.
        unsafe { client.set_stream(stream) };
        Ok(Self {
            ordinal,
            stream,
            client,
            operation: Mutex::new(()),
        })
    }

    pub(crate) fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub(crate) fn executes<T>(&self, operation: impl FnOnce() -> T) -> T {
        let _guard = self
            .operation
            .lock()
            .expect("ROCm model stream operation previously panicked; refusing reuse");
        self.stream.executes(operation)
    }

    pub(crate) fn drop_handles(&self, operation: impl FnOnce()) {
        // A replacement/error path may drop an expert inside executes().
        // Enqueuing tensor DropOps must not recursively lock that operation.
        // Fusion's server owns its handle synchronization; this does not run
        // allocator cleanup, which only occurs after the last Arc is gone.
        self.stream.executes(operation);
    }

    fn release_unused_pages(&self) -> Result<()> {
        self.stream.executes(|| {
            // Fusion must consume deferred DropOps before the allocator sees
            // handles as unused. Failure stops cleanup, retaining ambiguous work.
            <crate::burn_models::InferBackend as Backend>::sync(&AmdDevice::new(self.ordinal))
                .map_err(|error| {
                    anyhow::anyhow!("Burn ROCm fusion synchronization failed: {error:?}")
                })?;
            cubecl::future::block_on(self.client.sync()).map_err(|error| {
                anyhow::anyhow!("ROCm pre-cleanup synchronization failed: {error:?}")
            })?;
            self.client.memory_cleanup();
            cubecl::future::block_on(self.client.sync()).map_err(|error| {
                anyhow::anyhow!("ROCm pool-cleanup synchronization failed: {error:?}")
            })
        })
    }
}

impl Drop for RocmModelResidency {
    fn drop(&mut self) {
        if let Err(error) = self.release_unused_pages() {
            if std::thread::panicking() {
                eprintln!("Burn ROCm cleanup also failed during unwind: {error:#}");
            } else {
                panic!("Burn ROCm cleanup failed: {error:#}");
            }
        }
    }
}

/// An instantaneous free/total observation, not a memory reservation. HIP's
/// current-device setting is thread-local; restore it on every post-select path.
pub(crate) fn memory_snapshot(ordinal: usize) -> Result<(usize, usize)> {
    let ordinal = i32::try_from(ordinal).context("ROCm ordinal exceeds HIP int")?;
    let mut previous = -1;
    // SAFETY: all pointers refer to live host scalars; no raw GPU allocation is
    // manufactured or dereferenced by this read-only device query.
    unsafe {
        let status = cubecl_hip_sys::hipGetDevice(&mut previous);
        if status != cubecl_hip_sys::HIP_SUCCESS {
            bail!("HIP get-device failed: {status}");
        }
        let status = cubecl_hip_sys::hipSetDevice(ordinal);
        if status != cubecl_hip_sys::HIP_SUCCESS {
            bail!("HIP select device {ordinal} failed: {status}");
        }
        let mut free = 0;
        let mut total = 0;
        let query = cubecl_hip_sys::hipMemGetInfo(&mut free, &mut total);
        let restore = cubecl_hip_sys::hipSetDevice(previous);
        if restore != cubecl_hip_sys::HIP_SUCCESS {
            bail!("HIP restore device {previous} failed: {restore}; memory query status {query}");
        }
        if query != cubecl_hip_sys::HIP_SUCCESS {
            bail!("HIP memory query failed: {query}");
        }
        if total == 0 || free > total {
            bail!("HIP memory query returned invalid free/total capacity");
        }
        Ok((free, total))
    }
}

pub(crate) fn mlp_memory_snapshot(ordinal: usize) -> Result<(usize, usize, usize)> {
    let (free, total) = memory_snapshot(ordinal)?;
    let client = HipRuntime::client(&AmdDevice::new(ordinal));
    let max_page = usize::try_from(client.properties().memory.max_page_size)
        .context("MLP ROCm maximum tensor size exceeds usize")?;
    Ok((free, total, max_page))
}

/// Real inventory for an explicitly requested neural device. This is a planning
/// observation, not the native Data/Search lease or a reservation/reset proof.
pub(crate) fn planning_device(
    ordinal: usize,
    profile_id: usize,
) -> Result<neoethos_core::system::AcceleratorDevice> {
    use neoethos_core::system::{
        AcceleratorBackend, AcceleratorDevice, AcceleratorDeviceClass, TrainingPrecision,
    };
    let (_, total) = memory_snapshot(ordinal)?;
    let device = i32::try_from(ordinal).context("ROCm ordinal exceeds HIP int")?;
    let mut name = [0_i8; 256];
    let mut uuid = cubecl_hip_sys::hipUUID { bytes: [0; 16] };
    // SAFETY: bounded, live host output arrays; HIP receives an actual selected
    // ordinal, not a caller-created native device/context handle.
    unsafe {
        let status = cubecl_hip_sys::hipDeviceGetName(name.as_mut_ptr(), name.len() as i32, device);
        anyhow::ensure!(
            status == cubecl_hip_sys::HIP_SUCCESS,
            "HIP device-name query failed: {status}"
        );
        let status = cubecl_hip_sys::hipDeviceGetUuid(&mut uuid, device);
        anyhow::ensure!(
            status == cubecl_hip_sys::HIP_SUCCESS,
            "HIP UUID query failed: {status}"
        );
    }
    let end = name
        .iter()
        .position(|&byte| byte == 0)
        .context("HIP device name is not terminated")?;
    let bytes = name[..end]
        .iter()
        .map(|&byte| byte as u8)
        .collect::<Vec<_>>();
    let name = std::str::from_utf8(&bytes).context("HIP device name is not UTF-8")?;
    anyhow::ensure!(
        !name.trim().is_empty() && uuid.bytes.iter().any(|&byte| byte != 0),
        "HIP device identity is empty"
    );
    let uuid = uuid
        .bytes
        .iter()
        .map(|&byte| format!("{:02x}", byte as u8))
        .collect::<String>();
    Ok(AcceleratorDevice {
        id: profile_id,
        // stable_id already hashes the inventory name. Include the observed
        // UUID so two identically named cards cannot share this profile binding.
        name: format!("{name} [HIP UUID {uuid}]"),
        backend: AcceleratorBackend::Rocm,
        device_class: AcceleratorDeviceClass::Other,
        backend_index: ordinal,
        memory_gb: total as f64 / 1024.0_f64.powi(3),
        supported_precisions: vec![TrainingPrecision::Fp32],
        compute_capability: None,
        source: "HIP runtime device/name/UUID/total-memory observation; not a reservation"
            .to_string(),
    })
}
