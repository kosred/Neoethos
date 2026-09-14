//! An owned HIP runtime lease, separate from CUDA admission and numerical proof.
//!
//! This module neither imports `cust` nor grants Data/Search execution authority.
//! Native code owns the actual HIP resources and issues a nonreused registry key.
//! No safe Rust constructor accepts a raw pointer, stream, or lease key.
//! Identity uses modern HIP device/stream APIs, not deprecated context handles.

use std::cell::Cell;
use std::fmt;
use std::io::Write as _;
use std::marker::PhantomData;
use std::num::NonZeroU64;
use std::rc::Rc;

#[cfg(feature = "hip-native-kernels")]
#[path = "hip_feature_store_v1.rs"]
pub mod feature_store_v1;

pub const HIP_RUNTIME_IDENTITY_SCHEMA_V1: &str = "neoethos.hip-runtime-owned-lease.v1";

/// Compiler/artifact metadata for the host runtime bridge only. Its presence is
/// neither device execution evidence nor HIP Data/Search admission authority.
pub const fn hip_runtime_build_manifest_v1() -> Option<&'static str> {
    option_env!("NEOETHOS_HIP_RUNTIME_BUILD_MANIFEST_V1")
}

/// Exact compiled Session kernel identity, not hardware parity evidence.
pub const fn hip_session_build_manifest_v1() -> Option<&'static str> {
    option_env!("NEOETHOS_HIP_SESSION_BUILD_MANIFEST_V1")
}

/// Exact compiled SMC producer identity; does not assert device execution.
pub const fn hip_smc_build_manifest_v1() -> Option<&'static str> {
    option_env!("NEOETHOS_HIP_SMC_BUILD_MANIFEST_V1")
}

/// Source closure and artifact identity of the shared native HIP kernel archive.
/// This is not a CPU/CUDA/HIP parity receipt or a Search input admission.
pub const fn hip_native_build_manifest_v1() -> Option<&'static str> {
    option_env!("NEOETHOS_HIP_NATIVE_BUILD_MANIFEST_V1")
}

const ABI_VERSION_V1: u32 = 1;
const HIP_AMD_BACKEND_V1: u32 = 2;
const STATUS_OK_V1: i32 = 0;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RawHipRuntimeFactsV1 {
    pub(crate) abi_version: u32,
    pub(crate) backend_kind: u32,
    pub(crate) lease_id: u64,
    pub(crate) device_ordinal: i32,
    pub(crate) runtime_version: i32,
    pub(crate) driver_version: i32,
    pub(crate) warp_size: u32,
    pub(crate) uuid: [u8; 16],
    pub(crate) stream_handle: u64,
    pub(crate) stream_id: u64,
    pub(crate) free_memory_bytes: u64,
    pub(crate) total_memory_bytes: u64,
    pub(crate) current_pool_handle: u64,
    pub(crate) default_pool_handle: u64,
    pub(crate) pool_reserved_bytes: u64,
    pub(crate) pool_used_bytes: u64,
    pub(crate) architecture: [u8; 256],
}

impl RawHipRuntimeFactsV1 {
    pub(crate) fn empty() -> Self {
        // Every field is an integer or an integer array; zero is a valid value.
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Default)]
struct RawHipRuntimeErrorV1 {
    abi_version: u32,
    operation: u32,
    backend_status: i32,
    reserved: u32,
}

unsafe extern "C" {
    fn neoethos_hip_runtime_lease_create_v1(
        ordinal: i32,
        lease: *mut u64,
        facts: *mut RawHipRuntimeFactsV1,
        error: *mut RawHipRuntimeErrorV1,
    ) -> i32;
    fn neoethos_hip_runtime_lease_query_v1(
        lease: u64,
        facts: *mut RawHipRuntimeFactsV1,
        error: *mut RawHipRuntimeErrorV1,
    ) -> i32;
    fn neoethos_hip_runtime_lease_synchronize_v1(
        lease: u64,
        error: *mut RawHipRuntimeErrorV1,
    ) -> i32;
    fn neoethos_hip_runtime_lease_close_v1(lease: u64, error: *mut RawHipRuntimeErrorV1) -> i32;
    fn neoethos_hip_runtime_buffer_create_v1(
        lease: u64,
        bytes: u64,
        upload: *const u8,
        buffer: *mut u64,
        error: *mut RawHipRuntimeErrorV1,
    ) -> i32;
    fn neoethos_hip_runtime_buffer_free_v1(
        lease: u64,
        buffer: u64,
        error: *mut RawHipRuntimeErrorV1,
    ) -> i32;
    fn neoethos_hip_runtime_buffer_read_v1(
        lease: u64,
        buffer: u64,
        output: *mut u8,
        bytes: u64,
        error: *mut RawHipRuntimeErrorV1,
    ) -> i32;
    #[cfg(feature = "hip-session-kernels")]
    fn neoethos_hip_runtime_session_f64_v2(
        lease: u64,
        rows: u64,
        buffers: *const u64,
        error: *mut RawHipRuntimeErrorV1,
    ) -> i32;
    #[cfg(feature = "hip-native-kernels")]
    fn neoethos_hip_runtime_smc_parent_f64_v3(
        lease: u64,
        rows: u64,
        inputs: *const u64,
        outputs: *const u64,
        hashes: *mut u8,
        error: *mut RawHipRuntimeErrorV1,
    ) -> i32;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HipRuntimeErrorV1 {
    InvalidInput(&'static str),
    OrdinalOutOfRange,
    InvalidNativeFacts(&'static str),
    Quarantined,
    Native {
        operation: &'static str,
        status: i32,
        native_operation: u32,
        backend_status: i32,
        diagnostic_header_valid: bool,
    },
}

impl fmt::Display for HipRuntimeErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(reason) => write!(f, "invalid HIP input: {reason}"),
            Self::OrdinalOutOfRange => f.write_str("HIP ordinal exceeds the native i32 ABI"),
            Self::InvalidNativeFacts(reason) => write!(f, "invalid HIP runtime facts: {reason}"),
            Self::Quarantined => {
                f.write_str("HIP lease is quarantined; no backend retry is allowed")
            }
            Self::Native {
                operation,
                status,
                native_operation,
                backend_status,
                diagnostic_header_valid,
            } => {
                write!(
                    f,
                    "HIP {operation} failed: status={status}, native_operation={native_operation}, backend_status={backend_status}, diagnostic_header_valid={diagnostic_header_valid}"
                )
            }
        }
    }
}

impl std::error::Error for HipRuntimeErrorV1 {}

fn native_error_v1(
    operation: &'static str,
    status: i32,
    error: &RawHipRuntimeErrorV1,
) -> HipRuntimeErrorV1 {
    HipRuntimeErrorV1::Native {
        operation,
        status,
        native_operation: error.operation,
        backend_status: error.backend_status,
        diagnostic_header_valid: error.abi_version == ABI_VERSION_V1 && error.reserved == 0,
    }
}

/// Immutable facts about one genuine HIP owner, not an allocation capability.
#[derive(Clone, PartialEq, Eq)]
pub struct HipRuntimeIdentityV1 {
    lease_id: NonZeroU64,
    device_ordinal: u32,
    runtime_version: i32,
    driver_version: i32,
    warp_size: u32,
    uuid: [u8; 16],
    architecture: String,
    stream_id: u64,
    total_memory_bytes: u64,
    // Keep raw native handles private, including in Debug output.
    stream_handle: u64,
    current_pool_handle: u64,
    default_pool_handle: u64,
}

impl fmt::Debug for HipRuntimeIdentityV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HipRuntimeIdentityV1")
            .field("schema", &HIP_RUNTIME_IDENTITY_SCHEMA_V1)
            .field("lease_id", &self.lease_id)
            .field("device_ordinal", &self.device_ordinal)
            .field("uuid", &self.uuid)
            .field("architecture", &self.architecture)
            .field("stream_id", &self.stream_id)
            .field("runtime_version", &self.runtime_version)
            .field("driver_version", &self.driver_version)
            .field("warp_size", &self.warp_size)
            .field("total_memory_bytes", &self.total_memory_bytes)
            .finish_non_exhaustive()
    }
}

impl HipRuntimeIdentityV1 {
    pub const fn schema(&self) -> &'static str {
        HIP_RUNTIME_IDENTITY_SCHEMA_V1
    }
    pub const fn lease_id(&self) -> u64 {
        self.lease_id.get()
    }
    pub const fn device_ordinal(&self) -> u32 {
        self.device_ordinal
    }
    pub const fn runtime_version(&self) -> i32 {
        self.runtime_version
    }
    pub const fn driver_version(&self) -> i32 {
        self.driver_version
    }
    pub const fn warp_size(&self) -> u32 {
        self.warp_size
    }
    pub const fn device_uuid(&self) -> [u8; 16] {
        self.uuid
    }
    pub fn architecture(&self) -> &str {
        &self.architecture
    }
    pub const fn stream_id(&self) -> u64 {
        self.stream_id
    }
    pub const fn total_memory_bytes(&self) -> u64 {
        self.total_memory_bytes
    }
}

/// A measured snapshot, not a reservation or permission to allocate its bytes.
/// Pool counters are queried separately, not atomically: concurrent allocations
/// can make used bytes exceed the earlier reserved-byte observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HipRuntimeMemorySnapshotV1 {
    lease_id: u64,
    free_memory_bytes: u64,
    total_memory_bytes: u64,
    pool_reserved_bytes: u64,
    pool_used_bytes: u64,
}

impl HipRuntimeMemorySnapshotV1 {
    pub const fn lease_id(&self) -> u64 {
        self.lease_id
    }
    pub const fn free_memory_bytes(&self) -> u64 {
        self.free_memory_bytes
    }
    pub const fn total_memory_bytes(&self) -> u64 {
        self.total_memory_bytes
    }
    pub const fn pool_reserved_bytes(&self) -> u64 {
        self.pool_reserved_bytes
    }
    pub const fn pool_used_bytes(&self) -> u64 {
        self.pool_used_bytes
    }
}

pub(crate) fn validate_facts_v1(
    raw: &RawHipRuntimeFactsV1,
    lease_id: NonZeroU64,
    ordinal: u32,
) -> Result<(HipRuntimeIdentityV1, HipRuntimeMemorySnapshotV1), HipRuntimeErrorV1> {
    let invalid = HipRuntimeErrorV1::InvalidNativeFacts;
    if raw.abi_version != ABI_VERSION_V1 || raw.backend_kind != HIP_AMD_BACKEND_V1 {
        return Err(invalid("ABI or backend is not HIP AMD v1"));
    }
    if raw.lease_id != lease_id.get() || u32::try_from(raw.device_ordinal).ok() != Some(ordinal) {
        return Err(invalid(
            "lease or device does not match the requested owner",
        ));
    }
    if raw.runtime_version <= 0 || raw.driver_version <= 0 || raw.warp_size == 0 {
        return Err(invalid("runtime, driver, or warp identity is empty"));
    }
    if raw.uuid == [0; 16] || raw.stream_handle <= 2 || raw.stream_id == 0 {
        return Err(invalid("device or stream identity is empty"));
    }
    if raw.current_pool_handle == 0 || raw.current_pool_handle != raw.default_pool_handle {
        return Err(invalid("the run does not retain the default device pool"));
    }
    // Fully occupied memory is a valid observation. Never fabricate headroom.
    if raw.total_memory_bytes == 0 || raw.free_memory_bytes > raw.total_memory_bytes {
        return Err(invalid("memory snapshot is inconsistent"));
    }
    let end = raw
        .architecture
        .iter()
        .position(|&byte| byte == 0)
        .ok_or(invalid("architecture is not NUL terminated"))?;
    if raw.architecture[end..].iter().any(|&byte| byte != 0) {
        return Err(invalid("architecture contains noncanonical trailing bytes"));
    }
    let architecture = std::str::from_utf8(&raw.architecture[..end])
        .map_err(|_| invalid("architecture is not UTF-8"))?;
    let base = architecture.split(':').next().unwrap_or_default();
    let suffix = base
        .strip_prefix("gfx")
        .ok_or(invalid("architecture is not an AMD gfx target"))?;
    if suffix.is_empty()
        || !suffix.bytes().all(|byte| byte.is_ascii_hexdigit())
        || !architecture
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b":+-_".contains(&byte))
    {
        return Err(invalid("architecture is malformed"));
    }
    Ok((
        HipRuntimeIdentityV1 {
            lease_id,
            device_ordinal: ordinal,
            runtime_version: raw.runtime_version,
            driver_version: raw.driver_version,
            warp_size: raw.warp_size,
            uuid: raw.uuid,
            architecture: architecture.to_owned(),
            stream_id: raw.stream_id,
            total_memory_bytes: raw.total_memory_bytes,
            stream_handle: raw.stream_handle,
            current_pool_handle: raw.current_pool_handle,
            default_pool_handle: raw.default_pool_handle,
        },
        HipRuntimeMemorySnapshotV1 {
            lease_id: lease_id.get(),
            free_memory_bytes: raw.free_memory_bytes,
            total_memory_bytes: raw.total_memory_bytes,
            pool_reserved_bytes: raw.pool_reserved_bytes,
            pool_used_bytes: raw.pool_used_bytes,
        },
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeaseStateV1 {
    Active,
    Quarantined,
    Closed,
}

/// Unique, thread-affine owner of an actual HIP stream on its validated device.
///
/// This is deliberately neither Clone, Send, nor Sync. Native errors quarantine
/// the lease permanently; no method or destructor retries ambiguous destruction.
#[derive(Debug)]
#[must_use = "the HIP lease owns runtime resources; explicitly close it to observe cleanup errors"]
pub struct HipRunLeaseV1 {
    identity: HipRuntimeIdentityV1,
    initial_memory: HipRuntimeMemorySnapshotV1,
    state: Cell<LeaseStateV1>,
    _thread_affine: PhantomData<Rc<()>>,
}

impl HipRunLeaseV1 {
    pub fn acquire(ordinal: u32) -> Result<Self, HipRuntimeErrorV1> {
        let native_ordinal =
            i32::try_from(ordinal).map_err(|_| HipRuntimeErrorV1::OrdinalOutOfRange)?;
        let mut lease = 0;
        let mut facts = RawHipRuntimeFactsV1::empty();
        let mut error = RawHipRuntimeErrorV1::default();
        // SAFETY: all outputs are valid exclusive stack slots. Native creation
        // publishes a registry key only after owning and validating its resources.
        let status = unsafe {
            neoethos_hip_runtime_lease_create_v1(native_ordinal, &mut lease, &mut facts, &mut error)
        };
        if status != STATUS_OK_V1 {
            return Err(native_error_v1("create", status, &error));
        }
        let lease = NonZeroU64::new(lease).ok_or(HipRuntimeErrorV1::InvalidNativeFacts(
            "successful creation returned a zero lease",
        ))?;
        // Malformed success is not authority to destroy an unknown native key.
        // The native registry retains it rather than risking an invalid release.
        let (identity, initial_memory) = validate_facts_v1(&facts, lease, ordinal)?;
        Ok(Self {
            identity,
            initial_memory,
            state: Cell::new(LeaseStateV1::Active),
            _thread_affine: PhantomData,
        })
    }

    pub fn identity(&self) -> &HipRuntimeIdentityV1 {
        &self.identity
    }
    pub const fn initial_memory_snapshot(&self) -> HipRuntimeMemorySnapshotV1 {
        self.initial_memory
    }
    pub fn is_quarantined(&self) -> bool {
        self.state.get() == LeaseStateV1::Quarantined
    }

    /// Revalidate the exact owner and return newly measured memory, without
    /// replacing immutable identity or treating available bytes as a reservation.
    pub fn revalidate(&self) -> Result<HipRuntimeMemorySnapshotV1, HipRuntimeErrorV1> {
        self.require_active()?;
        let mut facts = RawHipRuntimeFactsV1::empty();
        let mut error = RawHipRuntimeErrorV1::default();
        // SAFETY: the private key belongs to this unique owner and outputs are valid.
        let status = unsafe {
            neoethos_hip_runtime_lease_query_v1(self.identity.lease_id(), &mut facts, &mut error)
        };
        if status != STATUS_OK_V1 {
            self.state.set(LeaseStateV1::Quarantined);
            return Err(native_error_v1("query", status, &error));
        }
        match validate_facts_v1(&facts, self.identity.lease_id, self.identity.device_ordinal) {
            Ok((identity, memory)) if identity == self.identity => Ok(memory),
            result => {
                self.state.set(LeaseStateV1::Quarantined);
                Err(result
                    .err()
                    .unwrap_or(HipRuntimeErrorV1::InvalidNativeFacts(
                        "immutable owner identity changed",
                    )))
            }
        }
    }

    /// Wait for this run stream. Completion is not evidence of numerical parity.
    pub fn synchronize(&self) -> Result<(), HipRuntimeErrorV1> {
        self.require_active()?;
        let mut error = RawHipRuntimeErrorV1::default();
        // SAFETY: the native registry owns the resources for this private key.
        let status = unsafe {
            neoethos_hip_runtime_lease_synchronize_v1(self.identity.lease_id(), &mut error)
        };
        if status == STATUS_OK_V1 {
            Ok(())
        } else {
            self.state.set(LeaseStateV1::Quarantined);
            Err(native_error_v1("synchronize", status, &error))
        }
    }

    pub fn try_close(self) -> Result<(), HipRuntimeCloseErrorV1> {
        if let Err(error) = self.require_active() {
            return Err(HipRuntimeCloseErrorV1 { error, owner: self });
        }
        // Disarm before FFI: even an ambiguous destruction result is never retried.
        self.state.set(LeaseStateV1::Quarantined);
        let mut error = RawHipRuntimeErrorV1::default();
        // SAFETY: native close validates the registry key and owns all destruction.
        let status =
            unsafe { neoethos_hip_runtime_lease_close_v1(self.identity.lease_id(), &mut error) };
        if status == STATUS_OK_V1 {
            self.state.set(LeaseStateV1::Closed);
            Ok(())
        } else {
            if status == -7 {
                // Native close refused before touching resources: a live native
                // consumer still owns a borrow. The returned owner is reusable.
                self.state.set(LeaseStateV1::Active);
            }
            Err(HipRuntimeCloseErrorV1 {
                error: native_error_v1("close", status, &error),
                owner: self,
            })
        }
    }

    fn require_active(&self) -> Result<(), HipRuntimeErrorV1> {
        if self.state.get() == LeaseStateV1::Active {
            Ok(())
        } else {
            Err(HipRuntimeErrorV1::Quarantined)
        }
    }

    fn resource_status(
        &self,
        operation: &'static str,
        status: i32,
        error: &RawHipRuntimeErrorV1,
    ) -> Result<(), HipRuntimeErrorV1> {
        if status == STATUS_OK_V1 {
            return Ok(());
        }
        // Native -6 is a definite allocation-capacity refusal, not a driver
        // fault. -1 rejects arguments before submission. -8 is a completed
        // producer semantic refusal; it publishes no accepted outputs and is
        // not a runtime fault.
        if status != -6 && status != -1 && status != -8 {
            self.state.set(LeaseStateV1::Quarantined);
        }
        Err(native_error_v1(operation, status, error))
    }

    fn create_buffer(
        &self,
        bytes: usize,
        upload: Option<&[u8]>,
    ) -> Result<HipDeviceBufferV1<'_>, HipRuntimeErrorV1> {
        self.require_active()?;
        if bytes == 0 || upload.is_some_and(|slice| slice.len() != bytes) {
            return Err(HipRuntimeErrorV1::InvalidInput(
                "buffer size must be nonzero and exact",
            ));
        }
        let mut key = 0;
        let mut error = RawHipRuntimeErrorV1::default();
        // SAFETY: input is valid for this call; native code stages it in its own
        // pinned allocation before returning, including asynchronous failures.
        let status = unsafe {
            neoethos_hip_runtime_buffer_create_v1(
                self.identity.lease_id(),
                bytes as u64,
                upload.map_or(std::ptr::null(), |slice| slice.as_ptr()),
                &mut key,
                &mut error,
            )
        };
        self.resource_status("allocate/upload", status, &error)?;
        let Some(key) = NonZeroU64::new(key) else {
            self.state.set(LeaseStateV1::Quarantined);
            return Err(HipRuntimeErrorV1::InvalidNativeFacts(
                "successful allocation has no buffer key",
            ));
        };
        Ok(HipDeviceBufferV1 {
            owner: self,
            key,
            bytes,
            active: true,
        })
    }

    /// Allocate an output. Readback is refused until a typed producer writes it.
    pub fn allocate_bytes(&self, bytes: usize) -> Result<HipDeviceBufferV1<'_>, HipRuntimeErrorV1> {
        self.create_buffer(bytes, None)
    }

    /// Stage once and enqueue H2D on this stream; does not synchronize the GPU.
    pub fn upload_bytes(&self, bytes: &[u8]) -> Result<HipDeviceBufferV1<'_>, HipRuntimeErrorV1> {
        self.create_buffer(bytes.len(), Some(bytes))
    }

    /// Execute the existing f64 Session-v2 kernel, retaining all data on device.
    #[cfg(feature = "hip-session-kernels")]
    pub fn launch_session_f64_v2(
        &self,
        rows: usize,
        lanes: [&HipDeviceBufferV1<'_>; 6],
        values: &HipDeviceBufferV1<'_>,
        validity: &HipDeviceBufferV1<'_>,
    ) -> Result<(), HipRuntimeErrorV1> {
        self.require_active()?;
        let sizes =
            rows.checked_mul(184)
                .filter(|_| rows > 0)
                .ok_or(HipRuntimeErrorV1::InvalidInput(
                    "Session row count overflows",
                ))?;
        let all = [
            lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5], values, validity,
        ];
        for (index, buffer) in all.iter().enumerate() {
            let expected = if index < 6 {
                rows * 8
            } else if index == 6 {
                sizes
            } else {
                rows * 23
            };
            if !std::ptr::eq(buffer.owner, self) || !buffer.active || buffer.bytes != expected {
                return Err(HipRuntimeErrorV1::InvalidInput(
                    "Session buffers must have exact size and same lease",
                ));
            }
        }
        let keys = all.map(|buffer| buffer.key.get());
        let mut error = RawHipRuntimeErrorV1::default();
        // SAFETY: no caller pointers reach the device ABI. Native registry
        // resolves and checks eight same-lease buffers before enqueueing work.
        let status = unsafe {
            neoethos_hip_runtime_session_f64_v2(
                self.identity.lease_id(),
                rows as u64,
                keys.as_ptr(),
                &mut error,
            )
        };
        self.resource_status("Session-v2 launch", status, &error)
    }

    /// Run the original SMC-v3 producer on existing same-lease OHLCV buffers.
    /// Only 100 control bytes cross to the host: completed device error, then
    /// the three 32-byte producer hashes. Native code publishes outputs only
    /// after both completion boundaries succeed and the semantic error is zero.
    #[cfg(feature = "hip-native-kernels")]
    pub fn launch_smc_parent_f64_v3(
        &self,
        rows: usize,
        inputs: [&HipDeviceBufferV1<'_>; 5],
        outputs: [&HipDeviceBufferV1<'_>; 7],
    ) -> Result<[u8; 96], HipRuntimeErrorV1> {
        self.require_active()?;
        let extents = smc_output_extents_v3(rows)?;
        let input_keys = self.checked_buffer_keys_v1(inputs, [rows * 8; 5])?;
        let output_keys = self.checked_buffer_keys_v1(outputs, extents)?;
        let mut hashes = [0; 96];
        let mut error = RawHipRuntimeErrorV1::default();
        // SAFETY: arrays are exact stack extents, keys are private owned values.
        // No user pointer is retained by async work: native readback stages its
        // control data in native pinned storage until completion is known.
        let status = unsafe {
            neoethos_hip_runtime_smc_parent_f64_v3(
                self.identity.lease_id(),
                rows as u64,
                input_keys.as_ptr(),
                output_keys.as_ptr(),
                hashes.as_mut_ptr(),
                &mut error,
            )
        };
        self.resource_status("SMC-v3 completed seal", status, &error)?;
        Ok(hashes)
    }

    #[cfg(feature = "hip-native-kernels")]
    fn checked_buffer_keys_v1<const N: usize>(
        &self,
        buffers: [&HipDeviceBufferV1<'_>; N],
        expected_bytes: [usize; N],
    ) -> Result<[u64; N], HipRuntimeErrorV1> {
        self.require_active()?;
        for (buffer, expected) in buffers.iter().zip(expected_bytes) {
            if expected == 0
                || !std::ptr::eq(buffer.owner, self)
                || !buffer.active
                || buffer.bytes != expected
            {
                return Err(HipRuntimeErrorV1::InvalidInput(
                    "buffers must be active, exact sized, and owned by this lease",
                ));
            }
        }
        Ok(buffers.map(|buffer| buffer.key.get()))
    }

    #[cfg(all(test, feature = "hip-device-fixtures"))]
    pub(crate) fn checked_population_keys_v1<const N: usize>(
        &self,
        buffers: [&HipDeviceBufferV1<'_>; N],
        expected_bytes: [usize; N],
    ) -> Result<[u64; N], HipRuntimeErrorV1> {
        self.checked_buffer_keys_v1(buffers, expected_bytes)
    }

    #[cfg(all(test, feature = "hip-device-fixtures"))]
    pub(crate) fn quarantine_after_population_failure_v1(&self) {
        self.state.set(LeaseStateV1::Quarantined);
    }
}

#[cfg(feature = "hip-native-kernels")]
fn smc_output_extents_v3(rows: usize) -> Result<[usize; 7], HipRuntimeErrorV1> {
    // The unchanged kernel stores FVG/OB birth rows in signed int. This is an
    // actual representation boundary, not a hardware-independent search budget.
    if rows == 0 || rows - 1 > i32::MAX as usize {
        return Err(HipRuntimeErrorV1::InvalidInput(
            "SMC-v3 last row must fit its signed 32-bit birth index",
        ));
    }
    let values = rows
        .checked_mul(368)
        .ok_or(HipRuntimeErrorV1::InvalidInput(
            "SMC-v3 output size overflows",
        ))?;
    Ok([values, rows * 46, rows * 8, rows * 8, rows * 11, 96, 4])
}

impl Drop for HipRunLeaseV1 {
    fn drop(&mut self) {
        if self.state.get() != LeaseStateV1::Active {
            return;
        }
        self.state.set(LeaseStateV1::Quarantined);
        let mut error = RawHipRuntimeErrorV1::default();
        // SAFETY: one best-effort native close. On any failure the native registry
        // retains/quarantines resources; no raw resource destructor exists in Rust.
        let status =
            unsafe { neoethos_hip_runtime_lease_close_v1(self.identity.lease_id(), &mut error) };
        if status == STATUS_OK_V1 {
            self.state.set(LeaseStateV1::Closed);
        } else {
            // Explicit close returns this error. Drop cannot return it, so keep
            // one diagnostic without panicking or attempting another release.
            let _ = writeln!(
                std::io::stderr().lock(),
                "neoethos HIP lease retained after Drop cleanup failure: {}",
                native_error_v1("close", status, &error)
            );
        }
    }
}

/// One native-owned allocation borrowing its run. There are no public raw
/// pointers or constructors, and the run cannot be closed while it is borrowed.
#[derive(Debug)]
#[must_use]
pub struct HipDeviceBufferV1<'lease> {
    owner: &'lease HipRunLeaseV1,
    key: NonZeroU64,
    bytes: usize,
    active: bool,
}

impl HipDeviceBufferV1<'_> {
    pub const fn len_bytes(&self) -> usize {
        self.bytes
    }
    pub fn lease_id(&self) -> u64 {
        self.owner.identity.lease_id()
    }

    /// Explicit diagnostic/terminal boundary. Waits only for the owning stream;
    /// ordinary producer composition never needs a feature download.
    pub fn read_bytes(&self) -> Result<Vec<u8>, HipRuntimeErrorV1> {
        self.owner.require_active()?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(self.bytes)
            .map_err(|_| HipRuntimeErrorV1::InvalidInput("host readback allocation failed"))?;
        bytes.resize(self.bytes, 0);
        let mut error = RawHipRuntimeErrorV1::default();
        // SAFETY: exact writable capacity. On failed async copy the native
        // pinned staging, not this Rust vector, remains referenced by the GPU.
        let status = unsafe {
            neoethos_hip_runtime_buffer_read_v1(
                self.lease_id(),
                self.key.get(),
                bytes.as_mut_ptr(),
                self.bytes as u64,
                &mut error,
            )
        };
        self.owner.resource_status("readback", status, &error)?;
        Ok(bytes)
    }

    pub fn try_close(mut self) -> Result<(), HipRuntimeErrorV1> {
        self.release()
    }

    fn release(&mut self) -> Result<(), HipRuntimeErrorV1> {
        if !self.active {
            return Ok(());
        }
        self.active = false; // Never retry uncertain destruction from Drop.
        self.owner.require_active()?;
        let mut error = RawHipRuntimeErrorV1::default();
        // SAFETY: private key and still-borrowed owner; free is stream ordered.
        let status = unsafe {
            neoethos_hip_runtime_buffer_free_v1(self.lease_id(), self.key.get(), &mut error)
        };
        self.owner.resource_status("buffer free", status, &error)
    }
}

impl Drop for HipDeviceBufferV1<'_> {
    fn drop(&mut self) {
        if self.active
            && let Err(error) = self.release()
        {
            let _ = writeln!(
                std::io::stderr().lock(),
                "neoethos HIP buffer retained: {error}"
            );
        }
    }
}

/// Failed close retains its owner. Only an explicit native BUSY refusal leaves
/// it active for a later close; uncertain destruction is never retried.
#[derive(Debug)]
pub struct HipRuntimeCloseErrorV1 {
    error: HipRuntimeErrorV1,
    owner: HipRunLeaseV1,
}

impl HipRuntimeCloseErrorV1 {
    pub fn error(&self) -> &HipRuntimeErrorV1 {
        &self.error
    }
    pub fn owner(&self) -> &HipRunLeaseV1 {
        &self.owner
    }
    pub fn into_parts(self) -> (HipRuntimeErrorV1, HipRunLeaseV1) {
        (self.error, self.owner)
    }
}

impl fmt::Display for HipRuntimeCloseErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}
impl std::error::Error for HipRuntimeCloseErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts() -> RawHipRuntimeFactsV1 {
        let mut raw = RawHipRuntimeFactsV1::empty();
        raw.abi_version = 1;
        raw.backend_kind = 2;
        raw.lease_id = 7;
        raw.device_ordinal = 1;
        raw.runtime_version = 70_200_000;
        raw.driver_version = 70_200_000;
        raw.warp_size = 64;
        raw.uuid = [9; 16];
        raw.stream_handle = 12;
        raw.stream_id = 13;
        raw.free_memory_bytes = 100;
        raw.total_memory_bytes = 200;
        raw.current_pool_handle = 14;
        raw.default_pool_handle = 14;
        raw.pool_reserved_bytes = 30;
        raw.pool_used_bytes = 20;
        raw.architecture[..6].copy_from_slice(b"gfx942");
        raw
    }

    fn decode(
        raw: &RawHipRuntimeFactsV1,
    ) -> Result<(HipRuntimeIdentityV1, HipRuntimeMemorySnapshotV1), HipRuntimeErrorV1> {
        validate_facts_v1(raw, NonZeroU64::new(7).unwrap(), 1)
    }

    #[test]
    fn hip_runtime_native_layout_is_exact() {
        use std::mem::{align_of, offset_of, size_of};
        assert_eq!(size_of::<RawHipRuntimeFactsV1>(), 368);
        assert_eq!(align_of::<RawHipRuntimeFactsV1>(), 8);
        assert_eq!(offset_of!(RawHipRuntimeFactsV1, lease_id), 8);
        assert_eq!(offset_of!(RawHipRuntimeFactsV1, uuid), 32);
        assert_eq!(offset_of!(RawHipRuntimeFactsV1, stream_handle), 48);
        assert_eq!(offset_of!(RawHipRuntimeFactsV1, stream_id), 56);
        assert_eq!(offset_of!(RawHipRuntimeFactsV1, architecture), 112);
        assert_eq!(size_of::<RawHipRuntimeErrorV1>(), 16);
        assert_eq!(offset_of!(RawHipRuntimeErrorV1, backend_status), 8);
    }

    #[test]
    #[cfg(feature = "hip-native-kernels")]
    fn hip_smc_extents_match_original_kernel_and_signed_birth_indices() {
        assert!(smc_output_extents_v3(0).is_err());
        assert_eq!(
            smc_output_extents_v3(1).unwrap(),
            [368, 46, 8, 8, 11, 96, 4]
        );
        let extents = smc_output_extents_v3(31).unwrap();
        assert_eq!(extents.iter().sum::<usize>(), 31 * 441 + 100);
        assert!(smc_output_extents_v3(i32::MAX as usize + 1).is_ok());
        assert!(smc_output_extents_v3(i32::MAX as usize + 2).is_err());
        assert!(smc_output_extents_v3(usize::MAX).is_err());
    }

    #[test]
    #[cfg(feature = "hip-native-kernels")]
    fn hip_native_linked_manifest_matches_rust_and_never_claims_device_execution() {
        use sha2::{Digest, Sha256};
        unsafe extern "C" {
            fn neoethos_hip_native_build_manifest_sha256_v1() -> *const u8;
        }
        let raw = hip_native_build_manifest_v1().expect("native build metadata missing");
        let manifest: serde_json::Value = serde_json::from_str(raw).unwrap();
        assert_eq!(manifest["schema"], "neoethos.hip-native-kernels-build.v1");
        assert_eq!(manifest["backend"], "amd-hip");
        assert_eq!(manifest["device_executed"], false);
        assert_eq!(
            manifest["device_fixtures"],
            cfg!(feature = "hip-device-fixtures")
        );
        // SAFETY: the linked native manifest accessor returns a static 32-byte
        // array; this host-only check neither acquires nor executes a device.
        let linked = unsafe { neoethos_hip_native_build_manifest_sha256_v1() };
        assert!(!linked.is_null());
        let linked = unsafe { std::slice::from_raw_parts(linked, 32) };
        assert_eq!(linked, Sha256::digest(raw.as_bytes()).as_slice());
        let smc: serde_json::Value =
            serde_json::from_str(hip_smc_build_manifest_v1().unwrap()).unwrap();
        assert_eq!(smc["schema"], "neoethos.hip-smc-kernels-build.v1");
        assert_eq!(smc["semantic_version"], 3);
        assert_eq!(smc["artifact_sha256"], manifest["artifact_sha256"]);
        assert_eq!(smc["device_executed"], false);
        assert_eq!(smc["target"], manifest["target"]);
    }

    #[test]
    fn hip_runtime_memory_changes_do_not_rewrite_identity() {
        let mut raw = facts();
        let (identity, initial) = decode(&raw).unwrap();
        raw.free_memory_bytes = 0;
        raw.pool_reserved_bytes = 100;
        raw.pool_used_bytes = 90;
        let (later_identity, later) = decode(&raw).unwrap();
        assert_eq!(identity, later_identity);
        assert_ne!(initial, later);
        assert_eq!(later.free_memory_bytes(), 0);
        assert_eq!(identity.architecture(), "gfx942");
        assert!(!format!("{identity:?}").contains("stream_handle"));
    }

    #[test]
    fn hip_runtime_accepts_non_atomic_pool_counter_observations() {
        let mut raw = facts();
        let identity = decode(&raw).unwrap().0;
        raw.pool_reserved_bytes = 30;
        raw.pool_used_bytes = 31;
        let (later_identity, memory) = decode(&raw).unwrap();
        assert_eq!(identity, later_identity);
        assert_eq!(memory.pool_reserved_bytes(), 30);
        assert_eq!(memory.pool_used_bytes(), 31);
    }

    #[test]
    fn hip_runtime_facts_reject_detached_and_malformed_authority() {
        let mutations: &[fn(&mut RawHipRuntimeFactsV1)] = &[
            |v| v.abi_version = 2,
            |v| v.backend_kind = 1,
            |v| v.lease_id = 8,
            |v| v.device_ordinal = -1,
            |v| v.device_ordinal = 0,
            |v| v.runtime_version = 0,
            |v| v.driver_version = -1,
            |v| v.warp_size = 0,
            |v| v.uuid = [0; 16],
            |v| v.stream_handle = 0,
            |v| v.stream_handle = 1,
            |v| v.stream_handle = 2,
            |v| v.stream_id = 0,
            |v| v.current_pool_handle = 0,
            |v| v.default_pool_handle = 15,
            |v| v.total_memory_bytes = 0,
            |v| v.free_memory_bytes = 201,
            |v| v.architecture = [b'x'; 256],
            |v| v.architecture[0] = 0xff,
            |v| v.architecture[8] = b'x',
            |v| v.architecture[..6].copy_from_slice(b"sm_120"),
        ];
        for (index, mutate) in mutations.iter().enumerate() {
            let mut raw = facts();
            mutate(&mut raw);
            assert!(decode(&raw).is_err(), "accepted invalid case {index}");
        }
    }

    #[test]
    fn hip_runtime_immutable_tuple_includes_private_handles_and_build_device_facts() {
        let identity = decode(&facts()).unwrap().0;
        let mutations: &[fn(&mut RawHipRuntimeFactsV1)] = &[
            |v| v.stream_handle += 1,
            |v| v.stream_id += 1,
            |v| {
                v.current_pool_handle += 1;
                v.default_pool_handle += 1;
            },
            |v| v.uuid[0] += 1,
            |v| v.total_memory_bytes += 1,
            |v| v.runtime_version += 1,
            |v| v.driver_version += 1,
            |v| v.warp_size = 32,
            |v| v.architecture[5] = b'1',
        ];
        for mutate in mutations {
            let mut raw = facts();
            mutate(&mut raw);
            assert_ne!(identity, decode(&raw).unwrap().0);
        }
    }

    #[test]
    fn hip_runtime_native_errors_preserve_backend_status_and_bad_diagnostics() {
        let raw = RawHipRuntimeErrorV1 {
            abi_version: 2,
            operation: 17,
            backend_status: 999,
            reserved: 1,
        };
        assert_eq!(
            native_error_v1("query", -4, &raw),
            HipRuntimeErrorV1::Native {
                operation: "query",
                status: -4,
                native_operation: 17,
                backend_status: 999,
                diagnostic_header_valid: false,
            }
        );
    }
}
