//! One-shot, vendor-neutral Discovery run-device admission.

#[cfg(feature = "cuda")]
use crate::cuda_build_manifest_v1;
#[cfg(feature = "cuda")]
use crate::physical_gpu_inventory_v1::PhysicalGpuInventoryRecordV1;
use crate::physical_gpu_inventory_v1::{
    PhysicalGpuInventoryErrorV1, SealedNoPhysicalGpuReceiptV1, SealedPhysicalGpuInventoryReceiptV1,
    probe_physical_gpu_inventory_v1, seal_no_physical_gpu_receipt_v1,
};
use crate::{CudaDeviceEnumerationErrorV1, probe_cuda_device_count_v1};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[cfg(feature = "cuda")]
use cust::context::Context;
#[cfg(feature = "cuda")]
use cust::device::{Device, DeviceAttribute};
#[cfg(feature = "cuda")]
use cust::memory::mem_get_info;
#[cfg(feature = "cuda")]
use cust::stream::{Stream, StreamFlags};
#[cfg(feature = "cuda")]
use cust::sys::cudaError_enum::CUDA_SUCCESS;
#[cfg(feature = "cuda")]
use cust::sys::{CUuuid, cuDeviceGetPCIBusId, cuDeviceGetUuid};
#[cfg(feature = "cuda")]
use cust::{CudaApiVersion, CudaFlags, init};
#[cfg(feature = "cuda")]
use serde_json::Value;
#[cfg(feature = "cuda")]
use std::ffi::CStr;
#[cfg(feature = "cuda")]
use std::sync::Arc;

const RUN_DEVICE_ADMISSION_SCHEMA_V1: &str = "neoethos.discovery-run-device-admission.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscoveryRunDeviceAdmissionErrorCodeV1 {
    PhysicalInventoryFailure,
    BaselineCudaContradiction,
    VisibleGpuWithoutStrictBackend,
    VisibleGpuBuildIncompatible,
    CudaEnumerationFailure,
    NoCompatibleCudaOrdinal,
    NativeDeviceIdentityFailure,
    NativeContextFailure,
    NativeStreamFailure,
    NativeMemorySnapshotFailure,
    NativeBuildManifestFailure,
    ProbeCounterOverflow,
    ProbeCounterMismatch,
}

#[derive(Debug, Error)]
#[error("discovery run-device admission failed ({code:?}): {detail}")]
pub struct DiscoveryRunDeviceAdmissionErrorV1 {
    code: DiscoveryRunDeviceAdmissionErrorCodeV1,
    detail: String,
}

impl DiscoveryRunDeviceAdmissionErrorV1 {
    fn new(code: DiscoveryRunDeviceAdmissionErrorCodeV1, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }

    pub const fn code(&self) -> DiscoveryRunDeviceAdmissionErrorCodeV1 {
        self.code
    }
}

impl From<PhysicalGpuInventoryErrorV1> for DiscoveryRunDeviceAdmissionErrorV1 {
    fn from(error: PhysicalGpuInventoryErrorV1) -> Self {
        Self::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::PhysicalInventoryFailure,
            error.to_string(),
        )
    }
}

#[derive(Debug)]
pub struct RunDeviceAcquisitionCountersV1 {
    physical_inventory_probe_count: u32,
    cuda_enumeration_count: u32,
    primary_context_acquisition_count: u32,
    run_stream_creation_count: u32,
}

impl RunDeviceAcquisitionCountersV1 {
    fn new() -> Self {
        Self {
            physical_inventory_probe_count: 0,
            cuda_enumeration_count: 0,
            primary_context_acquisition_count: 0,
            run_stream_creation_count: 0,
        }
    }

    fn increment(counter: &mut u32) -> Result<(), DiscoveryRunDeviceAdmissionErrorV1> {
        *counter = counter.checked_add(1).ok_or_else(|| {
            DiscoveryRunDeviceAdmissionErrorV1::new(
                DiscoveryRunDeviceAdmissionErrorCodeV1::ProbeCounterOverflow,
                "run-device acquisition counter overflow",
            )
        })?;
        Ok(())
    }

    fn record_physical_inventory_probe_v1(
        &mut self,
    ) -> Result<(), DiscoveryRunDeviceAdmissionErrorV1> {
        Self::increment(&mut self.physical_inventory_probe_count)
    }

    fn record_cuda_enumeration_v1(&mut self) -> Result<(), DiscoveryRunDeviceAdmissionErrorV1> {
        Self::increment(&mut self.cuda_enumeration_count)
    }

    #[cfg(feature = "cuda")]
    fn record_primary_context_acquisition_v1(
        &mut self,
    ) -> Result<(), DiscoveryRunDeviceAdmissionErrorV1> {
        Self::increment(&mut self.primary_context_acquisition_count)
    }

    #[cfg(feature = "cuda")]
    fn record_run_stream_creation_v1(&mut self) -> Result<(), DiscoveryRunDeviceAdmissionErrorV1> {
        Self::increment(&mut self.run_stream_creation_count)
    }

    fn seal_exact_once_v1(&self) -> Result<(), DiscoveryRunDeviceAdmissionErrorV1> {
        let base_exact =
            self.physical_inventory_probe_count == 1 && self.cuda_enumeration_count == 1;
        let route_exact = matches!(
            (
                self.primary_context_acquisition_count,
                self.run_stream_creation_count,
            ),
            (0, 0) | (1, 1)
        );
        if base_exact && route_exact {
            Ok(())
        } else {
            Err(DiscoveryRunDeviceAdmissionErrorV1::new(
                DiscoveryRunDeviceAdmissionErrorCodeV1::ProbeCounterMismatch,
                "run-device admission did not perform exactly one allowed acquisition sequence",
            ))
        }
    }

    pub(crate) fn require_exact_single_run_device_acquisition_v1(
        &self,
    ) -> Result<(), DiscoveryRunDeviceAdmissionErrorV1> {
        self.seal_exact_once_v1()
    }

    pub const fn physical_inventory_probe_count(&self) -> u32 {
        self.physical_inventory_probe_count
    }

    pub const fn cuda_enumeration_count(&self) -> u32 {
        self.cuda_enumeration_count
    }

    pub const fn primary_context_acquisition_count(&self) -> u32 {
        self.primary_context_acquisition_count
    }

    pub const fn run_stream_creation_count(&self) -> u32 {
        self.run_stream_creation_count
    }
}

#[derive(Debug)]
pub struct SealedCpuNoPhysicalGpuRunDeviceAdmissionV1 {
    pub(crate) no_physical_gpu_receipt: SealedNoPhysicalGpuReceiptV1,
    pub(crate) probe_counters: RunDeviceAcquisitionCountersV1,
    pub(crate) admission_identity_sha256: [u8; 32],
}

impl SealedCpuNoPhysicalGpuRunDeviceAdmissionV1 {
    pub const fn no_physical_gpu_receipt(&self) -> &SealedNoPhysicalGpuReceiptV1 {
        &self.no_physical_gpu_receipt
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub(crate) struct SealedCudaNativeBuildIdentityV1 {
    pub(crate) manifest_sha256: [u8; 32],
    pub(crate) artifact_sha256: [u8; 32],
    pub(crate) nvcc_version: String,
    pub(crate) sass_targets: Vec<String>,
    pub(crate) selected_sass_target: String,
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct SealedNativeCudaRunDeviceAdmissionV1 {
    pub(crate) physical_inventory_identity_sha256: [u8; 32],
    pub(crate) pci_identity: PhysicalGpuInventoryRecordV1,
    pub(crate) device_uuid: [u8; 16],
    pub(crate) ordinal: u32,
    pub(crate) run_stream: Arc<Stream>,
    pub(crate) primary_context: Arc<Context>,
    pub(crate) cuda_build_identity: SealedCudaNativeBuildIdentityV1,
    pub(crate) sass_target: String,
    pub(crate) driver_version: String,
    pub(crate) context_api_version: String,
    pub(crate) compute_capability_major: u16,
    pub(crate) compute_capability_minor: u16,
    pub(crate) free_memory_bytes_snapshot: u64,
    pub(crate) probe_counters: RunDeviceAcquisitionCountersV1,
    pub(crate) admission_identity_sha256: [u8; 32],
}

#[derive(Debug)]
pub enum SealedDiscoveryRunDeviceAdmissionV1 {
    CpuNoPhysicalGpu(SealedCpuNoPhysicalGpuRunDeviceAdmissionV1),
    #[cfg(feature = "cuda")]
    NativeCuda(Box<SealedNativeCudaRunDeviceAdmissionV1>),
}

impl SealedDiscoveryRunDeviceAdmissionV1 {
    pub fn probe_counters(&self) -> &RunDeviceAcquisitionCountersV1 {
        match self {
            Self::CpuNoPhysicalGpu(admission) => &admission.probe_counters,
            #[cfg(feature = "cuda")]
            Self::NativeCuda(admission) => &admission.probe_counters,
        }
    }

    pub const fn admission_identity_sha256(&self) -> [u8; 32] {
        match self {
            Self::CpuNoPhysicalGpu(admission) => admission.admission_identity_sha256,
            #[cfg(feature = "cuda")]
            Self::NativeCuda(admission) => admission.admission_identity_sha256,
        }
    }
}

enum CompletePhysicalInventoryV1 {
    CompleteNoPhysicalGpu(SealedPhysicalGpuInventoryReceiptV1),
    CompletePhysicalGpuSet(SealedPhysicalGpuInventoryReceiptV1),
}

enum CudaEnumerationEvidenceV1 {
    ExactCudaDeviceCount(u32),
    NativeAdapterUnavailable,
    RuntimeUnavailable(i32),
    InvalidEnumeration,
}

enum ClassifiedDiscoveryRunDeviceV1 {
    CpuNoPhysicalGpu(SealedNoPhysicalGpuReceiptV1),
    NativeCuda(Box<ClassifiedNativeCudaRunDeviceV1>),
}

struct ClassifiedNativeCudaRunDeviceV1 {
    inventory: SealedPhysicalGpuInventoryReceiptV1,
    candidate: NativeCudaCandidateV1,
}

pub fn acquire_discovery_run_device_admission_v1()
-> Result<SealedDiscoveryRunDeviceAdmissionV1, DiscoveryRunDeviceAdmissionErrorV1> {
    let mut counters = RunDeviceAcquisitionCountersV1::new();
    let inventory = probe_physical_gpu_inventory_v1()?;
    counters.record_physical_inventory_probe_v1()?;
    let cuda_enumeration = probe_cuda_device_count_v1();
    counters.record_cuda_enumeration_v1()?;
    let classified = classify_discovery_run_device_admission_v1(inventory, cuda_enumeration)?;

    match classified {
        ClassifiedDiscoveryRunDeviceV1::CpuNoPhysicalGpu(no_physical_gpu_receipt) => {
            counters.seal_exact_once_v1()?;
            let admission_identity_sha256 = hash_cpu_admission_v1(&no_physical_gpu_receipt);
            Ok(SealedDiscoveryRunDeviceAdmissionV1::CpuNoPhysicalGpu(
                SealedCpuNoPhysicalGpuRunDeviceAdmissionV1 {
                    no_physical_gpu_receipt,
                    probe_counters: counters,
                    admission_identity_sha256,
                },
            ))
        }
        ClassifiedDiscoveryRunDeviceV1::NativeCuda(native) => {
            let ClassifiedNativeCudaRunDeviceV1 {
                inventory,
                candidate,
            } = *native;
            #[cfg(feature = "cuda")]
            {
                let primary_context = retain_primary_context_once_v1(&candidate)?;
                counters.record_primary_context_acquisition_v1()?;
                let run_stream = create_run_stream_once_v1()?;
                counters.record_run_stream_creation_v1()?;
                counters.seal_exact_once_v1()?;
                seal_native_cuda_run_device_admission_v1(
                    inventory,
                    candidate,
                    primary_context,
                    run_stream,
                    counters,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                reject_featureless_native_admission_v1(inventory, candidate, counters)
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn reject_featureless_native_admission_v1(
    _inventory: SealedPhysicalGpuInventoryReceiptV1,
    _candidate: NativeCudaCandidateV1,
    _counters: RunDeviceAcquisitionCountersV1,
) -> Result<SealedDiscoveryRunDeviceAdmissionV1, DiscoveryRunDeviceAdmissionErrorV1> {
    Err(DiscoveryRunDeviceAdmissionErrorV1::new(
        DiscoveryRunDeviceAdmissionErrorCodeV1::VisibleGpuWithoutStrictBackend,
        "strict CUDA admission cannot be sealed without the native adapter",
    ))
}

fn classify_discovery_run_device_admission_v1(
    inventory: SealedPhysicalGpuInventoryReceiptV1,
    cuda_probe: Result<u32, CudaDeviceEnumerationErrorV1>,
) -> Result<ClassifiedDiscoveryRunDeviceV1, DiscoveryRunDeviceAdmissionErrorV1> {
    let physical = if inventory.records().is_empty() {
        CompletePhysicalInventoryV1::CompleteNoPhysicalGpu(inventory)
    } else {
        CompletePhysicalInventoryV1::CompletePhysicalGpuSet(inventory)
    };
    let cuda = match cuda_probe {
        Ok(count) => CudaEnumerationEvidenceV1::ExactCudaDeviceCount(count),
        Err(CudaDeviceEnumerationErrorV1::NativeAdapterUnavailable) => {
            CudaEnumerationEvidenceV1::NativeAdapterUnavailable
        }
        Err(CudaDeviceEnumerationErrorV1::RuntimeFailure(status)) => {
            CudaEnumerationEvidenceV1::RuntimeUnavailable(status)
        }
        Err(CudaDeviceEnumerationErrorV1::InvalidNativeOutput) => {
            CudaEnumerationEvidenceV1::InvalidEnumeration
        }
    };

    match (physical, cuda) {
        (
            CompletePhysicalInventoryV1::CompleteNoPhysicalGpu(inventory),
            CudaEnumerationEvidenceV1::NativeAdapterUnavailable
            | CudaEnumerationEvidenceV1::RuntimeUnavailable(_)
            | CudaEnumerationEvidenceV1::ExactCudaDeviceCount(0),
        ) => Ok(ClassifiedDiscoveryRunDeviceV1::CpuNoPhysicalGpu(
            seal_no_physical_gpu_receipt_v1(inventory)?,
        )),
        (
            CompletePhysicalInventoryV1::CompleteNoPhysicalGpu(_),
            CudaEnumerationEvidenceV1::ExactCudaDeviceCount(_),
        ) => Err(DiscoveryRunDeviceAdmissionErrorV1::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::BaselineCudaContradiction,
            "complete zero-physical-GPU PCI inventory contradicts positive CUDA enumeration",
        )),
        (
            CompletePhysicalInventoryV1::CompleteNoPhysicalGpu(_),
            CudaEnumerationEvidenceV1::InvalidEnumeration,
        ) => Err(DiscoveryRunDeviceAdmissionErrorV1::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::CudaEnumerationFailure,
            "CUDA adapter returned invalid enumeration evidence",
        )),
        (
            CompletePhysicalInventoryV1::CompletePhysicalGpuSet(_),
            CudaEnumerationEvidenceV1::NativeAdapterUnavailable,
        ) => Err(DiscoveryRunDeviceAdmissionErrorV1::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::VisibleGpuWithoutStrictBackend,
            "physical GPU exists but this binary has no strict native CUDA adapter",
        )),
        (
            CompletePhysicalInventoryV1::CompletePhysicalGpuSet(_),
            CudaEnumerationEvidenceV1::RuntimeUnavailable(status),
        ) => Err(DiscoveryRunDeviceAdmissionErrorV1::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::CudaEnumerationFailure,
            format!("physical GPU exists but CUDA enumeration failed with status {status}"),
        )),
        (
            CompletePhysicalInventoryV1::CompletePhysicalGpuSet(_),
            CudaEnumerationEvidenceV1::InvalidEnumeration,
        ) => Err(DiscoveryRunDeviceAdmissionErrorV1::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::CudaEnumerationFailure,
            "physical GPU exists but CUDA enumeration produced invalid output",
        )),
        (
            CompletePhysicalInventoryV1::CompletePhysicalGpuSet(_),
            CudaEnumerationEvidenceV1::ExactCudaDeviceCount(0),
        ) => Err(DiscoveryRunDeviceAdmissionErrorV1::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::NoCompatibleCudaOrdinal,
            "physical GPU exists but CUDA reported no device ordinal",
        )),
        (
            CompletePhysicalInventoryV1::CompletePhysicalGpuSet(inventory),
            CudaEnumerationEvidenceV1::ExactCudaDeviceCount(count),
        ) => {
            let candidate = match select_lowest_compatible_cuda_ordinal_v1(&inventory, count) {
                Err(error)
                    if error.code()
                        == DiscoveryRunDeviceAdmissionErrorCodeV1::VisibleGpuBuildIncompatible =>
                {
                    return Err(error);
                }
                result => result?,
            };
            Ok(ClassifiedDiscoveryRunDeviceV1::NativeCuda(Box::new(
                ClassifiedNativeCudaRunDeviceV1 {
                    inventory,
                    candidate,
                },
            )))
        }
    }
}

fn hash_cpu_admission_v1(receipt: &SealedNoPhysicalGpuReceiptV1) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(RUN_DEVICE_ADMISSION_SCHEMA_V1.as_bytes());
    hasher.update(b"cpu-no-physical-gpu");
    hasher.update(receipt.inventory_identity_sha256());
    hasher.update(receipt.platform().wire_name_for_admission_v1().as_bytes());
    hasher.finalize().into()
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct NativeCudaCandidateV1 {
    device: Device,
    ordinal: u32,
    pci_identity: PhysicalGpuInventoryRecordV1,
    device_uuid: [u8; 16],
    compute_capability_major: u16,
    compute_capability_minor: u16,
    cuda_build_identity: SealedCudaNativeBuildIdentityV1,
}

#[cfg(not(feature = "cuda"))]
struct NativeCudaCandidateV1;

#[cfg(feature = "cuda")]
fn select_lowest_compatible_cuda_ordinal_v1(
    inventory: &SealedPhysicalGpuInventoryReceiptV1,
    count: u32,
) -> Result<NativeCudaCandidateV1, DiscoveryRunDeviceAdmissionErrorV1> {
    init(CudaFlags::empty()).map_err(|error| {
        DiscoveryRunDeviceAdmissionErrorV1::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::CudaEnumerationFailure,
            error.to_string(),
        )
    })?;
    let build = parse_cuda_build_manifest_v1(cuda_build_manifest_v1().ok_or_else(|| {
        DiscoveryRunDeviceAdmissionErrorV1::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::VisibleGpuWithoutStrictBackend,
            "native CUDA build manifest is absent",
        )
    })?)?;
    let mut saw_build_incompatible = false;

    for ordinal in 0..count {
        let device = Device::get_device(ordinal).map_err(native_identity_error_v1)?;
        let (pci_domain, pci_bus, pci_device, pci_function) = cuda_pci_location_v1(device)?;
        let pci_identity = inventory
            .records()
            .iter()
            .find(|record| {
                record.vendor_id() == 0x10de
                    && record.pci_domain() == pci_domain
                    && record.pci_bus() == pci_bus
                    && record.pci_device() == pci_device
                    && record.pci_function() == pci_function
            })
            .cloned()
            .ok_or_else(|| {
                DiscoveryRunDeviceAdmissionErrorV1::new(
                    DiscoveryRunDeviceAdmissionErrorCodeV1::BaselineCudaContradiction,
                    format!(
                        "CUDA ordinal {ordinal} PCI identity is absent from the complete physical inventory"
                    ),
                )
            })?;
        let major = device
            .get_attribute(DeviceAttribute::ComputeCapabilityMajor)
            .map_err(native_identity_error_v1)?;
        let minor = device
            .get_attribute(DeviceAttribute::ComputeCapabilityMinor)
            .map_err(native_identity_error_v1)?;
        let major = u16::try_from(major).map_err(|_| {
            DiscoveryRunDeviceAdmissionErrorV1::new(
                DiscoveryRunDeviceAdmissionErrorCodeV1::NativeDeviceIdentityFailure,
                "negative or oversized CUDA compute-capability major",
            )
        })?;
        let minor = u16::try_from(minor).map_err(|_| {
            DiscoveryRunDeviceAdmissionErrorV1::new(
                DiscoveryRunDeviceAdmissionErrorCodeV1::NativeDeviceIdentityFailure,
                "negative or oversized CUDA compute-capability minor",
            )
        })?;
        let sass_target = format!("sm_{major}{minor}");
        if !build
            .sass_targets
            .iter()
            .any(|target| target == &sass_target)
        {
            saw_build_incompatible = true;
            continue;
        }
        let mut build = build;
        build.selected_sass_target = sass_target;
        return Ok(NativeCudaCandidateV1 {
            device,
            ordinal,
            pci_identity,
            device_uuid: cuda_device_uuid_v1(device)?,
            compute_capability_major: major,
            compute_capability_minor: minor,
            cuda_build_identity: build,
        });
    }

    let code = if saw_build_incompatible {
        DiscoveryRunDeviceAdmissionErrorCodeV1::VisibleGpuBuildIncompatible
    } else {
        DiscoveryRunDeviceAdmissionErrorCodeV1::NoCompatibleCudaOrdinal
    };
    Err(DiscoveryRunDeviceAdmissionErrorV1::new(
        code,
        "no visible CUDA ordinal has an exact physical-inventory and native-SASS binding",
    ))
}

#[cfg(not(feature = "cuda"))]
fn select_lowest_compatible_cuda_ordinal_v1(
    _inventory: &SealedPhysicalGpuInventoryReceiptV1,
    _count: u32,
) -> Result<NativeCudaCandidateV1, DiscoveryRunDeviceAdmissionErrorV1> {
    Err(DiscoveryRunDeviceAdmissionErrorV1::new(
        DiscoveryRunDeviceAdmissionErrorCodeV1::VisibleGpuWithoutStrictBackend,
        "physical GPU exists but the strict CUDA adapter is not compiled",
    ))
}

#[cfg(feature = "cuda")]
fn retain_primary_context_once_v1(
    candidate: &NativeCudaCandidateV1,
) -> Result<Arc<Context>, DiscoveryRunDeviceAdmissionErrorV1> {
    Context::new(candidate.device)
        .map(Arc::new)
        .map_err(|error| {
            DiscoveryRunDeviceAdmissionErrorV1::new(
                DiscoveryRunDeviceAdmissionErrorCodeV1::NativeContextFailure,
                error.to_string(),
            )
        })
}

#[cfg(feature = "cuda")]
fn create_run_stream_once_v1() -> Result<Arc<Stream>, DiscoveryRunDeviceAdmissionErrorV1> {
    Stream::new(StreamFlags::NON_BLOCKING, None)
        .map(Arc::new)
        .map_err(|error| {
            DiscoveryRunDeviceAdmissionErrorV1::new(
                DiscoveryRunDeviceAdmissionErrorCodeV1::NativeStreamFailure,
                error.to_string(),
            )
        })
}

#[cfg(feature = "cuda")]
fn seal_native_cuda_run_device_admission_v1(
    inventory: SealedPhysicalGpuInventoryReceiptV1,
    candidate: NativeCudaCandidateV1,
    primary_context: Arc<Context>,
    run_stream: Arc<Stream>,
    counters: RunDeviceAcquisitionCountersV1,
) -> Result<SealedDiscoveryRunDeviceAdmissionV1, DiscoveryRunDeviceAdmissionErrorV1> {
    let (free_memory_bytes, _) = mem_get_info().map_err(|error| {
        DiscoveryRunDeviceAdmissionErrorV1::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::NativeMemorySnapshotFailure,
            error.to_string(),
        )
    })?;
    let free_memory_bytes_snapshot = u64::try_from(free_memory_bytes).map_err(|_| {
        DiscoveryRunDeviceAdmissionErrorV1::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::NativeMemorySnapshotFailure,
            "CUDA free-memory snapshot exceeds the u64 workspace ABI",
        )
    })?;
    let driver = CudaApiVersion::get().map_err(native_identity_error_v1)?;
    let context_api = primary_context
        .get_api_version()
        .map_err(native_identity_error_v1)?;
    let driver_version = format!("cuDriverGetVersion={}.{}", driver.major(), driver.minor());
    let context_api_version = format!(
        "cuCtxGetApiVersion={}.{}",
        context_api.major(),
        context_api.minor()
    );
    let admission_identity_sha256 = hash_native_admission_v1(
        inventory.inventory_identity_sha256(),
        &candidate,
        primary_context.as_raw() as usize,
        run_stream.as_inner() as usize,
        free_memory_bytes_snapshot,
        &driver_version,
        &context_api_version,
    );
    Ok(SealedDiscoveryRunDeviceAdmissionV1::NativeCuda(Box::new(
        SealedNativeCudaRunDeviceAdmissionV1 {
            physical_inventory_identity_sha256: inventory.inventory_identity_sha256(),
            pci_identity: candidate.pci_identity,
            device_uuid: candidate.device_uuid,
            ordinal: candidate.ordinal,
            primary_context,
            run_stream,
            sass_target: candidate.cuda_build_identity.selected_sass_target.clone(),
            cuda_build_identity: candidate.cuda_build_identity,
            driver_version,
            context_api_version,
            compute_capability_major: candidate.compute_capability_major,
            compute_capability_minor: candidate.compute_capability_minor,
            free_memory_bytes_snapshot,
            probe_counters: counters,
            admission_identity_sha256,
        },
    )))
}

#[cfg(feature = "cuda")]
fn native_identity_error_v1(error: cust::error::CudaError) -> DiscoveryRunDeviceAdmissionErrorV1 {
    DiscoveryRunDeviceAdmissionErrorV1::new(
        DiscoveryRunDeviceAdmissionErrorCodeV1::NativeDeviceIdentityFailure,
        error.to_string(),
    )
}

#[cfg(feature = "cuda")]
fn cuda_device_uuid_v1(device: Device) -> Result<[u8; 16], DiscoveryRunDeviceAdmissionErrorV1> {
    let mut uuid = CUuuid::default();
    // SAFETY: `uuid` is a valid exclusive output and `device` was returned by
    // the same initialized driver enumeration.
    let status = unsafe { cuDeviceGetUuid(&mut uuid, device.as_raw()) };
    if status != CUDA_SUCCESS {
        return Err(DiscoveryRunDeviceAdmissionErrorV1::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::NativeDeviceIdentityFailure,
            format!("cuDeviceGetUuid failed with status {}", status as i32),
        ));
    }
    let bytes = uuid.bytes.map(|byte| byte as u8);
    if bytes.iter().all(|byte| *byte == 0) {
        return Err(DiscoveryRunDeviceAdmissionErrorV1::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::NativeDeviceIdentityFailure,
            "CUDA returned an all-zero device UUID",
        ));
    }
    Ok(bytes)
}

#[cfg(feature = "cuda")]
fn cuda_pci_location_v1(
    device: Device,
) -> Result<(u16, u8, u8, u8), DiscoveryRunDeviceAdmissionErrorV1> {
    let mut buffer = [0_i8; 32];
    // SAFETY: the fixed buffer is writable and its size is passed exactly.
    let status = unsafe {
        cuDeviceGetPCIBusId(
            buffer.as_mut_ptr(),
            i32::try_from(buffer.len()).expect("fixed PCI buffer fits i32"),
            device.as_raw(),
        )
    };
    if status != CUDA_SUCCESS {
        return Err(DiscoveryRunDeviceAdmissionErrorV1::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::NativeDeviceIdentityFailure,
            format!("cuDeviceGetPCIBusId failed with status {}", status as i32),
        ));
    }
    // SAFETY: successful CUDA writes a NUL-terminated bus identifier into the
    // provided buffer.
    let text = unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_str()
        .map_err(|error| {
            DiscoveryRunDeviceAdmissionErrorV1::new(
                DiscoveryRunDeviceAdmissionErrorCodeV1::NativeDeviceIdentityFailure,
                error.to_string(),
            )
        })?;
    parse_cuda_pci_bus_id_v1(text)
}

#[cfg(feature = "cuda")]
fn parse_cuda_pci_bus_id_v1(
    text: &str,
) -> Result<(u16, u8, u8, u8), DiscoveryRunDeviceAdmissionErrorV1> {
    let (domain, remainder) = text.split_once(':').ok_or_else(|| {
        DiscoveryRunDeviceAdmissionErrorV1::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::NativeDeviceIdentityFailure,
            format!("malformed CUDA PCI bus id {text:?}"),
        )
    })?;
    let (bus, slot_and_function) = remainder.split_once(':').ok_or_else(|| {
        DiscoveryRunDeviceAdmissionErrorV1::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::NativeDeviceIdentityFailure,
            format!("malformed CUDA PCI bus id {text:?}"),
        )
    })?;
    let (slot, function) = slot_and_function.split_once('.').ok_or_else(|| {
        DiscoveryRunDeviceAdmissionErrorV1::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::NativeDeviceIdentityFailure,
            format!("malformed CUDA PCI bus id {text:?}"),
        )
    })?;
    let parse = |field: &str, label: &'static str| {
        u16::from_str_radix(field, 16).map_err(|error| {
            DiscoveryRunDeviceAdmissionErrorV1::new(
                DiscoveryRunDeviceAdmissionErrorCodeV1::NativeDeviceIdentityFailure,
                format!("invalid CUDA PCI {label} {field:?}: {error}"),
            )
        })
    };
    let domain = parse(domain, "domain")?;
    let bus = u8::try_from(parse(bus, "bus")?).map_err(|_| {
        DiscoveryRunDeviceAdmissionErrorV1::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::NativeDeviceIdentityFailure,
            "CUDA PCI bus exceeds u8",
        )
    })?;
    let slot = u8::try_from(parse(slot, "device")?).map_err(|_| {
        DiscoveryRunDeviceAdmissionErrorV1::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::NativeDeviceIdentityFailure,
            "CUDA PCI device exceeds u8",
        )
    })?;
    let function = u8::try_from(parse(function, "function")?).map_err(|_| {
        DiscoveryRunDeviceAdmissionErrorV1::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::NativeDeviceIdentityFailure,
            "CUDA PCI function exceeds u8",
        )
    })?;
    Ok((domain, bus, slot, function))
}

#[cfg(feature = "cuda")]
fn parse_cuda_build_manifest_v1(
    manifest: &str,
) -> Result<SealedCudaNativeBuildIdentityV1, DiscoveryRunDeviceAdmissionErrorV1> {
    let value: Value = serde_json::from_str(manifest).map_err(|error| {
        DiscoveryRunDeviceAdmissionErrorV1::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::NativeBuildManifestFailure,
            error.to_string(),
        )
    })?;
    let required_text = |field: &'static str| -> Result<&str, DiscoveryRunDeviceAdmissionErrorV1> {
        value.get(field).and_then(Value::as_str).ok_or_else(|| {
            DiscoveryRunDeviceAdmissionErrorV1::new(
                DiscoveryRunDeviceAdmissionErrorCodeV1::NativeBuildManifestFailure,
                format!("CUDA build manifest is missing text field {field}"),
            )
        })
    };
    if required_text("schema")? != "neoethos.cuda-native-build.v1" {
        return Err(DiscoveryRunDeviceAdmissionErrorV1::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::NativeBuildManifestFailure,
            "CUDA build manifest schema mismatch",
        ));
    }
    let sass_targets = required_string_array_v1(&value, "sass_targets")?;
    let ptx_targets = required_string_array_v1(&value, "ptx_targets")?;
    if sass_targets.is_empty() || !ptx_targets.is_empty() {
        return Err(DiscoveryRunDeviceAdmissionErrorV1::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::NativeBuildManifestFailure,
            "CUDA build must contain exact SASS and no PTX fallback",
        ));
    }
    let artifact_sha256 = value
        .get("artifact")
        .and_then(|artifact| artifact.get("sha256"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            DiscoveryRunDeviceAdmissionErrorV1::new(
                DiscoveryRunDeviceAdmissionErrorCodeV1::NativeBuildManifestFailure,
                "CUDA build manifest is missing artifact.sha256",
            )
        })?;
    Ok(SealedCudaNativeBuildIdentityV1 {
        manifest_sha256: Sha256::digest(manifest.as_bytes()).into(),
        artifact_sha256: parse_sha256_hex_v1(artifact_sha256)?,
        nvcc_version: required_text("nvcc_version")?.to_string(),
        sass_targets,
        selected_sass_target: String::new(),
    })
}

#[cfg(feature = "cuda")]
fn required_string_array_v1(
    value: &Value,
    field: &'static str,
) -> Result<Vec<String>, DiscoveryRunDeviceAdmissionErrorV1> {
    value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| {
            DiscoveryRunDeviceAdmissionErrorV1::new(
                DiscoveryRunDeviceAdmissionErrorCodeV1::NativeBuildManifestFailure,
                format!("CUDA build manifest is missing array field {field}"),
            )
        })?
        .iter()
        .map(|entry| {
            entry.as_str().map(str::to_owned).ok_or_else(|| {
                DiscoveryRunDeviceAdmissionErrorV1::new(
                    DiscoveryRunDeviceAdmissionErrorCodeV1::NativeBuildManifestFailure,
                    format!("CUDA build manifest field {field} has a non-text entry"),
                )
            })
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn parse_sha256_hex_v1(text: &str) -> Result<[u8; 32], DiscoveryRunDeviceAdmissionErrorV1> {
    if text.len() != 64 || !text.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(DiscoveryRunDeviceAdmissionErrorV1::new(
            DiscoveryRunDeviceAdmissionErrorCodeV1::NativeBuildManifestFailure,
            "CUDA artifact SHA256 is not 64 hexadecimal characters",
        ));
    }
    let mut output = [0_u8; 32];
    for (index, byte) in output.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).map_err(|error| {
            DiscoveryRunDeviceAdmissionErrorV1::new(
                DiscoveryRunDeviceAdmissionErrorCodeV1::NativeBuildManifestFailure,
                error.to_string(),
            )
        })?;
    }
    Ok(output)
}

#[cfg(feature = "cuda")]
fn hash_native_admission_v1(
    inventory_identity_sha256: [u8; 32],
    candidate: &NativeCudaCandidateV1,
    context_handle: usize,
    stream_handle: usize,
    free_memory_bytes_snapshot: u64,
    driver_version: &str,
    context_api_version: &str,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(RUN_DEVICE_ADMISSION_SCHEMA_V1.as_bytes());
    hasher.update(b"native-cuda");
    hasher.update(inventory_identity_sha256);
    hasher.update(candidate.ordinal.to_le_bytes());
    hasher.update(candidate.device_uuid);
    hasher.update(candidate.pci_identity.pci_domain().to_le_bytes());
    hasher.update([
        candidate.pci_identity.pci_bus(),
        candidate.pci_identity.pci_device(),
        candidate.pci_identity.pci_function(),
    ]);
    hasher.update(candidate.cuda_build_identity.manifest_sha256);
    hasher.update(candidate.cuda_build_identity.artifact_sha256);
    hasher.update(candidate.cuda_build_identity.nvcc_version.as_bytes());
    hasher.update((context_handle as u64).to_le_bytes());
    hasher.update((stream_handle as u64).to_le_bytes());
    hasher.update(free_memory_bytes_snapshot.to_le_bytes());
    hasher.update(driver_version.as_bytes());
    hasher.update(context_api_version.as_bytes());
    hasher.finalize().into()
}
