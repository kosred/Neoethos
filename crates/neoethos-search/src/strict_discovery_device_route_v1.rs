//! Run-bound, fail-closed device authority for Discovery population work.
//!
//! Configuration strings do not prove hardware absence or select a CUDA
//! ordinal. One real native probe either seals one exact compatible ordinal or
//! proves that a loaded CUDA runtime enumerated exactly zero devices. Every
//! other state is an error and cannot authorize CPU execution.

#[cfg(feature = "gpu-b-native")]
use sha2::{Digest, Sha256};
use std::fmt;

#[cfg(feature = "gpu-b-native")]
const PROBE_HASH_DOMAIN_V1: &[u8] = b"neoethos.search.strict-cuda-probe.v1\0";
#[cfg(feature = "gpu-b-native")]
const DEVICE_HASH_DOMAIN_V1: &[u8] = b"neoethos.search.strict-cuda-device.v1\0";

#[cfg(any(test, feature = "gpu-b-native"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StrictCudaProbeFailureKindV1 {
    DeviceLost,
    UnsupportedRuntime,
    UnreadableDeviceIdentity,
}

#[cfg(any(test, feature = "gpu-b-native"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CudaOrdinalProbeOutcomeV1 {
    Compatible {
        ordinal: u32,
    },
    BuildIncompatible {
        ordinal: u32,
    },
    Fault {
        ordinal: u32,
        failure: StrictCudaProbeFailureKindV1,
    },
}

#[cfg(any(test, feature = "gpu-b-native"))]
impl CudaOrdinalProbeOutcomeV1 {
    const fn ordinal(&self) -> u32 {
        match self {
            Self::Compatible { ordinal }
            | Self::BuildIncompatible { ordinal }
            | Self::Fault { ordinal, .. } => *ordinal,
        }
    }
}

#[cfg(any(test, feature = "gpu-b-native"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StrictDiscoveryProbeObservationV1 {
    pub(crate) native_adapter_compiled: bool,
    pub(crate) runtime_loaded: bool,
    pub(crate) reported_device_count: u32,
    pub(crate) ordinal_outcomes: Vec<CudaOrdinalProbeOutcomeV1>,
}

#[cfg(any(test, feature = "gpu-b-native"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NoCompatibleGpuReasonV1 {
    NoVisibleCudaOrdinal,
}

#[cfg(any(test, feature = "gpu-b-native"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UnsealedStrictDiscoveryDeviceRouteV1 {
    NativeCuda { selected_ordinal: u32 },
    CpuNoCompatibleGpu { reason: NoCompatibleGpuReasonV1 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StrictDiscoveryDeviceRouteErrorCodeV1 {
    NativeAdapterNotCompiled,
    #[cfg(any(test, feature = "gpu-b-native"))]
    CudaRuntimeUnavailable,
    #[cfg(any(test, feature = "gpu-b-native"))]
    IncompleteCudaProbe,
    #[cfg(any(test, feature = "gpu-b-native"))]
    CudaProbeFault,
    #[cfg(any(test, feature = "gpu-b-native"))]
    VisibleGpuBuildIncompatible,
    #[cfg(feature = "gpu-b-native")]
    MissingCudaBuildManifest,
    #[cfg(feature = "gpu-b-native")]
    DeviceIdentityMismatch,
    #[cfg(feature = "gpu-b-native")]
    UnreadableDeviceMemory,
    #[cfg(feature = "gpu-b-native")]
    WrongDeviceRoute,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrictDiscoveryDeviceRouteErrorV1 {
    code: StrictDiscoveryDeviceRouteErrorCodeV1,
    message: String,
}

impl StrictDiscoveryDeviceRouteErrorV1 {
    #[cfg(test)]
    pub(crate) const fn code(&self) -> &'static str {
        match self.code {
            StrictDiscoveryDeviceRouteErrorCodeV1::NativeAdapterNotCompiled => {
                "native_adapter_not_compiled"
            }
            #[cfg(any(test, feature = "gpu-b-native"))]
            StrictDiscoveryDeviceRouteErrorCodeV1::CudaRuntimeUnavailable => {
                "cuda_runtime_unavailable"
            }
            #[cfg(any(test, feature = "gpu-b-native"))]
            StrictDiscoveryDeviceRouteErrorCodeV1::IncompleteCudaProbe => "incomplete_cuda_probe",
            #[cfg(any(test, feature = "gpu-b-native"))]
            StrictDiscoveryDeviceRouteErrorCodeV1::CudaProbeFault => "cuda_probe_fault",
            #[cfg(any(test, feature = "gpu-b-native"))]
            StrictDiscoveryDeviceRouteErrorCodeV1::VisibleGpuBuildIncompatible => {
                "visible_gpu_build_incompatible"
            }
            #[cfg(feature = "gpu-b-native")]
            StrictDiscoveryDeviceRouteErrorCodeV1::MissingCudaBuildManifest => {
                "missing_cuda_build_manifest"
            }
            #[cfg(feature = "gpu-b-native")]
            StrictDiscoveryDeviceRouteErrorCodeV1::DeviceIdentityMismatch => {
                "device_identity_mismatch"
            }
            #[cfg(feature = "gpu-b-native")]
            StrictDiscoveryDeviceRouteErrorCodeV1::UnreadableDeviceMemory => {
                "unreadable_device_memory"
            }
            #[cfg(feature = "gpu-b-native")]
            StrictDiscoveryDeviceRouteErrorCodeV1::WrongDeviceRoute => "wrong_device_route",
        }
    }
}

impl fmt::Display for StrictDiscoveryDeviceRouteErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for StrictDiscoveryDeviceRouteErrorV1 {}

fn route_error(
    code: StrictDiscoveryDeviceRouteErrorCodeV1,
    message: impl Into<String>,
) -> StrictDiscoveryDeviceRouteErrorV1 {
    StrictDiscoveryDeviceRouteErrorV1 {
        code,
        message: message.into(),
    }
}

#[cfg(any(test, feature = "gpu-b-native"))]
pub(crate) fn classify_strict_discovery_probe_observation_v1(
    observation: &StrictDiscoveryProbeObservationV1,
) -> Result<UnsealedStrictDiscoveryDeviceRouteV1, StrictDiscoveryDeviceRouteErrorV1> {
    use StrictDiscoveryDeviceRouteErrorCodeV1::{
        CudaProbeFault, CudaRuntimeUnavailable, IncompleteCudaProbe, NativeAdapterNotCompiled,
        VisibleGpuBuildIncompatible,
    };

    if !observation.native_adapter_compiled {
        return Err(route_error(
            NativeAdapterNotCompiled,
            "strict Discovery routing requires the compiled native CUDA adapter; its absence cannot prove that no GPU exists",
        ));
    }
    if !observation.runtime_loaded {
        return Err(route_error(
            CudaRuntimeUnavailable,
            "the CUDA runtime did not load; a driver/runtime fault cannot authorize CPU execution",
        ));
    }

    let expected_count = usize::try_from(observation.reported_device_count).map_err(|_| {
        route_error(
            IncompleteCudaProbe,
            "reported CUDA device count does not fit this process",
        )
    })?;
    if observation.ordinal_outcomes.len() != expected_count {
        return Err(route_error(
            IncompleteCudaProbe,
            "strict CUDA probe did not return exactly one outcome per reported ordinal",
        ));
    }

    let mut seen = vec![false; expected_count];
    let mut compatible = Vec::new();
    for outcome in &observation.ordinal_outcomes {
        let ordinal = usize::try_from(outcome.ordinal()).map_err(|_| {
            route_error(
                IncompleteCudaProbe,
                "CUDA ordinal does not fit this process",
            )
        })?;
        if ordinal >= expected_count || seen[ordinal] {
            return Err(route_error(
                IncompleteCudaProbe,
                "strict CUDA probe returned an out-of-range or duplicate ordinal",
            ));
        }
        seen[ordinal] = true;
        match outcome {
            CudaOrdinalProbeOutcomeV1::Compatible { ordinal } => compatible.push(*ordinal),
            CudaOrdinalProbeOutcomeV1::BuildIncompatible { .. } => {}
            CudaOrdinalProbeOutcomeV1::Fault { failure, .. } => {
                return Err(route_error(
                    CudaProbeFault,
                    format!(
                        "strict CUDA probe failed for a visible ordinal ({failure:?}); refusing CPU substitution"
                    ),
                ));
            }
        }
    }
    if seen.iter().any(|was_seen| !was_seen) {
        return Err(route_error(
            IncompleteCudaProbe,
            "strict CUDA probe omitted a reported ordinal",
        ));
    }
    if expected_count == 0 {
        return Ok(UnsealedStrictDiscoveryDeviceRouteV1::CpuNoCompatibleGpu {
            reason: NoCompatibleGpuReasonV1::NoVisibleCudaOrdinal,
        });
    }
    compatible.sort_unstable();
    compatible
        .first()
        .copied()
        .map(|selected_ordinal| UnsealedStrictDiscoveryDeviceRouteV1::NativeCuda {
            selected_ordinal,
        })
        .ok_or_else(|| {
            route_error(
                VisibleGpuBuildIncompatible,
                "one or more CUDA devices are visible, but the reviewed build has no compatible SASS image; refusing CPU execution",
            )
        })
}

#[cfg(any(test, feature = "gpu-b-adapter"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StrictNativeFailureKindV1 {
    AllocationPressure,
    DeviceLost,
    Unsupported,
    WrongShape,
}

#[cfg(any(test, feature = "gpu-b-adapter"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StrictNativeFailureActionV1 {
    RetrySameOrdinal {
        selected_ordinal: u32,
        next_batch_size: usize,
    },
    FailLoud {
        selected_ordinal: u32,
        failure: StrictNativeFailureKindV1,
    },
}

#[cfg(any(test, feature = "gpu-b-adapter"))]
pub(crate) fn decide_strict_native_failure_v1(
    failure: StrictNativeFailureKindV1,
    selected_ordinal: u32,
    batch_size: usize,
    retry_index: u32,
    max_retries: u32,
) -> StrictNativeFailureActionV1 {
    if failure == StrictNativeFailureKindV1::AllocationPressure
        && retry_index < max_retries
        && batch_size > 4
    {
        return StrictNativeFailureActionV1::RetrySameOrdinal {
            selected_ordinal,
            next_batch_size: batch_size.div_ceil(2).max(4),
        };
    }
    StrictNativeFailureActionV1::FailLoud {
        selected_ordinal,
        failure,
    }
}

/// Exact native CUDA device selected by the sealed run probe.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ExactCudaDeviceOrdinalV1 {
    selected_ordinal: u32,
    pre_parent_free_memory_bytes: u64,
    cuda_device_identity_sha256: String,
    cuda_build_manifest_sha256: String,
    probe_receipt_identity_sha256: String,
}

impl ExactCudaDeviceOrdinalV1 {
    pub const fn selected_ordinal(&self) -> u32 {
        self.selected_ordinal
    }

    /// Free-memory snapshot captured on this exact ordinal after device/build
    /// admission and before any Discovery parent allocation.
    pub const fn pre_parent_free_memory_bytes(&self) -> u64 {
        self.pre_parent_free_memory_bytes
    }

    pub fn cuda_device_identity_sha256(&self) -> &str {
        &self.cuda_device_identity_sha256
    }

    pub fn cuda_build_manifest_sha256(&self) -> &str {
        &self.cuda_build_manifest_sha256
    }

    pub fn probe_receipt_identity_sha256(&self) -> &str {
        &self.probe_receipt_identity_sha256
    }
}

/// Proof that a real loaded CUDA runtime enumerated exactly zero devices.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct SealedNoCompatibleGpuProbeReceiptV1 {
    probe_receipt_identity_sha256: String,
    native_adapter_compiled: bool,
    runtime_loaded: bool,
    reported_device_count: u32,
    ordinal_observation_manifest_sha256: String,
}

impl SealedNoCompatibleGpuProbeReceiptV1 {
    pub fn probe_receipt_identity_sha256(&self) -> &str {
        &self.probe_receipt_identity_sha256
    }
}

/// CPU authority retained by an exact population run. Legacy V1 acquisition
/// can still carry its CUDA-zero receipt for existing callers, while the
/// prepared V3 entrypoint carries the stronger cross-vendor physical-inventory
/// absence proof. Neither variant is caller-constructible.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SealedCpuDiscoveryRouteReceiptV2 {
    _sealed: (),
    #[cfg(feature = "gpu-b-native")]
    kind: SealedCpuDiscoveryRouteReceiptKindV2,
}

#[cfg(feature = "gpu-b-native")]
#[derive(Clone, Debug, PartialEq, Eq)]
enum SealedCpuDiscoveryRouteReceiptKindV2 {
    LegacyCudaZero(SealedNoCompatibleGpuProbeReceiptV1),
    #[cfg(feature = "gpu-cuda")]
    PhysicalGpuAbsence {
        platform: neoethos_gpu_cuda::PhysicalGpuInventoryPlatformV1,
        inventory_identity_sha256: [u8; 32],
    },
}

impl SealedCpuDiscoveryRouteReceiptV2 {
    pub(crate) fn authority_is_nonzero_v2(&self) -> bool {
        #[cfg(feature = "gpu-b-native")]
        {
            match &self.kind {
                SealedCpuDiscoveryRouteReceiptKindV2::LegacyCudaZero(receipt) => {
                    receipt.probe_receipt_identity_sha256().len() == 64
                }
                #[cfg(feature = "gpu-cuda")]
                SealedCpuDiscoveryRouteReceiptKindV2::PhysicalGpuAbsence {
                    platform,
                    inventory_identity_sha256,
                } => {
                    let _supported_platform = platform;
                    *inventory_identity_sha256 != [0; 32]
                }
            }
        }
        #[cfg(not(feature = "gpu-b-native"))]
        {
            false
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SealedStrictDiscoveryDeviceRouteV1 {
    _sealed: (),
    #[cfg(feature = "gpu-b-native")]
    kind: SealedStrictDiscoveryDeviceRouteKindV1,
}

#[cfg(feature = "gpu-b-native")]
#[derive(Clone, Debug, PartialEq, Eq)]
enum SealedStrictDiscoveryDeviceRouteKindV1 {
    NativeCuda(ExactCudaDeviceOrdinalV1),
    CpuNoCompatibleGpu(SealedCpuDiscoveryRouteReceiptV2),
}

/// One opaque, one-shot result of the real native device probe.
///
/// The carrier intentionally is not `Clone`, `Default`, or deserializable.
/// A public entrypoint acquires it once, then moves it into the exact
/// population run. Evaluations can inspect only the run-owned sealed route and
/// therefore cannot trigger a second probe or replace the selected ordinal.
#[derive(Debug)]
pub struct SealedStrictDiscoveryDeviceAdmissionV1 {
    route: SealedStrictDiscoveryDeviceRouteV1,
}

impl SealedStrictDiscoveryDeviceAdmissionV1 {
    pub(crate) fn into_route_v1(self) -> SealedStrictDiscoveryDeviceRouteV1 {
        self.route
    }

    #[cfg(feature = "gpu-cuda")]
    pub(crate) fn from_no_physical_gpu_admission_v1(
        admission: neoethos_gpu_cuda::run_device_admission_v1::SealedCpuNoPhysicalGpuRunDeviceAdmissionV1,
    ) -> Result<Self, StrictDiscoveryDeviceRouteErrorV1> {
        if admission
            .no_physical_gpu_receipt()
            .inventory_identity_sha256()
            == [0; 32]
        {
            return Err(route_error(
                StrictDiscoveryDeviceRouteErrorCodeV1::IncompleteCudaProbe,
                "prepared CPU route does not carry a complete physical-GPU inventory identity",
            ));
        }
        Ok(Self {
            route: SealedStrictDiscoveryDeviceRouteV1 {
                _sealed: (),
                kind: SealedStrictDiscoveryDeviceRouteKindV1::CpuNoCompatibleGpu(
                    SealedCpuDiscoveryRouteReceiptV2 {
                        _sealed: (),
                        kind: SealedCpuDiscoveryRouteReceiptKindV2::PhysicalGpuAbsence {
                            platform: admission.no_physical_gpu_receipt().platform(),
                            inventory_identity_sha256: admission
                                .no_physical_gpu_receipt()
                                .inventory_identity_sha256(),
                        },
                    },
                ),
            },
        })
    }
}

impl SealedStrictDiscoveryDeviceRouteV1 {
    pub(crate) fn population_auto_sizing_route_v1(
        &self,
    ) -> Result<crate::PopulationAutoSizingRouteV1, StrictDiscoveryDeviceRouteErrorV1> {
        #[cfg(feature = "gpu-b-native")]
        {
            return Ok(match &self.kind {
                SealedStrictDiscoveryDeviceRouteKindV1::NativeCuda(ordinal) => {
                    crate::PopulationAutoSizingRouteV1::NativeCuda {
                        selected_ordinal: ordinal.selected_ordinal(),
                        pre_parent_free_memory_bytes: ordinal.pre_parent_free_memory_bytes(),
                        cuda_device_identity_sha256: ordinal
                            .cuda_device_identity_sha256()
                            .to_owned(),
                        cuda_build_manifest_sha256: ordinal.cuda_build_manifest_sha256().to_owned(),
                        probe_receipt_identity_sha256: ordinal
                            .probe_receipt_identity_sha256()
                            .to_owned(),
                    }
                }
                SealedStrictDiscoveryDeviceRouteKindV1::CpuNoCompatibleGpu(receipt) => {
                    let authority = match &receipt.kind {
                        SealedCpuDiscoveryRouteReceiptKindV2::LegacyCudaZero(receipt) => {
                            crate::PopulationAutoCpuAuthorityV1::LegacyCudaZero {
                                probe_receipt_identity_sha256: receipt
                                    .probe_receipt_identity_sha256()
                                    .to_owned(),
                            }
                        }
                        #[cfg(feature = "gpu-cuda")]
                        SealedCpuDiscoveryRouteReceiptKindV2::PhysicalGpuAbsence {
                            platform,
                            inventory_identity_sha256,
                        } => {
                            let platform = match platform {
                                neoethos_gpu_cuda::PhysicalGpuInventoryPlatformV1::WindowsSetupApi => {
                                    "windows-setupapi"
                                }
                                neoethos_gpu_cuda::PhysicalGpuInventoryPlatformV1::LinuxProcFsExhaustive => {
                                    "linux-procfs-exhaustive"
                                }
                            };
                            crate::PopulationAutoCpuAuthorityV1::PhysicalGpuAbsence {
                                platform: platform.to_owned(),
                                inventory_identity_sha256: hex_lower(inventory_identity_sha256),
                            }
                        }
                    };
                    crate::PopulationAutoSizingRouteV1::CpuNoCompatibleGpu { authority }
                }
            });
        }
        #[cfg(not(feature = "gpu-b-native"))]
        {
            let _ = self;
            Err(route_error(
                StrictDiscoveryDeviceRouteErrorCodeV1::NativeAdapterNotCompiled,
                "native adapter is not compiled; no exact population-auto device route can be read",
            ))
        }
    }

    pub(crate) fn require_cpu_route_receipt_v1(
        &self,
    ) -> Result<&SealedCpuDiscoveryRouteReceiptV2, StrictDiscoveryDeviceRouteErrorV1> {
        #[cfg(feature = "gpu-b-native")]
        {
            match &self.kind {
                SealedStrictDiscoveryDeviceRouteKindV1::CpuNoCompatibleGpu(receipt) => Ok(receipt),
                SealedStrictDiscoveryDeviceRouteKindV1::NativeCuda(_) => Err(route_error(
                    StrictDiscoveryDeviceRouteErrorCodeV1::WrongDeviceRoute,
                    "a compatible CUDA ordinal is sealed for this run; refusing CPU substitution",
                )),
            }
        }
        #[cfg(not(feature = "gpu-b-native"))]
        {
            let _ = self;
            Err(route_error(
                StrictDiscoveryDeviceRouteErrorCodeV1::NativeAdapterNotCompiled,
                "native adapter is not compiled; CPU authority cannot be sealed",
            ))
        }
    }

    pub(crate) fn require_exact_cuda_device_ordinal_v1(
        &self,
    ) -> Result<&ExactCudaDeviceOrdinalV1, StrictDiscoveryDeviceRouteErrorV1> {
        #[cfg(feature = "gpu-b-native")]
        {
            match &self.kind {
                SealedStrictDiscoveryDeviceRouteKindV1::NativeCuda(ordinal) => Ok(ordinal),
                SealedStrictDiscoveryDeviceRouteKindV1::CpuNoCompatibleGpu(_) => Err(route_error(
                    StrictDiscoveryDeviceRouteErrorCodeV1::WrongDeviceRoute,
                    "the real run probe enumerated no CUDA device; native execution is unavailable",
                )),
            }
        }
        #[cfg(not(feature = "gpu-b-native"))]
        {
            let _ = self;
            Err(route_error(
                StrictDiscoveryDeviceRouteErrorCodeV1::NativeAdapterNotCompiled,
                "native adapter is not compiled; exact CUDA authority is unavailable",
            ))
        }
    }
}

#[cfg(feature = "gpu-b-native")]
fn hex_lower(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(feature = "gpu-b-native")]
fn ordinal_observation_manifest_sha256(observation: &StrictDiscoveryProbeObservationV1) -> String {
    let mut hasher = Sha256::new();
    hasher.update(PROBE_HASH_DOMAIN_V1);
    hasher.update([u8::from(observation.native_adapter_compiled)]);
    hasher.update([u8::from(observation.runtime_loaded)]);
    hasher.update(observation.reported_device_count.to_le_bytes());
    for outcome in &observation.ordinal_outcomes {
        match outcome {
            CudaOrdinalProbeOutcomeV1::Compatible { ordinal } => {
                hasher.update([0]);
                hasher.update(ordinal.to_le_bytes());
            }
            CudaOrdinalProbeOutcomeV1::BuildIncompatible { ordinal } => {
                hasher.update([1]);
                hasher.update(ordinal.to_le_bytes());
            }
            CudaOrdinalProbeOutcomeV1::Fault { ordinal, failure } => {
                hasher.update([2]);
                hasher.update(ordinal.to_le_bytes());
                hasher.update([match failure {
                    StrictCudaProbeFailureKindV1::DeviceLost => 0,
                    StrictCudaProbeFailureKindV1::UnsupportedRuntime => 1,
                    StrictCudaProbeFailureKindV1::UnreadableDeviceIdentity => 2,
                }]);
            }
        }
    }
    hex_lower(&hasher.finalize())
}

#[cfg(feature = "gpu-b-native")]
fn seal_no_compatible_gpu_probe_receipt_v1(
    observation: &StrictDiscoveryProbeObservationV1,
) -> Result<SealedNoCompatibleGpuProbeReceiptV1, StrictDiscoveryDeviceRouteErrorV1> {
    if !observation.native_adapter_compiled {
        return Err(route_error(
            StrictDiscoveryDeviceRouteErrorCodeV1::NativeAdapterNotCompiled,
            "native adapter is not compiled; CPU authority cannot be sealed",
        ));
    }
    if !observation.runtime_loaded {
        return Err(route_error(
            StrictDiscoveryDeviceRouteErrorCodeV1::CudaRuntimeUnavailable,
            "CUDA runtime is unavailable; CPU authority cannot be sealed",
        ));
    }
    if observation.reported_device_count != 0 {
        return Err(route_error(
            StrictDiscoveryDeviceRouteErrorCodeV1::VisibleGpuBuildIncompatible,
            "a visible CUDA device cannot be transformed into no-GPU CPU authority",
        ));
    }
    if !observation.ordinal_outcomes.is_empty() {
        return Err(route_error(
            StrictDiscoveryDeviceRouteErrorCodeV1::IncompleteCudaProbe,
            "zero-device probe contains impossible ordinal observations",
        ));
    }
    match classify_strict_discovery_probe_observation_v1(observation)? {
        UnsealedStrictDiscoveryDeviceRouteV1::CpuNoCompatibleGpu {
            reason: NoCompatibleGpuReasonV1::NoVisibleCudaOrdinal,
        } => {}
        UnsealedStrictDiscoveryDeviceRouteV1::NativeCuda { .. } => {
            return Err(route_error(
                StrictDiscoveryDeviceRouteErrorCodeV1::VisibleGpuBuildIncompatible,
                "native CUDA route cannot be sealed as CPU authority",
            ));
        }
    }
    let ordinal_observation_manifest_sha256 = ordinal_observation_manifest_sha256(observation);
    let mut hasher = Sha256::new();
    hasher.update(PROBE_HASH_DOMAIN_V1);
    hasher.update(ordinal_observation_manifest_sha256.as_bytes());
    let probe_receipt_identity_sha256 = hex_lower(&hasher.finalize());
    Ok(SealedNoCompatibleGpuProbeReceiptV1 {
        probe_receipt_identity_sha256,
        native_adapter_compiled: observation.native_adapter_compiled,
        runtime_loaded: observation.runtime_loaded,
        reported_device_count: observation.reported_device_count,
        ordinal_observation_manifest_sha256,
    })
}

#[cfg(feature = "gpu-b-native")]
fn probe_real_strict_discovery_device_route_v1()
-> Result<SealedStrictDiscoveryDeviceRouteV1, StrictDiscoveryDeviceRouteErrorV1> {
    let reported_device_count = match neoethos_gpu_cuda::probe_cuda_device_count_v1() {
        Ok(count) => count,
        Err(neoethos_gpu_cuda::CudaDeviceEnumerationErrorV1::NativeAdapterUnavailable) => {
            let observation = StrictDiscoveryProbeObservationV1 {
                native_adapter_compiled: false,
                runtime_loaded: false,
                reported_device_count: 0,
                ordinal_outcomes: Vec::new(),
            };
            classify_strict_discovery_probe_observation_v1(&observation)?;
            return Err(route_error(
                StrictDiscoveryDeviceRouteErrorCodeV1::NativeAdapterNotCompiled,
                "native CUDA adapter is unavailable",
            ));
        }
        Err(
            neoethos_gpu_cuda::CudaDeviceEnumerationErrorV1::RuntimeFailure(_)
            | neoethos_gpu_cuda::CudaDeviceEnumerationErrorV1::InvalidNativeOutput,
        ) => {
            let observation = StrictDiscoveryProbeObservationV1 {
                native_adapter_compiled: true,
                runtime_loaded: false,
                reported_device_count: 0,
                ordinal_outcomes: Vec::new(),
            };
            classify_strict_discovery_probe_observation_v1(&observation)?;
            return Err(route_error(
                StrictDiscoveryDeviceRouteErrorCodeV1::CudaRuntimeUnavailable,
                "fallible CUDA enumeration failed",
            ));
        }
    };
    let runtime_loaded = true;
    let reported_device_capacity = usize::try_from(reported_device_count).map_err(|_| {
        route_error(
            StrictDiscoveryDeviceRouteErrorCodeV1::IncompleteCudaProbe,
            "reported CUDA device count does not fit this process",
        )
    })?;
    let mut ordinal_outcomes = Vec::with_capacity(reported_device_capacity);
    let mut identities = Vec::with_capacity(reported_device_capacity);
    let cuda_build_manifest = if reported_device_count == 0 {
        None
    } else {
        Some(neoethos_gpu_cuda::cuda_build_manifest_v1().ok_or_else(|| {
            route_error(
                StrictDiscoveryDeviceRouteErrorCodeV1::MissingCudaBuildManifest,
                "native CUDA build did not embed its reviewed build manifest",
            )
        })?)
    };

    if let Some(cuda_build_manifest) = cuda_build_manifest {
        for ordinal in 0..reported_device_count {
            let native_ordinal = i32::try_from(ordinal).map_err(|_| {
                route_error(
                    StrictDiscoveryDeviceRouteErrorCodeV1::IncompleteCudaProbe,
                    format!("CUDA ordinal {ordinal} does not fit the native ABI"),
                )
            })?;
            let session = match neoethos_gpu_cuda::PopulationSession::create(native_ordinal, 1) {
                Ok(session) => session,
                Err(_) => {
                    ordinal_outcomes.push(CudaOrdinalProbeOutcomeV1::Fault {
                        ordinal,
                        failure: StrictCudaProbeFailureKindV1::UnsupportedRuntime,
                    });
                    continue;
                }
            };
            let identity = match session.read_device_identity_v1() {
                Ok(identity) => identity,
                Err(_) => {
                    ordinal_outcomes.push(CudaOrdinalProbeOutcomeV1::Fault {
                        ordinal,
                        failure: StrictCudaProbeFailureKindV1::UnreadableDeviceIdentity,
                    });
                    continue;
                }
            };
            if identity.selected_device_ordinal() != ordinal {
                ordinal_outcomes.push(CudaOrdinalProbeOutcomeV1::Fault {
                    ordinal,
                    failure: StrictCudaProbeFailureKindV1::DeviceLost,
                });
                continue;
            }
            match crate::native_population_residency_receipt_v1::validate_cuda_build_manifest_v1(
                cuda_build_manifest,
                identity,
            ) {
                Ok(()) => {
                    ordinal_outcomes.push(CudaOrdinalProbeOutcomeV1::Compatible { ordinal });
                    identities.push((ordinal, identity));
                }
                Err(_) => {
                    ordinal_outcomes.push(CudaOrdinalProbeOutcomeV1::BuildIncompatible { ordinal })
                }
            }
        }
    }

    let observation = StrictDiscoveryProbeObservationV1 {
        native_adapter_compiled: true,
        runtime_loaded,
        reported_device_count,
        ordinal_outcomes,
    };
    match classify_strict_discovery_probe_observation_v1(&observation)? {
        UnsealedStrictDiscoveryDeviceRouteV1::CpuNoCompatibleGpu { .. } => {
            Ok(SealedStrictDiscoveryDeviceRouteV1 {
                _sealed: (),
                kind: SealedStrictDiscoveryDeviceRouteKindV1::CpuNoCompatibleGpu(
                    SealedCpuDiscoveryRouteReceiptV2 {
                        _sealed: (),
                        kind: SealedCpuDiscoveryRouteReceiptKindV2::LegacyCudaZero(
                            seal_no_compatible_gpu_probe_receipt_v1(&observation)?,
                        ),
                    },
                ),
            })
        }
        UnsealedStrictDiscoveryDeviceRouteV1::NativeCuda { selected_ordinal } => {
            let identity = identities
                .into_iter()
                .find_map(|(ordinal, identity)| (ordinal == selected_ordinal).then_some(identity))
                .ok_or_else(|| {
                    route_error(
                        StrictDiscoveryDeviceRouteErrorCodeV1::DeviceIdentityMismatch,
                        "selected CUDA ordinal has no exact compatible identity",
                    )
                })?;
            let mut device_hasher = Sha256::new();
            device_hasher.update(DEVICE_HASH_DOMAIN_V1);
            device_hasher.update(identity.selected_device_ordinal().to_le_bytes());
            device_hasher.update(identity.compute_capability_major().to_le_bytes());
            device_hasher.update(identity.compute_capability_minor().to_le_bytes());
            device_hasher.update(identity.multiprocessor_count().to_le_bytes());
            device_hasher.update(identity.total_global_memory_bytes().to_le_bytes());
            device_hasher.update(identity.pci_domain_id().to_le_bytes());
            device_hasher.update(identity.pci_bus_id().to_le_bytes());
            device_hasher.update(identity.pci_device_id().to_le_bytes());
            device_hasher.update(identity.uuid());
            device_hasher.update(identity.name_bytes());
            let cuda_device_identity_sha256 = hex_lower(&device_hasher.finalize());
            let cuda_build_manifest = cuda_build_manifest.ok_or_else(|| {
                route_error(
                    StrictDiscoveryDeviceRouteErrorCodeV1::MissingCudaBuildManifest,
                    "selected CUDA ordinal has no reviewed build manifest",
                )
            })?;
            let cuda_build_manifest_sha256 =
                hex_lower(&Sha256::digest(cuda_build_manifest.as_bytes()));
            let selected_ordinal_usize = usize::try_from(selected_ordinal).map_err(|_| {
                route_error(
                    StrictDiscoveryDeviceRouteErrorCodeV1::UnreadableDeviceMemory,
                    "selected CUDA ordinal does not fit the free-memory probe",
                )
            })?;
            let pre_parent_free_memory_bytes = neoethos_gpu_cuda::device_free_memory_bytes(
                selected_ordinal_usize,
            )
            .ok_or_else(|| {
                route_error(
                    StrictDiscoveryDeviceRouteErrorCodeV1::UnreadableDeviceMemory,
                    "selected CUDA ordinal has no readable pre-parent free-memory snapshot",
                )
            })?;
            let ordinal_observation_manifest_sha256 =
                ordinal_observation_manifest_sha256(&observation);
            let mut probe_hasher = Sha256::new();
            probe_hasher.update(PROBE_HASH_DOMAIN_V1);
            probe_hasher.update(ordinal_observation_manifest_sha256.as_bytes());
            probe_hasher.update(cuda_device_identity_sha256.as_bytes());
            probe_hasher.update(cuda_build_manifest_sha256.as_bytes());
            probe_hasher.update(pre_parent_free_memory_bytes.to_le_bytes());
            let probe_receipt_identity_sha256 = hex_lower(&probe_hasher.finalize());
            Ok(SealedStrictDiscoveryDeviceRouteV1 {
                _sealed: (),
                kind: SealedStrictDiscoveryDeviceRouteKindV1::NativeCuda(
                    ExactCudaDeviceOrdinalV1 {
                        selected_ordinal,
                        pre_parent_free_memory_bytes,
                        cuda_device_identity_sha256,
                        cuda_build_manifest_sha256,
                        probe_receipt_identity_sha256,
                    },
                ),
            })
        }
    }
}

#[cfg(not(feature = "gpu-b-native"))]
fn probe_real_strict_discovery_device_route_v1()
-> Result<SealedStrictDiscoveryDeviceRouteV1, StrictDiscoveryDeviceRouteErrorV1> {
    Err(route_error(
        StrictDiscoveryDeviceRouteErrorCodeV1::NativeAdapterNotCompiled,
        "native adapter is not compiled; physical GPU absence cannot be proved",
    ))
}

pub fn acquire_strict_discovery_device_admission_v1()
-> Result<SealedStrictDiscoveryDeviceAdmissionV1, StrictDiscoveryDeviceRouteErrorV1> {
    Ok(SealedStrictDiscoveryDeviceAdmissionV1 {
        route: probe_real_strict_discovery_device_route_v1()?,
    })
}
