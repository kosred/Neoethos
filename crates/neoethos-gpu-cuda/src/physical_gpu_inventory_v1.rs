//! Fail-closed physical PCI GPU inventory for discovery run admission.
//!
//! This is deliberately independent from CUDA. A missing or broken CUDA
//! runtime cannot prove that a host has no GPU, so CPU admission starts from a
//! complete OS PCI inventory instead.

use pci_info::PciDevice;
use pci_info::PciInfo;
#[cfg(target_os = "linux")]
use pci_info::enumerators::LinuxProcFsPciEnumerator;
#[cfg(target_os = "windows")]
use pci_info::enumerators::WindowsSetupApiPciEnumerator;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

const PHYSICAL_GPU_INVENTORY_SCHEMA_V1: &str = "neoethos.physical-gpu-inventory.v1";
const PCI_BASE_CLASS_DISPLAY_CONTROLLER: u8 = 0x03;
const PCI_BASE_CLASS_PROCESSING_ACCELERATOR: u8 = 0x12;
const NVIDIA_VENDOR_ID: u16 = 0x10de;
const AMD_VENDOR_ID: u16 = 0x1002;
const INTEL_VENDOR_ID: u16 = 0x8086;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PhysicalGpuInventoryPlatformV1 {
    WindowsSetupApi,
    LinuxProcFsExhaustive,
}

impl PhysicalGpuInventoryPlatformV1 {
    const fn wire_name(self) -> &'static str {
        match self {
            Self::WindowsSetupApi => "windows-setupapi",
            Self::LinuxProcFsExhaustive => "linux-procfs-exhaustive",
        }
    }

    pub(crate) const fn wire_name_for_admission_v1(self) -> &'static str {
        self.wire_name()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PhysicalGpuInventoryRecordV1 {
    platform: PhysicalGpuInventoryPlatformV1,
    vendor_id: u16,
    device_id: u16,
    base_class: u8,
    subclass: u8,
    pci_domain: u16,
    pci_bus: u8,
    pci_device: u8,
    pci_function: u8,
}

impl PhysicalGpuInventoryRecordV1 {
    pub const fn platform(&self) -> PhysicalGpuInventoryPlatformV1 {
        self.platform
    }

    pub const fn vendor_id(&self) -> u16 {
        self.vendor_id
    }

    pub const fn device_id(&self) -> u16 {
        self.device_id
    }

    pub const fn base_class(&self) -> u8 {
        self.base_class
    }

    pub const fn subclass(&self) -> u8 {
        self.subclass
    }

    pub const fn pci_domain(&self) -> u16 {
        self.pci_domain
    }

    pub const fn pci_bus(&self) -> u8 {
        self.pci_bus
    }

    pub const fn pci_device(&self) -> u8 {
        self.pci_device
    }

    pub const fn pci_function(&self) -> u8 {
        self.pci_function
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalGpuInventoryErrorCodeV1 {
    UnsupportedPlatform,
    FatalEnumeration,
    PartialEnumeration,
    MissingClass,
    MissingLocation,
    AmbiguousLocation,
    AmbiguousAdapter,
    IncompleteInventory,
    PhysicalGpuPresent,
}

#[derive(Debug, Error)]
#[error("physical GPU inventory failed ({code:?}): {detail}")]
pub struct PhysicalGpuInventoryErrorV1 {
    code: PhysicalGpuInventoryErrorCodeV1,
    detail: String,
}

impl PhysicalGpuInventoryErrorV1 {
    fn new(code: PhysicalGpuInventoryErrorCodeV1, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }

    pub const fn code(&self) -> PhysicalGpuInventoryErrorCodeV1 {
        self.code
    }
}

#[derive(Debug)]
pub struct SealedPhysicalGpuInventoryReceiptV1 {
    platform: PhysicalGpuInventoryPlatformV1,
    records: Vec<PhysicalGpuInventoryRecordV1>,
    enumerated_pci_device_count: u64,
    evaluated_gpu_class_adapter_count: u64,
    inventory_identity_sha256: [u8; 32],
    complete: bool,
}

impl SealedPhysicalGpuInventoryReceiptV1 {
    pub const fn platform(&self) -> PhysicalGpuInventoryPlatformV1 {
        self.platform
    }

    pub fn records(&self) -> &[PhysicalGpuInventoryRecordV1] {
        &self.records
    }

    pub const fn inventory_identity_sha256(&self) -> [u8; 32] {
        self.inventory_identity_sha256
    }

    pub const fn enumerated_pci_device_count(&self) -> u64 {
        self.enumerated_pci_device_count
    }

    pub const fn evaluated_gpu_class_adapter_count(&self) -> u64 {
        self.evaluated_gpu_class_adapter_count
    }

    pub const fn is_complete(&self) -> bool {
        self.complete
    }
}

#[derive(Debug)]
pub struct SealedNoPhysicalGpuReceiptV1 {
    platform: PhysicalGpuInventoryPlatformV1,
    inventory_identity_sha256: [u8; 32],
}

impl SealedNoPhysicalGpuReceiptV1 {
    pub const fn platform(&self) -> PhysicalGpuInventoryPlatformV1 {
        self.platform
    }

    pub const fn inventory_identity_sha256(&self) -> [u8; 32] {
        self.inventory_identity_sha256
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum PhysicalAdapterDispositionV1 {
    PhysicalGpu,
    ProvenSoftwareOrVirtual,
    Ambiguous,
    NotGpuClass,
}

impl PhysicalAdapterDispositionV1 {
    const fn wire_code(self) -> u8 {
        match self {
            Self::PhysicalGpu => 1,
            Self::ProvenSoftwareOrVirtual => 2,
            Self::Ambiguous => 3,
            Self::NotGpuClass => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct EvaluatedPciGpuClassAdapterV1 {
    vendor_id: u16,
    device_id: u16,
    base_class: u8,
    subclass: u8,
    pci_domain: u16,
    pci_bus: u8,
    pci_device: u8,
    pci_function: u8,
    disposition: PhysicalAdapterDispositionV1,
}

pub fn probe_physical_gpu_inventory_v1()
-> Result<SealedPhysicalGpuInventoryReceiptV1, PhysicalGpuInventoryErrorV1> {
    #[cfg(target_os = "windows")]
    {
        let enumerated = PciInfo::enumerate_pci_with_enumerator(WindowsSetupApiPciEnumerator)
            .map_err(|error| {
                PhysicalGpuInventoryErrorV1::new(
                    PhysicalGpuInventoryErrorCodeV1::FatalEnumeration,
                    error.to_string(),
                )
            })?;
        let devices = enumerated
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| {
                PhysicalGpuInventoryErrorV1::new(
                    PhysicalGpuInventoryErrorCodeV1::PartialEnumeration,
                    error.to_string(),
                )
            })?;
        seal_complete_physical_gpu_inventory_v1(
            PhysicalGpuInventoryPlatformV1::WindowsSetupApi,
            devices,
        )
    }

    #[cfg(target_os = "linux")]
    {
        let enumerated = PciInfo::enumerate_pci_with_enumerator(
            LinuxProcFsPciEnumerator::Exhaustive,
        )
        .map_err(|error| {
            PhysicalGpuInventoryErrorV1::new(
                PhysicalGpuInventoryErrorCodeV1::FatalEnumeration,
                error.to_string(),
            )
        })?;
        let devices = enumerated
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| {
                PhysicalGpuInventoryErrorV1::new(
                    PhysicalGpuInventoryErrorCodeV1::PartialEnumeration,
                    error.to_string(),
                )
            })?;
        seal_complete_physical_gpu_inventory_v1(
            PhysicalGpuInventoryPlatformV1::LinuxProcFsExhaustive,
            devices,
        )
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        Err(PhysicalGpuInventoryErrorV1::new(
            PhysicalGpuInventoryErrorCodeV1::UnsupportedPlatform,
            "only Windows SetupAPI and Linux exhaustive PCI inventory are reviewed",
        ))
    }
}

fn seal_complete_physical_gpu_inventory_v1(
    platform: PhysicalGpuInventoryPlatformV1,
    devices: Vec<PciDevice>,
) -> Result<SealedPhysicalGpuInventoryReceiptV1, PhysicalGpuInventoryErrorV1> {
    let mut seen_locations = BTreeSet::new();
    let mut records = Vec::new();
    let mut evaluated_gpu_class_adapters = Vec::new();
    let enumerated_pci_device_count = u64::try_from(devices.len()).map_err(|_| {
        PhysicalGpuInventoryErrorV1::new(
            PhysicalGpuInventoryErrorCodeV1::FatalEnumeration,
            "PCI device count exceeds the inventory wire type",
        )
    })?;

    for device in devices {
        let base_class = device.device_class_code().map_err(|error| {
            PhysicalGpuInventoryErrorV1::new(
                PhysicalGpuInventoryErrorCodeV1::MissingClass,
                error.to_string(),
            )
        })?;
        let subclass = device.device_subclass_code().map_err(|error| {
            PhysicalGpuInventoryErrorV1::new(
                PhysicalGpuInventoryErrorCodeV1::MissingClass,
                error.to_string(),
            )
        })?;
        let location = device.location().map_err(|error| {
            PhysicalGpuInventoryErrorV1::new(
                PhysicalGpuInventoryErrorCodeV1::MissingLocation,
                error.to_string(),
            )
        })?;
        let location_key = (
            location.segment(),
            location.bus(),
            location.device(),
            location.function(),
        );
        if !seen_locations.insert(location_key) {
            return Err(PhysicalGpuInventoryErrorV1::new(
                PhysicalGpuInventoryErrorCodeV1::AmbiguousLocation,
                format!(
                    "duplicate PCI location {:04x}:{:02x}:{:02x}.{}",
                    location_key.0, location_key.1, location_key.2, location_key.3
                ),
            ));
        }

        let vendor_id = device.vendor_id();
        let device_id = device.device_id();
        let disposition =
            classify_physical_pci_adapter_v1(vendor_id, device_id, base_class, subclass);
        if matches!(
            base_class,
            PCI_BASE_CLASS_DISPLAY_CONTROLLER | PCI_BASE_CLASS_PROCESSING_ACCELERATOR
        ) {
            evaluated_gpu_class_adapters.push(EvaluatedPciGpuClassAdapterV1 {
                vendor_id,
                device_id,
                base_class,
                subclass,
                pci_domain: location.segment(),
                pci_bus: location.bus(),
                pci_device: location.device(),
                pci_function: location.function(),
                disposition,
            });
        }
        match disposition {
            PhysicalAdapterDispositionV1::PhysicalGpu => {
                records.push(PhysicalGpuInventoryRecordV1 {
                    platform,
                    vendor_id,
                    device_id,
                    base_class,
                    subclass,
                    pci_domain: location.segment(),
                    pci_bus: location.bus(),
                    pci_device: location.device(),
                    pci_function: location.function(),
                });
            }
            PhysicalAdapterDispositionV1::ProvenSoftwareOrVirtual => {}
            PhysicalAdapterDispositionV1::Ambiguous => Err(PhysicalGpuInventoryErrorV1::new(
                PhysicalGpuInventoryErrorCodeV1::AmbiguousAdapter,
                format!(
                    "unreviewed PCI accelerator {:04x}:{:04x} class {:02x}:{:02x}",
                    vendor_id, device_id, base_class, subclass
                ),
            ))?,
            PhysicalAdapterDispositionV1::NotGpuClass => {}
        }
    }

    records.sort();
    evaluated_gpu_class_adapters.sort();
    let evaluated_gpu_class_adapter_count = u64::try_from(evaluated_gpu_class_adapters.len())
        .map_err(|_| {
            PhysicalGpuInventoryErrorV1::new(
                PhysicalGpuInventoryErrorCodeV1::FatalEnumeration,
                "GPU-class adapter count exceeds the inventory wire type",
            )
        })?;
    let inventory_identity_sha256 = hash_complete_inventory_v1(
        platform,
        enumerated_pci_device_count,
        &records,
        &evaluated_gpu_class_adapters,
    );
    Ok(SealedPhysicalGpuInventoryReceiptV1 {
        platform,
        records,
        enumerated_pci_device_count,
        evaluated_gpu_class_adapter_count,
        inventory_identity_sha256,
        complete: true,
    })
}

fn classify_physical_pci_adapter_v1(
    vendor_id: u16,
    device_id: u16,
    base_class: u8,
    _subclass: u8,
) -> PhysicalAdapterDispositionV1 {
    if !matches!(
        base_class,
        PCI_BASE_CLASS_DISPLAY_CONTROLLER | PCI_BASE_CLASS_PROCESSING_ACCELERATOR
    ) {
        return PhysicalAdapterDispositionV1::NotGpuClass;
    }

    if is_reviewed_software_or_virtual_adapter_v1(vendor_id, device_id) {
        return PhysicalAdapterDispositionV1::ProvenSoftwareOrVirtual;
    }

    match vendor_id {
        NVIDIA_VENDOR_ID | AMD_VENDOR_ID | INTEL_VENDOR_ID => {
            PhysicalAdapterDispositionV1::PhysicalGpu
        }
        _ => PhysicalAdapterDispositionV1::Ambiguous,
    }
}

fn is_reviewed_software_or_virtual_adapter_v1(vendor_id: u16, device_id: u16) -> bool {
    matches!(
        (vendor_id, device_id),
        (0x1414, 0x008c)
            | (0x15ad, 0x0405 | 0x0710 | 0x0770)
            | (0x80ee, 0xbeef)
            | (0x1b36, 0x0100)
            | (0x1af4, 0x1050)
    )
}

fn hash_complete_inventory_v1(
    platform: PhysicalGpuInventoryPlatformV1,
    enumerated_pci_device_count: u64,
    records: &[PhysicalGpuInventoryRecordV1],
    evaluated_gpu_class_adapters: &[EvaluatedPciGpuClassAdapterV1],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(PHYSICAL_GPU_INVENTORY_SCHEMA_V1.as_bytes());
    hasher.update(platform.wire_name().as_bytes());
    hasher.update(enumerated_pci_device_count.to_le_bytes());
    hasher.update((records.len() as u64).to_le_bytes());
    for record in records {
        hasher.update(record.platform.wire_name().as_bytes());
        hasher.update(record.vendor_id.to_le_bytes());
        hasher.update(record.device_id.to_le_bytes());
        hasher.update([record.base_class, record.subclass]);
        hasher.update(record.pci_domain.to_le_bytes());
        hasher.update([record.pci_bus, record.pci_device, record.pci_function]);
    }
    hasher.update((evaluated_gpu_class_adapters.len() as u64).to_le_bytes());
    for adapter in evaluated_gpu_class_adapters {
        hasher.update(adapter.vendor_id.to_le_bytes());
        hasher.update(adapter.device_id.to_le_bytes());
        hasher.update([adapter.base_class, adapter.subclass]);
        hasher.update(adapter.pci_domain.to_le_bytes());
        hasher.update([adapter.pci_bus, adapter.pci_device, adapter.pci_function]);
        hasher.update([adapter.disposition.wire_code()]);
    }
    hasher.finalize().into()
}

pub(crate) fn seal_no_physical_gpu_receipt_v1(
    inventory: SealedPhysicalGpuInventoryReceiptV1,
) -> Result<SealedNoPhysicalGpuReceiptV1, PhysicalGpuInventoryErrorV1> {
    if !inventory.is_complete() {
        return Err(PhysicalGpuInventoryErrorV1::new(
            PhysicalGpuInventoryErrorCodeV1::IncompleteInventory,
            "PCI inventory is not complete",
        ));
    }
    if !inventory.records().is_empty() {
        return Err(PhysicalGpuInventoryErrorV1::new(
            PhysicalGpuInventoryErrorCodeV1::PhysicalGpuPresent,
            "complete PCI inventory contains a physical GPU",
        ));
    }
    Ok(SealedNoPhysicalGpuReceiptV1 {
        platform: inventory.platform(),
        inventory_identity_sha256: inventory.inventory_identity_sha256(),
    })
}
