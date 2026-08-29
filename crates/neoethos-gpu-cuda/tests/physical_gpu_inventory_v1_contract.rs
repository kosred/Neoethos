use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-gpu-cuda"))
}

fn read(relative: &str) -> String {
    let path = manifest_dir().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read required source {}: {error}", path.display()))
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let (_, tail) = source
        .split_once(start)
        .unwrap_or_else(|| panic!("missing source boundary {start:?}"));
    tail.split_once(end)
        .unwrap_or_else(|| panic!("missing source boundary {end:?} after {start:?}"))
        .0
}

fn require_all(source: &str, required: &[&str]) {
    for token in required {
        assert!(source.contains(token), "missing required token {token:?}");
    }
}

fn declaration_window<'a>(source: &'a str, declaration: &str) -> &'a str {
    let start = source
        .find(declaration)
        .unwrap_or_else(|| panic!("missing declaration {declaration:?}"));
    let before = source[..start]
        .rfind("\n\n")
        .map(|offset| offset + 2)
        .unwrap_or(0);
    let body_end = source[start..]
        .find("\n}")
        .map(|offset| start + offset + 2)
        .unwrap_or_else(|| panic!("unterminated declaration {declaration:?}"));
    &source[before..body_end]
}

#[test]
fn manifest_pins_the_reviewed_pci_info_release_without_default_features() {
    let manifest = read("Cargo.toml");
    let dependency = manifest
        .lines()
        .find(|line| line.trim_start().starts_with("pci-info"))
        .expect("missing exact pci-info dependency");
    require_all(
        dependency,
        &[
            "pci-info",
            "version = \"=0.3.4\"",
            "default-features = false",
        ],
    );
}

#[test]
fn inventory_uses_windows_setupapi_and_linux_exhaustive_enumeration() {
    let source = read("src/physical_gpu_inventory_v1.rs");
    require_all(
        &source,
        &[
            "pci_info::PciInfo",
            "PciInfo::enumerate_pci",
            "WindowsSetupApiPciEnumerator",
            "LinuxProcFsPciEnumerator::Exhaustive",
            "cfg(target_os = \"windows\")",
            "cfg(target_os = \"linux\")",
            "UnsupportedPlatform",
        ],
    );
    for forbidden in [
        "LinuxProcFsPciEnumerator::Fastest",
        "LinuxProcFsPciEnumerator::HeadersOnly",
        "SkipNoncommonHeaders",
    ] {
        assert!(
            !source.contains(forbidden),
            "incomplete Linux inventory mode remained reachable via {forbidden:?}"
        );
    }
}

#[test]
fn supported_platform_inventory_branches_return_their_tail_expression() {
    let source = read("src/physical_gpu_inventory_v1.rs");
    assert_eq!(
        source
            .matches("seal_complete_physical_gpu_inventory_v1(")
            .count(),
        3,
        "expected the helper definition plus exact Windows and Linux calls"
    );
    assert!(
        !source.contains("return seal_complete_physical_gpu_inventory_v1("),
        "a supported-platform cfg tail retained a needless return"
    );
}

#[test]
fn every_physical_record_binds_platform_pci_identity_class_and_location() {
    let source = read("src/physical_gpu_inventory_v1.rs");
    let record = section(&source, "pub struct PhysicalGpuInventoryRecordV1 {", "\n}");
    require_all(
        record,
        &[
            "platform:",
            "vendor_id: u16",
            "device_id: u16",
            "base_class: u8",
            "subclass: u8",
            "pci_domain: u16",
            "pci_bus: u8",
            "pci_device: u8",
            "pci_function: u8",
        ],
    );
    require_all(
        &source,
        &[
            "PCI_BASE_CLASS_DISPLAY_CONTROLLER: u8 = 0x03",
            "PCI_BASE_CLASS_PROCESSING_ACCELERATOR: u8 = 0x12",
            "NVIDIA_VENDOR_ID: u16 = 0x10de",
            "AMD_VENDOR_ID: u16 = 0x1002",
            "INTEL_VENDOR_ID: u16 = 0x8086",
        ],
    );
}

#[test]
fn fatal_partial_missing_class_or_missing_location_inventory_fails_closed() {
    let source = read("src/physical_gpu_inventory_v1.rs");
    require_all(
        &source,
        &[
            "PhysicalGpuInventoryErrorCodeV1",
            "FatalEnumeration",
            "PartialEnumeration",
            "MissingClass",
            "MissingLocation",
            "AmbiguousLocation",
            "collect::<Result<Vec<_>, _>>()",
        ],
    );
    for forbidden in [
        ".filter_map(Result::ok)",
        ".flatten()",
        ".unwrap_or_default()",
        ".unwrap_or(false)",
    ] {
        assert!(
            !source.contains(forbidden),
            "inventory loss was silently accepted via {forbidden:?}"
        );
    }
}

#[test]
fn non_compute_adapters_are_excluded_only_after_positive_proof() {
    let source = read("src/physical_gpu_inventory_v1.rs");
    require_all(
        &source,
        &[
            "enum PhysicalAdapterDispositionV1",
            "PhysicalGpu",
            "ProvenNonCompute",
            "Ambiguous",
            "PhysicalAdapterDispositionV1::Ambiguous => Err(",
            "PhysicalAdapterDispositionV1::ProvenNonCompute =>",
            "ASPEED_VENDOR_ID: u16 = 0x1a03",
            "ASPEED_AST_GRAPHICS_DEVICE_ID: u16 = 0x2000",
            "is_reviewed_non_compute_adapter_v1(vendor_id, device_id, base_class, subclass)",
            "aspeed_bmc_display_is_non_compute_only_for_exact_class",
        ],
    );
    let classifier = section(&source, "fn classify_physical_pci_adapter_v1(", "\n}");
    assert!(
        !classifier.contains("_ => PhysicalAdapterDispositionV1::ProvenNonCompute"),
        "unknown adapters were treated as proven non-compute devices"
    );
}

#[test]
fn complete_zero_inventory_is_the_only_source_of_cpu_absence_authority() {
    let source = read("src/physical_gpu_inventory_v1.rs");
    let sealer = section(
        &source,
        "pub(crate) fn seal_no_physical_gpu_receipt_v1(",
        "\n}",
    );
    require_all(
        sealer,
        &[
            "SealedPhysicalGpuInventoryReceiptV1",
            "inventory.is_complete()",
            "inventory.records().is_empty()",
            "SealedNoPhysicalGpuReceiptV1",
        ],
    );
    for forbidden in ["card_present: bool", "gpu_present: bool", "assume_no_gpu"] {
        assert!(
            !sealer.contains(forbidden),
            "caller-controlled absence authority remained via {forbidden:?}"
        );
    }
}

#[test]
fn inventory_and_zero_gpu_receipts_are_opaque_and_non_reconstructible() {
    let source = read("src/physical_gpu_inventory_v1.rs");
    for declaration in [
        "pub struct SealedPhysicalGpuInventoryReceiptV1 {",
        "pub struct SealedNoPhysicalGpuReceiptV1 {",
    ] {
        let window = declaration_window(&source, declaration);
        for forbidden in ["Clone", "Default", "Deserialize", "pub "] {
            let body = section(window, declaration, "\n}");
            if forbidden == "pub " {
                assert!(
                    !body.contains(forbidden),
                    "sealed receipt exposed a public field"
                );
            } else {
                assert!(
                    !window.contains(forbidden),
                    "sealed receipt gained reconstructible trait {forbidden:?}"
                );
            }
        }
    }
    for forbidden in [
        "pub fn from_records",
        "pub fn from_sha256",
        "pub fn new_no_physical_gpu",
    ] {
        assert!(
            !source.contains(forbidden),
            "caller could mint physical inventory authority via {forbidden:?}"
        );
    }
}

#[test]
fn inventory_authority_does_not_use_lossy_runtime_or_configuration_proxies() {
    let source = read("src/physical_gpu_inventory_v1.rs");
    for forbidden in [
        "nvidia-smi",
        "wgpu",
        "HardwareProbe",
        "std::env",
        "std::process::Command",
        "wmi",
        "CUDA_VISIBLE_DEVICES",
    ] {
        assert!(
            !source.contains(forbidden),
            "physical inventory used non-authoritative proxy {forbidden:?}"
        );
    }
}
