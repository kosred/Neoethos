#[cfg(feature = "gpu-b-adapter")]
use neoethos_gpu_contracts::device::NeoPopulationMetricRow;
#[cfg(feature = "gpu-b-adapter")]
use neoethos_gpu_cuda::{CudaPopulationDeviceIdentityV1, PopulationResidencyCountersV1};
use serde::Serialize;
#[cfg(feature = "gpu-b-adapter")]
use sha2::{Digest, Sha256};
#[cfg(feature = "gpu-b-adapter")]
use std::fmt;

#[cfg(feature = "gpu-b-adapter")]
pub const NATIVE_POPULATION_RESIDENCY_RECEIPT_SCHEMA_VERSION_V1: u16 = 1;

#[cfg(feature = "gpu-b-adapter")]
const NATIVE_RESIDENCY_HASH_DOMAIN_V1: &[u8] =
    b"neoethos.search.native-population-residency-receipt.v1\0";

#[cfg(feature = "gpu-b-adapter")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativePopulationResidencyReceiptErrorCodeV1 {
    NoSuccessfulNativePopulation,
    ParentUploadCountMismatch,
    ParentIdentityMismatch,
    StreamCreationCountMismatch,
    MissingViewBinding,
    TransferAccountingMismatch,
    SynchronizationMismatch,
    InvalidDeviceIdentity,
    NativeAbiMismatch,
    MissingCudaBuildManifest,
    InvalidCudaBuildManifest,
}

#[cfg(feature = "gpu-b-adapter")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct NativePopulationResidencyReceiptErrorV1 {
    code: NativePopulationResidencyReceiptErrorCodeV1,
    message: String,
}

#[cfg(feature = "gpu-b-adapter")]
impl fmt::Display for NativePopulationResidencyReceiptErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

#[cfg(feature = "gpu-b-adapter")]
impl std::error::Error for NativePopulationResidencyReceiptErrorV1 {}

#[cfg(feature = "gpu-b-adapter")]
fn error(
    code: NativePopulationResidencyReceiptErrorCodeV1,
    message: impl Into<String>,
) -> NativePopulationResidencyReceiptErrorV1 {
    NativePopulationResidencyReceiptErrorV1 {
        code,
        message: message.into(),
    }
}

/// Immutable evidence emitted only after a run-scoped native session returned
/// an exact output. It records transfers and synchronization; it is not a GPU
/// capability, benchmark result, or promotion permit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct NativePopulationResidencyReceiptV1 {
    schema_version: u16,
    parent_dataset_identity_sha256: String,
    successful_native_population_count: u64,
    parent_upload_count: u64,
    parent_upload_bytes: u64,
    view_binding_count: u64,
    full_binding_count: u64,
    range_binding_count: u64,
    ordered_binding_count: u64,
    ordered_index_upload_bytes: u64,
    adaptive_upload_bytes: u64,
    stream_creation_count: u64,
    explicit_synchronization_count: u64,
    metric_rows_readback_count: u64,
    metric_rows_readback_rows: u64,
    metric_rows_readback_bytes: u64,
    diagnostic_readback_count: u64,
    diagnostic_readback_rows: u64,
    diagnostic_readback_bytes: u64,
    accepted_trade_total_readback_count: u64,
    accepted_trade_total_readback_bytes: u64,
    selected_device_ordinal: u32,
    device_name: String,
    device_uuid_hex: String,
    compute_capability_major: u32,
    compute_capability_minor: u32,
    multiprocessor_count: u32,
    total_global_memory_bytes: u64,
    pci_domain_id: i32,
    pci_bus_id: i32,
    pci_device_id: i32,
    device_identity_sha256: String,
    native_abi_version: u32,
    cuda_build_manifest_sha256: String,
    identity_sha256: String,
}

impl NativePopulationResidencyReceiptV1 {
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub fn parent_dataset_identity_sha256(&self) -> &str {
        &self.parent_dataset_identity_sha256
    }

    pub const fn successful_native_population_count(&self) -> u64 {
        self.successful_native_population_count
    }

    pub const fn parent_upload_count(&self) -> u64 {
        self.parent_upload_count
    }

    pub const fn parent_upload_bytes(&self) -> u64 {
        self.parent_upload_bytes
    }

    pub const fn view_binding_count(&self) -> u64 {
        self.view_binding_count
    }

    pub const fn full_binding_count(&self) -> u64 {
        self.full_binding_count
    }

    pub const fn range_binding_count(&self) -> u64 {
        self.range_binding_count
    }

    pub const fn ordered_binding_count(&self) -> u64 {
        self.ordered_binding_count
    }

    pub const fn ordered_index_upload_bytes(&self) -> u64 {
        self.ordered_index_upload_bytes
    }

    pub const fn adaptive_upload_bytes(&self) -> u64 {
        self.adaptive_upload_bytes
    }

    pub const fn stream_creation_count(&self) -> u64 {
        self.stream_creation_count
    }

    pub const fn explicit_synchronization_count(&self) -> u64 {
        self.explicit_synchronization_count
    }

    pub const fn metric_rows_readback_count(&self) -> u64 {
        self.metric_rows_readback_count
    }

    pub const fn metric_rows_readback_rows(&self) -> u64 {
        self.metric_rows_readback_rows
    }

    pub const fn metric_rows_readback_bytes(&self) -> u64 {
        self.metric_rows_readback_bytes
    }

    pub const fn diagnostic_readback_count(&self) -> u64 {
        self.diagnostic_readback_count
    }

    pub const fn diagnostic_readback_rows(&self) -> u64 {
        self.diagnostic_readback_rows
    }

    pub const fn diagnostic_readback_bytes(&self) -> u64 {
        self.diagnostic_readback_bytes
    }

    pub const fn accepted_trade_total_readback_count(&self) -> u64 {
        self.accepted_trade_total_readback_count
    }

    pub const fn accepted_trade_total_readback_bytes(&self) -> u64 {
        self.accepted_trade_total_readback_bytes
    }

    pub const fn selected_device_ordinal(&self) -> u32 {
        self.selected_device_ordinal
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn device_uuid_hex(&self) -> &str {
        &self.device_uuid_hex
    }

    pub fn device_identity_sha256(&self) -> &str {
        &self.device_identity_sha256
    }

    pub const fn native_abi_version(&self) -> u32 {
        self.native_abi_version
    }

    pub fn cuda_build_manifest_sha256(&self) -> &str {
        &self.cuda_build_manifest_sha256
    }

    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }
}

#[cfg(feature = "gpu-b-adapter")]
fn hex_lower(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(feature = "gpu-b-adapter")]
fn manifest_error(message: impl Into<String>) -> NativePopulationResidencyReceiptErrorV1 {
    error(
        NativePopulationResidencyReceiptErrorCodeV1::InvalidCudaBuildManifest,
        message,
    )
}

#[cfg(feature = "gpu-b-adapter")]
fn required_manifest_string<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<&'a str, NativePopulationResidencyReceiptErrorV1> {
    object
        .get(field)
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| manifest_error(format!("CUDA build manifest has no non-empty `{field}`")))
}

#[cfg(feature = "gpu-b-adapter")]
fn required_manifest_string_array(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<Vec<String>, NativePopulationResidencyReceiptErrorV1> {
    object
        .get(field)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| manifest_error(format!("CUDA build manifest `{field}` is not an array")))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .filter(|entry| !entry.is_empty())
                .map(str::to_owned)
                .ok_or_else(|| {
                    manifest_error(format!(
                        "CUDA build manifest `{field}` contains a non-string or empty entry"
                    ))
                })
        })
        .collect()
}

#[cfg(feature = "gpu-b-adapter")]
pub(crate) fn validate_cuda_build_manifest_v1(
    manifest: &str,
    device_identity: CudaPopulationDeviceIdentityV1,
) -> Result<(), NativePopulationResidencyReceiptErrorV1> {
    const ROOT_FIELDS: [&str; 11] = [
        "schema",
        "resolution_mode",
        "architectures",
        "gencode",
        "sass_targets",
        "ptx_targets",
        "precision_flags",
        "optimization",
        "nvcc_version",
        "cuobjdump_version",
        "artifact",
    ];
    const PRECISION_FLAGS: [&str; 4] = [
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    ];

    let value: serde_json::Value = serde_json::from_str(manifest).map_err(|source| {
        manifest_error(format!("CUDA build manifest is invalid JSON: {source}"))
    })?;
    let object = value
        .as_object()
        .ok_or_else(|| manifest_error("CUDA build manifest root is not an object"))?;
    if object.len() != ROOT_FIELDS.len()
        || ROOT_FIELDS.iter().any(|field| !object.contains_key(*field))
    {
        return Err(manifest_error(
            "CUDA build manifest v1 does not have the exact reviewed field set",
        ));
    }
    if required_manifest_string(object, "schema")? != "neoethos.cuda-native-build.v1" {
        return Err(manifest_error(
            "CUDA build manifest schema is not reviewed v1",
        ));
    }
    if !matches!(
        required_manifest_string(object, "resolution_mode")?,
        "host_auto" | "cross_release_explicit"
    ) {
        return Err(manifest_error(
            "CUDA build manifest has an unsupported resolution mode",
        ));
    }

    let architecture_values = object
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| manifest_error("CUDA build manifest `architectures` is not an array"))?;
    let architectures = architecture_values
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| u16::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    manifest_error(
                        "CUDA build manifest contains an invalid architecture identifier",
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if architectures.is_empty()
        || architectures
            .windows(2)
            .any(|window| window[0] >= window[1])
    {
        return Err(manifest_error(
            "CUDA build manifest architectures are empty, duplicated, or unsorted",
        ));
    }

    let expected_gencode = architectures
        .iter()
        .map(|architecture| {
            format!("--generate-code=arch=compute_{architecture},code=sm_{architecture}")
        })
        .collect::<Vec<_>>();
    let expected_sass_targets = architectures
        .iter()
        .map(|architecture| format!("sm_{architecture}"))
        .collect::<Vec<_>>();
    let gencode = required_manifest_string_array(object, "gencode")?;
    let sass_targets = required_manifest_string_array(object, "sass_targets")?;
    let ptx_targets = required_manifest_string_array(object, "ptx_targets")?;
    if gencode != expected_gencode
        || sass_targets != expected_sass_targets
        || !ptx_targets.is_empty()
    {
        return Err(manifest_error(
            "CUDA build manifest does not prove the exact SASS-only architecture plan",
        ));
    }
    let required_sass_target = format!(
        "sm_{}{}",
        device_identity.compute_capability_major(),
        device_identity.compute_capability_minor()
    );
    if !sass_targets
        .iter()
        .any(|target| target == &required_sass_target)
    {
        return Err(manifest_error(format!(
            "CUDA build manifest has no exact `{required_sass_target}` image for the selected device"
        )));
    }

    let precision_flags = required_manifest_string_array(object, "precision_flags")?;
    let expected_precision_flags = PRECISION_FLAGS
        .iter()
        .map(|flag| (*flag).to_owned())
        .collect::<Vec<_>>();
    if precision_flags != expected_precision_flags {
        return Err(manifest_error(
            "CUDA build manifest does not bind the reviewed f64 precision flags",
        ));
    }
    if !matches!(
        required_manifest_string(object, "optimization")?,
        "-O3" | "-lineinfo"
    ) {
        return Err(manifest_error(
            "CUDA build manifest has an unsupported optimization mode",
        ));
    }
    for field in ["nvcc_version", "cuobjdump_version"] {
        required_manifest_string(object, field)?;
    }

    let artifact = object
        .get("artifact")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| manifest_error("CUDA build manifest `artifact` is not an object"))?;
    if artifact.len() != 3
        || ["logical_name", "sha256", "byte_len"]
            .iter()
            .any(|field| !artifact.contains_key(*field))
    {
        return Err(manifest_error(
            "CUDA build manifest artifact does not have the exact reviewed field set",
        ));
    }
    if !matches!(
        required_manifest_string(artifact, "logical_name")?,
        "libneoethos_gpu_cuda_native.a" | "neoethos_gpu_cuda_native.lib"
    ) {
        return Err(manifest_error(
            "CUDA build manifest names an unexpected native artifact",
        ));
    }
    let artifact_sha256 = required_manifest_string(artifact, "sha256")?;
    if artifact_sha256.len() != 64
        || !artifact_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(manifest_error(
            "CUDA build manifest artifact SHA-256 is not canonical lowercase hex",
        ));
    }
    if artifact
        .get("byte_len")
        .and_then(serde_json::Value::as_u64)
        .filter(|byte_len| *byte_len > 0)
        .is_none()
    {
        return Err(manifest_error(
            "CUDA build manifest artifact has no positive byte length",
        ));
    }
    Ok(())
}

#[cfg(feature = "gpu-b-adapter")]
pub(crate) fn seal_native_population_residency_receipt_v1(
    expected_parent_dataset_identity_sha256: &str,
    observed_parent_dataset_identity_sha256: &str,
    successful_native_population_count: u64,
    counters: PopulationResidencyCountersV1,
    device_identity: CudaPopulationDeviceIdentityV1,
    native_abi_version: u32,
    cuda_build_manifest_v1: &str,
) -> Result<NativePopulationResidencyReceiptV1, NativePopulationResidencyReceiptErrorV1> {
    if successful_native_population_count == 0 {
        return Err(error(
            NativePopulationResidencyReceiptErrorCodeV1::NoSuccessfulNativePopulation,
            "native population residency has no exact successful output",
        ));
    }
    if expected_parent_dataset_identity_sha256 != observed_parent_dataset_identity_sha256 {
        return Err(error(
            NativePopulationResidencyReceiptErrorCodeV1::ParentIdentityMismatch,
            "native population session parent identity differs from the sealed Search parent",
        ));
    }
    if counters.parent_upload_count() != 1 {
        return Err(error(
            NativePopulationResidencyReceiptErrorCodeV1::ParentUploadCountMismatch,
            format!(
                "native population parent upload count is {}; expected exactly one",
                counters.parent_upload_count()
            ),
        ));
    }
    if counters.parent_upload_bytes() == 0 {
        return Err(error(
            NativePopulationResidencyReceiptErrorCodeV1::TransferAccountingMismatch,
            "native population parent upload has zero accounted bytes",
        ));
    }
    if counters.stream_creation_count() != 1 {
        return Err(error(
            NativePopulationResidencyReceiptErrorCodeV1::StreamCreationCountMismatch,
            format!(
                "native population stream creation count is {}; expected exactly one",
                counters.stream_creation_count()
            ),
        ));
    }
    if counters.view_binding_count() == 0 {
        return Err(error(
            NativePopulationResidencyReceiptErrorCodeV1::MissingViewBinding,
            "native population session has no exact view binding",
        ));
    }
    let exact_view_binding_total = counters
        .full_binding_count()
        .checked_add(counters.range_binding_count())
        .and_then(|total| total.checked_add(counters.ordered_binding_count()))
        .ok_or_else(|| {
            error(
                NativePopulationResidencyReceiptErrorCodeV1::TransferAccountingMismatch,
                "native population view-binding counters overflow",
            )
        })?;
    if counters.view_binding_count() != exact_view_binding_total
        || (counters.ordered_binding_count() == 0 && counters.ordered_index_upload_bytes() != 0)
        || (counters.ordered_binding_count() != 0 && counters.ordered_index_upload_bytes() == 0)
        || counters.ordered_index_upload_bytes() % std::mem::size_of::<u64>() as u64 != 0
        || counters.ordered_index_upload_bytes() / (std::mem::size_of::<u64>() as u64)
            < counters.ordered_binding_count()
        || counters.adaptive_upload_bytes() % std::mem::size_of::<f64>() as u64 != 0
    {
        return Err(error(
            NativePopulationResidencyReceiptErrorCodeV1::TransferAccountingMismatch,
            "native population view and ordered-index transfer counters are inconsistent",
        ));
    }
    let expected_metric_bytes = counters
        .metric_rows_readback_rows()
        .checked_mul(std::mem::size_of::<NeoPopulationMetricRow>() as u64);
    if counters.metric_rows_readback_count() == 0
        || counters.metric_rows_readback_rows() < counters.metric_rows_readback_count()
        || expected_metric_bytes != Some(counters.metric_rows_readback_bytes())
        || counters.diagnostic_readback_count() != 0
        || counters.diagnostic_readback_rows() != 0
        || counters.diagnostic_readback_bytes() != 0
        || counters.accepted_trade_total_readback_count() != 0
        || counters.accepted_trade_total_readback_bytes() != 0
    {
        return Err(error(
            NativePopulationResidencyReceiptErrorCodeV1::TransferAccountingMismatch,
            "native population D2H readback count, row, and byte accounting is inconsistent",
        ));
    }
    if counters.explicit_synchronization_count() != counters.metric_rows_readback_count()
        || successful_native_population_count != counters.metric_rows_readback_count()
    {
        return Err(error(
            NativePopulationResidencyReceiptErrorCodeV1::SynchronizationMismatch,
            "native population successes, synchronization, and D2H boundaries disagree",
        ));
    }

    let device_name = std::str::from_utf8(device_identity.name_bytes()).map_err(|_| {
        error(
            NativePopulationResidencyReceiptErrorCodeV1::InvalidDeviceIdentity,
            "native CUDA device name is not valid UTF-8",
        )
    })?;
    if device_name.is_empty()
        || device_identity.compute_capability_major() == 0
        || device_identity.compute_capability_minor() > 9
        || device_identity.multiprocessor_count() == 0
        || device_identity.total_global_memory_bytes() == 0
        || device_identity.uuid().iter().all(|byte| *byte == 0)
    {
        return Err(error(
            NativePopulationResidencyReceiptErrorCodeV1::InvalidDeviceIdentity,
            "native CUDA session returned incomplete physical-device identity",
        ));
    }
    if native_abi_version != neoethos_gpu_contracts::ABI_VERSION {
        return Err(error(
            NativePopulationResidencyReceiptErrorCodeV1::NativeAbiMismatch,
            format!(
                "native CUDA ABI {native_abi_version} differs from Rust ABI {}",
                neoethos_gpu_contracts::ABI_VERSION
            ),
        ));
    }
    if cuda_build_manifest_v1.is_empty() {
        return Err(error(
            NativePopulationResidencyReceiptErrorCodeV1::MissingCudaBuildManifest,
            "successful native CUDA population has no embedded build manifest",
        ));
    }
    validate_cuda_build_manifest_v1(cuda_build_manifest_v1, device_identity)?;

    let device_uuid_hex = hex_lower(device_identity.uuid());
    let mut device_hasher = Sha256::new();
    device_hasher.update(b"neoethos.search.native-population-device.v1\0");
    device_hasher.update(device_identity.selected_device_ordinal().to_le_bytes());
    device_hasher.update(device_identity.compute_capability_major().to_le_bytes());
    device_hasher.update(device_identity.compute_capability_minor().to_le_bytes());
    device_hasher.update(device_identity.multiprocessor_count().to_le_bytes());
    device_hasher.update(device_identity.total_global_memory_bytes().to_le_bytes());
    device_hasher.update(device_identity.pci_domain_id().to_le_bytes());
    device_hasher.update(device_identity.pci_bus_id().to_le_bytes());
    device_hasher.update(device_identity.pci_device_id().to_le_bytes());
    device_hasher.update(device_identity.uuid());
    device_hasher.update((device_identity.name_bytes().len() as u64).to_le_bytes());
    device_hasher.update(device_identity.name_bytes());
    let device_identity_sha256 = hex_lower(&device_hasher.finalize());
    let cuda_build_manifest_sha256 = hex_lower(&Sha256::digest(cuda_build_manifest_v1.as_bytes()));

    let mut hasher = Sha256::new();
    hasher.update(NATIVE_RESIDENCY_HASH_DOMAIN_V1);
    hasher.update(NATIVE_POPULATION_RESIDENCY_RECEIPT_SCHEMA_VERSION_V1.to_le_bytes());
    hasher.update((expected_parent_dataset_identity_sha256.len() as u64).to_le_bytes());
    hasher.update(expected_parent_dataset_identity_sha256.as_bytes());
    hasher.update(successful_native_population_count.to_le_bytes());
    for value in [
        counters.parent_upload_count(),
        counters.parent_upload_bytes(),
        counters.view_binding_count(),
        counters.full_binding_count(),
        counters.range_binding_count(),
        counters.ordered_binding_count(),
        counters.ordered_index_upload_bytes(),
        counters.adaptive_upload_bytes(),
        counters.stream_creation_count(),
        counters.explicit_synchronization_count(),
        counters.metric_rows_readback_count(),
        counters.metric_rows_readback_rows(),
        counters.metric_rows_readback_bytes(),
        counters.diagnostic_readback_count(),
        counters.diagnostic_readback_rows(),
        counters.diagnostic_readback_bytes(),
        counters.accepted_trade_total_readback_count(),
        counters.accepted_trade_total_readback_bytes(),
    ] {
        hasher.update(value.to_le_bytes());
    }
    hasher.update((device_identity_sha256.len() as u64).to_le_bytes());
    hasher.update(device_identity_sha256.as_bytes());
    hasher.update(native_abi_version.to_le_bytes());
    hasher.update((cuda_build_manifest_sha256.len() as u64).to_le_bytes());
    hasher.update(cuda_build_manifest_sha256.as_bytes());
    let identity_sha256 = hex_lower(&hasher.finalize());

    Ok(NativePopulationResidencyReceiptV1 {
        schema_version: NATIVE_POPULATION_RESIDENCY_RECEIPT_SCHEMA_VERSION_V1,
        parent_dataset_identity_sha256: expected_parent_dataset_identity_sha256.to_owned(),
        successful_native_population_count,
        parent_upload_count: counters.parent_upload_count(),
        parent_upload_bytes: counters.parent_upload_bytes(),
        view_binding_count: counters.view_binding_count(),
        full_binding_count: counters.full_binding_count(),
        range_binding_count: counters.range_binding_count(),
        ordered_binding_count: counters.ordered_binding_count(),
        ordered_index_upload_bytes: counters.ordered_index_upload_bytes(),
        adaptive_upload_bytes: counters.adaptive_upload_bytes(),
        stream_creation_count: counters.stream_creation_count(),
        explicit_synchronization_count: counters.explicit_synchronization_count(),
        metric_rows_readback_count: counters.metric_rows_readback_count(),
        metric_rows_readback_rows: counters.metric_rows_readback_rows(),
        metric_rows_readback_bytes: counters.metric_rows_readback_bytes(),
        diagnostic_readback_count: counters.diagnostic_readback_count(),
        diagnostic_readback_rows: counters.diagnostic_readback_rows(),
        diagnostic_readback_bytes: counters.diagnostic_readback_bytes(),
        accepted_trade_total_readback_count: counters.accepted_trade_total_readback_count(),
        accepted_trade_total_readback_bytes: counters.accepted_trade_total_readback_bytes(),
        selected_device_ordinal: device_identity.selected_device_ordinal(),
        device_name: device_name.to_owned(),
        device_uuid_hex,
        compute_capability_major: device_identity.compute_capability_major(),
        compute_capability_minor: device_identity.compute_capability_minor(),
        multiprocessor_count: device_identity.multiprocessor_count(),
        total_global_memory_bytes: device_identity.total_global_memory_bytes(),
        pci_domain_id: device_identity.pci_domain_id(),
        pci_bus_id: device_identity.pci_bus_id(),
        pci_device_id: device_identity.pci_device_id(),
        device_identity_sha256,
        native_abi_version,
        cuda_build_manifest_sha256,
        identity_sha256,
    })
}
