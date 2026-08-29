use anyhow::{Context, Result, bail};
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use cudarc::driver::{CudaContext, sys::CUdevice_attribute};

use super::super::adaptive_impl::PassiveAggressiveCudaDeviceIdentityV1;
use super::{
    DEVICE_ARITHMETIC_REDUCTION_FAULT, DEVICE_ARITHMETIC_UPDATE_FAULT, DEVICE_LABEL_MAP_FAULT,
    DEVICE_MEMORY_HEADROOM_PERCENT, DEVICE_MEMORY_MIN_HEADROOM_BYTES, DEVICE_MISSING_CLASS_0_FAULT,
    DEVICE_MISSING_CLASS_1_FAULT, DEVICE_MISSING_CLASS_2_FAULT, DEVICE_SCALER_ARITHMETIC_FAULT,
    DEVICE_SCALER_INPUT_FAULT, DEVICE_SCALER_OUTPUT_FAULT, DEVICE_TRANSFORM_ARITHMETIC_FAULT,
    PA_CUBE_UNITS,
};

pub(super) fn checked_mul(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .with_context(|| format!("online_pa CUDA {label} overflow"))
}

pub(super) fn checked_add(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_add(right)
        .with_context(|| format!("online_pa CUDA {label} overflow"))
}

pub(super) fn checked_ceil_div(value: usize, divisor: usize, label: &str) -> Result<usize> {
    if divisor == 0 {
        bail!("online_pa CUDA {label} divisor is zero");
    }
    Ok(checked_add(value, divisor - 1, label)? / divisor)
}

pub(super) fn bytes_for<T>(elements: usize, label: &str) -> Result<usize> {
    checked_mul(elements, core::mem::size_of::<T>(), label)
}

pub(super) fn checked_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("online_pa CUDA {label} exceeds u32"))
}

pub(super) fn checked_u64(value: usize, label: &str) -> Result<u64> {
    u64::try_from(value).with_context(|| format!("online_pa CUDA {label} exceeds u64"))
}

pub(super) fn cube_count_1d(elements: usize, label: &str) -> Result<u32> {
    checked_u32(checked_ceil_div(elements, PA_CUBE_UNITS, label)?, label)
}

pub(super) fn exact_cuda_ordinal(requested_policy: &str, effective_policy: &str) -> Result<usize> {
    let ordinal = match crate::common::parse_cuda_device_policy(effective_policy)? {
        crate::common::CudaDevicePolicy::Gpu { ordinal } => ordinal,
        crate::common::CudaDevicePolicy::Auto => {
            bail!("online_pa full CUDA pipeline requires an exact effective ordinal, not auto")
        }
        crate::common::CudaDevicePolicy::Cpu => {
            bail!("online_pa full CUDA pipeline cannot execute an effective CPU policy")
        }
    };
    let canonical_effective = format!("gpu:{ordinal}");
    if effective_policy != canonical_effective {
        bail!(
            "online_pa full CUDA effective policy must be canonical `{canonical_effective}`, got `{effective_policy}`"
        );
    }
    match crate::common::parse_cuda_device_policy(requested_policy)? {
        crate::common::CudaDevicePolicy::Gpu {
            ordinal: requested_ordinal,
        } if requested_ordinal == ordinal => {}
        crate::common::CudaDevicePolicy::Gpu {
            ordinal: requested_ordinal,
        } => bail!(
            "online_pa requested CUDA ordinal {requested_ordinal} resolved to different ordinal {ordinal}"
        ),
        crate::common::CudaDevicePolicy::Auto => {}
        crate::common::CudaDevicePolicy::Cpu => {
            bail!("online_pa CPU request cannot execute the full CUDA pipeline")
        }
    }
    Ok(ordinal)
}

pub(super) fn query_cuda_device_identity(
    cuda_ordinal: usize,
) -> Result<PassiveAggressiveCudaDeviceIdentityV1> {
    let context = CudaContext::new(cuda_ordinal)
        .with_context(|| format!("retain online_pa CUDA ordinal {cuda_ordinal} for identity"))?;
    if context.ordinal() != cuda_ordinal {
        bail!(
            "online_pa CUDA context ordinal drift: requested {cuda_ordinal}, context retained {}",
            context.ordinal()
        );
    }
    let uuid = context
        .uuid()
        .context("query online_pa CUDA physical UUID")?
        .bytes
        .map(|byte| byte as u8);
    let (compute_capability_major, compute_capability_minor) = context
        .compute_capability()
        .context("query online_pa CUDA compute capability")?;
    let total_memory_bytes = checked_u64(
        context
            .total_mem()
            .context("query online_pa CUDA total memory")?,
        "device total memory",
    )?;
    Ok(PassiveAggressiveCudaDeviceIdentityV1 {
        ordinal: checked_u32(cuda_ordinal, "device ordinal")?,
        uuid,
        pci_domain: context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)
            .context("query online_pa CUDA PCI domain")?,
        pci_bus: context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)
            .context("query online_pa CUDA PCI bus")?,
        pci_device: context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)
            .context("query online_pa CUDA PCI device")?,
        compute_capability_major,
        compute_capability_minor,
        multiprocessor_count: context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .context("query online_pa CUDA SM count")?,
        total_memory_bytes,
        name: context.name().context("query online_pa CUDA device name")?,
    })
}

pub(crate) fn validate_passive_aggressive_cuda_device_identity(
    effective_policy: &str,
    expected: &PassiveAggressiveCudaDeviceIdentityV1,
) -> Result<()> {
    let ordinal = exact_cuda_ordinal(effective_policy, effective_policy)?;
    let actual = query_cuda_device_identity(ordinal)?;
    if &actual != expected {
        bail!(
            "online_pa CUDA physical device identity mismatch for `{effective_policy}`: expected {expected:?}, found {actual:?}"
        );
    }
    Ok(())
}

pub(super) fn preflight_device_memory(
    client: &ComputeClient<CudaRuntime>,
    cuda_ordinal: usize,
    buffer_bytes: &[usize],
) -> Result<()> {
    let max_page_size = usize::try_from(client.properties().memory.max_page_size)
        .context("online_pa CUDA allocator page limit exceeds usize")?;
    if let Some(largest) = buffer_bytes.iter().copied().max()
        && largest > max_page_size
    {
        bail!("online_pa CUDA buffer bytes {largest} exceed allocator page limit {max_page_size}");
    }
    let planned_bytes = buffer_bytes.iter().try_fold(0usize, |total, bytes| {
        checked_add(total, *bytes, "planned device bytes")
    })?;
    let context = CudaContext::new(cuda_ordinal)
        .with_context(|| format!("retain online_pa CUDA ordinal {cuda_ordinal}"))?;
    let (free_bytes, total_bytes) = context
        .mem_get_info()
        .context("inspect online_pa CUDA device memory")?;
    if total_bytes == 0 {
        bail!("online_pa CUDA device reports zero total memory");
    }
    let headroom = total_bytes
        .checked_mul(DEVICE_MEMORY_HEADROOM_PERCENT)
        .context("online_pa CUDA device headroom overflow")?
        / 100;
    let headroom = headroom
        .max(DEVICE_MEMORY_MIN_HEADROOM_BYTES)
        .min(total_bytes);
    let usable_bytes = free_bytes.min(total_bytes).saturating_sub(headroom);
    if planned_bytes > usable_bytes {
        bail!(
            "online_pa CUDA planned bytes {planned_bytes} exceed usable device bytes {usable_bytes} after {headroom} bytes headroom"
        );
    }
    Ok(())
}

pub(super) fn read_f64_buffer(
    client: &ComputeClient<CudaRuntime>,
    handle: cubecl::server::Handle,
    label: &str,
) -> Result<Vec<f64>> {
    let bytes = client
        .read_one(handle)
        .with_context(|| format!("read online_pa CUDA {label}"))?;
    Ok(f64::from_bytes(&bytes).to_vec())
}

pub(super) fn read_u32_buffer(
    client: &ComputeClient<CudaRuntime>,
    handle: cubecl::server::Handle,
    label: &str,
) -> Result<Vec<u32>> {
    let bytes = client
        .read_one(handle)
        .with_context(|| format!("read online_pa CUDA {label}"))?;
    Ok(u32::from_bytes(&bytes).to_vec())
}

pub(super) fn read_arithmetic_status(
    client: &ComputeClient<CudaRuntime>,
    handle: cubecl::server::Handle,
) -> Result<u32> {
    let bytes = client
        .read_one(handle)
        .context("read online_pa CUDA device arithmetic status")?;
    let status = u32::from_bytes(&bytes);
    if status.len() != 1 {
        bail!(
            "online_pa CUDA arithmetic-status readback mismatch: expected 1 entry, received {}",
            status.len()
        );
    }
    Ok(status[0])
}

pub(super) fn fail_for_full_pipeline_status(status: u32) -> Result<()> {
    match status {
        0 => Ok(()),
        DEVICE_LABEL_MAP_FAULT => {
            bail!("online_pa CUDA label-map device fault: expected original labels -1, 0, or 1")
        }
        DEVICE_MISSING_CLASS_0_FAULT
        | DEVICE_MISSING_CLASS_1_FAULT
        | DEVICE_MISSING_CLASS_2_FAULT => bail!(
            "online_pa CUDA device-derived slack weights require all three classes; device fault code {status}"
        ),
        DEVICE_SCALER_INPUT_FAULT | DEVICE_SCALER_ARITHMETIC_FAULT | DEVICE_SCALER_OUTPUT_FAULT => {
            bail!("online_pa CUDA scaler device fault code {status}")
        }
        DEVICE_TRANSFORM_ARITHMETIC_FAULT => {
            bail!("online_pa CUDA scaler-transform device fault code {status}")
        }
        DEVICE_ARITHMETIC_REDUCTION_FAULT | DEVICE_ARITHMETIC_UPDATE_FAULT => {
            bail!("online_pa CUDA PB-v2 training device arithmetic fault code {status}")
        }
        other => bail!("online_pa CUDA full-pipeline device fault code {other}"),
    }
}
