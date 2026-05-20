use forex_core::{AcceleratorBackend, BackendKind, ResolvedWorkloadAssignment};

pub(crate) fn accelerator_backend_from_assignment(
    assignment: &ResolvedWorkloadAssignment,
) -> AcceleratorBackend {
    match assignment.device_assignment.backend {
        BackendKind::NativeCuda | BackendKind::CudaKernel => AcceleratorBackend::Cuda,
        BackendKind::BurnWgpu => AcceleratorBackend::Wgpu,
        BackendKind::NativeCpu
        | BackendKind::BurnCpu
        | BackendKind::CpuReference
        | BackendKind::LocalSurrogateFallback
        | BackendKind::ExternalRuntime
        | BackendKind::NativeTreeGpu
        | BackendKind::NativeTreeCpu
        | BackendKind::Unavailable => AcceleratorBackend::Cpu,
    }
}
