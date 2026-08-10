//! GPU failure classification and typed fallback/retry decisions.
//!
//! Stage 1 / Commit 0.2 of the GPU-native discovery redesign.  Availability,
//! correctness and allocation-pressure failures are intentionally distinct:
//! strict GPU execution forbids CPU fallback, but it still permits bounded,
//! deterministic GPU rebatching when an allocation is too large.

use crate::backend::{EvaluationBackend, FallbackPolicy};

/// Why a GPU evaluation attempt did not yield a usable result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuFailure {
    /// No usable adapter/device was found for the requested backend.
    NoAdapter,
    /// The compiled backend cannot execute on this machine.
    UnsupportedBackend,
    /// A device allocation could not be served.
    AllocationPressure,
    /// The device was lost mid-execution.
    DeviceLost,
    /// The GPU returned a result with an invalid shape.  This is an internal
    /// backend/kernel malfunction, not an availability fault.
    WrongShape,
    /// The GPU returned plausibly-shaped values that violate canonical parity.
    ParityViolation,
}

impl GpuFailure {
    pub fn is_availability(self) -> bool {
        matches!(
            self,
            GpuFailure::NoAdapter | GpuFailure::UnsupportedBackend | GpuFailure::DeviceLost
        )
    }

    pub fn is_correctness(self) -> bool {
        matches!(self, GpuFailure::WrongShape | GpuFailure::ParityViolation)
    }
}

/// Action selected by the typed GPU failure policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuAction {
    /// Retry the exact same logical work on the GPU with a smaller physical
    /// batch. IDs, ordering and RNG counters must remain unchanged.
    RetryOnGpu { next_batch_size: usize },
    /// Recompute on the CPU. Legal only when the resolved backend allows it.
    FallbackToCpu,
    /// Stop the work unit and surface the failure.
    FailLoud,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuRetryPolicy {
    pub max_retries: u32,
    pub min_batch_size: usize,
}

impl Default for GpuRetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            min_batch_size: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuAttempt {
    /// Zero-based retry count. `0` is the first failed attempt.
    pub retry_index: u32,
    pub batch_size: usize,
}

impl GpuAttempt {
    pub fn first(batch_size: usize) -> Self {
        Self {
            retry_index: 0,
            batch_size: batch_size.max(1),
        }
    }
}

/// Decide what a caller must do after a GPU failure.
///
/// Correctness failures always fail closed. Allocation pressure retries on the
/// same GPU while the bounded retry policy can still reduce the batch. Only
/// after GPU retries are exhausted may an optional-GPU run fall back to CPU.
pub fn decide_action(
    failure: GpuFailure,
    backend: EvaluationBackend,
    attempt: GpuAttempt,
    retry_policy: GpuRetryPolicy,
) -> GpuAction {
    if failure.is_correctness() {
        return GpuAction::FailLoud;
    }

    if failure == GpuFailure::AllocationPressure {
        let min_batch_size = retry_policy.min_batch_size.max(1);
        let retries_remain = attempt.retry_index < retry_policy.max_retries;
        let can_shrink = attempt.batch_size > min_batch_size;
        if retries_remain && can_shrink {
            let halved = attempt.batch_size.div_ceil(2);
            return GpuAction::RetryOnGpu {
                next_batch_size: halved.max(min_batch_size),
            };
        }
    }

    if backend.fallback == FallbackPolicy::AllowCpu {
        GpuAction::FallbackToCpu
    } else {
        GpuAction::FailLoud
    }
}

/// Legacy two-way decision retained while existing call sites migrate to
/// [`GpuAction`]. New code must use [`decide_action`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackDecision {
    RecomputeOnCpu,
    FailLoud,
}

/// Legacy compatibility wrapper. It has no batch context, so allocation
/// pressure cannot retry here; migrated call sites use [`decide_action`].
pub fn decide(failure: GpuFailure, require_gpu: bool) -> FallbackDecision {
    if failure.is_correctness() || require_gpu {
        FallbackDecision::FailLoud
    } else {
        FallbackDecision::RecomputeOnCpu
    }
}

/// Whether this run refuses a CPU recompute — read from the RESOLVED backend,
/// not from the process environment.
///
/// 2026-08-10 (env→config wave): this was `env::var("NEOETHOS_REQUIRE_GPU")`.
/// Two things were wrong with that beyond the env read itself. First, the
/// crate had two different answers to one question — `backend.rs` resolved
/// `FallbackPolicy` from config while every fallback site here consulted an
/// env var that config could not set. Second, the env value was read on a hot
/// path with no record of it in the resolved backend, so `device_summary`
/// could report `AllowCpu` while this function was failing the run loud.
///
/// Now there is one source: the backend installed by
/// `install_evaluation_backend_from_settings`. `gpu_required()` is
/// `Gpu + ForbidCpu`, which is exactly what `*_required` in
/// `system.enable_gpu_preference` / `models.prop_search_device` produces.
///
/// The function name is kept because `decide_env` and existing tests call it.
pub fn require_gpu() -> bool {
    crate::backend::current_evaluation_backend().gpu_required()
}

/// Name kept for the existing call sites; the `_env` suffix is now a lie about
/// history, not about behaviour — see [`require_gpu`].
pub fn decide_env(failure: GpuFailure) -> FallbackDecision {
    decide(failure, require_gpu())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::EvaluationBackend;

    #[test]
    fn correctness_faults_always_fail_loud() {
        for failure in [GpuFailure::WrongShape, GpuFailure::ParityViolation] {
            for backend in [
                EvaluationBackend::AUTO,
                EvaluationBackend::GPU_PREFERRED,
                EvaluationBackend::GPU_REQUIRED,
            ] {
                assert_eq!(
                    decide_action(
                        failure,
                        backend,
                        GpuAttempt::first(128),
                        GpuRetryPolicy::default(),
                    ),
                    GpuAction::FailLoud
                );
            }
        }
        assert!(!GpuFailure::WrongShape.is_availability());
        assert!(GpuFailure::WrongShape.is_correctness());
    }

    #[test]
    fn optional_gpu_availability_fault_falls_back() {
        for failure in [
            GpuFailure::NoAdapter,
            GpuFailure::UnsupportedBackend,
            GpuFailure::DeviceLost,
        ] {
            assert_eq!(
                decide_action(
                    failure,
                    EvaluationBackend::GPU_PREFERRED,
                    GpuAttempt::first(64),
                    GpuRetryPolicy::default(),
                ),
                GpuAction::FallbackToCpu
            );
        }
    }

    #[test]
    fn required_gpu_availability_fault_fails_loud() {
        for failure in [
            GpuFailure::NoAdapter,
            GpuFailure::UnsupportedBackend,
            GpuFailure::DeviceLost,
        ] {
            assert_eq!(
                decide_action(
                    failure,
                    EvaluationBackend::GPU_REQUIRED,
                    GpuAttempt::first(64),
                    GpuRetryPolicy::default(),
                ),
                GpuAction::FailLoud
            );
        }
    }

    #[test]
    fn allocation_pressure_rebatches_before_fallback() {
        let action = decide_action(
            GpuFailure::AllocationPressure,
            EvaluationBackend::GPU_REQUIRED,
            GpuAttempt::first(101),
            GpuRetryPolicy {
                max_retries: 3,
                min_batch_size: 8,
            },
        );
        assert_eq!(
            action,
            GpuAction::RetryOnGpu {
                next_batch_size: 51
            }
        );
    }

    #[test]
    fn required_gpu_fails_after_retries_are_exhausted() {
        let action = decide_action(
            GpuFailure::AllocationPressure,
            EvaluationBackend::GPU_REQUIRED,
            GpuAttempt {
                retry_index: 3,
                batch_size: 8,
            },
            GpuRetryPolicy {
                max_retries: 3,
                min_batch_size: 8,
            },
        );
        assert_eq!(action, GpuAction::FailLoud);
    }

    #[test]
    fn optional_gpu_may_fallback_after_retries_are_exhausted() {
        let action = decide_action(
            GpuFailure::AllocationPressure,
            EvaluationBackend::GPU_PREFERRED,
            GpuAttempt {
                retry_index: 3,
                batch_size: 8,
            },
            GpuRetryPolicy {
                max_retries: 3,
                min_batch_size: 8,
            },
        );
        assert_eq!(action, GpuAction::FallbackToCpu);
    }

    #[test]
    fn retry_sequence_is_deterministic() {
        let policy = GpuRetryPolicy {
            max_retries: 8,
            min_batch_size: 4,
        };
        let mut batch = 100;
        let mut sequence = Vec::new();
        for retry_index in 0..8 {
            match decide_action(
                GpuFailure::AllocationPressure,
                EvaluationBackend::GPU_REQUIRED,
                GpuAttempt {
                    retry_index,
                    batch_size: batch,
                },
                policy,
            ) {
                GpuAction::RetryOnGpu { next_batch_size } => {
                    sequence.push(next_batch_size);
                    batch = next_batch_size;
                }
                GpuAction::FailLoud => break,
                GpuAction::FallbackToCpu => panic!("strict GPU mode must never fall back"),
            }
        }
        assert_eq!(sequence, vec![50, 25, 13, 7, 4]);
    }

    /// Replaces `legacy_env_parser_handles_false_values`, deleted 2026-08-10
    /// with the env parser it tested. The question it was really asking —
    /// "does a strict-GPU run refuse the CPU?" — is now answered by the
    /// resolved backend, so that is what is pinned here.
    #[test]
    fn the_cpu_refusal_comes_from_the_resolved_backend() {
        // GPU + ForbidCpu is the only shape that refuses; nothing else does.
        assert!(EvaluationBackend::GPU_REQUIRED.gpu_required());
        assert!(!EvaluationBackend::GPU_PREFERRED.gpu_required());
        assert!(!EvaluationBackend::AUTO.gpu_required());
        // And `require_gpu()` reports whatever backend is installed for the
        // process rather than an ambient variable nothing recorded.
        assert_eq!(
            require_gpu(),
            crate::backend::current_evaluation_backend().gpu_required()
        );
    }

    #[test]
    fn legacy_wrong_shape_is_never_hidden() {
        assert_eq!(
            decide(GpuFailure::WrongShape, false),
            FallbackDecision::FailLoud
        );
        assert_eq!(
            decide(GpuFailure::WrongShape, true),
            FallbackDecision::FailLoud
        );
    }
}
