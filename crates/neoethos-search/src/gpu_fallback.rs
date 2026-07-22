//! GPU failure classification and the fallback/fail decision.
//!
//! Task 5 of the GPU remediation (2026-07-22): the codebase read
//! `NEOETHOS_REQUIRE_GPU` inline in five places and treated every GPU failure
//! the same way — warn, then recompute on the CPU. That is correct for an
//! *availability* failure (no device, an allocation that could not be served, a
//! driver that lost the device): the CPU evaluator is the canonical reference,
//! so a slow-but-correct recompute preserves the never-OOM invariant.
//!
//! It is NOT correct for a *correctness* failure. If a GPU launch returned a
//! number that disagrees with the CPU reference beyond tolerance, that number
//! is already wrong; silently swapping in a CPU recompute would hide a real
//! kernel bug, and accepting the GPU number would corrupt output. Such a result
//! must fail loud, regardless of any environment flag.
//!
//! This module is the single source of truth for both decisions. It is pure
//! logic with no GPU dependency, so every branch is unit-tested on any machine.

/// Why a GPU evaluation attempt did not yield a usable result.
///
/// The split is the whole point: `Availability` is a reason the GPU *could not
/// run*, `Correctness` is a reason its output *cannot be trusted*.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuFailure {
    /// No usable adapter/device was found for the requested backend.
    NoAdapter,
    /// The compiled backend cannot execute on this machine (e.g. a CUDA build
    /// on a box with no CUDA driver, or an unsupported wgpu backend).
    UnsupportedBackend,
    /// A device allocation could not be served (VRAM pressure, pool
    /// exhaustion — including the cubecl#243 pool panic surfaced as a failure).
    AllocationPressure,
    /// The device was lost mid-execution (driver reset, timeout, unplug).
    DeviceLost,
    /// The GPU launch produced the wrong SHAPE (e.g. a gene-count mismatch).
    /// The lane malfunctioned; a CPU recompute yields the correct result, so
    /// this is an availability-class fault, not a trusted-but-wrong number.
    WrongShape,
    /// The GPU produced a plausibly-shaped result that disagrees with the CPU
    /// reference beyond tolerance. The output is wrong and there is no safe
    /// fallback — the GPU already "succeeded" with a bad number.
    ParityViolation,
}

impl GpuFailure {
    /// True when the failure means the GPU could not run (so a CPU recompute is
    /// a valid substitute), as opposed to running and producing a wrong answer.
    pub fn is_availability(self) -> bool {
        matches!(
            self,
            GpuFailure::NoAdapter
                | GpuFailure::UnsupportedBackend
                | GpuFailure::AllocationPressure
                | GpuFailure::DeviceLost
                | GpuFailure::WrongShape
        )
    }
}

/// What the caller should do about a [`GpuFailure`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackDecision {
    /// Recompute this work on the CPU. Slow but correct; the never-OOM path.
    RecomputeOnCpu,
    /// Fail loud. Either the operator required a GPU and an availability fault
    /// must not be hidden behind a silent CPU run, or the GPU returned a
    /// number that cannot be trusted.
    FailLoud,
}

/// The operator's assertion that a GPU MUST be used for this process.
///
/// Set `NEOETHOS_REQUIRE_GPU=1` on a real GPU box so a device/driver misconfig
/// fails loud instead of silently running the whole search on the CPU (which
/// looks like "it works, just slowly" and wastes rented card-hours). Any
/// non-empty value counts as set, matching the historical `is_ok()` check.
pub fn require_gpu() -> bool {
    std::env::var("NEOETHOS_REQUIRE_GPU")
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false)
}

/// Decide what to do about a GPU failure.
///
/// - A correctness failure (parity violation) ALWAYS fails loud: a wrong number
///   is never safe to accept, and hiding it behind a CPU recompute would mask a
///   kernel bug.
/// - An availability failure falls back to the CPU UNLESS the operator required
///   a GPU, in which case it fails loud so a misconfigured box does not run the
///   whole workload slowly on the CPU without anyone noticing.
pub fn decide(failure: GpuFailure, require_gpu: bool) -> FallbackDecision {
    if !failure.is_availability() {
        // Correctness fault: never fall back, never accept.
        return FallbackDecision::FailLoud;
    }
    if require_gpu {
        FallbackDecision::FailLoud
    } else {
        FallbackDecision::RecomputeOnCpu
    }
}

/// Convenience: classify + decide against the live `NEOETHOS_REQUIRE_GPU` env.
pub fn decide_env(failure: GpuFailure) -> FallbackDecision {
    decide(failure, require_gpu())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn availability_faults_fall_back_when_gpu_is_optional() {
        for f in [
            GpuFailure::NoAdapter,
            GpuFailure::UnsupportedBackend,
            GpuFailure::AllocationPressure,
            GpuFailure::DeviceLost,
            GpuFailure::WrongShape,
        ] {
            assert_eq!(
                decide(f, false),
                FallbackDecision::RecomputeOnCpu,
                "{f:?} should recompute on CPU when GPU is optional"
            );
        }
    }

    #[test]
    fn availability_faults_fail_loud_when_gpu_is_required() {
        // A rented card sitting idle while the run silently uses the CPU is the
        // exact waste NEOETHOS_REQUIRE_GPU exists to prevent.
        for f in [
            GpuFailure::NoAdapter,
            GpuFailure::UnsupportedBackend,
            GpuFailure::AllocationPressure,
            GpuFailure::DeviceLost,
            GpuFailure::WrongShape,
        ] {
            assert_eq!(
                decide(f, true),
                FallbackDecision::FailLoud,
                "{f:?} must fail loud when a GPU is required"
            );
        }
    }

    #[test]
    fn parity_violation_always_fails_loud() {
        // A wrong-but-plausible number is never safe: fail closed whether or not
        // a GPU was required, and whatever the env says.
        assert_eq!(
            decide(GpuFailure::ParityViolation, false),
            FallbackDecision::FailLoud
        );
        assert_eq!(
            decide(GpuFailure::ParityViolation, true),
            FallbackDecision::FailLoud
        );
        assert!(!GpuFailure::ParityViolation.is_availability());
    }

    #[test]
    fn wrong_shape_is_availability_not_correctness() {
        // A gene-count mismatch means the lane malfunctioned; the CPU recompute
        // is correct, so it is availability-class, not a trusted-wrong number.
        assert!(GpuFailure::WrongShape.is_availability());
        assert_eq!(
            decide(GpuFailure::WrongShape, false),
            FallbackDecision::RecomputeOnCpu
        );
    }

    #[test]
    fn require_gpu_reads_any_nonempty_value() {
        // The historical check was `is_ok()`, so any set value counted. Keep an
        // empty string as "not required" (a common accidental unset shape).
        // SAFETY: single-threaded test; serialized by the #[serial]-free
        // convention of this file — no other test in this module reads the env.
        let key = "NEOETHOS_REQUIRE_GPU";
        let prev = std::env::var(key).ok();
        unsafe {
            std::env::set_var(key, "1");
            assert!(require_gpu());
            std::env::set_var(key, "0");
            assert!(require_gpu(), "any non-empty value counts as required");
            std::env::set_var(key, "");
            assert!(!require_gpu(), "empty string is not required");
            std::env::remove_var(key);
            assert!(!require_gpu(), "unset is not required");
            match prev {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
}
