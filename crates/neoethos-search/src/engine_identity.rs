//! Which arithmetic actually evaluated a population — recorded, not inferred.
//!
//! There are four population-evaluation engines in this workspace and they do
//! not agree with each other:
//!
//! | engine | agreement with the canonical CPU |
//! |---|---|
//! | canonical CPU (f64) | it *is* the reference |
//! | native CUDA prototype B (f64) | bit-exact at 4 096 / 20 000 / 200 000 bars |
//! | CubeCL f64 | ~0.19 % off at 200 000 bars |
//!
//! The removed f32 lane was measured 54 % wrong and took 129-430 more trades,
//! because rounding flipped stop/target comparisons and changed which trades
//! happened at all. It is no longer an executable search engine. Different f64
//! engines are still named because the remaining ~0.19 % CubeCL difference is
//! enough to make their artifacts non-interchangeable.
//!
//! So the engine is written down by the run-scoped, canonical-scope-bound
//! population receipt. A process-global observation cannot distinguish two
//! concurrent or sequential discovery runs and is therefore not evidence.

use crate::backend::AcceleratorHint;

/// A population-evaluation engine, named by both its implementation and its
/// precision — the two facts that decide whether two runs are comparable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub enum PopulationEvalEngine {
    /// The canonical CPU engine (`eval::fast_evaluate_strategy_core`), f64.
    #[serde(rename = "cpu_f64_canonical")]
    Cpu,
    /// Prototype B: the native CUDA engine in `neoethos-gpu-cuda`, f64
    /// throughout, proven bit-exact against the CPU on real EURUSD data.
    #[serde(rename = "cuda_native_f64_prototype_b")]
    CudaNativeF64,
    /// The sole CubeCL search engine, double precision end-to-end.
    #[serde(rename = "cubecl_f64")]
    CubeclF64,
}

impl PopulationEvalEngine {
    /// The name used in logs and in the persisted profile. Kept identical to
    /// the serde rename so a log line and an artifact can be grepped together.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu_f64_canonical",
            Self::CudaNativeF64 => "cuda_native_f64_prototype_b",
            Self::CubeclF64 => "cubecl_f64",
        }
    }

    /// Has this engine been shown to reproduce the canonical CPU result?
    ///
    /// Only two have. The answer is carried here rather than in a comment so a
    /// caller can gate on it instead of remembering the measurement.
    pub const fn reproduces_canonical_cpu(self) -> bool {
        match self {
            Self::Cpu | Self::CudaNativeF64 => true,
            Self::CubeclF64 => false,
        }
    }
}

/// Whether the native CUDA population engine (prototype B) is part of this
/// build, and if so whether it can actually run right now.
///
/// The distinction is the whole point. The strict backend dispatcher selects
/// Prototype B only when `runtime_available() && device_count() > 0`; it never
/// relies on the CubeCL entrypoint to intercept or substitute engines. So:
///
/// * on a `gpu-vulkan` / `gpu-rocm` build prototype B is not compiled in at all
///   and CubeCL *is* the production engine — nothing is wrong;
/// * on a `gpu-cuda` build prototype B is compiled in, and it failing the
///   runtime probe means the run is about to be evaluated by a different
///   engine than the one this build exists to use.
///
/// Conflating those two either breaks the Vulkan build or keeps the silent
/// substitution, so they are separate variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrototypeBReadiness {
    /// This binary contains no native CUDA population engine.
    NotCompiledIn,
    /// Compiled in and usable.
    Ready,
    /// Compiled in, but the runtime would not load or no device was enumerated.
    CompiledButUnavailable {
        runtime_loaded: bool,
        device_count: usize,
    },
}

/// `gpu-b-native` is the feature that actually links the CUDA engine.
/// `gpu-b-adapter` alone compiles the adapter against an honest no-CUDA stub
/// whose `device_count()` is always zero — that build contains no CUDA engine
/// either, so it must read as `NotCompiledIn` and not as a fault.
#[cfg(feature = "gpu-b-native")]
pub fn prototype_b_readiness() -> PrototypeBReadiness {
    let runtime_loaded = neoethos_gpu_cuda::runtime_available();
    let device_count = neoethos_gpu_cuda::device_count();
    if runtime_loaded && device_count > 0 {
        PrototypeBReadiness::Ready
    } else {
        PrototypeBReadiness::CompiledButUnavailable {
            runtime_loaded,
            device_count,
        }
    }
}

#[cfg(not(feature = "gpu-b-native"))]
pub fn prototype_b_readiness() -> PrototypeBReadiness {
    PrototypeBReadiness::NotCompiledIn
}

/// The accelerator families this binary actually contains a population engine
/// for.
///
/// `EvaluationBackend::accelerator_hint` is parsed and carried but no dispatch
/// arm reads it, so `cuda_required` and `vulkan_required` are today the same
/// instruction. That is the same defect shape as the prototype-B one — an arm
/// believing it named an engine — and this is what makes it checkable.
pub fn compiled_accelerators() -> Vec<AcceleratorHint> {
    let mut compiled = Vec::new();
    if cfg!(feature = "gpu-cuda") || cfg!(feature = "gpu-b-native") {
        compiled.push(AcceleratorHint::Cuda);
    }
    if cfg!(feature = "gpu-vulkan") {
        // cubecl/wgpu picks its own adapter backend at runtime, so a wgpu build
        // can legitimately satisfy any of these.
        compiled.extend([
            AcceleratorHint::Wgpu,
            AcceleratorHint::Vulkan,
            AcceleratorHint::Dx12,
            AcceleratorHint::Metal,
        ]);
    }
    if cfg!(feature = "gpu-rocm") {
        compiled.push(AcceleratorHint::Rocm);
    }
    compiled
}

/// Can this build possibly honour `hint`?
///
/// `Any` always can. A specific hint that names an accelerator this binary was
/// not built with cannot be honoured by any amount of runtime probing, and a
/// strict run must say so instead of quietly using whatever engine it has.
pub fn accelerator_hint_is_compiled(hint: AcceleratorHint) -> bool {
    hint == AcceleratorHint::Any || compiled_accelerators().contains(&hint)
}

/// Decide, before any kernel runs, whether a strict (`ForbidCpu`) population
/// evaluation may proceed — and on success, name the engine it will reach.
///
/// This is the whole of the fail-closed rule, kept as a pure function of
/// `(backend, readiness)` so all three readiness states are testable on a
/// machine with no GPU at all. The dispatch arm in `backend.rs` used to call
/// straight into `try_evaluate_population_cuda` with no precondition, so a
/// `gpu-cuda` build that could not enumerate a device substituted the distinct
/// CubeCL f64 lane and kept ranking strategies with different arithmetic.
pub fn strict_engine_preflight(
    backend: crate::backend::EvaluationBackend,
    readiness: PrototypeBReadiness,
) -> Result<PopulationEvalEngine, String> {
    // A hint this binary was not built with is a configuration fault and no
    // amount of runtime probing can satisfy it, so it is reported first.
    if !accelerator_hint_is_compiled(backend.accelerator_hint) {
        return Err(format!(
            "gpu_required requested accelerator {:?}, but this build contains a population engine \
             only for {:?}. Rebuild with the matching GPU feature, or drop the accelerator hint \
             (use `gpu_required`) to accept the engine that is present.",
            backend.accelerator_hint,
            compiled_accelerators()
        ));
    }

    match readiness {
        PrototypeBReadiness::Ready => Ok(PopulationEvalEngine::CudaNativeF64),
        PrototypeBReadiness::NotCompiledIn => Err(
            "strict Discovery routing requires the native CUDA population engine, but this build does not contain it"
                .to_string(),
        ),
        PrototypeBReadiness::CompiledButUnavailable {
            runtime_loaded,
            device_count,
        } => Err(format!(
            "gpu_required refuses to start: this build links the native CUDA population engine \
             (prototype B) but it is not usable right now — CUDA runtime loaded: {runtime_loaded}, \
             CUDA devices enumerated: {device_count}. The runtime state cannot be transformed into \
             CPU authority or a different GPU engine. {}",
            if runtime_loaded {
                "The CUDA runtime loaded but enumerated no device: check `nvidia-smi`, \
                 CUDA_VISIBLE_DEVICES, and that the card is not already claimed."
            } else {
                "The CUDA runtime did not load: check the driver and that libcudart is on the \
                 library search path."
            }
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_engine_has_a_distinct_serialized_identity() {
        let mut seen = std::collections::BTreeSet::new();
        for engine in [
            PopulationEvalEngine::Cpu,
            PopulationEvalEngine::CudaNativeF64,
            PopulationEvalEngine::CubeclF64,
        ] {
            assert!(
                seen.insert(engine.as_str()),
                "{engine:?} reuses an identity"
            );
            let json = serde_json::to_string(&engine).unwrap();
            assert_eq!(json, format!("\"{}\"", engine.as_str()));
        }
    }

    /// Only the two measured-exact engines may claim canonical arithmetic.
    #[test]
    fn only_cpu_and_native_cuda_reproduce_the_canonical_engine() {
        assert!(PopulationEvalEngine::Cpu.reproduces_canonical_cpu());
        assert!(PopulationEvalEngine::CudaNativeF64.reproduces_canonical_cpu());
        assert!(!PopulationEvalEngine::CubeclF64.reproduces_canonical_cpu());
    }

    /// The build's own answer has to match the feature set it was compiled
    /// with — this is what the strict-mode guard depends on.
    #[test]
    fn readiness_reports_not_compiled_in_exactly_when_the_engine_is_absent() {
        let readiness = prototype_b_readiness();
        if cfg!(feature = "gpu-b-native") {
            assert_ne!(readiness, PrototypeBReadiness::NotCompiledIn);
        } else {
            assert_eq!(readiness, PrototypeBReadiness::NotCompiledIn);
        }
    }

    #[test]
    fn any_hint_is_always_compiled_and_a_hint_this_build_lacks_is_not() {
        assert!(accelerator_hint_is_compiled(AcceleratorHint::Any));
        assert_eq!(
            accelerator_hint_is_compiled(AcceleratorHint::Cuda),
            cfg!(feature = "gpu-cuda") || cfg!(feature = "gpu-b-native")
        );
        assert_eq!(
            accelerator_hint_is_compiled(AcceleratorHint::Rocm),
            cfg!(feature = "gpu-rocm")
        );
    }

    /// The defect, pinned: a `gpu-cuda` build whose card vanished used to fall
    /// through to the distinct CubeCL f64 lane and keep ranking strategies. Strict mode
    /// must now refuse, and the refusal must say which half of the probe failed
    /// so the operator knows whether to look at the driver or at the card.
    #[test]
    fn strict_mode_fails_closed_when_prototype_b_is_compiled_but_unavailable() {
        for (runtime_loaded, device_count, expected_hint) in [
            (false, 0usize, "runtime did not load"),
            (true, 0usize, "enumerated no device"),
        ] {
            let error = strict_engine_preflight(
                crate::backend::EvaluationBackend::GPU_REQUIRED,
                PrototypeBReadiness::CompiledButUnavailable {
                    runtime_loaded,
                    device_count,
                },
            )
            .expect_err("a compiled-but-unavailable native engine must fail closed");
            assert!(
                error.contains(expected_hint),
                "the refusal has to name what was missing: {error}"
            );
            assert!(error.contains("cannot be transformed"), "{error}");
        }
    }

    #[test]
    fn strict_mode_refuses_a_build_without_the_native_engine() {
        let error = strict_engine_preflight(
            crate::backend::EvaluationBackend::GPU_REQUIRED,
            PrototypeBReadiness::NotCompiledIn,
        )
        .expect_err("a build without the native engine cannot execute strict Discovery");
        assert!(error.contains("does not contain it"), "{error}");
    }

    #[test]
    fn strict_mode_names_the_native_engine_when_it_is_ready() {
        let engine = strict_engine_preflight(
            crate::backend::EvaluationBackend::GPU_REQUIRED,
            PrototypeBReadiness::Ready,
        )
        .unwrap();
        assert_eq!(engine, PopulationEvalEngine::CudaNativeF64);
        assert!(engine.reproduces_canonical_cpu());
    }

    /// An accelerator this binary was not built with cannot be satisfied by any
    /// runtime probe, and today nothing reads the hint at all.
    #[test]
    fn a_hint_this_build_cannot_honour_is_refused_before_any_kernel_runs() {
        let uncompilable = [
            AcceleratorHint::Cuda,
            AcceleratorHint::Rocm,
            AcceleratorHint::Vulkan,
        ]
        .into_iter()
        .find(|hint| !accelerator_hint_is_compiled(*hint));
        let Some(hint) = uncompilable else {
            // A build that contains every engine has nothing to refuse.
            return;
        };
        let backend = crate::backend::EvaluationBackend {
            accelerator_hint: hint,
            ..crate::backend::EvaluationBackend::GPU_REQUIRED
        };
        let error = strict_engine_preflight(backend, PrototypeBReadiness::Ready)
            .expect_err("an uncompiled accelerator must be refused");
        assert!(error.contains("requested accelerator"), "{error}");
    }
}
