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
//! | CubeCL f32 | **54 % off at 200 000 bars** (net profit 3 940.88 against 8 506.33) |
//!
//! The f32 figure is not a rounding remark. `cubecl_eval.rs` records that the
//! f32 lane took 129-430 *more trades* over the same series, because rounding
//! flipped stop/target comparisons and changed which trades happen at all. Two
//! runs that used different engines therefore ranked different strategies, and
//! nothing in the artifacts said which engine had run — which is precisely how
//! the f32/f64 confusion survived long enough to be measured twice.
//!
//! So the engine is written down. Every lane records itself the moment it
//! executes, the first sighting of each engine is logged at INFO, and
//! [`observed_population_engines`] is persisted into the discovery run profile.
//! A run that cannot name its own arithmetic is not evidence.

use crate::backend::AcceleratorHint;
use std::sync::atomic::{AtomicU8, Ordering};

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
    /// The CubeCL kernel in single precision — the lane measured 54 % wrong at
    /// production series length.
    #[serde(rename = "cubecl_f32")]
    CubeclF32,
    /// The CubeCL kernel in double precision (`NEOETHOS_GPU_F64=1`).
    #[serde(rename = "cubecl_f64")]
    CubeclF64,
}

impl PopulationEvalEngine {
    const ORDERED: [Self; 4] = [
        Self::Cpu,
        Self::CudaNativeF64,
        Self::CubeclF32,
        Self::CubeclF64,
    ];

    const fn bit(self) -> u8 {
        match self {
            Self::Cpu => 1 << 0,
            Self::CudaNativeF64 => 1 << 1,
            Self::CubeclF32 => 1 << 2,
            Self::CubeclF64 => 1 << 3,
        }
    }

    /// The name used in logs and in the persisted profile. Kept identical to
    /// the serde rename so a log line and an artifact can be grepped together.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu_f64_canonical",
            Self::CudaNativeF64 => "cuda_native_f64_prototype_b",
            Self::CubeclF32 => "cubecl_f32",
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
            Self::CubeclF32 | Self::CubeclF64 => false,
        }
    }
}

fn observed_bits() -> &'static AtomicU8 {
    static OBSERVED: AtomicU8 = AtomicU8::new(0);
    &OBSERVED
}

/// Record that `engine` just evaluated a population.
///
/// A bitmask rather than a single slot: one discovery run drives the GA, the
/// Monte-Carlo quality screen, walk-forward and CPCV through this boundary, and
/// an optional backend is free to use different engines for different stages.
/// Collapsing that to "the" engine would be the same inference this module
/// exists to remove.
///
/// The first sighting of each engine is logged at INFO. Subsequent calls are a
/// single relaxed atomic, so this is safe on the per-generation hot path.
pub fn record_population_engine(engine: PopulationEvalEngine) {
    let previous = observed_bits().fetch_or(engine.bit(), Ordering::Relaxed);
    if previous & engine.bit() == 0 {
        tracing::info!(
            target: "neoethos_search::engine",
            engine = engine.as_str(),
            reproduces_canonical_cpu = engine.reproduces_canonical_cpu(),
            "population evaluated by this engine (first use in this process)"
        );
    }
}

/// Every engine that has evaluated a population in this process, in a stable
/// order so two profiles can be compared field-for-field.
pub fn observed_population_engines() -> Vec<PopulationEvalEngine> {
    let bits = observed_bits().load(Ordering::Relaxed);
    PopulationEvalEngine::ORDERED
        .into_iter()
        .filter(|engine| bits & engine.bit() != 0)
        .collect()
}

/// Clear the record. For tests that need to assert on a single run; production
/// never calls this, because forgetting which engine ran is the defect.
pub fn reset_observed_population_engines() {
    observed_bits().store(0, Ordering::Relaxed);
}

/// Whether the native CUDA population engine (prototype B) is part of this
/// build, and if so whether it can actually run right now.
///
/// The distinction is the whole point. `try_evaluate_population_cuda`
/// short-circuits to prototype B only when `runtime_available() && device_count()
/// > 0`, and otherwise continues into the CubeCL lane. So:
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
/// `gpu-cuda` build that could not enumerate a device fell through to the
/// CubeCL f32 lane and kept ranking strategies with different arithmetic.
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
        // Prototype B is not part of this build at all — `gpu-vulkan` and
        // `gpu-rocm` ship exactly like this, and CubeCL is their production
        // engine. Strict mode there is legitimate and must keep working.
        PrototypeBReadiness::NotCompiledIn => Ok(PopulationEvalEngine::CubeclF32),
        PrototypeBReadiness::CompiledButUnavailable {
            runtime_loaded,
            device_count,
        } => Err(format!(
            "gpu_required refuses to start: this build links the native CUDA population engine \
             (prototype B) but it is not usable right now — CUDA runtime loaded: {runtime_loaded}, \
             CUDA devices enumerated: {device_count}. Continuing would evaluate the population on \
             the CubeCL f32 lane instead, which is measured 54 % wrong at 200 000 bars (net profit \
             3 940.88 against the canonical 8 506.33) and takes 129-430 more trades, so it would \
             silently change which strategies this run selects. {} Or choose a non-strict backend \
             (`auto`) if CPU-equivalent results are acceptable.",
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
    fn every_engine_has_a_distinct_bit_and_survives_a_round_trip() {
        let mut seen = 0u8;
        for engine in PopulationEvalEngine::ORDERED {
            assert_eq!(seen & engine.bit(), 0, "{engine:?} reuses a bit");
            seen |= engine.bit();
            let json = serde_json::to_string(&engine).unwrap();
            assert_eq!(json, format!("\"{}\"", engine.as_str()));
        }
    }

    /// Only the two measured-exact engines may claim canonical arithmetic.
    #[test]
    fn only_cpu_and_native_cuda_reproduce_the_canonical_engine() {
        assert!(PopulationEvalEngine::Cpu.reproduces_canonical_cpu());
        assert!(PopulationEvalEngine::CudaNativeF64.reproduces_canonical_cpu());
        assert!(!PopulationEvalEngine::CubeclF32.reproduces_canonical_cpu());
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
    /// through to the CubeCL f32 lane and keep ranking strategies. Strict mode
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
            // And it has to say why falling through would not be harmless.
            assert!(error.contains("54 %"), "{error}");
        }
    }

    /// `gpu-vulkan` and `gpu-rocm` do not contain prototype B at all, so CubeCL
    /// is their production engine and strict mode there is legitimate.
    /// Conflating that with the case above would break those builds.
    #[test]
    fn strict_mode_still_runs_cubecl_where_the_native_engine_does_not_exist() {
        let engine = strict_engine_preflight(
            crate::backend::EvaluationBackend::GPU_REQUIRED,
            PrototypeBReadiness::NotCompiledIn,
        )
        .expect("a build without the native engine may use CubeCL under strict mode");
        assert_eq!(engine, PopulationEvalEngine::CubeclF32);
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

    /// The recorder accumulates; it is never cleared in production, which is
    /// what lets the discovery profile name every engine the run used.
    #[test]
    fn recording_an_engine_makes_it_observable() {
        record_population_engine(PopulationEvalEngine::CubeclF64);
        assert!(observed_population_engines().contains(&PopulationEvalEngine::CubeclF64));
        // Recording twice is idempotent, so a per-generation call site is safe.
        record_population_engine(PopulationEvalEngine::CubeclF64);
        assert_eq!(
            observed_population_engines()
                .iter()
                .filter(|engine| **engine == PopulationEvalEngine::CubeclF64)
                .count(),
            1
        );
    }
}
