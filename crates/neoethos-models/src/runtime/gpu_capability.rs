//! Honest per-model GPU capability.
//!
//! Before this module existed, `registry.rs` answered "does this model support
//! the GPU?" with three hand-written `cfg!(feature = …)` calls that named
//! features NO shipped build enables:
//!
//! * `lightgbm-gpu` — enabled by nothing, and `lightgbm3` provably does not
//!   build a GPU tree learner (see `tree_models::lightgbm::effective_device_type`).
//! * `reinforcement-learning-cuda` — enabled by nothing, because `gpu-cuda`
//!   spelled out `rlkit/cuda` + `candle-core/cuda` directly instead of going
//!   through the aggregate.
//! * `burn-wgpu-backend` — the VULKAN backend, used as the key for the whole
//!   Deep/Exit family and `sac`. A CUDA build enables `burn-cuda-backend`
//!   instead (or neither), so a `--features gpu-cuda` binary on an RTX 3090
//!   reported "no GPU" for every deep model.
//!
//! The net effect on a CUDA build was that the program advertised GPU support
//! for XGBoost and CatBoost alone, and the UI repeated it.
//!
//! The fix is structural, not a flag swap:
//!
//! 1. [`declared_gpu_path`] is the ONE table mapping a model to the Cargo
//!    feature(s) that compile its GPU code path in, and to the accelerator
//!    family that path can actually execute on.
//! 2. [`feature_compiled`] is the ONE place a feature name becomes a `cfg!`.
//!    Advertised and linked cannot drift, because there is a single conversion.
//! 3. [`accelerator_present`] answers the OTHER half of the honest question —
//!    a compiled CUDA path on a machine with no NVIDIA device is not a GPU
//!    capability, it is a CPU fallback waiting to happen.
//! 4. `gpu_capability_feature_reachability` (tests) fails the build when a
//!    feature named here is not reachable from any shipped aggregate. That is
//!    the general form of the bug and would have caught all three cases above.
//!
//! Compiling a GPU path in must never, by itself, change what a model
//! produces. Where a newly linked path would run by default, the spec carries
//! an [`GpuPathSpec::opt_in`] config value and the runtime default stays where
//! it was — see [`ModelGpuRuntimeOverrides`].

use std::sync::OnceLock;

use neoethos_core::system::HardwareProbe;
use neoethos_core::{AcceleratorBackend, Settings};

use crate::runtime::capabilities::ModelFamily;

/// The device family a compiled GPU code path can execute on.
///
/// This is a real distinction, not bookkeeping: a `gpu-vulkan` build on an
/// NVIDIA card has a Burn/wgpu path but no CUDA path, and a `gpu-cuda` build
/// on an AMD card has the opposite. Reporting "GPU" without saying which
/// family is how `burn-wgpu-backend` came to stand in for CUDA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuAccelerator {
    /// NVIDIA CUDA driver API — candle, cubecl-cuda, XGBoost `device=cuda`,
    /// CatBoost `task_type=GPU`, burn-cuda.
    Cuda,
    /// Any wgpu-family adapter: Vulkan, Metal, DX12 (and ROCm cards reached
    /// through radv/amdvlk). Burn's wgpu backend.
    Wgpu,
}

impl GpuAccelerator {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Wgpu => "wgpu",
        }
    }
}

/// A GPU code path that exists in the tree for one model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuPathSpec {
    /// Cargo features of `neoethos-models`, ANY of which compiles this path.
    ///
    /// More than one entry means genuinely alternative implementations (the
    /// Burn family compiles against either the CUDA or the wgpu backend), not
    /// two names for one capability.
    pub features: &'static [&'static str],
    /// Devices the compiled path executes on.
    pub accelerator: GpuAccelerator,
    /// Config value the operator sets before the compiled path is taken.
    /// `None` = the path runs whenever the hardware is present.
    pub opt_in: Option<&'static str>,
}

/// XGBoost family, plus the four meta models that train through the same
/// expert (`meta_blender`, `probability_calibrator`, `conformal_gate`,
/// `meta_stack` — their runtime backend string is literally
/// `xgboost_meta_blender`). `xgb/cuda` is what makes the linked libxgboost
/// CUDA-capable, and `xgboost-gpu` is the feature that turns it on; the
/// `xgboost` feature alone links a CPU-only library.
const XGBOOST_CUDA: GpuPathSpec = GpuPathSpec {
    features: &["xgboost-gpu"],
    accelerator: GpuAccelerator::Cuda,
    opt_in: None,
};

/// CatBoost. `catboost-rust` is declared with `features = ["gpu"]`
/// unconditionally, so the `catboost` feature IS the GPU capability; the
/// expert switches to `task_type=GPU` when a device is present.
const CATBOOST_CUDA: GpuPathSpec = GpuPathSpec {
    features: &["catboost"],
    accelerator: GpuAccelerator::Cuda,
    opt_in: None,
};

/// DQN. `reinforcement-learning-cuda` compiles the candle CUDA device
/// resolution in. Opt-in because the resolver treats `auto` as "take CUDA
/// device 0", so linking the path would otherwise move training onto the card
/// and change the learned policy.
const RL_CUDA: GpuPathSpec = GpuPathSpec {
    features: &["reinforcement-learning-cuda"],
    accelerator: GpuAccelerator::Cuda,
    opt_in: Some("models.gpu_runtime.rl_device = \"gpu\""),
};

/// `neat` / `neuro_evo` CubeCL fitness kernels.
const NEURO_EVOLUTION_CUDA: GpuPathSpec = GpuPathSpec {
    features: &["neuro-evolution-gpu"],
    accelerator: GpuAccelerator::Cuda,
    opt_in: Some("models.gpu_runtime.neuro_evolution_device = \"gpu\""),
};

/// `elasticnet` / `logistic` CubeCL elasticnet + softmax fit/predict kernels.
const STATISTICAL_CUDA: GpuPathSpec = GpuPathSpec {
    features: &["statistical-gpu"],
    accelerator: GpuAccelerator::Cuda,
    opt_in: Some("models.gpu_runtime.statistical_device = \"gpu\""),
};

/// Every Burn-backed model: the Deep and Exit families plus `sac`.
///
/// `burn-cuda-backend` wins over `burn-wgpu-backend` when both are compiled,
/// mirroring the backend alias selection in `burn_models.rs`, so the reported
/// accelerator is the one the binary will really use.
const BURN_NEURAL: GpuPathSpec = GpuPathSpec {
    features: &["burn-cuda-backend", "burn-wgpu-backend"],
    accelerator: if cfg!(feature = "burn-cuda-backend") {
        GpuAccelerator::Cuda
    } else {
        GpuAccelerator::Wgpu
    },
    opt_in: None,
};

/// The GPU code path that exists in the tree for `name`, whether or not this
/// binary compiled it.
///
/// `None` means there is no GPU implementation to advertise under ANY feature
/// combination — the honest answer for LightGBM (no GPU tree learner exists in
/// the linked library), for the linfa-backed `bayes_logit`, for the online and
/// anomaly families, and for `genetic` (strategy search runs in
/// `neoethos-search`, not here).
pub fn declared_gpu_path(name: &str, family: ModelFamily) -> Option<GpuPathSpec> {
    match name {
        // LightGBM's GPU/CUDA tree learner is not enabled in the built library
        // ("[LightGBM] [Fatal] GPU Tree Learner was not enabled in this build")
        // and the OpenCL path does not link. `effective_device_type` returns
        // "cpu" unconditionally, so there is nothing here to advertise.
        "lightgbm" => None,
        "xgboost" | "xgboost_rf" | "xgboost_dart" | "meta_blender"
        | "probability_calibrator" | "conformal_gate" | "meta_stack" => Some(XGBOOST_CUDA),
        "catboost" | "catboost_alt" => Some(CATBOOST_CUDA),
        "dqn" => Some(RL_CUDA),
        "neat" | "neuro_evo" => Some(NEURO_EVOLUTION_CUDA),
        "elasticnet" | "logistic" => Some(STATISTICAL_CUDA),
        // SAC is a Burn model like the Deep/Exit families, not an rlkit one.
        "sac" => Some(BURN_NEURAL),
        _ => match family {
            ModelFamily::Deep | ModelFamily::Exit => Some(BURN_NEURAL),
            _ => None,
        },
    }
}

/// The ONLY place a capability feature name becomes a `cfg!`.
///
/// Adding a model's GPU path means adding its feature here AND to
/// [`declared_gpu_path`]; the reachability test then proves a shipped build can
/// actually reach it.
pub fn feature_compiled(feature: &str) -> bool {
    match feature {
        "xgboost-gpu" => cfg!(feature = "xgboost-gpu"),
        "catboost" => cfg!(feature = "catboost"),
        "reinforcement-learning-cuda" => cfg!(feature = "reinforcement-learning-cuda"),
        "neuro-evolution-gpu" => cfg!(feature = "neuro-evolution-gpu"),
        "statistical-gpu" => cfg!(feature = "statistical-gpu"),
        "burn-cuda-backend" => cfg!(feature = "burn-cuda-backend"),
        "burn-wgpu-backend" => cfg!(feature = "burn-wgpu-backend"),
        _ => false,
    }
}

/// Is a GPU code path for this model linked into THIS binary?
pub fn compiled_gpu_path(name: &str, family: ModelFamily) -> Option<GpuPathSpec> {
    let spec = declared_gpu_path(name, family)?;
    spec.features
        .iter()
        .any(|feature| feature_compiled(feature))
        .then_some(spec)
}

/// Is a device of this family attached to the machine right now?
///
/// Probed once and cached: `HardwareProbe` shells out to `nvidia-smi`, which is
/// far too expensive for a per-model capability lookup.
///
/// `HardwareProbe` only reports backends the binary compiled a detector for
/// (`neoethos-core/gpu-cuda`, `…/gpu-rocm`, `…/gpu-wgpu`). That is the right
/// primary signal, but it is not the whole truth: the tree experts decide with
/// `tree_models::config::gpu_count()`, which shells out to `nvidia-smi`
/// regardless of features. CatBoost therefore really can reach a card from a
/// default build. Falling back to that count keeps this report and the
/// experts' own behaviour from disagreeing.
pub fn accelerator_present(accelerator: GpuAccelerator) -> bool {
    static PRESENT: OnceLock<(bool, bool)> = OnceLock::new();
    let (cuda, wgpu) = *PRESENT.get_or_init(|| {
        let mut probe = HardwareProbe::new();
        let profile = probe.detect();
        let cuda = profile
            .accelerator_devices
            .iter()
            .any(|device| device.backend == AcceleratorBackend::Cuda);
        let wgpu = profile.accelerator_devices.iter().any(|device| {
            device.backend.is_wgpu_family() || device.backend == AcceleratorBackend::Rocm
        });
        if cuda || wgpu {
            return (cuda, wgpu);
        }
        let counted = crate::tree_models::config::gpu_count() > 0;
        (counted, counted)
    });
    match accelerator {
        GpuAccelerator::Cuda => cuda,
        GpuAccelerator::Wgpu => wgpu,
    }
}

/// What this binary can actually do for one model, split into the two facts
/// that were previously conflated into a single misreported boolean.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelGpuCapability {
    pub model: String,
    /// A GPU code path exists in the tree for this model under some feature.
    pub declared: bool,
    /// That path is linked into THIS binary.
    pub compiled: bool,
    /// A device of the right family is attached to THIS machine.
    pub device_present: bool,
    /// Accelerator family of the compiled (or, if not compiled, declared) path.
    pub accelerator: Option<GpuAccelerator>,
    /// The Cargo feature that supplies the path in this binary, or the
    /// preferred one if none is compiled.
    pub feature: Option<&'static str>,
    /// Config value the operator must set before the path is taken.
    pub opt_in: Option<&'static str>,
}

impl ModelGpuCapability {
    /// The honest answer to "can this binary run this model on the GPU here?"
    pub fn usable(&self) -> bool {
        self.compiled && self.device_present
    }
}

/// Resolve the full capability record for one model.
pub fn model_gpu_capability(name: &str, family: ModelFamily) -> ModelGpuCapability {
    let declared = declared_gpu_path(name, family);
    let compiled = compiled_gpu_path(name, family);
    let feature = declared.and_then(|spec| {
        spec.features
            .iter()
            .find(|feature| feature_compiled(feature))
            .or_else(|| spec.features.first())
            .copied()
    });
    ModelGpuCapability {
        model: name.to_string(),
        declared: declared.is_some(),
        compiled: compiled.is_some(),
        device_present: compiled
            .map(|spec| accelerator_present(spec.accelerator))
            .unwrap_or(false),
        accelerator: declared.map(|spec| spec.accelerator),
        feature,
        opt_in: declared.and_then(|spec| spec.opt_in),
    }
}

/// The whole matrix, in `KNOWN_MODEL_NAMES` order.
///
/// This is what a diagnostics surface should print instead of a bare boolean:
/// it separates "the build cannot" from "this machine cannot" from "the
/// operator has not asked".
pub fn gpu_capability_matrix() -> Vec<ModelGpuCapability> {
    crate::runtime::capabilities::KNOWN_MODEL_NAMES
        .iter()
        .filter_map(|name| {
            crate::runtime::capabilities::model_capability(name)
                .map(|capability| model_gpu_capability(name, capability.family))
        })
        .collect()
}

// ============================================================================
// RUNTIME DEVICE POLICY (config, not env)
// ============================================================================

/// Process-wide device policy for the non-tree accelerated families, installed
/// once from the operator's `Settings` at startup by
/// [`install_model_gpu_runtime_from_settings`].
///
/// Mirrors `tree_models::config::TreeRuntimeOverrides`, which does the same job
/// for the tree models. Defaults reproduce a fresh `Settings`, so a build with
/// the CUDA kernels linked but no config change behaves exactly like a build
/// without them.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelGpuRuntimeOverrides {
    pub rl_device: String,
    pub statistical_device: String,
    pub neuro_evolution_device: String,
}

impl Default for ModelGpuRuntimeOverrides {
    fn default() -> Self {
        Self {
            rl_device: "cpu".to_string(),
            statistical_device: "auto".to_string(),
            neuro_evolution_device: "auto".to_string(),
        }
    }
}

impl ModelGpuRuntimeOverrides {
    pub fn from_settings(settings: &Settings) -> Self {
        let configured = &settings.models.gpu_runtime;
        let or_default = |value: &str, fallback: &str| {
            if value.trim().is_empty() {
                fallback.to_string()
            } else {
                value.trim().to_ascii_lowercase()
            }
        };
        Self {
            rl_device: or_default(&configured.rl_device, "cpu"),
            statistical_device: or_default(&configured.statistical_device, "auto"),
            neuro_evolution_device: or_default(&configured.neuro_evolution_device, "auto"),
        }
    }
}

static MODEL_GPU_RUNTIME: OnceLock<ModelGpuRuntimeOverrides> = OnceLock::new();

/// Install the non-tree GPU device policy from `Settings`. First install wins,
/// exactly like the tree-model equivalent it sits beside.
pub fn install_model_gpu_runtime_from_settings(settings: &Settings) {
    let _ = MODEL_GPU_RUNTIME.set(ModelGpuRuntimeOverrides::from_settings(settings));
}

/// Current policy (defaults when never installed — unit tests, and any
/// front-end that has not called the installer).
pub fn current_model_gpu_runtime() -> &'static ModelGpuRuntimeOverrides {
    MODEL_GPU_RUNTIME.get_or_init(ModelGpuRuntimeOverrides::default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_gpu_runtime_from_default_settings_matches_default() {
        let settings = Settings::default();
        assert_eq!(
            ModelGpuRuntimeOverrides::from_settings(&settings),
            ModelGpuRuntimeOverrides::default()
        );
    }

    #[test]
    fn rl_default_is_cpu_so_linking_cuda_changes_nothing() {
        // The whole point of the `rl_device` knob: wiring
        // `reinforcement-learning-cuda` into `gpu-cuda` must not move DQN
        // training onto the card until the operator asks.
        assert_eq!(ModelGpuRuntimeOverrides::default().rl_device, "cpu");
    }

    #[test]
    fn lightgbm_declares_no_gpu_path() {
        assert!(declared_gpu_path("lightgbm", ModelFamily::Tree).is_none());
    }

    #[test]
    fn meta_models_share_the_xgboost_cuda_path() {
        // They train through the XGBoost expert, so a CUDA-capable libxgboost
        // accelerates all six at once — the registry used to report `false`
        // for four of them purely because their family is `Meta`.
        for model in [
            "meta_blender",
            "probability_calibrator",
            "conformal_gate",
            "meta_stack",
        ] {
            assert_eq!(
                declared_gpu_path(model, ModelFamily::Meta),
                Some(XGBOOST_CUDA),
                "{model} trains through the XGBoost expert"
            );
        }
    }

    #[test]
    fn deep_and_exit_families_report_the_active_burn_backend() {
        let spec =
            declared_gpu_path("tabnet", ModelFamily::Deep).expect("deep models are Burn-backed");
        let active = crate::burn_models::active_burn_backend_name();
        if active != "ndarray_cpu" {
            assert_eq!(spec.accelerator.as_str(), active);
        }
    }

    #[test]
    fn every_declared_feature_is_convertible_to_a_cfg() {
        // `feature_compiled` returning `false` for an unknown name is a silent
        // "no GPU" — the exact failure mode this module exists to remove. Every
        // feature the table names must be a case in that match.
        const KNOWN: &[&str] = &[
            "xgboost-gpu",
            "catboost",
            "reinforcement-learning-cuda",
            "neuro-evolution-gpu",
            "statistical-gpu",
            "burn-cuda-backend",
            "burn-wgpu-backend",
        ];
        for capability in gpu_capability_matrix() {
            let Some(feature) = capability.feature else {
                continue;
            };
            assert!(
                KNOWN.contains(&feature),
                "{} names feature `{feature}`, which `feature_compiled` does not handle",
                capability.model
            );
        }
    }

    #[test]
    fn compiled_implies_declared() {
        for capability in gpu_capability_matrix() {
            if capability.compiled {
                assert!(capability.declared, "{} compiled but not declared", capability.model);
            }
            if capability.usable() {
                assert!(capability.compiled, "{} usable but not compiled", capability.model);
            }
        }
    }
}
