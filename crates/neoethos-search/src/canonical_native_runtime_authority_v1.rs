use std::sync::{Mutex, OnceLock};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::canonical_native_discovery_request_v1::CanonicalNativeDiscoveryRequestErrorV1;
use crate::eval::BacktestRuntimeOverrides;
use crate::genetic::{
    GeneStopBoundsOverrides, GeneticSearchRuntimeOverrides, SeenSignatureMemoryRuntimeOverrides,
    SmcSearchConfig, StrategyEvaluationRuntimeOverrides,
};

const SETTINGS_DOMAIN_V1: &[u8] = b"neoethos.canonical-native.startup-settings.v1\0";
const SNAPSHOT_DOMAIN_V1: &[u8] = b"neoethos.canonical-native.runtime-snapshot.v1\0";
const INSTALL_DOMAIN_V1: &[u8] = b"neoethos.canonical-native.runtime-install-receipt.v1\0";
const AUTHORITY_DOMAIN_V1: &[u8] = b"neoethos.canonical-native.gen0-runtime-authority.v1\0";

#[derive(Clone, PartialEq, Serialize)]
struct RuntimeSnapshotV1 {
    evaluation_backend: String,
    indicator_compute_policy: String,
    data_normalize_features: bool,
    feature_cube_mode: neoethos_core::config::FeatureCubeMode,
    genetic: GeneticSearchRuntimeOverrides,
    strategy_evaluation: StrategyEvaluationRuntimeOverrides,
    backtest: BacktestRuntimeOverrides,
    smc: SmcSearchConfig,
    stop_target: crate::StopTargetRuntimeOverrides,
    adaptive_stops_enabled: bool,
    adaptive_stops_rr: f64,
    gene_stop_bounds: GeneStopBoundsOverrides,
    seen_memory: SeenSignatureMemoryRuntimeOverrides,
}

impl RuntimeSnapshotV1 {
    const fn expected_adaptive_invariants_v1() -> (bool, f64) {
        (true, 2.0)
    }

    fn expected(
        settings: &neoethos_core::Settings,
    ) -> Result<Self, CanonicalNativeDiscoveryRequestErrorV1> {
        let backend = crate::EvaluationBackend::from_settings_and_process_env(settings)
            .map_err(|error| runtime_error(&format!("resolve evaluation backend: {error}")))?;
        let (adaptive_stops_enabled, adaptive_stops_rr) = Self::expected_adaptive_invariants_v1();
        Ok(Self {
            evaluation_backend: backend_tag(backend),
            indicator_compute_policy: if backend.device == crate::DevicePreference::Gpu {
                "gpu_only".to_owned()
            } else {
                "auto".to_owned()
            },
            data_normalize_features: settings.models.data_runtime.normalize_features,
            feature_cube_mode: settings.models.data_runtime.feature_cube_mode,
            genetic: GeneticSearchRuntimeOverrides::from_settings(settings),
            strategy_evaluation: StrategyEvaluationRuntimeOverrides::from_settings(settings),
            backtest: BacktestRuntimeOverrides::from_settings(settings),
            smc: SmcSearchConfig::from_settings(settings),
            stop_target: crate::StopTargetRuntimeOverrides::from_settings(settings),
            adaptive_stops_enabled,
            adaptive_stops_rr,
            gene_stop_bounds: GeneStopBoundsOverrides::from_settings(settings),
            // This constructor may resolve the configured zero sentinel from live RAM.
            // It is called exactly once under INSTALL_LOCK; later validation compares
            // the installed effective value rather than probing RAM again.
            seen_memory: SeenSignatureMemoryRuntimeOverrides::from_settings(settings),
        })
    }

    fn current() -> Self {
        Self {
            evaluation_backend: backend_tag(crate::current_evaluation_backend()),
            indicator_compute_policy: indicator_policy_tag(
                neoethos_data::core::hpc_ta::resolved_indicator_compute_policy(),
            )
            .to_owned(),
            data_normalize_features: neoethos_data::current_data_runtime_overrides()
                .normalize_features,
            feature_cube_mode: neoethos_data::current_feature_cube_policy(),
            genetic: crate::genetic::current_genetic_search_runtime_overrides(),
            strategy_evaluation: crate::genetic::current_strategy_evaluation_runtime_overrides(),
            backtest: crate::eval::current_backtest_runtime_overrides(),
            smc: SmcSearchConfig::current(),
            stop_target: crate::stop_target::current_stop_target_runtime_overrides(),
            adaptive_stops_enabled: crate::stop_target::adaptive_stops_enabled(),
            adaptive_stops_rr: crate::stop_target::adaptive_stops_rr(),
            gene_stop_bounds: crate::genetic::current_gene_stop_bounds_overrides(),
            seen_memory: crate::genetic::current_seen_signature_memory_runtime_overrides(),
        }
    }

    fn mismatch_class(&self, actual: &Self) -> Option<&'static str> {
        if self.evaluation_backend != actual.evaluation_backend {
            Some("evaluation_backend")
        } else if self.indicator_compute_policy != actual.indicator_compute_policy {
            Some("indicator_compute_policy")
        } else if self.data_normalize_features != actual.data_normalize_features {
            Some("data_normalization")
        } else if self.feature_cube_mode != actual.feature_cube_mode {
            Some("data_feature_cube_policy")
        } else if self.genetic != actual.genetic {
            Some("genetic")
        } else if self.strategy_evaluation != actual.strategy_evaluation {
            Some("strategy_evaluation")
        } else if self.backtest != actual.backtest {
            Some("backtest")
        } else if self.smc != actual.smc {
            Some("smc")
        } else if self.stop_target != actual.stop_target {
            Some("stop_target")
        } else if self.adaptive_stops_enabled != actual.adaptive_stops_enabled
            || self.adaptive_stops_rr.to_bits() != actual.adaptive_stops_rr.to_bits()
        {
            Some("adaptive_stop_policy")
        } else if self.gene_stop_bounds != actual.gene_stop_bounds {
            Some("gene_stop_bounds")
        } else if self.seen_memory != actual.seen_memory {
            Some("seen_signature_memory")
        } else {
            None
        }
    }

    fn identity_sha256(&self) -> Result<String, CanonicalNativeDiscoveryRequestErrorV1> {
        let bytes = canonical_json_bytes(self)?;
        Ok(domain_hash(SNAPSHOT_DOMAIN_V1, &[&bytes]))
    }
}

#[derive(Clone)]
pub struct CanonicalNativeRuntimeInstallReceiptV1 {
    startup_settings_sha256: String,
    runtime_snapshot_sha256: String,
    identity_sha256: String,
}

impl CanonicalNativeRuntimeInstallReceiptV1 {
    pub fn startup_settings_sha256(&self) -> &str {
        &self.startup_settings_sha256
    }

    pub fn runtime_snapshot_sha256(&self) -> &str {
        &self.runtime_snapshot_sha256
    }

    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    pub(crate) fn validate_current(
        &self,
        settings: &neoethos_core::Settings,
    ) -> Result<(), CanonicalNativeDiscoveryRequestErrorV1> {
        let installed = INSTALLED
            .get()
            .ok_or_else(|| runtime_error("install receipt absent"))?;
        let settings_sha256 = settings_identity_sha256(settings)?;
        if settings_sha256 != self.startup_settings_sha256
            || self.startup_settings_sha256 != installed.receipt.startup_settings_sha256
            || self.runtime_snapshot_sha256 != installed.receipt.runtime_snapshot_sha256
            || self.identity_sha256 != installed.receipt.identity_sha256
        {
            return Err(runtime_error(
                "startup Settings/install receipt identity mismatch",
            ));
        }
        validate_snapshot(&installed.snapshot, &RuntimeSnapshotV1::current())?;
        Ok(())
    }
}

pub struct CanonicalNativeGenerationZeroRuntimeAuthorityV1 {
    startup_settings_sha256: String,
    runtime_install_receipt_sha256: String,
    identity_sha256: String,
}

impl CanonicalNativeGenerationZeroRuntimeAuthorityV1 {
    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    pub fn runtime_install_receipt_sha256(&self) -> &str {
        &self.runtime_install_receipt_sha256
    }

    pub(crate) fn validate_current(
        &self,
        settings: &neoethos_core::Settings,
        receipt: &CanonicalNativeRuntimeInstallReceiptV1,
    ) -> Result<(), CanonicalNativeDiscoveryRequestErrorV1> {
        receipt.validate_current(settings)?;
        if self.startup_settings_sha256 != receipt.startup_settings_sha256
            || self.runtime_install_receipt_sha256 != receipt.identity_sha256
            || self.identity_sha256
                != generation_zero_authority_identity(
                    receipt,
                    self.startup_settings_sha256.as_str(),
                )
        {
            return Err(runtime_error(
                "Generation-zero runtime authority identity mismatch",
            ));
        }
        require_migration_disabled()
    }
}

#[derive(Clone)]
struct InstalledRuntimeAuthorityV1 {
    receipt: CanonicalNativeRuntimeInstallReceiptV1,
    snapshot: RuntimeSnapshotV1,
}

static INSTALL_LOCK: Mutex<()> = Mutex::new(());
static INSTALLED: OnceLock<InstalledRuntimeAuthorityV1> = OnceLock::new();

pub fn install_and_seal_canonical_native_runtime_authority_v1(
    settings: &neoethos_core::Settings,
) -> Result<CanonicalNativeRuntimeInstallReceiptV1, CanonicalNativeDiscoveryRequestErrorV1> {
    let _guard = INSTALL_LOCK
        .lock()
        .map_err(|_| runtime_error("runtime installer lock poisoned"))?;
    let settings_sha256 = settings_identity_sha256(settings)?;
    if let Some(installed) = INSTALLED.get() {
        if installed.receipt.startup_settings_sha256 != settings_sha256 {
            return Err(runtime_error(
                "conflicting startup Settings were already installed",
            ));
        }
        installed.receipt.validate_current(settings)?;
        require_migration_disabled()?;
        return Ok(installed.receipt.clone());
    }

    let expected = RuntimeSnapshotV1::expected(settings)?;
    let mut install_settings = settings.clone();
    install_settings.models.seen_signature_runtime.max_entries = expected.seen_memory.max_entries;
    invoke_runtime_installers(&install_settings)?;
    let actual = RuntimeSnapshotV1::current();
    validate_snapshot(&expected, &actual)?;
    require_migration_disabled()?;
    let runtime_snapshot_sha256 = actual.identity_sha256()?;
    let identity_sha256 = domain_hash(
        INSTALL_DOMAIN_V1,
        &[
            settings_sha256.as_bytes(),
            runtime_snapshot_sha256.as_bytes(),
        ],
    );
    let receipt = CanonicalNativeRuntimeInstallReceiptV1 {
        startup_settings_sha256: settings_sha256,
        runtime_snapshot_sha256,
        identity_sha256,
    };
    INSTALLED
        .set(InstalledRuntimeAuthorityV1 {
            receipt: receipt.clone(),
            snapshot: actual,
        })
        .map_err(|_| runtime_error("runtime install receipt raced with another installer"))?;
    Ok(receipt)
}

#[cfg(all(feature = "gpu-cuda", target_os = "linux"))]
pub(crate) fn seal_generation_zero_runtime_authority_v1(
    settings: &neoethos_core::Settings,
    receipt: &CanonicalNativeRuntimeInstallReceiptV1,
) -> Result<CanonicalNativeGenerationZeroRuntimeAuthorityV1, CanonicalNativeDiscoveryRequestErrorV1>
{
    receipt.validate_current(settings)?;
    require_migration_disabled()?;
    Ok(CanonicalNativeGenerationZeroRuntimeAuthorityV1 {
        startup_settings_sha256: receipt.startup_settings_sha256.clone(),
        runtime_install_receipt_sha256: receipt.identity_sha256.clone(),
        identity_sha256: generation_zero_authority_identity(
            receipt,
            receipt.startup_settings_sha256.as_str(),
        ),
    })
}

fn invoke_runtime_installers(
    settings: &neoethos_core::Settings,
) -> Result<(), CanonicalNativeDiscoveryRequestErrorV1> {
    neoethos_data::install_data_runtime_overrides(settings.models.data_runtime.normalize_features);
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::install_search_runtime_overrides_from_settings(settings);
    }))
    .map_err(|_| runtime_error("Search runtime installer rejected the startup Settings"))?;
    Ok(())
}

fn validate_snapshot(
    expected: &RuntimeSnapshotV1,
    actual: &RuntimeSnapshotV1,
) -> Result<(), CanonicalNativeDiscoveryRequestErrorV1> {
    if let Some(class) = expected.mismatch_class(actual) {
        return Err(runtime_error(&format!(
            "installed {class} snapshot mismatches startup Settings"
        )));
    }
    Ok(())
}

fn require_migration_disabled() -> Result<(), CanonicalNativeDiscoveryRequestErrorV1> {
    if crate::genetic::migration_enabled() {
        Err(CanonicalNativeDiscoveryRequestErrorV1::MigrationEnabled)
    } else {
        Ok(())
    }
}

fn settings_identity_sha256(
    settings: &neoethos_core::Settings,
) -> Result<String, CanonicalNativeDiscoveryRequestErrorV1> {
    let bytes = canonical_json_bytes(settings)?;
    Ok(domain_hash(SETTINGS_DOMAIN_V1, &[&bytes]))
}

fn generation_zero_authority_identity(
    receipt: &CanonicalNativeRuntimeInstallReceiptV1,
    settings_sha256: &str,
) -> String {
    domain_hash(
        AUTHORITY_DOMAIN_V1,
        &[
            settings_sha256.as_bytes(),
            receipt.identity_sha256.as_bytes(),
            b"migration=false",
        ],
    )
}

fn canonical_json_bytes(
    value: &impl Serialize,
) -> Result<Vec<u8>, CanonicalNativeDiscoveryRequestErrorV1> {
    let value = serde_json::to_value(value)
        .map_err(|error| runtime_error(&format!("serialize authority input: {error}")))?;
    serde_json::to_vec(&sort_json(value))
        .map_err(|error| runtime_error(&format!("encode canonical authority input: {error}")))
}

fn sort_json(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(sort_json).collect())
        }
        serde_json::Value::Object(values) => {
            let mut entries = values.into_iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            let mut sorted = serde_json::Map::new();
            for (key, value) in entries {
                sorted.insert(key, sort_json(value));
            }
            serde_json::Value::Object(sorted)
        }
        scalar => scalar,
    }
}

fn domain_hash(domain: &[u8], fields: &[&[u8]]) -> String {
    let mut digest = Sha256::new();
    digest.update(domain);
    for field in fields {
        digest.update((field.len() as u64).to_le_bytes());
        digest.update(field);
    }
    format!("{:x}", digest.finalize())
}

fn runtime_error(detail: &str) -> CanonicalNativeDiscoveryRequestErrorV1 {
    CanonicalNativeDiscoveryRequestErrorV1::RuntimeAuthority(detail.to_owned())
}

fn backend_tag(backend: crate::EvaluationBackend) -> String {
    let device = match backend.device {
        crate::DevicePreference::Cpu => "cpu",
        crate::DevicePreference::Auto => "auto",
        crate::DevicePreference::Gpu => "gpu",
    };
    let accelerator = match backend.accelerator_hint {
        crate::AcceleratorHint::Any => "any",
        crate::AcceleratorHint::Cuda => "cuda",
        crate::AcceleratorHint::Wgpu => "wgpu",
        crate::AcceleratorHint::Vulkan => "vulkan",
        crate::AcceleratorHint::Rocm => "rocm",
        crate::AcceleratorHint::Metal => "metal",
        crate::AcceleratorHint::Dx12 => "dx12",
    };
    format!("{device}:forbid_cpu:{accelerator}")
}

fn indicator_policy_tag(
    policy: neoethos_data::core::hpc_ta::IndicatorComputePolicy,
) -> &'static str {
    match policy {
        neoethos_data::core::hpc_ta::IndicatorComputePolicy::Auto => "auto",
        neoethos_data::core::hpc_ta::IndicatorComputePolicy::CpuOnly => "cpu_only",
        neoethos_data::core::hpc_ta::IndicatorComputePolicy::GpuOnly => "gpu_only",
    }
}

#[cfg(test)]
mod adaptive_invariant_tests {
    use super::*;

    #[test]
    fn expected_adaptive_invariants_are_literal_and_current_drift_cannot_self_validate() {
        let (enabled, rr) = RuntimeSnapshotV1::expected_adaptive_invariants_v1();
        assert!(enabled);
        assert_eq!(rr.to_bits(), 2.0_f64.to_bits());

        let mut settings = neoethos_core::Settings::default();
        settings.models.seen_signature_runtime.max_entries = 3_000_000;
        let expected = RuntimeSnapshotV1::expected(&settings).unwrap();
        assert!(expected.adaptive_stops_enabled);
        assert_eq!(expected.adaptive_stops_rr.to_bits(), 2.0_f64.to_bits());

        let mut changed_enabled = expected.clone();
        changed_enabled.adaptive_stops_enabled = false;
        assert!(validate_snapshot(&expected, &changed_enabled).is_err());

        let mut changed_rr = expected.clone();
        changed_rr.adaptive_stops_rr = 2.5;
        assert!(validate_snapshot(&expected, &changed_rr).is_err());
    }
}
