use std::sync::OnceLock;

use anyhow::{Result, bail};
use neoethos_dataset_contracts::CanonicalTimeframe;
use neoethos_feature_contracts::{
    FeatureContractError, RelevantDependencySetV1, RelevantDependencyV1, SemanticSourceEntryV1,
    SemanticSourceKindV1, SemanticSourceManifestV1, SemanticSourceSetV1,
};
use serde::{Deserialize, Serialize};

use super::regime_detection::{
    REGIME_FEATURE_NAMES_V3, REGIME_RETIRED_V2_FEATURE_NAMES, REGIME_V2_ARTIFACT_MIGRATION_POLICY,
};
use crate::core::all_indicators::ALL_INDICATORS;

macro_rules! embedded_source {
    ($canonical_path:literal, $manifest_relative_path:literal) => {
        SemanticSourceEntryV1::from_bytes(
            $canonical_path,
            SemanticSourceKindV1::Utf8Text,
            include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), $manifest_relative_path)),
        )?
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureSource {
    SmartMoneyConcept,
    ClassicTechnicalAnalysis,
    Quantitative,
    Session,
    Regime,
    Footprint,
}

/// Exhaustive identities for the scalar families that feed the production
/// `FeatureFrame`.
///
/// This is intentionally distinct from [`FeatureSource`]: the latter labels a
/// column, while this enum is the compiler-checked call graph used to invoke
/// every top-level value producer. Adding a producer therefore requires an
/// exhaustive dispatch change as well as one manifest row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProductionFeatureProducerId {
    SmartMoneyConcept,
    ClassicVectorTa,
    Quantitative,
    Session,
    Regime,
    Footprint,
}

/// Stable production emission order. Model artifacts depend on this order, so
/// it must never be inferred from completion order of parallel tasks.
pub const PRODUCTION_FEATURE_PRODUCER_ORDER: [ProductionFeatureProducerId; 6] = [
    ProductionFeatureProducerId::SmartMoneyConcept,
    ProductionFeatureProducerId::ClassicVectorTa,
    ProductionFeatureProducerId::Quantitative,
    ProductionFeatureProducerId::Session,
    ProductionFeatureProducerId::Regime,
    ProductionFeatureProducerId::Footprint,
];

pub const CLASSIC_VECTOR_TA_SEMANTIC_VERSION_V8: u32 = 8;
pub const CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V8: &str = "evwma/rolling-volume-sum/close/f64/v1;evwma/rolling-length-key/length/default-30/f64/v1;evwma/fixed-n/close/f64/v1";
pub const CLASSIC_VECTOR_TA_V7_ARTIFACT_MIGRATION_POLICY: &str = "refuse semantic-v7 and unversioned ClassicVectorTa artifacts; regenerate them under semantic-v8";
pub const CLASSIC_VECTOR_TA_MANIFEST_COMPLETENESS_DEBT_V8: &str = "v8 binds this EVWMA slice; the broader 341-file VectorTA source closure remains explicit manifest-completeness debt";
pub const CLASSIC_VECTOR_TA_SEMANTIC_VERSION_V9: u32 = 9;
pub const CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V9: &str = "classic-composite-v9;evwma/rolling-volume-sum/close/f64/v1;evwma/rolling-length-key/length/default-30/f64/v1;evwma/fixed-n/close/f64/v1;cci-cycle/creator-pine-v3/local-current-resolution/f64/v1;cci-cycle/floor-half;cci-cycle/sma-seeded-ema-rma;cci-cycle/startup-flat-zero-carry;cci-cycle/factor-zero-freeze;cci-cycle/finite-segment-reset-v1;cci-cycle/creator-source-sha256/d00a0186f28989a34eb1da24eb9fae9a8906736afe413e2492ded9dc4b2a9c9f;frama-f64-v3-finite-hlc-segment-reset-even-window-stable-fma-v2;frama/finite-hlc-segment-reset/v3;frama/evenized-window-seed/v1;frama/stable-affine-fma/v2;frama/host-source-sha256/6D2380A30ECA86E77DDD7B461F0A9D961450C82CDD52B19653F148852A3FF7FE;frama/cuda-source-sha256/AACB6789BEE22C5FDE46C1966EA956E8E46209B42720D4DD900A5CD94AB1AD02;fwma-f64-v2-p254-u192-fib-pow2-dd-fma-window-recovery;fwma/host-source-sha256/D5F2E5D59128C02858E0DDB236A9EAB6425883A3978A67A7221A3FCEF42F6AC3;fwma/cuda-source-sha256/C7716141216AC0EE144430092F570606821415D78B75DDA756F91A64415A24EE;fisher-f64-v2-openlibm-e-log-midpoint-finite-segment-reset-oN-deque-bounded-faithful-p1024;fisher/host-source-sha256/B97652FCFB1BD711DE5B33F90564AA0DB02D46E9187C1134F84578B19BC724D6;fisher/cuda-source-sha256/4F548C7B1A0A10864B6FB398C26BF355C887BD282FB3B63A989D6858FCEE158A;fisher/openlibm-e-log-source-sha256/8996B789A4CBBCEF7CF7D568C1BE558CE9110900A40CA6C46FB4ED46C343CAFD;fisher/openlibm-receipt-sha256/7F4F37742F7EE8C8A79A5F8D244D1EE41423197A2842C06BF2E62FC165FBE5B9;half-causal-estimator-f64-v2-neoethos-canonical-pine6-script24-utc-day-slot-session-proxy-cached-future-windows-stable-f64-registry-ratio-dl;hce/registry-ratio-dl/7-d2-l7/20-d5-l20/21-d5-l21/50-d13-l50/100-d25-l100/200-d50-l200/v2;hce/data-period-zero-unbounded-online-welford/v1;hce/finite-frame-effective-d-checked-fallible-allocation/v1;hce/public-retained-budget-64mib/v1;hce/creator-source-sha256/4B7FD8AEC6B333A4ECE967D7CFA6D957357CE436CB098E96EB1EB8A1480A8080;hce/host-source-sha256/3632A8F08DF17BDE65A06C17068A6FE79BDE8F11E3A054688A956F32C84FCC6B;hce/stable-math-source-sha256/F1CE1AE5272EAED95EFBCBC87034C4CF1B72BC25AF0A6504FA0BFC1D29E4F528;hce/cuda-source-sha256/B9B87151A498EAE775C75CF5669799A59C38130A0B729846B09139DD448E0796;hce/generic-fail-closed-wrapper-sha256/F93F18CEC0912BBE15B481BD0575AF7DB7225E368FA4DF7B658A45E868245B64;hce/raw-creator-receipt-sha256/D371BB32D723C17997EA210E230597FFFD1AD876C7A537DA3DFCD272EC4582AD;hce/receipt-toml-sha256/18D24B85AA160B571BDE2BB6D023046C7403EE309F9C841694C51A1F8B90650F;eacp-f64-v1-vector-ta-pearson-dft-sq2-decaying-max-cog50-biased-ema-finite-segment-reset;eacp/strict-cuda-exact-cooperative-cta/fmad-off/v1;eacp/cpu-source-sha256/0108F73AC2DE644855A5E93999D211C2634C50A18B182D4337672D808A7D06EE;eacp/cuda-source-sha256/12224A4C7F1B10612491E5BBD4608011E7D32F9737B5DF3A0259A3AB3E9688B0;eacp/strict-wrapper-source-sha256/4E55CA5A5203013D255BA49F5A0C5B6FE3F12682CAB16234AAB62063C3582C36";
pub const CLASSIC_VECTOR_TA_V8_ARTIFACT_MIGRATION_POLICY: &str = "refuse semantic-v8 and older or unversioned ClassicVectorTa artifacts; identical CCI Cycle column names changed values, so regenerate them under semantic-v9";
pub const CLASSIC_VECTOR_TA_MANIFEST_COMPLETENESS_DEBT_V9: &str = "v9 is the open pre-release composite authority for audited Classic repairs; it binds EVWMA-v8, CCI-Cycle-v9, FRAMA-v3, FWMA-v2, Fisher-v2, and HCE-v2 now and may append other audited repairs before the v9 release manifest freezes; the broader VectorTA source closure remains explicit manifest-completeness debt";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductionFeatureProducerManifestRowV1 {
    producer: ProductionFeatureProducerId,
    source: FeatureSource,
    semantic_version: u32,
    semantic_source_set: SemanticSourceSetV1,
}

impl ProductionFeatureProducerManifestRowV1 {
    pub const fn producer(&self) -> ProductionFeatureProducerId {
        self.producer
    }

    pub const fn source(&self) -> FeatureSource {
        self.source
    }

    pub const fn semantic_version(&self) -> u32 {
        self.semantic_version
    }

    pub fn semantic_sources(&self) -> &SemanticSourceManifestV1 {
        self.semantic_source_set.sources()
    }

    pub fn relevant_dependencies(&self) -> &RelevantDependencySetV1 {
        self.semantic_source_set.dependencies()
    }

    pub const fn semantic_source_set(&self) -> &SemanticSourceSetV1 {
        &self.semantic_source_set
    }
}

static PRODUCTION_FEATURE_PRODUCER_MANIFEST_V1: OnceLock<
    std::result::Result<Vec<ProductionFeatureProducerManifestRowV1>, FeatureContractError>,
> = OnceLock::new();

/// Returns the embedded, canonical source/dependency closure for every
/// production value producer. Source bytes are compiled into the binary, so
/// packaged runtimes never depend on `.git` or on mutable checkout files.
pub fn production_feature_producer_manifest_v1()
-> std::result::Result<&'static [ProductionFeatureProducerManifestRowV1], FeatureContractError> {
    match PRODUCTION_FEATURE_PRODUCER_MANIFEST_V1.get_or_init(build_production_manifest_v1) {
        Ok(rows) => Ok(rows),
        Err(error) => Err(error.clone()),
    }
}

static QUANTITATIVE_FEATURE_PRODUCER_MANIFEST_V3: OnceLock<
    std::result::Result<ProductionFeatureProducerManifestRowV1, FeatureContractError>,
> = OnceLock::new();

/// Distinct semantic authority for the typed Quant-v3 CPU reference used by
/// the resident GPU exact-parity route. The ordinary production manifest stays
/// on Quant-v2; selecting this row is an explicit whole-feature math decision.
pub fn quantitative_feature_producer_manifest_v3()
-> std::result::Result<&'static ProductionFeatureProducerManifestRowV1, FeatureContractError> {
    match QUANTITATIVE_FEATURE_PRODUCER_MANIFEST_V3
        .get_or_init(build_quantitative_feature_producer_manifest_v3)
    {
        Ok(row) => Ok(row),
        Err(error) => Err(error.clone()),
    }
}

fn build_quantitative_feature_producer_manifest_v3()
-> std::result::Result<ProductionFeatureProducerManifestRowV1, FeatureContractError> {
    producer_row(
        ProductionFeatureProducerId::Quantitative,
        FeatureSource::Quantitative,
        3,
        vec![
            embedded_source!(
                "crates/neoethos-data/src/core/quant_features.rs",
                "/src/core/quant_features.rs"
            ),
            embedded_source!(
                "crates/neoethos-data/src/core/quant_exact_math_v3.rs",
                "/src/core/quant_exact_math_v3.rs"
            ),
            embedded_source!(
                "crates/neoethos-data/src/core/gpu_resident_temporal_grid_v1.rs",
                "/src/core/gpu_resident_temporal_grid_v1.rs"
            ),
            embedded_source!(
                "crates/neoethos-data/src/core/gpu_resident_quant_v3.rs",
                "/src/core/gpu_resident_quant_v3.rs"
            ),
            embedded_source!(
                "crates/neoethos-data/src/core/timestamps.rs",
                "/src/core/timestamps.rs"
            ),
            embedded_source!(
                "crates/neoethos-dataset-contracts/src/temporal.rs",
                "/../neoethos-dataset-contracts/src/temporal.rs"
            ),
            embedded_source!("crates/neoethos-data/src/lib.rs", "/src/lib.rs"),
        ],
        Vec::new(),
    )
}

fn build_production_manifest_v1()
-> std::result::Result<Vec<ProductionFeatureProducerManifestRowV1>, FeatureContractError> {
    const CHRONO_CHECKSUM: [u8; 32] = [
        0x1a, 0xa7, 0x9e, 0x62, 0xe7, 0x69, 0x7b, 0x8e, 0x29, 0xb5, 0x13, 0xa6, 0x8a, 0xba, 0xcf,
        0x48, 0x5a, 0xdc, 0xd1, 0xfe, 0x82, 0x84, 0xa4, 0x31, 0x6c, 0x5a, 0xe8, 0x68, 0xe6, 0x63,
        0x33, 0x27,
    ];

    let chrono_dependency = || {
        RelevantDependencyV1::registry(
            "chrono",
            "0.4.45",
            "https://github.com/rust-lang/crates.io-index",
            CHRONO_CHECKSUM,
            vec!["default".to_owned()],
        )
    };
    let vector_ta_sources = source_manifest(vec![
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs",
            "/../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_single.rs",
            "/../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_single.rs"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/mod.rs",
            "/../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/mod.rs"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/src/indicators/registry.rs",
            "/../../vendor/vector-ta-0.2.9-patched/src/indicators/registry.rs"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/registry.rs",
            "/../../vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/registry.rs"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/elastic_volume_weighted_moving_average.rs",
            "/../../vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/elastic_volume_weighted_moving_average.rs"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cuda_f64.rs",
            "/../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cuda_f64.rs"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/kernels/cuda/moving_averages/elastic_volume_weighted_moving_average_kernel.cu",
            "/../../vendor/vector-ta-0.2.9-patched/kernels/cuda/moving_averages/elastic_volume_weighted_moving_average_kernel.cu"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/frama.rs",
            "/../../vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/frama.rs"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/kernels/cuda/moving_averages/frama_kernel.cu",
            "/../../vendor/vector-ta-0.2.9-patched/kernels/cuda/moving_averages/frama_kernel.cu"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/fwma.rs",
            "/../../vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/fwma.rs"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/kernels/cuda/moving_averages/fwma_kernel.cu",
            "/../../vendor/vector-ta-0.2.9-patched/kernels/cuda/moving_averages/fwma_kernel.cu"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/src/indicators/fisher.rs",
            "/../../vendor/vector-ta-0.2.9-patched/src/indicators/fisher.rs"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/kernels/cuda/oscillators/fisher_kernel.cu",
            "/../../vendor/vector-ta-0.2.9-patched/kernels/cuda/oscillators/fisher_kernel.cu"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/tests/fixtures/openlibm/e_log-82e90aef0657289192efe77be89791c07dea0775.c",
            "/../../vendor/vector-ta-0.2.9-patched/tests/fixtures/openlibm/e_log-82e90aef0657289192efe77be89791c07dea0775.c"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/tests/fixtures/openlibm/e_log-82e90aef0657289192efe77be89791c07dea0775.receipt.txt",
            "/../../vendor/vector-ta-0.2.9-patched/tests/fixtures/openlibm/e_log-82e90aef0657289192efe77be89791c07dea0775.receipt.txt"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/src/indicators/half_causal_estimator.rs",
            "/../../vendor/vector-ta-0.2.9-patched/src/indicators/half_causal_estimator.rs"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/src/indicators/half_causal_estimator_stable_math.rs",
            "/../../vendor/vector-ta-0.2.9-patched/src/indicators/half_causal_estimator_stable_math.rs"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/src/cuda/half_causal_estimator_wrapper.rs",
            "/../../vendor/vector-ta-0.2.9-patched/src/cuda/half_causal_estimator_wrapper.rs"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/kernels/cuda/half_causal_estimator_kernel.cu",
            "/../../vendor/vector-ta-0.2.9-patched/kernels/cuda/half_causal_estimator_kernel.cu"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/src/indicators/ehlers_autocorrelation_periodogram.rs",
            "/../../vendor/vector-ta-0.2.9-patched/src/indicators/ehlers_autocorrelation_periodogram.rs"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/kernels/cuda/ehlers_autocorrelation_periodogram_kernel.cu",
            "/../../vendor/vector-ta-0.2.9-patched/kernels/cuda/ehlers_autocorrelation_periodogram_kernel.cu"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/audit_receipts/half_causal_estimator/tradingview_pine_facade_script24_raw.json",
            "/../../vendor/vector-ta-0.2.9-patched/audit_receipts/half_causal_estimator/tradingview_pine_facade_script24_raw.json"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/audit_receipts/half_causal_estimator/script24_receipt.toml",
            "/../../vendor/vector-ta-0.2.9-patched/audit_receipts/half_causal_estimator/script24_receipt.toml"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/src/indicators/cci_cycle.rs",
            "/../../vendor/vector-ta-0.2.9-patched/src/indicators/cci_cycle.rs"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs",
            "/../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/src/cuda/cci_cycle_wrapper.rs",
            "/../../vendor/vector-ta-0.2.9-patched/src/cuda/cci_cycle_wrapper.rs"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/kernels/cuda/oscillators/cci_cycle_kernel.cu",
            "/../../vendor/vector-ta-0.2.9-patched/kernels/cuda/oscillators/cci_cycle_kernel.cu"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/build.rs",
            "/../../vendor/vector-ta-0.2.9-patched/build.rs"
        ),
        embedded_source!(
            "vendor/vector-ta-0.2.9-patched/src/indicators/pattern_recognition.rs",
            "/../../vendor/vector-ta-0.2.9-patched/src/indicators/pattern_recognition.rs"
        ),
    ])?;
    #[cfg(feature = "gpu-cuda")]
    let vector_ta_features = vec![
        "cuda-build-native".to_owned(),
        "default".to_owned(),
        "nightly-avx".to_owned(),
    ];
    #[cfg(not(feature = "gpu-cuda"))]
    let vector_ta_features = vec!["default".to_owned(), "nightly-avx".to_owned()];
    let vector_ta_dependency = RelevantDependencyV1::repository_path(
        "vector-ta",
        "0.2.9",
        "vendor/vector-ta-0.2.9-patched",
        *vector_ta_sources.identity().as_bytes(),
        vector_ta_features,
    )?;

    Ok(vec![
        producer_row(
            ProductionFeatureProducerId::SmartMoneyConcept,
            FeatureSource::SmartMoneyConcept,
            super::smc::SMC_SEMANTIC_VERSION,
            vec![
                embedded_source!("crates/neoethos-data/src/core/smc.rs", "/src/core/smc.rs"),
                embedded_source!(
                    "crates/neoethos-data/src/core/smc_log1p_exact_v1.rs",
                    "/src/core/smc_log1p_exact_v1.rs"
                ),
                embedded_source!("crates/neoethos-data/src/lib.rs", "/src/lib.rs"),
            ],
            vec![chrono_dependency()?],
        )?,
        producer_row(
            ProductionFeatureProducerId::ClassicVectorTa,
            FeatureSource::ClassicTechnicalAnalysis,
            // v9 is the composite Classic release authority. It preserves the
            // EVWMA-v8 identities and adds creator-aligned, local-resolution
            // CCI Cycle with finite-segment resets plus frozen FRAMA-v3,
            // FWMA-v2, Fisher-v2, HCE-v2, and cooperative EACP-v1 math.
            // Semantic-v8, older and unversioned feature/search artifacts must
            // not be reused.
            CLASSIC_VECTOR_TA_SEMANTIC_VERSION_V9,
            vec![
                SemanticSourceEntryV1::from_bytes(
                    "semantic-authority/classic-vector-ta-v9",
                    SemanticSourceKindV1::Utf8Text,
                    CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V9.as_bytes(),
                )?,
                embedded_source!(
                    "crates/neoethos-data/src/core/all_indicators.rs",
                    "/src/core/all_indicators.rs"
                ),
                embedded_source!(
                    "crates/neoethos-data/src/core/feature_budget.rs",
                    "/src/core/feature_budget.rs"
                ),
                embedded_source!(
                    "crates/neoethos-data/src/core/hpc_ta.rs",
                    "/src/core/hpc_ta.rs"
                ),
                embedded_source!(
                    "crates/neoethos-data/src/core/indicator_ledger.rs",
                    "/src/core/indicator_ledger.rs"
                ),
                embedded_source!(
                    "crates/neoethos-data/src/core/gpu_indicators.rs",
                    "/src/core/gpu_indicators.rs"
                ),
                embedded_source!(
                    "crates/neoethos-data/src/core/classic_cuda_plan.rs",
                    "/src/core/classic_cuda_plan.rs"
                ),
                embedded_source!(
                    "crates/neoethos-data/src/core/gpu_resident_classic_ta_v3.rs",
                    "/src/core/gpu_resident_classic_ta_v3.rs"
                ),
                embedded_source!("crates/neoethos-data/src/lib.rs", "/src/lib.rs"),
                embedded_source!(
                    "vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs",
                    "/../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs"
                ),
                embedded_source!(
                    "vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_single.rs",
                    "/../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_single.rs"
                ),
                embedded_source!(
                    "vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/mod.rs",
                    "/../../vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/mod.rs"
                ),
                embedded_source!(
                    "vendor/vector-ta-0.2.9-patched/src/indicators/registry.rs",
                    "/../../vendor/vector-ta-0.2.9-patched/src/indicators/registry.rs"
                ),
                embedded_source!(
                    "vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/registry.rs",
                    "/../../vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/registry.rs"
                ),
            ],
            vec![vector_ta_dependency],
        )?,
        producer_row(
            ProductionFeatureProducerId::Quantitative,
            FeatureSource::Quantitative,
            2,
            vec![
                embedded_source!(
                    "crates/neoethos-data/src/core/quant_features.rs",
                    "/src/core/quant_features.rs"
                ),
                embedded_source!(
                    "crates/neoethos-data/src/core/timestamps.rs",
                    "/src/core/timestamps.rs"
                ),
                embedded_source!("crates/neoethos-data/src/lib.rs", "/src/lib.rs"),
            ],
            Vec::new(),
        )?,
        producer_row(
            ProductionFeatureProducerId::Session,
            FeatureSource::Session,
            2,
            vec![
                embedded_source!(
                    "crates/neoethos-data/src/core/session_features.rs",
                    "/src/core/session_features.rs"
                ),
                embedded_source!(
                    "crates/neoethos-data/src/core/timestamps.rs",
                    "/src/core/timestamps.rs"
                ),
                embedded_source!("crates/neoethos-data/src/lib.rs", "/src/lib.rs"),
            ],
            vec![chrono_dependency()?],
        )?,
        producer_row(
            ProductionFeatureProducerId::Regime,
            FeatureSource::Regime,
            super::regime_detection::REGIME_SEMANTIC_VERSION,
            vec![
                embedded_source!(
                    "crates/neoethos-data/src/core/regime_detection.rs",
                    "/src/core/regime_detection.rs"
                ),
                embedded_source!(
                    "crates/neoethos-data/src/core/regime_exact_math_v1.rs",
                    "/src/core/regime_exact_math_v1.rs"
                ),
                embedded_source!("crates/neoethos-data/src/lib.rs", "/src/lib.rs"),
            ],
            Vec::new(),
        )?,
        producer_row(
            ProductionFeatureProducerId::Footprint,
            FeatureSource::Footprint,
            super::footprint_features::FOOTPRINT_SEMANTIC_VERSION,
            vec![
                embedded_source!(
                    "crates/neoethos-data/src/core/footprint_features.rs",
                    "/src/core/footprint_features.rs"
                ),
                embedded_source!(
                    "crates/neoethos-data/src/core/timestamps.rs",
                    "/src/core/timestamps.rs"
                ),
                embedded_source!("crates/neoethos-data/src/lib.rs", "/src/lib.rs"),
            ],
            Vec::new(),
        )?,
    ])
}

fn source_manifest(
    entries: Vec<SemanticSourceEntryV1>,
) -> std::result::Result<SemanticSourceManifestV1, FeatureContractError> {
    SemanticSourceManifestV1::new(entries)
}

fn producer_row(
    producer: ProductionFeatureProducerId,
    source: FeatureSource,
    semantic_version: u32,
    source_entries: Vec<SemanticSourceEntryV1>,
    dependencies: Vec<RelevantDependencyV1>,
) -> std::result::Result<ProductionFeatureProducerManifestRowV1, FeatureContractError> {
    let sources = source_manifest(source_entries)?;
    let dependencies = RelevantDependencySetV1::new(dependencies)?;
    Ok(ProductionFeatureProducerManifestRowV1 {
        producer,
        source,
        semantic_version,
        semantic_source_set: SemanticSourceSetV1::new(sources, dependencies),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureValueDtype {
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureValueKind {
    Continuous,
    Binary,
    SignedSignal,
    Ratio,
    Distance,
    State,
    Count,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureParameterKind {
    Timeframe,
    IndicatorId,
    ParameterSet,
    Period,
    LagBars,
    WindowBars,
    OutputLine,
    Session,
    Formula,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureOutputSchema {
    pub dtype: FeatureValueDtype,
    pub kind: FeatureValueKind,
    pub nullable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureParameterMetadata {
    pub name: String,
    pub kind: FeatureParameterKind,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureColumnMetadata {
    pub name: String,
    pub source: FeatureSource,
    pub output: FeatureOutputSchema,
    pub parameters: Vec<FeatureParameterMetadata>,
    pub requires_volume: bool,
}

const SMC_FEATURE_NAMES: &[&str] = &[
    "smc_ob",
    "smc_fvg",
    "smc_ifvg",
    "smc_liq_sweep",
    "smc_pd_array",
    "smc_killzone",
    "smc_displacement",
    "smc_breaker_block",
    "smc_mitigation_block",
    "smc_mss",
    "smc_volume_imbalance",
    "smc_bos",
    "smc_eqh",
    "smc_eql",
    "smc_inducement",
    "smc_asian_range",
    "smc_silver_bullet",
    "smc_judas_swing",
    "smc_nwog",
    "smc_ndog",
    "smc_ict_macro",
    "smc_fvg_strength",
    "smc_dealing_range_width",
    "smc_swing_range_pct",
    "smc_ob_strength",
    "smc_trend_bias",
    "smc_unicorn_model",
    "smc_rejection_block",
    "smc_propulsion_block",
    "smc_fib_time_ratio",
    "smc_fib_236",
    "smc_fib_382",
    "smc_fib_500",
    "smc_fib_618",
    "smc_fib_705",
    "smc_fib_786",
    "smc_fib_886",
    "smc_fib_1272",
    "smc_fib_1414",
    "smc_fib_1618",
    "smc_fib_2000",
    "smc_fib_2618",
    "smc_fvg_magnet_dist",
    "smc_fvg_magnet_age",
    "smc_fvg_inside",
    "smc_fvg_open_count",
];

const SESSION_FEATURE_NAMES: &[&str] = &[
    "session_london_open_dist",
    "session_london_high_dist",
    "session_london_low_dist",
    "session_london_range",
    "session_london_vwap_dist",
    "session_ny_open_dist",
    "session_ny_high_dist",
    "session_ny_low_dist",
    "session_ny_range",
    "session_ny_vwap_dist",
    "session_asian_open_dist",
    "session_asian_close_dist",
    "session_asian_range_norm",
    "session_london_ny_overlap",
    "session_vol_ratio",
    "session_prev_close_dist",
    "session_open_gap",
    "daily_range_pct",
    "daily_body_pct",
    "daily_position",
    "daily_high_dist",
    "daily_low_dist",
    "daily_vwap_dist",
];

const REGIME_FEATURE_NAMES: &[&str] = &REGIME_FEATURE_NAMES_V3;

const FOOTPRINT_FEATURE_NAMES: &[&str] = &[
    "fp_volume_z",
    "fp_absorption",
    "fp_effort_result_div",
    "fp_climax",
    "fp_delta_proxy",
    "fp_volprice_corr",
    "fp_fix_window",
];

const QUANT_EXACT_FEATURES: &[(&str, bool)] = &[
    ("quant_close", false),
    ("quant_log_return", false),
    ("quant_log_volatility", false),
    ("quant_vol_ratio", false),
    ("quant_hurst_100", false),
    ("quant_skewness_30", false),
    ("quant_kurtosis_30", false),
    ("quant_kyle_lambda", true),
    ("quant_vpin", true),
    ("quant_amihud_illiquidity", true),
    ("quant_roll_spread", false),
    ("quant_consec_up", false),
    ("quant_consec_down", false),
    ("quant_inside_bar", false),
    ("quant_outside_bar", false),
    ("quant_body_ratio", false),
    ("quant_upper_shadow", false),
    ("quant_lower_shadow", false),
    ("quant_prev_day_h_dist", false),
    ("quant_prev_day_l_dist", false),
    ("quant_prev_week_h_dist", false),
    ("quant_prev_week_l_dist", false),
    ("quant_amd_phase", false),
    ("quant_wyckoff", false),
    ("quant_engulfing_vol", true),
    ("quant_pivot_dist", false),
    ("quant_r1_dist", false),
    ("quant_r2_dist", false),
    ("quant_s1_dist", false),
    ("quant_s2_dist", false),
    ("quant_cam_r3_dist", false),
    ("quant_cam_s3_dist", false),
    ("quant_fractal_dim", false),
    ("quant_delta_volume", true),
    ("quant_cum_delta_zscore", true),
];

pub fn feature_column_metadata(name: &str) -> Option<FeatureColumnMetadata> {
    let (base_name, timeframe) = strip_timeframe_prefix(name);
    let mut metadata = feature_column_metadata_unprefixed(base_name)?;
    metadata.name = name.to_string();

    if let Some(timeframe) = timeframe {
        metadata.parameters.insert(
            0,
            parameter("timeframe", FeatureParameterKind::Timeframe, timeframe),
        );
    }

    Some(metadata)
}

pub fn feature_metadata_for_names(names: &[String]) -> Result<Vec<FeatureColumnMetadata>> {
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let Some(metadata) = feature_column_metadata(name) else {
            bail!("unregistered feature column: {name}");
        };
        out.push(metadata);
    }
    Ok(out)
}

pub fn validate_feature_names(names: &[String]) -> Result<()> {
    let unknown = unknown_feature_names(names);
    if !unknown.is_empty() {
        bail!("unregistered feature columns: {}", unknown.join(", "));
    }
    Ok(())
}

pub fn unknown_feature_names(names: &[String]) -> Vec<String> {
    names
        .iter()
        .filter(|name| feature_column_metadata(name).is_none())
        .cloned()
        .collect()
}

fn feature_column_metadata_unprefixed(name: &str) -> Option<FeatureColumnMetadata> {
    if REGIME_RETIRED_V2_FEATURE_NAMES.contains(&name) {
        debug_assert!(
            REGIME_V2_ARTIFACT_MIGRATION_POLICY.contains("refuse semantic-v2 Regime artifacts")
        );
        return None;
    }
    if SMC_FEATURE_NAMES.contains(&name) {
        return Some(group_metadata(
            name,
            FeatureSource::SmartMoneyConcept,
            infer_value_kind(name),
            false,
            smc_parameters(name),
        ));
    }

    if SESSION_FEATURE_NAMES.contains(&name) {
        return Some(group_metadata(
            name,
            FeatureSource::Session,
            infer_value_kind(name),
            false,
            session_parameters(name),
        ));
    }

    if REGIME_FEATURE_NAMES.contains(&name) {
        return Some(group_metadata(
            name,
            FeatureSource::Regime,
            regime_value_kind_v3(name),
            false,
            regime_parameters(name),
        ));
    }

    if FOOTPRINT_FEATURE_NAMES.contains(&name) {
        return Some(group_metadata(
            name,
            FeatureSource::Footprint,
            infer_value_kind(name),
            name != "fp_fix_window",
            vec![],
        ));
    }

    if let Some(metadata) = quant_metadata(name) {
        return Some(metadata);
    }

    classic_ta_metadata(name)
}

fn strip_timeframe_prefix(name: &str) -> (&str, Option<&str>) {
    let Some((candidate, rest)) = name.split_once('_') else {
        return (name, None);
    };

    if candidate.parse::<CanonicalTimeframe>().is_ok() {
        (rest, Some(candidate))
    } else {
        (name, None)
    }
}

fn quant_metadata(name: &str) -> Option<FeatureColumnMetadata> {
    for (candidate, requires_volume) in QUANT_EXACT_FEATURES {
        if name == *candidate {
            return Some(group_metadata(
                name,
                FeatureSource::Quantitative,
                infer_value_kind(name),
                *requires_volume,
                quant_exact_parameters(name),
            ));
        }
    }

    let parameterized = [
        (
            "quant_return_",
            FeatureParameterKind::LagBars,
            &[1, 2, 3, 5, 8, 13, 21][..],
            false,
        ),
        (
            "quant_realized_vol_",
            FeatureParameterKind::WindowBars,
            &[5, 10, 20, 50][..],
            false,
        ),
        (
            "quant_gk_vol_",
            FeatureParameterKind::WindowBars,
            &[10, 20][..],
            false,
        ),
        (
            "quant_parkinson_vol_",
            FeatureParameterKind::WindowBars,
            &[10, 20][..],
            false,
        ),
        (
            "quant_autocorr_",
            FeatureParameterKind::LagBars,
            &[1, 5, 10][..],
            false,
        ),
        (
            "quant_efficiency_ratio_",
            FeatureParameterKind::WindowBars,
            &[10, 20][..],
            false,
        ),
        (
            "quant_orb_",
            FeatureParameterKind::WindowBars,
            &[4, 8, 12][..],
            false,
        ),
        (
            "quant_zscore_",
            FeatureParameterKind::WindowBars,
            &[20, 50][..],
            false,
        ),
        (
            "quant_rvol_",
            FeatureParameterKind::WindowBars,
            &[10, 20, 50][..],
            true,
        ),
    ];

    for (prefix, kind, allowed_values, requires_volume) in parameterized {
        if let Some(value) = numeric_suffix(name, prefix, allowed_values) {
            return Some(group_metadata(
                name,
                FeatureSource::Quantitative,
                infer_value_kind(name),
                requires_volume,
                vec![parameter(parameter_name(kind), kind, value.to_string())],
            ));
        }
    }

    None
}

fn classic_ta_metadata(name: &str) -> Option<FeatureColumnMetadata> {
    if let Some((indicator_id, period, line)) = classic_multi_period_parts(name) {
        let mut parameters = vec![
            parameter(
                "indicator_id",
                FeatureParameterKind::IndicatorId,
                indicator_id,
            ),
            parameter("period", FeatureParameterKind::Period, period.to_string()),
        ];
        if let Some(line) = line {
            parameters.push(parameter(
                "output_line",
                FeatureParameterKind::OutputLine,
                line.to_string(),
            ));
        }

        return Some(group_metadata(
            name,
            FeatureSource::ClassicTechnicalAnalysis,
            infer_value_kind(name),
            classic_indicator_requires_volume(indicator_id),
            parameters,
        ));
    }

    if let Some((indicator_id, line)) = classic_default_parts(name) {
        let mut parameters = vec![
            parameter(
                "indicator_id",
                FeatureParameterKind::IndicatorId,
                indicator_id,
            ),
            parameter("params", FeatureParameterKind::ParameterSet, "default"),
        ];
        if let Some(line) = line {
            parameters.push(parameter(
                "output_line",
                FeatureParameterKind::OutputLine,
                line.to_string(),
            ));
        }

        return Some(group_metadata(
            name,
            FeatureSource::ClassicTechnicalAnalysis,
            infer_value_kind(name),
            classic_indicator_requires_volume(indicator_id),
            parameters,
        ));
    }

    None
}

fn classic_multi_period_parts(name: &str) -> Option<(&'static str, usize, Option<String>)> {
    let (indicator_id, suffix) = classic_indicator_and_suffix(name)?;
    let (period_text, output_text) = suffix
        .split_once('_')
        .map_or((suffix, None), |(period, output)| (period, Some(output)));
    let period = period_text.parse::<usize>().ok()?;
    if !crate::core::hpc_ta::ALT_PERIODS.contains(&period) {
        return None;
    }

    let output = match output_text {
        Some(output) => Some(validated_classic_output(indicator_id, output)?.to_string()),
        None => None,
    };
    Some((indicator_id, period, output))
}

fn classic_default_parts(name: &str) -> Option<(&'static str, Option<String>)> {
    let (indicator_id, suffix) = classic_indicator_and_suffix(name)?;
    if suffix.is_empty() {
        return Some((indicator_id, None));
    }

    Some((
        indicator_id,
        Some(validated_classic_output(indicator_id, suffix)?.to_string()),
    ))
}

/// Resolve the longest registered indicator prefix. Longest-match is
/// essential: `ema_deviation_corrected_t3` must never be classified as an
/// `ema` output merely because both are valid ids.
fn classic_indicator_and_suffix(name: &str) -> Option<(&'static str, &str)> {
    let indicator_id = ALL_INDICATORS
        .iter()
        .copied()
        .filter(|id| name == *id || name.starts_with(&format!("{id}_")))
        .max_by_key(|id| id.len())?;
    let suffix = name
        .strip_prefix(indicator_id)?
        .strip_prefix('_')
        .unwrap_or_default();
    Some((indicator_id, suffix))
}

fn validated_classic_output<'a>(indicator_id: &str, output: &'a str) -> Option<&'a str> {
    // Temporary read compatibility for the positional names emitted before
    // hpc_ta switched to semantic vector-ta output ids. The atomic Tasks 5B-9
    // migration removes this branch together with old artifacts.
    if output
        .strip_prefix("line")
        .is_some_and(|line| !line.is_empty() && line.chars().all(|ch| ch.is_ascii_digit()))
    {
        return Some(output);
    }

    if indicator_id == "pattern_recognition"
        && vector_ta::indicators::pattern_recognition::list_patterns()
            .iter()
            .any(|pattern| pattern.id == output)
    {
        return Some(output);
    }

    let outputs = crate::core::indicator_ledger::output_ids_for(indicator_id);
    outputs
        .into_iter()
        .flatten()
        .any(|declared| declared == output)
        .then_some(output)
}

fn numeric_suffix(name: &str, prefix: &str, allowed_values: &[usize]) -> Option<usize> {
    let value = name.strip_prefix(prefix)?.parse::<usize>().ok()?;
    allowed_values.contains(&value).then_some(value)
}

fn group_metadata(
    name: &str,
    source: FeatureSource,
    kind: FeatureValueKind,
    requires_volume: bool,
    parameters: Vec<FeatureParameterMetadata>,
) -> FeatureColumnMetadata {
    FeatureColumnMetadata {
        name: name.to_string(),
        source,
        output: FeatureOutputSchema {
            dtype: FeatureValueDtype::F64,
            kind,
            nullable: false,
        },
        parameters,
        requires_volume,
    }
}

fn parameter(
    name: &str,
    kind: FeatureParameterKind,
    value: impl ToString,
) -> FeatureParameterMetadata {
    FeatureParameterMetadata {
        name: name.to_string(),
        kind,
        value: value.to_string(),
    }
}

fn parameter_name(kind: FeatureParameterKind) -> &'static str {
    match kind {
        FeatureParameterKind::LagBars => "lag_bars",
        FeatureParameterKind::WindowBars => "window_bars",
        FeatureParameterKind::Period => "period",
        _ => "parameter",
    }
}

fn smc_parameters(name: &str) -> Vec<FeatureParameterMetadata> {
    let mut parameters = Vec::new();
    if name.contains("_fib_") || name == "smc_pd_array" || name == "smc_dealing_range_width" {
        parameters.push(parameter(
            "lookback_bars",
            FeatureParameterKind::WindowBars,
            40,
        ));
    }
    if matches!(name, "smc_eqh" | "smc_eql" | "smc_bos") {
        parameters.push(parameter(
            "swing_fractal",
            FeatureParameterKind::Formula,
            "5_bar",
        ));
    }
    parameters
}

fn session_parameters(name: &str) -> Vec<FeatureParameterMetadata> {
    let mut parameters = vec![parameter(
        "timestamp_policy",
        FeatureParameterKind::Formula,
        "utc_session_windows",
    )];

    if name.contains("london") {
        parameters.push(parameter(
            "session",
            FeatureParameterKind::Session,
            "London",
        ));
    } else if name.contains("_ny_") {
        parameters.push(parameter(
            "session",
            FeatureParameterKind::Session,
            "NewYork",
        ));
    } else if name.contains("asian") {
        parameters.push(parameter("session", FeatureParameterKind::Session, "Asian"));
    } else if name.starts_with("daily_") {
        parameters.push(parameter("session", FeatureParameterKind::Session, "Daily"));
    }

    parameters
}

fn regime_parameters(name: &str) -> Vec<FeatureParameterMetadata> {
    match name {
        "neoethos_custom_gk_vol_ratio_state_10_50_v3"
        | "neoethos_custom_gk_vol_ratio_offset_10_50_v3" => vec![
            parameter("short_window_bars", FeatureParameterKind::WindowBars, 10),
            parameter("long_window_bars", FeatureParameterKind::WindowBars, 50),
        ],
        "regime_wilder_adx_14_v3"
        | "neoethos_custom_wilder_di_dominance_direction_14_v3"
        | "neoethos_custom_wilder_adx_direction_state_14_25_v3"
        | "regime_dreiss_choppiness_index_14_v3" => {
            vec![parameter("period", FeatureParameterKind::Period, 14)]
        }
        "neoethos_custom_bollinger_keltner_squeeze_state_20_2_1p5_v3"
        | "neoethos_custom_bollinger_midline_atr_deviation_20_v3"
        | "neoethos_custom_directional_persistence_balance_20_v3" => vec![parameter(
            "window_bars",
            FeatureParameterKind::WindowBars,
            20,
        )],
        "neoethos_custom_candle_body_range_balance_8_v3" => vec![parameter(
            "window_bars",
            FeatureParameterKind::WindowBars,
            8,
        )],
        "neoethos_custom_standardized_cusum_up_50_0p5_3_v3"
        | "neoethos_custom_standardized_cusum_down_50_0p5_3_v3"
        | "neoethos_custom_standardized_cusum_signal_50_0p5_3_v3" => vec![parameter(
            "baseline_window_bars",
            FeatureParameterKind::WindowBars,
            50,
        )],
        "neoethos_custom_equal_width_log_return_entropy_30_10_v3" => vec![
            parameter("window_bars", FeatureParameterKind::WindowBars, 30),
            parameter("bin_count", FeatureParameterKind::Formula, 10),
        ],
        _ => Vec::new(),
    }
}

/// Semantic-v3 Regime output kinds are explicit. The generic name heuristic
/// cannot distinguish a continuous indicator whose namespace starts with
/// `regime_` from a categorical state.
fn regime_value_kind_v3(name: &str) -> FeatureValueKind {
    match name {
        "neoethos_custom_gk_vol_ratio_state_10_50_v3"
        | "neoethos_custom_wilder_adx_direction_state_14_25_v3"
        | "neoethos_custom_bollinger_keltner_squeeze_state_20_2_1p5_v3" => FeatureValueKind::State,
        "neoethos_custom_wilder_di_dominance_direction_14_v3"
        | "neoethos_custom_standardized_cusum_signal_50_0p5_3_v3" => FeatureValueKind::SignedSignal,
        "neoethos_custom_gk_vol_ratio_offset_10_50_v3"
        | "neoethos_custom_bollinger_midline_atr_deviation_20_v3" => FeatureValueKind::Distance,
        "neoethos_custom_directional_persistence_balance_20_v3"
        | "neoethos_custom_candle_body_range_balance_8_v3" => FeatureValueKind::Ratio,
        "regime_wilder_adx_14_v3"
        | "regime_dreiss_choppiness_index_14_v3"
        | "neoethos_custom_standardized_cusum_up_50_0p5_3_v3"
        | "neoethos_custom_standardized_cusum_down_50_0p5_3_v3"
        | "neoethos_custom_equal_width_log_return_entropy_30_10_v3" => FeatureValueKind::Continuous,
        _ => unreachable!("Regime-v3 registry admitted an unknown name `{name}`"),
    }
}

fn quant_exact_parameters(name: &str) -> Vec<FeatureParameterMetadata> {
    match name {
        "quant_hurst_100" => vec![parameter(
            "window_bars",
            FeatureParameterKind::WindowBars,
            100,
        )],
        "quant_skewness_30" | "quant_kurtosis_30" | "quant_fractal_dim" => {
            vec![parameter(
                "window_bars",
                FeatureParameterKind::WindowBars,
                30,
            )]
        }
        "quant_kyle_lambda" | "quant_amihud_illiquidity" | "quant_roll_spread" => {
            vec![parameter(
                "window_bars",
                FeatureParameterKind::WindowBars,
                20,
            )]
        }
        "quant_vpin" => vec![
            parameter("bucket_size_bars", FeatureParameterKind::WindowBars, 50),
            parameter("bucket_count", FeatureParameterKind::Formula, 10),
        ],
        "quant_prev_day_h_dist"
        | "quant_prev_day_l_dist"
        | "quant_pivot_dist"
        | "quant_r1_dist"
        | "quant_r2_dist"
        | "quant_s1_dist"
        | "quant_s2_dist"
        | "quant_cam_r3_dist"
        | "quant_cam_s3_dist" => {
            vec![parameter(
                "window_bars",
                FeatureParameterKind::WindowBars,
                24,
            )]
        }
        "quant_prev_week_h_dist" | "quant_prev_week_l_dist" => {
            vec![parameter(
                "window_bars",
                FeatureParameterKind::WindowBars,
                120,
            )]
        }
        "quant_amd_phase" => vec![parameter(
            "window_bars",
            FeatureParameterKind::WindowBars,
            20,
        )],
        "quant_wyckoff" => vec![parameter(
            "window_bars",
            FeatureParameterKind::WindowBars,
            30,
        )],
        _ => Vec::new(),
    }
}

fn classic_indicator_requires_volume(indicator_id: &str) -> bool {
    indicator_id.contains("volume")
        || indicator_id.contains("vwap")
        || matches!(
            indicator_id,
            "ad" | "adosc" | "mfi" | "obv" | "vpt" | "vosc" | "vpci" | "vwma" | "vwmacd"
        )
}

fn infer_value_kind(name: &str) -> FeatureValueKind {
    if name.contains("overlap")
        || name.contains("killzone")
        || name.contains("silver_bullet")
        || name.contains("ict_macro")
        || name.contains("inside_bar")
        || name.contains("outside_bar")
        || name.contains("squeeze")
    {
        FeatureValueKind::Binary
    } else if name.contains("state") || name.contains("phase") || name.contains("regime") {
        FeatureValueKind::State
    } else if name.starts_with("pattern_recognition_")
        || name.contains("signal")
        || name.contains("direction")
        || name.contains("bias")
        || name.contains("swing")
        || name.ends_with("_bos")
        || name.ends_with("_mss")
    {
        FeatureValueKind::SignedSignal
    } else if name.contains("ratio")
        || name.contains("_pct")
        || name.contains("_range")
        || name.contains("vol_")
        || name.contains("_vol")
    {
        FeatureValueKind::Ratio
    } else if name.contains("dist") || name.contains("zscore") || name.contains("z_score") {
        FeatureValueKind::Distance
    } else if name.contains("count") || name.contains("consec") {
        FeatureValueKind::Count
    } else {
        FeatureValueKind::Continuous
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_ta_pattern_columns_are_signed_signals() {
        assert_eq!(
            infer_value_kind("pattern_recognition_cdlharami"),
            FeatureValueKind::SignedSignal
        );
    }

    #[test]
    fn resolves_explicit_feature_groups() {
        for name in SMC_FEATURE_NAMES
            .iter()
            .chain(SESSION_FEATURE_NAMES)
            .chain(REGIME_FEATURE_NAMES)
            .chain(FOOTPRINT_FEATURE_NAMES)
        {
            assert!(
                feature_column_metadata(name).is_some(),
                "{name} should have registry metadata"
            );
        }
    }

    #[test]
    fn regime_v3_uses_truthful_explicit_output_kinds() {
        for (name, expected) in [
            ("regime_wilder_adx_14_v3", FeatureValueKind::Continuous),
            (
                "regime_dreiss_choppiness_index_14_v3",
                FeatureValueKind::Continuous,
            ),
            (
                "neoethos_custom_wilder_di_dominance_direction_14_v3",
                FeatureValueKind::SignedSignal,
            ),
            (
                "neoethos_custom_gk_vol_ratio_offset_10_50_v3",
                FeatureValueKind::Distance,
            ),
        ] {
            assert_eq!(
                feature_column_metadata(name)
                    .expect("registered Regime-v3 metadata")
                    .output
                    .kind,
                expected,
                "Regime-v3 kind drifted for {name}"
            );
        }
        for retired in REGIME_RETIRED_V2_FEATURE_NAMES {
            assert!(
                feature_column_metadata(retired).is_none(),
                "semantic-v2 Regime name `{retired}` must fail closed"
            );
        }
    }

    #[test]
    fn resolves_parameterized_quant_features() {
        let close = feature_column_metadata("quant_close").expect("swarm close metadata");
        assert_eq!(close.source, FeatureSource::Quantitative);

        let hmm_volatility =
            feature_column_metadata("quant_log_volatility").expect("HMM volatility metadata");
        assert_eq!(hmm_volatility.source, FeatureSource::Quantitative);
        assert!(!hmm_volatility.requires_volume);

        let rvol = feature_column_metadata("quant_rvol_20").expect("rvol metadata");
        assert_eq!(rvol.source, FeatureSource::Quantitative);
        assert!(rvol.requires_volume);
        assert_eq!(rvol.parameters[0].name, "window_bars");

        let prefixed = feature_column_metadata("H1_quant_return_13").expect("prefixed metadata");
        assert_eq!(prefixed.name, "H1_quant_return_13");
        assert_eq!(prefixed.parameters[0].kind, FeatureParameterKind::Timeframe);
        assert_eq!(prefixed.parameters[1].kind, FeatureParameterKind::LagBars);
    }

    #[test]
    fn resolves_vector_ta_defaults_and_period_variants() {
        let default_rsi = feature_column_metadata("rsi").expect("rsi metadata");
        assert_eq!(default_rsi.source, FeatureSource::ClassicTechnicalAnalysis);

        let period_line =
            feature_column_metadata("bollinger_bands_21_line2").expect("period line metadata");
        assert_eq!(period_line.parameters[1].kind, FeatureParameterKind::Period);
        assert_eq!(
            period_line.parameters[2].kind,
            FeatureParameterKind::OutputLine
        );

        // Static removal of a redundant sibling may leave one semantically
        // named output. It must keep that output id in both the feature name
        // and the typed source metadata; `len() > 1` is not a valid proxy for
        // whether an output id exists.
        let estimate = feature_column_metadata("half_causal_estimator_estimate")
            .expect("remaining named output has classic-TA source metadata");
        assert_eq!(estimate.source, FeatureSource::ClassicTechnicalAnalysis);
        assert!(estimate.parameters.iter().any(|parameter| {
            parameter.kind == FeatureParameterKind::OutputLine && parameter.value == "estimate"
        }));
        let estimate_50 = feature_column_metadata("half_causal_estimator_50_estimate")
            .expect("period variant of remaining named output has source metadata");
        assert!(estimate_50.parameters.iter().any(|parameter| {
            parameter.kind == FeatureParameterKind::Period && parameter.value == "50"
        }));
        assert!(
            feature_column_metadata("half_causal_estimator_expected_value").is_none(),
            "the disabled all-NaN output must not remain registered as a production source"
        );
    }

    #[test]
    fn rejects_unknown_feature_names() {
        let names = vec!["quant_return_1".to_string(), "quant_return_4".to_string()];
        let unknown = unknown_feature_names(&names);
        assert_eq!(unknown, vec!["quant_return_4".to_string()]);
        assert!(validate_feature_names(&names).is_err());
    }

    #[test]
    fn classic_vector_ta_v9_preserves_evwma_and_binds_all_audited_repairs() {
        let classic = production_feature_producer_manifest_v1()
            .expect("embedded producer manifest")
            .iter()
            .find(|row| row.producer() == ProductionFeatureProducerId::ClassicVectorTa)
            .expect("ClassicVectorTa manifest row");

        assert_eq!(classic.semantic_version(), 9);
        assert!(
            CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V8
                .contains("evwma/rolling-volume-sum/close/f64/v1")
        );
        assert!(
            CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V8
                .contains("evwma/rolling-length-key/length/default-30/f64/v1")
        );
        assert!(CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V8.contains("evwma/fixed-n/close/f64/v1"));
        assert!(CLASSIC_VECTOR_TA_V7_ARTIFACT_MIGRATION_POLICY.contains("semantic-v7"));
        assert!(CLASSIC_VECTOR_TA_V7_ARTIFACT_MIGRATION_POLICY.contains("unversioned"));
        assert!(CLASSIC_VECTOR_TA_V7_ARTIFACT_MIGRATION_POLICY.contains("refuse"));
        assert!(CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V9.contains("classic-composite-v9"));
        assert!(
            CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V9
                .contains("cci-cycle/creator-pine-v3/local-current-resolution/f64/v1")
        );
        assert!(CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V9.contains("floor-half"));
        assert!(CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V9.contains("sma-seeded-ema-rma"));
        assert!(CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V9.contains("startup-flat-zero-carry"));
        assert!(CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V9.contains("factor-zero-freeze"));
        assert!(CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V9.contains("finite-segment-reset-v1"));
        assert!(
            CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V9
                .contains("frama-f64-v3-finite-hlc-segment-reset-even-window-stable-fma-v2")
        );
        assert!(
            CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V9.contains("frama/finite-hlc-segment-reset/v3")
        );
        assert!(CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V9.contains("frama/evenized-window-seed/v1"));
        assert!(CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V9.contains("frama/stable-affine-fma/v2"));
        assert!(
            CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V9
                .contains("6D2380A30ECA86E77DDD7B461F0A9D961450C82CDD52B19653F148852A3FF7FE")
        );
        assert!(
            CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V9
                .contains("AACB6789BEE22C5FDE46C1966EA956E8E46209B42720D4DD900A5CD94AB1AD02")
        );
        assert!(
            CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V9
                .contains("fwma-f64-v2-p254-u192-fib-pow2-dd-fma-window-recovery")
        );
        assert!(
            CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V9
                .contains("D5F2E5D59128C02858E0DDB236A9EAB6425883A3978A67A7221A3FCEF42F6AC3")
        );
        assert!(
            CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V9
                .contains("C7716141216AC0EE144430092F570606821415D78B75DDA756F91A64415A24EE")
        );
        assert!(
            CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V9.contains(
                "fisher-f64-v2-openlibm-e-log-midpoint-finite-segment-reset-oN-deque-bounded-faithful-p1024"
            )
        );
        assert!(
            CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V9.contains(
                "half-causal-estimator-f64-v2-neoethos-canonical-pine6-script24-utc-day-slot-session-proxy-cached-future-windows-stable-f64-registry-ratio-dl"
            )
        );
        assert!(CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V9.contains("hce/data-period-zero"));
        assert!(CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V9.contains(
            "hce/registry-ratio-dl/7-d2-l7/20-d5-l20/21-d5-l21/50-d13-l50/100-d25-l100/200-d50-l200/v2"
        ));
        assert!(
            CLASSIC_VECTOR_TA_SEMANTIC_AUTHORITY_V9.contains("hce/public-retained-budget-64mib/v1")
        );
        assert!(CLASSIC_VECTOR_TA_V8_ARTIFACT_MIGRATION_POLICY.contains("semantic-v8"));
        assert!(CLASSIC_VECTOR_TA_V8_ARTIFACT_MIGRATION_POLICY.contains("unversioned"));
        assert!(CLASSIC_VECTOR_TA_V8_ARTIFACT_MIGRATION_POLICY.contains("refuse"));

        let source = include_str!("feature_registry.rs");
        for path in [
            [
                "vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/",
                "elastic_volume_weighted_moving_average.rs",
            ]
            .concat(),
            [
                "vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/",
                "cuda_f64.rs",
            ]
            .concat(),
            [
                "vendor/vector-ta-0.2.9-patched/kernels/cuda/moving_averages/",
                "elastic_volume_weighted_moving_average_kernel.cu",
            ]
            .concat(),
            ["crates/neoethos-data/src/core/", "gpu_indicators.rs"].concat(),
        ] {
            let canonical_entry = format!("\"{path}\",");
            assert_eq!(
                source.matches(&canonical_entry).count(),
                1,
                "ClassicVectorTa v9 must preserve EVWMA binding `{path}` exactly once"
            );
        }
        for path in [
            [
                "vendor/vector-ta-0.2.9-patched/src/indicators/",
                "cci_cycle.rs",
            ]
            .concat(),
            [
                "vendor/vector-ta-0.2.9-patched/src/cuda/",
                "neoethos_f64_wrapper.rs",
            ]
            .concat(),
            [
                "vendor/vector-ta-0.2.9-patched/src/cuda/",
                "cci_cycle_wrapper.rs",
            ]
            .concat(),
            [
                "vendor/vector-ta-0.2.9-patched/kernels/cuda/oscillators/",
                "cci_cycle_kernel.cu",
            ]
            .concat(),
            ["vendor/vector-ta-0.2.9-patched/", "build.rs"].concat(),
            ["crates/neoethos-data/src/core/", "hpc_ta.rs"].concat(),
            [
                "vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/",
                "frama.rs",
            ]
            .concat(),
            [
                "vendor/vector-ta-0.2.9-patched/kernels/cuda/moving_averages/",
                "frama_kernel.cu",
            ]
            .concat(),
            [
                "vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/",
                "fwma.rs",
            ]
            .concat(),
            [
                "vendor/vector-ta-0.2.9-patched/kernels/cuda/moving_averages/",
                "fwma_kernel.cu",
            ]
            .concat(),
            [
                "vendor/vector-ta-0.2.9-patched/src/indicators/",
                "fisher.rs",
            ]
            .concat(),
            [
                "vendor/vector-ta-0.2.9-patched/kernels/cuda/oscillators/",
                "fisher_kernel.cu",
            ]
            .concat(),
            [
                "vendor/vector-ta-0.2.9-patched/src/indicators/",
                "half_causal_estimator.rs",
            ]
            .concat(),
            [
                "vendor/vector-ta-0.2.9-patched/src/indicators/",
                "half_causal_estimator_stable_math.rs",
            ]
            .concat(),
            [
                "vendor/vector-ta-0.2.9-patched/kernels/cuda/",
                "half_causal_estimator_kernel.cu",
            ]
            .concat(),
            ["crates/neoethos-data/src/core/", "classic_cuda_plan.rs"].concat(),
            [
                "crates/neoethos-data/src/core/",
                "gpu_resident_classic_ta_v3.rs",
            ]
            .concat(),
        ] {
            let canonical_entry = format!("\"{path}\",");
            assert_eq!(
                source.matches(&canonical_entry).count(),
                1,
                "ClassicVectorTa v9 must bind `{path}` exactly once"
            );
        }
    }
}
