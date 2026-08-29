use std::fs;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("neoethos-data must remain under crates/")
        .to_path_buf()
}

fn source(relative: &str) -> String {
    fs::read_to_string(workspace_root().join(relative)).expect(relative)
}

#[test]
fn classic_v9_is_a_fail_closed_composite_authority() {
    let registry = source("crates/neoethos-data/src/core/feature_registry.rs");
    assert!(registry.contains("CLASSIC_VECTOR_TA_SEMANTIC_VERSION_V9: u32 = 9"));
    assert!(registry.contains("classic-composite-v9"));
    assert!(registry.contains("cci-cycle/creator-pine-v3/local-current-resolution/f64/v1"));
    assert!(registry.contains("floor-half"));
    assert!(registry.contains("sma-seeded-ema-rma"));
    assert!(registry.contains("startup-flat-zero-carry"));
    assert!(registry.contains("factor-zero-freeze"));
    assert!(registry.contains("finite-segment-reset-v1"));
    assert!(registry.contains("d00a0186f28989a34eb1da24eb9fae9a8906736afe413e2492ded9dc4b2a9c9f"));
    assert!(registry.contains("refuse semantic-v8"));
    assert!(registry.contains("unversioned ClassicVectorTa artifacts"));
    assert!(registry.contains("regenerate them under semantic-v9"));

    for path in [
        "vendor/vector-ta-0.2.9-patched/src/indicators/cci_cycle.rs",
        "vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cuda_f64.rs",
        "vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs",
        "vendor/vector-ta-0.2.9-patched/src/cuda/cci_cycle_wrapper.rs",
        "vendor/vector-ta-0.2.9-patched/kernels/cuda/oscillators/cci_cycle_kernel.cu",
        "vendor/vector-ta-0.2.9-patched/build.rs",
        "crates/neoethos-data/src/core/hpc_ta.rs",
    ] {
        let canonical_entry = format!("\"{path}\",");
        assert_eq!(
            registry.matches(&canonical_entry).count(),
            1,
            "ClassicVectorTa v9 must bind `{path}` exactly once"
        );
    }
    for shared_path in [
        "vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cpu_batch.rs",
        "vendor/vector-ta-0.2.9-patched/src/indicators/registry.rs",
    ] {
        assert!(
            registry.contains(&format!("\"{shared_path}\",")),
            "ClassicVectorTa v9 must retain shared source `{shared_path}`"
        );
    }
}

#[test]
fn classic_v9_binds_the_frozen_frama_v3_authority_and_sources() {
    let registry = source("crates/neoethos-data/src/core/feature_registry.rs");
    for token in [
        "frama-f64-v3-finite-hlc-segment-reset-even-window-stable-fma-v2",
        "frama/finite-hlc-segment-reset/v3",
        "frama/evenized-window-seed/v1",
        "frama/stable-affine-fma/v2",
        "6D2380A30ECA86E77DDD7B461F0A9D961450C82CDD52B19653F148852A3FF7FE",
        "AACB6789BEE22C5FDE46C1966EA956E8E46209B42720D4DD900A5CD94AB1AD02",
    ] {
        assert!(
            registry.contains(token),
            "ClassicVectorTa v9 authority must freeze `{token}`"
        );
    }

    for path in [
        "vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/frama.rs",
        "vendor/vector-ta-0.2.9-patched/kernels/cuda/moving_averages/frama_kernel.cu",
    ] {
        let canonical_entry = format!("\"{path}\",");
        assert_eq!(
            registry.matches(&canonical_entry).count(),
            1,
            "ClassicVectorTa v9 must bind frozen FRAMA source `{path}` exactly once"
        );
    }

    let host = source("vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/frama.rs");
    let cuda =
        source("vendor/vector-ta-0.2.9-patched/kernels/cuda/moving_averages/frama_kernel.cu");
    let identity = "frama-f64-v3-finite-hlc-segment-reset-even-window-stable-fma-v2";
    assert!(host.contains(identity));
    assert!(cuda.contains(identity));
}

#[test]
fn classic_v9_binds_the_frozen_fwma_v2_authority_and_sources() {
    let registry = source("crates/neoethos-data/src/core/feature_registry.rs");
    for token in [
        "fwma-f64-v2-p254-u192-fib-pow2-dd-fma-window-recovery",
        "D5F2E5D59128C02858E0DDB236A9EAB6425883A3978A67A7221A3FCEF42F6AC3",
        "C7716141216AC0EE144430092F570606821415D78B75DDA756F91A64415A24EE",
    ] {
        assert!(
            registry.contains(token),
            "ClassicVectorTa v9 authority must freeze `{token}`"
        );
    }

    for path in [
        "vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/fwma.rs",
        "vendor/vector-ta-0.2.9-patched/kernels/cuda/moving_averages/fwma_kernel.cu",
    ] {
        let canonical_entry = format!("\"{path}\",");
        assert_eq!(
            registry.matches(&canonical_entry).count(),
            1,
            "ClassicVectorTa v9 must bind frozen FWMA source `{path}` exactly once"
        );
    }

    let host = source("vendor/vector-ta-0.2.9-patched/src/indicators/moving_averages/fwma.rs");
    let cuda = source("vendor/vector-ta-0.2.9-patched/kernels/cuda/moving_averages/fwma_kernel.cu");
    let identity = "fwma-f64-v2-p254-u192-fib-pow2-dd-fma-window-recovery";
    assert!(host.contains(identity));
    assert!(cuda.contains(identity));
}

#[test]
fn classic_v9_binds_the_frozen_fisher_v2_authority_and_sources() {
    let registry = source("crates/neoethos-data/src/core/feature_registry.rs");
    for token in [
        "fisher-f64-v2-openlibm-e-log-midpoint-finite-segment-reset-oN-deque-bounded-faithful-p1024",
        "B97652FCFB1BD711DE5B33F90564AA0DB02D46E9187C1134F84578B19BC724D6",
        "4F548C7B1A0A10864B6FB398C26BF355C887BD282FB3B63A989D6858FCEE158A",
        "8996B789A4CBBCEF7CF7D568C1BE558CE9110900A40CA6C46FB4ED46C343CAFD",
        "7F4F37742F7EE8C8A79A5F8D244D1EE41423197A2842C06BF2E62FC165FBE5B9",
    ] {
        assert!(
            registry.contains(token),
            "ClassicVectorTa v9 Fisher authority must freeze `{token}`"
        );
    }

    for path in [
        "vendor/vector-ta-0.2.9-patched/src/indicators/fisher.rs",
        "vendor/vector-ta-0.2.9-patched/kernels/cuda/oscillators/fisher_kernel.cu",
        "vendor/vector-ta-0.2.9-patched/tests/fixtures/openlibm/e_log-82e90aef0657289192efe77be89791c07dea0775.c",
        "vendor/vector-ta-0.2.9-patched/tests/fixtures/openlibm/e_log-82e90aef0657289192efe77be89791c07dea0775.receipt.txt",
    ] {
        let canonical_entry = format!("\"{path}\",");
        assert_eq!(
            registry.matches(&canonical_entry).count(),
            1,
            "ClassicVectorTa v9 must bind frozen Fisher source `{path}` exactly once"
        );
    }
}

#[test]
fn classic_v9_binds_the_frozen_hce_v2_registry_ratio_authority_and_sources() {
    let registry = source("crates/neoethos-data/src/core/feature_registry.rs");
    for token in [
        "half-causal-estimator-f64-v2-neoethos-canonical-pine6-script24-utc-day-slot-session-proxy-cached-future-windows-stable-f64-registry-ratio-dl",
        "hce/registry-ratio-dl/7-d2-l7/20-d5-l20/21-d5-l21/50-d13-l50/100-d25-l100/200-d50-l200/v2",
        "hce/public-retained-budget-64mib/v1",
        "3632A8F08DF17BDE65A06C17068A6FE79BDE8F11E3A054688A956F32C84FCC6B",
        "B9B87151A498EAE775C75CF5669799A59C38130A0B729846B09139DD448E0796",
        "F1CE1AE5272EAED95EFBCBC87034C4CF1B72BC25AF0A6504FA0BFC1D29E4F528",
        "F93F18CEC0912BBE15B481BD0575AF7DB7225E368FA4DF7B658A45E868245B64",
        "D371BB32D723C17997EA210E230597FFFD1AD876C7A537DA3DFCD272EC4582AD",
        "18D24B85AA160B571BDE2BB6D023046C7403EE309F9C841694C51A1F8B90650F",
        "4B7FD8AEC6B333A4ECE967D7CFA6D957357CE436CB098E96EB1EB8A1480A8080",
    ] {
        assert!(
            registry.contains(token),
            "ClassicVectorTa v9 HCE authority must freeze `{token}`"
        );
    }

    for path in [
        "vendor/vector-ta-0.2.9-patched/src/indicators/half_causal_estimator.rs",
        "vendor/vector-ta-0.2.9-patched/src/indicators/half_causal_estimator_stable_math.rs",
        "vendor/vector-ta-0.2.9-patched/src/cuda/half_causal_estimator_wrapper.rs",
        "vendor/vector-ta-0.2.9-patched/kernels/cuda/half_causal_estimator_kernel.cu",
        "vendor/vector-ta-0.2.9-patched/audit_receipts/half_causal_estimator/tradingview_pine_facade_script24_raw.json",
        "vendor/vector-ta-0.2.9-patched/audit_receipts/half_causal_estimator/script24_receipt.toml",
        "crates/neoethos-data/src/core/classic_cuda_plan.rs",
        "crates/neoethos-data/src/core/gpu_resident_classic_ta_v3.rs",
    ] {
        let canonical_entry = format!("\"{path}\",");
        assert_eq!(
            registry.matches(&canonical_entry).count(),
            1,
            "ClassicVectorTa v9 must bind frozen HCE source `{path}` exactly once"
        );
    }
}

#[test]
fn classic_v9_binds_the_frozen_eacp_cooperative_cta_authority_and_sources() {
    let registry = source("crates/neoethos-data/src/core/feature_registry.rs");
    for token in [
        "eacp-f64-v1-vector-ta-pearson-dft-sq2-decaying-max-cog50-biased-ema-finite-segment-reset",
        "eacp/strict-cuda-exact-cooperative-cta/fmad-off/v1",
        "0108F73AC2DE644855A5E93999D211C2634C50A18B182D4337672D808A7D06EE",
        "12224A4C7F1B10612491E5BBD4608011E7D32F9737B5DF3A0259A3AB3E9688B0",
        "4E55CA5A5203013D255BA49F5A0C5B6FE3F12682CAB16234AAB62063C3582C36",
    ] {
        assert!(
            registry.contains(token),
            "ClassicVectorTa v9 EACP authority must freeze `{token}`"
        );
    }

    for path in [
        "vendor/vector-ta-0.2.9-patched/src/indicators/ehlers_autocorrelation_periodogram.rs",
        "vendor/vector-ta-0.2.9-patched/kernels/cuda/ehlers_autocorrelation_periodogram_kernel.cu",
        "vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs",
    ] {
        let canonical_entry = format!("\"{path}\",");
        assert_eq!(
            registry.matches(&canonical_entry).count(),
            1,
            "ClassicVectorTa v9 must bind frozen EACP source `{path}` exactly once"
        );
    }
}

#[test]
fn every_active_cci_cycle_route_declares_its_semantics() {
    let cpu = source("vendor/vector-ta-0.2.9-patched/src/indicators/cci_cycle.rs");
    assert!(cpu.contains("CCI_CYCLE_CLASSIC_SEMANTIC_VERSION: u32 = 9"));
    assert!(cpu.contains("CCI_CYCLE_CREATOR_AUDIT_ORACLE_URL"));
    assert!(cpu.contains("pine-facade.tradingview.com"));
    assert!(cpu.contains("CCI_CYCLE_CREATOR_AUDIT_ORACLE_SHA256"));
    assert!(cpu.contains("let half = length / 2;"));
    assert!(cpu.contains("if !close.is_finite()"));
    assert!(cpu.contains("previous_pf + self.factor * (f1 - self.previous_pf)"));
    assert!(!cpu.contains("length * 4"));
    assert!(!cpu.contains("init_matrix_prefixes("));
    assert!(!cpu.contains("init_matrix_prefixes,"));
    assert!(!cpu.contains("make_uninit_matrix("));
    assert!(!cpu.contains("make_uninit_matrix,"));

    let registry = source("vendor/vector-ta-0.2.9-patched/src/indicators/registry.rs");
    let params_start = registry
        .find("const PARAM_CCI_CYCLE")
        .expect("CCI Cycle registry params");
    let params_end = registry[params_start..]
        .find("const PARAM_CFO")
        .map(|offset| params_start + offset)
        .expect("next registry parameter family");
    let params = &registry[params_start..params_end];
    assert!(params.contains("default: Some(ParamValueStatic::Int(10))"));
    assert!(params.contains("min: Some(2.0)"));
    assert!(params.contains("default: Some(ParamValueStatic::Float(0.5))"));

    let dispatch = source("vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cuda_f64.rs");
    let marker = "indicator_id: \"cci_cycle\"";
    let marker_offset = dispatch.find(marker).expect("CCI Cycle strict-f64 row");
    let row_start = dispatch[..marker_offset]
        .rfind("F64KernelSpec {")
        .expect("CCI Cycle row start");
    let row_end = dispatch[marker_offset..]
        .find("},")
        .map(|offset| marker_offset + offset)
        .expect("CCI Cycle row end");
    let row = &dispatch[row_start..row_end];
    assert!(row.contains("kernel: F64Kernel::CciCycle"));
    assert!(row.contains("input: F64InputKind::CloseSlice"));
    assert!(row.contains("first_valid: F64FirstValidRule::Ignored"));

    let kernel =
        source("vendor/vector-ta-0.2.9-patched/kernels/cuda/oscillators/cci_cycle_kernel.cu");
    assert!(kernel.contains("CCI_CYCLE_F32_LEGACY_SEMANTIC_VERSION 8"));
    assert!(kernel.contains("NEO_CCICYC_CLASSIC_SEMANTIC_VERSION 9"));
    assert!(kernel.contains("const int half = length / 2;"));
    assert!(kernel.contains("length < 2"));
    assert!(kernel.contains("if (!isfinite(close))"));
    assert!(kernel.contains("previous_pf + factor * (f1 - previous_pf)"));

    let strict_wrapper = source("vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
    assert!(strict_wrapper.contains("CCI_CYCLE_MAX_LENGTH: usize = 200"));
    assert!(strict_wrapper.contains("CCI_CYCLE_PRODUCTION_FACTOR_V9: f64 = 0.5"));
    assert!(strict_wrapper.contains("CCI_CYCLE_CUSTOM_FACTOR_CUDA_SUPPORTED_V9: bool = false"));
    assert!(strict_wrapper.contains("F64Kernel::CciCycle => \"cci_cycle_neo_batch_f64\""));
    assert!(strict_wrapper.contains("F64Kernel::CciCycle => Some(CCI_CYCLE_MAX_LENGTH)"));
    assert!(strict_wrapper.contains("fn validate_cci_cycle_periods_v9("));

    let sync_start = strict_wrapper
        .find("    pub fn sweep(\n")
        .expect("strict synchronous sweep route");
    let resident_start = strict_wrapper
        .find("    pub fn sweep_resident_v3(\n")
        .expect("strict resident sweep route");
    let launch_start = strict_wrapper[resident_start..]
        .find("    fn launch_chunk(\n")
        .map(|offset| resident_start + offset)
        .expect("shared strict launch route");
    let sync = &strict_wrapper[sync_start..resident_start];
    let resident = &strict_wrapper[resident_start..launch_start];
    let validation_call = "Self::validate_cci_cycle_periods_v9(kernel, periods, cols)?;";
    let sync_validation = sync.find(validation_call).expect("sync CCI-v9 validation");
    assert!(
        sync_validation
            > sync
                .find("if let Some(max) = kernel.max_period()")
                .expect("sync generic maximum validation")
    );
    assert!(
        sync_validation
            < sync
                .find("DeviceBuffer::<f64>::uninitialized(output_elems)")
                .expect("sync output allocation")
    );
    let resident_validation = resident
        .find(validation_call)
        .expect("resident CCI-v9 validation");
    assert!(
        resident_validation
            > resident
                .find("if let Some(maximum) = kernel.max_period()")
                .expect("resident generic maximum validation")
    );
    assert!(
        resident_validation
            < resident
                .find("let output_id =")
                .expect("resident output-id resolution")
    );
    assert!(
        resident_validation
            < resident
                .find("DeviceBuffer::<f64>::uninitialized_async")
                .expect("resident output allocation")
    );

    assert!(kernel.contains("#define NEO_CCICYC_FACTOR 0.5"));
    let cci_signature_start = kernel
        .find("void cci_cycle_neo_batch_f64(")
        .expect("strict CCI-v9 entry point");
    let cci_signature_end = kernel[cci_signature_start..]
        .find('{')
        .map(|offset| cci_signature_start + offset)
        .expect("strict CCI-v9 entry signature end");
    assert!(!kernel[cci_signature_start..cci_signature_end].contains("factor"));

    let legacy_f32 = source("vendor/vector-ta-0.2.9-patched/src/cuda/cci_cycle_wrapper.rs");
    assert!(legacy_f32.contains("CCI_CYCLE_F32_SEMANTIC_VERSION: u32 = 8"));
    assert!(legacy_f32.contains("cci-cycle-vector-ta-legacy-f32-v8"));
}
