use std::fs;
use std::path::PathBuf;

fn source(relative: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fs::read_to_string(root.join(relative)).expect("read production source")
}

fn function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .expect("function signature is present");
    let open = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .expect("function has a body");
    let mut depth = 0_usize;
    for (offset, byte) in source.as_bytes()[open..].iter().copied().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("function body is balanced")
}

#[test]
fn gpu_run_plan_preflights_the_complete_admission_before_feature_allocation() {
    let hpc = source("src/core/hpc_ta.rs");
    let prepare = function_body(&hpc, "pub fn prepare_classic_ta_run_plan(");
    assert!(
        prepare.contains("build_classic_ta_admission_plan")
            && prepare.contains("build_exact_classic_cuda_plan")
            && prepare.contains("resolve_gpu_only_classic_plan"),
        "one allocation-free run plan must capture admission and resolve every CUDA route"
    );

    let data = source("src/lib.rs");
    let cube = function_body(
        &data,
        "fn prepare_multitimeframe_features_with_optional_cutoff(",
    );
    assert_eq!(
        cube.matches("prepare_classic_ta_run_plan(").count(),
        1,
        "the multi-timeframe cube must capture exactly one RAM/admission decision"
    );
    let preflight = cube
        .find("prepare_classic_ta_run_plan(")
        .expect("run preflight is present");
    let first_feature = cube
        .find("compute_hpc_feature_frame_sized_with_classic_plan(")
        .expect("feature compute consumes the captured plan");
    assert!(
        preflight < first_feature,
        "the complete CUDA graph must fail before any feature producer runs"
    );
    assert!(
        cube.contains("&classic_run_plan")
            && cube.contains("compute_aligned_higher_block(")
            && !cube.contains("compute_hpc_feature_frame_sized(&base_source"),
        "base and every direct higher timeframe must consume the same run plan"
    );
}

#[test]
fn frame_execution_consumes_the_frozen_admission_instead_of_reprobing_ram() {
    let hpc = source("src/core/hpc_ta.rs");
    let execute = function_body(
        &hpc,
        "pub fn compute_classic_ta_columns_sized_report_with_run_plan(",
    );
    assert!(
        execute.contains("run_plan.admission.clone()")
            && !execute.contains("VocabularyBudget::for_run")
            && !execute.contains("build_classic_ta_admission_plan("),
        "frame execution must reuse the one admitted graph byte-for-byte"
    );

    let cuda = source("src/core/classic_cuda_plan.rs");
    let resolve = function_body(&cuda, "pub(crate) fn resolve_gpu_only_classic_plan(");
    assert!(
        resolve.contains("preflight_exact_classic_cuda_plan")
            && resolve.contains("before the first CUDA context/launch")
            && !resolve.contains("GpuIndicatorEngine::new"),
        "preflight must produce the complete manifest without creating a CUDA engine"
    );
}
