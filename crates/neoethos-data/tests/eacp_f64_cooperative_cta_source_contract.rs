const EACP_CUDA: &str = include_str!(
    "../../../vendor/vector-ta-0.2.9-patched/kernels/cuda/ehlers_autocorrelation_periodogram_kernel.cu"
);
const F64_WRAPPER: &str =
    include_str!("../../../vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs");
const CLASSIC_PLAN: &str = include_str!("../src/core/classic_cuda_plan.rs");

const EACP_COOPERATIVE_CTA_SOURCE_CLOSURE: &str =
    "eacp/strict-cuda-exact-cooperative-cta/fmad-off/v1";

fn function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let signature_start = source
        .find(signature)
        .unwrap_or_else(|| panic!("missing source signature `{signature}`"));
    let open = source[signature_start..]
        .find('{')
        .map(|offset| signature_start + offset)
        .unwrap_or_else(|| panic!("missing body for `{signature}`"));

    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[open..].iter().copied().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[open..=open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated body for `{signature}`");
}

fn assert_before(source: &str, first: &str, second: &str) {
    let first_offset = source
        .find(first)
        .unwrap_or_else(|| panic!("missing first source token `{first}`"));
    let second_offset = source
        .find(second)
        .unwrap_or_else(|| panic!("missing second source token `{second}`"));
    assert!(
        first_offset < second_offset,
        "expected `{first}` before `{second}`"
    );
}

#[test]
fn strict_eacp_maps_one_cooperative_cta_to_each_parameter_tuple() {
    let entry = function_body(
        EACP_CUDA,
        "extern \"C\" __global__ void ehlers_autocorrelation_periodogram_outputs_f64",
    );
    assert!(entry.contains("const int row = static_cast<int>(blockIdx.x);"));
    assert!(!entry.contains("blockIdx.x * blockDim.x + threadIdx.x"));
    assert!(entry.contains("ehlers_autocorrelation_periodogram_exact_row_f64("));

    let helper = function_body(
        EACP_CUDA,
        "ehlers_autocorrelation_periodogram_exact_row_f64",
    );
    assert!(helper.contains("const int lane = static_cast<int>(threadIdx.x);"));
    assert!(helper.contains("const int block_width = static_cast<int>(blockDim.x);"));
    assert!(helper.contains("__shared__ PeriodogramState state;"));
    assert!(helper.matches("__syncthreads();").count() >= 6);
}

#[test]
fn legacy_eacp_exports_and_f32_isolation_are_unchanged() {
    for (signature, row_mapping) in [
        (
            "extern \"C\" __global__ void ehlers_autocorrelation_periodogram_batch_f64",
            "const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);",
        ),
        (
            "extern \"C\" __global__ void ehlers_autocorrelation_periodogram_neo_batch_f64",
            "const int row_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);",
        ),
    ] {
        let body = function_body(EACP_CUDA, signature);
        assert!(body.contains(row_mapping));
    }
    assert_eq!(
        EACP_CUDA
            .matches("ehlers_autocorrelation_periodogram_outputs_f64(")
            .count(),
        1
    );
    assert!(!EACP_CUDA.contains("ehlers_autocorrelation_periodogram_outputs_f32"));
}

#[test]
fn cooperative_workers_preserve_each_scalar_reduction_in_ascending_order() {
    let helper = function_body(
        EACP_CUDA,
        "ehlers_autocorrelation_periodogram_exact_row_f64",
    );

    assert!(
        helper.contains("for (int lag = 2 + lane; lag <= state.max_period; lag += block_width)")
    );
    assert!(helper.contains(
        "for (int period = state.min_period + lane; period <= state.max_period; period += block_width)"
    ));
    assert!(helper.contains("for (int k = 0; k < window; ++k)"));
    assert!(helper.contains("for (int n = 2; n <= state.max_period; ++n)"));
    assert!(helper.contains("avg3_sx = avg3_x0 + avg3_x1 + avg3_x2"));
    assert!(
        helper.contains("avg3_sxx = avg3_x0 * avg3_x0 + avg3_x1 * avg3_x1 + avg3_x2 * avg3_x2")
    );
    assert!(
        helper
            .contains("for (int period = state.min_period; period <= state.max_period; ++period)")
    );

    assert_before(helper, "state.corr[lag] =", "double local_max_pwr = 0.0;");
    assert_before(
        helper,
        "double local_max_pwr = 0.0;",
        "double weighted = 0.0;",
    );
    assert_before(
        helper,
        "double weighted = 0.0;",
        "out_dominant_cycle[i] = state.dom;",
    );
}

#[test]
fn strict_wrapper_launches_exactly_one_256_thread_block_per_tuple() {
    let method = function_body(
        F64_WRAPPER,
        "pub fn ehlers_autocorrelation_periodogram_all_outputs",
    );
    assert!(method.contains("let grid = GridSize::x(rows_u32);"));
    assert!(method.contains("let block = BlockSize::x(BAR_BLOCK_X);"));
    assert!(!method.contains("rows_u32.div_ceil(BLOCK_X)"));

    assert!(
        CLASSIC_PLAN
            .contains("let parameter_rows = [(min_period, max_period, avg_length, enhance)];")
    );
}

#[test]
fn source_closure_names_the_exact_math_and_cooperative_schedule_boundary() {
    assert_eq!(
        EACP_COOPERATIVE_CTA_SOURCE_CLOSURE,
        "eacp/strict-cuda-exact-cooperative-cta/fmad-off/v1"
    );
    assert!(EACP_CUDA.contains(EACP_COOPERATIVE_CTA_SOURCE_CLOSURE));
    assert!(F64_WRAPPER.contains("vector-ta.f64.native-sass.no-fast-math.no-fmad.v3"));
}
