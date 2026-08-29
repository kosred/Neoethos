use std::{fs, path::PathBuf};

const MANIFEST: &str = include_str!("../Cargo.toml");
const LIB: &str = include_str!("../src/lib.rs");
const NEAT_GPU: &str = include_str!("../src/evolution/neat_gpu.rs");
const NEAT_IMPL: &str = include_str!("../src/evolution/neat_impl.rs");
const CRFMNES_GPU: &str = include_str!("../src/evolution/crfmnes_gpu.rs");
const CRFMNES_IMPL: &str = include_str!("../src/evolution/crfmnes_impl.rs");
const LINEAR_GPU: &str = include_str!("../src/statistical/linear_gpu.rs");
const PATCHED_CUDA_COMMAND: &str =
    include_str!("../../../vendor/cubecl-cuda-0.10.0-patched/src/compute/command.rs");

fn lifecycle_source() -> String {
    fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("cubecl_lifecycle.rs"),
    )
    .unwrap_or_default()
}

fn function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("missing function signature `{signature}`"));
    let body_start = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .expect("function must have a body");
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[body_start..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[body_start..=body_start + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated function `{signature}`");
}

fn manifest_feature_block<'a>(manifest: &'a str, feature: &str) -> &'a str {
    let assignment = format!("{feature} =");
    let start = manifest
        .find(&assignment)
        .unwrap_or_else(|| panic!("missing {feature} feature"));
    let rest = &manifest[start..];
    let end = rest
        .find(']')
        .map(|offset| offset + 1)
        .unwrap_or_else(|| panic!("unterminated {feature} feature"));
    &rest[..end]
}

#[test]
fn model_cubecl_calls_share_an_exact_stream_and_ordinal_lifecycle() {
    let lifecycle = lifecycle_source();
    assert!(
        MANIFEST.contains("cubecl-common = { version = \"=0.10.0\", optional = true }"),
        "the model lifecycle needs CubeCL's exact StreamId type as a direct dependency"
    );
    for feature in ["neuro-evolution-gpu", "statistical-gpu"] {
        let block = manifest_feature_block(MANIFEST, feature);
        assert!(
            block.contains("dep:cubecl-common"),
            "{feature} must enable exact CubeCL stream tracking"
        );
    }
    assert!(
        LIB.contains("mod cubecl_lifecycle;"),
        "the models crate must compile the shared CubeCL lifecycle module"
    );
    assert!(
        lifecycle.contains("struct CubeClResidencyScope")
            && lifecycle.contains("HashMap<(StreamId, usize), ComputeClient<CudaRuntime>>"),
        "model CubeCL residency must track every exact stream and CUDA ordinal"
    );

    let record = function_body(&lifecycle, "fn record_cubecl_device(");
    assert!(
        record.contains("StreamId::current()") && record.contains("set_stream(stream)"),
        "cleanup clients must be pinned to the exact stream that allocated each pool"
    );
    let release = function_body(&lifecycle, "fn release_cubecl_devices(");
    assert!(
        release.matches("client.sync()").count() >= 2
            && release.contains("client.memory_cleanup()"),
        "final model cleanup must synchronize before and after releasing CubeCL pools"
    );
    let patched_cleanup = function_body(PATCHED_CUDA_COMMAND, "pub fn memory_cleanup(&mut self)");
    assert!(
        patched_cleanup.contains("memory_management_gpu.cleanup(true)")
            && patched_cleanup.contains("memory_management_cpu.cleanup(true)"),
        "the selected CubeCL cleanup must release device and pinned-host pools"
    );
}

#[test]
fn direct_model_kernels_self_clean_and_evolution_hot_loops_keep_residency() {
    for (source, signature) in [
        (NEAT_GPU, "fn try_population_scores_cuda("),
        (CRFMNES_GPU, "fn try_selection_losses_cuda("),
        (LINEAR_GPU, "fn try_fit_linear_softmax_cuda("),
        (LINEAR_GPU, "fn try_predict_linear_softmax_cuda("),
    ] {
        let body = function_body(source, signature);
        assert!(
            body.contains("cubecl_residency_scope()") && body.contains("cubecl_cuda_client("),
            "{signature} must self-scope and register its exact CUDA client"
        );
    }

    let neat = function_body(NEAT_IMPL, "fn evolve_population(");
    let neat_scope = neat
        .find("cubecl_training_residency_scope")
        .expect("NEAT training needs an outer CubeCL residency scope");
    let neat_loop = neat
        .find("for _generation in 0..self.generations")
        .expect("NEAT generation loop must remain visible");
    assert!(
        neat_scope < neat_loop,
        "NEAT must enter residency before its generation loop"
    );

    let crfmnes = function_body(
        CRFMNES_IMPL,
        "fn fit(&mut self, x: &FeatureFrame, y: &[i32], lease: &CpuLease)",
    );
    let crfmnes_scope = crfmnes
        .find("cubecl_training_residency_scope")
        .expect("CR-FM-NES training needs an outer CubeCL residency scope");
    let crfmnes_loop = crfmnes
        .find("for _ in 0..self.islands.max(1)")
        .expect("CR-FM-NES island loop must remain visible");
    assert!(
        crfmnes_scope < crfmnes_loop,
        "CR-FM-NES must enter residency before its island/generation loops"
    );
}
