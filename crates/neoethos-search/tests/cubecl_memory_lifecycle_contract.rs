use std::fs;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("neoethos-search must live under <repo>/crates")
        .to_path_buf()
}

fn read_repo(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("could not read repository source {relative}: {error}"))
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

#[test]
fn patched_cubecl_cleanup_releases_gpu_and_pinned_cpu_pools_after_draining_drops() {
    let root_manifest = read_repo("Cargo.toml");
    assert!(
        root_manifest.contains("cubecl-cuda = { path = \"vendor/cubecl-cuda-0.10.0-patched\" }"),
        "the CubeCL CUDA lifecycle repair must be a reproducible workspace patch, not a local \n\
         Cargo registry edit"
    );

    let command = read_repo("vendor/cubecl-cuda-0.10.0-patched/src/compute/command.rs");
    let cleanup = function_body(&command, "pub fn memory_cleanup(&mut self)");
    assert_eq!(
        cleanup.matches("drop_queue.flush").count(),
        2,
        "CubeCL's double-buffered pending-drop queue must be drained before pool cleanup"
    );
    assert!(
        cleanup.contains("memory_management_gpu.cleanup(true)")
            && cleanup.contains("memory_management_cpu.cleanup(true)"),
        "explicit cleanup must release both device pages and pinned-host staging pages"
    );
}

#[test]
fn cubecl_residency_is_scoped_and_the_last_outer_scope_synchronously_cleans_every_device() {
    let cubecl = read_repo("crates/neoethos-search/src/cubecl_eval.rs");
    assert!(
        cubecl.contains("struct CubeClResidencyScope"),
        "CubeCL residency needs an explicit outer-run RAII lifetime"
    );
    let enter = function_body(&cubecl, "fn cubecl_residency_scope()");
    assert!(
        enter.contains("active_residency_scopes"),
        "nested CubeCL evaluations must share one outer residency lifetime"
    );
    let drop_impl = function_body(&cubecl, "fn drop(&mut self)");
    assert!(
        drop_impl.contains("resident_device_cache::clear()")
            && drop_impl.contains("release_cubecl_devices"),
        "the last CubeCL scope must drop resident handles before releasing runtime pools"
    );
    let release = function_body(&cubecl, "fn release_cubecl_devices(");
    assert!(
        release.matches("client.sync()").count() >= 2
            && release.contains("client.memory_cleanup()"),
        "device cleanup must synchronize before and after the explicit pool release"
    );
    let cache_clear = function_body(&cubecl, "pub(super) fn clear()");
    assert!(
        cache_clear.contains("map.clear()")
            && cache_clear.contains("order.clear()")
            && cache_clear.contains("total_bytes = 0"),
        "resident input handles must all be dropped at the outer boundary"
    );
    let client = function_body(&cubecl, "fn create_gpu_client(");
    assert!(
        client.contains("record_cubecl_device"),
        "every selected CUDA ordinal must be recorded for exact cleanup"
    );
    for signature in [
        "pub(crate) fn try_evaluate_population_cuda(",
        "pub(crate) fn try_evaluate_ftmo_population_cuda(",
    ] {
        let body = function_body(&cubecl, signature);
        assert!(
            body.contains("cubecl_residency_scope()"),
            "{signature} must self-clean when called without an outer discovery scope"
        );
    }

    let discovery = read_repo("crates/neoethos-search/src/discovery.rs");
    for signature in [
        "pub fn run_discovery_cycle_with_holdout_and_progress<F>(",
        "pub fn run_discovery_cycle_with_progress<F>(",
    ] {
        let body = function_body(&discovery, signature);
        assert!(
            body.contains("cubecl_residency_scope()"),
            "{signature} must retain CubeCL residency for the run and release it on every exit"
        );
    }
}

#[test]
fn persistent_prototype_engines_own_a_scope_until_their_device_resources_drop() {
    for relative in [
        "crates/neoethos-search/src/gpu_native/prototype_a_engine.rs",
        "crates/neoethos-search/src/gpu_native/prototype_c_engine/device.rs",
    ] {
        let source = read_repo(relative);
        assert!(
            source.contains("_residency_scope: crate::cubecl_eval::CubeClResidencyScope"),
            "{relative} must retain an explicit CubeCL scope for the full persistent-engine lifetime"
        );
        assert!(
            source.contains("let residency_scope = crate::cubecl_eval::cubecl_residency_scope();"),
            "{relative} must enter residency before it creates the client"
        );
        assert!(
            source.contains("_residency_scope: residency_scope"),
            "{relative} must move the scope into the returned engine"
        );
    }
}
