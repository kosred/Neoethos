use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

const UPSTREAM_0_1_2_VCS_SHA1: &str = "53f7569c9c566046ea05eae2a945d67dd1c04b68";

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root must resolve")
}

fn read_workspace(relative: &str) -> String {
    let path = workspace_root().join(relative);
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn compact(source: &str) -> String {
    source.split_whitespace().collect()
}

fn package_block<'a>(lock: &'a str, package_name: &str) -> &'a str {
    let name_line = format!("name = \"{package_name}\"");
    lock.split("[[package]]")
        .find(|block| block.lines().any(|line| line.trim() == name_line))
        .unwrap_or_else(|| panic!("Cargo.lock must contain package {package_name}"))
}

fn winapi_features(manifest: &str) -> BTreeSet<&str> {
    let section_header = "[target.'cfg(target_os = \"windows\")'.dependencies.winapi]";
    let section = manifest
        .split_once(section_header)
        .expect("sklears-core must declare its Windows winapi dependency")
        .1
        .split("\n[")
        .next()
        .expect("Windows winapi section must terminate");
    let features = section
        .split_once("features = [")
        .expect("Windows winapi dependency must declare explicit features")
        .1
        .split_once(']')
        .expect("Windows winapi feature list must terminate")
        .0;

    features
        .lines()
        .filter_map(|line| {
            let feature = line.trim().trim_end_matches(',').trim_matches('"');
            (!feature.is_empty()).then_some(feature)
        })
        .collect()
}

fn cfg_method_body<'a>(source: &'a str, cfg_marker: &str) -> &'a str {
    let start = source
        .find(cfg_marker)
        .unwrap_or_else(|| panic!("missing cfg marker {cfg_marker}"));
    let after_marker = &source[start + cfg_marker.len()..];
    let end = after_marker.find("#[cfg(").unwrap_or(after_marker.len());
    &after_marker[..end]
}

#[test]
fn vendored_sklears_core_is_the_active_locked_0_1_2_package_surface() {
    let vendor_manifest = read_workspace("vendor/sklears-core/Cargo.toml");
    assert!(
        vendor_manifest.contains("name = \"sklears-core\"")
            && vendor_manifest.contains("version = \"0.1.2\""),
        "the path patch must vendor the actual selected sklears-core 0.1.2 source"
    );

    let vcs = read_workspace("vendor/sklears-core/.cargo_vcs_info.json");
    assert!(
        vcs.contains(UPSTREAM_0_1_2_VCS_SHA1),
        "the vendor must preserve upstream 0.1.2 provenance"
    );

    let system_info = read_workspace("vendor/sklears-core/src/system_info.rs");
    for public_item in [
        "pub struct SystemMemory",
        "pub fn system_memory()",
        "pub fn process_rss_bytes()",
    ] {
        assert!(
            system_info.contains(public_item),
            "0.1.2 public API is missing {public_item}"
        );
    }
    let lib = read_workspace("vendor/sklears-core/src/lib.rs");
    assert!(lib.contains("pub mod system_info;"));
    assert!(
        lib.contains(
            "pub use crate::system_info::{process_rss_bytes, system_memory, SystemMemory};"
        )
    );

    let root_manifest = read_workspace("Cargo.toml");
    assert!(root_manifest.contains("sklears-core = { path = \"vendor/sklears-core\" }"));

    let lock = read_workspace("Cargo.lock").replace("\r\n", "\n");
    let package = package_block(&lock, "sklears-core");
    assert!(package.contains("version = \"0.1.2\""));
    assert!(
        !package.contains("source = ") && !package.contains("checksum = "),
        "the selected sklears-core package must be the local path patch"
    );
    assert!(
        !lock
            .split("[[patch.unused]]")
            .skip(1)
            .any(|block| block.contains("name = \"sklears-core\"")),
        "the sklears-core path patch must not remain unused"
    );
}

#[test]
fn vendored_sklears_core_declares_the_complete_windows_api_feature_closure() {
    let vendor_manifest = read_workspace("vendor/sklears-core/Cargo.toml");
    let actual = winapi_features(&vendor_manifest);
    let expected = BTreeSet::from([
        "basetsd",
        "handleapi",
        "memoryapi",
        "minwinbase",
        "minwindef",
        "processthreadsapi",
        "psapi",
        "sysinfoapi",
        "winnt",
    ]);

    assert_eq!(
        actual, expected,
        "sklears-core must not rely on unrelated crates to activate winapi modules it imports"
    );
}

#[test]
fn vendored_sklears_core_keeps_x86_64_perf_counters_on_linux() {
    let source = compact(&read_workspace("vendor/sklears-core/src/benchmarking.rs"));
    let marker = "#[cfg(all(target_arch=\"x86_64\",target_os=\"linux\"))]";
    let body = cfg_method_body(&source, marker);

    assert!(body.contains("fnread_cache_counters(&self)->CacheStats"));
    assert!(
        body.contains("self.read_perf_counters().unwrap_or(CacheStats{"),
        "Linux x86_64 must preserve its platform counter path"
    );
}

#[test]
fn vendored_sklears_core_uses_an_explicit_zero_fallback_off_linux_x86_64() {
    let source = compact(&read_workspace("vendor/sklears-core/src/benchmarking.rs"));
    let marker = "#[cfg(all(target_arch=\"x86_64\",not(target_os=\"linux\")))]";
    let body = cfg_method_body(&source, marker);

    assert!(body.contains("fnread_cache_counters(&self)->CacheStats"));
    assert!(
        !body.contains("read_perf_counters"),
        "non-Linux x86_64 cannot call the Linux-only counter method"
    );
    for zero_field in [
        "l1_hits:0",
        "l1_misses:0",
        "l2_hits:0",
        "l2_misses:0",
        "l3_hits:0",
        "l3_misses:0",
        "branch_mispredictions:0",
        "tlb_misses:0",
    ] {
        assert!(
            body.contains(zero_field),
            "portable fallback must explicitly mark {zero_field}"
        );
    }
}
