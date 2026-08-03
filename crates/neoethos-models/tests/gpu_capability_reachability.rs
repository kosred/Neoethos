//! The general form of the "a CUDA build reports no GPU" bug.
//!
//! `registry.rs` advertised GPU support for the Deep/Exit family behind
//! `burn-wgpu-backend`, for `dqn` behind `reinforcement-learning-cuda`, and for
//! `lightgbm` behind `lightgbm-gpu`. NONE of those three is reachable from any
//! aggregate a shipped front-end can select, so on every real build the answer
//! was a silent `false`. Nothing failed; the report was simply wrong for
//! months.
//!
//! This test closes that class. It reads the feature graph out of
//! `Cargo.toml`, computes what a shipped front-end (`neoethos-cli`,
//! `neoethos-app`, `neoethos-desktop`) can actually turn on, and fails when a
//! model advertises a GPU capability whose implementing feature is not in that
//! set.
//!
//! It is a pure manifest + table check: no GPU, no toolchain, no network.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

use neoethos_models::runtime::capabilities::{KNOWN_MODEL_NAMES, model_capability};
use neoethos_models::runtime::gpu_capability::declared_gpu_path;

const THIS_CRATE: &str = "neoethos-models";

fn workspace_root() -> PathBuf {
    // <workspace>/crates/neoethos-models
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/neoethos-models sits two levels below the workspace root")
        .to_path_buf()
}

/// Strip `#` comments and collapse a manifest to one logical line per entry so
/// multi-line feature arrays parse with a flat scanner.
fn strip_comments(manifest: &str) -> String {
    manifest
        .lines()
        .map(|line| match line.find('#') {
            // A `#` inside a quoted string never occurs in the feature blocks
            // we read, so the naive split is safe here.
            Some(index) => &line[..index],
            None => line,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// `name = ["a", "b/c", "dep:d"]`, possibly spanning several lines, for every
/// entry of the `[features]` table.
fn parse_features(manifest: &str) -> HashMap<String, Vec<String>> {
    let cleaned = strip_comments(manifest);
    let Some(start) = cleaned.find("\n[features]") else {
        return HashMap::new();
    };
    let body = &cleaned[start + "\n[features]".len()..];
    // Stop at the next table header.
    let body = match body.find("\n[") {
        Some(end) => &body[..end],
        None => body,
    };

    let mut features = HashMap::new();
    let mut current: Option<(String, String)> = None;
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match current.as_mut() {
            Some((_, buffer)) => buffer.push_str(line),
            None => {
                let Some((name, rest)) = line.split_once('=') else {
                    continue;
                };
                current = Some((name.trim().to_string(), rest.trim().to_string()));
            }
        }
        let complete = current
            .as_ref()
            .is_some_and(|(_, buffer)| buffer.contains(']'));
        if complete {
            let (name, buffer) = current.take().expect("checked above");
            let inner = buffer
                .trim_start_matches('[')
                .trim_end_matches(']')
                .to_string();
            let deps = inner
                .split(',')
                .map(|entry| entry.trim().trim_matches('"').to_string())
                .filter(|entry| !entry.is_empty())
                .collect::<Vec<_>>();
            features.insert(name, deps);
        }
    }
    features
}

/// Every `neoethos-models` feature a shipped front-end can select, whether by
/// naming it as `neoethos-models/<feature>` in its own aggregate or by listing
/// it in the dependency declaration.
fn front_end_selected_features(root: &Path) -> BTreeSet<String> {
    let manifests = [
        root.join("crates/neoethos-cli/Cargo.toml"),
        root.join("crates/neoethos-app/Cargo.toml"),
        root.join("desktop/src-tauri/Cargo.toml"),
        root.join("crates/neoethos-trader/Cargo.toml"),
    ];

    let mut selected = BTreeSet::new();
    // `default-features` is never disabled for this dependency, so the default
    // aggregate is always on.
    selected.insert("default".to_string());

    let needle = format!("{THIS_CRATE}/");
    for manifest_path in manifests {
        let Ok(manifest) = std::fs::read_to_string(&manifest_path) else {
            continue;
        };
        let cleaned = strip_comments(&manifest);

        // `"neoethos-models/gpu-cuda"` inside a front-end aggregate.
        let mut cursor = 0usize;
        while let Some(offset) = cleaned[cursor..].find(&needle) {
            let start = cursor + offset + needle.len();
            let end = cleaned[start..]
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '-' || c == '_'))
                .map(|index| start + index)
                .unwrap_or(cleaned.len());
            if end > start {
                selected.insert(cleaned[start..end].to_string());
            }
            cursor = start.max(cursor + offset + 1);
        }

        // `neoethos-models = { path = "…", features = ["tree-models", …] }`
        for line in cleaned.lines() {
            let trimmed = line.trim();
            if !trimmed.starts_with(THIS_CRATE) {
                continue;
            }
            let Some(features_at) = trimmed.find("features") else {
                continue;
            };
            let tail = &trimmed[features_at..];
            let Some(open) = tail.find('[') else { continue };
            let Some(close) = tail.find(']') else { continue };
            for entry in tail[open + 1..close].split(',') {
                let entry = entry.trim().trim_matches('"');
                if !entry.is_empty() {
                    selected.insert(entry.to_string());
                }
            }
        }
    }
    selected
}

/// Transitive closure of `roots` over this crate's own feature graph.
///
/// `dep:x` and `other-crate/x` entries are not features of this crate, so they
/// terminate the walk — except `neoethos-models/x`, which cannot appear.
fn feature_closure(graph: &HashMap<String, Vec<String>>, roots: &BTreeSet<String>) -> HashSet<String> {
    let mut reached = HashSet::new();
    let mut queue = roots.iter().cloned().collect::<Vec<_>>();
    while let Some(feature) = queue.pop() {
        if !reached.insert(feature.clone()) {
            continue;
        }
        let Some(deps) = graph.get(&feature) else {
            continue;
        };
        for dep in deps {
            if dep.starts_with("dep:") {
                continue;
            }
            let name = dep.split('?').next().unwrap_or(dep);
            if name.contains('/') {
                continue;
            }
            queue.push(name.to_string());
        }
    }
    reached
}

#[test]
fn every_advertised_gpu_capability_is_reachable_from_a_shipped_build() {
    let root = workspace_root();
    let manifest = std::fs::read_to_string(root.join("crates/neoethos-models/Cargo.toml"))
        .expect("neoethos-models manifest must be readable");
    let graph = parse_features(&manifest);
    assert!(
        graph.contains_key("gpu-cuda"),
        "feature parser failed: `gpu-cuda` not found in the [features] table"
    );

    let roots = front_end_selected_features(&root);
    assert!(
        roots.contains("gpu-cuda"),
        "no shipped front-end selects `neoethos-models/gpu-cuda`; parsed roots: {roots:?}"
    );
    let reachable = feature_closure(&graph, &roots);

    let mut unreachable = Vec::new();
    for name in KNOWN_MODEL_NAMES {
        let Some(capability) = model_capability(name) else {
            continue;
        };
        let Some(spec) = declared_gpu_path(name, capability.family) else {
            continue;
        };
        // The table may offer alternatives (Burn compiles against either the
        // CUDA or the wgpu backend). One reachable alternative is enough.
        let any_reachable = spec
            .features
            .iter()
            .any(|feature| reachable.contains(*feature));
        if !any_reachable {
            unreachable.push(format!(
                "  {name}: advertises GPU via {:?}, none reachable from any shipped aggregate",
                spec.features
            ));
        }
    }

    assert!(
        unreachable.is_empty(),
        "models advertise GPU capabilities that no shipped build can compile.\n\
         This is the `lightgbm-gpu` / `reinforcement-learning-cuda` / `burn-wgpu-backend`\n\
         bug: the capability report keys on a feature nothing enables, so it silently\n\
         answers `false` forever. Either wire the feature into an aggregate, or stop\n\
         advertising it in `runtime::gpu_capability::declared_gpu_path`.\n{}",
        unreachable.join("\n")
    );
}

#[test]
fn no_orphan_gpu_features_remain_in_the_manifest() {
    // The other half of the same defect: a feature that names a GPU capability,
    // is defined, and is enabled by nothing. `lightgbm-gpu`,
    // `reinforcement-learning-cuda`, `neuro-evolution-gpu` and `statistical-gpu`
    // were all in this state. A defined-but-unreachable GPU feature is either a
    // capability that should be wired in or a name that should be deleted;
    // leaving it is how two ways to express one capability appear.
    let root = workspace_root();
    let manifest = std::fs::read_to_string(root.join("crates/neoethos-models/Cargo.toml"))
        .expect("neoethos-models manifest must be readable");
    let graph = parse_features(&manifest);
    let roots = front_end_selected_features(&root);
    let reachable = feature_closure(&graph, &roots);

    // Deliberate command-line-only opt-ins. Each entry needs a reason and a
    // reachable alternative, or it belongs in an aggregate.
    //
    // `burn-cuda-backend`: measured pathological for our tiny neural models
    // (2026-06-10, A6000 — 338 cubecl autotune passes against 14 real epochs),
    // so `gpu-cuda` deliberately leaves it off and the Deep/Exit family's
    // reachable GPU path is `burn-wgpu-backend` via `gpu-vulkan`. Built with
    // `--features gpu-cuda,burn-cuda-backend` when the models get big enough.
    const DELIBERATE_OPT_INS: &[&str] = &["burn-cuda-backend"];

    let orphans = graph
        .keys()
        .filter(|feature| {
            let lower = feature.to_ascii_lowercase();
            lower.contains("gpu") || lower.contains("cuda")
        })
        .filter(|feature| !reachable.contains(*feature))
        .filter(|feature| !DELIBERATE_OPT_INS.contains(&feature.as_str()))
        // The vendor aggregates themselves are roots, so anything left here is
        // a leaf capability nothing turns on.
        .cloned()
        .collect::<BTreeSet<_>>();

    assert!(
        orphans.is_empty(),
        "GPU features defined but enabled by nothing: {orphans:?}\n\
         Wire each into the aggregate that should provide it, or delete the name."
    );
}
