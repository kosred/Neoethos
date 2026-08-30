use std::fs;
use std::path::{Path, PathBuf};

fn manifest_dir() -> PathBuf {
    if let Some(manifest_dir) = option_env!("CARGO_MANIFEST_DIR") {
        return PathBuf::from(manifest_dir);
    }
    let cwd = std::env::current_dir().expect("current directory");
    if cwd.ends_with(Path::new("crates").join("neoethos-data")) {
        cwd
    } else {
        cwd.join("crates").join("neoethos-data")
    }
}

fn read(relative: &str) -> String {
    fs::read_to_string(manifest_dir().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

#[test]
fn exact_cpu_reference_owns_typed_quant_v4_without_rebinding_ordinary_quant_v2() {
    let registry = read("src/core/feature_registry.rs");
    let library = read("src/lib.rs");
    let quant = read("src/core/quant_features.rs");

    assert!(
        registry.contains("pub fn quantitative_feature_producer_manifest_v4"),
        "the corrected exact CPU reference needs a distinct Quant-v4 manifest"
    );
    let v4_manifest = registry
        .split("fn build_quantitative_feature_producer_manifest_v4")
        .nth(1)
        .expect("Quant-v3 manifest builder");
    assert!(
        v4_manifest.contains("ProductionFeatureProducerId::Quantitative")
            && v4_manifest.contains("FeatureSource::Quantitative")
            && v4_manifest.contains("\n        4,"),
        "the distinct quantitative manifest must be semantic-v4"
    );
    for path in [
        "crates/neoethos-data/src/core/quant_features.rs",
        "crates/neoethos-data/src/core/quant_exact_math_v3.rs",
        "crates/neoethos-data/src/core/gpu_resident_temporal_grid_v1.rs",
        "crates/neoethos-data/src/core/gpu_resident_quant_v3.rs",
        "crates/neoethos-data/src/core/timestamps.rs",
        "crates/neoethos-dataset-contracts/src/temporal.rs",
        "crates/neoethos-data/src/lib.rs",
    ] {
        assert!(
            v4_manifest.contains(path),
            "Quant-v3 manifest omitted `{path}`"
        );
    }

    let ordinary_quant = registry
        .split("producer_row(\n            ProductionFeatureProducerId::Quantitative,")
        .nth(1)
        .expect("ordinary Quant manifest row");
    assert!(
        ordinary_quant.contains("FeatureSource::Quantitative")
            && ordinary_quant.contains("\n            2,"),
        "ordinary production/Full must retain Quant-v2"
    );

    assert!(
        library.contains("enum MultiTimeframeFeatureMathAuthorityV3"),
        "one typed authority must govern both Classic planning and whole-feature math"
    );
    assert!(
        library.contains("compute_quant_feature_columns_v4_f64(\n                    source.ohlcv(),\n                    source.artifact().frame_timeframe(),"),
        "exact Quant-v4 must consume the typed timeframe from each canonical frame"
    );
    assert!(
        quant.contains("CumulativeDeltaValidityDependency::Prefix")
            && quant.contains("CumulativeDeltaValidityDependency::RollingWindow")
            && quant.contains("pub fn compute_quant_feature_columns_v4_f64"),
        "semantic-v2 must retain prefix validity while semantic-v4 owns rolling validity"
    );
    assert!(
        library.contains("MultiTimeframeFeatureMathAuthorityV3::CurrentProcessPolicy"),
        "ordinary production must retain an explicit current-policy authority"
    );
    assert!(
        library
            .contains("MultiTimeframeFeatureMathAuthorityV3::ResidentGpuExactParityCpuReferenceV3"),
        "the exact reference must select its distinct whole-feature authority"
    );
    assert!(
        library.contains("build_multitimeframe_feature_contract(\n        &contract_blocks,\n        feature_math_authority,"),
        "the selected authority must also bind the emitted multi-timeframe plan"
    );
}
