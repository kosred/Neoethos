use std::fs;
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-data"))
}

fn read(relative: impl AsRef<Path>) -> String {
    let path = crate_root().join(relative);
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

fn require_all(source: &str, tokens: &[&str]) {
    for token in tokens {
        assert!(source.contains(token), "missing source token `{token}`");
    }
}

fn require_in_order(source: &str, tokens: &[&str]) {
    let mut cursor = 0_usize;
    for token in tokens {
        let offset = source[cursor..]
            .find(token)
            .unwrap_or_else(|| panic!("missing ordered source token `{token}`"));
        cursor += offset + token.len();
    }
}

#[test]
fn smc_regime_and_footprint_drafts_use_the_existing_owner_authorities() {
    let store = read("src/core/gpu_resident_feature_store_v3.rs");
    require_all(
        &store,
        &[
            "fn resident_smc_producer_draft_v4(",
            "RESIDENT_SMC_COLUMN_NAMES_V3",
            "SMC_SEMANTIC_VERSION",
            "preflight_resident_smc_memory_v4(row_count)",
            "ResidentFeatureProducerV3::Smc",
            "fn resident_regime_producer_draft_v4(",
            "REGIME_FEATURE_NAMES_V3",
            "REGIME_SEMANTIC_VERSION",
            "preflight_resident_regime_memory_v4(row_count)",
            "ResidentFeatureProducerV3::Regime",
            "fn resident_footprint_producer_draft_v4(",
            "FOOTPRINT_FEATURE_NAMES",
            "FOOTPRINT_SEMANTIC_VERSION",
            "preflight_resident_footprint_memory_v4(row_count)",
            "ResidentFeatureProducerV3::Footprint",
        ],
    );
}

#[test]
fn regime_and_footprint_memory_are_sized_by_the_runtime_owners() {
    let regime = read("../neoethos-gpu-cuda/src/resident_regime_v3.rs");
    require_all(
        &regime,
        &[
            "pub struct ResidentRegimePreDeviceMemoryReceiptV4",
            "pub fn preflight_resident_regime_memory_v4(",
            "additional_retained_bytes: 0",
            "scratch_bytes: 0",
        ],
    );
    let footprint = read("../neoethos-gpu-cuda/src/resident_footprint_v2.rs");
    let footprint_preflight = footprint
        .split("pub fn preflight_resident_footprint_memory_v4(")
        .nth(1)
        .and_then(|tail| tail.split("impl ResidentFootprintRuntimeReceiptV2").next())
        .expect("isolate Footprint pre-device memory authority");
    require_all(
        footprint_preflight,
        &[
            ".checked_add(1)",
            ".checked_mul(FOOTPRINT_PREFIX_SERIES_V2)",
            "std::mem::size_of::<f64>()",
        ],
    );
}

#[test]
fn classic_draft_uses_the_same_recipe_and_pre_device_memory_receipt() {
    let classic = read("src/core/gpu_resident_classic_ta_v3.rs");
    require_all(
        &classic,
        &[
            "pub(crate) fn into_resident_feature_recipe_draft_v4(",
            "ResidentClassicTaPreDeviceMemoryReceiptV4",
            "recipe.launches()",
            "memory.launch_plans()",
            "ResidentProducerDraftV4::from_owner_preflight(",
            "ResidentFeatureProducerV3::ClassicTa",
        ],
    );
}

#[test]
fn smc_memory_is_sized_by_the_runtime_owner_before_device_acquisition() {
    let smc = read("../neoethos-gpu-cuda/src/resident_smc_v3.rs");
    require_all(
        &smc,
        &[
            "pub struct ResidentSmcPreDeviceMemoryReceiptV4",
            "pub fn preflight_resident_smc_memory_v4(",
            "checked_parent_bytes_v3(rows)?",
            "GENERATED_PARENT_HASH_BYTES_V3",
            "RESIDENT_SMC_COLUMN_NAMES_V3.len()",
        ],
    );
}

#[test]
fn factory_appends_all_seven_column_drafts_in_canonical_order() {
    let store = read("src/core/gpu_resident_feature_store_v3.rs");
    let resolve = store
        .split_once("fn resolve(")
        .expect("crate-owned factory resolve")
        .1
        .split_once("fn prepare_smc(")
        .expect("factory resolve terminator")
        .0;
    require_in_order(
        resolve,
        &[
            "resident_smc_producer_draft_v4",
            "classic_draft",
            "quant_draft",
            "session_draft",
            "resident_regime_producer_draft_v4",
            "resident_footprint_producer_draft_v4",
            "htf_draft",
        ],
    );
    assert!(resolve.contains("into_materialization_v4()"));
}

#[test]
fn runtime_appends_every_admitted_family_before_normalization() {
    let store = read("src/core/gpu_resident_feature_store_v3.rs");
    let materialize = store
        .split_once("pub fn materialize_gpu_only_feature_store_v3(")
        .expect("strict materializer")
        .1
        .split_once("#[cfg(test)]")
        .expect("strict materializer terminator")
        .0;
    require_in_order(
        materialize,
        &[
            "pending_smc_batch.append_to",
            "append_resident_classic_ta_recipe_v4",
            "quant_runtime.append_to",
            "session_runtime.append_to",
            "regime_input.append_to",
            "append_resident_footprint_v2",
            "prepared_htf_append.append_to",
            "apply_resident_robust_normalization_v2",
        ],
    );
}
