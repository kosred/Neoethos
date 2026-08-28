use std::fs;
use std::path::{Path, PathBuf};

fn crate_dir() -> PathBuf {
    let source = PathBuf::from(file!());
    let source = if source.is_absolute() {
        source
    } else {
        std::env::current_dir()
            .expect("current directory")
            .join(source)
    };
    source
        .parent()
        .and_then(Path::parent)
        .expect("test lives below the gpu-cuda crate")
        .to_path_buf()
}

fn read(relative: &str) -> String {
    let path = crate_dir().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read required source {}: {error}", path.display()))
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let (_, tail) = source
        .split_once(start)
        .unwrap_or_else(|| panic!("missing source boundary {start:?}"));
    tail.split_once(end)
        .unwrap_or_else(|| panic!("missing source boundary {end:?} after {start:?}"))
        .0
}

fn normalized(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

#[test]
fn adaptive_base_shape_check_has_one_warning_clean_condition() {
    let source = read("src/population.rs");
    let validation = section(
        &source,
        "impl PopulationDatasetView<'_> {",
        "pub struct PopulationParentDatasetV1",
    );

    assert!(
        normalized(validation)
            .contains("self.adaptive_base_pips.is_some_and(|base|base.len()!=bars)"),
        "adaptive-base validation must be expressed as one warning-clean condition"
    );
    assert!(
        !validation.contains("if let Some(base) = self.adaptive_base_pips {\n            if"),
        "the nested adaptive-base condition triggers clippy::collapsible_if"
    );
}

#[test]
fn immutable_parent_constructor_accepts_one_typed_input() {
    let population = read("src/population.rs");
    let library = read("src/lib.rs");
    let parent = section(
        &population,
        "pub struct PopulationParentDatasetInputV1",
        "pub enum PopulationViewKindV1",
    );
    assert!(parent.contains("pub fn new(input: PopulationParentDatasetInputV1)"));
    for field in [
        "pub close: Arc<[f64]>",
        "pub high: Arc<[f64]>",
        "pub low: Arc<[f64]>",
        "pub indicators_feature_major: Arc<[f64]>",
        "pub feature_count: usize",
        "pub months: Arc<[i64]>",
        "pub days: Arc<[i64]>",
        "pub timestamps: Arc<[i64]>",
        "pub smc_rows: Arc<[i8]>",
    ] {
        assert!(
            parent.contains(field),
            "typed immutable input omits {field:?}"
        );
    }

    let gpu_tests = read("tests/population_parent_views_v1_contract.rs");
    let search = read("../neoethos-search/src/population_execution_evidence_v1.rs");
    for caller in [&gpu_tests, &search] {
        assert!(caller.contains("PopulationParentDatasetInputV1 {"));
        assert!(caller.contains("PopulationParentDatasetV1::new("));
    }
    assert!(library.contains("PopulationParentDatasetInputV1"));
}

#[test]
fn first_hit_test_uses_a_named_function_pointer_type() {
    let source = read("src/lib.rs");
    let test = section(
        &source,
        "fn first_hit_f64_abi_contract_is_stable()",
        "assert_eq!(size_of::<CudaFirstHitEvent>()",
    );
    assert!(test.contains("type WarpFirstHitFn = fn("));
    assert!(test.contains("let _: WarpFirstHitFn = warp_first_hit;"));
    assert!(
        !test.contains("let _: fn("),
        "the anonymous composite function pointer triggers clippy::type_complexity"
    );
}
