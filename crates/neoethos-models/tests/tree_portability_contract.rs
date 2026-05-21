use neoethos_models::tree_models::{CatBoostExpert, LightGBMExpert, XGBoostExpert};

#[test]
#[cfg(not(feature = "python-onnx-export"))]
fn default_build_keeps_tree_stack_python_free() {
    let python_export_enabled = std::hint::black_box(cfg!(feature = "python-onnx-export"));
    assert!(
        !python_export_enabled,
        "default tree-model build should stay Python-free unless explicitly enabled"
    );
}

#[test]
fn tree_experts_construct_without_python_runtime_requirements() {
    let xgboost = XGBoostExpert::new(1, None);
    let lightgbm = LightGBMExpert::new(2, None);
    let catboost = CatBoostExpert::new(3);

    assert_eq!(xgboost.idx, 1);
    assert_eq!(lightgbm.idx, 2);
    assert_eq!(catboost.idx, 3);
}

#[test]
#[allow(clippy::assertions_on_constants)]
fn compiled_tree_feature_set_is_not_empty() {
    let any_tree_backend = cfg!(feature = "xgboost")
        || cfg!(feature = "lightgbm")
        || cfg!(feature = "catboost")
        || cfg!(feature = "sklears-tree");

    assert!(
        any_tree_backend,
        "neoethos-models should expose at least one compiled tree backend in the default tree stack"
    );
}
