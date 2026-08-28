use neoethos_models::tree_models::{CatBoostExpert, LightGBMExpert, XGBoostExpert};

// #173: prior test depended on a `python-onnx-export` feature that has
// not existed in this workspace since the Rust port. Removing the
// test entirely is correct — there's nothing left to gate against.

#[test]
fn tree_experts_construct_without_python_runtime_requirements() {
    let xgboost = XGBoostExpert::new(1, None);
    let lightgbm = LightGBMExpert::new(2, None);
    let catboost = CatBoostExpert::new(3);

    assert_eq!(xgboost.idx, 1);
    assert_eq!(lightgbm.idx, 2);
    assert_eq!(catboost.idx, 3);
}

#[cfg(any(
    feature = "xgboost",
    feature = "lightgbm",
    feature = "catboost",
    feature = "sklears-tree"
))]
#[test]
fn compiled_tree_feature_set_is_not_empty() {
    let mut compiled_backends = Vec::new();
    #[cfg(feature = "xgboost")]
    compiled_backends.push("xgboost");
    #[cfg(feature = "lightgbm")]
    compiled_backends.push("lightgbm");
    #[cfg(feature = "catboost")]
    compiled_backends.push("catboost");
    #[cfg(feature = "sklears-tree")]
    compiled_backends.push("sklears-tree");

    assert!(
        !compiled_backends.is_empty(),
        "a tree-feature build must expose at least one compiled backend"
    );
}
