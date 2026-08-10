// Integration test for neoethos-models crate
//
// 2026-08-09 (batch D4): the registry-catalogue tests (`list_models_by_category`,
// `get_model_info`, `is_valid_model`) and the `tch`-gated hardware test were
// removed with the code they exercised. They were the ONLY callers of that API,
// which is what made it dead: a test suite is not a consumer. The `tch` test
// additionally referenced `logical_cores` / `ram_gb` / `gpu_list`, fields
// `HardwareInfo` has never had — it could not have compiled even with the
// feature on.
//
// The surviving capability surface is covered by the unit tests in
// `src/registry.rs` and `src/hardware.rs`.

#[test]
fn test_compilation() {
    // This test just verifies that all modules compile
    println!("✓ All neoethos-models modules compiled successfully");
}

#[test]
fn capability_lookup_is_reachable_from_outside_the_crate() {
    // `get_model_capability` is the one live export of `registry`; it is on the
    // train path via `runtime::dispatch::build_dispatch_plan`. Pin that it stays
    // publicly reachable.
    let capability =
        neoethos_models::registry::get_model_capability("lightgbm").expect("lightgbm capability");
    assert_eq!(capability.name, "lightgbm");
    assert!(neoethos_models::registry::get_model_capability("nonexistent").is_none());
}
