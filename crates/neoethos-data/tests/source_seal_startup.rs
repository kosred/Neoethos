#[test]
fn startup_preflight_is_publicly_reachable_without_exposing_seal_internals() {
    let preflight: fn() -> anyhow::Result<()> =
        neoethos_data::initialize_source_seal_before_runtime;
    let _ = preflight;
}

#[test]
fn source_seal_slot_limit_is_publicly_reachable_and_platform_stable() {
    let slot_limit: fn() -> usize = neoethos_data::source_seal_slot_limit;
    assert_eq!(slot_limit(), 8);
}
