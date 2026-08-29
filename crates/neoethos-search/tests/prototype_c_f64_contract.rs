const HOST_SOURCE: &str = include_str!("../src/gpu_native/prototype_c_engine.rs");
const DEVICE_SOURCE: &str = include_str!("../src/gpu_native/prototype_c_engine/device.rs");

#[test]
fn prototype_c_is_f64_from_host_projection_through_device_readback() {
    for (surface, source) in [("host", HOST_SOURCE), ("device", DEVICE_SOURCE)] {
        assert!(
            !source.contains("f32"),
            "Prototype C {surface} source still contains an f32 precision surface"
        );
    }

    for required in [
        "pub close_pips: Vec<f64>",
        "pub adaptive_base_pips: Vec<f64>",
        "metrics: &[f64]",
    ] {
        assert!(
            HOST_SOURCE.contains(required),
            "Prototype C host projection is missing canonical f64 surface `{required}`"
        );
    }

    for required in [
        "Array<f64>",
        "RuntimeCell::<f64>",
        "f64::as_bytes",
        "f64::from_bytes",
        "size_of::<f64>()",
        "pad_f64",
    ] {
        assert!(
            DEVICE_SOURCE.contains(required),
            "Prototype C device pipeline is missing f64 surface `{required}`"
        );
    }
}
