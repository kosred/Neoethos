const F64_WRAPPER: &str = include_str!("../src/cuda/neoethos_f64_wrapper.rs");
const F64_REGISTRY: &str = include_str!("../src/indicators/dispatch/cuda_f64.rs");

fn function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let signature_start = source
        .find(signature)
        .unwrap_or_else(|| panic!("missing function signature {signature}"));
    let open = source[signature_start..]
        .find('{')
        .map(|offset| signature_start + offset)
        .unwrap_or_else(|| panic!("missing body for {signature}"));

    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[open..].iter().copied().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[open..=open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated body for {signature}");
}

#[test]
fn swept_window_kernels_are_not_mislabeled_period_invariant() {
    let invariant = function_body(F64_WRAPPER, "pub fn is_period_invariant(self) -> bool");

    for kernel in [
        "F64Kernel::Adosc",
        "F64Kernel::Ao",
        "F64Kernel::Apo",
        "F64Kernel::AtrPercentile",
        "F64Kernel::CciCycle",
        "F64Kernel::GarmanKlassVolatility",
    ] {
        assert!(
            !invariant.contains(kernel),
            "{kernel} consumes the requested anchor and must not be period-invariant"
        );
    }
}

#[test]
fn cci_cycle_host_bound_matches_the_compiled_device_ring() {
    let max_period = function_body(F64_WRAPPER, "pub fn max_period(self) -> Option<usize>");

    assert!(F64_WRAPPER.contains("pub const CCI_CYCLE_MAX_LENGTH: usize = 200;"));
    assert!(max_period.contains("F64Kernel::CciCycle => Some(CCI_CYCLE_MAX_LENGTH)"));
}

#[test]
fn registry_comments_describe_anchor_routing_instead_of_duplicate_columns() {
    assert!(!F64_REGISTRY.contains(
        "PERIOD-INVARIANT ids in this block (their CPU batch arm never reads the\n    // swept `period`): apo"
    ));
    assert!(
        !F64_REGISTRY
            .contains("// PERIOD-INVARIANT:\n    // cpu_batch.rs:3454 reads `length` (10)")
    );
}
