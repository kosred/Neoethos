const NATIVE_RECEIPT: &str = include_str!("../src/native_population_residency_receipt_v1.rs");

fn source_between<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start = source
        .find(start)
        .unwrap_or_else(|| panic!("missing source boundary `{start}`"));
    let tail = &source[start..];
    let end = tail
        .find(end)
        .unwrap_or_else(|| panic!("missing source boundary `{end}` after `{start}`"));
    &tail[..end]
}

#[test]
fn strict_host_metrics_receipt_allows_only_the_single_metric_readback_boundary() {
    let seal = source_between(
        NATIVE_RECEIPT,
        "pub(crate) fn seal_native_population_residency_receipt_v1(",
        "\n    let device_name =",
    );

    for required in [
        "counters.diagnostic_readback_count() != 0",
        "counters.diagnostic_readback_rows() != 0",
        "counters.diagnostic_readback_bytes() != 0",
        "counters.accepted_trade_total_readback_count() != 0",
        "counters.accepted_trade_total_readback_bytes() != 0",
        "expected_metric_bytes != Some(counters.metric_rows_readback_bytes())",
        "counters.explicit_synchronization_count() != counters.metric_rows_readback_count()",
        "successful_native_population_count != counters.metric_rows_readback_count()",
    ] {
        assert!(
            seal.contains(required),
            "strict host-metrics receipt is missing `{required}`"
        );
    }

    assert!(!seal.contains("expected_control_bytes"));
    assert!(!seal.contains(
        "counters.explicit_synchronization_count() != counters.accepted_trade_total_readback_count()"
    ));
}
