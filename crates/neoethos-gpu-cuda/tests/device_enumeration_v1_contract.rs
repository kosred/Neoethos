use std::fs;
use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-gpu-cuda"))
}

fn read(relative: &str) -> String {
    let path = manifest_dir().join(relative);
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

fn require_all(source: &str, required: &[&str]) {
    for token in required {
        assert!(source.contains(token), "missing required token {token:?}");
    }
}

#[test]
fn c_abi_exposes_one_fallible_exact_count_probe() {
    let header = read("native/neoethos_gpu_cuda.h");
    require_all(
        &header,
        &[
            "NEO_CUDA_DEVICE_PROBE_OK",
            "NEO_CUDA_DEVICE_PROBE_INVALID_OUTPUT",
            "NEO_CUDA_DEVICE_PROBE_ADAPTER_UNAVAILABLE",
            "neoethos_gpu_cuda_probe_device_count_v1(std::uint32_t* out_count)",
        ],
    );
}

#[test]
fn native_probe_preserves_cuda_success_with_exact_zero_or_more_devices() {
    let native = read("native/smoke.cu");
    let probe = section(
        &native,
        "extern \"C\" std::int32_t neoethos_gpu_cuda_probe_device_count_v1(",
        "\n}",
    );
    require_all(
        probe,
        &[
            "std::uint32_t* out_count",
            "out_count == nullptr",
            "cudaGetDeviceCount(&count)",
            "status != cudaSuccess",
            "count < 0",
            "*out_count = static_cast<std::uint32_t>(count)",
            "NEO_CUDA_DEVICE_PROBE_OK",
        ],
    );
    assert!(
        !probe.contains("count > 0"),
        "zero devices is a successful exact enumeration, not runtime failure"
    );
}

#[test]
fn native_cuda_error_is_returned_and_never_collapsed_to_zero_devices() {
    let native = read("native/smoke.cu");
    let probe = section(
        &native,
        "extern \"C\" std::int32_t neoethos_gpu_cuda_probe_device_count_v1(",
        "\n}",
    );
    require_all(
        probe,
        &[
            "static_cast<std::int32_t>(status)",
            "if (status != cudaSuccess)",
        ],
    );
    let error_at = probe
        .find("if (status != cudaSuccess)")
        .expect("CUDA error branch");
    let write_at = probe
        .find("*out_count =")
        .expect("successful exact count write");
    assert!(
        error_at < write_at,
        "the out count was written before CUDA success was established"
    );
}

#[test]
fn no_cuda_stub_reports_adapter_unavailable_and_never_successful_zero() {
    let stub = read("native/stub.cpp");
    let probe = section(
        &stub,
        "extern \"C\" std::int32_t neoethos_gpu_cuda_probe_device_count_v1(",
        "\n}",
    );
    require_all(probe, &["NEO_CUDA_DEVICE_PROBE_ADAPTER_UNAVAILABLE"]);
    for forbidden in ["*out_count = 0", "NEO_CUDA_DEVICE_PROBE_OK"] {
        assert!(
            !probe.contains(forbidden),
            "stub fabricated successful zero-device evidence via {forbidden:?}"
        );
    }
}

#[test]
fn rust_wrapper_returns_typed_adapter_runtime_or_exact_count_evidence() {
    let rust = read("src/lib.rs");
    require_all(
        &rust,
        &[
            "pub enum CudaDeviceEnumerationErrorV1",
            "NativeAdapterUnavailable",
            "RuntimeFailure(i32)",
            "InvalidNativeOutput",
            "pub fn probe_cuda_device_count_v1() -> Result<u32, CudaDeviceEnumerationErrorV1>",
            "neoethos_gpu_cuda_probe_device_count_v1(&mut count)",
            "NEO_CUDA_DEVICE_PROBE_ADAPTER_UNAVAILABLE",
        ],
    );
    let wrapper = section(&rust, "pub fn probe_cuda_device_count_v1()", "\n}");
    for forbidden in ["max(0)", "unwrap_or(0)", "unwrap_or_default()"] {
        assert!(
            !wrapper.contains(forbidden),
            "fallible enumeration collapsed a native failure via {forbidden:?}"
        );
    }
}

#[test]
fn strict_search_route_uses_only_the_fallible_probe_for_cpu_authority() {
    let search = manifest_dir().join("../neoethos-search/src/strict_discovery_device_route_v1.rs");
    let route = fs::read_to_string(&search)
        .unwrap_or_else(|error| panic!("read strict Search route {}: {error}", search.display()));
    let probe = section(
        &route,
        "fn probe_real_strict_discovery_device_route_v1(",
        "\n}",
    );
    require_all(probe, &["neoethos_gpu_cuda::probe_cuda_device_count_v1()"]);
    for forbidden in [
        "neoethos_gpu_cuda::runtime_available()",
        "neoethos_gpu_cuda::device_count()",
    ] {
        assert!(
            !probe.contains(forbidden),
            "strict route used lossy legacy probe {forbidden:?}"
        );
    }
}
