from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")

for relative in (
    "crates/neoethos-search/src/gpu_native/prototype_c_gpu.rs",
    "crates/neoethos-search/src/gpu_native/signal_trace_gpu.rs",
):
    replace_once(
        Path(relative),
        '#[cfg(all(test, feature = "gpu-vulkan", not(feature = "gpu-cuda")))]\nmod tests {',
        '#[cfg(all(test, any(feature = "gpu-cuda", feature = "gpu-vulkan")))]\nmod tests {',
        f"CUDA/WGPU direct test cfg for {relative}",
    )

preflight = Path("scripts/gpu-bench/preflight.sh")
text = preflight.read_text(encoding="utf-8")
needle = '''NEOETHOS_RUN_CUDA_SMOKE=1 cargo test \\
  -p neoethos-gpu-cuda --features cuda \\
  tests::real_cuda_smoke_is_explicitly_gpu_gated -- --exact --nocapture
'''
insert = needle + '''
# Direct CUDA correctness probes for the CubeCL compact-event and trace kernels.
cargo test -p neoethos-search --features gpu-cuda \\
  gpu_event_first_hit_matches_reference_when_adapter_is_available -- --nocapture
cargo test -p neoethos-search --features gpu-cuda \\
  direct_gpu_trace_matches_cpu_when_an_adapter_is_available -- --nocapture
'''
if text.count(needle) != 1:
    raise RuntimeError(f"A6000 CUDA probe anchor: expected one match, found {text.count(needle)}")
preflight.write_text(text.replace(needle, insert, 1), encoding="utf-8")
print("enabled direct CUDA correctness probes for Prototype C and signal traces")
