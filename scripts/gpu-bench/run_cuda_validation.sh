#!/usr/bin/env bash
# Named, count-pinned real-CUDA gates shared by paid preflight and GPU CI.

set -euo pipefail

group="${1:-all}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gate="$script_dir/run_cuda_test_gate.sh"
log_dir="${NEOETHOS_GPU_VALIDATION_LOG_DIR:-cache/gpu-bench/cuda-validation}"
mkdir -p "$log_dir"

run_gate() {
  local label="$1"
  local expected_passed="$2"
  local expected_ignored="$3"
  shift 3
  bash "$gate" "$log_dir/${label}.log" "$expected_passed" "$expected_ignored" "$@"
}

native_env=(
  env
  -u CUDA_ARCHS
  -u CUDA_ARCH
  -u CMAKE_CUDA_ARCHITECTURES
  -u NVCC_ARGS
  -u NVCC_PREPEND_FLAGS
  -u NVCC_APPEND_FLAGS
  -u CUDAFLAGS
)

run_native() {
  run_gate native-abi-f64 1 0 \
    "${native_env[@]}" \
    cargo test -p neoethos-gpu-cuda --features cuda --lib \
    tests::real_cuda_smoke_executes_f64_first_hit_without_narrowing -- \
    --exact --nocapture --test-threads=1

  run_gate native-b-parity 3 0 \
    "${native_env[@]}" \
    NEOETHOS_RUN_CUDA_SEARCH_TESTS=1 NEOETHOS_REQUIRE_GPU=1 \
    cargo test -p neoethos-search --features gpu-b-native --lib \
    eval::trailing_parity_tests:: -- --nocapture --test-threads=1
}

run_cubecl() {
  run_gate cubecl-population-parity 7 0 \
    env NEOETHOS_RUN_CUDA_SEARCH_TESTS=1 NEOETHOS_REQUIRE_GPU=1 \
    cargo test -p neoethos-search --features gpu-cuda --lib \
    eval::gpu_cpu_parity_tests:: -- --nocapture --test-threads=1

  run_gate cubecl-trailing-parity 1 0 \
    env NEOETHOS_RUN_CUDA_SEARCH_TESTS=1 NEOETHOS_REQUIRE_GPU=1 \
    cargo test -p neoethos-search --features gpu-cuda --lib \
    eval::cubecl_trailing_parity_tests::gpu_cubecl_trailing_stop_matches_cpu -- \
    --exact --nocapture --test-threads=1

  run_gate cubecl-fused-parity 1 0 \
    env NEOETHOS_RUN_CUDA_SEARCH_TESTS=1 NEOETHOS_REQUIRE_GPU=1 \
    cargo test -p neoethos-search --features gpu-cuda --lib \
    cubecl_eval::fused_parity_tests::fused_path_is_byte_identical_to_windowed_path -- \
    --exact --nocapture --test-threads=1

  # This direct engine test is the only current Prototype A proof. The CLI
  # benchmark dispatcher is not accepted as A evidence.
  run_gate prototype-a-direct 1 0 \
    env NEOETHOS_RUN_CUDA_SEARCH_TESTS=1 NEOETHOS_REQUIRE_GPU=1 \
    cargo test -p neoethos-search --features gpu-cuda --lib \
    gpu_native::prototype_a::tests::direct_prototype_a_engine_is_resident_and_matches_cpu_fixture -- \
    --exact --nocapture --test-threads=1

  run_gate prototype-c-device 7 0 \
    env NEOETHOS_RUN_CUDA_SEARCH_TESTS=1 NEOETHOS_REQUIRE_GPU=1 \
    cargo test -p neoethos-search --features gpu-cuda --lib \
    gpu_native::prototype_c_engine::device_tests:: -- --nocapture --test-threads=1
}

run_data() {
  # Two audit/profiler tests are intentionally ignored; all 67 active tests
  # must execute, including every current resident f64 CUDA family.
  run_gate data-resident-f64-suite 67 2 \
    env NEOETHOS_REQUIRE_GPU=1 \
    cargo test -p neoethos-data --features gpu-cuda --lib \
    core::gpu_indicators::tests:: -- --nocapture --test-threads=1

  run_gate data-hpc-sweep-parity 1 0 \
    env NEOETHOS_REQUIRE_GPU=1 \
    cargo test -p neoethos-data --features gpu-cuda --lib \
    core::hpc_ta::tests::gpu_cpu_indicator_sweep_parity -- \
    --exact --nocapture --test-threads=1
}

case "$group" in
  native) run_native ;;
  cubecl) run_cubecl ;;
  data) run_data ;;
  all) run_native; run_cubecl; run_data ;;
  *)
    printf 'unknown CUDA validation group %q (expected native|cubecl|data|all)\n' "$group" >&2
    exit 64
    ;;
esac
