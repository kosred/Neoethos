#!/usr/bin/env bash
# Count-pinned Compute Sanitizer gates for every paid search CUDA family.

set -euo pipefail

group="${1:-all}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sanitizer_gate="$script_dir/run_compute_sanitizer_gate.sh"
log_dir="${NEOETHOS_GPU_SANITIZER_LOG_DIR:-cache/gpu-bench/cuda-memcheck}"
mkdir -p "$log_dir"

# These are shell exports, not temporary assignments on the background
# telemetry command. Every subsequently launched Cargo test must inherit them.
export NEOETHOS_RUN_CUDA_SEARCH_TESTS=1
export NEOETHOS_REQUIRE_GPU=1

telemetry_log="${NEOETHOS_GPU_TELEMETRY_LOG:-}"
telemetry_pid=""

stop_telemetry() {
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" >/dev/null 2>&1 || true
    wait "$telemetry_pid" >/dev/null 2>&1 || true
    telemetry_pid=""
  fi
}

start_telemetry() {
  [[ -n "$telemetry_log" ]] || return 0
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'nvidia-smi is required when NEOETHOS_GPU_TELEMETRY_LOG is set\n' >&2
    exit 69
  fi
  mkdir -p "$(dirname "$telemetry_log")"
  : >"$telemetry_log"
  nvidia-smi \
    --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used \
    --format=csv,noheader,nounits \
    --loop-ms=100 >"$telemetry_log" 2>&1 &
  telemetry_pid=$!

  local attempt
  for attempt in {1..50}; do
    if [[ -s "$telemetry_log" ]] && kill -0 "$telemetry_pid" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$telemetry_pid" >/dev/null 2>&1; then
      wait "$telemetry_pid" >/dev/null 2>&1 || true
      printf 'nvidia-smi telemetry exited before its first sample (log: %s)\n' \
        "$telemetry_log" >&2
      exit 94
    fi
    sleep 0.02
  done
  printf 'nvidia-smi telemetry produced no sample (log: %s)\n' "$telemetry_log" >&2
  exit 95
}

trap stop_telemetry EXIT
start_telemetry

run_sanitizer() {
  local label="$1"
  local expected_passed="$2"
  local expected_ignored="$3"
  shift 3
  bash "$sanitizer_gate" \
    "$log_dir/${label}-tests.log" \
    "$log_dir/${label}-memcheck.log" \
    "$expected_passed" "$expected_ignored" "$@"
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
  run_sanitizer native-abi-f64 1 0 \
    "${native_env[@]}" \
    cargo test -p neoethos-gpu-cuda --features cuda --lib \
    tests::real_cuda_smoke_executes_f64_first_hit_without_narrowing -- \
    --exact --nocapture --test-threads=1

  run_sanitizer native-b 3 0 \
    "${native_env[@]}" \
    cargo test -p neoethos-search --features gpu-b-native --lib \
    eval::trailing_parity_tests:: -- --nocapture --test-threads=1
}

run_cubecl() {
  run_sanitizer cubecl-population 7 0 \
    env \
    cargo test -p neoethos-search --features gpu-cuda --lib \
    eval::gpu_cpu_parity_tests:: -- --nocapture --test-threads=1

  run_sanitizer prototype-a-direct 1 0 \
    env \
    cargo test -p neoethos-search --features gpu-cuda --lib \
    gpu_native::prototype_a::tests::direct_prototype_a_engine_is_resident_and_matches_cpu_fixture -- \
    --exact --nocapture --test-threads=1

  run_sanitizer prototype-c-device 7 0 \
    env \
    cargo test -p neoethos-search --features gpu-cuda --lib \
    gpu_native::prototype_c_engine::device_tests:: -- --nocapture --test-threads=1
}

case "$group" in
  native) run_native ;;
  cubecl) run_cubecl ;;
  all) run_native; run_cubecl ;;
  *)
    printf 'unknown CUDA memcheck validation group %q (expected native|cubecl|all)\n' \
      "$group" >&2
    exit 64
    ;;
esac
