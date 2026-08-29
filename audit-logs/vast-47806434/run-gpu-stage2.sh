#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/neoethos/dependency-upgrade-probe
log_dir=/workspace/neoethos/audit-logs/compatible/gpu-stage2
target=/workspace/neoethos/targets/compatible-gpu/target

mkdir -p "$log_dir" "$target"

if [[ ! -f "$repo/Cargo.toml" ]]; then
  echo "GPU_STAGE2_PREFLIGHT_ERROR missing=$repo/Cargo.toml" >&2
  exit 2
fi

cd "$repo"

if [[ -f /root/.cargo/env ]]; then
  source /root/.cargo/env
fi

export CARGO_BUILD_JOBS=62
export CARGO_TARGET_DIR="$target"
export RUST_TEST_THREADS=1
export CUDA_ARCHS=89
export NEOETHOS_CUDA_ARCH=compute_89,code=sm_89
export CUDA_FAST_MATH=0
export CUDACXX=/usr/local/cuda/bin/nvcc

run_probe() {
  local label=$1
  shift
  local log="$log_dir/$label.log"
  echo "PROBE_START label=$label utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$log"
  "$@" 2>&1 | tee -a "$log"
  local status=${PIPESTATUS[0]}
  echo "PROBE_EXIT label=$label status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log"
  return "$status"
}

summary="$log_dir/summary.log"
exec > >(tee -a "$summary") 2>&1

echo "GPU_STAGE2_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "GPU_STAGE2_REPO=$repo"
echo "GPU_STAGE2_PWD=$PWD"
echo "CUDA_ARCHS=$CUDA_ARCHS"
echo "NEOETHOS_CUDA_ARCH=$NEOETHOS_CUDA_ARCH"
echo "CUDA_FAST_MATH=$CUDA_FAST_MATH"
echo "CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS"
git rev-parse HEAD
git status --short
sha256sum Cargo.lock
rustc +nightly-2026-04-07 -Vv
cargo +nightly-2026-04-07 -Vv
"$CUDACXX" --version
nvidia-smi --query-gpu=name,uuid,pci.bus_id,compute_cap,driver_version,memory.total --format=csv,noheader

run_probe data-cuda-all-targets-compile \
  cargo +nightly-2026-04-07 test -p neoethos-data --features gpu-cuda --all-targets --no-run -j 62
data_compile_status=$?

run_probe data-cuda-real-parity \
  env NEOETHOS_REQUIRE_GPU=1 cargo +nightly-2026-04-07 test -p neoethos-data --features gpu-cuda -j 62 gpu_cpu_indicator_sweep_parity -- --nocapture --test-threads=1
data_parity_status=$?

run_probe data-f64-cpu-reference \
  cargo +nightly-2026-04-07 test -p neoethos-data --features gpu-cuda --test f64_lane_cpu_reference -j 62 -- --nocapture --test-threads=1
data_reference_status=$?

run_probe search-cuda-all-targets-compile \
  cargo +nightly-2026-04-07 test -p neoethos-search --features gpu-cuda --all-targets --no-run -j 62
search_compile_status=$?

run_probe cli-nvidia-all-targets-compile \
  cargo +nightly-2026-04-07 test -p neoethos-cli --features gpu-nvidia --all-targets --no-run -j 62
cli_compile_status=$?

run_probe app-nvidia-all-targets-compile \
  cargo +nightly-2026-04-07 test -p neoethos-app --features gpu-nvidia --all-targets --no-run -j 62
app_compile_status=$?

run_probe cli-nvidia-bench-aggregate-compile \
  cargo +nightly-2026-04-07 test -p neoethos-cli --features gpu-nvidia,gpu-bench-cuda --all-targets --no-run -j 62
cli_aggregate_status=$?

skip_status=0
if grep -Eqi 'CUDA lane[^[:cntrl:]]*SKIPPED|GPU test[^[:cntrl:]]*skipped|adapter[^[:cntrl:]]*skipped|CPU fallback' "$log_dir/data-cuda-real-parity.log"; then
  echo "PROBE_ERROR reason=required-data-gpu-probe-reported-skip-or-fallback"
  skip_status=1
fi

echo "DATA_CUDA_ALL_TARGETS_COMPILE_EXIT=$data_compile_status"
echo "DATA_CUDA_REAL_PARITY_EXIT=$data_parity_status"
echo "DATA_F64_CPU_REFERENCE_EXIT=$data_reference_status"
echo "SEARCH_CUDA_ALL_TARGETS_COMPILE_EXIT=$search_compile_status"
echo "CLI_NVIDIA_ALL_TARGETS_COMPILE_EXIT=$cli_compile_status"
echo "APP_NVIDIA_ALL_TARGETS_COMPILE_EXIT=$app_compile_status"
echo "CLI_NVIDIA_BENCH_AGGREGATE_COMPILE_EXIT=$cli_aggregate_status"
echo "REQUIRED_DATA_GPU_SKIP_SCAN_EXIT=$skip_status"
df -h /workspace
echo "GPU_STAGE2_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ $data_compile_status -ne 0 || $data_parity_status -ne 0 || $data_reference_status -ne 0 || $search_compile_status -ne 0 || $cli_compile_status -ne 0 || $app_compile_status -ne 0 || $cli_aggregate_status -ne 0 || $skip_status -ne 0 ]]; then
  exit 1
fi
