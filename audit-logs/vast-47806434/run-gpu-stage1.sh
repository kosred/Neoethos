#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/neoethos/dependency-upgrade-probe
log_dir=/workspace/neoethos/audit-logs/compatible/gpu-stage1-attempt2
target=/workspace/neoethos/targets/compatible-gpu/target

mkdir -p "$log_dir" "$target"

if [[ ! -f "$repo/Cargo.toml" ]]; then
  echo "GPU_STAGE1_PREFLIGHT_ERROR missing=$repo/Cargo.toml" >&2
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

echo "GPU_STAGE1_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "GPU_STAGE1_REPO=$repo"
echo "GPU_STAGE1_PWD=$PWD"
echo "CUDA_ARCHS=$CUDA_ARCHS"
echo "NEOETHOS_CUDA_ARCH=$NEOETHOS_CUDA_ARCH"
echo "CUDA_FAST_MATH=$CUDA_FAST_MATH"
echo "CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS"
rustc +nightly-2026-04-07 -Vv
cargo +nightly-2026-04-07 -Vv
"$CUDACXX" --version
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total --format=csv,noheader

run_probe gpu-contracts \
  cargo +nightly-2026-04-07 test -p neoethos-gpu-contracts --all-targets -j 62 -- --nocapture --test-threads=1
contracts_status=$?

run_probe native-cuda-smoke \
  env NEOETHOS_RUN_CUDA_SMOKE=1 cargo +nightly-2026-04-07 test -p neoethos-gpu-cuda --features cuda --all-targets -j 62 -- --nocapture --test-threads=1
smoke_status=$?

run_probe prototype-b-native \
  env NEOETHOS_REQUIRE_GPU=1 cargo +nightly-2026-04-07 test -p neoethos-search --features gpu-b-native --all-targets -j 62 prototype_b -- --nocapture --test-threads=1
prototype_status=$?

sanitizer_status=127
if command -v compute-sanitizer >/dev/null 2>&1; then
  run_probe native-cuda-memcheck \
    env NEOETHOS_RUN_CUDA_SMOKE=1 compute-sanitizer --tool memcheck --target-processes all --require-cuda-init no --error-exitcode 86 --log-file "$log_dir/compute-sanitizer-native.log" \
    cargo +nightly-2026-04-07 test -p neoethos-gpu-cuda --features cuda -j 62 tests::real_cuda_smoke_is_explicitly_gpu_gated -- --exact --nocapture --test-threads=1
  sanitizer_status=$?
else
  echo "PROBE_ERROR label=native-cuda-memcheck reason=compute-sanitizer-missing"
fi

skip_status=0
if grep -Eqi 'CUDA smoke skipped|GPU test skipped|adapter[^[:cntrl:]]*skipped' "$log_dir/native-cuda-smoke.log" "$log_dir/prototype-b-native.log"; then
  echo "PROBE_ERROR reason=required-gpu-probe-reported-skip"
  skip_status=1
fi

echo "GPU_CONTRACTS_EXIT=$contracts_status"
echo "NATIVE_CUDA_SMOKE_EXIT=$smoke_status"
echo "PROTOTYPE_B_NATIVE_EXIT=$prototype_status"
echo "NATIVE_CUDA_MEMCHECK_EXIT=$sanitizer_status"
echo "REQUIRED_GPU_SKIP_SCAN_EXIT=$skip_status"
df -h /workspace
echo "GPU_STAGE1_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ $contracts_status -ne 0 || $smoke_status -ne 0 || $prototype_status -ne 0 || $sanitizer_status -ne 0 || $skip_status -ne 0 ]]; then
  exit 1
fi
