#!/usr/bin/env bash
# Bootstrap and gate a rented NVIDIA box, cheapest-first.
#
# Runs ON the rented machine. The ordering is deliberate and is about money:
# the question that justifies renting at all — "is the warp-cooperative
# Prototype B kernel correct?" — is answered in stage 1, which needs only
# rustc, a C++ compiler and nvcc. The CubeCL search/data stack is exercised only
# in stage 3, and only if stage 1 and 2 passed. If the budget dies early,
# you still leave with the answer you paid for.
#
# Nothing here fabricates a result. Every filtered test binary has an exact
# expected count and rejects skip/fallback/substitution output. CubeCL search
# tests also set their explicit real-device switch.
#
# Usage:
#   scripts/gpu-bench/remote_bootstrap.sh [stage1|stage2|stage3|all]

set -euo pipefail

STAGE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# mold 1.0.3 splits `.init_array` and leaves nvcc's fatbin-registration
# constructors outside the dynamic tag, so CUDA kernels build, link, and then
# fail at launch with "invalid device function". Every CUDA build here uses the
# default linker instead. Test harnesses remain on the portable CPU baseline;
# they are not private payloads behind the x86-64 v3 launcher.
export RUSTFLAGS="${RUSTFLAGS:--C link-arg=-fuse-ld=bfd}"
RESULTS="${NEOETHOS_RESULTS_DIR:-cache/gpu-bench/remote}"
mkdir -p "$RESULTS"

log() { printf '\n== %s ==\n' "$1" | tee -a "$RESULTS/run.log"; }
record() { tee -a "$RESULTS/run.log"; }

fail() {
  printf '\nFAILED at: %s\n' "$1" | tee -a "$RESULTS/run.log" >&2
  printf 'Everything before this point is proven; everything after is not.\n' >&2
  exit 1
}

# ---------------------------------------------------------------------------
# Stage 0 — environment, recorded before anything is built
# ---------------------------------------------------------------------------

log "stage 0: environment"
{
  date -u +'utc=%Y-%m-%dT%H:%M:%SZ'
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi \
    --query-gpu=name,uuid,driver_version,memory.total,compute_cap,clocks.max.sm \
    --format=csv,noheader || echo "nvidia-smi: MISSING"
  command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -n2 || echo "nvcc: MISSING"
  command -v cargo >/dev/null 2>&1 && cargo --version || echo "cargo: MISSING"
  command -v g++ >/dev/null 2>&1 && g++ --version | head -n1 || echo "g++: MISSING"
  echo "cpus=$(nproc 2>/dev/null || echo unknown)"
  free -h 2>/dev/null | head -n2 || true
  df -h . 2>/dev/null | tail -n1 || true
} | record > "$RESULTS/environment.txt" 2>&1 || true
cat "$RESULTS/environment.txt"

for tool in nvidia-smi nvcc g++ compute-sanitizer; do
  command -v "$tool" >/dev/null 2>&1 \
    || fail "stage 0: $tool is missing; this box cannot complete the paid CUDA proof"
done
bash "$SCRIPT_DIR/check_cuda_hardware.sh" \
  || fail "stage 0: unsupported NVIDIA hardware"
if ! command -v cargo >/dev/null 2>&1; then
  log "stage 0: installing rustup"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
fi

# bindgen (pulled in transitively) needs libclang; cmake is needed by several
# native dependencies. Installing them up front avoids a build that dies ten
# minutes in on a metered box.
if command -v apt-get >/dev/null 2>&1; then
  log "stage 0: build dependencies"
  DEBIAN_FRONTEND=noninteractive apt-get update -qq >/dev/null 2>&1 || true
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    git curl build-essential pkg-config libssl-dev libclang-dev clang cmake \
    >/dev/null 2>&1 || true
fi

# ---------------------------------------------------------------------------
# Stage 1 — the cheap gate: native CUDA only, no CubeCL, no libtorch
# ---------------------------------------------------------------------------

run_stage1() {
  log "stage 1: contracts and ABI"
  cargo test -p neoethos-gpu-contracts 2>&1 | record \
    || fail "stage 1: gpu-contracts"

  log "stage 1: f64 native ABI plus all three native-B parity tests"
  NEOETHOS_GPU_VALIDATION_LOG_DIR="$RESULTS/stage1-validation" \
    bash "$SCRIPT_DIR/run_cuda_validation.sh" native 2>&1 | record \
    || fail "stage 1: strict native CUDA validation"

  log "stage 1: PASSED — native ABI 1/1 and native-B parity 3/3"
}

# ---------------------------------------------------------------------------
# Stage 2 — memory correctness
# ---------------------------------------------------------------------------

run_stage2() {
  log "stage 2: Compute Sanitizer over native ABI 1/1 and native-B parity 3/3"
  NEOETHOS_GPU_SANITIZER_LOG_DIR="$RESULTS/stage2-native-memcheck" \
  NEOETHOS_GPU_TELEMETRY_LOG="$RESULTS/stage2-native-telemetry.csv" \
    bash "$SCRIPT_DIR/run_cuda_memcheck_validation.sh" native 2>&1 | record \
    || fail "stage 2: native ABI/native-B memcheck or leak check"
  log "stage 2: PASSED — zero memcheck errors and zero leaked bytes"
}

# ---------------------------------------------------------------------------
# Stage 3 — CubeCL search engines and the complete current data CUDA suites
# ---------------------------------------------------------------------------

run_stage3() {
  log "stage 3: CubeCL f64 population/trailing/fused, direct A, and C 7/7"
  NEOETHOS_GPU_VALIDATION_LOG_DIR="$RESULTS/stage3-cubecl-validation" \
    bash "$SCRIPT_DIR/run_cuda_validation.sh" cubecl 2>&1 | record \
    || fail "stage 3: strict CubeCL/A/C validation"

  log "stage 3: Compute Sanitizer over CubeCL 7/7, direct A 1/1, and resident C 7/7"
  NEOETHOS_GPU_SANITIZER_LOG_DIR="$RESULTS/stage3-search-memcheck" \
  NEOETHOS_GPU_TELEMETRY_LOG="$RESULTS/stage3-search-telemetry.csv" \
    bash "$SCRIPT_DIR/run_cuda_memcheck_validation.sh" cubecl 2>&1 | record \
    || fail "stage 3: CubeCL/A/C memcheck or leak check"

  log "stage 3: complete current resident-f64 data suite and HPC sweep parity"
  NEOETHOS_GPU_VALIDATION_LOG_DIR="$RESULTS/stage3-data-validation" \
    bash "$SCRIPT_DIR/run_cuda_validation.sh" data 2>&1 | record \
    || fail "stage 3: strict data CUDA validation"

  log "stage 3: PASSED — direct engine tests, not the misattributed CLI A benchmark"
}

case "$STAGE" in
  stage1) run_stage1 ;;
  stage2) run_stage1; run_stage2 ;;
  stage3) run_stage3 ;;
  all) run_stage1; run_stage2; run_stage3 ;;
  *) fail "unknown stage '$STAGE' (expected stage1|stage2|stage3|all)" ;;
esac

log "results in $RESULTS"
