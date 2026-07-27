#!/usr/bin/env bash
# Bootstrap and gate a rented NVIDIA box, cheapest-first.
#
# Runs ON the rented machine. The ordering is deliberate and is about money:
# the question that justifies renting at all — "is the warp-cooperative
# Prototype B kernel correct?" — is answered in stage 1, which needs only
# rustc, a C++ compiler and nvcc. The heavy CubeCL/libtorch stack is provisioned
# only in stage 3, and only if stage 1 and 2 passed. If the budget dies early,
# you still leave with the answer you paid for.
#
# Nothing here fabricates a result. `NEOETHOS_REQUIRE_GPU=1` turns a
# "no adapter, skipping" outcome into a hard failure, so a green run cannot mean
# the tests quietly did nothing on a box you are being billed for.
#
# Usage:
#   scripts/gpu-bench/remote_bootstrap.sh [stage1|stage2|stage3|all]

set -euo pipefail

STAGE="${1:-all}"

# The repo pins mold on Linux for link speed. mold 1.0.3 splits `.init_array`
# and leaves nvcc's fatbin-registration constructors outside the dynamic tag,
# so CUDA kernels build, link, and then fail at launch with "invalid device
# function". Every CUDA build here uses the default linker instead. RUSTFLAGS
# replaces .cargo/config.toml's list, so target-cpu is restated.
export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=x86-64-v3 -C link-arg=-fuse-ld=bfd}"
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
    --query-gpu=name,uuid,driver_version,memory.total,clocks.max.sm \
    --format=csv,noheader || echo "nvidia-smi: MISSING"
  command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -n2 || echo "nvcc: MISSING"
  command -v cargo >/dev/null 2>&1 && cargo --version || echo "cargo: MISSING"
  command -v g++ >/dev/null 2>&1 && g++ --version | head -n1 || echo "g++: MISSING"
  echo "cpus=$(nproc 2>/dev/null || echo unknown)"
  free -h 2>/dev/null | head -n2 || true
  df -h . 2>/dev/null | tail -n1 || true
} | record > "$RESULTS/environment.txt" 2>&1 || true
cat "$RESULTS/environment.txt"

for tool in nvidia-smi nvcc g++; do
  command -v "$tool" >/dev/null 2>&1 || fail "stage 0: $tool is missing; this box cannot prove Prototype B"
done
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

  log "stage 1: Rust -> C ABI -> CUDA smoke"
  NEOETHOS_RUN_CUDA_SMOKE=1 cargo test -p neoethos-gpu-cuda --features cuda -- --nocapture 2>&1 \
    | tee "$RESULTS/stage1-cuda-smoke.log" | record \
    || fail "stage 1: CUDA smoke"
  grep -qi "skipped" "$RESULTS/stage1-cuda-smoke.log" \
    && fail "stage 1: CUDA smoke reported a skip on a rented GPU"

  log "stage 1: Prototype B population parity against the canonical oracle"
  NEOETHOS_REQUIRE_GPU=1 cargo test -p neoethos-search --features gpu-b-native \
    prototype_b -- --nocapture --test-threads=1 2>&1 \
    | tee "$RESULTS/stage1-prototype-b.log" | record \
    || fail "stage 1: Prototype B parity"
  grep -qi "skipped" "$RESULTS/stage1-prototype-b.log" \
    && fail "stage 1: Prototype B parity reported a skip while a GPU was required"

  log "stage 1: PASSED — the B kernel is correct on real CUDA"
}

# ---------------------------------------------------------------------------
# Stage 2 — memory correctness
# ---------------------------------------------------------------------------

run_stage2() {
  command -v compute-sanitizer >/dev/null 2>&1 \
    || fail "stage 2: compute-sanitizer is missing"
  log "stage 2: Compute Sanitizer memcheck over the B population path"
  NEOETHOS_REQUIRE_GPU=1 compute-sanitizer \
    --tool memcheck --target-processes all --require-cuda-init no --error-exitcode 86 \
    --log-file "$RESULTS/stage2-memcheck.log" \
    cargo test -p neoethos-search --features gpu-b-native prototype_b -- --test-threads=1 2>&1 \
    | record || fail "stage 2: memcheck"
  log "stage 2: PASSED — no invalid access in the B population path"
}

# ---------------------------------------------------------------------------
# Stage 3 — full stack: CubeCL CUDA plus the attributed A/B/C matrix
# ---------------------------------------------------------------------------

run_stage3() {
  log "stage 3: full GPU stack build (CubeCL CUDA + libtorch)"
  cargo build --release -p neoethos-cli --features gpu-nvidia 2>&1 | record \
    || fail "stage 3: full-stack build (see docs/ for the libtorch recipe)"

  log "stage 3: Prototype C on CUDA"
  NEOETHOS_REQUIRE_GPU=1 cargo test -p neoethos-search --features gpu-cuda \
    gpu_native:: -- --nocapture --test-threads=1 2>&1 \
    | tee "$RESULTS/stage3-gpu-native.log" | record \
    || fail "stage 3: gpu_native suite on CUDA"

  log "stage 3: attributed matrix"
  local sha
  sha="$(git rev-parse HEAD)"
  ./target/release/neoethos-cli bench-matrix \
    --candidate-sha "$sha" \
    --runs-root "$RESULTS/runs" \
    --out "$RESULTS/matrix.json" 2>&1 | record \
    || fail "stage 3: matrix generation"
  log "stage 3: matrix written; execute the printed commands, then bench-collate"
}

case "$STAGE" in
  stage1) run_stage1 ;;
  stage2) run_stage1; run_stage2 ;;
  stage3) run_stage3 ;;
  all) run_stage1; run_stage2; run_stage3 ;;
  *) fail "unknown stage '$STAGE' (expected stage1|stage2|stage3|all)" ;;
esac

log "results in $RESULTS"
