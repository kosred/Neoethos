#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-cache/gpu-bench/preflight.json}"
EXPECTED_GPU="${NEOETHOS_EXPECT_GPU_SUBSTRING:-RTX A6000}"
MIN_VRAM_MIB="${NEOETHOS_MIN_VRAM_MIB:-45000}"
MIN_RAM_KIB="${NEOETHOS_MIN_RAM_KIB:-16000000}"
MIN_DISK_KIB="${NEOETHOS_MIN_DISK_KIB:-20000000}"
mkdir -p "$(dirname "$OUT")"
PROBE_LOG_DIR="$(dirname "$OUT")/preflight-logs"
mkdir -p "$PROBE_LOG_DIR"

# Python is deliberately absent: the paid-run path is Rust-only. The legacy
# helpers under scripts/gpu-bench/*.py are isolated tooling and are never called
# from here.
required=(git cargo rustc g++ nvidia-smi nvcc nsys ncu compute-sanitizer)
missing=()
for command_name in "${required[@]}"; do
  command -v "$command_name" >/dev/null 2>&1 || missing+=("$command_name")
done
if ((${#missing[@]})); then
  printf 'Missing required tools: %s\n' "${missing[*]}" >&2
  exit 20
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)"
GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n1 | xargs)"
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | xargs)"
VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | xargs)"
POWER_LIMIT_W="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -n1 | xargs)"
MAX_SM_CLOCK_MHZ="$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -n1 | xargs)"
TEMP_C="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -n1 | xargs)"
CUDA_TOOLKIT="$(nvcc --version | sed -n 's/.*release \([^,]*\).*/\1/p' | tail -n1)"
RAM_KIB="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
DISK_KIB="$(df -Pk . | awk 'NR==2 {print $4}')"

if [[ "${NEOETHOS_ALLOW_OTHER_GPU:-0}" != "1" && "$GPU_NAME" != *"$EXPECTED_GPU"* ]]; then
  printf 'Unexpected GPU: %s (expected substring %s)\n' "$GPU_NAME" "$EXPECTED_GPU" >&2
  exit 21
fi
if (( VRAM_MIB < MIN_VRAM_MIB )); then
  printf 'Insufficient VRAM: %s MiB < %s MiB\n' "$VRAM_MIB" "$MIN_VRAM_MIB" >&2
  exit 22
fi
if (( RAM_KIB < MIN_RAM_KIB )); then
  printf 'Insufficient RAM: %s KiB < %s KiB\n' "$RAM_KIB" "$MIN_RAM_KIB" >&2
  exit 23
fi
if (( DISK_KIB < MIN_DISK_KIB )); then
  printf 'Insufficient disk: %s KiB < %s KiB\n' "$DISK_KIB" "$MIN_DISK_KIB" >&2
  exit 24
fi

nvidia-smi -L >/dev/null
nsys status --environment >/tmp/neoethos-nsys-status.txt 2>&1 || {
  cat /tmp/neoethos-nsys-status.txt >&2
  exit 25
}
if ! find /usr/local /opt -path '*CUPTI*' -name 'libcupti.so*' -print -quit 2>/dev/null | grep -q .; then
  printf 'CUPTI library not found under /usr/local or /opt\n' >&2
  exit 26
fi

run_required_cuda_probe() {
  local label="$1"
  shift
  local log="$PROBE_LOG_DIR/${label}.stdout.log"
  set +e
  "$@" 2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  set -e
  if (( status != 0 )); then
    return "$status"
  fi
  if grep -Eqi 'GPU test skipped|CUDA smoke skipped|adapter.*skipped' "$log"; then
    printf 'Required CUDA probe reported a skip: %s\n' "$label" >&2
    return 27
  fi
}

# Real Rust -> C ABI -> CUDA allocation/upload/kernel/readback smoke path.
run_required_cuda_probe prototype-b-smoke \
  env NEOETHOS_RUN_CUDA_SMOKE=1 cargo test \
  -p neoethos-gpu-cuda --features cuda \
  tests::real_cuda_smoke_is_explicitly_gpu_gated -- --exact --nocapture

# Direct CUDA correctness probes for the CubeCL compact-event and trace kernels.
run_required_cuda_probe prototype-c-direct \
  env NEOETHOS_REQUIRE_GPU=1 cargo test -p neoethos-search --features gpu-cuda \
  gpu_event_first_hit_matches_reference_when_adapter_is_available -- --nocapture
run_required_cuda_probe signal-trace-direct \
  env NEOETHOS_REQUIRE_GPU=1 cargo test -p neoethos-search --features gpu-cuda \
  direct_gpu_trace_matches_cpu_when_an_adapter_is_available -- --nocapture
run_required_cuda_probe trade-trace-direct \
  env NEOETHOS_REQUIRE_GPU=1 cargo test -p neoethos-search --features gpu-cuda \
  direct_trade_trace_levels_four_through_nine_match_cpu -- --nocapture

# Execute memcheck rather than merely checking that the binary exists.
# `cargo test` launches the test binary as a child, so track all descendants;
# the non-zero sanitizer exit code turns detected memory errors into a hard gate.
run_required_cuda_probe compute-sanitizer-native \
  env NEOETHOS_RUN_CUDA_SMOKE=1 compute-sanitizer \
  --tool memcheck \
  --target-processes all \
  --require-cuda-init no \
  --error-exitcode 86 \
  --log-file "$PROBE_LOG_DIR/compute-sanitizer-native.log" \
  cargo test -p neoethos-gpu-cuda --features cuda \
  tests::real_cuda_smoke_is_explicitly_gpu_gated -- --exact --nocapture
run_required_cuda_probe compute-sanitizer-gpu-native \
  env NEOETHOS_REQUIRE_GPU=1 compute-sanitizer \
  --tool memcheck \
  --target-processes all \
  --require-cuda-init no \
  --error-exitcode 86 \
  --log-file "$PROBE_LOG_DIR/compute-sanitizer-gpu-native.log" \
  cargo test -p neoethos-search --features gpu-cuda \
  gpu_native:: -- --nocapture --test-threads=1

# The preflight report is written by the Rust CLI, not by an inline script.
cargo run --quiet --release -p neoethos-cli -- bench-preflight-report \
  --out "$OUT" \
  --gpu "$GPU_NAME" \
  --gpu-uuid "$GPU_UUID" \
  --driver "$DRIVER" \
  --vram-mib "$VRAM_MIB" \
  --cuda-toolkit "$CUDA_TOOLKIT" \
  --ram-kib "$RAM_KIB" \
  --disk-kib "$DISK_KIB" \
  --power-limit-watts "$POWER_LIMIT_W" \
  --max-sm-clock-mhz "$MAX_SM_CLOCK_MHZ" \
  --temperature-celsius "$TEMP_C"
