#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-cache/gpu-bench/preflight.json}"
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
GPU_COMPUTE_CAPABILITY="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | xargs)"
GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n1 | xargs)"
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | xargs)"
VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | xargs)"
POWER_LIMIT_W="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -n1 | xargs)"
MAX_SM_CLOCK_MHZ="$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -n1 | xargs)"
TEMP_C="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -n1 | xargs)"
CUDA_TOOLKIT="$(nvcc --version | sed -n 's/.*release \([^,]*\).*/\1/p' | tail -n1)"
RAM_KIB="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
DISK_KIB="$(df -Pk . | awk 'NR==2 {print $4}')"

bash "$SCRIPT_DIR/check_cuda_hardware.sh" \
  "$GPU_NAME" "$GPU_COMPUTE_CAPABILITY" "$VRAM_MIB"
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

# Every paid-device correctness gate has a pinned passing/ignored test count.
# The shared runner also rejects skip, fallback and substitution diagnostics.
NEOETHOS_GPU_VALIDATION_LOG_DIR="$PROBE_LOG_DIR/cuda-validation" \
  bash "$SCRIPT_DIR/run_cuda_validation.sh" all

# Normal paid-device suites have passed above. Re-run the native ABI, native-B,
# CubeCL population, direct Prototype A, and full resident Prototype C binaries
# under memcheck. Every pinned test binary must exit cleanly and report zero
# memory errors and zero leaked bytes.
NEOETHOS_GPU_SANITIZER_LOG_DIR="$PROBE_LOG_DIR/cuda-memcheck" \
  bash "$SCRIPT_DIR/run_cuda_memcheck_validation.sh" all

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
