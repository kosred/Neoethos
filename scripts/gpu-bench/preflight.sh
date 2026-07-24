#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-cache/gpu-bench/preflight.json}"
EXPECTED_GPU="${NEOETHOS_EXPECT_GPU_SUBSTRING:-RTX A6000}"
MIN_VRAM_MIB="${NEOETHOS_MIN_VRAM_MIB:-45000}"
MIN_RAM_KIB="${NEOETHOS_MIN_RAM_KIB:-16000000}"
MIN_DISK_KIB="${NEOETHOS_MIN_DISK_KIB:-20000000}"
mkdir -p "$(dirname "$OUT")"

required=(git cargo rustc python3 g++ nvidia-smi nvcc nsys ncu compute-sanitizer)
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

# Real Rust -> C ABI -> CUDA allocation/upload/kernel/readback smoke path.
NEOETHOS_RUN_CUDA_SMOKE=1 cargo test \
  -p neoethos-gpu-cuda --features cuda \
  tests::real_cuda_smoke_is_explicitly_gpu_gated -- --exact --nocapture

# Direct CUDA correctness probes for the CubeCL compact-event and trace kernels.
cargo test -p neoethos-search --features gpu-cuda \
  gpu_event_first_hit_matches_reference_when_adapter_is_available -- --nocapture
cargo test -p neoethos-search --features gpu-cuda \
  direct_gpu_trace_matches_cpu_when_an_adapter_is_available -- --nocapture

python3 - "$OUT" "$GPU_NAME" "$GPU_UUID" "$DRIVER" "$VRAM_MIB" \
  "$CUDA_TOOLKIT" "$RAM_KIB" "$DISK_KIB" "$POWER_LIMIT_W" \
  "$MAX_SM_CLOCK_MHZ" "$TEMP_C" <<'PY'
import json, pathlib, sys
(
    out, gpu, uuid, driver, vram, toolkit, ram, disk,
    power_limit, max_sm_clock, temperature,
) = sys.argv[1:]
payload = {
    "schema_version": 1,
    "gpu_visible": True,
    "gpu": gpu,
    "gpu_uuid": uuid,
    "driver_version": driver,
    "cuda_toolkit_version": toolkit or None,
    "vram_bytes": int(vram) * 1024 * 1024,
    "ram_bytes": int(ram) * 1024,
    "disk_free_bytes": int(disk) * 1024,
    "power_limit_watts": float(power_limit),
    "max_sm_clock_mhz": int(max_sm_clock),
    "temperature_celsius": int(temperature),
    "nsight_environment_checked": True,
    "cupti_present": True,
    "compute_sanitizer_present": True,
    "cuda_smoke_passed": True,
}
path = pathlib.Path(out)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(path)
PY
