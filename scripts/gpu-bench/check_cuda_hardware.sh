#!/usr/bin/env bash
# Enforce the default physical-card floor for paid NVIDIA validation.

set -euo pipefail

GPU_NAME="${1:-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)}"
GPU_COMPUTE_CAPABILITY="${2:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | xargs)}"
VRAM_MIB="${3:-$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | xargs)}"

MIN_COMPUTE_CAPABILITY="${NEOETHOS_MIN_COMPUTE_CAPABILITY:-86}"
MIN_VRAM_MIB="${NEOETHOS_MIN_VRAM_MIB:-24000}"
ALLOW_UNSUPPORTED="${NEOETHOS_ALLOW_OTHER_GPU:-0}"

SUPPORTED_GPU_SUBSTRINGS=("RTX 3090" "RTX A6000")
if [[ -n "${NEOETHOS_EXPECT_GPU_SUBSTRING:-}" ]]; then
  # Benchmark sessions may intentionally narrow the default correctness-card
  # policy, for example to A6000 plus NEOETHOS_MIN_VRAM_MIB=45000.
  SUPPORTED_GPU_SUBSTRINGS=("$NEOETHOS_EXPECT_GPU_SUBSTRING")
fi

if [[ "$ALLOW_UNSUPPORTED" == "1" ]]; then
  printf 'WARNING: explicit NEOETHOS_ALLOW_OTHER_GPU=1 bypasses name/CC/VRAM policy for %s (CC %s, %s MiB)\n' \
    "$GPU_NAME" "$GPU_COMPUTE_CAPABILITY" "$VRAM_MIB" >&2
  exit 0
fi

name_supported=0
for supported in "${SUPPORTED_GPU_SUBSTRINGS[@]}"; do
  if [[ "$GPU_NAME" == *"$supported"* ]]; then
    name_supported=1
    break
  fi
done
if (( name_supported == 0 )); then
  printf 'Unsupported GPU: %s (accepted by default: %s; explicit override: NEOETHOS_ALLOW_OTHER_GPU=1)\n' \
    "$GPU_NAME" "${SUPPORTED_GPU_SUBSTRINGS[*]}" >&2
  exit 21
fi

if [[ ! "$GPU_COMPUTE_CAPABILITY" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
  printf 'Unknown NVIDIA compute capability for %s: %s\n' \
    "$GPU_NAME" "$GPU_COMPUTE_CAPABILITY" >&2
  exit 28
fi
compute_capability_code=$((10 * BASH_REMATCH[1] + BASH_REMATCH[2]))
if [[ ! "$MIN_COMPUTE_CAPABILITY" =~ ^[0-9]+$ ]] \
  || (( compute_capability_code < MIN_COMPUTE_CAPABILITY )); then
  printf 'Insufficient compute capability: %s (code %s) < %s\n' \
    "$GPU_COMPUTE_CAPABILITY" "$compute_capability_code" "$MIN_COMPUTE_CAPABILITY" >&2
  exit 28
fi

if [[ ! "$MIN_VRAM_MIB" =~ ^[0-9]+$ ]]; then
  printf 'NEOETHOS_MIN_VRAM_MIB must be an integer, got: %s\n' "$MIN_VRAM_MIB" >&2
  exit 64
fi
if [[ ! "$VRAM_MIB" =~ ^[0-9]+$ ]] || (( VRAM_MIB < MIN_VRAM_MIB )); then
  printf 'Insufficient physical VRAM: %s MiB < %s MiB\n' "$VRAM_MIB" "$MIN_VRAM_MIB" >&2
  exit 22
fi

printf 'CUDA hardware accepted: %s, compute capability %s, %s MiB VRAM\n' \
  "$GPU_NAME" "$GPU_COMPUTE_CAPABILITY" "$VRAM_MIB"
