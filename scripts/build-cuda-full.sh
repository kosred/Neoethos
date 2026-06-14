#!/usr/bin/env bash
# =============================================================================
# build-cuda-full.sh — the ONE correct way to build neoethos-cli with the FULL
# GPU stack on a Hyperstack (Ubuntu 22.04) NVIDIA box. Idempotent: re-running
# skips already-done steps. Captures EVERY gotcha learned the hard way so we
# never fumble the build again.
#
# What "FULL / everything enabled" means here:
#   - cubecl-CUDA backend (GA eval on GPU)        [feature gpu-nvidia]
#   - libtorch 2.9.0+cu130 (tch device enum + ML) [feature gpu-nvidia]
#   - lightgbm + candle + ort CUDA                [feature gpu-nvidia]
#   - vector-ta nightly-AVX (AVX2/AVX512 TA on CPU, NOT SSE2) [always, nightly]
#   - vector-ta CUDA kernels (compute_86)         [OPT-IN: BUILD_VTA_CUDA=1 — see note]
#
# Usage (run as a normal user with sudo, from the repo root):
#   bash scripts/build-cuda-full.sh
#   BUILD_VTA_CUDA=1 bash scripts/build-cuda-full.sh   # also compile vector-ta CUDA kernels
#
# Target GPU arch defaults to the RTX A6000 (Ampere, sm_86). Override:
#   CUDA_ARCH=89 bash scripts/build-cuda-full.sh       # RTX 40xx (Ada)
#   CUDA_ARCH=80 ...                                    # A100
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBTORCH_DIR="${LIBTORCH_DIR:-$HOME/libtorch}"
LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.9.0}"   # MUST match tch in Cargo.toml (tch 0.22 -> 2.9.0)
LIBTORCH_CUDA="${LIBTORCH_CUDA:-cu130}"          # cu130 needs NVIDIA driver >= 580
CUDA_TOOLKIT="${CUDA_TOOLKIT:-/usr/local/cuda-12.2}"  # nvcc/nvrtc + headers (cubecl + vector-ta)
CUDA_ARCH="${CUDA_ARCH:-86}"                     # A6000 = sm_86; 89=Ada, 80=A100, 90=Hopper
BUILD_VTA_CUDA="${BUILD_VTA_CUDA:-0}"

log() { printf '\n\033[1;32m▶ %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33m! %s\033[0m\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

[[ "$EUID" -eq 0 ]] && { echo "Run as a normal user with sudo, not root." >&2; exit 1; }

# -----------------------------------------------------------------------------
log "1/7 — NVIDIA driver check (need >= 580 for libtorch ${LIBTORCH_CUDA})"
if ! nvidia-smi >/dev/null 2>&1; then
  warn "nvidia-smi failed. If you JUST upgraded the driver, REBOOT first (DKMS module mismatch)."
  exit 1
fi
DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
echo "driver major = $DRV"
if [[ "${LIBTORCH_CUDA}" == "cu13"* && "$DRV" -lt 580 ]]; then
  warn "Driver $DRV < 580 — libtorch ${LIBTORCH_CUDA} needs >=580. Install: sudo apt-get install -y nvidia-driver-595-server-open && sudo reboot"
  exit 1
fi

# -----------------------------------------------------------------------------
log "2/7 — apt build deps (incl the non-obvious ones that bit us)"
export DEBIAN_FRONTEND=noninteractive
sudo -E apt-get update -qq
sudo -E apt-get install -y -qq \
  build-essential pkg-config curl unzip git \
  clang libclang-dev libstdc++-12-dev libssl-dev python3-pip dpkg-dev \
  mold \
  libnccl2 libnccl-dev \
  cuda-libraries-13-0 \
  patchelf
# GOTCHA #1: a dist-upgrade can leave binutils half-installed -> `cannot find 'ld'`.
sudo -E apt-get install --reinstall -y -qq binutils binutils-common binutils-x86-64-linux-gnu
# GOTCHA #2 (covered by deps above):
#   - mold            : .cargo/config.toml sets -C link-arg=-fuse-ld=mold (Linux). Missing -> "cannot find 'ld'".
#   - libnccl-dev     : libtorch ${LIBTORCH_CUDA} links -lnccl.
#   - cuda-libraries-13-0 : provides libcufft.so.12 / libcusparse.so.12 / libcusolver.so.12 (CUDA-13 sonames)
#                       that libtorch_cuda.so needs but does NOT bundle (it bundles only cublas/cudart/cudnn .so.13).
echo "ld: $(ld --version | head -1)   mold: $(mold --version | head -1)"

# -----------------------------------------------------------------------------
log "3/7 — cmake>=3.28 (lightgbm needs it; Ubuntu 22.04 ships 3.22) + rustup"
if ! cmake --version 2>/dev/null | grep -qE 'cmake version (3\.(2[89]|[3-9][0-9])|[4-9])'; then
  pip3 install --user --quiet 'cmake>=3.28'
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! have rustup; then
  curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  source "$HOME/.cargo/env"
fi
# NOTE: the repo's rust-toolchain.toml pins NIGHTLY (needed for -Zthreads AND
# vector-ta's nightly-avx AVX2/AVX512 TA kernels). cargo auto-installs it.

# -----------------------------------------------------------------------------
log "4/7 — libtorch ${LIBTORCH_VERSION}+${LIBTORCH_CUDA}"
if [[ ! -d "$LIBTORCH_DIR" ]]; then
  # GOTCHA #3: the Linux file is `libtorch-shared-with-deps-` (NOT `cxx11-abi-`, that 404s).
  #            pytorch.org 403s curl HEAD/range probes — just GET the real filename.
  URL="https://download.pytorch.org/libtorch/${LIBTORCH_CUDA}/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2B${LIBTORCH_CUDA}.zip"
  echo "downloading $URL"
  curl -fL --retry 4 -o /tmp/libtorch.zip "$URL"
  ( cd "$(dirname "$LIBTORCH_DIR")" && unzip -q -o /tmp/libtorch.zip && rm -f /tmp/libtorch.zip )
fi
echo "libtorch: $(du -sh "$LIBTORCH_DIR" | cut -f1) at $LIBTORCH_DIR"

# -----------------------------------------------------------------------------
log "5/7 — build env"
export PATH="${CUDA_TOOLKIT}/bin:$PATH"
export LIBTORCH="$LIBTORCH_DIR"
export LIBTORCH_BYPASS_VERSION_CHECK=1
export CUDA_PATH="$CUDA_TOOLKIT" CUDA_ROOT="$CUDA_TOOLKIT" CUDA_HOME="$CUDA_TOOLKIT"
export CUDACXX="${CUDA_TOOLKIT}/bin/nvcc" NVCC="${CUDA_TOOLKIT}/bin/nvcc"
export CARGO_PROFILE_RELEASE_DEBUG=0
# For vector-ta CUDA kernel compile (only used when BUILD_VTA_CUDA=1):
export CUDA_ARCH="$CUDA_ARCH"            # build.rs: -arch compute_${CUDA_ARCH} (default in crate is compute_89 = Ada-only!)
export NVCC_ARGS="-include float.h"      # GOTCHA #4: 3 kernels use DBL/FLT_EPSILON without <cfloat>

cd "$REPO_DIR"

# gpu-nvidia already pulls cubecl-CUDA + libtorch + lightgbm/candle/ort CUDA.
# vector-ta nightly-avx is ALWAYS on (it's a default feature of the dep in
# crates/neoethos-data/Cargo.toml) — no flag needed.
FEATURES="gpu-nvidia"
if [[ "$BUILD_VTA_CUDA" == "1" ]]; then
  warn "BUILD_VTA_CUDA=1 is a PLACEHOLDER. vector-ta's CUDA TA path is NOT yet"
  warn "wired into hpc_ta (task #22) — there is no 'ta-cuda' feature to enable."
  warn "To VERIFY the 207 kernels compile for compute_${CUDA_ARCH}, build vector-ta"
  warn "standalone (it is a [patch] dep, so copy it out of the workspace first):"
  cat <<EOF
    cp -r vendor/vector-ta-0.2.9-patched /tmp/vta && printf '\n[workspace]\n' >> /tmp/vta/Cargo.toml
    cd /tmp/vta && CUDA_ARCH=${CUDA_ARCH} NVCC=${CUDA_TOOLKIT}/bin/nvcc CUDA_PATH=${CUDA_TOOLKIT} \\
      NVCC_ARGS='-include float.h' cargo build --release --features cuda-build-ptx
EOF
fi

# -----------------------------------------------------------------------------
log "6/7 — cargo build --release -p neoethos-cli --features ${FEATURES}"
cargo build --release -p neoethos-cli --features "$FEATURES"
BIN="$REPO_DIR/target/release/neoethos-cli"
echo "built: $(du -h "$BIN" | cut -f1)  $BIN"

# -----------------------------------------------------------------------------
log "7/7 — RUNTIME requirements (the binary CPU-falls-back silently without these)"
cat <<EOF
The CUDA build needs THREE things at RUNTIME or it silently runs on CPU:

  export CUDA_PATH=${CUDA_TOOLKIT}                          # NVRTC needs cuda_runtime.h (cubecl compiles kernels at runtime)
  export LD_PRELOAD="$LIBTORCH_DIR/lib/libtorch_cuda.so $LIBTORCH_DIR/lib/libc10_cuda.so"
                                                            # mold --as-needed DROPS libtorch_cuda -> tch sees 0 CUDA devices
  export LD_LIBRARY_PATH="$LIBTORCH_DIR/lib:/usr/local/cuda-13.0/targets/x86_64-linux/lib:${CUDA_TOOLKIT}/lib64:\$LD_LIBRARY_PATH"

Then run, e.g.:
  $BIN discover --symbol EURUSD --base H1 --config config.yaml --root data --out /tmp/x.json

For a SELF-CONTAINED, portable bundle (rpath + launcher), see make-cuda-bundle (memory gpu-cuda-build-recipe.md).
Verify the GPU is actually used: nvidia-smi shows mem>1MiB + sm%>0 during eval; logs have NO "0 CUDA devices" / "panicked".
EOF
log "DONE."
