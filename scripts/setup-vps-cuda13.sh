#!/usr/bin/env bash
# One-shot VPS bootstrap for forex-ai on Ubuntu 22.04 with NVIDIA L40 / A100 / RTX-A6000 / H100.
#
# What this does (in order):
#   1. Full system update (apt update + dist-upgrade)
#   2. Install build deps (cmake>=3.28 via pip, clang, libstdc++-12-dev, X11/Wayland libs, etc.)
#   3. Install NVIDIA driver 595 + CUDA toolkit 13.0 (matches our libtorch cu130 release)
#   4. Download libtorch 2.9.0+cu130 (~3 GB) and unpack into ~/libtorch
#   5. Install rustup with stable toolchain
#   6. Pull the real EURUSD/USDJPY/etc. dataset (~672 MB) into ~/data
#   7. Clone the repo at the requested branch
#   8. Persist all env vars in ~/.bashrc so subsequent shells just work
#   9. Reboot if a new kernel/driver was installed
#
# Total runtime on a fresh L40 VM: ~10-15 min. Disk usage: ~12 GB.
#
# Re-run safe: every step is idempotent — it'll skip what's already installed.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/kosred/forex-ai/master/scripts/setup-vps-cuda13.sh | bash
# OR clone first then:
#   bash scripts/setup-vps-cuda13.sh

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/kosred/forex-ai.git}"
REPO_BRANCH="${REPO_BRANCH:-claude/happy-gould-23d649}"
DATA_RELEASE_URL="${DATA_RELEASE_URL:-https://github.com/kosred/forex-ai/releases/download/dataset-v1/data.zip}"
LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.9.0}"
LIBTORCH_CUDA="${LIBTORCH_CUDA:-cu130}"
NVIDIA_DRIVER_PKG="${NVIDIA_DRIVER_PKG:-nvidia-driver-595-server-open}"
CUDA_TOOLKIT_PKG="${CUDA_TOOLKIT_PKG:-cuda-toolkit-13-0}"

HOME_DIR="$HOME"
LIBTORCH_DIR="$HOME_DIR/libtorch"
DATA_DIR="$HOME_DIR/data"
REPO_DIR="$HOME_DIR/forex-ai"

REBOOT_NEEDED=0

log() { printf '\n\033[1;32m▶ %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33m! %s\033[0m\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

if [[ "$EUID" -eq 0 ]]; then
  echo "Please run as a regular user with sudo, not as root." >&2
  exit 1
fi

log "Step 1/9 — apt update + dist-upgrade"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get -y -qq -o Dpkg::Options::="--force-confnew" dist-upgrade

log "Step 2/9 — build dependencies (clang, libstdc++-12-dev, X11/audio dev libs, python3-pip)"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  build-essential pkg-config curl unzip git \
  clang libclang-dev libstdc++-12-dev libssl-dev \
  libasound2-dev libudev-dev \
  libxcb-shape0-dev libxcb-xfixes0-dev \
  libxkbcommon-dev libxkbcommon-x11-dev \
  libx11-dev libxrandr-dev libxcursor-dev \
  libxi-dev libgl1-mesa-dev \
  python3-pip dpkg-dev

# Ubuntu 22.04 ships cmake 3.22; lightgbm-sys requires >= 3.28.
if ! cmake --version 2>/dev/null | grep -qE 'cmake version (3\.(2[89]|[3-9][0-9])|[4-9])'; then
  log "  Installing cmake>=3.28 via pip (Ubuntu 22.04's apt cmake is too old)"
  pip3 install --user --quiet 'cmake>=3.28'
fi
# Make sure ~/.local/bin is on PATH for current and future shells.
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) export PATH="$HOME/.local/bin:$PATH" ;;
esac

log "Step 3/9 — NVIDIA driver $NVIDIA_DRIVER_PKG + CUDA toolkit $CUDA_TOOLKIT_PKG"
if ! dpkg -s "$NVIDIA_DRIVER_PKG" >/dev/null 2>&1; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$NVIDIA_DRIVER_PKG"
  REBOOT_NEEDED=1
else
  log "  Driver $NVIDIA_DRIVER_PKG already installed."
fi
if ! dpkg -s "$CUDA_TOOLKIT_PKG" >/dev/null 2>&1; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$CUDA_TOOLKIT_PKG"
else
  log "  Toolkit $CUDA_TOOLKIT_PKG already installed."
fi

log "Step 4/9 — libtorch $LIBTORCH_VERSION+$LIBTORCH_CUDA (~3 GB)"
if [[ ! -f "$LIBTORCH_DIR/lib/libtorch.so" ]]; then
  cd "$HOME_DIR"
  url="https://download.pytorch.org/libtorch/$LIBTORCH_CUDA/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2B${LIBTORCH_CUDA}.zip"
  log "  Downloading $url"
  curl -fsSL --retry 3 -o libtorch.zip "$url"
  unzip -q libtorch.zip
  rm libtorch.zip
else
  log "  $LIBTORCH_DIR already present, skipping download."
fi

log "Step 5/9 — Rust stable toolchain"
if ! have rustup; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal -q
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"

log "Step 6/9 — dataset (~672 MB → $DATA_DIR)"
mkdir -p "$DATA_DIR"
if [[ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]]; then
  cd "$HOME_DIR"
  log "  Downloading $DATA_RELEASE_URL"
  curl -fsSL --retry 3 -o data.zip "$DATA_RELEASE_URL"
  log "  Extracting"
  unzip -q data.zip -d "$DATA_DIR"
  rm data.zip
  # Some releases nest a top-level data/ dir; flatten if so.
  if [[ -d "$DATA_DIR/data" ]] && [[ -z "$(ls "$DATA_DIR" | grep -v '^data$' || true)" ]]; then
    mv "$DATA_DIR/data"/* "$DATA_DIR/"
    rmdir "$DATA_DIR/data"
  fi
else
  log "  $DATA_DIR already populated, skipping download."
fi

log "Step 7/9 — clone forex-ai @ $REPO_BRANCH"
if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone --branch "$REPO_BRANCH" --depth 1 "$REPO_URL" "$REPO_DIR"
else
  cd "$REPO_DIR"
  git fetch origin "$REPO_BRANCH"
  git checkout "$REPO_BRANCH"
  git pull --rebase --autostash || true
fi

log "Step 8/9 — persist env vars in ~/.bashrc"
SETUP_MARK="# --- forex-ai vps env (managed by setup-vps-cuda13.sh) ---"
if ! grep -qF "$SETUP_MARK" "$HOME/.bashrc"; then
  cat >> "$HOME/.bashrc" <<EOF

$SETUP_MARK
export LIBTORCH="$LIBTORCH_DIR"
export TORCH_CUDA_VERSION=$LIBTORCH_CUDA
export PATH="\$HOME/.local/bin:/usr/local/cuda-${CUDA_TOOLKIT_PKG##*-}.0/bin:\$PATH"
export LD_LIBRARY_PATH="\$LIBTORCH/lib:/usr/local/cuda-${CUDA_TOOLKIT_PKG##*-}.0/lib64:\$HOME/forex-ai/target/release:\$HOME/forex-ai/target/release/deps:\${LD_LIBRARY_PATH:-}"
export FOREX_BOT_DATA_ROOT="$DATA_DIR"
# --- end forex-ai vps env ---
EOF
  log "  Appended env block to ~/.bashrc"
else
  log "  ~/.bashrc already contains the forex-ai env block."
fi

log "Step 9/9 — done"
cat <<EOF

================================================================
Bootstrap complete.
================================================================
Repo:         $REPO_DIR    (branch: $REPO_BRANCH)
Dataset:      $DATA_DIR
libtorch:     $LIBTORCH_DIR
Driver:       \$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "(needs reboot)")

Next steps:
  source ~/.bashrc           # load env vars in this shell
  cd $REPO_DIR
  cargo build --release -p forex-cli --features "forex-search/gpu forex-models/neuro-evolution-gpu forex-models/statistical-gpu"
  ./target/release/forex-cli migrate-data --root \$FOREX_BOT_DATA_ROOT
  ./target/release/forex-cli search --symbol EURUSD --base H4 --higher D1 --genes 64 --generations 5 --root \$FOREX_BOT_DATA_ROOT

EOF

if [[ $REBOOT_NEEDED -eq 1 ]]; then
  warn "A new NVIDIA driver was installed. REBOOT NOW so the kernel module loads:"
  warn "    sudo reboot"
fi
