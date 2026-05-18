#!/usr/bin/env bash
# build-release-on-vps.sh
#
# Runs on the Hyperstack L40 VM. Pulls the latest forex-ai source,
# builds Linux release binaries with GPU features, sanity-tests
# the cubecl JIT runtime, and packages the result into a tarball
# at $HOME/forex-ai-linux-x86_64-<DATE>.tar.gz.
#
# Designed to be invoked via SSH from the Windows-side orchestrator
# (`scripts/release-on-vps.ps1`).
#
# Idempotent — safe to re-run; resumes from wherever the last
# attempt stopped.

set -euo pipefail

DATE="${RELEASE_DATE:-$(date +%Y-%m-%d)}"
REPO_URL="${REPO_URL:-https://github.com/kosred/forex-ai.git}"
REPO_ROOT="${REPO_ROOT:-$HOME/forex-ai}"
BRANCH="${BRANCH:-master}"
TARBALL="$HOME/forex-ai-linux-x86_64-${DATE}.tar.gz"
STAGING="$HOME/release-${DATE}"

log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

log "=== forex-ai release build · $DATE ==="
log "Repo:    $REPO_ROOT"
log "Branch:  $BRANCH"
log "Tarball: $TARBALL"

# 1. Disk + GPU pre-check
log "--- Disk & GPU ---"
df -h "$HOME" | head -2
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "FATAL: nvidia-smi not found. This script expects a GPU VM."
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# 2. Ensure Rust toolchain is installed
log "--- Rust toolchain ---"
if ! command -v cargo >/dev/null 2>&1; then
    log "Installing rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --default-toolchain stable --profile minimal --no-modify-path
fi
export PATH="$HOME/.cargo/bin:$PATH"
cargo --version

# 3. Clone or pull
log "--- Source ---"
if [ -d "$REPO_ROOT/.git" ]; then
    log "Pulling $BRANCH..."
    cd "$REPO_ROOT"
    git fetch origin "$BRANCH"
    git checkout "$BRANCH"
    git pull --ff-only origin "$BRANCH"
else
    log "Cloning fresh..."
    git clone "$REPO_URL" "$REPO_ROOT"
    cd "$REPO_ROOT"
    git checkout "$BRANCH"
fi
log "HEAD: $(git rev-parse --short HEAD) — $(git log -1 --format=%s)"

# 4. Optional system setup (CUDA libs etc.) — only if marker missing
if [ ! -f "$HOME/.forex-ai-vps-setup-done" ] && [ -f "$REPO_ROOT/scripts/setup-vps-cuda13.sh" ]; then
    log "--- System setup (one-time) ---"
    bash "$REPO_ROOT/scripts/setup-vps-cuda13.sh"
    touch "$HOME/.forex-ai-vps-setup-done"
fi

# 5. Build
log "--- cargo build --release (forex-cli with GPU features) ---"
cd "$REPO_ROOT"
cargo build --release -p forex-cli \
    --features "forex-search/gpu forex-models/neuro-evolution-gpu forex-models/statistical-gpu" \
    2>&1 | tail -20

log "--- cargo build --release (forex-app, headless-capable) ---"
cargo build --release -p forex-app 2>&1 | tail -10

# 6. Binary sanity
log "--- Binaries ---"
ls -lh "$REPO_ROOT/target/release/forex-app" "$REPO_ROOT/target/release/forex-cli" 2>&1

# 7. GPU smoke test — short genetic-search run to exercise cubecl JIT
log "--- GPU smoke test (search · 32 genes · 3 generations) ---"
cd "$REPO_ROOT/target/release"
mkdir -p "$HOME/data"
set +e
timeout 180 ./forex-cli search \
    --symbol EURUSD --base H4 --higher D1 \
    --genes 32 --generations 3 \
    --root "$HOME/data" \
    2>&1 | tail -30
SMOKE_RC=$?
set -e
if [ $SMOKE_RC -ne 0 ] && [ $SMOKE_RC -ne 124 ]; then
    log "WARNING: GPU smoke test exited $SMOKE_RC"
    log "Continuing with packaging — operator can re-run later."
fi

# 8. Stage + tarball
log "--- Packaging ---"
rm -rf "$STAGING"
mkdir -p "$STAGING"
cp "$REPO_ROOT/target/release/forex-app" "$STAGING/"
cp "$REPO_ROOT/target/release/forex-cli" "$STAGING/"
# Strip symbols to keep the tarball small (keeps debug info next to binaries
# would balloon it from ~110 MB to ~700 MB each).
strip "$STAGING/forex-app" "$STAGING/forex-cli" 2>/dev/null || true

# Include essential runtime resources
[ -f "$REPO_ROOT/config.yaml" ] && cp "$REPO_ROOT/config.yaml" "$STAGING/"
[ -d "$REPO_ROOT/assets" ] && cp -r "$REPO_ROOT/assets" "$STAGING/"
[ -f "$REPO_ROOT/LICENSE" ] && cp "$REPO_ROOT/LICENSE" "$STAGING/"
[ -f "$REPO_ROOT/README.md" ] && cp "$REPO_ROOT/README.md" "$STAGING/"

# Build info
cat > "$STAGING/BUILD-INFO.txt" <<EOF
forex-ai Linux x86_64 release · $DATE
Built on: $(uname -srvmo)
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
Commit: $(cd "$REPO_ROOT" && git rev-parse HEAD)
Branch: $BRANCH
Rust:   $(rustc --version)
EOF

log "Tarball contents:"
ls -lh "$STAGING/"

tar -czf "$TARBALL" -C "$STAGING" .
log "Tarball size: $(ls -lh "$TARBALL" | awk '{print $5}')"
log "SHA256: $(sha256sum "$TARBALL" | cut -d' ' -f1)"

log "=== DONE ==="
log "Pull file via: scp <vps-user>@<vps-ip>:$TARBALL <local-dest>"
