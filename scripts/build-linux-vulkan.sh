#!/usr/bin/env bash
#
# scripts/build-linux-vulkan.sh
#
# Build the NeoEthos backend + CLI on Linux with the gpu-vulkan backend and
# package a portable tarball (headless server + TUI). Mirror of the Windows
# build-cargo-release.ps1 + make-release-bundle.ps1 flow.
#
# Produces:  dist/neoethos-<ver>-linux-x64-vulkan.tar.gz
#
# ---------------------------------------------------------------------------
# PREREQUISITES (apt; run once). The Rust crates pull a chain of native libs:
#
#   sudo apt-get update
#   sudo apt-get install -y \
#     build-essential cmake clang libclang-dev pkg-config \
#     libwayland-dev wayland-protocols libwayland-bin libxkbcommon-dev \
#     libxkbcommon-x11-dev libvulkan-dev libx11-dev libxcursor-dev \
#     libxrandr-dev libxi-dev libgl1-mesa-dev libxcb1-dev \
#     libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev \
#     libgtk-3-dev libdbus-1-dev libudev-dev
#
#   rustup component add rustfmt        # catboost-rust build.rs needs it
#
# Why each matters (hard-won, 2026-06-02):
#   * wayland / xcb / xkbcommon : neoethos-app links the legacy egui->winit tree.
#   * dbus / udev               : clipboard + device crates (libdbus-sys).
#   * vulkan (libvulkan-dev)    : gpu-vulkan -> wgpu -> ash (loads libvulkan at runtime).
#   * rustfmt                   : catboost-rust 0.3.8 build.rs formats bindings.
#
# GOTCHA: catboost-rust 0.3.8 build.rs does
#   out_dir.ancestors().find(|p| p.ends_with("target")).unwrap()
# i.e. it REQUIRES a path component named exactly "target". The repo default
# target dir (./target) satisfies this. If you override CARGO_TARGET_DIR, its
# FINAL component MUST be "target" (e.g. ~/build/target, NOT ~/neoethos-target).
# Also: do NOT `mv` a populated target dir to a new path then reuse it — build
# scripts bake absolute -L paths; a clean build is required after a move.
# ---------------------------------------------------------------------------

set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || { echo "cannot cd to repo root"; exit 1; }

VER="$(grep -m1 '^version' crates/neoethos-app/Cargo.toml | sed 's/.*"\(.*\)".*/\1/')"
NAME="neoethos-${VER}-linux-x64-vulkan"

# gpu-vulkan build.rs guard only checks that VULKAN_SDK is *set*; on Linux the
# loader/headers come from libvulkan-dev under /usr.
export VULKAN_SDK="${VULKAN_SDK:-/usr}"
# Help bindgen find libclang if it is not auto-detected.
if [ -z "${LIBCLANG_PATH:-}" ]; then
  for d in /usr/lib/llvm-*/lib /usr/lib/x86_64-linux-gnu; do
    [ -e "$d/libclang.so" ] || [ -e "$d/libclang.so.1" ] && { export LIBCLANG_PATH="$d"; break; }
  done
fi

echo "== NeoEthos $VER Linux gpu-vulkan build =="
echo "   repo=$REPO  VULKAN_SDK=$VULKAN_SDK  LIBCLANG_PATH=${LIBCLANG_PATH:-<auto>}"
echo "   rustc=$(rustc --version)"

cargo build --release -p neoethos-app --features gpu-vulkan || { echo "neoethos-app build FAILED"; exit 1; }
cargo build --release -p neoethos-cli                       || { echo "neoethos-cli build FAILED"; exit 1; }

REL="$REPO/target/release"
STAGE="$REPO/dist/$NAME"
rm -rf "$STAGE"; mkdir -p "$STAGE/bin"
cp "$REL/neoethos-app" "$STAGE/bin/"
cp "$REL/neoethos-cli" "$STAGE/bin/" 2>/dev/null || true
cp "$REPO/config.yaml" "$STAGE/config.yaml" 2>/dev/null || true

# Native ML sidecars are dynamic libs (the Linux mirror of catboostmodel.dll).
# xgboost lands in target/release/deps; catboost under build/.../out/libs.
for so in \
  "$REL/deps/libxgboost.so" \
  $(find "$REL/build" -name 'libcatboostmodel.so' 2>/dev/null) ; do
  [ -f "$so" ] && cp "$so" "$STAGE/bin/" && echo "   sidecar: $(basename "$so")"
done

cat > "$STAGE/run-backend.sh" <<'LAUNCH'
#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$DIR/bin:${LD_LIBRARY_PATH:-}"
exec "$DIR/bin/neoethos-app" --server --config "$DIR/config.yaml" "$@"
LAUNCH
chmod +x "$STAGE/run-backend.sh"

{
  echo "NeoEthos $VER Linux x64 (gpu-vulkan) - runtime deps"
  echo "uname: $(uname -a)"; echo "glibc: $(ldd --version | head -1)"; echo
  echo "== neoethos-app (LD_LIBRARY_PATH=bin) =="
  LD_LIBRARY_PATH="$STAGE/bin" ldd "$STAGE/bin/neoethos-app" 2>&1
} > "$STAGE/RUNTIME-DEPS.txt"

cat > "$STAGE/README.txt" <<EOF
NeoEthos $VER - Linux x64 (gpu-vulkan)

Run the backend:   ./run-backend.sh         (sets LD_LIBRARY_PATH for bin/*.so)
Run the TUI:       ./bin/neoethos-cli

Built against $(ldd --version | head -1). Won't run on an older glibc.
The GA discovery kernel is CUDA-only, so under Vulkan the heavy search runs on
CPU (Vulkan accelerates the Burn ML models only). Full GPU discovery = CUDA on
the L40 VPS.
EOF

tar -C "$REPO/dist" -czf "$REPO/dist/${NAME}.tar.gz" "$NAME"
echo "== packaged: $REPO/dist/${NAME}.tar.gz ($(du -h "$REPO/dist/${NAME}.tar.gz" | cut -f1)) =="
