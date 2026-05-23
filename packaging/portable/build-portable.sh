#!/usr/bin/env bash
# packaging/portable/build-portable.sh
#
# The lowest-friction shippable artifact: a plain portable tarball that
# unpacks anywhere and runs. Zero packaging tooling required beyond
# `cargo` and `tar` — used as the v0.4.10 ship-gate §5.1.7 "at least
# one of (.deb, .AppImage, .tar.gz)" check, and as the fallback path
# when AppImage / cargo-deb tooling is unavailable.
#
# Outputs:
#   dist/neoethos-<version>-<os>-<arch>-portable.tar.gz
#
# Layout inside the tarball:
#   neoethos-<version>/neoethos-app        (release binary)
#   neoethos-<version>/README.md           (top-level project README)
#   neoethos-<version>/LICENSE             (license file)
#   neoethos-<version>/config.yaml         (default config, copy-on-edit)
#
# Required tools: cargo, tar (BSD or GNU). On macOS also `lipo` if you
# want a universal binary — not done here; this script ships the host
# architecture only.
#
# Environment:
#   NEOETHOS_VERSION  (optional)  - version override; defaults to the
#                                   `version =` line in
#                                   crates/neoethos-app/Cargo.toml.
#   NEOETHOS_OUTDIR   (optional)  - output directory; default `dist/`.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ── version detection (same mechanism as appimage/build.sh) ────────────
if [[ -z "${NEOETHOS_VERSION:-}" ]]; then
    NEOETHOS_VERSION="$(grep -m1 '^version = ' crates/neoethos-app/Cargo.toml \
        | sed -E 's/version = "([^"]+)"/\1/')"
fi

# ── platform/arch detection ────────────────────────────────────────────
OS_TAG="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "${OS_TAG}" in
    linux*)   OS_TAG="linux" ;;
    darwin*)  OS_TAG="macos" ;;
    msys*|mingw*|cygwin*) OS_TAG="windows" ;;
esac
ARCH_TAG="$(uname -m)"
case "${ARCH_TAG}" in
    arm64|aarch64) ARCH_TAG="aarch64" ;;
    x86_64|amd64)  ARCH_TAG="x86_64" ;;
esac

OUTDIR="${NEOETHOS_OUTDIR:-${REPO_ROOT}/dist}"
STEM="neoethos-${NEOETHOS_VERSION}-${OS_TAG}-${ARCH_TAG}-portable"
TARBALL="${OUTDIR}/${STEM}.tar.gz"

echo "[portable] version=${NEOETHOS_VERSION} os=${OS_TAG} arch=${ARCH_TAG}"
echo "[portable] output → ${TARBALL}"

# ── Step 1: release build ──────────────────────────────────────────────
echo "[portable] step 1/3 — cargo build --release -p neoethos-app"
cargo build --release -p neoethos-app

# ── Step 2: stage the bundle in a clean dir ────────────────────────────
STAGING="$(mktemp -d -t neoethos-portable-XXXXXX)"
trap 'rm -rf "${STAGING}"' EXIT

INNER="${STAGING}/neoethos-${NEOETHOS_VERSION}"
mkdir -p "${INNER}"

# Binary name differs on Windows.
BIN_NAME="neoethos-app"
[[ "${OS_TAG}" == "windows" ]] && BIN_NAME="neoethos-app.exe"

cp "target/release/${BIN_NAME}" "${INNER}/"
cp README.md "${INNER}/" 2>/dev/null || echo "[portable] (no README.md to bundle — ok)"
cp LICENSE "${INNER}/"   2>/dev/null || echo "[portable] (no LICENSE to bundle — ok)"
cp config.yaml "${INNER}/" 2>/dev/null || echo "[portable] (no config.yaml to bundle — ok)"

# ── Step 3: package ────────────────────────────────────────────────────
mkdir -p "${OUTDIR}"
echo "[portable] step 3/3 — tar czf ${TARBALL}"
tar czf "${TARBALL}" -C "${STAGING}" "neoethos-${NEOETHOS_VERSION}"

echo "[portable] ✓ done"
echo "[portable]   ${TARBALL}"
echo "[portable]   $(du -h "${TARBALL}" | cut -f1)  ($(wc -c < "${TARBALL}") bytes)"
