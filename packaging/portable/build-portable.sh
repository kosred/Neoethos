#!/usr/bin/env bash
# packaging/portable/build-portable.sh
#
# The lowest-friction shippable artifact: a plain portable tarball that
# unpacks anywhere and runs. Zero packaging tooling required beyond
# `cargo` and `tar` — used as the v0.4.5 ship-gate §5.1.7 "at least
# one of (.deb, .AppImage, .tar.gz)" check, and as the fallback path
# when AppImage / cargo-deb tooling is unavailable.
#
# Outputs:
#   dist/forex-ai-<version>-<os>-<arch>-portable.tar.gz
#
# Layout inside the tarball:
#   forex-ai-<version>/forex-app           (release binary)
#   forex-ai-<version>/README.md           (top-level project README)
#   forex-ai-<version>/LICENSE             (Apache-2.0)
#   forex-ai-<version>/config.yaml         (default config, copy-on-edit)
#
# Required tools: cargo, tar (BSD or GNU). On macOS also `lipo` if you
# want a universal binary — not done here; this script ships the host
# architecture only.
#
# Environment:
#   FOREX_AI_VERSION  (optional)  — version override; defaults to the
#                                   `version =` line in
#                                   crates/forex-app/Cargo.toml.
#   FOREX_AI_OUTDIR   (optional)  — output directory; default `dist/`.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ── version detection (same mechanism as appimage/build.sh) ────────────
if [[ -z "${FOREX_AI_VERSION:-}" ]]; then
    FOREX_AI_VERSION="$(grep -m1 '^version = ' crates/forex-app/Cargo.toml \
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

OUTDIR="${FOREX_AI_OUTDIR:-${REPO_ROOT}/dist}"
STEM="forex-ai-${FOREX_AI_VERSION}-${OS_TAG}-${ARCH_TAG}-portable"
TARBALL="${OUTDIR}/${STEM}.tar.gz"

echo "[portable] version=${FOREX_AI_VERSION} os=${OS_TAG} arch=${ARCH_TAG}"
echo "[portable] output → ${TARBALL}"

# ── Step 1: release build ──────────────────────────────────────────────
echo "[portable] step 1/3 — cargo build --release -p forex-app"
cargo build --release -p forex-app

# ── Step 2: stage the bundle in a clean dir ────────────────────────────
STAGING="$(mktemp -d -t forex-ai-portable-XXXXXX)"
trap 'rm -rf "${STAGING}"' EXIT

INNER="${STAGING}/forex-ai-${FOREX_AI_VERSION}"
mkdir -p "${INNER}"

# Binary name differs on Windows.
BIN_NAME="forex-app"
[[ "${OS_TAG}" == "windows" ]] && BIN_NAME="forex-app.exe"

cp "target/release/${BIN_NAME}" "${INNER}/"
cp README.md "${INNER}/" 2>/dev/null || echo "[portable] (no README.md to bundle — ok)"
cp LICENSE "${INNER}/"   2>/dev/null || echo "[portable] (no LICENSE to bundle — ok)"
cp config.yaml "${INNER}/" 2>/dev/null || echo "[portable] (no config.yaml to bundle — ok)"

# ── Step 3: package ────────────────────────────────────────────────────
mkdir -p "${OUTDIR}"
echo "[portable] step 3/3 — tar czf ${TARBALL}"
tar czf "${TARBALL}" -C "${STAGING}" "forex-ai-${FOREX_AI_VERSION}"

echo "[portable] ✓ done"
echo "[portable]   ${TARBALL}"
echo "[portable]   $(du -h "${TARBALL}" | cut -f1)  ($(wc -c < "${TARBALL}") bytes)"
