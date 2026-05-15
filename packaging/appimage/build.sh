#!/usr/bin/env bash
# packaging/appimage/build.sh
#
# Build forex-app as a portable AppImage. Zero paid certificates required —
# AppImages are GPG-signed via `appimagetool -s` per the no-paid-certs
# strategy spec §3 (Linux AppImage path).
#
# Spec:
#   - AppImage packaging guide: https://docs.appimage.org/packaging-guide/manual.html
#   - appimagetool: https://github.com/AppImage/appimagetool
#   - Strategy doc: docs/audits/research/installer_no_paid_certs_strategy.md §3
#
# Outputs:
#   forex-app-<version>-x86_64.AppImage         (the AppImage bundle)
#   forex-app-<version>-x86_64.AppImage.asc     (detached GPG sig, when GPG_KEY_ID set)
#   forex-app-<version>-x86_64.AppImage.zsync   (delta-update manifest, optional)
#
# Required tools (PATH):
#   - cargo (Rust toolchain)
#   - appimagetool (downloaded automatically into /tmp if missing)
#   - gpg (only if GPG_KEY_ID is set for signing)
#
# Environment variables:
#   GPG_KEY_ID            (optional) — GPG key fingerprint for `appimagetool -s`.
#                                       TODO(release-time): set in CI from
#                                       secrets.GPG_PRIVATE_KEY after import.
#   APPIMAGETOOL_SIGN_PASSPHRASE (optional) — passphrase for the GPG key, per
#                                              appimagetool docs.
#   FOREX_AI_VERSION      (optional) — version string; defaults to the
#                                       `version =` line in crates/forex-app/Cargo.toml.

set -euo pipefail

# Resolve repo root from this script's location so the script can be invoked
# from anywhere.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
APPDIR="${SCRIPT_DIR}/forex-app.AppDir"

cd "${REPO_ROOT}"

# ── Version detection ────────────────────────────────────────────────────────
if [[ -z "${FOREX_AI_VERSION:-}" ]]; then
    FOREX_AI_VERSION="$(grep -m1 '^version = ' crates/forex-app/Cargo.toml \
        | sed -E 's/version = "([^"]+)"/\1/')"
fi
echo "[appimage] version = ${FOREX_AI_VERSION}"

# ── Step 1: cargo build --release -p forex-app ───────────────────────────────
echo "[appimage] step 1/4 — cargo build --release -p forex-app"
cargo build --release -p forex-app

# ── Step 2: stage the binary inside the AppDir ───────────────────────────────
echo "[appimage] step 2/4 — staging binary into AppDir"
install -Dm755 \
    "${REPO_ROOT}/target/release/forex-app" \
    "${APPDIR}/usr/bin/forex-app"

# Stage runtime assets per installer_infrastructure_spec.md §8.
install -Dm644 \
    "${REPO_ROOT}/assets/symbol_metadata/defaults.json" \
    "${APPDIR}/usr/share/forex-ai/symbol_metadata/defaults.json"

# Top-level icon expected by appimagetool. Use the real icon if it exists,
# otherwise fail-loudly so the operator can't ship a no-icon AppImage by
# accident.
if [[ -f "${APPDIR}/forex-app.png" ]]; then
    echo "[appimage] using existing forex-app.png"
else
    echo "[appimage] ERROR: ${APPDIR}/forex-app.png missing — see forex-app.png.TODO" >&2
    echo "[appimage] (TODO(real-icon): replace placeholder with a 256x256 PNG)" >&2
    exit 1
fi

# Ensure AppRun is executable (git may strip the +x bit on Windows hosts).
chmod 0755 "${APPDIR}/AppRun"

# ── Step 3: download appimagetool if missing ─────────────────────────────────
APPIMAGETOOL="${APPIMAGETOOL:-/tmp/appimagetool-x86_64.AppImage}"
if [[ ! -x "${APPIMAGETOOL}" ]]; then
    echo "[appimage] step 3a/4 — fetching appimagetool"
    # Continuous-release URL per appimagetool README; pinned mirror would be
    # an enhancement, but the project deliberately ships only this URL.
    curl -fL -o "${APPIMAGETOOL}" \
        "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"
    chmod +x "${APPIMAGETOOL}"
fi

# ── Step 4: build the AppImage (with optional GPG signing) ───────────────────
OUT_NAME="forex-app-${FOREX_AI_VERSION}-x86_64.AppImage"
OUT_PATH="${REPO_ROOT}/${OUT_NAME}"

echo "[appimage] step 4/4 — running appimagetool"
APPIMAGETOOL_ARGS=()
if [[ -n "${GPG_KEY_ID:-}" ]]; then
    # `-s --sign-key=<id>` signs the inner SquashFS image. The detached .asc
    # is generated separately in CI after the AppImage is written.
    APPIMAGETOOL_ARGS+=("-s" "--sign-key=${GPG_KEY_ID}")
    echo "[appimage]   signing with GPG key ${GPG_KEY_ID}"
else
    echo "[appimage]   WARNING: GPG_KEY_ID not set — AppImage will be UNSIGNED."
    echo "[appimage]   (Release CI sets GPG_KEY_ID from the imported GPG_PRIVATE_KEY secret.)"
fi

"${APPIMAGETOOL}" "${APPIMAGETOOL_ARGS[@]}" "${APPDIR}" "${OUT_PATH}"

# ── Detached .asc next to the AppImage (idempotent, ignored if no GPG_KEY_ID) ─
if [[ -n "${GPG_KEY_ID:-}" ]]; then
    echo "[appimage] writing detached signature ${OUT_PATH}.asc"
    gpg --detach-sign --armor --local-user "${GPG_KEY_ID}" "${OUT_PATH}"
fi

echo "[appimage] done — ${OUT_PATH}"
