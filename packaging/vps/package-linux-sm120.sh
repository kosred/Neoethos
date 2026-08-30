#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf 'neoethos-vps-release: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command is unavailable: $1"
}

require_nonempty_file() {
  [[ -f "$1" && -s "$1" ]] || fail "required non-empty file is missing: $1"
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

json_escape() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  value="${value//$'\r'/\\r}"
  value="${value//$'\t'/\\t}"
  printf '%s' "${value}"
}

usage() {
  cat >&2 <<'USAGE'
Usage: packaging/vps/package-linux-sm120.sh VERSION [SOURCE_RELEASE] [OUTPUT_DIR]

Packages an already-built Linux `gpu-nvidia-full` app/CLI pair for the exact
Blackwell sm_120 target. It does not compile or modify the source tree.

  VERSION         Release label, for example v0.5.6-rtx5090.1
  SOURCE_RELEASE  Cargo profile directory (default: target/release)
  OUTPUT_DIR      Artifact directory (default: dist)
USAGE
}

[[ $# -ge 1 && $# -le 3 ]] || {
  usage
  exit 2
}

VERSION="$1"
[[ "${VERSION}" =~ ^v?[0-9A-Za-z][0-9A-Za-z._+-]*$ ]] \
  || fail "VERSION contains characters unsafe for an artifact name: ${VERSION}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
SOURCE_RELEASE="${2:-${REPO_ROOT}/target/release}"
[[ -d "${SOURCE_RELEASE}" ]] || fail "Cargo profile directory does not exist: ${SOURCE_RELEASE}"
SOURCE_RELEASE="$(cd -- "${SOURCE_RELEASE}" && pwd -P)"

OUTPUT_DIR="${3:-${REPO_ROOT}/dist}"
mkdir -p -- "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd -- "${OUTPUT_DIR}" && pwd -P)"

case "${SOURCE_RELEASE}" in
  "${REPO_ROOT}"/*) ;;
  *) fail "SOURCE_RELEASE must be inside the current repository: ${SOURCE_RELEASE}" ;;
esac

for command_name in git find grep install cp ln rm chmod readelf ldd realpath sha256sum tar sort awk nvidia-smi python3; do
  require_command "${command_name}"
done

git -C "${REPO_ROOT}" diff --quiet -- \
  || fail "tracked source has unstaged changes; refusing an ambiguous release"
git -C "${REPO_ROOT}" diff --cached --quiet -- \
  || fail "tracked source has staged changes; refusing an ambiguous release"
GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse --verify HEAD)"
[[ "${GIT_SHA}" =~ ^[0-9a-f]{40}$ ]] || fail "git did not return a full source SHA"

require_nonempty_file "${SOURCE_RELEASE}/neoethos-app"
require_nonempty_file "${SOURCE_RELEASE}/neoethos-cli"
require_nonempty_file "${SOURCE_RELEASE}/libxgboost.so"
require_nonempty_file "${SOURCE_RELEASE}/libcatboostmodel.so"
require_nonempty_file "${REPO_ROOT}/config.yaml"
require_nonempty_file "${REPO_ROOT}/assets/symbol_metadata/defaults.json"
grep -Fq '    gpu_only: true' "${REPO_ROOT}/config.yaml" \
  || fail "bundled config does not forbid CPU fallback for GPU tree models"

mapfile -d '' CUDA_BUILD_MANIFESTS < <(
  find "${SOURCE_RELEASE}/build" -type f \
    -path '*/out/neoethos_cuda_build_manifest_v1.json' -print0
)
[[ ${#CUDA_BUILD_MANIFESTS[@]} -eq 1 ]] \
  || fail "expected exactly one CUDA build manifest under ${SOURCE_RELEASE}/build; found ${#CUDA_BUILD_MANIFESTS[@]}"
CUDA_BUILD_MANIFEST="${CUDA_BUILD_MANIFESTS[0]}"
require_nonempty_file "${CUDA_BUILD_MANIFEST}"
for exact_cuda_fact in \
  '"schema":"neoethos.cuda-native-build.v1"' \
  '"architectures":[120]' \
  '"gencode":["--generate-code=arch=compute_120,code=sm_120"]' \
  '"sass_targets":["sm_120"]' \
  '"ptx_targets":[]'; do
  grep -Fq -- "${exact_cuda_fact}" "${CUDA_BUILD_MANIFEST}" \
    || fail "CUDA build manifest is not exact sm_120 SASS with no PTX: missing ${exact_cuda_fact}"
done
CUDA_BUILD_MANIFEST_BYTES="$(<"${CUDA_BUILD_MANIFEST}")"
grep -aFq -- "${CUDA_BUILD_MANIFEST_BYTES}" "${SOURCE_RELEASE}/neoethos-app" \
  || fail "neoethos-app does not embed the selected exact sm_120 CUDA manifest"
grep -aFq -- "${CUDA_BUILD_MANIFEST_BYTES}" "${SOURCE_RELEASE}/neoethos-cli" \
  || fail "neoethos-cli does not embed the selected exact sm_120 CUDA manifest"
CUDA_MANIFEST_SHA256="$(sha256sum "${CUDA_BUILD_MANIFEST}" | awk '{print $1}')"

GPU_FACTS="$(nvidia-smi \
  --query-gpu=name,compute_cap,driver_version \
  --format=csv,noheader,nounits | awk 'NR == 1 { print; exit }')"
[[ -n "${GPU_FACTS}" ]] || fail "nvidia-smi returned no physical GPU"
IFS=',' read -r GPU_NAME GPU_COMPUTE_CAP DRIVER_VERSION <<<"${GPU_FACTS}"
GPU_NAME="$(trim "${GPU_NAME}")"
GPU_COMPUTE_CAP="$(trim "${GPU_COMPUTE_CAP}")"
DRIVER_VERSION="$(trim "${DRIVER_VERSION}")"
[[ "${GPU_COMPUTE_CAP}" == "12.0" ]] \
  || fail "physical packaging GPU is compute ${GPU_COMPUTE_CAP}, not the required 12.0"

CUDA_ROOT="${CUDA_PATH:-/usr/local/cuda}"
CUDA_LIB_DIR="${NEOETHOS_CUDA_LIB_DIR:-${CUDA_ROOT}/lib64}"
[[ -d "${CUDA_LIB_DIR}" ]] || fail "CUDA redistributable directory is missing: ${CUDA_LIB_DIR}"
CUDA_LIB_DIR="$(cd -- "${CUDA_LIB_DIR}" && pwd -P)"

BUNDLE_NAME="NeoEthos-${VERSION}-linux-x86_64-sm120"
RELEASE_ASSET_DIR="${OUTPUT_DIR}/${BUNDLE_NAME}.release-assets"
[[ ! -e "${RELEASE_ASSET_DIR}" ]] \
  || fail "refusing to overwrite existing release asset set: ${RELEASE_ASSET_DIR}"

STAGING_ROOT="$(mktemp -d "${OUTPUT_DIR}/.neoethos-vps-release.XXXXXX")"
cleanup() {
  if [[ -n "${STAGING_ROOT:-}" && -d "${STAGING_ROOT}" ]]; then
    case "${STAGING_ROOT}" in
      "${OUTPUT_DIR}"/.neoethos-vps-release.*) rm -rf -- "${STAGING_ROOT}" ;;
      *) printf 'neoethos-vps-release: refusing unsafe cleanup path: %s\n' "${STAGING_ROOT}" >&2 ;;
    esac
  fi
}
trap cleanup EXIT

BUNDLE_DIR="${STAGING_ROOT}/${BUNDLE_NAME}"
PUBLISH_DIR="${STAGING_ROOT}/release-assets"
mkdir -p -- \
  "${BUNDLE_DIR}/assets/symbol_metadata" \
  "${BUNDLE_DIR}/evidence" \
  "${PUBLISH_DIR}"
ARCHIVE_PATH="${PUBLISH_DIR}/${BUNDLE_NAME}.tar.gz"
ARCHIVE_CHECKSUM_PATH="${PUBLISH_DIR}/${BUNDLE_NAME}.tar.gz.sha256"
EXTERNAL_MANIFEST_PATH="${PUBLISH_DIR}/${BUNDLE_NAME}.MANIFEST.json"
EXTERNAL_SUMS_PATH="${PUBLISH_DIR}/${BUNDLE_NAME}.SHA256SUMS"

install -m 0755 "${SOURCE_RELEASE}/neoethos-app" "${BUNDLE_DIR}/neoethos-app"
install -m 0755 "${SOURCE_RELEASE}/neoethos-cli" "${BUNDLE_DIR}/neoethos-cli"
install -m 0644 "${SOURCE_RELEASE}/libxgboost.so" "${BUNDLE_DIR}/libxgboost.so"
install -m 0644 "${SOURCE_RELEASE}/libcatboostmodel.so" "${BUNDLE_DIR}/libcatboostmodel.so"

CUDA_RUNTIME_PATTERNS=(
  'libcudart.so'
  'libnvrtc.so'
  'libnvrtc-builtins.so'
  'libcublas.so'
  'libcublasLt.so'
  'libcurand.so'
  'libcusparse.so'
  'libcusolver.so'
  'libnvJitLink.so'
)
shopt -s nullglob
for runtime_pattern in "${CUDA_RUNTIME_PATTERNS[@]}"; do
  runtime_matches=("${CUDA_LIB_DIR}/${runtime_pattern}"*)
  [[ ${#runtime_matches[@]} -gt 0 ]] \
    || fail "required CUDA runtime family is absent: ${runtime_pattern}*"
  for runtime_path in "${runtime_matches[@]}"; do
    cp -a -- "${runtime_path}" "${BUNDLE_DIR}/"
  done
done
shopt -u nullglob

while IFS= read -r -d '' runtime_link; do
  resolved_runtime="$(realpath -e -- "${runtime_link}")"
  case "${resolved_runtime}" in
    "${BUNDLE_DIR}"/*) ;;
    *) fail "refusing CUDA runtime symlink outside the bundle: ${runtime_link} -> ${resolved_runtime}" ;;
  esac
  rm -f -- "${runtime_link}"
  ln -- "${resolved_runtime}" "${runtime_link}"
done < <(find "${BUNDLE_DIR}" -maxdepth 1 -type l -name 'lib*.so*' -print0)
if find "${BUNDLE_DIR}" -maxdepth 1 -type l -name 'lib*.so*' -print -quit | grep -q .; then
  fail "runtime staging retained a symbolic link after confinement conversion"
fi

if find "${BUNDLE_DIR}" -maxdepth 1 \( -type f -o -type l \) -name 'libcuda.so*' \
  -print -quit | grep -q .; then
  fail "libcuda.so is a host driver and must never be bundled"
fi
for staged_runtime in "${BUNDLE_DIR}"/lib*.so*; do
  [[ -e "${staged_runtime}" ]] || fail "staged runtime is a dangling symlink: ${staged_runtime}"
done

if ! awk '
  $0 == "models:" {
    print
    print "  prop_search_device: cuda_required"
    inserted_discovery = 1
    next
  }
  $0 ~ /^  prop_search_device:/ { next }
  $0 ~ /^  enable_gpu_preference:/ {
    print "  enable_gpu_preference: cuda_required"
    replaced_global = 1
    next
  }
  { print }
  END {
    if (!inserted_discovery || !replaced_global) exit 42
  }
' "${REPO_ROOT}/config.yaml" >"${BUNDLE_DIR}/config.yaml"; then
  fail "could not seal the VPS config to the current CUDA-required settings authority"
fi
chmod 0644 "${BUNDLE_DIR}/config.yaml"
[[ "$(grep -Fxc '  prop_search_device: cuda_required' "${BUNDLE_DIR}/config.yaml")" -eq 1 ]] \
  || fail "VPS config does not contain one exact CUDA-required Discovery policy"
[[ "$(grep -Fxc '  enable_gpu_preference: cuda_required' "${BUNDLE_DIR}/config.yaml")" -eq 1 ]] \
  || fail "VPS config does not contain one exact CUDA-required global policy"
install -m 0644 \
  "${REPO_ROOT}/assets/symbol_metadata/defaults.json" \
  "${BUNDLE_DIR}/assets/symbol_metadata/defaults.json"
install -m 0644 "${CUDA_BUILD_MANIFEST}" "${BUNDLE_DIR}/evidence/cuda-build-manifest.json"

cat >"${BUNDLE_DIR}/run-neoethos-app.sh" <<'APP_WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG_FILE="${BUNDLE_DIR}/config.yaml"
NEOETHOS_BOT_SYMBOL_METADATA="${BUNDLE_DIR}/assets/symbol_metadata/defaults.json"
export CONFIG_FILE NEOETHOS_BOT_SYMBOL_METADATA
unset NEOETHOS_REQUIRE_GPU
cd -- "${BUNDLE_DIR}"
exec "${BUNDLE_DIR}/neoethos-app" --config "${CONFIG_FILE}" "$@"
APP_WRAPPER

cat >"${BUNDLE_DIR}/run-neoethos-cli.sh" <<'CLI_WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG_FILE="${BUNDLE_DIR}/config.yaml"
NEOETHOS_BOT_SYMBOL_METADATA="${BUNDLE_DIR}/assets/symbol_metadata/defaults.json"
export CONFIG_FILE NEOETHOS_BOT_SYMBOL_METADATA
unset NEOETHOS_REQUIRE_GPU
cd -- "${BUNDLE_DIR}"
exec "${BUNDLE_DIR}/neoethos-cli" "$@"
CLI_WRAPPER
chmod 0755 "${BUNDLE_DIR}/run-neoethos-app.sh" "${BUNDLE_DIR}/run-neoethos-cli.sh"
bash -n "${BUNDLE_DIR}/run-neoethos-app.sh"
bash -n "${BUNDLE_DIR}/run-neoethos-cli.sh"

for executable_name in neoethos-app neoethos-cli; do
  readelf -d "${BUNDLE_DIR}/${executable_name}" \
    >"${BUNDLE_DIR}/evidence/${executable_name}.readelf.txt"
  grep -Fq '$ORIGIN' "${BUNDLE_DIR}/evidence/${executable_name}.readelf.txt" \
    || fail "${executable_name} does not carry executable-relative RUNPATH"

  env -u LD_LIBRARY_PATH ldd "${BUNDLE_DIR}/${executable_name}" \
    >"${BUNDLE_DIR}/evidence/${executable_name}.ldd.txt"
  if grep -Fq 'not found' "${BUNDLE_DIR}/evidence/${executable_name}.ldd.txt"; then
    fail "${executable_name} has an unresolved dynamic dependency"
  fi
done

CLI_LDD_EVIDENCE="${BUNDLE_DIR}/evidence/neoethos-cli.ldd.txt"
for tree_runtime in libxgboost.so libcatboostmodel.so; do
  resolved_path="$(awk -v library="${tree_runtime}" \
    '$1 == library && $2 == "=>" { print $3; exit }' \
    "${CLI_LDD_EVIDENCE}")"
  [[ -n "${resolved_path}" ]] || fail "neoethos-cli did not resolve ${tree_runtime}"
  [[ "$(realpath "${resolved_path}")" == "$(realpath "${BUNDLE_DIR}/${tree_runtime}")" ]] \
    || fail "neoethos-cli resolved ${tree_runtime} outside the bundle: ${resolved_path}"
done

env -u LD_LIBRARY_PATH "${BUNDLE_DIR}/neoethos-app" --version \
  >"${BUNDLE_DIR}/evidence/neoethos-app.version.txt"
env -u LD_LIBRARY_PATH "${BUNDLE_DIR}/neoethos-cli" --version \
  >"${BUNDLE_DIR}/evidence/neoethos-cli.version.txt"
APP_VERSION="$(tr -d '\r\n' <"${BUNDLE_DIR}/evidence/neoethos-app.version.txt")"
CLI_VERSION="$(tr -d '\r\n' <"${BUNDLE_DIR}/evidence/neoethos-cli.version.txt")"

FILE_ROWS="${STAGING_ROOT}/payload-files.tsv"
(
  cd -- "${BUNDLE_DIR}"
  find . -type f ! -name 'MANIFEST.json' ! -name 'SHA256SUMS' -print0 \
    | LC_ALL=C sort -z \
    | while IFS= read -r -d '' relative_path; do
        file_sha256="$(sha256sum "${relative_path}" | awk '{print $1}')"
        file_bytes="$(wc -c <"${relative_path}")"
        printf '%s\t%s\t%s\n' "${relative_path#./}" "${file_bytes//[[:space:]]/}" "${file_sha256}"
      done
) >"${FILE_ROWS}"

{
  printf '{\n'
  printf '  "schema": "neoethos.vps-release-bundle.v1",\n'
  printf '  "version": 1,\n'
  printf '  "release_label": "%s",\n' "$(json_escape "${VERSION}")"
  printf '  "git_sha": "%s",\n' "${GIT_SHA}"
  printf '  "target": "linux-x86_64",\n'
  printf '  "cuda_target": "sm_120",\n'
  printf '  "cuda_ptx_embedded": false,\n'
  printf '  "cuda_manifest_sha256": "%s",\n' "${CUDA_MANIFEST_SHA256}"
  printf '  "gpu_name": "%s",\n' "$(json_escape "${GPU_NAME}")"
  printf '  "gpu_compute_capability": "%s",\n' "$(json_escape "${GPU_COMPUTE_CAP}")"
  printf '  "driver_version": "%s",\n' "$(json_escape "${DRIVER_VERSION}")"
  printf '  "dependency_verification_scope": "packaging_host_ldd_only",\n'
  printf '  "app_version": "%s",\n' "$(json_escape "${APP_VERSION}")"
  printf '  "cli_version": "%s",\n' "$(json_escape "${CLI_VERSION}")"
  printf '  "files": [\n'
  first_row=1
  while IFS=$'\t' read -r relative_path file_bytes file_sha256; do
    if [[ ${first_row} -eq 0 ]]; then
      printf ',\n'
    fi
    first_row=0
    printf '    {"path":"%s","bytes":%s,"sha256":"%s"}' \
      "$(json_escape "${relative_path}")" "${file_bytes}" "${file_sha256}"
  done <"${FILE_ROWS}"
  printf '\n  ]\n}\n'
} >"${BUNDLE_DIR}/MANIFEST.json"
python3 -m json.tool "${BUNDLE_DIR}/MANIFEST.json" >/dev/null

(
  cd -- "${BUNDLE_DIR}"
  find . -type f ! -name 'SHA256SUMS' -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum >SHA256SUMS
  sha256sum -c SHA256SUMS >/dev/null
)

install -m 0644 "${BUNDLE_DIR}/MANIFEST.json" "${EXTERNAL_MANIFEST_PATH}"
install -m 0644 "${BUNDLE_DIR}/SHA256SUMS" "${EXTERNAL_SUMS_PATH}"
tar -czf "${ARCHIVE_PATH}" -C "${STAGING_ROOT}" "${BUNDLE_NAME}"
(
  cd -- "${PUBLISH_DIR}"
  sha256sum "$(basename -- "${ARCHIVE_PATH}")" \
    >"$(basename -- "${ARCHIVE_CHECKSUM_PATH}")"
  sha256sum -c "$(basename -- "${ARCHIVE_CHECKSUM_PATH}")" >/dev/null
)

require_nonempty_file "${ARCHIVE_PATH}"
require_nonempty_file "${ARCHIVE_CHECKSUM_PATH}"
tar -tzf "${ARCHIVE_PATH}" | grep -Fqx "${BUNDLE_NAME}/MANIFEST.json"
tar -tzf "${ARCHIVE_PATH}" | grep -Fqx "${BUNDLE_NAME}/SHA256SUMS"
mv -- "${PUBLISH_DIR}" "${RELEASE_ASSET_DIR}"

FINAL_ARCHIVE_PATH="${RELEASE_ASSET_DIR}/$(basename -- "${ARCHIVE_PATH}")"
FINAL_ARCHIVE_CHECKSUM_PATH="${RELEASE_ASSET_DIR}/$(basename -- "${ARCHIVE_CHECKSUM_PATH}")"
FINAL_MANIFEST_PATH="${RELEASE_ASSET_DIR}/$(basename -- "${EXTERNAL_MANIFEST_PATH}")"
FINAL_SUMS_PATH="${RELEASE_ASSET_DIR}/$(basename -- "${EXTERNAL_SUMS_PATH}")"

printf 'release_bundle=%s\n' "${FINAL_ARCHIVE_PATH}"
printf 'release_bundle_sha256=%s\n' "$(awk '{print $1}' "${FINAL_ARCHIVE_CHECKSUM_PATH}")"
printf 'release_manifest=%s\n' "${FINAL_MANIFEST_PATH}"
printf 'release_sha256sums=%s\n' "${FINAL_SUMS_PATH}"
