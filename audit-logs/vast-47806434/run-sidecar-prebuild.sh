#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/neoethos/dependency-upgrade-probe
log=/workspace/neoethos/audit-logs/compatible/sidecar-release-prebuild.log

mkdir -p "$(dirname "$log")"
exec > >(tee -a "$log") 2>&1

if [[ -f /root/.cargo/env ]]; then
  source /root/.cargo/env
fi

export CARGO_BUILD_JOBS=62

echo "SIDECAR_PREBUILD_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
rustc +nightly-2026-04-07 -Vv
cargo +nightly-2026-04-07 -Vv

echo "MCP_RELEASE_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cargo +nightly-2026-04-07 build --release --manifest-path "$repo/mcp/Cargo.toml" -j 62
mcp_status=$?
echo "MCP_RELEASE_EXIT=$mcp_status"
echo "MCP_RELEASE_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "MESH_RELEASE_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cargo +nightly-2026-04-07 build --release --manifest-path "$repo/mesh/Cargo.toml" -j 62
mesh_status=$?
echo "MESH_RELEASE_EXIT=$mesh_status"
echo "MESH_RELEASE_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

ls -l "$repo/mcp/target/release/neoethos-mcp" "$repo/mesh/target/release/neoethos-mesh"
artifact_status=$?
echo "SIDECAR_ARTIFACTS_EXIT=$artifact_status"
df -h /workspace
echo "SIDECAR_PREBUILD_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ $mcp_status -ne 0 || $mesh_status -ne 0 || $artifact_status -ne 0 ]]; then
  exit 1
fi
