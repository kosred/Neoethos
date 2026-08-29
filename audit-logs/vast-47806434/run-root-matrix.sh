#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/neoethos/dependency-upgrade-probe
log=/workspace/neoethos/audit-logs/compatible/root-all-targets-fixed-3.log
target=/workspace/neoethos/targets/compatible-root-fixed/target

mkdir -p "$(dirname "$log")" "$target"
exec > >(tee -a "$log") 2>&1

if [[ -f /root/.cargo/env ]]; then
  # rustup installs this environment file; tmux starts a non-login shell here.
  source /root/.cargo/env
fi

echo "ROOT_MATRIX_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "ROOT_MATRIX_REPO=$repo"
echo "ROOT_MATRIX_TARGET=$target"
echo "ROOT_MATRIX_BUILD_JOBS=62"
echo "ROOT_MATRIX_TEST_THREADS=62"

cd "$repo"
export CARGO_BUILD_JOBS=62
export CARGO_TARGET_DIR="$target"
export RUST_TEST_THREADS=62

git status --short
rustc +nightly-2026-04-07 -Vv
cargo +nightly-2026-04-07 -Vv
df -h /workspace
free -h
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader

echo "ROOT_CHECK_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cargo +nightly-2026-04-07 check --workspace --all-targets -j 62
check_status=$?
echo "ROOT_CHECK_EXIT=$check_status"
echo "ROOT_CHECK_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "ROOT_TEST_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cargo +nightly-2026-04-07 test --workspace --all-targets --no-fail-fast -j 62 -- --test-threads=62
test_status=$?
echo "ROOT_TEST_EXIT=$test_status"
echo "ROOT_TEST_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

df -h /workspace
free -h
echo "ROOT_MATRIX_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ $check_status -ne 0 || $test_status -ne 0 ]]; then
  exit 1
fi
