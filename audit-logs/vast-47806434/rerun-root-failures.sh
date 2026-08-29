#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/neoethos/dependency-upgrade-probe
target=/workspace/neoethos/targets/compatible-root-fixed/target
log=/workspace/neoethos/audit-logs/compatible/root-failures-isolated-2.log

cd "$repo" || exit 97
if [[ -f /root/.cargo/env ]]; then
  source /root/.cargo/env
fi
export CARGO_BUILD_JOBS=62
export RUST_TEST_THREADS=1

{
  rustc +nightly-2026-04-07 -Vv
  cargo +nightly-2026-04-07 -Vv

  cargo +nightly-2026-04-07 test -p neoethos-models \
    tree_models::config::tests::per_model_threads_never_exceed_resolved_cpu_budget \
    --target-dir "$target" -- --exact --nocapture --test-threads=1
  echo "ISOLATED_MODEL_BUDGET_EXIT=$?"

  cargo +nightly-2026-04-07 test -p neoethos-models --test tree_models_integration \
    lightgbm_tests::test_gpu_only_mode \
    --target-dir "$target" -- --exact --nocapture --test-threads=1
  echo "ISOLATED_GPU_ONLY_EXIT=$?"

  cargo +nightly-2026-04-07 test -p neoethos-search \
    discovery::tests::identical_configs_produce_identical_profile_json_apart_from_ambient_state \
    --target-dir "$target" -- --exact --nocapture --test-threads=1
  echo "ISOLATED_PROFILE_EXIT=$?"
} 2>&1 | tee "$log"

exit 0
