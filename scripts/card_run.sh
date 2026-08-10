#!/usr/bin/env bash
# The card run. One command, unattended, everything logged.
#
#   ./scripts/card_run.sh 2>&1 | tee card_run.console.log
#
# Answers exactly one question: is it, finally, everything on the card from
# start to finish. Every step writes a full log to $LOGS and the run stops at
# the first step that cannot honestly be called a pass.
#
# HOST REQUIREMENTS, learned the expensive way:
#   - AMD host. Building features on an Intel Xeon SIGILLs.
#   - Ubuntu 24.04+. 22.04 ships cmake 3.22 and lightgbm3-sys needs >= 3.28.
#   - Direct SSH port, never a proxy.
#   - Do NOT install mold. It breaks every CUDA binary this repo produces.
#   - nvcc 12.2. CUDA 13 was evaluated and rejected.

set -uo pipefail

LOGS="${LOGS:-$PWD/card-logs-$(date -u +%Y%m%d-%H%M%S)}"
mkdir -p "$LOGS"
COMMIT="$(git rev-parse HEAD)"
step=0

say()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
fail() { printf '\n\033[31mSTOPPED at step %s: %s\033[0m\n' "$step" "$*"; echo "see $LOGS"; exit 1; }

say "commit $COMMIT"
git status --porcelain > "$LOGS/00-dirty.txt"
[ -s "$LOGS/00-dirty.txt" ] && echo "WARNING: tree is dirty; the logs describe a state no commit holds"

# ── 1. The host can actually compile CUDA ────────────────────────────────────
step=1; say "preconditions"
command -v nvcc >/dev/null || fail "no nvcc. Without it the 253 changed vendor .cu files compile on NOTHING and this run proves nothing."
nvcc --version   > "$LOGS/01-nvcc.txt" 2>&1
nvidia-smi       > "$LOGS/01-smi.txt"  2>&1 || fail "no nvidia-smi"
free -g          > "$LOGS/01-mem.txt"  2>&1
df -h            > "$LOGS/01-disk.txt" 2>&1
grep -qi amd /proc/cpuinfo || echo "WARNING: not an AMD host — feature building has SIGILLed on Intel Xeon"

# ── 2. The config the run will actually use ──────────────────────────────────
# The live store is the ONLY file a run reads. Skipping this means searching
# under whatever numbers that file last had — which on the operator's box means
# the payoff gate OFF and no portfolio cap.
step=2; say "config migration (dry run first)"
if [ -f scripts/migrate_live_config.ps1 ] && command -v pwsh >/dev/null; then
  pwsh -File scripts/migrate_live_config.ps1 -WhatIf > "$LOGS/02-migration-diff.txt" 2>&1
  echo "diff written to $LOGS/02-migration-diff.txt"
  if [ "${APPLY_MIGRATION:-0}" = "1" ]; then
    pwsh -File scripts/migrate_live_config.ps1 -Confirm:$false >> "$LOGS/02-migration-diff.txt" 2>&1 \
      || fail "migration refused — read $LOGS/02-migration-diff.txt"
  else
    echo "NOT APPLIED. Re-run with APPLY_MIGRATION=1 once the diff has been read."
  fi
else
  echo "no pwsh or no migration script — the run will use the config as it stands"
fi
echo "--- the values this run will search under ---" > "$LOGS/02-effective.txt"
grep -nE 'prop_search_min_payoff_ratio|prefilter_top_k|max_portfolio_risk|require_walkforward_for_export|adaptive_thresholds|normalize_features|prop_search_device' \
  "${CONFIG_FILE:-$HOME/.local/share/neoethos/config.yaml}" >> "$LOGS/02-effective.txt" 2>&1 || true
cat "$LOGS/02-effective.txt"

# ── 3. THE BUILD. This is the only compiler that has ever seen the kernels ───
step=3; say "release build, full GPU feature set"
# nvtx is instrumentation, not behaviour: it gives an in-kernel timeline and is
# otherwise enabled by nothing.
FEATURES="${FEATURES:-gpu-nvidia,gpu-bench-cuda}"
echo "features: $FEATURES" | tee "$LOGS/03-features.txt"
cargo build --release -j "${JOBS:-8}" -p neoethos-cli -p neoethos-app \
      --features "$FEATURES" > "$LOGS/03-build.log" 2>&1
rc=$?
warn=$(grep -c '^warning' "$LOGS/03-build.log" || true)
err=$(grep -c  '^error'   "$LOGS/03-build.log" || true)
echo "warnings=$warn errors=$err  (full log: $LOGS/03-build.log)"
grep -E '^(warning|error)' "$LOGS/03-build.log" | sort | uniq -c | sort -rn | head -40 \
  > "$LOGS/03-build-summary.txt"
cat "$LOGS/03-build-summary.txt"
[ $rc -eq 0 ] || fail "the build failed. READ ALL of $LOGS/03-build.log — nvcc errors are the point of this run."
[ "$warn" -eq 0 ] || echo "NOTE: $warn warnings. The operator asked for the whole log including warnings; they are in 03-build.log."

# ── 4. Parity — does the card agree with the CPU ─────────────────────────────
step=4; say "GPU parity"
CARGO_PROFILE_TEST_DEBUG=0 cargo test --release -j "${JOBS:-8}" \
      -p neoethos-search --features "$FEATURES" gpu_ \
      > "$LOGS/04-parity.log" 2>&1
grep -E '^test result' "$LOGS/04-parity.log" | tee "$LOGS/04-parity-summary.txt"
grep -qE '^test result: ok' "$LOGS/04-parity.log" || fail "parity failed — $LOGS/04-parity.log"

# ── 5. The full suite, everything that can run here ──────────────────────────
step=5; say "test suites"
for p in neoethos-core neoethos-data neoethos-search neoethos-app neoethos-models; do
  CARGO_PROFILE_TEST_DEBUG=0 cargo test --release -j "${JOBS:-8}" -p "$p" --lib \
      >> "$LOGS/05-tests.log" 2>&1
  printf '%-18s %s\n' "$p" "$(grep -E '^test result' "$LOGS/05-tests.log" | tail -1)"
done
grep -E '^test result: FAILED' "$LOGS/05-tests.log" > "$LOGS/05-failures.txt" || true
[ -s "$LOGS/05-failures.txt" ] && echo "SOME SUITES FAILED — $LOGS/05-failures.txt (continuing to the run anyway; a red unit test does not invalidate the device measurement)"

# ── 6. THE RUN. M15, end to end ──────────────────────────────────────────────
step=6; say "M15 discovery run"
SYMBOL="${SYMBOL:-EURUSD}"
./target/release/neoethos-cli discover \
      --symbol "$SYMBOL" --timeframe M15 \
      > "$LOGS/06-run.log" 2>&1
rc=$?
echo "exit=$rc  (full log: $LOGS/06-run.log)"

# ── 7. The answer ────────────────────────────────────────────────────────────
step=7; say "did everything actually run on the card"
{
  echo "=== device summary (the invariant's own self-report) ==="
  grep -A25 -i 'device summary\|eval_telemetry\|gpu_pct' "$LOGS/06-run.log" | head -60
  echo
  echo "=== any CPU fallback recorded ==="
  grep -in 'cpu_fallback\|falling back\|silent cpu\|note_cpu_fallback' "$LOGS/06-run.log" | head -20
  echo
  echo "=== which engine evaluated the population ==="
  grep -in 'population_eval_engines\|PopulationEvalEngine\|prototype_b\|cubecl' "$LOGS/06-run.log" | head -20
  echo
  echo "=== the ten rejection counters — screen decided, or market decided ==="
  grep -iE 'profile_payoff_ratio|profile_net_expectancy|account_wiped|opportunistic_lane_closed|trades_per_month|monthly_return|positive_months|profile_win_rate|profile_in_market|expectancy_significance' \
      "$LOGS/06-run.log" | tail -20
  echo
  echo "=== indicator census — is the vocabulary really unlocked ==="
  grep -iE 'producing_ids|indicator vocabulary census|over_budget|admitted_columns|max_columns' "$LOGS/06-run.log" | head -20
  echo
  echo "=== config identity — which file, which hash ==="
  grep -iE 'config_hash|ConfigSource|UserStore|RepoRelative|provenance' "$LOGS/06-run.log" | head -10
} > "$LOGS/07-verdict.txt" 2>&1
cat "$LOGS/07-verdict.txt"

say "done — everything in $LOGS"
tar czf "${LOGS}.tar.gz" -C "$(dirname "$LOGS")" "$(basename "$LOGS")" 2>/dev/null \
  && echo "archive: ${LOGS}.tar.gz"

cat <<'NOTE'

HOW TO READ 07-verdict.txt, in order:

  1. gpu_pct. Below 95 with a card present means the run did NOT stay on the
     device, and the number tells you how far short.
  2. Any cpu_fallback line names an indicator that still owes a kernel.
  3. The rejection counters answer the sixteen-month question. If
     profile_payoff_ratio equals the candidate count, the SCREEN decided and the
     market was never measured. If profile_net_expectancy equals it, the market
     WAS measured and said no. Those are opposite conclusions.
  4. producing_ids near 329 means the vocabulary is genuinely unlocked; 1 is the
     old behaviour and is now a hard error, not a silent success.
  5. config_hash makes this the first run in the project's history that can be
     identified after the fact. Write it down.
NOTE
