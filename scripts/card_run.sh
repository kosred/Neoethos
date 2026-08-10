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
step=2; say "the config this run will actually use"

# migrate_live_config.ps1 is [CmdletBinding()] with ONE switch, -Apply. It has no
# SupportsShouldProcess, so -WhatIf and -Confirm are parameter-binding ERRORS.
# The first version of this script passed both, swallowed the error into the diff
# file, and printed "diff written to ..." over a file containing nothing but a
# PowerShell binding failure. Report-only is the default; -Apply is the other mode.
#
# And the apply path does not belong in an unattended script at all: the migration
# calls Assert-Interactive and Read-Host by design, because it rewrites the only
# file a run reads. Run it yourself, before this script, and re-run this script
# afterwards.
if [ -f scripts/migrate_live_config.ps1 ] && command -v pwsh >/dev/null; then
  pwsh -File scripts/migrate_live_config.ps1 > "$LOGS/02-migration-report.txt" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    head -20 "$LOGS/02-migration-report.txt"
    fail "the migration report itself failed (exit $rc). Fix that before spending card time."
  fi
  echo "migration report: $LOGS/02-migration-report.txt"
fi

# WHICH FILE WILL Settings::load() OPEN, AND WHAT IS IN IT.
# The cwd-relative fallback was deleted, so on a fresh box with no user store the
# run takes COMPILED DEFAULTS — and that is not a neutral outcome. Against the repo
# profile it flips trading_mode and discovery_mode risky->prop_firm, preset
# none->ftmo, require_walkforward_for_export false->true, prop_firm_min_pass_rate
# 0.0->0.40, multi_resolution_enabled false->true, prop_search_generations
# 20000->50, portfolio_size 50->3000, and max_portfolio_risk 0.34->0.0 = NO CAP.
# The old version of this step let that pass with `|| true` and an empty report.
STORE="${CONFIG_FILE:-}"
if [ -z "$STORE" ]; then
  case "$(uname -s)" in
    Linux)  STORE="${XDG_DATA_HOME:-$HOME/.local/share}/neoethos/config.yaml" ;;
    Darwin) STORE="$HOME/Library/Application Support/neoethos/config.yaml" ;;
    *)      STORE="${LOCALAPPDATA:-$HOME}/neoethos/config.yaml" ;;
  esac
fi
{
  echo "resolved store path: $STORE"
  if [ -f "$STORE" ]; then
    echo "EXISTS — this run searches under the values below"
    grep -nE 'prop_search_min_payoff_ratio|prefilter_top_k|max_portfolio_risk|require_walkforward_for_export|adaptive_thresholds|normalize_features|prop_search_device|trading_mode|discovery_mode|prop_firm_min_pass_rate' "$STORE"
  else
    echo "ABSENT — Settings::load() will fall through to COMPILED DEFAULTS."
  fi
} > "$LOGS/02-effective.txt" 2>&1
cat "$LOGS/02-effective.txt"

if [ ! -f "$STORE" ] && [ "${ALLOW_COMPILED_DEFAULTS:-0}" != "1" ]; then
  fail "no config store at $STORE. This run would search under compiled defaults —
        prop_firm rules on, 50 generations instead of 20000, and max_portfolio_risk
        0.0 which means NO CAP. Copy a config there, or set
        ALLOW_COMPILED_DEFAULTS=1 if that is genuinely what you want measured."
fi

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
  echo "=== vector-ta host-fallback debt (four wrappers compute on the host) ==="
  # rvi, mass, net_myrsi and vosc each return a DeviceArray built from a HOST
  # computation. They are now counted by vector_ta::cuda::host_fallback::record,
  # but NOTHING READS THE COUNTER YET — the reader belongs in the device summary
  # and that crate was being written when this landed. Until it exists, these
  # greps are the only surface. Absence of a line here is NOT evidence of zero.
  grep -in 'host_fallback\|host fallback\|computed on the host' "$LOGS/06-run.log" | head -10
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
