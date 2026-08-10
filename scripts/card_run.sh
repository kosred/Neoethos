#!/usr/bin/env bash
# The card run — TWO STAGES, because they answer different questions and the
# second is worthless until the first is clean.
#
#   ./scripts/card_run.sh cuda     # stage A: build, parity, tests. STOPS there.
#   ./scripts/card_run.sh search   # stage B: the M15 run + the verdict.
#
# The operator's sequencing, and the reason for the split: all repairs and the
# single config land FIRST, on the dev box. THEN the card, and the card is for
# SOLVING CUDA PROBLEMS — 253 changed .cu files that no compiler has ever seen,
# plus four wrapper edits under cfg(feature="cuda"). Expect stage A to fail the
# first time; that is what it is for. Read the log, fix, re-run stage A. Only
# when it is clean does stage B run the actual searches.
#
# Running them as one command would burn hours of card time on a discovery run
# launched from a build nobody had read.
#
# THE CONFIG MIGRATION IS NOT HERE ANY MORE. It rewrites the only file a run
# reads, it calls Read-Host by design, and it belongs to the "one config" work
# that happens BEFORE the card is rented:
#     pwsh -File scripts/migrate_live_config.ps1          # report, read the diff
#     pwsh -File scripts/migrate_live_config.ps1 -Apply   # then apply
# Stage A refuses to start if the store still carries the pre-migration values.
#
# HOST REQUIREMENTS, learned the expensive way:
#   - AMD host. Building features on an Intel Xeon SIGILLs.
#   - Ubuntu 24.04+. 22.04 ships cmake 3.22 and lightgbm3-sys needs >= 3.28.
#   - Direct SSH port, never a proxy.
#   - Do NOT install mold. It breaks every CUDA binary this repo produces.
#   - nvcc 12.2. CUDA 13 was evaluated and rejected.

set -uo pipefail

STAGE="${1:-cuda}"
case "$STAGE" in
  cuda|search) ;;
  *) echo "usage: $0 [cuda|search]"; echo "  cuda   — build, parity, tests. Fix CUDA here."; echo "  search — the M15 run. Only after cuda is clean."; exit 2 ;;
esac

LOGS="${LOGS:-$PWD/card-logs-$STAGE-$(date -u +%Y%m%d-%H%M%S)}"
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
step=2; say "the config this run will use — the migration must already have happened"

# The migration is NOT run here. It rewrites the only file a run reads and calls
# Read-Host by design; it belongs to the "one config" work that lands BEFORE the
# card is rented. This step only REFUSES to proceed if it clearly never ran.
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
    echo "EXISTS — this run uses the values below"
    grep -nE 'prop_search_min_payoff_ratio|prefilter_top_k|max_portfolio_risk|require_walkforward_for_export|adaptive_thresholds|normalize_features|prop_search_device|trading_mode|discovery_mode|prop_firm_min_pass_rate' "$STORE"
  else
    echo "ABSENT — Settings::load() falls through to COMPILED DEFAULTS."
  fi
} > "$LOGS/02-effective.txt" 2>&1
cat "$LOGS/02-effective.txt"

if [ ! -f "$STORE" ] && [ "${ALLOW_COMPILED_DEFAULTS:-0}" != "1" ]; then
  fail "no config store at $STORE. This run would use compiled defaults —
        prop_firm rules on, 50 generations instead of 20000, and
        max_portfolio_risk 0.0 which means NO CAP. Copy a config there, or set
        ALLOW_COMPILED_DEFAULTS=1 if that is genuinely what you want measured."
fi

# The three fingerprints of a store the migration never touched. Each is a value
# the migration asks about by name, and each makes a search meaningless: the
# payoff gate off, no portfolio cap, and a prefilter keeping a fifth of what it
# should. Stage B refuses on these; stage A only warns, because a CUDA build does
# not care what the search thresholds say.
if [ -f "$STORE" ]; then
  stale=0
  grep -qE '^\s*prop_search_min_payoff_ratio:\s*0\.0*\s*$' "$STORE" && { echo "STALE: prop_search_min_payoff_ratio is 0.0 — the payoff gate is OFF"; stale=1; }
  grep -qE '^\s*max_portfolio_risk:\s*0\.0*\s*$'          "$STORE" && { echo "STALE: max_portfolio_risk is 0.0 — that means NO CAP, not no risk"; stale=1; }
  grep -qE '^\s*prefilter_top_k:\s*50\s*$'                 "$STORE" && { echo "STALE: prefilter_top_k is 50 against a ~1795-column vocabulary"; stale=1; }
  if [ "$stale" = "1" ]; then
    echo
    echo "The migration has not been applied to this store. Run it on the dev box:"
    echo "    pwsh -File scripts/migrate_live_config.ps1          # read the diff"
    echo "    pwsh -File scripts/migrate_live_config.ps1 -Apply"
    [ "$STAGE" = "search" ] && fail "refusing to search under pre-migration values — the answer would be about the config, not the market"
    echo "(stage cuda continues: a CUDA build does not depend on these.)"
  fi
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

# ── 5b. THE BURN-CUDA A/B — the gate on model scale ──────────────────────────
# burn-cuda-backend is deliberately EXCLUDED from the gpu-cuda aggregate, on a
# 2026-06-10 A6000 measurement: 14 real epochs in 74 minutes for one combination,
# with burn-tensor dtype panics, against ~17 minutes TOTAL on burn-ndarray CPU.
# That decision is documented in neoethos-models/Cargo.toml and must NOT be
# overturned by argument.
#
# But kernel-launch overhead is a FIXED cost and the work has grown by orders of
# magnitude since — 1,795 feature columns now, against a few hundred then — and a
# 2026-08-01 observation recorded a training run sitting at GPU 0% / 1 MiB for
# over an hour at 799,880 rows. The ratio may have inverted. This step is the
# only thing that can say so, and it is the gate on whether a ~2B ensemble is
# reachable at all: today every burn neural model trains on the CPU, so scale is
# blocked regardless of which card is rented.
step=5b; say "burn-cuda A/B — does the neural side belong on the card yet"
if [ "${SKIP_BURN_AB:-0}" != "1" ]; then
  for variant in "without:$FEATURES" "with:$FEATURES,burn-cuda-backend"; do
    name="${variant%%:*}"; feats="${variant##*:}"
    echo "--- burn $name cuda ---"
    /usr/bin/time -f "%e s wall  %M KB peak"       cargo test --release -j "${JOBS:-8}" -p neoethos-models --features "$feats"         burn_ -- --nocapture > "$LOGS/05b-burn-$name.log" 2>&1
    tail -3 "$LOGS/05b-burn-$name.log"
  done
  echo "compare wall time and peak memory in $LOGS/05b-burn-*.log"
  echo "If WITH is now faster, add burn-cuda-backend to the gpu-cuda aggregate and"
  echo "say the 2026-06-10 measurement expired. If it is still slower, leave the"
  echo "exclusion and record the new number beside the old one."
fi

# ═══ END OF STAGE A ═══════════════════════════════════════════════════════════
# Everything above answers "does this build, and does the card agree with the
# CPU". That is the CUDA-fixing loop: run stage A, read the whole log, fix, run
# it again. Nothing below runs until it is clean, because a discovery run
# launched from a shaky build costs hours and proves nothing.
if [ "$STAGE" = "cuda" ]; then
  say "STAGE A COMPLETE — build, parity and suites are green"
  cat <<'NEXT'

Read these before going further:
  03-build.log            every warning, not just the errors
  04-parity-summary.txt   the card agreeing with the CPU
  05b-burn-*.log          with vs without burn-cuda — the gate on model scale

When stage A is clean and the migration has been applied on the dev box:
  ./scripts/card_run.sh search 2>&1 | tee card_search.console.log
NEXT
  tar czf "${LOGS}.tar.gz" -C "$(dirname "$LOGS")" "$(basename "$LOGS")" 2>/dev/null     && echo "archive: ${LOGS}.tar.gz"
  exit 0
fi

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
