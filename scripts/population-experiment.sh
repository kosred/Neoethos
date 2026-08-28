#!/usr/bin/env bash
# population-experiment.sh — does a bigger GA population find BETTER strategies,
# or just more of the same?
#
# ── The question ─────────────────────────────────────────────────────────────
# config ships prop_search_population: 200 (default 100). The card sustains
# 42 M candidate-bars/s at population 256 and 843-966 M at 16 384-131 072, so
# the search currently asks the card for ~1-2 % of what it can do. Whether
# raising the population IMPROVES RESULTS (not just utilisation) is an open
# empirical question: a GA population is a SEARCH parameter — more candidates,
# different survivors — not a batching parameter. Never raise it on vibes.
#
# ── Design: three arms, wall-clock-fair ──────────────────────────────────────
#   A  population 100,  seed 1337   (baseline: today's shape)
#   B  population 4096, seed 1337   (the lever under test; same seed, same data)
#   C  population 100,  seed 7331   (seed-noise yardstick)
#
# Same data, same binary, same wall-clock budget per combo (prop_search_max_
# hours binds first; generations 20000 is effectively "until the clock").
# Wall-clock-fair is deliberate: on a rented card the budget is hours, so the
# operative question is "at fixed hours, does breadth beat depth?" — arm B
# runs fewer generations of a much wider population, and its per-candidate
# cost is ~20-40x cheaper because the launches actually load the card.
#
# NOTE on "same seed": A and B share seed 1337 but are NOT nested — a 4096
# population draws a different random stream than a 100 population. The seed
# equality removes seed variance WITHIN an arm definition, not between arms;
# that is what arm C is for.
#
# ── Pre-registered interpretation (decide BEFORE looking) ────────────────────
# Let d(X,Y) = the difference between arms X and Y on each headline number:
#   exported-portfolio size, walkforward pass rate, CPCV phi, PBO,
#   best/median holdout (forward-test) net profit of exported strategies.
#
#   * "POPULATION WAS NEVER THE CONSTRAINT": |d(B,A)| <= |d(C,A)| on the
#     holdout metrics — the bigger search moved results no more than reseeding
#     did. Then the lever to pursue is elsewhere (objective, gates, data), and
#     population stays at today's value with the card win taken purely as
#     faster wall-clock via the batching lanes.
#   * "POPULATION WAS A CONSTRAINT": d(B,A) exceeds the seed spread IN THE
#     DIRECTION OF better holdout results (more strategies clearing the
#     walkforward+CPCV+PBO gates, or clearly better forward-test metrics).
#     Then raise prop_search_population (or flip prop_search_population_auto)
#     as a DECISION, recorded with these numbers.
#   * "BIGGER IS WORSE" is a real possible outcome: 40x more trials against
#     the same validation budget is 40x more selection pressure on luck. If B
#     shows more exported strategies but WORSE holdout/PBO, the experiment has
#     measured overfitting amplification — population stays, and the
#     validation budget (candidate_count) becomes the next question.
#
# ── Safety ───────────────────────────────────────────────────────────────────
# * REFUSES to start while anything is using the card (build/parity jobs).
# * Backs up config.yaml and restores it on ANY exit.
# * All output under ~/logs/ (standing rule: never grep-filter live output).
#
# Usage, on the GPU box, from the repo root (~/Neoethos):
#   bash scripts/population-experiment.sh [SYMBOL]      # default EURUSD
#
# Cost estimate: 3 arms x (combos for one symbol) x prop_search_max_hours.
# With max_hours 1.0 and ~6 timeframes that is up to ~18 GPU-hours — trim
# prop_search_max_hours or the symbol's timeframe list first if budget is
# tight; whatever you pick applies identically to all three arms.

set -euo pipefail

SYMBOL="${1:-EURUSD}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$REPO_ROOT/config.yaml"
CLI="$REPO_ROOT/target/release/neoethos-cli"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_ROOT="$HOME/logs/pop-experiment-$STAMP"
STATE_DIR="${NEOETHOS_STATE_DIR:-$HOME/.local/share/neoethos}"

SEED_MAIN=1337
SEED_ALT=7331

fail() { echo "FATAL: $*" >&2; exit 1; }

[ -f "$CONFIG" ] || fail "config.yaml not found at $CONFIG — run from the repo root checkout"
[ -x "$CLI" ] || fail "release CLI not built at $CLI — build first (cargo build --release -p neoethos-cli --features gpu-nvidia), and NOT while this experiment runs"

# ── The card must be idle: this script must never race the build/parity job ──
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found — this experiment is for the GPU box"
BUSY="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l)"
[ "$BUSY" -eq 0 ] || fail "the card has $BUSY compute process(es) right now — do not run this while a build/parity/search job occupies it"

mkdir -p "$OUT_ROOT"
cp "$CONFIG" "$OUT_ROOT/config.yaml.orig"

restore_config() {
  cp "$OUT_ROOT/config.yaml.orig" "$CONFIG"
  echo "config.yaml restored from $OUT_ROOT/config.yaml.orig"
}
trap restore_config EXIT

# Targeted, anchored, count-checked single-key edits. sed keeps every comment
# in config.yaml intact (a YAML-library rewrite would strip them all).
set_key() { # set_key <key> <value>
  local key="$1" value="$2" count
  count="$(grep -cE "^[[:space:]]*${key}:" "$CONFIG" || true)"
  [ "$count" -eq 1 ] || fail "expected exactly 1 '${key}:' line in config.yaml, found $count — refusing a blind edit"
  sed -i -E "s|^([[:space:]]*)${key}:.*|\1${key}: ${value}|" "$CONFIG"
  grep -E "^[[:space:]]*${key}: ${value}$" "$CONFIG" >/dev/null || fail "edit of ${key} did not take"
}

git -C "$REPO_ROOT" rev-parse HEAD > "$OUT_ROOT/commit.txt" 2>/dev/null || true
nvidia-smi > "$OUT_ROOT/nvidia-smi.txt"

run_arm() { # run_arm <name> <population> <seed>
  local name="$1" pop="$2" seed="$3"
  local arm_dir="$OUT_ROOT/arm-$name"
  mkdir -p "$arm_dir"
  echo "=== arm $name: population=$pop seed=$seed symbol=$SYMBOL ==="

  set_key "prop_search_population" "$pop"
  set_key "prop_search_population_auto" "false"
  set_key "seed" "$seed"     # models.search_runtime.seed — the GA's deterministic seed
  cp "$CONFIG" "$arm_dir/config.yaml.used"

  # Everything the run writes after this marker is this arm's artifact set.
  touch "$arm_dir/started.marker"

  # GPU utilisation, once a second, for the whole arm. The point is to SEE the
  # difference between 256-sized launches and card-sized ones, not to infer it.
  nvidia-smi dmon -s um -d 1 > "$arm_dir/dmon.log" 2>&1 &
  local dmon_pid=$!

  local t0 t1
  t0=$(date +%s)
  ( cd "$REPO_ROOT" && "$CLI" discover --symbol "$SYMBOL" ) > "$arm_dir/discover.log" 2>&1 \
    || echo "WARN: discover exited non-zero for arm $name — see $arm_dir/discover.log" | tee -a "$arm_dir/WARN"
  t1=$(date +%s)
  kill "$dmon_pid" 2>/dev/null || true
  echo "$((t1 - t0))" > "$arm_dir/wall_seconds.txt"

  # Snapshot every artifact this run touched (portfolio JSON, funnel JSON,
  # ledgers, exports) without hardcoding the store layout.
  if [ -d "$STATE_DIR" ]; then
    ( cd "$STATE_DIR" && find . -type f -newer "$arm_dir/started.marker" \
        \( -name '*.json' -o -name '*.yaml' \) -print0 \
      | tar --null -T - -czf "$arm_dir/artifacts.tgz" ) || true
    ( cd "$STATE_DIR" && find . -type f -newer "$arm_dir/started.marker" | sort ) \
      > "$arm_dir/artifact_paths.txt" || true
  fi

  # Headline numbers, straight from the log — the funnel lines and the
  # population-resolution lines are tracing::info!/warn! output.
  grep -E "stage 1 fast-evaluation slice|SearchStarted|quality-screen submission sizes|population|walkforward|cpcv|PBO|exported|portfolio" \
    "$arm_dir/discover.log" > "$arm_dir/headlines.txt" || true
  echo "arm $name done in $(cat "$arm_dir/wall_seconds.txt")s"
}

run_arm A 100  "$SEED_MAIN"
run_arm B 4096 "$SEED_MAIN"
run_arm C 100  "$SEED_ALT"

echo
echo "All three arms complete. Results under: $OUT_ROOT"
echo
echo "Compare, per arm (A=pop100/s$SEED_MAIN, B=pop4096/s$SEED_MAIN, C=pop100/s$SEED_ALT):"
echo "  1. wall_seconds.txt and dmon.log        — did B actually load the card?"
echo "  2. headlines.txt / funnel JSONs         — candidates per stage, exported count"
echo "  3. artifacts.tgz portfolio + forward-test JSONs — holdout metrics + PBO"
echo "  4. strategy hashes across arms          — overlap = 'more of the same'"
echo
echo "Verdict rule (pre-registered in this script's header):"
echo "  |d(B,A)| <= |d(C,A)| on holdout metrics  => population was never the constraint"
echo "  d(B,A) >> d(C,A) toward better holdout   => population was a constraint; raise it as a recorded decision"
echo "  d(B,A) toward worse holdout / higher PBO => bigger search amplified overfitting; keep 100, question the validation budget next"
