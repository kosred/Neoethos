#!/usr/bin/env bash
# Rented-NVIDIA benchmark run, driven entirely by the built Rust CLI.
#
# No Python is invoked anywhere on this path. The legacy helpers under
# scripts/gpu-bench/*.py remain in the tree as isolated tooling and are not
# called from here.
#
# The four passes stay separate processes on purpose: clean timing must not be
# contaminated by diagnostics counters or by a profiler attaching to the run.
#
# Usage:
#   scripts/gpu-bench/run_rented.sh <candidate-sha> [input-csv-dir]
#
# Environment:
#   NEOETHOS_BENCH_ROOT       output root (default cache/gpu-bench)
#   NEOETHOS_BENCH_TIMEFRAMES comma-separated timeframes (default H1,M30,M15,M5,M1)
#   NEOETHOS_BENCH_PROTOTYPES comma-separated prototypes (default a,b,c)
#   NEOETHOS_BENCH_POPULATION tiny-fixture population (default 256)
#   NEOETHOS_CLI_FEATURES     cargo features for the runner (default gpu-nvidia)

set -euo pipefail

CANDIDATE_SHA="${1:-}"
INPUT_DIR="${2:-cache/gpu-bench/input}"
if [[ -z "$CANDIDATE_SHA" ]]; then
  printf 'usage: %s <candidate-sha> [input-csv-dir]\n' "$0" >&2
  exit 2
fi

ROOT="${NEOETHOS_BENCH_ROOT:-cache/gpu-bench}"
TIMEFRAMES="${NEOETHOS_BENCH_TIMEFRAMES:-H1,M30,M15,M5,M1}"
PROTOTYPES="${NEOETHOS_BENCH_PROTOTYPES:-a,b,c}"
POPULATION="${NEOETHOS_BENCH_POPULATION:-256}"
FEATURES="${NEOETHOS_CLI_FEATURES:-gpu-nvidia}"
SNAPSHOT_DIR="$ROOT/snapshots"
RUNS_DIR="$ROOT/runs"
CLI="cargo run --quiet --release -p neoethos-cli --features $FEATURES --"

mkdir -p "$SNAPSHOT_DIR" "$RUNS_DIR"

printf '== preflight ==\n'
scripts/gpu-bench/preflight.sh "$ROOT/preflight.json"

printf '== snapshots ==\n'
IFS=',' read -r -a TF_LIST <<<"$TIMEFRAMES"
for timeframe in "${TF_LIST[@]}"; do
  csv="$INPUT_DIR/EURUSD_${timeframe}.csv"
  if [[ ! -f "$csv" ]]; then
    printf 'missing canonical CSV for %s: %s\n' "$timeframe" "$csv" >&2
    exit 30
  fi
  $CLI bench-prepare \
    --csv "$csv" \
    --out "$SNAPSHOT_DIR/${timeframe}.json" \
    --timeframe "$timeframe" \
    --population "$POPULATION"
done

printf '== matrix ==\n'
$CLI bench-matrix \
  --candidate-sha "$CANDIDATE_SHA" \
  --snapshot-dir "$SNAPSHOT_DIR" \
  --fixture snapshot \
  --timeframes "$TIMEFRAMES" \
  --prototypes "$PROTOTYPES" \
  --runs-root "$RUNS_DIR" \
  --out "$ROOT/matrix.json"

printf '== execution ==\n'
printf 'The matrix manifest lists one command per executable job, each already\n'
printf 'attributed to its pinned worktree and pass. Run them from %s in the\n' "$ROOT/matrix.json"
printf 'order printed above; blocked jobs stay blocked and must not be faked.\n'

printf '== collation ==\n'
$CLI bench-collate --reports "$RUNS_DIR" --out "$ROOT/summary.json"
