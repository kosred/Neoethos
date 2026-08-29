#!/usr/bin/env bash
# Rented-NVIDIA benchmark run, driven entirely by the built Rust CLI.
#
# No private parser or Python conversion is invoked anywhere on this path.
# Source CSV enters only through the shared admitted importer and every paid
# snapshot reopens the exact published canonical Vortex identity.
#
# The four passes stay separate processes on purpose: clean timing must not be
# contaminated by diagnostics counters or by a profiler attaching to the run.
#
# Usage:
#   scripts/gpu-bench/run_rented.sh <candidate-sha> [source-dir]
#
# Environment:
#   NEOETHOS_BENCH_ROOT       output root (default cache/gpu-bench)
#   NEOETHOS_BENCH_TIMEFRAMES comma-separated timeframes (default H1,M30,M15,M5,M1)
#   NEOETHOS_BENCH_PROTOTYPES comma-separated prototypes (default b,c)
#   NEOETHOS_BENCH_POPULATION tiny-fixture population (default 256)
#   NEOETHOS_CLI_FEATURES     cargo features for the runner (default gpu-nvidia)

set -euo pipefail

CANDIDATE_SHA="${1:-}"
SOURCE_DIR="${2:-cache/gpu-bench/input}"
if [[ -z "$CANDIDATE_SHA" ]]; then
  printf 'usage: %s <candidate-sha> [source-dir]\n' "$0" >&2
  exit 2
fi

ROOT="${NEOETHOS_BENCH_ROOT:-cache/gpu-bench}"
TIMEFRAMES="${NEOETHOS_BENCH_TIMEFRAMES:-H1,M30,M15,M5,M1}"
PROTOTYPES="${NEOETHOS_BENCH_PROTOTYPES:-b,c}"
POPULATION="${NEOETHOS_BENCH_POPULATION:-256}"
FEATURES="${NEOETHOS_CLI_FEATURES:-gpu-nvidia}"
SNAPSHOT_DIR="$ROOT/snapshots"
RUNS_DIR="$ROOT/runs"
CLI="cargo run --quiet --release -p neoethos-cli --features $FEATURES --"

# The current CLI A benchmark enters the strict aggregate dispatcher and can
# execute native B while labelling the receipt A. Direct Prototype A hardware
# proof lives in run_cuda_validation.sh; an attributed A timing run remains
# blocked until the CLI has its own direct A entrypoint.
normalized_prototypes="${PROTOTYPES,,}"
case ",${normalized_prototypes// /}," in
  *,a,*|*,prototype-a,*|*,prototype_a,*)
    printf 'Prototype A CLI benchmarking is disabled: the current command can execute B while labelling the result A. Run the direct A CUDA validation gate instead.\n' >&2
    exit 32
    ;;
esac

mkdir -p "$SNAPSHOT_DIR" "$RUNS_DIR"
IMPORT_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/neoethos-bench-import.XXXXXX")"
cleanup_import_root() {
  case "$IMPORT_ROOT" in
    "${TMPDIR:-/tmp}"/neoethos-bench-import.*) rm -rf -- "$IMPORT_ROOT" ;;
    *) printf 'refusing to clean unexpected import root: %s\n' "$IMPORT_ROOT" >&2 ;;
  esac
}
trap cleanup_import_root EXIT

printf '== preflight ==\n'
scripts/gpu-bench/preflight.sh "$ROOT/preflight.json"

printf '== snapshots ==\n'
IFS=',' read -r -a TF_LIST <<<"$TIMEFRAMES"
for timeframe in "${TF_LIST[@]}"; do
  csv="$SOURCE_DIR/EURUSD_${timeframe}.csv"
  if [[ ! -f "$csv" ]]; then
    printf 'missing explicitly bar-open source CSV for %s: %s\n' "$timeframe" "$csv" >&2
    exit 30
  fi
  import_output="$($CLI import \
    --source "$csv" \
    --format csv \
    --source-namespace "gpu-bench-${CANDIDATE_SHA}" \
    --symbol EURUSD \
    --timeframe "$timeframe" \
    --bar-timestamps bar_open \
    --root "$IMPORT_ROOT")"
  printf '%s\n' "$import_output"
  identity="$(printf '%s\n' "$import_output" | sed -n 's/^  dataset identity:   //p')"
  if [[ "$identity" != d1-* ]]; then
    printf 'shared import did not return an exact canonical identity for %s\n' "$timeframe" >&2
    exit 31
  fi
  $CLI bench-prepare \
    --data-root "$IMPORT_ROOT" \
    --symbol EURUSD \
    --dataset-identity "$identity" \
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
