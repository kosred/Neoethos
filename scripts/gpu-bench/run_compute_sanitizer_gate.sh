#!/usr/bin/env bash
# Run one non-vacuous Rust GPU test binary under Compute Sanitizer memcheck.

set -euo pipefail

if (( $# < 5 )); then
  printf 'usage: %s <test-log> <sanitizer-log> <expected-passed> <expected-ignored> <command> [args...]\n' \
    "$0" >&2
  exit 64
fi

test_log="$1"
sanitizer_log="$2"
expected_passed="$3"
expected_ignored="$4"
shift 4

if ! command -v compute-sanitizer >/dev/null 2>&1; then
  printf 'compute-sanitizer is required for the paid CUDA memory gate\n' >&2
  exit 69
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$(dirname "$sanitizer_log")"
rm -f -- "$sanitizer_log"

set +e
bash "$script_dir/run_cuda_test_gate.sh" \
  "$test_log" "$expected_passed" "$expected_ignored" \
  compute-sanitizer \
  --tool memcheck \
  --leak-check full \
  --target-processes all \
  --require-cuda-init no \
  --error-exitcode 86 \
  --log-file "$sanitizer_log" \
  "$@"
test_gate_status=$?
set -e

if [[ ! -s "$sanitizer_log" ]]; then
  printf 'Compute Sanitizer produced no report: %s\n' "$sanitizer_log" >&2
  exit 91
fi
if ! grep -Fq 'ERROR SUMMARY: 0 errors' "$sanitizer_log"; then
  printf 'Compute Sanitizer did not report ERROR SUMMARY: 0 errors (log: %s)\n' \
    "$sanitizer_log" >&2
  exit 92
fi
if grep -Eq 'ERROR SUMMARY: [1-9][0-9,]* errors?' "$sanitizer_log"; then
  printf 'Compute Sanitizer reported memory errors (log: %s)\n' "$sanitizer_log" >&2
  exit 92
fi
if ! grep -Fq 'LEAK SUMMARY: 0 bytes leaked' "$sanitizer_log"; then
  printf 'Compute Sanitizer did not report LEAK SUMMARY: 0 bytes leaked (log: %s)\n' \
    "$sanitizer_log" >&2
  exit 93
fi
if grep -Eq 'LEAK SUMMARY: [1-9][0-9,]* bytes leaked' "$sanitizer_log"; then
  printf 'Compute Sanitizer reported leaked device memory (log: %s)\n' \
    "$sanitizer_log" >&2
  exit 93
fi
if (( test_gate_status != 0 )); then
  printf 'Compute Sanitizer command/test gate exited %s despite clean summaries (test log: %s)\n' \
    "$test_gate_status" "$test_log" >&2
  exit "$test_gate_status"
fi

printf 'Compute Sanitizer gate accepted: zero errors and zero leaked bytes (log: %s)\n' \
  "$sanitizer_log"
