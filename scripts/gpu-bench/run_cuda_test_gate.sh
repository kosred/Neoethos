#!/usr/bin/env bash
# Execute one filtered Rust GPU test binary and prove that it was non-vacuous.

set -euo pipefail
export CARGO_TERM_COLOR=never

if (( $# < 4 )); then
  printf 'usage: %s <log> <expected-passed> <expected-ignored> <command> [args...]\n' "$0" >&2
  exit 64
fi

log_path="$1"
expected_passed="$2"
expected_ignored="$3"
shift 3

if [[ ! "$expected_passed" =~ ^[0-9]+$ || ! "$expected_ignored" =~ ^[0-9]+$ ]]; then
  printf 'expected counts must be non-negative integers\n' >&2
  exit 64
fi
if (( expected_passed == 0 )); then
  printf 'a required GPU gate may not expect zero passing tests\n' >&2
  exit 64
fi

mkdir -p "$(dirname "$log_path")"
set +e
"$@" 2>&1 | tee "$log_path"
command_status=${PIPESTATUS[0]}
set -e

# A paid-device result is invalid if the selected test says it skipped, used a
# fallback, or substituted another engine/CPU path even when Cargo returned 0.
if grep -Eqi '(^|[^[:alnum:]_])(skip|skipped|skipping|fallback|substitut(e|ed|ion|ing))([^[:alnum:]_]|$)' "$log_path"; then
  printf 'required CUDA command reported skip/fallback/substitution text (log: %s)\n' \
    "$log_path" >&2
  exit 87
fi

read -r observed_passed observed_failed observed_ignored observed_measured result_lines \
  < <(awk '
    /^test result: / {
      result_lines += 1
      for (i = 1; i < NF; i += 1) {
        if ($(i + 1) == "passed;") passed += $i
        if ($(i + 1) == "failed;") failed += $i
        if ($(i + 1) == "ignored;") ignored += $i
        if ($(i + 1) == "measured;") measured += $i
      }
    }
    END { print passed + 0, failed + 0, ignored + 0, measured + 0, result_lines + 0 }
  ' "$log_path")

read -r observed_selected running_lines \
  < <(awk '
    /^running [0-9]+ tests?$/ { selected += $2; running_lines += 1 }
    END { print selected + 0, running_lines + 0 }
  ' "$log_path")

expected_selected=$((expected_passed + expected_ignored))
if (( result_lines != 1 || running_lines != 1 )); then
  printf 'expected exactly one --lib test result, observed result_lines=%s running_lines=%s (log: %s)\n' \
    "$result_lines" "$running_lines" "$log_path" >&2
  exit 88
fi
if (( observed_failed != 0 || observed_measured != 0 )); then
  printf 'required CUDA command reported failed=%s measured=%s (log: %s)\n' \
    "$observed_failed" "$observed_measured" "$log_path" >&2
  exit 89
fi
if (( observed_passed != expected_passed || observed_ignored != expected_ignored \
      || observed_selected != expected_selected )); then
  printf 'CUDA test-count mismatch: expected passed=%s ignored=%s selected=%s; observed passed=%s ignored=%s selected=%s (log: %s)\n' \
    "$expected_passed" "$expected_ignored" "$expected_selected" \
    "$observed_passed" "$observed_ignored" "$observed_selected" "$log_path" >&2
  exit 90
fi
if (( command_status != 0 )); then
  printf 'required CUDA command exited %s after the exact test count ran (log: %s)\n' \
    "$command_status" "$log_path" >&2
  exit "$command_status"
fi

printf 'CUDA gate accepted: passed=%s ignored=%s selected=%s (log: %s)\n' \
  "$observed_passed" "$observed_ignored" "$observed_selected" "$log_path"
